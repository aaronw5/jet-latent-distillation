"""A6 — learned pair context in the weights AND the values (all formulas, no ParT internals). Per head h:
  w_ij   = softmax_j K_h(ln kT_ij, ln z_ij, ln ΔR_ij, ln m²_ij)                       which neighbours count (A3's kernel)
  ctx_i  = Σ_j w_ij u(x_j)                                                              what the neighbours are (standardized inputs)
  s_i    = g_h(x_i) + ctx_i · c_h,   α = (1 − α_cls) softmax(s), α_cls jet-level        the weights (A3)
  v_i    = f_h(x_i) + ctx_i · D_h                                                       the 16 values: own formula + neighbour context
  o_h    = Σ_i α_i v_i + α_cls · e_h + b_h                                              then ParT's fixed downstream
Start: A3's weight formulas, the values of a per-head model fitted on comparable weights (VALUES_TAG, e.g. S12A2h),
D = 0. Everything tuned together toward ParT's probabilities (+ λ on the head outputs).

  ALPHA_TAG=A3 VALUES_TAG=S12A2h OUT_TAG=A6 python -m jetdistill.research.values_pair [n_fit n_dev epochs]"""
import json, os, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, after, downstream, rows_of_split, OUT
from .heads_attn import feats
from .alpha_fit import rel_feats
from .alpha_e2e import zfeat
from .alpha_pair import pair_feats, NU


def run(n_fit=40000, n_dev=20000, epochs=40, n_jet=100000, lr=3e-4, lam=0.01, bs=250, device='mps', log=print):
    import torch
    t0 = time.time(); at, vt, tag = os.environ.get('ALPHA_TAG', 'A3'), os.environ.get('VALUES_TAG', 'S12A2h'), os.environ.get('OUT_TAG', 'A6')
    model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    Ofa, _, Mfa, Lfa, _ = extract(model, 'fit', n_jet, device); _, _, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    Ff, okf = feats(Jf, rf, model); Fd, okd = feats(Jd, rd, model)
    Ff = np.concatenate([Ff, rel_feats(Ff, okf)], -1); Fd = np.concatenate([Fd, rel_feats(Fd, okd)], -1)
    Xf, Xd = np.asarray(Jf['x'][rf], np.float32), np.asarray(Jd['x'][rd], np.float32)
    Zf_all, Zd = zfeat(Jf, Jd, n_jet, rd, Mfa, Md); Zf = Zf_all[:n_fit]
    PA = np.load(OUT / f'{at}_alpha.npz'); PV = np.load(OUT / f'{vt}_model.npz'); Wv0, knv = PV['W'], PV['kn'][:, :38]
    T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
    knt, mut, sdt = T(PA['kn']), T(PA['mu']), T(PA['sd']); knpt, mupt, sdpt = T(PA['knp']), T(PA['mup']), T(PA['sdp']); muut, sdut = T(PA['muu']), T(PA['sdu']); knvt = T(knv)
    def phi(F): return torch.cat([(F - mut) / sdt] + [torch.clamp(F - knt[k], min=0) / sdt for k in range(knt.shape[0])], -1)
    def phip(Q): return torch.cat([(Q - mupt) / sdpt] + [torch.clamp(Q - knpt[k], min=0) / sdpt for k in range(knpt.shape[0])], -1)
    def phiv(F): X = F[..., :38]; return torch.cat([X] + [torch.clamp(X - knvt[k], min=0) for k in range(5)], -1)                 # the values' own terms (228)
    Wc, Wz, Ak, Cu = (torch.nn.Parameter(T(x)) for x in (PA['coef'].T, PA['cz'].T, PA['ak'], PA['cu']))
    Wv = torch.nn.Parameter(T(Wv0)); Dv = torch.nn.Parameter(torch.zeros(16, NU, 16, device=device))
    def heads(F, ok, X, Z):
        n, P = F.shape[0], F.shape[1]; kap = phip(pair_feats(X)) @ Ak
        pair = ok[:, :, None] & ok[:, None] & ~torch.eye(P, dtype=torch.bool, device=device)[None]
        w = torch.softmax(kap.masked_fill(~pair[..., None], -1e9), 2) * pair[..., None]                          # (n, P, P, 16)
        ctx = torch.einsum('nijh,njk->nihk', w, ((F[..., :NU] - muut) / sdut) * ok[..., None])                  # (n, P, 16, NU) per head
        s = phi(F) @ Wc + torch.einsum('nihk,kh->nih', ctx, Cu)
        rel = torch.softmax(s.masked_fill(~ok[..., None], -1e9), 1); acls = torch.sigmoid(-(Z @ Wz))            # (n, P, 16), (n, 16)
        al = (1 - acls)[:, None] * rel                                                                            # particle weights
        v = torch.einsum('npk,hko->npho', phiv(F), Wv[:, :228]) + torch.einsum('nphk,hko->npho', ctx, Dv)      # (n, P, 16 heads, 16)
        return torch.einsum('nph,npho->nho', al, v) + acls[..., None] * Wv[:, 228][None] + Wv[:, 229][None]
    Fft, Fdt, okft, okdt = T(Ff, torch.float16), T(Fd, torch.float16), T(okf, torch.bool), T(okd, torch.bool); Xft, Xdt = T(Xf, torch.float16), T(Xd, torch.float16)
    Zft, Zdt = T(Zf), T(Zd); pf = torch.softmax(T(Lfa[:n_fit]), 1); Oft = T(Ofa[:, :n_fit].transpose(1, 0, 2, 3).reshape(n_fit, 16, 16)); vo = Oft.var(0) + 1e-6
    od, of = np.argsort(okd.sum(1)), np.argsort(okf.sum(1))
    def run_b(Fx, okx, Xx, Zx, i):
        P = int(okx[i].sum(1).max()); return heads(Fx[i, :P].float(), okx[i, :P], Xx[i, :P].float(), Zx[i])
    def agree():
        with torch.no_grad():
            pred = np.empty(n_dev, int)
            for a in range(0, n_dev, bs):
                i = torch.from_numpy(od[a:a + bs]).to(device); o = run_b(Fdt, okdt, Xdt, Zdt, i); pred[od[a:a + bs]] = downstream(model, o[:, :8], o[:, 8:]).argmax(1).cpu().numpy()
        return float((pred == ref).mean())
    params = [Wc, Wz, Ak, Cu, Wv, Dv]; a0 = agree(); best = (a0, [p.detach().clone() for p in params], 0); path = [(0, a0)]
    log(f'  start ({at} weights, {vt} values, no value context): {100 * a0:.2f}%, {time.time() - t0:.0f} s')
    opt = torch.optim.Adam([{'params': [Wc, Wz, Ak, Cu], 'lr': lr}, {'params': [Wv], 'lr': lr / 3}, {'params': [Dv], 'lr': lr}])
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, bs)):
            i = torch.from_numpy(of[a:a + bs]).to(device); o = run_b(Fft, okft, Xft, Zft, i); L = downstream(model, o[:, :8], o[:, 8:])
            loss = -(pf[i] * torch.log_softmax(L, 1)).sum(1).mean() + lam * ((o - Oft[i]) ** 2 / vo).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if device == 'mps': torch.mps.empty_cache()
        ag = agree(); path.append((ep + 1, ag))
        if ag > best[0]: best = (ag, [p.detach().clone() for p in params], ep + 1)
        log(f'  A6 epoch {ep + 1}: all formulas with learned pair context in weights and values {100 * ag:.2f}%, {time.time() - t0:.0f} s')
    np.savez(OUT / f'{tag}_model.npz', **{k: v.cpu().numpy() for k, v in zip(('coef', 'cz', 'ak', 'cu', 'Wv', 'Dv'), best[1])}, kn=PA['kn'], mu=PA['mu'], sd=PA['sd'],
             knp=PA['knp'], mup=PA['mup'], sdp=PA['sdp'], muu=PA['muu'], sdu=PA['sdu'], knv=knv)
    r = dict(experiment=tag, alpha=at, values=vt, n_fit=n_fit, epochs=epochs, lr=lr, lam=lam, start=a0, best=best[0], best_epoch=best[2], path=path, seconds=time.time() - t0)
    (OUT / f'{tag}_values_pair.json').write_text(json.dumps(r, indent=1)); log(f'{tag}: all formulas, learned pair context in weights and values: {100 * best[0]:.2f}% (start {100 * a0:.2f}%)'); return r


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; run(*a)
