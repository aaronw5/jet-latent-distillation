"""A7 — stacked learned context (formula message passing). ParT builds each particle's context over 8 blocks; one
learned hop (A6) stops at ~83 %. Here L hops, each a small formula layer:
  h⁰_i = standardized own + neighbourhood inputs of particle i (38)
  hop l:  w^l_ij = softmax_j K^l(ln kT_ij, ln z_ij, ln ΔR_ij, ln m²_ij)   (one kernel formula per hop: hinge terms, 24 → 1)
          h^l_i  = h^{l−1}_i + Σ_j w^l_ij (h^{l−1}_j · M^l)                  (what is gathered: a linear map of the neighbours' state)
  then per head: score s_i = φ(x_i)·a_h + h^L_i·c_h;  value v_i = φ_v(x_i)·W_h + h^L_i·D_h;  α as before (jet-level α_cls)
Every piece is a formula or a linear map; the state h^L_i reads as "particle i and what sits around it, L hops out".
Start: A3's weight formulas and S12A2h's values (as A6), kernels and maps at 0 → the start equals A6's start.

  ALPHA_TAG=A3 VALUES_TAG=S12A2h OUT_TAG=A7 HOPS=3 python -m jetdistill.research.stack [n_fit n_dev epochs]"""
import json, os, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, downstream, rows_of_split, OUT
from .heads_attn import feats
from .alpha_fit import rel_feats
from .alpha_e2e import zfeat
from .alpha_pair import pair_feats, NU


def run(n_fit=40000, n_dev=20000, epochs=40, n_jet=100000, lr=3e-4, lam=0.01, bs=250, device='mps', log=print):
    import torch
    t0 = time.time(); at, vt, tag, L = os.environ.get('ALPHA_TAG', 'A3'), os.environ.get('VALUES_TAG', 'S12A2h'), os.environ.get('OUT_TAG', 'A7'), int(os.environ.get('HOPS', 3))
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
    def phiv(F): X = F[..., :38]; return torch.cat([X] + [torch.clamp(X - knvt[k], min=0) for k in range(5)], -1)
    Wc, Wz = torch.nn.Parameter(T(PA['coef'].T)), torch.nn.Parameter(T(PA['cz'].T)); Wv = torch.nn.Parameter(T(Wv0))
    Ks = [torch.nn.Parameter(torch.zeros(24, 1, device=device)) for _ in range(L)]; Ms = [torch.nn.Parameter(torch.zeros(NU, NU, device=device)) for _ in range(L)]
    Cu = torch.nn.Parameter(torch.zeros(NU, 16, device=device)); Dv = torch.nn.Parameter(torch.zeros(16, NU, 16, device=device))
    def heads(F, ok, X, Z):
        n, P = F.shape[0], F.shape[1]; pp = phip(pair_feats(X)); pair = ok[:, :, None] & ok[:, None] & ~torch.eye(P, dtype=torch.bool, device=device)[None]
        h = ((F[..., :NU] - muut) / sdut) * ok[..., None]
        for l in range(L):
            w = torch.softmax((pp @ Ks[l])[..., 0].masked_fill(~pair, -1e9), 2) * pair                         # (n, P, P)
            h = h + torch.einsum('nij,njk->nik', w, h @ Ms[l]) * ok[..., None]
        s = phi(F) @ Wc + h @ Cu; rel = torch.softmax(s.masked_fill(~ok[..., None], -1e9), 1); acls = torch.sigmoid(-(Z @ Wz)); al = (1 - acls)[:, None] * rel
        v = torch.einsum('npk,hko->npho', phiv(F), Wv[:, :228]) + torch.einsum('npk,hko->npho', h, Dv)
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
    params = [Wc, Wz, Wv, Cu, Dv] + Ks + Ms; a0 = agree(); best = (a0, [p.detach().clone() for p in params], 0); path = [(0, a0)]
    log(f'  start ({at} weights, {vt} values, {L} hops at 0): {100 * a0:.2f}%, {time.time() - t0:.0f} s')
    opt = torch.optim.Adam([{'params': [Wc, Wz, Cu, Dv] + Ks + Ms, 'lr': lr}, {'params': [Wv], 'lr': lr / 3}])
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, bs)):
            i = torch.from_numpy(of[a:a + bs]).to(device); o = run_b(Fft, okft, Xft, Zft, i); Lg = downstream(model, o[:, :8], o[:, 8:])
            loss = -(pf[i] * torch.log_softmax(Lg, 1)).sum(1).mean() + lam * ((o - Oft[i]) ** 2 / vo).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if device == 'mps': torch.mps.empty_cache()
        ag = agree(); path.append((ep + 1, ag))
        if ag > best[0]: best = (ag, [p.detach().clone() for p in params], ep + 1)
        log(f'  A7 epoch {ep + 1}: all formulas, {L} learned hops: {100 * ag:.2f}%, {time.time() - t0:.0f} s')
    names = ['coef', 'cz', 'Wv', 'cu', 'Dv'] + [f'K{l}' for l in range(L)] + [f'M{l}' for l in range(L)]
    np.savez(OUT / f'{tag}_model.npz', **{k: v.cpu().numpy() for k, v in zip(names, best[1])}, kn=PA['kn'], mu=PA['mu'], sd=PA['sd'], knp=PA['knp'], mup=PA['mup'], sdp=PA['sdp'], muu=PA['muu'], sdu=PA['sdu'], knv=knv, hops=L)
    r = dict(experiment=tag, hops=L, alpha=at, values=vt, n_fit=n_fit, epochs=epochs, lr=lr, start=a0, best=best[0], best_epoch=best[2], path=path, seconds=time.time() - t0)
    (OUT / f'{tag}_stack.json').write_text(json.dumps(r, indent=1)); log(f'{tag}: all formulas with {L} learned hops: {100 * best[0]:.2f}% (start {100 * a0:.2f}%)'); return r


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; run(*a)
