"""A3 — the weight formulas with a learnable pair interaction (a formula attention). Per head h, the score of particle i
  s_i = g_h(x_i) + c_h · Σ_j w_ij u(x_j),     w_ij = softmax_j K_h(ln kT_ij, ln z_ij, ln ΔR_ij, ln m²_ij)   (j ≠ i, real)
g_h: the S11/A2 score formula (start), K_h: hinge terms of the 4 pair variables (which neighbours count), u: the
neighbours' standardized own + neighbourhood inputs (what about them counts), c_h: their weights (start 0, so the start
is exactly the source formula). α_cls from the jet-level formula as before. Tuned end to end through ParT's values and
downstream toward ParT's probabilities (decision-focused, as A2); then written like S11 for the values re-tune.

  ALPHA_TAG=A2 OUT_TAG=A3 python -m jetdistill.research.alpha_pair [n_fit n_dev epochs]"""
import json, os, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, after, rows_of_split, OUT
from .heads_attn import feats
from .alpha_fit import rel_feats
from .alpha_e2e import zfeat
from .weights import internals
from ..config import RESULTS

NU = 38                                                                      # u: the first 38 inputs (own 18 + neighbourhood 20)


def pair_feats(x):
    """(n, P, P, 4) torch: ln kT, ln z, ln ΔR, ln m² of every pair (massless, jet coordinates); x (n, P, 3) pt, Δη, Δφ"""
    import torch
    pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]; deta = eta[:, :, None] - eta[:, None]; dphi = phi[:, :, None] - phi[:, None]
    dR = torch.sqrt(deta ** 2 + dphi ** 2).clamp(min=1e-6); pmin = torch.minimum(pt[:, :, None], pt[:, None]).clamp(min=1e-6)
    m2 = (2 * pt[:, :, None] * pt[:, None] * (torch.cosh(deta) - torch.cos(dphi))).clamp(min=1e-8)
    return torch.stack([torch.log(pmin * dR), torch.log(pmin / (pt[:, :, None] + pt[:, None]).clamp(min=1e-6)), torch.log(dR), torch.log(m2)], -1)


def run(n_fit=20000, n_dev=20000, epochs=30, n_jet=100000, lr=3e-4, bs=250, device='mps', log=print):
    import torch
    t0 = time.time(); src, tag = os.environ.get('ALPHA_TAG', 'A2'), os.environ.get('OUT_TAG', 'A3')
    model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    _, _, Mfa, Lfa, _ = extract(model, 'fit', n_jet, device); _, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    Ff, okf = feats(Jf, rf, model); Fd, okd = feats(Jd, rd, model)
    Ff = np.concatenate([Ff, rel_feats(Ff, okf)], -1); Fd = np.concatenate([Fd, rel_feats(Fd, okd)], -1)
    Xf, Xd = np.asarray(Jf['x'][rf], np.float32), np.asarray(Jd['x'][rd], np.float32)
    Zf_all, Zd = zfeat(Jf, Jd, n_jet, rd, Mfa, Md); Zf = Zf_all[:n_fit]
    P0 = np.load(OUT / f'{src}_alpha.npz'); kn, mu, sd, COEF, CZ = P0['kn'], P0['mu'], P0['sd'], P0['coef'], P0['cz']
    D = RESULTS / '_cls' / 'full'
    _, Vf = internals(model, np.load(D / 'fit' / 'x_f16.npy', mmap_mode='r')[:n_fit], np.load(D / 'fit' / 'mask.npy')[:n_fit], device)
    _, Vd = internals(model, np.load(D / 'dev' / 'x_f16.npy', mmap_mode='r')[rd], np.load(D / 'dev' / 'mask.npy')[rd], device)
    T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
    with torch.no_grad():                                                   # thresholds of the pair variables (sample of real pairs)
        xs = T(Xf[:500]); okp = T(okf[:500], torch.bool); pf_ = pair_feats(xs)[okp[:, :, None] & okp[:, None]].cpu().numpy()
    knp = np.quantile(pf_, np.linspace(.15, .85, 5), axis=0).astype(np.float32); mup, sdp = pf_.mean(0), pf_.std(0) + 1e-6
    muu, sdu = Ff[okf][:, :NU].mean(0), Ff[okf][:, :NU].std(0) + 1e-6
    knt, mut, sdt, knpt, mupt, sdpt, muut, sdut = (T(a) for a in (kn, mu, sd, knp, mup, sdp, muu, sdu))
    def phi(F): return torch.cat([(F - mut) / sdt] + [torch.clamp(F - knt[k], min=0) / sdt for k in range(len(kn))], -1)
    def phip(Q): return torch.cat([(Q - mupt) / sdpt] + [torch.clamp(Q - knpt[k], min=0) / sdpt for k in range(len(knp))], -1)   # (…, 24)
    Wc = torch.nn.Parameter(T(COEF.T)); Wz = torch.nn.Parameter(T(CZ.T)); Ak = torch.nn.Parameter(torch.zeros(24, 16, device=device))
    Cu = torch.nn.Parameter(torch.zeros(NU, 16, device=device))
    def scores(F, ok, X):
        n, P = F.shape[0], F.shape[1]; s = phi(F) @ Wc                                                   # (n, P, 16) own part
        kap = phip(pair_feats(X)) @ Ak                                                                   # (n, P, P, 16) which neighbours count
        pair = ok[:, :, None] & ok[:, None] & ~torch.eye(P, dtype=torch.bool, device=device)[None]
        w = torch.softmax(kap.masked_fill(~pair[..., None], -1e9), 2) * pair[..., None]
        u = ((F[..., :NU] - muut) / sdut) * ok[..., None]                                               # (n, P, NU)
        return s + torch.einsum('nijh,njk,kh->nih', w, u, Cu)                                           # + Σ_j w_ij u(x_j)·c_h
    def logits(F, ok, X, Z, V):
        P = int(ok.sum(1).max()); F, ok, X, V = F[:, :P].float(), ok[:, :P], X[:, :P].float(), V[:, :, :1 + P]
        rel = torch.softmax(scores(F, ok, X).masked_fill(~ok[..., None], -1e9), 1); acls = torch.sigmoid(-(Z @ Wz)); cls = model.cls_token[0, 0].expand(F.shape[0], -1)
        for b, blk in enumerate(model.cls_blocks):
            a = torch.cat([acls[:, 8 * b:8 * b + 8, None], (1 - acls[:, 8 * b:8 * b + 8, None]) * rel[:, :, 8 * b:8 * b + 8].permute(0, 2, 1)], 2)
            cls = after(blk, torch.einsum('nhp,nphd->nhd', a, V[:, b].float()), cls)
        return model.fc(model.norm(cls))
    Fft, Fdt, okft, okdt = T(Ff, torch.float16), T(Fd, torch.float16), T(okf, torch.bool), T(okd, torch.bool); Xft, Xdt = T(Xf, torch.float16), T(Xd, torch.float16)
    Zft, Zdt = T(Zf), T(Zd); Vft, Vdt = T(Vf.transpose(1, 0, 2, 3, 4), torch.float16), T(Vd.transpose(1, 0, 2, 3, 4), torch.float16); pf = torch.softmax(T(Lfa[:n_fit]), 1)
    od = np.argsort(okd.sum(1)); of = np.argsort(okf.sum(1))                                             # batches of similar multiplicity (smaller P², faster)
    def agree():
        with torch.no_grad():
            pred = np.empty(n_dev, int)
            for a in range(0, n_dev, bs):
                i = torch.from_numpy(od[a:a + bs]).to(device); pred[od[a:a + bs]] = logits(Fdt[i], okdt[i], Xdt[i], Zdt[i], Vdt[i]).argmax(1).cpu().numpy()
        return float((pred == ref).mean())
    a0 = agree(); best = (a0, [p.detach().clone() for p in (Wc, Wz, Ak, Cu)], 0); path = [(0, a0)]
    log(f'  start ({src} weight formulas, no pair term, ParT values): {100 * a0:.2f}%, {time.time() - t0:.0f} s')
    opt = torch.optim.Adam([{'params': [Wc, Wz], 'lr': lr}, {'params': [Ak, Cu], 'lr': 3 * lr}])
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, bs)):
            i = torch.from_numpy(of[a:a + bs]).to(device)
            loss = -(pf[i] * torch.log_softmax(logits(Fft[i], okft[i], Xft[i], Zft[i], Vft[i]), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if device == 'mps': torch.mps.empty_cache()
        ag = agree(); path.append((ep + 1, ag))
        if ag > best[0]: best = (ag, [p.detach().clone() for p in (Wc, Wz, Ak, Cu)], ep + 1)
        log(f'  A3 epoch {ep + 1}: formula weights with pair interaction + ParT values {100 * ag:.2f}%, {time.time() - t0:.0f} s')
    for p, v in zip((Wc, Wz, Ak, Cu), best[1]): p.data.copy_(v)
    np.savez(OUT / f'{tag}_alpha.npz', kn=kn, mu=mu, sd=sd, coef=Wc.detach().cpu().numpy().T, cz=Wz.detach().cpu().numpy().T, ak=Ak.detach().cpu().numpy(), cu=Cu.detach().cpu().numpy(),
             knp=knp, mup=mup, sdp=sdp, muu=muu, sdu=sdu)
    def alpha_of(J, rows, Z, chunk=2000):
        A = np.zeros((2, len(rows), 8, 129), np.float16)
        with torch.no_grad():
            for a in range(0, len(rows), chunk):
                r = rows[a:a + chunk]; F, ok = feats(J, r, model); F = np.concatenate([F, rel_feats(F, ok)], -1); X = np.asarray(J['x'][r], np.float32)
                for c0 in range(0, len(r), bs):
                    sl = slice(c0, c0 + bs); okc = ok[sl]; P = int(okc.sum(1).max()); Ft, okt, Xt = T(F[sl, :P]), T(okc[:, :P], torch.bool), T(X[sl, :P])
                    rel = torch.softmax(scores(Ft, okt, Xt).masked_fill(~okt[..., None], -1e9), 1).cpu().numpy(); acls = torch.sigmoid(-(T(Z[a + c0:a + c0 + len(okc)]) @ Wz)).cpu().numpy()
                    for hh in range(16):
                        A[hh // 8, a + c0:a + c0 + len(okc), hh % 8, 0] = acls[:, hh]; A[hh // 8, a + c0:a + c0 + len(okc), hh % 8, 1:1 + P] = (1 - acls[:, hh:hh + 1]) * rel[:, :, hh]
        return A
    np.save(OUT / f'{tag}_alpha_fit.npy', alpha_of(Jf, rows_of_split('fit', n_jet), Zf_all)); np.save(OUT / f'{tag}_alpha_dev.npy', alpha_of(Jd, rd, Zd))
    r = dict(experiment=tag, source=src, n_fit=n_fit, epochs=epochs, lr=lr, start=a0, best=best[0], best_epoch=best[2], path=path, seconds=time.time() - t0)
    (OUT / f'{tag}_alpha_pair.json').write_text(json.dumps(r, indent=1)); log(f'{tag}: weight formulas with a learnable pair interaction, with ParT values {100 * best[0]:.2f}% (start {100 * a0:.2f}%)'); return r


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; run(*a)
