"""A2 — decision-focused α: the S11-type weight formulas (per head: score g(x_i) on per-particle terms, class-token share
from a jet-level formula) tuned end to end — through ParT's own values and downstream — toward ParT's class
probabilities, starting from their least-squares fit (ALPHA_TAG's saved coefficients). The values are ParT's, fixed:
this measures how well formula weights can serve ParT's decision. Then the tuned weights are written like S11's
(…_alpha_fit/dev.npy) so S12 (formula values re-tuned on them) can run on top.

  ALPHA_TAG=S11 OUT_TAG=A2 python -m jetdistill.research.alpha_e2e [n_fit n_dev epochs]"""
import json, os, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, after, rows_of_split, OUT
from .heads_attn import feats
from .alpha_fit import rel_feats
from .weights import internals
from .ceiling_jet import matrix
from ..config import RESULTS


def zfeat(Jf, Jd, rows_fit_jets, rd, Mf, Md):
    """the jet-level inputs of α_cls, exactly as alpha_fit builds them"""
    from sklearn.preprocessing import QuantileTransformer
    keys = [k for k in Jf['Q']]; Q100 = matrix(Jf, keys)[:100000]; okq = np.isfinite(Q100).all(0) & (Q100.std(0) > 0)        # the transform of S11: always its 100k jets
    qt = QuantileTransformer(n_quantiles=500, output_distribution='normal', subsample=50000, random_state=0).fit(Q100[:, okq]); Qf, Qd = Q100[:rows_fit_jets], matrix(Jd, keys)[rd]
    Z = lambda Q, M: np.concatenate([np.clip(qt.transform(Q[:, okq]), -5, 5), np.log(M.sum(1, keepdims=True)), np.ones((len(M), 1))], 1).astype(np.float32)
    return Z(Qf, Mf), Z(Qd, Md)


def run(n_fit=20000, n_dev=20000, epochs=30, n_jet=100000, lr=3e-4, bs=500, device='mps', log=print):
    import torch
    t0 = time.time(); src, tag = os.environ.get('ALPHA_TAG', 'S11'), os.environ.get('OUT_TAG', 'A2')
    model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    _, _, Mfa, Lfa, _ = extract(model, 'fit', n_jet, device); _, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    Ff, okf = feats(Jf, rf, model); Fd, okd = feats(Jd, rd, model)
    Ff = np.concatenate([Ff, rel_feats(Ff, okf)], -1); Fd = np.concatenate([Fd, rel_feats(Fd, okd)], -1)
    Zf_all, Zd = zfeat(Jf, Jd, n_jet, rd, Mfa, Md); Zf = Zf_all[:n_fit]
    P0 = np.load(OUT / f'{src}_alpha.npz'); kn, mu, sd, COEF, CZ = P0['kn'], P0['mu'], P0['sd'], P0['coef'], P0['cz']
    D = RESULTS / '_cls' / 'full'
    _, Vf = internals(model, np.load(D / 'fit' / 'x_f16.npy', mmap_mode='r')[:n_fit], np.load(D / 'fit' / 'mask.npy')[:n_fit], device)
    _, Vd = internals(model, np.load(D / 'dev' / 'x_f16.npy', mmap_mode='r')[rd], np.load(D / 'dev' / 'mask.npy')[rd], device)
    log(f'  inputs, ParT values, start formulas ({src}), {time.time() - t0:.0f} s')
    T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
    knt, mut, sdt = T(kn), T(mu), T(sd)
    def phi(F): return torch.cat([(F - mut) / sdt] + [torch.clamp(F - knt[k], min=0) / sdt for k in range(len(kn))], -1)
    Wc = torch.nn.Parameter(T(COEF.T)); Wz = torch.nn.Parameter(T(CZ.T))                                      # (terms, 16), (jet inputs, 16)
    def logits(F, ok, Z, V):
        n = F.shape[0]; P = int(ok.sum(1).max()); F, ok, V = F[:, :P].float(), ok[:, :P], V[:, :, :1 + P]
        s = (phi(F) @ Wc).masked_fill(~ok[..., None], -1e9); rel = torch.softmax(s, 1)                         # (n, P, 16)
        acls = torch.sigmoid(-(Z @ Wz))                                                                          # (n, 16): α_cls = 1 / (1 + e^u)
        cls = model.cls_token[0, 0].expand(n, -1)
        for b, blk in enumerate(model.cls_blocks):
            a = torch.cat([acls[:, 8 * b:8 * b + 8, None], (1 - acls[:, 8 * b:8 * b + 8, None]) * rel[:, :, 8 * b:8 * b + 8].permute(0, 2, 1)], 2)   # (n, 8, 1+P)
            cls = after(blk, torch.einsum('nhp,nphd->nhd', a, V[:, b].float()), cls)
        return model.fc(model.norm(cls))
    Fft, Fdt, okft, okdt = T(Ff, torch.float16), T(Fd, torch.float16), T(okf, torch.bool), T(okd, torch.bool)
    Zft, Zdt = T(Zf), T(Zd); Vft, Vdt = T(Vf.transpose(1, 0, 2, 3, 4), torch.float16), T(Vd.transpose(1, 0, 2, 3, 4), torch.float16); pf = torch.softmax(T(Lfa[:n_fit]), 1)        # values: (jets, block, 1+P, head, 16)
    def agree():
        with torch.no_grad():
            return float((np.concatenate([logits(Fdt[a:a + bs], okdt[a:a + bs], Zdt[a:a + bs], Vdt[a:a + bs]).argmax(1).cpu().numpy() for a in range(0, n_dev, bs)]) == ref).mean())
    if os.environ.get('RESUME') == '1':                                      # the tuned formulas already saved: only write the weights
        Ps = np.load(OUT / f'{tag}_alpha.npz'); Wc.data.copy_(T(Ps['coef'].T)); Wz.data.copy_(T(Ps['cz'].T)); epochs = 0
    a0 = agree(); best = (a0, Wc.detach().clone(), Wz.detach().clone(), 0); path = [(0, a0)]
    log(f'  start ({src} weight formulas, ParT values): {100 * a0:.2f}%, {time.time() - t0:.0f} s')
    opt = torch.optim.Adam([Wc, Wz], lr)
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, bs)):
            L = logits(Fft[a:a + bs], okft[a:a + bs], Zft[a:a + bs], Vft[a:a + bs]); loss = -(pf[a:a + bs] * torch.log_softmax(L, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if device == 'mps': torch.mps.empty_cache()
        ag = agree(); path.append((ep + 1, ag))
        if ag > best[0]: best = (ag, Wc.detach().clone(), Wz.detach().clone(), ep + 1)
        log(f'  A2 epoch {ep + 1}: formula weights + ParT values {100 * ag:.2f}%, {time.time() - t0:.0f} s')
    COEF2, CZ2 = best[1].cpu().numpy().T, best[2].cpu().numpy().T
    np.savez(OUT / f'{tag}_alpha.npz', kn=kn, mu=mu, sd=sd, coef=COEF2, cz=CZ2)
    # the tuned formula α of the 100k fit jets and the dev jets, for S12 on top
    def alpha_of(J, rows, Z, chunk=2000):                                   # in batches of bs jets on the GPU (the full term matrix of 2000 jets does not fit)
        A = np.zeros((2, len(rows), 8, 129), np.float16)
        with torch.no_grad():
            for a in range(0, len(rows), chunk):
                r = rows[a:a + chunk]; F, ok = feats(J, r, model); F = np.concatenate([F, rel_feats(F, ok)], -1)
                for c0 in range(0, len(r), bs):
                    sl = slice(c0, c0 + bs); okc = ok[sl]; P = int(okc.sum(1).max()); Ft, okt = T(F[sl, :P]), T(okc[:, :P], torch.bool)
                    rel = torch.softmax((phi(Ft) @ best[1]).masked_fill(~okt[..., None], -1e9), 1).cpu().numpy(); acls = torch.sigmoid(-(T(Z[a + c0:a + c0 + len(okc)]) @ best[2])).cpu().numpy()
                    for hh in range(16):
                        A[hh // 8, a + c0:a + c0 + len(okc), hh % 8, 0] = acls[:, hh]; A[hh // 8, a + c0:a + c0 + len(okc), hh % 8, 1:1 + P] = (1 - acls[:, hh:hh + 1]) * rel[:, :, hh]
        if device == 'mps': torch.mps.empty_cache()
        return A
    np.save(OUT / f'{tag}_alpha_fit.npy', alpha_of(Jf, rows_of_split('fit', n_jet), Zf_all)); np.save(OUT / f'{tag}_alpha_dev.npy', alpha_of(Jd, rd, Zd))
    r = dict(experiment=tag, source=src, n_fit=n_fit, epochs=epochs, lr=lr, start=a0, best=best[0], best_epoch=best[3], path=path, seconds=time.time() - t0)
    if os.environ.get('RESUME') == '1': r['best'] = max(r['best'], json.loads((OUT / f'{tag}_alpha_e2e.json').read_text())['best']) if (OUT / f'{tag}_alpha_e2e.json').exists() else r['best']
    (OUT / f'{tag}_alpha_e2e.json').write_text(json.dumps(r, indent=1)); log(f'{tag}: formula weights tuned for the decision, with ParT values {100 * best[0]:.2f}% (start {100 * a0:.2f}%)'); return r


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; run(*a)
