"""S11 — ParT's class-attention weights α as formulas, on top of S5 (S5's value formulas kept). Per head:
  α_i = (1 − α_cls) · softmax_i(g(x_i)),   α_cls = 1 / (1 + exp(u(jet)))
  g   per-particle score formula: hinge terms of own + neighbourhood + ParT pair-kernel context + within-jet relative
      inputs (rank, difference to the jet's maximum, standardized within the jet); least squares to ParT's
      log(α_i / α_cls), centred within each jet and weighted by ParT's α_i (the particles that carry the head count)
  u   jet-level formula of the class token's share: ridge on the quantile-normalized jet quantities (+ ln n)
Measured on ParT's exact downstream (balanced dev jets): ParT's values with (formula relative weights + ParT's α_cls),
(formula relative + formula α_cls); S5's formula values with formula α in block 1, and in both blocks.
Saves the formulas (research/results/S11_alpha.npz) for S12 (S5's values re-tuned on top).

  python -m jetdistill.research.alpha_fit [n_fit n_dev]"""
import json, os, sys, time
TAG = os.environ.get('ALPHA_TAG', 'S11')
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, downstream, phi_pooled, rows_of_split, OUT
from .heads_attn import feats
from .weights import internals, logits as part_logits
from .ceiling_jet import matrix

REL = [(2, 'ln pT/pT_jet'), (13, 'tanh d0'), (4, 'ΔR'), (7, 'charge')]           # inputs also given relative to their jet


def rel_feats(F, ok):
    """within-jet relative inputs: rank (0 = largest, / n), x − max over the jet, (x − mean)/std over the jet"""
    out = []; n = ok.sum(1, keepdims=True).clip(1)
    for f, _ in REL:
        x = np.abs(F[..., f]) if f == 13 else F[..., f]; xm = np.where(ok, x, -np.inf)
        rank = np.argsort(np.argsort(-xm, 1), 1) / n; mx = xm.max(1, keepdims=True)
        mu = np.where(ok, x, 0).sum(1, keepdims=True) / n; sd = np.sqrt(np.where(ok, (x - mu) ** 2, 0).sum(1, keepdims=True) / n) + 1e-6
        out += [rank, x - mx, (x - mu) / sd]
    return (np.stack(out, -1) * ok[..., None]).astype(np.float32)


def run(n_fit=20000, n_dev=20000, n_jet=100000, device='mps', log=print):
    import torch
    t0 = time.time(); model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    _, Af, Mf, _, _ = extract(model, 'fit', n_jet, device); _, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    Ff, okf = feats(Jf, rf, model); Fd, okd = feats(Jd, rd, model); assert (okf == Mf[:n_fit]).all() and (okd == Md).all()
    Ff = np.concatenate([Ff, rel_feats(Ff, okf)], -1); Fd = np.concatenate([Fd, rel_feats(Fd, okd)], -1); nf = Ff.shape[-1]
    kn = np.quantile(Ff[okf][::5], np.linspace(.15, .85, 5), axis=0).astype(np.float32); mu, sd = Ff[okf].mean(0), Ff[okf].std(0) + 1e-6
    phi = lambda F: np.concatenate([(F - mu) / sd] + [np.maximum(0, F - k) / sd for k in kn], -1).astype(np.float32)
    log(f'  {nf} inputs per particle ({len(REL) * 3} within-jet relative), {6 * nf} terms, {time.time() - t0:.0f} s')
    jid = np.repeat(np.arange(n_fit), okf.sum(1)); cnt = np.bincount(jid, minlength=n_fit)
    B = phi(Ff[okf]); Bd = phi(Fd[okd])
    # jet-level inputs for α_cls
    keys = [k for k in Jf['Q']]; Qf, Qd = matrix(Jf, keys)[:n_jet], matrix(Jd, keys)[rd]; okq = np.isfinite(Qf).all(0) & (Qf.std(0) > 0)
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=500, output_distribution='normal', subsample=50000, random_state=0).fit(Qf[:, okq])
    Zf = np.concatenate([np.clip(qt.transform(Qf[:, okq]), -5, 5), np.log(Mf.sum(1, keepdims=True)), np.ones((n_jet, 1))], 1); Zd = np.concatenate([np.clip(qt.transform(Qd[:, okq]), -5, 5), np.log(Md.sum(1, keepdims=True)), np.ones((n_dev, 1))], 1)
    Gz = Zf.T.astype(np.float64) @ Zf; Gz += 1.0 * np.diag(np.diag(Gz)).mean() * 1e-3 * np.eye(len(Gz))
    res = dict(n_fit=n_fit, n_dev=n_dev, n_jet=n_jet, inputs=nf, heads=[]); Anew = Ad.copy(); Arel = Ad.copy(); COEF = np.zeros((16, B.shape[1])); CZ = np.zeros((16, Zf.shape[1]))
    for b in range(2):
        for h in range(8):
            a = Af[b, :n_fit, h]; acls = a[:, 0]; ap = a[:, 1:][okf]
            y = np.log(np.maximum(ap, 1e-12)) - np.log(np.maximum(acls, 1e-12))[jid]                     # log(α_i / α_cls)
            w = ap / np.maximum(1 - acls, 1e-9)[jid] + 1.0 / cnt[jid]                                     # importance: the relative weight (+ a floor)
            jm = lambda v: np.bincount(jid, v * w, minlength=n_fit) / np.bincount(jid, w, minlength=n_fit)  # weighted mean within each jet
            yc = y - jm(y)[jid]; Bc = B - np.stack([jm(B[:, c]) for c in range(B.shape[1])], 1)[jid]
            Bw = Bc * np.sqrt(w)[:, None]; G = Bw.T.astype(np.float64) @ Bw; dd = np.sqrt(np.maximum(np.diag(G), 1e-12))
            coef = np.linalg.solve(G / dd[:, None] / dd[None] + 1e-4 * np.eye(len(dd)), (Bw.T.astype(np.float64) @ (yc * np.sqrt(w))) / dd) / dd
            # α_cls: u = log((1 − α_cls)/α_cls), jet-level ridge
            ua = Af[b, :, h, 0]; u = np.log(np.maximum(1 - ua, 1e-9)) - np.log(np.maximum(ua, 1e-9)); cz = np.linalg.solve(Gz, Zf.T.astype(np.float64) @ u)
            COEF[8 * b + h] = coef; CZ[8 * b + h] = cz
            ud = Zd @ cz; uad = Ad[b, :, h, 0]; u_true = np.log(np.maximum(1 - uad, 1e-9)) - np.log(np.maximum(uad, 1e-9)); r2u = 1 - ((ud - u_true) ** 2).mean() / u_true.var()
            # dev: relative weights from the formula, α_cls from ParT (Arel) or from the jet formula (Anew)
            g = np.full((n_dev, 128), -np.inf, np.float32); g[okd] = Bd @ coef; rel = np.exp(g - g.max(1, keepdims=True)); rel /= rel.sum(1, keepdims=True)
            Arel[b, :, h, 1:] = (1 - Ad[b, :, h, :1]) * rel; acls_f = 1 / (1 + np.exp(ud)); Anew[b, :, h, 0] = acls_f; Anew[b, :, h, 1:] = (1 - acls_f)[:, None] * rel
            tv = 0.5 * np.abs(Arel[b, :, h, 1:] - Ad[b, :, h, 1:]).sum(1).mean()                           # how far the formula's weights are from ParT's (total variation)
            res['heads'].append(dict(block=b + 1, head=h + 1, tv_relative=float(tv), r2_alpha_cls_logit=float(r2u)))
            log(f'  block {b + 1} head {h + 1}: weights off ParT’s by {100 * tv:.1f} % (total variation); α_cls logit R² {r2u:.3f}, {time.time() - t0:.0f} s')
    np.savez(OUT / f'{TAG}_alpha.npz', kn=kn, mu=mu, sd=sd, coef=COEF, cz=CZ)
    def alpha_of(J, rows, Z, chunk=4000):                                   # the formula α (2, n, 8, 129) of the jets `rows`
        A = np.zeros((2, len(rows), 8, 129), np.float16)
        for a in range(0, len(rows), chunk):
            r = rows[a:a + chunk]; F, ok = feats(J, r, model); F = np.concatenate([F, rel_feats(F, ok)], -1); Bp = phi(F[ok]); acls = 1 / (1 + np.exp(Z[a:a + chunk] @ CZ.T))     # (c, 16)
            for hh in range(16):
                g = np.full((len(r), 128), -np.inf, np.float32); g[ok] = Bp @ COEF[hh]; rel = np.exp(g - g.max(1, keepdims=True)); rel /= rel.sum(1, keepdims=True)
                A[hh // 8, a:a + len(r), hh % 8, 0] = acls[:, hh]; A[hh // 8, a:a + len(r), hh % 8, 1:] = (1 - acls[:, hh:hh + 1]) * rel
        return A
    np.save(OUT / f'{TAG}_alpha_fit.npy', alpha_of(Jf, rows_of_split('fit', n_jet), Zf)); np.save(OUT / f'{TAG}_alpha_dev.npy', alpha_of(Jd, rd, Zd)); log(f'  formula α of {n_jet} fit + {n_dev} dev jets saved, {time.time() - t0:.0f} s')
    # the agreements: ParT's values with formula weights, S5's formula values with formula weights
    S, V = internals(model, *(np.load((__import__('jetdistill.config', fromlist=['RESULTS']).RESULTS / '_cls' / 'full' / 'dev' / f), mmap_mode='r')[rd] for f in ('x_f16.npy', 'mask.npy')), device)
    toS = lambda A: np.where(A > 0, np.log(np.maximum(A, 1e-30)), -np.inf).astype(np.float32)
    def part_with(A, blocks):
        Sn = S.copy()
        for b in blocks: Sn[b] = toS(A[b])
        return float((part_logits(model, Sn, V, device) == ref).mean())
    P5 = np.load(OUT / 'S5_model.npz'); W5 = torch.from_numpy(P5['W']).to(device)
    def s5_with(A):
        P, _ = phi_pooled(Jd, rd, A, P5['kn'])
        with torch.no_grad(): return float((np.concatenate([downstream(model, *torch.einsum('nhk,hko->nho', torch.from_numpy(P[a:a + 5000]).to(device), W5).split(8, 1)).argmax(1).cpu().numpy() for a in range(0, n_dev, 5000)]) == ref).mean())
    mix = lambda Aform, blocks: np.stack([Aform[b] if b in blocks else Ad[b] for b in range(2)])
    r = res['agreement'] = dict(
        part_values_formula_rel_block1=part_with(Arel, [0]), part_values_formula_rel_and_cls_block1=part_with(Anew, [0]), part_values_formula_all_blocks=part_with(Anew, [0, 1]),
        s5_values_part_alpha=s5_with(Ad), s5_values_formula_alpha_block1=s5_with(mix(Anew, [0])), s5_values_formula_alpha_both=s5_with(Anew), s5_values_formula_rel_part_cls_block1=s5_with(mix(Arel, [0])))
    for k, v in r.items(): log(f'  {k}: {100 * v:.2f}%')
    res['seconds'] = time.time() - t0; return res


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; r = run(*a); (OUT / f'{TAG}_alpha.json').write_text(json.dumps(r, indent=1)); print('RESULT', json.dumps(r['agreement']))
