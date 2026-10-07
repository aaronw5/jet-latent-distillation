"""S4 — ParT's class attention head by head, as formulas (the user's idea; JEDI with per-head neurons).

ParT's two class blocks: each head h of block b returns o_bh = Σ_{j ∈ [cls, particles]} α_bhj · v_bhj (16 numbers),
v = W_v · LN(token). Everything after the heads is a fixed chain of ParT's own operations (out_proj, per-head scale,
LayerNorms, MLP, residuals; block 2; final LN; last layer) — kept exactly (downstream()).
Formula per head: o_bh ≈ W_bh · [Σ_i α_bhi φ(particle i), α_bh,self, 1], φ = hinge terms of each particle's physics
(18 own inputs + 20 neighbourhood features; 5 thresholds each) — a sum over particles of per-particle functions, as in
JEDI-linear, with the head's weights α. Weights: 'oracle' = ParT's own α (are the head values formula-shaped?).

  python -m jetdistill.research.heads [n_fit n_dev]"""
import json, sys, time, pathlib
import numpy as np
from ..config import RESULTS
from ..pipeline import jets
from ..part.clsfit import particle_features, PFEAT
from ..part.network import ParTNetwork
from .nbr import nbr_features, NBR

OUT = pathlib.Path(__file__).resolve().parents[2] / 'research' / 'results'


def attend(blk, cls, x, m):
    """one class block's attention, by hand: head outputs o (N, 8, 16) and weights α (N, 8, 1 + P)"""
    import torch
    n = x.shape[1]; u = blk.pre_attn_norm(torch.cat([cls, x], 0)); W, b = blk.attn.in_proj_weight, blk.attn.in_proj_bias
    q = (cls @ W[:128].T + b[:128]).view(1, n, 8, 16); k = (u @ W[128:256].T + b[128:256]).view(-1, n, 8, 16); v = (u @ W[256:].T + b[256:]).view(-1, n, 8, 16)   # the query: the raw class token (ParT: attn(x_cls, u, u))
    s = torch.einsum('qnhd,pnhd->nhp', q, k) / 4.0
    s = s.masked_fill(~torch.cat([torch.ones_like(m[:, :1]), m], 1)[:, None], -float('inf')); a = torch.softmax(s, -1)
    return torch.einsum('nhp,pnhd->nhd', a, v), a


def after(blk, o, cls):
    """the rest of a class block from its head outputs o (N, 8, 16): out_proj, per-head scale, post-attn LN, residual, MLP"""
    import torch
    n = o.shape[0]; y = o.reshape(n, 128) @ blk.attn.out_proj.weight.T + blk.attn.out_proj.bias
    y = torch.einsum('nhd,h->ndh', y.view(n, 8, 16), blk.c_attn).reshape(n, 128); y = blk.post_attn_norm(y); c1 = y + cls
    z = blk.post_fc_norm(blk.act(blk.fc1(blk.pre_fc_norm(c1)))); z = blk.fc2(z)
    return z + blk.w_resid * c1


def downstream(model, o1, o2):
    """class scores from the 2 × 8 head outputs, by ParT's own operations"""
    cls0 = model.cls_token[0, 0].expand(o1.shape[0], -1)
    c1 = after(model.cls_blocks[0], o1, cls0); c2 = after(model.cls_blocks[1], o2, c1)
    return model.fc(model.norm(c2))


def rows_of_split(which, n):
    """the extracted jets used: the first n of the (shuffled) fit split; for dev a random n of the 100k (the dev split is
    stored sorted by file, so its first jets are mostly H→bb / H→cc)"""
    total = len(np.load(RESULTS / '_cls' / 'full' / which / 'mask.npy'))
    return np.arange(min(n, total)) if which == 'fit' else np.sort(np.random.default_rng(0).choice(total, min(n, total), replace=False))


def extract(model, which, n, device='mps'):
    """ParT's head outputs and weights of both class blocks for the jets rows_of_split(which, n)"""
    import torch
    d = RESULTS / '_cls' / 'full' / which; X = np.load(d / 'x_f16.npy', mmap_mode='r'); rows = rows_of_split(which, n); n = len(rows); M = np.load(d / 'mask.npy')[rows]; L = np.load(d / 'L.npy')[rows]
    O = np.zeros((2, n, 8, 16), np.float32); A = np.zeros((2, n, 8, 129), np.float32); Ld = np.zeros((n, 10), np.float32)
    with torch.no_grad():
        for a in range(0, n, 1000):
            x = torch.from_numpy(np.asarray(X[rows[a:a + 1000]], np.float32)).to(device).permute(1, 0, 2); m = torch.from_numpy(M[a:a + 1000]).to(device)
            cls = model.cls_token.expand(1, x.shape[1], -1); os_ = []
            for b, blk in enumerate(model.cls_blocks):
                o, al = attend(blk, cls, x, m); O[b, a:a + 1000] = o.cpu().numpy(); A[b, a:a + 1000] = al.cpu().numpy(); os_.append(o)
                cls = after(blk, o, cls[0])[None]
            Ld[a:a + 1000] = downstream(model, *os_).cpu().numpy()
    return O, A, M, L, Ld


def phi_pooled(J, rows, A, kn=None, chunk=2000):
    """Σ_i α_bhi φ(particle i) for all 16 heads: (n, 16, K), and the thresholds"""
    out = None; n = len(rows)
    for a in range(0, n, chunk):
        r = rows[a:a + chunk]; F, ok = particle_features(J, r, ctx=[]); F = np.concatenate([F, nbr_features(J['x'][r], J['ext'][r], J['jet'][r])], -1)
        if kn is None: kn = np.quantile(F[ok][::3], np.linspace(.15, .85, 5), axis=0).astype(np.float32)
        Phi = np.concatenate([F] + [np.maximum(0, F - t) for t in kn], -1) * ok[..., None]       # (c, 128, K)
        w = A[:, a:a + len(r), :, 1:].transpose(1, 0, 2, 3).reshape(len(r), 16, 128)             # particle weights of the 16 heads
        P = np.einsum('nhp,npk->nhk', w, Phi, optimize=True)
        if out is None: out = np.zeros((n, 16, P.shape[-1] + 2), np.float32)
        sl = slice(a, a + len(r)); out[sl, :, :-2] = P; out[sl, :, -2] = A[:, sl, :, 0].transpose(1, 0, 2).reshape(len(r), 16); out[sl, :, -1] = 1
    return out, kn


def formula_weights(model, J, rows, A_true, M, params=None, n_score_fit=20000, chunk=2000, log=print):
    """S9: block-1 attention weights from per-particle score formulas. Per head h: s_i = φ(particle i)·w_h with φ the
    hinge terms of its own inputs, neighbourhood and ParT pair-kernel context; the class token's own score is 0 (in
    block 1 it is a constant, absorbed by the formula's intercept); α = softmax over [0, s_1, …, s_n]. Least squares on
    the first n_score_fit jets to ParT's log(α_i / α_self). Returns A (n, 8, 129) and the formula's parameters."""
    from .nbr import nbr_features, pk_features
    def feats_rows(r):
        F, ok = particle_features(J, r, ctx=[])
        return np.concatenate([F, nbr_features(J['x'][r], J['ext'][r], J['jet'][r]), pk_features(J['x'][r], J['ext'][r], J['jet'][r], model)], -1), ok
    if params is None:
        n = len(rows); n_score_fit = min(n_score_fit, n); F, ok = feats_rows(rows[:n_score_fit]); kn = np.quantile(F[ok][::7], np.linspace(.15, .85, 5), axis=0).astype(np.float32); mu, sd = F[ok].mean(0), F[ok].std(0) + 1e-6
        params = dict(kn=kn, mu=mu, sd=sd)
        phi = lambda F: np.concatenate([np.ones(F.shape[:-1] + (1,), np.float32), (F - mu) / sd] + [np.maximum(0, F - t) / sd for t in kn], -1)
        B = phi(F[ok]).astype(np.float64); al = A_true[:n_score_fit]
        Y = (np.log(np.maximum(al[..., 1:], 1e-12)) - np.log(np.maximum(al[..., :1], 1e-12))).transpose(0, 2, 1)[ok]       # (particles, 8)
        import os; rdg = float(os.environ.get('JOINT_RIDGE', 1e-6))
        G = B.T @ B; d = np.sqrt(np.maximum(np.diag(G), 1e-12)); W = np.linalg.solve(G / d[:, None] / d[None] + rdg * np.eye(len(d)), (B.T @ Y) / d[:, None]) / d[:, None]
        params['W'] = W.astype(np.float32); params['r2'] = 1 - ((B @ W - Y) ** 2).mean(0) / Y.var(0)
        log(f'  block-1 score formulas: {B.shape[1]} terms per particle, R² per head ' + ' '.join(f'{v:.2f}' for v in params['r2']))
    kn, mu, sd, W = params['kn'], params['mu'], params['sd'], params['W']
    phi = lambda F: np.concatenate([np.ones(F.shape[:-1] + (1,), np.float32), (F - mu) / sd] + [np.maximum(0, F - t) / sd for t in kn], -1)
    n = len(rows); A = np.zeros((n, 8, 129), np.float32)
    for a in range(0, n, chunk):
        r = rows[a:a + chunk]; F, ok = feats_rows(r); s = (phi(F) @ W).transpose(0, 2, 1)                              # (c, 8, 128)
        S = np.concatenate([np.zeros((len(r), 8, 1), np.float32), np.where(ok[:, None], s, -np.inf)], -1)
        e = np.exp(S - S.max(-1, keepdims=True)); A[a:a + len(r)] = e / e.sum(-1, keepdims=True)
    return A, params


def run(n_fit=40000, n_dev=20000, steps=0, uniform='', weights='', lr=3e-4, lam=0.01, device='mps', log=print):
    import torch
    t0 = time.time(); model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    Of, Af, Mf, Lf, Lfd = extract(model, 'fit', n_fit, device); Od, Ad, Md, Ld, Ldd = extract(model, 'dev', n_dev, device)
    ref = Ld.argmax(1); res = dict(n_fit=n_fit, n_dev=n_dev, downstream_check=float((Ldd.argmax(1) == ref).mean()))
    log(f'  downstream from ParT\'s own head outputs: same class {100 * res["downstream_check"]:.2f}% (must be 100), {time.time() - t0:.0f} s')
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rows_f, rows_d = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev); n_fit, n_dev = len(rows_f), len(rows_d)
    if weights == 'formula':                                                 # S9: block-1 weights from score formulas, block 2 uniform
        Af[0], prm = formula_weights(model, Jf, rows_f, Af[0], Mf, log=log); Ad[0], _ = formula_weights(model, Jd, rows_d, None, Md, params=prm)
        uniform = '2'; res['weights'] = 'formula'; res['score_r2'] = [float(v) for v in prm['r2']]
        log(f'  block-1 weights from the score formulas, block 2 uniform, {time.time() - t0:.0f} s')
    for b in (int(c) - 1 for c in uniform):                                 # S8: these blocks' weights uniform over [class token, particles]
        for A_, M_ in ((Af, Mf), (Ad, Md)):
            okm = np.concatenate([np.ones((len(M_), 1), bool), M_], 1); A_[b] = (okm / okm.sum(1, keepdims=True))[:, None, :]
    res['uniform_blocks'] = uniform
    Pf, kn = phi_pooled(Jf, rows_f, Af); Pd, _ = phi_pooled(Jd, rows_d, Ad, kn); log(f'  pooled terms per head: {Pf.shape[-1]}, {time.time() - t0:.0f} s')
    Of_, Od_ = Of.transpose(1, 0, 2, 3).reshape(n_fit, 16, 16), Od.transpose(1, 0, 2, 3).reshape(n_dev, 16, 16)
    pred = np.zeros_like(Od_); r2 = np.zeros((16, 16))
    for h in range(16):
        B = Pf[:, h].astype(np.float64); G = B.T @ B; d = np.sqrt(np.maximum(np.diag(G), 1e-12))
        W = np.linalg.solve(G / d[:, None] / d[None] + 1e-7 * np.eye(len(d)), (B.T @ Of_[:, h]) / d[:, None]) / d[:, None]
        pred[:, h] = Pd[:, h] @ W; r2[h] = 1 - ((pred[:, h] - Od_[:, h]) ** 2).mean(0) / Od_[:, h].var(0).clip(1e-12)
    T = lambda q: torch.from_numpy(np.ascontiguousarray(q, np.float32)).to(device)
    def agree(o):
        with torch.no_grad():
            return float((np.concatenate([downstream(model, T(o[a:a + 5000, :8]), T(o[a:a + 5000, 8:])).argmax(1).cpu().numpy() for a in range(0, n_dev, 5000)]) == ref).mean())
    heads = []
    for h in range(16):
        o = Od_.copy(); o[:, h] = pred[:, h]; heads.append(dict(block=h // 8 + 1, head=h % 8 + 1, r2_median=float(np.median(r2[h])), r2_min=float(r2[h].min()), agreement_alone=agree(o)))
        log(f"  block {heads[-1]['block']} head {heads[-1]['head']}: R² median {heads[-1]['r2_median']:.3f} (min {heads[-1]['r2_min']:.3f}); only this head as a formula: {100 * heads[-1]['agreement_alone']:.2f}%")
    res['heads'] = heads; res['all_block1'] = agree(np.concatenate([pred[:, :8], Od_[:, 8:]], 1)); res['all_block2'] = agree(np.concatenate([Od_[:, :8], pred[:, 8:]], 1)); res['all'] = agree(pred)
    log(f"  all block-1 heads as formulas: {100 * res['all_block1']:.2f}%; all block-2 heads: {100 * res['all_block2']:.2f}%; all 16 heads: {100 * res['all']:.2f}% (ParT's own weights), {time.time() - t0:.0f} s")
    if steps:                                                               # S5: tune all heads' coefficients toward ParT's probabilities
        Ms, Us = [], []                                                      # per head: whitened terms (uncorrelated, unit spread) — Adam on raw hinge terms diverges
        for h in range(16):
            B = Pf[:, h].astype(np.float64); G = B.T @ B / len(B); d = np.sqrt(np.maximum(np.diag(G), 1e-12))
            W0 = np.linalg.solve(G / d[:, None] / d[None] + 1e-7 * np.eye(len(d)), (B.T @ Of_[:, h] / len(B)) / d[:, None]) / d[:, None]
            ev, Q = np.linalg.eigh(G / d[:, None] / d[None]); ev = np.maximum(ev, 1e-6 * ev.max())
            Mh = Q / np.sqrt(ev)[None] / d[:, None]; Ms.append(Mh.astype(np.float32)); Us.append((np.sqrt(ev)[:, None] * (Q.T @ (d[:, None] * W0))).astype(np.float32))
        Mt = T(np.stack(Ms)); W = torch.nn.Parameter(T(np.stack(Us))); opt = torch.optim.Adam([W], lr)
        Pft, Pdt = T(Pf), T(Pd); pf = torch.softmax(T(Lf), 1); Oft = T(Of_); vo = Oft.var(0) + 1e-6
        def heads_of(P): return torch.einsum('nhk,hko->nho', P, torch.einsum('hkj,hjo->hko', Mt, W))
        def agree_t():
            with torch.no_grad():
                return float((torch.cat([downstream(model, *heads_of(Pdt[a:a + 5000]).split(8, 1)).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
        best = (agree_t(), 0); path = [best]; bestW = W.detach().clone()
        tag = 'S9' if weights == 'formula' else f'S8_uniform{uniform}' if uniform else 'S5'
        def save(Wb):
            extra = dict(score_W=prm['W'], score_kn=prm['kn'], score_mu=prm['mu'], score_sd=prm['sd']) if weights == 'formula' else {}
            np.savez(OUT / f'{tag}_model.npz', W=torch.einsum('hkj,hjo->hko', Mt, Wb).cpu().numpy(), kn=kn, uniform=uniform, weights=weights or 'part', **extra)
        OUT.mkdir(parents=True, exist_ok=True); save(bestW)
        for i in range(steps):
            for a in range(0, n_fit, 5000):
                o = heads_of(Pft[a:a + 5000]); Lg = downstream(model, o[:, :8], o[:, 8:])
                loss = -(pf[a:a + 5000] * torch.log_softmax(Lg, 1)).sum(1).mean() + lam * ((o - Oft[a:a + 5000]) ** 2 / vo).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if i % 10 == 9 or i == steps - 1:
                ag = agree_t(); path.append((ag, i + 1))
                if ag > best[0]: best = (ag, i + 1); save(W.detach())
                log(f'  S5 tuning epoch {i + 1}: all 16 heads as formulas {100 * ag:.2f}%, {time.time() - t0:.0f} s')
        res['tuned'] = dict(best=best[0], epoch=best[1], path=path, lr=lr, lam=lam)
    res['seconds'] = time.time() - t0
    return res


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:4]]; r = run(*a, uniform=sys.argv[4] if len(sys.argv) > 4 else '', weights=sys.argv[5] if len(sys.argv) > 5 else ''); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ('S9_heads_formula.json' if r.get('weights') else f'S8_heads_uniform{r["uniform_blocks"]}.json' if r['uniform_blocks'] else 'S5_heads_tuned.json' if r.get('tuned') else 'S4_heads_oracle.json')).write_text(json.dumps(r, indent=1)); print('RESULT', json.dumps({k: v for k, v in r.items() if k != 'heads'}))
