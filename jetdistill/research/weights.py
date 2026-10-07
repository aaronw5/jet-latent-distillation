"""S7 — how precise must ParT's class-attention weights be? ParT's own per-particle values v (both blocks), its own
downstream; only the weights α are swapped, block by block:
  uniform     α ∝ 1 over [class token, particles] (a plain mean)
  uniform_p   the particles uniform, the class token keeping ParT's own self-weight (its share)
  formula     α from a per-particle score formula (hinge terms of own + neighbourhood + ParT pair-kernel context,
              least squares to ParT's scores, centred within each jet), ParT's own self-score
Block 2's query depends on block 1's output; here block 2's true scores are used where block 2 is kept.

  python -m jetdistill.research.weights [n_fit n_dev]"""
import json, sys, time, pathlib
import numpy as np
from ..config import RESULTS
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import after
from .heads_attn import feats

OUT = pathlib.Path(__file__).resolve().parents[2] / 'research' / 'results'


def internals(model, X, M, device='mps'):
    """per jet: both blocks' scores s (2, N, 8, 1+P) and values v (2, N, 1+P, 8, 16) of [class token, particles]"""
    import torch
    N = len(M); S = np.full((2, N, 8, 129), -np.inf, np.float32); V = np.zeros((2, N, 129, 8, 16), np.float16)
    with torch.no_grad():
        for a in range(0, N, 1000):
            x = torch.from_numpy(np.asarray(X[a:a + 1000], np.float32)).to(device).permute(1, 0, 2); m = torch.from_numpy(M[a:a + 1000]).to(device); n = x.shape[1]
            cls = model.cls_token.expand(1, n, -1)
            for b, blk in enumerate(model.cls_blocks):
                u = blk.pre_attn_norm(torch.cat([cls, x], 0)); W, bi = blk.attn.in_proj_weight, blk.attn.in_proj_bias
                q = (cls @ W[:128].T + bi[:128]).view(1, n, 8, 16); k = (u @ W[128:256].T + bi[128:256]).view(-1, n, 8, 16); v = (u @ W[256:].T + bi[256:]).view(-1, n, 8, 16)
                s = (torch.einsum('qnhd,pnhd->nhp', q, k) / 4.0).masked_fill(~torch.cat([torch.ones_like(m[:, :1]), m], 1)[:, None], -float('inf'))
                S[b, a:a + n] = s.cpu().numpy(); V[b, a:a + n] = v.permute(1, 0, 2, 3).cpu().numpy()
                o = torch.einsum('nhp,pnhd->nhd', torch.softmax(s, -1), v); cls = after(blk, o, cls[0])[None]
    return S, V


def logits(model, S, V, device='mps'):
    import torch
    out = []
    with torch.no_grad():
        for a in range(0, S.shape[1], 2000):
            n = min(2000, S.shape[1] - a); cls = model.cls_token[0, 0].expand(n, -1)
            for b, blk in enumerate(model.cls_blocks):
                al = torch.softmax(torch.from_numpy(S[b, a:a + n]).to(device), -1); v = torch.from_numpy(V[b, a:a + n].astype(np.float32)).to(device)
                cls = after(blk, torch.einsum('nhp,nphd->nhd', al, v), cls)
            out.append(model.fc(model.norm(cls)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def run(n_fit=20000, n_dev=10000, device='mps', log=print):
    t0 = time.time(); model = ParTNetwork('full').model; model.eval(); D = RESULTS / '_cls' / 'full'
    Xf, Mf = np.load(D / 'fit' / 'x_f16.npy', mmap_mode='r')[:n_fit], np.load(D / 'fit' / 'mask.npy')[:n_fit]
    Xd, Md, Ld = np.load(D / 'dev' / 'x_f16.npy', mmap_mode='r')[:n_dev], np.load(D / 'dev' / 'mask.npy')[:n_dev], np.load(D / 'dev' / 'L.npy')[:n_dev]
    Sf, _ = internals(model, Xf, Mf, device); Sd, Vd = internals(model, Xd, Md, device); ref = Ld.argmax(1)
    res = dict(n_fit=n_fit, n_dev=n_dev, exact=float((logits(model, Sd, Vd, device) == ref).mean()))
    log(f'  ParT\'s own weights and values: {100 * res["exact"]:.2f}% (must be 100), {time.time() - t0:.0f} s')
    real = np.isfinite(Sd)
    def swap(b, Snew):
        S = Sd.copy(); S[b] = Snew; return float((logits(model, S, Vd, device) == ref).mean())
    for b in range(2):
        res[f'block{b + 1}_uniform'] = swap(b, np.where(real[b], 0.0, -np.inf))
        al = np.exp(Sd[b] - Sd[b].max(-1, keepdims=True)); al /= al.sum(-1, keepdims=True); selfw = al[..., 0]; npart = real[b, ..., 1:].sum(-1)
        u = np.where(real[b], 0.0, -np.inf); u[..., 0] = np.log(np.maximum(selfw, 1e-12) / np.maximum(1 - selfw, 1e-12) * npart)   # particles uniform, own self-weight
        res[f'block{b + 1}_uniform_particles'] = swap(b, u)
        log(f"  block {b + 1} weights uniform: {100 * res[f'block{b + 1}_uniform']:.2f}%; particles uniform (ParT's self-weight): {100 * res[f'block{b + 1}_uniform_particles']:.2f}%")
    # formula scores per head: hinge(own + nbr + pk) per particle, least squares within jets
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); Ff, okf = feats(Jf, n_fit, model); Fd, okd = feats(Jd, n_dev, model)
    kn = np.quantile(Ff[okf][::7], np.linspace(.15, .85, 5), axis=0).astype(np.float32); mu, sd = Ff[okf].mean(0), Ff[okf].std(0) + 1e-6
    phi = lambda F: np.concatenate([(F - mu) / sd] + [np.maximum(0, F - t) / sd for t in kn], -1)
    Bf, Bd = phi(Ff[okf]).astype(np.float32), phi(Fd[okd]).astype(np.float32)
    jf = np.repeat(np.arange(n_fit), okf.sum(1)); cf = np.bincount(jf)
    jm = np.zeros((n_fit, Bf.shape[1]), np.float64); np.add.at(jm, jf, Bf); Bfc = Bf - (jm / cf[:, None])[jf]          # centred within each jet
    G = Bfc.T.astype(np.float64) @ Bfc; dd = np.sqrt(np.maximum(np.diag(G), 1e-12))
    for b in range(2):
        Snew = Sd[b].copy()
        for h in range(8):
            y = Sf[b, :, h, 1:][okf]; yc = y - (np.bincount(jf, y) / cf)[jf]
            w = np.linalg.solve(G / dd[:, None] / dd[None] + 1e-6 * np.eye(len(dd)), (Bfc.T.astype(np.float64) @ yc) / dd) / dd
            p = Bd @ w; s = Snew[:, h, 1:].copy(); s[okd] = p
            js = np.where(okd, s, 0).sum(1) / okd.sum(1); ts = np.where(okd, Sd[b, :, h, 1:], 0).sum(1) / okd.sum(1)
            s = s - js[:, None] + ts[:, None]                               # keep each jet's mean score (the self-weight balance) as ParT's
            Snew[:, h, 1:] = np.where(okd, s, -np.inf)
        res[f'block{b + 1}_formula'] = swap(b, Snew)
        log(f"  block {b + 1} weights from per-particle score formulas (ParT's per-jet mean score kept): {100 * res[f'block{b + 1}_formula']:.2f}%, {time.time() - t0:.0f} s")
    res['seconds'] = time.time() - t0
    return res


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; r = run(*a); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'S7_weights.json').write_text(json.dumps(r, indent=1)); print('RESULT', json.dumps(r))
