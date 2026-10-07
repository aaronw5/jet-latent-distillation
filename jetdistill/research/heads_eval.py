"""The per-head formula models that keep ParT's class-attention weights (S5: both blocks' weights; S8: block 1's,
block 2 a plain average) on a split: agreement with ParT and the paper's metrics. On the test jets ParT's particle
blocks are run once per file (read_root → run_internals) for the weights; the formulas take the stored per-particle
arrays. Alignment check: ParT's logits from the files must match the stored ones.

  python -m jetdistill.research.heads_eval S5|S8_uniform2 dev|full_test"""
import json, sys, time, pathlib
import numpy as np
from ..config import RESULTS, DATA
from ..pipeline import jets
from ..metrics import softmax, paper_metrics
from ..part.network import ParTNetwork
from .heads import extract, attend, downstream, phi_pooled, rows_of_split, OUT


def uniformize(A, M, blocks):
    for b in blocks:
        okm = np.concatenate([np.ones((len(M), 1), bool), M], 1); A[b] = (okm / okm.sum(1, keepdims=True))[:, None, :]
    return A


def weights_from_files(net, model, rows, J, device, log, t0):
    """ParT's class-attention weights (2, n, 8, 129) and logits of the test jets `rows` (global, within one file)"""
    import torch
    from ..part.data import read_root
    files = json.loads((DATA / 'test' / 'files.json').read_text()); off = np.cumsum([0] + [f['jets'] for f in files])
    fi = int(np.searchsorted(off, rows[0], side='right') - 1); assert rows[-1] < off[fi + 1]
    Jr = read_root(files[fi]['file'], start=int(rows[0] - off[fi]), stop=int(rows[-1] - off[fi] + 1))
    R = net.run_internals(Jr); X, M = R['x'], R['mask']; n = len(rows); A = np.zeros((2, n, 8, 129), np.float32)
    with torch.no_grad():
        for a in range(0, n, 1000):
            x = torch.from_numpy(np.asarray(X[a:a + 1000], np.float32)).to(device).permute(1, 0, 2); m = torch.from_numpy(M[a:a + 1000]).to(device)
            cls = model.cls_token.expand(1, x.shape[1], -1)
            for b, blk in enumerate(model.cls_blocks):
                o, al = attend(blk, cls, x, m); A[b, a:a + 1000] = al.cpu().numpy(); cls = blk(x, x_cls=cls, padding_mask=~m)
    return A, M, R['logits']


def evaluate(tag='S5', which='full_test', device='mps', chunk=5000, log=print):
    import torch
    t0 = time.time(); Pz = np.load(OUT / f'{tag}_model.npz'); W, kn, uni = Pz['W'], Pz['kn'], str(Pz['uniform']); blocks = [int(c) - 1 for c in uni]
    net = ParTNetwork('full'); model = net.model; model.eval(); T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); Wt = T(W)
    J = jets('full', which)
    if which == 'dev':
        rows = rows_of_split('dev', 20000); _, A, M, _, _ = extract(model, 'dev', 20000, device); A = uniformize(A, M, blocks)
        P, _ = phi_pooled(J, rows, A, kn); L = np.concatenate([downstream(model, *torch.einsum('nhk,hko->nho', T(P[a:a + chunk]), Wt).split(8, 1)).cpu().numpy() for a in range(0, len(rows), chunk)])
    else:
        rows = np.arange(len(J['y'])); L = np.empty((len(rows), 10), np.float32); Lp = np.empty_like(L)
        for a in range(0, len(rows), chunk):
            r = rows[a:a + chunk]; A, M, Lp[r] = weights_from_files(net, model, r, J, device, log, t0); A = uniformize(A, M, blocks)
            P, _ = phi_pooled(J, r, A, kn); L[r] = downstream(model, *torch.einsum('nhk,hko->nho', T(P), Wt).split(8, 1)).cpu().numpy()
            if a % 100000 == 0: log(f'  {tag} on {which}: {a + len(r)} of {len(rows)} jets; ParT from the files vs stored: same class {100 * (Lp[:a + len(r)].argmax(1) == J["net"][:a + len(r)]).mean():.2f}%, {time.time() - t0:.0f} s')
        agree_check = float((Lp.argmax(1) == J['net']).mean()); assert agree_check > .999, agree_check
    Pr = softmax(L.astype(np.float64)); pred = Pr.argmax(1); netc, y = J['net'][rows], J['y'][rows]
    r = dict(model=tag, which=which, n=len(pred), agreement=float((pred == netc).mean()), **paper_metrics(Pr, y))
    if which == 'full_test':
        r['part'] = paper_metrics(J['P'], y); r['per_class'] = [dict(c=int(c), agreement=float((pred[netc == c] == c).mean()), accuracy=float((pred[y == c] == c).mean())) for c in range(10)]
        np.save(OUT / f'{tag}_test_logits.npy', L.astype(np.float16))
    (OUT / f'{tag}_metrics_{which}.json').write_text(json.dumps(r, indent=1))
    log(f'{tag} on {which} ({len(pred)} jets): same class as ParT {100 * r["agreement"]:.2f}%, accuracy {100 * r["accuracy"]:.2f}%, AUC {r["auc"]:.4f}, {time.time() - t0:.0f} s')
    return r


if __name__ == '__main__':
    evaluate(*sys.argv[1:])
