"""The all-formula class attention (S10 model, research/results/S10_model.npz) on a split: agreement with ParT and the
paper's metrics. Checked on the balanced dev sample first (must give the tuning's validation agreement).

  python -m jetdistill.research.joint_eval dev|full_test [model.npz]"""
import json, sys, time, pathlib
import numpy as np
from ..config import RESULTS
from ..pipeline import jets
from ..metrics import softmax, paper_metrics
from ..part.network import ParTNetwork
from .heads import downstream, rows_of_split
from .joint import Model, feats_rows, OUT


def logits(which, path, device='mps', chunk=2000, log=print):
    import torch
    t0 = time.time(); Pz = np.load(path); model = ParTNetwork('full').model; model.eval()
    mdl = Model(dict(kn=Pz['kns'], mu=Pz['mus'], sd=Pz['sds']), Pz['knv'], Pz['muv'], Pz['sdv'], device)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); Ws, Wv, cs, b = T(Pz['Ws']), T(Pz['Wv']), T(Pz['cself']), T(Pz['bias'])
    J = jets('full', which); rows = rows_of_split('dev', 20000) if which == 'dev' else np.arange(len(J['y'])); n = len(rows); L = np.empty((n, 10), np.float32)
    with torch.no_grad():
        for a in range(0, n, chunk):
            r = rows[a:a + chunk]; F, ok = feats_rows(J, r, model); P = int(ok.sum(1).max())
            o = mdl.heads(torch.from_numpy(F[:, :P]).to(device).half().float(), torch.from_numpy(ok[:, :P]).to(device), Ws, Wv, cs, b)
            L[a:a + len(r)] = downstream(model, o[:, :8], o[:, 8:]).cpu().numpy()
            if a % 100000 == 0: log(f'  {which}: {a + len(r)} of {n} jets, {time.time() - t0:.0f} s')
    return L, J, rows


def evaluate(which='full_test', path=None, log=print):
    path = pathlib.Path(path) if path else OUT / 'S10_model.npz'
    L, J, rows = logits(which, path, log=log); P = softmax(L.astype(np.float64)); pred = P.argmax(1); net, y = J['net'][rows], J['y'][rows]
    r = dict(model=str(path.name), which=which, n=len(pred), agreement=float((pred == net).mean()), **paper_metrics(P, y))
    if which == 'full_test':
        r['part'] = paper_metrics(J['P'], y); r['per_class'] = [dict(c=int(c), agreement=float((pred[net == c] == c).mean()), accuracy=float((pred[y == c] == c).mean())) for c in range(10)]
        np.save(path.with_name(path.stem + '_test_logits.npy'), L.astype(np.float16))
    (path.with_name(path.stem + f'_metrics_{which}.json')).write_text(json.dumps(r, indent=1))
    log(f'{path.name} on {which} ({len(pred)} jets): same class as ParT {100 * r["agreement"]:.2f}%, accuracy {100 * r["accuracy"]:.2f}%, AUC {r["auc"]:.4f}')
    return r


if __name__ == '__main__':
    evaluate(*sys.argv[1:])
