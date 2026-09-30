"""The cached jets of a ParT network (the counterpart of pipeline.jets for JEDI): particles, labels, the network's outputs
(stored with the dataset, see data.py) and every observable, in results/_jets/<network>/<which>/.

which: 'fit' (training jets, the first N_STEP4_FIT of the shuffled 'fit' split), 'dev' (validation), 'explain', 'test'
(the step-1 test split), 'full_test' (the whole test folder: millions of jets). Observables are written one file per
quantity (float32) and, for 'full_test', read back only when used; particles are kept for the first 20,000 test jets
(the export check and the example jets need them)."""
import json, time
import numpy as np
from .. import config
from ..config import DATA, RESULTS, NETWORKS
from ..observables import compute, library

KEEP_X = 20000
LAZY = ('full_test',)


def rows(which):
    S = np.load(DATA / 'splits.npz')
    if which == 'full_test': return 'test', np.arange(len(np.load(DATA / 'test' / 'label.npy', mmap_mode='r')))
    stop = dict(fit=config.N_STEP4_FIT, dev=config.N_DEV, explain=config.N_EXPLAIN, test=config.N_TEST_SPLIT)[which]
    return ('test' if which == 'test' else 'train'), S[which][:stop]


class LazyQ(dict):
    """observables read from their files when used (as float64)"""
    def __init__(self, d, ids):
        super().__init__({k: None for k in ids}); self.d = d

    def __getitem__(self, k):
        if k not in self: raise KeyError(k)
        return np.load(self.d / 'Q' / f'{k}.npy', mmap_mode='r').astype(np.float64)

    def items(self): return ((k, self[k]) for k in self)
    def values(self): return (self[k] for k in self)


def build(n, which, log=print, chunk=50000):
    t0 = time.time(); folder, r = rows(which)
    if which == 'full_test' and config.N_FULL_TEST: r = r[:config.N_FULL_TEST]
    d = RESULTS / '_jets' / str(n) / which; (d / 'Q').mkdir(parents=True, exist_ok=True); src = DATA / folder
    X = np.load(src / 'x_f16.npy', mmap_mode='r'); E = np.load(src / 'ext_f16.npy', mmap_mode='r'); jet = np.load(src / 'jet.npy')[r]
    net = np.load(src / f'net_{n}.npz'); np.save(d / 'Z.npy', net['Z'][r]); np.save(d / 'L.npy', net['L'][r])
    np.save(d / 'y.npy', np.load(src / 'label.npy')[r]); np.save(d / 'jet.npy', jet)
    keep = r[:KEEP_X] if which in LAZY else r; order = np.argsort(keep)          # sorted reads of the memory-mapped files
    for name, A in (('x', X), ('ext', E)):
        a = np.empty((len(keep),) + A.shape[1:], np.float32); a[order] = A[keep[order]]; np.save(d / f'{name}.npy', a)
    ids = list(library(n)); out = {k: np.lib.format.open_memmap(d / 'Q' / f'{k}.npy', 'w+', np.float32, (len(r),)) for k in ids}
    for s in range(0, len(r), chunk):
        rr = r[s:s + chunk]; o = np.argsort(rr); xs = np.empty((len(rr),) + X.shape[1:], np.float32); es = np.empty((len(rr),) + E.shape[1:], np.float32)
        xs[o] = X[rr[o]]; es[o] = E[rr[o]]
        Q = compute(xs, n, jet=jet[s:s + chunk], ext=es)
        for k in ids: out[k][s:s + len(rr)] = Q[k]
        if len(r) > chunk: log(f'  observables {which}: {s + len(rr)} / {len(r)} jets, {time.time() - t0:.0f} s')
    for a in out.values(): a.flush()
    (d / 'meta.json').write_text(json.dumps(dict(folder=folder, n_jets=int(len(r)), ids=ids)))
    log(f'jets {which}: {len(r)} in {time.time() - t0:.0f} s')


def jets(n, which, log=print):
    """dict(x, jet, ext, y, Q, net, P, L, Z, H) as pipeline.jets gives it (H = Z: no activation)"""
    from ..metrics import softmax
    d = RESULTS / '_jets' / str(n) / which
    if not (d / 'meta.json').exists(): build(n, which, log)
    M = json.loads((d / 'meta.json').read_text()); lazy = which in LAZY
    Q = LazyQ(d, M['ids']) if lazy else {k: np.load(d / 'Q' / f'{k}.npy').astype(np.float64) for k in M['ids']}
    L = np.load(d / 'L.npy').astype(np.float64); Z = np.load(d / 'Z.npy', mmap_mode='r' if lazy else None)
    Z = Z if lazy else Z.astype(np.float64)
    return dict(x=np.load(d / 'x.npy'), jet=np.load(d / 'jet.npy'), ext=np.load(d / 'ext.npy'), y=np.load(d / 'y.npy').astype(int), Q=Q,
                net=L.argmax(1), P=softmax(L), L=L, Z=Z, H=Z)
