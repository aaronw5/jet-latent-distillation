"""The class fit: ParT's per-particle embeddings (the input of its two class-attention blocks) written as formulas of
each particle's own inputs, fed through ParT's own class-attention blocks, final LayerNorm and last layer (all fixed).

  python -m jetdistill.part.clsfit extract [n_fit n_dev]   # embeddings of the first n_fit training / n_dev validation jets

extract: ParT is run again from the ROOT files (float32 inputs) for the rows of the cached 'fit' and 'dev' jets (the same
order); results/_cls/<net>/<which>/: x_f16.npy (J, 128, 128) the embeddings (0 in empty slots), mask.npy, z.npy, L.npy."""
import json, sys, time
import numpy as np
from .. import config
from ..config import DATA, RESULTS


def log(*a):
    print(*a, flush=True)


def rows_of(which, n):
    S = np.load(DATA / 'splits.npz'); return S[which][:n]


def extract(net='full', n_fit=20000, n_dev=10000, log=log):
    from .data import read_root
    from .network import ParTNetwork
    files = json.loads((DATA / 'train' / 'files.json').read_text()); off = np.cumsum([0] + [f['jets'] for f in files])
    model = ParTNetwork(net); Lc = np.load(DATA / 'train' / f'net_{net}.npz')['L']
    for which, n in ((w, k) for w, k in (('fit', n_fit), ('dev', n_dev)) if k):
        t0 = time.time(); r = rows_of(which, n); d = RESULTS / '_cls' / str(net) / which; d.mkdir(parents=True, exist_ok=True)
        X = np.lib.format.open_memmap(d / 'x_f16.npy', 'w+', np.float16, (len(r), 128, 128)); M = np.zeros((len(r), 128), bool)
        Z = np.zeros((len(r), 128), np.float32); L = np.zeros((len(r), 10), np.float32)
        for fi, f in enumerate(files):                     # every file once: its rows among r
            pos = np.flatnonzero((r >= off[fi]) & (r < off[fi + 1]))
            if not len(pos): continue
            J = read_root(f['file']); J = {k: v[r[pos] - off[fi]] for k, v in J.items()}
            R = model.run_internals(J); X[pos] = R['x']; M[pos] = R['mask']; Z[pos] = R['z']; L[pos] = R['logits']
            log(f'  {which}: {f["file"]}: {len(pos)} jets, {time.time() - t0:.0f} s')
        X.flush(); np.save(d / 'mask.npy', M); np.save(d / 'z.npy', Z); np.save(d / 'L.npy', L)
        agree = (L.argmax(1) == Lc[r].argmax(1)).mean(); dl = np.abs(L - Lc[r]).max()
        log(f'{which}: {len(r)} jets in {time.time() - t0:.0f} s; logits vs the stored ones: max |diff| {dl:.1e}, same class {100 * agree:.2f}%')


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'extract': extract('full', *(int(x) for x in a))      # n_dev = 0: skip; the validation split is stored sorted: take all of it
