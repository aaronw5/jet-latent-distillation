"""Step 1: each of the network's neurons as a sum of terms of observables (MARS-style forward selection, Friedman 1991).

Target (JEDI): the neuron's value before its ReLU, z, clipped below at −0.2·(spread of its positive values): below 0 the
ReLU erases every difference, so the fit only follows z a little way into the negative range. ParT: the neuron itself
(no activation follows it). R² is always measured on what the last layer reads (JEDI: max(0, ·)).
Candidate terms of an observable Q: Q itself, max(0, Q − t) and max(0, t − Q) for every candidate threshold t (the 5 %,
10 %, …, 95 % quantiles of Q on the fitting jets; for mass observables also m_W, m_Z, m_H, m_t and ×½, ×1.5, ×2 when the
setup offers them), and products of an already chosen threshold term with a threshold term of one of the 10 observables
most correlated with the current residual.
Each step adds the candidate that lowers the squared error most (on 12,000 of the fitting jets), refits all coefficients
by least squares on all 40,000 fitting jets, and scores R² of max(0, formula) against the neuron on 25,000 validation jets;
the kept length is the best validation R² (stop after 5 steps without improvement, or 100 terms). Only the network's
neuron values are used, never the classes.
ParT: the jet-level observables are candidates as above; ParT's per-particle and pair inputs (~35,000 quantities) are
screened at every step: the 100 whose value or distance from its median correlates most with the current residual (on
6,000 of the selection jets) join the candidates. Neurons are fitted in parallel processes.
Output: results/<setup>/n<N>/step1.json (terms in the order chosen; later steps take the first K of each neuron)."""
import json, os, re, sys, time
import numpy as np
from . import config, data
from .config import SETUPS, MASSES, MASS_MULT, RESULTS, RELU, TAGGER, net_dir
from .formula import act
from .network import Network
from .observables import library, compute, mass_ids
from .observables.library import is_block


def log(*a):
    print(*a, flush=True)

N_KNOTS, MAX_TERMS, N_SELECT, STALL = 19, (10 if config.SMOKE else 100), 12000, 5
N_SCREEN, N_TOP, N_CACHE = 6000, 100, 700      # ParT blocks: screening jets, screened candidates per step, kept candidate matrices


def splits(n, untrained=False, sizes=None):
    """{split: (particles, network outputs, observables)} for the fitting, validation and test jets of step 1"""
    sizes = sizes or dict(fit=config.N_STEP1_FIT, dev=config.N_DEV, test=config.N_TEST_SPLIT)
    if TAGGER == 'part' and not untrained:          # the network outputs are stored with the jets
        from .pipeline import jets, sub
        out = {s: (lambda J: (J['x'], dict(z=J['Z'], h=J['H'], logits=J['L']), J['Q']))(sub(jets(n, s), N)) for s, N in sizes.items()}
        return out, Network(n)
    net = Network(n, untrained=untrained); out = {}
    for s, N in sizes.items():
        x = data.particles(s, n, stop=N); out[s] = (x, net.run(x), compute(x, n))
    return out, net


def candidate_observables(setup, n, Q, lowlevel=False, log=log):
    """the observables step 1 may use in this setup (finite and not constant on every split)"""
    lib = library(n); drop = mass_ids(n) if setup.no_mass else set()
    if setup.strict: drop |= strict_equivalents(Q['fit'], drop, log)   # also every observable that is (m / ΣpT)² in disguise
    keep = [k for k in lib if k not in drop and not is_block(k) and (not lowlevel or re.fullmatch(r'(pt|eta|phi)_\d+', k))]
    return [k for k in keep if all(np.isfinite(Q[s][k]).all() for s in Q) and np.ptp(Q['fit'][k]) > 0]


def strict_equivalents(Qfit, drop=(), log=None):
    """observables (not already dropped) with |correlation| > 0.98 with (m / ΣpT)² on the training jets"""
    ref = Qfit['mass_over_sum_pt_sq']; out = set()
    for k, v in Qfit.items():
        if k in drop or not (np.ptp(v) > 0 and np.isfinite(v).all()): continue
        r = np.corrcoef(v, ref)[0, 1]
        if abs(r) > 0.98:
            out.add(k)
            if log: log(f'strict: excluded {k} (correlation {r:.4f})')
    return out


def thresholds(v, mass_like, offer_masses):
    ks = set(np.unique(np.round(np.quantile(v, np.linspace(.05, .95, N_KNOTS)), 12)).tolist())
    if mass_like and offer_masses:
        lo, hi = np.quantile(v, [.02, .98])
        ks |= {float(c * m) for m in MASSES.values() for c in MASS_MULT if lo < c * m < hi}
    return sorted(ks)


SMOOTH = False          # step 1 of a setup with gelu=True: threshold terms are s·GELU((±(Q − t))/s), s = the local threshold spacing


def widths(kn):
    kn = np.asarray(kn, np.float64)
    if len(kn) < 2: return [max(abs(kn[0]) * 0.1, 1e-6) if len(kn) else 1.0] * len(kn)
    d = np.diff(kn); w = np.concatenate([[d[0]], (d[:-1] + d[1:]) / 2, [d[-1]]]); return [float(max(x, 1e-9)) for x in w]


def cols(v, kn):
    """candidate columns of one quantity: Q, then the threshold terms above and below each threshold; descriptors (kind, t, s)"""
    if SMOOTH:
        from .formula import gelu
        ws = widths(kn)
        M = [v] + [s * gelu((v - t) / s) for t, s in zip(kn, ws)] + [s * gelu((t - v) / s) for t, s in zip(kn, ws)]
        return M, [('lin', None, None)] + [('gt', t, s) for t, s in zip(kn, ws)] + [('lt', t, s) for t, s in zip(kn, ws)]
    return [v] + [np.maximum(0, v - t) for t in kn] + [np.maximum(0, t - v) for t in kn], [('lin', None, None)] + [('gt', t, None) for t in kn] + [('lt', t, None) for t in kn]


def _one(v, kind, t, s=None):
    from .formula import factor
    return factor(v, kind, t, s)


def term_value(tm, Q):
    v = _one(Q[tm['q']], tm['kind'], tm.get('t'), tm.get('s'))
    return v * _one(Q[tm['q2']], tm['kind2'], tm.get('t2'), tm.get('s2')) if tm.get('q2') else v


class Screen:
    """ParT's per-particle and pair inputs on the first N_SCREEN selection jets, standardised, for the screening; their
    candidate terms are built when a quantity is screened in (the last N_CACHE are kept)"""
    def __init__(self, n, J, sel, Qfit, drop, offer_masses):
        from .observables.compute import block_matrix
        rows = sel[:N_SCREEN]; ids, M = block_matrix(n, J['x'][rows], J['jet'][rows], J['ext'][rows])
        ok = (np.ptp(M, 0) > 0) & np.array([k not in drop for k in ids]); self.ids = [k for k, o in zip(ids, ok) if o]; M = M[:, ok]
        A = np.abs(M - np.median(M, 0)); self.V = ((M - M.mean(0)) / M.std(0)).astype(np.float32); del M
        self.A = ((A - A.mean(0)) / np.maximum(A.std(0), 1e-12)).astype(np.float32); del A
        self.Qfit, self.sel, self.offer, self.mids, self.cache, self.knots = Qfit, sel, offer_masses, mass_ids(n), {}, {}

    def top(self, r):
        rs = (r[:N_SCREEN] - r[:N_SCREEN].mean()).astype(np.float32)
        sc = np.maximum(np.abs(rs @ self.V), np.abs(rs @ self.A)); return [self.ids[i] for i in np.argsort(-sc)[:N_TOP]]

    def cand(self, k):
        if k not in self.cache:
            v = self.Qfit[k]; kn = self.knots.setdefault(k, thresholds(v, k in self.mids, self.offer)); vs = v[self.sel]
            if len(self.cache) >= N_CACHE: self.cache.pop(next(iter(self.cache)))
            M, D = cols(vs, kn); self.cache[k] = (np.stack(M, 1).astype(np.float32), D)
        return self.cache[k]


def fit_neuron(z, Q, keys, knots, sel, log=None, screen=None):
    """forward selection for one neuron; z = {split: pre-activation}. Returns (terms, coefficients incl. intercept, path)"""
    zf, zd = z['fit'], z['dev']; NF = len(zf)
    zpos = zf[zf > 0]; floor = -0.2 * (zpos.std() if len(zpos) > 50 else zf.std()); target = np.maximum(zf, floor) if RELU else zf
    cache = {}
    for k in keys:
        v = Q['fit'][k][sel]
        M, D = cols(v, knots[k]); cache[k] = (np.stack(M, 1).astype(np.float32), D)
    chosen, Bsel, path = [], [np.ones(len(sel))], []; best = (-np.inf, 0); stall = 0
    ys = target[sel]; hd = act(zd)
    def gain(C, Qb, r):
        Cp = C - Qb @ (Qb.T @ C); nn = (Cp ** 2).sum(0); g = np.where(nn > 1e-9, (Cp.T @ r) ** 2 / np.maximum(nn, 1e-12), 0); i = int(g.argmax()); return float(g[i]), i
    for _ in range(MAX_TERMS):
        Qb, _ = np.linalg.qr(np.stack(Bsel, 1)); r = ys - Qb @ (Qb.T @ ys); Qb32, r32 = Qb.astype(np.float32), r.astype(np.float32)
        cand, rel = [], []
        get = lambda k: cache[k] if k in cache else screen.cand(k)
        for k in keys + (screen.top(r) if screen else []):
            C, D = get(k); g, i = gain(C, Qb32, r32); cand.append((g, dict(q=k, kind=D[i][0], t=D[i][1], **({'s': D[i][2]} if D[i][2] else {}))))
            v = C[:, 0]; rel.append((abs(np.corrcoef(v, r)[0, 1]) if np.ptp(v) > 0 else 0, k))
        top = [k for _, k in sorted(rel, reverse=True)[:10]]
        for tm in chosen:                                   # products: a chosen threshold term × a threshold term of a related observable
            if tm.get('q2') is not None or tm['kind'] == 'lin': continue
            pv = term_value(tm, {tm['q']: Q['fit'][tm['q']][sel]}).astype(np.float32)
            for k in top:
                if k == tm['q']: continue
                C, D = get(k); g, i = gain(C[:, 1:] * pv[:, None], Qb32, r32)
                cand.append((g, dict(q=tm['q'], kind=tm['kind'], t=tm['t'], **({'s': tm['s']} if tm.get('s') else {}), q2=k, kind2=D[i + 1][0], t2=D[i + 1][1], **({'s2': D[i + 1][2]} if D[i + 1][2] else {}))))
        g, tm = max(cand, key=lambda c: c[0])
        if g <= 0: break
        chosen.append(tm); Bsel.append(term_value(tm, {k: Q['fit'][k][sel] for k in (tm['q'], tm.get('q2')) if k}))
        coef = _lstsq(chosen, Q['fit'], target, NF)
        Bd = np.stack([np.ones(len(zd))] + [term_value(t, Q['dev']) for t in chosen], 1)
        r2 = 1 - ((hd - act(Bd @ coef)) ** 2).mean() / max(hd.var(), 1e-12); path.append(round(float(r2), 4))
        if r2 > best[0] + 1e-4: best, stall = (r2, len(chosen)), 0
        else: stall += 1
        if stall >= STALL or r2 > .9995: break
    chosen = chosen[:best[1]]
    return chosen, _lstsq(chosen, Q['fit'], target, NF), path


def _lstsq(terms, Qf, target, NF):
    B = np.stack([np.ones(NF)] + [term_value(t, Qf) for t in terms], 1); return np.linalg.lstsq(B, target, rcond=None)[0]


_G = {}          # the state the neuron fits share (inherited by the worker processes)


def _one_neuron(j):
    Q, Z, keys, knots, sel, screen = (_G[k] for k in ('Q', 'Z', 'keys', 'knots', 'sel', 'screen'))
    zj = {s: Z[s][:, j] for s in Z}; t0 = time.time()
    if np.ptp(zj['fit']) < 1e-9: return dict(neuron=j, intercept=float(zj['fit'].mean()), terms=[], constant=True)
    terms, coef, path = fit_neuron(zj, Q, keys, knots, sel, screen=screen)
    return _finish(j, terms, coef, path, zj, Q, t0)


def _finish(j, terms, coef, path, zj, Q, t0):
    """a neuron's step-1 result: R² on the test split, each term's importance (R² lost without it), the log lines"""
    Bt = np.stack([np.ones(len(zj['test']))] + [term_value(t, Q['test']) for t in terms], 1); ht = act(zj['test']); zt = Bt @ coef
    r2h = lambda z: float(1 - ((ht - act(z)) ** 2).mean() / max(ht.var(), 1e-12)); r2 = r2h(zt)
    imp = [round(r2 - r2h(zt - c * Bt[:, i + 1]), 5) for i, c in enumerate(coef[1:])]         # R² lost when the term alone is left out
    log(f'neuron {j}: {len(terms)} terms, R² on test jets {r2:.4f}, {time.time() - t0:.0f} s')
    if r2 >= 0.95:                                   # the main quantities of a well-described neuron (R² lost without them)
        from .observables import symbol
        by = {}
        for t, m in zip(terms, imp):
            for q in {t['q'], t.get('q2')} - {None}: by[q] = by.get(q, 0) + m
        log(f'  neuron {j} features: ' + ', '.join(f'{symbol(q)} ({v:.3f})' for q, v in sorted(by.items(), key=lambda x: -x[1])[:8]))
    return dict(neuron=j, intercept=float(coef[0]), terms=[dict(t, coef=float(c), importance=m) for t, c, m in zip(terms, coef[1:], imp)], dev_r2_path=path, test_r2=r2)


def _one_thread():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)          # one BLAS thread per worker process (no oversubscription)


def run(setup_name, n, untrained=False, lowlevel=False, out=None, log=log, workers=None):
    global SMOOTH
    t0 = time.time(); setup = SETUPS[setup_name]; SMOOTH = bool(setup.gelu)
    S, net = splits(n, untrained)
    Q = {s: S[s][2] for s in S}; Z = {s: S[s][1]['z'] for s in S}
    keys = candidate_observables(setup, n, Q, lowlevel, log); mids = mass_ids(n)
    knots = {k: thresholds(Q['fit'][k], k in mids, setup.masses_as_thresholds) for k in keys}
    sel = np.random.default_rng(0).choice(len(Z['fit']), min(N_SELECT, len(Z['fit'])), replace=False); screen = None
    if TAGGER == 'part' and not lowlevel:
        from .pipeline import jets, sub
        drop = mids if setup.no_mass else set()
        screen = Screen(n, sub(jets(n, 'fit'), config.N_STEP1_FIT), sel, Q['fit'], drop, setup.masses_as_thresholds)
        log(f'screening: {len(screen.ids)} per-particle and pair quantities, {time.time() - t0:.0f} s')
    log(f'{len(keys)} observables, {sum(1 + 2 * len(v) for v in knots.values())} candidate terms, {time.time() - t0:.0f} s')
    _G.update(Q=Q, Z=Z, keys=keys, knots=knots, sel=sel, screen=screen)
    workers = workers or int(os.environ.get('JETDISTILL_WORKERS', 4 if TAGGER == 'part' else 1))
    gpu = os.environ.get('JETDISTILL_STEP1_DEVICE')
    if gpu and screen is not None:               # the same selection on a GPU (mars_gpu), all neurons in lockstep
        from . import mars_gpu
        from .observables.compute import block_matrix
        from .pipeline import jets, sub
        Jf = sub(jets(n, 'fit'), config.N_STEP1_FIT); bids, BM = block_matrix(n, Jf['x'][sel], Jf['jet'][sel], Jf['ext'][sel])
        R = mars_gpu.select(Z, Q, keys, knots, sel, screen, (bids, BM), device=gpu, log=log); del BM
        neurons = []
        for j in range(Z['fit'].shape[1]):
            zj = {s_: Z[s_][:, j] for s_ in Z}
            neurons.append(_finish(j, *R[j], zj, Q, time.time()) if j in R else dict(neuron=j, intercept=float(zj['fit'].mean()), terms=[], constant=True))
    elif workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers, initializer=_one_thread) as pool: neurons = pool.map(_one_neuron, range(Z['fit'].shape[1]), chunksize=1)
    else:
        neurons = [_one_neuron(j) for j in range(Z['fit'].shape[1])]
    used = sorted({t['q'] for nr in neurons for t in nr['terms']} | {t['q2'] for nr in neurons for t in nr['terms'] if t.get('q2')})
    res = dict(setup=setup_name, n=n, untrained=untrained, lowlevel=lowlevel, observables=keys, n_thresholds={k: len(v) for k, v in knots.items()},
               screened=dict(n=len(screen.ids), used=[k for k in used if is_block(k)]) if screen else None,
               jets=dict(fit=len(Z['fit']), dev=len(Z['dev']), test=len(Z['test'])), neurons=neurons)
    out = out or RESULTS / setup_name / net_dir(n) / 'step1.json'; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1))
    return res


def load(setup_name, n):
    """the step-1 neurons; a setup with step1_from reuses that setup's step 1"""
    src = SETUPS[setup_name].step1_from or setup_name
    return json.loads((RESULTS / src / net_dir(n) / 'step1.json').read_text())['neurons']


if __name__ == '__main__':
    run(sys.argv[1], config.parse_net(sys.argv[2]))
