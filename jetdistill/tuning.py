"""Steps 2 and 3: all coefficients tuned together, then pruning with retraining.

Step 2 keeps the terms and thresholds of step 1 (the first K terms of each neuron, coefficients refitted by least squares)
and trains every coefficient and intercept at once with Adam, full batch, on 100,000 fitting jets (ParT: 50,000). Targets:
  probabilities  soft cross-entropy to the network's class probabilities, through the network's own last layer
                 (JEDI: formula neurons rounded to the network's fixed-point grid, straight-through gradient)
  decisions      cross-entropy to the network's class (one-hot)
  labels         cross-entropy to the true class
  neurons        no class scores: only R (below)
plus λ·R for the first three, R = mean over neurons of (h_j − h_j,network)² / Var(h_j,network) (keeps each formula neuron
close to the network's; the neurons (16 / 128) reach the probabilities only through 5 / 10 class scores, so without R the
tuning can move them in directions the probabilities do not see).
The kept step (checked every 50 of 800) is the one with the best validation score: agreement with the network's class
(probabilities, decisions), accuracy (labels) or mean neuron R² (neurons).
Step 3 (ParT) first replaces whole neurons by constants (prune_neurons: 114 of ParT's 128 neurons change no decision),
then removes the weakest terms (|coefficient| × spread of the term; ParT: × the norm of the neuron's weights in the last
layer, i.e. the size of the term's effect on the class scores: 114 of ParT's 128 neurons change no decision), 15 % at a time, retraining after each cut; a cut is
kept while the validation score stays within TOL of step 2's (0.1 point; 0.005 of R² for 'neurons'), else the cut size
is halved, down to 2 %."""
import numpy as np
from . import formula as F

from .config import SMOKE, RELU, NC, TAGGER


def log(*a):
    print(*a, flush=True)
STEPS, LR = (800 if not SMOKE else 100), 3e-3
STEPS_BY_TARGET = dict(labels=500) if not SMOKE else {}     # as in the original runs
TOL = dict(probabilities=0.001, decisions=0.001, labels=0.001, neurons=0.005)


def refit(neurons, Q, Z, ridge=1e-6):
    """least-squares coefficients of the given terms to each neuron's pre-activation Z (clipped below as in step 1)"""
    out = []
    for nr in neurons:                                  # neuron by neuron (memory: all float64 matrices at once are ~30 GB at 300k jets)
        B = F.bases([nr], Q)[0]
        z = Z[:, nr['neuron']]; zp = z[z > 0]; tgt = np.maximum(z, -0.2 * (zp.std() if len(zp) > 50 else z.std())) if RELU else z
        if not nr['terms']: out.append(dict(nr, intercept=float(tgt.mean()))); continue
        sd = B.std(0) + 1e-12; Bn = (B - B.mean(0)) / sd
        c = np.linalg.solve(Bn.T @ Bn + ridge * len(Bn) * np.eye(Bn.shape[1]), Bn.T @ (tgt - tgt.mean())) / sd
        out.append(dict(nr, intercept=float(tgt.mean() - (B.mean(0) * c).sum()), terms=[dict(t, coef=float(a)) for t, a in zip(nr['terms'], c)]))
    return out


def neuron_r2(H, Hn):
    """mean over neurons of 1 − MSE / Var (neurons constant in the network left out)"""
    v = Hn.var(0); ok = v > 1e-9
    return float(np.mean(1 - ((H - Hn) ** 2).mean(0)[ok] / v[ok]))


def train(formula, Qf, Qd, target, fit, dev, last, lam=0.0, steps=None, lr=LR, history=None):
    """fit/dev: dict(P=network probabilities, y=true class, net=network class, H=network neurons (ReLU)) of those jets.
    Returns (formula with trained coefficients, validation score of the kept step). history: a list that receives, every 50
    steps, the loss and the score on the training and on the validation jets (analysis/convergence.py)."""
    import jax, jax.numpy as jnp
    steps = steps or STEPS_BY_TARGET.get(target, STEPS)
    K7, b7 = (jnp.asarray(x, jnp.float32) for x in last[:2]); rounds = last[2] is not None
    if rounds: i7, f7 = (jnp.asarray(x, jnp.float32) for x in last[2:])
    a = jax.nn.relu if RELU else (lambda x: x)
    # neuron by neuron (the same float32 arrays; the float64 matrices of all neurons at once need twice the memory)
    B = [jnp.asarray(F.bases([nr], Qf)[0], jnp.float32) for nr in formula]; Bd = [jnp.asarray(F.bases([nr], Qd)[0], jnp.float32) for nr in formula]
    sd = [jnp.asarray(np.asarray(b).std(0) + 1e-9) for b in B]
    th = [(jnp.asarray(c, jnp.float32) * s, jnp.asarray(c0, jnp.float32)) for (c, c0), s in zip(F.coefs(formula), sd)]   # scaled coefficients

    def hidden(th, Bs):
        return jnp.stack([a(b @ (c / s) + c0) for (c, c0), b, s in zip(th, Bs, sd)], 1)

    def logits(th, Bs):
        h = hidden(th, Bs)
        if not rounds: return h @ K7 + b7
        sc = 2.0 ** f7; hq = (jnp.floor(h * sc + 0.5) / sc) % (2.0 ** i7)
        return (h + jax.lax.stop_gradient(hq - h)) @ K7 + b7

    # the jets' matrices and targets are arguments of the jitted functions, not closure constants: jax embeds closed-over
    # arrays in the compiled program (about 3x their size in memory: 25 GB for 8 GB of matrices at 100k jets)
    Hn = jnp.asarray(fit['H'], jnp.float32); vn = jnp.asarray(np.asarray(fit['H']).var(0) + 1e-6, jnp.float32)
    R = lambda th, Bs, Hs: jnp.mean((hidden(th, Bs) - Hs) ** 2 / vn)
    T = {'probabilities': fit['P'], 'decisions': np.eye(NC)[fit['net']], 'labels': np.eye(NC)[fit['y']]}.get(target)
    data = (B, Hn, None if T is None else jnp.asarray(T, jnp.float32))
    if target == 'neurons':
        loss = lambda th, D: R(th, D[0], D[1])
    else:
        loss = (lambda th, D: -jnp.mean(jnp.sum(D[2] * jax.nn.log_softmax(logits(th, D[0])), 1)) + lam * R(th, D[0], D[1])) if lam > 0 else \
               (lambda th, D: -jnp.mean(jnp.sum(D[2] * jax.nn.log_softmax(logits(th, D[0])), 1)))
    out = jax.jit(lambda th, Bs: hidden(th, Bs) if target == 'neurons' else logits(th, Bs))
    if target == 'neurons':
        score = lambda th: neuron_r2(np.asarray(out(th, Bd)), dev['H'])
    else:
        ref = dev['y'] if target == 'labels' else dev['net']
        score = lambda th: float((np.asarray(out(th, Bd)).argmax(1) == ref).mean())
    if history is not None:
        ref_f = fit['y'] if target == 'labels' else fit['net']
        score_fit = (lambda th: neuron_r2(np.asarray(out(th, B)), fit['H'])) if target == 'neurons' else (lambda th: float((np.asarray(out(th, B)).argmax(1) == ref_f).mean()))
    vg = jax.jit(jax.value_and_grad(loss)); g = lambda th: vg(th, data); tm = jax.tree_util.tree_map
    m_, v_ = tm(jnp.zeros_like, th), tm(jnp.zeros_like, th); best = (-np.inf, th)
    for i in range(steps):
        _, gr = g(th); m_ = tm(lambda a, b: .9 * a + .1 * b, m_, gr); v_ = tm(lambda a, b: .999 * a + .001 * b * b, v_, gr)
        th = tm(lambda p, a, b: p - lr * (a / (1 - .9 ** (i + 1))) / (jnp.sqrt(b / (1 - .999 ** (i + 1))) + 1e-8), th, m_, v_)
        if i % 50 == 49 or i == steps - 1:
            s = score(th)
            if history is not None: history.append(dict(step=i + 1, loss=float(loss(th, data)), train=score_fit(th), val=s))
            if s > best[0]: best = (s, th)
    return F.with_coefs(formula, [(np.asarray(c / s, np.float64), float(c0)) for (c, c0), s in zip(best[1], sd)]), best[0]


def prune(formula, Qf, Qd, target, fit, dev, last, lam=0.0, frac=0.15, min_frac=0.02, log=log):
    """step 3; returns (pruned formula, path of (terms, validation score))"""
    cur, ref = train(formula, Qf, Qd, target, fit, dev, last, lam); path = [(F.n_terms(cur), ref)]
    log(f'step 2: {path[-1][0]} terms, validation {ref:.4f}')
    if TAGGER == 'part' and target != 'neurons':
        cur = prune_neurons(cur, ref, Qf, Qd, target, fit, dev, last, lam, path, log)
    while frac >= min_frac:
        wn = np.linalg.norm(np.asarray(last[0]), axis=1) if TAGGER == 'part' else np.ones(len(cur))   # ParT: size of the term's effect on the class scores
        imp = sorted((abs(t['coef']) * float(b[:, i].std()) * float(wn[nr['neuron']]), j, i) for j, (nr, b) in enumerate(zip(cur, F.bases(cur, Qf))) for i, t in enumerate(nr['terms']))
        if not imp: break
        cut = {(j, i) for _, j, i in imp[:max(1, int(len(imp) * frac))]}
        trial, s = train([dict(nr, terms=[t for i, t in enumerate(nr['terms']) if (j, i) not in cut]) for j, nr in enumerate(cur)], Qf, Qd, target, fit, dev, last, lam)
        if s >= ref - TOL[target]:
            cur = trial; path.append((F.n_terms(cur), s)); log(f'  kept a cut of {len(cut)}: {path[-1][0]} terms, validation {s:.4f}')
        else:
            frac /= 2
    return cur, path


def decision_flips(formula, Q, last):
    """per neuron: the share of jets whose class (the formula's) changes when that neuron is fixed at its mean"""
    H = F.hidden(formula, Q); L = F.logits_from_h(H, *last); pred = L.argmax(1); K = np.asarray(last[0])
    return np.array([((L + np.outer(H[:, j].mean() - H[:, j], K[j])).argmax(1) != pred).mean() if np.ptp(H[:, j]) > 0 else 0.0 for j in range(H.shape[1])])


def prune_neurons(cur, ref, Qf, Qd, target, fit, dev, last, lam, path, log=log):
    """step 3a (ParT): whole neurons replaced by constants (no terms; the intercept is retrained), the neurons whose
    fixing changes fewest validation decisions first; the first batch is every neuron that changes fewer than 0.1 %
    of them; a batch is kept while the validation score stays within TOL of step 2's, else it is halved"""
    k = None
    while True:
        fl = decision_flips(cur, Qd, last); live = [j for j, nr in enumerate(cur) if nr['terms']]
        order = sorted(live, key=lambda j: fl[j])
        if k is None: k = max(1, int((fl[live] < 0.001).sum()))
        k = min(k, len(order))
        if k < 1: break
        drop = set(order[:k]); Hf = F.hidden(cur, Qf)
        trial = [dict(nr, terms=[], intercept=float(Hf[:, j].mean())) if j in drop else nr for j, nr in enumerate(cur)]
        trial, sc = train(trial, Qf, Qd, target, fit, dev, last, lam)
        if sc >= ref - TOL[target]:
            cur = trial; path.append((F.n_terms(cur), sc))
            log(f'  kept {sum(1 for nr in cur if nr["terms"])} of {len(cur)} neurons (cut {k}): {path[-1][0]} terms, validation {sc:.4f}')
        else:
            k //= 2
    return cur


def targets(net_run, y):
    """the tuning targets of a set of jets from the network's outputs"""
    L = net_run['logits']; P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    return dict(P=P, net=L.argmax(1), y=y, H=F.act(net_run['z']))
