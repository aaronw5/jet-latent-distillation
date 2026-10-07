"""Text for the per-head model pages: definitions of the per-particle inputs, how the output is composed, the exact
Python formulation of a model, and the Python-like text of one value neuron."""

DEFS = {
    'ln pT': 'ln pT_i', 'ln E': 'ln E_i', 'ln pT/pT_jet': 'ln(pT_i / pT_jet)', 'ln E/E_jet': 'ln(E_i / E_jet)', 'ΔR': 'ΔR_i = √(Δη_i² + Δφ_i²), to the jet axis', 'Δη': 'η_i − η_jet', 'Δφ': 'φ_i − φ_jet',
    'charge': 'q_i ∈ {−1, 0, +1}', 'charged hadron': '1 if particle i is a charged hadron, else 0', 'neutral hadron': '1 if a neutral hadron', 'photon': '1 if a photon', 'electron': '1 if an electron', 'muon': '1 if a muon',
    'tanh d0': 'tanh(d0_i), d0 the transverse impact parameter', 'σ(d0)': 'min(σ(d0_i), 1)', 'tanh dz': 'tanh(dz_i), dz the longitudinal impact parameter', 'σ(dz)': 'min(σ(dz_i), 1)',
    'ln(1 + pT rank)': 'ln(1 + r_i), r_i = 0 for the hardest particle of the jet, 1 for the next, …',
    'ln(1+n within 0.1)': 'ln(1 + #{j ≠ i : ΔR_ij < 0.1})', 'ln(1+n within 0.2)': 'ln(1 + #{j ≠ i : ΔR_ij < 0.2})',
    'pT share within 0.1': 'Σ_{j ≠ i, ΔR_ij < 0.1} pT_j / Σ_j pT_j', 'pT share within 0.2': 'Σ_{j ≠ i, ΔR_ij < 0.2} pT_j / Σ_j pT_j', 'pT share within 0.4': 'Σ_{j ≠ i, ΔR_ij < 0.4} pT_j / Σ_j pT_j',
    'ΔR nearest': 'min_{j ≠ i} ΔR_ij (capped at 1.5)', 'ln pT nearest / pT': 'ln(pT_nn(i) / pT_i), nn(i) the nearest other particle (clipped to ±10)',
    'ΔR hardest': 'ΔR between i and the hardest particle of the jet (capped at 1.5)', 'ΔR 2nd hardest': 'ΔR between i and the second-hardest particle (capped at 1.5)',
    'ΔR nearest displaced': 'min ΔR_ij over the other particles j that are charged and displaced: q_j ≠ 0 and |d0_j| / σ(d0_j) > 3; 1.5 if the jet has none',
    'ΔR nearest lepton': 'min ΔR_ij over the other particles j that are electrons or muons; 1.5 if none', 'ΔR nearest photon': 'min ΔR_ij over the other photons j; 1.5 if none',
    'prong index': 'k ∈ {0, 1, 2}: the 3 hardest particles seed 3 prongs; every particle belongs to the prong whose seed is nearest in ΔR; k is the index of particle i’s prong (0 = the hardest seed)',
    'prong pT share': 'Σ_{j ∈ prong(i)} pT_j / Σ_j pT_j', 'prong ln mass': 'ln(1 + m(prong(i))), the invariant mass of the summed 4-vectors of the prong’s particles',
    'prong charge': 'Σ_{j ∈ prong(i)} q_j', 'prong n displaced': '#{j ∈ prong(i) : q_j ≠ 0, |d0_j| / σ(d0_j) > 3}', 'pT share in prong': 'pT_i / Σ_{j ∈ prong(i)} pT_j',
    'ΔR prong axis': 'ΔR between i and the seed particle of its prong (capped at 1.5)', 'ln kT with nearest': 'ln(min(pT_i, pT_nn(i)) · ΔR_i,nn(i))'}

PK_DEF = ('Context seen through ParT’s pair kernels: ParT adds to every particle-attention logit a learned bias U_h(i, j) = f_h(ln kT_ij, ln z_ij, ln ΔR_ij, ln m²_ij), one per head h = 1…8 '
          '(kT_ij = min(pT_i, pT_j)·ΔR_ij, z_ij = min(pT_i, pT_j)/(pT_i + pT_j), m_ij the pair mass). With w_hij = softmax_j U_h(i, j) over the other particles, “hk: P” = Σ_{j ≠ i} w_hij P_j for the property P of particle j '
          '(ln pT_j/pT_jet, q_j, its type, tanh d0_j, [charged and |d0_j|/σ > 3], ΔR_j to the axis), and “hk: ln ΔR to i” = Σ_j w_hij ln ΔR_ij.')

COMPOSE = {
    False: """for every jet:
    X[i, :]          = the per-particle inputs of particle i (definitions at the bottom of the page)
    for each class block b (1, 2) and head h (1..8):
        alpha[b, h, :]  = ParT's attention weights over [class token, particles]            # kept from ParT
        o[b, h, :]      = sum_i alpha[b, h, i] * f_bh(X[i, :]) + alpha_cls * c_bh + bias_bh   # 16 value neurons: FORMULAS
                          f_bh(x) = W_bh . [x, max(0, x - t1), ..., max(0, x - t5)]           # per feature, 5 thresholds
    class scores     = ParT_downstream(o)   # fixed arithmetic: out-projection, LayerNorms, MLP, residuals, final LN, last layer""",
    True: """for every jet:
    X[i, :]          = the per-particle inputs of particle i (definitions at the bottom of the page)
    block 1, head h:   s_i = g_h(X[i, :])                       # score FORMULA; the class token's own score is 0
                       alpha = softmax([0, s_1, ..., s_n])      # per jet
    block 2, head h:   alpha = [1, 1, ..., 1] / (n + 1)         # a plain average
    o[b, h, :]       = sum_i alpha_i * f_bh(X[i, :]) + alpha_cls * c_bh + bias_bh   # 16 value neurons: FORMULAS
    class scores     = ParT_downstream(o)   # fixed arithmetic: out-projection, LayerNorms, MLP, residuals, final LN, last layer"""}

HEADS_PY = '''"""Model {tag}: ParT's class attention written per head. Exact formulation; the coefficients are in {tag}_model.npz.

Inputs per jet: X[i, f], the {nv} per-particle features of particle i (definitions below); alpha[b, h, :] the attention
weights of head h of class block b over [the class token, the particles] (they sum to 1).{uniform_note}
In this model the weights are ParT's own (computed from its particle blocks); the values are the formulas below.
The 16 outputs of every head go through ParT's fixed arithmetic (jetdistill/research/heads.py: downstream) to the class scores.
"""
import numpy as np
P = np.load('{tag}_model.npz'); W = P['W']; KN = P['kn'][:, :{nv}]   # W: (16 heads, {nterms} terms, 16 outputs); KN: (5 thresholds, {nv} features)


def terms(X):
    """X: (n_particles, {nv}) -> (n_particles, {nv6}): every feature, then max(0, feature - t) at its 5 thresholds"""
    return np.concatenate([X] + [np.maximum(0, X - KN[k]) for k in range(5)], 1)


def head_values(b, h, X, alpha):
    """the 16 values of head h (0-7) of class block b (0-1); alpha[0] = the class token's own weight, alpha[1:] the particles'"""
    pooled = alpha[1:] @ terms(X)                     # the weighted sum over the particles of every term
    z = np.concatenate([pooled, [alpha[0], 1.0]])     # + the class token's share, + the intercept
    return z @ W[8 * b + h]


def neuron(b, h, o, X, alpha):
    """one value neuron (o = 0..15) written out: sum over the particles of alpha_i * f(X_i) + c * alpha_cls + bias"""
    w = W[8 * b + h][:, o]; t = terms(X)
    return float(alpha[1:] @ (t @ w[:-2]) + w[-2] * alpha[0] + w[-1])


# the {nv} features: index, name, definition
{feats}
'''


def python_export(tag, m):
    """an exact Python formulation of a per-head model (None for the joint models for now)"""
    if m['kind'] != 'heads': return None
    nv = m['nv']; feats = '\n'.join(f'#  {k:2d}  {n:24s} {DEFS.get(n, "")}' for k, n in enumerate(m['names'][:nv]))
    note = ' Block 2 is a plain average: alpha[1, h] = 1/(n + 1) in every slot.' if m['uniform'] else ''
    return HEADS_PY.format(tag=tag, nv=nv, nv6=6 * nv, nterms=m['W'].shape[1], uniform_note=note, feats=feats)


def neuron_snippet(tag, block, head, o, terms, share=.99):
    """Python-like text of one value neuron: the terms covering `share` of its total importance, largest first"""
    tot = sum(t['importance'] for t in terms); acc, lines = 0.0, []
    for t in terms:
        if acc >= share * tot and len(lines) >= 3: break
        acc += t['importance']; x = f'x["{t["feature"]}"]'
        thr = t['term'].split('− ')[1].rstrip(')') if t['term'] != 'linear' else None
        lines.append(f'      {t["coef"]:+.4g} * ' + (x if thr is None else f'max(0, {x} - {thr})'))
    return (f'# value neuron {o + 1} of block {block}, head {head}:  sum over the particles i of alpha_i * f(x_i)   (+ c * alpha_cls + bias)\n'
            f'def f(x):   # x: the inputs of one particle\n    return (\n' + '\n'.join(lines) + f'\n      # ... {max(0, len(terms) - len(lines))} smaller terms; exact: {tag}_formulas.py\n    )')
