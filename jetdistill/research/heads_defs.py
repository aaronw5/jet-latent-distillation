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

# plain words: every input describes ONE particle i of the jet (the particle being summed over by a head);
# "nearest" = nearest TO PARTICLE i among the OTHER particles of the same jet, by angle ΔR in (η, φ)
WORDS = {
    'ln pT': 'how hard particle i is (its transverse momentum, log)', 'ln E': 'its energy (log)',
    'ln pT/pT_jet': 'the fraction of the jet’s pT carried by particle i (log)', 'ln E/E_jet': 'the fraction of the jet’s energy carried by particle i (log)',
    'ΔR': 'how far particle i is from the jet axis (the jet’s direction)', 'Δη': 'its offset from the jet axis in pseudorapidity', 'Δφ': 'its offset from the jet axis in azimuth',
    'charge': 'the electric charge of particle i', 'charged hadron': 'is particle i a charged hadron (e.g. a pion)?', 'neutral hadron': 'is it a neutral hadron?', 'photon': 'is it a photon?',
    'electron': 'is it an electron?', 'muon': 'is it a muon?',
    'tanh d0': 'how far particle i’s track misses the collision point, sideways — large for tracks from a b or c hadron decay', 'σ(d0)': 'the uncertainty of that miss distance',
    'tanh dz': 'how far its track misses the collision point along the beam', 'σ(dz)': 'the uncertainty of that',
    'ln(1 + pT rank)': 'where particle i stands in the jet’s pT ordering: 0 for the hardest, larger for softer particles',
    'ln(1+n within 0.1)': 'how many other particles of the jet are within ΔR 0.1 of particle i (a tight cone around it)', 'ln(1+n within 0.2)': 'the same within ΔR 0.2',
    'pT share within 0.1': 'what fraction of the jet’s pT sits in the other particles within ΔR 0.1 of particle i — is i inside a dense, hard core?',
    'pT share within 0.2': 'the same within ΔR 0.2', 'pT share within 0.4': 'the same within ΔR 0.4',
    'ΔR nearest': 'the angle from particle i to the closest other particle of the jet — is i isolated or in a cluster?',
    'ln pT nearest / pT': 'how hard that closest other particle is compared with particle i',
    'ΔR hardest': 'the angle from particle i to the hardest particle of the jet', 'ΔR 2nd hardest': 'the angle from particle i to the second-hardest particle',
    'ΔR nearest displaced': 'the angle from particle i to the closest OTHER particle of the jet that is a displaced track (charged, its track misses the collision point by more than 3σ — typical of b/c-hadron decays). Small: i sits next to a secondary vertex. 1.5 if the jet has no displaced track.',
    'ΔR nearest lepton': 'the angle from particle i to the closest other electron or muon in the jet (1.5 if none)', 'ΔR nearest photon': 'the angle from particle i to the closest other photon in the jet (1.5 if none)',
    'prong index': 'which of the jet’s 3 prongs particle i belongs to (the prongs are seeded by the 3 hardest particles; each particle joins the nearest seed): 0, 1 or 2',
    'prong pT share': 'the fraction of the jet’s pT in particle i’s prong', 'prong ln mass': 'the mass of particle i’s prong (log)', 'prong charge': 'the total charge of particle i’s prong',
    'prong n displaced': 'how many displaced tracks are in particle i’s prong', 'pT share in prong': 'the fraction of its prong’s pT that particle i carries — the prong’s leader or a soft companion?',
    'ΔR prong axis': 'the angle from particle i to the seed (hardest particle) of its prong', 'ln kT with nearest': 'how hard and wide the splitting between particle i and its closest neighbour is (kT = softer pT × angle)'}


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


def neuron_snippet(tag, block, head, o, inputs, share=.99):
    """one value neuron, an input per line: its piece as slopes per segment (a hinge coefficient is a change of slope, so
    the piece is shown as the slope on each interval between its kinks) and its net typical contribution"""
    tot = sum(d['importance'] for d in inputs) or 1.0; acc, lines = 0.0, []
    for d in inputs:
        if acc >= share * tot and len(lines) >= 3: break
        acc += d['importance']; k, s = d['knots'], d['slopes']
        if not k: seg = f'slope {s[0]:+.3g}'
        else: seg = ', '.join([f'slope {s[0]:+.3g} below {k[0]:.3g}'] + [f'{s[i + 1]:+.3g} from {k[i]:.3g}' for i in range(len(k))])
        lines.append(f'  {d["feature"]:<24s} {seg:<70s} # typical contribution ±{d["importance"]:.3g}')
    return (f'# value neuron {o + 1} of block {block}, head {head}:  sum over the particles i of alpha_i * f(x_i)   (+ c * alpha_cls + bias)\n'
            f'# f(x) = sum of one piecewise-linear piece per input (the slope on each interval between kinks); typical contribution = how much that input moves the neuron across jets\n'
            f'f(x) =\n' + '\n'.join(lines) + f'\n  # ... {max(0, len(inputs) - len(lines))} smaller inputs; exact: {tag}_formulas.py')


NOTATION = [
    ('jet, particles', 'a jet has n particles i = 1…n (up to 128); x_i = the per-particle inputs of particle i (defined in the table at the bottom).'),
    ('class token', 'ParT’s learned 128-number vector c that “reads” the jet. In class block 1 it is a constant (the same for every jet); class block 2 reads with the output of block 1.'),
    ('class blocks b, heads h', 'ParT ends with 2 class-attention blocks (b = 1, 2), each with 8 heads (h = 1…8): 16 heads in total.'),
    ('score s_bhi', 'how strongly head h of block b looks at particle i: s_bhi = q_bh · k_bhi / 4, with q_bh = W_q,bh · c (the query of the class token) and k_bhi = W_k,bh · LN(e_i) (the key of particle i; e_i its embedding after ParT’s 8 particle blocks, LN a LayerNorm). The class token has its own score s_bh0 = q_bh · k_bh0.'),
    ('attention weight α_bhi', 'α_bhi = exp(s_bhi) / Σ_{j=0…n} exp(s_bhj), i = 0 (the class token itself) … n: a softmax over [class token, particles] of this jet; the n + 1 weights of a head sum to 1. Per jet and per head.'),
    ('terms t(x)', 'for every input x_f: x_f itself and max(0, x_f − θ_f,k) at 5 thresholds θ_f,1…5 (the 15–85 % quantiles of x_f over particles): 6 terms per input.'),
    ('value neuron o_bh,m', 'm = 1…16: o_bh,m = Σ_{i=1…n} α_bhi · f_bh,m(x_i) + c_bh,m · α_bh0 + b_bh,m, with f_bh,m(x) = Σ_terms w · t(x): a sum over the particles, weighted by the head’s attention, of a formula of each particle’s inputs; plus the class token’s share α_bh0 times a constant.'),
    ('ParT_downstream', 'the 16 × 16 = 256 head outputs o go through ParT’s own fixed operations: out-projection, per-head scale, LayerNorm, residual and a 128→512→128 MLP in each class block, the final LayerNorm and the last layer → 10 class scores. Not fitted.'),
]
