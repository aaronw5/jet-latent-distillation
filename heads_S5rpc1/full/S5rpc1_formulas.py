"""Model S5rpc1: ParT's class attention written per head. Exact formulation; the coefficients are in S5rpc1_model.npz.

Inputs per jet: X[i, f], the 38 per-particle features of particle i (definitions below); alpha[b, h, :] the attention
weights of head h of class block b over [the class token, the particles] (they sum to 1).
In this model the weights are ParT's own (computed from its particle blocks); the values are the formulas below.
The 16 outputs of every head go through ParT's fixed arithmetic (jetdistill/research/heads.py: downstream) to the class scores.
"""
import numpy as np
P = np.load('S5rpc1_model.npz'); W = P['W']; KN = P['kn'][:, :38]   # W: (16 heads, 230 terms, 16 outputs); KN: (5 thresholds, 38 features)


def terms(X):
    """X: (n_particles, 38) -> (n_particles, 228): every feature, then max(0, feature - t) at its 5 thresholds"""
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


# the 38 features: index, name, definition
#   0  ln pT                    ln pT_i
#   1  ln E                     ln E_i
#   2  ln pT/pT_jet             ln(pT_i / pT_jet)
#   3  ln E/E_jet               ln(E_i / E_jet)
#   4  ΔR                       ΔR_i = √(Δη_i² + Δφ_i²), to the jet axis
#   5  Δη                       η_i − η_jet
#   6  Δφ                       φ_i − φ_jet
#   7  charge                   q_i ∈ {−1, 0, +1}
#   8  charged hadron           1 if particle i is a charged hadron, else 0
#   9  neutral hadron           1 if a neutral hadron
#  10  photon                   1 if a photon
#  11  electron                 1 if an electron
#  12  muon                     1 if a muon
#  13  tanh d0                  tanh(d0_i), d0 the transverse impact parameter
#  14  σ(d0)                    min(σ(d0_i), 1)
#  15  tanh dz                  tanh(dz_i), dz the longitudinal impact parameter
#  16  σ(dz)                    min(σ(dz_i), 1)
#  17  ln(1 + pT rank)          ln(1 + r_i), r_i = 0 for the hardest particle of the jet, 1 for the next, …
#  18  ln(1+n within 0.1)       ln(1 + #{j ≠ i : ΔR_ij < 0.1})
#  19  ln(1+n within 0.2)       ln(1 + #{j ≠ i : ΔR_ij < 0.2})
#  20  pT share within 0.1      Σ_{j ≠ i, ΔR_ij < 0.1} pT_j / Σ_j pT_j
#  21  pT share within 0.2      Σ_{j ≠ i, ΔR_ij < 0.2} pT_j / Σ_j pT_j
#  22  pT share within 0.4      Σ_{j ≠ i, ΔR_ij < 0.4} pT_j / Σ_j pT_j
#  23  ΔR nearest               min_{j ≠ i} ΔR_ij (capped at 1.5)
#  24  ln pT nearest / pT       ln(pT_nn(i) / pT_i), nn(i) the nearest other particle (clipped to ±10)
#  25  ΔR hardest               ΔR between i and the hardest particle of the jet (capped at 1.5)
#  26  ΔR 2nd hardest           ΔR between i and the second-hardest particle (capped at 1.5)
#  27  ΔR nearest displaced     min ΔR_ij over the other particles j that are charged and displaced: q_j ≠ 0 and |d0_j| / σ(d0_j) > 3; 1.5 if the jet has none
#  28  ΔR nearest lepton        min ΔR_ij over the other particles j that are electrons or muons; 1.5 if none
#  29  ΔR nearest photon        min ΔR_ij over the other photons j; 1.5 if none
#  30  prong index              k ∈ {0, 1, 2}: the 3 hardest particles seed 3 prongs; every particle belongs to the prong whose seed is nearest in ΔR; k is the index of particle i’s prong (0 = the hardest seed)
#  31  prong pT share           Σ_{j ∈ prong(i)} pT_j / Σ_j pT_j
#  32  prong ln mass            ln(1 + m(prong(i))), the invariant mass of the summed 4-vectors of the prong’s particles
#  33  prong charge             Σ_{j ∈ prong(i)} q_j
#  34  prong n displaced        #{j ∈ prong(i) : q_j ≠ 0, |d0_j| / σ(d0_j) > 3}
#  35  pT share in prong        pT_i / Σ_{j ∈ prong(i)} pT_j
#  36  ΔR prong axis            ΔR between i and the seed particle of its prong (capped at 1.5)
#  37  ln kT with nearest       ln(min(pT_i, pT_nn(i)) · ΔR_i,nn(i))
