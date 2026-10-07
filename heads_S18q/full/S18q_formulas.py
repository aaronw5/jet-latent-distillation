"""Model S18q: ParT's 128 class-token neurons written directly as per-particle formulas with ParT's attention weights.
neuron_n = b_n + sum_h [ sum_i alpha[h, i] * f_hn(x_i) + c_hn * alpha_cls[h] ];  class scores = FC_W @ neurons + FC_B (ParT's last layer).
x_i: the 38 per-particle inputs of particle i (jetdistill/part/clsfit.py particle_features + research/nbr.py nbr_features, first 38 columns);
alpha[h]: head h's attention weights of the particles (h = 0..7 block 1, 8..15 block 2), alpha_cls[h] its weight on the class token (ParT's own).
f_hn(x) = sum over terms w * t(x), t(x) in {x_f, max(0, x_f - KN[k, f]), max(0, KN[k, f] - x_f)}; the coefficients are in S18q_model.npz."""
import numpy as np
P = np.load('S18q_model.npz'); W = P['W'].reshape(16, 420, 128); KN = P['knv']          # W[h]: rows = terms [x (38), x > KN_k (5 x 38), x < KN_k (5 x 38), alpha_cls, 1]
FC_W = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
FC_B = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
NAMES = ["ln pT", "ln E", "ln pT/pT_jet", "ln E/E_jet", "ΔR", "Δη", "Δφ", "charge", "charged hadron", "neutral hadron", "photon", "electron", "muon", "tanh d0", "σ(d0)", "tanh dz", "σ(dz)", "ln(1 + pT rank)", "ln(1+n within 0.1)", "ln(1+n within 0.2)", "pT share within 0.1", "pT share within 0.2", "pT share within 0.4", "ΔR nearest", "ln pT nearest / pT", "ΔR hardest", "ΔR 2nd hardest", "ΔR nearest displaced", "ΔR nearest lepton", "ΔR nearest photon", "prong index", "prong pT share", "prong ln mass", "prong charge", "prong n displaced", "pT share in prong", "ΔR prong axis", "ln kT with nearest"]


def terms(x):                                                   # x: (n, 38) -> (n, 418)
    return np.concatenate([x] + [np.maximum(0, x - KN[k]) for k in range(5)] + [np.maximum(0, KN[k] - x) for k in range(5)], -1)


def neurons(X, alpha, alpha_cls):
    """X: (n, 38) per-particle inputs of one jet; alpha: (16, n); alpha_cls: (16,) -> the 128 neurons"""
    T = terms(X); out = W[:, -1].sum(0).copy()
    for h in range(16): out += alpha[h] @ (T @ W[h, :-2]) + alpha_cls[h] * W[h, -2]
    return out


def class_scores(X, alpha, alpha_cls): return FC_W @ neurons(X, alpha, alpha_cls) + FC_B
