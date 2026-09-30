"""Vectorised numpy implementation of every observable of library.py (same algorithms and tie-breaking as the
plain-Python code of the exported files; tests/test_observables.py compares them).

compute(X, n) -> {id: array (J,)} for particles X (J, N, 3) = (pT [GeV], Δη, Δφ), hardest first, padding pT = 0 at the end.
ParT networks ('kin', 'full'; N = 128) also take jet = (J, 4) jet pT, η, φ, energy and ext = (J, N, 6) charge, particle
type, d0, σ(d0), dz, σ(dz) (see part/data.py) for the extra observables of library.extras."""
import itertools
import numpy as np
from .library import KH, KS, TOPK, RINGS, ring_tag, library, PTYPES, H3, H4, HSD, LNEPS, BETAS, PPAIR, BLOCK_RX, is_block
from ..config import n_particles

ZCUT, R0, EPS = 0.1, 0.8, 1e-30


def compute(X, n, ids=None, chunk=1000, jet=None, ext=None):
    """all observables (or only `ids`) of the jets X, in chunks of `chunk` jets"""
    net, n = n, n_particles(n); X = np.asarray(X, np.float64).reshape(len(X), n, 3); out = {}
    blocks = [k for k in ids if is_block(k)] if ids is not None else []
    if blocks:                                   # ParT's per-particle and pair inputs: computed directly
        B = block_values(blocks, X, jet, ext); rest = [k for k in ids if not is_block(k)]
        if rest: B.update(compute(X, net, rest, chunk, jet, ext))
        return {k: B[k] for k in ids}
    for i0 in range(0, len(X), chunk):
        O = _chunk(X[i0:i0 + chunk], n)
        if not isinstance(net, int): O.update(_extras(X[i0:i0 + chunk], net, np.asarray(jet[i0:i0 + chunk], np.float64), None if ext is None else np.asarray(ext[i0:i0 + chunk], np.float64)))
        for k, v in O.items():
            if ids is None or k in ids: out.setdefault(k, []).append(v)
    O = {k: np.nan_to_num(np.concatenate(v)) for k, v in out.items()}
    lib = [k for k in library(net) if not is_block(k)]; missing = set(ids or lib) - set(O)
    assert not missing, f'no numpy implementation for {sorted(missing)}'
    return {k: O[k] for k in (ids or lib)}


def _m4(px, py, pz, E):
    return np.sqrt(np.maximum(E * E - px * px - py * py - pz * pz, 0.0))


def _p4(pt, eta, phi):
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1)


def _kmeans(pt, eta, phi, k, iters=6):
    """farthest-point initialisation (pT-weighted) then 6 pT-weighted k-means updates in (Δη, Δφ).
    Returns the axes (J, k, 2) and the distance of every particle to every axis (J, n, k)."""
    J = len(pt); ax = np.zeros((J, k, 2)); pts = np.stack([eta, phi], -1)
    ax[:, 0] = pts[np.arange(J), pt.argmax(1)]
    for j in range(1, k):
        d = np.min(np.linalg.norm(pts[:, :, None] - ax[:, None, :j], axis=-1), -1) * pt
        ax[:, j] = pts[np.arange(J), d.argmax(1)]
    for _ in range(iters):
        a = np.linalg.norm(pts[:, :, None] - ax[:, None], axis=-1).argmin(-1)
        for j in range(k):
            w = pt * (a == j); s = w.sum(1, keepdims=True)
            ax[:, j] = np.where(s > 0, (w[..., None] * pts).sum(1) / np.maximum(s, 1e-12), ax[:, j])
    return ax, np.linalg.norm(pts[:, :, None] - ax[:, None], axis=-1)


def _yphi(v):
    return 0.5 * np.log(np.maximum(v[..., 3] + v[..., 2], 1e-300) / np.maximum(v[..., 3] - v[..., 2], 1e-300)), np.arctan2(v[..., 1], v[..., 0])


def _dR2(u, v):
    (ya, pa), (yb, pb) = _yphi(u), _yphi(v); dp = np.mod(pa - pb + np.pi, 2 * np.pi) - np.pi
    return (ya - yb) ** 2 + dp ** 2


def _ca_tree(pt, eta, phi, batch=128):
    """Cambridge/Aachen clustering (E-scheme, rapidity-azimuth distance) of the real particles of each jet (the first
    slots; padding pT = 0 at the end). Returns V (J, 2H, 4) four-vectors of every node, kids (J, 2H, 2) its two children
    (-1 for a particle) and the root node of each jet. Jets are clustered in batches of similar multiplicity, each batch
    only over its real slots; pairwise distances are kept and only the merged node's row is recomputed after a merge
    (the same merges, in the same order, as recomputing every distance)."""
    J, H = pt.shape; V = np.zeros((J, 2 * H, 4)); kids = -np.ones((J, 2 * H, 2), int); root = np.zeros(J, int)
    nreal = (pt > 0).sum(1); order = np.argsort(nreal, kind='stable')
    for i0 in range(0, J, batch):
        idx = order[i0:i0 + batch]; h = max(int(nreal[idx].max()), 1)
        v, k, r = _ca_batch(pt[idx, :h], eta[idx, :h], phi[idx, :h], H)
        V[idx] = v; kids[idx] = k; root[idx] = r
    return V, kids, root


def _ca_batch(pt, eta, phi, H):
    J, h = pt.shape; rows = np.arange(J); real = pt > 0
    V = np.zeros((J, 2 * H, 4)); V[:, :h] = _p4(pt, eta, phi); kids = -np.ones((J, 2 * H, 2), int)
    slot_node = np.tile(np.arange(h), (J, 1)); active = real.copy(); cols = np.arange(h)
    Y, PH = _yphi(V[:, :h])
    def d2(ya, pa, yb, pb):                     # _dR2 on cached rapidities and azimuths
        dp = np.mod(pa - pb + np.pi, 2 * np.pi) - np.pi; return (ya - yb) ** 2 + dp ** 2
    D = np.where(active[:, :, None] & active[:, None] & np.triu(np.ones((h, h), bool), 1), d2(Y[:, :, None], PH[:, :, None], Y[:, None], PH[:, None]), np.inf)
    for s in range(h - 1):
        go = active.sum(1) >= 2
        if not go.any(): break
        f = D.reshape(J, -1).argmin(1); a, b = f // h, f % h
        new = H + s; na, nb = slot_node[rows, a], slot_node[rows, b]
        V[go, new] = V[rows, na][go] + V[rows, nb][go]; kids[go, new, 0] = na[go]; kids[go, new, 1] = nb[go]
        g = np.flatnonzero(go); ag, bg = a[g], b[g]
        slot_node[g, ag] = new; active[g, bg] = False; D[g, bg, :] = np.inf; D[g, :, bg] = np.inf
        yn, pn = _yphi(V[g, new]); Y[g, ag] = yn; PH[g, ag] = pn
        dn = d2(yn[:, None], pn[:, None], Y[g], PH[g]); act = active[g]
        D[g, ag, :] = np.where(act & (cols > ag[:, None]), dn, np.inf); D[g, :, ag] = np.where(act & (cols < ag[:, None]), dn, np.inf)
    return V, kids, slot_node[rows, active.argmax(1)]


def _softdrop(pt, eta, phi, tree=None):
    """soft drop (β = 0, z_cut = 0.1) on the C/A tree: follow the harder branch until min(pT1, pT2)/(pT1 + pT2) > z_cut.
    Returns the groomed mass, z_g, R_g and the number of removed branches."""
    J, H = pt.shape; rows = np.arange(J)
    V, kids, cur = tree if tree is not None else _ca_tree(pt, eta, phi); done = np.zeros(J, bool)
    mass, zg, rg, nrm = np.zeros(J), np.zeros(J), np.zeros(J), np.zeros(J)
    for _ in range(H):
        c1, c2 = kids[rows, cur, 0], kids[rows, cur, 1]; done |= c1 < 0
        if done.all(): break
        v1, v2 = V[rows, np.maximum(c1, 0)], V[rows, np.maximum(c2, 0)]
        p1, p2 = np.hypot(v1[:, 0], v1[:, 1]), np.hypot(v2[:, 0], v2[:, 1]); zz = np.minimum(p1, p2) / np.maximum(p1 + p2, 1e-300)
        stop = ~done & (zz > ZCUT); vc = V[rows, cur]
        mass = np.where(stop, _m4(*vc.T), mass); zg = np.where(stop, zz, zg); rg = np.where(stop, np.sqrt(_dR2(v1, v2)), rg)
        done |= stop; step = ~done
        nrm += step; cur = np.where(step, np.where(p1 >= p2, c1, c2), cur)
    return mass, zg, rg, nrm


def _ecf(zt, et, pht):
    """e2, e2(β=2), e3, e3(β=2), ₁e₃, ₂e₃ of the given particles, their distance matrix and pT shares"""
    h = zt.shape[1]; R = np.sqrt((et[:, :, None] - et[:, None]) ** 2 + (pht[:, :, None] - pht[:, None]) ** 2)
    iu = np.triu_indices(h, 1); w2 = zt[:, iu[0]] * zt[:, iu[1]]; rr = R[:, iu[0], iu[1]]
    I, J, K = np.array(list(itertools.combinations(range(h), 3))).T
    a, b, c = R[:, I, J], R[:, I, K], R[:, J, K]; w3 = zt[:, I] * zt[:, J] * zt[:, K]; s3 = np.sort(np.stack([a, b, c], -1), -1)
    return ((w2 * rr).sum(1), (w2 * rr ** 2).sum(1), (w3 * a * b * c).sum(1), (w3 * (a * b * c) ** 2).sum(1),
            (w3 * s3[..., 0]).sum(1), (w3 * s3[..., 0] * s3[..., 1]).sum(1), R, zt)


def _chunk(X, n):
    pt, eta, phi = X[..., 0], X[..., 1], X[..., 2]; J = len(X); rows = np.arange(J)
    real = pt > 0; nreal = real.sum(1); tot = pt.sum(1); z = pt / tot[:, None]; dr = np.hypot(eta, phi)
    P4 = _p4(pt, eta, phi)
    mass_of = lambda k: _m4(*P4[:, :k].sum(1).T)
    dist = lambda i, j: np.hypot(eta[:, i] - eta[:, j], phi[:, i] - phi[:, j])
    pm = lambda i, j: np.sqrt(np.maximum(2 * pt[:, i] * pt[:, j] * (np.cosh(eta[:, i] - eta[:, j]) - np.cos(phi[:, i] - phi[:, j])), 0))
    O = {}
    # ---- whole jet ----
    m = mass_of(n); zs = -np.sort(-z, 1)
    O['sum_pt'] = tot; O['log_sum_pt'] = np.log(tot); O['mass'] = m; O['mass_over_sum_pt'] = m / tot; O['mass_over_sum_pt_sq'] = (m / tot) ** 2
    O['max_dr'] = np.where(real, dr, -np.inf).max(1); O['pt_dispersion'] = np.sqrt((z * z).sum(1))
    O['z_1st'], O['z_2nd'], O['z_3rd'] = zs[:, 0], zs[:, 1], zs[:, 2]; O['z_top5'] = zs[:, :5].sum(1)
    O['girth'] = (z * dr).sum(1); O['girth2'] = (z * dr ** 2).sum(1)
    O['mean_eta'] = (z * eta).sum(1); O['mean_eta2'] = (z * eta ** 2).sum(1); O['mean_phi'] = (z * phi).sum(1); O['mean_phi2'] = (z * phi ** 2).sum(1)
    O['n_particles'] = nreal.astype(float)
    for c in (1, 5, 10, 50): O[f'n_pt_above_{c}'] = (pt > c).sum(1).astype(float)
    tau_ax = {k: _kmeans(pt, eta, phi, k) for k in (1, 2, 3, 4)}
    tau = {k: (z * D.min(-1)).sum(1) / R0 for k, (_, D) in tau_ax.items()}
    O['tau21'] = tau[2] / np.maximum(tau[1], 1e-12); O['tau32'] = tau[3] / np.maximum(tau[2], 1e-12)
    # e2 / e3 (library 'e2', 'C2', 'D2'): the 24 hardest by pT (stable sort); the other ECFs: the first 24 / 12 slots
    h3 = min(n, H3); top = np.argsort(-pt, 1, kind='stable')[:, :h3]
    e2, _, e3, _, _, _, R, zt = _ecf(*(np.take_along_axis(a, top, 1) for a in (z, eta, phi)))
    O['e2'] = e2; O['C2'] = e3 / np.maximum(e2 ** 2, 1e-12); O['D2'] = e3 / np.maximum(e2 ** 3, 1e-12)
    e2, e2b2, e3, e3b2, g31, g32, R, zt = _ecf(z[:, :h3], eta[:, :h3], phi[:, :h3])
    iua = np.triu_indices(n, 1)
    O['e2_sq'] = (z[:, iua[0]] * z[:, iua[1]] * ((eta[:, iua[0]] - eta[:, iua[1]]) ** 2 + (phi[:, iua[0]] - phi[:, iua[1]]) ** 2)).sum(1)
    O['LHA'] = (z * np.sqrt(dr / 0.8)).sum(1); O['centroid_offset'] = np.hypot(O['mean_eta'], O['mean_phi'])
    ta, tb, tc = O['mean_eta2'], (z * eta * phi).sum(1), O['mean_phi2']
    disc = np.sqrt(np.maximum((ta - tc) ** 2 / 4 + tb ** 2, 0)); lam1, lam2 = (ta + tc) / 2 + disc, np.maximum((ta + tc) / 2 - disc, 0)
    O['planar_flow'] = 4 * (ta * tc - tb ** 2) / np.maximum((ta + tc) ** 2, 1e-12); O['eccentricity'] = 1 - lam2 / np.maximum(lam1, 1e-12)
    O['width'] = ta + tc; O['lam1'] = lam1; O['lam2'] = lam2; O['orientation_deg'] = np.degrees(0.5 * np.arctan2(2 * tb, ta - tc))
    # ---- the two / three hardest ----
    O['dr01'] = dist(0, 1); O['pt1_over_pt0'] = pt[:, 1] / np.maximum(pt[:, 0], 1e-9)
    O['pt_balance01'] = np.minimum(pt[:, 0], pt[:, 1]) / np.maximum(pt[:, 0] + pt[:, 1], 1e-9); O['pt1_dr01'] = pt[:, 1] * dist(0, 1)
    m01 = pm(0, 1)
    if n >= 3:                                   # (the networks here see 8 or 64 particles)
        m02, m12 = pm(0, 2), pm(1, 2); m012 = np.sqrt(m01 ** 2 + m02 ** 2 + m12 ** 2)
        O['m01'] = m01; O['m012'] = m012; O['min_pair_mass'] = np.minimum(np.minimum(m01, m02), m12); O['max_pair_mass'] = np.maximum(np.maximum(m01, m02), m12)
        r2 = real[:, 2]; a01 = dist(0, 1); a02 = np.where(r2, dist(0, 2), 0.0); a12 = np.where(r2, dist(1, 2), 0.0)
        O['dr02'] = a02; O['dr12'] = a12
        O['dr_min_012'] = np.where(r2, np.minimum(np.minimum(a01, a02), a12), a01); O['dr_max_012'] = np.where(r2, np.maximum(np.maximum(a01, a02), a12), a01)
        O['pt2_over_pt0'] = pt[:, 2] / np.maximum(pt[:, 0], 1e-9)
        O['mratio_min_012'] = O['min_pair_mass'] / np.maximum(m012, 1e-9); O['mratio_max_012'] = O['max_pair_mass'] / np.maximum(m012, 1e-9)
    # ---- rings, integrated jet shape, counting ----
    for lo, hi in RINGS:
        ring = real & (dr >= lo) & (dr < (hi or 10)); t = ring_tag(lo, hi)
        O[f'n_dr_{t}'] = ring.sum(1).astype(float); O[f'z_dr_{t}'] = (z * ring).sum(1)
    for r in (0.1, 0.2, 0.3): O[f'psi_{r:g}'.replace('.', 'p')] = (z * (real & (dr < r))).sum(1)
    cz = np.cumsum(z, 1)
    for f in (0.5, 0.9): O[f'n_for_{int(100 * f)}pct'] = np.minimum((cz < f).sum(1) + 1, nreal).astype(float)
    O['pt_entropy'] = np.where(real, -z * np.log(np.where(real, z, 1.0)), 0.0).sum(1)
    # ---- each of the hardest particles ----
    for i in range(min(n, KH)):
        r = real[:, i]
        O[f'pt_{i}'], O[f'eta_{i}'], O[f'phi_{i}'] = pt[:, i], eta[:, i], phi[:, i]
        O[f'dr_{i}'] = np.where(r, dr[:, i], 0.0); O[f'z_{i}'] = z[:, i]
        O[f'abseta_{i}'] = np.abs(eta[:, i]); O[f'absphi_{i}'] = np.abs(phi[:, i]); O[f'zdr_{i}'] = z[:, i] * dr[:, i]
        if i >= 2:
            O[f'dr0_{i}'] = np.where(r, dist(0, i), 0.0); O[f'dr1_{i}'] = np.where(r, dist(1, i), 0.0)
            O[f'ptdr0_{i}'] = np.where(r, pt[:, i] * dist(0, i), 0.0); O[f'pair_mass_0_{i}'] = pm(0, i)
    # ---- the softest real particles beyond the hardest KH ----
    if n > KH:
        for s in range(1, KS + 1):
            j = nreal - s; ok = j >= KH; jj = np.clip(j, 0, n - 1)
            g = lambda a: np.where(ok, a[rows, jj], 0.0)
            O[f'soft{s}_pt'] = g(pt); O[f'soft{s}_z'] = g(z); O[f'soft{s}_abseta'] = g(np.abs(eta)); O[f'soft{s}_absphi'] = g(np.abs(phi)); O[f'soft{s}_dr'] = g(dr)
            O[f'soft{s}_dr0'] = np.where(ok, np.hypot(eta[rows, jj] - eta[:, 0], phi[rows, jj] - phi[:, 0]), 0.0)
    # ---- the k hardest ----
    for k in TOPK:
        if k >= n: break
        spk = pt[:, :k].sum(1)
        O[f'mass_top{k}'] = mass_of(k); O[f'sum_pt_top{k}'] = spk; O[f'z_top{k}_slots'] = spk / tot
        O[f'girth2_top{k}'] = (pt[:, :k] * dr[:, :k] ** 2).sum(1) / np.maximum(spk, 1e-9); O[f'n_real_top{k}'] = real[:, :k].sum(1).astype(float)
    # ---- energy correlation functions ----
    h4 = min(n, H4); C4 = np.array(list(itertools.combinations(range(h4), 4))).T
    d6 = np.stack([R[:, C4[p], C4[q]] for p, q in itertools.combinations(range(4), 2)], -1); w4 = zt[:, C4[0]] * zt[:, C4[1]] * zt[:, C4[2]] * zt[:, C4[3]]
    s6 = np.sort(d6, -1); e4 = (w4 * np.prod(d6, -1)).sum(1); g41, g42 = (w4 * s6[..., 0]).sum(1), (w4 * s6[..., 0] * s6[..., 1]).sum(1)
    O['e3'] = e3; O['e4'] = e4
    O['C3'] = e4 * e2 / np.maximum(e3 ** 2, EPS); O['D3'] = e4 * e2 ** 3 / np.maximum(e3 ** 3, EPS)
    O['N2'] = g32 / np.maximum(e2 ** 2, EPS); O['N3'] = g42 / np.maximum(g31 ** 2, EPS)
    O['M2'] = g31 / np.maximum(e2, EPS); O['M3'] = g41 / np.maximum(g31, EPS)
    O['C2_b2'] = e3b2 / np.maximum(e2b2 ** 2, EPS); O['D2_b2'] = e3b2 / np.maximum(e2b2 ** 3, EPS)
    # ---- N-subjettiness and subjets ----
    tau2 = {k: (z * D.min(-1) ** 2).sum(1) / R0 ** 2 for k, (_, D) in tau_ax.items()}
    for k in (1, 2, 3, 4): O[f'tau{k}'] = tau[k]
    O['tau43'] = tau[4] / np.maximum(tau[3], 1e-12); O['tau21_b2'] = tau2[2] / np.maximum(tau2[1], 1e-12)
    for k in (2, 3):
        ax, D = tau_ax[k]; near = D.argmin(-1)
        S = np.stack([(z * real * (near == j)).sum(1) for j in range(k)], 1); V = np.stack([(P4 * (real & (near == j))[..., None]).sum(1) for j in range(k)], 1)
        o = np.argsort(-S, 1, kind='stable'); S = np.take_along_axis(S, o, 1); V = np.take_along_axis(V, o[..., None], 1); A = np.take_along_axis(ax, o[..., None], 1)
        msj = _m4(*np.moveaxis(V, -1, 0))
        drs = {(p, q): np.hypot(A[:, p, 0] - A[:, q, 0], A[:, p, 1] - A[:, q, 1]) for p, q in itertools.combinations(range(k), 2)}
        if k == 2:
            O['sj2_zsoft'] = S[:, 1]; O['sj2_dr'] = drs[0, 1]; O['sj2_mass1'] = msj[:, 0]; O['sj2_mass2'] = msj[:, 1]
        else:
            for p in range(3): O[f'sj3_z{p + 1}'] = S[:, p]; O[f'sj3_mass{p + 1}'] = msj[:, p]
            for (p, q), v in drs.items(): O[f'sj3_dr{p + 1}{q + 1}'] = v
            dd = np.stack(list(drs.values()), -1); O['sj3_dr_min'] = dd.min(-1); O['sj3_dr_max'] = dd.max(-1)
            mp = np.stack([_m4(*(V[:, p] + V[:, q]).T) for p, q in itertools.combinations(range(3), 2)], -1)
            O['sj3_pair_mass_min'] = mp.min(-1); O['sj3_pair_mass_max'] = mp.max(-1)
            O['sj3_pairmin_over_m'] = mp.min(-1) / np.maximum(m, 1e-9); O['sj3_pairmax_over_m'] = mp.max(-1) / np.maximum(m, 1e-9)
    # ---- soft drop ----
    hs = min(n, HSD); O['sd_mass'], O['sd_zg'], O['sd_rg'], O['sd_nremoved'] = _softdrop(pt[:, :hs], eta[:, :hs], phi[:, :hs])
    return O


def _extras(X, net, jet, ext):
    """library.extras (jet-level quantities; blocks A and B are computed by block_values when used)"""
    pt, eta, phi = X[..., 0], X[..., 1], X[..., 2]; J, n = pt.shape; real = pt > 0; z = pt / pt.sum(1, keepdims=True); rows = np.arange(J)
    E = ext[..., 0]
    O = dict(jet_pt=jet[:, 0], jet_abs_eta=np.abs(jet[:, 1]), jet_e=jet[:, 3], sum_e=np.where(real, E, 0).sum(1))
    # ---- pair summaries ----
    iu = np.triu_indices(n, 1); P = _pairs(pt, eta, phi, iu); ok = real[:, iu[0]] & real[:, iu[1]]
    w = np.where(ok, z[:, iu[0]] * z[:, iu[1]], 0.0); ws = np.maximum(w.sum(1), 1e-300)
    for f in PPAIR: O[f'pair_mean_{f}'] = (w * P[f]).sum(1) / ws
    O['pair_max_lnkt'] = np.where(ok, P['lnkt'], LNEPS).max(1); O['pair_max_lnm2'] = np.where(ok, P['lnm2'], LNEPS).max(1)
    kt = np.minimum(pt[:, iu[0]], pt[:, iu[1]]) * np.hypot(eta[:, iu[0]] - eta[:, iu[1]], phi[:, iu[0]] - phi[:, iu[1]])
    for c in (1, 3, 10, 30): O[f'n_pairs_kt_above_{c}'] = (ok & (kt > c)).sum(1).astype(float)
    del P, w, kt
    # ---- primary Lund plane ----
    hs = min(n, HSD); V, kids, cur = _ca_tree(pt[:, :hs], eta[:, :hs], phi[:, :hs]); root = cur.copy()
    L = {(k, f): np.full(J, LNEPS) for k in (1, 2, 3) for f in ('lndelta', 'lnkt', 'lnz')}
    maxkt, maxd, nsp, n1, n5 = np.full(J, LNEPS), np.full(J, LNEPS), np.zeros(J), np.zeros(J), np.zeros(J); done = np.zeros(J, bool)
    for step in range(hs):
        c1, c2 = kids[rows, cur, 0], kids[rows, cur, 1]; done |= c1 < 0
        if done.all(): break
        v1, v2 = V[rows, np.maximum(c1, 0)], V[rows, np.maximum(c2, 0)]
        p1, p2 = np.hypot(v1[:, 0], v1[:, 1]), np.hypot(v2[:, 0], v2[:, 1]); d = np.sqrt(_dR2(v1, v2)); pm = np.minimum(p1, p2)
        lnd, lnk, lnz = np.log(np.maximum(d, 1e-8)), np.log(np.maximum(pm * d, 1e-8)), np.log(np.maximum(pm / np.maximum(p1 + p2, 1e-8), 1e-8))
        live = ~done
        if step < 3:
            for f, v in (('lndelta', lnd), ('lnkt', lnk), ('lnz', lnz)): L[step + 1, f] = np.where(live, v, L[step + 1, f])
        up = live & (lnk > maxkt); maxkt = np.where(up, lnk, maxkt); maxd = np.where(up, lnd, maxd)
        nsp += live; n1 += live & (pm * d > 1); n5 += live & (pm * d > 5)
        cur = np.where(live, np.where(p1 >= p2, c1, c2), cur)
    for (k, f), v in L.items(): O[f'lund{k}_{f}'] = v
    O['lund_max_lnkt'], O['lund_max_lndelta'], O['n_lund'], O['n_lund_kt_above_1'], O['n_lund_kt_above_5'] = maxkt, maxd, nsp, n1, n5
    # ---- energy correlations ----
    h3, h4 = min(n, H3), min(n, H4)
    G = {b: _ecfb(z, eta, phi, h3, h4, b) for b in (0.5, 1, 2)}
    for g in ('g31', 'g32', 'g41', 'g42', 'g43'): O[f'ecf_{g}'] = G[1][g]
    for b, t in BETAS:
        e = G[b]; mx = lambda x: np.maximum(x, 1e-30)
        O[f'e2_{t}'], O[f'e3_{t}'], O[f'e4_{t}'] = e['e2'], e['e3'], e['e4']
        if b != 2: O[f'C2_{t}'] = e['e3'] / mx(e['e2'] ** 2); O[f'D2_{t}'] = e['e3'] / mx(e['e2'] ** 3)
        O[f'C3_{t}'] = e['e4'] * e['e2'] / mx(e['e3'] ** 2); O[f'D3_{t}'] = e['e4'] * e['e2'] ** 3 / mx(e['e3'] ** 3)
        O[f'N2_{t}'] = e['g32'] / mx(e['e2'] ** 2); O[f'N3_{t}'] = e['g42'] / mx(e['g31'] ** 2)
        O[f'M2_{t}'] = e['g31'] / mx(e['e2']); O[f'M3_{t}'] = e['g41'] / mx(e['g31'])
    # ---- N-subjettiness, 4 subjets ----
    ax = {k: _kmeans(pt, eta, phi, k) for k in (2, 3, 4, 5)}
    t1 = {k: (z * D.min(-1)).sum(1) / R0 for k, (_, D) in ax.items()}; t2 = {k: (z * D.min(-1) ** 2).sum(1) / R0 ** 2 for k, (_, D) in ax.items()}
    O['tau5'] = t1[5]; O['tau54'] = t1[5] / np.maximum(t1[4], 1e-12)
    O['tau32_b2'] = t2[3] / np.maximum(t2[2], 1e-12); O['tau43_b2'] = t2[4] / np.maximum(t2[3], 1e-12)
    A4, D4 = ax[4]; near = D4.argmin(-1); P4 = _p4(pt, eta, phi)
    S = np.stack([(z * real * (near == j)).sum(1) for j in range(4)], 1); Vs = np.stack([(P4 * (real & (near == j))[..., None]).sum(1) for j in range(4)], 1)
    o = np.argsort(-S, 1, kind='stable'); S = np.take_along_axis(S, o, 1); Vs = np.take_along_axis(Vs, o[..., None], 1); A = np.take_along_axis(A4, o[..., None], 1)
    prs = list(itertools.combinations(range(4), 2))
    O['sj4_zsoft'] = S[:, 3]; O['sj4_dr_min'] = np.stack([np.hypot(A[:, p, 0] - A[:, q, 0], A[:, p, 1] - A[:, q, 1]) for p, q in prs], -1).min(-1)
    mp = np.stack([_m4(*(Vs[:, p] + Vs[:, q]).T) for p, q in prs], -1); O['sj4_pair_mass_min'] = mp.min(-1); O['sj4_pair_mass_max'] = mp.max(-1)
    if net != 'full': return O
    # ---- particle types, charge ----
    q, typ, d0, d0e, dz, dze = (ext[..., i] for i in range(1, 7)); ch = real & (q != 0); qq = np.where(real, q, 0.0)
    conds = dict(charged_had=typ == 1, neutral_had=typ == 2, photon=typ == 3, electron=typ == 4, muon=typ == 5, charged=q != 0, neutral=q == 0)
    for k in PTYPES:
        c = real & conds[k]; O[f'n_{k}'] = c.sum(1).astype(float); O[f'z_{k}'] = (z * c).sum(1)
    for c in (1, 10): O[f'n_charged_pt_above_{c}'] = (ch & (pt > c)).sum(1).astype(float)
    lep = real & ((typ == 4) | (typ == 5)); O['n_lepton'] = lep.sum(1).astype(float)
    O['jet_charge'] = (qq * z).sum(1); O['jet_charge_k05'] = (qq * z ** 0.5).sum(1); O['jet_charge_k03'] = (qq * z ** 0.3).sum(1); O['sum_charge'] = qq.sum(1)
    ich = ch.argmax(1); hasch = ch.any(1); O['lead_charge'] = np.where(hasch, q[rows, ich], 0.0)
    il = lep.argmax(1); has = lep.any(1); dr = np.hypot(eta, phi)
    t0, tz = ch & (d0e > 0), ch & (dze > 0)                  # tracks with a measured uncertainty (a few charged particles have σ = 0)
    sd0, sdz = np.where(t0, d0 / np.where(t0, d0e, 1.0), 0.0), np.where(tz, dz / np.where(tz, dze, 1.0), 0.0); s3 = np.sqrt(sd0 ** 2 + sdz ** 2); t3 = t0 & tz
    near_l = real & (np.hypot(eta - eta[rows, il][:, None], phi - phi[rows, il][:, None]) < 0.2); near_l[rows, il] = False
    O['lep_z'] = np.where(has, z[rows, il], 0.0); O['lep_dr'] = np.where(has, dr[rows, il], 0.0); O['lep_ptrel'] = np.where(has, pt[rows, il] * dr[rows, il], 0.0)
    O['lep_sd0'] = np.where(has, sd0[rows, il], 0.0)
    O['lep_iso'] = np.where(has, (pt * near_l).sum(1) / np.maximum(pt[rows, il], 1e-9), 0.0)
    for w, v, tk in (('d0', sd0, t0), ('dz', sdz, tz), ('3d', s3, t3)):
        order = np.argsort(-np.where(tk, np.abs(v), -1.0), 1, kind='stable'); vs = np.take_along_axis(v, order, 1); cs = np.take_along_axis(tk, order, 1)
        for r in (1, 2, 3): O[f'sip_{w}_{r}'] = np.where(cs[:, r - 1], vs[:, r - 1], 0.0)
    for w, v, tk, cs in (('d0', sd0, t0, (2, 3, 5, 10)), ('dz', sdz, tz, (2, 5)), ('3d', s3, t3, (3, 10))):
        for c in cs: O[f'n_s{w}_above_{c}'] = (tk & (np.abs(v) > c)).sum(1).astype(float)
    O['max_abs_d0'] = np.where(ch, np.abs(d0), 0.0).max(1); O['max_abs_dz'] = np.where(ch, np.abs(dz), 0.0).max(1)
    O['lead_ch_sd0'] = np.where(hasch, sd0[rows, ich], 0.0); O['lead_ch_sdz'] = np.where(hasch, sdz[rows, ich], 0.0)
    for c in (3, 5):
        disp = t0 & (np.abs(sd0) > c); O[f'z_displaced{c}'] = (z * disp).sum(1); O[f'mass_displaced{c}'] = _m4(*(P4 * disp[..., None]).sum(1).T)
    first2 = lambda m: m & (np.cumsum(m, 1) <= 2)
    for k, m in (('charged', ch), ('neutral', real & (q == 0)), ('2photon', first2(real & (typ == 3))), ('2charged', first2(ch))):
        O[f'mass_{k}'] = _m4(*(P4 * m[..., None]).sum(1).T)
    O['mass_2photon'] = np.where((real & (typ == 3)).sum(1) >= 2, O['mass_2photon'], 0.0)
    O.update(_plus(pt, eta, phi, jet, ext, (V, kids, root), ax, mp, P4))
    return O


# ---------------------------------------------------------------- more quantities of 'full' (library.plus; python_code.PLUS)
def _genkt(pt, eta, phi, sel, p, R, nstop=0, snap=0, batch=256):
    """generalized kT (p = 1 kT, -1 anti-kT) of the selected particles, as python_code.genkt: returns the (J, n) label of
    each particle (its pseudojet; -1 if not selected) and the exclusive merging scales dm[:, m] (m + 1 -> m pseudojets)"""
    J, n = pt.shape; cnt = sel.sum(1); order = np.argsort(cnt, kind='stable')
    lab = -np.ones((J, n), int); dm = np.zeros((J, 4)); dk = np.zeros((J, 4), bool)
    for i0 in range(0, J, batch):
        idx = order[i0:i0 + batch]; h = int(cnt[idx].max())
        if h == 0: continue
        pos = np.argsort(~sel[idx], 1, kind='stable')[:, :h]; live = np.take_along_axis(sel[idx], pos, 1)
        tk = lambda a: np.take_along_axis(a[idx], pos, 1)
        l, d, k = _genkt_batch(tk(pt), tk(eta), tk(phi), live, p, R, nstop, snap)
        sub = lab[idx]; np.put_along_axis(sub, pos, np.where(live, l, -1), 1); lab[idx] = sub; dm[idx] = d; dk[idx] = k
    return lab, dm, dk


def _genkt_batch(pt, eta, phi, live, p, R, nstop, snap):
    J, h = pt.shape; rows = np.arange(J); R2 = R * R; cols = np.arange(h)
    V = _p4(pt, eta, phi); Y, PH = _yphi(V); PT = np.hypot(V[..., 0], V[..., 1])
    kp = lambda t: t * t if p == 1 else 1.0 / np.maximum(t * t, 1e-300)
    KP = kp(PT)
    def d2(ya, pa, yb, pb):
        dp = np.mod(pa - pb + np.pi, 2 * np.pi) - np.pi; return (ya - yb) ** 2 + dp ** 2
    D = np.where(live[:, :, None] & live[:, None] & np.triu(np.ones((h, h), bool), 1),
                 np.minimum(KP[:, :, None], KP[:, None]) * d2(Y[:, :, None], PH[:, :, None], Y[:, None], PH[:, None]) / R2, np.inf)
    mem = np.tile(cols, (J, 1)); act = live.copy(); part = mem.copy(); jid = -np.ones((J, h), int); nj = np.zeros(J, int)
    dm = np.zeros((J, 4)); dk = np.zeros((J, 4), bool)
    for _ in range(h + 1):
        na = act.sum(1)
        if snap: part = np.where((na == snap)[:, None], mem, part)
        go = na > nstop
        if not go.any(): break
        flat = D.reshape(J, -1); f = flat.argmin(1); dpair = flat[rows, f]; a, b = f // h, f % h
        if nstop:
            mg = go
            for m in (1, 2, 3): w = mg & (na == m + 1); dm[w, m] = dpair[w]; dk[w, m] = True
        else:
            DB = np.where(act, KP, np.inf); ab = DB.argmin(1); mg = go & (dpair <= DB[rows, ab]); fin = go & ~mg
        g = np.flatnonzero(mg)
        if len(g):
            ag, bg = a[g], b[g]; V[g, ag] = V[g, ag] + V[g, bg]; act[g, bg] = False; D[g, bg, :] = np.inf; D[g, :, bg] = np.inf
            mem[g] = np.where(mem[g] == bg[:, None], ag[:, None], mem[g])
            vn = V[g, ag]; yn, pn = _yphi(vn); kn = kp(np.hypot(vn[:, 0], vn[:, 1])); Y[g, ag] = yn; PH[g, ag] = pn; KP[g, ag] = kn
            dn = np.minimum(kn[:, None], KP[g]) * d2(yn[:, None], pn[:, None], Y[g], PH[g]) / R2; ac = act[g]
            D[g, ag, :] = np.where(ac & (cols > ag[:, None]), dn, np.inf); D[g, :, ag] = np.where(ac & (cols < ag[:, None]), dn, np.inf)
        if not nstop:
            g = np.flatnonzero(fin)
            if len(g):
                abg = ab[g]; act[g, abg] = False; D[g, abg, :] = np.inf; D[g, :, abg] = np.inf; jid[g, abg] = nj[g]; nj[g] += 1
    if not nstop: return np.take_along_axis(jid, mem, 1), dm, dk
    return (part if snap else mem), dm, dk


def _rank(lab, pt, keep, min_pt=None):
    """the groups of a (J, n) labelling ranked by Σ pT: `keep` masks (J, n) (empty if missing) and the number of groups"""
    J, n = lab.shape; ok = lab >= 0; r, c = np.nonzero(ok)
    W = max(n, int(lab.max()) + 1); spt = np.zeros((J, W)); cnt = np.zeros((J, W), int); np.add.at(spt, (r, lab[ok]), pt[r, c]); np.add.at(cnt, (r, lab[ok]), 1)
    ex = cnt > 0
    if min_pt is not None: ex &= spt > min_pt
    order = np.argsort(-np.where(ex, spt, -np.inf), 1, kind='stable')
    masks = [(lab == order[:, k:k + 1]) & np.take_along_axis(ex, order[:, k:k + 1], 1) for k in range(keep)]
    return masks, ex.sum(1)


def _plus(pt, eta, phi, jet, ext, tree, ax, mp4, P4):
    from scipy.special import erfc
    J, n = pt.shape; rows = np.arange(J); real = pt > 0; tot = pt.sum(1); z = pt / tot[:, None]; O = {}
    E, q, typ, d0, d0e, dz, dze = (ext[..., i] for i in range(7))
    ch = real & (q != 0); t0 = ch & (d0e > 0); tz = ch & (dze > 0); t3 = t0 & tz; lep = real & ((typ == 4) | (typ == 5))
    c0 = np.where(t0, d0 / np.where(t0, np.minimum(d0e, 1.0), 1.0), 0.0)
    c3 = np.where(t3, np.sqrt(c0 ** 2 + (dz / np.where(tz, np.minimum(dze, 1.0), 1.0)) ** 2), 0.0)
    lnp = np.where(t0 & (c0 > 0), -np.log(np.maximum(erfc(np.abs(c0) / np.sqrt(2.0)), 1e-300)), 0.0)
    PE = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta + jet[:, 1:2]), E], -1)
    def emass(m, w=None):
        v = ((m * (1.0 if w is None else w))[..., None] * PE).sum(1); return np.sqrt(np.maximum(v[:, 3] * v[:, 3] - v[:, 0] * v[:, 0] - v[:, 1] * v[:, 1] - v[:, 2] * v[:, 2], 0.0))
    def kth(v, m, k):
        s = -np.sort(-np.where(m, v, -np.inf), 1); r = s[:, k - 1]; return np.where(np.isfinite(r), r, 0.0)
    def cen(m):
        w = np.maximum((pt * m).sum(1), 1e-9); return (pt * eta * m).sum(1) / w, (pt * phi * m).sum(1) / w
    def flav(m, pre):
        tr = m & t0; dsp = tr & (c0 > 3)
        for k in (1, 2, 3): O[f'{pre}sd0_{k}'] = kth(c0, tr, k)
        O[pre + 'n_disp3'] = dsp.sum(1).astype(float); O[pre + 'mass_disp3'] = emass(dsp); O[pre + 'z_disp3'] = (z * dsp).sum(1)
        O[pre + 'jp'] = (lnp * tr).sum(1); O[pre + 'n_lep'] = (m & lep).sum(1).astype(float)
        O[pre + 'charge'] = (q * np.sqrt(pt) * m).sum(1) / np.sqrt(np.maximum((pt * m).sum(1), 1e-9))
    def subfeats(pre, masks, mins=True, dr=True):
        cs = []
        for r, m in enumerate(masks):
            Q = f'{pre}_{r + 1}_'; O[Q + 'z'] = (z * m).sum(1); O[Q + 'mass'] = emass(m); flav(m, Q); cs.append((cen(m), m.any(1)))
        if dr:
            for a in range(len(masks)):
                for b in range(a + 1, len(masks)):
                    (ea, pa), oa = cs[a]; (eb, pb), ob = cs[b]; O[f'{pre}_dr{a + 1}{b + 1}'] = np.where(oa & ob, np.hypot(ea - eb, pa - pb), 0.0)
        if mins:
            for v in ('sd0_1', 'sd0_2', 'n_disp3', 'mass_disp3', 'jp'): O[f'{pre}_min12_{v}'] = np.minimum(O[f'{pre}_1_{v}'], O[f'{pre}_2_{v}'])
    # ---- subjets: exclusive kT (2) and its merging scales; anti-kT R = 0.2
    lab, dm, dk = _genkt(pt, eta, phi, real, 1, 1.0, nstop=1, snap=2); subfeats('kt2', _rank(lab, pt, 2)[0])
    for m in (1, 2, 3): O[f'ktd_ln_d{m}{m + 1}'] = np.where(dk[:, m], np.log(np.maximum(dm[:, m], 1e-12) / (tot * tot)), 0.0)
    lab = _genkt(pt, eta, phi, real, -1, 0.2)[0]; masks, ng = _rank(lab, pt, 3, min_pt=10.0); O['ak02_n'] = ng.astype(float); subfeats('ak02', masks)
    # ---- secondary-vertex proxies
    lab = _genkt(pt, eta, phi, t0 & (c0 > 3), -1, 0.1)[0]; masks, ng = _rank(lab, pt, 2); O['sv_n'] = ng.astype(float)
    for r, m in enumerate(masks):
        Q = f'sv_{r + 1}_'; (ce, cp) = cen(m)
        O[Q + 'n'] = m.sum(1).astype(float); O[Q + 'mass'] = emass(m); O[Q + 'z'] = (z * m).sum(1); O[Q + 'dr'] = np.where(m.any(1), np.hypot(ce, cp), 0.0); O[Q + 'sd0_sum'] = (c0 * m).sum(1)
    # ---- sums over tracks and pairs
    ed = (-np.inf, -3.0, -1.0, 1.0, 3.0, 10.0, np.inf)
    for b in range(6):
        m = t0 & (c0 > ed[b]) & (c0 <= ed[b + 1]); O[f'sdb_{b}_z'] = (z * m).sum(1); O[f'sdb_{b}_n'] = m.sum(1).astype(float)
    O['sdb_jp_all'] = lnp.sum(1); O['sdb_jp_top3'] = -np.sort(-lnp, 1)[:, :3].sum(1)
    h = min(n, 40); iu = np.triu_indices(h, 1); a, b = iu; ok = real[:, a] & real[:, b]
    d = np.hypot(eta[:, a] - eta[:, b], phi[:, a] - phi[:, b]); w = np.where(ok, z[:, a] * z[:, b], 0.0)
    ld, lk = np.log(np.maximum(d, 1e-6)), np.log(np.maximum(np.minimum(pt[:, a], pt[:, b]) * d, 1e-6))
    for k, (lo, hi) in enumerate(zip((-9.0, -3.0, -2.0, -1.0), (-3.0, -2.0, -1.0, 9.0))): O[f'pz_lnd{k}'] = (w * ((ld > lo) & (ld <= hi))).sum(1)
    for k, (lo, hi) in enumerate(zip((-9.0, 0.0, 1.0, 2.0, 3.0), (0.0, 1.0, 2.0, 3.0, 9.0))): O[f'pz_lnkt{k}'] = (w * ((lk > lo) & (lk <= hi))).sum(1)
    # ---- prongs (the C/A tree reversed)
    V, kids, root = tree; H2 = V.shape[1]; H = H2 // 2; npt = np.hypot(V[..., 0], V[..., 1]); ptj = npt[rows, root]; K = 16
    opn = -np.ones((J, K), int); opn[:, 0] = root; isp = np.zeros((J, H2), bool); SP = np.zeros((J, K, 5)); nsp = np.zeros(J, int)
    while (opn >= 0).any():
        for k in range(K):
            j = np.flatnonzero(opn[:, k] >= 0)
            if not len(j): continue
            nd = opn[j, k]; ca, cb = kids[j, nd, 0], kids[j, nd, 1]; leaf = ca < 0
            isp[j[leaf], nd[leaf]] = True; opn[j[leaf], k] = -1; j, nd, ca, cb = j[~leaf], nd[~leaf], ca[~leaf], cb[~leaf]
            if not len(j): continue
            pa, pb = npt[j, ca], npt[j, cb]; hard = np.where(pa >= pb, ca, cb); soft = np.where(pa >= pb, cb, ca)
            dr = np.sqrt(_dR2(V[j, ca], V[j, cb])); zz = np.minimum(pa, pb) / np.maximum(ptj[j], 1e-12); stop = dr <= 0.1
            isp[j[stop], nd[stop]] = True; opn[j[stop], k] = -1; go = ~stop; opn[j[go], k] = hard[go]; sp = go & (zz >= 0.1)
            for jj, ss, nn, z_, d_, pa_, pb_ in zip(j[sp], soft[sp], nd[sp], zz[sp], dr[sp], pa[sp], pb[sp]):
                opn[jj, np.flatnonzero(opn[jj] < 0)[0]] = ss
                if nsp[jj] < K: SP[jj, nsp[jj]] = (npt[jj, nn], z_, d_, min(pa_, pb_) * d_, _m4(*V[jj, nn])); nsp[jj] += 1
    ln = np.where(isp, np.arange(H2)[None], -1)
    for nd in range(H2 - 1, H - 1, -1):
        g = np.flatnonzero(ln[:, nd] >= 0)
        if len(g):
            for c in (0, 1):
                kc = kids[g, nd, c]; okc = kc >= 0; ln[g[okc], kc[okc]] = ln[g[okc], nd]
    lp = -np.ones((J, n), int); lp[:, :H] = ln[:, :H]
    masks, ng = _rank(np.where(real, lp, -1), pt, 4); O['dc_n'] = isp.sum(1).astype(float); subfeats('dc', masks, mins=False, dr=False)
    tags = np.stack([O[f'dc_{r + 1}_sd0_2'] for r in range(4)], 1); st = -np.sort(-tags, 1)
    O['dc_ntag'] = (tags > 3).sum(1).astype(float); O['dc_tag_max'], O['dc_tag_2nd'] = st[:, 0], st[:, 1]
    md = np.stack([O[f'dc_{r + 1}_mass_disp3'] for r in range(4)], 1); sm = -np.sort(-md, 1)
    O['dc_mass_disp_max'], O['dc_mass_disp_2nd'], O['dc_mass_disp_sum'] = sm[:, 0], sm[:, 1], md.sum(1)
    pm = np.stack([np.where(masks[a].any(1) & masks[b].any(1), emass(masks[a] | masks[b]), np.nan) for a in range(4) for b in range(a + 1, 4)], 1); has = np.isfinite(pm).any(1)
    O['dc_pair_mass_min'] = np.where(has, np.nanmin(np.where(has[:, None], pm, 0.0), 1), 0.0); O['dc_pair_mass_max'] = np.where(has, np.nanmax(np.where(has[:, None], pm, 0.0), 1), 0.0)
    o = np.argsort(-np.where(np.arange(K)[None] < nsp[:, None], SP[..., 0], -np.inf), 1, kind='stable'); SPo = np.take_along_axis(SP, o[..., None], 1)
    for r in (0, 1):
        okr = nsp > r
        for c, nm in enumerate(('z', 'dr', 'kt', 'mass')): O[f'dc_split{r + 1}_{nm}'] = np.where(okr, SPo[:, r, c + 1], 0.0)
    # ---- k-means subjets (the τ_N axes): groups ranked by pT share as subjets(k)
    def kgroups(k):
        near = ax[k][1].argmin(-1); S = np.stack([(z * real * (near == j)).sum(1) for j in range(k)], 1); order = np.argsort(-S, 1, kind='stable')
        return [real & (near == order[:, r:r + 1]) for r in range(k)]
    KG = {k: kgroups(k) for k in (2, 3, 4)}
    for k in (2, 3, 4):
        n2 = np.zeros(J)
        for r, m in enumerate(KG[k]):
            Q = f'sjf_{k}_{r + 1}_'; tr = m & t0; d3 = tr & (np.abs(c0) > 3)
            O[Q + 'n_d3'] = d3.sum(1).astype(float); O[Q + 'n_d5'] = (tr & (np.abs(c0) > 5)).sum(1).astype(float)
            O[Q + 'mass_d3'] = emass(d3); O[Q + 'z_d3'] = (z * d3).sum(1)
            O[Q + 'maxsd0'] = np.where(tr, np.abs(c0), 0.0).max(1); O[Q + 'max3d'] = np.where(m & t3, c3, 0.0).max(1); n2 += d3.sum(1) >= 2
        O[f'sjf_{k}_n2disp'] = n2
    # ---- jet-level vertex-like
    order = np.argsort(-np.where(t0, np.abs(c0), -1.0), 1, kind='stable'); vs = np.take_along_axis(c0, order, 1); cs_ = np.take_along_axis(t0, order, 1)
    for r in (4, 5, 6): O[f'jd_sd0_{r}'] = np.where(cs_[:, r - 1], vs[:, r - 1], 0.0); O[f'jd_3d_{r}'] = kth(c3, t3, r)
    ab = -np.sort(-np.where(t0, np.abs(c0), 0.0), 1); O['jd_sum_abs_sd0_top3'] = ab[:, :3].sum(1); O['jd_sum_abs_sd0_top5'] = ab[:, :5].sum(1)
    d3 = t0 & (np.abs(c0) > 3); mx = np.where(d3.any(1), np.where(d3, np.abs(c0), 0.0).max(1), 1.0)
    O['jd_mass_d3_sigw'] = emass(d3, np.abs(c0) / mx[:, None]); zd = (z * d3).sum(1); O['jd_mass_d3_over_z'] = np.where(zd > 0, emass(d3) / np.where(zd > 0, zd, 1.0), 0.0)
    O['jd_n_d3_pt1'] = (d3 & (pt > 1.0)).sum(1).astype(float)
    # ---- mass resolution
    O['mres_mass_e'] = emass(real)
    for k in (2, 3):
        for r, m in enumerate(KG[k]): O[f'mres_sj{k}_mass{r + 1}_e'] = emass(m)
    def sd(zc, beta, prongs=False):
        cur = root.copy(); done = np.zeros(J, bool); res = np.zeros((J, 3)); pm_ = np.zeros((J, 2))
        for _ in range(H):
            c1, c2 = kids[rows, cur, 0], kids[rows, cur, 1]; done |= c1 < 0
            if done.all(): break
            v1, v2 = V[rows, np.maximum(c1, 0)], V[rows, np.maximum(c2, 0)]; p1, p2 = np.hypot(v1[:, 0], v1[:, 1]), np.hypot(v2[:, 0], v2[:, 1])
            zz = np.minimum(p1, p2) / np.maximum(p1 + p2, 1e-300); d = np.sqrt(_dR2(v1, v2)); stop = ~done & (zz > zc * (d / 0.8) ** beta)
            res = np.where(stop[:, None], np.stack([_m4(*V[rows, cur].T), zz, d], 1), res)
            m1, m2 = _m4(*v1.T), _m4(*v2.T); pm_ = np.where(stop[:, None], np.stack([np.where(p1 >= p2, m1, m2), np.where(p1 >= p2, m2, m1)], 1), pm_)
            done |= stop; cur = np.where(~done, np.where(p1 >= p2, c1, c2), cur)
        return pm_ if prongs else res
    for t, beta, zc in (('b0z005', 0, 0.05), ('b0z02', 0, 0.2), ('b1z01', 1, 0.1), ('b2z01', 2, 0.1)):
        res = sd(zc, beta); O[f'mres_sd_mass_{t}'], O[f'mres_sd_zg_{t}'], O[f'mres_sd_rg_{t}'] = res[:, 0], res[:, 1], res[:, 2]
    pm_ = sd(0.1, 0, True); O['mres_sd_prong_mass1'], O['mres_sd_prong_mass2'] = pm_[:, 0], pm_[:, 1]
    rv = V[rows, root]; dcut = _m4(*rv.T) / np.maximum(np.hypot(rv[:, 0], rv[:, 1]), 1e-300); PV = V.copy()
    for nd in range(H, H2):
        c1, c2 = kids[:, nd, 0], kids[:, nd, 1]; ok = c1 >= 0
        if not ok.any(): break
        va, vb = PV[rows, np.maximum(c1, 0)], PV[rows, np.maximum(c2, 0)]; pa, pb = np.hypot(va[:, 0], va[:, 1]), np.hypot(vb[:, 0], vb[:, 1]); v = va + vb
        zz = np.minimum(pa, pb) / np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-300); pr = (zz < 0.1) & (np.sqrt(_dR2(va, vb)) > dcut)
        PV[ok, nd] = np.where(pr[:, None], np.where((pa >= pb)[:, None], va, vb), v)[ok]
    O['mres_pruned_mass'] = _m4(*PV[rows, root].T)
    lab = _genkt(pt, eta, phi, real, 1, 0.2)[0]; ok = lab >= 0; r_, c_ = np.nonzero(ok); spt = np.zeros((J, n)); np.add.at(spt, (r_, lab[ok]), pt[r_, c_])
    keep = ok & (np.take_along_axis(spt, np.maximum(lab, 0), 1) > 0.03 * tot[:, None]); O['mres_trimmed_mass'] = emass(keep)
    # ---- charge per subjet, prong counting
    for k in (2, 3):
        Qk = {}
        for r, m in enumerate(KG[k]):
            w = np.maximum((pt * m).sum(1), 1e-9)
            for ka, t in ((0.3, '03'), (0.5, '05'), (1.0, '1')): Qk[r, t] = (q * pt ** ka * m).sum(1) / w ** ka; O[f'sjq_{k}_{r + 1}_k{t}'] = Qk[r, t]
            O[f'sjq_{k}_{r + 1}_nch'] = (m & (q != 0)).sum(1).astype(float)
        for t in ('03', '05', '1'): O[f'sjq_{k}_sumabs_k{t}'] = np.abs(Qk[0, t] + Qk[1, t]); O[f'sjq_{k}_prod_k{t}'] = Qk[0, t] * Qk[1, t]
    c1, c2 = kids[..., 0], kids[..., 1]; okn = c1 >= 0; R_ = rows[:, None]
    va, vb = V[R_, np.maximum(c1, 0)], V[R_, np.maximum(c2, 0)]; ktn = np.where(okn, np.minimum(npt[R_, np.maximum(c1, 0)], npt[R_, np.maximum(c2, 0)]) * np.sqrt(_dR2(va, vb)), 0.0)
    for c in (2, 5, 10, 20):
        spl = okn & (ktn > c); reach = np.zeros((J, H2), bool); reach[rows, root] = True
        for nd in range(H2 - 1, H - 1, -1):
            g = np.flatnonzero(reach[:, nd] & spl[:, nd])
            if len(g): reach[g, c1[g, nd]] = True; reach[g, c2[g, nd]] = True
        O[f'nca_kt_above_{c}'] = (reach & ~spl).sum(1).astype(float)
    mps = -np.sort(-mp4, 1); mj = np.maximum(_m4(*(P4 * real[..., None]).sum(1).T), 1e-9)
    O['nca_sj4_pair_mass_2nd'], O['nca_sj4_pairmax_over_mass'], O['nca_sj4_pair2nd_over_mass'] = mps[:, 1], mps[:, 0] / mj, mps[:, 1] / mj
    # ---- the hardest lepton with its nearest subjet
    il = lep.argmax(1); has = lep.any(1); cols = np.arange(n)[None]
    for k in (2, 3):
        D = ax[k][1]; near = D.argmin(-1); jl = D[rows, il].argmin(-1); g = real & (near == jl[:, None]) & (cols != il[:, None]); tr = g & t0
        Q = f'lepsj_{k}_'
        O[Q + 'mass'] = np.where(has, emass(g | ((cols == il[:, None]) & has[:, None])), 0.0); O[Q + 'dr'] = np.where(has, D[rows, il, jl], 0.0)
        O[Q + 'n_d3'] = np.where(has, (tr & (np.abs(c0) > 3)).sum(1), 0).astype(float); O[Q + 'maxsd0'] = np.where(has, np.where(tr, np.abs(c0), 0.0).max(1), 0.0)
    return O


def _pairs(pt, eta, phi, iu):
    """ParT's pair inputs for the pairs iu (massless, from pT, Δη, Δφ); a pair with an empty slot: ln 1e-8"""
    a, b = iu; pa, pb = pt[:, a], pt[:, b]; de, dp = eta[:, a] - eta[:, b], phi[:, a] - phi[:, b]; ok = (pa > 0) & (pb > 0)
    d = np.hypot(de, dp); pm = np.minimum(pa, pb); m2 = 2 * pa * pb * (np.cosh(de) - np.cos(dp))
    lg = lambda x: np.where(ok, np.log(np.maximum(x, 1e-8)), LNEPS)
    return dict(lndelta=lg(d), lnkt=lg(pm * d), lnz=lg(pm / np.maximum(pa + pb, 1e-8)), lnm2=lg(m2))


def _ecfb(z, eta, phi, h3, h4, beta):
    """e2, e3, e4 and the generalized ₁e₃, ₂e₃, ₁e₄, ₂e₄, ₃e₄ (products of the smallest angles) with angular exponent β:
    pairs and triplets of the first h3 slots, quadruplets of the first h4"""
    zt, et, pht = z[:, :h3], eta[:, :h3], phi[:, :h3]
    R = np.sqrt((et[:, :, None] - et[:, None]) ** 2 + (pht[:, :, None] - pht[:, None]) ** 2) ** beta
    iu = np.triu_indices(h3, 1); o = dict(e2=(zt[:, iu[0]] * zt[:, iu[1]] * R[:, iu[0], iu[1]]).sum(1))
    I, Jx, K = _combos(h3, 3); a, b, c = R[:, I, Jx], R[:, I, K], R[:, Jx, K]; w3 = zt[:, I] * zt[:, Jx] * zt[:, K]; s3 = np.sort(np.stack([a, b, c], -1), -1)
    o['e3'] = (w3 * a * b * c).sum(1); o['g31'] = (w3 * s3[..., 0]).sum(1); o['g32'] = (w3 * s3[..., 0] * s3[..., 1]).sum(1); del a, b, c, s3, w3
    C4 = _combos(h4, 4); d6 = np.stack([R[:, C4[p], C4[q]] for p, q in itertools.combinations(range(4), 2)], -1)
    w4 = zt[:, C4[0]] * zt[:, C4[1]] * zt[:, C4[2]] * zt[:, C4[3]]; s6 = np.sort(d6, -1)
    o['e4'] = (w4 * np.prod(d6, -1)).sum(1); o['g41'] = (w4 * s6[..., 0]).sum(1); o['g42'] = (w4 * s6[..., 0] * s6[..., 1]).sum(1)
    o['g43'] = (w4 * s6[..., 0] * s6[..., 1] * s6[..., 2]).sum(1)
    return o


_COMBOS = {}


def _combos(h, k):
    if (h, k) not in _COMBOS: _COMBOS[h, k] = np.array(list(itertools.combinations(range(h), k))).reshape(-1, k).T
    return _COMBOS[h, k]


def block_values(ids, X, jet=None, ext=None):
    """blocks A (per-particle ParT inputs) and B (pair inputs) of the jets X (J, 128, 3) (may be memory-mapped: only the
    slots an id needs are read), for the given ids"""
    out = {}; jet = None if jet is None else np.asarray(jet, np.float64); col = {}
    def part(i):
        if i not in col: col[i] = (np.asarray(X[:, i], np.float64), None if ext is None else np.asarray(ext[:, i], np.float64))
        return col[i]
    pairs = {}
    for q in ids:
        m = BLOCK_RX.fullmatch(q); f = q.split('_')[0]
        if m.group(1) is not None:
            i = int(m.group(1)); x, e = part(i); pt = x[:, 0]; r = pt > 0
            lg = lambda v: np.where(r, np.log(np.maximum(np.where(r, v, 1.0), 1e-300)), LNEPS)
            v = dict(lnpt=lambda: lg(pt), lne=lambda: lg(e[:, 0]), lnptrel=lambda: lg(pt / jet[:, 0]), lnerel=lambda: lg(e[:, 0] / jet[:, 3]),
                     dr=lambda: np.hypot(x[:, 1], x[:, 2]), eta=lambda: x[:, 1], phi=lambda: x[:, 2],
                     charge=lambda: e[:, 1], ischhad=lambda: (e[:, 2] == 1) * 1.0, isnhad=lambda: (e[:, 2] == 2) * 1.0, isphoton=lambda: (e[:, 2] == 3) * 1.0,
                     iselectron=lambda: (e[:, 2] == 4) * 1.0, ismuon=lambda: (e[:, 2] == 5) * 1.0, td0=lambda: np.tanh(e[:, 3]), d0err=lambda: np.clip(e[:, 4], 0, 1),
                     tdz=lambda: np.tanh(e[:, 5]), dzerr=lambda: np.clip(e[:, 6], 0, 1))[f]()
            out[q] = np.where(r, v, LNEPS if f.startswith('ln') else 0.0)
        else:
            i, j = int(m.group(2)), int(m.group(3))
            if (i, j) not in pairs:
                (a, _), (b, _) = part(i), part(j); P = np.stack([a, b], 1)
                pairs[i, j] = _pairs(P[..., 0], P[..., 1], P[..., 2], (np.array([0]), np.array([1])))
            out[q] = pairs[i, j][f][:, 0]
    return out


def block_matrix(n, X, jet, ext, dtype=np.float32):
    """every block quantity of the jets X at once: (ids, (J, ids) array); for step 1's screening of blocks A and B"""
    X = np.asarray(X, np.float64); J = len(X); lib = library(n); ids = [k for k in lib if is_block(k)]; col = {k: c for c, k in enumerate(ids)}
    M = np.empty((J, len(ids)), dtype); pt = X[..., 0]
    if any(BLOCK_RX.fullmatch(k).group(1) is None for k in ids):
        iu = np.triu_indices(X.shape[1], 1); P = _pairs(pt, X[..., 1], X[..., 2], iu)
        for f in PPAIR:
            for c, (i, j) in enumerate(zip(*iu)): M[:, col[f'{f}_{i}_{j}']] = P[f][:, c]
        del P
    per = [k for k in ids if BLOCK_RX.fullmatch(k).group(1) is not None]
    for k, v in block_values(per, X, jet, ext).items(): M[:, col[k]] = v
    return ids, M
