"""The observables the formulas may use: one table, keyed by a short id.

Every observable is computed from the N particles the network sees, each (pT [GeV], Δη, Δφ) relative to the jet axis,
hardest first, empty slots (pT = 0) at the end. Each entry has
    label  the name shown on the pages,
    expr   a plain-Python expression, evaluated inside `quantities()` of an exported file (helpers: python_code.py),
    desc   a one-line description (units in brackets).
The numpy implementation (compute.py) follows the same algorithms and tie-breaking; tests/test_observables.py checks
that both give the same values. Every observable with units of mass has 'mass' in its id (the setups without mass
observables remove them by that rule, see config.py)."""
import functools, itertools, re
from typing import NamedTuple
from ..config import n_particles, TAGGER

KH, KS = 15, 10            # per-particle observables: the KH hardest; the KS softest real particles beyond them (N > KH)
TOPK = (2, 3, 5, 10, 15, 20, 30, 40, 50)
# ECF e2/e3 on the first H3 slots, e4 on the first H4, C/A clustering (soft drop, Lund plane) on the first HSD
H3, H4, HSD = (24, 12, 20) if TAGGER == 'jedi' else (32, 16, 128)
RINGS = ((0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, None))


class Observable(NamedTuple):
    label: str
    expr: str
    desc: str


def ring_tag(lo, hi):
    return (f'{lo:g}_{hi:g}' if hi else f'{lo:g}_up').replace('.', 'p')


@functools.lru_cache(maxsize=None)
def library(n):
    """{id: Observable} for a network n (JEDI: n particles; ParT: 'kin' or 'full', 128 particles plus jet and particle extras)"""
    net, n = n, n_particles(n)
    L = {}

    def add(id_, label, expr, desc):
        assert id_ not in L, id_
        L[id_] = Observable(label, expr, desc)

    # ---- whole jet ----
    add('sum_pt', 'Σᵢ pTᵢ', 'tot', 'total pT of the particles [GeV]')
    add('log_sum_pt', 'log Σᵢ pTᵢ', 'math.log(tot)', 'natural log of the total pT')
    add('mass', 'mass', 'mass_of(n)', 'invariant mass of all particles (massless four-vectors) [GeV]')
    add('mass_over_sum_pt', 'm / Σᵢ pTᵢ', 'mass_of(n) / tot', 'jet mass / total pT')
    add('mass_over_sum_pt_sq', '(m / Σᵢ pTᵢ)²', '(mass_of(n) / tot) ** 2', '(jet mass / total pT) squared')
    add('max_dr', 'maxᵢ ΔRᵢ', 'max(dr[i] for i in real)', 'largest distance ΔR of a particle from the jet axis')
    add('pt_dispersion', '√(Σᵢ pTᵢ²) / Σᵢ pTᵢ', 'math.sqrt(sum(x * x for x in z))', '√(Σ pTᵢ²) / Σ pTᵢ')
    add('z_1st', 'pT₀ / Σᵢ pTᵢ', 'zs[0]', 'largest pT share')
    add('z_2nd', 'pT₁ / Σᵢ pTᵢ', 'zs[1]', '2nd-largest pT share')
    add('z_3rd', 'pT₂ / Σᵢ pTᵢ', 'zs[2]', '3rd-largest pT share')
    add('z_top5', '(pT₀+pT₁+pT₂+pT₃+pT₄) / Σᵢ pTᵢ', 'sum(zs[:5])', 'pT share of the 5 largest')
    add('girth', 'Σᵢ pTᵢ·ΔRᵢ / Σᵢ pTᵢ', 'sum(z[i] * dr[i] for i in P)', 'pT-weighted mean ΔR')
    add('girth2', 'Σᵢ pTᵢ·ΔRᵢ² / Σᵢ pTᵢ', 'sum(z[i] * dr[i] ** 2 for i in P)', 'pT-weighted mean ΔR²')
    add('mean_eta', 'Σᵢ pTᵢ·Δηᵢ / Σᵢ pTᵢ', 'sum(z[i] * eta[i] for i in P)', 'pT-weighted mean Δη')
    add('mean_eta2', 'Σᵢ pTᵢ·Δηᵢ² / Σᵢ pTᵢ', 'sum(z[i] * eta[i] ** 2 for i in P)', 'pT-weighted mean Δη²')
    add('mean_phi', 'Σᵢ pTᵢ·Δφᵢ / Σᵢ pTᵢ', 'sum(z[i] * phi[i] for i in P)', 'pT-weighted mean Δφ')
    add('mean_phi2', 'Σᵢ pTᵢ·Δφᵢ² / Σᵢ pTᵢ', 'sum(z[i] * phi[i] ** 2 for i in P)', 'pT-weighted mean Δφ²')
    add('n_particles', 'number of particles', 'len(real)', 'number of real particles (pT > 0)')
    for c in (1, 5, 10, 50):
        add(f'n_pt_above_{c}', f'Σᵢ [pTᵢ > {c} GeV]', f'sum(1 for x in pt if x > {c})', f'number of particles with pT > {c} GeV')
    add('tau21', 'τ21', 'tau(2) / max(tau(1), 1e-12)', 'N-subjettiness τ2/τ1 (axes from pT-weighted k-means)')
    add('tau32', 'τ32', 'tau(3) / max(tau(2), 1e-12)', 'N-subjettiness τ3/τ2')
    add('e2', 'Σᵢ<ⱼ pTᵢpTⱼ·ΔRᵢⱼ / (Σᵢ pTᵢ)²', 'e2', 'energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ')
    add('C2', 'C2', 'e3 / max(e2 ** 2, 1e-12)', 'energy correlation ratio e3/e2²')
    add('D2', 'D2', 'e3 / max(e2 ** 3, 1e-12)', 'energy correlation ratio e3/e2³')
    add('e2_sq', 'Σᵢ<ⱼ zᵢzⱼΔRᵢⱼ²', 'sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j)', 'Σ_{i<j} zᵢzⱼΔRᵢⱼ²')
    add('LHA', 'LHA = Σᵢ zᵢ(ΔRᵢ/0.8)^½', 'sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P)', 'Les Houches angularity')
    add('centroid_offset', '√((Σᵢ zᵢΔηᵢ)² + (Σᵢ zᵢΔφᵢ)²)', 'math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P))', 'distance of the pT centroid from the jet axis')
    add('planar_flow', 'planar flow 4λ₁λ₂/(λ₁+λ₂)²', '4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12)', 'planar flow of the pT-weighted (Δη, Δφ) tensor')
    add('eccentricity', '1 − λ₂/λ₁', '1 - lam2 / max(lam1, 1e-12)', '1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor')
    add('width', 'λ₁ + λ₂', 'ta + tc', 'λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor')
    add('lam1', 'λ₁', 'lam1', 'larger eigenvalue of the pT-weighted (Δη, Δφ) tensor')
    add('lam2', 'λ₂', 'lam2', 'smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor')
    add('orientation_deg', '½·atan2(2c, a − b) [deg]', 'math.degrees(0.5 * math.atan2(2 * tb, ta - tc))', 'direction of the major axis [degrees]')
    # ---- the two / three hardest particles ----
    add('dr01', 'ΔR₀₁', 'math.sqrt(dist2(0, 1))', 'ΔR between particles 0 and 1')
    add('pt1_over_pt0', 'pT₁ / pT₀', 'pt[1] / max(pt[0], 1e-9)', 'pT1 / pT0')
    add('pt_balance01', 'min(pT₀, pT₁)/(pT₀ + pT₁)', 'min(pt[0], pt[1]) / max(pt[0] + pt[1], 1e-9)', 'min(pT0, pT1) / (pT0 + pT1)')
    add('pt1_dr01', 'pT₁·ΔR₀₁', 'pt[1] * math.sqrt(dist2(0, 1))', 'pT1 · ΔR01')
    add('m01', 'm₀₁', 'pair_mass(0, 1)', 'mass of particles 0 and 1 [GeV]')
    add('m012', 'm₀₁₂', 'math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2)', 'mass of particles 0, 1 and 2 [GeV]')
    add('min_pair_mass', 'min(m₀₁, m₀₂, m₁₂)', 'min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2))', 'smallest pair mass among particles 0, 1, 2 [GeV]')
    add('max_pair_mass', 'max(m₀₁, m₀₂, m₁₂)', 'max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2))', 'largest pair mass among particles 0, 1, 2 [GeV]')
    if n >= 3:
        add('dr02', 'ΔR₀₂', 'math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0', 'ΔR between particles 0 and 2')
        add('dr12', 'ΔR₁₂', 'math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0', 'ΔR between particles 1 and 2')
        d3 = 'math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))'
        add('dr_min_012', 'min(ΔR₀₁, ΔR₀₂, ΔR₁₂)', f'min({d3}) if pt[2] > 0 else math.sqrt(dist2(0, 1))', 'smallest distance among the 3 hardest')
        add('dr_max_012', 'max(ΔR₀₁, ΔR₀₂, ΔR₁₂)', f'max({d3}) if pt[2] > 0 else math.sqrt(dist2(0, 1))', 'largest distance among the 3 hardest')
        add('pt2_over_pt0', 'pT₂ / pT₀', 'pt[2] / max(pt[0], 1e-9)', 'pT2 / pT0')
        m012 = 'max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9)'
        add('mratio_min_012', 'min(m₀₁, m₀₂, m₁₂) / m₀₁₂', f'min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / {m012}', 'smallest pair mass / mass of the 3 hardest (dimensionless)')
        add('mratio_max_012', 'max(m₀₁, m₀₂, m₁₂) / m₀₁₂', f'max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / {m012}', 'largest pair mass / mass of the 3 hardest (dimensionless)')
    # ---- rings, integrated jet shape, counting ----
    for lo, hi in RINGS:
        rng, t = (f'{lo:g} ≤ ΔRᵢ < {hi:g}' if hi else f'ΔRᵢ ≥ {lo:g}'), ring_tag(lo, hi)
        cond = f'{lo:g} <= dr[i] < {hi:g}' if hi else f'{lo:g} <= dr[i] < 10'
        add(f'n_dr_{t}', f'Σᵢ [{rng}]', f'sum(1 for i in real if {cond})', f'number of particles with {rng.replace("ᵢ", "")}')
        add(f'z_dr_{t}', f'Σᵢ pTᵢ·[{rng}] / Σᵢ pTᵢ', f'sum(z[i] for i in real if {cond})', f'pT share of the particles with {rng.replace("ᵢ", "")}')
    for r in (0.1, 0.2, 0.3):
        add(f'psi_{r:g}'.replace('.', 'p'), f'Ψ({r:g}) = Σᵢ zᵢ [ΔRᵢ < {r:g}]', f'sum(z[i] for i in real if dr[i] < {r:g})', f'pT share within ΔR < {r:g} of the jet axis')
    for f in (0.5, 0.9):
        add(f'n_for_{int(100 * f)}pct', f'number of particles for {int(100 * f)}% of ΣpT', f'ncum({f:g})', f'number of hardest particles that carry {int(100 * f)}% of the jet pT')
    add('pt_entropy', 'pT entropy −Σᵢ zᵢ ln zᵢ', '-sum(z[i] * math.log(z[i]) for i in real)', 'pT entropy −Σ zᵢ ln zᵢ')
    # ---- each of the hardest particles ----
    for i in range(min(n, KH) if isinstance(net, int) else 0):          # ParT: block A (all 128 particles, see extras)
        add(f'pt_{i}', f'pT of particle {i}', f'pt[{i}]', f'pT of particle {i} [GeV]')
        add(f'eta_{i}', f'Δη of particle {i}', f'eta[{i}]', f'Δη of particle {i}')
        add(f'phi_{i}', f'Δφ of particle {i}', f'phi[{i}]', f'Δφ of particle {i}')
        add(f'dr_{i}', f'ΔR of particle {i}', f'dr[{i}] if pt[{i}] > 0 else 0.0', f'ΔR of particle {i} from the jet axis')
        add(f'z_{i}', f'pT share of particle {i}', f'z[{i}]', f'pT of particle {i} / total pT')
        add(f'abseta_{i}', f'|Δη| of particle {i}', f'abs(eta[{i}])', f'|Δη| of particle {i}')
        add(f'absphi_{i}', f'|Δφ| of particle {i}', f'abs(phi[{i}])', f'|Δφ| of particle {i}')
        add(f'zdr_{i}', f'pT share × ΔR of particle {i}', f'z[{i}] * dr[{i}]', f'pT share × ΔR of particle {i} (its term in ΣzΔR)')
        if i >= 2:
            add(f'dr0_{i}', f'ΔR between particles 0 and {i}', f'math.sqrt(dist2(0, {i})) if pt[{i}] > 0 else 0.0', f'ΔR between particle {i} and the hardest particle')
            add(f'dr1_{i}', f'ΔR between particles 1 and {i}', f'math.sqrt(dist2(1, {i})) if pt[{i}] > 0 else 0.0', f'ΔR between particle {i} and the 2nd-hardest particle')
            add(f'ptdr0_{i}', f'pT of particle {i} × ΔR to particle 0', f'pt[{i}] * math.sqrt(dist2(0, {i})) if pt[{i}] > 0 else 0.0', f'pT{i} · ΔR(0, {i}) [GeV]')
            add(f'pair_mass_0_{i}', f'mass of particles 0 and {i}', f'pair_mass(0, {i})', f'mass of particles 0 and {i} [GeV]')
    # ---- the softest real particles beyond the hardest KH (a particle among the KH hardest is named by its hardest index) ----
    if n > KH and isinstance(net, int):
        for s in range(1, KS + 1):
            for nm, w, d in (('pT', 'pt', 'pT [GeV]'), ('pT share', 'z', 'pT share'), ('|Δη|', 'abseta', '|Δη|'), ('|Δφ|', 'absphi', '|Δφ|'), ('ΔR', 'dr', 'ΔR from the jet axis')):
                add(f'soft{s}_{w}', f'{nm} of softest particle {s}', f'softp({s}, {w!r})', f'{d} of the {s}. softest real particle (0 if it is among the {KH} hardest)')
            add(f'soft{s}_dr0', f'ΔR between particle 0 and softest particle {s}', f"softp({s}, 'dr0')", f'ΔR between the hardest and the {s}. softest real particle (0 if among the {KH} hardest)')
    # ---- the k hardest particles ----
    for k in TOPK:
        if k >= n: break
        add(f'mass_top{k}', f'mass of the {k} hardest particles', f'mass_of({k})', f'mass of the {k} hardest particles [GeV]')
        add(f'sum_pt_top{k}', f'Σ pT of the {k} hardest particles', f'sum(pt[:{k}])', f'total pT of the {k} hardest particles [GeV]')
        add(f'z_top{k}_slots', f'pT share of the {k} hardest particles', f'sum(pt[:{k}]) / tot', f'pT share of the {k} hardest particles')
        add(f'girth2_top{k}', f'Σ zΔR² of the {k} hardest particles', f'sum(pt[i] * dr[i] ** 2 for i in range({k})) / max(sum(pt[:{k}]), 1e-9)', f'pT-weighted mean ΔR² of the {k} hardest particles')
        add(f'n_real_top{k}', f'number of real particles among the {k} hardest', f'sum(1 for x in pt[:{k}] if x > 0)', f'number of real particles among the {k} hardest')
    # ---- energy correlation functions (β = 1 unless noted) ----
    E = lambda s: f"ecf('{s}')"; mx = lambda s: f'max({s}, 1e-30)'
    add('e3', 'e₃ = Σᵢ<ⱼ<ₖ zᵢzⱼzₖ ΔRᵢⱼΔRᵢₖΔRⱼₖ', E('e3'), f'energy correlation e3 (β=1, {H3} hardest)')
    add('e4', f'e₄ (hardest {H4})', E('e4'), f'energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, {H4} hardest)')
    add('C3', 'C₃ = e₄e₂/e₃²', f"{E('e4')} * {E('e2')} / {mx(E('e3') + ' ** 2')}", 'energy correlation ratio e4·e2/e3² (small = three-prong)')
    add('D3', 'D₃ = e₄e₂³/e₃³', f"{E('e4')} * {E('e2')} ** 3 / {mx(E('e3') + ' ** 3')}", 'energy correlation ratio e4·e2³/e3³ (small = three-prong)')
    add('N2', 'N₂ = ₂e₃/(₁e₂)²', f"{E('g32')} / {mx(E('e2') + ' ** 2')}", 'generalized ECF ratio N2 (two smallest angles; small = two-prong)')
    add('N3', 'N₃ = ₂e₄/(₁e₃)²', f"{E('g42')} / {mx(E('g31') + ' ** 2')}", 'generalized ECF ratio N3 (small = three-prong)')
    add('M2', 'M₂ = ₁e₃/₁e₂', f"{E('g31')} / {mx(E('e2'))}", 'generalized ECF ratio M2 (smallest angle)')
    add('M3', 'M₃ = ₁e₄/₁e₃', f"{E('g41')} / {mx(E('g31'))}", 'generalized ECF ratio M3')
    add('C2_b2', 'C₂ (β=2) = e₃/e₂²', f"{E('e3b2')} / {mx(E('e2b2') + ' ** 2')}", 'energy correlation ratio e3/e2² with β = 2')
    add('D2_b2', 'D₂ (β=2) = e₃/e₂³', f"{E('e3b2')} / {mx(E('e2b2') + ' ** 3')}", 'energy correlation ratio e3/e2³ with β = 2')
    # ---- N-subjettiness, subjets ----
    for k in (1, 2, 3, 4): add(f'tau{k}', f'τ{k}', f'tau_n({k})', f'N-subjettiness τ{k} (β=1)')
    add('tau43', 'τ43', 'tau_n(4) / max(tau_n(3), 1e-12)', 'N-subjettiness τ4/τ3')
    add('tau21_b2', 'τ21 (β=2)', 'tau_n(2, 2) / max(tau_n(1, 2), 1e-12)', 'N-subjettiness τ2/τ1 with β = 2 (same axes)')
    add('sj2_zsoft', 'pT share of the softer of 2 subjets', 'subjets(2)["z"][1]', 'pT share of the softer of the 2 N-subjettiness subjets')
    add('sj2_dr', 'ΔR between 2 subjet axes', 'subjets(2)["dr"][0]', 'distance between the 2 subjet axes')
    add('sj2_mass1', 'mass of the harder of 2 subjets', 'subjets(2)["mass"][0]', 'mass of the harder of 2 subjets [GeV]')
    add('sj2_mass2', 'mass of the softer of 2 subjets', 'subjets(2)["mass"][1]', 'mass of the softer of 2 subjets [GeV]')
    for p in range(3):
        add(f'sj3_z{p + 1}', f'pT share of subjet {p + 1} of 3', f'subjets(3)["z"][{p}]', f'pT share of subjet {p + 1} of 3 (by pT)')
        add(f'sj3_mass{p + 1}', f'mass of subjet {p + 1} of 3', f'subjets(3)["mass"][{p}]', f'mass of subjet {p + 1} of 3 [GeV]')
    for c, (p, q) in enumerate(itertools.combinations(range(3), 2)):
        add(f'sj3_dr{p + 1}{q + 1}', f'ΔR between subjets {p + 1} and {q + 1} of 3', f'subjets(3)["dr"][{c}]', f'distance between subjet axes {p + 1} and {q + 1} (of 3)')
    add('sj3_dr_min', 'smallest ΔR among 3 subjets', 'min(subjets(3)["dr"])', 'smallest distance among the 3 subjet axes')
    add('sj3_dr_max', 'largest ΔR among 3 subjets', 'max(subjets(3)["dr"])', 'largest distance among the 3 subjet axes')
    add('sj3_pair_mass_min', 'smallest pair mass among 3 subjets', 'min(subjets(3)["mpair"])', 'smallest mass of two of the 3 subjets [GeV]')
    add('sj3_pair_mass_max', 'largest pair mass among 3 subjets', 'max(subjets(3)["mpair"])', 'largest mass of two of the 3 subjets [GeV]')
    add('sj3_pairmin_over_m', 'smallest subjet-pair m / jet m (3 subjets)', 'min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9)', 'smallest subjet-pair mass / jet mass (dimensionless)')
    add('sj3_pairmax_over_m', 'largest subjet-pair m / jet m (3 subjets)', 'max(subjets(3)["mpair"]) / max(mass_of(n), 1e-9)', 'largest subjet-pair mass / jet mass (dimensionless)')
    # ---- soft drop ----
    add('sd_mass', 'soft-drop mass (β=0, z_cut=0.1)', 'softdrop("mass")', f'soft-drop groomed mass, C/A on the {HSD} hardest, β=0, z_cut=0.1 [GeV]')
    add('sd_zg', 'soft-drop z_g', 'softdrop("zg")', 'momentum sharing of the soft-drop splitting')
    add('sd_rg', 'soft-drop R_g', 'softdrop("rg")', 'angle of the soft-drop splitting')
    add('sd_nremoved', 'soft-drop: number of removed branches', 'softdrop("removed")', 'number of branches removed by soft drop')
    if not isinstance(net, int): extras(add, net)
    return L


PAIR_BLOCK = False                   # block B (ParT's pair inputs for all 8,128 pairs): off for now (step 1 cost)
LNEPS = -18.420680743952367          # ln(1e-8): ParT's floor for its logarithms; also the value of a missing particle or pair
BETAS = ((0.5, 'b05'), (2, 'b2'))   # the extra angular exponents of the energy correlations
PPART = dict(kin=['lnpt', 'lne', 'lnptrel', 'lnerel', 'dr', 'eta', 'phi'],
             full=['lnpt', 'lne', 'lnptrel', 'lnerel', 'dr', 'eta', 'phi', 'charge', 'ischhad', 'isnhad', 'isphoton', 'iselectron', 'ismuon',
                   'td0', 'd0err', 'tdz', 'dzerr'])
PPAIR = ['lndelta', 'lnkt', 'lnz', 'lnm2']
PDESC = dict(lnpt='ln pT [GeV]', lne='ln E [GeV]', lnptrel='ln(pT / pT of the jet)', lnerel='ln(E / E of the jet)', dr='ΔR from the jet axis', eta='Δη',
             phi='Δφ', charge='charge', ischhad='1 if a charged hadron', isnhad='1 if a neutral hadron', isphoton='1 if a photon', iselectron='1 if an electron',
             ismuon='1 if a muon', td0='tanh(d0 [mm])', d0err='σ(d0), clipped to [0, 1]', tdz='tanh(dz [mm])', dzerr='σ(dz), clipped to [0, 1]')
QDESC = dict(lndelta='ln ΔRᵢⱼ', lnkt='ln kT = ln(min(pTᵢ, pTⱼ)·ΔRᵢⱼ)', lnz='ln z = ln(min(pTᵢ, pTⱼ)/(pTᵢ + pTⱼ))', lnm2='ln mᵢⱼ² (massless)')
PTYPES = dict(charged_had=('charged hadrons', 'ptype[i] == 1'), neutral_had=('neutral hadrons', 'ptype[i] == 2'), photon=('photons', 'ptype[i] == 3'),
              electron=('electrons', 'ptype[i] == 4'), muon=('muons', 'ptype[i] == 5'), charged=('charged particles', 'charge[i] != 0'),
              neutral=('neutral particles', 'charge[i] == 0'))
BLOCK_RX = re.compile(r'(?:' + '|'.join(PPART['full']) + r')_(\d+)|(?:' + '|'.join(PPAIR) + r')_(\d+)_(\d+)')


def is_block(q):
    """ParT's per-particle inputs (block A) and pair inputs (block B): computed from the particles when used"""
    return TAGGER == 'part' and BLOCK_RX.fullmatch(q) is not None


def extras(add, net):
    """ParT: everything the network sees beyond (pT, Δη, Δφ). Block A: its per-particle inputs for all 128 slots; block B:
    its pair inputs for all pairs; then jet-level quantities: the jet, pair summaries, the primary Lund plane, more energy
    correlations and N-subjettiness, and for 'full' particle types, charges and track impact parameters.
    Energies and the jet pT / η / energy are the file's values."""
    N = 128
    for i in range(N):                                            # ---- block A: per particle
        for f in PPART[net]:
            add(f'{f}_{i}', f'{PDESC[f]} of particle {i}', f'pfeat({i}, {f!r})', f'{PDESC[f]} of particle {i} (ParT input; empty slot: {"ln 1e-8" if f.startswith("ln") else "0"})')
    for i in range(N if PAIR_BLOCK else 0):                       # ---- block B: every pair (off: PAIR_BLOCK)
        for j in range(i + 1, N):
            for f in PPAIR:
                add(f'{f}_{i}_{j}', f'{QDESC[f]} of particles {i}, {j}', f'pairf({i}, {j}, {f!r})', f'{QDESC[f]} of particles {i} and {j} (ParT pair input; either slot empty: ln 1e-8)')
    # ---- the jet ----
    add('jet_pt', 'pT of the jet', 'jet_pt', 'jet pT [GeV]')
    add('jet_abs_eta', '|η_jet|', 'abs(jet_eta)', 'absolute pseudorapidity of the jet axis')
    add('jet_e', 'E of the jet', 'jet_energy', 'jet energy [GeV]')
    add('sum_e', 'Σᵢ Eᵢ', 'sum(energy[i] for i in real)', 'total energy of the particles [GeV]')
    # ---- pair summaries (over all pairs of real particles; weights zᵢzⱼ) ----
    for f in PPAIR:
        add(f'pair_mean_{f}', f'Σ zᵢzⱼ {QDESC[f].split(" =")[0]} / Σ zᵢzⱼ', f'pairsum({f!r})', f'zᵢzⱼ-weighted mean of {QDESC[f]} over all pairs')
    add('pair_max_lnkt', 'max ln kT of a pair', "pairmax('lnkt')", 'largest ln kT among all pairs')
    add('pair_max_lnm2', 'max ln m² of a pair', "pairmax('lnm2')", 'largest ln m² among all pairs')
    for c in (1, 3, 10, 30):
        add(f'n_pairs_kt_above_{c}', f'Σᵢ<ⱼ [kTᵢⱼ > {c} GeV]', f'paircount({c})', f'number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > {c} GeV')
    # ---- primary Lund plane (C/A on all particles, follow the harder branch) ----
    for k in (1, 2, 3):
        for f, d in (('lndelta', 'ln Δ'), ('lnkt', 'ln kT'), ('lnz', 'ln z')):
            add(f'lund{k}_{f}', f'{d} of primary splitting {k}', f'lund({k}, {f!r})', f'{d} of the {k}. primary C/A splitting (ln 1e-8 if none)')
    add('lund_max_lnkt', 'max ln kT of the primary splittings', "lund(0, 'maxkt')", 'largest ln kT among the primary splittings')
    add('lund_max_lndelta', 'ln Δ of the hardest-kT splitting', "lund(0, 'maxdelta')", 'ln Δ of the primary splitting with the largest kT')
    add('n_lund', 'number of primary splittings', "lund(0, 'n')", 'number of primary C/A splittings')
    for c in (1, 5):
        add(f'n_lund_kt_above_{c}', f'primary splittings with kT > {c} GeV', f"lund(0, 'n{c}')", f'number of primary splittings with kT > {c} GeV')
    # ---- more energy correlations (3- and 4-point; β = 0.5 and 2) ----
    for g, d in (('g31', '₁e₃'), ('g32', '₂e₃'), ('g41', '₁e₄'), ('g42', '₂e₄'), ('g43', '₃e₄')):
        add(f'ecf_{g}', d, f"ecfb({g!r}, 1)", f'generalized energy correlation {d} (β=1; products of the smallest angles)')
    for b, t in BETAS:
        E = lambda nm: f'ecfb({nm!r}, {b})'; mx = lambda x: f'max({x}, 1e-30)'
        add(f'e2_{t}', f'e₂ (β={b})', E('e2'), f'energy correlation e2 with β = {b}')
        add(f'e3_{t}', f'e₃ (β={b})', E('e3'), f'energy correlation e3 with β = {b}')
        add(f'e4_{t}', f'e₄ (β={b})', E('e4'), f'energy correlation e4 with β = {b}')
        if b != 2:
            add(f'C2_{t}', f'C₂ (β={b})', f"{E('e3')} / {mx(E('e2') + ' ** 2')}", f'e3/e2² with β = {b}')
            add(f'D2_{t}', f'D₂ (β={b})', f"{E('e3')} / {mx(E('e2') + ' ** 3')}", f'e3/e2³ with β = {b}')
        add(f'C3_{t}', f'C₃ (β={b})', f"{E('e4')} * {E('e2')} / {mx(E('e3') + ' ** 2')}", f'e4·e2/e3² with β = {b}')
        add(f'D3_{t}', f'D₃ (β={b})', f"{E('e4')} * {E('e2')} ** 3 / {mx(E('e3') + ' ** 3')}", f'e4·e2³/e3³ with β = {b}')
        add(f'N2_{t}', f'N₂ (β={b})', f"{E('g32')} / {mx(E('e2') + ' ** 2')}", f'₂e₃/(e2)² with β = {b}')
        add(f'N3_{t}', f'N₃ (β={b})', f"{E('g42')} / {mx(E('g31') + ' ** 2')}", f'₂e₄/(₁e₃)² with β = {b}')
        add(f'M2_{t}', f'M₂ (β={b})', f"{E('g31')} / {mx(E('e2'))}", f'₁e₃/e2 with β = {b}')
        add(f'M3_{t}', f'M₃ (β={b})', f"{E('g41')} / {mx(E('g31'))}", f'₁e₄/₁e₃ with β = {b}')
    # ---- N-subjettiness and subjets (more prongs) ----
    add('tau5', 'τ5', 'tau_n(5)', 'N-subjettiness τ5 (β=1)')
    add('tau54', 'τ54', 'tau_n(5) / max(tau_n(4), 1e-12)', 'N-subjettiness τ5/τ4')
    add('tau32_b2', 'τ32 (β=2)', 'tau_n(3, 2) / max(tau_n(2, 2), 1e-12)', 'N-subjettiness τ3/τ2 with β = 2')
    add('tau43_b2', 'τ43 (β=2)', 'tau_n(4, 2) / max(tau_n(3, 2), 1e-12)', 'N-subjettiness τ4/τ3 with β = 2')
    add('sj4_zsoft', 'pT share of the softest of 4 subjets', 'subjets(4)["z"][3]', 'pT share of the softest of 4 subjets')
    add('sj4_dr_min', 'smallest ΔR among 4 subjets', 'min(subjets(4)["dr"])', 'smallest distance among the 4 subjet axes')
    add('sj4_pair_mass_min', 'smallest pair mass among 4 subjets', 'min(subjets(4)["mpair"])', 'smallest mass of two of the 4 subjets [GeV]')
    add('sj4_pair_mass_max', 'largest pair mass among 4 subjets', 'max(subjets(4)["mpair"])', 'largest mass of two of the 4 subjets [GeV]')
    if net != 'full': return
    # ---- particle types and charge ----
    for k, (nm, cond) in PTYPES.items():
        add(f'n_{k}', f'number of {nm}', f'sum(1 for i in real if {cond})', f'number of {nm}')
        add(f'z_{k}', f'pT share of {nm}', f'sum(z[i] for i in real if {cond})', f'pT share of {nm}')
    for c in (1, 10):
        add(f'n_charged_pt_above_{c}', f'charged particles with pT > {c} GeV', f'sum(1 for i in real if charge[i] != 0 and pt[i] > {c})', f'number of charged particles with pT > {c} GeV')
    add('n_lepton', 'number of leptons', 'sum(1 for i in real if ptype[i] in (4, 5))', 'number of electrons and muons')
    add('jet_charge', 'Σᵢ qᵢ zᵢ', 'sum(charge[i] * z[i] for i in real)', 'pT-weighted jet charge (κ = 1)')
    add('jet_charge_k05', 'Σᵢ qᵢ zᵢ^0.5', 'sum(charge[i] * z[i] ** 0.5 for i in real)', 'jet charge with κ = 0.5')
    add('jet_charge_k03', 'Σᵢ qᵢ zᵢ^0.3', 'sum(charge[i] * z[i] ** 0.3 for i in real)', 'jet charge with κ = 0.3')
    add('sum_charge', 'Σᵢ qᵢ', 'sum(charge[i] for i in real)', 'total charge of the particles')
    add('lead_charge', 'charge of the hardest charged particle', 'next((charge[i] for i in real if charge[i] != 0), 0.0)', 'charge of the hardest charged particle')
    # ---- the hardest lepton ----
    add('lep_z', 'pT share of the hardest lepton', "lepton('z')", 'pT share of the hardest electron or muon (0 if none)')
    add('lep_dr', 'ΔR of the hardest lepton', "lepton('dr')", 'ΔR of the hardest lepton from the jet axis (0 if none)')
    add('lep_ptrel', 'pT·ΔR of the hardest lepton', "lepton('ptrel')", 'pT × ΔR from the jet axis of the hardest lepton [GeV] (0 if none)')
    add('lep_sd0', 'd0/σ of the hardest lepton', "lepton('sd0')", 'signed d0/σ(d0) of the hardest lepton (0 if none)')
    add('lep_iso', 'isolation of the hardest lepton', "lepton('iso')", 'Σ pT of the other particles within ΔR < 0.2 of the hardest lepton / its pT (0 if none)')
    # ---- tracks: impact parameters (charged particles; σ floored at 1e-6) ----
    for w, lab in (('d0', 'd0'), ('dz', 'dz'), ('3d', '3D')):
        for r in (1, 2, 3):
            add(f'sip_{w}_{r}', f'{lab} significance of track {r}', f'sip({w!r}, {r})', f'the {r}. largest {lab} significance among the charged particles' + (' (√((d0/σ)² + (dz/σ)²))' if w == '3d' else f', signed ({lab}/σ)') + ' (0 if fewer)')
    for w, cs in (('d0', (2, 3, 5, 10)), ('dz', (2, 5)), ('3d', (3, 10))):
        for c in cs:
            add(f'n_s{w}_above_{c}', f'tracks with {w} significance > {c}', f'nsig({w!r}, {c})', f'number of charged particles with {w} significance > {c}')
    add('max_abs_d0', 'max |d0| [mm]', 'max([abs(d0[i]) for i in real if charge[i] != 0] or [0.0])', 'largest |d0| among the charged particles [mm]')
    add('max_abs_dz', 'max |dz| [mm]', 'max([abs(dz[i]) for i in real if charge[i] != 0] or [0.0])', 'largest |dz| among the charged particles [mm]')
    add('lead_ch_sd0', 'd0/σ of the hardest track', "leadtrack('d0')", 'signed d0/σ of the hardest charged particle (0 if none)')
    add('lead_ch_sdz', 'dz/σ of the hardest track', "leadtrack('dz')", 'signed dz/σ of the hardest charged particle (0 if none)')
    for c in (3, 5):
        add(f'z_displaced{c}', f'pT share of tracks with |d0/σ| > {c}', f"displaced({c}, 'z')", f'pT share of the charged particles with |d0|/σ > {c}')
        add(f'mass_displaced{c}', f'mass of tracks with |d0/σ| > {c}', f"displaced({c}, 'mass')", f'invariant mass of the charged particles with |d0|/σ > {c} [GeV]')
    # ---- masses of particle subsets ----
    add('mass_charged', 'mass of the charged particles', "subset_mass('charged')", 'invariant mass of all charged particles [GeV]')
    add('mass_neutral', 'mass of the neutral particles', "subset_mass('neutral')", 'invariant mass of all neutral particles [GeV]')
    add('mass_2photon', 'mass of the 2 hardest photons', "subset_mass('photon2')", 'invariant mass of the 2 hardest photons [GeV] (0 if fewer)')
    add('mass_2charged', 'mass of the 2 hardest charged', "subset_mass('charged2')", 'invariant mass of the 2 hardest charged particles [GeV]')


def mass_ids(n):
    """observables with units of mass, and the dimensionless ratios built from masses (removed without mass observables);
    ParT: also its pair masses ln m²ᵢⱼ and their summaries"""
    return {k for k, o in library(n).items() if 'mass' in k or k in ('m01', 'm012') or k.startswith('lnm2_') or k.endswith('_lnm2')}
