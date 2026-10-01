"""Particle Transformer (ParT) jet tagger, JetClass, input features 'full': tuned on the network's predictions (from 100 if-statements per neuron, pruned), with normalized weights (how much each one matters), as if-statements.

Input:  the particles of a jet (up to 128), hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
        energy: each particle's energy [GeV]; jet_pt, jet_eta, jet_energy: the jet's pT [GeV], pseudorapidity, energy [GeV].
Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the training jets, so share_k is the fraction of the neuron's
             average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron 16:  12.6%
  neuron 24:  11.0%
  neuron 18:   9.9%
  neuron 97:   9.7%
  neuron 120:   8.7%
  neuron 68:   8.4%
  neuron 104:   8.4%
  neuron 115:   7.6%
  neuron 52:   6.8%
  neuron 78:   5.9%
  neuron 83:   4.9%
  neuron 81:   4.8%
  neuron 124:   0.9%
  neuron 25:   0.2%
  neuron 70:   0.1%
  neuron 123:   0.1%
  neuron  0:   0.0%
  neuron 90:   0.0%
  neuron 21:   0.0%
  neuron 99:   0.0%
  neuron 84:   0.0%
  neuron 55:   0.0%
  neuron 79:   0.0%
  neuron 105:   0.0%
  neuron 106:   0.0%
  neuron 113:   0.0%
  neuron 34:   0.0%
  neuron 73:   0.0%
  neuron 10:   0.0%
  neuron 11:   0.0%
  neuron  6:   0.0%
  neuron 86:   0.0%
  neuron 32:   0.0%
  neuron 17:   0.0%
  neuron 38:   0.0%
  neuron 42:   0.0%
  neuron 54:   0.0%
  neuron 29:   0.0%
  neuron 118:   0.0%
  neuron 48:   0.0%
  neuron 45:   0.0%
  neuron 39:   0.0%
  neuron 27:   0.0%
  neuron 63:   0.0%
  neuron 30:   0.0%
  neuron 43:   0.0%
  neuron 121:   0.0%
  neuron 80:   0.0%
  neuron 126:   0.0%
  neuron 60:   0.0%
  neuron 61:   0.0%
  neuron 102:   0.0%
  neuron 107:   0.0%
  neuron  7:   0.0%
  neuron 92:   0.0%
  neuron 35:   0.0%
  neuron 15:   0.0%
  neuron 59:   0.0%
  neuron 101:   0.0%
  neuron 28:   0.0%
  neuron 103:   0.0%
  neuron 69:   0.0%
  neuron 67:   0.0%
  neuron  1:   0.0%
  neuron 100:   0.0%
  neuron 110:   0.0%
  neuron  3:   0.0%
  neuron 37:   0.0%
  neuron 50:   0.0%
  neuron 44:   0.0%
  neuron 36:   0.0%
  neuron 12:   0.0%
  neuron 114:   0.0%
  neuron  2:   0.0%
  neuron 58:   0.0%
  neuron 112:   0.0%
  neuron 76:   0.0%
  neuron 13:   0.0%
  neuron 19:   0.0%
  neuron 111:   0.0%
  neuron 74:   0.0%
  neuron 108:   0.0%
  neuron  5:   0.0%
  neuron  8:   0.0%
  neuron 91:   0.0%
  neuron  4:   0.0%
  neuron 117:   0.0%
  neuron 53:   0.0%
  neuron 66:   0.0%
  neuron 33:   0.0%
  neuron 65:   0.0%
  neuron 82:   0.0%
  neuron 31:   0.0%
  neuron 71:   0.0%
  neuron 94:   0.0%
  neuron  9:   0.0%
  neuron 26:   0.0%
  neuron 14:   0.0%
  neuron 57:   0.0%
  neuron 49:   0.0%
  neuron 77:   0.0%
  neuron 22:   0.0%
  neuron 122:   0.0%
  neuron 72:   0.0%
  neuron 51:   0.0%
  neuron 23:   0.0%
  neuron 89:   0.0%
  neuron 125:   0.0%
  neuron 75:   0.0%
  neuron 87:   0.0%
  neuron 93:   0.0%
  neuron 40:   0.0%
  neuron 20:   0.0%
  neuron 119:   0.0%
  neuron 62:   0.0%
  neuron 95:   0.0%
  neuron 96:   0.0%
  neuron 116:   0.0%
  neuron 56:   0.0%
  neuron 85:   0.0%
  neuron 109:   0.0%
  neuron 98:   0.0%
  neuron 88:   0.0%
  neuron 46:   0.0%
  neuron 47:   0.0%
  neuron 64:   0.0%
  neuron 127:   0.0%
  neuron 41:   0.0%
                  built from if-statements.
3. logits():      the network's own last layer: the 128 numbers multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Whole test file (2,000,000 jets): accuracy 74.99% (the network: 86.03%); same class as the network for 79.75% of jets.

Quantities:
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3_b05                 e4·e2/e3² with β = 0.5
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3_b05                 e4·e2³/e3³ with β = 0.5
  Q.D3_b2                  e4·e2³/e3³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M2_b2                  ₁e₃/e2 with β = 2
  Q.M3                     generalized ECF ratio M3
  Q.M3_b2                  ₁e₄/₁e₃ with β = 2
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N2_b05                 ₂e₃/(e2)² with β = 0.5
  Q.N2_b2                  ₂e₃/(e2)² with β = 2
  Q.N3_b05                 ₂e₄/(₁e₃)² with β = 0.5
  Q.ak02_1_n_lep           hardest anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_1_z               pT share of the hardest anti-kT 0.2 subjet (0 if none)
  Q.ak02_2_n_lep           2nd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_2_z               pT share of the 2nd anti-kT 0.2 subjet (0 if none)
  Q.ak02_3_z               pT share of the 3rd anti-kT 0.2 subjet (0 if none)
  Q.ak02_dr13              distance between the pT-weighted centres of the hardest and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr23              distance between the pT-weighted centres of the 2nd and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_min12_jp          the smaller of the two hardest anti-kT 0.2 subjets: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.ak02_n                 number of anti-kT R = 0.2 subjets with pT > 10 GeV
  Q.dc_1_n_lep             hardest prong: number of its electrons and muons
  Q.dc_1_z                 pT share of the hardest prong (0 if none)
  Q.dc_1_z_disp3           hardest prong: pT share (of the jet) of its tracks with d0/σ > 3
  Q.dc_2_jp                2nd prong: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.dc_2_n_lep             2nd prong: number of its electrons and muons
  Q.dc_3_z                 pT share of the 3rd prong (0 if none)
  Q.dc_ntag                number of the 4 hardest prongs whose 2nd largest d0/σ is above 3
  Q.dc_pair_mass_min       smallest mass (E) of two of the 4 hardest prongs [GeV] (0 if fewer than 2)
  Q.dc_split2_dr           ΔR of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_tag_2nd             second largest 2nd-largest d0/σ among the 4 hardest prongs (double tag)
  Q.dr02                   ΔR between particles 0 and 2
  Q.dr_19                  ΔR from the jet axis of particle 19 (ParT input; empty slot: 0)
  Q.dr_29                  ΔR from the jet axis of particle 29 (ParT input; empty slot: 0)
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.e3                     energy correlation e3 (β=1, 32 hardest)
  Q.e3_b05                 energy correlation e3 with β = 0.5
  Q.e3_b2                  energy correlation e3 with β = 2
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 16 hardest)
  Q.e4_b05                 energy correlation e4 with β = 0.5
  Q.ecf_g31                generalized energy correlation ₁e₃ (β=1; products of the smallest angles)
  Q.ecf_g41                generalized energy correlation ₁e₄ (β=1; products of the smallest angles)
  Q.ecf_g42                generalized energy correlation ₂e₄ (β=1; products of the smallest angles)
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.ischhad_0              1 if a charged hadron of particle 0 (ParT input; empty slot: 0)
  Q.jd_3d_4                the 4th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_5                the 5th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_6                the 6th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_sum_abs_sd0_top3    sum of the 3 largest |d0/σ|
  Q.jd_sum_abs_sd0_top5    sum of the 5 largest |d0/σ|
  Q.jet_abs_eta            absolute pseudorapidity of the jet axis
  Q.jet_charge_k03         jet charge with κ = 0.3
  Q.jet_charge_k05         jet charge with κ = 0.5
  Q.jet_e                  jet energy [GeV]
  Q.kt2_1_n_disp3          hardest kT subjet: number of its tracks with d0/σ > 3
  Q.kt2_2_n_lep            2nd kT subjet: number of its electrons and muons
  Q.kt2_min12_jp           the smaller of the two hardest kT subjets: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.ktd_ln_d34             ln of the exclusive-kT merging scale from 4 to 3 subjets, min(pT²)ΔR², over (Σ pT)² (0 if fewer particles)
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lep_dr                 ΔR of the hardest lepton from the jet axis (0 if none)
  Q.lep_iso                Σ pT of the other particles within ΔR < 0.2 of the hardest lepton / its pT (0 if none)
  Q.lep_ptrel              pT × ΔR from the jet axis of the hardest lepton [GeV] (0 if none)
  Q.lep_z                  pT share of the hardest electron or muon (0 if none)
  Q.lepsj_2_dr             distance of the hardest lepton from its nearest subjet axis; 2 subjets (0 if no lepton)
  Q.lepsj_2_maxsd0         largest |d0/σ| in the lepton’s nearest subjet (without the lepton); 2 subjets (0 if no lepton)
  Q.lepsj_2_n_d3           number of tracks with |d0/σ| > 3 in the lepton’s nearest subjet (without the lepton); 2 subjets (0 if no lepton)
  Q.lepsj_3_dr             distance of the hardest lepton from its nearest subjet axis; 3 subjets (0 if no lepton)
  Q.lepsj_3_maxsd0         largest |d0/σ| in the lepton’s nearest subjet (without the lepton); 3 subjets (0 if no lepton)
  Q.lepsj_3_n_d3           number of tracks with |d0/σ| > 3 in the lepton’s nearest subjet (without the lepton); 3 subjets (0 if no lepton)
  Q.lne_0                  ln E [GeV] of particle 0 (ParT input; empty slot: ln 1e-8)
  Q.lne_1                  ln E [GeV] of particle 1 (ParT input; empty slot: ln 1e-8)
  Q.lne_16                 ln E [GeV] of particle 16 (ParT input; empty slot: ln 1e-8)
  Q.lne_4                  ln E [GeV] of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lne_8                  ln E [GeV] of particle 8 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_25             ln(pT / pT of the jet) of particle 25 (ParT input; empty slot: ln 1e-8)
  Q.lund3_lndelta          ln Δ of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund_max_lndelta       ln Δ of the primary splitting with the largest kT
  Q.lund_max_lnkt          largest ln kT among the primary splittings
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_2charged          invariant mass of the 2 hardest charged particles [GeV]
  Q.mass_2photon           invariant mass of the 2 hardest photons [GeV] (0 if fewer)
  Q.mass_charged           invariant mass of all charged particles [GeV]
  Q.mass_displaced3        invariant mass of the charged particles with |d0|/σ > 3 [GeV]
  Q.mass_displaced5        invariant mass of the charged particles with |d0|/σ > 5 [GeV]
  Q.mass_neutral           invariant mass of all neutral particles [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_abs_d0             largest |d0| among the charged particles [mm]
  Q.max_abs_dz             largest |dz| among the charged particles [mm]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.mres_pruned_mass       pruned mass: C/A tree, a merge with z < 0.1 and ΔR > m/pT of the jet keeps only its harder branch [GeV]
  Q.mres_sd_mass_b0z005    soft-drop mass, C/A, β = 0, z_cut = 0.05 [GeV]
  Q.mres_sd_mass_b0z02     soft-drop mass, C/A, β = 0, z_cut = 0.2 [GeV]
  Q.mres_sd_mass_b1z01     soft-drop mass, C/A, β = 1, z_cut = 0.1 [GeV]
  Q.mres_sd_mass_b2z01     soft-drop mass, C/A, β = 2, z_cut = 0.1 [GeV]
  Q.mres_sd_prong_mass1    mass of the harder branch at the soft-drop splitting (β = 0, z_cut = 0.1) [GeV]
  Q.mres_sd_prong_mass2    mass of the softer branch at the soft-drop splitting (β = 0, z_cut = 0.1) [GeV]
  Q.mres_sd_rg_b2z01       soft-drop R_g, β = 2, z_cut = 0.1
  Q.n_charged_had          number of charged hadrons
  Q.n_charged_pt_above_1   number of charged particles with pT > 1 GeV
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_dr_0p4_up            number of particles with ΔR ≥ 0.4
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_lepton               number of electrons and muons
  Q.n_lund                 number of primary C/A splittings
  Q.n_lund_kt_above_1      number of primary splittings with kT > 1 GeV
  Q.n_lund_kt_above_5      number of primary splittings with kT > 5 GeV
  Q.n_muon                 number of muons
  Q.n_neutral              number of neutral particles
  Q.n_neutral_had          number of neutral hadrons
  Q.n_pairs_kt_above_1     number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 1 GeV
  Q.n_pairs_kt_above_10    number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 10 GeV
  Q.n_pairs_kt_above_3     number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 3 GeV
  Q.n_particles            number of real particles (pT > 0)
  Q.n_photon               number of photons
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.n_s3d_above_10         number of charged particles with 3d significance > 10
  Q.n_s3d_above_3          number of charged particles with 3d significance > 3
  Q.n_sd0_above_3          number of charged particles with d0 significance > 3
  Q.n_sdz_above_2          number of charged particles with dz significance > 2
  Q.n_sdz_above_5          number of charged particles with dz significance > 5
  Q.nca_sj4_pair2nd_over_mass second largest mass of two of the 4 subjets over the jet mass
  Q.nca_sj4_pair_mass_2nd  second largest mass of two of the 4 subjets [GeV]
  Q.pair_max_lnkt          largest ln kT among all pairs
  Q.pair_max_lnm2          largest ln m² among all pairs
  Q.pair_mean_lndelta      zᵢzⱼ-weighted mean of ln ΔRᵢⱼ over all pairs
  Q.pair_mean_lnkt         zᵢzⱼ-weighted mean of ln kT = ln(min(pTᵢ, pTⱼ)·ΔRᵢⱼ) over all pairs
  Q.pair_mean_lnm2         zᵢzⱼ-weighted mean of ln mᵢⱼ² (massless) over all pairs
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pz_lnd0                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -9 < ln ΔRᵢⱼ ≤ -3
  Q.pz_lnd2                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -2 < ln ΔRᵢⱼ ≤ -1
  Q.pz_lnd3                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -1 < ln ΔRᵢⱼ ≤ 9
  Q.pz_lnkt0               Σ zᵢzⱼ over the pairs of the 40 hardest particles with -9 < ln kT ≤ 0
  Q.pz_lnkt1               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 0 < ln kT ≤ 1
  Q.pz_lnkt3               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 2 < ln kT ≤ 3
  Q.sd_mass                soft-drop groomed mass, C/A on the 128 hardest, β=0, z_cut=0.1 [GeV]
  Q.sdb_0_n                number of tracks with d0/σ ≤ -3
  Q.sdb_0_z                pT share of the tracks with d0/σ ≤ -3
  Q.sdb_2_n                number of tracks with -1 < d0/σ ≤ 1
  Q.sdb_2_z                pT share of the tracks with -1 < d0/σ ≤ 1
  Q.sdb_3_n                number of tracks with 1 < d0/σ ≤ 3
  Q.sdb_4_n                number of tracks with 3 < d0/σ ≤ 10
  Q.sdb_4_z                pT share of the tracks with 3 < d0/σ ≤ 10
  Q.sdb_5_z                pT share of the tracks with d0/σ > 10
  Q.sdb_jp_top3            the 3 largest −ln P(|d0/σ|) summed
  Q.sip_3d_1               the 1. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sip_3d_2               the 2. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sip_3d_3               the 3. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmax_over_m     largest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj4_dr_min             smallest distance among the 4 subjet axes
  Q.sj4_pair_mass_max      largest mass of two of the 4 subjets [GeV]
  Q.sj4_pair_mass_min      smallest mass of two of the 4 subjets [GeV]
  Q.sjf_2_1_max3d          subjet 1 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_1_n_d3           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_2_1_n_d5           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_2_1_z_d3           subjet 1 of 2 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_2_2_max3d          subjet 2 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_2_z_d3           subjet 2 of 2 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_3_1_max3d          subjet 1 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_1_n_d3           subjet 1 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_3_1_n_d5           subjet 1 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_3_2_max3d          subjet 2 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_2_maxsd0         subjet 2 of 3 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_3_3_max3d          subjet 3 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_3_maxsd0         subjet 3 of 3 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_4_1_max3d          subjet 1 of 4 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_4_1_z_d3           subjet 1 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_2_z_d3           subjet 2 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_3_z_d3           subjet 3 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_n2disp           number of the 4 subjets with at least 2 tracks with |d0/σ| > 3
  Q.sjq_2_1_nch            number of charged particles in subjet 1 of 2
  Q.sjq_2_2_k1             Σ q pT^κ / (Σ pT)^κ of subjet 2 of 2, κ = 1
  Q.sjq_2_2_nch            number of charged particles in subjet 2 of 2
  Q.sjq_2_prod_k03         product of the charges (κ = 0.3) of the two hardest of 2 subjets
  Q.sjq_2_prod_k05         product of the charges (κ = 0.5) of the two hardest of 2 subjets
  Q.sjq_2_prod_k1          product of the charges (κ = 1) of the two hardest of 2 subjets
  Q.sjq_2_sumabs_k03       |sum| of the charges (κ = 0.3) of the two hardest of 2 subjets
  Q.sjq_2_sumabs_k1        |sum| of the charges (κ = 1) of the two hardest of 2 subjets
  Q.sjq_3_2_k1             Σ q pT^κ / (Σ pT)^κ of subjet 2 of 3, κ = 1
  Q.sjq_3_2_nch            number of charged particles in subjet 2 of 3
  Q.sjq_3_3_k1             Σ q pT^κ / (Σ pT)^κ of subjet 3 of 3, κ = 1
  Q.sjq_3_prod_k1          product of the charges (κ = 1) of the two hardest of 3 subjets
  Q.sjq_3_sumabs_k1        |sum| of the charges (κ = 1) of the two hardest of 3 subjets
  Q.sum_e                  total energy of the particles [GeV]
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sv_1_dr                hardest displaced-track cluster: ΔR of its centre from the jet axis (0 if none)
  Q.sv_1_n                 hardest displaced-track cluster: number of tracks (0 if none)
  Q.sv_1_sd0_sum           hardest displaced-track cluster: Σ d0/σ of its tracks (0 if none)
  Q.sv_2_z                 2nd displaced-track cluster: pT share of the jet (0 if none)
  Q.sv_n                   number of anti-kT R = 0.1 clusters of the tracks with d0/σ > 3
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau32_b2               N-subjettiness τ3/τ2 with β = 2
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.tau43                  N-subjettiness τ4/τ3
  Q.tau5                   N-subjettiness τ5 (β=1)
  Q.tau54                  N-subjettiness τ5/τ4
  Q.z_charged              pT share of charged particles
  Q.z_charged_had          pT share of charged hadrons
  Q.z_displaced3           pT share of the charged particles with |d0|/σ > 3
  Q.z_displaced5           pT share of the charged particles with |d0|/σ > 5
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.z_dr_0p4_up            pT share of the particles with ΔR ≥ 0.4
  Q.z_electron             pT share of electrons
  Q.z_muon                 pT share of muons
  Q.z_neutral_had          pT share of neutral hadrons
  Q.z_photon               pT share of photons
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
"""
import math
from types import SimpleNamespace

CLASSES = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
W = [[0.05088143050670624, 0.1155136302113533, 0.16600550711154938, 0.2029360979795456, 0.22491300106048584, -0.5611751079559326, -0.04644721373915672, -0.14651615917682648, -0.17138755321502686, -0.6625961065292358], [0.014363662339746952, 0.014363058842718601, 0.014369546435773373, 0.014362843707203865, 0.014363950118422508, 0.014355706982314587, 0.014365989714860916, 0.01436514314264059, 0.014367412775754929, 0.01435252744704485], [-0.010149345733225346, -0.010141393169760704, -0.010149979032576084, -0.010147258639335632, -0.010150057263672352, -0.010159865953028202, -0.010142926126718521, -0.010150880552828312, -0.01014940906316042, -0.010166633874177933], [-0.008032542653381824, -0.008032522164285183, -0.008034964092075825, -0.008032361976802349, -0.008031551726162434, -0.008037381805479527, -0.008030678145587444, -0.008033300749957561, -0.00803344789892435, -0.008044585585594177], [-0.012415152043104172, -0.012414411641657352, -0.012417037971317768, -0.012415740638971329, -0.01241601724177599, -0.012423457577824593, -0.01241646334528923, -0.012417135760188103, -0.01241734903305769, -0.0124318553134799], [0.018808551132678986, 0.0188076701015234, 0.018821457400918007, 0.018816113471984863, 0.018818994984030724, 0.018798161298036575, 0.018818842247128487, 0.018819760531187057, 0.018815426155924797, 0.018800225108861923], [-0.029233312234282494, -0.02923615463078022, -0.029235955327749252, -0.029231488704681396, -0.029236609116196632, -0.029213139787316322, -0.02923823520541191, -0.029242046177387238, -0.029235266149044037, -0.029218951240181923], [-0.01816708594560623, -0.018166257068514824, -0.01817971281707287, -0.018165934830904007, -0.018169455230236053, -0.018170276656746864, -0.018178338184952736, -0.018171850591897964, -0.018172437325119972, -0.018172523006796837], [-0.01325061172246933, -0.013249002397060394, -0.013256094418466091, -0.013253879733383656, -0.013255167752504349, -0.013265701942145824, -0.01325475424528122, -0.013255455531179905, -0.013254094868898392, -0.01326675433665514], [0.007193062454462051, 0.007191248703747988, 0.0071927872486412525, 0.007192373741418123, 0.007192266173660755, 0.007190669886767864, 0.0071923392824828625, 0.0071928612887859344, 0.007192203775048256, 0.007190892938524485], [0.040606606751680374, 0.04060528799891472, 0.040621329098939896, 0.0406026765704155, 0.04061054065823555, 0.040587007999420166, 0.040617380291223526, 0.04060754552483559, 0.04060862585902214, 0.040591832250356674], [0.03218542039394379, 0.03219390660524368, 0.032195158302783966, 0.03218289092183113, 0.032182727009058, 0.032209232449531555, 0.03218784183263779, 0.03219090402126312, 0.03219301998615265, 0.03221141919493675], [-0.011495490558445454, -0.01149073801934719, -0.011493385769426823, -0.011495158076286316, -0.011493068188428879, -0.011506449431180954, -0.011498279869556427, -0.011494604870676994, -0.0114969527348876, -0.01151067204773426], [-0.011318986304104328, -0.011318260803818703, -0.01132102683186531, -0.011318862438201904, -0.011318323202431202, -0.011305463500320911, -0.01131997536867857, -0.011319609358906746, -0.011318108066916466, -0.011304453946650028], [-0.004866324365139008, -0.004865212831646204, -0.004869204014539719, -0.004865723196417093, -0.004865332040935755, -0.004869276657700539, -0.004865934140980244, -0.004867125302553177, -0.0048656705766916275, -0.004868981894105673], [0.015495091676712036, 0.015491540543735027, 0.015497160144150257, 0.015491385944187641, 0.015494843013584614, 0.015490030869841576, 0.015493087470531464, 0.015494529157876968, 0.01549362763762474, 0.015485921874642372], [0.10181883722543716, 1.3344638347625732, -0.19571681320667267, -0.3669143319129944, -0.3562762439250946, -0.036993853747844696, -0.14310306310653687, -0.14991150796413422, 0.8553245067596436, 2.018296241760254], [0.041684847325086594, 0.041672565042972565, 0.04170159623026848, 0.04169374331831932, 0.04169904440641403, 0.041702792048454285, 0.04168292507529259, 0.04168177396059036, 0.04169892892241478, 0.04170497506856918], [-0.7004212737083435, 0.4346098303794861, 0.752185583114624, -0.4526638686656952, -0.2220054715871811, 0.8997800946235657, 0.06621253490447998, 0.007195856422185898, -0.5604346394538879, 1.0857001543045044], [-0.034533847123384476, -0.03456304594874382, -0.03455730527639389, -0.03455520421266556, -0.03454622998833656, -0.03455745428800583, -0.034558266401290894, -0.0345601849257946, -0.03455345705151558, -0.03455956652760506], [-0.004943703766912222, -0.004944415297359228, -0.004946902394294739, -0.004945553373545408, -0.004945577122271061, -0.004944073036313057, -0.004945517983287573, -0.004945946391671896, -0.004945364780724049, -0.0049460334703326225], [-0.07831130921840668, -0.0870489850640297, -0.06785894930362701, -0.0728980228304863, -0.08164841681718826, 0.3600445091724396, -0.08139730244874954, -0.07096368819475174, -0.04609576240181923, 0.34154894948005676], [-0.012095765210688114, -0.012096457183361053, -0.012104102410376072, -0.012092809192836285, -0.012102092616260052, -0.012093005701899529, -0.012098447419703007, -0.012101216241717339, -0.012098347768187523, -0.012091463431715965], [-0.01280591357499361, -0.012803261168301105, -0.012812280096113682, -0.012803354300558567, -0.012802564539015293, -0.012824933975934982, -0.012811262160539627, -0.012816931121051311, -0.012792502529919147, -0.012840954586863518], [-0.33045393228530884, -0.41349226236343384, -0.5988553762435913, -0.2296663522720337, -0.3896954357624054, 1.228258728981018, 0.4463769793510437, 0.4786914885044098, -0.27336615324020386, 1.2449630498886108], [0.028429275378584862, -0.2898458242416382, -0.1922299861907959, -0.13386254012584686, -0.13978658616542816, 0.6962875127792358, 0.24276822805404663, -0.3037947416305542, 0.042212020605802536, 0.8521035313606262], [0.004781550727784634, 0.004777733236551285, 0.004776466637849808, 0.004778571892529726, 0.0047787828370928764, 0.004800261929631233, 0.004778334405273199, 0.004776317626237869, 0.0047810254618525505, 0.004802254494279623], [0.025863366201519966, 0.025864634662866592, 0.025872958824038506, 0.025866467505693436, 0.025869235396385193, 0.02587084285914898, 0.025869624689221382, 0.025872958824038506, 0.025871092453598976, 0.02587881311774254], [-0.009269620291888714, -0.009266220964491367, -0.00927056185901165, -0.009268919005990028, -0.009270304813981056, -0.009250542148947716, -0.00927410926669836, -0.009269274771213531, -0.009276112541556358, -0.009254143573343754], [-0.029541729018092155, -0.029533857479691505, -0.029549315571784973, -0.029537053778767586, -0.029545925557613373, -0.029535967856645584, -0.029546840116381645, -0.029547076672315598, -0.02954772301018238, -0.029544223099946976], [0.01868555136024952, 0.018681535497307777, 0.018684424459934235, 0.018681181594729424, 0.018684279173612595, 0.018685655668377876, 0.018681949004530907, 0.018685100600123405, 0.01868364028632641, 0.018701747059822083], [-0.006075814366340637, -0.006076902616769075, -0.006073683965951204, -0.006074128672480583, -0.006074940320104361, -0.006055628880858421, -0.006074316333979368, -0.00607713358476758, -0.006072286516427994, -0.006063251290470362], [-0.032577916979789734, -0.03256373852491379, -0.03257916122674942, -0.03258126974105835, -0.03257816657423973, -0.03257230296730995, -0.03257627785205841, -0.03257874771952629, -0.03257722407579422, -0.03259680047631264], [-0.009837334975600243, -0.009838376194238663, -0.00983874499797821, -0.009837915189564228, -0.009837055578827858, -0.009810620918869972, -0.009836745448410511, -0.009838344529271126, -0.009837547317147255, -0.009811862371861935], [-0.05506289377808571, -0.05505332723259926, -0.0550570972263813, -0.055062003433704376, -0.055064670741558075, -0.05506820231676102, -0.05506540462374687, -0.05506938323378563, -0.05506429076194763, -0.05507828667759895], [0.01526290737092495, 0.015261156484484673, 0.015265882946550846, 0.01526274811476469, 0.015265767462551594, 0.015249042771756649, 0.015252772718667984, 0.015256480313837528, 0.015245750546455383, 0.015257726423442364], [0.012341375462710857, 0.012340879999101162, 0.01234623696655035, 0.012340350076556206, 0.012341116555035114, 0.012348196469247341, 0.012341818772256374, 0.012343371286988258, 0.012340524233877659, 0.01234869658946991], [0.014874011278152466, 0.01487020030617714, 0.014873611740767956, 0.014871367253363132, 0.01487270649522543, 0.014873671345412731, 0.0148730194196105, 0.014873931184411049, 0.014873447827994823, 0.014879658818244934], [0.04144299775362015, 0.04144911468029022, 0.041437678039073944, 0.041453976184129715, 0.041456323117017746, 0.0414309948682785, 0.04145108163356781, 0.04145601764321327, 0.04145520180463791, 0.04143337532877922], [-0.021387668326497078, -0.021386567503213882, -0.021392228081822395, -0.02138524129986763, -0.02138705551624298, -0.0213723573833704, -0.021392829716205597, -0.02138928510248661, -0.02138681337237358, -0.021377239376306534], [-0.005647566635161638, -0.005646564532071352, -0.005647305864840746, -0.005646994803100824, -0.005647668149322271, -0.005644064862281084, -0.005647450685501099, -0.00564760435372591, -0.00564756290987134, -0.005645777564495802], [0.00015941228775773197, 0.00015910410729702562, 0.0001596047804923728, 0.00015934955445118248, 0.00015951339446473867, 0.00016594961925875396, 0.00015913501556497067, 0.00015932154201436788, 0.00015945253835525364, 0.00016687207971699536], [-0.036483775824308395, -0.03648171201348305, -0.03648370876908302, -0.03647801652550697, -0.036482084542512894, -0.03646915778517723, -0.036481283605098724, -0.03648832440376282, -0.03648745268583298, -0.03649463504552841], [0.016732843592762947, 0.01673162914812565, 0.01673036627471447, 0.01673085428774357, 0.016732288524508476, 0.016719134524464607, 0.01673193834722042, 0.016732821241021156, 0.016729524359107018, 0.016723206266760826], [0.01795245334506035, 0.017953241243958473, 0.017957551404833794, 0.017952224239706993, 0.01795322261750698, 0.017939135432243347, 0.017953837290406227, 0.01795400120317936, 0.017953557893633842, 0.017944641411304474], [-0.026753153651952744, -0.026743266731500626, -0.026751846075057983, -0.026751892641186714, -0.026752987876534462, -0.026754610240459442, -0.02675226889550686, -0.026755936443805695, -0.026747679337859154, -0.026757702231407166], [-0.0022174238692969084, -0.002215974498540163, -0.00221756799146533, -0.0022168499417603016, -0.0022173395846039057, -0.0022310472559183836, -0.0022171474993228912, -0.002216631080955267, -0.0022173605393618345, -0.0022332819644361734], [-0.001149711082689464, -0.001149944611825049, -0.0011494186474010348, -0.001149537623859942, -0.001149741350673139, -0.0011465927818790078, -0.001149659394286573, -0.0011497549712657928, -0.0011494933860376477, -0.0011466926662251353], [0.021839484572410583, 0.021835932508111, 0.02184072509407997, 0.021838931366801262, 0.021838178858160973, 0.021825015544891357, 0.021839775145053864, 0.021842898800969124, 0.021842587739229202, 0.021828046068549156], [0.005777475889772177, 0.005777162965387106, 0.005776997655630112, 0.005776918027549982, 0.005777184851467609, 0.005770324729382992, 0.005776988808065653, 0.005777470767498016, 0.005777243059128523, 0.005771378055214882], [-0.012268471531569958, -0.0122578339651227, -0.012265805155038834, -0.012269191443920135, -0.0122604975476861, -0.01225277129560709, -0.012262878008186817, -0.012265702709555626, -0.012259438633918762, -0.012250489555299282], [0.00451144902035594, 0.004510563798248768, 0.0045146336778998375, 0.004514096304774284, 0.004513600375503302, 0.004529430530965328, 0.004510956816375256, 0.004513333085924387, 0.004512472078204155, 0.0045332047156989574], [0.5141751766204834, -0.47030913829803467, -0.13294865190982819, 0.5316269397735596, 0.5521713495254517, 1.0491759777069092, -0.2833074927330017, -0.5185812711715698, -0.2555696666240692, 0.8928049802780151], [0.014851833693683147, 0.01484174095094204, 0.014856684021651745, 0.01484659593552351, 0.014845733530819416, 0.014856180176138878, 0.014847379177808762, 0.014845588244497776, 0.014847424812614918, 0.01484605297446251], [-0.023973839357495308, -0.023968631401658058, -0.023978380486369133, -0.023972466588020325, -0.02397889830172062, -0.02399195358157158, -0.023974129930138588, -0.02398003824055195, -0.023976339027285576, -0.023996273055672646], [0.09215790778398514, 0.09214489161968231, 0.09219290316104889, 0.09215540438890457, 0.09218885749578476, 0.09217965602874756, 0.0921855941414833, 0.09217298030853271, 0.09219004958868027, 0.09215881675481796], [0.002212311839684844, 0.002211266662925482, 0.002213289262726903, 0.0022114799357950687, 0.0022132587619125843, 0.0022205295972526073, 0.002211734652519226, 0.0022119893692433834, 0.0022122992668300867, 0.0022212781477719545], [0.006999255158007145, 0.006997383665293455, 0.0070020463317632675, 0.00700021767988801, 0.007000548765063286, 0.007023555226624012, 0.006998357828706503, 0.006999206729233265, 0.007001797668635845, 0.007028547115623951], [0.01473083533346653, 0.014730117283761501, 0.014737415127456188, 0.014735491015017033, 0.014735469594597816, 0.01474799308925867, 0.01473494153469801, 0.014736142940819263, 0.014736202545464039, 0.014752699993550777], [-0.016031896695494652, -0.016031455248594284, -0.016034558415412903, -0.016033604741096497, -0.016029655933380127, -0.01604415476322174, -0.016035910695791245, -0.01604337804019451, -0.016032349318265915, -0.016043635085225105], [0.027231954038143158, 0.027232777327299118, 0.027239888906478882, 0.02723032981157303, 0.027242669835686684, 0.027226855978369713, 0.0272381454706192, 0.027233341708779335, 0.027240147814154625, 0.027231575921177864], [-0.021430788561701775, -0.02142929472029209, -0.02143288217484951, -0.021433880552649498, -0.021429911255836487, -0.021416708827018738, -0.021436864510178566, -0.021432528272271156, -0.021431410685181618, -0.021416008472442627], [-0.0035393184516578913, -0.0035393673460930586, -0.0035400905180722475, -0.00353915523737669, -0.0035404576919972897, -0.0035303947515785694, -0.003540021600201726, -0.0035401228815317154, -0.003539324039593339, -0.003530394984409213], [0.019556595012545586, 0.019553828984498978, 0.019562555477023125, 0.019553976133465767, 0.01955513469874859, 0.019556447863578796, 0.01955641433596611, 0.01955699920654297, 0.01955803483724594, 0.019557829946279526], [-0.0001403831847710535, -0.0001236218522535637, -0.0001437000755686313, -0.00013777091226074845, -0.00014067505253478885, -0.00010860648762900382, -0.0001294568064622581, -0.0001344299380434677, -0.00012457351840566844, -0.00011891438771272078], [-0.004237177781760693, -0.0042359549552202225, -0.004238830413669348, -0.004231867380440235, -0.004233108833432198, -0.004247313365340233, -0.004234891850501299, -0.004236309789121151, -0.004236453678458929, -0.004249528516083956], [-0.01024045143276453, -0.01023187953978777, -0.010242581367492676, -0.010237667709589005, -0.01023886352777481, -0.010235396213829517, -0.010233930312097073, -0.010242711752653122, -0.0102331368252635, -0.010239296592772007], [0.015129251405596733, 0.015124914236366749, 0.015132180415093899, 0.015125013887882233, 0.015129351988434792, 0.015154209919273853, 0.015127970837056637, 0.015127060003578663, 0.015127996914088726, 0.015156702138483524], [-0.4938729703426361, -0.6113813519477844, -0.11253375560045242, -0.8149121999740601, 0.26271408796310425, 0.15879219770431519, 0.35816672444343567, 0.5000133514404297, 0.9971407055854797, -0.6801400780677795], [0.011747613549232483, 0.011743798851966858, 0.011748673394322395, 0.01174565777182579, 0.01174602098762989, 0.011760194785892963, 0.011747940443456173, 0.011747756972908974, 0.011745310388505459, 0.011766991578042507], [0.04877988621592522, -0.2460460066795349, 0.23093295097351074, 0.15028433501720428, 0.19125987589359283, 0.8021718263626099, -0.32731199264526367, -0.29373741149902344, -0.5081436634063721, 0.754387617111206], [-0.006467494647949934, -0.006463835947215557, -0.006465374957770109, -0.006466877646744251, -0.006467239931225777, -0.006460108328610659, -0.006465700920671225, -0.0064676315523684025, -0.006465076003223658, -0.006459739059209824], [0.008141957223415375, 0.008141485042870045, 0.008140197955071926, 0.008142313919961452, 0.008142837323248386, 0.008133607916533947, 0.008143505081534386, 0.008144007064402103, 0.008143291808664799, 0.008133530616760254], [0.040391359478235245, 0.040375858545303345, 0.04039027914404869, 0.04038543999195099, 0.04038925841450691, 0.0403793603181839, 0.04038756713271141, 0.04039502888917923, 0.040382400155067444, 0.04039273411035538], [0.010121549479663372, 0.01012034434825182, 0.01012284867465496, 0.010121201165020466, 0.01012073177844286, 0.010117500089108944, 0.01012113131582737, 0.01012139581143856, 0.010122552514076233, 0.010120649822056293], [-0.0028682355768978596, -0.002867114031687379, -0.002868199022486806, -0.0028682579286396503, -0.002868285635486245, -0.002862118650227785, -0.0028685186989605427, -0.00286908564157784, -0.0028680183459073305, -0.0028624190017580986], [0.009369571693241596, 0.00936775840818882, 0.00937194935977459, 0.009372282773256302, 0.009372487664222717, 0.009378915652632713, 0.00937176588922739, 0.009372388944029808, 0.009372265078127384, 0.00938424188643694], [0.004101733211427927, 0.004103126935660839, 0.00410164101049304, 0.004102480597794056, 0.0041061099618673325, 0.004103580955415964, 0.004105870146304369, 0.0041028279811143875, 0.00410590460523963, 0.004098229575902224], [0.265907883644104, -0.7480407953262329, 0.3953246772289276, -0.18204957246780396, -0.705207347869873, 1.0098243951797485, -0.0645923763513565, -0.2998703420162201, 0.6627867817878723, 0.8632068634033203], [0.035488858819007874, 0.03547275438904762, 0.03549369052052498, 0.0354849249124527, 0.035487640649080276, 0.0354759618639946, 0.0354837067425251, 0.03548821806907654, 0.035501085221767426, 0.03547490015625954], [-0.014816180802881718, -0.014811282977461815, -0.014819124713540077, -0.01481393538415432, -0.01481710560619831, -0.014815201982855797, -0.01481673028320074, -0.01481681689620018, -0.014817516319453716, -0.014815224334597588], [0.6465077996253967, 0.314510703086853, -0.7656049132347107, -0.5250392556190491, -0.5656489729881287, -0.339816153049469, 0.5627880692481995, -0.41398632526397705, 0.4847275912761688, -0.13104437291622162], [0.007712486200034618, 0.007710773032158613, 0.007714828941971064, 0.007711485959589481, 0.007714370731264353, 0.007727183401584625, 0.007713278289884329, 0.00771233020350337, 0.007712431717664003, 0.007732660509645939], [-0.1501917839050293, 0.049915995448827744, -0.9527811408042908, 0.08355074375867844, -0.06172167509794235, 0.720667839050293, 0.3810296952724457, 0.48599594831466675, -0.8601528406143188, -0.10748595744371414], [0.10870547592639923, 0.10866495221853256, 0.10869935899972916, 0.10868468135595322, 0.10867515206336975, 0.10872345417737961, 0.10867581516504288, 0.10869436711072922, 0.10868892073631287, 0.10873434692621231], [0.001328901736997068, 0.0013279394479468465, 0.0013298472622409463, 0.0013294548261910677, 0.001328888232819736, 0.0013378487201407552, 0.0013284055748954415, 0.0013289300259202719, 0.001329375198110938, 0.0013388348743319511], [0.02887844294309616, 0.028868671506643295, 0.028873968869447708, 0.028873158618807793, 0.028874479234218597, 0.028865516185760498, 0.02887343056499958, 0.028879662975668907, 0.028875045478343964, 0.02887771837413311], [-0.0030257455073297024, -0.0030240416526794434, -0.0030253506265580654, -0.003024330595508218, -0.0030249119736254215, -0.0030108275823295116, -0.003026125952601433, -0.0030247049871832132, -0.0030253299046307802, -0.0030102955643087626], [-0.0015024296008050442, -0.001503165578469634, -0.0015015527606010437, -0.001503726001828909, -0.0015030221547931433, -0.0014929373282939196, -0.0015037397388368845, -0.0015031005023047328, -0.0015011944342404604, -0.0014915406936779618], [-0.003556514158844948, -0.0035568897146731615, -0.003556461539119482, -0.0035565963480621576, -0.0035565525759011507, -0.0035370979458093643, -0.003557572141289711, -0.0035579369869083166, -0.003557161893695593, -0.0035349850077182055], [-0.22817395627498627, -0.1659400910139084, 0.005098015069961548, 0.10460185259580612, -0.02178771048784256, 0.9134813547134399, -0.19292452931404114, -0.15784138441085815, -0.3231997489929199, 0.6261100172996521], [-0.009312987327575684, -0.009311968460679054, -0.009318014606833458, -0.00931156799197197, -0.009317419491708279, -0.009332631714642048, -0.009316817857325077, -0.0093145240098238, -0.009312930516898632, -0.009335395880043507], [0.015650015324354172, 0.01564880460500717, 0.015651056542992592, 0.01564791239798069, 0.01564808376133442, 0.015638887882232666, 0.0156512800604105, 0.015651531517505646, 0.01564822718501091, 0.015642037615180016], [0.005768756847828627, 0.00576499430462718, 0.005766183603554964, 0.005765360314399004, 0.0057692271657288074, 0.005770198535174131, 0.005769030191004276, 0.005764967296272516, 0.00576759735122323, 0.005770391784608364], [0.007837331853806973, 0.007833653129637241, 0.007839139550924301, 0.007835190743207932, 0.007836655713617802, 0.007832185365259647, 0.007835864089429379, 0.007833114825189114, 0.007839205674827099, 0.007834256626665592], [-0.006863060873001814, -0.006861493457108736, -0.006861700210720301, -0.006861040368676186, -0.00686428789049387, -0.006859966088086367, -0.006862738635390997, -0.006862489506602287, -0.006861082278192043, -0.006858571898192167], [-0.006496533751487732, -0.006494121626019478, -0.006495033390820026, -0.006494896486401558, -0.006496809422969818, -0.006479903124272823, -0.006496520712971687, -0.006497570779174566, -0.0064943754114210606, -0.006477709859609604], [0.5964194536209106, -0.35329675674438477, -0.2049149125814438, -0.3345659673213959, -0.3492302894592285, 1.064683198928833, -0.37962546944618225, 1.1715655326843262, 0.20428466796875, 0.7683817148208618], [-0.0011296926531940699, -0.0011297534219920635, -0.0011298557510599494, -0.0011296110460534692, -0.0011297379387542605, -0.0011262971675023437, -0.0011299499310553074, -0.001129903830587864, -0.001130119664594531, -0.0011259133461862803], [-0.08836476504802704, -0.08950844407081604, -0.08881659805774689, -0.08831381052732468, -0.08818111568689346, -0.08935540914535522, -0.08861667662858963, -0.08874557912349701, -0.08808904886245728, -0.08743861317634583], [0.012120550498366356, 0.012121576815843582, 0.012124677188694477, 0.01211944967508316, 0.012119777500629425, 0.012111358344554901, 0.01212262362241745, 0.01212262362241745, 0.012122491374611855, 0.01210837997496128], [-0.010503215715289116, -0.010503153316676617, -0.010500754229724407, -0.010501082986593246, -0.010501948185265064, -0.01050981879234314, -0.010503025725483894, -0.010503110475838184, -0.010503125376999378, -0.01051250658929348], [-0.021646879613399506, -0.02164457179605961, -0.021647173911333084, -0.021645724773406982, -0.021645814180374146, -0.021636253222823143, -0.021650280803442, -0.021651307120919228, -0.021644091233611107, -0.021645337343215942], [0.025319060310721397, 0.02531399019062519, 0.025323916226625443, 0.02531825378537178, 0.02531934343278408, 0.025328006595373154, 0.025319432839751244, 0.025323761627078056, 0.025314034894108772, 0.02533205971121788], [0.32737863063812256, -0.026592819020152092, 0.22137123346328735, -1.0014634132385254, -0.5075631141662598, 0.13055816292762756, 0.4256267547607422, 0.24512150883674622, -1.4211052656173706, -1.0893663167953491], [-0.04466724023222923, -0.04466211050748825, -0.04466484859585762, -0.044663649052381516, -0.04466557875275612, -0.04465842992067337, -0.0446665957570076, -0.044671252369880676, -0.04466283321380615, -0.044694412499666214], [-0.045833002775907516, -0.04582158103585243, -0.04584401845932007, -0.04582836106419563, -0.04584681987762451, -0.04584993049502373, -0.0458441860973835, -0.04584534466266632, -0.0458347387611866, -0.04585269093513489], [-0.010243255645036697, -0.010234097018837929, -0.010233195498585701, -0.010238947346806526, -0.010239929892122746, -0.010222965851426125, -0.010235476307570934, -0.010237225331366062, -0.010235852561891079, -0.01022410299628973], [0.009653285145759583, 0.009656237438321114, 0.00965618435293436, 0.009653765708208084, 0.009653080254793167, 0.009633833542466164, 0.009654659777879715, 0.00965622253715992, 0.00965513102710247, 0.009633171372115612], [0.0012451318325474858, 0.0012452565133571625, 0.0012451671063899994, 0.0012451461516320705, 0.0012455715332180262, 0.0012411274947226048, 0.0012456629192456603, 0.0012453942326828837, 0.0012450963258743286, 0.001240548794157803], [-0.019480161368846893, -0.019482124596834183, -0.019487539306282997, -0.01947782002389431, -0.01948321983218193, -0.019465336576104164, -0.019484449177980423, -0.01948034204542637, -0.019485432654619217, -0.019469408318400383], [-0.013004000298678875, -0.013002610765397549, -0.013003001920878887, -0.013003070838749409, -0.013003259897232056, -0.01299121044576168, -0.013004143722355366, -0.013004024513065815, -0.013004918582737446, -0.012990687042474747], [-0.010130255483090878, -0.01012806873768568, -0.010134799405932426, -0.010134600102901459, -0.010133529081940651, -0.010152596049010754, -0.010134615004062653, -0.010136242024600506, -0.010134573094546795, -0.010156993754208088], [0.030198294669389725, 0.030181577429175377, 0.03019983135163784, 0.030195649713277817, 0.03018971160054207, 0.030207810923457146, 0.030188677832484245, 0.030196240171790123, 0.030196746811270714, 0.03022271767258644], [-0.011922081001102924, -0.011915285140275955, -0.011920320801436901, -0.011919545009732246, -0.011921748518943787, -0.011920370161533356, -0.011920128017663956, -0.011919813230633736, -0.01192003209143877, -0.011923297308385372], [0.12737838923931122, 0.2418745756149292, 0.0887438952922821, 0.23817116022109985, -1.4971548318862915, -0.6255540251731873, 0.4396445155143738, 0.23777012526988983, -0.18206532299518585, -0.7693063020706177], [0.004819788038730621, 0.004820294678211212, 0.004823074210435152, 0.004819885827600956, 0.004821108188480139, 0.004812580067664385, 0.00482161296531558, 0.004821083042770624, 0.004820918198674917, 0.004812291823327541], [-0.011602181009948254, -0.011600669473409653, -0.011603194288909435, -0.011600593104958534, -0.011603521183133125, -0.011606484651565552, -0.011602000333368778, -0.01160296332091093, -0.01160038448870182, -0.01161534059792757], [0.029875915497541428, 0.02986983396112919, 0.029880637302994728, 0.029879551380872726, 0.02988334931433201, 0.029888277873396873, 0.02988089621067047, 0.02988274395465851, 0.029879415407776833, 0.02988959103822708], [0.005753437522798777, 0.005746463779360056, 0.0057527609169483185, 0.005750806070864201, 0.005754326935857534, 0.005771062336862087, 0.0057534510269761086, 0.005748684052377939, 0.005753851030021906, 0.005772699601948261], [-0.39496922492980957, 1.1834187507629395, 0.14213189482688904, 0.05709841847419739, 0.06485891342163086, 1.6388710737228394, -0.36524149775505066, -0.3587712049484253, 0.9327880144119263, 0.19021300971508026], [0.025552652776241302, 0.025552161037921906, 0.02556631714105606, 0.025556715205311775, 0.025558318942785263, 0.025549616664648056, 0.025562912225723267, 0.025562746450304985, 0.025565262883901596, 0.025550853461027145], [0.004893604665994644, 0.004892186261713505, 0.004895914811640978, 0.004892543889582157, 0.004893985111266375, 0.0049073221161961555, 0.004893142729997635, 0.0048935930244624615, 0.0048931497149169445, 0.004910826217383146], [0.06983538717031479, 0.008842358365654945, -0.10315163433551788, -0.01950705610215664, 0.006566979922354221, -0.47543075680732727, 0.09395948797464371, 0.03395254537463188, -0.26232582330703735, 0.8729899525642395], [-0.18234017491340637, 0.02506617270410061, 0.39638662338256836, 0.39043328166007996, 0.3973049521446228, -1.04948091506958, -0.6416453123092651, 0.08480850607156754, 0.09194150567054749, -1.0923004150390625], [0.005064285360276699, 0.005066038575023413, 0.005064961034804583, 0.0050638169050216675, 0.0050651440396904945, 0.005055146291851997, 0.005065148696303368, 0.005064853932708502, 0.005065022502094507, 0.00505316024646163], [-0.017247922718524933, -0.01724379137158394, -0.017246266826987267, -0.017246266826987267, -0.01724977418780327, -0.01727081835269928, -0.017238706350326538, -0.01724896766245365, -0.01724177412688732, -0.01726522482931614], [0.0003500595921650529, 0.00034901333856396377, 0.0003499234444461763, 0.0003499090962577611, 0.0003495076089166105, 0.00036177996662445366, 0.00034911223337985575, 0.00034897346631623805, 0.0003497837169561535, 0.0003635325701907277]]
B = [0.4043571949005127, -0.4064284861087799, 0.14156678318977356, 0.12869247794151306, -0.07702489197254181, -0.9789527058601379, 0.3035426437854767, 0.09887856990098953, -0.4320669174194336, -1.4954330921173096]


def quantities(pt, eta, phi, energy, jet_pt, jet_eta, jet_energy, charge, ptype, d0, d0err, dz, dzerr):
    pt, eta, phi, energy, charge, ptype, d0, d0err, dz, dzerr = [float(x) for x in pt], [float(x) for x in eta], [float(x) for x in phi], [float(x) for x in energy], [float(x) for x in charge], [float(x) for x in ptype], [float(x) for x in d0], [float(x) for x in d0err], [float(x) for x in dz], [float(x) for x in dzerr]
    jet_pt, jet_eta, jet_energy = float(jet_pt), float(jet_eta), float(jet_energy)
    n = len(pt)
    P = range(n)
    real = [i for i in P if pt[i] > 0]
    tot = sum(pt)
    z = [x / tot for x in pt]
    zs = sorted(z, reverse=True)
    dr = [math.hypot(eta[i], phi[i]) for i in P]

    def dist2(i, j):
        return (eta[i] - eta[j]) ** 2 + (phi[i] - phi[j]) ** 2

    def mass_of(k):
        E = sum(pt[i] * math.cosh(eta[i]) for i in range(k))
        px = sum(pt[i] * math.cos(phi[i]) for i in range(k))
        py = sum(pt[i] * math.sin(phi[i]) for i in range(k))
        pz = sum(pt[i] * math.sinh(eta[i]) for i in range(k))
        return math.sqrt(max(E * E - px * px - py * py - pz * pz, 0.0))

    def pair_mass(i, j):
        return math.sqrt(max(2 * pt[i] * pt[j] * (math.cosh(eta[i] - eta[j]) - math.cos(phi[i] - phi[j])), 0.0))

    def tau(k):
        axes = [(eta[max(P, key=lambda i: pt[i])], phi[max(P, key=lambda i: pt[i])])]
        for _ in range(1, k):
            far = max(P, key=lambda i: pt[i] * min(math.hypot(eta[i] - a, phi[i] - b) for a, b in axes))
            axes.append((eta[far], phi[far]))
        for _ in range(6):
            nearest = [min(range(k), key=lambda j: math.hypot(eta[i] - axes[j][0], phi[i] - axes[j][1])) for i in P]
            for j in range(k):
                w = sum(pt[i] for i in P if nearest[i] == j)
                if w > 0:
                    axes[j] = (sum(pt[i] * eta[i] for i in P if nearest[i] == j) / w, sum(pt[i] * phi[i] for i in P if nearest[i] == j) / w)
        return sum(z[i] * min(math.hypot(eta[i] - a, phi[i] - b) for a, b in axes) for i in P) / 0.8

    ta = sum(z[i] * eta[i] ** 2 for i in P)
    tb = sum(z[i] * eta[i] * phi[i] for i in P)
    tc = sum(z[i] * phi[i] ** 2 for i in P)
    disc = math.sqrt(max((ta - tc) ** 2 / 4 + tb ** 2, 0.0))
    lam1, lam2 = (ta + tc) / 2 + disc, max((ta + tc) / 2 - disc, 0.0)

    hard = sorted(P, key=lambda i: -pt[i])[:32]
    R = {(i, j): math.sqrt(dist2(i, j)) for i in hard for j in hard}
    e2 = sum(z[i] * z[j] * R[i, j] for i in hard for j in hard if i < j)
    e3 = sum(z[i] * z[j] * z[k] * R[i, j] * R[i, k] * R[j, k] for i in hard for j in hard for k in hard if i < j < k)

    _memo = {}

    def m4(v):
        return math.sqrt(max(v[3] ** 2 - v[0] ** 2 - v[1] ** 2 - v[2] ** 2, 0.0))

    def ecf(name):
        if 'ecf' not in _memo:
            h3, h4 = min(n, 32), min(n, 16)
            Rm = {(i, j): math.sqrt(dist2(i, j)) for i in range(h3) for j in range(h3)}
            o = dict(e2=0.0, e2b2=0.0, e3=0.0, e3b2=0.0, g31=0.0, g32=0.0, e4=0.0, g41=0.0, g42=0.0)
            for i in range(h3):
                for j in range(i + 1, h3):
                    w = z[i] * z[j]
                    o['e2'] += w * Rm[i, j]
                    o['e2b2'] += w * Rm[i, j] ** 2
                    for k in range(j + 1, h3):
                        a, b, c = Rm[i, j], Rm[i, k], Rm[j, k]
                        w3 = z[i] * z[j] * z[k]
                        s = sorted((a, b, c))
                        o['e3'] += w3 * a * b * c
                        o['e3b2'] += w3 * (a * b * c) ** 2
                        o['g31'] += w3 * s[0]
                        o['g32'] += w3 * s[0] * s[1]
            for i in range(h4):
                for j in range(i + 1, h4):
                    for k in range(j + 1, h4):
                        for l in range(k + 1, h4):
                            d = (Rm[i, j], Rm[i, k], Rm[i, l], Rm[j, k], Rm[j, l], Rm[k, l])
                            w4 = z[i] * z[j] * z[k] * z[l]
                            s = sorted(d)
                            o['e4'] += w4 * d[0] * d[1] * d[2] * d[3] * d[4] * d[5]
                            o['g41'] += w4 * s[0]
                            o['g42'] += w4 * s[0] * s[1]
            _memo['ecf'] = o
        return _memo['ecf'][name]

    def kaxes(k):
        if ('ax', k) not in _memo:
            hard = max(P, key=lambda i: pt[i])
            axes = [(eta[hard], phi[hard])]
            for _ in range(1, k):
                far = max(P, key=lambda i: min(math.hypot(eta[i] - a, phi[i] - b) for a, b in axes) * pt[i])
                axes.append((eta[far], phi[far]))
            for _ in range(6):
                near = [min(range(k), key=lambda j: math.hypot(eta[i] - axes[j][0], phi[i] - axes[j][1])) for i in P]
                for j in range(k):
                    w = sum(pt[i] for i in P if near[i] == j)
                    if w > 0:
                        axes[j] = (sum(pt[i] * eta[i] for i in P if near[i] == j) / w, sum(pt[i] * phi[i] for i in P if near[i] == j) / w)
            dist = [[math.hypot(eta[i] - a, phi[i] - b) for a, b in axes] for i in P]
            _memo['ax', k] = (axes, dist)
        return _memo['ax', k]

    def tau_n(k, beta=1):
        axes, dist = kaxes(k)
        return sum(z[i] * min(dist[i]) ** beta for i in P) / 0.8 ** beta

    def subjets(k):
        if ('sj', k) not in _memo:
            axes, dist = kaxes(k)
            S, V = [0.0] * k, [[0.0] * 4 for _ in range(k)]
            for i in real:
                j = min(range(k), key=lambda a: dist[i][a])
                S[j] += z[i]
                for c, x in enumerate((pt[i] * math.cos(phi[i]), pt[i] * math.sin(phi[i]), pt[i] * math.sinh(eta[i]), pt[i] * math.cosh(eta[i]))):
                    V[j][c] += x
            order = sorted(range(k), key=lambda j: -S[j])
            pairs = [(p, q) for p in range(k) for q in range(p + 1, k)]
            _memo['sj', k] = dict(z=[S[j] for j in order], mass=[m4(V[j]) for j in order],
                                  dr=[math.hypot(axes[order[p]][0] - axes[order[q]][0], axes[order[p]][1] - axes[order[q]][1]) for p, q in pairs],
                                  mpair=[m4([V[order[p]][c] + V[order[q]][c] for c in range(4)]) for p, q in pairs])
        return _memo['sj', k]

    def ca_dR2(u, v):
        ya, pa = 0.5 * math.log(max(u[3] + u[2], 1e-300) / max(u[3] - u[2], 1e-300)), math.atan2(u[1], u[0])
        yb, pb = 0.5 * math.log(max(v[3] + v[2], 1e-300) / max(v[3] - v[2], 1e-300)), math.atan2(v[1], v[0])
        return (ya - yb) ** 2 + ((pa - pb + math.pi) % (2 * math.pi) - math.pi) ** 2

    def ca_tree():
        if 'ca' not in _memo:
            node = {i: (pt[i] * math.cos(phi[i]), pt[i] * math.sin(phi[i]), pt[i] * math.sinh(eta[i]), pt[i] * math.cosh(eta[i])) for i in range(min(n, 128)) if pt[i] > 0}
            live, kids, new = list(node), {}, 1000
            yp = {i: (0.5 * math.log(max(v[3] + v[2], 1e-300) / max(v[3] - v[2], 1e-300)), math.atan2(v[1], v[0])) for i, v in node.items()}
            while len(live) > 1:                                  # Cambridge/Aachen: merge the closest pair
                best = None
                for a in range(len(live)):
                    ya, pa = yp[live[a]]
                    for b in range(a + 1, len(live)):
                        yb, pb = yp[live[b]]
                        d = (ya - yb) ** 2 + ((pa - pb + math.pi) % (2 * math.pi) - math.pi) ** 2
                        if best is None or d < best[0]:
                            best = (d, a, b)
                _, a, b = best
                v = tuple(x + y for x, y in zip(node[live[a]], node[live[b]]))
                node[new] = v
                yp[new] = (0.5 * math.log(max(v[3] + v[2], 1e-300) / max(v[3] - v[2], 1e-300)), math.atan2(v[1], v[0]))
                kids[new] = (live[a], live[b])
                live[a] = new
                live.pop(b)
                new += 1
            _memo['ca'] = (node, kids, live[0])
        return _memo['ca']

    def softdrop(what):
        if 'sd' not in _memo:
            node, kids, cur = ca_tree()
            removed, res = 0, (0.0, 0.0, 0.0)
            while cur in kids:                                    # soft drop, beta = 0, z_cut = 0.1
                c1, c2 = kids[cur]
                p1, p2 = math.hypot(node[c1][0], node[c1][1]), math.hypot(node[c2][0], node[c2][1])
                zz = min(p1, p2) / max(p1 + p2, 1e-300)
                if zz > 0.1:
                    res = (m4(node[cur]), zz, math.sqrt(ca_dR2(node[c1], node[c2])))
                    break
                cur = c1 if p1 >= p2 else c2
                removed += 1
            _memo['sd'] = dict(mass=res[0], zg=res[1], rg=res[2], removed=float(removed))
        return _memo['sd'][what]

    def softp(s, what):
        j = len(real) - s
        if j < 15:
            return 0.0
        return dict(pt=pt[j], z=z[j], abseta=abs(eta[j]), absphi=abs(phi[j]), dr=dr[j], dr0=math.sqrt(dist2(0, j)))[what]

    def ncum(f):
        c = 0.0
        for k, i in enumerate(real):
            c += z[i]
            if c >= f:
                return float(k + 1)
        return float(len(real))

    LNEPS = math.log(1e-8)

    def lg(x):
        return math.log(max(x, 1e-8))

    def pairv(i, j, f):
        if pt[i] <= 0 or pt[j] <= 0:
            return LNEPS
        de, dp = eta[i] - eta[j], phi[i] - phi[j]
        d, pm = math.hypot(de, dp), min(pt[i], pt[j])
        if f == 'lndelta':
            return lg(d)
        if f == 'lnkt':
            return lg(pm * d)
        if f == 'lnz':
            return lg(pm / max(pt[i] + pt[j], 1e-8))
        return lg(2 * pt[i] * pt[j] * (math.cosh(de) - math.cos(dp)))

    def pairf(i, j, f):
        return pairv(i, j, f)

    def pfeat(i, f):
        if pt[i] <= 0:
            return LNEPS if f.startswith('ln') else 0.0
        if f == 'lnpt':
            return math.log(pt[i])
        if f == 'lne':
            return math.log(energy[i])
        if f == 'lnptrel':
            return math.log(pt[i] / jet_pt)
        if f == 'lnerel':
            return math.log(energy[i] / jet_energy)
        if f == 'dr':
            return math.hypot(eta[i], phi[i])
        if f in ('eta', 'phi'):
            return eta[i] if f == 'eta' else phi[i]
        if f == 'charge':
            return charge[i]
        if f.startswith('is'):
            return 1.0 if ptype[i] == dict(ischhad=1, isnhad=2, isphoton=3, iselectron=4, ismuon=5)[f] else 0.0
        return dict(td0=lambda: math.tanh(d0[i]), d0err=lambda: min(max(d0err[i], 0.0), 1.0), tdz=lambda: math.tanh(dz[i]),
                    dzerr=lambda: min(max(dzerr[i], 0.0), 1.0))[f]()

    def pairstats():
        if 'pairs' not in _memo:
            s = {f: 0.0 for f in ('lndelta', 'lnkt', 'lnz', 'lnm2')}
            w_sum, mx, cnt = 0.0, {'lnkt': LNEPS, 'lnm2': LNEPS}, {1: 0, 3: 0, 10: 0, 30: 0}
            for a in range(len(real)):
                for b in range(a + 1, len(real)):
                    i, j = real[a], real[b]
                    w = z[i] * z[j]
                    w_sum += w
                    for f in s:
                        v = pairv(i, j, f)
                        s[f] += w * v
                        if f in mx and v > mx[f]:
                            mx[f] = v
                    kt = min(pt[i], pt[j]) * math.hypot(eta[i] - eta[j], phi[i] - phi[j])
                    for c in cnt:
                        if kt > c:
                            cnt[c] += 1
            _memo['pairs'] = dict(mean={f: v / max(w_sum, 1e-300) for f, v in s.items()}, max=mx, count=cnt)
        return _memo['pairs']

    def pairsum(f):
        return pairstats()['mean'][f]

    def pairmax(f):
        return pairstats()['max'][f]

    def paircount(c):
        return float(pairstats()['count'][c])

    def lund(k, what):
        if 'lund' not in _memo:
            node, kids, cur = ca_tree()
            seq = []
            while cur in kids:
                c1, c2 = kids[cur]
                p1, p2 = math.hypot(node[c1][0], node[c1][1]), math.hypot(node[c2][0], node[c2][1])
                d, pm = math.sqrt(ca_dR2(node[c1], node[c2])), min(p1, p2)
                seq.append((lg(d), lg(pm * d), lg(pm / max(p1 + p2, 1e-8)), pm * d))
                cur = c1 if p1 >= p2 else c2
            best = None
            for s in seq:
                if best is None or s[1] > best[1]:
                    best = s
            _memo['lund'] = dict(seq=seq, maxkt=best[1] if best else LNEPS, maxdelta=best[0] if best else LNEPS, n=float(len(seq)),
                                 n1=float(sum(1 for s in seq if s[3] > 1)), n5=float(sum(1 for s in seq if s[3] > 5)))
        L = _memo['lund']
        if k == 0:
            return L[what]
        if len(L['seq']) < k:
            return LNEPS
        return L['seq'][k - 1][('lndelta', 'lnkt', 'lnz').index(what)]

    def ecfb(name, beta):
        if ('ecfb', beta) not in _memo:
            h3, h4 = min(n, 32), min(n, 16)
            Rb = {(i, j): math.sqrt(dist2(i, j)) ** beta for i in range(h3) for j in range(h3)}
            o = dict(e2=0.0, e3=0.0, g31=0.0, g32=0.0, e4=0.0, g41=0.0, g42=0.0, g43=0.0)
            for i in range(h3):
                for j in range(i + 1, h3):
                    o['e2'] += z[i] * z[j] * Rb[i, j]
                    for k in range(j + 1, h3):
                        a, b, c = Rb[i, j], Rb[i, k], Rb[j, k]
                        w3 = z[i] * z[j] * z[k]
                        s = sorted((a, b, c))
                        o['e3'] += w3 * a * b * c
                        o['g31'] += w3 * s[0]
                        o['g32'] += w3 * s[0] * s[1]
            for i in range(h4):
                for j in range(i + 1, h4):
                    for k in range(j + 1, h4):
                        for l in range(k + 1, h4):
                            d = (Rb[i, j], Rb[i, k], Rb[i, l], Rb[j, k], Rb[j, l], Rb[k, l])
                            w4 = z[i] * z[j] * z[k] * z[l]
                            s = sorted(d)
                            o['e4'] += w4 * d[0] * d[1] * d[2] * d[3] * d[4] * d[5]
                            o['g41'] += w4 * s[0]
                            o['g42'] += w4 * s[0] * s[1]
                            o['g43'] += w4 * s[0] * s[1] * s[2]
            _memo['ecfb', beta] = o
        return _memo['ecfb', beta][name]

    ch = [i for i in real if charge[i] != 0]
    # tracks with a measured uncertainty (a few charged particles have sigma = 0: left out of every significance)
    tk = dict(d0=[i for i in ch if d0err[i] > 0], dz=[i for i in ch if dzerr[i] > 0])
    tk['3d'] = [i for i in tk['d0'] if dzerr[i] > 0]
    sd0 = {i: d0[i] / d0err[i] for i in tk['d0']}
    sdz = {i: dz[i] / dzerr[i] for i in tk['dz']}

    def sig(w, i):
        if i not in tk[w]:
            return 0.0
        return sd0[i] if w == 'd0' else sdz[i] if w == 'dz' else math.sqrt(sd0[i] ** 2 + sdz[i] ** 2)

    def sip(w, r):
        s = sorted(tk[w], key=lambda i: -abs(sig(w, i)))
        return sig(w, s[r - 1]) if len(s) >= r else 0.0

    def nsig(w, c):
        return float(sum(1 for i in tk[w] if abs(sig(w, i)) > c))

    def leadtrack(w):
        return sig(w, ch[0]) if ch else 0.0

    def lepton(what):
        lep = [i for i in real if ptype[i] in (4, 5)]
        if not lep:
            return 0.0
        i = lep[0]
        if what == 'z':
            return z[i]
        if what == 'dr':
            return dr[i]
        if what == 'ptrel':
            return pt[i] * dr[i]
        if what == 'sd0':
            return sig('d0', i)
        return sum(pt[j] for j in real if j != i and math.hypot(eta[j] - eta[i], phi[j] - phi[i]) < 0.2) / max(pt[i], 1e-9)

    def set_mass(ids):
        E = sum(pt[i] * math.cosh(eta[i]) for i in ids)
        px = sum(pt[i] * math.cos(phi[i]) for i in ids)
        py = sum(pt[i] * math.sin(phi[i]) for i in ids)
        pz = sum(pt[i] * math.sinh(eta[i]) for i in ids)
        return math.sqrt(max(E * E - px * px - py * py - pz * pz, 0.0))

    def displaced(c, what):
        ids = [i for i in tk['d0'] if abs(sd0[i]) > c]
        return sum(z[i] for i in ids) if what == 'z' else set_mass(ids)

    def subset_mass(what):
        if what == 'charged':
            return set_mass(ch)
        if what == 'neutral':
            return set_mass([i for i in real if charge[i] == 0])
        if what == 'photon2':
            ph = [i for i in real if ptype[i] == 3]
            return set_mass(ph[:2]) if len(ph) >= 2 else 0.0
        return set_mass(ch[:2])

    # ---- more quantities of the 'full' network (pq(id)); ParT's view of the track uncertainties: clipped to [0, 1];
    # tracks with sigma = 0 have no significance; four-vectors with the particles' energies (eta + the jet's eta)
    SQ2 = math.sqrt(2.0)
    c0 = {i: d0[i] / min(d0err[i], 1.0) for i in tk['d0']}
    c3 = {i: math.sqrt(c0[i] ** 2 + (dz[i] / min(dzerr[i], 1.0)) ** 2) for i in tk['3d']}
    lnp = {i: -math.log(max(math.erfc(abs(c0[i]) / SQ2), 1e-300)) for i in tk['d0'] if c0[i] > 0}
    p4e = {i: (pt[i] * math.cos(phi[i]), pt[i] * math.sin(phi[i]), pt[i] * math.sinh(eta[i] + jet_eta), energy[i]) for i in real}

    def emass(ids, w=None):
        v = [0.0, 0.0, 0.0, 0.0]
        for i in ids:
            f = 1.0 if w is None else w[i]
            for c in range(4):
                v[c] += f * p4e[i][c]
        return math.sqrt(max(v[3] * v[3] - v[0] * v[0] - v[1] * v[1] - v[2] * v[2], 0.0))

    def ranked(groups):
        return sorted(groups, key=lambda g: -sum(pt[i] for i in g))

    def cen(g):
        w = max(sum(pt[i] for i in g), 1e-9)
        return (sum(pt[i] * eta[i] for i in g) / w, sum(pt[i] * phi[i] for i in g) / w)

    def flav(g, pre, out):
        tr = [i for i in g if i in c0]
        dsp = [i for i in tr if c0[i] > 3]
        s = sorted((c0[i] for i in tr), reverse=True)
        for k in (1, 2, 3):
            out[pre + 'sd0_' + str(k)] = s[k - 1] if len(s) >= k else 0.0
        out[pre + 'n_disp3'] = float(len(dsp))
        out[pre + 'mass_disp3'] = emass(dsp)
        out[pre + 'z_disp3'] = sum(z[i] for i in dsp)
        out[pre + 'jp'] = sum(lnp.get(i, 0.0) for i in tr)
        out[pre + 'n_lep'] = float(sum(1 for i in g if ptype[i] in (4, 5)))
        out[pre + 'charge'] = sum(charge[i] * math.sqrt(pt[i]) for i in g) / math.sqrt(max(sum(pt[i] for i in g), 1e-9))

    def subfeats(pre, groups, keep, out):
        gs = ranked(groups)[:keep]
        gs += [[]] * (keep - len(gs))
        cs = []
        for r, g in enumerate(gs):
            q = pre + '_' + str(r + 1) + '_'
            out[q + 'z'] = sum(z[i] for i in g)
            out[q + 'mass'] = emass(g)
            flav(g, q, out)
            cs.append(cen(g) if g else None)
        for a in range(keep):
            for b in range(a + 1, keep):
                out[pre + '_dr' + str(a + 1) + str(b + 1)] = math.hypot(cs[a][0] - cs[b][0], cs[a][1] - cs[b][1]) if cs[a] and cs[b] else 0.0
        for v in ('sd0_1', 'sd0_2', 'n_disp3', 'mass_disp3', 'jp'):
            out[pre + '_min12_' + v] = min(out[pre + '_1_' + v], out[pre + '_2_' + v])

    def genkt(ids, p, R, nstop=0, snap=0):
        # generalized kT (p = 1: kT, p = -1: anti-kT), E-scheme, rapidity-azimuth distances, massless four-vectors;
        # nstop > 0: exclusive (pairwise merges only) down to nstop pseudojets, snap: the partition when snap remain;
        # nstop = 0: inclusive (a pseudojet whose beam distance is the smallest becomes a jet; ties merge)
        key = ('gk', tuple(ids), p, R, nstop, snap)
        if key in _memo:
            return _memo[key]
        R2 = R * R
        node = {i: (pt[i] * math.cos(phi[i]), pt[i] * math.sin(phi[i]), pt[i] * math.sinh(eta[i]), pt[i] * math.cosh(eta[i])) for i in ids}

        def yk(v):
            t = math.hypot(v[0], v[1])
            return (0.5 * math.log(max(v[3] + v[2], 1e-300) / max(v[3] - v[2], 1e-300)), math.atan2(v[1], v[0]), t * t if p == 1 else 1.0 / max(t * t, 1e-300))
        live, mem, yp, jets, dms, new = list(ids), {i: [i] for i in ids}, {i: yk(node[i]) for i in ids}, [], {}, 1000
        part = [[i] for i in ids]
        while live and len(live) > nstop:
            if snap and len(live) == snap:
                part = [mem[x] for x in live]
            best = None
            for a in range(len(live)):
                ya, pa, ka = yp[live[a]]
                for b in range(a + 1, len(live)):
                    yb, pb, kb = yp[live[b]]
                    d = min(ka, kb) * ((ya - yb) ** 2 + ((pa - pb + math.pi) % (2 * math.pi) - math.pi) ** 2) / R2
                    if best is None or d < best[0]:
                        best = (d, a, b)
            if not nstop:
                bb = None
                for a in range(len(live)):
                    if bb is None or yp[live[a]][2] < bb[0]:
                        bb = (yp[live[a]][2], a)
                if best is None or bb[0] < best[0]:
                    jets.append(mem[live[bb[1]]])
                    live.pop(bb[1])
                    continue
            d, a, b = best
            if nstop:
                dms[len(live) - 1] = d
            v = tuple(x + y for x, y in zip(node[live[a]], node[live[b]]))
            node[new], yp[new], mem[new] = v, yk(v), mem[live[a]] + mem[live[b]]
            live[a] = new
            live.pop(b)
            new += 1
        _memo[key] = (part if snap else jets if not nstop else [mem[x] for x in live], dms)
        return _memo[key]

    def kgroups(k):
        axes, dist = kaxes(k)
        near = {i: min(range(k), key=lambda a: dist[i][a]) for i in real}
        S = [0.0] * k
        for i in real:
            S[near[i]] += z[i]
        order = sorted(range(k), key=lambda j: -S[j])
        return [[i for i in real if near[i] == j] for j in order]

    def g_kt2():
        out = {}
        subfeats('kt2', genkt(real, 1, 1.0, 1, 2)[0], 2, out)
        return out

    def g_ktd():
        dms = genkt(real, 1, 1.0, 1, 2)[1]
        return {'ktd_ln_d' + str(m) + str(m + 1): math.log(max(dms[m], 1e-12) / (tot * tot)) if m in dms else 0.0 for m in (1, 2, 3)}

    def g_ak02():
        jets = [g for g in genkt(real, -1, 0.2)[0] if sum(pt[i] for i in g) > 10.0]
        out = {'ak02_n': float(len(jets))}
        subfeats('ak02', jets, 3, out)
        return out

    def g_sv():
        jets = genkt([i for i in tk['d0'] if c0[i] > 3], -1, 0.1)[0]
        out = {'sv_n': float(len(jets))}
        gs = ranked(jets)[:2]
        gs += [[]] * (2 - len(gs))
        for r, g in enumerate(gs):
            q = 'sv_' + str(r + 1) + '_'
            out[q + 'n'] = float(len(g))
            out[q + 'mass'] = emass(g)
            out[q + 'z'] = sum(z[i] for i in g)
            out[q + 'dr'] = math.hypot(*cen(g)) if g else 0.0
            out[q + 'sd0_sum'] = sum(c0[i] for i in g)
        return out

    def g_sdb():
        E, out = (-math.inf, -3.0, -1.0, 1.0, 3.0, 10.0, math.inf), {}
        for b in range(6):
            ids = [i for i in tk['d0'] if E[b] < c0[i] <= E[b + 1]]
            out['sdb_' + str(b) + '_z'] = sum(z[i] for i in ids)
            out['sdb_' + str(b) + '_n'] = float(len(ids))
        out['sdb_jp_all'] = sum(lnp.values())
        out['sdb_jp_top3'] = sum(sorted(lnp.values(), reverse=True)[:3])
        return out

    def g_pz():
        LD, LK = (-9.0, -3.0, -2.0, -1.0, 9.0), (-9.0, 0.0, 1.0, 2.0, 3.0, 9.0)
        out = {'pz_lnd' + str(b): 0.0 for b in range(4)}
        out.update({'pz_lnkt' + str(b): 0.0 for b in range(5)})
        top = [i for i in real if i < 40]
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                i, j = top[a], top[b]
                d = math.hypot(eta[i] - eta[j], phi[i] - phi[j])
                w = z[i] * z[j]
                ld, lk = math.log(max(d, 1e-6)), math.log(max(min(pt[i], pt[j]) * d, 1e-6))
                for k in range(4):
                    if LD[k] < ld <= LD[k + 1]:
                        out['pz_lnd' + str(k)] += w
                for k in range(5):
                    if LK[k] < lk <= LK[k + 1]:
                        out['pz_lnkt' + str(k)] += w
        return out

    def g_dc():
        # prongs: reverse the C/A tree; a node whose branches are closer than 0.1 is a prong; else the branch with less
        # than 10 % of the jet's pT is dropped, or (both above) both branches are declustered further
        node, kids, root = ca_tree()
        ptn = lambda k: math.hypot(node[k][0], node[k][1])
        ptj = ptn(root)
        prongs, splits, stack = [], [], [root]
        while stack:
            k = stack.pop()
            while True:
                if k not in kids:
                    prongs.append(k)
                    break
                a, b = kids[k]
                pa, pb = ptn(a), ptn(b)
                d = math.sqrt(ca_dR2(node[a], node[b]))
                if d <= 0.1:
                    prongs.append(k)
                    break
                hard, soft = (a, b) if pa >= pb else (b, a)
                zz = min(pa, pb) / max(ptj, 1e-12)
                if zz >= 0.1:
                    splits.append((ptn(k), zz, d, min(pa, pb) * d, m4(node[k])))
                    stack.append(soft)
                k = hard

        def leaves(k):
            return leaves(kids[k][0]) + leaves(kids[k][1]) if k in kids else [k]
        groups = [leaves(k) for k in prongs]
        out = {'dc_n': float(len(groups))}
        gs = ranked(groups)[:4]
        gs += [[]] * (4 - len(gs))
        for r, g in enumerate(gs):
            q = 'dc_' + str(r + 1) + '_'
            out[q + 'z'] = sum(z[i] for i in g)
            out[q + 'mass'] = emass(g)
            flav(g, q, out)
        tags = [out['dc_' + str(r + 1) + '_sd0_2'] for r in range(4)]
        st = sorted(tags, reverse=True)
        out['dc_ntag'], out['dc_tag_max'], out['dc_tag_2nd'] = float(sum(1 for t in tags if t > 3)), st[0], st[1]
        md = [out['dc_' + str(r + 1) + '_mass_disp3'] for r in range(4)]
        sm = sorted(md, reverse=True)
        out['dc_mass_disp_max'], out['dc_mass_disp_2nd'], out['dc_mass_disp_sum'] = sm[0], sm[1], sum(md)
        pm = [emass(gs[a] + gs[b]) for a in range(4) for b in range(a + 1, 4) if gs[a] and gs[b]]
        out['dc_pair_mass_min'], out['dc_pair_mass_max'] = (min(pm), max(pm)) if pm else (0.0, 0.0)
        sp = sorted(splits, key=lambda t: -t[0])
        for r in (0, 1):
            t = sp[r] if len(sp) > r else (0.0, 0.0, 0.0, 0.0, 0.0)
            for c, nm in enumerate(('z', 'dr', 'kt', 'mass')):
                out['dc_split' + str(r + 1) + '_' + nm] = t[c + 1]
        return out

    def g_sjf():
        out = {}
        for k in (2, 3, 4):
            n2 = 0
            for r, g in enumerate(kgroups(k)):
                q = 'sjf_' + str(k) + '_' + str(r + 1) + '_'
                tr = [i for i in g if i in c0]
                d3 = [i for i in tr if abs(c0[i]) > 3]
                out[q + 'n_d3'] = float(len(d3))
                out[q + 'n_d5'] = float(sum(1 for i in tr if abs(c0[i]) > 5))
                out[q + 'mass_d3'] = emass(d3)
                out[q + 'z_d3'] = sum(z[i] for i in d3)
                out[q + 'maxsd0'] = max([abs(c0[i]) for i in tr] or [0.0])
                out[q + 'max3d'] = max([c3[i] for i in g if i in c3] or [0.0])
                n2 += len(d3) >= 2
            out['sjf_' + str(k) + '_n2disp'] = float(n2)
        return out

    def g_jd():
        tr, out = tk['d0'], {}
        s = sorted(tr, key=lambda i: -abs(c0[i]))
        for r in (4, 5, 6):
            out['jd_sd0_' + str(r)] = c0[s[r - 1]] if len(s) >= r else 0.0
        s3 = sorted((c3[i] for i in tk['3d']), reverse=True)
        for r in (4, 5, 6):
            out['jd_3d_' + str(r)] = s3[r - 1] if len(s3) >= r else 0.0
        a = sorted((abs(c0[i]) for i in tr), reverse=True)
        out['jd_sum_abs_sd0_top3'], out['jd_sum_abs_sd0_top5'] = sum(a[:3]), sum(a[:5])
        d3 = [i for i in tr if abs(c0[i]) > 3]
        mx = max([abs(c0[i]) for i in d3] or [1.0])
        out['jd_mass_d3_sigw'] = emass(d3, {i: abs(c0[i]) / mx for i in d3})
        zd = sum(z[i] for i in d3)
        out['jd_mass_d3_over_z'] = emass(d3) / zd if zd > 0 else 0.0
        out['jd_n_d3_pt1'] = float(sum(1 for i in d3 if pt[i] > 1.0))
        return out

    def g_mres():
        out = {'mres_mass_e': emass(real)}
        for k in (2, 3):
            for r, g in enumerate(kgroups(k)):
                out['mres_sj' + str(k) + '_mass' + str(r + 1) + '_e'] = emass(g)
        node, kids, root = ca_tree()
        ptn = lambda k: math.hypot(node[k][0], node[k][1])
        for b, zc, t in ((0, 0.05, 'b0z005'), (0, 0.2, 'b0z02'), (1, 0.1, 'b1z01'), (2, 0.1, 'b2z01')):
            cur, res = root, (0.0, 0.0, 0.0)
            while cur in kids:
                c1, c2 = kids[cur]
                p1, p2 = ptn(c1), ptn(c2)
                zz, d = min(p1, p2) / max(p1 + p2, 1e-300), math.sqrt(ca_dR2(node[c1], node[c2]))
                if zz > zc * (d / 0.8) ** b:
                    res = (m4(node[cur]), zz, d)
                    break
                cur = c1 if p1 >= p2 else c2
            out['mres_sd_mass_' + t], out['mres_sd_zg_' + t], out['mres_sd_rg_' + t] = res
        cur, pm = root, (0.0, 0.0)
        while cur in kids:                                # the two prongs of the soft-drop splitting (beta = 0, z_cut = 0.1)
            c1, c2 = kids[cur]
            p1, p2 = ptn(c1), ptn(c2)
            if min(p1, p2) / max(p1 + p2, 1e-300) > 0.1:
                pm = (m4(node[c1]), m4(node[c2])) if p1 >= p2 else (m4(node[c2]), m4(node[c1]))
                break
            cur = c1 if p1 >= p2 else c2
        out['mres_sd_prong_mass1'], out['mres_sd_prong_mass2'] = pm
        rv = node[root]                                   # pruning on the C/A tree: z_cut = 0.1, D_cut = m / pT of the jet
        dcut = m4(rv) / max(math.hypot(rv[0], rv[1]), 1e-300)
        pv = {i: node[i] for i in node if i not in kids}
        for k in sorted(kids):
            a, b = kids[k]
            va, vb = pv[a], pv[b]
            pa, pb = math.hypot(va[0], va[1]), math.hypot(vb[0], vb[1])
            v = tuple(x + y for x, y in zip(va, vb))
            zz = min(pa, pb) / max(math.hypot(v[0], v[1]), 1e-300)
            pv[k] = (va if pa >= pb else vb) if zz < 0.1 and math.sqrt(ca_dR2(va, vb)) > dcut else v
        out['mres_pruned_mass'] = m4(pv[root])
        keep = [i for g in genkt(real, 1, 0.2)[0] if sum(pt[j] for j in g) > 0.03 * tot for i in g]   # trimming
        out['mres_trimmed_mass'] = emass(keep)
        return out

    def g_sjq():
        out = {}
        for k in (2, 3):
            Q = {}
            for r, g in enumerate(kgroups(k)):
                w = max(sum(pt[i] for i in g), 1e-9)
                for ka, t in ((0.3, '03'), (0.5, '05'), (1.0, '1')):
                    Q[r, t] = sum(charge[i] * pt[i] ** ka for i in g) / w ** ka
                    out['sjq_' + str(k) + '_' + str(r + 1) + '_k' + t] = Q[r, t]
                out['sjq_' + str(k) + '_' + str(r + 1) + '_nch'] = float(sum(1 for i in g if charge[i] != 0))
            for t in ('03', '05', '1'):
                out['sjq_' + str(k) + '_sumabs_k' + t] = abs(Q[0, t] + Q[1, t])
                out['sjq_' + str(k) + '_prod_k' + t] = Q[0, t] * Q[1, t]
        return out

    def g_nca():
        node, kids, root = ca_tree()
        out = {}
        for c in (2, 5, 10, 20):
            cnt, stack = 0, [root]
            while stack:
                k = stack.pop()
                if k in kids:
                    a, b = kids[k]
                    if min(math.hypot(node[a][0], node[a][1]), math.hypot(node[b][0], node[b][1])) * math.sqrt(ca_dR2(node[a], node[b])) > c:
                        stack += [a, b]
                        continue
                cnt += 1
            out['nca_kt_above_' + str(c)] = float(cnt)
        mp = sorted(subjets(4)['mpair'], reverse=True)
        mj = max(mass_of(n), 1e-9)
        out['nca_sj4_pair_mass_2nd'], out['nca_sj4_pairmax_over_mass'], out['nca_sj4_pair2nd_over_mass'] = mp[1], mp[0] / mj, mp[1] / mj
        return out

    def g_lepsj():
        out, lep = {}, [i for i in real if ptype[i] in (4, 5)]
        for k in (2, 3):
            q = 'lepsj_' + str(k) + '_'
            if not lep:
                out[q + 'mass'] = out[q + 'dr'] = out[q + 'n_d3'] = out[q + 'maxsd0'] = 0.0
                continue
            l = lep[0]
            axes, dist = kaxes(k)
            j = min(range(k), key=lambda a: dist[l][a])
            g = [i for i in real if i != l and min(range(k), key=lambda a: dist[i][a]) == j]
            tr = [i for i in g if i in c0]
            out[q + 'mass'] = emass(g + [l])
            out[q + 'dr'] = dist[l][j]
            out[q + 'n_d3'] = float(sum(1 for i in tr if abs(c0[i]) > 3))
            out[q + 'maxsd0'] = max([abs(c0[i]) for i in tr] or [0.0])
        return out

    def pq(key):
        g = key.split('_')[0]
        if ('pq', g) not in _memo:
            _memo['pq', g] = dict(kt2=g_kt2, ktd=g_ktd, ak02=g_ak02, sv=g_sv, sdb=g_sdb, pz=g_pz, dc=g_dc, sjf=g_sjf, jd=g_jd,
                                  mres=g_mres, sjq=g_sjq, nca=g_nca, lepsj=g_lepsj)[g]()
        return _memo['pq', g][key]

    return SimpleNamespace(
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3_b05=ecfb('e4', 0.5) * ecfb('e2', 0.5) / max(ecfb('e3', 0.5) ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3_b05=ecfb('e4', 0.5) * ecfb('e2', 0.5) ** 3 / max(ecfb('e3', 0.5) ** 3, 1e-30),
        D3_b2=ecfb('e4', 2) * ecfb('e2', 2) ** 3 / max(ecfb('e3', 2) ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M2_b2=ecfb('g31', 2) / max(ecfb('e2', 2), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        M3_b2=ecfb('g41', 2) / max(ecfb('g31', 2), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N2_b05=ecfb('g32', 0.5) / max(ecfb('e2', 0.5) ** 2, 1e-30),
        N2_b2=ecfb('g32', 2) / max(ecfb('e2', 2) ** 2, 1e-30),
        N3_b05=ecfb('g42', 0.5) / max(ecfb('g31', 0.5) ** 2, 1e-30),
        ak02_1_n_lep=pq('ak02_1_n_lep'),
        ak02_1_z=pq('ak02_1_z'),
        ak02_2_n_lep=pq('ak02_2_n_lep'),
        ak02_2_z=pq('ak02_2_z'),
        ak02_3_z=pq('ak02_3_z'),
        ak02_dr13=pq('ak02_dr13'),
        ak02_dr23=pq('ak02_dr23'),
        ak02_min12_jp=pq('ak02_min12_jp'),
        ak02_n=pq('ak02_n'),
        dc_1_n_lep=pq('dc_1_n_lep'),
        dc_1_z=pq('dc_1_z'),
        dc_1_z_disp3=pq('dc_1_z_disp3'),
        dc_2_jp=pq('dc_2_jp'),
        dc_2_n_lep=pq('dc_2_n_lep'),
        dc_3_z=pq('dc_3_z'),
        dc_ntag=pq('dc_ntag'),
        dc_pair_mass_min=pq('dc_pair_mass_min'),
        dc_split2_dr=pq('dc_split2_dr'),
        dc_tag_2nd=pq('dc_tag_2nd'),
        dr02=math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        dr_19=pfeat(19, 'dr'),
        dr_29=pfeat(29, 'dr'),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        e3=ecf('e3'),
        e3_b05=ecfb('e3', 0.5),
        e3_b2=ecfb('e3', 2),
        e4=ecf('e4'),
        e4_b05=ecfb('e4', 0.5),
        ecf_g31=ecfb('g31', 1),
        ecf_g41=ecfb('g41', 1),
        ecf_g42=ecfb('g42', 1),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        ischhad_0=pfeat(0, 'ischhad'),
        jd_3d_4=pq('jd_3d_4'),
        jd_3d_5=pq('jd_3d_5'),
        jd_3d_6=pq('jd_3d_6'),
        jd_sum_abs_sd0_top3=pq('jd_sum_abs_sd0_top3'),
        jd_sum_abs_sd0_top5=pq('jd_sum_abs_sd0_top5'),
        jet_abs_eta=abs(jet_eta),
        jet_charge_k03=sum(charge[i] * z[i] ** 0.3 for i in real),
        jet_charge_k05=sum(charge[i] * z[i] ** 0.5 for i in real),
        jet_e=jet_energy,
        kt2_1_n_disp3=pq('kt2_1_n_disp3'),
        kt2_2_n_lep=pq('kt2_2_n_lep'),
        kt2_min12_jp=pq('kt2_min12_jp'),
        ktd_ln_d34=pq('ktd_ln_d34'),
        lam1=lam1,
        lep_dr=lepton('dr'),
        lep_iso=lepton('iso'),
        lep_ptrel=lepton('ptrel'),
        lep_z=lepton('z'),
        lepsj_2_dr=pq('lepsj_2_dr'),
        lepsj_2_maxsd0=pq('lepsj_2_maxsd0'),
        lepsj_2_n_d3=pq('lepsj_2_n_d3'),
        lepsj_3_dr=pq('lepsj_3_dr'),
        lepsj_3_maxsd0=pq('lepsj_3_maxsd0'),
        lepsj_3_n_d3=pq('lepsj_3_n_d3'),
        lne_0=pfeat(0, 'lne'),
        lne_1=pfeat(1, 'lne'),
        lne_16=pfeat(16, 'lne'),
        lne_4=pfeat(4, 'lne'),
        lne_8=pfeat(8, 'lne'),
        lnptrel_25=pfeat(25, 'lnptrel'),
        lund3_lndelta=lund(3, 'lndelta'),
        lund_max_lndelta=lund(0, 'maxdelta'),
        lund_max_lnkt=lund(0, 'maxkt'),
        mass=mass_of(n),
        mass_2charged=subset_mass('charged2'),
        mass_2photon=subset_mass('photon2'),
        mass_charged=subset_mass('charged'),
        mass_displaced3=displaced(3, 'mass'),
        mass_displaced5=displaced(5, 'mass'),
        mass_neutral=subset_mass('neutral'),
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        max_abs_d0=max([abs(d0[i]) for i in real if charge[i] != 0] or [0.0]),
        max_abs_dz=max([abs(dz[i]) for i in real if charge[i] != 0] or [0.0]),
        max_dr=max(dr[i] for i in real),
        mres_pruned_mass=pq('mres_pruned_mass'),
        mres_sd_mass_b0z005=pq('mres_sd_mass_b0z005'),
        mres_sd_mass_b0z02=pq('mres_sd_mass_b0z02'),
        mres_sd_mass_b1z01=pq('mres_sd_mass_b1z01'),
        mres_sd_mass_b2z01=pq('mres_sd_mass_b2z01'),
        mres_sd_prong_mass1=pq('mres_sd_prong_mass1'),
        mres_sd_prong_mass2=pq('mres_sd_prong_mass2'),
        mres_sd_rg_b2z01=pq('mres_sd_rg_b2z01'),
        n_charged_had=sum(1 for i in real if ptype[i] == 1),
        n_charged_pt_above_1=sum(1 for i in real if charge[i] != 0 and pt[i] > 1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        n_for_90pct=ncum(0.9),
        n_lepton=sum(1 for i in real if ptype[i] in (4, 5)),
        n_lund=lund(0, 'n'),
        n_lund_kt_above_1=lund(0, 'n1'),
        n_lund_kt_above_5=lund(0, 'n5'),
        n_muon=sum(1 for i in real if ptype[i] == 5),
        n_neutral=sum(1 for i in real if charge[i] == 0),
        n_neutral_had=sum(1 for i in real if ptype[i] == 2),
        n_pairs_kt_above_1=paircount(1),
        n_pairs_kt_above_10=paircount(10),
        n_pairs_kt_above_3=paircount(3),
        n_particles=len(real),
        n_photon=sum(1 for i in real if ptype[i] == 3),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        n_s3d_above_10=nsig('3d', 10),
        n_s3d_above_3=nsig('3d', 3),
        n_sd0_above_3=nsig('d0', 3),
        n_sdz_above_2=nsig('dz', 2),
        n_sdz_above_5=nsig('dz', 5),
        nca_sj4_pair2nd_over_mass=pq('nca_sj4_pair2nd_over_mass'),
        nca_sj4_pair_mass_2nd=pq('nca_sj4_pair_mass_2nd'),
        pair_max_lnkt=pairmax('lnkt'),
        pair_max_lnm2=pairmax('lnm2'),
        pair_mean_lndelta=pairsum('lndelta'),
        pair_mean_lnkt=pairsum('lnkt'),
        pair_mean_lnm2=pairsum('lnm2'),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pz_lnd0=pq('pz_lnd0'),
        pz_lnd2=pq('pz_lnd2'),
        pz_lnd3=pq('pz_lnd3'),
        pz_lnkt0=pq('pz_lnkt0'),
        pz_lnkt1=pq('pz_lnkt1'),
        pz_lnkt3=pq('pz_lnkt3'),
        sd_mass=softdrop("mass"),
        sdb_0_n=pq('sdb_0_n'),
        sdb_0_z=pq('sdb_0_z'),
        sdb_2_n=pq('sdb_2_n'),
        sdb_2_z=pq('sdb_2_z'),
        sdb_3_n=pq('sdb_3_n'),
        sdb_4_n=pq('sdb_4_n'),
        sdb_4_z=pq('sdb_4_z'),
        sdb_5_z=pq('sdb_5_z'),
        sdb_jp_top3=pq('sdb_jp_top3'),
        sip_3d_1=sip('3d', 1),
        sip_3d_2=sip('3d', 2),
        sip_3d_3=sip('3d', 3),
        sj2_dr=subjets(2)["dr"][0],
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr_min=min(subjets(3)["dr"]),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass3=subjets(3)["mass"][2],
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmax_over_m=max(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj4_dr_min=min(subjets(4)["dr"]),
        sj4_pair_mass_max=max(subjets(4)["mpair"]),
        sj4_pair_mass_min=min(subjets(4)["mpair"]),
        sjf_2_1_max3d=pq('sjf_2_1_max3d'),
        sjf_2_1_n_d3=pq('sjf_2_1_n_d3'),
        sjf_2_1_n_d5=pq('sjf_2_1_n_d5'),
        sjf_2_1_z_d3=pq('sjf_2_1_z_d3'),
        sjf_2_2_max3d=pq('sjf_2_2_max3d'),
        sjf_2_2_z_d3=pq('sjf_2_2_z_d3'),
        sjf_3_1_max3d=pq('sjf_3_1_max3d'),
        sjf_3_1_n_d3=pq('sjf_3_1_n_d3'),
        sjf_3_1_n_d5=pq('sjf_3_1_n_d5'),
        sjf_3_2_max3d=pq('sjf_3_2_max3d'),
        sjf_3_2_maxsd0=pq('sjf_3_2_maxsd0'),
        sjf_3_3_max3d=pq('sjf_3_3_max3d'),
        sjf_3_3_maxsd0=pq('sjf_3_3_maxsd0'),
        sjf_4_1_max3d=pq('sjf_4_1_max3d'),
        sjf_4_1_z_d3=pq('sjf_4_1_z_d3'),
        sjf_4_2_z_d3=pq('sjf_4_2_z_d3'),
        sjf_4_3_z_d3=pq('sjf_4_3_z_d3'),
        sjf_4_n2disp=pq('sjf_4_n2disp'),
        sjq_2_1_nch=pq('sjq_2_1_nch'),
        sjq_2_2_k1=pq('sjq_2_2_k1'),
        sjq_2_2_nch=pq('sjq_2_2_nch'),
        sjq_2_prod_k03=pq('sjq_2_prod_k03'),
        sjq_2_prod_k05=pq('sjq_2_prod_k05'),
        sjq_2_prod_k1=pq('sjq_2_prod_k1'),
        sjq_2_sumabs_k03=pq('sjq_2_sumabs_k03'),
        sjq_2_sumabs_k1=pq('sjq_2_sumabs_k1'),
        sjq_3_2_k1=pq('sjq_3_2_k1'),
        sjq_3_2_nch=pq('sjq_3_2_nch'),
        sjq_3_3_k1=pq('sjq_3_3_k1'),
        sjq_3_prod_k1=pq('sjq_3_prod_k1'),
        sjq_3_sumabs_k1=pq('sjq_3_sumabs_k1'),
        sum_e=sum(energy[i] for i in real),
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sv_1_dr=pq('sv_1_dr'),
        sv_1_n=pq('sv_1_n'),
        sv_1_sd0_sum=pq('sv_1_sd0_sum'),
        sv_2_z=pq('sv_2_z'),
        sv_n=pq('sv_n'),
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau32_b2=tau_n(3, 2) / max(tau_n(2, 2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        tau5=tau_n(5),
        tau54=tau_n(5) / max(tau_n(4), 1e-12),
        z_charged=sum(z[i] for i in real if charge[i] != 0),
        z_charged_had=sum(z[i] for i in real if ptype[i] == 1),
        z_displaced3=displaced(3, 'z'),
        z_displaced5=displaced(5, 'z'),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        z_dr_0p4_up=sum(z[i] for i in real if 0.4 <= dr[i] < 10),
        z_electron=sum(z[i] for i in real if ptype[i] == 4),
        z_muon=sum(z[i] for i in real if ptype[i] == 5),
        z_neutral_had=sum(z[i] for i in real if ptype[i] == 2),
        z_photon=sum(z[i] for i in real if ptype[i] == 3),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top3_slots=sum(pt[:3]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
    )


def neuron_0(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.00518969
    )
    return z


def neuron_1(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.325267e-06
    )
    return z


def neuron_2(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.447303e-06
    )
    return z


def neuron_3(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.399839e-06
    )
    return z


def neuron_4(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.79214e-06
    )
    return z


def neuron_5(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.156924e-06
    )
    return z


def neuron_6(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.770106e-05
    )
    return z


def neuron_7(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.430825e-06
    )
    return z


def neuron_8(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.926471e-06
    )
    return z


def neuron_9(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.443662e-06
    )
    return z


def neuron_10(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.570105e-05
    )
    return z


def neuron_11(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.887713e-05
    )
    return z


def neuron_12(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.099298e-06
    )
    return z


def neuron_13(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.422911e-06
    )
    return z


def neuron_14(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.845304e-06
    )
    return z


def neuron_15(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.459527e-06
    )
    return z


def neuron_16(Q):
    # scale S = 14.64; each line: share * term / its average size
    z = 14.64339 * (0.01477773
        - 0.1388152 * Q.z_charged_had / 0.5066138   # -13.9%  z_charged_had
        - 0.0878859 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # -8.8%  n_s3d_above_3 < 10
        + 0.08690503 * max(0.0, 0.5187302 - Q.lep_z) / 0.4405854   # +8.7%  lep_z < 0.5187
        - 0.06144841 * max(0.0, 0.8615361 - Q.z_charged) / 0.2675383   # -6.1%  z_charged < 0.8615
        - 0.05938406 * max(0.0, 0.2801368 - Q.z_displaced5) / 0.1998239   # -5.9%  z_displaced5 < 0.2801
        + 0.05211568 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # +5.2%  lep_ptrel < 27.32
        + 0.05021756 * max(0.0, 0.2801368 - Q.z_displaced5) * max(0.0, 172.888 - Q.jd_3d_4) / 31.78805   # +5.0%  z_displaced5 < 0.2801 and jd_3d_4 < 172.9
        + 0.02810798 * Q.lep_ptrel / 7.315413   # +2.8%  lep_ptrel
        + 0.02683806 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # +2.7%  e3_b2 < 0.000791
        + 0.02362958 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # +2.4%  sip_3d_2 < 226.3
        - 0.02231651 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # -2.2%  e3_b2 < 0.0004127
        - 0.0195643 * max(0.0, 0.1373477 - Q.z_electron) / 0.1179076   # -2.0%  z_electron < 0.1373
        + 0.01879008 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # +1.9%  n_pairs_kt_above_3 < 34
        + 0.01827846 * max(0.0, 175.9957 - Q.sip_3d_3) / 142.4273   # +1.8%  sip_3d_3 < 176
        + 0.01783324 * max(0.0, 0.09802954 - Q.sjq_2_prod_k1) / 0.1166532   # +1.8%  sjq_2_prod_k1 < 0.09803
        - 0.01706411 * max(0.0, 14.85973 - Q.jd_3d_4) / 8.639828   # -1.7%  jd_3d_4 < 14.86
        + 0.01665748 * Q.n_sdz_above_2 / 3.817083   # +1.7%  n_sdz_above_2
        - 0.01437572 * max(0.0, 0.03898651 - Q.z_muon) / 0.03219295   # -1.4%  z_muon < 0.03899
        + 0.0127015 * max(0.0, 16.0 - Q.sdb_2_n) / 6.019063   # +1.3%  sdb_2_n < 16
        - 0.01264768 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # -1.3%  n_pairs_kt_above_3 < 28
        + 0.01243719 * max(0.0, Q.mass_top40 - 112.406) / 12.32657   # +1.2%  mass_top40 > 112.4
        - 0.01142254 * max(0.0, 1.777286 - Q.mass_displaced3) / 0.7377974   # -1.1%  mass_displaced3 < 1.777
        + 0.01057413 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, 0.1270192 - Q.sdb_4_z) / 0.7401208   # +1.1%  n_s3d_above_3 < 10 and sdb_4_z < 0.127
        + 0.009963768 * max(0.0, 62.00562 - Q.mass_charged) / 9.422672   # +1.0%  mass_charged < 62.01
        + 0.009963218 * max(0.0, Q.sjq_2_sumabs_k1 - 0.2432922) / 0.1551258   # +1.0%  sjq_2_sumabs_k1 > 0.2433
        + 0.009111006 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # +0.9%  max_abs_d0 < 5.812
        - 0.008887995 * max(0.0, Q.mres_sd_mass_b0z005 - 159.9242) / 3.119054   # -0.9%  mres_sd_mass_b0z005 > 159.9
        - 0.008685969 * max(0.0, 18.0 - Q.n_photon) / 4.188877   # -0.9%  n_photon < 18
        + 0.00815626 * max(0.0, 2.957031 - Q.max_abs_dz) / 1.247099   # +0.8%  max_abs_dz < 2.957
        + 0.008041166 * max(0.0, Q.M3 - 0.02716047) / 0.01123924   # +0.8%  M3 > 0.02716
        + 0.006687722 * max(0.0, Q.mass_neutral - 34.02385) / 15.79354   # +0.7%  mass_neutral > 34.02
        + 0.005935819 * max(0.0, 7.0 - Q.n_pairs_kt_above_10) / 2.958697   # +0.6%  n_pairs_kt_above_10 < 7
        + 0.005717587 * max(0.0, 46.61784 - Q.mres_sd_prong_mass1) / 22.17314   # +0.6%  mres_sd_prong_mass1 < 46.62
        + 0.005112507 * max(0.0, Q.mres_sd_mass_b0z005 - 122.1) / 9.418671   # +0.5%  mres_sd_mass_b0z005 > 122.1
        - 0.00482585 * max(0.0, 9.408315 - Q.jd_sum_abs_sd0_top5) / 0.5127701   # -0.5%  jd_sum_abs_sd0_top5 < 9.408
        - 0.004751388 * max(0.0, Q.lep_z - 0.221436) / 0.03510013   # -0.5%  lep_z > 0.2214
        - 0.004572359 * max(0.0, 0.2801368 - Q.z_displaced5) * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 0.229834   # -0.5%  z_displaced5 < 0.2801 and n_lund_kt_above_5 > 1
        - 0.004357099 * max(0.0, 175.9957 - Q.sip_3d_3) * max(0.0, Q.mres_sd_prong_mass2 - 1.685874e-06) / 1399.378   # -0.4%  sip_3d_3 < 176 and mres_sd_prong_mass2 > 1.686e-06
        + 0.004151693 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.n_lund - 5.0) / 0.001819184   # +0.4%  e3_b2 < 0.0004127 and n_lund > 5
        + 0.004128054 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.n_neutral - 10.0) / 62.20264   # +0.4%  n_s3d_above_3 < 10 and n_neutral > 10
        + 0.004015587 * max(0.0, 0.08321691 - Q.pz_lnd2) / 0.01880965   # +0.4%  pz_lnd2 < 0.08322
        + 0.003939777 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.sv_1_n - 1.0) / 1.06826   # +0.4%  n_s3d_above_3 < 10 and sv_1_n > 1
        - 0.003891977 * max(0.0, 0.09802954 - Q.sjq_2_prod_k1) * max(0.0, 1.0 - Q.ak02_2_n_lep) / 0.09023323   # -0.4%  sjq_2_prod_k1 < 0.09803 and ak02_2_n_lep < 1
        - 0.003837042 * max(0.0, 0.01841101 - Q.dr_min_012) / 0.005231118   # -0.4%  dr_min_012 < 0.01841
        - 0.003795831 * max(0.0, 0.09802954 - Q.sjq_2_prod_k1) * max(0.0, 0.200234 - Q.dc_1_z_disp3) / 0.01941435   # -0.4%  sjq_2_prod_k1 < 0.09803 and dc_1_z_disp3 < 0.2002
        - 0.003714246 * max(0.0, Q.mass_neutral - 34.02385) * max(0.0, 2.912651 - Q.jd_3d_5) / 8.763977   # -0.4%  mass_neutral > 34.02 and jd_3d_5 < 2.913
        - 0.003333022 * max(0.0, 0.4436035 - Q.max_abs_dz) / 0.08080909   # -0.3%  max_abs_dz < 0.4436
        - 0.003109091 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.3%  lep_ptrel > 43.21
        - 0.002997924 * max(0.0, 0.01553665 - Q.z_dr_0p4_up) / 0.004779005   # -0.3%  z_dr_0p4_up < 0.01554
        + 0.002657282 * max(0.0, 6.0 - Q.n_dr_0p1_0p2) / 0.8571967   # +0.3%  n_dr_0p1_0p2 < 6
        + 0.002656833 * max(0.0, 91.15481 - Q.mres_sd_mass_b2z01) / 9.5029   # +0.3%  mres_sd_mass_b2z01 < 91.15
        - 0.002298292 * max(0.0, Q.sjq_2_sumabs_k1 - 0.2432922) * max(0.0, 2.0 - Q.n_lepton) / 0.1700657   # -0.2%  sjq_2_sumabs_k1 > 0.2433 and n_lepton < 2
        - 0.002289245 * max(0.0, 7.54918 - Q.sj2_mass2) / 0.8421773   # -0.2%  sj2_mass2 < 7.549
        + 0.002243321 * max(0.0, 3.101398 - Q.sjf_3_2_max3d) / 0.6332004   # +0.2%  sjf_3_2_max3d < 3.101
        - 0.002218478 * max(0.0, 16.0 - Q.sdb_2_n) * max(0.0, 105.4151 - Q.dc_pair_mass_min) / 387.233   # -0.2%  sdb_2_n < 16 and dc_pair_mass_min < 105.4
        + 0.002206897 * max(0.0, Q.mres_sd_mass_b0z005 - 178.8957) / 1.714218   # +0.2%  mres_sd_mass_b0z005 > 178.9
        + 0.001902171 * max(0.0, 0.01556887 - Q.z_dr_0p05_0p1) / 0.002231501   # +0.2%  z_dr_0p05_0p1 < 0.01557
        + 0.001843827 * max(0.0, Q.mass_2charged - 20.76537) / 3.761489   # +0.2%  mass_2charged > 20.77
        + 0.001789991 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +0.2%  n_pairs_kt_above_1 < 80
        + 0.001756887 * max(0.0, 3.0 - Q.sjq_2_2_nch) / 0.2391967   # +0.2%  sjq_2_2_nch < 3
        + 0.001448549 * max(0.0, Q.mres_sd_rg_b2z01 - 0.457968) / 0.03191439   # +0.1%  mres_sd_rg_b2z01 > 0.458
        - 0.001418121 * max(0.0, 16.0 - Q.sdb_2_n) * max(0.0, 0.8479222 - Q.tau43) / 0.4389785   # -0.1%  sdb_2_n < 16 and tau43 < 0.8479
        + 0.001272667 * max(0.0, Q.ak02_dr23 - 0.3194837) / 0.08880139   # +0.1%  ak02_dr23 > 0.3195
        + 0.001265293 * max(0.0, Q.mres_sd_mass_b0z02 - 93.75785) / 14.10696   # +0.1%  mres_sd_mass_b0z02 > 93.76
        + 0.001154567 * max(0.0, 0.01841101 - Q.dr_min_012) * max(0.0, 3.652671 - Q.sjf_4_1_max3d) / 0.004013044   # +0.1%  dr_min_012 < 0.01841 and sjf_4_1_max3d < 3.653
        - 0.0009651014 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, Q.lepsj_2_n_d3 - 2.0) / 48.03862   # -0.1%  sip_3d_2 < 447.1 and lepsj_2_n_d3 > 2
        + 0.0008552464 * max(0.0, 3.577424 - Q.lne_4) / 0.1166942   # +0.1%  lne_4 < 3.577
        + 0.0008238434 * max(0.0, Q.mass_top5 - 62.84493) / 3.692784   # +0.1%  mass_top5 > 62.84
        - 0.000567226 * max(0.0, Q.mass_2charged - 45.73288) / 0.6358753   # -0.1%  mass_2charged > 45.73
        + 0.0003896638 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.mass_2photon - 22.18431) / 0.000102149   # +0.0%  e3_b2 < 0.0004127 and mass_2photon > 22.18
        - 0.0002821426 * max(0.0, Q.jd_3d_6 - 30.57088) / 4.999257   # -0.0%  jd_3d_6 > 30.57
    )
    return z


def neuron_17(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.035347e-05
    )
    return z


def neuron_18(Q):
    # scale S = 11.27; each line: share * term / its average size
    z = 11.27334 * (0.2228076
        - 0.07574061 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -7.6%  lep_z < 0.2214
        + 0.06340555 * max(0.0, 3.0 - Q.lepsj_3_n_d3) / 2.624773   # +6.3%  lepsj_3_n_d3 < 3
        - 0.04734165 * max(0.0, 3.710567 - Q.sjf_3_1_max3d) / 0.7122498   # -4.7%  sjf_3_1_max3d < 3.711
        + 0.03723927 * max(0.0, 0.08090366 - Q.z_displaced5) / 0.04032225   # +3.7%  z_displaced5 < 0.0809
        - 0.03511071 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.08090366 - Q.z_displaced5) / 0.00723244   # -3.5%  lep_z < 0.2214 and z_displaced5 < 0.0809
        - 0.03399963 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # -3.4%  e3_b2 < 0.000791
        - 0.03350205 * max(0.0, 0.004135872 - Q.lep_z) / 0.002390373   # -3.4%  lep_z < 0.004136
        - 0.03298963 * max(0.0, 46.74905 - Q.lepsj_3_maxsd0) / 40.371   # -3.3%  lepsj_3_maxsd0 < 46.75
        - 0.02887373 * max(0.0, 586.9572 - Q.lepsj_3_maxsd0) / 546.6092   # -2.9%  lepsj_3_maxsd0 < 587
        + 0.02875863 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 1.761851   # +2.9%  lepsj_3_maxsd0 < 2.413
        - 0.02859788 * max(0.0, 29.0 - Q.n_for_90pct) / 10.0757   # -2.9%  n_for_90pct < 29
        - 0.02851313 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # -2.9%  n_s3d_above_3 < 10
        + 0.02683696 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.3829918 - Q.N2) / 0.01292108   # +2.7%  lep_z < 0.2214 and N2 < 0.383
        - 0.02130945 * max(0.0, Q.mass - 110.2019) / 16.73354   # -2.1%  mass > 110.2
        + 0.02093048 * max(0.0, 2.924357 - Q.sjf_3_1_max3d) / 0.3728193   # +2.1%  sjf_3_1_max3d < 2.924
        + 0.01908379 * max(0.0, 2.0 - Q.lepsj_3_n_d3) / 1.694207   # +1.9%  lepsj_3_n_d3 < 2
        - 0.01791406 * max(0.0, 126.8853 - Q.mass_top40) / 23.76162   # -1.8%  mass_top40 < 126.9
        + 0.01769444 * max(0.0, 55.92931 - Q.sjf_2_2_max3d) / 33.91681   # +1.8%  sjf_2_2_max3d < 55.93
        - 0.01721578 * max(0.0, 6.185635 - Q.lep_iso) / 4.883283   # -1.7%  lep_iso < 6.186
        - 0.01717028 * max(0.0, 4.516968 - Q.sjf_2_2_max3d) / 1.242753   # -1.7%  sjf_2_2_max3d < 4.517
        + 0.01649032 * max(0.0, 0.03663959 - Q.tau5) / 0.009671449   # +1.6%  tau5 < 0.03664
        - 0.01591481 * max(0.0, Q.n_real_top50 - 26.0) / 12.00248   # -1.6%  n_real_top50 > 26
        - 0.01567548 * max(0.0, 0.0828821 - Q.sjf_2_1_z_d3) / 0.04834541   # -1.6%  sjf_2_1_z_d3 < 0.08288
        - 0.01417767 * max(0.0, Q.mass - 100.4835) / 22.56274   # -1.4%  mass > 100.5
        + 0.01375298 * max(0.0, 71.08287 - Q.sjf_3_1_max3d) / 42.52076   # +1.4%  sjf_3_1_max3d < 71.08
        + 0.01350687 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +1.4%  lep_iso < 1.362
        + 0.01340222 * max(0.0, 16.0 - Q.sjq_2_1_nch) / 5.009663   # +1.3%  sjq_2_1_nch < 16
        + 0.0130745 * max(0.0, 2.409613 - Q.mass_displaced3) / 1.070706   # +1.3%  mass_displaced3 < 2.41
        + 0.01268618 * max(0.0, Q.mass - 164.4374) / 3.038568   # +1.3%  mass > 164.4
        + 0.0120231 * max(0.0, 6.0 - Q.n_s3d_above_10) / 3.631173   # +1.2%  n_s3d_above_10 < 6
        + 0.01201355 * max(0.0, 1.0 - Q.sjf_3_1_n_d3) / 0.4519233   # +1.2%  sjf_3_1_n_d3 < 1
        + 0.0108405 * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0001737301   # +1.1%  e3_b2 < 0.0002537
        - 0.01074831 * max(0.0, 4.606241 - Q.sip_3d_3) / 1.172198   # -1.1%  sip_3d_3 < 4.606
        + 0.01003705 * max(0.0, 3.0 - Q.n_sdz_above_5) / 1.349857   # +1.0%  n_sdz_above_5 < 3
        - 0.009717051 * max(0.0, 16.0 - Q.sjq_2_1_nch) * max(0.0, 1651.766 - Q.sum_e) / 3918.801   # -1.0%  sjq_2_1_nch < 16 and sum_e < 1652
        - 0.00937068 * max(0.0, 3.0 - Q.lepsj_3_n_d3) * max(0.0, 0.08178299 - Q.lep_dr) / 0.1447323   # -0.9%  lepsj_3_n_d3 < 3 and lep_dr < 0.08178
        + 0.00914985 * max(0.0, 51.56287 - Q.sv_1_sd0_sum) / 27.26317   # +0.9%  sv_1_sd0_sum < 51.56
        - 0.0087656 * max(0.0, 0.849996 - Q.ak02_1_z) / 0.1554704   # -0.9%  ak02_1_z < 0.85
        - 0.008454922 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.sdb_2_z - 0.2509165) / 0.0155742   # -0.8%  lep_z < 0.2214 and sdb_2_z > 0.2509
        - 0.007966823 * max(0.0, 17.21875 - Q.max_abs_dz) / 11.28733   # -0.8%  max_abs_dz < 17.22
        + 0.007959829 * max(0.0, 0.624781 - Q.psi_0p1) / 0.2784205   # +0.8%  psi_0p1 < 0.6248
        - 0.00768566 * max(0.0, 0.03624058 - Q.z_displaced3) / 0.01247855   # -0.8%  z_displaced3 < 0.03624
        - 0.007030256 * max(0.0, 0.05934469 - Q.sdb_0_z) / 0.03278889   # -0.7%  sdb_0_z < 0.05934
        - 0.006513481 * max(0.0, 813.043 - Q.sdb_jp_top3) / 410.6254   # -0.7%  sdb_jp_top3 < 813
        - 0.006064735 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # -0.6%  n_s3d_above_3 < 3
        - 0.005900234 * max(0.0, 114.4658 - Q.mass_top20) / 26.77686   # -0.6%  mass_top20 < 114.5
        - 0.005869027 * max(0.0, Q.mres_sd_mass_b2z01 - 121.6067) / 9.197708   # -0.6%  mres_sd_mass_b2z01 > 121.6
        + 0.005854995 * max(0.0, 55.67805 - Q.sjf_2_1_max3d) / 27.50165   # +0.6%  sjf_2_1_max3d < 55.68
        - 0.005324845 * max(0.0, 9.090532 - Q.mass_displaced3) * max(0.0, Q.sjq_2_2_nch - 4.0) / 17.22371   # -0.5%  mass_displaced3 < 9.091 and sjq_2_2_nch > 4
        + 0.004962714 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.60135   # +0.5%  n_dr_0p1_0p2 < 17
        + 0.004537026 * max(0.0, Q.M2 - 0.07013948) / 0.0151198   # +0.5%  M2 > 0.07014
        - 0.004518497 * max(0.0, 0.01299032 - Q.lam1) / 0.000791932   # -0.5%  lam1 < 0.01299
        - 0.004404683 * max(0.0, Q.mass - 182.8592) / 1.688248   # -0.4%  mass > 182.9
        + 0.004090781 * max(0.0, 6.465318 - Q.sj2_mass2) / 0.5968622   # +0.4%  sj2_mass2 < 6.465
        - 0.004087331 * max(0.0, Q.n_real_top50 - 26.0) * max(0.0, 3.0 - Q.n_lund_kt_above_5) / 8.186317   # -0.4%  n_real_top50 > 26 and n_lund_kt_above_5 < 3
        + 0.003909931 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +0.4%  n_dr_0p4_up < 4
        - 0.003407752 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.jet_charge_k05 - -0.2841306) / 0.06110708   # -0.3%  lep_z < 0.2214 and jet_charge_k05 > -0.2841
        + 0.003354785 * max(0.0, Q.mres_sd_mass_b2z01 - 177.5398) / 1.721011   # +0.3%  mres_sd_mass_b2z01 > 177.5
        - 0.003292396 * max(0.0, Q.sj3_pair_mass_max - 128.6605) / 2.418855   # -0.3%  sj3_pair_mass_max > 128.7
        + 0.003272474 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.lne_4 - 3.751708) / 0.03644274   # +0.3%  lep_z < 0.2214 and lne_4 > 3.752
        - 0.003150938 * max(0.0, Q.z_displaced3 - 0.2092108) / 0.02926899   # -0.3%  z_displaced3 > 0.2092
        + 0.003067733 * max(0.0, 0.3603262 - Q.z_charged_had) / 0.02270075   # +0.3%  z_charged_had < 0.3603
        + 0.002521906 * max(0.0, 0.1032185 - Q.pz_lnd0) / 0.01871678   # +0.3%  pz_lnd0 < 0.1032
        - 0.002328385 * max(0.0, Q.ak02_dr23 - 0.3194837) / 0.08880139   # -0.2%  ak02_dr23 > 0.3195
        + 0.002233912 * max(0.0, Q.mres_sd_mass_b2z01 - 111.4443) / 13.01666   # +0.2%  mres_sd_mass_b2z01 > 111.4
        + 0.002122757 * max(0.0, 3.0 - Q.sjq_3_2_nch) / 0.2817433   # +0.2%  sjq_3_2_nch < 3
        + 0.002060991 * max(0.0, Q.sjf_4_1_z_d3 - 0.181327) / 0.01992241   # +0.2%  sjf_4_1_z_d3 > 0.1813
        + 0.002014474 * max(0.0, 1.348343 - Q.sjf_3_3_max3d) / 0.2007926   # +0.2%  sjf_3_3_max3d < 1.348
        + 0.001904661 * max(0.0, 2.078395 - Q.sjf_3_1_max3d) / 0.1080215   # +0.2%  sjf_3_1_max3d < 2.078
        - 0.001785039 * max(0.0, 0.2072106 - Q.sj3_pairmin_over_m) / 0.01530396   # -0.2%  sj3_pairmin_over_m < 0.2072
        - 0.001775791 * max(0.0, Q.e3_b05 - 0.008870191) / 0.001172079   # -0.2%  e3_b05 > 0.00887
        - 0.001526892 * max(0.0, 51.56287 - Q.sv_1_sd0_sum) * max(0.0, Q.dc_1_n_lep - 0.0) / 6.332546   # -0.2%  sv_1_sd0_sum < 51.56 and dc_1_n_lep > 0
        - 0.001468904 * max(0.0, Q.N2 - 0.4137858) / 0.003073306   # -0.1%  N2 > 0.4138
    )
    return z


def neuron_19(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.40782e-06
    )
    return z


def neuron_20(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.253686e-07
    )
    return z


def neuron_21(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.001682464
    )
    return z


def neuron_22(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.044846e-06
    )
    return z


def neuron_23(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.978799e-07
    )
    return z


def neuron_24(Q):
    # scale S = 17.28; each line: share * term / its average size
    z = 17.28352 * (-0.02817077
        - 0.07834602 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # -7.8%  lep_z < 0.3397
        + 0.06447638 * max(0.0, 119.8459 - Q.mres_pruned_mass) / 32.90211   # +6.4%  mres_pruned_mass < 119.8
        - 0.04926575 * max(0.0, 86.14266 - Q.mres_pruned_mass) / 13.9802   # -4.9%  mres_pruned_mass < 86.14
        - 0.04850226 * max(0.0, 0.103005 - Q.lepsj_2_dr) * max(0.0, 14.38858 - Q.lep_iso) / 1.091303   # -4.9%  lepsj_2_dr < 0.103 and lep_iso < 14.39
        + 0.04429122 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +4.4%  lep_iso < 0.4382
        - 0.03533999 * max(0.0, 171.3819 - Q.mres_pruned_mass) / 78.34654   # -3.5%  mres_pruned_mass < 171.4
        - 0.03479071 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -3.5%  mass < 164.4
        - 0.03435203 * max(0.0, 0.004135872 - Q.lep_z) * max(0.0, 2.907433 - Q.lep_iso) / 0.006825218   # -3.4%  lep_z < 0.004136 and lep_iso < 2.907
        + 0.03131112 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.8036986 - Q.lepsj_3_maxsd0) / 0.1622685   # +3.1%  lep_z < 0.3397 and lepsj_3_maxsd0 < 0.8037
        + 0.03097769 * Q.pair_max_lnkt / 2.706794   # +3.1%  pair_max_lnkt
        + 0.026815 * max(0.0, 118.2746 - Q.mres_sd_mass_b2z01) / 22.65141   # +2.7%  mres_sd_mass_b2z01 < 118.3
        + 0.01999197 * max(0.0, Q.tau21 - 0.135772) / 0.3001429   # +2.0%  tau21 > 0.1358
        + 0.01985041 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 127.2317 - Q.mass) / 5.805704   # +2.0%  lep_z < 0.3397 and mass < 127.2
        + 0.0194543 * max(0.0, 18.80005 - Q.mass_displaced3) / 13.42744   # +1.9%  mass_displaced3 < 18.8
        + 0.01939016 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, 0.1822601 - Q.lepsj_2_dr) / 0.05233764   # +1.9%  lep_iso < 0.4382 and lepsj_2_dr < 0.1823
        + 0.01868943 * max(0.0, 134.2224 - Q.mass_top30) / 33.8221   # +1.9%  mass_top30 < 134.2
        - 0.01820534 * max(0.0, 81.19466 - Q.mres_sd_mass_b2z01) / 6.499995   # -1.8%  mres_sd_mass_b2z01 < 81.19
        + 0.01681298 * max(0.0, 17.0 - Q.n_dr_0p2_0p4) / 7.328103   # +1.7%  n_dr_0p2_0p4 < 17
        - 0.01662866 * max(0.0, 84.11398 - Q.jd_sum_abs_sd0_top5) * max(0.0, 6.771002 - Q.jd_3d_5) / 138.5038   # -1.7%  jd_sum_abs_sd0_top5 < 84.11 and jd_3d_5 < 6.771
        - 0.01660677 * max(0.0, 18.80005 - Q.mass_displaced3) * max(0.0, 2325.187 - Q.sip_3d_2) / 28111.09   # -1.7%  mass_displaced3 < 18.8 and sip_3d_2 < 2325
        + 0.01580463 * max(0.0, 60.18017 - Q.mres_pruned_mass) / 6.5815   # +1.6%  mres_pruned_mass < 60.18
        - 0.01561738 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # -1.6%  e3_b2 < 0.0004127
        + 0.01525227 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +1.5%  lep_ptrel < 43.21
        - 0.01503691 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.tau21 - 0.1757731) / 0.07577322   # -1.5%  lep_z < 0.3397 and tau21 > 0.1758
        + 0.01299573 * max(0.0, 110.2019 - Q.mass) / 12.0043   # +1.3%  mass < 110.2
        + 0.01267825 * max(0.0, 18.80005 - Q.mass_displaced3) * max(0.0, 4.004982 - Q.jd_3d_5) / 24.96886   # +1.3%  mass_displaced3 < 18.8 and jd_3d_5 < 4.005
        + 0.01227465 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # +1.2%  n_pairs_kt_above_1 < 366
        - 0.01128013 * max(0.0, 83.63398 - Q.sj3_pair_mass_max) * max(0.0, 0.4381892 - Q.lep_iso) / 2.327307   # -1.1%  sj3_pair_mass_max < 83.63 and lep_iso < 0.4382
        - 0.01121897 * max(0.0, 0.103005 - Q.lepsj_2_dr) / 0.08254947   # -1.1%  lepsj_2_dr < 0.103
        - 0.01113954 * max(0.0, 164.4374 - Q.mass) * max(0.0, 3.0 - Q.sv_1_n) / 110.1758   # -1.1%  mass < 164.4 and sv_1_n < 3
        + 0.01060686 * max(0.0, 84.11398 - Q.jd_sum_abs_sd0_top5) / 28.0218   # +1.1%  jd_sum_abs_sd0_top5 < 84.11
        + 0.009827258 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # +1.0%  max_abs_d0 < 10.52
        - 0.009607277 * max(0.0, 12.15228 - Q.lep_ptrel) / 8.819929   # -1.0%  lep_ptrel < 12.15
        + 0.008823491 * max(0.0, 83.63398 - Q.sj3_pair_mass_max) * max(0.0, 0.1822601 - Q.lepsj_2_dr) / 1.279459   # +0.9%  sj3_pair_mass_max < 83.63 and lepsj_2_dr < 0.1823
        - 0.008464185 * max(0.0, 0.1468781 - Q.z_dr_0p2_0p4) / 0.048751   # -0.8%  z_dr_0p2_0p4 < 0.1469
        - 0.008275862 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, 5.0 - Q.sdb_0_n) / 0.001102698   # -0.8%  e3_b2 < 0.0004127 and sdb_0_n < 5
        + 0.00792432 * max(0.0, 0.212136 - Q.z_neutral_had) / 0.08633882   # +0.8%  z_neutral_had < 0.2121
        + 0.007595019 * max(0.0, 83.63398 - Q.sj3_pair_mass_max) * max(0.0, 14.38858 - Q.lep_iso) / 98.03357   # +0.8%  sj3_pair_mass_max < 83.63 and lep_iso < 14.39
        + 0.007560571 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +0.8%  lep_iso < 1.362
        - 0.007497956 * max(0.0, 0.004135872 - Q.lep_z) / 0.002390373   # -0.7%  lep_z < 0.004136
        - 0.007358645 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 4.004982 - Q.jd_3d_5) / 10.57583   # -0.7%  max_abs_d0 < 10.52 and jd_3d_5 < 4.005
        + 0.007088536 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 15.17086 - Q.D2_b2) / 0.5885   # +0.7%  z_displaced3 < 0.1062 and D2_b2 < 15.17
        - 0.006399083 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 7.0 - Q.n_neutral_had) / 0.7710798   # -0.6%  lep_z < 0.3397 and n_neutral_had < 7
        + 0.006213513 * max(0.0, 0.004135872 - Q.lep_z) * max(0.0, 0.01435877 - Q.sdb_4_z) / 2.61867e-05   # +0.6%  lep_z < 0.004136 and sdb_4_z < 0.01436
        - 0.006149864 * max(0.0, 164.4374 - Q.mass) * max(0.0, 0.0309457 - Q.M3_b2) / 0.822219   # -0.6%  mass < 164.4 and M3_b2 < 0.03095
        + 0.005970699 * max(0.0, 164.4374 - Q.mass) * max(0.0, 24.0 - Q.n_dr_0p2_0p4) / 828.5246   # +0.6%  mass < 164.4 and n_dr_0p2_0p4 < 24
        - 0.00544705 * max(0.0, 83.63398 - Q.sj3_pair_mass_max) / 7.654671   # -0.5%  sj3_pair_mass_max < 83.63
        + 0.005241388 * max(0.0, Q.n_s3d_above_10 - 1.0) / 1.904323   # +0.5%  n_s3d_above_10 > 1
        - 0.00501464 * max(0.0, 0.456416 - Q.tau32) / 0.0169303   # -0.5%  tau32 < 0.4564
        + 0.004941355 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +0.5%  n_dr_0p4_up < 4
        + 0.004790923 * max(0.0, 128.6605 - Q.sj3_pair_mass_max) / 38.7386   # +0.5%  sj3_pair_mass_max < 128.7
        + 0.004651001 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.456416 - Q.tau32) / 0.003646089   # +0.5%  lep_z < 0.3397 and tau32 < 0.4564
        - 0.004583354 * max(0.0, 7.452974e-05 - Q.e3_b2) / 3.411896e-05   # -0.5%  e3_b2 < 7.453e-05
        + 0.004235578 * max(0.0, 55.13212 - Q.mres_sd_mass_b2z01) / 2.457281   # +0.4%  mres_sd_mass_b2z01 < 55.13
        + 0.00394733 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.lne_8 - 2.11682) / 0.0003149447   # +0.4%  e3_b2 < 0.0004127 and lne_8 > 2.117
        + 0.003490724 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.sdb_2_z - 0.2968888) / 2.687353   # +0.3%  lep_ptrel < 43.21 and sdb_2_z > 0.2969
        - 0.00330744 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.3%  mass < 90.09
        + 0.003079464 * max(0.0, 1.0 - Q.n_s3d_above_3) / 0.17383   # +0.3%  n_s3d_above_3 < 1
        + 0.00299176 * max(0.0, 6.161303 - Q.dc_2_jp) / 3.330118   # +0.3%  dc_2_jp < 6.161
        + 0.002983649 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # +0.3%  n_s3d_above_3 < 3
        + 0.00297748 * max(0.0, 2.953464 - Q.jd_3d_4) / 0.5545459   # +0.3%  jd_3d_4 < 2.953
        - 0.002927772 * max(0.0, Q.z_charged_had - 0.5398733) / 0.0536258   # -0.3%  z_charged_had > 0.5399
        + 0.002645674 * max(0.0, Q.sjq_2_2_k1 - 0.277232) / 0.04421729   # +0.3%  sjq_2_2_k1 > 0.2772
        - 0.002585394 * max(0.0, 0.2423129 - Q.N2) / 0.01410631   # -0.3%  N2 < 0.2423
        - 0.002546927 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.3683248 - Q.sj3_dr12) / 0.02771715   # -0.3%  lep_z < 0.3397 and sj3_dr12 < 0.3683
        + 0.002499404 * max(0.0, 7.028704 - Q.sip_3d_1) / 1.023735   # +0.2%  sip_3d_1 < 7.029
        - 0.002271486 * max(0.0, 74.51927 - Q.mass_top30) / 2.669461   # -0.2%  mass_top30 < 74.52
        + 0.002121284 * max(0.0, -0.6014774 - Q.sjq_3_2_k1) / 0.01396761   # +0.2%  sjq_3_2_k1 < -0.6015
        + 0.001940569 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, Q.sdb_4_n - 0.0) / 0.1221333   # +0.2%  lep_iso < 0.4382 and sdb_4_n > 0
        + 0.001591269 * max(0.0, Q.dc_split2_dr - 0.2293319) / 0.01625115   # +0.2%  dc_split2_dr > 0.2293
        + 0.001584185 * max(0.0, 5.0 - Q.n_lund) / 0.1017067   # +0.2%  n_lund < 5
        - 0.001531794 * max(0.0, 0.006142967 - Q.lam1) / 0.0001406198   # -0.2%  lam1 < 0.006143
        - 0.001498507 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 0.1166862 - Q.pz_lnd2) / 0.001637201   # -0.1%  z_displaced3 < 0.1062 and pz_lnd2 < 0.1167
        - 0.001404998 * max(0.0, Q.sjq_2_2_k1 - 0.277232) * max(0.0, 2.907433 - Q.lep_iso) / 0.1043629   # -0.1%  sjq_2_2_k1 > 0.2772 and lep_iso < 2.907
        - 0.001399114 * Q.sjq_2_prod_k1 / 0.05054899   # -0.1%  sjq_2_prod_k1
        - 0.001303164 * max(0.0, Q.dc_split2_dr - 0.3591078) / 0.004116954   # -0.1%  dc_split2_dr > 0.3591
        - 0.001215872 * max(0.0, -0.6014774 - Q.sjq_3_2_k1) * max(0.0, 0.4381892 - Q.lep_iso) / 0.003672589   # -0.1%  sjq_3_2_k1 < -0.6015 and lep_iso < 0.4382
        + 0.001088892 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, Q.mass_2photon - 3.013) / 0.1897934   # +0.1%  z_displaced3 < 0.1062 and mass_2photon > 3.013
        + 0.001069977 * max(0.0, 0.103005 - Q.lepsj_2_dr) * max(0.0, Q.jet_e - 926.0771) / 10.59847   # +0.1%  lepsj_2_dr < 0.103 and jet_e > 926.1
        - 0.001011428 * max(0.0, 0.004135872 - Q.lep_z) * max(0.0, Q.n_pairs_kt_above_10 - 4.0) / 0.0104063   # -0.1%  lep_z < 0.004136 and n_pairs_kt_above_10 > 4
        - 0.0007929172 * max(0.0, Q.mass_top15 - 70.93762) * max(0.0, Q.sv_2_z - 0.00726873) / 0.09933311   # -0.1%  mass_top15 > 70.94 and sv_2_z > 0.007269
        + 0.0005264139 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.e4 - 4.06096e-07) / 3.568567e-07   # +0.1%  lep_z < 0.3397 and e4 > 4.061e-07
    )
    return z


def neuron_25(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.03136871
    )
    return z


def neuron_26(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.231261e-06
    )
    return z


def neuron_27(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.308501e-06
    )
    return z


def neuron_28(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.617208e-06
    )
    return z


def neuron_29(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.183865e-06
    )
    return z


def neuron_30(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.603055e-06
    )
    return z


def neuron_31(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.509523e-06
    )
    return z


def neuron_32(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.441348e-05
    )
    return z


def neuron_33(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.567326e-06
    )
    return z


def neuron_34(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.388486e-05
    )
    return z


def neuron_35(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.743301e-06
    )
    return z


def neuron_36(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.831948e-06
    )
    return z


def neuron_37(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.357436e-06
    )
    return z


def neuron_38(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.411176e-06
    )
    return z


def neuron_39(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.274277e-06
    )
    return z


def neuron_40(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.77554e-07
    )
    return z


def neuron_41(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.814691e-08
    )
    return z


def neuron_42(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.779052e-06
    )
    return z


def neuron_43(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.407709e-06
    )
    return z


def neuron_44(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.324349e-06
    )
    return z


def neuron_45(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.849337e-06
    )
    return z


def neuron_46(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.368759e-07
    )
    return z


def neuron_47(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.492075e-07
    )
    return z


def neuron_48(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.791778e-06
    )
    return z


def neuron_49(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.243815e-06
    )
    return z


def neuron_50(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.175116e-06
    )
    return z


def neuron_51(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.012202e-06
    )
    return z


def neuron_52(Q):
    # scale S = 10.9; each line: share * term / its average size
    z = 10.90385 * (-0.09812106
        + 0.1008522 * max(0.0, 159.9242 - Q.mres_sd_mass_b0z005) / 57.19246   # +10.1%  mres_sd_mass_b0z005 < 159.9
        - 0.05928173 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -5.9%  n_lepton < 1
        + 0.05461783 * max(0.0, 0.3014662 - Q.lep_dr) / 0.2238267   # +5.5%  lep_dr < 0.3015
        - 0.04439072 * max(0.0, 131.0776 - Q.mres_sd_mass_b0z005) / 32.45252   # -4.4%  mres_sd_mass_b0z005 < 131.1
        - 0.04086264 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -4.1%  lep_z < 0.2214
        - 0.03957859 * max(0.0, 0.103005 - Q.lepsj_2_dr) / 0.08254947   # -4.0%  lepsj_2_dr < 0.103
        + 0.03957056 * max(0.0, 0.1342762 - Q.z_displaced3) / 0.07089169   # +4.0%  z_displaced3 < 0.1343
        - 0.03868654 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 30.57088 - Q.jd_3d_6) / 2.01762   # -3.9%  z_displaced3 < 0.1343 and jd_3d_6 < 30.57
        + 0.03409331 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 5.367501 - Q.jd_3d_6) / 0.2552407   # +3.4%  z_displaced3 < 0.1343 and jd_3d_6 < 5.368
        + 0.02908995 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +2.9%  lep_iso < 1.362
        + 0.02760161 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 1.761851   # +2.8%  lepsj_3_maxsd0 < 2.413
        - 0.02611469 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 577.991 - Q.sip_3d_3) / 38.5554   # -2.6%  z_displaced3 < 0.1343 and sip_3d_3 < 578
        + 0.02591674 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # +2.6%  lep_z < 0.3397
        + 0.02578232 * max(0.0, 0.3565533 - Q.tau21_b2) / 0.1109794   # +2.6%  tau21_b2 < 0.3566
        - 0.02310073 * max(0.0, 12.15228 - Q.lep_ptrel) / 8.819929   # -2.3%  lep_ptrel < 12.15
        + 0.02258634 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.n_charged_pt_above_1 - 11.0) / 2.519562   # +2.3%  lep_z < 0.3397 and n_charged_pt_above_1 > 11
        - 0.0217165 * max(0.0, Q.lne_16 - 0.4468807) / 1.728342   # -2.2%  lne_16 > 0.4469
        + 0.02033661 * max(0.0, 1.0 - Q.dc_1_n_lep) / 0.78823   # +2.0%  dc_1_n_lep < 1
        - 0.01984703 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # -2.0%  lep_ptrel < 27.32
        + 0.01869655 * max(0.0, 0.2837384 - Q.ak02_2_z) / 0.0942207   # +1.9%  ak02_2_z < 0.2837
        - 0.01763871 * max(0.0, 159.9242 - Q.mres_sd_mass_b0z005) * max(0.0, 0.3565533 - Q.tau21_b2) / 5.887371   # -1.8%  mres_sd_mass_b0z005 < 159.9 and tau21_b2 < 0.3566
        - 0.01677353 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 10.56503   # -1.7%  n_dr_0p2_0p4 < 21
        + 0.01321416 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # +1.3%  e3_b2 < 0.0004127
        + 0.01295692 * max(0.0, Q.N2_b2 - 0.08291719) / 0.0935801   # +1.3%  N2_b2 > 0.08292
        + 0.01221877 * max(0.0, 0.3788785 - Q.z_neutral_had) * max(0.0, Q.ecf_g41 - 6.216954e-05) / 3.09401e-05   # +1.2%  z_neutral_had < 0.3789 and ecf_g41 > 6.217e-05
        + 0.01209762 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # +1.2%  n_s3d_above_3 < 3
        - 0.01083105 * max(0.0, 125.2765 - Q.sv_1_sd0_sum) / 79.48195   # -1.1%  sv_1_sd0_sum < 125.3
        - 0.01067401 * max(0.0, 84.11398 - Q.jd_sum_abs_sd0_top5) / 28.0218   # -1.1%  jd_sum_abs_sd0_top5 < 84.11
        - 0.01002682 * max(0.0, Q.ak02_1_z - 0.6342743) / 0.116529   # -1.0%  ak02_1_z > 0.6343
        - 0.009958067 * max(0.0, 0.3788785 - Q.z_neutral_had) * max(0.0, Q.pz_lnkt1 - 0.03633353) / 0.01470373   # -1.0%  z_neutral_had < 0.3789 and pz_lnkt1 > 0.03633
        + 0.009844256 * max(0.0, 0.103005 - Q.lepsj_2_dr) * max(0.0, 0.2250047 - Q.C3_b05) / 0.009818842   # +1.0%  lepsj_2_dr < 0.103 and C3_b05 < 0.225
        + 0.009610903 * max(0.0, Q.mres_sd_mass_b0z005 - 81.62059) / 31.49572   # +1.0%  mres_sd_mass_b0z005 > 81.62
        + 0.009014796 * max(0.0, Q.mres_sd_mass_b0z005 - 81.62059) * max(0.0, 0.07650476 - Q.ak02_3_z) / 0.9745751   # +0.9%  mres_sd_mass_b0z005 > 81.62 and ak02_3_z < 0.0765
        - 0.008602373 * max(0.0, Q.sdb_2_n - 9.0) / 2.899327   # -0.9%  sdb_2_n > 9
        + 0.007945809 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, Q.N2 - 0.1824346) / 0.009842026   # +0.8%  z_displaced3 < 0.1343 and N2 > 0.1824
        + 0.007815641 * max(0.0, 0.3014662 - Q.lep_dr) * max(0.0, Q.sdb_2_z - 0.1530389) / 0.04087335   # +0.8%  lep_dr < 0.3015 and sdb_2_z > 0.153
        - 0.006474649 * max(0.0, 10.0 - Q.n_dr_0p4_up) / 5.21002   # -0.6%  n_dr_0p4_up < 10
        + 0.006462107 * max(0.0, Q.lund3_lndelta - -2.817283) * max(0.0, 0.3689526 - Q.sv_1_dr) / 0.3049417   # +0.6%  lund3_lndelta > -2.817 and sv_1_dr < 0.369
        + 0.006038889 * max(0.0, 0.1122946 - Q.pz_lnd0) / 0.02302722   # +0.6%  pz_lnd0 < 0.1123
        - 0.005851182 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sjq_2_prod_k05 - -0.5057096) / 4.880743   # -0.6%  n_dr_0p2_0p4 < 21 and sjq_2_prod_k05 > -0.5057
        + 0.005291884 * max(0.0, 3.029824 - Q.sjf_2_2_max3d) / 0.5243565   # +0.5%  sjf_2_2_max3d < 3.03
        - 0.005090544 * max(0.0, 115.614 - Q.mass_top50) / 15.03711   # -0.5%  mass_top50 < 115.6
        + 0.004933569 * max(0.0, 79.27954 - Q.mass_top50) / 2.845865   # +0.5%  mass_top50 < 79.28
        + 0.004885655 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.1145956 - Q.sjq_3_prod_k1) / 1.507298   # +0.5%  n_dr_0p2_0p4 < 21 and sjq_3_prod_k1 < 0.1146
        - 0.004812866 * max(0.0, 21.7295 - Q.sj2_mass2) / 8.475837   # -0.5%  sj2_mass2 < 21.73
        - 0.004439806 * max(0.0, Q.jet_charge_k03 - -0.2409073) / 0.4302075   # -0.4%  jet_charge_k03 > -0.2409
        + 0.004008012 * max(0.0, 7.0 - Q.n_lund) / 0.2630033   # +0.4%  n_lund < 7
        + 0.003996198 * max(0.0, 0.3740528 - Q.sj2_dr) / 0.05324013   # +0.4%  sj2_dr < 0.3741
        - 0.003664833 * max(0.0, 1.421532 - Q.D2) / 0.1381636   # -0.4%  D2 < 1.422
        + 0.003463125 * max(0.0, Q.sdb_2_n - 9.0) * max(0.0, 2.01745 - Q.D2_b2) / 1.203096   # +0.3%  sdb_2_n > 9 and D2_b2 < 2.017
        + 0.003444292 * max(0.0, Q.mres_sd_mass_b0z005 - 81.62059) * max(0.0, 0.03385157 - Q.sjf_2_2_z_d3) / 0.6012181   # +0.3%  mres_sd_mass_b0z005 > 81.62 and sjf_2_2_z_d3 < 0.03385
        + 0.003283534 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, Q.n_lund_kt_above_1 - 3.0) / 0.1680127   # +0.3%  z_displaced3 < 0.1343 and n_lund_kt_above_1 > 3
        - 0.003006112 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.sj3_pair_mass_min - 48.40547) / 1.423551   # -0.3%  lep_z < 0.3397 and sj3_pair_mass_min > 48.41
        - 0.002986291 * max(0.0, Q.mres_sd_mass_b0z005 - 81.62059) * max(0.0, 0.7544983 - Q.tau32) / 5.345875   # -0.3%  mres_sd_mass_b0z005 > 81.62 and tau32 < 0.7545
        - 0.002791063 * max(0.0, Q.mass_2charged - 14.28253) / 5.706501   # -0.3%  mass_2charged > 14.28
        + 0.002755125 * max(0.0, -0.6337755 - Q.sjq_2_2_k1) / 0.01353013   # +0.3%  sjq_2_2_k1 < -0.6338
        + 0.002591094 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sjq_3_sumabs_k1 - 0.3376898) / 1.666965   # +0.3%  n_dr_0p2_0p4 < 21 and sjq_3_sumabs_k1 > 0.3377
        + 0.002556743 * max(0.0, Q.sjq_2_2_k1 - 0.6477929) / 0.01300326   # +0.3%  sjq_2_2_k1 > 0.6478
        - 0.002059366 * max(0.0, 159.9242 - Q.mres_sd_mass_b0z005) * max(0.0, Q.mass_displaced3 - 3.208089) / 243.7029   # -0.2%  mres_sd_mass_b0z005 < 159.9 and mass_displaced3 > 3.208
        + 0.00187461 * max(0.0, Q.lep_ptrel - 18.7678) / 2.829169   # +0.2%  lep_ptrel > 18.77
        - 0.001782 * max(0.0, 0.001184159 - Q.D3_b2) * max(0.0, 5.367501 - Q.jd_3d_6) / 0.0004088142   # -0.2%  D3_b2 < 0.001184 and jd_3d_6 < 5.368
        + 0.001702736 * max(0.0, 0.103005 - Q.lepsj_2_dr) * max(0.0, Q.z_photon - 0.3318968) / 0.002233468   # +0.2%  lepsj_2_dr < 0.103 and z_photon > 0.3319
        + 0.001450064 * max(0.0, Q.lund_max_lndelta - -0.615738) / 0.02343634   # +0.1%  lund_max_lndelta > -0.6157
        - 0.00128753 * max(0.0, Q.N3_b05 - 0.7591346) / 0.1643393   # -0.1%  N3_b05 > 0.7591
        - 0.001193185 * max(0.0, Q.sj3_mass1 - 36.36236) / 1.189826   # -0.1%  sj3_mass1 > 36.36
        + 0.001184295 * max(0.0, Q.sj3_pair_mass_min - 80.02563) / 0.702582   # +0.1%  sj3_pair_mass_min > 80.03
        - 0.00106898 * max(0.0, 7.0 - Q.n_lund) * max(0.0, 2.591685 - Q.jd_3d_6) / 0.2956722   # -0.1%  n_lund < 7 and jd_3d_6 < 2.592
        - 0.0008558021 * max(0.0, Q.mass - 182.8592) / 1.688248   # -0.1%  mass > 182.9
        - 0.0007672953 * max(0.0, Q.lep_dr - 0.4191372) / 0.007569263   # -0.1%  lep_dr > 0.4191
    )
    return z


def neuron_53(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.889363e-06
    )
    return z


def neuron_54(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.190659e-05
    )
    return z


def neuron_55(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.06694e-05
    )
    return z


def neuron_56(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.072331e-07
    )
    return z


def neuron_57(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.959648e-06
    )
    return z


def neuron_58(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.617151e-06
    )
    return z


def neuron_59(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.029332e-06
    )
    return z


def neuron_60(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.666766e-06
    )
    return z


def neuron_61(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.840559e-06
    )
    return z


def neuron_62(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.422976e-07
    )
    return z


def neuron_63(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.308512e-06
    )
    return z


def neuron_64(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.258629e-06
    )
    return z


def neuron_65(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.791054e-06
    )
    return z


def neuron_66(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.709779e-06
    )
    return z


def neuron_67(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.219511e-06
    )
    return z


def neuron_68(Q):
    # scale S = 14.18; each line: share * term / its average size
    z = 14.18366 * (0.3161182
        - 0.1817006 * Q.n_particles / 39.33801   # -18.2%  n_particles
        - 0.1082881 * max(0.0, 158.2019 - Q.mres_sd_mass_b2z01) / 55.4697   # -10.8%  mres_sd_mass_b2z01 < 158.2
        - 0.1001801 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -10.0%  mass < 164.4
        + 0.07895293 * max(0.0, 118.2746 - Q.mres_sd_mass_b2z01) / 22.65141   # +7.9%  mres_sd_mass_b2z01 < 118.3
        + 0.04558505 * Q.n_photon / 16.03902   # +4.6%  n_photon
        - 0.03930027 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # -3.9%  e3_b2 < 0.0004127
        - 0.0296321 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # -3.0%  lep_iso < 0.4382
        + 0.029246 * max(0.0, 0.05613495 - Q.tau5) / 0.02517241   # +2.9%  tau5 < 0.05613
        - 0.02715305 * max(0.0, 76.1529 - Q.mres_sd_mass_b2z01) / 5.369584   # -2.7%  mres_sd_mass_b2z01 < 76.15
        - 0.02570159 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) / 44.19386   # -2.6%  sj3_pair_mass_min < 80.03
        + 0.0246892 * max(0.0, 110.2019 - Q.mass) / 12.0043   # +2.5%  mass < 110.2
        - 0.01821026 * max(0.0, 0.05966366 - Q.M2_b2) / 0.0281728   # -1.8%  M2_b2 < 0.05966
        + 0.015983 * max(0.0, 0.04607888 - Q.lep_dr) * max(0.0, 0.7828545 - Q.lepsj_2_maxsd0) / 0.02040788   # +1.6%  lep_dr < 0.04608 and lepsj_2_maxsd0 < 0.7829
        + 0.01524365 * max(0.0, Q.z_charged_had - 0.265564) / 0.2499185   # +1.5%  z_charged_had > 0.2656
        - 0.01488458 * max(0.0, Q.mres_pruned_mass - 86.14266) / 22.56548   # -1.5%  mres_pruned_mass > 86.14
        - 0.01480332 * max(0.0, Q.mres_sd_mass_b0z02 - 86.60355) / 17.13177   # -1.5%  mres_sd_mass_b0z02 > 86.6
        + 0.01312784 * max(0.0, Q.mres_sd_mass_b0z02 - 47.97531) / 39.20836   # +1.3%  mres_sd_mass_b0z02 > 47.98
        + 0.01310074 * max(0.0, 0.04607888 - Q.lep_dr) / 0.02662022   # +1.3%  lep_dr < 0.04608
        - 0.01268595 * max(0.0, Q.lep_ptrel - 6.983043) / 5.140178   # -1.3%  lep_ptrel > 6.983
        + 0.01097661 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.lnptrel_25 - -6.691703) / 0.0003839776   # +1.1%  e3_b2 < 0.0004127 and lnptrel_25 > -6.692
        - 0.01081344 * max(0.0, Q.sj3_pair_mass_max - 83.63398) / 16.36149   # -1.1%  sj3_pair_mass_max > 83.63
        - 0.01073744 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 0.7828545 - Q.lepsj_2_maxsd0) / 0.02902708   # -1.1%  z_displaced3 < 0.1062 and lepsj_2_maxsd0 < 0.7829
        + 0.009953736 * max(0.0, 55.13212 - Q.mres_sd_mass_b2z01) / 2.457281   # +1.0%  mres_sd_mass_b2z01 < 55.13
        - 0.008713373 * max(0.0, 0.1339824 - Q.pz_lnd2) / 0.04287826   # -0.9%  pz_lnd2 < 0.134
        + 0.008139157 * max(0.0, Q.jet_abs_eta - 0.5325716) / 0.3171687   # +0.8%  jet_abs_eta > 0.5326
        - 0.007880013 * max(0.0, 1.774856 - Q.D2_b2) / 0.4426441   # -0.8%  D2_b2 < 1.775
        - 0.007299441 * max(0.0, 3.38061 - Q.pair_mean_lnm2) / 0.9985192   # -0.7%  pair_mean_lnm2 < 3.381
        + 0.007193222 * max(0.0, Q.ktd_ln_d34 - -9.359695) / 0.5550767   # +0.7%  ktd_ln_d34 > -9.36
        + 0.007040537 * max(0.0, 3.0 - Q.n_sdz_above_5) / 1.349857   # +0.7%  n_sdz_above_5 < 3
        + 0.00690124 * max(0.0, Q.mres_pruned_mass - 124.3145) / 6.78236   # +0.7%  mres_pruned_mass > 124.3
        + 0.006826055 * max(0.0, 0.7981752 - Q.tau32) / 0.1480334   # +0.7%  tau32 < 0.7982
        - 0.005922079 * max(0.0, Q.sj4_pair_mass_max - 69.83554) * max(0.0, 0.2037349 - Q.dc_3_z) / 2.386338   # -0.6%  sj4_pair_mass_max > 69.84 and dc_3_z < 0.2037
        + 0.0058822 * max(0.0, Q.lep_ptrel - 18.7678) / 2.829169   # +0.6%  lep_ptrel > 18.77
        + 0.004715303 * max(0.0, Q.pz_lnd3 - 0.01024929) / 0.08508837   # +0.5%  pz_lnd3 > 0.01025
        + 0.004578225 * max(0.0, 3.38061 - Q.pair_mean_lnm2) * max(0.0, Q.z_top30_slots - 0.8316085) / 0.1174666   # +0.5%  pair_mean_lnm2 < 3.381 and z_top30_slots > 0.8316
        + 0.004124625 * max(0.0, Q.z_charged_had - 0.265564) * max(0.0, Q.pz_lnkt3 - 0.0) / 0.01381879   # +0.4%  z_charged_had > 0.2656 and pz_lnkt3 > 0
        + 0.003916393 * max(0.0, Q.sj3_pair_mass_max - 128.6605) / 2.418855   # +0.4%  sj3_pair_mass_max > 128.7
        - 0.003827939 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.4%  mass < 90.09
        - 0.003579718 * max(0.0, Q.nca_sj4_pair_mass_2nd - 72.72359) / 3.203001   # -0.4%  nca_sj4_pair_mass_2nd > 72.72
        - 0.003387386 * max(0.0, Q.mres_sd_rg_b2z01 - 0.457968) / 0.03191439   # -0.3%  mres_sd_rg_b2z01 > 0.458
        - 0.003225165 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, Q.n_lund - 10.0) / 0.08342847   # -0.3%  z_displaced3 < 0.1062 and n_lund > 10
        - 0.003205263 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # -0.3%  n_pairs_kt_above_1 < 80
        + 0.002958037 * max(0.0, Q.mres_sd_mass_b0z02 - 134.2023) / 3.925253   # +0.3%  mres_sd_mass_b0z02 > 134.2
        + 0.002884334 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 0.02258554 - Q.dr_29) / 0.0003980039   # +0.3%  z_displaced3 < 0.1062 and dr_29 < 0.02259
        - 0.002830982 * max(0.0, Q.mass_top20 - 130.7093) / 2.260459   # -0.3%  mass_top20 > 130.7
        + 0.002742298 * max(0.0, 117.4867 - Q.mass) * max(0.0, 1.380731 - Q.D2_b2) / 2.980715   # +0.3%  mass < 117.5 and D2_b2 < 1.381
        - 0.002716122 * max(0.0, 56.73313 - Q.sj3_pair_mass_max) / 1.542034   # -0.3%  sj3_pair_mass_max < 56.73
        + 0.002678039 * max(0.0, 0.03259227 - Q.M3) / 0.00464829   # +0.3%  M3 < 0.03259
        - 0.002530259 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.C2_b2 - 0.01891146) / 0.9633021   # -0.3%  mass < 117.5 and C2_b2 > 0.01891
        + 0.002465678 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.sj3_mass1 - 17.31003) / 0.001436151   # +0.2%  e3_b2 < 0.0004127 and sj3_mass1 > 17.31
        + 0.002308609 * max(0.0, 0.7981752 - Q.tau32) * max(0.0, 0.3190414 - Q.sj3_dr_min) / 0.01394611   # +0.2%  tau32 < 0.7982 and sj3_dr_min < 0.319
        + 0.002161345 * max(0.0, Q.lep_ptrel - 6.983043) * max(0.0, 0.1682645 - Q.C2_b2) / 0.5328009   # +0.2%  lep_ptrel > 6.983 and C2_b2 < 0.1683
        - 0.001971216 * max(0.0, Q.sdb_2_n - 12.0) / 1.5393   # -0.2%  sdb_2_n > 12
        - 0.001953151 * max(0.0, 3.0 - Q.sjq_2_2_nch) / 0.2391967   # -0.2%  sjq_2_2_nch < 3
        + 0.001794137 * max(0.0, 0.005938474 - Q.sum_z_dr2_top2) / 0.001014729   # +0.2%  sum_z_dr2_top2 < 0.005938
        - 0.001726385 * max(0.0, 0.05613495 - Q.tau5) * max(0.0, Q.dc_ntag - 0.0) / 0.007175136   # -0.2%  tau5 < 0.05613 and dc_ntag > 0
        + 0.001634886 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.n_lund_kt_above_5 - 2.0) / 0.0001088025   # +0.2%  e3_b2 < 0.0004127 and n_lund_kt_above_5 > 2
        - 0.00156362 * max(0.0, 0.1339824 - Q.pz_lnd2) * max(0.0, Q.sj3_dr13 - 0.4691911) / 0.003699286   # -0.2%  pz_lnd2 < 0.134 and sj3_dr13 > 0.4692
        + 0.001499452 * max(0.0, Q.sj4_pair_mass_max - 69.83554) * max(0.0, Q.nca_sj4_pair2nd_over_mass - 0.4488456) / 0.8081534   # +0.1%  sj4_pair_mass_max > 69.84 and nca_sj4_pair2nd_over_mass > 0.4488
        + 0.00149783 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.ak02_dr23 - 0.3194837) / 1.450982e-05   # +0.1%  e3_b2 < 0.0004127 and ak02_dr23 > 0.3195
        - 0.001021407 * max(0.0, 0.2403736 - Q.sj2_dr) / 0.01094304   # -0.1%  sj2_dr < 0.2404
        - 0.0007336696 * max(0.0, Q.dc_split2_dr - 0.3591078) / 0.004116954   # -0.1%  dc_split2_dr > 0.3591
        - 0.000671124 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.1%  lep_ptrel > 43.21
        + 0.0003357149 * max(0.0, Q.sj3_pair_mass_max - 128.6605) * max(0.0, Q.e4 - 7.184834e-06) / 1.995926e-05   # +0.0%  sj3_pair_mass_max > 128.7 and e4 > 7.185e-06
        + 1.483996e-05 * max(0.0, 130.0968 - Q.mres_sd_mass_b2z01) / 31.31128   # +0.0%  mres_sd_mass_b2z01 < 130.1
    )
    return z


def neuron_69(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.926144e-06
    )
    return z


def neuron_70(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.01601256
    )
    return z


def neuron_71(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.029709e-06
    )
    return z


def neuron_72(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.16412e-06
    )
    return z


def neuron_73(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.69964e-05
    )
    return z


def neuron_74(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.493399e-06
    )
    return z


def neuron_75(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.97384e-06
    )
    return z


def neuron_76(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.542789e-06
    )
    return z


def neuron_77(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.087486e-06
    )
    return z


def neuron_78(Q):
    # scale S = 8.318; each line: share * term / its average size
    z = 8.317814 * (0.02412665
        - 0.1307779 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # -13.1%  lep_z < 0.3397
        - 0.1160041 * max(0.0, 18.7678 - Q.lep_ptrel) / 14.28156   # -11.6%  lep_ptrel < 18.77
        + 0.08405505 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.103005 - Q.lepsj_2_dr) / 0.02325369   # +8.4%  lep_z < 0.3397 and lepsj_2_dr < 0.103
        - 0.07294169 * max(0.0, 0.103005 - Q.lepsj_2_dr) / 0.08254947   # -7.3%  lepsj_2_dr < 0.103
        + 0.05695237 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +5.7%  lep_ptrel < 43.21
        - 0.04578424 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -4.6%  n_lepton < 1
        + 0.0453515 * max(0.0, 9.0 - Q.n_s3d_above_3) / 5.43421   # +4.5%  n_s3d_above_3 < 9
        + 0.03190799 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 1.761851   # +3.2%  lepsj_3_maxsd0 < 2.413
        + 0.02980191 * max(0.0, 0.5083429 - Q.N2_b05) / 0.07626804   # +3.0%  N2_b05 < 0.5083
        - 0.02564681 * max(0.0, 586.9572 - Q.lepsj_3_maxsd0) / 546.6092   # -2.6%  lepsj_3_maxsd0 < 587
        + 0.02146993 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, -7.886954 - Q.ktd_ln_d34) / 20.90232   # +2.1%  lep_ptrel < 18.77 and ktd_ln_d34 < -7.887
        + 0.02143184 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 8.17233e-05 - Q.ecf_g42) / 1.595489e-05   # +2.1%  lep_z < 0.3397 and ecf_g42 < 8.172e-05
        + 0.0201562 * max(0.0, 2.0 - Q.lepsj_3_n_d3) / 1.694207   # +2.0%  lepsj_3_n_d3 < 2
        - 0.01857089 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 22.18431 - Q.mass_2photon) / 237.1798   # -1.9%  lep_ptrel < 18.77 and mass_2photon < 22.18
        + 0.01689432 * max(0.0, 99.0 - Q.n_pairs_kt_above_3) / 39.22243   # +1.7%  n_pairs_kt_above_3 < 99
        + 0.01590308 * max(0.0, Q.mass - 114.0172) / 14.73611   # +1.6%  mass > 114
        - 0.01485241 * max(0.0, -8.400697 - Q.ktd_ln_d34) / 1.139688   # -1.5%  ktd_ln_d34 < -8.401
        + 0.01427939 * max(0.0, 195.0 - Q.n_pairs_kt_above_1) / 37.35244   # +1.4%  n_pairs_kt_above_1 < 195
        + 0.01377522 * max(0.0, Q.lund3_lndelta - -2.817283) / 1.325774   # +1.4%  lund3_lndelta > -2.817
        + 0.01203857 * max(0.0, Q.sdb_2_n - 4.0) / 6.662073   # +1.2%  sdb_2_n > 4
        + 0.01181842 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 749.0582 - Q.sum_pt_top30) / 2343.327   # +1.2%  lep_ptrel < 18.77 and sum_pt_top30 < 749.1
        + 0.0117386 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +1.2%  lep_iso < 1.362
        + 0.01063977 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 0.1570831 - Q.sjf_4_2_z_d3) / 1.931364   # +1.1%  lep_ptrel < 18.77 and sjf_4_2_z_d3 < 0.1571
        - 0.009997697 * max(0.0, Q.z_top15_slots - 0.8008865) / 0.07132828   # -1.0%  z_top15_slots > 0.8009
        + 0.009596024 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.mass_top20 - 86.78877) / 3.685704   # +1.0%  lep_z < 0.3397 and mass_top20 > 86.79
        + 0.009504375 * max(0.0, Q.mres_sd_mass_b0z005 - 122.1) / 9.418671   # +1.0%  mres_sd_mass_b0z005 > 122.1
        - 0.009374552 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, 9.0 - Q.n_neutral_had) / 195.063   # -0.9%  sj4_pair_mass_max < 115.5 and n_neutral_had < 9
        + 0.009171219 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.1956014 - Q.pz_lnd2) / 0.02306375   # +0.9%  lep_z < 0.3397 and pz_lnd2 < 0.1956
        + 0.009155707 * max(0.0, 22.0 - Q.n_neutral) / 4.591883   # +0.9%  n_neutral < 22
        - 0.00913661 * max(0.0, 10.83159 - Q.mass_2charged) / 5.345466   # -0.9%  mass_2charged < 10.83
        - 0.00830711 * max(0.0, Q.mres_sd_mass_b0z005 - 159.9242) / 3.119054   # -0.8%  mres_sd_mass_b0z005 > 159.9
        + 0.008108875 * max(0.0, 1.839882 - Q.mass_2charged) / 0.4692955   # +0.8%  mass_2charged < 1.84
        - 0.006468129 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, Q.ak02_2_z - 0.08487383) / 1.979896   # -0.6%  lep_ptrel < 18.77 and ak02_2_z > 0.08487
        - 0.005989926 * max(0.0, 95.14961 - Q.mass) / 6.371707   # -0.6%  mass < 95.15
        - 0.005879937 * max(0.0, 9.0 - Q.sdb_2_n) * max(0.0, Q.sjq_2_prod_k03 - -0.9998116) / 1.185663   # -0.6%  sdb_2_n < 9 and sjq_2_prod_k03 > -0.9998
        + 0.005621241 * max(0.0, 70.88236 - Q.mass_top40) / 1.88455   # +0.6%  mass_top40 < 70.88
        + 0.005487147 * max(0.0, Q.mass_displaced5 - 9.380468) / 3.157509   # +0.5%  mass_displaced5 > 9.38
        + 0.005046743 * max(0.0, Q.mass_top15 - 83.68425) / 10.75757   # +0.5%  mass_top15 > 83.68
        + 0.00479571 * max(0.0, 9.0 - Q.sdb_2_n) / 1.347003   # +0.5%  sdb_2_n < 9
        + 0.004559271 * max(0.0, Q.n_charged_had - 22.0) / 1.800493   # +0.5%  n_charged_had > 22
        - 0.003786499 * max(0.0, 2.0 - Q.lepsj_3_n_d3) * max(0.0, Q.jet_e - 835.8719) / 277.0793   # -0.4%  lepsj_3_n_d3 < 2 and jet_e > 835.9
        + 0.003548469 * max(0.0, 72.862 - Q.sj4_pair_mass_max) / 6.86631   # +0.4%  sj4_pair_mass_max < 72.86
        + 0.003541538 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 0.1631992 - Q.sj4_dr_min) / 0.889978   # +0.4%  lep_ptrel < 18.77 and sj4_dr_min < 0.1632
        + 0.003323683 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 226.3008 - Q.sip_3d_2) / 106.0959   # +0.3%  n_s3d_above_3 > 4 and sip_3d_2 < 226.3
        - 0.003108403 * max(0.0, Q.mass_displaced5 - 9.380468) * max(0.0, 113.5019 - Q.mass_charged) / 112.3241   # -0.3%  mass_displaced5 > 9.38 and mass_charged < 113.5
        - 0.003020449 * max(0.0, Q.pair_max_lnm2 - 7.347625) / 0.1021051   # -0.3%  pair_max_lnm2 > 7.348
        - 0.002793069 * max(0.0, Q.mass_displaced5 - 21.12768) / 1.493669   # -0.3%  mass_displaced5 > 21.13
        + 0.002228346 * max(0.0, Q.mass_top20 - 130.7093) / 2.260459   # +0.2%  mass_top20 > 130.7
        - 0.002036681 * max(0.0, Q.mass - 164.4374) / 3.038568   # -0.2%  mass > 164.4
        - 0.001302144 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # -0.1%  mass_top10 > 103.5
        - 0.001166571 * max(0.0, Q.z_top15_slots - 0.8008865) * max(0.0, Q.z_displaced5 - 0.2184442) / 0.001733934   # -0.1%  z_top15_slots > 0.8009 and z_displaced5 > 0.2184
        + 0.001151706 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, Q.z_top50_slots - 0.9572293) / 0.04314808   # +0.1%  n_s3d_above_3 > 4 and z_top50_slots > 0.9572
    )
    return z


def neuron_79(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.602388e-05
    )
    return z


def neuron_80(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.707706e-06
    )
    return z


def neuron_81(Q):
    # scale S = 6.864; each line: share * term / its average size
    z = 6.864494 * (0.001856389
        - 0.1788559 * max(0.0, 158.2019 - Q.mres_sd_mass_b2z01) / 55.4697   # -17.9%  mres_sd_mass_b2z01 < 158.2
        + 0.09386681 * max(0.0, 115.0388 - Q.mres_sd_mass_b2z01) / 20.62743   # +9.4%  mres_sd_mass_b2z01 < 115
        + 0.07335793 * max(0.0, 125.2732 - Q.mres_sd_mass_b2z01) / 27.56259   # +7.3%  mres_sd_mass_b2z01 < 125.3
        - 0.05938556 * max(0.0, 91.15481 - Q.mres_sd_mass_b2z01) / 9.5029   # -5.9%  mres_sd_mass_b2z01 < 91.15
        + 0.05357465 * max(0.0, 0.01162881 - Q.ecf_g31) / 0.005313715   # +5.4%  ecf_g31 < 0.01163
        + 0.03900135 * max(0.0, Q.z_displaced3 - 0.03624058) / 0.08737704   # +3.9%  z_displaced3 > 0.03624
        - 0.03677113 * max(0.0, 0.005262883 - Q.lepsj_3_dr) / 0.003463974   # -3.7%  lepsj_3_dr < 0.005263
        - 0.03381629 * max(0.0, 226.3008 - Q.sip_3d_2) * max(0.0, 6.771002 - Q.jd_3d_5) / 605.5688   # -3.4%  sip_3d_2 < 226.3 and jd_3d_5 < 6.771
        + 0.02830381 * max(0.0, Q.n_s3d_above_3 - 2.0) / 2.23187   # +2.8%  n_s3d_above_3 > 2
        + 0.02674159 * max(0.0, 4.004982 - Q.jd_3d_5) / 1.451954   # +2.7%  jd_3d_5 < 4.005
        - 0.02460361 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.z_charged_had - 0.265564) / 0.02462507   # -2.5%  z_displaced3 > 0.03624 and z_charged_had > 0.2656
        - 0.02458684 * max(0.0, 5.0 - Q.sdb_0_n) / 3.42113   # -2.5%  sdb_0_n < 5
        + 0.0245632 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # +2.5%  sip_3d_2 < 226.3
        + 0.02339315 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # +2.3%  max_abs_d0 < 10.52
        - 0.0222949 * max(0.0, 0.01162881 - Q.ecf_g31) * max(0.0, Q.sjq_2_prod_k05 - -0.5057096) / 0.00247391   # -2.2%  ecf_g31 < 0.01163 and sjq_2_prod_k05 > -0.5057
        + 0.02107517 * max(0.0, 3.0 - Q.sdb_0_n) / 1.623507   # +2.1%  sdb_0_n < 3
        - 0.02003033 * max(0.0, 0.06117886 - Q.sum_zz_dr2) / 0.02616533   # -2.0%  sum_zz_dr2 < 0.06118
        - 0.01895511 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 6.43167 - Q.jd_3d_4) / 19.90242   # -1.9%  max_abs_d0 < 10.52 and jd_3d_4 < 6.432
        - 0.01814397 * max(0.0, 15.0 - Q.n_dr_0p4_up) / 9.48698   # -1.8%  n_dr_0p4_up < 15
        + 0.01559301 * max(0.0, 0.005262883 - Q.lepsj_3_dr) * max(0.0, 2.953464 - Q.jd_3d_4) / 0.002212736   # +1.6%  lepsj_3_dr < 0.005263 and jd_3d_4 < 2.953
        + 0.01523533 * max(0.0, 0.06416437 - Q.z_displaced3) * max(0.0, 1.0 - Q.dc_2_n_lep) / 0.02413578   # +1.5%  z_displaced3 < 0.06416 and dc_2_n_lep < 1
        + 0.01461917 * max(0.0, 76.1529 - Q.mres_sd_mass_b2z01) / 5.369584   # +1.5%  mres_sd_mass_b2z01 < 76.15
        + 0.01238664 * max(0.0, 0.04510459 - Q.sdb_4_z) * max(0.0, 0.004135872 - Q.lep_z) / 8.841111e-05   # +1.2%  sdb_4_z < 0.0451 and lep_z < 0.004136
        + 0.01012551 * max(0.0, Q.mass_top40 - 115.7429) / 10.91182   # +1.0%  mass_top40 > 115.7
        - 0.009969493 * max(0.0, 1.777286 - Q.mass_displaced3) / 0.7377974   # -1.0%  mass_displaced3 < 1.777
        + 0.009155841 * max(0.0, Q.n_s3d_above_10 - 3.0) / 0.9107233   # +0.9%  n_s3d_above_10 > 3
        + 0.008304386 * max(0.0, 0.01162881 - Q.ecf_g31) * max(0.0, Q.sjq_2_sumabs_k1 - 0.1588551) / 0.001231486   # +0.8%  ecf_g31 < 0.01163 and sjq_2_sumabs_k1 > 0.1589
        - 0.007655423 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 7.0 - Q.n_s3d_above_3) / 0.1116745   # -0.8%  z_displaced3 > 0.03624 and n_s3d_above_3 < 7
        + 0.006831864 * max(0.0, Q.n_s3d_above_3 - 2.0) * max(0.0, 4.0 - Q.n_lund_kt_above_5) / 4.131823   # +0.7%  n_s3d_above_3 > 2 and n_lund_kt_above_5 < 4
        - 0.006501907 * max(0.0, Q.sv_n - 1.0) / 0.43993   # -0.7%  sv_n > 1
        - 0.006464536 * max(0.0, Q.sjf_4_n2disp - 1.0) / 0.2405067   # -0.6%  sjf_4_n2disp > 1
        + 0.00643821 * max(0.0, Q.n_s3d_above_3 - 6.0) / 0.6254467   # +0.6%  n_s3d_above_3 > 6
        - 0.006046638 * max(0.0, Q.n_s3d_above_3 - 2.0) * max(0.0, 0.4982257 - Q.z_photon) / 0.5738664   # -0.6%  n_s3d_above_3 > 2 and z_photon < 0.4982
        + 0.005356005 * max(0.0, 0.01162881 - Q.ecf_g31) * max(0.0, Q.n_s3d_above_3 - 4.0) / 0.005782324   # +0.5%  ecf_g31 < 0.01163 and n_s3d_above_3 > 4
        + 0.004962562 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.n_lund_kt_above_1 - 2.0) / 0.2755117   # +0.5%  z_displaced3 > 0.03624 and n_lund_kt_above_1 > 2
        + 0.004943512 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 1262.672 - Q.jd_sum_abs_sd0_top3) / 63.42777   # +0.5%  z_displaced3 > 0.03624 and jd_sum_abs_sd0_top3 < 1263
        - 0.004836394 * Q.lepsj_2_n_d3 / 0.6197967   # -0.5%  lepsj_2_n_d3
        + 0.004298303 * max(0.0, 158.2019 - Q.mres_sd_mass_b2z01) * max(0.0, Q.mass_2charged - 24.4079) / 104.2811   # +0.4%  mres_sd_mass_b2z01 < 158.2 and mass_2charged > 24.41
        - 0.004288579 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # -0.4%  mass_top50 > 161.1
        + 0.004150335 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # +0.4%  n_pairs_kt_above_3 < 28
        + 0.00319292 * max(0.0, 59.5457 - Q.sj4_pair_mass_max) / 2.921503   # +0.3%  sj4_pair_mass_max < 59.55
        - 0.00276266 * max(0.0, Q.sjf_2_1_n_d3 - 5.0) / 0.1972067   # -0.3%  sjf_2_1_n_d3 > 5
        + 0.002693763 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # +0.3%  n_pairs_kt_above_1 < 58
        + 0.001865664 * max(0.0, 0.06074238 - Q.tau1) / 0.001113326   # +0.2%  tau1 < 0.06074
    )
    return z


def neuron_82(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.942075e-06
    )
    return z


def neuron_83(Q):
    # scale S = 7.69; each line: share * term / its average size
    z = 7.690351 * (0.1568895
        - 0.1146835 * max(0.0, Q.mass - 79.47361) / 38.31536   # -11.5%  mass > 79.47
        - 0.08652666 * max(0.0, Q.mres_sd_mass_b2z01 - 81.19466) / 31.20897   # -8.7%  mres_sd_mass_b2z01 > 81.19
        + 0.07297353 * max(0.0, 0.0925671 - Q.sum_z_dr2_top50) / 0.05494881   # +7.3%  sum_z_dr2_top50 < 0.09257
        - 0.07164922 * max(0.0, 139.2565 - Q.mres_sd_mass_b2z01) / 38.89774   # -7.2%  mres_sd_mass_b2z01 < 139.3
        + 0.06595765 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) / 36.55147   # +6.6%  sj4_pair_mass_max < 115.5
        + 0.04735274 * max(0.0, Q.mass_top40 - 78.33213) / 34.93312   # +4.7%  mass_top40 > 78.33
        - 0.04024811 * max(0.0, 1.139376 - Q.D3_b05) / 0.7023584   # -4.0%  D3_b05 < 1.139
        - 0.03497228 * max(0.0, 6.0 - Q.n_s3d_above_10) / 3.631173   # -3.5%  n_s3d_above_10 < 6
        - 0.03266538 * max(0.0, 3.53193e-06 - Q.e4) / 2.734524e-06   # -3.3%  e4 < 3.532e-06
        + 0.03197959 * max(0.0, Q.mres_sd_mass_b2z01 - 130.0968) / 7.118079   # +3.2%  mres_sd_mass_b2z01 > 130.1
        + 0.03196947 * max(0.0, 13.03663 - Q.mass_displaced3) * max(0.0, 6.771002 - Q.jd_3d_5) / 38.55232   # +3.2%  mass_displaced3 < 13.04 and jd_3d_5 < 6.771
        + 0.02849207 * max(0.0, Q.tau5 - 0.01467699) / 0.01818842   # +2.8%  tau5 > 0.01468
        + 0.02327135 * max(0.0, Q.mass - 117.4867) / 13.09248   # +2.3%  mass > 117.5
        - 0.02148202 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 6.771002 - Q.jd_3d_5) / 24.1852   # -2.1%  max_abs_d0 < 10.52 and jd_3d_5 < 6.771
        - 0.02042759 * max(0.0, Q.mass - 95.14961) / 26.15323   # -2.0%  mass > 95.15
        - 0.01999739 * max(0.0, Q.mass_top40 - 78.33213) * max(0.0, 0.1275041 - Q.lep_z) / 3.263902   # -2.0%  mass_top40 > 78.33 and lep_z < 0.1275
        + 0.01960007 * max(0.0, 0.06416437 - Q.z_displaced3) / 0.02655242   # +2.0%  z_displaced3 < 0.06416
        + 0.01954836 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # +2.0%  max_abs_d0 < 10.52
        - 0.01952853 * max(0.0, 162.7874 - Q.mass_top30) / 59.65527   # -2.0%  mass_top30 < 162.8
        - 0.01859641 * max(0.0, Q.mass_top15 - 77.22442) / 13.98893   # -1.9%  mass_top15 > 77.22
        - 0.01819069 * max(0.0, 0.02160244 - Q.M3_b2) / 0.009628213   # -1.8%  M3_b2 < 0.0216
        - 0.0170634 * max(0.0, 0.5109872 - Q.sdb_2_z) / 0.2120326   # -1.7%  sdb_2_z < 0.511
        - 0.01523257 * max(0.0, 24.0 - Q.n_pt_above_5) / 4.457557   # -1.5%  n_pt_above_5 < 24
        - 0.01369531 * max(0.0, 185.1889 - Q.jd_sum_abs_sd0_top5) / 79.19909   # -1.4%  jd_sum_abs_sd0_top5 < 185.2
        + 0.01215947 * max(0.0, 6.0 - Q.n_s3d_above_10) * max(0.0, Q.lne_0 - 4.380463) / 3.270558   # +1.2%  n_s3d_above_10 < 6 and lne_0 > 4.38
        - 0.01136119 * max(0.0, 75.78208 - Q.sj4_pair_mass_max) / 8.111726   # -1.1%  sj4_pair_mass_max < 75.78
        + 0.01012974 * max(0.0, 139.2565 - Q.mres_sd_mass_b2z01) * max(0.0, 1.0 - Q.dc_2_n_lep) / 32.88249   # +1.0%  mres_sd_mass_b2z01 < 139.3 and dc_2_n_lep < 1
        + 0.008960677 * max(0.0, 7.028704 - Q.sip_3d_1) / 1.023735   # +0.9%  sip_3d_1 < 7.029
        - 0.007895157 * max(0.0, 75.78208 - Q.sj4_pair_mass_max) * max(0.0, 0.07303924 - Q.sjf_4_3_z_d3) / 0.535129   # -0.8%  sj4_pair_mass_max < 75.78 and sjf_4_3_z_d3 < 0.07304
        + 0.007818362 * max(0.0, 0.1530389 - Q.sdb_2_z) / 0.01282076   # +0.8%  sdb_2_z < 0.153
        + 0.007315812 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 172.888 - Q.jd_3d_4) / 66.43449   # +0.7%  n_s3d_above_3 > 6 and jd_3d_4 < 172.9
        + 0.006900268 * max(0.0, Q.mass_top40 - 155.8928) / 2.693823   # +0.7%  mass_top40 > 155.9
        + 0.005917721 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # +0.6%  n_s3d_above_3 < 3
        + 0.00533111 * max(0.0, 10.0 - Q.sdb_2_n) / 1.820263   # +0.5%  sdb_2_n < 10
        - 0.005033321 * max(0.0, 0.258375 - Q.N2) / 0.01851814   # -0.5%  N2 < 0.2584
        + 0.004042147 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 0.4866692 - Q.sj3_pairmin_over_m) / 0.118201   # +0.4%  n_s3d_above_3 > 6 and sj3_pairmin_over_m < 0.4867
        + 0.00385687 * max(0.0, Q.pair_max_lnm2 - 7.524613) / 0.07146798   # +0.4%  pair_max_lnm2 > 7.525
        + 0.00302409 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 1292.625 - Q.jd_sum_abs_sd0_top5) / 301.995   # +0.3%  n_s3d_above_3 > 6 and jd_sum_abs_sd0_top5 < 1293
        - 0.002945643 * max(0.0, Q.mres_sd_mass_b2z01 - 158.2019) / 3.171488   # -0.3%  mres_sd_mass_b2z01 > 158.2
        - 0.002869583 * max(0.0, 0.1530389 - Q.sdb_2_z) * max(0.0, 4.004982 - Q.jd_3d_5) / 0.01187781   # -0.3%  sdb_2_z < 0.153 and jd_3d_5 < 4.005
        - 0.002672285 * max(0.0, 162.7874 - Q.mass_top30) * max(0.0, Q.mass_displaced5 - 9.380468) / 137.2553   # -0.3%  mass_top30 < 162.8 and mass_displaced5 > 9.38
        + 0.002271137 * max(0.0, 0.03767806 - Q.sv_2_z) * max(0.0, Q.max_dr - 0.8033751) / 0.002523979   # +0.2%  sv_2_z < 0.03768 and max_dr > 0.8034
        - 0.001918845 * max(0.0, 0.06074238 - Q.tau1) / 0.001113326   # -0.2%  tau1 < 0.06074
        - 0.001472668 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # -0.1%  mass_top10 > 103.5
    )
    return z


def neuron_84(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.427514e-05
    )
    return z


def neuron_85(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.185525e-07
    )
    return z


def neuron_86(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.709903e-05
    )
    return z


def neuron_87(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.619162e-06
    )
    return z


def neuron_88(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.99698e-07
    )
    return z


def neuron_89(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.735374e-06
    )
    return z


def neuron_90(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.001229865
    )
    return z


def neuron_91(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.992639e-06
    )
    return z


def neuron_92(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.724736e-06
    )
    return z


def neuron_93(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.46439e-07
    )
    return z


def neuron_94(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.322172e-06
    )
    return z


def neuron_95(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.369519e-07
    )
    return z


def neuron_96(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.278031e-07
    )
    return z


def neuron_97(Q):
    # scale S = 25.45; each line: share * term / its average size
    z = 25.44852 * (-0.07217458
        - 0.1179393 * max(0.0, Q.mres_sd_mass_b0z005 - 76.40585) / 35.53887   # -11.8%  mres_sd_mass_b0z005 > 76.41
        + 0.09051693 * Q.sj3_pairmax_over_m / 0.807958   # +9.1%  sj3_pairmax_over_m
        + 0.07808938 * max(0.0, Q.mres_sd_mass_b0z005 - 91.82593) / 24.37401   # +7.8%  mres_sd_mass_b0z005 > 91.83
        - 0.07580807 * max(0.0, 171.3819 - Q.mres_pruned_mass) / 78.34654   # -7.6%  mres_pruned_mass < 171.4
        + 0.05951495 * max(0.0, 182.8592 - Q.mass) / 69.61635   # +6.0%  mass < 182.9
        - 0.05884293 * max(0.0, 149.0507 - Q.mass) / 39.08704   # -5.9%  mass < 149.1
        + 0.05467635 * max(0.0, 124.3145 - Q.mres_pruned_mass) / 36.36896   # +5.5%  mres_pruned_mass < 124.3
        - 0.04938132 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -4.9%  mass < 164.4
        + 0.04678819 * max(0.0, Q.mres_sd_mass_b0z005 - 53.57509) / 55.22856   # +4.7%  mres_sd_mass_b0z005 > 53.58
        + 0.03559604 * max(0.0, 39.09615 - Q.mass_displaced3) / 31.84697   # +3.6%  mass_displaced3 < 39.1
        + 0.03417829 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) / 26.86577   # +3.4%  sj3_pair_mass_max < 114.8
        - 0.0183088 * max(0.0, 112.1947 - Q.mres_pruned_mass) / 27.53635   # -1.8%  mres_pruned_mass < 112.2
        - 0.01821367 * max(0.0, Q.mres_sd_mass_b0z005 - 97.29337) / 20.97048   # -1.8%  mres_sd_mass_b0z005 > 97.29
        - 0.01742811 * max(0.0, Q.mres_pruned_mass - 70.04065) / 33.4731   # -1.7%  mres_pruned_mass > 70.04
        - 0.01740627 * max(0.0, 6.0 - Q.n_sd0_above_3) / 3.16582   # -1.7%  n_sd0_above_3 < 6
        + 0.0132266 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # +1.3%  e3_b2 < 0.000791
        - 0.0125537 * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0001737301   # -1.3%  e3_b2 < 0.0002537
        + 0.01137768 * max(0.0, 126.8853 - Q.mass_top40) / 23.76162   # +1.1%  mass_top40 < 126.9
        + 0.01078724 * max(0.0, Q.mres_sd_mass_b0z005 - 122.1) / 9.418671   # +1.1%  mres_sd_mass_b0z005 > 122.1
        - 0.01052114 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # -1.1%  sip_3d_2 < 447.1
        - 0.01047641 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # -1.0%  max_abs_d0 < 5.812
        + 0.009699684 * max(0.0, 18.80005 - Q.mass_displaced3) / 13.42744   # +1.0%  mass_displaced3 < 18.8
        + 0.009086682 * max(0.0, 5.8125 - Q.max_abs_d0) * max(0.0, 4.004982 - Q.jd_3d_5) / 5.484604   # +0.9%  max_abs_d0 < 5.812 and jd_3d_5 < 4.005
        + 0.008413674 * max(0.0, 100.4835 - Q.mass) / 8.115061   # +0.8%  mass < 100.5
        + 0.008192458 * max(0.0, 9.0 - Q.n_s3d_above_3) / 5.43421   # +0.8%  n_s3d_above_3 < 9
        - 0.00775928 * max(0.0, 122.0585 - Q.mass_top50) / 18.91058   # -0.8%  mass_top50 < 122.1
        - 0.007413767 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 18.7678 - Q.lep_ptrel) / 453.8353   # -0.7%  mass_displaced3 < 39.1 and lep_ptrel < 18.77
        - 0.007369202 * max(0.0, 172.888 - Q.jd_3d_4) / 152.4729   # -0.7%  jd_3d_4 < 172.9
        - 0.006621658 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -0.7%  lep_z < 0.2214
        - 0.006405793 * max(0.0, 182.8592 - Q.mass) * max(0.0, Q.n_s3d_above_3 - 1.0) / 168.7405   # -0.6%  mass < 182.9 and n_s3d_above_3 > 1
        + 0.00580072 * max(0.0, 111.0 - Q.n_pairs_kt_above_3) / 47.95244   # +0.6%  n_pairs_kt_above_3 < 111
        + 0.00541823 * max(0.0, 0.2113485 - Q.pz_lnd3) / 0.1202167   # +0.5%  pz_lnd3 < 0.2113
        + 0.004982002 * max(0.0, 122.0585 - Q.mass_top50) * max(0.0, 1.0 - Q.ak02_2_n_lep) / 16.31653   # +0.5%  mass_top50 < 122.1 and ak02_2_n_lep < 1
        + 0.004531847 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 0.18807 - Q.pz_lnd3) / 3.368618   # +0.5%  mass_displaced3 < 39.1 and pz_lnd3 < 0.1881
        - 0.00448658 * max(0.0, 9.621843e-05 - Q.e3_b2) / 4.881439e-05   # -0.4%  e3_b2 < 9.622e-05
        + 0.004478384 * max(0.0, Q.sd_mass - 119.4443) / 9.070113   # +0.4%  sd_mass > 119.4
        + 0.004056168 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) / 103.3282   # +0.4%  n_pairs_kt_above_1 < 325
        + 0.003766221 * max(0.0, 5.0 - Q.sjf_3_1_n_d5) / 3.728713   # +0.4%  sjf_3_1_n_d5 < 5
        + 0.003418755 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, 1.0 - Q.dc_1_n_lep) / 249.927   # +0.3%  sip_3d_2 < 447.1 and dc_1_n_lep < 1
        - 0.00330685 * max(0.0, Q.mres_sd_mass_b0z005 - 159.9242) / 3.119054   # -0.3%  mres_sd_mass_b0z005 > 159.9
        + 0.003268189 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.jet_charge_k03 - -0.05990128) / 5.503613e-05   # +0.3%  e3_b2 < 0.0002537 and jet_charge_k03 > -0.0599
        - 0.003262381 * max(0.0, Q.nca_sj4_pair_mass_2nd - 72.72359) / 3.203001   # -0.3%  nca_sj4_pair_mass_2nd > 72.72
        - 0.003173375 * max(0.0, 0.2740506 - Q.sdb_2_z) / 0.05180874   # -0.3%  sdb_2_z < 0.2741
        + 0.002991993 * max(0.0, Q.lund3_lndelta - -1.780944) / 0.4380167   # +0.3%  lund3_lndelta > -1.781
        + 0.00293642 * max(0.0, 7.746064 - Q.ak02_min12_jp) / 4.717722   # +0.3%  ak02_min12_jp < 7.746
        + 0.002402177 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +0.2%  mass < 95.15
        - 0.002326308 * max(0.0, 122.0585 - Q.mass_top50) * max(0.0, 0.125305 - Q.sjq_2_prod_k05) / 3.922517   # -0.2%  mass_top50 < 122.1 and sjq_2_prod_k05 < 0.1253
        - 0.002235994 * max(0.0, Q.N2_b05 - 0.4353632) / 0.03052074   # -0.2%  N2_b05 > 0.4354
        - 0.002222551 * max(0.0, 100.4835 - Q.mass) * max(0.0, 6.185635 - Q.lep_iso) / 42.98093   # -0.2%  mass < 100.5 and lep_iso < 6.186
        + 0.001971476 * max(0.0, Q.nca_sj4_pair_mass_2nd - 83.41384) / 1.649755   # +0.2%  nca_sj4_pair_mass_2nd > 83.41
        + 0.001946826 * max(0.0, 100.4835 - Q.mass) * max(0.0, 1.477152 - Q.lep_ptrel) / 8.351593   # +0.2%  mass < 100.5 and lep_ptrel < 1.477
        + 0.001674284 * max(0.0, 7.075642 - Q.pair_max_lnm2) / 0.6225895   # +0.2%  pair_max_lnm2 < 7.076
        + 0.001635364 * max(0.0, Q.lep_ptrel - 3.53503) / 6.079822   # +0.2%  lep_ptrel > 3.535
        + 0.001555285 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, 0.3709098 - Q.sdb_2_z) / 1.803054e-05   # +0.2%  e3_b2 < 0.0002537 and sdb_2_z < 0.3709
        + 0.001198232 * max(0.0, 0.2740506 - Q.sdb_2_z) * max(0.0, 0.1348361 - Q.pz_lnkt1) / 0.002128601   # +0.1%  sdb_2_z < 0.2741 and pz_lnkt1 < 0.1348
        - 0.001160777 * max(0.0, 3.734077 - Q.lund_max_lnkt) / 0.2313255   # -0.1%  lund_max_lnkt < 3.734
        - 0.001153974 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) * max(0.0, Q.n_photon - 12.0) / 94.74492   # -0.1%  sj3_pair_mass_max < 114.8 and n_photon > 12
        - 0.0009692193 * max(0.0, 71.80762 - Q.mass_top50) / 1.927765   # -0.1%  mass_top50 < 71.81
        + 0.0008947261 * max(0.0, 122.0585 - Q.mass_top50) * max(0.0, Q.sum_e - 926.2598) / 2366.251   # +0.1%  mass_top50 < 122.1 and sum_e > 926.3
        - 0.0007898779 * max(0.0, Q.z_displaced5 - 0.1339658) / 0.03671738   # -0.1%  z_displaced5 > 0.134
        + 0.0007322481 * max(0.0, 0.0007909605 - Q.e3_b2) * max(0.0, Q.sjq_3_sumabs_k1 - 0.4372817) / 6.220365e-05   # +0.1%  e3_b2 < 0.000791 and sjq_3_sumabs_k1 > 0.4373
        + 0.0006865057 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # +0.1%  sj2_mass2 < 1.852
        + 0.000620596 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.D2 - 3.038341) / 11.93622   # +0.1%  mass_displaced3 < 39.1 and D2 > 3.038
        + 0.0005635014 * max(0.0, 63.7508 - Q.sj3_pair_mass_max) / 2.405567   # +0.1%  sj3_pair_mass_max < 63.75
        - 0.0005528244 * max(0.0, Q.mres_sd_prong_mass1 - 69.04524) / 2.206937   # -0.1%  mres_sd_prong_mass1 > 69.05
        - 0.0002355386 * max(0.0, 126.8853 - Q.mass_top40) * max(0.0, Q.n_muon - 1.0) / 0.4724669   # -0.0%  mass_top40 < 126.9 and n_muon > 1
    )
    return z


def neuron_98(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.990169e-07
    )
    return z


def neuron_99(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.001934242
    )
    return z


def neuron_100(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.264077e-06
    )
    return z


def neuron_101(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.878263e-06
    )
    return z


def neuron_102(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.490515e-06
    )
    return z


def neuron_103(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.41438e-06
    )
    return z


def neuron_104(Q):
    # scale S = 6.755; each line: share * term / its average size
    z = 6.755028 * (-0.2741644
        + 0.1213453 * max(0.0, Q.mres_sd_mass_b2z01 - 76.1529) / 35.12032   # +12.1%  mres_sd_mass_b2z01 > 76.15
        - 0.09470373 * max(0.0, Q.mres_sd_mass_b2z01 - 86.09683) / 27.65834   # -9.5%  mres_sd_mass_b2z01 > 86.1
        + 0.06441517 * max(0.0, -1.352792 - Q.pair_mean_lndelta) / 0.8695512   # +6.4%  pair_mean_lndelta < -1.353
        + 0.06261276 * max(0.0, 58.0 - Q.n_pt_above_1) / 20.6963   # +6.3%  n_pt_above_1 < 58
        + 0.05154921 * max(0.0, Q.mass_displaced3 - 3.208089) / 6.234439   # +5.2%  mass_displaced3 > 3.208
        + 0.04821857 * max(0.0, 117.4867 - Q.mass) / 15.64803   # +4.8%  mass < 117.5
        + 0.04241782 * max(0.0, 0.2827395 - Q.pz_lnd0) / 0.1575044   # +4.2%  pz_lnd0 < 0.2827
        + 0.033565 * max(0.0, 6.0 - Q.n_s3d_above_10) / 3.631173   # +3.4%  n_s3d_above_10 < 6
        - 0.03291664 * max(0.0, Q.mass_displaced3 - 6.341631) / 5.164503   # -3.3%  mass_displaced3 > 6.342
        + 0.0301383 * max(0.0, 18.0 - Q.sjq_2_1_nch) / 6.63145   # +3.0%  sjq_2_1_nch < 18
        + 0.0245047 * max(0.0, 15.0 - Q.sjq_2_2_nch) / 8.033983   # +2.5%  sjq_2_2_nch < 15
        + 0.02299826 * max(0.0, 58.0 - Q.n_pt_above_1) * max(0.0, 0.4490565 - Q.ak02_dr13) / 6.760883   # +2.3%  n_pt_above_1 < 58 and ak02_dr13 < 0.4491
        - 0.0195001 * max(0.0, 1.914834 - Q.dc_tag_2nd) / 1.481015   # -2.0%  dc_tag_2nd < 1.915
        - 0.01884363 * max(0.0, 149.0507 - Q.mass) * max(0.0, 1401.904 - Q.sum_e) / 20921.9   # -1.9%  mass < 149.1 and sum_e < 1402
        - 0.01799528 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # -1.8%  n_pairs_kt_above_3 < 34
        + 0.0173608 * max(0.0, 135.5368 - Q.mass_top50) / 28.82074   # +1.7%  mass_top50 < 135.5
        - 0.01723392 * max(0.0, Q.mres_sd_mass_b0z005 - 91.82593) / 24.37401   # -1.7%  mres_sd_mass_b0z005 > 91.83
        - 0.01719775 * max(0.0, 14.23948 - Q.ak02_min12_jp) / 10.12039   # -1.7%  ak02_min12_jp < 14.24
        + 0.01718275 * max(0.0, 99.20396 - Q.mass_charged) / 37.01153   # +1.7%  mass_charged < 99.2
        - 0.01557876 * max(0.0, 0.05398263 - Q.tau3) / 0.0141203   # -1.6%  tau3 < 0.05398
        - 0.01365017 * max(0.0, 58.0 - Q.n_pt_above_1) * max(0.0, Q.pair_max_lnkt - 1.555555) / 24.2913   # -1.4%  n_pt_above_1 < 58 and pair_max_lnkt > 1.556
        + 0.01352068 * max(0.0, 58.0 - Q.n_pt_above_1) * max(0.0, 1.0 - Q.lepsj_2_n_d3) / 16.6829   # +1.4%  n_pt_above_1 < 58 and lepsj_2_n_d3 < 1
        - 0.01263154 * max(0.0, 4.516968 - Q.sjf_2_2_max3d) * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0008440434   # -1.3%  sjf_2_2_max3d < 4.517 and e3_b2 < 0.000791
        + 0.01242163 * max(0.0, 149.0507 - Q.mass) / 39.08704   # +1.2%  mass < 149.1
        + 0.01240156 * max(0.0, Q.pair_max_lnm2 - 5.908788) / 0.8228665   # +1.2%  pair_max_lnm2 > 5.909
        + 0.0117588 * max(0.0, 0.0230226 - Q.tau5) / 0.00273951   # +1.2%  tau5 < 0.02302
        - 0.01090329 * max(0.0, Q.mass_top5 - 17.63354) / 26.12147   # -1.1%  mass_top5 > 17.63
        - 0.01013653 * max(0.0, 5.245219 - Q.sjf_3_2_max3d) / 1.779833   # -1.0%  sjf_3_2_max3d < 5.245
        + 0.01002338 * max(0.0, 61.73838 - Q.sjf_3_2_maxsd0) / 41.35113   # +1.0%  sjf_3_2_maxsd0 < 61.74
        + 0.009550504 * max(0.0, 16.96512 - Q.sjf_2_2_max3d) / 8.191057   # +1.0%  sjf_2_2_max3d < 16.97
        + 0.009276666 * max(0.0, Q.z_top3_slots - 0.5395924) / 0.0406864   # +0.9%  z_top3_slots > 0.5396
        - 0.008095313 * max(0.0, 99.20396 - Q.mass_charged) * max(0.0, 1.0 - Q.ak02_2_n_lep) / 31.82896   # -0.8%  mass_charged < 99.2 and ak02_2_n_lep < 1
        + 0.007055007 * max(0.0, 0.1798521 - Q.C2) / 0.04587768   # +0.7%  C2 < 0.1799
        + 0.006235849 * max(0.0, Q.tau21_b2 - 0.3898586) / 0.04953334   # +0.6%  tau21_b2 > 0.3899
        - 0.006203959 * max(0.0, 146.0 - Q.n_pairs_kt_above_1) / 20.10573   # -0.6%  n_pairs_kt_above_1 < 146
        - 0.005847619 * max(0.0, Q.lep_ptrel - 6.983043) * max(0.0, 0.2385164 - Q.ak02_dr23) / 0.7252318   # -0.6%  lep_ptrel > 6.983 and ak02_dr23 < 0.2385
        + 0.005775145 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, 1.0 - Q.kt2_2_n_lep) / 3.68679   # +0.6%  n_pairs_kt_above_3 < 34 and kt2_2_n_lep < 1
        + 0.00572428 * max(0.0, Q.mres_sd_mass_b0z005 - 91.82593) * max(0.0, 3.0 - Q.sv_1_n) / 40.5646   # +0.6%  mres_sd_mass_b0z005 > 91.83 and sv_1_n < 3
        - 0.005560018 * max(0.0, 3.32421 - Q.sjf_2_1_max3d) / 0.386295   # -0.6%  sjf_2_1_max3d < 3.324
        - 0.00551802 * max(0.0, 0.2827395 - Q.pz_lnd0) * max(0.0, Q.n_pairs_kt_above_10 - 1.0) / 1.06713   # -0.6%  pz_lnd0 < 0.2827 and n_pairs_kt_above_10 > 1
        - 0.005166143 * max(0.0, 0.4501484 - Q.z_charged_had) / 0.04690479   # -0.5%  z_charged_had < 0.4501
        - 0.004580434 * max(0.0, 58.0 - Q.n_pt_above_1) * max(0.0, Q.sjf_2_1_n_d5 - 1.0) / 19.38064   # -0.5%  n_pt_above_1 < 58 and sjf_2_1_n_d5 > 1
        - 0.004240199 * max(0.0, 0.4680886 - Q.tau32_b2) / 0.05978538   # -0.4%  tau32_b2 < 0.4681
        - 0.0040029 * max(0.0, Q.sjq_2_sumabs_k03 - 0.6667228) / 0.2276103   # -0.4%  sjq_2_sumabs_k03 > 0.6667
        + 0.00361836 * max(0.0, Q.mass_displaced3 - 3.208089) * max(0.0, 0.1014193 - Q.ak02_3_z) / 0.4174877   # +0.4%  mass_displaced3 > 3.208 and ak02_3_z < 0.1014
        - 0.003600138 * max(0.0, Q.sjf_2_1_z_d3 - 0.1149585) / 0.03815928   # -0.4%  sjf_2_1_z_d3 > 0.115
        - 0.00321505 * max(0.0, 18.0 - Q.sjq_2_1_nch) * max(0.0, Q.dr02 - 0.04115773) / 0.9302163   # -0.3%  sjq_2_1_nch < 18 and dr02 > 0.04116
        + 0.003106941 * max(0.0, Q.mass_top30 - 126.5007) / 5.381458   # +0.3%  mass_top30 > 126.5
        + 0.002731921 * max(0.0, 2.0 - Q.sjq_2_2_nch) / 0.09739333   # +0.3%  sjq_2_2_nch < 2
        + 0.002409353 * max(0.0, Q.nca_sj4_pair_mass_2nd - 77.3635) / 2.396009   # +0.2%  nca_sj4_pair_mass_2nd > 77.36
        + 0.002378129 * max(0.0, Q.D2 - 3.597891) / 0.2539167   # +0.2%  D2 > 3.598
        - 0.002214306 * max(0.0, Q.sj2_mass1 - 53.51926) / 6.081546   # -0.2%  sj2_mass1 > 53.52
        + 0.001909864 * max(0.0, Q.mres_sd_mass_b0z005 - 159.9242) / 3.119054   # +0.2%  mres_sd_mass_b0z005 > 159.9
        + 0.001577138 * max(0.0, Q.sjf_2_1_z_d3 - 0.1149585) * max(0.0, Q.ischhad_0 - 0.0) / 0.02018831   # +0.2%  sjf_2_1_z_d3 > 0.115 and ischhad_0 > 0
        + 0.001415755 * max(0.0, Q.sj4_pair_mass_max - 128.0079) / 1.174806   # +0.1%  sj4_pair_mass_max > 128
        + 0.001264962 * max(0.0, 1.226724 - Q.D2) / 0.07974157   # +0.1%  D2 < 1.227
    )
    return z


def neuron_105(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.732383e-05
    )
    return z


def neuron_106(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.160934e-05
    )
    return z


def neuron_107(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.15856e-05
    )
    return z


def neuron_108(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.470701e-06
    )
    return z


def neuron_109(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.200155e-07
    )
    return z


def neuron_110(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.579609e-06
    )
    return z


def neuron_111(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.669616e-06
    )
    return z


def neuron_112(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.237823e-06
    )
    return z


def neuron_113(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.665928e-05
    )
    return z


def neuron_114(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.677418e-06
    )
    return z


def neuron_115(Q):
    # scale S = 11.94; each line: share * term / its average size
    z = 11.93773 * (0.1118897
        - 0.08364832 * max(0.0, 39.09615 - Q.mass_displaced3) / 31.84697   # -8.4%  mass_displaced3 < 39.1
        - 0.05863305 * max(0.0, 18.7678 - Q.lep_ptrel) / 14.28156   # -5.9%  lep_ptrel < 18.77
        - 0.05715165 * Q.sj3_pairmax_over_m / 0.807958   # -5.7%  sj3_pairmax_over_m
        + 0.04555992 * max(0.0, 0.5187302 - Q.lep_z) / 0.4405854   # +4.6%  lep_z < 0.5187
        - 0.04524392 * max(0.0, 0.5626523 - Q.LHA) / 0.1600511   # -4.5%  LHA < 0.5627
        + 0.0427015 * max(0.0, 131.3917 - Q.mass) / 24.784   # +4.3%  mass < 131.4
        + 0.03925208 * max(0.0, 0.120439 - Q.M2) / 0.04313262   # +3.9%  M2 < 0.1204
        + 0.03747922 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 18.7678 - Q.lep_ptrel) / 453.8353   # +3.7%  mass_displaced3 < 39.1 and lep_ptrel < 18.77
        - 0.03652648 * max(0.0, 161.1264 - Q.mass_top50) / 50.68933   # -3.7%  mass_top50 < 161.1
        + 0.0337338 * Q.z_neutral_had / 0.1513027   # +3.4%  z_neutral_had
        + 0.03017781 * max(0.0, 0.9367772 - Q.pair_mean_lnkt) / 0.4427064   # +3.0%  pair_mean_lnkt < 0.9368
        - 0.02433278 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) / 26.86577   # -2.4%  sj3_pair_mass_max < 114.8
        + 0.0233576 * max(0.0, 0.09733903 - Q.tau2) / 0.03188557   # +2.3%  tau2 < 0.09734
        + 0.02282692 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # +2.3%  n_lepton < 1
        - 0.02189667 * max(0.0, Q.pair_max_lnkt - 1.555555) / 1.17245   # -2.2%  pair_max_lnkt > 1.556
        + 0.02101442 * max(0.0, 0.00228569 - Q.e3) / 0.001253813   # +2.1%  e3 < 0.002286
        - 0.02031217 * max(0.0, 101.849 - Q.sj4_pair_mass_max) / 24.8879   # -2.0%  sj4_pair_mass_max < 101.8
        + 0.01803131 * max(0.0, 6.767937 - Q.mass_2charged) / 3.01163   # +1.8%  mass_2charged < 6.768
        + 0.0168536 * max(0.0, 21.04127 - Q.mres_sd_prong_mass2) / 11.57923   # +1.7%  mres_sd_prong_mass2 < 21.04
        - 0.01592075 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # -1.6%  e3_b2 < 0.0004127
        - 0.01438096 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # -1.4%  e3_b2 < 0.000791
        - 0.01282714 * max(0.0, Q.n_sd0_above_3 - 2.0) / 1.826197   # -1.3%  n_sd0_above_3 > 2
        - 0.01264103 * max(0.0, 3.0 - Q.lepsj_3_n_d3) / 2.624773   # -1.3%  lepsj_3_n_d3 < 3
        - 0.01249766 * max(0.0, 20.76537 - Q.mass_2charged) / 12.0384   # -1.2%  mass_2charged < 20.77
        + 0.01235869 * max(0.0, Q.n_s3d_above_3 - 4.0) / 1.224997   # +1.2%  n_s3d_above_3 > 4
        - 0.01191521 * max(0.0, 0.932165 - Q.dc_1_z) / 0.3086674   # -1.2%  dc_1_z < 0.9322
        - 0.01132057 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 14.38858 - Q.lep_iso) / 5.237586   # -1.1%  lep_z < 0.5187 and lep_iso < 14.39
        - 0.01115806 * max(0.0, 0.8939856 - Q.tau43) / 0.102551   # -1.1%  tau43 < 0.894
        + 0.01053198 * max(0.0, 118.8408 - Q.mass_top50) * max(0.0, 9.356714 - Q.jd_3d_6) / 120.1591   # +1.1%  mass_top50 < 118.8 and jd_3d_6 < 9.357
        + 0.01042611 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # +1.0%  sip_3d_2 < 447.1
        + 0.01022099 * max(0.0, 0.2243273 - Q.N2_b2) / 0.06973244   # +1.0%  N2_b2 < 0.2243
        - 0.01013302 * max(0.0, 101.849 - Q.sj4_pair_mass_max) * max(0.0, 9.356714 - Q.jd_3d_6) / 169.6293   # -1.0%  sj4_pair_mass_max < 101.8 and jd_3d_6 < 9.357
        - 0.009388536 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) * max(0.0, 509.6197 - Q.sv_1_sd0_sum) / 693.4354   # -0.9%  lepsj_3_maxsd0 < 2.413 and sv_1_sd0_sum < 509.6
        + 0.009292918 * max(0.0, 75.78208 - Q.sj4_pair_mass_max) / 8.111726   # +0.9%  sj4_pair_mass_max < 75.78
        + 0.006950616 * max(0.0, 428.9828 - Q.sum_pt_top10) / 28.27138   # +0.7%  sum_pt_top10 < 429
        + 0.006934707 * max(0.0, 25.64898 - Q.sj4_pair_mass_min) / 11.12072   # +0.7%  sj4_pair_mass_min < 25.65
        + 0.006903595 * max(0.0, Q.lund_max_lnkt - 3.388322) / 0.5830947   # +0.7%  lund_max_lnkt > 3.388
        - 0.006777744 * max(0.0, 0.02745856 - Q.sum_z_dr2_top3) / 0.01202135   # -0.7%  sum_z_dr2_top3 < 0.02746
        - 0.00660858 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # -0.7%  n_pairs_kt_above_1 < 80
        + 0.006431496 * max(0.0, 118.8408 - Q.mass_top50) * max(0.0, 0.103005 - Q.lepsj_2_dr) / 1.51288   # +0.6%  mass_top50 < 118.8 and lepsj_2_dr < 0.103
        - 0.006237217 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, 9.982976 - Q.jd_3d_4) / 1917.09   # -0.6%  sip_3d_2 < 447.1 and jd_3d_4 < 9.983
        + 0.006162972 * max(0.0, 411.0 - Q.n_pairs_kt_above_1) / 159.5699   # +0.6%  n_pairs_kt_above_1 < 411
        + 0.006073855 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 0.1845735 - Q.sj4_dr_min) / 0.03477143   # +0.6%  lep_z < 0.5187 and sj4_dr_min < 0.1846
        + 0.006031435 * max(0.0, 93.50967 - Q.mass_top20) / 12.41212   # +0.6%  mass_top20 < 93.51
        - 0.005980839 * max(0.0, 7.065114 - Q.mres_sd_prong_mass2) / 1.50217   # -0.6%  mres_sd_prong_mass2 < 7.065
        - 0.005968935 * max(0.0, 5.460258 - Q.sj3_mass3) / 1.669879   # -0.6%  sj3_mass3 < 5.46
        - 0.005784699 * max(0.0, 1.398635 - Q.mass_2charged) / 0.2801595   # -0.6%  mass_2charged < 1.399
        + 0.005119228 * max(0.0, 6.402344 - Q.max_abs_dz) / 3.33271   # +0.5%  max_abs_dz < 6.402
        - 0.004829663 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 0.8786609 - Q.tau54) / 0.02005551   # -0.5%  lep_z < 0.5187 and tau54 < 0.8787
        - 0.004799849 * max(0.0, Q.ktd_ln_d34 - -9.531553) / 0.6451699   # -0.5%  ktd_ln_d34 > -9.532
        + 0.004156768 * max(0.0, 118.8408 - Q.mass_top50) / 16.89652   # +0.4%  mass_top50 < 118.8
        + 0.004107012 * max(0.0, 0.3479096 - Q.sjf_3_3_maxsd0) / 0.04351868   # +0.4%  sjf_3_3_maxsd0 < 0.3479
        - 0.003879315 * max(0.0, 6.402344 - Q.max_abs_dz) * max(0.0, 0.01012269 - Q.lepsj_3_dr) / 0.02407785   # -0.4%  max_abs_dz < 6.402 and lepsj_3_dr < 0.01012
        - 0.003730842 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.ak02_n - 2.0) / 0.3373298   # -0.4%  lep_z < 0.5187 and ak02_n > 2
        + 0.003702271 * max(0.0, Q.pt_entropy - 3.22434) / 0.06429873   # +0.4%  pt_entropy > 3.224
        + 0.003527986 * max(0.0, Q.psi_0p3 - 0.9756505) / 0.00473588   # +0.4%  psi_0p3 > 0.9757
        - 0.003403303 * max(0.0, Q.mres_sd_mass_b1z01 - 88.79082) / 24.90103   # -0.3%  mres_sd_mass_b1z01 > 88.79
        - 0.003113803 * max(0.0, 0.9367772 - Q.pair_mean_lnkt) * max(0.0, 428.9828 - Q.sum_pt_top10) / 14.08647   # -0.3%  pair_mean_lnkt < 0.9368 and sum_pt_top10 < 429
        - 0.003060365 * max(0.0, Q.tau4 - 0.04996) / 0.004242161   # -0.3%  tau4 > 0.04996
        - 0.002874843 * max(0.0, 0.09733903 - Q.tau2) * max(0.0, Q.kt2_1_n_disp3 - 0.0) / 0.02717373   # -0.3%  tau2 < 0.09734 and kt2_1_n_disp3 > 0
        + 0.002657058 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.lne_1 - 4.635336) / 0.08574602   # +0.3%  lep_z < 0.5187 and lne_1 > 4.635
        + 0.002618368 * max(0.0, Q.sj3_pair_mass_min - 59.49644) * max(0.0, 4.0 - Q.n_lund_kt_above_5) / 2.905603   # +0.3%  sj3_pair_mass_min > 59.5 and n_lund_kt_above_5 < 4
        - 0.002600361 * max(0.0, 94.51361 - Q.mass_top50) / 6.251054   # -0.3%  mass_top50 < 94.51
        + 0.00258857 * max(0.0, Q.ktd_ln_d34 - -9.531553) * max(0.0, 0.102661 - Q.pz_lnkt3) / 0.02974445   # +0.3%  ktd_ln_d34 > -9.532 and pz_lnkt3 < 0.1027
        + 0.001807418 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.sdb_5_z - 0.04013001) / 5.763019e-06   # +0.2%  e3_b2 < 0.0004127 and sdb_5_z > 0.04013
        - 0.0008294695 * max(0.0, Q.z_neutral_had - 0.3788785) / 0.004557199   # -0.1%  z_neutral_had > 0.3789
    )
    return z


def neuron_116(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.399317e-07
    )
    return z


def neuron_117(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.862138e-06
    )
    return z


def neuron_118(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.233224e-06
    )
    return z


def neuron_119(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.336862e-07
    )
    return z


def neuron_120(Q):
    # scale S = 9.631; each line: share * term / its average size
    z = 9.630885 * (0.1278778
        - 0.07087527 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 10.0 - Q.n_s3d_above_3) / 15.42709   # -7.1%  lep_ptrel < 3.535 and n_s3d_above_3 < 10
        - 0.05229664 * max(0.0, 111.701 - Q.sd_mass) / 26.44189   # -5.2%  sd_mass < 111.7
        + 0.04771289 * max(0.0, 88.81751 - Q.sd_mass) / 15.02663   # +4.8%  sd_mass < 88.82
        - 0.04390926 * max(0.0, 12.15228 - Q.lep_ptrel) / 8.819929   # -4.4%  lep_ptrel < 12.15
        - 0.04046072 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # -4.0%  lep_iso < 0.4382
        + 0.0403088 * max(0.0, 178.725 - Q.mass_top50) / 66.98604   # +4.0%  mass_top50 < 178.7
        + 0.03227423 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, 1.0 - Q.dc_1_n_lep) / 0.2724246   # +3.2%  lep_iso < 0.4382 and dc_1_n_lep < 1
        - 0.03049577 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) / 16.68515   # -3.0%  n_pairs_kt_above_3 < 62
        - 0.02879013 * max(0.0, 68.20711 - Q.mass_charged) / 12.83892   # -2.9%  mass_charged < 68.21
        + 0.02811416 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 172.888 - Q.jd_3d_4) / 351.393   # +2.8%  lep_ptrel < 3.535 and jd_3d_4 < 172.9
        - 0.02714552 * max(0.0, 0.004135872 - Q.lep_z) / 0.002390373   # -2.7%  lep_z < 0.004136
        + 0.02623612 * max(0.0, 63.87145 - Q.mass_neutral) / 22.26348   # +2.6%  mass_neutral < 63.87
        - 0.02538741 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 69.1565 - Q.mass_neutral) / 10.38924   # -2.5%  lep_z < 0.5187 and mass_neutral < 69.16
        - 0.02445369 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 3.151933 - Q.jd_3d_6) / 9.359889   # -2.4%  lep_ptrel < 12.15 and jd_3d_6 < 3.152
        - 0.0239608 * max(0.0, 0.06573337 - Q.lepsj_2_dr) / 0.04985349   # -2.4%  lepsj_2_dr < 0.06573
        + 0.02329227 * max(0.0, 0.08178299 - Q.lep_dr) / 0.04892529   # +2.3%  lep_dr < 0.08178
        + 0.02322641 * max(0.0, 68.20711 - Q.mass_charged) * max(0.0, 5.367501 - Q.jd_3d_6) / 44.32806   # +2.3%  mass_charged < 68.21 and jd_3d_6 < 5.368
        - 0.02230706 * max(0.0, 3.53503 - Q.lep_ptrel) / 2.299439   # -2.2%  lep_ptrel < 3.535
        + 0.02221622 * max(0.0, 0.0008210142 - Q.lepsj_2_dr) / 0.0004891649   # +2.2%  lepsj_2_dr < 0.000821
        - 0.02174779 * max(0.0, 114.0172 - Q.mass) / 13.82223   # -2.2%  mass < 114
        - 0.02034295 * max(0.0, 0.3014662 - Q.lep_dr) / 0.2238267   # -2.0%  lep_dr < 0.3015
        + 0.01963856 * max(0.0, 2.907433 - Q.lep_iso) * max(0.0, 5.367501 - Q.jd_3d_6) / 6.467594   # +2.0%  lep_iso < 2.907 and jd_3d_6 < 5.368
        - 0.01945186 * max(0.0, 105.7234 - Q.mass) / 10.09077   # -1.9%  mass < 105.7
        - 0.01720497 * max(0.0, 0.06573337 - Q.lepsj_2_dr) * max(0.0, 5.367501 - Q.jd_3d_6) / 0.1450967   # -1.7%  lepsj_2_dr < 0.06573 and jd_3d_6 < 5.368
        + 0.01710812 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 18.0 - Q.sdb_2_n) / 61.14732   # +1.7%  lep_ptrel < 12.15 and sdb_2_n < 18
        + 0.01695478 * max(0.0, 2.907433 - Q.lep_iso) / 2.168962   # +1.7%  lep_iso < 2.907
        + 0.01675253 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, Q.n_photon - 5.0) / 28.0407   # +1.7%  lep_ptrel < 3.535 and n_photon > 5
        - 0.01624762 * max(0.0, 11.0 - Q.n_pairs_kt_above_10) / 5.855997   # -1.6%  n_pairs_kt_above_10 < 11
        + 0.01516558 * max(0.0, 0.1589878 - Q.pz_lnkt0) / 0.02919723   # +1.5%  pz_lnkt0 < 0.159
        - 0.01205185 * max(0.0, 1.0 - Q.ak02_1_n_lep) / 0.7451267   # -1.2%  ak02_1_n_lep < 1
        + 0.01167742 * max(0.0, 114.0172 - Q.mass) * max(0.0, 5.367501 - Q.jd_3d_6) / 48.98244   # +1.2%  mass < 114 and jd_3d_6 < 5.368
        - 0.0109588 * max(0.0, 78.4753 - Q.sd_mass) / 11.39114   # -1.1%  sd_mass < 78.48
        + 0.009487645 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.2710171 - Q.sjf_2_1_z_d3) / 3.530667   # +0.9%  n_pairs_kt_above_3 < 62 and sjf_2_1_z_d3 < 0.271
        + 0.0091376 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 5.8125 - Q.max_abs_d0) / 6.645755   # +0.9%  lep_ptrel < 3.535 and max_abs_d0 < 5.812
        - 0.00899658 * max(0.0, Q.lund3_lndelta - -2.036501) / 0.629841   # -0.9%  lund3_lndelta > -2.037
        + 0.007549393 * max(0.0, 11.0 - Q.n_pairs_kt_above_10) * max(0.0, 3.151933 - Q.jd_3d_6) / 7.085947   # +0.8%  n_pairs_kt_above_10 < 11 and jd_3d_6 < 3.152
        + 0.007461329 * max(0.0, 0.6800935 - Q.tau32) / 0.0806267   # +0.7%  tau32 < 0.6801
        + 0.006332637 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, 1.0 - Q.dc_2_n_lep) / 12.75663   # +0.6%  n_pairs_kt_above_3 < 62 and dc_2_n_lep < 1
        + 0.006190473 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.mass_displaced3 - 0.0) / 3.662943   # +0.6%  lep_z < 0.5187 and mass_displaced3 > 0
        + 0.005875858 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.1822601 - Q.lepsj_2_dr) / 2.79471   # +0.6%  n_pairs_kt_above_3 < 62 and lepsj_2_dr < 0.1823
        + 0.005087873 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 5.0 - Q.sdb_3_n) / 20.52156   # +0.5%  lep_ptrel < 12.15 and sdb_3_n < 5
        + 0.004986315 * max(0.0, 71.96396 - Q.mass) / 1.934957   # +0.5%  mass < 71.96
        + 0.004818421 * max(0.0, 114.0172 - Q.mass) * max(0.0, 0.1271955 - Q.z_neutral_had) / 0.6099698   # +0.5%  mass < 114 and z_neutral_had < 0.1272
        - 0.004731475 * max(0.0, Q.mass_top20 - 114.4658) / 4.613131   # -0.5%  mass_top20 > 114.5
        - 0.004527505 * max(0.0, 0.1589878 - Q.pz_lnkt0) * max(0.0, Q.sjq_2_prod_k05 - -0.3383985) / 0.008728143   # -0.5%  pz_lnkt0 < 0.159 and sjq_2_prod_k05 > -0.3384
        - 0.004340573 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # -0.4%  n_pairs_kt_above_1 < 80
        - 0.004265574 * max(0.0, Q.pair_mean_lndelta - -1.790445) / 0.06864152   # -0.4%  pair_mean_lndelta > -1.79
        - 0.004171854 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, Q.pz_lnd2 - 0.008992646) / 0.9741691   # -0.4%  lep_ptrel < 12.15 and pz_lnd2 > 0.008993
        - 0.003822887 * max(0.0, 4.912483 - Q.kt2_min12_jp) / 1.902446   # -0.4%  kt2_min12_jp < 4.912
        - 0.003709992 * max(0.0, 0.05805236 - Q.M2) * max(0.0, 6.771002 - Q.jd_3d_5) / 0.01364138   # -0.4%  M2 < 0.05805 and jd_3d_5 < 6.771
        - 0.003178084 * max(0.0, Q.mass_top50 - 135.5368) / 6.535045   # -0.3%  mass_top50 > 135.5
        - 0.003147664 * max(0.0, 0.3667049 - Q.N2_b05) / 0.008239872   # -0.3%  N2_b05 < 0.3667
        - 0.003092915 * max(0.0, 0.03277088 - Q.pz_lnd2) / 0.003265594   # -0.3%  pz_lnd2 < 0.03277
        + 0.002761844 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # +0.3%  mass_top50 > 161.1
        - 0.002756289 * max(0.0, 3.029824 - Q.sjf_2_2_max3d) / 0.5243565   # -0.3%  sjf_2_2_max3d < 3.03
        + 0.002693089 * max(0.0, 0.05805236 - Q.M2) / 0.003596358   # +0.3%  M2 < 0.05805
        - 0.002674585 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.3%  mass < 90.09
        - 0.002522211 * max(0.0, Q.lam1 - 0.04304553) / 0.005099272   # -0.3%  lam1 > 0.04305
        - 0.002413173 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # -0.2%  lep_ptrel > 27.32
        + 0.002290973 * max(0.0, Q.sjq_3_3_k1 - 0.4037488) / 0.02997137   # +0.2%  sjq_3_3_k1 > 0.4037
        - 0.001871674 * max(0.0, 4.6892 - Q.mres_sd_prong_mass2) / 0.7312792   # -0.2%  mres_sd_prong_mass2 < 4.689
        - 0.001760532 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # -0.2%  mass_top10 > 103.5
        + 0.001659532 * max(0.0, Q.e3 - 0.00228569) / 0.000293828   # +0.2%  e3 > 0.002286
        + 0.001593834 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, -0.2912597 - Q.sjq_3_3_k1) / 0.01773337   # +0.2%  lep_z < 0.5187 and sjq_3_3_k1 < -0.2913
        - 0.001552718 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, Q.sv_2_z - 0.0) / 0.01309156   # -0.2%  lep_ptrel < 3.535 and sv_2_z > 0
        + 0.001538812 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, Q.n_muon - 0.0) / 0.7787752   # +0.2%  lep_ptrel < 12.15 and n_muon > 0
        + 0.001415173 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, Q.e4_b05 - 4.363663e-05) / 3.240758e-05   # +0.1%  lep_ptrel < 3.535 and e4_b05 > 4.364e-05
        + 0.001270992 * max(0.0, 0.05066241 - Q.pz_lnd0) / 0.003045628   # +0.1%  pz_lnd0 < 0.05066
        - 0.001145133 * max(0.0, 0.06573337 - Q.lepsj_2_dr) * max(0.0, Q.sjq_3_3_k1 - 0.4037488) / 0.00137647   # -0.1%  lepsj_2_dr < 0.06573 and sjq_3_3_k1 > 0.4037
        - 0.001121775 * max(0.0, Q.pair_mean_lnm2 - 4.787589) / 0.03990488   # -0.1%  pair_mean_lnm2 > 4.788
        + 0.001062784 * max(0.0, 0.05066241 - Q.pz_lnd0) * max(0.0, 0.0538419 - Q.dr_19) / 3.863467e-05   # +0.1%  pz_lnd0 < 0.05066 and dr_19 < 0.05384
        - 0.0009230557 * max(0.0, Q.ak02_1_n_lep - 1.0) / 0.05023   # -0.1%  ak02_1_n_lep > 1
        + 0.0007869551 * max(0.0, Q.mass_top10 - 103.4976) * max(0.0, Q.sum_pt_top40 - 502.3395) / 282.4757   # +0.1%  mass_top10 > 103.5 and sum_pt_top40 > 502.3
        - 0.0005060188 * max(0.0, Q.sj3_dr13 - 0.7462286) / 0.01200383   # -0.1%  sj3_dr13 > 0.7462
    )
    return z


def neuron_121(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.73052e-06
    )
    return z


def neuron_122(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.395902e-06
    )
    return z


def neuron_123(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.0191935
    )
    return z


def neuron_124(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.1131311
    )
    return z


def neuron_125(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.190763e-06
    )
    return z


def neuron_126(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.632632e-06
    )
    return z


def neuron_127(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.966697e-07
    )
    return z


def class_token(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q), neuron_16(Q), neuron_17(Q), neuron_18(Q), neuron_19(Q), neuron_20(Q), neuron_21(Q), neuron_22(Q), neuron_23(Q), neuron_24(Q), neuron_25(Q), neuron_26(Q), neuron_27(Q), neuron_28(Q), neuron_29(Q), neuron_30(Q), neuron_31(Q), neuron_32(Q), neuron_33(Q), neuron_34(Q), neuron_35(Q), neuron_36(Q), neuron_37(Q), neuron_38(Q), neuron_39(Q), neuron_40(Q), neuron_41(Q), neuron_42(Q), neuron_43(Q), neuron_44(Q), neuron_45(Q), neuron_46(Q), neuron_47(Q), neuron_48(Q), neuron_49(Q), neuron_50(Q), neuron_51(Q), neuron_52(Q), neuron_53(Q), neuron_54(Q), neuron_55(Q), neuron_56(Q), neuron_57(Q), neuron_58(Q), neuron_59(Q), neuron_60(Q), neuron_61(Q), neuron_62(Q), neuron_63(Q), neuron_64(Q), neuron_65(Q), neuron_66(Q), neuron_67(Q), neuron_68(Q), neuron_69(Q), neuron_70(Q), neuron_71(Q), neuron_72(Q), neuron_73(Q), neuron_74(Q), neuron_75(Q), neuron_76(Q), neuron_77(Q), neuron_78(Q), neuron_79(Q), neuron_80(Q), neuron_81(Q), neuron_82(Q), neuron_83(Q), neuron_84(Q), neuron_85(Q), neuron_86(Q), neuron_87(Q), neuron_88(Q), neuron_89(Q), neuron_90(Q), neuron_91(Q), neuron_92(Q), neuron_93(Q), neuron_94(Q), neuron_95(Q), neuron_96(Q), neuron_97(Q), neuron_98(Q), neuron_99(Q), neuron_100(Q), neuron_101(Q), neuron_102(Q), neuron_103(Q), neuron_104(Q), neuron_105(Q), neuron_106(Q), neuron_107(Q), neuron_108(Q), neuron_109(Q), neuron_110(Q), neuron_111(Q), neuron_112(Q), neuron_113(Q), neuron_114(Q), neuron_115(Q), neuron_116(Q), neuron_117(Q), neuron_118(Q), neuron_119(Q), neuron_120(Q), neuron_121(Q), neuron_122(Q), neuron_123(Q), neuron_124(Q), neuron_125(Q), neuron_126(Q), neuron_127(Q)]


H_AVG = [0.00518969027325511, 5.325267011357937e-06, 5.447302555694478e-06, 8.39983931655297e-06, 2.7921398668695474e-06, 2.1569237560470356e-06, 1.770106246112846e-05, 6.430824669223512e-06, 2.926471324826707e-06, 2.44366196966439e-06, 1.5701054508099332e-05, 1.8877131878980435e-05, 5.099298050481593e-06, 4.422911388246575e-06, 2.845303924914333e-06, 6.4595269577694125e-06, 1.260690592402373, 1.0353472134738695e-05, 1.067999195218418, 1.4078198091738159e-06, 8.253686019088491e-07, 0.001682463800534606, 1.0448460443512886e-06, 6.978798978707346e-07, 1.0889400067671244, 0.03136871010065079, 3.2312607345375e-06, 6.308500815066509e-06, 9.617207979317755e-06, 9.183864676742814e-06, 8.60305499372771e-06, 3.5095233670290327e-06, 1.4413480130315293e-05, 2.56732573689078e-06, 1.3884864529245533e-05, 6.743300673406338e-06, 4.831947990169283e-06, 4.357436409918591e-06, 9.411175597051624e-06, 8.274277206510305e-06, 7.775539643262164e-07, 3.8146914960179856e-08, 8.779052222962491e-06, 9.407708603248466e-06, 3.3243493362533627e-06, 6.8493368416966405e-06, 1.3687592570477136e-07, 2.4920754526647215e-07, 8.791777872829698e-06, 2.2438150608650176e-06, 5.17511625730549e-06, 2.012202230616822e-06, 0.7298259413165791, 1.889362692963914e-06, 1.1906591680599377e-05, 2.0669400328188203e-05, 7.072331413837674e-07, 1.9596477613958996e-06, 3.6171511510474375e-06, 6.0293318711046595e-06, 4.666765562433284e-06, 5.840559424541425e-06, 9.422975608686102e-07, 8.308512406074442e-06, 1.2586291404659278e-06, 5.791054263681872e-06, 2.7097789825347718e-06, 5.219511422183132e-06, 0.9426195526525165, 6.926144124008715e-06, 0.016012562438845634, 3.0297092052933294e-06, 1.1641204764600843e-06, 1.699640415608883e-05, 4.493399046623381e-06, 1.9738399714697152e-06, 5.542788585444214e-06, 3.087486447839183e-06, 0.6365336076171539, 3.602388096624054e-05, 9.707706340122968e-06, 0.5651591579804759, 2.9420746159303235e-06, 0.7015632093690842, 7.427513628499582e-05, 8.185525075532496e-07, 1.709903335722629e-05, 1.6191623899430851e-06, 2.996980015268491e-07, 1.7353742123304983e-06, 0.0012298653600737453, 3.992638994532172e-06, 6.724735612806398e-06, 8.464390361950791e-07, 2.3221721221489133e-06, 4.3695186491277127e-07, 4.2780305875567137e-07, 0.9960086942508197, 4.990168918084237e-07, 0.0019342415034770966, 6.26407654635841e-06, 8.878262633515988e-06, 5.490514922712464e-06, 3.414380216781865e-06, 0.862080934126616, 2.732382927206345e-05, 2.160934491257649e-05, 1.1585601896513253e-05, 4.470701242098585e-06, 5.200154760132136e-07, 3.57960925612133e-06, 3.6696162624139106e-06, 5.237822733761277e-06, 2.665928150236141e-05, 4.67741847387515e-06, 0.949144181789996, 3.399316597096913e-07, 2.8621379897231236e-06, 8.233223525166977e-06, 6.33686227047292e-07, 0.9054951940971818, 5.730519660573918e-06, 2.3959016743901884e-06, 0.019193504005670547, 0.11313110589981079, 1.1907625321327941e-06, 7.632632332388312e-06, 1.966697027455666e-07]
T = [4.096379870168239, 5.898393282971542, 3.92365533806661, 4.350462559821134, 4.8923009416898875, 8.019026540706392, 3.4891850597297327, 4.288040247864444, 6.872729809370931, 9.842270838549904]


def logits(h):
    # rounded to a multiple of 2^-20 (the formula's class scores are exact binary fractions): removes floating-point noise
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        0.4043572 + T[0] * (   # class QCD
            - 0.1826123 * h[18] / H_AVG[18]
            + 0.1450156 * h[97] / H_AVG[97]
            - 0.1136453 * h[68] / H_AVG[68]
            + 0.09160732 * h[52] / H_AVG[52]
            + 0.08919578 * h[81] / H_AVG[81]
            - 0.08784452 * h[24] / H_AVG[24]
            - 0.08730702 * h[120] / H_AVG[120]
            + 0.06889666 * h[104] / H_AVG[104]
            + 0.04131924 * h[78] / H_AVG[78]
            + 0.03133549 * h[16] / H_AVG[16]
            + 0.02951398 * h[115] / H_AVG[115]
            - 0.02572248 * h[83] / H_AVG[83]
            - 0.00503575 * h[124] / H_AVG[124]
            + 0.0003272123 * h[123] / H_AVG[123]
            + 0.0002177019 * h[25] / H_AVG[25]
            + 0.0001906784 * h[70] / H_AVG[70]
            - 6.850518e-05 * h[90] / H_AVG[90]
            + 6.446152e-05 * h[0] / H_AVG[0]
            - 4.172435e-05 * h[99] / H_AVG[99]
            - 3.2164e-05 * h[21] / H_AVG[21]
            + 1.971036e-06 * h[84] / H_AVG[84]
            + 4.650078e-07 * h[55] / H_AVG[55]
            + 3.120918e-07 * h[79] / H_AVG[79]
            - 2.979411e-07 * h[105] / H_AVG[105]
            - 2.417796e-07 * h[106] / H_AVG[106]
            + 1.965308e-07 * h[113] / H_AVG[113]
            - 1.866382e-07 * h[34] / H_AVG[34]
            + 1.675889e-07 * h[73] / H_AVG[73]
            + 1.556415e-07 * h[10] / H_AVG[10]
            + 1.483184e-07 * h[11] / H_AVG[11]
            - 1.263215e-07 * h[6] / H_AVG[6]
            + 1.205439e-07 * h[86] / H_AVG[86]
            - 1.146283e-07 * h[32] / H_AVG[32]
            + 1.053571e-07 * h[17] / H_AVG[17]
            + 9.521269e-08 * h[38] / H_AVG[38]
            - 7.818928e-08 * h[42] / H_AVG[42]
            - 6.968268e-08 * h[54] / H_AVG[54]
            - 6.623098e-08 * h[29] / H_AVG[29]
            + 6.004694e-08 * h[118] / H_AVG[118]
            + 4.687258e-08 * h[48] / H_AVG[48]
            - 4.473251e-08 * h[45] / H_AVG[45]
            - 4.320095e-08 * h[39] / H_AVG[39]
            + 3.983006e-08 * h[27] / H_AVG[27]
            + 3.966581e-08 * h[63] / H_AVG[63]
            + 3.924266e-08 * h[30] / H_AVG[30]
            + 3.84285e-08 * h[43] / H_AVG[43]
            + 3.574619e-08 * h[121] / H_AVG[121]
            - 3.511177e-08 * h[80] / H_AVG[80]
            - 3.213741e-08 * h[126] / H_AVG[126]
            + 3.102377e-08 * h[60] / H_AVG[60]
            - 3.055571e-08 * h[61] / H_AVG[61]
            - 2.901404e-08 * h[102] / H_AVG[102]
            - 2.897053e-08 * h[107] / H_AVG[107]
            - 2.852014e-08 * h[7] / H_AVG[7]
            + 2.569152e-08 * h[92] / H_AVG[92]
            + 2.51252e-08 * h[35] / H_AVG[35]
            + 2.4434e-08 * h[15] / H_AVG[15]
            - 2.359684e-08 * h[59] / H_AVG[59]
            - 2.276408e-08 * h[101] / H_AVG[101]
            - 2.17626e-08 * h[28] / H_AVG[28]
            + 2.110373e-08 * h[103] / H_AVG[103]
            + 1.986282e-08 * h[69] / H_AVG[69]
            + 1.927734e-08 * h[67] / H_AVG[67]
            + 1.867267e-08 * h[1] / H_AVG[1]
            + 1.853443e-08 * h[100] / H_AVG[100]
            - 1.702268e-08 * h[110] / H_AVG[110]
            - 1.647115e-08 * h[3] / H_AVG[3]
            + 1.582191e-08 * h[37] / H_AVG[37]
            - 1.549924e-08 * h[50] / H_AVG[50]
            + 1.456902e-08 * h[44] / H_AVG[44]
            + 1.455746e-08 * h[36] / H_AVG[36]
            - 1.430994e-08 * h[12] / H_AVG[12]
            - 1.361313e-08 * h[114] / H_AVG[114]
            - 1.349644e-08 * h[2] / H_AVG[2]
            + 1.30075e-08 * h[58] / H_AVG[58]
            - 1.295302e-08 * h[112] / H_AVG[112]
            + 1.267791e-08 * h[76] / H_AVG[76]
            - 1.222125e-08 * h[13] / H_AVG[13]
            - 1.186839e-08 * h[19] / H_AVG[19]
            - 1.164923e-08 * h[111] / H_AVG[111]
            + 1.110253e-08 * h[74] / H_AVG[74]
            + 1.053539e-08 * h[108] / H_AVG[108]
            + 9.903528e-09 * h[5] / H_AVG[5]
            - 9.466294e-09 * h[8] / H_AVG[8]
            - 9.077136e-09 * h[91] / H_AVG[91]
            - 8.462311e-09 * h[4] / H_AVG[4]
            - 8.106436e-09 * h[117] / H_AVG[117]
            + 6.850073e-09 * h[53] / H_AVG[53]
            - 6.774118e-09 * h[66] / H_AVG[66]
            - 6.165357e-09 * h[33] / H_AVG[33]
            - 5.9901e-09 * h[65] / H_AVG[65]
            + 5.53921e-09 * h[82] / H_AVG[82]
            - 5.20538e-09 * h[31] / H_AVG[31]
            - 4.783401e-09 * h[71] / H_AVG[71]
            + 4.442858e-09 * h[94] / H_AVG[94]
            + 4.290963e-09 * h[9] / H_AVG[9]
            + 3.77173e-09 * h[26] / H_AVG[26]
            - 3.3801e-09 * h[14] / H_AVG[14]
            + 3.348341e-09 * h[57] / H_AVG[57]
            + 3.164645e-09 * h[49] / H_AVG[49]
            + 3.091521e-09 * h[77] / H_AVG[77]
            - 3.085215e-09 * h[22] / H_AVG[22]
            + 2.862185e-09 * h[122] / H_AVG[122]
            + 2.313804e-09 * h[72] / H_AVG[72]
            + 2.21609e-09 * h[51] / H_AVG[51]
            - 2.18168e-09 * h[23] / H_AVG[23]
            - 1.506668e-09 * h[89] / H_AVG[89]
            + 1.47212e-09 * h[125] / H_AVG[125]
            - 1.382059e-09 * h[75] / H_AVG[75]
            - 1.195976e-09 * h[87] / H_AVG[87]
            + 1.192004e-09 * h[93] / H_AVG[93]
            - 1.071992e-09 * h[40] / H_AVG[40]
            - 9.960936e-10 * h[20] / H_AVG[20]
            + 8.900234e-10 * h[119] / H_AVG[119]
            - 8.141557e-10 * h[62] / H_AVG[62]
            - 7.320677e-10 * h[95] / H_AVG[95]
            - 6.784617e-10 * h[96] / H_AVG[96]
            + 3.999626e-10 * h[116] / H_AVG[116]
            + 3.819519e-10 * h[56] / H_AVG[56]
            + 2.655456e-10 * h[85] / H_AVG[85]
            + 1.580634e-10 * h[109] / H_AVG[109]
            - 1.37618e-10 * h[98] / H_AVG[98]
            - 1.099203e-10 * h[88] / H_AVG[88]
            - 7.409272e-11 * h[46] / H_AVG[46]
            - 6.994387e-11 * h[47] / H_AVG[47]
            - 4.31333e-11 * h[64] / H_AVG[64]
            + 1.680658e-11 * h[127] / H_AVG[127]
            + 1.484503e-12 * h[41] / H_AVG[41]
        ),
        -0.4064285 + T[1] * (   # class Hbb
            + 0.2852211 * h[16] / H_AVG[16]
            + 0.1816732 * h[120] / H_AVG[120]
            - 0.09770458 * h[68] / H_AVG[68]
            - 0.0807259 * h[78] / H_AVG[78]
            + 0.07869312 * h[18] / H_AVG[18]
            - 0.07633744 * h[24] / H_AVG[24]
            - 0.05965805 * h[97] / H_AVG[97]
            - 0.05819276 * h[52] / H_AVG[52]
            + 0.03892142 * h[115] / H_AVG[115]
            + 0.03013509 * h[81] / H_AVG[81]
            + 0.005937079 * h[83] / H_AVG[83]
            - 0.003886679 * h[104] / H_AVG[104]
            - 0.001541452 * h[25] / H_AVG[25]
            - 0.0006679492 * h[70] / H_AVG[70]
            + 0.0004807689 * h[124] / H_AVG[124]
            + 0.0001016345 * h[0] / H_AVG[0]
            - 3.459993e-05 * h[90] / H_AVG[90]
            - 2.935222e-05 * h[99] / H_AVG[99]
            + 2.877323e-05 * h[123] / H_AVG[123]
            - 2.482994e-05 * h[21] / H_AVG[21]
            + 1.368356e-06 * h[84] / H_AVG[84]
            + 3.22898e-07 * h[55] / H_AVG[55]
            + 2.166465e-07 * h[79] / H_AVG[79]
            - 2.068936e-07 * h[105] / H_AVG[105]
            - 1.678719e-07 * h[106] / H_AVG[106]
            + 1.364133e-07 * h[113] / H_AVG[113]
            - 1.29596e-07 * h[34] / H_AVG[34]
            + 1.163443e-07 * h[73] / H_AVG[73]
            + 1.080881e-07 * h[10] / H_AVG[10]
            + 1.030329e-07 * h[11] / H_AVG[11]
            - 8.773762e-08 * h[6] / H_AVG[6]
            + 8.368828e-08 * h[86] / H_AVG[86]
            - 7.957367e-08 * h[32] / H_AVG[32]
            + 7.314801e-08 * h[17] / H_AVG[17]
            + 6.613409e-08 * h[38] / H_AVG[38]
            - 5.429866e-08 * h[42] / H_AVG[42]
            - 4.838347e-08 * h[54] / H_AVG[54]
            - 4.598455e-08 * h[29] / H_AVG[29]
            + 4.169356e-08 * h[118] / H_AVG[118]
            + 3.254728e-08 * h[48] / H_AVG[48]
            - 3.105484e-08 * h[45] / H_AVG[45]
            - 3.000112e-08 * h[39] / H_AVG[39]
            + 2.766297e-08 * h[27] / H_AVG[27]
            + 2.754364e-08 * h[63] / H_AVG[63]
            + 2.724781e-08 * h[30] / H_AVG[30]
            + 2.66863e-08 * h[43] / H_AVG[43]
            + 2.482492e-08 * h[121] / H_AVG[121]
            - 2.437674e-08 * h[80] / H_AVG[80]
            - 2.231379e-08 * h[126] / H_AVG[126]
            + 2.154637e-08 * h[60] / H_AVG[60]
            - 2.121918e-08 * h[61] / H_AVG[61]
            - 2.014783e-08 * h[102] / H_AVG[102]
            - 2.010177e-08 * h[107] / H_AVG[107]
            - 1.980607e-08 * h[7] / H_AVG[7]
            + 1.784114e-08 * h[92] / H_AVG[92]
            + 1.744722e-08 * h[35] / H_AVG[35]
            + 1.69653e-08 * h[15] / H_AVG[15]
            - 1.638734e-08 * h[59] / H_AVG[59]
            - 1.580935e-08 * h[101] / H_AVG[101]
            - 1.510838e-08 * h[28] / H_AVG[28]
            + 1.465341e-08 * h[103] / H_AVG[103]
            + 1.379007e-08 * h[69] / H_AVG[69]
            + 1.33841e-08 * h[67] / H_AVG[67]
            + 1.296745e-08 * h[1] / H_AVG[1]
            + 1.287308e-08 * h[100] / H_AVG[100]
            - 1.182329e-08 * h[110] / H_AVG[110]
            - 1.143903e-08 * h[3] / H_AVG[3]
            + 1.098536e-08 * h[37] / H_AVG[37]
            - 1.075475e-08 * h[50] / H_AVG[50]
            + 1.011849e-08 * h[44] / H_AVG[44]
            + 1.010962e-08 * h[36] / H_AVG[36]
            - 9.93401e-09 * h[12] / H_AVG[12]
            - 9.448806e-09 * h[114] / H_AVG[114]
            - 9.365811e-09 * h[2] / H_AVG[2]
            + 9.033148e-09 * h[58] / H_AVG[58]
            - 8.99381e-09 * h[112] / H_AVG[112]
            + 8.802991e-09 * h[76] / H_AVG[76]
            - 8.487e-09 * h[13] / H_AVG[13]
            - 8.249457e-09 * h[19] / H_AVG[19]
            - 8.089422e-09 * h[111] / H_AVG[111]
            + 7.709684e-09 * h[74] / H_AVG[74]
            + 7.318968e-09 * h[108] / H_AVG[108]
            + 6.877587e-09 * h[5] / H_AVG[5]
            - 6.573455e-09 * h[8] / H_AVG[8]
            - 6.303298e-09 * h[91] / H_AVG[91]
            - 5.876647e-09 * h[4] / H_AVG[4]
            - 5.629112e-09 * h[117] / H_AVG[117]
            + 4.75408e-09 * h[53] / H_AVG[53]
            - 4.700625e-09 * h[66] / H_AVG[66]
            - 4.282237e-09 * h[33] / H_AVG[33]
            - 4.158869e-09 * h[65] / H_AVG[65]
            + 3.846076e-09 * h[82] / H_AVG[82]
            - 3.615736e-09 * h[31] / H_AVG[31]
            - 3.320149e-09 * h[71] / H_AVG[71]
            + 3.084076e-09 * h[94] / H_AVG[94]
            + 2.979283e-09 * h[9] / H_AVG[9]
            + 2.61734e-09 * h[26] / H_AVG[26]
            - 2.346912e-09 * h[14] / H_AVG[14]
            + 2.32477e-09 * h[57] / H_AVG[57]
            + 2.197698e-09 * h[49] / H_AVG[49]
            + 2.147763e-09 * h[77] / H_AVG[77]
            - 2.142776e-09 * h[22] / H_AVG[22]
            + 1.987185e-09 * h[122] / H_AVG[122]
            + 1.606822e-09 * h[72] / H_AVG[72]
            + 1.538752e-09 * h[51] / H_AVG[51]
            - 1.514843e-09 * h[23] / H_AVG[23]
            - 1.046477e-09 * h[89] / H_AVG[89]
            + 1.022727e-09 * h[125] / H_AVG[125]
            - 9.594518e-10 * h[75] / H_AVG[75]
            - 8.301268e-10 * h[87] / H_AVG[87]
            + 8.272958e-10 * h[93] / H_AVG[93]
            - 7.443567e-10 * h[40] / H_AVG[40]
            - 6.918774e-10 * h[20] / H_AVG[20]
            + 6.173639e-10 * h[119] / H_AVG[119]
            - 5.654315e-10 * h[62] / H_AVG[62]
            - 5.082981e-10 * h[95] / H_AVG[95]
            - 4.710105e-10 * h[96] / H_AVG[96]
            + 2.777995e-10 * h[116] / H_AVG[116]
            + 2.651368e-10 * h[56] / H_AVG[56]
            + 1.842855e-10 * h[85] / H_AVG[85]
            + 1.097846e-10 * h[109] / H_AVG[109]
            - 9.55796e-11 * h[98] / H_AVG[98]
            - 7.637601e-11 * h[88] / H_AVG[88]
            - 5.142308e-11 * h[46] / H_AVG[46]
            - 4.858524e-11 * h[47] / H_AVG[47]
            - 2.637906e-11 * h[64] / H_AVG[64]
            + 1.163713e-11 * h[127] / H_AVG[127]
            + 1.02898e-12 * h[41] / H_AVG[41]
        ),
        0.1415668 + T[2] * (   # class Hcc
            + 0.2047411 * h[18] / H_AVG[18]
            - 0.1703606 * h[83] / H_AVG[83]
            - 0.1662015 * h[24] / H_AVG[24]
            - 0.1102769 * h[81] / H_AVG[81]
            + 0.06413342 * h[78] / H_AVG[78]
            - 0.06288482 * h[16] / H_AVG[16]
            - 0.05201706 * h[97] / H_AVG[97]
            + 0.0486383 * h[104] / H_AVG[104]
            + 0.03280098 * h[120] / H_AVG[120]
            - 0.02703513 * h[68] / H_AVG[68]
            - 0.02472933 * h[52] / H_AVG[52]
            + 0.02146742 * h[115] / H_AVG[115]
            + 0.01142905 * h[124] / H_AVG[124]
            - 0.001536834 * h[25] / H_AVG[25]
            + 0.0009424447 * h[70] / H_AVG[70]
            - 0.000504591 * h[123] / H_AVG[123]
            + 0.00021957 * h[0] / H_AVG[0]
            - 4.378385e-05 * h[99] / H_AVG[99]
            - 2.909792e-05 * h[21] / H_AVG[21]
            + 2.057688e-06 * h[84] / H_AVG[84]
            + 1.597967e-06 * h[90] / H_AVG[90]
            + 4.856624e-07 * h[55] / H_AVG[55]
            + 3.258748e-07 * h[79] / H_AVG[79]
            - 3.110402e-07 * h[105] / H_AVG[105]
            - 2.524837e-07 * h[106] / H_AVG[106]
            + 2.051928e-07 * h[113] / H_AVG[113]
            - 1.948337e-07 * h[34] / H_AVG[34]
            + 1.749617e-07 * h[73] / H_AVG[73]
            + 1.625519e-07 * h[10] / H_AVG[10]
            + 1.548944e-07 * h[11] / H_AVG[11]
            - 1.318942e-07 * h[6] / H_AVG[6]
            + 1.258309e-07 * h[86] / H_AVG[86]
            - 1.19679e-07 * h[32] / H_AVG[32]
            + 1.100393e-07 * h[17] / H_AVG[17]
            + 9.939132e-08 * h[38] / H_AVG[38]
            - 8.163112e-08 * h[42] / H_AVG[42]
            - 7.276398e-08 * h[54] / H_AVG[54]
            - 6.916431e-08 * h[29] / H_AVG[29]
            + 6.27002e-08 * h[118] / H_AVG[118]
            + 4.893875e-08 * h[48] / H_AVG[48]
            - 4.669941e-08 * h[45] / H_AVG[45]
            - 4.511233e-08 * h[39] / H_AVG[39]
            + 4.159886e-08 * h[27] / H_AVG[27]
            + 4.142457e-08 * h[63] / H_AVG[63]
            + 4.09677e-08 * h[30] / H_AVG[30]
            + 4.011423e-08 * h[43] / H_AVG[43]
            + 3.733974e-08 * h[121] / H_AVG[121]
            - 3.666472e-08 * h[80] / H_AVG[80]
            - 3.354892e-08 * h[126] / H_AVG[126]
            + 3.239891e-08 * h[60] / H_AVG[60]
            - 3.190393e-08 * h[61] / H_AVG[61]
            - 3.029168e-08 * h[102] / H_AVG[102]
            - 3.021614e-08 * h[107] / H_AVG[107]
            - 2.979633e-08 * h[7] / H_AVG[7]
            + 2.682428e-08 * h[92] / H_AVG[92]
            + 2.623636e-08 * h[35] / H_AVG[35]
            + 2.551303e-08 * h[15] / H_AVG[15]
            - 2.46397e-08 * h[59] / H_AVG[59]
            - 2.376061e-08 * h[101] / H_AVG[101]
            - 2.272292e-08 * h[28] / H_AVG[28]
            + 2.203697e-08 * h[103] / H_AVG[103]
            + 2.073908e-08 * h[69] / H_AVG[69]
            + 2.012985e-08 * h[67] / H_AVG[67]
            + 1.950265e-08 * h[1] / H_AVG[1]
            + 1.935693e-08 * h[100] / H_AVG[100]
            - 1.777877e-08 * h[110] / H_AVG[110]
            - 1.720141e-08 * h[3] / H_AVG[3]
            + 1.651797e-08 * h[37] / H_AVG[37]
            - 1.617802e-08 * h[50] / H_AVG[50]
            + 1.521468e-08 * h[44] / H_AVG[44]
            + 1.520429e-08 * h[36] / H_AVG[36]
            - 1.493714e-08 * h[12] / H_AVG[12]
            - 1.42103e-08 * h[114] / H_AVG[114]
            - 1.409145e-08 * h[2] / H_AVG[2]
            + 1.358617e-08 * h[58] / H_AVG[58]
            - 1.352929e-08 * h[112] / H_AVG[112]
            + 1.323937e-08 * h[76] / H_AVG[76]
            - 1.276154e-08 * h[13] / H_AVG[13]
            - 1.239927e-08 * h[19] / H_AVG[19]
            - 1.216112e-08 * h[111] / H_AVG[111]
            + 1.159276e-08 * h[74] / H_AVG[74]
            + 1.100247e-08 * h[108] / H_AVG[108]
            + 1.034659e-08 * h[5] / H_AVG[5]
            - 9.887102e-09 * h[8] / H_AVG[8]
            - 9.481839e-09 * h[91] / H_AVG[91]
            - 8.836175e-09 * h[4] / H_AVG[4]
            - 8.464032e-09 * h[117] / H_AVG[117]
            + 7.153958e-09 * h[53] / H_AVG[53]
            - 7.073795e-09 * h[66] / H_AVG[66]
            - 6.437687e-09 * h[33] / H_AVG[33]
            - 6.256232e-09 * h[65] / H_AVG[65]
            + 5.78481e-09 * h[82] / H_AVG[82]
            - 5.432622e-09 * h[31] / H_AVG[31]
            - 4.992336e-09 * h[71] / H_AVG[71]
            + 4.639508e-09 * h[94] / H_AVG[94]
            + 4.479685e-09 * h[9] / H_AVG[9]
            + 3.933579e-09 * h[26] / H_AVG[26]
            - 3.530984e-09 * h[14] / H_AVG[14]
            + 3.497133e-09 * h[57] / H_AVG[57]
            + 3.303683e-09 * h[49] / H_AVG[49]
            + 3.227542e-09 * h[77] / H_AVG[77]
            - 3.22325e-09 * h[22] / H_AVG[22]
            + 2.989592e-09 * h[122] / H_AVG[122]
            + 2.415139e-09 * h[72] / H_AVG[72]
            + 2.315279e-09 * h[51] / H_AVG[51]
            - 2.278853e-09 * h[23] / H_AVG[23]
            - 1.57297e-09 * h[89] / H_AVG[89]
            + 1.537129e-09 * h[125] / H_AVG[125]
            - 1.442881e-09 * h[75] / H_AVG[75]
            - 1.248462e-09 * h[87] / H_AVG[87]
            + 1.243922e-09 * h[93] / H_AVG[93]
            - 1.119131e-09 * h[40] / H_AVG[40]
            - 1.040616e-09 * h[20] / H_AVG[20]
            + 9.290942e-10 * h[119] / H_AVG[119]
            - 8.501814e-10 * h[62] / H_AVG[62]
            - 7.641427e-10 * h[95] / H_AVG[95]
            - 7.081649e-10 * h[96] / H_AVG[96]
            + 4.178541e-10 * h[116] / H_AVG[116]
            + 3.989422e-10 * h[56] / H_AVG[56]
            + 2.774326e-10 * h[85] / H_AVG[85]
            + 1.650263e-10 * h[109] / H_AVG[109]
            - 1.436969e-10 * h[98] / H_AVG[98]
            - 1.146921e-10 * h[88] / H_AVG[88]
            - 7.735941e-11 * h[46] / H_AVG[46]
            - 7.300432e-11 * h[47] / H_AVG[47]
            - 4.609607e-11 * h[64] / H_AVG[64]
            + 1.75396e-11 * h[127] / H_AVG[127]
            + 1.551724e-12 * h[41] / H_AVG[41]
        ),
        0.1286925 + T[3] * (   # class Hgg
            - 0.1984484 * h[104] / H_AVG[104]
            - 0.1765679 * h[68] / H_AVG[68]
            - 0.1111249 * h[18] / H_AVG[18]
            - 0.1063256 * h[16] / H_AVG[16]
            + 0.0891848 * h[52] / H_AVG[52]
            - 0.07659659 * h[97] / H_AVG[97]
            - 0.06820671 * h[81] / H_AVG[81]
            - 0.0574865 * h[24] / H_AVG[24]
            + 0.05196201 * h[115] / H_AVG[115]
            - 0.0266364 * h[78] / H_AVG[78]
            + 0.01347354 * h[83] / H_AVG[83]
            + 0.01188433 * h[120] / H_AVG[120]
            + 0.01015298 * h[124] / H_AVG[124]
            - 0.0009652066 * h[25] / H_AVG[25]
            + 0.0005531452 * h[70] / H_AVG[70]
            + 0.0002420836 * h[0] / H_AVG[0]
            - 8.606183e-05 * h[123] / H_AVG[123]
            - 3.926484e-05 * h[99] / H_AVG[99]
            + 2.957069e-05 * h[90] / H_AVG[90]
            - 2.819201e-05 * h[21] / H_AVG[21]
            + 1.855566e-06 * h[84] / H_AVG[84]
            + 4.378378e-07 * h[55] / H_AVG[55]
            + 2.938319e-07 * h[79] / H_AVG[79]
            - 2.805177e-07 * h[105] / H_AVG[105]
            - 2.276358e-07 * h[106] / H_AVG[106]
            + 1.850365e-07 * h[113] / H_AVG[113]
            - 1.75735e-07 * h[34] / H_AVG[34]
            + 1.57778e-07 * h[73] / H_AVG[73]
            + 1.465373e-07 * h[10] / H_AVG[10]
            + 1.396451e-07 * h[11] / H_AVG[11]
            - 1.189364e-07 * h[6] / H_AVG[6]
            + 1.134829e-07 * h[86] / H_AVG[86]
            - 1.079447e-07 * h[32] / H_AVG[32]
            + 9.922508e-08 * h[17] / H_AVG[17]
            + 8.967567e-08 * h[38] / H_AVG[38]
            - 7.361112e-08 * h[42] / H_AVG[42]
            - 6.56092e-08 * h[54] / H_AVG[54]
            - 6.235298e-08 * h[29] / H_AVG[29]
            + 5.654687e-08 * h[118] / H_AVG[118]
            + 4.413394e-08 * h[48] / H_AVG[48]
            - 4.211799e-08 * h[45] / H_AVG[45]
            - 4.067324e-08 * h[39] / H_AVG[39]
            + 3.750834e-08 * h[27] / H_AVG[27]
            + 3.734418e-08 * h[63] / H_AVG[63]
            + 3.69421e-08 * h[30] / H_AVG[30]
            + 3.617983e-08 * h[43] / H_AVG[43]
            + 3.366384e-08 * h[121] / H_AVG[121]
            - 3.30561e-08 * h[80] / H_AVG[80]
            - 3.025757e-08 * h[126] / H_AVG[126]
            + 2.921013e-08 * h[60] / H_AVG[60]
            - 2.87753e-08 * h[61] / H_AVG[61]
            - 2.731805e-08 * h[102] / H_AVG[102]
            - 2.726707e-08 * h[107] / H_AVG[107]
            - 2.685276e-08 * h[7] / H_AVG[7]
            + 2.418779e-08 * h[92] / H_AVG[92]
            + 2.365755e-08 * h[35] / H_AVG[35]
            + 2.300147e-08 * h[15] / H_AVG[15]
            - 2.222107e-08 * h[59] / H_AVG[59]
            - 2.143022e-08 * h[101] / H_AVG[101]
            - 2.049003e-08 * h[28] / H_AVG[28]
            + 1.987056e-08 * h[103] / H_AVG[103]
            + 1.869965e-08 * h[69] / H_AVG[69]
            + 1.814639e-08 * h[67] / H_AVG[67]
            + 1.758111e-08 * h[1] / H_AVG[1]
            + 1.745037e-08 * h[100] / H_AVG[100]
            - 1.602657e-08 * h[110] / H_AVG[110]
            - 1.550882e-08 * h[3] / H_AVG[3]
            + 1.489521e-08 * h[37] / H_AVG[37]
            - 1.459488e-08 * h[50] / H_AVG[50]
            + 1.371796e-08 * h[44] / H_AVG[44]
            + 1.370611e-08 * h[36] / H_AVG[36]
            - 1.347379e-08 * h[12] / H_AVG[12]
            - 1.281535e-08 * h[114] / H_AVG[114]
            - 1.270559e-08 * h[2] / H_AVG[2]
            + 1.225169e-08 * h[58] / H_AVG[58]
            - 1.220175e-08 * h[112] / H_AVG[112]
            + 1.194093e-08 * h[76] / H_AVG[76]
            - 1.150736e-08 * h[13] / H_AVG[13]
            - 1.118214e-08 * h[19] / H_AVG[19]
            - 1.096809e-08 * h[111] / H_AVG[111]
            + 1.045374e-08 * h[74] / H_AVG[74]
            + 9.920578e-09 * h[108] / H_AVG[108]
            + 9.328875e-09 * h[5] / H_AVG[5]
            - 8.915626e-09 * h[8] / H_AVG[8]
            - 8.545696e-09 * h[91] / H_AVG[91]
            - 7.968459e-09 * h[4] / H_AVG[4]
            - 7.631947e-09 * h[117] / H_AVG[117]
            + 6.447729e-09 * h[53] / H_AVG[53]
            - 6.376751e-09 * h[66] / H_AVG[66]
            - 5.80562e-09 * h[33] / H_AVG[33]
            - 5.633188e-09 * h[65] / H_AVG[65]
            + 5.215024e-09 * h[82] / H_AVG[82]
            - 4.900007e-09 * h[31] / H_AVG[31]
            - 4.503604e-09 * h[71] / H_AVG[71]
            + 4.182236e-09 * h[94] / H_AVG[94]
            + 4.039968e-09 * h[9] / H_AVG[9]
            + 3.549234e-09 * h[26] / H_AVG[26]
            - 3.182296e-09 * h[14] / H_AVG[14]
            + 3.153219e-09 * h[57] / H_AVG[57]
            + 2.97953e-09 * h[49] / H_AVG[49]
            + 2.911496e-09 * h[77] / H_AVG[77]
            - 2.904317e-09 * h[22] / H_AVG[22]
            + 2.694439e-09 * h[122] / H_AVG[122]
            + 2.178765e-09 * h[72] / H_AVG[72]
            + 2.087887e-09 * h[51] / H_AVG[51]
            - 2.053851e-09 * h[23] / H_AVG[23]
            - 1.418706e-09 * h[89] / H_AVG[89]
            + 1.386014e-09 * h[125] / H_AVG[125]
            - 1.301352e-09 * h[75] / H_AVG[75]
            - 1.1256e-09 * h[87] / H_AVG[87]
            + 1.121726e-09 * h[93] / H_AVG[93]
            - 1.009282e-09 * h[40] / H_AVG[40]
            - 9.382691e-10 * h[20] / H_AVG[20]
            + 8.376596e-10 * h[119] / H_AVG[119]
            - 7.665708e-10 * h[62] / H_AVG[62]
            - 6.891093e-10 * h[95] / H_AVG[95]
            - 6.386761e-10 * h[96] / H_AVG[96]
            + 3.766109e-10 * h[116] / H_AVG[116]
            + 3.595093e-10 * h[56] / H_AVG[56]
            + 2.501409e-10 * h[85] / H_AVG[85]
            + 1.488337e-10 * h[109] / H_AVG[109]
            - 1.295713e-10 * h[98] / H_AVG[98]
            - 1.035898e-10 * h[88] / H_AVG[88]
            - 6.974739e-11 * h[46] / H_AVG[46]
            - 6.584896e-11 * h[47] / H_AVG[47]
            - 3.98584e-11 * h[64] / H_AVG[64]
            + 1.581821e-11 * h[127] / H_AVG[127]
            + 1.397252e-12 * h[41] / H_AVG[41]
        ),
        -0.07702489 + T[4] * (   # class H4q
            - 0.2904596 * h[115] / H_AVG[115]
            - 0.09180836 * h[16] / H_AVG[16]
            - 0.091754 * h[78] / H_AVG[78]
            - 0.08943859 * h[104] / H_AVG[104]
            - 0.08673934 * h[24] / H_AVG[24]
            + 0.08237207 * h[52] / H_AVG[52]
            - 0.07109873 * h[97] / H_AVG[97]
            - 0.06534383 * h[81] / H_AVG[81]
            + 0.05061819 * h[68] / H_AVG[68]
            - 0.04846424 * h[18] / H_AVG[18]
            + 0.01200446 * h[120] / H_AVG[120]
            + 0.009187405 * h[124] / H_AVG[124]
            - 0.00885098 * h[83] / H_AVG[83]
            - 0.0008962909 * h[25] / H_AVG[25]
            + 0.000625996 * h[70] / H_AVG[70]
            + 0.0002385848 * h[0] / H_AVG[0]
            - 3.486367e-05 * h[99] / H_AVG[99]
            - 2.807892e-05 * h[21] / H_AVG[21]
            + 2.576361e-05 * h[123] / H_AVG[123]
            - 5.477167e-06 * h[90] / H_AVG[90]
            + 1.649911e-06 * h[84] / H_AVG[84]
            + 3.894872e-07 * h[55] / H_AVG[55]
            + 2.613091e-07 * h[79] / H_AVG[79]
            - 2.494603e-07 * h[105] / H_AVG[105]
            - 2.025059e-07 * h[106] / H_AVG[106]
            + 1.645107e-07 * h[113] / H_AVG[113]
            - 1.562793e-07 * h[34] / H_AVG[34]
            + 1.403168e-07 * h[73] / H_AVG[73]
            + 1.30333e-07 * h[10] / H_AVG[10]
            + 1.241783e-07 * h[11] / H_AVG[11]
            - 1.057823e-07 * h[6] / H_AVG[6]
            + 1.009189e-07 * h[86] / H_AVG[86]
            - 9.598035e-08 * h[32] / H_AVG[32]
            + 8.82468e-08 * h[17] / H_AVG[17]
            + 7.974831e-08 * h[38] / H_AVG[38]
            - 6.546574e-08 * h[42] / H_AVG[42]
            - 5.835842e-08 * h[54] / H_AVG[54]
            - 5.546384e-08 * h[29] / H_AVG[29]
            + 5.029051e-08 * h[118] / H_AVG[118]
            + 3.92446e-08 * h[48] / H_AVG[48]
            - 3.745481e-08 * h[45] / H_AVG[45]
            - 3.617161e-08 * h[39] / H_AVG[39]
            + 3.335774e-08 * h[27] / H_AVG[27]
            + 3.321016e-08 * h[63] / H_AVG[63]
            + 3.285609e-08 * h[30] / H_AVG[30]
            + 3.217555e-08 * h[43] / H_AVG[43]
            + 2.993733e-08 * h[121] / H_AVG[121]
            - 2.940132e-08 * h[80] / H_AVG[80]
            - 2.691191e-08 * h[126] / H_AVG[126]
            + 2.598678e-08 * h[60] / H_AVG[60]
            - 2.55836e-08 * h[61] / H_AVG[61]
            - 2.429259e-08 * h[102] / H_AVG[102]
            - 2.424948e-08 * h[107] / H_AVG[107]
            - 2.388336e-08 * h[7] / H_AVG[7]
            + 2.150915e-08 * h[92] / H_AVG[92]
            + 2.104156e-08 * h[35] / H_AVG[35]
            + 2.045854e-08 * h[15] / H_AVG[15]
            - 1.975515e-08 * h[59] / H_AVG[59]
            - 1.905832e-08 * h[101] / H_AVG[101]
            - 1.822342e-08 * h[28] / H_AVG[28]
            + 1.767059e-08 * h[103] / H_AVG[103]
            + 1.662911e-08 * h[69] / H_AVG[69]
            + 1.614124e-08 * h[67] / H_AVG[67]
            + 1.563515e-08 * h[1] / H_AVG[1]
            + 1.55181e-08 * h[100] / H_AVG[100]
            - 1.425552e-08 * h[110] / H_AVG[110]
            - 1.378978e-08 * h[3] / H_AVG[3]
            + 1.324671e-08 * h[37] / H_AVG[37]
            - 1.296926e-08 * h[50] / H_AVG[50]
            + 1.219933e-08 * h[44] / H_AVG[44]
            + 1.218887e-08 * h[36] / H_AVG[36]
            - 1.197935e-08 * h[12] / H_AVG[12]
            - 1.139811e-08 * h[114] / H_AVG[114]
            - 1.130152e-08 * h[2] / H_AVG[2]
            + 1.089476e-08 * h[58] / H_AVG[58]
            - 1.084922e-08 * h[112] / H_AVG[112]
            + 1.061867e-08 * h[76] / H_AVG[76]
            - 1.023239e-08 * h[13] / H_AVG[13]
            - 9.941103e-09 * h[19] / H_AVG[19]
            - 9.753483e-09 * h[111] / H_AVG[111]
            + 9.295521e-09 * h[74] / H_AVG[74]
            + 8.821215e-09 * h[108] / H_AVG[108]
            + 8.296942e-09 * h[5] / H_AVG[5]
            - 7.928962e-09 * h[8] / H_AVG[8]
            - 7.604007e-09 * h[91] / H_AVG[91]
            - 7.086084e-09 * h[4] / H_AVG[4]
            - 6.788397e-09 * h[117] / H_AVG[117]
            + 5.733289e-09 * h[53] / H_AVG[53]
            - 5.671167e-09 * h[66] / H_AVG[66]
            - 5.162178e-09 * h[33] / H_AVG[33]
            - 5.010763e-09 * h[65] / H_AVG[65]
            + 4.639178e-09 * h[82] / H_AVG[82]
            - 4.357897e-09 * h[31] / H_AVG[31]
            - 4.005039e-09 * h[71] / H_AVG[71]
            + 3.719735e-09 * h[94] / H_AVG[94]
            + 3.592475e-09 * h[9] / H_AVG[9]
            + 3.156284e-09 * h[26] / H_AVG[26]
            - 2.829619e-09 * h[14] / H_AVG[14]
            + 2.804122e-09 * h[57] / H_AVG[57]
            + 2.64966e-09 * h[49] / H_AVG[49]
            + 2.591328e-09 * h[77] / H_AVG[77]
            - 2.584637e-09 * h[22] / H_AVG[22]
            + 2.396726e-09 * h[122] / H_AVG[122]
            + 1.937584e-09 * h[72] / H_AVG[72]
            + 1.856443e-09 * h[51] / H_AVG[51]
            - 1.826268e-09 * h[23] / H_AVG[23]
            - 1.261564e-09 * h[89] / H_AVG[89]
            + 1.232832e-09 * h[125] / H_AVG[125]
            - 1.157234e-09 * h[75] / H_AVG[75]
            - 1.001129e-09 * h[87] / H_AVG[87]
            + 9.9816e-10 * h[93] / H_AVG[93]
            - 8.976077e-10 * h[40] / H_AVG[40]
            - 8.343567e-10 * h[20] / H_AVG[20]
            + 7.453421e-10 * h[119] / H_AVG[119]
            - 6.819214e-10 * h[62] / H_AVG[62]
            - 6.130783e-10 * h[95] / H_AVG[95]
            - 5.681079e-10 * h[96] / H_AVG[96]
            + 3.34985e-10 * h[116] / H_AVG[116]
            + 3.199496e-10 * h[56] / H_AVG[56]
            + 2.223422e-10 * h[85] / H_AVG[85]
            + 1.323951e-10 * h[109] / H_AVG[109]
            - 1.152338e-10 * h[98] / H_AVG[98]
            - 9.20738e-11 * h[88] / H_AVG[88]
            - 6.203633e-11 * h[46] / H_AVG[46]
            - 5.856635e-11 * h[47] / H_AVG[47]
            - 3.619109e-11 * h[64] / H_AVG[64]
            + 1.405015e-11 * h[127] / H_AVG[127]
            + 1.24378e-12 * h[41] / H_AVG[41]
        ),
        -0.9789527 + T[5] * (   # class Hqql
            + 0.1850586 * h[120] / H_AVG[120]
            + 0.1667908 * h[24] / H_AVG[24]
            + 0.1322397 * h[97] / H_AVG[97]
            + 0.1198355 * h[18] / H_AVG[18]
            + 0.09548738 * h[52] / H_AVG[52]
            + 0.08015776 * h[78] / H_AVG[78]
            - 0.07404153 * h[115] / H_AVG[115]
            + 0.0630493 * h[83] / H_AVG[83]
            - 0.02394932 * h[81] / H_AVG[81]
            + 0.01866569 * h[68] / H_AVG[68]
            - 0.0148059 * h[124] / H_AVG[124]
            + 0.01403558 * h[104] / H_AVG[104]
            - 0.005815893 * h[16] / H_AVG[16]
            + 0.002723727 * h[25] / H_AVG[25]
            + 0.001601794 * h[70] / H_AVG[70]
            - 0.001137941 * h[123] / H_AVG[123]
            - 0.0003631769 * h[0] / H_AVG[0]
            + 0.0001400992 * h[90] / H_AVG[90]
            + 7.554057e-05 * h[21] / H_AVG[21]
            - 2.155311e-05 * h[99] / H_AVG[99]
            + 1.007036e-06 * h[84] / H_AVG[84]
            + 2.375972e-07 * h[55] / H_AVG[55]
            + 1.593687e-07 * h[79] / H_AVG[79]
            - 1.52168e-07 * h[105] / H_AVG[105]
            - 1.235545e-07 * h[106] / H_AVG[106]
            + 1.00426e-07 * h[113] / H_AVG[113]
            - 9.535004e-08 * h[34] / H_AVG[34]
            + 8.558444e-08 * h[73] / H_AVG[73]
            + 7.946835e-08 * h[10] / H_AVG[10]
            + 7.582191e-08 * h[11] / H_AVG[11]
            - 6.448459e-08 * h[6] / H_AVG[6]
            + 6.155017e-08 * h[86] / H_AVG[86]
            - 5.854579e-08 * h[32] / H_AVG[32]
            + 5.384303e-08 * h[17] / H_AVG[17]
            + 4.862365e-08 * h[38] / H_AVG[38]
            - 3.992562e-08 * h[42] / H_AVG[42]
            - 3.562308e-08 * h[54] / H_AVG[54]
            - 3.382634e-08 * h[29] / H_AVG[29]
            + 3.068663e-08 * h[118] / H_AVG[118]
            + 2.392818e-08 * h[48] / H_AVG[48]
            - 2.285207e-08 * h[45] / H_AVG[45]
            - 2.205265e-08 * h[39] / H_AVG[39]
            + 2.035237e-08 * h[27] / H_AVG[27]
            + 2.026243e-08 * h[63] / H_AVG[63]
            + 2.004654e-08 * h[30] / H_AVG[30]
            + 1.961444e-08 * h[43] / H_AVG[43]
            + 1.825815e-08 * h[121] / H_AVG[121]
            - 1.793505e-08 * h[80] / H_AVG[80]
            - 1.643863e-08 * h[126] / H_AVG[126]
            + 1.584498e-08 * h[60] / H_AVG[60]
            - 1.55986e-08 * h[61] / H_AVG[61]
            - 1.481404e-08 * h[102] / H_AVG[102]
            - 1.476977e-08 * h[107] / H_AVG[107]
            - 1.457158e-08 * h[7] / H_AVG[7]
            + 1.311473e-08 * h[92] / H_AVG[92]
            + 1.282311e-08 * h[35] / H_AVG[35]
            + 1.247761e-08 * h[15] / H_AVG[15]
            - 1.206325e-08 * h[59] / H_AVG[59]
            - 1.163594e-08 * h[101] / H_AVG[101]
            - 1.109416e-08 * h[28] / H_AVG[28]
            + 1.078428e-08 * h[103] / H_AVG[103]
            + 1.015744e-08 * h[69] / H_AVG[69]
            + 9.863737e-09 * h[67] / H_AVG[67]
            + 9.533323e-09 * h[1] / H_AVG[1]
            + 9.460809e-09 * h[100] / H_AVG[100]
            - 8.689122e-09 * h[110] / H_AVG[110]
            - 8.419066e-09 * h[3] / H_AVG[3]
            + 8.082163e-09 * h[37] / H_AVG[37]
            - 7.907383e-09 * h[50] / H_AVG[50]
            + 7.440534e-09 * h[36] / H_AVG[36]
            + 7.436807e-09 * h[44] / H_AVG[44]
            - 7.31695e-09 * h[12] / H_AVG[12]
            - 6.953033e-09 * h[114] / H_AVG[114]
            - 6.901569e-09 * h[2] / H_AVG[2]
            + 6.652394e-09 * h[58] / H_AVG[58]
            - 6.631416e-09 * h[112] / H_AVG[112]
            + 6.48275e-09 * h[76] / H_AVG[76]
            - 6.235553e-09 * h[13] / H_AVG[13]
            - 6.066905e-09 * h[19] / H_AVG[19]
            - 5.944956e-09 * h[111] / H_AVG[111]
            + 5.669262e-09 * h[74] / H_AVG[74]
            + 5.370975e-09 * h[108] / H_AVG[108]
            + 5.05625e-09 * h[5] / H_AVG[5]
            - 4.841198e-09 * h[8] / H_AVG[8]
            - 4.646677e-09 * h[91] / H_AVG[91]
            - 4.325716e-09 * h[4] / H_AVG[4]
            - 4.142568e-09 * h[117] / H_AVG[117]
            + 3.500264e-09 * h[53] / H_AVG[53]
            - 3.458732e-09 * h[66] / H_AVG[66]
            - 3.140912e-09 * h[33] / H_AVG[33]
            - 3.067258e-09 * h[65] / H_AVG[65]
            + 2.835001e-09 * h[82] / H_AVG[82]
            - 2.650243e-09 * h[31] / H_AVG[31]
            - 2.440726e-09 * h[71] / H_AVG[71]
            + 2.268066e-09 * h[94] / H_AVG[94]
            + 2.191234e-09 * h[9] / H_AVG[9]
            + 1.934262e-09 * h[26] / H_AVG[26]
            - 1.727712e-09 * h[14] / H_AVG[14]
            + 1.71638e-09 * h[57] / H_AVG[57]
            + 1.614603e-09 * h[49] / H_AVG[49]
            + 1.579961e-09 * h[77] / H_AVG[77]
            - 1.575669e-09 * h[22] / H_AVG[22]
            + 1.466196e-09 * h[122] / H_AVG[122]
            + 1.180754e-09 * h[72] / H_AVG[72]
            + 1.136563e-09 * h[51] / H_AVG[51]
            - 1.116128e-09 * h[23] / H_AVG[23]
            - 7.654531e-10 * h[89] / H_AVG[89]
            + 7.506496e-10 * h[125] / H_AVG[125]
            - 7.04495e-10 * h[75] / H_AVG[75]
            + 6.090666e-10 * h[93] / H_AVG[93]
            - 6.079315e-10 * h[87] / H_AVG[87]
            - 5.47269e-10 * h[40] / H_AVG[40]
            - 5.088751e-10 * h[20] / H_AVG[20]
            + 4.560457e-10 * h[119] / H_AVG[119]
            - 4.148487e-10 * h[62] / H_AVG[62]
            - 3.737954e-10 * h[95] / H_AVG[95]
            - 3.456931e-10 * h[96] / H_AVG[96]
            + 2.040083e-10 * h[116] / H_AVG[116]
            + 1.958382e-10 * h[56] / H_AVG[56]
            + 1.365626e-10 * h[85] / H_AVG[85]
            + 8.048427e-11 * h[109] / H_AVG[109]
            - 7.008847e-11 * h[98] / H_AVG[98]
            - 5.579609e-11 * h[88] / H_AVG[88]
            - 3.808151e-11 * h[46] / H_AVG[46]
            - 3.56327e-11 * h[47] / H_AVG[47]
            - 1.704637e-11 * h[64] / H_AVG[64]
            + 8.872792e-12 * h[127] / H_AVG[127]
            + 7.894307e-13 * h[41] / H_AVG[41]
        ),
        0.3035426 + T[6] * (   # class Zqq
            + 0.1393098 * h[24] / H_AVG[24]
            + 0.1195941 * h[115] / H_AVG[115]
            - 0.1083664 * h[97] / H_AVG[97]
            + 0.1051606 * h[104] / H_AVG[104]
            + 0.0967604 * h[68] / H_AVG[68]
            - 0.09478558 * h[120] / H_AVG[120]
            + 0.09115734 * h[81] / H_AVG[81]
            + 0.07661285 * h[83] / H_AVG[83]
            - 0.05925887 * h[52] / H_AVG[52]
            - 0.05170511 * h[16] / H_AVG[16]
            - 0.0208043 * h[124] / H_AVG[124]
            + 0.02026689 * h[18] / H_AVG[18]
            - 0.01178362 * h[78] / H_AVG[78]
            + 0.002182552 * h[25] / H_AVG[25]
            - 0.0015021 * h[70] / H_AVG[70]
            + 0.0005168576 * h[123] / H_AVG[123]
            - 6.908394e-05 * h[0] / H_AVG[0]
            - 6.800189e-05 * h[90] / H_AVG[90]
            - 4.912495e-05 * h[99] / H_AVG[99]
            - 3.924928e-05 * h[21] / H_AVG[21]
            + 2.313409e-06 * h[84] / H_AVG[84]
            + 5.460934e-07 * h[55] / H_AVG[55]
            + 3.663494e-07 * h[79] / H_AVG[79]
            - 3.497844e-07 * h[105] / H_AVG[105]
            - 2.839238e-07 * h[106] / H_AVG[106]
            + 2.30658e-07 * h[113] / H_AVG[113]
            - 2.191273e-07 * h[34] / H_AVG[34]
            + 1.967346e-07 * h[73] / H_AVG[73]
            + 1.82775e-07 * h[10] / H_AVG[10]
            + 1.741421e-07 * h[11] / H_AVG[11]
            - 1.483291e-07 * h[6] / H_AVG[6]
            + 1.414966e-07 * h[86] / H_AVG[86]
            - 1.345694e-07 * h[32] / H_AVG[32]
            + 1.236859e-07 * h[17] / H_AVG[17]
            + 1.118036e-07 * h[38] / H_AVG[38]
            - 9.178966e-08 * h[42] / H_AVG[42]
            - 8.180998e-08 * h[54] / H_AVG[54]
            - 7.777007e-08 * h[29] / H_AVG[29]
            + 7.050818e-08 * h[118] / H_AVG[118]
            + 5.503017e-08 * h[48] / H_AVG[48]
            - 5.251521e-08 * h[45] / H_AVG[45]
            - 5.07311e-08 * h[39] / H_AVG[39]
            + 4.677268e-08 * h[27] / H_AVG[27]
            + 4.65681e-08 * h[63] / H_AVG[63]
            + 4.606286e-08 * h[30] / H_AVG[30]
            + 4.511346e-08 * h[43] / H_AVG[43]
            + 4.198366e-08 * h[121] / H_AVG[121]
            - 4.122351e-08 * h[80] / H_AVG[80]
            - 3.770987e-08 * h[126] / H_AVG[126]
            + 3.643087e-08 * h[60] / H_AVG[60]
            - 3.588324e-08 * h[61] / H_AVG[61]
            - 3.406847e-08 * h[102] / H_AVG[102]
            - 3.39862e-08 * h[107] / H_AVG[107]
            - 3.350401e-08 * h[7] / H_AVG[7]
            + 3.016484e-08 * h[92] / H_AVG[92]
            + 2.947795e-08 * h[35] / H_AVG[35]
            + 2.868235e-08 * h[15] / H_AVG[15]
            - 2.771015e-08 * h[59] / H_AVG[59]
            - 2.672504e-08 * h[101] / H_AVG[101]
            - 2.556214e-08 * h[28] / H_AVG[28]
            + 2.477661e-08 * h[103] / H_AVG[103]
            + 2.332004e-08 * h[69] / H_AVG[69]
            + 2.26301e-08 * h[67] / H_AVG[67]
            + 2.192567e-08 * h[1] / H_AVG[1]
            + 2.176355e-08 * h[100] / H_AVG[100]
            - 1.99894e-08 * h[110] / H_AVG[110]
            - 1.9333e-08 * h[3] / H_AVG[3]
            + 1.857403e-08 * h[37] / H_AVG[37]
            - 1.818815e-08 * h[50] / H_AVG[50]
            + 1.710566e-08 * h[44] / H_AVG[44]
            + 1.709139e-08 * h[36] / H_AVG[36]
            - 1.680426e-08 * h[12] / H_AVG[12]
            - 1.59795e-08 * h[114] / H_AVG[114]
            - 1.58351e-08 * h[2] / H_AVG[2]
            + 1.527535e-08 * h[58] / H_AVG[58]
            - 1.521367e-08 * h[112] / H_AVG[112]
            + 1.488764e-08 * h[76] / H_AVG[76]
            - 1.434927e-08 * h[13] / H_AVG[13]
            - 1.39436e-08 * h[19] / H_AVG[19]
            - 1.367661e-08 * h[111] / H_AVG[111]
            + 1.303407e-08 * h[74] / H_AVG[74]
            + 1.237054e-08 * h[108] / H_AVG[108]
            + 1.163332e-08 * h[5] / H_AVG[5]
            - 1.111711e-08 * h[8] / H_AVG[8]
            - 1.066114e-08 * h[91] / H_AVG[91]
            - 9.935988e-09 * h[4] / H_AVG[4]
            - 9.516986e-09 * h[117] / H_AVG[117]
            + 8.039724e-09 * h[53] / H_AVG[53]
            - 7.947899e-09 * h[66] / H_AVG[66]
            - 7.23783e-09 * h[33] / H_AVG[33]
            - 7.028715e-09 * h[65] / H_AVG[65]
            + 6.503822e-09 * h[82] / H_AVG[82]
            - 6.109723e-09 * h[31] / H_AVG[31]
            - 5.61426e-09 * h[71] / H_AVG[71]
            + 5.215036e-09 * h[94] / H_AVG[94]
            + 5.037178e-09 * h[9] / H_AVG[9]
            + 4.425115e-09 * h[26] / H_AVG[26]
            - 3.967993e-09 * h[14] / H_AVG[14]
            + 3.930521e-09 * h[57] / H_AVG[57]
            + 3.715049e-09 * h[49] / H_AVG[49]
            + 3.633175e-09 * h[77] / H_AVG[77]
            - 3.622913e-09 * h[22] / H_AVG[22]
            + 3.35995e-09 * h[122] / H_AVG[122]
            + 2.716973e-09 * h[72] / H_AVG[72]
            + 2.601455e-09 * h[51] / H_AVG[51]
            - 2.56241e-09 * h[23] / H_AVG[23]
            - 1.769387e-09 * h[89] / H_AVG[89]
            + 1.728595e-09 * h[125] / H_AVG[125]
            - 1.622728e-09 * h[75] / H_AVG[75]
            - 1.404279e-09 * h[87] / H_AVG[87]
            + 1.399505e-09 * h[93] / H_AVG[93]
            - 1.258517e-09 * h[40] / H_AVG[40]
            - 1.169865e-09 * h[20] / H_AVG[20]
            + 1.04491e-09 * h[119] / H_AVG[119]
            - 9.560266e-10 * h[62] / H_AVG[62]
            - 8.594232e-10 * h[95] / H_AVG[95]
            - 7.965274e-10 * h[96] / H_AVG[96]
            + 4.697426e-10 * h[116] / H_AVG[116]
            + 4.48303e-10 * h[56] / H_AVG[56]
            + 3.1164e-10 * h[85] / H_AVG[85]
            + 1.856491e-10 * h[109] / H_AVG[109]
            - 1.616034e-10 * h[98] / H_AVG[98]
            - 1.291613e-10 * h[88] / H_AVG[88]
            - 8.697564e-11 * h[46] / H_AVG[46]
            - 8.211195e-11 * h[47] / H_AVG[47]
            - 4.669804e-11 * h[64] / H_AVG[64]
            + 1.967789e-11 * h[127] / H_AVG[127]
            + 1.739807e-12 * h[41] / H_AVG[41]
        ),
        0.09887857 + T[7] * (   # class Wqq
            + 0.2721265 * h[97] / H_AVG[97]
            + 0.1215628 * h[24] / H_AVG[24]
            + 0.1099156 * h[68] / H_AVG[68]
            - 0.08826271 * h[52] / H_AVG[52]
            + 0.07951345 * h[83] / H_AVG[83]
            - 0.07576086 * h[120] / H_AVG[120]
            - 0.05456296 * h[81] / H_AVG[81]
            + 0.05262967 * h[115] / H_AVG[115]
            + 0.04927999 * h[104] / H_AVG[104]
            - 0.04451394 * h[78] / H_AVG[78]
            - 0.04407422 * h[16] / H_AVG[16]
            + 0.002237498 * h[124] / H_AVG[124]
            - 0.002222379 * h[25] / H_AVG[25]
            + 0.001792233 * h[18] / H_AVG[18]
            - 0.001096885 * h[70] / H_AVG[70]
            - 0.0001773242 * h[0] / H_AVG[0]
            + 0.0001519735 * h[123] / H_AVG[123]
            - 4.527095e-05 * h[90] / H_AVG[90]
            - 4.00312e-05 * h[99] / H_AVG[99]
            - 2.784345e-05 * h[21] / H_AVG[21]
            + 1.882746e-06 * h[84] / H_AVG[84]
            + 4.442963e-07 * h[55] / H_AVG[55]
            + 2.98137e-07 * h[79] / H_AVG[79]
            - 2.846498e-07 * h[105] / H_AVG[105]
            - 2.310351e-07 * h[106] / H_AVG[106]
            + 1.877338e-07 * h[113] / H_AVG[113]
            - 1.783171e-07 * h[34] / H_AVG[34]
            + 1.601128e-07 * h[73] / H_AVG[73]
            + 1.486883e-07 * h[10] / H_AVG[10]
            + 1.417132e-07 * h[11] / H_AVG[11]
            - 1.207114e-07 * h[6] / H_AVG[6]
            + 1.151608e-07 * h[86] / H_AVG[86]
            - 1.095076e-07 * h[32] / H_AVG[32]
            + 1.006406e-07 * h[17] / H_AVG[17]
            + 9.098559e-08 * h[38] / H_AVG[38]
            - 7.47038e-08 * h[42] / H_AVG[42]
            - 6.658532e-08 * h[54] / H_AVG[54]
            - 6.328214e-08 * h[29] / H_AVG[29]
            + 5.737617e-08 * h[118] / H_AVG[118]
            + 4.478454e-08 * h[48] / H_AVG[48]
            - 4.273757e-08 * h[45] / H_AVG[45]
            - 4.127314e-08 * h[39] / H_AVG[39]
            + 3.806391e-08 * h[27] / H_AVG[27]
            + 3.789367e-08 * h[63] / H_AVG[63]
            + 3.748774e-08 * h[30] / H_AVG[30]
            + 3.671083e-08 * h[43] / H_AVG[43]
            + 3.416195e-08 * h[121] / H_AVG[121]
            - 3.354383e-08 * h[80] / H_AVG[80]
            - 3.070284e-08 * h[126] / H_AVG[126]
            + 2.963863e-08 * h[60] / H_AVG[60]
            - 2.919235e-08 * h[61] / H_AVG[61]
            - 2.772288e-08 * h[102] / H_AVG[102]
            - 2.765935e-08 * h[107] / H_AVG[107]
            - 2.725254e-08 * h[7] / H_AVG[7]
            + 2.454557e-08 * h[92] / H_AVG[92]
            + 2.399209e-08 * h[35] / H_AVG[35]
            + 2.334104e-08 * h[15] / H_AVG[15]
            - 2.255829e-08 * h[59] / H_AVG[59]
            - 2.174638e-08 * h[101] / H_AVG[101]
            - 2.078911e-08 * h[28] / H_AVG[28]
            + 2.016421e-08 * h[103] / H_AVG[103]
            + 1.897526e-08 * h[69] / H_AVG[69]
            + 1.841304e-08 * h[67] / H_AVG[67]
            + 1.78399e-08 * h[1] / H_AVG[1]
            + 1.770903e-08 * h[100] / H_AVG[100]
            - 1.626198e-08 * h[110] / H_AVG[110]
            - 1.573643e-08 * h[3] / H_AVG[3]
            + 1.511465e-08 * h[37] / H_AVG[37]
            - 1.480313e-08 * h[50] / H_AVG[50]
            + 1.391903e-08 * h[44] / H_AVG[44]
            + 1.390904e-08 * h[36] / H_AVG[36]
            - 1.366928e-08 * h[12] / H_AVG[12]
            - 1.30022e-08 * h[114] / H_AVG[114]
            - 1.289515e-08 * h[2] / H_AVG[2]
            + 1.243059e-08 * h[58] / H_AVG[58]
            - 1.238138e-08 * h[112] / H_AVG[112]
            + 1.21149e-08 * h[76] / H_AVG[76]
            - 1.167564e-08 * h[13] / H_AVG[13]
            - 1.134656e-08 * h[19] / H_AVG[19]
            - 1.112858e-08 * h[111] / H_AVG[111]
            + 1.060612e-08 * h[74] / H_AVG[74]
            + 1.006756e-08 * h[108] / H_AVG[108]
            + 9.466513e-09 * h[5] / H_AVG[5]
            - 9.046489e-09 * h[8] / H_AVG[8]
            - 8.67285e-09 * h[91] / H_AVG[91]
            - 8.085367e-09 * h[4] / H_AVG[4]
            - 7.744629e-09 * h[117] / H_AVG[117]
            + 6.541147e-09 * h[53] / H_AVG[53]
            - 6.472767e-09 * h[66] / H_AVG[66]
            - 5.890391e-09 * h[33] / H_AVG[33]
            - 5.721192e-09 * h[65] / H_AVG[65]
            + 5.29152e-09 * h[82] / H_AVG[82]
            - 4.973797e-09 * h[31] / H_AVG[31]
            - 4.569697e-09 * h[71] / H_AVG[71]
            + 4.241994e-09 * h[94] / H_AVG[94]
            + 4.099057e-09 * h[9] / H_AVG[9]
            + 3.599203e-09 * h[26] / H_AVG[26]
            - 3.229552e-09 * h[14] / H_AVG[14]
            + 3.198659e-09 * h[57] / H_AVG[57]
            + 3.023194e-09 * h[49] / H_AVG[49]
            + 2.954129e-09 * h[77] / H_AVG[77]
            - 2.948645e-09 * h[22] / H_AVG[22]
            + 2.734249e-09 * h[122] / H_AVG[122]
            + 2.210941e-09 * h[72] / H_AVG[72]
            + 2.117923e-09 * h[51] / H_AVG[51]
            - 2.08596e-09 * h[23] / H_AVG[23]
            - 1.439901e-09 * h[89] / H_AVG[89]
            + 1.406479e-09 * h[125] / H_AVG[125]
            - 1.320677e-09 * h[75] / H_AVG[75]
            - 1.142127e-09 * h[87] / H_AVG[87]
            + 1.137978e-09 * h[93] / H_AVG[93]
            - 1.024085e-09 * h[40] / H_AVG[40]
            - 9.520034e-10 * h[20] / H_AVG[20]
            + 8.4954e-10 * h[119] / H_AVG[119]
            - 7.779426e-10 * h[62] / H_AVG[62]
            - 6.992886e-10 * h[95] / H_AVG[95]
            - 6.482403e-10 * h[96] / H_AVG[96]
            + 3.821883e-10 * h[116] / H_AVG[116]
            + 3.648268e-10 * h[56] / H_AVG[56]
            + 2.536821e-10 * h[85] / H_AVG[85]
            + 1.510304e-10 * h[109] / H_AVG[109]
            - 1.314916e-10 * h[98] / H_AVG[98]
            - 1.050541e-10 * h[88] / H_AVG[88]
            - 7.075573e-11 * h[46] / H_AVG[46]
            - 6.682018e-11 * h[47] / H_AVG[47]
            - 3.945799e-11 * h[64] / H_AVG[64]
            + 1.600557e-11 * h[127] / H_AVG[127]
            + 1.417343e-12 * h[41] / H_AVG[41]
        ),
        -0.4320669 + T[8] * (   # class Tbqq
            - 0.1782564 * h[104] / H_AVG[104]
            + 0.1568954 * h[16] / H_AVG[16]
            + 0.1367614 * h[68] / H_AVG[68]
            + 0.1228966 * h[120] / H_AVG[120]
            - 0.08780377 * h[83] / H_AVG[83]
            - 0.08708967 * h[18] / H_AVG[18]
            + 0.06138552 * h[78] / H_AVG[78]
            - 0.04331312 * h[24] / H_AVG[24]
            + 0.03986018 * h[81] / H_AVG[81]
            + 0.02960531 * h[97] / H_AVG[97]
            - 0.02713934 * h[52] / H_AVG[52]
            - 0.02514376 * h[115] / H_AVG[115]
            + 0.001513437 * h[124] / H_AVG[124]
            - 0.001183908 * h[70] / H_AVG[70]
            - 0.0007325985 * h[123] / H_AVG[123]
            + 0.0001926653 * h[25] / H_AVG[25]
            - 0.000129417 * h[0] / H_AVG[0]
            - 5.783614e-05 * h[90] / H_AVG[90]
            - 2.479153e-05 * h[99] / H_AVG[99]
            - 1.128437e-05 * h[21] / H_AVG[21]
            + 1.174626e-06 * h[84] / H_AVG[84]
            + 2.772571e-07 * h[55] / H_AVG[55]
            + 1.860814e-07 * h[79] / H_AVG[79]
            - 1.775655e-07 * h[105] / H_AVG[105]
            - 1.441143e-07 * h[106] / H_AVG[106]
            + 1.17133e-07 * h[113] / H_AVG[113]
            - 1.112455e-07 * h[34] / H_AVG[34]
            + 9.986652e-08 * h[73] / H_AVG[73]
            + 9.27722e-08 * h[10] / H_AVG[10]
            + 8.842365e-08 * h[11] / H_AVG[11]
            - 7.52969e-08 * h[6] / H_AVG[6]
            + 7.183978e-08 * h[86] / H_AVG[86]
            - 6.832091e-08 * h[32] / H_AVG[32]
            + 6.281764e-08 * h[17] / H_AVG[17]
            + 5.67667e-08 * h[38] / H_AVG[38]
            - 4.660815e-08 * h[42] / H_AVG[42]
            - 4.153757e-08 * h[54] / H_AVG[54]
            - 3.948392e-08 * h[29] / H_AVG[29]
            + 3.579421e-08 * h[118] / H_AVG[118]
            + 2.794162e-08 * h[48] / H_AVG[48]
            - 2.665664e-08 * h[45] / H_AVG[45]
            - 2.57482e-08 * h[39] / H_AVG[39]
            + 2.374716e-08 * h[27] / H_AVG[27]
            + 2.364391e-08 * h[63] / H_AVG[63]
            + 2.338756e-08 * h[30] / H_AVG[30]
            + 2.290014e-08 * h[43] / H_AVG[43]
            + 2.131646e-08 * h[121] / H_AVG[121]
            - 2.092969e-08 * h[80] / H_AVG[80]
            - 1.914816e-08 * h[126] / H_AVG[126]
            + 1.849678e-08 * h[60] / H_AVG[60]
            - 1.821277e-08 * h[61] / H_AVG[61]
            - 1.729112e-08 * h[102] / H_AVG[102]
            - 1.725494e-08 * h[107] / H_AVG[107]
            - 1.700398e-08 * h[7] / H_AVG[7]
            + 1.531127e-08 * h[92] / H_AVG[92]
            + 1.495864e-08 * h[35] / H_AVG[35]
            + 1.456212e-08 * h[15] / H_AVG[15]
            - 1.406491e-08 * h[59] / H_AVG[59]
            - 1.356804e-08 * h[101] / H_AVG[101]
            - 1.298033e-08 * h[28] / H_AVG[28]
            + 1.257604e-08 * h[103] / H_AVG[103]
            + 1.183659e-08 * h[69] / H_AVG[69]
            + 1.148899e-08 * h[67] / H_AVG[67]
            + 1.113245e-08 * h[1] / H_AVG[1]
            + 1.104892e-08 * h[100] / H_AVG[100]
            - 1.014884e-08 * h[110] / H_AVG[110]
            - 9.818467e-09 * h[3] / H_AVG[3]
            + 9.430038e-09 * h[37] / H_AVG[37]
            - 9.231269e-09 * h[50] / H_AVG[50]
            + 8.684162e-09 * h[44] / H_AVG[44]
            + 8.676141e-09 * h[36] / H_AVG[36]
            - 8.530291e-09 * h[12] / H_AVG[12]
            - 8.112494e-09 * h[114] / H_AVG[114]
            - 8.044388e-09 * h[2] / H_AVG[2]
            + 7.755735e-09 * h[58] / H_AVG[58]
            - 7.723728e-09 * h[112] / H_AVG[112]
            + 7.558639e-09 * h[76] / H_AVG[76]
            - 7.283713e-09 * h[13] / H_AVG[13]
            - 7.07798e-09 * h[19] / H_AVG[19]
            - 6.943829e-09 * h[111] / H_AVG[111]
            + 6.618137e-09 * h[74] / H_AVG[74]
            + 6.280649e-09 * h[108] / H_AVG[108]
            + 5.904996e-09 * h[5] / H_AVG[5]
            - 5.643715e-09 * h[8] / H_AVG[8]
            - 5.410248e-09 * h[91] / H_AVG[91]
            - 5.044717e-09 * h[4] / H_AVG[4]
            - 4.830963e-09 * h[117] / H_AVG[117]
            + 4.081664e-09 * h[53] / H_AVG[53]
            - 4.03472e-09 * h[66] / H_AVG[66]
            - 3.674841e-09 * h[33] / H_AVG[33]
            - 3.569693e-09 * h[65] / H_AVG[65]
            + 3.301534e-09 * h[82] / H_AVG[82]
            - 3.100781e-09 * h[31] / H_AVG[31]
            - 2.850003e-09 * h[71] / H_AVG[71]
            + 2.648727e-09 * h[94] / H_AVG[94]
            + 2.557254e-09 * h[9] / H_AVG[9]
            + 2.247832e-09 * h[26] / H_AVG[26]
            - 2.014383e-09 * h[14] / H_AVG[14]
            + 1.996449e-09 * h[57] / H_AVG[57]
            + 1.88616e-09 * h[49] / H_AVG[49]
            + 1.844525e-09 * h[77] / H_AVG[77]
            - 1.839285e-09 * h[22] / H_AVG[22]
            + 1.7058e-09 * h[122] / H_AVG[122]
            + 1.379332e-09 * h[72] / H_AVG[72]
            + 1.321164e-09 * h[51] / H_AVG[51]
            - 1.298993e-09 * h[23] / H_AVG[23]
            - 8.981885e-10 * h[89] / H_AVG[89]
            + 8.775609e-10 * h[125] / H_AVG[125]
            - 8.236915e-10 * h[75] / H_AVG[75]
            - 7.127445e-10 * h[87] / H_AVG[87]
            + 7.103319e-10 * h[93] / H_AVG[93]
            - 6.389433e-10 * h[40] / H_AVG[40]
            - 5.93905e-10 * h[20] / H_AVG[20]
            + 5.305223e-10 * h[119] / H_AVG[119]
            - 4.852652e-10 * h[62] / H_AVG[62]
            - 4.362113e-10 * h[95] / H_AVG[95]
            - 4.042518e-10 * h[96] / H_AVG[96]
            + 2.384471e-10 * h[116] / H_AVG[116]
            + 2.27655e-10 * h[56] / H_AVG[56]
            + 1.583306e-10 * h[85] / H_AVG[85]
            + 9.420847e-11 * h[109] / H_AVG[109]
            - 8.205601e-11 * h[98] / H_AVG[98]
            - 6.546234e-11 * h[88] / H_AVG[88]
            - 4.416051e-11 * h[46] / H_AVG[46]
            - 4.168103e-11 * h[47] / H_AVG[47]
            - 2.281362e-11 * h[64] / H_AVG[64]
            + 1.000939e-11 * h[127] / H_AVG[127]
            + 8.850373e-13 * h[41] / H_AVG[41]
        ),
        -1.495433 + T[9] * (   # class Tbl
            + 0.2585224 * h[16] / H_AVG[16]
            + 0.1377416 * h[24] / H_AVG[24]
            + 0.1178109 * h[18] / H_AVG[18]
            - 0.0954172 * h[104] / H_AVG[104]
            + 0.07775796 * h[97] / H_AVG[97]
            - 0.07418843 * h[115] / H_AVG[115]
            + 0.06620344 * h[52] / H_AVG[52]
            - 0.06513876 * h[68] / H_AVG[68]
            + 0.05582657 * h[78] / H_AVG[78]
            + 0.01749972 * h[120] / H_AVG[120]
            - 0.01255535 * h[124] / H_AVG[124]
            - 0.007661666 * h[83] / H_AVG[83]
            - 0.00752478 * h[81] / H_AVG[81]
            + 0.002715775 * h[25] / H_AVG[25]
            + 0.001702426 * h[123] / H_AVG[123]
            + 0.001227326 * h[70] / H_AVG[70]
            - 0.0003493776 * h[0] / H_AVG[0]
            + 7.823713e-05 * h[90] / H_AVG[90]
            + 5.838528e-05 * h[21] / H_AVG[21]
            - 1.718378e-05 * h[99] / H_AVG[99]
            + 8.205686e-07 * h[84] / H_AVG[84]
            + 1.935394e-07 * h[55] / H_AVG[55]
            + 1.298424e-07 * h[79] / H_AVG[79]
            - 1.240793e-07 * h[105] / H_AVG[105]
            - 1.006726e-07 * h[106] / H_AVG[106]
            + 8.186281e-08 * h[113] / H_AVG[113]
            - 7.770103e-08 * h[34] / H_AVG[34]
            + 6.975334e-08 * h[73] / H_AVG[73]
            + 6.475483e-08 * h[10] / H_AVG[10]
            + 6.178038e-08 * h[11] / H_AVG[11]
            - 5.254951e-08 * h[6] / H_AVG[6]
            + 5.016943e-08 * h[86] / H_AVG[86]
            - 4.773627e-08 * h[32] / H_AVG[32]
            + 4.38711e-08 * h[17] / H_AVG[17]
            + 3.961858e-08 * h[38] / H_AVG[38]
            - 3.255227e-08 * h[42] / H_AVG[42]
            - 2.902926e-08 * h[54] / H_AVG[54]
            - 2.756784e-08 * h[29] / H_AVG[29]
            + 2.500314e-08 * h[118] / H_AVG[118]
            + 1.949828e-08 * h[48] / H_AVG[48]
            - 1.862096e-08 * h[45] / H_AVG[45]
            - 1.797158e-08 * h[39] / H_AVG[39]
            + 1.658728e-08 * h[27] / H_AVG[27]
            + 1.651006e-08 * h[63] / H_AVG[63]
            + 1.634706e-08 * h[30] / H_AVG[30]
            + 1.598483e-08 * h[43] / H_AVG[43]
            + 1.487661e-08 * h[121] / H_AVG[121]
            - 1.461267e-08 * h[80] / H_AVG[80]
            - 1.33891e-08 * h[126] / H_AVG[126]
            + 1.2912e-08 * h[60] / H_AVG[60]
            - 1.27086e-08 * h[61] / H_AVG[61]
            - 1.207486e-08 * h[102] / H_AVG[102]
            - 1.203507e-08 * h[107] / H_AVG[107]
            - 1.187371e-08 * h[7] / H_AVG[7]
            + 1.068743e-08 * h[92] / H_AVG[92]
            + 1.045363e-08 * h[35] / H_AVG[35]
            + 1.016348e-08 * h[15] / H_AVG[15]
            - 9.82826e-09 * h[59] / H_AVG[59]
            - 9.482852e-09 * h[101] / H_AVG[101]
            - 9.042529e-09 * h[28] / H_AVG[28]
            + 8.78794e-09 * h[103] / H_AVG[103]
            + 8.280597e-09 * h[69] / H_AVG[69]
            + 8.037838e-09 * h[67] / H_AVG[67]
            + 7.76559e-09 * h[1] / H_AVG[1]
            + 7.706333e-09 * h[100] / H_AVG[100]
            - 7.080975e-09 * h[110] / H_AVG[110]
            - 6.865613e-09 * h[3] / H_AVG[3]
            + 6.587623e-09 * h[37] / H_AVG[37]
            - 6.44137e-09 * h[50] / H_AVG[50]
            + 6.062448e-09 * h[36] / H_AVG[36]
            + 6.061026e-09 * h[44] / H_AVG[44]
            - 5.9637e-09 * h[12] / H_AVG[12]
            - 5.666401e-09 * h[114] / H_AVG[114]
            - 5.626824e-09 * h[2] / H_AVG[2]
            + 5.421792e-09 * h[58] / H_AVG[58]
            - 5.405311e-09 * h[112] / H_AVG[112]
            + 5.284844e-09 * h[76] / H_AVG[76]
            - 5.079986e-09 * h[13] / H_AVG[13]
            - 4.943335e-09 * h[19] / H_AVG[19]
            - 4.843479e-09 * h[111] / H_AVG[111]
            + 4.62049e-09 * h[74] / H_AVG[74]
            + 4.375721e-09 * h[108] / H_AVG[108]
            + 4.12005e-09 * h[5] / H_AVG[5]
            - 3.944697e-09 * h[8] / H_AVG[8]
            - 3.787019e-09 * h[91] / H_AVG[91]
            - 3.526775e-09 * h[4] / H_AVG[4]
            - 3.377748e-09 * h[117] / H_AVG[117]
            + 2.849909e-09 * h[53] / H_AVG[53]
            - 2.819088e-09 * h[66] / H_AVG[66]
            - 2.559394e-09 * h[33] / H_AVG[33]
            - 2.500363e-09 * h[65] / H_AVG[65]
            + 2.311465e-09 * h[82] / H_AVG[82]
            - 2.162013e-09 * h[31] / H_AVG[31]
            - 1.988477e-09 * h[71] / H_AVG[71]
            + 1.848404e-09 * h[94] / H_AVG[94]
            + 1.785372e-09 * h[9] / H_AVG[9]
            + 1.576601e-09 * h[26] / H_AVG[26]
            - 1.407575e-09 * h[14] / H_AVG[14]
            + 1.399421e-09 * h[57] / H_AVG[57]
            + 1.315744e-09 * h[49] / H_AVG[49]
            + 1.2856e-09 * h[77] / H_AVG[77]
            - 1.283618e-09 * h[22] / H_AVG[22]
            + 1.195441e-09 * h[122] / H_AVG[122]
            + 9.620147e-10 * h[72] / H_AVG[72]
            + 9.267907e-10 * h[51] / H_AVG[51]
            - 9.105057e-10 * h[23] / H_AVG[23]
            - 6.232832e-10 * h[89] / H_AVG[89]
            + 6.113542e-10 * h[125] / H_AVG[125]
            - 5.740501e-10 * h[75] / H_AVG[75]
            + 4.962559e-10 * h[93] / H_AVG[93]
            - 4.952269e-10 * h[87] / H_AVG[87]
            - 4.460248e-10 * h[40] / H_AVG[40]
            - 4.147722e-10 * h[20] / H_AVG[20]
            + 3.716703e-10 * h[119] / H_AVG[119]
            - 3.379995e-10 * h[62] / H_AVG[62]
            - 3.044893e-10 * h[95] / H_AVG[95]
            - 2.815594e-10 * h[96] / H_AVG[96]
            + 1.662066e-10 * h[116] / H_AVG[116]
            + 1.596137e-10 * h[56] / H_AVG[56]
            + 1.113469e-10 * h[85] / H_AVG[85]
            + 6.554428e-11 * h[109] / H_AVG[109]
            - 5.708538e-11 * h[98] / H_AVG[98]
            - 4.541754e-11 * h[88] / H_AVG[88]
            - 3.105813e-11 * h[46] / H_AVG[46]
            - 2.90344e-11 * h[47] / H_AVG[47]
            - 1.520677e-11 * h[64] / H_AVG[64]
            + 7.264161e-12 * h[127] / H_AVG[127]
            + 6.467669e-13 * h[41] / H_AVG[41]
        ),
    ]]


def classify(pt, eta, phi, energy, jet_pt, jet_eta, jet_energy, charge, ptype, d0, d0err, dz, dzerr):
    s = logits(class_token(quantities(pt, eta, phi, energy, jet_pt, jet_eta, jet_energy, charge, ptype, d0, d0err, dz, dzerr)))
    m = max(s)
    e = [math.exp(x - m) for x in s]
    p = [x / sum(e) for x in e]
    return CLASSES[s.index(m)], s, p


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    energy = [447.116, 239.595, 123.342, 44.45, 22.892, 11.973, 6.796, 4.414]
    jet_pt = 830.0
    jet_eta = 0.4
    jet_energy = 900.578
    charge = [1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0]
    ptype = [1.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0, 4.0]
    d0 = [0.01, -0.3, 0.0, 0.02, 0.0, 0.2, 0.0, -0.01]
    d0err = [0.01, 0.02, 0.0, 0.03, 0.0, 0.02, 0.0, 0.01]
    dz = [0.02, 0.1, 0.0, -0.02, 0.0, 0.3, 0.0, 0.01]
    dzerr = [0.01, 0.02, 0.0, 0.03, 0.0, 0.02, 0.0, 0.01]
    c, s, p = classify(pt, eta, phi, energy, jet_pt, jet_eta, jet_energy, charge, ptype, d0, d0err, dz, dzerr)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
