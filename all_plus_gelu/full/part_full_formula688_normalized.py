"""Particle Transformer (ParT) jet tagger, JetClass, input features 'full': smaller version of the 869 (tuned on the network's predictions; step 4), with normalized weights (how much each one matters), as if-statements.

Input:  the particles of a jet (up to 128), hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
        energy: each particle's energy [GeV]; jet_pt, jet_eta, jet_energy: the jet's pT [GeV], pseudorapidity, energy [GeV].
Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the training jets, so share_k is the fraction of the neuron's
             average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron 16:  12.0%
  neuron 18:   9.5%
  neuron 97:   9.2%
  neuron 24:   8.9%
  neuron 120:   8.7%
  neuron 104:   8.3%
  neuron 115:   7.6%
  neuron 83:   7.4%
  neuron 52:   7.2%
  neuron 68:   7.0%
  neuron 124:   5.6%
  neuron 78:   3.6%
  neuron 81:   3.1%
  neuron 70:   1.6%
  neuron 25:   0.2%
  neuron 123:   0.1%
  neuron  0:   0.0%
  neuron 21:   0.0%
  neuron 99:   0.0%
  neuron 90:   0.0%
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
  neuron 22:   0.0%
  neuron 77:   0.0%
  neuron 122:   0.0%
  neuron 72:   0.0%
  neuron 51:   0.0%
  neuron 23:   0.0%
  neuron 89:   0.0%
  neuron 125:   0.0%
  neuron 75:   0.0%
  neuron 93:   0.0%
  neuron 87:   0.0%
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

Whole test file (2,000,000 jets): accuracy 75.25% (the network: 86.03%); same class as the network for 80.09% of jets.

Quantities:
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b05                 e3/e2² with β = 0.5
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3_b2                  e4·e2/e3² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M2_b2                  ₁e₃/e2 with β = 2
  Q.M3                     generalized ECF ratio M3
  Q.M3_b2                  ₁e₄/₁e₃ with β = 2
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N2_b05                 ₂e₃/(e2)² with β = 0.5
  Q.N2_b2                  ₂e₃/(e2)² with β = 2
  Q.N3_b05                 ₂e₄/(₁e₃)² with β = 0.5
  Q.N3_b2                  ₂e₄/(₁e₃)² with β = 2
  Q.ak02_1_n_disp3         hardest anti-kT 0.2 subjet: number of its tracks with d0/σ > 3
  Q.ak02_2_n_lep           2nd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_2_z               pT share of the 2nd anti-kT 0.2 subjet (0 if none)
  Q.ak02_3_n_lep           3rd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_3_z               pT share of the 3rd anti-kT 0.2 subjet (0 if none)
  Q.ak02_dr12              distance between the pT-weighted centres of the hardest and 2nd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr13              distance between the pT-weighted centres of the hardest and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr23              distance between the pT-weighted centres of the 2nd and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_min12_n_disp3     the smaller of the two hardest anti-kT 0.2 subjets: number of its tracks with d0/σ > 3
  Q.ak02_n                 number of anti-kT R = 0.2 subjets with pT > 10 GeV
  Q.dc_1_n_disp3           hardest prong: number of its tracks with d0/σ > 3
  Q.dc_1_n_lep             hardest prong: number of its electrons and muons
  Q.dc_1_z                 pT share of the hardest prong (0 if none)
  Q.dc_2_jp                2nd prong: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.dc_2_n_lep             2nd prong: number of its electrons and muons
  Q.dc_3_charge            3rd prong: Σ q √pT / √(Σ pT) of its particles
  Q.dc_3_z                 pT share of the 3rd prong (0 if none)
  Q.dc_4_z                 pT share of the 4th prong (0 if none)
  Q.dc_n                   number of prongs: reverse the C/A tree; ΔR ≤ 0.1 is a prong, a branch with < 10 % of the jet pT is dropped, else both branches are declustered
  Q.dc_split1_kt           kT = min(pT)·ΔR [GeV] of the hardest hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_split2_dr           ΔR of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_split2_mass         mass of the splitting node [GeV] of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dr12                   ΔR between particles 1 and 2
  Q.dr_0                   ΔR from the jet axis of particle 0 (ParT input; empty slot: 0)
  Q.dr_max_012             largest distance among the 3 hardest
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.e3                     energy correlation e3 (β=1, 32 hardest)
  Q.e3_b05                 energy correlation e3 with β = 0.5
  Q.e3_b2                  energy correlation e3 with β = 2
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 16 hardest)
  Q.e4_b2                  energy correlation e4 with β = 2
  Q.ecf_g41                generalized energy correlation ₁e₄ (β=1; products of the smallest angles)
  Q.ecf_g42                generalized energy correlation ₂e₄ (β=1; products of the smallest angles)
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.ismuon_0               1 if a muon of particle 0 (ParT input; empty slot: 0)
  Q.jd_3d_4                the 4th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_5                the 5th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_6                the 6th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_n_d3_pt1            number of tracks with |d0/σ| > 3 and pT > 1 GeV
  Q.jd_sum_abs_sd0_top3    sum of the 3 largest |d0/σ|
  Q.jd_sum_abs_sd0_top5    sum of the 5 largest |d0/σ|
  Q.jet_abs_eta            absolute pseudorapidity of the jet axis
  Q.jet_charge_k03         jet charge with κ = 0.3
  Q.jet_charge_k05         jet charge with κ = 0.5
  Q.kt2_1_n_lep            hardest kT subjet: number of its electrons and muons
  Q.kt2_2_n_lep            2nd kT subjet: number of its electrons and muons
  Q.kt2_2_z_disp3          2nd kT subjet: pT share (of the jet) of its tracks with d0/σ > 3
  Q.kt2_dr12               distance between the pT-weighted centres of the hardest and 2nd kT subjet (0 if missing)
  Q.ktd_ln_d23             ln of the exclusive-kT merging scale from 3 to 2 subjets, min(pT²)ΔR², over (Σ pT)² (0 if fewer particles)
  Q.ktd_ln_d34             ln of the exclusive-kT merging scale from 4 to 3 subjets, min(pT²)ΔR², over (Σ pT)² (0 if fewer particles)
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lep_dr                 ΔR of the hardest lepton from the jet axis (0 if none)
  Q.lep_iso                Σ pT of the other particles within ΔR < 0.2 of the hardest lepton / its pT (0 if none)
  Q.lep_ptrel              pT × ΔR from the jet axis of the hardest lepton [GeV] (0 if none)
  Q.lep_z                  pT share of the hardest electron or muon (0 if none)
  Q.lepsj_2_dr             distance of the hardest lepton from its nearest subjet axis; 2 subjets (0 if no lepton)
  Q.lepsj_2_maxsd0         largest |d0/σ| in the lepton’s nearest subjet (without the lepton); 2 subjets (0 if no lepton)
  Q.lepsj_2_n_d3           number of tracks with |d0/σ| > 3 in the lepton’s nearest subjet (without the lepton); 2 subjets (0 if no lepton)
  Q.lepsj_3_dr             distance of the hardest lepton from its nearest subjet axis; 3 subjets (0 if no lepton)
  Q.lepsj_3_mass           mass (E) of the hardest lepton with its nearest subjet [GeV]; 3 subjets (0 if no lepton)
  Q.lepsj_3_maxsd0         largest |d0/σ| in the lepton’s nearest subjet (without the lepton); 3 subjets (0 if no lepton)
  Q.lepsj_3_n_d3           number of tracks with |d0/σ| > 3 in the lepton’s nearest subjet (without the lepton); 3 subjets (0 if no lepton)
  Q.lne_0                  ln E [GeV] of particle 0 (ParT input; empty slot: ln 1e-8)
  Q.lne_17                 ln E [GeV] of particle 17 (ParT input; empty slot: ln 1e-8)
  Q.lne_21                 ln E [GeV] of particle 21 (ParT input; empty slot: ln 1e-8)
  Q.lne_5                  ln E [GeV] of particle 5 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_1               ln(E / E of the jet) of particle 1 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_78              ln(E / E of the jet) of particle 78 (ParT input; empty slot: ln 1e-8)
  Q.lund2_lndelta          ln Δ of the 2. primary C/A splitting (ln 1e-8 if none)
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
  Q.mres_sd_rg_b1z01       soft-drop R_g, β = 1, z_cut = 0.1
  Q.n_charged_had          number of charged hadrons
  Q.n_charged_pt_above_1   number of charged particles with pT > 1 GeV
  Q.n_charged_pt_above_10  number of charged particles with pT > 10 GeV
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_dr_0p4_up            number of particles with ΔR ≥ 0.4
  Q.n_electron             number of electrons
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_lepton               number of electrons and muons
  Q.n_lund                 number of primary C/A splittings
  Q.n_lund_kt_above_1      number of primary splittings with kT > 1 GeV
  Q.n_lund_kt_above_5      number of primary splittings with kT > 5 GeV
  Q.n_muon                 number of muons
  Q.n_neutral              number of neutral particles
  Q.n_neutral_had          number of neutral hadrons
  Q.n_pairs_kt_above_1     number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 1 GeV
  Q.n_pairs_kt_above_3     number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 3 GeV
  Q.n_particles            number of real particles (pT > 0)
  Q.n_photon               number of photons
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.n_s3d_above_10         number of charged particles with 3d significance > 10
  Q.n_s3d_above_3          number of charged particles with 3d significance > 3
  Q.n_sd0_above_2          number of charged particles with d0 significance > 2
  Q.n_sd0_above_3          number of charged particles with d0 significance > 3
  Q.n_sd0_above_5          number of charged particles with d0 significance > 5
  Q.n_sdz_above_2          number of charged particles with dz significance > 2
  Q.n_sdz_above_5          number of charged particles with dz significance > 5
  Q.nca_sj4_pair_mass_2nd  second largest mass of two of the 4 subjets [GeV]
  Q.nca_sj4_pairmax_over_mass largest mass of two of the 4 subjets over the jet mass
  Q.pair_max_lnkt          largest ln kT among all pairs
  Q.pair_max_lnm2          largest ln m² among all pairs
  Q.pair_mean_lndelta      zᵢzⱼ-weighted mean of ln ΔRᵢⱼ over all pairs
  Q.pair_mean_lnkt         zᵢzⱼ-weighted mean of ln kT = ln(min(pTᵢ, pTⱼ)·ΔRᵢⱼ) over all pairs
  Q.pair_mean_lnm2         zᵢzⱼ-weighted mean of ln mᵢⱼ² (massless) over all pairs
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.pt1_over_pt0           pT1 / pT0
  Q.pz_lnd0                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -9 < ln ΔRᵢⱼ ≤ -3
  Q.pz_lnd2                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -2 < ln ΔRᵢⱼ ≤ -1
  Q.pz_lnd3                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -1 < ln ΔRᵢⱼ ≤ 9
  Q.pz_lnkt0               Σ zᵢzⱼ over the pairs of the 40 hardest particles with -9 < ln kT ≤ 0
  Q.pz_lnkt1               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 0 < ln kT ≤ 1
  Q.pz_lnkt3               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 2 < ln kT ≤ 3
  Q.pz_lnkt4               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 3 < ln kT ≤ 9
  Q.sd_mass                soft-drop groomed mass, C/A on the 128 hardest, β=0, z_cut=0.1 [GeV]
  Q.sdb_0_n                number of tracks with d0/σ ≤ -3
  Q.sdb_2_n                number of tracks with -1 < d0/σ ≤ 1
  Q.sdb_2_z                pT share of the tracks with -1 < d0/σ ≤ 1
  Q.sdb_4_z                pT share of the tracks with 3 < d0/σ ≤ 10
  Q.sdb_5_n                number of tracks with d0/σ > 10
  Q.sdb_5_z                pT share of the tracks with d0/σ > 10
  Q.sip_3d_1               the 1. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sip_3d_2               the 2. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sip_3d_3               the 3. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.sj4_dr_min             smallest distance among the 4 subjet axes
  Q.sj4_pair_mass_max      largest mass of two of the 4 subjets [GeV]
  Q.sj4_zsoft              pT share of the softest of 4 subjets
  Q.sjf_2_1_max3d          subjet 1 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_1_n_d3           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_2_1_n_d5           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_2_1_z_d3           subjet 1 of 2 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_2_2_max3d          subjet 2 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_2_n_d3           subjet 2 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_2_n2disp           number of the 2 subjets with at least 2 tracks with |d0/σ| > 3
  Q.sjf_3_1_max3d          subjet 1 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_1_n_d3           subjet 1 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_3_3_n_d3           subjet 3 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_4_1_max3d          subjet 1 of 4 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_4_1_n_d3           subjet 1 of 4 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_4_1_z_d3           subjet 1 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_2_z_d3           subjet 2 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_n2disp           number of the 4 subjets with at least 2 tracks with |d0/σ| > 3
  Q.sjq_2_1_nch            number of charged particles in subjet 1 of 2
  Q.sjq_2_2_k1             Σ q pT^κ / (Σ pT)^κ of subjet 2 of 2, κ = 1
  Q.sjq_2_2_nch            number of charged particles in subjet 2 of 2
  Q.sjq_2_prod_k03         product of the charges (κ = 0.3) of the two hardest of 2 subjets
  Q.sjq_2_prod_k05         product of the charges (κ = 0.5) of the two hardest of 2 subjets
  Q.sjq_2_prod_k1          product of the charges (κ = 1) of the two hardest of 2 subjets
  Q.sjq_2_sumabs_k03       |sum| of the charges (κ = 0.3) of the two hardest of 2 subjets
  Q.sjq_2_sumabs_k1        |sum| of the charges (κ = 1) of the two hardest of 2 subjets
  Q.sjq_3_1_k05            Σ q pT^κ / (Σ pT)^κ of subjet 1 of 3, κ = 0.5
  Q.sjq_3_2_k1             Σ q pT^κ / (Σ pT)^κ of subjet 2 of 3, κ = 1
  Q.sjq_3_2_nch            number of charged particles in subjet 2 of 3
  Q.sjq_3_3_k1             Σ q pT^κ / (Σ pT)^κ of subjet 3 of 3, κ = 1
  Q.sjq_3_3_nch            number of charged particles in subjet 3 of 3
  Q.sjq_3_sumabs_k1        |sum| of the charges (κ = 1) of the two hardest of 3 subjets
  Q.sum_e                  total energy of the particles [GeV]
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sv_1_dr                hardest displaced-track cluster: ΔR of its centre from the jet axis (0 if none)
  Q.sv_1_n                 hardest displaced-track cluster: number of tracks (0 if none)
  Q.sv_1_sd0_sum           hardest displaced-track cluster: Σ d0/σ of its tracks (0 if none)
  Q.sv_1_z                 hardest displaced-track cluster: pT share of the jet (0 if none)
  Q.sv_2_n                 2nd displaced-track cluster: number of tracks (0 if none)
  Q.sv_2_sd0_sum           2nd displaced-track cluster: Σ d0/σ of its tracks (0 if none)
  Q.sv_2_z                 2nd displaced-track cluster: pT share of the jet (0 if none)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau32_b2               N-subjettiness τ3/τ2 with β = 2
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.tau43                  N-subjettiness τ4/τ3
  Q.tau5                   N-subjettiness τ5 (β=1)
  Q.tau54                  N-subjettiness τ5/τ4
  Q.z_charged_had          pT share of charged hadrons
  Q.z_displaced3           pT share of the charged particles with |d0|/σ > 3
  Q.z_displaced5           pT share of the charged particles with |d0|/σ > 5
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.z_dr_0p4_up            pT share of the particles with ΔR ≥ 0.4
  Q.z_electron             pT share of electrons
  Q.z_muon                 pT share of muons
  Q.z_neutral              pT share of neutral particles
  Q.z_neutral_had          pT share of neutral hadrons
  Q.z_photon               pT share of photons
  Q.z_top30_slots          pT share of the 30 hardest particles
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
        C2_b05=ecfb('e3', 0.5) / max(ecfb('e2', 0.5) ** 2, 1e-30),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3_b2=ecfb('e4', 2) * ecfb('e2', 2) / max(ecfb('e3', 2) ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M2_b2=ecfb('g31', 2) / max(ecfb('e2', 2), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        M3_b2=ecfb('g41', 2) / max(ecfb('g31', 2), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N2_b05=ecfb('g32', 0.5) / max(ecfb('e2', 0.5) ** 2, 1e-30),
        N2_b2=ecfb('g32', 2) / max(ecfb('e2', 2) ** 2, 1e-30),
        N3_b05=ecfb('g42', 0.5) / max(ecfb('g31', 0.5) ** 2, 1e-30),
        N3_b2=ecfb('g42', 2) / max(ecfb('g31', 2) ** 2, 1e-30),
        ak02_1_n_disp3=pq('ak02_1_n_disp3'),
        ak02_2_n_lep=pq('ak02_2_n_lep'),
        ak02_2_z=pq('ak02_2_z'),
        ak02_3_n_lep=pq('ak02_3_n_lep'),
        ak02_3_z=pq('ak02_3_z'),
        ak02_dr12=pq('ak02_dr12'),
        ak02_dr13=pq('ak02_dr13'),
        ak02_dr23=pq('ak02_dr23'),
        ak02_min12_n_disp3=pq('ak02_min12_n_disp3'),
        ak02_n=pq('ak02_n'),
        dc_1_n_disp3=pq('dc_1_n_disp3'),
        dc_1_n_lep=pq('dc_1_n_lep'),
        dc_1_z=pq('dc_1_z'),
        dc_2_jp=pq('dc_2_jp'),
        dc_2_n_lep=pq('dc_2_n_lep'),
        dc_3_charge=pq('dc_3_charge'),
        dc_3_z=pq('dc_3_z'),
        dc_4_z=pq('dc_4_z'),
        dc_n=pq('dc_n'),
        dc_split1_kt=pq('dc_split1_kt'),
        dc_split2_dr=pq('dc_split2_dr'),
        dc_split2_mass=pq('dc_split2_mass'),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        dr_0=pfeat(0, 'dr'),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        e3=ecf('e3'),
        e3_b05=ecfb('e3', 0.5),
        e3_b2=ecfb('e3', 2),
        e4=ecf('e4'),
        e4_b2=ecfb('e4', 2),
        ecf_g41=ecfb('g41', 1),
        ecf_g42=ecfb('g42', 1),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        ismuon_0=pfeat(0, 'ismuon'),
        jd_3d_4=pq('jd_3d_4'),
        jd_3d_5=pq('jd_3d_5'),
        jd_3d_6=pq('jd_3d_6'),
        jd_n_d3_pt1=pq('jd_n_d3_pt1'),
        jd_sum_abs_sd0_top3=pq('jd_sum_abs_sd0_top3'),
        jd_sum_abs_sd0_top5=pq('jd_sum_abs_sd0_top5'),
        jet_abs_eta=abs(jet_eta),
        jet_charge_k03=sum(charge[i] * z[i] ** 0.3 for i in real),
        jet_charge_k05=sum(charge[i] * z[i] ** 0.5 for i in real),
        kt2_1_n_lep=pq('kt2_1_n_lep'),
        kt2_2_n_lep=pq('kt2_2_n_lep'),
        kt2_2_z_disp3=pq('kt2_2_z_disp3'),
        kt2_dr12=pq('kt2_dr12'),
        ktd_ln_d23=pq('ktd_ln_d23'),
        ktd_ln_d34=pq('ktd_ln_d34'),
        lam2=lam2,
        lep_dr=lepton('dr'),
        lep_iso=lepton('iso'),
        lep_ptrel=lepton('ptrel'),
        lep_z=lepton('z'),
        lepsj_2_dr=pq('lepsj_2_dr'),
        lepsj_2_maxsd0=pq('lepsj_2_maxsd0'),
        lepsj_2_n_d3=pq('lepsj_2_n_d3'),
        lepsj_3_dr=pq('lepsj_3_dr'),
        lepsj_3_mass=pq('lepsj_3_mass'),
        lepsj_3_maxsd0=pq('lepsj_3_maxsd0'),
        lepsj_3_n_d3=pq('lepsj_3_n_d3'),
        lne_0=pfeat(0, 'lne'),
        lne_17=pfeat(17, 'lne'),
        lne_21=pfeat(21, 'lne'),
        lne_5=pfeat(5, 'lne'),
        lnerel_1=pfeat(1, 'lnerel'),
        lnerel_78=pfeat(78, 'lnerel'),
        lund2_lndelta=lund(2, 'lndelta'),
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
        mres_sd_rg_b1z01=pq('mres_sd_rg_b1z01'),
        n_charged_had=sum(1 for i in real if ptype[i] == 1),
        n_charged_pt_above_1=sum(1 for i in real if charge[i] != 0 and pt[i] > 1),
        n_charged_pt_above_10=sum(1 for i in real if charge[i] != 0 and pt[i] > 10),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        n_electron=sum(1 for i in real if ptype[i] == 4),
        n_for_90pct=ncum(0.9),
        n_lepton=sum(1 for i in real if ptype[i] in (4, 5)),
        n_lund=lund(0, 'n'),
        n_lund_kt_above_1=lund(0, 'n1'),
        n_lund_kt_above_5=lund(0, 'n5'),
        n_muon=sum(1 for i in real if ptype[i] == 5),
        n_neutral=sum(1 for i in real if charge[i] == 0),
        n_neutral_had=sum(1 for i in real if ptype[i] == 2),
        n_pairs_kt_above_1=paircount(1),
        n_pairs_kt_above_3=paircount(3),
        n_particles=len(real),
        n_photon=sum(1 for i in real if ptype[i] == 3),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        n_s3d_above_10=nsig('3d', 10),
        n_s3d_above_3=nsig('3d', 3),
        n_sd0_above_2=nsig('d0', 2),
        n_sd0_above_3=nsig('d0', 3),
        n_sd0_above_5=nsig('d0', 5),
        n_sdz_above_2=nsig('dz', 2),
        n_sdz_above_5=nsig('dz', 5),
        nca_sj4_pair_mass_2nd=pq('nca_sj4_pair_mass_2nd'),
        nca_sj4_pairmax_over_mass=pq('nca_sj4_pairmax_over_mass'),
        pair_max_lnkt=pairmax('lnkt'),
        pair_max_lnm2=pairmax('lnm2'),
        pair_mean_lndelta=pairsum('lndelta'),
        pair_mean_lnkt=pairsum('lnkt'),
        pair_mean_lnm2=pairsum('lnm2'),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        pz_lnd0=pq('pz_lnd0'),
        pz_lnd2=pq('pz_lnd2'),
        pz_lnd3=pq('pz_lnd3'),
        pz_lnkt0=pq('pz_lnkt0'),
        pz_lnkt1=pq('pz_lnkt1'),
        pz_lnkt3=pq('pz_lnkt3'),
        pz_lnkt4=pq('pz_lnkt4'),
        sd_mass=softdrop("mass"),
        sdb_0_n=pq('sdb_0_n'),
        sdb_2_n=pq('sdb_2_n'),
        sdb_2_z=pq('sdb_2_z'),
        sdb_4_z=pq('sdb_4_z'),
        sdb_5_n=pq('sdb_5_n'),
        sdb_5_z=pq('sdb_5_z'),
        sip_3d_1=sip('3d', 1),
        sip_3d_2=sip('3d', 2),
        sip_3d_3=sip('3d', 3),
        sj2_dr=subjets(2)["dr"][0],
        sj2_mass2=subjets(2)["mass"][1],
        sj2_zsoft=subjets(2)["z"][1],
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr_max=max(subjets(3)["dr"]),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_z1=subjets(3)["z"][0],
        sj4_dr_min=min(subjets(4)["dr"]),
        sj4_pair_mass_max=max(subjets(4)["mpair"]),
        sj4_zsoft=subjets(4)["z"][3],
        sjf_2_1_max3d=pq('sjf_2_1_max3d'),
        sjf_2_1_n_d3=pq('sjf_2_1_n_d3'),
        sjf_2_1_n_d5=pq('sjf_2_1_n_d5'),
        sjf_2_1_z_d3=pq('sjf_2_1_z_d3'),
        sjf_2_2_max3d=pq('sjf_2_2_max3d'),
        sjf_2_2_n_d3=pq('sjf_2_2_n_d3'),
        sjf_2_n2disp=pq('sjf_2_n2disp'),
        sjf_3_1_max3d=pq('sjf_3_1_max3d'),
        sjf_3_1_n_d3=pq('sjf_3_1_n_d3'),
        sjf_3_3_n_d3=pq('sjf_3_3_n_d3'),
        sjf_4_1_max3d=pq('sjf_4_1_max3d'),
        sjf_4_1_n_d3=pq('sjf_4_1_n_d3'),
        sjf_4_1_z_d3=pq('sjf_4_1_z_d3'),
        sjf_4_2_z_d3=pq('sjf_4_2_z_d3'),
        sjf_4_n2disp=pq('sjf_4_n2disp'),
        sjq_2_1_nch=pq('sjq_2_1_nch'),
        sjq_2_2_k1=pq('sjq_2_2_k1'),
        sjq_2_2_nch=pq('sjq_2_2_nch'),
        sjq_2_prod_k03=pq('sjq_2_prod_k03'),
        sjq_2_prod_k05=pq('sjq_2_prod_k05'),
        sjq_2_prod_k1=pq('sjq_2_prod_k1'),
        sjq_2_sumabs_k03=pq('sjq_2_sumabs_k03'),
        sjq_2_sumabs_k1=pq('sjq_2_sumabs_k1'),
        sjq_3_1_k05=pq('sjq_3_1_k05'),
        sjq_3_2_k1=pq('sjq_3_2_k1'),
        sjq_3_2_nch=pq('sjq_3_2_nch'),
        sjq_3_3_k1=pq('sjq_3_3_k1'),
        sjq_3_3_nch=pq('sjq_3_3_nch'),
        sjq_3_sumabs_k1=pq('sjq_3_sumabs_k1'),
        sum_e=sum(energy[i] for i in real),
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sv_1_dr=pq('sv_1_dr'),
        sv_1_n=pq('sv_1_n'),
        sv_1_sd0_sum=pq('sv_1_sd0_sum'),
        sv_1_z=pq('sv_1_z'),
        sv_2_n=pq('sv_2_n'),
        sv_2_sd0_sum=pq('sv_2_sd0_sum'),
        sv_2_z=pq('sv_2_z'),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau32_b2=tau_n(3, 2) / max(tau_n(2, 2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        tau5=tau_n(5),
        tau54=tau_n(5) / max(tau_n(4), 1e-12),
        z_charged_had=sum(z[i] for i in real if ptype[i] == 1),
        z_displaced3=displaced(3, 'z'),
        z_displaced5=displaced(5, 'z'),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        z_dr_0p4_up=sum(z[i] for i in real if 0.4 <= dr[i] < 10),
        z_electron=sum(z[i] for i in real if ptype[i] == 4),
        z_muon=sum(z[i] for i in real if ptype[i] == 5),
        z_neutral=sum(z[i] for i in real if charge[i] == 0),
        z_neutral_had=sum(z[i] for i in real if ptype[i] == 2),
        z_photon=sum(z[i] for i in real if ptype[i] == 3),
        z_top30_slots=sum(pt[:30]) / tot,
    )


def neuron_0(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.00521
    )
    return z


def neuron_1(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.32e-06
    )
    return z


def neuron_2(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.45e-06
    )
    return z


def neuron_3(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.39e-06
    )
    return z


def neuron_4(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.79e-06
    )
    return z


def neuron_5(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.16e-06
    )
    return z


def neuron_6(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.77e-05
    )
    return z


def neuron_7(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.43e-06
    )
    return z


def neuron_8(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.92e-06
    )
    return z


def neuron_9(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.44e-06
    )
    return z


def neuron_10(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.57e-05
    )
    return z


def neuron_11(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.89e-05
    )
    return z


def neuron_12(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.1e-06
    )
    return z


def neuron_13(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.42e-06
    )
    return z


def neuron_14(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.84e-06
    )
    return z


def neuron_15(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.45e-06
    )
    return z


def neuron_16(Q):
    # scale S = 15.3; each line: share * term / its average size
    z = 15.30374 * (-0.06449403
        - 0.1064798 * (146.0039 * 0.5 * ((263.0 - Q.jd_sum_abs_sd0_top3) / 146.0039) * (1 + math.erf(((263.0 - Q.jd_sum_abs_sd0_top3) / 146.0039) / math.sqrt(2)))) / 120.7066   # -10.6%  jd_sum_abs_sd0_top3 < 263
        + 0.1037701 * (12.22004 * 0.5 * ((27.4 - Q.lep_ptrel) / 12.22004) * (1 + math.erf(((27.4 - Q.lep_ptrel) / 12.22004) / math.sqrt(2)))) / 21.43145   # +10.4%  lep_ptrel < 27.4
        + 0.09025953 * (148.6964 * 0.5 * ((289.0 - Q.jd_sum_abs_sd0_top5) / 148.6964) * (1 + math.erf(((289.0 - Q.jd_sum_abs_sd0_top5) / 148.6964) / math.sqrt(2)))) / 132.8181   # +9.0%  jd_sum_abs_sd0_top5 < 289
        - 0.08146742 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 6.361002   # -8.1%  n_s3d_above_3 < 10
        - 0.06582357 * (0.05422308 * 0.5 * ((Q.z_neutral - 0.139) / 0.05422308) * (1 + math.erf(((Q.z_neutral - 0.139) / 0.05422308) / math.sqrt(2)))) / 0.2664939   # -6.6%  z_neutral > 0.139
        - 0.06567021 * (0.08093679 * 0.5 * ((0.281 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.281 - Q.z_displaced5) / 0.08093679) / math.sqrt(2)))) / 0.1994047   # -6.6%  z_displaced5 < 0.281
        + 0.06423544 * (0.09329774 * 0.5 * ((0.723 - Q.z_charged_had) / 0.09329774) * (1 + math.erf(((0.723 - Q.z_charged_had) / 0.09329774) / math.sqrt(2)))) / 0.21992   # +6.4%  z_charged_had < 0.723
        + 0.05250881 * (0.08093679 * 0.5 * ((0.28 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.28 - Q.z_displaced5) / 0.08093679) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.0 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.0 - Q.jd_3d_4) / 115.0698) / math.sqrt(2)))) / 29.11526   # +5.3%  z_displaced5 < 0.28 and jd_3d_4 < 172
        + 0.04306912 * Q.lep_ptrel / 7.315413   # +4.3%  lep_ptrel
        + 0.01748839 * (164.9991 * 0.5 * ((228.0 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((228.0 - Q.sip_3d_2) / 164.9991) / math.sqrt(2)))) / 135.1706   # +1.7%  sip_3d_2 < 228
        - 0.01658603 * (0.07843207 * 0.5 * ((Q.lep_z - 0.13) / 0.07843207) * (1 + math.erf(((Q.lep_z - 0.13) / 0.07843207) / math.sqrt(2)))) / 0.05603274   # -1.7%  lep_z > 0.13
        - 0.01596466 * (17.91682 * 0.5 * ((33.6 - Q.jd_3d_4) / 17.91682) * (1 + math.erf(((33.6 - Q.jd_3d_4) / 17.91682) / math.sqrt(2)))) / 22.62213   # -1.6%  jd_3d_4 < 33.6
        + 0.01573014 * (0.0439522 * 0.5 * ((0.0968 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.0968 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) / 0.11409   # +1.6%  sjq_2_prod_k1 < 0.0968
        + 0.01475494 * (112.8732 * 0.5 * ((207.0 - Q.sip_3d_1) / 112.8732) * (1 + math.erf(((207.0 - Q.sip_3d_1) / 112.8732) / math.sqrt(2)))) / 96.49823   # +1.5%  sip_3d_1 < 207
        + 0.01300585 * (3.705078 * 0.5 * ((5.79 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.79 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) / 2.745353   # +1.3%  max_abs_d0 < 5.79
        - 0.01230131 * (4.214028 * 0.5 * ((10.1 - Q.jd_3d_4) / 4.214028) * (1 + math.erf(((10.1 - Q.jd_3d_4) / 4.214028) / math.sqrt(2)))) / 5.020161   # -1.2%  jd_3d_4 < 10.1
        - 0.01178463 * (0.560057 * 0.5 * ((1.78 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.78 - Q.mass_displaced5) / 0.560057) / math.sqrt(2)))) / 0.8197678   # -1.2%  mass_displaced5 < 1.78
        + 0.01144694 * (6.5 * 0.5 * ((41.2 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((41.2 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2)))) / 7.68338   # +1.1%  n_pairs_kt_above_3 < 41.2
        - 0.009434906 * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.6533455   # -0.9%  kt2_2_n_lep < 1
        + 0.008862129 * (248.3135 * 0.5 * ((178.0 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((178.0 - Q.sip_3d_3) / 248.3135) / math.sqrt(2)))) / 109.374   # +0.9%  sip_3d_3 < 178
        + 0.007796549 * (0.03043217 * 0.5 * ((0.111 - Q.sv_1_z) / 0.03043217) * (1 + math.erf(((0.111 - Q.sv_1_z) / 0.03043217) / math.sqrt(2)))) / 0.07504174   # +0.8%  sv_1_z < 0.111
        + 0.00758785 * Q.n_photon / 16.03902   # +0.8%  n_photon
        + 0.00745579 * (0.0002686389 * 0.5 * ((0.00041 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.00041 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.001717815 * 0.5 * ((Q.M3 - 0.0252) / 0.001717815) * (1 + math.erf(((Q.M3 - 0.0252) / 0.001717815) / math.sqrt(2)))) / 4.194907e-06   # +0.7%  e3_b2 < 0.00041 and M3 > 0.0252
        - 0.007385368 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.151853 * 0.5 * ((0.139 - Q.z_electron) / 0.151853) * (1 + math.erf(((0.139 - Q.z_electron) / 0.151853) / math.sqrt(2)))) / 0.6314177   # -0.7%  n_s3d_above_3 < 10 and z_electron < 0.139
        - 0.007328746 * (0.0439522 * 0.5 * ((0.0995 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.0995 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) * (1.0 * 0.5 * ((4.99 - Q.sjf_4_1_n_d3) / 1.0) * (1 + math.erf(((4.99 - Q.sjf_4_1_n_d3) / 1.0) / math.sqrt(2)))) / 0.4415639   # -0.7%  sjq_2_prod_k1 < 0.0995 and sjf_4_1_n_d3 < 4.99
        + 0.007146186 * (3.043206 * 0.5 * ((56.2 - Q.mass_charged) / 3.043206) * (1 + math.erf(((56.2 - Q.mass_charged) / 3.043206) / math.sqrt(2)))) / 6.792757   # +0.7%  mass_charged < 56.2
        + 0.006569953 * (3.485807 * 0.5 * ((Q.mass_top40 - 118.0) / 3.485807) * (1 + math.erf(((Q.mass_top40 - 118.0) / 3.485807) / math.sqrt(2)))) / 10.05449   # +0.7%  mass_top40 > 118
        - 0.006360496 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.2798636 * 0.5 * ((1.71 - Q.sjq_2_sumabs_k03) / 0.2798636) * (1 + math.erf(((1.71 - Q.sjq_2_sumabs_k03) / 0.2798636) / math.sqrt(2)))) / 6.532845   # -0.6%  n_s3d_above_3 < 10 and sjq_2_sumabs_k03 < 1.71
        - 0.00631966 * (0.03445022 * 0.5 * ((0.0493 - Q.lepsj_3_dr) / 0.03445022) * (1 + math.erf(((0.0493 - Q.lepsj_3_dr) / 0.03445022) / math.sqrt(2)))) / 0.03663426   # -0.6%  lepsj_3_dr < 0.0493
        - 0.006221586 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2)))) / 9.427083   # -0.6%  n_dr_0p4_up < 15
        + 0.005877347 * (0.02907957 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.212) / 0.02907957) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.212) / 0.02907957) / math.sqrt(2)))) / 0.1716515   # +0.6%  sjq_2_sumabs_k1 > 0.212
        + 0.005357418 * (1.0 * 0.5 * ((9.98 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((9.98 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sv_1_n - 1.01) / 1.0) * (1 + math.erf(((Q.sv_1_n - 1.01) / 1.0) / math.sqrt(2)))) / 1.406322   # +0.5%  n_s3d_above_3 < 9.98 and sv_1_n > 1.01
        + 0.005136044 * (5.55459 * 0.5 * ((98.2 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((98.2 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) * (0.03294892 * 0.5 * ((0.0872 - Q.kt2_2_z_disp3) / 0.03294892) * (1 + math.erf(((0.0872 - Q.kt2_2_z_disp3) / 0.03294892) / math.sqrt(2)))) / 0.9899332   # +0.5%  mres_sd_mass_b2z01 < 98.2 and kt2_2_z_disp3 < 0.0872
        - 0.005103411 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 156.0) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 156.0) / 19.14168) / math.sqrt(2)))) / 3.773009   # -0.5%  mres_sd_mass_b2z01 > 156
        - 0.0049184 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.5) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.5) / 15.88428) / math.sqrt(2)))) / 0.9432322   # -0.5%  lep_ptrel > 43.5
        + 0.004776037 * (3.20498 * 0.5 * ((Q.mass_neutral - 34.8) / 3.20498) * (1 + math.erf(((Q.mass_neutral - 34.8) / 3.20498) / math.sqrt(2)))) / 15.29105   # +0.5%  mass_neutral > 34.8
        - 0.004405938 * (1.0 * 0.5 * ((Q.sdb_2_n - 5.75) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 5.75) / 1.0) / math.sqrt(2)))) / 5.147125   # -0.4%  sdb_2_n > 5.75
        + 0.0043426 * (1.0 * 0.5 * ((Q.dc_n - 0.986) / 1.0) * (1 + math.erf(((Q.dc_n - 0.986) / 1.0) / math.sqrt(2)))) * (0.003546451 * 0.5 * ((0.0419 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0419 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) / 0.01087693   # +0.4%  dc_n > 0.986 and M2_b2 < 0.0419
        - 0.00433444 * (1.0 * 0.5 * ((Q.dc_n - 1.03) / 1.0) * (1 + math.erf(((Q.dc_n - 1.03) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((18.0 - Q.n_photon) / 1.0) * (1 + math.erf(((18.0 - Q.n_photon) / 1.0) / math.sqrt(2)))) / 3.644679   # -0.4%  dc_n > 1.03 and n_photon < 18
        + 0.004126765 * (0.0002686389 * 0.5 * ((0.000414 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000414 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.5 * 0.5 * ((Q.n_lund - 7.08) / 1.5) * (1 + math.erf(((Q.n_lund - 7.08) / 1.5) / math.sqrt(2)))) / 0.001113844   # +0.4%  e3_b2 < 0.000414 and n_lund > 7.08
        + 0.004096935 * (21.5 * 0.5 * ((80.8 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.8 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 5.313427   # +0.4%  n_pairs_kt_above_1 < 80.8
        - 0.003870006 * (6.0 * 0.5 * ((28.1 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.1 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) / 3.701599   # -0.4%  n_pairs_kt_above_3 < 28.1
        + 0.003793279 * (0.01578816 * 0.5 * ((0.0834 - Q.pz_lnd2) / 0.01578816) * (1 + math.erf(((0.0834 - Q.pz_lnd2) / 0.01578816) / math.sqrt(2)))) / 0.01884784   # +0.4%  pz_lnd2 < 0.0834
        - 0.0036965 * (1.0 * 0.5 * ((-0.0551 - Q.n_sd0_above_5) / 1.0) * (1 + math.erf(((-0.0551 - Q.n_sd0_above_5) / 1.0) / math.sqrt(2)))) / 0.0347057   # -0.4%  n_sd0_above_5 < -0.0551
        - 0.003644712 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) / 1.215201   # -0.4%  n_dr_0p4_up < 4
        + 0.003226975 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 0.989) / 1.0) * (1 + math.erf(((Q.n_muon - 0.989) / 1.0) / math.sqrt(2)))) / 0.9317885   # +0.3%  n_s3d_above_3 < 10 and n_muon > 0.989
        + 0.003214555 * (0.08093679 * 0.5 * ((0.282 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.282 - Q.z_displaced5) / 0.08093679) / math.sqrt(2)))) * (0.1240004 * 0.5 * ((7.09 - Q.pair_max_lnm2) / 0.1240004) * (1 + math.erf(((7.09 - Q.pair_max_lnm2) / 0.1240004) / math.sqrt(2)))) / 0.1351503   # +0.3%  z_displaced5 < 0.282 and pair_max_lnm2 < 7.09
        - 0.003103414 * (0.1750793 * 0.5 * ((0.452 - Q.max_abs_dz) / 0.1750793) * (1 + math.erf(((0.452 - Q.max_abs_dz) / 0.1750793) / math.sqrt(2)))) / 0.08077184   # -0.3%  max_abs_dz < 0.452
        - 0.002941912 * (0.09329774 * 0.5 * ((Q.z_charged_had - 0.72) / 0.09329774) * (1 + math.erf(((Q.z_charged_had - 0.72) / 0.09329774) / math.sqrt(2)))) / 0.01023233   # -0.3%  z_charged_had > 0.72
        + 0.002933468 * (1.5 * 0.5 * ((Q.n_dr_0p2_0p4 - 9.34) / 1.5) * (1 + math.erf(((Q.n_dr_0p2_0p4 - 9.34) / 1.5) / math.sqrt(2)))) / 4.623381   # +0.3%  n_dr_0p2_0p4 > 9.34
        + 0.002436428 * (1.0 * 0.5 * ((0.00403 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.00403 - Q.sv_2_n) / 1.0) / math.sqrt(2)))) / 0.04503196   # +0.2%  sv_2_n < 0.00403
        + 0.002411601 * (1.0 * 0.5 * ((0.991 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.991 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) * (5.497214 * 0.5 * ((8.21 - Q.sjf_4_1_max3d) / 5.497214) * (1 + math.erf(((8.21 - Q.sjf_4_1_max3d) / 5.497214) / math.sqrt(2)))) / 1.932278   # +0.2%  kt2_2_n_lep < 0.991 and sjf_4_1_max3d < 8.21
        - 0.002281974 * (0.0002686389 * 0.5 * ((0.000411 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000411 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.04590778 * 0.5 * ((0.19 - Q.dr_max_012) / 0.04590778) * (1 + math.erf(((0.19 - Q.dr_max_012) / 0.04590778) / math.sqrt(2)))) / 1.973036e-05   # -0.2%  e3_b2 < 0.000411 and dr_max_012 < 0.19
        - 0.002218784 * (1.0 * 0.5 * ((3.03 - Q.dc_1_n_disp3) / 1.0) * (1 + math.erf(((3.03 - Q.dc_1_n_disp3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_pt_above_50 - 1.07) / 1.0) * (1 + math.erf(((Q.n_pt_above_50 - 1.07) / 1.0) / math.sqrt(2)))) / 3.743737   # -0.2%  dc_1_n_disp3 < 3.03 and n_pt_above_50 > 1.07
        + 0.00198833 * (1.0 * 0.5 * ((3.65 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((3.65 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) / 0.3842031   # +0.2%  sjq_2_2_nch < 3.65
        - 0.001932659 * (0.06782196 * 0.5 * ((0.0379 - Q.z_muon) / 0.06782196) * (1 + math.erf(((0.0379 - Q.z_muon) / 0.06782196) / math.sqrt(2)))) * (0.7263644 * 0.5 * ((Q.mres_sd_prong_mass2 - -0.105) / 0.7263644) * (1 + math.erf(((Q.mres_sd_prong_mass2 - -0.105) / 0.7263644) / math.sqrt(2)))) / 0.2404627   # -0.2%  z_muon < 0.0379 and mres_sd_prong_mass2 > -0.105
        + 0.001518705 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 180.0) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 180.0) / 18.97147) / math.sqrt(2)))) / 1.85935   # +0.2%  mres_sd_mass_b0z005 > 180
        + 0.001513437 * (0.004067431 * 0.5 * ((Q.sdb_5_z - 0.0135) / 0.004067431) * (1 + math.erf(((Q.sdb_5_z - 0.0135) / 0.004067431) / math.sqrt(2)))) / 0.0273128   # +0.2%  sdb_5_z > 0.0135
        + 0.001508093 * (5.119423 * 0.5 * ((Q.mass_top5 - 63.7) / 5.119423) * (1 + math.erf(((Q.mass_top5 - 63.7) / 5.119423) / math.sqrt(2)))) / 3.556159   # +0.2%  mass_top5 > 63.7
        - 0.00141375 * (3.20498 * 0.5 * ((Q.mass_neutral - 34.5) / 3.20498) * (1 + math.erf(((Q.mass_neutral - 34.5) / 3.20498) / math.sqrt(2)))) * (0.203947 * 0.5 * ((2.37 - Q.jd_3d_6) / 0.203947) * (1 + math.erf(((2.37 - Q.jd_3d_6) / 0.203947) / math.sqrt(2)))) / 5.663786   # -0.1%  mass_neutral > 34.5 and jd_3d_6 < 2.37
        - 0.00140127 * (1.0 * 0.5 * ((0.00438 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.00438 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2)))) / 0.03191172   # -0.1%  dc_1_n_lep < 0.00438
        - 0.001325071 * (1.5 * 0.5 * ((Q.lepsj_2_n_d3 - 2.18) / 1.5) * (1 + math.erf(((Q.lepsj_2_n_d3 - 2.18) / 1.5) / math.sqrt(2)))) / 0.3508399   # -0.1%  lepsj_2_n_d3 > 2.18
        - 0.001285043 * (1.118837 * 0.5 * ((6.58 - Q.sj2_mass2) / 1.118837) * (1 + math.erf(((6.58 - Q.sj2_mass2) / 1.118837) / math.sqrt(2)))) / 0.6223406   # -0.1%  sj2_mass2 < 6.58
        - 0.00107691 * (0.02907957 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.22) / 0.02907957) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.22) / 0.02907957) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.46 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.46 - Q.sip_3d_3) / 2.093785) / math.sqrt(2)))) / 0.1686874   # -0.1%  sjq_2_sumabs_k1 > 0.22 and sip_3d_3 < 4.46
        + 0.0009686044 * (0.008519048 * 0.5 * ((0.0152 - Q.z_dr_0p05_0p1) / 0.008519048) * (1 + math.erf(((0.0152 - Q.z_dr_0p05_0p1) / 0.008519048) / math.sqrt(2)))) / 0.002076089   # +0.1%  z_dr_0p05_0p1 < 0.0152
        + 0.000688345 * (1.0 * 0.5 * ((Q.sdb_0_n - 3.09) / 1.0) * (1 + math.erf(((Q.sdb_0_n - 3.09) / 1.0) / math.sqrt(2)))) / 0.2764896   # +0.1%  sdb_0_n > 3.09
        - 0.0003148208 * (14.45336 * 0.5 * ((Q.jd_3d_6 - 30.9) / 14.45336) * (1 + math.erf(((Q.jd_3d_6 - 30.9) / 14.45336) / math.sqrt(2)))) / 5.708455   # -0.0%  jd_3d_6 > 30.9
    )
    return z


def neuron_17(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.04e-05
    )
    return z


def neuron_18(Q):
    # scale S = 8.823; each line: share * term / its average size
    z = 8.822578 * (0.06528704
        + 0.06502779 * (14.89455 * 0.5 * ((146.0 - Q.mass_top20) / 14.89455) * (1 + math.erf(((146.0 - Q.mass_top20) / 14.89455) / math.sqrt(2)))) / 54.63931   # +6.5%  mass_top20 < 146
        + 0.06310875 * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) / 2.614   # +6.3%  lepsj_3_n_d3 < 3
        - 0.05298465 * (0.1060766 * 0.5 * ((0.221 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) / 0.1669504   # -5.3%  lep_z < 0.221
        - 0.0471594 * (2.5 * 0.5 * ((Q.n_particles - 20.9) / 2.5) * (1 + math.erf(((Q.n_particles - 20.9) / 2.5) / math.sqrt(2)))) / 18.82658   # -4.7%  n_particles > 20.9
        - 0.04520615 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 6.361002   # -4.5%  n_s3d_above_3 < 10
        - 0.04406806 * (540.2081 * 0.5 * ((586.0 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.0 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2)))) / 469.5578   # -4.4%  lepsj_3_maxsd0 < 586
        + 0.04060576 * (1.0 * 0.5 * ((8.97 - Q.jd_n_d3_pt1) / 1.0) * (1 + math.erf(((8.97 - Q.jd_n_d3_pt1) / 1.0) / math.sqrt(2)))) / 5.844168   # +4.1%  jd_n_d3_pt1 < 8.97
        - 0.03433632 * (4.146938 * 0.5 * ((Q.mass - 110.0) / 4.146938) * (1 + math.erf(((Q.mass - 110.0) / 4.146938) / math.sqrt(2)))) / 16.82972   # -3.4%  mass > 110
        + 0.03351813 * (5.001896 * 0.5 * ((2.45 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.45 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) / 1.247748   # +3.4%  lepsj_3_maxsd0 < 2.45
        - 0.03298493 * (0.006229165 * 0.5 * ((0.0041 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.0041 - Q.lep_z) / 0.006229165) / math.sqrt(2)))) / 0.001807529   # -3.3%  lep_z < 0.0041
        + 0.03245218 * (0.005001016 * 0.5 * ((0.0503 - Q.tau5) / 0.005001016) * (1 + math.erf(((0.0503 - Q.tau5) / 0.005001016) / math.sqrt(2)))) / 0.02002181   # +3.2%  tau5 < 0.0503
        - 0.02987957 * (0.1060766 * 0.5 * ((0.221 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.02653106 * 0.5 * ((0.104 - Q.z_displaced5) / 0.02653106) * (1 + math.erf(((0.104 - Q.z_displaced5) / 0.02653106) / math.sqrt(2)))) / 0.009763514   # -3.0%  lep_z < 0.221 and z_displaced5 < 0.104
        + 0.02893132 * (1.234622 * 0.5 * ((1.38 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.38 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) / 0.8565397   # +2.9%  lep_iso < 1.38
        - 0.0279908 * (1.0 * 0.5 * ((0.00941 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.00941 - Q.n_lepton) / 1.0) / math.sqrt(2)))) / 0.05677036   # -2.8%  n_lepton < 0.00941
        + 0.02561862 * (0.1060766 * 0.5 * ((0.222 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.222 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.01348041 * 0.5 * ((0.383 - Q.N2) / 0.01348041) * (1 + math.erf(((0.383 - Q.N2) / 0.01348041) / math.sqrt(2)))) / 0.01269788   # +2.6%  lep_z < 0.222 and N2 < 0.383
        + 0.0247526 * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.04808774 * 0.5 * ((0.0987 - Q.lepsj_3_dr) / 0.04808774) * (1 + math.erf(((0.0987 - Q.lepsj_3_dr) / 0.04808774) / math.sqrt(2)))) / 0.2298755   # +2.5%  lepsj_3_n_d3 < 3 and lepsj_3_dr < 0.0987
        - 0.02321821 * (5.97473 * 0.5 * ((115.0 - Q.mass_top20) / 5.97473) * (1 + math.erf(((115.0 - Q.mass_top20) / 5.97473) / math.sqrt(2)))) / 27.1677   # -2.3%  mass_top20 < 115
        - 0.02230647 * (1.234622 * 0.5 * ((1.35 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.35 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (0.4913027 * 0.5 * ((-5.15 - Q.ktd_ln_d23) / 0.4913027) * (1 + math.erf(((-5.15 - Q.ktd_ln_d23) / 0.4913027) / math.sqrt(2)))) / 2.269903   # -2.2%  lep_iso < 1.35 and ktd_ln_d23 < -5.15
        - 0.02158435 * (0.01141967 * 0.5 * ((0.0353 - Q.z_displaced3) / 0.01141967) * (1 + math.erf(((0.0353 - Q.z_displaced3) / 0.01141967) / math.sqrt(2)))) / 0.0119767   # -2.2%  z_displaced3 < 0.0353
        + 0.01986054 * (0.01670814 * 0.5 * ((0.0629 - Q.z_displaced5) / 0.01670814) * (1 + math.erf(((0.0629 - Q.z_displaced5) / 0.01670814) / math.sqrt(2)))) / 0.02891439   # +2.0%  z_displaced5 < 0.0629
        - 0.01902446 * (1.0 * 0.5 * ((3.03 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((3.03 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 0.9171847   # -1.9%  n_s3d_above_3 < 3.03
        - 0.01738205 * (2.463307 * 0.5 * ((3.59 - Q.sjf_3_1_max3d) / 2.463307) * (1 + math.erf(((3.59 - Q.sjf_3_1_max3d) / 2.463307) / math.sqrt(2)))) / 0.5146125   # -1.7%  sjf_3_1_max3d < 3.59
        - 0.01730895 * (0.1060766 * 0.5 * ((0.222 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.222 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.02343699 * 0.5 * ((Q.sdb_2_z - 0.225) / 0.02343699) * (1 + math.erf(((Q.sdb_2_z - 0.225) / 0.02343699) / math.sqrt(2)))) / 0.01809355   # -1.7%  lep_z < 0.222 and sdb_2_z > 0.225
        + 0.01561129 * (5.55459 * 0.5 * ((Q.mres_sd_mass_b2z01 - 96.9) / 5.55459) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 96.9) / 5.55459) / math.sqrt(2)))) / 20.68046   # +1.6%  mres_sd_mass_b2z01 > 96.9
        - 0.01399947 * (2.0 * 0.5 * ((27.1 - Q.n_for_90pct) / 2.0) * (1 + math.erf(((27.1 - Q.n_for_90pct) / 2.0) / math.sqrt(2)))) / 8.57718   # -1.4%  n_for_90pct < 27.1
        - 0.01320934 * (0.3893833 * 0.5 * ((Q.lund3_lndelta - -2.81) / 0.3893833) * (1 + math.erf(((Q.lund3_lndelta - -2.81) / 0.3893833) / math.sqrt(2)))) / 1.312392   # -1.3%  lund3_lndelta > -2.81
        + 0.0130186 * (0.008548367 * 0.5 * ((Q.N2_b05 - 0.435) / 0.008548367) * (1 + math.erf(((Q.N2_b05 - 0.435) / 0.008548367) / math.sqrt(2)))) / 0.0307106   # +1.3%  N2_b05 > 0.435
        - 0.0111694 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0) / 6.991655) / math.sqrt(2)))) / 7.245805   # -1.1%  mres_sd_mass_b2z01 > 130
        - 0.01059848 * (0.1060766 * 0.5 * ((0.22 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.22 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.03286901 * 0.5 * ((0.484 - Q.ak02_dr12) / 0.03286901) * (1 + math.erf(((0.484 - Q.ak02_dr12) / 0.03286901) / math.sqrt(2)))) / 0.02331819   # -1.1%  lep_z < 0.22 and ak02_dr12 < 0.484
        + 0.01046274 * (20.79165 * 0.5 * ((57.3 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((57.3 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) / 30.66722   # +1.0%  sv_1_sd0_sum < 57.3
        + 0.01027212 * (16.90428 * 0.5 * ((Q.mass - 165.0) / 16.90428) * (1 + math.erf(((Q.mass - 165.0) / 16.90428) / math.sqrt(2)))) / 3.17988   # +1.0%  mass > 165
        - 0.01005037 * (4.85476 * 0.5 * ((13.3 - Q.mass_displaced3) / 4.85476) * (1 + math.erf(((13.3 - Q.mass_displaced3) / 4.85476) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 11.6) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 11.6) / 1.0) / math.sqrt(2)))) / 58.72199   # -1.0%  mass_displaced3 < 13.3 and n_charged_pt_above_1 > 11.6
        + 0.009955562 * (1.5 * 0.5 * ((5.97 - Q.n_s3d_above_10) / 1.5) * (1 + math.erf(((5.97 - Q.n_s3d_above_10) / 1.5) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.238 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.238 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2)))) / 0.3852356   # +1.0%  n_s3d_above_10 < 5.97 and z_neutral_had < 0.238
        + 0.009391961 * (5.898438 * 0.5 * ((9.51 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((9.51 - Q.max_abs_d0) / 5.898438) / math.sqrt(2)))) / 4.932221   # +0.9%  max_abs_d0 < 9.51
        + 0.009356156 * (0.006229165 * 0.5 * ((0.00471 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.00471 - Q.lep_z) / 0.006229165) / math.sqrt(2)))) * (117.3924 * 0.5 * ((127.0 - Q.sjf_3_1_max3d) / 117.3924) * (1 + math.erf(((127.0 - Q.sjf_3_1_max3d) / 117.3924) / math.sqrt(2)))) / 0.1537159   # +0.9%  lep_z < 0.00471 and sjf_3_1_max3d < 127
        + 0.009207091 * (1.0 * 0.5 * ((Q.n_lepton - 0.94) / 1.0) * (1 + math.erf(((Q.n_lepton - 0.94) / 1.0) / math.sqrt(2)))) / 0.2491726   # +0.9%  n_lepton > 0.94
        - 0.008988986 * (0.001754707 * 0.5 * ((0.00932 - Q.sum_z_dr2_top2) / 0.001754707) * (1 + math.erf(((0.00932 - Q.sum_z_dr2_top2) / 0.001754707) / math.sqrt(2)))) / 0.002215252   # -0.9%  sum_z_dr2_top2 < 0.00932
        + 0.008269734 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 158.0) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 158.0) / 19.14168) / math.sqrt(2)))) / 3.524656   # +0.8%  mres_sd_mass_b2z01 > 158
        + 0.008071412 * (1.0 * 0.5 * ((2.01 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((2.01 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2)))) / 0.7222177   # +0.8%  n_sdz_above_5 < 2.01
        + 0.006414746 * (1.0 * 0.5 * ((4.08 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.08 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) / 1.257658   # +0.6%  n_dr_0p4_up < 4.08
        + 0.005657389 * (4.85476 * 0.5 * ((13.0 - Q.mass_displaced3) / 4.85476) * (1 + math.erf(((13.0 - Q.mass_displaced3) / 4.85476) / math.sqrt(2)))) * (0.06729888 * 0.5 * ((Q.jet_abs_eta - 0.413) / 0.06729888) * (1 + math.erf(((Q.jet_abs_eta - 0.413) / 0.06729888) / math.sqrt(2)))) / 3.418682   # +0.6%  mass_displaced3 < 13 and jet_abs_eta > 0.413
        - 0.004829282 * (0.005299632 * 0.5 * ((Q.z_displaced5 - 0.0108) / 0.005299632) * (1 + math.erf(((Q.z_displaced5 - 0.0108) / 0.005299632) / math.sqrt(2)))) / 0.08538421   # -0.5%  z_displaced5 > 0.0108
        - 0.004801372 * (11.17405 * 0.5 * ((Q.sj3_pair_mass_max - 129.0) / 11.17405) * (1 + math.erf(((Q.sj3_pair_mass_max - 129.0) / 11.17405) / math.sqrt(2)))) / 2.567302   # -0.5%  sj3_pair_mass_max > 129
        + 0.004643334 * (1.268269 * 0.5 * ((5.13 - Q.sj2_mass2) / 1.268269) * (1 + math.erf(((5.13 - Q.sj2_mass2) / 1.268269) / math.sqrt(2)))) / 0.3724198   # +0.5%  sj2_mass2 < 5.13
        + 0.004339866 * (3.069852 * 0.5 * ((-11.1 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((-11.1 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) / 0.0001126141   # +0.4%  sjf_2_2_max3d < -11.1
        + 0.004068772 * (0.1783594 * 0.5 * ((Q.ktd_ln_d34 - -9.17) / 0.1783594) * (1 + math.erf(((Q.ktd_ln_d34 - -9.17) / 0.1783594) / math.sqrt(2)))) / 0.4655909   # +0.4%  ktd_ln_d34 > -9.17
        + 0.004000314 * (0.03770036 * 0.5 * ((0.359 - Q.z_charged_had) / 0.03770036) * (1 + math.erf(((0.359 - Q.z_charged_had) / 0.03770036) / math.sqrt(2)))) / 0.02262377   # +0.4%  z_charged_had < 0.359
        + 0.003771977 * (0.1763298 * 0.5 * ((2.06 - Q.sjf_3_1_max3d) / 0.1763298) * (1 + math.erf(((2.06 - Q.sjf_3_1_max3d) / 0.1763298) / math.sqrt(2)))) / 0.1046496   # +0.4%  sjf_3_1_max3d < 2.06
        + 0.003447623 * (1.0 * 0.5 * ((1.96 - Q.sjq_3_3_nch) / 1.0) * (1 + math.erf(((1.96 - Q.sjq_3_3_nch) / 1.0) / math.sqrt(2)))) / 0.3540969   # +0.3%  sjq_3_3_nch < 1.96
        - 0.002293525 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 179.0) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 179.0) / 18.97147) / math.sqrt(2)))) / 1.908943   # -0.2%  mres_sd_mass_b0z005 > 179
        - 0.002162273 * (0.005001016 * 0.5 * ((0.0503 - Q.tau5) / 0.005001016) * (1 + math.erf(((0.0503 - Q.tau5) / 0.005001016) / math.sqrt(2)))) * (0.01243073 * 0.5 * ((Q.ak02_3_z - 0.017) / 0.01243073) * (1 + math.erf(((Q.ak02_3_z - 0.017) / 0.01243073) / math.sqrt(2)))) / 0.0003283447   # -0.2%  tau5 < 0.0503 and ak02_3_z > 0.017
        - 0.002131745 * (0.01539698 * 0.5 * ((Q.N2 - 0.396) / 0.01539698) * (1 + math.erf(((Q.N2 - 0.396) / 0.01539698) / math.sqrt(2)))) / 0.005373568   # -0.2%  N2 > 0.396
        - 0.002018062 * (0.001001007 * 0.5 * ((0.000643 - Q.sum_z_dr2_top3) / 0.001001007) * (1 + math.erf(((0.000643 - Q.sum_z_dr2_top3) / 0.001001007) / math.sqrt(2)))) / 2.422383e-05   # -0.2%  sum_z_dr2_top3 < 0.000643
        - 0.001614888 * (20.79165 * 0.5 * ((52.8 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((52.8 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_1_n_lep - 0.000424) / 1.0) * (1 + math.erf(((Q.dc_1_n_lep - 0.000424) / 1.0) / math.sqrt(2)))) / 5.609243   # -0.2%  sv_1_sd0_sum < 52.8 and dc_1_n_lep > 0.000424
        + 0.001429497 * (0.005299632 * 0.5 * ((Q.z_displaced5 - -0.000394) / 0.005299632) * (1 + math.erf(((Q.z_displaced5 - -0.000394) / 0.005299632) / math.sqrt(2)))) * (0.003791252 * 0.5 * ((0.0194 - Q.dr12) / 0.003791252) * (1 + math.erf(((0.0194 - Q.dr12) / 0.003791252) / math.sqrt(2)))) / 0.0002638462   # +0.1%  z_displaced5 > -0.000394 and dr12 < 0.0194
        - 0.0007246935 * (0.117671 * 0.5 * ((Q.lep_dr - 0.428) / 0.117671) * (1 + math.erf(((Q.lep_dr - 0.428) / 0.117671) / math.sqrt(2)))) / 0.00883103   # -0.1%  lep_dr > 0.428
        - 0.0007078383 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 157.0) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 157.0) / 19.14168) / math.sqrt(2)))) * (0.04048364 * 0.5 * ((0.274 - Q.ak02_dr23) / 0.04048364) * (1 + math.erf(((0.274 - Q.ak02_dr23) / 0.04048364) / math.sqrt(2)))) / 0.08946933   # -0.1%  mres_sd_mass_b2z01 > 157 and ak02_dr23 < 0.274
    )
    return z


def neuron_19(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.41e-06
    )
    return z


def neuron_20(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.21e-07
    )
    return z


def neuron_21(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.00168
    )
    return z


def neuron_22(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.05e-06
    )
    return z


def neuron_23(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.98e-07
    )
    return z


def neuron_24(Q):
    # scale S = 9.359; each line: share * term / its average size
    z = 9.358981 * (0.09456157
        - 0.09362579 * (18.42178 * 0.5 * ((183.0 - Q.mass) / 18.42178) * (1 + math.erf(((183.0 - Q.mass) / 18.42178) / math.sqrt(2)))) / 69.54302   # -9.4%  mass < 183
        - 0.07083021 * (0.1486471 * 0.5 * ((0.34 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.34 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) / 0.2705709   # -7.1%  lep_z < 0.34
        + 0.06308061 * (4.167088 * 0.5 * ((119.0 - Q.mres_pruned_mass) / 4.167088) * (1 + math.erf(((119.0 - Q.mres_pruned_mass) / 4.167088) / math.sqrt(2)))) / 32.26067   # +6.3%  mres_pruned_mass < 119
        - 0.06238548 * (0.05826336 * 0.5 * ((0.103 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) / 0.07858204   # -6.2%  lepsj_2_dr < 0.103
        + 0.06003314 * (3.283929 * 0.5 * ((118.0 - Q.mres_sd_mass_b2z01) / 3.283929) * (1 + math.erf(((118.0 - Q.mres_sd_mass_b2z01) / 3.283929) / math.sqrt(2)))) / 22.47396   # +6.0%  mres_sd_mass_b2z01 < 118
        - 0.05948839 * (4.971962 * 0.5 * ((81.3 - Q.mres_sd_mass_b2z01) / 4.971962) * (1 + math.erf(((81.3 - Q.mres_sd_mass_b2z01) / 4.971962) / math.sqrt(2)))) / 6.534633   # -5.9%  mres_sd_mass_b2z01 < 81.3
        + 0.0387783 * (0.6671377 * 0.5 * ((0.448 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.448 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2326445   # +3.9%  lep_iso < 0.448
        + 0.03651705 * (2.0 * 0.5 * ((17.0 - Q.n_dr_0p2_0p4) / 2.0) * (1 + math.erf(((17.0 - Q.n_dr_0p2_0p4) / 2.0) / math.sqrt(2)))) / 7.318253   # +3.7%  n_dr_0p2_0p4 < 17
        + 0.03490295 * (5.001896 * 0.5 * ((2.42 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.42 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) / 1.22803   # +3.5%  lepsj_3_maxsd0 < 2.42
        - 0.03440294 * (1.0 * 0.5 * ((9.06 - Q.n_sd0_above_3) / 1.0) * (1 + math.erf(((9.06 - Q.n_sd0_above_3) / 1.0) / math.sqrt(2)))) / 5.897006   # -3.4%  n_sd0_above_3 < 9.06
        - 0.03370163 * (5.270719 * 0.5 * ((86.1 - Q.mres_pruned_mass) / 5.270719) * (1 + math.erf(((86.1 - Q.mres_pruned_mass) / 5.270719) / math.sqrt(2)))) / 13.95633   # -3.4%  mres_pruned_mass < 86.1
        + 0.02724732 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2)))) / 15.93795   # +2.7%  mass < 118
        - 0.02655764 * (6.875137 * 0.5 * ((19.2 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((19.2 - Q.mass_displaced3) / 6.875137) / math.sqrt(2)))) * (0.09779513 * 0.5 * ((0.671 - Q.sdb_2_z) / 0.09779513) * (1 + math.erf(((0.671 - Q.sdb_2_z) / 0.09779513) / math.sqrt(2)))) / 4.375924   # -2.7%  mass_displaced3 < 19.2 and sdb_2_z < 0.671
        - 0.02546614 * (12.49365 * 0.5 * ((128.0 - Q.sj4_pair_mass_max) / 12.49365) * (1 + math.erf(((128.0 - Q.sj4_pair_mass_max) / 12.49365) / math.sqrt(2)))) / 47.95515   # -2.5%  sj4_pair_mass_max < 128
        + 0.02348875 * (10.51039 * 0.5 * ((69.2 - Q.mres_sd_mass_b2z01) / 10.51039) * (1 + math.erf(((69.2 - Q.mres_sd_mass_b2z01) / 10.51039) / math.sqrt(2)))) / 4.344481   # +2.3%  mres_sd_mass_b2z01 < 69.2
        - 0.02297574 * (0.1486471 * 0.5 * ((0.339 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.339 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 0.994) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 0.994) / 1.0) / math.sqrt(2)))) / 0.2953702   # -2.3%  lep_z < 0.339 and n_lund_kt_above_5 > 0.994
        - 0.02258108 * (0.0002686389 * 0.5 * ((0.000412 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000412 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) / 0.0002863631   # -2.3%  e3_b2 < 0.000412
        + 0.02186826 * (6.875137 * 0.5 * ((18.6 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.6 - Q.mass_displaced3) / 6.875137) / math.sqrt(2)))) / 13.11953   # +2.2%  mass_displaced3 < 18.6
        + 0.01557935 * (0.1026619 * 0.5 * ((0.919 - Q.N3_b05) / 0.1026619) * (1 + math.erf(((0.919 - Q.N3_b05) / 0.1026619) / math.sqrt(2)))) / 0.258982   # +1.6%  N3_b05 < 0.919
        - 0.01553969 * (0.0257051 * 0.5 * ((0.148 - Q.z_dr_0p2_0p4) / 0.0257051) * (1 + math.erf(((0.148 - Q.z_dr_0p2_0p4) / 0.0257051) / math.sqrt(2)))) / 0.04930021   # -1.6%  z_dr_0p2_0p4 < 0.148
        - 0.01424161 * (0.01304025 * 0.5 * ((0.0126 - Q.lep_z) / 0.01304025) * (1 + math.erf(((0.0126 - Q.lep_z) / 0.01304025) / math.sqrt(2)))) / 0.006470242   # -1.4%  lep_z < 0.0126
        + 0.01353565 * (0.4384575 * 0.5 * ((Q.pair_max_lnkt - 2.33) / 0.4384575) * (1 + math.erf(((Q.pair_max_lnkt - 2.33) / 0.4384575) / math.sqrt(2)))) / 0.4762403   # +1.4%  pair_max_lnkt > 2.33
        + 0.01127511 * (15.88428 * 0.5 * ((43.0 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.0 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.08912976 * 0.5 * ((Q.lne_0 - 4.82) / 0.08912976) * (1 + math.erf(((Q.lne_0 - 4.82) / 0.08912976) / math.sqrt(2)))) / 15.20512   # +1.1%  lep_ptrel < 43 and lne_0 > 4.82
        + 0.01077006 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2)))) * (1.15674 * 0.5 * ((Q.lne_17 - -0.102) / 1.15674) * (1 + math.erf(((Q.lne_17 - -0.102) / 1.15674) / math.sqrt(2)))) / 25.84532   # +1.1%  mass < 118 and lne_17 > -0.102
        + 0.009799553 * (8.537109 * 0.5 * ((18.2 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((18.2 - Q.max_abs_d0) / 8.537109) / math.sqrt(2)))) / 10.85371   # +1.0%  max_abs_d0 < 18.2
        + 0.009635497 * (14.176 * 0.5 * ((Q.mres_sd_prong_mass1 - 69.0) / 14.176) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 69.0) / 14.176) / math.sqrt(2)))) / 2.348397   # +1.0%  mres_sd_prong_mass1 > 69
        + 0.009633784 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2)))) * (0.01776485 * 0.5 * ((0.272 - Q.N2_b2) / 0.01776485) * (1 + math.erf(((0.272 - Q.N2_b2) / 0.01776485) / math.sqrt(2)))) / 1.456582   # +1.0%  mass < 118 and N2_b2 < 0.272
        + 0.009551586 * (3.24083 * 0.5 * ((6.92 - Q.dc_2_jp) / 3.24083) * (1 + math.erf(((6.92 - Q.dc_2_jp) / 3.24083) / math.sqrt(2)))) / 3.709258   # +1.0%  dc_2_jp < 6.92
        - 0.009419095 * (0.071994 * 0.5 * ((0.458 - Q.N3_b2) / 0.071994) * (1 + math.erf(((0.458 - Q.N3_b2) / 0.071994) / math.sqrt(2)))) / 0.1912216   # -0.9%  N3_b2 < 0.458
        + 0.008244056 * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.99) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.99) / 1.0) / math.sqrt(2)))) / 0.4310389   # +0.8%  n_lund_kt_above_5 > 1.99
        - 0.007145915 * (0.7154015 * 0.5 * ((Q.mass_displaced3 - 1.76) / 0.7154015) * (1 + math.erf(((Q.mass_displaced3 - 1.76) / 0.7154015) / math.sqrt(2)))) / 6.887588   # -0.7%  mass_displaced3 > 1.76
        + 0.006851494 * (0.0002686389 * 0.5 * ((0.000412 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000412 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_neutral_had - 0.998) / 1.0) * (1 + math.erf(((Q.n_neutral_had - 0.998) / 1.0) / math.sqrt(2)))) / 0.0008306089   # +0.7%  e3_b2 < 0.000412 and n_neutral_had > 0.998
        - 0.006303797 * (15.148 * 0.5 * ((Q.mres_sd_prong_mass1 - 83.9) / 15.148) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 83.9) / 15.148) / math.sqrt(2)))) / 1.242045   # -0.6%  mres_sd_prong_mass1 > 83.9
        + 0.00602048 * (0.560057 * 0.5 * ((1.77 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.77 - Q.mass_displaced5) / 0.560057) / math.sqrt(2)))) / 0.8142422   # +0.6%  mass_displaced5 < 1.77
        - 0.005548793 * (1.0 * 0.5 * ((2.03 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((2.03 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2)))) / 0.3372146   # -0.6%  n_lund_kt_above_5 < 2.03
        - 0.005272691 * (1.0 * 0.5 * ((9.03 - Q.n_sd0_above_3) / 1.0) * (1 + math.erf(((9.03 - Q.n_sd0_above_3) / 1.0) / math.sqrt(2)))) * (0.05422308 * 0.5 * ((Q.z_neutral - 0.141) / 0.05422308) * (1 + math.erf(((Q.z_neutral - 0.141) / 0.05422308) / math.sqrt(2)))) / 1.586721   # -0.5%  n_sd0_above_3 < 9.03 and z_neutral > 0.141
        + 0.004953935 * (0.2334666 * 0.5 * ((-0.605 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.605 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2)))) / 0.01948058   # +0.5%  sjq_3_2_k1 < -0.605
        + 0.00483177 * (1.5 * 0.5 * ((7.06 - Q.n_lund) / 1.5) * (1 + math.erf(((7.06 - Q.n_lund) / 1.5) / math.sqrt(2)))) / 0.3055435   # +0.5%  n_lund < 7.06
        + 0.004775151 * (0.001965812 * 0.5 * ((Q.M3 - 0.0324) / 0.001965812) * (1 + math.erf(((Q.M3 - 0.0324) / 0.001965812) / math.sqrt(2)))) / 0.008215175   # +0.5%  M3 > 0.0324
        - 0.004736732 * (5.061224 * 0.5 * ((88.9 - Q.mass) / 5.061224) * (1 + math.erf(((88.9 - Q.mass) / 5.061224) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.04 - Q.sv_1_n) / 1.0) * (1 + math.erf(((3.04 - Q.sv_1_n) / 1.0) / math.sqrt(2)))) / 11.28015   # -0.5%  mass < 88.9 and sv_1_n < 3.04
        - 0.004568897 * (6.170706 * 0.5 * ((76.1 - Q.mass_top30) / 6.170706) * (1 + math.erf(((76.1 - Q.mass_top30) / 6.170706) / math.sqrt(2)))) / 2.96946   # -0.5%  mass_top30 < 76.1
        + 0.004540994 * (0.06488792 * 0.5 * ((Q.dc_split2_dr - 0.279) / 0.06488792) * (1 + math.erf(((Q.dc_split2_dr - 0.279) / 0.06488792) / math.sqrt(2)))) / 0.0108416   # +0.5%  dc_split2_dr > 0.279
        + 0.004318327 * (0.2398529 * 0.5 * ((Q.sjq_3_2_k1 - 0.614) / 0.2398529) * (1 + math.erf(((Q.sjq_3_2_k1 - 0.614) / 0.2398529) / math.sqrt(2)))) / 0.01981134   # +0.4%  sjq_3_2_k1 > 0.614
        - 0.004315021 * (0.4384575 * 0.5 * ((2.35 - Q.pair_max_lnkt) / 0.4384575) * (1 + math.erf(((2.35 - Q.pair_max_lnkt) / 0.4384575) / math.sqrt(2)))) / 0.1412035   # -0.4%  pair_max_lnkt < 2.35
        + 0.003736807 * (0.0002686389 * 0.5 * ((0.00041 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.00041 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.05415789 * 0.5 * ((-2.05 - Q.lnerel_1) / 0.05415789) * (1 + math.erf(((-2.05 - Q.lnerel_1) / 0.05415789) / math.sqrt(2)))) / 6.029777e-05   # +0.4%  e3_b2 < 0.00041 and lnerel_1 < -2.05
        - 0.003329354 * (0.02561531 * 0.5 * ((0.108 - Q.z_displaced3) / 0.02561531) * (1 + math.erf(((0.108 - Q.z_displaced3) / 0.02561531) / math.sqrt(2)))) * (2.8693e-06 * 0.5 * ((2.04e-05 - Q.ecf_g42) / 2.8693e-06) * (1 + math.erf(((2.04e-05 - Q.ecf_g42) / 2.8693e-06) / math.sqrt(2)))) / 3.50499e-07   # -0.3%  z_displaced3 < 0.108 and ecf_g42 < 2.04e-05
        - 0.003236432 * (0.002611265 * 0.5 * ((0.0105 - Q.M2_b2) / 0.002611265) * (1 + math.erf(((0.0105 - Q.M2_b2) / 0.002611265) / math.sqrt(2)))) / 0.0003597352   # -0.3%  M2_b2 < 0.0105
        - 0.003127637 * (0.07842787 * 0.5 * ((Q.dc_split2_dr - 0.354) / 0.07842787) * (1 + math.erf(((Q.dc_split2_dr - 0.354) / 0.07842787) / math.sqrt(2)))) / 0.005351279   # -0.3%  dc_split2_dr > 0.354
        - 0.002779338 * (0.6671377 * 0.5 * ((0.464 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.464 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (0.02224606 * 0.5 * ((0.255 - Q.sj2_dr) / 0.02224606) * (1 + math.erf(((0.255 - Q.sj2_dr) / 0.02224606) / math.sqrt(2)))) / 0.003251472   # -0.3%  lep_iso < 0.464 and sj2_dr < 0.255
        + 0.002593132 * (0.2508028 * 0.5 * ((-0.661 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.661 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2)))) / 0.01733505   # +0.3%  sjq_3_3_k1 < -0.661
        + 0.002276934 * (0.02561531 * 0.5 * ((0.1 - Q.z_displaced3) / 0.02561531) * (1 + math.erf(((0.1 - Q.z_displaced3) / 0.02561531) / math.sqrt(2)))) * (0.9519596 * 0.5 * ((Q.mass_2photon - 2.47) / 0.9519596) * (1 + math.erf(((Q.mass_2photon - 2.47) / 0.9519596) / math.sqrt(2)))) / 0.1869279   # +0.2%  z_displaced3 < 0.1 and mass_2photon > 2.47
        + 0.002266883 * (0.004691441 * 0.5 * ((2.34e-05 - Q.z_muon) / 0.004691441) * (1 + math.erf(((2.34e-05 - Q.z_muon) / 0.004691441) / math.sqrt(2)))) / 2.31108e-05   # +0.2%  z_muon < 2.34e-05
        + 0.002087031 * (0.1486471 * 0.5 * ((0.34 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.34 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.05582565 * 0.5 * ((0.458 - Q.tau32) / 0.05582565) * (1 + math.erf(((0.458 - Q.tau32) / 0.05582565) / math.sqrt(2)))) / 0.003727574   # +0.2%  lep_z < 0.34 and tau32 < 0.458
        - 0.002013938 * (6.785484 * 0.5 * ((Q.mass_top5 - 67.7) / 6.785484) * (1 + math.erf(((Q.mass_top5 - 67.7) / 6.785484) / math.sqrt(2)))) / 2.899754   # -0.2%  mass_top5 > 67.7
        + 0.001321108 * (0.6671377 * 0.5 * ((0.444 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.444 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.00632) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.00632) / 1e-06) / math.sqrt(2)))) / 0.005088157   # +0.1%  lep_iso < 0.444 and ak02_3_n_lep > 0.00632
        - 0.001266729 * (15.148 * 0.5 * ((Q.mres_sd_prong_mass1 - 83.7) / 15.148) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 83.7) / 15.148) / math.sqrt(2)))) * (0.0947145 * 0.5 * ((0.0946 - Q.dc_4_z) / 0.0947145) * (1 + math.erf(((0.0946 - Q.dc_4_z) / 0.0947145) / math.sqrt(2)))) / 0.06478302   # -0.1%  mres_sd_prong_mass1 > 83.7 and dc_4_z < 0.0946
        + 0.000624211 * (3.143401 * 0.5 * ((Q.mass_top15 - 65.1) / 3.143401) * (1 + math.erf(((Q.mass_top15 - 65.1) / 3.143401) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.00436) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.00436) / 1e-06) / math.sqrt(2)))) / 1.738684   # +0.1%  mass_top15 > 65.1 and ak02_3_n_lep > 0.00436
    )
    return z


def neuron_25(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.0305
    )
    return z


def neuron_26(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.23e-06
    )
    return z


def neuron_27(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.3e-06
    )
    return z


def neuron_28(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.61e-06
    )
    return z


def neuron_29(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.18e-06
    )
    return z


def neuron_30(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.6e-06
    )
    return z


def neuron_31(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.51e-06
    )
    return z


def neuron_32(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.44e-05
    )
    return z


def neuron_33(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.56e-06
    )
    return z


def neuron_34(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.39e-05
    )
    return z


def neuron_35(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.74e-06
    )
    return z


def neuron_36(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.84e-06
    )
    return z


def neuron_37(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.35e-06
    )
    return z


def neuron_38(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.41e-06
    )
    return z


def neuron_39(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.27e-06
    )
    return z


def neuron_40(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.83e-07
    )
    return z


def neuron_41(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.12e-08
    )
    return z


def neuron_42(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.78e-06
    )
    return z


def neuron_43(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.4e-06
    )
    return z


def neuron_44(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.33e-06
    )
    return z


def neuron_45(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.85e-06
    )
    return z


def neuron_46(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.31e-07
    )
    return z


def neuron_47(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.46e-07
    )
    return z


def neuron_48(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.8e-06
    )
    return z


def neuron_49(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.24e-06
    )
    return z


def neuron_50(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.17e-06
    )
    return z


def neuron_51(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.02e-06
    )
    return z


def neuron_52(Q):
    # scale S = 9.847; each line: share * term / its average size
    z = 9.84702 * (-0.1360818
        + 0.1713194 * (18.66842 * 0.5 * ((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) / 56.99275   # +17.1%  mres_sd_mass_b0z005 < 160
        + 0.07152667 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.3) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.3) / 5.1259) / math.sqrt(2)))) / 31.72633   # +7.2%  mres_sd_mass_b0z005 > 81.3
        - 0.0657165 * (0.1486471 * 0.5 * ((0.339 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.339 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) / 0.2696298   # -6.6%  lep_z < 0.339
        - 0.06258438 * (1.5 * 0.5 * ((5.99 - Q.n_sd0_above_3) / 1.5) * (1 + math.erf(((5.99 - Q.n_sd0_above_3) / 1.5) / math.sqrt(2)))) / 3.128272   # -6.3%  n_sd0_above_3 < 5.99
        + 0.0532848 * (0.03097976 * 0.5 * ((0.134 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.134 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) / 0.07052372   # +5.3%  z_displaced3 < 0.134
        + 0.04347452 * (1.5 * 0.5 * ((7.01 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.01 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2)))) / 3.658927   # +4.3%  n_s3d_above_3 < 7.01
        - 0.03951516 * (5.001896 * 0.5 * ((1.59 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((1.59 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) * (0.0330605 * 0.5 * ((0.103 - Q.lep_dr) / 0.0330605) * (1 + math.erf(((0.103 - Q.lep_dr) / 0.0330605) / math.sqrt(2)))) / 0.05949641   # -4.0%  lepsj_3_maxsd0 < 1.59 and lep_dr < 0.103
        - 0.03138203 * (5.001896 * 0.5 * ((-15.4 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((-15.4 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) / 0.01174979   # -3.1%  lepsj_3_maxsd0 < -15.4
        - 0.03113968 * (0.03060878 * 0.5 * ((0.0667 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0667 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) / 0.04961699   # -3.1%  lepsj_2_dr < 0.0667
        - 0.03064753 * (4.488781 * 0.5 * ((126.0 - Q.mres_sd_mass_b0z005) / 4.488781) * (1 + math.erf(((126.0 - Q.mres_sd_mass_b0z005) / 4.488781) / math.sqrt(2)))) / 28.47045   # -3.1%  mres_sd_mass_b0z005 < 126
        + 0.02578802 * (1.234622 * 0.5 * ((1.5 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.5 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) / 0.9546435   # +2.6%  lep_iso < 1.5
        - 0.01869181 * (20.79165 * 0.5 * ((52.1 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((52.1 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) / 27.14729   # -1.9%  sv_1_sd0_sum < 52.1
        + 0.01782921 * (1.0 * 0.5 * ((1.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2)))) / 0.667546   # +1.8%  dc_1_n_lep < 1
        - 0.01777934 * (1.234622 * 0.5 * ((1.42 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.42 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (4.0 * 0.5 * ((23.9 - Q.n_dr_0p2_0p4) / 4.0) * (1 + math.erf(((23.9 - Q.n_dr_0p2_0p4) / 4.0) / math.sqrt(2)))) / 11.82929   # -1.8%  lep_iso < 1.42 and n_dr_0p2_0p4 < 23.9
        + 0.01673639 * (2.5 * 0.5 * ((Q.n_particles - 25.9) / 2.5) * (1 + math.erf(((Q.n_particles - 25.9) / 2.5) / math.sqrt(2)))) / 14.58439   # +1.7%  n_particles > 25.9
        - 0.01616734 * (1.5 * 0.5 * ((5.91 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((5.91 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2)))) / 2.287359   # -1.6%  n_dr_0p4_up < 5.91
        - 0.01583028 * (0.03097976 * 0.5 * ((0.134 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.134 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (110.8422 * 0.5 * ((116.0 - Q.sjf_2_2_max3d) / 110.8422) * (1 + math.erf(((116.0 - Q.sjf_2_2_max3d) / 110.8422) / math.sqrt(2)))) / 5.144592   # -1.6%  z_displaced3 < 0.134 and sjf_2_2_max3d < 116
        + 0.01500915 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.5) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.5) / 5.1259) / math.sqrt(2)))) * (0.03966789 * 0.5 * ((0.134 - Q.ak02_3_z) / 0.03966789) * (1 + math.erf(((0.134 - Q.ak02_3_z) / 0.03966789) / math.sqrt(2)))) / 2.176662   # +1.5%  mres_sd_mass_b0z005 > 81.5 and ak02_3_z < 0.134
        - 0.01467084 * (17.1726 * 0.5 * ((36.4 - Q.sjf_2_1_max3d) / 17.1726) * (1 + math.erf(((36.4 - Q.sjf_2_1_max3d) / 17.1726) / math.sqrt(2)))) / 15.85775   # -1.5%  sjf_2_1_max3d < 36.4
        - 0.01460739 * (0.03097976 * 0.5 * ((0.135 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.135 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (401.9953 * 0.5 * ((578.0 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((578.0 - Q.sip_3d_3) / 401.9953) / math.sqrt(2)))) / 35.69212   # -1.5%  z_displaced3 < 0.135 and sip_3d_3 < 578
        + 0.01440592 * (0.0002686389 * 0.5 * ((0.000414 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000414 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) / 0.000288324   # +1.4%  e3_b2 < 0.000414
        + 0.01422424 * (3.069852 * 0.5 * ((4.65 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.65 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) / 1.094268   # +1.4%  sjf_2_2_max3d < 4.65
        - 0.01386362 * (18.66842 * 0.5 * ((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) * (0.04741882 * 0.5 * ((0.466 - Q.tau21_b2) / 0.04741882) * (1 + math.erf(((0.466 - Q.tau21_b2) / 0.04741882) / math.sqrt(2)))) / 10.03789   # -1.4%  mres_sd_mass_b0z005 < 160 and tau21_b2 < 0.466
        + 0.01366104 * (0.1159667 * 0.5 * ((0.485 - Q.sv_1_dr) / 0.1159667) * (1 + math.erf(((0.485 - Q.sv_1_dr) / 0.1159667) / math.sqrt(2)))) / 0.3440424   # +1.4%  sv_1_dr < 0.485
        + 0.0116253 * (0.1486471 * 0.5 * ((0.339 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.339 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 11.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 11.0) / 1.0) / math.sqrt(2)))) / 2.472453   # +1.2%  lep_z < 0.339 and n_charged_pt_above_1 > 11
        - 0.01125763 * (1.0 * 0.5 * ((8.95 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((8.95 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) * (0.3035439 * 0.5 * ((Q.sjq_2_prod_k03 - -1.02) / 0.3035439) * (1 + math.erf(((Q.sjq_2_prod_k03 - -1.02) / 0.3035439) / math.sqrt(2)))) / 3.379699   # -1.1%  n_lund_kt_above_1 < 8.95 and sjq_2_prod_k03 > -1.02
        + 0.01104158 * (11.49171 * 0.5 * ((72.9 - Q.mass) / 11.49171) * (1 + math.erf(((72.9 - Q.mass) / 11.49171) / math.sqrt(2)))) / 2.237174   # +1.1%  mass < 72.9
        + 0.01100896 * (4.481852 * 0.5 * ((6.48 - Q.sjf_2_1_max3d) / 4.481852) * (1 + math.erf(((6.48 - Q.sjf_2_1_max3d) / 4.481852) / math.sqrt(2)))) / 1.348326   # +1.1%  sjf_2_1_max3d < 6.48
        - 0.01018979 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 82.1) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 82.1) / 5.1259) / math.sqrt(2)))) * (0.03693697 * 0.5 * ((0.584 - Q.tau32_b2) / 0.03693697) * (1 + math.erf(((0.584 - Q.tau32_b2) / 0.03693697) / math.sqrt(2)))) / 4.8473   # -1.0%  mres_sd_mass_b0z005 > 82.1 and tau32_b2 < 0.584
        + 0.009584823 * (0.0002686389 * 0.5 * ((0.000416 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000416 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.01565591 * 0.5 * ((0.168 - Q.pz_lnkt1) / 0.01565591) * (1 + math.erf(((0.168 - Q.pz_lnkt1) / 0.01565591) / math.sqrt(2)))) / 1.880118e-05   # +1.0%  e3_b2 < 0.000416 and pz_lnkt1 < 0.168
        + 0.008876501 * (3.304036 * 0.5 * ((116.0 - Q.mass_top50) / 3.304036) * (1 + math.erf(((116.0 - Q.mass_top50) / 3.304036) / math.sqrt(2)))) / 15.25429   # +0.9%  mass_top50 < 116
        - 0.008796155 * (24.25363 * 0.5 * ((53.0 - Q.mres_sd_mass_b0z005) / 24.25363) * (1 + math.erf(((53.0 - Q.mres_sd_mass_b0z005) / 24.25363) / math.sqrt(2)))) / 3.749606   # -0.9%  mres_sd_mass_b0z005 < 53
        - 0.007228292 * (0.0408915 * 0.5 * ((Q.lep_dr - 0.0653) / 0.0408915) * (1 + math.erf(((Q.lep_dr - 0.0653) / 0.0408915) / math.sqrt(2)))) / 0.06843955   # -0.7%  lep_dr > 0.0653
        + 0.006929973 * (0.03097976 * 0.5 * ((0.134 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.134 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (0.02727943 * 0.5 * ((Q.N2 - 0.183) / 0.02727943) * (1 + math.erf(((Q.N2 - 0.183) / 0.02727943) / math.sqrt(2)))) / 0.009734604   # +0.7%  z_displaced3 < 0.134 and N2 > 0.183
        + 0.006187645 * (21.5 * 0.5 * ((79.8 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((79.8 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 5.163547   # +0.6%  n_pairs_kt_above_1 < 79.8
        - 0.005793945 * (2.497115 * 0.5 * ((Q.mass_displaced5 - 6.01) / 2.497115) * (1 + math.erf(((Q.mass_displaced5 - 6.01) / 2.497115) / math.sqrt(2)))) / 3.989727   # -0.6%  mass_displaced5 > 6.01
        - 0.005788784 * (8.859433e-05 * 0.5 * ((0.000272 - Q.ecf_g41) / 8.859433e-05) * (1 + math.erf(((0.000272 - Q.ecf_g41) / 8.859433e-05) / math.sqrt(2)))) / 8.096914e-05   # -0.6%  ecf_g41 < 0.000272
        - 0.005518904 * (0.08168809 * 0.5 * ((Q.jet_abs_eta - 0.83) / 0.08168809) * (1 + math.erf(((Q.jet_abs_eta - 0.83) / 0.08168809) / math.sqrt(2)))) / 0.1692983   # -0.6%  jet_abs_eta > 0.83
        - 0.005366393 * (0.09740396 * 0.5 * ((1.31 - Q.D2) / 0.09740396) * (1 + math.erf(((1.31 - Q.D2) / 0.09740396) / math.sqrt(2)))) / 0.1024089   # -0.5%  D2 < 1.31
        - 0.005146361 * (8.307776 * 0.5 * ((Q.mass_top50 - 137.0) / 8.307776) * (1 + math.erf(((Q.mass_top50 - 137.0) / 8.307776) / math.sqrt(2)))) / 6.37438   # -0.5%  mass_top50 > 137
        - 0.005010558 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 177.0) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 177.0) / 18.97147) / math.sqrt(2)))) / 2.013839   # -0.5%  mres_sd_mass_b0z005 > 177
        + 0.004484163 * (0.2681069 * 0.5 * ((Q.sjq_2_2_k1 - 0.646) / 0.2681069) * (1 + math.erf(((Q.sjq_2_2_k1 - 0.646) / 0.2681069) / math.sqrt(2)))) / 0.02082813   # +0.4%  sjq_2_2_k1 > 0.646
        + 0.003967066 * (0.09740396 * 0.5 * ((1.32 - Q.D2) / 0.09740396) * (1 + math.erf(((1.32 - Q.D2) / 0.09740396) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjq_3_2_nch - 4.05) / 1.0) * (1 + math.erf(((Q.sjq_3_2_nch - 4.05) / 1.0) / math.sqrt(2)))) / 0.2206993   # +0.4%  D2 < 1.32 and sjq_3_2_nch > 4.05
        - 0.002946027 * (0.08734879 * 0.5 * ((1.6 - Q.jd_3d_6) / 0.08734879) * (1 + math.erf(((1.6 - Q.jd_3d_6) / 0.08734879) / math.sqrt(2)))) / 0.1198743   # -0.3%  jd_3d_6 < 1.6
        + 0.002945564 * (1.0 * 0.5 * ((7.78 - Q.n_lund) / 1.0) * (1 + math.erf(((7.78 - Q.n_lund) / 1.0) / math.sqrt(2)))) / 0.3831575   # +0.3%  n_lund < 7.78
        + 0.00270984 * (0.2610212 * 0.5 * ((-0.606 - Q.sjq_2_2_k1) / 0.2610212) * (1 + math.erf(((-0.606 - Q.sjq_2_2_k1) / 0.2610212) / math.sqrt(2)))) / 0.02320335   # +0.3%  sjq_2_2_k1 < -0.606
        + 0.002553705 * (8.859433e-05 * 0.5 * ((Q.ecf_g41 - 0.000269) / 8.859433e-05) * (1 + math.erf(((Q.ecf_g41 - 0.000269) / 8.859433e-05) / math.sqrt(2)))) / 3.112176e-05   # +0.3%  ecf_g41 > 0.000269
        + 0.002472006 * (0.0326228 * 0.5 * ((Q.kt2_dr12 - 0.494) / 0.0326228) * (1 + math.erf(((Q.kt2_dr12 - 0.494) / 0.0326228) / math.sqrt(2)))) / 0.02045537   # +0.2%  kt2_dr12 > 0.494
        + 0.002425807 * (0.0496275 * 0.5 * ((0.198 - Q.kt2_dr12) / 0.0496275) * (1 + math.erf(((0.198 - Q.kt2_dr12) / 0.0496275) / math.sqrt(2)))) / 0.008814379   # +0.2%  kt2_dr12 < 0.198
        - 0.002242752 * (0.03097976 * 0.5 * ((0.136 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.136 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_3_3_n_d3 - 0.971) / 1.0) * (1 + math.erf(((Q.sjf_3_3_n_d3 - 0.971) / 1.0) / math.sqrt(2)))) / 0.01840369   # -0.2%  z_displaced3 < 0.136 and sjf_3_3_n_d3 > 0.971
        + 0.001802145 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.0) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.0) / 18.97147) / math.sqrt(2)))) * (0.03559215 * 0.5 * ((0.624 - Q.tau32_b2) / 0.03559215) * (1 + math.erf(((0.624 - Q.tau32_b2) / 0.03559215) / math.sqrt(2)))) / 0.4669935   # +0.2%  mres_sd_mass_b0z005 > 178 and tau32_b2 < 0.624
        + 0.001780434 * (0.08149619 * 0.5 * ((0.242 - Q.pt1_over_pt0) / 0.08149619) * (1 + math.erf(((0.242 - Q.pt1_over_pt0) / 0.08149619) / math.sqrt(2)))) / 0.00958031   # +0.2%  pt1_over_pt0 < 0.242
        + 0.001604314 * (11.90664 * 0.5 * ((Q.sj3_pair_mass_min - 80.7) / 11.90664) * (1 + math.erf(((Q.sj3_pair_mass_min - 80.7) / 11.90664) / math.sqrt(2)))) / 0.8060057   # +0.2%  sj3_pair_mass_min > 80.7
        - 0.001275802 * (0.03097976 * 0.5 * ((0.134 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.134 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (0.0001714083 * 0.5 * ((0.00019 - Q.C3_b2) / 0.0001714083) * (1 + math.erf(((0.00019 - Q.C3_b2) / 0.0001714083) / math.sqrt(2)))) / 1.869471e-06   # -0.1%  z_displaced3 < 0.134 and C3_b2 < 0.00019
        - 0.0008983923 * (7.767937 * 0.5 * ((Q.sj3_mass1 - 43.5) / 7.767937) * (1 + math.erf(((Q.sj3_mass1 - 43.5) / 7.767937) / math.sqrt(2)))) / 0.7898649   # -0.1%  sj3_mass1 > 43.5
        + 0.0006851069 * (6.0 * 0.5 * ((28.8 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.8 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) * (1.375431 * 0.5 * ((Q.mass_2photon - 3.9) / 1.375431) * (1 + math.erf(((Q.mass_2photon - 3.9) / 1.375431) / math.sqrt(2)))) / 4.750888   # +0.1%  n_pairs_kt_above_3 < 28.8 and mass_2photon > 3.9
    )
    return z


def neuron_53(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.9e-06
    )
    return z


def neuron_54(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.19e-05
    )
    return z


def neuron_55(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.07e-05
    )
    return z


def neuron_56(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.13e-07
    )
    return z


def neuron_57(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.95e-06
    )
    return z


def neuron_58(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.61e-06
    )
    return z


def neuron_59(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.03e-06
    )
    return z


def neuron_60(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.67e-06
    )
    return z


def neuron_61(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.84e-06
    )
    return z


def neuron_62(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.41e-07
    )
    return z


def neuron_63(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.31e-06
    )
    return z


def neuron_64(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.25e-06
    )
    return z


def neuron_65(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.79e-06
    )
    return z


def neuron_66(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.71e-06
    )
    return z


def neuron_67(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.23e-06
    )
    return z


def neuron_68(Q):
    # scale S = 11.07; each line: share * term / its average size
    z = 11.06845 * (0.1029955
        - 0.06311583 * (11.90664 * 0.5 * ((79.9 - Q.sj3_pair_mass_min) / 11.90664) * (1 + math.erf(((79.9 - Q.sj3_pair_mass_min) / 11.90664) / math.sqrt(2)))) / 43.93675   # -6.3%  sj3_pair_mass_min < 79.9
        - 0.06305644 * (19.14168 * 0.5 * ((158.0 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.0 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2)))) / 54.95566   # -6.3%  mres_sd_mass_b2z01 < 158
        + 0.05366177 * (7.0 * 0.5 * ((64.9 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((64.9 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) / 26.99784   # +5.4%  n_pt_above_1 < 64.9
        + 0.05048408 * (18.57486 * 0.5 * ((Q.mres_pruned_mass - 41.6) / 18.57486) * (1 + math.erf(((Q.mres_pruned_mass - 41.6) / 18.57486) / math.sqrt(2)))) / 56.38551   # +5.0%  mres_pruned_mass > 41.6
        - 0.05006901 * (16.26099 * 0.5 * ((161.0 - Q.mass_top50) / 16.26099) * (1 + math.erf(((161.0 - Q.mass_top50) / 16.26099) / math.sqrt(2)))) / 50.38057   # -5.0%  mass_top50 < 161
        - 0.04124111 * (5.270719 * 0.5 * ((Q.mres_pruned_mass - 86.1) / 5.270719) * (1 + math.erf(((Q.mres_pruned_mass - 86.1) / 5.270719) / math.sqrt(2)))) / 22.59778   # -4.1%  mres_pruned_mass > 86.1
        + 0.03956531 * (3.499256 * 0.5 * ((122.0 - Q.mres_sd_mass_b2z01) / 3.499256) * (1 + math.erf(((122.0 - Q.mres_sd_mass_b2z01) / 3.499256) / math.sqrt(2)))) / 25.16819   # +4.0%  mres_sd_mass_b2z01 < 122
        - 0.03662188 * (0.0138316 * 0.5 * ((0.251 - Q.pz_lnd2) / 0.0138316) * (1 + math.erf(((0.251 - Q.pz_lnd2) / 0.0138316) / math.sqrt(2)))) / 0.129504   # -3.7%  pz_lnd2 < 0.251
        - 0.03328777 * (0.6671377 * 0.5 * ((0.438 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.438 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2260392   # -3.3%  lep_iso < 0.438
        + 0.03181418 * (12.47875 * 0.5 * ((Q.mres_pruned_mass - 131.0) / 12.47875) * (1 + math.erf(((Q.mres_pruned_mass - 131.0) / 12.47875) / math.sqrt(2)))) / 5.948203   # +3.2%  mres_pruned_mass > 131
        + 0.03125822 * (4.590131 * 0.5 * ((107.0 - Q.mres_sd_mass_b2z01) / 4.590131) * (1 + math.erf(((107.0 - Q.mres_sd_mass_b2z01) / 4.590131) / math.sqrt(2)))) / 16.24319   # +3.1%  mres_sd_mass_b2z01 < 107
        - 0.03059536 * (6.036668 * 0.5 * ((76.2 - Q.mres_sd_mass_b2z01) / 6.036668) * (1 + math.erf(((76.2 - Q.mres_sd_mass_b2z01) / 6.036668) / math.sqrt(2)))) / 5.418291   # -3.1%  mres_sd_mass_b2z01 < 76.2
        + 0.02968547 * (0.0003783019 * 0.5 * ((0.000807 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.000807 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) / 0.0006611107   # +3.0%  e3_b2 < 0.000807
        - 0.02519224 * (0.02630693 * 0.5 * ((0.237 - Q.pz_lnd3) / 0.02630693) * (1 + math.erf(((0.237 - Q.pz_lnd3) / 0.02630693) / math.sqrt(2)))) / 0.1437314   # -2.5%  pz_lnd3 < 0.237
        - 0.02310778 * (20.04228 * 0.5 * ((Q.mres_pruned_mass - 149.0) / 20.04228) * (1 + math.erf(((Q.mres_pruned_mass - 149.0) / 20.04228) / math.sqrt(2)))) / 3.898891   # -2.3%  mres_pruned_mass > 149
        + 0.02118514 * (0.03429006 * 0.5 * ((0.0802 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.0802 - Q.lep_dr) / 0.03429006) / math.sqrt(2)))) / 0.04756321   # +2.1%  lep_dr < 0.0802
        - 0.01899294 * (0.2410611 * 0.5 * ((3.38 - Q.pair_mean_lnm2) / 0.2410611) * (1 + math.erf(((3.38 - Q.pair_mean_lnm2) / 0.2410611) / math.sqrt(2)))) / 0.9963144   # -1.9%  pair_mean_lnm2 < 3.38
        - 0.0176558 * (0.0002686389 * 0.5 * ((0.000403 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000403 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.03798057 * 0.5 * ((Q.z_top30_slots - 0.83) / 0.03798057) * (1 + math.erf(((Q.z_top30_slots - 0.83) / 0.03798057) / math.sqrt(2)))) / 3.632386e-05   # -1.8%  e3_b2 < 0.000403 and z_top30_slots > 0.83
        + 0.01682114 * (3.317894 * 0.5 * ((117.0 - Q.mass) / 3.317894) * (1 + math.erf(((117.0 - Q.mass) / 3.317894) / math.sqrt(2)))) / 15.3871   # +1.7%  mass < 117
        - 0.0163903 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.9) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.9) / 4.308625) / math.sqrt(2)))) / 5.431594   # -1.6%  lep_ptrel > 6.9
        + 0.01620733 * (7.585662 * 0.5 * ((Q.lep_ptrel - 18.7) / 7.585662) * (1 + math.erf(((Q.lep_ptrel - 18.7) / 7.585662) / math.sqrt(2)))) / 2.989833   # +1.6%  lep_ptrel > 18.7
        + 0.01614602 * (0.2410611 * 0.5 * ((3.38 - Q.pair_mean_lnm2) / 0.2410611) * (1 + math.erf(((3.38 - Q.pair_mean_lnm2) / 0.2410611) / math.sqrt(2)))) * (0.06955833 * 0.5 * ((0.378 - Q.z_neutral_had) / 0.06955833) * (1 + math.erf(((0.378 - Q.z_neutral_had) / 0.06955833) / math.sqrt(2)))) / 0.2468389   # +1.6%  pair_mean_lnm2 < 3.38 and z_neutral_had < 0.378
        - 0.01534308 * (0.298047 * 0.5 * ((2.29 - Q.D2_b2) / 0.298047) * (1 + math.erf(((2.29 - Q.D2_b2) / 0.298047) / math.sqrt(2)))) / 0.725744   # -1.5%  D2_b2 < 2.29
        - 0.01461397 * (2.0 * 0.5 * ((11.1 - Q.n_sd0_above_2) / 2.0) * (1 + math.erf(((11.1 - Q.n_sd0_above_2) / 2.0) / math.sqrt(2)))) * (0.117671 * 0.5 * ((0.414 - Q.lep_dr) / 0.117671) * (1 + math.erf(((0.414 - Q.lep_dr) / 0.117671) / math.sqrt(2)))) / 2.361372   # -1.5%  n_sd0_above_2 < 11.1 and lep_dr < 0.414
        + 0.01448648 * (0.0216761 * 0.5 * ((0.819 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.819 - Q.tau32) / 0.0216761) / math.sqrt(2)))) / 0.1631158   # +1.4%  tau32 < 0.819
        + 0.01251523 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.998 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((0.998 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2)))) / 18.03701   # +1.3%  n_pt_above_1 < 65 and ak02_2_n_lep < 0.998
        + 0.01201522 * (0.04853964 * 0.5 * ((0.19 - Q.C2) / 0.04853964) * (1 + math.erf(((0.19 - Q.C2) / 0.04853964) / math.sqrt(2)))) / 0.05134741   # +1.2%  C2 < 0.19
        - 0.01181888 * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 19.9) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 19.9) / 1.0) / math.sqrt(2)))) / 2.575131   # -1.2%  n_charged_pt_above_1 > 19.9
        - 0.01156423 * (0.02099671 * 0.5 * ((0.0829 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.0829 - Q.z_displaced3) / 0.02099671) / math.sqrt(2)))) / 0.03720874   # -1.2%  z_displaced3 < 0.0829
        + 0.01125575 * (19.93276 * 0.5 * ((55.3 - Q.mres_sd_mass_b2z01) / 19.93276) * (1 + math.erf(((55.3 - Q.mres_sd_mass_b2z01) / 19.93276) / math.sqrt(2)))) / 3.016553   # +1.1%  mres_sd_mass_b2z01 < 55.3
        - 0.01055173 * (2.973269 * 0.5 * ((Q.sj4_pair_mass_max - 72.8) / 2.973269) * (1 + math.erf(((Q.sj4_pair_mass_max - 72.8) / 2.973269) / math.sqrt(2)))) / 15.08931   # -1.1%  sj4_pair_mass_max > 72.8
        + 0.01033701 * (0.01332924 * 0.5 * ((0.907 - Q.tau43) / 0.01332924) * (1 + math.erf(((0.907 - Q.tau43) / 0.01332924) / math.sqrt(2)))) / 0.1132818   # +1.0%  tau43 < 0.907
        - 0.009876813 * (0.1659628 * 0.5 * ((1.46 - Q.jet_abs_eta) / 0.1659628) * (1 + math.erf(((1.46 - Q.jet_abs_eta) / 0.1659628) / math.sqrt(2)))) / 0.7386554   # -1.0%  jet_abs_eta < 1.46
        + 0.008154601 * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.267) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.267) / 0.0649922) / math.sqrt(2)))) / 0.2479637   # +0.8%  z_charged_had > 0.267
        - 0.007922892 * (0.0218375 * 0.5 * ((0.0762 - Q.ak02_3_z) / 0.0218375) * (1 + math.erf(((0.0762 - Q.ak02_3_z) / 0.0218375) / math.sqrt(2)))) * (1032.513 * 0.5 * ((1160.0 - Q.lepsj_2_maxsd0) / 1032.513) * (1 + math.erf(((1160.0 - Q.lepsj_2_maxsd0) / 1032.513) / math.sqrt(2)))) / 47.14738   # -0.8%  ak02_3_z < 0.0762 and lepsj_2_maxsd0 < 1160
        + 0.007698881 * (0.0216761 * 0.5 * ((0.819 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.819 - Q.tau32) / 0.0216761) / math.sqrt(2)))) * (11.90664 * 0.5 * ((80.2 - Q.sj3_pair_mass_min) / 11.90664) * (1 + math.erf(((80.2 - Q.sj3_pair_mass_min) / 11.90664) / math.sqrt(2)))) / 5.393333   # +0.8%  tau32 < 0.819 and sj3_pair_mass_min < 80.2
        - 0.007613956 * (5.061224 * 0.5 * ((90.4 - Q.mass) / 5.061224) * (1 + math.erf(((90.4 - Q.mass) / 5.061224) / math.sqrt(2)))) * (5.001896 * 0.5 * ((2.43 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.43 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) / 7.141922   # -0.8%  mass < 90.4 and lepsj_3_maxsd0 < 2.43
        + 0.007572224 * (0.0138316 * 0.5 * ((0.25 - Q.pz_lnd2) / 0.0138316) * (1 + math.erf(((0.25 - Q.pz_lnd2) / 0.0138316) / math.sqrt(2)))) * (115.0698 * 0.5 * ((170.0 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((170.0 - Q.jd_3d_4) / 115.0698) / math.sqrt(2)))) / 17.7194   # +0.8%  pz_lnd2 < 0.25 and jd_3d_4 < 170
        + 0.006745178 * (7.0 * 0.5 * ((65.1 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.1 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.997 - Q.n_muon) / 1.0) * (1 + math.erf(((0.997 - Q.n_muon) / 1.0) / math.sqrt(2)))) / 17.40295   # +0.7%  n_pt_above_1 < 65.1 and n_muon < 0.997
        + 0.006198055 * (1.369872 * 0.5 * ((Q.sj3_mass1 - 17.3) / 1.369872) * (1 + math.erf(((Q.sj3_mass1 - 17.3) / 1.369872) / math.sqrt(2)))) / 6.017794   # +0.6%  sj3_mass1 > 17.3
        + 0.006046584 * (0.007115881 * 0.5 * ((Q.sjf_4_1_z_d3 - 0.00808) / 0.007115881) * (1 + math.erf(((Q.sjf_4_1_z_d3 - 0.00808) / 0.007115881) / math.sqrt(2)))) / 0.06373934   # +0.6%  sjf_4_1_z_d3 > 0.00808
        + 0.005352085 * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.266) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.266) / 0.0649922) / math.sqrt(2)))) * (0.01039253 * 0.5 * ((0.922 - Q.tau54) / 0.01039253) * (1 + math.erf(((0.922 - Q.tau54) / 0.01039253) / math.sqrt(2)))) / 0.01828373   # +0.5%  z_charged_had > 0.266 and tau54 < 0.922
        - 0.005231341 * (5.270719 * 0.5 * ((Q.mres_pruned_mass - 86.4) / 5.270719) * (1 + math.erf(((Q.mres_pruned_mass - 86.4) / 5.270719) / math.sqrt(2)))) * (0.02798345 * 0.5 * ((0.172 - Q.dc_3_z) / 0.02798345) * (1 + math.erf(((0.172 - Q.dc_3_z) / 0.02798345) / math.sqrt(2)))) / 1.996649   # -0.5%  mres_pruned_mass > 86.4 and dc_3_z < 0.172
        + 0.005086125 * (5.061224 * 0.5 * ((90.3 - Q.mass) / 5.061224) * (1 + math.erf(((90.3 - Q.mass) / 5.061224) / math.sqrt(2)))) / 5.026384   # +0.5%  mass < 90.3
        + 0.004566408 * (21.5 * 0.5 * ((79.2 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((79.2 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 5.074603   # +0.5%  n_pairs_kt_above_1 < 79.2
        + 0.004346585 * (3.317894 * 0.5 * ((117.0 - Q.mass) / 3.317894) * (1 + math.erf(((117.0 - Q.mass) / 3.317894) / math.sqrt(2)))) * (0.3667324 * 0.5 * ((2.59 - Q.D2_b2) / 0.3667324) * (1 + math.erf(((2.59 - Q.D2_b2) / 0.3667324) / math.sqrt(2)))) / 9.719182   # +0.4%  mass < 117 and D2_b2 < 2.59
        + 0.004332664 * (0.1783594 * 0.5 * ((Q.ktd_ln_d34 - -9.17) / 0.1783594) * (1 + math.erf(((Q.ktd_ln_d34 - -9.17) / 0.1783594) / math.sqrt(2)))) / 0.4655909   # +0.4%  ktd_ln_d34 > -9.17
        + 0.003821026 * (22.10989 * 0.5 * ((Q.mres_pruned_mass - 171.0) / 22.10989) * (1 + math.erf(((Q.mres_pruned_mass - 171.0) / 22.10989) / math.sqrt(2)))) / 2.013944   # +0.4%  mres_pruned_mass > 171
        - 0.003340299 * (0.02099671 * 0.5 * ((0.0826 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.0826 - Q.z_displaced3) / 0.02099671) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.00259 - Q.n_muon) / 1.0) * (1 + math.erf(((0.00259 - Q.n_muon) / 1.0) / math.sqrt(2)))) / 0.0008234282   # -0.3%  z_displaced3 < 0.0826 and n_muon < 0.00259
        + 0.003237492 * (0.05861841 * 0.5 * ((Q.ak02_dr23 - 0.369) / 0.05861841) * (1 + math.erf(((Q.ak02_dr23 - 0.369) / 0.05861841) / math.sqrt(2)))) / 0.0729817   # +0.3%  ak02_dr23 > 0.369
        - 0.003080653 * (0.04853964 * 0.5 * ((Q.C2 - 0.193) / 0.04853964) * (1 + math.erf(((Q.C2 - 0.193) / 0.04853964) / math.sqrt(2)))) / 0.01363922   # -0.3%  C2 > 0.193
        - 0.003033418 * (1.0 * 0.5 * ((Q.sdb_2_n - 11.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 11.0) / 1.0) / math.sqrt(2)))) * (1.739103 * 0.5 * ((4.02 - Q.jd_3d_4) / 1.739103) * (1 + math.erf(((4.02 - Q.jd_3d_4) / 1.739103) / math.sqrt(2)))) / 1.844793   # -0.3%  sdb_2_n > 11 and jd_3d_4 < 4.02
        + 0.002925416 * (0.2297361 * 0.5 * ((-5.68e-05 - Q.ak02_dr13) / 0.2297361) * (1 + math.erf(((-5.68e-05 - Q.ak02_dr13) / 0.2297361) / math.sqrt(2)))) / 0.008727713   # +0.3%  ak02_dr13 < -5.68e-05
        + 0.002783373 * (11.17405 * 0.5 * ((Q.sj3_pair_mass_max - 129.0) / 11.17405) * (1 + math.erf(((Q.sj3_pair_mass_max - 129.0) / 11.17405) / math.sqrt(2)))) / 2.567302   # +0.3%  sj3_pair_mass_max > 129
        + 0.002779597 * (0.001201988 * 0.5 * ((0.00282 - Q.sum_z_dr2_top3) / 0.001201988) * (1 + math.erf(((0.00282 - Q.sum_z_dr2_top3) / 0.001201988) / math.sqrt(2)))) / 0.0002366602   # +0.3%  sum_z_dr2_top3 < 0.00282
        - 0.002757226 * (0.03746729 * 0.5 * ((Q.z_displaced3 - 0.163) / 0.03746729) * (1 + math.erf(((Q.z_displaced3 - 0.163) / 0.03746729) / math.sqrt(2)))) / 0.03999766   # -0.3%  z_displaced3 > 0.163
        - 0.002549999 * (0.02719315 * 0.5 * ((0.154 - Q.sdb_2_z) / 0.02719315) * (1 + math.erf(((0.154 - Q.sdb_2_z) / 0.02719315) / math.sqrt(2)))) / 0.01306691   # -0.3%  sdb_2_z < 0.154
        + 0.00248873 * (0.0002686389 * 0.5 * ((0.000437 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000437 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 2.04) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 2.04) / 1.0) / math.sqrt(2)))) / 0.000112895   # +0.2%  e3_b2 < 0.000437 and n_lund_kt_above_5 > 2.04
        - 0.002431304 * (1.0 * 0.5 * ((2.98 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((2.98 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) / 0.2424393   # -0.2%  sjq_2_2_nch < 2.98
        - 0.002418089 * (5.345127 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 77.4) / 5.345127) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 77.4) / 5.345127) / math.sqrt(2)))) / 2.433135   # -0.2%  nca_sj4_pair_mass_2nd > 77.4
        - 0.002260217 * (10.8986 * 0.5 * ((67.3 - Q.mass_top30) / 10.8986) * (1 + math.erf(((67.3 - Q.mass_top30) / 10.8986) / math.sqrt(2)))) / 2.001368   # -0.2%  mass_top30 < 67.3
        - 0.002092689 * (0.179073 * 0.5 * ((Q.lep_z - 0.517) / 0.179073) * (1 + math.erf(((Q.lep_z - 0.517) / 0.179073) / math.sqrt(2)))) / 0.01015913   # -0.2%  lep_z > 0.517
        + 0.001632667 * (1.504711 * 0.5 * ((Q.sj3_mass2 - 15.3) / 1.504711) * (1 + math.erf(((Q.sj3_mass2 - 15.3) / 1.504711) / math.sqrt(2)))) / 1.849651   # +0.2%  sj3_mass2 > 15.3
        + 0.001450868 * (0.0216761 * 0.5 * ((0.817 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.817 - Q.tau32) / 0.0216761) / math.sqrt(2)))) * (0.01950041 * 0.5 * ((Q.sv_1_dr - 0.132) / 0.01950041) * (1 + math.erf(((Q.sv_1_dr - 0.132) / 0.01950041) / math.sqrt(2)))) / 0.01372552   # +0.1%  tau32 < 0.817 and sv_1_dr > 0.132
        - 0.001387389 * (0.02593915 * 0.5 * ((Q.mres_sd_rg_b1z01 - 0.471) / 0.02593915) * (1 + math.erf(((Q.mres_sd_rg_b1z01 - 0.471) / 0.02593915) / math.sqrt(2)))) / 0.02505096   # -0.1%  mres_sd_rg_b1z01 > 0.471
        - 0.001378924 * (0.02434011 * 0.5 * ((Q.sj3_pairmin_over_m - 0.461) / 0.02434011) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.461) / 0.02434011) / math.sqrt(2)))) / 0.007065993   # -0.1%  sj3_pairmin_over_m > 0.461
        - 0.001353927 * (10.67504 * 0.5 * ((Q.mass_top10 - 103.0) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 103.0) / 10.67504) / math.sqrt(2)))) / 2.067017   # -0.1%  mass_top10 > 103
        - 0.001192899 * (0.00254714 * 0.5 * ((0.0388 - Q.M3) / 0.00254714) * (1 + math.erf(((0.0388 - Q.M3) / 0.00254714) / math.sqrt(2)))) * (0.06042313 * 0.5 * ((Q.sjq_3_sumabs_k1 - 0.439) / 0.06042313) * (1 + math.erf(((Q.sjq_3_sumabs_k1 - 0.439) / 0.06042313) / math.sqrt(2)))) / 0.0006056668   # -0.1%  M3 < 0.0388 and sjq_3_sumabs_k1 > 0.439
        - 0.000930796 * (0.07842787 * 0.5 * ((Q.dc_split2_dr - 0.362) / 0.07842787) * (1 + math.erf(((Q.dc_split2_dr - 0.362) / 0.07842787) / math.sqrt(2)))) / 0.004905937   # -0.1%  dc_split2_dr > 0.362
        + 0.0008888698 * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 19.9) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 19.9) / 1.0) / math.sqrt(2)))) * (1.5 * 0.5 * ((17.1 - Q.n_dr_0p1_0p2) / 1.5) * (1 + math.erf(((17.1 - Q.n_dr_0p1_0p2) / 1.5) / math.sqrt(2)))) / 7.626674   # +0.1%  n_charged_pt_above_1 > 19.9 and n_dr_0p1_0p2 < 17.1
        + 0.0003830644 * (12.47875 * 0.5 * ((Q.mres_pruned_mass - 131.0) / 12.47875) * (1 + math.erf(((Q.mres_pruned_mass - 131.0) / 12.47875) / math.sqrt(2)))) * (0.002064355 * 0.5 * ((1.6e-05 - Q.lepsj_3_dr) / 0.002064355) * (1 + math.erf(((1.6e-05 - Q.lepsj_3_dr) / 0.002064355) / math.sqrt(2)))) / 9.506565e-05   # +0.0%  mres_pruned_mass > 131 and lepsj_3_dr < 1.6e-05
    )
    return z


def neuron_69(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.92e-06
    )
    return z


def neuron_70(Q):
    # scale S = 2.503; each line: share * term / its average size
    z = 2.502737 * (0.004475101
        - 0.1022099 * (0.03060878 * 0.5 * ((0.0616 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0616 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) / 0.04503599   # -10.2%  lepsj_2_dr < 0.0616
        - 0.0914569 * (3.563419 * 0.5 * ((Q.mres_sd_mass_b0z005 - 122.0) / 3.563419) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 122.0) / 3.563419) / math.sqrt(2)))) / 9.458369   # -9.1%  mres_sd_mass_b0z005 > 122
        + 0.08945688 * (1.0 * 0.5 * ((1.68 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((1.68 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.0330605 * 0.5 * ((0.176 - Q.lep_dr) / 0.0330605) * (1 + math.erf(((0.176 - Q.lep_dr) / 0.0330605) / math.sqrt(2)))) / 0.1776881   # +8.9%  lepsj_3_n_d3 < 1.68 and lep_dr < 0.176
        - 0.07483203 * (0.0134569 * 0.5 * ((Q.C2 - 0.066) / 0.0134569) * (1 + math.erf(((Q.C2 - 0.066) / 0.0134569) / math.sqrt(2)))) / 0.08436256   # -7.5%  C2 > 0.066
        + 0.06963562 * (11.41538 * 0.5 * ((Q.mres_sd_mass_b0z005 - 79.6) / 11.41538) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 79.6) / 11.41538) / math.sqrt(2)))) / 32.88295   # +7.0%  mres_sd_mass_b0z005 > 79.6
        + 0.06248055 * (0.009408518 * 0.5 * ((0.111 - Q.pz_lnd0) / 0.009408518) * (1 + math.erf(((0.111 - Q.pz_lnd0) / 0.009408518) / math.sqrt(2)))) / 0.02237087   # +6.2%  pz_lnd0 < 0.111
        + 0.05869575 * (0.0134569 * 0.5 * ((Q.C2 - 0.0645) / 0.0134569) * (1 + math.erf(((Q.C2 - 0.0645) / 0.0134569) / math.sqrt(2)))) * (0.0329306 * 0.5 * ((0.434 - Q.ak02_2_z) / 0.0329306) * (1 + math.erf(((0.434 - Q.ak02_2_z) / 0.0329306) / math.sqrt(2)))) / 0.02037448   # +5.9%  C2 > 0.0645 and ak02_2_z < 0.434
        + 0.05694339 * (1.0 * 0.5 * ((18.6 - Q.n_neutral) / 1.0) * (1 + math.erf(((18.6 - Q.n_neutral) / 1.0) / math.sqrt(2)))) / 2.735399   # +5.7%  n_neutral < 18.6
        + 0.05376174 * (3.764443 * 0.5 * ((Q.mass_top50 - 127.0) / 3.764443) * (1 + math.erf(((Q.mass_top50 - 127.0) / 3.764443) / math.sqrt(2)))) / 8.570159   # +5.4%  mass_top50 > 127
        - 0.04772197 * (1.0 * 0.5 * ((18.9 - Q.n_neutral) / 1.0) * (1 + math.erf(((18.9 - Q.n_neutral) / 1.0) / math.sqrt(2)))) * (3.102391 * 0.5 * ((7.31 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((7.31 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 14.00182   # -4.8%  n_neutral < 18.9 and jd_3d_6 < 7.31
        + 0.03534021 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 162.0) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 162.0) / 18.66842) / math.sqrt(2)))) / 3.15883   # +3.5%  mres_sd_mass_b0z005 > 162
        - 0.03447447 * (3.974361 * 0.5 * ((Q.sj3_pair_mass_min - 40.4) / 3.974361) * (1 + math.erf(((Q.sj3_pair_mass_min - 40.4) / 3.974361) / math.sqrt(2)))) / 7.250464   # -3.4%  sj3_pair_mass_min > 40.4
        - 0.02854116 * (0.1486471 * 0.5 * ((0.233 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.233 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 0.189) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 0.189) / 1.0) / math.sqrt(2)))) / 0.2012141   # -2.9%  lep_z < 0.233 and sjf_2_2_n_d3 > 0.189
        - 0.02241409 * (0.02343699 * 0.5 * ((0.221 - Q.sdb_2_z) / 0.02343699) * (1 + math.erf(((0.221 - Q.sdb_2_z) / 0.02343699) / math.sqrt(2)))) / 0.03099258   # -2.2%  sdb_2_z < 0.221
        + 0.02053558 * (1.0 * 0.5 * ((Q.sjf_4_n2disp - 1.11) / 1.0) * (1 + math.erf(((Q.sjf_4_n2disp - 1.11) / 1.0) / math.sqrt(2)))) / 0.2705008   # +2.1%  sjf_4_n2disp > 1.11
        - 0.01699313 * (19.33798 * 0.5 * ((Q.mres_sd_mass_b2z01 - 175.0) / 19.33798) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 175.0) / 19.33798) / math.sqrt(2)))) / 2.06453   # -1.7%  mres_sd_mass_b2z01 > 175
        - 0.01607532 * (3.329534 * 0.5 * ((Q.mass_2charged - 14.7) / 3.329534) * (1 + math.erf(((Q.mass_2charged - 14.7) / 3.329534) / math.sqrt(2)))) / 5.556946   # -1.6%  mass_2charged > 14.7
        + 0.01598013 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.64) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.64) / 0.2498329) / math.sqrt(2)))) / 0.01904479   # +1.6%  sjq_3_3_k1 > 0.64
        - 0.01591844 * (0.009408518 * 0.5 * ((0.115 - Q.pz_lnd0) / 0.009408518) * (1 + math.erf(((0.115 - Q.pz_lnd0) / 0.009408518) / math.sqrt(2)))) * (0.01282622 * 0.5 * ((Q.z_neutral_had - 0.12) / 0.01282622) * (1 + math.erf(((Q.z_neutral_had - 0.12) / 0.01282622) / math.sqrt(2)))) / 0.001861666   # -1.6%  pz_lnd0 < 0.115 and z_neutral_had > 0.12
        + 0.01468062 * (0.1486471 * 0.5 * ((0.452 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.452 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.03538975 * 0.5 * ((0.591 - Q.tau32) / 0.03538975) * (1 + math.erf(((0.591 - Q.tau32) / 0.03538975) / math.sqrt(2)))) / 0.01524553   # +1.5%  lep_z < 0.452 and tau32 < 0.591
        + 0.01357906 * (0.2334666 * 0.5 * ((-0.59 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.59 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2)))) / 0.02072244   # +1.4%  sjq_3_2_k1 < -0.59
        + 0.01264148 * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 1.71) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 1.71) / 1.0) / math.sqrt(2)))) / 0.3409299   # +1.3%  lepsj_2_n_d3 > 1.71
        - 0.01180053 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.0926) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.0926) / 0.02108391) / math.sqrt(2)))) * (0.01372201 * 0.5 * ((Q.pz_lnkt1 - 0.0651) / 0.01372201) * (1 + math.erf(((Q.pz_lnkt1 - 0.0651) / 0.01372201) / math.sqrt(2)))) / 0.001757953   # -1.2%  z_displaced5 > 0.0926 and pz_lnkt1 > 0.0651
        - 0.01119811 * (1.234622 * 0.5 * ((1.79 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.79 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.388) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.388) / 0.179816) / math.sqrt(2)))) / 0.04601957   # -1.1%  lep_iso < 1.79 and sjq_3_3_k1 > 0.388
        + 0.009223819 * (8.963785 * 0.5 * ((23.6 - Q.mass_charged) / 8.963785) * (1 + math.erf(((23.6 - Q.mass_charged) / 8.963785) / math.sqrt(2)))) / 0.5685909   # +0.9%  mass_charged < 23.6
        + 0.008332914 * (0.2508028 * 0.5 * ((-0.673 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.673 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2)))) / 0.01642133   # +0.8%  sjq_3_3_k1 < -0.673
        - 0.005076238 * (0.04964267 * 0.5 * ((Q.sj4_dr_min - 0.261) / 0.04964267) * (1 + math.erf(((Q.sj4_dr_min - 0.261) / 0.04964267) / math.sqrt(2)))) / 0.007746639   # -0.5%  sj4_dr_min > 0.261
    )
    return z


def neuron_71(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.03e-06
    )
    return z


def neuron_72(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.16e-06
    )
    return z


def neuron_73(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.7e-05
    )
    return z


def neuron_74(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.49e-06
    )
    return z


def neuron_75(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.97e-06
    )
    return z


def neuron_76(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.54e-06
    )
    return z


def neuron_77(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.09e-06
    )
    return z


def neuron_78(Q):
    # scale S = 3.03; each line: share * term / its average size
    z = 3.029907 * (-0.0336644
        - 0.3029438 * (7.585662 * 0.5 * ((17.8 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((17.8 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) / 13.28353   # -30.3%  lep_ptrel < 17.8
        + 0.08497914 * (1.0 * 0.5 * ((5.73 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((5.73 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 2.657161   # +8.5%  n_s3d_above_3 < 5.73
        + 0.05795921 * (0.009441413 * 0.5 * ((0.488 - Q.N2_b05) / 0.009441413) * (1 + math.erf(((0.488 - Q.N2_b05) / 0.009441413) / math.sqrt(2)))) / 0.05973164   # +5.8%  N2_b05 < 0.488
        - 0.0534621 * (0.2260337 * 0.5 * ((-8.43 - Q.ktd_ln_d34) / 0.2260337) * (1 + math.erf(((-8.43 - Q.ktd_ln_d34) / 0.2260337) / math.sqrt(2)))) / 1.117139   # -5.3%  ktd_ln_d34 < -8.43
        + 0.05042947 * (7.585662 * 0.5 * ((20.5 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((20.5 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.3051516 * 0.5 * ((-7.89 - Q.ktd_ln_d34) / 0.3051516) * (1 + math.erf(((-7.89 - Q.ktd_ln_d34) / 0.3051516) / math.sqrt(2)))) / 22.87374   # +5.0%  lep_ptrel < 20.5 and ktd_ln_d34 < -7.89
        + 0.04587071 * (0.05826336 * 0.5 * ((0.102 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.102 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) * (0.2429919 * 0.5 * ((6.3 - Q.lne_0) / 0.2429919) * (1 + math.erf(((6.3 - Q.lne_0) / 0.2429919) / math.sqrt(2)))) / 0.08526626   # +4.6%  lepsj_2_dr < 0.102 and lne_0 < 6.3
        + 0.04228635 * (11.5 * 0.5 * ((99.0 - Q.n_pairs_kt_above_3) / 11.5) * (1 + math.erf(((99.0 - Q.n_pairs_kt_above_3) / 11.5) / math.sqrt(2)))) / 39.18157   # +4.2%  n_pairs_kt_above_3 < 99
        + 0.03760374 * (0.3893833 * 0.5 * ((Q.lund3_lndelta - -2.76) / 0.3893833) * (1 + math.erf(((Q.lund3_lndelta - -2.76) / 0.3893833) / math.sqrt(2)))) / 1.264549   # +3.8%  lund3_lndelta > -2.76
        + 0.0370839 * (2.54381e-06 * 0.5 * ((3.49e-06 - Q.e4) / 2.54381e-06) * (1 + math.erf(((3.49e-06 - Q.e4) / 2.54381e-06) / math.sqrt(2)))) / 2.416361e-06   # +3.7%  e4 < 3.49e-06
        - 0.03574161 * (5.898438 * 0.5 * ((10.2 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.2 - Q.max_abs_d0) / 5.898438) / math.sqrt(2)))) / 5.414689   # -3.6%  max_abs_d0 < 10.2
        + 0.02614034 * (0.04874922 * 0.5 * ((0.13 - Q.sjf_4_1_z_d3) / 0.04874922) * (1 + math.erf(((0.13 - Q.sjf_4_1_z_d3) / 0.04874922) / math.sqrt(2)))) / 0.09114247   # +2.6%  sjf_4_1_z_d3 < 0.13
        + 0.02583157 * (27.0 * 0.5 * ((201.0 - Q.n_pairs_kt_above_1) / 27.0) * (1 + math.erf(((201.0 - Q.n_pairs_kt_above_1) / 27.0) / math.sqrt(2)))) / 39.72958   # +2.6%  n_pairs_kt_above_1 < 201
        + 0.02323497 * (4.488781 * 0.5 * ((Q.mres_sd_mass_b0z005 - 125.0) / 4.488781) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 125.0) / 4.488781) / math.sqrt(2)))) / 8.638012   # +2.3%  mres_sd_mass_b0z005 > 125
        + 0.02322298 * (2.0 * 0.5 * ((11.0 - Q.n_sdz_above_2) / 2.0) * (1 + math.erf(((11.0 - Q.n_sdz_above_2) / 2.0) / math.sqrt(2)))) * (0.06215671 * 0.5 * ((0.309 - Q.sj4_dr_min) / 0.06215671) * (1 + math.erf(((0.309 - Q.sj4_dr_min) / 0.06215671) / math.sqrt(2)))) / 1.398877   # +2.3%  n_sdz_above_2 < 11 and sj4_dr_min < 0.309
        - 0.02292338 * (1.0 * 0.5 * ((10.1 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((10.1 - Q.sdb_2_n) / 1.0) / math.sqrt(2)))) / 1.872122   # -2.3%  sdb_2_n < 10.1
        - 0.02238395 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 163.0) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 163.0) / 18.66842) / math.sqrt(2)))) / 3.055013   # -2.2%  mres_sd_mass_b0z005 > 163
        - 0.01786822 * (5.197 * 0.5 * ((95.5 - Q.mass) / 5.197) * (1 + math.erf(((95.5 - Q.mass) / 5.197) / math.sqrt(2)))) / 6.475962   # -1.8%  mass < 95.5
        + 0.01503578 * (0.1486471 * 0.5 * ((0.299 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.299 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.01443111 * 0.5 * ((0.194 - Q.pz_lnd2) / 0.01443111) * (1 + math.erf(((0.194 - Q.pz_lnd2) / 0.01443111) / math.sqrt(2)))) / 0.01922237   # +1.5%  lep_z < 0.299 and pz_lnd2 < 0.194
        + 0.014429 * (1.5 * 0.5 * ((Q.n_charged_had - 22.7) / 1.5) * (1 + math.erf(((Q.n_charged_had - 22.7) / 1.5) / math.sqrt(2)))) / 1.607299   # +1.4%  n_charged_had > 22.7
        + 0.01168429 * (7.585662 * 0.5 * ((20.3 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((20.3 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (16.61023 * 0.5 * ((616.0 - Q.sum_pt_top30) / 16.61023) * (1 + math.erf(((616.0 - Q.sum_pt_top30) / 16.61023) / math.sqrt(2)))) / 878.469   # +1.2%  lep_ptrel < 20.3 and sum_pt_top30 < 616
        + 0.01004902 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.49) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.49) / 1.0) / math.sqrt(2)))) * (405.0217 * 0.5 * ((779.0 - Q.jd_sum_abs_sd0_top5) / 405.0217) * (1 + math.erf(((779.0 - Q.jd_sum_abs_sd0_top5) / 405.0217) / math.sqrt(2)))) / 255.8621   # +1.0%  n_s3d_above_3 > 4.49 and jd_sum_abs_sd0_top5 < 779
        + 0.009315367 * (17.59858 * 0.5 * ((Q.mass_top50 - 183.0) / 17.59858) * (1 + math.erf(((Q.mass_top50 - 183.0) / 17.59858) / math.sqrt(2)))) / 1.501314   # +0.9%  mass_top50 > 183
        - 0.008228311 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.27) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.27) / 1.0) / math.sqrt(2)))) * (0.02818185 * 0.5 * ((Q.sj3_pairmin_over_m - 0.0864) / 0.02818185) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.0864) / 0.02818185) / math.sqrt(2)))) / 0.2503114   # -0.8%  n_s3d_above_3 > 4.27 and sj3_pairmin_over_m > 0.0864
        - 0.007962799 * (3.793098 * 0.5 * ((Q.mass_displaced5 - 8.44) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 8.44) / 3.793098) / math.sqrt(2)))) * (14.29796 * 0.5 * ((109.0 - Q.mass_charged) / 14.29796) * (1 + math.erf(((109.0 - Q.mass_charged) / 14.29796) / math.sqrt(2)))) / 111.697   # -0.8%  mass_displaced5 > 8.44 and mass_charged < 109
        + 0.007408987 * (3.793098 * 0.5 * ((Q.mass_displaced5 - 8.86) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 8.86) / 3.793098) / math.sqrt(2)))) / 3.385904   # +0.7%  mass_displaced5 > 8.86
        + 0.003499471 * (12.30925 * 0.5 * ((Q.mass_displaced3 - 37.3) / 12.30925) * (1 + math.erf(((Q.mass_displaced3 - 37.3) / 12.30925) / math.sqrt(2)))) / 0.9639157   # +0.3%  mass_displaced3 > 37.3
        + 0.002421496 * (2.0 * 0.5 * ((Q.sjf_2_1_n_d3 - 6.96) / 2.0) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 6.96) / 2.0) / math.sqrt(2)))) / 0.1132239   # +0.2%  sjf_2_1_n_d3 > 6.96
    )
    return z


def neuron_79(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.6e-05
    )
    return z


def neuron_80(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.71e-06
    )
    return z


def neuron_81(Q):
    # scale S = 5.547; each line: share * term / its average size
    z = 5.547484 * (0.1501582
        - 0.1938516 * (19.14168 * 0.5 * ((157.0 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((157.0 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2)))) / 54.03964   # -19.4%  mres_sd_mass_b2z01 < 157
        + 0.1579294 * (3.499256 * 0.5 * ((121.0 - Q.mres_sd_mass_b2z01) / 3.499256) * (1 + math.erf(((121.0 - Q.mres_sd_mass_b2z01) / 3.499256) / math.sqrt(2)))) / 24.47237   # +15.8%  mres_sd_mass_b2z01 < 121
        - 0.1040436 * (1.0 * 0.5 * ((10.1 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.1 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 6.456156   # -10.4%  n_s3d_above_3 < 10.1
        + 0.07273828 * (1.0 * 0.5 * ((6.11 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((6.11 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 2.945361   # +7.3%  n_s3d_above_3 < 6.11
        + 0.05486317 * (3.705078 * 0.5 * ((6.2 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((6.2 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) / 3.013392   # +5.5%  max_abs_d0 < 6.2
        - 0.05354556 * (3.5 * 0.5 * ((14.5 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((14.5 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2)))) / 8.974113   # -5.4%  n_dr_0p4_up < 14.5
        + 0.04730918 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.0314) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.0314) / 0.01141967) / math.sqrt(2)))) / 0.09018794   # +4.7%  z_displaced3 > 0.0314
        - 0.03929417 * (29.76053 * 0.5 * ((44.4 - Q.lep_iso) / 29.76053) * (1 + math.erf(((44.4 - Q.lep_iso) / 29.76053) / math.sqrt(2)))) / 37.07207   # -3.9%  lep_iso < 44.4
        - 0.02951492 * (3.705078 * 0.5 * ((6.05 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((6.05 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (6.084951 * 0.5 * ((18.7 - Q.sip_3d_3) / 6.084951) * (1 + math.erf(((18.7 - Q.sip_3d_3) / 6.084951) / math.sqrt(2)))) / 37.38209   # -3.0%  max_abs_d0 < 6.05 and sip_3d_3 < 18.7
        + 0.02719173 * (0.009443246 * 0.5 * ((0.0156 - Q.lepsj_3_dr) / 0.009443246) * (1 + math.erf(((0.0156 - Q.lepsj_3_dr) / 0.009443246) / math.sqrt(2)))) / 0.01062293   # +2.7%  lepsj_3_dr < 0.0156
        - 0.02596385 * (5.26174 * 0.5 * ((91.1 - Q.mres_sd_mass_b2z01) / 5.26174) * (1 + math.erf(((91.1 - Q.mres_sd_mass_b2z01) / 5.26174) / math.sqrt(2)))) / 9.475925   # -2.6%  mres_sd_mass_b2z01 < 91.1
        - 0.02391279 * (0.2381965 * 0.5 * ((Q.lne_5 - 2.73) / 0.2381965) * (1 + math.erf(((Q.lne_5 - 2.73) / 0.2381965) / math.sqrt(2)))) / 0.850358   # -2.4%  lne_5 > 2.73
        - 0.02256142 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.0405) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.0405) / 0.01141967) / math.sqrt(2)))) * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.267) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.267) / 0.0649922) / math.sqrt(2)))) / 0.02388532   # -2.3%  z_displaced3 > 0.0405 and z_charged_had > 0.267
        - 0.02227814 * (0.5788858 * 0.5 * ((1.9 - Q.mass_displaced3) / 0.5788858) * (1 + math.erf(((1.9 - Q.mass_displaced3) / 0.5788858) / math.sqrt(2)))) / 0.7973396   # -2.2%  mass_displaced3 < 1.9
        + 0.01903079 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.21) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.21) / 1.0) / math.sqrt(2)))) * (0.006796583 * 0.5 * ((0.0634 - Q.M2_b2) / 0.006796583) * (1 + math.erf(((0.0634 - Q.M2_b2) / 0.006796583) / math.sqrt(2)))) / 0.06598313   # +1.9%  n_s3d_above_3 > 2.21 and M2_b2 < 0.0634
        + 0.01580575 * (1.0 * 0.5 * ((9.69 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((9.69 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.235 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.235 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2)))) / 0.6353781   # +1.6%  n_s3d_above_3 < 9.69 and z_neutral_had < 0.235
        + 0.01522554 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 3.46) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 3.46) / 1.0) / math.sqrt(2)))) / 0.7893779   # +1.5%  n_s3d_above_10 > 3.46
        - 0.01009783 * (1.0 * 0.5 * ((Q.sjf_4_n2disp - 0.94) / 1.0) * (1 + math.erf(((Q.sjf_4_n2disp - 0.94) / 1.0) / math.sqrt(2)))) / 0.3077888   # -1.0%  sjf_4_n2disp > 0.94
        + 0.007281828 * (0.02426199 * 0.5 * ((0.212 - Q.sdb_2_z) / 0.02426199) * (1 + math.erf(((0.212 - Q.sdb_2_z) / 0.02426199) / math.sqrt(2)))) / 0.02805266   # +0.7%  sdb_2_z < 0.212
        - 0.00717488 * (0.01141967 * 0.5 * ((Q.z_displaced3 - -0.00479) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - -0.00479) / 0.01141967) / math.sqrt(2)))) * (1.5 * 0.5 * ((6.84 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((6.84 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2)))) / 0.1708263   # -0.7%  z_displaced3 > -0.00479 and n_s3d_above_3 < 6.84
        - 0.007134027 * (0.01559439 * 0.5 * ((0.167 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.167 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.2297361 * 0.5 * ((Q.ak02_dr13 - 0.0205) / 0.2297361) * (1 + math.erf(((Q.ak02_dr13 - 0.0205) / 0.2297361) / math.sqrt(2)))) / 0.01383773   # -0.7%  pz_lnd0 < 0.167 and ak02_dr13 > 0.0205
        - 0.006490475 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.0323) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.0323) / 0.01141967) / math.sqrt(2)))) * (0.0964495 * 0.5 * ((0.572 - Q.sjq_2_sumabs_k1) / 0.0964495) * (1 + math.erf(((0.572 - Q.sjq_2_sumabs_k1) / 0.0964495) / math.sqrt(2)))) / 0.02466151   # -0.6%  z_displaced3 > 0.0323 and sjq_2_sumabs_k1 < 0.572
        - 0.005952199 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.0312) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.0312) / 0.01141967) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.01 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.01 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.05266304   # -0.6%  z_displaced3 > 0.0312 and ak02_2_n_lep < 1.01
        - 0.005940636 * (0.02426199 * 0.5 * ((0.209 - Q.sdb_2_z) / 0.02426199) * (1 + math.erf(((0.209 - Q.sdb_2_z) / 0.02426199) / math.sqrt(2)))) * (5.085197 * 0.5 * ((10.5 - Q.jd_3d_5) / 5.085197) * (1 + math.erf(((10.5 - Q.jd_3d_5) / 5.085197) / math.sqrt(2)))) / 0.1160408   # -0.6%  sdb_2_z < 0.209 and jd_3d_5 < 10.5
        - 0.005623892 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.0866) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.0866) / 0.02108391) / math.sqrt(2)))) / 0.05056475   # -0.6%  z_displaced5 > 0.0866
        - 0.005470943 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 3.13) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 3.13) / 1.0) / math.sqrt(2)))) * (2.0 * 0.5 * ((4.33 - Q.lepsj_2_n_d3) / 2.0) * (1 + math.erf(((4.33 - Q.lepsj_2_n_d3) / 2.0) / math.sqrt(2)))) / 2.508262   # -0.5%  n_s3d_above_10 > 3.13 and lepsj_2_n_d3 < 4.33
        + 0.004387334 * (0.04445521 * 0.5 * ((0.262 - Q.LHA) / 0.04445521) * (1 + math.erf(((0.262 - Q.LHA) / 0.04445521) / math.sqrt(2)))) / 0.006631788   # +0.4%  LHA < 0.262
        + 0.003601385 * (1.0 * 0.5 * ((4.38 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.38 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) * (0.006013287 * 0.5 * ((0.0622 - Q.pz_lnkt3) / 0.006013287) * (1 + math.erf(((0.0622 - Q.pz_lnkt3) / 0.006013287) / math.sqrt(2)))) / 0.02331228   # +0.4%  n_dr_0p4_up < 4.38 and pz_lnkt3 < 0.0622
        - 0.003384022 * (5.55459 * 0.5 * ((96.2 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.2 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) * (0.01467775 * 0.5 * ((0.223 - Q.N2_b2) / 0.01467775) * (1 + math.erf(((0.223 - Q.N2_b2) / 0.01467775) / math.sqrt(2)))) / 0.4788981   # -0.3%  mres_sd_mass_b2z01 < 96.2 and N2_b2 < 0.223
        - 0.00240059 * (17.59858 * 0.5 * ((Q.mass_top50 - 175.0) / 17.59858) * (1 + math.erf(((Q.mass_top50 - 175.0) / 17.59858) / math.sqrt(2)))) / 1.913396   # -0.2%  mass_top50 > 175
    )
    return z


def neuron_82(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.95e-06
    )
    return z


def neuron_83(Q):
    # scale S = 7.94; each line: share * term / its average size
    z = 7.940361 * (-0.1460891
        + 0.08633027 * (10.09897 * 0.5 * ((116.0 - Q.sj4_pair_mass_max) / 10.09897) * (1 + math.erf(((116.0 - Q.sj4_pair_mass_max) / 10.09897) / math.sqrt(2)))) / 36.85449   # +8.6%  sj4_pair_mass_max < 116
        + 0.08253794 * (16.90428 * 0.5 * ((165.0 - Q.mass) / 16.90428) * (1 + math.erf(((165.0 - Q.mass) / 16.90428) / math.sqrt(2)))) / 52.85331   # +8.3%  mass < 165
        - 0.06297959 * (0.00905862 * 0.5 * ((0.0653 - Q.tau5) / 0.00905862) * (1 + math.erf(((0.0653 - Q.tau5) / 0.00905862) / math.sqrt(2)))) / 0.03356246   # -6.3%  tau5 < 0.0653
        + 0.05412676 * (3.384707 * 0.5 * ((6.65 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.65 - Q.jd_3d_5) / 3.384707) / math.sqrt(2)))) / 3.114392   # +5.4%  jd_3d_5 < 6.65
        - 0.04712961 * (5.197 * 0.5 * ((Q.mass - 94.6) / 5.197) * (1 + math.erf(((Q.mass - 94.6) / 5.197) / math.sqrt(2)))) / 26.54086   # -4.7%  mass > 94.6
        + 0.04595897 * (10.14805 * 0.5 * ((26.4 - Q.mass_displaced3) / 10.14805) * (1 + math.erf(((26.4 - Q.mass_displaced3) / 10.14805) / math.sqrt(2)))) / 19.8332   # +4.6%  mass_displaced3 < 26.4
        - 0.04536816 * (2.0 * 0.5 * ((8.03 - Q.n_s3d_above_10) / 2.0) * (1 + math.erf(((8.03 - Q.n_s3d_above_10) / 2.0) / math.sqrt(2)))) / 5.458176   # -4.5%  n_s3d_above_10 < 8.03
        - 0.04455321 * (3.705078 * 0.5 * ((5.98 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.98 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (3.384707 * 0.5 * ((6.53 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.53 - Q.jd_3d_5) / 3.384707) / math.sqrt(2)))) / 10.59187   # -4.5%  max_abs_d0 < 5.98 and jd_3d_5 < 6.53
        + 0.0405005 * (1.5 * 0.5 * ((9.98 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((9.98 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2)))) / 5.186913   # +4.1%  n_dr_0p4_up < 9.98
        + 0.04006929 * (3.705078 * 0.5 * ((5.75 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.75 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) / 2.719356   # +4.0%  max_abs_d0 < 5.75
        + 0.03236091 * (1.951776e-05 * 0.5 * ((Q.ecf_g41 - 8.42e-05) / 1.951776e-05) * (1 + math.erf(((Q.ecf_g41 - 8.42e-05) / 1.951776e-05) / math.sqrt(2)))) / 0.0001304352   # +3.2%  ecf_g41 > 8.42e-05
        + 0.02769234 * (0.02099671 * 0.5 * ((0.083 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.083 - Q.z_displaced3) / 0.02099671) / math.sqrt(2)))) / 0.03726902   # +2.8%  z_displaced3 < 0.083
        - 0.02481426 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.5) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.5) / 6.345399) / math.sqrt(2)))) * (0.1060766 * 0.5 * ((0.224 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.224 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) / 5.916942   # -2.5%  mass_top40 > 78.5 and lep_z < 0.224
        + 0.02471228 * (6.230332 * 0.5 * ((13.7 - Q.sip_3d_2) / 6.230332) * (1 + math.erf(((13.7 - Q.sip_3d_2) / 6.230332) / math.sqrt(2)))) / 4.312624   # +2.5%  sip_3d_2 < 13.7
        - 0.02452417 * (8.494762 * 0.5 * ((20.9 - Q.sip_3d_2) / 8.494762) * (1 + math.erf(((20.9 - Q.sip_3d_2) / 8.494762) / math.sqrt(2)))) / 7.758197   # -2.5%  sip_3d_2 < 20.9
        - 0.02436606 * (3.373048 * 0.5 * ((Q.mass_top15 - 82.9) / 3.373048) * (1 + math.erf(((Q.mass_top15 - 82.9) / 3.373048) / math.sqrt(2)))) / 11.11927   # -2.4%  mass_top15 > 82.9
        - 0.02289569 * (0.09779513 * 0.5 * ((0.67 - Q.sdb_2_z) / 0.09779513) * (1 + math.erf(((0.67 - Q.sdb_2_z) / 0.09779513) / math.sqrt(2)))) / 0.3550781   # -2.3%  sdb_2_z < 0.67
        + 0.01831374 * (1.0 * 0.5 * ((Q.sjf_2_n2disp - -0.0463) / 1.0) * (1 + math.erf(((Q.sjf_2_n2disp - -0.0463) / 1.0) / math.sqrt(2)))) / 0.7653561   # +1.8%  sjf_2_n2disp > -0.0463
        + 0.01813634 * (3.153038 * 0.5 * ((Q.mass - 121.0) / 3.153038) * (1 + math.erf(((Q.mass - 121.0) / 3.153038) / math.sqrt(2)))) / 11.61364   # +1.8%  mass > 121
        - 0.01655339 * (6.345399 * 0.5 * ((Q.mass_top40 - 80.0) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 80.0) / 6.345399) / math.sqrt(2)))) * (0.01771551 * 0.5 * ((0.119 - Q.C2_b2) / 0.01771551) * (1 + math.erf(((0.119 - Q.C2_b2) / 0.01771551) / math.sqrt(2)))) / 1.5629   # -1.7%  mass_top40 > 80 and C2_b2 < 0.119
        - 0.01648334 * (2.952984 * 0.5 * ((75.6 - Q.sj4_pair_mass_max) / 2.952984) * (1 + math.erf(((75.6 - Q.sj4_pair_mass_max) / 2.952984) / math.sqrt(2)))) / 8.029674   # -1.6%  sj4_pair_mass_max < 75.6
        + 0.01597014 * (1.0 * 0.5 * ((3.11 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((3.11 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 0.960672   # +1.6%  n_s3d_above_3 < 3.11
        - 0.01369102 * (3.705078 * 0.5 * ((5.82 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.82 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.5 * 0.5 * ((2.0 - Q.lepsj_2_n_d3) / 1.5) * (1 + math.erf(((2.0 - Q.lepsj_2_n_d3) / 1.5) / math.sqrt(2)))) / 4.419171   # -1.4%  max_abs_d0 < 5.82 and lepsj_2_n_d3 < 2
        - 0.01258105 * (1.951776e-05 * 0.5 * ((Q.ecf_g41 - 8.18e-05) / 1.951776e-05) * (1 + math.erf(((Q.ecf_g41 - 8.18e-05) / 1.951776e-05) / math.sqrt(2)))) * (57.98295 * 0.5 * ((768.0 - Q.sum_pt_top40) / 57.98295) * (1 + math.erf(((768.0 - Q.sum_pt_top40) / 57.98295) / math.sqrt(2)))) / 0.02317822   # -1.3%  ecf_g41 > 8.18e-05 and sum_pt_top40 < 768
        - 0.01241581 * (0.001766249 * 0.5 * ((0.0172 - Q.M3_b2) / 0.001766249) * (1 + math.erf(((0.0172 - Q.M3_b2) / 0.001766249) / math.sqrt(2)))) / 0.006085558   # -1.2%  M3_b2 < 0.0172
        + 0.01215057 * (5.171003 * 0.5 * ((7.28 - Q.sip_3d_1) / 5.171003) * (1 + math.erf(((7.28 - Q.sip_3d_1) / 5.171003) / math.sqrt(2)))) / 0.9276915   # +1.2%  sip_3d_1 < 7.28
        + 0.01157091 * (1.0 * 0.5 * ((1.99 - Q.sdb_5_n) / 1.0) * (1 + math.erf(((1.99 - Q.sdb_5_n) / 1.0) / math.sqrt(2)))) / 1.033489   # +1.2%  sdb_5_n < 1.99
        - 0.01085187 * (0.01159153 * 0.5 * ((0.0445 - Q.z_dr_0p4_up) / 0.01159153) * (1 + math.erf(((0.0445 - Q.z_dr_0p4_up) / 0.01159153) / math.sqrt(2)))) / 0.02106792   # -1.1%  z_dr_0p4_up < 0.0445
        + 0.009957998 * (0.03060878 * 0.5 * ((0.0658 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0658 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) / 0.0488087   # +1.0%  lepsj_2_dr < 0.0658
        - 0.009926239 * (20.79165 * 0.5 * ((48.7 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((48.7 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) / 24.8637   # -1.0%  sv_1_sd0_sum < 48.7
        + 0.009686811 * (0.2916549 * 0.5 * ((Q.lne_21 - 1.09) / 0.2916549) * (1 + math.erf(((Q.lne_21 - 1.09) / 0.2916549) / math.sqrt(2)))) * (0.02507469 * 0.5 * ((0.155 - Q.sj4_zsoft) / 0.02507469) * (1 + math.erf(((0.155 - Q.sj4_zsoft) / 0.02507469) / math.sqrt(2)))) / 0.06518371   # +1.0%  lne_21 > 1.09 and sj4_zsoft < 0.155
        - 0.009011889 * (1.0 * 0.5 * ((5.01 - Q.n_s3d_above_10) / 1.0) * (1 + math.erf(((5.01 - Q.n_s3d_above_10) / 1.0) / math.sqrt(2)))) * (0.02426199 * 0.5 * ((Q.sdb_2_z - 0.208) / 0.02426199) * (1 + math.erf(((Q.sdb_2_z - 0.208) / 0.02426199) / math.sqrt(2)))) / 0.4616623   # -0.9%  n_s3d_above_10 < 5.01 and sdb_2_z > 0.208
        + 0.008345121 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.07) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.07) / 1.0) / math.sqrt(2)))) * (248.3135 * 0.5 * ((182.0 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((182.0 - Q.sip_3d_3) / 248.3135) / math.sqrt(2)))) / 63.71468   # +0.8%  n_s3d_above_3 > 5.07 and sip_3d_3 < 182
        + 0.007786898 * (15.58693 * 0.5 * ((Q.mass_top40 - 159.0) / 15.58693) * (1 + math.erf(((Q.mass_top40 - 159.0) / 15.58693) / math.sqrt(2)))) / 2.576283   # +0.8%  mass_top40 > 159
        + 0.006987724 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.99) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.99) / 1.0) / math.sqrt(2)))) * (0.02937692 * 0.5 * ((0.496 - Q.sj3_pairmin_over_m) / 0.02937692) * (1 + math.erf(((0.496 - Q.sj3_pairmin_over_m) / 0.02937692) / math.sqrt(2)))) / 0.1744813   # +0.7%  n_s3d_above_3 > 4.99 and sj3_pairmin_over_m < 0.496
        + 0.006355411 * (0.04874922 * 0.5 * ((0.126 - Q.sjf_4_1_z_d3) / 0.04874922) * (1 + math.erf(((0.126 - Q.sjf_4_1_z_d3) / 0.04874922) / math.sqrt(2)))) * (6.566474 * 0.5 * ((13.2 - Q.lepsj_3_mass) / 6.566474) * (1 + math.erf(((13.2 - Q.lepsj_3_mass) / 6.566474) / math.sqrt(2)))) / 0.873084   # +0.6%  sjf_4_1_z_d3 < 0.126 and lepsj_3_mass < 13.2
        - 0.006260837 * (8.537109 * 0.5 * ((17.0 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((17.0 - Q.max_abs_d0) / 8.537109) / math.sqrt(2)))) * (0.1631384 * 0.5 * ((0.618 - Q.D3) / 0.1631384) * (1 + math.erf(((0.618 - Q.D3) / 0.1631384) / math.sqrt(2)))) / 3.087783   # -0.6%  max_abs_d0 < 17 and D3 < 0.618
        - 0.006235746 * (15.47377 * 0.5 * ((54.8 - Q.mass) / 15.47377) * (1 + math.erf(((54.8 - Q.mass) / 15.47377) / math.sqrt(2)))) / 0.9883048   # -0.6%  mass < 54.8
        + 0.005506689 * (0.2024667 * 0.5 * ((Q.pair_max_lnm2 - 7.52) / 0.2024667) * (1 + math.erf(((Q.pair_max_lnm2 - 7.52) / 0.2024667) / math.sqrt(2)))) / 0.07361128   # +0.6%  pair_max_lnm2 > 7.52
        - 0.004719495 * (0.01556476 * 0.5 * ((0.259 - Q.N2) / 0.01556476) * (1 + math.erf(((0.259 - Q.N2) / 0.01556476) / math.sqrt(2)))) / 0.01873725   # -0.5%  N2 < 0.259
        + 0.003508619 * (6.345399 * 0.5 * ((Q.mass_top40 - 81.4) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 81.4) / 6.345399) / math.sqrt(2)))) * (0.01759885 * 0.5 * ((Q.sj3_z1 - 0.492) / 0.01759885) * (1 + math.erf(((Q.sj3_z1 - 0.492) / 0.01759885) / math.sqrt(2)))) / 3.473779   # +0.4%  mass_top40 > 81.4 and sj3_z1 > 0.492
        + 0.002905995 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2)))) / 0.2342604   # +0.3%  sjf_2_1_n_d3 > 5
        - 0.00282469 * (6.036668 * 0.5 * ((85.8 - Q.mres_sd_mass_b2z01) / 6.036668) * (1 + math.erf(((85.8 - Q.mres_sd_mass_b2z01) / 6.036668) / math.sqrt(2)))) / 7.76092   # -0.3%  mres_sd_mass_b2z01 < 85.8
        + 0.002715103 * (0.02719315 * 0.5 * ((0.156 - Q.sdb_2_z) / 0.02719315) * (1 + math.erf(((0.156 - Q.sdb_2_z) / 0.02719315) / math.sqrt(2)))) / 0.01347431   # +0.3%  sdb_2_z < 0.156
        + 0.002261595 * (15.58693 * 0.5 * ((Q.mass_top40 - 154.0) / 15.58693) * (1 + math.erf(((Q.mass_top40 - 154.0) / 15.58693) / math.sqrt(2)))) * (0.01417841 * 0.5 * ((0.106 - Q.C2_b2) / 0.01417841) * (1 + math.erf(((0.106 - Q.C2_b2) / 0.01417841) / math.sqrt(2)))) / 0.0588783   # +0.2%  mass_top40 > 154 and C2_b2 < 0.106
        + 0.002074789 * (4.245069 * 0.5 * ((125.0 - Q.mres_sd_mass_b2z01) / 4.245069) * (1 + math.erf(((125.0 - Q.mres_sd_mass_b2z01) / 4.245069) / math.sqrt(2)))) * (1.0 * 0.5 * ((-0.000129 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((-0.000129 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.6034642   # +0.2%  mres_sd_mass_b2z01 < 125 and dc_2_n_lep < -0.000129
        + 0.001844432 * (1.5 * 0.5 * ((10.5 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((10.5 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2)))) * (0.3391285 * 0.5 * ((Q.max_dr - 0.922) / 0.3391285) * (1 + math.erf(((Q.max_dr - 0.922) / 0.3391285) / math.sqrt(2)))) / 0.4817585   # +0.2%  n_dr_0p4_up < 10.5 and max_dr > 0.922
        + 0.001445682 * (0.09626377 * 0.5 * ((Q.sjf_2_1_z_d3 - 0.37) / 0.09626377) * (1 + math.erf(((Q.sjf_2_1_z_d3 - 0.37) / 0.09626377) / math.sqrt(2)))) / 0.006105979   # +0.1%  sjf_2_1_z_d3 > 0.37
    )
    return z


def neuron_84(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.43e-05
    )
    return z


def neuron_85(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.15e-07
    )
    return z


def neuron_86(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.71e-05
    )
    return z


def neuron_87(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.61e-06
    )
    return z


def neuron_88(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.03e-07
    )
    return z


def neuron_89(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.74e-06
    )
    return z


def neuron_90(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.00023
    )
    return z


def neuron_91(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.99e-06
    )
    return z


def neuron_92(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.72e-06
    )
    return z


def neuron_93(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.47e-07
    )
    return z


def neuron_94(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.32e-06
    )
    return z


def neuron_95(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.35e-07
    )
    return z


def neuron_96(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.23e-07
    )
    return z


def neuron_97(Q):
    # scale S = 12.54; each line: share * term / its average size
    z = 12.53634 * (-0.1196522
        - 0.222501 * (6.036668 * 0.5 * ((Q.mres_sd_mass_b2z01 - 75.7) / 6.036668) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 75.7) / 6.036668) / math.sqrt(2)))) / 35.44279   # -22.3%  mres_sd_mass_b2z01 > 75.7
        + 0.121576 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 90.9) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 90.9) / 5.26174) / math.sqrt(2)))) / 24.42495   # +12.2%  mres_sd_mass_b2z01 > 90.9
        + 0.0821305 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 53.2) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 53.2) / 19.93276) / math.sqrt(2)))) / 54.47701   # +8.2%  mres_sd_mass_b2z01 > 53.2
        + 0.04542956 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0) / 6.991655) / math.sqrt(2)))) / 7.245805   # +4.5%  mres_sd_mass_b2z01 > 130
        - 0.03842691 * (14.05251 * 0.5 * ((Q.mres_sd_mass_b2z01 - 139.0) / 14.05251) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 139.0) / 14.05251) / math.sqrt(2)))) / 5.9917   # -3.8%  mres_sd_mass_b2z01 > 139
        + 0.03565352 * (10.53967 * 0.5 * ((93.8 - Q.nca_sj4_pair_mass_2nd) / 10.53967) * (1 + math.erf(((93.8 - Q.nca_sj4_pair_mass_2nd) / 10.53967) / math.sqrt(2)))) / 37.56005   # +3.6%  nca_sj4_pair_mass_2nd < 93.8
        - 0.032368 * (5.5191 * 0.5 * ((Q.mres_pruned_mass - 76.5) / 5.5191) * (1 + math.erf(((Q.mres_pruned_mass - 76.5) / 5.5191) / math.sqrt(2)))) / 28.77845   # -3.2%  mres_pruned_mass > 76.5
        + 0.03132326 * (12.30925 * 0.5 * ((39.1 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.1 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) / 31.66766   # +3.1%  mass_displaced3 < 39.1
        - 0.02450944 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.98) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.98) / 1.0) / math.sqrt(2)))) / 1.688234   # -2.5%  n_s3d_above_3 > 2.98
        + 0.0235907 * (6.875137 * 0.5 * ((18.9 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.9 - Q.mass_displaced3) / 6.875137) / math.sqrt(2)))) / 13.38194   # +2.4%  mass_displaced3 < 18.9
        - 0.02284692 * (289.0822 * 0.5 * ((446.0 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((446.0 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) / 296.4976   # -2.3%  sip_3d_2 < 446
        + 0.02212003 * (4.883139 * 0.5 * ((Q.mass_top50 - 89.5) / 4.883139) * (1 + math.erf(((Q.mass_top50 - 89.5) / 4.883139) / math.sqrt(2)))) / 28.61756   # +2.2%  mass_top50 > 89.5
        + 0.01994767 * (0.01301382 * 0.5 * ((0.119 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.119 - Q.tau2) / 0.01301382) / math.sqrt(2)))) / 0.04809052   # +2.0%  tau2 < 0.119
        + 0.01981003 * (5.286881 * 0.5 * ((101.0 - Q.mass) / 5.286881) * (1 + math.erf(((101.0 - Q.mass) / 5.286881) / math.sqrt(2)))) / 8.305858   # +2.0%  mass < 101
        - 0.01965329 * (3.838712e-05 * 0.5 * ((0.00013 - Q.e3_b2) / 3.838712e-05) * (1 + math.erf(((0.00013 - Q.e3_b2) / 3.838712e-05) / math.sqrt(2)))) / 7.289356e-05   # -2.0%  e3_b2 < 0.00013
        + 0.01837621 * (1.0 * 0.5 * ((Q.n_sd0_above_3 - 1.96) / 1.0) * (1 + math.erf(((Q.n_sd0_above_3 - 1.96) / 1.0) / math.sqrt(2)))) / 1.857825   # +1.8%  n_sd0_above_3 > 1.96
        - 0.01716172 * (0.0002686389 * 0.5 * ((0.000418 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000418 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (115.0698 * 0.5 * ((174.0 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((174.0 - Q.jd_3d_4) / 115.0698) / math.sqrt(2)))) / 0.04185701   # -1.7%  e3_b2 < 0.000418 and jd_3d_4 < 174
        - 0.01708858 * (2.991814 * 0.5 * ((65.8 - Q.nca_sj4_pair_mass_2nd) / 2.991814) * (1 + math.erf(((65.8 - Q.nca_sj4_pair_mass_2nd) / 2.991814) / math.sqrt(2)))) / 13.73257   # -1.7%  nca_sj4_pair_mass_2nd < 65.8
        + 0.01443467 * (5.725739 * 0.5 * ((Q.mres_pruned_mass - 125.0) / 5.725739) * (1 + math.erf(((Q.mres_pruned_mass - 125.0) / 5.725739) / math.sqrt(2)))) / 6.702142   # +1.4%  mres_pruned_mass > 125
        - 0.01386289 * (22.10989 * 0.5 * ((172.0 - Q.mres_pruned_mass) / 22.10989) * (1 + math.erf(((172.0 - Q.mres_pruned_mass) / 22.10989) / math.sqrt(2)))) / 78.63792   # -1.4%  mres_pruned_mass < 172
        + 0.01315872 * (35.5 * 0.5 * ((284.0 - Q.n_pairs_kt_above_1) / 35.5) * (1 + math.erf(((284.0 - Q.n_pairs_kt_above_1) / 35.5) / math.sqrt(2)))) / 79.69186   # +1.3%  n_pairs_kt_above_1 < 284
        + 0.0128379 * (0.0002686389 * 0.5 * ((0.000404 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000404 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((9.85 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((9.85 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 0.001798215   # +1.3%  e3_b2 < 0.000404 and n_s3d_above_3 < 9.85
        + 0.01111244 * (289.0822 * 0.5 * ((449.0 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((449.0 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.929175 * 0.5 * ((4.01 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((4.01 - Q.jd_3d_5) / 1.929175) / math.sqrt(2)))) / 453.776   # +1.1%  sip_3d_2 < 449 and jd_3d_5 < 4.01
        + 0.01061758 * (4.879578 * 0.5 * ((121.0 - Q.mass_top30) / 4.879578) * (1 + math.erf(((121.0 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) / 23.14879   # +1.1%  mass_top30 < 121
        - 0.009947652 * (175.2971 * 0.5 * ((84.2 - Q.sv_2_sd0_sum) / 175.2971) * (1 + math.erf(((84.2 - Q.sv_2_sd0_sum) / 175.2971) / math.sqrt(2)))) / 47.05928   # -1.0%  sv_2_sd0_sum < 84.2
        + 0.008514182 * (0.02188437 * 0.5 * ((Q.z_top30_slots - 0.896) / 0.02188437) * (1 + math.erf(((Q.z_top30_slots - 0.896) / 0.02188437) / math.sqrt(2)))) / 0.06712996   # +0.9%  z_top30_slots > 0.896
        - 0.007949311 * (16.90428 * 0.5 * ((Q.mass - 164.0) / 16.90428) * (1 + math.erf(((Q.mass - 164.0) / 16.90428) / math.sqrt(2)))) / 3.288951   # -0.8%  mass > 164
        + 0.007506455 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 51.7) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 51.7) / 19.93276) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.237 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.237 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2)))) / 5.47113   # +0.8%  mres_sd_mass_b2z01 > 51.7 and z_neutral_had < 0.237
        - 0.007426761 * (0.01301382 * 0.5 * ((0.119 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.119 - Q.tau2) / 0.01301382) / math.sqrt(2)))) * (0.05096652 * 0.5 * ((0.128 - Q.sjq_2_prod_k05) / 0.05096652) * (1 + math.erf(((0.128 - Q.sjq_2_prod_k05) / 0.05096652) / math.sqrt(2)))) / 0.009968347   # -0.7%  tau2 < 0.119 and sjq_2_prod_k05 < 0.128
        + 0.006529578 * (0.0002686389 * 0.5 * ((0.00041 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.00041 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.08904938 * 0.5 * ((Q.jet_charge_k03 - -0.0536) / 0.08904938) * (1 + math.erf(((Q.jet_charge_k03 - -0.0536) / 0.08904938) / math.sqrt(2)))) / 8.926606e-05   # +0.7%  e3_b2 < 0.00041 and jet_charge_k03 > -0.0536
        + 0.006484675 * (3.289335 * 0.5 * ((Q.mass - 125.0) / 3.289335) * (1 + math.erf(((Q.mass - 125.0) / 3.289335) / math.sqrt(2)))) / 10.17448   # +0.6%  mass > 125
        - 0.005995183 * (1.0 * 0.5 * ((6.01 - Q.n_neutral_had) / 1.0) * (1 + math.erf(((6.01 - Q.n_neutral_had) / 1.0) / math.sqrt(2)))) / 2.184815   # -0.6%  n_neutral_had < 6.01
        + 0.004983188 * (0.1277784 * 0.5 * ((Q.lund3_lndelta - -1.89) / 0.1277784) * (1 + math.erf(((Q.lund3_lndelta - -1.89) / 0.1277784) / math.sqrt(2)))) / 0.5162886   # +0.5%  lund3_lndelta > -1.89
        + 0.004848377 * (12.30925 * 0.5 * ((38.6 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((38.6 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (0.03286901 * 0.5 * ((0.489 - Q.ak02_dr12) / 0.03286901) * (1 + math.erf(((0.489 - Q.ak02_dr12) / 0.03286901) / math.sqrt(2)))) / 4.82388   # +0.5%  mass_displaced3 < 38.6 and ak02_dr12 < 0.489
        + 0.004801794 * (1.0 * 0.5 * ((1.01 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.01 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2)))) * (0.1648594 * 0.5 * ((Q.D2 - 0.837) / 0.1648594) * (1 + math.erf(((Q.D2 - 0.837) / 0.1648594) / math.sqrt(2)))) / 1.01684   # +0.5%  ak02_2_n_lep < 1.01 and D2 > 0.837
        - 0.003745988 * (4.879578 * 0.5 * ((116.0 - Q.mass_top30) / 4.879578) * (1 + math.erf(((116.0 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 0.925) / 1.0) * (1 + math.erf(((Q.n_muon - 0.925) / 1.0) / math.sqrt(2)))) / 3.109998   # -0.4%  mass_top30 < 116 and n_muon > 0.925
        - 0.003561836 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 89.7) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 89.7) / 5.26174) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.5 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.5 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 51.26564   # -0.4%  mres_sd_mass_b2z01 > 89.7 and jd_3d_6 < 5.5
        + 0.003349152 * (12.30925 * 0.5 * ((39.2 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.2 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (23.09414 * 0.5 * ((Q.sum_pt_top5 - 220.0) / 23.09414) * (1 + math.erf(((Q.sum_pt_top5 - 220.0) / 23.09414) / math.sqrt(2)))) / 4904.917   # +0.3%  mass_displaced3 < 39.2 and sum_pt_top5 > 220
        + 0.002735898 * (6.93786 * 0.5 * ((Q.sj3_pair_mass_max - 120.0) / 6.93786) * (1 + math.erf(((Q.sj3_pair_mass_max - 120.0) / 6.93786) / math.sqrt(2)))) / 3.554211   # +0.3%  sj3_pair_mass_max > 120
        + 0.002539145 * (0.0297357 * 0.5 * ((0.651 - Q.tau32) / 0.0297357) * (1 + math.erf(((0.651 - Q.tau32) / 0.0297357) / math.sqrt(2)))) / 0.06830809   # +0.3%  tau32 < 0.651
        - 0.002414774 * (0.1064494 * 0.5 * ((3.63 - Q.lund_max_lnkt) / 0.1064494) * (1 + math.erf(((3.63 - Q.lund_max_lnkt) / 0.1064494) / math.sqrt(2)))) / 0.1978589   # -0.2%  lund_max_lnkt < 3.63
        - 0.002311951 * (3.93783 * 0.5 * ((73.1 - Q.sj3_pair_mass_max) / 3.93783) * (1 + math.erf(((73.1 - Q.sj3_pair_mass_max) / 3.93783) / math.sqrt(2)))) / 4.256005   # -0.2%  sj3_pair_mass_max < 73.1
        - 0.002240941 * (0.0331444 * 0.5 * ((Q.z_displaced5 - 0.171) / 0.0331444) * (1 + math.erf(((Q.z_displaced5 - 0.171) / 0.0331444) / math.sqrt(2)))) / 0.02860814   # -0.2%  z_displaced5 > 0.171
        - 0.001974386 * (20.07181 * 0.5 * ((Q.mres_sd_mass_b1z01 - 180.0) / 20.07181) * (1 + math.erf(((Q.mres_sd_mass_b1z01 - 180.0) / 20.07181) / math.sqrt(2)))) / 1.793591   # -0.2%  mres_sd_mass_b1z01 > 180
        + 0.001823548 * (5.119423 * 0.5 * ((Q.mass_top5 - 63.5) / 5.119423) * (1 + math.erf(((Q.mass_top5 - 63.5) / 5.119423) / math.sqrt(2)))) / 3.594435   # +0.2%  mass_top5 > 63.5
        - 0.001798237 * (0.06488792 * 0.5 * ((Q.dc_split2_dr - 0.281) / 0.06488792) * (1 + math.erf(((Q.dc_split2_dr - 0.281) / 0.06488792) / math.sqrt(2)))) / 0.01063363   # -0.2%  dc_split2_dr > 0.281
        + 0.001708885 * (2.077045 * 0.5 * ((1.84 - Q.sj2_mass2) / 2.077045) * (1 + math.erf(((1.84 - Q.sj2_mass2) / 2.077045) / math.sqrt(2)))) / 0.08889277   # +0.2%  sj2_mass2 < 1.84
        + 0.001405717 * (19.15449 * 0.5 * ((Q.mres_sd_mass_b0z02 - 134.0) / 19.15449) * (1 + math.erf(((Q.mres_sd_mass_b0z02 - 134.0) / 19.15449) / math.sqrt(2)))) / 4.541893   # +0.1%  mres_sd_mass_b0z02 > 134
        + 0.001319939 * (4.879578 * 0.5 * ((120.0 - Q.mass_top30) / 4.879578) * (1 + math.erf(((120.0 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (52.10352 * 0.5 * ((Q.sum_e - 931.0) / 52.10352) * (1 + math.erf(((Q.sum_e - 931.0) / 52.10352) / math.sqrt(2)))) / 2843.162   # +0.1%  mass_top30 < 120 and sum_e > 931
        - 0.0009060022 * (4.879578 * 0.5 * ((119.0 - Q.mass_top30) / 4.879578) * (1 + math.erf(((119.0 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_min12_n_disp3 - 0.00984) / 1.0) * (1 + math.erf(((Q.ak02_min12_n_disp3 - 0.00984) / 1.0) / math.sqrt(2)))) / 2.756783   # -0.1%  mass_top30 < 119 and ak02_min12_n_disp3 > 0.00984
        - 0.0005960458 * (3.289335 * 0.5 * ((Q.mass - 127.0) / 3.289335) * (1 + math.erf(((Q.mass - 127.0) / 3.289335) / math.sqrt(2)))) * (0.01084662 * 0.5 * ((Q.pz_lnkt4 - 0.0251) / 0.01084662) * (1 + math.erf(((Q.pz_lnkt4 - 0.0251) / 0.01084662) / math.sqrt(2)))) / 0.1404555   # -0.1%  mass > 127 and pz_lnkt4 > 0.0251
        + 4.291345e-05 * (5.294312 * 0.5 * ((Q.mres_sd_mass_b2z01 - 104.0) / 5.294312) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 104.0) / 5.294312) / math.sqrt(2)))) * (3.2258e-08 * 0.5 * ((Q.e4_b2 - -1.71e-08) / 3.2258e-08) * (1 + math.erf(((Q.e4_b2 - -1.71e-08) / 3.2258e-08) / math.sqrt(2)))) / 6.576741e-06   # +0.0%  mres_sd_mass_b2z01 > 104 and e4_b2 > -1.71e-08
    )
    return z


def neuron_98(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.97e-07
    )
    return z


def neuron_99(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.00193
    )
    return z


def neuron_100(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.27e-06
    )
    return z


def neuron_101(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.88e-06
    )
    return z


def neuron_102(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.49e-06
    )
    return z


def neuron_103(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.42e-06
    )
    return z


def neuron_104(Q):
    # scale S = 6.897; each line: share * term / its average size
    z = 6.896618 * (-0.1383287
        + 0.1256896 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.3) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.3) / 19.93276) / math.sqrt(2)))) / 52.53533   # +12.6%  mres_sd_mass_b2z01 > 55.3
        + 0.08889343 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 3.23) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 3.23) / 1.000219) / math.sqrt(2)))) / 6.236664   # +8.9%  mass_displaced3 > 3.23
        + 0.07535711 * (0.171005 * 0.5 * ((-1.36 - Q.pair_mean_lndelta) / 0.171005) * (1 + math.erf(((-1.36 - Q.pair_mean_lndelta) / 0.171005) / math.sqrt(2)))) / 0.8604457   # +7.5%  pair_mean_lndelta < -1.36
        - 0.06852381 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.0) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.0) / 5.26174) / math.sqrt(2)))) / 24.35992   # -6.9%  mres_sd_mass_b2z01 > 91
        - 0.06740684 * (1.566771 * 0.5 * ((Q.mass_displaced3 - 4.44) / 1.566771) * (1 + math.erf(((Q.mass_displaced3 - 4.44) / 1.566771) / math.sqrt(2)))) / 5.803735   # -6.7%  mass_displaced3 > 4.44
        + 0.06054768 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2)))) / 15.93795   # +6.1%  mass < 118
        + 0.04962139 * (5.5 * 0.5 * ((58.2 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.2 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (0.04737588 * 0.5 * ((0.18 - Q.ak02_3_z) / 0.04737588) * (1 + math.erf(((0.18 - Q.ak02_3_z) / 0.04737588) / math.sqrt(2)))) / 3.259236   # +5.0%  n_pt_above_1 < 58.2 and ak02_3_z < 0.18
        - 0.03397907 * (0.01003572 * 0.5 * ((Q.pz_lnd0 - 0.0503) / 0.01003572) * (1 + math.erf(((Q.pz_lnd0 - 0.0503) / 0.01003572) / math.sqrt(2)))) / 0.07997976   # -3.4%  pz_lnd0 > 0.0503
        - 0.02477809 * (13.49273 * 0.5 * ((149.0 - Q.mass) / 13.49273) * (1 + math.erf(((149.0 - Q.mass) / 13.49273) / math.sqrt(2)))) * (202.2502 * 0.5 * ((1400.0 - Q.sum_e) / 202.2502) * (1 + math.erf(((1400.0 - Q.sum_e) / 202.2502) / math.sqrt(2)))) / 20588.55   # -2.5%  mass < 149 and sum_e < 1400
        + 0.02210488 * (11.52979 * 0.5 * ((99.3 - Q.mass_charged) / 11.52979) * (1 + math.erf(((99.3 - Q.mass_charged) / 11.52979) / math.sqrt(2)))) / 36.91257   # +2.2%  mass_charged < 99.3
        - 0.02186436 * (0.009790654 * 0.5 * ((Q.C2_b05 - 0.224) / 0.009790654) * (1 + math.erf(((Q.C2_b05 - 0.224) / 0.009790654) / math.sqrt(2)))) / 0.05385362   # -2.2%  C2_b05 > 0.224
        + 0.02090934 * (1.5 * 0.5 * ((6.05 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.05 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2)))) / 3.204528   # +2.1%  jd_n_d3_pt1 < 6.05
        - 0.01970407 * (0.05354637 * 0.5 * ((0.775 - Q.z_charged_had) / 0.05354637) * (1 + math.erf(((0.775 - Q.z_charged_had) / 0.05354637) / math.sqrt(2)))) / 0.2707001   # -2.0%  z_charged_had < 0.775
        + 0.01831776 * (19.4821 * 0.5 * ((31.1 - Q.sjf_2_2_max3d) / 19.4821) * (1 + math.erf(((31.1 - Q.sjf_2_2_max3d) / 19.4821) / math.sqrt(2)))) / 15.8309   # +1.8%  sjf_2_2_max3d < 31.1
        + 0.01770302 * (1.5 * 0.5 * ((18.1 - Q.sjq_2_1_nch) / 1.5) * (1 + math.erf(((18.1 - Q.sjq_2_1_nch) / 1.5) / math.sqrt(2)))) / 6.708294   # +1.8%  sjq_2_1_nch < 18.1
        - 0.01737464 * (3.069852 * 0.5 * ((4.66 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.66 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) / 1.099323   # -1.7%  sjf_2_2_max3d < 4.66
        + 0.01644019 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.84) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.84) / 4.308625) / math.sqrt(2)))) / 5.451042   # +1.6%  lep_ptrel > 6.84
        - 0.01596155 * (0.6671377 * 0.5 * ((0.458 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.458 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2393059   # -1.6%  lep_iso < 0.458
        + 0.01522836 * (5.5 * 0.5 * ((58.4 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.4 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (0.194914 * 0.5 * ((3.65 - Q.pair_max_lnkt) / 0.194914) * (1 + math.erf(((3.65 - Q.pair_max_lnkt) / 0.194914) / math.sqrt(2)))) / 20.39304   # +1.5%  n_pt_above_1 < 58.4 and pair_max_lnkt < 3.65
        - 0.01469886 * Q.n_charged_pt_above_10 / 8.309213   # -1.5%  n_charged_pt_above_10
        - 0.01447004 * (19.88146 * 0.5 * ((10.5 - Q.dc_2_jp) / 19.88146) * (1 + math.erf(((10.5 - Q.dc_2_jp) / 19.88146) / math.sqrt(2)))) / 4.536107   # -1.4%  dc_2_jp < 10.5
        - 0.01436752 * (6.5 * 0.5 * ((40.7 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((40.7 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2)))) / 7.506611   # -1.4%  n_pairs_kt_above_3 < 40.7
        + 0.01389149 * (0.002048025 * 0.5 * ((0.0207 - Q.tau5) / 0.002048025) * (1 + math.erf(((0.0207 - Q.tau5) / 0.002048025) / math.sqrt(2)))) / 0.001991773   # +1.4%  tau5 < 0.0207
        - 0.01356867 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.97) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.97) / 4.308625) / math.sqrt(2)))) * (0.02843722 * 0.5 * ((0.102 - Q.ak02_3_z) / 0.02843722) * (1 + math.erf(((0.102 - Q.ak02_3_z) / 0.02843722) / math.sqrt(2)))) / 0.4050992   # -1.4%  lep_ptrel > 6.97 and ak02_3_z < 0.102
        - 0.01311793 * (0.01171084 * 0.5 * ((0.0258 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.0258 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) / 0.01185706   # -1.3%  sjf_2_1_z_d3 < 0.0258
        + 0.01293724 * (0.07989133 * 0.5 * ((Q.N3_b05 - 0.843) / 0.07989133) * (1 + math.erf(((Q.N3_b05 - 0.843) / 0.07989133) / math.sqrt(2)))) / 0.1381164   # +1.3%  N3_b05 > 0.843
        + 0.01255343 * (4.590131 * 0.5 * ((107.0 - Q.mres_sd_mass_b2z01) / 4.590131) * (1 + math.erf(((107.0 - Q.mres_sd_mass_b2z01) / 4.590131) / math.sqrt(2)))) / 16.24319   # +1.3%  mres_sd_mass_b2z01 < 107
        + 0.01216214 * (1.0 * 0.5 * ((1.01 - Q.lepsj_2_n_d3) / 1.0) * (1 + math.erf(((1.01 - Q.lepsj_2_n_d3) / 1.0) / math.sqrt(2)))) / 0.6710211   # +1.2%  lepsj_2_n_d3 < 1.01
        - 0.01028512 * (1.5 * 0.5 * ((18.0 - Q.sjq_2_1_nch) / 1.5) * (1 + math.erf(((18.0 - Q.sjq_2_1_nch) / 1.5) / math.sqrt(2)))) * (777.2244 * 0.5 * ((1040.0 - Q.sv_1_sd0_sum) / 777.2244) * (1 + math.erf(((1040.0 - Q.sv_1_sd0_sum) / 777.2244) / math.sqrt(2)))) / 5333.276   # -1.0%  sjq_2_1_nch < 18 and sv_1_sd0_sum < 1040
        + 0.008912868 * (0.06999318 * 0.5 * ((Q.lund2_lndelta - -1.06) / 0.06999318) * (1 + math.erf(((Q.lund2_lndelta - -1.06) / 0.06999318) / math.sqrt(2)))) / 0.2337211   # +0.9%  lund2_lndelta > -1.06
        + 0.007732756 * (0.01171084 * 0.5 * ((0.0277 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.0277 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) * (95.61515 * 0.5 * ((41.3 - Q.dc_2_jp) / 95.61515) * (1 + math.erf(((41.3 - Q.dc_2_jp) / 95.61515) / math.sqrt(2)))) / 0.2882695   # +0.8%  sjf_2_1_z_d3 < 0.0277 and dc_2_jp < 41.3
        - 0.007636266 * (0.00390151 * 0.5 * ((0.0733 - Q.M2) / 0.00390151) * (1 + math.erf(((0.0733 - Q.M2) / 0.00390151) / math.sqrt(2)))) * (0.04921085 * 0.5 * ((Q.N3_b05 - 0.756) / 0.04921085) * (1 + math.erf(((Q.N3_b05 - 0.756) / 0.04921085) / math.sqrt(2)))) / 0.004660567   # -0.8%  M2 < 0.0733 and N3_b05 > 0.756
        + 0.00629994 * (0.1799842 * 0.5 * ((2.1 - Q.sjf_2_2_max3d) / 0.1799842) * (1 + math.erf(((2.1 - Q.sjf_2_2_max3d) / 0.1799842) / math.sqrt(2)))) / 0.175904   # +0.6%  sjf_2_2_max3d < 2.1
        - 0.006174679 * (0.03993816 * 0.5 * ((0.469 - Q.tau32_b2) / 0.03993816) * (1 + math.erf(((0.469 - Q.tau32_b2) / 0.03993816) / math.sqrt(2)))) / 0.06014746   # -0.6%  tau32_b2 < 0.469
        + 0.005819976 * (0.01129976 * 0.5 * ((Q.N2 - 0.335) / 0.01129976) * (1 + math.erf(((Q.N2 - 0.335) / 0.01129976) / math.sqrt(2)))) / 0.02193341   # +0.6%  N2 > 0.335
        - 0.005781677 * (3.069852 * 0.5 * ((4.46 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.46 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) * (164.9991 * 0.5 * ((216.0 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((216.0 - Q.sip_3d_2) / 164.9991) / math.sqrt(2)))) / 155.7579   # -0.6%  sjf_2_2_max3d < 4.46 and sip_3d_2 < 216
        + 0.0054744 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.0) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.0) / 18.66842) / math.sqrt(2)))) / 3.495819   # +0.5%  mres_sd_mass_b0z005 > 159
        - 0.005106806 * (13.49273 * 0.5 * ((149.0 - Q.mass) / 13.49273) * (1 + math.erf(((149.0 - Q.mass) / 13.49273) / math.sqrt(2)))) * (0.07882828 * 0.5 * ((Q.sjq_2_sumabs_k03 - 0.743) / 0.07882828) * (1 + math.erf(((Q.sjq_2_sumabs_k03 - 0.743) / 0.07882828) / math.sqrt(2)))) / 7.100744   # -0.5%  mass < 149 and sjq_2_sumabs_k03 > 0.743
        - 0.004773327 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 54.7) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 54.7) / 19.93276) / math.sqrt(2)))) * (0.05887118 * 0.5 * ((0.388 - Q.ak02_dr13) / 0.05887118) * (1 + math.erf(((0.388 - Q.ak02_dr13) / 0.05887118) / math.sqrt(2)))) / 8.572867   # -0.5%  mres_sd_mass_b2z01 > 54.7 and ak02_dr13 < 0.388
        + 0.004286059 * (6.5 * 0.5 * ((40.7 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((40.7 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.994 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.994 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) / 4.379157   # +0.4%  n_pairs_kt_above_3 < 40.7 and kt2_2_n_lep < 0.994
        - 0.004173798 * (1.5 * 0.5 * ((Q.sjf_3_1_n_d3 - 3.94) / 1.5) * (1 + math.erf(((Q.sjf_3_1_n_d3 - 3.94) / 1.5) / math.sqrt(2)))) / 0.2794669   # -0.4%  sjf_3_1_n_d3 > 3.94
        + 0.003892547 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 1.6) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 1.6) / 1.000219) / math.sqrt(2)))) * (0.02843722 * 0.5 * ((0.101 - Q.ak02_3_z) / 0.02843722) * (1 + math.erf(((0.101 - Q.ak02_3_z) / 0.02843722) / math.sqrt(2)))) / 0.4628519   # +0.4%  mass_displaced3 > 1.6 and ak02_3_z < 0.101
        - 0.00369702 * (5.5 * 0.5 * ((57.3 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((57.3 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_1_n_d5 - 1.12) / 1.0) * (1 + math.erf(((Q.sjf_2_1_n_d5 - 1.12) / 1.0) / math.sqrt(2)))) / 18.74775   # -0.4%  n_pt_above_1 < 57.3 and sjf_2_1_n_d5 > 1.12
        - 0.001998567 * (1.0 * 0.5 * ((Q.sdb_2_n - 9.13) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 9.13) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.02 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((3.02 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2)))) / 1.56629   # -0.2%  sdb_2_n > 9.13 and n_lund_kt_above_5 < 3.02
        + 0.001847208 * (0.08021924 * 0.5 * ((Q.lund_max_lndelta - -0.623) / 0.08021924) * (1 + math.erf(((Q.lund_max_lndelta - -0.623) / 0.08021924) / math.sqrt(2)))) / 0.0249305   # +0.2%  lund_max_lndelta > -0.623
        + 0.001822418 * (8.295007 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 83.4) / 8.295007) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 83.4) / 8.295007) / math.sqrt(2)))) / 1.780243   # +0.2%  nca_sj4_pair_mass_2nd > 83.4
        + 0.001108837 * (12.49365 * 0.5 * ((Q.sj4_pair_mass_max - 127.0) / 12.49365) * (1 + math.erf(((Q.sj4_pair_mass_max - 127.0) / 12.49365) / math.sqrt(2)))) / 1.40574   # +0.1%  sj4_pair_mass_max > 127
        - 0.000724734 * (8.295007 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 84.4) / 8.295007) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 84.4) / 8.295007) / math.sqrt(2)))) * (1.512315e-07 * 0.5 * ((Q.e4 - 5.35e-07) / 1.512315e-07) * (1 + math.erf(((Q.e4 - 5.35e-07) / 1.512315e-07) / math.sqrt(2)))) / 2.108951e-05   # -0.1%  nca_sj4_pair_mass_2nd > 84.4 and e4 > 5.35e-07
        - 0.0002724371 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 160.0) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 160.0) / 18.66842) / math.sqrt(2)))) * (1.842068 * 0.5 * ((Q.lnerel_78 - -18.4) / 1.842068) * (1 + math.erf(((Q.lnerel_78 - -18.4) / 1.842068) / math.sqrt(2)))) / 4.931481   # -0.0%  mres_sd_mass_b0z005 > 160 and lnerel_78 > -18.4
        - 6.077464e-06 * (6.58003 * 0.5 * ((Q.mass_top30 - 127.0) / 6.58003) * (1 + math.erf(((Q.mass_top30 - 127.0) / 6.58003) / math.sqrt(2)))) * (0.9957982 * 0.5 * ((-5.57 - Q.sip_3d_2) / 0.9957982) * (1 + math.erf(((-5.57 - Q.sip_3d_2) / 0.9957982) / math.sqrt(2)))) / 6.092143e-13   # -0.0%  mass_top30 > 127 and sip_3d_2 < -5.57
    )
    return z


def neuron_105(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.73e-05
    )
    return z


def neuron_106(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.16e-05
    )
    return z


def neuron_107(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.16e-05
    )
    return z


def neuron_108(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.46e-06
    )
    return z


def neuron_109(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.17e-07
    )
    return z


def neuron_110(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.58e-06
    )
    return z


def neuron_111(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.67e-06
    )
    return z


def neuron_112(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.24e-06
    )
    return z


def neuron_113(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.67e-05
    )
    return z


def neuron_114(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.68e-06
    )
    return z


def neuron_115(Q):
    # scale S = 9.31; each line: share * term / its average size
    z = 9.30986 * (0.01922693
        - 0.09847618 * (0.01854575 * 0.5 * ((Q.N2 - 0.225) / 0.01854575) * (1 + math.erf(((Q.N2 - 0.225) / 0.01854575) / math.sqrt(2)))) / 0.0898823   # -9.8%  N2 > 0.225
        + 0.08105387 * (0.179073 * 0.5 * ((0.517 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.517 - Q.lep_z) / 0.179073) / math.sqrt(2)))) / 0.4361851   # +8.1%  lep_z < 0.517
        - 0.06427358 * (6.832602 * 0.5 * ((108.0 - Q.sj4_pair_mass_max) / 6.832602) * (1 + math.erf(((108.0 - Q.sj4_pair_mass_max) / 6.832602) / math.sqrt(2)))) / 29.9189   # -6.4%  sj4_pair_mass_max < 108
        - 0.06127594 * (0.0003783019 * 0.5 * ((0.000805 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.000805 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (0.06955833 * 0.5 * ((0.374 - Q.z_neutral_had) / 0.06955833) * (1 + math.erf(((0.374 - Q.z_neutral_had) / 0.06955833) / math.sqrt(2)))) / 0.0001505199   # -6.1%  e3_b2 < 0.000805 and z_neutral_had < 0.374
        + 0.05855556 * (5.110145 * 0.5 * ((131.0 - Q.mass) / 5.110145) * (1 + math.erf(((131.0 - Q.mass) / 5.110145) / math.sqrt(2)))) / 24.44592   # +5.9%  mass < 131
        + 0.05742131 * (0.04122765 * 0.5 * ((Q.N2_b05 - 0.303) / 0.04122765) * (1 + math.erf(((Q.N2_b05 - 0.303) / 0.04122765) / math.sqrt(2)))) / 0.1370729   # +5.7%  N2_b05 > 0.303
        + 0.04991514 * (0.008001329 * 0.5 * ((0.12 - Q.M2) / 0.008001329) * (1 + math.erf(((0.12 - Q.M2) / 0.008001329) / math.sqrt(2)))) / 0.0426333   # +5.0%  M2 < 0.12
        + 0.0351391 * (0.0003783019 * 0.5 * ((0.000779 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.000779 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (0.1133023 * 0.5 * ((1.04 - Q.pair_mean_lnkt) / 0.1133023) * (1 + math.erf(((1.04 - Q.pair_mean_lnkt) / 0.1133023) / math.sqrt(2)))) / 0.0003587063   # +3.5%  e3_b2 < 0.000779 and pair_mean_lnkt < 1.04
        - 0.03459005 * (12.30925 * 0.5 * ((39.0 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.0 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) / 31.57142   # -3.5%  mass_displaced3 < 39
        + 0.03115735 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2)))) / 15.93795   # +3.1%  mass < 118
        - 0.02739216 * (5.413174 * 0.5 * ((115.0 - Q.sj3_pair_mass_max) / 5.413174) * (1 + math.erf(((115.0 - Q.sj3_pair_mass_max) / 5.413174) / math.sqrt(2)))) / 27.01453   # -2.7%  sj3_pair_mass_max < 115
        - 0.02686079 * (0.006239604 * 0.5 * ((0.0214 - Q.lam2) / 0.006239604) * (1 + math.erf(((0.0214 - Q.lam2) / 0.006239604) / math.sqrt(2)))) / 0.01534173   # -2.7%  lam2 < 0.0214
        + 0.02523519 * (0.008548367 * 0.5 * ((Q.N2_b05 - 0.435) / 0.008548367) * (1 + math.erf(((Q.N2_b05 - 0.435) / 0.008548367) / math.sqrt(2)))) / 0.0307106   # +2.5%  N2_b05 > 0.435
        - 0.02373053 * (0.6671377 * 0.5 * ((0.468 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.468 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2460222   # -2.4%  lep_iso < 0.468
        + 0.01991092 * (3.876948 * 0.5 * ((6.77 - Q.mass_2charged) / 3.876948) * (1 + math.erf(((6.77 - Q.mass_2charged) / 3.876948) / math.sqrt(2)))) / 2.843066   # +2.0%  mass_2charged < 6.77
        + 0.01740563 * (0.006229165 * 0.5 * ((0.00454 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.00454 - Q.lep_z) / 0.006229165) / math.sqrt(2)))) / 0.002059009   # +1.7%  lep_z < 0.00454
        - 0.01661826 * (36.85684 * 0.5 * ((75.9 - Q.sv_1_sd0_sum) / 36.85684) * (1 + math.erf(((75.9 - Q.sv_1_sd0_sum) / 36.85684) / math.sqrt(2)))) / 42.15631   # -1.7%  sv_1_sd0_sum < 75.9
        - 0.01499591 * (4.06561 * 0.5 * ((24.4 - Q.mass_2charged) / 4.06561) * (1 + math.erf(((24.4 - Q.mass_2charged) / 4.06561) / math.sqrt(2)))) / 14.83633   # -1.5%  mass_2charged < 24.4
        - 0.01439761 * (0.8395288 * 0.5 * ((1.84 - Q.mass_2charged) / 0.8395288) * (1 + math.erf(((1.84 - Q.mass_2charged) / 0.8395288) / math.sqrt(2)))) / 0.4323863   # -1.4%  mass_2charged < 1.84
        + 0.01396487 * (0.09740396 * 0.5 * ((1.33 - Q.D2) / 0.09740396) * (1 + math.erf(((1.33 - Q.D2) / 0.09740396) / math.sqrt(2)))) / 0.1083425   # +1.4%  D2 < 1.33
        - 0.01375267 * (1.0 * 0.5 * ((3.01 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.01 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) / 2.623677   # -1.4%  lepsj_3_n_d3 < 3.01
        + 0.01357806 * (0.09724171 * 0.5 * ((3.23 - Q.pair_max_lnkt) / 0.09724171) * (1 + math.erf(((3.23 - Q.pair_max_lnkt) / 0.09724171) / math.sqrt(2)))) / 0.5825339   # +1.4%  pair_max_lnkt < 3.23
        - 0.01264975 * (5.468702 * 0.5 * ((Q.sd_mass - 89.7) / 5.468702) * (1 + math.erf(((Q.sd_mass - 89.7) / 5.468702) / math.sqrt(2)))) / 22.73502   # -1.3%  sd_mass > 89.7
        - 0.01209957 * (0.01206829 * 0.5 * ((0.895 - Q.tau43) / 0.01206829) * (1 + math.erf(((0.895 - Q.tau43) / 0.01206829) / math.sqrt(2)))) / 0.1033443   # -1.2%  tau43 < 0.895
        + 0.01113096 * (2.35787 * 0.5 * ((Q.sj3_pair_mass_min - 26.6) / 2.35787) * (1 + math.erf(((Q.sj3_pair_mass_min - 26.6) / 2.35787) / math.sqrt(2)))) / 13.68926   # +1.1%  sj3_pair_mass_min > 26.6
        - 0.009392643 * (0.179073 * 0.5 * ((0.521 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.521 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2)))) / 0.5499635   # -0.9%  lep_z < 0.521 and dc_n > 1
        - 0.009057785 * (0.1723618 * 0.5 * ((Q.ktd_ln_d34 - -9.52) / 0.1723618) * (1 + math.erf(((Q.ktd_ln_d34 - -9.52) / 0.1723618) / math.sqrt(2)))) / 0.6388387   # -0.9%  ktd_ln_d34 > -9.52
        + 0.008931331 * (0.009766867 * 0.5 * ((0.097 - Q.tau2) / 0.009766867) * (1 + math.erf(((0.097 - Q.tau2) / 0.009766867) / math.sqrt(2)))) / 0.03161576   # +0.9%  tau2 < 0.097
        - 0.008905058 * (0.179073 * 0.5 * ((0.519 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.519 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_n - 0.999) / 1.0) * (1 + math.erf(((Q.ak02_n - 0.999) / 1.0) / math.sqrt(2)))) / 0.7085884   # -0.9%  lep_z < 0.519 and ak02_n > 0.999
        + 0.008252475 * (3.260893 * 0.5 * ((87.5 - Q.mass_top20) / 3.260893) * (1 + math.erf(((87.5 - Q.mass_top20) / 3.260893) / math.sqrt(2)))) / 9.369438   # +0.8%  mass_top20 < 87.5
        + 0.008025774 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.01) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.01) / 1.0) / math.sqrt(2)))) * (248.3135 * 0.5 * ((178.0 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((178.0 - Q.sip_3d_3) / 248.3135) / math.sqrt(2)))) / 90.34926   # +0.8%  n_s3d_above_3 > 4.01 and sip_3d_3 < 178
        + 0.007684188 * (0.8395288 * 0.5 * ((1.85 - Q.mass_2charged) / 0.8395288) * (1 + math.erf(((1.85 - Q.mass_2charged) / 0.8395288) / math.sqrt(2)))) * (0.04514027 * 0.5 * ((0.102 - Q.sjf_4_2_z_d3) / 0.04514027) * (1 + math.erf(((0.102 - Q.sjf_4_2_z_d3) / 0.04514027) / math.sqrt(2)))) / 0.03649935   # +0.8%  mass_2charged < 1.85 and sjf_4_2_z_d3 < 0.102
        - 0.00764064 * (6.622935 * 0.5 * ((80.4 - Q.dc_split1_kt) / 6.622935) * (1 + math.erf(((80.4 - Q.dc_split1_kt) / 6.622935) / math.sqrt(2)))) / 31.47491   # -0.8%  dc_split1_kt < 80.4
        + 0.007190291 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.9) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.9) / 1.0) / math.sqrt(2)))) / 1.277492   # +0.7%  n_s3d_above_3 > 3.9
        - 0.006895584 * (0.6262913 * 0.5 * ((5.49 - Q.sj3_mass3) / 0.6262913) * (1 + math.erf(((5.49 - Q.sj3_mass3) / 0.6262913) / math.sqrt(2)))) / 1.684959   # -0.7%  sj3_mass3 < 5.49
        + 0.006656787 * (12.30925 * 0.5 * ((39.4 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.4 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (12.86093 * 0.5 * ((Q.dc_split2_mass - 17.9) / 12.86093) * (1 + math.erf(((Q.dc_split2_mass - 17.9) / 12.86093) / math.sqrt(2)))) / 380.2071   # +0.7%  mass_displaced3 < 39.4 and dc_split2_mass > 17.9
        + 0.006473477 * (0.02680828 * 0.5 * ((Q.dc_1_z - 0.616) / 0.02680828) * (1 + math.erf(((Q.dc_1_z - 0.616) / 0.02680828) / math.sqrt(2)))) / 0.07847287   # +0.6%  dc_1_z > 0.616
        + 0.006376763 * (1.0 * 0.5 * ((0.983 - Q.sjq_3_3_nch) / 1.0) * (1 + math.erf(((0.983 - Q.sjq_3_3_nch) / 1.0) / math.sqrt(2)))) / 0.1132954   # +0.6%  sjq_3_3_nch < 0.983
        + 0.006374714 * (14.06404 * 0.5 * ((Q.mass_neutral - 26.3) / 14.06404) * (1 + math.erf(((Q.mass_neutral - 26.3) / 14.06404) / math.sqrt(2)))) / 20.89708   # +0.6%  mass_neutral > 26.3
        + 0.006236485 * (1.721646 * 0.5 * ((5.8 - Q.mass_2photon) / 1.721646) * (1 + math.erf(((5.8 - Q.mass_2photon) / 1.721646) / math.sqrt(2)))) / 2.962286   # +0.6%  mass_2photon < 5.8
        + 0.006078989 * (0.0321369 * 0.5 * ((0.854 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.854 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) * (1.0 * 0.5 * ((4.01 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((4.01 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2)))) / 0.241857   # +0.6%  nca_sj4_pairmax_over_mass < 0.854 and n_lund_kt_above_5 < 4.01
        - 0.005493004 * (21.5 * 0.5 * ((77.8 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((77.8 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 4.870391   # -0.5%  n_pairs_kt_above_1 < 77.8
        - 0.005386794 * (0.179073 * 0.5 * ((0.519 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.519 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.008350939 * 0.5 * ((0.888 - Q.tau54) / 0.008350939) * (1 + math.erf(((0.888 - Q.tau54) / 0.008350939) / math.sqrt(2)))) / 0.02238853   # -0.5%  lep_z < 0.519 and tau54 < 0.888
        + 0.005183093 * (0.179073 * 0.5 * ((0.517 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.517 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.02439358 * 0.5 * ((0.185 - Q.sj4_dr_min) / 0.02439358) * (1 + math.erf(((0.185 - Q.sj4_dr_min) / 0.02439358) / math.sqrt(2)))) / 0.03446705   # +0.5%  lep_z < 0.517 and sj4_dr_min < 0.185
        - 0.004326833 * (0.003950899 * 0.5 * ((Q.tau4 - 0.0507) / 0.003950899) * (1 + math.erf(((Q.tau4 - 0.0507) / 0.003950899) / math.sqrt(2)))) / 0.004077147   # -0.4%  tau4 > 0.0507
        - 0.004247199 * (0.009691066 * 0.5 * ((0.111 - Q.dr_0) / 0.009691066) * (1 + math.erf(((0.111 - Q.dr_0) / 0.009691066) / math.sqrt(2)))) / 0.02671678   # -0.4%  dr_0 < 0.111
        + 0.003950949 * (14.63984 * 0.5 * ((414.0 - Q.sum_pt_top10) / 14.63984) * (1 + math.erf(((414.0 - Q.sum_pt_top10) / 14.63984) / math.sqrt(2)))) / 22.70542   # +0.4%  sum_pt_top10 < 414
        + 0.00390975 * (0.0164642 * 0.5 * ((0.616 - Q.nca_sj4_pairmax_over_mass) / 0.0164642) * (1 + math.erf(((0.616 - Q.nca_sj4_pairmax_over_mass) / 0.0164642) / math.sqrt(2)))) / 0.00887786   # +0.4%  nca_sj4_pairmax_over_mass < 0.616
        - 0.003097723 * (0.009766867 * 0.5 * ((0.0955 - Q.tau2) / 0.009766867) * (1 + math.erf(((0.0955 - Q.tau2) / 0.009766867) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_1_n_disp3 - 0.903) / 1.0) * (1 + math.erf(((Q.ak02_1_n_disp3 - 0.903) / 1.0) / math.sqrt(2)))) / 0.01471396   # -0.3%  tau2 < 0.0955 and ak02_1_n_disp3 > 0.903
        + 0.0027486 * (0.1723618 * 0.5 * ((Q.ktd_ln_d34 - -9.54) / 0.1723618) * (1 + math.erf(((Q.ktd_ln_d34 - -9.54) / 0.1723618) / math.sqrt(2)))) * (0.01029976 * 0.5 * ((0.103 - Q.pz_lnkt3) / 0.01029976) * (1 + math.erf(((0.103 - Q.pz_lnkt3) / 0.01029976) / math.sqrt(2)))) / 0.0301048   # +0.3%  ktd_ln_d34 > -9.54 and pz_lnkt3 < 0.103
        + 0.002306547 * (0.03292552 * 0.5 * ((0.0563 - Q.sj2_zsoft) / 0.03292552) * (1 + math.erf(((0.0563 - Q.sj2_zsoft) / 0.03292552) / math.sqrt(2)))) / 0.001460791   # +0.2%  sj2_zsoft < 0.0563
        + 0.002126132 * (0.008554338 * 0.5 * ((0.0788 - Q.pz_lnd0) / 0.008554338) * (1 + math.erf(((0.0788 - Q.pz_lnd0) / 0.008554338) / math.sqrt(2)))) / 0.009425709   # +0.2%  pz_lnd0 < 0.0788
        + 0.001469441 * (4.06561 * 0.5 * ((24.9 - Q.mass_2charged) / 4.06561) * (1 + math.erf(((24.9 - Q.mass_2charged) / 4.06561) / math.sqrt(2)))) * (0.01273628 * 0.5 * ((Q.sdb_5_z - 0.0394) / 0.01273628) * (1 + math.erf(((Q.sdb_5_z - 0.0394) / 0.01273628) / math.sqrt(2)))) / 0.2814875   # +0.1%  mass_2charged < 24.9 and sdb_5_z > 0.0394
    )
    return z


def neuron_116(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.4e-07
    )
    return z


def neuron_117(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.86e-06
    )
    return z


def neuron_118(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.23e-06
    )
    return z


def neuron_119(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.35e-07
    )
    return z


def neuron_120(Q):
    # scale S = 9.298; each line: share * term / its average size
    z = 9.297555 * (0.03947274
        + 0.1456794 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.4) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.4) / 5.26174) / math.sqrt(2)))) / 24.10075   # +14.6%  mres_sd_mass_b2z01 > 91.4
        - 0.1295323 * (5.55459 * 0.5 * ((Q.mres_sd_mass_b2z01 - 96.7) / 5.55459) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 96.7) / 5.55459) / math.sqrt(2)))) / 20.80024   # -13.0%  mres_sd_mass_b2z01 > 96.7
        + 0.05440043 * (9.37985 * 0.5 * ((75.7 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((75.7 - Q.mass_neutral) / 9.37985) / math.sqrt(2)))) / 32.21599   # +5.4%  mass_neutral < 75.7
        - 0.05102266 * (2.752946 * 0.5 * ((3.66 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.66 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (1.0 * 0.5 * ((9.97 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((9.97 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 14.46299   # -5.1%  lep_ptrel < 3.66 and n_s3d_above_3 < 9.97
        - 0.05044939 * (2.752946 * 0.5 * ((3.47 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.47 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) / 2.021793   # -5.0%  lep_ptrel < 3.47
        - 0.04416207 * (5.892378 * 0.5 * ((12.4 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.4 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) / 8.830092   # -4.4%  lep_ptrel < 12.4
        - 0.03436062 * (0.179073 * 0.5 * ((0.518 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.518 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (9.37985 * 0.5 * ((76.1 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((76.1 - Q.mass_neutral) / 9.37985) / math.sqrt(2)))) / 12.83011   # -3.4%  lep_z < 0.518 and mass_neutral < 76.1
        + 0.03395256 * (2.752946 * 0.5 * ((3.55 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.55 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (401.9953 * 0.5 * ((574.0 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((574.0 - Q.sip_3d_3) / 401.9953) / math.sqrt(2)))) / 974.3079   # +3.4%  lep_ptrel < 3.55 and sip_3d_3 < 574
        - 0.03331774 * (0.03060878 * 0.5 * ((0.0646 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0646 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) / 0.04773089   # -3.3%  lepsj_2_dr < 0.0646
        - 0.03030957 * (3.529599 * 0.5 * ((110.0 - Q.mass_top30) / 3.529599) * (1 + math.erf(((110.0 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) / 15.65583   # -3.0%  mass_top30 < 110
        - 0.02424504 * (0.03060878 * 0.5 * ((0.0687 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0687 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.25 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.25 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 0.1273557   # -2.4%  lepsj_2_dr < 0.0687 and jd_3d_6 < 5.25
        - 0.0218628 * (5.892378 * 0.5 * ((12.5 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.5 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (1.929175 * 0.5 * ((3.85 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((3.85 - Q.jd_3d_5) / 1.929175) / math.sqrt(2)))) / 10.58701   # -2.2%  lep_ptrel < 12.5 and jd_3d_5 < 3.85
        + 0.02006401 * (0.03429006 * 0.5 * ((0.0797 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.0797 - Q.lep_dr) / 0.03429006) / math.sqrt(2)))) / 0.04722689   # +2.0%  lep_dr < 0.0797
        - 0.01879939 * (5.110145 * 0.5 * ((131.0 - Q.mass) / 5.110145) * (1 + math.erf(((131.0 - Q.mass) / 5.110145) / math.sqrt(2)))) / 24.44592   # -1.9%  mass < 131
        + 0.0186277 * (11.4891 * 0.5 * ((32.8 - Q.mass_displaced5) / 11.4891) * (1 + math.erf(((32.8 - Q.mass_displaced5) / 11.4891) / math.sqrt(2)))) / 26.97696   # +1.9%  mass_displaced5 < 32.8
        + 0.01839899 * (2.752946 * 0.5 * ((3.68 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.68 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (2.0 * 0.5 * ((Q.n_photon - 5.04) / 2.0) * (1 + math.erf(((Q.n_photon - 5.04) / 2.0) / math.sqrt(2)))) / 26.43981   # +1.8%  lep_ptrel < 3.68 and n_photon > 5.04
        + 0.01801489 * (2.41177 * 0.5 * ((2.81 - Q.lep_iso) / 2.41177) * (1 + math.erf(((2.81 - Q.lep_iso) / 2.41177) / math.sqrt(2)))) / 1.836561   # +1.8%  lep_iso < 2.81
        + 0.01743291 * (0.0005676896 * 0.5 * ((0.00228 - Q.e3) / 0.0005676896) * (1 + math.erf(((0.00228 - Q.e3) / 0.0005676896) / math.sqrt(2)))) / 0.001237278   # +1.7%  e3 < 0.00228
        - 0.01721193 * (3.277554 * 0.5 * ((68.5 - Q.mass_charged) / 3.277554) * (1 + math.erf(((68.5 - Q.mass_charged) / 3.277554) / math.sqrt(2)))) / 13.01048   # -1.7%  mass_charged < 68.5
        - 0.01603605 * (4.146938 * 0.5 * ((110.0 - Q.mass) / 4.146938) * (1 + math.erf(((110.0 - Q.mass) / 4.146938) / math.sqrt(2)))) / 11.92768   # -1.6%  mass < 110
        + 0.0156469 * (2.41177 * 0.5 * ((2.83 - Q.lep_iso) / 2.41177) * (1 + math.erf(((2.83 - Q.lep_iso) / 2.41177) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.58 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.58 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 5.251911   # +1.6%  lep_iso < 2.83 and jd_3d_6 < 5.58
        - 0.01540577 * (8.0 * 0.5 * ((62.4 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.4 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) / 16.87114   # -1.5%  n_pairs_kt_above_3 < 62.4
        + 0.01403221 * (3.277554 * 0.5 * ((67.9 - Q.mass_charged) / 3.277554) * (1 + math.erf(((67.9 - Q.mass_charged) / 3.277554) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.6 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.6 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 42.22176   # +1.4%  mass_charged < 67.9 and jd_3d_6 < 5.6
        + 0.01329103 * (0.179073 * 0.5 * ((0.513 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.513 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (4.4703e-08 * 0.5 * ((Q.mass_displaced3 - -0.485) / 4.4703e-08) * (1 + math.erf(((Q.mass_displaced3 - -0.485) / 4.4703e-08) / math.sqrt(2)))) / 3.80228   # +1.3%  lep_z < 0.513 and mass_displaced3 > -0.485
        + 0.01177894 * (5.892378 * 0.5 * ((11.9 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((11.9 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (1.5 * 0.5 * ((15.9 - Q.sdb_2_n) / 1.5) * (1 + math.erf(((15.9 - Q.sdb_2_n) / 1.5) / math.sqrt(2)))) / 43.28669   # +1.2%  lep_ptrel < 11.9 and sdb_2_n < 15.9
        + 0.01153388 * (3.529599 * 0.5 * ((109.0 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.0 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.33 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.33 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 44.68204   # +1.2%  mass_top30 < 109 and jd_3d_6 < 5.33
        - 0.01047621 * (3.705078 * 0.5 * ((5.86 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.86 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) / 2.790922   # -1.0%  max_abs_d0 < 5.86
        - 0.01022288 * (3.705078 * 0.5 * ((5.8 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ismuon_0 - 1.0) / 1.0) * (1 + math.erf(((Q.ismuon_0 - 1.0) / 1.0) / math.sqrt(2)))) / 0.3927593   # -1.0%  max_abs_d0 < 5.8 and ismuon_0 > 1
        - 0.009745013 * (0.3071165 * 0.5 * ((Q.lund3_lndelta - -2.49) / 0.3071165) * (1 + math.erf(((Q.lund3_lndelta - -2.49) / 0.3071165) / math.sqrt(2)))) / 1.015749   # -1.0%  lund3_lndelta > -2.49
        + 0.009204602 * (5.110145 * 0.5 * ((132.0 - Q.mass) / 5.110145) * (1 + math.erf(((132.0 - Q.mass) / 5.110145) / math.sqrt(2)))) * (1.929175 * 0.5 * ((4.2 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((4.2 - Q.jd_3d_5) / 1.929175) / math.sqrt(2)))) / 47.02214   # +0.9%  mass < 132 and jd_3d_5 < 4.2
        + 0.008856444 * (0.008162106 * 0.5 * ((0.158 - Q.pz_lnkt0) / 0.008162106) * (1 + math.erf(((0.158 - Q.pz_lnkt0) / 0.008162106) / math.sqrt(2)))) / 0.02859141   # +0.9%  pz_lnkt0 < 0.158
        + 0.005967311 * (8.0 * 0.5 * ((62.9 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.9 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) * (0.08055962 * 0.5 * ((0.272 - Q.sjf_2_1_z_d3) / 0.08055962) * (1 + math.erf(((0.272 - Q.sjf_2_1_z_d3) / 0.08055962) / math.sqrt(2)))) / 3.626235   # +0.6%  n_pairs_kt_above_3 < 62.9 and sjf_2_1_z_d3 < 0.272
        + 0.005950438 * (0.02757255 * 0.5 * ((0.69 - Q.tau32) / 0.02757255) * (1 + math.erf(((0.69 - Q.tau32) / 0.02757255) / math.sqrt(2)))) / 0.0852458   # +0.6%  tau32 < 0.69
        + 0.005882933 * (3.705078 * 0.5 * ((5.84 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.84 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.989 - Q.n_electron) / 1.0) * (1 + math.erf(((0.989 - Q.n_electron) / 1.0) / math.sqrt(2)))) / 1.703953   # +0.6%  max_abs_d0 < 5.84 and n_electron < 0.989
        - 0.005868685 * (1.0 * 0.5 * ((-0.00345 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((-0.00345 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2)))) / 0.03925498   # -0.6%  kt2_1_n_lep < -0.00345
        + 0.005597918 * (3.529599 * 0.5 * ((108.0 - Q.mass_top30) / 3.529599) * (1 + math.erf(((108.0 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (0.0139093 * 0.5 * ((0.128 - Q.z_neutral_had) / 0.0139093) * (1 + math.erf(((0.128 - Q.z_neutral_had) / 0.0139093) / math.sqrt(2)))) / 0.5894332   # +0.6%  mass_top30 < 108 and z_neutral_had < 0.128
        + 0.00487348 * (0.002623498 * 0.5 * ((Q.e3_b05 - 0.00575) / 0.002623498) * (1 + math.erf(((Q.e3_b05 - 0.00575) / 0.002623498) / math.sqrt(2)))) / 0.002531366   # +0.5%  e3_b05 > 0.00575
        - 0.004491927 * (21.5 * 0.5 * ((81.9 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((81.9 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 5.480832   # -0.4%  n_pairs_kt_above_1 < 81.9
        - 0.004440284 * (0.08036477 * 0.5 * ((Q.pair_mean_lndelta - -1.72) / 0.08036477) * (1 + math.erf(((Q.pair_mean_lndelta - -1.72) / 0.08036477) / math.sqrt(2)))) / 0.05299587   # -0.4%  pair_mean_lndelta > -1.72
        + 0.003727403 * (0.03060878 * 0.5 * ((0.0623 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0623 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (0.2385164 * 0.5 * ((Q.ak02_dr23 - 0.0104) / 0.2385164) * (1 + math.erf(((Q.ak02_dr23 - 0.0104) / 0.2385164) / math.sqrt(2)))) / 0.009266238   # +0.4%  lepsj_2_dr < 0.0623 and ak02_dr23 > 0.0104
        + 0.003397441 * (5.892378 * 0.5 * ((12.2 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.2 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.09724841 * 0.5 * ((0.432 - Q.sjq_3_1_k05) / 0.09724841) * (1 + math.erf(((0.432 - Q.sjq_3_1_k05) / 0.09724841) / math.sqrt(2)))) / 3.928842   # +0.3%  lep_ptrel < 12.2 and sjq_3_1_k05 < 0.432
        + 0.003101263 * (0.01151941 * 0.5 * ((Q.N2 - 0.327) / 0.01151941) * (1 + math.erf(((Q.N2 - 0.327) / 0.01151941) / math.sqrt(2)))) / 0.02529312   # +0.3%  N2 > 0.327
        - 0.002776102 * (0.00189604 * 0.5 * ((0.0142 - Q.dr_min_012) / 0.00189604) * (1 + math.erf(((0.0142 - Q.dr_min_012) / 0.00189604) / math.sqrt(2)))) / 0.003083746   # -0.3%  dr_min_012 < 0.0142
        + 0.002406256 * (0.6671377 * 0.5 * ((0.442 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.442 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.000627 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.000627 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2)))) / 0.003364255   # +0.2%  lep_iso < 0.442 and dc_1_n_lep < 0.000627
        - 0.002202783 * (9.37985 * 0.5 * ((77.9 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((77.9 - Q.mass_neutral) / 9.37985) / math.sqrt(2)))) * (0.03369438 * 0.5 * ((Q.sj3_dr_max - 0.542) / 0.03369438) * (1 + math.erf(((Q.sj3_dr_max - 0.542) / 0.03369438) / math.sqrt(2)))) / 2.303768   # -0.2%  mass_neutral < 77.9 and sj3_dr_max > 0.542
        - 0.002056493 * (0.009387389 * 0.5 * ((0.0394 - Q.pz_lnd2) / 0.009387389) * (1 + math.erf(((0.0394 - Q.pz_lnd2) / 0.009387389) / math.sqrt(2)))) / 0.004674903   # -0.2%  pz_lnd2 < 0.0394
        + 0.001597614 * (8.0 * 0.5 * ((63.5 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((63.5 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.00191 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((0.00191 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.6402545   # +0.2%  n_pairs_kt_above_3 < 63.5 and dc_2_n_lep < 0.00191
        - 0.001584658 * (0.03214999 * 0.5 * ((0.339 - Q.N2_b05) / 0.03214999) * (1 + math.erf(((0.339 - Q.N2_b05) / 0.03214999) / math.sqrt(2)))) / 0.005497553   # -0.2%  N2_b05 < 0.339
        + 0.001574996 * (1.0 * 0.5 * ((-0.00233 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((-0.00233 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2)))) * (0.0812128 * 0.5 * ((Q.sjq_2_prod_k05 - -0.239) / 0.0812128) * (1 + math.erf(((Q.sjq_2_prod_k05 - -0.239) / 0.0812128) / math.sqrt(2)))) / 0.008135339   # +0.2%  kt2_1_n_lep < -0.00233 and sjq_2_prod_k05 > -0.239
        + 0.001464742 * (0.179073 * 0.5 * ((0.516 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.516 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.09432096 * 0.5 * ((-0.289 - Q.sjq_3_3_k1) / 0.09432096) * (1 + math.erf(((-0.289 - Q.sjq_3_3_k1) / 0.09432096) / math.sqrt(2)))) / 0.01810973   # +0.1%  lep_z < 0.516 and sjq_3_3_k1 < -0.289
        + 0.001279871 * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.402) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.402) / 0.179816) / math.sqrt(2)))) / 0.03595067   # +0.1%  sjq_3_3_k1 > 0.402
        - 0.001223682 * (10.67504 * 0.5 * ((Q.mass_top10 - 106.0) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 106.0) / 10.67504) / math.sqrt(2)))) * (0.04703017 * 0.5 * ((0.672 - Q.planar_flow) / 0.04703017) * (1 + math.erf(((0.672 - Q.planar_flow) / 0.04703017) / math.sqrt(2)))) / 0.5148076   # -0.1%  mass_top10 > 106 and planar_flow < 0.672
        - 0.0004593564 * (3.529599 * 0.5 * ((109.0 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.0 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (0.005973664 * 0.5 * ((Q.sv_2_z - 0.0116) / 0.005973664) * (1 + math.erf(((Q.sv_2_z - 0.0116) / 0.005973664) / math.sqrt(2)))) / 0.02905368   # -0.0%  mass_top30 < 109 and sv_2_z > 0.0116
    )
    return z


def neuron_121(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.73e-06
    )
    return z


def neuron_122(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.39e-06
    )
    return z


def neuron_123(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.0224
    )
    return z


def neuron_124(Q):
    # scale S = 8.874; each line: share * term / its average size
    z = 8.873819 * (-0.004631602
        + 0.112851 * (18.42178 * 0.5 * ((183.0 - Q.mass) / 18.42178) * (1 + math.erf(((183.0 - Q.mass) / 18.42178) / math.sqrt(2)))) / 69.54302   # +11.3%  mass < 183
        - 0.1077981 * (4.488781 * 0.5 * ((125.0 - Q.mres_sd_mass_b0z005) / 4.488781) * (1 + math.erf(((125.0 - Q.mres_sd_mass_b0z005) / 4.488781) / math.sqrt(2)))) / 27.72699   # -10.8%  mres_sd_mass_b0z005 < 125
        + 0.1046879 * (18.66842 * 0.5 * ((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) / 56.99275   # +10.5%  mres_sd_mass_b0z005 < 160
        - 0.08525475 * (1.234622 * 0.5 * ((1.52 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.52 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) / 0.971162   # -8.5%  lep_iso < 1.52
        + 0.0709053 * (0.1060766 * 0.5 * ((0.219 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.219 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) / 0.1651445   # +7.1%  lep_z < 0.219
        + 0.0530201 * (5.31786 * 0.5 * ((91.5 - Q.mres_sd_mass_b0z005) / 5.31786) * (1 + math.erf(((91.5 - Q.mres_sd_mass_b0z005) / 5.31786) / math.sqrt(2)))) / 10.22806   # +5.3%  mres_sd_mass_b0z005 < 91.5
        - 0.05276496 * (137.5 * 0.5 * ((700.0 - Q.n_pairs_kt_above_1) / 137.5) * (1 + math.erf(((700.0 - Q.n_pairs_kt_above_1) / 137.5) / math.sqrt(2)))) / 393.4678   # -5.3%  n_pairs_kt_above_1 < 700
        + 0.04927657 * (0.01610679 * 0.5 * ((0.0268 - Q.lepsj_3_dr) / 0.01610679) * (1 + math.erf(((0.0268 - Q.lepsj_3_dr) / 0.01610679) / math.sqrt(2)))) / 0.01926306   # +4.9%  lepsj_3_dr < 0.0268
        - 0.03967457 * (1.5 * 0.5 * ((6.84 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((6.84 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2)))) / 3.52065   # -4.0%  n_s3d_above_3 < 6.84
        + 0.038991 * (1.0 * 0.5 * ((0.933 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.933 - Q.n_lepton) / 1.0) / math.sqrt(2)))) / 0.4607178   # +3.9%  n_lepton < 0.933
        - 0.02590957 * (43.0 * 0.5 * ((362.0 - Q.n_pairs_kt_above_1) / 43.0) * (1 + math.erf(((362.0 - Q.n_pairs_kt_above_1) / 43.0) / math.sqrt(2)))) / 126.3279   # -2.6%  n_pairs_kt_above_1 < 362
        - 0.02462728 * (2.5 * 0.5 * ((30.6 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((30.6 - Q.n_for_90pct) / 2.5) / math.sqrt(2)))) / 11.38219   # -2.5%  n_for_90pct < 30.6
        - 0.02156172 * (6.235814 * 0.5 * ((75.3 - Q.mres_sd_mass_b0z005) / 6.235814) * (1 + math.erf(((75.3 - Q.mres_sd_mass_b0z005) / 6.235814) / math.sqrt(2)))) / 5.923679   # -2.2%  mres_sd_mass_b0z005 < 75.3
        - 0.01700913 * (1.0 * 0.5 * ((0.969 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.969 - Q.n_lepton) / 1.0) / math.sqrt(2)))) * (0.01040872 * 0.5 * ((0.0177 - Q.sdb_4_z) / 0.01040872) * (1 + math.erf(((0.0177 - Q.sdb_4_z) / 0.01040872) / math.sqrt(2)))) / 0.006185897   # -1.7%  n_lepton < 0.969 and sdb_4_z < 0.0177
        + 0.01564993 * (2.5 * 0.5 * ((31.3 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((31.3 - Q.n_for_90pct) / 2.5) / math.sqrt(2)))) * (0.1673111 * 0.5 * ((Q.sjq_2_prod_k05 - -0.501) / 0.1673111) * (1 + math.erf(((Q.sjq_2_prod_k05 - -0.501) / 0.1673111) / math.sqrt(2)))) / 5.361955   # +1.6%  n_for_90pct < 31.3 and sjq_2_prod_k05 > -0.501
        - 0.01544955 * (0.01710086 * 0.5 * ((0.0594 - Q.z_displaced3) / 0.01710086) * (1 + math.erf(((0.0594 - Q.z_displaced3) / 0.01710086) / math.sqrt(2)))) / 0.02384287   # -1.5%  z_displaced3 < 0.0594
        + 0.01436136 * (0.0001198329 * 0.5 * ((0.000264 - Q.e3_b2) / 0.0001198329) * (1 + math.erf(((0.000264 - Q.e3_b2) / 0.0001198329) / math.sqrt(2)))) / 0.0001767546   # +1.4%  e3_b2 < 0.000264
        - 0.01376883 * (0.0439522 * 0.5 * ((0.0969 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.0969 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) / 0.1141889   # -1.4%  sjq_2_prod_k1 < 0.0969
        - 0.01285086 * (5.197 * 0.5 * ((97.7 - Q.mass) / 5.197) * (1 + math.erf(((97.7 - Q.mass) / 5.197) / math.sqrt(2)))) / 7.172089   # -1.3%  mass < 97.7
        - 0.01246081 * (0.04339825 * 0.5 * ((0.536 - Q.sj2_dr) / 0.04339825) * (1 + math.erf(((0.536 - Q.sj2_dr) / 0.04339825) / math.sqrt(2)))) / 0.1650372   # -1.2%  sj2_dr < 0.536
        - 0.01202467 * (0.02246636 * 0.5 * ((Q.z_photon - 0.12) / 0.02246636) * (1 + math.erf(((Q.z_photon - 0.12) / 0.02246636) / math.sqrt(2)))) / 0.1394833   # -1.2%  z_photon > 0.12
        + 0.01112459 * (5.898438 * 0.5 * ((8.78 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((8.78 - Q.max_abs_d0) / 5.898438) / math.sqrt(2)))) / 4.426799   # +1.1%  max_abs_d0 < 8.78
        - 0.01032827 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.33) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.33) / 1.0) / math.sqrt(2)))) / 1.532629   # -1.0%  n_s3d_above_3 > 3.33
        + 0.008641144 * (0.01670814 * 0.5 * ((Q.z_displaced5 - 0.0565) / 0.01670814) * (1 + math.erf(((Q.z_displaced5 - 0.0565) / 0.01670814) / math.sqrt(2)))) / 0.06183866   # +0.9%  z_displaced5 > 0.0565
        - 0.00834005 * (3.244489 * 0.5 * ((69.0 - Q.sj4_pair_mass_max) / 3.244489) * (1 + math.erf(((69.0 - Q.sj4_pair_mass_max) / 3.244489) / math.sqrt(2)))) / 5.441771   # -0.8%  sj4_pair_mass_max < 69
        + 0.007695357 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2)))) * (6.000774 * 0.5 * ((16.5 - Q.jd_3d_4) / 6.000774) * (1 + math.erf(((16.5 - Q.jd_3d_4) / 6.000774) / math.sqrt(2)))) / 192.3583   # +0.8%  mass < 118 and jd_3d_4 < 16.5
        - 0.00752328 * (0.1486471 * 0.5 * ((0.338 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.338 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.02653106 * 0.5 * ((Q.z_displaced5 - 0.093) / 0.02653106) * (1 + math.erf(((Q.z_displaced5 - 0.093) / 0.02653106) / math.sqrt(2)))) / 0.01200723   # -0.8%  lep_z < 0.338 and z_displaced5 > 0.093
        - 0.006459152 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.21) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.21) / 1.0) / math.sqrt(2)))) * (401.9953 * 0.5 * ((580.0 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((580.0 - Q.sip_3d_3) / 401.9953) / math.sqrt(2)))) / 625.0528   # -0.6%  n_s3d_above_3 > 3.21 and sip_3d_3 < 580
        - 0.005869185 * (0.03580654 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.281) / 0.03580654) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.281) / 0.03580654) / math.sqrt(2)))) / 0.1374198   # -0.6%  sjq_2_sumabs_k1 > 0.281
        - 0.00517108 * (0.02550179 * 0.5 * ((0.177 - Q.sdb_2_z) / 0.02550179) * (1 + math.erf(((0.177 - Q.sdb_2_z) / 0.02550179) / math.sqrt(2)))) / 0.01820922   # -0.5%  sdb_2_z < 0.177
        - 0.004333271 * (0.02505229 * 0.5 * ((Q.sdb_2_z - 0.347) / 0.02505229) * (1 + math.erf(((Q.sdb_2_z - 0.347) / 0.02505229) / math.sqrt(2)))) / 0.06192055   # -0.4%  sdb_2_z > 0.347
        - 0.004330143 * (1.0 * 0.5 * ((5.99 - Q.n_dr_0p1_0p2) / 1.0) * (1 + math.erf(((5.99 - Q.n_dr_0p1_0p2) / 1.0) / math.sqrt(2)))) / 0.8538867   # -0.4%  n_dr_0p1_0p2 < 5.99
        - 0.003809294 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.09) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.09) / 1.0) / math.sqrt(2)))) * (5.898438 * 0.5 * ((9.91 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((9.91 - Q.max_abs_d0) / 5.898438) / math.sqrt(2)))) / 5.208472   # -0.4%  n_s3d_above_3 > 3.09 and max_abs_d0 < 9.91
        + 0.003701739 * (5.001896 * 0.5 * ((-8.49 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((-8.49 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) / 0.2906952   # +0.4%  lepsj_3_maxsd0 < -8.49
        + 0.002927441 * (15.88428 * 0.5 * ((44.7 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((44.7 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.05556614 * 0.5 * ((Q.jet_charge_k05 - 0.0289) / 0.05556614) * (1 + math.erf(((Q.jet_charge_k05 - 0.0289) / 0.05556614) / math.sqrt(2)))) / 6.069528   # +0.3%  lep_ptrel < 44.7 and jet_charge_k05 > 0.0289
        + 0.002492545 * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 0.928) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 0.928) / 1.0) / math.sqrt(2)))) / 0.5241325   # +0.2%  lepsj_2_n_d3 > 0.928
        + 0.002282414 * (0.1605002 * 0.5 * ((Q.pair_max_lnm2 - 7.34) / 0.1605002) * (1 + math.erf(((Q.pair_max_lnm2 - 7.34) / 0.1605002) / math.sqrt(2)))) / 0.1044007   # +0.2%  pair_max_lnm2 > 7.34
        + 0.001935119 * (2.0 * 0.5 * ((Q.sjf_2_1_n_d3 - 7.22) / 2.0) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 7.22) / 2.0) / math.sqrt(2)))) / 0.1010112   # +0.2%  sjf_2_1_n_d3 > 7.22
        - 0.001294247 * (0.2857178 * 0.5 * ((Q.dc_3_charge - 0.642) / 0.2857178) * (1 + math.erf(((Q.dc_3_charge - 0.642) / 0.2857178) / math.sqrt(2)))) / 0.02065632   # -0.1%  dc_3_charge > 0.642
        - 0.0008436088 * (0.07655927 * 0.5 * ((Q.sj3_dr12 - 0.615) / 0.07655927) * (1 + math.erf(((Q.sj3_dr12 - 0.615) / 0.07655927) / math.sqrt(2)))) / 0.004205636   # -0.1%  sj3_dr12 > 0.615
    )
    return z


def neuron_125(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.19e-06
    )
    return z


def neuron_126(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.63e-06
    )
    return z


def neuron_127(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.01e-07
    )
    return z


def class_token(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q), neuron_16(Q), neuron_17(Q), neuron_18(Q), neuron_19(Q), neuron_20(Q), neuron_21(Q), neuron_22(Q), neuron_23(Q), neuron_24(Q), neuron_25(Q), neuron_26(Q), neuron_27(Q), neuron_28(Q), neuron_29(Q), neuron_30(Q), neuron_31(Q), neuron_32(Q), neuron_33(Q), neuron_34(Q), neuron_35(Q), neuron_36(Q), neuron_37(Q), neuron_38(Q), neuron_39(Q), neuron_40(Q), neuron_41(Q), neuron_42(Q), neuron_43(Q), neuron_44(Q), neuron_45(Q), neuron_46(Q), neuron_47(Q), neuron_48(Q), neuron_49(Q), neuron_50(Q), neuron_51(Q), neuron_52(Q), neuron_53(Q), neuron_54(Q), neuron_55(Q), neuron_56(Q), neuron_57(Q), neuron_58(Q), neuron_59(Q), neuron_60(Q), neuron_61(Q), neuron_62(Q), neuron_63(Q), neuron_64(Q), neuron_65(Q), neuron_66(Q), neuron_67(Q), neuron_68(Q), neuron_69(Q), neuron_70(Q), neuron_71(Q), neuron_72(Q), neuron_73(Q), neuron_74(Q), neuron_75(Q), neuron_76(Q), neuron_77(Q), neuron_78(Q), neuron_79(Q), neuron_80(Q), neuron_81(Q), neuron_82(Q), neuron_83(Q), neuron_84(Q), neuron_85(Q), neuron_86(Q), neuron_87(Q), neuron_88(Q), neuron_89(Q), neuron_90(Q), neuron_91(Q), neuron_92(Q), neuron_93(Q), neuron_94(Q), neuron_95(Q), neuron_96(Q), neuron_97(Q), neuron_98(Q), neuron_99(Q), neuron_100(Q), neuron_101(Q), neuron_102(Q), neuron_103(Q), neuron_104(Q), neuron_105(Q), neuron_106(Q), neuron_107(Q), neuron_108(Q), neuron_109(Q), neuron_110(Q), neuron_111(Q), neuron_112(Q), neuron_113(Q), neuron_114(Q), neuron_115(Q), neuron_116(Q), neuron_117(Q), neuron_118(Q), neuron_119(Q), neuron_120(Q), neuron_121(Q), neuron_122(Q), neuron_123(Q), neuron_124(Q), neuron_125(Q), neuron_126(Q), neuron_127(Q)]


H_AVG = [0.005210000000000001, 5.319999999999999e-06, 5.4499999999999995e-06, 8.389999999999998e-06, 2.7899999999999995e-06, 2.1599999999999996e-06, 1.7700000000000003e-05, 6.429999999999999e-06, 2.919999999999999e-06, 2.44e-06, 1.5699999999999995e-05, 1.89e-05, 5.100000000000001e-06, 4.420000000000002e-06, 2.84e-06, 6.45e-06, 1.2669106801592802, 1.0399999999999997e-05, 1.078519199759497, 1.4099999999999998e-06, 8.210000000000001e-07, 0.0016800000000000003, 1.05e-06, 6.979999999999999e-07, 0.928680698007098, 0.030499999999999992, 3.2300000000000004e-06, 6.299999999999999e-06, 9.61e-06, 9.180000000000002e-06, 8.6e-06, 3.51e-06, 1.4400000000000001e-05, 2.56e-06, 1.3899999999999999e-05, 6.740000000000001e-06, 4.84e-06, 4.350000000000001e-06, 9.410000000000001e-06, 8.269999999999999e-06, 7.83e-07, 4.1200000000000005e-08, 8.780000000000002e-06, 9.400000000000001e-06, 3.3300000000000003e-06, 6.849999999999999e-06, 1.31e-07, 2.46e-07, 8.799999999999999e-06, 2.2399999999999997e-06, 5.1700000000000005e-06, 2.02e-06, 0.8173106260448579, 1.9000000000000002e-06, 1.1900000000000001e-05, 2.07e-05, 7.129999999999999e-07, 1.95e-06, 3.6100000000000006e-06, 6.030000000000001e-06, 4.670000000000001e-06, 5.839999999999998e-06, 9.41e-07, 8.31e-06, 1.25e-06, 5.789999999999998e-06, 2.7099999999999995e-06, 5.230000000000001e-06, 0.829558851257542, 6.9200000000000015e-06, 0.2617462014715054, 3.0299999999999994e-06, 1.16e-06, 1.7000000000000003e-05, 4.49e-06, 1.97e-06, 5.5399999999999995e-06, 3.0899999999999996e-06, 0.40737365142725557, 3.6e-05, 9.709999999999999e-06, 0.38540902159950763, 2.95e-06, 1.1258660761266643, 7.43e-05, 8.15e-07, 1.7099999999999996e-05, 1.61e-06, 3.0299999999999995e-07, 1.74e-06, 0.00023, 3.99e-06, 6.720000000000002e-06, 8.469999999999998e-07, 2.32e-06, 4.35e-07, 4.2299999999999996e-07, 0.9929166934911581, 4.969999999999999e-07, 0.00193, 6.269999999999999e-06, 8.879999999999998e-06, 5.490000000000001e-06, 3.4199999999999994e-06, 0.9100198588845018, 2.730000000000001e-05, 2.16e-05, 1.16e-05, 4.4600000000000005e-06, 5.170000000000001e-07, 3.5799999999999996e-06, 3.67e-06, 5.239999999999999e-06, 2.67e-05, 4.68e-06, 1.012083017780519, 3.4000000000000003e-07, 2.86e-06, 8.23e-06, 6.350000000000001e-07, 0.9651064469913785, 5.729999999999999e-06, 2.3899999999999996e-06, 0.0224, 0.7546792976449075, 1.19e-06, 7.63e-06, 2.0100000000000007e-07]
T = [4.101499773326386, 5.772352839123414, 4.347026398319477, 4.527057249915543, 5.039030814368431, 8.929467406653387, 4.009884244196475, 4.43500643693444, 7.194604057657564, 10.489370324111812]


def logits(h):
    # rounded to a multiple of 2^-20 (the formula's class scores are exact binary fractions): removes floating-point noise
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        0.4043572 + T[0] * (   # class QCD
            - 0.1841809 * h[18] / H_AVG[18]
            + 0.1443849 * h[97] / H_AVG[97]
            + 0.1024603 * h[52] / H_AVG[52]
            - 0.09988948 * h[68] / H_AVG[68]
            - 0.09293853 * h[120] / H_AVG[120]
            - 0.07482292 * h[24] / H_AVG[24]
            + 0.0726371 * h[104] / H_AVG[104]
            + 0.06075093 * h[81] / H_AVG[81]
            - 0.04122781 * h[83] / H_AVG[83]
            - 0.03355074 * h[124] / H_AVG[124]
            + 0.03145078 * h[16] / H_AVG[16]
            + 0.0314318 * h[115] / H_AVG[115]
            + 0.02641079 * h[78] / H_AVG[78]
            + 0.003112995 * h[70] / H_AVG[70]
            + 0.0003814002 * h[123] / H_AVG[123]
            + 0.0002114087 * h[25] / H_AVG[25]
            + 6.4633e-05 * h[0] / H_AVG[0]
            - 4.158089e-05 * h[99] / H_AVG[99]
            - 3.20768e-05 * h[21] / H_AVG[21]
            - 1.279532e-05 * h[90] / H_AVG[90]
            + 1.969235e-06 * h[84] / H_AVG[84]
            + 4.651149e-07 * h[55] / H_AVG[55]
            + 3.114955e-07 * h[79] / H_AVG[79]
            - 2.973097e-07 * h[105] / H_AVG[105]
            - 2.413734e-07 * h[106] / H_AVG[106]
            + 1.965853e-07 * h[113] / H_AVG[113]
            - 1.866084e-07 * h[34] / H_AVG[34]
            + 1.674151e-07 * h[73] / H_AVG[73]
            + 1.554367e-07 * h[10] / H_AVG[10]
            + 1.483127e-07 * h[11] / H_AVG[11]
            - 1.261562e-07 * h[6] / H_AVG[6]
            + 1.204002e-07 * h[86] / H_AVG[86]
            - 1.143782e-07 * h[32] / H_AVG[32]
            + 1.056985e-07 * h[17] / H_AVG[17]
            + 9.508195e-08 * h[38] / H_AVG[38]
            - 7.81001e-08 * h[42] / H_AVG[42]
            - 6.955716e-08 * h[54] / H_AVG[54]
            - 6.612047e-08 * h[29] / H_AVG[29]
            + 5.994851e-08 * h[118] / H_AVG[118]
            + 4.685785e-08 * h[48] / H_AVG[48]
            - 4.4681e-08 * h[45] / H_AVG[45]
            - 4.312472e-08 * h[39] / H_AVG[39]
            + 3.972674e-08 * h[27] / H_AVG[27]
            + 3.962338e-08 * h[63] / H_AVG[63]
            + 3.917975e-08 * h[30] / H_AVG[30]
            + 3.834908e-08 * h[43] / H_AVG[43]
            + 3.569833e-08 * h[121] / H_AVG[121]
            - 3.507622e-08 * h[80] / H_AVG[80]
            - 3.208623e-08 * h[126] / H_AVG[126]
            + 3.100652e-08 * h[60] / H_AVG[60]
            - 3.051464e-08 * h[61] / H_AVG[61]
            - 2.89751e-08 * h[102] / H_AVG[102]
            - 2.897032e-08 * h[107] / H_AVG[107]
            - 2.848089e-08 * h[7] / H_AVG[7]
            + 2.564138e-08 * h[92] / H_AVG[92]
            + 2.508156e-08 * h[35] / H_AVG[35]
            + 2.436751e-08 * h[15] / H_AVG[15]
            - 2.357e-08 * h[59] / H_AVG[59]
            - 2.274011e-08 * h[101] / H_AVG[101]
            - 2.171914e-08 * h[28] / H_AVG[28]
            + 2.111208e-08 * h[103] / H_AVG[103]
            + 1.982043e-08 * h[69] / H_AVG[69]
            + 1.929196e-08 * h[67] / H_AVG[67]
            + 1.863091e-08 * h[1] / H_AVG[1]
            + 1.85288e-08 * h[100] / H_AVG[100]
            - 1.700329e-08 * h[110] / H_AVG[110]
            - 1.643131e-08 * h[3] / H_AVG[3]
            + 1.577519e-08 * h[37] / H_AVG[37]
            - 1.546459e-08 * h[50] / H_AVG[50]
            + 1.457556e-08 * h[44] / H_AVG[44]
            + 1.456352e-08 * h[36] / H_AVG[36]
            - 1.429404e-08 * h[12] / H_AVG[12]
            - 1.360364e-08 * h[114] / H_AVG[114]
            - 1.348627e-08 * h[2] / H_AVG[2]
            + 1.296558e-08 * h[58] / H_AVG[58]
            - 1.294223e-08 * h[112] / H_AVG[112]
            + 1.265572e-08 * h[76] / H_AVG[76]
            - 1.219796e-08 * h[13] / H_AVG[13]
            - 1.187193e-08 * h[19] / H_AVG[19]
            - 1.163591e-08 * h[111] / H_AVG[111]
            + 1.108028e-08 * h[74] / H_AVG[74]
            + 1.049705e-08 * h[108] / H_AVG[108]
            + 9.905272e-09 * h[5] / H_AVG[5]
            - 9.43357e-09 * h[8] / H_AVG[8]
            - 9.059813e-09 * h[91] / H_AVG[91]
            - 8.44527e-09 * h[4] / H_AVG[4]
            - 8.090269e-09 * h[117] / H_AVG[117]
            + 6.88004e-09 * h[53] / H_AVG[53]
            - 6.766214e-09 * h[66] / H_AVG[66]
            - 6.14009e-09 * h[33] / H_AVG[33]
            - 5.981534e-09 * h[65] / H_AVG[65]
            + 5.547199e-09 * h[82] / H_AVG[82]
            - 5.199588e-09 * h[31] / H_AVG[31]
            - 4.777889e-09 * h[71] / H_AVG[71]
            + 4.433161e-09 * h[94] / H_AVG[94]
            + 4.279184e-09 * h[9] / H_AVG[9]
            + 3.765552e-09 * h[26] / H_AVG[26]
            - 3.369587e-09 * h[14] / H_AVG[14]
            + 3.327697e-09 * h[57] / H_AVG[57]
            + 3.15532e-09 * h[49] / H_AVG[49]
            - 3.096563e-09 * h[22] / H_AVG[22]
            + 3.090176e-09 * h[77] / H_AVG[77]
            + 2.85157e-09 * h[122] / H_AVG[122]
            + 2.302736e-09 * h[72] / H_AVG[72]
            + 2.221901e-09 * h[51] / H_AVG[51]
            - 2.179332e-09 * h[23] / H_AVG[23]
            - 1.508798e-09 * h[89] / H_AVG[89]
            + 1.46934e-09 * h[125] / H_AVG[125]
            - 1.377648e-09 * h[75] / H_AVG[75]
            + 1.191305e-09 * h[93] / H_AVG[93]
            - 1.187724e-09 * h[87] / H_AVG[87]
            - 1.078153e-09 * h[40] / H_AVG[40]
            - 9.895845e-10 * h[20] / H_AVG[20]
            + 8.907553e-10 * h[119] / H_AVG[119]
            - 8.120197e-10 * h[62] / H_AVG[62]
            - 7.278878e-10 * h[95] / H_AVG[95]
            - 6.700071e-10 * h[96] / H_AVG[96]
            + 3.995436e-10 * h[116] / H_AVG[116]
            + 3.845857e-10 * h[56] / H_AVG[56]
            + 2.640631e-10 * h[85] / H_AVG[85]
            + 1.569507e-10 * h[109] / H_AVG[109]
            - 1.368907e-10 * h[98] / H_AVG[98]
            - 1.109926e-10 * h[88] / H_AVG[88]
            - 7.082349e-11 * h[46] / H_AVG[46]
            - 6.895744e-11 * h[47] / H_AVG[47]
            - 4.27841e-11 * h[64] / H_AVG[64]
            + 1.715518e-11 * h[127] / H_AVG[127]
            + 1.601313e-12 * h[41] / H_AVG[41]
        ),
        -0.4064285 + T[1] * (   # class Hbb
            + 0.2928869 * h[16] / H_AVG[16]
            + 0.1978613 * h[120] / H_AVG[120]
            - 0.0878631 * h[68] / H_AVG[68]
            + 0.08120346 * h[18] / H_AVG[18]
            - 0.06659133 * h[52] / H_AVG[52]
            - 0.0665244 * h[24] / H_AVG[24]
            - 0.06077145 * h[97] / H_AVG[97]
            - 0.05279166 * h[78] / H_AVG[78]
            + 0.04240856 * h[115] / H_AVG[115]
            + 0.02099928 * h[81] / H_AVG[81]
            - 0.01115691 * h[70] / H_AVG[70]
            + 0.009735844 * h[83] / H_AVG[83]
            - 0.004192397 * h[104] / H_AVG[104]
            + 0.00327716 * h[124] / H_AVG[124]
            - 0.001531489 * h[25] / H_AVG[25]
            + 0.0001042601 * h[0] / H_AVG[0]
            + 3.431336e-05 * h[123] / H_AVG[123]
            - 2.992736e-05 * h[99] / H_AVG[99]
            - 2.533495e-05 * h[21] / H_AVG[21]
            - 6.6119e-06 * h[90] / H_AVG[90]
            + 1.398703e-06 * h[84] / H_AVG[84]
            + 3.304371e-07 * h[55] / H_AVG[55]
            + 2.212303e-07 * h[79] / H_AVG[79]
            - 2.112268e-07 * h[105] / H_AVG[105]
            - 1.714632e-07 * h[106] / H_AVG[106]
            + 1.396048e-07 * h[113] / H_AVG[113]
            - 1.325701e-07 * h[34] / H_AVG[34]
            + 1.189098e-07 * h[73] / H_AVG[73]
            + 1.104408e-07 * h[10] / H_AVG[10]
            + 1.054102e-07 * h[11] / H_AVG[11]
            - 8.964801e-08 * h[6] / H_AVG[6]
            + 8.552046e-08 * h[86] / H_AVG[86]
            - 8.123513e-08 * h[32] / H_AVG[32]
            + 7.508111e-08 * h[17] / H_AVG[17]
            + 6.75697e-08 * h[38] / H_AVG[38]
            - 5.549027e-08 * h[42] / H_AVG[42]
            - 4.941256e-08 * h[54] / H_AVG[54]
            - 4.696886e-08 * h[29] / H_AVG[29]
            + 4.258727e-08 * h[118] / H_AVG[118]
            + 3.328906e-08 * h[48] / H_AVG[48]
            - 3.1736e-08 * h[45] / H_AVG[45]
            - 3.064035e-08 * h[39] / H_AVG[39]
            + 2.82289e-08 * h[27] / H_AVG[27]
            + 2.81501e-08 * h[63] / H_AVG[63]
            + 2.783288e-08 * h[30] / H_AVG[30]
            + 2.724666e-08 * h[43] / H_AVG[43]
            + 2.536468e-08 * h[121] / H_AVG[121]
            - 2.491489e-08 * h[80] / H_AVG[80]
            - 2.279315e-08 * h[126] / H_AVG[126]
            + 2.20321e-08 * h[60] / H_AVG[60]
            - 2.168043e-08 * h[61] / H_AVG[61]
            - 2.058583e-08 * h[102] / H_AVG[102]
            - 2.056623e-08 * h[107] / H_AVG[107]
            - 2.023595e-08 * h[7] / H_AVG[7]
            + 1.821787e-08 * h[92] / H_AVG[92]
            + 1.781946e-08 * h[35] / H_AVG[35]
            + 1.731017e-08 * h[15] / H_AVG[15]
            - 1.674701e-08 * h[59] / H_AVG[59]
            - 1.615771e-08 * h[101] / H_AVG[101]
            - 1.54267e-08 * h[28] / H_AVG[28]
            + 1.499802e-08 * h[103] / H_AVG[103]
            + 1.407868e-08 * h[69] / H_AVG[69]
            + 1.370382e-08 * h[67] / H_AVG[67]
            + 1.323749e-08 * h[1] / H_AVG[1]
            + 1.31666e-08 * h[100] / H_AVG[100]
            - 1.208277e-08 * h[110] / H_AVG[110]
            - 1.167511e-08 * h[3] / H_AVG[3]
            + 1.120607e-08 * h[37] / H_AVG[37]
            - 1.097871e-08 * h[50] / H_AVG[50]
            + 1.035701e-08 * h[44] / H_AVG[44]
            + 1.034758e-08 * h[36] / H_AVG[36]
            - 1.015232e-08 * h[12] / H_AVG[12]
            - 9.660451e-09 * h[114] / H_AVG[114]
            - 9.575054e-09 * h[2] / H_AVG[2]
            + 9.21214e-09 * h[58] / H_AVG[58]
            - 9.194012e-09 * h[112] / H_AVG[112]
            + 8.990681e-09 * h[76] / H_AVG[76]
            - 8.666607e-09 * h[13] / H_AVG[13]
            - 8.44264e-09 * h[19] / H_AVG[19]
            - 8.26692e-09 * h[111] / H_AVG[111]
            + 7.872067e-09 * h[74] / H_AVG[74]
            + 7.460878e-09 * h[108] / H_AVG[108]
            + 7.037783e-09 * h[5] / H_AVG[5]
            - 6.702135e-09 * h[8] / H_AVG[8]
            - 6.436674e-09 * h[91] / H_AVG[91]
            - 6.000362e-09 * h[4] / H_AVG[4]
            - 5.747728e-09 * h[117] / H_AVG[117]
            + 4.885236e-09 * h[53] / H_AVG[53]
            - 4.803655e-09 * h[66] / H_AVG[66]
            - 4.363254e-09 * h[33] / H_AVG[33]
            - 4.248905e-09 * h[65] / H_AVG[65]
            + 3.940643e-09 * h[82] / H_AVG[82]
            - 3.695188e-09 * h[31] / H_AVG[31]
            - 3.392971e-09 * h[71] / H_AVG[71]
            + 3.148469e-09 * h[94] / H_AVG[94]
            + 3.039774e-09 * h[9] / H_AVG[9]
            + 2.673447e-09 * h[26] / H_AVG[26]
            - 2.393687e-09 * h[14] / H_AVG[14]
            + 2.363836e-09 * h[57] / H_AVG[57]
            + 2.241867e-09 * h[49] / H_AVG[49]
            - 2.200364e-09 * h[22] / H_AVG[22]
            + 2.196446e-09 * h[77] / H_AVG[77]
            + 2.025574e-09 * h[122] / H_AVG[122]
            + 1.636096e-09 * h[72] / H_AVG[72]
            + 1.578445e-09 * h[51] / H_AVG[51]
            - 1.548186e-09 * h[23] / H_AVG[23]
            - 1.072178e-09 * h[89] / H_AVG[89]
            + 1.04439e-09 * h[125] / H_AVG[125]
            - 9.784944e-10 * h[75] / H_AVG[75]
            + 8.459203e-10 * h[93] / H_AVG[93]
            - 8.434528e-10 * h[87] / H_AVG[87]
            - 7.659372e-10 * h[40] / H_AVG[40]
            - 7.032427e-10 * h[20] / H_AVG[20]
            + 6.32152e-10 * h[119] / H_AVG[119]
            - 5.769822e-10 * h[62] / H_AVG[62]
            - 5.170768e-10 * h[95] / H_AVG[95]
            - 4.758915e-10 * h[96] / H_AVG[96]
            + 2.839224e-10 * h[116] / H_AVG[116]
            + 2.731353e-10 * h[56] / H_AVG[56]
            + 1.874921e-10 * h[85] / H_AVG[85]
            + 1.115312e-10 * h[109] / H_AVG[109]
            - 9.727185e-11 * h[98] / H_AVG[98]
            - 7.890356e-11 * h[88] / H_AVG[88]
            - 5.029018e-11 * h[46] / H_AVG[46]
            - 4.900712e-11 * h[47] / H_AVG[47]
            - 2.677025e-11 * h[64] / H_AVG[64]
            + 1.215305e-11 * h[127] / H_AVG[127]
            + 1.135601e-12 * h[41] / H_AVG[41]
        ),
        0.1415668 + T[2] * (   # class Hcc
            - 0.2467673 * h[83] / H_AVG[83]
            + 0.186621 * h[18] / H_AVG[18]
            - 0.127937 * h[24] / H_AVG[24]
            + 0.06881596 * h[124] / H_AVG[124]
            - 0.06787882 * h[81] / H_AVG[81]
            - 0.05704031 * h[16] / H_AVG[16]
            - 0.0468052 * h[97] / H_AVG[97]
            + 0.04634253 * h[104] / H_AVG[104]
            + 0.03704713 * h[78] / H_AVG[78]
            + 0.03155546 * h[120] / H_AVG[120]
            - 0.02499648 * h[52] / H_AVG[52]
            - 0.02147523 * h[68] / H_AVG[68]
            + 0.02066152 * h[115] / H_AVG[115]
            + 0.0139051 * h[70] / H_AVG[70]
            - 0.001348741 * h[25] / H_AVG[25]
            - 0.000531535 * h[123] / H_AVG[123]
            + 0.000198961 * h[0] / H_AVG[0]
            - 3.943294e-05 * h[99] / H_AVG[99]
            - 2.622552e-05 * h[21] / H_AVG[21]
            + 1.857905e-06 * h[84] / H_AVG[84]
            + 4.390112e-07 * h[55] / H_AVG[55]
            + 2.939418e-07 * h[79] / H_AVG[79]
            - 2.805022e-07 * h[105] / H_AVG[105]
            + 2.697346e-07 * h[90] / H_AVG[90]
            - 2.27795e-07 * h[106] / H_AVG[106]
            + 1.854913e-07 * h[113] / H_AVG[113]
            - 1.760499e-07 * h[34] / H_AVG[34]
            + 1.57955e-07 * h[73] / H_AVG[73]
            + 1.467106e-07 * h[10] / H_AVG[10]
            + 1.399781e-07 * h[11] / H_AVG[11]
            - 1.190415e-07 * h[6] / H_AVG[6]
            + 1.135822e-07 * h[86] / H_AVG[86]
            - 1.07922e-07 * h[32] / H_AVG[32]
            + 9.976857e-08 * h[17] / H_AVG[17]
            + 8.970006e-08 * h[38] / H_AVG[38]
            - 7.368875e-08 * h[42] / H_AVG[42]
            - 6.56409e-08 * h[54] / H_AVG[54]
            - 6.24019e-08 * h[29] / H_AVG[29]
            + 5.657146e-08 * h[118] / H_AVG[118]
            + 4.421376e-08 * h[48] / H_AVG[48]
            - 4.215529e-08 * h[45] / H_AVG[45]
            - 4.069764e-08 * h[39] / H_AVG[39]
            + 3.749681e-08 * h[27] / H_AVG[27]
            + 3.739679e-08 * h[63] / H_AVG[63]
            + 3.696459e-08 * h[30] / H_AVG[30]
            + 3.617771e-08 * h[43] / H_AVG[43]
            + 3.370005e-08 * h[121] / H_AVG[121]
            - 3.310164e-08 * h[80] / H_AVG[80]
            - 3.027104e-08 * h[126] / H_AVG[126]
            + 2.926375e-08 * h[60] / H_AVG[60]
            - 2.879394e-08 * h[61] / H_AVG[61]
            - 2.733891e-08 * h[102] / H_AVG[102]
            - 2.730719e-08 * h[107] / H_AVG[107]
            - 2.689092e-08 * h[7] / H_AVG[7]
            + 2.419472e-08 * h[92] / H_AVG[92]
            + 2.366953e-08 * h[35] / H_AVG[35]
            + 2.299427e-08 * h[15] / H_AVG[15]
            - 2.224242e-08 * h[59] / H_AVG[59]
            - 2.145069e-08 * h[101] / H_AVG[101]
            - 2.049449e-08 * h[28] / H_AVG[28]
            + 1.992346e-08 * h[103] / H_AVG[103]
            + 1.870263e-08 * h[69] / H_AVG[69]
            + 1.820585e-08 * h[67] / H_AVG[67]
            + 1.758581e-08 * h[1] / H_AVG[1]
            + 1.748821e-08 * h[100] / H_AVG[100]
            - 1.604899e-08 * h[110] / H_AVG[110]
            - 1.550792e-08 * h[3] / H_AVG[3]
            + 1.488379e-08 * h[37] / H_AVG[37]
            - 1.458795e-08 * h[50] / H_AVG[50]
            + 1.375622e-08 * h[44] / H_AVG[44]
            + 1.374636e-08 * h[36] / H_AVG[36]
            - 1.348422e-08 * h[12] / H_AVG[12]
            - 1.283339e-08 * h[114] / H_AVG[114]
            - 1.272534e-08 * h[2] / H_AVG[2]
            + 1.223873e-08 * h[58] / H_AVG[58]
            - 1.221671e-08 * h[112] / H_AVG[112]
            + 1.194393e-08 * h[76] / H_AVG[76]
            - 1.151107e-08 * h[13] / H_AVG[13]
            - 1.1209e-08 * h[19] / H_AVG[19]
            - 1.097785e-08 * h[111] / H_AVG[111]
            + 1.045579e-08 * h[74] / H_AVG[74]
            + 9.907136e-09 * h[108] / H_AVG[108]
            + 9.35222e-09 * h[5] / H_AVG[5]
            - 8.904431e-09 * h[8] / H_AVG[8]
            - 8.552715e-09 * h[91] / H_AVG[91]
            - 7.969479e-09 * h[4] / H_AVG[4]
            - 7.633985e-09 * h[117] / H_AVG[117]
            + 6.493565e-09 * h[53] / H_AVG[53]
            - 6.385375e-09 * h[66] / H_AVG[66]
            - 5.794119e-09 * h[33] / H_AVG[33]
            - 5.645889e-09 * h[65] / H_AVG[65]
            + 5.235474e-09 * h[82] / H_AVG[82]
            - 4.904187e-09 * h[31] / H_AVG[31]
            - 4.506549e-09 * h[71] / H_AVG[71]
            + 4.183734e-09 * h[94] / H_AVG[94]
            + 4.037335e-09 * h[9] / H_AVG[9]
            + 3.54909e-09 * h[26] / H_AVG[26]
            - 3.181149e-09 * h[14] / H_AVG[14]
            + 3.140995e-09 * h[57] / H_AVG[57]
            + 2.976857e-09 * h[49] / H_AVG[49]
            - 2.923678e-09 * h[22] / H_AVG[22]
            + 2.915573e-09 * h[77] / H_AVG[77]
            + 2.691779e-09 * h[122] / H_AVG[122]
            + 2.172204e-09 * h[72] / H_AVG[72]
            + 2.097885e-09 * h[51] / H_AVG[51]
            - 2.057262e-09 * h[23] / H_AVG[23]
            - 1.423558e-09 * h[89] / H_AVG[89]
            + 1.386535e-09 * h[125] / H_AVG[125]
            - 1.29982e-09 * h[75] / H_AVG[75]
            + 1.123517e-09 * h[93] / H_AVG[93]
            - 1.120493e-09 * h[87] / H_AVG[87]
            - 1.01721e-09 * h[40] / H_AVG[40]
            - 9.342954e-10 * h[20] / H_AVG[20]
            + 8.403453e-10 * h[119] / H_AVG[119]
            - 7.663227e-10 * h[62] / H_AVG[62]
            - 6.866394e-10 * h[95] / H_AVG[95]
            - 6.32018e-10 * h[96] / H_AVG[96]
            + 3.772338e-10 * h[116] / H_AVG[116]
            + 3.630241e-10 * h[56] / H_AVG[56]
            + 2.493257e-10 * h[85] / H_AVG[85]
            + 1.480901e-10 * h[109] / H_AVG[109]
            - 1.291776e-10 * h[98] / H_AVG[98]
            - 1.046625e-10 * h[88] / H_AVG[88]
            - 6.682762e-11 * h[46] / H_AVG[46]
            - 6.504607e-11 * h[47] / H_AVG[47]
            - 4.132137e-11 * h[64] / H_AVG[64]
            + 1.617994e-11 * h[127] / H_AVG[127]
            + 1.512693e-12 * h[41] / H_AVG[41]
        ),
        0.1286925 + T[3] * (   # class Hgg
            - 0.2013121 * h[104] / H_AVG[104]
            - 0.1493283 * h[68] / H_AVG[68]
            - 0.1078419 * h[18] / H_AVG[18]
            - 0.1026821 * h[16] / H_AVG[16]
            + 0.09597942 * h[52] / H_AVG[52]
            - 0.07338015 * h[97] / H_AVG[97]
            + 0.06508685 * h[124] / H_AVG[124]
            + 0.05324629 * h[115] / H_AVG[115]
            - 0.04711376 * h[24] / H_AVG[24]
            - 0.04469899 * h[81] / H_AVG[81]
            + 0.02077883 * h[83] / H_AVG[83]
            - 0.01638199 * h[78] / H_AVG[78]
            + 0.0121726 * h[120] / H_AVG[120]
            + 0.008689166 * h[70] / H_AVG[70]
            - 0.0009018679 * h[25] / H_AVG[25]
            + 0.0002335506 * h[0] / H_AVG[0]
            - 9.652143e-05 * h[123] / H_AVG[123]
            - 3.765043e-05 * h[99] / H_AVG[99]
            - 2.70526e-05 * h[21] / H_AVG[21]
            + 5.314363e-06 * h[90] / H_AVG[90]
            + 1.783779e-06 * h[84] / H_AVG[84]
            + 4.213812e-07 * h[55] / H_AVG[55]
            + 2.821827e-07 * h[79] / H_AVG[79]
            - 2.6934e-07 * h[105] / H_AVG[105]
            - 2.186614e-07 * h[106] / H_AVG[106]
            + 1.7809e-07 * h[113] / H_AVG[113]
            - 1.690639e-07 * h[34] / H_AVG[34]
            + 1.516554e-07 * h[73] / H_AVG[73]
            + 1.408116e-07 * h[10] / H_AVG[10]
            + 1.343603e-07 * h[11] / H_AVG[11]
            - 1.1429e-07 * h[6] / H_AVG[6]
            + 1.090622e-07 * h[86] / H_AVG[86]
            - 1.036369e-07 * h[32] / H_AVG[32]
            + 9.578296e-08 * h[17] / H_AVG[17]
            + 8.616677e-08 * h[38] / H_AVG[38]
            - 7.074728e-08 * h[42] / H_AVG[42]
            - 6.301496e-08 * h[54] / H_AVG[54]
            - 5.989545e-08 * h[29] / H_AVG[29]
            + 5.431977e-08 * h[118] / H_AVG[118]
            + 4.245199e-08 * h[48] / H_AVG[48]
            - 4.047894e-08 * h[45] / H_AVG[45]
            - 3.906643e-08 * h[39] / H_AVG[39]
            + 3.599662e-08 * h[27] / H_AVG[27]
            + 3.589386e-08 * h[63] / H_AVG[63]
            + 3.548843e-08 * h[30] / H_AVG[30]
            + 3.474001e-08 * h[43] / H_AVG[43]
            + 3.234772e-08 * h[121] / H_AVG[121]
            - 3.177413e-08 * h[80] / H_AVG[80]
            - 2.906723e-08 * h[126] / H_AVG[126]
            + 2.809013e-08 * h[60] / H_AVG[60]
            - 2.765016e-08 * h[61] / H_AVG[61]
            - 2.624995e-08 * h[102] / H_AVG[102]
            - 2.623598e-08 * h[107] / H_AVG[107]
            - 2.580196e-08 * h[7] / H_AVG[7]
            + 2.322789e-08 * h[92] / H_AVG[92]
            + 2.272357e-08 * h[35] / H_AVG[35]
            + 2.207161e-08 * h[15] / H_AVG[15]
            - 2.135662e-08 * h[59] / H_AVG[59]
            - 2.059829e-08 * h[101] / H_AVG[101]
            - 1.967599e-08 * h[28] / H_AVG[28]
            + 1.912687e-08 * h[103] / H_AVG[103]
            + 1.795426e-08 * h[69] / H_AVG[69]
            + 1.747356e-08 * h[67] / H_AVG[67]
            + 1.687859e-08 * h[1] / H_AVG[1]
            + 1.678551e-08 * h[100] / H_AVG[100]
            - 1.540307e-08 * h[110] / H_AVG[110]
            - 1.488638e-08 * h[3] / H_AVG[3]
            + 1.428973e-08 * h[37] / H_AVG[37]
            - 1.401169e-08 * h[50] / H_AVG[50]
            + 1.320525e-08 * h[44] / H_AVG[44]
            + 1.31934e-08 * h[36] / H_AVG[36]
            - 1.294998e-08 * h[12] / H_AVG[12]
            - 1.232224e-08 * h[114] / H_AVG[114]
            - 1.221601e-08 * h[2] / H_AVG[2]
            + 1.175049e-08 * h[58] / H_AVG[58]
            - 1.173065e-08 * h[112] / H_AVG[112]
            + 1.146936e-08 * h[76] / H_AVG[76]
            - 1.105119e-08 * h[13] / H_AVG[13]
            - 1.076258e-08 * h[19] / H_AVG[19]
            - 1.054134e-08 * h[111] / H_AVG[111]
            + 1.003835e-08 * h[74] / H_AVG[74]
            + 9.510769e-09 * h[108] / H_AVG[108]
            + 8.977754e-09 * h[5] / H_AVG[5]
            - 8.548893e-09 * h[8] / H_AVG[8]
            - 8.206911e-09 * h[91] / H_AVG[91]
            - 7.651751e-09 * h[4] / H_AVG[4]
            - 7.328756e-09 * h[117] / H_AVG[117]
            + 6.231097e-09 * h[53] / H_AVG[53]
            - 6.128502e-09 * h[66] / H_AVG[66]
            - 5.56323e-09 * h[33] / H_AVG[33]
            - 5.412459e-09 * h[65] / H_AVG[65]
            + 5.025093e-09 * h[82] / H_AVG[82]
            - 4.709503e-09 * h[31] / H_AVG[31]
            - 4.328339e-09 * h[71] / H_AVG[71]
            + 4.015333e-09 * h[94] / H_AVG[94]
            + 3.876556e-09 * h[9] / H_AVG[9]
            + 3.409453e-09 * h[26] / H_AVG[26]
            - 3.052458e-09 * h[14] / H_AVG[14]
            + 3.015298e-09 * h[57] / H_AVG[57]
            + 2.858434e-09 * h[49] / H_AVG[49]
            - 2.804791e-09 * h[22] / H_AVG[22]
            + 2.8002e-09 * h[77] / H_AVG[77]
            + 2.582954e-09 * h[122] / H_AVG[122]
            + 2.086363e-09 * h[72] / H_AVG[72]
            + 2.014217e-09 * h[51] / H_AVG[51]
            - 1.974073e-09 * h[23] / H_AVG[23]
            - 1.366998e-09 * h[89] / H_AVG[89]
            + 1.331095e-09 * h[125] / H_AVG[125]
            - 1.248155e-09 * h[75] / H_AVG[75]
            + 1.078683e-09 * h[93] / H_AVG[93]
            - 1.075571e-09 * h[87] / H_AVG[87]
            - 9.767044e-10 * h[40] / H_AVG[40]
            - 8.96896e-10 * h[20] / H_AVG[20]
            + 8.066525e-10 * h[119] / H_AVG[119]
            - 7.356534e-10 * h[62] / H_AVG[62]
            - 6.592699e-10 * h[95] / H_AVG[95]
            - 6.068713e-10 * h[96] / H_AVG[96]
            + 3.619926e-10 * h[116] / H_AVG[116]
            + 3.483025e-10 * h[56] / H_AVG[56]
            + 2.3934e-10 * h[85] / H_AVG[85]
            + 1.421985e-10 * h[109] / H_AVG[109]
            - 1.240136e-10 * h[98] / H_AVG[98]
            - 1.006457e-10 * h[88] / H_AVG[88]
            - 6.414925e-11 * h[46] / H_AVG[46]
            - 6.24658e-11 * h[47] / H_AVG[47]
            - 3.804097e-11 * h[64] / H_AVG[64]
            + 1.553586e-11 * h[127] / H_AVG[127]
            + 1.450214e-12 * h[41] / H_AVG[41]
        ),
        -0.07702489 + T[4] * (   # class H4q
            - 0.3007017 * h[115] / H_AVG[115]
            - 0.09166297 * h[104] / H_AVG[104]
            - 0.0895748 * h[16] / H_AVG[16]
            + 0.08955998 * h[52] / H_AVG[52]
            - 0.07181989 * h[24] / H_AVG[24]
            - 0.06881414 * h[97] / H_AVG[97]
            + 0.05950307 * h[124] / H_AVG[124]
            - 0.05701154 * h[78] / H_AVG[78]
            - 0.04751651 * h[18] / H_AVG[18]
            - 0.04326352 * h[81] / H_AVG[81]
            + 0.04324974 * h[68] / H_AVG[68]
            - 0.01379042 * h[83] / H_AVG[83]
            + 0.01242218 * h[120] / H_AVG[120]
            + 0.009934757 * h[70] / H_AVG[70]
            - 0.0008460934 * h[25] / H_AVG[25]
            + 0.0002325441 * h[0] / H_AVG[0]
            - 3.377426e-05 * h[99] / H_AVG[99]
            + 2.919219e-05 * h[123] / H_AVG[123]
            - 2.722137e-05 * h[21] / H_AVG[21]
            + 1.602404e-06 * h[84] / H_AVG[84]
            - 9.944717e-07 * h[90] / H_AVG[90]
            + 3.787056e-07 * h[55] / H_AVG[55]
            + 2.535319e-07 * h[79] / H_AVG[79]
            - 2.419851e-07 * h[105] / H_AVG[105]
            - 1.965242e-07 * h[106] / H_AVG[106]
            + 1.599644e-07 * h[113] / H_AVG[113]
            - 1.518941e-07 * h[34] / H_AVG[34]
            + 1.362598e-07 * h[73] / H_AVG[73]
            + 1.265294e-07 * h[10] / H_AVG[10]
            + 1.207084e-07 * h[11] / H_AVG[11]
            - 1.026959e-07 * h[6] / H_AVG[6]
            + 9.798583e-08 * h[86] / H_AVG[86]
            - 9.309838e-08 * h[32] / H_AVG[32]
            + 8.60622e-08 * h[17] / H_AVG[17]
            + 7.741647e-08 * h[38] / H_AVG[38]
            - 6.356633e-08 * h[42] / H_AVG[42]
            - 5.662773e-08 * h[54] / H_AVG[54]
            - 5.382614e-08 * h[29] / H_AVG[29]
            + 4.8807e-08 * h[118] / H_AVG[118]
            + 3.813749e-08 * h[48] / H_AVG[48]
            - 3.63677e-08 * h[45] / H_AVG[45]
            - 3.510019e-08 * h[39] / H_AVG[39]
            + 3.234276e-08 * h[27] / H_AVG[27]
            + 3.224889e-08 * h[63] / H_AVG[63]
            + 3.188804e-08 * h[30] / H_AVG[30]
            + 3.121305e-08 * h[43] / H_AVG[43]
            + 2.906296e-08 * h[121] / H_AVG[121]
            - 2.855194e-08 * h[80] / H_AVG[80]
            - 2.611926e-08 * h[126] / H_AVG[126]
            + 2.524757e-08 * h[60] / H_AVG[60]
            - 2.483626e-08 * h[61] / H_AVG[61]
            - 2.358301e-08 * h[102] / H_AVG[102]
            - 2.357263e-08 * h[107] / H_AVG[107]
            - 2.318493e-08 * h[7] / H_AVG[7]
            + 2.086812e-08 * h[92] / H_AVG[92]
            + 2.041886e-08 * h[35] / H_AVG[35]
            + 1.983352e-08 * h[15] / H_AVG[15]
            - 1.918203e-08 * h[59] / H_AVG[59]
            - 1.850699e-08 * h[101] / H_AVG[101]
            - 1.767952e-08 * h[28] / H_AVG[28]
            + 1.718429e-08 * h[103] / H_AVG[103]
            + 1.613058e-08 * h[69] / H_AVG[69]
            + 1.570272e-08 * h[67] / H_AVG[67]
            + 1.516486e-08 * h[1] / H_AVG[1]
            + 1.508048e-08 * h[100] / H_AVG[100]
            - 1.384193e-08 * h[110] / H_AVG[110]
            - 1.337256e-08 * h[3] / H_AVG[3]
            + 1.283903e-08 * h[37] / H_AVG[37]
            - 1.257916e-08 * h[50] / H_AVG[50]
            + 1.186423e-08 * h[44] / H_AVG[44]
            + 1.185367e-08 * h[36] / H_AVG[36]
            - 1.163213e-08 * h[12] / H_AVG[12]
            - 1.107232e-08 * h[114] / H_AVG[114]
            - 1.097787e-08 * h[2] / H_AVG[2]
            + 1.05566e-08 * h[58] / H_AVG[58]
            - 1.053768e-08 * h[112] / H_AVG[112]
            + 1.030428e-08 * h[76] / H_AVG[76]
            - 9.927899e-09 * h[13] / H_AVG[13]
            - 9.666578e-09 * h[19] / H_AVG[19]
            - 9.470465e-09 * h[111] / H_AVG[111]
            + 9.018021e-09 * h[74] / H_AVG[74]
            + 8.543853e-09 * h[108] / H_AVG[108]
            + 8.066835e-09 * h[5] / H_AVG[5]
            - 7.681058e-09 * h[8] / H_AVG[8]
            - 7.377709e-09 * h[91] / H_AVG[91]
            - 6.874474e-09 * h[4] / H_AVG[4]
            - 6.585804e-09 * h[117] / H_AVG[117]
            + 5.597682e-09 * h[53] / H_AVG[53]
            - 5.50648e-09 * h[66] / H_AVG[66]
            - 4.997561e-09 * h[33] / H_AVG[33]
            - 4.863971e-09 * h[65] / H_AVG[65]
            + 4.516224e-09 * h[82] / H_AVG[82]
            - 4.231576e-09 * h[31] / H_AVG[31]
            - 3.888791e-09 * h[71] / H_AVG[71]
            + 3.608043e-09 * h[94] / H_AVG[94]
            + 3.48264e-09 * h[9] / H_AVG[9]
            + 3.063182e-09 * h[26] / H_AVG[26]
            - 2.742103e-09 * h[14] / H_AVG[14]
            + 2.709067e-09 * h[57] / H_AVG[57]
            + 2.568132e-09 * h[49] / H_AVG[49]
            - 2.521754e-09 * h[22] / H_AVG[22]
            + 2.517921e-09 * h[77] / H_AVG[77]
            + 2.321205e-09 * h[122] / H_AVG[122]
            + 1.874506e-09 * h[72] / H_AVG[72]
            + 1.80937e-09 * h[51] / H_AVG[51]
            - 1.773395e-09 * h[23] / H_AVG[23]
            - 1.228094e-09 * h[89] / H_AVG[89]
            + 1.196167e-09 * h[125] / H_AVG[125]
            - 1.121351e-09 * h[75] / H_AVG[75]
            + 9.697372e-10 * h[93] / H_AVG[93]
            - 9.664772e-10 * h[87] / H_AVG[87]
            - 8.775743e-10 * h[40] / H_AVG[40]
            - 8.057738e-10 * h[20] / H_AVG[20]
            + 7.25139e-10 * h[119] / H_AVG[119]
            - 6.611531e-10 * h[62] / H_AVG[62]
            - 5.925674e-10 * h[95] / H_AVG[95]
            - 5.453728e-10 * h[96] / H_AVG[96]
            + 3.25296e-10 * h[116] / H_AVG[116]
            + 3.131661e-10 * h[56] / H_AVG[56]
            + 2.14931e-10 * h[85] / H_AVG[85]
            + 1.277945e-10 * h[109] / H_AVG[109]
            - 1.114261e-10 * h[98] / H_AVG[98]
            - 9.037764e-11 * h[88] / H_AVG[88]
            - 5.764432e-11 * h[46] / H_AVG[46]
            - 5.612912e-11 * h[47] / H_AVG[47]
            - 3.489636e-11 * h[64] / H_AVG[64]
            + 1.394138e-11 * h[127] / H_AVG[127]
            + 1.304209e-12 * h[41] / H_AVG[41]
        ),
        -0.9789527 + T[5] * (   # class Hqql
            + 0.1771309 * h[120] / H_AVG[120]
            + 0.1277411 * h[24] / H_AVG[24]
            + 0.118388 * h[97] / H_AVG[97]
            + 0.1086773 * h[18] / H_AVG[18]
            + 0.09603066 * h[52] / H_AVG[52]
            + 0.09086493 * h[83] / H_AVG[83]
            - 0.08869751 * h[124] / H_AVG[124]
            - 0.0709015 * h[115] / H_AVG[115]
            + 0.04606947 * h[78] / H_AVG[78]
            + 0.02351377 * h[70] / H_AVG[70]
            + 0.014752 * h[68] / H_AVG[68]
            - 0.01466697 * h[81] / H_AVG[81]
            + 0.01330544 * h[104] / H_AVG[104]
            - 0.005248679 * h[16] / H_AVG[16]
            + 0.002378279 * h[25] / H_AVG[25]
            - 0.001192641 * h[123] / H_AVG[123]
            - 0.000327424 * h[0] / H_AVG[0]
            + 6.773918e-05 * h[21] / H_AVG[21]
            + 2.352892e-05 * h[90] / H_AVG[90]
            - 1.931313e-05 * h[99] / H_AVG[99]
            + 9.046623e-07 * h[84] / H_AVG[84]
            + 2.136879e-07 * h[55] / H_AVG[55]
            + 1.430247e-07 * h[79] / H_AVG[79]
            - 1.365339e-07 * h[105] / H_AVG[105]
            - 1.10909e-07 * h[106] / H_AVG[106]
            + 9.032437e-08 * h[113] / H_AVG[113]
            - 8.572158e-08 * h[34] / H_AVG[34]
            + 7.687459e-08 * h[73] / H_AVG[73]
            + 7.136103e-08 * h[10] / H_AVG[10]
            + 6.817366e-08 * h[11] / H_AVG[11]
            - 5.790632e-08 * h[6] / H_AVG[6]
            + 5.527769e-08 * h[86] / H_AVG[86]
            - 5.252734e-08 * h[32] / H_AVG[32]
            + 4.857054e-08 * h[17] / H_AVG[17]
            + 4.366057e-08 * h[38] / H_AVG[38]
            - 3.585871e-08 * h[42] / H_AVG[42]
            - 3.197327e-08 * h[54] / H_AVG[54]
            - 3.036465e-08 * h[29] / H_AVG[29]
            + 2.754705e-08 * h[118] / H_AVG[118]
            + 2.150858e-08 * h[48] / H_AVG[48]
            - 2.052408e-08 * h[45] / H_AVG[45]
            - 1.979395e-08 * h[39] / H_AVG[39]
            + 1.825264e-08 * h[27] / H_AVG[27]
            + 1.819975e-08 * h[63] / H_AVG[63]
            + 1.799622e-08 * h[30] / H_AVG[30]
            + 1.760014e-08 * h[43] / H_AVG[43]
            + 1.639508e-08 * h[121] / H_AVG[121]
            - 1.611021e-08 * h[80] / H_AVG[80]
            - 1.475747e-08 * h[126] / H_AVG[126]
            + 1.423931e-08 * h[60] / H_AVG[60]
            - 1.400684e-08 * h[61] / H_AVG[61]
            - 1.330236e-08 * h[102] / H_AVG[102]
            - 1.328034e-08 * h[107] / H_AVG[107]
            - 1.308419e-08 * h[7] / H_AVG[7]
            + 1.176927e-08 * h[92] / H_AVG[92]
            + 1.151004e-08 * h[35] / H_AVG[35]
            + 1.118888e-08 * h[15] / H_AVG[15]
            - 1.083449e-08 * h[59] / H_AVG[59]
            - 1.04516e-08 * h[101] / H_AVG[101]
            - 9.955544e-09 * h[28] / H_AVG[28]
            + 9.700666e-09 * h[103] / H_AVG[103]
            + 9.113707e-09 * h[69] / H_AVG[69]
            + 8.875839e-09 * h[67] / H_AVG[67]
            + 8.552846e-09 * h[1] / H_AVG[1]
            + 8.504227e-09 * h[100] / H_AVG[100]
            - 7.804038e-09 * h[110] / H_AVG[110]
            - 7.551809e-09 * h[3] / H_AVG[3]
            + 7.245726e-09 * h[37] / H_AVG[37]
            - 7.094133e-09 * h[50] / H_AVG[50]
            + 6.693039e-09 * h[36] / H_AVG[36]
            + 6.689909e-09 * h[44] / H_AVG[44]
            - 6.571824e-09 * h[12] / H_AVG[12]
            - 6.247554e-09 * h[114] / H_AVG[114]
            - 6.20096e-09 * h[2] / H_AVG[2]
            + 5.96231e-09 * h[58] / H_AVG[58]
            - 5.957758e-09 * h[112] / H_AVG[112]
            + 5.818846e-09 * h[76] / H_AVG[76]
            - 5.596095e-09 * h[13] / H_AVG[13]
            - 5.456766e-09 * h[19] / H_AVG[19]
            - 5.339371e-09 * h[111] / H_AVG[111]
            + 5.087378e-09 * h[74] / H_AVG[74]
            + 4.81181e-09 * h[108] / H_AVG[108]
            + 4.547195e-09 * h[5] / H_AVG[5]
            - 4.33798e-09 * h[8] / H_AVG[8]
            - 4.170148e-09 * h[91] / H_AVG[91]
            - 3.881692e-09 * h[4] / H_AVG[4]
            - 3.717416e-09 * h[117] / H_AVG[117]
            + 3.161078e-09 * h[53] / H_AVG[53]
            - 3.106336e-09 * h[66] / H_AVG[66]
            - 2.812619e-09 * h[33] / H_AVG[33]
            - 2.754021e-09 * h[65] / H_AVG[65]
            + 2.552805e-09 * h[82] / H_AVG[82]
            - 2.38035e-09 * h[31] / H_AVG[31]
            - 2.192082e-09 * h[71] / H_AVG[71]
            + 2.034911e-09 * h[94] / H_AVG[94]
            + 1.964869e-09 * h[9] / H_AVG[9]
            + 1.736369e-09 * h[26] / H_AVG[26]
            - 1.548664e-09 * h[14] / H_AVG[14]
            + 1.533791e-09 * h[57] / H_AVG[57]
            + 1.447514e-09 * h[49] / H_AVG[49]
            - 1.421995e-09 * h[22] / H_AVG[22]
            + 1.420025e-09 * h[77] / H_AVG[77]
            + 1.31346e-09 * h[122] / H_AVG[122]
            + 1.056612e-09 * h[72] / H_AVG[72]
            + 1.024636e-09 * h[51] / H_AVG[51]
            - 1.002501e-09 * h[23] / H_AVG[23]
            - 6.892405e-10 * h[89] / H_AVG[89]
            + 6.736823e-10 * h[125] / H_AVG[125]
            - 6.314345e-10 * h[75] / H_AVG[75]
            + 5.473292e-10 * h[93] / H_AVG[93]
            - 5.42858e-10 * h[87] / H_AVG[87]
            - 4.949122e-10 * h[40] / H_AVG[40]
            - 4.545718e-10 * h[20] / H_AVG[20]
            + 4.103968e-10 * h[119] / H_AVG[119]
            - 3.72038e-10 * h[62] / H_AVG[62]
            - 3.34184e-10 * h[95] / H_AVG[95]
            - 3.069611e-10 * h[96] / H_AVG[96]
            + 1.832447e-10 * h[116] / H_AVG[116]
            + 1.773048e-10 * h[56] / H_AVG[56]
            + 1.221066e-10 * h[85] / H_AVG[85]
            + 7.185904e-11 * h[109] / H_AVG[109]
            - 6.268791e-11 * h[98] / H_AVG[98]
            - 5.065924e-11 * h[88] / H_AVG[88]
            - 3.273064e-11 * h[46] / H_AVG[46]
            - 3.158775e-11 * h[47] / H_AVG[47]
            - 1.520338e-11 * h[64] / H_AVG[64]
            + 8.143573e-12 * h[127] / H_AVG[127]
            + 7.656811e-13 * h[41] / H_AVG[41]
        ),
        0.3035426 + T[6] * (   # class Zqq
            - 0.1207607 * h[124] / H_AVG[124]
            + 0.110965 * h[115] / H_AVG[115]
            + 0.1069827 * h[83] / H_AVG[83]
            + 0.10338 * h[24] / H_AVG[24]
            + 0.09659351 * h[104] / H_AVG[104]
            - 0.09400183 * h[97] / H_AVG[97]
            - 0.08790701 * h[120] / H_AVG[120]
            + 0.074097 * h[68] / H_AVG[68]
            - 0.05774486 * h[52] / H_AVG[52]
            + 0.05409223 * h[81] / H_AVG[81]
            - 0.04521298 * h[16] / H_AVG[16]
            - 0.02136537 * h[70] / H_AVG[70]
            + 0.01780887 * h[18] / H_AVG[18]
            - 0.006562093 * h[78] / H_AVG[78]
            + 0.001846545 * h[25] / H_AVG[25]
            + 0.0005248761 * h[123] / H_AVG[123]
            - 6.034837e-05 * h[0] / H_AVG[0]
            - 4.265215e-05 * h[99] / H_AVG[99]
            - 3.41026e-05 * h[21] / H_AVG[21]
            - 1.106582e-05 * h[90] / H_AVG[90]
            + 2.013677e-06 * h[84] / H_AVG[84]
            + 4.758845e-07 * h[55] / H_AVG[55]
            + 3.185662e-07 * h[79] / H_AVG[79]
            - 3.040981e-07 * h[105] / H_AVG[105]
            - 2.469484e-07 * h[106] / H_AVG[106]
            + 2.010127e-07 * h[113] / H_AVG[113]
            - 1.908806e-07 * h[34] / H_AVG[34]
            + 1.712241e-07 * h[73] / H_AVG[73]
            + 1.590302e-07 * h[10] / H_AVG[10]
            + 1.517127e-07 * h[11] / H_AVG[11]
            - 1.290603e-07 * h[6] / H_AVG[6]
            + 1.231297e-07 * h[86] / H_AVG[86]
            - 1.169855e-07 * h[32] / H_AVG[32]
            + 1.081085e-07 * h[17] / H_AVG[17]
            + 9.72733e-08 * h[38] / H_AVG[38]
            - 7.987903e-08 * h[42] / H_AVG[42]
            - 7.114723e-08 * h[54] / H_AVG[54]
            - 6.764285e-08 * h[29] / H_AVG[29]
            + 6.13284e-08 * h[118] / H_AVG[118]
            + 4.792907e-08 * h[48] / H_AVG[48]
            - 4.570033e-08 * h[45] / H_AVG[45]
            - 4.412065e-08 * h[39] / H_AVG[39]
            + 4.064422e-08 * h[27] / H_AVG[27]
            + 4.05283e-08 * h[63] / H_AVG[63]
            + 4.006718e-08 * h[30] / H_AVG[30]
            + 3.922313e-08 * h[43] / H_AVG[43]
            + 3.652861e-08 * h[121] / H_AVG[121]
            - 3.587895e-08 * h[80] / H_AVG[80]
            - 3.280178e-08 * h[126] / H_AVG[126]
            + 3.172215e-08 * h[60] / H_AVG[60]
            - 3.122067e-08 * h[61] / H_AVG[61]
            - 2.964176e-08 * h[102] / H_AVG[102]
            - 2.960971e-08 * h[107] / H_AVG[107]
            - 2.914965e-08 * h[7] / H_AVG[7]
            + 2.622934e-08 * h[92] / H_AVG[92]
            + 2.563757e-08 * h[35] / H_AVG[35]
            + 2.492102e-08 * h[15] / H_AVG[15]
            - 2.411455e-08 * h[59] / H_AVG[59]
            - 2.325924e-08 * h[101] / H_AVG[101]
            - 2.222613e-08 * h[28] / H_AVG[28]
            + 2.159475e-08 * h[103] / H_AVG[103]
            + 2.027384e-08 * h[69] / H_AVG[69]
            + 1.973107e-08 * h[67] / H_AVG[67]
            + 1.905967e-08 * h[1] / H_AVG[1]
            + 1.895537e-08 * h[100] / H_AVG[100]
            - 1.73956e-08 * h[110] / H_AVG[110]
            - 1.680283e-08 * h[3] / H_AVG[3]
            + 1.613454e-08 * h[37] / H_AVG[37]
            - 1.58107e-08 * h[50] / H_AVG[50]
            + 1.490973e-08 * h[44] / H_AVG[44]
            + 1.489679e-08 * h[36] / H_AVG[36]
            - 1.462417e-08 * h[12] / H_AVG[12]
            - 1.391217e-08 * h[114] / H_AVG[114]
            - 1.378567e-08 * h[2] / H_AVG[2]
            + 1.32655e-08 * h[58] / H_AVG[58]
            - 1.324362e-08 * h[112] / H_AVG[112]
            + 1.29479e-08 * h[76] / H_AVG[76]
            - 1.247774e-08 * h[13] / H_AVG[13]
            - 1.215176e-08 * h[19] / H_AVG[19]
            - 1.190189e-08 * h[111] / H_AVG[111]
            + 1.133297e-08 * h[74] / H_AVG[74]
            + 1.073841e-08 * h[108] / H_AVG[108]
            + 1.013713e-08 * h[5] / H_AVG[5]
            - 9.65212e-09 * h[8] / H_AVG[8]
            - 9.270618e-09 * h[91] / H_AVG[91]
            - 8.639135e-09 * h[4] / H_AVG[4]
            - 8.274982e-09 * h[117] / H_AVG[117]
            + 7.035121e-09 * h[53] / H_AVG[53]
            - 6.916397e-09 * h[66] / H_AVG[66]
            - 6.279999e-09 * h[33] / H_AVG[33]
            - 6.114896e-09 * h[65] / H_AVG[65]
            + 5.674521e-09 * h[82] / H_AVG[82]
            - 5.317074e-09 * h[31] / H_AVG[31]
            - 4.885696e-09 * h[71] / H_AVG[71]
            + 4.533598e-09 * h[94] / H_AVG[94]
            + 4.376512e-09 * h[9] / H_AVG[9]
            + 3.848994e-09 * h[26] / H_AVG[26]
            - 3.446297e-09 * h[14] / H_AVG[14]
            + 3.40329e-09 * h[57] / H_AVG[57]
            + 3.227139e-09 * h[49] / H_AVG[49]
            - 3.168014e-09 * h[22] / H_AVG[22]
            + 3.163966e-09 * h[77] / H_AVG[77]
            + 2.916446e-09 * h[122] / H_AVG[122]
            + 2.355795e-09 * h[72] / H_AVG[72]
            + 2.272418e-09 * h[51] / H_AVG[51]
            - 2.230055e-09 * h[23] / H_AVG[23]
            - 1.543729e-09 * h[89] / H_AVG[89]
            + 1.503167e-09 * h[125] / H_AVG[125]
            - 1.409263e-09 * h[75] / H_AVG[75]
            + 1.218581e-09 * h[93] / H_AVG[93]
            - 1.215013e-09 * h[87] / H_AVG[87]
            - 1.102763e-09 * h[40] / H_AVG[40]
            - 1.012565e-09 * h[20] / H_AVG[20]
            + 9.111089e-10 * h[119] / H_AVG[119]
            - 8.307373e-10 * h[62] / H_AVG[62]
            - 7.444832e-10 * h[95] / H_AVG[95]
            - 6.853136e-10 * h[96] / H_AVG[96]
            + 4.088269e-10 * h[116] / H_AVG[116]
            + 3.932699e-10 * h[56] / H_AVG[56]
            + 2.699955e-10 * h[85] / H_AVG[85]
            + 1.606051e-10 * h[109] / H_AVG[109]
            - 1.400502e-10 * h[98] / H_AVG[98]
            - 1.136275e-10 * h[88] / H_AVG[88]
            - 7.24326e-11 * h[46] / H_AVG[46]
            - 7.052977e-11 * h[47] / H_AVG[47]
            - 4.035553e-11 * h[64] / H_AVG[64]
            + 1.749965e-11 * h[127] / H_AVG[127]
            + 1.63505e-12 * h[41] / H_AVG[41]
        ),
        0.09887857 + T[7] * (   # class Wqq
            + 0.2622921 * h[97] / H_AVG[97]
            + 0.1233744 * h[83] / H_AVG[83]
            + 0.100237 * h[24] / H_AVG[24]
            - 0.09556739 * h[52] / H_AVG[52]
            + 0.09352647 * h[68] / H_AVG[68]
            - 0.07807258 * h[120] / H_AVG[120]
            + 0.05425992 * h[115] / H_AVG[115]
            + 0.05029653 * h[104] / H_AVG[104]
            - 0.04282395 * h[16] / H_AVG[16]
            - 0.03597606 * h[81] / H_AVG[81]
            - 0.02754433 * h[78] / H_AVG[78]
            - 0.01733586 * h[70] / H_AVG[70]
            + 0.01443137 * h[124] / H_AVG[124]
            - 0.002089228 * h[25] / H_AVG[25]
            + 0.001749912 * h[18] / H_AVG[18]
            - 0.0001721191 * h[0] / H_AVG[0]
            + 0.000171485 * h[123] / H_AVG[123]
            - 3.861978e-05 * h[99] / H_AVG[99]
            - 2.688136e-05 * h[21] / H_AVG[21]
            - 8.185674e-06 * h[90] / H_AVG[90]
            + 1.820965e-06 * h[84] / H_AVG[84]
            + 4.302092e-07 * h[55] / H_AVG[55]
            + 2.880663e-07 * h[79] / H_AVG[79]
            - 2.749771e-07 * h[105] / H_AVG[105]
            - 2.232825e-07 * h[106] / H_AVG[106]
            + 1.8179e-07 * h[113] / H_AVG[113]
            - 1.72596e-07 * h[34] / H_AVG[34]
            + 1.548398e-07 * h[73] / H_AVG[73]
            + 1.437514e-07 * h[10] / H_AVG[10]
            + 1.371831e-07 * h[11] / H_AVG[11]
            - 1.167043e-07 * h[6] / H_AVG[6]
            + 1.11351e-07 * h[86] / H_AVG[86]
            - 1.057798e-07 * h[32] / H_AVG[32]
            + 9.774291e-08 * h[17] / H_AVG[17]
            + 8.795954e-08 * h[38] / H_AVG[38]
            - 7.223608e-08 * h[42] / H_AVG[42]
            - 6.434319e-08 * h[54] / H_AVG[54]
            - 6.115936e-08 * h[29] / H_AVG[29]
            + 5.545313e-08 * h[118] / H_AVG[118]
            + 4.334098e-08 * h[48] / H_AVG[48]
            - 4.132534e-08 * h[45] / H_AVG[45]
            - 3.988481e-08 * h[39] / H_AVG[39]
            + 3.675297e-08 * h[27] / H_AVG[27]
            + 3.664452e-08 * h[63] / H_AVG[63]
            + 3.623261e-08 * h[30] / H_AVG[30]
            + 3.546523e-08 * h[43] / H_AVG[43]
            + 3.302691e-08 * h[121] / H_AVG[121]
            - 3.243993e-08 * h[80] / H_AVG[80]
            - 2.967518e-08 * h[126] / H_AVG[126]
            + 2.867633e-08 * h[60] / H_AVG[60]
            - 2.822227e-08 * h[61] / H_AVG[61]
            - 2.680169e-08 * h[102] / H_AVG[102]
            - 2.677602e-08 * h[107] / H_AVG[107]
            - 2.634607e-08 * h[7] / H_AVG[7]
            + 2.371548e-08 * h[92] / H_AVG[92]
            + 2.318569e-08 * h[35] / H_AVG[35]
            + 2.253429e-08 * h[15] / H_AVG[15]
            - 2.181317e-08 * h[59] / H_AVG[59]
            - 2.102987e-08 * h[101] / H_AVG[101]
            - 2.008514e-08 * h[28] / H_AVG[28]
            + 1.95281e-08 * h[103] / H_AVG[103]
            + 1.833018e-08 * h[69] / H_AVG[69]
            + 1.783865e-08 * h[67] / H_AVG[67]
            + 1.723167e-08 * h[1] / H_AVG[1]
            + 1.713839e-08 * h[100] / H_AVG[100]
            - 1.572481e-08 * h[110] / H_AVG[110]
            - 1.519714e-08 * h[3] / H_AVG[3]
            + 1.458884e-08 * h[37] / H_AVG[37]
            - 1.429844e-08 * h[50] / H_AVG[50]
            + 1.348066e-08 * h[44] / H_AVG[44]
            + 1.347054e-08 * h[36] / H_AVG[36]
            - 1.321813e-08 * h[12] / H_AVG[12]
            - 1.257827e-08 * h[114] / H_AVG[114]
            - 1.247401e-08 * h[2] / H_AVG[2]
            + 1.19949e-08 * h[58] / H_AVG[58]
            - 1.197606e-08 * h[112] / H_AVG[112]
            + 1.170754e-08 * h[76] / H_AVG[76]
            - 1.128131e-08 * h[13] / H_AVG[13]
            - 1.098755e-08 * h[19] / H_AVG[19]
            - 1.076092e-08 * h[111] / H_AVG[111]
            + 1.02469e-08 * h[74] / H_AVG[74]
            + 9.71064e-09 * h[108] / H_AVG[108]
            + 9.165868e-09 * h[5] / H_AVG[5]
            - 8.727367e-09 * h[8] / H_AVG[8]
            - 8.379909e-09 * h[91] / H_AVG[91]
            - 7.811445e-09 * h[4] / H_AVG[4]
            - 7.482396e-09 * h[117] / H_AVG[117]
            + 6.359995e-09 * h[53] / H_AVG[53]
            - 6.258784e-09 * h[66] / H_AVG[66]
            - 5.678946e-09 * h[33] / H_AVG[33]
            - 5.530597e-09 * h[65] / H_AVG[65]
            + 5.129953e-09 * h[82] / H_AVG[82]
            - 4.80963e-09 * h[31] / H_AVG[31]
            - 4.418691e-09 * h[71] / H_AVG[71]
            + 4.097587e-09 * h[94] / H_AVG[94]
            + 3.957284e-09 * h[9] / H_AVG[9]
            + 3.478576e-09 * h[26] / H_AVG[26]
            - 3.116712e-09 * h[14] / H_AVG[14]
            + 3.077437e-09 * h[57] / H_AVG[57]
            + 2.918042e-09 * h[49] / H_AVG[49]
            - 2.864996e-09 * h[22] / H_AVG[22]
            + 2.858561e-09 * h[77] / H_AVG[77]
            + 2.63713e-09 * h[122] / H_AVG[122]
            + 2.130109e-09 * h[72] / H_AVG[72]
            + 2.055675e-09 * h[51] / H_AVG[51]
            - 2.017183e-09 * h[23] / H_AVG[23]
            - 1.395897e-09 * h[89] / H_AVG[89]
            + 1.359001e-09 * h[125] / H_AVG[125]
            - 1.274429e-09 * h[75] / H_AVG[75]
            + 1.100997e-09 * h[93] / H_AVG[93]
            - 1.098031e-09 * h[87] / H_AVG[87]
            - 9.970841e-10 * h[40] / H_AVG[40]
            - 9.155842e-10 * h[20] / H_AVG[20]
            + 8.230911e-10 * h[119] / H_AVG[119]
            - 7.511276e-10 * h[62] / H_AVG[62]
            - 6.730955e-10 * h[95] / H_AVG[95]
            - 6.197223e-10 * h[96] / H_AVG[96]
            + 3.695977e-10 * h[116] / H_AVG[116]
            + 3.556136e-10 * h[56] / H_AVG[56]
            + 2.442111e-10 * h[85] / H_AVG[85]
            + 1.451788e-10 * h[109] / H_AVG[109]
            - 1.266204e-10 * h[98] / H_AVG[98]
            - 1.026919e-10 * h[88] / H_AVG[88]
            - 6.547424e-11 * h[46] / H_AVG[46]
            - 6.377437e-11 * h[47] / H_AVG[47]
            - 3.788888e-11 * h[64] / H_AVG[64]
            + 1.581591e-11 * h[127] / H_AVG[127]
            + 1.480054e-12 * h[41] / H_AVG[41]
        ),
        -0.4320669 + T[8] * (   # class Tbqq
            - 0.1797505 * h[104] / H_AVG[104]
            + 0.1506156 * h[16] / H_AVG[16]
            - 0.1346032 * h[83] / H_AVG[83]
            + 0.1251271 * h[120] / H_AVG[120]
            + 0.1149732 * h[68] / H_AVG[68]
            - 0.0840129 * h[18] / H_AVG[18]
            + 0.03752839 * h[78] / H_AVG[78]
            - 0.03528615 * h[24] / H_AVG[24]
            - 0.02903284 * h[52] / H_AVG[52]
            + 0.02819303 * h[97] / H_AVG[97]
            + 0.02596646 * h[81] / H_AVG[81]
            - 0.02561159 * h[115] / H_AVG[115]
            - 0.01848673 * h[70] / H_AVG[70]
            + 0.009644221 * h[124] / H_AVG[124]
            - 0.0008167369 * h[123] / H_AVG[123]
            + 0.0001789489 * h[25] / H_AVG[25]
            - 0.000124111 * h[0] / H_AVG[0]
            - 2.363047e-05 * h[99] / H_AVG[99]
            - 1.076374e-05 * h[21] / H_AVG[21]
            - 1.033218e-05 * h[90] / H_AVG[90]
            + 1.12245e-06 * h[84] / H_AVG[84]
            + 2.652452e-07 * h[55] / H_AVG[55]
            + 1.776386e-07 * h[79] / H_AVG[79]
            - 1.694736e-07 * h[105] / H_AVG[105]
            - 1.376073e-07 * h[106] / H_AVG[106]
            + 1.120636e-07 * h[113] / H_AVG[113]
            - 1.063844e-07 * h[34] / H_AVG[34]
            + 9.541884e-08 * h[73] / H_AVG[73]
            + 8.861578e-08 * h[10] / H_AVG[10]
            + 8.457006e-08 * h[11] / H_AVG[11]
            - 7.192393e-08 * h[6] / H_AVG[6]
            + 6.862967e-08 * h[86] / H_AVG[86]
            - 6.520331e-08 * h[32] / H_AVG[32]
            + 6.027696e-08 * h[17] / H_AVG[17]
            + 5.422028e-08 * h[38] / H_AVG[38]
            - 4.452779e-08 * h[42] / H_AVG[42]
            - 3.965728e-08 * h[54] / H_AVG[54]
            - 3.77016e-08 * h[29] / H_AVG[29]
            + 3.417945e-08 * h[118] / H_AVG[118]
            + 2.671652e-08 * h[48] / H_AVG[48]
            - 2.546653e-08 * h[45] / H_AVG[45]
            - 2.458356e-08 * h[39] / H_AVG[39]
            + 2.265418e-08 * h[27] / H_AVG[27]
            + 2.259016e-08 * h[63] / H_AVG[63]
            + 2.233331e-08 * h[30] / H_AVG[30]
            + 2.18577e-08 * h[43] / H_AVG[43]
            + 2.036095e-08 * h[121] / H_AVG[121]
            - 1.999805e-08 * h[80] / H_AVG[80]
            - 1.828519e-08 * h[126] / H_AVG[126]
            + 1.768151e-08 * h[60] / H_AVG[60]
            - 1.739629e-08 * h[61] / H_AVG[61]
            - 1.6516e-08 * h[102] / H_AVG[102]
            - 1.650346e-08 * h[107] / H_AVG[107]
            - 1.624117e-08 * h[7] / H_AVG[7]
            + 1.461597e-08 * h[92] / H_AVG[92]
            + 1.428242e-08 * h[35] / H_AVG[35]
            + 1.389012e-08 * h[15] / H_AVG[15]
            - 1.343716e-08 * h[59] / H_AVG[59]
            - 1.296357e-08 * h[101] / H_AVG[101]
            - 1.239032e-08 * h[28] / H_AVG[28]
            + 1.203318e-08 * h[103] / H_AVG[103]
            + 1.129701e-08 * h[69] / H_AVG[69]
            + 1.099705e-08 * h[67] / H_AVG[67]
            + 1.062388e-08 * h[1] / H_AVG[1]
            + 1.056459e-08 * h[100] / H_AVG[100]
            - 9.695857e-09 * h[110] / H_AVG[110]
            - 9.368219e-09 * h[3] / H_AVG[3]
            + 8.992781e-09 * h[37] / H_AVG[37]
            - 8.80956e-09 * h[50] / H_AVG[50]
            + 8.309748e-09 * h[44] / H_AVG[44]
            + 8.301796e-09 * h[36] / H_AVG[36]
            - 8.149783e-09 * h[12] / H_AVG[12]
            - 7.753832e-09 * h[114] / H_AVG[114]
            - 7.688301e-09 * h[2] / H_AVG[2]
            + 7.39411e-09 * h[58] / H_AVG[58]
            - 7.381249e-09 * h[112] / H_AVG[112]
            + 7.216846e-09 * h[76] / H_AVG[76]
            - 6.953272e-09 * h[13] / H_AVG[13]
            - 6.771794e-09 * h[19] / H_AVG[19]
            - 6.633868e-09 * h[111] / H_AVG[111]
            + 6.317271e-09 * h[74] / H_AVG[74]
            + 5.985303e-09 * h[108] / H_AVG[108]
            + 5.648861e-09 * h[5] / H_AVG[5]
            - 5.379303e-09 * h[8] / H_AVG[8]
            - 5.164786e-09 * h[91] / H_AVG[91]
            - 4.815332e-09 * h[4] / H_AVG[4]
            - 4.611386e-09 * h[117] / H_AVG[117]
            + 3.921009e-09 * h[53] / H_AVG[53]
            - 3.854528e-09 * h[66] / H_AVG[66]
            - 3.500418e-09 * h[33] / H_AVG[33]
            - 3.40937e-09 * h[65] / H_AVG[65]
            + 3.162325e-09 * h[82] / H_AVG[82]
            - 2.96246e-09 * h[31] / H_AVG[31]
            - 2.72276e-09 * h[71] / H_AVG[71]
            + 2.527861e-09 * h[94] / H_AVG[94]
            + 2.439186e-09 * h[9] / H_AVG[9]
            + 2.14643e-09 * h[26] / H_AVG[26]
            - 1.920676e-09 * h[14] / H_AVG[14]
            + 1.897742e-09 * h[57] / H_AVG[57]
            + 1.798713e-09 * h[49] / H_AVG[49]
            - 1.765666e-09 * h[22] / H_AVG[22]
            + 1.763439e-09 * h[77] / H_AVG[77]
            + 1.625472e-09 * h[122] / H_AVG[122]
            + 1.312959e-09 * h[72] / H_AVG[72]
            + 1.266949e-09 * h[51] / H_AVG[51]
            - 1.241092e-09 * h[23] / H_AVG[23]
            - 8.602922e-10 * h[89] / H_AVG[89]
            + 8.377635e-10 * h[125] / H_AVG[125]
            - 7.853102e-10 * h[75] / H_AVG[75]
            + 6.790026e-10 * h[93] / H_AVG[93]
            - 6.770048e-10 * h[87] / H_AVG[87]
            - 6.146331e-10 * h[40] / H_AVG[40]
            - 5.643319e-10 * h[20] / H_AVG[20]
            + 5.078383e-10 * h[119] / H_AVG[119]
            - 4.629169e-10 * h[62] / H_AVG[62]
            - 4.148346e-10 * h[95] / H_AVG[95]
            - 3.818307e-10 * h[96] / H_AVG[96]
            + 2.278252e-10 * h[116] / H_AVG[116]
            + 2.192434e-10 * h[56] / H_AVG[56]
            + 1.505907e-10 * h[85] / H_AVG[85]
            + 8.947189e-11 * h[109] / H_AVG[109]
            - 7.806816e-11 * h[98] / H_AVG[98]
            - 6.322265e-11 * h[88] / H_AVG[88]
            - 4.03739e-11 * h[46] / H_AVG[46]
            - 3.930381e-11 * h[47] / H_AVG[47]
            - 2.164357e-11 * h[64] / H_AVG[64]
            + 9.772119e-12 * h[127] / H_AVG[127]
            + 9.131072e-13 * h[41] / H_AVG[41]
        ),
        -1.495433 + T[9] * (   # class Tbl
            + 0.2437707 * h[16] / H_AVG[16]
            + 0.1116319 * h[18] / H_AVG[18]
            + 0.1102233 * h[24] / H_AVG[24]
            - 0.09450948 * h[104] / H_AVG[104]
            - 0.0785878 * h[124] / H_AVG[124]
            - 0.0742277 * h[115] / H_AVG[115]
            + 0.07273449 * h[97] / H_AVG[97]
            + 0.06956557 * h[52] / H_AVG[52]
            - 0.05378933 * h[68] / H_AVG[68]
            + 0.0335242 * h[78] / H_AVG[78]
            + 0.01882459 * h[70] / H_AVG[70]
            + 0.01750113 * h[120] / H_AVG[120]
            - 0.0115369 * h[83] / H_AVG[83]
            - 0.00481494 * h[81] / H_AVG[81]
            + 0.002477666 * h[25] / H_AVG[25]
            + 0.001864266 * h[123] / H_AVG[123]
            - 0.000329107 * h[0] / H_AVG[0]
            + 5.470321e-05 * h[21] / H_AVG[21]
            - 1.608834e-05 * h[99] / H_AVG[99]
            + 1.372869e-05 * h[90] / H_AVG[90]
            + 7.702047e-07 * h[84] / H_AVG[84]
            + 1.818686e-07 * h[55] / H_AVG[55]
            + 1.217515e-07 * h[79] / H_AVG[79]
            - 1.163232e-07 * h[105] / H_AVG[105]
            - 9.442112e-08 * h[106] / H_AVG[106]
            + 7.692993e-08 * h[113] / H_AVG[113]
            - 7.298705e-08 * h[34] / H_AVG[34]
            + 6.546403e-08 * h[73] / H_AVG[73]
            + 6.075596e-08 * h[10] / H_AVG[10]
            + 5.803931e-08 * h[11] / H_AVG[11]
            - 4.930472e-08 * h[6] / H_AVG[6]
            + 4.707709e-08 * h[86] / H_AVG[86]
            - 4.474949e-08 * h[32] / H_AVG[32]
            + 4.134965e-08 * h[17] / H_AVG[17]
            + 3.716983e-08 * h[38] / H_AVG[38]
            - 3.054739e-08 * h[42] / H_AVG[42]
            - 2.722334e-08 * h[54] / H_AVG[54]
            - 2.585627e-08 * h[29] / H_AVG[29]
            + 2.345149e-08 * h[118] / H_AVG[118]
            + 1.831252e-08 * h[48] / H_AVG[48]
            - 1.74739e-08 * h[45] / H_AVG[45]
            - 1.685418e-08 * h[39] / H_AVG[39]
            + 1.554302e-08 * h[27] / H_AVG[27]
            + 1.549431e-08 * h[63] / H_AVG[63]
            + 1.533314e-08 * h[30] / H_AVG[30]
            + 1.498642e-08 * h[43] / H_AVG[43]
            + 1.39576e-08 * h[121] / H_AVG[121]
            - 1.371444e-08 * h[80] / H_AVG[80]
            - 1.255878e-08 * h[126] / H_AVG[126]
            + 1.212384e-08 * h[60] / H_AVG[60]
            - 1.192345e-08 * h[61] / H_AVG[61]
            - 1.132889e-08 * h[102] / H_AVG[102]
            - 1.130665e-08 * h[107] / H_AVG[107]
            - 1.113978e-08 * h[7] / H_AVG[7]
            + 1.002105e-08 * h[92] / H_AVG[92]
            + 9.803932e-09 * h[35] / H_AVG[35]
            + 9.522421e-09 * h[15] / H_AVG[15]
            - 9.222967e-09 * h[59] / H_AVG[59]
            - 8.899586e-09 * h[101] / H_AVG[101]
            - 8.478328e-09 * h[28] / H_AVG[28]
            + 8.259375e-09 * h[103] / H_AVG[103]
            + 7.762867e-09 * h[69] / H_AVG[69]
            + 7.557132e-09 * h[67] / H_AVG[67]
            + 7.279316e-09 * h[1] / H_AVG[1]
            + 7.23776e-09 * h[100] / H_AVG[100]
            - 6.644868e-09 * h[110] / H_AVG[110]
            - 6.434521e-09 * h[3] / H_AVG[3]
            + 6.170677e-09 * h[37] / H_AVG[37]
            - 6.03802e-09 * h[50] / H_AVG[50]
            + 5.697929e-09 * h[36] / H_AVG[36]
            + 5.696782e-09 * h[44] / H_AVG[44]
            - 5.596564e-09 * h[12] / H_AVG[12]
            - 5.319769e-09 * h[114] / H_AVG[114]
            - 5.282315e-09 * h[2] / H_AVG[2]
            + 5.077259e-09 * h[58] / H_AVG[58]
            - 5.07396e-09 * h[112] / H_AVG[112]
            + 4.956322e-09 * h[76] / H_AVG[76]
            - 4.763459e-09 * h[13] / H_AVG[13]
            - 4.645559e-09 * h[19] / H_AVG[19]
            - 4.545156e-09 * h[111] / H_AVG[111]
            + 4.332168e-09 * h[74] / H_AVG[74]
            + 4.095951e-09 * h[108] / H_AVG[108]
            + 3.871394e-09 * h[5] / H_AVG[5]
            - 3.69316e-09 * h[8] / H_AVG[8]
            - 3.551045e-09 * h[91] / H_AVG[91]
            - 3.306669e-09 * h[4] / H_AVG[4]
            - 3.167004e-09 * h[117] / H_AVG[117]
            + 2.689151e-09 * h[53] / H_AVG[53]
            - 2.645392e-09 * h[66] / H_AVG[66]
            - 2.39465e-09 * h[33] / H_AVG[33]
            - 2.345686e-09 * h[65] / H_AVG[65]
            + 2.174711e-09 * h[82] / H_AVG[82]
            - 2.028912e-09 * h[31] / H_AVG[31]
            - 1.865985e-09 * h[71] / H_AVG[71]
            + 1.732752e-09 * h[94] / H_AVG[94]
            + 1.67272e-09 * h[9] / H_AVG[9]
            + 1.478762e-09 * h[26] / H_AVG[26]
            - 1.318278e-09 * h[14] / H_AVG[14]
            + 1.306624e-09 * h[57] / H_AVG[57]
            + 1.232475e-09 * h[49] / H_AVG[49]
            - 1.210372e-09 * h[22] / H_AVG[22]
            + 1.207273e-09 * h[77] / H_AVG[77]
            + 1.11893e-09 * h[122] / H_AVG[122]
            + 8.994721e-10 * h[72] / H_AVG[72]
            + 8.72986e-10 * h[51] / H_AVG[51]
            - 8.544828e-10 * h[23] / H_AVG[23]
            - 5.863911e-10 * h[89] / H_AVG[89]
            + 5.732718e-10 * h[125] / H_AVG[125]
            - 5.375886e-10 * h[75] / H_AVG[75]
            + 4.6595e-10 * h[93] / H_AVG[93]
            - 4.620464e-10 * h[87] / H_AVG[87]
            - 4.214403e-10 * h[40] / H_AVG[40]
            - 3.871246e-10 * h[20] / H_AVG[20]
            + 3.494647e-10 * h[119] / H_AVG[119]
            - 3.167113e-10 * h[62] / H_AVG[62]
            - 2.844288e-10 * h[95] / H_AVG[95]
            - 2.612236e-10 * h[96] / H_AVG[96]
            + 1.559845e-10 * h[116] / H_AVG[116]
            + 1.509882e-10 * h[56] / H_AVG[56]
            + 1.040244e-10 * h[85] / H_AVG[85]
            + 6.114416e-11 * h[109] / H_AVG[109]
            - 5.334724e-11 * h[98] / H_AVG[98]
            - 4.308522e-11 * h[88] / H_AVG[88]
            - 2.789109e-11 * h[46] / H_AVG[46]
            - 2.68926e-11 * h[47] / H_AVG[47]
            - 1.417082e-11 * h[64] / H_AVG[64]
            + 6.966104e-12 * h[127] / H_AVG[127]
            + 6.554378e-13 * h[41] / H_AVG[41]
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
