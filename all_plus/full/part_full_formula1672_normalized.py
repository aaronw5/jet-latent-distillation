"""Particle Transformer (ParT) jet tagger, JetClass, input features 'full': tuned on the network's predictions (from 100 if-statements per neuron; the smallest without loss on the validation jets), with normalized weights (how much each one matters), as if-statements.

Input:  the particles of a jet (up to 128), hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
        energy: each particle's energy [GeV]; jet_pt, jet_eta, jet_energy: the jet's pT [GeV], pseudorapidity, energy [GeV].
Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the training jets, so share_k is the fraction of the neuron's
             average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron 16:  11.8%
  neuron 24:   9.1%
  neuron 18:   9.1%
  neuron 97:   8.9%
  neuron 104:   8.8%
  neuron 120:   8.5%
  neuron 115:   7.7%
  neuron 68:   7.4%
  neuron 52:   6.4%
  neuron 83:   5.5%
  neuron 124:   5.2%
  neuron 78:   4.6%
  neuron 81:   3.1%
  neuron 70:   1.4%
  neuron 90:   0.8%
  neuron 25:   0.8%
  neuron 123:   0.6%
  neuron  0:   0.0%
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

Whole test file (2,000,000 jets): accuracy 75.38% (the network: 86.03%); same class as the network for 80.21% of jets.

Quantities:
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b05                 e3/e2² with β = 0.5
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3_b05                 e4·e2/e3² with β = 0.5
  Q.C3_b2                  e4·e2/e3² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b05                 e3/e2³ with β = 0.5
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.D3_b05                 e4·e2³/e3³ with β = 0.5
  Q.D3_b2                  e4·e2³/e3³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M2_b05                 ₁e₃/e2 with β = 0.5
  Q.M2_b2                  ₁e₃/e2 with β = 2
  Q.M3                     generalized ECF ratio M3
  Q.M3_b05                 ₁e₄/₁e₃ with β = 0.5
  Q.M3_b2                  ₁e₄/₁e₃ with β = 2
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N2_b05                 ₂e₃/(e2)² with β = 0.5
  Q.N2_b2                  ₂e₃/(e2)² with β = 2
  Q.N3_b05                 ₂e₄/(₁e₃)² with β = 0.5
  Q.ak02_1_n_disp3         hardest anti-kT 0.2 subjet: number of its tracks with d0/σ > 3
  Q.ak02_1_n_lep           hardest anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_1_z               pT share of the hardest anti-kT 0.2 subjet (0 if none)
  Q.ak02_2_charge          2nd anti-kT 0.2 subjet: Σ q √pT / √(Σ pT) of its particles
  Q.ak02_2_n_lep           2nd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_2_sd0_1           2nd anti-kT 0.2 subjet: the largest signed d0/σ among its tracks (0 if none)
  Q.ak02_2_sd0_3           2nd anti-kT 0.2 subjet: the 3rd largest signed d0/σ among its tracks (0 if fewer)
  Q.ak02_2_z               pT share of the 2nd anti-kT 0.2 subjet (0 if none)
  Q.ak02_3_n_disp3         3rd anti-kT 0.2 subjet: number of its tracks with d0/σ > 3
  Q.ak02_3_n_lep           3rd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_3_z               pT share of the 3rd anti-kT 0.2 subjet (0 if none)
  Q.ak02_3_z_disp3         3rd anti-kT 0.2 subjet: pT share (of the jet) of its tracks with d0/σ > 3
  Q.ak02_dr12              distance between the pT-weighted centres of the hardest and 2nd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr13              distance between the pT-weighted centres of the hardest and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr23              distance between the pT-weighted centres of the 2nd and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_min12_jp          the smaller of the two hardest anti-kT 0.2 subjets: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.ak02_min12_n_disp3     the smaller of the two hardest anti-kT 0.2 subjets: number of its tracks with d0/σ > 3
  Q.ak02_n                 number of anti-kT R = 0.2 subjets with pT > 10 GeV
  Q.charge_52              charge of particle 52 (ParT input; empty slot: 0)
  Q.d0err_0                σ(d0), clipped to [0, 1] of particle 0 (ParT input; empty slot: 0)
  Q.d0err_52               σ(d0), clipped to [0, 1] of particle 52 (ParT input; empty slot: 0)
  Q.dc_1_n_disp3           hardest prong: number of its tracks with d0/σ > 3
  Q.dc_1_n_lep             hardest prong: number of its electrons and muons
  Q.dc_1_sd0_1             hardest prong: the largest signed d0/σ among its tracks (0 if none)
  Q.dc_1_z                 pT share of the hardest prong (0 if none)
  Q.dc_1_z_disp3           hardest prong: pT share (of the jet) of its tracks with d0/σ > 3
  Q.dc_2_jp                2nd prong: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.dc_2_mass_disp3        2nd prong: mass (E) of its tracks with d0/σ > 3 [GeV]
  Q.dc_2_n_disp3           2nd prong: number of its tracks with d0/σ > 3
  Q.dc_2_n_lep             2nd prong: number of its electrons and muons
  Q.dc_2_z                 pT share of the 2nd prong (0 if none)
  Q.dc_3_charge            3rd prong: Σ q √pT / √(Σ pT) of its particles
  Q.dc_3_n_lep             3rd prong: number of its electrons and muons
  Q.dc_3_z                 pT share of the 3rd prong (0 if none)
  Q.dc_4_mass_disp3        4th prong: mass (E) of its tracks with d0/σ > 3 [GeV]
  Q.dc_4_z                 pT share of the 4th prong (0 if none)
  Q.dc_mass_disp_2nd       second largest displaced-track mass among the 4 hardest prongs [GeV]
  Q.dc_n                   number of prongs: reverse the C/A tree; ΔR ≤ 0.1 is a prong, a branch with < 10 % of the jet pT is dropped, else both branches are declustered
  Q.dc_ntag                number of the 4 hardest prongs whose 2nd largest d0/σ is above 3
  Q.dc_pair_mass_min       smallest mass (E) of two of the 4 hardest prongs [GeV] (0 if fewer than 2)
  Q.dc_split1_z            z (of the jet pT) of the hardest hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_split2_dr           ΔR of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_split2_mass         mass of the splitting node [GeV] of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_tag_2nd             second largest 2nd-largest d0/σ among the 4 hardest prongs (double tag)
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr02                   ΔR between particles 0 and 2
  Q.dr12                   ΔR between particles 1 and 2
  Q.dr_1                   ΔR from the jet axis of particle 1 (ParT input; empty slot: 0)
  Q.dr_10                  ΔR from the jet axis of particle 10 (ParT input; empty slot: 0)
  Q.dr_15                  ΔR from the jet axis of particle 15 (ParT input; empty slot: 0)
  Q.dr_19                  ΔR from the jet axis of particle 19 (ParT input; empty slot: 0)
  Q.dr_28                  ΔR from the jet axis of particle 28 (ParT input; empty slot: 0)
  Q.dr_29                  ΔR from the jet axis of particle 29 (ParT input; empty slot: 0)
  Q.dr_3                   ΔR from the jet axis of particle 3 (ParT input; empty slot: 0)
  Q.dr_30                  ΔR from the jet axis of particle 30 (ParT input; empty slot: 0)
  Q.dr_50                  ΔR from the jet axis of particle 50 (ParT input; empty slot: 0)
  Q.dr_52                  ΔR from the jet axis of particle 52 (ParT input; empty slot: 0)
  Q.dr_77                  ΔR from the jet axis of particle 77 (ParT input; empty slot: 0)
  Q.dr_8                   ΔR from the jet axis of particle 8 (ParT input; empty slot: 0)
  Q.dr_max_012             largest distance among the 3 hardest
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.dzerr_0                σ(dz), clipped to [0, 1] of particle 0 (ParT input; empty slot: 0)
  Q.dzerr_37               σ(dz), clipped to [0, 1] of particle 37 (ParT input; empty slot: 0)
  Q.dzerr_60               σ(dz), clipped to [0, 1] of particle 60 (ParT input; empty slot: 0)
  Q.dzerr_8                σ(dz), clipped to [0, 1] of particle 8 (ParT input; empty slot: 0)
  Q.dzerr_9                σ(dz), clipped to [0, 1] of particle 9 (ParT input; empty slot: 0)
  Q.e2_b2                  energy correlation e2 with β = 2
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.e3                     energy correlation e3 (β=1, 32 hardest)
  Q.e3_b05                 energy correlation e3 with β = 0.5
  Q.e3_b2                  energy correlation e3 with β = 2
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 16 hardest)
  Q.e4_b05                 energy correlation e4 with β = 0.5
  Q.e4_b2                  energy correlation e4 with β = 2
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.ecf_g31                generalized energy correlation ₁e₃ (β=1; products of the smallest angles)
  Q.ecf_g41                generalized energy correlation ₁e₄ (β=1; products of the smallest angles)
  Q.ecf_g42                generalized energy correlation ₂e₄ (β=1; products of the smallest angles)
  Q.ecf_g43                generalized energy correlation ₃e₄ (β=1; products of the smallest angles)
  Q.eta_1                  Δη of particle 1 (ParT input; empty slot: 0)
  Q.eta_11                 Δη of particle 11 (ParT input; empty slot: 0)
  Q.eta_18                 Δη of particle 18 (ParT input; empty slot: 0)
  Q.eta_28                 Δη of particle 28 (ParT input; empty slot: 0)
  Q.eta_29                 Δη of particle 29 (ParT input; empty slot: 0)
  Q.eta_4                  Δη of particle 4 (ParT input; empty slot: 0)
  Q.eta_60                 Δη of particle 60 (ParT input; empty slot: 0)
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.ischhad_0              1 if a charged hadron of particle 0 (ParT input; empty slot: 0)
  Q.ischhad_4              1 if a charged hadron of particle 4 (ParT input; empty slot: 0)
  Q.iselectron_1           1 if an electron of particle 1 (ParT input; empty slot: 0)
  Q.iselectron_10          1 if an electron of particle 10 (ParT input; empty slot: 0)
  Q.iselectron_12          1 if an electron of particle 12 (ParT input; empty slot: 0)
  Q.iselectron_29          1 if an electron of particle 29 (ParT input; empty slot: 0)
  Q.iselectron_3           1 if an electron of particle 3 (ParT input; empty slot: 0)
  Q.ismuon_14              1 if a muon of particle 14 (ParT input; empty slot: 0)
  Q.ismuon_3               1 if a muon of particle 3 (ParT input; empty slot: 0)
  Q.ismuon_4               1 if a muon of particle 4 (ParT input; empty slot: 0)
  Q.isnhad_29              1 if a neutral hadron of particle 29 (ParT input; empty slot: 0)
  Q.isnhad_34              1 if a neutral hadron of particle 34 (ParT input; empty slot: 0)
  Q.isnhad_38              1 if a neutral hadron of particle 38 (ParT input; empty slot: 0)
  Q.isnhad_8               1 if a neutral hadron of particle 8 (ParT input; empty slot: 0)
  Q.isphoton_19            1 if a photon of particle 19 (ParT input; empty slot: 0)
  Q.isphoton_32            1 if a photon of particle 32 (ParT input; empty slot: 0)
  Q.isphoton_53            1 if a photon of particle 53 (ParT input; empty slot: 0)
  Q.jd_3d_4                the 4th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_5                the 5th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_6                the 6th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_mass_d3_over_z      mass (E) of the tracks with |d0/σ| > 3 over their pT share [GeV] (0 if none)
  Q.jd_n_d3_pt1            number of tracks with |d0/σ| > 3 and pT > 1 GeV
  Q.jd_sum_abs_sd0_top3    sum of the 3 largest |d0/σ|
  Q.jd_sum_abs_sd0_top5    sum of the 5 largest |d0/σ|
  Q.jet_abs_eta            absolute pseudorapidity of the jet axis
  Q.jet_charge             pT-weighted jet charge (κ = 1)
  Q.jet_charge_k03         jet charge with κ = 0.3
  Q.jet_charge_k05         jet charge with κ = 0.5
  Q.jet_e                  jet energy [GeV]
  Q.kt2_1_n_disp3          hardest kT subjet: number of its tracks with d0/σ > 3
  Q.kt2_1_sd0_1            hardest kT subjet: the largest signed d0/σ among its tracks (0 if none)
  Q.kt2_1_sd0_3            hardest kT subjet: the 3rd largest signed d0/σ among its tracks (0 if fewer)
  Q.kt2_1_z                pT share of the hardest kT subjet (0 if none)
  Q.kt2_2_charge           2nd kT subjet: Σ q √pT / √(Σ pT) of its particles
  Q.kt2_2_n_lep            2nd kT subjet: number of its electrons and muons
  Q.kt2_2_z_disp3          2nd kT subjet: pT share (of the jet) of its tracks with d0/σ > 3
  Q.kt2_dr12               distance between the pT-weighted centres of the hardest and 2nd kT subjet (0 if missing)
  Q.kt2_min12_jp           the smaller of the two hardest kT subjets: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.kt2_min12_mass_disp3   the smaller of the two hardest kT subjets: mass (E) of its tracks with d0/σ > 3 [GeV]
  Q.kt2_min12_n_disp3      the smaller of the two hardest kT subjets: number of its tracks with d0/σ > 3
  Q.kt2_min12_sd0_1        the smaller of the two hardest kT subjets: the largest signed d0/σ among its tracks (0 if none)
  Q.ktd_ln_d34             ln of the exclusive-kT merging scale from 4 to 3 subjets, min(pT²)ΔR², over (Σ pT)² (0 if fewer particles)
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
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
  Q.lne_1                  ln E [GeV] of particle 1 (ParT input; empty slot: ln 1e-8)
  Q.lne_10                 ln E [GeV] of particle 10 (ParT input; empty slot: ln 1e-8)
  Q.lne_16                 ln E [GeV] of particle 16 (ParT input; empty slot: ln 1e-8)
  Q.lne_17                 ln E [GeV] of particle 17 (ParT input; empty slot: ln 1e-8)
  Q.lne_3                  ln E [GeV] of particle 3 (ParT input; empty slot: ln 1e-8)
  Q.lne_4                  ln E [GeV] of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lne_5                  ln E [GeV] of particle 5 (ParT input; empty slot: ln 1e-8)
  Q.lne_8                  ln E [GeV] of particle 8 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_3               ln(E / E of the jet) of particle 3 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_4               ln(E / E of the jet) of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_42              ln(E / E of the jet) of particle 42 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_65              ln(E / E of the jet) of particle 65 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_1                 ln pT [GeV] of particle 1 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_30                ln pT [GeV] of particle 30 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_78                ln pT [GeV] of particle 78 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_2              ln(pT / pT of the jet) of particle 2 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_25             ln(pT / pT of the jet) of particle 25 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_40             ln(pT / pT of the jet) of particle 40 (ParT input; empty slot: ln 1e-8)
  Q.lund1_lndelta          ln Δ of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund1_lnz              ln z of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund2_lndelta          ln Δ of the 2. primary C/A splitting (ln 1e-8 if none)
  Q.lund2_lnkt             ln kT of the 2. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lndelta          ln Δ of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lnkt             ln kT of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lnz              ln z of the 3. primary C/A splitting (ln 1e-8 if none)
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
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_abs_d0             largest |d0| among the charged particles [mm]
  Q.max_abs_dz             largest |dz| among the charged particles [mm]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.mean_eta               pT-weighted mean Δη
  Q.min_pair_mass          smallest pair mass among particles 0, 1, 2 [GeV]
  Q.mres_pruned_mass       pruned mass: C/A tree, a merge with z < 0.1 and ΔR > m/pT of the jet keeps only its harder branch [GeV]
  Q.mres_sd_mass_b0z005    soft-drop mass, C/A, β = 0, z_cut = 0.05 [GeV]
  Q.mres_sd_mass_b0z02     soft-drop mass, C/A, β = 0, z_cut = 0.2 [GeV]
  Q.mres_sd_mass_b1z01     soft-drop mass, C/A, β = 1, z_cut = 0.1 [GeV]
  Q.mres_sd_mass_b2z01     soft-drop mass, C/A, β = 2, z_cut = 0.1 [GeV]
  Q.mres_sd_prong_mass1    mass of the harder branch at the soft-drop splitting (β = 0, z_cut = 0.1) [GeV]
  Q.mres_sd_prong_mass2    mass of the softer branch at the soft-drop splitting (β = 0, z_cut = 0.1) [GeV]
  Q.mres_sd_rg_b0z005      soft-drop R_g, β = 0, z_cut = 0.05
  Q.mres_sd_rg_b1z01       soft-drop R_g, β = 1, z_cut = 0.1
  Q.mres_sd_rg_b2z01       soft-drop R_g, β = 2, z_cut = 0.1
  Q.mres_sd_zg_b0z005      soft-drop z_g, β = 0, z_cut = 0.05
  Q.mres_sd_zg_b1z01       soft-drop z_g, β = 1, z_cut = 0.1
  Q.mres_sd_zg_b2z01       soft-drop z_g, β = 2, z_cut = 0.1
  Q.n_charged_had          number of charged hadrons
  Q.n_charged_pt_above_1   number of charged particles with pT > 1 GeV
  Q.n_charged_pt_above_10  number of charged particles with pT > 10 GeV
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
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
  Q.n_sd0_above_10         number of charged particles with d0 significance > 10
  Q.n_sd0_above_2          number of charged particles with d0 significance > 2
  Q.n_sd0_above_3          number of charged particles with d0 significance > 3
  Q.n_sd0_above_5          number of charged particles with d0 significance > 5
  Q.n_sdz_above_2          number of charged particles with dz significance > 2
  Q.n_sdz_above_5          number of charged particles with dz significance > 5
  Q.nca_kt_above_10        number of C/A subjets when every branching with kT = min(pT)·ΔR > 10 GeV is split
  Q.nca_kt_above_2         number of C/A subjets when every branching with kT = min(pT)·ΔR > 2 GeV is split
  Q.nca_sj4_pair2nd_over_mass second largest mass of two of the 4 subjets over the jet mass
  Q.nca_sj4_pair_mass_2nd  second largest mass of two of the 4 subjets [GeV]
  Q.nca_sj4_pairmax_over_mass largest mass of two of the 4 subjets over the jet mass
  Q.pair_max_lnkt          largest ln kT among all pairs
  Q.pair_max_lnm2          largest ln m² among all pairs
  Q.pair_mean_lndelta      zᵢzⱼ-weighted mean of ln ΔRᵢⱼ over all pairs
  Q.pair_mean_lnkt         zᵢzⱼ-weighted mean of ln kT = ln(min(pTᵢ, pTⱼ)·ΔRᵢⱼ) over all pairs
  Q.pair_mean_lnm2         zᵢzⱼ-weighted mean of ln mᵢⱼ² (massless) over all pairs
  Q.pair_mean_lnz          zᵢzⱼ-weighted mean of ln z = ln(min(pTᵢ, pTⱼ)/(pTᵢ + pTⱼ)) over all pairs
  Q.phi_1                  Δφ of particle 1 (ParT input; empty slot: 0)
  Q.phi_3                  Δφ of particle 3 (ParT input; empty slot: 0)
  Q.phi_4                  Δφ of particle 4 (ParT input; empty slot: 0)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pz_lnd0                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -9 < ln ΔRᵢⱼ ≤ -3
  Q.pz_lnd1                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -3 < ln ΔRᵢⱼ ≤ -2
  Q.pz_lnd2                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -2 < ln ΔRᵢⱼ ≤ -1
  Q.pz_lnd3                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -1 < ln ΔRᵢⱼ ≤ 9
  Q.pz_lnkt0               Σ zᵢzⱼ over the pairs of the 40 hardest particles with -9 < ln kT ≤ 0
  Q.pz_lnkt1               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 0 < ln kT ≤ 1
  Q.pz_lnkt2               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 1 < ln kT ≤ 2
  Q.pz_lnkt3               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 2 < ln kT ≤ 3
  Q.pz_lnkt4               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 3 < ln kT ≤ 9
  Q.sd_mass                soft-drop groomed mass, C/A on the 128 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_nremoved            number of branches removed by soft drop
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
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmax_over_m     largest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj4_dr_min             smallest distance among the 4 subjet axes
  Q.sj4_pair_mass_max      largest mass of two of the 4 subjets [GeV]
  Q.sj4_pair_mass_min      smallest mass of two of the 4 subjets [GeV]
  Q.sjf_2_1_mass_d3        subjet 1 of 2 (k-means axes, by pT): mass (E) of the tracks with |d0/σ| > 3 [GeV]
  Q.sjf_2_1_max3d          subjet 1 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_1_maxsd0         subjet 1 of 2 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_2_1_n_d3           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_2_1_n_d5           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_2_1_z_d3           subjet 1 of 2 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_2_2_max3d          subjet 2 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_2_maxsd0         subjet 2 of 2 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_2_2_n_d3           subjet 2 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_2_2_n_d5           subjet 2 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_2_2_z_d3           subjet 2 of 2 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_3_1_max3d          subjet 1 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_1_n_d3           subjet 1 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_3_1_n_d5           subjet 1 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_3_1_z_d3           subjet 1 of 3 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_3_2_mass_d3        subjet 2 of 3 (k-means axes, by pT): mass (E) of the tracks with |d0/σ| > 3 [GeV]
  Q.sjf_3_2_max3d          subjet 2 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_2_maxsd0         subjet 2 of 3 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_3_3_mass_d3        subjet 3 of 3 (k-means axes, by pT): mass (E) of the tracks with |d0/σ| > 3 [GeV]
  Q.sjf_3_3_max3d          subjet 3 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_3_maxsd0         subjet 3 of 3 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_3_3_n_d3           subjet 3 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_3_3_z_d3           subjet 3 of 3 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_3_n2disp           number of the 3 subjets with at least 2 tracks with |d0/σ| > 3
  Q.sjf_4_1_max3d          subjet 1 of 4 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_4_1_n_d3           subjet 1 of 4 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_4_1_z_d3           subjet 1 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_2_mass_d3        subjet 2 of 4 (k-means axes, by pT): mass (E) of the tracks with |d0/σ| > 3 [GeV]
  Q.sjf_4_2_z_d3           subjet 2 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_3_n_d3           subjet 3 of 4 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_4_3_z_d3           subjet 3 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_4_maxsd0         subjet 4 of 4 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_4_4_z_d3           subjet 4 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
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
  Q.sjq_3_3_nch            number of charged particles in subjet 3 of 3
  Q.sjq_3_prod_k1          product of the charges (κ = 1) of the two hardest of 3 subjets
  Q.sjq_3_sumabs_k03       |sum| of the charges (κ = 0.3) of the two hardest of 3 subjets
  Q.sjq_3_sumabs_k1        |sum| of the charges (κ = 1) of the two hardest of 3 subjets
  Q.sum_e                  total energy of the particles [GeV]
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sv_1_dr                hardest displaced-track cluster: ΔR of its centre from the jet axis (0 if none)
  Q.sv_1_n                 hardest displaced-track cluster: number of tracks (0 if none)
  Q.sv_1_sd0_sum           hardest displaced-track cluster: Σ d0/σ of its tracks (0 if none)
  Q.sv_2_dr                2nd displaced-track cluster: ΔR of its centre from the jet axis (0 if none)
  Q.sv_2_n                 2nd displaced-track cluster: number of tracks (0 if none)
  Q.sv_2_sd0_sum           2nd displaced-track cluster: Σ d0/σ of its tracks (0 if none)
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
  Q.tau43_b2               N-subjettiness τ4/τ3 with β = 2
  Q.tau5                   N-subjettiness τ5 (β=1)
  Q.tau54                  N-subjettiness τ5/τ4
  Q.td0_25                 tanh(d0 [mm]) of particle 25 (ParT input; empty slot: 0)
  Q.tdz_0                  tanh(dz [mm]) of particle 0 (ParT input; empty slot: 0)
  Q.tdz_1                  tanh(dz [mm]) of particle 1 (ParT input; empty slot: 0)
  Q.tdz_13                 tanh(dz [mm]) of particle 13 (ParT input; empty slot: 0)
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.z_charged              pT share of charged particles
  Q.z_charged_had          pT share of charged hadrons
  Q.z_displaced3           pT share of the charged particles with |d0|/σ > 3
  Q.z_displaced5           pT share of the charged particles with |d0|/σ > 5
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.z_dr_0p4_up            pT share of the particles with ΔR ≥ 0.4
  Q.z_electron             pT share of electrons
  Q.z_muon                 pT share of muons
  Q.z_neutral              pT share of neutral particles
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
        C2_b05=ecfb('e3', 0.5) / max(ecfb('e2', 0.5) ** 2, 1e-30),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3_b05=ecfb('e4', 0.5) * ecfb('e2', 0.5) / max(ecfb('e3', 0.5) ** 2, 1e-30),
        C3_b2=ecfb('e4', 2) * ecfb('e2', 2) / max(ecfb('e3', 2) ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b05=ecfb('e3', 0.5) / max(ecfb('e2', 0.5) ** 3, 1e-30),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        D3_b05=ecfb('e4', 0.5) * ecfb('e2', 0.5) ** 3 / max(ecfb('e3', 0.5) ** 3, 1e-30),
        D3_b2=ecfb('e4', 2) * ecfb('e2', 2) ** 3 / max(ecfb('e3', 2) ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M2_b05=ecfb('g31', 0.5) / max(ecfb('e2', 0.5), 1e-30),
        M2_b2=ecfb('g31', 2) / max(ecfb('e2', 2), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        M3_b05=ecfb('g41', 0.5) / max(ecfb('g31', 0.5), 1e-30),
        M3_b2=ecfb('g41', 2) / max(ecfb('g31', 2), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N2_b05=ecfb('g32', 0.5) / max(ecfb('e2', 0.5) ** 2, 1e-30),
        N2_b2=ecfb('g32', 2) / max(ecfb('e2', 2) ** 2, 1e-30),
        N3_b05=ecfb('g42', 0.5) / max(ecfb('g31', 0.5) ** 2, 1e-30),
        ak02_1_n_disp3=pq('ak02_1_n_disp3'),
        ak02_1_n_lep=pq('ak02_1_n_lep'),
        ak02_1_z=pq('ak02_1_z'),
        ak02_2_charge=pq('ak02_2_charge'),
        ak02_2_n_lep=pq('ak02_2_n_lep'),
        ak02_2_sd0_1=pq('ak02_2_sd0_1'),
        ak02_2_sd0_3=pq('ak02_2_sd0_3'),
        ak02_2_z=pq('ak02_2_z'),
        ak02_3_n_disp3=pq('ak02_3_n_disp3'),
        ak02_3_n_lep=pq('ak02_3_n_lep'),
        ak02_3_z=pq('ak02_3_z'),
        ak02_3_z_disp3=pq('ak02_3_z_disp3'),
        ak02_dr12=pq('ak02_dr12'),
        ak02_dr13=pq('ak02_dr13'),
        ak02_dr23=pq('ak02_dr23'),
        ak02_min12_jp=pq('ak02_min12_jp'),
        ak02_min12_n_disp3=pq('ak02_min12_n_disp3'),
        ak02_n=pq('ak02_n'),
        charge_52=pfeat(52, 'charge'),
        d0err_0=pfeat(0, 'd0err'),
        d0err_52=pfeat(52, 'd0err'),
        dc_1_n_disp3=pq('dc_1_n_disp3'),
        dc_1_n_lep=pq('dc_1_n_lep'),
        dc_1_sd0_1=pq('dc_1_sd0_1'),
        dc_1_z=pq('dc_1_z'),
        dc_1_z_disp3=pq('dc_1_z_disp3'),
        dc_2_jp=pq('dc_2_jp'),
        dc_2_mass_disp3=pq('dc_2_mass_disp3'),
        dc_2_n_disp3=pq('dc_2_n_disp3'),
        dc_2_n_lep=pq('dc_2_n_lep'),
        dc_2_z=pq('dc_2_z'),
        dc_3_charge=pq('dc_3_charge'),
        dc_3_n_lep=pq('dc_3_n_lep'),
        dc_3_z=pq('dc_3_z'),
        dc_4_mass_disp3=pq('dc_4_mass_disp3'),
        dc_4_z=pq('dc_4_z'),
        dc_mass_disp_2nd=pq('dc_mass_disp_2nd'),
        dc_n=pq('dc_n'),
        dc_ntag=pq('dc_ntag'),
        dc_pair_mass_min=pq('dc_pair_mass_min'),
        dc_split1_z=pq('dc_split1_z'),
        dc_split2_dr=pq('dc_split2_dr'),
        dc_split2_mass=pq('dc_split2_mass'),
        dc_tag_2nd=pq('dc_tag_2nd'),
        dr01=math.sqrt(dist2(0, 1)),
        dr02=math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        dr_1=pfeat(1, 'dr'),
        dr_10=pfeat(10, 'dr'),
        dr_15=pfeat(15, 'dr'),
        dr_19=pfeat(19, 'dr'),
        dr_28=pfeat(28, 'dr'),
        dr_29=pfeat(29, 'dr'),
        dr_3=pfeat(3, 'dr'),
        dr_30=pfeat(30, 'dr'),
        dr_50=pfeat(50, 'dr'),
        dr_52=pfeat(52, 'dr'),
        dr_77=pfeat(77, 'dr'),
        dr_8=pfeat(8, 'dr'),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        dzerr_0=pfeat(0, 'dzerr'),
        dzerr_37=pfeat(37, 'dzerr'),
        dzerr_60=pfeat(60, 'dzerr'),
        dzerr_8=pfeat(8, 'dzerr'),
        dzerr_9=pfeat(9, 'dzerr'),
        e2_b2=ecfb('e2', 2),
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        e3=ecf('e3'),
        e3_b05=ecfb('e3', 0.5),
        e3_b2=ecfb('e3', 2),
        e4=ecf('e4'),
        e4_b05=ecfb('e4', 0.5),
        e4_b2=ecfb('e4', 2),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        ecf_g31=ecfb('g31', 1),
        ecf_g41=ecfb('g41', 1),
        ecf_g42=ecfb('g42', 1),
        ecf_g43=ecfb('g43', 1),
        eta_1=pfeat(1, 'eta'),
        eta_11=pfeat(11, 'eta'),
        eta_18=pfeat(18, 'eta'),
        eta_28=pfeat(28, 'eta'),
        eta_29=pfeat(29, 'eta'),
        eta_4=pfeat(4, 'eta'),
        eta_60=pfeat(60, 'eta'),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        ischhad_0=pfeat(0, 'ischhad'),
        ischhad_4=pfeat(4, 'ischhad'),
        iselectron_1=pfeat(1, 'iselectron'),
        iselectron_10=pfeat(10, 'iselectron'),
        iselectron_12=pfeat(12, 'iselectron'),
        iselectron_29=pfeat(29, 'iselectron'),
        iselectron_3=pfeat(3, 'iselectron'),
        ismuon_14=pfeat(14, 'ismuon'),
        ismuon_3=pfeat(3, 'ismuon'),
        ismuon_4=pfeat(4, 'ismuon'),
        isnhad_29=pfeat(29, 'isnhad'),
        isnhad_34=pfeat(34, 'isnhad'),
        isnhad_38=pfeat(38, 'isnhad'),
        isnhad_8=pfeat(8, 'isnhad'),
        isphoton_19=pfeat(19, 'isphoton'),
        isphoton_32=pfeat(32, 'isphoton'),
        isphoton_53=pfeat(53, 'isphoton'),
        jd_3d_4=pq('jd_3d_4'),
        jd_3d_5=pq('jd_3d_5'),
        jd_3d_6=pq('jd_3d_6'),
        jd_mass_d3_over_z=pq('jd_mass_d3_over_z'),
        jd_n_d3_pt1=pq('jd_n_d3_pt1'),
        jd_sum_abs_sd0_top3=pq('jd_sum_abs_sd0_top3'),
        jd_sum_abs_sd0_top5=pq('jd_sum_abs_sd0_top5'),
        jet_abs_eta=abs(jet_eta),
        jet_charge=sum(charge[i] * z[i] for i in real),
        jet_charge_k03=sum(charge[i] * z[i] ** 0.3 for i in real),
        jet_charge_k05=sum(charge[i] * z[i] ** 0.5 for i in real),
        jet_e=jet_energy,
        kt2_1_n_disp3=pq('kt2_1_n_disp3'),
        kt2_1_sd0_1=pq('kt2_1_sd0_1'),
        kt2_1_sd0_3=pq('kt2_1_sd0_3'),
        kt2_1_z=pq('kt2_1_z'),
        kt2_2_charge=pq('kt2_2_charge'),
        kt2_2_n_lep=pq('kt2_2_n_lep'),
        kt2_2_z_disp3=pq('kt2_2_z_disp3'),
        kt2_dr12=pq('kt2_dr12'),
        kt2_min12_jp=pq('kt2_min12_jp'),
        kt2_min12_mass_disp3=pq('kt2_min12_mass_disp3'),
        kt2_min12_n_disp3=pq('kt2_min12_n_disp3'),
        kt2_min12_sd0_1=pq('kt2_min12_sd0_1'),
        ktd_ln_d34=pq('ktd_ln_d34'),
        lam1=lam1,
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
        lne_1=pfeat(1, 'lne'),
        lne_10=pfeat(10, 'lne'),
        lne_16=pfeat(16, 'lne'),
        lne_17=pfeat(17, 'lne'),
        lne_3=pfeat(3, 'lne'),
        lne_4=pfeat(4, 'lne'),
        lne_5=pfeat(5, 'lne'),
        lne_8=pfeat(8, 'lne'),
        lnerel_3=pfeat(3, 'lnerel'),
        lnerel_4=pfeat(4, 'lnerel'),
        lnerel_42=pfeat(42, 'lnerel'),
        lnerel_65=pfeat(65, 'lnerel'),
        lnpt_1=pfeat(1, 'lnpt'),
        lnpt_30=pfeat(30, 'lnpt'),
        lnpt_78=pfeat(78, 'lnpt'),
        lnptrel_2=pfeat(2, 'lnptrel'),
        lnptrel_25=pfeat(25, 'lnptrel'),
        lnptrel_40=pfeat(40, 'lnptrel'),
        lund1_lndelta=lund(1, 'lndelta'),
        lund1_lnz=lund(1, 'lnz'),
        lund2_lndelta=lund(2, 'lndelta'),
        lund2_lnkt=lund(2, 'lnkt'),
        lund3_lndelta=lund(3, 'lndelta'),
        lund3_lnkt=lund(3, 'lnkt'),
        lund3_lnz=lund(3, 'lnz'),
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
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        max_abs_d0=max([abs(d0[i]) for i in real if charge[i] != 0] or [0.0]),
        max_abs_dz=max([abs(dz[i]) for i in real if charge[i] != 0] or [0.0]),
        max_dr=max(dr[i] for i in real),
        mean_eta=sum(z[i] * eta[i] for i in P),
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        mres_pruned_mass=pq('mres_pruned_mass'),
        mres_sd_mass_b0z005=pq('mres_sd_mass_b0z005'),
        mres_sd_mass_b0z02=pq('mres_sd_mass_b0z02'),
        mres_sd_mass_b1z01=pq('mres_sd_mass_b1z01'),
        mres_sd_mass_b2z01=pq('mres_sd_mass_b2z01'),
        mres_sd_prong_mass1=pq('mres_sd_prong_mass1'),
        mres_sd_prong_mass2=pq('mres_sd_prong_mass2'),
        mres_sd_rg_b0z005=pq('mres_sd_rg_b0z005'),
        mres_sd_rg_b1z01=pq('mres_sd_rg_b1z01'),
        mres_sd_rg_b2z01=pq('mres_sd_rg_b2z01'),
        mres_sd_zg_b0z005=pq('mres_sd_zg_b0z005'),
        mres_sd_zg_b1z01=pq('mres_sd_zg_b1z01'),
        mres_sd_zg_b2z01=pq('mres_sd_zg_b2z01'),
        n_charged_had=sum(1 for i in real if ptype[i] == 1),
        n_charged_pt_above_1=sum(1 for i in real if charge[i] != 0 and pt[i] > 1),
        n_charged_pt_above_10=sum(1 for i in real if charge[i] != 0 and pt[i] > 10),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
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
        n_sd0_above_10=nsig('d0', 10),
        n_sd0_above_2=nsig('d0', 2),
        n_sd0_above_3=nsig('d0', 3),
        n_sd0_above_5=nsig('d0', 5),
        n_sdz_above_2=nsig('dz', 2),
        n_sdz_above_5=nsig('dz', 5),
        nca_kt_above_10=pq('nca_kt_above_10'),
        nca_kt_above_2=pq('nca_kt_above_2'),
        nca_sj4_pair2nd_over_mass=pq('nca_sj4_pair2nd_over_mass'),
        nca_sj4_pair_mass_2nd=pq('nca_sj4_pair_mass_2nd'),
        nca_sj4_pairmax_over_mass=pq('nca_sj4_pairmax_over_mass'),
        pair_max_lnkt=pairmax('lnkt'),
        pair_max_lnm2=pairmax('lnm2'),
        pair_mean_lndelta=pairsum('lndelta'),
        pair_mean_lnkt=pairsum('lnkt'),
        pair_mean_lnm2=pairsum('lnm2'),
        pair_mean_lnz=pairsum('lnz'),
        phi_1=pfeat(1, 'phi'),
        phi_3=pfeat(3, 'phi'),
        phi_4=pfeat(4, 'phi'),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pz_lnd0=pq('pz_lnd0'),
        pz_lnd1=pq('pz_lnd1'),
        pz_lnd2=pq('pz_lnd2'),
        pz_lnd3=pq('pz_lnd3'),
        pz_lnkt0=pq('pz_lnkt0'),
        pz_lnkt1=pq('pz_lnkt1'),
        pz_lnkt2=pq('pz_lnkt2'),
        pz_lnkt3=pq('pz_lnkt3'),
        pz_lnkt4=pq('pz_lnkt4'),
        sd_mass=softdrop("mass"),
        sd_nremoved=softdrop("removed"),
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
        sj2_zsoft=subjets(2)["z"][1],
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr_min=min(subjets(3)["dr"]),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmax_over_m=max(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj4_dr_min=min(subjets(4)["dr"]),
        sj4_pair_mass_max=max(subjets(4)["mpair"]),
        sj4_pair_mass_min=min(subjets(4)["mpair"]),
        sjf_2_1_mass_d3=pq('sjf_2_1_mass_d3'),
        sjf_2_1_max3d=pq('sjf_2_1_max3d'),
        sjf_2_1_maxsd0=pq('sjf_2_1_maxsd0'),
        sjf_2_1_n_d3=pq('sjf_2_1_n_d3'),
        sjf_2_1_n_d5=pq('sjf_2_1_n_d5'),
        sjf_2_1_z_d3=pq('sjf_2_1_z_d3'),
        sjf_2_2_max3d=pq('sjf_2_2_max3d'),
        sjf_2_2_maxsd0=pq('sjf_2_2_maxsd0'),
        sjf_2_2_n_d3=pq('sjf_2_2_n_d3'),
        sjf_2_2_n_d5=pq('sjf_2_2_n_d5'),
        sjf_2_2_z_d3=pq('sjf_2_2_z_d3'),
        sjf_3_1_max3d=pq('sjf_3_1_max3d'),
        sjf_3_1_n_d3=pq('sjf_3_1_n_d3'),
        sjf_3_1_n_d5=pq('sjf_3_1_n_d5'),
        sjf_3_1_z_d3=pq('sjf_3_1_z_d3'),
        sjf_3_2_mass_d3=pq('sjf_3_2_mass_d3'),
        sjf_3_2_max3d=pq('sjf_3_2_max3d'),
        sjf_3_2_maxsd0=pq('sjf_3_2_maxsd0'),
        sjf_3_3_mass_d3=pq('sjf_3_3_mass_d3'),
        sjf_3_3_max3d=pq('sjf_3_3_max3d'),
        sjf_3_3_maxsd0=pq('sjf_3_3_maxsd0'),
        sjf_3_3_n_d3=pq('sjf_3_3_n_d3'),
        sjf_3_3_z_d3=pq('sjf_3_3_z_d3'),
        sjf_3_n2disp=pq('sjf_3_n2disp'),
        sjf_4_1_max3d=pq('sjf_4_1_max3d'),
        sjf_4_1_n_d3=pq('sjf_4_1_n_d3'),
        sjf_4_1_z_d3=pq('sjf_4_1_z_d3'),
        sjf_4_2_mass_d3=pq('sjf_4_2_mass_d3'),
        sjf_4_2_z_d3=pq('sjf_4_2_z_d3'),
        sjf_4_3_n_d3=pq('sjf_4_3_n_d3'),
        sjf_4_3_z_d3=pq('sjf_4_3_z_d3'),
        sjf_4_4_maxsd0=pq('sjf_4_4_maxsd0'),
        sjf_4_4_z_d3=pq('sjf_4_4_z_d3'),
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
        sjq_3_3_nch=pq('sjq_3_3_nch'),
        sjq_3_prod_k1=pq('sjq_3_prod_k1'),
        sjq_3_sumabs_k03=pq('sjq_3_sumabs_k03'),
        sjq_3_sumabs_k1=pq('sjq_3_sumabs_k1'),
        sum_e=sum(energy[i] for i in real),
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sv_1_dr=pq('sv_1_dr'),
        sv_1_n=pq('sv_1_n'),
        sv_1_sd0_sum=pq('sv_1_sd0_sum'),
        sv_2_dr=pq('sv_2_dr'),
        sv_2_n=pq('sv_2_n'),
        sv_2_sd0_sum=pq('sv_2_sd0_sum'),
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
        tau43_b2=tau_n(4, 2) / max(tau_n(3, 2), 1e-12),
        tau5=tau_n(5),
        tau54=tau_n(5) / max(tau_n(4), 1e-12),
        td0_25=pfeat(25, 'td0'),
        tdz_0=pfeat(0, 'tdz'),
        tdz_1=pfeat(1, 'tdz'),
        tdz_13=pfeat(13, 'tdz'),
        lam1_plus_lam2=ta + tc,
        z_charged=sum(z[i] for i in real if charge[i] != 0),
        z_charged_had=sum(z[i] for i in real if ptype[i] == 1),
        z_displaced3=displaced(3, 'z'),
        z_displaced5=displaced(5, 'z'),
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        z_dr_0p4_up=sum(z[i] for i in real if 0.4 <= dr[i] < 10),
        z_electron=sum(z[i] for i in real if ptype[i] == 4),
        z_muon=sum(z[i] for i in real if ptype[i] == 5),
        z_neutral=sum(z[i] for i in real if charge[i] == 0),
        z_neutral_had=sum(z[i] for i in real if ptype[i] == 2),
        z_photon=sum(z[i] for i in real if ptype[i] == 3),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top3_slots=sum(pt[:3]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
    )


def neuron_0(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.005196407
    )
    return z


def neuron_1(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.325267e-06
    )
    return z


def neuron_2(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.4473e-06
    )
    return z


def neuron_3(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.399837e-06
    )
    return z


def neuron_4(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.792139e-06
    )
    return z


def neuron_5(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.156926e-06
    )
    return z


def neuron_6(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.770106e-05
    )
    return z


def neuron_7(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.430826e-06
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
    z = 1.0 * (5.0993e-06
    )
    return z


def neuron_13(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.422913e-06
    )
    return z


def neuron_14(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.845304e-06
    )
    return z


def neuron_15(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.459526e-06
    )
    return z


def neuron_16(Q):
    # scale S = 15.03; each line: share * term / its average size
    z = 15.03195 * (-0.03041921
        - 0.1334216 * Q.z_charged_had / 0.5066138   # -13.3%  z_charged_had
        + 0.108631 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # +10.9%  lep_ptrel < 27.32
        - 0.09096887 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # -9.1%  n_s3d_above_3 < 10
        + 0.08044234 * max(0.0, 0.5187302 - Q.lep_z) / 0.4405854   # +8.0%  lep_z < 0.5187
        - 0.06196154 * max(0.0, 0.8615361 - Q.z_charged) / 0.2675383   # -6.2%  z_charged < 0.8615
        - 0.0512603 * max(0.0, 0.2801368 - Q.z_displaced5) / 0.1998239   # -5.1%  z_displaced5 < 0.2801
        + 0.05052095 * Q.lep_ptrel / 7.315413   # +5.1%  lep_ptrel
        + 0.04366474 * max(0.0, 0.2801368 - Q.z_displaced5) * max(0.0, 172.888 - Q.jd_3d_4) / 31.78805   # +4.4%  z_displaced5 < 0.2801 and jd_3d_4 < 172.9
        + 0.0204418 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # +2.0%  e3_b2 < 0.000791
        + 0.01883453 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # +1.9%  sip_3d_2 < 226.3
        + 0.01736008 * max(0.0, 0.09802954 - Q.sjq_2_prod_k1) / 0.1166532   # +1.7%  sjq_2_prod_k1 < 0.09803
        + 0.01728461 * max(0.0, 175.9957 - Q.sip_3d_3) / 142.4273   # +1.7%  sip_3d_3 < 176
        - 0.01682857 * max(0.0, 14.85973 - Q.jd_3d_4) / 8.639828   # -1.7%  jd_3d_4 < 14.86
        + 0.01418238 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # +1.4%  n_pairs_kt_above_3 < 34
        - 0.01409798 * max(0.0, 0.1373477 - Q.z_electron) / 0.1179076   # -1.4%  z_electron < 0.1373
        + 0.0124611 * max(0.0, Q.mass_top40 - 112.406) / 12.32657   # +1.2%  mass_top40 > 112.4
        - 0.01205453 * max(0.0, 1.777286 - Q.mass_displaced3) / 0.7377974   # -1.2%  mass_displaced3 < 1.777
        + 0.01175053 * Q.n_sdz_above_2 / 3.817083   # +1.2%  n_sdz_above_2
        + 0.009103658 * max(0.0, Q.sjq_2_sumabs_k1 - 0.2432922) / 0.1551258   # +0.9%  sjq_2_sumabs_k1 > 0.2433
        + 0.008824049 * max(0.0, 62.00562 - Q.mass_charged) / 9.422672   # +0.9%  mass_charged < 62.01
        + 0.008805028 * max(0.0, 16.0 - Q.sdb_2_n) / 6.019063   # +0.9%  sdb_2_n < 16
        - 0.008766484 * max(0.0, 0.03898651 - Q.z_muon) / 0.03219295   # -0.9%  z_muon < 0.03899
        - 0.00853716 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # -0.9%  n_pairs_kt_above_3 < 28
        + 0.008210916 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, 0.1270192 - Q.sdb_4_z) / 0.7401208   # +0.8%  n_s3d_above_3 < 10 and sdb_4_z < 0.127
        + 0.007906793 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # +0.8%  max_abs_d0 < 5.812
        - 0.00702858 * max(0.0, Q.mres_sd_mass_b0z005 - 159.9242) / 3.119054   # -0.7%  mres_sd_mass_b0z005 > 159.9
        + 0.006895927 * max(0.0, 2.957031 - Q.max_abs_dz) / 1.247099   # +0.7%  max_abs_dz < 2.957
        + 0.006829724 * max(0.0, 7.0 - Q.n_pairs_kt_above_10) / 2.958697   # +0.7%  n_pairs_kt_above_10 < 7
        - 0.006468285 * max(0.0, 18.0 - Q.n_photon) / 4.188877   # -0.6%  n_photon < 18
        + 0.006291955 * max(0.0, Q.mass_neutral - 34.02385) / 15.79354   # +0.6%  mass_neutral > 34.02
        - 0.006027933 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # -0.6%  e3_b2 < 0.0004127
        + 0.005780212 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.n_neutral - 10.0) / 62.20264   # +0.6%  n_s3d_above_3 < 10 and n_neutral > 10
        + 0.005725463 * max(0.0, Q.M3 - 0.02716047) / 0.01123924   # +0.6%  M3 > 0.02716
        - 0.005114389 * max(0.0, 0.2801368 - Q.z_displaced5) * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 0.229834   # -0.5%  z_displaced5 < 0.2801 and n_lund_kt_above_5 > 1
        + 0.004671899 * max(0.0, 46.61784 - Q.mres_sd_prong_mass1) / 22.17314   # +0.5%  mres_sd_prong_mass1 < 46.62
        - 0.004598199 * max(0.0, Q.lep_z - 0.221436) / 0.03510013   # -0.5%  lep_z > 0.2214
        - 0.004443291 * max(0.0, 0.156893 - Q.pz_lnd0) / 0.05001376   # -0.4%  pz_lnd0 < 0.1569
        - 0.004372103 * max(0.0, 0.09802954 - Q.sjq_2_prod_k1) * max(0.0, 1.0 - Q.ak02_2_n_lep) / 0.09023323   # -0.4%  sjq_2_prod_k1 < 0.09803 and ak02_2_n_lep < 1
        - 0.004361533 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.4%  lep_ptrel > 43.21
        - 0.004277512 * max(0.0, 9.408315 - Q.jd_sum_abs_sd0_top5) / 0.5127701   # -0.4%  jd_sum_abs_sd0_top5 < 9.408
        + 0.004089148 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.sv_1_n - 1.0) / 1.06826   # +0.4%  n_s3d_above_3 < 10 and sv_1_n > 1
        + 0.004014026 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +0.4%  n_pairs_kt_above_1 < 80
        + 0.003903283 * max(0.0, 0.08321691 - Q.pz_lnd2) / 0.01880965   # +0.4%  pz_lnd2 < 0.08322
        + 0.003694146 * max(0.0, 2.0 - Q.sjf_4_n2disp) / 1.231643   # +0.4%  sjf_4_n2disp < 2
        - 0.003594603 * max(0.0, 175.9957 - Q.sip_3d_3) * max(0.0, Q.mres_sd_prong_mass2 - 1.685874e-06) / 1399.378   # -0.4%  sip_3d_3 < 176 and mres_sd_prong_mass2 > 1.686e-06
        - 0.003217998 * max(0.0, 0.4436035 - Q.max_abs_dz) / 0.08080909   # -0.3%  max_abs_dz < 0.4436
        - 0.003182155 * max(0.0, Q.mass_neutral - 34.02385) * max(0.0, 2.912651 - Q.jd_3d_5) / 8.763977   # -0.3%  mass_neutral > 34.02 and jd_3d_5 < 2.913
        + 0.002998332 * max(0.0, 91.15481 - Q.mres_sd_mass_b2z01) / 9.5029   # +0.3%  mres_sd_mass_b2z01 < 91.15
        + 0.002939131 * max(0.0, Q.mres_sd_mass_b0z005 - 122.1) / 9.418671   # +0.3%  mres_sd_mass_b0z005 > 122.1
        - 0.002919294 * max(0.0, 0.0002703514 - Q.ecf_g41) / 8.44015e-05   # -0.3%  ecf_g41 < 0.0002704
        - 0.002905663 * max(0.0, 0.01841101 - Q.dr_min_012) / 0.005231118   # -0.3%  dr_min_012 < 0.01841
        - 0.002842586 * max(0.0, 0.8860453 - Q.D3_b05) / 0.4670665   # -0.3%  D3_b05 < 0.886
        - 0.002746736 * max(0.0, 0.09802954 - Q.sjq_2_prod_k1) * max(0.0, 0.200234 - Q.dc_1_z_disp3) / 0.01941435   # -0.3%  sjq_2_prod_k1 < 0.09803 and dc_1_z_disp3 < 0.2002
        + 0.002688201 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.n_lund - 5.0) / 0.001819184   # +0.3%  e3_b2 < 0.0004127 and n_lund > 5
        + 0.002339938 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # +0.2%  sip_3d_2 < 447.1
        - 0.002016106 * max(0.0, 7.54918 - Q.sj2_mass2) / 0.8421773   # -0.2%  sj2_mass2 < 7.549
        + 0.001974148 * max(0.0, 6.0 - Q.n_dr_0p1_0p2) / 0.8571967   # +0.2%  n_dr_0p1_0p2 < 6
        + 0.001841952 * max(0.0, 3.101398 - Q.sjf_3_2_max3d) / 0.6332004   # +0.2%  sjf_3_2_max3d < 3.101
        - 0.001782177 * max(0.0, Q.sjq_2_sumabs_k1 - 0.2432922) * max(0.0, 2.0 - Q.n_lepton) / 0.1700657   # -0.2%  sjq_2_sumabs_k1 > 0.2433 and n_lepton < 2
        + 0.001747007 * max(0.0, Q.mres_sd_mass_b0z005 - 178.8957) / 1.714218   # +0.2%  mres_sd_mass_b0z005 > 178.9
        - 0.001704282 * max(0.0, 0.01553665 - Q.z_dr_0p4_up) / 0.004779005   # -0.2%  z_dr_0p4_up < 0.01554
        + 0.001656749 * max(0.0, 3.0 - Q.sjq_2_2_nch) / 0.2391967   # +0.2%  sjq_2_2_nch < 3
        + 0.001630796 * max(0.0, Q.mres_sd_mass_b0z02 - 93.75785) / 14.10696   # +0.2%  mres_sd_mass_b0z02 > 93.76
        + 0.001615629 * max(0.0, Q.mres_sd_rg_b2z01 - 0.457968) / 0.03191439   # +0.2%  mres_sd_rg_b2z01 > 0.458
        - 0.001512502 * max(0.0, 16.0 - Q.sdb_2_n) * max(0.0, 105.4151 - Q.dc_pair_mass_min) / 387.233   # -0.2%  sdb_2_n < 16 and dc_pair_mass_min < 105.4
        + 0.00145511 * max(0.0, Q.ak02_dr23 - 0.3194837) / 0.08880139   # +0.1%  ak02_dr23 > 0.3195
        + 0.001453057 * max(0.0, 0.01556887 - Q.z_dr_0p05_0p1) / 0.002231501   # +0.1%  z_dr_0p05_0p1 < 0.01557
        - 0.001296316 * max(0.0, 16.0 - Q.sdb_2_n) * max(0.0, 0.8479222 - Q.tau43) / 0.4389785   # -0.1%  sdb_2_n < 16 and tau43 < 0.8479
        + 0.001266374 * max(0.0, Q.mass_2charged - 20.76537) / 3.761489   # +0.1%  mass_2charged > 20.77
        + 0.00117731 * max(0.0, Q.sjf_4_1_n_d3 - 2.0) / 0.5063833   # +0.1%  sjf_4_1_n_d3 > 2
        - 0.0008917846 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, Q.lepsj_2_n_d3 - 2.0) / 48.03862   # -0.1%  sip_3d_2 < 447.1 and lepsj_2_n_d3 > 2
        + 0.0008715967 * max(0.0, 0.01841101 - Q.dr_min_012) * max(0.0, 3.652671 - Q.sjf_4_1_max3d) / 0.004013044   # +0.1%  dr_min_012 < 0.01841 and sjf_4_1_max3d < 3.653
        + 0.0006653098 * max(0.0, 16.0 - Q.sdb_2_n) * max(0.0, Q.sjq_3_sumabs_k03 - 0.5318777) / 1.714196   # +0.1%  sdb_2_n < 16 and sjq_3_sumabs_k03 > 0.5319
        + 0.0006568884 * max(0.0, Q.sdb_0_n - 3.0) / 0.25801   # +0.1%  sdb_0_n > 3
        + 0.0006060346 * max(0.0, 1.141642 - Q.sjf_2_2_maxsd0) / 0.07431875   # +0.1%  sjf_2_2_maxsd0 < 1.142
        + 0.0005918086 * max(0.0, 3.577424 - Q.lne_4) / 0.1166942   # +0.1%  lne_4 < 3.577
        + 0.0005752856 * max(0.0, Q.mass_top5 - 62.84493) / 3.692784   # +0.1%  mass_top5 > 62.84
        - 0.0005570212 * max(0.0, Q.mass_2charged - 45.73288) / 0.6358753   # -0.1%  mass_2charged > 45.73
        - 0.0004582928 * max(0.0, 0.1506626 - Q.N2) / 0.001585014   # -0.0%  N2 < 0.1507
        + 0.0002794657 * max(0.0, Q.ak02_min12_n_disp3 - 1.0) / 0.05098   # +0.0%  ak02_min12_n_disp3 > 1
        - 0.0002500069 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) * max(0.0, 0.6513932 - Q.tau32) / 0.3152391   # -0.0%  n_pairs_kt_above_1 < 80 and tau32 < 0.6514
        - 0.0002350673 * max(0.0, Q.jd_3d_6 - 30.57088) / 4.999257   # -0.0%  jd_3d_6 > 30.57
        + 0.0002180164 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.mass_2photon - 22.18431) / 0.000102149   # +0.0%  e3_b2 < 0.0004127 and mass_2photon > 22.18
        + 0.0001985254 * max(0.0, -0.1170754 - Q.tdz_1) / 0.01260556   # +0.0%  tdz_1 < -0.1171
        + 0.000194932 * max(0.0, 0.8860453 - Q.D3_b05) * max(0.0, Q.n_muon - 0.0) / 0.09201753   # +0.0%  D3_b05 < 0.886 and n_muon > 0
        - 0.0001883689 * max(0.0, 1.00791 - Q.D2) / 0.03617881   # -0.0%  D2 < 1.008
        + 0.0001636579 * max(0.0, Q.M3 - 0.02716047) * max(0.0, 486.2148 - Q.jd_mass_d3_over_z) / 4.718954   # +0.0%  M3 > 0.02716 and jd_mass_d3_over_z < 486.2
        - 0.0001263895 * max(0.0, Q.sjf_2_1_n_d3 - 5.0) / 0.1972067   # -0.0%  sjf_2_1_n_d3 > 5
        + 0.0001253815 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, 1.0 - Q.kt2_2_n_lep) / 2.55124   # +0.0%  n_pairs_kt_above_3 < 28 and kt2_2_n_lep < 1
        + 0.0001212437 * max(0.0, 1.113424 - Q.sjf_2_1_maxsd0) / 0.01846321   # +0.0%  sjf_2_1_maxsd0 < 1.113
        + 0.0001196116 * max(0.0, Q.mass_2charged - 20.76537) * max(0.0, Q.M2_b2 - 0.01015545) / 0.06931777   # +0.0%  mass_2charged > 20.77 and M2_b2 > 0.01016
        + 0.0001097594 * max(0.0, 0.1530389 - Q.sdb_2_z) / 0.01282076   # +0.0%  sdb_2_z < 0.153
        + 9.885565e-05 * max(0.0, Q.sjq_2_sumabs_k1 - 0.2432922) * max(0.0, 1.0 - Q.ischhad_4) / 0.0606615   # +0.0%  sjq_2_sumabs_k1 > 0.2433 and ischhad_4 < 1
        + 8.97687e-05 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.sjf_4_3_z_d3 - 0.07303924) / 0.007363526   # +0.0%  n_s3d_above_3 < 10 and sjf_4_3_z_d3 > 0.07304
        - 6.298153e-05 * max(0.0, 1.141642 - Q.sjf_2_2_maxsd0) * max(0.0, Q.iselectron_1 - 0.0) / 0.003368357   # -0.0%  sjf_2_2_maxsd0 < 1.142 and iselectron_1 > 0
        - 5.04038e-05 * max(0.0, 0.302405 - Q.N2_b05) / 0.002473216   # -0.0%  N2_b05 < 0.3024
        - 3.658183e-05 * max(0.0, 46.61784 - Q.mres_sd_prong_mass1) * max(0.0, Q.dc_mass_disp_2nd - 0.0) / 0.699533   # -0.0%  mres_sd_prong_mass1 < 46.62 and dc_mass_disp_2nd > 0
        - 3.356227e-05 * max(0.0, Q.lep_ptrel - 43.20788) * max(0.0, 13.96555 - Q.kt2_1_sd0_1) / 4.664102   # -0.0%  lep_ptrel > 43.21 and kt2_1_sd0_1 < 13.97
        - 1.003044e-05 * max(0.0, Q.sjq_2_sumabs_k1 - 0.2432922) * max(0.0, Q.iselectron_29 - 0.0) / 0.0004300931   # -0.0%  sjq_2_sumabs_k1 > 0.2433 and iselectron_29 > 0
    )
    return z


def neuron_17(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.035347e-05
    )
    return z


def neuron_18(Q):
    # scale S = 11.98; each line: share * term / its average size
    z = 11.97548 * (0.1655977
        - 0.06228146 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -6.2%  lep_z < 0.2214
        + 0.05941264 * max(0.0, 3.0 - Q.lepsj_3_n_d3) / 2.624773   # +5.9%  lepsj_3_n_d3 < 3
        - 0.04103341 * max(0.0, 3.710567 - Q.sjf_3_1_max3d) / 0.7122498   # -4.1%  sjf_3_1_max3d < 3.711
        - 0.03082843 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # -3.1%  e3_b2 < 0.000791
        + 0.0306216 * max(0.0, 0.08090366 - Q.z_displaced5) / 0.04032225   # +3.1%  z_displaced5 < 0.0809
        - 0.03008425 * max(0.0, 46.74905 - Q.lepsj_3_maxsd0) / 40.371   # -3.0%  lepsj_3_maxsd0 < 46.75
        - 0.02821423 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.08090366 - Q.z_displaced5) / 0.00723244   # -2.8%  lep_z < 0.2214 and z_displaced5 < 0.0809
        - 0.02603237 * max(0.0, 0.004135872 - Q.lep_z) / 0.002390373   # -2.6%  lep_z < 0.004136
        - 0.02573819 * max(0.0, 586.9572 - Q.lepsj_3_maxsd0) / 546.6092   # -2.6%  lepsj_3_maxsd0 < 587
        - 0.02566625 * max(0.0, 29.0 - Q.n_for_90pct) / 10.0757   # -2.6%  n_for_90pct < 29
        + 0.02538216 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 1.761851   # +2.5%  lepsj_3_maxsd0 < 2.413
        - 0.02458523 * max(0.0, Q.mass - 100.4835) / 22.56274   # -2.5%  mass > 100.5
        - 0.02326176 * max(0.0, 4.516968 - Q.sjf_2_2_max3d) / 1.242753   # -2.3%  sjf_2_2_max3d < 4.517
        + 0.02268067 * max(0.0, 2.0 - Q.lepsj_3_n_d3) / 1.694207   # +2.3%  lepsj_3_n_d3 < 2
        + 0.02231784 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +2.2%  lep_iso < 1.362
        - 0.02215599 * max(0.0, 6.185635 - Q.lep_iso) / 4.883283   # -2.2%  lep_iso < 6.186
        + 0.02189121 * max(0.0, 55.92931 - Q.sjf_2_2_max3d) / 33.91681   # +2.2%  sjf_2_2_max3d < 55.93
        + 0.02172265 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.3829918 - Q.N2) / 0.01292108   # +2.2%  lep_z < 0.2214 and N2 < 0.383
        + 0.01938603 * max(0.0, 2.924357 - Q.sjf_3_1_max3d) / 0.3728193   # +1.9%  sjf_3_1_max3d < 2.924
        + 0.01865918 * max(0.0, 0.03663959 - Q.tau5) / 0.009671449   # +1.9%  tau5 < 0.03664
        - 0.01863472 * max(0.0, 3.0 - Q.lepsj_3_n_d3) * max(0.0, 0.08178299 - Q.lep_dr) / 0.1447323   # -1.9%  lepsj_3_n_d3 < 3 and lep_dr < 0.08178
        - 0.01669658 * max(0.0, Q.mres_sd_mass_b2z01 - 121.6067) / 9.197708   # -1.7%  mres_sd_mass_b2z01 > 121.6
        - 0.01588117 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # -1.6%  n_s3d_above_3 < 10
        - 0.01552644 * max(0.0, Q.n_real_top50 - 26.0) / 12.00248   # -1.6%  n_real_top50 > 26
        + 0.01514714 * max(0.0, Q.mres_sd_mass_b2z01 - 111.4443) / 13.01666   # +1.5%  mres_sd_mass_b2z01 > 111.4
        + 0.01408796 * max(0.0, 6.0 - Q.n_s3d_above_10) / 3.631173   # +1.4%  n_s3d_above_10 < 6
        - 0.01334862 * max(0.0, 0.0828821 - Q.sjf_2_1_z_d3) / 0.04834541   # -1.3%  sjf_2_1_z_d3 < 0.08288
        - 0.01322209 * max(0.0, 0.03624058 - Q.z_displaced3) / 0.01247855   # -1.3%  z_displaced3 < 0.03624
        + 0.01321041 * max(0.0, 71.08287 - Q.sjf_3_1_max3d) / 42.52076   # +1.3%  sjf_3_1_max3d < 71.08
        - 0.01274851 * max(0.0, 4.606241 - Q.sip_3d_3) / 1.172198   # -1.3%  sip_3d_3 < 4.606
        + 0.0111706 * max(0.0, 2.409613 - Q.mass_displaced3) / 1.070706   # +1.1%  mass_displaced3 < 2.41
        + 0.01096002 * max(0.0, 16.0 - Q.sjq_2_1_nch) / 5.009663   # +1.1%  sjq_2_1_nch < 16
        + 0.010085 * max(0.0, 146.3096 - Q.mass_top30) / 44.39811   # +1.0%  mass_top30 < 146.3
        + 0.010006 * max(0.0, 1.0 - Q.sjf_3_1_n_d3) / 0.4519233   # +1.0%  sjf_3_1_n_d3 < 1
        + 0.009761073 * max(0.0, Q.mass - 164.4374) / 3.038568   # +1.0%  mass > 164.4
        - 0.009506904 * max(0.0, Q.mass - 110.2019) / 16.73354   # -1.0%  mass > 110.2
        - 0.008937375 * max(0.0, 114.4658 - Q.mass_top20) / 26.77686   # -0.9%  mass_top20 < 114.5
        + 0.008694874 * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0001737301   # +0.9%  e3_b2 < 0.0002537
        - 0.008288622 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.sdb_2_z - 0.2509165) / 0.0155742   # -0.8%  lep_z < 0.2214 and sdb_2_z > 0.2509
        + 0.008282604 * max(0.0, 3.0 - Q.n_sdz_above_5) / 1.349857   # +0.8%  n_sdz_above_5 < 3
        - 0.008039137 * max(0.0, 126.8853 - Q.mass_top40) / 23.76162   # -0.8%  mass_top40 < 126.9
        - 0.007915453 * max(0.0, 16.0 - Q.sjq_2_1_nch) * max(0.0, 1651.766 - Q.sum_e) / 3918.801   # -0.8%  sjq_2_1_nch < 16 and sum_e < 1652
        - 0.007776321 * max(0.0, 0.05934469 - Q.sdb_0_z) / 0.03278889   # -0.8%  sdb_0_z < 0.05934
        + 0.007541552 * max(0.0, 51.56287 - Q.sv_1_sd0_sum) / 27.26317   # +0.8%  sv_1_sd0_sum < 51.56
        + 0.00749354 * max(0.0, 55.67805 - Q.sjf_2_1_max3d) / 27.50165   # +0.7%  sjf_2_1_max3d < 55.68
        + 0.007144416 * max(0.0, 4.0 - Q.jd_n_d3_pt1) / 1.678117   # +0.7%  jd_n_d3_pt1 < 4
        + 0.006815921 * max(0.0, 0.624781 - Q.psi_0p1) / 0.2784205   # +0.7%  psi_0p1 < 0.6248
        - 0.00641822 * max(0.0, 17.21875 - Q.max_abs_dz) / 11.28733   # -0.6%  max_abs_dz < 17.22
        - 0.006085897 * max(0.0, 0.849996 - Q.ak02_1_z) / 0.1554704   # -0.6%  ak02_1_z < 0.85
        - 0.006080174 * max(0.0, Q.lund3_lndelta - -2.4279) / 0.9642284   # -0.6%  lund3_lndelta > -2.428
        - 0.005725152 * max(0.0, 813.043 - Q.sdb_jp_top3) / 410.6254   # -0.6%  sdb_jp_top3 < 813
        - 0.005606694 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # -0.6%  n_s3d_above_3 < 3
        + 0.005602217 * max(0.0, 9.090532 - Q.mass_displaced3) / 5.595081   # +0.6%  mass_displaced3 < 9.091
        - 0.004804628 * max(0.0, 9.090532 - Q.mass_displaced3) * max(0.0, Q.sjq_2_2_nch - 4.0) / 17.22371   # -0.5%  mass_displaced3 < 9.091 and sjq_2_2_nch > 4
        - 0.004650234 * max(0.0, Q.n_real_top50 - 26.0) * max(0.0, 3.0 - Q.n_lund_kt_above_5) / 8.186317   # -0.5%  n_real_top50 > 26 and n_lund_kt_above_5 < 3
        + 0.004609659 * max(0.0, 0.5726046 - Q.sdb_2_z) / 0.2660293   # +0.5%  sdb_2_z < 0.5726
        - 0.004511189 * max(0.0, Q.z_displaced3 - 0.2092108) / 0.02926899   # -0.5%  z_displaced3 > 0.2092
        + 0.004386048 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +0.4%  n_dr_0p4_up < 4
        - 0.004133471 * max(0.0, 0.01299032 - Q.lam1) / 0.000791932   # -0.4%  lam1 < 0.01299
        - 0.003678149 * max(0.0, Q.mass - 182.8592) / 1.688248   # -0.4%  mass > 182.9
        + 0.003653506 * max(0.0, Q.M2 - 0.07013948) / 0.0151198   # +0.4%  M2 > 0.07014
        - 0.003521657 * max(0.0, Q.sj3_pair_mass_max - 128.6605) / 2.418855   # -0.4%  sj3_pair_mass_max > 128.7
        + 0.003405724 * max(0.0, 6.465318 - Q.sj2_mass2) / 0.5968622   # +0.3%  sj2_mass2 < 6.465
        + 0.003369508 * max(0.0, Q.mres_sd_mass_b2z01 - 177.5398) / 1.721011   # +0.3%  mres_sd_mass_b2z01 > 177.5
        + 0.003199224 * max(0.0, 0.3603262 - Q.z_charged_had) / 0.02270075   # +0.3%  z_charged_had < 0.3603
        - 0.002963968 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.jet_charge_k05 - -0.2841306) / 0.06110708   # -0.3%  lep_z < 0.2214 and jet_charge_k05 > -0.2841
        - 0.002956769 * max(0.0, 3.32421 - Q.sjf_2_1_max3d) / 0.386295   # -0.3%  sjf_2_1_max3d < 3.324
        + 0.002619735 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # +0.3%  sip_3d_2 < 226.3
        + 0.002348414 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.lne_4 - 3.751708) / 0.03644274   # +0.2%  lep_z < 0.2214 and lne_4 > 3.752
        - 0.002316302 * max(0.0, Q.ak02_dr23 - 0.3194837) / 0.08880139   # -0.2%  ak02_dr23 > 0.3195
        + 0.002307534 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.60135   # +0.2%  n_dr_0p1_0p2 < 17
        - 0.002219208 * max(0.0, Q.e3_b05 - 0.008870191) / 0.001172079   # -0.2%  e3_b05 > 0.00887
        + 0.001794859 * max(0.0, Q.sjf_4_1_z_d3 - 0.181327) / 0.01992241   # +0.2%  sjf_4_1_z_d3 > 0.1813
        - 0.001656376 * max(0.0, 0.2072106 - Q.sj3_pairmin_over_m) / 0.01530396   # -0.2%  sj3_pairmin_over_m < 0.2072
        + 0.001563821 * max(0.0, Q.lnptrel_2 - -2.51737) / 0.1593883   # +0.2%  lnptrel_2 > -2.517
        - 0.001510248 * max(0.0, 1.262841 - Q.mass_displaced5) / 0.5499328   # -0.2%  mass_displaced5 < 1.263
        + 0.001442835 * max(0.0, 3.0 - Q.sjq_3_2_nch) / 0.2817433   # +0.1%  sjq_3_2_nch < 3
        + 0.001405535 * max(0.0, 1.348343 - Q.sjf_3_3_max3d) / 0.2007926   # +0.1%  sjf_3_3_max3d < 1.348
        - 0.001144996 * max(0.0, 3.0 - Q.lepsj_3_n_d3) * max(0.0, Q.lepsj_3_dr - 0.0) / 0.03423189   # -0.1%  lepsj_3_n_d3 < 3 and lepsj_3_dr > 0
        + 0.001078742 * max(0.0, 2.078395 - Q.sjf_3_1_max3d) / 0.1080215   # +0.1%  sjf_3_1_max3d < 2.078
        - 0.001031459 * max(0.0, 0.01435877 - Q.sdb_4_z) / 0.01017083   # -0.1%  sdb_4_z < 0.01436
        - 0.001006984 * max(0.0, 51.56287 - Q.sv_1_sd0_sum) * max(0.0, Q.dc_1_n_lep - 0.0) / 6.332546   # -0.1%  sv_1_sd0_sum < 51.56 and dc_1_n_lep > 0
        - 0.0009381008 * max(0.0, Q.N2 - 0.4137858) / 0.003073306   # -0.1%  N2 > 0.4138
        + 0.0007284236 * max(0.0, Q.n_real_top50 - 26.0) * max(0.0, Q.d0err_0 - 0.0) / 0.07211032   # +0.1%  n_real_top50 > 26 and d0err_0 > 0
        + 0.0006781038 * max(0.0, Q.sv_2_n - 1.0) / 0.07863667   # +0.1%  sv_2_n > 1
        + 0.0006215228 * max(0.0, Q.mass_neutral - 59.66563) / 4.46544   # +0.1%  mass_neutral > 59.67
        - 0.0005611501 * max(0.0, 0.5726046 - Q.sdb_2_z) * max(0.0, 0.3249512 - Q.eta_28) / 0.08881337   # -0.1%  sdb_2_z < 0.5726 and eta_28 < 0.325
        - 0.0004741441 * max(0.0, 2.0 - Q.sjq_2_2_nch) / 0.09739333   # -0.0%  sjq_2_2_nch < 2
        + 0.0003756525 * max(0.0, 0.1032185 - Q.pz_lnd0) / 0.01871678   # +0.0%  pz_lnd0 < 0.1032
        - 0.0003481902 * max(0.0, Q.ak02_3_z - 0.07650476) / 0.0139231   # -0.0%  ak02_3_z > 0.0765
        - 0.0002914116 * max(0.0, 101.0 - Q.n_pairs_kt_above_1) / 8.83225   # -0.0%  n_pairs_kt_above_1 < 101
        - 0.0002531999 * max(0.0, 51.56287 - Q.sv_1_sd0_sum) * max(0.0, Q.sjf_3_3_z_d3 - 0.006421567) / 0.1438139   # -0.0%  sv_1_sd0_sum < 51.56 and sjf_3_3_z_d3 > 0.006422
        + 0.0002295894 * max(0.0, Q.lepsj_3_dr - 0.09790963) / 0.006471985   # +0.0%  lepsj_3_dr > 0.09791
        + 0.0002099663 * max(0.0, Q.mass_charged - 113.5019) / 1.042604   # +0.0%  mass_charged > 113.5
        - 0.0002091641 * max(0.0, Q.C2_b05 - 0.3223395) / 0.008478391   # -0.0%  C2_b05 > 0.3223
        + 0.00015787 * max(0.0, Q.nca_sj4_pair_mass_2nd - 72.72359) / 3.203001   # +0.0%  nca_sj4_pair_mass_2nd > 72.72
        - 0.0001459512 * Q.sjf_4_n2disp / 0.8042467   # -0.0%  sjf_4_n2disp
        - 8.837695e-05 * max(0.0, 0.08090366 - Q.z_displaced5) * max(0.0, 0.1912751 - Q.z_neutral_had) / 0.002894085   # -0.0%  z_displaced5 < 0.0809 and z_neutral_had < 0.1913
        - 1.439515e-06 * max(0.0, Q.mres_sd_mass_b0z02 - 17.38085) / 60.49884   # -0.0%  mres_sd_mass_b0z02 > 17.38
    )
    return z


def neuron_19(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.407819e-06
    )
    return z


def neuron_20(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.253685e-07
    )
    return z


def neuron_21(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.001681615
    )
    return z


def neuron_22(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.044848e-06
    )
    return z


def neuron_23(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.978821e-07
    )
    return z


def neuron_24(Q):
    # scale S = 15.8; each line: share * term / its average size
    z = 15.8025 * (0.02167006
        - 0.06692662 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # -6.7%  lep_z < 0.3397
        - 0.05534925 * max(0.0, 0.103005 - Q.lepsj_2_dr) * max(0.0, 14.38858 - Q.lep_iso) / 1.091303   # -5.5%  lepsj_2_dr < 0.103 and lep_iso < 14.39
        + 0.05158304 * max(0.0, 119.8459 - Q.mres_pruned_mass) / 32.90211   # +5.2%  mres_pruned_mass < 119.8
        - 0.04486659 * max(0.0, 86.14266 - Q.mres_pruned_mass) / 13.9802   # -4.5%  mres_pruned_mass < 86.14
        - 0.04419753 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -4.4%  mass < 164.4
        - 0.04237526 * max(0.0, 0.004135872 - Q.lep_z) * max(0.0, 2.907433 - Q.lep_iso) / 0.006825218   # -4.2%  lep_z < 0.004136 and lep_iso < 2.907
        + 0.04004508 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +4.0%  lep_iso < 0.4382
        + 0.03001833 * max(0.0, 118.2746 - Q.mres_sd_mass_b2z01) / 22.65141   # +3.0%  mres_sd_mass_b2z01 < 118.3
        + 0.02924852 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, 0.1822601 - Q.lepsj_2_dr) / 0.05233764   # +2.9%  lep_iso < 0.4382 and lepsj_2_dr < 0.1823
        + 0.0268695 * Q.pair_max_lnkt / 2.706794   # +2.7%  pair_max_lnkt
        - 0.02410551 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # -2.4%  e3_b2 < 0.0004127
        + 0.02369704 * max(0.0, 128.6605 - Q.sj3_pair_mass_max) / 38.7386   # +2.4%  sj3_pair_mass_max < 128.7
        + 0.02198411 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.8036986 - Q.lepsj_3_maxsd0) / 0.1622685   # +2.2%  lep_z < 0.3397 and lepsj_3_maxsd0 < 0.8037
        - 0.02151576 * max(0.0, 171.3819 - Q.mres_pruned_mass) / 78.34654   # -2.2%  mres_pruned_mass < 171.4
        + 0.02106261 * max(0.0, 17.0 - Q.n_dr_0p2_0p4) / 7.328103   # +2.1%  n_dr_0p2_0p4 < 17
        + 0.02096638 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +2.1%  lep_iso < 1.362
        - 0.02085263 * max(0.0, 81.19466 - Q.mres_sd_mass_b2z01) / 6.499995   # -2.1%  mres_sd_mass_b2z01 < 81.19
        - 0.01919353 * max(0.0, 18.80005 - Q.mass_displaced3) * max(0.0, 2325.187 - Q.sip_3d_2) / 28111.09   # -1.9%  mass_displaced3 < 18.8 and sip_3d_2 < 2325
        + 0.01861447 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 127.2317 - Q.mass) / 5.805704   # +1.9%  lep_z < 0.3397 and mass < 127.2
        + 0.01696509 * max(0.0, 18.80005 - Q.mass_displaced3) * max(0.0, 4.004982 - Q.jd_3d_5) / 24.96886   # +1.7%  mass_displaced3 < 18.8 and jd_3d_5 < 4.005
        + 0.01644483 * max(0.0, 18.80005 - Q.mass_displaced3) / 13.42744   # +1.6%  mass_displaced3 < 18.8
        + 0.01558335 * max(0.0, 110.2019 - Q.mass) / 12.0043   # +1.6%  mass < 110.2
        + 0.01487675 * max(0.0, 60.18017 - Q.mres_pruned_mass) / 6.5815   # +1.5%  mres_pruned_mass < 60.18
        - 0.01448975 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.tau21 - 0.1757731) / 0.07577322   # -1.4%  lep_z < 0.3397 and tau21 > 0.1758
        + 0.01436675 * max(0.0, Q.tau21 - 0.135772) / 0.3001429   # +1.4%  tau21 > 0.1358
        - 0.01412631 * max(0.0, 84.11398 - Q.jd_sum_abs_sd0_top5) * max(0.0, 6.771002 - Q.jd_3d_5) / 138.5038   # -1.4%  jd_sum_abs_sd0_top5 < 84.11 and jd_3d_5 < 6.771
        - 0.0136996 * max(0.0, 83.63398 - Q.sj3_pair_mass_max) * max(0.0, 0.4381892 - Q.lep_iso) / 2.327307   # -1.4%  sj3_pair_mass_max < 83.63 and lep_iso < 0.4382
        + 0.0125046 * max(0.0, 84.11398 - Q.jd_sum_abs_sd0_top5) / 28.0218   # +1.3%  jd_sum_abs_sd0_top5 < 84.11
        + 0.01078477 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # +1.1%  max_abs_d0 < 10.52
        - 0.01005912 * max(0.0, 0.1468781 - Q.z_dr_0p2_0p4) / 0.048751   # -1.0%  z_dr_0p2_0p4 < 0.1469
        + 0.009885498 * max(0.0, 83.63398 - Q.sj3_pair_mass_max) * max(0.0, 14.38858 - Q.lep_iso) / 98.03357   # +1.0%  sj3_pair_mass_max < 83.63 and lep_iso < 14.39
        + 0.009541341 * max(0.0, 134.2224 - Q.mass_top30) / 33.8221   # +1.0%  mass_top30 < 134.2
        - 0.009206684 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, 5.0 - Q.sdb_0_n) / 0.001102698   # -0.9%  e3_b2 < 0.0004127 and sdb_0_n < 5
        - 0.009074162 * max(0.0, 164.4374 - Q.mass) * max(0.0, 3.0 - Q.sv_1_n) / 110.1758   # -0.9%  mass < 164.4 and sv_1_n < 3
        - 0.008413907 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 7.0 - Q.n_neutral_had) / 0.7710798   # -0.8%  lep_z < 0.3397 and n_neutral_had < 7
        + 0.008262343 * max(0.0, 83.63398 - Q.sj3_pair_mass_max) * max(0.0, 0.1822601 - Q.lepsj_2_dr) / 1.279459   # +0.8%  sj3_pair_mass_max < 83.63 and lepsj_2_dr < 0.1823
        + 0.0081356 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 15.17086 - Q.D2_b2) / 0.5885   # +0.8%  z_displaced3 < 0.1062 and D2_b2 < 15.17
        - 0.007819224 * max(0.0, 117.4867 - Q.mass) / 15.64803   # -0.8%  mass < 117.5
        - 0.007424623 * max(0.0, 83.63398 - Q.sj3_pair_mass_max) / 7.654671   # -0.7%  sj3_pair_mass_max < 83.63
        + 0.007285763 * max(0.0, 0.212136 - Q.z_neutral_had) / 0.08633882   # +0.7%  z_neutral_had < 0.2121
        - 0.006683989 * max(0.0, 0.004135872 - Q.lep_z) / 0.002390373   # -0.7%  lep_z < 0.004136
        - 0.006440329 * max(0.0, 0.1061578 - Q.z_displaced3) / 0.05189409   # -0.6%  z_displaced3 < 0.1062
        - 0.006184798 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # -0.6%  lep_ptrel < 43.21
        + 0.005576033 * max(0.0, 55.13212 - Q.mres_sd_mass_b2z01) / 2.457281   # +0.6%  mres_sd_mass_b2z01 < 55.13
        + 0.005359644 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +0.5%  n_dr_0p4_up < 4
        + 0.005114516 * max(0.0, 0.004135872 - Q.lep_z) * max(0.0, 0.01435877 - Q.sdb_4_z) / 2.61867e-05   # +0.5%  lep_z < 0.004136 and sdb_4_z < 0.01436
        + 0.005096155 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.lne_8 - 2.11682) / 0.0003149447   # +0.5%  e3_b2 < 0.0004127 and lne_8 > 2.117
        - 0.005092889 * max(0.0, 0.103005 - Q.lepsj_2_dr) / 0.08254947   # -0.5%  lepsj_2_dr < 0.103
        - 0.004582955 * max(0.0, Q.mass_top15 - 70.93762) / 17.7702   # -0.5%  mass_top15 > 70.94
        - 0.004468643 * max(0.0, 7.452974e-05 - Q.e3_b2) / 3.411896e-05   # -0.4%  e3_b2 < 7.453e-05
        + 0.004389062 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.456416 - Q.tau32) / 0.003646089   # +0.4%  lep_z < 0.3397 and tau32 < 0.4564
        + 0.004335639 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.sdb_2_z - 0.2968888) / 2.687353   # +0.4%  lep_ptrel < 43.21 and sdb_2_z > 0.2969
        + 0.004334293 * max(0.0, 6.161303 - Q.dc_2_jp) / 3.330118   # +0.4%  dc_2_jp < 6.161
        - 0.004215294 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.4%  mass < 90.09
        - 0.004128812 * max(0.0, 0.456416 - Q.tau32) / 0.0169303   # -0.4%  tau32 < 0.4564
        - 0.003844579 * max(0.0, 0.2423129 - Q.N2) / 0.01410631   # -0.4%  N2 < 0.2423
        - 0.003292139 * max(0.0, 164.4374 - Q.mass) * max(0.0, 0.0309457 - Q.M3_b2) / 0.822219   # -0.3%  mass < 164.4 and M3_b2 < 0.03095
        - 0.003093457 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 4.004982 - Q.jd_3d_5) / 10.57583   # -0.3%  max_abs_d0 < 10.52 and jd_3d_5 < 4.005
        - 0.002775656 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.3683248 - Q.sj3_dr12) / 0.02771715   # -0.3%  lep_z < 0.3397 and sj3_dr12 < 0.3683
        + 0.00275523 * max(0.0, Q.dc_split2_dr - 0.2293319) / 0.01625115   # +0.3%  dc_split2_dr > 0.2293
        - 0.002552863 * max(0.0, Q.n_s3d_above_10 - 1.0) / 1.904323   # -0.3%  n_s3d_above_10 > 1
        + 0.002408554 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # +0.2%  n_s3d_above_3 < 3
        - 0.002396913 * max(0.0, 74.51927 - Q.mass_top30) / 2.669461   # -0.2%  mass_top30 < 74.52
        + 0.002395063 * max(0.0, 18.80005 - Q.mass_displaced3) * max(0.0, 0.3090294 - Q.z_photon) / 1.208322   # +0.2%  mass_displaced3 < 18.8 and z_photon < 0.309
        + 0.002351479 * max(0.0, 164.4374 - Q.mass) * max(0.0, 24.0 - Q.n_dr_0p2_0p4) / 828.5246   # +0.2%  mass < 164.4 and n_dr_0p2_0p4 < 24
        - 0.002192317 * max(0.0, Q.z_charged_had - 0.5398733) / 0.0536258   # -0.2%  z_charged_had > 0.5399
        + 0.002109798 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # +0.2%  n_pairs_kt_above_1 < 366
        + 0.00210717 * max(0.0, 5.0 - Q.n_lund) / 0.1017067   # +0.2%  n_lund < 5
        - 0.001968967 * max(0.0, 12.15228 - Q.lep_ptrel) / 8.819929   # -0.2%  lep_ptrel < 12.15
        - 0.001955701 * max(0.0, 0.006142967 - Q.lam1) / 0.0001406198   # -0.2%  lam1 < 0.006143
        + 0.001880382 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 1.761851   # +0.2%  lepsj_3_maxsd0 < 2.413
        + 0.001704737 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, Q.sdb_4_n - 0.0) / 0.1221333   # +0.2%  lep_iso < 0.4382 and sdb_4_n > 0
        - 0.001667107 * max(0.0, Q.dc_split2_dr - 0.3591078) / 0.004116954   # -0.2%  dc_split2_dr > 0.3591
        - 0.001584256 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 0.1166862 - Q.pz_lnd2) / 0.001637201   # -0.2%  z_displaced3 < 0.1062 and pz_lnd2 < 0.1167
        + 0.00154707 * max(0.0, -0.6014774 - Q.sjq_3_2_k1) / 0.01396761   # +0.2%  sjq_3_2_k1 < -0.6015
        + 0.001512485 * max(0.0, Q.sjq_2_2_k1 - 0.277232) / 0.04421729   # +0.2%  sjq_2_2_k1 > 0.2772
        - 0.001282091 * max(0.0, Q.mass_top15 - 70.93762) * max(0.0, Q.sv_2_z - 0.00726873) / 0.09933311   # -0.1%  mass_top15 > 70.94 and sv_2_z > 0.007269
        + 0.001239935 * max(0.0, 7.028704 - Q.sip_3d_1) / 1.023735   # +0.1%  sip_3d_1 < 7.029
        - 0.001187253 * max(0.0, 0.004135872 - Q.lep_z) * max(0.0, Q.n_pairs_kt_above_10 - 4.0) / 0.0104063   # -0.1%  lep_z < 0.004136 and n_pairs_kt_above_10 > 4
        + 0.001168506 * max(0.0, 0.103005 - Q.lepsj_2_dr) * max(0.0, Q.jet_e - 926.0771) / 10.59847   # +0.1%  lepsj_2_dr < 0.103 and jet_e > 926.1
        + 0.001165967 * Q.sjq_2_prod_k1 / 0.05054899   # +0.1%  sjq_2_prod_k1
        + 0.001085626 * max(0.0, 2.953464 - Q.jd_3d_4) / 0.5545459   # +0.1%  jd_3d_4 < 2.953
        + 0.001012436 * max(0.0, 1.0 - Q.n_s3d_above_3) / 0.17383   # +0.1%  n_s3d_above_3 < 1
        - 0.0007491269 * max(0.0, 0.01057938 - Q.C2_b2) / 0.0004084311   # -0.1%  C2_b2 < 0.01058
        - 0.0006117614 * max(0.0, -0.6014774 - Q.sjq_3_2_k1) * max(0.0, 0.4381892 - Q.lep_iso) / 0.003672589   # -0.1%  sjq_3_2_k1 < -0.6015 and lep_iso < 0.4382
        + 0.0005986687 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.e4 - 4.06096e-07) / 3.568567e-07   # +0.1%  lep_z < 0.3397 and e4 > 4.061e-07
        - 0.0005907705 * max(0.0, Q.mass_top15 - 70.93762) * max(0.0, 0.1572038 - Q.sv_2_dr) / 1.813829   # -0.1%  mass_top15 > 70.94 and sv_2_dr < 0.1572
        + 0.0004934081 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, Q.mass_2photon - 3.013) / 0.1897934   # +0.0%  z_displaced3 < 0.1062 and mass_2photon > 3.013
        + 0.0003934295 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.kt2_min12_n_disp3 - 1.0) / 2.276104e-05   # +0.0%  e3_b2 < 0.0004127 and kt2_min12_n_disp3 > 1
        - 0.0003384745 * max(0.0, Q.dr_8 - 0.3729651) / 0.01308837   # -0.0%  dr_8 > 0.373
        - 0.0003286834 * max(0.0, Q.sjq_2_2_k1 - 0.277232) * max(0.0, 2.907433 - Q.lep_iso) / 0.1043629   # -0.0%  sjq_2_2_k1 > 0.2772 and lep_iso < 2.907
        - 0.0002823926 * max(0.0, Q.mass_top15 - 70.93762) * max(0.0, Q.dc_1_n_lep - 0.0) / 5.165161   # -0.0%  mass_top15 > 70.94 and dc_1_n_lep > 0
        + 0.0002735442 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) * max(0.0, Q.ak02_3_n_lep - 0.0) / 0.04367839   # +0.0%  lepsj_3_maxsd0 < 2.413 and ak02_3_n_lep > 0
        + 0.0002093982 * max(0.0, Q.n_sd0_above_10 - 6.0) / 0.15998   # +0.0%  n_sd0_above_10 > 6
        - 0.0001527212 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.ak02_3_n_disp3 - 1.0) / 1.116382   # -0.0%  lep_ptrel < 43.21 and ak02_3_n_disp3 > 1
        - 0.0001086304 * max(0.0, Q.mass_top15 - 70.93762) * max(0.0, Q.sjf_3_2_mass_d3 - 0.0) / 91.98028   # -0.0%  mass_top15 > 70.94 and sjf_3_2_mass_d3 > 0
        + 7.944885e-05 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.sjq_2_sumabs_k1 - 0.5557674) / 0.005741854   # +0.0%  lep_z < 0.3397 and sjq_2_sumabs_k1 > 0.5558
        + 7.320974e-05 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, 9.442313 - Q.mres_sd_prong_mass2) / 0.821252   # +0.0%  lep_iso < 0.4382 and mres_sd_prong_mass2 < 9.442
        - 3.967575e-05 * max(0.0, 0.456416 - Q.tau32) * max(0.0, Q.dr_10 - 0.2577145) / 0.001113074   # -0.0%  tau32 < 0.4564 and dr_10 > 0.2577
        - 1.402904e-05 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) * max(0.0, Q.dc_3_n_lep - 0.0) / 0.03701267   # -0.0%  lepsj_3_maxsd0 < 2.413 and dc_3_n_lep > 0
    )
    return z


def neuron_25(Q):
    # scale S = 3.578; each line: share * term / its average size
    z = 3.577733 * (0.0650941
        + 0.08666722 * max(0.0, Q.mres_sd_mass_b2z01 - 76.1529) / 35.12032   # +8.7%  mres_sd_mass_b2z01 > 76.15
        + 0.04626134 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # +4.6%  lep_z < 0.3397
        - 0.04610741 * max(0.0, Q.mres_sd_mass_b2z01 - 55.13212) / 53.2288   # -4.6%  mres_sd_mass_b2z01 > 55.13
        + 0.04356673 * max(0.0, 73.90849 - Q.mres_sd_mass_b0z02) / 22.54232   # +4.4%  mres_sd_mass_b0z02 < 73.91
        + 0.04145748 * max(0.0, 99.0 - Q.n_pairs_kt_above_3) / 39.22243   # +4.1%  n_pairs_kt_above_3 < 99
        - 0.03327657 * max(0.0, Q.z_neutral_had - 0.02425618) / 0.1284684   # -3.3%  z_neutral_had > 0.02426
        - 0.03288856 * max(0.0, 2.0 - Q.n_sd0_above_3) / 0.5539567   # -3.3%  n_sd0_above_3 < 2
        - 0.03104005 * max(0.0, 3.0 - Q.dc_1_n_disp3) / 2.37235   # -3.1%  dc_1_n_disp3 < 3
        - 0.03087396 * max(0.0, 23.0 - Q.n_pairs_kt_above_10) / 16.58492   # -3.1%  n_pairs_kt_above_10 < 23
        - 0.03069071 * max(0.0, Q.mres_sd_mass_b2z01 - 102.264) / 17.61325   # -3.1%  mres_sd_mass_b2z01 > 102.3
        - 0.029667 * max(0.0, 47.97531 - Q.mres_sd_mass_b0z02) / 12.33932   # -3.0%  mres_sd_mass_b0z02 < 47.98
        - 0.02948608 * Q.z_photon / 0.2524126   # -2.9%  z_photon
        - 0.02345139 * max(0.0, 110.2019 - Q.mass) / 12.0043   # -2.3%  mass < 110.2
        - 0.0226863 * max(0.0, Q.N2_b05 - 0.4265629) / 0.03557833   # -2.3%  N2_b05 > 0.4266
        - 0.02104083 * max(0.0, Q.z_charged_had - 0.265564) / 0.2499185   # -2.1%  z_charged_had > 0.2656
        - 0.02023184 * max(0.0, 131.3917 - Q.mass) / 24.784   # -2.0%  mass < 131.4
        - 0.02006401 * max(0.0, Q.mres_sd_mass_b2z01 - 91.15481) / 24.25173   # -2.0%  mres_sd_mass_b2z01 > 91.15
        + 0.01899403 * max(0.0, 90.08945 - Q.mass) / 4.972097   # +1.9%  mass < 90.09
        + 0.01825883 * max(0.0, 76.15079 - Q.mass_neutral) * max(0.0, 1.0 - Q.ak02_2_n_lep) / 25.43291   # +1.8%  mass_neutral < 76.15 and ak02_2_n_lep < 1
        - 0.01793605 * max(0.0, 0.8501496 - Q.sj3_pairmax_over_m) / 0.06258068   # -1.8%  sj3_pairmax_over_m < 0.8501
        + 0.01770047 * max(0.0, 0.03624058 - Q.z_displaced3) / 0.01247855   # +1.8%  z_displaced3 < 0.03624
        - 0.01702982 * max(0.0, 0.0417856 - Q.M2_b2) * max(0.0, 5.0 - Q.sjf_4_1_n_d3) / 0.05097173   # -1.7%  M2_b2 < 0.04179 and sjf_4_1_n_d3 < 5
        + 0.01586766 * max(0.0, 35.45098 - Q.sv_1_sd0_sum) / 17.17693   # +1.6%  sv_1_sd0_sum < 35.45
        - 0.01526856 * max(0.0, 79.47361 - Q.mass) / 2.857843   # -1.5%  mass < 79.47
        + 0.01501242 * max(0.0, Q.N2_b05 - 0.302405) / 0.1383194   # +1.5%  N2_b05 > 0.3024
        - 0.01330468 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.1014193 - Q.ak02_3_z) / 0.01904535   # -1.3%  lep_z < 0.3397 and ak02_3_z < 0.1014
        + 0.01273887 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # +1.3%  lep_ptrel < 27.32
        + 0.0123288 * max(0.0, 0.05414784 - Q.sum_z_dr2_top20) / 0.02427967   # +1.2%  sum_z_dr2_top20 < 0.05415
        + 0.01188038 * max(0.0, 2.0 - Q.n_s3d_above_3) * max(0.0, 0.006470637 - Q.lepsj_2_dr) / 0.002241326   # +1.2%  n_s3d_above_3 < 2 and lepsj_2_dr < 0.006471
        + 0.01141598 * max(0.0, 2.0 - Q.n_s3d_above_3) / 0.4717633   # +1.1%  n_s3d_above_3 < 2
        + 0.01056559 * max(0.0, Q.mres_sd_mass_b2z01 - 107.2089) / 15.02375   # +1.1%  mres_sd_mass_b2z01 > 107.2
        + 0.01041315 * max(0.0, 7.075642 - Q.pair_max_lnm2) / 0.6225895   # +1.0%  pair_max_lnm2 < 7.076
        - 0.01041288 * max(0.0, 2.0 - Q.n_sdz_above_5) / 0.7314567   # -1.0%  n_sdz_above_5 < 2
        - 0.00974505 * max(0.0, 0.0417856 - Q.M2_b2) / 0.01370017   # -1.0%  M2_b2 < 0.04179
        + 0.009714043 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +1.0%  mass < 95.15
        - 0.007503744 * max(0.0, 86.60355 - Q.mres_sd_mass_b0z02) / 28.89097   # -0.8%  mres_sd_mass_b0z02 < 86.6
        - 0.00731681 * max(0.0, Q.N2_b05 - 0.302405) * max(0.0, Q.lne_1 - 4.222261) / 0.05438067   # -0.7%  N2_b05 > 0.3024 and lne_1 > 4.222
        - 0.007224223 * max(0.0, 6.43167 - Q.jd_3d_4) / 2.68944   # -0.7%  jd_3d_4 < 6.432
        + 0.006185343 * max(0.0, 0.170055 - Q.sdb_0_z) / 0.1233616   # +0.6%  sdb_0_z < 0.1701
        - 0.006151176 * max(0.0, 69.01131 - Q.nca_sj4_pair_mass_2nd) / 16.06985   # -0.6%  nca_sj4_pair_mass_2nd < 69.01
        + 0.0060024 * max(0.0, 9.76395e-05 - Q.e4_b05) / 6.241811e-05   # +0.6%  e4_b05 < 9.764e-05
        + 0.005868188 * max(0.0, 5.543454 - Q.mres_sd_prong_mass2) / 0.9673864   # +0.6%  mres_sd_prong_mass2 < 5.543
        + 0.00565493 * max(0.0, 2.655772 - Q.sjf_2_2_max3d) / 0.3641563   # +0.6%  sjf_2_2_max3d < 2.656
        + 0.004857684 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.pz_lnkt4 - 0.0) / 0.003508002   # +0.5%  lep_z < 0.3397 and pz_lnkt4 > 0
        + 0.00476483 * max(0.0, Q.sj2_mass1 - 77.42768) / 2.022907   # +0.5%  sj2_mass1 > 77.43
        + 0.004711672 * max(0.0, 0.6552778 - Q.z_charged_had) / 0.1652864   # +0.5%  z_charged_had < 0.6553
        + 0.00460717 * max(0.0, 3.0 - Q.dc_1_n_disp3) * max(0.0, 0.4046458 - Q.dr_52) / 0.8420906   # +0.5%  dc_1_n_disp3 < 3 and dr_52 < 0.4046
        + 0.004538138 * max(0.0, 0.2100115 - Q.pz_lnd2) / 0.09382831   # +0.5%  pz_lnd2 < 0.21
        + 0.004502607 * max(0.0, 2.552762 - Q.jd_3d_5) / 0.4820055   # +0.5%  jd_3d_5 < 2.553
        - 0.004467613 * max(0.0, 9.0 - Q.n_sd0_above_2) / 5.032627   # -0.4%  n_sd0_above_2 < 9
        + 0.004405085 * max(0.0, 0.06871203 - Q.pz_lnd0) / 0.006611837   # +0.4%  pz_lnd0 < 0.06871
        - 0.004232552 * max(0.0, 0.0417856 - Q.M2_b2) * max(0.0, 2.0 - Q.dc_2_n_disp3) / 0.02229523   # -0.4%  M2_b2 < 0.04179 and dc_2_n_disp3 < 2
        - 0.003836507 * max(0.0, 11.07803 - Q.sj3_mass3) / 5.735451   # -0.4%  sj3_mass3 < 11.08
        + 0.00383582 * max(0.0, 0.6552778 - Q.z_charged_had) * max(0.0, Q.sj2_mass2 - 6.465318) / 1.684754   # +0.4%  z_charged_had < 0.6553 and sj2_mass2 > 6.465
        - 0.003668456 * max(0.0, 0.6275286 - Q.nca_sj4_pair2nd_over_mass) / 0.1317611   # -0.4%  nca_sj4_pair2nd_over_mass < 0.6275
        - 0.003184034 * max(0.0, 0.05058744 - Q.dr12) / 0.01588854   # -0.3%  dr12 < 0.05059
        + 0.003048601 * max(0.0, 76.91486 - Q.sj3_pair_mass_max) / 5.293372   # +0.3%  sj3_pair_mass_max < 76.91
        - 0.003046096 * max(0.0, 0.5068038 - Q.tau32) / 0.02583179   # -0.3%  tau32 < 0.5068
        - 0.002970782 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -0.3%  lep_z < 0.2214
        + 0.002942266 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.sj4_dr_min - 0.1169491) / 0.01025673   # +0.3%  lep_z < 0.3397 and sj4_dr_min > 0.1169
        + 0.002913069 * max(0.0, Q.mres_sd_mass_b2z01 - 130.0968) / 7.118079   # +0.3%  mres_sd_mass_b2z01 > 130.1
        + 0.002714048 * max(0.0, 0.1104943 - Q.N2_b2) / 0.01277198   # +0.3%  N2_b2 < 0.1105
        + 0.00254841 * max(0.0, 0.04303099 - Q.sum_z_dr2) / 0.01222829   # +0.3%  sum_z_dr2 < 0.04303
        + 0.002387957 * max(0.0, 76.15079 - Q.mass_neutral) / 32.75986   # +0.2%  mass_neutral < 76.15
        - 0.001915099 * max(0.0, Q.sj3_pair_mass_max - 114.7848) / 4.421747   # -0.2%  sj3_pair_mass_max > 114.8
        - 0.001907053 * max(0.0, 58.87188 - Q.mass_charged) / 7.933493   # -0.2%  mass_charged < 58.87
        + 0.001832374 * max(0.0, 2.0 - Q.n_sdz_above_5) * max(0.0, Q.ak02_2_sd0_3 - 0.0) / 0.1797089   # +0.2%  n_sdz_above_5 < 2 and ak02_2_sd0_3 > 0
        - 0.001785981 * max(0.0, 131.3917 - Q.mass) * max(0.0, -0.02569022 - Q.sjq_2_prod_k1) / 0.7478403   # -0.2%  mass < 131.4 and sjq_2_prod_k1 < -0.02569
        - 0.001758991 * max(0.0, Q.sj2_mass1 - 77.42768) * max(0.0, 0.3689526 - Q.sv_1_dr) / 0.3334605   # -0.2%  sj2_mass1 > 77.43 and sv_1_dr < 0.369
        + 0.001728113 * max(0.0, 3.0 - Q.dc_1_n_disp3) * max(0.0, Q.sdb_2_n - 14.0) / 2.509633   # +0.2%  dc_1_n_disp3 < 3 and sdb_2_n > 14
        - 0.001601505 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.2241366 - Q.pz_lnd2) / 0.02945891   # -0.2%  lep_z < 0.3397 and pz_lnd2 < 0.2241
        + 0.001529231 * max(0.0, 0.0417856 - Q.M2_b2) * max(0.0, Q.mass_displaced3 - 6.341631) / 0.08418037   # +0.2%  M2_b2 < 0.04179 and mass_displaced3 > 6.342
        + 0.001501933 * max(0.0, Q.n_neutral - 17.0) / 5.247987   # +0.2%  n_neutral > 17
        - 0.001423214 * max(0.0, 0.00878048 - Q.sum_z_dr2_top3) / 0.001780555   # -0.1%  sum_z_dr2_top3 < 0.00878
        - 0.001365912 * max(0.0, Q.mres_sd_rg_b0z005 - 0.6103335) / 0.007392822   # -0.1%  mres_sd_rg_b0z005 > 0.6103
        - 0.001334472 * max(0.0, 18.0 - Q.n_neutral) / 2.453357   # -0.1%  n_neutral < 18
        - 0.001061572 * max(0.0, 99.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.sj3_dr13 - 0.3807968) / 3.354488   # -0.1%  n_pairs_kt_above_3 < 99 and sj3_dr13 > 0.3808
        - 0.0009429112 * max(0.0, 0.8501496 - Q.sj3_pairmax_over_m) * max(0.0, Q.kt2_2_z_disp3 - 0.01515512) / 0.0006237107   # -0.1%  sj3_pairmax_over_m < 0.8501 and kt2_2_z_disp3 > 0.01516
        - 0.0009398128 * max(0.0, 7.075642 - Q.pair_max_lnm2) * max(0.0, Q.dc_3_charge - 0.1143891) / 0.03676761   # -0.1%  pair_max_lnm2 < 7.076 and dc_3_charge > 0.1144
        + 0.0008690746 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # +0.1%  sj2_mass2 < 1.852
        + 0.0008612975 * max(0.0, 0.0417856 - Q.M2_b2) * max(0.0, Q.mass_2photon - 9.82024) / 0.02398497   # +0.1%  M2_b2 < 0.04179 and mass_2photon > 9.82
        + 0.0008554472 * max(0.0, 55.75601 - Q.mass_top40) / 0.8096506   # +0.1%  mass_top40 < 55.76
        - 0.0007348308 * max(0.0, Q.mass_top40 - 173.1022) / 1.415364   # -0.1%  mass_top40 > 173.1
        - 0.0006984958 * max(0.0, 21.21062 - Q.sj2_mass1) / 1.735779   # -0.1%  sj2_mass1 < 21.21
        + 0.0006635143 * max(0.0, 4.482525 - Q.lund_max_lnkt) / 0.678685   # +0.1%  lund_max_lnkt < 4.483
        - 0.0006489293 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.sj3_pair_mass_max - 73.24742) / 6.564178   # -0.1%  lep_z < 0.3397 and sj3_pair_mass_max > 73.25
        + 0.0005974452 * max(0.0, 0.0255865 - Q.sum_z_dr2_top10) / 0.006952385   # +0.1%  sum_z_dr2_top10 < 0.02559
        - 0.0005848014 * max(0.0, Q.lnpt_30 - 1.274457) / 0.09409909   # -0.1%  lnpt_30 > 1.274
        - 0.0004751239 * max(0.0, 76.15079 - Q.mass_neutral) * max(0.0, 2.0 - Q.lepsj_2_n_d3) / 54.2555   # -0.0%  mass_neutral < 76.15 and lepsj_2_n_d3 < 2
        + 0.0004732163 * max(0.0, 76.15079 - Q.mass_neutral) * max(0.0, 126.0617 - Q.dc_1_sd0_1) / 3440.702   # +0.0%  mass_neutral < 76.15 and dc_1_sd0_1 < 126.1
        + 0.0004487961 * max(0.0, 16.47324 - Q.dc_1_sd0_1) / 10.71466   # +0.0%  dc_1_sd0_1 < 16.47
        - 0.0002740852 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.sv_n - 0.0) / 0.3191875   # -0.0%  lep_z < 0.3397 and sv_n > 0
        + 0.0002714435 * max(0.0, 0.2100115 - Q.pz_lnd2) * max(0.0, Q.eta_29 - 0.06262207) / 0.004231314   # +0.0%  pz_lnd2 < 0.21 and eta_29 > 0.06262
        + 0.0002227566 * max(0.0, Q.sj3_pair_mass_max - 114.7848) * max(0.0, Q.lepsj_3_n_d3 - 3.0) / 0.7571707   # +0.0%  sj3_pair_mass_max > 114.8 and lepsj_3_n_d3 > 3
        + 0.0002003873 * max(0.0, Q.mres_sd_rg_b0z005 - 0.6103335) * max(0.0, Q.dzerr_8 - 0.0) / 0.0002039414   # +0.0%  mres_sd_rg_b0z005 > 0.6103 and dzerr_8 > 0
        + 0.0001455177 * max(0.0, Q.mass_top40 - 173.1022) * max(0.0, 0.09209404 - Q.C2_b2) / 0.01507746   # +0.0%  mass_top40 > 173.1 and C2_b2 < 0.09209
        + 0.0001216031 * max(0.0, -0.6014774 - Q.sjq_3_2_k1) / 0.01396761   # +0.0%  sjq_3_2_k1 < -0.6015
        + 5.510547e-05 * max(0.0, 2.0 - Q.n_sdz_above_5) * max(0.0, Q.iselectron_10 - 0.0) / 0.003586667   # +0.0%  n_sdz_above_5 < 2 and iselectron_10 > 0
        - 3.500197e-05 * max(0.0, 0.8501496 - Q.sj3_pairmax_over_m) * max(0.0, Q.dc_4_mass_disp3 - 0.0) / 0.004465992   # -0.0%  sj3_pairmax_over_m < 0.8501 and dc_4_mass_disp3 > 0
    )
    return z


def neuron_26(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.23126e-06
    )
    return z


def neuron_27(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.308499e-06
    )
    return z


def neuron_28(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.61721e-06
    )
    return z


def neuron_29(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.183866e-06
    )
    return z


def neuron_30(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.603051e-06
    )
    return z


def neuron_31(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.509522e-06
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
    z = 1.0 * (1.388487e-05
    )
    return z


def neuron_35(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.743296e-06
    )
    return z


def neuron_36(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.831947e-06
    )
    return z


def neuron_37(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.357436e-06
    )
    return z


def neuron_38(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.411179e-06
    )
    return z


def neuron_39(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.274279e-06
    )
    return z


def neuron_40(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.775539e-07
    )
    return z


def neuron_41(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.814745e-08
    )
    return z


def neuron_42(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.779049e-06
    )
    return z


def neuron_43(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.407709e-06
    )
    return z


def neuron_44(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.324348e-06
    )
    return z


def neuron_45(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.849337e-06
    )
    return z


def neuron_46(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.368751e-07
    )
    return z


def neuron_47(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.492076e-07
    )
    return z


def neuron_48(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.791779e-06
    )
    return z


def neuron_49(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.243815e-06
    )
    return z


def neuron_50(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.175119e-06
    )
    return z


def neuron_51(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.012201e-06
    )
    return z


def neuron_52(Q):
    # scale S = 10.65; each line: share * term / its average size
    z = 10.64916 * (-0.08467921
        + 0.1314661 * max(0.0, 159.9242 - Q.mres_sd_mass_b0z005) / 57.19246   # +13.1%  mres_sd_mass_b0z005 < 159.9
        + 0.05270866 * max(0.0, 0.3014662 - Q.lep_dr) / 0.2238267   # +5.3%  lep_dr < 0.3015
        - 0.05192031 * max(0.0, 131.0776 - Q.mres_sd_mass_b0z005) / 32.45252   # -5.2%  mres_sd_mass_b0z005 < 131.1
        + 0.03982137 * max(0.0, 0.1342762 - Q.z_displaced3) / 0.07089169   # +4.0%  z_displaced3 < 0.1343
        - 0.03487509 * max(0.0, 0.103005 - Q.lepsj_2_dr) / 0.08254947   # -3.5%  lepsj_2_dr < 0.103
        - 0.03377381 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 30.57088 - Q.jd_3d_6) / 2.01762   # -3.4%  z_displaced3 < 0.1343 and jd_3d_6 < 30.57
        - 0.03143339 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -3.1%  n_lepton < 1
        - 0.03113528 * max(0.0, 12.15228 - Q.lep_ptrel) / 8.819929   # -3.1%  lep_ptrel < 12.15
        + 0.02971781 * max(0.0, 0.3565533 - Q.tau21_b2) / 0.1109794   # +3.0%  tau21_b2 < 0.3566
        + 0.02884112 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.n_charged_pt_above_1 - 11.0) / 2.519562   # +2.9%  lep_z < 0.3397 and n_charged_pt_above_1 > 11
        + 0.02713335 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 5.367501 - Q.jd_3d_6) / 0.2552407   # +2.7%  z_displaced3 < 0.1343 and jd_3d_6 < 5.368
        + 0.02643639 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # +2.6%  e3_b2 < 0.0004127
        - 0.02543016 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 577.991 - Q.sip_3d_3) / 38.5554   # -2.5%  z_displaced3 < 0.1343 and sip_3d_3 < 578
        - 0.02353775 * max(0.0, Q.lne_16 - 0.4468807) / 1.728342   # -2.4%  lne_16 > 0.4469
        + 0.0225999 * max(0.0, 0.2837384 - Q.ak02_2_z) / 0.0942207   # +2.3%  ak02_2_z < 0.2837
        + 0.0218052 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 1.761851   # +2.2%  lepsj_3_maxsd0 < 2.413
        + 0.02163369 * max(0.0, 1.0 - Q.dc_1_n_lep) / 0.78823   # +2.2%  dc_1_n_lep < 1
        - 0.02063531 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -2.1%  lep_z < 0.2214
        - 0.01897495 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 10.56503   # -1.9%  n_dr_0p2_0p4 < 21
        - 0.01782827 * max(0.0, 159.9242 - Q.mres_sd_mass_b0z005) * max(0.0, 0.3565533 - Q.tau21_b2) / 5.887371   # -1.8%  mres_sd_mass_b0z005 < 159.9 and tau21_b2 < 0.3566
        + 0.01517745 * max(0.0, Q.N2_b2 - 0.08291719) / 0.0935801   # +1.5%  N2_b2 > 0.08292
        - 0.01399342 * max(0.0, 125.2765 - Q.sv_1_sd0_sum) / 79.48195   # -1.4%  sv_1_sd0_sum < 125.3
        - 0.0139707 * max(0.0, Q.ak02_1_z - 0.6342743) / 0.116529   # -1.4%  ak02_1_z > 0.6343
        + 0.01236444 * max(0.0, Q.mres_sd_mass_b0z005 - 81.62059) / 31.49572   # +1.2%  mres_sd_mass_b0z005 > 81.62
        - 0.01226175 * max(0.0, Q.sdb_2_n - 9.0) / 2.899327   # -1.2%  sdb_2_n > 9
        - 0.01222143 * max(0.0, 10.0 - Q.n_dr_0p4_up) / 5.21002   # -1.2%  n_dr_0p4_up < 10
        + 0.01151638 * max(0.0, Q.mres_sd_mass_b0z005 - 81.62059) * max(0.0, 0.07650476 - Q.ak02_3_z) / 0.9745751   # +1.2%  mres_sd_mass_b0z005 > 81.62 and ak02_3_z < 0.0765
        + 0.01109797 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # +1.1%  lep_z < 0.3397
        - 0.01087207 * max(0.0, 0.3788785 - Q.z_neutral_had) * max(0.0, Q.pz_lnkt1 - 0.03633353) / 0.01470373   # -1.1%  z_neutral_had < 0.3789 and pz_lnkt1 > 0.03633
        - 0.01000851 * max(0.0, 21.7295 - Q.sj2_mass2) / 8.475837   # -1.0%  sj2_mass2 < 21.73
        - 0.009401649 * max(0.0, 84.11398 - Q.jd_sum_abs_sd0_top5) / 28.0218   # -0.9%  jd_sum_abs_sd0_top5 < 84.11
        + 0.008259433 * max(0.0, 115.614 - Q.mass_top50) / 15.03711   # +0.8%  mass_top50 < 115.6
        + 0.008014085 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # +0.8%  n_s3d_above_3 < 3
        + 0.007496289 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, Q.N2 - 0.1824346) / 0.009842026   # +0.7%  z_displaced3 < 0.1343 and N2 > 0.1824
        - 0.006423602 * max(0.0, Q.lund3_lndelta - -2.817283) / 1.325774   # -0.6%  lund3_lndelta > -2.817
        + 0.006403869 * max(0.0, 79.27954 - Q.mass_top50) / 2.845865   # +0.6%  mass_top50 < 79.28
        + 0.006328668 * max(0.0, Q.lund3_lndelta - -2.817283) * max(0.0, 0.3689526 - Q.sv_1_dr) / 0.3049417   # +0.6%  lund3_lndelta > -2.817 and sv_1_dr < 0.369
        - 0.006308688 * max(0.0, 0.3788785 - Q.z_neutral_had) / 0.2321329   # -0.6%  z_neutral_had < 0.3789
        - 0.006281456 * max(0.0, 159.9242 - Q.mres_sd_mass_b0z005) * max(0.0, Q.mass_displaced3 - 3.208089) / 243.7029   # -0.6%  mres_sd_mass_b0z005 < 159.9 and mass_displaced3 > 3.208
        + 0.006209 * max(0.0, 0.103005 - Q.lepsj_2_dr) * max(0.0, 0.2250047 - Q.C3_b05) / 0.009818842   # +0.6%  lepsj_2_dr < 0.103 and C3_b05 < 0.225
        - 0.005184929 * max(0.0, Q.mres_sd_mass_b0z005 - 81.62059) * max(0.0, 0.7544983 - Q.tau32) / 5.345875   # -0.5%  mres_sd_mass_b0z005 > 81.62 and tau32 < 0.7545
        - 0.004242473 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sjq_2_prod_k05 - -0.5057096) / 4.880743   # -0.4%  n_dr_0p2_0p4 < 21 and sjq_2_prod_k05 > -0.5057
        + 0.004083807 * max(0.0, 0.3788785 - Q.z_neutral_had) * max(0.0, Q.ecf_g41 - 6.216954e-05) / 3.09401e-05   # +0.4%  z_neutral_had < 0.3789 and ecf_g41 > 6.217e-05
        + 0.004045549 * max(0.0, Q.sdb_2_n - 9.0) * max(0.0, 2.01745 - Q.D2_b2) / 1.203096   # +0.4%  sdb_2_n > 9 and D2_b2 < 2.017
        - 0.004030005 * max(0.0, Q.mass_2charged - 14.28253) / 5.706501   # -0.4%  mass_2charged > 14.28
        - 0.003900604 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # -0.4%  lep_ptrel < 27.32
        + 0.003832845 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +0.4%  lep_iso < 1.362
        - 0.003803397 * max(0.0, 1.45443 - Q.D2_b05) / 0.06234826   # -0.4%  D2_b05 < 1.454
        + 0.003753428 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.1145956 - Q.sjq_3_prod_k1) / 1.507298   # +0.4%  n_dr_0p2_0p4 < 21 and sjq_3_prod_k1 < 0.1146
        - 0.003506519 * max(0.0, Q.jet_charge_k03 - -0.2409073) / 0.4302075   # -0.4%  jet_charge_k03 > -0.2409
        + 0.00343275 * max(0.0, 0.3740528 - Q.sj2_dr) / 0.05324013   # +0.3%  sj2_dr < 0.3741
        - 0.003374135 * max(0.0, 65.0404 - Q.mass_charged) / 11.01768   # -0.3%  mass_charged < 65.04
        + 0.003291365 * max(0.0, Q.lep_ptrel - 18.7678) / 2.829169   # +0.3%  lep_ptrel > 18.77
        + 0.003270869 * max(0.0, 7.0 - Q.n_lund) / 0.2630033   # +0.3%  n_lund < 7
        + 0.003265055 * max(0.0, 21.7295 - Q.sj2_mass2) * max(0.0, 0.7817308 - Q.nca_sj4_pairmax_over_mass) / 0.6842131   # +0.3%  sj2_mass2 < 21.73 and nca_sj4_pairmax_over_mass < 0.7817
        - 0.003089221 * max(0.0, 6.767937 - Q.mass_2charged) / 3.01163   # -0.3%  mass_2charged < 6.768
        + 0.003071004 * max(0.0, 0.3014662 - Q.lep_dr) * max(0.0, Q.sdb_2_z - 0.1530389) / 0.04087335   # +0.3%  lep_dr < 0.3015 and sdb_2_z > 0.153
        + 0.002919358 * max(0.0, 3.029824 - Q.sjf_2_2_max3d) / 0.5243565   # +0.3%  sjf_2_2_max3d < 3.03
        + 0.002309665 * max(0.0, -0.6337755 - Q.sjq_2_2_k1) / 0.01353013   # +0.2%  sjq_2_2_k1 < -0.6338
        + 0.002285741 * max(0.0, Q.mres_sd_mass_b0z005 - 81.62059) * max(0.0, 3.151933 - Q.jd_3d_6) / 23.93055   # +0.2%  mres_sd_mass_b0z005 > 81.62 and jd_3d_6 < 3.152
        - 0.002246162 * max(0.0, 131.0776 - Q.mres_sd_mass_b0z005) * max(0.0, 1.204663 - Q.D2_b2) / 4.256639   # -0.2%  mres_sd_mass_b0z005 < 131.1 and D2_b2 < 1.205
        - 0.002197164 * max(0.0, Q.mass - 182.8592) / 1.688248   # -0.2%  mass > 182.9
        + 0.002178943 * max(0.0, Q.mres_sd_mass_b0z005 - 81.62059) * max(0.0, 0.03385157 - Q.sjf_2_2_z_d3) / 0.6012181   # +0.2%  mres_sd_mass_b0z005 > 81.62 and sjf_2_2_z_d3 < 0.03385
        - 0.002147973 * max(0.0, 131.0776 - Q.mres_sd_mass_b0z005) * max(0.0, 23.30856 - Q.sip_3d_3) / 502.2844   # -0.2%  mres_sd_mass_b0z005 < 131.1 and sip_3d_3 < 23.31
        - 0.002111194 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.sj3_pair_mass_min - 48.40547) / 1.423551   # -0.2%  lep_z < 0.3397 and sj3_pair_mass_min > 48.41
        + 0.002101902 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sjq_3_sumabs_k1 - 0.3376898) / 1.666965   # +0.2%  n_dr_0p2_0p4 < 21 and sjq_3_sumabs_k1 > 0.3377
        + 0.002017062 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # +0.2%  n_pairs_kt_above_3 < 28
        + 0.001871249 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) * max(0.0, 0.3014662 - Q.lep_dr) / 0.4601307   # +0.2%  lepsj_3_maxsd0 < 2.413 and lep_dr < 0.3015
        + 0.00179529 * max(0.0, Q.sjq_2_2_k1 - 0.6477929) / 0.01300326   # +0.2%  sjq_2_2_k1 > 0.6478
        + 0.001694842 * max(0.0, Q.lund_max_lndelta - -0.615738) / 0.02343634   # +0.2%  lund_max_lndelta > -0.6157
        + 0.001582722 * max(0.0, 7.104488e-06 - Q.e3_b2) / 8.290971e-07   # +0.2%  e3_b2 < 7.104e-06
        + 0.001570926 * max(0.0, Q.sj3_pair_mass_min - 80.02563) / 0.702582   # +0.2%  sj3_pair_mass_min > 80.03
        - 0.001546556 * max(0.0, Q.sj3_mass1 - 36.36236) / 1.189826   # -0.2%  sj3_mass1 > 36.36
        + 0.001515813 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +0.2%  n_pairs_kt_above_1 < 80
        - 0.001393052 * max(0.0, 0.001184159 - Q.D3_b2) * max(0.0, 5.367501 - Q.jd_3d_6) / 0.0004088142   # -0.1%  D3_b2 < 0.001184 and jd_3d_6 < 5.368
        - 0.001385561 * max(0.0, 0.03986247 - Q.sjf_2_1_z_d3) / 0.02018817   # -0.1%  sjf_2_1_z_d3 < 0.03986
        + 0.001282908 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, Q.n_lund_kt_above_1 - 3.0) / 0.1680127   # +0.1%  z_displaced3 < 0.1343 and n_lund_kt_above_1 > 3
        - 0.001170804 * max(0.0, Q.N3_b05 - 0.7591346) / 0.1643393   # -0.1%  N3_b05 > 0.7591
        - 0.001167136 * max(0.0, Q.ak02_3_z - 0.1014193) / 0.00958741   # -0.1%  ak02_3_z > 0.1014
        + 0.001108979 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 5.460234 - Q.D2_b2) / 26.9768   # +0.1%  lep_ptrel < 12.15 and D2_b2 < 5.46
        + 0.0009766087 * max(0.0, 0.01143413 - Q.tau4) / 0.0001505084   # +0.1%  tau4 < 0.01143
        - 0.0009760052 * max(0.0, 1.421532 - Q.D2) / 0.1381636   # -0.1%  D2 < 1.422
        + 0.0009522777 * max(0.0, 0.1122946 - Q.pz_lnd0) / 0.02302722   # +0.1%  pz_lnd0 < 0.1123
        + 0.0008733814 * max(0.0, 65.0404 - Q.mass_charged) * max(0.0, 1.0 - Q.isphoton_19) / 6.791118   # +0.1%  mass_charged < 65.04 and isphoton_19 < 1
        - 0.0006888555 * max(0.0, 0.1891627 - Q.z_charged_had) / 0.003239614   # -0.1%  z_charged_had < 0.1892
        + 0.0006450722 * max(0.0, 0.103005 - Q.lepsj_2_dr) * max(0.0, Q.z_photon - 0.3318968) / 0.002233468   # +0.1%  lepsj_2_dr < 0.103 and z_photon > 0.3319
        - 0.0006218183 * max(0.0, 7.0 - Q.n_lund) * max(0.0, 2.591685 - Q.jd_3d_6) / 0.2956722   # -0.1%  n_lund < 7 and jd_3d_6 < 2.592
        - 0.0006141301 * max(0.0, Q.lep_dr - 0.4191372) / 0.007569263   # -0.1%  lep_dr > 0.4191
        + 0.000551339 * max(0.0, 70.88236 - Q.mass_top40) / 1.88455   # +0.1%  mass_top40 < 70.88
        - 0.0004765837 * max(0.0, 0.001184159 - Q.D3_b2) / 0.0001406867   # -0.0%  D3_b2 < 0.001184
        + 0.0003074441 * max(0.0, Q.sjq_3_3_k1 - 0.6535817) / 0.01230651   # +0.0%  sjq_3_3_k1 > 0.6536
        - 0.0002835639 * max(0.0, 159.9242 - Q.mres_sd_mass_b0z005) * max(0.0, 1.251841 - Q.mass_displaced3) / 34.94578   # -0.0%  mres_sd_mass_b0z005 < 159.9 and mass_displaced3 < 1.252
        + 0.0002480766 * max(0.0, Q.sj2_mass1 - 91.2852) / 1.019309   # +0.0%  sj2_mass1 > 91.29
        + 0.0002439008 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.jet_e - 926.0771) / 1297.905   # +0.0%  n_dr_0p2_0p4 < 21 and jet_e > 926.1
        + 0.0001555629 * max(0.0, Q.mres_sd_mass_b0z005 - 81.62059) * max(0.0, Q.pz_lnd1 - 0.05758556) / 0.5323798   # +0.0%  mres_sd_mass_b0z005 > 81.62 and pz_lnd1 > 0.05759
        + 9.693336e-05 * max(0.0, Q.mass_top15 - 133.1648) / 1.058612   # +0.0%  mass_top15 > 133.2
        + 6.58563e-05 * max(0.0, 0.3377343 - Q.tau32_b2) / 0.02407155   # +0.0%  tau32_b2 < 0.3377
        - 5.276893e-05 * max(0.0, Q.mass - 182.8592) * max(0.0, 1135.303 - Q.sum_e) / 355.9363   # -0.0%  mass > 182.9 and sum_e < 1135
        - 4.584958e-05 * max(0.0, 3.0 - Q.n_s3d_above_3) * max(0.0, Q.dzerr_0 - 0.02830505) / 0.004573864   # -0.0%  n_s3d_above_3 < 3 and dzerr_0 > 0.02831
        - 2.62187e-05 * max(0.0, 1.421532 - Q.D2) * max(0.0, Q.ak02_2_sd0_1 - 18.03212) / 16.59933   # -0.0%  D2 < 1.422 and ak02_2_sd0_1 > 18.03
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
    z = 1.0 * (-7.072322e-07
    )
    return z


def neuron_57(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.959645e-06
    )
    return z


def neuron_58(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.61715e-06
    )
    return z


def neuron_59(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.029333e-06
    )
    return z


def neuron_60(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.666764e-06
    )
    return z


def neuron_61(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.840563e-06
    )
    return z


def neuron_62(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.422978e-07
    )
    return z


def neuron_63(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.308511e-06
    )
    return z


def neuron_64(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.258627e-06
    )
    return z


def neuron_65(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.791052e-06
    )
    return z


def neuron_66(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.709777e-06
    )
    return z


def neuron_67(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.219509e-06
    )
    return z


def neuron_68(Q):
    # scale S = 13.82; each line: share * term / its average size
    z = 13.81904 * (0.3396869
        - 0.1741062 * Q.n_particles / 39.33801   # -17.4%  n_particles
        - 0.1311352 * max(0.0, 158.2019 - Q.mres_sd_mass_b2z01) / 55.4697   # -13.1%  mres_sd_mass_b2z01 < 158.2
        - 0.09096913 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -9.1%  mass < 164.4
        + 0.06357776 * max(0.0, 118.2746 - Q.mres_sd_mass_b2z01) / 22.65141   # +6.4%  mres_sd_mass_b2z01 < 118.3
        + 0.04056729 * Q.n_photon / 16.03902   # +4.1%  n_photon
        + 0.02976996 * max(0.0, 130.0968 - Q.mres_sd_mass_b2z01) / 31.31128   # +3.0%  mres_sd_mass_b2z01 < 130.1
        + 0.02931967 * max(0.0, 0.05613495 - Q.tau5) / 0.02517241   # +2.9%  tau5 < 0.05613
        - 0.02843418 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # -2.8%  e3_b2 < 0.0004127
        - 0.02704868 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) / 44.19386   # -2.7%  sj3_pair_mass_min < 80.03
        - 0.02668036 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # -2.7%  lep_iso < 0.4382
        + 0.02481743 * max(0.0, 110.2019 - Q.mass) / 12.0043   # +2.5%  mass < 110.2
        - 0.02456215 * max(0.0, 76.1529 - Q.mres_sd_mass_b2z01) / 5.369584   # -2.5%  mres_sd_mass_b2z01 < 76.15
        + 0.01713851 * max(0.0, 0.04607888 - Q.lep_dr) * max(0.0, 0.7828545 - Q.lepsj_2_maxsd0) / 0.02040788   # +1.7%  lep_dr < 0.04608 and lepsj_2_maxsd0 < 0.7829
        - 0.01495324 * max(0.0, Q.mres_pruned_mass - 86.14266) / 22.56548   # -1.5%  mres_pruned_mass > 86.14
        + 0.01373129 * max(0.0, 0.04607888 - Q.lep_dr) / 0.02662022   # +1.4%  lep_dr < 0.04608
        - 0.01306705 * max(0.0, 0.05966366 - Q.M2_b2) / 0.0281728   # -1.3%  M2_b2 < 0.05966
        - 0.01284097 * max(0.0, Q.mres_sd_mass_b0z02 - 86.60355) / 17.13177   # -1.3%  mres_sd_mass_b0z02 > 86.6
        + 0.01143171 * max(0.0, Q.mres_sd_mass_b0z02 - 47.97531) / 39.20836   # +1.1%  mres_sd_mass_b0z02 > 47.98
        - 0.01099317 * max(0.0, Q.sj3_pair_mass_max - 83.63398) / 16.36149   # -1.1%  sj3_pair_mass_max > 83.63
        + 0.01092472 * max(0.0, Q.z_charged_had - 0.265564) / 0.2499185   # +1.1%  z_charged_had > 0.2656
        - 0.01069069 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 0.7828545 - Q.lepsj_2_maxsd0) / 0.02902708   # -1.1%  z_displaced3 < 0.1062 and lepsj_2_maxsd0 < 0.7829
        - 0.009113427 * max(0.0, 4.004982 - Q.jd_3d_5) / 1.451954   # -0.9%  jd_3d_5 < 4.005
        + 0.00873102 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.lnptrel_25 - -6.691703) / 0.0003839776   # +0.9%  e3_b2 < 0.0004127 and lnptrel_25 > -6.692
        + 0.008705425 * max(0.0, 55.13212 - Q.mres_sd_mass_b2z01) / 2.457281   # +0.9%  mres_sd_mass_b2z01 < 55.13
        + 0.008020348 * max(0.0, Q.jet_abs_eta - 0.5325716) / 0.3171687   # +0.8%  jet_abs_eta > 0.5326
        + 0.007498167 * max(0.0, Q.lep_ptrel - 18.7678) / 2.829169   # +0.7%  lep_ptrel > 18.77
        + 0.007437321 * max(0.0, Q.mres_pruned_mass - 124.3145) / 6.78236   # +0.7%  mres_pruned_mass > 124.3
        - 0.00734222 * max(0.0, 1163.885 - Q.lepsj_2_maxsd0) / 1076.442   # -0.7%  lepsj_2_maxsd0 < 1164
        + 0.006644779 * max(0.0, 0.7981752 - Q.tau32) / 0.1480334   # +0.7%  tau32 < 0.7982
        - 0.006459244 * max(0.0, Q.lep_ptrel - 6.983043) / 5.140178   # -0.6%  lep_ptrel > 6.983
        + 0.00621265 * max(0.0, Q.ktd_ln_d34 - -9.359695) / 0.5550767   # +0.6%  ktd_ln_d34 > -9.36
        - 0.006154085 * max(0.0, 0.1339824 - Q.pz_lnd2) / 0.04287826   # -0.6%  pz_lnd2 < 0.134
        + 0.005808012 * max(0.0, 3.0 - Q.n_sdz_above_5) / 1.349857   # +0.6%  n_sdz_above_5 < 3
        - 0.005432064 * max(0.0, Q.sj4_pair_mass_max - 69.83554) * max(0.0, 0.2037349 - Q.dc_3_z) / 2.386338   # -0.5%  sj4_pair_mass_max > 69.84 and dc_3_z < 0.2037
        - 0.005040174 * max(0.0, 1.774856 - Q.D2_b2) / 0.4426441   # -0.5%  D2_b2 < 1.775
        + 0.004530038 * max(0.0, 0.03259227 - Q.M3) / 0.00464829   # +0.5%  M3 < 0.03259
        - 0.003983223 * max(0.0, Q.nca_sj4_pair_mass_2nd - 72.72359) / 3.203001   # -0.4%  nca_sj4_pair_mass_2nd > 72.72
        + 0.003667346 * max(0.0, Q.sj3_pair_mass_max - 128.6605) / 2.418855   # +0.4%  sj3_pair_mass_max > 128.7
        - 0.003330602 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.C2_b2 - 0.01891146) / 0.9633021   # -0.3%  mass < 117.5 and C2_b2 > 0.01891
        + 0.003330573 * max(0.0, Q.z_charged_had - 0.265564) * max(0.0, Q.pz_lnkt3 - 0.0) / 0.01381879   # +0.3%  z_charged_had > 0.2656 and pz_lnkt3 > 0
        - 0.003196831 * max(0.0, Q.sdb_2_n - 12.0) / 1.5393   # -0.3%  sdb_2_n > 12
        - 0.003186679 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, Q.n_lund - 10.0) / 0.08342847   # -0.3%  z_displaced3 < 0.1062 and n_lund > 10
        - 0.00310818 * max(0.0, 103.4976 - Q.mass_top10) / 38.48054   # -0.3%  mass_top10 < 103.5
        - 0.002966032 * max(0.0, Q.mres_sd_rg_b2z01 - 0.457968) / 0.03191439   # -0.3%  mres_sd_rg_b2z01 > 0.458
        - 0.00291395 * max(0.0, Q.sj4_pair_mass_max - 69.83554) / 16.93977   # -0.3%  sj4_pair_mass_max > 69.84
        - 0.002853911 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.3%  mass < 90.09
        - 0.002660344 * max(0.0, 3.38061 - Q.pair_mean_lnm2) / 0.9985192   # -0.3%  pair_mean_lnm2 < 3.381
        + 0.002630893 * max(0.0, Q.mres_sd_mass_b0z02 - 134.2023) / 3.925253   # +0.3%  mres_sd_mass_b0z02 > 134.2
        + 0.002522407 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) * max(0.0, Q.mass_displaced5 - 0.0) / 257.9521   # +0.3%  sj3_pair_mass_min < 80.03 and mass_displaced5 > 0
        - 0.00250268 * max(0.0, 56.73313 - Q.sj3_pair_mass_max) / 1.542034   # -0.3%  sj3_pair_mass_max < 56.73
        + 0.002374063 * max(0.0, Q.pz_lnd3 - 0.01024929) / 0.08508837   # +0.2%  pz_lnd3 > 0.01025
        - 0.002244689 * max(0.0, 3.0 - Q.sjq_2_2_nch) / 0.2391967   # -0.2%  sjq_2_2_nch < 3
        + 0.002084327 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.sj3_mass1 - 17.31003) / 0.001436151   # +0.2%  e3_b2 < 0.0004127 and sj3_mass1 > 17.31
        - 0.001940696 * max(0.0, Q.mass_top20 - 130.7093) / 2.260459   # -0.2%  mass_top20 > 130.7
        + 0.001892324 * max(0.0, 0.7981752 - Q.tau32) * max(0.0, 0.3190414 - Q.sj3_dr_min) / 0.01394611   # +0.2%  tau32 < 0.7982 and sj3_dr_min < 0.319
        - 0.001886031 * max(0.0, 0.1061578 - Q.z_displaced3) / 0.05189409   # -0.2%  z_displaced3 < 0.1062
        + 0.001805711 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.n_lund_kt_above_5 - 2.0) / 0.0001088025   # +0.2%  e3_b2 < 0.0004127 and n_lund_kt_above_5 > 2
        + 0.001785441 * max(0.0, 3.38061 - Q.pair_mean_lnm2) * max(0.0, Q.z_top30_slots - 0.8316085) / 0.1174666   # +0.2%  pair_mean_lnm2 < 3.381 and z_top30_slots > 0.8316
        + 0.001784384 * max(0.0, 0.005938474 - Q.sum_z_dr2_top2) / 0.001014729   # +0.2%  sum_z_dr2_top2 < 0.005938
        + 0.001783034 * max(0.0, 7.0 - Q.sdb_2_n) / 0.63474   # +0.2%  sdb_2_n < 7
        + 0.001738187 * max(0.0, 117.4867 - Q.mass) * max(0.0, 1.380731 - Q.D2_b2) / 2.980715   # +0.2%  mass < 117.5 and D2_b2 < 1.381
        + 0.001718383 * max(0.0, Q.sj4_pair_mass_max - 69.83554) * max(0.0, Q.nca_sj4_pair2nd_over_mass - 0.4488456) / 0.8081534   # +0.2%  sj4_pair_mass_max > 69.84 and nca_sj4_pair2nd_over_mass > 0.4488
        - 0.001600679 * max(0.0, 0.1339824 - Q.pz_lnd2) * max(0.0, Q.sj3_dr13 - 0.4691911) / 0.003699286   # -0.2%  pz_lnd2 < 0.134 and sj3_dr13 > 0.4692
        + 0.001533912 * max(0.0, Q.z_charged_had - 0.265564) * max(0.0, Q.z_photon - 0.09084052) / 0.03440287   # +0.2%  z_charged_had > 0.2656 and z_photon > 0.09084
        - 0.001529414 * max(0.0, Q.n_dr_0p2_0p4 - 7.0) / 5.937073   # -0.2%  n_dr_0p2_0p4 > 7
        + 0.001504344 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 0.02258554 - Q.dr_29) / 0.0003980039   # +0.2%  z_displaced3 < 0.1062 and dr_29 < 0.02259
        + 0.001478638 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.ak02_dr23 - 0.3194837) / 1.450982e-05   # +0.1%  e3_b2 < 0.0004127 and ak02_dr23 > 0.3195
        + 0.001316668 * max(0.0, Q.lep_ptrel - 6.983043) * max(0.0, 0.1682645 - Q.C2_b2) / 0.5328009   # +0.1%  lep_ptrel > 6.983 and C2_b2 < 0.1683
        - 0.001267749 * max(0.0, Q.sj4_pair_mass_max - 69.83554) * max(0.0, 0.3689526 - Q.sv_1_dr) / 3.155583   # -0.1%  sj4_pair_mass_max > 69.84 and sv_1_dr < 0.369
        - 0.001230213 * max(0.0, Q.ak02_dr12 - 0.3882673) / 0.04986031   # -0.1%  ak02_dr12 > 0.3883
        - 0.001080219 * max(0.0, Q.n_s3d_above_3 - 5.0) / 0.8834567   # -0.1%  n_s3d_above_3 > 5
        + 0.001054492 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +0.1%  n_pairs_kt_above_1 < 80
        - 0.0009087876 * max(0.0, 0.1530389 - Q.sdb_2_z) / 0.01282076   # -0.1%  sdb_2_z < 0.153
        - 0.0008889011 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.1%  lep_ptrel > 43.21
        - 0.0008667619 * max(0.0, Q.n_s3d_above_3 - 5.0) * max(0.0, 226.3008 - Q.sip_3d_2) / 73.04149   # -0.1%  n_s3d_above_3 > 5 and sip_3d_2 < 226.3
        - 0.0007762216 * max(0.0, 0.03259227 - Q.M3) * max(0.0, 4.004982 - Q.jd_3d_5) / 0.005123874   # -0.1%  M3 < 0.03259 and jd_3d_5 < 4.005
        + 0.0007530724 * max(0.0, 0.7981752 - Q.tau32) * max(0.0, Q.sjq_3_sumabs_k1 - 0.4372817) / 0.0178362   # +0.1%  tau32 < 0.7982 and sjq_3_sumabs_k1 > 0.4373
        - 0.0007349622 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.sjf_4_3_z_d3 - 0.0) / 0.5551258   # -0.1%  mass < 164.4 and sjf_4_3_z_d3 > 0
        + 0.0007286037 * max(0.0, 117.4867 - Q.mass) / 15.64803   # +0.1%  mass < 117.5
        - 0.0007103352 * max(0.0, Q.dc_split2_dr - 0.3591078) / 0.004116954   # -0.1%  dc_split2_dr > 0.3591
        - 0.0005898207 * max(0.0, 0.2403736 - Q.sj2_dr) / 0.01094304   # -0.1%  sj2_dr < 0.2404
        - 0.0005679329 * max(0.0, Q.mres_pruned_mass - 171.3819) / 1.692539   # -0.1%  mres_pruned_mass > 171.4
        - 0.0003883841 * max(0.0, 0.4436035 - Q.max_abs_dz) / 0.08080909   # -0.0%  max_abs_dz < 0.4436
        - 0.0003835438 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.n_lund_kt_above_1 - 3.0) / 27.29162   # -0.0%  mass < 117.5 and n_lund_kt_above_1 > 3
        + 0.0003701943 * max(0.0, Q.lep_dr - 0.114659) / 0.05013662   # +0.0%  lep_dr > 0.1147
        + 0.0003374879 * max(0.0, Q.sj3_pair_mass_max - 128.6605) * max(0.0, Q.jet_e - 551.9263) / 1033.674   # +0.0%  sj3_pair_mass_max > 128.7 and jet_e > 551.9
        + 0.0003015561 * max(0.0, Q.sj3_pair_mass_max - 128.6605) * max(0.0, Q.e4 - 7.184834e-06) / 1.995926e-05   # +0.0%  sj3_pair_mass_max > 128.7 and e4 > 7.185e-06
        + 0.0002680401 * max(0.0, Q.z_displaced3 - 0.1681173) / 0.03847366   # +0.0%  z_displaced3 > 0.1681
        + 0.0002629636 * max(0.0, Q.sjf_3_1_n_d3 - 6.0) / 0.06264333   # +0.0%  sjf_3_1_n_d3 > 6
        - 0.0001748916 * max(0.0, 1.774856 - Q.D2_b2) * max(0.0, Q.sjf_4_4_z_d3 - 0.0) / 0.00241727   # -0.0%  D2_b2 < 1.775 and sjf_4_4_z_d3 > 0
        - 0.0001229111 * max(0.0, 117.4867 - Q.mass) * max(0.0, 1.0 - Q.ak02_2_n_lep) / 13.59497   # -0.0%  mass < 117.5 and ak02_2_n_lep < 1
        - 0.0001053912 * max(0.0, Q.sjf_3_1_n_d3 - 6.0) * max(0.0, 0.3989771 - Q.dr_10) / 0.01424444   # -0.0%  sjf_3_1_n_d3 > 6 and dr_10 < 0.399
        + 9.858567e-05 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, Q.ak02_3_n_lep - 0.0) / 0.00217889   # +0.0%  z_displaced3 < 0.1062 and ak02_3_n_lep > 0
        + 9.144969e-05 * max(0.0, Q.n_s3d_above_3 - 5.0) * max(0.0, Q.mean_eta - -0.02869681) / 0.01467691   # +0.0%  n_s3d_above_3 > 5 and mean_eta > -0.0287
        - 7.100526e-05 * max(0.0, Q.sj3_pair_mass_max - 128.6605) * max(0.0, Q.sjq_3_sumabs_k03 - 0.8323072) / 0.4833799   # -0.0%  sj3_pair_mass_max > 128.7 and sjq_3_sumabs_k03 > 0.8323
        - 3.687804e-05 * max(0.0, Q.z_charged_had - 0.265564) * max(0.0, Q.isnhad_29 - 0.0) / 0.01543715   # -0.0%  z_charged_had > 0.2656 and isnhad_29 > 0
        - 3.520094e-05 * max(0.0, 0.05613495 - Q.tau5) * max(0.0, Q.dc_ntag - 0.0) / 0.007175136   # -0.0%  tau5 < 0.05613 and dc_ntag > 0
        + 3.414381e-05 * max(0.0, Q.ak02_dr12 - 0.3882673) * max(0.0, Q.dr_77 - 0.0) / 0.0004115578   # +0.0%  ak02_dr12 > 0.3883 and dr_77 > 0
        + 1.64003e-05 * max(0.0, Q.mres_sd_mass_b0z02 - 47.97531) * max(0.0, Q.ismuon_4 - 0.0) / 0.3782379   # +0.0%  mres_sd_mass_b0z02 > 47.98 and ismuon_4 > 0
    )
    return z


def neuron_69(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.926143e-06
    )
    return z


def neuron_70(Q):
    # scale S = 4.711; each line: share * term / its average size
    z = 4.711199 * (0.01062481
        + 0.08049114 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +8.0%  lep_iso < 0.4382
        - 0.06464278 * max(0.0, 0.2801368 - Q.z_displaced5) / 0.1998239   # -6.5%  z_displaced5 < 0.2801
        - 0.05550214 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # -5.6%  lep_iso < 1.362
        - 0.0398558 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -4.0%  n_lepton < 1
        + 0.0390611 * max(0.0, 0.02148541 - Q.lam2) / 0.01555082   # +3.9%  lam2 < 0.02149
        + 0.03357657 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # +3.4%  n_s3d_above_3 < 10
        + 0.03332974 * max(0.0, 16.0 - Q.n_photon) / 3.000387   # +3.3%  n_photon < 16
        - 0.02983325 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.7941848 - Q.ak02_dr23) / 0.1520744   # -3.0%  lep_z < 0.3397 and ak02_dr23 < 0.7942
        - 0.02870619 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, 586.9572 - Q.lepsj_3_maxsd0) / 3579.903   # -2.9%  n_s3d_above_3 < 10 and lepsj_3_maxsd0 < 587
        + 0.0260848 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 162.7874 - Q.mass_top30) / 16.52391   # +2.6%  lep_z < 0.3397 and mass_top30 < 162.8
        - 0.02512964 * max(0.0, Q.mres_sd_mass_b0z005 - 69.14897) / 41.5427   # -2.5%  mres_sd_mass_b0z005 > 69.15
        - 0.02474222 * max(0.0, 16.0 - Q.n_photon) * max(0.0, 4.004982 - Q.jd_3d_5) / 5.382743   # -2.5%  n_photon < 16 and jd_3d_5 < 4.005
        - 0.02466004 * max(0.0, Q.mass - 120.653) / 11.75087   # -2.5%  mass > 120.7
        + 0.02361002 * max(0.0, Q.mass - 149.0507) / 4.967491   # +2.4%  mass > 149.1
        - 0.02251704 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -2.3%  lep_z < 0.2214
        + 0.01935192 * max(0.0, 0.1122946 - Q.pz_lnd0) / 0.02302722   # +1.9%  pz_lnd0 < 0.1123
        - 0.01906995 * max(0.0, 49.25971 - Q.mass_neutral) / 11.9268   # -1.9%  mass_neutral < 49.26
        + 0.01805412 * max(0.0, Q.mres_sd_mass_b0z005 - 86.65765) / 27.85055   # +1.8%  mres_sd_mass_b0z005 > 86.66
        + 0.0179866 * max(0.0, Q.mres_sd_mass_b0z005 - 86.65765) * max(0.0, 4.279435 - Q.D2_b2) / 71.37394   # +1.8%  mres_sd_mass_b0z005 > 86.66 and D2_b2 < 4.279
        - 0.01788507 * max(0.0, 114.4658 - Q.mass_top20) / 26.77686   # -1.8%  mass_top20 < 114.5
        + 0.01684523 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.M2_b05 - 0.0815014) / 0.3548306   # +1.7%  n_s3d_above_3 < 10 and M2_b05 > 0.0815
        + 0.01582376 * max(0.0, 1.874473e-05 - Q.ecf_g42) / 5.221994e-06   # +1.6%  ecf_g42 < 1.874e-05
        - 0.01484438 * max(0.0, 0.06573337 - Q.lepsj_2_dr) / 0.04985349   # -1.5%  lepsj_2_dr < 0.06573
        + 0.01477292 * max(0.0, 2.0 - Q.lepsj_3_n_d3) * max(0.0, 0.114659 - Q.lep_dr) / 0.138271   # +1.5%  lepsj_3_n_d3 < 2 and lep_dr < 0.1147
        + 0.01449464 * max(0.0, 0.2801368 - Q.z_displaced5) * max(0.0, Q.n_lund_kt_above_1 - 2.0) / 0.663796   # +1.4%  z_displaced5 < 0.2801 and n_lund_kt_above_1 > 2
        + 0.01426827 * max(0.0, 1.529925 - Q.lepsj_3_maxsd0) / 1.073625   # +1.4%  lepsj_3_maxsd0 < 1.53
        - 0.01378154 * max(0.0, Q.mres_sd_mass_b0z005 - 86.65765) * max(0.0, 0.1807551 - Q.ak02_3_z) / 2.921705   # -1.4%  mres_sd_mass_b0z005 > 86.66 and ak02_3_z < 0.1808
        + 0.01361945 * max(0.0, 4.0 - Q.n_s3d_above_3) * max(0.0, Q.z_photon - 0.05998812) / 0.2993287   # +1.4%  n_s3d_above_3 < 4 and z_photon > 0.05999
        + 0.01348372 * max(0.0, 2.0 - Q.lepsj_3_n_d3) / 1.694207   # +1.3%  lepsj_3_n_d3 < 2
        + 0.0124446 * max(0.0, Q.N2 - 0.2875658) / 0.04585899   # +1.2%  N2 > 0.2876
        - 0.009988316 * max(0.0, Q.z_displaced5 - 0.08090366) / 0.05238512   # -1.0%  z_displaced5 > 0.0809
        - 0.00996444 * max(0.0, Q.n_s3d_above_3 - 1.0) / 2.933937   # -1.0%  n_s3d_above_3 > 1
        - 0.009494119 * max(0.0, Q.N2 - 0.2875658) * max(0.0, 172.888 - Q.jd_3d_4) / 6.769955   # -0.9%  N2 > 0.2876 and jd_3d_4 < 172.9
        - 0.008882892 * max(0.0, Q.sj3_pair_mass_min - 48.40547) / 4.844786   # -0.9%  sj3_pair_mass_min > 48.41
        + 0.008219783 * max(0.0, Q.sdb_2_z - 0.3448514) / 0.06273059   # +0.8%  sdb_2_z > 0.3449
        + 0.0079945 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # +0.8%  lep_z < 0.3397
        + 0.007988449 * max(0.0, Q.sjf_4_n2disp - 1.0) / 0.2405067   # +0.8%  sjf_4_n2disp > 1
        + 0.007947806 * max(0.0, 0.2385164 - Q.ak02_dr23) / 0.1230174   # +0.8%  ak02_dr23 < 0.2385
        - 0.007829975 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # -0.8%  sip_3d_2 < 447.1
        - 0.006687826 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, 6.43167 - Q.jd_3d_4) / 1024.924   # -0.7%  sip_3d_2 < 447.1 and jd_3d_4 < 6.432
        + 0.006424921 * max(0.0, Q.sj3_pair_mass_min - 48.40547) * max(0.0, 4.0 - Q.n_lund_kt_above_5) / 5.411714   # +0.6%  sj3_pair_mass_min > 48.41 and n_lund_kt_above_5 < 4
        + 0.006415796 * max(0.0, 4.0 - Q.n_s3d_above_3) / 1.46489   # +0.6%  n_s3d_above_3 < 4
        + 0.006143585 * max(0.0, 0.1828918 - Q.mres_sd_zg_b0z005) / 0.02909288   # +0.6%  mres_sd_zg_b0z005 < 0.1829
        - 0.005991971 * max(0.0, 0.2509165 - Q.sdb_2_z) / 0.04197991   # -0.6%  sdb_2_z < 0.2509
        - 0.005717565 * max(0.0, Q.mres_sd_mass_b0z005 - 159.9242) / 3.119054   # -0.6%  mres_sd_mass_b0z005 > 159.9
        - 0.005590426 * max(0.0, 0.1122946 - Q.pz_lnd0) * max(0.0, Q.z_neutral_had - 0.08823774) / 0.002226703   # -0.6%  pz_lnd0 < 0.1123 and z_neutral_had > 0.08824
        + 0.005511564 * max(0.0, Q.z_displaced5 - 0.08090366) * max(0.0, 1.0 - Q.dc_1_n_lep) / 0.03602952   # +0.6%  z_displaced5 > 0.0809 and dc_1_n_lep < 1
        + 0.005045208 * max(0.0, 22.56857 - Q.mass_charged) / 0.407248   # +0.5%  mass_charged < 22.57
        - 0.004939708 * max(0.0, 0.7947468 - Q.tau43) / 0.04387733   # -0.5%  tau43 < 0.7947
        + 0.004852139 * max(0.0, -0.6014774 - Q.sjq_3_2_k1) / 0.01396761   # +0.5%  sjq_3_2_k1 < -0.6015
        - 0.004815379 * max(0.0, 4.183454 - Q.sv_2_sd0_sum) / 2.861891   # -0.5%  sv_2_sd0_sum < 4.183
        - 0.004258939 * max(0.0, Q.n_charged_had - 21.0) / 2.12364   # -0.4%  n_charged_had > 21
        - 0.004070627 * max(0.0, 1.362094 - Q.lep_iso) * max(0.0, Q.sjq_3_3_k1 - 0.2939497) / 0.04264811   # -0.4%  lep_iso < 1.362 and sjq_3_3_k1 > 0.2939
        - 0.003640789 * max(0.0, 0.08996752 - Q.sj2_zsoft) / 0.003665091   # -0.4%  sj2_zsoft < 0.08997
        - 0.00332881 * max(0.0, 4.696862e-06 - Q.e3_b2) / 4.050596e-07   # -0.3%  e3_b2 < 4.697e-06
        - 0.00323194 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, -0.6014774 - Q.sjq_3_2_k1) / 0.003672589   # -0.3%  lep_iso < 0.4382 and sjq_3_2_k1 < -0.6015
        - 0.003231271 * max(0.0, 1.529925 - Q.lepsj_3_maxsd0) * max(0.0, Q.mass_top3 - 16.326) / 14.88457   # -0.3%  lepsj_3_maxsd0 < 1.53 and mass_top3 > 16.33
        - 0.003187563 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # -0.3%  lep_ptrel > 27.32
        - 0.002894663 * max(0.0, Q.lep_dr - 0.08178299) / 0.06076808   # -0.3%  lep_dr > 0.08178
        + 0.002822955 * max(0.0, Q.sjq_3_3_k1 - 0.6535817) / 0.01230651   # +0.3%  sjq_3_3_k1 > 0.6536
        + 0.002650095 * max(0.0, 0.1122946 - Q.pz_lnd0) * max(0.0, Q.sj3_mass2 - 1.824785) / 0.2876784   # +0.3%  pz_lnd0 < 0.1123 and sj3_mass2 > 1.825
        - 0.002547045 * max(0.0, Q.sv_2_n - 1.0) / 0.07863667   # -0.3%  sv_2_n > 1
        + 0.002432933 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # +0.2%  sj2_mass2 < 1.852
        - 0.002072092 * max(0.0, 79.47361 - Q.mass) / 2.857843   # -0.2%  mass < 79.47
        + 0.001801977 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # +0.2%  n_pairs_kt_above_1 < 58
        - 0.001608503 * max(0.0, Q.mres_sd_mass_b0z005 - 178.8957) * max(0.0, Q.sj3_mass2 - 12.74295) / 21.53091   # -0.2%  mres_sd_mass_b0z005 > 178.9 and sj3_mass2 > 12.74
        - 0.001562865 * max(0.0, Q.z_displaced5 - 0.08090366) * max(0.0, 0.00953824 - Q.sjf_2_2_z_d3) / 0.0001650916   # -0.2%  z_displaced5 > 0.0809 and sjf_2_2_z_d3 < 0.009538
        + 0.001405108 * max(0.0, Q.lep_dr - 0.08178299) * max(0.0, Q.ak02_2_z - 0.131395) / 0.007209622   # +0.1%  lep_dr > 0.08178 and ak02_2_z > 0.1314
        + 0.001383829 * max(0.0, Q.mass_2charged - 24.4079) / 2.942332   # +0.1%  mass_2charged > 24.41
        + 0.001381177 * max(0.0, Q.N2 - 0.2875658) * max(0.0, Q.sjq_3_2_k1 - 0.6148962) / 0.0003098244   # +0.1%  N2 > 0.2876 and sjq_3_2_k1 > 0.6149
        + 0.001336921 * max(0.0, Q.n_s3d_above_3 - 1.0) * max(0.0, 0.8410552 - Q.tau54) / 0.07333908   # +0.1%  n_s3d_above_3 > 1 and tau54 < 0.8411
        + 0.001281881 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, 8.17233e-05 - Q.ecf_g42) / 9.735043e-05   # +0.1%  lep_ptrel > 27.32 and ecf_g42 < 8.172e-05
        + 0.001083347 * max(0.0, Q.sj3_pair_mass_min - 80.02563) / 0.702582   # +0.1%  sj3_pair_mass_min > 80.03
        - 0.001072315 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.sjf_3_3_n_d3 - 2.0) / 0.2897733   # -0.1%  n_s3d_above_3 < 10 and sjf_3_3_n_d3 > 2
        - 0.001032284 * max(0.0, Q.nca_sj4_pair_mass_2nd - 83.41384) / 1.649755   # -0.1%  nca_sj4_pair_mass_2nd > 83.41
        - 0.000981744 * max(0.0, Q.lep_dr - 0.08178299) * max(0.0, Q.sjf_4_3_z_d3 - 0.0) / 0.001055292   # -0.1%  lep_dr > 0.08178 and sjf_4_3_z_d3 > 0
        - 0.0009530923 * max(0.0, 16.0 - Q.n_photon) * max(0.0, 0.4091922 - Q.ak02_2_z) / 0.6580645   # -0.1%  n_photon < 16 and ak02_2_z < 0.4092
        + 0.0009449383 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.lepsj_2_n_d3 - 0.0) / 0.0926259   # +0.1%  lep_z < 0.2214 and lepsj_2_n_d3 > 0
        + 0.0008465527 * max(0.0, 2.0 - Q.lepsj_3_n_d3) * max(0.0, Q.lepsj_3_dr - 0.0) / 0.01942047   # +0.1%  lepsj_3_n_d3 < 2 and lepsj_3_dr > 0
        - 0.0008047623 * max(0.0, Q.sj3_pair_mass_min - 48.40547) * max(0.0, 2.755351 - Q.lne_10) / 0.2893877   # -0.1%  sj3_pair_mass_min > 48.41 and lne_10 < 2.755
        + 0.0007516324 * max(0.0, Q.kt2_dr12 - 0.6465173) / 0.003279283   # +0.1%  kt2_dr12 > 0.6465
        - 0.0006404791 * Q.pz_lnkt1 / 0.1020861   # -0.1%  pz_lnkt1
        + 0.0006384197 * max(0.0, 0.2801368 - Q.z_displaced5) * max(0.0, 0.006832265 - Q.C3_b2) / 0.0006903204   # +0.1%  z_displaced5 < 0.2801 and C3_b2 < 0.006832
        + 0.0004791208 * max(0.0, Q.lund1_lndelta - -0.3624079) / 0.0928942   # +0.0%  lund1_lndelta > -0.3624
        - 0.000471254 * max(0.0, Q.sjq_3_2_nch - 5.0) / 2.173277   # -0.0%  sjq_3_2_nch > 5
        + 0.0004704428 * max(0.0, Q.sjq_3_2_nch - 5.0) * max(0.0, 0.1016846 - Q.dzerr_9) / 0.1593706   # +0.0%  sjq_3_2_nch > 5 and dzerr_9 < 0.1017
        + 0.0003213123 * max(0.0, 0.01470468 - Q.tau3) / 0.0001776565   # +0.0%  tau3 < 0.0147
        + 0.0002929339 * max(0.0, Q.ak02_1_n_disp3 - 3.0) / 0.05635333   # +0.0%  ak02_1_n_disp3 > 3
        - 0.000247445 * max(0.0, Q.mres_sd_mass_b0z005 - 178.8957) / 1.714218   # -0.0%  mres_sd_mass_b0z005 > 178.9
        - 0.0002049692 * max(0.0, Q.mres_sd_mass_b0z005 - 125.8718) / 8.385721   # -0.0%  mres_sd_mass_b0z005 > 125.9
        - 0.0001666162 * max(0.0, Q.mass_displaced5 - 32.61679) / 0.6696441   # -0.0%  mass_displaced5 > 32.62
        + 0.000141156 * max(0.0, 0.2738646 - Q.kt2_dr12) / 0.02095188   # +0.0%  kt2_dr12 < 0.2739
        + 0.0001394196 * max(0.0, 0.3078242 - Q.sj3_dr12) / 0.06524017   # +0.0%  sj3_dr12 < 0.3078
        + 0.0001062718 * max(0.0, Q.mass_2photon - 22.18431) / 0.4406114   # +0.0%  mass_2photon > 22.18
        + 7.242064e-05 * max(0.0, 0.02148541 - Q.lam2) * max(0.0, Q.dc_mass_disp_2nd - 0.0) / 0.000724687   # +0.0%  lam2 < 0.02149 and dc_mass_disp_2nd > 0
        + 4.408717e-05 * max(0.0, 16.0 - Q.n_photon) * max(0.0, -2.411946 - Q.lnerel_3) / 1.077537   # +0.0%  n_photon < 16 and lnerel_3 < -2.412
        - 2.579495e-05 * max(0.0, Q.z_displaced5 - 0.08090366) * max(0.0, Q.iselectron_29 - 0.0) / 0.0002627898   # -0.0%  z_displaced5 > 0.0809 and iselectron_29 > 0
        + 1.553819e-05 * max(0.0, 0.1122946 - Q.pz_lnd0) * max(0.0, Q.jd_3d_6 - 0.9115584) / 0.2751505   # +0.0%  pz_lnd0 < 0.1123 and jd_3d_6 > 0.9116
        + 2.756252e-06 * max(0.0, Q.lep_dr - 0.08178299) * max(0.0, Q.ismuon_14 - 0.0) / 0.0005743788   # +0.0%  lep_dr > 0.08178 and ismuon_14 > 0
        - 1.89074e-06 * max(0.0, 79.47361 - Q.mass) * max(0.0, Q.C2_b2 - 0.02858957) / 0.1953539   # -0.0%  mass < 79.47 and C2_b2 > 0.02859
    )
    return z


def neuron_71(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.029709e-06
    )
    return z


def neuron_72(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.164122e-06
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
    z = 1.0 * (-5.542788e-06
    )
    return z


def neuron_77(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.087488e-06
    )
    return z


def neuron_78(Q):
    # scale S = 8.1; each line: share * term / its average size
    z = 8.099838 * (0.03136723
        - 0.1192454 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # -11.9%  lep_z < 0.3397
        - 0.1191059 * max(0.0, 18.7678 - Q.lep_ptrel) / 14.28156   # -11.9%  lep_ptrel < 18.77
        + 0.07877603 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.103005 - Q.lepsj_2_dr) / 0.02325369   # +7.9%  lep_z < 0.3397 and lepsj_2_dr < 0.103
        - 0.05275021 * max(0.0, 0.103005 - Q.lepsj_2_dr) / 0.08254947   # -5.3%  lepsj_2_dr < 0.103
        + 0.05242144 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +5.2%  lep_ptrel < 43.21
        - 0.05043141 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -5.0%  n_lepton < 1
        + 0.03235998 * max(0.0, 0.5083429 - Q.N2_b05) / 0.07626804   # +3.2%  N2_b05 < 0.5083
        + 0.0210951 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 8.17233e-05 - Q.ecf_g42) / 1.595489e-05   # +2.1%  lep_z < 0.3397 and ecf_g42 < 8.172e-05
        + 0.02071581 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 1.761851   # +2.1%  lepsj_3_maxsd0 < 2.413
        + 0.02064299 * max(0.0, 9.0 - Q.n_s3d_above_3) / 5.43421   # +2.1%  n_s3d_above_3 < 9
        - 0.017475 * max(0.0, Q.z_top15_slots - 0.8008865) / 0.07132828   # -1.7%  z_top15_slots > 0.8009
        - 0.01611416 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 22.18431 - Q.mass_2photon) / 237.1798   # -1.6%  lep_ptrel < 18.77 and mass_2photon < 22.18
        + 0.01439483 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, -7.886954 - Q.ktd_ln_d34) / 20.90232   # +1.4%  lep_ptrel < 18.77 and ktd_ln_d34 < -7.887
        + 0.01393847 * max(0.0, 99.0 - Q.n_pairs_kt_above_3) / 39.22243   # +1.4%  n_pairs_kt_above_3 < 99
        + 0.01385355 * max(0.0, 195.0 - Q.n_pairs_kt_above_1) / 37.35244   # +1.4%  n_pairs_kt_above_1 < 195
        + 0.01348964 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 0.1570831 - Q.sjf_4_2_z_d3) / 1.931364   # +1.3%  lep_ptrel < 18.77 and sjf_4_2_z_d3 < 0.1571
        + 0.01318414 * max(0.0, 2.220372 - Q.lepsj_2_maxsd0) / 1.502045   # +1.3%  lepsj_2_maxsd0 < 2.22
        + 0.01256607 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 749.0582 - Q.sum_pt_top30) / 2343.327   # +1.3%  lep_ptrel < 18.77 and sum_pt_top30 < 749.1
        + 0.0120555 * max(0.0, Q.sdb_2_n - 4.0) / 6.662073   # +1.2%  sdb_2_n > 4
        - 0.01194636 * max(0.0, 586.9572 - Q.lepsj_3_maxsd0) / 546.6092   # -1.2%  lepsj_3_maxsd0 < 587
        + 0.01183075 * max(0.0, Q.mass - 114.0172) / 14.73611   # +1.2%  mass > 114
        - 0.01102539 * max(0.0, -8.400697 - Q.ktd_ln_d34) / 1.139688   # -1.1%  ktd_ln_d34 < -8.401
        - 0.0105459 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, 9.0 - Q.n_neutral_had) / 195.063   # -1.1%  sj4_pair_mass_max < 115.5 and n_neutral_had < 9
        - 0.009872976 * max(0.0, 0.0142807 - Q.M3_b2) / 0.004049821   # -1.0%  M3_b2 < 0.01428
        + 0.008355743 * max(0.0, 2.0 - Q.n_lepton) / 1.440613   # +0.8%  n_lepton < 2
        + 0.008307583 * max(0.0, 2.0 - Q.lepsj_3_n_d3) / 1.694207   # +0.8%  lepsj_3_n_d3 < 2
        - 0.008294547 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) / 36.55147   # -0.8%  sj4_pair_mass_max < 115.5
        - 0.00819818 * max(0.0, 10.83159 - Q.mass_2charged) / 5.345466   # -0.8%  mass_2charged < 10.83
        + 0.008008145 * max(0.0, 1.839882 - Q.mass_2charged) / 0.4692955   # +0.8%  mass_2charged < 1.84
        + 0.007821074 * max(0.0, Q.mres_sd_mass_b0z005 - 122.1) / 9.418671   # +0.8%  mres_sd_mass_b0z005 > 122.1
        - 0.007582782 * max(0.0, Q.mres_sd_mass_b0z005 - 159.9242) / 3.119054   # -0.8%  mres_sd_mass_b0z005 > 159.9
        + 0.007543956 * max(0.0, Q.mass_top15 - 83.68425) / 10.75757   # +0.8%  mass_top15 > 83.68
        + 0.007458042 * max(0.0, 10.0 - Q.n_sdz_above_2) / 6.271463   # +0.7%  n_sdz_above_2 < 10
        - 0.00734427 * max(0.0, 95.14961 - Q.mass) / 6.371707   # -0.7%  mass < 95.15
        + 0.006813062 * max(0.0, 2.0 - Q.lepsj_3_n_d3) * max(0.0, 0.09790963 - Q.lepsj_3_dr) / 0.1531856   # +0.7%  lepsj_3_n_d3 < 2 and lepsj_3_dr < 0.09791
        + 0.006770188 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 0.1631992 - Q.sj4_dr_min) / 0.889978   # +0.7%  lep_ptrel < 18.77 and sj4_dr_min < 0.1632
        + 0.006750669 * max(0.0, 22.0 - Q.n_neutral) / 4.591883   # +0.7%  n_neutral < 22
        - 0.006301036 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 131.3722 - Q.lepsj_2_maxsd0) / 1680.384   # -0.6%  lep_ptrel < 18.77 and lepsj_2_maxsd0 < 131.4
        - 0.006214978 * max(0.0, 9.0 - Q.sdb_2_n) * max(0.0, Q.sjq_2_prod_k03 - -0.9998116) / 1.185663   # -0.6%  sdb_2_n < 9 and sjq_2_prod_k03 > -0.9998
        + 0.006146142 * max(0.0, Q.lund3_lndelta - -2.817283) / 1.325774   # +0.6%  lund3_lndelta > -2.817
        + 0.005969927 * max(0.0, 0.0772633 - Q.sdb_0_z) / 0.04579627   # +0.6%  sdb_0_z < 0.07726
        + 0.005704552 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 226.3008 - Q.sip_3d_2) / 106.0959   # +0.6%  n_s3d_above_3 > 4 and sip_3d_2 < 226.3
        - 0.00544942 * max(0.0, 0.2040425 - Q.sdb_2_z) / 0.02557338   # -0.5%  sdb_2_z < 0.204
        + 0.005283482 * max(0.0, 9.0 - Q.sdb_2_n) / 1.347003   # +0.5%  sdb_2_n < 9
        + 0.005135527 * max(0.0, Q.n_charged_had - 22.0) / 1.800493   # +0.5%  n_charged_had > 22
        + 0.005108993 * max(0.0, 7.0 - Q.n_pairs_kt_above_10) / 2.958697   # +0.5%  n_pairs_kt_above_10 < 7
        - 0.004944967 * max(0.0, 10.98011 - Q.dc_2_jp) / 6.863088   # -0.5%  dc_2_jp < 10.98
        + 0.004640232 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.mass_top20 - 86.78877) / 3.685704   # +0.5%  lep_z < 0.3397 and mass_top20 > 86.79
        - 0.004170851 * max(0.0, Q.pair_max_lnm2 - 7.347625) / 0.1021051   # -0.4%  pair_max_lnm2 > 7.348
        + 0.004032721 * max(0.0, 0.2801368 - Q.z_displaced5) / 0.1998239   # +0.4%  z_displaced5 < 0.2801
        - 0.0040025 * max(0.0, 2.0 - Q.lepsj_3_n_d3) * max(0.0, Q.jet_e - 835.8719) / 277.0793   # -0.4%  lepsj_3_n_d3 < 2 and jet_e > 835.9
        - 0.003934231 * max(0.0, 0.8153708 - Q.kt2_1_z) / 0.1186238   # -0.4%  kt2_1_z < 0.8154
        + 0.003897222 * max(0.0, Q.mass_displaced5 - 9.380468) / 3.157509   # +0.4%  mass_displaced5 > 9.38
        - 0.003727158 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, Q.ak02_2_z - 0.08487383) / 1.979896   # -0.4%  lep_ptrel < 18.77 and ak02_2_z > 0.08487
        + 0.003672445 * max(0.0, Q.lund3_lndelta - -2.817283) * max(0.0, 172.888 - Q.jd_3d_4) / 199.4197   # +0.4%  lund3_lndelta > -2.817 and jd_3d_4 < 172.9
        - 0.003267677 * max(0.0, Q.n_s3d_above_3 - 4.0) / 1.224997   # -0.3%  n_s3d_above_3 > 4
        + 0.003021563 * max(0.0, Q.sjf_2_1_n_d3 - 5.0) / 0.1972067   # +0.3%  sjf_2_1_n_d3 > 5
        + 0.002973959 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, Q.z_top50_slots - 0.9572293) / 0.04314808   # +0.3%  n_s3d_above_3 > 4 and z_top50_slots > 0.9572
        - 0.002910322 * max(0.0, Q.mass_displaced5 - 21.12768) / 1.493669   # -0.3%  mass_displaced5 > 21.13
        - 0.002903308 * max(0.0, 37.19471 - Q.sj3_pair_mass_min) / 9.096396   # -0.3%  sj3_pair_mass_min < 37.19
        - 0.002885282 * max(0.0, Q.sdb_2_z - 0.2740506) / 0.09626986   # -0.3%  sdb_2_z > 0.2741
        + 0.002876154 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.1956014 - Q.pz_lnd2) / 0.02306375   # +0.3%  lep_z < 0.3397 and pz_lnd2 < 0.1956
        + 0.002831035 * max(0.0, Q.mass_top20 - 130.7093) / 2.260459   # +0.3%  mass_top20 > 130.7
        + 0.00276609 * max(0.0, 72.862 - Q.sj4_pair_mass_max) / 6.86631   # +0.3%  sj4_pair_mass_max < 72.86
        + 0.002726187 * max(0.0, 46.19491 - Q.mass_neutral) / 10.1597   # +0.3%  mass_neutral < 46.19
        + 0.00272043 * max(0.0, Q.n_s3d_above_3 - 6.0) / 0.6254467   # +0.3%  n_s3d_above_3 > 6
        + 0.002715644 * max(0.0, 2.0 - Q.lepsj_3_n_d3) * max(0.0, 1.0 - Q.dc_2_n_lep) / 1.456113   # +0.3%  lepsj_3_n_d3 < 2 and dc_2_n_lep < 1
        + 0.002579799 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.ak02_n - 1.0) / 0.468621   # +0.3%  lep_z < 0.3397 and ak02_n > 1
        + 0.00238869 * max(0.0, 0.103005 - Q.lepsj_2_dr) * max(0.0, 0.1164034 - Q.jet_charge_k03) / 0.02637386   # +0.2%  lepsj_2_dr < 0.103 and jet_charge_k03 < 0.1164
        + 0.002255253 * max(0.0, 70.88236 - Q.mass_top40) / 1.88455   # +0.2%  mass_top40 < 70.88
        - 0.00217973 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 0.863865 - Q.tau32) / 0.2380229   # -0.2%  n_s3d_above_3 > 4 and tau32 < 0.8639
        + 0.002110456 * max(0.0, 2.097213e-06 - Q.e4) / 1.474625e-06   # +0.2%  e4 < 2.097e-06
        + 0.002075238 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, Q.nca_sj4_pairmax_over_mass - 0.6162845) / 1.424901   # +0.2%  lep_ptrel < 18.77 and nca_sj4_pairmax_over_mass > 0.6163
        - 0.001937963 * max(0.0, Q.mass_displaced5 - 9.380468) * max(0.0, 113.5019 - Q.mass_charged) / 112.3241   # -0.2%  mass_displaced5 > 9.38 and mass_charged < 113.5
        + 0.001856395 * max(0.0, 0.007652966 - Q.z_dr_0_0p05) / 0.002629287   # +0.2%  z_dr_0_0p05 < 0.007653
        + 0.00177793 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # +0.2%  max_abs_d0 < 10.52
        - 0.001538801 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # -0.2%  mass_top10 > 103.5
        - 0.001378674 * max(0.0, Q.n_sd0_above_5 - 4.0) / 0.7095433   # -0.1%  n_sd0_above_5 > 4
        - 0.001302548 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # -0.1%  lep_iso < 1.362
        + 0.001027582 * max(0.0, Q.mass_displaced3 - 39.09615) / 0.6617151   # +0.1%  mass_displaced3 > 39.1
        - 0.0009283873 * max(0.0, Q.z_top15_slots - 0.8008865) * max(0.0, Q.z_displaced5 - 0.2184442) / 0.001733934   # -0.1%  z_top15_slots > 0.8009 and z_displaced5 > 0.2184
        + 0.0004367841 * max(0.0, 0.1020077 - Q.pz_lnkt1) / 0.01583928   # +0.0%  pz_lnkt1 < 0.102
        + 0.0004004839 * max(0.0, 0.181327 - Q.sjf_4_1_z_d3) / 0.1342297   # +0.0%  sjf_4_1_z_d3 < 0.1813
        - 0.0003304931 * max(0.0, -2.049856 - Q.pair_mean_lndelta) / 0.3197094   # -0.0%  pair_mean_lndelta < -2.05
        + 0.000318961 * max(0.0, 2.0 - Q.lepsj_3_n_d3) * max(0.0, Q.ak02_3_n_lep - 0.0) / 0.06022333   # +0.0%  lepsj_3_n_d3 < 2 and ak02_3_n_lep > 0
        - 0.0002560599 * max(0.0, Q.mass_displaced3 - 39.09615) * max(0.0, 0.004703917 - Q.sjq_2_prod_k05) / 0.06411494   # -0.0%  mass_displaced3 > 39.1 and sjq_2_prod_k05 < 0.004704
        - 0.0001749191 * max(0.0, Q.n_muon - 1.0) / 0.02322   # -0.0%  n_muon > 1
        + 0.0001552808 * max(0.0, Q.mass - 164.4374) / 3.038568   # +0.0%  mass > 164.4
        + 0.0001349604 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, Q.dc_3_n_lep - 0.0) / 1.379311   # +0.0%  sj4_pair_mass_max < 115.5 and dc_3_n_lep > 0
        + 0.0001140826 * max(0.0, Q.mass_displaced5 - 9.380468) * max(0.0, 0.00489684 - Q.sjq_2_prod_k03) / 0.6193119   # +0.0%  mass_displaced5 > 9.38 and sjq_2_prod_k03 < 0.004897
        + 7.56664e-05 * max(0.0, 0.5083429 - Q.N2_b05) * max(0.0, 0.0 - Q.eta_28) / 0.003357017   # +0.0%  N2_b05 < 0.5083 and eta_28 < 0
        + 6.155944e-05 * max(0.0, Q.sjq_3_3_k1 - 0.6535817) / 0.01230651   # +0.0%  sjq_3_3_k1 > 0.6536
        - 5.210529e-05 * max(0.0, Q.mass_top40 - 173.1022) / 1.415364   # -0.0%  mass_top40 > 173.1
        + 4.956609e-05 * max(0.0, Q.mass_displaced5 - 9.380468) * max(0.0, 6.771002 - Q.jd_3d_5) / 1.084855   # +0.0%  mass_displaced5 > 9.38 and jd_3d_5 < 6.771
        + 4.14362e-05 * max(0.0, 0.5083429 - Q.N2_b05) * max(0.0, Q.kt2_min12_sd0_1 - 66.72222) / 0.950028   # +0.0%  N2_b05 < 0.5083 and kt2_min12_sd0_1 > 66.72
        + 2.872106e-05 * max(0.0, -0.1170754 - Q.tdz_1) / 0.01260556   # +0.0%  tdz_1 < -0.1171
        + 4.402306e-06 * max(0.0, Q.mass_displaced5 - 21.12768) * max(0.0, Q.charge_52 - 0.0) / 0.03019485   # +0.0%  mass_displaced5 > 21.13 and charge_52 > 0
        - 3.909013e-06 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, Q.ak02_3_z_disp3 - 0.01033072) / 0.01420718   # -0.0%  lep_ptrel < 18.77 and ak02_3_z_disp3 > 0.01033
        - 1.968849e-06 * max(0.0, -0.5140858 - Q.lund1_lndelta) / 0.155591   # -0.0%  lund1_lndelta < -0.5141
        - 1.905381e-06 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 3.0 - Q.sjf_4_3_n_d3) / 2.183207   # -0.0%  n_s3d_above_3 > 4 and sjf_4_3_n_d3 < 3
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
    # scale S = 5.787; each line: share * term / its average size
    z = 5.786751 * (0.01208891
        - 0.1753976 * max(0.0, 158.2019 - Q.mres_sd_mass_b2z01) / 55.4697   # -17.5%  mres_sd_mass_b2z01 < 158.2
        + 0.08784543 * max(0.0, 125.2732 - Q.mres_sd_mass_b2z01) / 27.56259   # +8.8%  mres_sd_mass_b2z01 < 125.3
        + 0.06554724 * max(0.0, 115.0388 - Q.mres_sd_mass_b2z01) / 20.62743   # +6.6%  mres_sd_mass_b2z01 < 115
        - 0.06078351 * max(0.0, 91.15481 - Q.mres_sd_mass_b2z01) / 9.5029   # -6.1%  mres_sd_mass_b2z01 < 91.15
        + 0.03447271 * max(0.0, 0.01162881 - Q.ecf_g31) / 0.005313715   # +3.4%  ecf_g31 < 0.01163
        - 0.03317123 * max(0.0, 226.3008 - Q.sip_3d_2) * max(0.0, 6.771002 - Q.jd_3d_5) / 605.5688   # -3.3%  sip_3d_2 < 226.3 and jd_3d_5 < 6.771
        - 0.03047669 * max(0.0, 15.0 - Q.n_dr_0p4_up) / 9.48698   # -3.0%  n_dr_0p4_up < 15
        - 0.02845836 * max(0.0, 0.06117886 - Q.sum_zz_dr2) / 0.02616533   # -2.8%  sum_zz_dr2 < 0.06118
        + 0.02812611 * max(0.0, 4.004982 - Q.jd_3d_5) / 1.451954   # +2.8%  jd_3d_5 < 4.005
        - 0.02521962 * max(0.0, 0.005262883 - Q.lepsj_3_dr) / 0.003463974   # -2.5%  lepsj_3_dr < 0.005263
        + 0.02462829 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # +2.5%  sip_3d_2 < 226.3
        + 0.02226825 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # +2.2%  max_abs_d0 < 10.52
        - 0.0211987 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.z_charged_had - 0.265564) / 0.02462507   # -2.1%  z_displaced3 > 0.03624 and z_charged_had > 0.2656
        + 0.01839432 * max(0.0, 3.0 - Q.sdb_0_n) / 1.623507   # +1.8%  sdb_0_n < 3
        + 0.01786756 * max(0.0, 0.01162881 - Q.ecf_g31) * max(0.0, 1.0 - Q.ak02_2_n_lep) / 0.004333109   # +1.8%  ecf_g31 < 0.01163 and ak02_2_n_lep < 1
        - 0.01781894 * max(0.0, 5.0 - Q.sdb_0_n) / 3.42113   # -1.8%  sdb_0_n < 5
        + 0.01754287 * max(0.0, 0.04510459 - Q.sdb_4_z) * max(0.0, 0.004135872 - Q.lep_z) / 8.841111e-05   # +1.8%  sdb_4_z < 0.0451 and lep_z < 0.004136
        + 0.01722059 * max(0.0, Q.z_displaced3 - 0.03624058) / 0.08737704   # +1.7%  z_displaced3 > 0.03624
        + 0.01608219 * max(0.0, 96.62031 - Q.mres_sd_mass_b2z01) / 11.56393   # +1.6%  mres_sd_mass_b2z01 < 96.62
        - 0.01557608 * max(0.0, 0.06416437 - Q.z_displaced3) / 0.02655242   # -1.6%  z_displaced3 < 0.06416
        - 0.01538352 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 6.43167 - Q.jd_3d_4) / 19.90242   # -1.5%  max_abs_d0 < 10.52 and jd_3d_4 < 6.432
        - 0.01480439 * max(0.0, 14.38858 - Q.lep_iso) / 12.08589   # -1.5%  lep_iso < 14.39
        + 0.01215268 * max(0.0, Q.n_s3d_above_3 - 6.0) / 0.6254467   # +1.2%  n_s3d_above_3 > 6
        + 0.01207071 * max(0.0, 76.1529 - Q.mres_sd_mass_b2z01) / 5.369584   # +1.2%  mres_sd_mass_b2z01 < 76.15
        + 0.01019501 * max(0.0, 0.06416437 - Q.z_displaced3) * max(0.0, 1.0 - Q.dc_2_n_lep) / 0.02413578   # +1.0%  z_displaced3 < 0.06416 and dc_2_n_lep < 1
        + 0.01012533 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 1.0 - Q.ak02_min12_n_disp3) / 5.28269   # +1.0%  max_abs_d0 < 10.52 and ak02_min12_n_disp3 < 1
        - 0.007985087 * max(0.0, 2.044945 - Q.sjf_2_2_max3d) / 0.1617863   # -0.8%  sjf_2_2_max3d < 2.045
        - 0.007952563 * max(0.0, 0.01162881 - Q.ecf_g31) * max(0.0, Q.sjq_2_prod_k05 - -0.5057096) / 0.00247391   # -0.8%  ecf_g31 < 0.01163 and sjq_2_prod_k05 > -0.5057
        - 0.007115871 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 7.0 - Q.n_s3d_above_3) / 0.1116745   # -0.7%  z_displaced3 > 0.03624 and n_s3d_above_3 < 7
        - 0.007017927 * max(0.0, 0.04510459 - Q.sdb_4_z) * max(0.0, Q.jet_abs_eta - 0.1941339) / 0.01976022   # -0.7%  sdb_4_z < 0.0451 and jet_abs_eta > 0.1941
        - 0.006801595 * max(0.0, Q.sv_n - 1.0) / 0.43993   # -0.7%  sv_n > 1
        + 0.00673606 * max(0.0, 0.04510459 - Q.sdb_4_z) / 0.03498975   # +0.7%  sdb_4_z < 0.0451
        + 0.006687196 * max(0.0, Q.n_s3d_above_10 - 3.0) / 0.9107233   # +0.7%  n_s3d_above_10 > 3
        + 0.006345131 * max(0.0, 0.01162881 - Q.ecf_g31) * max(0.0, Q.sjq_2_sumabs_k1 - 0.1588551) / 0.001231486   # +0.6%  ecf_g31 < 0.01163 and sjq_2_sumabs_k1 > 0.1589
        + 0.006342699 * max(0.0, Q.sj3_pairmin_over_m - 0.2477126) / 0.09094898   # +0.6%  sj3_pairmin_over_m > 0.2477
        + 0.006049832 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.n_lund_kt_above_1 - 2.0) / 0.2755117   # +0.6%  z_displaced3 > 0.03624 and n_lund_kt_above_1 > 2
        - 0.004877242 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # -0.5%  mass_top50 > 161.1
        - 0.004535497 * max(0.0, 2.0 - Q.lepsj_3_n_d3) / 1.694207   # -0.5%  lepsj_3_n_d3 < 2
        + 0.004346503 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # +0.4%  n_pairs_kt_above_1 < 58
        - 0.00423593 * max(0.0, 0.1339824 - Q.pz_lnd2) / 0.04287826   # -0.4%  pz_lnd2 < 0.134
        + 0.004139293 * max(0.0, 0.01162881 - Q.ecf_g31) * max(0.0, Q.n_s3d_above_3 - 4.0) / 0.005782324   # +0.4%  ecf_g31 < 0.01163 and n_s3d_above_3 > 4
        - 0.00413735 * max(0.0, Q.sjf_4_n2disp - 1.0) / 0.2405067   # -0.4%  sjf_4_n2disp > 1
        + 0.004110158 * max(0.0, 0.005262883 - Q.lepsj_3_dr) * max(0.0, 2.953464 - Q.jd_3d_4) / 0.002212736   # +0.4%  lepsj_3_dr < 0.005263 and jd_3d_4 < 2.953
        + 0.003912848 * max(0.0, 12.49716 - Q.sjf_2_1_mass_d3) / 10.17129   # +0.4%  sjf_2_1_mass_d3 < 12.5
        + 0.003698569 * max(0.0, Q.mass_top40 - 115.7429) / 10.91182   # +0.4%  mass_top40 > 115.7
        - 0.003602516 * max(0.0, Q.mass_top50 - 129.5874) / 7.866327   # -0.4%  mass_top50 > 129.6
        + 0.00311882 * max(0.0, 0.003472016 - Q.sum_z_dr2_top2) / 0.0003923846   # +0.3%  sum_z_dr2_top2 < 0.003472
        - 0.002961465 * max(0.0, 1.777286 - Q.mass_displaced3) / 0.7377974   # -0.3%  mass_displaced3 < 1.777
        + 0.002919852 * max(0.0, 0.1076451 - Q.tau2) / 0.0393622   # +0.3%  tau2 < 0.1076
        - 0.002722446 * max(0.0, 1.514826 - Q.min_pair_mass) / 0.4657734   # -0.3%  min_pair_mass < 1.515
        + 0.002665891 * max(0.0, Q.n_s3d_above_3 - 2.0) * max(0.0, 981.6443 - Q.jet_e) / 399.6224   # +0.3%  n_s3d_above_3 > 2 and jet_e < 981.6
        + 0.002656421 * max(0.0, Q.n_s3d_above_3 - 2.0) / 2.23187   # +0.3%  n_s3d_above_3 > 2
        - 0.002615057 * max(0.0, 0.06416437 - Q.z_displaced3) * max(0.0, Q.eccentricity - 0.7717404) / 0.00194909   # -0.3%  z_displaced3 < 0.06416 and eccentricity > 0.7717
        - 0.00259834 * Q.dc_ntag / 0.2920233   # -0.3%  dc_ntag
        + 0.002585411 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 1262.672 - Q.jd_sum_abs_sd0_top3) / 63.42777   # +0.3%  z_displaced3 > 0.03624 and jd_sum_abs_sd0_top3 < 1263
        - 0.002580516 * Q.lepsj_2_n_d3 / 0.6197967   # -0.3%  lepsj_2_n_d3
        + 0.002480589 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 16.23838 - Q.sip_3d_3) / 0.2292165   # +0.2%  z_displaced3 > 0.03624 and sip_3d_3 < 16.24
        - 0.002104394 * max(0.0, 0.1339824 - Q.pz_lnd2) * max(0.0, 15.0 - Q.sjq_2_2_nch) / 0.3887088   # -0.2%  pz_lnd2 < 0.134 and sjq_2_2_nch < 15
        - 0.001993694 * max(0.0, 15.0 - Q.n_dr_0p4_up) * max(0.0, Q.nca_sj4_pair2nd_over_mass - 0.4488456) / 0.6294244   # -0.2%  n_dr_0p4_up < 15 and nca_sj4_pair2nd_over_mass > 0.4488
        - 0.001879026 * max(0.0, 59.5457 - Q.sj4_pair_mass_max) / 2.921503   # -0.2%  sj4_pair_mass_max < 59.55
        - 0.00166201 * max(0.0, Q.sjf_2_1_n_d3 - 5.0) / 0.1972067   # -0.2%  sjf_2_1_n_d3 > 5
        - 0.001590447 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # -0.2%  n_pairs_kt_above_3 < 28
        + 0.001590204 * max(0.0, 0.01162881 - Q.ecf_g31) * max(0.0, Q.sjf_4_3_z_d3 - 0.005113028) / 5.142948e-05   # +0.2%  ecf_g31 < 0.01163 and sjf_4_3_z_d3 > 0.005113
        + 0.001383775 * max(0.0, 158.2019 - Q.mres_sd_mass_b2z01) * max(0.0, Q.mass_2charged - 24.4079) / 104.2811   # +0.1%  mres_sd_mass_b2z01 < 158.2 and mass_2charged > 24.41
        + 0.001375446 * max(0.0, 0.06074238 - Q.tau1) / 0.001113326   # +0.1%  tau1 < 0.06074
        + 0.001289493 * max(0.0, Q.pair_mean_lndelta - -1.352792) / 0.01174331   # +0.1%  pair_mean_lndelta > -1.353
        - 0.0009597773 * max(0.0, Q.n_s3d_above_3 - 2.0) * max(0.0, Q.dc_n - 2.0) / 1.1914   # -0.1%  n_s3d_above_3 > 2 and dc_n > 2
        + 0.0009221863 * max(0.0, Q.n_s3d_above_3 - 2.0) * max(0.0, 4.0 - Q.n_lund_kt_above_5) / 4.131823   # +0.1%  n_s3d_above_3 > 2 and n_lund_kt_above_5 < 4
        - 0.000878768 * max(0.0, 226.3008 - Q.sip_3d_2) * max(0.0, Q.lepsj_2_n_d3 - 2.0) / 19.01415   # -0.1%  sip_3d_2 < 226.3 and lepsj_2_n_d3 > 2
        + 0.0008178719 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.isphoton_32 - 0.0) / 0.03002655   # +0.1%  z_displaced3 > 0.03624 and isphoton_32 > 0
        + 0.0008078743 * max(0.0, 2.0 - Q.lepsj_3_n_d3) * max(0.0, Q.ak02_min12_n_disp3 - 1.0) / 0.05119667   # +0.1%  lepsj_3_n_d3 < 2 and ak02_min12_n_disp3 > 1
        - 0.00073969 * max(0.0, Q.n_s3d_above_3 - 2.0) * max(0.0, 0.4982257 - Q.z_photon) / 0.5738664   # -0.1%  n_s3d_above_3 > 2 and z_photon < 0.4982
        + 0.0006235293 * max(0.0, Q.n_s3d_above_3 - 10.0) / 0.12486   # +0.1%  n_s3d_above_3 > 10
        - 0.0005610191 * max(0.0, Q.n_s3d_above_3 - 2.0) * max(0.0, 0.8501496 - Q.sj3_pairmax_over_m) / 0.1363705   # -0.1%  n_s3d_above_3 > 2 and sj3_pairmax_over_m < 0.8501
        + 0.0005437133 * max(0.0, 6.465318 - Q.sj2_mass2) / 0.5968622   # +0.1%  sj2_mass2 < 6.465
        + 0.0005084566 * max(0.0, 0.2040425 - Q.sdb_2_z) / 0.02557338   # +0.1%  sdb_2_z < 0.204
        + 0.0004572362 * max(0.0, 2.055951 - Q.sjf_2_1_maxsd0) / 0.15069   # +0.0%  sjf_2_1_maxsd0 < 2.056
        + 0.0004128697 * max(0.0, -0.1170754 - Q.tdz_1) / 0.01260556   # +0.0%  tdz_1 < -0.1171
        + 0.0003559315 * max(0.0, 15.0 - Q.n_dr_0p4_up) * max(0.0, Q.n_muon - 1.0) / 0.2205367   # +0.0%  n_dr_0p4_up < 15 and n_muon > 1
        - 0.0003367682 * max(0.0, Q.sjf_2_2_n_d3 - 5.0) / 0.06871333   # -0.0%  sjf_2_2_n_d3 > 5
        + 0.0003116388 * max(0.0, 125.2732 - Q.mres_sd_mass_b2z01) * max(0.0, Q.dr_3 - 0.1090736) / 0.7749741   # +0.0%  mres_sd_mass_b2z01 < 125.3 and dr_3 > 0.1091
        + 0.0002908147 * max(0.0, 0.6403502 - Q.max_dr) / 0.07137735   # +0.0%  max_dr < 0.6404
        + 0.0002638386 * max(0.0, 0.1325326 - Q.pz_lnd0) / 0.03416029   # +0.0%  pz_lnd0 < 0.1325
        + 0.000255218 * max(0.0, 0.01162881 - Q.ecf_g31) * max(0.0, 577.991 - Q.sip_3d_3) / 2.807757   # +0.0%  ecf_g31 < 0.01163 and sip_3d_3 < 578
        + 0.0002366809 * max(0.0, Q.mass_top40 - 115.7429) * max(0.0, Q.tau43 - 0.6363796) / 1.68249   # +0.0%  mass_top40 > 115.7 and tau43 > 0.6364
        + 0.0002179043 * max(0.0, Q.sjf_2_1_n_d3 - 5.0) * max(0.0, Q.kt2_2_charge - 0.06887309) / 0.0414089   # +0.0%  sjf_2_1_n_d3 > 5 and kt2_2_charge > 0.06887
        + 0.0002022287 * max(0.0, Q.mass_top40 - 115.7429) * max(0.0, 0.0 - Q.tdz_13) / 0.4599529   # +0.0%  mass_top40 > 115.7 and tdz_13 < 0
        + 0.0001626879 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.ismuon_3 - 0.0) / 0.001924006   # +0.0%  z_displaced3 > 0.03624 and ismuon_3 > 0
        + 0.0001583473 * max(0.0, 0.1339824 - Q.pz_lnd2) * max(0.0, Q.iselectron_3 - 0.0) / 0.0005178098   # +0.0%  pz_lnd2 < 0.134 and iselectron_3 > 0
        - 0.0001440826 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.nca_kt_above_2 - 4.0) / 0.02092076   # -0.0%  z_displaced3 > 0.03624 and nca_kt_above_2 > 4
        + 0.0001418821 * max(0.0, 0.1339824 - Q.pz_lnd2) * max(0.0, Q.sjf_4_4_z_d3 - 0.003370318) / 0.0001323675   # +0.0%  pz_lnd2 < 0.134 and sjf_4_4_z_d3 > 0.00337
        - 0.0001358363 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, Q.eta_18 - 0.3122559) / 0.003493027   # -0.0%  n_s3d_above_3 > 6 and eta_18 > 0.3123
        - 0.0001060931 * max(0.0, 0.2040425 - Q.sdb_2_z) * max(0.0, 0.6363796 - Q.tau43) / 0.0002127447   # -0.0%  sdb_2_z < 0.204 and tau43 < 0.6364
        - 0.0001052524 * max(0.0, Q.sv_n - 1.0) * max(0.0, 0.8953628 - Q.tau54) / 0.02142264   # -0.0%  sv_n > 1 and tau54 < 0.8954
        - 4.761341e-05 * max(0.0, 0.06117886 - Q.sum_zz_dr2) * max(0.0, Q.jd_3d_6 - 16.11752) / 0.1159432   # -0.0%  sum_zz_dr2 < 0.06118 and jd_3d_6 > 16.12
        - 4.268463e-05 * max(0.0, 59.5457 - Q.sj4_pair_mass_max) * max(0.0, Q.ak02_2_sd0_3 - -1.729543) / 5.335037   # -0.0%  sj4_pair_mass_max < 59.55 and ak02_2_sd0_3 > -1.73
        + 9.52999e-06 * max(0.0, 1.777286 - Q.mass_displaced3) * max(0.0, Q.kt2_1_sd0_3 - 1.376447) / 0.04921625   # +0.0%  mass_displaced3 < 1.777 and kt2_1_sd0_3 > 1.376
        + 4.876652e-06 * max(0.0, -0.1170754 - Q.tdz_1) * max(0.0, Q.kt2_1_sd0_3 - 13.98058) / 0.1263218   # +0.0%  tdz_1 < -0.1171 and kt2_1_sd0_3 > 13.98
        - 3.98803e-06 * max(0.0, Q.sjf_2_1_n_d3 - 5.0) * max(0.0, 2.627723 - Q.ak02_2_sd0_3) / 2.518858   # -0.0%  sjf_2_1_n_d3 > 5 and ak02_2_sd0_3 < 2.628
        - 2.043059e-06 * max(0.0, Q.mass_top40 - 115.7429) * max(0.0, Q.isnhad_34 - 0.0) / 0.712341   # -0.0%  mass_top40 > 115.7 and isnhad_34 > 0
    )
    return z


def neuron_82(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.942073e-06
    )
    return z


def neuron_83(Q):
    # scale S = 8.357; each line: share * term / its average size
    z = 8.35711 * (0.1582698
        - 0.1168018 * max(0.0, Q.mass - 79.47361) / 38.31536   # -11.7%  mass > 79.47
        + 0.07508575 * max(0.0, 0.0925671 - Q.sum_z_dr2_top50) / 0.05494881   # +7.5%  sum_z_dr2_top50 < 0.09257
        - 0.06305767 * max(0.0, Q.mres_sd_mass_b2z01 - 81.19466) / 31.20897   # -6.3%  mres_sd_mass_b2z01 > 81.19
        + 0.05756424 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) / 36.55147   # +5.8%  sj4_pair_mass_max < 115.5
        - 0.056683 * max(0.0, 139.2565 - Q.mres_sd_mass_b2z01) / 38.89774   # -5.7%  mres_sd_mass_b2z01 < 139.3
        - 0.03382217 * max(0.0, 1.139376 - Q.D3_b05) / 0.7023584   # -3.4%  D3_b05 < 1.139
        - 0.03088862 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 6.771002 - Q.jd_3d_5) / 24.1852   # -3.1%  max_abs_d0 < 10.52 and jd_3d_5 < 6.771
        - 0.02906902 * max(0.0, 6.0 - Q.n_s3d_above_10) / 3.631173   # -2.9%  n_s3d_above_10 < 6
        + 0.02812606 * max(0.0, Q.mass - 117.4867) / 13.09248   # +2.8%  mass > 117.5
        + 0.02724417 * max(0.0, Q.tau5 - 0.01467699) / 0.01818842   # +2.7%  tau5 > 0.01468
        - 0.02685515 * max(0.0, 0.02160244 - Q.M3_b2) / 0.009628213   # -2.7%  M3_b2 < 0.0216
        + 0.02546465 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # +2.5%  max_abs_d0 < 10.52
        - 0.02533629 * max(0.0, 3.53193e-06 - Q.e4) / 2.734524e-06   # -2.5%  e4 < 3.532e-06
        + 0.02452899 * max(0.0, 13.03663 - Q.mass_displaced3) * max(0.0, 6.771002 - Q.jd_3d_5) / 38.55232   # +2.5%  mass_displaced3 < 13.04 and jd_3d_5 < 6.771
        - 0.02435958 * max(0.0, 0.5109872 - Q.sdb_2_z) / 0.2120326   # -2.4%  sdb_2_z < 0.511
        + 0.02340906 * max(0.0, Q.mres_sd_mass_b2z01 - 130.0968) / 7.118079   # +2.3%  mres_sd_mass_b2z01 > 130.1
        - 0.0229348 * max(0.0, Q.mass_top40 - 78.33213) * max(0.0, 0.1275041 - Q.lep_z) / 3.263902   # -2.3%  mass_top40 > 78.33 and lep_z < 0.1275
        - 0.02262722 * max(0.0, Q.mass - 95.14961) / 26.15323   # -2.3%  mass > 95.15
        + 0.01504905 * max(0.0, 0.06416437 - Q.z_displaced3) / 0.02655242   # +1.5%  z_displaced3 < 0.06416
        + 0.01409317 * max(0.0, Q.mass_top40 - 78.33213) / 34.93312   # +1.4%  mass_top40 > 78.33
        - 0.01393085 * max(0.0, 0.03767806 - Q.sv_2_z) / 0.03276122   # -1.4%  sv_2_z < 0.03768
        + 0.01338236 * max(0.0, 0.06573337 - Q.lepsj_2_dr) / 0.04985349   # +1.3%  lepsj_2_dr < 0.06573
        - 0.01254051 * max(0.0, Q.mass_top15 - 77.22442) / 13.98893   # -1.3%  mass_top15 > 77.22
        - 0.01252911 * max(0.0, 24.0 - Q.n_pt_above_5) / 4.457557   # -1.3%  n_pt_above_5 < 24
        + 0.01058641 * max(0.0, 6.0 - Q.n_s3d_above_10) * max(0.0, Q.lne_0 - 4.380463) / 3.270558   # +1.1%  n_s3d_above_10 < 6 and lne_0 > 4.38
        + 0.0105521 * max(0.0, 7.028704 - Q.sip_3d_1) / 1.023735   # +1.1%  sip_3d_1 < 7.029
        - 0.009716707 * max(0.0, 6.0 - Q.n_s3d_above_10) * max(0.0, Q.sdb_2_z - 0.2277628) / 0.5379899   # -1.0%  n_s3d_above_10 < 6 and sdb_2_z > 0.2278
        - 0.008936872 * max(0.0, 75.78208 - Q.sj4_pair_mass_max) * max(0.0, 0.07303924 - Q.sjf_4_3_z_d3) / 0.535129   # -0.9%  sj4_pair_mass_max < 75.78 and sjf_4_3_z_d3 < 0.07304
        - 0.008553061 * max(0.0, 75.78208 - Q.sj4_pair_mass_max) / 8.111726   # -0.9%  sj4_pair_mass_max < 75.78
        + 0.008192453 * max(0.0, Q.mass_top40 - 155.8928) / 2.693823   # +0.8%  mass_top40 > 155.9
        + 0.0079983 * max(0.0, 0.1681173 - Q.z_displaced3) / 0.09545189   # +0.8%  z_displaced3 < 0.1681
        + 0.007392422 * max(0.0, 139.2565 - Q.mres_sd_mass_b2z01) * max(0.0, 1.0 - Q.dc_2_n_lep) / 32.88249   # +0.7%  mres_sd_mass_b2z01 < 139.3 and dc_2_n_lep < 1
        - 0.00730044 * max(0.0, 6.185635 - Q.lep_iso) / 4.883283   # -0.7%  lep_iso < 6.186
        + 0.006717363 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 172.888 - Q.jd_3d_4) / 66.43449   # +0.7%  n_s3d_above_3 > 6 and jd_3d_4 < 172.9
        - 0.006494326 * max(0.0, 185.1889 - Q.jd_sum_abs_sd0_top5) / 79.19909   # -0.6%  jd_sum_abs_sd0_top5 < 185.2
        - 0.005902317 * max(0.0, 9.0 - Q.n_sd0_above_3) / 5.844283   # -0.6%  n_sd0_above_3 < 9
        - 0.005730733 * max(0.0, 1.262841 - Q.mass_displaced5) / 0.5499328   # -0.6%  mass_displaced5 < 1.263
        + 0.00567374 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # +0.6%  n_s3d_above_3 < 3
        - 0.004978807 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, 0.3093201 - Q.z_neutral_had) / 6.179422   # -0.5%  sj4_pair_mass_max < 115.5 and z_neutral_had < 0.3093
        - 0.004929741 * max(0.0, 185.1889 - Q.jd_sum_abs_sd0_top5) * max(0.0, 0.003370318 - Q.sjf_4_4_z_d3) / 0.2445938   # -0.5%  jd_sum_abs_sd0_top5 < 185.2 and sjf_4_4_z_d3 < 0.00337
        + 0.004710011 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 0.4866692 - Q.sj3_pairmin_over_m) / 0.118201   # +0.5%  n_s3d_above_3 > 6 and sj3_pairmin_over_m < 0.4867
        + 0.004595476 * max(0.0, 0.1681173 - Q.z_displaced3) * max(0.0, 12.91148 - Q.lepsj_3_mass) / 0.9795791   # +0.5%  z_displaced3 < 0.1681 and lepsj_3_mass < 12.91
        + 0.003908174 * max(0.0, 46.74905 - Q.lepsj_3_maxsd0) / 40.371   # +0.4%  lepsj_3_maxsd0 < 46.75
        + 0.003899323 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +0.4%  n_dr_0p4_up < 4
        + 0.00381768 * max(0.0, 0.1530389 - Q.sdb_2_z) / 0.01282076   # +0.4%  sdb_2_z < 0.153
        - 0.003763172 * max(0.0, 162.7874 - Q.mass_top30) * max(0.0, Q.mass_displaced5 - 9.380468) / 137.2553   # -0.4%  mass_top30 < 162.8 and mass_displaced5 > 9.38
        + 0.003596661 * max(0.0, 27.26979 - Q.mres_sd_prong_mass1) / 8.356534   # +0.4%  mres_sd_prong_mass1 < 27.27
        + 0.003492358 * Q.pz_lnd2 / 0.1223855   # +0.3%  pz_lnd2
        + 0.003359522 * max(0.0, 0.02160244 - Q.M3_b2) * max(0.0, 6.0 - Q.sjq_3_3_nch) / 0.02377485   # +0.3%  M3_b2 < 0.0216 and sjq_3_3_nch < 6
        - 0.00331986 * max(0.0, 0.258375 - Q.N2) / 0.01851814   # -0.3%  N2 < 0.2584
        + 0.003198358 * max(0.0, Q.sjf_2_1_n_d3 - 5.0) / 0.1972067   # +0.3%  sjf_2_1_n_d3 > 5
        + 0.002728373 * max(0.0, 10.0 - Q.sdb_2_n) / 1.820263   # +0.3%  sdb_2_n < 10
        + 0.002579721 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 1292.625 - Q.jd_sum_abs_sd0_top5) / 301.995   # +0.3%  n_s3d_above_3 > 6 and jd_sum_abs_sd0_top5 < 1293
        - 0.002535133 * max(0.0, 0.1530389 - Q.sdb_2_z) * max(0.0, 4.004982 - Q.jd_3d_5) / 0.01187781   # -0.3%  sdb_2_z < 0.153 and jd_3d_5 < 4.005
        + 0.002524563 * max(0.0, 139.2565 - Q.mres_sd_mass_b2z01) * max(0.0, Q.lund1_lnz - -2.478371) / 7.228734   # +0.3%  mres_sd_mass_b2z01 < 139.3 and lund1_lnz > -2.478
        + 0.00241978 * max(0.0, Q.pair_max_lnm2 - 7.524613) / 0.07146798   # +0.2%  pair_max_lnm2 > 7.525
        - 0.002187559 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 4.821289 - Q.lne_3) / 0.5362893   # -0.2%  n_s3d_above_3 > 6 and lne_3 < 4.821
        - 0.002127275 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 106.3958   # -0.2%  sj4_pair_mass_max < 115.5 and n_dr_0p2_0p4 > 9
        + 0.002116265 * max(0.0, 0.03767806 - Q.sv_2_z) * max(0.0, Q.max_dr - 0.8033751) / 0.002523979   # +0.2%  sv_2_z < 0.03768 and max_dr > 0.8034
        + 0.002093044 * max(0.0, 6.0 - Q.n_s3d_above_10) * max(0.0, 5.969698e-05 - Q.ecf_g42) / 0.000137147   # +0.2%  n_s3d_above_10 < 6 and ecf_g42 < 5.97e-05
        + 0.002032328 * max(0.0, Q.n_s3d_above_3 - 6.0) / 0.6254467   # +0.2%  n_s3d_above_3 > 6
        - 0.001938451 * max(0.0, 24.0 - Q.n_pt_above_5) * max(0.0, 0.103005 - Q.lepsj_2_dr) / 0.3875471   # -0.2%  n_pt_above_5 < 24 and lepsj_2_dr < 0.103
        + 0.001818791 * max(0.0, Q.mres_sd_mass_b2z01 - 81.19466) * max(0.0, 16.9414 - Q.jd_3d_5) / 315.0758   # +0.2%  mres_sd_mass_b2z01 > 81.19 and jd_3d_5 < 16.94
        - 0.001671978 * max(0.0, 4.498447 - Q.dc_2_jp) / 2.19927   # -0.2%  dc_2_jp < 4.498
        - 0.001603685 * max(0.0, 0.02210827 - Q.dr_max_012) / 0.001265414   # -0.2%  dr_max_012 < 0.02211
        + 0.001596002 * max(0.0, Q.mass - 79.47361) * max(0.0, 0.2535652 - Q.dr_15) / 2.625286   # +0.2%  mass > 79.47 and dr_15 < 0.2536
        - 0.001457762 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # -0.1%  mass_top10 > 103.5
        - 0.001389449 * max(0.0, 0.06074238 - Q.tau1) / 0.001113326   # -0.1%  tau1 < 0.06074
        - 0.001222954 * max(0.0, Q.mres_sd_mass_b2z01 - 158.2019) / 3.171488   # -0.1%  mres_sd_mass_b2z01 > 158.2
        - 0.001124068 * max(0.0, 86.14266 - Q.mres_pruned_mass) / 13.9802   # -0.1%  mres_pruned_mass < 86.14
        + 0.001119011 * max(0.0, 2.0 - Q.lepsj_3_n_d3) / 1.694207   # +0.1%  lepsj_3_n_d3 < 2
        + 0.001083731 * max(0.0, Q.mres_sd_mass_b2z01 - 130.0968) * max(0.0, 10.7744 - Q.jd_3d_5) / 33.3057   # +0.1%  mres_sd_mass_b2z01 > 130.1 and jd_3d_5 < 10.77
        - 0.0008737614 * max(0.0, 162.7874 - Q.mass_top30) / 59.65527   # -0.1%  mass_top30 < 162.8
        - 0.00053991 * max(0.0, 5.0 - Q.n_charged_pt_above_10) / 0.18785   # -0.1%  n_charged_pt_above_10 < 5
        - 0.0005260441 * max(0.0, Q.sjf_4_n2disp - 2.0) / 0.03589   # -0.1%  sjf_4_n2disp > 2
        - 0.0004690482 * max(0.0, 13.03663 - Q.mass_displaced3) / 8.661134   # -0.0%  mass_displaced3 < 13.04
        + 0.0004097134 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, Q.tdz_1 - 0.05116378) / 0.5124873   # +0.0%  sj4_pair_mass_max < 115.5 and tdz_1 > 0.05116
        - 0.0003505605 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 14.85973 - Q.jd_3d_4) / 0.3569867   # -0.0%  n_s3d_above_3 > 6 and jd_3d_4 < 14.86
        - 0.0002780721 * max(0.0, Q.sjf_2_1_n_d3 - 5.0) * max(0.0, 0.1300849 - Q.dc_tag_2nd) / 0.0098277   # -0.0%  sjf_2_1_n_d3 > 5 and dc_tag_2nd < 0.1301
        + 0.0002027289 * max(0.0, -0.1170754 - Q.tdz_1) / 0.01260556   # +0.0%  tdz_1 < -0.1171
        - 0.0001666181 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, Q.ak02_3_n_lep - 0.0) / 1.355152   # -0.0%  sj4_pair_mass_max < 115.5 and ak02_3_n_lep > 0
        + 0.0001139018 * max(0.0, 162.7874 - Q.mass_top30) * max(0.0, Q.dc_tag_2nd - 1.067968) / 19.40899   # +0.0%  mass_top30 < 162.8 and dc_tag_2nd > 1.068
    )
    return z


def neuron_84(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.427511e-05
    )
    return z


def neuron_85(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.18552e-07
    )
    return z


def neuron_86(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.709903e-05
    )
    return z


def neuron_87(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.619163e-06
    )
    return z


def neuron_88(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.996992e-07
    )
    return z


def neuron_89(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.735376e-06
    )
    return z


def neuron_90(Q):
    # scale S = 5.437; each line: share * term / its average size
    z = 5.436602 * (-0.01997816
        - 0.08231052 * max(0.0, 0.1275041 - Q.lep_z) / 0.0936134   # -8.2%  lep_z < 0.1275
        + 0.06207969 * max(0.0, 2.0 - Q.lepsj_3_n_d3) * max(0.0, 0.04607888 - Q.lep_dr) / 0.05289158   # +6.2%  lepsj_3_n_d3 < 2 and lep_dr < 0.04608
        + 0.04930045 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +4.9%  lep_iso < 1.362
        + 0.03777193 * max(0.0, 0.07664127 - Q.e2_b2) / 0.04435455   # +3.8%  e2_b2 < 0.07664
        - 0.03739826 * max(0.0, 1.362094 - Q.lep_iso) * max(0.0, 2.885768e-05 - Q.ecf_g43) / 2.185758e-05   # -3.7%  lep_iso < 1.362 and ecf_g43 < 2.886e-05
        - 0.03352336 * max(0.0, 0.004135872 - Q.lep_z) * max(0.0, 0.4381892 - Q.lep_iso) / 0.001028156   # -3.4%  lep_z < 0.004136 and lep_iso < 0.4382
        + 0.03305959 * max(0.0, Q.mres_sd_mass_b0z005 - 76.40585) / 35.53887   # +3.3%  mres_sd_mass_b0z005 > 76.41
        + 0.03295829 * max(0.0, 18.7678 - Q.lep_ptrel) / 14.28156   # +3.3%  lep_ptrel < 18.77
        - 0.03274776 * max(0.0, 1.362094 - Q.lep_iso) * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 2.200342   # -3.3%  lep_iso < 1.362 and lepsj_3_maxsd0 < 2.413
        - 0.02692028 * max(0.0, 0.06573337 - Q.lepsj_2_dr) / 0.04985349   # -2.7%  lepsj_2_dr < 0.06573
        + 0.0257321 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 3.157646 - Q.sip_3d_3) / 7.046859   # +2.6%  lep_ptrel < 18.77 and sip_3d_3 < 3.158
        + 0.02482478 * max(0.0, 7.0 - Q.n_s3d_above_3) / 3.67411   # +2.5%  n_s3d_above_3 < 7
        - 0.02401506 * Q.pz_lnd2 / 0.1223855   # -2.4%  pz_lnd2
        - 0.02400189 * max(0.0, Q.sj3_pairmin_over_m - 0.2477126) / 0.09094898   # -2.4%  sj3_pairmin_over_m > 0.2477
        + 0.02324206 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.sj3_pairmin_over_m - 0.2278414) / 0.02903772   # +2.3%  lep_z < 0.3397 and sj3_pairmin_over_m > 0.2278
        + 0.0232355 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # +2.3%  lep_z < 0.3397
        - 0.02277107 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, 3.157646 - Q.sip_3d_3) / 17.98527   # -2.3%  lep_ptrel < 43.21 and sip_3d_3 < 3.158
        + 0.02117672 * max(0.0, 1.529925 - Q.lepsj_3_maxsd0) / 1.073625   # +2.1%  lepsj_3_maxsd0 < 1.53
        - 0.02094596 * max(0.0, Q.mres_sd_mass_b0z005 - 122.1) / 9.418671   # -2.1%  mres_sd_mass_b0z005 > 122.1
        + 0.01996261 * max(0.0, 128.0079 - Q.sj4_pair_mass_max) / 48.13337   # +2.0%  sj4_pair_mass_max < 128
        - 0.01881062 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 11.53372 - Q.lepsj_3_maxsd0) / 139.2357   # -1.9%  lep_ptrel < 18.77 and lepsj_3_maxsd0 < 11.53
        + 0.01850526 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.06573337 - Q.lepsj_2_dr) / 0.01426412   # +1.9%  lep_z < 0.3397 and lepsj_2_dr < 0.06573
        + 0.01828038 * max(0.0, 0.1122946 - Q.pz_lnd0) * max(0.0, 0.3788785 - Q.z_neutral_had) / 0.004972581   # +1.8%  pz_lnd0 < 0.1123 and z_neutral_had < 0.3789
        - 0.01638613 * max(0.0, 0.0740332 - Q.M2) / 0.009166169   # -1.6%  M2 < 0.07403
        + 0.01478975 * max(0.0, 2.0 - Q.lepsj_3_n_d3) / 1.694207   # +1.5%  lepsj_3_n_d3 < 2
        + 0.01470056 * max(0.0, Q.mres_sd_mass_b0z005 - 141.5589) / 5.429569   # +1.5%  mres_sd_mass_b0z005 > 141.6
        - 0.01394226 * max(0.0, 0.1122946 - Q.pz_lnd0) * max(0.0, 172.888 - Q.jd_3d_4) / 3.438001   # -1.4%  pz_lnd0 < 0.1123 and jd_3d_4 < 172.9
        - 0.01372133 * max(0.0, 11.53372 - Q.lepsj_3_maxsd0) / 9.357067   # -1.4%  lepsj_3_maxsd0 < 11.53
        - 0.01361948 * max(0.0, 0.07664127 - Q.e2_b2) * max(0.0, 0.9926336 - Q.sjq_3_sumabs_k1) / 0.02865038   # -1.4%  e2_b2 < 0.07664 and sjq_3_sumabs_k1 < 0.9926
        - 0.01286791 * max(0.0, Q.mres_sd_mass_b0z005 - 97.29337) / 20.97048   # -1.3%  mres_sd_mass_b0z005 > 97.29
        - 0.01135123 * max(0.0, 0.3261071 - Q.z_displaced3) * max(0.0, 0.04607888 - Q.lep_dr) / 0.006631095   # -1.1%  z_displaced3 < 0.3261 and lep_dr < 0.04608
        + 0.01112443 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +1.1%  lep_ptrel < 43.21
        + 0.0111232 * max(0.0, Q.mres_sd_mass_b0z005 - 91.82593) / 24.37401   # +1.1%  mres_sd_mass_b0z005 > 91.83
        - 0.01080983 * max(0.0, 3.113281 - Q.max_abs_d0) / 1.417031   # -1.1%  max_abs_d0 < 3.113
        - 0.01009513 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 0.890007 - Q.nca_sj4_pairmax_over_mass) / 2.631817   # -1.0%  lep_ptrel < 18.77 and nca_sj4_pairmax_over_mass < 0.89
        + 0.01000149 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # +1.0%  n_pairs_kt_above_1 < 366
        + 0.007746944 * max(0.0, 0.3667049 - Q.N2_b05) / 0.008239872   # +0.8%  N2_b05 < 0.3667
        + 0.006985107 * max(0.0, 0.006470637 - Q.lepsj_2_dr) / 0.004025048   # +0.7%  lepsj_2_dr < 0.006471
        - 0.00665816 * max(0.0, 0.07664127 - Q.e2_b2) * max(0.0, 0.2012988 - Q.kt2_min12_mass_disp3) / 0.008390312   # -0.7%  e2_b2 < 0.07664 and kt2_min12_mass_disp3 < 0.2013
        + 0.006203428 * max(0.0, 11.53372 - Q.lepsj_3_maxsd0) * max(0.0, 0.3539053 - Q.dc_2_z) / 1.368866   # +0.6%  lepsj_3_maxsd0 < 11.53 and dc_2_z < 0.3539
        - 0.005529223 * max(0.0, Q.mass_2charged - 20.76537) / 3.761489   # -0.6%  mass_2charged > 20.77
        + 0.005473111 * max(0.0, 128.0079 - Q.sj4_pair_mass_max) * max(0.0, Q.sum_pt_top20 - 407.2186) / 7610.075   # +0.5%  sj4_pair_mass_max < 128 and sum_pt_top20 > 407.2
        + 0.00450221 * max(0.0, 0.3261071 - Q.z_displaced3) / 0.2272573   # +0.5%  z_displaced3 < 0.3261
        - 0.00381419 * max(0.0, 0.0003489585 - Q.e3) / 3.154871e-05   # -0.4%  e3 < 0.000349
        - 0.003748886 * max(0.0, 0.3603262 - Q.z_charged_had) / 0.02270075   # -0.4%  z_charged_had < 0.3603
        - 0.003639356 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.jet_charge - 0.09659934) / 0.008365432   # -0.4%  lep_z < 0.3397 and jet_charge > 0.0966
        + 0.003530593 * Q.lepsj_2_n_d3 / 0.6197967   # +0.4%  lepsj_2_n_d3
        - 0.003212579 * max(0.0, Q.dc_split2_mass - 32.73553) / 7.572462   # -0.3%  dc_split2_mass > 32.74
        - 0.003192517 * max(0.0, 10.0 - Q.sdb_2_n) / 1.820263   # -0.3%  sdb_2_n < 10
        - 0.002886577 * Q.dc_1_n_lep / 0.2450033   # -0.3%  dc_1_n_lep
        - 0.002792652 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # -0.3%  n_dr_0p4_up < 4
        - 0.00276602 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, Q.mres_sd_rg_b1z01 - 0.2472393) / 1.983668   # -0.3%  lep_ptrel < 18.77 and mres_sd_rg_b1z01 > 0.2472
        - 0.002721235 * max(0.0, Q.n_charged_had - 21.0) / 2.12364   # -0.3%  n_charged_had > 21
        - 0.002651807 * max(0.0, Q.mres_sd_mass_b0z005 - 131.0776) / 7.225802   # -0.3%  mres_sd_mass_b0z005 > 131.1
        + 0.002639561 * max(0.0, -0.6550393 - Q.sjq_3_3_k1) / 0.01225981   # +0.3%  sjq_3_3_k1 < -0.655
        - 0.002481601 * max(0.0, 23.0 - Q.n_pt_above_1) * max(0.0, 0.2748617 - Q.dc_2_z) / 0.09585139   # -0.2%  n_pt_above_1 < 23 and dc_2_z < 0.2749
        - 0.002259015 * Q.z_displaced5 / 0.09296654   # -0.2%  z_displaced5
        + 0.002194551 * max(0.0, 23.0 - Q.n_pt_above_1) / 0.7353867   # +0.2%  n_pt_above_1 < 23
        + 0.002127246 * max(0.0, 0.07664127 - Q.e2_b2) * max(0.0, 4.0 - Q.sd_nremoved) / 0.09721485   # +0.2%  e2_b2 < 0.07664 and sd_nremoved < 4
        + 0.002010095 * max(0.0, Q.sjq_3_3_k1 - 0.6535817) / 0.01230651   # +0.2%  sjq_3_3_k1 > 0.6536
        + 0.001969784 * max(0.0, Q.sjf_3_n2disp - 1.0) * max(0.0, Q.nca_sj4_pair2nd_over_mass - 0.3369906) / 0.03631358   # +0.2%  sjf_3_n2disp > 1 and nca_sj4_pair2nd_over_mass > 0.337
        - 0.00194968 * max(0.0, 0.06074238 - Q.tau1) / 0.001113326   # -0.2%  tau1 < 0.06074
        + 0.001868091 * max(0.0, 10.0 - Q.sdb_2_n) * max(0.0, Q.z_charged - 0.5767344) / 0.1823335   # +0.2%  sdb_2_n < 10 and z_charged > 0.5767
        - 0.001685894 * max(0.0, 0.004135872 - Q.lep_z) / 0.002390373   # -0.2%  lep_z < 0.004136
        + 0.001622751 * max(0.0, 0.3261071 - Q.z_displaced3) * max(0.0, 0.0309457 - Q.M3_b2) / 0.003999011   # +0.2%  z_displaced3 < 0.3261 and M3_b2 < 0.03095
        + 0.001561619 * max(0.0, 115.7429 - Q.mass_top40) / 16.18885   # +0.2%  mass_top40 < 115.7
        + 0.001401 * max(0.0, Q.psi_0p3 - 0.9925964) / 0.0007268581   # +0.1%  psi_0p3 > 0.9926
        + 0.001303492 * max(0.0, 0.06573337 - Q.lepsj_2_dr) * max(0.0, Q.sdb_5_z - 0.0) / 0.001615915   # +0.1%  lepsj_2_dr < 0.06573 and sdb_5_z > 0
        - 0.001240312 * max(0.0, 0.3667049 - Q.N2_b05) * max(0.0, 1.0 - Q.ak02_2_n_lep) / 0.004560542   # -0.1%  N2_b05 < 0.3667 and ak02_2_n_lep < 1
        - 0.00123809 * max(0.0, Q.n_pairs_kt_above_3 - 47.0) / 43.67114   # -0.1%  n_pairs_kt_above_3 > 47
        + 0.001227398 * max(0.0, 0.1122946 - Q.pz_lnd0) / 0.02302722   # +0.1%  pz_lnd0 < 0.1123
        - 0.001178145 * max(0.0, Q.sjf_3_n2disp - 1.0) / 0.2247667   # -0.1%  sjf_3_n2disp > 1
        - 0.001040939 * max(0.0, 128.0079 - Q.sj4_pair_mass_max) * max(0.0, Q.lne_5 - 3.452465) / 12.43289   # -0.1%  sj4_pair_mass_max < 128 and lne_5 > 3.452
        - 0.001009075 * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.3730633   # -0.1%  n_dr_0p1_0p2 < 4
        - 0.0009958697 * max(0.0, 0.06573337 - Q.lepsj_2_dr) * max(0.0, Q.sjq_3_3_k1 - 0.4037488) / 0.00137647   # -0.1%  lepsj_2_dr < 0.06573 and sjq_3_3_k1 > 0.4037
        + 0.0008955628 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # +0.1%  n_pairs_kt_above_1 < 58
        - 0.0008712018 * max(0.0, 0.06573337 - Q.lepsj_2_dr) * max(0.0, Q.mass_displaced3 - 9.090532) / 0.1957631   # -0.1%  lepsj_2_dr < 0.06573 and mass_displaced3 > 9.091
        - 0.0007404504 * max(0.0, Q.mres_sd_mass_b0z005 - 97.29337) * max(0.0, 0.07659457 - Q.tau3) / 0.4039779   # -0.1%  mres_sd_mass_b0z005 > 97.29 and tau3 < 0.07659
        + 0.0006799017 * max(0.0, 0.3265243 - Q.z_charged) / 0.003638213   # +0.1%  z_charged < 0.3265
        - 0.0006549863 * max(0.0, 0.001440167 - Q.sum_z_dr2_top2) / 8.35316e-05   # -0.1%  sum_z_dr2_top2 < 0.00144
        - 0.0006364109 * max(0.0, 0.2077175 - Q.tau21) / 0.008580176   # -0.1%  tau21 < 0.2077
        - 0.0006112887 * max(0.0, Q.mres_sd_mass_b0z005 - 97.29337) * max(0.0, 3.157646 - Q.sip_3d_3) / 4.674094   # -0.1%  mres_sd_mass_b0z005 > 97.29 and sip_3d_3 < 3.158
        + 0.000585177 * max(0.0, Q.mass_2charged - 20.76537) * max(0.0, Q.sv_2_dr - 0.08408739) / 0.247501   # +0.1%  mass_2charged > 20.77 and sv_2_dr > 0.08409
        + 0.0005584973 * max(0.0, 115.7429 - Q.mass_top40) * max(0.0, -0.009407043 - Q.phi_1) / 0.3430248   # +0.1%  mass_top40 < 115.7 and phi_1 < -0.009407
        - 0.0005094846 * max(0.0, Q.mass - 182.8592) / 1.688248   # -0.1%  mass > 182.9
        + 0.0004370853 * max(0.0, 10.0 - Q.sdb_2_n) * max(0.0, Q.dc_3_charge - 0.0) / 0.06552326   # +0.0%  sdb_2_n < 10 and dc_3_charge > 0
        + 0.0003108117 * max(0.0, Q.sum_pt_top10 - 383.2656) / 98.57957   # +0.0%  sum_pt_top10 > 383.3
        + 0.0002741818 * max(0.0, Q.mres_sd_mass_b0z005 - 76.40585) * max(0.0, -0.1329965 - Q.tdz_13) / 0.863148   # +0.0%  mres_sd_mass_b0z005 > 76.41 and tdz_13 < -0.133
        - 0.0001892887 * max(0.0, Q.mass - 164.4374) / 3.038568   # -0.0%  mass > 164.4
        + 0.0001793621 * max(0.0, 0.6359875 - Q.sjf_4_4_maxsd0) / 0.1735084   # +0.0%  sjf_4_4_maxsd0 < 0.636
        - 0.0001716319 * max(0.0, -0.6550393 - Q.sjq_3_3_k1) * max(0.0, 0.1136997 - Q.dc_2_mass_disp3) / 0.001228263   # -0.0%  sjq_3_3_k1 < -0.655 and dc_2_mass_disp3 < 0.1137
        + 0.0001623693 * max(0.0, 0.3603262 - Q.z_charged_had) * max(0.0, 0.0 - Q.td0_25) / 0.0002515501   # +0.0%  z_charged_had < 0.3603 and td0_25 < 0
        - 0.0001289222 * max(0.0, 0.0740332 - Q.M2) * max(0.0, Q.mres_sd_prong_mass2 - 13.86227) / 0.006105662   # -0.0%  M2 < 0.07403 and mres_sd_prong_mass2 > 13.86
        - 0.0001035788 * max(0.0, 0.3667049 - Q.N2_b05) * max(0.0, Q.sjq_2_prod_k03 - 0.1042503) / 0.0006235042   # -0.0%  N2_b05 < 0.3667 and sjq_2_prod_k03 > 0.1043
        + 5.508006e-05 * max(0.0, 0.6359875 - Q.sjf_4_4_maxsd0) * max(0.0, 0.0 - Q.eta_60) / 0.0006301541   # +0.0%  sjf_4_4_maxsd0 < 0.636 and eta_60 < 0
        - 3.575231e-05 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # -0.0%  sip_3d_2 < 447.1
        - 2.927168e-05 * max(0.0, 0.06573337 - Q.lepsj_2_dr) * max(0.0, -0.03515625 - Q.phi_3) / 0.001560354   # -0.0%  lepsj_2_dr < 0.06573 and phi_3 < -0.03516
        + 1.31948e-05 * max(0.0, Q.n_charged_had - 21.0) * max(0.0, Q.dc_2_mass_disp3 - 0.1136997) / 2.975406   # +0.0%  n_charged_had > 21 and dc_2_mass_disp3 > 0.1137
        - 8.217034e-07 * max(0.0, 10.0 - Q.sdb_2_n) * max(0.0, Q.e4_b2 - 7.0969e-08) / 3.452937e-07   # -0.0%  sdb_2_n < 10 and e4_b2 > 7.097e-08
    )
    return z


def neuron_91(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.992637e-06
    )
    return z


def neuron_92(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.724735e-06
    )
    return z


def neuron_93(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.464388e-07
    )
    return z


def neuron_94(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.322172e-06
    )
    return z


def neuron_95(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.369518e-07
    )
    return z


def neuron_96(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.278042e-07
    )
    return z


def neuron_97(Q):
    # scale S = 25.8; each line: share * term / its average size
    z = 25.80128 * (-0.07408164
        - 0.1162286 * max(0.0, Q.mres_sd_mass_b0z005 - 76.40585) / 35.53887   # -11.6%  mres_sd_mass_b0z005 > 76.41
        + 0.08929563 * max(0.0, Q.mres_sd_mass_b0z005 - 91.82593) / 24.37401   # +8.9%  mres_sd_mass_b0z005 > 91.83
        + 0.08303304 * Q.sj3_pairmax_over_m / 0.807958   # +8.3%  sj3_pairmax_over_m
        - 0.06633478 * max(0.0, 171.3819 - Q.mres_pruned_mass) / 78.34654   # -6.6%  mres_pruned_mass < 171.4
        + 0.05847522 * max(0.0, 182.8592 - Q.mass) / 69.61635   # +5.8%  mass < 182.9
        - 0.05821202 * max(0.0, 149.0507 - Q.mass) / 39.08704   # -5.8%  mass < 149.1
        - 0.0556067 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -5.6%  mass < 164.4
        + 0.05449009 * max(0.0, 124.3145 - Q.mres_pruned_mass) / 36.36896   # +5.4%  mres_pruned_mass < 124.3
        + 0.04301147 * max(0.0, Q.mres_sd_mass_b0z005 - 53.57509) / 55.22856   # +4.3%  mres_sd_mass_b0z005 > 53.58
        + 0.02928574 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) / 26.86577   # +2.9%  sj3_pair_mass_max < 114.8
        - 0.02449913 * max(0.0, Q.mres_sd_mass_b0z005 - 97.29337) / 20.97048   # -2.4%  mres_sd_mass_b0z005 > 97.29
        + 0.02304552 * max(0.0, 39.09615 - Q.mass_displaced3) / 31.84697   # +2.3%  mass_displaced3 < 39.1
        - 0.02056564 * max(0.0, 112.1947 - Q.mres_pruned_mass) / 27.53635   # -2.1%  mres_pruned_mass < 112.2
        + 0.01749474 * max(0.0, 9.0 - Q.n_s3d_above_3) / 5.43421   # +1.7%  n_s3d_above_3 < 9
        - 0.01651871 * max(0.0, 6.0 - Q.n_sd0_above_3) / 3.16582   # -1.7%  n_sd0_above_3 < 6
        + 0.01593681 * max(0.0, 126.8853 - Q.mass_top40) / 23.76162   # +1.6%  mass_top40 < 126.9
        + 0.01489198 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # +1.5%  e3_b2 < 0.000791
        - 0.01378749 * max(0.0, Q.mres_pruned_mass - 70.04065) / 33.4731   # -1.4%  mres_pruned_mass > 70.04
        + 0.01247735 * max(0.0, 18.80005 - Q.mass_displaced3) / 13.42744   # +1.2%  mass_displaced3 < 18.8
        - 0.01167758 * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0001737301   # -1.2%  e3_b2 < 0.0002537
        - 0.01146651 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # -1.1%  max_abs_d0 < 5.812
        + 0.01070241 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 0.18807 - Q.pz_lnd3) / 3.368618   # +1.1%  mass_displaced3 < 39.1 and pz_lnd3 < 0.1881
        + 0.009625271 * max(0.0, 5.8125 - Q.max_abs_d0) * max(0.0, 4.004982 - Q.jd_3d_5) / 5.484604   # +1.0%  max_abs_d0 < 5.812 and jd_3d_5 < 4.005
        - 0.009306546 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # -0.9%  sip_3d_2 < 447.1
        - 0.008732014 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 18.7678 - Q.lep_ptrel) / 453.8353   # -0.9%  mass_displaced3 < 39.1 and lep_ptrel < 18.77
        + 0.00831872 * max(0.0, Q.mres_sd_mass_b0z005 - 122.1) / 9.418671   # +0.8%  mres_sd_mass_b0z005 > 122.1
        - 0.007810496 * max(0.0, 172.888 - Q.jd_3d_4) / 152.4729   # -0.8%  jd_3d_4 < 172.9
        - 0.006204226 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -0.6%  lep_z < 0.2214
        + 0.005794568 * max(0.0, 111.0 - Q.n_pairs_kt_above_3) / 47.95244   # +0.6%  n_pairs_kt_above_3 < 111
        + 0.005669482 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) / 103.3282   # +0.6%  n_pairs_kt_above_1 < 325
        + 0.005576319 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +0.6%  mass < 95.15
        + 0.005514004 * max(0.0, 122.0585 - Q.mass_top50) * max(0.0, 1.0 - Q.ak02_2_n_lep) / 16.31653   # +0.6%  mass_top50 < 122.1 and ak02_2_n_lep < 1
        - 0.005201803 * max(0.0, 122.0585 - Q.mass_top50) / 18.91058   # -0.5%  mass_top50 < 122.1
        + 0.004857193 * max(0.0, Q.sd_mass - 119.4443) / 9.070113   # +0.5%  sd_mass > 119.4
        - 0.004670386 * max(0.0, 9.621843e-05 - Q.e3_b2) / 4.881439e-05   # -0.5%  e3_b2 < 9.622e-05
        - 0.00382303 * max(0.0, 182.8592 - Q.mass) * max(0.0, Q.n_s3d_above_3 - 1.0) / 168.7405   # -0.4%  mass < 182.9 and n_s3d_above_3 > 1
        - 0.003491454 * max(0.0, Q.mres_sd_mass_b0z005 - 159.9242) / 3.119054   # -0.3%  mres_sd_mass_b0z005 > 159.9
        + 0.003281169 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, 1.0 - Q.dc_1_n_lep) / 249.927   # +0.3%  sip_3d_2 < 447.1 and dc_1_n_lep < 1
        - 0.003198465 * max(0.0, Q.nca_sj4_pair_mass_2nd - 72.72359) / 3.203001   # -0.3%  nca_sj4_pair_mass_2nd > 72.72
        + 0.002861128 * max(0.0, 100.4835 - Q.mass) / 8.115061   # +0.3%  mass < 100.5
        + 0.002796688 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.jet_charge_k03 - -0.05990128) / 5.503613e-05   # +0.3%  e3_b2 < 0.0002537 and jet_charge_k03 > -0.0599
        + 0.002708991 * max(0.0, 5.0 - Q.sjf_3_1_n_d5) / 3.728713   # +0.3%  sjf_3_1_n_d5 < 5
        - 0.002416186 * max(0.0, 0.2113485 - Q.pz_lnd3) / 0.1202167   # -0.2%  pz_lnd3 < 0.2113
        - 0.002122769 * max(0.0, 122.0585 - Q.mass_top50) * max(0.0, 0.125305 - Q.sjq_2_prod_k05) / 3.922517   # -0.2%  mass_top50 < 122.1 and sjq_2_prod_k05 < 0.1253
        - 0.002103567 * max(0.0, Q.N2_b05 - 0.4353632) / 0.03052074   # -0.2%  N2_b05 > 0.4354
        - 0.002058721 * max(0.0, Q.lep_ptrel - 3.53503) / 6.079822   # -0.2%  lep_ptrel > 3.535
        + 0.001953001 * max(0.0, 100.4835 - Q.mass) * max(0.0, 1.477152 - Q.lep_ptrel) / 8.351593   # +0.2%  mass < 100.5 and lep_ptrel < 1.477
        + 0.001932436 * max(0.0, Q.nca_sj4_pair_mass_2nd - 83.41384) / 1.649755   # +0.2%  nca_sj4_pair_mass_2nd > 83.41
        + 0.001929517 * max(0.0, 7.746064 - Q.ak02_min12_jp) / 4.717722   # +0.2%  ak02_min12_jp < 7.746
        + 0.001927271 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 4.0 - Q.lepsj_2_n_d3) / 0.6051566   # +0.2%  lep_z < 0.2214 and lepsj_2_n_d3 < 4
        - 0.001557482 * max(0.0, 0.2740506 - Q.sdb_2_z) / 0.05180874   # -0.2%  sdb_2_z < 0.2741
        + 0.001545321 * max(0.0, Q.lund3_lndelta - -1.780944) / 0.4380167   # +0.2%  lund3_lndelta > -1.781
        - 0.001503263 * max(0.0, Q.mres_pruned_mass - 70.04065) * max(0.0, Q.lnerel_4 - -3.643775) / 21.7593   # -0.2%  mres_pruned_mass > 70.04 and lnerel_4 > -3.644
        - 0.00143226 * max(0.0, 100.4835 - Q.mass) * max(0.0, 6.185635 - Q.lep_iso) / 42.98093   # -0.1%  mass < 100.5 and lep_iso < 6.186
        - 0.001359535 * max(0.0, 71.80762 - Q.mass_top50) / 1.927765   # -0.1%  mass_top50 < 71.81
        + 0.001172509 * max(0.0, 7.075642 - Q.pair_max_lnm2) / 0.6225895   # +0.1%  pair_max_lnm2 < 7.076
        - 0.001066524 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) * max(0.0, Q.n_photon - 12.0) / 94.74492   # -0.1%  sj3_pair_mass_max < 114.8 and n_photon > 12
        + 0.001060303 * max(0.0, 0.2740506 - Q.sdb_2_z) * max(0.0, 0.1348361 - Q.pz_lnkt1) / 0.002128601   # +0.1%  sdb_2_z < 0.2741 and pz_lnkt1 < 0.1348
        - 0.0009824843 * max(0.0, 3.734077 - Q.lund_max_lnkt) / 0.2313255   # -0.1%  lund_max_lnkt < 3.734
        + 0.000947131 * max(0.0, 0.6513932 - Q.tau32) / 0.06839918   # +0.1%  tau32 < 0.6514
        - 0.0009116294 * max(0.0, Q.lnerel_42 - -6.337363) / 0.25676   # -0.1%  lnerel_42 > -6.337
        - 0.0009093888 * max(0.0, 0.103005 - Q.lepsj_2_dr) / 0.08254947   # -0.1%  lepsj_2_dr < 0.103
        + 0.0008982462 * max(0.0, Q.N2 - 0.2052214) / 0.1062466   # +0.1%  N2 > 0.2052
        + 0.0007701552 * max(0.0, Q.lund3_lndelta - -1.780944) * max(0.0, Q.z_photon - 0.2047275) / 0.03411439   # +0.1%  lund3_lndelta > -1.781 and z_photon > 0.2047
        - 0.0007484376 * max(0.0, 0.6513932 - Q.tau32) * max(0.0, 3.151933 - Q.jd_3d_6) / 0.0733668   # -0.1%  tau32 < 0.6514 and jd_3d_6 < 3.152
        + 0.0007361525 * max(0.0, 63.7508 - Q.sj3_pair_mass_max) / 2.405567   # +0.1%  sj3_pair_mass_max < 63.75
        + 0.0007314442 * max(0.0, 122.0585 - Q.mass_top50) * max(0.0, Q.sum_e - 926.2598) / 2366.251   # +0.1%  mass_top50 < 122.1 and sum_e > 926.3
        + 0.0006923356 * max(0.0, Q.mass_charged - 62.00562) / 11.67561   # +0.1%  mass_charged > 62.01
        - 0.0006348151 * max(0.0, 182.8592 - Q.mass) * max(0.0, Q.dc_ntag - 0.0) / 16.94701   # -0.1%  mass < 182.9 and dc_ntag > 0
        + 0.0005898613 * max(0.0, 7.746064 - Q.ak02_min12_jp) * max(0.0, Q.mass_2photon - 0.2753928) / 24.14226   # +0.1%  ak02_min12_jp < 7.746 and mass_2photon > 0.2754
        - 0.0005328717 * max(0.0, Q.mres_sd_prong_mass1 - 69.04524) / 2.206937   # -0.1%  mres_sd_prong_mass1 > 69.05
        - 0.000526243 * max(0.0, 65.88119 - Q.nca_sj4_pair_mass_2nd) * max(0.0, Q.nca_kt_above_10 - 1.0) / 5.220674   # -0.1%  nca_sj4_pair_mass_2nd < 65.88 and nca_kt_above_10 > 1
        - 0.0005090358 * max(0.0, Q.pair_mean_lnm2 - 2.09826) / 0.7077607   # -0.1%  pair_mean_lnm2 > 2.098
        - 0.0005054092 * max(0.0, Q.z_displaced5 - 0.1339658) / 0.03671738   # -0.1%  z_displaced5 > 0.134
        + 0.0004813874 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, 0.3709098 - Q.sdb_2_z) / 1.803054e-05   # +0.0%  e3_b2 < 0.0002537 and sdb_2_z < 0.3709
        + 0.0004258507 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.D2 - 3.038341) / 11.93622   # +0.0%  mass_displaced3 < 39.1 and D2 > 3.038
        - 0.0003967341 * max(0.0, Q.sjf_2_1_n_d3 - 5.0) / 0.1972067   # -0.0%  sjf_2_1_n_d3 > 5
        - 0.0003931392 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 1.761851   # -0.0%  lepsj_3_maxsd0 < 2.413
        - 0.0003929644 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.sum_pt_top10 - 324.6547) / 4846.941   # -0.0%  mass_displaced3 < 39.1 and sum_pt_top10 > 324.7
        + 0.000385061 * max(0.0, 8.0 - Q.n_photon) / 0.3050067   # +0.0%  n_photon < 8
        - 0.0003631708 * max(0.0, 124.3145 - Q.mres_pruned_mass) * max(0.0, 0.09642216 - Q.ak02_2_charge) / 8.762255   # -0.0%  mres_pruned_mass < 124.3 and ak02_2_charge < 0.09642
        + 0.0003257059 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # +0.0%  sj2_mass2 < 1.852
        + 0.0002718904 * max(0.0, 0.0007909605 - Q.e3_b2) * max(0.0, Q.sjq_3_sumabs_k1 - 0.4372817) / 6.220365e-05   # +0.0%  e3_b2 < 0.000791 and sjq_3_sumabs_k1 > 0.4373
        - 0.0002551905 * max(0.0, Q.pair_mean_lnm2 - 2.09826) * max(0.0, 0.171574 - Q.dc_3_z) / 0.08287162   # -0.0%  pair_mean_lnm2 > 2.098 and dc_3_z < 0.1716
        - 0.0002029391 * max(0.0, 65.88119 - Q.nca_sj4_pair_mass_2nd) / 13.7977   # -0.0%  nca_sj4_pair_mass_2nd < 65.88
        - 0.0001828506 * max(0.0, 126.8853 - Q.mass_top40) * max(0.0, Q.n_muon - 1.0) / 0.4724669   # -0.0%  mass_top40 < 126.9 and n_muon > 1
        + 0.0001628139 * max(0.0, 1.365419e-05 - Q.e3_b2) / 2.488876e-06   # +0.0%  e3_b2 < 1.365e-05
        - 0.0001161041 * max(0.0, Q.sj3_pair_mass_min - 68.11898) * max(0.0, 0.08408739 - Q.sv_2_dr) / 0.0623899   # -0.0%  sj3_pair_mass_min > 68.12 and sv_2_dr < 0.08409
        + 9.349036e-05 * max(0.0, Q.dc_split2_dr - 0.3591078) / 0.004116954   # +0.0%  dc_split2_dr > 0.3591
        + 6.98939e-05 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.jd_3d_6 - 16.11752) / 225.9739   # +0.0%  mass < 164.4 and jd_3d_6 > 16.12
        + 5.821045e-05 * max(0.0, 1.851735 - Q.sj2_mass2) * max(0.0, Q.mass_2photon - 1.504865) / 0.1404703   # +0.0%  sj2_mass2 < 1.852 and mass_2photon > 1.505
        + 5.257145e-05 * max(0.0, 182.8592 - Q.mass) * max(0.0, Q.ak02_3_n_lep - 0.0) / 2.256165   # +0.0%  mass < 182.9 and ak02_3_n_lep > 0
        + 3.501775e-05 * max(0.0, 5.0 - Q.sjf_3_1_n_d5) * max(0.0, Q.dzerr_37 - 0.06842041) / 0.04290707   # +0.0%  sjf_3_1_n_d5 < 5 and dzerr_37 > 0.06842
        + 3.298639e-05 * max(0.0, 0.0007909605 - Q.e3_b2) * max(0.0, Q.dr_28 - 0.3368505) / 1.879537e-05   # +0.0%  e3_b2 < 0.000791 and dr_28 > 0.3369
        + 2.747191e-05 * max(0.0, Q.lep_ptrel - 3.53503) * max(0.0, Q.mass_2photon - 0.4121793) / 17.80228   # +0.0%  lep_ptrel > 3.535 and mass_2photon > 0.4122
        + 1.119609e-05 * max(0.0, 0.135772 - Q.tau21) / 0.001647604   # +0.0%  tau21 < 0.1358
        - 9.740631e-06 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.kt2_min12_mass_disp3 - 0.0) / 0.01344401   # -0.0%  lep_z < 0.2214 and kt2_min12_mass_disp3 > 0
        + 5.039451e-06 * max(0.0, Q.sj3_pair_mass_min - 68.11898) / 1.584065   # +0.0%  sj3_pair_mass_min > 68.12
        + 3.172117e-06 * max(0.0, Q.sj3_pair_mass_min - 68.11898) * max(0.0, Q.e4_b2 - 7.0969e-08) / 1.309508e-06   # +0.0%  sj3_pair_mass_min > 68.12 and e4_b2 > 7.097e-08
    )
    return z


def neuron_98(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.990171e-07
    )
    return z


def neuron_99(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.001934241
    )
    return z


def neuron_100(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.264077e-06
    )
    return z


def neuron_101(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.878262e-06
    )
    return z


def neuron_102(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.490515e-06
    )
    return z


def neuron_103(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.414379e-06
    )
    return z


def neuron_104(Q):
    # scale S = 7.033; each line: share * term / its average size
    z = 7.032548 * (-0.2082113
        + 0.09598365 * max(0.0, Q.mres_sd_mass_b2z01 - 76.1529) / 35.12032   # +9.6%  mres_sd_mass_b2z01 > 76.15
        - 0.06969087 * max(0.0, Q.mres_sd_mass_b2z01 - 86.09683) / 27.65834   # -7.0%  mres_sd_mass_b2z01 > 86.1
        + 0.06687602 * max(0.0, 58.0 - Q.n_pt_above_1) / 20.6963   # +6.7%  n_pt_above_1 < 58
        + 0.06061967 * max(0.0, -1.352792 - Q.pair_mean_lndelta) / 0.8695512   # +6.1%  pair_mean_lndelta < -1.353
        + 0.04814139 * max(0.0, Q.mass_displaced3 - 3.208089) / 6.234439   # +4.8%  mass_displaced3 > 3.208
        + 0.0471355 * max(0.0, 117.4867 - Q.mass) / 15.64803   # +4.7%  mass < 117.5
        + 0.04150875 * max(0.0, 0.2827395 - Q.pz_lnd0) / 0.1575044   # +4.2%  pz_lnd0 < 0.2827
        + 0.03053121 * max(0.0, 149.0507 - Q.mass) / 39.08704   # +3.1%  mass < 149.1
        - 0.03013848 * max(0.0, Q.mass_displaced3 - 6.341631) / 5.164503   # -3.0%  mass_displaced3 > 6.342
        + 0.0278599 * max(0.0, 6.0 - Q.n_s3d_above_10) / 3.631173   # +2.8%  n_s3d_above_10 < 6
        - 0.02301266 * max(0.0, 149.0507 - Q.mass) * max(0.0, 1401.904 - Q.sum_e) / 20921.9   # -2.3%  mass < 149.1 and sum_e < 1402
        + 0.02058263 * max(0.0, 18.0 - Q.sjq_2_1_nch) / 6.63145   # +2.1%  sjq_2_1_nch < 18
        + 0.01951271 * max(0.0, 58.0 - Q.n_pt_above_1) * max(0.0, 0.4490565 - Q.ak02_dr13) / 6.760883   # +2.0%  n_pt_above_1 < 58 and ak02_dr13 < 0.4491
        - 0.01792849 * max(0.0, 1.914834 - Q.dc_tag_2nd) / 1.481015   # -1.8%  dc_tag_2nd < 1.915
        - 0.01749734 * max(0.0, 14.23948 - Q.ak02_min12_jp) / 10.12039   # -1.7%  ak02_min12_jp < 14.24
        - 0.0173657 * max(0.0, 0.05398263 - Q.tau3) / 0.0141203   # -1.7%  tau3 < 0.05398
        - 0.01709503 * max(0.0, Q.mres_sd_mass_b0z005 - 91.82593) / 24.37401   # -1.7%  mres_sd_mass_b0z005 > 91.83
        + 0.01678268 * max(0.0, 99.20396 - Q.mass_charged) / 37.01153   # +1.7%  mass_charged < 99.2
        + 0.01600505 * max(0.0, 15.0 - Q.sjq_2_2_nch) / 8.033983   # +1.6%  sjq_2_2_nch < 15
        + 0.01305043 * max(0.0, 0.0230226 - Q.tau5) / 0.00273951   # +1.3%  tau5 < 0.02302
        - 0.01150519 * max(0.0, 5.245219 - Q.sjf_3_2_max3d) / 1.779833   # -1.2%  sjf_3_2_max3d < 5.245
        + 0.01145369 * max(0.0, Q.pair_max_lnm2 - 5.908788) / 0.8228665   # +1.1%  pair_max_lnm2 > 5.909
        + 0.0113342 * max(0.0, 58.0 - Q.n_pt_above_1) * max(0.0, 1.0 - Q.lepsj_2_n_d3) / 16.6829   # +1.1%  n_pt_above_1 < 58 and lepsj_2_n_d3 < 1
        + 0.01126454 * max(0.0, Q.z_top3_slots - 0.5395924) / 0.0406864   # +1.1%  z_top3_slots > 0.5396
        + 0.01090397 * max(0.0, 61.73838 - Q.sjf_3_2_maxsd0) / 41.35113   # +1.1%  sjf_3_2_maxsd0 < 61.74
        - 0.0108852 * max(0.0, 58.0 - Q.n_pt_above_1) * max(0.0, Q.pair_max_lnkt - 1.555555) / 24.2913   # -1.1%  n_pt_above_1 < 58 and pair_max_lnkt > 1.556
        - 0.01087754 * max(0.0, 99.20396 - Q.mass_charged) * max(0.0, 1.0 - Q.ak02_2_n_lep) / 31.82896   # -1.1%  mass_charged < 99.2 and ak02_2_n_lep < 1
        - 0.01032957 * max(0.0, 4.516968 - Q.sjf_2_2_max3d) * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0008440434   # -1.0%  sjf_2_2_max3d < 4.517 and e3_b2 < 0.000791
        + 0.01025423 * max(0.0, 135.5368 - Q.mass_top50) / 28.82074   # +1.0%  mass_top50 < 135.5
        - 0.01010546 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # -1.0%  n_pairs_kt_above_3 < 34
        - 0.009623705 * max(0.0, Q.mass_top5 - 17.63354) / 26.12147   # -1.0%  mass_top5 > 17.63
        + 0.00873382 * max(0.0, Q.lep_ptrel - 6.983043) / 5.140178   # +0.9%  lep_ptrel > 6.983
        - 0.008343193 * max(0.0, 58.0 - Q.n_pt_above_1) * max(0.0, Q.z_neutral - 0.1384639) / 5.157668   # -0.8%  n_pt_above_1 < 58 and z_neutral > 0.1385
        - 0.008332157 * max(0.0, 4.0 - Q.sjf_3_1_n_d3) / 2.66867   # -0.8%  sjf_3_1_n_d3 < 4
        + 0.008297439 * max(0.0, 16.96512 - Q.sjf_2_2_max3d) / 8.191057   # +0.8%  sjf_2_2_max3d < 16.97
        - 0.008286352 * max(0.0, Q.lep_ptrel - 6.983043) * max(0.0, 0.2385164 - Q.ak02_dr23) / 0.7252318   # -0.8%  lep_ptrel > 6.983 and ak02_dr23 < 0.2385
        + 0.008146068 * max(0.0, 1.546763 - Q.lepsj_2_maxsd0) / 1.012326   # +0.8%  lepsj_2_maxsd0 < 1.547
        + 0.007296249 * max(0.0, 0.1798521 - Q.C2) / 0.04587768   # +0.7%  C2 < 0.1799
        + 0.006014495 * max(0.0, Q.tau21_b2 - 0.3898586) / 0.04953334   # +0.6%  tau21_b2 > 0.3899
        - 0.005529258 * max(0.0, 0.2827395 - Q.pz_lnd0) * max(0.0, Q.n_pairs_kt_above_10 - 1.0) / 1.06713   # -0.6%  pz_lnd0 < 0.2827 and n_pairs_kt_above_10 > 1
        - 0.00547932 * max(0.0, 1.546763 - Q.lepsj_2_maxsd0) * max(0.0, 0.0008210142 - Q.lepsj_2_dr) / 0.0007561966   # -0.5%  lepsj_2_maxsd0 < 1.547 and lepsj_2_dr < 0.000821
        - 0.005380958 * max(0.0, 0.4680886 - Q.tau32_b2) / 0.05978538   # -0.5%  tau32_b2 < 0.4681
        - 0.005145906 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -0.5%  sip_3d_2 < 226.3
        - 0.004924703 * max(0.0, 3.32421 - Q.sjf_2_1_max3d) / 0.386295   # -0.5%  sjf_2_1_max3d < 3.324
        - 0.00461048 * max(0.0, 0.312717 - Q.N2) * max(0.0, 4.662928e-05 - Q.e3_b2) / 8.831951e-07   # -0.5%  N2 < 0.3127 and e3_b2 < 4.663e-05
        + 0.004591269 * max(0.0, Q.mres_sd_mass_b0z005 - 91.82593) * max(0.0, 3.0 - Q.sv_1_n) / 40.5646   # +0.5%  mres_sd_mass_b0z005 > 91.83 and sv_1_n < 3
        - 0.00436563 * max(0.0, 0.4501484 - Q.z_charged_had) / 0.04690479   # -0.4%  z_charged_had < 0.4501
        + 0.00436439 * max(0.0, 14.23948 - Q.ak02_min12_jp) * max(0.0, Q.pz_lnkt1 - 0.06778673) / 0.3775963   # +0.4%  ak02_min12_jp < 14.24 and pz_lnkt1 > 0.06779
        - 0.004218827 * max(0.0, 58.0 - Q.n_pt_above_1) * max(0.0, Q.sjf_2_1_n_d5 - 1.0) / 19.38064   # -0.4%  n_pt_above_1 < 58 and sjf_2_1_n_d5 > 1
        + 0.00412271 * max(0.0, Q.D2 - 3.597891) / 0.2539167   # +0.4%  D2 > 3.598
        + 0.003796073 * max(0.0, Q.mass_displaced3 - 3.208089) * max(0.0, 0.1014193 - Q.ak02_3_z) / 0.4174877   # +0.4%  mass_displaced3 > 3.208 and ak02_3_z < 0.1014
        - 0.003725183 * max(0.0, 146.0 - Q.n_pairs_kt_above_1) / 20.10573   # -0.4%  n_pairs_kt_above_1 < 146
        + 0.003449219 * max(0.0, 0.312717 - Q.N2) / 0.03999115   # +0.3%  N2 < 0.3127
        - 0.00340569 * max(0.0, 0.1040701 - Q.sv_1_dr) / 0.03940747   # -0.3%  sv_1_dr < 0.1041
        + 0.003262032 * max(0.0, 18.0 - Q.sjq_2_1_nch) * max(0.0, 0.3258728 - Q.dr01) / 1.262314   # +0.3%  sjq_2_1_nch < 18 and dr01 < 0.3259
        + 0.00305548 * max(0.0, Q.mass_top30 - 126.5007) / 5.381458   # +0.3%  mass_top30 > 126.5
        + 0.002724712 * max(0.0, 2.0 - Q.sjq_2_2_nch) / 0.09739333   # +0.3%  sjq_2_2_nch < 2
        - 0.002654985 * max(0.0, Q.sjq_2_sumabs_k03 - 0.6667228) / 0.2276103   # -0.3%  sjq_2_sumabs_k03 > 0.6667
        - 0.002476577 * max(0.0, 18.0 - Q.sjq_2_1_nch) * max(0.0, Q.dr02 - 0.04115773) / 0.9302163   # -0.2%  sjq_2_1_nch < 18 and dr02 > 0.04116
        + 0.002196392 * max(0.0, 0.2234678 - Q.C2_b05) / 0.00926781   # +0.2%  C2_b05 < 0.2235
        + 0.002175406 * max(0.0, Q.nca_sj4_pair_mass_2nd - 77.3635) / 2.396009   # +0.2%  nca_sj4_pair_mass_2nd > 77.36
        + 0.002149921 * max(0.0, 0.3988697 - Q.sdb_2_z) / 0.124307   # +0.2%  sdb_2_z < 0.3989
        + 0.002100345 * max(0.0, 8.0 - Q.sdb_2_n) / 0.95158   # +0.2%  sdb_2_n < 8
        - 0.001978399 * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 1.107657   # -0.2%  n_lund_kt_above_5 > 1
        - 0.001966704 * max(0.0, 149.0507 - Q.mass) * max(0.0, 0.3265797 - Q.tau21_b2) / 3.621553   # -0.2%  mass < 149.1 and tau21_b2 < 0.3266
        + 0.001905941 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, 1.0 - Q.kt2_2_n_lep) / 3.68679   # +0.2%  n_pairs_kt_above_3 < 34 and kt2_2_n_lep < 1
        + 0.001890873 * max(0.0, 0.8080863 - Q.sjf_4_4_maxsd0) / 0.2381004   # +0.2%  sjf_4_4_maxsd0 < 0.8081
        + 0.001821184 * max(0.0, Q.mres_sd_mass_b0z005 - 159.9242) / 3.119054   # +0.2%  mres_sd_mass_b0z005 > 159.9
        - 0.001713146 * max(0.0, Q.sj2_mass1 - 53.51926) / 6.081546   # -0.2%  sj2_mass1 > 53.52
        + 0.001510621 * max(0.0, 3.0 - Q.n_sdz_above_5) / 1.349857   # +0.2%  n_sdz_above_5 < 3
        + 0.001454143 * max(0.0, Q.pz_lnd2 - 0.1811493) / 0.01345995   # +0.1%  pz_lnd2 > 0.1811
        + 0.001259164 * max(0.0, Q.sj4_pair_mass_max - 128.0079) / 1.174806   # +0.1%  sj4_pair_mass_max > 128
        + 0.001245963 * max(0.0, Q.sjf_2_1_z_d3 - 0.1149585) * max(0.0, Q.ischhad_0 - 0.0) / 0.02018831   # +0.1%  sjf_2_1_z_d3 > 0.115 and ischhad_0 > 0
        + 0.001228679 * max(0.0, Q.lep_ptrel - 6.983043) * max(0.0, 28.89659 - Q.mass_2charged) / 19.1244   # +0.1%  lep_ptrel > 6.983 and mass_2charged < 28.9
        + 0.001072613 * max(0.0, 1.789401 - Q.jd_3d_5) / 0.1199535   # +0.1%  jd_3d_5 < 1.789
        + 0.0009501233 * max(0.0, Q.lund_max_lndelta - -0.529318) / 0.01270035   # +0.1%  lund_max_lndelta > -0.5293
        + 0.000924491 * max(0.0, 5.245219 - Q.sjf_3_2_max3d) * max(0.0, Q.jet_charge - 0.09659934) / 0.09332462   # +0.1%  sjf_3_2_max3d < 5.245 and jet_charge > 0.0966
        - 0.0009243509 * max(0.0, 18.0 - Q.sjq_2_1_nch) * max(0.0, Q.sjf_2_2_n_d5 - 1.0) / 4.496593   # -0.1%  sjq_2_1_nch < 18 and sjf_2_2_n_d5 > 1
        - 0.000842277 * max(0.0, Q.sjf_2_1_z_d3 - 0.1149585) * max(0.0, 0.09802954 - Q.sjq_2_prod_k1) / 0.004869423   # -0.1%  sjf_2_1_z_d3 > 0.115 and sjq_2_prod_k1 < 0.09803
        + 0.0008098487 * max(0.0, 0.01637527 - Q.pz_lnd2) / 0.0007696198   # +0.1%  pz_lnd2 < 0.01638
        + 0.000777593 * max(0.0, 1.226724 - Q.D2) / 0.07974157   # +0.1%  D2 < 1.227
        - 0.0007388778 * max(0.0, Q.sjf_2_1_z_d3 - 0.1149585) / 0.03815928   # -0.1%  sjf_2_1_z_d3 > 0.115
        + 0.00066733 * max(0.0, 1.0 - Q.sjq_3_2_nch) / 0.02006667   # +0.1%  sjq_3_2_nch < 1
        + 0.0006158602 * max(0.0, 0.02312549 - Q.sjf_3_1_z_d3) * max(0.0, 0.08897484 - Q.dr_19) / 0.0001881455   # +0.1%  sjf_3_1_z_d3 < 0.02313 and dr_19 < 0.08897
        - 0.000585574 * max(0.0, 6.216954e-05 - Q.ecf_g41) / 1.118908e-06   # -0.1%  ecf_g41 < 6.217e-05
        - 0.0005769646 * max(0.0, 149.0507 - Q.mass) * max(0.0, Q.lnerel_65 - -18.42068) / 10.15003   # -0.1%  mass < 149.1 and lnerel_65 > -18.42
        - 0.0005599222 * max(0.0, Q.sjf_2_1_z_d3 - 0.1149585) * max(0.0, 0.9542229 - Q.sjq_2_sumabs_k1) / 0.02212272   # -0.1%  sjf_2_1_z_d3 > 0.115 and sjq_2_sumabs_k1 < 0.9542
        - 0.0004392176 * max(0.0, 5.245219 - Q.sjf_3_2_max3d) * max(0.0, 0.325219 - Q.mres_sd_zg_b1z01) / 0.2261735   # -0.0%  sjf_3_2_max3d < 5.245 and mres_sd_zg_b1z01 < 0.3252
        - 0.0004079651 * max(0.0, 4.516968 - Q.sjf_2_2_max3d) / 1.242753   # -0.0%  sjf_2_2_max3d < 4.517
        + 0.0004036443 * max(0.0, 0.02312549 - Q.sjf_3_1_z_d3) / 0.01242051   # +0.0%  sjf_3_1_z_d3 < 0.02313
        + 0.0003076067 * max(0.0, Q.mass_top5 - 17.63354) * max(0.0, Q.eta_1 - -0.0289917) / 1.739555   # +0.0%  mass_top5 > 17.63 and eta_1 > -0.02899
        - 0.0002704632 * max(0.0, 0.1010123 - Q.z_neutral_had) / 0.02050406   # -0.0%  z_neutral_had < 0.101
        - 0.0002410186 * max(0.0, Q.mass_top5 - 17.63354) * max(0.0, Q.lnpt_78 - -18.42068) / 4.441788   # -0.0%  mass_top5 > 17.63 and lnpt_78 > -18.42
        - 0.0001869085 * max(0.0, Q.D3 - 1.263537) / 0.04007117   # -0.0%  D3 > 1.264
        - 0.000168254 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.iselectron_1 - 0.0) / 0.2092567   # -0.0%  n_pairs_kt_above_3 < 34 and iselectron_1 > 0
        - 0.0001322547 * max(0.0, 0.3988697 - Q.sdb_2_z) * max(0.0, Q.e3 - 0.001381331) / 6.243017e-05   # -0.0%  sdb_2_z < 0.3989 and e3 > 0.001381
        + 6.695705e-05 * max(0.0, 1.546763 - Q.lepsj_2_maxsd0) * max(0.0, Q.eta_4 - 0.104187) / 0.01320538   # +0.0%  lepsj_2_maxsd0 < 1.547 and eta_4 > 0.1042
        - 6.100917e-05 * max(0.0, Q.tau21_b2 - 0.3898586) * max(0.0, 0.09854228 - Q.dr_50) / 0.00367081   # -0.0%  tau21_b2 > 0.3899 and dr_50 < 0.09854
        - 4.868234e-05 * max(0.0, Q.sj4_pair_mass_max - 128.0079) * max(0.0, Q.sv_2_sd0_sum - 1490.247) / 170.984   # -0.0%  sj4_pair_mass_max > 128 and sv_2_sd0_sum > 1490
        + 3.466825e-05 * max(0.0, 18.0 - Q.sjq_2_1_nch) * max(0.0, Q.dzerr_60 - 0.0) / 0.006278939   # +0.0%  sjq_2_1_nch < 18 and dzerr_60 > 0
    )
    return z


def neuron_105(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.732383e-05
    )
    return z


def neuron_106(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.160935e-05
    )
    return z


def neuron_107(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.15856e-05
    )
    return z


def neuron_108(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.470702e-06
    )
    return z


def neuron_109(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.200158e-07
    )
    return z


def neuron_110(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.579609e-06
    )
    return z


def neuron_111(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.669617e-06
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
    # scale S = 12.6; each line: share * term / its average size
    z = 12.60397 * (0.115241
        - 0.0831814 * Q.sj3_pairmax_over_m / 0.807958   # -8.3%  sj3_pairmax_over_m
        - 0.07252796 * max(0.0, 39.09615 - Q.mass_displaced3) / 31.84697   # -7.3%  mass_displaced3 < 39.1
        + 0.04964714 * max(0.0, 0.5187302 - Q.lep_z) / 0.4405854   # +5.0%  lep_z < 0.5187
        + 0.04559523 * max(0.0, 131.3917 - Q.mass) / 24.784   # +4.6%  mass < 131.4
        - 0.04266402 * max(0.0, 0.5626523 - Q.LHA) / 0.1600511   # -4.3%  LHA < 0.5627
        - 0.03901203 * max(0.0, 18.7678 - Q.lep_ptrel) / 14.28156   # -3.9%  lep_ptrel < 18.77
        + 0.03597467 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 18.7678 - Q.lep_ptrel) / 453.8353   # +3.6%  mass_displaced3 < 39.1 and lep_ptrel < 18.77
        + 0.03524413 * Q.z_neutral_had / 0.1513027   # +3.5%  z_neutral_had
        + 0.03368542 * max(0.0, 0.120439 - Q.M2) / 0.04313262   # +3.4%  M2 < 0.1204
        - 0.03201237 * max(0.0, 161.1264 - Q.mass_top50) / 50.68933   # -3.2%  mass_top50 < 161.1
        - 0.03054371 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) / 26.86577   # -3.1%  sj3_pair_mass_max < 114.8
        + 0.02875448 * max(0.0, 0.9367772 - Q.pair_mean_lnkt) / 0.4427064   # +2.9%  pair_mean_lnkt < 0.9368
        + 0.02197483 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # +2.2%  n_lepton < 1
        + 0.02190436 * max(0.0, 0.00228569 - Q.e3) / 0.001253813   # +2.2%  e3 < 0.002286
        + 0.02036889 * max(0.0, 0.09733903 - Q.tau2) / 0.03188557   # +2.0%  tau2 < 0.09734
        - 0.01989103 * max(0.0, Q.pair_max_lnkt - 1.555555) / 1.17245   # -2.0%  pair_max_lnkt > 1.556
        - 0.01510744 * max(0.0, 101.849 - Q.sj4_pair_mass_max) * max(0.0, 9.356714 - Q.jd_3d_6) / 169.6293   # -1.5%  sj4_pair_mass_max < 101.8 and jd_3d_6 < 9.357
        + 0.01506375 * max(0.0, 6.767937 - Q.mass_2charged) / 3.01163   # +1.5%  mass_2charged < 6.768
        + 0.01504236 * max(0.0, 0.2243273 - Q.N2_b2) / 0.06973244   # +1.5%  N2_b2 < 0.2243
        - 0.01409992 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # -1.4%  e3_b2 < 0.0004127
        + 0.01324839 * max(0.0, 21.04127 - Q.mres_sd_prong_mass2) / 11.57923   # +1.3%  mres_sd_prong_mass2 < 21.04
        + 0.01283257 * max(0.0, 118.8408 - Q.mass_top50) * max(0.0, 9.356714 - Q.jd_3d_6) / 120.1591   # +1.3%  mass_top50 < 118.8 and jd_3d_6 < 9.357
        - 0.01240509 * max(0.0, 0.932165 - Q.dc_1_z) / 0.3086674   # -1.2%  dc_1_z < 0.9322
        - 0.01201513 * max(0.0, 3.0 - Q.lepsj_3_n_d3) / 2.624773   # -1.2%  lepsj_3_n_d3 < 3
        - 0.0115393 * max(0.0, 0.8939856 - Q.tau43) / 0.102551   # -1.2%  tau43 < 0.894
        - 0.01118843 * max(0.0, Q.n_sd0_above_3 - 2.0) / 1.826197   # -1.1%  n_sd0_above_3 > 2
        - 0.01027708 * max(0.0, 20.76537 - Q.mass_2charged) / 12.0384   # -1.0%  mass_2charged < 20.77
        - 0.009562078 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # -1.0%  e3_b2 < 0.000791
        - 0.00904406 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # -0.9%  lep_iso < 0.4382
        + 0.008988817 * max(0.0, 75.78208 - Q.sj4_pair_mass_max) / 8.111726   # +0.9%  sj4_pair_mass_max < 75.78
        + 0.008935554 * max(0.0, Q.n_s3d_above_3 - 4.0) / 1.224997   # +0.9%  n_s3d_above_3 > 4
        + 0.008826664 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # +0.9%  sip_3d_2 < 447.1
        - 0.008811512 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, 9.982976 - Q.jd_3d_4) / 1917.09   # -0.9%  sip_3d_2 < 447.1 and jd_3d_4 < 9.983
        - 0.008273955 * max(0.0, 101.849 - Q.sj4_pair_mass_max) / 24.8879   # -0.8%  sj4_pair_mass_max < 101.8
        - 0.008229906 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) * max(0.0, 509.6197 - Q.sv_1_sd0_sum) / 693.4354   # -0.8%  lepsj_3_maxsd0 < 2.413 and sv_1_sd0_sum < 509.6
        + 0.006720696 * max(0.0, 428.9828 - Q.sum_pt_top10) / 28.27138   # +0.7%  sum_pt_top10 < 429
        - 0.006522662 * max(0.0, 172.888 - Q.jd_3d_4) / 152.4729   # -0.7%  jd_3d_4 < 172.9
        - 0.006474238 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # -0.6%  n_pairs_kt_above_1 < 80
        - 0.006305375 * max(0.0, 0.02745856 - Q.sum_z_dr2_top3) / 0.01202135   # -0.6%  sum_z_dr2_top3 < 0.02746
        + 0.006125819 * max(0.0, 118.8408 - Q.mass_top50) * max(0.0, 0.103005 - Q.lepsj_2_dr) / 1.51288   # +0.6%  mass_top50 < 118.8 and lepsj_2_dr < 0.103
        + 0.005875738 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 0.1845735 - Q.sj4_dr_min) / 0.03477143   # +0.6%  lep_z < 0.5187 and sj4_dr_min < 0.1846
        + 0.005834667 * max(0.0, 25.64898 - Q.sj4_pair_mass_min) / 11.12072   # +0.6%  sj4_pair_mass_min < 25.65
        - 0.005673798 * max(0.0, 5.460258 - Q.sj3_mass3) / 1.669879   # -0.6%  sj3_mass3 < 5.46
        - 0.005657142 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 14.38858 - Q.lep_iso) / 5.237586   # -0.6%  lep_z < 0.5187 and lep_iso < 14.39
        - 0.005509487 * max(0.0, 1.398635 - Q.mass_2charged) / 0.2801595   # -0.6%  mass_2charged < 1.399
        + 0.005507819 * max(0.0, 93.50967 - Q.mass_top20) / 12.41212   # +0.6%  mass_top20 < 93.51
        - 0.005409152 * max(0.0, Q.ktd_ln_d34 - -9.531553) / 0.6451699   # -0.5%  ktd_ln_d34 > -9.532
        - 0.005339792 * max(0.0, 7.065114 - Q.mres_sd_prong_mass2) / 1.50217   # -0.5%  mres_sd_prong_mass2 < 7.065
        + 0.005239723 * max(0.0, Q.lund_max_lnkt - 3.388322) / 0.5830947   # +0.5%  lund_max_lnkt > 3.388
        + 0.00505254 * max(0.0, 6.402344 - Q.max_abs_dz) / 3.33271   # +0.5%  max_abs_dz < 6.402
        - 0.004882783 * max(0.0, 0.1713451 - Q.pz_lnd0) / 0.06050411   # -0.5%  pz_lnd0 < 0.1713
        + 0.004656519 * max(0.0, 1.0 - Q.dc_1_n_lep) / 0.78823   # +0.5%  dc_1_n_lep < 1
        + 0.004344601 * max(0.0, 411.0 - Q.n_pairs_kt_above_1) / 159.5699   # +0.4%  n_pairs_kt_above_1 < 411
        - 0.004213493 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 0.8786609 - Q.tau54) / 0.02005551   # -0.4%  lep_z < 0.5187 and tau54 < 0.8787
        + 0.003963647 * max(0.0, 0.3479096 - Q.sjf_3_3_maxsd0) / 0.04351868   # +0.4%  sjf_3_3_maxsd0 < 0.3479
        - 0.003953068 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.ak02_n - 2.0) / 0.3373298   # -0.4%  lep_z < 0.5187 and ak02_n > 2
        - 0.003709541 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.dc_n - 1.0) / 0.5909868   # -0.4%  lep_z < 0.5187 and dc_n > 1
        - 0.003546252 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # -0.4%  max_abs_d0 < 5.812
        - 0.003390524 * max(0.0, Q.tau4 - 0.04996) / 0.004242161   # -0.3%  tau4 > 0.04996
        + 0.003323263 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 1.761851   # +0.3%  lepsj_3_maxsd0 < 2.413
        - 0.003303635 * max(0.0, 6.402344 - Q.max_abs_dz) * max(0.0, 0.01012269 - Q.lepsj_3_dr) / 0.02407785   # -0.3%  max_abs_dz < 6.402 and lepsj_3_dr < 0.01012
        + 0.003202124 * max(0.0, Q.pt_entropy - 3.22434) / 0.06429873   # +0.3%  pt_entropy > 3.224
        + 0.003180616 * max(0.0, Q.psi_0p3 - 0.9756505) / 0.00473588   # +0.3%  psi_0p3 > 0.9757
        - 0.003152809 * max(0.0, 0.0007909605 - Q.e3_b2) * max(0.0, 1.0 - Q.isphoton_53) / 0.0005913749   # -0.3%  e3_b2 < 0.000791 and isphoton_53 < 1
        - 0.003078798 * max(0.0, 0.09733903 - Q.tau2) * max(0.0, Q.kt2_1_n_disp3 - 0.0) / 0.02717373   # -0.3%  tau2 < 0.09734 and kt2_1_n_disp3 > 0
        - 0.002421862 * max(0.0, Q.mres_sd_mass_b1z01 - 88.79082) / 24.90103   # -0.2%  mres_sd_mass_b1z01 > 88.79
        + 0.002327815 * max(0.0, Q.sj3_pair_mass_min - 59.49644) * max(0.0, 4.0 - Q.n_lund_kt_above_5) / 2.905603   # +0.2%  sj3_pair_mass_min > 59.5 and n_lund_kt_above_5 < 4
        - 0.002241272 * max(0.0, 0.9367772 - Q.pair_mean_lnkt) * max(0.0, 428.9828 - Q.sum_pt_top10) / 14.08647   # -0.2%  pair_mean_lnkt < 0.9368 and sum_pt_top10 < 429
        + 0.002129702 * max(0.0, 0.01013989 - Q.M3_b2) / 0.001758274   # +0.2%  M3_b2 < 0.01014
        + 0.001981159 * max(0.0, Q.ktd_ln_d34 - -9.531553) * max(0.0, 0.102661 - Q.pz_lnkt3) / 0.02974445   # +0.2%  ktd_ln_d34 > -9.532 and pz_lnkt3 < 0.1027
        + 0.001655528 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.lne_1 - 4.635336) / 0.08574602   # +0.2%  lep_z < 0.5187 and lne_1 > 4.635
        - 0.001559545 * max(0.0, 118.8408 - Q.mass_top50) / 16.89652   # -0.2%  mass_top50 < 118.8
        + 0.001335453 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.sdb_5_z - 0.04013001) / 5.763019e-06   # +0.1%  e3_b2 < 0.0004127 and sdb_5_z > 0.04013
        + 0.001302921 * max(0.0, 0.06871203 - Q.pz_lnd0) / 0.006611837   # +0.1%  pz_lnd0 < 0.06871
        + 0.001298862 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 0.2509165 - Q.sdb_2_z) / 0.01790895   # +0.1%  lep_z < 0.5187 and sdb_2_z < 0.2509
        + 0.001247492 * max(0.0, Q.n_sd0_above_3 - 2.0) * max(0.0, 0.9433644 - Q.tau43_b2) / 0.4260101   # +0.1%  n_sd0_above_3 > 2 and tau43_b2 < 0.9434
        + 0.001223874 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.sdb_3_n - 2.0) / 0.5084269   # +0.1%  lep_z < 0.5187 and sdb_3_n > 2
        - 0.001127893 * max(0.0, 40.24169 - Q.mass_neutral) / 7.173374   # -0.1%  mass_neutral < 40.24
        - 0.001074441 * max(0.0, Q.z_neutral_had - 0.3788785) / 0.004557199   # -0.1%  z_neutral_had > 0.3789
        - 0.0009186869 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.lam1_plus_lam2 - 0.04303099) / 0.2409715   # -0.1%  mass_displaced3 < 39.1 and lam1_plus_lam2 > 0.04303
        + 0.0008555117 * max(0.0, 0.1713451 - Q.pz_lnd0) * max(0.0, Q.z_displaced5 - 0.1046203) / 0.00164923   # +0.1%  pz_lnd0 < 0.1713 and z_displaced5 > 0.1046
        + 0.0006966727 * max(0.0, 94.51361 - Q.mass_top50) / 6.251054   # +0.1%  mass_top50 < 94.51
        + 0.0006684043 * max(0.0, Q.jet_abs_eta - 1.465946) / 0.02086851   # +0.1%  jet_abs_eta > 1.466
        - 0.000667353 * max(0.0, 133.1648 - Q.mass_top15) / 52.0744   # -0.1%  mass_top15 < 133.2
        - 0.0005523203 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 0.1525194 - Q.pz_lnkt1) / 0.06292119   # -0.1%  n_s3d_above_3 > 4 and pz_lnkt1 < 0.1525
        - 0.0004507976 * max(0.0, 0.006220408 - Q.psi_0p1) / 0.0004886818   # -0.0%  psi_0p1 < 0.00622
        + 0.0003606942 * max(0.0, 1.398635 - Q.mass_2charged) * max(0.0, Q.pz_lnd1 - 0.0906501) / 0.003706533   # +0.0%  mass_2charged < 1.399 and pz_lnd1 > 0.09065
        - 0.0003450687 * max(0.0, Q.n_neutral_had - 5.0) / 0.6360767   # -0.0%  n_neutral_had > 5
        - 0.0003263915 * max(0.0, Q.mass_displaced5 - 21.12768) / 1.493669   # -0.0%  mass_displaced5 > 21.13
        + 0.0002597302 * max(0.0, Q.ktd_ln_d34 - -9.531553) * max(0.0, Q.sjq_2_sumabs_k03 - 1.261566) / 0.05052534   # +0.0%  ktd_ln_d34 > -9.532 and sjq_2_sumabs_k03 > 1.262
        - 0.0002254303 * max(0.0, Q.lam1 - 0.04304553) / 0.005099272   # -0.0%  lam1 > 0.04305
        - 0.0002219011 * max(0.0, Q.sj3_pair_mass_min - 59.49644) / 2.657693   # -0.0%  sj3_pair_mass_min > 59.5
        + 0.0001998778 * max(0.0, 118.8408 - Q.mass_top50) * max(0.0, Q.n_dr_0_0p05 - 2.0) / 72.90777   # +0.0%  mass_top50 < 118.8 and n_dr_0_0p05 > 2
        + 0.0001922709 * max(0.0, Q.ktd_ln_d34 - -7.075141) / 0.02510929   # +0.0%  ktd_ln_d34 > -7.075
        + 0.000173886 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) * max(0.0, Q.sjf_3_3_n_d3 - 3.0) / 1.02452   # +0.0%  sj3_pair_mass_max < 114.8 and sjf_3_3_n_d3 > 3
        - 0.0001213138 * max(0.0, 0.2243273 - Q.N2_b2) * max(0.0, Q.d0err_52 - 0.0) / 0.0001524602   # -0.0%  N2_b2 < 0.2243 and d0err_52 > 0
        - 0.0001055091 * max(0.0, 0.1713451 - Q.pz_lnd0) * max(0.0, Q.jd_3d_5 - 1.108533) / 1.378019   # -0.0%  pz_lnd0 < 0.1713 and jd_3d_5 > 1.109
        - 5.28332e-05 * max(0.0, 0.1713451 - Q.pz_lnd0) * max(0.0, 1.0 - Q.kt2_2_n_lep) / 0.04632292   # -0.0%  pz_lnd0 < 0.1713 and kt2_2_n_lep < 1
        - 4.058923e-05 * max(0.0, 0.8939856 - Q.tau43) * max(0.0, Q.dc_mass_disp_2nd - 0.0) / 0.006007192   # -0.0%  tau43 < 0.894 and dc_mass_disp_2nd > 0
        + 1.294767e-05 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.dc_tag_2nd - 0.0) / 13.25572   # +0.0%  mass_displaced3 < 39.1 and dc_tag_2nd > 0
    )
    return z


def neuron_116(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.399319e-07
    )
    return z


def neuron_117(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.862136e-06
    )
    return z


def neuron_118(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.233224e-06
    )
    return z


def neuron_119(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.336861e-07
    )
    return z


def neuron_120(Q):
    # scale S = 9.25; each line: share * term / its average size
    z = 9.249655 * (0.1501698
        - 0.07316788 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 10.0 - Q.n_s3d_above_3) / 15.42709   # -7.3%  lep_ptrel < 3.535 and n_s3d_above_3 < 10
        - 0.05790902 * max(0.0, 111.701 - Q.sd_mass) / 26.44189   # -5.8%  sd_mass < 111.7
        + 0.05299638 * max(0.0, 88.81751 - Q.sd_mass) / 15.02663   # +5.3%  sd_mass < 88.82
        - 0.05106953 * max(0.0, 12.15228 - Q.lep_ptrel) / 8.819929   # -5.1%  lep_ptrel < 12.15
        - 0.03585635 * max(0.0, 114.0172 - Q.mass) / 13.82223   # -3.6%  mass < 114
        + 0.03573956 * max(0.0, 178.725 - Q.mass_top50) / 66.98604   # +3.6%  mass_top50 < 178.7
        - 0.03099231 * max(0.0, 0.004135872 - Q.lep_z) / 0.002390373   # -3.1%  lep_z < 0.004136
        + 0.03089453 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 172.888 - Q.jd_3d_4) / 351.393   # +3.1%  lep_ptrel < 3.535 and jd_3d_4 < 172.9
        - 0.02715749 * max(0.0, 3.53503 - Q.lep_ptrel) / 2.299439   # -2.7%  lep_ptrel < 3.535
        + 0.02631094 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, 1.0 - Q.dc_1_n_lep) / 0.2724246   # +2.6%  lep_iso < 0.4382 and dc_1_n_lep < 1
        - 0.02597257 * max(0.0, 0.06573337 - Q.lepsj_2_dr) / 0.04985349   # -2.6%  lepsj_2_dr < 0.06573
        + 0.025764 * max(0.0, 0.08178299 - Q.lep_dr) / 0.04892529   # +2.6%  lep_dr < 0.08178
        - 0.02575025 * max(0.0, 68.20711 - Q.mass_charged) / 12.83892   # -2.6%  mass_charged < 68.21
        - 0.02552857 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # -2.6%  lep_iso < 0.4382
        - 0.02436688 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 3.151933 - Q.jd_3d_6) / 9.359889   # -2.4%  lep_ptrel < 12.15 and jd_3d_6 < 3.152
        + 0.02391359 * max(0.0, 2.907433 - Q.lep_iso) * max(0.0, 5.367501 - Q.jd_3d_6) / 6.467594   # +2.4%  lep_iso < 2.907 and jd_3d_6 < 5.368
        - 0.0228671 * max(0.0, 0.06573337 - Q.lepsj_2_dr) * max(0.0, 5.367501 - Q.jd_3d_6) / 0.1450967   # -2.3%  lepsj_2_dr < 0.06573 and jd_3d_6 < 5.368
        - 0.02162718 * max(0.0, 0.5187302 - Q.lep_z) / 0.4405854   # -2.2%  lep_z < 0.5187
        - 0.02132659 * max(0.0, 105.7234 - Q.mass) / 10.09077   # -2.1%  mass < 105.7
        + 0.01895456 * max(0.0, 0.0008210142 - Q.lepsj_2_dr) / 0.0004891649   # +1.9%  lepsj_2_dr < 0.000821
        + 0.01884107 * max(0.0, 68.20711 - Q.mass_charged) * max(0.0, 5.367501 - Q.jd_3d_6) / 44.32806   # +1.9%  mass_charged < 68.21 and jd_3d_6 < 5.368
        + 0.0188025 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, Q.n_photon - 5.0) / 28.0407   # +1.9%  lep_ptrel < 3.535 and n_photon > 5
        + 0.01673704 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 18.0 - Q.sdb_2_n) / 61.14732   # +1.7%  lep_ptrel < 12.15 and sdb_2_n < 18
        - 0.0161215 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) / 16.68515   # -1.6%  n_pairs_kt_above_3 < 62
        - 0.01609004 * max(0.0, 11.0 - Q.n_pairs_kt_above_10) / 5.855997   # -1.6%  n_pairs_kt_above_10 < 11
        - 0.01480322 * max(0.0, 0.3014662 - Q.lep_dr) / 0.2238267   # -1.5%  lep_dr < 0.3015
        + 0.01470819 * max(0.0, 114.0172 - Q.mass) * max(0.0, 5.367501 - Q.jd_3d_6) / 48.98244   # +1.5%  mass < 114 and jd_3d_6 < 5.368
        + 0.01423951 * max(0.0, 0.1589878 - Q.pz_lnkt0) / 0.02919723   # +1.4%  pz_lnkt0 < 0.159
        + 0.01373172 * max(0.0, 2.907433 - Q.lep_iso) / 2.168962   # +1.4%  lep_iso < 2.907
        - 0.01201228 * max(0.0, 78.4753 - Q.sd_mass) / 11.39114   # -1.2%  sd_mass < 78.48
        - 0.01026834 * max(0.0, 1.0 - Q.ak02_1_n_lep) / 0.7451267   # -1.0%  ak02_1_n_lep < 1
        + 0.009393751 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 5.8125 - Q.max_abs_d0) / 6.645755   # +0.9%  lep_ptrel < 3.535 and max_abs_d0 < 5.812
        + 0.009054326 * max(0.0, 63.87145 - Q.mass_neutral) / 22.26348   # +0.9%  mass_neutral < 63.87
        + 0.00903642 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.mass_displaced3 - 0.0) / 3.662943   # +0.9%  lep_z < 0.5187 and mass_displaced3 > 0
        - 0.008258483 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 69.1565 - Q.mass_neutral) / 10.38924   # -0.8%  lep_z < 0.5187 and mass_neutral < 69.16
        + 0.007517693 * max(0.0, 11.0 - Q.n_pairs_kt_above_10) * max(0.0, 3.151933 - Q.jd_3d_6) / 7.085947   # +0.8%  n_pairs_kt_above_10 < 11 and jd_3d_6 < 3.152
        - 0.007482477 * max(0.0, Q.lund3_lndelta - -2.036501) / 0.629841   # -0.7%  lund3_lndelta > -2.037
        + 0.006518426 * max(0.0, 0.05805236 - Q.M2) / 0.003596358   # +0.7%  M2 < 0.05805
        + 0.006332277 * max(0.0, 71.96396 - Q.mass) / 1.934957   # +0.6%  mass < 71.96
        + 0.006294212 * max(0.0, 0.6800935 - Q.tau32) / 0.0806267   # +0.6%  tau32 < 0.6801
        - 0.005933137 * max(0.0, 0.05805236 - Q.M2) * max(0.0, 6.771002 - Q.jd_3d_5) / 0.01364138   # -0.6%  M2 < 0.05805 and jd_3d_5 < 6.771
        + 0.00515485 * max(0.0, 114.0172 - Q.mass) * max(0.0, 0.1271955 - Q.z_neutral_had) / 0.6099698   # +0.5%  mass < 114 and z_neutral_had < 0.1272
        + 0.004690738 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 5.0 - Q.sdb_3_n) / 20.52156   # +0.5%  lep_ptrel < 12.15 and sdb_3_n < 5
        + 0.004660495 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 0.7095008 - Q.planar_flow) / 0.1108101   # +0.5%  lep_z < 0.5187 and planar_flow < 0.7095
        - 0.004401105 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # -0.4%  n_pairs_kt_above_1 < 80
        + 0.004294816 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.1822601 - Q.lepsj_2_dr) / 2.79471   # +0.4%  n_pairs_kt_above_3 < 62 and lepsj_2_dr < 0.1823
        - 0.00394491 * max(0.0, Q.mass_top20 - 114.4658) / 4.613131   # -0.4%  mass_top20 > 114.5
        - 0.003934941 * max(0.0, 0.3667049 - Q.N2_b05) / 0.008239872   # -0.4%  N2_b05 < 0.3667
        + 0.003897732 * max(0.0, 91.2852 - Q.sj2_mass1) / 51.87338   # +0.4%  sj2_mass1 < 91.29
        + 0.003794111 * max(0.0, 3.0 - Q.n_s3d_above_10) / 1.281233   # +0.4%  n_s3d_above_10 < 3
        - 0.003777748 * max(0.0, 4.912483 - Q.kt2_min12_jp) / 1.902446   # -0.4%  kt2_min12_jp < 4.912
        - 0.00280397 * max(0.0, Q.lam1 - 0.04304553) / 0.005099272   # -0.3%  lam1 > 0.04305
        + 0.002748635 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, 1.0 - Q.dc_2_n_lep) / 12.75663   # +0.3%  n_pairs_kt_above_3 < 62 and dc_2_n_lep < 1
        - 0.00260428 * max(0.0, 0.1589878 - Q.pz_lnkt0) * max(0.0, Q.sjq_2_prod_k05 - -0.3383985) / 0.008728143   # -0.3%  pz_lnkt0 < 0.159 and sjq_2_prod_k05 > -0.3384
        - 0.002581265 * max(0.0, 3.029824 - Q.sjf_2_2_max3d) / 0.5243565   # -0.3%  sjf_2_2_max3d < 3.03
        - 0.002576772 * max(0.0, 4.6892 - Q.mres_sd_prong_mass2) / 0.7312792   # -0.3%  mres_sd_prong_mass2 < 4.689
        - 0.002521697 * max(0.0, Q.pair_mean_lndelta - -1.790445) / 0.06864152   # -0.3%  pair_mean_lndelta > -1.79
        + 0.002313619 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.2710171 - Q.sjf_2_1_z_d3) / 3.530667   # +0.2%  n_pairs_kt_above_3 < 62 and sjf_2_1_z_d3 < 0.271
        + 0.002215982 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # +0.2%  lep_ptrel > 27.32
        - 0.002069739 * max(0.0, Q.mass_top50 - 135.5368) / 6.535045   # -0.2%  mass_top50 > 135.5
        - 0.001959529 * max(0.0, 0.03277088 - Q.pz_lnd2) / 0.003265594   # -0.2%  pz_lnd2 < 0.03277
        - 0.001892313 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # -0.2%  mass_top10 > 103.5
        - 0.001870351 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, Q.pz_lnd2 - 0.008992646) / 0.9741691   # -0.2%  lep_ptrel < 12.15 and pz_lnd2 > 0.008993
        + 0.001834911 * Q.sjf_3_n2disp / 0.81625   # +0.2%  sjf_3_n2disp
        - 0.001791352 * max(0.0, 0.1589878 - Q.pz_lnkt0) * max(0.0, 0.7462286 - Q.sj3_dr13) / 0.009970705   # -0.2%  pz_lnkt0 < 0.159 and sj3_dr13 < 0.7462
        - 0.0017198 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.2%  mass < 90.09
        + 0.00165514 * max(0.0, 63.87145 - Q.mass_neutral) * max(0.0, 4.004982 - Q.jd_3d_5) / 36.04702   # +0.2%  mass_neutral < 63.87 and jd_3d_5 < 4.005
        + 0.001653033 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, Q.e4_b05 - 4.363663e-05) / 3.240758e-05   # +0.2%  lep_ptrel < 3.535 and e4_b05 > 4.364e-05
        + 0.00162636 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, Q.n_muon - 0.0) / 0.7787752   # +0.2%  lep_ptrel < 12.15 and n_muon > 0
        + 0.001495813 * max(0.0, Q.sjq_3_3_k1 - 0.4037488) / 0.02997137   # +0.1%  sjq_3_3_k1 > 0.4037
        - 0.001482604 * max(0.0, Q.pair_mean_lnm2 - 4.787589) / 0.03990488   # -0.1%  pair_mean_lnm2 > 4.788
        - 0.001424534 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, Q.sv_2_z - 0.0) / 0.01309156   # -0.1%  lep_ptrel < 3.535 and sv_2_z > 0
        + 0.001293931 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # +0.1%  mass_top50 > 161.1
        + 0.001262189 * max(0.0, Q.e3 - 0.00228569) / 0.000293828   # +0.1%  e3 > 0.002286
        - 0.00123731 * max(0.0, 0.01394245 - Q.dr_min_012) / 0.002973681   # -0.1%  dr_min_012 < 0.01394
        + 0.001078459 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, -0.2912597 - Q.sjq_3_3_k1) / 0.01773337   # +0.1%  lep_z < 0.5187 and sjq_3_3_k1 < -0.2913
        - 0.001022607 * max(0.0, Q.ak02_1_n_lep - 1.0) / 0.05023   # -0.1%  ak02_1_n_lep > 1
        + 0.0009540828 * max(0.0, 0.05066241 - Q.pz_lnd0) / 0.003045628   # +0.1%  pz_lnd0 < 0.05066
        + 0.0008550669 * max(0.0, 0.05066241 - Q.pz_lnd0) * max(0.0, 0.0538419 - Q.dr_19) / 3.863467e-05   # +0.1%  pz_lnd0 < 0.05066 and dr_19 < 0.05384
        + 0.0007475651 * max(0.0, Q.mass_top10 - 103.4976) * max(0.0, Q.sum_pt_top40 - 502.3395) / 282.4757   # +0.1%  mass_top10 > 103.5 and sum_pt_top40 > 502.3
        - 0.0006810044 * max(0.0, 0.06573337 - Q.lepsj_2_dr) * max(0.0, Q.sjq_3_3_k1 - 0.4037488) / 0.00137647   # -0.1%  lepsj_2_dr < 0.06573 and sjq_3_3_k1 > 0.4037
        - 0.0006730678 * max(0.0, 3.0 - Q.n_sd0_above_5) / 1.183617   # -0.1%  n_sd0_above_5 < 3
        - 0.0006636217 * max(0.0, 124.9603 - Q.jd_sum_abs_sd0_top5) / 47.50041   # -0.1%  jd_sum_abs_sd0_top5 < 125
        - 0.0006609707 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.1%  lep_ptrel > 43.21
        + 0.0006360537 * max(0.0, Q.lam1 - 0.04304553) * max(0.0, Q.lund3_lnkt - 0.3409807) / 0.007994184   # +0.1%  lam1 > 0.04305 and lund3_lnkt > 0.341
        - 0.0005424958 * max(0.0, 3.0 - Q.n_s3d_above_10) * max(0.0, Q.sdb_4_z - 0.02669833) / 0.01095775   # -0.1%  n_s3d_above_10 < 3 and sdb_4_z > 0.0267
        - 0.0004840514 * max(0.0, Q.sj3_dr13 - 0.7462286) / 0.01200383   # -0.0%  sj3_dr13 > 0.7462
        - 0.0003873405 * max(0.0, Q.lep_ptrel - 6.983043) / 5.140178   # -0.0%  lep_ptrel > 6.983
        + 0.0002720584 * max(0.0, 3.0 - Q.n_sd0_above_5) * max(0.0, Q.ak02_3_n_lep - 0.0) / 0.04567   # +0.0%  n_sd0_above_5 < 3 and ak02_3_n_lep > 0
        - 0.0002569583 * max(0.0, Q.lam1 - 0.04304553) * max(0.0, Q.dc_3_n_lep - 0.0) / 0.0004429441   # -0.0%  lam1 > 0.04305 and dc_3_n_lep > 0
        + 0.0001272203 * max(0.0, 0.2977501 - Q.max_dr) / 0.003051838   # +0.0%  max_dr < 0.2978
        - 8.263499e-05 * max(0.0, Q.mass_top10 - 103.4976) * max(0.0, 0.3728877 - Q.ak02_dr23) / 0.2125584   # -0.0%  mass_top10 > 103.5 and ak02_dr23 < 0.3729
        - 7.946723e-05 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.jd_3d_6 - 30.57088) / 2.411398   # -0.0%  lep_z < 0.5187 and jd_3d_6 > 30.57
        - 6.28109e-05 * max(0.0, 0.6800935 - Q.tau32) * max(0.0, Q.sjf_4_2_mass_d3 - 3.450645) / 0.1935294   # -0.0%  tau32 < 0.6801 and sjf_4_2_mass_d3 > 3.451
        - 6.225472e-05 * max(0.0, 0.01394245 - Q.dr_min_012) * max(0.0, Q.iselectron_1 - 0.0) / 5.126283e-05   # -0.0%  dr_min_012 < 0.01394 and iselectron_1 > 0
        + 6.220276e-05 * max(0.0, 0.6800935 - Q.tau32) * max(0.0, -0.4042365 - Q.sjq_3_3_k1) / 0.002623629   # +0.0%  tau32 < 0.6801 and sjq_3_3_k1 < -0.4042
        + 3.778894e-05 * max(0.0, 0.7954618 - Q.ak02_1_z) / 0.1173141   # +0.0%  ak02_1_z < 0.7955
        + 2.731706e-05 * max(0.0, 0.6800935 - Q.tau32) * max(0.0, 3.928781 - Q.sj2_mass2) / 0.02302154   # +0.0%  tau32 < 0.6801 and sj2_mass2 < 3.929
        + 1.054898e-05 * max(0.0, 0.05066241 - Q.pz_lnd0) * max(0.0, Q.lund2_lnkt - 4.507197) / 4.005918e-05   # +0.0%  pz_lnd0 < 0.05066 and lund2_lnkt > 4.507
        + 3.90533e-06 * max(0.0, Q.sjq_3_3_k1 - 0.4037488) * max(0.0, Q.iselectron_12 - 0.0) / 0.0001697947   # +0.0%  sjq_3_3_k1 > 0.4037 and iselectron_12 > 0
    )
    return z


def neuron_121(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.730522e-06
    )
    return z


def neuron_122(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.3959e-06
    )
    return z


def neuron_123(Q):
    # scale S = 3.52; each line: share * term / its average size
    z = 3.52007 * (0.01084279
        - 0.07656787 * max(0.0, 0.3803178 - Q.z_displaced5) / 0.2926897   # -7.7%  z_displaced5 < 0.3803
        + 0.06374499 * max(0.0, 0.3803178 - Q.z_displaced5) * max(0.0, 0.06457187 - Q.lep_z) / 0.01371315   # +6.4%  z_displaced5 < 0.3803 and lep_z < 0.06457
        - 0.06005243 * max(0.0, 0.06457187 - Q.lep_z) / 0.044696   # -6.0%  lep_z < 0.06457
        + 0.04936032 * max(0.0, Q.mres_sd_mass_b1z01 - 20.80297) / 82.85566   # +4.9%  mres_sd_mass_b1z01 > 20.8
        - 0.04589699 * max(0.0, 3.0 - Q.dc_1_n_disp3) / 2.37235   # -4.6%  dc_1_n_disp3 < 3
        - 0.04081544 * max(0.0, Q.mres_sd_mass_b1z01 - 73.23918) / 36.20294   # -4.1%  mres_sd_mass_b1z01 > 73.24
        - 0.034097 * max(0.0, 3.969624 - Q.jd_3d_4) / 1.144973   # -3.4%  jd_3d_4 < 3.97
        + 0.03386625 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) / 7.619143   # +3.4%  n_pairs_kt_above_3 < 41
        - 0.03356597 * max(0.0, 2.907433 - Q.lep_iso) / 2.168962   # -3.4%  lep_iso < 2.907
        + 0.02996522 * max(0.0, Q.mres_sd_mass_b1z01 - 105.3331) / 15.43055   # +3.0%  mres_sd_mass_b1z01 > 105.3
        - 0.02844399 * max(0.0, Q.mres_sd_mass_b1z01 - 88.79082) / 24.90103   # -2.8%  mres_sd_mass_b1z01 > 88.79
        + 0.02732801 * max(0.0, 0.02926638 - Q.M2_b2) / 0.006137723   # +2.7%  M2_b2 < 0.02927
        + 0.02634341 * max(0.0, 3.0 - Q.dc_1_n_disp3) * max(0.0, 0.0146198 - Q.lepsj_2_dr) / 0.02333296   # +2.6%  dc_1_n_disp3 < 3 and lepsj_2_dr < 0.01462
        - 0.02516973 * max(0.0, Q.lep_z - 0.221436) * max(0.0, 0.04178742 - Q.lepsj_2_dr) / 0.0009641362   # -2.5%  lep_z > 0.2214 and lepsj_2_dr < 0.04179
        + 0.02155071 * max(0.0, 4.606241 - Q.sip_3d_3) / 1.172198   # +2.2%  sip_3d_3 < 4.606
        + 0.02125025 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +2.1%  lep_iso < 0.4382
        + 0.0211501 * max(0.0, 0.3803178 - Q.z_displaced5) * max(0.0, Q.M2_b05 - 0.1152127) / 0.007983796   # +2.1%  z_displaced5 < 0.3803 and M2_b05 > 0.1152
        + 0.02043594 * max(0.0, 13.03663 - Q.mass_displaced3) / 8.661134   # +2.0%  mass_displaced3 < 13.04
        + 0.01755043 * max(0.0, Q.lep_z - 0.1275041) / 0.0515702   # +1.8%  lep_z > 0.1275
        - 0.01728727 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.1294021 - Q.sdb_0_z) / 0.7392055   # -1.7%  n_pairs_kt_above_3 < 41 and sdb_0_z < 0.1294
        + 0.01697042 * max(0.0, Q.mres_sd_mass_b2z01 - 130.0968) / 7.118079   # +1.7%  mres_sd_mass_b2z01 > 130.1
        - 0.01648968 * max(0.0, 0.02926638 - Q.M2_b2) * max(0.0, 1.0 - Q.kt2_2_n_lep) / 0.00458044   # -1.6%  M2_b2 < 0.02927 and kt2_2_n_lep < 1
        - 0.01581075 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # -1.6%  lep_ptrel > 27.32
        + 0.01513508 * max(0.0, Q.mres_sd_mass_b1z01 - 73.23918) * max(0.0, 67.99453 - Q.sj2_mass1) / 820.1913   # +1.5%  mres_sd_mass_b1z01 > 73.24 and sj2_mass1 < 67.99
        + 0.01288412 * max(0.0, 226.3008 - Q.sip_3d_2) * max(0.0, 9.982976 - Q.jd_3d_4) / 926.2612   # +1.3%  sip_3d_2 < 226.3 and jd_3d_4 < 9.983
        + 0.01024668 * Q.n_s3d_above_10 / 2.62949   # +1.0%  n_s3d_above_10
        + 0.01004511 * max(0.0, 3.0 - Q.sjq_3_2_nch) * max(0.0, 0.1527273 - Q.sjf_3_3_mass_d3) / 0.03427039   # +1.0%  sjq_3_2_nch < 3 and sjf_3_3_mass_d3 < 0.1527
        + 0.009490631 * Q.n_s3d_above_3 / 3.760107   # +0.9%  n_s3d_above_3
        + 0.0093893 * max(0.0, Q.sjq_2_sumabs_k1 - 0.4723988) / 0.07281504   # +0.9%  sjq_2_sumabs_k1 > 0.4724
        + 0.009113802 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # +0.9%  lep_ptrel > 43.21
        - 0.008395685 * max(0.0, 4.606241 - Q.sip_3d_3) * max(0.0, 0.186324 - Q.sjq_2_prod_k05) / 0.295086   # -0.8%  sip_3d_3 < 4.606 and sjq_2_prod_k05 < 0.1863
        - 0.008242341 * max(0.0, Q.mres_sd_mass_b2z01 - 158.2019) / 3.171488   # -0.8%  mres_sd_mass_b2z01 > 158.2
        + 0.007995091 * max(0.0, 0.3603262 - Q.z_charged_had) / 0.02270075   # +0.8%  z_charged_had < 0.3603
        + 0.007563517 * max(0.0, 6.983043 - Q.lep_ptrel) * max(0.0, 3.969624 - Q.jd_3d_4) / 5.84821   # +0.8%  lep_ptrel < 6.983 and jd_3d_4 < 3.97
        - 0.007436971 * max(0.0, 6.983043 - Q.lep_ptrel) / 4.807808   # -0.7%  lep_ptrel < 6.983
        - 0.007405786 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -0.7%  sip_3d_2 < 226.3
        + 0.006257242 * max(0.0, 5.0 - Q.sjq_2_1_nch) / 0.1836433   # +0.6%  sjq_2_1_nch < 5
        - 0.005952131 * max(0.0, Q.sjq_2_sumabs_k1 - 0.4723988) * max(0.0, 1.0 - Q.kt2_2_n_lep) / 0.03607478   # -0.6%  sjq_2_sumabs_k1 > 0.4724 and kt2_2_n_lep < 1
        + 0.005717497 * max(0.0, Q.sjq_2_sumabs_k1 - 0.2130551) / 0.1710108   # +0.6%  sjq_2_sumabs_k1 > 0.2131
        - 0.00564984 * max(0.0, 0.6800935 - Q.tau32) / 0.0806267   # -0.6%  tau32 < 0.6801
        - 0.005530799 * max(0.0, Q.lep_z - 0.1275041) * max(0.0, 0.676889 - Q.sj3_dr13) / 0.01453481   # -0.6%  lep_z > 0.1275 and sj3_dr13 < 0.6769
        - 0.005191934 * max(0.0, 3.0 - Q.sjq_3_2_nch) * max(0.0, 4.606241 - Q.sip_3d_3) / 0.4524366   # -0.5%  sjq_3_2_nch < 3 and sip_3d_3 < 4.606
        + 0.004294199 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) * max(0.0, 11.53372 - Q.lepsj_3_maxsd0) / 26.52676   # +0.4%  n_pairs_kt_above_1 < 58 and lepsj_3_maxsd0 < 11.53
        - 0.004120808 * max(0.0, Q.sjq_2_sumabs_k1 - 0.4723988) * max(0.0, 1.013911 - Q.sjq_2_sumabs_k03) / 0.01613688   # -0.4%  sjq_2_sumabs_k1 > 0.4724 and sjq_2_sumabs_k03 < 1.014
        + 0.00400141 * max(0.0, -2.572052 - Q.pair_mean_lnz) / 0.02771534   # +0.4%  pair_mean_lnz < -2.572
        - 0.003977024 * max(0.0, 0.02926638 - Q.M2_b2) * max(0.0, Q.dc_ntag - 0.0) / 0.002087832   # -0.4%  M2_b2 < 0.02927 and dc_ntag > 0
        - 0.003903405 * max(0.0, 3.0 - Q.sjq_3_2_nch) / 0.2817433   # -0.4%  sjq_3_2_nch < 3
        - 0.003802713 * max(0.0, 0.6800935 - Q.tau32) * max(0.0, 586.9572 - Q.lepsj_3_maxsd0) / 44.05529   # -0.4%  tau32 < 0.6801 and lepsj_3_maxsd0 < 587
        + 0.003255795 * max(0.0, Q.mass_top5 - 68.52153) / 2.713876   # +0.3%  mass_top5 > 68.52
        - 0.003246445 * max(0.0, 0.08996752 - Q.sj2_zsoft) / 0.003665091   # -0.3%  sj2_zsoft < 0.08997
        - 0.003139443 * max(0.0, 2.907433 - Q.lep_iso) * max(0.0, 1.0 - Q.ak02_2_n_lep) / 1.909807   # -0.3%  lep_iso < 2.907 and ak02_2_n_lep < 1
        + 0.003074644 * max(0.0, 226.3008 - Q.sip_3d_2) * max(0.0, 0.09802954 - Q.sjq_2_prod_k1) / 17.74315   # +0.3%  sip_3d_2 < 226.3 and sjq_2_prod_k1 < 0.09803
        - 0.003043436 * max(0.0, 5.543454 - Q.mres_sd_prong_mass2) / 0.9673864   # -0.3%  mres_sd_prong_mass2 < 5.543
        + 0.003038517 * max(0.0, Q.lep_z - 0.221436) / 0.03510013   # +0.3%  lep_z > 0.2214
        + 0.002998599 * max(0.0, 3.0 - Q.dc_1_n_disp3) * max(0.0, -5.255594 - Q.lnptrel_40) / 18.38021   # +0.3%  dc_1_n_disp3 < 3 and lnptrel_40 < -5.256
        - 0.002734946 * max(0.0, Q.mres_sd_mass_b2z01 - 130.0968) * max(0.0, 28.85058 - Q.lepsj_2_maxsd0) / 144.1949   # -0.3%  mres_sd_mass_b2z01 > 130.1 and lepsj_2_maxsd0 < 28.85
        + 0.002633318 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, 10.59762 - Q.min_pair_mass) / 13.72124   # +0.3%  lep_ptrel > 27.32 and min_pair_mass < 10.6
        - 0.002562796 * max(0.0, 14.92637 - Q.mass_neutral) / 0.561096   # -0.3%  mass_neutral < 14.93
        - 0.002544926 * max(0.0, 0.3265243 - Q.z_charged) / 0.003638213   # -0.3%  z_charged < 0.3265
        + 0.002460411 * max(0.0, 5.0 - Q.n_dr_0p1_0p2) / 0.5919233   # +0.2%  n_dr_0p1_0p2 < 5
        - 0.002246641 * max(0.0, 5.543454 - Q.mres_sd_prong_mass2) * max(0.0, 0.4355562 - Q.z_dr_0p05_0p1) / 0.2370129   # -0.2%  mres_sd_prong_mass2 < 5.543 and z_dr_0p05_0p1 < 0.4356
        + 0.002039467 * max(0.0, Q.mres_sd_mass_b1z01 - 73.23918) * max(0.0, 3.969624 - Q.jd_3d_4) / 28.611   # +0.2%  mres_sd_mass_b1z01 > 73.24 and jd_3d_4 < 3.97
        - 0.001971877 * max(0.0, Q.mres_sd_mass_b1z01 - 73.23918) * max(0.0, -0.3493007 - Q.lund2_lndelta) / 23.61909   # -0.2%  mres_sd_mass_b1z01 > 73.24 and lund2_lndelta < -0.3493
        - 0.001926336 * max(0.0, Q.sjq_2_sumabs_k1 - 0.4723988) * max(0.0, 0.0008210142 - Q.lepsj_2_dr) / 2.189494e-05   # -0.2%  sjq_2_sumabs_k1 > 0.4724 and lepsj_2_dr < 0.000821
        + 0.001836751 * max(0.0, 0.135772 - Q.tau21) / 0.001647604   # +0.2%  tau21 < 0.1358
        + 0.001795993 * max(0.0, 226.3008 - Q.sip_3d_2) * max(0.0, Q.lepsj_2_n_d3 - 2.0) / 19.01415   # +0.2%  sip_3d_2 < 226.3 and lepsj_2_n_d3 > 2
        - 0.001739617 * max(0.0, 0.302405 - Q.N2_b05) / 0.002473216   # -0.2%  N2_b05 < 0.3024
        + 0.001624289 * max(0.0, 172.888 - Q.jd_3d_4) / 152.4729   # +0.2%  jd_3d_4 < 172.9
        - 0.001604333 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, Q.ecf_g41 - 6.216954e-05) / 0.0002440357   # -0.2%  lep_ptrel > 27.32 and ecf_g41 > 6.217e-05
        - 0.001449071 * max(0.0, 0.06457187 - Q.lep_z) * max(0.0, 4.606241 - Q.sip_3d_3) / 0.05651432   # -0.1%  lep_z < 0.06457 and sip_3d_3 < 4.606
        - 0.001426773 * max(0.0, 0.07823361 - Q.pz_lnkt0) / 0.002331445   # -0.1%  pz_lnkt0 < 0.07823
        - 0.001426046 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # -0.1%  n_pairs_kt_above_1 < 58
        + 0.00133057 * max(0.0, Q.mres_sd_mass_b1z01 - 73.23918) * max(0.0, 0.3603262 - Q.z_charged_had) / 0.6417097   # +0.1%  mres_sd_mass_b1z01 > 73.24 and z_charged_had < 0.3603
        + 0.001306789 * max(0.0, 14.92637 - Q.mass_neutral) * max(0.0, 0.04397771 - Q.mres_sd_zg_b2z01) / 0.003673321   # +0.1%  mass_neutral < 14.93 and mres_sd_zg_b2z01 < 0.04398
        + 0.001249906 * max(0.0, Q.sjf_2_1_n_d3 - 7.0) / 0.05848333   # +0.1%  sjf_2_1_n_d3 > 7
        - 0.001193945 * max(0.0, 226.3008 - Q.sip_3d_2) * max(0.0, Q.ak02_1_n_lep - 0.0) / 45.58267   # -0.1%  sip_3d_2 < 226.3 and ak02_1_n_lep > 0
        - 0.001175384 * max(0.0, Q.pair_mean_lndelta - -1.523797) / 0.02402574   # -0.1%  pair_mean_lndelta > -1.524
        + 0.001174482 * max(0.0, 2.224391 - Q.sjf_2_2_max3d) / 0.2112922   # +0.1%  sjf_2_2_max3d < 2.224
        - 0.001167085 * max(0.0, Q.sjq_2_sumabs_k1 - 0.4723988) * max(0.0, Q.lund3_lnz - -4.227393) / 0.07539916   # -0.1%  sjq_2_sumabs_k1 > 0.4724 and lund3_lnz > -4.227
        + 0.001094671 * max(0.0, 0.02926638 - Q.M2_b2) * max(0.0, 1.546763 - Q.lepsj_2_maxsd0) / 0.006867676   # +0.1%  M2_b2 < 0.02927 and lepsj_2_maxsd0 < 1.547
        + 0.0009858218 * max(0.0, 3.0 - Q.sjq_3_2_nch) * max(0.0, 0.03240401 - Q.ak02_2_z) / 0.001792477   # +0.1%  sjq_3_2_nch < 3 and ak02_2_z < 0.0324
        - 0.0009749912 * max(0.0, Q.mres_sd_mass_b2z01 - 158.2019) * max(0.0, 2.953464 - Q.jd_3d_4) / 0.6607881   # -0.1%  mres_sd_mass_b2z01 > 158.2 and jd_3d_4 < 2.953
        + 0.0009413945 * max(0.0, 5.0 - Q.n_lund) / 0.1017067   # +0.1%  n_lund < 5
        - 0.00092308 * max(0.0, 0.02926638 - Q.M2_b2) * max(0.0, Q.dc_pair_mass_min - 105.4151) / 0.02230033   # -0.1%  M2_b2 < 0.02927 and dc_pair_mass_min > 105.4
        + 0.0008816229 * max(0.0, 0.6800935 - Q.tau32) * max(0.0, Q.eta_29 - 0.2073975) / 0.001621309   # +0.1%  tau32 < 0.6801 and eta_29 > 0.2074
        - 0.0008248834 * max(0.0, Q.mass_top5 - 68.52153) * max(0.0, Q.n_lund_kt_above_5 - 2.0) / 0.7598323   # -0.1%  mass_top5 > 68.52 and n_lund_kt_above_5 > 2
        + 0.0007016437 * max(0.0, Q.lep_z - 0.1275041) * max(0.0, Q.lund2_lndelta - -1.052328) / 0.01077732   # +0.1%  lep_z > 0.1275 and lund2_lndelta > -1.052
        - 0.0006479496 * max(0.0, Q.M3_b05 - 0.08895667) / 0.001302358   # -0.1%  M3_b05 > 0.08896
        + 0.0003940539 * max(0.0, Q.lep_z - 0.221436) * max(0.0, -0.1245117 - Q.eta_11) / 0.001302488   # +0.0%  lep_z > 0.2214 and eta_11 < -0.1245
        - 0.0003934056 * max(0.0, Q.lep_ptrel - 43.20788) * max(0.0, Q.eta_29 - 0.0) / 0.03414251   # -0.0%  lep_ptrel > 43.21 and eta_29 > 0
        + 0.000284623 * max(0.0, Q.mres_sd_mass_b1z01 - 73.23918) * max(0.0, Q.lnpt_1 - 4.37968) / 3.103499   # +0.0%  mres_sd_mass_b1z01 > 73.24 and lnpt_1 > 4.38
        + 0.0002428215 * max(0.0, 13.03663 - Q.mass_displaced3) * max(0.0, Q.iselectron_1 - 0.0) / 0.1996362   # +0.0%  mass_displaced3 < 13.04 and iselectron_1 > 0
        + 0.0002229456 * max(0.0, Q.lep_z - 0.1275041) * max(0.0, Q.pz_lnkt2 - 0.07998846) / 0.0007692691   # +0.0%  lep_z > 0.1275 and pz_lnkt2 > 0.07999
        + 0.0002170986 * max(0.0, 0.302405 - Q.N2_b05) * max(0.0, 0.1645421 - Q.sj2_zsoft) / 1.126197e-05   # +0.0%  N2_b05 < 0.3024 and sj2_zsoft < 0.1645
        + 0.0001481921 * max(0.0, 0.05462126 - Q.pz_lnd2) / 0.00878118   # +0.0%  pz_lnd2 < 0.05462
        - 0.0001121026 * max(0.0, Q.lep_ptrel - 43.20788) * max(0.0, -0.2275391 - Q.phi_4) / 0.008637302   # -0.0%  lep_ptrel > 43.21 and phi_4 < -0.2275
        + 5.409411e-05 * max(0.0, 0.02926638 - Q.M2_b2) * max(0.0, Q.dr_1 - 0.1975394) / 0.0001116294   # +0.0%  M2_b2 < 0.02927 and dr_1 > 0.1975
        - 4.420726e-05 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, 21.98452 - Q.jd_3d_4) / 20.83747   # -0.0%  lep_ptrel > 27.32 and jd_3d_4 < 21.98
        - 2.172453e-05 * max(0.0, 0.08996752 - Q.sj2_zsoft) * max(0.0, Q.pz_lnkt3 - 0.1312056) / 5.374028e-07   # -0.0%  sj2_zsoft < 0.08997 and pz_lnkt3 > 0.1312
        - 1.375039e-05 * max(0.0, Q.mres_sd_mass_b2z01 - 130.0968) * max(0.0, 0.2632495 - Q.dc_split1_z) / 0.3181632   # -0.0%  mres_sd_mass_b2z01 > 130.1 and dc_split1_z < 0.2632
    )
    return z


def neuron_124(Q):
    # scale S = 9.998; each line: share * term / its average size
    z = 9.997952 * (-0.02896422
        + 0.09459214 * max(0.0, Q.mres_sd_mass_b2z01 - 91.15481) / 24.25173   # +9.5%  mres_sd_mass_b2z01 > 91.15
        - 0.06539449 * max(0.0, Q.mres_sd_mass_b2z01 - 76.1529) / 35.12032   # -6.5%  mres_sd_mass_b2z01 > 76.15
        + 0.06227107 * max(0.0, 0.04982189 - Q.lepsj_3_dr) / 0.04047859   # +6.2%  lepsj_3_dr < 0.04982
        + 0.05088934 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # +5.1%  lep_z < 0.2214
        + 0.0407054 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # +4.1%  n_lepton < 1
        - 0.03342788 * max(0.0, Q.mres_sd_mass_b0z005 - 76.40585) / 35.53887   # -3.3%  mres_sd_mass_b0z005 > 76.41
        + 0.02967486 * max(0.0, 55.83114 - Q.jd_sum_abs_sd0_top5) / 15.94498   # +3.0%  jd_sum_abs_sd0_top5 < 55.83
        - 0.02905551 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # -2.9%  n_pairs_kt_above_1 < 366
        - 0.02799774 * max(0.0, 0.04982189 - Q.lepsj_3_dr) * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 0.08394198   # -2.8%  lepsj_3_dr < 0.04982 and lepsj_3_maxsd0 < 2.413
        - 0.02547203 * max(0.0, Q.mres_sd_mass_b2z01 - 107.2089) / 15.02375   # -2.5%  mres_sd_mass_b2z01 > 107.2
        - 0.02305296 * max(0.0, 48.72265 - Q.jd_sum_abs_sd0_top3) / 14.29606   # -2.3%  jd_sum_abs_sd0_top3 < 48.72
        - 0.02285388 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # -2.3%  lep_iso < 0.4382
        - 0.02232191 * max(0.0, 0.09802954 - Q.sjq_2_prod_k1) / 0.1166532   # -2.2%  sjq_2_prod_k1 < 0.09803
        - 0.02204081 * max(0.0, 0.8036986 - Q.lepsj_3_maxsd0) / 0.5472878   # -2.2%  lepsj_3_maxsd0 < 0.8037
        + 0.02142704 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +2.1%  lep_ptrel < 43.21
        - 0.02137855 * max(0.0, Q.z_photon - 0.1149688) / 0.14382   # -2.1%  z_photon > 0.115
        - 0.02107152 * max(0.0, Q.mass_top40 - 70.88236) / 41.46808   # -2.1%  mass_top40 > 70.88
        + 0.020579 * max(0.0, Q.mres_sd_mass_b0z005 - 86.65765) / 27.85055   # +2.1%  mres_sd_mass_b0z005 > 86.66
        - 0.02020877 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # -2.0%  lep_iso < 1.362
        - 0.01901282 * max(0.0, Q.mres_sd_mass_b0z005 - 125.8718) / 8.385721   # -1.9%  mres_sd_mass_b0z005 > 125.9
        + 0.01796038 * max(0.0, Q.mres_sd_mass_b0z005 - 91.82593) / 24.37401   # +1.8%  mres_sd_mass_b0z005 > 91.83
        + 0.01758765 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 1.0 - Q.ak02_min12_n_disp3) / 5.28269   # +1.8%  max_abs_d0 < 10.52 and ak02_min12_n_disp3 < 1
        - 0.01653816 * max(0.0, 0.04884386 - Q.z_displaced3) / 0.01848687   # -1.7%  z_displaced3 < 0.04884
        + 0.01563397 * max(0.0, 0.04178742 - Q.lepsj_2_dr) / 0.03006022   # +1.6%  lepsj_2_dr < 0.04179
        + 0.01197017 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, 142.9952 - Q.sj3_pair_mass_max) / 1932.344   # +1.2%  lep_ptrel < 43.21 and sj3_pair_mass_max < 143
        - 0.01088293 * max(0.0, 1.0 - Q.n_lepton) * max(0.0, 2.953464 - Q.jd_3d_4) / 0.3573652   # -1.1%  n_lepton < 1 and jd_3d_4 < 2.953
        + 0.01023738 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # +1.0%  lep_z < 0.3397
        + 0.009758597 * max(0.0, 1.0 - Q.kt2_2_n_lep) / 0.77195   # +1.0%  kt2_2_n_lep < 1
        - 0.009713308 * max(0.0, Q.sjq_2_sumabs_k1 - 0.2769039) / 0.1391447   # -1.0%  sjq_2_sumabs_k1 > 0.2769
        + 0.008498723 * max(0.0, 0.09802954 - Q.sjq_2_prod_k1) * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 0.2046735   # +0.8%  sjq_2_prod_k1 < 0.09803 and lepsj_3_maxsd0 < 2.413
        - 0.008409762 * max(0.0, Q.n_s3d_above_10 - 2.0) / 1.323447   # -0.8%  n_s3d_above_10 > 2
        + 0.008222791 * max(0.0, 0.1076451 - Q.tau2) / 0.0393622   # +0.8%  tau2 < 0.1076
        + 0.008048864 * max(0.0, Q.z_displaced5 - 0.04748739) / 0.06559798   # +0.8%  z_displaced5 > 0.04749
        + 0.007602427 * max(0.0, Q.lep_dr - 0.04607888) / 0.07416711   # +0.8%  lep_dr > 0.04608
        + 0.007360467 * max(0.0, Q.mass_top40 - 70.88236) * max(0.0, -0.4311181 - Q.lund1_lndelta) / 5.382582   # +0.7%  mass_top40 > 70.88 and lund1_lndelta < -0.4311
        - 0.007122035 * max(0.0, 15.0 - Q.n_dr_0p4_up) / 9.48698   # -0.7%  n_dr_0p4_up < 15
        - 0.007094758 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # -0.7%  max_abs_d0 < 10.52
        - 0.006904367 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.z_displaced5 - 0.1046203) / 0.01140989   # -0.7%  lep_z < 0.3397 and z_displaced5 > 0.1046
        + 0.006862507 * max(0.0, Q.n_photon - 5.0) / 11.0945   # +0.7%  n_photon > 5
        + 0.006690312 * max(0.0, 35.45098 - Q.sv_1_sd0_sum) / 17.17693   # +0.7%  sv_1_sd0_sum < 35.45
        - 0.006558073 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 172.888 - Q.jd_3d_4) / 196.7144   # -0.7%  n_s3d_above_3 > 3 and jd_3d_4 < 172.9
        + 0.00623947 * max(0.0, Q.z_photon - 0.1149688) * max(0.0, 6.771002 - Q.jd_3d_5) / 0.5173637   # +0.6%  z_photon > 0.115 and jd_3d_5 < 6.771
        - 0.006061023 * max(0.0, 1.0 - Q.n_s3d_above_3) / 0.17383   # -0.6%  n_s3d_above_3 < 1
        - 0.005759423 * max(0.0, 1.864422 - Q.sjf_2_2_max3d) / 0.1211692   # -0.6%  sjf_2_2_max3d < 1.864
        + 0.005223258 * max(0.0, Q.sjq_2_sumabs_k1 - 0.2769039) * max(0.0, 2.953464 - Q.jd_3d_4) / 0.08678409   # +0.5%  sjq_2_sumabs_k1 > 0.2769 and jd_3d_4 < 2.953
        - 0.00498638 * max(0.0, Q.mres_sd_mass_b0z005 - 111.9362) / 13.23299   # -0.5%  mres_sd_mass_b0z005 > 111.9
        - 0.004983383 * max(0.0, 66.70506 - Q.sj4_pair_mass_max) / 4.704439   # -0.5%  sj4_pair_mass_max < 66.71
        - 0.00482036 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 0.06762785 - Q.M2_b2) / 0.05887994   # -0.5%  n_s3d_above_3 > 3 and M2_b2 < 0.06763
        - 0.004810113 * max(0.0, 0.1530389 - Q.sdb_2_z) / 0.01282076   # -0.5%  sdb_2_z < 0.153
        - 0.004716399 * max(0.0, 1.0 - Q.kt2_2_n_lep) * max(0.0, Q.sdb_2_z - 0.3208052) / 0.05728729   # -0.5%  kt2_2_n_lep < 1 and sdb_2_z > 0.3208
        + 0.004518955 * max(0.0, Q.n_s3d_above_3 - 3.0) / 1.66923   # +0.5%  n_s3d_above_3 > 3
        - 0.004459293 * max(0.0, Q.mres_sd_prong_mass1 - 69.04524) / 2.206937   # -0.4%  mres_sd_prong_mass1 > 69.05
        + 0.003937319 * max(0.0, Q.mres_sd_mass_b0z005 - 159.9242) / 3.119054   # +0.4%  mres_sd_mass_b0z005 > 159.9
        - 0.003504634 * max(0.0, 0.03330871 - Q.sjq_2_prod_k1) / 0.05754286   # -0.4%  sjq_2_prod_k1 < 0.03331
        + 0.003337247 * max(0.0, Q.sjq_2_sumabs_k1 - 0.2769039) * max(0.0, 0.8036986 - Q.lepsj_3_maxsd0) / 0.07416281   # +0.3%  sjq_2_sumabs_k1 > 0.2769 and lepsj_3_maxsd0 < 0.8037
        + 0.003332043 * max(0.0, Q.mres_sd_prong_mass1 - 84.19324) / 1.074491   # +0.3%  mres_sd_prong_mass1 > 84.19
        + 0.003129403 * max(0.0, 0.1076451 - Q.tau2) * max(0.0, Q.jet_charge_k03 - -0.05990128) / 0.01248115   # +0.3%  tau2 < 0.1076 and jet_charge_k03 > -0.0599
        + 0.003048069 * max(0.0, Q.sjq_2_prod_k05 - 0.004703917) / 0.05337879   # +0.3%  sjq_2_prod_k05 > 0.004704
        - 0.002951864 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # -0.3%  sj3_pair_mass_max < 73.25
        - 0.002832987 * max(0.0, Q.sjq_2_sumabs_k1 - 0.1103483) / 0.2373469   # -0.3%  sjq_2_sumabs_k1 > 0.1103
        + 0.002478413 * Q.lepsj_2_n_d3 / 0.6197967   # +0.2%  lepsj_2_n_d3
        - 0.00231334 * max(0.0, 0.06871203 - Q.pz_lnd0) / 0.006611837   # -0.2%  pz_lnd0 < 0.06871
        - 0.002206444 * max(0.0, Q.lne_17 - -0.1601665) / 2.20813   # -0.2%  lne_17 > -0.1602
        - 0.002187721 * max(0.0, 1.777787 - Q.mass_displaced5) / 0.8212198   # -0.2%  mass_displaced5 < 1.778
        + 0.002158041 * max(0.0, 0.5498426 - Q.tau32) / 0.03559481   # +0.2%  tau32 < 0.5498
        + 0.001890782 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) * max(0.0, 0.221369 - Q.C2_b2) / 20.27193   # +0.2%  n_pairs_kt_above_1 < 366 and C2_b2 < 0.2214
        - 0.00180978 * max(0.0, Q.mass_top40 - 70.88236) * max(0.0, 0.07650476 - Q.ak02_3_z) / 1.529251   # -0.2%  mass_top40 > 70.88 and ak02_3_z < 0.0765
        - 0.001742194 * max(0.0, 100.4835 - Q.mass) / 8.115061   # -0.2%  mass < 100.5
        + 0.001709166 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +0.2%  mass < 95.15
        + 0.001476704 * max(0.0, Q.sjf_2_1_n_d3 - 7.0) / 0.05848333   # +0.1%  sjf_2_1_n_d3 > 7
        - 0.001355381 * max(0.0, 0.006220408 - Q.psi_0p1) / 0.0004886818   # -0.1%  psi_0p1 < 0.00622
        + 0.001327967 * max(0.0, 15.0 - Q.n_dr_0p4_up) * max(0.0, Q.dc_ntag - 0.0) / 2.69226   # +0.1%  n_dr_0p4_up < 15 and dc_ntag > 0
        - 0.001165344 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.8096444 - Q.tau43) / 0.007638829   # -0.1%  lep_z < 0.2214 and tau43 < 0.8096
        - 0.001078815 * max(0.0, 2.413264 - Q.lepsj_3_maxsd0) / 1.761851   # -0.1%  lepsj_3_maxsd0 < 2.413
        + 0.001017502 * max(0.0, 0.1530389 - Q.sdb_2_z) * max(0.0, 0.04891969 - Q.dr_30) / 0.0003140444   # +0.1%  sdb_2_z < 0.153 and dr_30 < 0.04892
        - 0.0008980485 * max(0.0, 117.4867 - Q.mass) / 15.64803   # -0.1%  mass < 117.5
        - 0.0008367404 * max(0.0, -0.06334099 - Q.sjq_3_prod_k1) / 0.02047404   # -0.1%  sjq_3_prod_k1 < -0.06334
        + 0.000829704 * max(0.0, 0.1076451 - Q.tau2) * max(0.0, Q.sjf_4_3_z_d3 - 0.009093044) / 0.0003852683   # +0.1%  tau2 < 0.1076 and sjf_4_3_z_d3 > 0.009093
        + 0.0008023371 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.dc_4_z - 0.0) / 0.001599002   # +0.1%  lep_z < 0.2214 and dc_4_z > 0
        - 0.0007278345 * max(0.0, 35.45098 - Q.sv_1_sd0_sum) * max(0.0, 0.556239 - Q.nca_sj4_pair2nd_over_mass) / 1.189016   # -0.1%  sv_1_sd0_sum < 35.45 and nca_sj4_pair2nd_over_mass < 0.5562
        - 0.0006250227 * max(0.0, Q.tau21_b2 - 0.3898586) / 0.04953334   # -0.1%  tau21_b2 > 0.3899
        + 0.0006123126 * max(0.0, Q.lund_max_lndelta - -0.529318) / 0.01270035   # +0.1%  lund_max_lndelta > -0.5293
        - 0.0006071888 * max(0.0, Q.dc_3_charge - 0.6283033) / 0.01365463   # -0.1%  dc_3_charge > 0.6283
        - 0.000557731 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 578.2954 - Q.sip_3d_1) / 374.5995   # -0.1%  n_s3d_above_3 > 3 and sip_3d_1 < 578.3
        - 0.0005296937 * max(0.0, 15.0 - Q.n_dr_0p4_up) * max(0.0, 0.05462126 - Q.pz_lnd2) / 0.07283672   # -0.1%  n_dr_0p4_up < 15 and pz_lnd2 < 0.05462
        - 0.0004172223 * max(0.0, Q.tau21_b2 - 0.3898586) * max(0.0, 1.0 - Q.isnhad_8) / 0.04255731   # -0.0%  tau21_b2 > 0.3899 and isnhad_8 < 1
        + 0.0003787654 * max(0.0, Q.mass_top40 - 70.88236) * max(0.0, 4.004982 - Q.jd_3d_5) / 48.02072   # +0.0%  mass_top40 > 70.88 and jd_3d_5 < 4.005
        + 0.0001084565 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # +0.0%  n_pairs_kt_above_1 < 58
        - 9.668635e-05 * max(0.0, Q.sjq_2_sumabs_k1 - 0.6652978) / 0.03515981   # -0.0%  sjq_2_sumabs_k1 > 0.6653
        - 8.767899e-05 * max(0.0, Q.mass_top40 - 70.88236) * max(0.0, Q.sdb_4_z - 0.0) / 1.021464   # -0.0%  mass_top40 > 70.88 and sdb_4_z > 0
        + 8.10575e-05 * max(0.0, Q.mass_top40 - 70.88236) * max(0.0, 3.0 - Q.sv_n) / 70.05374   # +0.0%  mass_top40 > 70.88 and sv_n < 3
        + 5.317468e-05 * max(0.0, Q.n_particles - 67.0) / 0.39903   # +0.0%  n_particles > 67
        + 3.507221e-05 * max(0.0, 1.0 - Q.kt2_2_n_lep) * max(0.0, -0.04205891 - Q.tdz_0) / 0.00750527   # +0.0%  kt2_2_n_lep < 1 and tdz_0 < -0.04206
        + 3.113979e-05 * max(0.0, 0.5498426 - Q.tau32) * max(0.0, Q.isnhad_38 - 0.0) / 0.0007443742   # +0.0%  tau32 < 0.5498 and isnhad_38 > 0
        + 2.716747e-05 * max(0.0, Q.sjf_4_1_n_d3 - 2.0) / 0.5063833   # +0.0%  sjf_4_1_n_d3 > 2
        + 2.404813e-06 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 2.627723 - Q.ak02_2_sd0_3) / 19.97614   # +0.0%  n_s3d_above_3 > 3 and ak02_2_sd0_3 < 2.628
    )
    return z


def neuron_125(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.190763e-06
    )
    return z


def neuron_126(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.63263e-06
    )
    return z


def neuron_127(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.966689e-07
    )
    return z


def class_token(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q), neuron_16(Q), neuron_17(Q), neuron_18(Q), neuron_19(Q), neuron_20(Q), neuron_21(Q), neuron_22(Q), neuron_23(Q), neuron_24(Q), neuron_25(Q), neuron_26(Q), neuron_27(Q), neuron_28(Q), neuron_29(Q), neuron_30(Q), neuron_31(Q), neuron_32(Q), neuron_33(Q), neuron_34(Q), neuron_35(Q), neuron_36(Q), neuron_37(Q), neuron_38(Q), neuron_39(Q), neuron_40(Q), neuron_41(Q), neuron_42(Q), neuron_43(Q), neuron_44(Q), neuron_45(Q), neuron_46(Q), neuron_47(Q), neuron_48(Q), neuron_49(Q), neuron_50(Q), neuron_51(Q), neuron_52(Q), neuron_53(Q), neuron_54(Q), neuron_55(Q), neuron_56(Q), neuron_57(Q), neuron_58(Q), neuron_59(Q), neuron_60(Q), neuron_61(Q), neuron_62(Q), neuron_63(Q), neuron_64(Q), neuron_65(Q), neuron_66(Q), neuron_67(Q), neuron_68(Q), neuron_69(Q), neuron_70(Q), neuron_71(Q), neuron_72(Q), neuron_73(Q), neuron_74(Q), neuron_75(Q), neuron_76(Q), neuron_77(Q), neuron_78(Q), neuron_79(Q), neuron_80(Q), neuron_81(Q), neuron_82(Q), neuron_83(Q), neuron_84(Q), neuron_85(Q), neuron_86(Q), neuron_87(Q), neuron_88(Q), neuron_89(Q), neuron_90(Q), neuron_91(Q), neuron_92(Q), neuron_93(Q), neuron_94(Q), neuron_95(Q), neuron_96(Q), neuron_97(Q), neuron_98(Q), neuron_99(Q), neuron_100(Q), neuron_101(Q), neuron_102(Q), neuron_103(Q), neuron_104(Q), neuron_105(Q), neuron_106(Q), neuron_107(Q), neuron_108(Q), neuron_109(Q), neuron_110(Q), neuron_111(Q), neuron_112(Q), neuron_113(Q), neuron_114(Q), neuron_115(Q), neuron_116(Q), neuron_117(Q), neuron_118(Q), neuron_119(Q), neuron_120(Q), neuron_121(Q), neuron_122(Q), neuron_123(Q), neuron_124(Q), neuron_125(Q), neuron_126(Q), neuron_127(Q)]


H_AVG = [0.005196406971663237, 5.325267466105288e-06, 5.447299827210372e-06, 8.399836588068865e-06, 2.79213873000117e-06, 2.156925575036439e-06, 1.7701060642139055e-05, 6.430825578718213e-06, 2.926471324826707e-06, 2.44366196966439e-06, 1.5701054508099332e-05, 1.887712824100163e-05, 5.099300324218348e-06, 4.422912752488628e-06, 2.8453036975406576e-06, 6.45952559352736e-06, 1.2819195803913568, 1.0353470315749291e-05, 1.0619896374685476, 1.4078186723054387e-06, 8.253685450654302e-07, 0.0016816153656691313, 1.0448479770275299e-06, 6.978821147640701e-07, 0.9832458398332604, 0.16279899575905568, 3.231259825042798e-06, 6.308498996077105e-06, 9.617209798307158e-06, 9.183865586237516e-06, 8.603051355748903e-06, 3.509522457534331e-06, 1.4413479220820591e-05, 2.567326419011806e-06, 1.3884865438740235e-05, 6.7432956711854786e-06, 4.83194662592723e-06, 4.35743550042389e-06, 9.411179235030431e-06, 8.274279025499709e-06, 7.775539074827975e-07, 3.8147447867231676e-08, 8.779049494478386e-06, 9.407708603248466e-06, 3.324348426758661e-06, 6.849337296443991e-06, 1.3687507305348845e-07, 2.492075736881816e-07, 8.7917787823244e-06, 2.243815288238693e-06, 5.175118531042244e-06, 2.012200639001094e-06, 0.7484921208659663, 1.8893630340244272e-06, 1.1906588952115271e-05, 2.0669402147177607e-05, 7.072322318890656e-07, 1.959645487659145e-06, 3.6171502415527357e-06, 6.029332780599361e-06, 4.666764198191231e-06, 5.840562607772881e-06, 9.422977882422856e-07, 8.30851149657974e-06, 1.2586272077896865e-06, 5.791052444692468e-06, 2.7097767087980174e-06, 5.2195086936990265e-06, 0.8968695078928819, 6.9261427597666625e-06, 0.244685585411519, 3.029709432667005e-06, 1.1641216133284615e-06, 1.6996402337099425e-05, 4.493399046623381e-06, 1.973840426217066e-06, 5.542788130696863e-06, 3.08748758470756e-06, 0.5385801018364945, 3.602387732826173e-05, 9.707706340122968e-06, 0.3992075231543757, 2.9420730243145954e-06, 0.8701448058277815, 7.427511445712298e-05, 8.185519959624799e-07, 1.7099031538236886e-05, 1.6191634131246246e-06, 2.996991952386452e-07, 1.735375576572551e-06, 0.18551360102091696, 3.992637175542768e-06, 6.7247351580590475e-06, 8.464388088214037e-07, 2.322171894775238e-06, 4.36951779647643e-07, 4.278041956240486e-07, 0.9981332629612899, 4.990170623386803e-07, 0.0019342413870617747, 6.264077455853112e-06, 8.878261724021286e-06, 5.490515377459815e-06, 3.4143788525398122e-06, 0.9896458210038155, 2.7323825634084642e-05, 2.1609346731565893e-05, 1.1585600987018552e-05, 4.4707021515932865e-06, 5.200158170737268e-07, 3.57960925612133e-06, 3.669617399282288e-06, 5.237822733761277e-06, 2.6659277864382602e-05, 4.677418019127799e-06, 1.0537851234950786, 3.399319439267856e-07, 2.8621359433600446e-06, 8.233223525166977e-06, 6.336860565170355e-07, 0.9693102866894728, 5.730521934310673e-06, 2.3959003101481358e-06, 0.19597973771106206, 0.7195232383162837, 1.1907628731933073e-06, 7.632629603904206e-06, 1.9666887851599313e-07]
T = [4.199810879636352, 5.990838728041529, 4.236818137344312, 4.681489958124058, 5.237586759460459, 9.215784653022755, 4.070481045987272, 4.486392992782892, 7.3611509465445915, 11.094193234141564]


def logits(h):
    # rounded to a multiple of 2^-20 (the formula's class scores are exact binary fractions): removes floating-point noise
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        0.4043572 + T[0] * (   # class QCD
            - 0.1771128 * h[18] / H_AVG[18]
            + 0.1417459 * h[97] / H_AVG[97]
            - 0.1054666 * h[68] / H_AVG[68]
            + 0.09163652 * h[52] / H_AVG[52]
            - 0.09115833 * h[120] / H_AVG[120]
            - 0.07736478 * h[24] / H_AVG[24]
            + 0.07714369 * h[104] / H_AVG[104]
            + 0.06145295 * h[81] / H_AVG[81]
            + 0.0340998 * h[78] / H_AVG[78]
            + 0.03196083 * h[115] / H_AVG[115]
            - 0.03123902 * h[124] / H_AVG[124]
            - 0.03111773 * h[83] / H_AVG[83]
            + 0.03107844 * h[16] / H_AVG[16]
            - 0.01007888 * h[90] / H_AVG[90]
            + 0.003258795 * h[123] / H_AVG[123]
            + 0.00284197 * h[70] / H_AVG[70]
            + 0.001102016 * h[25] / H_AVG[25]
            + 6.295536e-05 * h[0] / H_AVG[0]
            - 4.069678e-05 * h[99] / H_AVG[99]
            - 3.135605e-05 * h[21] / H_AVG[21]
            + 1.922494e-06 * h[84] / H_AVG[84]
            + 4.535559e-07 * h[55] / H_AVG[55]
            + 3.044057e-07 * h[79] / H_AVG[79]
            - 2.906035e-07 * h[105] / H_AVG[105]
            - 2.358252e-07 * h[106] / H_AVG[106]
            + 1.916907e-07 * h[113] / H_AVG[113]
            - 1.820417e-07 * h[34] / H_AVG[34]
            + 1.634616e-07 * h[73] / H_AVG[73]
            + 1.518084e-07 * h[10] / H_AVG[10]
            + 1.446656e-07 * h[11] / H_AVG[11]
            - 1.232105e-07 * h[6] / H_AVG[6]
            + 1.175752e-07 * h[86] / H_AVG[86]
            - 1.118053e-07 * h[32] / H_AVG[32]
            + 1.027624e-07 * h[17] / H_AVG[17]
            + 9.286787e-08 * h[38] / H_AVG[38]
            - 7.626364e-08 * h[42] / H_AVG[42]
            - 6.796655e-08 * h[54] / H_AVG[54]
            - 6.459988e-08 * h[29] / H_AVG[29]
            + 5.856813e-08 * h[118] / H_AVG[118]
            + 4.571823e-08 * h[48] / H_AVG[48]
            - 4.363086e-08 * h[45] / H_AVG[45]
            - 4.213702e-08 * h[39] / H_AVG[39]
            + 3.884913e-08 * h[27] / H_AVG[27]
            + 3.868893e-08 * h[63] / H_AVG[63]
            + 3.827619e-08 * h[30] / H_AVG[30]
            + 3.74821e-08 * h[43] / H_AVG[43]
            + 3.486586e-08 * h[121] / H_AVG[121]
            - 3.424705e-08 * h[80] / H_AVG[80]
            - 3.134594e-08 * h[126] / H_AVG[126]
            + 3.025972e-08 * h[60] / H_AVG[60]
            - 2.980321e-08 * h[61] / H_AVG[61]
            - 2.829949e-08 * h[102] / H_AVG[102]
            - 2.825705e-08 * h[107] / H_AVG[107]
            - 2.781777e-08 * h[7] / H_AVG[7]
            + 2.50588e-08 * h[92] / H_AVG[92]
            + 2.450641e-08 * h[35] / H_AVG[35]
            + 2.383225e-08 * h[15] / H_AVG[15]
            - 2.301571e-08 * h[59] / H_AVG[59]
            - 2.220345e-08 * h[101] / H_AVG[101]
            - 2.122664e-08 * h[28] / H_AVG[28]
            + 2.058399e-08 * h[103] / H_AVG[103]
            + 1.937365e-08 * h[69] / H_AVG[69]
            + 1.880258e-08 * h[67] / H_AVG[67]
            + 1.821281e-08 * h[1] / H_AVG[1]
            + 1.807797e-08 * h[100] / H_AVG[100]
            - 1.660345e-08 * h[110] / H_AVG[110]
            - 1.60655e-08 * h[3] / H_AVG[3]
            + 1.543225e-08 * h[37] / H_AVG[37]
            - 1.511754e-08 * h[50] / H_AVG[50]
            + 1.421021e-08 * h[44] / H_AVG[44]
            + 1.419894e-08 * h[36] / H_AVG[36]
            - 1.395752e-08 * h[12] / H_AVG[12]
            - 1.327787e-08 * h[114] / H_AVG[114]
            - 1.316405e-08 * h[2] / H_AVG[2]
            + 1.268715e-08 * h[58] / H_AVG[58]
            - 1.263402e-08 * h[112] / H_AVG[112]
            + 1.236569e-08 * h[76] / H_AVG[76]
            - 1.192027e-08 * h[13] / H_AVG[13]
            - 1.157609e-08 * h[19] / H_AVG[19]
            - 1.136235e-08 * h[111] / H_AVG[111]
            + 1.08291e-08 * h[74] / H_AVG[74]
            + 1.027593e-08 * h[108] / H_AVG[108]
            + 9.659636e-09 * h[5] / H_AVG[5]
            - 9.233162e-09 * h[8] / H_AVG[8]
            - 8.853584e-09 * h[91] / H_AVG[91]
            - 8.253902e-09 * h[4] / H_AVG[4]
            - 7.906789e-09 * h[117] / H_AVG[117]
            + 6.681374e-09 * h[53] / H_AVG[53]
            - 6.607282e-09 * h[66] / H_AVG[66]
            - 6.013521e-09 * h[33] / H_AVG[33]
            - 5.842577e-09 * h[65] / H_AVG[65]
            + 5.40279e-09 * h[82] / H_AVG[82]
            - 5.077183e-09 * h[31] / H_AVG[31]
            - 4.665598e-09 * h[71] / H_AVG[71]
            + 4.333441e-09 * h[94] / H_AVG[94]
            + 4.185287e-09 * h[9] / H_AVG[9]
            + 3.67884e-09 * h[26] / H_AVG[26]
            - 3.296856e-09 * h[14] / H_AVG[14]
            + 3.265875e-09 * h[57] / H_AVG[57]
            + 3.086708e-09 * h[49] / H_AVG[49]
            + 3.015386e-09 * h[77] / H_AVG[77]
            - 3.009239e-09 * h[22] / H_AVG[22]
            + 2.791695e-09 * h[122] / H_AVG[122]
            + 2.256823e-09 * h[72] / H_AVG[72]
            + 2.161512e-09 * h[51] / H_AVG[51]
            - 2.127957e-09 * h[23] / H_AVG[23]
            - 1.469563e-09 * h[89] / H_AVG[89]
            + 1.435865e-09 * h[125] / H_AVG[125]
            - 1.348022e-09 * h[75] / H_AVG[75]
            - 1.166523e-09 * h[87] / H_AVG[87]
            + 1.162648e-09 * h[93] / H_AVG[93]
            - 1.045592e-09 * h[40] / H_AVG[40]
            - 9.715622e-10 * h[20] / H_AVG[20]
            + 8.681041e-10 * h[119] / H_AVG[119]
            - 7.941053e-10 * h[62] / H_AVG[62]
            - 7.140385e-10 * h[95] / H_AVG[95]
            - 6.617547e-10 * h[96] / H_AVG[96]
            + 3.901128e-10 * h[116] / H_AVG[116]
            + 3.725449e-10 * h[56] / H_AVG[56]
            + 2.590058e-10 * h[85] / H_AVG[85]
            + 1.541708e-10 * h[109] / H_AVG[109]
            - 1.342289e-10 * h[98] / H_AVG[98]
            - 1.072136e-10 * h[88] / H_AVG[88]
            - 7.226755e-11 * h[46] / H_AVG[46]
            - 6.822134e-11 * h[47] / H_AVG[47]
            - 4.207096e-11 * h[64] / H_AVG[64]
            + 1.63926e-11 * h[127] / H_AVG[127]
            + 1.447963e-12 * h[41] / H_AVG[41]
        ),
        -0.4064285 + T[1] * (   # class Hbb
            + 0.2855486 * h[16] / H_AVG[16]
            + 0.1914757 * h[120] / H_AVG[120]
            - 0.09152797 * h[68] / H_AVG[68]
            + 0.07704282 * h[18] / H_AVG[18]
            - 0.06786438 * h[24] / H_AVG[24]
            - 0.06724933 * h[78] / H_AVG[78]
            - 0.05886275 * h[97] / H_AVG[97]
            - 0.05876017 * h[52] / H_AVG[52]
            + 0.0425456 * h[115] / H_AVG[115]
            + 0.02095784 * h[81] / H_AVG[81]
            - 0.01004933 * h[70] / H_AVG[70]
            - 0.007876461 * h[25] / H_AVG[25]
            + 0.007250094 * h[83] / H_AVG[83]
            - 0.005138537 * h[90] / H_AVG[90]
            - 0.004392953 * h[104] / H_AVG[104]
            + 0.003010546 * h[124] / H_AVG[124]
            + 0.0002892622 * h[123] / H_AVG[123]
            + 0.0001001956 * h[0] / H_AVG[0]
            - 2.889928e-05 * h[99] / H_AVG[99]
            - 2.443446e-05 * h[21] / H_AVG[21]
            + 1.347241e-06 * h[84] / H_AVG[84]
            + 3.179154e-07 * h[55] / H_AVG[55]
            + 2.133034e-07 * h[79] / H_AVG[79]
            - 2.03701e-07 * h[105] / H_AVG[105]
            - 1.652814e-07 * h[106] / H_AVG[106]
            + 1.343082e-07 * h[113] / H_AVG[113]
            - 1.275962e-07 * h[34] / H_AVG[34]
            + 1.14549e-07 * h[73] / H_AVG[73]
            + 1.064201e-07 * h[10] / H_AVG[10]
            + 1.01443e-07 * h[11] / H_AVG[11]
            - 8.638372e-08 * h[6] / H_AVG[6]
            + 8.239686e-08 * h[86] / H_AVG[86]
            - 7.834575e-08 * h[32] / H_AVG[32]
            + 7.201924e-08 * h[17] / H_AVG[17]
            + 6.51136e-08 * h[38] / H_AVG[38]
            - 5.346075e-08 * h[42] / H_AVG[42]
            - 4.763684e-08 * h[54] / H_AVG[54]
            - 4.527496e-08 * h[29] / H_AVG[29]
            + 4.105018e-08 * h[118] / H_AVG[118]
            + 3.204504e-08 * h[48] / H_AVG[48]
            - 3.057563e-08 * h[45] / H_AVG[45]
            - 2.953817e-08 * h[39] / H_AVG[39]
            + 2.723609e-08 * h[27] / H_AVG[27]
            + 2.711861e-08 * h[63] / H_AVG[63]
            + 2.682733e-08 * h[30] / H_AVG[30]
            + 2.62745e-08 * h[43] / H_AVG[43]
            + 2.444186e-08 * h[121] / H_AVG[121]
            - 2.400058e-08 * h[80] / H_AVG[80]
            - 2.196946e-08 * h[126] / H_AVG[126]
            + 2.121388e-08 * h[60] / H_AVG[60]
            - 2.089176e-08 * h[61] / H_AVG[61]
            - 1.983693e-08 * h[102] / H_AVG[102]
            - 1.979158e-08 * h[107] / H_AVG[107]
            - 1.950045e-08 * h[7] / H_AVG[7]
            + 1.756583e-08 * h[92] / H_AVG[92]
            + 1.717798e-08 * h[35] / H_AVG[35]
            + 1.67035e-08 * h[15] / H_AVG[15]
            - 1.613447e-08 * h[59] / H_AVG[59]
            - 1.556539e-08 * h[101] / H_AVG[101]
            - 1.487524e-08 * h[28] / H_AVG[28]
            + 1.442729e-08 * h[103] / H_AVG[103]
            + 1.357727e-08 * h[69] / H_AVG[69]
            + 1.317756e-08 * h[67] / H_AVG[67]
            + 1.276735e-08 * h[1] / H_AVG[1]
            + 1.267444e-08 * h[100] / H_AVG[100]
            - 1.164084e-08 * h[110] / H_AVG[110]
            - 1.126251e-08 * h[3] / H_AVG[3]
            + 1.081584e-08 * h[37] / H_AVG[37]
            - 1.058879e-08 * h[50] / H_AVG[50]
            + 9.96235e-09 * h[44] / H_AVG[44]
            + 9.95361e-09 * h[36] / H_AVG[36]
            - 9.780721e-09 * h[12] / H_AVG[12]
            - 9.302999e-09 * h[114] / H_AVG[114]
            - 9.221281e-09 * h[2] / H_AVG[2]
            + 8.893754e-09 * h[58] / H_AVG[58]
            - 8.855025e-09 * h[112] / H_AVG[112]
            + 8.66715e-09 * h[76] / H_AVG[76]
            - 8.356039e-09 * h[13] / H_AVG[13]
            - 8.122152e-09 * h[19] / H_AVG[19]
            - 7.964595e-09 * h[111] / H_AVG[111]
            + 7.590714e-09 * h[74] / H_AVG[74]
            + 7.20603e-09 * h[108] / H_AVG[108]
            + 6.771463e-09 * h[5] / H_AVG[5]
            - 6.47202e-09 * h[8] / H_AVG[8]
            - 6.206028e-09 * h[91] / H_AVG[91]
            - 5.785961e-09 * h[4] / H_AVG[4]
            - 5.542245e-09 * h[117] / H_AVG[117]
            + 4.68072e-09 * h[53] / H_AVG[53]
            - 4.628085e-09 * h[66] / H_AVG[66]
            - 4.216158e-09 * h[33] / H_AVG[33]
            - 4.094692e-09 * h[65] / H_AVG[65]
            + 3.786725e-09 * h[82] / H_AVG[82]
            - 3.55994e-09 * h[31] / H_AVG[31]
            - 3.268915e-09 * h[71] / H_AVG[71]
            + 3.036485e-09 * h[94] / H_AVG[94]
            + 2.933309e-09 * h[9] / H_AVG[9]
            + 2.576951e-09 * h[26] / H_AVG[26]
            - 2.310696e-09 * h[14] / H_AVG[14]
            + 2.288893e-09 * h[57] / H_AVG[57]
            + 2.163785e-09 * h[49] / H_AVG[49]
            + 2.114621e-09 * h[77] / H_AVG[77]
            - 2.109714e-09 * h[22] / H_AVG[22]
            + 1.956519e-09 * h[122] / H_AVG[122]
            + 1.582029e-09 * h[72] / H_AVG[72]
            + 1.515006e-09 * h[51] / H_AVG[51]
            - 1.491472e-09 * h[23] / H_AVG[23]
            - 1.03033e-09 * h[89] / H_AVG[89]
            + 1.006946e-09 * h[125] / H_AVG[125]
            - 9.446466e-10 * h[75] / H_AVG[75]
            - 8.173175e-10 * h[87] / H_AVG[87]
            + 8.145295e-10 * h[93] / H_AVG[93]
            - 7.328704e-10 * h[40] / H_AVG[40]
            - 6.812009e-10 * h[20] / H_AVG[20]
            + 6.078371e-10 * h[119] / H_AVG[119]
            - 5.567064e-10 * h[62] / H_AVG[62]
            - 5.004544e-10 * h[95] / H_AVG[95]
            - 4.637435e-10 * h[96] / H_AVG[96]
            + 2.73513e-10 * h[116] / H_AVG[116]
            + 2.610451e-10 * h[56] / H_AVG[56]
            + 1.814416e-10 * h[85] / H_AVG[85]
            + 1.080906e-10 * h[109] / H_AVG[109]
            - 9.410473e-11 * h[98] / H_AVG[98]
            - 7.519774e-11 * h[88] / H_AVG[88]
            - 5.062925e-11 * h[46] / H_AVG[46]
            - 4.783552e-11 * h[47] / H_AVG[47]
            - 2.597196e-11 * h[64] / H_AVG[64]
            + 1.14575e-11 * h[127] / H_AVG[127]
            + 1.013116e-12 * h[41] / H_AVG[41]
        ),
        0.1415668 + T[2] * (   # class Hcc
            - 0.1956793 * h[83] / H_AVG[83]
            + 0.1885409 * h[18] / H_AVG[18]
            - 0.1389774 * h[24] / H_AVG[24]
            - 0.07213792 * h[81] / H_AVG[81]
            + 0.06731688 * h[124] / H_AVG[124]
            - 0.05921737 * h[16] / H_AVG[16]
            + 0.05170841 * h[104] / H_AVG[104]
            + 0.05025328 * h[78] / H_AVG[78]
            - 0.048275 * h[97] / H_AVG[97]
            + 0.03251731 * h[120] / H_AVG[120]
            - 0.02382167 * h[68] / H_AVG[68]
            - 0.02348721 * h[52] / H_AVG[52]
            + 0.02207246 * h[115] / H_AVG[115]
            + 0.01333689 * h[70] / H_AVG[70]
            - 0.007386404 * h[25] / H_AVG[25]
            - 0.004771418 * h[123] / H_AVG[123]
            + 0.000223222 * h[90] / H_AVG[90]
            + 0.0002036038 * h[0] / H_AVG[0]
            - 4.054758e-05 * h[99] / H_AVG[99]
            - 2.693357e-05 * h[21] / H_AVG[21]
            + 1.905594e-06 * h[84] / H_AVG[84]
            + 4.497649e-07 * h[55] / H_AVG[55]
            + 3.017879e-07 * h[79] / H_AVG[79]
            - 2.880498e-07 * h[105] / H_AVG[105]
            - 2.338215e-07 * h[106] / H_AVG[106]
            + 1.90026e-07 * h[113] / H_AVG[113]
            - 1.804327e-07 * h[34] / H_AVG[34]
            + 1.620295e-07 * h[73] / H_AVG[73]
            + 1.50537e-07 * h[10] / H_AVG[10]
            + 1.434454e-07 * h[11] / H_AVG[11]
            - 1.221453e-07 * h[6] / H_AVG[6]
            + 1.165301e-07 * h[86] / H_AVG[86]
            - 1.10833e-07 * h[32] / H_AVG[32]
            + 1.019058e-07 * h[17] / H_AVG[17]
            + 9.204488e-08 * h[38] / H_AVG[38]
            - 7.559736e-08 * h[42] / H_AVG[42]
            - 6.738564e-08 * h[54] / H_AVG[54]
            - 6.405206e-08 * h[29] / H_AVG[29]
            + 5.806574e-08 * h[118] / H_AVG[118]
            + 4.532147e-08 * h[48] / H_AVG[48]
            - 4.324765e-08 * h[45] / H_AVG[45]
            - 4.177788e-08 * h[39] / H_AVG[39]
            + 3.852408e-08 * h[27] / H_AVG[27]
            + 3.836268e-08 * h[63] / H_AVG[63]
            + 3.793957e-08 * h[30] / H_AVG[30]
            + 3.71492e-08 * h[43] / H_AVG[43]
            + 3.45798e-08 * h[121] / H_AVG[121]
            - 3.395466e-08 * h[80] / H_AVG[80]
            - 3.106916e-08 * h[126] / H_AVG[126]
            + 3.000415e-08 * h[60] / H_AVG[60]
            - 2.954578e-08 * h[61] / H_AVG[61]
            - 2.805269e-08 * h[102] / H_AVG[102]
            - 2.798273e-08 * h[107] / H_AVG[107]
            - 2.759395e-08 * h[7] / H_AVG[7]
            + 2.484157e-08 * h[92] / H_AVG[92]
            + 2.429709e-08 * h[35] / H_AVG[35]
            + 2.362724e-08 * h[15] / H_AVG[15]
            - 2.281847e-08 * h[59] / H_AVG[59]
            - 2.200435e-08 * h[101] / H_AVG[101]
            - 2.104337e-08 * h[28] / H_AVG[28]
            + 2.040811e-08 * h[103] / H_AVG[103]
            + 1.920616e-08 * h[69] / H_AVG[69]
            + 1.864195e-08 * h[67] / H_AVG[67]
            + 1.806112e-08 * h[1] / H_AVG[1]
            + 1.792617e-08 * h[100] / H_AVG[100]
            - 1.646466e-08 * h[110] / H_AVG[110]
            - 1.592997e-08 * h[3] / H_AVG[3]
            + 1.529705e-08 * h[37] / H_AVG[37]
            - 1.498223e-08 * h[50] / H_AVG[50]
            + 1.409009e-08 * h[44] / H_AVG[44]
            + 1.408046e-08 * h[36] / H_AVG[36]
            - 1.383308e-08 * h[12] / H_AVG[12]
            - 1.315995e-08 * h[114] / H_AVG[114]
            - 1.304988e-08 * h[2] / H_AVG[2]
            + 1.258195e-08 * h[58] / H_AVG[58]
            - 1.252928e-08 * h[112] / H_AVG[112]
            + 1.226079e-08 * h[76] / H_AVG[76]
            - 1.181828e-08 * h[13] / H_AVG[13]
            - 1.148277e-08 * h[19] / H_AVG[19]
            - 1.126224e-08 * h[111] / H_AVG[111]
            + 1.073589e-08 * h[74] / H_AVG[74]
            + 1.018923e-08 * h[108] / H_AVG[108]
            + 9.581833e-09 * h[5] / H_AVG[5]
            - 9.156301e-09 * h[8] / H_AVG[8]
            - 8.780989e-09 * h[91] / H_AVG[91]
            - 8.18305e-09 * h[4] / H_AVG[4]
            - 7.83841e-09 * h[117] / H_AVG[117]
            + 6.625177e-09 * h[53] / H_AVG[53]
            - 6.550932e-09 * h[66] / H_AVG[66]
            - 5.961849e-09 * h[33] / H_AVG[33]
            - 5.793803e-09 * h[65] / H_AVG[65]
            + 5.357225e-09 * h[82] / H_AVG[82]
            - 5.03107e-09 * h[31] / H_AVG[31]
            - 4.62333e-09 * h[71] / H_AVG[71]
            + 4.29658e-09 * h[94] / H_AVG[94]
            + 4.148571e-09 * h[9] / H_AVG[9]
            + 3.642829e-09 * h[26] / H_AVG[26]
            - 3.269993e-09 * h[14] / H_AVG[14]
            + 3.23864e-09 * h[57] / H_AVG[57]
            + 3.059493e-09 * h[49] / H_AVG[49]
            + 2.98898e-09 * h[77] / H_AVG[77]
            - 2.985011e-09 * h[22] / H_AVG[22]
            + 2.768616e-09 * h[122] / H_AVG[122]
            + 2.236627e-09 * h[72] / H_AVG[72]
            + 2.144144e-09 * h[51] / H_AVG[51]
            - 2.110419e-09 * h[23] / H_AVG[23]
            - 1.456706e-09 * h[89] / H_AVG[89]
            + 1.423513e-09 * h[125] / H_AVG[125]
            - 1.336231e-09 * h[75] / H_AVG[75]
            - 1.156183e-09 * h[87] / H_AVG[87]
            + 1.151978e-09 * h[93] / H_AVG[93]
            - 1.036411e-09 * h[40] / H_AVG[40]
            - 9.636991e-10 * h[20] / H_AVG[20]
            + 8.604203e-10 * h[119] / H_AVG[119]
            - 7.873407e-10 * h[62] / H_AVG[62]
            - 7.076613e-10 * h[95] / H_AVG[95]
            - 6.558229e-10 * h[96] / H_AVG[96]
            + 3.869689e-10 * h[116] / H_AVG[116]
            + 3.69454e-10 * h[56] / H_AVG[56]
            + 2.569261e-10 * h[85] / H_AVG[85]
            + 1.528285e-10 * h[109] / H_AVG[109]
            - 1.330756e-10 * h[98] / H_AVG[98]
            - 1.062151e-10 * h[88] / H_AVG[88]
            - 7.164097e-11 * h[46] / H_AVG[46]
            - 6.760824e-11 * h[47] / H_AVG[47]
            - 4.268883e-11 * h[64] / H_AVG[64]
            + 1.62431e-11 * h[127] / H_AVG[127]
            + 1.437049e-12 * h[41] / H_AVG[41]
        ),
        0.1286925 + T[3] * (   # class Hgg
            - 0.2117048 * h[104] / H_AVG[104]
            - 0.1561191 * h[68] / H_AVG[68]
            - 0.1026862 * h[18] / H_AVG[18]
            - 0.1004711 * h[16] / H_AVG[16]
            + 0.08499828 * h[52] / H_AVG[52]
            - 0.07133229 * h[97] / H_AVG[97]
            + 0.06000778 * h[124] / H_AVG[124]
            + 0.0536114 * h[115] / H_AVG[115]
            - 0.04823646 * h[24] / H_AVG[24]
            - 0.04477199 * h[81] / H_AVG[81]
            - 0.02094382 * h[78] / H_AVG[78]
            + 0.01552951 * h[83] / H_AVG[83]
            + 0.01182232 * h[120] / H_AVG[120]
            + 0.007854852 * h[70] / H_AVG[70]
            - 0.004655075 * h[25] / H_AVG[25]
            + 0.004145062 * h[90] / H_AVG[90]
            - 0.0008166177 * h[123] / H_AVG[123]
            + 0.000225257 * h[0] / H_AVG[0]
            - 3.648843e-05 * h[99] / H_AVG[99]
            - 2.618535e-05 * h[21] / H_AVG[21]
            + 1.724359e-06 * h[84] / H_AVG[84]
            + 4.068784e-07 * h[55] / H_AVG[55]
            + 2.730551e-07 * h[79] / H_AVG[79]
            - 2.606823e-07 * h[105] / H_AVG[105]
            - 2.115397e-07 * h[106] / H_AVG[106]
            + 1.719526e-07 * h[113] / H_AVG[113]
            - 1.633088e-07 * h[34] / H_AVG[34]
            + 1.466215e-07 * h[73] / H_AVG[73]
            + 1.361756e-07 * h[10] / H_AVG[10]
            + 1.297708e-07 * h[11] / H_AVG[11]
            - 1.105264e-07 * h[6] / H_AVG[6]
            + 1.054585e-07 * h[86] / H_AVG[86]
            - 1.00312e-07 * h[32] / H_AVG[32]
            + 9.220888e-08 * h[17] / H_AVG[17]
            + 8.333475e-08 * h[38] / H_AVG[38]
            - 6.840607e-08 * h[42] / H_AVG[42]
            - 6.096997e-08 * h[54] / H_AVG[54]
            - 5.794402e-08 * h[29] / H_AVG[29]
            + 5.254845e-08 * h[118] / H_AVG[118]
            + 4.101324e-08 * h[48] / H_AVG[48]
            - 3.913983e-08 * h[45] / H_AVG[45]
            - 3.779725e-08 * h[39] / H_AVG[39]
            + 3.485612e-08 * h[27] / H_AVG[27]
            + 3.470357e-08 * h[63] / H_AVG[63]
            + 3.432992e-08 * h[30] / H_AVG[30]
            + 3.362156e-08 * h[43] / H_AVG[43]
            + 3.128348e-08 * h[121] / H_AVG[121]
            - 3.071871e-08 * h[80] / H_AVG[80]
            - 2.811805e-08 * h[126] / H_AVG[126]
            + 2.714468e-08 * h[60] / H_AVG[60]
            - 2.674062e-08 * h[61] / H_AVG[61]
            - 2.53864e-08 * h[102] / H_AVG[102]
            - 2.533902e-08 * h[107] / H_AVG[107]
            - 2.495401e-08 * h[7] / H_AVG[7]
            + 2.247747e-08 * h[92] / H_AVG[92]
            + 2.198472e-08 * h[35] / H_AVG[35]
            + 2.137503e-08 * h[15] / H_AVG[15]
            - 2.064982e-08 * h[59] / H_AVG[59]
            - 1.991489e-08 * h[101] / H_AVG[101]
            - 1.904119e-08 * h[28] / H_AVG[28]
            + 1.846551e-08 * h[103] / H_AVG[103]
            + 1.73774e-08 * h[69] / H_AVG[69]
            + 1.686325e-08 * h[67] / H_AVG[67]
            + 1.633796e-08 * h[1] / H_AVG[1]
            + 1.621646e-08 * h[100] / H_AVG[100]
            - 1.489333e-08 * h[110] / H_AVG[110]
            - 1.441219e-08 * h[3] / H_AVG[3]
            + 1.384197e-08 * h[37] / H_AVG[37]
            - 1.356289e-08 * h[50] / H_AVG[50]
            + 1.274796e-08 * h[44] / H_AVG[44]
            + 1.273695e-08 * h[36] / H_AVG[36]
            - 1.252107e-08 * h[12] / H_AVG[12]
            - 1.190918e-08 * h[114] / H_AVG[114]
            - 1.180717e-08 * h[2] / H_AVG[2]
            + 1.138537e-08 * h[58] / H_AVG[58]
            - 1.133896e-08 * h[112] / H_AVG[112]
            + 1.109659e-08 * h[76] / H_AVG[76]
            - 1.069368e-08 * h[13] / H_AVG[13]
            - 1.039145e-08 * h[19] / H_AVG[19]
            - 1.019254e-08 * h[111] / H_AVG[111]
            + 9.714556e-09 * h[74] / H_AVG[74]
            + 9.219097e-09 * h[108] / H_AVG[108]
            + 8.669239e-09 * h[5] / H_AVG[5]
            - 8.285204e-09 * h[8] / H_AVG[8]
            - 7.941427e-09 * h[91] / H_AVG[91]
            - 7.405008e-09 * h[4] / H_AVG[4]
            - 7.092288e-09 * h[117] / H_AVG[117]
            + 5.991812e-09 * h[53] / H_AVG[53]
            - 5.925847e-09 * h[66] / H_AVG[66]
            - 5.395107e-09 * h[33] / H_AVG[33]
            - 5.234865e-09 * h[65] / H_AVG[65]
            + 4.846268e-09 * h[82] / H_AVG[82]
            - 4.553527e-09 * h[31] / H_AVG[31]
            - 4.185155e-09 * h[71] / H_AVG[71]
            + 3.88651e-09 * h[94] / H_AVG[94]
            + 3.754303e-09 * h[9] / H_AVG[9]
            + 3.298268e-09 * h[26] / H_AVG[26]
            - 2.957276e-09 * h[14] / H_AVG[14]
            + 2.930252e-09 * h[57] / H_AVG[57]
            + 2.768849e-09 * h[49] / H_AVG[49]
            + 2.705625e-09 * h[77] / H_AVG[77]
            - 2.698959e-09 * h[22] / H_AVG[22]
            + 2.503914e-09 * h[122] / H_AVG[122]
            + 2.024707e-09 * h[72] / H_AVG[72]
            + 1.940251e-09 * h[51] / H_AVG[51]
            - 1.90863e-09 * h[23] / H_AVG[23]
            - 1.31839e-09 * h[89] / H_AVG[89]
            + 1.28801e-09 * h[125] / H_AVG[125]
            - 1.209334e-09 * h[75] / H_AVG[75]
            - 1.04601e-09 * h[87] / H_AVG[87]
            + 1.042408e-09 * h[93] / H_AVG[93]
            - 9.379157e-10 * h[40] / H_AVG[40]
            - 8.719242e-10 * h[20] / H_AVG[20]
            + 7.784286e-10 * h[119] / H_AVG[119]
            - 7.123668e-10 * h[62] / H_AVG[62]
            - 6.403824e-10 * h[95] / H_AVG[95]
            - 5.93517e-10 * h[96] / H_AVG[96]
            + 3.499811e-10 * h[116] / H_AVG[116]
            + 3.340881e-10 * h[56] / H_AVG[56]
            + 2.324533e-10 * h[85] / H_AVG[85]
            + 1.383097e-10 * h[109] / H_AVG[109]
            - 1.204094e-10 * h[98] / H_AVG[98]
            - 9.626539e-11 * h[88] / H_AVG[88]
            - 6.481516e-11 * h[46] / H_AVG[46]
            - 6.11928e-11 * h[47] / H_AVG[47]
            - 3.703996e-11 * h[64] / H_AVG[64]
            + 1.469964e-11 * h[127] / H_AVG[127]
            + 1.298471e-12 * h[41] / H_AVG[41]
        ),
        -0.07702489 + T[4] * (   # class H4q
            - 0.3012226 * h[115] / H_AVG[115]
            - 0.09590442 * h[104] / H_AVG[104]
            - 0.08719999 * h[16] / H_AVG[16]
            + 0.07890961 * h[52] / H_AVG[52]
            - 0.07315705 * h[24] / H_AVG[24]
            - 0.07251634 * h[78] / H_AVG[78]
            - 0.06655324 * h[97] / H_AVG[97]
            + 0.05458051 * h[124] / H_AVG[124]
            - 0.04501453 * h[18] / H_AVG[18]
            + 0.04498642 * h[68] / H_AVG[68]
            - 0.04311362 * h[81] / H_AVG[81]
            + 0.01200332 * h[120] / H_AVG[120]
            - 0.01025411 * h[83] / H_AVG[83]
            + 0.008935133 * h[70] / H_AVG[70]
            - 0.004344962 * h[25] / H_AVG[25]
            - 0.0007717135 * h[90] / H_AVG[90]
            + 0.0002457229 * h[123] / H_AVG[123]
            + 0.0002231447 * h[0] / H_AVG[0]
            - 3.25653e-05 * h[99] / H_AVG[99]
            - 2.62146e-05 * h[21] / H_AVG[21]
            + 1.541141e-06 * h[84] / H_AVG[84]
            + 3.638104e-07 * h[55] / H_AVG[55]
            + 2.440823e-07 * h[79] / H_AVG[79]
            - 2.330147e-07 * h[105] / H_AVG[105]
            - 1.891558e-07 * h[106] / H_AVG[106]
            + 1.536654e-07 * h[113] / H_AVG[113]
            - 1.459767e-07 * h[34] / H_AVG[34]
            + 1.310665e-07 * h[73] / H_AVG[73]
            + 1.217409e-07 * h[10] / H_AVG[10]
            + 1.159919e-07 * h[11] / H_AVG[11]
            - 9.880867e-08 * h[6] / H_AVG[6]
            + 9.426586e-08 * h[86] / H_AVG[86]
            - 8.965288e-08 * h[32] / H_AVG[32]
            + 8.242915e-08 * h[17] / H_AVG[17]
            + 7.449096e-08 * h[38] / H_AVG[38]
            - 6.114992e-08 * h[42] / H_AVG[42]
            - 5.451115e-08 * h[54] / H_AVG[54]
            - 5.180741e-08 * h[29] / H_AVG[29]
            + 4.697513e-08 * h[118] / H_AVG[118]
            + 3.665742e-08 * h[48] / H_AVG[48]
            - 3.498562e-08 * h[45] / H_AVG[45]
            - 3.378702e-08 * h[39] / H_AVG[39]
            + 3.115863e-08 * h[27] / H_AVG[27]
            + 3.102079e-08 * h[63] / H_AVG[63]
            + 3.069005e-08 * h[30] / H_AVG[30]
            + 3.005439e-08 * h[43] / H_AVG[43]
            + 2.796374e-08 * h[121] / H_AVG[121]
            - 2.746305e-08 * h[80] / H_AVG[80]
            - 2.513775e-08 * h[126] / H_AVG[126]
            + 2.427361e-08 * h[60] / H_AVG[60]
            - 2.389702e-08 * h[61] / H_AVG[61]
            - 2.269111e-08 * h[102] / H_AVG[102]
            - 2.265084e-08 * h[107] / H_AVG[107]
            - 2.230886e-08 * h[7] / H_AVG[7]
            + 2.009116e-08 * h[92] / H_AVG[92]
            + 1.965439e-08 * h[35] / H_AVG[35]
            + 1.910982e-08 * h[15] / H_AVG[15]
            - 1.84528e-08 * h[59] / H_AVG[59]
            - 1.780191e-08 * h[101] / H_AVG[101]
            - 1.702205e-08 * h[28] / H_AVG[28]
            + 1.650566e-08 * h[103] / H_AVG[103]
            + 1.553284e-08 * h[69] / H_AVG[69]
            + 1.507713e-08 * h[67] / H_AVG[67]
            + 1.460441e-08 * h[1] / H_AVG[1]
            + 1.449508e-08 * h[100] / H_AVG[100]
            - 1.331573e-08 * h[110] / H_AVG[110]
            - 1.288069e-08 * h[3] / H_AVG[3]
            + 1.237342e-08 * h[37] / H_AVG[37]
            - 1.211427e-08 * h[50] / H_AVG[50]
            + 1.139509e-08 * h[44] / H_AVG[44]
            + 1.138532e-08 * h[36] / H_AVG[36]
            - 1.118962e-08 * h[12] / H_AVG[12]
            - 1.06467e-08 * h[114] / H_AVG[114]
            - 1.055647e-08 * h[2] / H_AVG[2]
            + 1.017652e-08 * h[58] / H_AVG[58]
            - 1.013399e-08 * h[112] / H_AVG[112]
            + 9.918635e-09 * h[76] / H_AVG[76]
            - 9.557829e-09 * h[13] / H_AVG[13]
            - 9.285732e-09 * h[19] / H_AVG[19]
            - 9.110491e-09 * h[111] / H_AVG[111]
            + 8.682718e-09 * h[74] / H_AVG[74]
            + 8.239681e-09 * h[108] / H_AVG[108]
            + 7.749976e-09 * h[5] / H_AVG[5]
            - 7.406248e-09 * h[8] / H_AVG[8]
            - 7.102713e-09 * h[91] / H_AVG[91]
            - 6.618934e-09 * h[4] / H_AVG[4]
            - 6.34087e-09 * h[117] / H_AVG[117]
            + 5.355325e-09 * h[53] / H_AVG[53]
            - 5.297293e-09 * h[66] / H_AVG[66]
            - 4.821864e-09 * h[33] / H_AVG[33]
            - 4.680429e-09 * h[65] / H_AVG[65]
            + 4.33334e-09 * h[82] / H_AVG[82]
            - 4.070604e-09 * h[31] / H_AVG[31]
            - 3.741009e-09 * h[71] / H_AVG[71]
            + 3.474513e-09 * h[94] / H_AVG[94]
            + 3.355642e-09 * h[9] / H_AVG[9]
            + 2.948207e-09 * h[26] / H_AVG[26]
            - 2.643077e-09 * h[14] / H_AVG[14]
            + 2.619259e-09 * h[57] / H_AVG[57]
            + 2.474983e-09 * h[49] / H_AVG[49]
            + 2.420497e-09 * h[77] / H_AVG[77]
            - 2.414251e-09 * h[22] / H_AVG[22]
            + 2.238722e-09 * h[122] / H_AVG[122]
            + 1.809851e-09 * h[72] / H_AVG[72]
            + 1.734056e-09 * h[51] / H_AVG[51]
            - 1.705877e-09 * h[23] / H_AVG[23]
            - 1.178397e-09 * h[89] / H_AVG[89]
            + 1.151558e-09 * h[125] / H_AVG[125]
            - 1.080944e-09 * h[75] / H_AVG[75]
            - 9.351304e-10 * h[87] / H_AVG[87]
            + 9.323564e-10 * h[93] / H_AVG[93]
            - 8.384332e-10 * h[40] / H_AVG[40]
            - 7.79352e-10 * h[20] / H_AVG[20]
            + 6.962055e-10 * h[119] / H_AVG[119]
            - 6.369661e-10 * h[62] / H_AVG[62]
            - 5.726612e-10 * h[95] / H_AVG[95]
            - 5.30657e-10 * h[96] / H_AVG[96]
            + 3.129015e-10 * h[116] / H_AVG[116]
            + 2.988567e-10 * h[56] / H_AVG[56]
            + 2.076842e-10 * h[85] / H_AVG[85]
            + 1.23667e-10 * h[109] / H_AVG[109]
            - 1.076371e-10 * h[98] / H_AVG[98]
            - 8.600421e-11 * h[88] / H_AVG[88]
            - 5.794625e-11 * h[46] / H_AVG[46]
            - 5.470539e-11 * h[47] / H_AVG[47]
            - 3.380516e-11 * h[64] / H_AVG[64]
            + 1.312384e-11 * h[127] / H_AVG[127]
            + 1.1618e-12 * h[41] / H_AVG[41]
        ),
        -0.9789527 + T[5] * (   # class Hqql
            + 0.1723754 * h[120] / H_AVG[120]
            + 0.1310448 * h[24] / H_AVG[24]
            + 0.1153126 * h[97] / H_AVG[97]
            + 0.103687 * h[18] / H_AVG[18]
            + 0.08521249 * h[52] / H_AVG[52]
            - 0.08193832 * h[124] / H_AVG[124]
            - 0.0715294 * h[115] / H_AVG[115]
            + 0.06804471 * h[83] / H_AVG[83]
            + 0.05901519 * h[78] / H_AVG[78]
            + 0.02129823 * h[70] / H_AVG[70]
            + 0.01838837 * h[90] / H_AVG[90]
            + 0.01545347 * h[68] / H_AVG[68]
            - 0.01472009 * h[81] / H_AVG[81]
            + 0.01402011 * h[104] / H_AVG[104]
            + 0.01230008 * h[25] / H_AVG[25]
            - 0.01011035 * h[123] / H_AVG[123]
            - 0.005145861 * h[16] / H_AVG[16]
            - 0.0003164239 * h[0] / H_AVG[0]
            + 6.569776e-05 * h[21] / H_AVG[21]
            - 1.875423e-05 * h[99] / H_AVG[99]
            + 8.762626e-07 * h[84] / H_AVG[84]
            + 2.067429e-07 * h[55] / H_AVG[55]
            + 1.386731e-07 * h[79] / H_AVG[79]
            - 1.324075e-07 * h[105] / H_AVG[105]
            - 1.075098e-07 * h[106] / H_AVG[106]
            + 8.738468e-08 * h[113] / H_AVG[113]
            - 8.296793e-08 * h[34] / H_AVG[34]
            + 7.447047e-08 * h[73] / H_AVG[73]
            + 6.914862e-08 * h[10] / H_AVG[10]
            + 6.59757e-08 * h[11] / H_AVG[11]
            - 5.611064e-08 * h[6] / H_AVG[6]
            + 5.355728e-08 * h[86] / H_AVG[86]
            - 5.094305e-08 * h[32] / H_AVG[32]
            + 4.685099e-08 * h[17] / H_AVG[17]
            + 4.230942e-08 * h[38] / H_AVG[38]
            - 3.474089e-08 * h[42] / H_AVG[42]
            - 3.099707e-08 * h[54] / H_AVG[54]
            - 2.943367e-08 * h[29] / H_AVG[29]
            + 2.670167e-08 * h[118] / H_AVG[118]
            + 2.082088e-08 * h[48] / H_AVG[48]
            - 1.988451e-08 * h[45] / H_AVG[45]
            - 1.918891e-08 * h[39] / H_AVG[39]
            + 1.770942e-08 * h[27] / H_AVG[27]
            + 1.763116e-08 * h[63] / H_AVG[63]
            + 1.74433e-08 * h[30] / H_AVG[30]
            + 1.706732e-08 * h[43] / H_AVG[43]
            + 1.588716e-08 * h[121] / H_AVG[121]
            - 1.560601e-08 * h[80] / H_AVG[80]
            - 1.430391e-08 * h[126] / H_AVG[126]
            + 1.378736e-08 * h[60] / H_AVG[60]
            - 1.357298e-08 * h[61] / H_AVG[61]
            - 1.289029e-08 * h[102] / H_AVG[102]
            - 1.285178e-08 * h[107] / H_AVG[107]
            - 1.267932e-08 * h[7] / H_AVG[7]
            + 1.141166e-08 * h[92] / H_AVG[92]
            + 1.11579e-08 * h[35] / H_AVG[35]
            + 1.085727e-08 * h[15] / H_AVG[15]
            - 1.049672e-08 * h[59] / H_AVG[59]
            - 1.01249e-08 * h[101] / H_AVG[101]
            - 9.653481e-09 * h[28] / H_AVG[28]
            + 9.383836e-09 * h[103] / H_AVG[103]
            + 8.8384e-09 * h[69] / H_AVG[69]
            + 8.582832e-09 * h[67] / H_AVG[67]
            + 8.29533e-09 * h[1] / H_AVG[1]
            + 8.232233e-09 * h[100] / H_AVG[100]
            - 7.560756e-09 * h[110] / H_AVG[110]
            - 7.325767e-09 * h[3] / H_AVG[3]
            + 7.032615e-09 * h[37] / H_AVG[37]
            - 6.880537e-09 * h[50] / H_AVG[50]
            + 6.474308e-09 * h[36] / H_AVG[36]
            + 6.471064e-09 * h[44] / H_AVG[44]
            - 6.366777e-09 * h[12] / H_AVG[12]
            - 6.050115e-09 * h[114] / H_AVG[114]
            - 6.005331e-09 * h[2] / H_AVG[2]
            + 5.788515e-09 * h[58] / H_AVG[58]
            - 5.770263e-09 * h[112] / H_AVG[112]
            + 5.640902e-09 * h[76] / H_AVG[76]
            - 5.425808e-09 * h[13] / H_AVG[13]
            - 5.279054e-09 * h[19] / H_AVG[19]
            - 5.172948e-09 * h[111] / H_AVG[111]
            + 4.933054e-09 * h[74] / H_AVG[74]
            + 4.673503e-09 * h[108] / H_AVG[108]
            + 4.399651e-09 * h[5] / H_AVG[5]
            - 4.212522e-09 * h[8] / H_AVG[8]
            - 4.04326e-09 * h[91] / H_AVG[91]
            - 3.763979e-09 * h[4] / H_AVG[4]
            - 3.604613e-09 * h[117] / H_AVG[117]
            + 3.045722e-09 * h[53] / H_AVG[53]
            - 3.00958e-09 * h[66] / H_AVG[66]
            - 2.733035e-09 * h[33] / H_AVG[33]
            - 2.668944e-09 * h[65] / H_AVG[65]
            + 2.466848e-09 * h[82] / H_AVG[82]
            - 2.306083e-09 * h[31] / H_AVG[31]
            - 2.123775e-09 * h[71] / H_AVG[71]
            + 1.973536e-09 * h[94] / H_AVG[94]
            + 1.906682e-09 * h[9] / H_AVG[9]
            + 1.683079e-09 * h[26] / H_AVG[26]
            - 1.503352e-09 * h[14] / H_AVG[14]
            + 1.49349e-09 * h[57] / H_AVG[57]
            + 1.404931e-09 * h[49] / H_AVG[49]
            + 1.374789e-09 * h[77] / H_AVG[77]
            - 1.371056e-09 * h[22] / H_AVG[22]
            + 1.275795e-09 * h[122] / H_AVG[122]
            + 1.027423e-09 * h[72] / H_AVG[72]
            + 9.889687e-10 * h[51] / H_AVG[51]
            - 9.711915e-10 * h[23] / H_AVG[23]
            - 6.660522e-10 * h[89] / H_AVG[89]
            + 6.531707e-10 * h[125] / H_AVG[125]
            - 6.130097e-10 * h[75] / H_AVG[75]
            + 5.299733e-10 * h[93] / H_AVG[93]
            - 5.289861e-10 * h[87] / H_AVG[87]
            - 4.762009e-10 * h[40] / H_AVG[40]
            - 4.427927e-10 * h[20] / H_AVG[20]
            + 3.968237e-10 * h[119] / H_AVG[119]
            - 3.609767e-10 * h[62] / H_AVG[62]
            - 3.252544e-10 * h[95] / H_AVG[95]
            - 3.008024e-10 * h[96] / H_AVG[96]
            + 1.775161e-10 * h[116] / H_AVG[116]
            + 1.704066e-10 * h[56] / H_AVG[56]
            + 1.188286e-10 * h[85] / H_AVG[85]
            + 7.003266e-11 * h[109] / H_AVG[109]
            - 6.098683e-11 * h[98] / H_AVG[98]
            - 4.855063e-11 * h[88] / H_AVG[88]
            - 3.313606e-11 * h[46] / H_AVG[46]
            - 3.100546e-11 * h[47] / H_AVG[47]
            - 1.483271e-11 * h[64] / H_AVG[64]
            + 7.720543e-12 * h[127] / H_AVG[127]
            + 6.869252e-13 * h[41] / H_AVG[41]
        ),
        0.3035426 + T[6] * (   # class Zqq
            + 0.1138172 * h[115] / H_AVG[115]
            - 0.1134212 * h[124] / H_AVG[124]
            + 0.1078247 * h[24] / H_AVG[24]
            + 0.1034816 * h[104] / H_AVG[104]
            - 0.09308895 * h[97] / H_AVG[97]
            - 0.08697555 * h[120] / H_AVG[120]
            + 0.08145254 * h[83] / H_AVG[83]
            + 0.07891667 * h[68] / H_AVG[68]
            + 0.05519476 * h[81] / H_AVG[81]
            - 0.05209542 * h[52] / H_AVG[52]
            - 0.04506755 * h[16] / H_AVG[16]
            - 0.01967545 * h[70] / H_AVG[70]
            + 0.01727487 * h[18] / H_AVG[18]
            + 0.009709522 * h[25] / H_AVG[25]
            - 0.008792603 * h[90] / H_AVG[90]
            - 0.008546451 * h[78] / H_AVG[78]
            + 0.004523828 * h[123] / H_AVG[123]
            - 5.929487e-05 * h[0] / H_AVG[0]
            - 4.210953e-05 * h[99] / H_AVG[99]
            - 3.362722e-05 * h[21] / H_AVG[21]
            + 1.983036e-06 * h[84] / H_AVG[84]
            + 4.681071e-07 * h[55] / H_AVG[55]
            + 3.140319e-07 * h[79] / H_AVG[79]
            - 2.998324e-07 * h[105] / H_AVG[105]
            - 2.433774e-07 * h[106] / H_AVG[106]
            + 1.977182e-07 * h[113] / H_AVG[113]
            - 1.878342e-07 * h[34] / H_AVG[34]
            + 1.686394e-07 * h[73] / H_AVG[73]
            + 1.566733e-07 * h[10] / H_AVG[10]
            + 1.492733e-07 * h[11] / H_AVG[11]
            - 1.271466e-07 * h[6] / H_AVG[6]
            + 1.212898e-07 * h[86] / H_AVG[86]
            - 1.153518e-07 * h[32] / H_AVG[32]
            + 1.060226e-07 * h[17] / H_AVG[17]
            + 9.583721e-08 * h[38] / H_AVG[38]
            - 7.868136e-08 * h[42] / H_AVG[42]
            - 7.012687e-08 * h[54] / H_AVG[54]
            - 6.666392e-08 * h[29] / H_AVG[29]
            + 6.043907e-08 * h[118] / H_AVG[118]
            + 4.717144e-08 * h[48] / H_AVG[48]
            - 4.501564e-08 * h[45] / H_AVG[45]
            - 4.348632e-08 * h[39] / H_AVG[39]
            + 4.009317e-08 * h[27] / H_AVG[27]
            + 3.991781e-08 * h[63] / H_AVG[63]
            + 3.948471e-08 * h[30] / H_AVG[30]
            + 3.867091e-08 * h[43] / H_AVG[43]
            + 3.598809e-08 * h[121] / H_AVG[121]
            - 3.533648e-08 * h[80] / H_AVG[80]
            - 3.23246e-08 * h[126] / H_AVG[126]
            + 3.122825e-08 * h[60] / H_AVG[60]
            - 3.075886e-08 * h[61] / H_AVG[61]
            - 2.920323e-08 * h[102] / H_AVG[102]
            - 2.913271e-08 * h[107] / H_AVG[107]
            - 2.871939e-08 * h[7] / H_AVG[7]
            + 2.585707e-08 * h[92] / H_AVG[92]
            + 2.526826e-08 * h[35] / H_AVG[35]
            + 2.458628e-08 * h[15] / H_AVG[15]
            - 2.375293e-08 * h[59] / H_AVG[59]
            - 2.29085e-08 * h[101] / H_AVG[101]
            - 2.191167e-08 * h[28] / H_AVG[28]
            + 2.123831e-08 * h[103] / H_AVG[103]
            + 1.998975e-08 * h[69] / H_AVG[69]
            + 1.939834e-08 * h[67] / H_AVG[67]
            + 1.879452e-08 * h[1] / H_AVG[1]
            + 1.865555e-08 * h[100] / H_AVG[100]
            - 1.713476e-08 * h[110] / H_AVG[110]
            - 1.657209e-08 * h[3] / H_AVG[3]
            + 1.592151e-08 * h[37] / H_AVG[37]
            - 1.559075e-08 * h[50] / H_AVG[50]
            + 1.466284e-08 * h[44] / H_AVG[44]
            + 1.46506e-08 * h[36] / H_AVG[36]
            - 1.440448e-08 * h[12] / H_AVG[12]
            - 1.36975e-08 * h[114] / H_AVG[114]
            - 1.357372e-08 * h[2] / H_AVG[2]
            + 1.309391e-08 * h[58] / H_AVG[58]
            - 1.304104e-08 * h[112] / H_AVG[112]
            + 1.276157e-08 * h[76] / H_AVG[76]
            - 1.230009e-08 * h[13] / H_AVG[13]
            - 1.195234e-08 * h[19] / H_AVG[19]
            - 1.172349e-08 * h[111] / H_AVG[111]
            + 1.11727e-08 * h[74] / H_AVG[74]
            + 1.060393e-08 * h[108] / H_AVG[108]
            + 9.972001e-09 * h[5] / H_AVG[5]
            - 9.529502e-09 * h[8] / H_AVG[8]
            - 9.138643e-09 * h[91] / H_AVG[91]
            - 8.517049e-09 * h[4] / H_AVG[4]
            - 8.157882e-09 * h[117] / H_AVG[117]
            + 6.891591e-09 * h[53] / H_AVG[53]
            - 6.812872e-09 * h[66] / H_AVG[66]
            - 6.204214e-09 * h[33] / H_AVG[33]
            - 6.024959e-09 * h[65] / H_AVG[65]
            + 5.575024e-09 * h[82] / H_AVG[82]
            - 5.237206e-09 * h[31] / H_AVG[31]
            - 4.812501e-09 * h[71] / H_AVG[71]
            + 4.470288e-09 * h[94] / H_AVG[94]
            + 4.31783e-09 * h[9] / H_AVG[9]
            + 3.793173e-09 * h[26] / H_AVG[26]
            - 3.401333e-09 * h[14] / H_AVG[14]
            + 3.369209e-09 * h[57] / H_AVG[57]
            + 3.184512e-09 * h[49] / H_AVG[49]
            + 3.11433e-09 * h[77] / H_AVG[77]
            - 3.105539e-09 * h[22] / H_AVG[22]
            + 2.880122e-09 * h[122] / H_AVG[122]
            + 2.32897e-09 * h[72] / H_AVG[72]
            + 2.229945e-09 * h[51] / H_AVG[51]
            - 2.196485e-09 * h[23] / H_AVG[23]
            - 1.516706e-09 * h[89] / H_AVG[89]
            + 1.481739e-09 * h[125] / H_AVG[125]
            - 1.39099e-09 * h[75] / H_AVG[75]
            - 1.203738e-09 * h[87] / H_AVG[87]
            + 1.199645e-09 * h[93] / H_AVG[93]
            - 1.078791e-09 * h[40] / H_AVG[40]
            - 1.002799e-09 * h[20] / H_AVG[20]
            + 8.956882e-10 * h[119] / H_AVG[119]
            - 8.194988e-10 * h[62] / H_AVG[62]
            - 7.366908e-10 * h[95] / H_AVG[95]
            - 6.82779e-10 * h[96] / H_AVG[96]
            + 4.026601e-10 * h[116] / H_AVG[116]
            + 3.842814e-10 * h[56] / H_AVG[56]
            + 2.671353e-10 * h[85] / H_AVG[85]
            + 1.591371e-10 * h[109] / H_AVG[109]
            - 1.385252e-10 * h[98] / H_AVG[98]
            - 1.107165e-10 * h[88] / H_AVG[88]
            - 7.455439e-11 * h[46] / H_AVG[46]
            - 7.038574e-11 * h[47] / H_AVG[47]
            - 4.002914e-11 * h[64] / H_AVG[64]
            + 1.686767e-11 * h[127] / H_AVG[127]
            + 1.49137e-12 * h[41] / H_AVG[41]
        ),
        0.09887857 + T[7] * (   # class Wqq
            + 0.26065 * h[97] / H_AVG[97]
            + 0.1049109 * h[24] / H_AVG[24]
            + 0.09995708 * h[68] / H_AVG[68]
            + 0.09425988 * h[83] / H_AVG[83]
            - 0.08651805 * h[52] / H_AVG[52]
            - 0.07751452 * h[120] / H_AVG[120]
            + 0.05584857 * h[115] / H_AVG[115]
            + 0.05407094 * h[104] / H_AVG[104]
            - 0.04283497 * h[16] / H_AVG[16]
            - 0.03683727 * h[81] / H_AVG[81]
            - 0.03599867 * h[78] / H_AVG[78]
            - 0.01602029 * h[70] / H_AVG[70]
            + 0.0136015 * h[124] / H_AVG[124]
            - 0.01102388 * h[25] / H_AVG[25]
            - 0.006526785 * h[90] / H_AVG[90]
            + 0.001703356 * h[18] / H_AVG[18]
            + 0.001483154 * h[123] / H_AVG[123]
            - 0.0001697037 * h[0] / H_AVG[0]
            - 3.826133e-05 * h[99] / H_AVG[99]
            - 2.659901e-05 * h[21] / H_AVG[21]
            + 1.799505e-06 * h[84] / H_AVG[84]
            + 4.24653e-07 * h[55] / H_AVG[55]
            + 2.849557e-07 * h[79] / H_AVG[79]
            - 2.720648e-07 * h[105] / H_AVG[105]
            - 2.208206e-07 * h[106] / H_AVG[106]
            + 1.794337e-07 * h[113] / H_AVG[113]
            - 1.704333e-07 * h[34] / H_AVG[34]
            + 1.530339e-07 * h[73] / H_AVG[73]
            + 1.421145e-07 * h[10] / H_AVG[10]
            + 1.354477e-07 * h[11] / H_AVG[11]
            - 1.153745e-07 * h[6] / H_AVG[6]
            + 1.100693e-07 * h[86] / H_AVG[86]
            - 1.046661e-07 * h[32] / H_AVG[32]
            + 9.619108e-08 * h[17] / H_AVG[17]
            + 8.696296e-08 * h[38] / H_AVG[38]
            - 7.140097e-08 * h[42] / H_AVG[42]
            - 6.364143e-08 * h[54] / H_AVG[54]
            - 6.048431e-08 * h[29] / H_AVG[29]
            + 5.483945e-08 * h[118] / H_AVG[118]
            + 4.280453e-08 * h[48] / H_AVG[48]
            - 4.084806e-08 * h[45] / H_AVG[45]
            - 3.944837e-08 * h[39] / H_AVG[39]
            + 3.638102e-08 * h[27] / H_AVG[27]
            + 3.621831e-08 * h[63] / H_AVG[63]
            + 3.583032e-08 * h[30] / H_AVG[30]
            + 3.508777e-08 * h[43] / H_AVG[43]
            + 3.265159e-08 * h[121] / H_AVG[121]
            - 3.206079e-08 * h[80] / H_AVG[80]
            - 2.93454e-08 * h[126] / H_AVG[126]
            + 2.832823e-08 * h[60] / H_AVG[60]
            - 2.790171e-08 * h[61] / H_AVG[61]
            - 2.64972e-08 * h[102] / H_AVG[102]
            - 2.643647e-08 * h[107] / H_AVG[107]
            - 2.604765e-08 * h[7] / H_AVG[7]
            + 2.346036e-08 * h[92] / H_AVG[92]
            + 2.293133e-08 * h[35] / H_AVG[35]
            + 2.230908e-08 * h[15] / H_AVG[15]
            - 2.156094e-08 * h[59] / H_AVG[59]
            - 2.078493e-08 * h[101] / H_AVG[101]
            - 1.986998e-08 * h[28] / H_AVG[28]
            + 1.92727e-08 * h[103] / H_AVG[103]
            + 1.813632e-08 * h[69] / H_AVG[69]
            + 1.759895e-08 * h[67] / H_AVG[67]
            + 1.705117e-08 * h[1] / H_AVG[1]
            + 1.692608e-08 * h[100] / H_AVG[100]
            - 1.5543e-08 * h[110] / H_AVG[110]
            - 1.504068e-08 * h[3] / H_AVG[3]
            + 1.444639e-08 * h[37] / H_AVG[37]
            - 1.414866e-08 * h[50] / H_AVG[50]
            + 1.330364e-08 * h[44] / H_AVG[44]
            + 1.329409e-08 * h[36] / H_AVG[36]
            - 1.306494e-08 * h[12] / H_AVG[12]
            - 1.242734e-08 * h[114] / H_AVG[114]
            - 1.232502e-08 * h[2] / H_AVG[2]
            + 1.1881e-08 * h[58] / H_AVG[58]
            - 1.183397e-08 * h[112] / H_AVG[112]
            + 1.157927e-08 * h[76] / H_AVG[76]
            - 1.115944e-08 * h[13] / H_AVG[13]
            - 1.08449e-08 * h[19] / H_AVG[19]
            - 1.063656e-08 * h[111] / H_AVG[111]
            + 1.01372e-08 * h[74] / H_AVG[74]
            + 9.622451e-09 * h[108] / H_AVG[108]
            + 9.047986e-09 * h[5] / H_AVG[5]
            - 8.646525e-09 * h[8] / H_AVG[8]
            - 8.289402e-09 * h[91] / H_AVG[91]
            - 7.727893e-09 * h[4] / H_AVG[4]
            - 7.402218e-09 * h[117] / H_AVG[117]
            + 6.25195e-09 * h[53] / H_AVG[53]
            - 6.186587e-09 * h[66] / H_AVG[66]
            - 5.629966e-09 * h[33] / H_AVG[33]
            - 5.468244e-09 * h[65] / H_AVG[65]
            + 5.057568e-09 * h[82] / H_AVG[82]
            - 4.753894e-09 * h[31] / H_AVG[31]
            - 4.367661e-09 * h[71] / H_AVG[71]
            + 4.054446e-09 * h[94] / H_AVG[94]
            + 3.917829e-09 * h[9] / H_AVG[9]
            + 3.440074e-09 * h[26] / H_AVG[26]
            - 3.086767e-09 * h[14] / H_AVG[14]
            + 3.057236e-09 * h[57] / H_AVG[57]
            + 2.889532e-09 * h[49] / H_AVG[49]
            + 2.823522e-09 * h[77] / H_AVG[77]
            - 2.818284e-09 * h[22] / H_AVG[22]
            + 2.61336e-09 * h[122] / H_AVG[122]
            + 2.113193e-09 * h[72] / H_AVG[72]
            + 2.024284e-09 * h[51] / H_AVG[51]
            - 1.993741e-09 * h[23] / H_AVG[23]
            - 1.376241e-09 * h[89] / H_AVG[89]
            + 1.344296e-09 * h[125] / H_AVG[125]
            - 1.262287e-09 * h[75] / H_AVG[75]
            - 1.091632e-09 * h[87] / H_AVG[87]
            + 1.087665e-09 * h[93] / H_AVG[93]
            - 9.788079e-10 * h[40] / H_AVG[40]
            - 9.099133e-10 * h[20] / H_AVG[20]
            + 8.119799e-10 * h[119] / H_AVG[119]
            - 7.435483e-10 * h[62] / H_AVG[62]
            - 6.683715e-10 * h[95] / H_AVG[95]
            - 6.195819e-10 * h[96] / H_AVG[96]
            + 3.652913e-10 * h[116] / H_AVG[116]
            + 3.486966e-10 * h[56] / H_AVG[56]
            + 2.424661e-10 * h[85] / H_AVG[85]
            + 1.443531e-10 * h[109] / H_AVG[109]
            - 1.256781e-10 * h[98] / H_AVG[98]
            - 1.004098e-10 * h[88] / H_AVG[88]
            - 6.762705e-11 * h[46] / H_AVG[46]
            - 6.386593e-11 * h[47] / H_AVG[47]
            - 3.771341e-11 * h[64] / H_AVG[64]
            + 1.529786e-11 * h[127] / H_AVG[127]
            + 1.354699e-12 * h[41] / H_AVG[41]
        ),
        -0.4320669 + T[8] * (   # class Tbqq
            - 0.1910558 * h[104] / H_AVG[104]
            + 0.1489519 * h[16] / H_AVG[16]
            + 0.1228288 * h[120] / H_AVG[120]
            + 0.1214898 * h[68] / H_AVG[68]
            - 0.1016767 * h[83] / H_AVG[83]
            - 0.08085363 * h[18] / H_AVG[18]
            + 0.04849293 * h[78] / H_AVG[78]
            - 0.03651414 * h[24] / H_AVG[24]
            + 0.02769992 * h[97] / H_AVG[97]
            + 0.02628759 * h[81] / H_AVG[81]
            - 0.02606355 * h[115] / H_AVG[115]
            - 0.02598668 * h[52] / H_AVG[52]
            - 0.01689076 * h[70] / H_AVG[70]
            + 0.008986917 * h[124] / H_AVG[124]
            - 0.008145187 * h[90] / H_AVG[90]
            - 0.006984036 * h[123] / H_AVG[123]
            + 0.0009335598 * h[25] / H_AVG[25]
            - 0.0001209864 * h[0] / H_AVG[0]
            - 2.314658e-05 * h[99] / H_AVG[99]
            - 1.053033e-05 * h[21] / H_AVG[21]
            + 1.096687e-06 * h[84] / H_AVG[84]
            + 2.588608e-07 * h[55] / H_AVG[55]
            + 1.737346e-07 * h[79] / H_AVG[79]
            - 1.657838e-07 * h[105] / H_AVG[105]
            - 1.345522e-07 * h[106] / H_AVG[106]
            + 1.093611e-07 * h[113] / H_AVG[113]
            - 1.038642e-07 * h[34] / H_AVG[34]
            + 9.324025e-08 * h[73] / H_AVG[73]
            + 8.661665e-08 * h[10] / H_AVG[10]
            + 8.255662e-08 * h[11] / H_AVG[11]
            - 7.030086e-08 * h[6] / H_AVG[6]
            + 6.707311e-08 * h[86] / H_AVG[86]
            - 6.378773e-08 * h[32] / H_AVG[32]
            + 5.864961e-08 * h[17] / H_AVG[17]
            + 5.300018e-08 * h[38] / H_AVG[38]
            - 4.351563e-08 * h[42] / H_AVG[42]
            - 3.878149e-08 * h[54] / H_AVG[54]
            - 3.686412e-08 * h[29] / H_AVG[29]
            + 3.341922e-08 * h[118] / H_AVG[118]
            + 2.608766e-08 * h[48] / H_AVG[48]
            - 2.488794e-08 * h[45] / H_AVG[45]
            - 2.403978e-08 * h[39] / H_AVG[39]
            + 2.21715e-08 * h[27] / H_AVG[27]
            + 2.20751e-08 * h[63] / H_AVG[63]
            + 2.183576e-08 * h[30] / H_AVG[30]
            + 2.138069e-08 * h[43] / H_AVG[43]
            + 1.990209e-08 * h[121] / H_AVG[121]
            - 1.954098e-08 * h[80] / H_AVG[80]
            - 1.787765e-08 * h[126] / H_AVG[126]
            + 1.726949e-08 * h[60] / H_AVG[60]
            - 1.700434e-08 * h[61] / H_AVG[61]
            - 1.614384e-08 * h[102] / H_AVG[102]
            - 1.611005e-08 * h[107] / H_AVG[107]
            - 1.587575e-08 * h[7] / H_AVG[7]
            + 1.429534e-08 * h[92] / H_AVG[92]
            + 1.39661e-08 * h[35] / H_AVG[35]
            + 1.35959e-08 * h[15] / H_AVG[15]
            - 1.313169e-08 * h[59] / H_AVG[59]
            - 1.266779e-08 * h[101] / H_AVG[101]
            - 1.211907e-08 * h[28] / H_AVG[28]
            + 1.17416e-08 * h[103] / H_AVG[103]
            + 1.105122e-08 * h[69] / H_AVG[69]
            + 1.072668e-08 * h[67] / H_AVG[67]
            + 1.03938e-08 * h[1] / H_AVG[1]
            + 1.031581e-08 * h[100] / H_AVG[100]
            - 9.475452e-09 * h[110] / H_AVG[110]
            - 9.166997e-09 * h[3] / H_AVG[3]
            + 8.804342e-09 * h[37] / H_AVG[37]
            - 8.618767e-09 * h[50] / H_AVG[50]
            + 8.107955e-09 * h[44] / H_AVG[44]
            + 8.100466e-09 * h[36] / H_AVG[36]
            - 7.9643e-09 * h[12] / H_AVG[12]
            - 7.574219e-09 * h[114] / H_AVG[114]
            - 7.510629e-09 * h[2] / H_AVG[2]
            + 7.241131e-09 * h[58] / H_AVG[58]
            - 7.21125e-09 * h[112] / H_AVG[112]
            + 7.057114e-09 * h[76] / H_AVG[76]
            - 6.800432e-09 * h[13] / H_AVG[13]
            - 6.608342e-09 * h[19] / H_AVG[19]
            - 6.4831e-09 * h[111] / H_AVG[111]
            + 6.179016e-09 * h[74] / H_AVG[74]
            + 5.863922e-09 * h[108] / H_AVG[108]
            + 5.513197e-09 * h[5] / H_AVG[5]
            - 5.269248e-09 * h[8] / H_AVG[8]
            - 5.051269e-09 * h[91] / H_AVG[91]
            - 4.709992e-09 * h[4] / H_AVG[4]
            - 4.510419e-09 * h[117] / H_AVG[117]
            + 3.810841e-09 * h[53] / H_AVG[53]
            - 3.767008e-09 * h[66] / H_AVG[66]
            - 3.431012e-09 * h[33] / H_AVG[33]
            - 3.332838e-09 * h[65] / H_AVG[65]
            + 3.082471e-09 * h[82] / H_AVG[82]
            - 2.89504e-09 * h[31] / H_AVG[31]
            - 2.660902e-09 * h[71] / H_AVG[71]
            + 2.472981e-09 * h[94] / H_AVG[94]
            + 2.387577e-09 * h[9] / H_AVG[9]
            + 2.098685e-09 * h[26] / H_AVG[26]
            - 1.880726e-09 * h[14] / H_AVG[14]
            + 1.86398e-09 * h[57] / H_AVG[57]
            + 1.761011e-09 * h[49] / H_AVG[49]
            + 1.72214e-09 * h[77] / H_AVG[77]
            - 1.71725e-09 * h[22] / H_AVG[22]
            + 1.592618e-09 * h[122] / H_AVG[122]
            + 1.287812e-09 * h[72] / H_AVG[72]
            + 1.233503e-09 * h[51] / H_AVG[51]
            - 1.212807e-09 * h[23] / H_AVG[23]
            - 8.385933e-10 * h[89] / H_AVG[89]
            + 8.193339e-10 * h[125] / H_AVG[125]
            - 7.690388e-10 * h[75] / H_AVG[75]
            - 6.654535e-10 * h[87] / H_AVG[87]
            + 6.632004e-10 * h[93] / H_AVG[93]
            - 5.965486e-10 * h[40] / H_AVG[40]
            - 5.544987e-10 * h[20] / H_AVG[20]
            + 4.953213e-10 * h[119] / H_AVG[119]
            - 4.530674e-10 * h[62] / H_AVG[62]
            - 4.072681e-10 * h[95] / H_AVG[95]
            - 3.774303e-10 * h[96] / H_AVG[96]
            + 2.226261e-10 * h[116] / H_AVG[116]
            + 2.125496e-10 * h[56] / H_AVG[56]
            + 1.478251e-10 * h[85] / H_AVG[85]
            + 8.795768e-11 * h[109] / H_AVG[109]
            - 7.661152e-11 * h[98] / H_AVG[98]
            - 6.111908e-11 * h[88] / H_AVG[88]
            - 4.123015e-11 * h[46] / H_AVG[46]
            - 3.891544e-11 * h[47] / H_AVG[47]
            - 2.129988e-11 * h[64] / H_AVG[64]
            + 9.345219e-12 * h[127] / H_AVG[127]
            + 8.263256e-13 * h[41] / H_AVG[41]
        ),
        -1.495433 + T[9] * (   # class Tbl
            + 0.2332115 * h[16] / H_AVG[16]
            + 0.1103374 * h[24] / H_AVG[24]
            + 0.1039285 * h[18] / H_AVG[18]
            - 0.09717577 * h[104] / H_AVG[104]
            - 0.07307278 * h[115] / H_AVG[115]
            - 0.07084206 * h[124] / H_AVG[124]
            + 0.06913052 * h[97] / H_AVG[97]
            + 0.06023489 * h[52] / H_AVG[52]
            - 0.05498344 * h[68] / H_AVG[68]
            + 0.04190535 * h[78] / H_AVG[78]
            + 0.01663823 * h[70] / H_AVG[70]
            + 0.01661909 * h[120] / H_AVG[120]
            + 0.01542143 * h[123] / H_AVG[123]
            + 0.01250398 * h[25] / H_AVG[25]
            + 0.01046961 * h[90] / H_AVG[90]
            - 0.008430387 * h[83] / H_AVG[83]
            - 0.004715431 * h[81] / H_AVG[81]
            - 0.0003103533 * h[0] / H_AVG[0]
            + 5.177068e-05 * h[21] / H_AVG[21]
            - 1.524468e-05 * h[99] / H_AVG[99]
            + 7.279715e-07 * h[84] / H_AVG[84]
            + 1.716995e-07 * h[55] / H_AVG[55]
            + 1.151903e-07 * h[79] / H_AVG[79]
            - 1.100776e-07 * h[105] / H_AVG[105]
            - 8.931219e-08 * h[106] / H_AVG[106]
            + 7.2625e-08 * h[113] / H_AVG[113]
            - 6.893287e-08 * h[34] / H_AVG[34]
            + 6.188203e-08 * h[73] / H_AVG[73]
            + 5.744758e-08 * h[10] / H_AVG[10]
            + 5.480877e-08 * h[11] / H_AVG[11]
            - 4.661956e-08 * h[6] / H_AVG[6]
            + 4.450806e-08 * h[86] / H_AVG[86]
            - 4.234948e-08 * h[32] / H_AVG[32]
            + 3.892047e-08 * h[17] / H_AVG[17]
            + 3.514784e-08 * h[38] / H_AVG[38]
            - 2.887891e-08 * h[42] / H_AVG[42]
            - 2.575345e-08 * h[54] / H_AVG[54]
            - 2.445695e-08 * h[29] / H_AVG[29]
            + 2.218167e-08 * h[118] / H_AVG[118]
            + 1.7298e-08 * h[48] / H_AVG[48]
            - 1.651968e-08 * h[45] / H_AVG[45]
            - 1.594359e-08 * h[39] / H_AVG[39]
            + 1.471549e-08 * h[27] / H_AVG[27]
            + 1.464698e-08 * h[63] / H_AVG[63]
            + 1.450237e-08 * h[30] / H_AVG[30]
            + 1.418103e-08 * h[43] / H_AVG[43]
            + 1.319787e-08 * h[121] / H_AVG[121]
            - 1.29637e-08 * h[80] / H_AVG[80]
            - 1.18782e-08 * h[126] / H_AVG[126]
            + 1.145494e-08 * h[60] / H_AVG[60]
            - 1.127451e-08 * h[61] / H_AVG[61]
            - 1.071228e-08 * h[102] / H_AVG[102]
            - 1.067697e-08 * h[107] / H_AVG[107]
            - 1.053383e-08 * h[7] / H_AVG[7]
            + 9.481407e-09 * h[92] / H_AVG[92]
            + 9.273983e-09 * h[35] / H_AVG[35]
            + 9.016583e-09 * h[15] / H_AVG[15]
            - 8.719193e-09 * h[59] / H_AVG[59]
            - 8.41276e-09 * h[101] / H_AVG[101]
            - 8.022128e-09 * h[28] / H_AVG[28]
            + 7.796263e-09 * h[103] / H_AVG[103]
            + 7.346173e-09 * h[69] / H_AVG[69]
            + 7.130806e-09 * h[67] / H_AVG[67]
            + 6.889284e-09 * h[1] / H_AVG[1]
            + 6.836714e-09 * h[100] / H_AVG[100]
            - 6.281924e-09 * h[110] / H_AVG[110]
            - 6.090862e-09 * h[3] / H_AVG[3]
            + 5.844242e-09 * h[37] / H_AVG[37]
            - 5.714497e-09 * h[50] / H_AVG[50]
            + 5.378331e-09 * h[36] / H_AVG[36]
            + 5.377069e-09 * h[44] / H_AVG[44]
            - 5.290729e-09 * h[12] / H_AVG[12]
            - 5.026976e-09 * h[114] / H_AVG[114]
            - 4.991864e-09 * h[2] / H_AVG[2]
            + 4.80997e-09 * h[58] / H_AVG[58]
            - 4.795349e-09 * h[112] / H_AVG[112]
            + 4.688477e-09 * h[76] / H_AVG[76]
            - 4.506737e-09 * h[13] / H_AVG[13]
            - 4.385502e-09 * h[19] / H_AVG[19]
            - 4.296919e-09 * h[111] / H_AVG[111]
            + 4.099092e-09 * h[74] / H_AVG[74]
            + 3.881944e-09 * h[108] / H_AVG[108]
            + 3.655127e-09 * h[5] / H_AVG[5]
            - 3.499558e-09 * h[8] / H_AVG[8]
            - 3.359672e-09 * h[91] / H_AVG[91]
            - 3.128796e-09 * h[4] / H_AVG[4]
            - 2.996584e-09 * h[117] / H_AVG[117]
            + 2.528312e-09 * h[53] / H_AVG[53]
            - 2.500967e-09 * h[66] / H_AVG[66]
            - 2.27058e-09 * h[33] / H_AVG[33]
            - 2.218209e-09 * h[65] / H_AVG[65]
            + 2.050627e-09 * h[82] / H_AVG[82]
            - 1.918041e-09 * h[31] / H_AVG[31]
            - 1.764088e-09 * h[71] / H_AVG[71]
            + 1.639821e-09 * h[94] / H_AVG[94]
            + 1.583902e-09 * h[9] / H_AVG[9]
            + 1.39869e-09 * h[26] / H_AVG[26]
            - 1.248737e-09 * h[14] / H_AVG[14]
            + 1.241502e-09 * h[57] / H_AVG[57]
            + 1.167269e-09 * h[49] / H_AVG[49]
            + 1.140528e-09 * h[77] / H_AVG[77]
            - 1.138771e-09 * h[22] / H_AVG[22]
            + 1.060541e-09 * h[122] / H_AVG[122]
            + 8.534572e-10 * h[72] / H_AVG[72]
            + 8.222065e-10 * h[51] / H_AVG[51]
            - 8.077624e-10 * h[23] / H_AVG[23]
            - 5.529493e-10 * h[89] / H_AVG[89]
            + 5.423662e-10 * h[125] / H_AVG[125]
            - 5.092717e-10 * h[75] / H_AVG[75]
            + 4.402559e-10 * h[93] / H_AVG[93]
            - 4.393434e-10 * h[87] / H_AVG[87]
            - 3.956932e-10 * h[40] / H_AVG[40]
            - 3.679673e-10 * h[20] / H_AVG[20]
            + 3.297292e-10 * h[119] / H_AVG[119]
            - 2.998581e-10 * h[62] / H_AVG[62]
            - 2.701292e-10 * h[95] / H_AVG[95]
            - 2.497876e-10 * h[96] / H_AVG[96]
            + 1.474512e-10 * h[116] / H_AVG[116]
            + 1.41602e-10 * h[56] / H_AVG[56]
            + 9.878194e-11 * h[85] / H_AVG[85]
            + 5.814799e-11 * h[109] / H_AVG[109]
            - 5.064361e-11 * h[98] / H_AVG[98]
            - 4.029257e-11 * h[88] / H_AVG[88]
            - 2.755321e-11 * h[46] / H_AVG[46]
            - 2.575802e-11 * h[47] / H_AVG[47]
            - 1.349074e-11 * h[64] / H_AVG[64]
            + 6.444411e-12 * h[127] / H_AVG[127]
            + 5.737906e-13 * h[41] / H_AVG[41]
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
