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
  neuron 16:  11.0%
  neuron 18:   9.6%
  neuron 97:   9.3%
  neuron 24:   8.7%
  neuron 120:   8.6%
  neuron 104:   8.1%
  neuron 115:   7.8%
  neuron 68:   7.1%
  neuron 52:   6.8%
  neuron 83:   6.8%
  neuron 124:   5.0%
  neuron 78:   4.1%
  neuron 81:   3.4%
  neuron 70:   1.5%
  neuron 90:   0.8%
  neuron 25:   0.8%
  neuron 123:   0.7%
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

Whole test file (2,000,000 jets): accuracy 75.46% (the network: 86.03%); same class as the network for 80.34% of jets.

Quantities:
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b05                 e3/e2² with β = 0.5
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3_b2                  e4·e2/e3² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.D3_b2                  e4·e2³/e3³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M2_b05                 ₁e₃/e2 with β = 0.5
  Q.M2_b2                  ₁e₃/e2 with β = 2
  Q.M3                     generalized ECF ratio M3
  Q.M3_b2                  ₁e₄/₁e₃ with β = 2
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N2_b05                 ₂e₃/(e2)² with β = 0.5
  Q.N2_b2                  ₂e₃/(e2)² with β = 2
  Q.N3_b05                 ₂e₄/(₁e₃)² with β = 0.5
  Q.N3_b2                  ₂e₄/(₁e₃)² with β = 2
  Q.ak02_1_mass            mass (E) of the hardest anti-kT 0.2 subjet [GeV] (0 if none)
  Q.ak02_1_n_disp3         hardest anti-kT 0.2 subjet: number of its tracks with d0/σ > 3
  Q.ak02_1_n_lep           hardest anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_1_z               pT share of the hardest anti-kT 0.2 subjet (0 if none)
  Q.ak02_2_n_lep           2nd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_2_sd0_3           2nd anti-kT 0.2 subjet: the 3rd largest signed d0/σ among its tracks (0 if fewer)
  Q.ak02_2_z               pT share of the 2nd anti-kT 0.2 subjet (0 if none)
  Q.ak02_3_mass            mass (E) of the 3rd anti-kT 0.2 subjet [GeV] (0 if none)
  Q.ak02_3_n_lep           3rd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_3_z               pT share of the 3rd anti-kT 0.2 subjet (0 if none)
  Q.ak02_3_z_disp3         3rd anti-kT 0.2 subjet: pT share (of the jet) of its tracks with d0/σ > 3
  Q.ak02_dr12              distance between the pT-weighted centres of the hardest and 2nd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr13              distance between the pT-weighted centres of the hardest and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr23              distance between the pT-weighted centres of the 2nd and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_min12_n_disp3     the smaller of the two hardest anti-kT 0.2 subjets: number of its tracks with d0/σ > 3
  Q.ak02_n                 number of anti-kT R = 0.2 subjets with pT > 10 GeV
  Q.charge_76              charge of particle 76 (ParT input; empty slot: 0)
  Q.d0err_10               σ(d0), clipped to [0, 1] of particle 10 (ParT input; empty slot: 0)
  Q.d0err_34               σ(d0), clipped to [0, 1] of particle 34 (ParT input; empty slot: 0)
  Q.d0err_49               σ(d0), clipped to [0, 1] of particle 49 (ParT input; empty slot: 0)
  Q.d0err_52               σ(d0), clipped to [0, 1] of particle 52 (ParT input; empty slot: 0)
  Q.d0err_60               σ(d0), clipped to [0, 1] of particle 60 (ParT input; empty slot: 0)
  Q.dc_1_charge            hardest prong: Σ q √pT / √(Σ pT) of its particles
  Q.dc_1_n_disp3           hardest prong: number of its tracks with d0/σ > 3
  Q.dc_1_n_lep             hardest prong: number of its electrons and muons
  Q.dc_1_sd0_1             hardest prong: the largest signed d0/σ among its tracks (0 if none)
  Q.dc_1_z                 pT share of the hardest prong (0 if none)
  Q.dc_2_charge            2nd prong: Σ q √pT / √(Σ pT) of its particles
  Q.dc_2_jp                2nd prong: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.dc_2_n_disp3           2nd prong: number of its tracks with d0/σ > 3
  Q.dc_2_n_lep             2nd prong: number of its electrons and muons
  Q.dc_3_charge            3rd prong: Σ q √pT / √(Σ pT) of its particles
  Q.dc_3_jp                3rd prong: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.dc_3_n_lep             3rd prong: number of its electrons and muons
  Q.dc_3_z                 pT share of the 3rd prong (0 if none)
  Q.dc_4_jp                4th prong: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.dc_4_mass_disp3        4th prong: mass (E) of its tracks with d0/σ > 3 [GeV]
  Q.dc_4_z                 pT share of the 4th prong (0 if none)
  Q.dc_mass_disp_2nd       second largest displaced-track mass among the 4 hardest prongs [GeV]
  Q.dc_n                   number of prongs: reverse the C/A tree; ΔR ≤ 0.1 is a prong, a branch with < 10 % of the jet pT is dropped, else both branches are declustered
  Q.dc_ntag                number of the 4 hardest prongs whose 2nd largest d0/σ is above 3
  Q.dc_pair_mass_min       smallest mass (E) of two of the 4 hardest prongs [GeV] (0 if fewer than 2)
  Q.dc_split1_dr           ΔR of the hardest hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_split1_kt           kT = min(pT)·ΔR [GeV] of the hardest hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_split1_mass         mass of the splitting node [GeV] of the hardest hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_split2_dr           ΔR of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_split2_kt           kT = min(pT)·ΔR [GeV] of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_split2_mass         mass of the splitting node [GeV] of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_tag_2nd             second largest 2nd-largest d0/σ among the 4 hardest prongs (double tag)
  Q.dc_tag_max             largest 2nd-largest d0/σ among the 4 hardest prongs
  Q.dr12                   ΔR between particles 1 and 2
  Q.dr_0                   ΔR from the jet axis of particle 0 (ParT input; empty slot: 0)
  Q.dr_1                   ΔR from the jet axis of particle 1 (ParT input; empty slot: 0)
  Q.dr_11                  ΔR from the jet axis of particle 11 (ParT input; empty slot: 0)
  Q.dr_18                  ΔR from the jet axis of particle 18 (ParT input; empty slot: 0)
  Q.dr_2                   ΔR from the jet axis of particle 2 (ParT input; empty slot: 0)
  Q.dr_21                  ΔR from the jet axis of particle 21 (ParT input; empty slot: 0)
  Q.dr_28                  ΔR from the jet axis of particle 28 (ParT input; empty slot: 0)
  Q.dr_3                   ΔR from the jet axis of particle 3 (ParT input; empty slot: 0)
  Q.dr_39                  ΔR from the jet axis of particle 39 (ParT input; empty slot: 0)
  Q.dr_68                  ΔR from the jet axis of particle 68 (ParT input; empty slot: 0)
  Q.dr_86                  ΔR from the jet axis of particle 86 (ParT input; empty slot: 0)
  Q.dr_94                  ΔR from the jet axis of particle 94 (ParT input; empty slot: 0)
  Q.dr_max_012             largest distance among the 3 hardest
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.dzerr_0                σ(dz), clipped to [0, 1] of particle 0 (ParT input; empty slot: 0)
  Q.dzerr_1                σ(dz), clipped to [0, 1] of particle 1 (ParT input; empty slot: 0)
  Q.dzerr_22               σ(dz), clipped to [0, 1] of particle 22 (ParT input; empty slot: 0)
  Q.dzerr_32               σ(dz), clipped to [0, 1] of particle 32 (ParT input; empty slot: 0)
  Q.dzerr_75               σ(dz), clipped to [0, 1] of particle 75 (ParT input; empty slot: 0)
  Q.dzerr_76               σ(dz), clipped to [0, 1] of particle 76 (ParT input; empty slot: 0)
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e3                     energy correlation e3 (β=1, 32 hardest)
  Q.e3_b05                 energy correlation e3 with β = 0.5
  Q.e3_b2                  energy correlation e3 with β = 2
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 16 hardest)
  Q.e4_b2                  energy correlation e4 with β = 2
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.ecf_g31                generalized energy correlation ₁e₃ (β=1; products of the smallest angles)
  Q.ecf_g41                generalized energy correlation ₁e₄ (β=1; products of the smallest angles)
  Q.ecf_g42                generalized energy correlation ₂e₄ (β=1; products of the smallest angles)
  Q.ecf_g43                generalized energy correlation ₃e₄ (β=1; products of the smallest angles)
  Q.eta_1                  Δη of particle 1 (ParT input; empty slot: 0)
  Q.eta_14                 Δη of particle 14 (ParT input; empty slot: 0)
  Q.eta_28                 Δη of particle 28 (ParT input; empty slot: 0)
  Q.eta_37                 Δη of particle 37 (ParT input; empty slot: 0)
  Q.eta_4                  Δη of particle 4 (ParT input; empty slot: 0)
  Q.eta_8                  Δη of particle 8 (ParT input; empty slot: 0)
  Q.eta_9                  Δη of particle 9 (ParT input; empty slot: 0)
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.ischhad_0              1 if a charged hadron of particle 0 (ParT input; empty slot: 0)
  Q.ischhad_10             1 if a charged hadron of particle 10 (ParT input; empty slot: 0)
  Q.iselectron_1           1 if an electron of particle 1 (ParT input; empty slot: 0)
  Q.iselectron_11          1 if an electron of particle 11 (ParT input; empty slot: 0)
  Q.iselectron_12          1 if an electron of particle 12 (ParT input; empty slot: 0)
  Q.iselectron_27          1 if an electron of particle 27 (ParT input; empty slot: 0)
  Q.ismuon_0               1 if a muon of particle 0 (ParT input; empty slot: 0)
  Q.ismuon_3               1 if a muon of particle 3 (ParT input; empty slot: 0)
  Q.isnhad_34              1 if a neutral hadron of particle 34 (ParT input; empty slot: 0)
  Q.isnhad_8               1 if a neutral hadron of particle 8 (ParT input; empty slot: 0)
  Q.isphoton_19            1 if a photon of particle 19 (ParT input; empty slot: 0)
  Q.isphoton_29            1 if a photon of particle 29 (ParT input; empty slot: 0)
  Q.isphoton_40            1 if a photon of particle 40 (ParT input; empty slot: 0)
  Q.isphoton_42            1 if a photon of particle 42 (ParT input; empty slot: 0)
  Q.isphoton_53            1 if a photon of particle 53 (ParT input; empty slot: 0)
  Q.jd_3d_4                the 4th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_5                the 5th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_6                the 6th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_n_d3_pt1            number of tracks with |d0/σ| > 3 and pT > 1 GeV
  Q.jd_sum_abs_sd0_top3    sum of the 3 largest |d0/σ|
  Q.jd_sum_abs_sd0_top5    sum of the 5 largest |d0/σ|
  Q.jet_abs_eta            absolute pseudorapidity of the jet axis
  Q.jet_charge             pT-weighted jet charge (κ = 1)
  Q.jet_charge_k03         jet charge with κ = 0.3
  Q.jet_charge_k05         jet charge with κ = 0.5
  Q.jet_e                  jet energy [GeV]
  Q.kt2_1_n_lep            hardest kT subjet: number of its electrons and muons
  Q.kt2_1_sd0_3            hardest kT subjet: the 3rd largest signed d0/σ among its tracks (0 if fewer)
  Q.kt2_1_z_disp3          hardest kT subjet: pT share (of the jet) of its tracks with d0/σ > 3
  Q.kt2_2_n_disp3          2nd kT subjet: number of its tracks with d0/σ > 3
  Q.kt2_2_n_lep            2nd kT subjet: number of its electrons and muons
  Q.kt2_2_z_disp3          2nd kT subjet: pT share (of the jet) of its tracks with d0/σ > 3
  Q.kt2_dr12               distance between the pT-weighted centres of the hardest and 2nd kT subjet (0 if missing)
  Q.kt2_min12_jp           the smaller of the two hardest kT subjets: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.kt2_min12_mass_disp3   the smaller of the two hardest kT subjets: mass (E) of its tracks with d0/σ > 3 [GeV]
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
  Q.lne_1                  ln E [GeV] of particle 1 (ParT input; empty slot: ln 1e-8)
  Q.lne_16                 ln E [GeV] of particle 16 (ParT input; empty slot: ln 1e-8)
  Q.lne_17                 ln E [GeV] of particle 17 (ParT input; empty slot: ln 1e-8)
  Q.lne_21                 ln E [GeV] of particle 21 (ParT input; empty slot: ln 1e-8)
  Q.lne_3                  ln E [GeV] of particle 3 (ParT input; empty slot: ln 1e-8)
  Q.lne_4                  ln E [GeV] of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lne_40                 ln E [GeV] of particle 40 (ParT input; empty slot: ln 1e-8)
  Q.lne_5                  ln E [GeV] of particle 5 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_0               ln(E / E of the jet) of particle 0 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_1               ln(E / E of the jet) of particle 1 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_4               ln(E / E of the jet) of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_78              ln(E / E of the jet) of particle 78 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_31                ln pT [GeV] of particle 31 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_32                ln pT [GeV] of particle 32 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_40                ln pT [GeV] of particle 40 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_43                ln pT [GeV] of particle 43 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_65                ln pT [GeV] of particle 65 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_25             ln(pT / pT of the jet) of particle 25 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_3              ln(pT / pT of the jet) of particle 3 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_31             ln(pT / pT of the jet) of particle 31 (ParT input; empty slot: ln 1e-8)
  Q.lund1_lndelta          ln Δ of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund1_lnkt             ln kT of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund2_lndelta          ln Δ of the 2. primary C/A splitting (ln 1e-8 if none)
  Q.lund2_lnz              ln z of the 2. primary C/A splitting (ln 1e-8 if none)
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
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_abs_d0             largest |d0| among the charged particles [mm]
  Q.max_abs_dz             largest |dz| among the charged particles [mm]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.mean_eta2              pT-weighted mean Δη²
  Q.min_pair_mass          smallest pair mass among particles 0, 1, 2 [GeV]
  Q.mres_pruned_mass       pruned mass: C/A tree, a merge with z < 0.1 and ΔR > m/pT of the jet keeps only its harder branch [GeV]
  Q.mres_sd_mass_b0z005    soft-drop mass, C/A, β = 0, z_cut = 0.05 [GeV]
  Q.mres_sd_mass_b0z02     soft-drop mass, C/A, β = 0, z_cut = 0.2 [GeV]
  Q.mres_sd_mass_b1z01     soft-drop mass, C/A, β = 1, z_cut = 0.1 [GeV]
  Q.mres_sd_mass_b2z01     soft-drop mass, C/A, β = 2, z_cut = 0.1 [GeV]
  Q.mres_sd_prong_mass1    mass of the harder branch at the soft-drop splitting (β = 0, z_cut = 0.1) [GeV]
  Q.mres_sd_prong_mass2    mass of the softer branch at the soft-drop splitting (β = 0, z_cut = 0.1) [GeV]
  Q.mres_sd_rg_b0z02       soft-drop R_g, β = 0, z_cut = 0.2
  Q.mres_sd_rg_b1z01       soft-drop R_g, β = 1, z_cut = 0.1
  Q.mres_sd_zg_b1z01       soft-drop z_g, β = 1, z_cut = 0.1
  Q.mres_sd_zg_b2z01       soft-drop z_g, β = 2, z_cut = 0.1
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
  Q.n_pairs_kt_above_10    number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 10 GeV
  Q.n_pairs_kt_above_3     number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 3 GeV
  Q.n_pairs_kt_above_30    number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 30 GeV
  Q.n_particles            number of real particles (pT > 0)
  Q.n_photon               number of photons
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.n_s3d_above_10         number of charged particles with 3d significance > 10
  Q.n_s3d_above_3          number of charged particles with 3d significance > 3
  Q.n_sd0_above_10         number of charged particles with d0 significance > 10
  Q.n_sd0_above_2          number of charged particles with d0 significance > 2
  Q.n_sd0_above_3          number of charged particles with d0 significance > 3
  Q.n_sd0_above_5          number of charged particles with d0 significance > 5
  Q.n_sdz_above_2          number of charged particles with dz significance > 2
  Q.n_sdz_above_5          number of charged particles with dz significance > 5
  Q.nca_kt_above_2         number of C/A subjets when every branching with kT = min(pT)·ΔR > 2 GeV is split
  Q.nca_kt_above_20        number of C/A subjets when every branching with kT = min(pT)·ΔR > 20 GeV is split
  Q.nca_sj4_pair2nd_over_mass second largest mass of two of the 4 subjets over the jet mass
  Q.nca_sj4_pair_mass_2nd  second largest mass of two of the 4 subjets [GeV]
  Q.nca_sj4_pairmax_over_mass largest mass of two of the 4 subjets over the jet mass
  Q.pair_max_lnkt          largest ln kT among all pairs
  Q.pair_max_lnm2          largest ln m² among all pairs
  Q.pair_mean_lndelta      zᵢzⱼ-weighted mean of ln ΔRᵢⱼ over all pairs
  Q.pair_mean_lnkt         zᵢzⱼ-weighted mean of ln kT = ln(min(pTᵢ, pTⱼ)·ΔRᵢⱼ) over all pairs
  Q.pair_mean_lnm2         zᵢzⱼ-weighted mean of ln mᵢⱼ² (massless) over all pairs
  Q.phi_1                  Δφ of particle 1 (ParT input; empty slot: 0)
  Q.phi_3                  Δφ of particle 3 (ParT input; empty slot: 0)
  Q.phi_4                  Δφ of particle 4 (ParT input; empty slot: 0)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.pt1_over_pt0           pT1 / pT0
  Q.pt_balance01           min(pT0, pT1) / (pT0 + pT1)
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pz_lnd0                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -9 < ln ΔRᵢⱼ ≤ -3
  Q.pz_lnd1                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -3 < ln ΔRᵢⱼ ≤ -2
  Q.pz_lnd2                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -2 < ln ΔRᵢⱼ ≤ -1
  Q.pz_lnd3                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -1 < ln ΔRᵢⱼ ≤ 9
  Q.pz_lnkt0               Σ zᵢzⱼ over the pairs of the 40 hardest particles with -9 < ln kT ≤ 0
  Q.pz_lnkt1               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 0 < ln kT ≤ 1
  Q.pz_lnkt3               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 2 < ln kT ≤ 3
  Q.pz_lnkt4               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 3 < ln kT ≤ 9
  Q.sd_mass                soft-drop groomed mass, C/A on the 128 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sdb_0_n                number of tracks with d0/σ ≤ -3
  Q.sdb_1_n                number of tracks with -3 < d0/σ ≤ -1
  Q.sdb_2_n                number of tracks with -1 < d0/σ ≤ 1
  Q.sdb_2_z                pT share of the tracks with -1 < d0/σ ≤ 1
  Q.sdb_4_n                number of tracks with 3 < d0/σ ≤ 10
  Q.sdb_4_z                pT share of the tracks with 3 < d0/σ ≤ 10
  Q.sdb_5_n                number of tracks with d0/σ > 10
  Q.sdb_5_z                pT share of the tracks with d0/σ > 10
  Q.sdb_jp_all             Σ −ln P(|d0/σ|) over the tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.sip_3d_1               the 1. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sip_3d_2               the 2. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sip_3d_3               the 3. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.sj3_z2                 pT share of subjet 2 of 3 (by pT)
  Q.sj4_dr_min             smallest distance among the 4 subjet axes
  Q.sj4_pair_mass_max      largest mass of two of the 4 subjets [GeV]
  Q.sj4_pair_mass_min      smallest mass of two of the 4 subjets [GeV]
  Q.sj4_zsoft              pT share of the softest of 4 subjets
  Q.sjf_2_1_mass_d3        subjet 1 of 2 (k-means axes, by pT): mass (E) of the tracks with |d0/σ| > 3 [GeV]
  Q.sjf_2_1_max3d          subjet 1 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_1_maxsd0         subjet 1 of 2 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_2_1_n_d3           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_2_1_n_d5           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_2_1_z_d3           subjet 1 of 2 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_2_2_max3d          subjet 2 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_2_maxsd0         subjet 2 of 2 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_2_2_n_d3           subjet 2 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_2_2_z_d3           subjet 2 of 2 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_2_n2disp           number of the 2 subjets with at least 2 tracks with |d0/σ| > 3
  Q.sjf_3_1_max3d          subjet 1 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_1_n_d3           subjet 1 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_3_1_z_d3           subjet 1 of 3 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_3_2_mass_d3        subjet 2 of 3 (k-means axes, by pT): mass (E) of the tracks with |d0/σ| > 3 [GeV]
  Q.sjf_3_2_max3d          subjet 2 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_2_n_d3           subjet 2 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_3_3_n_d3           subjet 3 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_3_3_z_d3           subjet 3 of 3 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_3_n2disp           number of the 3 subjets with at least 2 tracks with |d0/σ| > 3
  Q.sjf_4_1_mass_d3        subjet 1 of 4 (k-means axes, by pT): mass (E) of the tracks with |d0/σ| > 3 [GeV]
  Q.sjf_4_1_max3d          subjet 1 of 4 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_4_1_n_d3           subjet 1 of 4 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_4_1_z_d3           subjet 1 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_2_z_d3           subjet 2 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_3_n_d3           subjet 3 of 4 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_4_3_n_d5           subjet 3 of 4 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_4_3_z_d3           subjet 3 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_4_mass_d3        subjet 4 of 4 (k-means axes, by pT): mass (E) of the tracks with |d0/σ| > 3 [GeV]
  Q.sjf_4_4_max3d          subjet 4 of 4 (k-means axes, by pT): largest 3D significance (0 if no track)
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
  Q.sjq_3_1_k05            Σ q pT^κ / (Σ pT)^κ of subjet 1 of 3, κ = 0.5
  Q.sjq_3_2_k1             Σ q pT^κ / (Σ pT)^κ of subjet 2 of 3, κ = 1
  Q.sjq_3_2_nch            number of charged particles in subjet 2 of 3
  Q.sjq_3_3_k1             Σ q pT^κ / (Σ pT)^κ of subjet 3 of 3, κ = 1
  Q.sjq_3_3_nch            number of charged particles in subjet 3 of 3
  Q.sjq_3_prod_k1          product of the charges (κ = 1) of the two hardest of 3 subjets
  Q.sjq_3_sumabs_k1        |sum| of the charges (κ = 1) of the two hardest of 3 subjets
  Q.sum_charge             total charge of the particles
  Q.sum_e                  total energy of the particles [GeV]
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sv_1_dr                hardest displaced-track cluster: ΔR of its centre from the jet axis (0 if none)
  Q.sv_1_n                 hardest displaced-track cluster: number of tracks (0 if none)
  Q.sv_1_sd0_sum           hardest displaced-track cluster: Σ d0/σ of its tracks (0 if none)
  Q.sv_1_z                 hardest displaced-track cluster: pT share of the jet (0 if none)
  Q.sv_2_dr                2nd displaced-track cluster: ΔR of its centre from the jet axis (0 if none)
  Q.sv_2_n                 2nd displaced-track cluster: number of tracks (0 if none)
  Q.sv_2_sd0_sum           2nd displaced-track cluster: Σ d0/σ of its tracks (0 if none)
  Q.sv_2_z                 2nd displaced-track cluster: pT share of the jet (0 if none)
  Q.sv_n                   number of anti-kT R = 0.1 clusters of the tracks with d0/σ > 3
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
  Q.td0_13                 tanh(d0 [mm]) of particle 13 (ParT input; empty slot: 0)
  Q.td0_50                 tanh(d0 [mm]) of particle 50 (ParT input; empty slot: 0)
  Q.tdz_0                  tanh(dz [mm]) of particle 0 (ParT input; empty slot: 0)
  Q.tdz_1                  tanh(dz [mm]) of particle 1 (ParT input; empty slot: 0)
  Q.tdz_10                 tanh(dz [mm]) of particle 10 (ParT input; empty slot: 0)
  Q.tdz_13                 tanh(dz [mm]) of particle 13 (ParT input; empty slot: 0)
  Q.tdz_2                  tanh(dz [mm]) of particle 2 (ParT input; empty slot: 0)
  Q.tdz_72                 tanh(dz [mm]) of particle 72 (ParT input; empty slot: 0)
  Q.z_charged              pT share of charged particles
  Q.z_charged_had          pT share of charged hadrons
  Q.z_displaced3           pT share of the charged particles with |d0|/σ > 3
  Q.z_displaced5           pT share of the charged particles with |d0|/σ > 5
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.z_dr_0p4_up            pT share of the particles with ΔR ≥ 0.4
  Q.z_electron             pT share of electrons
  Q.z_muon                 pT share of muons
  Q.z_neutral              pT share of neutral particles
  Q.z_neutral_had          pT share of neutral hadrons
  Q.z_photon               pT share of photons
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
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
        D3_b2=ecfb('e4', 2) * ecfb('e2', 2) ** 3 / max(ecfb('e3', 2) ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M2_b05=ecfb('g31', 0.5) / max(ecfb('e2', 0.5), 1e-30),
        M2_b2=ecfb('g31', 2) / max(ecfb('e2', 2), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        M3_b2=ecfb('g41', 2) / max(ecfb('g31', 2), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N2_b05=ecfb('g32', 0.5) / max(ecfb('e2', 0.5) ** 2, 1e-30),
        N2_b2=ecfb('g32', 2) / max(ecfb('e2', 2) ** 2, 1e-30),
        N3_b05=ecfb('g42', 0.5) / max(ecfb('g31', 0.5) ** 2, 1e-30),
        N3_b2=ecfb('g42', 2) / max(ecfb('g31', 2) ** 2, 1e-30),
        ak02_1_mass=pq('ak02_1_mass'),
        ak02_1_n_disp3=pq('ak02_1_n_disp3'),
        ak02_1_n_lep=pq('ak02_1_n_lep'),
        ak02_1_z=pq('ak02_1_z'),
        ak02_2_n_lep=pq('ak02_2_n_lep'),
        ak02_2_sd0_3=pq('ak02_2_sd0_3'),
        ak02_2_z=pq('ak02_2_z'),
        ak02_3_mass=pq('ak02_3_mass'),
        ak02_3_n_lep=pq('ak02_3_n_lep'),
        ak02_3_z=pq('ak02_3_z'),
        ak02_3_z_disp3=pq('ak02_3_z_disp3'),
        ak02_dr12=pq('ak02_dr12'),
        ak02_dr13=pq('ak02_dr13'),
        ak02_dr23=pq('ak02_dr23'),
        ak02_min12_n_disp3=pq('ak02_min12_n_disp3'),
        ak02_n=pq('ak02_n'),
        charge_76=pfeat(76, 'charge'),
        d0err_10=pfeat(10, 'd0err'),
        d0err_34=pfeat(34, 'd0err'),
        d0err_49=pfeat(49, 'd0err'),
        d0err_52=pfeat(52, 'd0err'),
        d0err_60=pfeat(60, 'd0err'),
        dc_1_charge=pq('dc_1_charge'),
        dc_1_n_disp3=pq('dc_1_n_disp3'),
        dc_1_n_lep=pq('dc_1_n_lep'),
        dc_1_sd0_1=pq('dc_1_sd0_1'),
        dc_1_z=pq('dc_1_z'),
        dc_2_charge=pq('dc_2_charge'),
        dc_2_jp=pq('dc_2_jp'),
        dc_2_n_disp3=pq('dc_2_n_disp3'),
        dc_2_n_lep=pq('dc_2_n_lep'),
        dc_3_charge=pq('dc_3_charge'),
        dc_3_jp=pq('dc_3_jp'),
        dc_3_n_lep=pq('dc_3_n_lep'),
        dc_3_z=pq('dc_3_z'),
        dc_4_jp=pq('dc_4_jp'),
        dc_4_mass_disp3=pq('dc_4_mass_disp3'),
        dc_4_z=pq('dc_4_z'),
        dc_mass_disp_2nd=pq('dc_mass_disp_2nd'),
        dc_n=pq('dc_n'),
        dc_ntag=pq('dc_ntag'),
        dc_pair_mass_min=pq('dc_pair_mass_min'),
        dc_split1_dr=pq('dc_split1_dr'),
        dc_split1_kt=pq('dc_split1_kt'),
        dc_split1_mass=pq('dc_split1_mass'),
        dc_split2_dr=pq('dc_split2_dr'),
        dc_split2_kt=pq('dc_split2_kt'),
        dc_split2_mass=pq('dc_split2_mass'),
        dc_tag_2nd=pq('dc_tag_2nd'),
        dc_tag_max=pq('dc_tag_max'),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        dr_0=pfeat(0, 'dr'),
        dr_1=pfeat(1, 'dr'),
        dr_11=pfeat(11, 'dr'),
        dr_18=pfeat(18, 'dr'),
        dr_2=pfeat(2, 'dr'),
        dr_21=pfeat(21, 'dr'),
        dr_28=pfeat(28, 'dr'),
        dr_3=pfeat(3, 'dr'),
        dr_39=pfeat(39, 'dr'),
        dr_68=pfeat(68, 'dr'),
        dr_86=pfeat(86, 'dr'),
        dr_94=pfeat(94, 'dr'),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        dzerr_0=pfeat(0, 'dzerr'),
        dzerr_1=pfeat(1, 'dzerr'),
        dzerr_22=pfeat(22, 'dzerr'),
        dzerr_32=pfeat(32, 'dzerr'),
        dzerr_75=pfeat(75, 'dzerr'),
        dzerr_76=pfeat(76, 'dzerr'),
        e2=e2,
        e3=ecf('e3'),
        e3_b05=ecfb('e3', 0.5),
        e3_b2=ecfb('e3', 2),
        e4=ecf('e4'),
        e4_b2=ecfb('e4', 2),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        ecf_g31=ecfb('g31', 1),
        ecf_g41=ecfb('g41', 1),
        ecf_g42=ecfb('g42', 1),
        ecf_g43=ecfb('g43', 1),
        eta_1=pfeat(1, 'eta'),
        eta_14=pfeat(14, 'eta'),
        eta_28=pfeat(28, 'eta'),
        eta_37=pfeat(37, 'eta'),
        eta_4=pfeat(4, 'eta'),
        eta_8=pfeat(8, 'eta'),
        eta_9=pfeat(9, 'eta'),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        ischhad_0=pfeat(0, 'ischhad'),
        ischhad_10=pfeat(10, 'ischhad'),
        iselectron_1=pfeat(1, 'iselectron'),
        iselectron_11=pfeat(11, 'iselectron'),
        iselectron_12=pfeat(12, 'iselectron'),
        iselectron_27=pfeat(27, 'iselectron'),
        ismuon_0=pfeat(0, 'ismuon'),
        ismuon_3=pfeat(3, 'ismuon'),
        isnhad_34=pfeat(34, 'isnhad'),
        isnhad_8=pfeat(8, 'isnhad'),
        isphoton_19=pfeat(19, 'isphoton'),
        isphoton_29=pfeat(29, 'isphoton'),
        isphoton_40=pfeat(40, 'isphoton'),
        isphoton_42=pfeat(42, 'isphoton'),
        isphoton_53=pfeat(53, 'isphoton'),
        jd_3d_4=pq('jd_3d_4'),
        jd_3d_5=pq('jd_3d_5'),
        jd_3d_6=pq('jd_3d_6'),
        jd_n_d3_pt1=pq('jd_n_d3_pt1'),
        jd_sum_abs_sd0_top3=pq('jd_sum_abs_sd0_top3'),
        jd_sum_abs_sd0_top5=pq('jd_sum_abs_sd0_top5'),
        jet_abs_eta=abs(jet_eta),
        jet_charge=sum(charge[i] * z[i] for i in real),
        jet_charge_k03=sum(charge[i] * z[i] ** 0.3 for i in real),
        jet_charge_k05=sum(charge[i] * z[i] ** 0.5 for i in real),
        jet_e=jet_energy,
        kt2_1_n_lep=pq('kt2_1_n_lep'),
        kt2_1_sd0_3=pq('kt2_1_sd0_3'),
        kt2_1_z_disp3=pq('kt2_1_z_disp3'),
        kt2_2_n_disp3=pq('kt2_2_n_disp3'),
        kt2_2_n_lep=pq('kt2_2_n_lep'),
        kt2_2_z_disp3=pq('kt2_2_z_disp3'),
        kt2_dr12=pq('kt2_dr12'),
        kt2_min12_jp=pq('kt2_min12_jp'),
        kt2_min12_mass_disp3=pq('kt2_min12_mass_disp3'),
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
        lne_1=pfeat(1, 'lne'),
        lne_16=pfeat(16, 'lne'),
        lne_17=pfeat(17, 'lne'),
        lne_21=pfeat(21, 'lne'),
        lne_3=pfeat(3, 'lne'),
        lne_4=pfeat(4, 'lne'),
        lne_40=pfeat(40, 'lne'),
        lne_5=pfeat(5, 'lne'),
        lnerel_0=pfeat(0, 'lnerel'),
        lnerel_1=pfeat(1, 'lnerel'),
        lnerel_4=pfeat(4, 'lnerel'),
        lnerel_78=pfeat(78, 'lnerel'),
        lnpt_31=pfeat(31, 'lnpt'),
        lnpt_32=pfeat(32, 'lnpt'),
        lnpt_40=pfeat(40, 'lnpt'),
        lnpt_43=pfeat(43, 'lnpt'),
        lnpt_65=pfeat(65, 'lnpt'),
        lnptrel_25=pfeat(25, 'lnptrel'),
        lnptrel_3=pfeat(3, 'lnptrel'),
        lnptrel_31=pfeat(31, 'lnptrel'),
        lund1_lndelta=lund(1, 'lndelta'),
        lund1_lnkt=lund(1, 'lnkt'),
        lund2_lndelta=lund(2, 'lndelta'),
        lund2_lnz=lund(2, 'lnz'),
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
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        max_abs_d0=max([abs(d0[i]) for i in real if charge[i] != 0] or [0.0]),
        max_abs_dz=max([abs(dz[i]) for i in real if charge[i] != 0] or [0.0]),
        max_dr=max(dr[i] for i in real),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        mres_pruned_mass=pq('mres_pruned_mass'),
        mres_sd_mass_b0z005=pq('mres_sd_mass_b0z005'),
        mres_sd_mass_b0z02=pq('mres_sd_mass_b0z02'),
        mres_sd_mass_b1z01=pq('mres_sd_mass_b1z01'),
        mres_sd_mass_b2z01=pq('mres_sd_mass_b2z01'),
        mres_sd_prong_mass1=pq('mres_sd_prong_mass1'),
        mres_sd_prong_mass2=pq('mres_sd_prong_mass2'),
        mres_sd_rg_b0z02=pq('mres_sd_rg_b0z02'),
        mres_sd_rg_b1z01=pq('mres_sd_rg_b1z01'),
        mres_sd_zg_b1z01=pq('mres_sd_zg_b1z01'),
        mres_sd_zg_b2z01=pq('mres_sd_zg_b2z01'),
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
        n_pairs_kt_above_10=paircount(10),
        n_pairs_kt_above_3=paircount(3),
        n_pairs_kt_above_30=paircount(30),
        n_particles=len(real),
        n_photon=sum(1 for i in real if ptype[i] == 3),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        n_s3d_above_10=nsig('3d', 10),
        n_s3d_above_3=nsig('3d', 3),
        n_sd0_above_10=nsig('d0', 10),
        n_sd0_above_2=nsig('d0', 2),
        n_sd0_above_3=nsig('d0', 3),
        n_sd0_above_5=nsig('d0', 5),
        n_sdz_above_2=nsig('dz', 2),
        n_sdz_above_5=nsig('dz', 5),
        nca_kt_above_2=pq('nca_kt_above_2'),
        nca_kt_above_20=pq('nca_kt_above_20'),
        nca_sj4_pair2nd_over_mass=pq('nca_sj4_pair2nd_over_mass'),
        nca_sj4_pair_mass_2nd=pq('nca_sj4_pair_mass_2nd'),
        nca_sj4_pairmax_over_mass=pq('nca_sj4_pairmax_over_mass'),
        pair_max_lnkt=pairmax('lnkt'),
        pair_max_lnm2=pairmax('lnm2'),
        pair_mean_lndelta=pairsum('lndelta'),
        pair_mean_lnkt=pairsum('lnkt'),
        pair_mean_lnm2=pairsum('lnm2'),
        phi_1=pfeat(1, 'phi'),
        phi_3=pfeat(3, 'phi'),
        phi_4=pfeat(4, 'phi'),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        pt_balance01=min(pt[0], pt[1]) / max(pt[0] + pt[1], 1e-9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pz_lnd0=pq('pz_lnd0'),
        pz_lnd1=pq('pz_lnd1'),
        pz_lnd2=pq('pz_lnd2'),
        pz_lnd3=pq('pz_lnd3'),
        pz_lnkt0=pq('pz_lnkt0'),
        pz_lnkt1=pq('pz_lnkt1'),
        pz_lnkt3=pq('pz_lnkt3'),
        pz_lnkt4=pq('pz_lnkt4'),
        sd_mass=softdrop("mass"),
        sd_rg=softdrop("rg"),
        sdb_0_n=pq('sdb_0_n'),
        sdb_1_n=pq('sdb_1_n'),
        sdb_2_n=pq('sdb_2_n'),
        sdb_2_z=pq('sdb_2_z'),
        sdb_4_n=pq('sdb_4_n'),
        sdb_4_z=pq('sdb_4_z'),
        sdb_5_n=pq('sdb_5_n'),
        sdb_5_z=pq('sdb_5_z'),
        sdb_jp_all=pq('sdb_jp_all'),
        sip_3d_1=sip('3d', 1),
        sip_3d_2=sip('3d', 2),
        sip_3d_3=sip('3d', 3),
        sj2_dr=subjets(2)["dr"][0],
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        sj2_zsoft=subjets(2)["z"][1],
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        sj3_dr_max=max(subjets(3)["dr"]),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_z1=subjets(3)["z"][0],
        sj3_z2=subjets(3)["z"][1],
        sj4_dr_min=min(subjets(4)["dr"]),
        sj4_pair_mass_max=max(subjets(4)["mpair"]),
        sj4_pair_mass_min=min(subjets(4)["mpair"]),
        sj4_zsoft=subjets(4)["z"][3],
        sjf_2_1_mass_d3=pq('sjf_2_1_mass_d3'),
        sjf_2_1_max3d=pq('sjf_2_1_max3d'),
        sjf_2_1_maxsd0=pq('sjf_2_1_maxsd0'),
        sjf_2_1_n_d3=pq('sjf_2_1_n_d3'),
        sjf_2_1_n_d5=pq('sjf_2_1_n_d5'),
        sjf_2_1_z_d3=pq('sjf_2_1_z_d3'),
        sjf_2_2_max3d=pq('sjf_2_2_max3d'),
        sjf_2_2_maxsd0=pq('sjf_2_2_maxsd0'),
        sjf_2_2_n_d3=pq('sjf_2_2_n_d3'),
        sjf_2_2_z_d3=pq('sjf_2_2_z_d3'),
        sjf_2_n2disp=pq('sjf_2_n2disp'),
        sjf_3_1_max3d=pq('sjf_3_1_max3d'),
        sjf_3_1_n_d3=pq('sjf_3_1_n_d3'),
        sjf_3_1_z_d3=pq('sjf_3_1_z_d3'),
        sjf_3_2_mass_d3=pq('sjf_3_2_mass_d3'),
        sjf_3_2_max3d=pq('sjf_3_2_max3d'),
        sjf_3_2_n_d3=pq('sjf_3_2_n_d3'),
        sjf_3_3_n_d3=pq('sjf_3_3_n_d3'),
        sjf_3_3_z_d3=pq('sjf_3_3_z_d3'),
        sjf_3_n2disp=pq('sjf_3_n2disp'),
        sjf_4_1_mass_d3=pq('sjf_4_1_mass_d3'),
        sjf_4_1_max3d=pq('sjf_4_1_max3d'),
        sjf_4_1_n_d3=pq('sjf_4_1_n_d3'),
        sjf_4_1_z_d3=pq('sjf_4_1_z_d3'),
        sjf_4_2_z_d3=pq('sjf_4_2_z_d3'),
        sjf_4_3_n_d3=pq('sjf_4_3_n_d3'),
        sjf_4_3_n_d5=pq('sjf_4_3_n_d5'),
        sjf_4_3_z_d3=pq('sjf_4_3_z_d3'),
        sjf_4_4_mass_d3=pq('sjf_4_4_mass_d3'),
        sjf_4_4_max3d=pq('sjf_4_4_max3d'),
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
        sjq_3_1_k05=pq('sjq_3_1_k05'),
        sjq_3_2_k1=pq('sjq_3_2_k1'),
        sjq_3_2_nch=pq('sjq_3_2_nch'),
        sjq_3_3_k1=pq('sjq_3_3_k1'),
        sjq_3_3_nch=pq('sjq_3_3_nch'),
        sjq_3_prod_k1=pq('sjq_3_prod_k1'),
        sjq_3_sumabs_k1=pq('sjq_3_sumabs_k1'),
        sum_charge=sum(charge[i] for i in real),
        sum_e=sum(energy[i] for i in real),
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sv_1_dr=pq('sv_1_dr'),
        sv_1_n=pq('sv_1_n'),
        sv_1_sd0_sum=pq('sv_1_sd0_sum'),
        sv_1_z=pq('sv_1_z'),
        sv_2_dr=pq('sv_2_dr'),
        sv_2_n=pq('sv_2_n'),
        sv_2_sd0_sum=pq('sv_2_sd0_sum'),
        sv_2_z=pq('sv_2_z'),
        sv_n=pq('sv_n'),
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
        td0_13=pfeat(13, 'td0'),
        td0_50=pfeat(50, 'td0'),
        tdz_0=pfeat(0, 'tdz'),
        tdz_1=pfeat(1, 'tdz'),
        tdz_10=pfeat(10, 'tdz'),
        tdz_13=pfeat(13, 'tdz'),
        tdz_2=pfeat(2, 'tdz'),
        tdz_72=pfeat(72, 'tdz'),
        z_charged=sum(z[i] for i in real if charge[i] != 0),
        z_charged_had=sum(z[i] for i in real if ptype[i] == 1),
        z_displaced3=displaced(3, 'z'),
        z_displaced5=displaced(5, 'z'),
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        z_dr_0p4_up=sum(z[i] for i in real if 0.4 <= dr[i] < 10),
        z_electron=sum(z[i] for i in real if ptype[i] == 4),
        z_muon=sum(z[i] for i in real if ptype[i] == 5),
        z_neutral=sum(z[i] for i in real if charge[i] == 0),
        z_neutral_had=sum(z[i] for i in real if ptype[i] == 2),
        z_photon=sum(z[i] for i in real if ptype[i] == 3),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
    )


def neuron_0(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.005213402
    )
    return z


def neuron_1(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.325262e-06
    )
    return z


def neuron_2(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.447298e-06
    )
    return z


def neuron_3(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.399834e-06
    )
    return z


def neuron_4(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.792135e-06
    )
    return z


def neuron_5(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.156917e-06
    )
    return z


def neuron_6(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.770105e-05
    )
    return z


def neuron_7(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.430819e-06
    )
    return z


def neuron_8(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.926468e-06
    )
    return z


def neuron_9(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.443659e-06
    )
    return z


def neuron_10(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.570104e-05
    )
    return z


def neuron_11(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.887712e-05
    )
    return z


def neuron_12(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.099297e-06
    )
    return z


def neuron_13(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.422908e-06
    )
    return z


def neuron_14(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.845303e-06
    )
    return z


def neuron_15(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.459521e-06
    )
    return z


def neuron_16(Q):
    # scale S = 16.05; each line: share * term / its average size
    z = 16.04942 * (0.1123632
        - 0.1202089 * Q.z_charged_had / 0.5066138   # -12.0%  z_charged_had
        + 0.1040603 * (12.22004 * 0.5 * ((27.3236 - Q.lep_ptrel) / 12.22004) * (1 + math.erf(((27.3236 - Q.lep_ptrel) / 12.22004) / math.sqrt(2)))) / 21.35932   # +10.4%  lep_ptrel < 27.32
        - 0.08814837 * (146.0039 * 0.5 * ((262.8757 - Q.jd_sum_abs_sd0_top3) / 146.0039) * (1 + math.erf(((262.8757 - Q.jd_sum_abs_sd0_top3) / 146.0039) / math.sqrt(2)))) / 120.6265   # -8.8%  jd_sum_abs_sd0_top3 < 262.9
        - 0.07655538 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 6.361002   # -7.7%  n_s3d_above_3 < 10
        + 0.07282296 * (148.6964 * 0.5 * ((288.7714 - Q.jd_sum_abs_sd0_top5) / 148.6964) * (1 + math.erf(((288.7714 - Q.jd_sum_abs_sd0_top5) / 148.6964) / math.sqrt(2)))) / 132.6717   # +7.3%  jd_sum_abs_sd0_top5 < 288.8
        - 0.0581426 * (0.08093679 * 0.5 * ((0.2801368 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.2801368 - Q.z_displaced5) / 0.08093679) / math.sqrt(2)))) / 0.1986202   # -5.8%  z_displaced5 < 0.2801
        - 0.05483689 * (0.05422308 * 0.5 * ((Q.z_neutral - 0.1384639) / 0.05422308) * (1 + math.erf(((Q.z_neutral - 0.1384639) / 0.05422308) / math.sqrt(2)))) / 0.2670039   # -5.5%  z_neutral > 0.1385
        + 0.04723392 * (0.08093679 * 0.5 * ((0.2801368 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.2801368 - Q.z_displaced5) / 0.08093679) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2)))) / 29.32309   # +4.7%  z_displaced5 < 0.2801 and jd_3d_4 < 172.9
        + 0.04287673 * Q.lep_ptrel / 7.315413   # +4.3%  lep_ptrel
        + 0.0190911 * (164.9991 * 0.5 * ((226.3008 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((226.3008 - Q.sip_3d_2) / 164.9991) / math.sqrt(2)))) / 133.7832   # +1.9%  sip_3d_2 < 226.3
        + 0.0159569 * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) / 0.1153058   # +1.6%  sjq_2_prod_k1 < 0.09803
        - 0.01434554 * (4.214028 * 0.5 * ((9.982976 - Q.jd_3d_4) / 4.214028) * (1 + math.erf(((9.982976 - Q.jd_3d_4) / 4.214028) / math.sqrt(2)))) / 4.932055   # -1.4%  jd_3d_4 < 9.983
        - 0.01403748 * (0.07843207 * 0.5 * ((Q.lep_z - 0.1275041) / 0.07843207) * (1 + math.erf(((Q.lep_z - 0.1275041) / 0.07843207) / math.sqrt(2)))) / 0.05672893   # -1.4%  lep_z > 0.1275
        + 0.0138931 * (112.8732 * 0.5 * ((206.0654 - Q.sip_3d_1) / 112.8732) * (1 + math.erf(((206.0654 - Q.sip_3d_1) / 112.8732) / math.sqrt(2)))) / 95.8954   # +1.4%  sip_3d_1 < 206.1
        - 0.01336864 * (17.91682 * 0.5 * ((33.44111 - Q.jd_3d_4) / 17.91682) * (1 + math.erf(((33.44111 - Q.jd_3d_4) / 17.91682) / math.sqrt(2)))) / 22.47608   # -1.3%  jd_3d_4 < 33.44
        - 0.01220151 * (0.560057 * 0.5 * ((1.777787 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.777787 - Q.mass_displaced5) / 0.560057) / math.sqrt(2)))) / 0.8185445   # -1.2%  mass_displaced5 < 1.778
        + 0.01192531 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) / 2.75999   # +1.2%  max_abs_d0 < 5.812
        + 0.00993858 * (6.5 * 0.5 * ((41.0 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((41.0 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2)))) / 7.611779   # +1.0%  n_pairs_kt_above_3 < 41
        - 0.008749707 * (0.03445022 * 0.5 * ((0.04982189 - Q.lepsj_3_dr) / 0.03445022) * (1 + math.erf(((0.04982189 - Q.lepsj_3_dr) / 0.03445022) / math.sqrt(2)))) / 0.03714658   # -0.9%  lepsj_3_dr < 0.04982
        + 0.008265065 * (248.3135 * 0.5 * ((175.9957 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((175.9957 - Q.sip_3d_3) / 248.3135) / math.sqrt(2)))) / 107.6838   # +0.8%  sip_3d_3 < 176
        + 0.007264997 * (0.02907957 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) / math.sqrt(2)))) / 0.1710697   # +0.7%  sjq_2_sumabs_k1 > 0.2131
        + 0.007077302 * (0.03043217 * 0.5 * ((0.1114751 - Q.sv_1_z) / 0.03043217) * (1 + math.erf(((0.1114751 - Q.sv_1_z) / 0.03043217) / math.sqrt(2)))) / 0.07544827   # +0.7%  sv_1_z < 0.1115
        + 0.00677884 * (3.485807 * 0.5 * ((Q.mass_top40 - 119.1279) / 3.485807) * (1 + math.erf(((Q.mass_top40 - 119.1279) / 3.485807) / math.sqrt(2)))) / 9.653102   # +0.7%  mass_top40 > 119.1
        - 0.006475787 * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.6533455   # -0.6%  kt2_2_n_lep < 1
        + 0.006418854 * Q.n_photon / 16.03902   # +0.6%  n_photon
        - 0.006310335 * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) * (1.0 * 0.5 * ((5.0 - Q.sjf_4_1_n_d3) / 1.0) * (1 + math.erf(((5.0 - Q.sjf_4_1_n_d3) / 1.0) / math.sqrt(2)))) / 0.4371997   # -0.6%  sjq_2_prod_k1 < 0.09803 and sjf_4_1_n_d3 < 5
        + 0.006061864 * (3.043206 * 0.5 * ((55.84091 - Q.mass_charged) / 3.043206) * (1 + math.erf(((55.84091 - Q.mass_charged) / 3.043206) / math.sqrt(2)))) / 6.64845   # +0.6%  mass_charged < 55.84
        + 0.006013242 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.001717815 * 0.5 * ((Q.M3 - 0.02540381) / 0.001717815) * (1 + math.erf(((Q.M3 - 0.02540381) / 0.001717815) / math.sqrt(2)))) / 4.185796e-06   # +0.6%  e3_b2 < 0.0004127 and M3 > 0.0254
        - 0.005634021 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.151853 * 0.5 * ((0.1373477 - Q.z_electron) / 0.151853) * (1 + math.erf(((0.1373477 - Q.z_electron) / 0.151853) / math.sqrt(2)))) / 0.6215838   # -0.6%  n_s3d_above_3 < 10 and z_electron < 0.1373
        - 0.005516613 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.2798636 * 0.5 * ((1.720031 - Q.sjq_2_sumabs_k03) / 0.2798636) * (1 + math.erf(((1.720031 - Q.sjq_2_sumabs_k03) / 0.2798636) / math.sqrt(2)))) / 6.593886   # -0.6%  n_s3d_above_3 < 10 and sjq_2_sumabs_k03 < 1.72
        - 0.005483482 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2)))) / 9.427083   # -0.5%  n_dr_0p4_up < 15
        + 0.005453566 * (0.03445022 * 0.5 * ((0.04982189 - Q.lepsj_3_dr) / 0.03445022) * (1 + math.erf(((0.04982189 - Q.lepsj_3_dr) / 0.03445022) / math.sqrt(2)))) * (467.791 * 0.5 * ((593.8527 - Q.dc_1_sd0_1) / 467.791) * (1 + math.erf(((593.8527 - Q.dc_1_sd0_1) / 467.791) / math.sqrt(2)))) / 17.9989   # +0.5%  lepsj_3_dr < 0.04982 and dc_1_sd0_1 < 593.9
        + 0.00516227 * (3.20498 * 0.5 * ((Q.mass_neutral - 34.02385) / 3.20498) * (1 + math.erf(((Q.mass_neutral - 34.02385) / 3.20498) / math.sqrt(2)))) / 15.79025   # +0.5%  mass_neutral > 34.02
        - 0.005126156 * (1.0 * 0.5 * ((Q.sdb_2_n - 5.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 5.0) / 1.0) / math.sqrt(2)))) / 5.768261   # -0.5%  sdb_2_n > 5
        + 0.004963165 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sv_1_n - 1.0) / 1.0) * (1 + math.erf(((Q.sv_1_n - 1.0) / 1.0) / math.sqrt(2)))) / 1.409945   # +0.5%  n_s3d_above_3 < 10 and sv_1_n > 1
        + 0.004824019 * (1.0 * 0.5 * ((4.0 - Q.n_sd0_above_5) / 1.0) * (1 + math.erf(((4.0 - Q.n_sd0_above_5) / 1.0) / math.sqrt(2)))) / 1.837681   # +0.5%  n_sd0_above_5 < 4
        - 0.004592495 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) / 1.215201   # -0.5%  n_dr_0p4_up < 4
        - 0.004577707 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) / 0.9643417   # -0.5%  lep_ptrel > 43.21
        + 0.004221521 * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2)))) * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) / 0.01061807   # +0.4%  dc_n > 1 and M2_b2 < 0.04179
        - 0.004070676 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) / math.sqrt(2)))) / 3.50049   # -0.4%  mres_sd_mass_b2z01 > 158.2
        + 0.003990597 * (5.55459 * 0.5 * ((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) * (0.03294892 * 0.5 * ((0.08598434 - Q.kt2_2_z_disp3) / 0.03294892) * (1 + math.erf(((0.08598434 - Q.kt2_2_z_disp3) / 0.03294892) / math.sqrt(2)))) / 0.9243343   # +0.4%  mres_sd_mass_b2z01 < 96.62 and kt2_2_z_disp3 < 0.08598
        - 0.00397654 * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((18.0 - Q.n_photon) / 1.0) * (1 + math.erf(((18.0 - Q.n_photon) / 1.0) / math.sqrt(2)))) / 3.738296   # -0.4%  dc_n > 1 and n_photon < 18
        + 0.003573558 * (0.08093679 * 0.5 * ((0.2801368 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.2801368 - Q.z_displaced5) / 0.08093679) / math.sqrt(2)))) * (0.1240004 * 0.5 * ((7.075642 - Q.pair_max_lnm2) / 0.1240004) * (1 + math.erf(((7.075642 - Q.pair_max_lnm2) / 0.1240004) / math.sqrt(2)))) / 0.1319907   # +0.4%  z_displaced5 < 0.2801 and pair_max_lnm2 < 7.076
        - 0.00346674 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) / 3.676114   # -0.3%  n_pairs_kt_above_3 < 28
        + 0.003280444 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 1.0) / 1.0) * (1 + math.erf(((Q.n_muon - 1.0) / 1.0) / math.sqrt(2)))) / 0.9200341   # +0.3%  n_s3d_above_3 < 10 and n_muon > 1
        + 0.003159957 * (1.5 * 0.5 * ((Q.n_dr_0p2_0p4 - 9.0) / 1.5) * (1 + math.erf(((Q.n_dr_0p2_0p4 - 9.0) / 1.5) / math.sqrt(2)))) / 4.79497   # +0.3%  n_dr_0p2_0p4 > 9
        - 0.003109382 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) / 297.4519   # -0.3%  sip_3d_2 < 447.1
        + 0.002938564 * (0.01578816 * 0.5 * ((0.08321691 - Q.pz_lnd2) / 0.01578816) * (1 + math.erf(((0.08321691 - Q.pz_lnd2) / 0.01578816) / math.sqrt(2)))) / 0.01877458   # +0.3%  pz_lnd2 < 0.08322
        + 0.002849302 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.5 * 0.5 * ((Q.n_lund - 7.0) / 1.5) * (1 + math.erf(((Q.n_lund - 7.0) / 1.5) / math.sqrt(2)))) / 0.001128545   # +0.3%  e3_b2 < 0.0004127 and n_lund > 7
        - 0.002819545 * (1.0 * 0.5 * ((3.0 - Q.dc_1_n_disp3) / 1.0) * (1 + math.erf(((3.0 - Q.dc_1_n_disp3) / 1.0) / math.sqrt(2)))) / 2.348369   # -0.3%  dc_1_n_disp3 < 3
        - 0.002526215 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2261637   # -0.3%  lep_iso < 0.4382
        - 0.002500565 * (0.1750793 * 0.5 * ((0.4436035 - Q.max_abs_dz) / 0.1750793) * (1 + math.erf(((0.4436035 - Q.max_abs_dz) / 0.1750793) / math.sqrt(2)))) / 0.0781359   # -0.3%  max_abs_dz < 0.4436
        + 0.002488384 * (1.0 * 0.5 * ((0.0 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.0 - Q.sv_2_n) / 1.0) / math.sqrt(2)))) / 0.04355367   # +0.2%  sv_2_n < 0
        - 0.002378848 * (1.0 * 0.5 * ((0.0 - Q.n_sd0_above_5) / 1.0) * (1 + math.erf(((0.0 - Q.n_sd0_above_5) / 1.0) / math.sqrt(2)))) / 0.02964454   # -0.2%  n_sd0_above_5 < 0
        + 0.002369226 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 5.193195   # +0.2%  n_pairs_kt_above_1 < 80
        - 0.002131543 * (0.06782196 * 0.5 * ((0.03898651 - Q.z_muon) / 0.06782196) * (1 + math.erf(((0.03898651 - Q.z_muon) / 0.06782196) / math.sqrt(2)))) * (0.7263644 * 0.5 * ((Q.mres_sd_prong_mass2 - 1.685874e-06) / 0.7263644) * (1 + math.erf(((Q.mres_sd_prong_mass2 - 1.685874e-06) / 0.7263644) / math.sqrt(2)))) / 0.2467929   # -0.2%  z_muon < 0.03899 and mres_sd_prong_mass2 > 1.686e-06
        - 0.002034537 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.04590778 * 0.5 * ((0.2054406 - Q.dr_max_012) / 0.04590778) * (1 + math.erf(((0.2054406 - Q.dr_max_012) / 0.04590778) / math.sqrt(2)))) / 2.220967e-05   # -0.2%  e3_b2 < 0.0004127 and dr_max_012 < 0.2054
        - 0.001735843 * (1.0 * 0.5 * ((3.0 - Q.dc_1_n_disp3) / 1.0) * (1 + math.erf(((3.0 - Q.dc_1_n_disp3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_pt_above_50 - 1.0) / 1.0) * (1 + math.erf(((Q.n_pt_above_50 - 1.0) / 1.0) / math.sqrt(2)))) / 3.821168   # -0.2%  dc_1_n_disp3 < 3 and n_pt_above_50 > 1
        + 0.001722467 * (1.278204 * 0.5 * ((3.101398 - Q.sjf_3_2_max3d) / 1.278204) * (1 + math.erf(((3.101398 - Q.sjf_3_2_max3d) / 1.278204) / math.sqrt(2)))) / 0.5582653   # +0.2%  sjf_3_2_max3d < 3.101
        + 0.001710634 * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) * (5.497214 * 0.5 * ((7.621251 - Q.sjf_4_1_max3d) / 5.497214) * (1 + math.erf(((7.621251 - Q.sjf_4_1_max3d) / 5.497214) / math.sqrt(2)))) / 1.714476   # +0.2%  kt2_2_n_lep < 1 and sjf_4_1_max3d < 7.621
        + 0.00168923 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) / math.sqrt(2)))) / 1.914224   # +0.2%  mres_sd_mass_b0z005 > 178.9
        - 0.001675736 * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2)))) / 1.238945   # -0.2%  dc_n > 1
        - 0.0015022 * (3.20498 * 0.5 * ((Q.mass_neutral - 34.02385) / 3.20498) * (1 + math.erf(((Q.mass_neutral - 34.02385) / 3.20498) / math.sqrt(2)))) * (0.203947 * 0.5 * ((2.345351 - Q.jd_3d_6) / 0.203947) * (1 + math.erf(((2.345351 - Q.jd_3d_6) / 0.203947) / math.sqrt(2)))) / 5.565482   # -0.2%  mass_neutral > 34.02 and jd_3d_6 < 2.345
        + 0.001501214 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) / 0.4568459   # +0.2%  sjq_2_2_nch < 4
        - 0.001474647 * (1.5 * 0.5 * ((Q.lepsj_2_n_d3 - 2.0) / 1.5) * (1 + math.erf(((Q.lepsj_2_n_d3 - 2.0) / 1.5) / math.sqrt(2)))) / 0.3817964   # -0.1%  lepsj_2_n_d3 > 2
        + 0.001294716 * (0.06782196 * 0.5 * ((0.03898651 - Q.z_muon) / 0.06782196) * (1 + math.erf(((0.03898651 - Q.z_muon) / 0.06782196) / math.sqrt(2)))) / 0.02348138   # +0.1%  z_muon < 0.03899
        + 0.001255362 * (5.119423 * 0.5 * ((Q.mass_top5 - 62.84493) / 5.119423) * (1 + math.erf(((Q.mass_top5 - 62.84493) / 5.119423) / math.sqrt(2)))) / 3.722504   # +0.1%  mass_top5 > 62.84
        + 0.001192275 * (5.55459 * 0.5 * ((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) / 11.56313   # +0.1%  mres_sd_mass_b2z01 < 96.62
        + 0.00118665 * (0.004067431 * 0.5 * ((Q.sdb_5_z - 0.01083984) / 0.004067431) * (1 + math.erf(((Q.sdb_5_z - 0.01083984) / 0.004067431) / math.sqrt(2)))) / 0.0284804   # +0.1%  sdb_5_z > 0.01084
        - 0.001162962 * (5.55459 * 0.5 * ((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) * (11.51919 * 0.5 * ((15.37467 - Q.dc_pair_mass_min) / 11.51919) * (1 + math.erf(((15.37467 - Q.dc_pair_mass_min) / 11.51919) / math.sqrt(2)))) / 107.5313   # -0.1%  mres_sd_mass_b2z01 < 96.62 and dc_pair_mass_min < 15.37
        - 0.00102414 * (1.0 * 0.5 * ((9.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((9.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2)))) / 1.346192   # -0.1%  sdb_2_n < 9
        - 0.0009704938 * (0.02907957 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.0) / 1.0) / math.sqrt(2)))) / 0.138117   # -0.1%  sjq_2_sumabs_k1 > 0.2131 and n_lund_kt_above_5 > 1
        - 0.0009257298 * (1.0 * 0.5 * ((0.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2)))) / 0.0301033   # -0.1%  dc_1_n_lep < 0
        - 0.0009257221 * (16.26099 * 0.5 * ((Q.mass_top50 - 161.1264) / 16.26099) * (1 + math.erf(((Q.mass_top50 - 161.1264) / 16.26099) / math.sqrt(2)))) / 3.008129   # -0.1%  mass_top50 > 161.1
        + 0.000775505 * (1.0 * 0.5 * ((Q.sdb_0_n - 3.0) / 1.0) * (1 + math.erf(((Q.sdb_0_n - 3.0) / 1.0) / math.sqrt(2)))) / 0.2883269   # +0.1%  sdb_0_n > 3
        - 0.0007541813 * (0.02907957 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2)))) / 0.1856394   # -0.1%  sjq_2_sumabs_k1 > 0.2131 and sip_3d_3 < 4.606
        - 0.0007417541 * (5.877113 * 0.5 * ((40.03188 - Q.mres_sd_prong_mass1) / 5.877113) * (1 + math.erf(((40.03188 - Q.mres_sd_prong_mass1) / 5.877113) / math.sqrt(2)))) / 17.02396   # -0.1%  mres_sd_prong_mass1 < 40.03
        + 0.0006989181 * (0.02550179 * 0.5 * ((0.1792389 - Q.sdb_2_z) / 0.02550179) * (1 + math.erf(((0.1792389 - Q.sdb_2_z) / 0.02550179) / math.sqrt(2)))) / 0.01876489   # +0.1%  sdb_2_z < 0.1792
        + 0.0006937123 * (0.008519048 * 0.5 * ((0.01556887 - Q.z_dr_0p05_0p1) / 0.008519048) * (1 + math.erf(((0.01556887 - Q.z_dr_0p05_0p1) / 0.008519048) / math.sqrt(2)))) / 0.002150629   # +0.1%  z_dr_0p05_0p1 < 0.01557
        - 0.0006193255 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) / 0.06043165   # -0.1%  pz_lnd0 < 0.1713
        + 0.0005232982 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.002556514 * 0.5 * ((Q.sjf_4_3_z_d3 - 0.001692006) / 0.002556514) * (1 + math.erf(((Q.sjf_4_3_z_d3 - 0.001692006) / 0.002556514) / math.sqrt(2)))) / 0.04728083   # +0.1%  n_s3d_above_3 < 10 and sjf_4_3_z_d3 > 0.001692
        - 0.0004770667 * (1.118837 * 0.5 * ((6.465318 - Q.sj2_mass2) / 1.118837) * (1 + math.erf(((6.465318 - Q.sj2_mass2) / 1.118837) / math.sqrt(2)))) / 0.5990726   # -0.0%  sj2_mass2 < 6.465
        + 0.0004737843 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (4.06561 * 0.5 * ((Q.mass_2charged - 24.4079) / 4.06561) * (1 + math.erf(((Q.mass_2charged - 24.4079) / 4.06561) / math.sqrt(2)))) / 17.41533   # +0.0%  n_s3d_above_3 < 10 and mass_2charged > 24.41
        + 0.0004719665 * (0.3133958 * 0.5 * ((0.8906353 - Q.sjf_2_2_maxsd0) / 0.3133958) * (1 + math.erf(((0.8906353 - Q.sjf_2_2_maxsd0) / 0.3133958) / math.sqrt(2)))) / 0.04638082   # +0.0%  sjf_2_2_maxsd0 < 0.8906
        + 0.0004592008 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) / 0.0002870086   # +0.0%  e3_b2 < 0.0004127
        - 0.0004110545 * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) * (81.78161 * 0.5 * ((Q.sv_1_sd0_sum - 125.2765) / 81.78161) * (1 + math.erf(((Q.sv_1_sd0_sum - 125.2765) / 81.78161) / math.sqrt(2)))) / 31.72718   # -0.0%  sjq_2_prod_k1 < 0.09803 and sv_1_sd0_sum > 125.3
        + 0.000276101 * (0.01685699 * 0.5 * ((0.0815014 - Q.M2_b05) / 0.01685699) * (1 + math.erf(((0.0815014 - Q.M2_b05) / 0.01685699) / math.sqrt(2)))) / 0.001265906   # +0.0%  M2_b05 < 0.0815
        - 0.0002610835 * (10.55249 * 0.5 * ((Q.mass_2charged - 45.73288) / 10.55249) * (1 + math.erf(((Q.mass_2charged - 45.73288) / 10.55249) / math.sqrt(2)))) / 0.789651   # -0.0%  mass_2charged > 45.73
        - 0.0002130001 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 2.0) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 2.0) / 1.0) / math.sqrt(2)))) / 0.0001305496   # -0.0%  e3_b2 < 0.0004127 and sjf_2_2_n_d3 > 2
        - 0.0002127669 * (14.45336 * 0.5 * ((Q.jd_3d_6 - 30.57088) / 14.45336) * (1 + math.erf(((Q.jd_3d_6 - 30.57088) / 14.45336) / math.sqrt(2)))) / 5.748674   # -0.0%  jd_3d_6 > 30.57
        - 0.0001528897 * (1.0 * 0.5 * ((0.0 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.0 - Q.sv_2_n) / 1.0) / math.sqrt(2)))) * (0.0225807 * 0.5 * ((0.8090312 - Q.tau43_b2) / 0.0225807) * (1 + math.erf(((0.8090312 - Q.tau43_b2) / 0.0225807) / math.sqrt(2)))) / 0.00557655   # -0.0%  sv_2_n < 0 and tau43_b2 < 0.809
        + 0.0001078339 * (0.07843207 * 0.5 * ((Q.lep_z - 0.1275041) / 0.07843207) * (1 + math.erf(((Q.lep_z - 0.1275041) / 0.07843207) / math.sqrt(2)))) * (3.620548 * 0.5 * ((4.394157 - Q.sv_1_sd0_sum) / 3.620548) * (1 + math.erf(((4.394157 - Q.sv_1_sd0_sum) / 3.620548) / math.sqrt(2)))) / 0.07475856   # +0.0%  lep_z > 0.1275 and sv_1_sd0_sum < 4.394
        - 9.072871e-05 * (0.002042627 * 0.5 * ((0.01892655 - Q.tau5) / 0.002042627) * (1 + math.erf(((0.01892655 - Q.tau5) / 0.002042627) / math.sqrt(2)))) / 0.001510389   # -0.0%  tau5 < 0.01893
        + 8.463078e-05 * (3.499256 * 0.5 * ((Q.mres_sd_mass_b2z01 - 121.6067) / 3.499256) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 121.6067) / 3.499256) / math.sqrt(2)))) / 9.205751   # +0.0%  mres_sd_mass_b2z01 > 121.6
        + 8.285893e-05 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.1703457   # +0.0%  n_pairs_kt_above_3 < 28 and kt2_2_n_lep < 0
        + 6.104445e-05 * (0.3128853 * 0.5 * ((1.113424 - Q.sjf_2_1_maxsd0) / 0.3128853) * (1 + math.erf(((1.113424 - Q.sjf_2_1_maxsd0) / 0.3128853) / math.sqrt(2)))) / 0.02185049   # +0.0%  sjf_2_1_maxsd0 < 1.113
        + 4.578952e-05 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.05798278 * 0.5 * ((3.751708 - Q.lne_4) / 0.05798278) * (1 + math.erf(((3.751708 - Q.lne_4) / 0.05798278) / math.sqrt(2)))) / 0.01690873   # +0.0%  pz_lnd0 < 0.1713 and lne_4 < 3.752
        - 3.663666e-05 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dc_mass_disp_2nd - 0.0) / 1e-06) * (1 + math.erf(((Q.dc_mass_disp_2nd - 0.0) / 1e-06) / math.sqrt(2)))) / 1.426391e-05   # -0.0%  e3_b2 < 0.0004127 and dc_mass_disp_2nd > 0
        - 2.267047e-05 * (1.0 * 0.5 * ((0.0 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.0 - Q.sv_2_n) / 1.0) / math.sqrt(2)))) * (0.03988647 * 0.5 * ((Q.d0err_49 - 0.0) / 0.03988647) * (1 + math.erf(((Q.d0err_49 - 0.0) / 0.03988647) / math.sqrt(2)))) / 0.0003107066   # -0.0%  sv_2_n < 0 and d0err_49 > 0
        + 2.232302e-05 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (0.2811803 * 0.5 * ((2.552762 - Q.jd_3d_5) / 0.2811803) * (1 + math.erf(((2.552762 - Q.jd_3d_5) / 0.2811803) / math.sqrt(2)))) / 0.4137323   # +0.0%  lep_ptrel > 43.21 and jd_3d_5 < 2.553
    )
    return z


def neuron_17(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.035346e-05
    )
    return z


def neuron_18(Q):
    # scale S = 9.684; each line: share * term / its average size
    z = 9.68352 * (0.04164339
        - 0.05715945 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) / 0.1673441   # -5.7%  lep_z < 0.2214
        + 0.05530779 * (14.89455 * 0.5 * ((145.6038 - Q.mass_top20) / 14.89455) * (1 + math.erf(((145.6038 - Q.mass_top20) / 14.89455) / math.sqrt(2)))) / 54.26043   # +5.5%  mass_top20 < 145.6
        + 0.05360489 * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) / 2.614   # +5.4%  lepsj_3_n_d3 < 3
        - 0.04474719 * (2.5 * 0.5 * ((Q.n_particles - 21.0) / 2.5) * (1 + math.erf(((Q.n_particles - 21.0) / 2.5) / math.sqrt(2)))) / 18.73609   # -4.5%  n_particles > 21
        - 0.04138226 * (540.2081 * 0.5 * ((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2)))) / 470.5515   # -4.1%  lepsj_3_maxsd0 < 587
        + 0.0375013 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) / 0.8420682   # +3.8%  lep_iso < 1.362
        + 0.03471078 * (0.005001016 * 0.5 * ((0.05045808 - Q.tau5) / 0.005001016) * (1 + math.erf(((0.05045808 - Q.tau5) / 0.005001016) / math.sqrt(2)))) / 0.02015638   # +3.5%  tau5 < 0.05046
        + 0.0341389 * (1.0 * 0.5 * ((9.0 - Q.jd_n_d3_pt1) / 1.0) * (1 + math.erf(((9.0 - Q.jd_n_d3_pt1) / 1.0) / math.sqrt(2)))) / 5.872064   # +3.4%  jd_n_d3_pt1 < 9
        - 0.0325756 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 6.361002   # -3.3%  n_s3d_above_3 < 10
        + 0.0315536 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) / 1.223619   # +3.2%  lepsj_3_maxsd0 < 2.413
        - 0.02929428 * (5.97473 * 0.5 * ((114.4658 - Q.mass_top20) / 5.97473) * (1 + math.erf(((114.4658 - Q.mass_top20) / 5.97473) / math.sqrt(2)))) / 26.73897   # -2.9%  mass_top20 < 114.5
        - 0.02751743 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.02653106 * 0.5 * ((0.1046203 - Q.z_displaced5) / 0.02653106) * (1 + math.erf(((0.1046203 - Q.z_displaced5) / 0.02653106) / math.sqrt(2)))) / 0.009861993   # -2.8%  lep_z < 0.2214 and z_displaced5 < 0.1046
        + 0.02658811 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.01348041 * 0.5 * ((0.3829918 - Q.N2) / 0.01348041) * (1 + math.erf(((0.3829918 - Q.N2) / 0.01348041) / math.sqrt(2)))) / 0.01265775   # +2.7%  lep_z < 0.2214 and N2 < 0.383
        - 0.02539146 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2)))) / 0.001827643   # -2.5%  lep_z < 0.004136
        + 0.02228056 * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.04808774 * 0.5 * ((0.09790963 - Q.lepsj_3_dr) / 0.04808774) * (1 + math.erf(((0.09790963 - Q.lepsj_3_dr) / 0.04808774) / math.sqrt(2)))) / 0.2277222   # +2.2%  lepsj_3_n_d3 < 3 and lepsj_3_dr < 0.09791
        - 0.01930444 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2)))) / 0.0537686   # -1.9%  n_lepton < 0
        - 0.01914429 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (0.4913027 * 0.5 * ((-5.150973 - Q.ktd_ln_d23) / 0.4913027) * (1 + math.erf(((-5.150973 - Q.ktd_ln_d23) / 0.4913027) / math.sqrt(2)))) / 2.295652   # -1.9%  lep_iso < 1.362 and ktd_ln_d23 < -5.151
        - 0.01903788 * (4.146938 * 0.5 * ((Q.mass - 110.2019) / 4.146938) * (1 + math.erf(((Q.mass - 110.2019) / 4.146938) / math.sqrt(2)))) / 16.71886   # -1.9%  mass > 110.2
        - 0.0176122 * (2.0 * 0.5 * ((27.0 - Q.n_for_90pct) / 2.0) * (1 + math.erf(((27.0 - Q.n_for_90pct) / 2.0) / math.sqrt(2)))) / 8.500074   # -1.8%  n_for_90pct < 27
        + 0.0165379 * (0.01670814 * 0.5 * ((0.06245248 - Q.z_displaced5) / 0.01670814) * (1 + math.erf(((0.06245248 - Q.z_displaced5) / 0.01670814) / math.sqrt(2)))) / 0.02864346   # +1.7%  z_displaced5 < 0.06245
        - 0.01652655 * (0.01141967 * 0.5 * ((0.03624058 - Q.z_displaced3) / 0.01141967) * (1 + math.erf(((0.03624058 - Q.z_displaced3) / 0.01141967) / math.sqrt(2)))) / 0.01240284   # -1.7%  z_displaced3 < 0.03624
        + 0.01636608 * Q.n_lepton / 0.5904133   # +1.6%  n_lepton
        - 0.01417623 * (2.463307 * 0.5 * ((3.710567 - Q.sjf_3_1_max3d) / 2.463307) * (1 + math.erf(((3.710567 - Q.sjf_3_1_max3d) / 2.463307) / math.sqrt(2)))) / 0.5634309   # -1.4%  sjf_3_1_max3d < 3.711
        - 0.01340043 * (0.3893833 * 0.5 * ((Q.lund3_lndelta - -2.817283) / 0.3893833) * (1 + math.erf(((Q.lund3_lndelta - -2.817283) / 0.3893833) / math.sqrt(2)))) / 1.319379   # -1.3%  lund3_lndelta > -2.817
        - 0.01321018 * (1.0 * 0.5 * ((3.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((3.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 0.9010223   # -1.3%  n_s3d_above_3 < 3
        + 0.01309485 * (0.008548367 * 0.5 * ((Q.N2_b05 - 0.4353632) / 0.008548367) * (1 + math.erf(((Q.N2_b05 - 0.4353632) / 0.008548367) / math.sqrt(2)))) / 0.03051078   # +1.3%  N2_b05 > 0.4354
        - 0.01277294 * (3.069852 * 0.5 * ((4.516968 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.516968 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) / 1.027676   # -1.3%  sjf_2_2_max3d < 4.517
        - 0.01245469 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.02343699 * 0.5 * ((Q.sdb_2_z - 0.2277628) / 0.02343699) * (1 + math.erf(((Q.sdb_2_z - 0.2277628) / 0.02343699) / math.sqrt(2)))) / 0.01773813   # -1.2%  lep_z < 0.2214 and sdb_2_z > 0.2278
        + 0.01240012 * (1.5 * 0.5 * ((6.0 - Q.n_s3d_above_10) / 1.5) * (1 + math.erf(((6.0 - Q.n_s3d_above_10) / 1.5) / math.sqrt(2)))) / 3.601475   # +1.2%  n_s3d_above_10 < 6
        - 0.01205512 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) / math.sqrt(2)))) / 7.226273   # -1.2%  mres_sd_mass_b2z01 > 130.1
        + 0.01178672 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) / 26.78553   # +1.2%  sv_1_sd0_sum < 51.56
        - 0.01032753 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.03286901 * 0.5 * ((0.4839362 - Q.ak02_dr12) / 0.03286901) * (1 + math.erf(((0.4839362 - Q.ak02_dr12) / 0.03286901) / math.sqrt(2)))) / 0.02348986   # -1.0%  lep_z < 0.2214 and ak02_dr12 < 0.4839
        - 0.0100033 * (4.883139 * 0.5 * ((Q.mass_top50 - 89.68964) / 4.883139) * (1 + math.erf(((Q.mass_top50 - 89.68964) / 4.883139) / math.sqrt(2)))) / 28.47564   # -1.0%  mass_top50 > 89.69
        + 0.009959251 * (1.5 * 0.5 * ((6.0 - Q.n_s3d_above_10) / 1.5) * (1 + math.erf(((6.0 - Q.n_s3d_above_10) / 1.5) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.2370407 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.2370407 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2)))) / 0.3852355   # +1.0%  n_s3d_above_10 < 6 and z_neutral_had < 0.237
        + 0.009955443 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) / math.sqrt(2)))) / 3.50049   # +1.0%  mres_sd_mass_b2z01 > 158.2
        + 0.00830874 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2)))) * (117.3924 * 0.5 * ((132.2798 - Q.sjf_3_1_max3d) / 117.3924) * (1 + math.erf(((132.2798 - Q.sjf_3_1_max3d) / 117.3924) / math.sqrt(2)))) / 0.1376492   # +0.8%  lep_z < 0.004136 and sjf_3_1_max3d < 132.3
        - 0.008218101 * (188.4037 * 0.5 * ((828.2826 - Q.sdb_jp_all) / 188.4037) * (1 + math.erf(((828.2826 - Q.sdb_jp_all) / 188.4037) / math.sqrt(2)))) / 412.3994   # -0.8%  sdb_jp_all < 828.3
        + 0.008140683 * (16.90428 * 0.5 * ((Q.mass - 164.4374) / 16.90428) * (1 + math.erf(((Q.mass - 164.4374) / 16.90428) / math.sqrt(2)))) / 3.240687   # +0.8%  mass > 164.4
        - 0.007668713 * (0.03429006 * 0.5 * ((0.08178299 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.08178299 - Q.lep_dr) / 0.03429006) / math.sqrt(2)))) / 0.04862806   # -0.8%  lep_dr < 0.08178
        - 0.007545962 * (4.85476 * 0.5 * ((13.03663 - Q.mass_displaced3) / 4.85476) * (1 + math.erf(((13.03663 - Q.mass_displaced3) / 4.85476) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 12.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 12.0) / 1.0) / math.sqrt(2)))) / 54.49073   # -0.8%  mass_displaced3 < 13.04 and n_charged_pt_above_1 > 12
        + 0.007233986 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) / 1.215201   # +0.7%  n_dr_0p4_up < 4
        - 0.007017109 * (0.001754707 * 0.5 * ((0.009042742 - Q.sum_z_dr2_top2) / 0.001754707) * (1 + math.erf(((0.009042742 - Q.sum_z_dr2_top2) / 0.001754707) / math.sqrt(2)))) / 0.002102826   # -0.7%  sum_z_dr2_top2 < 0.009043
        + 0.006995686 * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2)))) / 5.641924   # +0.7%  max_abs_d0 < 10.52
        + 0.006727734 * (5.55459 * 0.5 * ((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) / math.sqrt(2)))) / 20.84806   # +0.7%  mres_sd_mass_b2z01 > 96.62
        + 0.006590706 * (0.07970627 * 0.5 * ((0.5726046 - Q.sdb_2_z) / 0.07970627) * (1 + math.erf(((0.5726046 - Q.sdb_2_z) / 0.07970627) / math.sqrt(2)))) / 0.2649225   # +0.7%  sdb_2_z < 0.5726
        - 0.005824784 * (4.85476 * 0.5 * ((13.03663 - Q.mass_displaced3) / 4.85476) * (1 + math.erf(((13.03663 - Q.mass_displaced3) / 4.85476) / math.sqrt(2)))) / 8.562723   # -0.6%  mass_displaced3 < 13.04
        + 0.005548822 * (50.53748 * 0.5 * ((124.9603 - Q.jd_sum_abs_sd0_top5) / 50.53748) * (1 + math.erf(((124.9603 - Q.jd_sum_abs_sd0_top5) / 50.53748) / math.sqrt(2)))) / 46.51091   # +0.6%  jd_sum_abs_sd0_top5 < 125
        + 0.00521472 * (4.85476 * 0.5 * ((13.03663 - Q.mass_displaced3) / 4.85476) * (1 + math.erf(((13.03663 - Q.mass_displaced3) / 4.85476) / math.sqrt(2)))) * (0.06729888 * 0.5 * ((Q.jet_abs_eta - 0.3952159) / 0.06729888) * (1 + math.erf(((Q.jet_abs_eta - 0.3952159) / 0.06729888) / math.sqrt(2)))) / 3.536451   # +0.5%  mass_displaced3 < 13.04 and jet_abs_eta > 0.3952
        + 0.004683941 * (1.0 * 0.5 * ((2.0 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((2.0 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2)))) / 0.7163451   # +0.5%  n_sdz_above_5 < 2
        + 0.004627368 * (3.838712e-05 * 0.5 * ((0.000125186 - Q.e3_b2) / 3.838712e-05) * (1 + math.erf(((0.000125186 - Q.e3_b2) / 3.838712e-05) / math.sqrt(2)))) / 6.920782e-05   # +0.5%  e3_b2 < 0.0001252
        - 0.004445206 * (4.481852 * 0.5 * ((6.220692 - Q.sjf_2_1_max3d) / 4.481852) * (1 + math.erf(((6.220692 - Q.sjf_2_1_max3d) / 4.481852) / math.sqrt(2)))) / 1.242535   # -0.4%  sjf_2_1_max3d < 6.221
        - 0.004280136 * (11.17405 * 0.5 * ((Q.sj3_pair_mass_max - 128.6605) / 11.17405) * (1 + math.erf(((Q.sj3_pair_mass_max - 128.6605) / 11.17405) / math.sqrt(2)))) / 2.60253   # -0.4%  sj3_pair_mass_max > 128.7
        - 0.003583052 * (0.005299632 * 0.5 * ((Q.z_displaced5 - 0.006242101) / 0.005299632) * (1 + math.erf(((Q.z_displaced5 - 0.006242101) / 0.005299632) / math.sqrt(2)))) / 0.0885628   # -0.4%  z_displaced5 > 0.006242
        + 0.003501855 * (1.268269 * 0.5 * ((5.311506 - Q.sj2_mass2) / 1.268269) * (1 + math.erf(((5.311506 - Q.sj2_mass2) / 1.268269) / math.sqrt(2)))) / 0.3989925   # +0.4%  sj2_mass2 < 5.312
        + 0.003226811 * (0.1783594 * 0.5 * ((Q.ktd_ln_d34 - -9.180886) / 0.1783594) * (1 + math.erf(((Q.ktd_ln_d34 - -9.180886) / 0.1783594) / math.sqrt(2)))) / 0.470454   # +0.3%  ktd_ln_d34 > -9.181
        + 0.003130427 * (0.03770036 * 0.5 * ((0.3603262 - Q.z_charged_had) / 0.03770036) * (1 + math.erf(((0.3603262 - Q.z_charged_had) / 0.03770036) / math.sqrt(2)))) / 0.0228886   # +0.3%  z_charged_had < 0.3603
        + 0.002932149 * (0.1763298 * 0.5 * ((2.078395 - Q.sjf_3_1_max3d) / 0.1763298) * (1 + math.erf(((2.078395 - Q.sjf_3_1_max3d) / 0.1763298) / math.sqrt(2)))) / 0.1082649   # +0.3%  sjf_3_1_max3d < 2.078
        - 0.002745058 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.06557918 * 0.5 * ((Q.jet_charge_k05 - -0.215816) / 0.06557918) * (1 + math.erf(((Q.jet_charge_k05 - -0.215816) / 0.06557918) / math.sqrt(2)))) / 0.0512185   # -0.3%  lep_z < 0.2214 and jet_charge_k05 > -0.2158
        + 0.002718533 * (1.0 * 0.5 * ((2.0 - Q.sjq_3_3_nch) / 1.0) * (1 + math.erf(((2.0 - Q.sjq_3_3_nch) / 1.0) / math.sqrt(2)))) / 0.3638722   # +0.3%  sjq_3_3_nch < 2
        + 0.002588098 * (0.6933428 * 0.5 * ((2.382955 - Q.mass_displaced5) / 0.6933428) * (1 + math.erf(((2.382955 - Q.mass_displaced5) / 0.6933428) / math.sqrt(2)))) / 1.165287   # +0.3%  mass_displaced5 < 2.383
        - 0.002341313 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) / math.sqrt(2)))) / 1.914224   # -0.2%  mres_sd_mass_b0z005 > 178.9
        - 0.002216142 * (0.01539698 * 0.5 * ((Q.N2 - 0.3970676) / 0.01539698) * (1 + math.erf(((Q.N2 - 0.3970676) / 0.01539698) / math.sqrt(2)))) / 0.005211996   # -0.2%  N2 > 0.3971
        + 0.002145036 * Q.sjf_4_n2disp / 0.8042467   # +0.2%  sjf_4_n2disp
        - 0.001942659 * (2.113111 * 0.5 * ((3.373037 - Q.sip_3d_1) / 2.113111) * (1 + math.erf(((3.373037 - Q.sip_3d_1) / 2.113111) / math.sqrt(2)))) / 0.1387434   # -0.2%  sip_3d_1 < 3.373
        - 0.00193665 * (0.001001007 * 0.5 * ((0.0006725139 - Q.sum_z_dr2_top3) / 0.001001007) * (1 + math.erf(((0.0006725139 - Q.sum_z_dr2_top3) / 0.001001007) / math.sqrt(2)))) / 2.532287e-05   # -0.2%  sum_z_dr2_top3 < 0.0006725
        + 0.001877287 * (0.07645116 * 0.5 * ((Q.lund2_lndelta - -1.124734) / 0.07645116) * (1 + math.erf(((Q.lund2_lndelta - -1.124734) / 0.07645116) / math.sqrt(2)))) / 0.2742816   # +0.2%  lund2_lndelta > -1.125
        - 0.001863076 * (0.005001016 * 0.5 * ((0.05045808 - Q.tau5) / 0.005001016) * (1 + math.erf(((0.05045808 - Q.tau5) / 0.005001016) / math.sqrt(2)))) * (0.01243073 * 0.5 * ((Q.ak02_3_z - 0.01956504) / 0.01243073) * (1 + math.erf(((Q.ak02_3_z - 0.01956504) / 0.01243073) / math.sqrt(2)))) / 0.0003142048   # -0.2%  tau5 < 0.05046 and ak02_3_z > 0.01957
        + 0.001576999 * (0.05815218 * 0.5 * ((Q.lne_3 - 3.907638) / 0.05815218) * (1 + math.erf(((Q.lne_3 - 3.907638) / 0.05815218) / math.sqrt(2)))) / 0.232701   # +0.2%  lne_3 > 3.908
        - 0.001361436 * (1.0 * 0.5 * ((0.0 - Q.sdb_4_n) / 1.0) * (1 + math.erf(((0.0 - Q.sdb_4_n) / 1.0) / math.sqrt(2)))) / 0.03888314   # -0.1%  sdb_4_n < 0
        + 0.001324854 * (5.307922 * 0.5 * ((85.02716 - Q.mass) / 5.307922) * (1 + math.erf(((85.02716 - Q.mass) / 5.307922) / math.sqrt(2)))) / 3.842916   # +0.1%  mass < 85.03
        - 0.001270149 * (18.98174 * 0.5 * ((14.38858 - Q.lep_iso) / 18.98174) * (1 + math.erf(((14.38858 - Q.lep_iso) / 18.98174) / math.sqrt(2)))) / 9.394448   # -0.1%  lep_iso < 14.39
        - 0.001086382 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_1_n_lep - 0.0) / 1.0) * (1 + math.erf(((Q.dc_1_n_lep - 0.0) / 1.0) / math.sqrt(2)))) / 5.427907   # -0.1%  sv_1_sd0_sum < 51.56 and dc_1_n_lep > 0
        + 0.001004957 * (0.005299632 * 0.5 * ((Q.z_displaced5 - 0.006242101) / 0.005299632) * (1 + math.erf(((Q.z_displaced5 - 0.006242101) / 0.005299632) / math.sqrt(2)))) * (0.003791252 * 0.5 * ((0.01937651 - Q.dr12) / 0.003791252) * (1 + math.erf(((0.01937651 - Q.dr12) / 0.003791252) / math.sqrt(2)))) / 0.0002517065   # +0.1%  z_displaced5 > 0.006242 and dr12 < 0.01938
        + 0.0009825997 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2)))) * (0.02420044 * 0.5 * ((0.04159546 - Q.dzerr_32) / 0.02420044) * (1 + math.erf(((0.04159546 - Q.dzerr_32) / 0.02420044) / math.sqrt(2)))) / 0.001708805   # +0.1%  n_lepton < 0 and dzerr_32 < 0.0416
        + 0.0008414077 * (188.4037 * 0.5 * ((828.2826 - Q.sdb_jp_all) / 188.4037) * (1 + math.erf(((828.2826 - Q.sdb_jp_all) / 188.4037) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.ischhad_0) / 1.0) * (1 + math.erf(((0.0 - Q.ischhad_0) / 1.0) / math.sqrt(2)))) / 26.56202   # +0.1%  sdb_jp_all < 828.3 and ischhad_0 < 0
        - 0.0008317888 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (1.043533 * 0.5 * ((1.296719 - Q.ak02_3_mass) / 1.043533) * (1 + math.erf(((1.296719 - Q.ak02_3_mass) / 1.043533) / math.sqrt(2)))) / 0.6632457   # -0.1%  lep_iso < 1.362 and ak02_3_mass < 1.297
        + 0.0008232154 * (1.0 * 0.5 * ((Q.sv_2_n - 1.0) / 1.0) * (1 + math.erf(((Q.sv_2_n - 1.0) / 1.0) / math.sqrt(2)))) / 0.1797016   # +0.1%  sv_2_n > 1
        - 0.0007476567 * (0.117671 * 0.5 * ((Q.lep_dr - 0.4191372) / 0.117671) * (1 + math.erf(((Q.lep_dr - 0.4191372) / 0.117671) / math.sqrt(2)))) / 0.009357382   # -0.1%  lep_dr > 0.4191
        - 0.0006902957 * (22.5 * 0.5 * ((123.0 - Q.n_pairs_kt_above_1) / 22.5) * (1 + math.erf(((123.0 - Q.n_pairs_kt_above_1) / 22.5) / math.sqrt(2)))) / 13.74748   # -0.1%  n_pairs_kt_above_1 < 123
        + 0.0006658869 * (3.069852 * 0.5 * ((4.516968 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.516968 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) * (0.2888875 * 0.5 * ((Q.dc_3_charge - -0.6306676) / 0.2888875) * (1 + math.erf(((Q.dc_3_charge - -0.6306676) / 0.2888875) / math.sqrt(2)))) / 0.6518959   # +0.1%  sjf_2_2_max3d < 4.517 and dc_3_charge > -0.6307
        - 0.0006587682 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) * (0.003401978 * 0.5 * ((Q.sjf_3_3_z_d3 - 0.0) / 0.003401978) * (1 + math.erf(((Q.sjf_3_3_z_d3 - 0.0) / 0.003401978) / math.sqrt(2)))) / 0.1732541   # -0.1%  sv_1_sd0_sum < 51.56 and sjf_3_3_z_d3 > 0
        - 0.0006224386 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) / math.sqrt(2)))) * (0.04048364 * 0.5 * ((0.2759933 - Q.ak02_dr23) / 0.04048364) * (1 + math.erf(((0.2759933 - Q.ak02_dr23) / 0.04048364) / math.sqrt(2)))) / 0.0847213   # -0.1%  mres_sd_mass_b2z01 > 158.2 and ak02_dr23 < 0.276
        + 0.0006168695 * (4.883139 * 0.5 * ((Q.mass_top50 - 89.68964) / 4.883139) * (1 + math.erf(((Q.mass_top50 - 89.68964) / 4.883139) / math.sqrt(2)))) * (0.1268041 * 0.5 * ((0.9100211 - Q.lnpt_32) / 0.1268041) * (1 + math.erf(((0.9100211 - Q.lnpt_32) / 0.1268041) / math.sqrt(2)))) / 101.0318   # +0.1%  mass_top50 > 89.69 and lnpt_32 < 0.91
        - 0.0006151832 * (0.1505243 * 0.5 * ((Q.lund_max_lnkt - 3.388322) / 0.1505243) * (1 + math.erf(((Q.lund_max_lnkt - 3.388322) / 0.1505243) / math.sqrt(2)))) / 0.5821184   # -0.1%  lund_max_lnkt > 3.388
        - 0.0005135064 * (0.07970627 * 0.5 * ((0.5726046 - Q.sdb_2_z) / 0.07970627) * (1 + math.erf(((0.5726046 - Q.sdb_2_z) / 0.07970627) / math.sqrt(2)))) * (0.1182861 * 0.5 * ((0.3249512 - Q.eta_28) / 0.1182861) * (1 + math.erf(((0.3249512 - Q.eta_28) / 0.1182861) / math.sqrt(2)))) / 0.08783141   # -0.1%  sdb_2_z < 0.5726 and eta_28 < 0.325
        - 0.0003583579 * (0.02426666 * 0.5 * ((0.04397771 - Q.mres_sd_zg_b2z01) / 0.02426666) * (1 + math.erf(((0.04397771 - Q.mres_sd_zg_b2z01) / 0.02426666) / math.sqrt(2)))) / 0.002429741   # -0.0%  mres_sd_zg_b2z01 < 0.04398
        + 0.0003558481 * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2)))) * (0.02455054 * 0.5 * ((0.1435369 - Q.pz_lnd1) / 0.02455054) * (1 + math.erf(((0.1435369 - Q.pz_lnd1) / 0.02455054) / math.sqrt(2)))) / 0.423967   # +0.0%  max_abs_d0 < 10.52 and pz_lnd1 < 0.1435
        + 0.0002703394 * (5.55459 * 0.5 * ((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) / math.sqrt(2)))) * (0.01806359 * 0.5 * ((Q.sjf_2_2_z_d3 - 0.04865675) / 0.01806359) * (1 + math.erf(((Q.sjf_2_2_z_d3 - 0.04865675) / 0.01806359) / math.sqrt(2)))) / 0.4076655   # +0.0%  mres_sd_mass_b2z01 > 96.62 and sjf_2_2_z_d3 > 0.04866
        + 0.0002553019 * (0.03429006 * 0.5 * ((0.08178299 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.08178299 - Q.lep_dr) / 0.03429006) / math.sqrt(2)))) * (0.1131331 * 0.5 * ((Q.sjf_4_1_mass_d3 - 0.0) / 0.1131331) * (1 + math.erf(((Q.sjf_4_1_mass_d3 - 0.0) / 0.1131331) / math.sqrt(2)))) / 0.1979642   # +0.0%  lep_dr < 0.08178 and sjf_4_1_mass_d3 > 0
        + 0.0002540275 * (0.05815218 * 0.5 * ((Q.lne_3 - 3.907638) / 0.05815218) * (1 + math.erf(((Q.lne_3 - 3.907638) / 0.05815218) / math.sqrt(2)))) * (0.02199402 * 0.5 * ((Q.eta_9 - 0.07116699) / 0.02199402) * (1 + math.erf(((Q.eta_9 - 0.07116699) / 0.02199402) / math.sqrt(2)))) / 0.005415495   # +0.0%  lne_3 > 3.908 and eta_9 > 0.07117
        + 0.0001755525 * (25.87632 * 0.5 * ((29.2558 - Q.mres_sd_mass_b2z01) / 25.87632) * (1 + math.erf(((29.2558 - Q.mres_sd_mass_b2z01) / 25.87632) / math.sqrt(2)))) / 1.005338   # +0.0%  mres_sd_mass_b2z01 < 29.26
        - 0.0001624001 * (50.53748 * 0.5 * ((124.9603 - Q.jd_sum_abs_sd0_top5) / 50.53748) * (1 + math.erf(((124.9603 - Q.jd_sum_abs_sd0_top5) / 50.53748) / math.sqrt(2)))) * (0.1431331 * 0.5 * ((Q.kt2_1_sd0_3 - 1.376447) / 0.1431331) * (1 + math.erf(((Q.kt2_1_sd0_3 - 1.376447) / 0.1431331) / math.sqrt(2)))) / 7.136288   # -0.0%  jd_sum_abs_sd0_top5 < 125 and kt2_1_sd0_3 > 1.376
        + 0.0001109305 * (0.05815218 * 0.5 * ((Q.lne_3 - 3.907638) / 0.05815218) * (1 + math.erf(((Q.lne_3 - 3.907638) / 0.05815218) / math.sqrt(2)))) * (0.2385164 * 0.5 * ((Q.ak02_dr23 - 0.0) / 0.2385164) * (1 + math.erf(((Q.ak02_dr23 - 0.0) / 0.2385164) / math.sqrt(2)))) / 0.04247289   # +0.0%  lne_3 > 3.908 and ak02_dr23 > 0
        + 8.339795e-05 * (0.03770036 * 0.5 * ((0.3603262 - Q.z_charged_had) / 0.03770036) * (1 + math.erf(((0.3603262 - Q.z_charged_had) / 0.03770036) / math.sqrt(2)))) * (1.721646 * 0.5 * ((Q.mass_2photon - 5.763861) / 1.721646) * (1 + math.erf(((Q.mass_2photon - 5.763861) / 1.721646) / math.sqrt(2)))) / 0.06653234   # +0.0%  z_charged_had < 0.3603 and mass_2photon > 5.764
        + 1.194269e-05 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) / math.sqrt(2)))) * (0.09796308 * 0.5 * ((-0.258759 - Q.jet_charge) / 0.09796308) * (1 + math.erf(((-0.258759 - Q.jet_charge) / 0.09796308) / math.sqrt(2)))) / 0.09380557   # +0.0%  mres_sd_mass_b2z01 > 130.1 and jet_charge < -0.2588
        - 1.005122e-05 * (16.90428 * 0.5 * ((Q.mass - 164.4374) / 16.90428) * (1 + math.erf(((Q.mass - 164.4374) / 16.90428) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dr_94 - 0.0) / 1e-06) * (1 + math.erf(((Q.dr_94 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.03896896   # -0.0%  mass > 164.4 and dr_94 > 0
        + 9.712439e-06 * (11.17405 * 0.5 * ((Q.sj3_pair_mass_max - 128.6605) / 11.17405) * (1 + math.erf(((Q.sj3_pair_mass_max - 128.6605) / 11.17405) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.tdz_72 - 0.0) / 1e-06) * (1 + math.erf(((Q.tdz_72 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.01501301   # +0.0%  sj3_pair_mass_max > 128.7 and tdz_72 > 0
        + 7.46545e-06 * (4.176093 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 72.72359) / 4.176093) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 72.72359) / 4.176093) / math.sqrt(2)))) / 3.226281   # +0.0%  nca_sj4_pair_mass_2nd > 72.72
    )
    return z


def neuron_19(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.407828e-06
    )
    return z


def neuron_20(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.253669e-07
    )
    return z


def neuron_21(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.001683665
    )
    return z


def neuron_22(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.044852e-06
    )
    return z


def neuron_23(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.978848e-07
    )
    return z


def neuron_24(Q):
    # scale S = 10.48; each line: share * term / its average size
    z = 10.48475 * (0.06308529
        - 0.0976267 * (18.42178 * 0.5 * ((182.8592 - Q.mass) / 18.42178) * (1 + math.erf(((182.8592 - Q.mass) / 18.42178) / math.sqrt(2)))) / 69.40895   # -9.8%  mass < 182.9
        - 0.07574958 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) / 0.2702483   # -7.6%  lep_z < 0.3397
        + 0.06013831 * Q.pair_max_lnkt / 2.706794   # +6.0%  pair_max_lnkt
        - 0.05095671 * (1.0 * 0.5 * ((9.0 - Q.n_sd0_above_3) / 1.0) * (1 + math.erf(((9.0 - Q.n_sd0_above_3) / 1.0) / math.sqrt(2)))) / 5.839859   # -5.1%  n_sd0_above_3 < 9
        + 0.04990459 * (4.167088 * 0.5 * ((119.8459 - Q.mres_pruned_mass) / 4.167088) * (1 + math.erf(((119.8459 - Q.mres_pruned_mass) / 4.167088) / math.sqrt(2)))) / 32.88832   # +5.0%  mres_pruned_mass < 119.8
        - 0.04981844 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) / 0.07858689   # -5.0%  lepsj_2_dr < 0.103
        + 0.04647331 * (3.283929 * 0.5 * ((118.2746 - Q.mres_sd_mass_b2z01) / 3.283929) * (1 + math.erf(((118.2746 - Q.mres_sd_mass_b2z01) / 3.283929) / math.sqrt(2)))) / 22.6514   # +4.6%  mres_sd_mass_b2z01 < 118.3
        - 0.04479673 * (4.971962 * 0.5 * ((81.19466 - Q.mres_sd_mass_b2z01) / 4.971962) * (1 + math.erf(((81.19466 - Q.mres_sd_mass_b2z01) / 4.971962) / math.sqrt(2)))) / 6.508575   # -4.5%  mres_sd_mass_b2z01 < 81.19
        + 0.03685787 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2261637   # +3.7%  lep_iso < 0.4382
        + 0.03288401 * (2.0 * 0.5 * ((17.0 - Q.n_dr_0p2_0p4) / 2.0) * (1 + math.erf(((17.0 - Q.n_dr_0p2_0p4) / 2.0) / math.sqrt(2)))) / 7.318253   # +3.3%  n_dr_0p2_0p4 < 17
        - 0.02587596 * (5.270719 * 0.5 * ((86.14266 - Q.mres_pruned_mass) / 5.270719) * (1 + math.erf(((86.14266 - Q.mres_pruned_mass) / 5.270719) / math.sqrt(2)))) / 13.9734   # -2.6%  mres_pruned_mass < 86.14
        - 0.02478391 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) / 0.0002870086   # -2.5%  e3_b2 < 0.0004127
        - 0.02297234 * (6.875137 * 0.5 * ((18.80005 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.80005 - Q.mass_displaced3) / 6.875137) / math.sqrt(2)))) * (0.09779513 * 0.5 * ((0.6703997 - Q.sdb_2_z) / 0.09779513) * (1 + math.erf(((0.6703997 - Q.sdb_2_z) / 0.09779513) / math.sqrt(2)))) / 4.252873   # -2.3%  mass_displaced3 < 18.8 and sdb_2_z < 0.6704
        + 0.02034828 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) / 1.223619   # +2.0%  lepsj_3_maxsd0 < 2.413
        + 0.01994139 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) / 15.65331   # +2.0%  mass < 117.5
        + 0.01840318 * (6.875137 * 0.5 * ((18.80005 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.80005 - Q.mass_displaced3) / 6.875137) / math.sqrt(2)))) / 13.2945   # +1.8%  mass_displaced3 < 18.8
        + 0.01790397 * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 2.114183   # +1.8%  n_s3d_above_3 < 5
        + 0.01667727 * (10.51039 * 0.5 * ((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) * (1 + math.erf(((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) / math.sqrt(2)))) / 4.332549   # +1.7%  mres_sd_mass_b2z01 < 69.12
        - 0.0162929 * (0.0257051 * 0.5 * ((0.1468781 - Q.z_dr_0p2_0p4) / 0.0257051) * (1 + math.erf(((0.1468781 - Q.z_dr_0p2_0p4) / 0.0257051) / math.sqrt(2)))) / 0.04873494   # -1.6%  z_dr_0p2_0p4 < 0.1469
        - 0.01339124 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) / 36.20948   # -1.3%  lep_ptrel < 43.21
        + 0.01237062 * (0.1026619 * 0.5 * ((0.9189173 - Q.N3_b05) / 0.1026619) * (1 + math.erf(((0.9189173 - Q.N3_b05) / 0.1026619) / math.sqrt(2)))) / 0.2589195   # +1.2%  N3_b05 < 0.9189
        - 0.01167136 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.0) / 1.0) / math.sqrt(2)))) / 0.2945968   # -1.2%  lep_z < 0.3397 and n_lund_kt_above_5 > 1
        + 0.01015512 * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.0) / 1.0) / math.sqrt(2)))) / 1.039761   # +1.0%  n_lund_kt_above_5 > 1
        - 0.009999641 * (0.01304025 * 0.5 * ((0.01245833 - Q.lep_z) / 0.01304025) * (1 + math.erf(((0.01245833 - Q.lep_z) / 0.01304025) / math.sqrt(2)))) / 0.006374399   # -1.0%  lep_z < 0.01246
        + 0.009986367 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) * (0.01776485 * 0.5 * ((0.2714017 - Q.N2_b2) / 0.01776485) * (1 + math.erf(((0.2714017 - Q.N2_b2) / 0.01776485) / math.sqrt(2)))) / 1.419035   # +1.0%  mass < 117.5 and N2_b2 < 0.2714
        + 0.009321226 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (5.110145 * 0.5 * ((131.3917 - Q.mass) / 5.110145) * (1 + math.erf(((131.3917 - Q.mass) / 5.110145) / math.sqrt(2)))) / 6.513183   # +0.9%  lep_z < 0.3397 and mass < 131.4
        + 0.008930581 * (2.113111 * 0.5 * ((3.373037 - Q.sip_3d_1) / 2.113111) * (1 + math.erf(((3.373037 - Q.sip_3d_1) / 2.113111) / math.sqrt(2)))) / 0.1387434   # +0.9%  sip_3d_1 < 3.373
        + 0.008854518 * (14.176 * 0.5 * ((Q.mres_sd_prong_mass1 - 69.04524) / 14.176) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 69.04524) / 14.176) / math.sqrt(2)))) / 2.343595   # +0.9%  mres_sd_prong_mass1 > 69.05
        - 0.008452835 * (0.7154015 * 0.5 * ((Q.mass_displaced3 - 2.409613) / 0.7154015) * (1 + math.erf(((Q.mass_displaced3 - 2.409613) / 0.7154015) / math.sqrt(2)))) / 6.576652   # -0.8%  mass_displaced3 > 2.41
        + 0.008434522 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.08912976 * 0.5 * ((Q.lne_0 - 4.817758) / 0.08912976) * (1 + math.erf(((Q.lne_0 - 4.817758) / 0.08912976) / math.sqrt(2)))) / 15.3503   # +0.8%  lep_ptrel < 43.21 and lne_0 > 4.818
        + 0.007261709 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_neutral_had - 1.0) / 1.0) * (1 + math.erf(((Q.n_neutral_had - 1.0) / 1.0) / math.sqrt(2)))) / 0.0008320061   # +0.7%  e3_b2 < 0.0004127 and n_neutral_had > 1
        + 0.007003504 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) * (1.15674 * 0.5 * ((Q.lne_17 - -0.1601665) / 1.15674) * (1 + math.erf(((Q.lne_17 - -0.1601665) / 1.15674) / math.sqrt(2)))) / 26.14959   # +0.7%  mass < 117.5 and lne_17 > -0.1602
        - 0.006886909 * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (395.2298 * 0.5 * ((509.6197 - Q.sv_1_sd0_sum) / 395.2298) * (1 + math.erf(((509.6197 - Q.sv_1_sd0_sum) / 395.2298) / math.sqrt(2)))) / 832.2983   # -0.7%  n_s3d_above_3 < 5 and sv_1_sd0_sum < 509.6
        - 0.006588176 * (0.071994 * 0.5 * ((0.4545826 - Q.N3_b2) / 0.071994) * (1 + math.erf(((0.4545826 - Q.N3_b2) / 0.071994) / math.sqrt(2)))) / 0.1886484   # -0.7%  N3_b2 < 0.4546
        - 0.006471367 * (3.02693 * 0.5 * ((78.76797 - Q.sj4_pair_mass_max) / 3.02693) * (1 + math.erf(((78.76797 - Q.sj4_pair_mass_max) / 3.02693) / math.sqrt(2)))) / 9.532447   # -0.6%  sj4_pair_mass_max < 78.77
        + 0.006004071 * (0.560057 * 0.5 * ((1.777787 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.777787 - Q.mass_displaced5) / 0.560057) / math.sqrt(2)))) / 0.8185445   # +0.6%  mass_displaced5 < 1.778
        - 0.005912555 * (15.148 * 0.5 * ((Q.mres_sd_prong_mass1 - 84.19324) / 15.148) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 84.19324) / 15.148) / math.sqrt(2)))) / 1.227187   # -0.6%  mres_sd_prong_mass1 > 84.19
        + 0.005331738 * (3.24083 * 0.5 * ((6.161303 - Q.dc_2_jp) / 3.24083) * (1 + math.erf(((6.161303 - Q.dc_2_jp) / 3.24083) / math.sqrt(2)))) / 3.139178   # +0.5%  dc_2_jp < 6.161
        + 0.005067104 * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2)))) / 0.01976846   # +0.5%  sjq_3_2_k1 < -0.6015
        - 0.005051694 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 1.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 1.0) / 1.0) / math.sqrt(2)))) / 1.915687   # -0.5%  n_s3d_above_10 > 1
        - 0.004952171 * (1.0 * 0.5 * ((9.0 - Q.n_sd0_above_3) / 1.0) * (1 + math.erf(((9.0 - Q.n_sd0_above_3) / 1.0) / math.sqrt(2)))) * (0.05422308 * 0.5 * ((Q.z_neutral - 0.1384639) / 0.05422308) * (1 + math.erf(((Q.z_neutral - 0.1384639) / 0.05422308) / math.sqrt(2)))) / 1.593141   # -0.5%  n_sd0_above_3 < 9 and z_neutral > 0.1385
        + 0.004897065 * (8.537109 * 0.5 * ((17.60938 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((17.60938 - Q.max_abs_d0) / 8.537109) / math.sqrt(2)))) / 10.41349   # +0.5%  max_abs_d0 < 17.61
        + 0.004363822 * (1.5 * 0.5 * ((7.0 - Q.n_lund) / 1.5) * (1 + math.erf(((7.0 - Q.n_lund) / 1.5) / math.sqrt(2)))) / 0.2969234   # +0.4%  n_lund < 7
        - 0.004256825 * (0.02561531 * 0.5 * ((0.1061578 - Q.z_displaced3) / 0.02561531) * (1 + math.erf(((0.1061578 - Q.z_displaced3) / 0.02561531) / math.sqrt(2)))) * (2.8693e-06 * 0.5 * ((2.139972e-05 - Q.ecf_g42) / 2.8693e-06) * (1 + math.erf(((2.139972e-05 - Q.ecf_g42) / 2.8693e-06) / math.sqrt(2)))) / 3.745832e-07   # -0.4%  z_displaced3 < 0.1062 and ecf_g42 < 2.14e-05
        - 0.004150412 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (0.02224606 * 0.5 * ((0.2644738 - Q.sj2_dr) / 0.02224606) * (1 + math.erf(((0.2644738 - Q.sj2_dr) / 0.02224606) / math.sqrt(2)))) / 0.003407478   # -0.4%  lep_iso < 0.4382 and sj2_dr < 0.2645
        + 0.003706964 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) * (0.01972221 * 0.5 * ((0.2848657 - Q.sj2_dr) / 0.01972221) * (1 + math.erf(((0.2848657 - Q.sj2_dr) / 0.01972221) / math.sqrt(2)))) / 0.001681303   # +0.4%  lepsj_2_dr < 0.103 and sj2_dr < 0.2849
        + 0.003659965 * (0.001965812 * 0.5 * ((Q.M3 - 0.03259227) / 0.001965812) * (1 + math.erf(((Q.M3 - 0.03259227) / 0.001965812) / math.sqrt(2)))) / 0.008118159   # +0.4%  M3 > 0.03259
        - 0.00353936 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2)))) / 4.97365   # -0.4%  mass < 90.09
        + 0.003461873 * (0.2398529 * 0.5 * ((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) * (1 + math.erf(((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) / math.sqrt(2)))) / 0.01973624   # +0.3%  sjq_3_2_k1 > 0.6149
        + 0.003409124 * (43.0 * 0.5 * ((366.0 - Q.n_pairs_kt_above_1) / 43.0) * (1 + math.erf(((366.0 - Q.n_pairs_kt_above_1) / 43.0) / math.sqrt(2)))) / 128.9288   # +0.3%  n_pairs_kt_above_1 < 366
        + 0.003408837 * (0.06488792 * 0.5 * ((Q.dc_split2_dr - 0.2806799) / 0.06488792) * (1 + math.erf(((Q.dc_split2_dr - 0.2806799) / 0.06488792) / math.sqrt(2)))) / 0.01066673   # +0.3%  dc_split2_dr > 0.2807
        - 0.00302049 * (0.4432951 * 0.5 * ((2.802481 - Q.sip_3d_1) / 0.4432951) * (1 + math.erf(((2.802481 - Q.sip_3d_1) / 0.4432951) / math.sqrt(2)))) / 0.07369928   # -0.3%  sip_3d_1 < 2.802
        - 0.002897656 * (0.01063134 * 0.5 * ((0.4075226 - Q.N2_b05) / 0.01063134) * (1 + math.erf(((0.4075226 - Q.N2_b05) / 0.01063134) / math.sqrt(2)))) / 0.01725708   # -0.3%  N2_b05 < 0.4075
        + 0.002682577 * (0.02561531 * 0.5 * ((0.1061578 - Q.z_displaced3) / 0.02561531) * (1 + math.erf(((0.1061578 - Q.z_displaced3) / 0.02561531) / math.sqrt(2)))) / 0.05174148   # +0.3%  z_displaced3 < 0.1062
        - 0.002661473 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.sv_1_n) / 1.0) * (1 + math.erf(((3.0 - Q.sv_1_n) / 1.0) / math.sqrt(2)))) / 11.74502   # -0.3%  mass < 90.09 and sv_1_n < 3
        - 0.002556322 * (0.002611265 * 0.5 * ((0.01015545 - Q.M2_b2) / 0.002611265) * (1 + math.erf(((0.01015545 - Q.M2_b2) / 0.002611265) / math.sqrt(2)))) / 0.0003252506   # -0.3%  M2_b2 < 0.01016
        + 0.002496262 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.05415789 * 0.5 * ((-2.05292 - Q.lnerel_1) / 0.05415789) * (1 + math.erf(((-2.05292 - Q.lnerel_1) / 0.05415789) / math.sqrt(2)))) / 6.037548e-05   # +0.2%  e3_b2 < 0.0004127 and lnerel_1 < -2.053
        - 0.002302462 * (0.01417841 * 0.5 * ((0.1049877 - Q.C2_b2) / 0.01417841) * (1 + math.erf(((0.1049877 - Q.C2_b2) / 0.01417841) / math.sqrt(2)))) / 0.04687556   # -0.2%  C2_b2 < 0.105
        + 0.002288821 * (0.2508028 * 0.5 * ((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2)))) / 0.01780014   # +0.2%  sjq_3_3_k1 < -0.655
        - 0.002242705 * (6.170706 * 0.5 * ((74.51927 - Q.mass_top30) / 6.170706) * (1 + math.erf(((74.51927 - Q.mass_top30) / 6.170706) / math.sqrt(2)))) / 2.72308   # -0.2%  mass_top30 < 74.52
        - 0.002194616 * (0.07842787 * 0.5 * ((Q.dc_split2_dr - 0.3591078) / 0.07842787) * (1 + math.erf(((Q.dc_split2_dr - 0.3591078) / 0.07842787) / math.sqrt(2)))) / 0.005063019   # -0.2%  dc_split2_dr > 0.3591
        - 0.002096201 * (0.01129976 * 0.5 * ((Q.N2 - 0.3357559) / 0.01129976) * (1 + math.erf(((Q.N2 - 0.3357559) / 0.01129976) / math.sqrt(2)))) / 0.02163059   # -0.2%  N2 > 0.3358
        + 0.00184952 * (12.49365 * 0.5 * ((128.0079 - Q.sj4_pair_mass_max) / 12.49365) * (1 + math.erf(((128.0079 - Q.sj4_pair_mass_max) / 12.49365) / math.sqrt(2)))) / 47.96266   # +0.2%  sj4_pair_mass_max < 128
        - 0.001775555 * (0.01435892 * 0.5 * ((Q.pz_lnd3 - 0.1187422) / 0.01435892) * (1 + math.erf(((Q.pz_lnd3 - 0.1187422) / 0.01435892) / math.sqrt(2)))) * (0.04200199 * 0.5 * ((Q.sj3_dr23 - 0.1712041) / 0.04200199) * (1 + math.erf(((Q.sj3_dr23 - 0.1712041) / 0.04200199) / math.sqrt(2)))) / 0.008119376   # -0.2%  pz_lnd3 > 0.1187 and sj3_dr23 > 0.1712
        + 0.001775302 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.05582565 * 0.5 * ((0.456416 - Q.tau32) / 0.05582565) * (1 + math.erf(((0.456416 - Q.tau32) / 0.05582565) / math.sqrt(2)))) / 0.003667966   # +0.2%  lep_z < 0.3397 and tau32 < 0.4564
        - 0.001664902 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.0947145 * 0.5 * ((Q.dc_4_z - 0.0) / 0.0947145) * (1 + math.erf(((Q.dc_4_z - 0.0) / 0.0947145) / math.sqrt(2)))) / 0.002189242   # -0.2%  lep_z < 0.3397 and dc_4_z > 0
        - 0.00142521 * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2)))) * (0.02222194 * 0.5 * ((Q.sj3_z2 - 0.1801324) / 0.02222194) * (1 + math.erf(((Q.sj3_z2 - 0.1801324) / 0.02222194) / math.sqrt(2)))) / 0.00223002   # -0.1%  sjq_3_2_k1 < -0.6015 and sj3_z2 > 0.1801
        - 0.001306241 * (6.785484 * 0.5 * ((Q.mass_top5 - 68.52153) / 6.785484) * (1 + math.erf(((Q.mass_top5 - 68.52153) / 6.785484) / math.sqrt(2)))) / 2.773241   # -0.1%  mass_top5 > 68.52
        + 0.001296648 * (0.0947145 * 0.5 * ((Q.dc_4_z - 0.0) / 0.0947145) * (1 + math.erf(((Q.dc_4_z - 0.0) / 0.0947145) / math.sqrt(2)))) / 0.007314922   # +0.1%  dc_4_z > 0
        + 0.001295775 * (0.02561531 * 0.5 * ((0.1061578 - Q.z_displaced3) / 0.02561531) * (1 + math.erf(((0.1061578 - Q.z_displaced3) / 0.02561531) / math.sqrt(2)))) * (0.9519596 * 0.5 * ((Q.mass_2photon - 3.013) / 0.9519596) * (1 + math.erf(((Q.mass_2photon - 3.013) / 0.9519596) / math.sqrt(2)))) / 0.1903982   # +0.1%  z_displaced3 < 0.1062 and mass_2photon > 3.013
        + 0.001195984 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2)))) / 0.005033892   # +0.1%  lep_iso < 0.4382 and ak02_3_n_lep > 0
        + 0.001071854 * (1.0 * 0.5 * ((Q.n_sd0_above_10 - 4.0) / 1.0) * (1 + math.erf(((Q.n_sd0_above_10 - 4.0) / 1.0) / math.sqrt(2)))) / 0.4504918   # +0.1%  n_sd0_above_10 > 4
        + 0.001059097 * (0.004691441 * 0.5 * ((0.0 - Q.z_muon) / 0.004691441) * (1 + math.erf(((0.0 - Q.z_muon) / 0.004691441) / math.sqrt(2)))) / 1.382939e-05   # +0.1%  z_muon < 0
        - 0.0009796649 * (15.148 * 0.5 * ((Q.mres_sd_prong_mass1 - 84.19324) / 15.148) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 84.19324) / 15.148) / math.sqrt(2)))) * (0.0947145 * 0.5 * ((0.0947145 - Q.dc_4_z) / 0.0947145) * (1 + math.erf(((0.0947145 - Q.dc_4_z) / 0.0947145) / math.sqrt(2)))) / 0.06345791   # -0.1%  mres_sd_prong_mass1 > 84.19 and dc_4_z < 0.09471
        + 0.0009524759 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) * (0.02849867 * 0.5 * ((Q.z_dr_0p1_0p2 - 0.07090906) / 0.02849867) * (1 + math.erf(((Q.z_dr_0p1_0p2 - 0.07090906) / 0.02849867) / math.sqrt(2)))) / 2.72544   # +0.1%  mass < 117.5 and z_dr_0p1_0p2 > 0.07091
        + 0.000903153 * (3.143401 * 0.5 * ((Q.mass_top15 - 74.10377) / 3.143401) * (1 + math.erf(((Q.mass_top15 - 74.10377) / 3.143401) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2)))) / 1.373799   # +0.1%  mass_top15 > 74.1 and ak02_3_n_lep > 0
        + 0.0008842155 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (1.364915e-06 * 0.5 * ((Q.ecf_g43 - 6.840827e-06) / 1.364915e-06) * (1 + math.erf(((Q.ecf_g43 - 6.840827e-06) / 1.364915e-06) / math.sqrt(2)))) / 0.000126541   # +0.1%  lep_ptrel < 43.21 and ecf_g43 > 6.841e-06
        - 0.0008712888 * (2.624262 * 0.5 * ((4.636903 - Q.sip_3d_2) / 2.624262) * (1 + math.erf(((4.636903 - Q.sip_3d_2) / 2.624262) / math.sqrt(2)))) / 0.6633386   # -0.1%  sip_3d_2 < 4.637
        + 0.0008164451 * (3.143401 * 0.5 * ((Q.mass_top15 - 74.10377) / 3.143401) * (1 + math.erf(((Q.mass_top15 - 74.10377) / 3.143401) / math.sqrt(2)))) / 15.7856   # +0.1%  mass_top15 > 74.1
        + 0.0007216709 * (4.971962 * 0.5 * ((81.19466 - Q.mres_sd_mass_b2z01) / 4.971962) * (1 + math.erf(((81.19466 - Q.mres_sd_mass_b2z01) / 4.971962) / math.sqrt(2)))) * (0.02709999 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.185133) / 0.02709999) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.185133) / 0.02709999) / math.sqrt(2)))) / 1.549239   # +0.1%  mres_sd_mass_b2z01 < 81.19 and sjq_2_sumabs_k1 > 0.1851
        - 0.0005686524 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.1300849 * 0.5 * ((Q.dc_tag_2nd - 0.0) / 0.1300849) * (1 + math.erf(((Q.dc_tag_2nd - 0.0) / 0.1300849) / math.sqrt(2)))) / 0.2196251   # -0.1%  lep_z < 0.3397 and dc_tag_2nd > 0
        + 0.0005510414 * (0.1340495 * 0.5 * ((Q.dc_tag_2nd - 0.1300849) / 0.1340495) * (1 + math.erf(((Q.dc_tag_2nd - 0.1300849) / 0.1340495) / math.sqrt(2)))) / 0.6818312   # +0.1%  dc_tag_2nd > 0.1301
        + 0.0005171961 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 1.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 1.0) / 1.0) / math.sqrt(2)))) * (0.3893833 * 0.5 * ((Q.lund3_lndelta - -2.817283) / 0.3893833) * (1 + math.erf(((Q.lund3_lndelta - -2.817283) / 0.3893833) / math.sqrt(2)))) / 2.659021   # +0.1%  n_s3d_above_10 > 1 and lund3_lndelta > -2.817
        + 0.0004024386 * (0.002798005 * 0.5 * ((-0.003234757 - Q.sjq_2_prod_k1) / 0.002798005) * (1 + math.erf(((-0.003234757 - Q.sjq_2_prod_k1) / 0.002798005) / math.sqrt(2)))) / 0.03094846   # +0.0%  sjq_2_prod_k1 < -0.003235
        - 0.0003965934 * (1.0 * 0.5 * ((9.0 - Q.n_sd0_above_3) / 1.0) * (1 + math.erf(((9.0 - Q.n_sd0_above_3) / 1.0) / math.sqrt(2)))) * (0.003395811 * 0.5 * ((Q.sv_2_z - 0.00726873) / 0.003395811) * (1 + math.erf(((Q.sv_2_z - 0.00726873) / 0.003395811) / math.sqrt(2)))) / 0.008415699   # -0.0%  n_sd0_above_3 < 9 and sv_2_z > 0.007269
        - 0.0003744465 * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.0) / 1.0) / math.sqrt(2)))) * (0.00991452 * 0.5 * ((-0.01644749 - Q.tdz_0) / 0.00991452) * (1 + math.erf(((-0.01644749 - Q.tdz_0) / 0.00991452) / math.sqrt(2)))) / 0.01342035   # -0.0%  n_lund_kt_above_5 > 1 and tdz_0 < -0.01645
        - 0.0003646779 * (0.01282622 * 0.5 * ((0.1010123 - Q.z_neutral_had) / 0.01282622) * (1 + math.erf(((0.1010123 - Q.z_neutral_had) / 0.01282622) / math.sqrt(2)))) / 0.020496   # -0.0%  z_neutral_had < 0.101
        - 0.000345 * (1.0 * 0.5 * ((0.0 - Q.sdb_4_n) / 1.0) * (1 + math.erf(((0.0 - Q.sdb_4_n) / 1.0) / math.sqrt(2)))) / 0.03888314   # -0.0%  sdb_4_n < 0
        - 0.0003432725 * (0.7154015 * 0.5 * ((Q.mass_displaced3 - 2.409613) / 0.7154015) * (1 + math.erf(((Q.mass_displaced3 - 2.409613) / 0.7154015) / math.sqrt(2)))) * (0.0361397 * 0.5 * ((Q.z_neutral_had - 0.2675458) / 0.0361397) * (1 + math.erf(((Q.z_neutral_had - 0.2675458) / 0.0361397) / math.sqrt(2)))) / 0.06059834   # -0.0%  mass_displaced3 > 2.41 and z_neutral_had > 0.2675
        + 0.0003153133 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.1852805 * 0.5 * ((Q.sjq_2_2_k1 - 0.3796859) / 0.1852805) * (1 + math.erf(((Q.sjq_2_2_k1 - 0.3796859) / 0.1852805) / math.sqrt(2)))) / 0.006522582   # +0.0%  lep_z < 0.3397 and sjq_2_2_k1 > 0.3797
        - 0.000268832 * (6.785484 * 0.5 * ((Q.mass_top5 - 68.52153) / 6.785484) * (1 + math.erf(((Q.mass_top5 - 68.52153) / 6.785484) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2)))) / 0.2443172   # -0.0%  mass_top5 > 68.52 and ak02_3_n_lep > 0
        - 0.0002607569 * (0.003868055 * 0.5 * ((Q.psi_0p3 - 0.9925964) / 0.003868055) * (1 + math.erf(((Q.psi_0p3 - 0.9925964) / 0.003868055) / math.sqrt(2)))) / 0.0007066244   # -0.0%  psi_0p3 > 0.9926
        + 0.000195545 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dc_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.dc_3_n_lep - 0.0) / 1e-06) / math.sqrt(2)))) / 0.02719422   # +0.0%  lepsj_3_maxsd0 < 2.413 and dc_3_n_lep > 0
        - 0.0001665662 * (0.01168879 * 0.5 * ((Q.tdz_2 - -0.02221619) / 0.01168879) * (1 + math.erf(((Q.tdz_2 - -0.02221619) / 0.01168879) / math.sqrt(2)))) / 0.04741814   # -0.0%  tdz_2 > -0.02222
        + 0.0001296021 * (0.01435892 * 0.5 * ((Q.pz_lnd3 - 0.1187422) / 0.01435892) * (1 + math.erf(((Q.pz_lnd3 - 0.1187422) / 0.01435892) / math.sqrt(2)))) / 0.02297606   # +0.0%  pz_lnd3 > 0.1187
        + 5.284514e-05 * (0.2508028 * 0.5 * ((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2)))) * (0.04756694 * 0.5 * ((-0.04769897 - Q.eta_37) / 0.04756694) * (1 + math.erf(((-0.04769897 - Q.eta_37) / 0.04756694) / math.sqrt(2)))) / 0.0005099405   # +0.0%  sjq_3_3_k1 < -0.655 and eta_37 < -0.0477
        - 4.866392e-05 * (8.537109 * 0.5 * ((17.60938 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((17.60938 - Q.max_abs_d0) / 8.537109) / math.sqrt(2)))) * (10.60212 * 0.5 * ((Q.sjf_3_2_mass_d3 - 14.85065) / 10.60212) * (1 + math.erf(((Q.sjf_3_2_mass_d3 - 14.85065) / 10.60212) / math.sqrt(2)))) / 30.33824   # -0.0%  max_abs_d0 < 17.61 and sjf_3_2_mass_d3 > 14.85
        + 3.505861e-05 * (3.283929 * 0.5 * ((118.2746 - Q.mres_sd_mass_b2z01) / 3.283929) * (1 + math.erf(((118.2746 - Q.mres_sd_mass_b2z01) / 3.283929) / math.sqrt(2)))) * (9.32425 * 0.5 * ((Q.lnpt_43 - -0.08134564) / 9.32425) * (1 + math.erf(((Q.lnpt_43 - -0.08134564) / 9.32425) / math.sqrt(2)))) / 9.876325   # +0.0%  mres_sd_mass_b2z01 < 118.3 and lnpt_43 > -0.08135
        + 1.552927e-05 * (0.06488792 * 0.5 * ((Q.dc_split2_dr - 0.2806799) / 0.06488792) * (1 + math.erf(((Q.dc_split2_dr - 0.2806799) / 0.06488792) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dr_86 - 0.0) / 1e-06) * (1 + math.erf(((Q.dr_86 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.0001187642   # +0.0%  dc_split2_dr > 0.2807 and dr_86 > 0
        - 4.64818e-06 * (0.7154015 * 0.5 * ((Q.mass_displaced3 - 2.409613) / 0.7154015) * (1 + math.erf(((Q.mass_displaced3 - 2.409613) / 0.7154015) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dzerr_75 - 0.0) / 1e-06) * (1 + math.erf(((Q.dzerr_75 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.008395698   # -0.0%  mass_displaced3 > 2.41 and dzerr_75 > 0
    )
    return z


def neuron_25(Q):
    # scale S = 3.738; each line: share * term / its average size
    z = 3.738343 * (0.02441123
        - 0.1441119 * (7.657485 * 0.5 * ((93.75785 - Q.mres_sd_mass_b0z02) / 7.657485) * (1 + math.erf(((93.75785 - Q.mres_sd_mass_b0z02) / 7.657485) / math.sqrt(2)))) / 33.00258   # -14.4%  mres_sd_mass_b0z02 < 93.76
        + 0.0719327 * (6.347532 * 0.5 * ((80.36442 - Q.mres_sd_mass_b0z02) / 6.347532) * (1 + math.erf(((80.36442 - Q.mres_sd_mass_b0z02) / 6.347532) / math.sqrt(2)))) / 25.61713   # +7.2%  mres_sd_mass_b0z02 < 80.36
        + 0.06063357 * (7.801503 * 0.5 * ((101.9185 - Q.mres_sd_mass_b0z02) / 7.801503) * (1 + math.erf(((101.9185 - Q.mres_sd_mass_b0z02) / 7.801503) / math.sqrt(2)))) / 38.1565   # +6.1%  mres_sd_mass_b0z02 < 101.9
        - 0.04762766 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) / 0.2702483   # -4.8%  lep_z < 0.3397
        + 0.04629545 * (0.05999923 * 0.5 * ((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) * (1 + math.erf(((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) / math.sqrt(2)))) * (0.01992156 * 0.5 * ((0.04178742 - Q.lepsj_2_dr) / 0.01992156) * (1 + math.erf(((0.04178742 - Q.lepsj_2_dr) / 0.01992156) / math.sqrt(2)))) / 0.004254508   # +4.6%  sjf_3_1_z_d3 < 0.1941 and lepsj_2_dr < 0.04179
        - 0.04460986 * (0.05999923 * 0.5 * ((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) * (1 + math.erf(((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) / math.sqrt(2)))) / 0.1407024   # -4.5%  sjf_3_1_z_d3 < 0.1941
        + 0.03483074 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2)))) / 2.484045   # +3.5%  lep_z < 0.3397 and n_dr_0p4_up < 15
        + 0.03290548 * Q.pair_max_lnkt / 2.706794   # +3.3%  pair_max_lnkt
        - 0.02506119 * (0.04122765 * 0.5 * ((Q.N2_b05 - 0.302405) / 0.04122765) * (1 + math.erf(((Q.N2_b05 - 0.302405) / 0.04122765) / math.sqrt(2)))) * (1.0 * 0.5 * ((2.0 - Q.n_electron) / 1.0) * (1 + math.erf(((2.0 - Q.n_electron) / 1.0) / math.sqrt(2)))) / 0.2231253   # -2.5%  N2_b05 > 0.3024 and n_electron < 2
        - 0.02307493 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) * (1.0 * 0.5 * ((5.0 - Q.sjf_4_1_n_d3) / 1.0) * (1 + math.erf(((5.0 - Q.sjf_4_1_n_d3) / 1.0) / math.sqrt(2)))) / 0.05083318   # -2.3%  M2_b2 < 0.04179 and sjf_4_1_n_d3 < 5
        + 0.02140039 * (29.74772 * 0.5 * ((163.95 - Q.mres_sd_mass_b0z02) / 29.74772) * (1 + math.erf(((163.95 - Q.mres_sd_mass_b0z02) / 29.74772) / math.sqrt(2)))) / 90.04006   # +2.1%  mres_sd_mass_b0z02 < 164
        - 0.02057546 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2261637   # -2.1%  lep_iso < 0.4382
        - 0.01995127 * (6.531597 * 0.5 * ((79.47361 - Q.mass) / 6.531597) * (1 + math.erf(((79.47361 - Q.mass) / 6.531597) / math.sqrt(2)))) / 2.901611   # -2.0%  mass < 79.47
        - 0.01969689 * (1.079379e-06 * 0.5 * ((2.097213e-06 - Q.e4) / 1.079379e-06) * (1 + math.erf(((2.097213e-06 - Q.e4) / 1.079379e-06) / math.sqrt(2)))) / 1.408465e-06   # -2.0%  e4 < 2.097e-06
        + 0.01856192 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.01374893 * 0.5 * ((0.2241366 - Q.pz_lnd2) / 0.01374893) * (1 + math.erf(((0.2241366 - Q.pz_lnd2) / 0.01374893) / math.sqrt(2)))) / 0.02898826   # +1.9%  lep_z < 0.3397 and pz_lnd2 < 0.2241
        + 0.01844525 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.2765194 * 0.5 * ((2.605573 - Q.jd_3d_4) / 0.2765194) * (1 + math.erf(((2.605573 - Q.jd_3d_4) / 0.2765194) / math.sqrt(2)))) / 0.09640299   # +1.8%  lep_z < 0.3397 and jd_3d_4 < 2.606
        + 0.0167036 * (12.96659 * 0.5 * ((63.78353 - Q.mres_sd_mass_b0z02) / 12.96659) * (1 + math.erf(((63.78353 - Q.mres_sd_mass_b0z02) / 12.96659) / math.sqrt(2)))) / 18.41368   # +1.7%  mres_sd_mass_b0z02 < 63.78
        + 0.01509095 * Q.n_dr_0p2_0p4 / 11.50196   # +1.5%  n_dr_0p2_0p4
        + 0.01496122 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2)))) / 4.97365   # +1.5%  mass < 90.09
        + 0.01474682 * (1.0 * 0.5 * ((1.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((1.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) / 0.6926634   # +1.5%  lepsj_3_n_d3 < 1
        - 0.0146896 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.03966789 * 0.5 * ((0.1333792 - Q.ak02_3_z) / 0.03966789) * (1 + math.erf(((0.1333792 - Q.ak02_3_z) / 0.03966789) / math.sqrt(2)))) / 0.02612471   # -1.5%  lep_z < 0.3397 and ak02_3_z < 0.1334
        - 0.0139972 * (16.47315 * 0.5 * ((47.97531 - Q.mres_sd_mass_b0z02) / 16.47315) * (1 + math.erf(((47.97531 - Q.mres_sd_mass_b0z02) / 16.47315) / math.sqrt(2)))) / 12.3957   # -1.4%  mres_sd_mass_b0z02 < 47.98
        - 0.01359041 * (1.0 * 0.5 * ((2.0 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((2.0 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2)))) / 0.7163451   # -1.4%  n_sdz_above_5 < 2
        + 0.01148779 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) * (0.06215671 * 0.5 * ((0.3112717 - Q.sj4_dr_min) / 0.06215671) * (1 + math.erf(((0.3112717 - Q.sj4_dr_min) / 0.06215671) / math.sqrt(2)))) / 0.003058245   # +1.1%  M2_b2 < 0.04179 and sj4_dr_min < 0.3113
        - 0.01145101 * (3.556438 * 0.5 * ((76.91486 - Q.sj3_pair_mass_max) / 3.556438) * (1 + math.erf(((76.91486 - Q.sj3_pair_mass_max) / 3.556438) / math.sqrt(2)))) / 5.300402   # -1.1%  sj3_pair_mass_max < 76.91
        + 0.0113889 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) / 3.676114   # +1.1%  n_pairs_kt_above_3 < 28
        + 0.01098345 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (3.244489 * 0.5 * ((Q.sj4_pair_mass_max - 66.70506) / 3.244489) * (1 + math.erf(((Q.sj4_pair_mass_max - 66.70506) / 3.244489) / math.sqrt(2)))) / 5.216817   # +1.1%  lep_z < 0.3397 and sj4_pair_mass_max > 66.71
        - 0.01021413 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) / 8.618957   # -1.0%  lep_ptrel < 12.15
        + 0.00837662 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2)))) / 31.65413   # +0.8%  sj3_pair_mass_max < 120.6
        + 0.008054732 * (5.197 * 0.5 * ((95.14961 - Q.mass) / 5.197) * (1 + math.erf(((95.14961 - Q.mass) / 5.197) / math.sqrt(2)))) / 6.369586   # +0.8%  mass < 95.15
        - 0.007765286 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sdb_2_n - 3.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 3.0) / 1.0) / math.sqrt(2)))) / 2.252578   # -0.8%  lep_z < 0.3397 and sdb_2_n > 3
        - 0.007604892 * (0.02314387 * 0.5 * ((0.2509165 - Q.sdb_2_z) / 0.02314387) * (1 + math.erf(((0.2509165 - Q.sdb_2_z) / 0.02314387) / math.sqrt(2)))) / 0.04199024   # -0.8%  sdb_2_z < 0.2509
        - 0.007532688 * (0.03445022 * 0.5 * ((0.04982189 - Q.lepsj_3_dr) / 0.03445022) * (1 + math.erf(((0.04982189 - Q.lepsj_3_dr) / 0.03445022) / math.sqrt(2)))) / 0.03714658   # -0.8%  lepsj_3_dr < 0.04982
        - 0.007026635 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) * (1.0 * 0.5 * ((2.0 - Q.dc_2_n_disp3) / 1.0) * (1 + math.erf(((2.0 - Q.dc_2_n_disp3) / 1.0) / math.sqrt(2)))) / 0.02156071   # -0.7%  M2_b2 < 0.04179 and dc_2_n_disp3 < 2
        + 0.006600881 * (0.01476889 * 0.5 * ((0.1811493 - Q.pz_lnd2) / 0.01476889) * (1 + math.erf(((0.1811493 - Q.pz_lnd2) / 0.01476889) / math.sqrt(2)))) / 0.07223697   # +0.7%  pz_lnd2 < 0.1811
        + 0.006354669 * (3.003641 * 0.5 * ((40.24169 - Q.mass_neutral) / 3.003641) * (1 + math.erf(((40.24169 - Q.mass_neutral) / 3.003641) / math.sqrt(2)))) / 7.174533   # +0.6%  mass_neutral < 40.24
        + 0.00587909 * (0.07010728 * 0.5 * ((4.046329 - Q.lund_max_lnkt) / 0.07010728) * (1 + math.erf(((4.046329 - Q.lund_max_lnkt) / 0.07010728) / math.sqrt(2)))) / 0.3709323   # +0.6%  lund_max_lnkt < 4.046
        - 0.005529348 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (0.6891756 * 0.5 * ((6.112692 - Q.sj3_mass3) / 0.6891756) * (1 + math.erf(((6.112692 - Q.sj3_mass3) / 0.6891756) / math.sqrt(2)))) / 0.4864021   # -0.6%  lep_iso < 0.4382 and sj3_mass3 < 6.113
        + 0.005169513 * (0.002590499 * 0.5 * ((0.0255865 - Q.sum_z_dr2_top10) / 0.002590499) * (1 + math.erf(((0.0255865 - Q.sum_z_dr2_top10) / 0.002590499) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.005225227   # +0.5%  sum_z_dr2_top10 < 0.02559 and ak02_2_n_lep < 1
        + 0.005155152 * (0.007664379 * 0.5 * ((0.05414784 - Q.sum_z_dr2_top20) / 0.007664379) * (1 + math.erf(((0.05414784 - Q.sum_z_dr2_top20) / 0.007664379) / math.sqrt(2)))) / 0.02418133   # +0.5%  sum_z_dr2_top20 < 0.05415
        + 0.005148056 * (3.799482 * 0.5 * ((127.2317 - Q.mass) / 3.799482) * (1 + math.erf(((127.2317 - Q.mass) / 3.799482) / math.sqrt(2)))) * (0.01108125 * 0.5 * ((0.1325326 - Q.pz_lnd0) / 0.01108125) * (1 + math.erf(((0.1325326 - Q.pz_lnd0) / 0.01108125) / math.sqrt(2)))) / 0.5167883   # +0.5%  mass < 127.2 and pz_lnd0 < 0.1325
        - 0.005063017 * (4.146938 * 0.5 * ((110.2019 - Q.mass) / 4.146938) * (1 + math.erf(((110.2019 - Q.mass) / 4.146938) / math.sqrt(2)))) / 12.01898   # -0.5%  mass < 110.2
        - 0.005005084 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) / 0.01369289   # -0.5%  M2_b2 < 0.04179
        + 0.004873286 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2)))) / 1.091313   # +0.5%  mass < 56.49
        + 0.004683176 * (0.002590499 * 0.5 * ((0.0255865 - Q.sum_z_dr2_top10) / 0.002590499) * (1 + math.erf(((0.0255865 - Q.sum_z_dr2_top10) / 0.002590499) / math.sqrt(2)))) / 0.00694583   # +0.5%  sum_z_dr2_top10 < 0.02559
        + 0.004677994 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.isphoton_29 - 1.0) / 1.0) * (1 + math.erf(((Q.isphoton_29 - 1.0) / 1.0) / math.sqrt(2)))) / 3.48461   # +0.5%  sj3_pair_mass_max < 120.6 and isphoton_29 > 1
        - 0.004562107 * (3.799482 * 0.5 * ((127.2317 - Q.mass) / 3.799482) * (1 + math.erf(((127.2317 - Q.mass) / 3.799482) / math.sqrt(2)))) / 21.74576   # -0.5%  mass < 127.2
        - 0.004499586 * (0.0125626 * 0.5 * ((Q.C2_b05 - 0.3223395) / 0.0125626) * (1 + math.erf(((Q.C2_b05 - 0.3223395) / 0.0125626) / math.sqrt(2)))) / 0.008534453   # -0.4%  C2_b05 > 0.3223
        - 0.003871838 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2)))) / 0.003263441   # -0.4%  lep_iso < 0.4382 and dc_1_n_lep < 0
        - 0.003847281 * (0.02184861 * 0.5 * ((0.6020351 - Q.nca_sj4_pair2nd_over_mass) / 0.02184861) * (1 + math.erf(((0.6020351 - Q.nca_sj4_pair2nd_over_mass) / 0.02184861) / math.sqrt(2)))) * (0.01280829 * 0.5 * ((Q.z_neutral_had - 0.05062943) / 0.01280829) * (1 + math.erf(((Q.z_neutral_had - 0.05062943) / 0.01280829) / math.sqrt(2)))) / 0.01044598   # -0.4%  nca_sj4_pair2nd_over_mass < 0.602 and z_neutral_had > 0.05063
        - 0.003557048 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) * (0.01117424 * 0.5 * ((0.03578969 - Q.pz_lnkt4) / 0.01117424) * (1 + math.erf(((0.03578969 - Q.pz_lnkt4) / 0.01117424) / math.sqrt(2)))) / 0.0003466384   # -0.4%  M2_b2 < 0.04179 and pz_lnkt4 < 0.03579
        + 0.003548778 * (13.55261 * 0.5 * ((35.45098 - Q.sv_1_sd0_sum) / 13.55261) * (1 + math.erf(((35.45098 - Q.sv_1_sd0_sum) / 13.55261) / math.sqrt(2)))) / 16.9383   # +0.4%  sv_1_sd0_sum < 35.45
        - 0.003413593 * (0.1359916 * 0.5 * ((Q.pair_max_lnm2 - 7.203613) / 0.1359916) * (1 + math.erf(((Q.pair_max_lnm2 - 7.203613) / 0.1359916) / math.sqrt(2)))) / 0.1347303   # -0.3%  pair_max_lnm2 > 7.204
        + 0.003403989 * (0.04122765 * 0.5 * ((Q.N2_b05 - 0.302405) / 0.04122765) * (1 + math.erf(((Q.N2_b05 - 0.302405) / 0.04122765) / math.sqrt(2)))) / 0.137646   # +0.3%  N2_b05 > 0.3024
        + 0.003374726 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2)))) * (702.9612 * 0.5 * ((379.5631 - Q.sv_2_sd0_sum) / 702.9612) * (1 + math.erf(((379.5631 - Q.sv_2_sd0_sum) / 702.9612) / math.sqrt(2)))) / 7700.853   # +0.3%  sj3_pair_mass_max < 120.6 and sv_2_sd0_sum < 379.6
        - 0.003350995 * (3.003641 * 0.5 * ((40.24169 - Q.mass_neutral) / 3.003641) * (1 + math.erf(((40.24169 - Q.mass_neutral) / 3.003641) / math.sqrt(2)))) * (467.791 * 0.5 * ((593.8527 - Q.dc_1_sd0_1) / 467.791) * (1 + math.erf(((593.8527 - Q.dc_1_sd0_1) / 467.791) / math.sqrt(2)))) / 3496.972   # -0.3%  mass_neutral < 40.24 and dc_1_sd0_1 < 593.9
        + 0.003069038 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) * (3.347501 * 0.5 * ((Q.mass_displaced3 - 9.090532) / 3.347501) * (1 + math.erf(((Q.mass_displaced3 - 9.090532) / 3.347501) / math.sqrt(2)))) / 0.07485035   # +0.3%  M2_b2 < 0.04179 and mass_displaced3 > 9.091
        + 0.003059782 * (3.5 * 0.5 * ((Q.n_dr_0p4_up - 15.0) / 3.5) * (1 + math.erf(((Q.n_dr_0p4_up - 15.0) / 3.5) / math.sqrt(2)))) / 0.6807532   # +0.3%  n_dr_0p4_up > 15
        - 0.002954809 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.sdb_4_n) / 1.0) * (1 + math.erf(((1.0 - Q.sdb_4_n) / 1.0) / math.sqrt(2)))) / 0.1574999   # -0.3%  lep_z < 0.3397 and sdb_4_n < 1
        + 0.00289656 * (0.01141967 * 0.5 * ((0.03624058 - Q.z_displaced3) / 0.01141967) * (1 + math.erf(((0.03624058 - Q.z_displaced3) / 0.01141967) / math.sqrt(2)))) / 0.01240284   # +0.3%  z_displaced3 < 0.03624
        + 0.002825325 * (6.955739 * 0.5 * ((125.6411 - Q.mres_sd_mass_b0z02) / 6.955739) * (1 + math.erf(((125.6411 - Q.mres_sd_mass_b0z02) / 6.955739) / math.sqrt(2)))) / 55.70318   # +0.3%  mres_sd_mass_b0z02 < 125.6
        - 0.002293747 * (0.02184861 * 0.5 * ((0.6020351 - Q.nca_sj4_pair2nd_over_mass) / 0.02184861) * (1 + math.erf(((0.6020351 - Q.nca_sj4_pair2nd_over_mass) / 0.02184861) / math.sqrt(2)))) / 0.1079316   # -0.2%  nca_sj4_pair2nd_over_mass < 0.602
        - 0.002203361 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2)))) * (0.1767711 * 0.5 * ((Q.lne_1 - 3.745821) / 0.1767711) * (1 + math.erf(((Q.lne_1 - 3.745821) / 0.1767711) / math.sqrt(2)))) / 29.49341   # -0.2%  sj3_pair_mass_max < 120.6 and lne_1 > 3.746
        + 0.001926671 * (3.521114 * 0.5 * ((Q.mass_top20 - 93.50967) / 3.521114) * (1 + math.erf(((Q.mass_top20 - 93.50967) / 3.521114) / math.sqrt(2)))) / 11.20857   # +0.2%  mass_top20 > 93.51
        - 0.001782806 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2)))) * (3.006676 * 0.5 * ((6.43167 - Q.jd_3d_4) / 3.006676) * (1 + math.erf(((6.43167 - Q.jd_3d_4) / 3.006676) / math.sqrt(2)))) / 97.42529   # -0.2%  sj3_pair_mass_max < 120.6 and jd_3d_4 < 6.432
        - 0.001657167 * (0.04018594 * 0.5 * ((0.5498426 - Q.tau32) / 0.04018594) * (1 + math.erf(((0.5498426 - Q.tau32) / 0.04018594) / math.sqrt(2)))) * (0.0005676896 * 0.5 * ((Q.e3 - 0.00228569) / 0.0005676896) * (1 + math.erf(((Q.e3 - 0.00228569) / 0.0005676896) / math.sqrt(2)))) / 3.801504e-05   # -0.2%  tau32 < 0.5498 and e3 > 0.002286
        - 0.001591525 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.523415 * 0.5 * ((2.133094 - Q.sjf_4_4_mass_d3) / 1.523415) * (1 + math.erf(((2.133094 - Q.sjf_4_4_mass_d3) / 1.523415) / math.sqrt(2)))) / 0.4827891   # -0.2%  lep_z < 0.3397 and sjf_4_4_mass_d3 < 2.133
        - 0.001443427 * (1.0 * 0.5 * ((10.0 - Q.n_photon) / 1.0) * (1 + math.erf(((10.0 - Q.n_photon) / 1.0) / math.sqrt(2)))) / 0.6708074   # -0.1%  n_photon < 10
        - 0.001356717 * (0.04018594 * 0.5 * ((0.5498426 - Q.tau32) / 0.04018594) * (1 + math.erf(((0.5498426 - Q.tau32) / 0.04018594) / math.sqrt(2)))) * (1.0 * 0.5 * ((5.0 - Q.n_pairs_kt_above_10) / 1.0) * (1 + math.erf(((5.0 - Q.n_pairs_kt_above_10) / 1.0) / math.sqrt(2)))) / 0.03769255   # -0.1%  tau32 < 0.5498 and n_pairs_kt_above_10 < 5
        + 0.001312091 * (1.0 * 0.5 * ((Q.dc_1_n_disp3 - 1.0) / 1.0) * (1 + math.erf(((Q.dc_1_n_disp3 - 1.0) / 1.0) / math.sqrt(2)))) / 0.3682282   # +0.1%  dc_1_n_disp3 > 1
        + 0.001171492 * (2.0 * 0.5 * ((26.0 - Q.n_neutral) / 2.0) * (1 + math.erf(((26.0 - Q.n_neutral) / 2.0) / math.sqrt(2)))) / 7.338909   # +0.1%  n_neutral < 26
        - 0.001089812 * (3.799482 * 0.5 * ((127.2317 - Q.mass) / 3.799482) * (1 + math.erf(((127.2317 - Q.mass) / 3.799482) / math.sqrt(2)))) * (0.005274752 * 0.5 * ((-0.01099959 - Q.sjq_2_prod_k1) / 0.005274752) * (1 + math.erf(((-0.01099959 - Q.sjq_2_prod_k1) / 0.005274752) / math.sqrt(2)))) / 0.7730101   # -0.1%  mass < 127.2 and sjq_2_prod_k1 < -0.011
        - 0.0009731099 * (0.04018594 * 0.5 * ((0.5498426 - Q.tau32) / 0.04018594) * (1 + math.erf(((0.5498426 - Q.tau32) / 0.04018594) / math.sqrt(2)))) / 0.03573005   # -0.1%  tau32 < 0.5498
        - 0.0009360846 * (3.003641 * 0.5 * ((40.24169 - Q.mass_neutral) / 3.003641) * (1 + math.erf(((40.24169 - Q.mass_neutral) / 3.003641) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 0.0) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 0.0) / 1.0) / math.sqrt(2)))) / 2.798119   # -0.1%  mass_neutral < 40.24 and lepsj_2_n_d3 > 0
        + 0.0008183648 * (6.104222 * 0.5 * ((0.0 - Q.ak02_1_mass) / 6.104222) * (1 + math.erf(((0.0 - Q.ak02_1_mass) / 6.104222) / math.sqrt(2)))) / 0.06109526   # +0.1%  ak02_1_mass < 0
        - 0.0008134523 * (2.295472 * 0.5 * ((21.21062 - Q.sj2_mass1) / 2.295472) * (1 + math.erf(((21.21062 - Q.sj2_mass1) / 2.295472) / math.sqrt(2)))) / 1.736417   # -0.1%  sj2_mass1 < 21.21
        - 0.0008126082 * (0.04018594 * 0.5 * ((0.5498426 - Q.tau32) / 0.04018594) * (1 + math.erf(((0.5498426 - Q.tau32) / 0.04018594) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_pairs_kt_above_30 - 1.0) / 1.0) * (1 + math.erf(((Q.n_pairs_kt_above_30 - 1.0) / 1.0) / math.sqrt(2)))) / 0.01007209   # -0.1%  tau32 < 0.5498 and n_pairs_kt_above_30 > 1
        - 0.00080837 * (0.09626377 * 0.5 * ((Q.sjf_2_1_z_d3 - 0.3672808) / 0.09626377) * (1 + math.erf(((Q.sjf_2_1_z_d3 - 0.3672808) / 0.09626377) / math.sqrt(2)))) / 0.006253243   # -0.1%  sjf_2_1_z_d3 > 0.3673
        + 0.0006068981 * (2.113111 * 0.5 * ((3.373037 - Q.sip_3d_1) / 2.113111) * (1 + math.erf(((3.373037 - Q.sip_3d_1) / 2.113111) / math.sqrt(2)))) / 0.1387434   # +0.1%  sip_3d_1 < 3.373
        - 0.0005442144 * (0.01476889 * 0.5 * ((0.1811493 - Q.pz_lnd2) / 0.01476889) * (1 + math.erf(((0.1811493 - Q.pz_lnd2) / 0.01476889) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_4_3_n_d5 - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_4_3_n_d5 - 1.0) / 1.0) / math.sqrt(2)))) / 0.02363155   # -0.1%  pz_lnd2 < 0.1811 and sjf_4_3_n_d5 > 1
        - 0.000514148 * (3.421198 * 0.5 * ((69.01131 - Q.nca_sj4_pair_mass_2nd) / 3.421198) * (1 + math.erf(((69.01131 - Q.nca_sj4_pair_mass_2nd) / 3.421198) / math.sqrt(2)))) * (0.1712927 * 0.5 * ((Q.dc_3_charge - 0.1143891) / 0.1712927) * (1 + math.erf(((Q.dc_3_charge - 0.1143891) / 0.1712927) / math.sqrt(2)))) / 0.9528865   # -0.1%  nca_sj4_pair_mass_2nd < 69.01 and dc_3_charge > 0.1144
        - 0.0004654976 * (9.707626 * 0.5 * ((16.47324 - Q.dc_1_sd0_1) / 9.707626) * (1 + math.erf(((16.47324 - Q.dc_1_sd0_1) / 9.707626) / math.sqrt(2)))) * (0.03500366 * 0.5 * ((Q.dzerr_22 - 0.0) / 0.03500366) * (1 + math.erf(((Q.dzerr_22 - 0.0) / 0.03500366) / math.sqrt(2)))) / 0.3181887   # -0.0%  dc_1_sd0_1 < 16.47 and dzerr_22 > 0
        - 0.0004238814 * (9.707626 * 0.5 * ((16.47324 - Q.dc_1_sd0_1) / 9.707626) * (1 + math.erf(((16.47324 - Q.dc_1_sd0_1) / 9.707626) / math.sqrt(2)))) / 10.07866   # -0.0%  dc_1_sd0_1 < 16.47
        - 0.0004108452 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (2.54381e-06 * 0.5 * ((3.53193e-06 - Q.e4) / 2.54381e-06) * (1 + math.erf(((3.53193e-06 - Q.e4) / 2.54381e-06) / math.sqrt(2)))) / 1.990101e-06   # -0.0%  lep_ptrel > 43.21 and e4 < 3.532e-06
        - 0.0004046161 * (0.3121171 * 0.5 * ((2.17433 - Q.sip_3d_1) / 0.3121171) * (1 + math.erf(((2.17433 - Q.sip_3d_1) / 0.3121171) / math.sqrt(2)))) / 0.0160733   # -0.0%  sip_3d_1 < 2.174
        - 0.000329119 * (0.09718233 * 0.5 * ((Q.z_displaced3 - 0.4232894) / 0.09718233) * (1 + math.erf(((Q.z_displaced3 - 0.4232894) / 0.09718233) / math.sqrt(2)))) / 0.006313196   # -0.0%  z_displaced3 > 0.4233
        + 0.0002760628 * (3.799482 * 0.5 * ((127.2317 - Q.mass) / 3.799482) * (1 + math.erf(((127.2317 - Q.mass) / 3.799482) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.5099508   # +0.0%  mass < 127.2 and dc_2_n_lep < 0
        - 0.0002108198 * (0.001664982 * 0.5 * ((0.007018285 - Q.sum_z_dr2_top3) / 0.001664982) * (1 + math.erf(((0.007018285 - Q.sum_z_dr2_top3) / 0.001664982) / math.sqrt(2)))) / 0.001200679   # -0.0%  sum_z_dr2_top3 < 0.007018
        + 0.0002003606 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2)))) * (0.02812676 * 0.5 * ((Q.sj3_dr13 - 0.3524911) / 0.02812676) * (1 + math.erf(((Q.sj3_dr13 - 0.3524911) / 0.02812676) / math.sqrt(2)))) / 2.705599   # +0.0%  sj3_pair_mass_max < 120.6 and sj3_dr13 > 0.3525
        + 0.0001721222 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) / 0.9643417   # +0.0%  lep_ptrel > 43.21
        + 0.0001710152 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.ischhad_10) / 1.0) * (1 + math.erf(((0.0 - Q.ischhad_10) / 1.0) / math.sqrt(2)))) / 0.02494768   # +0.0%  lep_z < 0.3397 and ischhad_10 < 0
        - 0.0001333348 * (1.0 * 0.5 * ((Q.n_dr_0p4_up - 2.0) / 1.0) * (1 + math.erf(((Q.n_dr_0p4_up - 2.0) / 1.0) / math.sqrt(2)))) / 4.529206   # -0.0%  n_dr_0p4_up > 2
        - 0.0001261635 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.02288284 * 0.5 * ((0.212136 - Q.z_neutral_had) / 0.02288284) * (1 + math.erf(((0.212136 - Q.z_neutral_had) / 0.02288284) / math.sqrt(2)))) / 0.671426   # -0.0%  lep_ptrel < 12.15 and z_neutral_had < 0.2121
        + 9.903161e-05 * (4.196564 * 0.5 * ((Q.sj4_pair_mass_max - 92.5542) / 4.196564) * (1 + math.erf(((Q.sj4_pair_mass_max - 92.5542) / 4.196564) / math.sqrt(2)))) / 6.408398   # +0.0%  sj4_pair_mass_max > 92.55
        + 5.882037e-05 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 1.0) / 1.0) / math.sqrt(2)))) / 0.4067986   # +0.0%  lep_ptrel > 43.21 and lepsj_2_n_d3 > 1
        - 4.569473e-05 * (1.0 * 0.5 * ((Q.n_dr_0p4_up - 2.0) / 1.0) * (1 + math.erf(((Q.n_dr_0p4_up - 2.0) / 1.0) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.charge_76 - 0.0) / 1e-06) * (1 + math.erf(((Q.charge_76 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.03477909   # -0.0%  n_dr_0p4_up > 2 and charge_76 > 0
        - 3.725877e-05 * (0.04122765 * 0.5 * ((Q.N2_b05 - 0.302405) / 0.04122765) * (1 + math.erf(((Q.N2_b05 - 0.302405) / 0.04122765) / math.sqrt(2)))) * (0.009933948 * 0.5 * ((Q.kt2_2_z_disp3 - 0.02323978) / 0.009933948) * (1 + math.erf(((Q.kt2_2_z_disp3 - 0.02323978) / 0.009933948) / math.sqrt(2)))) / 0.001008843   # -0.0%  N2_b05 > 0.3024 and kt2_2_z_disp3 > 0.02324
        + 1.941098e-05 * (0.0125626 * 0.5 * ((Q.C2_b05 - 0.3223395) / 0.0125626) * (1 + math.erf(((Q.C2_b05 - 0.3223395) / 0.0125626) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dc_mass_disp_2nd - 0.0) / 1e-06) * (1 + math.erf(((Q.dc_mass_disp_2nd - 0.0) / 1e-06) / math.sqrt(2)))) / 0.001116217   # +0.0%  C2_b05 > 0.3223 and dc_mass_disp_2nd > 0
        - 4.374174e-06 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.01535144 * 0.5 * ((0.1124664 - Q.sj4_zsoft) / 0.01535144) * (1 + math.erf(((0.1124664 - Q.sj4_zsoft) / 0.01535144) / math.sqrt(2)))) / 0.01558626   # -0.0%  lep_z < 0.3397 and sj4_zsoft < 0.1125
        + 1.895295e-07 * (3.421198 * 0.5 * ((69.01131 - Q.nca_sj4_pair_mass_2nd) / 3.421198) * (1 + math.erf(((69.01131 - Q.nca_sj4_pair_mass_2nd) / 3.421198) / math.sqrt(2)))) / 16.05602   # +0.0%  nca_sj4_pair_mass_2nd < 69.01
    )
    return z


def neuron_26(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.231261e-06
    )
    return z


def neuron_27(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.308491e-06
    )
    return z


def neuron_28(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.617203e-06
    )
    return z


def neuron_29(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.183854e-06
    )
    return z


def neuron_30(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.603046e-06
    )
    return z


def neuron_31(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.509519e-06
    )
    return z


def neuron_32(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.441347e-05
    )
    return z


def neuron_33(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.567321e-06
    )
    return z


def neuron_34(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.388485e-05
    )
    return z


def neuron_35(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.743291e-06
    )
    return z


def neuron_36(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.831943e-06
    )
    return z


def neuron_37(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.357431e-06
    )
    return z


def neuron_38(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.411165e-06
    )
    return z


def neuron_39(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.27427e-06
    )
    return z


def neuron_40(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.775515e-07
    )
    return z


def neuron_41(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.814669e-08
    )
    return z


def neuron_42(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.779036e-06
    )
    return z


def neuron_43(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.407703e-06
    )
    return z


def neuron_44(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.324341e-06
    )
    return z


def neuron_45(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.849329e-06
    )
    return z


def neuron_46(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.368757e-07
    )
    return z


def neuron_47(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.492068e-07
    )
    return z


def neuron_48(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.791772e-06
    )
    return z


def neuron_49(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.243813e-06
    )
    return z


def neuron_50(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.175112e-06
    )
    return z


def neuron_51(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.012201e-06
    )
    return z


def neuron_52(Q):
    # scale S = 10.67; each line: share * term / its average size
    z = 10.67076 * (-0.1308172
        + 0.1883265 * (18.66842 * 0.5 * ((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) / 56.92363   # +18.8%  mres_sd_mass_b0z005 < 159.9
        + 0.0562878 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) / 31.48666   # +5.6%  mres_sd_mass_b0z005 > 81.62
        - 0.05367881 * (1.5 * 0.5 * ((6.0 - Q.n_sd0_above_3) / 1.5) * (1 + math.erf(((6.0 - Q.n_sd0_above_3) / 1.5) / math.sqrt(2)))) / 3.13619   # -5.4%  n_sd0_above_3 < 6
        + 0.04409345 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) / 0.07071767   # +4.4%  z_displaced3 < 0.1343
        - 0.04146026 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) / 0.2702483   # -4.1%  lep_z < 0.3397
        - 0.03592869 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) * (0.0330605 * 0.5 * ((0.114659 - Q.lep_dr) / 0.0330605) * (1 + math.erf(((0.114659 - Q.lep_dr) / 0.0330605) / math.sqrt(2)))) / 0.1110396   # -3.6%  lepsj_3_maxsd0 < 2.413 and lep_dr < 0.1147
        + 0.03415004 * (1.5 * 0.5 * ((7.0 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.0 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2)))) / 3.650329   # +3.4%  n_s3d_above_3 < 7
        - 0.03115415 * (14.42334 * 0.5 * ((141.5589 - Q.mres_sd_mass_b0z005) / 14.42334) * (1 + math.erf(((141.5589 - Q.mres_sd_mass_b0z005) / 14.42334) / math.sqrt(2)))) / 40.76608   # -3.1%  mres_sd_mass_b0z005 < 141.6
        - 0.0304372 * (4.488781 * 0.5 * ((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) * (1 + math.erf(((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) / math.sqrt(2)))) / 28.3744   # -3.0%  mres_sd_mass_b0z005 < 125.9
        + 0.024004 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) / 1.223619   # +2.4%  lepsj_3_maxsd0 < 2.413
        + 0.0232456 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) / 0.8420682   # +2.3%  lep_iso < 1.362
        - 0.02280807 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) / 0.04874886   # -2.3%  lepsj_2_dr < 0.06573
        - 0.01819827 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) / 26.78553   # -1.8%  sv_1_sd0_sum < 51.56
        - 0.01725094 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2)))) / 1.708113   # -1.7%  lne_16 > 0.4469
        + 0.01628165 * (1.0 * 0.5 * ((1.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2)))) / 0.667546   # +1.6%  dc_1_n_lep < 1
        - 0.01504706 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (4.0 * 0.5 * ((24.0 - Q.n_dr_0p2_0p4) / 4.0) * (1 + math.erf(((24.0 - Q.n_dr_0p2_0p4) / 4.0) / math.sqrt(2)))) / 11.27712   # -1.5%  lep_iso < 1.362 and n_dr_0p2_0p4 < 24
        - 0.01427206 * (1.5 * 0.5 * ((6.0 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((6.0 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2)))) / 2.341129   # -1.4%  n_dr_0p4_up < 6
        - 0.01332691 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (110.8422 * 0.5 * ((114.1102 - Q.sjf_2_2_max3d) / 110.8422) * (1 + math.erf(((114.1102 - Q.sjf_2_2_max3d) / 110.8422) / math.sqrt(2)))) / 5.045129   # -1.3%  z_displaced3 < 0.1343 and sjf_2_2_max3d < 114.1
        + 0.01304014 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.03966789 * 0.5 * ((0.1333792 - Q.ak02_3_z) / 0.03966789) * (1 + math.erf(((0.1333792 - Q.ak02_3_z) / 0.03966789) / math.sqrt(2)))) / 2.153733   # +1.3%  mres_sd_mass_b0z005 > 81.62 and ak02_3_z < 0.1334
        + 0.01201258 * (3.069852 * 0.5 * ((4.516968 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.516968 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) / 1.027676   # +1.2%  sjf_2_2_max3d < 4.517
        + 0.01189581 * (2.5 * 0.5 * ((Q.n_particles - 26.0) / 2.5) * (1 + math.erf(((Q.n_particles - 26.0) / 2.5) / math.sqrt(2)))) / 14.50415   # +1.2%  n_particles > 26
        + 0.01172383 * (11.49171 * 0.5 * ((71.96396 - Q.mass) / 11.49171) * (1 + math.erf(((71.96396 - Q.mass) / 11.49171) / math.sqrt(2)))) / 2.139873   # +1.2%  mass < 71.96
        + 0.01147878 * (0.1159667 * 0.5 * ((0.4849193 - Q.sv_1_dr) / 0.1159667) * (1 + math.erf(((0.4849193 - Q.sv_1_dr) / 0.1159667) / math.sqrt(2)))) / 0.3439647   # +1.1%  sv_1_dr < 0.4849
        - 0.01122018 * (18.66842 * 0.5 * ((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) * (0.04741882 * 0.5 * ((0.4690304 - Q.tau21_b2) / 0.04741882) * (1 + math.erf(((0.4690304 - Q.tau21_b2) / 0.04741882) / math.sqrt(2)))) / 10.1517   # -1.1%  mres_sd_mass_b0z005 < 159.9 and tau21_b2 < 0.469
        - 0.0111267 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (401.9953 * 0.5 * ((577.991 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((577.991 - Q.sip_3d_3) / 401.9953) / math.sqrt(2)))) / 35.444   # -1.1%  z_displaced3 < 0.1343 and sip_3d_3 < 578
        + 0.01112143 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 11.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 11.0) / 1.0) / math.sqrt(2)))) / 2.477971   # +1.1%  lep_z < 0.3397 and n_charged_pt_above_1 > 11
        - 0.01068744 * (17.1726 * 0.5 * ((34.81061 - Q.sjf_2_1_max3d) / 17.1726) * (1 + math.erf(((34.81061 - Q.sjf_2_1_max3d) / 17.1726) / math.sqrt(2)))) / 14.92418   # -1.1%  sjf_2_1_max3d < 34.81
        + 0.01024221 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) / 0.0002870086   # +1.0%  e3_b2 < 0.0004127
        - 0.0101067 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) / 3.720581   # -1.0%  n_lund_kt_above_1 < 9
        + 0.009593593 * Q.ecf_g41 / 0.0002120905   # +1.0%  ecf_g41
        - 0.009234734 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.03693697 * 0.5 * ((0.5838518 - Q.tau32_b2) / 0.03693697) * (1 + math.erf(((0.5838518 - Q.tau32_b2) / 0.03693697) / math.sqrt(2)))) / 4.886321   # -0.9%  mres_sd_mass_b0z005 > 81.62 and tau32_b2 < 0.5839
        - 0.009044198 * (0.0408915 * 0.5 * ((Q.lep_dr - 0.04607888) / 0.0408915) * (1 + math.erf(((Q.lep_dr - 0.04607888) / 0.0408915) / math.sqrt(2)))) / 0.07708341   # -0.9%  lep_dr > 0.04608
        + 0.008143678 * (0.7202602 * 0.5 * ((1.462588 - Q.D3_b2) / 0.7202602) * (1 + math.erf(((1.462588 - Q.D3_b2) / 0.7202602) / math.sqrt(2)))) / 1.191742   # +0.8%  D3_b2 < 1.463
        + 0.008100708 * (4.481852 * 0.5 * ((6.220692 - Q.sjf_2_1_max3d) / 4.481852) * (1 + math.erf(((6.220692 - Q.sjf_2_1_max3d) / 4.481852) / math.sqrt(2)))) / 1.242535   # +0.8%  sjf_2_1_max3d < 6.221
        + 0.007698116 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.01565591 * 0.5 * ((0.1681753 - Q.pz_lnkt1) / 0.01565591) * (1 + math.erf(((0.1681753 - Q.pz_lnkt1) / 0.01565591) / math.sqrt(2)))) / 1.863622e-05   # +0.8%  e3_b2 < 0.0004127 and pz_lnkt1 < 0.1682
        + 0.007378709 * (3.304036 * 0.5 * ((115.614 - Q.mass_top50) / 3.304036) * (1 + math.erf(((115.614 - Q.mass_top50) / 3.304036) / math.sqrt(2)))) / 15.0409   # +0.7%  mass_top50 < 115.6
        - 0.007254569 * (24.25363 * 0.5 * ((53.57509 - Q.mres_sd_mass_b0z005) / 24.25363) * (1 + math.erf(((53.57509 - Q.mres_sd_mass_b0z005) / 24.25363) / math.sqrt(2)))) / 3.824245   # -0.7%  mres_sd_mass_b0z005 < 53.58
        + 0.007210214 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) * (0.05566634 * 0.5 * ((0.3093201 - Q.z_neutral_had) / 0.05566634) * (1 + math.erf(((0.3093201 - Q.z_neutral_had) / 0.05566634) / math.sqrt(2)))) / 0.6586621   # +0.7%  n_lund_kt_above_1 < 9 and z_neutral_had < 0.3093
        - 0.006123025 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) * (0.3035439 * 0.5 * ((Q.sjq_2_prod_k03 - -0.9998116) / 0.3035439) * (1 + math.erf(((Q.sjq_2_prod_k03 - -0.9998116) / 0.3035439) / math.sqrt(2)))) / 3.351014   # -0.6%  n_lund_kt_above_1 < 9 and sjq_2_prod_k03 > -0.9998
        + 0.005742778 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (0.02727943 * 0.5 * ((Q.N2 - 0.1824346) / 0.02727943) * (1 + math.erf(((Q.N2 - 0.1824346) / 0.02727943) / math.sqrt(2)))) / 0.009798652   # +0.6%  z_displaced3 < 0.1343 and N2 > 0.1824
        - 0.005031221 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) / math.sqrt(2)))) / 1.914224   # -0.5%  mres_sd_mass_b0z005 > 178.9
        - 0.004606716 * (1.0 * 0.5 * ((1.0 - Q.ak02_1_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_1_n_lep) / 1.0) / math.sqrt(2)))) / 0.6330793   # -0.5%  ak02_1_n_lep < 1
        + 0.004216919 * (22.16789 * 0.5 * ((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) * (1 + math.erf(((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) / math.sqrt(2)))) / 6.631784   # +0.4%  lepsj_3_maxsd0 < 11.53
        + 0.004192032 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 5.193195   # +0.4%  n_pairs_kt_above_1 < 80
        + 0.004054516 * (0.2681069 * 0.5 * ((Q.sjq_2_2_k1 - 0.6477929) / 0.2681069) * (1 + math.erf(((Q.sjq_2_2_k1 - 0.6477929) / 0.2681069) / math.sqrt(2)))) / 0.02066679   # +0.4%  sjq_2_2_k1 > 0.6478
        + 0.004013238 * (1.0 * 0.5 * ((2.0 - Q.sum_charge) / 1.0) * (1 + math.erf(((2.0 - Q.sum_charge) / 1.0) / math.sqrt(2)))) / 2.130099   # +0.4%  sum_charge < 2
        + 0.004011847 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2)))) * (0.05296894 * 0.5 * ((0.1145956 - Q.sjq_3_prod_k1) / 0.05296894) * (1 + math.erf(((0.1145956 - Q.sjq_3_prod_k1) / 0.05296894) / math.sqrt(2)))) / 0.2118377   # +0.4%  lne_16 > 0.4469 and sjq_3_prod_k1 < 0.1146
        - 0.003926156 * (0.09740396 * 0.5 * ((1.325034 - Q.D2) / 0.09740396) * (1 + math.erf(((1.325034 - Q.D2) / 0.09740396) / math.sqrt(2)))) / 0.1068502   # -0.4%  D2 < 1.325
        - 0.003924967 * (18.66842 * 0.5 * ((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2)))) / 54.17346   # -0.4%  mres_sd_mass_b0z005 < 159.9 and dc_n > 1
        - 0.003921496 * (3.304036 * 0.5 * ((115.614 - Q.mass_top50) / 3.304036) * (1 + math.erf(((115.614 - Q.mass_top50) / 3.304036) / math.sqrt(2)))) * (1.0 * 0.5 * ((6.0 - Q.sdb_1_n) / 1.0) * (1 + math.erf(((6.0 - Q.sdb_1_n) / 1.0) / math.sqrt(2)))) / 59.04629   # -0.4%  mass_top50 < 115.6 and sdb_1_n < 6
        + 0.003678924 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) / 0.5234339   # +0.4%  n_lund_kt_above_1 < 9 and nca_sj4_pairmax_over_mass < 0.8538
        + 0.003562196 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2)))) / 2.198028   # +0.4%  lep_ptrel > 27.32
        + 0.003526221 * (0.09740396 * 0.5 * ((1.325034 - Q.D2) / 0.09740396) * (1 + math.erf(((1.325034 - Q.D2) / 0.09740396) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjq_3_2_nch - 4.0) / 1.0) * (1 + math.erf(((Q.sjq_3_2_nch - 4.0) / 1.0) / math.sqrt(2)))) / 0.2273369   # +0.4%  D2 < 1.325 and sjq_3_2_nch > 4
        - 0.003230504 * (2.497115 * 0.5 * ((Q.mass_displaced5 - 6.387683) / 2.497115) * (1 + math.erf(((Q.mass_displaced5 - 6.387683) / 2.497115) / math.sqrt(2)))) / 3.882933   # -0.3%  mass_displaced5 > 6.388
        - 0.003124925 * (0.08734879 * 0.5 * ((1.608106 - Q.jd_3d_6) / 0.08734879) * (1 + math.erf(((1.608106 - Q.jd_3d_6) / 0.08734879) / math.sqrt(2)))) / 0.1222875   # -0.3%  jd_3d_6 < 1.608
        + 0.002958098 * (0.2610212 * 0.5 * ((-0.6337755 - Q.sjq_2_2_k1) / 0.2610212) * (1 + math.erf(((-0.6337755 - Q.sjq_2_2_k1) / 0.2610212) / math.sqrt(2)))) / 0.02066928   # +0.3%  sjq_2_2_k1 < -0.6338
        + 0.002812207 * (1.0 * 0.5 * ((8.0 - Q.n_lund) / 1.0) * (1 + math.erf(((8.0 - Q.n_lund) / 1.0) / math.sqrt(2)))) / 0.412674   # +0.3%  n_lund < 8
        - 0.002377012 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2)))) / 1.091313   # -0.2%  mass < 56.49
        + 0.002331881 * (0.0326228 * 0.5 * ((Q.kt2_dr12 - 0.4918357) / 0.0326228) * (1 + math.erf(((Q.kt2_dr12 - 0.4918357) / 0.0326228) / math.sqrt(2)))) / 0.02088838   # +0.2%  kt2_dr12 > 0.4918
        + 0.002255123 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2)))) * (0.02550179 * 0.5 * ((Q.sdb_2_z - 0.1792389) / 0.02550179) * (1 + math.erf(((Q.sdb_2_z - 0.1792389) / 0.02550179) / math.sqrt(2)))) / 0.2474472   # +0.2%  lne_16 > 0.4469 and sdb_2_z > 0.1792
        - 0.002173022 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) / 8.618957   # -0.2%  lep_ptrel < 12.15
        - 0.001893336 * (0.08168809 * 0.5 * ((Q.jet_abs_eta - 0.8317778) / 0.08168809) * (1 + math.erf(((Q.jet_abs_eta - 0.8317778) / 0.08168809) / math.sqrt(2)))) / 0.1685869   # -0.2%  jet_abs_eta > 0.8318
        - 0.001808913 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_3_3_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_3_3_n_d3 - 1.0) / 1.0) / math.sqrt(2)))) / 0.01752883   # -0.2%  z_displaced3 < 0.1343 and sjf_3_3_n_d3 > 1
        + 0.001803243 * (0.01743646 * 0.5 * ((0.3394107 - Q.sj2_dr) / 0.01743646) * (1 + math.erf(((0.3394107 - Q.sj2_dr) / 0.01743646) / math.sqrt(2)))) / 0.03755481   # +0.2%  sj2_dr < 0.3394
        + 0.00179369 * (0.0496275 * 0.5 * ((0.1915984 - Q.kt2_dr12) / 0.0496275) * (1 + math.erf(((0.1915984 - Q.kt2_dr12) / 0.0496275) / math.sqrt(2)))) / 0.008113201   # +0.2%  kt2_dr12 < 0.1916
        - 0.0017668 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.01952245 * 0.5 * ((0.1701336 - Q.dr_21) / 0.01952245) * (1 + math.erf(((0.1701336 - Q.dr_21) / 0.01952245) / math.sqrt(2)))) / 0.3612983   # -0.2%  lep_ptrel < 12.15 and dr_21 < 0.1701
        + 0.001684075 * (0.007529243 * 0.5 * ((0.01782783 - Q.z_displaced3) / 0.007529243) * (1 + math.erf(((0.01782783 - Q.z_displaced3) / 0.007529243) / math.sqrt(2)))) / 0.004958575   # +0.2%  z_displaced3 < 0.01783
        + 0.001647554 * (18.66842 * 0.5 * ((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) * (36.85684 * 0.5 * ((77.03428 - Q.sv_1_sd0_sum) / 36.85684) * (1 + math.erf(((77.03428 - Q.sv_1_sd0_sum) / 36.85684) / math.sqrt(2)))) / 2632.17   # +0.2%  mres_sd_mass_b0z005 < 159.9 and sv_1_sd0_sum < 77.03
        + 0.001633357 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) / math.sqrt(2)))) * (0.03559215 * 0.5 * ((0.6203012 - Q.tau32_b2) / 0.03559215) * (1 + math.erf(((0.6203012 - Q.tau32_b2) / 0.03559215) / math.sqrt(2)))) / 0.4505721   # +0.2%  mres_sd_mass_b0z005 > 178.9 and tau32_b2 < 0.6203
        - 0.00160738 * (8.307776 * 0.5 * ((Q.mass_top50 - 135.5368) / 8.307776) * (1 + math.erf(((Q.mass_top50 - 135.5368) / 8.307776) / math.sqrt(2)))) / 6.670994   # -0.2%  mass_top50 > 135.5
        - 0.001577624 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) * (0.02830505 * 0.5 * ((Q.dzerr_0 - 0.0) / 0.02830505) * (1 + math.erf(((Q.dzerr_0 - 0.0) / 0.02830505) / math.sqrt(2)))) / 0.07528029   # -0.2%  n_lund_kt_above_1 < 9 and dzerr_0 > 0
        + 0.001383013 * (0.08149619 * 0.5 * ((0.2477181 - Q.pt1_over_pt0) / 0.08149619) * (1 + math.erf(((0.2477181 - Q.pt1_over_pt0) / 0.08149619) / math.sqrt(2)))) / 0.01013904   # +0.1%  pt1_over_pt0 < 0.2477
        - 0.001364239 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (0.0001714083 * 0.5 * ((0.0002104854 - Q.C3_b2) / 0.0001714083) * (1 + math.erf(((0.0002104854 - Q.C3_b2) / 0.0001714083) / math.sqrt(2)))) / 2.1757e-06   # -0.1%  z_displaced3 < 0.1343 and C3_b2 < 0.0002105
        + 0.001302987 * (11.90664 * 0.5 * ((Q.sj3_pair_mass_min - 80.02563) / 11.90664) * (1 + math.erf(((Q.sj3_pair_mass_min - 80.02563) / 11.90664) / math.sqrt(2)))) / 0.8401129   # +0.1%  sj3_pair_mass_min > 80.03
        + 0.001216234 * (3.100743 * 0.5 * ((65.0404 - Q.mass_charged) / 3.100743) * (1 + math.erf(((65.0404 - Q.mass_charged) / 3.100743) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.isphoton_19) / 1.0) * (1 + math.erf(((1.0 - Q.isphoton_19) / 1.0) / math.sqrt(2)))) / 5.712548   # +0.1%  mass_charged < 65.04 and isphoton_19 < 1
        + 0.001143859 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.9414577 * 0.5 * ((3.157646 - Q.sip_3d_3) / 0.9414577) * (1 + math.erf(((3.157646 - Q.sip_3d_3) / 0.9414577) / math.sqrt(2)))) / 7.224904   # +0.1%  mres_sd_mass_b0z005 > 81.62 and sip_3d_3 < 3.158
        + 0.0010981 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2)))) * (1.751638 * 0.5 * ((5.460234 - Q.D2_b2) / 1.751638) * (1 + math.erf(((5.460234 - Q.D2_b2) / 1.751638) / math.sqrt(2)))) / 1.142363   # +0.1%  mass < 56.49 and D2_b2 < 5.46
        - 0.0009878109 * (3.100743 * 0.5 * ((65.0404 - Q.mass_charged) / 3.100743) * (1 + math.erf(((65.0404 - Q.mass_charged) / 3.100743) / math.sqrt(2)))) / 11.01486   # -0.1%  mass_charged < 65.04
        + 0.0009266237 * (0.005494283 * 0.5 * ((Q.pz_lnd1 - 0.05758556) / 0.005494283) * (1 + math.erf(((Q.pz_lnd1 - 0.05758556) / 0.005494283) / math.sqrt(2)))) / 0.02718738   # +0.1%  pz_lnd1 > 0.05759
        + 0.0009100705 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.0003494239 * 0.5 * ((Q.C3_b2 - 0.0007327206) / 0.0003494239) * (1 + math.erf(((Q.C3_b2 - 0.0007327206) / 0.0003494239) / math.sqrt(2)))) / 2.547669e-06   # +0.1%  e3_b2 < 0.0004127 and C3_b2 > 0.0007327
        - 0.0008899626 * (7.767937 * 0.5 * ((Q.sj3_mass1 - 44.1303) / 7.767937) * (1 + math.erf(((Q.sj3_mass1 - 44.1303) / 7.767937) / math.sqrt(2)))) / 0.750979   # -0.1%  sj3_mass1 > 44.13
        - 0.0008026207 * (1.0 * 0.5 * ((0.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((0.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) / 0.01335943   # -0.1%  lepsj_3_n_d3 < 0
        + 0.0007852757 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2)))) * (0.09869392 * 0.5 * ((Q.sjq_3_sumabs_k1 - 0.5890081) / 0.09869392) * (1 + math.erf(((Q.sjq_3_sumabs_k1 - 0.5890081) / 0.09869392) / math.sqrt(2)))) / 0.06091976   # +0.1%  lne_16 > 0.4469 and sjq_3_sumabs_k1 > 0.589
        + 0.0007015835 * (4.488781 * 0.5 * ((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) * (1 + math.erf(((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) / math.sqrt(2)))) * (0.1152886 * 0.5 * ((-0.3970734 - Q.dc_2_charge) / 0.1152886) * (1 + math.erf(((-0.3970734 - Q.dc_2_charge) / 0.1152886) / math.sqrt(2)))) / 1.61135   # +0.1%  mres_sd_mass_b0z005 < 125.9 and dc_2_charge < -0.3971
        + 0.0006711616 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 0.0) / 1.0) * (1 + math.erf(((Q.n_muon - 0.0) / 1.0) / math.sqrt(2)))) / 0.3030557   # +0.1%  lne_16 > 0.4469 and n_muon > 0
        - 0.0006172875 * (0.01266368 * 0.5 * ((0.05462126 - Q.pz_lnd2) / 0.01266368) * (1 + math.erf(((0.05462126 - Q.pz_lnd2) / 0.01266368) / math.sqrt(2)))) / 0.00873136   # -0.1%  pz_lnd2 < 0.05462
        + 0.0005854524 * (13.85751 * 0.5 * ((Q.sj2_mass1 - 91.2852) / 13.85751) * (1 + math.erf(((Q.sj2_mass1 - 91.2852) / 13.85751) / math.sqrt(2)))) / 1.19043   # +0.1%  sj2_mass1 > 91.29
        + 0.0005013535 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.0792551 * 0.5 * ((Q.lepsj_2_dr - 0.1822601) / 0.0792551) * (1 + math.erf(((Q.lepsj_2_dr - 0.1822601) / 0.0792551) / math.sqrt(2)))) / 0.003391776   # +0.1%  lep_z < 0.3397 and lepsj_2_dr > 0.1823
        + 0.0004817213 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) * (1.375431 * 0.5 * ((Q.mass_2photon - 4.18513) / 1.375431) * (1 + math.erf(((Q.mass_2photon - 4.18513) / 1.375431) / math.sqrt(2)))) / 4.211216   # +0.0%  n_pairs_kt_above_3 < 28 and mass_2photon > 4.185
        - 0.0004594507 * (0.05582565 * 0.5 * ((0.456416 - Q.tau32) / 0.05582565) * (1 + math.erf(((0.456416 - Q.tau32) / 0.05582565) / math.sqrt(2)))) / 0.01722171   # -0.0%  tau32 < 0.4564
        - 0.0004532459 * (0.0408915 * 0.5 * ((Q.lep_dr - 0.04607888) / 0.0408915) * (1 + math.erf(((Q.lep_dr - 0.04607888) / 0.0408915) / math.sqrt(2)))) * (0.03296266 * 0.5 * ((0.04394648 - Q.sv_1_dr) / 0.03296266) * (1 + math.erf(((0.04394648 - Q.sv_1_dr) / 0.03296266) / math.sqrt(2)))) / 0.0006851243   # -0.0%  lep_dr > 0.04608 and sv_1_dr < 0.04395
        + 0.0003759098 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) / 3.676114   # +0.0%  n_pairs_kt_above_3 < 28
        - 0.0002807119 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2)))) * (0.02417016 * 0.5 * ((Q.z_photon - 0.3318968) / 0.02417016) * (1 + math.erf(((Q.z_photon - 0.3318968) / 0.02417016) / math.sqrt(2)))) / 0.04023263   # -0.0%  lne_16 > 0.4469 and z_photon > 0.3319
        - 0.0002703587 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (4.565952 * 0.5 * ((Q.sj3_pair_mass_min - 48.40547) / 4.565952) * (1 + math.erf(((Q.sj3_pair_mass_min - 48.40547) / 4.565952) / math.sqrt(2)))) / 1.404019   # -0.0%  lep_z < 0.3397 and sj3_pair_mass_min > 48.41
        - 0.0001552467 * (5.345127 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) / math.sqrt(2)))) / 2.438634   # -0.0%  nca_sj4_pair_mass_2nd > 77.36
        - 0.0001254948 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.06842582 * 0.5 * ((Q.lne_1 - 4.635336) / 0.06842582) * (1 + math.erf(((Q.lne_1 - 4.635336) / 0.06842582) / math.sqrt(2)))) / 1.668933   # -0.0%  lep_ptrel < 12.15 and lne_1 > 4.635
        - 8.279526e-05 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) / math.sqrt(2)))) / 0.01794643   # -0.0%  sjq_3_3_k1 > 0.6536
        + 5.921403e-05 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.036273 * 0.5 * ((Q.td0_13 - -0.06278656) / 0.036273) * (1 + math.erf(((Q.td0_13 - -0.06278656) / 0.036273) / math.sqrt(2)))) / 2.705098   # +0.0%  mres_sd_mass_b0z005 > 81.62 and td0_13 > -0.06279
        - 5.586174e-05 * (0.02937692 * 0.5 * ((Q.sj3_pairmin_over_m - 0.4866692) / 0.02937692) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.4866692) / 0.02937692) / math.sqrt(2)))) / 0.003991075   # -0.0%  sj3_pairmin_over_m > 0.4867
        - 3.556727e-05 * (1.0 * 0.5 * ((2.0 - Q.sum_charge) / 1.0) * (1 + math.erf(((2.0 - Q.sum_charge) / 1.0) / math.sqrt(2)))) * (0.01620483 * 0.5 * ((Q.d0err_34 - 0.02580261) / 0.01620483) * (1 + math.erf(((Q.d0err_34 - 0.02580261) / 0.01620483) / math.sqrt(2)))) / 0.01271546   # -0.0%  sum_charge < 2 and d0err_34 > 0.0258
    )
    return z


def neuron_53(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.889358e-06
    )
    return z


def neuron_54(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.190658e-05
    )
    return z


def neuron_55(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.066937e-05
    )
    return z


def neuron_56(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.072324e-07
    )
    return z


def neuron_57(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.959646e-06
    )
    return z


def neuron_58(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.617147e-06
    )
    return z


def neuron_59(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.029329e-06
    )
    return z


def neuron_60(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.666754e-06
    )
    return z


def neuron_61(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.840554e-06
    )
    return z


def neuron_62(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.422957e-07
    )
    return z


def neuron_63(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.308504e-06
    )
    return z


def neuron_64(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.258623e-06
    )
    return z


def neuron_65(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.791053e-06
    )
    return z


def neuron_66(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.709773e-06
    )
    return z


def neuron_67(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.219507e-06
    )
    return z


def neuron_68(Q):
    # scale S = 12.04; each line: share * term / its average size
    z = 12.04156 * (0.1415438
        - 0.06908113 * (19.14168 * 0.5 * ((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2)))) / 55.1407   # -6.9%  mres_sd_mass_b2z01 < 158.2
        - 0.0545134 * (11.90664 * 0.5 * ((80.02563 - Q.sj3_pair_mass_min) / 11.90664) * (1 + math.erf(((80.02563 - Q.sj3_pair_mass_min) / 11.90664) / math.sqrt(2)))) / 44.05633   # -5.5%  sj3_pair_mass_min < 80.03
        + 0.04856352 * (18.57486 * 0.5 * ((Q.mres_pruned_mass - 41.77934) / 18.57486) * (1 + math.erf(((Q.mres_pruned_mass - 41.77934) / 18.57486) / math.sqrt(2)))) / 56.22948   # +4.9%  mres_pruned_mass > 41.78
        + 0.04219665 * (3.499256 * 0.5 * ((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) * (1 + math.erf(((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) / math.sqrt(2)))) / 24.89273   # +4.2%  mres_sd_mass_b2z01 < 121.6
        + 0.04197078 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) / 27.0933   # +4.2%  n_pt_above_1 < 65
        - 0.04118488 * (16.26099 * 0.5 * ((161.1264 - Q.mass_top50) / 16.26099) * (1 + math.erf(((161.1264 - Q.mass_top50) / 16.26099) / math.sqrt(2)))) / 50.49522   # -4.1%  mass_top50 < 161.1
        - 0.04096558 * (5.270719 * 0.5 * ((Q.mres_pruned_mass - 86.14266) / 5.270719) * (1 + math.erf(((Q.mres_pruned_mass - 86.14266) / 5.270719) / math.sqrt(2)))) / 22.57228   # -4.1%  mres_pruned_mass > 86.14
        + 0.03314666 * (12.47875 * 0.5 * ((Q.mres_pruned_mass - 131.2974) / 12.47875) * (1 + math.erf(((Q.mres_pruned_mass - 131.2974) / 12.47875) / math.sqrt(2)))) / 5.901305   # +3.3%  mres_pruned_mass > 131.3
        - 0.03196707 * (0.0138316 * 0.5 * ((0.2513409 - Q.pz_lnd2) / 0.0138316) * (1 + math.erf(((0.2513409 - Q.pz_lnd2) / 0.0138316) / math.sqrt(2)))) / 0.1298261   # -3.2%  pz_lnd2 < 0.2513
        - 0.0317503 * Q.C2 / 0.1492705   # -3.2%  C2
        - 0.02869077 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2261637   # -2.9%  lep_iso < 0.4382
        - 0.02834484 * (6.036668 * 0.5 * ((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) * (1 + math.erf(((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) / math.sqrt(2)))) / 5.409032   # -2.8%  mres_sd_mass_b2z01 < 76.15
        + 0.02780961 * (4.590131 * 0.5 * ((107.2089 - Q.mres_sd_mass_b2z01) / 4.590131) * (1 + math.erf(((107.2089 - Q.mres_sd_mass_b2z01) / 4.590131) / math.sqrt(2)))) / 16.34801   # +2.8%  mres_sd_mass_b2z01 < 107.2
        + 0.02767882 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) / 0.0006447212   # +2.8%  e3_b2 < 0.000791
        - 0.02517853 * (20.04228 * 0.5 * ((Q.mres_pruned_mass - 149.272) / 20.04228) * (1 + math.erf(((Q.mres_pruned_mass - 149.272) / 20.04228) / math.sqrt(2)))) / 3.864607   # -2.5%  mres_pruned_mass > 149.3
        + 0.02244621 * (0.03429006 * 0.5 * ((0.08178299 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.08178299 - Q.lep_dr) / 0.03429006) / math.sqrt(2)))) / 0.04862806   # +2.2%  lep_dr < 0.08178
        - 0.01793477 * (0.02630693 * 0.5 * ((0.2376554 - Q.pz_lnd3) / 0.02630693) * (1 + math.erf(((0.2376554 - Q.pz_lnd3) / 0.02630693) / math.sqrt(2)))) / 0.1443517   # -1.8%  pz_lnd3 < 0.2377
        - 0.01639131 * (0.2410611 * 0.5 * ((3.38061 - Q.pair_mean_lnm2) / 0.2410611) * (1 + math.erf(((3.38061 - Q.pair_mean_lnm2) / 0.2410611) / math.sqrt(2)))) / 0.9968028   # -1.6%  pair_mean_lnm2 < 3.381
        - 0.01562234 * (2.0 * 0.5 * ((11.0 - Q.n_sd0_above_2) / 2.0) * (1 + math.erf(((11.0 - Q.n_sd0_above_2) / 2.0) / math.sqrt(2)))) * (0.117671 * 0.5 * ((0.4191372 - Q.lep_dr) / 0.117671) * (1 + math.erf(((0.4191372 - Q.lep_dr) / 0.117671) / math.sqrt(2)))) / 2.36372   # -1.6%  n_sd0_above_2 < 11 and lep_dr < 0.4191
        + 0.0146953 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) / 15.65331   # +1.5%  mass < 117.5
        + 0.01469278 * (0.0216761 * 0.5 * ((0.820003 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.820003 - Q.tau32) / 0.0216761) / math.sqrt(2)))) / 0.163867   # +1.5%  tau32 < 0.82
        - 0.01417511 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.03798057 * 0.5 * ((Q.z_top30_slots - 0.8316085) / 0.03798057) * (1 + math.erf(((Q.z_top30_slots - 0.8316085) / 0.03798057) / math.sqrt(2)))) / 3.710454e-05   # -1.4%  e3_b2 < 0.0004127 and z_top30_slots > 0.8316
        - 0.01352911 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.983043) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.983043) / 4.308625) / math.sqrt(2)))) / 5.404766   # -1.4%  lep_ptrel > 6.983
        - 0.01322853 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) / 0.0002870086   # -1.3%  e3_b2 < 0.0004127
        - 0.0131309 * (0.298047 * 0.5 * ((2.286291 - Q.D2_b2) / 0.298047) * (1 + math.erf(((2.286291 - Q.D2_b2) / 0.298047) / math.sqrt(2)))) / 0.7235156   # -1.3%  D2_b2 < 2.286
        - 0.01303886 * (0.1659628 * 0.5 * ((1.465946 - Q.jet_abs_eta) / 0.1659628) * (1 + math.erf(((1.465946 - Q.jet_abs_eta) / 0.1659628) / math.sqrt(2)))) / 0.7440077   # -1.3%  jet_abs_eta < 1.466
        - 0.01287089 * (0.02099671 * 0.5 * ((0.08304558 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.08304558 - Q.z_displaced3) / 0.02099671) / math.sqrt(2)))) / 0.0372965   # -1.3%  z_displaced3 < 0.08305
        + 0.01254667 * (0.2410611 * 0.5 * ((3.38061 - Q.pair_mean_lnm2) / 0.2410611) * (1 + math.erf(((3.38061 - Q.pair_mean_lnm2) / 0.2410611) / math.sqrt(2)))) * (0.06955833 * 0.5 * ((0.3788785 - Q.z_neutral_had) / 0.06955833) * (1 + math.erf(((0.3788785 - Q.z_neutral_had) / 0.06955833) / math.sqrt(2)))) / 0.2478187   # +1.3%  pair_mean_lnm2 < 3.381 and z_neutral_had < 0.3789
        + 0.01225751 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2)))) / 18.07823   # +1.2%  n_pt_above_1 < 65 and ak02_2_n_lep < 1
        - 0.01121175 * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 20.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 20.0) / 1.0) / math.sqrt(2)))) / 2.533146   # -1.1%  n_charged_pt_above_1 > 20
        + 0.01098223 * (19.93276 * 0.5 * ((55.13212 - Q.mres_sd_mass_b2z01) / 19.93276) * (1 + math.erf(((55.13212 - Q.mres_sd_mass_b2z01) / 19.93276) / math.sqrt(2)))) / 2.995945   # +1.1%  mres_sd_mass_b2z01 < 55.13
        - 0.01081224 * (2.973269 * 0.5 * ((Q.sj4_pair_mass_max - 72.862) / 2.973269) * (1 + math.erf(((Q.sj4_pair_mass_max - 72.862) / 2.973269) / math.sqrt(2)))) / 15.05222   # -1.1%  sj4_pair_mass_max > 72.86
        + 0.01056125 * (2.0 * 0.5 * ((11.0 - Q.n_sd0_above_2) / 2.0) * (1 + math.erf(((11.0 - Q.n_sd0_above_2) / 2.0) / math.sqrt(2)))) / 6.877544   # +1.1%  n_sd0_above_2 < 11
        + 0.009591286 * (7.585662 * 0.5 * ((Q.lep_ptrel - 18.7678) / 7.585662) * (1 + math.erf(((Q.lep_ptrel - 18.7678) / 7.585662) / math.sqrt(2)))) / 2.977753   # +1.0%  lep_ptrel > 18.77
        + 0.009017309 * (0.01332924 * 0.5 * ((0.9062492 - Q.tau43) / 0.01332924) * (1 + math.erf(((0.9062492 - Q.tau43) / 0.01332924) / math.sqrt(2)))) / 0.1126426   # +0.9%  tau43 < 0.9062
        + 0.008841628 * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.265564) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.265564) / 0.0649922) / math.sqrt(2)))) / 0.2492585   # +0.9%  z_charged_had > 0.2656
        - 0.00750911 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2)))) * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) / 6.973082   # -0.8%  mass < 90.09 and lepsj_3_maxsd0 < 2.413
        - 0.00724025 * (0.0218375 * 0.5 * ((0.07650476 - Q.ak02_3_z) / 0.0218375) * (1 + math.erf(((0.07650476 - Q.ak02_3_z) / 0.0218375) / math.sqrt(2)))) * (1032.513 * 0.5 * ((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) * (1 + math.erf(((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) / math.sqrt(2)))) / 47.58341   # -0.7%  ak02_3_z < 0.0765 and lepsj_2_maxsd0 < 1164
        + 0.006287597 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.n_muon) / 1.0) * (1 + math.erf(((1.0 - Q.n_muon) / 1.0) / math.sqrt(2)))) / 17.39762   # +0.6%  n_pt_above_1 < 65 and n_muon < 1
        + 0.006168663 * (1.369872 * 0.5 * ((Q.sj3_mass1 - 17.31003) / 1.369872) * (1 + math.erf(((Q.sj3_mass1 - 17.31003) / 1.369872) / math.sqrt(2)))) / 6.012784   # +0.6%  sj3_mass1 > 17.31
        + 0.005844686 * (0.007115881 * 0.5 * ((Q.sjf_4_1_z_d3 - 0.00625791) / 0.007115881) * (1 + math.erf(((Q.sjf_4_1_z_d3 - 0.00625791) / 0.007115881) / math.sqrt(2)))) / 0.064594   # +0.6%  sjf_4_1_z_d3 > 0.006258
        + 0.005619807 * (0.0138316 * 0.5 * ((0.2513409 - Q.pz_lnd2) / 0.0138316) * (1 + math.erf(((0.2513409 - Q.pz_lnd2) / 0.0138316) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2)))) / 18.29233   # +0.6%  pz_lnd2 < 0.2513 and jd_3d_4 < 172.9
        + 0.005443125 * (0.0216761 * 0.5 * ((0.820003 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.820003 - Q.tau32) / 0.0216761) / math.sqrt(2)))) * (11.90664 * 0.5 * ((80.02563 - Q.sj3_pair_mass_min) / 11.90664) * (1 + math.erf(((80.02563 - Q.sj3_pair_mass_min) / 11.90664) / math.sqrt(2)))) / 5.398574   # +0.5%  tau32 < 0.82 and sj3_pair_mass_min < 80.03
        - 0.004619125 * (5.270719 * 0.5 * ((Q.mres_pruned_mass - 86.14266) / 5.270719) * (1 + math.erf(((Q.mres_pruned_mass - 86.14266) / 5.270719) / math.sqrt(2)))) * (0.02798345 * 0.5 * ((0.171574 - Q.dc_3_z) / 0.02798345) * (1 + math.erf(((0.171574 - Q.dc_3_z) / 0.02798345) / math.sqrt(2)))) / 2.004849   # -0.5%  mres_pruned_mass > 86.14 and dc_3_z < 0.1716
        + 0.004551735 * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.265564) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.265564) / 0.0649922) / math.sqrt(2)))) * (0.01039253 * 0.5 * ((0.9216755 - Q.tau54) / 0.01039253) * (1 + math.erf(((0.9216755 - Q.tau54) / 0.01039253) / math.sqrt(2)))) / 0.01824663   # +0.5%  z_charged_had > 0.2656 and tau54 < 0.9217
        + 0.004438104 * (22.10989 * 0.5 * ((Q.mres_pruned_mass - 171.3819) / 22.10989) * (1 + math.erf(((Q.mres_pruned_mass - 171.3819) / 22.10989) / math.sqrt(2)))) / 1.992102   # +0.4%  mres_pruned_mass > 171.4
        + 0.004404252 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2)))) / 4.97365   # +0.4%  mass < 90.09
        + 0.004250345 * (0.1783594 * 0.5 * ((Q.ktd_ln_d34 - -9.180886) / 0.1783594) * (1 + math.erf(((Q.ktd_ln_d34 - -9.180886) / 0.1783594) / math.sqrt(2)))) / 0.470454   # +0.4%  ktd_ln_d34 > -9.181
        + 0.003641104 * (0.00254714 * 0.5 * ((0.0391767 - Q.M3) / 0.00254714) * (1 + math.erf(((0.0391767 - Q.M3) / 0.00254714) / math.sqrt(2)))) / 0.008438771   # +0.4%  M3 < 0.03918
        + 0.003463054 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) * (0.3667324 * 0.5 * ((2.613544 - Q.D2_b2) / 0.3667324) * (1 + math.erf(((2.613544 - Q.D2_b2) / 0.3667324) / math.sqrt(2)))) / 10.09948   # +0.3%  mass < 117.5 and D2_b2 < 2.614
        + 0.003341166 * (0.0138316 * 0.5 * ((0.2513409 - Q.pz_lnd2) / 0.0138316) * (1 + math.erf(((0.2513409 - Q.pz_lnd2) / 0.0138316) / math.sqrt(2)))) * (11.72898 * 0.5 * ((Q.lnptrel_25 - -18.42068) / 11.72898) * (1 + math.erf(((Q.lnptrel_25 - -18.42068) / 11.72898) / math.sqrt(2)))) / 1.158805   # +0.3%  pz_lnd2 < 0.2513 and lnptrel_25 > -18.42
        - 0.003297572 * (11.90664 * 0.5 * ((80.02563 - Q.sj3_pair_mass_min) / 11.90664) * (1 + math.erf(((80.02563 - Q.sj3_pair_mass_min) / 11.90664) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2)))) / 30.51937   # -0.3%  sj3_pair_mass_min < 80.03 and dc_2_n_lep < 1
        - 0.00309146 * (1.0 * 0.5 * ((Q.sdb_2_n - 11.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 11.0) / 1.0) / math.sqrt(2)))) * (1.739103 * 0.5 * ((3.969624 - Q.jd_3d_4) / 1.739103) * (1 + math.erf(((3.969624 - Q.jd_3d_4) / 1.739103) / math.sqrt(2)))) / 1.779546   # -0.3%  sdb_2_n > 11 and jd_3d_4 < 3.97
        + 0.002965571 * (0.05861841 * 0.5 * ((Q.ak02_dr23 - 0.3728877) / 0.05861841) * (1 + math.erf(((Q.ak02_dr23 - 0.3728877) / 0.05861841) / math.sqrt(2)))) / 0.07180454   # +0.3%  ak02_dr23 > 0.3729
        - 0.00278695 * (0.02099671 * 0.5 * ((0.08304558 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.08304558 - Q.z_displaced3) / 0.02099671) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.n_muon) / 1.0) * (1 + math.erf(((0.0 - Q.n_muon) / 1.0) / math.sqrt(2)))) / 0.0007870856   # -0.3%  z_displaced3 < 0.08305 and n_muon < 0
        - 0.002687912 * (1.5 * 0.5 * ((Q.n_dr_0p2_0p4 - 7.0) / 1.5) * (1 + math.erf(((Q.n_dr_0p2_0p4 - 7.0) / 1.5) / math.sqrt(2)))) / 5.939596   # -0.3%  n_dr_0p2_0p4 > 7
        + 0.002677582 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) / 0.0005925947   # +0.3%  e3_b2 < 0.0004127 and n_s3d_above_3 > 2
        + 0.002620289 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 5.193195   # +0.3%  n_pairs_kt_above_1 < 80
        + 0.002609221 * (0.001201988 * 0.5 * ((0.002792418 - Q.sum_z_dr2_top3) / 0.001201988) * (1 + math.erf(((0.002792418 - Q.sum_z_dr2_top3) / 0.001201988) / math.sqrt(2)))) / 0.0002325086   # +0.3%  sum_z_dr2_top3 < 0.002792
        - 0.002557493 * (5.345127 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) / math.sqrt(2)))) / 2.438634   # -0.3%  nca_sj4_pair_mass_2nd > 77.36
        + 0.002443469 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.983043) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.983043) / 4.308625) / math.sqrt(2)))) * (0.0531045 * 0.5 * ((0.221369 - Q.C2_b2) / 0.0531045) * (1 + math.erf(((0.221369 - Q.C2_b2) / 0.0531045) / math.sqrt(2)))) / 0.8215827   # +0.2%  lep_ptrel > 6.983 and C2_b2 < 0.2214
        - 0.002370944 * (1.0 * 0.5 * ((3.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((3.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) / 0.2451407   # -0.2%  sjq_2_2_nch < 3
        + 0.002358941 * (11.17405 * 0.5 * ((Q.sj3_pair_mass_max - 128.6605) / 11.17405) * (1 + math.erf(((Q.sj3_pair_mass_max - 128.6605) / 11.17405) / math.sqrt(2)))) / 2.60253   # +0.2%  sj3_pair_mass_max > 128.7
        + 0.002358288 * (0.2297361 * 0.5 * ((0.0 - Q.ak02_dr13) / 0.2297361) * (1 + math.erf(((0.0 - Q.ak02_dr13) / 0.2297361) / math.sqrt(2)))) / 0.008715326   # +0.2%  ak02_dr13 < 0
        + 0.00233511 * (10.93201 * 0.5 * ((Q.mass_top10 - 27.42853) / 10.93201) * (1 + math.erf(((Q.mass_top10 - 27.42853) / 10.93201) / math.sqrt(2)))) / 40.50349   # +0.2%  mass_top10 > 27.43
        + 0.002300197 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) / 0.9643417   # +0.2%  lep_ptrel > 43.21
        + 0.002170266 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 2.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 2.0) / 1.0) / math.sqrt(2)))) / 0.0001051181   # +0.2%  e3_b2 < 0.0004127 and n_lund_kt_above_5 > 2
        + 0.002101001 * (0.0218375 * 0.5 * ((0.07650476 - Q.ak02_3_z) / 0.0218375) * (1 + math.erf(((0.07650476 - Q.ak02_3_z) / 0.0218375) / math.sqrt(2)))) / 0.05013548   # +0.2%  ak02_3_z < 0.0765
        - 0.002070333 * (10.8986 * 0.5 * ((67.20576 - Q.mass_top30) / 10.8986) * (1 + math.erf(((67.20576 - Q.mass_top30) / 10.8986) / math.sqrt(2)))) / 1.991323   # -0.2%  mass_top30 < 67.21
        - 0.00205648 * (1.0 * 0.5 * ((Q.n_lund_kt_above_1 - 4.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_1 - 4.0) / 1.0) / math.sqrt(2)))) / 1.55976   # -0.2%  n_lund_kt_above_1 > 4
        + 0.001869946 * (1.0 * 0.5 * ((Q.sdb_2_n - 11.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 11.0) / 1.0) / math.sqrt(2)))) / 1.924272   # +0.2%  sdb_2_n > 11
        + 0.001717708 * (1.504711 * 0.5 * ((Q.sj3_mass2 - 15.2797) / 1.504711) * (1 + math.erf(((Q.sj3_mass2 - 15.2797) / 1.504711) / math.sqrt(2)))) / 1.854725   # +0.2%  sj3_mass2 > 15.28
        - 0.001699026 * (0.03746729 * 0.5 * ((Q.z_displaced3 - 0.1681173) / 0.03746729) * (1 + math.erf(((Q.z_displaced3 - 0.1681173) / 0.03746729) / math.sqrt(2)))) / 0.0386898   # -0.2%  z_displaced3 > 0.1681
        + 0.001594802 * (0.1750793 * 0.5 * ((0.4436035 - Q.max_abs_dz) / 0.1750793) * (1 + math.erf(((0.4436035 - Q.max_abs_dz) / 0.1750793) / math.sqrt(2)))) / 0.0781359   # +0.2%  max_abs_dz < 0.4436
        - 0.001584105 * (0.02719315 * 0.5 * ((0.1530389 - Q.sdb_2_z) / 0.02719315) * (1 + math.erf(((0.1530389 - Q.sdb_2_z) / 0.02719315) / math.sqrt(2)))) / 0.01287382   # -0.2%  sdb_2_z < 0.153
        - 0.001532655 * (0.02593915 * 0.5 * ((Q.mres_sd_rg_b1z01 - 0.4673199) / 0.02593915) * (1 + math.erf(((Q.mres_sd_rg_b1z01 - 0.4673199) / 0.02593915) / math.sqrt(2)))) / 0.02595905   # -0.2%  mres_sd_rg_b1z01 > 0.4673
        - 0.001319613 * (0.02434011 * 0.5 * ((Q.sj3_pairmin_over_m - 0.4610803) / 0.02434011) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.4610803) / 0.02434011) / math.sqrt(2)))) / 0.007054082   # -0.1%  sj3_pairmin_over_m > 0.4611
        - 0.001303864 * (0.2410611 * 0.5 * ((3.38061 - Q.pair_mean_lnm2) / 0.2410611) * (1 + math.erf(((3.38061 - Q.pair_mean_lnm2) / 0.2410611) / math.sqrt(2)))) * (6.12936 * 0.5 * ((Q.lnptrel_31 - -6.668647) / 6.12936) * (1 + math.erf(((Q.lnptrel_31 - -6.668647) / 6.12936) / math.sqrt(2)))) / 0.6844746   # -0.1%  pair_mean_lnm2 < 3.381 and lnptrel_31 > -6.669
        + 0.001301222 * (0.0216761 * 0.5 * ((0.820003 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.820003 - Q.tau32) / 0.0216761) / math.sqrt(2)))) * (0.01950041 * 0.5 * ((Q.sv_1_dr - 0.1227033) / 0.01950041) * (1 + math.erf(((Q.sv_1_dr - 0.1227033) / 0.01950041) / math.sqrt(2)))) / 0.01462745   # +0.1%  tau32 < 0.82 and sv_1_dr > 0.1227
        - 0.001161627 * (1.0 * 0.5 * ((Q.sdb_2_n - 11.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 11.0) / 1.0) / math.sqrt(2)))) * (0.01645773 * 0.5 * ((0.5838314 - Q.nca_sj4_pair2nd_over_mass) / 0.01645773) * (1 + math.erf(((0.5838314 - Q.nca_sj4_pair2nd_over_mass) / 0.01645773) / math.sqrt(2)))) / 0.1653239   # -0.1%  sdb_2_n > 11 and nca_sj4_pair2nd_over_mass < 0.5838
        + 0.001128371 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (0.1600921 * 0.5 * ((Q.mass_displaced5 - 6.5555e-08) / 0.1600921) * (1 + math.erf(((Q.mass_displaced5 - 6.5555e-08) / 0.1600921) / math.sqrt(2)))) / 149.2472   # +0.1%  n_pt_above_1 < 65 and mass_displaced5 > 6.555e-08
        - 0.0009377315 * (0.00254714 * 0.5 * ((0.0391767 - Q.M3) / 0.00254714) * (1 + math.erf(((0.0391767 - Q.M3) / 0.00254714) / math.sqrt(2)))) * (0.06042313 * 0.5 * ((Q.sjq_3_sumabs_k1 - 0.4372817) / 0.06042313) * (1 + math.erf(((Q.sjq_3_sumabs_k1 - 0.4372817) / 0.06042313) / math.sqrt(2)))) / 0.0006284477   # -0.1%  M3 < 0.03918 and sjq_3_sumabs_k1 > 0.4373
        - 0.0009282339 * (10.67504 * 0.5 * ((Q.mass_top10 - 103.4976) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 103.4976) / 10.67504) / math.sqrt(2)))) / 2.014523   # -0.1%  mass_top10 > 103.5
        - 0.0009226552 * (0.07842787 * 0.5 * ((Q.dc_split2_dr - 0.3591078) / 0.07842787) * (1 + math.erf(((Q.dc_split2_dr - 0.3591078) / 0.07842787) / math.sqrt(2)))) / 0.005063019   # -0.1%  dc_split2_dr > 0.3591
        + 0.0008569937 * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 20.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 20.0) / 1.0) / math.sqrt(2)))) * (1.5 * 0.5 * ((17.0 - Q.n_dr_0p1_0p2) / 1.5) * (1 + math.erf(((17.0 - Q.n_dr_0p1_0p2) / 1.5) / math.sqrt(2)))) / 7.346064   # +0.1%  n_charged_pt_above_1 > 20 and n_dr_0p1_0p2 < 17
        - 0.0007862394 * (1.0 * 0.5 * ((Q.n_neutral_had - 6.0) / 1.0) * (1 + math.erf(((Q.n_neutral_had - 6.0) / 1.0) / math.sqrt(2)))) / 0.3921539   # -0.1%  n_neutral_had > 6
        + 0.000725634 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.ak02_1_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_1_n_lep) / 1.0) / math.sqrt(2)))) / 16.31506   # +0.1%  n_pt_above_1 < 65 and ak02_1_n_lep < 1
        - 0.0005556663 * (0.08347678 * 0.5 * ((Q.tau21_b2 - 0.5868564) / 0.08347678) * (1 + math.erf(((Q.tau21_b2 - 0.5868564) / 0.08347678) / math.sqrt(2)))) / 0.01353943   # -0.1%  tau21_b2 > 0.5869
        - 0.0005513576 * (3.274853e-06 * 0.5 * ((1.008706e-05 - Q.e3_b2) / 3.274853e-06) * (1 + math.erf(((1.008706e-05 - Q.e3_b2) / 3.274853e-06) / math.sqrt(2)))) / 1.48515e-06   # -0.1%  e3_b2 < 1.009e-05
        - 0.0004761576 * (0.179073 * 0.5 * ((Q.lep_z - 0.5187302) / 0.179073) * (1 + math.erf(((Q.lep_z - 0.5187302) / 0.179073) / math.sqrt(2)))) / 0.01004697   # -0.0%  lep_z > 0.5187
        + 0.0003335461 * (0.188475 * 0.5 * ((1.634536 - Q.dc_tag_max) / 0.188475) * (1 + math.erf(((1.634536 - Q.dc_tag_max) / 0.188475) / math.sqrt(2)))) / 0.4736477   # +0.0%  dc_tag_max < 1.635
        - 0.0002783746 * (0.007115881 * 0.5 * ((Q.sjf_4_1_z_d3 - 0.00625791) / 0.007115881) * (1 + math.erf(((Q.sjf_4_1_z_d3 - 0.00625791) / 0.007115881) / math.sqrt(2)))) * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) / 17.41223   # -0.0%  sjf_4_1_z_d3 > 0.006258 and sip_3d_2 < 447.1
        + 0.0002489766 * (12.47875 * 0.5 * ((Q.mres_pruned_mass - 131.2974) / 12.47875) * (1 + math.erf(((Q.mres_pruned_mass - 131.2974) / 12.47875) / math.sqrt(2)))) * (0.002064355 * 0.5 * ((0.0 - Q.lepsj_3_dr) / 0.002064355) * (1 + math.erf(((0.0 - Q.lepsj_3_dr) / 0.002064355) / math.sqrt(2)))) / 6.782104e-05   # +0.0%  mres_pruned_mass > 131.3 and lepsj_3_dr < 0
        - 0.0002084942 * (0.298047 * 0.5 * ((2.286291 - Q.D2_b2) / 0.298047) * (1 + math.erf(((2.286291 - Q.D2_b2) / 0.298047) / math.sqrt(2)))) * (0.003370318 * 0.5 * ((Q.sjf_4_4_z_d3 - 0.0) / 0.003370318) * (1 + math.erf(((Q.sjf_4_4_z_d3 - 0.0) / 0.003370318) / math.sqrt(2)))) / 0.003860902   # -0.0%  D2_b2 < 2.286 and sjf_4_4_z_d3 > 0
        + 0.0001866772 * (29.74772 * 0.5 * ((Q.mres_sd_mass_b0z02 - 163.95) / 29.74772) * (1 + math.erf(((Q.mres_sd_mass_b0z02 - 163.95) / 29.74772) / math.sqrt(2)))) / 2.635068   # +0.0%  mres_sd_mass_b0z02 > 164
        - 0.0001609814 * (3.499256 * 0.5 * ((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) * (1 + math.erf(((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2)))) / 17.97101   # -0.0%  mres_sd_mass_b2z01 < 121.6 and dc_2_n_lep < 1
        - 0.0001604154 * (19.14168 * 0.5 * ((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2)))) * (0.002556514 * 0.5 * ((Q.sjf_4_3_z_d3 - 0.001692006) / 0.002556514) * (1 + math.erf(((Q.sjf_4_3_z_d3 - 0.001692006) / 0.002556514) / math.sqrt(2)))) / 0.5377318   # -0.0%  mres_sd_mass_b2z01 < 158.2 and sjf_4_3_z_d3 > 0.001692
        + 0.0001272601 * (0.03746729 * 0.5 * ((Q.z_displaced3 - 0.1681173) / 0.03746729) * (1 + math.erf(((Q.z_displaced3 - 0.1681173) / 0.03746729) / math.sqrt(2)))) * (6.000774 * 0.5 * ((14.85973 - Q.jd_3d_4) / 6.000774) * (1 + math.erf(((14.85973 - Q.jd_3d_4) / 6.000774) / math.sqrt(2)))) / 0.1042357   # +0.0%  z_displaced3 > 0.1681 and jd_3d_4 < 14.86
        + 3.995305e-05 * (0.07842787 * 0.5 * ((Q.dc_split2_dr - 0.3591078) / 0.07842787) * (1 + math.erf(((Q.dc_split2_dr - 0.3591078) / 0.07842787) / math.sqrt(2)))) * (0.2807389 * 0.5 * ((Q.lund3_lnkt - 4.319803) / 0.2807389) * (1 + math.erf(((Q.lund3_lnkt - 4.319803) / 0.2807389) / math.sqrt(2)))) / 0.0001469637   # +0.0%  dc_split2_dr > 0.3591 and lund3_lnkt > 4.32
    )
    return z


def neuron_69(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.92614e-06
    )
    return z


def neuron_70(Q):
    # scale S = 4.89; each line: share * term / its average size
    z = 4.889726 * (0.002847913
        - 0.06015029 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (22.16789 * 0.5 * ((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) * (1 + math.erf(((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) / math.sqrt(2)))) / 1.151755   # -6.0%  lep_z < 0.2214 and lepsj_3_maxsd0 < 11.53
        + 0.03957851 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) / 1.651857   # +4.0%  lepsj_3_n_d3 < 2
        - 0.0390636 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) / 0.8420682   # -3.9%  lep_iso < 1.362
        - 0.03816092 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) / 14.13297   # -3.8%  lep_ptrel < 18.77
        + 0.03687418 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (16.47782 * 0.5 * ((162.7874 - Q.mass_top30) / 16.47782) * (1 + math.erf(((162.7874 - Q.mass_top30) / 16.47782) / math.sqrt(2)))) / 16.21778   # +3.7%  lep_z < 0.3397 and mass_top30 < 162.8
        - 0.03510339 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) / 0.04874886   # -3.5%  lepsj_2_dr < 0.06573
        - 0.03437818 * (0.0134569 * 0.5 * ((Q.C2 - 0.05691773) / 0.0134569) * (1 + math.erf(((Q.C2 - 0.05691773) / 0.0134569) / math.sqrt(2)))) / 0.09283547   # -3.4%  C2 > 0.05692
        + 0.03245336 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2261637   # +3.2%  lep_iso < 0.4382
        + 0.02820304 * (0.009408518 * 0.5 * ((0.1122946 - Q.pz_lnd0) / 0.009408518) * (1 + math.erf(((0.1122946 - Q.pz_lnd0) / 0.009408518) / math.sqrt(2)))) / 0.02301262   # +2.8%  pz_lnd0 < 0.1123
        + 0.02599758 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (22.16789 * 0.5 * ((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) * (1 + math.erf(((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) / math.sqrt(2)))) / 1.835745   # +2.6%  lep_z < 0.3397 and lepsj_3_maxsd0 < 11.53
        + 0.0241185 * (0.0134569 * 0.5 * ((Q.C2 - 0.05691773) / 0.0134569) * (1 + math.erf(((Q.C2 - 0.05691773) / 0.0134569) / math.sqrt(2)))) * (0.0329306 * 0.5 * ((0.4091922 - Q.ak02_2_z) / 0.0329306) * (1 + math.erf(((0.4091922 - Q.ak02_2_z) / 0.0329306) / math.sqrt(2)))) / 0.01970912   # +2.4%  C2 > 0.05692 and ak02_2_z < 0.4092
        - 0.0240326 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (540.2081 * 0.5 * ((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2)))) / 3080.945   # -2.4%  n_s3d_above_3 < 10 and lepsj_3_maxsd0 < 587
        + 0.02363012 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.0330605 * 0.5 * ((0.114659 - Q.lep_dr) / 0.0330605) * (1 + math.erf(((0.114659 - Q.lep_dr) / 0.0330605) / math.sqrt(2)))) / 0.1349844   # +2.4%  lepsj_3_n_d3 < 2 and lep_dr < 0.1147
        + 0.02164951 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) / 0.1673441   # +2.2%  lep_z < 0.2214
        + 0.02127267 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) / 31.48666   # +2.1%  mres_sd_mass_b0z005 > 81.62
        - 0.02080119 * (1.0 * 0.5 * ((18.0 - Q.n_neutral) / 1.0) * (1 + math.erf(((18.0 - Q.n_neutral) / 1.0) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 7.414209   # -2.1%  n_neutral < 18 and jd_3d_6 < 5.368
        + 0.0200246 * (1.0 * 0.5 * ((18.0 - Q.n_neutral) / 1.0) * (1 + math.erf(((18.0 - Q.n_neutral) / 1.0) / math.sqrt(2)))) / 2.453083   # +2.0%  n_neutral < 18
        + 0.01941431 * (0.8047829 * 0.5 * ((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) * (1 + math.erf(((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) / math.sqrt(2)))) / 1.042224   # +1.9%  lepsj_3_maxsd0 < 1.53
        - 0.0192655 * (3.563419 * 0.5 * ((Q.mres_sd_mass_b0z005 - 122.1) / 3.563419) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 122.1) / 3.563419) / math.sqrt(2)))) / 9.428572   # -1.9%  mres_sd_mass_b0z005 > 122.1
        + 0.01663032 * (3.764443 * 0.5 * ((Q.mass_top50 - 125.4927) / 3.764443) * (1 + math.erf(((Q.mass_top50 - 125.4927) / 3.764443) / math.sqrt(2)))) / 9.005445   # +1.7%  mass_top50 > 125.5
        - 0.01657928 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) / 297.4519   # -1.7%  sip_3d_2 < 447.1
        + 0.01550557 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.387908 * 0.5 * ((3.151933 - Q.jd_3d_6) / 1.387908) * (1 + math.erf(((3.151933 - Q.jd_3d_6) / 1.387908) / math.sqrt(2)))) / 326.1951   # +1.6%  sip_3d_2 < 447.1 and jd_3d_6 < 3.152
        - 0.01400564 * (3.974361 * 0.5 * ((Q.sj3_pair_mass_min - 44.1643) / 3.974361) * (1 + math.erf(((Q.sj3_pair_mass_min - 44.1643) / 3.974361) / math.sqrt(2)))) / 6.024343   # -1.4%  sj3_pair_mass_min > 44.16
        + 0.01326404 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) / 0.05253524   # +1.3%  z_displaced5 > 0.0809
        + 0.01209552 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2)))) / 0.0537686   # +1.2%  n_lepton < 0
        - 0.01205245 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) * (0.01372201 * 0.5 * ((Q.pz_lnkt1 - 0.03633353) / 0.01372201) * (1 + math.erf(((Q.pz_lnkt1 - 0.03633353) / 0.01372201) / math.sqrt(2)))) / 0.003240112   # -1.2%  z_displaced5 > 0.0809 and pz_lnkt1 > 0.03633
        - 0.01176278 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.02942179 * 0.5 * ((Q.tau21 - 0.2377332) / 0.02942179) * (1 + math.erf(((Q.tau21 - 0.2377332) / 0.02942179) / math.sqrt(2)))) / 5.62464e-05   # -1.2%  e3_b2 < 0.0004127 and tau21 > 0.2377
        + 0.01169379 * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2)))) / 0.01976846   # +1.2%  sjq_3_2_k1 < -0.6015
        + 0.01091866 * (1.0 * 0.5 * ((2.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((2.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 0.4679696   # +1.1%  n_s3d_above_3 < 2
        - 0.01011841 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.04737588 * 0.5 * ((0.1807551 - Q.ak02_3_z) / 0.04737588) * (1 + math.erf(((0.1807551 - Q.ak02_3_z) / 0.04737588) / math.sqrt(2)))) / 3.366431   # -1.0%  mres_sd_mass_b0z005 > 81.62 and ak02_3_z < 0.1808
        + 0.0100085 * (0.004103638 * 0.5 * ((Q.M2 - 0.06228948) / 0.004103638) * (1 + math.erf(((Q.M2 - 0.06228948) / 0.004103638) / math.sqrt(2)))) / 0.0202169   # +1.0%  M2 > 0.06229
        - 0.009991765 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 1.0) / 1.0) / math.sqrt(2)))) / 0.2216248   # -1.0%  lep_z < 0.3397 and sjf_2_2_n_d3 > 1
        + 0.009702107 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) / math.sqrt(2)))) / 3.38788   # +1.0%  mres_sd_mass_b0z005 > 159.9
        + 0.00947213 * (1.0 * 0.5 * ((Q.sjf_4_n2disp - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_4_n2disp - 1.0) / 1.0) / math.sqrt(2)))) / 0.2811624   # +0.9%  sjf_4_n2disp > 1
        - 0.009434645 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2)))) / 0.001827643   # -0.9%  lep_z < 0.004136
        + 0.009382551 * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 1.0) / 1.0) / math.sqrt(2)))) / 0.5054993   # +0.9%  lepsj_2_n_d3 > 1
        - 0.008894233 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.01417841 * 0.5 * ((0.1049877 - Q.C2_b2) / 0.01417841) * (1 + math.erf(((0.1049877 - Q.C2_b2) / 0.01417841) / math.sqrt(2)))) / 0.007787697   # -0.9%  lep_z < 0.2214 and C2_b2 < 0.105
        - 0.008622135 * (0.009408518 * 0.5 * ((0.1122946 - Q.pz_lnd0) / 0.009408518) * (1 + math.erf(((0.1122946 - Q.pz_lnd0) / 0.009408518) / math.sqrt(2)))) * (0.01282622 * 0.5 * ((Q.z_neutral_had - 0.1010123) / 0.01282622) * (1 + math.erf(((Q.z_neutral_had - 0.1010123) / 0.01282622) / math.sqrt(2)))) / 0.002028153   # -0.9%  pz_lnd0 < 0.1123 and z_neutral_had > 0.101
        - 0.008412077 * (0.02343699 * 0.5 * ((0.2277628 - Q.sdb_2_z) / 0.02343699) * (1 + math.erf(((0.2277628 - Q.sdb_2_z) / 0.02343699) / math.sqrt(2)))) / 0.0333137   # -0.8%  sdb_2_z < 0.2278
        + 0.007598484 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.06755743 * 0.5 * ((0.5017201 - Q.ak02_dr23) / 0.06755743) * (1 + math.erf(((0.5017201 - Q.ak02_dr23) / 0.06755743) / math.sqrt(2)))) / 5.979818   # +0.8%  mres_sd_mass_b0z005 > 81.62 and ak02_dr23 < 0.5017
        - 0.007487231 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) / 24.48148   # -0.7%  n_s3d_above_3 < 10 and n_lund_kt_above_1 < 9
        + 0.007299313 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.03538975 * 0.5 * ((0.5871757 - Q.tau32) / 0.03538975) * (1 + math.erf(((0.5871757 - Q.tau32) / 0.03538975) / math.sqrt(2)))) / 0.01038461   # +0.7%  lep_z < 0.3397 and tau32 < 0.5872
        - 0.006835184 * (19.33798 * 0.5 * ((Q.mres_sd_mass_b2z01 - 177.5398) / 19.33798) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 177.5398) / 19.33798) / math.sqrt(2)))) / 1.926528   # -0.7%  mres_sd_mass_b2z01 > 177.5
        - 0.006620822 * (1.5 * 0.5 * ((Q.n_charged_pt_above_1 - 22.0) / 1.5) * (1 + math.erf(((Q.n_charged_pt_above_1 - 22.0) / 1.5) / math.sqrt(2)))) / 1.827424   # -0.7%  n_charged_pt_above_1 > 22
        - 0.006360836 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2)))) / 0.9830025   # -0.6%  lep_iso < 1.362 and sip_3d_3 < 4.606
        - 0.006356164 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2)))) / 0.03447537   # -0.6%  lepsj_3_n_d3 < 2 and sjq_3_2_k1 < -0.6015
        + 0.006106451 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) / 0.2702483   # +0.6%  lep_z < 0.3397
        + 0.006072431 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (11.4891 * 0.5 * ((32.61679 - Q.mass_displaced5) / 11.4891) * (1 + math.erf(((32.61679 - Q.mass_displaced5) / 11.4891) / math.sqrt(2)))) / 0.007834152   # +0.6%  e3_b2 < 0.0004127 and mass_displaced5 < 32.62
        - 0.00585686 * (2.44598e-06 * 0.5 * ((1.874473e-05 - Q.ecf_g42) / 2.44598e-06) * (1 + math.erf(((1.874473e-05 - Q.ecf_g42) / 2.44598e-06) / math.sqrt(2)))) / 5.212439e-06   # -0.6%  ecf_g42 < 1.874e-05
        + 0.005703268 * (0.01961104 * 0.5 * ((0.3273298 - Q.sj3_dr12) / 0.01961104) * (1 + math.erf(((0.3273298 - Q.sj3_dr12) / 0.01961104) / math.sqrt(2)))) / 0.07554041   # +0.6%  sj3_dr12 < 0.3273
        + 0.005618659 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) / 0.0002870086   # +0.6%  e3_b2 < 0.0004127
        - 0.00554519 * (1.0 * 0.5 * ((Q.n_electron - 1.0) / 1.0) * (1 + math.erf(((Q.n_electron - 1.0) / 1.0) / math.sqrt(2)))) / 0.1872867   # -0.6%  n_electron > 1
        + 0.00549838 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) / math.sqrt(2)))) / 0.01794643   # +0.5%  sjq_3_3_k1 > 0.6536
        - 0.005431928 * (3.329534 * 0.5 * ((Q.mass_2charged - 14.28253) / 3.329534) * (1 + math.erf(((Q.mass_2charged - 14.28253) / 3.329534) / math.sqrt(2)))) / 5.701311   # -0.5%  mass_2charged > 14.28
        + 0.005260936 * (0.2398529 * 0.5 * ((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) * (1 + math.erf(((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) / math.sqrt(2)))) / 0.01973624   # +0.5%  sjq_3_2_k1 > 0.6149
        + 0.005229827 * (3.710857 * 0.5 * ((Q.mass_top30 - 91.4679) / 3.710857) * (1 + math.erf(((Q.mass_top30 - 91.4679) / 3.710857) / math.sqrt(2)))) / 19.75551   # +0.5%  mass_top30 > 91.47
        + 0.004905844 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 6.361002   # +0.5%  n_s3d_above_3 < 10
        + 0.004695594 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.01565591 * 0.5 * ((0.1681753 - Q.pz_lnkt1) / 0.01565591) * (1 + math.erf(((0.1681753 - Q.pz_lnkt1) / 0.01565591) / math.sqrt(2)))) / 0.8631618   # +0.5%  lep_ptrel < 18.77 and pz_lnkt1 < 0.1682
        - 0.004660743 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) / math.sqrt(2)))) / 0.03063719   # -0.5%  lep_iso < 1.362 and sjq_3_3_k1 > 0.4037
        - 0.004275399 * (0.01743442 * 0.5 * ((0.3281884 - Q.kt2_dr12) / 0.01743442) * (1 + math.erf(((0.3281884 - Q.kt2_dr12) / 0.01743442) / math.sqrt(2)))) / 0.03862065   # -0.4%  kt2_dr12 < 0.3282
        + 0.00419296 * (8.963785 * 0.5 * ((22.56857 - Q.mass_charged) / 8.963785) * (1 + math.erf(((22.56857 - Q.mass_charged) / 8.963785) / math.sqrt(2)))) / 0.5104165   # +0.4%  mass_charged < 22.57
        - 0.004065077 * (0.560057 * 0.5 * ((Q.mass_displaced5 - 1.777787) / 0.560057) * (1 + math.erf(((Q.mass_displaced5 - 1.777787) / 0.560057) / math.sqrt(2)))) / 5.31072   # -0.4%  mass_displaced5 > 1.778
        - 0.00387983 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2)))) / 1.091313   # -0.4%  mass < 56.49
        - 0.003841593 * (1.0 * 0.5 * ((0.0 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.0 - Q.sv_2_n) / 1.0) / math.sqrt(2)))) / 0.04355367   # -0.4%  sv_2_n < 0
        + 0.003823071 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) / math.sqrt(2)))) / 1.914224   # +0.4%  mres_sd_mass_b0z005 > 178.9
        + 0.003657619 * (0.179073 * 0.5 * ((Q.lep_z - 0.5187302) / 0.179073) * (1 + math.erf(((Q.lep_z - 0.5187302) / 0.179073) / math.sqrt(2)))) * (0.7363527 * 0.5 * ((0.7363527 - Q.dc_4_jp) / 0.7363527) * (1 + math.erf(((0.7363527 - Q.dc_4_jp) / 0.7363527) / math.sqrt(2)))) / 0.006127839   # +0.4%  lep_z > 0.5187 and dc_4_jp < 0.7364
        - 0.003528069 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) * (0.1097229 * 0.5 * ((-2.222088 - Q.lnptrel_3) / 0.1097229) * (1 + math.erf(((-2.222088 - Q.lnptrel_3) / 0.1097229) / math.sqrt(2)))) / 0.02236909   # -0.4%  z_displaced5 > 0.0809 and lnptrel_3 < -2.222
        + 0.003384235 * (0.0134569 * 0.5 * ((Q.C2 - 0.05691773) / 0.0134569) * (1 + math.erf(((Q.C2 - 0.05691773) / 0.0134569) / math.sqrt(2)))) * (0.005748204 * 0.5 * ((0.0548897 - Q.pz_lnkt3) / 0.005748204) * (1 + math.erf(((0.0548897 - Q.pz_lnkt3) / 0.005748204) / math.sqrt(2)))) / 0.001597553   # +0.3%  C2 > 0.05692 and pz_lnkt3 < 0.05489
        + 0.003378492 * (0.2508028 * 0.5 * ((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2)))) / 0.01780014   # +0.3%  sjq_3_3_k1 < -0.655
        - 0.003346827 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2)))) / 2.198028   # -0.3%  lep_ptrel > 27.32
        + 0.003275963 * (2.077045 * 0.5 * ((1.851735 - Q.sj2_mass2) / 2.077045) * (1 + math.erf(((1.851735 - Q.sj2_mass2) / 2.077045) / math.sqrt(2)))) / 0.08951668   # +0.3%  sj2_mass2 < 1.852
        + 0.003136445 * (11.49171 * 0.5 * ((71.96396 - Q.mass) / 11.49171) * (1 + math.erf(((71.96396 - Q.mass) / 11.49171) / math.sqrt(2)))) / 2.139873   # +0.3%  mass < 71.96
        - 0.002954817 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (1.387908 * 0.5 * ((3.151933 - Q.jd_3d_6) / 1.387908) * (1 + math.erf(((3.151933 - Q.jd_3d_6) / 1.387908) / math.sqrt(2)))) / 12.95323   # -0.3%  lep_ptrel < 18.77 and jd_3d_6 < 3.152
        - 0.002949002 * (16.26099 * 0.5 * ((Q.mass_top50 - 161.1264) / 16.26099) * (1 + math.erf(((Q.mass_top50 - 161.1264) / 16.26099) / math.sqrt(2)))) / 3.008129   # -0.3%  mass_top50 > 161.1
        - 0.002812637 * (0.009408518 * 0.5 * ((0.1122946 - Q.pz_lnd0) / 0.009408518) * (1 + math.erf(((0.1122946 - Q.pz_lnd0) / 0.009408518) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2)))) / 3.172196   # -0.3%  pz_lnd0 < 0.1123 and jd_3d_4 < 172.9
        - 0.002697379 * (14.05251 * 0.5 * ((Q.mres_sd_mass_b2z01 - 139.2565) / 14.05251) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 139.2565) / 14.05251) / math.sqrt(2)))) / 5.947884   # -0.3%  mres_sd_mass_b2z01 > 139.3
        - 0.002240127 * (0.04964267 * 0.5 * ((Q.sj4_dr_min - 0.249115) / 0.04964267) * (1 + math.erf(((Q.sj4_dr_min - 0.249115) / 0.04964267) / math.sqrt(2)))) / 0.008951805   # -0.2%  sj4_dr_min > 0.2491
        + 0.002183064 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2)))) / 20.4272   # +0.2%  mres_sd_mass_b0z005 > 81.62 and sip_3d_3 < 4.606
        - 0.002106596 * (5.143623 * 0.5 * ((Q.mres_sd_mass_b1z01 - 88.79082) / 5.143623) * (1 + math.erf(((Q.mres_sd_mass_b1z01 - 88.79082) / 5.143623) / math.sqrt(2)))) / 24.90775   # -0.2%  mres_sd_mass_b1z01 > 88.79
        + 0.002102861 * (1.0 * 0.5 * ((Q.sjq_3_2_nch - 5.0) / 1.0) * (1 + math.erf(((Q.sjq_3_2_nch - 5.0) / 1.0) / math.sqrt(2)))) / 2.171173   # +0.2%  sjq_3_2_nch > 5
        + 0.002089903 * (5.345127 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) / math.sqrt(2)))) / 2.438634   # +0.2%  nca_sj4_pair_mass_2nd > 77.36
        + 0.001826354 * (0.01743442 * 0.5 * ((0.3281884 - Q.kt2_dr12) / 0.01743442) * (1 + math.erf(((0.3281884 - Q.kt2_dr12) / 0.01743442) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.000784222   # +0.2%  kt2_dr12 < 0.3282 and dc_2_n_lep < 0
        - 0.001708451 * (2.210526e-06 * 0.5 * ((4.696862e-06 - Q.e3_b2) / 2.210526e-06) * (1 + math.erf(((4.696862e-06 - Q.e3_b2) / 2.210526e-06) / math.sqrt(2)))) / 3.939189e-07   # -0.2%  e3_b2 < 4.697e-06
        + 0.001701925 * (0.560057 * 0.5 * ((Q.mass_displaced5 - 1.777787) / 0.560057) * (1 + math.erf(((Q.mass_displaced5 - 1.777787) / 0.560057) / math.sqrt(2)))) * (5.37501 * 0.5 * ((9.356714 - Q.jd_3d_6) / 5.37501) * (1 + math.erf(((9.356714 - Q.jd_3d_6) / 5.37501) / math.sqrt(2)))) / 10.41514   # +0.2%  mass_displaced5 > 1.778 and jd_3d_6 < 9.357
        + 0.00169532 * (0.003553237 * 0.5 * ((0.01143413 - Q.tau4) / 0.003553237) * (1 + math.erf(((0.01143413 - Q.tau4) / 0.003553237) / math.sqrt(2)))) / 0.0001826239   # +0.2%  tau4 < 0.01143
        + 0.001658709 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.751638 * 0.5 * ((Q.D2_b2 - 5.460234) / 1.751638) * (1 + math.erf(((Q.D2_b2 - 5.460234) / 1.751638) / math.sqrt(2)))) / 0.5630631   # +0.2%  lep_z < 0.3397 and D2_b2 > 5.46
        + 0.001557111 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2)))) * (0.04060641 * 0.5 * ((0.1322296 - Q.dr_39) / 0.04060641) * (1 + math.erf(((0.1322296 - Q.dr_39) / 0.04060641) / math.sqrt(2)))) / 0.2308704   # +0.2%  lep_ptrel > 27.32 and dr_39 < 0.1322
        + 0.001454461 * (0.0795439 * 0.5 * ((Q.ak02_dr23 - 0.5718354) / 0.0795439) * (1 + math.erf(((Q.ak02_dr23 - 0.5718354) / 0.0795439) / math.sqrt(2)))) / 0.02767855   # +0.1%  ak02_dr23 > 0.5718
        - 0.001162273 * (0.179073 * 0.5 * ((Q.lep_z - 0.5187302) / 0.179073) * (1 + math.erf(((Q.lep_z - 0.5187302) / 0.179073) / math.sqrt(2)))) / 0.01004697   # -0.1%  lep_z > 0.5187
        - 0.001041914 * (3.710857 * 0.5 * ((Q.mass_top30 - 91.4679) / 3.710857) * (1 + math.erf(((Q.mass_top30 - 91.4679) / 3.710857) / math.sqrt(2)))) * (0.007817624 * 0.5 * ((Q.sjf_4_3_z_d3 - 0.01507439) / 0.007817624) * (1 + math.erf(((Q.sjf_4_3_z_d3 - 0.01507439) / 0.007817624) / math.sqrt(2)))) / 0.2161443   # -0.1%  mass_top30 > 91.47 and sjf_4_3_z_d3 > 0.01507
        - 0.0008550536 * (2.0 * 0.5 * ((8.0 - Q.n_charged_pt_above_1) / 2.0) * (1 + math.erf(((8.0 - Q.n_charged_pt_above_1) / 2.0) / math.sqrt(2)))) / 0.1037746   # -0.1%  n_charged_pt_above_1 < 8
        + 0.0007616969 * (0.117671 * 0.5 * ((Q.lep_dr - 0.4191372) / 0.117671) * (1 + math.erf(((Q.lep_dr - 0.4191372) / 0.117671) / math.sqrt(2)))) / 0.009357382   # +0.1%  lep_dr > 0.4191
        + 0.0007275836 * (11.41538 * 0.5 * ((Q.mres_sd_mass_b0z005 - 69.14897) / 11.41538) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 69.14897) / 11.41538) / math.sqrt(2)))) / 41.33029   # +0.1%  mres_sd_mass_b0z005 > 69.15
        - 0.0006818396 * (0.02505229 * 0.5 * ((Q.sdb_2_z - 0.3448514) / 0.02505229) * (1 + math.erf(((Q.sdb_2_z - 0.3448514) / 0.02505229) / math.sqrt(2)))) / 0.06277194   # -0.1%  sdb_2_z > 0.3449
        + 0.0005256257 * (0.5664529 * 0.5 * ((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) * (1 + math.erf(((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) / math.sqrt(2)))) / 0.1032402   # +0.1%  pair_mean_lnm2 > 4.069
        + 0.0004192833 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) * (0.02203204 * 0.5 * ((Q.dr_3 - 0.1990664) / 0.02203204) * (1 + math.erf(((Q.dr_3 - 0.1990664) / 0.02203204) / math.sqrt(2)))) / 0.001268327   # +0.0%  z_displaced5 > 0.0809 and dr_3 > 0.1991
        - 0.0003245991 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.04373169 * 0.5 * ((Q.eta_14 - 0.1429443) / 0.04373169) * (1 + math.erf(((Q.eta_14 - 0.1429443) / 0.04373169) / math.sqrt(2)))) / 0.005215056   # -0.0%  lep_z < 0.3397 and eta_14 > 0.1429
        - 5.190923e-05 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) * (0.05310694 * 0.5 * ((0.4358177 - Q.pt1_over_pt0) / 0.05310694) * (1 + math.erf(((0.4358177 - Q.pt1_over_pt0) / 0.05310694) / math.sqrt(2)))) / 0.001200902   # -0.0%  z_displaced5 > 0.0809 and pt1_over_pt0 < 0.4358
        + 4.331466e-05 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2)))) * (0.2399741 * 0.5 * ((-4.934509 - Q.lund3_lnz) / 0.2399741) * (1 + math.erf(((-4.934509 - Q.lund3_lnz) / 0.2399741) / math.sqrt(2)))) / 4.571564   # +0.0%  lep_ptrel > 27.32 and lund3_lnz < -4.935
        - 1.779322e-05 * (11.90664 * 0.5 * ((Q.sj3_pair_mass_min - 80.02563) / 11.90664) * (1 + math.erf(((Q.sj3_pair_mass_min - 80.02563) / 11.90664) / math.sqrt(2)))) / 0.8401129   # -0.0%  sj3_pair_mass_min > 80.03
    )
    return z


def neuron_71(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.029707e-06
    )
    return z


def neuron_72(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.164118e-06
    )
    return z


def neuron_73(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.699639e-05
    )
    return z


def neuron_74(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.493394e-06
    )
    return z


def neuron_75(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.973839e-06
    )
    return z


def neuron_76(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.542785e-06
    )
    return z


def neuron_77(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.087487e-06
    )
    return z


def neuron_78(Q):
    # scale S = 4.625; each line: share * term / its average size
    z = 4.625364 * (-0.01866388
        - 0.2178925 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) / 14.13297   # -21.8%  lep_ptrel < 18.77
        - 0.06276965 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) / 0.2702483   # -6.3%  lep_z < 0.3397
        + 0.06134159 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) / 36.20948   # +6.1%  lep_ptrel < 43.21
        + 0.05137477 * (1.0 * 0.5 * ((6.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((6.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 2.857   # +5.1%  n_s3d_above_3 < 6
        + 0.0354854 * (2.54381e-06 * 0.5 * ((3.53193e-06 - Q.e4) / 2.54381e-06) * (1 + math.erf(((3.53193e-06 - Q.e4) / 2.54381e-06) / math.sqrt(2)))) / 2.457263e-06   # +3.5%  e4 < 3.532e-06
        + 0.03032508 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.3051516 * 0.5 * ((-7.886954 - Q.ktd_ln_d34) / 0.3051516) * (1 + math.erf(((-7.886954 - Q.ktd_ln_d34) / 0.3051516) / math.sqrt(2)))) / 20.68074   # +3.0%  lep_ptrel < 18.77 and ktd_ln_d34 < -7.887
        + 0.02801987 * (0.009441413 * 0.5 * ((0.4854269 - Q.N2_b05) / 0.009441413) * (1 + math.erf(((0.4854269 - Q.N2_b05) / 0.009441413) / math.sqrt(2)))) / 0.0577829   # +2.8%  N2_b05 < 0.4854
        + 0.02785708 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) * (0.2429919 * 0.5 * ((6.304449 - Q.lne_0) / 0.2429919) * (1 + math.erf(((6.304449 - Q.lne_0) / 0.2429919) / math.sqrt(2)))) / 0.08666831   # +2.8%  lepsj_2_dr < 0.103 and lne_0 < 6.304
        - 0.02643947 * (0.2260337 * 0.5 * ((-8.400697 - Q.ktd_ln_d34) / 0.2260337) * (1 + math.erf(((-8.400697 - Q.ktd_ln_d34) / 0.2260337) / math.sqrt(2)))) / 1.139062   # -2.6%  ktd_ln_d34 < -8.401
        + 0.02495901 * (11.5 * 0.5 * ((99.0 - Q.n_pairs_kt_above_3) / 11.5) * (1 + math.erf(((99.0 - Q.n_pairs_kt_above_3) / 11.5) / math.sqrt(2)))) / 39.18157   # +2.5%  n_pairs_kt_above_3 < 99
        - 0.02265833 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 6.361002   # -2.3%  n_s3d_above_3 < 10
        + 0.02148139 * (4.488781 * 0.5 * ((Q.mres_sd_mass_b0z005 - 125.8718) / 4.488781) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 125.8718) / 4.488781) / math.sqrt(2)))) / 8.417972   # +2.1%  mres_sd_mass_b0z005 > 125.9
        + 0.02131696 * (0.3893833 * 0.5 * ((Q.lund3_lndelta - -2.817283) / 0.3893833) * (1 + math.erf(((Q.lund3_lndelta - -2.817283) / 0.3893833) / math.sqrt(2)))) / 1.319379   # +2.1%  lund3_lndelta > -2.817
        - 0.0201744 * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2)))) / 5.641924   # -2.0%  max_abs_d0 < 10.52
        + 0.01977726 * (0.04874922 * 0.5 * ((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) * (1 + math.erf(((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) / math.sqrt(2)))) / 0.08923496   # +2.0%  sjf_4_1_z_d3 < 0.1277
        - 0.01796891 * (540.2081 * 0.5 * ((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2)))) / 470.5515   # -1.8%  lepsj_3_maxsd0 < 587
        + 0.01704609 * (27.0 * 0.5 * ((195.0 - Q.n_pairs_kt_above_1) / 27.0) * (1 + math.erf(((195.0 - Q.n_pairs_kt_above_1) / 27.0) / math.sqrt(2)))) / 37.29428   # +1.7%  n_pairs_kt_above_1 < 195
        + 0.01580989 * (2.0 * 0.5 * ((10.0 - Q.n_sdz_above_2) / 2.0) * (1 + math.erf(((10.0 - Q.n_sdz_above_2) / 2.0) / math.sqrt(2)))) * (0.06215671 * 0.5 * ((0.3112717 - Q.sj4_dr_min) / 0.06215671) * (1 + math.erf(((0.3112717 - Q.sj4_dr_min) / 0.06215671) / math.sqrt(2)))) / 1.228448   # +1.6%  n_sdz_above_2 < 10 and sj4_dr_min < 0.3113
        - 0.01359694 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) / math.sqrt(2)))) / 3.38788   # -1.4%  mres_sd_mass_b0z005 > 159.9
        + 0.01318063 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.05549181 * 0.5 * ((0.1570831 - Q.sjf_4_2_z_d3) / 0.05549181) * (1 + math.erf(((0.1570831 - Q.sjf_4_2_z_d3) / 0.05549181) / math.sqrt(2)))) / 1.898221   # +1.3%  lep_ptrel < 18.77 and sjf_4_2_z_d3 < 0.1571
        + 0.01292916 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) / 1.651857   # +1.3%  lepsj_3_n_d3 < 2
        + 0.01219165 * (3.02693 * 0.5 * ((Q.sj4_pair_mass_max - 78.76797) / 3.02693) * (1 + math.erf(((Q.sj4_pair_mass_max - 78.76797) / 3.02693) / math.sqrt(2)))) / 11.81894   # +1.2%  sj4_pair_mass_max > 78.77
        - 0.01091902 * (2.0 * 0.5 * ((10.0 - Q.n_sdz_above_2) / 2.0) * (1 + math.erf(((10.0 - Q.n_sdz_above_2) / 2.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2)))) / 4.462522   # -1.1%  n_sdz_above_2 < 10 and dc_2_n_lep < 1
        + 0.01089147 * (1.5 * 0.5 * ((Q.n_charged_had - 23.0) / 1.5) * (1 + math.erf(((Q.n_charged_had - 23.0) / 1.5) / math.sqrt(2)))) / 1.522546   # +1.1%  n_charged_had > 23
        - 0.01081164 * (1.0 * 0.5 * ((10.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((10.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2)))) / 1.81815   # -1.1%  sdb_2_n < 10
        - 0.01046027 * (5.197 * 0.5 * ((95.14961 - Q.mass) / 5.197) * (1 + math.erf(((95.14961 - Q.mass) / 5.197) / math.sqrt(2)))) / 6.369586   # -1.0%  mass < 95.15
        - 0.01022764 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) / 1.234147   # -1.0%  n_s3d_above_3 > 4
        + 0.009331169 * (164.9991 * 0.5 * ((226.3008 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((226.3008 - Q.sip_3d_2) / 164.9991) / math.sqrt(2)))) / 133.7832   # +0.9%  sip_3d_2 < 226.3
        + 0.009211082 * (0.560057 * 0.5 * ((1.777787 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.777787 - Q.mass_displaced5) / 0.560057) / math.sqrt(2)))) / 0.8185445   # +0.9%  mass_displaced5 < 1.778
        + 0.008562269 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2)))) / 1983.284   # +0.9%  lep_ptrel < 18.77 and jd_3d_4 < 172.9
        + 0.007519352 * (2.0 * 0.5 * ((10.0 - Q.n_sdz_above_2) / 2.0) * (1 + math.erf(((10.0 - Q.n_sdz_above_2) / 2.0) / math.sqrt(2)))) / 6.236875   # +0.8%  n_sdz_above_2 < 10
        + 0.00732269 * (1.5 * 0.5 * ((22.0 - Q.n_neutral) / 1.5) * (1 + math.erf(((22.0 - Q.n_neutral) / 1.5) / math.sqrt(2)))) / 4.588734   # +0.7%  n_neutral < 22
        + 0.007176012 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (3.260893 * 0.5 * ((Q.mass_top20 - 86.78877) / 3.260893) * (1 + math.erf(((Q.mass_top20 - 86.78877) / 3.260893) / math.sqrt(2)))) / 3.624198   # +0.7%  lep_z < 0.3397 and mass_top20 > 86.79
        + 0.007165195 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (405.0217 * 0.5 * ((803.1736 - Q.jd_sum_abs_sd0_top5) / 405.0217) * (1 + math.erf(((803.1736 - Q.jd_sum_abs_sd0_top5) / 405.0217) / math.sqrt(2)))) / 317.7424   # +0.7%  n_s3d_above_3 > 4 and jd_sum_abs_sd0_top5 < 803.2
        + 0.007140703 * (18.98174 * 0.5 * ((14.38858 - Q.lep_iso) / 18.98174) * (1 + math.erf(((14.38858 - Q.lep_iso) / 18.98174) / math.sqrt(2)))) / 9.394448   # +0.7%  lep_iso < 14.39
        + 0.00706002 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (16.61023 * 0.5 * ((607.8256 - Q.sum_pt_top30) / 16.61023) * (1 + math.erf(((607.8256 - Q.sum_pt_top30) / 16.61023) / math.sqrt(2)))) / 726.0065   # +0.7%  lep_ptrel < 18.77 and sum_pt_top30 < 607.8
        - 0.007011318 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2)))) / 0.0537686   # -0.7%  n_lepton < 0
        - 0.006093094 * (0.06660296 * 0.5 * ((-2.115543 - Q.pair_mean_lndelta) / 0.06660296) * (1 + math.erf(((-2.115543 - Q.pair_mean_lndelta) / 0.06660296) / math.sqrt(2)))) / 0.2852413   # -0.6%  pair_mean_lndelta < -2.116
        - 0.005741764 * (3.793098 * 0.5 * ((Q.mass_displaced5 - 9.380468) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 9.380468) / 3.793098) / math.sqrt(2)))) * (14.29796 * 0.5 * ((113.5019 - Q.mass_charged) / 14.29796) * (1 + math.erf(((113.5019 - Q.mass_charged) / 14.29796) / math.sqrt(2)))) / 116.4344   # -0.6%  mass_displaced5 > 9.38 and mass_charged < 113.5
        + 0.005376794 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) / 0.8420682   # +0.5%  lep_iso < 1.362
        - 0.005210217 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) / 0.07858689   # -0.5%  lepsj_2_dr < 0.103
        + 0.004790214 * (17.59858 * 0.5 * ((Q.mass_top50 - 178.725) / 17.59858) * (1 + math.erf(((Q.mass_top50 - 178.725) / 17.59858) / math.sqrt(2)))) / 1.705883   # +0.5%  mass_top50 > 178.7
        + 0.004651144 * (0.03825143 * 0.5 * ((0.06059953 - Q.sjq_2_prod_k03) / 0.03825143) * (1 + math.erf(((0.06059953 - Q.sjq_2_prod_k03) / 0.03825143) / math.sqrt(2)))) / 0.2567996   # +0.5%  sjq_2_prod_k03 < 0.0606
        + 0.004494755 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) * (0.001358032 * 0.5 * ((0.03179932 - Q.dzerr_0) / 0.001358032) * (1 + math.erf(((0.03179932 - Q.dzerr_0) / 0.001358032) / math.sqrt(2)))) / 0.001087592   # +0.4%  lepsj_2_dr < 0.103 and dzerr_0 < 0.0318
        + 0.00444087 * (3.793098 * 0.5 * ((Q.mass_displaced5 - 9.380468) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 9.380468) / 3.793098) / math.sqrt(2)))) / 3.261523   # +0.4%  mass_displaced5 > 9.38
        + 0.004417567 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.04808774 * 0.5 * ((0.09790963 - Q.lepsj_3_dr) / 0.04808774) * (1 + math.erf(((0.09790963 - Q.lepsj_3_dr) / 0.04808774) / math.sqrt(2)))) / 0.1456636   # +0.4%  lepsj_3_n_d3 < 2 and lepsj_3_dr < 0.09791
        - 0.004245483 * (0.04148383 * 0.5 * ((-0.4705779 - Q.lund1_lndelta) / 0.04148383) * (1 + math.erf(((-0.4705779 - Q.lund1_lndelta) / 0.04148383) / math.sqrt(2)))) / 0.1763665   # -0.4%  lund1_lndelta < -0.4706
        + 0.004028117 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (1.0 * 0.5 * ((4.0 - Q.ak02_n) / 1.0) * (1 + math.erf(((4.0 - Q.ak02_n) / 1.0) / math.sqrt(2)))) / 18.67207   # +0.4%  lep_ptrel < 18.77 and ak02_n < 4
        - 0.004012921 * (0.001268491 * 0.5 * ((0.0142807 - Q.M3_b2) / 0.001268491) * (1 + math.erf(((0.0142807 - Q.M3_b2) / 0.001268491) / math.sqrt(2)))) / 0.004045429   # -0.4%  M3_b2 < 0.01428
        - 0.003992692 * (16.90428 * 0.5 * ((Q.mass - 164.4374) / 16.90428) * (1 + math.erf(((Q.mass - 164.4374) / 16.90428) / math.sqrt(2)))) / 3.240687   # -0.4%  mass > 164.4
        + 0.003138291 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.01443111 * 0.5 * ((0.1956014 - Q.pz_lnd2) / 0.01443111) * (1 + math.erf(((0.1956014 - Q.pz_lnd2) / 0.01443111) / math.sqrt(2)))) / 0.02270136   # +0.3%  lep_z < 0.3397 and pz_lnd2 < 0.1956
        - 0.003088336 * (3.373048 * 0.5 * ((Q.mass_top15 - 83.68425) / 3.373048) * (1 + math.erf(((Q.mass_top15 - 83.68425) / 3.373048) / math.sqrt(2)))) / 10.76171   # -0.3%  mass_top15 > 83.68
        - 0.002970478 * (3.357558 * 0.5 * ((Q.max_pair_mass - 31.44492) / 3.357558) * (1 + math.erf(((Q.max_pair_mass - 31.44492) / 3.357558) / math.sqrt(2)))) / 3.652923   # -0.3%  max_pair_mass > 31.44
        - 0.002805884 * (10.09897 * 0.5 * ((Q.sj4_pair_mass_max - 115.5142) / 10.09897) * (1 + math.erf(((Q.sj4_pair_mass_max - 115.5142) / 10.09897) / math.sqrt(2)))) / 2.222395   # -0.3%  sj4_pair_mass_max > 115.5
        - 0.002785533 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (0.02818185 * 0.5 * ((Q.sj3_pairmin_over_m - 0.1383534) / 0.02818185) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.1383534) / 0.02818185) / math.sqrt(2)))) / 0.2106931   # -0.3%  n_s3d_above_3 > 4 and sj3_pairmin_over_m > 0.1384
        - 0.002601479 * (11.28806 * 0.5 * ((70.88236 - Q.mass_top40) / 11.28806) * (1 + math.erf(((70.88236 - Q.mass_top40) / 11.28806) / math.sqrt(2)))) / 2.101479   # -0.3%  mass_top40 < 70.88
        + 0.002414932 * (10.67504 * 0.5 * ((Q.mass_top10 - 103.4976) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 103.4976) / 10.67504) / math.sqrt(2)))) / 2.014523   # +0.2%  mass_top10 > 103.5
        - 0.0020946 * (1.0 * 0.5 * ((Q.n_sd0_above_5 - 5.0) / 1.0) * (1 + math.erf(((Q.n_sd0_above_5 - 5.0) / 1.0) / math.sqrt(2)))) / 0.4848347   # -0.2%  n_sd0_above_5 > 5
        + 0.002062813 * (2.0 * 0.5 * ((Q.sjf_2_1_n_d3 - 7.0) / 2.0) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 7.0) / 2.0) / math.sqrt(2)))) / 0.110583   # +0.2%  sjf_2_1_n_d3 > 7
        + 0.001586289 * (12.30925 * 0.5 * ((Q.mass_displaced3 - 39.09615) / 12.30925) * (1 + math.erf(((Q.mass_displaced3 - 39.09615) / 12.30925) / math.sqrt(2)))) / 0.8447299   # +0.2%  mass_displaced3 > 39.1
        + 0.001261086 * (3.793098 * 0.5 * ((Q.mass_displaced5 - 9.380468) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 9.380468) / 3.793098) / math.sqrt(2)))) * (6.000774 * 0.5 * ((14.85973 - Q.jd_3d_4) / 6.000774) * (1 + math.erf(((14.85973 - Q.jd_3d_4) / 6.000774) / math.sqrt(2)))) / 4.574776   # +0.1%  mass_displaced5 > 9.38 and jd_3d_4 < 14.86
        - 0.001173206 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (98.63368 * 0.5 * ((Q.jet_e - 1134.686) / 98.63368) * (1 + math.erf(((Q.jet_e - 1134.686) / 98.63368) / math.sqrt(2)))) / 2709.635   # -0.1%  lep_ptrel < 43.21 and jet_e > 1135
        + 0.0008887158 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) / math.sqrt(2)))) / 0.01794643   # +0.1%  sjq_3_3_k1 > 0.6536
        + 0.0008605858 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_electron - 1.0) / 1.0) * (1 + math.erf(((Q.n_electron - 1.0) / 1.0) / math.sqrt(2)))) / 2.628079   # +0.1%  lep_ptrel < 18.77 and n_electron > 1
        - 0.0007765327 * (0.06585074 * 0.5 * ((-0.1170754 - Q.tdz_1) / 0.06585074) * (1 + math.erf(((-0.1170754 - Q.tdz_1) / 0.06585074) / math.sqrt(2)))) / 0.01660446   # -0.1%  tdz_1 < -0.1171
        + 0.0006408114 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2)))) / 0.001136453   # +0.1%  lepsj_2_dr < 0.103 and ak02_3_n_lep > 0
        + 0.0005635989 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (0.05461387 * 0.5 * ((0.2184442 - Q.z_displaced5) / 0.05461387) * (1 + math.erf(((0.2184442 - Q.z_displaced5) / 0.05461387) / math.sqrt(2)))) / 0.06539754   # +0.1%  n_s3d_above_3 > 4 and z_displaced5 < 0.2184
        - 0.000488685 * (0.0033606 * 0.5 * ((Q.M2_b05 - 0.1409711) / 0.0033606) * (1 + math.erf(((Q.M2_b05 - 0.1409711) / 0.0033606) / math.sqrt(2)))) / 0.009183911   # -0.0%  M2_b05 > 0.141
        - 0.0004020992 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_4_3_n_d3 - 0.0) / 1.0) * (1 + math.erf(((Q.sjf_4_3_n_d3 - 0.0) / 1.0) / math.sqrt(2)))) / 0.4486658   # -0.0%  lep_iso < 1.362 and sjf_4_3_n_d3 > 0
        + 0.000277893 * (0.005026783 * 0.5 * ((0.09704114 - Q.pz_lnkt1) / 0.005026783) * (1 + math.erf(((0.09704114 - Q.pz_lnkt1) / 0.005026783) / math.sqrt(2)))) / 0.01347961   # +0.0%  pz_lnkt1 < 0.09704
        - 0.0001875543 * (3.858926 * 0.5 * ((Q.mres_sd_mass_b0z005 - 111.9362) / 3.858926) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 111.9362) / 3.858926) / math.sqrt(2)))) / 13.22014   # -0.0%  mres_sd_mass_b0z005 > 111.9
        + 2.911754e-05 * (12.30925 * 0.5 * ((Q.mass_displaced3 - 39.09615) / 12.30925) * (1 + math.erf(((Q.mass_displaced3 - 39.09615) / 12.30925) / math.sqrt(2)))) * (0.006172711 * 0.5 * ((Q.lepsj_3_dr - 0.01012269) / 0.006172711) * (1 + math.erf(((Q.lepsj_3_dr - 0.01012269) / 0.006172711) / math.sqrt(2)))) / 0.01743932   # +0.0%  mass_displaced3 > 39.1 and lepsj_3_dr > 0.01012
    )
    return z


def neuron_79(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.602387e-05
    )
    return z


def neuron_80(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.707701e-06
    )
    return z


def neuron_81(Q):
    # scale S = 7.189; each line: share * term / its average size
    z = 7.189197 * (0.07237603
        - 0.115694 * (19.14168 * 0.5 * ((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2)))) / 55.1407   # -11.6%  mres_sd_mass_b2z01 < 158.2
        + 0.1061448 * (3.499256 * 0.5 * ((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) * (1 + math.erf(((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) / math.sqrt(2)))) / 24.89273   # +10.6%  mres_sd_mass_b2z01 < 121.6
        - 0.06049323 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 6.361002   # -6.0%  n_s3d_above_3 < 10
        - 0.05483913 * (5.26174 * 0.5 * ((91.15481 - Q.mres_sd_mass_b2z01) / 5.26174) * (1 + math.erf(((91.15481 - Q.mres_sd_mass_b2z01) / 5.26174) / math.sqrt(2)))) / 9.495189   # -5.5%  mres_sd_mass_b2z01 < 91.15
        + 0.05439573 * (1.0 * 0.5 * ((6.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((6.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 2.857   # +5.4%  n_s3d_above_3 < 6
        - 0.04296934 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) / 36.20948   # -4.3%  lep_ptrel < 43.21
        + 0.03716842 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) / 0.08745274   # +3.7%  z_displaced3 > 0.03624
        - 0.03235721 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2)))) / 9.427083   # -3.2%  n_dr_0p4_up < 15
        + 0.03144984 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) / 2.75999   # +3.1%  max_abs_d0 < 5.812
        + 0.02613715 * (5.55459 * 0.5 * ((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) / 11.56313   # +2.6%  mres_sd_mass_b2z01 < 96.62
        - 0.02468919 * (29.76053 * 0.5 * ((44.14912 - Q.lep_iso) / 29.76053) * (1 + math.erf(((44.14912 - Q.lep_iso) / 29.76053) / math.sqrt(2)))) / 36.80686   # -2.5%  lep_iso < 44.15
        - 0.02100579 * (0.5788858 * 0.5 * ((1.777286 - Q.mass_displaced3) / 0.5788858) * (1 + math.erf(((1.777286 - Q.mass_displaced3) / 0.5788858) / math.sqrt(2)))) / 0.7349753   # -2.1%  mass_displaced3 < 1.777
        - 0.02012211 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.265564) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.265564) / 0.0649922) / math.sqrt(2)))) / 0.02458276   # -2.0%  z_displaced3 > 0.03624 and z_charged_had > 0.2656
        + 0.01951408 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) / 297.4519   # +2.0%  sip_3d_2 < 447.1
        - 0.01655138 * (0.2381965 * 0.5 * ((Q.lne_5 - 2.737811) / 0.2381965) * (1 + math.erf(((Q.lne_5 - 2.737811) / 0.2381965) / math.sqrt(2)))) / 0.8428346   # -1.7%  lne_5 > 2.738
        + 0.0162055 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) / 0.2702483   # +1.6%  lep_z < 0.3397
        + 0.01612875 * (6.036668 * 0.5 * ((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) * (1 + math.erf(((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) / math.sqrt(2)))) / 5.409032   # +1.6%  mres_sd_mass_b2z01 < 76.15
        + 0.0142787 * (1.0 * 0.5 * ((3.0 - Q.sdb_0_n) / 1.0) * (1 + math.erf(((3.0 - Q.sdb_0_n) / 1.0) / math.sqrt(2)))) / 1.59319   # +1.4%  sdb_0_n < 3
        + 0.01391522 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.2370407 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.2370407 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2)))) / 0.6755992   # +1.4%  n_s3d_above_3 < 10 and z_neutral_had < 0.237
        + 0.01321869 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 3.0) / 1.0) / math.sqrt(2)))) / 0.9275361   # +1.3%  n_s3d_above_10 > 3
        - 0.01313524 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (6.084951 * 0.5 * ((16.23838 - Q.sip_3d_3) / 6.084951) * (1 + math.erf(((16.23838 - Q.sip_3d_3) / 6.084951) / math.sqrt(2)))) / 29.44929   # -1.3%  max_abs_d0 < 5.812 and sip_3d_3 < 16.24
        - 0.01293213 * Q.sv_n / 1.137077   # -1.3%  sv_n
        + 0.01154192 * (0.009443246 * 0.5 * ((0.0176083 - Q.lepsj_3_dr) / 0.009443246) * (1 + math.erf(((0.0176083 - Q.lepsj_3_dr) / 0.009443246) / math.sqrt(2)))) / 0.01235423   # +1.2%  lepsj_3_dr < 0.01761
        + 0.01122279 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) / 2.235664   # +1.1%  n_s3d_above_3 > 2
        - 0.01091856 * (1.0 * 0.5 * ((1.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((1.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) / 0.6926634   # -1.1%  lepsj_3_n_d3 < 1
        - 0.01060882 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.ak02_min12_n_disp3) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_min12_n_disp3) / 1.0) / math.sqrt(2)))) / 221.9615   # -1.1%  sip_3d_2 < 447.1 and ak02_min12_n_disp3 < 1
        + 0.009950387 * (11.4891 * 0.5 * ((32.61679 - Q.mass_displaced5) / 11.4891) * (1 + math.erf(((32.61679 - Q.mass_displaced5) / 11.4891) / math.sqrt(2)))) / 26.79867   # +1.0%  mass_displaced5 < 32.62
        + 0.009615436 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (0.006796583 * 0.5 * ((0.05966366 - Q.M2_b2) / 0.006796583) * (1 + math.erf(((0.05966366 - Q.M2_b2) / 0.006796583) / math.sqrt(2)))) / 0.06188117   # +1.0%  n_s3d_above_3 > 2 and M2_b2 < 0.05966
        + 0.008805362 * (3.360928 * 0.5 * ((Q.mass_top40 - 115.7429) / 3.360928) * (1 + math.erf(((Q.mass_top40 - 115.7429) / 3.360928) / math.sqrt(2)))) / 10.91322   # +0.9%  mass_top40 > 115.7
        - 0.00791211 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) / 0.06043165   # -0.8%  pz_lnd0 < 0.1713
        - 0.007547103 * (76.66736 * 0.5 * ((164.4318 - Q.jd_sum_abs_sd0_top3) / 76.66736) * (1 + math.erf(((164.4318 - Q.jd_sum_abs_sd0_top3) / 76.66736) / math.sqrt(2)))) / 68.54094   # -0.8%  jd_sum_abs_sd0_top3 < 164.4
        - 0.00704678 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_charged_pt_above_10 - 3.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_10 - 3.0) / 1.0) / math.sqrt(2)))) / 1481.365   # -0.7%  sip_3d_2 < 447.1 and n_charged_pt_above_10 > 3
        + 0.006928601 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) / 1.215201   # +0.7%  n_dr_0p4_up < 4
        + 0.006492177 * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2)))) * (0.05285263 * 0.5 * ((0.1270192 - Q.sdb_4_z) / 0.05285263) * (1 + math.erf(((0.1270192 - Q.sdb_4_z) / 0.05285263) / math.sqrt(2)))) / 0.1339419   # +0.6%  dc_n > 1 and sdb_4_z < 0.127
        - 0.005713846 * (10.51039 * 0.5 * ((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) * (1 + math.erf(((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) / math.sqrt(2)))) / 4.332549   # -0.6%  mres_sd_mass_b2z01 < 69.12
        + 0.005668705 * (0.008567977 * 0.5 * ((0.4920044 - Q.pt_balance01) / 0.008567977) * (1 + math.erf(((0.4920044 - Q.pt_balance01) / 0.008567977) / math.sqrt(2)))) / 0.1256134   # +0.6%  pt_balance01 < 0.492
        + 0.005102908 * (0.00773283 * 0.5 * ((0.07991731 - Q.tau2) / 0.00773283) * (1 + math.erf(((0.07991731 - Q.tau2) / 0.00773283) / math.sqrt(2)))) / 0.02050542   # +0.5%  tau2 < 0.07992
        + 0.005080449 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2)))) / 0.0537686   # +0.5%  n_lepton < 0
        - 0.005079246 * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2)))) / 1.238945   # -0.5%  dc_n > 1
        - 0.00465406 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.2297361 * 0.5 * ((Q.ak02_dr13 - 0.0) / 0.2297361) * (1 + math.erf(((Q.ak02_dr13 - 0.0) / 0.2297361) / math.sqrt(2)))) / 0.0151563   # -0.5%  pz_lnd0 < 0.1713 and ak02_dr13 > 0
        - 0.004581018 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (0.0964495 * 0.5 * ((0.5557674 - Q.sjq_2_sumabs_k1) / 0.0964495) * (1 + math.erf(((0.5557674 - Q.sjq_2_sumabs_k1) / 0.0964495) / math.sqrt(2)))) / 0.02290881   # -0.5%  z_displaced3 > 0.03624 and sjq_2_sumabs_k1 < 0.5558
        - 0.004439105 * (0.01706594 * 0.5 * ((0.7339085 - Q.max_dr) / 0.01706594) * (1 + math.erf(((0.7339085 - Q.max_dr) / 0.01706594) / math.sqrt(2)))) / 0.1173315   # -0.4%  max_dr < 0.7339
        - 0.004419765 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 3.0) / 1.0) / math.sqrt(2)))) * (2.0 * 0.5 * ((4.0 - Q.lepsj_2_n_d3) / 2.0) * (1 + math.erf(((4.0 - Q.lepsj_2_n_d3) / 2.0) / math.sqrt(2)))) / 2.363852   # -0.4%  n_s3d_above_10 > 3 and lepsj_2_n_d3 < 4
        - 0.004406044 * (1.0 * 0.5 * ((Q.sjf_4_n2disp - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_4_n2disp - 1.0) / 1.0) / math.sqrt(2)))) / 0.2811624   # -0.4%  sjf_4_n2disp > 1
        - 0.004036198 * (0.2210883 * 0.5 * ((1.514826 - Q.min_pair_mass) / 0.2210883) * (1 + math.erf(((1.514826 - Q.min_pair_mass) / 0.2210883) / math.sqrt(2)))) / 0.4642144   # -0.4%  min_pair_mass < 1.515
        - 0.004008787 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (8.422628 * 0.5 * ((23.30856 - Q.sip_3d_3) / 8.422628) * (1 + math.erf(((23.30856 - Q.sip_3d_3) / 8.422628) / math.sqrt(2)))) / 4616.07   # -0.4%  sip_3d_2 < 447.1 and sip_3d_3 < 23.31
        - 0.003867941 * (0.02426199 * 0.5 * ((0.2040425 - Q.sdb_2_z) / 0.02426199) * (1 + math.erf(((0.2040425 - Q.sdb_2_z) / 0.02426199) / math.sqrt(2)))) * (5.085197 * 0.5 * ((10.7744 - Q.jd_3d_5) / 5.085197) * (1 + math.erf(((10.7744 - Q.jd_3d_5) / 5.085197) / math.sqrt(2)))) / 0.1135765   # -0.4%  sdb_2_z < 0.204 and jd_3d_5 < 10.77
        - 0.003800636 * (0.1799842 * 0.5 * ((2.044945 - Q.sjf_2_2_max3d) / 0.1799842) * (1 + math.erf(((2.044945 - Q.sjf_2_2_max3d) / 0.1799842) / math.sqrt(2)))) / 0.1618068   # -0.4%  sjf_2_2_max3d < 2.045
        + 0.003723086 * (0.04445521 * 0.5 * ((0.2686235 - Q.LHA) / 0.04445521) * (1 + math.erf(((0.2686235 - Q.LHA) / 0.04445521) / math.sqrt(2)))) / 0.007316391   # +0.4%  LHA < 0.2686
        - 0.003606734 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) / 3.676114   # -0.4%  n_pairs_kt_above_3 < 28
        - 0.003250498 * (16.26099 * 0.5 * ((Q.mass_top50 - 161.1264) / 16.26099) * (1 + math.erf(((Q.mass_top50 - 161.1264) / 16.26099) / math.sqrt(2)))) / 3.008129   # -0.3%  mass_top50 > 161.1
        + 0.003024403 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 5.193195   # +0.3%  n_pairs_kt_above_1 < 80
        - 0.002874727 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) / 0.05253524   # -0.3%  z_displaced5 > 0.0809
        - 0.002860593 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.05007501   # -0.3%  z_displaced3 > 0.03624 and ak02_2_n_lep < 1
        + 0.002839301 * (0.02426199 * 0.5 * ((0.2040425 - Q.sdb_2_z) / 0.02426199) * (1 + math.erf(((0.2040425 - Q.sdb_2_z) / 0.02426199) / math.sqrt(2)))) / 0.02559335   # +0.3%  sdb_2_z < 0.204
        - 0.002838559 * (1.0 * 0.5 * ((0.0 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.0 - Q.sv_2_n) / 1.0) / math.sqrt(2)))) / 0.04355367   # -0.3%  sv_2_n < 0
        - 0.002565695 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (0.02572129 * 0.5 * ((0.9433644 - Q.tau43_b2) / 0.02572129) * (1 + math.erf(((0.9433644 - Q.tau43_b2) / 0.02572129) / math.sqrt(2)))) / 0.02017023   # -0.3%  z_displaced3 > 0.03624 and tau43_b2 < 0.9434
        + 0.002516153 * (1.0 * 0.5 * ((Q.sjf_4_1_n_d3 - 3.0) / 1.0) * (1 + math.erf(((Q.sjf_4_1_n_d3 - 3.0) / 1.0) / math.sqrt(2)))) / 0.3200949   # +0.3%  sjf_4_1_n_d3 > 3
        - 0.002515223 * (5.55459 * 0.5 * ((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) * (0.01467775 * 0.5 * ((0.2243273 - Q.N2_b2) / 0.01467775) * (1 + math.erf(((0.2243273 - Q.N2_b2) / 0.01467775) / math.sqrt(2)))) / 0.4964292   # -0.3%  mres_sd_mass_b2z01 < 96.62 and N2_b2 < 0.2243
        - 0.002391028 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2)))) / 6.231872   # -0.2%  n_dr_0p4_up < 15 and dc_1_n_lep < 1
        + 0.002218693 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (10.60708 * 0.5 * ((16.11752 - Q.jd_3d_6) / 10.60708) * (1 + math.erf(((16.11752 - Q.jd_3d_6) / 10.60708) / math.sqrt(2)))) / 33.72373   # +0.2%  max_abs_d0 < 5.812 and jd_3d_6 < 16.12
        - 0.00207638 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (1.5 * 0.5 * ((7.0 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.0 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2)))) / 0.1108979   # -0.2%  z_displaced3 > 0.03624 and n_s3d_above_3 < 7
        + 0.00201075 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) * (0.006013287 * 0.5 * ((0.06661369 - Q.pz_lnkt3) / 0.006013287) * (1 + math.erf(((0.06661369 - Q.pz_lnkt3) / 0.006013287) / math.sqrt(2)))) / 0.02246715   # +0.2%  n_dr_0p4_up < 4 and pz_lnkt3 < 0.06661
        - 0.001880339 * Q.sjq_2_prod_k05 / 0.1587328   # -0.2%  sjq_2_prod_k05
        - 0.001652743 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_electron - 1.0) / 1.0) * (1 + math.erf(((Q.n_electron - 1.0) / 1.0) / math.sqrt(2)))) / 0.4579033   # -0.2%  n_s3d_above_3 > 2 and n_electron > 1
        - 0.001565433 * (3.579676 * 0.5 * ((63.34656 - Q.sj4_pair_mass_max) / 3.579676) * (1 + math.erf(((63.34656 - Q.sj4_pair_mass_max) / 3.579676) / math.sqrt(2)))) / 3.788053   # -0.2%  sj4_pair_mass_max < 63.35
        + 0.001456463 * (4.193136 * 0.5 * ((6.64573 - Q.sjf_2_1_mass_d3) / 4.193136) * (1 + math.erf(((6.64573 - Q.sjf_2_1_mass_d3) / 4.193136) / math.sqrt(2)))) / 4.695909   # +0.1%  sjf_2_1_mass_d3 < 6.646
        + 0.001453468 * (19.14168 * 0.5 * ((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2)))) * (5.386244 * 0.5 * ((Q.mass_2charged - 28.89659) / 5.386244) * (1 + math.erf(((Q.mass_2charged - 28.89659) / 5.386244) / math.sqrt(2)))) / 72.89715   # +0.1%  mres_sd_mass_b2z01 < 158.2 and mass_2charged > 28.9
        - 0.001345774 * (17.59858 * 0.5 * ((Q.mass_top50 - 178.725) / 17.59858) * (1 + math.erf(((Q.mass_top50 - 178.725) / 17.59858) / math.sqrt(2)))) / 1.705883   # -0.1%  mass_top50 > 178.7
        - 0.001330279 * (4.106456 * 0.5 * ((5.245219 - Q.sjf_3_2_max3d) / 4.106456) * (1 + math.erf(((5.245219 - Q.sjf_3_2_max3d) / 4.106456) / math.sqrt(2)))) / 1.450133   # -0.1%  sjf_3_2_max3d < 5.245
        + 0.00126172 * (0.01154615 * 0.5 * ((0.01545532 - Q.psi_0p1) / 0.01154615) * (1 + math.erf(((0.01545532 - Q.psi_0p1) / 0.01154615) / math.sqrt(2)))) / 0.0015287   # +0.1%  psi_0p1 < 0.01546
        - 0.001038661 * (0.005213255 * 0.5 * ((0.0 - Q.z_displaced3) / 0.005213255) * (1 + math.erf(((0.0 - Q.z_displaced3) / 0.005213255) / math.sqrt(2)))) / 5.985572e-05   # -0.1%  z_displaced3 < 0
        + 0.001026016 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (401.9953 * 0.5 * ((577.991 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((577.991 - Q.sip_3d_3) / 401.9953) / math.sqrt(2)))) / 898.4184   # +0.1%  n_s3d_above_3 > 2 and sip_3d_3 < 578
        + 0.0009720117 * (19.14168 * 0.5 * ((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2)))) * (0.0467133 * 0.5 * ((0.5068038 - Q.tau32) / 0.0467133) * (1 + math.erf(((0.5068038 - Q.tau32) / 0.0467133) / math.sqrt(2)))) / 0.9489034   # +0.1%  mres_sd_mass_b2z01 < 158.2 and tau32 < 0.5068
        + 0.0007786383 * (0.06585074 * 0.5 * ((-0.1170754 - Q.tdz_1) / 0.06585074) * (1 + math.erf(((-0.1170754 - Q.tdz_1) / 0.06585074) / math.sqrt(2)))) / 0.01660446   # +0.1%  tdz_1 < -0.1171
        + 0.0007586197 * (11.52979 * 0.5 * ((Q.mass_charged - 99.20396) / 11.52979) * (1 + math.erf(((Q.mass_charged - 99.20396) / 11.52979) / math.sqrt(2)))) / 2.252362   # +0.1%  mass_charged > 99.2
        - 0.0007200462 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) * (0.01480848 * 0.5 * ((Q.nca_sj4_pair2nd_over_mass - 0.4488456) / 0.01480848) * (1 + math.erf(((Q.nca_sj4_pair2nd_over_mass - 0.4488456) / 0.01480848) / math.sqrt(2)))) / 0.08260434   # -0.1%  n_dr_0p4_up < 4 and nca_sj4_pair2nd_over_mass > 0.4488
        - 0.0006099978 * (5.02206 * 0.5 * ((Q.mass_top50 - 129.5874) / 5.02206) * (1 + math.erf(((Q.mass_top50 - 129.5874) / 5.02206) / math.sqrt(2)))) / 7.907933   # -0.1%  mass_top50 > 129.6
        - 0.0006013277 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2)))) / 0.2342604   # -0.1%  sjf_2_1_n_d3 > 5
        - 0.0004976912 * (0.01683601 * 0.5 * ((0.1339824 - Q.pz_lnd2) / 0.01683601) * (1 + math.erf(((0.1339824 - Q.pz_lnd2) / 0.01683601) / math.sqrt(2)))) / 0.04289153   # -0.0%  pz_lnd2 < 0.134
        + 0.0003976688 * (0.2210883 * 0.5 * ((1.514826 - Q.min_pair_mass) / 0.2210883) * (1 + math.erf(((1.514826 - Q.min_pair_mass) / 0.2210883) / math.sqrt(2)))) * (9.707626 * 0.5 * ((16.47324 - Q.dc_1_sd0_1) / 9.707626) * (1 + math.erf(((16.47324 - Q.dc_1_sd0_1) / 9.707626) / math.sqrt(2)))) / 4.632587   # +0.0%  min_pair_mass < 1.515 and dc_1_sd0_1 < 16.47
        + 0.0003694491 * (0.02426199 * 0.5 * ((0.2040425 - Q.sdb_2_z) / 0.02426199) * (1 + math.erf(((0.2040425 - Q.sdb_2_z) / 0.02426199) / math.sqrt(2)))) * (0.01654934 * 0.5 * ((Q.tau54 - 0.8015743) / 0.01654934) * (1 + math.erf(((Q.tau54 - 0.8015743) / 0.01654934) / math.sqrt(2)))) / 0.001477717   # +0.0%  sdb_2_z < 0.204 and tau54 > 0.8016
        - 0.0002937887 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.isnhad_8 - 1.0) / 1.0) * (1 + math.erf(((Q.isnhad_8 - 1.0) / 1.0) / math.sqrt(2)))) / 0.3073611   # -0.0%  n_s3d_above_3 > 2 and isnhad_8 > 1
        - 0.0002859265 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (0.01789856 * 0.5 * ((Q.eta_8 - 0.04946899) / 0.01789856) * (1 + math.erf(((Q.eta_8 - 0.04946899) / 0.01789856) / math.sqrt(2)))) / 0.002985747   # -0.0%  z_displaced3 > 0.03624 and eta_8 > 0.04947
        + 0.0002666248 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.02367609 * 0.5 * ((0.7382159 - Q.tau43) / 0.02367609) * (1 + math.erf(((0.7382159 - Q.tau43) / 0.02367609) / math.sqrt(2)))) / 0.002183915   # +0.0%  pz_lnd0 < 0.1713 and tau43 < 0.7382
        + 0.0002198045 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 3.0) / 1.0) / math.sqrt(2)))) * (0.06646541 * 0.5 * ((0.3206143 - Q.pt1_over_pt0) / 0.06646541) * (1 + math.erf(((0.3206143 - Q.pt1_over_pt0) / 0.06646541) / math.sqrt(2)))) / 0.01156029   # +0.0%  n_s3d_above_10 > 3 and pt1_over_pt0 < 0.3206
        + 0.0002100742 * (0.00837824 * 0.5 * ((0.2743483 - Q.C2_b05) / 0.00837824) * (1 + math.erf(((0.2743483 - Q.C2_b05) / 0.00837824) / math.sqrt(2)))) / 0.02950575   # +0.0%  C2_b05 < 0.2743
        - 0.0001646868 * (14.45336 * 0.5 * ((Q.jd_3d_6 - 30.57088) / 14.45336) * (1 + math.erf(((Q.jd_3d_6 - 30.57088) / 14.45336) / math.sqrt(2)))) / 5.748674   # -0.0%  jd_3d_6 > 30.57
        + 0.000154285 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) * (0.05748496 * 0.5 * ((0.2338497 - Q.lep_dr) / 0.05748496) * (1 + math.erf(((0.2338497 - Q.lep_dr) / 0.05748496) / math.sqrt(2)))) / 0.6043239   # +0.0%  n_pairs_kt_above_3 < 28 and lep_dr < 0.2338
        + 0.0001327134 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ismuon_3 - 0.0) / 1e-06) * (1 + math.erf(((Q.ismuon_3 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.001924117   # +0.0%  z_displaced3 > 0.03624 and ismuon_3 > 0
        + 0.00011872 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dr_68 - 0.0) / 1e-06) * (1 + math.erf(((Q.dr_68 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.0006481155   # +0.0%  z_displaced3 > 0.03624 and dr_68 > 0
        - 8.685761e-05 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.001692006 * 0.5 * ((Q.sjf_4_3_z_d3 - 0.0) / 0.001692006) * (1 + math.erf(((Q.sjf_4_3_z_d3 - 0.0) / 0.001692006) / math.sqrt(2)))) / 0.04813776   # -0.0%  n_s3d_above_3 < 10 and sjf_4_3_z_d3 > 0
        + 6.300378e-05 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.nca_kt_above_2 - 4.0) / 1.0) * (1 + math.erf(((Q.nca_kt_above_2 - 4.0) / 1.0) / math.sqrt(2)))) / 0.851345   # +0.0%  n_s3d_above_3 > 2 and nca_kt_above_2 > 4
        - 5.873112e-05 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2)))) * (17.93691 * 0.5 * ((28.56971 - Q.dc_1_sd0_1) / 17.93691) * (1 + math.erf(((28.56971 - Q.dc_1_sd0_1) / 17.93691) / math.sqrt(2)))) / 1.81338   # -0.0%  sjf_2_1_n_d3 > 5 and dc_1_sd0_1 < 28.57
        - 5.75361e-05 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.isnhad_34 - 0.0) / 1e-06) * (1 + math.erf(((Q.isnhad_34 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.1143614   # -0.0%  n_s3d_above_3 > 2 and isnhad_34 > 0
        + 2.977574e-05 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (275.2718 * 0.5 * ((578.2954 - Q.sip_3d_1) / 275.2718) * (1 + math.erf(((578.2954 - Q.sip_3d_1) / 275.2718) / math.sqrt(2)))) / 508.0896   # +0.0%  n_s3d_above_3 > 2 and sip_3d_1 < 578.3
        - 2.313111e-05 * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 5.0) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 5.0) / 1.0) / math.sqrt(2)))) / 0.07765456   # -0.0%  sjf_2_2_n_d3 > 5
        + 6.18857e-06 * (0.06585074 * 0.5 * ((-0.1170754 - Q.tdz_1) / 0.06585074) * (1 + math.erf(((-0.1170754 - Q.tdz_1) / 0.06585074) / math.sqrt(2)))) * (5.901273 * 0.5 * ((Q.kt2_1_sd0_3 - 4.457634) / 5.901273) * (1 + math.erf(((Q.kt2_1_sd0_3 - 4.457634) / 5.901273) / math.sqrt(2)))) / 0.1802731   # +0.0%  tdz_1 < -0.1171 and kt2_1_sd0_3 > 4.458
        + 5.556412e-06 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.td0_50 - 0.0) / 1e-06) * (1 + math.erf(((Q.td0_50 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.0002074911   # +0.0%  z_displaced5 > 0.0809 and td0_50 > 0
    )
    return z


def neuron_82(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.942071e-06
    )
    return z


def neuron_83(Q):
    # scale S = 8.539; each line: share * term / its average size
    z = 8.539027 * (-0.08780096
        + 0.05919573 * (16.90428 * 0.5 * ((164.4374 - Q.mass) / 16.90428) * (1 + math.erf(((164.4374 - Q.mass) / 16.90428) / math.sqrt(2)))) / 52.34277   # +5.9%  mass < 164.4
        - 0.05654808 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (3.384707 * 0.5 * ((6.771002 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.771002 - Q.jd_3d_5) / 3.384707) / math.sqrt(2)))) / 10.84308   # -5.7%  max_abs_d0 < 5.812 and jd_3d_5 < 6.771
        + 0.05511382 * (10.09897 * 0.5 * ((115.5142 - Q.sj4_pair_mass_max) / 10.09897) * (1 + math.erf(((115.5142 - Q.sj4_pair_mass_max) / 10.09897) / math.sqrt(2)))) / 36.41562   # +5.5%  sj4_pair_mass_max < 115.5
        - 0.04740541 * (0.00905862 * 0.5 * ((0.06519357 - Q.tau5) / 0.00905862) * (1 + math.erf(((0.06519357 - Q.tau5) / 0.00905862) / math.sqrt(2)))) / 0.03346063   # -4.7%  tau5 < 0.06519
        + 0.03878334 * (1.5 * 0.5 * ((10.0 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((10.0 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2)))) / 5.202348   # +3.9%  n_dr_0p4_up < 10
        + 0.03237199 * (3.384707 * 0.5 * ((6.771002 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.771002 - Q.jd_3d_5) / 3.384707) / math.sqrt(2)))) / 3.213863   # +3.2%  jd_3d_5 < 6.771
        - 0.03088895 * (5.197 * 0.5 * ((Q.mass - 95.14961) / 5.197) * (1 + math.erf(((Q.mass - 95.14961) / 5.197) / math.sqrt(2)))) / 26.15535   # -3.1%  mass > 95.15
        + 0.02982685 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) / 2.75999   # +3.0%  max_abs_d0 < 5.812
        - 0.02925682 * (2.0 * 0.5 * ((8.0 - Q.n_s3d_above_10) / 2.0) * (1 + math.erf(((8.0 - Q.n_s3d_above_10) / 2.0) / math.sqrt(2)))) / 5.429064   # -2.9%  n_s3d_above_10 < 8
        - 0.02815107 * (6.036668 * 0.5 * ((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) * (1 + math.erf(((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) / math.sqrt(2)))) / 5.409032   # -2.8%  mres_sd_mass_b2z01 < 76.15
        + 0.02613999 * (1.951776e-05 * 0.5 * ((Q.ecf_g41 - 8.500146e-05) / 1.951776e-05) * (1 + math.erf(((Q.ecf_g41 - 8.500146e-05) / 1.951776e-05) / math.sqrt(2)))) / 0.0001297133   # +2.6%  ecf_g41 > 8.5e-05
        - 0.02439069 * (1.0 * 0.5 * ((2.0 - Q.sjf_2_n2disp) / 1.0) * (1 + math.erf(((2.0 - Q.sjf_2_n2disp) / 1.0) / math.sqrt(2)))) / 1.110043   # -2.4%  sjf_2_n2disp < 2
        - 0.02363777 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.33213) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.33213) / 6.345399) / math.sqrt(2)))) * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) / 5.859203   # -2.4%  mass_top40 > 78.33 and lep_z < 0.2214
        + 0.02346043 * (10.14805 * 0.5 * ((26.78691 - Q.mass_displaced3) / 10.14805) * (1 + math.erf(((26.78691 - Q.mass_displaced3) / 10.14805) / math.sqrt(2)))) * (3.384707 * 0.5 * ((6.771002 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.771002 - Q.jd_3d_5) / 3.384707) / math.sqrt(2)))) / 78.12837   # +2.3%  mass_displaced3 < 26.79 and jd_3d_5 < 6.771
        - 0.02296279 * (0.09779513 * 0.5 * ((0.6703997 - Q.sdb_2_z) / 0.09779513) * (1 + math.erf(((0.6703997 - Q.sdb_2_z) / 0.09779513) / math.sqrt(2)))) / 0.3554605   # -2.3%  sdb_2_z < 0.6704
        + 0.02292024 * (0.02099671 * 0.5 * ((0.08304558 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.08304558 - Q.z_displaced3) / 0.02099671) / math.sqrt(2)))) / 0.0372965   # +2.3%  z_displaced3 < 0.08305
        + 0.02050476 * (6.230332 * 0.5 * ((13.50273 - Q.sip_3d_2) / 6.230332) * (1 + math.erf(((13.50273 - Q.sip_3d_2) / 6.230332) / math.sqrt(2)))) / 4.217535   # +2.1%  sip_3d_2 < 13.5
        + 0.02039157 * (10.14805 * 0.5 * ((26.78691 - Q.mass_displaced3) / 10.14805) * (1 + math.erf(((26.78691 - Q.mass_displaced3) / 10.14805) / math.sqrt(2)))) / 20.19226   # +2.0%  mass_displaced3 < 26.79
        + 0.02005994 * (10.51039 * 0.5 * ((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) * (1 + math.erf(((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) / math.sqrt(2)))) / 4.332549   # +2.0%  mres_sd_mass_b2z01 < 69.12
        - 0.01981282 * (8.494762 * 0.5 * ((20.75111 - Q.sip_3d_2) / 8.494762) * (1 + math.erf(((20.75111 - Q.sip_3d_2) / 8.494762) / math.sqrt(2)))) / 7.680124   # -2.0%  sip_3d_2 < 20.75
        - 0.01915272 * (3.373048 * 0.5 * ((Q.mass_top15 - 83.68425) / 3.373048) * (1 + math.erf(((Q.mass_top15 - 83.68425) / 3.373048) / math.sqrt(2)))) / 10.76171   # -1.9%  mass_top15 > 83.68
        + 0.01550814 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (5.085197 * 0.5 * ((10.7744 - Q.jd_3d_5) / 5.085197) * (1 + math.erf(((10.7744 - Q.jd_3d_5) / 5.085197) / math.sqrt(2)))) / 20.62538   # +1.6%  max_abs_d0 < 5.812 and jd_3d_5 < 10.77
        - 0.01414616 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.33213) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.33213) / 6.345399) / math.sqrt(2)))) * (0.01771551 * 0.5 * ((0.1204509 - Q.C2_b2) / 0.01771551) * (1 + math.erf(((0.1204509 - Q.C2_b2) / 0.01771551) / math.sqrt(2)))) / 1.678843   # -1.4%  mass_top40 > 78.33 and C2_b2 < 0.1205
        + 0.0139565 * (16.47782 * 0.5 * ((162.7874 - Q.mass_top30) / 16.47782) * (1 + math.erf(((162.7874 - Q.mass_top30) / 16.47782) / math.sqrt(2)))) / 59.44714   # +1.4%  mass_top30 < 162.8
        + 0.01370946 * (4.245069 * 0.5 * ((125.2732 - Q.mres_sd_mass_b2z01) / 4.245069) * (1 + math.erf(((125.2732 - Q.mres_sd_mass_b2z01) / 4.245069) / math.sqrt(2)))) / 27.53555   # +1.4%  mres_sd_mass_b2z01 < 125.3
        - 0.01368477 * (0.01159153 * 0.5 * ((0.04504618 - Q.z_dr_0p4_up) / 0.01159153) * (1 + math.erf(((0.04504618 - Q.z_dr_0p4_up) / 0.01159153) / math.sqrt(2)))) / 0.02142498   # -1.4%  z_dr_0p4_up < 0.04505
        + 0.01353634 * (3.153038 * 0.5 * ((Q.mass - 120.653) / 3.153038) * (1 + math.erf(((Q.mass - 120.653) / 3.153038) / math.sqrt(2)))) / 11.75055   # +1.4%  mass > 120.7
        - 0.01305382 * (0.001766249 * 0.5 * ((0.01724773 - Q.M3_b2) / 0.001766249) * (1 + math.erf(((0.01724773 - Q.M3_b2) / 0.001766249) / math.sqrt(2)))) / 0.006121328   # -1.3%  M3_b2 < 0.01725
        + 0.01147523 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) / 0.04874886   # +1.1%  lepsj_2_dr < 0.06573
        + 0.01097691 * (1.0 * 0.5 * ((3.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((3.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 0.9010223   # +1.1%  n_s3d_above_3 < 3
        + 0.01073765 * Q.pz_lnd2 / 0.1223855   # +1.1%  pz_lnd2
        - 0.01064498 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.33213) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.33213) / 6.345399) / math.sqrt(2)))) * (18.22661 * 0.5 * ((27.25083 - Q.jd_3d_5) / 18.22661) * (1 + math.erf(((27.25083 - Q.jd_3d_5) / 18.22661) / math.sqrt(2)))) / 601.5774   # -1.1%  mass_top40 > 78.33 and jd_3d_5 < 27.25
        + 0.01057812 * (1.0 * 0.5 * ((2.0 - Q.sdb_5_n) / 1.0) * (1 + math.erf(((2.0 - Q.sdb_5_n) / 1.0) / math.sqrt(2)))) / 1.040282   # +1.1%  sdb_5_n < 2
        - 0.01055825 * (1.951776e-05 * 0.5 * ((Q.ecf_g41 - 8.500146e-05) / 1.951776e-05) * (1 + math.erf(((Q.ecf_g41 - 8.500146e-05) / 1.951776e-05) / math.sqrt(2)))) * (57.98295 * 0.5 * ((767.6359 - Q.sum_pt_top40) / 57.98295) * (1 + math.erf(((767.6359 - Q.sum_pt_top40) / 57.98295) / math.sqrt(2)))) / 0.02265369   # -1.1%  ecf_g41 > 8.5e-05 and sum_pt_top40 < 767.6
        + 0.0102637 * (5.171003 * 0.5 * ((7.028704 - Q.sip_3d_1) / 5.171003) * (1 + math.erf(((7.028704 - Q.sip_3d_1) / 5.171003) / math.sqrt(2)))) / 0.8646436   # +1.0%  sip_3d_1 < 7.029
        - 0.01022954 * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_10) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_10) / 1.0) / math.sqrt(2)))) * (0.02426199 * 0.5 * ((Q.sdb_2_z - 0.2040425) / 0.02426199) * (1 + math.erf(((Q.sdb_2_z - 0.2040425) / 0.02426199) / math.sqrt(2)))) / 0.4688356   # -1.0%  n_s3d_above_10 < 5 and sdb_2_z > 0.204
        - 0.008759823 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.5 * 0.5 * ((2.0 - Q.lepsj_2_n_d3) / 1.5) * (1 + math.erf(((2.0 - Q.lepsj_2_n_d3) / 1.5) / math.sqrt(2)))) / 4.411501   # -0.9%  max_abs_d0 < 5.812 and lepsj_2_n_d3 < 2
        + 0.00824219 * (0.2916549 * 0.5 * ((Q.lne_21 - 1.025706) / 0.2916549) * (1 + math.erf(((Q.lne_21 - 1.025706) / 0.2916549) / math.sqrt(2)))) * (0.02507469 * 0.5 * ((0.1545914 - Q.sj4_zsoft) / 0.02507469) * (1 + math.erf(((0.1545914 - Q.sj4_zsoft) / 0.02507469) / math.sqrt(2)))) / 0.06959425   # +0.8%  lne_21 > 1.026 and sj4_zsoft < 0.1546
        - 0.008238681 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) / 26.78553   # -0.8%  sv_1_sd0_sum < 51.56
        + 0.007931465 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.0) / 1.0) / math.sqrt(2)))) * (248.3135 * 0.5 * ((175.9957 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((175.9957 - Q.sip_3d_3) / 248.3135) / math.sqrt(2)))) / 61.75029   # +0.8%  n_s3d_above_3 > 5 and sip_3d_3 < 176
        + 0.007475788 * (8.537109 * 0.5 * ((17.60938 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((17.60938 - Q.max_abs_d0) / 8.537109) / math.sqrt(2)))) / 10.41349   # +0.7%  max_abs_d0 < 17.61
        - 0.007311539 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2)))) / 1.091313   # -0.7%  mass < 56.49
        + 0.007055188 * (15.58693 * 0.5 * ((Q.mass_top40 - 155.8928) / 15.58693) * (1 + math.erf(((Q.mass_top40 - 155.8928) / 15.58693) / math.sqrt(2)))) / 2.899488   # +0.7%  mass_top40 > 155.9
        + 0.006972511 * (1.0 * 0.5 * ((11.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((11.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2)))) / 2.365772   # +0.7%  sdb_2_n < 11
        - 0.006833296 * (2.952984 * 0.5 * ((75.78208 - Q.sj4_pair_mass_max) / 2.952984) * (1 + math.erf(((75.78208 - Q.sj4_pair_mass_max) / 2.952984) / math.sqrt(2)))) * (22.16789 * 0.5 * ((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) * (1 + math.erf(((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) / math.sqrt(2)))) / 57.77006   # -0.7%  sj4_pair_mass_max < 75.78 and lepsj_3_maxsd0 < 11.53
        - 0.006400027 * (8.537109 * 0.5 * ((17.60938 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((17.60938 - Q.max_abs_d0) / 8.537109) / math.sqrt(2)))) * (0.1631384 * 0.5 * ((0.611688 - Q.D3) / 0.1631384) * (1 + math.erf(((0.611688 - Q.D3) / 0.1631384) / math.sqrt(2)))) / 3.176755   # -0.6%  max_abs_d0 < 17.61 and D3 < 0.6117
        + 0.005875087 * (81.90557 * 0.5 * ((185.1889 - Q.jd_sum_abs_sd0_top5) / 81.90557) * (1 + math.erf(((185.1889 - Q.jd_sum_abs_sd0_top5) / 81.90557) / math.sqrt(2)))) / 76.95359   # +0.6%  jd_sum_abs_sd0_top5 < 185.2
        + 0.005045197 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.0) / 1.0) / math.sqrt(2)))) * (0.02937692 * 0.5 * ((0.4866692 - Q.sj3_pairmin_over_m) / 0.02937692) * (1 + math.erf(((0.4866692 - Q.sj3_pairmin_over_m) / 0.02937692) / math.sqrt(2)))) / 0.1662124   # +0.5%  n_s3d_above_3 > 5 and sj3_pairmin_over_m < 0.4867
        - 0.004962519 * (0.01556476 * 0.5 * ((0.258375 - Q.N2) / 0.01556476) * (1 + math.erf(((0.258375 - Q.N2) / 0.01556476) / math.sqrt(2)))) / 0.01854925   # -0.5%  N2 < 0.2584
        + 0.004690509 * (0.04874922 * 0.5 * ((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) * (1 + math.erf(((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) / math.sqrt(2)))) * (6.566474 * 0.5 * ((12.91148 - Q.lepsj_3_mass) / 6.566474) * (1 + math.erf(((12.91148 - Q.lepsj_3_mass) / 6.566474) / math.sqrt(2)))) / 0.8641791   # +0.5%  sjf_4_1_z_d3 < 0.1277 and lepsj_3_mass < 12.91
        + 0.004189562 * (0.2024667 * 0.5 * ((Q.pair_max_lnm2 - 7.524613) / 0.2024667) * (1 + math.erf(((Q.pair_max_lnm2 - 7.524613) / 0.2024667) / math.sqrt(2)))) / 0.07291812   # +0.4%  pair_max_lnm2 > 7.525
        + 0.004122308 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.33213) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.33213) / 6.345399) / math.sqrt(2)))) * (0.01759885 * 0.5 * ((Q.sj3_z1 - 0.4972199) / 0.01759885) * (1 + math.erf(((Q.sj3_z1 - 0.4972199) / 0.01759885) / math.sqrt(2)))) / 3.650562   # +0.4%  mass_top40 > 78.33 and sj3_z1 > 0.4972
        - 0.003702392 * (0.2916549 * 0.5 * ((Q.lne_21 - 1.025706) / 0.2916549) * (1 + math.erf(((Q.lne_21 - 1.025706) / 0.2916549) / math.sqrt(2)))) / 0.8001974   # -0.4%  lne_21 > 1.026
        + 0.003351845 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.33213) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.33213) / 6.345399) / math.sqrt(2)))) / 34.88496   # +0.3%  mass_top40 > 78.33
        + 0.00322362 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2)))) / 0.2342604   # +0.3%  sjf_2_1_n_d3 > 5
        - 0.003057597 * (0.003751777 * 0.5 * ((0.007652966 - Q.z_dr_0_0p05) / 0.003751777) * (1 + math.erf(((0.007652966 - Q.z_dr_0_0p05) / 0.003751777) / math.sqrt(2)))) / 0.002571422   # -0.3%  z_dr_0_0p05 < 0.007653
        - 0.003003944 * (81.90557 * 0.5 * ((185.1889 - Q.jd_sum_abs_sd0_top5) / 81.90557) * (1 + math.erf(((185.1889 - Q.jd_sum_abs_sd0_top5) / 81.90557) / math.sqrt(2)))) * (0.003268159 * 0.5 * ((0.003370318 - Q.sjf_4_4_z_d3) / 0.003268159) * (1 + math.erf(((0.003370318 - Q.sjf_4_4_z_d3) / 0.003268159) / math.sqrt(2)))) / 0.202671   # -0.3%  jd_sum_abs_sd0_top5 < 185.2 and sjf_4_4_z_d3 < 0.00337
        + 0.002978181 * (0.04874922 * 0.5 * ((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) * (1 + math.erf(((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) / math.sqrt(2)))) / 0.08923496   # +0.3%  sjf_4_1_z_d3 < 0.1277
        + 0.002905339 * (10.14805 * 0.5 * ((26.78691 - Q.mass_displaced3) / 10.14805) * (1 + math.erf(((26.78691 - Q.mass_displaced3) / 10.14805) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.2370407 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.2370407 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2)))) / 2.152046   # +0.3%  mass_displaced3 < 26.79 and z_neutral_had < 0.237
        + 0.002639735 * (0.02719315 * 0.5 * ((0.1530389 - Q.sdb_2_z) / 0.02719315) * (1 + math.erf(((0.1530389 - Q.sdb_2_z) / 0.02719315) / math.sqrt(2)))) / 0.01287382   # +0.3%  sdb_2_z < 0.153
        - 0.002434628 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.0) / 1.0) / math.sqrt(2)))) / 0.8926235   # -0.2%  n_s3d_above_3 > 5
        + 0.002395715 * (0.007063147 * 0.5 * ((0.01810676 - Q.z_displaced5) / 0.007063147) * (1 + math.erf(((0.01810676 - Q.z_displaced5) / 0.007063147) / math.sqrt(2)))) / 0.005827334   # +0.2%  z_displaced5 < 0.01811
        + 0.002352085 * (3.153038 * 0.5 * ((Q.mass - 120.653) / 3.153038) * (1 + math.erf(((Q.mass - 120.653) / 3.153038) / math.sqrt(2)))) * (5.085197 * 0.5 * ((10.7744 - Q.jd_3d_5) / 5.085197) * (1 + math.erf(((10.7744 - Q.jd_3d_5) / 5.085197) / math.sqrt(2)))) / 55.06482   # +0.2%  mass > 120.7 and jd_3d_5 < 10.77
        + 0.002224626 * (4.245069 * 0.5 * ((125.2732 - Q.mres_sd_mass_b2z01) / 4.245069) * (1 + math.erf(((125.2732 - Q.mres_sd_mass_b2z01) / 4.245069) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.6073311   # +0.2%  mres_sd_mass_b2z01 < 125.3 and dc_2_n_lep < 0
        - 0.00221851 * (2.952984 * 0.5 * ((75.78208 - Q.sj4_pair_mass_max) / 2.952984) * (1 + math.erf(((75.78208 - Q.sj4_pair_mass_max) / 2.952984) / math.sqrt(2)))) / 8.111448   # -0.2%  sj4_pair_mass_max < 75.78
        - 0.002049931 * (0.560057 * 0.5 * ((1.777787 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.777787 - Q.mass_displaced5) / 0.560057) / math.sqrt(2)))) / 0.8185445   # -0.2%  mass_displaced5 < 1.778
        + 0.001841959 * (15.58693 * 0.5 * ((Q.mass_top40 - 155.8928) / 15.58693) * (1 + math.erf(((Q.mass_top40 - 155.8928) / 15.58693) / math.sqrt(2)))) * (0.01417841 * 0.5 * ((0.1049877 - Q.C2_b2) / 0.01417841) * (1 + math.erf(((0.1049877 - Q.C2_b2) / 0.01417841) / math.sqrt(2)))) / 0.05255163   # +0.2%  mass_top40 > 155.9 and C2_b2 < 0.105
        - 0.001813405 * (1.0 * 0.5 * ((11.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((11.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2)))) * (0.76083 * 0.5 * ((6.306889 - Q.mres_sd_prong_mass2) / 0.76083) * (1 + math.erf(((6.306889 - Q.mres_sd_prong_mass2) / 0.76083) / math.sqrt(2)))) / 4.675332   # -0.2%  sdb_2_n < 11 and mres_sd_prong_mass2 < 6.307
        + 0.001581115 * (1.5 * 0.5 * ((10.0 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((10.0 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2)))) * (0.3391285 * 0.5 * ((Q.max_dr - 0.9206713) / 0.3391285) * (1 + math.erf(((Q.max_dr - 0.9206713) / 0.3391285) / math.sqrt(2)))) / 0.4436428   # +0.2%  n_dr_0p4_up < 10 and max_dr > 0.9207
        + 0.001481089 * (0.09626377 * 0.5 * ((Q.sjf_2_1_z_d3 - 0.3672808) / 0.09626377) * (1 + math.erf(((Q.sjf_2_1_z_d3 - 0.3672808) / 0.09626377) / math.sqrt(2)))) / 0.006253243   # +0.1%  sjf_2_1_z_d3 > 0.3673
        + 0.001478766 * (3.24083 * 0.5 * ((6.161303 - Q.dc_2_jp) / 3.24083) * (1 + math.erf(((6.161303 - Q.dc_2_jp) / 3.24083) / math.sqrt(2)))) / 3.139178   # +0.1%  dc_2_jp < 6.161
        + 0.001458834 * (0.2916549 * 0.5 * ((Q.lne_21 - 1.025706) / 0.2916549) * (1 + math.erf(((Q.lne_21 - 1.025706) / 0.2916549) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((3.0 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2)))) / 0.5686422   # +0.1%  lne_21 > 1.026 and n_lund_kt_above_5 < 3
        - 0.001363311 * (0.006687614 * 0.5 * ((0.02210827 - Q.dr_max_012) / 0.006687614) * (1 + math.erf(((0.02210827 - Q.dr_max_012) / 0.006687614) / math.sqrt(2)))) / 0.001227145   # -0.1%  dr_max_012 < 0.02211
        - 0.001326735 * (0.09779513 * 0.5 * ((0.6703997 - Q.sdb_2_z) / 0.09779513) * (1 + math.erf(((0.6703997 - Q.sdb_2_z) / 0.09779513) / math.sqrt(2)))) * (0.1379966 * 0.5 * ((Q.ak02_dr23 - 0.2385164) / 0.1379966) * (1 + math.erf(((Q.ak02_dr23 - 0.2385164) / 0.1379966) / math.sqrt(2)))) / 0.04421949   # -0.1%  sdb_2_z < 0.6704 and ak02_dr23 > 0.2385
        + 0.001271519 * (0.04905322 * 0.5 * ((0.3377343 - Q.tau32_b2) / 0.04905322) * (1 + math.erf(((0.3377343 - Q.tau32_b2) / 0.04905322) / math.sqrt(2)))) / 0.02418625   # +0.1%  tau32_b2 < 0.3377
        + 0.001262785 * (0.2916549 * 0.5 * ((Q.lne_21 - 1.025706) / 0.2916549) * (1 + math.erf(((Q.lne_21 - 1.025706) / 0.2916549) / math.sqrt(2)))) * (0.08786014 * 0.5 * ((-1.279477 - Q.lnerel_0) / 0.08786014) * (1 + math.erf(((-1.279477 - Q.lnerel_0) / 0.08786014) / math.sqrt(2)))) / 0.4562226   # +0.1%  lne_21 > 1.026 and lnerel_0 < -1.279
        + 0.001242118 * (76.60049 * 0.5 * ((Q.sum_e - 1049.334) / 76.60049) * (1 + math.erf(((Q.sum_e - 1049.334) / 76.60049) / math.sqrt(2)))) / 90.07363   # +0.1%  sum_e > 1049
        - 0.000987774 * (1.0 * 0.5 * ((Q.sjf_4_n2disp - 2.0) / 1.0) * (1 + math.erf(((Q.sjf_4_n2disp - 2.0) / 1.0) / math.sqrt(2)))) / 0.107617   # -0.1%  sjf_4_n2disp > 2
        - 0.0009481272 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2)))) * (2.0 * 0.5 * ((9.0 - Q.n_neutral_had) / 2.0) * (1 + math.erf(((9.0 - Q.n_neutral_had) / 2.0) / math.sqrt(2)))) / 1.043051   # -0.1%  sjf_2_1_n_d3 > 5 and n_neutral_had < 9
        - 0.0007610683 * (0.001766249 * 0.5 * ((0.01724773 - Q.M3_b2) / 0.001766249) * (1 + math.erf(((0.01724773 - Q.M3_b2) / 0.001766249) / math.sqrt(2)))) * (0.004593578 * 0.5 * ((Q.sjf_4_2_z_d3 - 0.004340802) / 0.004593578) * (1 + math.erf(((Q.sjf_4_2_z_d3 - 0.004340802) / 0.004593578) / math.sqrt(2)))) / 0.0001744244   # -0.1%  M3_b2 < 0.01725 and sjf_4_2_z_d3 > 0.004341
        + 0.0006315386 * (1.0 * 0.5 * ((Q.sdb_0_n - 3.0) / 1.0) * (1 + math.erf(((Q.sdb_0_n - 3.0) / 1.0) / math.sqrt(2)))) / 0.2883269   # +0.1%  sdb_0_n > 3
        - 0.0006134384 * (1.0 * 0.5 * ((5.0 - Q.n_charged_pt_above_10) / 1.0) * (1 + math.erf(((5.0 - Q.n_charged_pt_above_10) / 1.0) / math.sqrt(2)))) / 0.2005604   # -0.1%  n_charged_pt_above_10 < 5
        + 0.0005805952 * (5.740575 * 0.5 * ((6.185635 - Q.lep_iso) / 5.740575) * (1 + math.erf(((6.185635 - Q.lep_iso) / 5.740575) / math.sqrt(2)))) / 4.188125   # +0.1%  lep_iso < 6.186
        - 0.0003644558 * (3.311555 * 0.5 * ((80.27571 - Q.mass_top20) / 3.311555) * (1 + math.erf(((80.27571 - Q.mass_top20) / 3.311555) / math.sqrt(2)))) / 6.45104   # -0.0%  mass_top20 < 80.28
        + 0.00029817 * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_10) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_10) / 1.0) / math.sqrt(2)))) / 2.763271   # +0.0%  n_s3d_above_10 < 5
        - 0.0002981626 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.0) / 1.0) / math.sqrt(2)))) * (0.02367418 * 0.5 * ((Q.z_top40_slots - 0.9370976) / 0.02367418) * (1 + math.erf(((Q.z_top40_slots - 0.9370976) / 0.02367418) / math.sqrt(2)))) / 0.03647236   # -0.0%  n_s3d_above_3 > 5 and z_top40_slots > 0.9371
        + 0.0002875426 * (16.47782 * 0.5 * ((162.7874 - Q.mass_top30) / 16.47782) * (1 + math.erf(((162.7874 - Q.mass_top30) / 16.47782) / math.sqrt(2)))) * (0.1300849 * 0.5 * ((Q.dc_tag_2nd - 0.0) / 0.1300849) * (1 + math.erf(((Q.dc_tag_2nd - 0.0) / 0.1300849) / math.sqrt(2)))) / 38.53705   # +0.0%  mass_top30 < 162.8 and dc_tag_2nd > 0
        - 0.0002770048 * (0.06585074 * 0.5 * ((-0.1170754 - Q.tdz_1) / 0.06585074) * (1 + math.erf(((-0.1170754 - Q.tdz_1) / 0.06585074) / math.sqrt(2)))) / 0.01660446   # -0.0%  tdz_1 < -0.1171
        - 0.0002695786 * (10.09897 * 0.5 * ((115.5142 - Q.sj4_pair_mass_max) / 10.09897) * (1 + math.erf(((115.5142 - Q.sj4_pair_mass_max) / 10.09897) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2)))) / 1.349772   # -0.0%  sj4_pair_mass_max < 115.5 and ak02_3_n_lep > 0
        + 0.0002061338 * (3.384707 * 0.5 * ((6.771002 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.771002 - Q.jd_3d_5) / 3.384707) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.nca_kt_above_20 - 1.0) / 1.0) * (1 + math.erf(((Q.nca_kt_above_20 - 1.0) / 1.0) / math.sqrt(2)))) / 1.070093   # +0.0%  jd_3d_5 < 6.771 and nca_kt_above_20 > 1
        + 0.0001745145 * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_10) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_10) / 1.0) / math.sqrt(2)))) * (0.06371583 * 0.5 * ((Q.sv_2_dr - 0.2172495) / 0.06371583) * (1 + math.erf(((Q.sv_2_dr - 0.2172495) / 0.06371583) / math.sqrt(2)))) / 0.04622265   # +0.0%  n_s3d_above_10 < 5 and sv_2_dr > 0.2172
        - 0.0001444734 * (0.02719315 * 0.5 * ((0.1530389 - Q.sdb_2_z) / 0.02719315) * (1 + math.erf(((0.1530389 - Q.sdb_2_z) / 0.02719315) / math.sqrt(2)))) * (0.01728058 * 0.5 * ((Q.eta_4 - -0.06317139) / 0.01728058) * (1 + math.erf(((Q.eta_4 - -0.06317139) / 0.01728058) / math.sqrt(2)))) / 0.001098556   # -0.0%  sdb_2_z < 0.153 and eta_4 > -0.06317
        - 0.0001201278 * (1.0 * 0.5 * ((11.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((11.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2)))) * (26.14378 * 0.5 * ((Q.jd_3d_5 - 53.39461) / 26.14378) * (1 + math.erf(((Q.jd_3d_5 - 53.39461) / 26.14378) / math.sqrt(2)))) / 17.94712   # -0.0%  sdb_2_n < 11 and jd_3d_5 > 53.39
        - 7.572016e-05 * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 5.0) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 5.0) / 1.0) / math.sqrt(2)))) / 0.07765456   # -0.0%  sjf_2_2_n_d3 > 5
        + 5.919698e-05 * (1.0 * 0.5 * ((Q.sdb_0_n - 3.0) / 1.0) * (1 + math.erf(((Q.sdb_0_n - 3.0) / 1.0) / math.sqrt(2)))) * (0.08551811 * 0.5 * ((Q.tdz_10 - 0.1420982) / 0.08551811) * (1 + math.erf(((Q.tdz_10 - 0.1420982) / 0.08551811) / math.sqrt(2)))) / 0.01611687   # +0.0%  sdb_0_n > 3 and tdz_10 > 0.1421
        + 3.449639e-05 * (16.47782 * 0.5 * ((162.7874 - Q.mass_top30) / 16.47782) * (1 + math.erf(((162.7874 - Q.mass_top30) / 16.47782) / math.sqrt(2)))) * (3.793098 * 0.5 * ((Q.mass_displaced5 - 9.380468) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 9.380468) / 3.793098) / math.sqrt(2)))) / 142.6357   # +0.0%  mass_top30 < 162.8 and mass_displaced5 > 9.38
        + 3.099826e-05 * (0.05184471 * 0.5 * ((-0.5637295 - Q.lund1_lndelta) / 0.05184471) * (1 + math.erf(((-0.5637295 - Q.lund1_lndelta) / 0.05184471) / math.sqrt(2)))) / 0.134722   # +0.0%  lund1_lndelta < -0.5637
        + 1.881112e-05 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ismuon_3 - 0.0) / 1e-06) * (1 + math.erf(((Q.ismuon_3 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.003459209   # +0.0%  sjf_2_1_n_d3 > 5 and ismuon_3 > 0
        + 1.601158e-05 * (0.003751777 * 0.5 * ((0.007652966 - Q.z_dr_0_0p05) / 0.003751777) * (1 + math.erf(((0.007652966 - Q.z_dr_0_0p05) / 0.003751777) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.iselectron_11 - 0.0) / 1e-06) * (1 + math.erf(((Q.iselectron_11 - 0.0) / 1e-06) / math.sqrt(2)))) / 1.573661e-05   # +0.0%  z_dr_0_0p05 < 0.007653 and iselectron_11 > 0
        + 1.238306e-06 * (1.5 * 0.5 * ((10.0 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((10.0 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dc_4_mass_disp3 - 0.0) / 1e-06) * (1 + math.erf(((Q.dc_4_mass_disp3 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.1206473   # +0.0%  n_dr_0p4_up < 10 and dc_4_mass_disp3 > 0
    )
    return z


def neuron_84(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.427509e-05
    )
    return z


def neuron_85(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.185526e-07
    )
    return z


def neuron_86(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.709902e-05
    )
    return z


def neuron_87(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.619161e-06
    )
    return z


def neuron_88(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.996977e-07
    )
    return z


def neuron_89(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.735373e-06
    )
    return z


def neuron_90(Q):
    # scale S = 5.469; each line: share * term / its average size
    z = 5.468953 * (-0.002829909
        - 0.1067911 * (0.07843207 * 0.5 * ((0.1275041 - Q.lep_z) / 0.07843207) * (1 + math.erf(((0.1275041 - Q.lep_z) / 0.07843207) / math.sqrt(2)))) / 0.08845467   # -10.7%  lep_z < 0.1275
        + 0.06009847 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) / 0.2702483   # +6.0%  lep_z < 0.3397
        + 0.04993999 * (16.90428 * 0.5 * ((164.4374 - Q.mass) / 16.90428) * (1 + math.erf(((164.4374 - Q.mass) / 16.90428) / math.sqrt(2)))) / 52.34277   # +5.0%  mass < 164.4
        + 0.04080198 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) / 14.13297   # +4.1%  lep_ptrel < 18.77
        - 0.0403258 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (29.76053 * 0.5 * ((44.14912 - Q.lep_iso) / 29.76053) * (1 + math.erf(((44.14912 - Q.lep_iso) / 29.76053) / math.sqrt(2)))) / 1.933715   # -4.0%  lepsj_2_dr < 0.06573 and lep_iso < 44.15
        + 0.03834037 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2)))) / 41.25115   # +3.8%  mass_top40 > 70.88
        + 0.03244989 * (0.08158173 * 0.5 * ((0.3261071 - Q.z_displaced3) / 0.08158173) * (1 + math.erf(((0.3261071 - Q.z_displaced3) / 0.08158173) / math.sqrt(2)))) / 0.2262928   # +3.2%  z_displaced3 < 0.3261
        - 0.03003684 * (3.560123 * 0.5 * ((120.8334 - Q.mres_sd_mass_b1z01) / 3.560123) * (1 + math.erf(((120.8334 - Q.mres_sd_mass_b1z01) / 3.560123) / math.sqrt(2)))) / 26.69326   # -3.0%  mres_sd_mass_b1z01 < 120.8
        - 0.02878716 * (0.08158173 * 0.5 * ((0.3261071 - Q.z_displaced3) / 0.08158173) * (1 + math.erf(((0.3261071 - Q.z_displaced3) / 0.08158173) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2)))) / 33.2928   # -2.9%  z_displaced3 < 0.3261 and jd_3d_4 < 172.9
        + 0.0276796 * (13.85645 * 0.5 * ((137.6422 - Q.mres_sd_mass_b1z01) / 13.85645) * (1 + math.erf(((137.6422 - Q.mres_sd_mass_b1z01) / 13.85645) / math.sqrt(2)))) / 39.59806   # +2.8%  mres_sd_mass_b1z01 < 137.6
        - 0.02433935 * (6.549481 * 0.5 * ((129.1614 - Q.mres_sd_mass_b1z01) / 6.549481) * (1 + math.erf(((129.1614 - Q.mres_sd_mass_b1z01) / 6.549481) / math.sqrt(2)))) / 32.89124   # -2.4%  mres_sd_mass_b1z01 < 129.2
        + 0.0242552 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) / 36.20948   # +2.4%  lep_ptrel < 43.21
        + 0.02368353 * (5.297564 * 0.5 * ((78.89137 - Q.mres_sd_mass_b1z01) / 5.297564) * (1 + math.erf(((78.89137 - Q.mres_sd_mass_b1z01) / 5.297564) / math.sqrt(2)))) / 7.459324   # +2.4%  mres_sd_mass_b1z01 < 78.89
        + 0.02287425 * (1.5 * 0.5 * ((7.0 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.0 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2)))) / 3.650329   # +2.3%  n_s3d_above_3 < 7
        - 0.02228229 * (0.01294944 * 0.5 * ((0.08643515 - Q.tau3) / 0.01294944) * (1 + math.erf(((0.08643515 - Q.tau3) / 0.01294944) / math.sqrt(2)))) / 0.03980155   # -2.2%  tau3 < 0.08644
        + 0.02226608 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) / 0.8420682   # +2.2%  lep_iso < 1.362
        + 0.02118996 * (12.49365 * 0.5 * ((128.0079 - Q.sj4_pair_mass_max) / 12.49365) * (1 + math.erf(((128.0079 - Q.sj4_pair_mass_max) / 12.49365) / math.sqrt(2)))) * (0.06955833 * 0.5 * ((0.3788785 - Q.z_neutral_had) / 0.06955833) * (1 + math.erf(((0.3788785 - Q.z_neutral_had) / 0.06955833) / math.sqrt(2)))) / 11.1099   # +2.1%  sj4_pair_mass_max < 128 and z_neutral_had < 0.3789
        + 0.02048501 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) / 0.04874886   # +2.0%  lepsj_2_dr < 0.06573
        - 0.01848458 * Q.pz_lnd2 / 0.1223855   # -1.8%  pz_lnd2
        - 0.018395 * (0.001626539 * 0.5 * ((1.0 - Q.z_top20_slots) / 0.001626539) * (1 + math.erf(((1.0 - Q.z_top20_slots) / 0.001626539) / math.sqrt(2)))) / 0.1051858   # -1.8%  z_top20_slots < 1
        - 0.01787906 * (5.539613 * 0.5 * ((94.12155 - Q.mres_sd_mass_b1z01) / 5.539613) * (1 + math.erf(((94.12155 - Q.mres_sd_mass_b1z01) / 5.539613) / math.sqrt(2)))) / 12.42898   # -1.8%  mres_sd_mass_b1z01 < 94.12
        + 0.01769564 * (4.902193 * 0.5 * ((126.8853 - Q.mass_top40) / 4.902193) * (1 + math.erf(((126.8853 - Q.mass_top40) / 4.902193) / math.sqrt(2)))) / 23.72702   # +1.8%  mass_top40 < 126.9
        - 0.01743658 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) / 0.4378545   # -1.7%  lep_z < 0.5187
        + 0.01572884 * (0.8047829 * 0.5 * ((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) * (1 + math.erf(((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) / math.sqrt(2)))) / 1.042224   # +1.6%  lepsj_3_maxsd0 < 1.53
        - 0.01414646 * (5.286881 * 0.5 * ((Q.mass - 100.4835) / 5.286881) * (1 + math.erf(((Q.mass - 100.4835) / 5.286881) / math.sqrt(2)))) / 22.55633   # -1.4%  mass > 100.5
        - 0.01285991 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2)))) / 0.001827643   # -1.3%  lep_z < 0.004136
        - 0.01174845 * (18.42178 * 0.5 * ((182.8592 - Q.mass) / 18.42178) * (1 + math.erf(((182.8592 - Q.mass) / 18.42178) / math.sqrt(2)))) / 69.40895   # -1.2%  mass < 182.9
        + 0.011157 * (3.485807 * 0.5 * ((119.1279 - Q.mass_top40) / 3.485807) * (1 + math.erf(((119.1279 - Q.mass_top40) / 3.485807) / math.sqrt(2)))) / 18.30296   # +1.1%  mass_top40 < 119.1
        - 0.009579988 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) / 1.651857   # -1.0%  lepsj_3_n_d3 < 2
        - 0.009519101 * (11.68693 * 0.5 * ((141.9283 - Q.mass_top40) / 11.68693) * (1 + math.erf(((141.9283 - Q.mass_top40) / 11.68693) / math.sqrt(2)))) / 35.72399   # -1.0%  mass_top40 < 141.9
        + 0.008448172 * (1.5 * 0.5 * ((Q.n_for_90pct - 11.0) / 1.5) * (1 + math.erf(((Q.n_for_90pct - 11.0) / 1.5) / math.sqrt(2)))) / 9.938058   # +0.8%  n_for_90pct > 11
        - 0.00817891 * (1.0 * 0.5 * ((10.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((10.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2)))) * (0.7057249 * 0.5 * ((0.6893409 - Q.dc_3_jp) / 0.7057249) * (1 + math.erf(((0.6893409 - Q.dc_3_jp) / 0.7057249) / math.sqrt(2)))) / 0.9358215   # -0.8%  sdb_2_n < 10 and dc_3_jp < 0.6893
        - 0.007685523 * (3.244489 * 0.5 * ((66.70506 - Q.sj4_pair_mass_max) / 3.244489) * (1 + math.erf(((66.70506 - Q.sj4_pair_mass_max) / 3.244489) / math.sqrt(2)))) / 4.710216   # -0.8%  sj4_pair_mass_max < 66.71
        + 0.007260806 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) / 1.215201   # +0.7%  n_dr_0p4_up < 4
        + 0.007050801 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2)))) * (0.03537257 * 0.5 * ((0.9798274 - Q.ak02_1_z) / 0.03537257) * (1 + math.erf(((0.9798274 - Q.ak02_1_z) / 0.03537257) / math.sqrt(2)))) / 13.94398   # +0.7%  mass_top40 > 70.88 and ak02_1_z < 0.9798
        - 0.007015601 * (0.02862925 * 0.5 * ((0.2665611 - Q.tau21) / 0.02862925) * (1 + math.erf(((0.2665611 - Q.tau21) / 0.02862925) / math.sqrt(2)))) / 0.02037111   # -0.7%  tau21 < 0.2666
        - 0.006301919 * (0.02005207 * 0.5 * ((Q.sj3_pairmin_over_m - 0.2674688) / 0.02005207) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.2674688) / 0.02005207) / math.sqrt(2)))) / 0.07858923   # -0.6%  sj3_pairmin_over_m > 0.2675
        - 0.006296009 * (22.11238 * 0.5 * ((34.60954 - Q.sjf_2_1_mass_d3) / 22.11238) * (1 + math.erf(((34.60954 - Q.sjf_2_1_mass_d3) / 22.11238) / math.sqrt(2)))) / 28.6976   # -0.6%  sjf_2_1_mass_d3 < 34.61
        - 0.005923531 * Q.dc_1_n_lep / 0.2450033   # -0.6%  dc_1_n_lep
        + 0.005681955 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.01981371 * 0.5 * ((Q.sj3_pairmin_over_m - 0.2477126) / 0.01981371) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.2477126) / 0.01981371) / math.sqrt(2)))) / 0.02489857   # +0.6%  lep_z < 0.3397 and sj3_pairmin_over_m > 0.2477
        - 0.005422024 * Q.tau21 / 0.4342672   # -0.5%  tau21
        - 0.005123472 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.05461387 * 0.5 * ((0.2184442 - Q.z_displaced5) / 0.05461387) * (1 + math.erf(((0.2184442 - Q.z_displaced5) / 0.05461387) / math.sqrt(2)))) / 5.422994   # -0.5%  lep_ptrel < 43.21 and z_displaced5 < 0.2184
        - 0.005121217 * (0.0278003 * 0.5 * ((0.4239562 - Q.z_charged_had) / 0.0278003) * (1 + math.erf(((0.4239562 - Q.z_charged_had) / 0.0278003) / math.sqrt(2)))) / 0.03851381   # -0.5%  z_charged_had < 0.424
        - 0.00509934 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (0.2012988 * 0.5 * ((0.2012988 - Q.kt2_min12_mass_disp3) / 0.2012988) * (1 + math.erf(((0.2012988 - Q.kt2_min12_mass_disp3) / 0.2012988) / math.sqrt(2)))) / 48.35027   # -0.5%  sip_3d_2 < 447.1 and kt2_min12_mass_disp3 < 0.2013
        + 0.005007362 * (0.01958217 * 0.5 * ((0.3667049 - Q.N2_b05) / 0.01958217) * (1 + math.erf(((0.3667049 - Q.N2_b05) / 0.01958217) / math.sqrt(2)))) / 0.008411412   # +0.5%  N2_b05 < 0.3667
        - 0.004486911 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_3_n2disp - 0.0) / 1.0) * (1 + math.erf(((Q.sjf_3_n2disp - 0.0) / 1.0) / math.sqrt(2)))) / 0.2030189   # -0.4%  lep_z < 0.3397 and sjf_3_n2disp > 0
        - 0.004419117 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) / 2.75999   # -0.4%  max_abs_d0 < 5.812
        + 0.004403984 * (0.1314845 * 0.5 * ((Q.pair_mean_lnm2 - 2.708356) / 0.1314845) * (1 + math.erf(((Q.pair_mean_lnm2 - 2.708356) / 0.1314845) / math.sqrt(2)))) / 0.3872486   # +0.4%  pair_mean_lnm2 > 2.708
        - 0.004222705 * (0.0331444 * 0.5 * ((0.1339658 - Q.z_displaced5) / 0.0331444) * (1 + math.erf(((0.1339658 - Q.z_displaced5) / 0.0331444) / math.sqrt(2)))) / 0.07747698   # -0.4%  z_displaced5 < 0.134
        - 0.003977372 * (0.01203596 * 0.5 * ((0.0318986 - Q.e2) / 0.01203596) * (1 + math.erf(((0.0318986 - Q.e2) / 0.01203596) / math.sqrt(2)))) / 0.0006714177   # -0.4%  e2 < 0.0319
        + 0.003835012 * (0.006899392 * 0.5 * ((0.006470637 - Q.lepsj_2_dr) / 0.006899392) * (1 + math.erf(((0.006470637 - Q.lepsj_2_dr) / 0.006899392) / math.sqrt(2)))) / 0.003371104   # +0.4%  lepsj_2_dr < 0.006471
        + 0.003604459 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.05461387 * 0.5 * ((0.2184442 - Q.z_displaced5) / 0.05461387) * (1 + math.erf(((0.2184442 - Q.z_displaced5) / 0.05461387) / math.sqrt(2)))) / 2.149422   # +0.4%  lep_ptrel < 18.77 and z_displaced5 < 0.2184
        - 0.003580902 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2)))) * (1.739103 * 0.5 * ((3.969624 - Q.jd_3d_4) / 1.739103) * (1 + math.erf(((3.969624 - Q.jd_3d_4) / 1.739103) / math.sqrt(2)))) / 31.12022   # -0.4%  mass_top40 > 70.88 and jd_3d_4 < 3.97
        - 0.003305906 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) / 297.4519   # -0.3%  sip_3d_2 < 447.1
        - 0.002952374 * (4.06561 * 0.5 * ((Q.mass_2charged - 24.4079) / 4.06561) * (1 + math.erf(((Q.mass_2charged - 24.4079) / 4.06561) / math.sqrt(2)))) * (0.1334042 * 0.5 * ((0.5033607 - Q.sv_2_dr) / 0.1334042) * (1 + math.erf(((0.5033607 - Q.sv_2_dr) / 0.1334042) / math.sqrt(2)))) / 1.260942   # -0.3%  mass_2charged > 24.41 and sv_2_dr < 0.5034
        + 0.00277493 * (1.0 * 0.5 * ((10.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((10.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2)))) / 1.81815   # +0.3%  sdb_2_n < 10
        - 0.002693798 * (2.340241 * 0.5 * ((Q.mass_displaced3 - 6.341631) / 2.340241) * (1 + math.erf(((Q.mass_displaced3 - 6.341631) / 2.340241) / math.sqrt(2)))) / 5.210803   # -0.3%  mass_displaced3 > 6.342
        - 0.00264261 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.05691776 * 0.5 * ((Q.jet_charge - 0.2072767) / 0.05691776) * (1 + math.erf(((Q.jet_charge - 0.2072767) / 0.05691776) / math.sqrt(2)))) / 0.003258505   # -0.3%  lep_z < 0.3397 and jet_charge > 0.2073
        + 0.002615255 * (0.06646541 * 0.5 * ((0.3206143 - Q.pt1_over_pt0) / 0.06646541) * (1 + math.erf(((0.3206143 - Q.pt1_over_pt0) / 0.06646541) / math.sqrt(2)))) / 0.01895531   # +0.3%  pt1_over_pt0 < 0.3206
        + 0.002599581 * (0.04445521 * 0.5 * ((0.2686235 - Q.LHA) / 0.04445521) * (1 + math.erf(((0.2686235 - Q.LHA) / 0.04445521) / math.sqrt(2)))) / 0.007316391   # +0.3%  LHA < 0.2686
        + 0.002578129 * (0.06042313 * 0.5 * ((Q.sjq_3_sumabs_k1 - 0.4372817) / 0.06042313) * (1 + math.erf(((Q.sjq_3_sumabs_k1 - 0.4372817) / 0.06042313) / math.sqrt(2)))) / 0.09359343   # +0.3%  sjq_3_sumabs_k1 > 0.4373
        + 0.002298542 * (1.0 * 0.5 * ((Q.n_charged_had - 18.0) / 1.0) * (1 + math.erf(((Q.n_charged_had - 18.0) / 1.0) / math.sqrt(2)))) / 3.361264   # +0.2%  n_charged_had > 18
        + 0.002148228 * (0.03675411 * 0.5 * ((Q.z_neutral - 0.570175) / 0.03675411) * (1 + math.erf(((Q.z_neutral - 0.570175) / 0.03675411) / math.sqrt(2)))) / 0.01347288   # +0.2%  z_neutral > 0.5702
        - 0.002043204 * (0.01268163 * 0.5 * ((0.02077199 - Q.mres_sd_rg_b0z02) / 0.01268163) * (1 + math.erf(((0.02077199 - Q.mres_sd_rg_b0z02) / 0.01268163) / math.sqrt(2)))) / 0.001595095   # -0.2%  mres_sd_rg_b0z02 < 0.02077
        - 0.002020024 * (43.0 * 0.5 * ((366.0 - Q.n_pairs_kt_above_1) / 43.0) * (1 + math.erf(((366.0 - Q.n_pairs_kt_above_1) / 43.0) / math.sqrt(2)))) / 128.9288   # -0.2%  n_pairs_kt_above_1 < 366
        - 0.001970699 * (0.08158173 * 0.5 * ((0.3261071 - Q.z_displaced3) / 0.08158173) * (1 + math.erf(((0.3261071 - Q.z_displaced3) / 0.08158173) / math.sqrt(2)))) * (0.0408915 * 0.5 * ((0.04607888 - Q.lep_dr) / 0.0408915) * (1 + math.erf(((0.04607888 - Q.lep_dr) / 0.0408915) / math.sqrt(2)))) / 0.00586613   # -0.2%  z_displaced3 < 0.3261 and lep_dr < 0.04608
        - 0.001968168 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2)))) * (0.007119697 * 0.5 * ((0.177312 - Q.M2_b05) / 0.007119697) * (1 + math.erf(((0.177312 - Q.M2_b05) / 0.007119697) / math.sqrt(2)))) / 1.674327   # -0.2%  mass_top40 > 70.88 and M2_b05 < 0.1773
        - 0.001934355 * (7.099104e-05 * 0.5 * ((0.0003489585 - Q.e3) / 7.099104e-05) * (1 + math.erf(((0.0003489585 - Q.e3) / 7.099104e-05) / math.sqrt(2)))) / 3.156875e-05   # -0.2%  e3 < 0.000349
        - 0.001910967 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (0.005517122 * 0.5 * ((0.9919867 - Q.z_top40_slots) / 0.005517122) * (1 + math.erf(((0.9919867 - Q.z_top40_slots) / 0.005517122) / math.sqrt(2)))) / 0.0006927117   # -0.2%  lepsj_2_dr < 0.06573 and z_top40_slots < 0.992
        - 0.0017603 * (12.49365 * 0.5 * ((128.0079 - Q.sj4_pair_mass_max) / 12.49365) * (1 + math.erf(((128.0079 - Q.sj4_pair_mass_max) / 12.49365) / math.sqrt(2)))) / 47.96266   # -0.2%  sj4_pair_mass_max < 128
        + 0.001657645 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) / math.sqrt(2)))) / 0.01794643   # +0.2%  sjq_3_3_k1 > 0.6536
        - 0.001530269 * (0.04445521 * 0.5 * ((0.2686235 - Q.LHA) / 0.04445521) * (1 + math.erf(((0.2686235 - Q.LHA) / 0.04445521) / math.sqrt(2)))) * (0.5801757 * 0.5 * ((1.544662 - Q.N3_b2) / 0.5801757) * (1 + math.erf(((1.544662 - Q.N3_b2) / 0.5801757) / math.sqrt(2)))) / 0.003989983   # -0.2%  LHA < 0.2686 and N3_b2 < 1.545
        + 0.001343071 * (4.06561 * 0.5 * ((Q.mass_2charged - 24.4079) / 4.06561) * (1 + math.erf(((Q.mass_2charged - 24.4079) / 4.06561) / math.sqrt(2)))) / 2.96147   # +0.1%  mass_2charged > 24.41
        + 0.001256848 * (12.49365 * 0.5 * ((128.0079 - Q.sj4_pair_mass_max) / 12.49365) * (1 + math.erf(((128.0079 - Q.sj4_pair_mass_max) / 12.49365) / math.sqrt(2)))) * (540.2081 * 0.5 * ((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2)))) / 22749.79   # +0.1%  sj4_pair_mass_max < 128 and lepsj_3_maxsd0 < 587
        - 0.001227864 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (1.0 * 0.5 * ((6.0 - Q.n_neutral_had) / 1.0) * (1 + math.erf(((6.0 - Q.n_neutral_had) / 1.0) / math.sqrt(2)))) / 1.862681   # -0.1%  lep_iso < 1.362 and n_neutral_had < 6
        - 0.001195537 * (0.1314845 * 0.5 * ((Q.pair_mean_lnm2 - 2.708356) / 0.1314845) * (1 + math.erf(((Q.pair_mean_lnm2 - 2.708356) / 0.1314845) / math.sqrt(2)))) * (0.0330605 * 0.5 * ((Q.lep_dr - 0.114659) / 0.0330605) * (1 + math.erf(((Q.lep_dr - 0.114659) / 0.0330605) / math.sqrt(2)))) / 0.02029191   # -0.1%  pair_mean_lnm2 > 2.708 and lep_dr > 0.1147
        + 0.001100374 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2)))) * (0.2508028 * 0.5 * ((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2)))) / 0.6761457   # +0.1%  mass_top40 > 70.88 and sjq_3_3_k1 < -0.655
        - 0.0009603171 * (0.01869595 * 0.5 * ((Q.LHA - 0.5052443) / 0.01869595) * (1 + math.erf(((Q.LHA - 0.5052443) / 0.01869595) / math.sqrt(2)))) / 0.007621843   # -0.1%  LHA > 0.5052
        + 0.0008969674 * (0.002177355 * 0.5 * ((0.01919357 - Q.M3_b2) / 0.002177355) * (1 + math.erf(((0.01919357 - Q.M3_b2) / 0.002177355) / math.sqrt(2)))) * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) / 0.0008488608   # +0.1%  M3_b2 < 0.01919 and sjq_2_prod_k1 < 0.09803
        + 0.0008037019 * (0.003868055 * 0.5 * ((Q.psi_0p3 - 0.9925964) / 0.003868055) * (1 + math.erf(((Q.psi_0p3 - 0.9925964) / 0.003868055) / math.sqrt(2)))) / 0.0007066244   # +0.1%  psi_0p3 > 0.9926
        + 0.0007750195 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_3_2_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_3_2_n_d3 - 1.0) / 1.0) / math.sqrt(2)))) / 162.0062   # +0.1%  sip_3d_2 < 447.1 and sjf_3_2_n_d3 > 1
        + 0.0007449454 * (5.286881 * 0.5 * ((Q.mass - 100.4835) / 5.286881) * (1 + math.erf(((Q.mass - 100.4835) / 5.286881) / math.sqrt(2)))) * (0.006231177 * 0.5 * ((0.03947764 - Q.sj4_zsoft) / 0.006231177) * (1 + math.erf(((0.03947764 - Q.sj4_zsoft) / 0.006231177) / math.sqrt(2)))) / 0.1594808   # +0.1%  mass > 100.5 and sj4_zsoft < 0.03948
        - 0.0007101834 * (5.297564 * 0.5 * ((78.89137 - Q.mres_sd_mass_b1z01) / 5.297564) * (1 + math.erf(((78.89137 - Q.mres_sd_mass_b1z01) / 5.297564) / math.sqrt(2)))) * (0.7811923 * 0.5 * ((6.006437 - Q.sj3_mass2) / 0.7811923) * (1 + math.erf(((6.006437 - Q.sj3_mass2) / 0.7811923) / math.sqrt(2)))) / 12.18113   # -0.1%  mres_sd_mass_b1z01 < 78.89 and sj3_mass2 < 6.006
        - 0.0006320473 * (0.06215671 * 0.5 * ((Q.sj4_dr_min - 0.3112717) / 0.06215671) * (1 + math.erf(((Q.sj4_dr_min - 0.3112717) / 0.06215671) / math.sqrt(2)))) / 0.004539938   # -0.1%  sj4_dr_min > 0.3113
        + 0.0006236098 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (0.001294243 * 0.5 * ((Q.sdb_5_z - 0.0) / 0.001294243) * (1 + math.erf(((Q.sdb_5_z - 0.0) / 0.001294243) / math.sqrt(2)))) / 0.001571837   # +0.1%  lepsj_2_dr < 0.06573 and sdb_5_z > 0
        - 0.000593941 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2)))) * (4.522214 * 0.5 * ((Q.dc_split2_kt - 13.50797) / 4.522214) * (1 + math.erf(((Q.dc_split2_kt - 13.50797) / 4.522214) / math.sqrt(2)))) / 530.8062   # -0.1%  mass_top40 > 70.88 and dc_split2_kt > 13.51
        + 0.0005914013 * (4.902193 * 0.5 * ((126.8853 - Q.mass_top40) / 4.902193) * (1 + math.erf(((126.8853 - Q.mass_top40) / 4.902193) / math.sqrt(2)))) * (0.02635594 * 0.5 * ((Q.eccentricity - 0.7995541) / 0.02635594) * (1 + math.erf(((Q.eccentricity - 0.7995541) / 0.02635594) / math.sqrt(2)))) / 1.382039   # +0.1%  mass_top40 < 126.9 and eccentricity > 0.7996
        + 0.0005721134 * (0.002177355 * 0.5 * ((0.01919357 - Q.M3_b2) / 0.002177355) * (1 + math.erf(((0.01919357 - Q.M3_b2) / 0.002177355) / math.sqrt(2)))) / 0.007625123   # +0.1%  M3_b2 < 0.01919
        - 0.0005596518 * (0.01958217 * 0.5 * ((0.3667049 - Q.N2_b05) / 0.01958217) * (1 + math.erf(((0.3667049 - Q.N2_b05) / 0.01958217) / math.sqrt(2)))) * (0.03181498 * 0.5 * ((Q.z_charged_had - 0.3945478) / 0.03181498) * (1 + math.erf(((Q.z_charged_had - 0.3945478) / 0.03181498) / math.sqrt(2)))) / 0.0005423004   # -0.1%  N2_b05 < 0.3667 and z_charged_had > 0.3945
        - 0.0004755193 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 5.193195   # -0.0%  n_pairs_kt_above_1 < 80
        - 0.0004248439 * (0.1228932 * 0.5 * ((3.521178 - Q.lund_max_lnkt) / 0.1228932) * (1 + math.erf(((3.521178 - Q.lund_max_lnkt) / 0.1228932) / math.sqrt(2)))) / 0.1681741   # -0.0%  lund_max_lnkt < 3.521
        + 0.0003947619 * (0.8047829 * 0.5 * ((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) * (1 + math.erf(((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_ntag - 0.0) / 1.0) * (1 + math.erf(((Q.dc_ntag - 0.0) / 1.0) / math.sqrt(2)))) / 0.2121719   # +0.0%  lepsj_3_maxsd0 < 1.53 and dc_ntag > 0
        - 0.0003703712 * (1.5 * 0.5 * ((7.0 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.0 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_1_n_lep - 0.0) / 1.0) * (1 + math.erf(((Q.dc_1_n_lep - 0.0) / 1.0) / math.sqrt(2)))) / 0.6753784   # -0.0%  n_s3d_above_3 < 7 and dc_1_n_lep > 0
        + 0.0003597088 * (0.001059952 * 0.5 * ((0.001673521 - Q.sum_z_dr2_top3) / 0.001059952) * (1 + math.erf(((0.001673521 - Q.sum_z_dr2_top3) / 0.001059952) / math.sqrt(2)))) / 9.400143e-05   # +0.0%  sum_z_dr2_top3 < 0.001674
        + 0.0002504287 * (3.560123 * 0.5 * ((120.8334 - Q.mres_sd_mass_b1z01) / 3.560123) * (1 + math.erf(((120.8334 - Q.mres_sd_mass_b1z01) / 3.560123) / math.sqrt(2)))) * (0.01029703 * 0.5 * ((-0.009407043 - Q.phi_1) / 0.01029703) * (1 + math.erf(((-0.009407043 - Q.phi_1) / 0.01029703) / math.sqrt(2)))) / 0.5685484   # +0.0%  mres_sd_mass_b1z01 < 120.8 and phi_1 < -0.009407
        - 0.0001814075 * (0.01958217 * 0.5 * ((0.3667049 - Q.N2_b05) / 0.01958217) * (1 + math.erf(((0.3667049 - Q.N2_b05) / 0.01958217) / math.sqrt(2)))) * (0.03825143 * 0.5 * ((Q.sjq_2_prod_k03 - 0.06059953) / 0.03825143) * (1 + math.erf(((Q.sjq_2_prod_k03 - 0.06059953) / 0.03825143) / math.sqrt(2)))) / 0.0007294772   # -0.0%  N2_b05 < 0.3667 and sjq_2_prod_k03 > 0.0606
        + 7.228337e-05 * (5.286881 * 0.5 * ((Q.mass - 100.4835) / 5.286881) * (1 + math.erf(((Q.mass - 100.4835) / 5.286881) / math.sqrt(2)))) * (0.07571368 * 0.5 * ((-0.1329965 - Q.tdz_13) / 0.07571368) * (1 + math.erf(((-0.1329965 - Q.tdz_13) / 0.07571368) / math.sqrt(2)))) / 0.6544268   # +0.0%  mass > 100.5 and tdz_13 < -0.133
        + 3.979168e-05 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (3.793098 * 0.5 * ((Q.mass_displaced5 - 9.380468) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 9.380468) / 3.793098) / math.sqrt(2)))) / 0.1400313   # +0.0%  lepsj_2_dr < 0.06573 and mass_displaced5 > 9.38
        - 2.32409e-05 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) / math.sqrt(2)))) * (0.1175438 * 0.5 * ((Q.dc_3_charge - 0.0) / 0.1175438) * (1 + math.erf(((Q.dc_3_charge - 0.0) / 0.1175438) / math.sqrt(2)))) / 0.002946696   # -0.0%  sjq_3_3_k1 > 0.6536 and dc_3_charge > 0
        + 1.046876e-05 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.01429749 * 0.5 * ((-0.03515625 - Q.phi_3) / 0.01429749) * (1 + math.erf(((-0.03515625 - Q.phi_3) / 0.01429749) / math.sqrt(2)))) / 0.4269498   # +0.0%  lep_ptrel < 18.77 and phi_3 < -0.03516
    )
    return z


def neuron_91(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.992636e-06
    )
    return z


def neuron_92(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.72473e-06
    )
    return z


def neuron_93(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.464408e-07
    )
    return z


def neuron_94(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.322168e-06
    )
    return z


def neuron_95(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.369498e-07
    )
    return z


def neuron_96(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.278005e-07
    )
    return z


def neuron_97(Q):
    # scale S = 14.12; each line: share * term / its average size
    z = 14.11976 * (-0.1037192
        - 0.1746002 * (6.036668 * 0.5 * ((Q.mres_sd_mass_b2z01 - 76.1529) / 6.036668) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 76.1529) / 6.036668) / math.sqrt(2)))) / 35.08087   # -17.5%  mres_sd_mass_b2z01 > 76.15
        + 0.1185772 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) / math.sqrt(2)))) / 24.25944   # +11.9%  mres_sd_mass_b2z01 > 91.15
        + 0.08358079 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) / math.sqrt(2)))) / 52.69014   # +8.4%  mres_sd_mass_b2z01 > 55.13
        + 0.04460884 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) / math.sqrt(2)))) / 7.226273   # +4.5%  mres_sd_mass_b2z01 > 130.1
        + 0.0396135 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) / 31.66395   # +4.0%  mass_displaced3 < 39.1
        - 0.03717824 * (14.05251 * 0.5 * ((Q.mres_sd_mass_b2z01 - 139.2565) / 14.05251) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 139.2565) / 14.05251) / math.sqrt(2)))) / 5.947884   # -3.7%  mres_sd_mass_b2z01 > 139.3
        - 0.03593446 * (10.51039 * 0.5 * ((Q.mres_sd_mass_b2z01 - 69.12133) / 10.51039) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 69.12133) / 10.51039) / math.sqrt(2)))) / 40.76287   # -3.6%  mres_sd_mass_b2z01 > 69.12
        + 0.03064715 * (10.53967 * 0.5 * ((93.95351 - Q.nca_sj4_pair_mass_2nd) / 10.53967) * (1 + math.erf(((93.95351 - Q.nca_sj4_pair_mass_2nd) / 10.53967) / math.sqrt(2)))) / 37.70718   # +3.1%  nca_sj4_pair_mass_2nd < 93.95
        - 0.03043872 * (5.5191 * 0.5 * ((Q.mres_pruned_mass - 76.10016) / 5.5191) * (1 + math.erf(((Q.mres_pruned_mass - 76.10016) / 5.5191) / math.sqrt(2)))) / 29.05566   # -3.0%  mres_pruned_mass > 76.1
        + 0.02492719 * (0.01301382 * 0.5 * ((0.1193588 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.1193588 - Q.tau2) / 0.01301382) / math.sqrt(2)))) / 0.04837718   # +2.5%  tau2 < 0.1194
        - 0.02155086 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) / 297.4519   # -2.2%  sip_3d_2 < 447.1
        - 0.02103157 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) / 1.677331   # -2.1%  n_s3d_above_3 > 3
        + 0.02021088 * (6.875137 * 0.5 * ((18.80005 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.80005 - Q.mass_displaced3) / 6.875137) / math.sqrt(2)))) / 13.2945   # +2.0%  mass_displaced3 < 18.8
        + 0.01746464 * (4.883139 * 0.5 * ((Q.mass_top50 - 89.68964) / 4.883139) * (1 + math.erf(((Q.mass_top50 - 89.68964) / 4.883139) / math.sqrt(2)))) / 28.47564   # +1.7%  mass_top50 > 89.69
        + 0.01677335 * (1.0 * 0.5 * ((Q.n_sd0_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_sd0_above_3 - 2.0) / 1.0) / math.sqrt(2)))) / 1.832491   # +1.7%  n_sd0_above_3 > 2
        - 0.01528745 * (22.10989 * 0.5 * ((171.3819 - Q.mres_pruned_mass) / 22.10989) * (1 + math.erf(((171.3819 - Q.mres_pruned_mass) / 22.10989) / math.sqrt(2)))) / 78.04698   # -1.5%  mres_pruned_mass < 171.4
        + 0.01487691 * (5.286881 * 0.5 * ((100.4835 - Q.mass) / 5.286881) * (1 + math.erf(((100.4835 - Q.mass) / 5.286881) / math.sqrt(2)))) / 8.121469   # +1.5%  mass < 100.5
        - 0.01444389 * (2.991814 * 0.5 * ((65.88119 - Q.nca_sj4_pair_mass_2nd) / 2.991814) * (1 + math.erf(((65.88119 - Q.nca_sj4_pair_mass_2nd) / 2.991814) / math.sqrt(2)))) / 13.78938   # -1.4%  nca_sj4_pair_mass_2nd < 65.88
        - 0.01431721 * (3.838712e-05 * 0.5 * ((0.000125186 - Q.e3_b2) / 3.838712e-05) * (1 + math.erf(((0.000125186 - Q.e3_b2) / 3.838712e-05) / math.sqrt(2)))) / 6.920782e-05   # -1.4%  e3_b2 < 0.0001252
        + 0.01318768 * (5.725739 * 0.5 * ((Q.mres_pruned_mass - 124.3145) / 5.725739) * (1 + math.erf(((Q.mres_pruned_mass - 124.3145) / 5.725739) / math.sqrt(2)))) / 6.835692   # +1.3%  mres_pruned_mass > 124.3
        - 0.01235746 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2)))) / 0.04076886   # -1.2%  e3_b2 < 0.0004127 and jd_3d_4 < 172.9
        - 0.01222351 * (5.294312 * 0.5 * ((Q.mres_sd_mass_b2z01 - 102.264) / 5.294312) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 102.264) / 5.294312) / math.sqrt(2)))) / 17.59744   # -1.2%  mres_sd_mass_b2z01 > 102.3
        + 0.01165793 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 0.001892503   # +1.2%  e3_b2 < 0.0004127 and n_s3d_above_3 < 10
        - 0.01135863 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) / 446.9928   # -1.1%  mass_displaced3 < 39.1 and lep_ptrel < 18.77
        + 0.01128676 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.929175 * 0.5 * ((4.004982 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((4.004982 - Q.jd_3d_5) / 1.929175) / math.sqrt(2)))) / 450.0733   # +1.1%  sip_3d_2 < 447.1 and jd_3d_5 < 4.005
        + 0.01088867 * (4.879578 * 0.5 * ((121.0623 - Q.mass_top30) / 4.879578) * (1 + math.erf(((121.0623 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) / 23.19549   # +1.1%  mass_top30 < 121.1
        + 0.009492972 * (35.5 * 0.5 * ((288.0 - Q.n_pairs_kt_above_1) / 35.5) * (1 + math.erf(((288.0 - Q.n_pairs_kt_above_1) / 35.5) / math.sqrt(2)))) / 81.89092   # +0.9%  n_pairs_kt_above_1 < 288
        - 0.008823808 * (175.2971 * 0.5 * ((84.32402 - Q.sv_2_sd0_sum) / 175.2971) * (1 + math.erf(((84.32402 - Q.sv_2_sd0_sum) / 175.2971) / math.sqrt(2)))) / 47.14683   # -0.9%  sv_2_sd0_sum < 84.32
        + 0.00838229 * (3.289335 * 0.5 * ((Q.mass - 123.7928) / 3.289335) * (1 + math.erf(((Q.mass - 123.7928) / 3.289335) / math.sqrt(2)))) / 10.58264   # +0.8%  mass > 123.8
        - 0.006840991 * (16.90428 * 0.5 * ((Q.mass - 164.4374) / 16.90428) * (1 + math.erf(((Q.mass - 164.4374) / 16.90428) / math.sqrt(2)))) / 3.240687   # -0.7%  mass > 164.4
        + 0.006790002 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.2370407 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.2370407 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2)))) / 5.144219   # +0.7%  mres_sd_mass_b2z01 > 55.13 and z_neutral_had < 0.237
        - 0.006645564 * (0.01301382 * 0.5 * ((0.1193588 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.1193588 - Q.tau2) / 0.01301382) / math.sqrt(2)))) * (0.05096652 * 0.5 * ((0.125305 - Q.sjq_2_prod_k05) / 0.05096652) * (1 + math.erf(((0.125305 - Q.sjq_2_prod_k05) / 0.05096652) / math.sqrt(2)))) / 0.009914395   # -0.7%  tau2 < 0.1194 and sjq_2_prod_k05 < 0.1253
        - 0.005791633 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) / 0.0002870086   # -0.6%  e3_b2 < 0.0004127
        - 0.005608027 * (1.0 * 0.5 * ((6.0 - Q.n_neutral_had) / 1.0) * (1 + math.erf(((6.0 - Q.n_neutral_had) / 1.0) / math.sqrt(2)))) / 2.176559   # -0.6%  n_neutral_had < 6
        + 0.005471345 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.08904938 * 0.5 * ((Q.jet_charge_k03 - -0.05990128) / 0.08904938) * (1 + math.erf(((Q.jet_charge_k03 - -0.05990128) / 0.08904938) / math.sqrt(2)))) / 9.108633e-05   # +0.5%  e3_b2 < 0.0004127 and jet_charge_k03 > -0.0599
        + 0.005198026 * (0.02188437 * 0.5 * ((Q.z_top30_slots - 0.8942764) / 0.02188437) * (1 + math.erf(((Q.z_top30_slots - 0.8942764) / 0.02188437) / math.sqrt(2)))) / 0.06859716   # +0.5%  z_top30_slots > 0.8943
        + 0.004436211 * (0.1277784 * 0.5 * ((Q.lund3_lndelta - -1.902701) / 0.1277784) * (1 + math.erf(((Q.lund3_lndelta - -1.902701) / 0.1277784) / math.sqrt(2)))) / 0.5257705   # +0.4%  lund3_lndelta > -1.903
        + 0.004021748 * (1.0 * 0.5 * ((1.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.6885179   # +0.4%  ak02_2_n_lep < 1
        + 0.003920551 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (0.03286901 * 0.5 * ((0.4839362 - Q.ak02_dr12) / 0.03286901) * (1 + math.erf(((0.4839362 - Q.ak02_dr12) / 0.03286901) / math.sqrt(2)))) / 4.76361   # +0.4%  mass_displaced3 < 39.1 and ak02_dr12 < 0.4839
        - 0.003914215 * (0.01301382 * 0.5 * ((0.1193588 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.1193588 - Q.tau2) / 0.01301382) / math.sqrt(2)))) * (0.08093679 * 0.5 * ((0.2801368 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.2801368 - Q.z_displaced5) / 0.08093679) / math.sqrt(2)))) / 0.009465605   # -0.4%  tau2 < 0.1194 and z_displaced5 < 0.2801
        + 0.003787454 * (1.0 * 0.5 * ((1.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2)))) * (0.1648594 * 0.5 * ((Q.D2 - 0.7107791) / 0.1648594) * (1 + math.erf(((Q.D2 - 0.7107791) / 0.1648594) / math.sqrt(2)))) / 1.085578   # +0.4%  ak02_2_n_lep < 1 and D2 > 0.7108
        - 0.003707217 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 46.72742   # -0.4%  mres_sd_mass_b2z01 > 91.15 and jd_3d_6 < 5.368
        - 0.003675901 * (6.469856 * 0.5 * ((Q.mass_top50 - 79.27954) / 6.469856) * (1 + math.erf(((Q.mass_top50 - 79.27954) / 6.469856) / math.sqrt(2)))) / 36.77225   # -0.4%  mass_top50 > 79.28
        + 0.003598799 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (23.09414 * 0.5 * ((Q.sum_pt_top5 - 219.4219) / 23.09414) * (1 + math.erf(((Q.sum_pt_top5 - 219.4219) / 23.09414) / math.sqrt(2)))) / 4906.65   # +0.4%  mass_displaced3 < 39.1 and sum_pt_top5 > 219.4
        - 0.003298768 * (4.879578 * 0.5 * ((121.0623 - Q.mass_top30) / 4.879578) * (1 + math.erf(((121.0623 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 1.0) / 1.0) * (1 + math.erf(((Q.n_muon - 1.0) / 1.0) / math.sqrt(2)))) / 3.402555   # -0.3%  mass_top30 < 121.1 and n_muon > 1
        - 0.002573782 * (0.0331444 * 0.5 * ((Q.z_displaced5 - 0.1339658) / 0.0331444) * (1 + math.erf(((Q.z_displaced5 - 0.1339658) / 0.0331444) / math.sqrt(2)))) / 0.03695703   # -0.3%  z_displaced5 > 0.134
        + 0.002494245 * (6.93786 * 0.5 * ((Q.sj3_pair_mass_max - 120.6471) / 6.93786) * (1 + math.erf(((Q.sj3_pair_mass_max - 120.6471) / 6.93786) / math.sqrt(2)))) / 3.456044   # +0.2%  sj3_pair_mass_max > 120.6
        + 0.002235717 * (0.1256484 * 0.5 * ((Q.pair_mean_lnm2 - 2.09826) / 0.1256484) * (1 + math.erf(((Q.pair_mean_lnm2 - 2.09826) / 0.1256484) / math.sqrt(2)))) / 0.7075889   # +0.2%  pair_mean_lnm2 > 2.098
        - 0.00221681 * (3.93783 * 0.5 * ((73.24742 - Q.sj3_pair_mass_max) / 3.93783) * (1 + math.erf(((73.24742 - Q.sj3_pair_mass_max) / 3.93783) / math.sqrt(2)))) / 4.292878   # -0.2%  sj3_pair_mass_max < 73.25
        + 0.002085084 * (0.0297357 * 0.5 * ((0.6513932 - Q.tau32) / 0.0297357) * (1 + math.erf(((0.6513932 - Q.tau32) / 0.0297357) / math.sqrt(2)))) / 0.06846556   # +0.2%  tau32 < 0.6514
        - 0.001999043 * (9.332349 * 0.5 * ((Q.lnpt_31 - -0.1515499) / 9.332349) * (1 + math.erf(((Q.lnpt_31 - -0.1515499) / 9.332349) / math.sqrt(2)))) / 0.6109509   # -0.2%  lnpt_31 > -0.1515
        - 0.00168566 * (0.02298615 * 0.5 * ((0.2740506 - Q.sdb_2_z) / 0.02298615) * (1 + math.erf(((0.2740506 - Q.sdb_2_z) / 0.02298615) / math.sqrt(2)))) / 0.05180903   # -0.2%  sdb_2_z < 0.2741
        - 0.001630014 * (0.06488792 * 0.5 * ((Q.dc_split2_dr - 0.2806799) / 0.06488792) * (1 + math.erf(((Q.dc_split2_dr - 0.2806799) / 0.06488792) / math.sqrt(2)))) / 0.01066673   # -0.2%  dc_split2_dr > 0.2807
        + 0.00156995 * (5.286881 * 0.5 * ((100.4835 - Q.mass) / 5.286881) * (1 + math.erf(((100.4835 - Q.mass) / 5.286881) / math.sqrt(2)))) * (1.518442 * 0.5 * ((1.477152 - Q.lep_ptrel) / 1.518442) * (1 + math.erf(((1.477152 - Q.lep_ptrel) / 1.518442) / math.sqrt(2)))) / 6.990026   # +0.2%  mass < 100.5 and lep_ptrel < 1.477
        - 0.001491285 * (2.328125 * 0.5 * ((6.402344 - Q.max_abs_dz) / 2.328125) * (1 + math.erf(((6.402344 - Q.max_abs_dz) / 2.328125) / math.sqrt(2)))) / 3.297216   # -0.1%  max_abs_dz < 6.402
        + 0.001412565 * (5.119423 * 0.5 * ((Q.mass_top5 - 62.84493) / 5.119423) * (1 + math.erf(((Q.mass_top5 - 62.84493) / 5.119423) / math.sqrt(2)))) / 3.722504   # +0.1%  mass_top5 > 62.84
        - 0.001348007 * (20.07181 * 0.5 * ((Q.mres_sd_mass_b1z01 - 176.9461) / 20.07181) * (1 + math.erf(((Q.mres_sd_mass_b1z01 - 176.9461) / 20.07181) / math.sqrt(2)))) / 1.944249   # -0.1%  mres_sd_mass_b1z01 > 176.9
        + 0.001295913 * (19.15449 * 0.5 * ((Q.mres_sd_mass_b0z02 - 134.2023) / 19.15449) * (1 + math.erf(((Q.mres_sd_mass_b0z02 - 134.2023) / 19.15449) / math.sqrt(2)))) / 4.52144   # +0.1%  mres_sd_mass_b0z02 > 134.2
        + 0.001253292 * (4.879578 * 0.5 * ((121.0623 - Q.mass_top30) / 4.879578) * (1 + math.erf(((121.0623 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (52.10352 * 0.5 * ((Q.sum_e - 926.2598) / 52.10352) * (1 + math.erf(((Q.sum_e - 926.2598) / 52.10352) / math.sqrt(2)))) / 2981.252   # +0.1%  mass_top30 < 121.1 and sum_e > 926.3
        - 0.001251841 * (0.1064494 * 0.5 * ((3.634108 - Q.lund_max_lnkt) / 0.1064494) * (1 + math.erf(((3.634108 - Q.lund_max_lnkt) / 0.1064494) / math.sqrt(2)))) / 0.199093   # -0.1%  lund_max_lnkt < 3.634
        + 0.001232402 * (22.10989 * 0.5 * ((171.3819 - Q.mres_pruned_mass) / 22.10989) * (1 + math.erf(((171.3819 - Q.mres_pruned_mass) / 22.10989) / math.sqrt(2)))) * (0.1367866 * 0.5 * ((Q.mass_2photon - 0.2753928) / 0.1367866) * (1 + math.erf(((Q.mass_2photon - 0.2753928) / 0.1367866) / math.sqrt(2)))) / 334.2153   # +0.1%  mres_pruned_mass < 171.4 and mass_2photon > 0.2754
        - 0.001150786 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.983043) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.983043) / 4.308625) / math.sqrt(2)))) / 5.404766   # -0.1%  lep_ptrel > 6.983
        + 0.001127491 * (2.077045 * 0.5 * ((1.851735 - Q.sj2_mass2) / 2.077045) * (1 + math.erf(((1.851735 - Q.sj2_mass2) / 2.077045) / math.sqrt(2)))) / 0.08951668   # +0.1%  sj2_mass2 < 1.852
        - 0.0008558667 * (0.1256484 * 0.5 * ((Q.pair_mean_lnm2 - 2.09826) / 0.1256484) * (1 + math.erf(((Q.pair_mean_lnm2 - 2.09826) / 0.1256484) / math.sqrt(2)))) * (0.1068554 * 0.5 * ((0.4825848 - Q.dr_28) / 0.1068554) * (1 + math.erf(((0.4825848 - Q.dr_28) / 0.1068554) / math.sqrt(2)))) / 0.2379703   # -0.1%  pair_mean_lnm2 > 2.098 and dr_28 < 0.4826
        - 0.0008509897 * (3.838712e-05 * 0.5 * ((0.000125186 - Q.e3_b2) / 3.838712e-05) * (1 + math.erf(((0.000125186 - Q.e3_b2) / 3.838712e-05) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_ntag - 0.0) / 1.0) * (1 + math.erf(((Q.dc_ntag - 0.0) / 1.0) / math.sqrt(2)))) / 1.65298e-05   # -0.1%  e3_b2 < 0.0001252 and dc_ntag > 0
        - 0.0007245458 * (0.01301382 * 0.5 * ((0.1193588 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.1193588 - Q.tau2) / 0.01301382) / math.sqrt(2)))) * (0.02453987 * 0.5 * ((-0.0190396 - Q.sjq_2_2_k1) / 0.02453987) * (1 + math.erf(((-0.0190396 - Q.sjq_2_2_k1) / 0.02453987) / math.sqrt(2)))) / 0.006020944   # -0.1%  tau2 < 0.1194 and sjq_2_2_k1 < -0.01904
        - 0.0007191282 * (4.879578 * 0.5 * ((121.0623 - Q.mass_top30) / 4.879578) * (1 + math.erf(((121.0623 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_min12_n_disp3 - 0.0) / 1.0) * (1 + math.erf(((Q.ak02_min12_n_disp3 - 0.0) / 1.0) / math.sqrt(2)))) / 2.968073   # -0.1%  mass_top30 < 121.1 and ak02_min12_n_disp3 > 0
        + 0.0005647615 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.02505229 * 0.5 * ((0.3448514 - Q.sdb_2_z) / 0.02505229) * (1 + math.erf(((0.3448514 - Q.sdb_2_z) / 0.02505229) / math.sqrt(2)))) / 2.525739e-05   # +0.1%  e3_b2 < 0.0004127 and sdb_2_z < 0.3449
        - 0.000487659 * (3.289335 * 0.5 * ((Q.mass - 123.7928) / 3.289335) * (1 + math.erf(((Q.mass - 123.7928) / 3.289335) / math.sqrt(2)))) * (0.01084662 * 0.5 * ((Q.pz_lnkt4 - 0.02508543) / 0.01084662) * (1 + math.erf(((Q.pz_lnkt4 - 0.02508543) / 0.01084662) / math.sqrt(2)))) / 0.1531759   # -0.0%  mass > 123.8 and pz_lnkt4 > 0.02509
        + 0.0004676832 * (4.883139 * 0.5 * ((Q.mass_top50 - 89.68964) / 4.883139) * (1 + math.erf(((Q.mass_top50 - 89.68964) / 4.883139) / math.sqrt(2)))) * (0.01033072 * 0.5 * ((0.01033072 - Q.ak02_3_z_disp3) / 0.01033072) * (1 + math.erf(((0.01033072 - Q.ak02_3_z_disp3) / 0.01033072) / math.sqrt(2)))) / 0.2141335   # +0.0%  mass_top50 > 89.69 and ak02_3_z_disp3 < 0.01033
        - 0.000458158 * (14.29796 * 0.5 * ((Q.mass_charged - 113.5019) / 14.29796) * (1 + math.erf(((Q.mass_charged - 113.5019) / 14.29796) / math.sqrt(2)))) / 1.247881   # -0.0%  mass_charged > 113.5
        - 0.0003905462 * (3.082358 * 0.5 * ((Q.mass_charged - 58.87188) / 3.082358) * (1 + math.erf(((Q.mass_charged - 58.87188) / 3.082358) / math.sqrt(2)))) / 13.32028   # -0.0%  mass_charged > 58.87
        + 0.0003805749 * (0.003267481 * 0.5 * ((0.01181938 - Q.z_dr_0p4_up) / 0.003267481) * (1 + math.erf(((0.01181938 - Q.z_dr_0p4_up) / 0.003267481) / math.sqrt(2)))) / 0.003171787   # +0.0%  z_dr_0p4_up < 0.01182
        + 0.000250594 * (3.289335 * 0.5 * ((Q.mass - 123.7928) / 3.289335) * (1 + math.erf(((Q.mass - 123.7928) / 3.289335) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2)))) / 0.2990214   # +0.0%  mass > 123.8 and dc_1_n_lep < 0
        + 0.0002440744 * (0.02188437 * 0.5 * ((Q.z_top30_slots - 0.8942764) / 0.02188437) * (1 + math.erf(((Q.z_top30_slots - 0.8942764) / 0.02188437) / math.sqrt(2)))) * (0.0219027 * 0.5 * ((0.147768 - Q.dc_3_z) / 0.0219027) * (1 + math.erf(((0.147768 - Q.dc_3_z) / 0.0219027) / math.sqrt(2)))) / 0.00814385   # +0.0%  z_top30_slots > 0.8943 and dc_3_z < 0.1478
        + 0.0002116213 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) / math.sqrt(2)))) * (5.3089e-08 * 0.5 * ((7.0969e-08 - Q.e4_b2) / 5.3089e-08) * (1 + math.erf(((7.0969e-08 - Q.e4_b2) / 5.3089e-08) / math.sqrt(2)))) / 2.400806e-07   # +0.0%  mres_sd_mass_b2z01 > 130.1 and e4_b2 < 7.097e-08
        - 0.0001883212 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) / math.sqrt(2)))) * (0.2610212 * 0.5 * ((-0.6337755 - Q.sjq_2_2_k1) / 0.2610212) * (1 + math.erf(((-0.6337755 - Q.sjq_2_2_k1) / 0.2610212) / math.sqrt(2)))) / 0.1110951   # -0.0%  mres_sd_mass_b2z01 > 130.1 and sjq_2_2_k1 < -0.6338
        + 0.0001845692 * (1.0 * 0.5 * ((Q.lepsj_3_n_d3 - 3.0) / 1.0) * (1 + math.erf(((Q.lepsj_3_n_d3 - 3.0) / 1.0) / math.sqrt(2)))) / 0.0922171   # +0.0%  lepsj_3_n_d3 > 3
        + 0.0001440206 * (2.077045 * 0.5 * ((1.851735 - Q.sj2_mass2) / 2.077045) * (1 + math.erf(((1.851735 - Q.sj2_mass2) / 2.077045) / math.sqrt(2)))) * (0.3881729 * 0.5 * ((Q.mass_2photon - 1.818724) / 0.3881729) * (1 + math.erf(((Q.mass_2photon - 1.818724) / 0.3881729) / math.sqrt(2)))) / 0.211829   # +0.0%  sj2_mass2 < 1.852 and mass_2photon > 1.819
        + 0.0001085018 * (22.10989 * 0.5 * ((171.3819 - Q.mres_pruned_mass) / 22.10989) * (1 + math.erf(((171.3819 - Q.mres_pruned_mass) / 22.10989) / math.sqrt(2)))) * (14.45336 * 0.5 * ((Q.jd_3d_6 - 30.57088) / 14.45336) * (1 + math.erf(((Q.jd_3d_6 - 30.57088) / 14.45336) / math.sqrt(2)))) / 359.4133   # +0.0%  mres_pruned_mass < 171.4 and jd_3d_6 > 30.57
        + 6.853814e-05 * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2)))) / 0.01976846   # +0.0%  sjq_3_2_k1 < -0.6015
        + 6.393914e-05 * (0.02188437 * 0.5 * ((Q.z_top30_slots - 0.8942764) / 0.02188437) * (1 + math.erf(((Q.z_top30_slots - 0.8942764) / 0.02188437) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2)))) / 0.002743061   # +0.0%  z_top30_slots > 0.8943 and ak02_3_n_lep > 0
        - 4.583972e-05 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (0.0136261 * 0.5 * ((Q.eta_4 - -0.03231812) / 0.0136261) * (1 + math.erf(((Q.eta_4 - -0.03231812) / 0.0136261) / math.sqrt(2)))) / 1.895345   # -0.0%  mass_displaced3 < 39.1 and eta_4 > -0.03232
        + 4.400533e-05 * (0.04000118 * 0.5 * ((0.135772 - Q.tau21) / 0.04000118) * (1 + math.erf(((0.135772 - Q.tau21) / 0.04000118) / math.sqrt(2)))) / 0.002007792   # +0.0%  tau21 < 0.1358
        + 3.620607e-05 * (5.294312 * 0.5 * ((Q.mres_sd_mass_b2z01 - 102.264) / 5.294312) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 102.264) / 5.294312) / math.sqrt(2)))) * (3.2258e-08 * 0.5 * ((Q.e4_b2 - 1.788e-08) / 3.2258e-08) * (1 + math.erf(((Q.e4_b2 - 1.788e-08) / 3.2258e-08) / math.sqrt(2)))) / 6.409875e-06   # +0.0%  mres_sd_mass_b2z01 > 102.3 and e4_b2 > 1.788e-08
        + 2.684837e-05 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (0.004196167 * 0.5 * ((Q.d0err_10 - 0.02780151) / 0.004196167) * (1 + math.erf(((Q.d0err_10 - 0.02780151) / 0.004196167) / math.sqrt(2)))) / 0.000415436   # +0.0%  n_s3d_above_3 > 3 and d0err_10 > 0.0278
    )
    return z


def neuron_98(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.990162e-07
    )
    return z


def neuron_99(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.00193424
    )
    return z


def neuron_100(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.264072e-06
    )
    return z


def neuron_101(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.878259e-06
    )
    return z


def neuron_102(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.490507e-06
    )
    return z


def neuron_103(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.414372e-06
    )
    return z


def neuron_104(Q):
    # scale S = 6.828; each line: share * term / its average size
    z = 6.828248 * (-0.1273925
        + 0.1077219 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) / math.sqrt(2)))) / 52.69014   # +10.8%  mres_sd_mass_b2z01 > 55.13
        + 0.08710343 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 3.208089) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 3.208089) / 1.000219) / math.sqrt(2)))) / 6.245495   # +8.7%  mass_displaced3 > 3.208
        + 0.06869394 * (0.171005 * 0.5 * ((-1.352792 - Q.pair_mean_lndelta) / 0.171005) * (1 + math.erf(((-1.352792 - Q.pair_mean_lndelta) / 0.171005) / math.sqrt(2)))) / 0.867331   # +6.9%  pair_mean_lndelta < -1.353
        - 0.0650327 * (1.566771 * 0.5 * ((Q.mass_displaced3 - 4.41005) / 1.566771) * (1 + math.erf(((Q.mass_displaced3 - 4.41005) / 1.566771) / math.sqrt(2)))) / 5.814614   # -6.5%  mass_displaced3 > 4.41
        - 0.05371132 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) / math.sqrt(2)))) / 24.25944   # -5.4%  mres_sd_mass_b2z01 > 91.15
        + 0.0507059 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) / 15.65331   # +5.1%  mass < 117.5
        + 0.04360661 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (0.04737588 * 0.5 * ((0.1807551 - Q.ak02_3_z) / 0.04737588) * (1 + math.erf(((0.1807551 - Q.ak02_3_z) / 0.04737588) / math.sqrt(2)))) / 3.24802   # +4.4%  n_pt_above_1 < 58 and ak02_3_z < 0.1808
        - 0.03412874 * (0.01003572 * 0.5 * ((Q.pz_lnd0 - 0.05066241) / 0.01003572) * (1 + math.erf(((Q.pz_lnd0 - 0.05066241) / 0.01003572) / math.sqrt(2)))) / 0.07967157   # -3.4%  pz_lnd0 > 0.05066
        - 0.02983825 * (0.009790654 * 0.5 * ((Q.C2_b05 - 0.2234678) / 0.009790654) * (1 + math.erf(((Q.C2_b05 - 0.2234678) / 0.009790654) / math.sqrt(2)))) / 0.05425106   # -3.0%  C2_b05 > 0.2235
        - 0.02371866 * (13.49273 * 0.5 * ((149.0507 - Q.mass) / 13.49273) * (1 + math.erf(((149.0507 - Q.mass) / 13.49273) / math.sqrt(2)))) * (202.2502 * 0.5 * ((1401.904 - Q.sum_e) / 202.2502) * (1 + math.erf(((1401.904 - Q.sum_e) / 202.2502) / math.sqrt(2)))) / 20678.93   # -2.4%  mass < 149.1 and sum_e < 1402
        - 0.02028721 * (13.49273 * 0.5 * ((149.0507 - Q.mass) / 13.49273) * (1 + math.erf(((149.0507 - Q.mass) / 13.49273) / math.sqrt(2)))) * (18.34092 * 0.5 * ((Q.lnpt_65 - -0.07975792) / 18.34092) * (1 + math.erf(((Q.lnpt_65 - -0.07975792) / 18.34092) / math.sqrt(2)))) / 110.8442   # -2.0%  mass < 149.1 and lnpt_65 > -0.07976
        - 0.0197553 * (0.05354637 * 0.5 * ((0.7758695 - Q.z_charged_had) / 0.05354637) * (1 + math.erf(((0.7758695 - Q.z_charged_had) / 0.05354637) / math.sqrt(2)))) / 0.2715281   # -2.0%  z_charged_had < 0.7759
        + 0.01931084 * (1.5 * 0.5 * ((18.0 - Q.sjq_2_1_nch) / 1.5) * (1 + math.erf(((18.0 - Q.sjq_2_1_nch) / 1.5) / math.sqrt(2)))) / 6.622259   # +1.9%  sjq_2_1_nch < 18
        + 0.01830973 * (11.52979 * 0.5 * ((99.20396 - Q.mass_charged) / 11.52979) * (1 + math.erf(((99.20396 - Q.mass_charged) / 11.52979) / math.sqrt(2)))) / 36.82531   # +1.8%  mass_charged < 99.2
        + 0.01608407 * (0.002048025 * 0.5 * ((0.02098847 - Q.tau5) / 0.002048025) * (1 + math.erf(((0.02098847 - Q.tau5) / 0.002048025) / math.sqrt(2)))) / 0.002077462   # +1.6%  tau5 < 0.02099
        + 0.01500667 * (19.4821 * 0.5 * ((30.48974 - Q.sjf_2_2_max3d) / 19.4821) * (1 + math.erf(((30.48974 - Q.sjf_2_2_max3d) / 19.4821) / math.sqrt(2)))) / 15.39846   # +1.5%  sjf_2_2_max3d < 30.49
        - 0.01490297 * (3.069852 * 0.5 * ((4.516968 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.516968 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) / 1.027676   # -1.5%  sjf_2_2_max3d < 4.517
        - 0.01474917 * (6.5 * 0.5 * ((41.0 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((41.0 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2)))) / 7.611779   # -1.5%  n_pairs_kt_above_3 < 41
        + 0.01436909 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.983043) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.983043) / 4.308625) / math.sqrt(2)))) / 5.404766   # +1.4%  lep_ptrel > 6.983
        - 0.01433921 * (19.88146 * 0.5 * ((10.98011 - Q.dc_2_jp) / 19.88146) * (1 + math.erf(((10.98011 - Q.dc_2_jp) / 19.88146) / math.sqrt(2)))) / 4.832356   # -1.4%  dc_2_jp < 10.98
        + 0.01288489 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (0.194914 * 0.5 * ((3.660196 - Q.pair_max_lnkt) / 0.194914) * (1 + math.erf(((3.660196 - Q.pair_max_lnkt) / 0.194914) / math.sqrt(2)))) / 20.24649   # +1.3%  n_pt_above_1 < 58 and pair_max_lnkt < 3.66
        - 0.01268238 * Q.n_charged_pt_above_10 / 8.309213   # -1.3%  n_charged_pt_above_10
        + 0.0123552 * (0.07989133 * 0.5 * ((Q.N3_b05 - 0.828622) / 0.07989133) * (1 + math.erf(((Q.N3_b05 - 0.828622) / 0.07989133) / math.sqrt(2)))) / 0.1424352   # +1.2%  N3_b05 > 0.8286
        + 0.01229461 * (1.0 * 0.5 * ((1.0 - Q.lepsj_2_n_d3) / 1.0) * (1 + math.erf(((1.0 - Q.lepsj_2_n_d3) / 1.0) / math.sqrt(2)))) / 0.6621907   # +1.2%  lepsj_2_n_d3 < 1
        - 0.0118954 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.983043) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.983043) / 4.308625) / math.sqrt(2)))) * (0.02843722 * 0.5 * ((0.1014193 - Q.ak02_3_z) / 0.02843722) * (1 + math.erf(((0.1014193 - Q.ak02_3_z) / 0.02843722) / math.sqrt(2)))) / 0.4020861   # -1.2%  lep_ptrel > 6.983 and ak02_3_z < 0.1014
        - 0.01169559 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2261637   # -1.2%  lep_iso < 0.4382
        + 0.01042817 * (1.5 * 0.5 * ((6.0 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.0 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2)))) / 3.161762   # +1.0%  jd_n_d3_pt1 < 6
        - 0.00970493 * (0.01171084 * 0.5 * ((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) / 0.01210615   # -1.0%  sjf_2_1_z_d3 < 0.02623
        + 0.00945489 * (4.590131 * 0.5 * ((107.2089 - Q.mres_sd_mass_b2z01) / 4.590131) * (1 + math.erf(((107.2089 - Q.mres_sd_mass_b2z01) / 4.590131) / math.sqrt(2)))) / 16.34801   # +0.9%  mres_sd_mass_b2z01 < 107.2
        - 0.009193095 * (1.5 * 0.5 * ((18.0 - Q.sjq_2_1_nch) / 1.5) * (1 + math.erf(((18.0 - Q.sjq_2_1_nch) / 1.5) / math.sqrt(2)))) * (777.2244 * 0.5 * ((1031.057 - Q.sv_1_sd0_sum) / 777.2244) * (1 + math.erf(((1031.057 - Q.sv_1_sd0_sum) / 777.2244) / math.sqrt(2)))) / 5273.189   # -0.9%  sjq_2_1_nch < 18 and sv_1_sd0_sum < 1031
        - 0.00830713 * (0.03993816 * 0.5 * ((0.4680886 - Q.tau32_b2) / 0.03993816) * (1 + math.erf(((0.4680886 - Q.tau32_b2) / 0.03993816) / math.sqrt(2)))) / 0.05982784   # -0.8%  tau32_b2 < 0.4681
        + 0.007946755 * (0.06999318 * 0.5 * ((Q.lund2_lndelta - -1.052328) / 0.06999318) * (1 + math.erf(((Q.lund2_lndelta - -1.052328) / 0.06999318) / math.sqrt(2)))) / 0.2290993   # +0.8%  lund2_lndelta > -1.052
        - 0.00764291 * (0.00390151 * 0.5 * ((0.0740332 - Q.M2) / 0.00390151) * (1 + math.erf(((0.0740332 - Q.M2) / 0.00390151) / math.sqrt(2)))) * (0.04921085 * 0.5 * ((Q.N3_b05 - 0.7045747) / 0.04921085) * (1 + math.erf(((Q.N3_b05 - 0.7045747) / 0.04921085) / math.sqrt(2)))) / 0.005109293   # -0.8%  M2 < 0.07403 and N3_b05 > 0.7046
        - 0.006695549 * (3.069852 * 0.5 * ((4.516968 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.516968 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) * (164.9991 * 0.5 * ((226.3008 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((226.3008 - Q.sip_3d_2) / 164.9991) / math.sqrt(2)))) / 170.3646   # -0.7%  sjf_2_2_max3d < 4.517 and sip_3d_2 < 226.3
        - 0.006053059 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) / math.sqrt(2)))) * (0.05887118 * 0.5 * ((0.3853337 - Q.ak02_dr13) / 0.05887118) * (1 + math.erf(((0.3853337 - Q.ak02_dr13) / 0.05887118) / math.sqrt(2)))) / 8.400821   # -0.6%  mres_sd_mass_b2z01 > 55.13 and ak02_dr13 < 0.3853
        + 0.00602869 * (0.01129976 * 0.5 * ((Q.N2 - 0.3357559) / 0.01129976) * (1 + math.erf(((Q.N2 - 0.3357559) / 0.01129976) / math.sqrt(2)))) / 0.02163059   # +0.6%  N2 > 0.3358
        + 0.006000409 * (0.01171084 * 0.5 * ((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) * (95.61515 * 0.5 * ((45.92423 - Q.dc_2_jp) / 95.61515) * (1 + math.erf(((45.92423 - Q.dc_2_jp) / 95.61515) / math.sqrt(2)))) / 0.3087844   # +0.6%  sjf_2_1_z_d3 < 0.02623 and dc_2_jp < 45.92
        + 0.005498514 * (0.1799842 * 0.5 * ((2.044945 - Q.sjf_2_2_max3d) / 0.1799842) * (1 + math.erf(((2.044945 - Q.sjf_2_2_max3d) / 0.1799842) / math.sqrt(2)))) / 0.1618068   # +0.5%  sjf_2_2_max3d < 2.045
        - 0.004804056 * (1.696027 * 0.5 * ((3.32421 - Q.sjf_2_1_max3d) / 1.696027) * (1 + math.erf(((3.32421 - Q.sjf_2_1_max3d) / 1.696027) / math.sqrt(2)))) / 0.312688   # -0.5%  sjf_2_1_max3d < 3.324
        - 0.004529059 * (13.49273 * 0.5 * ((149.0507 - Q.mass) / 13.49273) * (1 + math.erf(((149.0507 - Q.mass) / 13.49273) / math.sqrt(2)))) * (0.07882828 * 0.5 * ((Q.sjq_2_sumabs_k03 - 0.7416858) / 0.07882828) * (1 + math.erf(((Q.sjq_2_sumabs_k03 - 0.7416858) / 0.07882828) / math.sqrt(2)))) / 7.128835   # -0.5%  mass < 149.1 and sjq_2_sumabs_k03 > 0.7417
        + 0.004375302 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) / 20.63145   # +0.4%  n_pt_above_1 < 58
        + 0.004202132 * (0.00390151 * 0.5 * ((0.0740332 - Q.M2) / 0.00390151) * (1 + math.erf(((0.0740332 - Q.M2) / 0.00390151) / math.sqrt(2)))) / 0.009166666   # +0.4%  M2 < 0.07403
        + 0.004162355 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) / math.sqrt(2)))) / 3.38788   # +0.4%  mres_sd_mass_b0z005 > 159.9
        + 0.004008195 * (6.5 * 0.5 * ((41.0 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((41.0 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2)))) * (0.04359399 * 0.5 * ((0.10729 - Q.sdb_5_z) / 0.04359399) * (1 + math.erf(((0.10729 - Q.sdb_5_z) / 0.04359399) / math.sqrt(2)))) / 0.6506156   # +0.4%  n_pairs_kt_above_3 < 41 and sdb_5_z < 0.1073
        - 0.003982686 * (1.5 * 0.5 * ((Q.sjf_3_1_n_d3 - 4.0) / 1.5) * (1 + math.erf(((Q.sjf_3_1_n_d3 - 4.0) / 1.5) / math.sqrt(2)))) / 0.2690222   # -0.4%  sjf_3_1_n_d3 > 4
        + 0.003569729 * (6.5 * 0.5 * ((41.0 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((41.0 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) / 4.469311   # +0.4%  n_pairs_kt_above_3 < 41 and kt2_2_n_lep < 1
        - 0.003376171 * (24.5 * 0.5 * ((169.0 - Q.n_pairs_kt_above_1) / 24.5) * (1 + math.erf(((169.0 - Q.n_pairs_kt_above_1) / 24.5) / math.sqrt(2)))) / 27.54883   # -0.3%  n_pairs_kt_above_1 < 169
        + 0.003291559 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (3.876948 * 0.5 * ((6.767937 - Q.mass_2charged) / 3.876948) * (1 + math.erf(((6.767937 - Q.mass_2charged) / 3.876948) / math.sqrt(2)))) / 56.75724   # +0.3%  n_pt_above_1 < 58 and mass_2charged < 6.768
        - 0.003269262 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_1_n_d5 - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_2_1_n_d5 - 1.0) / 1.0) / math.sqrt(2)))) / 20.31445   # -0.3%  n_pt_above_1 < 58 and sjf_2_1_n_d5 > 1
        - 0.00303647 * (0.002703061 * 0.5 * ((0.03663959 - Q.tau5) / 0.002703061) * (1 + math.erf(((0.03663959 - Q.tau5) / 0.002703061) / math.sqrt(2)))) / 0.009665218   # -0.3%  tau5 < 0.03664
        + 0.00286551 * (0.009790654 * 0.5 * ((Q.C2_b05 - 0.2234678) / 0.009790654) * (1 + math.erf(((Q.C2_b05 - 0.2234678) / 0.009790654) / math.sqrt(2)))) * (1.5 * 0.5 * ((6.0 - Q.n_s3d_above_10) / 1.5) * (1 + math.erf(((6.0 - Q.n_s3d_above_10) / 1.5) / math.sqrt(2)))) / 0.169875   # +0.3%  C2_b05 > 0.2235 and n_s3d_above_10 < 6
        + 0.002654756 * (1.0 * 0.5 * ((Q.sdb_2_n - 9.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 9.0) / 1.0) / math.sqrt(2)))) / 2.900138   # +0.3%  sdb_2_n > 9
        + 0.002594435 * (0.009790654 * 0.5 * ((Q.C2_b05 - 0.2234678) / 0.009790654) * (1 + math.erf(((Q.C2_b05 - 0.2234678) / 0.009790654) / math.sqrt(2)))) * (0.006538313 * 0.5 * ((Q.pz_lnkt1 - 0.07470686) / 0.006538313) * (1 + math.erf(((Q.pz_lnkt1 - 0.07470686) / 0.006538313) / math.sqrt(2)))) / 0.001962747   # +0.3%  C2_b05 > 0.2235 and pz_lnkt1 > 0.07471
        - 0.002532311 * (0.01282622 * 0.5 * ((0.1010123 - Q.z_neutral_had) / 0.01282622) * (1 + math.erf(((0.1010123 - Q.z_neutral_had) / 0.01282622) / math.sqrt(2)))) / 0.020496   # -0.3%  z_neutral_had < 0.101
        + 0.002460728 * (0.430485 * 0.5 * ((2.508382 - Q.sjf_4_4_max3d) / 0.430485) * (1 + math.erf(((2.508382 - Q.sjf_4_4_max3d) / 0.430485) / math.sqrt(2)))) / 1.019744   # +0.2%  sjf_4_4_max3d < 2.508
        - 0.002121138 * (0.1006646 * 0.5 * ((1.226724 - Q.D2) / 0.1006646) * (1 + math.erf(((1.226724 - Q.D2) / 0.1006646) / math.sqrt(2)))) / 0.07990341   # -0.2%  D2 < 1.227
        + 0.002112684 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 3.208089) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 3.208089) / 1.000219) / math.sqrt(2)))) * (0.02843722 * 0.5 * ((0.1014193 - Q.ak02_3_z) / 0.02843722) * (1 + math.erf(((0.1014193 - Q.ak02_3_z) / 0.02843722) / math.sqrt(2)))) / 0.4169126   # +0.2%  mass_displaced3 > 3.208 and ak02_3_z < 0.1014
        - 0.002041261 * (1.0 * 0.5 * ((Q.sdb_2_n - 9.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 9.0) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((3.0 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2)))) / 1.573108   # -0.2%  sdb_2_n > 9 and n_lund_kt_above_5 < 3
        + 0.002004635 * (0.01926185 * 0.5 * ((Q.sd_rg - 0.3591864) / 0.01926185) * (1 + math.erf(((Q.sd_rg - 0.3591864) / 0.01926185) / math.sqrt(2)))) / 0.05680271   # +0.2%  sd_rg > 0.3592
        + 0.001988334 * (1.0 * 0.5 * ((2.0 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((2.0 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2)))) / 0.7163451   # +0.2%  n_sdz_above_5 < 2
        - 0.001816027 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2)))) / 0.0537686   # -0.2%  n_lepton < 0
        + 0.00162914 * (5.726106 * 0.5 * ((4.183454 - Q.sv_2_sd0_sum) / 5.726106) * (1 + math.erf(((4.183454 - Q.sv_2_sd0_sum) / 5.726106) / math.sqrt(2)))) / 2.238841   # +0.2%  sv_2_sd0_sum < 4.183
        + 0.001599302 * (0.08021924 * 0.5 * ((Q.lund_max_lndelta - -0.615738) / 0.08021924) * (1 + math.erf(((Q.lund_max_lndelta - -0.615738) / 0.08021924) / math.sqrt(2)))) / 0.02382256   # +0.2%  lund_max_lndelta > -0.6157
        + 0.001583028 * (8.295007 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 83.41384) / 8.295007) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 83.41384) / 8.295007) / math.sqrt(2)))) / 1.778785   # +0.2%  nca_sj4_pair_mass_2nd > 83.41
        + 0.001576599 * (6.58003 * 0.5 * ((Q.mass_top30 - 126.5007) / 6.58003) * (1 + math.erf(((Q.mass_top30 - 126.5007) / 6.58003) / math.sqrt(2)))) / 5.441799   # +0.2%  mass_top30 > 126.5
        - 0.001436219 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 2.0) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 2.0) / 1.0) / math.sqrt(2)))) / 8.202956   # -0.1%  n_pt_above_1 < 58 and sjf_2_2_n_d3 > 2
        + 0.001411507 * (0.01171084 * 0.5 * ((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) * (0.01528826 * 0.5 * ((0.106665 - Q.dr_18) / 0.01528826) * (1 + math.erf(((0.106665 - Q.dr_18) / 0.01528826) / math.sqrt(2)))) / 0.0002372133   # +0.1%  sjf_2_1_z_d3 < 0.02623 and dr_18 < 0.1067
        + 0.001378848 * (0.01268564 * 0.5 * ((0.03385157 - Q.sjf_2_2_z_d3) / 0.01268564) * (1 + math.erf(((0.03385157 - Q.sjf_2_2_z_d3) / 0.01268564) / math.sqrt(2)))) / 0.02168765   # +0.1%  sjf_2_2_z_d3 < 0.03385
        - 0.001354698 * (1.5 * 0.5 * ((6.0 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.0 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.06009795   # -0.1%  jd_n_d3_pt1 < 6 and ak02_2_n_lep < 0
        - 0.001253775 * (1.5 * 0.5 * ((6.0 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.0 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.isphoton_40 - 0.0) / 1.0) * (1 + math.erf(((Q.isphoton_40 - 0.0) / 1.0) / math.sqrt(2)))) / 0.5755465   # -0.1%  jd_n_d3_pt1 < 6 and isphoton_40 > 0
        + 0.00112215 * (0.03993816 * 0.5 * ((0.4680886 - Q.tau32_b2) / 0.03993816) * (1 + math.erf(((0.4680886 - Q.tau32_b2) / 0.03993816) / math.sqrt(2)))) * (5.0 * 0.5 * ((17.0 - Q.n_pairs_kt_above_10) / 5.0) * (1 + math.erf(((17.0 - Q.n_pairs_kt_above_10) / 5.0) / math.sqrt(2)))) / 0.5101588   # +0.1%  tau32_b2 < 0.4681 and n_pairs_kt_above_10 < 17
        + 0.001041313 * (1.729885 * 0.5 * ((3.928781 - Q.sj2_mass2) / 1.729885) * (1 + math.erf(((3.928781 - Q.sj2_mass2) / 1.729885) / math.sqrt(2)))) / 0.2363684   # +0.1%  sj2_mass2 < 3.929
        - 0.001026926 * (1.0 * 0.5 * ((Q.sdb_2_n - 9.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 9.0) / 1.0) / math.sqrt(2)))) * (0.01417841 * 0.5 * ((0.1049877 - Q.C2_b2) / 0.01417841) * (1 + math.erf(((0.1049877 - Q.C2_b2) / 0.01417841) / math.sqrt(2)))) / 0.1179817   # -0.1%  sdb_2_n > 9 and C2_b2 < 0.105
        + 0.001025655 * (1.696027 * 0.5 * ((3.32421 - Q.sjf_2_1_max3d) / 1.696027) * (1 + math.erf(((3.32421 - Q.sjf_2_1_max3d) / 1.696027) / math.sqrt(2)))) * (4.85271 * 0.5 * ((16.06148 - Q.mass_2photon) / 4.85271) * (1 + math.erf(((16.06148 - Q.mass_2photon) / 4.85271) / math.sqrt(2)))) / 3.637499   # +0.1%  sjf_2_1_max3d < 3.324 and mass_2photon < 16.06
        + 0.001006667 * (0.00390151 * 0.5 * ((0.0740332 - Q.M2) / 0.00390151) * (1 + math.erf(((0.0740332 - Q.M2) / 0.00390151) / math.sqrt(2)))) * (0.04063514 * 0.5 * ((Q.dr_21 - 0.0) / 0.04063514) * (1 + math.erf(((Q.dr_21 - 0.0) / 0.04063514) / math.sqrt(2)))) / 0.001777187   # +0.1%  M2 < 0.07403 and dr_21 > 0
        + 0.0009826414 * (0.01171084 * 0.5 * ((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) * (0.1472193 * 0.5 * ((Q.D2_b2 - 0.2571861) / 0.1472193) * (1 + math.erf(((Q.D2_b2 - 0.2571861) / 0.1472193) / math.sqrt(2)))) / 0.06202829   # +0.1%  sjf_2_1_z_d3 < 0.02623 and D2_b2 > 0.2572
        + 0.0009772896 * (0.002703061 * 0.5 * ((0.03663959 - Q.tau5) / 0.002703061) * (1 + math.erf(((0.03663959 - Q.tau5) / 0.002703061) / math.sqrt(2)))) * (0.001358032 * 0.5 * ((0.03421021 - Q.dzerr_1) / 0.001358032) * (1 + math.erf(((0.03421021 - Q.dzerr_1) / 0.001358032) / math.sqrt(2)))) / 0.0001415858   # +0.1%  tau5 < 0.03664 and dzerr_1 < 0.03421
        - 0.0009462571 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) / math.sqrt(2)))) * (316.716 * 0.5 * ((Q.kt2_min12_jp - 271.4622) / 316.716) * (1 + math.erf(((Q.kt2_min12_jp - 271.4622) / 316.716) / math.sqrt(2)))) / 5073.904   # -0.1%  mres_sd_mass_b2z01 > 55.13 and kt2_min12_jp > 271.5
        - 0.0008604474 * (567.5171 * 0.5 * ((131.3722 - Q.lepsj_2_maxsd0) / 567.5171) * (1 + math.erf(((131.3722 - Q.lepsj_2_maxsd0) / 567.5171) / math.sqrt(2)))) / 70.54231   # -0.1%  lepsj_2_maxsd0 < 131.4
        + 0.0008602447 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (0.05422306 * 0.5 * ((0.8615361 - Q.z_charged) / 0.05422306) * (1 + math.erf(((0.8615361 - Q.z_charged) / 0.05422306) / math.sqrt(2)))) / 5.127343   # +0.1%  n_pt_above_1 < 58 and z_charged < 0.8615
        - 0.0007715043 * (8.295007 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 83.41384) / 8.295007) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 83.41384) / 8.295007) / math.sqrt(2)))) * (1.512315e-07 * 0.5 * ((Q.e4 - 5.29882e-07) / 1.512315e-07) * (1 + math.erf(((Q.e4 - 5.29882e-07) / 1.512315e-07) / math.sqrt(2)))) / 2.18676e-05   # -0.1%  nca_sj4_pair_mass_2nd > 83.41 and e4 > 5.299e-07
        + 0.000754743 * (12.49365 * 0.5 * ((Q.sj4_pair_mass_max - 128.0079) / 12.49365) * (1 + math.erf(((Q.sj4_pair_mass_max - 128.0079) / 12.49365) / math.sqrt(2)))) / 1.34552   # +0.1%  sj4_pair_mass_max > 128
        + 0.0006512027 * (0.03993816 * 0.5 * ((0.4680886 - Q.tau32_b2) / 0.03993816) * (1 + math.erf(((0.4680886 - Q.tau32_b2) / 0.03993816) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) / 0.1542815   # +0.1%  tau32_b2 < 0.4681 and lepsj_3_n_d3 < 3
        - 0.0006247591 * (0.1006646 * 0.5 * ((1.226724 - Q.D2) / 0.1006646) * (1 + math.erf(((1.226724 - Q.D2) / 0.1006646) / math.sqrt(2)))) * (0.1809026 * 0.5 * ((Q.lund1_lndelta - -1.227355) / 0.1809026) * (1 + math.erf(((Q.lund1_lndelta - -1.227355) / 0.1809026) / math.sqrt(2)))) / 0.04176895   # -0.1%  D2 < 1.227 and lund1_lndelta > -1.227
        + 0.0005814421 * (13.49273 * 0.5 * ((149.0507 - Q.mass) / 13.49273) * (1 + math.erf(((149.0507 - Q.mass) / 13.49273) / math.sqrt(2)))) / 38.84933   # +0.1%  mass < 149.1
        + 0.0005402126 * (1.5 * 0.5 * ((6.0 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.0 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2)))) * (0.02700916 * 0.5 * ((0.3709098 - Q.sdb_2_z) / 0.02700916) * (1 + math.erf(((0.3709098 - Q.sdb_2_z) / 0.02700916) / math.sqrt(2)))) / 0.2501954   # +0.1%  jd_n_d3_pt1 < 6 and sdb_2_z < 0.3709
        - 0.0004376775 * (0.01003572 * 0.5 * ((Q.pz_lnd0 - 0.05066241) / 0.01003572) * (1 + math.erf(((Q.pz_lnd0 - 0.05066241) / 0.01003572) / math.sqrt(2)))) * (0.00461869 * 0.5 * ((0.008363276 - Q.sjq_2_prod_k1) / 0.00461869) * (1 + math.erf(((0.008363276 - Q.sjq_2_prod_k1) / 0.00461869) / math.sqrt(2)))) / 0.00259516   # -0.0%  pz_lnd0 > 0.05066 and sjq_2_prod_k1 < 0.008363
        + 0.0004354323 * (0.01268564 * 0.5 * ((0.03385157 - Q.sjf_2_2_z_d3) / 0.01268564) * (1 + math.erf(((0.03385157 - Q.sjf_2_2_z_d3) / 0.01268564) / math.sqrt(2)))) * (0.09214564 * 0.5 * ((Q.dc_1_charge - 0.4233202) / 0.09214564) * (1 + math.erf(((Q.dc_1_charge - 0.4233202) / 0.09214564) / math.sqrt(2)))) / 0.001180928   # +0.0%  sjf_2_2_z_d3 < 0.03385 and dc_1_charge > 0.4233
        + 0.000426333 * (1.5 * 0.5 * ((6.0 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.0 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2)))) * (0.005637773 * 0.5 * ((Q.mean_eta2 - 0.03501387) / 0.005637773) * (1 + math.erf(((Q.mean_eta2 - 0.03501387) / 0.005637773) / math.sqrt(2)))) / 0.005770517   # +0.0%  jd_n_d3_pt1 < 6 and mean_eta2 > 0.03501
        + 0.0004263046 * (6.58003 * 0.5 * ((Q.mass_top30 - 126.5007) / 6.58003) * (1 + math.erf(((Q.mass_top30 - 126.5007) / 6.58003) / math.sqrt(2)))) * (0.9957982 * 0.5 * ((3.041925 - Q.sip_3d_2) / 0.9957982) * (1 + math.erf(((3.041925 - Q.sip_3d_2) / 0.9957982) / math.sqrt(2)))) / 0.385599   # +0.0%  mass_top30 > 126.5 and sip_3d_2 < 3.042
        + 0.0003037156 * (19.88146 * 0.5 * ((10.98011 - Q.dc_2_jp) / 19.88146) * (1 + math.erf(((10.98011 - Q.dc_2_jp) / 19.88146) / math.sqrt(2)))) * (0.02390684 * 0.5 * ((Q.C2_b2 - 0.1404188) / 0.02390684) * (1 + math.erf(((Q.C2_b2 - 0.1404188) / 0.02390684) / math.sqrt(2)))) / 0.1062317   # +0.0%  dc_2_jp < 10.98 and C2_b2 > 0.1404
        + 0.000276802 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 3.208089) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 3.208089) / 1.000219) / math.sqrt(2)))) * (0.2385164 * 0.5 * ((0.0 - Q.ak02_dr23) / 0.2385164) * (1 + math.erf(((0.0 - Q.ak02_dr23) / 0.2385164) / math.sqrt(2)))) / 0.05127996   # +0.0%  mass_displaced3 > 3.208 and ak02_dr23 < 0
        - 0.0002540453 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) / math.sqrt(2)))) * (1.842068 * 0.5 * ((Q.lnerel_78 - -18.42068) / 1.842068) * (1 + math.erf(((Q.lnerel_78 - -18.42068) / 1.842068) / math.sqrt(2)))) / 4.914902   # -0.0%  mres_sd_mass_b0z005 > 159.9 and lnerel_78 > -18.42
        + 0.0001766522 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 3.208089) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 3.208089) / 1.000219) / math.sqrt(2)))) * (0.04684085 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.3582549) / 0.04684085) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.3582549) / 0.04684085) / math.sqrt(2)))) / 0.3452394   # +0.0%  mass_displaced3 > 3.208 and sjq_2_sumabs_k1 > 0.3583
        - 0.000136585 * (0.01171084 * 0.5 * ((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) * (0.04330444 * 0.5 * ((-0.1473389 - Q.eta_1) / 0.04330444) * (1 + math.erf(((-0.1473389 - Q.eta_1) / 0.04330444) / math.sqrt(2)))) / 0.0001006034   # -0.0%  sjf_2_1_z_d3 < 0.02623 and eta_1 < -0.1473
        - 0.0001246937 * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2)))) / 0.01976846   # -0.0%  sjq_3_2_k1 < -0.6015
        + 3.481903e-05 * (1.5 * 0.5 * ((18.0 - Q.sjq_2_1_nch) / 1.5) * (1 + math.erf(((18.0 - Q.sjq_2_1_nch) / 1.5) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.d0err_60 - 0.0) / 1e-06) * (1 + math.erf(((Q.d0err_60 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.002242132   # +0.0%  sjq_2_1_nch < 18 and d0err_60 > 0
        + 2.306718e-05 * (0.02801514 * 0.5 * ((Q.eta_9 - 0.09516602) / 0.02801514) * (1 + math.erf(((Q.eta_9 - 0.09516602) / 0.02801514) / math.sqrt(2)))) / 0.02319483   # +0.0%  eta_9 > 0.09517
        + 1.062639e-05 * (6.58003 * 0.5 * ((Q.mass_top30 - 126.5007) / 6.58003) * (1 + math.erf(((Q.mass_top30 - 126.5007) / 6.58003) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.iselectron_27 - 0.0) / 1e-06) * (1 + math.erf(((Q.iselectron_27 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.02367939   # +0.0%  mass_top30 > 126.5 and iselectron_27 > 0
    )
    return z


def neuron_105(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.732381e-05
    )
    return z


def neuron_106(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.160933e-05
    )
    return z


def neuron_107(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.15856e-05
    )
    return z


def neuron_108(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.470698e-06
    )
    return z


def neuron_109(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.200149e-07
    )
    return z


def neuron_110(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.5796e-06
    )
    return z


def neuron_111(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.669612e-06
    )
    return z


def neuron_112(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.23782e-06
    )
    return z


def neuron_113(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.665927e-05
    )
    return z


def neuron_114(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.677414e-06
    )
    return z


def neuron_115(Q):
    # scale S = 10.6; each line: share * term / its average size
    z = 10.59975 * (0.0400455
        - 0.07304177 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) / 31.66395   # -7.3%  mass_displaced3 < 39.1
        + 0.07157645 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) / 0.4378545   # +7.2%  lep_z < 0.5187
        - 0.06361388 * (0.01854575 * 0.5 * ((Q.N2 - 0.2250586) / 0.01854575) * (1 + math.erf(((Q.N2 - 0.2250586) / 0.01854575) / math.sqrt(2)))) / 0.08983552   # -6.4%  N2 > 0.2251
        + 0.04595833 * (0.008001329 * 0.5 * ((0.120439 - Q.M2) / 0.008001329) * (1 + math.erf(((0.120439 - Q.M2) / 0.008001329) / math.sqrt(2)))) / 0.04305026   # +4.6%  M2 < 0.1204
        - 0.03783838 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (0.06955833 * 0.5 * ((0.3788785 - Q.z_neutral_had) / 0.06955833) * (1 + math.erf(((0.3788785 - Q.z_neutral_had) / 0.06955833) / math.sqrt(2)))) / 0.0001502728   # -3.8%  e3_b2 < 0.000791 and z_neutral_had < 0.3789
        + 0.03664093 * (5.110145 * 0.5 * ((131.3917 - Q.mass) / 5.110145) * (1 + math.erf(((131.3917 - Q.mass) / 5.110145) / math.sqrt(2)))) / 24.73868   # +3.7%  mass < 131.4
        + 0.03402263 * (0.04122765 * 0.5 * ((Q.N2_b05 - 0.302405) / 0.04122765) * (1 + math.erf(((Q.N2_b05 - 0.302405) / 0.04122765) / math.sqrt(2)))) / 0.137646   # +3.4%  N2_b05 > 0.3024
        - 0.02973144 * (6.832602 * 0.5 * ((107.8099 - Q.sj4_pair_mass_max) / 6.832602) * (1 + math.erf(((107.8099 - Q.sj4_pair_mass_max) / 6.832602) / math.sqrt(2)))) / 29.75732   # -3.0%  sj4_pair_mass_max < 107.8
        + 0.0256433 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (0.1133023 * 0.5 * ((1.030977 - Q.pair_mean_lnkt) / 0.1133023) * (1 + math.erf(((1.030977 - Q.pair_mean_lnkt) / 0.1133023) / math.sqrt(2)))) / 0.0003603174   # +2.6%  e3_b2 < 0.000791 and pair_mean_lnkt < 1.031
        - 0.02419345 * (5.413174 * 0.5 * ((114.7848 - Q.sj3_pair_mass_max) / 5.413174) * (1 + math.erf(((114.7848 - Q.sj3_pair_mass_max) / 5.413174) / math.sqrt(2)))) / 26.84218   # -2.4%  sj3_pair_mass_max < 114.8
        - 0.02113667 * (0.006239604 * 0.5 * ((0.02148541 - Q.lam2) / 0.006239604) * (1 + math.erf(((0.02148541 - Q.lam2) / 0.006239604) / math.sqrt(2)))) / 0.01542465   # -2.1%  lam2 < 0.02149
        + 0.02112416 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) / 15.65331   # +2.1%  mass < 117.5
        - 0.02109246 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2261637   # -2.1%  lep_iso < 0.4382
        + 0.02050533 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) / 446.9928   # +2.1%  mass_displaced3 < 39.1 and lep_ptrel < 18.77
        + 0.01672459 * (3.876948 * 0.5 * ((6.767937 - Q.mass_2charged) / 3.876948) * (1 + math.erf(((6.767937 - Q.mass_2charged) / 3.876948) / math.sqrt(2)))) / 2.841815   # +1.7%  mass_2charged < 6.768
        + 0.01647962 * (0.008548367 * 0.5 * ((Q.N2_b05 - 0.4353632) / 0.008548367) * (1 + math.erf(((Q.N2_b05 - 0.4353632) / 0.008548367) / math.sqrt(2)))) / 0.03051078   # +1.6%  N2_b05 > 0.4354
        - 0.01633037 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) / 0.0006447212   # -1.6%  e3_b2 < 0.000791
        - 0.01540059 * (6.832602 * 0.5 * ((107.8099 - Q.sj4_pair_mass_max) / 6.832602) * (1 + math.erf(((107.8099 - Q.sj4_pair_mass_max) / 6.832602) / math.sqrt(2)))) * (5.37501 * 0.5 * ((9.356714 - Q.jd_3d_6) / 5.37501) * (1 + math.erf(((9.356714 - Q.jd_3d_6) / 5.37501) / math.sqrt(2)))) / 185.6063   # -1.5%  sj4_pair_mass_max < 107.8 and jd_3d_6 < 9.357
        - 0.01490624 * (0.05566634 * 0.5 * ((0.3093201 - Q.z_neutral_had) / 0.05566634) * (1 + math.erf(((0.3093201 - Q.z_neutral_had) / 0.05566634) / math.sqrt(2)))) / 0.1666885   # -1.5%  z_neutral_had < 0.3093
        - 0.01455122 * (36.85684 * 0.5 * ((77.03428 - Q.sv_1_sd0_sum) / 36.85684) * (1 + math.erf(((77.03428 - Q.sv_1_sd0_sum) / 36.85684) / math.sqrt(2)))) / 42.99541   # -1.5%  sv_1_sd0_sum < 77.03
        + 0.01425677 * (3.764443 * 0.5 * ((125.4927 - Q.mass_top50) / 3.764443) * (1 + math.erf(((125.4927 - Q.mass_top50) / 3.764443) / math.sqrt(2)))) / 21.21586   # +1.4%  mass_top50 < 125.5
        + 0.01379857 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2)))) / 0.001827643   # +1.4%  lep_z < 0.004136
        - 0.01301072 * (4.06561 * 0.5 * ((24.4079 - Q.mass_2charged) / 4.06561) * (1 + math.erf(((24.4079 - Q.mass_2charged) / 4.06561) / math.sqrt(2)))) / 14.84264   # -1.3%  mass_2charged < 24.41
        + 0.01293971 * (3.360928 * 0.5 * ((115.7429 - Q.mass_top40) / 3.360928) * (1 + math.erf(((115.7429 - Q.mass_top40) / 3.360928) / math.sqrt(2)))) * (10.60708 * 0.5 * ((16.11752 - Q.jd_3d_6) / 10.60708) * (1 + math.erf(((16.11752 - Q.jd_3d_6) / 10.60708) / math.sqrt(2)))) / 198.4906   # +1.3%  mass_top40 < 115.7 and jd_3d_6 < 16.12
        - 0.01201509 * (3.360928 * 0.5 * ((115.7429 - Q.mass_top40) / 3.360928) * (1 + math.erf(((115.7429 - Q.mass_top40) / 3.360928) / math.sqrt(2)))) / 16.18745   # -1.2%  mass_top40 < 115.7
        + 0.01200202 * (0.09724171 * 0.5 * ((3.234227 - Q.pair_max_lnkt) / 0.09724171) * (1 + math.erf(((3.234227 - Q.pair_max_lnkt) / 0.09724171) / math.sqrt(2)))) / 0.5859137   # +1.2%  pair_max_lnkt < 3.234
        + 0.01190048 * (2.202631e-05 * 0.5 * ((8.17233e-05 - Q.ecf_g42) / 2.202631e-05) * (1 + math.erf(((8.17233e-05 - Q.ecf_g42) / 2.202631e-05) / math.sqrt(2)))) / 5.709e-05   # +1.2%  ecf_g42 < 8.172e-05
        - 0.0115573 * (0.8395288 * 0.5 * ((1.839882 - Q.mass_2charged) / 0.8395288) * (1 + math.erf(((1.839882 - Q.mass_2charged) / 0.8395288) / math.sqrt(2)))) / 0.4323313   # -1.2%  mass_2charged < 1.84
        + 0.01149146 * (0.09740396 * 0.5 * ((1.325034 - Q.D2) / 0.09740396) * (1 + math.erf(((1.325034 - Q.D2) / 0.09740396) / math.sqrt(2)))) / 0.1068502   # +1.1%  D2 < 1.325
        - 0.01127434 * (5.468702 * 0.5 * ((Q.sd_mass - 88.81751) / 5.468702) * (1 + math.erf(((Q.sd_mass - 88.81751) / 5.468702) / math.sqrt(2)))) / 23.25878   # -1.1%  sd_mass > 88.82
        + 0.01107958 * Q.mass_neutral / 45.13363   # +1.1%  mass_neutral
        - 0.010768 * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) / 2.614   # -1.1%  lepsj_3_n_d3 < 3
        - 0.01047702 * (0.01206829 * 0.5 * ((0.8939856 - Q.tau43) / 0.01206829) * (1 + math.erf(((0.8939856 - Q.tau43) / 0.01206829) / math.sqrt(2)))) / 0.1025303   # -1.0%  tau43 < 0.894
        - 0.009584046 * (6.622935 * 0.5 * ((80.33914 - Q.dc_split1_kt) / 6.622935) * (1 + math.erf(((80.33914 - Q.dc_split1_kt) / 6.622935) / math.sqrt(2)))) / 31.42612   # -1.0%  dc_split1_kt < 80.34
        - 0.009531652 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2)))) / 0.5472155   # -1.0%  lep_z < 0.5187 and dc_n > 1
        + 0.009347259 * (2.35787 * 0.5 * ((Q.sj3_pair_mass_min - 26.64603) / 2.35787) * (1 + math.erf(((Q.sj3_pair_mass_min - 26.64603) / 2.35787) / math.sqrt(2)))) / 13.66164   # +0.9%  sj3_pair_mass_min > 26.65
        + 0.009274225 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) / 2.535129   # +0.9%  mass_displaced3 < 39.1 and lepsj_2_dr < 0.103
        + 0.008396196 * (0.01467775 * 0.5 * ((0.2243273 - Q.N2_b2) / 0.01467775) * (1 + math.erf(((0.2243273 - Q.N2_b2) / 0.01467775) / math.sqrt(2)))) * (5.37501 * 0.5 * ((9.356714 - Q.jd_3d_6) / 5.37501) * (1 + math.erf(((9.356714 - Q.jd_3d_6) / 5.37501) / math.sqrt(2)))) / 0.3809856   # +0.8%  N2_b2 < 0.2243 and jd_3d_6 < 9.357
        - 0.007790455 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_n - 1.0) / 1.0) * (1 + math.erf(((Q.ak02_n - 1.0) / 1.0) / math.sqrt(2)))) / 0.7077218   # -0.8%  lep_z < 0.5187 and ak02_n > 1
        - 0.007515822 * (0.1723618 * 0.5 * ((Q.ktd_ln_d34 - -9.531553) / 0.1723618) * (1 + math.erf(((Q.ktd_ln_d34 - -9.531553) / 0.1723618) / math.sqrt(2)))) / 0.6451565   # -0.8%  ktd_ln_d34 > -9.532
        + 0.007279651 * (0.02680828 * 0.5 * ((Q.dc_1_z - 0.6122903) / 0.02680828) * (1 + math.erf(((Q.dc_1_z - 0.6122903) / 0.02680828) / math.sqrt(2)))) / 0.08031388   # +0.7%  dc_1_z > 0.6123
        - 0.006571124 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 5.193195   # -0.7%  n_pairs_kt_above_1 < 80
        + 0.006441619 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (248.3135 * 0.5 * ((175.9957 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((175.9957 - Q.sip_3d_3) / 248.3135) / math.sqrt(2)))) / 89.03212   # +0.6%  n_s3d_above_3 > 4 and sip_3d_3 < 176
        + 0.006371938 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (12.86093 * 0.5 * ((Q.dc_split2_mass - 17.56529) / 12.86093) * (1 + math.erf(((Q.dc_split2_mass - 17.56529) / 12.86093) / math.sqrt(2)))) / 380.9171   # +0.6%  mass_displaced3 < 39.1 and dc_split2_mass > 17.57
        + 0.006156535 * (0.8395288 * 0.5 * ((1.839882 - Q.mass_2charged) / 0.8395288) * (1 + math.erf(((1.839882 - Q.mass_2charged) / 0.8395288) / math.sqrt(2)))) * (0.04514027 * 0.5 * ((0.1015913 - Q.sjf_4_2_z_d3) / 0.04514027) * (1 + math.erf(((0.1015913 - Q.sjf_4_2_z_d3) / 0.04514027) / math.sqrt(2)))) / 0.03593049   # +0.6%  mass_2charged < 1.84 and sjf_4_2_z_d3 < 0.1016
        - 0.006044796 * (459.2021 * 0.5 * ((804.4651 - Q.sip_3d_2) / 459.2021) * (1 + math.erf(((804.4651 - Q.sip_3d_2) / 459.2021) / math.sqrt(2)))) * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) / 499.2437   # -0.6%  sip_3d_2 < 804.5 and lep_iso < 1.362
        + 0.006030164 * (0.009766867 * 0.5 * ((0.09733903 - Q.tau2) / 0.009766867) * (1 + math.erf(((0.09733903 - Q.tau2) / 0.009766867) / math.sqrt(2)))) / 0.03185308   # +0.6%  tau2 < 0.09734
        - 0.005899113 * (0.6262913 * 0.5 * ((5.460258 - Q.sj3_mass3) / 0.6262913) * (1 + math.erf(((5.460258 - Q.sj3_mass3) / 0.6262913) / math.sqrt(2)))) / 1.668581   # -0.6%  sj3_mass3 < 5.46
        + 0.005898382 * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) * (1.0 * 0.5 * ((4.0 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((4.0 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2)))) / 0.2400857   # +0.6%  nca_sj4_pairmax_over_mass < 0.8538 and n_lund_kt_above_5 < 4
        + 0.005669038 * (3.260893 * 0.5 * ((86.78877 - Q.mass_top20) / 3.260893) * (1 + math.erf(((86.78877 - Q.mass_top20) / 3.260893) / math.sqrt(2)))) / 9.046583   # +0.6%  mass_top20 < 86.79
        + 0.005418362 * (1.0 * 0.5 * ((1.0 - Q.sjq_3_3_nch) / 1.0) * (1 + math.erf(((1.0 - Q.sjq_3_3_nch) / 1.0) / math.sqrt(2)))) / 0.1142307   # +0.5%  sjq_3_3_nch < 1
        + 0.00513622 * (1.721646 * 0.5 * ((5.763861 - Q.mass_2photon) / 1.721646) * (1 + math.erf(((5.763861 - Q.mass_2photon) / 1.721646) / math.sqrt(2)))) / 2.936504   # +0.5%  mass_2photon < 5.764
        - 0.005049213 * (48.0 * 0.5 * ((411.0 - Q.n_pairs_kt_above_1) / 48.0) * (1 + math.erf(((411.0 - Q.n_pairs_kt_above_1) / 48.0) / math.sqrt(2)))) / 159.3984   # -0.5%  n_pairs_kt_above_1 < 411
        + 0.00483479 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) / 1.234147   # +0.5%  n_s3d_above_3 > 4
        + 0.004800119 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.02439358 * 0.5 * ((0.1845735 - Q.sj4_dr_min) / 0.02439358) * (1 + math.erf(((0.1845735 - Q.sj4_dr_min) / 0.02439358) / math.sqrt(2)))) / 0.03445084   # +0.5%  lep_z < 0.5187 and sj4_dr_min < 0.1846
        - 0.004732113 * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) / 0.1490021   # -0.5%  nca_sj4_pairmax_over_mass < 0.8538
        - 0.004728144 * (0.01467775 * 0.5 * ((0.2243273 - Q.N2_b2) / 0.01467775) * (1 + math.erf(((0.2243273 - Q.N2_b2) / 0.01467775) / math.sqrt(2)))) / 0.06971979   # -0.5%  N2_b2 < 0.2243
        - 0.004665081 * (0.009691066 * 0.5 * ((0.1117947 - Q.dr_0) / 0.009691066) * (1 + math.erf(((0.1117947 - Q.dr_0) / 0.009691066) / math.sqrt(2)))) / 0.02715196   # -0.5%  dr_0 < 0.1118
        - 0.00452962 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.008350939 * 0.5 * ((0.8870464 - Q.tau54) / 0.008350939) * (1 + math.erf(((0.8870464 - Q.tau54) / 0.008350939) / math.sqrt(2)))) / 0.0221123   # -0.5%  lep_z < 0.5187 and tau54 < 0.887
        - 0.004092976 * (0.003950899 * 0.5 * ((Q.tau4 - 0.04996) / 0.003950899) * (1 + math.erf(((Q.tau4 - 0.04996) / 0.003950899) / math.sqrt(2)))) / 0.00425693   # -0.4%  tau4 > 0.04996
        + 0.003890829 * (14.63984 * 0.5 * ((414.6641 - Q.sum_pt_top10) / 14.63984) * (1 + math.erf(((414.6641 - Q.sum_pt_top10) / 14.63984) / math.sqrt(2)))) / 22.93628   # +0.4%  sum_pt_top10 < 414.7
        - 0.003882802 * (13.56554 * 0.5 * ((133.1648 - Q.mass_top15) / 13.56554) * (1 + math.erf(((133.1648 - Q.mass_top15) / 13.56554) / math.sqrt(2)))) / 51.89694   # -0.4%  mass_top15 < 133.2
        + 0.003197295 * (0.0164642 * 0.5 * ((0.6162845 - Q.nca_sj4_pairmax_over_mass) / 0.0164642) * (1 + math.erf(((0.6162845 - Q.nca_sj4_pairmax_over_mass) / 0.0164642) / math.sqrt(2)))) / 0.008934618   # +0.3%  nca_sj4_pairmax_over_mass < 0.6163
        + 0.00316589 * (459.2021 * 0.5 * ((804.4651 - Q.sip_3d_2) / 459.2021) * (1 + math.erf(((804.4651 - Q.sip_3d_2) / 459.2021) / math.sqrt(2)))) / 585.8706   # +0.3%  sip_3d_2 < 804.5
        + 0.002943007 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 1.0) / 1.0) * (1 + math.erf(((Q.n_muon - 1.0) / 1.0) / math.sqrt(2)))) / 9.35737e-05   # +0.3%  e3_b2 < 0.000791 and n_muon > 1
        - 0.002931297 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) / 0.06043165   # -0.3%  pz_lnd0 < 0.1713
        + 0.002848494 * (3.360928 * 0.5 * ((115.7429 - Q.mass_top40) / 3.360928) * (1 + math.erf(((115.7429 - Q.mass_top40) / 3.360928) / math.sqrt(2)))) * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 3.801973   # +0.3%  mass_top40 < 115.7 and lep_iso < 0.4382
        + 0.002446894 * (0.1723618 * 0.5 * ((Q.ktd_ln_d34 - -9.531553) / 0.1723618) * (1 + math.erf(((Q.ktd_ln_d34 - -9.531553) / 0.1723618) / math.sqrt(2)))) * (0.01029976 * 0.5 * ((0.102661 - Q.pz_lnkt3) / 0.01029976) * (1 + math.erf(((0.102661 - Q.pz_lnkt3) / 0.01029976) / math.sqrt(2)))) / 0.02969278   # +0.2%  ktd_ln_d34 > -9.532 and pz_lnkt3 < 0.1027
        - 0.002332308 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.9748591 * 0.5 * ((Q.sj4_pair_mass_min - 8.513203) / 0.9748591) * (1 + math.erf(((Q.sj4_pair_mass_min - 8.513203) / 0.9748591) / math.sqrt(2)))) / 3.726274   # -0.2%  lep_z < 0.5187 and sj4_pair_mass_min > 8.513
        + 0.00233103 * (0.008554338 * 0.5 * ((0.07722421 - Q.pz_lnd0) / 0.008554338) * (1 + math.erf(((0.07722421 - Q.pz_lnd0) / 0.008554338) / math.sqrt(2)))) / 0.008946562   # +0.2%  pz_lnd0 < 0.07722
        + 0.002106698 * (18.98174 * 0.5 * ((14.38858 - Q.lep_iso) / 18.98174) * (1 + math.erf(((14.38858 - Q.lep_iso) / 18.98174) / math.sqrt(2)))) / 9.394448   # +0.2%  lep_iso < 14.39
        + 0.001825449 * (0.03292552 * 0.5 * ((0.057042 - Q.sj2_zsoft) / 0.03292552) * (1 + math.erf(((0.057042 - Q.sj2_zsoft) / 0.03292552) / math.sqrt(2)))) / 0.00149658   # +0.2%  sj2_zsoft < 0.05704
        - 0.00163316 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) / 1.223619   # -0.2%  lepsj_3_maxsd0 < 2.413
        - 0.001461865 * (0.009766867 * 0.5 * ((0.09733903 - Q.tau2) / 0.009766867) * (1 + math.erf(((0.09733903 - Q.tau2) / 0.009766867) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_1_n_disp3 - 1.0) / 1.0) * (1 + math.erf(((Q.ak02_1_n_disp3 - 1.0) / 1.0) / math.sqrt(2)))) / 0.01409511   # -0.1%  tau2 < 0.09734 and ak02_1_n_disp3 > 1
        + 0.001437914 * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2)))) / 5.641924   # +0.1%  max_abs_d0 < 10.52
        + 0.001200398 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (0.02572129 * 0.5 * ((0.9433644 - Q.tau43_b2) / 0.02572129) * (1 + math.erf(((0.9433644 - Q.tau43_b2) / 0.02572129) / math.sqrt(2)))) / 0.2833147   # +0.1%  n_s3d_above_3 > 4 and tau43_b2 < 0.9434
        + 0.001198994 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.0331444 * 0.5 * ((Q.z_displaced5 - 0.1339658) / 0.0331444) * (1 + math.erf(((Q.z_displaced5 - 0.1339658) / 0.0331444) / math.sqrt(2)))) / 0.001268893   # +0.1%  pz_lnd0 < 0.1713 and z_displaced5 > 0.134
        + 0.001175544 * (3.20498 * 0.5 * ((34.02385 - Q.mass_neutral) / 3.20498) * (1 + math.erf(((34.02385 - Q.mass_neutral) / 3.20498) / math.sqrt(2)))) / 4.687056   # +0.1%  mass_neutral < 34.02
        + 0.001075296 * (0.1312538 * 0.5 * ((Q.jet_abs_eta - 1.323111) / 0.1312538) * (1 + math.erf(((Q.jet_abs_eta - 1.323111) / 0.1312538) / math.sqrt(2)))) / 0.03908429   # +0.1%  jet_abs_eta > 1.323
        + 0.001040213 * (4.06561 * 0.5 * ((24.4079 - Q.mass_2charged) / 4.06561) * (1 + math.erf(((24.4079 - Q.mass_2charged) / 4.06561) / math.sqrt(2)))) * (0.01273628 * 0.5 * ((Q.sdb_5_z - 0.04013001) / 0.01273628) * (1 + math.erf(((Q.sdb_5_z - 0.04013001) / 0.01273628) / math.sqrt(2)))) / 0.2719393   # +0.1%  mass_2charged < 24.41 and sdb_5_z > 0.04013
        + 0.0009204893 * (0.05566634 * 0.5 * ((0.3093201 - Q.z_neutral_had) / 0.05566634) * (1 + math.erf(((0.3093201 - Q.z_neutral_had) / 0.05566634) / math.sqrt(2)))) * (0.06273278 * 0.5 * ((Q.dr12 - 0.2116473) / 0.06273278) * (1 + math.erf(((Q.dr12 - 0.2116473) / 0.06273278) / math.sqrt(2)))) / 0.008075291   # +0.1%  z_neutral_had < 0.3093 and dr12 > 0.2116
        + 0.0006908085 * (2.952984 * 0.5 * ((75.78208 - Q.sj4_pair_mass_max) / 2.952984) * (1 + math.erf(((75.78208 - Q.sj4_pair_mass_max) / 2.952984) / math.sqrt(2)))) / 8.111448   # +0.1%  sj4_pair_mass_max < 75.78
        + 0.0006657223 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.1093626 * 0.5 * ((0.3699565 - Q.sv_2_dr) / 0.1093626) * (1 + math.erf(((0.3699565 - Q.sv_2_dr) / 0.1093626) / math.sqrt(2)))) / 0.017108   # +0.1%  pz_lnd0 < 0.1713 and sv_2_dr < 0.37
        + 0.0006601696 * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) * (0.1568179 * 0.5 * ((Q.sjq_2_sumabs_k03 - 1.261566) / 0.1568179) * (1 + math.erf(((Q.sjq_2_sumabs_k03 - 1.261566) / 0.1568179) / math.sqrt(2)))) / 0.00951657   # +0.1%  nca_sj4_pairmax_over_mass < 0.8538 and sjq_2_sumabs_k03 > 1.262
        - 0.0006598234 * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.isphoton_53) / 1.0) * (1 + math.erf(((0.0 - Q.isphoton_53) / 1.0) / math.sqrt(2)))) / 0.003533033   # -0.1%  nca_sj4_pairmax_over_mass < 0.8538 and isphoton_53 < 0
        - 0.0006476156 * (3.764443 * 0.5 * ((125.4927 - Q.mass_top50) / 3.764443) * (1 + math.erf(((125.4927 - Q.mass_top50) / 3.764443) / math.sqrt(2)))) * (0.01637522 * 0.5 * ((0.1243111 - Q.pz_lnd1) / 0.01637522) * (1 + math.erf(((0.1243111 - Q.pz_lnd1) / 0.01637522) / math.sqrt(2)))) / 1.073547   # -0.1%  mass_top50 < 125.5 and pz_lnd1 < 0.1243
        - 0.0005270607 * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) * (0.03352269 * 0.5 * ((0.6856395 - Q.z_charged_had) / 0.03352269) * (1 + math.erf(((0.6856395 - Q.z_charged_had) / 0.03352269) / math.sqrt(2)))) / 0.02892945   # -0.1%  nca_sj4_pairmax_over_mass < 0.8538 and z_charged_had < 0.6856
        - 0.0005128314 * (1.729885 * 0.5 * ((3.928781 - Q.sj2_mass2) / 1.729885) * (1 + math.erf(((3.928781 - Q.sj2_mass2) / 1.729885) / math.sqrt(2)))) / 0.2363684   # -0.1%  sj2_mass2 < 3.929
        - 0.0003919827 * (9.321454 * 0.5 * ((Q.mass_displaced5 - 21.12768) / 9.321454) * (1 + math.erf(((Q.mass_displaced5 - 21.12768) / 9.321454) / math.sqrt(2)))) / 1.847765   # -0.0%  mass_displaced5 > 21.13
        + 0.0002262663 * (6.832602 * 0.5 * ((107.8099 - Q.sj4_pair_mass_max) / 6.832602) * (1 + math.erf(((107.8099 - Q.sj4_pair_mass_max) / 6.832602) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_3_3_n_d3 - 3.0) / 1.0) * (1 + math.erf(((Q.sjf_3_3_n_d3 - 3.0) / 1.0) / math.sqrt(2)))) / 1.770164   # +0.0%  sj4_pair_mass_max < 107.8 and sjf_3_3_n_d3 > 3
        - 0.0002186458 * (0.01467775 * 0.5 * ((0.2243273 - Q.N2_b2) / 0.01467775) * (1 + math.erf(((0.2243273 - Q.N2_b2) / 0.01467775) / math.sqrt(2)))) * (0.03500366 * 0.5 * ((Q.d0err_52 - 0.03500366) / 0.03500366) * (1 + math.erf(((Q.d0err_52 - 0.03500366) / 0.03500366) / math.sqrt(2)))) / 0.0004529147   # -0.0%  N2_b2 < 0.2243 and d0err_52 > 0.035
        + 0.000203957 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.002034628   # +0.0%  pz_lnd0 < 0.1713 and kt2_2_n_lep < 0
        + 0.0001644637 * (0.4768076 * 0.5 * ((Q.ktd_ln_d34 - -7.075141) / 0.4768076) * (1 + math.erf(((Q.ktd_ln_d34 - -7.075141) / 0.4768076) / math.sqrt(2)))) / 0.03026856   # +0.0%  ktd_ln_d34 > -7.075
        - 0.0001518964 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (0.07359024 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.4723988) / 0.07359024) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.4723988) / 0.07359024) / math.sqrt(2)))) / 0.06867315   # -0.0%  n_s3d_above_3 > 4 and sjq_2_sumabs_k1 > 0.4724
        + 0.0001293996 * (4.883139 * 0.5 * ((89.68964 - Q.mass_top50) / 4.883139) * (1 + math.erf(((89.68964 - Q.mass_top50) / 4.883139) / math.sqrt(2)))) / 4.92169   # +0.0%  mass_top50 < 89.69
        - 8.824643e-05 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.2411197 * 0.5 * ((Q.jd_3d_6 - 0.9115584) / 0.2411197) * (1 + math.erf(((Q.jd_3d_6 - 0.9115584) / 0.2411197) / math.sqrt(2)))) / 0.6840455   # -0.0%  pz_lnd0 < 0.1713 and jd_3d_6 > 0.9116
        - 7.728583e-05 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (0.002122726 * 0.5 * ((Q.sjf_2_2_z_d3 - 0.0) / 0.002122726) * (1 + math.erf(((Q.sjf_2_2_z_d3 - 0.0) / 0.002122726) / math.sqrt(2)))) / 1.851598e-05   # -0.0%  e3_b2 < 0.000791 and sjf_2_2_z_d3 > 0
        - 3.770976e-05 * (0.01206829 * 0.5 * ((0.8939856 - Q.tau43) / 0.01206829) * (1 + math.erf(((0.8939856 - Q.tau43) / 0.01206829) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dc_mass_disp_2nd - 0.0) / 1e-06) * (1 + math.erf(((Q.dc_mass_disp_2nd - 0.0) / 1e-06) / math.sqrt(2)))) / 0.00600558   # -0.0%  tau43 < 0.894 and dc_mass_disp_2nd > 0
        + 2.872184e-05 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (0.1300849 * 0.5 * ((Q.dc_tag_2nd - 0.0) / 0.1300849) * (1 + math.erf(((Q.dc_tag_2nd - 0.0) / 0.1300849) / math.sqrt(2)))) / 13.11513   # +0.0%  mass_displaced3 < 39.1 and dc_tag_2nd > 0
    )
    return z


def neuron_116(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.399297e-07
    )
    return z


def neuron_117(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.862133e-06
    )
    return z


def neuron_118(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.233216e-06
    )
    return z


def neuron_119(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.336853e-07
    )
    return z


def neuron_120(Q):
    # scale S = 9.579; each line: share * term / its average size
    z = 9.579279 * (0.05472787
        + 0.128468 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) / math.sqrt(2)))) / 24.25944   # +12.8%  mres_sd_mass_b2z01 > 91.15
        - 0.1009614 * (5.55459 * 0.5 * ((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) / math.sqrt(2)))) / 20.84806   # -10.1%  mres_sd_mass_b2z01 > 96.62
        - 0.04979459 * (2.752946 * 0.5 * ((3.53503 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.53503 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 13.88745   # -5.0%  lep_ptrel < 3.535 and n_s3d_above_3 < 10
        - 0.04019013 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) / 8.618957   # -4.0%  lep_ptrel < 12.15
        + 0.0369205 * (9.37985 * 0.5 * ((76.15079 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((76.15079 - Q.mass_neutral) / 9.37985) / math.sqrt(2)))) / 32.62296   # +3.7%  mass_neutral < 76.15
        - 0.03274405 * (2.752946 * 0.5 * ((3.53503 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.53503 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) / 2.071478   # -3.3%  lep_ptrel < 3.535
        + 0.03258373 * (2.752946 * 0.5 * ((3.53503 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.53503 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (401.9953 * 0.5 * ((577.991 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((577.991 - Q.sip_3d_3) / 401.9953) / math.sqrt(2)))) / 977.6443   # +3.3%  lep_ptrel < 3.535 and sip_3d_3 < 578
        - 0.0290732 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) / 0.04874886   # -2.9%  lepsj_2_dr < 0.06573
        - 0.02507433 * (4.146938 * 0.5 * ((110.2019 - Q.mass) / 4.146938) * (1 + math.erf(((110.2019 - Q.mass) / 4.146938) / math.sqrt(2)))) / 12.01898   # -2.5%  mass < 110.2
        - 0.02495369 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 0.1259735   # -2.5%  lepsj_2_dr < 0.06573 and jd_3d_6 < 5.368
        - 0.0241531 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) / 0.4378545   # -2.4%  lep_z < 0.5187
        + 0.0240928 * (2.41177 * 0.5 * ((2.907433 - Q.lep_iso) / 2.41177) * (1 + math.erf(((2.907433 - Q.lep_iso) / 2.41177) / math.sqrt(2)))) / 1.920664   # +2.4%  lep_iso < 2.907
        + 0.02362947 * (0.03429006 * 0.5 * ((0.08178299 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.08178299 - Q.lep_dr) / 0.03429006) / math.sqrt(2)))) / 0.04862806   # +2.4%  lep_dr < 0.08178
        - 0.02336032 * (3.529599 * 0.5 * ((109.3251 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.3251 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) / 15.24743   # -2.3%  mass_top30 < 109.3
        - 0.02226302 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (9.37985 * 0.5 * ((76.15079 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((76.15079 - Q.mass_neutral) / 9.37985) / math.sqrt(2)))) / 12.87172   # -2.2%  lep_z < 0.5187 and mass_neutral < 76.15
        - 0.0207032 * (10.51039 * 0.5 * ((Q.mres_sd_mass_b2z01 - 69.12133) / 10.51039) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 69.12133) / 10.51039) / math.sqrt(2)))) / 40.76287   # -2.1%  mres_sd_mass_b2z01 > 69.12
        + 0.01878432 * (2.41177 * 0.5 * ((2.907433 - Q.lep_iso) / 2.41177) * (1 + math.erf(((2.907433 - Q.lep_iso) / 2.41177) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 5.067295   # +1.9%  lep_iso < 2.907 and jd_3d_6 < 5.368
        + 0.01636263 * (0.0005676896 * 0.5 * ((0.00228569 - Q.e3) / 0.0005676896) * (1 + math.erf(((0.00228569 - Q.e3) / 0.0005676896) / math.sqrt(2)))) / 0.001242197   # +1.6%  e3 < 0.002286
        - 0.01619207 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (1.929175 * 0.5 * ((4.004982 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((4.004982 - Q.jd_3d_5) / 1.929175) / math.sqrt(2)))) / 11.26363   # -1.6%  lep_ptrel < 12.15 and jd_3d_5 < 4.005
        + 0.01615518 * (2.752946 * 0.5 * ((3.53503 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.53503 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (2.0 * 0.5 * ((Q.n_photon - 5.0) / 2.0) * (1 + math.erf(((Q.n_photon - 5.0) / 2.0) / math.sqrt(2)))) / 25.16315   # +1.6%  lep_ptrel < 3.535 and n_photon > 5
        + 0.01493518 * (3.277554 * 0.5 * ((68.20711 - Q.mass_charged) / 3.277554) * (1 + math.erf(((68.20711 - Q.mass_charged) / 3.277554) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 39.78624   # +1.5%  mass_charged < 68.21 and jd_3d_6 < 5.368
        + 0.01484006 * Q.e3_b05 / 0.007432188   # +1.5%  e3_b05
        - 0.01413903 * (3.277554 * 0.5 * ((68.20711 - Q.mass_charged) / 3.277554) * (1 + math.erf(((68.20711 - Q.mass_charged) / 3.277554) / math.sqrt(2)))) / 12.83428   # -1.4%  mass_charged < 68.21
        - 0.01394406 * (5.110145 * 0.5 * ((131.3917 - Q.mass) / 5.110145) * (1 + math.erf(((131.3917 - Q.mass) / 5.110145) / math.sqrt(2)))) / 24.73868   # -1.4%  mass < 131.4
        + 0.01187675 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (1.5 * 0.5 * ((16.0 - Q.sdb_2_n) / 1.5) * (1 + math.erf(((16.0 - Q.sdb_2_n) / 1.5) / math.sqrt(2)))) / 45.09043   # +1.2%  lep_ptrel < 12.15 and sdb_2_n < 16
        - 0.01185489 * (8.0 * 0.5 * ((62.0 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.0 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) / 16.66856   # -1.2%  n_pairs_kt_above_3 < 62
        + 0.01175484 * (3.529599 * 0.5 * ((109.3251 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.3251 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2)))) / 45.7697   # +1.2%  mass_top30 < 109.3 and jd_3d_6 < 5.368
        + 0.01071836 * (11.4891 * 0.5 * ((32.61679 - Q.mass_displaced5) / 11.4891) * (1 + math.erf(((32.61679 - Q.mass_displaced5) / 11.4891) / math.sqrt(2)))) / 26.79867   # +1.1%  mass_displaced5 < 32.62
        + 0.01057051 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (4.4703e-08 * 0.5 * ((Q.mass_displaced3 - 0.0) / 4.4703e-08) * (1 + math.erf(((Q.mass_displaced3 - 0.0) / 4.4703e-08) / math.sqrt(2)))) / 3.638459   # +1.1%  lep_z < 0.5187 and mass_displaced3 > 0
        - 0.0103869 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2)))) / 0.001827643   # -1.0%  lep_z < 0.004136
        - 0.01029948 * (2.41177 * 0.5 * ((2.907433 - Q.lep_iso) / 2.41177) * (1 + math.erf(((2.907433 - Q.lep_iso) / 2.41177) / math.sqrt(2)))) * (1032.513 * 0.5 * ((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) * (1 + math.erf(((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) / math.sqrt(2)))) / 1895.95   # -1.0%  lep_iso < 2.907 and lepsj_2_maxsd0 < 1164
        - 0.009489832 * (2.752946 * 0.5 * ((3.53503 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.53503 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (3.652904e-06 * 0.5 * ((7.184834e-06 - Q.e4) / 3.652904e-06) * (1 + math.erf(((7.184834e-06 - Q.e4) / 3.652904e-06) / math.sqrt(2)))) / 1.251385e-05   # -0.9%  lep_ptrel < 3.535 and e4 < 7.185e-06
        - 0.008584666 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ismuon_0 - 1.0) / 1.0) * (1 + math.erf(((Q.ismuon_0 - 1.0) / 1.0) / math.sqrt(2)))) / 0.3939196   # -0.9%  max_abs_d0 < 5.812 and ismuon_0 > 1
        + 0.008391094 * (5.110145 * 0.5 * ((131.3917 - Q.mass) / 5.110145) * (1 + math.erf(((131.3917 - Q.mass) / 5.110145) / math.sqrt(2)))) * (1.929175 * 0.5 * ((4.004982 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((4.004982 - Q.jd_3d_5) / 1.929175) / math.sqrt(2)))) / 41.9565   # +0.8%  mass < 131.4 and jd_3d_5 < 4.005
        - 0.008332484 * (0.3071165 * 0.5 * ((Q.lund3_lndelta - -2.4279) / 0.3071165) * (1 + math.erf(((Q.lund3_lndelta - -2.4279) / 0.3071165) / math.sqrt(2)))) / 0.9592431   # -0.8%  lund3_lndelta > -2.428
        - 0.006638471 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) / 2.75999   # -0.7%  max_abs_d0 < 5.812
        + 0.006588915 * (0.008162106 * 0.5 * ((0.1589878 - Q.pz_lnkt0) / 0.008162106) * (1 + math.erf(((0.1589878 - Q.pz_lnkt0) / 0.008162106) / math.sqrt(2)))) / 0.0291817   # +0.7%  pz_lnkt0 < 0.159
        - 0.006397955 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (1.387908 * 0.5 * ((3.151933 - Q.jd_3d_6) / 1.387908) * (1 + math.erf(((3.151933 - Q.jd_3d_6) / 1.387908) / math.sqrt(2)))) / 7.966652   # -0.6%  lep_ptrel < 12.15 and jd_3d_6 < 3.152
        - 0.006021043 * (1.0 * 0.5 * ((0.0 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2)))) / 0.03808846   # -0.6%  kt2_1_n_lep < 0
        + 0.005695337 * (0.02757255 * 0.5 * ((0.6800935 - Q.tau32) / 0.02757255) * (1 + math.erf(((0.6800935 - Q.tau32) / 0.02757255) / math.sqrt(2)))) / 0.08068051   # +0.6%  tau32 < 0.6801
        + 0.005437741 * (0.006899392 * 0.5 * ((0.006470637 - Q.lepsj_2_dr) / 0.006899392) * (1 + math.erf(((0.006470637 - Q.lepsj_2_dr) / 0.006899392) / math.sqrt(2)))) / 0.003371104   # +0.5%  lepsj_2_dr < 0.006471
        + 0.005336904 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.n_electron) / 1.0) * (1 + math.erf(((1.0 - Q.n_electron) / 1.0) / math.sqrt(2)))) / 1.713975   # +0.5%  max_abs_d0 < 5.812 and n_electron < 1
        + 0.004575182 * (3.529599 * 0.5 * ((109.3251 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.3251 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (0.0139093 * 0.5 * ((0.1271955 - Q.z_neutral_had) / 0.0139093) * (1 + math.erf(((0.1271955 - Q.z_neutral_had) / 0.0139093) / math.sqrt(2)))) / 0.6105978   # +0.5%  mass_top30 < 109.3 and z_neutral_had < 0.1272
        - 0.004498813 * (1.5 * 0.5 * ((11.0 - Q.n_pairs_kt_above_10) / 1.5) * (1 + math.erf(((11.0 - Q.n_pairs_kt_above_10) / 1.5) / math.sqrt(2)))) / 5.848932   # -0.4%  n_pairs_kt_above_10 < 11
        + 0.00405117 * (0.003235318 * 0.5 * ((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) * (1 + math.erf(((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) / math.sqrt(2)))) / 0.0003162318   # +0.4%  lepsj_2_dr < 0.000821
        + 0.003741036 * (13.85751 * 0.5 * ((91.2852 - Q.sj2_mass1) / 13.85751) * (1 + math.erf(((91.2852 - Q.sj2_mass1) / 13.85751) / math.sqrt(2)))) / 51.70226   # +0.4%  sj2_mass1 < 91.29
        + 0.003569515 * (0.01151941 * 0.5 * ((Q.N2 - 0.3245983) / 0.01151941) * (1 + math.erf(((Q.N2 - 0.3245983) / 0.01151941) / math.sqrt(2)))) / 0.02635739   # +0.4%  N2 > 0.3246
        - 0.003274216 * (0.08036477 * 0.5 * ((Q.pair_mean_lndelta - -1.714682) / 0.08036477) * (1 + math.erf(((Q.pair_mean_lndelta - -1.714682) / 0.08036477) / math.sqrt(2)))) / 0.05193348   # -0.3%  pair_mean_lndelta > -1.715
        - 0.003211475 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2)))) / 5.193195   # -0.3%  n_pairs_kt_above_1 < 80
        + 0.003113666 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (0.2385164 * 0.5 * ((Q.ak02_dr23 - 0.0) / 0.2385164) * (1 + math.erf(((Q.ak02_dr23 - 0.0) / 0.2385164) / math.sqrt(2)))) / 0.009996905   # +0.3%  lepsj_2_dr < 0.06573 and ak02_dr23 > 0
        + 0.003004502 * (8.0 * 0.5 * ((62.0 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.0 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) * (0.0792551 * 0.5 * ((0.1822601 - Q.lepsj_2_dr) / 0.0792551) * (1 + math.erf(((0.1822601 - Q.lepsj_2_dr) / 0.0792551) / math.sqrt(2)))) / 2.746722   # +0.3%  n_pairs_kt_above_3 < 62 and lepsj_2_dr < 0.1823
        - 0.00272833 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2261637   # -0.3%  lep_iso < 0.4382
        + 0.0026683 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.09724841 * 0.5 * ((0.4305249 - Q.sjq_3_1_k05) / 0.09724841) * (1 + math.erf(((0.4305249 - Q.sjq_3_1_k05) / 0.09724841) / math.sqrt(2)))) / 3.89982   # +0.3%  lep_ptrel < 12.15 and sjq_3_1_k05 < 0.4305
        + 0.00246731 * (36.4359 * 0.5 * ((28.96878 - Q.sv_2_sd0_sum) / 36.4359) * (1 + math.erf(((28.96878 - Q.sv_2_sd0_sum) / 36.4359) / math.sqrt(2)))) / 17.25512   # +0.2%  sv_2_sd0_sum < 28.97
        - 0.002422357 * (1.0 * 0.5 * ((3.0 - Q.n_sd0_above_5) / 1.0) * (1 + math.erf(((3.0 - Q.n_sd0_above_5) / 1.0) / math.sqrt(2)))) / 1.169248   # -0.2%  n_sd0_above_5 < 3
        - 0.002405221 * (0.00189604 * 0.5 * ((0.01394245 - Q.dr_min_012) / 0.00189604) * (1 + math.erf(((0.01394245 - Q.dr_min_012) / 0.00189604) / math.sqrt(2)))) / 0.002966383   # -0.2%  dr_min_012 < 0.01394
        - 0.002383281 * (0.02132677 * 0.5 * ((Q.dc_split1_dr - 0.2818852) / 0.02132677) * (1 + math.erf(((Q.dc_split1_dr - 0.2818852) / 0.02132677) / math.sqrt(2)))) / 0.09939548   # -0.2%  dc_split1_dr > 0.2819
        - 0.002183777 * (0.2156905 * 0.5 * ((2.41868 - Q.sjf_2_2_max3d) / 0.2156905) * (1 + math.erf(((2.41868 - Q.sjf_2_2_max3d) / 0.2156905) / math.sqrt(2)))) / 0.2736596   # -0.2%  sjf_2_2_max3d < 2.419
        - 0.002037393 * (1.0 * 0.5 * ((0.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.03307822   # -0.2%  kt2_2_n_lep < 0
        + 0.0019132 * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) / math.sqrt(2)))) / 0.0357357   # +0.2%  sjq_3_3_k1 > 0.4037
        - 0.001810314 * (9.37985 * 0.5 * ((76.15079 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((76.15079 - Q.mass_neutral) / 9.37985) / math.sqrt(2)))) * (0.03369438 * 0.5 * ((Q.sj3_dr_max - 0.5700825) / 0.03369438) * (1 + math.erf(((Q.sj3_dr_max - 0.5700825) / 0.03369438) / math.sqrt(2)))) / 1.91354   # -0.2%  mass_neutral < 76.15 and sj3_dr_max > 0.5701
        - 0.001781713 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2)))) / 2.198028   # -0.2%  lep_ptrel > 27.32
        + 0.001727829 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2)))) / 0.003263441   # +0.2%  lep_iso < 0.4382 and dc_1_n_lep < 0
        + 0.001634338 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.kt2_2_n_disp3 - 0.0) / 1.0) * (1 + math.erf(((Q.kt2_2_n_disp3 - 0.0) / 1.0) / math.sqrt(2)))) / 0.2725932   # +0.2%  lep_z < 0.5187 and kt2_2_n_disp3 > 0
        + 0.001533518 * (8.0 * 0.5 * ((62.0 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.0 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.6015744   # +0.2%  n_pairs_kt_above_3 < 62 and dc_2_n_lep < 0
        + 0.001440408 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2)))) / 4.97365   # +0.1%  mass < 90.09
        - 0.001389154 * (164.9991 * 0.5 * ((226.3008 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((226.3008 - Q.sip_3d_2) / 164.9991) / math.sqrt(2)))) / 133.7832   # -0.1%  sip_3d_2 < 226.3
        - 0.001331713 * (0.7185011 * 0.5 * ((Q.pair_mean_lnm2 - 4.787589) / 0.7185011) * (1 + math.erf(((Q.pair_mean_lnm2 - 4.787589) / 0.7185011) / math.sqrt(2)))) / 0.05138015   # -0.1%  pair_mean_lnm2 > 4.788
        + 0.001270457 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.09432096 * 0.5 * ((-0.2912597 - Q.sjq_3_3_k1) / 0.09432096) * (1 + math.erf(((-0.2912597 - Q.sjq_3_3_k1) / 0.09432096) / math.sqrt(2)))) / 0.01808236   # +0.1%  lep_z < 0.5187 and sjq_3_3_k1 < -0.2913
        - 0.001090929 * (0.03214999 * 0.5 * ((0.3436326 - Q.N2_b05) / 0.03214999) * (1 + math.erf(((0.3436326 - Q.N2_b05) / 0.03214999) / math.sqrt(2)))) / 0.005971875   # -0.1%  N2_b05 < 0.3436
        - 0.001080943 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) / math.sqrt(2)))) / 0.001629875   # -0.1%  lepsj_2_dr < 0.06573 and sjq_3_3_k1 > 0.4037
        + 0.0009999631 * (1.0 * 0.5 * ((0.0 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2)))) * (0.0812128 * 0.5 * ((Q.sjq_2_prod_k05 - -0.2405169) / 0.0812128) * (1 + math.erf(((Q.sjq_2_prod_k05 - -0.2405169) / 0.0812128) / math.sqrt(2)))) / 0.00799683   # +0.1%  kt2_1_n_lep < 0 and sjq_2_prod_k05 > -0.2405
        - 0.0009312109 * (20.30421 * 0.5 * ((Q.dc_split1_mass - 154.6884) / 20.30421) * (1 + math.erf(((Q.dc_split1_mass - 154.6884) / 20.30421) / math.sqrt(2)))) / 3.783855   # -0.1%  dc_split1_mass > 154.7
        - 0.0009297332 * Q.lund3_lnkt / 2.035383   # -0.1%  lund3_lnkt
        - 0.0009141128 * (10.67504 * 0.5 * ((Q.mass_top10 - 103.4976) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 103.4976) / 10.67504) / math.sqrt(2)))) / 2.014523   # -0.1%  mass_top10 > 103.5
        - 0.0006743954 * (3.277554 * 0.5 * ((68.20711 - Q.mass_charged) / 3.277554) * (1 + math.erf(((68.20711 - Q.mass_charged) / 3.277554) / math.sqrt(2)))) * (0.04703017 * 0.5 * ((0.5081296 - Q.planar_flow) / 0.04703017) * (1 + math.erf(((0.5081296 - Q.planar_flow) / 0.04703017) / math.sqrt(2)))) / 1.522082   # -0.1%  mass_charged < 68.21 and planar_flow < 0.5081
        - 0.0006729736 * (10.67504 * 0.5 * ((Q.mass_top10 - 103.4976) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 103.4976) / 10.67504) / math.sqrt(2)))) * (0.04703017 * 0.5 * ((0.5081296 - Q.planar_flow) / 0.04703017) * (1 + math.erf(((0.5081296 - Q.planar_flow) / 0.04703017) / math.sqrt(2)))) / 0.3545125   # -0.1%  mass_top10 > 103.5 and planar_flow < 0.5081
        - 0.0005214437 * (3.529599 * 0.5 * ((109.3251 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.3251 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (0.005973664 * 0.5 * ((Q.sv_2_z - 0.01148432) / 0.005973664) * (1 + math.erf(((Q.sv_2_z - 0.01148432) / 0.005973664) / math.sqrt(2)))) / 0.02988809   # -0.1%  mass_top30 < 109.3 and sv_2_z > 0.01148
        + 0.000514843 * (0.01538599 * 0.5 * ((0.07823361 - Q.pz_lnkt0) / 0.01538599) * (1 + math.erf(((0.07823361 - Q.pz_lnkt0) / 0.01538599) / math.sqrt(2)))) / 0.002510226   # +0.1%  pz_lnkt0 < 0.07823
        - 0.0005123258 * (0.9486798 * 0.5 * ((4.6892 - Q.mres_sd_prong_mass2) / 0.9486798) * (1 + math.erf(((4.6892 - Q.mres_sd_prong_mass2) / 0.9486798) / math.sqrt(2)))) / 0.7356014   # -0.1%  mres_sd_prong_mass2 < 4.689
        + 0.0004570661 * (0.009024806 * 0.5 * ((0.05997821 - Q.pz_lnd0) / 0.009024806) * (1 + math.erf(((0.05997821 - Q.pz_lnd0) / 0.009024806) / math.sqrt(2)))) / 0.004680032   # +0.0%  pz_lnd0 < 0.05998
        + 0.0004402606 * (0.008162106 * 0.5 * ((0.1589878 - Q.pz_lnkt0) / 0.008162106) * (1 + math.erf(((0.1589878 - Q.pz_lnkt0) / 0.008162106) / math.sqrt(2)))) * (0.04133456 * 0.5 * ((Q.sj3_dr13 - 0.1541) / 0.04133456) * (1 + math.erf(((Q.sj3_dr13 - 0.1541) / 0.04133456) / math.sqrt(2)))) / 0.007762142   # +0.0%  pz_lnkt0 < 0.159 and sj3_dr13 > 0.1541
        - 0.0003802214 * (8.0 * 0.5 * ((62.0 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.0 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) * (0.08055962 * 0.5 * ((0.2710171 - Q.sjf_2_1_z_d3) / 0.08055962) * (1 + math.erf(((0.2710171 - Q.sjf_2_1_z_d3) / 0.08055962) / math.sqrt(2)))) / 3.516929   # -0.0%  n_pairs_kt_above_3 < 62 and sjf_2_1_z_d3 < 0.271
        - 0.0003722952 * (0.2186175 * 0.5 * ((-3.643775 - Q.lnerel_4) / 0.2186175) * (1 + math.erf(((-3.643775 - Q.lnerel_4) / 0.2186175) / math.sqrt(2)))) / 0.02285097   # -0.0%  lnerel_4 < -3.644
        - 0.0003530476 * (0.009387389 * 0.5 * ((0.03277088 - Q.pz_lnd2) / 0.009387389) * (1 + math.erf(((0.03277088 - Q.pz_lnd2) / 0.009387389) / math.sqrt(2)))) / 0.003229967   # -0.0%  pz_lnd2 < 0.03277
        - 0.0003208881 * (0.02153986 * 0.5 * ((0.4646004 - Q.z_neutral) / 0.02153986) * (1 + math.erf(((0.4646004 - Q.z_neutral) / 0.02153986) / math.sqrt(2)))) / 0.09957918   # -0.0%  z_neutral < 0.4646
        + 0.000288724 * (1.0 * 0.5 * ((0.0 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2)))) * (0.01040872 * 0.5 * ((Q.sdb_4_z - 0.01435877) / 0.01040872) * (1 + math.erf(((Q.sdb_4_z - 0.01435877) / 0.01040872) / math.sqrt(2)))) / 0.0008001338   # +0.0%  kt2_1_n_lep < 0 and sdb_4_z > 0.01436
        + 0.0002751017 * (1.0 * 0.5 * ((3.0 - Q.n_sd0_above_5) / 1.0) * (1 + math.erf(((3.0 - Q.n_sd0_above_5) / 1.0) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2)))) / 0.04526133   # +0.0%  n_sd0_above_5 < 3 and ak02_3_n_lep > 0
        + 0.0002543428 * (1.5 * 0.5 * ((7.0 - Q.n_dr_0p1_0p2) / 1.5) * (1 + math.erf(((7.0 - Q.n_dr_0p1_0p2) / 1.5) / math.sqrt(2)))) / 1.165522   # +0.0%  n_dr_0p1_0p2 < 7
        + 0.0002267231 * (13.56554 * 0.5 * ((Q.mass_top15 - 133.1648) / 13.56554) * (1 + math.erf(((Q.mass_top15 - 133.1648) / 13.56554) / math.sqrt(2)))) * (10.58345 * 0.5 * ((Q.sum_pt_top40 - 569.6282) / 10.58345) * (1 + math.erf(((Q.sum_pt_top40 - 569.6282) / 10.58345) / math.sqrt(2)))) / 140.9367   # +0.0%  mass_top15 > 133.2 and sum_pt_top40 > 569.6
        - 0.0002156641 * (13.56554 * 0.5 * ((Q.mass_top15 - 133.1648) / 13.56554) * (1 + math.erf(((Q.mass_top15 - 133.1648) / 13.56554) / math.sqrt(2)))) / 1.236077   # -0.0%  mass_top15 > 133.2
        + 0.0001741175 * (5.416144 * 0.5 * ((Q.sj4_pair_mass_max - 101.849) / 5.416144) * (1 + math.erf(((Q.sj4_pair_mass_max - 101.849) / 5.416144) / math.sqrt(2)))) / 4.112215   # +0.0%  sj4_pair_mass_max > 101.8
        + 0.0001491332 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) / 0.9643417   # +0.0%  lep_ptrel > 43.21
        - 0.0001233202 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (14.45336 * 0.5 * ((Q.jd_3d_6 - 30.57088) / 14.45336) * (1 + math.erf(((Q.jd_3d_6 - 30.57088) / 14.45336) / math.sqrt(2)))) / 2.727837   # -0.0%  lep_z < 0.5187 and jd_3d_6 > 30.57
        - 9.486307e-05 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.007382626 * 0.5 * ((Q.pz_lnd2 - 0.008992646) / 0.007382626) * (1 + math.erf(((Q.pz_lnd2 - 0.008992646) / 0.007382626) / math.sqrt(2)))) / 0.9531053   # -0.0%  lep_ptrel < 12.15 and pz_lnd2 > 0.008993
        + 7.422993e-05 * (0.01519451 * 0.5 * ((Q.psi_0p2 - 0.9892758) / 0.01519451) * (1 + math.erf(((Q.psi_0p2 - 0.9892758) / 0.01519451) / math.sqrt(2)))) / 0.0003783519   # +0.0%  psi_0p2 > 0.9893
        - 4.218035e-05 * (0.01538599 * 0.5 * ((0.07823361 - Q.pz_lnkt0) / 0.01538599) * (1 + math.erf(((0.07823361 - Q.pz_lnkt0) / 0.01538599) / math.sqrt(2)))) * (0.1714986 * 0.5 * ((Q.lund2_lnz - -0.9953121) / 0.1714986) * (1 + math.erf(((Q.lund2_lnz - -0.9953121) / 0.1714986) / math.sqrt(2)))) / 3.735839e-05   # -0.0%  pz_lnkt0 < 0.07823 and lund2_lnz > -0.9953
        - 2.798257e-05 * (1.0 * 0.5 * ((0.0 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.isphoton_42) / 1.0) * (1 + math.erf(((1.0 - Q.isphoton_42) / 1.0) / math.sqrt(2)))) / 0.02625444   # -0.0%  kt2_1_n_lep < 0 and isphoton_42 < 1
        - 2.385864e-05 * (0.00189604 * 0.5 * ((0.01394245 - Q.dr_min_012) / 0.00189604) * (1 + math.erf(((0.01394245 - Q.dr_min_012) / 0.00189604) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.iselectron_1 - 0.0) / 1e-06) * (1 + math.erf(((Q.iselectron_1 - 0.0) / 1e-06) / math.sqrt(2)))) / 5.117587e-05   # -0.0%  dr_min_012 < 0.01394 and iselectron_1 > 0
        + 6.584363e-07 * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.iselectron_12 - 0.0) / 1e-06) * (1 + math.erf(((Q.iselectron_12 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.0002012655   # +0.0%  sjq_3_3_k1 > 0.4037 and iselectron_12 > 0
    )
    return z


def neuron_121(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.730511e-06
    )
    return z


def neuron_122(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.395899e-06
    )
    return z


def neuron_123(Q):
    # scale S = 3.738; each line: share * term / its average size
    z = 3.738316 * (-0.05932106
        + 0.07263073 * (0.100181 * 0.5 * ((0.3803178 - Q.z_displaced5) / 0.100181) * (1 + math.erf(((0.3803178 - Q.z_displaced5) / 0.100181) / math.sqrt(2)))) * (0.04864387 * 0.5 * ((0.06457187 - Q.lep_z) / 0.04864387) * (1 + math.erf(((0.06457187 - Q.lep_z) / 0.04864387) / math.sqrt(2)))) / 0.01236481   # +7.3%  z_displaced5 < 0.3803 and lep_z < 0.06457
        - 0.06494629 * (0.04864387 * 0.5 * ((0.06457187 - Q.lep_z) / 0.04864387) * (1 + math.erf(((0.06457187 - Q.lep_z) / 0.04864387) / math.sqrt(2)))) / 0.04047035   # -6.5%  lep_z < 0.06457
        - 0.05094996 * (0.100181 * 0.5 * ((0.3803178 - Q.z_displaced5) / 0.100181) * (1 + math.erf(((0.3803178 - Q.z_displaced5) / 0.100181) / math.sqrt(2)))) / 0.2913591   # -5.1%  z_displaced5 < 0.3803
        + 0.04658949 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.117671 * 0.5 * ((0.4191372 - Q.lep_dr) / 0.117671) * (1 + math.erf(((0.4191372 - Q.lep_dr) / 0.117671) / math.sqrt(2)))) / 0.1501102   # +4.7%  lep_z < 0.5187 and lep_dr < 0.4191
        - 0.04339373 * (5.10267 * 0.5 * ((Q.mres_sd_mass_b0z005 - 86.65765) / 5.10267) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 86.65765) / 5.10267) / math.sqrt(2)))) / 27.85294   # -4.3%  mres_sd_mass_b0z005 > 86.66
        + 0.04241775 * (6.875137 * 0.5 * ((18.80005 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.80005 - Q.mass_displaced3) / 6.875137) / math.sqrt(2)))) / 13.2945   # +4.2%  mass_displaced3 < 18.8
        + 0.03689882 * (4.539253 * 0.5 * ((Q.mres_sd_mass_b0z005 - 107.7493) / 4.539253) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 107.7493) / 4.539253) / math.sqrt(2)))) / 15.19886   # +3.7%  mres_sd_mass_b0z005 > 107.7
        - 0.02843232 * (0.04864387 * 0.5 * ((0.06457187 - Q.lep_z) / 0.04864387) * (1 + math.erf(((0.06457187 - Q.lep_z) / 0.04864387) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.lepsj_2_n_d3) / 1.0) * (1 + math.erf(((1.0 - Q.lepsj_2_n_d3) / 1.0) / math.sqrt(2)))) / 0.03001067   # -2.8%  lep_z < 0.06457 and lepsj_2_n_d3 < 1
        + 0.02826444 * (24.25363 * 0.5 * ((Q.mres_sd_mass_b0z005 - 53.57509) / 24.25363) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 53.57509) / 24.25363) / math.sqrt(2)))) / 54.35711   # +2.8%  mres_sd_mass_b0z005 > 53.58
        + 0.02770165 * (0.005331567 * 0.5 * ((Q.M2_b05 - 0.1209237) / 0.005331567) * (1 + math.erf(((Q.M2_b05 - 0.1209237) / 0.005331567) / math.sqrt(2)))) / 0.02184554   # +2.8%  M2_b05 > 0.1209
        + 0.02422139 * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2)))) / 140.6595   # +2.4%  jd_3d_4 < 172.9
        - 0.02366523 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.04048364 * 0.5 * ((Q.ak02_dr23 - 0.2759933) / 0.04048364) * (1 + math.erf(((Q.ak02_dr23 - 0.2759933) / 0.04048364) / math.sqrt(2)))) / 0.04768377   # -2.4%  lep_z < 0.5187 and ak02_dr23 > 0.276
        + 0.02343419 * (0.02363876 * 0.5 * ((0.4744137 - Q.z_charged_had) / 0.02363876) * (1 + math.erf(((0.4744137 - Q.z_charged_had) / 0.02363876) / math.sqrt(2)))) / 0.05600666   # +2.3%  z_charged_had < 0.4744
        + 0.02233603 * (0.04048364 * 0.5 * ((Q.ak02_dr23 - 0.2759933) / 0.04048364) * (1 + math.erf(((Q.ak02_dr23 - 0.2759933) / 0.04048364) / math.sqrt(2)))) / 0.1050422   # +2.2%  ak02_dr23 > 0.276
        + 0.02023188 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.sv_1_n) / 1.0) * (1 + math.erf(((3.0 - Q.sv_1_n) / 1.0) / math.sqrt(2)))) / 0.01153394   # +2.0%  M2_b2 < 0.02927 and sv_1_n < 3
        - 0.02014699 * (1032.513 * 0.5 * ((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) * (1 + math.erf(((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) / math.sqrt(2)))) / 936.2819   # -2.0%  lepsj_2_maxsd0 < 1164
        - 0.01940127 * (1.400504 * 0.5 * ((14.42172 - Q.sj2_mass2) / 1.400504) * (1 + math.erf(((14.42172 - Q.sj2_mass2) / 1.400504) / math.sqrt(2)))) / 3.645471   # -1.9%  sj2_mass2 < 14.42
        + 0.0186821 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2261637   # +1.9%  lep_iso < 0.4382
        - 0.01846456 * (1.0 * 0.5 * ((4.0 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((4.0 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2)))) / 2.068967   # -1.8%  n_sdz_above_5 < 4
        + 0.01763393 * (0.04864387 * 0.5 * ((0.06457187 - Q.lep_z) / 0.04864387) * (1 + math.erf(((0.06457187 - Q.lep_z) / 0.04864387) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2)))) / 0.04567881   # +1.8%  lep_z < 0.06457 and sip_3d_3 < 4.606
        - 0.01565177 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2)))) / 2.198028   # -1.6%  lep_ptrel > 27.32
        + 0.01522205 * Q.sjf_2_1_z_d3 / 0.08151038   # +1.5%  sjf_2_1_z_d3
        + 0.01437564 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (0.03294892 * 0.5 * ((0.08598434 - Q.kt2_2_z_disp3) / 0.03294892) * (1 + math.erf(((0.08598434 - Q.kt2_2_z_disp3) / 0.03294892) / math.sqrt(2)))) / 0.03735847   # +1.4%  sjq_2_2_nch < 4 and kt2_2_z_disp3 < 0.08598
        - 0.01424519 * (0.01140748 * 0.5 * ((0.0573796 - Q.pz_lnd3) / 0.01140748) * (1 + math.erf(((0.0573796 - Q.pz_lnd3) / 0.01140748) / math.sqrt(2)))) * (0.05999923 * 0.5 * ((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) * (1 + math.erf(((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) / math.sqrt(2)))) / 0.002245723   # -1.4%  pz_lnd3 < 0.05738 and sjf_3_1_z_d3 < 0.1941
        + 0.01416727 * (0.04864387 * 0.5 * ((0.06457187 - Q.lep_z) / 0.04864387) * (1 + math.erf(((0.06457187 - Q.lep_z) / 0.04864387) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_min12_n_disp3 - 1.0) / 1.0) * (1 + math.erf(((Q.ak02_min12_n_disp3 - 1.0) / 1.0) / math.sqrt(2)))) / 0.007154413   # +1.4%  lep_z < 0.06457 and ak02_min12_n_disp3 > 1
        + 0.01296914 * (22.5 * 0.5 * ((123.0 - Q.n_pairs_kt_above_1) / 22.5) * (1 + math.erf(((123.0 - Q.n_pairs_kt_above_1) / 22.5) / math.sqrt(2)))) / 13.74748   # +1.3%  n_pairs_kt_above_1 < 123
        - 0.01284218 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) / 0.4378545   # -1.3%  lep_z < 0.5187
        + 0.01275308 * (0.01140748 * 0.5 * ((0.0573796 - Q.pz_lnd3) / 0.01140748) * (1 + math.erf(((0.0573796 - Q.pz_lnd3) / 0.01140748) / math.sqrt(2)))) / 0.0146607   # +1.3%  pz_lnd3 < 0.05738
        - 0.01218464 * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2)))) / 1.048811   # -1.2%  sip_3d_3 < 4.606
        - 0.01185484 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.003871142   # -1.2%  M2_b2 < 0.02927 and kt2_2_n_lep < 1
        + 0.011311 * (23.0 * 0.5 * ((58.0 - Q.n_pairs_kt_above_1) / 23.0) * (1 + math.erf(((58.0 - Q.n_pairs_kt_above_1) / 23.0) / math.sqrt(2)))) / 2.480147   # +1.1%  n_pairs_kt_above_1 < 58
        - 0.009044226 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) / math.sqrt(2)))) / 3.38788   # -0.9%  mres_sd_mass_b0z005 > 159.9
        - 0.008771853 * (23.0 * 0.5 * ((58.0 - Q.n_pairs_kt_above_1) / 23.0) * (1 + math.erf(((58.0 - Q.n_pairs_kt_above_1) / 23.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.sjf_3_1_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.sjf_3_1_n_d3) / 1.0) / math.sqrt(2)))) / 5.74675   # -0.9%  n_pairs_kt_above_1 < 58 and sjf_3_1_n_d3 < 3
        - 0.00825721 * (6.235814 * 0.5 * ((Q.mres_sd_mass_b0z005 - 76.40585) / 6.235814) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 76.40585) / 6.235814) / math.sqrt(2)))) / 35.49666   # -0.8%  mres_sd_mass_b0z005 > 76.41
        + 0.008186448 * (0.01092519 * 0.5 * ((0.04286245 - Q.pz_lnd2) / 0.01092519) * (1 + math.erf(((0.04286245 - Q.pz_lnd2) / 0.01092519) / math.sqrt(2)))) / 0.005498889   # +0.8%  pz_lnd2 < 0.04286
        - 0.007818295 * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.6533455   # -0.8%  kt2_2_n_lep < 1
        - 0.007750346 * (0.02363876 * 0.5 * ((0.4744137 - Q.z_charged_had) / 0.02363876) * (1 + math.erf(((0.4744137 - Q.z_charged_had) / 0.02363876) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.dc_ntag) / 1.0) * (1 + math.erf(((1.0 - Q.dc_ntag) / 1.0) / math.sqrt(2)))) / 0.03673241   # -0.8%  z_charged_had < 0.4744 and dc_ntag < 1
        - 0.007661316 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (0.01968664 * 0.5 * ((0.06216672 - Q.pz_lnkt4) / 0.01968664) * (1 + math.erf(((0.06216672 - Q.pz_lnkt4) / 0.01968664) / math.sqrt(2)))) / 0.01868498   # -0.8%  sjq_2_2_nch < 4 and pz_lnkt4 < 0.06217
        + 0.007551764 * (0.01140748 * 0.5 * ((0.0573796 - Q.pz_lnd3) / 0.01140748) * (1 + math.erf(((0.0573796 - Q.pz_lnd3) / 0.01140748) / math.sqrt(2)))) * (3.006676 * 0.5 * ((6.43167 - Q.jd_3d_4) / 3.006676) * (1 + math.erf(((6.43167 - Q.jd_3d_4) / 3.006676) / math.sqrt(2)))) / 0.04682183   # +0.8%  pz_lnd3 < 0.05738 and jd_3d_4 < 6.432
        - 0.006930784 * (0.0964495 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) / math.sqrt(2)))) * (0.7733816 * 0.5 * ((0.7828545 - Q.lepsj_2_maxsd0) / 0.7733816) * (1 + math.erf(((0.7828545 - Q.lepsj_2_maxsd0) / 0.7733816) / math.sqrt(2)))) / 0.02186179   # -0.7%  sjq_2_sumabs_k1 > 0.5558 and lepsj_2_maxsd0 < 0.7829
        + 0.005420789 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 1.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 1.0) / 1.0) / math.sqrt(2)))) / 1.915687   # +0.5%  n_s3d_above_10 > 1
        + 0.005325537 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) / 0.9643417   # +0.5%  lep_ptrel > 43.21
        - 0.005266872 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2)))) * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) / 0.0007553545   # -0.5%  M2_b2 < 0.02927 and sjq_2_prod_k1 < 0.09803
        - 0.00514598 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 1.203998   # -0.5%  sjq_2_2_nch < 4 and n_s3d_above_3 < 5
        + 0.004996221 * (0.0964495 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) / math.sqrt(2)))) / 0.05500747   # +0.5%  sjq_2_sumabs_k1 > 0.5558
        - 0.004952605 * (1.278204 * 0.5 * ((3.101398 - Q.sjf_3_2_max3d) / 1.278204) * (1 + math.erf(((3.101398 - Q.sjf_3_2_max3d) / 1.278204) / math.sqrt(2)))) / 0.5582653   # -0.5%  sjf_3_2_max3d < 3.101
        - 0.004924305 * (0.02314658 * 0.5 * ((Q.z_neutral - 0.4865925) / 0.02314658) * (1 + math.erf(((Q.z_neutral - 0.4865925) / 0.02314658) / math.sqrt(2)))) / 0.03163256   # -0.5%  z_neutral > 0.4866
        + 0.004182521 * (2.0 * 0.5 * ((3.0 - Q.sjq_2_1_nch) / 2.0) * (1 + math.erf(((3.0 - Q.sjq_2_1_nch) / 2.0) / math.sqrt(2)))) / 0.08397528   # +0.4%  sjq_2_1_nch < 3
        - 0.004017307 * (0.0106503 * 0.5 * ((0.09036178 - Q.pz_lnkt0) / 0.0106503) * (1 + math.erf(((0.09036178 - Q.pz_lnkt0) / 0.0106503) / math.sqrt(2)))) * (0.05391589 * 0.5 * ((0.2792084 - Q.dr_2) / 0.05391589) * (1 + math.erf(((0.2792084 - Q.dr_2) / 0.05391589) / math.sqrt(2)))) / 0.0003048898   # -0.4%  pz_lnkt0 < 0.09036 and dr_2 < 0.2792
        + 0.003639663 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (0.03240401 * 0.5 * ((Q.ak02_2_z - 0.0) / 0.03240401) * (1 + math.erf(((Q.ak02_2_z - 0.0) / 0.03240401) / math.sqrt(2)))) / 0.0710246   # +0.4%  sjq_2_2_nch < 4 and ak02_2_z > 0
        - 0.0035379 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (0.2342104 * 0.5 * ((Q.lund3_lnz - -4.699653) / 0.2342104) * (1 + math.erf(((Q.lund3_lnz - -4.699653) / 0.2342104) / math.sqrt(2)))) / 0.5412713   # -0.4%  sjq_2_2_nch < 4 and lund3_lnz > -4.7
        + 0.003415709 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (0.03240401 * 0.5 * ((Q.ak02_2_z - 0.0) / 0.03240401) * (1 + math.erf(((Q.ak02_2_z - 0.0) / 0.03240401) / math.sqrt(2)))) / 0.2815275   # +0.3%  lep_ptrel > 43.21 and ak02_2_z > 0
        - 0.003330036 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2)))) / 0.006133635   # -0.3%  M2_b2 < 0.02927
        + 0.003288218 * (7.843552 * 0.5 * ((Q.mres_sd_mass_b0z005 - 131.0776) / 7.843552) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 131.0776) / 7.843552) / math.sqrt(2)))) / 7.363783   # +0.3%  mres_sd_mass_b0z005 > 131.1
        + 0.003174866 * (0.02363876 * 0.5 * ((0.4744137 - Q.z_charged_had) / 0.02363876) * (1 + math.erf(((0.4744137 - Q.z_charged_had) / 0.02363876) / math.sqrt(2)))) * (164.9991 * 0.5 * ((226.3008 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((226.3008 - Q.sip_3d_2) / 164.9991) / math.sqrt(2)))) / 8.462044   # +0.3%  z_charged_had < 0.4744 and sip_3d_2 < 226.3
        - 0.003139604 * (4.934653 * 0.5 * ((14.92637 - Q.mass_neutral) / 4.934653) * (1 + math.erf(((14.92637 - Q.mass_neutral) / 4.934653) / math.sqrt(2)))) / 0.5831325   # -0.3%  mass_neutral < 14.93
        - 0.003036183 * (0.02757255 * 0.5 * ((0.6800935 - Q.tau32) / 0.02757255) * (1 + math.erf(((0.6800935 - Q.tau32) / 0.02757255) / math.sqrt(2)))) / 0.08068051   # -0.3%  tau32 < 0.6801
        - 0.002974039 * (7.17138 * 0.5 * ((Q.mass_top10 - 95.24893) / 7.17138) * (1 + math.erf(((Q.mass_top10 - 95.24893) / 7.17138) / math.sqrt(2)))) * (9.373106 * 0.5 * ((Q.lnpt_40 - 0.0508341) / 9.373106) * (1 + math.erf(((Q.lnpt_40 - 0.0508341) / 9.373106) / math.sqrt(2)))) / 1.242204   # -0.3%  mass_top10 > 95.25 and lnpt_40 > 0.05083
        - 0.002953864 * (0.0964495 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) / math.sqrt(2)))) * (0.1568179 * 0.5 * ((1.261566 - Q.sjq_2_sumabs_k03) / 0.1568179) * (1 + math.erf(((1.261566 - Q.sjq_2_sumabs_k03) / 0.1568179) / math.sqrt(2)))) / 0.01919331   # -0.3%  sjq_2_sumabs_k1 > 0.5558 and sjq_2_sumabs_k03 < 1.262
        + 0.002884272 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (257.2011 * 0.5 * ((482.5817 - Q.jd_sum_abs_sd0_top5) / 257.2011) * (1 + math.erf(((482.5817 - Q.jd_sum_abs_sd0_top5) / 257.2011) / math.sqrt(2)))) / 141.0045   # +0.3%  sjq_2_2_nch < 4 and jd_sum_abs_sd0_top5 < 482.6
        - 0.002770474 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2)))) / 1.091313   # -0.3%  mass < 56.49
        + 0.002700463 * (0.0106503 * 0.5 * ((0.09036178 - Q.pz_lnkt0) / 0.0106503) * (1 + math.erf(((0.09036178 - Q.pz_lnkt0) / 0.0106503) / math.sqrt(2)))) / 0.003899851   # +0.3%  pz_lnkt0 < 0.09036
        - 0.002695827 * (0.1721579 * 0.5 * ((2.133633 - Q.pt_entropy) / 0.1721579) * (1 + math.erf(((2.133633 - Q.pt_entropy) / 0.1721579) / math.sqrt(2)))) / 0.06417642   # -0.3%  pt_entropy < 2.134
        + 0.002509042 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) / 0.8420682   # +0.3%  lep_iso < 1.362
        + 0.002495509 * (16.90428 * 0.5 * ((Q.mass - 164.4374) / 16.90428) * (1 + math.erf(((Q.mass - 164.4374) / 16.90428) / math.sqrt(2)))) / 3.240687   # +0.2%  mass > 164.4
        - 0.002136078 * (0.5664529 * 0.5 * ((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) * (1 + math.erf(((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) / math.sqrt(2)))) * (1.0 * 0.5 * ((2.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((2.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) / 0.04638776   # -0.2%  pair_mean_lnm2 > 4.069 and n_s3d_above_3 < 2
        - 0.002127963 * (0.1667799 * 0.5 * ((-0.3680108 - Q.sjq_3_2_k1) / 0.1667799) * (1 + math.erf(((-0.3680108 - Q.sjq_3_2_k1) / 0.1667799) / math.sqrt(2)))) / 0.03583271   # -0.2%  sjq_3_2_k1 < -0.368
        + 0.002055369 * (2.0 * 0.5 * ((5.0 - Q.n_lund) / 2.0) * (1 + math.erf(((5.0 - Q.n_lund) / 2.0) / math.sqrt(2)))) / 0.1380555   # +0.2%  n_lund < 5
        - 0.002005958 * (0.005331567 * 0.5 * ((Q.M2_b05 - 0.1209237) / 0.005331567) * (1 + math.erf(((Q.M2_b05 - 0.1209237) / 0.005331567) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2)))) / 0.001391255   # -0.2%  M2_b05 > 0.1209 and ak02_3_n_lep > 0
        + 0.001925873 * (7.17138 * 0.5 * ((Q.mass_top10 - 95.24893) / 7.17138) * (1 + math.erf(((Q.mass_top10 - 95.24893) / 7.17138) / math.sqrt(2)))) / 2.939738   # +0.2%  mass_top10 > 95.25
        - 0.001818786 * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) * (0.2881506 * 0.5 * ((Q.lund1_lnkt - 3.599502) / 0.2881506) * (1 + math.erf(((Q.lund1_lnkt - 3.599502) / 0.2881506) / math.sqrt(2)))) / 0.09608778   # -0.2%  kt2_2_n_lep < 1 and lund1_lnkt > 3.6
        + 0.001786727 * (1.0 * 0.5 * ((4.0 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((4.0 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2)))) * (0.01992156 * 0.5 * ((Q.lepsj_2_dr - 0.04178742) / 0.01992156) * (1 + math.erf(((Q.lepsj_2_dr - 0.04178742) / 0.01992156) / math.sqrt(2)))) / 0.04061747   # +0.2%  n_sdz_above_5 < 4 and lepsj_2_dr > 0.04179
        - 0.001771042 * (0.5664529 * 0.5 * ((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) * (1 + math.erf(((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) / math.sqrt(2)))) * (4.214136 * 0.5 * ((10.59762 - Q.min_pair_mass) / 4.214136) * (1 + math.erf(((10.59762 - Q.min_pair_mass) / 4.214136) / math.sqrt(2)))) / 0.7103049   # -0.2%  pair_mean_lnm2 > 4.069 and min_pair_mass < 10.6
        - 0.001444694 * (0.2398529 * 0.5 * ((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) * (1 + math.erf(((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) / math.sqrt(2)))) / 0.01973624   # -0.1%  sjq_3_2_k1 > 0.6149
        + 0.00143406 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (2.0 * 0.5 * ((4.0 - Q.lepsj_2_n_d3) / 2.0) * (1 + math.erf(((4.0 - Q.lepsj_2_n_d3) / 2.0) / math.sqrt(2)))) / 1.481239   # +0.1%  lep_z < 0.5187 and lepsj_2_n_d3 < 4
        + 0.001406865 * (0.03953735 * 0.5 * ((Q.mres_sd_rg_b1z01 - 0.5299323) / 0.03953735) * (1 + math.erf(((Q.mres_sd_rg_b1z01 - 0.5299323) / 0.03953735) / math.sqrt(2)))) / 0.01384814   # +0.1%  mres_sd_rg_b1z01 > 0.5299
        + 0.001317022 * (0.100181 * 0.5 * ((0.3803178 - Q.z_displaced5) / 0.100181) * (1 + math.erf(((0.3803178 - Q.z_displaced5) / 0.100181) / math.sqrt(2)))) * (18.73129 * 0.5 * ((-18.42068 - Q.lne_40) / 18.73129) * (1 + math.erf(((-18.42068 - Q.lne_40) / 18.73129) / math.sqrt(2)))) / 0.3674922   # +0.1%  z_displaced5 < 0.3803 and lne_40 < -18.42
        + 0.001283076 * (24.25363 * 0.5 * ((Q.mres_sd_mass_b0z005 - 53.57509) / 24.25363) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 53.57509) / 24.25363) / math.sqrt(2)))) * (13.98191 * 0.5 * ((12.49716 - Q.sjf_2_1_mass_d3) / 13.98191) * (1 + math.erf(((12.49716 - Q.sjf_2_1_mass_d3) / 13.98191) / math.sqrt(2)))) / 427.102   # +0.1%  mres_sd_mass_b0z005 > 53.58 and sjf_2_1_mass_d3 < 12.5
        + 0.001272197 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.0) / 1.0) / math.sqrt(2)))) / 0.8926235   # +0.1%  n_s3d_above_3 > 5
        - 0.001255144 * (0.5664529 * 0.5 * ((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) * (1 + math.erf(((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) / math.sqrt(2)))) / 0.1032402   # -0.1%  pair_mean_lnm2 > 4.069
        + 0.001163386 * (0.02363876 * 0.5 * ((0.4744137 - Q.z_charged_had) / 0.02363876) * (1 + math.erf(((0.4744137 - Q.z_charged_had) / 0.02363876) / math.sqrt(2)))) * (0.000830722 * 0.5 * ((0.00678572 - Q.M3_b2) / 0.000830722) * (1 + math.erf(((0.00678572 - Q.M3_b2) / 0.000830722) / math.sqrt(2)))) / 1.874727e-05   # +0.1%  z_charged_had < 0.4744 and M3_b2 < 0.006786
        + 0.001044997 * (0.1721579 * 0.5 * ((2.133633 - Q.pt_entropy) / 0.1721579) * (1 + math.erf(((2.133633 - Q.pt_entropy) / 0.1721579) / math.sqrt(2)))) * (0.002556514 * 0.5 * ((0.001692006 - Q.sjf_4_3_z_d3) / 0.002556514) * (1 + math.erf(((0.001692006 - Q.sjf_4_3_z_d3) / 0.002556514) / math.sqrt(2)))) / 5.751589e-05   # +0.1%  pt_entropy < 2.134 and sjf_4_3_z_d3 < 0.001692
        + 0.0008709203 * (0.0964495 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.0) / 1.0) / math.sqrt(2)))) / 0.03804243   # +0.1%  sjq_2_sumabs_k1 > 0.5558 and n_lund_kt_above_5 > 1
        + 0.0008560855 * (0.003235318 * 0.5 * ((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) * (1 + math.erf(((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2)))) / 0.0003736527   # +0.1%  lepsj_2_dr < 0.000821 and sip_3d_3 < 4.606
        - 0.0008507987 * (5.02206 * 0.5 * ((Q.mass_top50 - 129.5874) / 5.02206) * (1 + math.erf(((Q.mass_top50 - 129.5874) / 5.02206) / math.sqrt(2)))) / 7.907933   # -0.1%  mass_top50 > 129.6
        + 0.0008439781 * (2.0 * 0.5 * ((Q.sjf_2_1_n_d3 - 7.0) / 2.0) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 7.0) / 2.0) / math.sqrt(2)))) / 0.110583   # +0.1%  sjf_2_1_n_d3 > 7
        - 0.0008012715 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2)))) / 0.6767019   # -0.1%  lep_ptrel > 43.21 and sip_3d_3 < 4.606
        + 0.0007848709 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2)))) * (0.02245414 * 0.5 * ((Q.dr_1 - 0.1975394) / 0.02245414) * (1 + math.erf(((Q.dr_1 - 0.1975394) / 0.02245414) / math.sqrt(2)))) / 0.0001122774   # +0.1%  M2_b2 < 0.02927 and dr_1 > 0.1975
        + 0.0007825111 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) / 0.4568459   # +0.1%  sjq_2_2_nch < 4
        + 0.0005872228 * (0.02363876 * 0.5 * ((0.4744137 - Q.z_charged_had) / 0.02363876) * (1 + math.erf(((0.4744137 - Q.z_charged_had) / 0.02363876) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.iselectron_1 - 0.0) / 1e-06) * (1 + math.erf(((Q.iselectron_1 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.002999809   # +0.1%  z_charged_had < 0.4744 and iselectron_1 > 0
        - 0.0005497402 * (0.1664357 * 0.5 * ((-3.425158 - Q.lnerel_4) / 0.1664357) * (1 + math.erf(((-3.425158 - Q.lnerel_4) / 0.1664357) / math.sqrt(2)))) / 0.03801393   # -0.1%  lnerel_4 < -3.425
        - 0.0005253654 * (0.100181 * 0.5 * ((0.3803178 - Q.z_displaced5) / 0.100181) * (1 + math.erf(((0.3803178 - Q.z_displaced5) / 0.100181) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.kt2_1_n_lep - 0.0) / 1.0) * (1 + math.erf(((Q.kt2_1_n_lep - 0.0) / 1.0) / math.sqrt(2)))) / 0.07990589   # -0.1%  z_displaced5 < 0.3803 and kt2_1_n_lep > 0
        - 0.0004719085 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2)))) * (0.06161476 * 0.5 * ((0.676889 - Q.sj3_dr13) / 0.06161476) * (1 + math.erf(((0.676889 - Q.sj3_dr13) / 0.06161476) / math.sqrt(2)))) / 0.5738303   # -0.0%  lep_ptrel > 27.32 and sj3_dr13 < 0.6769
        + 0.0003845788 * (0.02757255 * 0.5 * ((0.6800935 - Q.tau32) / 0.02757255) * (1 + math.erf(((0.6800935 - Q.tau32) / 0.02757255) / math.sqrt(2)))) * (0.1102744 * 0.5 * ((Q.dr_11 - 0.5195105) / 0.1102744) * (1 + math.erf(((Q.dr_11 - 0.5195105) / 0.1102744) / math.sqrt(2)))) / 0.0008322867   # +0.0%  tau32 < 0.6801 and dr_11 > 0.5195
        - 0.0003842779 * (6.785484 * 0.5 * ((Q.mass_top5 - 68.52153) / 6.785484) * (1 + math.erf(((Q.mass_top5 - 68.52153) / 6.785484) / math.sqrt(2)))) / 2.773241   # -0.0%  mass_top5 > 68.52
        + 0.0003841883 * (0.0964495 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) / math.sqrt(2)))) * (0.0131537 * 0.5 * ((0.1142534 - Q.dr_11) / 0.0131537) * (1 + math.erf(((0.1142534 - Q.dr_11) / 0.0131537) / math.sqrt(2)))) / 0.001008097   # +0.0%  sjq_2_sumabs_k1 > 0.5558 and dr_11 < 0.1143
        + 0.0003507158 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sdb_0_n - 2.0) / 1.0) * (1 + math.erf(((Q.sdb_0_n - 2.0) / 1.0) / math.sqrt(2)))) / 0.003358768   # +0.0%  M2_b2 < 0.02927 and sdb_0_n > 2
        - 0.0002588997 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (0.05374146 * 0.5 * ((-0.1588135 - Q.phi_4) / 0.05374146) * (1 + math.erf(((-0.1588135 - Q.phi_4) / 0.05374146) / math.sqrt(2)))) / 0.0193899   # -0.0%  lep_ptrel > 43.21 and phi_4 < -0.1588
        + 0.0001538542 * (0.003235318 * 0.5 * ((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) * (1 + math.erf(((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) / math.sqrt(2)))) / 0.0003162318   # +0.0%  lepsj_2_dr < 0.000821
        + 0.0001399537 * (0.03953735 * 0.5 * ((Q.mres_sd_rg_b1z01 - 0.5299323) / 0.03953735) * (1 + math.erf(((Q.mres_sd_rg_b1z01 - 0.5299323) / 0.03953735) / math.sqrt(2)))) * (0.07882828 * 0.5 * ((0.7416858 - Q.sjq_2_sumabs_k03) / 0.07882828) * (1 + math.erf(((0.7416858 - Q.sjq_2_sumabs_k03) / 0.07882828) / math.sqrt(2)))) / 0.003025222   # +0.0%  mres_sd_rg_b1z01 > 0.5299 and sjq_2_sumabs_k03 < 0.7417
    )
    return z


def neuron_124(Q):
    # scale S = 9.345; each line: share * term / its average size
    z = 9.345175 * (-0.05520418
        - 0.06987463 * (4.488781 * 0.5 * ((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) * (1 + math.erf(((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) / math.sqrt(2)))) / 28.3744   # -7.0%  mres_sd_mass_b0z005 < 125.9
        + 0.0639464 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) / 0.1673441   # +6.4%  lep_z < 0.2214
        + 0.05771459 * (5.31786 * 0.5 * ((91.82593 - Q.mres_sd_mass_b0z005) / 5.31786) * (1 + math.erf(((91.82593 - Q.mres_sd_mass_b0z005) / 5.31786) / math.sqrt(2)))) / 10.34254   # +5.8%  mres_sd_mass_b0z005 < 91.83
        + 0.0535211 * (18.42178 * 0.5 * ((182.8592 - Q.mass) / 18.42178) * (1 + math.erf(((182.8592 - Q.mass) / 18.42178) / math.sqrt(2)))) / 69.40895   # +5.4%  mass < 182.9
        + 0.04691498 * (18.66842 * 0.5 * ((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) / 56.92363   # +4.7%  mres_sd_mass_b0z005 < 159.9
        - 0.04435736 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) / 0.8420682   # -4.4%  lep_iso < 1.362
        + 0.04289436 * (0.01610679 * 0.5 * ((0.02900919 - Q.lepsj_3_dr) / 0.01610679) * (1 + math.erf(((0.02900919 - Q.lepsj_3_dr) / 0.01610679) / math.sqrt(2)))) / 0.02129097   # +4.3%  lepsj_3_dr < 0.02901
        - 0.03923231 * (0.7649624 * 0.5 * ((0.8036986 - Q.lepsj_3_maxsd0) / 0.7649624) * (1 + math.erf(((0.8036986 - Q.lepsj_3_maxsd0) / 0.7649624) / math.sqrt(2)))) / 0.4748477   # -3.9%  lepsj_3_maxsd0 < 0.8037
        + 0.03291063 * (14.42334 * 0.5 * ((141.5589 - Q.mres_sd_mass_b0z005) / 14.42334) * (1 + math.erf(((141.5589 - Q.mres_sd_mass_b0z005) / 14.42334) / math.sqrt(2)))) / 40.76608   # +3.3%  mres_sd_mass_b0z005 < 141.6
        + 0.03264253 * (1.0 * 0.5 * ((1.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((1.0 - Q.n_lepton) / 1.0) / math.sqrt(2)))) / 0.4923146   # +3.3%  n_lepton < 1
        - 0.02801641 * (137.5 * 0.5 * ((702.0 - Q.n_pairs_kt_above_1) / 137.5) * (1 + math.erf(((702.0 - Q.n_pairs_kt_above_1) / 137.5) / math.sqrt(2)))) / 395.2812   # -2.8%  n_pairs_kt_above_1 < 702
        - 0.02760426 * (5.227956 * 0.5 * ((102.8576 - Q.mres_sd_mass_b0z005) / 5.227956) * (1 + math.erf(((102.8576 - Q.mres_sd_mass_b0z005) / 5.227956) / math.sqrt(2)))) / 14.8042   # -2.8%  mres_sd_mass_b0z005 < 102.9
        + 0.02400932 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) / 0.2702483   # +2.4%  lep_z < 0.3397
        + 0.02376129 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) / 0.07858689   # +2.4%  lepsj_2_dr < 0.103
        - 0.01942724 * (6.235814 * 0.5 * ((76.40585 - Q.mres_sd_mass_b0z005) / 6.235814) * (1 + math.erf(((76.40585 - Q.mres_sd_mass_b0z005) / 6.235814) / math.sqrt(2)))) / 6.136088   # -1.9%  mres_sd_mass_b0z005 < 76.41
        - 0.01859737 * (43.0 * 0.5 * ((366.0 - Q.n_pairs_kt_above_1) / 43.0) * (1 + math.erf(((366.0 - Q.n_pairs_kt_above_1) / 43.0) / math.sqrt(2)))) / 128.9288   # -1.9%  n_pairs_kt_above_1 < 366
        - 0.01744268 * (1.5 * 0.5 * ((7.0 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.0 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2)))) / 3.650329   # -1.7%  n_s3d_above_3 < 7
        - 0.01718711 * (2.5 * 0.5 * ((31.0 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((31.0 - Q.n_for_90pct) / 2.5) / math.sqrt(2)))) / 11.71735   # -1.7%  n_for_90pct < 31
        - 0.01621239 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) / 0.2261637   # -1.6%  lep_iso < 0.4382
        - 0.0161678 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.0947145 * 0.5 * ((0.0947145 - Q.dc_4_z) / 0.0947145) * (1 + math.erf(((0.0947145 - Q.dc_4_z) / 0.0947145) / math.sqrt(2)))) / 0.01231544   # -1.6%  lep_z < 0.2214 and dc_4_z < 0.09471
        - 0.01582676 * (0.01710086 * 0.5 * ((0.06416437 - Q.z_displaced3) / 0.01710086) * (1 + math.erf(((0.06416437 - Q.z_displaced3) / 0.01710086) / math.sqrt(2)))) / 0.02644806   # -1.6%  z_displaced3 < 0.06416
        + 0.01554773 * (0.0001198329 * 0.5 * ((0.0002536827 - Q.e3_b2) / 0.0001198329) * (1 + math.erf(((0.0002536827 - Q.e3_b2) / 0.0001198329) / math.sqrt(2)))) / 0.0001673774   # +1.6%  e3_b2 < 0.0002537
        + 0.01393217 * (0.01992156 * 0.5 * ((0.04178742 - Q.lepsj_2_dr) / 0.01992156) * (1 + math.erf(((0.04178742 - Q.lepsj_2_dr) / 0.01992156) / math.sqrt(2)))) / 0.02937909   # +1.4%  lepsj_2_dr < 0.04179
        - 0.01360008 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) / 1.677331   # -1.4%  n_s3d_above_3 > 3
        + 0.0135932 * (0.0947145 * 0.5 * ((0.0947145 - Q.dc_4_z) / 0.0947145) * (1 + math.erf(((0.0947145 - Q.dc_4_z) / 0.0947145) / math.sqrt(2)))) / 0.07423689   # +1.4%  dc_4_z < 0.09471
        - 0.01243172 * (1.0 * 0.5 * ((1.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((1.0 - Q.n_lepton) / 1.0) / math.sqrt(2)))) * (0.01040872 * 0.5 * ((0.01435877 - Q.sdb_4_z) / 0.01040872) * (1 + math.erf(((0.01435877 - Q.sdb_4_z) / 0.01040872) / math.sqrt(2)))) / 0.004924561   # -1.2%  n_lepton < 1 and sdb_4_z < 0.01436
        + 0.01053727 * (0.01670814 * 0.5 * ((Q.z_displaced5 - 0.06245248) / 0.01670814) * (1 + math.erf(((Q.z_displaced5 - 0.06245248) / 0.01670814) / math.sqrt(2)))) / 0.05938468   # +1.1%  z_displaced5 > 0.06245
        + 0.01050691 * (2.5 * 0.5 * ((31.0 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((31.0 - Q.n_for_90pct) / 2.5) / math.sqrt(2)))) * (0.1673111 * 0.5 * ((Q.sjq_2_prod_k05 - -0.5057096) / 0.1673111) * (1 + math.erf(((Q.sjq_2_prod_k05 - -0.5057096) / 0.1673111) / math.sqrt(2)))) / 5.296283   # +1.1%  n_for_90pct < 31 and sjq_2_prod_k05 > -0.5057
        - 0.01013433 * (0.9305981 * 0.5 * ((3.029824 - Q.sjf_2_2_max3d) / 0.9305981) * (1 + math.erf(((3.029824 - Q.sjf_2_2_max3d) / 0.9305981) / math.sqrt(2)))) / 0.4838409   # -1.0%  sjf_2_2_max3d < 3.03
        - 0.009955847 * (0.04339825 * 0.5 * ((0.5454864 - Q.sj2_dr) / 0.04339825) * (1 + math.erf(((0.5454864 - Q.sj2_dr) / 0.04339825) / math.sqrt(2)))) / 0.173069   # -1.0%  sj2_dr < 0.5455
        - 0.008584846 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) / 15.65331   # -0.9%  mass < 117.5
        + 0.008520895 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) * (6.000774 * 0.5 * ((14.85973 - Q.jd_3d_4) / 6.000774) * (1 + math.erf(((14.85973 - Q.jd_3d_4) / 6.000774) / math.sqrt(2)))) / 165.7552   # +0.9%  mass < 117.5 and jd_3d_4 < 14.86
        - 0.00739264 * (0.02246636 * 0.5 * ((Q.z_photon - 0.1149688) / 0.02246636) * (1 + math.erf(((Q.z_photon - 0.1149688) / 0.02246636) / math.sqrt(2)))) / 0.1437282   # -0.7%  z_photon > 0.115
        - 0.007231474 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.02653106 * 0.5 * ((Q.z_displaced5 - 0.1046203) / 0.02653106) * (1 + math.erf(((Q.z_displaced5 - 0.1046203) / 0.02653106) / math.sqrt(2)))) / 0.01115566   # -0.7%  lep_z < 0.3397 and z_displaced5 > 0.1046
        + 0.007052251 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) / 36.20948   # +0.7%  lep_ptrel < 43.21
        - 0.006678425 * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) / 0.1153058   # -0.7%  sjq_2_prod_k1 < 0.09803
        - 0.006594152 * (5.286881 * 0.5 * ((100.4835 - Q.mass) / 5.286881) * (1 + math.erf(((100.4835 - Q.mass) / 5.286881) / math.sqrt(2)))) / 8.121469   # -0.7%  mass < 100.5
        - 0.006372439 * (3.244489 * 0.5 * ((66.70506 - Q.sj4_pair_mass_max) / 3.244489) * (1 + math.erf(((66.70506 - Q.sj4_pair_mass_max) / 3.244489) / math.sqrt(2)))) / 4.710216   # -0.6%  sj4_pair_mass_max < 66.71
        + 0.006241044 * (0.01355407 * 0.5 * ((0.03067242 - Q.kt2_1_z_disp3) / 0.01355407) * (1 + math.erf(((0.03067242 - Q.kt2_1_z_disp3) / 0.01355407) / math.sqrt(2)))) / 0.01840745   # +0.6%  kt2_1_z_disp3 < 0.03067
        + 0.006035363 * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2)))) / 5.641924   # +0.6%  max_abs_d0 < 10.52
        + 0.0058793 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) / 3.720581   # +0.6%  n_lund_kt_above_1 < 9
        - 0.005869963 * (0.02550179 * 0.5 * ((0.1792389 - Q.sdb_2_z) / 0.02550179) * (1 + math.erf(((0.1792389 - Q.sdb_2_z) / 0.02550179) / math.sqrt(2)))) / 0.01876489   # -0.6%  sdb_2_z < 0.1792
        + 0.004869072 * (18.42178 * 0.5 * ((182.8592 - Q.mass) / 18.42178) * (1 + math.erf(((182.8592 - Q.mass) / 18.42178) / math.sqrt(2)))) * (1110.683 * 0.5 * ((1490.247 - Q.sv_2_sd0_sum) / 1110.683) * (1 + math.erf(((1490.247 - Q.sv_2_sd0_sum) / 1110.683) / math.sqrt(2)))) / 87317.72   # +0.5%  mass < 182.9 and sv_2_sd0_sum < 1490
        - 0.004519792 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (401.9953 * 0.5 * ((577.991 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((577.991 - Q.sip_3d_3) / 401.9953) / math.sqrt(2)))) / 659.7798   # -0.5%  n_s3d_above_3 > 3 and sip_3d_3 < 578
        + 0.004404809 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2)))) / 9.427083   # +0.4%  n_dr_0p4_up < 15
        - 0.004381465 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (14.3347 * 0.5 * ((142.9952 - Q.sj3_pair_mass_max) / 14.3347) * (1 + math.erf(((142.9952 - Q.sj3_pair_mass_max) / 14.3347) / math.sqrt(2)))) / 1903.528   # -0.4%  lep_ptrel < 43.21 and sj3_pair_mass_max < 143
        - 0.004343793 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2)))) / 5.844349   # -0.4%  n_s3d_above_3 > 3 and max_abs_d0 < 10.52
        + 0.003971654 * (34.56457 * 0.5 * ((84.11398 - Q.jd_sum_abs_sd0_top5) / 34.56457) * (1 + math.erf(((84.11398 - Q.jd_sum_abs_sd0_top5) / 34.56457) / math.sqrt(2)))) / 27.38042   # +0.4%  jd_sum_abs_sd0_top5 < 84.11
        - 0.003941887 * (3.93783 * 0.5 * ((73.24742 - Q.sj3_pair_mass_max) / 3.93783) * (1 + math.erf(((73.24742 - Q.sj3_pair_mass_max) / 3.93783) / math.sqrt(2)))) / 4.292878   # -0.4%  sj3_pair_mass_max < 73.25
        + 0.00391684 * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 1.0) / 1.0) / math.sqrt(2)))) / 0.5054993   # +0.4%  lepsj_2_n_d3 > 1
        - 0.003652211 * (0.008959514 * 0.5 * ((0.2915083 - Q.C2_b05) / 0.008959514) * (1 + math.erf(((0.2915083 - Q.C2_b05) / 0.008959514) / math.sqrt(2)))) / 0.03982705   # -0.4%  C2_b05 < 0.2915
        - 0.003456192 * (0.03580654 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.2769039) / 0.03580654) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.2769039) / 0.03580654) / math.sqrt(2)))) / 0.1392552   # -0.3%  sjq_2_sumabs_k1 > 0.2769
        - 0.003432967 * (13.98191 * 0.5 * ((12.49716 - Q.sjf_2_1_mass_d3) / 13.98191) * (1 + math.erf(((12.49716 - Q.sjf_2_1_mass_d3) / 13.98191) / math.sqrt(2)))) / 8.225304   # -0.3%  sjf_2_1_mass_d3 < 12.5
        + 0.003402826 * (0.01806359 * 0.5 * ((0.04865675 - Q.sjf_2_2_z_d3) / 0.01806359) * (1 + math.erf(((0.04865675 - Q.sjf_2_2_z_d3) / 0.01806359) / math.sqrt(2)))) / 0.03307428   # +0.3%  sjf_2_2_z_d3 < 0.04866
        - 0.003311022 * (1.696027 * 0.5 * ((3.32421 - Q.sjf_2_1_max3d) / 1.696027) * (1 + math.erf(((3.32421 - Q.sjf_2_1_max3d) / 1.696027) / math.sqrt(2)))) / 0.312688   # -0.3%  sjf_2_1_max3d < 3.324
        - 0.003233386 * (1.0 * 0.5 * ((6.0 - Q.n_dr_0p1_0p2) / 1.0) * (1 + math.erf(((6.0 - Q.n_dr_0p1_0p2) / 1.0) / math.sqrt(2)))) / 0.8564846   # -0.3%  n_dr_0p1_0p2 < 6
        + 0.003134256 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.05556614 * 0.5 * ((Q.jet_charge_k05 - 0.01860317) / 0.05556614) * (1 + math.erf(((Q.jet_charge_k05 - 0.01860317) / 0.05556614) / math.sqrt(2)))) / 6.014681   # +0.3%  lep_ptrel < 43.21 and jet_charge_k05 > 0.0186
        - 0.002957791 * (0.02505229 * 0.5 * ((Q.sdb_2_z - 0.3448514) / 0.02505229) * (1 + math.erf(((Q.sdb_2_z - 0.3448514) / 0.02505229) / math.sqrt(2)))) / 0.06277194   # -0.3%  sdb_2_z > 0.3449
        + 0.002842004 * (5.197 * 0.5 * ((95.14961 - Q.mass) / 5.197) * (1 + math.erf(((95.14961 - Q.mass) / 5.197) / math.sqrt(2)))) / 6.369586   # +0.3%  mass < 95.15
        + 0.002319047 * (0.000494246 * 0.5 * ((Q.ecf_g31 - 0.006877516) / 0.000494246) * (1 + math.erf(((Q.ecf_g31 - 0.006877516) / 0.000494246) / math.sqrt(2)))) / 0.001331765   # +0.2%  ecf_g31 > 0.006878
        - 0.002167509 * (1.459864 * 0.5 * ((2.219501 - Q.mres_sd_prong_mass2) / 1.459864) * (1 + math.erf(((2.219501 - Q.mres_sd_prong_mass2) / 1.459864) / math.sqrt(2)))) / 0.2453042   # -0.2%  mres_sd_prong_mass2 < 2.22
        + 0.002115218 * (2.5 * 0.5 * ((31.0 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((31.0 - Q.n_for_90pct) / 2.5) / math.sqrt(2)))) * (0.03966789 * 0.5 * ((0.1333792 - Q.ak02_3_z) / 0.03966789) * (1 + math.erf(((0.1333792 - Q.ak02_3_z) / 0.03966789) / math.sqrt(2)))) / 1.32536   # +0.2%  n_for_90pct < 31 and ak02_3_z < 0.1334
        + 0.002063099 * (0.009806371 * 0.5 * ((Q.sjq_2_prod_k1 - 0.02149519) / 0.009806371) * (1 + math.erf(((Q.sjq_2_prod_k1 - 0.02149519) / 0.009806371) / math.sqrt(2)))) / 0.01202141   # +0.2%  sjq_2_prod_k1 > 0.0215
        - 0.001989799 * (0.02425343 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.1343792) / 0.02425343) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.1343792) / 0.02425343) / math.sqrt(2)))) / 0.219983   # -0.2%  sjq_2_sumabs_k1 > 0.1344
        - 0.001577665 * (0.05547861 * 0.5 * ((Q.tdz_2 - -0.05727976) / 0.05547861) * (1 + math.erf(((Q.tdz_2 - -0.05727976) / 0.05547861) / math.sqrt(2)))) / 0.07191244   # -0.2%  tdz_2 > -0.05728
        + 0.001506215 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_ntag - 0.0) / 1.0) * (1 + math.erf(((Q.dc_ntag - 0.0) / 1.0) / math.sqrt(2)))) / 2.345103   # +0.2%  n_dr_0p4_up < 15 and dc_ntag > 0
        + 0.00146121 * (2.0 * 0.5 * ((Q.sjf_2_1_n_d3 - 7.0) / 2.0) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 7.0) / 2.0) / math.sqrt(2)))) / 0.110583   # +0.1%  sjf_2_1_n_d3 > 7
        - 0.00145347 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (0.007964189 * 0.5 * ((0.06762785 - Q.M2_b2) / 0.007964189) * (1 + math.erf(((0.06762785 - Q.M2_b2) / 0.007964189) / math.sqrt(2)))) / 0.05903832   # -0.1%  n_s3d_above_3 > 3 and M2_b2 < 0.06763
        - 0.001452476 * (0.5788858 * 0.5 * ((1.777286 - Q.mass_displaced3) / 0.5788858) * (1 + math.erf(((1.777286 - Q.mass_displaced3) / 0.5788858) / math.sqrt(2)))) / 0.7349753   # -0.1%  mass_displaced3 < 1.777
        + 0.001436387 * (0.1605002 * 0.5 * ((Q.pair_max_lnm2 - 7.347625) / 0.1605002) * (1 + math.erf(((Q.pair_max_lnm2 - 7.347625) / 0.1605002) / math.sqrt(2)))) / 0.1028733   # +0.1%  pair_max_lnm2 > 7.348
        - 0.001298328 * (0.007727658 * 0.5 * ((0.006220408 - Q.psi_0p1) / 0.007727658) * (1 + math.erf(((0.006220408 - Q.psi_0p1) / 0.007727658) / math.sqrt(2)))) / 0.0004461023   # -0.1%  psi_0p1 < 0.00622
        - 0.001255052 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.01119633 * 0.5 * ((0.8709334 - Q.tau43) / 0.01119633) * (1 + math.erf(((0.8709334 - Q.tau43) / 0.01119633) / math.sqrt(2)))) / 0.01305516   # -0.1%  lep_z < 0.2214 and tau43 < 0.8709
        - 0.001167011 * (0.2857178 * 0.5 * ((Q.dc_3_charge - 0.6283033) / 0.2857178) * (1 + math.erf(((Q.dc_3_charge - 0.6283033) / 0.2857178) / math.sqrt(2)))) / 0.02187731   # -0.1%  dc_3_charge > 0.6283
        - 0.001160903 * (0.08408739 * 0.5 * ((0.0 - Q.sv_2_dr) / 0.08408739) * (1 + math.erf(((0.0 - Q.sv_2_dr) / 0.08408739) / math.sqrt(2)))) / 0.000853123   # -0.1%  sv_2_dr < 0
        + 0.001025593 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) / 1.223619   # +0.1%  lepsj_3_maxsd0 < 2.413
        - 0.0009853151 * (5.197 * 0.5 * ((95.14961 - Q.mass) / 5.197) * (1 + math.erf(((95.14961 - Q.mass) / 5.197) / math.sqrt(2)))) * (0.5801757 * 0.5 * ((1.544662 - Q.N3_b2) / 0.5801757) * (1 + math.erf(((1.544662 - Q.N3_b2) / 0.5801757) / math.sqrt(2)))) / 4.644813   # -0.1%  mass < 95.15 and N3_b2 < 1.545
        + 0.0009436144 * (9.773951 * 0.5 * ((56.73313 - Q.sj3_pair_mass_max) / 9.773951) * (1 + math.erf(((56.73313 - Q.sj3_pair_mass_max) / 9.773951) / math.sqrt(2)))) / 1.704515   # +0.1%  sj3_pair_mass_max < 56.73
        + 0.0008678389 * (1.0 * 0.5 * ((Q.sjf_4_1_n_d3 - 2.0) / 1.0) * (1 + math.erf(((Q.sjf_4_1_n_d3 - 2.0) / 1.0) / math.sqrt(2)))) / 0.5418366   # +0.1%  sjf_4_1_n_d3 > 2
        - 0.0008577098 * (0.07655927 * 0.5 * ((Q.sj3_dr12 - 0.5966255) / 0.07655927) * (1 + math.erf(((Q.sj3_dr12 - 0.5966255) / 0.07655927) / math.sqrt(2)))) / 0.005195002   # -0.1%  sj3_dr12 > 0.5966
        - 0.0007201927 * (0.1223989 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.6652978) / 0.1223989) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.6652978) / 0.1223989) / math.sqrt(2)))) / 0.03608245   # -0.1%  sjq_2_sumabs_k1 > 0.6653
        + 0.0006908911 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) / 26.78553   # +0.1%  sv_1_sd0_sum < 51.56
        + 0.0006846084 * (0.03580654 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.2769039) / 0.03580654) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.2769039) / 0.03580654) / math.sqrt(2)))) * (0.7649624 * 0.5 * ((0.8036986 - Q.lepsj_3_maxsd0) / 0.7649624) * (1 + math.erf(((0.8036986 - Q.lepsj_3_maxsd0) / 0.7649624) / math.sqrt(2)))) / 0.06463002   # +0.1%  sjq_2_sumabs_k1 > 0.2769 and lepsj_3_maxsd0 < 0.8037
        - 0.0006530167 * (2.5 * 0.5 * ((31.0 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((31.0 - Q.n_for_90pct) / 2.5) / math.sqrt(2)))) * (0.1509271 * 0.5 * ((Q.lund1_lndelta - -1.046452) / 0.1509271) * (1 + math.erf(((Q.lund1_lndelta - -1.046452) / 0.1509271) / math.sqrt(2)))) / 6.266679   # -0.1%  n_for_90pct < 31 and lund1_lndelta > -1.046
        - 0.0006470894 * (11.49171 * 0.5 * ((71.96396 - Q.mass) / 11.49171) * (1 + math.erf(((71.96396 - Q.mass) / 11.49171) / math.sqrt(2)))) / 2.139873   # -0.1%  mass < 71.96
        + 0.0005636861 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (0.02145059 * 0.5 * ((0.7981752 - Q.tau32) / 0.02145059) * (1 + math.erf(((0.7981752 - Q.tau32) / 0.02145059) / math.sqrt(2)))) / 0.2462208   # +0.1%  n_s3d_above_3 > 3 and tau32 < 0.7982
        + 0.0004898033 * (0.01804395 * 0.5 * ((0.1244388 - Q.mres_sd_zg_b1z01) / 0.01804395) * (1 + math.erf(((0.1244388 - Q.mres_sd_zg_b1z01) / 0.01804395) / math.sqrt(2)))) / 0.01223636   # +0.0%  mres_sd_zg_b1z01 < 0.1244
        + 0.000402978 * (1.0 * 0.5 * ((0.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) / 0.03307822   # +0.0%  kt2_2_n_lep < 0
        - 0.0003472235 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.003882417 * 0.5 * ((-0.006552442 - Q.sjq_2_prod_k1) / 0.003882417) * (1 + math.erf(((-0.006552442 - Q.sjq_2_prod_k1) / 0.003882417) / math.sqrt(2)))) / 0.6848801   # -0.0%  lep_ptrel < 43.21 and sjq_2_prod_k1 < -0.006552
        - 0.0002654533 * (0.04339825 * 0.5 * ((0.5454864 - Q.sj2_dr) / 0.04339825) * (1 + math.erf(((0.5454864 - Q.sj2_dr) / 0.04339825) / math.sqrt(2)))) * (0.0820479 * 0.5 * ((4.855929 - Q.lne_1) / 0.0820479) * (1 + math.erf(((4.855929 - Q.lne_1) / 0.0820479) / math.sqrt(2)))) / 0.05662539   # -0.0%  sj2_dr < 0.5455 and lne_1 < 4.856
        + 0.0002295152 * (0.04808774 * 0.5 * ((Q.lepsj_3_dr - 0.09790963) / 0.04808774) * (1 + math.erf(((Q.lepsj_3_dr - 0.09790963) / 0.04808774) / math.sqrt(2)))) * (540.2081 * 0.5 * ((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2)))) / 3.586681   # +0.0%  lepsj_3_dr > 0.09791 and lepsj_3_maxsd0 < 587
        - 0.0002177732 * (0.02271025 * 0.5 * ((0.1465679 - Q.tau21_b2) / 0.02271025) * (1 + math.erf(((0.1465679 - Q.tau21_b2) / 0.02271025) / math.sqrt(2)))) / 0.01448528   # -0.0%  tau21_b2 < 0.1466
        + 0.0001139428 * (0.01806359 * 0.5 * ((0.04865675 - Q.sjf_2_2_z_d3) / 0.01806359) * (1 + math.erf(((0.04865675 - Q.sjf_2_2_z_d3) / 0.01806359) / math.sqrt(2)))) * (0.007274976 * 0.5 * ((0.0 - Q.tdz_0) / 0.007274976) * (1 + math.erf(((0.0 - Q.tdz_0) / 0.007274976) / math.sqrt(2)))) / 0.0004934027   # +0.0%  sjf_2_2_z_d3 < 0.04866 and tdz_0 < 0
        + 6.624189e-05 * (12.30925 * 0.5 * ((Q.mass_displaced3 - 39.09615) / 12.30925) * (1 + math.erf(((Q.mass_displaced3 - 39.09615) / 12.30925) / math.sqrt(2)))) / 0.8447299   # +0.0%  mass_displaced3 > 39.1
        - 5.499499e-05 * (0.02246636 * 0.5 * ((Q.z_photon - 0.1149688) / 0.02246636) * (1 + math.erf(((Q.z_photon - 0.1149688) / 0.02246636) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dzerr_76 - 0.0) / 1e-06) * (1 + math.erf(((Q.dzerr_76 - 0.0) / 1e-06) / math.sqrt(2)))) / 0.0001838092   # -0.0%  z_photon > 0.115 and dzerr_76 > 0
        + 1.553114e-05 * (0.02365497 * 0.5 * ((Q.psi_0p3 - 0.8413991) / 0.02365497) * (1 + math.erf(((Q.psi_0p3 - 0.8413991) / 0.02365497) / math.sqrt(2)))) / 0.07920452   # +0.0%  psi_0p3 > 0.8414
        - 4.179209e-06 * (0.04808774 * 0.5 * ((Q.lepsj_3_dr - 0.09790963) / 0.04808774) * (1 + math.erf(((Q.lepsj_3_dr - 0.09790963) / 0.04808774) / math.sqrt(2)))) / 0.008924141   # -0.0%  lepsj_3_dr > 0.09791
        - 3.871925e-06 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (1.26387 * 0.5 * ((2.627723 - Q.ak02_2_sd0_3) / 1.26387) * (1 + math.erf(((2.627723 - Q.ak02_2_sd0_3) / 1.26387) / math.sqrt(2)))) / 19.90112   # -0.0%  n_s3d_above_3 > 3 and ak02_2_sd0_3 < 2.628
    )
    return z


def neuron_125(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.19076e-06
    )
    return z


def neuron_126(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.632629e-06
    )
    return z


def neuron_127(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.9667e-07
    )
    return z


def class_token(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q), neuron_16(Q), neuron_17(Q), neuron_18(Q), neuron_19(Q), neuron_20(Q), neuron_21(Q), neuron_22(Q), neuron_23(Q), neuron_24(Q), neuron_25(Q), neuron_26(Q), neuron_27(Q), neuron_28(Q), neuron_29(Q), neuron_30(Q), neuron_31(Q), neuron_32(Q), neuron_33(Q), neuron_34(Q), neuron_35(Q), neuron_36(Q), neuron_37(Q), neuron_38(Q), neuron_39(Q), neuron_40(Q), neuron_41(Q), neuron_42(Q), neuron_43(Q), neuron_44(Q), neuron_45(Q), neuron_46(Q), neuron_47(Q), neuron_48(Q), neuron_49(Q), neuron_50(Q), neuron_51(Q), neuron_52(Q), neuron_53(Q), neuron_54(Q), neuron_55(Q), neuron_56(Q), neuron_57(Q), neuron_58(Q), neuron_59(Q), neuron_60(Q), neuron_61(Q), neuron_62(Q), neuron_63(Q), neuron_64(Q), neuron_65(Q), neuron_66(Q), neuron_67(Q), neuron_68(Q), neuron_69(Q), neuron_70(Q), neuron_71(Q), neuron_72(Q), neuron_73(Q), neuron_74(Q), neuron_75(Q), neuron_76(Q), neuron_77(Q), neuron_78(Q), neuron_79(Q), neuron_80(Q), neuron_81(Q), neuron_82(Q), neuron_83(Q), neuron_84(Q), neuron_85(Q), neuron_86(Q), neuron_87(Q), neuron_88(Q), neuron_89(Q), neuron_90(Q), neuron_91(Q), neuron_92(Q), neuron_93(Q), neuron_94(Q), neuron_95(Q), neuron_96(Q), neuron_97(Q), neuron_98(Q), neuron_99(Q), neuron_100(Q), neuron_101(Q), neuron_102(Q), neuron_103(Q), neuron_104(Q), neuron_105(Q), neuron_106(Q), neuron_107(Q), neuron_108(Q), neuron_109(Q), neuron_110(Q), neuron_111(Q), neuron_112(Q), neuron_113(Q), neuron_114(Q), neuron_115(Q), neuron_116(Q), neuron_117(Q), neuron_118(Q), neuron_119(Q), neuron_120(Q), neuron_121(Q), neuron_122(Q), neuron_123(Q), neuron_124(Q), neuron_125(Q), neuron_126(Q), neuron_127(Q)]


H_AVG = [0.005213402211666107, 5.325262009137077e-06, 5.447297553473618e-06, 8.39983385958476e-06, 2.7921348646486877e-06, 2.1569171622104477e-06, 1.7701049728202634e-05, 6.43081875750795e-06, 2.926468141595251e-06, 2.4436594685539603e-06, 1.57010381371947e-05, 1.887711914605461e-05, 5.099296686239541e-06, 4.422908205015119e-06, 2.845302788045956e-06, 6.4595205913065e-06, 1.1932521929064712, 1.0353457582823467e-05, 1.1106555627568493, 1.4078282219998073e-06, 8.253668966062833e-07, 0.0016836648574098945, 1.044851615006337e-06, 6.978847864047566e-07, 0.9289592050510637, 0.15842534962165986, 3.2312607345375e-06, 6.3084908106247894e-06, 9.617203431844246e-06, 9.183853762806393e-06, 8.603045898780692e-06, 3.5095190469291992e-06, 1.4413466487894766e-05, 2.567321189417271e-06, 1.3884848158340901e-05, 6.743290668964619e-06, 4.831943442695774e-06, 4.357430952950381e-06, 9.411164683115203e-06, 8.274269930552691e-06, 7.775515200592054e-07, 3.814669113921809e-08, 8.77903585205786e-06, 9.407703146280255e-06, 3.324341378174722e-06, 6.849329110991675e-06, 1.3687571254195063e-07, 2.4920677788031753e-07, 8.791771506366786e-06, 2.243812787128263e-06, 5.175112164579332e-06, 2.0122010937484447e-06, 0.7913817403420338, 1.8893584865509183e-06, 1.190658349514706e-05, 2.066937304334715e-05, 7.072324024193222e-07, 1.9596457150328206e-06, 3.6171466035739286e-06, 6.029329142620554e-06, 4.666753739002161e-06, 5.840553967573214e-06, 9.422956850357878e-07, 8.308504220622126e-06, 1.258623342437204e-06, 5.791052899439819e-06, 2.7097730708192103e-06, 5.219506874709623e-06, 0.8603143000774194, 6.926140031282557e-06, 0.2609537600388446, 3.0297067041828996e-06, 1.1641183164101676e-06, 1.69963896041736e-05, 4.493394499149872e-06, 1.9738386072276626e-06, 5.542784947465407e-06, 3.0874869025865337e-06, 0.47144373984373145, 3.602386641432531e-05, 9.707700883154757e-06, 0.43426584748544617, 2.9420714326988673e-06, 1.0572172095992778, 7.427509262925014e-05, 8.185525643966685e-07, 1.7099020624300465e-05, 1.6191605709536816e-06, 2.996976888880454e-07, 1.7353728480884456e-06, 0.17332890881892243, 3.992636266048066e-06, 6.724729701090837e-06, 8.464407983410638e-07, 2.3221680294227554e-06, 4.369498469714017e-07, 4.2780052922353207e-07, 1.0282156972522611, 4.990161528439785e-07, 0.0019342397572472692, 6.264072453632252e-06, 8.87825899553718e-06, 5.490507192007499e-06, 3.414371576582198e-06, 0.9001446928678405, 2.7323811082169414e-05, 2.1609332179650664e-05, 1.1585598258534446e-05, 4.470697604119778e-06, 5.20014907579025e-07, 3.5796001611743122e-06, 3.669611942314077e-06, 5.237820005277172e-06, 2.6659268769435585e-05, 4.677413926401641e-06, 1.0584868132579215, 3.3992972703345004e-07, 2.862132987502264e-06, 8.233216249209363e-06, 6.336852607091714e-07, 0.9729515287895683, 5.7305110203742515e-06, 2.395899400653434e-06, 0.21701453386205916, 0.6905891026004972, 1.1907604857697152e-06, 7.632628694409505e-06, 1.9667004380607978e-07]
T = [4.228600763648476, 5.85305473624375, 4.384531818911559, 4.584807084974399, 5.156991041288111, 9.315514647584372, 4.084561774335694, 4.55224470602118, 7.295875568723221, 10.798278655194101]


def logits(h):
    # rounded to a multiple of 2^-20 (the formula's class scores are exact binary fractions): removes floating-point noise
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        0.4043572 + T[0] * (   # class QCD
            - 0.1839679 * h[18] / H_AVG[18]
            + 0.1450238 * h[97] / H_AVG[97]
            - 0.1004791 * h[68] / H_AVG[68]
            + 0.09622778 * h[52] / H_AVG[52]
            - 0.09087779 * h[120] / H_AVG[120]
            - 0.0725957 * h[24] / H_AVG[24]
            + 0.06968928 * h[104] / H_AVG[104]
            + 0.0663946 * h[81] / H_AVG[81]
            - 0.03755033 * h[83] / H_AVG[83]
            + 0.03188486 * h[115] / H_AVG[115]
            - 0.02977868 * h[124] / H_AVG[124]
            + 0.02964588 * h[78] / H_AVG[78]
            + 0.02873186 * h[16] / H_AVG[16]
            - 0.009352773 * h[90] / H_AVG[90]
            + 0.003583997 * h[123] / H_AVG[123]
            + 0.003010285 * h[70] / H_AVG[70]
            + 0.001065108 * h[25] / H_AVG[25]
            + 6.273124e-05 * h[0] / H_AVG[0]
            - 4.041967e-05 * h[99] / H_AVG[99]
            - 3.118053e-05 * h[21] / H_AVG[21]
            + 1.909404e-06 * h[84] / H_AVG[84]
            + 4.504673e-07 * h[55] / H_AVG[55]
            + 3.023331e-07 * h[79] / H_AVG[79]
            - 2.886248e-07 * h[105] / H_AVG[105]
            - 2.342195e-07 * h[106] / H_AVG[106]
            + 1.903855e-07 * h[113] / H_AVG[113]
            - 1.808021e-07 * h[34] / H_AVG[34]
            + 1.623486e-07 * h[73] / H_AVG[73]
            + 1.507747e-07 * h[10] / H_AVG[10]
            + 1.436806e-07 * h[11] / H_AVG[11]
            - 1.223715e-07 * h[6] / H_AVG[6]
            + 1.167746e-07 * h[86] / H_AVG[86]
            - 1.11044e-07 * h[32] / H_AVG[32]
            + 1.020627e-07 * h[17] / H_AVG[17]
            + 9.223545e-08 * h[38] / H_AVG[38]
            - 7.574429e-08 * h[42] / H_AVG[42]
            - 6.750378e-08 * h[54] / H_AVG[54]
            - 6.415997e-08 * h[29] / H_AVG[29]
            + 5.816933e-08 * h[118] / H_AVG[118]
            + 4.540693e-08 * h[48] / H_AVG[48]
            - 4.333376e-08 * h[45] / H_AVG[45]
            - 4.185009e-08 * h[39] / H_AVG[39]
            + 3.858459e-08 * h[27] / H_AVG[27]
            + 3.842549e-08 * h[63] / H_AVG[63]
            + 3.801557e-08 * h[30] / H_AVG[30]
            + 3.722688e-08 * h[43] / H_AVG[43]
            + 3.462842e-08 * h[121] / H_AVG[121]
            - 3.401386e-08 * h[80] / H_AVG[80]
            - 3.113252e-08 * h[126] / H_AVG[126]
            + 3.005363e-08 * h[60] / H_AVG[60]
            - 2.960026e-08 * h[61] / H_AVG[61]
            - 2.810678e-08 * h[102] / H_AVG[102]
            - 2.806466e-08 * h[107] / H_AVG[107]
            - 2.762834e-08 * h[7] / H_AVG[7]
            + 2.488817e-08 * h[92] / H_AVG[92]
            + 2.433955e-08 * h[35] / H_AVG[35]
            + 2.366997e-08 * h[15] / H_AVG[15]
            - 2.2859e-08 * h[59] / H_AVG[59]
            - 2.205228e-08 * h[101] / H_AVG[101]
            - 2.108211e-08 * h[28] / H_AVG[28]
            + 2.04438e-08 * h[103] / H_AVG[103]
            + 1.924174e-08 * h[69] / H_AVG[69]
            + 1.867455e-08 * h[67] / H_AVG[67]
            + 1.808879e-08 * h[1] / H_AVG[1]
            + 1.795488e-08 * h[100] / H_AVG[100]
            - 1.649037e-08 * h[110] / H_AVG[110]
            - 1.595611e-08 * h[3] / H_AVG[3]
            + 1.532717e-08 * h[37] / H_AVG[37]
            - 1.501459e-08 * h[50] / H_AVG[50]
            + 1.411344e-08 * h[44] / H_AVG[44]
            + 1.410226e-08 * h[36] / H_AVG[36]
            - 1.386249e-08 * h[12] / H_AVG[12]
            - 1.318746e-08 * h[114] / H_AVG[114]
            - 1.307442e-08 * h[2] / H_AVG[2]
            + 1.260076e-08 * h[58] / H_AVG[58]
            - 1.254799e-08 * h[112] / H_AVG[112]
            + 1.228149e-08 * h[76] / H_AVG[76]
            - 1.18391e-08 * h[13] / H_AVG[13]
            - 1.149736e-08 * h[19] / H_AVG[19]
            - 1.128497e-08 * h[111] / H_AVG[111]
            + 1.075536e-08 * h[74] / H_AVG[74]
            + 1.020596e-08 * h[108] / H_AVG[108]
            + 9.593832e-09 * h[5] / H_AVG[5]
            - 9.170289e-09 * h[8] / H_AVG[8]
            - 8.793304e-09 * h[91] / H_AVG[91]
            - 8.197695e-09 * h[4] / H_AVG[4]
            - 7.852949e-09 * h[117] / H_AVG[117]
            + 6.635868e-09 * h[53] / H_AVG[53]
            - 6.562289e-09 * h[66] / H_AVG[66]
            - 5.972566e-09 * h[33] / H_AVG[33]
            - 5.802799e-09 * h[65] / H_AVG[65]
            + 5.366003e-09 * h[82] / H_AVG[82]
            - 5.04261e-09 * h[31] / H_AVG[31]
            - 4.633829e-09 * h[71] / H_AVG[71]
            + 4.30393e-09 * h[94] / H_AVG[94]
            + 4.156788e-09 * h[9] / H_AVG[9]
            + 3.653794e-09 * h[26] / H_AVG[26]
            - 3.274409e-09 * h[14] / H_AVG[14]
            + 3.24364e-09 * h[57] / H_AVG[57]
            + 3.065689e-09 * h[49] / H_AVG[49]
            + 2.994855e-09 * h[77] / H_AVG[77]
            - 2.988762e-09 * h[22] / H_AVG[22]
            + 2.772687e-09 * h[122] / H_AVG[122]
            + 2.241451e-09 * h[72] / H_AVG[72]
            + 2.146796e-09 * h[51] / H_AVG[51]
            - 2.113477e-09 * h[23] / H_AVG[23]
            - 1.459556e-09 * h[89] / H_AVG[89]
            + 1.426087e-09 * h[125] / H_AVG[125]
            - 1.338843e-09 * h[75] / H_AVG[75]
            - 1.158579e-09 * h[87] / H_AVG[87]
            + 1.154734e-09 * h[93] / H_AVG[93]
            - 1.03847e-09 * h[40] / H_AVG[40]
            - 9.649455e-10 * h[20] / H_AVG[20]
            + 8.621927e-10 * h[119] / H_AVG[119]
            - 7.886969e-10 * h[62] / H_AVG[62]
            - 7.091739e-10 * h[95] / H_AVG[95]
            - 6.572435e-10 * h[96] / H_AVG[96]
            + 3.874542e-10 * h[116] / H_AVG[116]
            + 3.700086e-10 * h[56] / H_AVG[56]
            + 2.572425e-10 * h[85] / H_AVG[85]
            + 1.531209e-10 * h[109] / H_AVG[109]
            - 1.333148e-10 * h[98] / H_AVG[98]
            - 1.064831e-10 * h[88] / H_AVG[88]
            - 7.177586e-11 * h[46] / H_AVG[46]
            - 6.775664e-11 * h[47] / H_AVG[47]
            - 4.17844e-11 * h[64] / H_AVG[64]
            + 1.628109e-11 * h[127] / H_AVG[127]
            + 1.438076e-12 * h[41] / H_AVG[41]
        ),
        -0.4064285 + T[1] * (   # class Hbb
            + 0.2720548 * h[16] / H_AVG[16]
            + 0.1967193 * h[120] / H_AVG[120]
            - 0.08986421 * h[68] / H_AVG[68]
            + 0.08247007 * h[18] / H_AVG[18]
            - 0.06562683 * h[24] / H_AVG[24]
            - 0.06358971 * h[52] / H_AVG[52]
            - 0.06206422 * h[97] / H_AVG[97]
            - 0.06025215 * h[78] / H_AVG[78]
            + 0.04374144 * h[115] / H_AVG[115]
            + 0.02333504 * h[81] / H_AVG[81]
            - 0.01096976 * h[70] / H_AVG[70]
            + 0.009016155 * h[83] / H_AVG[83]
            - 0.007845292 * h[25] / H_AVG[25]
            - 0.004914052 * h[90] / H_AVG[90]
            - 0.004089725 * h[104] / H_AVG[104]
            + 0.002957503 * h[124] / H_AVG[124]
            + 0.0003278494 * h[123] / H_AVG[123]
            + 0.0001028897 * h[0] / H_AVG[0]
            - 2.957956e-05 * h[99] / H_AVG[99]
            - 2.504014e-05 * h[21] / H_AVG[21]
            + 1.378955e-06 * h[84] / H_AVG[84]
            + 3.253988e-07 * h[55] / H_AVG[55]
            + 2.183246e-07 * h[79] / H_AVG[79]
            - 2.084961e-07 * h[105] / H_AVG[105]
            - 1.691721e-07 * h[106] / H_AVG[106]
            + 1.374699e-07 * h[113] / H_AVG[113]
            - 1.305997e-07 * h[34] / H_AVG[34]
            + 1.172454e-07 * h[73] / H_AVG[73]
            + 1.089252e-07 * h[10] / H_AVG[10]
            + 1.038309e-07 * h[11] / H_AVG[11]
            - 8.841719e-08 * h[6] / H_AVG[6]
            + 8.433648e-08 * h[86] / H_AVG[86]
            - 8.018998e-08 * h[32] / H_AVG[32]
            + 7.371452e-08 * h[17] / H_AVG[17]
            + 6.66463e-08 * h[38] / H_AVG[38]
            - 5.471916e-08 * h[42] / H_AVG[42]
            - 4.875822e-08 * h[54] / H_AVG[54]
            - 4.63407e-08 * h[29] / H_AVG[29]
            + 4.201649e-08 * h[118] / H_AVG[118]
            + 3.279937e-08 * h[48] / H_AVG[48]
            - 3.129536e-08 * h[45] / H_AVG[45]
            - 3.023348e-08 * h[39] / H_AVG[39]
            + 2.787721e-08 * h[27] / H_AVG[27]
            + 2.775697e-08 * h[63] / H_AVG[63]
            + 2.745884e-08 * h[30] / H_AVG[30]
            + 2.6893e-08 * h[43] / H_AVG[43]
            + 2.501718e-08 * h[121] / H_AVG[121]
            - 2.456555e-08 * h[80] / H_AVG[80]
            - 2.248663e-08 * h[126] / H_AVG[126]
            + 2.171322e-08 * h[60] / H_AVG[60]
            - 2.138353e-08 * h[61] / H_AVG[61]
            - 2.030387e-08 * h[102] / H_AVG[102]
            - 2.025748e-08 * h[107] / H_AVG[107]
            - 1.995948e-08 * h[7] / H_AVG[7]
            + 1.797933e-08 * h[92] / H_AVG[92]
            + 1.758234e-08 * h[35] / H_AVG[35]
            + 1.70967e-08 * h[15] / H_AVG[15]
            - 1.651427e-08 * h[59] / H_AVG[59]
            - 1.59318e-08 * h[101] / H_AVG[101]
            - 1.522541e-08 * h[28] / H_AVG[28]
            + 1.476688e-08 * h[103] / H_AVG[103]
            + 1.389688e-08 * h[69] / H_AVG[69]
            + 1.348776e-08 * h[67] / H_AVG[67]
            + 1.306789e-08 * h[1] / H_AVG[1]
            + 1.297279e-08 * h[100] / H_AVG[100]
            - 1.191484e-08 * h[110] / H_AVG[110]
            - 1.152763e-08 * h[3] / H_AVG[3]
            + 1.107044e-08 * h[37] / H_AVG[37]
            - 1.083804e-08 * h[50] / H_AVG[50]
            + 1.019685e-08 * h[44] / H_AVG[44]
            + 1.018792e-08 * h[36] / H_AVG[36]
            - 1.001096e-08 * h[12] / H_AVG[12]
            - 9.521989e-09 * h[114] / H_AVG[114]
            - 9.438351e-09 * h[2] / H_AVG[2]
            + 9.103109e-09 * h[58] / H_AVG[58]
            - 9.063473e-09 * h[112] / H_AVG[112]
            + 8.871175e-09 * h[76] / H_AVG[76]
            - 8.552735e-09 * h[13] / H_AVG[13]
            - 8.313408e-09 * h[19] / H_AVG[19]
            - 8.152074e-09 * h[111] / H_AVG[111]
            + 7.769396e-09 * h[74] / H_AVG[74]
            + 7.375656e-09 * h[108] / H_AVG[108]
            + 6.93084e-09 * h[5] / H_AVG[5]
            - 6.624367e-09 * h[8] / H_AVG[8]
            - 6.352119e-09 * h[91] / H_AVG[91]
            - 5.922157e-09 * h[4] / H_AVG[4]
            - 5.672706e-09 * h[117] / H_AVG[117]
            + 4.790895e-09 * h[53] / H_AVG[53]
            - 4.737026e-09 * h[66] / H_AVG[66]
            - 4.3154e-09 * h[33] / H_AVG[33]
            - 4.191083e-09 * h[65] / H_AVG[65]
            + 3.875864e-09 * h[82] / H_AVG[82]
            - 3.643739e-09 * h[31] / H_AVG[31]
            - 3.345864e-09 * h[71] / H_AVG[71]
            + 3.10796e-09 * h[94] / H_AVG[94]
            + 3.002358e-09 * h[9] / H_AVG[9]
            + 2.637614e-09 * h[26] / H_AVG[26]
            - 2.36509e-09 * h[14] / H_AVG[14]
            + 2.342775e-09 * h[57] / H_AVG[57]
            + 2.214719e-09 * h[49] / H_AVG[49]
            + 2.1644e-09 * h[77] / H_AVG[77]
            - 2.159386e-09 * h[22] / H_AVG[22]
            + 2.002576e-09 * h[122] / H_AVG[122]
            + 1.619266e-09 * h[72] / H_AVG[72]
            + 1.550671e-09 * h[51] / H_AVG[51]
            - 1.526588e-09 * h[23] / H_AVG[23]
            - 1.054583e-09 * h[89] / H_AVG[89]
            + 1.030648e-09 * h[125] / H_AVG[125]
            - 9.668832e-10 * h[75] / H_AVG[75]
            - 8.365562e-10 * h[87] / H_AVG[87]
            + 8.337059e-10 * h[93] / H_AVG[93]
            - 7.501202e-10 * h[40] / H_AVG[40]
            - 6.972354e-10 * h[20] / H_AVG[20]
            + 6.221451e-10 * h[119] / H_AVG[119]
            - 5.698102e-10 * h[62] / H_AVG[62]
            - 5.122331e-10 * h[95] / H_AVG[95]
            - 4.746562e-10 * h[96] / H_AVG[96]
            + 2.799498e-10 * h[116] / H_AVG[116]
            + 2.671903e-10 * h[56] / H_AVG[56]
            + 1.85713e-10 * h[85] / H_AVG[85]
            + 1.106349e-10 * h[109] / H_AVG[109]
            - 9.631982e-11 * h[98] / H_AVG[98]
            - 7.696754e-11 * h[88] / H_AVG[88]
            - 5.182133e-11 * h[46] / H_AVG[46]
            - 4.896144e-11 * h[47] / H_AVG[47]
            - 2.658327e-11 * h[64] / H_AVG[64]
            + 1.172729e-11 * h[127] / H_AVG[127]
            + 1.036945e-12 * h[41] / H_AVG[41]
        ),
        0.1415668 + T[2] * (   # class Hcc
            - 0.2297387 * h[83] / H_AVG[83]
            + 0.1905378 * h[18] / H_AVG[18]
            - 0.1268806 * h[24] / H_AVG[24]
            - 0.07582932 * h[81] / H_AVG[81]
            + 0.06243318 * h[124] / H_AVG[124]
            - 0.05326441 * h[16] / H_AVG[16]
            - 0.04805456 * h[97] / H_AVG[97]
            + 0.04544753 * h[104] / H_AVG[104]
            + 0.04250701 * h[78] / H_AVG[78]
            + 0.03153984 * h[120] / H_AVG[120]
            - 0.02399644 * h[52] / H_AVG[52]
            - 0.0220809 * h[68] / H_AVG[68]
            + 0.02142401 * h[115] / H_AVG[115]
            + 0.01374441 * h[70] / H_AVG[70]
            - 0.006945805 * h[25] / H_AVG[25]
            - 0.00510554 * h[123] / H_AVG[123]
            + 0.0002015343 * h[90] / H_AVG[90]
            + 0.0001973879 * h[0] / H_AVG[0]
            - 3.918151e-05 * h[99] / H_AVG[99]
            - 2.605791e-05 * h[21] / H_AVG[21]
            + 1.841395e-06 * h[84] / H_AVG[84]
            + 4.346119e-07 * h[55] / H_AVG[55]
            + 2.916206e-07 * h[79] / H_AVG[79]
            - 2.783453e-07 * h[105] / H_AVG[105]
            - 2.25944e-07 * h[106] / H_AVG[106]
            + 1.83624e-07 * h[113] / H_AVG[113]
            - 1.743537e-07 * h[34] / H_AVG[34]
            + 1.565706e-07 * h[73] / H_AVG[73]
            + 1.454653e-07 * h[10] / H_AVG[10]
            + 1.386127e-07 * h[11] / H_AVG[11]
            - 1.180302e-07 * h[6] / H_AVG[6]
            + 1.126042e-07 * h[86] / H_AVG[86]
            - 1.070989e-07 * h[32] / H_AVG[32]
            + 9.847248e-08 * h[17] / H_AVG[17]
            + 8.894378e-08 * h[38] / H_AVG[38]
            - 7.30504e-08 * h[42] / H_AVG[42]
            - 6.511541e-08 * h[54] / H_AVG[54]
            - 6.189409e-08 * h[29] / H_AVG[29]
            + 5.610947e-08 * h[118] / H_AVG[118]
            + 4.379457e-08 * h[48] / H_AVG[48]
            - 4.17906e-08 * h[45] / H_AVG[45]
            - 4.037035e-08 * h[39] / H_AVG[39]
            + 3.722617e-08 * h[27] / H_AVG[27]
            + 3.707022e-08 * h[63] / H_AVG[63]
            + 3.666137e-08 * h[30] / H_AVG[30]
            + 3.589763e-08 * h[43] / H_AVG[43]
            + 3.341476e-08 * h[121] / H_AVG[121]
            - 3.281072e-08 * h[80] / H_AVG[80]
            - 3.002244e-08 * h[126] / H_AVG[126]
            + 2.899326e-08 * h[60] / H_AVG[60]
            - 2.855035e-08 * h[61] / H_AVG[61]
            - 2.710756e-08 * h[102] / H_AVG[102]
            - 2.703999e-08 * h[107] / H_AVG[107]
            - 2.666429e-08 * h[7] / H_AVG[7]
            + 2.400464e-08 * h[92] / H_AVG[92]
            + 2.347851e-08 * h[35] / H_AVG[35]
            + 2.283122e-08 * h[15] / H_AVG[15]
            - 2.20497e-08 * h[59] / H_AVG[59]
            - 2.126303e-08 * h[101] / H_AVG[101]
            - 2.033441e-08 * h[28] / H_AVG[28]
            + 1.972052e-08 * h[103] / H_AVG[103]
            + 1.85591e-08 * h[69] / H_AVG[69]
            + 1.80139e-08 * h[67] / H_AVG[67]
            + 1.745263e-08 * h[1] / H_AVG[1]
            + 1.732223e-08 * h[100] / H_AVG[100]
            - 1.590993e-08 * h[110] / H_AVG[110]
            - 1.539329e-08 * h[3] / H_AVG[3]
            + 1.478168e-08 * h[37] / H_AVG[37]
            - 1.447747e-08 * h[50] / H_AVG[50]
            + 1.361537e-08 * h[44] / H_AVG[44]
            + 1.360609e-08 * h[36] / H_AVG[36]
            - 1.336703e-08 * h[12] / H_AVG[12]
            - 1.271659e-08 * h[114] / H_AVG[114]
            - 1.261023e-08 * h[2] / H_AVG[2]
            + 1.215806e-08 * h[58] / H_AVG[58]
            - 1.210717e-08 * h[112] / H_AVG[112]
            + 1.184772e-08 * h[76] / H_AVG[76]
            - 1.142012e-08 * h[13] / H_AVG[13]
            - 1.1096e-08 * h[19] / H_AVG[19]
            - 1.08828e-08 * h[111] / H_AVG[111]
            + 1.037419e-08 * h[74] / H_AVG[74]
            + 9.84595e-09 * h[108] / H_AVG[108]
            + 9.258987e-09 * h[5] / H_AVG[5]
            - 8.847818e-09 * h[8] / H_AVG[8]
            - 8.485157e-09 * h[91] / H_AVG[91]
            - 7.907354e-09 * h[4] / H_AVG[4]
            - 7.574329e-09 * h[117] / H_AVG[117]
            + 6.401961e-09 * h[53] / H_AVG[53]
            - 6.330225e-09 * h[66] / H_AVG[66]
            - 5.760984e-09 * h[33] / H_AVG[33]
            - 5.598612e-09 * h[65] / H_AVG[65]
            + 5.176739e-09 * h[82] / H_AVG[82]
            - 4.86157e-09 * h[31] / H_AVG[31]
            - 4.467567e-09 * h[71] / H_AVG[71]
            + 4.151823e-09 * h[94] / H_AVG[94]
            + 4.008803e-09 * h[9] / H_AVG[9]
            + 3.520104e-09 * h[26] / H_AVG[26]
            - 3.159826e-09 * h[14] / H_AVG[14]
            + 3.129531e-09 * h[57] / H_AVG[57]
            + 2.956416e-09 * h[49] / H_AVG[49]
            + 2.888282e-09 * h[77] / H_AVG[77]
            - 2.884456e-09 * h[22] / H_AVG[22]
            + 2.675341e-09 * h[122] / H_AVG[122]
            + 2.161269e-09 * h[72] / H_AVG[72]
            + 2.071909e-09 * h[51] / H_AVG[51]
            - 2.039327e-09 * h[23] / H_AVG[23]
            - 1.407627e-09 * h[89] / H_AVG[89]
            + 1.375553e-09 * h[125] / H_AVG[125]
            - 1.291212e-09 * h[75] / H_AVG[75]
            - 1.11723e-09 * h[87] / H_AVG[87]
            + 1.113171e-09 * h[93] / H_AVG[93]
            - 1.001491e-09 * h[40] / H_AVG[40]
            - 9.312304e-10 * h[20] / H_AVG[20]
            + 8.314319e-10 * h[119] / H_AVG[119]
            - 7.608137e-10 * h[62] / H_AVG[62]
            - 6.838173e-10 * h[95] / H_AVG[95]
            - 6.33723e-10 * h[96] / H_AVG[96]
            + 3.739296e-10 * h[116] / H_AVG[116]
            + 3.570073e-10 * h[56] / H_AVG[56]
            + 2.482705e-10 * h[85] / H_AVG[85]
            + 1.476795e-10 * h[109] / H_AVG[109]
            - 1.285921e-10 * h[98] / H_AVG[98]
            - 1.026362e-10 * h[88] / H_AVG[88]
            - 6.922773e-11 * h[46] / H_AVG[46]
            - 6.533033e-11 * h[47] / H_AVG[47]
            - 4.125053e-11 * h[64] / H_AVG[64]
            + 1.569597e-11 * h[127] / H_AVG[127]
            + 1.388608e-12 * h[41] / H_AVG[41]
        ),
        0.1286925 + T[3] * (   # class Hgg
            - 0.1966194 * h[104] / H_AVG[104]
            - 0.1529139 * h[68] / H_AVG[68]
            - 0.1096564 * h[18] / H_AVG[18]
            - 0.09549395 * h[16] / H_AVG[16]
            + 0.09176392 * h[52] / H_AVG[52]
            - 0.07503172 * h[97] / H_AVG[97]
            + 0.05880923 * h[124] / H_AVG[124]
            + 0.05498618 * h[115] / H_AVG[115]
            - 0.04973091 * h[81] / H_AVG[81]
            - 0.04653427 * h[24] / H_AVG[24]
            + 0.01926609 * h[83] / H_AVG[83]
            - 0.01871968 * h[78] / H_AVG[78]
            + 0.01211698 * h[120] / H_AVG[120]
            + 0.008553743 * h[70] / H_AVG[70]
            - 0.004625542 * h[25] / H_AVG[25]
            + 0.003954479 * h[90] / H_AVG[90]
            - 0.0009233354 * h[123] / H_AVG[123]
            + 0.0002307594 * h[0] / H_AVG[0]
            - 3.725786e-05 * h[99] / H_AVG[99]
            - 2.677012e-05 * h[21] / H_AVG[21]
            + 1.760721e-06 * h[84] / H_AVG[84]
            + 4.154579e-07 * h[55] / H_AVG[55]
            + 2.788131e-07 * h[79] / H_AVG[79]
            - 2.661794e-07 * h[105] / H_AVG[105]
            - 2.160004e-07 * h[106] / H_AVG[106]
            + 1.755786e-07 * h[113] / H_AVG[113]
            - 1.667524e-07 * h[34] / H_AVG[34]
            + 1.497133e-07 * h[73] / H_AVG[73]
            + 1.390471e-07 * h[10] / H_AVG[10]
            + 1.325073e-07 * h[11] / H_AVG[11]
            - 1.128571e-07 * h[6] / H_AVG[6]
            + 1.076823e-07 * h[86] / H_AVG[86]
            - 1.024272e-07 * h[32] / H_AVG[32]
            + 9.415323e-08 * h[17] / H_AVG[17]
            + 8.509195e-08 * h[38] / H_AVG[38]
            - 6.984848e-08 * h[42] / H_AVG[42]
            - 6.225566e-08 * h[54] / H_AVG[54]
            - 5.916584e-08 * h[29] / H_AVG[29]
            + 5.365652e-08 * h[118] / H_AVG[118]
            + 4.187807e-08 * h[48] / H_AVG[48]
            - 3.996515e-08 * h[45] / H_AVG[45]
            - 3.859426e-08 * h[39] / H_AVG[39]
            + 3.559111e-08 * h[27] / H_AVG[27]
            + 3.543536e-08 * h[63] / H_AVG[63]
            + 3.505383e-08 * h[30] / H_AVG[30]
            + 3.433054e-08 * h[43] / H_AVG[43]
            + 3.194312e-08 * h[121] / H_AVG[121]
            - 3.136648e-08 * h[80] / H_AVG[80]
            - 2.871099e-08 * h[126] / H_AVG[126]
            + 2.771703e-08 * h[60] / H_AVG[60]
            - 2.730447e-08 * h[61] / H_AVG[61]
            - 2.59217e-08 * h[102] / H_AVG[102]
            - 2.587335e-08 * h[107] / H_AVG[107]
            - 2.548021e-08 * h[7] / H_AVG[7]
            + 2.295145e-08 * h[92] / H_AVG[92]
            + 2.24483e-08 * h[35] / H_AVG[35]
            + 2.182577e-08 * h[15] / H_AVG[15]
            - 2.108527e-08 * h[59] / H_AVG[59]
            - 2.033484e-08 * h[101] / H_AVG[101]
            - 1.944271e-08 * h[28] / H_AVG[28]
            + 1.885487e-08 * h[103] / H_AVG[103]
            + 1.774384e-08 * h[69] / H_AVG[69]
            + 1.721885e-08 * h[67] / H_AVG[67]
            + 1.668247e-08 * h[1] / H_AVG[1]
            + 1.655841e-08 * h[100] / H_AVG[100]
            - 1.520736e-08 * h[110] / H_AVG[110]
            - 1.471611e-08 * h[3] / H_AVG[3]
            + 1.413385e-08 * h[37] / H_AVG[37]
            - 1.384888e-08 * h[50] / H_AVG[50]
            + 1.301676e-08 * h[44] / H_AVG[44]
            + 1.300554e-08 * h[36] / H_AVG[36]
            - 1.27851e-08 * h[12] / H_AVG[12]
            - 1.21603e-08 * h[114] / H_AVG[114]
            - 1.205615e-08 * h[2] / H_AVG[2]
            + 1.162545e-08 * h[58] / H_AVG[58]
            - 1.157807e-08 * h[112] / H_AVG[112]
            + 1.133059e-08 * h[76] / H_AVG[76]
            - 1.091917e-08 * h[13] / H_AVG[13]
            - 1.061065e-08 * h[19] / H_AVG[19]
            - 1.040747e-08 * h[111] / H_AVG[111]
            + 9.919403e-09 * h[74] / H_AVG[74]
            + 9.413497e-09 * h[108] / H_AVG[108]
            + 8.852019e-09 * h[5] / H_AVG[5]
            - 8.45991e-09 * h[8] / H_AVG[8]
            - 8.108892e-09 * h[91] / H_AVG[91]
            - 7.561152e-09 * h[4] / H_AVG[4]
            - 7.24184e-09 * h[117] / H_AVG[117]
            + 6.118151e-09 * h[53] / H_AVG[53]
            - 6.050801e-09 * h[66] / H_AVG[66]
            - 5.508866e-09 * h[33] / H_AVG[33]
            - 5.345256e-09 * h[65] / H_AVG[65]
            + 4.948462e-09 * h[82] / H_AVG[82]
            - 4.649546e-09 * h[31] / H_AVG[31]
            - 4.273406e-09 * h[71] / H_AVG[71]
            + 3.968461e-09 * h[94] / H_AVG[94]
            + 3.833468e-09 * h[9] / H_AVG[9]
            + 3.367821e-09 * h[26] / H_AVG[26]
            - 3.019638e-09 * h[14] / H_AVG[14]
            + 2.992044e-09 * h[57] / H_AVG[57]
            + 2.827234e-09 * h[49] / H_AVG[49]
            + 2.76268e-09 * h[77] / H_AVG[77]
            - 2.755883e-09 * h[22] / H_AVG[22]
            + 2.556715e-09 * h[122] / H_AVG[122]
            + 2.067397e-09 * h[72] / H_AVG[72]
            + 1.981167e-09 * h[51] / H_AVG[51]
            - 1.948886e-09 * h[23] / H_AVG[23]
            - 1.34619e-09 * h[89] / H_AVG[89]
            + 1.315168e-09 * h[125] / H_AVG[125]
            - 1.234835e-09 * h[75] / H_AVG[75]
            - 1.068066e-09 * h[87] / H_AVG[87]
            + 1.064393e-09 * h[93] / H_AVG[93]
            - 9.576912e-10 * h[40] / H_AVG[40]
            - 8.903092e-10 * h[20] / H_AVG[20]
            + 7.948428e-10 * h[119] / H_AVG[119]
            - 7.273874e-10 * h[62] / H_AVG[62]
            - 6.538837e-10 * h[95] / H_AVG[95]
            - 6.060277e-10 * h[96] / H_AVG[96]
            + 3.573591e-10 * h[116] / H_AVG[116]
            + 3.411333e-10 * h[56] / H_AVG[56]
            + 2.373554e-10 * h[85] / H_AVG[85]
            + 1.412261e-10 * h[109] / H_AVG[109]
            - 1.229483e-10 * h[98] / H_AVG[98]
            - 9.829491e-11 * h[88] / H_AVG[88]
            - 6.618226e-11 * h[46] / H_AVG[46]
            - 6.248301e-11 * h[47] / H_AVG[47]
            - 3.782093e-11 * h[64] / H_AVG[64]
            + 1.500971e-11 * h[127] / H_AVG[127]
            + 1.325826e-12 * h[41] / H_AVG[41]
        ),
        -0.07702489 + T[4] * (   # class H4q
            - 0.3072952 * h[115] / H_AVG[115]
            - 0.08859435 * h[104] / H_AVG[104]
            + 0.08473513 * h[52] / H_AVG[52]
            - 0.0824371 * h[16] / H_AVG[16]
            - 0.07019814 * h[24] / H_AVG[24]
            - 0.06963054 * h[97] / H_AVG[97]
            - 0.06446891 * h[78] / H_AVG[78]
            + 0.05320437 * h[124] / H_AVG[124]
            - 0.04781308 * h[18] / H_AVG[18]
            - 0.04763282 * h[81] / H_AVG[81]
            + 0.04382724 * h[68] / H_AVG[68]
            - 0.01265335 * h[83] / H_AVG[83]
            + 0.01223671 * h[120] / H_AVG[120]
            + 0.009678121 * h[70] / H_AVG[70]
            - 0.004294314 * h[25] / H_AVG[25]
            - 0.0007322953 * h[90] / H_AVG[90]
            + 0.0002763491 * h[123] / H_AVG[123]
            + 0.0002273733 * h[0] / H_AVG[0]
            - 3.307421e-05 * h[99] / H_AVG[99]
            - 2.665674e-05 * h[21] / H_AVG[21]
            + 1.565226e-06 * h[84] / H_AVG[84]
            + 3.694957e-07 * h[55] / H_AVG[55]
            + 2.478969e-07 * h[79] / H_AVG[79]
            - 2.366562e-07 * h[105] / H_AVG[105]
            - 1.921119e-07 * h[106] / H_AVG[106]
            + 1.560669e-07 * h[113] / H_AVG[113]
            - 1.482579e-07 * h[34] / H_AVG[34]
            + 1.331147e-07 * h[73] / H_AVG[73]
            + 1.236433e-07 * h[10] / H_AVG[10]
            + 1.178046e-07 * h[11] / H_AVG[11]
            - 1.003528e-07 * h[6] / H_AVG[6]
            + 9.573903e-08 * h[86] / H_AVG[86]
            - 9.105393e-08 * h[32] / H_AVG[32]
            + 8.371728e-08 * h[17] / H_AVG[17]
            + 7.565502e-08 * h[38] / H_AVG[38]
            - 6.21055e-08 * h[42] / H_AVG[42]
            - 5.536305e-08 * h[54] / H_AVG[54]
            - 5.261701e-08 * h[29] / H_AVG[29]
            + 4.770923e-08 * h[118] / H_AVG[118]
            + 3.723029e-08 * h[48] / H_AVG[48]
            - 3.553235e-08 * h[45] / H_AVG[45]
            - 3.431502e-08 * h[39] / H_AVG[39]
            + 3.164555e-08 * h[27] / H_AVG[27]
            + 3.150557e-08 * h[63] / H_AVG[63]
            + 3.116967e-08 * h[30] / H_AVG[30]
            + 3.052408e-08 * h[43] / H_AVG[43]
            + 2.840071e-08 * h[121] / H_AVG[121]
            - 2.789224e-08 * h[80] / H_AVG[80]
            - 2.553061e-08 * h[126] / H_AVG[126]
            + 2.465291e-08 * h[60] / H_AVG[60]
            - 2.427046e-08 * h[61] / H_AVG[61]
            - 2.304571e-08 * h[102] / H_AVG[102]
            - 2.300483e-08 * h[107] / H_AVG[107]
            - 2.265749e-08 * h[7] / H_AVG[7]
            + 2.040514e-08 * h[92] / H_AVG[92]
            + 1.996154e-08 * h[35] / H_AVG[35]
            + 1.940846e-08 * h[15] / H_AVG[15]
            - 1.874118e-08 * h[59] / H_AVG[59]
            - 1.808012e-08 * h[101] / H_AVG[101]
            - 1.728807e-08 * h[28] / H_AVG[28]
            + 1.676358e-08 * h[103] / H_AVG[103]
            + 1.577559e-08 * h[69] / H_AVG[69]
            + 1.531276e-08 * h[67] / H_AVG[67]
            + 1.483264e-08 * h[1] / H_AVG[1]
            + 1.47216e-08 * h[100] / H_AVG[100]
            - 1.35238e-08 * h[110] / H_AVG[110]
            - 1.308199e-08 * h[3] / H_AVG[3]
            + 1.256678e-08 * h[37] / H_AVG[37]
            - 1.230358e-08 * h[50] / H_AVG[50]
            + 1.157315e-08 * h[44] / H_AVG[44]
            + 1.156325e-08 * h[36] / H_AVG[36]
            - 1.136449e-08 * h[12] / H_AVG[12]
            - 1.081308e-08 * h[114] / H_AVG[114]
            - 1.072144e-08 * h[2] / H_AVG[2]
            + 1.033555e-08 * h[58] / H_AVG[58]
            - 1.029236e-08 * h[112] / H_AVG[112]
            + 1.007364e-08 * h[76] / H_AVG[76]
            - 9.707192e-09 * h[13] / H_AVG[13]
            - 9.430918e-09 * h[19] / H_AVG[19]
            - 9.25286e-09 * h[111] / H_AVG[111]
            + 8.818406e-09 * h[74] / H_AVG[74]
            + 8.368446e-09 * h[108] / H_AVG[108]
            + 7.871065e-09 * h[5] / H_AVG[5]
            - 7.521988e-09 * h[8] / H_AVG[8]
            - 7.213716e-09 * h[91] / H_AVG[91]
            - 6.722369e-09 * h[4] / H_AVG[4]
            - 6.439961e-09 * h[117] / H_AVG[117]
            + 5.439007e-09 * h[53] / H_AVG[53]
            - 5.380075e-09 * h[66] / H_AVG[66]
            - 4.897213e-09 * h[33] / H_AVG[33]
            - 4.753578e-09 * h[65] / H_AVG[65]
            + 4.401061e-09 * h[82] / H_AVG[82]
            - 4.134217e-09 * h[31] / H_AVG[31]
            - 3.799471e-09 * h[71] / H_AVG[71]
            + 3.528808e-09 * h[94] / H_AVG[94]
            + 3.408082e-09 * h[9] / H_AVG[9]
            + 2.994284e-09 * h[26] / H_AVG[26]
            - 2.684384e-09 * h[14] / H_AVG[14]
            + 2.660194e-09 * h[57] / H_AVG[57]
            + 2.51366e-09 * h[49] / H_AVG[49]
            + 2.458325e-09 * h[77] / H_AVG[77]
            - 2.45199e-09 * h[22] / H_AVG[22]
            + 2.273709e-09 * h[122] / H_AVG[122]
            + 1.838131e-09 * h[72] / H_AVG[72]
            + 1.761157e-09 * h[51] / H_AVG[51]
            - 1.732544e-09 * h[23] / H_AVG[23]
            - 1.196811e-09 * h[89] / H_AVG[89]
            + 1.169553e-09 * h[125] / H_AVG[125]
            - 1.097836e-09 * h[75] / H_AVG[75]
            - 9.497434e-10 * h[87] / H_AVG[87]
            + 9.469299e-10 * h[93] / H_AVG[93]
            - 8.51534e-10 * h[40] / H_AVG[40]
            - 7.915305e-10 * h[20] / H_AVG[20]
            + 7.070852e-10 * h[119] / H_AVG[119]
            - 6.469195e-10 * h[62] / H_AVG[62]
            - 5.816084e-10 * h[95] / H_AVG[95]
            - 5.389458e-10 * h[96] / H_AVG[96]
            + 3.177896e-10 * h[116] / H_AVG[116]
            + 3.035274e-10 * h[56] / H_AVG[56]
            + 2.109301e-10 * h[85] / H_AVG[85]
            + 1.255996e-10 * h[109] / H_AVG[109]
            - 1.093191e-10 * h[98] / H_AVG[98]
            - 8.734789e-11 * h[88] / H_AVG[88]
            - 5.885214e-11 * h[46] / H_AVG[46]
            - 5.556018e-11 * h[47] / H_AVG[47]
            - 3.433337e-11 * h[64] / H_AVG[64]
            + 1.332903e-11 * h[127] / H_AVG[127]
            + 1.179934e-12 * h[41] / H_AVG[41]
        ),
        -0.9789527 + T[5] * (   # class Hqql
            + 0.1711706 * h[120] / H_AVG[120]
            + 0.1224841 * h[24] / H_AVG[24]
            + 0.1175162 * h[97] / H_AVG[97]
            + 0.1072776 * h[18] / H_AVG[18]
            + 0.08913074 * h[52] / H_AVG[52]
            + 0.08178855 * h[83] / H_AVG[83]
            - 0.0778014 * h[124] / H_AVG[124]
            - 0.07107935 * h[115] / H_AVG[115]
            + 0.05110565 * h[78] / H_AVG[78]
            + 0.02247109 * h[70] / H_AVG[70]
            + 0.01699667 * h[90] / H_AVG[90]
            - 0.01584137 * h[81] / H_AVG[81]
            + 0.01466491 * h[68] / H_AVG[68]
            + 0.01261565 * h[104] / H_AVG[104]
            + 0.01184149 * h[25] / H_AVG[25]
            - 0.01107565 * h[123] / H_AVG[123]
            - 0.004738654 * h[16] / H_AVG[16]
            - 0.0003140601 * h[0] / H_AVG[0]
            + 6.507362e-05 * h[21] / H_AVG[21]
            - 1.855343e-05 * h[99] / H_AVG[99]
            + 8.668812e-07 * h[84] / H_AVG[84]
            + 2.045293e-07 * h[55] / H_AVG[55]
            + 1.371885e-07 * h[79] / H_AVG[79]
            - 1.309899e-07 * h[105] / H_AVG[105]
            - 1.063587e-07 * h[106] / H_AVG[106]
            + 8.644913e-08 * h[113] / H_AVG[113]
            - 8.207959e-08 * h[34] / H_AVG[34]
            + 7.367315e-08 * h[73] / H_AVG[73]
            + 6.840826e-08 * h[10] / H_AVG[10]
            + 6.526934e-08 * h[11] / H_AVG[11]
            - 5.55099e-08 * h[6] / H_AVG[6]
            + 5.298387e-08 * h[86] / H_AVG[86]
            - 5.039762e-08 * h[32] / H_AVG[32]
            + 4.634935e-08 * h[17] / H_AVG[17]
            + 4.18564e-08 * h[38] / H_AVG[38]
            - 3.436891e-08 * h[42] / H_AVG[42]
            - 3.066521e-08 * h[54] / H_AVG[54]
            - 2.911852e-08 * h[29] / H_AVG[29]
            + 2.641579e-08 * h[118] / H_AVG[118]
            + 2.059795e-08 * h[48] / H_AVG[48]
            - 1.967161e-08 * h[45] / H_AVG[45]
            - 1.898346e-08 * h[39] / H_AVG[39]
            + 1.75198e-08 * h[27] / H_AVG[27]
            + 1.744239e-08 * h[63] / H_AVG[63]
            + 1.725654e-08 * h[30] / H_AVG[30]
            + 1.688459e-08 * h[43] / H_AVG[43]
            + 1.571704e-08 * h[121] / H_AVG[121]
            - 1.543893e-08 * h[80] / H_AVG[80]
            - 1.415077e-08 * h[126] / H_AVG[126]
            + 1.363972e-08 * h[60] / H_AVG[60]
            - 1.342765e-08 * h[61] / H_AVG[61]
            - 1.275227e-08 * h[102] / H_AVG[102]
            - 1.271418e-08 * h[107] / H_AVG[107]
            - 1.254356e-08 * h[7] / H_AVG[7]
            + 1.128948e-08 * h[92] / H_AVG[92]
            + 1.103844e-08 * h[35] / H_AVG[35]
            + 1.074102e-08 * h[15] / H_AVG[15]
            - 1.038434e-08 * h[59] / H_AVG[59]
            - 1.00165e-08 * h[101] / H_AVG[101]
            - 9.550127e-09 * h[28] / H_AVG[28]
            + 9.283355e-09 * h[103] / H_AVG[103]
            + 8.743774e-09 * h[69] / H_AVG[69]
            + 8.490943e-09 * h[67] / H_AVG[67]
            + 8.206514e-09 * h[1] / H_AVG[1]
            + 8.144094e-09 * h[100] / H_AVG[100]
            - 7.479793e-09 * h[110] / H_AVG[110]
            - 7.247337e-09 * h[3] / H_AVG[3]
            + 6.957318e-09 * h[37] / H_AVG[37]
            - 6.806867e-09 * h[50] / H_AVG[50]
            + 6.404991e-09 * h[36] / H_AVG[36]
            + 6.401773e-09 * h[44] / H_AVG[44]
            - 6.298611e-09 * h[12] / H_AVG[12]
            - 5.985338e-09 * h[114] / H_AVG[114]
            - 5.941037e-09 * h[2] / H_AVG[2]
            + 5.726538e-09 * h[58] / H_AVG[58]
            - 5.708484e-09 * h[112] / H_AVG[112]
            + 5.580509e-09 * h[76] / H_AVG[76]
            - 5.367715e-09 * h[13] / H_AVG[13]
            - 5.222573e-09 * h[19] / H_AVG[19]
            - 5.11756e-09 * h[111] / H_AVG[111]
            + 4.880237e-09 * h[74] / H_AVG[74]
            + 4.623465e-09 * h[108] / H_AVG[108]
            + 4.352532e-09 * h[5] / H_AVG[5]
            - 4.167419e-09 * h[8] / H_AVG[8]
            - 3.999973e-09 * h[91] / H_AVG[91]
            - 3.723677e-09 * h[4] / H_AVG[4]
            - 3.566019e-09 * h[117] / H_AVG[117]
            + 3.013108e-09 * h[53] / H_AVG[53]
            - 2.977356e-09 * h[66] / H_AVG[66]
            - 2.703771e-09 * h[33] / H_AVG[33]
            - 2.640371e-09 * h[65] / H_AVG[65]
            + 2.440437e-09 * h[82] / H_AVG[82]
            - 2.281392e-09 * h[31] / H_AVG[31]
            - 2.101036e-09 * h[71] / H_AVG[71]
            + 1.952404e-09 * h[94] / H_AVG[94]
            + 1.886267e-09 * h[9] / H_AVG[9]
            + 1.665061e-09 * h[26] / H_AVG[26]
            - 1.487257e-09 * h[14] / H_AVG[14]
            + 1.477501e-09 * h[57] / H_AVG[57]
            + 1.389889e-09 * h[49] / H_AVG[49]
            + 1.36007e-09 * h[77] / H_AVG[77]
            - 1.356382e-09 * h[22] / H_AVG[22]
            + 1.262136e-09 * h[122] / H_AVG[122]
            + 1.016421e-09 * h[72] / H_AVG[72]
            + 9.783813e-10 * h[51] / H_AVG[51]
            - 9.607978e-10 * h[23] / H_AVG[23]
            - 6.589205e-10 * h[89] / H_AVG[89]
            + 6.461767e-10 * h[125] / H_AVG[125]
            - 6.064464e-10 * h[75] / H_AVG[75]
            + 5.243008e-10 * h[93] / H_AVG[93]
            - 5.23322e-10 * h[87] / H_AVG[87]
            - 4.711013e-10 * h[40] / H_AVG[40]
            - 4.380514e-10 * h[20] / H_AVG[20]
            + 3.925749e-10 * h[119] / H_AVG[119]
            - 3.571113e-10 * h[62] / H_AVG[62]
            - 3.217709e-10 * h[95] / H_AVG[95]
            - 2.975795e-10 * h[96] / H_AVG[96]
            + 1.756145e-10 * h[116] / H_AVG[116]
            + 1.685823e-10 * h[56] / H_AVG[56]
            + 1.175565e-10 * h[85] / H_AVG[85]
            + 6.928279e-11 * h[109] / H_AVG[109]
            - 6.033381e-11 * h[98] / H_AVG[98]
            - 4.803061e-11 * h[88] / H_AVG[88]
            - 3.278146e-11 * h[46] / H_AVG[46]
            - 3.067342e-11 * h[47] / H_AVG[47]
            - 1.467387e-11 * h[64] / H_AVG[64]
            + 7.637934e-12 * h[127] / H_AVG[127]
            + 6.795576e-13 * h[41] / H_AVG[41]
        ),
        0.3035426 + T[6] * (   # class Zqq
            + 0.1139309 * h[115] / H_AVG[115]
            - 0.1084849 * h[124] / H_AVG[124]
            + 0.1015203 * h[24] / H_AVG[24]
            + 0.09862286 * h[83] / H_AVG[83]
            - 0.09556395 * h[97] / H_AVG[97]
            + 0.09379847 * h[104] / H_AVG[104]
            - 0.08700132 * h[120] / H_AVG[120]
            + 0.07543917 * h[68] / H_AVG[68]
            + 0.05983497 * h[81] / H_AVG[81]
            - 0.05489068 * h[52] / H_AVG[52]
            - 0.04180572 * h[16] / H_AVG[16]
            - 0.02091125 * h[70] / H_AVG[70]
            + 0.01800421 * h[18] / H_AVG[18]
            + 0.0094161 * h[25] / H_AVG[25]
            - 0.008186777 * h[90] / H_AVG[90]
            - 0.007455309 * h[78] / H_AVG[78]
            + 0.004992108 * h[123] / H_AVG[123]
            - 5.928372e-05 * h[0] / H_AVG[0]
            - 4.196433e-05 * h[99] / H_AVG[99]
            - 3.355214e-05 * h[21] / H_AVG[21]
            + 1.976199e-06 * h[84] / H_AVG[84]
            + 4.664927e-07 * h[55] / H_AVG[55]
            + 3.129492e-07 * h[79] / H_AVG[79]
            - 2.987987e-07 * h[105] / H_AVG[105]
            - 2.425382e-07 * h[106] / H_AVG[106]
            + 1.970366e-07 * h[113] / H_AVG[113]
            - 1.871865e-07 * h[34] / H_AVG[34]
            + 1.680579e-07 * h[73] / H_AVG[73]
            + 1.56133e-07 * h[10] / H_AVG[10]
            + 1.487586e-07 * h[11] / H_AVG[11]
            - 1.267082e-07 * h[6] / H_AVG[6]
            + 1.208716e-07 * h[86] / H_AVG[86]
            - 1.149541e-07 * h[32] / H_AVG[32]
            + 1.05657e-07 * h[17] / H_AVG[17]
            + 9.550669e-08 * h[38] / H_AVG[38]
            - 7.841e-08 * h[42] / H_AVG[42]
            - 6.988509e-08 * h[54] / H_AVG[54]
            - 6.643402e-08 * h[29] / H_AVG[29]
            + 6.023067e-08 * h[118] / H_AVG[118]
            + 4.700879e-08 * h[48] / H_AVG[48]
            - 4.48604e-08 * h[45] / H_AVG[45]
            - 4.333636e-08 * h[39] / H_AVG[39]
            + 3.995491e-08 * h[27] / H_AVG[27]
            + 3.978017e-08 * h[63] / H_AVG[63]
            + 3.934857e-08 * h[30] / H_AVG[30]
            + 3.853758e-08 * h[43] / H_AVG[43]
            + 3.586396e-08 * h[121] / H_AVG[121]
            - 3.521464e-08 * h[80] / H_AVG[80]
            - 3.221316e-08 * h[126] / H_AVG[126]
            + 3.112053e-08 * h[60] / H_AVG[60]
            - 3.065278e-08 * h[61] / H_AVG[61]
            - 2.910252e-08 * h[102] / H_AVG[102]
            - 2.903227e-08 * h[107] / H_AVG[107]
            - 2.862035e-08 * h[7] / H_AVG[7]
            + 2.576791e-08 * h[92] / H_AVG[92]
            + 2.518113e-08 * h[35] / H_AVG[35]
            + 2.450151e-08 * h[15] / H_AVG[15]
            - 2.367103e-08 * h[59] / H_AVG[59]
            - 2.282952e-08 * h[101] / H_AVG[101]
            - 2.183612e-08 * h[28] / H_AVG[28]
            + 2.116505e-08 * h[103] / H_AVG[103]
            + 1.992083e-08 * h[69] / H_AVG[69]
            + 1.933146e-08 * h[67] / H_AVG[67]
            + 1.872971e-08 * h[1] / H_AVG[1]
            + 1.859122e-08 * h[100] / H_AVG[100]
            - 1.707565e-08 * h[110] / H_AVG[110]
            - 1.651496e-08 * h[3] / H_AVG[3]
            + 1.586661e-08 * h[37] / H_AVG[37]
            - 1.553698e-08 * h[50] / H_AVG[50]
            + 1.461226e-08 * h[44] / H_AVG[44]
            + 1.460009e-08 * h[36] / H_AVG[36]
            - 1.435482e-08 * h[12] / H_AVG[12]
            - 1.365027e-08 * h[114] / H_AVG[114]
            - 1.352692e-08 * h[2] / H_AVG[2]
            + 1.304875e-08 * h[58] / H_AVG[58]
            - 1.299608e-08 * h[112] / H_AVG[112]
            + 1.271757e-08 * h[76] / H_AVG[76]
            - 1.225767e-08 * h[13] / H_AVG[13]
            - 1.191122e-08 * h[19] / H_AVG[19]
            - 1.168306e-08 * h[111] / H_AVG[111]
            + 1.113418e-08 * h[74] / H_AVG[74]
            + 1.056737e-08 * h[108] / H_AVG[108]
            + 9.937586e-09 * h[5] / H_AVG[5]
            - 9.496641e-09 * h[8] / H_AVG[8]
            - 9.107137e-09 * h[91] / H_AVG[91]
            - 8.487677e-09 * h[4] / H_AVG[4]
            - 8.12975e-09 * h[117] / H_AVG[117]
            + 6.867817e-09 * h[53] / H_AVG[53]
            - 6.789377e-09 * h[66] / H_AVG[66]
            - 6.182814e-09 * h[33] / H_AVG[33]
            - 6.004189e-09 * h[65] / H_AVG[65]
            + 5.555802e-09 * h[82] / H_AVG[82]
            - 5.219147e-09 * h[31] / H_AVG[31]
            - 4.795907e-09 * h[71] / H_AVG[71]
            + 4.45487e-09 * h[94] / H_AVG[94]
            + 4.302941e-09 * h[9] / H_AVG[9]
            + 3.780098e-09 * h[26] / H_AVG[26]
            - 3.389606e-09 * h[14] / H_AVG[14]
            + 3.357594e-09 * h[57] / H_AVG[57]
            + 3.17353e-09 * h[49] / H_AVG[49]
            + 3.103594e-09 * h[77] / H_AVG[77]
            - 3.094844e-09 * h[22] / H_AVG[22]
            + 2.870192e-09 * h[122] / H_AVG[122]
            + 2.320935e-09 * h[72] / H_AVG[72]
            + 2.222259e-09 * h[51] / H_AVG[51]
            - 2.188921e-09 * h[23] / H_AVG[23]
            - 1.511475e-09 * h[89] / H_AVG[89]
            + 1.476628e-09 * h[125] / H_AVG[125]
            - 1.386193e-09 * h[75] / H_AVG[75]
            - 1.199586e-09 * h[87] / H_AVG[87]
            + 1.195512e-09 * h[93] / H_AVG[93]
            - 1.075069e-09 * h[40] / H_AVG[40]
            - 9.993402e-10 * h[20] / H_AVG[20]
            + 8.925993e-10 * h[119] / H_AVG[119]
            - 8.16672e-10 * h[62] / H_AVG[62]
            - 7.341479e-10 * h[95] / H_AVG[95]
            - 6.804194e-10 * h[96] / H_AVG[96]
            + 4.012694e-10 * h[116] / H_AVG[116]
            + 3.829567e-10 * h[56] / H_AVG[56]
            + 2.662146e-10 * h[85] / H_AVG[85]
            + 1.585882e-10 * h[109] / H_AVG[109]
            - 1.380474e-10 * h[98] / H_AVG[98]
            - 1.103343e-10 * h[88] / H_AVG[88]
            - 7.429772e-11 * h[46] / H_AVG[46]
            - 7.014288e-11 * h[47] / H_AVG[47]
            - 3.989103e-11 * h[64] / H_AVG[64]
            + 1.680962e-11 * h[127] / H_AVG[127]
            + 1.4862e-12 * h[41] / H_AVG[41]
        ),
        0.09887857 + T[7] * (   # class Wqq
            + 0.2646216 * h[97] / H_AVG[97]
            + 0.1128681 * h[83] / H_AVG[83]
            + 0.09768475 * h[24] / H_AVG[24]
            + 0.09449594 * h[68] / H_AVG[68]
            - 0.09015239 * h[52] / H_AVG[52]
            - 0.07668019 * h[120] / H_AVG[120]
            + 0.05528625 * h[115] / H_AVG[115]
            + 0.04846946 * h[104] / H_AVG[104]
            - 0.03949263 * h[81] / H_AVG[81]
            - 0.03929539 * h[16] / H_AVG[16]
            - 0.03105545 * h[78] / H_AVG[78]
            - 0.01683826 * h[70] / H_AVG[70]
            + 0.0128657 * h[124] / H_AVG[124]
            - 0.01057254 * h[25] / H_AVG[25]
            - 0.006009887 * h[90] / H_AVG[90]
            + 0.001755643 * h[18] / H_AVG[18]
            + 0.001618585 * h[123] / H_AVG[123]
            - 0.0001677958 * h[0] / H_AVG[0]
            - 3.770782e-05 * h[99] / H_AVG[99]
            - 2.624619e-05 * h[21] / H_AVG[21]
            + 1.773473e-06 * h[84] / H_AVG[84]
            + 4.185095e-07 * h[55] / H_AVG[55]
            + 2.808335e-07 * h[79] / H_AVG[79]
            - 2.68129e-07 * h[105] / H_AVG[105]
            - 2.176261e-07 * h[106] / H_AVG[106]
            + 1.76838e-07 * h[113] / H_AVG[113]
            - 1.679677e-07 * h[34] / H_AVG[34]
            + 1.5082e-07 * h[73] / H_AVG[73]
            + 1.400585e-07 * h[10] / H_AVG[10]
            + 1.334883e-07 * h[11] / H_AVG[11]
            - 1.137054e-07 * h[6] / H_AVG[6]
            + 1.08477e-07 * h[86] / H_AVG[86]
            - 1.031519e-07 * h[32] / H_AVG[32]
            + 9.479949e-08 * h[17] / H_AVG[17]
            + 8.570484e-08 * h[38] / H_AVG[38]
            - 7.036799e-08 * h[42] / H_AVG[42]
            - 6.272078e-08 * h[54] / H_AVG[54]
            - 5.960928e-08 * h[29] / H_AVG[29]
            + 5.40461e-08 * h[118] / H_AVG[118]
            + 4.218529e-08 * h[48] / H_AVG[48]
            - 4.025711e-08 * h[45] / H_AVG[45]
            - 3.887768e-08 * h[39] / H_AVG[39]
            + 3.585469e-08 * h[27] / H_AVG[27]
            + 3.569435e-08 * h[63] / H_AVG[63]
            + 3.531198e-08 * h[30] / H_AVG[30]
            + 3.458017e-08 * h[43] / H_AVG[43]
            + 3.21792e-08 * h[121] / H_AVG[121]
            - 3.159699e-08 * h[80] / H_AVG[80]
            - 2.892089e-08 * h[126] / H_AVG[126]
            + 2.791838e-08 * h[60] / H_AVG[60]
            - 2.749805e-08 * h[61] / H_AVG[61]
            - 2.611385e-08 * h[102] / H_AVG[102]
            - 2.605404e-08 * h[107] / H_AVG[107]
            - 2.567083e-08 * h[7] / H_AVG[7]
            + 2.312097e-08 * h[92] / H_AVG[92]
            + 2.259959e-08 * h[35] / H_AVG[35]
            + 2.198635e-08 * h[15] / H_AVG[15]
            - 2.124903e-08 * h[59] / H_AVG[59]
            - 2.048425e-08 * h[101] / H_AVG[101]
            - 1.958254e-08 * h[28] / H_AVG[28]
            + 1.899387e-08 * h[103] / H_AVG[103]
            + 1.787395e-08 * h[69] / H_AVG[69]
            + 1.734436e-08 * h[67] / H_AVG[67]
            + 1.680449e-08 * h[1] / H_AVG[1]
            + 1.668122e-08 * h[100] / H_AVG[100]
            - 1.531812e-08 * h[110] / H_AVG[110]
            - 1.48231e-08 * h[3] / H_AVG[3]
            + 1.42374e-08 * h[37] / H_AVG[37]
            - 1.394398e-08 * h[50] / H_AVG[50]
            + 1.311116e-08 * h[44] / H_AVG[44]
            + 1.310177e-08 * h[36] / H_AVG[36]
            - 1.287593e-08 * h[12] / H_AVG[12]
            - 1.224756e-08 * h[114] / H_AVG[114]
            - 1.214673e-08 * h[2] / H_AVG[2]
            + 1.170912e-08 * h[58] / H_AVG[58]
            - 1.166278e-08 * h[112] / H_AVG[112]
            + 1.141176e-08 * h[76] / H_AVG[76]
            - 1.0998e-08 * h[13] / H_AVG[13]
            - 1.068809e-08 * h[19] / H_AVG[19]
            - 1.048268e-08 * h[111] / H_AVG[111]
            + 9.990549e-09 * h[74] / H_AVG[74]
            + 9.483245e-09 * h[108] / H_AVG[108]
            + 8.917066e-09 * h[5] / H_AVG[5]
            - 8.521437e-09 * h[8] / H_AVG[8]
            - 8.169488e-09 * h[91] / H_AVG[91]
            - 7.616093e-09 * h[4] / H_AVG[4]
            - 7.295132e-09 * h[117] / H_AVG[117]
            + 6.161496e-09 * h[53] / H_AVG[53]
            - 6.097085e-09 * h[66] / H_AVG[66]
            - 5.548513e-09 * h[33] / H_AVG[33]
            - 5.389142e-09 * h[65] / H_AVG[65]
            + 4.984404e-09 * h[82] / H_AVG[82]
            - 4.685121e-09 * h[31] / H_AVG[31]
            - 4.304476e-09 * h[71] / H_AVG[71]
            + 3.995789e-09 * h[94] / H_AVG[94]
            + 3.861151e-09 * h[9] / H_AVG[9]
            + 3.390312e-09 * h[26] / H_AVG[26]
            - 3.042114e-09 * h[14] / H_AVG[14]
            + 3.013011e-09 * h[57] / H_AVG[57]
            + 2.84773e-09 * h[49] / H_AVG[49]
            + 2.782677e-09 * h[77] / H_AVG[77]
            - 2.777525e-09 * h[22] / H_AVG[22]
            + 2.575555e-09 * h[122] / H_AVG[122]
            + 2.082618e-09 * h[72] / H_AVG[72]
            + 1.995001e-09 * h[51] / H_AVG[51]
            - 1.964908e-09 * h[23] / H_AVG[23]
            - 1.35633e-09 * h[89] / H_AVG[89]
            + 1.324847e-09 * h[125] / H_AVG[125]
            - 1.244026e-09 * h[75] / H_AVG[75]
            - 1.075839e-09 * h[87] / H_AVG[87]
            + 1.071933e-09 * h[93] / H_AVG[93]
            - 9.646457e-10 * h[40] / H_AVG[40]
            - 8.967489e-10 * h[20] / H_AVG[20]
            + 8.00233e-10 * h[119] / H_AVG[119]
            - 7.327907e-10 * h[62] / H_AVG[62]
            - 6.587e-10 * h[95] / H_AVG[95]
            - 6.10614e-10 * h[96] / H_AVG[96]
            + 3.600047e-10 * h[116] / H_AVG[116]
            + 3.436526e-10 * h[56] / H_AVG[56]
            + 2.389588e-10 * h[85] / H_AVG[85]
            + 1.422647e-10 * h[109] / H_AVG[109]
            - 1.238598e-10 * h[98] / H_AVG[98]
            - 9.895684e-11 * h[88] / H_AVG[88]
            - 6.664909e-11 * h[46] / H_AVG[46]
            - 6.294186e-11 * h[47] / H_AVG[47]
            - 3.716774e-11 * h[64] / H_AVG[64]
            + 1.507666e-11 * h[127] / H_AVG[127]
            + 1.335075e-12 * h[41] / H_AVG[41]
        ),
        -0.4320669 + T[8] * (   # class Tbqq
            - 0.175332 * h[104] / H_AVG[104]
            + 0.1398897 * h[16] / H_AVG[16]
            - 0.1246414 * h[83] / H_AVG[83]
            + 0.1243932 * h[120] / H_AVG[120]
            + 0.1175807 * h[68] / H_AVG[68]
            - 0.08531531 * h[18] / H_AVG[18]
            + 0.04282785 * h[78] / H_AVG[78]
            - 0.03480679 * h[24] / H_AVG[24]
            + 0.028852 * h[81] / H_AVG[81]
            + 0.02879006 * h[97] / H_AVG[97]
            - 0.02772158 * h[52] / H_AVG[52]
            - 0.02641407 * h[115] / H_AVG[115]
            - 0.01817493 * h[70] / H_AVG[70]
            + 0.008702698 * h[124] / H_AVG[124]
            - 0.007802835 * h[123] / H_AVG[123]
            - 0.007678292 * h[90] / H_AVG[90]
            + 0.0009166075 * h[25] / H_AVG[25]
            - 0.0001224681 * h[0] / H_AVG[0]
            - 2.335365e-05 * h[99] / H_AVG[99]
            - 1.063749e-05 * h[21] / H_AVG[21]
            + 1.106499e-06 * h[84] / H_AVG[84]
            + 2.611764e-07 * h[55] / H_AVG[55]
            + 1.752889e-07 * h[79] / H_AVG[79]
            - 1.672669e-07 * h[105] / H_AVG[105]
            - 1.357559e-07 * h[106] / H_AVG[106]
            + 1.103395e-07 * h[113] / H_AVG[113]
            - 1.047934e-07 * h[34] / H_AVG[34]
            + 9.407438e-08 * h[73] / H_AVG[73]
            + 8.739151e-08 * h[10] / H_AVG[10]
            + 8.329521e-08 * h[11] / H_AVG[11]
            - 7.092979e-08 * h[6] / H_AVG[6]
            + 6.767317e-08 * h[86] / H_AVG[86]
            - 6.435838e-08 * h[32] / H_AVG[32]
            + 5.917427e-08 * h[17] / H_AVG[17]
            + 5.347429e-08 * h[38] / H_AVG[38]
            - 4.39049e-08 * h[42] / H_AVG[42]
            - 3.912845e-08 * h[54] / H_AVG[54]
            - 3.719389e-08 * h[29] / H_AVG[29]
            + 3.371819e-08 * h[118] / H_AVG[118]
            + 2.632104e-08 * h[48] / H_AVG[48]
            - 2.511058e-08 * h[45] / H_AVG[45]
            - 2.425484e-08 * h[39] / H_AVG[39]
            + 2.236984e-08 * h[27] / H_AVG[27]
            + 2.227259e-08 * h[63] / H_AVG[63]
            + 2.203111e-08 * h[30] / H_AVG[30]
            + 2.157197e-08 * h[43] / H_AVG[43]
            + 2.008012e-08 * h[121] / H_AVG[121]
            - 1.97158e-08 * h[80] / H_AVG[80]
            - 1.80376e-08 * h[126] / H_AVG[126]
            + 1.742396e-08 * h[60] / H_AVG[60]
            - 1.715645e-08 * h[61] / H_AVG[61]
            - 1.628825e-08 * h[102] / H_AVG[102]
            - 1.625418e-08 * h[107] / H_AVG[107]
            - 1.601777e-08 * h[7] / H_AVG[7]
            + 1.442323e-08 * h[92] / H_AVG[92]
            + 1.409105e-08 * h[35] / H_AVG[35]
            + 1.371753e-08 * h[15] / H_AVG[15]
            - 1.324917e-08 * h[59] / H_AVG[59]
            - 1.278112e-08 * h[101] / H_AVG[101]
            - 1.222749e-08 * h[28] / H_AVG[28]
            + 1.184663e-08 * h[103] / H_AVG[103]
            + 1.115009e-08 * h[69] / H_AVG[69]
            + 1.082265e-08 * h[67] / H_AVG[67]
            + 1.048678e-08 * h[1] / H_AVG[1]
            + 1.040809e-08 * h[100] / H_AVG[100]
            - 9.560204e-09 * h[110] / H_AVG[110]
            - 9.24901e-09 * h[3] / H_AVG[3]
            + 8.883104e-09 * h[37] / H_AVG[37]
            - 8.695868e-09 * h[50] / H_AVG[50]
            + 8.180479e-09 * h[44] / H_AVG[44]
            + 8.172935e-09 * h[36] / H_AVG[36]
            - 8.03555e-09 * h[12] / H_AVG[12]
            - 7.641978e-09 * h[114] / H_AVG[114]
            - 7.577823e-09 * h[2] / H_AVG[2]
            + 7.305909e-09 * h[58] / H_AVG[58]
            - 7.275764e-09 * h[112] / H_AVG[112]
            + 7.120249e-09 * h[76] / H_AVG[76]
            - 6.861267e-09 * h[13] / H_AVG[13]
            - 6.667511e-09 * h[19] / H_AVG[19]
            - 6.541094e-09 * h[111] / H_AVG[111]
            + 6.234292e-09 * h[74] / H_AVG[74]
            + 5.91638e-09 * h[108] / H_AVG[108]
            + 5.562501e-09 * h[5] / H_AVG[5]
            - 5.316385e-09 * h[8] / H_AVG[8]
            - 5.096461e-09 * h[91] / H_AVG[91]
            - 4.752125e-09 * h[4] / H_AVG[4]
            - 4.550769e-09 * h[117] / H_AVG[117]
            + 3.844927e-09 * h[53] / H_AVG[53]
            - 3.800706e-09 * h[66] / H_AVG[66]
            - 3.461702e-09 * h[33] / H_AVG[33]
            - 3.362657e-09 * h[65] / H_AVG[65]
            + 3.110048e-09 * h[82] / H_AVG[82]
            - 2.920939e-09 * h[31] / H_AVG[31]
            - 2.684706e-09 * h[71] / H_AVG[71]
            + 2.495102e-09 * h[94] / H_AVG[94]
            + 2.408936e-09 * h[9] / H_AVG[9]
            + 2.117462e-09 * h[26] / H_AVG[26]
            - 1.897552e-09 * h[14] / H_AVG[14]
            + 1.880657e-09 * h[57] / H_AVG[57]
            + 1.776764e-09 * h[49] / H_AVG[49]
            + 1.737547e-09 * h[77] / H_AVG[77]
            - 1.73262e-09 * h[22] / H_AVG[22]
            + 1.606866e-09 * h[122] / H_AVG[122]
            + 1.299331e-09 * h[72] / H_AVG[72]
            + 1.244539e-09 * h[51] / H_AVG[51]
            - 1.223663e-09 * h[23] / H_AVG[23]
            - 8.460948e-10 * h[89] / H_AVG[89]
            + 8.266628e-10 * h[125] / H_AVG[125]
            - 7.759186e-10 * h[75] / H_AVG[75]
            - 6.71406e-10 * h[87] / H_AVG[87]
            + 6.691355e-10 * h[93] / H_AVG[93]
            - 6.01884e-10 * h[40] / H_AVG[40]
            - 5.594586e-10 * h[20] / H_AVG[20]
            + 4.997523e-10 * h[119] / H_AVG[119]
            - 4.571199e-10 * h[62] / H_AVG[62]
            - 4.109101e-10 * h[95] / H_AVG[95]
            - 3.808038e-10 * h[96] / H_AVG[96]
            + 2.246164e-10 * h[116] / H_AVG[116]
            + 2.144513e-10 * h[56] / H_AVG[56]
            + 1.491478e-10 * h[85] / H_AVG[85]
            + 8.874448e-11 * h[109] / H_AVG[109]
            - 7.729682e-11 * h[98] / H_AVG[98]
            - 6.166559e-11 * h[88] / H_AVG[88]
            - 4.159923e-11 * h[46] / H_AVG[46]
            - 3.926349e-11 * h[47] / H_AVG[47]
            - 2.149038e-11 * h[64] / H_AVG[64]
            + 9.428886e-12 * h[127] / H_AVG[127]
            + 8.337021e-13 * h[41] / H_AVG[41]
        ),
        -1.495433 + T[9] * (   # class Tbl
            + 0.2230297 * h[16] / H_AVG[16]
            + 0.1116695 * h[18] / H_AVG[18]
            + 0.1071022 * h[24] / H_AVG[24]
            - 0.09080959 * h[104] / H_AVG[104]
            - 0.07541022 * h[115] / H_AVG[115]
            + 0.07316556 * h[97] / H_AVG[97]
            - 0.06985658 * h[124] / H_AVG[124]
            + 0.06543168 * h[52] / H_AVG[52]
            - 0.05418773 * h[68] / H_AVG[68]
            + 0.03768688 * h[78] / H_AVG[78]
            + 0.01823071 * h[70] / H_AVG[70]
            + 0.0175446 * h[123] / H_AVG[123]
            + 0.01713866 * h[120] / H_AVG[120]
            + 0.01250151 * h[25] / H_AVG[25]
            - 0.01052353 * h[83] / H_AVG[83]
            + 0.01005002 * h[90] / H_AVG[90]
            - 0.005270108 * h[81] / H_AVG[81]
            - 0.000319901 * h[0] / H_AVG[0]
            + 5.325423e-05 * h[21] / H_AVG[21]
            - 1.566243e-05 * h[99] / H_AVG[99]
            + 7.479205e-07 * h[84] / H_AVG[84]
            + 1.764045e-07 * h[55] / H_AVG[55]
            + 1.183469e-07 * h[79] / H_AVG[79]
            - 1.130941e-07 * h[105] / H_AVG[105]
            - 9.175963e-08 * h[106] / H_AVG[106]
            + 7.461518e-08 * h[113] / H_AVG[113]
            - 7.082181e-08 * h[34] / H_AVG[34]
            + 6.357779e-08 * h[73] / H_AVG[73]
            + 5.902181e-08 * h[10] / H_AVG[10]
            + 5.631072e-08 * h[11] / H_AVG[11]
            - 4.789709e-08 * h[6] / H_AVG[6]
            + 4.572772e-08 * h[86] / H_AVG[86]
            - 4.350998e-08 * h[32] / H_AVG[32]
            + 3.998699e-08 * h[17] / H_AVG[17]
            + 3.611097e-08 * h[38] / H_AVG[38]
            - 2.967026e-08 * h[42] / H_AVG[42]
            - 2.645918e-08 * h[54] / H_AVG[54]
            - 2.512714e-08 * h[29] / H_AVG[29]
            + 2.278951e-08 * h[118] / H_AVG[118]
            + 1.777202e-08 * h[48] / H_AVG[48]
            - 1.697236e-08 * h[45] / H_AVG[45]
            - 1.638049e-08 * h[39] / H_AVG[39]
            + 1.511873e-08 * h[27] / H_AVG[27]
            + 1.504835e-08 * h[63] / H_AVG[63]
            + 1.489978e-08 * h[30] / H_AVG[30]
            + 1.456963e-08 * h[43] / H_AVG[43]
            + 1.355952e-08 * h[121] / H_AVG[121]
            - 1.331895e-08 * h[80] / H_AVG[80]
            - 1.220371e-08 * h[126] / H_AVG[126]
            + 1.176883e-08 * h[60] / H_AVG[60]
            - 1.158345e-08 * h[61] / H_AVG[61]
            - 1.100582e-08 * h[102] / H_AVG[102]
            - 1.096956e-08 * h[107] / H_AVG[107]
            - 1.082248e-08 * h[7] / H_AVG[7]
            + 9.741226e-09 * h[92] / H_AVG[92]
            + 9.528119e-09 * h[35] / H_AVG[35]
            + 9.263665e-09 * h[15] / H_AVG[15]
            - 8.958127e-09 * h[59] / H_AVG[59]
            - 8.643299e-09 * h[101] / H_AVG[101]
            - 8.24196e-09 * h[28] / H_AVG[28]
            + 8.009894e-09 * h[103] / H_AVG[103]
            + 7.547484e-09 * h[69] / H_AVG[69]
            + 7.326215e-09 * h[67] / H_AVG[67]
            + 7.07807e-09 * h[1] / H_AVG[1]
            + 7.024061e-09 * h[100] / H_AVG[100]
            - 6.454056e-09 * h[110] / H_AVG[110]
            - 6.257774e-09 * h[3] / H_AVG[3]
            + 6.004391e-09 * h[37] / H_AVG[37]
            - 5.871089e-09 * h[50] / H_AVG[50]
            + 5.525714e-09 * h[36] / H_AVG[36]
            + 5.52441e-09 * h[44] / H_AVG[44]
            - 5.435712e-09 * h[12] / H_AVG[12]
            - 5.16473e-09 * h[114] / H_AVG[114]
            - 5.128658e-09 * h[2] / H_AVG[2]
            + 4.941776e-09 * h[58] / H_AVG[58]
            - 4.926758e-09 * h[112] / H_AVG[112]
            + 4.816956e-09 * h[76] / H_AVG[76]
            - 4.630234e-09 * h[13] / H_AVG[13]
            - 4.505712e-09 * h[19] / H_AVG[19]
            - 4.414665e-09 * h[111] / H_AVG[111]
            + 4.211419e-09 * h[74] / H_AVG[74]
            + 3.988321e-09 * h[108] / H_AVG[108]
            + 3.755277e-09 * h[5] / H_AVG[5]
            - 3.595456e-09 * h[8] / H_AVG[8]
            - 3.451739e-09 * h[91] / H_AVG[91]
            - 3.214532e-09 * h[4] / H_AVG[4]
            - 3.078699e-09 * h[117] / H_AVG[117]
            + 2.597591e-09 * h[53] / H_AVG[53]
            - 2.569499e-09 * h[66] / H_AVG[66]
            - 2.332798e-09 * h[33] / H_AVG[33]
            - 2.278997e-09 * h[65] / H_AVG[65]
            + 2.106821e-09 * h[82] / H_AVG[82]
            - 1.970601e-09 * h[31] / H_AVG[31]
            - 1.812429e-09 * h[71] / H_AVG[71]
            + 1.684756e-09 * h[94] / H_AVG[94]
            + 1.627305e-09 * h[9] / H_AVG[9]
            + 1.437019e-09 * h[26] / H_AVG[26]
            - 1.282957e-09 * h[14] / H_AVG[14]
            + 1.275524e-09 * h[57] / H_AVG[57]
            + 1.199255e-09 * h[49] / H_AVG[49]
            + 1.171782e-09 * h[77] / H_AVG[77]
            - 1.169981e-09 * h[22] / H_AVG[22]
            + 1.089604e-09 * h[122] / H_AVG[122]
            + 8.768427e-10 * h[72] / H_AVG[72]
            + 8.447383e-10 * h[51] / H_AVG[51]
            - 8.299014e-10 * h[23] / H_AVG[23]
            - 5.681014e-10 * h[89] / H_AVG[89]
            + 5.57228e-10 * h[125] / H_AVG[125]
            - 5.232272e-10 * h[75] / H_AVG[75]
            + 4.523216e-10 * h[93] / H_AVG[93]
            - 4.513823e-10 * h[87] / H_AVG[87]
            - 4.065354e-10 * h[40] / H_AVG[40]
            - 3.780503e-10 * h[20] / H_AVG[20]
            + 3.387646e-10 * h[119] / H_AVG[119]
            - 3.080747e-10 * h[62] / H_AVG[62]
            - 2.775305e-10 * h[95] / H_AVG[95]
            - 2.566305e-10 * h[96] / H_AVG[96]
            + 1.514909e-10 * h[116] / H_AVG[116]
            + 1.454824e-10 * h[56] / H_AVG[56]
            + 1.01489e-10 * h[85] / H_AVG[85]
            + 5.974136e-11 * h[109] / H_AVG[109]
            - 5.203134e-11 * h[98] / H_AVG[98]
            - 4.139653e-11 * h[88] / H_AVG[88]
            - 2.830841e-11 * h[46] / H_AVG[46]
            - 2.646381e-11 * h[47] / H_AVG[47]
            - 1.38604e-11 * h[64] / H_AVG[64]
            + 6.621052e-12 * h[127] / H_AVG[127]
            + 5.89503e-13 * h[41] / H_AVG[41]
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
