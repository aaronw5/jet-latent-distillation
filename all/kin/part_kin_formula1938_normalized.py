"""Particle Transformer (ParT) jet tagger, JetClass, input features 'kin': tuned on the network's predictions (from 100 if-statements per neuron; the smallest without loss on the validation jets), with normalized weights (how much each one matters), as if-statements.

Input:  the particles of a jet (up to 128), hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
        energy: each particle's energy [GeV]; jet_pt, jet_eta, jet_energy: the jet's pT [GeV], pseudorapidity, energy [GeV].
Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the training jets, so share_k is the fraction of the neuron's
             average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron 54:  17.2%
  neuron 103:  11.0%
  neuron 27:   9.8%
  neuron 121:   9.1%
  neuron 33:   8.3%
  neuron 86:   8.2%
  neuron 51:   8.0%
  neuron 38:   6.5%
  neuron 22:   5.5%
  neuron 97:   4.1%
  neuron 104:   3.2%
  neuron  4:   2.4%
  neuron  9:   2.4%
  neuron 70:   1.5%
  neuron 81:   1.3%
  neuron 69:   0.6%
  neuron 102:   0.5%
  neuron 28:   0.3%
  neuron 30:   0.2%
  neuron 35:   0.1%
  neuron 57:   0.0%
  neuron 82:   0.0%
  neuron  5:   0.0%
  neuron 25:   0.0%
  neuron  2:   0.0%
  neuron 107:   0.0%
  neuron 18:   0.0%
  neuron 20:   0.0%
  neuron 24:   0.0%
  neuron 80:   0.0%
  neuron 50:   0.0%
  neuron 116:   0.0%
  neuron 63:   0.0%
  neuron 47:   0.0%
  neuron 96:   0.0%
  neuron 88:   0.0%
  neuron 125:   0.0%
  neuron 95:   0.0%
  neuron 37:   0.0%
  neuron 101:   0.0%
  neuron 58:   0.0%
  neuron 113:   0.0%
  neuron 45:   0.0%
  neuron 112:   0.0%
  neuron 94:   0.0%
  neuron 126:   0.0%
  neuron 99:   0.0%
  neuron 73:   0.0%
  neuron  0:   0.0%
  neuron  6:   0.0%
  neuron 49:   0.0%
  neuron 114:   0.0%
  neuron 77:   0.0%
  neuron 72:   0.0%
  neuron 29:   0.0%
  neuron 83:   0.0%
  neuron 12:   0.0%
  neuron 44:   0.0%
  neuron 19:   0.0%
  neuron 100:   0.0%
  neuron 65:   0.0%
  neuron 64:   0.0%
  neuron 106:   0.0%
  neuron 74:   0.0%
  neuron  7:   0.0%
  neuron 90:   0.0%
  neuron 66:   0.0%
  neuron 71:   0.0%
  neuron 17:   0.0%
  neuron 53:   0.0%
  neuron 23:   0.0%
  neuron 67:   0.0%
  neuron 55:   0.0%
  neuron 123:   0.0%
  neuron 109:   0.0%
  neuron 92:   0.0%
  neuron 15:   0.0%
  neuron 46:   0.0%
  neuron  8:   0.0%
  neuron 14:   0.0%
  neuron 59:   0.0%
  neuron 41:   0.0%
  neuron 39:   0.0%
  neuron 16:   0.0%
  neuron 110:   0.0%
  neuron 85:   0.0%
  neuron 119:   0.0%
  neuron 89:   0.0%
  neuron 60:   0.0%
  neuron 43:   0.0%
  neuron 105:   0.0%
  neuron 91:   0.0%
  neuron 13:   0.0%
  neuron 10:   0.0%
  neuron 48:   0.0%
  neuron 84:   0.0%
  neuron 75:   0.0%
  neuron 31:   0.0%
  neuron 62:   0.0%
  neuron 42:   0.0%
  neuron 11:   0.0%
  neuron 36:   0.0%
  neuron 122:   0.0%
  neuron 98:   0.0%
  neuron  3:   0.0%
  neuron 26:   0.0%
  neuron 40:   0.0%
  neuron 61:   0.0%
  neuron 68:   0.0%
  neuron 117:   0.0%
  neuron 120:   0.0%
  neuron 93:   0.0%
  neuron  1:   0.0%
  neuron 78:   0.0%
  neuron 111:   0.0%
  neuron 118:   0.0%
  neuron 124:   0.0%
  neuron 21:   0.0%
  neuron 115:   0.0%
  neuron 87:   0.0%
  neuron 108:   0.0%
  neuron 76:   0.0%
  neuron 127:   0.0%
  neuron 56:   0.0%
  neuron 32:   0.0%
  neuron 34:   0.0%
  neuron 79:   0.0%
  neuron 52:   0.0%
4. classify():    softmax of the logits; the class is the largest logit.

Whole test file (2,000,000 jets): accuracy 62.44% (the network: 74.52%); same class as the network for 71.91% of jets.

Quantities:
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b05                 e3/e2² with β = 0.5
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
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
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.N3_b05                 ₂e₄/(₁e₃)² with β = 0.5
  Q.N3_b2                  ₂e₄/(₁e₃)² with β = 2
  Q.centroid_offset        distance of the pT centroid from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr02                   ΔR between particles 0 and 2
  Q.dr12                   ΔR between particles 1 and 2
  Q.dr_0                   ΔR from the jet axis of particle 0 (ParT input; empty slot: 0)
  Q.dr_1                   ΔR from the jet axis of particle 1 (ParT input; empty slot: 0)
  Q.dr_10                  ΔR from the jet axis of particle 10 (ParT input; empty slot: 0)
  Q.dr_11                  ΔR from the jet axis of particle 11 (ParT input; empty slot: 0)
  Q.dr_12                  ΔR from the jet axis of particle 12 (ParT input; empty slot: 0)
  Q.dr_13                  ΔR from the jet axis of particle 13 (ParT input; empty slot: 0)
  Q.dr_14                  ΔR from the jet axis of particle 14 (ParT input; empty slot: 0)
  Q.dr_16                  ΔR from the jet axis of particle 16 (ParT input; empty slot: 0)
  Q.dr_17                  ΔR from the jet axis of particle 17 (ParT input; empty slot: 0)
  Q.dr_18                  ΔR from the jet axis of particle 18 (ParT input; empty slot: 0)
  Q.dr_19                  ΔR from the jet axis of particle 19 (ParT input; empty slot: 0)
  Q.dr_2                   ΔR from the jet axis of particle 2 (ParT input; empty slot: 0)
  Q.dr_21                  ΔR from the jet axis of particle 21 (ParT input; empty slot: 0)
  Q.dr_22                  ΔR from the jet axis of particle 22 (ParT input; empty slot: 0)
  Q.dr_23                  ΔR from the jet axis of particle 23 (ParT input; empty slot: 0)
  Q.dr_25                  ΔR from the jet axis of particle 25 (ParT input; empty slot: 0)
  Q.dr_3                   ΔR from the jet axis of particle 3 (ParT input; empty slot: 0)
  Q.dr_31                  ΔR from the jet axis of particle 31 (ParT input; empty slot: 0)
  Q.dr_32                  ΔR from the jet axis of particle 32 (ParT input; empty slot: 0)
  Q.dr_33                  ΔR from the jet axis of particle 33 (ParT input; empty slot: 0)
  Q.dr_34                  ΔR from the jet axis of particle 34 (ParT input; empty slot: 0)
  Q.dr_35                  ΔR from the jet axis of particle 35 (ParT input; empty slot: 0)
  Q.dr_37                  ΔR from the jet axis of particle 37 (ParT input; empty slot: 0)
  Q.dr_38                  ΔR from the jet axis of particle 38 (ParT input; empty slot: 0)
  Q.dr_4                   ΔR from the jet axis of particle 4 (ParT input; empty slot: 0)
  Q.dr_40                  ΔR from the jet axis of particle 40 (ParT input; empty slot: 0)
  Q.dr_41                  ΔR from the jet axis of particle 41 (ParT input; empty slot: 0)
  Q.dr_46                  ΔR from the jet axis of particle 46 (ParT input; empty slot: 0)
  Q.dr_47                  ΔR from the jet axis of particle 47 (ParT input; empty slot: 0)
  Q.dr_5                   ΔR from the jet axis of particle 5 (ParT input; empty slot: 0)
  Q.dr_53                  ΔR from the jet axis of particle 53 (ParT input; empty slot: 0)
  Q.dr_55                  ΔR from the jet axis of particle 55 (ParT input; empty slot: 0)
  Q.dr_57                  ΔR from the jet axis of particle 57 (ParT input; empty slot: 0)
  Q.dr_6                   ΔR from the jet axis of particle 6 (ParT input; empty slot: 0)
  Q.dr_7                   ΔR from the jet axis of particle 7 (ParT input; empty slot: 0)
  Q.dr_8                   ΔR from the jet axis of particle 8 (ParT input; empty slot: 0)
  Q.dr_9                   ΔR from the jet axis of particle 9 (ParT input; empty slot: 0)
  Q.dr_max_012             largest distance among the 3 hardest
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_b05                 energy correlation e2 with β = 0.5
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
  Q.ecf_g32                generalized energy correlation ₂e₃ (β=1; products of the smallest angles)
  Q.ecf_g41                generalized energy correlation ₁e₄ (β=1; products of the smallest angles)
  Q.ecf_g42                generalized energy correlation ₂e₄ (β=1; products of the smallest angles)
  Q.ecf_g43                generalized energy correlation ₃e₄ (β=1; products of the smallest angles)
  Q.eta_0                  Δη of particle 0 (ParT input; empty slot: 0)
  Q.eta_1                  Δη of particle 1 (ParT input; empty slot: 0)
  Q.eta_12                 Δη of particle 12 (ParT input; empty slot: 0)
  Q.eta_14                 Δη of particle 14 (ParT input; empty slot: 0)
  Q.eta_17                 Δη of particle 17 (ParT input; empty slot: 0)
  Q.eta_19                 Δη of particle 19 (ParT input; empty slot: 0)
  Q.eta_2                  Δη of particle 2 (ParT input; empty slot: 0)
  Q.eta_20                 Δη of particle 20 (ParT input; empty slot: 0)
  Q.eta_23                 Δη of particle 23 (ParT input; empty slot: 0)
  Q.eta_25                 Δη of particle 25 (ParT input; empty slot: 0)
  Q.eta_26                 Δη of particle 26 (ParT input; empty slot: 0)
  Q.eta_27                 Δη of particle 27 (ParT input; empty slot: 0)
  Q.eta_28                 Δη of particle 28 (ParT input; empty slot: 0)
  Q.eta_29                 Δη of particle 29 (ParT input; empty slot: 0)
  Q.eta_31                 Δη of particle 31 (ParT input; empty slot: 0)
  Q.eta_33                 Δη of particle 33 (ParT input; empty slot: 0)
  Q.eta_37                 Δη of particle 37 (ParT input; empty slot: 0)
  Q.eta_38                 Δη of particle 38 (ParT input; empty slot: 0)
  Q.eta_39                 Δη of particle 39 (ParT input; empty slot: 0)
  Q.eta_4                  Δη of particle 4 (ParT input; empty slot: 0)
  Q.eta_40                 Δη of particle 40 (ParT input; empty slot: 0)
  Q.eta_41                 Δη of particle 41 (ParT input; empty slot: 0)
  Q.eta_42                 Δη of particle 42 (ParT input; empty slot: 0)
  Q.eta_50                 Δη of particle 50 (ParT input; empty slot: 0)
  Q.eta_55                 Δη of particle 55 (ParT input; empty slot: 0)
  Q.eta_57                 Δη of particle 57 (ParT input; empty slot: 0)
  Q.eta_64                 Δη of particle 64 (ParT input; empty slot: 0)
  Q.eta_73                 Δη of particle 73 (ParT input; empty slot: 0)
  Q.eta_77                 Δη of particle 77 (ParT input; empty slot: 0)
  Q.eta_81                 Δη of particle 81 (ParT input; empty slot: 0)
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.jet_abs_eta            absolute pseudorapidity of the jet axis
  Q.jet_e                  jet energy [GeV]
  Q.jet_pt                 jet pT [GeV]
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lne_0                  ln E [GeV] of particle 0 (ParT input; empty slot: ln 1e-8)
  Q.lne_1                  ln E [GeV] of particle 1 (ParT input; empty slot: ln 1e-8)
  Q.lne_12                 ln E [GeV] of particle 12 (ParT input; empty slot: ln 1e-8)
  Q.lne_16                 ln E [GeV] of particle 16 (ParT input; empty slot: ln 1e-8)
  Q.lne_2                  ln E [GeV] of particle 2 (ParT input; empty slot: ln 1e-8)
  Q.lne_20                 ln E [GeV] of particle 20 (ParT input; empty slot: ln 1e-8)
  Q.lne_22                 ln E [GeV] of particle 22 (ParT input; empty slot: ln 1e-8)
  Q.lne_27                 ln E [GeV] of particle 27 (ParT input; empty slot: ln 1e-8)
  Q.lne_29                 ln E [GeV] of particle 29 (ParT input; empty slot: ln 1e-8)
  Q.lne_3                  ln E [GeV] of particle 3 (ParT input; empty slot: ln 1e-8)
  Q.lne_38                 ln E [GeV] of particle 38 (ParT input; empty slot: ln 1e-8)
  Q.lne_4                  ln E [GeV] of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lne_49                 ln E [GeV] of particle 49 (ParT input; empty slot: ln 1e-8)
  Q.lne_6                  ln E [GeV] of particle 6 (ParT input; empty slot: ln 1e-8)
  Q.lne_8                  ln E [GeV] of particle 8 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_0               ln(E / E of the jet) of particle 0 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_1               ln(E / E of the jet) of particle 1 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_12              ln(E / E of the jet) of particle 12 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_27              ln(E / E of the jet) of particle 27 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_4               ln(E / E of the jet) of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_50              ln(E / E of the jet) of particle 50 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_67              ln(E / E of the jet) of particle 67 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_77              ln(E / E of the jet) of particle 77 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_9               ln(E / E of the jet) of particle 9 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_0                 ln pT [GeV] of particle 0 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_1                 ln pT [GeV] of particle 1 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_10                ln pT [GeV] of particle 10 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_12                ln pT [GeV] of particle 12 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_18                ln pT [GeV] of particle 18 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_22                ln pT [GeV] of particle 22 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_29                ln pT [GeV] of particle 29 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_34                ln pT [GeV] of particle 34 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_4                 ln pT [GeV] of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_5                 ln pT [GeV] of particle 5 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_6                 ln pT [GeV] of particle 6 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_7                 ln pT [GeV] of particle 7 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_0              ln(pT / pT of the jet) of particle 0 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_14             ln(pT / pT of the jet) of particle 14 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_29             ln(pT / pT of the jet) of particle 29 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_30             ln(pT / pT of the jet) of particle 30 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_77             ln(pT / pT of the jet) of particle 77 (ParT input; empty slot: ln 1e-8)
  Q.log_sum_pt             natural log of the total pT
  Q.lund1_lndelta          ln Δ of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund1_lnkt             ln kT of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund1_lnz              ln z of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund2_lndelta          ln Δ of the 2. primary C/A splitting (ln 1e-8 if none)
  Q.lund2_lnkt             ln kT of the 2. primary C/A splitting (ln 1e-8 if none)
  Q.lund2_lnz              ln z of the 2. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lndelta          ln Δ of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lnkt             ln kT of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lnz              ln z of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund_max_lndelta       ln Δ of the primary splitting with the largest kT
  Q.lund_max_lnkt          largest ln kT among the primary splittings
  Q.m01                    mass of particles 0 and 1 [GeV]
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.min_pair_mass          smallest pair mass among particles 0, 1, 2 [GeV]
  Q.mratio_max_012         largest pair mass / mass of the 3 hardest (dimensionless)
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_dr_0p4_up            number of particles with ΔR ≥ 0.4
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_lund                 number of primary C/A splittings
  Q.n_lund_kt_above_1      number of primary splittings with kT > 1 GeV
  Q.n_lund_kt_above_5      number of primary splittings with kT > 5 GeV
  Q.n_pairs_kt_above_1     number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 1 GeV
  Q.n_pairs_kt_above_10    number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 10 GeV
  Q.n_pairs_kt_above_3     number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 3 GeV
  Q.n_pairs_kt_above_30    number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 30 GeV
  Q.n_particles            number of real particles (pT > 0)
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.n_real_top20           number of real particles among the 20 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pair_max_lnkt          largest ln kT among all pairs
  Q.pair_max_lnm2          largest ln m² among all pairs
  Q.pair_mean_lndelta      zᵢzⱼ-weighted mean of ln ΔRᵢⱼ over all pairs
  Q.pair_mean_lnkt         zᵢzⱼ-weighted mean of ln kT = ln(min(pTᵢ, pTⱼ)·ΔRᵢⱼ) over all pairs
  Q.pair_mean_lnm2         zᵢzⱼ-weighted mean of ln mᵢⱼ² (massless) over all pairs
  Q.pair_mean_lnz          zᵢzⱼ-weighted mean of ln z = ln(min(pTᵢ, pTⱼ)/(pTᵢ + pTⱼ)) over all pairs
  Q.phi_0                  Δφ of particle 0 (ParT input; empty slot: 0)
  Q.phi_13                 Δφ of particle 13 (ParT input; empty slot: 0)
  Q.phi_14                 Δφ of particle 14 (ParT input; empty slot: 0)
  Q.phi_17                 Δφ of particle 17 (ParT input; empty slot: 0)
  Q.phi_18                 Δφ of particle 18 (ParT input; empty slot: 0)
  Q.phi_19                 Δφ of particle 19 (ParT input; empty slot: 0)
  Q.phi_2                  Δφ of particle 2 (ParT input; empty slot: 0)
  Q.phi_20                 Δφ of particle 20 (ParT input; empty slot: 0)
  Q.phi_22                 Δφ of particle 22 (ParT input; empty slot: 0)
  Q.phi_24                 Δφ of particle 24 (ParT input; empty slot: 0)
  Q.phi_26                 Δφ of particle 26 (ParT input; empty slot: 0)
  Q.phi_27                 Δφ of particle 27 (ParT input; empty slot: 0)
  Q.phi_29                 Δφ of particle 29 (ParT input; empty slot: 0)
  Q.phi_30                 Δφ of particle 30 (ParT input; empty slot: 0)
  Q.phi_31                 Δφ of particle 31 (ParT input; empty slot: 0)
  Q.phi_33                 Δφ of particle 33 (ParT input; empty slot: 0)
  Q.phi_39                 Δφ of particle 39 (ParT input; empty slot: 0)
  Q.phi_47                 Δφ of particle 47 (ParT input; empty slot: 0)
  Q.phi_51                 Δφ of particle 51 (ParT input; empty slot: 0)
  Q.phi_54                 Δφ of particle 54 (ParT input; empty slot: 0)
  Q.phi_58                 Δφ of particle 58 (ParT input; empty slot: 0)
  Q.phi_68                 Δφ of particle 68 (ParT input; empty slot: 0)
  Q.phi_72                 Δφ of particle 72 (ParT input; empty slot: 0)
  Q.phi_8                  Δφ of particle 8 (ParT input; empty slot: 0)
  Q.phi_80                 Δφ of particle 80 (ParT input; empty slot: 0)
  Q.phi_9                  Δφ of particle 9 (ParT input; empty slot: 0)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt1_over_pt0           pT1 / pT0
  Q.pt2_over_pt0           pT2 / pT0
  Q.pt_balance01           min(pT0, pT1) / (pT0 + pT1)
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.sd_mass                soft-drop groomed mass, C/A on the 128 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_nremoved            number of branches removed by soft drop
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmax_over_m     largest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.sj3_z2                 pT share of subjet 2 of 3 (by pT)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.sj4_dr_min             smallest distance among the 4 subjet axes
  Q.sj4_pair_mass_max      largest mass of two of the 4 subjets [GeV]
  Q.sj4_pair_mass_min      smallest mass of two of the 4 subjets [GeV]
  Q.sj4_zsoft              pT share of the softest of 4 subjets
  Q.sum_e                  total energy of the particles [GeV]
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
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
  Q.z_1st                  largest pT share
  Q.z_2nd                  2nd-largest pT share
  Q.z_3rd                  3rd-largest pT share
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.z_dr_0p4_up            pT share of the particles with ΔR ≥ 0.4
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5                 pT share of the 5 largest
"""
import math
from types import SimpleNamespace

CLASSES = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
W = [[-0.01822192780673504, -0.018223922699689865, -0.01821100153028965, -0.018214771524071693, -0.018210945650935173, -0.018211685121059418, -0.018219465389847755, -0.018225939944386482, -0.018215201795101166, -0.018212569877505302], [-0.0021634986624121666, -0.0021633917931467295, -0.002162517048418522, -0.0021630555856972933, -0.002162644639611244, -0.0021621491760015488, -0.002163545461371541, -0.0021637133322656155, -0.002163414377719164, -0.002162326592952013], [0.017690112814307213, 0.017594698816537857, 0.017827976495027542, 0.017818087711930275, 0.017786001786589622, 0.017602840438485146, 0.017682665959000587, 0.017519237473607063, 0.017570585012435913, 0.017652569338679314], [0.003339486662298441, 0.0033395031932741404, 0.003341465489938855, 0.003340712282806635, 0.0033393441699445248, 0.0033387362491339445, 0.0033412263728678226, 0.003341742791235447, 0.003340923460200429, 0.0033391264732927084], [-0.7301616072654724, 0.09791335463523865, -0.17450349032878876, 0.5759408473968506, -0.3328898549079895, 0.8614599108695984, 0.04895413666963577, -0.5553299784660339, -0.1560581773519516, 0.3810499310493469], [0.03559383004903793, 0.01339625846594572, 0.01709081418812275, 0.010995804332196712, 0.01666048914194107, 0.17591996490955353, 0.05799321457743645, 0.06227768957614899, 0.06743571907281876, 0.21951113641262054], [0.018246276304125786, 0.018255265429615974, 0.018248165026307106, 0.018253302201628685, 0.0182547178119421, 0.018246028572320938, 0.018244193866848946, 0.01825674995779991, 0.018255237489938736, 0.018244821578264236], [-0.021627917885780334, -0.02162504941225052, -0.02162834070622921, -0.02162730135023594, -0.021627742797136307, -0.021626966074109077, -0.021629316732287407, -0.021628115326166153, -0.021629445254802704, -0.021614614874124527], [-0.005787142552435398, -0.005787602625787258, -0.005785602144896984, -0.005787078756839037, -0.00578576372936368, -0.005787025671452284, -0.005787552334368229, -0.005788770969957113, -0.005788708105683327, -0.005787767935544252], [0.652646005153656, -0.4205797612667084, -0.5709133744239807, 0.3899630904197693, 0.503659725189209, 0.3480503261089325, 0.27262064814567566, 0.09483171254396439, 0.559780478477478, 0.08770892024040222], [-0.006678011734038591, -0.006680261809378862, -0.006679002195596695, -0.006681141909211874, -0.00668038334697485, -0.0066800653003156185, -0.006681004539132118, -0.006681437138468027, -0.006681452505290508, -0.00667764525860548], [0.00417762016877532, 0.0041787344962358475, 0.004178027156740427, 0.004178134258836508, 0.0041779931634664536, 0.004178047180175781, 0.00417699757963419, 0.004178883042186499, 0.004178313072770834, 0.00417578499764204], [-0.020552951842546463, -0.020560309290885925, -0.020556747913360596, -0.020556751638650894, -0.020558906719088554, -0.020547229796648026, -0.020548561587929726, -0.020560935139656067, -0.020552705973386765, -0.020548036321997643], [-0.005080202128738165, -0.005064697004854679, -0.005076479632407427, -0.005083804950118065, -0.005072861909866333, -0.00508100213482976, -0.005074361804872751, -0.005071224644780159, -0.005075542256236076, -0.005059056915342808], [-0.008157660253345966, -0.008158886805176735, -0.008157528936862946, -0.008152023889124393, -0.008158408105373383, -0.008154698647558689, -0.008159095421433449, -0.008157763630151749, -0.008156364783644676, -0.008153160102665424], [-0.007940243929624557, -0.007945303805172443, -0.007941817864775658, -0.007944189943373203, -0.00794413685798645, -0.007938317954540253, -0.007941094227135181, -0.007945743389427662, -0.007940758019685745, -0.007939577102661133], [0.008016807958483696, 0.008016076870262623, 0.00801553763449192, 0.008015750907361507, 0.008018454536795616, 0.008015566505491734, 0.008015834726393223, 0.008019541390240192, 0.008016758598387241, 0.008014238439500332], [-0.013684825040400028, -0.013685612939298153, -0.01368161104619503, -0.013684019446372986, -0.01368232537060976, -0.013684242032468319, -0.013684989884495735, -0.01368303969502449, -0.013686083257198334, -0.013682350516319275], [0.033392854034900665, 0.03339868411421776, 0.03339498117566109, 0.03339378908276558, 0.03339028358459473, 0.03339065983891487, 0.033392127603292465, 0.033391110599040985, 0.03339020907878876, 0.033373937010765076], [0.019665826112031937, 0.019675256684422493, 0.0196735430508852, 0.01967230997979641, 0.019672317430377007, 0.019664710387587547, 0.019662583246827126, 0.0196772962808609, 0.019666779786348343, 0.019663453102111816], [-0.03545346483588219, -0.03547246754169464, -0.035448115319013596, -0.03545939177274704, -0.03546028584241867, -0.035473406314849854, -0.03547651320695877, -0.03546923026442528, -0.03547871857881546, -0.03544959798455238], [-0.006010076962411404, -0.006009729579091072, -0.00600992888212204, -0.006010605487972498, -0.006010303273797035, -0.006010015029460192, -0.006009962875396013, -0.006010419689118862, -0.006010278128087521, -0.006009513046592474], [-0.01968284510076046, -0.3850700557231903, -0.4174414277076721, -0.12699845433235168, -0.44063764810562134, -0.9030249714851379, 0.6422889232635498, 0.4206008017063141, 0.6969190835952759, -0.7567481994628906], [-0.010041841305792332, -0.01004061009734869, -0.010037001222372055, -0.010041097179055214, -0.01004259567707777, -0.010037705302238464, -0.010039426386356354, -0.010043739341199398, -0.010043205693364143, -0.01003845315426588], [0.02973105199635029, 0.0297258198261261, 0.02973085828125477, 0.029724184423685074, 0.0297243595123291, 0.0297330841422081, 0.029739633202552795, 0.029740966856479645, 0.029742281883955002, 0.029729565605521202], [-0.10716275870800018, -0.10716168582439423, -0.10715947300195694, -0.10715314000844955, -0.10715781897306442, -0.10715153813362122, -0.10716372728347778, -0.10721394419670105, -0.10716139525175095, -0.10715746879577637], [0.0035146994050592184, 0.0035109359305351973, 0.0035086972638964653, 0.003516048425808549, 0.0035107817966490984, 0.0035190158523619175, 0.0035186787135899067, 0.0035132162738591433, 0.00351566425524652, 0.0035112930927425623], [0.1827966421842575, 0.37277817726135254, 0.6375513076782227, -0.32910364866256714, -1.211993932723999, -0.9366849660873413, -0.14195820689201355, -0.3353545367717743, 0.20392297208309174, 0.6652587056159973], [0.19311262667179108, -0.07580959796905518, -0.14977802336215973, 0.10071705281734467, -0.0020063917618244886, -0.41682669520378113, 0.2222684919834137, 0.1532585471868515, 0.07081278413534164, -0.5620132684707642], [0.015384555794298649, 0.01538464892655611, 0.015387404710054398, 0.01538782473653555, 0.015387521125376225, 0.015385253354907036, 0.015391474589705467, 0.015391100198030472, 0.015393689274787903, 0.015386110171675682], [-0.010788452811539173, 0.14575301110744476, 0.20134232938289642, 0.0680728480219841, 0.08563748002052307, -0.006119611673057079, -0.40822291374206543, 0.14496514201164246, -0.2028258740901947, -0.07857867330312729], [0.009070524014532566, 0.009070336818695068, 0.009067150764167309, 0.009067947044968605, 0.009069480001926422, 0.009070522151887417, 0.009067258797585964, 0.009068550541996956, 0.009067261591553688, 0.009066819213330746], [0.0018678844207897782, 0.001867974060587585, 0.001867777551524341, 0.0018679294735193253, 0.001867991522885859, 0.0018677652115002275, 0.0018678574124351144, 0.0018679788336157799, 0.001868174527771771, 0.0018678316846489906], [-0.6367899775505066, -0.2902272939682007, 0.3991955518722534, -0.7664453983306885, 0.6174276471138, -0.28449350595474243, 0.2254040390253067, 0.4396357238292694, 0.38732296228408813, -1.2867321968078613], [-0.0007123086834326386, -0.0007122252718545496, -0.000712116074282676, -0.0007122382521629333, -0.0007121257367543876, -0.0007121964590623975, -0.0007121816161088645, -0.0007122099632397294, -0.0007122386596165597, -0.0007120600785128772], [0.04072640836238861, -0.14691993594169617, -0.14109927415847778, -0.04632819816470146, -0.051218125969171524, -0.045430704951286316, 0.3713523745536804, -0.19551438093185425, 0.12331623584032059, 0.003497796133160591], [-0.01868424005806446, -0.018685217946767807, -0.01868485100567341, -0.018685700371861458, -0.018688926473259926, -0.0186846312135458, -0.018684430047869682, -0.018690546974539757, -0.018689321354031563, -0.01868521422147751], [0.02287447825074196, 0.022875268012285233, 0.022873403504490852, 0.02287120185792446, 0.02287515625357628, 0.022869952023029327, 0.022870389744639397, 0.022878386080265045, 0.02287987992167473, 0.022867199033498764], [-0.7244415283203125, 0.04974116384983063, 0.5026509165763855, 0.5843549966812134, 0.8379828929901123, -0.49803459644317627, -0.2477058321237564, -0.3551958501338959, -0.5128905773162842, -0.10538041591644287], [-0.007355937268584967, -0.007355969864875078, -0.007355424575507641, -0.007354626897722483, -0.007355374284088612, -0.00735339755192399, -0.00735624460503459, -0.007356829009950161, -0.0073567540384829044, -0.007354075089097023], [-0.003163968212902546, -0.0031636846251785755, -0.0031637954525649548, -0.00316424248740077, -0.0031626573763787746, -0.0031647616997361183, -0.0031648289877921343, -0.003164444351568818, -0.0031641831155866385, -0.0031632031314074993], [-0.008611319586634636, -0.00861409492790699, -0.008611460216343403, -0.008612758480012417, -0.008610130287706852, -0.008613231591880322, -0.008612018078565598, -0.00861653033643961, -0.008609403856098652, -0.008608912117779255], [0.004705532919615507, 0.00470559811219573, 0.004706521984189749, 0.004706752020865679, 0.004707112908363342, 0.004706323146820068, 0.004706587176769972, 0.0047096493653953075, 0.004705905448645353, 0.004705831874161959], [0.010036836378276348, 0.010043825954198837, 0.010036594234406948, 0.01003705058246851, 0.010037864558398724, 0.010036447085440159, 0.010037296451628208, 0.010036631487309933, 0.010037360712885857, 0.010039448738098145], [-0.013583594001829624, -0.013582632876932621, -0.013583643361926079, -0.01357878465205431, -0.013581246137619019, -0.01358005777001381, -0.013582324609160423, -0.013585453853011131, -0.013582068495452404, -0.01357833668589592], [0.01913236640393734, 0.01913473941385746, 0.019127588719129562, 0.01912786066532135, 0.0191288311034441, 0.01912512630224228, 0.01913389377295971, 0.019141754135489464, 0.019132938235998154, 0.019129866734147072], [-0.00886219646781683, -0.008867224678397179, -0.008863775059580803, -0.008865917101502419, -0.008865689858794212, -0.008861817419528961, -0.008862567134201527, -0.008869116194546223, -0.008867104537785053, -0.00886549148708582], [-0.02597908116877079, -0.025989992544054985, -0.025974569842219353, -0.025980457663536072, -0.02596946433186531, -0.02596847526729107, -0.025958193466067314, -0.025968749076128006, -0.025966674089431763, -0.025969648733735085], [0.03465237468481064, 0.034653522074222565, 0.034651387482881546, 0.034652743488550186, 0.034652579575777054, 0.0346507728099823, 0.03464736044406891, 0.034652434289455414, 0.034652501344680786, 0.034652791917324066], [0.020106464624404907, 0.020117033272981644, 0.02010435238480568, 0.02011319436132908, 0.020116068422794342, 0.02010980248451233, 0.020116133615374565, 0.02011800929903984, 0.0201143566519022, 0.020105473697185516], [0.030619027093052864, 0.030623290687799454, 0.030618948861956596, 0.030622249469161034, 0.030626052990555763, 0.030619880184531212, 0.03063071519136429, 0.030629459768533707, 0.030631830915808678, 0.03061787225306034], [0.6071838736534119, -0.41654443740844727, -0.5706896781921387, 0.08883025497198105, -0.07930935174226761, -0.9962469339370728, 0.15032067894935608, 0.5623717308044434, -1.1138765811920166, -0.6446731090545654], [-0.00012487095955293626, -0.0001254517846973613, -0.0001249985652975738, -0.0001246535830432549, -0.00012441100261639804, -0.00012451224029064178, -0.00012456397234927863, -0.00012472158414311707, -0.00012555021385196596, -0.0001249833730980754], [0.020106060430407524, 0.020105356350541115, 0.02010284550487995, 0.02010459452867508, 0.02010633423924446, 0.020103102549910545, 0.020104102790355682, 0.02011043392121792, 0.020108135417103767, 0.02010476402938366], [0.02310413122177124, 0.6079930067062378, 0.20754943788051605, 0.43534186482429504, 0.5062739849090576, -0.8788766860961914, -0.3940946161746979, -0.8613174557685852, 0.2429743856191635, -0.9003410339355469], [-0.015769699588418007, -0.015768295153975487, -0.015766168013215065, -0.015767144039273262, -0.015768762677907944, -0.015769632533192635, -0.015768462792038918, -0.015768542885780334, -0.015764642506837845, -0.015759117901325226], [-0.0012883529998362064, -0.001287571620196104, -0.0012871461221948266, -0.001288249040953815, -0.0012886381009593606, -0.0012879460118710995, -0.0012883504386991262, -0.0012887736083939672, -0.0012884620809927583, -0.0012879533460363746], [0.08179951459169388, 0.04561958834528923, -0.029204299673438072, -0.019306713715195656, -0.007240188308060169, 0.2950687110424042, -0.023803260177373886, -0.11810196936130524, -0.06002580747008324, -0.13669542968273163], [-0.02473408170044422, -0.024741698056459427, -0.024739455431699753, -0.024733373895287514, -0.024738244712352753, -0.02473480813205242, -0.024728452786803246, -0.024743221700191498, -0.024734344333410263, -0.024728262796998024], [0.010730873793363571, 0.0107371611520648, 0.010731196030974388, 0.010733330622315407, 0.010737022385001183, 0.010731196030974388, 0.010733856819570065, 0.010737661272287369, 0.010735167190432549, 0.010731359012424946], [-0.004755547270178795, -0.00475843483582139, -0.004757223185151815, -0.004757545422762632, -0.004757593851536512, -0.004757157526910305, -0.0047575621865689754, -0.004757613874971867, -0.004758594091981649, -0.004756270442157984], [-0.0033044111914932728, -0.0033063816372305155, -0.0033028842881321907, -0.003303102683275938, -0.003303257282823324, -0.003302432596683502, -0.003303623292595148, -0.0033041341230273247, -0.003303295699879527, -0.0033043241128325462], [0.0038714278489351273, 0.0038719524163752794, 0.0038713670801371336, 0.003872249973937869, 0.0038715905975550413, 0.0038710408844053745, 0.0038722336757928133, 0.0038727496284991503, 0.0038716646376997232, 0.0038711889646947384], [-0.04036068916320801, -0.04035858437418938, -0.04036026820540428, -0.04036056250333786, -0.04036441817879677, -0.0403621569275856, -0.04037446156144142, -0.040379807353019714, -0.04038351774215698, -0.04036084935069084], [-0.017434054985642433, -0.017437592148780823, -0.01743323542177677, -0.017439380288124084, -0.017442597076296806, -0.01743393763899803, -0.017447680234909058, -0.017448334023356438, -0.017443815246224403, -0.017435451969504356], [-0.016098417341709137, -0.016089821234345436, -0.01609937660396099, -0.01606983132660389, -0.01609262265264988, -0.016088780015707016, -0.01607394404709339, -0.01608751341700554, -0.016087446361780167, -0.016096316277980804], [0.013288096524775028, 0.013288808055222034, 0.013291570357978344, 0.01329100877046585, 0.013291077688336372, 0.013288279995322227, 0.013293620198965073, 0.013294842094182968, 0.013289370574057102, 0.013287545181810856], [-0.01764504425227642, -0.017646029591560364, -0.017641613259911537, -0.017646517604589462, -0.017645729705691338, -0.017644574865698814, -0.017644323408603668, -0.01764608360826969, -0.017644891515374184, -0.017633333802223206], [0.025505203753709793, 0.025505084544420242, 0.025505224242806435, 0.025511054322123528, 0.02550896629691124, 0.025505289435386658, 0.02550497092306614, 0.02550577186048031, 0.025504974648356438, 0.025506317615509033], [-0.12790870666503906, 0.15491053462028503, 0.319791316986084, 0.18899649381637573, 0.016976622864603996, -0.03744540736079216, -0.6537112593650818, 0.2738505005836487, -0.15752913057804108, 0.03444282338023186], [0.12283387780189514, -0.09733038395643234, -0.3012671172618866, 0.018763240426778793, -0.305300772190094, 0.626775324344635, 0.5360485315322876, -0.3623552918434143, 0.38106241822242737, -0.6032472252845764], [-0.01938798278570175, -0.01940065249800682, -0.01939309947192669, -0.01939949207007885, -0.01939977891743183, -0.019388142973184586, -0.01939922198653221, -0.019400889053940773, -0.019394807517528534, -0.01939561776816845], [-0.014001138508319855, -0.014009620063006878, -0.014002918265759945, -0.014018616639077663, -0.014010156504809856, -0.014015336520969868, -0.014004342257976532, -0.01400501374155283, -0.01398791279643774, -0.014011384919285774], [0.018517756834626198, 0.01852189190685749, 0.018517402932047844, 0.018525544553995132, 0.018521133810281754, 0.018521204590797424, 0.01852981373667717, 0.018527543172240257, 0.01853221468627453, 0.01851748116314411], [-0.01899322122335434, -0.01899128593504429, -0.01898963749408722, -0.01899065636098385, -0.018989766016602516, -0.01898777484893799, -0.018991833552718163, -0.01899363286793232, -0.018989989534020424, -0.018987072631716728], [-0.01782926730811596, -0.01783001236617565, -0.017829440534114838, -0.017832497134804726, -0.017829913645982742, -0.017830073833465576, -0.017830194905400276, -0.01782948523759842, -0.017830463126301765, -0.017824159935116768], [0.01329454779624939, 0.013294263742864132, 0.013294086791574955, 0.013295377604663372, 0.013296278193593025, 0.013294103555381298, 0.013294344767928123, 0.013294347561895847, 0.013294262811541557, 0.013293863274157047], [0.018563099205493927, 0.018568722531199455, 0.018563173711299896, 0.018570678308606148, 0.018571030348539352, 0.018566787242889404, 0.01857629604637623, 0.018577612936496735, 0.0185712780803442, 0.018564624711871147], [-0.0027080420404672623, -0.002700114855542779, -0.002708581043407321, -0.0027036366518586874, -0.002707288134843111, -0.0027063239831477404, -0.00270058773458004, -0.0027051481883972883, -0.0027114541735500097, -0.0027127156499773264], [-0.000788459088653326, -0.0007884168298915029, -0.000788362929597497, -0.0007884791702963412, -0.0007884528604336083, -0.0007884257356636226, -0.000788454432040453, -0.0007884571678005159, -0.0007884662481956184, -0.000788146338891238], [-0.0383637361228466, -0.03837263584136963, -0.0383622832596302, -0.03837363421916962, -0.03837105259299278, -0.03836354240775108, -0.038363393396139145, -0.03837582841515541, -0.03837454691529274, -0.03835878521203995], [0.19712476432323456, 0.19650736451148987, -0.05028354749083519, 0.013763533905148506, -0.22452189028263092, 0.22577643394470215, -0.7084058523178101, 0.7227587699890137, 0.15770402550697327, -0.4560408294200897], [-0.08011911809444427, 0.02251986227929592, 0.01804065890610218, 0.024065611883997917, 0.021116161718964577, -0.28466135263442993, -0.23452894389629364, -0.024605069309473038, -0.13380549848079681, -0.24936148524284363], [-0.02026117965579033, -0.02026262879371643, -0.02025504596531391, -0.020254330709576607, -0.020257487893104553, -0.02025483176112175, -0.020260442048311234, -0.020257193595170975, -0.020253067836165428, -0.02024381421506405], [-0.011013191193342209, -0.011012709699571133, -0.011012966744601727, -0.011013230308890343, -0.011015108786523342, -0.01101342961192131, -0.011018672026693821, -0.011016933247447014, -0.011013190262019634, -0.01101356279104948], [-0.00762165104970336, -0.007622865028679371, -0.0076221502386033535, -0.007619764190167189, -0.00762370927259326, -0.007619301788508892, -0.007619859650731087, -0.007620904594659805, -0.007620513904839754, -0.007622901350259781], [-0.42233145236968994, 0.7840976119041443, -0.24303372204303741, -0.88011234998703, -0.6209192872047424, 0.34952279925346375, 0.46825921535491943, 0.31027328968048096, -0.6176993250846863, 0.1076078712940216], [-0.0027476849500089884, -0.0027478800620883703, -0.002747537335380912, -0.0027476726099848747, -0.002748073311522603, -0.002747628139331937, -0.002747769234701991, -0.0027488977648317814, -0.0027477573603391647, -0.0027476500254124403], [-0.024717632681131363, -0.024717355147004128, -0.024712905287742615, -0.02471422590315342, -0.02471006102859974, -0.024714330211281776, -0.0247123371809721, -0.024722199887037277, -0.024717146530747414, -0.024713700637221336], [-0.005362497642636299, -0.005364603828638792, -0.005362572148442268, -0.005363781470805407, -0.005363092292100191, -0.0053619504906237125, -0.0053622545674443245, -0.005362613592296839, -0.005365235731005669, -0.005361699499189854], [0.013102655299007893, 0.013106008060276508, 0.013105000369250774, 0.013105159625411034, 0.01310571376234293, 0.013105015270411968, 0.013103732839226723, 0.013103228993713856, 0.013106252066791058, 0.013103261590003967], [-0.005281541030853987, -0.0052826907485723495, -0.005282113794237375, -0.005281960591673851, -0.005281844642013311, -0.005280881188809872, -0.005284786224365234, -0.005282757803797722, -0.005284030921757221, -0.005280324257910252], [-0.010326736606657505, -0.010325828567147255, -0.01032628770917654, -0.010326847434043884, -0.010330545715987682, -0.010327182710170746, -0.010331585071980953, -0.010329922661185265, -0.010328950360417366, -0.010326411575078964], [0.004907586146146059, 0.0049073281697928905, 0.004907458554953337, 0.004907212220132351, 0.004907319322228432, 0.004906886722892523, 0.004907205235213041, 0.004907739348709583, 0.004907521884888411, 0.004906508140265942], [0.023828741163015366, 0.023835796862840652, 0.023827631026506424, 0.02382688969373703, 0.023829922080039978, 0.02382573112845421, 0.02382785826921463, 0.02383507788181305, 0.023831456899642944, 0.02382831647992134], [0.02623964659869671, 0.026240283623337746, 0.026236925274133682, 0.02623894438147545, 0.026243600994348526, 0.026237716898322105, 0.02624412626028061, 0.02625321038067341, 0.026240935549139977, 0.026241207495331764], [0.024050815030932426, 0.024053281173110008, 0.02404228411614895, 0.0240490585565567, 0.024046337231993675, 0.024050455540418625, 0.024052076041698456, 0.024057194590568542, 0.02405555173754692, 0.02404623106122017], [-0.7974182963371277, 0.417285293340683, -0.34986114501953125, 0.7146439552307129, 0.47957372665405273, -0.6708701848983765, -0.012628917582333088, 0.15504565834999084, 0.03689786419272423, 0.83966064453125], [-0.01624131388962269, -0.016240600496530533, -0.01624242030084133, -0.01624237932264805, -0.016242587938904762, -0.01624259725213051, -0.016241664066910744, -0.01624295860528946, -0.016242649406194687, -0.016232334077358246], [-0.027475876733660698, -0.027470527216792107, -0.02746943198144436, -0.027474690228700638, -0.02747246064245701, -0.027473263442516327, -0.027478428557515144, -0.027475683018565178, -0.02747616171836853, -0.027469893917441368], [-0.01168819796293974, -0.01168834138661623, -0.011686468496918678, -0.011688286438584328, -0.01168467290699482, -0.011686833575367928, -0.011692681349813938, -0.01169157586991787, -0.011691599152982235, -0.011687551625072956], [-0.026952596381306648, -0.026952259242534637, -0.026949690654873848, -0.026952261105179787, -0.026949221268296242, -0.026950672268867493, -0.026952987536787987, -0.02695518173277378, -0.026954136788845062, -0.026943011209368706], [0.04095865786075592, -0.18762946128845215, -0.08523557335138321, -0.07476040720939636, 0.06491943448781967, -0.20579494535923004, 0.5840845108032227, -0.401909738779068, -0.19501402974128723, 0.136881485581398], [0.2421182096004486, -0.0014151217183098197, -0.3830263912677765, -0.3444571793079376, 0.8199025392532349, 0.23997116088867188, -0.41431915760040283, -0.09347639977931976, 1.58009934425354, 1.6220146417617798], [0.41530078649520874, 0.21218039095401764, 0.20151068270206451, -0.0005621737218461931, 0.5641027092933655, -0.9787026047706604, 0.39356529712677, -0.2914770841598511, -0.6633836627006531, -0.493389755487442], [-0.009708738885819912, -0.00970923900604248, -0.009708632715046406, -0.009709158912301064, -0.009709948673844337, -0.009709184989333153, -0.00970876682549715, -0.00971321202814579, -0.009708845987915993, -0.009709127247333527], [-0.013385162688791752, -0.013383692130446434, -0.013379674404859543, -0.013380816206336021, -0.013383695855736732, -0.013383348472416401, -0.013384350575506687, -0.013387680053710938, -0.013393436558544636, -0.013380923308432102], [0.06010771170258522, 0.06010660156607628, 0.060102418065071106, 0.06010894104838371, 0.06010524183511734, 0.06010112166404724, 0.06010574474930763, 0.06011044606566429, 0.060119856148958206, 0.06009618937969208], [-0.010830544866621494, -0.010829861275851727, -0.010829060338437557, -0.010830957442522049, -0.010832421481609344, -0.010829875245690346, -0.010830050334334373, -0.010829842649400234, -0.010829993523657322, -0.010828985832631588], [-0.011340397410094738, -0.011346141807734966, -0.011341491714119911, -0.011342886835336685, -0.011347676627337933, -0.011341413483023643, -0.011341188102960587, -0.011347842402756214, -0.011341807432472706, -0.011342903599143028], [-0.009183632209897041, -0.009188700467348099, -0.009182300418615341, -0.009188900701701641, -0.009188295342028141, -0.00918642245233059, -0.009189276956021786, -0.00918852910399437, -0.009188701398670673, -0.009182964451611042], [-0.004019890911877155, -0.00401925528421998, -0.004019048996269703, -0.004018697887659073, -0.004018252249807119, -0.004018106963485479, -0.004018272273242474, -0.00401958217844367, -0.004018615465611219, -0.004019020590931177], [-0.01771109364926815, -0.0177206601947546, -0.017713403329253197, -0.01771070435643196, -0.0177154578268528, -0.0177104864269495, -0.01771259307861328, -0.01772184856235981, -0.0177199374884367, -0.017710503190755844], [-0.02076590433716774, -0.02076009474694729, -0.020758183673024178, -0.02075691521167755, -0.0207471065223217, -0.02075764164328575, -0.02075139246881008, -0.020754573866724968, -0.02075064554810524, -0.02075173147022724], [-0.015971852466464043, -0.01597461849451065, -0.015971714630723, -0.015971509739756584, -0.015979627147316933, -0.015972740948200226, -0.015977902337908745, -0.015985021367669106, -0.015973515808582306, -0.0159689262509346], [-0.0016564277466386557, -0.0016565148252993822, -0.0016562426462769508, -0.0016561999218538404, -0.001656138920225203, -0.0016560513759031892, -0.0016564460238441825, -0.001657115644775331, -0.001656818320043385, -0.0016562623204663396], [-0.028466004878282547, -0.028467625379562378, -0.02847333252429962, -0.02847820520401001, -0.028473371639847755, -0.02846546098589897, -0.02847299538552761, -0.02848082408308983, -0.028472794219851494, -0.028463147580623627], [-0.003105201292783022, -0.0031035577412694693, -0.0031054813880473375, -0.0031036098953336477, -0.003101455047726631, -0.0031028049997985363, -0.003102765651419759, -0.0031031612306833267, -0.0030990620143711567, -0.003097119042649865], [0.002793496008962393, 0.0027935642283409834, 0.002793569816276431, 0.0027937861159443855, 0.002794236410409212, 0.0027945046313107014, 0.002793822670355439, 0.0027952329255640507, 0.0027939234860241413, 0.002793449442833662], [0.01670181192457676, 0.016700079664587975, 0.01669982261955738, 0.01670078933238983, 0.016703862696886063, 0.016700534150004387, 0.01670100726187229, 0.01669890247285366, 0.016697639599442482, 0.01669909991323948], [-0.010562526993453503, -0.01056961715221405, -0.010562905110418797, -0.010563572868704796, -0.010570688173174858, -0.010564573109149933, -0.010562860406935215, -0.010570823214948177, -0.010562917217612267, -0.010563739575445652], [0.0003188322589267045, -0.5406482815742493, -0.1503109484910965, -0.5665838122367859, 0.2653762400150299, 0.5949655771255493, 0.11226224899291992, -0.15949249267578125, -0.9399242997169495, 0.7696388363838196], [-0.004386436194181442, -0.004388194065541029, -0.004385435022413731, -0.0043860930018126965, -0.004388401284813881, -0.004388163331896067, -0.0043883961625397205, -0.004388401284813881, -0.004388377536088228, -0.004385439679026604], [-0.010054892860352993, -0.010061154142022133, -0.010055548511445522, -0.010052181780338287, -0.010045730508863926, -0.010063434951007366, -0.010062504559755325, -0.010057845152914524, -0.010059034451842308, -0.010055452585220337], [0.002185689751058817, 0.002184717683121562, 0.0021851113997399807, 0.0021857917308807373, 0.00218459265306592, 0.002184698823839426, 0.0021855728700757027, 0.002186086028814316, 0.002184987301006913, 0.0021852036006748676], [0.02659710869193077, 0.02660137228667736, 0.026597576215863228, 0.02659769542515278, 0.02659866213798523, 0.02659984678030014, 0.02660195343196392, 0.02660340443253517, 0.026601431891322136, 0.026595404371619225], [0.03880545124411583, 0.03880423679947853, 0.03880372270941734, 0.038806408643722534, 0.03880389407277107, 0.038803450763225555, 0.0388050377368927, 0.03881329670548439, 0.03880557417869568, 0.03880667686462402], [-0.002133610425516963, -0.002135121263563633, -0.002133541740477085, -0.0021350851748138666, -0.002134423004463315, -0.002134710783138871, -0.002134815091267228, -0.0021339948289096355, -0.00213481648825109, -0.0021338986698538065]]
B = [0.44403335452079773, 0.209792822599411, 0.3583735525608063, 0.05534523352980614, -0.11267533898353577, -0.4036940634250641, 0.3007381856441498, 0.03525549918413162, -0.5069618225097656, -1.3898615837097168]


def quantities(pt, eta, phi, energy, jet_pt, jet_eta, jet_energy):
    pt, eta, phi, energy = [float(x) for x in pt], [float(x) for x in eta], [float(x) for x in phi], [float(x) for x in energy]
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

    return SimpleNamespace(
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b05=ecfb('e3', 0.5) / max(ecfb('e2', 0.5) ** 2, 1e-30),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
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
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        N3_b05=ecfb('g42', 0.5) / max(ecfb('g31', 0.5) ** 2, 1e-30),
        N3_b2=ecfb('g42', 2) / max(ecfb('g31', 2) ** 2, 1e-30),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        dr01=math.sqrt(dist2(0, 1)),
        dr02=math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        dr_0=pfeat(0, 'dr'),
        dr_1=pfeat(1, 'dr'),
        dr_10=pfeat(10, 'dr'),
        dr_11=pfeat(11, 'dr'),
        dr_12=pfeat(12, 'dr'),
        dr_13=pfeat(13, 'dr'),
        dr_14=pfeat(14, 'dr'),
        dr_16=pfeat(16, 'dr'),
        dr_17=pfeat(17, 'dr'),
        dr_18=pfeat(18, 'dr'),
        dr_19=pfeat(19, 'dr'),
        dr_2=pfeat(2, 'dr'),
        dr_21=pfeat(21, 'dr'),
        dr_22=pfeat(22, 'dr'),
        dr_23=pfeat(23, 'dr'),
        dr_25=pfeat(25, 'dr'),
        dr_3=pfeat(3, 'dr'),
        dr_31=pfeat(31, 'dr'),
        dr_32=pfeat(32, 'dr'),
        dr_33=pfeat(33, 'dr'),
        dr_34=pfeat(34, 'dr'),
        dr_35=pfeat(35, 'dr'),
        dr_37=pfeat(37, 'dr'),
        dr_38=pfeat(38, 'dr'),
        dr_4=pfeat(4, 'dr'),
        dr_40=pfeat(40, 'dr'),
        dr_41=pfeat(41, 'dr'),
        dr_46=pfeat(46, 'dr'),
        dr_47=pfeat(47, 'dr'),
        dr_5=pfeat(5, 'dr'),
        dr_53=pfeat(53, 'dr'),
        dr_55=pfeat(55, 'dr'),
        dr_57=pfeat(57, 'dr'),
        dr_6=pfeat(6, 'dr'),
        dr_7=pfeat(7, 'dr'),
        dr_8=pfeat(8, 'dr'),
        dr_9=pfeat(9, 'dr'),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        e2=e2,
        e2_b05=ecfb('e2', 0.5),
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
        ecf_g32=ecfb('g32', 1),
        ecf_g41=ecfb('g41', 1),
        ecf_g42=ecfb('g42', 1),
        ecf_g43=ecfb('g43', 1),
        eta_0=pfeat(0, 'eta'),
        eta_1=pfeat(1, 'eta'),
        eta_12=pfeat(12, 'eta'),
        eta_14=pfeat(14, 'eta'),
        eta_17=pfeat(17, 'eta'),
        eta_19=pfeat(19, 'eta'),
        eta_2=pfeat(2, 'eta'),
        eta_20=pfeat(20, 'eta'),
        eta_23=pfeat(23, 'eta'),
        eta_25=pfeat(25, 'eta'),
        eta_26=pfeat(26, 'eta'),
        eta_27=pfeat(27, 'eta'),
        eta_28=pfeat(28, 'eta'),
        eta_29=pfeat(29, 'eta'),
        eta_31=pfeat(31, 'eta'),
        eta_33=pfeat(33, 'eta'),
        eta_37=pfeat(37, 'eta'),
        eta_38=pfeat(38, 'eta'),
        eta_39=pfeat(39, 'eta'),
        eta_4=pfeat(4, 'eta'),
        eta_40=pfeat(40, 'eta'),
        eta_41=pfeat(41, 'eta'),
        eta_42=pfeat(42, 'eta'),
        eta_50=pfeat(50, 'eta'),
        eta_55=pfeat(55, 'eta'),
        eta_57=pfeat(57, 'eta'),
        eta_64=pfeat(64, 'eta'),
        eta_73=pfeat(73, 'eta'),
        eta_77=pfeat(77, 'eta'),
        eta_81=pfeat(81, 'eta'),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        jet_abs_eta=abs(jet_eta),
        jet_e=jet_energy,
        jet_pt=jet_pt,
        lam1=lam1,
        lam2=lam2,
        lne_0=pfeat(0, 'lne'),
        lne_1=pfeat(1, 'lne'),
        lne_12=pfeat(12, 'lne'),
        lne_16=pfeat(16, 'lne'),
        lne_2=pfeat(2, 'lne'),
        lne_20=pfeat(20, 'lne'),
        lne_22=pfeat(22, 'lne'),
        lne_27=pfeat(27, 'lne'),
        lne_29=pfeat(29, 'lne'),
        lne_3=pfeat(3, 'lne'),
        lne_38=pfeat(38, 'lne'),
        lne_4=pfeat(4, 'lne'),
        lne_49=pfeat(49, 'lne'),
        lne_6=pfeat(6, 'lne'),
        lne_8=pfeat(8, 'lne'),
        lnerel_0=pfeat(0, 'lnerel'),
        lnerel_1=pfeat(1, 'lnerel'),
        lnerel_12=pfeat(12, 'lnerel'),
        lnerel_27=pfeat(27, 'lnerel'),
        lnerel_4=pfeat(4, 'lnerel'),
        lnerel_50=pfeat(50, 'lnerel'),
        lnerel_67=pfeat(67, 'lnerel'),
        lnerel_77=pfeat(77, 'lnerel'),
        lnerel_9=pfeat(9, 'lnerel'),
        lnpt_0=pfeat(0, 'lnpt'),
        lnpt_1=pfeat(1, 'lnpt'),
        lnpt_10=pfeat(10, 'lnpt'),
        lnpt_12=pfeat(12, 'lnpt'),
        lnpt_18=pfeat(18, 'lnpt'),
        lnpt_22=pfeat(22, 'lnpt'),
        lnpt_29=pfeat(29, 'lnpt'),
        lnpt_34=pfeat(34, 'lnpt'),
        lnpt_4=pfeat(4, 'lnpt'),
        lnpt_5=pfeat(5, 'lnpt'),
        lnpt_6=pfeat(6, 'lnpt'),
        lnpt_7=pfeat(7, 'lnpt'),
        lnptrel_0=pfeat(0, 'lnptrel'),
        lnptrel_14=pfeat(14, 'lnptrel'),
        lnptrel_29=pfeat(29, 'lnptrel'),
        lnptrel_30=pfeat(30, 'lnptrel'),
        lnptrel_77=pfeat(77, 'lnptrel'),
        log_sum_pt=math.log(tot),
        lund1_lndelta=lund(1, 'lndelta'),
        lund1_lnkt=lund(1, 'lnkt'),
        lund1_lnz=lund(1, 'lnz'),
        lund2_lndelta=lund(2, 'lndelta'),
        lund2_lnkt=lund(2, 'lnkt'),
        lund2_lnz=lund(2, 'lnz'),
        lund3_lndelta=lund(3, 'lndelta'),
        lund3_lnkt=lund(3, 'lnkt'),
        lund3_lnz=lund(3, 'lnz'),
        lund_max_lndelta=lund(0, 'maxdelta'),
        lund_max_lnkt=lund(0, 'maxkt'),
        m01=pair_mass(0, 1),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        mass=mass_of(n),
        mass_over_sum_pt=mass_of(n) / tot,
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top2=mass_of(2),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        mratio_max_012=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        n_for_90pct=ncum(0.9),
        n_lund=lund(0, 'n'),
        n_lund_kt_above_1=lund(0, 'n1'),
        n_lund_kt_above_5=lund(0, 'n5'),
        n_pairs_kt_above_1=paircount(1),
        n_pairs_kt_above_10=paircount(10),
        n_pairs_kt_above_3=paircount(3),
        n_pairs_kt_above_30=paircount(30),
        n_particles=len(real),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        n_real_top20=sum(1 for x in pt[:20] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pair_max_lnkt=pairmax('lnkt'),
        pair_max_lnm2=pairmax('lnm2'),
        pair_mean_lndelta=pairsum('lndelta'),
        pair_mean_lnkt=pairsum('lnkt'),
        pair_mean_lnm2=pairsum('lnm2'),
        pair_mean_lnz=pairsum('lnz'),
        phi_0=pfeat(0, 'phi'),
        phi_13=pfeat(13, 'phi'),
        phi_14=pfeat(14, 'phi'),
        phi_17=pfeat(17, 'phi'),
        phi_18=pfeat(18, 'phi'),
        phi_19=pfeat(19, 'phi'),
        phi_2=pfeat(2, 'phi'),
        phi_20=pfeat(20, 'phi'),
        phi_22=pfeat(22, 'phi'),
        phi_24=pfeat(24, 'phi'),
        phi_26=pfeat(26, 'phi'),
        phi_27=pfeat(27, 'phi'),
        phi_29=pfeat(29, 'phi'),
        phi_30=pfeat(30, 'phi'),
        phi_31=pfeat(31, 'phi'),
        phi_33=pfeat(33, 'phi'),
        phi_39=pfeat(39, 'phi'),
        phi_47=pfeat(47, 'phi'),
        phi_51=pfeat(51, 'phi'),
        phi_54=pfeat(54, 'phi'),
        phi_58=pfeat(58, 'phi'),
        phi_68=pfeat(68, 'phi'),
        phi_72=pfeat(72, 'phi'),
        phi_8=pfeat(8, 'phi'),
        phi_80=pfeat(80, 'phi'),
        phi_9=pfeat(9, 'phi'),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        pt_balance01=min(pt[0], pt[1]) / max(pt[0] + pt[1], 1e-9),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        sd_mass=softdrop("mass"),
        sd_nremoved=softdrop("removed"),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        sj2_dr=subjets(2)["dr"][0],
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        sj2_zsoft=subjets(2)["z"][1],
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        sj3_dr_max=max(subjets(3)["dr"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmax_over_m=max(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_z1=subjets(3)["z"][0],
        sj3_z2=subjets(3)["z"][1],
        sj3_z3=subjets(3)["z"][2],
        sj4_dr_min=min(subjets(4)["dr"]),
        sj4_pair_mass_max=max(subjets(4)["mpair"]),
        sj4_pair_mass_min=min(subjets(4)["mpair"]),
        sj4_zsoft=subjets(4)["z"][3],
        sum_e=sum(energy[i] for i in real),
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top40=sum(pt[:40]),
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
        z_1st=zs[0],
        z_2nd=zs[1],
        z_3rd=zs[2],
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        z_dr_0p4_up=sum(z[i] for i in real if 0.4 <= dr[i] < 10),
        z_top10_slots=sum(pt[:10]) / tot,
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top2_slots=sum(pt[:2]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top3_slots=sum(pt[:3]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5=sum(zs[:5]),
    )


def neuron_0(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.322344e-05
    )
    return z


def neuron_1(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.761036e-06
    )
    return z


def neuron_2(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.000625311
    )
    return z


def neuron_3(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.991911e-06
    )
    return z


def neuron_4(Q):
    # scale S = 3.507; each line: share * term / its average size
    z = 3.506706 * (-0.03919707
        + 0.1311114 * max(0.0, 149.0507 - Q.mass) / 39.08704   # +13.1%  mass < 149.1
        - 0.1070612 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) / 26.86577   # -10.7%  sj3_pair_mass_max < 114.8
        - 0.08122568 * max(0.0, 17.3274 - Q.m01) / 8.627026   # -8.1%  m01 < 17.33
        + 0.0614345 * max(0.0, 0.3258728 - Q.dr01) / 0.1941327   # +6.1%  dr01 < 0.3259
        - 0.04949965 * max(0.0, 0.04495465 - Q.M3) / 0.01249797   # -4.9%  M3 < 0.04495
        + 0.04164145 * max(0.0, 93.87663 - Q.sj3_pair_mass_max) / 12.55772   # +4.2%  sj3_pair_mass_max < 93.88
        - 0.03222896 * max(0.0, 104.3234 - Q.mass_top50) / 9.697487   # -3.2%  mass_top50 < 104.3
        - 0.02945514 * max(0.0, 0.03484839 - Q.sum_z_dr2_top10) / 0.01280026   # -2.9%  sum_z_dr2_top10 < 0.03485
        + 0.02825923 * max(0.0, 0.7045747 - Q.N3_b05) / 0.1131179   # +2.8%  N3_b05 < 0.7046
        + 0.02799242 * max(0.0, -1.332584 - Q.pair_mean_lnz) / 0.4204304   # +2.8%  pair_mean_lnz < -1.333
        + 0.02435374 * max(0.0, 15.07073 - Q.pt1_dr01) / 7.957755   # +2.4%  pt1_dr01 < 15.07
        + 0.02329821 * max(0.0, 0.02497878 - Q.M3_b2) / 0.01258377   # +2.3%  M3_b2 < 0.02498
        + 0.02260921 * max(0.0, Q.D3 - 0.1179386) / 0.2711908   # +2.3%  D3 > 0.1179
        - 0.0213076 * max(0.0, 411.0 - Q.n_pairs_kt_above_1) / 159.5699   # -2.1%  n_pairs_kt_above_1 < 411
        - 0.02104271 * max(0.0, 0.1301769 - Q.M2_b05) / 0.008681393   # -2.1%  M2_b05 < 0.1302
        - 0.02061973 * max(0.0, Q.sj3_pairmax_over_m - 0.6888277) / 0.1221148   # -2.1%  sj3_pairmax_over_m > 0.6888
        - 0.01476109 * max(0.0, Q.D3 - 0.3195268) / 0.1636318   # -1.5%  D3 > 0.3195
        + 0.01474674 * max(0.0, 149.0507 - Q.mass) * max(0.0, Q.lund_max_lndelta - -1.233804) / 7.162554   # +1.5%  mass < 149.1 and lund_max_lndelta > -1.234
        - 0.01406206 * max(0.0, 0.003965728 - Q.ecf_g32) / 0.001936808   # -1.4%  ecf_g32 < 0.003966
        - 0.01268616 * max(0.0, Q.jet_abs_eta - 0.5325716) / 0.3171687   # -1.3%  jet_abs_eta > 0.5326
        + 0.01221698 * max(0.0, 83.57316 - Q.mass_top40) / 3.715787   # +1.2%  mass_top40 < 83.57
        + 0.01131138 * max(0.0, 2.151069 - Q.sj3_mass3) / 0.3356229   # +1.1%  sj3_mass3 < 2.151
        - 0.01024675 * max(0.0, Q.tau32_b2 - 0.5838518) / 0.08955437   # -1.0%  tau32_b2 > 0.5839
        + 0.009741893 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +1.0%  n_dr_0p4_up < 4
        - 0.007535406 * max(0.0, 0.04863552 - Q.sum_z_dr) / 0.000892581   # -0.8%  sum_z_dr < 0.04864
        + 0.007444565 * max(0.0, 16.0 - Q.n_pairs_kt_above_3) / 1.257407   # +0.7%  n_pairs_kt_above_3 < 16
        + 0.006719453 * max(0.0, 17.3274 - Q.m01) * max(0.0, 0.3166168 - Q.N2_b2) / 1.247339   # +0.7%  m01 < 17.33 and N2_b2 < 0.3166
        - 0.006701215 * max(0.0, 2.961008 - Q.lund_max_lnkt) / 0.07658753   # -0.7%  lund_max_lnkt < 2.961
        - 0.006388954 * max(0.0, 95.14961 - Q.mass) / 6.371707   # -0.6%  mass < 95.15
        + 0.006309712 * max(0.0, 149.0507 - Q.mass) * max(0.0, 0.5498426 - Q.tau32) / 0.9230671   # +0.6%  mass < 149.1 and tau32 < 0.5498
        - 0.006243443 * max(0.0, 2.151069 - Q.sj3_mass3) * max(0.0, 0.03350185 - Q.sj4_zsoft) / 0.005318074   # -0.6%  sj3_mass3 < 2.151 and sj4_zsoft < 0.0335
        - 0.005603827 * max(0.0, 0.1301769 - Q.M2_b05) * max(0.0, 9.0 - Q.n_dr_0p1_0p2) / 0.03637397   # -0.6%  M2_b05 < 0.1302 and n_dr_0p1_0p2 < 9
        + 0.005424943 * max(0.0, 0.003001458 - Q.ecf_g32) / 0.001159814   # +0.5%  ecf_g32 < 0.003001
        + 0.005226809 * max(0.0, Q.max_pair_mass - 45.28191) / 1.37267   # +0.5%  max_pair_mass > 45.28
        + 0.005190205 * max(0.0, Q.sj3_pair_mass_min - 68.11898) / 1.584065   # +0.5%  sj3_pair_mass_min > 68.12
        - 0.005042129 * max(0.0, 0.05997694 - Q.tau2) / 0.009989817   # -0.5%  tau2 < 0.05998
        + 0.004839897 * max(0.0, 1.208316 - Q.D2_b05) / 0.01444539   # +0.5%  D2_b05 < 1.208
        + 0.004652447 * max(0.0, 0.02982432 - Q.dr02) / 0.005753371   # +0.5%  dr02 < 0.02982
        + 0.004284391 * max(0.0, Q.z_top2_slots - 0.3197804) / 0.09263154   # +0.4%  z_top2_slots > 0.3198
        - 0.004271578 * max(0.0, 0.07290954 - Q.sum_z_dr) / 0.002660742   # -0.4%  sum_z_dr < 0.07291
        - 0.0038868 * max(0.0, 732.7973 - Q.sum_e) / 42.60421   # -0.4%  sum_e < 732.8
        - 0.003875685 * max(0.0, Q.lund_max_lndelta - -0.6897565) / 0.03634831   # -0.4%  lund_max_lndelta > -0.6898
        + 0.003692257 * max(0.0, -1.332584 - Q.pair_mean_lnz) * max(0.0, Q.ecf_g41 - 0.0001767103) / 1.975501e-05   # +0.4%  pair_mean_lnz < -1.333 and ecf_g41 > 0.0001767
        - 0.003352032 * max(0.0, 0.04276413 - Q.M2) / 0.0009967452   # -0.3%  M2 < 0.04276
        + 0.002762672 * max(0.0, Q.sj3_z1 - 0.4593088) * max(0.0, 0.5444444 - Q.dr_13) / 0.05309438   # +0.3%  sj3_z1 > 0.4593 and dr_13 < 0.5444
        - 0.002719735 * max(0.0, Q.sj3_pair_mass_min - 80.02563) / 0.702582   # -0.3%  sj3_pair_mass_min > 80.03
        + 0.002694676 * max(0.0, Q.sj2_mass1 - 67.99453) / 3.209722   # +0.3%  sj2_mass1 > 67.99
        - 0.002604759 * max(0.0, 0.3150493 - Q.sj3_dr_max) / 0.01079671   # -0.3%  sj3_dr_max < 0.315
        - 0.002587888 * max(0.0, 428.9828 - Q.sum_pt_top10) / 28.27138   # -0.3%  sum_pt_top10 < 429
        - 0.002577694 * max(0.0, Q.m012 - 57.33311) * max(0.0, 8.588303 - Q.sj2_mass2) / 2.780324   # -0.3%  m012 > 57.33 and sj2_mass2 < 8.588
        + 0.002532103 * max(0.0, Q.z_top3_slots - 0.7102909) / 0.008607941   # +0.3%  z_top3_slots > 0.7103
        - 0.00240739 * max(0.0, 0.4353632 - Q.N2_b05) / 0.02763283   # -0.2%  N2_b05 < 0.4354
        - 0.002403109 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) * max(0.0, Q.mratio_max_012 - 0.7784366) / 0.7471323   # -0.2%  sj3_pair_mass_max < 114.8 and mratio_max_012 > 0.7784
        + 0.002375778 * max(0.0, 0.0259406 - Q.dr01) / 0.005036114   # +0.2%  dr01 < 0.02594
        - 0.002258903 * max(0.0, 15.07073 - Q.pt1_dr01) * max(0.0, 0.8939856 - Q.tau43) / 0.7254282   # -0.2%  pt1_dr01 < 15.07 and tau43 < 0.894
        - 0.002176656 * max(0.0, Q.tau21 - 0.3552591) / 0.1282842   # -0.2%  tau21 > 0.3553
        + 0.002042742 * max(0.0, Q.lund_max_lndelta - -0.6897565) * max(0.0, Q.lund3_lnz - -6.170656) / 0.08251702   # +0.2%  lund_max_lndelta > -0.6898 and lund3_lnz > -6.171
        - 0.001990825 * max(0.0, 2.151069 - Q.sj3_mass3) * max(0.0, 3.928781 - Q.sj2_mass2) / 0.2334801   # -0.2%  sj3_mass3 < 2.151 and sj2_mass2 < 3.929
        - 0.001976047 * max(0.0, 17.3274 - Q.m01) * max(0.0, 0.08435059 - Q.eta_17) / 1.094361   # -0.2%  m01 < 17.33 and eta_17 < 0.08435
        + 0.001815978 * max(0.0, Q.pair_mean_lnm2 - 4.787589) / 0.03990488   # +0.2%  pair_mean_lnm2 > 4.788
        - 0.00173388 * max(0.0, 0.02497878 - Q.M3_b2) * max(0.0, 0.0007909605 - Q.e3_b2) / 7.871547e-06   # -0.2%  M3_b2 < 0.02498 and e3_b2 < 0.000791
        - 0.001621861 * max(0.0, Q.m012 - 57.33311) / 1.633981   # -0.2%  m012 > 57.33
        - 0.001576005 * max(0.0, 142.9952 - Q.sj3_pair_mass_max) / 52.03908   # -0.2%  sj3_pair_mass_max < 143
        + 0.00157127 * max(0.0, 1.208316 - Q.D2_b05) * max(0.0, 0.0657426 - Q.sj4_dr_min) / 0.0002643703   # +0.2%  D2_b05 < 1.208 and sj4_dr_min < 0.06574
        - 0.001550348 * max(0.0, 15.07073 - Q.pt1_dr01) * max(0.0, 6.0 - Q.n_lund_kt_above_1) / 8.906999   # -0.2%  pt1_dr01 < 15.07 and n_lund_kt_above_1 < 6
        + 0.001528295 * max(0.0, Q.sj2_dr - 0.5076533) / 0.02194993   # +0.2%  sj2_dr > 0.5077
        - 0.001522405 * max(0.0, 0.04276413 - Q.M2) * max(0.0, Q.lnpt_6 - 2.807141) / 0.0001498569   # -0.2%  M2 < 0.04276 and lnpt_6 > 2.807
        + 0.001469096 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) * max(0.0, Q.pair_max_lnkt - 2.687142) / 4.212269   # +0.1%  sj3_pair_mass_max < 114.8 and pair_max_lnkt > 2.687
        + 0.001450279 * max(0.0, 1.87753 - Q.m012) / 0.05753931   # +0.1%  m012 < 1.878
        + 0.001430876 * max(0.0, Q.lne_29 - 1.660458) / 0.1174236   # +0.1%  lne_29 > 1.66
        + 0.00141597 * max(0.0, 0.004077497 - Q.sum_z_dr2_top3) / 0.0004626356   # +0.1%  sum_z_dr2_top3 < 0.004077
        + 0.00139019 * max(0.0, Q.sj3_pairmax_over_m - 0.6888277) * max(0.0, Q.sj2_mass2 - 33.1786) / 0.09375834   # +0.1%  sj3_pairmax_over_m > 0.6888 and sj2_mass2 > 33.18
        + 0.001377368 * max(0.0, Q.sj2_dr - 0.5076533) * max(0.0, Q.lund2_lnz - -4.435851) / 0.02441234   # +0.1%  sj2_dr > 0.5077 and lund2_lnz > -4.436
        - 0.001327711 * max(0.0, 0.1824346 - Q.N2) / 0.003903698   # -0.1%  N2 < 0.1824
        - 0.001290483 * max(0.0, Q.m012 - 69.67489) / 0.7542893   # -0.1%  m012 > 69.67
        + 0.001229261 * max(0.0, 149.0507 - Q.mass) * max(0.0, 0.4242331 - Q.pt2_over_pt0) / 3.489843   # +0.1%  mass < 149.1 and pt2_over_pt0 < 0.4242
        + 0.0008996901 * max(0.0, 4.0 - Q.n_dr_0p4_up) * max(0.0, Q.phi_24 - 0.05783997) / 0.03116743   # +0.1%  n_dr_0p4_up < 4 and phi_24 > 0.05784
        + 0.0008849367 * max(0.0, Q.n_lund - 10.0) / 1.682617   # +0.1%  n_lund > 10
        - 0.000806948 * max(0.0, Q.m012 - 69.67489) * max(0.0, 0.09604646 - Q.sj4_dr_min) / 0.01577993   # -0.1%  m012 > 69.67 and sj4_dr_min < 0.09605
        - 0.0006956737 * max(0.0, Q.m012 - 69.67489) * max(0.0, -3.686417 - Q.lund3_lnz) / 2.376917   # -0.1%  m012 > 69.67 and lund3_lnz < -3.686
        + 0.0006296317 * max(0.0, 104.3234 - Q.mass_top50) * max(0.0, 0.6363796 - Q.tau43) / 0.0719295   # +0.1%  mass_top50 < 104.3 and tau43 < 0.6364
        - 0.0005836194 * max(0.0, Q.sj3_z1 - 0.4593088) / 0.1565176   # -0.1%  sj3_z1 > 0.4593
        + 0.0005579652 * max(0.0, 0.1824346 - Q.N2) * max(0.0, 0.2460229 - Q.dr_13) / 0.0003235987   # +0.1%  N2 < 0.1824 and dr_13 < 0.246
        + 0.000514573 * max(0.0, Q.max_pair_mass - 45.28191) * max(0.0, 3.175446 - Q.lne_4) / 0.2289821   # +0.1%  max_pair_mass > 45.28 and lne_4 < 3.175
        - 0.0004874766 * max(0.0, Q.m012 - 69.67489) * max(0.0, Q.sj2_mass2 - 1.851735) / 12.0823   # -0.0%  m012 > 69.67 and sj2_mass2 > 1.852
        - 0.0004232512 * max(0.0, Q.max_pair_mass - 45.28191) * max(0.0, 0.7613542 - Q.lund1_lnkt) / 0.3249134   # -0.0%  max_pair_mass > 45.28 and lund1_lnkt < 0.7614
        - 0.000389139 * max(0.0, 2.09826 - Q.pair_mean_lnm2) / 0.2329459   # -0.0%  pair_mean_lnm2 < 2.098
        + 0.0003466667 * max(0.0, Q.lund_max_lndelta - -0.6897565) * max(0.0, Q.e4 - 1.2913e-08) / 2.367302e-07   # +0.0%  lund_max_lndelta > -0.6898 and e4 > 1.291e-08
        + 0.0003064577 * max(0.0, Q.m012 - 57.33311) * max(0.0, 0.1115643 - Q.sj3_dr13) / 0.007600278   # +0.0%  m012 > 57.33 and sj3_dr13 < 0.1116
        - 0.00029892 * max(0.0, 0.0259406 - Q.dr01) * max(0.0, Q.lund3_lnkt - 3.765889) / 0.0002008213   # -0.0%  dr01 < 0.02594 and lund3_lnkt > 3.766
        + 0.0002860059 * max(0.0, 732.7973 - Q.sum_e) * max(0.0, Q.C3_b2 - 2.435109e-06) / 0.5525005   # +0.0%  sum_e < 732.8 and C3_b2 > 2.435e-06
        + 0.0002759175 * max(0.0, Q.sj2_dr - 0.6647889) / 0.004152297   # +0.0%  sj2_dr > 0.6648
        + 0.0002209431 * max(0.0, 0.07290954 - Q.sum_z_dr) * max(0.0, 0.003536765 - Q.C3_b2) / 4.649996e-06   # +0.0%  sum_z_dr < 0.07291 and C3_b2 < 0.003537
        - 8.552407e-05 * max(0.0, 732.7973 - Q.sum_e) * max(0.0, 1.851735 - Q.sj2_mass2) / 4.097722   # -0.0%  sum_e < 732.8 and sj2_mass2 < 1.852
        + 6.51111e-05 * max(0.0, Q.tau21 - 0.3552591) * max(0.0, -0.3098145 - Q.phi_13) / 0.0009582121   # +0.0%  tau21 > 0.3553 and phi_13 < -0.3098
        - 5.465753e-05 * max(0.0, Q.sj2_dr - 0.6647889) * max(0.0, 0.0 - Q.phi_51) / 0.0001128924   # -0.0%  sj2_dr > 0.6648 and phi_51 < 0
        + 4.943923e-05 * max(0.0, 92.68149 - Q.mass_top40) / 5.990263   # +0.0%  mass_top40 < 92.68
        + 4.59798e-05 * max(0.0, Q.D3 - 0.3195268) * max(0.0, 0.0 - Q.eta_40) / 0.000346521   # +0.0%  D3 > 0.3195 and eta_40 < 0
        - 4.319149e-05 * max(0.0, Q.z_top2_slots - 0.3197804) * max(0.0, -0.01504822 - Q.eta_23) / 0.00416548   # -0.0%  z_top2_slots > 0.3198 and eta_23 < -0.01505
        + 3.631892e-05 * max(0.0, 2.09826 - Q.pair_mean_lnm2) * max(0.0, Q.lund1_lnz - -3.742431) / 0.1027051   # +0.0%  pair_mean_lnm2 < 2.098 and lund1_lnz > -3.742
    )
    return z


def neuron_5(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.0006186547
    )
    return z


def neuron_6(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.169769e-05
    )
    return z


def neuron_7(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.171535e-05
    )
    return z


def neuron_8(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.341724e-05
    )
    return z


def neuron_9(Q):
    # scale S = 4.249; each line: share * term / its average size
    z = 4.248972 * (0.05064078
        + 0.08962063 * max(0.0, Q.N2_b05 - 0.302405) / 0.1383194   # +9.0%  N2_b05 > 0.3024
        - 0.0786881 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -7.9%  mass < 164.4
        - 0.05267568 * max(0.0, 0.9148508 - Q.sj3_pairmax_over_m) / 0.1113978   # -5.3%  sj3_pairmax_over_m < 0.9149
        + 0.05134211 * max(0.0, 120.6471 - Q.sj3_pair_mass_max) / 31.70826   # +5.1%  sj3_pair_mass_max < 120.6
        + 0.04973894 * max(0.0, 0.2701529 - Q.tau21_b2) / 0.06119609   # +5.0%  tau21_b2 < 0.2702
        - 0.0368488 * max(0.0, Q.tau5 - 0.01467699) / 0.01818842   # -3.7%  tau5 > 0.01468
        - 0.03581894 * max(0.0, 88.0 - Q.n_pairs_kt_above_3) / 31.77884   # -3.6%  n_pairs_kt_above_3 < 88
        + 0.03544751 * max(0.0, Q.pair_mean_lnm2 - 1.827583) / 0.8979862   # +3.5%  pair_mean_lnm2 > 1.828
        - 0.03150822 * max(0.0, 0.3829918 - Q.N2) / 0.08586977   # -3.2%  N2 < 0.383
        + 0.03081329 * max(0.0, 117.4867 - Q.mass) / 15.64803   # +3.1%  mass < 117.5
        - 0.02603317 * max(0.0, Q.N2_b05 - 0.4265629) / 0.03557833   # -2.6%  N2_b05 > 0.4266
        - 0.02384009 * max(0.0, Q.n_particles - 26.0) / 14.5103   # -2.4%  n_particles > 26
        - 0.02300149 * max(0.0, 0.3829918 - Q.N2) * max(0.0, 0.09477716 - Q.dr_47) / 0.007225319   # -2.3%  N2 < 0.383 and dr_47 < 0.09478
        - 0.02238422 * max(0.0, Q.sd_mass - 135.3031) / 5.717996   # -2.2%  sd_mass > 135.3
        + 0.01914118 * max(0.0, Q.sd_mass - 127.8042) / 7.009137   # +1.9%  sd_mass > 127.8
        - 0.01815618 * max(0.0, 95.14961 - Q.mass) / 6.371707   # -1.8%  mass < 95.15
        - 0.015901 * max(0.0, 0.1259759 - Q.sj3_dr_min) / 0.01757215   # -1.6%  sj3_dr_min < 0.126
        + 0.01534592 * max(0.0, Q.tau5 - 0.03167074) / 0.007201204   # +1.5%  tau5 > 0.03167
        - 0.01458701 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # -1.5%  sd_mass > 94.56
        + 0.01369106 * max(0.0, 120.6471 - Q.sj3_pair_mass_max) * max(0.0, 12.0 - Q.n_dr_0p05_0p1) / 176.7552   # +1.4%  sj3_pair_mass_max < 120.6 and n_dr_0p05_0p1 < 12
        - 0.01362006 * max(0.0, 0.2701529 - Q.tau21_b2) * max(0.0, 0.02273044 - Q.dr_31) / 0.0007597281   # -1.4%  tau21_b2 < 0.2702 and dr_31 < 0.02273
        + 0.01313821 * max(0.0, Q.tau21 - 0.4790917) / 0.06636012   # +1.3%  tau21 > 0.4791
        + 0.01261291 * max(0.0, Q.sd_mass - 111.701) / 11.78149   # +1.3%  sd_mass > 111.7
        - 0.01200729 * max(0.0, 0.2116473 - Q.dr12) / 0.117423   # -1.2%  dr12 < 0.2116
        - 0.01102411 * max(0.0, 0.6079631 - Q.max_dr) / 0.05926836   # -1.1%  max_dr < 0.608
        + 0.01068935 * max(0.0, 0.01520104 - Q.sum_z_dr2_top3) / 0.004537457   # +1.1%  sum_z_dr2_top3 < 0.0152
        - 0.01035222 * max(0.0, 0.03086713 - Q.tau3) / 0.00283235   # -1.0%  tau3 < 0.03087
        - 0.009415216 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.lund_max_lndelta - -1.498181) / 4.181858   # -0.9%  mass < 117.5 and lund_max_lndelta > -1.498
        - 0.009351122 * max(0.0, Q.pair_mean_lndelta - -2.42577) / 0.3779626   # -0.9%  pair_mean_lndelta > -2.426
        + 0.009213259 * max(0.0, 0.1193588 - Q.tau2) / 0.04844327   # +0.9%  tau2 < 0.1194
        - 0.008960251 * max(0.0, 0.5969141 - Q.pt_dispersion) * max(0.0, 1651.766 - Q.sum_e) / 200.6166   # -0.9%  pt_dispersion < 0.5969 and sum_e < 1652
        + 0.008958411 * max(0.0, Q.mass_top20 - 130.7093) * max(0.0, 1.380731 - Q.D2_b2) / 0.7302023   # +0.9%  mass_top20 > 130.7 and D2_b2 < 1.381
        + 0.008783075 * max(0.0, 109.8208 - Q.sj3_pair_mass_max) / 23.01361   # +0.9%  sj3_pair_mass_max < 109.8
        - 0.008589651 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 1.43511   # -0.9%  n_dr_0p2_0p4 < 7
        + 0.008497632 * max(0.0, Q.lund_max_lnkt - 3.22013) / 0.7220204   # +0.8%  lund_max_lnkt > 3.22
        - 0.007501778 * max(0.0, Q.lne_0 - 5.157617) / 0.277972   # -0.8%  lne_0 > 5.158
        + 0.007314359 * max(0.0, Q.e3_b05 - 0.0061892) / 0.002290521   # +0.7%  e3_b05 > 0.006189
        + 0.007262537 * max(0.0, 117.4867 - Q.mass) * max(0.0, -0.3023635 - Q.lund1_lndelta) / 6.482175   # +0.7%  mass < 117.5 and lund1_lndelta < -0.3024
        - 0.006873489 * max(0.0, Q.mass_top20 - 130.7093) / 2.260459   # -0.7%  mass_top20 > 130.7
        + 0.006686087 * max(0.0, 0.6079631 - Q.max_dr) * max(0.0, Q.lund3_lndelta - -2.817283) / 0.04146321   # +0.7%  max_dr < 0.608 and lund3_lndelta > -2.817
        + 0.006652961 * max(0.0, Q.lnpt_6 - 2.880777) / 0.2833407   # +0.7%  lnpt_6 > 2.881
        - 0.006450175 * max(0.0, 0.2701529 - Q.tau21_b2) * max(0.0, Q.lne_6 - 2.784239) / 0.03982471   # -0.6%  tau21_b2 < 0.2702 and lne_6 > 2.784
        - 0.006276989 * max(0.0, Q.mass_top30 - 162.7874) * max(0.0, 1.380731 - Q.D2_b2) / 0.4020949   # -0.6%  mass_top30 > 162.8 and D2_b2 < 1.381
        + 0.006144564 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +0.6%  n_pairs_kt_above_1 < 80
        + 0.005124808 * max(0.0, 0.0153855 - Q.dr01) / 0.00178836   # +0.5%  dr01 < 0.01539
        - 0.005076261 * max(0.0, 175.9333 - Q.sd_mass) / 80.62387   # -0.5%  sd_mass < 175.9
        - 0.00474868 * max(0.0, Q.mass_over_sum_pt_sq - 0.04296187) / 0.008561693   # -0.5%  mass_over_sum_pt_sq > 0.04296
        + 0.004285944 * max(0.0, 0.2701529 - Q.tau21_b2) * max(0.0, 1.851735 - Q.sj2_mass2) / 0.006034104   # +0.4%  tau21_b2 < 0.2702 and sj2_mass2 < 1.852
        + 0.004280036 * max(0.0, 95.14961 - Q.mass) * max(0.0, Q.lnerel_0 - -1.362996) / 1.744454   # +0.4%  mass < 95.15 and lnerel_0 > -1.363
        + 0.004143596 * max(0.0, Q.sj4_pair_mass_max - 115.5142) / 2.08655   # +0.4%  sj4_pair_mass_max > 115.5
        - 0.004117729 * max(0.0, Q.n_particles - 26.0) * max(0.0, 1.070218 - Q.lne_49) / 121.9195   # -0.4%  n_particles > 26 and lne_49 < 1.07
        - 0.003966548 * max(0.0, 0.9123375 - Q.m01) / 0.05269771   # -0.4%  m01 < 0.9123
        - 0.003612904 * max(0.0, Q.pair_mean_lndelta - -1.629715) / 0.03696365   # -0.4%  pair_mean_lndelta > -1.63
        - 0.003588706 * max(0.0, 1.824785 - Q.sj3_mass2) / 0.06171879   # -0.4%  sj3_mass2 < 1.825
        + 0.003365305 * max(0.0, 2.43449 - Q.pair_max_lnkt) / 0.1466106   # +0.3%  pair_max_lnkt < 2.434
        + 0.003219934 * max(0.0, 0.2701529 - Q.tau21_b2) * max(0.0, 1.462588 - Q.D3_b2) / 0.06066647   # +0.3%  tau21_b2 < 0.2702 and D3_b2 < 1.463
        - 0.002958902 * max(0.0, 0.2116473 - Q.dr12) * max(0.0, Q.eta_1 - -0.05657959) / 0.007433029   # -0.3%  dr12 < 0.2116 and eta_1 > -0.05658
        - 0.002793861 * max(0.0, 5.0 - Q.n_dr_0p4_up) / 1.758997   # -0.3%  n_dr_0p4_up < 5
        + 0.002747562 * max(0.0, Q.C2_b05 - 0.3533901) / 0.003974948   # +0.3%  C2_b05 > 0.3534
        + 0.002731316 * max(0.0, Q.mass_top30 - 162.7874) / 1.306268   # +0.3%  mass_top30 > 162.8
        + 0.002381495 * max(0.0, 0.1854236 - Q.C2_b05) / 0.002874153   # +0.2%  C2_b05 < 0.1854
        + 0.002317313 * max(0.0, Q.ecf_g42 - 2.829762e-05) / 9.057051e-06   # +0.2%  ecf_g42 > 2.83e-05
        + 0.002285878 * max(0.0, Q.lund_max_lnkt - 4.323396) * max(0.0, Q.sj3_dr_min - 0.07110203) / 0.01246881   # +0.2%  lund_max_lnkt > 4.323 and sj3_dr_min > 0.0711
        - 0.002188697 * max(0.0, 0.3829918 - Q.N2) * max(0.0, Q.dr_31 - 0.0) / 0.01146476   # -0.2%  N2 < 0.383 and dr_31 > 0
        - 0.002031516 * max(0.0, Q.lund_max_lnkt - 4.323396) / 0.06128957   # -0.2%  lund_max_lnkt > 4.323
        - 0.001910483 * max(0.0, 0.3829918 - Q.N2) * max(0.0, Q.lam1 - 0.05242126) / 0.000265899   # -0.2%  N2 < 0.383 and lam1 > 0.05242
        - 0.001874491 * max(0.0, Q.pair_mean_lndelta - -2.049856) * max(0.0, -0.529318 - Q.lund_max_lndelta) / 0.05372291   # -0.2%  pair_mean_lndelta > -2.05 and lund_max_lndelta < -0.5293
        + 0.001810103 * max(0.0, 0.2317874 - Q.sj3_dr13) / 0.02459729   # +0.2%  sj3_dr13 < 0.2318
        + 0.001724387 * max(0.0, Q.lund_max_lnkt - 4.323396) * max(0.0, 4.0 - Q.n_pt_above_50) / 0.1039031   # +0.2%  lund_max_lnkt > 4.323 and n_pt_above_50 < 4
        - 0.001708985 * max(0.0, 0.9148508 - Q.sj3_pairmax_over_m) * max(0.0, Q.D2_b2 - 0.5503445) / 0.4412063   # -0.2%  sj3_pairmax_over_m < 0.9149 and D2_b2 > 0.5503
        + 0.001702387 * max(0.0, 0.9148508 - Q.sj3_pairmax_over_m) * max(0.0, Q.sd_nremoved - 5.0) / 0.01948275   # +0.2%  sj3_pairmax_over_m < 0.9149 and sd_nremoved > 5
        + 0.001609106 * max(0.0, 0.5969141 - Q.pt_dispersion) / 0.2647359   # +0.2%  pt_dispersion < 0.5969
        + 0.00160685 * max(0.0, Q.sd_mass - 135.3031) * max(0.0, 1.380731 - Q.D2_b2) / 1.315249   # +0.2%  sd_mass > 135.3 and D2_b2 < 1.381
        - 0.001596516 * max(0.0, Q.pair_mean_lndelta - -2.049856) / 0.1589658   # -0.2%  pair_mean_lndelta > -2.05
        + 0.001483645 * max(0.0, Q.mass_over_sum_pt_sq - 0.04296187) * max(0.0, -0.8095462 - Q.lund_max_lndelta) / 0.000366308   # +0.1%  mass_over_sum_pt_sq > 0.04296 and lund_max_lndelta < -0.8095
        + 0.001336803 * max(0.0, Q.mass_top20 - 130.7093) * max(0.0, 0.2901133 - Q.sj3_dr_min) / 0.05774644   # +0.1%  mass_top20 > 130.7 and sj3_dr_min < 0.2901
        - 0.001271271 * max(0.0, 0.03086713 - Q.tau3) * max(0.0, Q.phi_29 - -0.3354492) / 0.0009525773   # -0.1%  tau3 < 0.03087 and phi_29 > -0.3354
        - 0.001251161 * max(0.0, Q.sd_mass - 111.701) * max(0.0, 0.191004 - Q.dr01) / 0.8947287   # -0.1%  sd_mass > 111.7 and dr01 < 0.191
        + 0.0009795965 * max(0.0, 0.2116473 - Q.dr12) * max(0.0, Q.dr_10 - 0.3349968) / 0.002882465   # +0.1%  dr12 < 0.2116 and dr_10 > 0.335
        - 0.0009785804 * max(0.0, 0.03465331 - Q.M2) / 0.0004071737   # -0.1%  M2 < 0.03465
        + 0.0009560075 * max(0.0, Q.pair_mean_lndelta - -2.049856) * max(0.0, Q.mratio_max_012 - 0.8957738) / 0.0002699499   # +0.1%  pair_mean_lndelta > -2.05 and mratio_max_012 > 0.8958
        + 0.0007787463 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.mass_top2 - 35.44963) / 71.43292   # +0.1%  mass < 164.4 and mass_top2 > 35.45
        - 0.000328575 * max(0.0, Q.lund_max_lnkt - 4.323396) * max(0.0, 0.3125 - Q.phi_14) / 0.01962841   # -0.0%  lund_max_lnkt > 4.323 and phi_14 < 0.3125
        - 0.0002757331 * max(0.0, 0.2701529 - Q.tau21_b2) * max(0.0, Q.max_pair_mass - 45.28191) / 0.1832484   # -0.0%  tau21_b2 < 0.2702 and max_pair_mass > 45.28
        - 0.0002620673 * max(0.0, Q.sj4_pair_mass_max - 92.5542) / 6.396254   # -0.0%  sj4_pair_mass_max > 92.55
        - 0.0002389325 * max(0.0, -0.2019043 - Q.eta_1) / 0.004072277   # -0.0%  eta_1 < -0.2019
        + 0.0002238832 * max(0.0, 2.0 - Q.n_lund_kt_above_1) / 0.01792   # +0.0%  n_lund_kt_above_1 < 2
        - 0.0001806877 * max(0.0, Q.lnpt_6 - 2.880777) * max(0.0, Q.phi_17 - 0.07128906) / 0.01058555   # -0.0%  lnpt_6 > 2.881 and phi_17 > 0.07129
        - 0.0001802376 * max(0.0, 1.824785 - Q.sj3_mass2) * max(0.0, Q.sj2_mass2 - 5.311506) / 0.2192101   # -0.0%  sj3_mass2 < 1.825 and sj2_mass2 > 5.312
        + 0.0001458266 * max(0.0, 2.0 - Q.n_lund_kt_above_1) * max(0.0, -0.1276855 - Q.eta_19) / 0.0002895898   # +0.0%  n_lund_kt_above_1 < 2 and eta_19 < -0.1277
        + 0.0001363955 * max(0.0, Q.sd_rg - 0.4449101) / 0.02724738   # +0.0%  sd_rg > 0.4449
        + 0.0001087984 * max(0.0, 0.2116473 - Q.dr12) * max(0.0, Q.phi_47 - -0.07501221) / 0.01069012   # +0.0%  dr12 < 0.2116 and phi_47 > -0.07501
        + 0.0001058802 * max(0.0, 0.2701529 - Q.tau21_b2) * max(0.0, Q.eta_0 - 0.1077942) / 0.0004428031   # +0.0%  tau21_b2 < 0.2702 and eta_0 > 0.1078
        + 7.887452e-05 * max(0.0, 0.03465331 - Q.M2) * max(0.0, Q.phi_27 - 0.0) / 1.837542e-06   # +0.0%  M2 < 0.03465 and phi_27 > 0
        + 6.53724e-05 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.sj3_mass1 - 13.49356) / 295.674   # +0.0%  mass < 164.4 and sj3_mass1 > 13.49
        + 5.243748e-05 * max(0.0, Q.lund_max_lndelta - -0.4168051) / 0.004389625   # +0.0%  lund_max_lndelta > -0.4168
        - 3.97159e-05 * max(0.0, Q.tau5 - 0.01467699) * max(0.0, 1.851735 - Q.sj2_mass2) / 0.0003139224   # -0.0%  tau5 > 0.01468 and sj2_mass2 < 1.852
        - 3.835476e-05 * max(0.0, 2.0 - Q.n_lund_kt_above_1) * max(0.0, -0.3310547 - Q.phi_17) / 8.961019e-05   # -0.0%  n_lund_kt_above_1 < 2 and phi_17 < -0.3311
        + 2.47446e-05 * max(0.0, 0.2701529 - Q.tau21_b2) * max(0.0, Q.phi_0 - 0.1651611) / 0.0002309241   # +0.0%  tau21_b2 < 0.2702 and phi_0 > 0.1652
        - 8.125395e-06 * max(0.0, 0.03465331 - Q.M2) * max(0.0, 0.0 - Q.eta_19) / 1.21742e-05   # -0.0%  M2 < 0.03465 and eta_19 < 0
    )
    return z


def neuron_10(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.742235e-06
    )
    return z


def neuron_11(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.878501e-06
    )
    return z


def neuron_12(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.628307e-05
    )
    return z


def neuron_13(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.68468e-06
    )
    return z


def neuron_14(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.188985e-06
    )
    return z


def neuron_15(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.038257e-05
    )
    return z


def neuron_16(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.221253e-06
    )
    return z


def neuron_17(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.269395e-05
    )
    return z


def neuron_18(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.348156e-05
    )
    return z


def neuron_19(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.553789e-05
    )
    return z


def neuron_20(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.223184e-05
    )
    return z


def neuron_21(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.048267e-06
    )
    return z


def neuron_22(Q):
    # scale S = 12.48; each line: share * term / its average size
    z = 12.47719 * (0.1191731
        - 0.1043775 * max(0.0, Q.mass - 85.02716) / 33.73392   # -10.4%  mass > 85.03
        - 0.09387247 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -9.4%  mass < 164.4
        - 0.0799997 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # -8.0%  sd_mass > 88.82
        + 0.04336004 * max(0.0, 0.23825 - Q.C2) / 0.09330487   # +4.3%  C2 < 0.2383
        - 0.04195297 * max(0.0, 0.3701694 - Q.dr01) / 0.2308018   # -4.2%  dr01 < 0.3702
        + 0.04065036 * max(0.0, 0.1058963 - Q.dr02) / 0.04190221   # +4.1%  dr02 < 0.1059
        + 0.03841553 * max(0.0, Q.sd_mass - 62.03827) / 42.41619   # +3.8%  sd_mass > 62.04
        + 0.03247956 * max(0.0, 12.51777 - Q.m01) / 5.617849   # +3.2%  m01 < 12.52
        + 0.03237087 * max(0.0, Q.mass - 110.2019) / 16.73354   # +3.2%  mass > 110.2
        - 0.03210575 * max(0.0, 0.0001729927 - Q.e3_b2) / 0.0001069897   # -3.2%  e3_b2 < 0.000173
        + 0.02893211 * max(0.0, 0.05966366 - Q.M2_b2) / 0.0281728   # +2.9%  M2_b2 < 0.05966
        - 0.02786869 * max(0.0, 0.297969 - Q.tau21_b2) / 0.0757859   # -2.8%  tau21_b2 < 0.298
        + 0.02619256 * max(0.0, 35.44963 - Q.m01) / 22.30185   # +2.6%  m01 < 35.45
        - 0.02266013 * max(0.0, 0.07373689 - Q.dr02) / 0.02487709   # -2.3%  dr02 < 0.07374
        + 0.02169225 * max(0.0, Q.sd_mass - 123.2919) / 8.015693   # +2.2%  sd_mass > 123.3
        + 0.02027718 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # +2.0%  sd_mass > 106.8
        + 0.01960325 * max(0.0, Q.n_pairs_kt_above_3 - 47.0) / 43.67114   # +2.0%  n_pairs_kt_above_3 > 47
        - 0.01899571 * max(0.0, 0.3413762 - Q.dr02) / 0.1988175   # -1.9%  dr02 < 0.3414
        - 0.0186099 * max(0.0, 0.828622 - Q.N3_b05) / 0.1941975   # -1.9%  N3_b05 < 0.8286
        + 0.01589806 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) / 143.7916   # +1.6%  n_pairs_kt_above_3 < 220
        + 0.01484804 * max(0.0, 0.5935014 - Q.N3) / 0.1341665   # +1.5%  N3 < 0.5935
        - 0.01348789 * max(0.0, 702.0 - Q.n_pairs_kt_above_1) / 397.4307   # -1.3%  n_pairs_kt_above_1 < 702
        - 0.01277018 * max(0.0, 97.52633 - Q.sj3_pair_mass_max) / 14.66667   # -1.3%  sj3_pair_mass_max < 97.53
        - 0.01233798 * max(0.0, Q.C2_b05 - 0.1620185) / 0.107645   # -1.2%  C2_b05 > 0.162
        + 0.01204626 * max(0.0, Q.pair_mean_lnm2 - 2.708356) / 0.3870149   # +1.2%  pair_mean_lnm2 > 2.708
        + 0.01162393 * max(0.0, Q.pair_max_lnm2 - 5.908788) / 0.8228665   # +1.2%  pair_max_lnm2 > 5.909
        + 0.0104186 * max(0.0, 129.5874 - Q.mass_top50) / 24.20267   # +1.0%  mass_top50 < 129.6
        - 0.009652405 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.0856189 - Q.dr01) / 4.84313   # -1.0%  n_pairs_kt_above_3 < 220 and dr01 < 0.08562
        + 0.00954074 * max(0.0, Q.sj3_pair_mass_min - 44.1643) / 6.009845   # +1.0%  sj3_pair_mass_min > 44.16
        + 0.008333249 * max(0.0, 129.5874 - Q.mass_top50) * max(0.0, 0.297969 - Q.tau21_b2) / 1.696297   # +0.8%  mass_top50 < 129.6 and tau21_b2 < 0.298
        - 0.008279534 * max(0.0, Q.pair_mean_lnm2 - 2.708356) * max(0.0, 0.3219863 - Q.dr_57) / 0.1216276   # -0.8%  pair_mean_lnm2 > 2.708 and dr_57 < 0.322
        - 0.006838835 * max(0.0, Q.n_pairs_kt_above_3 - 47.0) * max(0.0, Q.jet_abs_eta - 0.1941339) / 23.38773   # -0.7%  n_pairs_kt_above_3 > 47 and jet_abs_eta > 0.1941
        - 0.00629063 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) / 103.3282   # -0.6%  n_pairs_kt_above_1 < 325
        - 0.006244154 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # -0.6%  sd_mass > 154.6
        + 0.0057721 * max(0.0, 137.452 - Q.mass) / 29.4899   # +0.6%  mass < 137.5
        - 0.005260009 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.sum_z_dr2_top3 - 0.00878048) / 0.4368809   # -0.5%  mass < 164.4 and sum_z_dr2_top3 > 0.00878
        - 0.005258219 * max(0.0, Q.mass_top50 - 115.614) / 12.67421   # -0.5%  mass_top50 > 115.6
        - 0.004809974 * max(0.0, Q.n_pairs_kt_above_3 - 88.0) / 24.56355   # -0.5%  n_pairs_kt_above_3 > 88
        - 0.00464053 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # -0.5%  n_pairs_kt_above_3 < 28
        - 0.004598601 * max(0.0, 47.0 - Q.n_particles) / 10.95631   # -0.5%  n_particles < 47
        + 0.004212969 * max(0.0, 0.1432984 - Q.sj3_dr_min) / 0.02409843   # +0.4%  sj3_dr_min < 0.1433
        + 0.003459801 * max(0.0, Q.n_pairs_kt_above_3 - 88.0) * max(0.0, Q.jet_abs_eta - 0.3274899) / 10.33515   # +0.3%  n_pairs_kt_above_3 > 88 and jet_abs_eta > 0.3275
        - 0.003112619 * max(0.0, 12.51777 - Q.m01) * max(0.0, 0.3956483 - Q.sj3_dr_min) / 1.10551   # -0.3%  m01 < 12.52 and sj3_dr_min < 0.3956
        + 0.002877388 * max(0.0, Q.psi_0p2 - 0.865232) / 0.02605204   # +0.3%  psi_0p2 > 0.8652
        + 0.002429941 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.sj3_pairmin_over_m - 0.3081017) / 6.88567   # +0.2%  n_pairs_kt_above_3 < 220 and sj3_pairmin_over_m > 0.3081
        + 0.002222798 * max(0.0, Q.e2 - 0.1081051) / 0.004484375   # +0.2%  e2 > 0.1081
        + 0.002169211 * max(0.0, Q.sj3_pair_mass_min - 44.1643) * max(0.0, 21.93246 - Q.sj3_mass2) / 36.09481   # +0.2%  sj3_pair_mass_min > 44.16 and sj3_mass2 < 21.93
        + 0.002086369 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.009042742 - Q.sum_z_dr2_top2) / 0.01700331   # +0.2%  n_pairs_kt_above_3 < 28 and sum_z_dr2_top2 < 0.009043
        + 0.002018098 * max(0.0, Q.n_pt_above_1 - 35.0) / 7.55322   # +0.2%  n_pt_above_1 > 35
        + 0.001956963 * max(0.0, Q.mass_top50 - 115.614) * max(0.0, 0.9206713 - Q.max_dr) / 2.247192   # +0.2%  mass_top50 > 115.6 and max_dr < 0.9207
        + 0.001835179 * max(0.0, Q.jet_abs_eta - 1.465946) / 0.02086851   # +0.2%  jet_abs_eta > 1.466
        - 0.001806437 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.4383089 - Q.sj3_dr13) / 0.5411985   # -0.2%  n_pairs_kt_above_3 < 28 and sj3_dr13 < 0.4383
        - 0.001667322 * max(0.0, Q.sj3_pair_mass_min - 44.1643) * max(0.0, 0.07749559 - Q.tau4) / 0.1344034   # -0.2%  sj3_pair_mass_min > 44.16 and tau4 < 0.0775
        + 0.001654291 * max(0.0, 35.44963 - Q.m01) * max(0.0, Q.z_2nd - 0.07694751) / 1.149652   # +0.2%  m01 < 35.45 and z_2nd > 0.07695
        + 0.001635865 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.3850339 - Q.psi_0p1) / 16.95156   # +0.2%  n_pairs_kt_above_3 < 220 and psi_0p1 < 0.385
        + 0.001565658 * max(0.0, Q.pair_mean_lnm2 - 2.708356) * max(0.0, 0.1631992 - Q.sj4_dr_min) / 0.0227674   # +0.2%  pair_mean_lnm2 > 2.708 and sj4_dr_min < 0.1632
        - 0.0015229 * max(0.0, 0.08335692 - Q.sj3_dr_min) / 0.005612375   # -0.2%  sj3_dr_min < 0.08336
        + 0.001384272 * max(0.0, Q.pair_mean_lnm2 - 2.708356) * max(0.0, Q.lne_1 - 4.222261) / 0.1843239   # +0.1%  pair_mean_lnm2 > 2.708 and lne_1 > 4.222
        + 0.001372322 * max(0.0, 2.961008 - Q.lund_max_lnkt) / 0.07658753   # +0.1%  lund_max_lnkt < 2.961
        + 0.001308319 * max(0.0, 271.1875 - Q.sum_pt_top2) / 67.44534   # +0.1%  sum_pt_top2 < 271.2
        - 0.00126501 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.sd_zg - 0.2652027) / 8.121777   # -0.1%  n_pairs_kt_above_3 < 220 and sd_zg > 0.2652
        - 0.00121632 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.2792084 - Q.dr_2) / 0.6800534   # -0.1%  n_pairs_kt_above_3 < 28 and dr_2 < 0.2792
        + 0.001203187 * max(0.0, 2.151069 - Q.sj3_mass3) / 0.3356229   # +0.1%  sj3_mass3 < 2.151
        - 0.001148178 * max(0.0, 2.151069 - Q.sj3_mass3) * max(0.0, Q.M2_b2 - 0.01934939) / 0.002130897   # -0.1%  sj3_mass3 < 2.151 and M2_b2 > 0.01935
        + 0.00113833 * max(0.0, 129.5874 - Q.mass_top50) * max(0.0, Q.ecf_g42 - 1.451882e-05) / 0.0001491842   # +0.1%  mass_top50 < 129.6 and ecf_g42 > 1.452e-05
        - 0.001133492 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.075147   # -0.1%  n_dr_0p2_0p4 < 6
        - 0.001127488 * max(0.0, Q.C2_b05 - 0.1620185) * max(0.0, 0.0004392119 - Q.ecf_g41) / 2.392481e-05   # -0.1%  C2_b05 > 0.162 and ecf_g41 < 0.0004392
        - 0.001086451 * max(0.0, Q.sj3_pair_mass_min - 80.02563) / 0.702582   # -0.1%  sj3_pair_mass_min > 80.03
        + 0.001028196 * max(0.0, Q.sj4_pair_mass_max - 48.76729) * max(0.0, Q.N3 - 0.5935014) / 3.485264   # +0.1%  sj4_pair_mass_max > 48.77 and N3 > 0.5935
        - 0.001014907 * max(0.0, 0.297969 - Q.tau21_b2) * max(0.0, Q.sj2_mass2 - 7.54918) / 0.4001559   # -0.1%  tau21_b2 < 0.298 and sj2_mass2 > 7.549
        - 0.0009912629 * max(0.0, Q.tau1 - 0.2694895) / 0.008456149   # -0.1%  tau1 > 0.2695
        - 0.0009857118 * max(0.0, 0.07373689 - Q.dr02) * max(0.0, 7.782712 - Q.D2_b2) / 0.1186596   # -0.1%  dr02 < 0.07374 and D2_b2 < 7.783
        - 0.0008715034 * max(0.0, 0.009257989 - Q.M3_b2) / 0.001384941   # -0.1%  M3_b2 < 0.009258
        - 0.0008603798 * max(0.0, Q.pair_max_lnm2 - 5.908788) * max(0.0, Q.n_lund_kt_above_5 - 2.0) / 0.2877964   # -0.1%  pair_max_lnm2 > 5.909 and n_lund_kt_above_5 > 2
        + 0.0007724678 * max(0.0, 0.0001729927 - Q.e3_b2) * max(0.0, 13.0 - Q.n_pt_above_10) / 0.0001662049   # +0.1%  e3_b2 < 0.000173 and n_pt_above_10 < 13
        + 0.0007690515 * max(0.0, Q.sj4_pair_mass_max - 48.76729) * max(0.0, 6.0 - Q.n_lund_kt_above_1) / 35.25129   # +0.1%  sj4_pair_mass_max > 48.77 and n_lund_kt_above_1 < 6
        - 0.00065134 * max(0.0, 5.0 - Q.n_lund) / 0.1017067   # -0.1%  n_lund < 5
        - 0.0006468041 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # -0.1%  sj2_mass2 < 1.852
        + 0.000619316 * max(0.0, 0.828622 - Q.N3_b05) * max(0.0, 6.0 - Q.n_dr_0_0p05) / 0.72594   # +0.1%  N3_b05 < 0.8286 and n_dr_0_0p05 < 6
        + 0.0006182951 * max(0.0, Q.sd_mass - 88.81751) * max(0.0, Q.C2_b2 - 0.04664369) / 1.176182   # +0.1%  sd_mass > 88.82 and C2_b2 > 0.04664
        - 0.0005715328 * max(0.0, Q.sj4_pair_mass_max - 48.76729) / 33.64482   # -0.1%  sj4_pair_mass_max > 48.77
        - 0.0005645186 * max(0.0, 2.151069 - Q.sj3_mass3) * max(0.0, 3.0 - Q.n_lund_kt_above_5) / 0.5301747   # -0.1%  sj3_mass3 < 2.151 and n_lund_kt_above_5 < 3
        + 0.0005487299 * max(0.0, 2.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.2912598 - Q.eta_38) / 0.02563852   # +0.1%  n_dr_0p1_0p2 < 2 and eta_38 < 0.2913
        - 0.0004904593 * max(0.0, Q.sj3_pair_mass_min - 80.02563) * max(0.0, 19.01265 - Q.sj3_mass2) / 1.907388   # -0.0%  sj3_pair_mass_min > 80.03 and sj3_mass2 < 19.01
        - 0.0004884289 * max(0.0, Q.sj3_pair_mass_min - 44.1643) * max(0.0, 13.0 - Q.n_pairs_kt_above_10) / 19.89765   # -0.0%  sj3_pair_mass_min > 44.16 and n_pairs_kt_above_10 < 13
        + 0.0004387589 * max(0.0, Q.sj4_pair_mass_max - 48.76729) * max(0.0, 0.310246 - Q.dr_max_012) / 3.738722   # +0.0%  sj4_pair_mass_max > 48.77 and dr_max_012 < 0.3102
        - 0.0004295174 * max(0.0, 2.0 - Q.n_dr_0p1_0p2) / 0.08729333   # -0.0%  n_dr_0p1_0p2 < 2
        - 0.0003332412 * max(0.0, 3.298199 - Q.sj3_mass2) / 0.1692566   # -0.0%  sj3_mass2 < 3.298
        + 0.0002690938 * max(0.0, Q.lne_38 - 1.040994) / 0.1507516   # +0.0%  lne_38 > 1.041
        + 0.0002340404 * max(0.0, 12.51777 - Q.m01) * max(0.0, 0.005525017 - Q.eta_12) / 0.3955334   # +0.0%  m01 < 12.52 and eta_12 < 0.005525
        + 0.0002026206 * max(0.0, Q.n_pt_above_5 - 32.0) / 0.3652267   # +0.0%  n_pt_above_5 > 32
        - 0.000176662 * max(0.0, 0.828622 - Q.N3_b05) * max(0.0, 0.2757029 - Q.pt_balance01) / 0.001508908   # -0.0%  N3_b05 < 0.8286 and pt_balance01 < 0.2757
        + 0.0001598928 * max(0.0, 97.52633 - Q.sj3_pair_mass_max) * max(0.0, 0.8610613 - Q.tau54) / 0.7277731   # +0.0%  sj3_pair_mass_max < 97.53 and tau54 < 0.8611
        - 0.000136187 * max(0.0, Q.sd_mass - 62.03827) * max(0.0, Q.tau21_b2 - 0.5868564) / 0.1545868   # -0.0%  sd_mass > 62.04 and tau21_b2 > 0.5869
        + 6.419905e-05 * max(0.0, Q.C2_b05 - 0.1620185) * max(0.0, Q.dr_2 - 0.1403176) / 0.005394771   # +0.0%  C2_b05 > 0.162 and dr_2 > 0.1403
        - 5.193232e-05 * max(0.0, 12.51777 - Q.m01) * max(0.0, Q.e3_b2 - 0.0007909605) / 0.0002048882   # -0.0%  m01 < 12.52 and e3_b2 > 0.000791
        + 1.802832e-05 * max(0.0, Q.sj3_pair_mass_min - 44.1643) * max(0.0, -0.232666 - Q.eta_42) / 0.1430042   # +0.0%  sj3_pair_mass_min > 44.16 and eta_42 < -0.2327
        + 5.13642e-06 * max(0.0, Q.n_pt_above_5 - 32.0) * max(0.0, Q.e4_b2 - 7.0969e-08) / 1.927777e-08   # +0.0%  n_pt_above_5 > 32 and e4_b2 > 7.097e-08
        - 2.68164e-06 * max(0.0, Q.sj3_pair_mass_min - 80.02563) * max(0.0, 0.04125977 - Q.eta_38) / 0.09832977   # -0.0%  sj3_pair_mass_min > 80.03 and eta_38 < 0.04126
        - 1.461592e-06 * max(0.0, 3.298199 - Q.sj3_mass2) * max(0.0, -0.1281738 - Q.eta_20) / 0.003567263   # -0.0%  sj3_mass2 < 3.298 and eta_20 < -0.1282
    )
    return z


def neuron_23(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.400643e-05
    )
    return z


def neuron_24(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.022347e-05
    )
    return z


def neuron_25(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.0001539363
    )
    return z


def neuron_26(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.593832e-06
    )
    return z


def neuron_27(Q):
    # scale S = 6.277; each line: share * term / its average size
    z = 6.277232 * (-0.1274414
        - 0.08824623 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) / 36.55147   # -8.8%  sj4_pair_mass_max < 115.5
        + 0.07075634 * max(0.0, 0.4954556 - Q.N2_b05) / 0.06560395   # +7.1%  N2_b05 < 0.4955
        + 0.05469584 * max(0.0, 3.38061 - Q.pair_mean_lnm2) / 0.9985192   # +5.5%  pair_mean_lnm2 < 3.381
        + 0.04149907 * max(0.0, 0.1335998 - Q.dr01) / 0.06067654   # +4.1%  dr01 < 0.1336
        + 0.03958574 * max(0.0, 1.549573 - Q.N3_b05) / 0.8053398   # +4.0%  N3_b05 < 1.55
        + 0.03868947 * max(0.0, Q.sd_mass - 83.61981) / 26.48984   # +3.9%  sd_mass > 83.62
        - 0.03377625 * max(0.0, 6.394276 - Q.pt1_dr01) / 2.478804   # -3.4%  pt1_dr01 < 6.394
        - 0.03140678 * max(0.0, Q.sd_mass - 42.31629) / 58.72614   # -3.1%  sd_mass > 42.32
        + 0.0303274 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) / 7.619143   # +3.0%  n_pairs_kt_above_3 < 41
        - 0.0303263 * max(0.0, 116.599 - Q.mass_top10) / 50.63712   # -3.0%  mass_top10 < 116.6
        + 0.03030101 * max(0.0, Q.sj3_pair_mass_max - 80.36029) / 18.3994   # +3.0%  sj3_pair_mass_max > 80.36
        - 0.02921614 * max(0.0, Q.pair_mean_lnm2 - 1.488114) / 1.170692   # -2.9%  pair_mean_lnm2 > 1.488
        + 0.02651195 * max(0.0, 0.1066509 - Q.M2) / 0.03065482   # +2.7%  M2 < 0.1067
        - 0.02486894 * max(0.0, 0.072817 - Q.sum_z_dr2_top2) / 0.05267127   # -2.5%  sum_z_dr2_top2 < 0.07282
        + 0.02422529 * max(0.0, 1.549573 - Q.N3_b05) * max(0.0, 36.36236 - Q.sj3_mass1) / 12.76947   # +2.4%  N3_b05 < 1.55 and sj3_mass1 < 36.36
        - 0.02250952 * max(0.0, 0.072817 - Q.sum_z_dr2_top2) * max(0.0, 12.0 - Q.n_dr_0p4_up) / 0.3696563   # -2.3%  sum_z_dr2_top2 < 0.07282 and n_dr_0p4_up < 12
        - 0.02040772 * max(0.0, 0.1066509 - Q.M2) * max(0.0, 0.06041764 - Q.lam1) / 0.0009107365   # -2.0%  M2 < 0.1067 and lam1 < 0.06042
        + 0.02032159 * max(0.0, 0.1066509 - Q.M2) * max(0.0, 0.5797033 - Q.tau21) / 0.00816618   # +2.0%  M2 < 0.1067 and tau21 < 0.5797
        - 0.0162371 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # -1.6%  n_pairs_kt_above_3 < 28
        + 0.01616768 * max(0.0, Q.mass_top50 - 118.8408) / 11.30684   # +1.6%  mass_top50 > 118.8
        + 0.01505518 * max(0.0, Q.sj3_pair_mass_max - 80.36029) * max(0.0, 0.05844876 - Q.tau3) / 0.1953071   # +1.5%  sj3_pair_mass_max > 80.36 and tau3 < 0.05845
        + 0.01419011 * max(0.0, 0.1458275 - Q.sj4_dr_min) / 0.04988689   # +1.4%  sj4_dr_min < 0.1458
        - 0.01412222 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, -6.393071 - Q.lnerel_50) / 371.1048   # -1.4%  sj4_pair_mass_max < 115.5 and lnerel_50 < -6.393
        + 0.01278011 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, Q.lnpt_0 - 3.990719) / 35.28135   # +1.3%  sj4_pair_mass_max < 115.5 and lnpt_0 > 3.991
        + 0.01162901 * max(0.0, 0.072817 - Q.sum_z_dr2_top2) * max(0.0, Q.M3 - 0.02540381) / 0.0007194409   # +1.2%  sum_z_dr2_top2 < 0.07282 and M3 > 0.0254
        - 0.01149016 * max(0.0, 0.01600475 - Q.dr_min_012) / 0.003959346   # -1.1%  dr_min_012 < 0.016
        + 0.01074354 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +1.1%  n_pairs_kt_above_1 < 80
        + 0.01073901 * max(0.0, 6.0 - Q.n_pairs_kt_above_10) / 2.34071   # +1.1%  n_pairs_kt_above_10 < 6
        - 0.0105404 * max(0.0, Q.sd_mass - 127.8042) / 7.009137   # -1.1%  sd_mass > 127.8
        + 0.01021339 * max(0.0, 0.008597002 - Q.C3_b2) / 0.004767795   # +1.0%  C3_b2 < 0.008597
        - 0.009531752 * max(0.0, 0.072817 - Q.sum_z_dr2_top2) * max(0.0, Q.sj3_dr_min - 0.08335692) / 0.005983708   # -1.0%  sum_z_dr2_top2 < 0.07282 and sj3_dr_min > 0.08336
        - 0.008832367 * max(0.0, Q.sj3_pair_mass_max - 128.6605) / 2.418855   # -0.9%  sj3_pair_mass_max > 128.7
        + 0.008452532 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, Q.n_lund - 5.0) / 204.1159   # +0.8%  sj4_pair_mass_max < 115.5 and n_lund > 5
        + 0.008066561 * max(0.0, 0.01600475 - Q.dr_min_012) * max(0.0, 3.338396 - Q.pair_max_lnkt) / 0.003421218   # +0.8%  dr_min_012 < 0.016 and pair_max_lnkt < 3.338
        + 0.007787277 * max(0.0, 223.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.n_lund_kt_above_1 - 3.0) / 56.99366   # +0.8%  n_pairs_kt_above_1 < 223 and n_lund_kt_above_1 > 3
        + 0.006415761 * max(0.0, 62.96042 - Q.mass_top10) / 9.406087   # +0.6%  mass_top10 < 62.96
        + 0.005848796 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, 4.02591 - Q.lne_3) / 7.856954   # +0.6%  sj4_pair_mass_max < 115.5 and lne_3 < 4.026
        + 0.005638367 * max(0.0, 0.07264571 - Q.tau2) / 0.01634431   # +0.6%  tau2 < 0.07265
        + 0.005548323 * max(0.0, 223.0 - Q.n_pairs_kt_above_1) / 49.27858   # +0.6%  n_pairs_kt_above_1 < 223
        - 0.005397909 * max(0.0, 0.08335692 - Q.sj3_dr_min) / 0.005612375   # -0.5%  sj3_dr_min < 0.08336
        - 0.005394072 * max(0.0, 0.4954556 - Q.N2_b05) * max(0.0, 0.04229114 - Q.C3_b2) / 0.002136635   # -0.5%  N2_b05 < 0.4955 and C3_b2 < 0.04229
        + 0.004852472 * max(0.0, Q.mass_top5 - 58.28268) / 4.708762   # +0.5%  mass_top5 > 58.28
        + 0.004819583 * max(0.0, 0.072817 - Q.sum_z_dr2_top2) * max(0.0, Q.n_dr_0p2_0p4 - 10.0) / 0.1599088   # +0.5%  sum_z_dr2_top2 < 0.07282 and n_dr_0p2_0p4 > 10
        + 0.00481072 * max(0.0, Q.sd_mass - 83.61981) * max(0.0, 0.06637116 - Q.tau4) / 0.561401   # +0.5%  sd_mass > 83.62 and tau4 < 0.06637
        + 0.004802816 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, Q.sj3_dr13 - 0.4088773) / 2.375169   # +0.5%  sj4_pair_mass_max < 115.5 and sj3_dr13 > 0.4089
        + 0.004627945 * max(0.0, 67.20576 - Q.mass_top30) / 1.774095   # +0.5%  mass_top30 < 67.21
        - 0.004584982 * max(0.0, 0.6240065 - Q.N3_b05) / 0.06867617   # -0.5%  N3_b05 < 0.624
        + 0.004572221 * max(0.0, 223.0 - Q.n_pairs_kt_above_1) * max(0.0, 0.4190095 - Q.dr_5) / 13.87615   # +0.5%  n_pairs_kt_above_1 < 223 and dr_5 < 0.419
        - 0.004516999 * max(0.0, 0.1458275 - Q.sj4_dr_min) * max(0.0, 0.1426884 - Q.sj3_z3) / 0.002512501   # -0.5%  sj4_dr_min < 0.1458 and sj3_z3 < 0.1427
        + 0.004362164 * max(0.0, 63.34656 - Q.sj4_pair_mass_max) * max(0.0, Q.z_3rd - 0.06584103) / 0.1467963   # +0.4%  sj4_pair_mass_max < 63.35 and z_3rd > 0.06584
        - 0.004315857 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) * max(0.0, 0.06632183 - Q.dr12) / 0.166601   # -0.4%  n_pairs_kt_above_1 < 80 and dr12 < 0.06632
        - 0.004122297 * max(0.0, Q.ecf_g42 - 2.829762e-05) / 9.057051e-06   # -0.4%  ecf_g42 > 2.83e-05
        + 0.003899605 * max(0.0, 63.34656 - Q.sj4_pair_mass_max) / 3.777391   # +0.4%  sj4_pair_mass_max < 63.35
        - 0.003653721 * max(0.0, 0.0121584 - Q.dr01) / 0.001057216   # -0.4%  dr01 < 0.01216
        - 0.003404688 * max(0.0, 0.5586581 - Q.pair_mean_lnkt) / 0.193039   # -0.3%  pair_mean_lnkt < 0.5587
        - 0.003264829 * max(0.0, Q.pair_mean_lnkt - 0.9367772) / 0.06159511   # -0.3%  pair_mean_lnkt > 0.9368
        + 0.002987815 * max(0.0, 6.0 - Q.n_dr_0p1_0p2) / 0.8571967   # +0.3%  n_dr_0p1_0p2 < 6
        + 0.002867184 * max(0.0, 2.0 - Q.n_dr_0p05_0p1) / 0.2769167   # +0.3%  n_dr_0p05_0p1 < 2
        + 0.002817836 * max(0.0, 2.0 - Q.n_dr_0_0p05) / 0.7938667   # +0.3%  n_dr_0_0p05 < 2
        - 0.002717607 * max(0.0, Q.tau4 - 0.05937965) / 0.002408288   # -0.3%  tau4 > 0.05938
        + 0.002694153 * max(0.0, Q.sj3_dr12 - 0.4152525) / 0.02699121   # +0.3%  sj3_dr12 > 0.4153
        + 0.002502991 * max(0.0, Q.sj3_dr_min - 0.3526989) / 0.01394539   # +0.3%  sj3_dr_min > 0.3527
        - 0.002135707 * max(0.0, Q.sj3_mass2 - 15.2797) / 1.8472   # -0.2%  sj3_mass2 > 15.28
        + 0.002068775 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.3413762 - Q.dr02) / 0.7579775   # +0.2%  n_pairs_kt_above_3 < 28 and dr02 < 0.3414
        - 0.002067837 * max(0.0, 0.4954556 - Q.N2_b05) * max(0.0, Q.dr01 - 0.4232714) / 0.001021785   # -0.2%  N2_b05 < 0.4955 and dr01 > 0.4233
        + 0.002039191 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.M3_b05 - 0.05410585) / 0.07748445   # +0.2%  n_pairs_kt_above_3 < 28 and M3_b05 > 0.05411
        - 0.002028278 * max(0.0, 0.072817 - Q.sum_z_dr2_top2) * max(0.0, Q.sj3_mass3 - 8.552496) / 0.0562161   # -0.2%  sum_z_dr2_top2 < 0.07282 and sj3_mass3 > 8.552
        + 0.002001946 * max(0.0, Q.sj3_pair_mass_max - 128.6605) * max(0.0, Q.lnpt_12 - 1.112834) / 3.462497   # +0.2%  sj3_pair_mass_max > 128.7 and lnpt_12 > 1.113
        - 0.001779849 * max(0.0, Q.sd_mass - 127.8042) * max(0.0, 0.05417009 - Q.tau4) / 0.05218182   # -0.2%  sd_mass > 127.8 and tau4 < 0.05417
        + 0.001560964 * max(0.0, 1.549573 - Q.N3_b05) * max(0.0, 0.4232714 - Q.dr01) / 0.2227896   # +0.2%  N3_b05 < 1.55 and dr01 < 0.4233
        - 0.001529202 * max(0.0, 116.599 - Q.mass_top10) * max(0.0, 0.01057938 - Q.C2_b2) / 0.0174707   # -0.2%  mass_top10 < 116.6 and C2_b2 < 0.01058
        - 0.001175044 * max(0.0, Q.sj3_dr_max - 0.8930677) / 0.01387638   # -0.1%  sj3_dr_max > 0.8931
        - 0.001087733 * max(0.0, 0.3951525 - Q.tau32) * max(0.0, 0.0415296 - Q.mean_eta2) / 0.0001658178   # -0.1%  tau32 < 0.3952 and mean_eta2 < 0.04153
        + 0.00104679 * max(0.0, Q.sj3_dr_min - 0.3526989) * max(0.0, 0.806533 - Q.sj3_pairmax_over_m) / 0.0005362591   # +0.1%  sj3_dr_min > 0.3527 and sj3_pairmax_over_m < 0.8065
        + 0.0009564843 * max(0.0, 0.072817 - Q.sum_z_dr2_top2) * max(0.0, 0.03403084 - Q.C2_b2) / 0.0003095677   # +0.1%  sum_z_dr2_top2 < 0.07282 and C2_b2 < 0.03403
        + 0.0009315897 * max(0.0, Q.pair_mean_lnm2 - 1.488114) * max(0.0, 0.3334961 - Q.eta_27) / 0.3991187   # +0.1%  pair_mean_lnm2 > 1.488 and eta_27 < 0.3335
        - 0.0007910404 * max(0.0, Q.min_pair_mass - 6.383481) / 0.6758576   # -0.1%  min_pair_mass > 6.383
        + 0.0007704595 * max(0.0, Q.sd_mass - 42.31629) * max(0.0, Q.mratio_max_012 - 0.7062246) / 4.213104   # +0.1%  sd_mass > 42.32 and mratio_max_012 > 0.7062
        - 0.000706704 * max(0.0, Q.sd_mass - 127.8042) * max(0.0, 0.2377332 - Q.tau21) / 0.05042618   # -0.1%  sd_mass > 127.8 and tau21 < 0.2377
        + 0.0005796057 * max(0.0, Q.lund2_lndelta - -1.411949) / 0.4841443   # +0.1%  lund2_lndelta > -1.412
        - 0.0005129911 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.4623202 - Q.planar_flow) / 0.9250383   # -0.1%  n_pairs_kt_above_3 < 41 and planar_flow < 0.4623
        - 0.0005038 * max(0.0, 2.0 - Q.n_dr_0_0p05) * max(0.0, 0.4806885 - Q.lnpt_18) / 1.053336   # -0.1%  n_dr_0_0p05 < 2 and lnpt_18 < 0.4807
        + 0.000501875 * max(0.0, 0.0121584 - Q.dr01) * max(0.0, Q.lnpt_18 - 1.844387) / 0.0001270307   # +0.1%  dr01 < 0.01216 and lnpt_18 > 1.844
        - 0.0004618912 * max(0.0, 0.3951525 - Q.tau32) / 0.009206105   # -0.0%  tau32 < 0.3952
        + 0.0004350092 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) * max(0.0, Q.ecf_g42 - 2.829762e-05) / 0.0002190726   # +0.0%  sj4_pair_mass_max < 115.5 and ecf_g42 > 2.83e-05
        + 0.0004081187 * max(0.0, Q.pair_mean_lndelta - -1.921258) / 0.1077194   # +0.0%  pair_mean_lndelta > -1.921
        + 0.000339285 * max(0.0, Q.dr_11 - 0.4092361) / 0.01361917   # +0.0%  dr_11 > 0.4092
        - 0.0003340608 * max(0.0, 1.549573 - Q.N3_b05) * max(0.0, 20.0 - Q.n_real_top20) / 0.0242976   # -0.0%  N3_b05 < 1.55 and n_real_top20 < 20
        - 0.0003219961 * max(0.0, 2.0 - Q.n_dr_0_0p05) * max(0.0, -5.280637 - Q.lnerel_12) / 0.131549   # -0.0%  n_dr_0_0p05 < 2 and lnerel_12 < -5.281
        + 0.0002529305 * max(0.0, 2.0 - Q.n_dr_0_0p05) * max(0.0, Q.e3_b2 - 0.000125186) / 0.0001609221   # +0.0%  n_dr_0_0p05 < 2 and e3_b2 > 0.0001252
        + 0.0001859905 * max(0.0, 63.34656 - Q.sj4_pair_mass_max) * max(0.0, Q.pt1_dr01 - 11.84401) / 4.621239   # +0.0%  sj4_pair_mass_max < 63.35 and pt1_dr01 > 11.84
        + 0.00016912 * max(0.0, 6.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.1385887 - Q.dr_4) / 0.05295299   # +0.0%  n_dr_0p1_0p2 < 6 and dr_4 < 0.1386
        + 0.0001623004 * max(0.0, Q.lund2_lndelta - -1.411949) * max(0.0, Q.dr_10 - 0.5063302) / 0.00392681   # +0.0%  lund2_lndelta > -1.412 and dr_10 > 0.5063
        - 0.0001256463 * max(0.0, 0.4954556 - Q.N2_b05) * max(0.0, Q.phi_47 - 0.0) / 0.0004041334   # -0.0%  N2_b05 < 0.4955 and phi_47 > 0
        - 8.783289e-05 * max(0.0, 2.0 - Q.n_dr_0p4_up) / 0.3859433   # -0.0%  n_dr_0p4_up < 2
        + 7.651331e-05 * max(0.0, 2.0 - Q.n_dr_0p4_up) * max(0.0, 0.1719087 - Q.sj3_z3) / 0.02155412   # +0.0%  n_dr_0p4_up < 2 and sj3_z3 < 0.1719
        - 6.507722e-05 * max(0.0, 63.34656 - Q.sj4_pair_mass_max) * max(0.0, 0.1783885 - Q.z_2nd) / 0.1443132   # -0.0%  sj4_pair_mass_max < 63.35 and z_2nd < 0.1784
        + 5.390091e-05 * max(0.0, Q.mass_top50 - 118.8408) * max(0.0, 0.0 - Q.phi_80) / 0.05361868   # +0.0%  mass_top50 > 118.8 and phi_80 < 0
        - 3.762465e-05 * max(0.0, Q.pair_mean_lnm2 - 1.488114) * max(0.0, 0.2677232 - Q.dr12) / 0.163863   # -0.0%  pair_mean_lnm2 > 1.488 and dr12 < 0.2677
        + 1.516052e-05 * max(0.0, 1.824785 - Q.sj3_mass2) / 0.06171879   # +0.0%  sj3_mass2 < 1.825
    )
    return z


def neuron_28(Q):
    # scale S = 1.481; each line: share * term / its average size
    z = 1.481257 * (-0.01677245
        + 0.08105572 * max(0.0, Q.lund_max_lnkt - 3.22013) / 0.7220204   # +8.1%  lund_max_lnkt > 3.22
        - 0.06631226 * max(0.0, 873.0 - Q.n_pairs_kt_above_1) / 556.2057   # -6.6%  n_pairs_kt_above_1 < 873
        - 0.06151204 * max(0.0, Q.mass - 90.08945) / 29.81377   # -6.2%  mass > 90.09
        - 0.06119648 * max(0.0, Q.pair_max_lnkt - 2.080306) / 0.69545   # -6.1%  pair_max_lnkt > 2.08
        + 0.05321566 * max(0.0, 0.07142062 - Q.sum_z_dr2_top3) / 0.05004712   # +5.3%  sum_z_dr2_top3 < 0.07142
        + 0.04558436 * max(0.0, Q.pair_max_lnm2 - 6.411593) / 0.466776   # +4.6%  pair_max_lnm2 > 6.412
        - 0.03877141 * max(0.0, 0.004407991 - Q.e3) / 0.003185681   # -3.9%  e3 < 0.004408
        - 0.03817506 * max(0.0, 0.3413762 - Q.dr02) / 0.1988175   # -3.8%  dr02 < 0.3414
        + 0.03562551 * max(0.0, 0.03877522 - Q.sum_z_dr2_top10) / 0.01565932   # +3.6%  sum_z_dr2_top10 < 0.03878
        + 0.03242675 * max(0.0, Q.e3_b05 - 0.001630164) / 0.005835361   # +3.2%  e3_b05 > 0.00163
        + 0.02660991 * max(0.0, 0.1620436 - Q.dr02) / 0.07427204   # +2.7%  dr02 < 0.162
        - 0.02576267 * max(0.0, 93.87663 - Q.sj3_pair_mass_max) / 12.55772   # -2.6%  sj3_pair_mass_max < 93.88
        + 0.02509769 * max(0.0, 42.26122 - Q.m01) * max(0.0, 15.17086 - Q.D2_b2) / 330.0492   # +2.5%  m01 < 42.26 and D2_b2 < 15.17
        + 0.02342043 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) * max(0.0, 14.0 - Q.n_lund) / 496.4341   # +2.3%  n_pairs_kt_above_1 < 325 and n_lund < 14
        + 0.02090446 * max(0.0, Q.mass - 114.0172) / 14.73611   # +2.1%  mass > 114
        + 0.01769657 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # +1.8%  sj3_pair_mass_max < 73.25
        - 0.01745571 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) / 103.3282   # -1.7%  n_pairs_kt_above_1 < 325
        - 0.01717727 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 1.43511   # -1.7%  n_dr_0p2_0p4 < 7
        + 0.01683603 * max(0.0, Q.dr01 - 0.1335998) / 0.08309635   # +1.7%  dr01 > 0.1336
        + 0.01526537 * max(0.0, 0.05279362 - Q.z_dr_0p2_0p4) / 0.01042015   # +1.5%  z_dr_0p2_0p4 < 0.05279
        - 0.01487587 * max(0.0, Q.mass - 95.14961) / 26.15323   # -1.5%  mass > 95.15
        - 0.0146519 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # -1.5%  sd_mass > 94.56
        - 0.01376553 * max(0.0, Q.lund_max_lnkt - 3.22013) * max(0.0, 0.3898758 - Q.pt2_over_pt0) / 0.03989405   # -1.4%  lund_max_lnkt > 3.22 and pt2_over_pt0 < 0.3899
        + 0.01281984 * max(0.0, 42.26122 - Q.m01) / 28.28232   # +1.3%  m01 < 42.26
        + 0.01195798 * max(0.0, Q.sd_mass - 119.4443) / 9.070113   # +1.2%  sd_mass > 119.4
        - 0.01179027 * Q.pair_mean_lnm2 / 2.57995   # -1.2%  pair_mean_lnm2
        - 0.01050658 * max(0.0, Q.mass_top40 - 122.7145) / 8.484825   # -1.1%  mass_top40 > 122.7
        + 0.01038029 * max(0.0, 42.26122 - Q.m01) * max(0.0, Q.sj3_pair_mass_min - 29.07349) / 320.6924   # +1.0%  m01 < 42.26 and sj3_pair_mass_min > 29.07
        - 0.009660235 * max(0.0, Q.lund3_lndelta - -2.817283) / 1.325774   # -1.0%  lund3_lndelta > -2.817
        + 0.008241274 * max(0.0, 120.6471 - Q.sj3_pair_mass_max) / 31.70826   # +0.8%  sj3_pair_mass_max < 120.6
        - 0.008138834 * max(0.0, Q.lund_max_lnkt - 4.115614) / 0.1287457   # -0.8%  lund_max_lnkt > 4.116
        - 0.007188149 * max(0.0, 5.0 - Q.n_dr_0p4_up) / 1.758997   # -0.7%  n_dr_0p4_up < 5
        - 0.006658305 * max(0.0, Q.pair_mean_lndelta - -1.986684) * max(0.0, Q.psi_0p3 - 0.6500863) / 0.02090813   # -0.7%  pair_mean_lndelta > -1.987 and psi_0p3 > 0.6501
        + 0.006447808 * max(0.0, Q.dr01 - 0.1335998) * max(0.0, 0.04229114 - Q.C3_b2) / 0.002723183   # +0.6%  dr01 > 0.1336 and C3_b2 < 0.04229
        + 0.006032879 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) * max(0.0, 0.072817 - Q.sum_z_dr2_top2) / 5.972269   # +0.6%  n_pairs_kt_above_1 < 325 and sum_z_dr2_top2 < 0.07282
        - 0.005983097 * max(0.0, 0.02098847 - Q.tau5) / 0.002077107   # -0.6%  tau5 < 0.02099
        + 0.005764424 * max(0.0, Q.M3_b05 - 0.03550507) / 0.0232179   # +0.6%  M3_b05 > 0.03551
        - 0.005502136 * max(0.0, 0.3398637 - Q.dr_max_012) / 0.152061   # -0.6%  dr_max_012 < 0.3399
        + 0.005054327 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # +0.5%  mass_top50 > 161.1
        - 0.004895676 * max(0.0, 3.294888 - Q.m01) / 0.785535   # -0.5%  m01 < 3.295
        + 0.00475304 * max(0.0, 0.06625964 - Q.M2) / 0.006053324   # +0.5%  M2 < 0.06626
        - 0.004713768 * max(0.0, Q.pair_mean_lndelta - -1.986684) / 0.1321814   # -0.5%  pair_mean_lndelta > -1.987
        - 0.004545972 * max(0.0, 0.003001458 - Q.ecf_g32) / 0.001159814   # -0.5%  ecf_g32 < 0.003001
        - 0.004225023 * max(0.0, Q.pair_max_lnm2 - 6.411593) * max(0.0, Q.lund2_lndelta - -1.20523) / 0.1691671   # -0.4%  pair_max_lnm2 > 6.412 and lund2_lndelta > -1.205
        - 0.003871846 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # -0.4%  sd_mass > 154.6
        - 0.003811162 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, 2.613544 - Q.D2_b2) / 1.110752   # -0.4%  n_dr_0p2_0p4 < 7 and D2_b2 < 2.614
        - 0.003702775 * max(0.0, 120.6471 - Q.sj3_pair_mass_max) * max(0.0, 2.13546 - Q.lne_29) / 290.9263   # -0.4%  sj3_pair_mass_max < 120.6 and lne_29 < 2.135
        + 0.003615837 * max(0.0, Q.pair_mean_lnkt - 0.9367772) / 0.06159511   # +0.4%  pair_mean_lnkt > 0.9368
        - 0.003505995 * max(0.0, 7.0 - Q.n_pairs_kt_above_10) / 2.958697   # -0.4%  n_pairs_kt_above_10 < 7
        + 0.003453025 * max(0.0, Q.mass_top40 - 122.7145) * max(0.0, 0.0007909605 - Q.e3_b2) / 0.002631102   # +0.3%  mass_top40 > 122.7 and e3_b2 < 0.000791
        + 0.003090474 * max(0.0, Q.tau1 - 0.1866173) / 0.03390354   # +0.3%  tau1 > 0.1866
        + 0.003024849 * max(0.0, 88.0 - Q.n_pairs_kt_above_3) / 31.77884   # +0.3%  n_pairs_kt_above_3 < 88
        + 0.002940271 * max(0.0, 0.1900649 - Q.sd_zg) / 0.01480835   # +0.3%  sd_zg < 0.1901
        + 0.002826519 * max(0.0, 0.11544 - Q.C2) / 0.01159687   # +0.3%  C2 < 0.1154
        - 0.00280919 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) * max(0.0, 0.1573433 - Q.M2_b05) / 4.100814   # -0.3%  n_pairs_kt_above_1 < 325 and M2_b05 < 0.1573
        + 0.002525112 * max(0.0, Q.M3_b05 - 0.03550507) * max(0.0, 0.4190095 - Q.dr_5) / 0.006707117   # +0.3%  M3_b05 > 0.03551 and dr_5 < 0.419
        + 0.002509371 * max(0.0, Q.M3_b05 - 0.03550507) * max(0.0, Q.sj2_zsoft - 0.2089827) / 0.001991061   # +0.3%  M3_b05 > 0.03551 and sj2_zsoft > 0.209
        - 0.002188163 * max(0.0, 8.0 - Q.n_lund) / 0.3988733   # -0.2%  n_lund < 8
        + 0.002073187 * max(0.0, 0.0184643 - Q.sj3_z3) / 0.0008523908   # +0.2%  sj3_z3 < 0.01846
        - 0.002008995 * max(0.0, Q.sj3_pair_mass_min - 37.19471) / 8.436037   # -0.2%  sj3_pair_mass_min > 37.19
        + 0.001974989 * max(0.0, 13.0 - Q.n_pt_above_10) / 1.505313   # +0.2%  n_pt_above_10 < 13
        - 0.001935759 * max(0.0, 1.053727 - Q.sj3_mass3) / 0.1175774   # -0.2%  sj3_mass3 < 1.054
        - 0.001918183 * max(0.0, Q.z_top2_slots - 0.4675914) / 0.03529666   # -0.2%  z_top2_slots > 0.4676
        - 0.001899551 * max(0.0, Q.mass_top20 - 121.2735) / 3.429677   # -0.2%  mass_top20 > 121.3
        + 0.001794157 * max(0.0, Q.lne_1 - 4.70445) / 0.1635936   # +0.2%  lne_1 > 4.704
        - 0.001780028 * max(0.0, Q.mass_top40 - 83.57316) * max(0.0, Q.sj2_dr - 0.5076533) / 1.330023   # -0.2%  mass_top40 > 83.57 and sj2_dr > 0.5077
        + 0.001741352 * max(0.0, Q.sj4_pair_mass_max - 115.5142) / 2.08655   # +0.2%  sj4_pair_mass_max > 115.5
        - 0.001732707 * max(0.0, 5.0 - Q.n_lund) / 0.1017067   # -0.2%  n_lund < 5
        + 0.00152292 * max(0.0, 0.3035327 - Q.pt_balance01) / 0.02443296   # +0.2%  pt_balance01 < 0.3035
        - 0.001303198 * max(0.0, 0.01374378 - Q.e2_b2) / 0.0007836237   # -0.1%  e2_b2 < 0.01374
        - 0.001264632 * max(0.0, Q.M3_b05 - 0.03550507) * max(0.0, 0.6550361 - Q.tau32_b2) / 0.002854556   # -0.1%  M3_b05 > 0.03551 and tau32_b2 < 0.655
        + 0.00115781 * max(0.0, 0.1620436 - Q.dr02) * max(0.0, Q.sj3_mass1 - 4.621465) / 1.16323   # +0.1%  dr02 < 0.162 and sj3_mass1 > 4.621
        + 0.001124809 * max(0.0, Q.mass_top40 - 83.57316) / 30.60852   # +0.1%  mass_top40 > 83.57
        - 0.001064108 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.1589855 - Q.dr_0) / 0.3435528   # -0.1%  n_pairs_kt_above_3 < 28 and dr_0 < 0.159
        + 0.001032996 * max(0.0, 3.294888 - Q.m01) * max(0.0, 0.1876221 - Q.eta_12) / 0.1658647   # +0.1%  m01 < 3.295 and eta_12 < 0.1876
        - 0.0009582782 * max(0.0, Q.dr01 - 0.1335998) * max(0.0, Q.n_dr_0p05_0p1 - 6.0) / 0.09108404   # -0.1%  dr01 > 0.1336 and n_dr_0p05_0p1 > 6
        + 0.0008442545 * max(0.0, 0.2686235 - Q.LHA) / 0.00660313   # +0.1%  LHA < 0.2686
        + 0.0008122051 * max(0.0, Q.mass_top20 - 121.2735) * max(0.0, 13.49356 - Q.sj3_mass1) / 4.233158   # +0.1%  mass_top20 > 121.3 and sj3_mass1 < 13.49
        + 0.0007307782 * max(0.0, 0.1620436 - Q.dr02) * max(0.0, Q.n_pt_above_5 - 16.0) / 0.4620088   # +0.1%  dr02 < 0.162 and n_pt_above_5 > 16
        - 0.0007197555 * max(0.0, 101.0 - Q.n_pairs_kt_above_1) / 8.83225   # -0.1%  n_pairs_kt_above_1 < 101
        + 0.000596839 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.tau54 - 0.7835935) / 0.08825674   # +0.1%  n_dr_0p2_0p4 < 7 and tau54 > 0.7836
        - 0.0005875504 * max(0.0, Q.sj3_pair_mass_max - 44.2029) / 48.7862   # -0.1%  sj3_pair_mass_max > 44.2
        + 0.0005606664 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # +0.1%  sj2_mass2 < 1.852
        + 0.0004925594 * max(0.0, 0.0815014 - Q.M2_b05) / 0.001014705   # +0.0%  M2_b05 < 0.0815
        + 0.000487214 * max(0.0, Q.m012 - 69.67489) / 0.7542893   # +0.0%  m012 > 69.67
        + 0.0004504189 * max(0.0, Q.mass_top20 - 121.2735) * max(0.0, 4.06096e-07 - Q.e4) / 5.104961e-08   # +0.0%  mass_top20 > 121.3 and e4 < 4.061e-07
        - 0.0004335448 * max(0.0, Q.ecf_g32 - 0.006299414) / 0.0001063595   # -0.0%  ecf_g32 > 0.006299
        + 0.0004159884 * max(0.0, 1.851735 - Q.sj2_mass2) * max(0.0, Q.lnptrel_14 - -4.925159) / 0.02226997   # +0.0%  sj2_mass2 < 1.852 and lnptrel_14 > -4.925
        - 0.0003692193 * max(0.0, Q.lne_29 - 1.557322) / 0.1506023   # -0.0%  lne_29 > 1.557
        + 0.000361505 * max(0.0, Q.jet_abs_eta - 1.465946) / 0.02086851   # +0.0%  jet_abs_eta > 1.466
        + 0.0003036826 * max(0.0, Q.sj4_pair_mass_max - 115.5142) * max(0.0, Q.e4 - 7.184834e-06) / 1.434126e-05   # +0.0%  sj4_pair_mass_max > 115.5 and e4 > 7.185e-06
        - 0.0002834015 * max(0.0, 4.0 - Q.n_lund_kt_above_1) / 0.2806233   # -0.0%  n_lund_kt_above_1 < 4
        - 0.0002767041 * max(0.0, 0.06625964 - Q.M2) * max(0.0, Q.dr_6 - 0.08715843) / 0.0007473583   # -0.0%  M2 < 0.06626 and dr_6 > 0.08716
        + 0.0001890811 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # +0.0%  n_pairs_kt_above_3 < 28
        - 0.0001535891 * max(0.0, Q.mass - 114.0172) * max(0.0, 2.306773e-05 - Q.e3_b2) / 1.22924e-05   # -0.0%  mass > 114 and e3_b2 < 2.307e-05
        + 5.085847e-05 * max(0.0, 873.0 - Q.n_pairs_kt_above_1) * max(0.0, 0.0 - Q.eta_33) / 20.6255   # +0.0%  n_pairs_kt_above_1 < 873 and eta_33 < 0
        - 3.105917e-05 * max(0.0, Q.pair_max_lnm2 - 6.411593) * max(0.0, Q.eta_81 - 0.0) / 0.0003349466   # -0.0%  pair_max_lnm2 > 6.412 and eta_81 > 0
        + 1.709864e-05 * max(0.0, 1.053727 - Q.sj3_mass3) * max(0.0, 0.0 - Q.eta_33) / 0.002837809   # +0.0%  sj3_mass3 < 1.054 and eta_33 < 0
        - 1.150029e-05 * max(0.0, Q.mass_top20 - 121.2735) * max(0.0, Q.phi_17 - -0.02920532) / 0.4145697   # -0.0%  mass_top20 > 121.3 and phi_17 > -0.02921
        + 3.692939e-06 * max(0.0, Q.m012 - 69.67489) * max(0.0, Q.sj4_zsoft - 0.1545914) / 0.0006627306   # +0.0%  m012 > 69.67 and sj4_zsoft > 0.1546
    )
    return z


def neuron_29(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.301974e-05
    )
    return z


def neuron_30(Q):
    # scale S = 2.006; each line: share * term / its average size
    z = 2.006075 * (-0.0361746
        - 0.1134509 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # -11.3%  sd_mass > 78.48
        + 0.07196419 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # +7.2%  sd_mass > 88.82
        - 0.05879093 * max(0.0, 873.0 - Q.n_pairs_kt_above_1) / 556.2057   # -5.9%  n_pairs_kt_above_1 < 873
        + 0.05854068 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # +5.9%  sd_mass > 94.56
        - 0.04969289 * max(0.0, 128.6605 - Q.sj3_pair_mass_max) / 38.7386   # -5.0%  sj3_pair_mass_max < 128.7
        - 0.0462672 * max(0.0, 120.653 - Q.mass) / 17.47277   # -4.6%  mass < 120.7
        - 0.04173518 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # -4.2%  sd_mass > 106.8
        - 0.04115938 * max(0.0, 97.12186 - Q.mass_top40) / 7.43888   # -4.1%  mass_top40 < 97.12
        + 0.03844922 * max(0.0, Q.z_top40_slots - 0.9077314) / 0.07624387   # +3.8%  z_top40_slots > 0.9077
        + 0.03659544 * max(0.0, 135.5368 - Q.mass_top50) / 28.82074   # +3.7%  mass_top50 < 135.5
        + 0.03534468 * max(0.0, 105.4845 - Q.sj3_pair_mass_max) / 19.86384   # +3.5%  sj3_pair_mass_max < 105.5
        + 0.03395581 * max(0.0, 110.2019 - Q.mass) / 12.0043   # +3.4%  mass < 110.2
        + 0.03384344 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # +3.4%  e3_b2 < 0.0004127
        + 0.026251 * max(0.0, Q.tau1 - 0.1246227) / 0.07463455   # +2.6%  tau1 > 0.1246
        + 0.01935259 * max(0.0, 164.4374 - Q.mass) / 52.54489   # +1.9%  mass < 164.4
        + 0.01588642 * max(0.0, Q.sd_mass - 62.03827) / 42.41619   # +1.6%  sd_mass > 62.04
        - 0.01444681 * max(0.0, 0.05997694 - Q.tau2) / 0.009989817   # -1.4%  tau2 < 0.05998
        + 0.01426848 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # +1.4%  n_pairs_kt_above_1 < 366
        + 0.01370717 * max(0.0, 79.54717 - Q.mass_top30) / 3.542433   # +1.4%  mass_top30 < 79.55
        + 0.01202241 * max(0.0, Q.z_top15_slots - 0.8008865) / 0.07132828   # +1.2%  z_top15_slots > 0.8009
        + 0.01026457 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +1.0%  mass < 95.15
        + 0.01020016 * max(0.0, 0.2613173 - Q.dr_25) * max(0.0, Q.lnpt_5 - 2.515047) / 0.08193385   # +1.0%  dr_25 < 0.2613 and lnpt_5 > 2.515
        + 0.009944927 * max(0.0, 0.08811137 - Q.tau2) / 0.02565187   # +1.0%  tau2 < 0.08811
        - 0.009690364 * max(0.0, 0.1631992 - Q.sj4_dr_min) / 0.06250144   # -1.0%  sj4_dr_min < 0.1632
        - 0.009136707 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # -0.9%  sd_mass > 175.9
        - 0.00911348 * max(0.0, 0.2613173 - Q.dr_25) / 0.1124484   # -0.9%  dr_25 < 0.2613
        + 0.00853423 * max(0.0, Q.sd_mass - 175.9333) * max(0.0, 0.8301995 - Q.tau32_b2) / 0.7418738   # +0.9%  sd_mass > 175.9 and tau32_b2 < 0.8302
        - 0.00816315 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # -0.8%  sj3_pair_mass_max < 73.25
        + 0.008046091 * max(0.0, 105.4845 - Q.sj3_pair_mass_max) * max(0.0, 0.08614914 - Q.dr_32) / 1.018636   # +0.8%  sj3_pair_mass_max < 105.5 and dr_32 < 0.08615
        - 0.007707893 * max(0.0, 137.452 - Q.mass) / 29.4899   # -0.8%  mass < 137.5
        - 0.007180556 * max(0.0, 88.0 - Q.n_pairs_kt_above_3) / 31.77884   # -0.7%  n_pairs_kt_above_3 < 88
        + 0.006531787 * max(0.0, Q.m012 - 49.92073) / 2.537789   # +0.7%  m012 > 49.92
        - 0.006318768 * max(0.0, Q.sj4_pair_mass_max - 69.83554) / 16.93977   # -0.6%  sj4_pair_mass_max > 69.84
        - 0.006279096 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, 0.8720575 - Q.sj3_z1) / 8.214036e-05   # -0.6%  e3_b2 < 0.0004127 and sj3_z1 < 0.8721
        - 0.005974553 * max(0.0, Q.mass_over_sum_pt - 0.2243345) / 0.01185973   # -0.6%  mass_over_sum_pt > 0.2243
        - 0.005589168 * max(0.0, Q.mass_top5 - 68.52153) / 2.713876   # -0.6%  mass_top5 > 68.52
        + 0.005209548 * max(0.0, Q.sd_mass - 106.7501) * max(0.0, Q.tau32 - 0.6513932) / 0.7440308   # +0.5%  sd_mass > 106.8 and tau32 > 0.6514
        + 0.004969725 * max(0.0, 9.76395e-05 - Q.e4_b05) / 6.241811e-05   # +0.5%  e4_b05 < 9.764e-05
        + 0.003837581 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.lund3_lndelta - -1.902701) / 22.39247   # +0.4%  mass < 164.4 and lund3_lndelta > -1.903
        + 0.003755564 * max(0.0, Q.lund_max_lndelta - -0.615738) / 0.02343634   # +0.4%  lund_max_lndelta > -0.6157
        - 0.003535172 * max(0.0, Q.m012 - 32.7897) / 6.653649   # -0.4%  m012 > 32.79
        - 0.003524017 * max(0.0, 105.4845 - Q.sj3_pair_mass_max) * max(0.0, Q.n_dr_0p4_up - 0.0) / 64.71968   # -0.4%  sj3_pair_mass_max < 105.5 and n_dr_0p4_up > 0
        + 0.003372959 * max(0.0, Q.z_1st - 0.5700296) / 0.006009751   # +0.3%  z_1st > 0.57
        + 0.0033412 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.n_pt_above_5 - 10.0) / 507.2171   # +0.3%  mass < 164.4 and n_pt_above_5 > 10
        - 0.003330886 * max(0.0, 0.2734424 - Q.N2) / 0.02340706   # -0.3%  N2 < 0.2734
        + 0.003151882 * max(0.0, 120.653 - Q.mass) * max(0.0, Q.eccentricity - 0.8244523) / 0.8135121   # +0.3%  mass < 120.7 and eccentricity > 0.8245
        + 0.003104859 * max(0.0, 0.02808351 - Q.tau3) / 0.002063919   # +0.3%  tau3 < 0.02808
        + 0.003046116 * max(0.0, 88.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.sj3_z2 - 0.1272236) / 4.409916   # +0.3%  n_pairs_kt_above_3 < 88 and sj3_z2 > 0.1272
        + 0.002724655 * max(0.0, Q.sd_mass - 175.9333) * max(0.0, Q.tau43 - 0.568406) / 0.3177352   # +0.3%  sd_mass > 175.9 and tau43 > 0.5684
        - 0.002551613 * max(0.0, 120.653 - Q.mass) * max(0.0, 1.851735 - Q.sj2_mass2) / 2.422172   # -0.3%  mass < 120.7 and sj2_mass2 < 1.852
        + 0.002516234 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # +0.3%  sj2_mass2 < 1.852
        - 0.00232476 * max(0.0, 1.63996 - Q.pt_entropy) / 0.01754758   # -0.2%  pt_entropy < 1.64
        - 0.002242806 * max(0.0, 0.08811137 - Q.tau2) * max(0.0, 0.0391767 - Q.M3) / 0.0001036649   # -0.2%  tau2 < 0.08811 and M3 < 0.03918
        - 0.00216779 * max(0.0, Q.z_top15_slots - 0.8008865) * max(0.0, Q.eta_26 - -0.21521) / 0.01695611   # -0.2%  z_top15_slots > 0.8009 and eta_26 > -0.2152
        + 0.001884564 * max(0.0, Q.sd_mass - 62.03827) * max(0.0, 21.7295 - Q.sj2_mass2) / 273.944   # +0.2%  sd_mass > 62.04 and sj2_mass2 < 21.73
        + 0.001808243 * max(0.0, 1.617171 - Q.sj3_mass3) * max(0.0, 0.6550361 - Q.tau32_b2) / 0.03365656   # +0.2%  sj3_mass3 < 1.617 and tau32_b2 < 0.655
        - 0.001784331 * max(0.0, 101.0 - Q.n_pairs_kt_above_1) / 8.83225   # -0.2%  n_pairs_kt_above_1 < 101
        - 0.001768115 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, 1.851735 - Q.sj2_mass2) / 2.352955e-05   # -0.2%  e3_b2 < 0.0004127 and sj2_mass2 < 1.852
        - 0.001718542 * max(0.0, 105.4845 - Q.sj3_pair_mass_max) * max(0.0, 0.1648004 - Q.dr_9) / 1.198179   # -0.2%  sj3_pair_mass_max < 105.5 and dr_9 < 0.1648
        + 0.001717358 * max(0.0, Q.mass_top5 - 68.52153) * max(0.0, 0.2690436 - Q.dr_32) / 0.4839361   # +0.2%  mass_top5 > 68.52 and dr_32 < 0.269
        - 0.001681019 * max(0.0, 0.2734424 - Q.N2) * max(0.0, Q.lne_8 - 2.942793) / 0.005833089   # -0.2%  N2 < 0.2734 and lne_8 > 2.943
        - 0.001679958 * max(0.0, 1.617171 - Q.sj3_mass3) / 0.2158965   # -0.2%  sj3_mass3 < 1.617
        - 0.001542756 * max(0.0, 0.08811137 - Q.tau2) * max(0.0, Q.dr12 - 0.2677232) / 0.0005540365   # -0.2%  tau2 < 0.08811 and dr12 > 0.2677
        + 0.001487922 * max(0.0, 1.617171 - Q.sj3_mass3) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 2.685454   # +0.1%  sj3_mass3 < 1.617 and n_dr_0p1_0p2 < 21
        - 0.001436324 * max(0.0, Q.sj3_pair_mass_min - 53.29621) / 3.744152   # -0.1%  sj3_pair_mass_min > 53.3
        - 0.00143624 * max(0.0, Q.sj3_pair_mass_min - 80.02563) / 0.702582   # -0.1%  sj3_pair_mass_min > 80.03
        + 0.00135786 * max(0.0, 95.14961 - Q.mass) * max(0.0, 1.851735 - Q.sj2_mass2) / 1.257722   # +0.1%  mass < 95.15 and sj2_mass2 < 1.852
        + 0.001327634 * max(0.0, Q.pair_mean_lndelta - -2.42577) * max(0.0, 0.3346349 - Q.dr_17) / 0.04852999   # +0.1%  pair_mean_lndelta > -2.426 and dr_17 < 0.3346
        - 0.001246255 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.sj3_dr_max - 0.6446256) / 1.463945   # -0.1%  mass < 164.4 and sj3_dr_max > 0.6446
        + 0.001039955 * max(0.0, 79.47361 - Q.mass) / 2.857843   # +0.1%  mass < 79.47
        - 0.0009609811 * max(0.0, 0.7591346 - Q.N3_b05) / 0.1472444   # -0.1%  N3_b05 < 0.7591
        - 0.0008060579 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, 4.0 - Q.n_dr_0p05_0p1) / 0.0002347834   # -0.1%  e3_b2 < 0.0004127 and n_dr_0p05_0p1 < 4
        + 0.0007956879 * max(0.0, 873.0 - Q.n_pairs_kt_above_1) * max(0.0, 0.0 - Q.phi_8) / 31.58415   # +0.1%  n_pairs_kt_above_1 < 873 and phi_8 < 0
        + 0.00077168 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.min_pair_mass - 10.59762) / 10.32656   # +0.1%  mass < 164.4 and min_pair_mass > 10.6
        - 0.0007370427 * max(0.0, Q.m012 - 49.92073) * max(0.0, Q.min_pair_mass - 1.514826) / 12.74992   # -0.1%  m012 > 49.92 and min_pair_mass > 1.515
        - 0.0007080272 * max(0.0, 110.2019 - Q.mass) * max(0.0, Q.min_pair_mass - 10.59762) / 1.239167   # -0.1%  mass < 110.2 and min_pair_mass > 10.6
        + 0.0006622373 * max(0.0, 0.2734424 - Q.N2) * max(0.0, 0.5339053 - Q.pt1_over_pt0) / 0.002591909   # +0.1%  N2 < 0.2734 and pt1_over_pt0 < 0.5339
        - 0.0005963674 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # -0.1%  n_pairs_kt_above_3 < 28
        + 0.0005711409 * max(0.0, Q.sj3_pair_mass_min - 53.29621) * max(0.0, Q.sj3_z3 - 0.009630718) / 0.5771387   # +0.1%  sj3_pair_mass_min > 53.3 and sj3_z3 > 0.009631
        + 0.0005580629 * max(0.0, Q.pair_mean_lndelta - -1.352792) / 0.01174331   # +0.1%  pair_mean_lndelta > -1.353
        + 0.0005267526 * max(0.0, Q.sd_mass - 78.4753) * max(0.0, 0.3039183 - Q.sj2_dr) / 0.1051652   # +0.1%  sd_mass > 78.48 and sj2_dr < 0.3039
        - 0.0004470661 * max(0.0, Q.sd_mass - 78.4753) * max(0.0, 0.2816529 - Q.pt2_over_pt0) / 0.6676513   # -0.0%  sd_mass > 78.48 and pt2_over_pt0 < 0.2817
        + 0.0003766413 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) * max(0.0, -0.008700943 - Q.phi_0) / 3.473565   # +0.0%  n_pairs_kt_above_1 < 366 and phi_0 < -0.008701
        + 0.0003739762 * max(0.0, Q.sd_mass - 106.7501) * max(0.0, 0.8673543 - Q.tau32_b2) / 5.813209   # +0.0%  sd_mass > 106.8 and tau32_b2 < 0.8674
        + 0.000337072 * max(0.0, Q.pair_mean_lndelta - -2.42577) / 0.3779626   # +0.0%  pair_mean_lndelta > -2.426
        + 0.0003324683 * max(0.0, 1.851735 - Q.sj2_mass2) * max(0.0, 0.2325538 - Q.dr_17) / 0.008891005   # +0.0%  sj2_mass2 < 1.852 and dr_17 < 0.2326
        - 0.0003242811 * max(0.0, 0.0005509685 - Q.sum_z_dr2_top2) / 1.65218e-05   # -0.0%  sum_z_dr2_top2 < 0.000551
        + 0.0003145019 * max(0.0, 0.08811137 - Q.tau2) * max(0.0, 0.5464273 - Q.tau32_b2) / 0.001241423   # +0.0%  tau2 < 0.08811 and tau32_b2 < 0.5464
        + 0.00031367 * max(0.0, 0.08811137 - Q.tau2) * max(0.0, Q.ecf_g41 - 6.216954e-05) / 2.826927e-06   # +0.0%  tau2 < 0.08811 and ecf_g41 > 6.217e-05
        + 0.0003014356 * max(0.0, Q.mass_top5 - 68.52153) * max(0.0, 0.7608692 - Q.tau54) / 0.06374681   # +0.0%  mass_top5 > 68.52 and tau54 < 0.7609
        - 0.0002330968 * max(0.0, 1.617171 - Q.sj3_mass3) * max(0.0, Q.sj3_dr13 - 0.1942334) / 0.07861331   # -0.0%  sj3_mass3 < 1.617 and sj3_dr13 > 0.1942
        + 0.0002128005 * max(0.0, Q.m012 - 49.92073) * max(0.0, Q.mratio_max_012 - 0.8335772) / 0.03712734   # +0.0%  m012 > 49.92 and mratio_max_012 > 0.8336
        + 0.0002054166 * max(0.0, Q.sd_mass - 78.4753) * max(0.0, -4.934509 - Q.lund3_lnz) / 11.43225   # +0.0%  sd_mass > 78.48 and lund3_lnz < -4.935
        - 0.0001867693 * max(0.0, 0.2734424 - Q.N2) * max(0.0, Q.n_dr_0p05_0p1 - 16.0) / 0.0005929536   # -0.0%  N2 < 0.2734 and n_dr_0p05_0p1 > 16
        + 8.655681e-05 * max(0.0, Q.mass_over_sum_pt - 0.2243345) * max(0.0, 0.01399549 - Q.sj4_zsoft) / 7.694656e-06   # +0.0%  mass_over_sum_pt > 0.2243 and sj4_zsoft < 0.014
        + 8.376883e-05 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, 0.8314744 - Q.tau43_b2) / 4.435258e-05   # +0.0%  e3_b2 < 0.0004127 and tau43_b2 < 0.8315
        - 7.306216e-05 * max(0.0, Q.n_dr_0_0p05 - 10.0) / 0.3092667   # -0.0%  n_dr_0_0p05 > 10
        - 6.38089e-05 * max(0.0, Q.m012 - 49.92073) * max(0.0, Q.phi_9 - 0.2814941) / 0.0298545   # -0.0%  m012 > 49.92 and phi_9 > 0.2815
        - 1.008599e-05 * max(0.0, 79.47361 - Q.mass) * max(0.0, 1.851735 - Q.sj2_mass2) / 0.7696854   # -0.0%  mass < 79.47 and sj2_mass2 < 1.852
        + 4.685709e-06 * max(0.0, Q.m012 - 49.92073) * max(0.0, Q.dr_34 - 0.2187045) / 0.1244754   # +0.0%  m012 > 49.92 and dr_34 > 0.2187
    )
    return z


def neuron_31(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.82251e-06
    )
    return z


def neuron_32(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.232198e-06
    )
    return z


def neuron_33(Q):
    # scale S = 13.97; each line: share * term / its average size
    z = 13.97184 * (0.1794235
        + 0.0942309 * Q.n_pairs_kt_above_1 / 331.4992   # +9.4%  n_pairs_kt_above_1
        - 0.07152899 * max(0.0, Q.mass - 79.47361) / 38.31536   # -7.2%  mass > 79.47
        - 0.07145191 * Q.n_pt_above_1 / 38.1855   # -7.1%  n_pt_above_1
        - 0.06714059 * max(0.0, 0.2250047 - Q.C3_b05) / 0.1186192   # -6.7%  C3_b05 < 0.225
        - 0.05167322 * max(0.0, Q.sd_mass - 83.61981) / 26.48984   # -5.2%  sd_mass > 83.62
        + 0.03881741 * max(0.0, Q.mass - 100.4835) / 22.56274   # +3.9%  mass > 100.5
        + 0.03720566 * max(0.0, Q.sd_mass - 42.31629) / 58.72614   # +3.7%  sd_mass > 42.32
        + 0.03672405 * max(0.0, 0.1271654 - Q.C3) / 0.08698511   # +3.7%  C3 < 0.1272
        - 0.03597139 * max(0.0, 1.33105 - Q.N3_b05) / 0.6024977   # -3.6%  N3_b05 < 1.331
        - 0.03119855 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) / 143.7916   # -3.1%  n_pairs_kt_above_3 < 220
        - 0.02870944 * max(0.0, Q.M3 - 0.01517988) / 0.02100446   # -2.9%  M3 > 0.01518
        - 0.01913927 * max(0.0, 462.0 - Q.n_pairs_kt_above_1) / 196.6704   # -1.9%  n_pairs_kt_above_1 < 462
        + 0.01871118 * max(0.0, Q.sd_mass - 111.701) / 11.78149   # +1.9%  sd_mass > 111.7
        - 0.01794986 * max(0.0, 0.01162881 - Q.ecf_g31) / 0.005313715   # -1.8%  ecf_g31 < 0.01163
        + 0.01785742 * max(0.0, Q.ecf_g31 - 0.00553358) / 0.00196484   # +1.8%  ecf_g31 > 0.005534
        - 0.01735337 * max(0.0, Q.n_pairs_kt_above_1 - 325.0) / 109.8274   # -1.7%  n_pairs_kt_above_1 > 325
        - 0.01733162 * max(0.0, Q.n_for_90pct - 14.0) / 7.689217   # -1.7%  n_for_90pct > 14
        + 0.0170075 * max(0.0, 0.863865 - Q.tau32) / 0.1990202   # +1.7%  tau32 < 0.8639
        + 0.01641753 * max(0.0, 0.05613495 - Q.tau5) / 0.02517241   # +1.6%  tau5 < 0.05613
        - 0.01311751 * max(0.0, 4.905275 - Q.lne_2) / 0.6888743   # -1.3%  lne_2 < 4.905
        - 0.01190748 * max(0.0, Q.n_lund - 10.0) / 1.682617   # -1.2%  n_lund > 10
        - 0.01144942 * max(0.0, 0.0417856 - Q.M2_b2) * max(0.0, 0.1869341 - Q.C3_b05) / 0.001027615   # -1.1%  M2_b2 < 0.04179 and C3_b05 < 0.1869
        - 0.01139089 * max(0.0, 0.006877516 - Q.ecf_g31) / 0.00161834   # -1.1%  ecf_g31 < 0.006878
        + 0.01053875 * max(0.0, 8.0 - Q.n_lund_kt_above_1) / 2.798053   # +1.1%  n_lund_kt_above_1 < 8
        - 0.01025012 * max(0.0, Q.mass - 110.2019) / 16.73354   # -1.0%  mass > 110.2
        - 0.009808088 * max(0.0, Q.pt_dispersion - 0.2706555) / 0.08161065   # -1.0%  pt_dispersion > 0.2707
        + 0.009695041 * max(0.0, 0.9062492 - Q.tau43) / 0.1126911   # +1.0%  tau43 < 0.9062
        - 0.009335577 * max(0.0, Q.pair_mean_lndelta - -2.42577) * max(0.0, Q.tau54 - 0.6772033) / 0.06242323   # -0.9%  pair_mean_lndelta > -2.426 and tau54 > 0.6772
        - 0.009144803 * max(0.0, Q.ecf_g31 - 0.00553358) * max(0.0, 1.462588 - Q.D3_b2) / 0.002610389   # -0.9%  ecf_g31 > 0.005534 and D3_b2 < 1.463
        - 0.009059344 * max(0.0, 0.6800935 - Q.tau32) / 0.0806267   # -0.9%  tau32 < 0.6801
        + 0.008372287 * max(0.0, 0.0417856 - Q.M2_b2) * max(0.0, 0.07139556 - Q.C3) / 0.0004787308   # +0.8%  M2_b2 < 0.04179 and C3 < 0.0714
        - 0.008286244 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # -0.8%  n_pairs_kt_above_3 < 28
        + 0.008260499 * max(0.0, 0.0417856 - Q.M2_b2) / 0.01370017   # +0.8%  M2_b2 < 0.04179
        + 0.007914367 * max(0.0, Q.mass - 79.47361) * max(0.0, Q.lund3_lndelta - -2.817283) / 58.19991   # +0.8%  mass > 79.47 and lund3_lndelta > -2.817
        - 0.007704721 * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 4.79296   # -0.8%  n_dr_0p2_0p4 > 9
        + 0.006403568 * max(0.0, Q.M3 - 0.01517988) * max(0.0, 0.2338039 - Q.dr_37) / 0.003851109   # +0.6%  M3 > 0.01518 and dr_37 < 0.2338
        + 0.006267942 * max(0.0, 0.03459382 - Q.M3) / 0.005695489   # +0.6%  M3 < 0.03459
        + 0.006223573 * Q.n_pairs_kt_above_10 / 6.942877   # +0.6%  n_pairs_kt_above_10
        + 0.006179829 * max(0.0, 0.4936772 - Q.D3) / 0.2369625   # +0.6%  D3 < 0.4937
        + 0.006154863 * max(0.0, 11.87755 - Q.sj2_mass2) / 2.372559   # +0.6%  sj2_mass2 < 11.88
        - 0.005963492 * max(0.0, Q.n_dr_0p1_0p2 - 7.0) / 6.188233   # -0.6%  n_dr_0p1_0p2 > 7
        - 0.005773173 * max(0.0, 111.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.2108315 - Q.dr_2) / 5.128502   # -0.6%  n_pairs_kt_above_3 < 111 and dr_2 < 0.2108
        + 0.00549135 * max(0.0, 101.849 - Q.sj4_pair_mass_max) / 24.8879   # +0.5%  sj4_pair_mass_max < 101.8
        - 0.004832621 * max(0.0, 101.0 - Q.n_pairs_kt_above_1) / 8.83225   # -0.5%  n_pairs_kt_above_1 < 101
        + 0.004427209 * max(0.0, Q.lne_1 - 3.745821) / 0.85571   # +0.4%  lne_1 > 3.746
        - 0.004313507 * max(0.0, 111.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.ecf_g41 - 6.216954e-05) / 0.006485243   # -0.4%  n_pairs_kt_above_3 < 111 and ecf_g41 > 6.217e-05
        - 0.004113214 * max(0.0, Q.sum_z_dr2_top20 - 0.01911708) / 0.01693635   # -0.4%  sum_z_dr2_top20 > 0.01912
        - 0.003967787 * max(0.0, Q.n_dr_0p4_up - 6.0) / 2.489417   # -0.4%  n_dr_0p4_up > 6
        + 0.003884897 * max(0.0, Q.n_lund - 10.0) * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 2.18948   # +0.4%  n_lund > 10 and n_lund_kt_above_5 > 1
        - 0.003622352 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.2082728 - Q.dr_4) / 0.1695459   # -0.4%  n_dr_0p2_0p4 < 7 and dr_4 < 0.2083
        + 0.003556877 * max(0.0, 0.02982432 - Q.dr02) / 0.005753371   # +0.4%  dr02 < 0.02982
        + 0.003434087 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.1451741 - Q.dr_0) / 0.3003125   # +0.3%  n_pairs_kt_above_3 < 28 and dr_0 < 0.1452
        - 0.003186884 * max(0.0, Q.ecf_g31 - 0.00553358) * max(0.0, -1.332584 - Q.pair_mean_lnz) / 0.0006437692   # -0.3%  ecf_g31 > 0.005534 and pair_mean_lnz < -1.333
        - 0.002951426 * max(0.0, 69.83554 - Q.sj4_pair_mass_max) / 5.726027   # -0.3%  sj4_pair_mass_max < 69.84
        + 0.002940218 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.0004392119 - Q.ecf_g41) / 0.001109348   # +0.3%  n_pairs_kt_above_3 < 28 and ecf_g41 < 0.0004392
        + 0.002807179 * max(0.0, 0.863865 - Q.tau32) * max(0.0, Q.eccentricity - 0.3996864) / 0.0724491   # +0.3%  tau32 < 0.8639 and eccentricity > 0.3997
        + 0.002608026 * max(0.0, 0.0259406 - Q.dr01) / 0.005036114   # +0.3%  dr01 < 0.02594
        - 0.002556627 * max(0.0, 101.849 - Q.sj4_pair_mass_max) * max(0.0, Q.C2_b2 - 0.02858957) / 1.41214   # -0.3%  sj4_pair_mass_max < 101.8 and C2_b2 > 0.02859
        + 0.002524661 * max(0.0, Q.lne_1 - 3.745821) * max(0.0, Q.sj2_zsoft - 0.1869576) / 0.08899312   # +0.3%  lne_1 > 3.746 and sj2_zsoft > 0.187
        + 0.00211876 * max(0.0, Q.mratio_max_012 - 0.787882) / 0.02179413   # +0.2%  mratio_max_012 > 0.7879
        + 0.002078016 * max(0.0, Q.sj3_pair_mass_min - 37.19471) * max(0.0, 0.221369 - Q.C2_b2) / 0.8673087   # +0.2%  sj3_pair_mass_min > 37.19 and C2_b2 < 0.2214
        + 0.002068729 * max(0.0, 0.004077497 - Q.sum_z_dr2_top3) / 0.0004626356   # +0.2%  sum_z_dr2_top3 < 0.004077
        + 0.001960358 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sd_zg - 0.2075336) / 0.1355966   # +0.2%  n_dr_0p2_0p4 < 7 and sd_zg > 0.2075
        - 0.001727674 * max(0.0, 7.0 - Q.n_pairs_kt_above_10) / 2.958697   # -0.2%  n_pairs_kt_above_10 < 7
        - 0.001587545 * max(0.0, Q.mass_top20 - 114.4658) / 4.613131   # -0.2%  mass_top20 > 114.5
        - 0.001533448 * max(0.0, 3.298199 - Q.sj3_mass2) / 0.1692566   # -0.2%  sj3_mass2 < 3.298
        - 0.001503247 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # -0.2%  sj2_mass2 < 1.852
        + 0.001421932 * max(0.0, 0.0417856 - Q.M2_b2) * max(0.0, 0.001184159 - Q.D3_b2) / 2.303917e-06   # +0.1%  M2_b2 < 0.04179 and D3_b2 < 0.001184
        - 0.001371123 * max(0.0, 111.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.1619198 - Q.dr_3) / 3.109635   # -0.1%  n_pairs_kt_above_3 < 111 and dr_3 < 0.1619
        + 0.001345072 * max(0.0, Q.m012 - 44.62237) / 3.452672   # +0.1%  m012 > 44.62
        + 0.001099736 * max(0.0, 0.0417856 - Q.M2_b2) * max(0.0, Q.n_lund_kt_above_1 - 3.0) / 0.02553276   # +0.1%  M2_b2 < 0.04179 and n_lund_kt_above_1 > 3
        - 0.001062151 * max(0.0, Q.pair_max_lnkt - 3.143912) / 0.07891759   # -0.1%  pair_max_lnkt > 3.144
        + 0.0009826591 * max(0.0, 1.00791 - Q.D2) / 0.03617881   # +0.1%  D2 < 1.008
        + 0.0009598272 * max(0.0, 101.849 - Q.sj4_pair_mass_max) * max(0.0, Q.lnpt_4 - 3.230063) / 7.274718   # +0.1%  sj4_pair_mass_max < 101.8 and lnpt_4 > 3.23
        + 0.00088475 * max(0.0, 0.6800935 - Q.tau32) * max(0.0, 0.04063514 - Q.dr_21) / 0.0004574256   # +0.1%  tau32 < 0.6801 and dr_21 < 0.04064
        - 0.0008345999 * max(0.0, 101.849 - Q.sj4_pair_mass_max) * max(0.0, 0.07750231 - Q.dr_35) / 1.172713   # -0.1%  sj4_pair_mass_max < 101.8 and dr_35 < 0.0775
        + 0.000798796 * max(0.0, 11.87755 - Q.sj2_mass2) * max(0.0, 0.07094338 - Q.C2_b2) / 0.06147835   # +0.1%  sj2_mass2 < 11.88 and C2_b2 < 0.07094
        - 0.0007824458 * max(0.0, 0.3436326 - Q.N2_b05) / 0.005384222   # -0.1%  N2_b05 < 0.3436
        - 0.00066944 * max(0.0, 0.1620185 - Q.C2_b05) / 0.001186273   # -0.1%  C2_b05 < 0.162
        - 0.0006458366 * max(0.0, Q.sj3_pair_mass_min - 37.19471) / 8.436037   # -0.1%  sj3_pair_mass_min > 37.19
        - 0.0006224937 * max(0.0, 0.1375297 - Q.M2_b05) / 0.01163631   # -0.1%  M2_b05 < 0.1375
        - 0.0005432847 * max(0.0, 111.0 - Q.n_pairs_kt_above_3) / 47.95244   # -0.1%  n_pairs_kt_above_3 < 111
        - 0.000502788 * max(0.0, Q.pair_mean_lndelta - -2.42577) / 0.3779626   # -0.1%  pair_mean_lndelta > -2.426
        + 0.000458123 * max(0.0, Q.pair_mean_lndelta - -2.42577) * max(0.0, Q.n_pt_above_50 - 3.0) / 0.05347676   # +0.0%  pair_mean_lndelta > -2.426 and n_pt_above_50 > 3
        - 0.0004536823 * max(0.0, Q.sd_mass - 83.61981) * max(0.0, 0.4370905 - Q.dr_6) / 5.695691   # -0.0%  sd_mass > 83.62 and dr_6 < 0.4371
        - 0.0004245678 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 1.43511   # -0.0%  n_dr_0p2_0p4 < 7
        - 0.0004162695 * max(0.0, Q.sj3_dr_min - 0.3526989) / 0.01394539   # -0.0%  sj3_dr_min > 0.3527
        - 0.0003885795 * max(0.0, Q.sj3_pair_mass_min - 80.02563) / 0.702582   # -0.0%  sj3_pair_mass_min > 80.03
        - 0.0003775557 * max(0.0, 0.0417856 - Q.M2_b2) * max(0.0, 0.11544 - Q.C2) / 0.0002554486   # -0.0%  M2_b2 < 0.04179 and C2 < 0.1154
        + 0.0003144944 * max(0.0, Q.sj3_dr_min - 0.3526989) * max(0.0, 0.072817 - Q.sum_z_dr2_top2) / 0.0005735006   # +0.0%  sj3_dr_min > 0.3527 and sum_z_dr2_top2 < 0.07282
        - 0.0002325744 * max(0.0, Q.mass - 79.47361) * max(0.0, 5.85048e-05 - Q.e3_b2) / 0.0004577699   # -0.0%  mass > 79.47 and e3_b2 < 5.85e-05
        + 0.0001940818 * max(0.0, 69.83554 - Q.sj4_pair_mass_max) * max(0.0, 1.892933e-05 - Q.D3_b2) / 1.155018e-05   # +0.0%  sj4_pair_mass_max < 69.84 and D3_b2 < 1.893e-05
        - 0.0001934899 * max(0.0, 0.03459382 - Q.M3) * max(0.0, -0.1281738 - Q.eta_20) / 0.0002044971   # -0.0%  M3 < 0.03459 and eta_20 < -0.1282
        + 0.0001588538 * max(0.0, Q.m012 - 44.62237) * max(0.0, 0.01408252 - Q.D3_b2) / 0.005798774   # +0.0%  m012 > 44.62 and D3_b2 < 0.01408
        - 0.0001471662 * max(0.0, 0.02982432 - Q.dr02) * max(0.0, -0.4705779 - Q.lund1_lndelta) / 0.0009992758   # -0.0%  dr02 < 0.02982 and lund1_lndelta < -0.4706
        - 0.0001127511 * max(0.0, Q.mass - 110.2019) * max(0.0, 0.007827335 - Q.D3_b2) / 0.01597111   # -0.0%  mass > 110.2 and D3_b2 < 0.007827
        + 9.161394e-05 * max(0.0, 56.50998 - Q.mass_top10) / 6.843086   # +0.0%  mass_top10 < 56.51
        - 6.043793e-05 * max(0.0, Q.n_lund - 10.0) * max(0.0, Q.dr_19 - 0.5935318) / 0.009215549   # -0.0%  n_lund > 10 and dr_19 > 0.5935
        - 2.440817e-05 * max(0.0, 0.2083105 - Q.sj2_dr) / 0.00695417   # -0.0%  sj2_dr < 0.2083
        + 6.281815e-06 * max(0.0, Q.sum_z_dr2_top20 - 0.01911708) * max(0.0, Q.phi_68 - 0.0) / 0.0001697794   # +0.0%  sum_z_dr2_top20 > 0.01912 and phi_68 > 0
    )
    return z


def neuron_34(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.463992e-07
    )
    return z


def neuron_35(Q):
    # scale S = 1.657; each line: share * term / its average size
    z = 1.65718 * (0.0005029725
        + 0.1523452 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # +15.2%  sd_mass > 78.48
        + 0.1017759 * max(0.0, Q.mass - 71.96396) / 44.90212   # +10.2%  mass > 71.96
        - 0.07598434 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # -7.6%  sd_mass > 94.56
        - 0.06710396 * max(0.0, Q.sj3_pair_mass_max - 56.73313) / 37.14971   # -6.7%  sj3_pair_mass_max > 56.73
        - 0.06250685 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # -6.3%  sd_mass > 88.82
        - 0.05570843 * max(0.0, Q.sd_mass - 62.03827) / 42.41619   # -5.6%  sd_mass > 62.04
        - 0.04833668 * max(0.0, Q.mass - 79.47361) / 38.31536   # -4.8%  mass > 79.47
        + 0.0414349 * max(0.0, Q.sj3_pair_mass_max - 76.91486) / 20.71932   # +4.1%  sj3_pair_mass_max > 76.91
        - 0.03669132 * max(0.0, Q.mass - 95.14961) / 26.15323   # -3.7%  mass > 95.15
        + 0.03438508 * max(0.0, Q.sj4_pair_mass_max - 78.76797) / 11.81635   # +3.4%  sj4_pair_mass_max > 78.77
        - 0.03102815 * max(0.0, Q.n_pairs_kt_above_3 - 5.0) / 75.94189   # -3.1%  n_pairs_kt_above_3 > 5
        - 0.02853132 * max(0.0, Q.sj4_pair_mass_max - 88.5845) / 7.683986   # -2.9%  sj4_pair_mass_max > 88.58
        + 0.02004416 * max(0.0, Q.sd_mass - 111.701) / 11.78149   # +2.0%  sd_mass > 111.7
        + 0.01604509 * max(0.0, 0.6203012 - Q.tau32_b2) * max(0.0, 6.750142 - Q.log_sum_pt) / 0.0426137   # +1.6%  tau32_b2 < 0.6203 and log_sum_pt < 6.75
        + 0.01432483 * max(0.0, 0.06762785 - Q.M2_b2) / 0.03555411   # +1.4%  M2_b2 < 0.06763
        + 0.01276869 * max(0.0, Q.mass - 110.2019) / 16.73354   # +1.3%  mass > 110.2
        + 0.01107354 * max(0.0, Q.n_pairs_kt_above_3 - 34.0) / 52.11327   # +1.1%  n_pairs_kt_above_3 > 34
        + 0.01015169 * max(0.0, Q.sd_mass - 127.8042) / 7.009137   # +1.0%  sd_mass > 127.8
        + 0.009940657 * max(0.0, Q.mass_top15 - 83.68425) / 10.75757   # +1.0%  mass_top15 > 83.68
        - 0.009342531 * max(0.0, Q.ecf_g31 - 0.004728078) / 0.002447847   # -0.9%  ecf_g31 > 0.004728
        + 0.009225017 * max(0.0, Q.lnptrel_30 - -5.77735) / 0.3703715   # +0.9%  lnptrel_30 > -5.777
        + 0.008764061 * max(0.0, 5.0 - Q.n_pairs_kt_above_10) / 1.790573   # +0.9%  n_pairs_kt_above_10 < 5
        - 0.008708055 * max(0.0, Q.sj3_pair_mass_max - 56.73313) * max(0.0, Q.sj3_z3 - 0.02858644) / 3.433443   # -0.9%  sj3_pair_mass_max > 56.73 and sj3_z3 > 0.02859
        - 0.006490825 * max(0.0, Q.N2_b2 - 0.1528932) * max(0.0, Q.lne_20 - 0.1290046) / 0.08063697   # -0.6%  N2_b2 > 0.1529 and lne_20 > 0.129
        - 0.006035442 * max(0.0, 0.6203012 - Q.tau32_b2) / 0.128098   # -0.6%  tau32_b2 < 0.6203
        - 0.00581174 * max(0.0, Q.mass - 105.7234) / 19.29853   # -0.6%  mass > 105.7
        + 0.005657833 * max(0.0, Q.mass_top20 - 97.09528) / 9.683208   # +0.6%  mass_top20 > 97.1
        - 0.005535174 * max(0.0, 23.0 - Q.n_pairs_kt_above_10) / 16.58492   # -0.6%  n_pairs_kt_above_10 < 23
        - 0.005392033 * max(0.0, Q.m012 - 36.38136) / 5.489191   # -0.5%  m012 > 36.38
        + 0.005299898 * max(0.0, Q.n_pairs_kt_above_3 - 78.0) / 28.31171   # +0.5%  n_pairs_kt_above_3 > 78
        - 0.005057883 * max(0.0, Q.mass - 110.2019) * max(0.0, Q.psi_0p3 - 0.6500863) / 2.274601   # -0.5%  mass > 110.2 and psi_0p3 > 0.6501
        - 0.004294395 * max(0.0, 0.5498426 - Q.tau32) / 0.03559481   # -0.4%  tau32 < 0.5498
        + 0.004074149 * max(0.0, Q.N2_b2 - 0.1528932) * max(0.0, 12.0 - Q.n_dr_0p05_0p1) / 0.2611676   # +0.4%  N2_b2 > 0.1529 and n_dr_0p05_0p1 < 12
        + 0.003703825 * max(0.0, Q.mass - 79.47361) * max(0.0, 0.1302765 - Q.sj4_dr_min) / 0.9947527   # +0.4%  mass > 79.47 and sj4_dr_min < 0.1303
        - 0.00317458 * max(0.0, 0.06762785 - Q.M2_b2) * max(0.0, Q.sj3_dr_max - 0.2345694) / 0.01083316   # -0.3%  M2_b2 < 0.06763 and sj3_dr_max > 0.2346
        - 0.00300851 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) / 103.3282   # -0.3%  n_pairs_kt_above_1 < 325
        + 0.002915483 * max(0.0, 0.06762785 - Q.M2_b2) * max(0.0, Q.D2_b05 - 1.102602) / 0.02484071   # +0.3%  M2_b2 < 0.06763 and D2_b05 > 1.103
        + 0.002901455 * max(0.0, Q.mass_top50 - 112.2327) / 14.27677   # +0.3%  mass_top50 > 112.2
        - 0.002840857 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # -0.3%  sd_mass > 175.9
        - 0.002722315 * max(0.0, Q.pair_mean_lnm2 - 4.069088) / 0.0918484   # -0.3%  pair_mean_lnm2 > 4.069
        + 0.002722014 * max(0.0, Q.sd_mass - 78.4753) * max(0.0, 0.6509029 - Q.tau43_b2) / 1.778509   # +0.3%  sd_mass > 78.48 and tau43_b2 < 0.6509
        + 0.002542054 * max(0.0, Q.sj3_pair_mass_max - 93.87663) / 11.02189   # +0.3%  sj3_pair_mass_max > 93.88
        + 0.002515649 * max(0.0, Q.M2_b2 - 0.02656308) / 0.01078221   # +0.3%  M2_b2 > 0.02656
        + 0.002399304 * max(0.0, Q.mass - 95.14961) * max(0.0, 1049.334 - Q.sum_e) / 5712.764   # +0.2%  mass > 95.15 and sum_e < 1049
        + 0.002327309 * max(0.0, Q.N2_b2 - 0.1528932) / 0.04637186   # +0.2%  N2_b2 > 0.1529
        + 0.002245779 * max(0.0, Q.M2_b2 - 0.02656308) * max(0.0, 36.36236 - Q.sj3_mass1) / 0.1526005   # +0.2%  M2_b2 > 0.02656 and sj3_mass1 < 36.36
        + 0.002152237 * max(0.0, Q.pair_mean_lnm2 - 4.069088) * max(0.0, 0.3168775 - Q.dr_32) / 0.02624815   # +0.2%  pair_mean_lnm2 > 4.069 and dr_32 < 0.3169
        - 0.002062763 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) * max(0.0, 0.8709334 - Q.tau43) / 9.413515   # -0.2%  n_pairs_kt_above_1 < 325 and tau43 < 0.8709
        - 0.00199054 * max(0.0, 3.734077 - Q.lund_max_lnkt) / 0.2313255   # -0.2%  lund_max_lnkt < 3.734
        + 0.001978846 * max(0.0, Q.M2_b2 - 0.02656308) * max(0.0, Q.jet_pt - 584.4842) / 0.5928601   # +0.2%  M2_b2 > 0.02656 and jet_pt > 584.5
        - 0.001972118 * max(0.0, 0.5440886 - Q.tau21) / 0.1533584   # -0.2%  tau21 < 0.5441
        - 0.001868988 * max(0.0, Q.n_pairs_kt_above_3 - 5.0) * max(0.0, 0.05157804 - Q.dr_32) / 0.5251018   # -0.2%  n_pairs_kt_above_3 > 5 and dr_32 < 0.05158
        + 0.001774714 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # +0.2%  sd_mass > 106.8
        + 0.001690274 * max(0.0, 0.01744666 - Q.mratio_min_012) / 0.0006584817   # +0.2%  mratio_min_012 < 0.01745
        + 0.001504971 * max(0.0, 0.06762785 - Q.M2_b2) * max(0.0, 26.98412 - Q.sj3_mass2) / 0.6137064   # +0.2%  M2_b2 < 0.06763 and sj3_mass2 < 26.98
        - 0.001495247 * max(0.0, Q.sj3_pair_mass_max - 76.91486) * max(0.0, Q.ecf_g41 - 0.0001644218) / 0.001479684   # -0.1%  sj3_pair_mass_max > 76.91 and ecf_g41 > 0.0001644
        - 0.001469142 * max(0.0, Q.mass_top20 - 97.09528) * max(0.0, 0.1192981 - Q.z_3rd) / 0.3630697   # -0.1%  mass_top20 > 97.1 and z_3rd < 0.1193
        - 0.001464709 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.lnpt_7 - 2.07354) / 74.33731   # -0.1%  n_pairs_kt_above_1 < 325 and lnpt_7 > 2.074
        - 0.001414062 * max(0.0, Q.sd_rg - 0.4731839) / 0.02090394   # -0.1%  sd_rg > 0.4732
        - 0.001320021 * max(0.0, 1.617171 - Q.sj3_mass3) * max(0.0, 0.3377343 - Q.tau32_b2) / 0.00574966   # -0.1%  sj3_mass3 < 1.617 and tau32_b2 < 0.3377
        + 0.001252356 * max(0.0, Q.n_pairs_kt_above_3 - 78.0) * max(0.0, Q.lne_0 - 4.992132) / 3.116845   # +0.1%  n_pairs_kt_above_3 > 78 and lne_0 > 4.992
        - 0.001208912 * max(0.0, Q.lam2 - 0.002431555) / 0.004388001   # -0.1%  lam2 > 0.002432
        - 0.001094989 * max(0.0, Q.mass - 95.14961) * max(0.0, 0.3035327 - Q.pt_balance01) / 0.5458512   # -0.1%  mass > 95.15 and pt_balance01 < 0.3035
        - 0.001048001 * max(0.0, 0.01744666 - Q.mratio_min_012) * max(0.0, Q.phi_39 - -0.282959) / 0.0001910501   # -0.1%  mratio_min_012 < 0.01745 and phi_39 > -0.283
        + 0.001030228 * max(0.0, Q.m012 - 36.38136) * max(0.0, Q.dr12 - 0.08983921) / 1.028785   # +0.1%  m012 > 36.38 and dr12 > 0.08984
        + 0.001004733 * max(0.0, Q.sd_mass - 62.03827) * max(0.0, 0.01566153 - Q.dr_25) / 0.06639843   # +0.1%  sd_mass > 62.04 and dr_25 < 0.01566
        - 0.0009674254 * max(0.0, Q.n_pairs_kt_above_3 - 5.0) * max(0.0, 3.298199 - Q.sj3_mass2) / 3.364659   # -0.1%  n_pairs_kt_above_3 > 5 and sj3_mass2 < 3.298
        + 0.0009527686 * max(0.0, Q.sd_mass - 175.9333) * max(0.0, Q.e4 - 3.53193e-06) / 2.785269e-05   # +0.1%  sd_mass > 175.9 and e4 > 3.532e-06
        + 0.0009128871 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # +0.1%  n_pairs_kt_above_1 < 58
        - 0.000871242 * max(0.0, Q.lam2 - 0.002431555) * max(0.0, 0.2530865 - Q.sj2_zsoft) / 0.0001559903   # -0.1%  lam2 > 0.002432 and sj2_zsoft < 0.2531
        - 0.0008553404 * max(0.0, Q.mass - 95.14961) * max(0.0, 0.0225905 - Q.D3_b2) / 0.1230172   # -0.1%  mass > 95.15 and D3_b2 < 0.02259
        - 0.0008035795 * max(0.0, Q.pair_mean_lndelta - -1.352792) / 0.01174331   # -0.1%  pair_mean_lndelta > -1.353
        - 0.0007769242 * max(0.0, Q.sj3_pairmin_over_m - 0.2477126) / 0.09094898   # -0.1%  sj3_pairmin_over_m > 0.2477
        + 0.000743799 * max(0.0, Q.m012 - 36.38136) * max(0.0, 0.294021 - Q.dr_17) / 0.6143771   # +0.1%  m012 > 36.38 and dr_17 < 0.294
        - 0.0006288633 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.tau2 - 0.07264571) / 0.3816425   # -0.1%  n_pairs_kt_above_1 < 325 and tau2 > 0.07265
        + 0.0006281564 * max(0.0, 0.06074238 - Q.tau1) / 0.001113326   # +0.1%  tau1 < 0.06074
        - 0.0005466078 * max(0.0, Q.ecf_g31 - 0.004728078) * max(0.0, Q.e3_b2 - 0.0007909605) / 4.555217e-07   # -0.1%  ecf_g31 > 0.004728 and e3_b2 > 0.000791
        - 0.0004904242 * max(0.0, Q.M2_b2 - 0.02656308) * max(0.0, 0.01566153 - Q.dr_25) / 1.001996e-05   # -0.0%  M2_b2 > 0.02656 and dr_25 < 0.01566
        + 0.0004873819 * max(0.0, Q.sj4_pair_mass_max - 78.76797) * max(0.0, Q.eta_26 - -0.05227661) / 1.349778   # +0.0%  sj4_pair_mass_max > 78.77 and eta_26 > -0.05228
        - 0.0004811765 * max(0.0, 1.617171 - Q.sj3_mass3) / 0.2158965   # -0.0%  sj3_mass3 < 1.617
        + 0.0003997289 * max(0.0, Q.D3_b2 - 0.2418808) / 0.2352366   # +0.0%  D3_b2 > 0.2419
        + 0.0003801423 * max(0.0, Q.ecf_g31 - 0.004728078) * max(0.0, 0.01399549 - Q.sj4_zsoft) / 9.270944e-07   # +0.0%  ecf_g31 > 0.004728 and sj4_zsoft < 0.014
        + 0.0003213924 * max(0.0, 1.617171 - Q.sj3_mass3) * max(0.0, 4.326415 - Q.sj3_mass2) / 0.1218581   # +0.0%  sj3_mass3 < 1.617 and sj3_mass2 < 4.326
        + 0.0002969112 * max(0.0, Q.N2_b2 - 0.1528932) * max(0.0, Q.eta_64 - 0.0) / 0.0007147843   # +0.0%  N2_b2 > 0.1529 and eta_64 > 0
        - 0.000295028 * max(0.0, Q.lne_0 - 6.304449) / 0.01340284   # -0.0%  lne_0 > 6.304
        - 0.0002645514 * max(0.0, 0.5498426 - Q.tau32) * max(0.0, 0.1326904 - Q.phi_33) / 0.005763642   # -0.0%  tau32 < 0.5498 and phi_33 < 0.1327
        + 0.0002415436 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.min_pair_mass - 10.59762) / 32.57011   # +0.0%  n_pairs_kt_above_1 < 325 and min_pair_mass > 10.6
        - 0.0001443622 * max(0.0, Q.lam2 - 0.01524581) / 0.0009195121   # -0.0%  lam2 > 0.01525
        - 0.00014421 * max(0.0, 5.152094 - Q.pair_max_lnm2) / 0.02599185   # -0.0%  pair_max_lnm2 < 5.152
        - 0.0001236489 * max(0.0, Q.pair_mean_lnm2 - 4.069088) * max(0.0, Q.eta_19 - -0.01043701) / 0.007252561   # -0.0%  pair_mean_lnm2 > 4.069 and eta_19 > -0.01044
        - 9.574791e-05 * max(0.0, Q.sd_mass - 175.9333) * max(0.0, 0.3898758 - Q.pt2_over_pt0) / 0.09064241   # -0.0%  sd_mass > 175.9 and pt2_over_pt0 < 0.3899
        - 9.435788e-05 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) * max(0.0, 0.4680886 - Q.tau32_b2) / 0.1378045   # -0.0%  n_pairs_kt_above_1 < 58 and tau32_b2 < 0.4681
        - 7.442839e-05 * max(0.0, Q.sd_mass - 127.8042) * max(0.0, Q.D3 - 0.611688) / 0.1450367   # -0.0%  sd_mass > 127.8 and D3 > 0.6117
        - 6.602002e-05 * max(0.0, Q.ecf_g31 - 0.004728078) * max(0.0, 0.002489417 - Q.dr_17) / 1.330796e-08   # -0.0%  ecf_g31 > 0.004728 and dr_17 < 0.002489
        + 6.593242e-05 * max(0.0, Q.N2_b2 - 0.1528932) * max(0.0, -0.2019043 - Q.eta_1) / 0.0001587138   # +0.0%  N2_b2 > 0.1529 and eta_1 < -0.2019
        - 6.145195e-05 * max(0.0, Q.sd_mass - 127.8042) * max(0.0, Q.D3_b2 - 0.7423282) / 0.3427126   # -0.0%  sd_mass > 127.8 and D3_b2 > 0.7423
        + 5.11032e-05 * max(0.0, Q.m012 - 36.38136) * max(0.0, -0.3327637 - Q.phi_30) / 0.03533451   # +0.0%  m012 > 36.38 and phi_30 < -0.3328
        + 2.088627e-05 * max(0.0, Q.m012 - 36.38136) * max(0.0, Q.eta_55 - 0.1072388) / 0.03096199   # +0.0%  m012 > 36.38 and eta_55 > 0.1072
        + 1.762391e-05 * max(0.0, Q.lnptrel_30 - -5.77735) * max(0.0, 3.928781 - Q.sj2_mass2) / 0.01751884   # +0.0%  lnptrel_30 > -5.777 and sj2_mass2 < 3.929
        + 7.522678e-06 * max(0.0, 0.5498426 - Q.tau32) * max(0.0, Q.eta_77 - 0.0) / 2.28649e-05   # +0.0%  tau32 < 0.5498 and eta_77 > 0
    )
    return z


def neuron_36(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.035494e-06
    )
    return z


def neuron_37(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.70117e-05
    )
    return z


def neuron_38(Q):
    # scale S = 20.24; each line: share * term / its average size
    z = 20.24211 * (-0.003165418
        + 0.3520236 * max(0.0, 0.06117886 - Q.sum_zz_dr2) / 0.02616533   # +35.2%  sum_zz_dr2 < 0.06118
        - 0.3290251 * max(0.0, 0.06154656 - Q.sum_z_dr2) / 0.02639251   # -32.9%  sum_z_dr2 < 0.06155
        + 0.04299539 * max(0.0, 164.4374 - Q.mass) / 52.54489   # +4.3%  mass < 164.4
        - 0.03349102 * max(0.0, 117.4867 - Q.mass) / 15.64803   # -3.3%  mass < 117.5
        + 0.02978352 * max(0.0, 19.0 - Q.n_dr_0p4_up) / 13.21017   # +3.0%  n_dr_0p4_up < 19
        - 0.02495654 * max(0.0, 0.1240514 - Q.e2) / 0.04306553   # -2.5%  e2 < 0.1241
        + 0.02063248 * max(0.0, 118.8408 - Q.mass_top50) / 16.89652   # +2.1%  mass_top50 < 118.8
        - 0.0152988 * max(0.0, Q.sd_mass - 127.8042) / 7.009137   # -1.5%  sd_mass > 127.8
        + 0.01499704 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # +1.5%  sd_mass > 94.56
        - 0.01379208 * max(0.0, Q.sd_mass - 42.31629) / 58.72614   # -1.4%  sd_mass > 42.32
        - 0.01069254 * max(0.0, 105.7234 - Q.mass) / 10.09077   # -1.1%  mass < 105.7
        - 0.0102938 * max(0.0, 0.05733843 - Q.sum_z_dr2_top30) / 0.02479037   # -1.0%  sum_z_dr2_top30 < 0.05734
        + 0.008028197 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # +0.8%  sd_mass > 78.48
        - 0.007265694 * max(0.0, 129.5874 - Q.mass_top50) / 24.20267   # -0.7%  mass_top50 < 129.6
        + 0.006482204 * max(0.0, 162.7874 - Q.mass_top30) * max(0.0, Q.n_dr_0p4_up - 0.0) / 266.7205   # +0.6%  mass_top30 < 162.8 and n_dr_0p4_up > 0
        - 0.006121882 * max(0.0, 162.7874 - Q.mass_top30) / 59.65527   # -0.6%  mass_top30 < 162.8
        + 0.006068017 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +0.6%  sd_mass > 154.6
        - 0.005784966 * max(0.0, 0.1024935 - Q.tau3) / 0.05489442   # -0.6%  tau3 < 0.1025
        - 0.005282142 * max(0.0, 161.1264 - Q.mass_top50) / 50.68933   # -0.5%  mass_top50 < 161.1
        - 0.004444376 * max(0.0, 0.001589861 - Q.e3) / 0.0006923942   # -0.4%  e3 < 0.00159
        + 0.00422058 * max(0.0, 0.7634316 - Q.max_dr) / 0.1364426   # +0.4%  max_dr < 0.7634
        - 0.003143519 * max(0.0, 0.0001729927 - Q.e3_b2) * max(0.0, 40.0 - Q.n_real_top40) / 0.0008303901   # -0.3%  e3_b2 < 0.000173 and n_real_top40 < 40
        - 0.003033072 * max(0.0, Q.lund2_lndelta - -1.411949) / 0.4841443   # -0.3%  lund2_lndelta > -1.412
        + 0.002159886 * max(0.0, 0.1894158 - Q.e2_b05) * max(0.0, 0.5868564 - Q.tau21_b2) / 0.009322675   # +0.2%  e2_b05 < 0.1894 and tau21_b2 < 0.5869
        + 0.002045245 * max(0.0, 0.1894158 - Q.e2_b05) / 0.03521773   # +0.2%  e2_b05 < 0.1894
        + 0.001943498 * max(0.0, 105.7234 - Q.mass) * max(0.0, 0.0281821 - Q.dr_37) / 0.2375643   # +0.2%  mass < 105.7 and dr_37 < 0.02818
        - 0.001916559 * max(0.0, Q.sd_mass - 127.8042) * max(0.0, 0.1204509 - Q.C2_b2) / 0.1980587   # -0.2%  sd_mass > 127.8 and C2_b2 < 0.1205
        + 0.001843238 * max(0.0, Q.mass_top15 - 119.5993) * max(0.0, 0.221369 - Q.C2_b2) / 0.2323189   # +0.2%  mass_top15 > 119.6 and C2_b2 < 0.2214
        - 0.0018427 * max(0.0, Q.mass_top15 - 119.5993) / 2.048216   # -0.2%  mass_top15 > 119.6
        - 0.001838131 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.jet_abs_eta - 0.6771968) / 3.785169   # -0.2%  mass < 117.5 and jet_abs_eta > 0.6772
        + 0.001578507 * max(0.0, Q.mass_top50 - 94.51361) / 24.98854   # +0.2%  mass_top50 > 94.51
        - 0.001571484 * max(0.0, Q.tau32 - 0.6800935) / 0.07219602   # -0.2%  tau32 > 0.6801
        - 0.001439596 * max(0.0, 162.7874 - Q.mass_top30) * max(0.0, Q.C2_b2 - 0.04016973) / 2.678344   # -0.1%  mass_top30 < 162.8 and C2_b2 > 0.04017
        + 0.001424153 * max(0.0, 117.4867 - Q.mass) * max(0.0, 0.02854284 - Q.C3) / 0.1186065   # +0.1%  mass < 117.5 and C3 < 0.02854
        - 0.001312323 * max(0.0, 0.1024935 - Q.tau3) * max(0.0, 0.007407379 - Q.sj4_zsoft) / 1.935355e-05   # -0.1%  tau3 < 0.1025 and sj4_zsoft < 0.007407
        - 0.001312048 * max(0.0, 162.7874 - Q.mass_top30) * max(0.0, Q.lnerel_0 - -1.516067) / 13.53375   # -0.1%  mass_top30 < 162.8 and lnerel_0 > -1.516
        + 0.001169968 * max(0.0, 0.007407379 - Q.sj4_zsoft) / 0.0002646498   # +0.1%  sj4_zsoft < 0.007407
        - 0.001078042 * max(0.0, Q.sd_mass - 42.31629) * max(0.0, Q.sj3_pairmin_over_m - 0.2278414) / 6.389667   # -0.1%  sd_mass > 42.32 and sj3_pairmin_over_m > 0.2278
        + 0.001032624 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.lund_max_lndelta - -1.233804) / 10.39437   # +0.1%  mass < 164.4 and lund_max_lndelta > -1.234
        + 0.0009928951 * max(0.0, 105.7234 - Q.mass) * max(0.0, Q.mass_top3 - 29.24995) / 27.42038   # +0.1%  mass < 105.7 and mass_top3 > 29.25
        - 0.0008898446 * max(0.0, Q.mass_top50 - 94.51361) * max(0.0, 25.64898 - Q.sj4_pair_mass_min) / 181.546   # -0.1%  mass_top50 > 94.51 and sj4_pair_mass_min < 25.65
        - 0.0008680411 * max(0.0, Q.mass_top50 - 94.51361) * max(0.0, 0.2677232 - Q.dr12) / 3.389041   # -0.1%  mass_top50 > 94.51 and dr12 < 0.2677
        + 0.0008276455 * max(0.0, Q.mass_top20 - 93.50967) * max(0.0, 0.4587485 - Q.dr_34) / 2.941458   # +0.1%  mass_top20 > 93.51 and dr_34 < 0.4587
        - 0.0007713744 * max(0.0, 0.0001729927 - Q.e3_b2) * max(0.0, Q.sj3_pairmin_over_m - 0.1383534) / 1.662716e-05   # -0.1%  e3_b2 < 0.000173 and sj3_pairmin_over_m > 0.1384
        - 0.0007368778 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.sj2_mass1 - 53.51926) / 90.7957   # -0.1%  mass < 164.4 and sj2_mass1 > 53.52
        - 0.0007109638 * max(0.0, 0.0001729927 - Q.e3_b2) / 0.0001069897   # -0.1%  e3_b2 < 0.000173
        - 0.0006997815 * max(0.0, 0.1024935 - Q.tau3) * max(0.0, Q.n_dr_0p2_0p4 - 21.0) / 0.02621905   # -0.1%  tau3 < 0.1025 and n_dr_0p2_0p4 > 21
        - 0.0006097256 * max(0.0, 0.00761379 - Q.M3_b2) / 0.0008114698   # -0.1%  M3_b2 < 0.007614
        + 0.0005922368 * max(0.0, Q.mass_top50 - 94.51361) * max(0.0, Q.sj3_dr13 - 0.2949571) / 5.434376   # +0.1%  mass_top50 > 94.51 and sj3_dr13 > 0.295
        + 0.0005588512 * max(0.0, Q.sj2_mass1 - 60.04967) / 4.606883   # +0.1%  sj2_mass1 > 60.05
        - 0.0005179556 * max(0.0, 0.1894158 - Q.e2_b05) * max(0.0, -1.411949 - Q.lund2_lndelta) / 0.008446075   # -0.1%  e2_b05 < 0.1894 and lund2_lndelta < -1.412
        - 0.0005102725 * max(0.0, Q.mass_top20 - 93.50967) / 11.20454   # -0.1%  mass_top20 > 93.51
        + 0.0004859068 * max(0.0, Q.mass_top50 - 94.51361) * max(0.0, Q.mass_top2 - 30.2961) / 148.494   # +0.0%  mass_top50 > 94.51 and mass_top2 > 30.3
        + 0.0004547982 * max(0.0, Q.sum_pt_top3 - 356.8516) / 23.60315   # +0.0%  sum_pt_top3 > 356.9
        + 0.0004157009 * max(0.0, Q.sd_mass - 127.8042) * max(0.0, 3.038341 - Q.D2) / 8.927783   # +0.0%  sd_mass > 127.8 and D2 < 3.038
        + 0.000411514 * max(0.0, Q.z_top30_slots - 0.9283051) / 0.04135436   # +0.0%  z_top30_slots > 0.9283
        + 0.0003838336 * max(0.0, 0.004729211 - Q.z_dr_0p4_up) * max(0.0, 0.03900679 - Q.mratio_min_012) / 4.51859e-06   # +0.0%  z_dr_0p4_up < 0.004729 and mratio_min_012 < 0.03901
        - 0.0003718186 * max(0.0, Q.sd_mass - 154.5947) * max(0.0, 0.9433644 - Q.tau43_b2) / 1.000422   # -0.0%  sd_mass > 154.6 and tau43_b2 < 0.9434
        - 0.0003156393 * max(0.0, 0.1024935 - Q.tau3) * max(0.0, Q.sj2_dr - 0.5454864) / 0.0007495353   # -0.0%  tau3 < 0.1025 and sj2_dr > 0.5455
        + 0.000292831 * max(0.0, Q.lund2_lndelta - -1.411949) * max(0.0, Q.z_dr_0p2_0p4 - 0.4683306) / 0.01139874   # +0.0%  lund2_lndelta > -1.412 and z_dr_0p2_0p4 > 0.4683
        - 0.0002924858 * max(0.0, 0.004729211 - Q.z_dr_0p4_up) / 0.0008408066   # -0.0%  z_dr_0p4_up < 0.004729
        + 0.0002885734 * max(0.0, Q.sd_mass - 42.31629) * max(0.0, Q.lund1_lndelta - -0.4705779) / 8.710501   # +0.0%  sd_mass > 42.32 and lund1_lndelta > -0.4706
        + 0.0002855188 * max(0.0, Q.sd_mass - 94.55722) * max(0.0, 0.3402972 - Q.sd_rg) / 0.07147237   # +0.0%  sd_mass > 94.56 and sd_rg < 0.3403
        + 0.0002511492 * max(0.0, Q.pair_max_lnm2 - 8.097646) / 0.0185767   # +0.0%  pair_max_lnm2 > 8.098
        - 0.0002505772 * max(0.0, 1.868856 - Q.m01) / 0.2758198   # -0.0%  m01 < 1.869
        - 0.0002433025 * max(0.0, 0.01545532 - Q.psi_0p1) / 0.001647647   # -0.0%  psi_0p1 < 0.01546
        - 0.0002420039 * max(0.0, 0.08996752 - Q.sj2_zsoft) / 0.003665091   # -0.0%  sj2_zsoft < 0.08997
        + 0.000241219 * max(0.0, 0.0001729927 - Q.e3_b2) * max(0.0, Q.n_lund - 11.0) / 0.0001123229   # +0.0%  e3_b2 < 0.000173 and n_lund > 11
        - 0.0002382981 * max(0.0, Q.mass_top50 - 94.51361) * max(0.0, Q.tau43_b2 - 0.4866492) / 5.450705   # -0.0%  mass_top50 > 94.51 and tau43_b2 > 0.4866
        - 0.0002147156 * max(0.0, 162.7874 - Q.mass_top30) * max(0.0, Q.eta_0 - 0.08172607) / 0.3524506   # -0.0%  mass_top30 < 162.8 and eta_0 > 0.08173
        - 0.0001955316 * max(0.0, Q.mass_top40 - 141.9283) * max(0.0, 10.0 - Q.n_lund) / 3.31173   # -0.0%  mass_top40 > 141.9 and n_lund < 10
        - 0.0001833007 * max(0.0, 0.2686235 - Q.LHA) / 0.00660313   # -0.0%  LHA < 0.2686
        + 0.0001815428 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, Q.sum_z_dr2_top5 - 0.02100028) / 0.1980051   # +0.0%  mass_top50 < 161.1 and sum_z_dr2_top5 > 0.021
        - 0.0001587393 * max(0.0, Q.mass_top40 - 141.9283) / 4.449513   # -0.0%  mass_top40 > 141.9
        - 0.0001327388 * max(0.0, Q.mass_top50 - 94.51361) * max(0.0, 0.806533 - Q.sj3_pairmax_over_m) / 1.183827   # -0.0%  mass_top50 > 94.51 and sj3_pairmax_over_m < 0.8065
        - 0.0001314488 * max(0.0, Q.sj2_mass1 - 77.42768) / 2.022907   # -0.0%  sj2_mass1 > 77.43
        + 0.0001008904 * max(0.0, Q.sd_mass - 154.5947) * max(0.0, 0.1776883 - Q.N3_b2) / 0.1514683   # +0.0%  sd_mass > 154.6 and N3_b2 < 0.1777
        - 9.901754e-05 * max(0.0, Q.lund_max_lndelta - -0.4168051) / 0.004389625   # -0.0%  lund_max_lndelta > -0.4168
        - 8.711882e-05 * max(0.0, Q.sd_mass - 154.5947) * max(0.0, Q.lnptrel_77 - -18.42068) / 5.372699   # -0.0%  sd_mass > 154.6 and lnptrel_77 > -18.42
        + 7.860774e-05 * max(0.0, Q.sd_mass - 127.8042) * max(0.0, 5.0 - Q.n_lund) / 0.5872355   # +0.0%  sd_mass > 127.8 and n_lund < 5
        - 7.486915e-05 * max(0.0, Q.mass_top50 - 94.51361) * max(0.0, 0.3533641 - Q.dr_12) / 3.293554   # -0.0%  mass_top50 > 94.51 and dr_12 < 0.3534
        + 7.354816e-05 * max(0.0, Q.sd_mass - 154.5947) * max(0.0, 11.0 - Q.n_lund) / 3.162842   # +0.0%  sd_mass > 154.6 and n_lund < 11
        - 6.164112e-05 * max(0.0, Q.sd_mass - 42.31629) * max(0.0, Q.eta_55 - 0.0) / 1.575882   # -0.0%  sd_mass > 42.32 and eta_55 > 0
        + 3.348253e-05 * max(0.0, Q.mass_top20 - 93.50967) * max(0.0, 0.6627397 - Q.sj3_pairmax_over_m) / 0.02010499   # +0.0%  mass_top20 > 93.51 and sj3_pairmax_over_m < 0.6627
        + 2.479207e-05 * max(0.0, Q.mass_top15 - 119.5993) * max(0.0, 0.09424246 - Q.z_2nd) / 0.01504788   # +0.0%  mass_top15 > 119.6 and z_2nd < 0.09424
        + 2.434349e-05 * max(0.0, Q.mass_top40 - 141.9283) * max(0.0, 0.09015482 - Q.dr_12) / 0.0101324   # +0.0%  mass_top40 > 141.9 and dr_12 < 0.09015
        + 1.780336e-05 * Q.dr_46 / 0.09048465   # +0.0%  dr_46
        - 2.984822e-06 * max(0.0, 0.1024935 - Q.tau3) * max(0.0, Q.dr_41 - 0.3126638) / 0.0009911504   # -0.0%  tau3 < 0.1025 and dr_41 > 0.3127
        + 2.794049e-06 * max(0.0, Q.mass_top15 - 119.5993) * max(0.0, Q.e4_b2 - 7.0969e-08) / 4.691616e-06   # +0.0%  mass_top15 > 119.6 and e4_b2 > 7.097e-08
        - 2.043544e-06 * max(0.0, Q.mass_top15 - 119.5993) * max(0.0, 0.08382604 - Q.dr_37) / 0.06943353   # -0.0%  mass_top15 > 119.6 and dr_37 < 0.08383
        - 1.837453e-06 * max(0.0, Q.pair_max_lnm2 - 8.097646) * max(0.0, Q.eta_28 - 0.09893799) / 0.0005865674   # -0.0%  pair_max_lnm2 > 8.098 and eta_28 > 0.09894
    )
    return z


def neuron_39(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.904104e-06
    )
    return z


def neuron_40(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.003766e-06
    )
    return z


def neuron_41(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.526475e-06
    )
    return z


def neuron_42(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.68051e-06
    )
    return z


def neuron_43(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.8981e-06
    )
    return z


def neuron_44(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.332824e-05
    )
    return z


def neuron_45(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.218037e-05
    )
    return z


def neuron_46(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.101478e-06
    )
    return z


def neuron_47(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.13496e-05
    )
    return z


def neuron_48(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.639793e-07
    )
    return z


def neuron_49(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.90266e-05
    )
    return z


def neuron_50(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.526367e-05
    )
    return z


def neuron_51(Q):
    # scale S = 6.265; each line: share * term / its average size
    z = 6.265136 * (0.06020328
        - 0.1067023 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # -10.7%  sd_mass > 78.48
        + 0.1013684 * max(0.0, 0.1400425 - Q.e2) / 0.05789227   # +10.1%  e2 < 0.14
        - 0.0590741 * Q.M2_b2 / 0.03249042   # -5.9%  M2_b2
        - 0.05887741 * max(0.0, 0.05045808 - Q.tau5) / 0.02019296   # -5.9%  tau5 < 0.05046
        - 0.05353774 * max(0.0, 127.8042 - Q.sd_mass) / 37.77282   # -5.4%  sd_mass < 127.8
        + 0.0385942 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # +3.9%  sd_mass > 106.8
        - 0.03586495 * max(0.0, 3.9754 - Q.lund_max_lnkt) / 0.3336278   # -3.6%  lund_max_lnkt < 3.975
        - 0.03187697 * max(0.0, 0.02938903 - Q.tau5) / 0.005467528   # -3.2%  tau5 < 0.02939
        + 0.03099579 * max(0.0, 105.3556 - Q.mass_top40) / 10.73691   # +3.1%  mass_top40 < 105.4
        + 0.025793 * max(0.0, 35.0 - Q.n_pt_above_5) / 13.58272   # +2.6%  n_pt_above_5 < 35
        + 0.02552562 * max(0.0, 100.4835 - Q.mass) / 8.115061   # +2.6%  mass < 100.5
        - 0.02185915 * max(0.0, 9.621843e-05 - Q.e3_b2) / 4.881439e-05   # -2.2%  e3_b2 < 9.622e-05
        + 0.02113278 * max(0.0, 128.6605 - Q.sj3_pair_mass_max) / 38.7386   # +2.1%  sj3_pair_mass_max < 128.7
        + 0.01991024 * max(0.0, Q.sd_mass - 106.7501) * max(0.0, 15.17086 - Q.D2_b2) / 188.2717   # +2.0%  sd_mass > 106.8 and D2_b2 < 15.17
        + 0.01946813 * max(0.0, 3.9754 - Q.lund_max_lnkt) * max(0.0, 1.0 - Q.n_pairs_kt_above_30) / 0.3293485   # +1.9%  lund_max_lnkt < 3.975 and n_pairs_kt_above_30 < 1
        - 0.01914497 * max(0.0, Q.mass_top30 - 79.54717) / 28.43367   # -1.9%  mass_top30 > 79.55
        - 0.01832913 * max(0.0, 105.7234 - Q.mass) / 10.09077   # -1.8%  mass < 105.7
        + 0.01809561 * Q.sum_e / 905.8561   # +1.8%  sum_e
        - 0.01587067 * max(0.0, 0.7544983 - Q.tau32) / 0.1196328   # -1.6%  tau32 < 0.7545
        + 0.01577185 * max(0.0, 0.02605908 - Q.lam1) / 0.004700161   # +1.6%  lam1 < 0.02606
        + 0.01552991 * max(0.0, 1.953274 - Q.D2) / 0.3946001   # +1.6%  D2 < 1.953
        - 0.01360882 * max(0.0, 3.0 - Q.n_dr_0_0p05) / 1.40206   # -1.4%  n_dr_0_0p05 < 3
        + 0.01262242 * max(0.0, 0.2009044 - Q.sd_rg) / 0.02300866   # +1.3%  sd_rg < 0.2009
        + 0.01154442 * max(0.0, 90.35243 - Q.sj3_pair_mass_max) / 10.69708   # +1.2%  sj3_pair_mass_max < 90.35
        - 0.01046863 * max(0.0, Q.pair_max_lnkt - 2.43449) / 0.4188007   # -1.0%  pair_max_lnkt > 2.434
        - 0.009992981 * max(0.0, Q.mass_top30 - 79.54717) * max(0.0, 0.7462286 - Q.sj3_dr13) / 8.357536   # -1.0%  mass_top30 > 79.55 and sj3_dr13 < 0.7462
        - 0.009485569 * max(0.0, 104.4937 - Q.mass_top15) * max(0.0, Q.M3_b2 - 0.00678572) / 0.2344338   # -0.9%  mass_top15 < 104.5 and M3_b2 > 0.006786
        + 0.008770206 * max(0.0, Q.sj3_pairmax_over_m - 0.7925455) / 0.04609974   # +0.9%  sj3_pairmax_over_m > 0.7925
        + 0.008355465 * max(0.0, 24.0 - Q.n_dr_0p2_0p4) / 13.19173   # +0.8%  n_dr_0p2_0p4 < 24
        - 0.007714554 * max(0.0, Q.pair_mean_lnm2 - 2.458746) / 0.4996178   # -0.8%  pair_mean_lnm2 > 2.459
        + 0.007535899 * max(0.0, 0.1072997 - Q.z_dr_0_0p05) / 0.06409855   # +0.8%  z_dr_0_0p05 < 0.1073
        - 0.007500341 * max(0.0, 8.588303 - Q.sj2_mass2) / 1.129506   # -0.8%  sj2_mass2 < 8.588
        + 0.007282529 * max(0.0, 9.621843e-05 - Q.e3_b2) * max(0.0, Q.lund3_lndelta - -2.817283) / 5.359263e-05   # +0.7%  e3_b2 < 9.622e-05 and lund3_lndelta > -2.817
        + 0.007017763 * max(0.0, Q.pair_max_lnm2 - 6.840102) / 0.2515609   # +0.7%  pair_max_lnm2 > 6.84
        + 0.006798664 * max(0.0, 0.02982432 - Q.dr02) / 0.005753371   # +0.7%  dr02 < 0.02982
        + 0.006070416 * max(0.0, Q.M2_b05 - 0.1506455) / 0.005094623   # +0.6%  M2_b05 > 0.1506
        - 0.005490236 * max(0.0, 0.1400425 - Q.e2) * max(0.0, -0.002507949 - Q.mean_eta) / 0.0002914804   # -0.5%  e2 < 0.14 and mean_eta < -0.002508
        + 0.00541923 * max(0.0, Q.mass_top30 - 134.2224) / 4.038118   # +0.5%  mass_top30 > 134.2
        + 0.004897824 * max(0.0, Q.pair_max_lnm2 - 6.840102) * max(0.0, 0.1762573 - Q.dr_31) / 0.02644282   # +0.5%  pair_max_lnm2 > 6.84 and dr_31 < 0.1763
        - 0.004798679 * max(0.0, 1.953274 - Q.D2) * max(0.0, 0.3258728 - Q.dr01) / 0.05630781   # -0.5%  D2 < 1.953 and dr01 < 0.3259
        + 0.004766152 * max(0.0, 5.0 - Q.n_lund_kt_above_1) / 0.64189   # +0.5%  n_lund_kt_above_1 < 5
        - 0.004755381 * max(0.0, 0.01108077 - Q.C3_b2) / 0.006695724   # -0.5%  C3_b2 < 0.01108
        - 0.004615359 * max(0.0, Q.pair_max_lnkt - 2.43449) * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 2.511701   # -0.5%  pair_max_lnkt > 2.434 and n_dr_0p1_0p2 < 17
        - 0.004516718 * max(0.0, 0.009042742 - Q.sum_z_dr2_top2) / 0.002108874   # -0.5%  sum_z_dr2_top2 < 0.009043
        + 0.004050134 * max(0.0, 90.35243 - Q.sj3_pair_mass_max) * max(0.0, 0.191004 - Q.dr01) / 1.212771   # +0.4%  sj3_pair_mass_max < 90.35 and dr01 < 0.191
        + 0.003943953 * max(0.0, 104.4937 - Q.mass_top15) / 26.57384   # +0.4%  mass_top15 < 104.5
        - 0.00385734 * max(0.0, 105.3556 - Q.mass_top40) * max(0.0, 0.4358177 - Q.pt1_over_pt0) / 0.560849   # -0.4%  mass_top40 < 105.4 and pt1_over_pt0 < 0.4358
        - 0.003524086 * max(0.0, 3.0 - Q.n_dr_0p05_0p1) / 0.5535833   # -0.4%  n_dr_0p05_0p1 < 3
        + 0.003355165 * max(0.0, 0.03574519 - Q.dr01) * max(0.0, 15.17086 - Q.D2_b2) / 0.09870477   # +0.3%  dr01 < 0.03575 and D2_b2 < 15.17
        + 0.0031996 * max(0.0, Q.m012 - 57.33311) / 1.633981   # +0.3%  m012 > 57.33
        + 0.003184942 * max(0.0, 0.7544983 - Q.tau32) * max(0.0, 0.1259759 - Q.sj3_dr_min) / 0.0008432644   # +0.3%  tau32 < 0.7545 and sj3_dr_min < 0.126
        + 0.003110441 * max(0.0, 0.05045808 - Q.tau5) * max(0.0, Q.sj3_dr_min - 0.1611011) / 0.001091756   # +0.3%  tau5 < 0.05046 and sj3_dr_min > 0.1611
        + 0.002745357 * max(0.0, 0.03574519 - Q.dr01) / 0.008740061   # +0.3%  dr01 < 0.03575
        - 0.002601622 * max(0.0, Q.pair_max_lnkt - 2.43449) * max(0.0, Q.sj3_z1 - 0.5904161) / 0.01484156   # -0.3%  pair_max_lnkt > 2.434 and sj3_z1 > 0.5904
        + 0.002560576 * max(0.0, 111.0 - Q.n_pairs_kt_above_3) / 47.95244   # +0.3%  n_pairs_kt_above_3 < 111
        - 0.002458701 * max(0.0, Q.pt_dispersion - 0.5051136) / 0.01184782   # -0.2%  pt_dispersion > 0.5051
        - 0.002414197 * max(0.0, 104.4937 - Q.mass_top15) * max(0.0, Q.pair_mean_lnm2 - 2.340672) / 6.692281   # -0.2%  mass_top15 < 104.5 and pair_mean_lnm2 > 2.341
        + 0.002336995 * max(0.0, Q.sj2_dr - 0.5076533) / 0.02194993   # +0.2%  sj2_dr > 0.5077
        + 0.002221049 * max(0.0, Q.mass_top30 - 79.54717) * max(0.0, -1.763117 - Q.lund3_lnz) / 56.61136   # +0.2%  mass_top30 > 79.55 and lund3_lnz < -1.763
        + 0.001995611 * max(0.0, Q.mass_top10 - 103.4976) * max(0.0, 4.0 - Q.n_lund_kt_above_5) / 3.498649   # +0.2%  mass_top10 > 103.5 and n_lund_kt_above_5 < 4
        - 0.001966755 * max(0.0, 0.03574519 - Q.dr01) * max(0.0, 0.1458275 - Q.sj4_dr_min) / 0.0004473576   # -0.2%  dr01 < 0.03575 and sj4_dr_min < 0.1458
        - 0.001954878 * max(0.0, 128.6605 - Q.sj3_pair_mass_max) * max(0.0, 1.618769 - Q.lnpt_29) / 328.6632   # -0.2%  sj3_pair_mass_max < 128.7 and lnpt_29 < 1.619
        + 0.001865131 * max(0.0, 0.2852651 - Q.D3) / 0.0873966   # +0.2%  D3 < 0.2853
        - 0.001846685 * max(0.0, Q.e4_b05 - 0.000141779) / 4.111841e-06   # -0.2%  e4_b05 > 0.0001418
        - 0.001636242 * max(0.0, 0.7544983 - Q.tau32) * max(0.0, 3.748488 - Q.sj3_mass3) / 0.06215929   # -0.2%  tau32 < 0.7545 and sj3_mass3 < 3.748
        + 0.001563326 * max(0.0, Q.tau5 - 0.05613495) / 0.001237361   # +0.2%  tau5 > 0.05613
        + 0.001137074 * max(0.0, Q.lne_1 - 5.29456) / 0.0281941   # +0.1%  lne_1 > 5.295
        - 0.001070616 * max(0.0, Q.mass_top30 - 79.54717) * max(0.0, -0.831543 - Q.lund1_lndelta) / 0.5616841   # -0.1%  mass_top30 > 79.55 and lund1_lndelta < -0.8315
        - 0.0009372164 * max(0.0, 16.0 - Q.n_pairs_kt_above_3) / 1.257407   # -0.1%  n_pairs_kt_above_3 < 16
        - 0.0009006179 * max(0.0, 0.02981375 - Q.dr_max_012) / 0.002629335   # -0.1%  dr_max_012 < 0.02981
        + 0.0008692246 * max(0.0, 8.588303 - Q.sj2_mass2) * max(0.0, Q.dr01 - 0.2849189) / 0.03191009   # +0.1%  sj2_mass2 < 8.588 and dr01 > 0.2849
        - 0.0007558093 * max(0.0, 2.961008 - Q.lund_max_lnkt) / 0.07658753   # -0.1%  lund_max_lnkt < 2.961
        + 0.0007211324 * max(0.0, Q.mass_top30 - 79.54717) * max(0.0, Q.D3 - 0.04661539) / 8.357887   # +0.1%  mass_top30 > 79.55 and D3 > 0.04662
        + 0.0006649407 * max(0.0, 0.4954556 - Q.N2_b05) * max(0.0, Q.n_pt_above_10 - 6.0) / 0.3628235   # +0.1%  N2_b05 < 0.4955 and n_pt_above_10 > 6
        + 0.0006053309 * max(0.0, 0.4954556 - Q.N2_b05) / 0.06560395   # +0.1%  N2_b05 < 0.4955
        + 0.0005649244 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # +0.1%  mass_top10 > 103.5
        - 0.0005543704 * max(0.0, Q.pair_max_lnm2 - 6.840102) * max(0.0, Q.dr_8 - 0.1767313) / 0.0225828   # -0.1%  pair_max_lnm2 > 6.84 and dr_8 > 0.1767
        + 0.0005395475 * max(0.0, 104.4937 - Q.mass_top15) * max(0.0, Q.C2_b2 - 0.08076625) / 0.9469303   # +0.1%  mass_top15 < 104.5 and C2_b2 > 0.08077
        + 0.0004627547 * max(0.0, Q.pair_mean_lnm2 - 4.069088) / 0.0918484   # +0.0%  pair_mean_lnm2 > 4.069
        - 0.0004598273 * max(0.0, Q.M2_b05 - 0.1506455) * max(0.0, 0.4580564 - Q.pt2_over_pt0) / 0.0001918601   # -0.0%  M2_b05 > 0.1506 and pt2_over_pt0 < 0.4581
        + 0.0004547406 * max(0.0, 0.05045808 - Q.tau5) * max(0.0, Q.sd_zg - 0.2652027) / 0.001151501   # +0.0%  tau5 < 0.05046 and sd_zg > 0.2652
        + 0.0003453147 * max(0.0, Q.mass_top10 - 103.4976) * max(0.0, 6.383481 - Q.min_pair_mass) / 6.546709   # +0.0%  mass_top10 > 103.5 and min_pair_mass < 6.383
        - 0.0003185188 * max(0.0, Q.m012 - 69.67489) * max(0.0, Q.lnpt_29 - 0.2696991) / 0.1781519   # -0.0%  m012 > 69.67 and lnpt_29 > 0.2697
        - 0.0003104504 * max(0.0, 3.9754 - Q.lund_max_lnkt) * max(0.0, -0.0848999 - Q.eta_0) / 0.0004282662   # -0.0%  lund_max_lnkt < 3.975 and eta_0 < -0.0849
        + 0.0003080076 * max(0.0, 70.93762 - Q.mass_top15) / 6.558806   # +0.0%  mass_top15 < 70.94
        + 0.0003037121 * max(0.0, Q.z_dr_0p1_0p2 - 0.5192063) / 0.05044446   # +0.0%  z_dr_0p1_0p2 > 0.5192
        + 0.0002816402 * max(0.0, 0.02981375 - Q.dr_max_012) * max(0.0, 0.03826904 - Q.eta_12) / 0.0002398773   # +0.0%  dr_max_012 < 0.02981 and eta_12 < 0.03827
        + 0.0002584954 * max(0.0, 0.1209237 - Q.M2_b05) / 0.00591466   # +0.0%  M2_b05 < 0.1209
        + 0.000249574 * max(0.0, Q.m012 - 69.67489) / 0.7542893   # +0.0%  m012 > 69.67
        + 0.0002222445 * max(0.0, Q.M2_b05 - 0.1506455) * max(0.0, 0.4240535 - Q.tau43_b2) / 5.159955e-05   # +0.0%  M2_b05 > 0.1506 and tau43_b2 < 0.4241
        + 0.0001901872 * max(0.0, Q.pair_max_lnm2 - 6.840102) * max(0.0, 0.1497803 - Q.eta_25) / 0.04508403   # +0.0%  pair_max_lnm2 > 6.84 and eta_25 < 0.1498
        + 0.0001899061 * max(0.0, Q.lnpt_6 - 3.082483) * max(0.0, -0.05612183 - Q.eta_14) / 0.005515627   # +0.0%  lnpt_6 > 3.082 and eta_14 < -0.05612
        - 0.0001391792 * max(0.0, 0.4954556 - Q.N2_b05) * max(0.0, Q.eta_41 - -0.1331787) / 0.009400749   # -0.0%  N2_b05 < 0.4955 and eta_41 > -0.1332
        + 0.0001364762 * max(0.0, Q.M2_b05 - 0.1506455) * max(0.0, 0.3886647 - Q.sj3_mass3) / 0.0001008306   # +0.0%  M2_b05 > 0.1506 and sj3_mass3 < 0.3887
        - 9.924502e-05 * max(0.0, Q.lnpt_6 - 3.082483) / 0.1508566   # -0.0%  lnpt_6 > 3.082
        - 9.402934e-05 * max(0.0, 0.03574519 - Q.dr01) * max(0.0, -0.07635498 - Q.eta_2) / 0.0001388104   # -0.0%  dr01 < 0.03575 and eta_2 < -0.07635
        - 3.95786e-05 * max(0.0, Q.sd_mass - 106.7501) * max(0.0, 0.0 - Q.phi_72) / 0.1630938   # -0.0%  sd_mass > 106.8 and phi_72 < 0
        - 3.815454e-05 * max(0.0, Q.sj2_dr - 0.5076533) * max(0.0, 1.325034 - Q.D2) / 0.0009775364   # -0.0%  sj2_dr > 0.5077 and D2 < 1.325
        - 3.324002e-05 * max(0.0, Q.pair_mean_lnm2 - 2.458746) * max(0.0, Q.eta_73 - 0.0) / 0.0004760423   # -0.0%  pair_mean_lnm2 > 2.459 and eta_73 > 0
    )
    return z


def neuron_52(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.688201e-07
    )
    return z


def neuron_53(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.044334e-06
    )
    return z


def neuron_54(Q):
    # scale S = 21.98; each line: share * term / its average size
    z = 21.98264 * (-0.004637246
        + 0.1021068 * max(0.0, 175.9333 - Q.sd_mass) / 80.62387   # +10.2%  sd_mass < 175.9
        - 0.09114069 * max(0.0, Q.sd_mass - 42.31629) / 58.72614   # -9.1%  sd_mass > 42.32
        + 0.0906985 * max(0.0, Q.mass - 79.47361) / 38.31536   # +9.1%  mass > 79.47
        - 0.07235722 * max(0.0, Q.m01 - 3.294888) / 12.92518   # -7.2%  m01 > 3.295
        - 0.06408849 * max(0.0, 123.2919 - Q.sd_mass) / 34.26699   # -6.4%  sd_mass < 123.3
        + 0.06164325 * max(0.0, Q.sd_mass - 83.61981) / 26.48984   # +6.2%  sd_mass > 83.62
        + 0.06095765 * max(0.0, Q.m01 - 4.577187) / 12.18994   # +6.1%  m01 > 4.577
        - 0.0388168 * max(0.0, Q.mass - 117.4867) / 13.09248   # -3.9%  mass > 117.5
        - 0.03689902 * max(0.0, 873.0 - Q.n_pairs_kt_above_1) / 556.2057   # -3.7%  n_pairs_kt_above_1 < 873
        + 0.0260549 * max(0.0, 4.787589 - Q.pair_mean_lnm2) / 2.254419   # +2.6%  pair_mean_lnm2 < 4.788
        + 0.02576564 * max(0.0, 0.07109528 - Q.sum_z_dr2_top5) / 0.0476681   # +2.6%  sum_z_dr2_top5 < 0.0711
        - 0.02348711 * max(0.0, 19.0 - Q.n_dr_0p2_0p4) / 8.90226   # -2.3%  n_dr_0p2_0p4 < 19
        - 0.02109861 * max(0.0, 1.166286 - Q.N3_b05) / 0.4582066   # -2.1%  N3_b05 < 1.166
        - 0.01820622 * max(0.0, 111.0 - Q.n_pairs_kt_above_3) / 47.95244   # -1.8%  n_pairs_kt_above_3 < 111
        + 0.01457868 * max(0.0, 0.0001729927 - Q.e3_b2) / 0.0001069897   # +1.5%  e3_b2 < 0.000173
        - 0.01450074 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # -1.5%  n_pairs_kt_above_1 < 366
        - 0.01332341 * max(0.0, 0.052589 - Q.e2_b2) / 0.02247618   # -1.3%  e2_b2 < 0.05259
        + 0.0128316 * max(0.0, 0.05500758 - Q.mass_over_sum_pt_sq) / 0.0209996   # +1.3%  mass_over_sum_pt_sq < 0.05501
        + 0.01236049 * max(0.0, Q.pt_entropy - 2.282975) / 0.5784992   # +1.2%  pt_entropy > 2.283
        - 0.01103955 * max(0.0, 0.6647889 - Q.sj2_dr) / 0.2815025   # -1.1%  sj2_dr < 0.6648
        - 0.009879085 * max(0.0, 0.07625067 - Q.M3_b05) / 0.02163642   # -1.0%  M3_b05 < 0.07625
        + 0.009685372 * max(0.0, 77.42768 - Q.sj2_mass1) / 39.01947   # +1.0%  sj2_mass1 < 77.43
        - 0.00961101 * max(0.0, 40.0 - Q.n_particles) / 6.510063   # -1.0%  n_particles < 40
        + 0.009409318 * max(0.0, Q.sj3_pair_mass_max - 90.35243) / 12.68545   # +0.9%  sj3_pair_mass_max > 90.35
        + 0.008327144 * max(0.0, Q.z_top10_slots - 0.7362734) / 0.06667364   # +0.8%  z_top10_slots > 0.7363
        - 0.00832581 * max(0.0, -2.115543 - Q.pair_mean_lndelta) / 0.285167   # -0.8%  pair_mean_lndelta < -2.116
        + 0.007724311 * max(0.0, 0.1724455 - Q.z_dr_0p2_0p4) / 0.06225009   # +0.8%  z_dr_0p2_0p4 < 0.1724
        - 0.007290823 * max(0.0, Q.sd_mass - 127.8042) / 7.009137   # -0.7%  sd_mass > 127.8
        - 0.00695767 * max(0.0, 195.0 - Q.n_pairs_kt_above_1) / 37.35244   # -0.7%  n_pairs_kt_above_1 < 195
        - 0.005702065 * max(0.0, Q.jet_abs_eta - 0.3274899) / 0.4554064   # -0.6%  jet_abs_eta > 0.3275
        - 0.00506767 * max(0.0, 2.102317 - Q.min_pair_mass) / 0.8511323   # -0.5%  min_pair_mass < 2.102
        + 0.005039599 * max(0.0, Q.dr01 - 0.1335998) / 0.08309635   # +0.5%  dr01 > 0.1336
        - 0.004620775 * max(0.0, Q.mass_top40 - 126.8853) / 7.342254   # -0.5%  mass_top40 > 126.9
        + 0.004564034 * max(0.0, 3.734077 - Q.lund_max_lnkt) / 0.2313255   # +0.5%  lund_max_lnkt < 3.734
        + 0.004545807 * max(0.0, 0.3740528 - Q.sj2_dr) / 0.05324013   # +0.5%  sj2_dr < 0.3741
        - 0.004230682 * max(0.0, 0.5498426 - Q.tau32) / 0.03559481   # -0.4%  tau32 < 0.5498
        + 0.0037473 * max(0.0, 0.02398667 - Q.M2_b2) / 0.003762047   # +0.4%  M2_b2 < 0.02399
        - 0.003653672 * max(0.0, 39.95503 - Q.sj2_mass1) / 9.944929   # -0.4%  sj2_mass1 < 39.96
        - 0.003411252 * max(0.0, 0.6647889 - Q.sj2_dr) * max(0.0, Q.M3_b2 - 0.004141442) / 0.003105169   # -0.3%  sj2_dr < 0.6648 and M3_b2 > 0.004141
        - 0.003390386 * max(0.0, Q.mass - 79.47361) * max(0.0, 17.0 - Q.n_pairs_kt_above_10) / 296.4903   # -0.3%  mass > 79.47 and n_pairs_kt_above_10 < 17
        + 0.00317865 * max(0.0, 0.4545826 - Q.N3_b2) / 0.1894157   # +0.3%  N3_b2 < 0.4546
        - 0.003044583 * Q.sum_z_dr2_top2 / 0.02204908   # -0.3%  sum_z_dr2_top2
        - 0.003020154 * max(0.0, 0.3886647 - Q.sj3_mass3) / 0.03562942   # -0.3%  sj3_mass3 < 0.3887
        - 0.002968064 * max(0.0, Q.n_lund_kt_above_1 - 5.0) / 0.9400367   # -0.3%  n_lund_kt_above_1 > 5
        - 0.002918896 * max(0.0, 0.03513777 - Q.M2_b2) / 0.009369713   # -0.3%  M2_b2 < 0.03514
        + 0.002859514 * max(0.0, 0.1573433 - Q.M2_b05) / 0.02356419   # +0.3%  M2_b05 < 0.1573
        + 0.002614791 * max(0.0, Q.mass - 164.4374) / 3.038568   # +0.3%  mass > 164.4
        + 0.001982974 * max(0.0, Q.C3_b05 - 0.1094329) / 0.02188935   # +0.2%  C3_b05 > 0.1094
        - 0.00188629 * max(0.0, Q.sj3_pair_mass_max - 128.6605) / 2.418855   # -0.2%  sj3_pair_mass_max > 128.7
        + 0.001883771 * max(0.0, 0.3886647 - Q.sj3_mass3) * max(0.0, 0.06149175 - Q.sj3_z3) / 0.001326498   # +0.2%  sj3_mass3 < 0.3887 and sj3_z3 < 0.06149
        + 0.00182882 * max(0.0, Q.sd_mass - 115.7091) / 10.28199   # +0.2%  sd_mass > 115.7
        + 0.001741312 * max(0.0, 19.01187 - Q.sj4_pair_mass_min) / 5.921968   # +0.2%  sj4_pair_mass_min < 19.01
        + 0.00170645 * max(0.0, 0.2083105 - Q.sj2_dr) / 0.00695417   # +0.2%  sj2_dr < 0.2083
        + 0.001673098 * max(0.0, 0.5498426 - Q.tau32) * max(0.0, 14.0 - Q.n_lund) / 0.1604382   # +0.2%  tau32 < 0.5498 and n_lund < 14
        - 0.001636283 * max(0.0, 4.326415 - Q.sj3_mass2) / 0.2966808   # -0.2%  sj3_mass2 < 4.326
        + 0.001617633 * max(0.0, Q.mass_top5 - 54.36876) / 5.778115   # +0.2%  mass_top5 > 54.37
        + 0.001594801 * max(0.0, 0.2542227 - Q.pair_mean_lnkt) / 0.08124362   # +0.2%  pair_mean_lnkt < 0.2542
        - 0.001512117 * max(0.0, 175.9333 - Q.sd_mass) * max(0.0, 0.03350185 - Q.sj4_zsoft) / 0.7131627   # -0.2%  sd_mass < 175.9 and sj4_zsoft < 0.0335
        + 0.001451852 * max(0.0, Q.lne_16 - 1.463328) / 0.7979125   # +0.1%  lne_16 > 1.463
        + 0.001363392 * max(0.0, Q.pair_mean_lnkt - 0.7891509) / 0.09822302   # +0.1%  pair_mean_lnkt > 0.7892
        + 0.001334473 * max(0.0, Q.lne_16 - 1.463328) * max(0.0, 0.8531024 - Q.tau43_b2) / 0.124609   # +0.1%  lne_16 > 1.463 and tau43_b2 < 0.8531
        - 0.001283529 * max(0.0, 111.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.dr02 - 0.017759) / 6.673578   # -0.1%  n_pairs_kt_above_3 < 111 and dr02 > 0.01776
        + 0.001206938 * max(0.0, 0.05063855 - Q.dr_min_012) / 0.02812661   # +0.1%  dr_min_012 < 0.05064
        + 0.001168402 * max(0.0, Q.sj3_dr_min - 0.3526989) / 0.01394539   # +0.1%  sj3_dr_min > 0.3527
        - 0.001123063 * max(0.0, Q.lne_16 - 1.463328) * max(0.0, 9.0 - Q.n_dr_0p1_0p2) / 0.8680983   # -0.1%  lne_16 > 1.463 and n_dr_0p1_0p2 < 9
        + 0.001075151 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) * max(0.0, 6.006437 - Q.sj3_mass2) / 159.4739   # +0.1%  n_pairs_kt_above_1 < 366 and sj3_mass2 < 6.006
        + 0.0009939842 * max(0.0, 0.05551408 - Q.dr02) / 0.01617261   # +0.1%  dr02 < 0.05551
        + 0.0008893527 * max(0.0, 0.7366642 - Q.sj3_pairmax_over_m) / 0.0110733   # +0.1%  sj3_pairmax_over_m < 0.7367
        + 0.0007692368 * max(0.0, Q.sj3_pair_mass_min - 80.02563) / 0.702582   # +0.1%  sj3_pair_mass_min > 80.03
        - 0.0007543314 * max(0.0, 0.1464694 - Q.dr_23) / 0.0394748   # -0.1%  dr_23 < 0.1465
        + 0.0006992398 * max(0.0, 0.007507913 - Q.sum_z_dr2_top15) / 0.0003727018   # +0.1%  sum_z_dr2_top15 < 0.007508
        - 0.0006322421 * max(0.0, Q.sj3_dr_min - 0.1991803) / 0.05760129   # -0.1%  sj3_dr_min > 0.1992
        - 0.0006229767 * max(0.0, Q.sj3_pair_mass_min - 59.49644) / 2.657693   # -0.1%  sj3_pair_mass_min > 59.5
        - 0.0005947449 * max(0.0, Q.lund2_lndelta - -0.737181) / 0.0784362   # -0.1%  lund2_lndelta > -0.7372
        + 0.0005907511 * max(0.0, Q.M3 - 0.04863496) / 0.002771826   # +0.1%  M3 > 0.04863
        + 0.0005824578 * max(0.0, 0.1076451 - Q.tau2) * max(0.0, 0.7045747 - Q.N3_b05) / 0.002445281   # +0.1%  tau2 < 0.1076 and N3_b05 < 0.7046
        - 0.0005138694 * max(0.0, 0.3886647 - Q.sj3_mass3) * max(0.0, Q.lund2_lnz - -5.443387) / 0.06593447   # -0.1%  sj3_mass3 < 0.3887 and lund2_lnz > -5.443
        - 0.0005123011 * max(0.0, 1.408071 - Q.max_pair_mass) / 0.04402196   # -0.1%  max_pair_mass < 1.408
        - 0.0004835 * max(0.0, Q.sd_mass - 42.31629) * max(0.0, 0.5339053 - Q.pt1_over_pt0) / 3.776029   # -0.0%  sd_mass > 42.32 and pt1_over_pt0 < 0.5339
        - 0.000481686 * max(0.0, 0.1985369 - Q.pt_balance01) / 0.006970235   # -0.0%  pt_balance01 < 0.1985
        - 0.0004390093 * max(0.0, 1.102602 - Q.D2_b05) / 0.006783992   # -0.0%  D2_b05 < 1.103
        + 0.0004228831 * max(0.0, Q.m01 - 3.294888) * max(0.0, 20.0 - Q.n_real_top20) / 8.186128   # +0.0%  m01 > 3.295 and n_real_top20 < 20
        - 0.000402168 * max(0.0, Q.m01 - 4.577187) * max(0.0, 0.1975394 - Q.dr_1) / 0.4305706   # -0.0%  m01 > 4.577 and dr_1 < 0.1975
        - 0.0003758965 * max(0.0, Q.C3_b05 - 0.1094329) * max(0.0, 0.07720947 - Q.eta_40) / 0.001774585   # -0.0%  C3_b05 > 0.1094 and eta_40 < 0.07721
        - 0.0002533647 * max(0.0, 19.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.01724773 - Q.M3_b2) / 0.04348534   # -0.0%  n_dr_0p2_0p4 < 19 and M3_b2 < 0.01725
        - 0.0002265749 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # -0.0%  n_pairs_kt_above_3 < 34
        + 0.0002242544 * max(0.0, 0.3886647 - Q.sj3_mass3) * max(0.0, Q.lund2_lnkt - 3.917562) / 0.002427378   # +0.0%  sj3_mass3 < 0.3887 and lund2_lnkt > 3.918
        + 0.0002105142 * max(0.0, Q.mass - 117.4867) * max(0.0, Q.mratio_max_012 - 0.7696297) / 0.3538346   # +0.0%  mass > 117.5 and mratio_max_012 > 0.7696
        + 0.000210412 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.phi_18 - 0.04797363) / 5.873287   # +0.0%  n_pairs_kt_above_1 < 366 and phi_18 > 0.04797
        - 0.0002055259 * max(0.0, Q.mass_top15 - 133.1648) / 1.058612   # -0.0%  mass_top15 > 133.2
        + 0.0001757446 * max(0.0, 0.1076451 - Q.tau2) / 0.0393622   # +0.0%  tau2 < 0.1076
        + 0.00015547 * max(0.0, Q.mass_top40 - 126.8853) * max(0.0, 0.009257989 - Q.M3_b2) / 0.02386565   # +0.0%  mass_top40 > 126.9 and M3_b2 < 0.009258
        - 0.0001418979 * max(0.0, Q.dr02 - 0.1620436) / 0.07871781   # -0.0%  dr02 > 0.162
        - 0.0001293518 * max(0.0, 175.9333 - Q.sd_mass) * max(0.0, Q.min_pair_mass - 10.59762) / 15.33759   # -0.0%  sd_mass < 175.9 and min_pair_mass > 10.6
        - 0.0001210337 * max(0.0, Q.sj4_pair_mass_max - 128.0079) / 1.174806   # -0.0%  sj4_pair_mass_max > 128
        - 0.000110649 * max(0.0, Q.sj3_dr_min - 0.462488) / 0.00364591   # -0.0%  sj3_dr_min > 0.4625
        + 9.043097e-05 * max(0.0, 123.2919 - Q.sd_mass) * max(0.0, Q.dr_10 - 0.5063302) / 0.2811536   # +0.0%  sd_mass < 123.3 and dr_10 > 0.5063
        - 6.789757e-05 * max(0.0, 1.281478 - Q.D2_b05) / 0.02358394   # -0.0%  D2_b05 < 1.281
        - 5.281813e-05 * max(0.0, 1.514826 - Q.min_pair_mass) / 0.4657734   # -0.0%  min_pair_mass < 1.515
        + 4.92647e-05 * max(0.0, Q.lne_1 - 5.29456) / 0.0281941   # +0.0%  lne_1 > 5.295
    )
    return z


def neuron_55(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.958066e-06
    )
    return z


def neuron_56(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.805158e-06
    )
    return z


def neuron_57(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.001688453
    )
    return z


def neuron_58(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.775588e-05
    )
    return z


def neuron_59(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.876595e-06
    )
    return z


def neuron_60(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.460409e-06
    )
    return z


def neuron_61(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.691285e-06
    )
    return z


def neuron_62(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.222401e-06
    )
    return z


def neuron_63(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.730003e-05
    )
    return z


def neuron_64(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.583109e-05
    )
    return z


def neuron_65(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.799774e-05
    )
    return z


def neuron_66(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.819647e-05
    )
    return z


def neuron_67(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.789744e-06
    )
    return z


def neuron_68(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.798125e-07
    )
    return z


def neuron_69(Q):
    # scale S = 3.69; each line: share * term / its average size
    z = 3.689929 * (-0.02099726
        + 0.08388552 * max(0.0, 94.55722 - Q.sd_mass) / 17.47876   # +8.4%  sd_mass < 94.56
        - 0.08167496 * max(0.0, 78.4753 - Q.sd_mass) / 11.39114   # -8.2%  sd_mass < 78.48
        - 0.07393188 * max(0.0, Q.mass - 56.49019) / 59.27462   # -7.4%  mass > 56.49
        + 0.0663559 * max(0.0, Q.sd_mass - 42.31629) / 58.72614   # +6.6%  sd_mass > 42.32
        + 0.06613363 * max(0.0, Q.mass - 95.14961) / 26.15323   # +6.6%  mass > 95.15
        - 0.05902408 * max(0.0, 115.7091 - Q.sd_mass) / 28.95052   # -5.9%  sd_mass < 115.7
        + 0.05814696 * Q.pair_mean_lndelta / 2.210599   # +5.8%  pair_mean_lndelta
        + 0.05052226 * max(0.0, 175.9333 - Q.sd_mass) / 80.62387   # +5.1%  sd_mass < 175.9
        - 0.04724224 * max(0.0, Q.mass_top40 - 92.68149) / 23.77466   # -4.7%  mass_top40 > 92.68
        + 0.03988405 * max(0.0, Q.mass - 95.14961) * max(0.0, 0.1049877 - Q.C2_b2) / 0.8639832   # +4.0%  mass > 95.15 and C2_b2 < 0.105
        + 0.03762044 * max(0.0, 88.81751 - Q.sd_mass) / 15.02663   # +3.8%  sd_mass < 88.82
        - 0.02690248 * max(0.0, 95.20959 - Q.mass_top30) / 8.191707   # -2.7%  mass_top30 < 95.21
        + 0.02542894 * max(0.0, 97.52633 - Q.sj3_pair_mass_max) / 14.66667   # +2.5%  sj3_pair_mass_max < 97.53
        + 0.02517075 * max(0.0, 106.7501 - Q.sd_mass) / 23.58889   # +2.5%  sd_mass < 106.8
        + 0.02208823 * max(0.0, Q.mass_top40 - 122.7145) / 8.484825   # +2.2%  mass_top40 > 122.7
        - 0.02192629 * max(0.0, 76.91486 - Q.sj3_pair_mass_max) / 5.293372   # -2.2%  sj3_pair_mass_max < 76.91
        - 0.01856437 * max(0.0, Q.mass - 79.47361) / 38.31536   # -1.9%  mass > 79.47
        - 0.01788224 * max(0.0, Q.mass - 114.0172) * max(0.0, 0.1204509 - Q.C2_b2) / 0.5063536   # -1.8%  mass > 114 and C2_b2 < 0.1205
        - 0.01413417 * max(0.0, Q.lnerel_0 - -1.362996) / 0.1441523   # -1.4%  lnerel_0 > -1.363
        - 0.01266639 * max(0.0, 33.62116 - Q.sj2_mass1) / 6.452718   # -1.3%  sj2_mass1 < 33.62
        + 0.01093641 * max(0.0, 83.81888 - Q.mass_top30) / 4.500625   # +1.1%  mass_top30 < 83.82
        + 0.00986203 * max(0.0, 0.1193588 - Q.tau2) / 0.04844327   # +1.0%  tau2 < 0.1194
        - 0.008903229 * max(0.0, Q.sj3_pair_mass_max - 128.6605) / 2.418855   # -0.9%  sj3_pair_mass_max > 128.7
        + 0.008121216 * max(0.0, 3.634108 - Q.lund_max_lnkt) / 0.1987071   # +0.8%  lund_max_lnkt < 3.634
        - 0.007814816 * max(0.0, Q.mass - 114.0172) / 14.73611   # -0.8%  mass > 114
        + 0.007593567 * max(0.0, Q.sj3_pairmin_over_m - 0.2878167) / 0.06686755   # +0.8%  sj3_pairmin_over_m > 0.2878
        - 0.007094136 * max(0.0, 175.9333 - Q.sd_mass) * max(0.0, Q.n_pt_above_5 - 14.0) / 529.0534   # -0.7%  sd_mass < 175.9 and n_pt_above_5 > 14
        + 0.006717598 * max(0.0, 8.0 - Q.n_lund) / 0.3988733   # +0.7%  n_lund < 8
        - 0.006605223 * Q.pt1_dr01 / 10.94415   # -0.7%  pt1_dr01
        + 0.005649678 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # +0.6%  sj3_pair_mass_max < 73.25
        + 0.005222183 * max(0.0, Q.mass_top40 - 101.351) * max(0.0, 0.3141459 - Q.dr12) / 2.957168   # +0.5%  mass_top40 > 101.4 and dr12 < 0.3141
        + 0.004912911 * max(0.0, 88.0 - Q.n_pairs_kt_above_3) / 31.77884   # +0.5%  n_pairs_kt_above_3 < 88
        + 0.004902264 * max(0.0, Q.mass_top50 - 56.41079) * max(0.0, 0.3886647 - Q.sj3_mass3) / 1.584122   # +0.5%  mass_top50 > 56.41 and sj3_mass3 < 0.3887
        + 0.003974007 * max(0.0, Q.lnerel_0 - -1.362996) * max(0.0, 0.2273903 - Q.z_2nd) / 0.0152671   # +0.4%  lnerel_0 > -1.363 and z_2nd < 0.2274
        + 0.003878068 * max(0.0, 175.9333 - Q.sd_mass) * max(0.0, 0.6591684 - Q.pt1_over_pt0) / 10.67619   # +0.4%  sd_mass < 175.9 and pt1_over_pt0 < 0.6592
        - 0.003330432 * max(0.0, Q.mass - 95.14961) * max(0.0, Q.lnptrel_0 - -1.266303) / 1.9484   # -0.3%  mass > 95.15 and lnptrel_0 > -1.266
        - 0.002800244 * max(0.0, Q.mass_top50 - 56.41079) / 57.67108   # -0.3%  mass_top50 > 56.41
        + 0.002668848 * max(0.0, Q.mass_top40 - 122.7145) * max(0.0, 0.2075244 - Q.sj3_z3) / 0.6181629   # +0.3%  mass_top40 > 122.7 and sj3_z3 < 0.2075
        - 0.002476335 * max(0.0, 8.0 - Q.n_lund) * max(0.0, 0.000125186 - Q.e3_b2) / 3.067844e-05   # -0.2%  n_lund < 8 and e3_b2 < 0.0001252
        - 0.002229824 * max(0.0, 0.2077175 - Q.tau21) / 0.008580176   # -0.2%  tau21 < 0.2077
        - 0.002135999 * max(0.0, Q.mass - 131.3917) / 8.323408   # -0.2%  mass > 131.4
        - 0.002011322 * max(0.0, Q.mass_top30 - 146.3096) / 2.526929   # -0.2%  mass_top30 > 146.3
        + 0.001948643 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # +0.2%  n_pairs_kt_above_1 < 58
        + 0.001793704 * max(0.0, 0.1648004 - Q.dr_9) / 0.04144238   # +0.2%  dr_9 < 0.1648
        - 0.001691326 * max(0.0, 0.06074238 - Q.tau1) / 0.001113326   # -0.2%  tau1 < 0.06074
        + 0.001608552 * max(0.0, Q.sj3_pairmin_over_m - 0.2878167) * max(0.0, 0.08614914 - Q.dr_32) / 0.001569223   # +0.2%  sj3_pairmin_over_m > 0.2878 and dr_32 < 0.08615
        + 0.001545863 * max(0.0, 0.2077175 - Q.tau21) * max(0.0, 0.5381046 - Q.sj3_dr_max) / 0.0008380664   # +0.2%  tau21 < 0.2077 and sj3_dr_max < 0.5381
        + 0.001486452 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) * max(0.0, 4.696862e-06 - Q.e3_b2) / 4.856521e-06   # +0.1%  n_pairs_kt_above_1 < 58 and e3_b2 < 4.697e-06
        + 0.0014514 * max(0.0, Q.mass - 182.8592) / 1.688248   # +0.1%  mass > 182.9
        + 0.001431633 * max(0.0, 0.0366743 - Q.tau3) / 0.00487328   # +0.1%  tau3 < 0.03667
        - 0.001185259 * max(0.0, Q.mass_top40 - 101.351) / 18.14537   # -0.1%  mass_top40 > 101.4
        - 0.001126119 * max(0.0, 1.851735 - Q.sj2_mass2) * max(0.0, 21.0 - Q.n_pt_above_10) / 0.7208369   # -0.1%  sj2_mass2 < 1.852 and n_pt_above_10 < 21
        - 0.001066447 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # -0.1%  sj2_mass2 < 1.852
        + 0.001004145 * max(0.0, Q.m01 - 17.3274) * max(0.0, -0.3948254 - Q.lund1_lndelta) / 1.411322   # +0.1%  m01 > 17.33 and lund1_lndelta < -0.3948
        + 0.0009559635 * max(0.0, Q.lnerel_0 - -1.362996) * max(0.0, Q.dr_0 - 0.1756108) / 0.0008107367   # +0.1%  lnerel_0 > -1.363 and dr_0 > 0.1756
        - 0.0009445216 * max(0.0, 78.4753 - Q.sd_mass) * max(0.0, Q.M3 - 0.04863496) / 0.08285467   # -0.1%  sd_mass < 78.48 and M3 > 0.04863
        - 0.000943857 * max(0.0, Q.m01 - 17.3274) * max(0.0, 0.1869778 - Q.dr_16) / 0.3123091   # -0.1%  m01 > 17.33 and dr_16 < 0.187
        + 0.000915862 * max(0.0, Q.C2 - 0.2035961) / 0.009390341   # +0.1%  C2 > 0.2036
        + 0.0008670617 * max(0.0, 33.62116 - Q.sj2_mass1) * max(0.0, 2.0 - Q.n_dr_0p05_0p1) / 2.570601   # +0.1%  sj2_mass1 < 33.62 and n_dr_0p05_0p1 < 2
        + 0.0008248957 * max(0.0, 88.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.1605483 - Q.dr_40) / 4.518911   # +0.1%  n_pairs_kt_above_3 < 88 and dr_40 < 0.1605
        - 0.0008121785 * max(0.0, Q.mass - 114.0172) * max(0.0, 0.8673543 - Q.tau32_b2) / 6.248034   # -0.1%  mass > 114 and tau32_b2 < 0.8674
        - 0.0008081919 * max(0.0, Q.m01 - 17.3274) * max(0.0, 0.002489417 - Q.dr_17) / 0.002119909   # -0.1%  m01 > 17.33 and dr_17 < 0.002489
        + 0.0007603568 * max(0.0, Q.mass - 114.0172) * max(0.0, 1.824785 - Q.sj3_mass2) / 0.4471415   # +0.1%  mass > 114 and sj3_mass2 < 1.825
        - 0.0007236286 * max(0.0, 94.55722 - Q.sd_mass) * max(0.0, 0.00278761 - Q.C3_b2) / 0.02279717   # -0.1%  sd_mass < 94.56 and C3_b2 < 0.002788
        + 0.0007176422 * max(0.0, 83.81888 - Q.mass_top30) * max(0.0, 0.4868628 - Q.pt1_over_pt0) / 0.2957715   # +0.1%  mass_top30 < 83.82 and pt1_over_pt0 < 0.4869
        + 0.0007047629 * max(0.0, Q.mass - 114.0172) * max(0.0, 0.1361601 - Q.pt_balance01) / 0.02915088   # +0.1%  mass > 114 and pt_balance01 < 0.1362
        + 0.0006643942 * max(0.0, Q.m01 - 17.3274) * max(0.0, 0.09069824 - Q.eta_1) / 0.940547   # +0.1%  m01 > 17.33 and eta_1 < 0.0907
        - 0.0006472719 * max(0.0, 175.9333 - Q.sd_mass) * max(0.0, Q.min_pair_mass - 10.59762) / 15.33759   # -0.1%  sd_mass < 175.9 and min_pair_mass > 10.6
        - 0.0004730238 * max(0.0, 8.0 - Q.n_lund) * max(0.0, Q.pt1_over_pt0 - 0.380649) / 0.02596386   # -0.0%  n_lund < 8 and pt1_over_pt0 > 0.3806
        + 0.0004454839 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # +0.0%  n_pairs_kt_above_3 < 34
        + 0.0003889159 * max(0.0, Q.mass - 79.47361) * max(0.0, 0.04944303 - Q.M2_b2) / 0.6063826   # +0.0%  mass > 79.47 and M2_b2 < 0.04944
        - 0.0003240546 * max(0.0, Q.n_dr_0_0p05 - 10.0) / 0.3092667   # -0.0%  n_dr_0_0p05 > 10
        - 0.0001126378 * max(0.0, Q.m01 - 17.3274) / 6.734154   # -0.0%  m01 > 17.33
        - 0.0001084293 * max(0.0, Q.lnerel_0 - -1.362996) * max(0.0, Q.jet_abs_eta - 1.098425) / 0.01155688   # -0.0%  lnerel_0 > -1.363 and jet_abs_eta > 1.098
        - 9.827331e-05 * max(0.0, Q.mass - 95.14961) * max(0.0, Q.mratio_max_012 - 0.8680029) / 0.06702601   # -0.0%  mass > 95.15 and mratio_max_012 > 0.868
        + 9.382353e-05 * max(0.0, Q.mass - 182.8592) * max(0.0, 0.4868628 - Q.pt1_over_pt0) / 0.08297038   # +0.0%  mass > 182.9 and pt1_over_pt0 < 0.4869
        - 9.379935e-05 * max(0.0, Q.sd_mass - 42.31629) * max(0.0, Q.e3_b2 - 0.0002536827) / 0.01264561   # -0.0%  sd_mass > 42.32 and e3_b2 > 0.0002537
        - 9.10241e-05 * max(0.0, 0.1193588 - Q.tau2) * max(0.0, 3.143912 - Q.pair_max_lnkt) / 0.02960322   # -0.0%  tau2 < 0.1194 and pair_max_lnkt < 3.144
        + 8.825745e-05 * max(0.0, 0.0366743 - Q.tau3) * max(0.0, Q.mratio_min_012 - 0.1991254) / 0.0002096794   # +0.0%  tau3 < 0.03667 and mratio_min_012 > 0.1991
        - 1.00987e-05 * max(0.0, Q.m01 - 17.3274) * max(0.0, Q.dr_55 - 0.1893821) / 0.08946805   # -0.0%  m01 > 17.33 and dr_55 > 0.1894
        + 8.416931e-06 * max(0.0, 0.2077175 - Q.tau21) * max(0.0, Q.n_dr_0p05_0p1 - 16.0) / 8.885844e-05   # +0.0%  tau21 < 0.2077 and n_dr_0p05_0p1 > 16
        - 5.803934e-06 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.n_dr_0p05_0p1 - 16.0) / 0.01324333   # -0.0%  n_pairs_kt_above_1 < 58 and n_dr_0p05_0p1 > 16
        + 1.728621e-06 * max(0.0, Q.mass_top30 - 146.3096) * max(0.0, Q.e4_b2 - 6.5e-11) / 3.962494e-06   # +0.0%  mass_top30 > 146.3 and e4_b2 > 6.5e-11
    )
    return z


def neuron_70(Q):
    # scale S = 4.474; each line: share * term / its average size
    z = 4.47376 * (-0.04051933
        + 0.1274795 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # +12.7%  sd_mass > 78.48
        - 0.08358616 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # -8.4%  sd_mass > 94.56
        - 0.08080596 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -8.1%  mass < 164.4
        + 0.06720862 * Q.z_top3_slots / 0.4630907   # +6.7%  z_top3_slots
        - 0.06016273 * max(0.0, Q.ecf_g31 - 0.001656914) / 0.004961522   # -6.0%  ecf_g31 > 0.001657
        + 0.05015006 * max(0.0, 109.8208 - Q.sj3_pair_mass_max) / 23.01361   # +5.0%  sj3_pair_mass_max < 109.8
        + 0.04642004 * max(0.0, 0.654849 - Q.z_top3_slots) / 0.2071834   # +4.6%  z_top3_slots < 0.6548
        + 0.03676901 * max(0.0, 131.3917 - Q.mass) / 24.784   # +3.7%  mass < 131.4
        + 0.02993765 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.0420789 - Q.sum_z_dr2_top2) / 0.3038921   # +3.0%  n_pairs_kt_above_3 < 47 and sum_z_dr2_top2 < 0.04208
        - 0.02436204 * max(0.0, 4.363663e-05 - Q.e4_b05) / 1.785748e-05   # -2.4%  e4_b05 < 4.364e-05
        - 0.02379053 * max(0.0, 117.4867 - Q.mass) / 15.64803   # -2.4%  mass < 117.5
        - 0.01947356 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # -1.9%  sd_mass > 106.8
        - 0.01923949 * max(0.0, 0.654849 - Q.z_top3_slots) * max(0.0, 0.221369 - Q.C2_b2) / 0.0312683   # -1.9%  z_top3_slots < 0.6548 and C2_b2 < 0.2214
        + 0.01834573 * max(0.0, 0.2054406 - Q.dr_max_012) / 0.07212785   # +1.8%  dr_max_012 < 0.2054
        - 0.01816114 * max(0.0, 76.91486 - Q.sj3_pair_mass_max) / 5.293372   # -1.8%  sj3_pair_mass_max < 76.91
        - 0.01809472 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) / 9.886423   # -1.8%  n_pairs_kt_above_3 < 47
        + 0.01326138 * max(0.0, 117.4867 - Q.mass) * max(0.0, 0.2273903 - Q.z_2nd) / 1.301829   # +1.3%  mass < 117.5 and z_2nd < 0.2274
        + 0.01252716 * max(0.0, Q.sd_mass - 135.3031) / 5.717996   # +1.3%  sd_mass > 135.3
        + 0.01232688 * max(0.0, 0.654849 - Q.z_top3_slots) * max(0.0, Q.tau43 - 0.568406) / 0.05004789   # +1.2%  z_top3_slots < 0.6548 and tau43 > 0.5684
        - 0.01208641 * max(0.0, 95.14961 - Q.mass) / 6.371707   # -1.2%  mass < 95.15
        + 0.01199742 * max(0.0, Q.mass_top50 - 99.54528) / 21.60123   # +1.2%  mass_top50 > 99.55
        + 0.01198205 * max(0.0, 161.1264 - Q.mass_top50) / 50.68933   # +1.2%  mass_top50 < 161.1
        - 0.01196078 * max(0.0, Q.mass_top50 - 129.5874) / 7.866327   # -1.2%  mass_top50 > 129.6
        + 0.01043165 * max(0.0, Q.mass - 71.96396) / 44.90212   # +1.0%  mass > 71.96
        - 0.01034393 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, Q.lne_12 - 1.787846) / 39.09795   # -1.0%  mass_top50 < 161.1 and lne_12 > 1.788
        - 0.01027932 * max(0.0, 0.654849 - Q.z_top3_slots) * max(0.0, 0.4410123 - Q.dr_max_012) / 0.04563847   # -1.0%  z_top3_slots < 0.6548 and dr_max_012 < 0.441
        - 0.009998468 * max(0.0, 146.0 - Q.n_pairs_kt_above_1) / 20.10573   # -1.0%  n_pairs_kt_above_1 < 146
        - 0.008654596 * max(0.0, Q.mass - 71.96396) * max(0.0, 23.0 - Q.n_pairs_kt_above_10) / 586.4939   # -0.9%  mass > 71.96 and n_pairs_kt_above_10 < 23
        - 0.008195858 * max(0.0, Q.pair_mean_lnm2 - 2.09826) / 0.7077607   # -0.8%  pair_mean_lnm2 > 2.098
        + 0.007759924 * max(0.0, Q.e3_b05 - 0.01046349) / 0.0007795921   # +0.8%  e3_b05 > 0.01046
        + 0.007216612 * max(0.0, 146.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.sj3_pairmin_over_m - 0.1636952) / 2.178039   # +0.7%  n_pairs_kt_above_1 < 146 and sj3_pairmin_over_m > 0.1637
        - 0.005993264 * max(0.0, 0.3667049 - Q.N2_b05) / 0.008239872   # -0.6%  N2_b05 < 0.3667
        - 0.005970208 * max(0.0, 0.1411752 - Q.C2) / 0.02255942   # -0.6%  C2 < 0.1412
        - 0.005957479 * max(0.0, Q.lund_max_lndelta - -0.9602344) / 0.122395   # -0.6%  lund_max_lndelta > -0.9602
        + 0.005909695 * max(0.0, 0.2234678 - Q.C2_b05) / 0.00926781   # +0.6%  C2_b05 < 0.2235
        + 0.005686756 * max(0.0, Q.mass - 71.96396) * max(0.0, 623.2168 - Q.sum_pt_top40) / 2091.232   # +0.6%  mass > 71.96 and sum_pt_top40 < 623.2
        + 0.005626225 * max(0.0, Q.ecf_g31 - 0.001656914) * max(0.0, Q.tau54 - 0.7835935) / 0.0003781799   # +0.6%  ecf_g31 > 0.001657 and tau54 > 0.7836
        - 0.004700391 * max(0.0, Q.m012 - 49.92073) / 2.537789   # -0.5%  m012 > 49.92
        - 0.004666184 * max(0.0, Q.e4_b05 - 7.503722e-05) / 1.004857e-05   # -0.5%  e4_b05 > 7.504e-05
        + 0.004120367 * max(0.0, Q.max_pair_mass - 55.24488) / 0.6579942   # +0.4%  max_pair_mass > 55.24
        - 0.003775546 * max(0.0, Q.mass_top50 - 99.54528) * max(0.0, 91.2852 - Q.sj2_mass1) / 746.4599   # -0.4%  mass_top50 > 99.55 and sj2_mass1 < 91.29
        + 0.003339952 * max(0.0, 83.57316 - Q.mass_top40) / 3.715787   # +0.3%  mass_top40 < 83.57
        + 0.00323997 * max(0.0, 0.4276838 - Q.tau32_b2) * max(0.0, 0.1686866 - Q.dr_32) / 0.00356876   # +0.3%  tau32_b2 < 0.4277 and dr_32 < 0.1687
        - 0.003238927 * max(0.0, Q.lund_max_lndelta - -1.303913) / 0.3311482   # -0.3%  lund_max_lndelta > -1.304
        - 0.003094901 * max(0.0, 0.00761379 - Q.M3_b2) / 0.0008114698   # -0.3%  M3_b2 < 0.007614
        + 0.003037999 * max(0.0, Q.pair_mean_lnkt - 0.6690549) / 0.140271   # +0.3%  pair_mean_lnkt > 0.6691
        - 0.003000157 * max(0.0, 0.007457486 - Q.sum_z_dr2_top5) / 0.001077856   # -0.3%  sum_z_dr2_top5 < 0.007457
        - 0.002695887 * max(0.0, 646.5833 - Q.jet_pt) / 60.98317   # -0.3%  jet_pt < 646.6
        + 0.002542295 * max(0.0, Q.e3_b05 - 0.01046349) * max(0.0, 0.68373 - Q.sj3_z1) / 0.0001313387   # +0.3%  e3_b05 > 0.01046 and sj3_z1 < 0.6837
        + 0.002328722 * max(0.0, 0.4276838 - Q.tau32_b2) / 0.04662994   # +0.2%  tau32_b2 < 0.4277
        + 0.002230979 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +0.2%  sd_mass > 154.6
        + 0.002170473 * max(0.0, Q.m012 - 49.92073) * max(0.0, 0.3613291 - Q.dr12) / 0.4882546   # +0.2%  m012 > 49.92 and dr12 < 0.3613
        + 0.001994307 * max(0.0, Q.m012 - 49.92073) * max(0.0, 0.3713204 - Q.sj3_pairmin_over_m) / 0.3645446   # +0.2%  m012 > 49.92 and sj3_pairmin_over_m < 0.3713
        + 0.001885684 * max(0.0, Q.pair_mean_lndelta - -2.049856) * max(0.0, 0.02915846 - Q.dr_22) / 0.0006518068   # +0.2%  pair_mean_lndelta > -2.05 and dr_22 < 0.02916
        - 0.001799257 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # -0.2%  mass_top10 > 103.5
        - 0.001724809 * max(0.0, Q.pair_mean_lndelta - -1.523797) / 0.02402574   # -0.2%  pair_mean_lndelta > -1.524
        - 0.001607753 * max(0.0, Q.mass - 182.8592) / 1.688248   # -0.2%  mass > 182.9
        + 0.001605345 * max(0.0, Q.sj3_pair_mass_max - 142.9952) / 1.384637   # +0.2%  sj3_pair_mass_max > 143
        + 0.001601474 * max(0.0, 95.14961 - Q.mass) * max(0.0, 4.621465 - Q.sj3_mass1) / 1.733417   # +0.2%  mass < 95.15 and sj3_mass1 < 4.621
        - 0.001595495 * max(0.0, 0.01467699 - Q.tau5) / 0.0006654988   # -0.2%  tau5 < 0.01468
        + 0.001534792 * max(0.0, 0.654849 - Q.z_top3_slots) * max(0.0, Q.lund3_lnkt - 3.058345) / 0.05245815   # +0.2%  z_top3_slots < 0.6548 and lund3_lnkt > 3.058
        - 0.001385212 * max(0.0, Q.pair_mean_lndelta - -2.049856) / 0.1589658   # -0.1%  pair_mean_lndelta > -2.05
        + 0.001305013 * max(0.0, 1.824785 - Q.sj3_mass2) / 0.06171879   # +0.1%  sj3_mass2 < 1.825
        - 0.00120544 * max(0.0, 0.6627397 - Q.sj3_pairmax_over_m) / 0.001074687   # -0.1%  sj3_pairmax_over_m < 0.6627
        - 0.001145979 * max(0.0, 4.363663e-05 - Q.e4_b05) * max(0.0, Q.dr_1 - 0.05473942) / 9.023853e-07   # -0.1%  e4_b05 < 4.364e-05 and dr_1 > 0.05474
        - 0.001138877 * max(0.0, 0.4276838 - Q.tau32_b2) * max(0.0, Q.sum_z_dr2_top2 - 0.0420789) / 0.0004079403   # -0.1%  tau32_b2 < 0.4277 and sum_z_dr2_top2 > 0.04208
        - 0.001131216 * max(0.0, Q.mass_top50 - 99.54528) * max(0.0, 3.928781 - Q.sj2_mass2) / 2.222121   # -0.1%  mass_top50 > 99.55 and sj2_mass2 < 3.929
        + 0.001115852 * max(0.0, Q.m012 - 49.92073) * max(0.0, Q.dr_2 - 0.0205536) / 0.6396363   # +0.1%  m012 > 49.92 and dr_2 > 0.02055
        - 0.001007135 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.z_3rd - 0.07296058) / 0.4659698   # -0.1%  mass < 117.5 and z_3rd > 0.07296
        + 0.001005069 * max(0.0, Q.mass_over_sum_pt - 0.2478639) / 0.00727487   # +0.1%  mass_over_sum_pt > 0.2479
        - 0.0007919559 * max(0.0, Q.mass - 182.8592) * max(0.0, 6.45188 - Q.log_sum_pt) / 0.04131746   # -0.1%  mass > 182.9 and log_sum_pt < 6.452
        + 0.0007565307 * max(0.0, Q.ecf_g31 - 0.001656914) * max(0.0, -0.1473389 - Q.eta_1) / 6.206641e-05   # +0.1%  ecf_g31 > 0.001657 and eta_1 < -0.1473
        - 0.0007152803 * max(0.0, Q.sd_mass - 154.5947) * max(0.0, 0.2652027 - Q.sd_zg) / 0.1412539   # -0.1%  sd_mass > 154.6 and sd_zg < 0.2652
        - 0.0007022811 * max(0.0, 0.4276838 - Q.tau32_b2) * max(0.0, Q.dr_10 - 0.3989771) / 0.001131288   # -0.1%  tau32_b2 < 0.4277 and dr_10 > 0.399
        + 0.0005649956 * max(0.0, Q.mass_top50 - 129.5874) * max(0.0, 288.0 - Q.n_pairs_kt_above_1) / 106.5669   # +0.1%  mass_top50 > 129.6 and n_pairs_kt_above_1 < 288
        - 0.0005584646 * max(0.0, 146.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.sum_e - 1247.266) / 951.9318   # -0.1%  n_pairs_kt_above_1 < 146 and sum_e > 1247
        - 0.0005366648 * max(0.0, Q.max_pair_mass - 55.24488) * max(0.0, Q.eta_23 - -0.3259277) / 0.2245444   # -0.1%  max_pair_mass > 55.24 and eta_23 > -0.3259
        - 0.0005033189 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.sj3_mass1 - 11.10125) / 372.8029   # -0.1%  mass < 164.4 and sj3_mass1 > 11.1
        + 0.0004657066 * max(0.0, Q.ecf_g31 - 0.001656914) * max(0.0, 0.01827585 - Q.dr_16) / 7.509115e-07   # +0.0%  ecf_g31 > 0.001657 and dr_16 < 0.01828
        + 0.0003610069 * max(0.0, Q.lund_max_lndelta - -0.9602344) * max(0.0, Q.dr_18 - 0.09105897) / 0.02511309   # +0.0%  lund_max_lndelta > -0.9602 and dr_18 > 0.09106
        - 0.0003389571 * max(0.0, Q.pair_mean_lndelta - -2.049856) * max(0.0, Q.dr_55 - 0.0) / 0.01603724   # -0.0%  pair_mean_lndelta > -2.05 and dr_55 > 0
        - 0.0002820276 * max(0.0, Q.sd_mass - 106.7501) * max(0.0, Q.phi_17 - 0.328125) / 0.1479392   # -0.0%  sd_mass > 106.8 and phi_17 > 0.3281
        - 0.0002727929 * max(0.0, 0.09835839 - Q.M2_b05) / 0.002220231   # -0.0%  M2_b05 < 0.09836
        - 0.0002639437 * max(0.0, Q.pt1_dr01 - 29.53862) * max(0.0, 0.1959828 - Q.dr_5) / 0.04939225   # -0.0%  pt1_dr01 > 29.54 and dr_5 < 0.196
        - 0.0002376167 * max(0.0, Q.ecf_g31 - 0.001656914) * max(0.0, 0.0184643 - Q.sj3_z3) / 1.419749e-06   # -0.0%  ecf_g31 > 0.001657 and sj3_z3 < 0.01846
        + 0.0001241635 * max(0.0, 0.00761379 - Q.M3_b2) * max(0.0, Q.eta_57 - 0.07013245) / 1.519827e-05   # +0.0%  M3_b2 < 0.007614 and eta_57 > 0.07013
        + 0.0001191183 * max(0.0, Q.max_pair_mass - 55.24488) * max(0.0, Q.jet_e - 551.9263) / 233.2114   # +0.0%  max_pair_mass > 55.24 and jet_e > 551.9
        - 0.000103588 * max(0.0, Q.mass_top50 - 129.5874) * max(0.0, 0.02915846 - Q.dr_22) / 0.004817342   # -0.0%  mass_top50 > 129.6 and dr_22 < 0.02916
        + 6.045642e-05 * max(0.0, Q.pt1_dr01 - 29.53862) / 1.088839   # +0.0%  pt1_dr01 > 29.54
        - 6.040098e-05 * max(0.0, Q.sd_mass - 154.5947) * max(0.0, Q.C3_b05 - 0.1643977) / 0.006666327   # -0.0%  sd_mass > 154.6 and C3_b05 > 0.1644
        - 3.232003e-05 * max(0.0, Q.mass - 71.96396) * max(0.0, Q.C2_b2 - 0.02858957) / 3.220159   # -0.0%  mass > 71.96 and C2_b2 > 0.02859
        - 2.370881e-05 * max(0.0, Q.sj3_pair_mass_min - 53.29621) / 3.744152   # -0.0%  sj3_pair_mass_min > 53.3
        + 1.867199e-07 * max(0.0, Q.mass - 182.8592) * max(0.0, Q.eta_23 - 0.3276367) / 0.02563513   # +0.0%  mass > 182.9 and eta_23 > 0.3276
    )
    return z


def neuron_71(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.181428e-05
    )
    return z


def neuron_72(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.547756e-05
    )
    return z


def neuron_73(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.301758e-05
    )
    return z


def neuron_74(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.363609e-05
    )
    return z


def neuron_75(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.582183e-06
    )
    return z


def neuron_76(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.739781e-07
    )
    return z


def neuron_77(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.005252e-05
    )
    return z


def neuron_78(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.790103e-06
    )
    return z


def neuron_79(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.50548e-07
    )
    return z


def neuron_80(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.80459e-05
    )
    return z


def neuron_81(Q):
    # scale S = 6.429; each line: share * term / its average size
    z = 6.429183 * (-0.06555793
        - 0.1986369 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # -19.9%  sd_mass > 78.48
        + 0.09476652 * Q.N2_b05 / 0.4382511   # +9.5%  N2_b05
        + 0.08819588 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # +8.8%  sd_mass > 94.56
        + 0.07478644 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # +7.5%  sd_mass > 88.82
        + 0.07217323 * max(0.0, Q.sd_mass - 62.03827) / 42.41619   # +7.2%  sd_mass > 62.04
        - 0.04718776 * max(0.0, 149.0507 - Q.mass) / 39.08704   # -4.7%  mass < 149.1
        - 0.02873715 * max(0.0, 0.000125186 - Q.e3_b2) / 6.986384e-05   # -2.9%  e3_b2 < 0.0001252
        + 0.02585548 * max(0.0, 135.5368 - Q.mass_top50) / 28.82074   # +2.6%  mass_top50 < 135.5
        - 0.02238442 * max(0.0, Q.mass_top50 - 108.5283) / 16.21581   # -2.2%  mass_top50 > 108.5
        - 0.02073227 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # -2.1%  sd_mass > 106.8
        + 0.01958021 * max(0.0, 0.06095651 - Q.e2_b2) / 0.02981164   # +2.0%  e2_b2 < 0.06096
        + 0.01829864 * max(0.0, Q.pair_mean_lnkt - 0.1701175) / 0.4479778   # +1.8%  pair_mean_lnkt > 0.1701
        + 0.01816465 * max(0.0, Q.mass_top50 - 71.80762) / 43.37124   # +1.8%  mass_top50 > 71.81
        - 0.01790034 * max(0.0, 0.0131933 - Q.e3_b05) / 0.006147996   # -1.8%  e3_b05 < 0.01319
        - 0.01631522 * max(0.0, 411.0 - Q.n_pairs_kt_above_1) / 159.5699   # -1.6%  n_pairs_kt_above_1 < 411
        - 0.0157998 * max(0.0, Q.sd_mass - 115.7091) / 10.28199   # -1.6%  sd_mass > 115.7
        + 0.01455048 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +1.5%  mass < 95.15
        - 0.01412634 * max(0.0, Q.N2 - 0.2734424) / 0.05469276   # -1.4%  N2 > 0.2734
        + 0.01032194 * max(0.0, 105.4845 - Q.sj3_pair_mass_max) / 19.86384   # +1.0%  sj3_pair_mass_max < 105.5
        - 0.01007688 * max(0.0, 71.96396 - Q.mass) / 1.934957   # -1.0%  mass < 71.96
        - 0.009300826 * max(0.0, Q.max_pair_mass - 25.35141) / 5.460395   # -0.9%  max_pair_mass > 25.35
        + 0.009224365 * max(0.0, 0.1651975 - Q.M2_b05) / 0.02986803   # +0.9%  M2_b05 < 0.1652
        + 0.00856752 * max(0.0, Q.mass - 110.2019) / 16.73354   # +0.9%  mass > 110.2
        - 0.007799882 * max(0.0, 76.91486 - Q.sj3_pair_mass_max) / 5.293372   # -0.8%  sj3_pair_mass_max < 76.91
        + 0.007793757 * max(0.0, 0.2638636 - Q.sj3_dr_min) / 0.09139393   # +0.8%  sj3_dr_min < 0.2639
        + 0.00768302 * max(0.0, Q.sd_mass - 123.2919) / 8.015693   # +0.8%  sd_mass > 123.3
        + 0.007191567 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # +0.7%  sj3_pair_mass_max < 73.25
        + 0.00687144 * max(0.0, 79.47361 - Q.mass) / 2.857843   # +0.7%  mass < 79.47
        + 0.006238 * max(0.0, 0.000125186 - Q.e3_b2) * max(0.0, 0.2303838 - Q.sj3_z3) / 8.29086e-06   # +0.6%  e3_b2 < 0.0001252 and sj3_z3 < 0.2304
        - 0.005962823 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # -0.6%  n_pairs_kt_above_3 < 34
        + 0.005923967 * max(0.0, 132.5189 - Q.mass_top40) / 28.13525   # +0.6%  mass_top40 < 132.5
        - 0.00548599 * max(0.0, 470.9609 - Q.sum_pt_top10) / 48.10208   # -0.5%  sum_pt_top10 < 471
        - 0.005115333 * max(0.0, Q.mass_top50 - 71.80762) * max(0.0, Q.sj3_dr13 - 0.07631451) / 16.66943   # -0.5%  mass_top50 > 71.81 and sj3_dr13 > 0.07631
        - 0.004592067 * max(0.0, 135.5368 - Q.mass_top50) * max(0.0, Q.dr01 - 0.009409147) / 2.944292   # -0.5%  mass_top50 < 135.5 and dr01 > 0.009409
        - 0.004576231 * max(0.0, 132.5189 - Q.mass_top40) * max(0.0, Q.tau32_b2 - 0.3377343) / 8.133784   # -0.5%  mass_top40 < 132.5 and tau32_b2 > 0.3377
        + 0.004529591 * max(0.0, 0.5871757 - Q.tau32) / 0.04593264   # +0.5%  tau32 < 0.5872
        + 0.004249586 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, 3.338396 - Q.pair_max_lnkt) / 5.710907   # +0.4%  n_pairs_kt_above_3 < 34 and pair_max_lnkt < 3.338
        + 0.003933736 * max(0.0, 149.0507 - Q.mass) * max(0.0, 0.897453 - Q.sj3_pairmax_over_m) / 3.503212   # +0.4%  mass < 149.1 and sj3_pairmax_over_m < 0.8975
        + 0.003591182 * max(0.0, Q.lnptrel_0 - -1.266303) / 0.1164165   # +0.4%  lnptrel_0 > -1.266
        - 0.003385455 * max(0.0, 0.000125186 - Q.e3_b2) * max(0.0, Q.lne_27 - 0.04953394) / 7.411781e-05   # -0.3%  e3_b2 < 0.0001252 and lne_27 > 0.04953
        - 0.003313174 * max(0.0, Q.mass_top10 - 89.15489) / 3.936474   # -0.3%  mass_top10 > 89.15
        + 0.002991662 * max(0.0, 0.3265797 - Q.tau21_b2) * max(0.0, Q.mass_top2 - 30.2961) / 0.522191   # +0.3%  tau21_b2 < 0.3266 and mass_top2 > 30.3
        - 0.002831368 * max(0.0, 0.3265797 - Q.tau21_b2) / 0.09224092   # -0.3%  tau21_b2 < 0.3266
        + 0.002683107 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # +0.3%  sj2_mass2 < 1.852
        + 0.002473897 * max(0.0, Q.m012 - 57.33311) / 1.633981   # +0.2%  m012 > 57.33
        - 0.002470167 * max(0.0, 95.14961 - Q.mass) * max(0.0, Q.n_dr_0_0p05 - 2.0) / 36.9883   # -0.2%  mass < 95.15 and n_dr_0_0p05 > 2
        + 0.002116373 * max(0.0, 135.5368 - Q.mass_top50) * max(0.0, Q.sum_e - 798.3291) / 5133.829   # +0.2%  mass_top50 < 135.5 and sum_e > 798.3
        - 0.001740758 * max(0.0, Q.pair_mean_lnm2 - 3.654683) / 0.143116   # -0.2%  pair_mean_lnm2 > 3.655
        - 0.001672055 * max(0.0, Q.mass_top50 - 146.203) * max(0.0, 0.6203012 - Q.tau32_b2) / 1.224425   # -0.2%  mass_top50 > 146.2 and tau32_b2 < 0.6203
        - 0.001626442 * max(0.0, Q.lund_max_lndelta - -0.8095462) / 0.06614362   # -0.2%  lund_max_lndelta > -0.8095
        + 0.001591777 * max(0.0, Q.N2 - 0.2734424) * max(0.0, Q.sj3_z1 - 0.4064022) / 0.01169291   # +0.2%  N2 > 0.2734 and sj3_z1 > 0.4064
        + 0.001500896 * max(0.0, Q.pair_mean_lnkt - 0.9367772) * max(0.0, 0.2325538 - Q.dr_17) / 0.005811503   # +0.2%  pair_mean_lnkt > 0.9368 and dr_17 < 0.2326
        - 0.001430538 * max(0.0, Q.mass_top50 - 71.80762) * max(0.0, Q.n_dr_0p05_0p1 - 5.0) / 96.21166   # -0.1%  mass_top50 > 71.81 and n_dr_0p05_0p1 > 5
        - 0.001361196 * max(0.0, Q.sd_mass - 123.2919) * max(0.0, 0.2116473 - Q.dr12) / 0.7052215   # -0.1%  sd_mass > 123.3 and dr12 < 0.2116
        - 0.001357916 * max(0.0, 0.5871757 - Q.tau32) * max(0.0, Q.sj2_mass1 - 77.42768) / 0.3282681   # -0.1%  tau32 < 0.5872 and sj2_mass1 > 77.43
        - 0.001175274 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.z_dr_0p2_0p4 - 0.1986699) / 0.1651849   # -0.1%  n_pairs_kt_above_3 < 34 and z_dr_0p2_0p4 > 0.1987
        + 0.0009736862 * max(0.0, Q.mass_top50 - 146.203) / 4.691111   # +0.1%  mass_top50 > 146.2
        + 0.000952296 * max(0.0, Q.pair_mean_lnm2 - 3.654683) * max(0.0, 0.4020128 - Q.dr_23) / 0.04068364   # +0.1%  pair_mean_lnm2 > 3.655 and dr_23 < 0.402
        - 0.000897944 * max(0.0, 0.3265797 - Q.tau21_b2) * max(0.0, Q.n_dr_0p05_0p1 - 7.0) / 0.127074   # -0.1%  tau21_b2 < 0.3266 and n_dr_0p05_0p1 > 7
        + 0.0008845972 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # +0.1%  sd_mass > 175.9
        - 0.0007732484 * max(0.0, 1.555555 - Q.pair_max_lnkt) / 0.02132551   # -0.1%  pair_max_lnkt < 1.556
        + 0.0007273737 * max(0.0, 149.0507 - Q.mass) * max(0.0, Q.min_pair_mass - 10.59762) / 7.118753   # +0.1%  mass < 149.1 and min_pair_mass > 10.6
        - 0.0007132725 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) * max(0.0, Q.ecf_g43 - 2.37474e-07) / 6.386888e-06   # -0.1%  sj3_pair_mass_max < 73.25 and ecf_g43 > 2.375e-07
        + 0.0006635037 * max(0.0, 16.48115 - Q.sj2_mass1) / 0.7890349   # +0.1%  sj2_mass1 < 16.48
        + 0.0006576377 * max(0.0, Q.sd_mass - 106.7501) * max(0.0, 0.5286514 - Q.max_dr) / 0.160105   # +0.1%  sd_mass > 106.8 and max_dr < 0.5287
        - 0.0006506638 * max(0.0, 149.0507 - Q.mass) * max(0.0, -2.500134 - Q.lnerel_1) / 1.471423   # -0.1%  mass < 149.1 and lnerel_1 < -2.5
        - 0.0006399314 * max(0.0, Q.lund_max_lndelta - -1.498181) / 0.4872254   # -0.1%  lund_max_lndelta > -1.498
        + 0.0006308956 * max(0.0, 0.5871757 - Q.tau32) * max(0.0, 3.516199 - Q.lne_4) / 0.005970061   # +0.1%  tau32 < 0.5872 and lne_4 < 3.516
        + 0.0006240602 * max(0.0, Q.mass_top50 - 108.5283) * max(0.0, 5.0 - Q.n_lund) / 1.509237   # +0.1%  mass_top50 > 108.5 and n_lund < 5
        - 0.0005426526 * max(0.0, Q.sd_rg - 0.4449101) / 0.02724738   # -0.1%  sd_rg > 0.4449
        + 0.0005307546 * max(0.0, 0.01202341 - Q.M3_b2) / 0.002697954   # +0.1%  M3_b2 < 0.01202
        - 0.000480726 * max(0.0, 0.5871757 - Q.tau32) * max(0.0, 20.0 - Q.n_real_top20) / 0.0202105   # -0.0%  tau32 < 0.5872 and n_real_top20 < 20
        - 0.0004570406 * max(0.0, Q.lnptrel_0 - -1.266303) * max(0.0, 0.4811724 - Q.max_dr) / 0.003428793   # -0.0%  lnptrel_0 > -1.266 and max_dr < 0.4812
        + 0.0004430943 * max(0.0, 149.0507 - Q.mass) * max(0.0, Q.ecf_g43 - 5.642202e-06) / 4.700717e-05   # +0.0%  mass < 149.1 and ecf_g43 > 5.642e-06
        - 0.000442786 * max(0.0, 135.5368 - Q.mass_top50) * max(0.0, Q.sd_zg - 0.2450652) / 1.845679   # -0.0%  mass_top50 < 135.5 and sd_zg > 0.2451
        - 0.0004230849 * max(0.0, Q.pair_max_lnkt - 2.98386) / 0.1266563   # -0.0%  pair_max_lnkt > 2.984
        - 0.0004084664 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # -0.0%  sd_mass > 154.6
        + 0.0003903418 * max(0.0, 0.01202341 - Q.M3_b2) * max(0.0, -0.03344727 - Q.eta_29) / 0.0001814308   # +0.0%  M3_b2 < 0.01202 and eta_29 < -0.03345
        + 0.0003031143 * max(0.0, Q.pair_mean_lnkt - 0.9367772) / 0.06159511   # +0.0%  pair_mean_lnkt > 0.9368
        - 0.0003006699 * max(0.0, 0.5871757 - Q.tau32) * max(0.0, 0.03403084 - Q.C2_b2) / 0.0001106386   # -0.0%  tau32 < 0.5872 and C2_b2 < 0.03403
        - 0.0002839289 * max(0.0, 0.2638636 - Q.sj3_dr_min) * max(0.0, 0.1648004 - Q.dr_9) / 0.004710811   # -0.0%  sj3_dr_min < 0.2639 and dr_9 < 0.1648
        + 0.0002723141 * max(0.0, Q.lnerel_9 - -3.64491) / 0.0924678   # +0.0%  lnerel_9 > -3.645
        - 0.0002604773 * max(0.0, 0.01202341 - Q.M3_b2) * max(0.0, 0.9036234 - Q.tau54) / 0.0001407024   # -0.0%  M3_b2 < 0.01202 and tau54 < 0.9036
        - 0.0002528537 * max(0.0, 0.5871757 - Q.tau32) * max(0.0, 0.02658081 - Q.eta_4) / 0.003923127   # -0.0%  tau32 < 0.5872 and eta_4 < 0.02658
        - 0.000197643 * max(0.0, 101.0 - Q.n_pairs_kt_above_1) / 8.83225   # -0.0%  n_pairs_kt_above_1 < 101
        + 0.0001900972 * max(0.0, 0.5871757 - Q.tau32) * max(0.0, Q.lam2 - 0.00972048) / 0.0002082107   # +0.0%  tau32 < 0.5872 and lam2 > 0.00972
        + 0.0001876804 * max(0.0, 470.9609 - Q.sum_pt_top10) * max(0.0, Q.max_dr - 0.7339085) / 5.119617   # +0.0%  sum_pt_top10 < 471 and max_dr > 0.7339
        - 0.0001865582 * max(0.0, 0.5871757 - Q.tau32) * max(0.0, Q.sj3_dr_max - 0.6054934) / 0.003707156   # -0.0%  tau32 < 0.5872 and sj3_dr_max > 0.6055
        + 0.0001696718 * max(0.0, Q.pair_mean_lndelta - -1.352792) / 0.01174331   # +0.0%  pair_mean_lndelta > -1.353
        + 0.00016071 * max(0.0, 0.000125186 - Q.e3_b2) * max(0.0, -0.09778748 - Q.eta_37) / 1.051829e-06   # +0.0%  e3_b2 < 0.0001252 and eta_37 < -0.09779
        + 0.0001284106 * max(0.0, Q.pt1_dr01 - 29.53862) / 1.088839   # +0.0%  pt1_dr01 > 29.54
        + 0.0001024552 * max(0.0, Q.mass_top50 - 146.203) * max(0.0, 0.4811724 - Q.max_dr) / 0.003268553   # +0.0%  mass_top50 > 146.2 and max_dr < 0.4812
        - 6.212508e-05 * max(0.0, 0.5871757 - Q.tau32) * max(0.0, 1.053727 - Q.sj3_mass3) / 0.003289301   # -0.0%  tau32 < 0.5872 and sj3_mass3 < 1.054
        - 5.122545e-05 * max(0.0, Q.lne_0 - 5.423297) / 0.1650486   # -0.0%  lne_0 > 5.423
        - 2.600698e-05 * max(0.0, 101.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.n_dr_0p05_0p1 - 16.0) / 0.09467   # -0.0%  n_pairs_kt_above_1 < 101 and n_dr_0p05_0p1 > 16
        - 1.208939e-05 * max(0.0, 16.48115 - Q.sj2_mass1) * max(0.0, Q.phi_26 - 0.3427734) / 0.001855891   # -0.0%  sj2_mass1 < 16.48 and phi_26 > 0.3428
        + 4.180579e-07 * max(0.0, 0.01202341 - Q.M3_b2) * max(0.0, 0.00586924 - Q.centroid_offset) / 2.05341e-06   # +0.0%  M3_b2 < 0.01202 and centroid_offset < 0.005869
    )
    return z


def neuron_82(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.0004023885
    )
    return z


def neuron_83(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.658984e-05
    )
    return z


def neuron_84(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.650188e-06
    )
    return z


def neuron_85(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.717487e-06
    )
    return z


def neuron_86(Q):
    # scale S = 7.756; each line: share * term / its average size
    z = 7.755501 * (0.1635962
        - 0.1781619 * max(0.0, Q.z_top15_slots - 0.6733431) / 0.1722908   # -17.8%  z_top15_slots > 0.6733
        + 0.08551245 * max(0.0, Q.z_top30_slots - 0.8695891) / 0.09042836   # +8.6%  z_top30_slots > 0.8696
        - 0.05725229 * Q.tau5 / 0.0321999   # -5.7%  tau5
        - 0.05580587 * max(0.0, Q.n_particles - 23.0) / 16.99045   # -5.6%  n_particles > 23
        + 0.04990732 * max(0.0, 0.07659457 - Q.tau3) / 0.03136078   # +5.0%  tau3 < 0.07659
        + 0.03912045 * max(0.0, 123.7928 - Q.mass) / 19.43787   # +3.9%  mass < 123.8
        + 0.03829955 * Q.M2_b2 / 0.03249042   # +3.8%  M2_b2
        - 0.03721861 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # -3.7%  e3_b2 < 0.000791
        + 0.03193074 * max(0.0, 131.3917 - Q.mass) / 24.784   # +3.2%  mass < 131.4
        - 0.03171382 * max(0.0, Q.M2_b05 - 0.1258758) / 0.01827883   # -3.2%  M2_b05 > 0.1259
        + 0.03164264 * max(0.0, Q.M3 - 0.02540381) / 0.01242824   # +3.2%  M3 > 0.0254
        + 0.02759933 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) / 143.7916   # +2.8%  n_pairs_kt_above_3 < 220
        - 0.02494927 * max(0.0, 95.14961 - Q.mass) / 6.371707   # -2.5%  mass < 95.15
        - 0.02274797 * max(0.0, 0.04863496 - Q.M3) / 0.01534745   # -2.3%  M3 < 0.04863
        + 0.01984921 * max(0.0, Q.lne_22 - 0.3560888) / 1.289173   # +2.0%  lne_22 > 0.3561
        + 0.01933037 * max(0.0, 131.3917 - Q.mass) * max(0.0, 0.2099278 - Q.N2_b2) / 1.35478   # +1.9%  mass < 131.4 and N2_b2 < 0.2099
        - 0.0145834 * max(0.0, 3.019755 - Q.D2_b2) / 1.204398   # -1.5%  D2_b2 < 3.02
        - 0.01426868 * max(0.0, 0.07659457 - Q.tau3) * max(0.0, 1650.879 - Q.jet_e) / 23.75171   # -1.4%  tau3 < 0.07659 and jet_e < 1651
        + 0.01051738 * max(0.0, 0.07991731 - Q.tau2) * max(0.0, 0.1189606 - Q.dr_53) / 0.002403157   # +1.1%  tau2 < 0.07992 and dr_53 < 0.119
        + 0.009807105 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +1.0%  n_dr_0p4_up < 4
        - 0.009611909 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -1.0%  mass < 90.09
        + 0.009447488 * max(0.0, Q.z_top30_slots - 0.8695891) * max(0.0, 0.2410564 - Q.sj3_dr_min) / 0.008128749   # +0.9%  z_top30_slots > 0.8696 and sj3_dr_min < 0.2411
        - 0.009193844 * max(0.0, 1.954943 - Q.D2_b05) / 0.3207466   # -0.9%  D2_b05 < 1.955
        + 0.008514371 * max(0.0, 0.06625964 - Q.M2) / 0.006053324   # +0.9%  M2 < 0.06626
        - 0.008392771 * max(0.0, 3.375095e-06 - Q.ecf_g43) / 9.649919e-07   # -0.8%  ecf_g43 < 3.375e-06
        - 0.007831534 * max(0.0, 0.07659457 - Q.tau3) * max(0.0, Q.lund3_lndelta - -2.817283) / 0.03503399   # -0.8%  tau3 < 0.07659 and lund3_lndelta > -2.817
        - 0.00727891 * max(0.0, 1.868856 - Q.m01) / 0.2758198   # -0.7%  m01 < 1.869
        + 0.007036281 * max(0.0, Q.lund2_lndelta - -1.124734) / 0.274477   # +0.7%  lund2_lndelta > -1.125
        + 0.007027358 * max(0.0, 4.0 - Q.n_dr_0_0p05) / 2.08424   # +0.7%  n_dr_0_0p05 < 4
        - 0.006843816 * max(0.0, 0.4545826 - Q.N3_b2) / 0.1894157   # -0.7%  N3_b2 < 0.4546
        - 0.006571949 * max(0.0, 54.0 - Q.n_pairs_kt_above_3) / 12.86598   # -0.7%  n_pairs_kt_above_3 < 54
        + 0.00623272 * max(0.0, 18.73268 - Q.sj3_mass1) / 4.11578   # +0.6%  sj3_mass1 < 18.73
        - 0.005338466 * max(0.0, 0.1259759 - Q.sj3_dr_min) / 0.01757215   # -0.5%  sj3_dr_min < 0.126
        - 0.005305282 * max(0.0, Q.M3 - 0.02540381) * max(0.0, Q.n_dr_0p05_0p1 - 1.0) / 0.06282844   # -0.5%  M3 > 0.0254 and n_dr_0p05_0p1 > 1
        - 0.005204007 * max(0.0, 0.07991731 - Q.tau2) / 0.0205282   # -0.5%  tau2 < 0.07992
        + 0.005116794 * max(0.0, 5.0 - Q.n_dr_0p1_0p2) / 0.5919233   # +0.5%  n_dr_0p1_0p2 < 5
        + 0.00470269 * max(0.0, Q.n_dr_0p4_up - 9.0) / 1.570153   # +0.5%  n_dr_0p4_up > 9
        - 0.004638782 * max(0.0, 0.03259227 - Q.M3) / 0.00464829   # -0.5%  M3 < 0.03259
        - 0.004543415 * max(0.0, 0.01600475 - Q.dr_min_012) / 0.003959346   # -0.5%  dr_min_012 < 0.016
        + 0.003941552 * max(0.0, 0.01919357 - Q.M3_b2) / 0.007639717   # +0.4%  M3_b2 < 0.01919
        - 0.003658334 * max(0.0, 0.07991731 - Q.tau2) * max(0.0, 0.7109507 - Q.sj3_z1) / 0.002033834   # -0.4%  tau2 < 0.07992 and sj3_z1 < 0.711
        + 0.003611648 * max(0.0, 1.617171 - Q.sj3_mass3) / 0.2158965   # +0.4%  sj3_mass3 < 1.617
        - 0.003608564 * max(0.0, 0.3265797 - Q.tau21_b2) / 0.09224092   # -0.4%  tau21_b2 < 0.3266
        - 0.003464843 * max(0.0, 67.20576 - Q.mass_top30) / 1.774095   # -0.3%  mass_top30 < 67.21
        + 0.003307242 * max(0.0, Q.lne_22 - 0.3560888) * max(0.0, 0.04908137 - Q.dr01) / 0.01810537   # +0.3%  lne_22 > 0.3561 and dr01 < 0.04908
        + 0.003267872 * max(0.0, 0.04863496 - Q.M3) * max(0.0, 10.0 - Q.n_pairs_kt_above_10) / 0.07268588   # +0.3%  M3 < 0.04863 and n_pairs_kt_above_10 < 10
        - 0.003221268 * max(0.0, 18.73268 - Q.sj3_mass1) * max(0.0, 0.02553503 - Q.dr_33) / 0.07049185   # -0.3%  sj3_mass1 < 18.73 and dr_33 < 0.02554
        + 0.003120404 * max(0.0, 92.68149 - Q.mass_top40) / 5.990263   # +0.3%  mass_top40 < 92.68
        + 0.003004205 * max(0.0, 0.01919357 - Q.M3_b2) * max(0.0, Q.sj2_mass2 - 3.928781) / 0.1110228   # +0.3%  M3_b2 < 0.01919 and sj2_mass2 > 3.929
        + 0.002350849 * max(0.0, 56.73313 - Q.sj3_pair_mass_max) / 1.542034   # +0.2%  sj3_pair_mass_max < 56.73
        - 0.002264905 * max(0.0, Q.lund_max_lndelta - -0.8095462) / 0.06614362   # -0.2%  lund_max_lndelta > -0.8095
        - 0.002263998 * max(0.0, 0.02252173 - Q.dr02) / 0.003357714   # -0.2%  dr02 < 0.02252
        - 0.002261057 * max(0.0, 0.004729211 - Q.z_dr_0p4_up) / 0.0008408066   # -0.2%  z_dr_0p4_up < 0.004729
        + 0.002257477 * max(0.0, 0.5546022 - Q.z_top10_slots) / 0.005826969   # +0.2%  z_top10_slots < 0.5546
        - 0.002135097 * max(0.0, 0.3265797 - Q.tau21_b2) * max(0.0, 5.311506 - Q.sj2_mass2) / 0.04229496   # -0.2%  tau21_b2 < 0.3266 and sj2_mass2 < 5.312
        + 0.002127165 * max(0.0, 0.008254376 - Q.z_dr_0p05_0p1) / 0.0009507355   # +0.2%  z_dr_0p05_0p1 < 0.008254
        + 0.002058544 * max(0.0, Q.pair_max_lnm2 - 7.752558) / 0.04350608   # +0.2%  pair_max_lnm2 > 7.753
        - 0.001879604 * max(0.0, 0.6922585 - Q.z_top10_slots) / 0.03483906   # -0.2%  z_top10_slots < 0.6923
        - 0.001448659 * max(0.0, 0.07991731 - Q.tau2) * max(0.0, Q.sum_z_dr2_top10 - 0.04905122) / 1.316657e-05   # -0.1%  tau2 < 0.07992 and sum_z_dr2_top10 > 0.04905
        - 0.001393428 * max(0.0, 3.019755 - Q.D2_b2) * max(0.0, Q.sj3_mass2 - 6.769801) / 5.55555   # -0.1%  D2_b2 < 3.02 and sj3_mass2 > 6.77
        - 0.001356013 * max(0.0, 5.0 - Q.n_lund_kt_above_1) / 0.64189   # -0.1%  n_lund_kt_above_1 < 5
        - 0.001154416 * max(0.0, Q.lam2 - 0.02148541) / 0.0004759345   # -0.1%  lam2 > 0.02149
        + 0.001064388 * max(0.0, Q.M2_b05 - 0.1258758) * max(0.0, 0.3969707 - Q.dr_53) / 0.00608326   # +0.1%  M2_b05 > 0.1259 and dr_53 < 0.397
        - 0.001063293 * max(0.0, Q.n_particles - 23.0) * max(0.0, -1.763117 - Q.lund3_lnz) / 28.4553   # -0.1%  n_particles > 23 and lund3_lnz < -1.763
        - 0.0009384077 * max(0.0, Q.lnerel_50 - -5.787293) / 0.02986882   # -0.1%  lnerel_50 > -5.787
        - 0.0008569885 * max(0.0, 0.07659457 - Q.tau3) * max(0.0, Q.n_lund_kt_above_1 - 6.0) / 0.007113277   # -0.1%  tau3 < 0.07659 and n_lund_kt_above_1 > 6
        - 0.0007980376 * max(0.0, 0.01600475 - Q.dr_min_012) * max(0.0, 0.2119274 - Q.dr_35) / 0.0005136813   # -0.1%  dr_min_012 < 0.016 and dr_35 < 0.2119
        - 0.0007575363 * max(0.0, 123.0 - Q.n_pairs_kt_above_1) / 13.76892   # -0.1%  n_pairs_kt_above_1 < 123
        + 0.0007540908 * max(0.0, 0.0007909605 - Q.e3_b2) * max(0.0, Q.sj2_mass1 - 60.04967) / 0.001562723   # +0.1%  e3_b2 < 0.000791 and sj2_mass1 > 60.05
        - 0.0007252224 * max(0.0, 0.4545826 - Q.N3_b2) * max(0.0, Q.dr_9 - 0.1346091) / 0.01410756   # -0.1%  N3_b2 < 0.4546 and dr_9 > 0.1346
        - 0.0006348466 * max(0.0, 0.01919357 - Q.M3_b2) * max(0.0, 0.4540665 - Q.dr_16) / 0.00177042   # -0.1%  M3_b2 < 0.01919 and dr_16 < 0.4541
        + 0.0004580527 * max(0.0, 0.4545826 - Q.N3_b2) * max(0.0, Q.dr_19 - 0.3410959) / 0.006648879   # +0.0%  N3_b2 < 0.4546 and dr_19 > 0.3411
        + 0.0004522287 * max(0.0, 0.01600475 - Q.dr_min_012) * max(0.0, Q.phi_54 - -0.1107788) / 0.0004622354   # +0.0%  dr_min_012 < 0.016 and phi_54 > -0.1108
        - 0.0004022798 * max(0.0, Q.lne_22 - 0.3560888) * max(0.0, 0.8202927 - Q.planar_flow) / 0.4024599   # -0.0%  lne_22 > 0.3561 and planar_flow < 0.8203
        - 0.0003998864 * max(0.0, 0.02252173 - Q.dr02) * max(0.0, Q.dr_14 - 0.2250629) / 0.0002168813   # -0.0%  dr02 < 0.02252 and dr_14 > 0.2251
        - 0.0003751344 * max(0.0, Q.n_particles - 23.0) * max(0.0, Q.dr_7 - 0.3042435) / 0.3114922   # -0.0%  n_particles > 23 and dr_7 > 0.3042
        + 0.0003321467 * max(0.0, Q.lne_22 - 0.3560888) * max(0.0, Q.dr_34 - 0.3708198) / 0.04488919   # +0.0%  lne_22 > 0.3561 and dr_34 > 0.3708
        - 0.0003184932 * max(0.0, 4.0 - Q.n_dr_0p4_up) * max(0.0, Q.sj4_pair_mass_min - 16.05703) / 1.446982   # -0.0%  n_dr_0p4_up < 4 and sj4_pair_mass_min > 16.06
        + 0.0002638409 * max(0.0, 0.03259227 - Q.M3) * max(0.0, Q.eta_50 - 0.03570557) / 0.0002083886   # +0.0%  M3 < 0.03259 and eta_50 > 0.03571
        + 0.0002412204 * max(0.0, 3.019755 - Q.D2_b2) * max(0.0, 0.3544847 - Q.pt1_dr01) / 0.0066829   # +0.0%  D2_b2 < 3.02 and pt1_dr01 < 0.3545
        - 0.0002262223 * max(0.0, Q.lund2_lndelta - -1.124734) * max(0.0, Q.dr_max_012 - 0.5651787) / 0.002619115   # -0.0%  lund2_lndelta > -1.125 and dr_max_012 > 0.5652
        + 0.0001656314 * max(0.0, 3.019755 - Q.D2_b2) * max(0.0, -0.2314453 - Q.phi_22) / 0.01397573   # +0.0%  D2_b2 < 3.02 and phi_22 < -0.2314
        + 0.0001535779 * max(0.0, 0.1445164 - Q.mass_over_sum_pt) / 0.008435599   # +0.0%  mass_over_sum_pt < 0.1445
        + 0.0001283548 * max(0.0, 0.07991731 - Q.tau2) * max(0.0, Q.lnpt_10 - 3.138208) / 0.0001549072   # +0.0%  tau2 < 0.07992 and lnpt_10 > 3.138
        + 8.198658e-05 * max(0.0, 0.008254376 - Q.z_dr_0p05_0p1) * max(0.0, -0.02503967 - Q.phi_19) / 7.006191e-05   # +0.0%  z_dr_0p05_0p1 < 0.008254 and phi_19 < -0.02504
        + 7.912411e-05 * max(0.0, Q.lne_20 - 2.544918) / 0.04190674   # +0.0%  lne_20 > 2.545
        + 4.424522e-05 * max(0.0, 0.07659457 - Q.tau3) * max(0.0, Q.eta_37 - 0.1695557) / 0.0003595731   # +0.0%  tau3 < 0.07659 and eta_37 > 0.1696
        + 4.139751e-05 * max(0.0, Q.pair_max_lnm2 - 7.752558) * max(0.0, 0.3615236 - Q.dr_13) / 0.005866326   # +0.0%  pair_max_lnm2 > 7.753 and dr_13 < 0.3615
        + 3.283498e-05 * max(0.0, 3.019755 - Q.D2_b2) * max(0.0, Q.eta_55 - 0.1072388) / 0.009473871   # +0.0%  D2_b2 < 3.02 and eta_55 > 0.1072
        + 2.604669e-06 * max(0.0, Q.pair_max_lnm2 - 7.752558) * max(0.0, Q.n_pairs_kt_above_30 - 1.0) / 0.02833342   # +0.0%  pair_max_lnm2 > 7.753 and n_pairs_kt_above_30 > 1
    )
    return z


def neuron_87(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.63959e-06
    )
    return z


def neuron_88(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.16205e-05
    )
    return z


def neuron_89(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.851283e-06
    )
    return z


def neuron_90(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.897432e-05
    )
    return z


def neuron_91(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.238619e-06
    )
    return z


def neuron_92(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.794084e-06
    )
    return z


def neuron_93(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.100003e-06
    )
    return z


def neuron_94(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.384212e-05
    )
    return z


def neuron_95(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.277345e-05
    )
    return z


def neuron_96(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.324139e-05
    )
    return z


def neuron_97(Q):
    # scale S = 4.309; each line: share * term / its average size
    z = 4.308863 * (-0.2499706
        + 0.1325063 * max(0.0, Q.pt_entropy - 1.63996) / 1.150155   # +13.3%  pt_entropy > 1.64
        + 0.08821089 * Q.M3 / 0.03605934   # +8.8%  M3
        + 0.05900317 * max(0.0, 0.2250047 - Q.C3_b05) / 0.1186192   # +5.9%  C3_b05 < 0.225
        + 0.04194854 * max(0.0, 0.02497878 - Q.M3_b2) / 0.01258377   # +4.2%  M3_b2 < 0.02498
        - 0.0418871 * max(0.0, 44.0 - Q.n_real_top50) / 8.918697   # -4.2%  n_real_top50 < 44
        - 0.03294814 * max(0.0, 23.0 - Q.n_pairs_kt_above_10) / 16.58492   # -3.3%  n_pairs_kt_above_10 < 23
        + 0.03249812 * max(0.0, 10.0 - Q.n_pairs_kt_above_10) / 5.075437   # +3.2%  n_pairs_kt_above_10 < 10
        + 0.02741876 * max(0.0, 123.0 - Q.n_pairs_kt_above_1) / 13.76892   # +2.7%  n_pairs_kt_above_1 < 123
        - 0.02654051 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) * max(0.0, 7.184834e-06 - Q.e4) / 0.0009319273   # -2.7%  n_pairs_kt_above_3 < 220 and e4 < 7.185e-06
        - 0.02377346 * max(0.0, 2.692859 - Q.D2) / 0.9215868   # -2.4%  D2 < 2.693
        + 0.02302242 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) / 143.7916   # +2.3%  n_pairs_kt_above_3 < 220
        + 0.02010191 * max(0.0, Q.m012 - 32.7897) / 6.653649   # +2.0%  m012 > 32.79
        + 0.01988897 * max(0.0, 23.0 - Q.n_pairs_kt_above_10) * max(0.0, Q.mass_top5 - 34.01347) / 207.0275   # +2.0%  n_pairs_kt_above_10 < 23 and mass_top5 > 34.01
        + 0.01774345 * max(0.0, 4.9989 - Q.lnpt_1) / 0.7363106   # +1.8%  lnpt_1 < 4.999
        + 0.01580288 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) / 9.886423   # +1.6%  n_pairs_kt_above_3 < 47
        + 0.01492673 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +1.5%  sd_mass > 154.6
        - 0.01488837 * max(0.0, 123.0 - Q.n_pairs_kt_above_1) * max(0.0, 3.0 - Q.n_pairs_kt_above_10) / 19.57475   # -1.5%  n_pairs_kt_above_1 < 123 and n_pairs_kt_above_10 < 3
        + 0.01456307 * max(0.0, Q.tau2 - 0.04957212) / 0.03609822   # +1.5%  tau2 > 0.04957
        - 0.01399624 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.dr_2 - 0.04584392) / 12.37236   # -1.4%  n_pairs_kt_above_3 < 220 and dr_2 > 0.04584
        + 0.01392969 * max(0.0, 110.7667 - Q.mass_top15) / 31.75754   # +1.4%  mass_top15 < 110.8
        - 0.01371672 * max(0.0, Q.mass - 123.7928) / 10.57625   # -1.4%  mass > 123.8
        - 0.01271668 * max(0.0, 0.3436326 - Q.N2_b05) / 0.005384222   # -1.3%  N2_b05 < 0.3436
        - 0.01256416 * max(0.0, 0.006832265 - Q.C3_b2) / 0.00348603   # -1.3%  C3_b2 < 0.006832
        - 0.0123402 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # -1.2%  sd_mass > 106.8
        + 0.01228061 * max(0.0, 44.0 - Q.n_real_top50) * max(0.0, Q.n_lund - 7.0) / 23.69225   # +1.2%  n_real_top50 < 44 and n_lund > 7
        - 0.01126325 * max(0.0, Q.mass_top5 - 44.18338) / 9.573881   # -1.1%  mass_top5 > 44.18
        - 0.01090848 * max(0.0, Q.max_pair_mass - 28.31178) / 4.498986   # -1.1%  max_pair_mass > 28.31
        - 0.01013856 * max(0.0, 0.4675349 - Q.D3_b2) / 0.3220351   # -1.0%  D3_b2 < 0.4675
        + 0.01001046 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.6052136 - Q.planar_flow) / 2.056674   # +1.0%  n_pairs_kt_above_3 < 47 and planar_flow < 0.6052
        + 0.009774836 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.dr02 - 0.1620436) / 10.24846   # +1.0%  n_pairs_kt_above_3 < 220 and dr02 > 0.162
        - 0.009279326 * max(0.0, 70.93762 - Q.mass_top15) / 6.558806   # -0.9%  mass_top15 < 70.94
        - 0.009208107 * max(0.0, Q.lund3_lndelta - -2.4279) / 0.9642284   # -0.9%  lund3_lndelta > -2.428
        - 0.008720197 * max(0.0, 78.33213 - Q.mass_top40) / 2.799359   # -0.9%  mass_top40 < 78.33
        - 0.008625345 * max(0.0, 4.9989 - Q.lnpt_1) * max(0.0, Q.tau21 - 0.1757731) / 0.1994879   # -0.9%  lnpt_1 < 4.999 and tau21 > 0.1758
        - 0.007755501 * max(0.0, 0.02139785 - Q.dr_min_012) / 0.006957017   # -0.8%  dr_min_012 < 0.0214
        + 0.00747489 * max(0.0, Q.m01 - 35.44963) / 2.286753   # +0.7%  m01 > 35.45
        + 0.007282106 * max(0.0, 0.1927916 - Q.sum_z_dr) / 0.05101873   # +0.7%  sum_z_dr < 0.1928
        - 0.007102976 * max(0.0, Q.sd_mass - 127.8042) / 7.009137   # -0.7%  sd_mass > 127.8
        - 0.006594683 * max(0.0, Q.pair_mean_lnz - -1.676789) / 0.09958353   # -0.7%  pair_mean_lnz > -1.677
        - 0.006481694 * max(0.0, 0.005508318 - Q.sum_z_dr2_top5) / 0.0006372793   # -0.6%  sum_z_dr2_top5 < 0.005508
        - 0.006422147 * max(0.0, Q.lund_max_lndelta - -0.615738) / 0.02343634   # -0.6%  lund_max_lndelta > -0.6157
        + 0.005846331 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.M3 - 0.02886227) / 1.883703   # +0.6%  n_pairs_kt_above_3 < 220 and M3 > 0.02886
        + 0.005649777 * max(0.0, 0.02497878 - Q.M3_b2) * max(0.0, 0.3179541 - Q.pt2_over_pt0) / 0.0004533683   # +0.6%  M3_b2 < 0.02498 and pt2_over_pt0 < 0.318
        - 0.005465875 * max(0.0, 0.1184723 - Q.sum_z_dr) / 0.01124185   # -0.5%  sum_z_dr < 0.1185
        + 0.005320897 * max(0.0, 12.0 - Q.n_dr_0p4_up) / 6.849897   # +0.5%  n_dr_0p4_up < 12
        + 0.004998216 * max(0.0, 0.2250047 - Q.C3_b05) * max(0.0, -0.4705779 - Q.lund1_lndelta) / 0.01950301   # +0.5%  C3_b05 < 0.225 and lund1_lndelta < -0.4706
        - 0.004987123 * max(0.0, 0.1184723 - Q.sum_z_dr) * max(0.0, Q.lnpt_0 - 4.993828) / 0.004139561   # -0.5%  sum_z_dr < 0.1185 and lnpt_0 > 4.994
        - 0.004738861 * max(0.0, Q.z_top20_slots - 0.9741142) / 0.004568937   # -0.5%  z_top20_slots > 0.9741
        + 0.004572753 * max(0.0, 4.9989 - Q.lnpt_1) * max(0.0, Q.z_dr_0p1_0p2 - 0.4056731) / 0.06860314   # +0.5%  lnpt_1 < 4.999 and z_dr_0p1_0p2 > 0.4057
        - 0.004461649 * max(0.0, 110.7667 - Q.mass_top15) * max(0.0, 0.2001251 - Q.pt2_over_pt0) / 0.3196475   # -0.4%  mass_top15 < 110.8 and pt2_over_pt0 < 0.2001
        + 0.004300132 * max(0.0, Q.m012 - 32.7897) * max(0.0, 0.3209146 - Q.dr_1) / 0.7397002   # +0.4%  m012 > 32.79 and dr_1 < 0.3209
        - 0.004185581 * max(0.0, Q.m012 - 32.7897) * max(0.0, 2.01745 - Q.D2_b2) / 6.088604   # -0.4%  m012 > 32.79 and D2_b2 < 2.017
        - 0.004169999 * max(0.0, Q.pt_entropy - 1.63996) * max(0.0, 6.0 - Q.n_lund_kt_above_1) / 0.9404317   # -0.4%  pt_entropy > 1.64 and n_lund_kt_above_1 < 6
        - 0.003721864 * max(0.0, 0.68373 - Q.sj3_z1) / 0.1058115   # -0.4%  sj3_z1 < 0.6837
        - 0.003703525 * max(0.0, Q.m01 - 35.44963) * max(0.0, 1.462588 - Q.D3_b2) / 2.205006   # -0.4%  m01 > 35.45 and D3_b2 < 1.463
        + 0.003549129 * max(0.0, 0.2001251 - Q.pt2_over_pt0) / 0.01182056   # +0.4%  pt2_over_pt0 < 0.2001
        - 0.003358397 * max(0.0, Q.N2_b2 - 0.255112) / 0.008335535   # -0.3%  N2_b2 > 0.2551
        - 0.003232377 * max(0.0, 0.02210827 - Q.dr_max_012) / 0.001265414   # -0.3%  dr_max_012 < 0.02211
        + 0.003159288 * max(0.0, Q.pt_entropy - 1.63996) * max(0.0, -0.603253 - Q.lund3_lndelta) / 0.9515233   # +0.3%  pt_entropy > 1.64 and lund3_lndelta < -0.6033
        - 0.002568581 * max(0.0, 0.006832265 - Q.C3_b2) * max(0.0, 0.00859211 - Q.mean_phi2) / 4.772795e-06   # -0.3%  C3_b2 < 0.006832 and mean_phi2 < 0.008592
        - 0.002559206 * max(0.0, 0.1338707 - Q.M2_b05) / 0.01007341   # -0.3%  M2_b05 < 0.1339
        - 0.002516975 * max(0.0, Q.pt_entropy - 1.63996) * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 1.540901   # -0.3%  pt_entropy > 1.64 and n_lund_kt_above_5 > 1
        - 0.002507416 * max(0.0, 123.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 4.63537   # -0.3%  n_pairs_kt_above_1 < 123 and n_lund_kt_above_5 > 1
        + 0.002418241 * max(0.0, Q.mass - 164.4374) / 3.038568   # +0.2%  mass > 164.4
        - 0.002244976 * max(0.0, 4.9989 - Q.lnpt_1) * max(0.0, -0.752531 - Q.lund_max_lndelta) / 0.2458641   # -0.2%  lnpt_1 < 4.999 and lund_max_lndelta < -0.7525
        - 0.002051758 * max(0.0, 220.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.min_pair_mass - 2.606496) / 166.571   # -0.2%  n_pairs_kt_above_3 < 220 and min_pair_mass > 2.606
        + 0.002051031 * max(0.0, Q.mass_top5 - 44.18338) * max(0.0, 0.4680886 - Q.tau32_b2) / 0.8623256   # +0.2%  mass_top5 > 44.18 and tau32_b2 < 0.4681
        + 0.001876444 * max(0.0, 2.692859 - Q.D2) * max(0.0, 0.5796538 - Q.tau43_b2) / 0.03041731   # +0.2%  D2 < 2.693 and tau43_b2 < 0.5797
        - 0.001765521 * max(0.0, Q.mass - 182.8592) / 1.688248   # -0.2%  mass > 182.9
        - 0.001642307 * max(0.0, 44.0 - Q.n_real_top50) * max(0.0, 0.2250629 - Q.dr_14) / 0.7286005   # -0.2%  n_real_top50 < 44 and dr_14 < 0.2251
        + 0.001571348 * max(0.0, Q.n_lund - 10.0) / 1.682617   # +0.2%  n_lund > 10
        + 0.001529444 * max(0.0, -2.572052 - Q.pair_mean_lnz) / 0.02771534   # +0.2%  pair_mean_lnz < -2.572
        - 0.001504015 * max(0.0, 0.2250047 - Q.C3_b05) * max(0.0, 0.8629612 - Q.psi_0p3) / 0.004262556   # -0.2%  C3_b05 < 0.225 and psi_0p3 < 0.863
        + 0.001458896 * max(0.0, Q.pair_mean_lnkt - 1.370103) / 0.01395839   # +0.1%  pair_mean_lnkt > 1.37
        - 0.001241871 * max(0.0, 0.02139785 - Q.dr_min_012) * max(0.0, 0.09477716 - Q.dr_47) / 0.0005231534   # -0.1%  dr_min_012 < 0.0214 and dr_47 < 0.09478
        + 0.001120683 * max(0.0, 2.692859 - Q.D2) * max(0.0, Q.sum_e - 1247.266) / 43.52122   # +0.1%  D2 < 2.693 and sum_e > 1247
        - 0.0009167617 * max(0.0, 6.0 - Q.n_dr_0p1_0p2) / 0.8571967   # -0.1%  n_dr_0p1_0p2 < 6
        - 0.0007673957 * max(0.0, 0.1184723 - Q.sum_z_dr) * max(0.0, -2.471159 - Q.lnerel_4) / 0.006332745   # -0.1%  sum_z_dr < 0.1185 and lnerel_4 < -2.471
        - 0.0007638007 * max(0.0, 0.07980588 - Q.tau21_b2) / 0.002810141   # -0.1%  tau21_b2 < 0.07981
        + 0.0007454969 * max(0.0, Q.sj3_pair_mass_max - 105.4845) * max(0.0, 0.3956483 - Q.sj3_dr_min) / 0.5883554   # +0.1%  sj3_pair_mass_max > 105.5 and sj3_dr_min < 0.3956
        + 0.0007404402 * max(0.0, 74.51927 - Q.mass_top30) / 2.669461   # +0.1%  mass_top30 < 74.52
        - 0.000635074 * max(0.0, 4.9989 - Q.lnpt_1) * max(0.0, Q.e3_b2 - 1.78923e-05) / 0.0001385901   # -0.1%  lnpt_1 < 4.999 and e3_b2 > 1.789e-05
        - 0.0005853754 * max(0.0, Q.sd_mass - 154.5947) * max(0.0, 0.2412858 - Q.pt2_over_pt0) / 0.04340868   # -0.1%  sd_mass > 154.6 and pt2_over_pt0 < 0.2413
        + 0.0005703652 * max(0.0, 4.621465 - Q.sj3_mass1) / 0.129745   # +0.1%  sj3_mass1 < 4.621
        + 0.0005693714 * max(0.0, 0.02858957 - Q.C2_b2) / 0.004118505   # +0.1%  C2_b2 < 0.02859
        - 0.0005481343 * max(0.0, Q.max_pair_mass - 55.24488) / 0.6579942   # -0.1%  max_pair_mass > 55.24
        - 0.0005291672 * max(0.0, 78.33213 - Q.mass_top40) * max(0.0, 2.102317 - Q.min_pair_mass) / 2.928426   # -0.1%  mass_top40 < 78.33 and min_pair_mass < 2.102
        - 0.0005246334 * max(0.0, 4.9989 - Q.lnpt_1) * max(0.0, Q.dr_19 - 0.1712001) / 0.06539796   # -0.1%  lnpt_1 < 4.999 and dr_19 > 0.1712
        - 0.000520645 * max(0.0, 0.2001251 - Q.pt2_over_pt0) * max(0.0, Q.z_dr_0p2_0p4 - 0.1724455) / 0.0006950108   # -0.1%  pt2_over_pt0 < 0.2001 and z_dr_0p2_0p4 > 0.1724
        - 0.0004843908 * max(0.0, Q.sj3_pair_mass_max - 105.4845) * max(0.0, 0.1500144 - Q.dr_31) / 0.1769266   # -0.0%  sj3_pair_mass_max > 105.5 and dr_31 < 0.15
        + 0.0004486767 * max(0.0, Q.mass - 123.7928) * max(0.0, 1.824785 - Q.sj3_mass2) / 0.3018868   # +0.0%  mass > 123.8 and sj3_mass2 < 1.825
        - 0.0004415694 * max(0.0, 0.2001251 - Q.pt2_over_pt0) * max(0.0, Q.C3_b2 - 0.04229114) / 0.0001890768   # -0.0%  pt2_over_pt0 < 0.2001 and C3_b2 > 0.04229
        - 0.0004298053 * max(0.0, Q.sd_mass - 127.8042) * max(0.0, 1.824785 - Q.sj3_mass2) / 0.1975212   # -0.0%  sd_mass > 127.8 and sj3_mass2 < 1.825
        - 0.0004131266 * max(0.0, Q.max_pair_mass - 55.24488) * max(0.0, 2.102317 - Q.min_pair_mass) / 0.2811229   # -0.0%  max_pair_mass > 55.24 and min_pair_mass < 2.102
        + 0.0003926334 * max(0.0, Q.mass_top5 - 68.52153) / 2.713876   # +0.0%  mass_top5 > 68.52
        + 0.0003437228 * max(0.0, 123.0 - Q.n_pairs_kt_above_1) * max(0.0, 6.310762 - Q.log_sum_pt) / 0.2251313   # +0.0%  n_pairs_kt_above_1 < 123 and log_sum_pt < 6.311
        - 0.0002418375 * max(0.0, Q.m01 - 35.44963) * max(0.0, 0.4959595 - Q.dr12) / 0.7258492   # -0.0%  m01 > 35.45 and dr12 < 0.496
        + 3.69009e-05 * max(0.0, 0.02858957 - Q.C2_b2) * max(0.0, Q.eta_31 - 0.0) / 0.0001147829   # +0.0%  C2_b2 < 0.02859 and eta_31 > 0
        + 2.25568e-05 * max(0.0, Q.sj3_pair_mass_max - 105.4845) / 6.720107   # +0.0%  sj3_pair_mass_max > 105.5
        + 1.799755e-05 * max(0.0, 74.51927 - Q.mass_top30) * max(0.0, 0.09605749 - Q.pt2_over_pt0) / 0.006288063   # +0.0%  mass_top30 < 74.52 and pt2_over_pt0 < 0.09606
    )
    return z


def neuron_98(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.067419e-06
    )
    return z


def neuron_99(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.580472e-05
    )
    return z


def neuron_100(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.544017e-05
    )
    return z


def neuron_101(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.024446e-05
    )
    return z


def neuron_102(Q):
    # scale S = 2.637; each line: share * term / its average size
    z = 2.636875 * (0.0470802
        - 0.1277682 * max(0.0, 94.55722 - Q.sd_mass) / 17.47876   # -12.8%  sd_mass < 94.56
        + 0.1064887 * max(0.0, 78.4753 - Q.sd_mass) / 11.39114   # +10.6%  sd_mass < 78.48
        + 0.06136432 * max(0.0, 106.7501 - Q.sd_mass) / 23.58889   # +6.1%  sd_mass < 106.8
        - 0.05581786 * max(0.0, 88.81751 - Q.sd_mass) / 15.02663   # -5.6%  sd_mass < 88.82
        + 0.05208444 * max(0.0, 149.0507 - Q.mass) / 39.08704   # +5.2%  mass < 149.1
        + 0.04545805 * max(0.0, 115.7091 - Q.sd_mass) / 28.95052   # +4.5%  sd_mass < 115.7
        - 0.03880489 * max(0.0, 96.97763 - Q.sj4_pair_mass_max) / 21.11159   # -3.9%  sj4_pair_mass_max < 96.98
        + 0.03555232 * max(0.0, 76.91486 - Q.sj3_pair_mass_max) / 5.293372   # +3.6%  sj3_pair_mass_max < 76.91
        - 0.03198087 * max(0.0, 97.52633 - Q.sj3_pair_mass_max) / 14.66667   # -3.2%  sj3_pair_mass_max < 97.53
        - 0.02774257 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) / 103.3282   # -2.8%  n_pairs_kt_above_1 < 325
        - 0.02675581 * max(0.0, 0.1526454 - Q.tau2) / 0.07687261   # -2.7%  tau2 < 0.1526
        + 0.02601828 * max(0.0, 131.3917 - Q.mass) / 24.784   # +2.6%  mass < 131.4
        + 0.01958349 * max(0.0, 99.0 - Q.n_pairs_kt_above_3) / 39.22243   # +2.0%  n_pairs_kt_above_3 < 99
        - 0.01951175 * max(0.0, 127.8042 - Q.sd_mass) / 37.77282   # -2.0%  sd_mass < 127.8
        - 0.01873361 * max(0.0, 110.2019 - Q.mass) / 12.0043   # -1.9%  mass < 110.2
        - 0.01820334 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # -1.8%  n_pairs_kt_above_3 < 34
        - 0.01697671 * max(0.0, Q.lnpt_22 - 1.005903) / 0.4993339   # -1.7%  lnpt_22 > 1.006
        + 0.016331 * max(0.0, 0.1209237 - Q.M2_b05) / 0.00591466   # +1.6%  M2_b05 < 0.1209
        + 0.01608734 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) / 7.619143   # +1.6%  n_pairs_kt_above_3 < 41
        - 0.01594058 * max(0.0, 62.03827 - Q.sd_mass) / 7.413901   # -1.6%  sd_mass < 62.04
        - 0.01393428 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # -1.4%  n_dr_0p4_up < 4
        + 0.01313061 * max(0.0, Q.pt1_dr01 - 4.704996) / 7.827878   # +1.3%  pt1_dr01 > 4.705
        + 0.01106177 * max(0.0, 55.06039 - Q.sj4_pair_mass_max) / 2.138873   # +1.1%  sj4_pair_mass_max < 55.06
        - 0.01072674 * max(0.0, Q.n_pt_above_5 - 14.0) / 8.10087   # -1.1%  n_pt_above_5 > 14
        + 0.009642297 * max(0.0, 71.96396 - Q.mass) / 1.934957   # +1.0%  mass < 71.96
        - 0.00931515 * max(0.0, Q.lnpt_0 - 4.46447) / 0.4854576   # -0.9%  lnpt_0 > 4.464
        - 0.008408917 * max(0.0, 79.47361 - Q.mass) / 2.857843   # -0.8%  mass < 79.47
        - 0.008333407 * max(0.0, 56.73313 - Q.sj3_pair_mass_max) / 1.542034   # -0.8%  sj3_pair_mass_max < 56.73
        - 0.008055879 * max(0.0, Q.lnptrel_0 - -1.266303) / 0.1164165   # -0.8%  lnptrel_0 > -1.266
        - 0.006537049 * max(0.0, 33.62116 - Q.sj2_mass1) / 6.452718   # -0.7%  sj2_mass1 < 33.62
        + 0.006208111 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +0.6%  mass < 95.15
        + 0.005856974 * max(0.0, 0.2410564 - Q.sj3_dr_min) / 0.07599059   # +0.6%  sj3_dr_min < 0.2411
        + 0.005850172 * max(0.0, 0.1526454 - Q.tau2) * max(0.0, 0.4951794 - Q.dr_9) / 0.02512966   # +0.6%  tau2 < 0.1526 and dr_9 < 0.4952
        - 0.005831666 * max(0.0, Q.lund_max_lnkt - 2.961008) / 0.9492621   # -0.6%  lund_max_lnkt > 2.961
        + 0.005776835 * max(0.0, 11.07803 - Q.sj3_mass3) / 5.735451   # +0.6%  sj3_mass3 < 11.08
        - 0.005199524 * max(0.0, 90.08945 - Q.mass) * max(0.0, 0.1783885 - Q.z_2nd) / 0.1818843   # -0.5%  mass < 90.09 and z_2nd < 0.1784
        - 0.004940603 * max(0.0, Q.dr01 - 0.2849189) / 0.03340072   # -0.5%  dr01 > 0.2849
        - 0.004657207 * max(0.0, 48.76729 - Q.sj4_pair_mass_max) / 1.362821   # -0.5%  sj4_pair_mass_max < 48.77
        - 0.004550362 * max(0.0, 0.1148075 - Q.dr_32) / 0.04792074   # -0.5%  dr_32 < 0.1148
        + 0.004463627 * max(0.0, 0.008254376 - Q.z_dr_0p05_0p1) / 0.0009507355   # +0.4%  z_dr_0p05_0p1 < 0.008254
        - 0.004363514 * max(0.0, 11.07803 - Q.sj3_mass3) * max(0.0, 4.0 - Q.n_dr_0p05_0p1) / 5.513278   # -0.4%  sj3_mass3 < 11.08 and n_dr_0p05_0p1 < 4
        - 0.003936557 * max(0.0, 0.2949917 - Q.tau21) / 0.02817973   # -0.4%  tau21 < 0.295
        - 0.003661187 * max(0.0, 11.07803 - Q.sj3_mass3) * max(0.0, Q.sj3_dr13 - 0.4088773) / 0.6139838   # -0.4%  sj3_mass3 < 11.08 and sj3_dr13 > 0.4089
        + 0.003518354 * max(0.0, 0.00391037 - Q.ecf_g31) / 0.0003411367   # +0.4%  ecf_g31 < 0.00391
        - 0.003435933 * max(0.0, 101.0 - Q.n_pairs_kt_above_1) / 8.83225   # -0.3%  n_pairs_kt_above_1 < 101
        + 0.003290266 * max(0.0, 97.52633 - Q.sj3_pair_mass_max) * max(0.0, 0.01312409 - Q.M3_b2) / 0.02241546   # +0.3%  sj3_pair_mass_max < 97.53 and M3_b2 < 0.01312
        + 0.003265037 * max(0.0, 97.52633 - Q.sj3_pair_mass_max) * max(0.0, 4.855929 - Q.lne_1) / 4.405418   # +0.3%  sj3_pair_mass_max < 97.53 and lne_1 < 4.856
        - 0.003183395 * max(0.0, 79.47361 - Q.mass) * max(0.0, 14.0 - Q.n_pt_above_10) / 12.46528   # -0.3%  mass < 79.47 and n_pt_above_10 < 14
        - 0.003053832 * max(0.0, 0.0815014 - Q.M2_b05) / 0.001014705   # -0.3%  M2_b05 < 0.0815
        + 0.002944484 * max(0.0, 0.1242239 - Q.tau21_b2) / 0.009482157   # +0.3%  tau21_b2 < 0.1242
        + 0.002905493 * max(0.0, 0.3886647 - Q.sj3_mass3) / 0.03562942   # +0.3%  sj3_mass3 < 0.3887
        + 0.002746267 * max(0.0, Q.lund_max_lnkt - 2.961008) * max(0.0, Q.tau32 - 0.731235) / 0.04243374   # +0.3%  lund_max_lnkt > 2.961 and tau32 > 0.7312
        + 0.002667344 * max(0.0, 0.03465331 - Q.M2) / 0.0004071737   # +0.3%  M2 < 0.03465
        - 0.002608186 * max(0.0, Q.lnpt_0 - 4.46447) * max(0.0, Q.dr01 - 0.3258728) / 0.01220346   # -0.3%  lnpt_0 > 4.464 and dr01 > 0.3259
        - 0.002494317 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.2%  mass < 90.09
        + 0.001993828 * max(0.0, Q.sj3_pair_mass_min - 59.49644) / 2.657693   # +0.2%  sj3_pair_mass_min > 59.5
        + 0.001815833 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # +0.2%  sj3_pair_mass_max < 73.25
        + 0.001814583 * max(0.0, 110.2019 - Q.mass) * max(0.0, Q.n_dr_0p2_0p4 - 11.0) / 4.10954   # +0.2%  mass < 110.2 and n_dr_0p2_0p4 > 11
        + 0.001808609 * max(0.0, 0.1242239 - Q.tau21_b2) * max(0.0, 0.4372584 - Q.sj3_dr_max) / 0.0003839797   # +0.2%  tau21_b2 < 0.1242 and sj3_dr_max < 0.4373
        + 0.00151834 * max(0.0, 0.2109554 - Q.sj3_dr23) / 0.01697534   # +0.2%  sj3_dr23 < 0.211
        + 0.00134093 * max(0.0, 99.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.eccentricity - 0.8461094) / 1.80147   # +0.1%  n_pairs_kt_above_3 < 99 and eccentricity > 0.8461
        + 0.00132419 * max(0.0, 0.01744666 - Q.mratio_min_012) / 0.0006584817   # +0.1%  mratio_min_012 < 0.01745
        - 0.001164642 * max(0.0, Q.dr_1 - 0.1975394) / 0.01806617   # -0.1%  dr_1 > 0.1975
        - 0.001051961 * max(0.0, 0.1148075 - Q.dr_32) * max(0.0, 13.0 - Q.n_pt_above_10) / 0.1331669   # -0.1%  dr_32 < 0.1148 and n_pt_above_10 < 13
        + 0.0009548741 * max(0.0, 0.1526454 - Q.tau2) * max(0.0, 0.8351741 - Q.sj3_pairmax_over_m) / 0.002859553   # +0.1%  tau2 < 0.1526 and sj3_pairmax_over_m < 0.8352
        - 0.0008806673 * max(0.0, 90.08945 - Q.mass) * max(0.0, Q.lund2_lnz - -4.435851) / 3.638158   # -0.1%  mass < 90.09 and lund2_lnz > -4.436
        + 0.0008257913 * max(0.0, Q.pair_mean_lndelta - -1.352792) / 0.01174331   # +0.1%  pair_mean_lndelta > -1.353
        + 0.0006474733 * max(0.0, 0.1526454 - Q.tau2) * max(0.0, Q.lnerel_67 - -18.42068) / 0.01301567   # +0.1%  tau2 < 0.1526 and lnerel_67 > -18.42
        - 0.000576928 * max(0.0, 0.5068038 - Q.tau32) * max(0.0, Q.pt1_dr01 - 29.53862) / 0.04928999   # -0.1%  tau32 < 0.5068 and pt1_dr01 > 29.54
        + 0.0005517699 * max(0.0, 76.91486 - Q.sj3_pair_mass_max) * max(0.0, 0.6550361 - Q.tau32_b2) / 0.6690683   # +0.1%  sj3_pair_mass_max < 76.91 and tau32_b2 < 0.655
        - 0.000549469 * max(0.0, 4.0 - Q.n_dr_0p4_up) * max(0.0, Q.min_pair_mass - 10.59762) / 0.4203529   # -0.1%  n_dr_0p4_up < 4 and min_pair_mass > 10.6
        + 0.0004991724 * max(0.0, Q.lnpt_0 - 4.46447) * max(0.0, 4.0 - Q.n_dr_0p05_0p1) / 0.5415633   # +0.0%  lnpt_0 > 4.464 and n_dr_0p05_0p1 < 4
        - 0.0004739558 * max(0.0, 0.5068038 - Q.tau32) / 0.02583179   # -0.0%  tau32 < 0.5068
        - 0.0004199906 * max(0.0, 110.2019 - Q.mass) * max(0.0, 0.7107791 - Q.D2) / 0.04959533   # -0.0%  mass < 110.2 and D2 < 0.7108
        - 0.0003937958 * max(0.0, 48.76729 - Q.sj4_pair_mass_max) * max(0.0, Q.z_dr_0p05_0p1 - 0.4355562) / 0.02753135   # -0.0%  sj4_pair_mass_max < 48.77 and z_dr_0p05_0p1 > 0.4356
        - 0.0003445865 * max(0.0, 0.3886647 - Q.sj3_mass3) * max(0.0, 0.731235 - Q.tau32) / 0.002402595   # -0.0%  sj3_mass3 < 0.3887 and tau32 < 0.7312
        - 0.0003361397 * max(0.0, 0.1242239 - Q.tau21_b2) * max(0.0, Q.n_pt_above_10 - 13.0) / 0.00954065   # -0.0%  tau21_b2 < 0.1242 and n_pt_above_10 > 13
        - 0.0002538134 * max(0.0, 0.2410564 - Q.sj3_dr_min) * max(0.0, -6.585249 - Q.lnerel_27) / 0.3361337   # -0.0%  sj3_dr_min < 0.2411 and lnerel_27 < -6.585
        - 0.0002383873 * max(0.0, 0.1526454 - Q.tau2) * max(0.0, Q.sj4_zsoft - 0.03350185) / 0.001889471   # -0.0%  tau2 < 0.1526 and sj4_zsoft > 0.0335
        + 0.0001688962 * max(0.0, Q.pair_mean_lnm2 - 4.069088) / 0.0918484   # +0.0%  pair_mean_lnm2 > 4.069
        + 0.0001128523 * max(0.0, 0.3886647 - Q.sj3_mass3) * max(0.0, Q.sj4_pair_mass_min - 4.004863) / 0.2345587   # +0.0%  sj3_mass3 < 0.3887 and sj4_pair_mass_min > 4.005
        - 8.132437e-05 * max(0.0, 94.55722 - Q.sd_mass) * max(0.0, Q.pt1_dr01 - 2.525555) / 61.33814   # -0.0%  sd_mass < 94.56 and pt1_dr01 > 2.526
        + 4.796575e-05 * max(0.0, 0.5068038 - Q.tau32) * max(0.0, Q.sj3_dr13 - 0.7462286) / 0.0001552012   # +0.0%  tau32 < 0.5068 and sj3_dr13 > 0.7462
        - 3.262863e-05 * max(0.0, 0.1242239 - Q.tau21_b2) * max(0.0, -0.3178711 - Q.eta_26) / 2.94369e-05   # -0.0%  tau21_b2 < 0.1242 and eta_26 < -0.3179
        - 1.715918e-05 * max(0.0, 0.03465331 - Q.M2) * max(0.0, Q.n_dr_0p05_0p1 - 16.0) / 8.959816e-07   # -0.0%  M2 < 0.03465 and n_dr_0p05_0p1 > 16
    )
    return z


def neuron_103(Q):
    # scale S = 8.145; each line: share * term / its average size
    z = 8.145164 * (0.1980581
        - 0.07459595 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -7.5%  mass < 164.4
        + 0.0639897 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) / 44.19386   # +6.4%  sj3_pair_mass_min < 80.03
        + 0.06260014 * max(0.0, 0.05613495 - Q.tau5) / 0.02517241   # +6.3%  tau5 < 0.05613
        - 0.06229388 * max(0.0, 68.11898 - Q.sj3_pair_mass_min) / 33.1687   # -6.2%  sj3_pair_mass_min < 68.12
        - 0.05723612 * Q.lne_1 / 4.588918   # -5.7%  lne_1
        - 0.05696974 * max(0.0, 164.4374 - Q.mass) * max(0.0, 0.1204509 - Q.C2_b2) / 3.410356   # -5.7%  mass < 164.4 and C2_b2 < 0.1205
        - 0.0436983 * max(0.0, 161.1264 - Q.mass_top50) / 50.68933   # -4.4%  mass_top50 < 161.1
        - 0.03549314 * max(0.0, 0.00228569 - Q.e3) / 0.001253813   # -3.5%  e3 < 0.002286
        - 0.03372864 * max(0.0, 0.09733903 - Q.tau2) / 0.03188557   # -3.4%  tau2 < 0.09734
        - 0.03254574 * max(0.0, 0.2243273 - Q.N2_b2) / 0.06973244   # -3.3%  N2_b2 < 0.2243
        + 0.02607384 * max(0.0, 0.09209404 - Q.C2_b2) / 0.0375889   # +2.6%  C2_b2 < 0.09209
        - 0.02373514 * max(0.0, 0.2159556 - Q.dr02) / 0.1079925   # -2.4%  dr02 < 0.216
        + 0.0182822 * max(0.0, 92.68149 - Q.mass_top40) / 5.990263   # +1.8%  mass_top40 < 92.68
        - 0.0177973 * max(0.0, Q.z_top5 - 0.5239396) / 0.1012136   # -1.8%  z_top5 > 0.5239
        - 0.01706186 * max(0.0, Q.n_pairs_kt_above_3 - 54.0) / 39.6507   # -1.7%  n_pairs_kt_above_3 > 54
        + 0.01637969 * max(0.0, Q.pair_mean_lndelta - -2.336539) / 0.3176941   # +1.6%  pair_mean_lndelta > -2.337
        + 0.01580227 * max(0.0, 223.0 - Q.n_pairs_kt_above_1) / 49.27858   # +1.6%  n_pairs_kt_above_1 < 223
        + 0.01517829 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.4974172 - Q.dr01) / 3.417565   # +1.5%  n_pairs_kt_above_3 < 47 and dr01 < 0.4974
        + 0.01415146 * max(0.0, Q.sd_mass - 123.2919) / 8.015693   # +1.4%  sd_mass > 123.3
        - 0.01321808 * max(0.0, 0.1573433 - Q.M2_b05) / 0.02356419   # -1.3%  M2_b05 < 0.1573
        - 0.01138813 * max(0.0, 0.03469679 - Q.z_dr_0p4_up) / 0.0150329   # -1.1%  z_dr_0p4_up < 0.0347
        + 0.01099097 * max(0.0, 0.09733903 - Q.tau2) * max(0.0, 11.0 - Q.n_pairs_kt_above_10) / 0.2199417   # +1.1%  tau2 < 0.09734 and n_pairs_kt_above_10 < 11
        - 0.01083119 * max(0.0, 0.04229114 - Q.C3_b2) / 0.03492918   # -1.1%  C3_b2 < 0.04229
        - 0.01079761 * max(0.0, 97.12186 - Q.mass_top40) / 7.43888   # -1.1%  mass_top40 < 97.12
        + 0.01070362 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.pair_max_lnkt - 1.555555) / 9.524956   # +1.1%  n_pairs_kt_above_3 < 47 and pair_max_lnkt > 1.556
        - 0.01067327 * max(0.0, 5.0 - Q.n_dr_0p4_up) / 1.758997   # -1.1%  n_dr_0p4_up < 5
        - 0.01039868 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # -1.0%  sd_mass > 154.6
        - 0.01011395 * max(0.0, Q.lund_max_lndelta - -1.388411) / 0.3966091   # -1.0%  lund_max_lndelta > -1.388
        + 0.008833147 * max(0.0, 0.07373689 - Q.dr02) / 0.02487709   # +0.9%  dr02 < 0.07374
        + 0.00865907 * max(0.0, 97.12186 - Q.mass_top40) * max(0.0, 0.1049877 - Q.C2_b2) / 0.3749185   # +0.9%  mass_top40 < 97.12 and C2_b2 < 0.105
        - 0.008478682 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # -0.8%  n_pairs_kt_above_3 < 34
        + 0.008408034 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) * max(0.0, 0.1975394 - Q.dr_1) / 3.944513   # +0.8%  sj3_pair_mass_min < 80.03 and dr_1 < 0.1975
        + 0.007674605 * max(0.0, 96.97763 - Q.sj4_pair_mass_max) / 21.11159   # +0.8%  sj4_pair_mass_max < 96.98
        + 0.007148734 * max(0.0, 0.03469679 - Q.z_dr_0p4_up) * max(0.0, 0.1049877 - Q.C2_b2) / 0.0009997393   # +0.7%  z_dr_0p4_up < 0.0347 and C2_b2 < 0.105
        - 0.006534076 * max(0.0, 0.1798005 - Q.sj3_dr_min) / 0.04059535   # -0.7%  sj3_dr_min < 0.1798
        + 0.006357161 * max(0.0, 0.1702808 - Q.dr_2) / 0.05898194   # +0.6%  dr_2 < 0.1703
        - 0.006312013 * max(0.0, 29.68311 - Q.mass_top5) / 6.003218   # -0.6%  mass_top5 < 29.68
        + 0.006198262 * max(0.0, 0.2159556 - Q.dr02) * max(0.0, Q.pt1_over_pt0 - 0.2477181) / 0.04429267   # +0.6%  dr02 < 0.216 and pt1_over_pt0 > 0.2477
        + 0.006075848 * max(0.0, Q.m01 - 8.773939) / 10.11753   # +0.6%  m01 > 8.774
        + 0.005746464 * max(0.0, 0.3436326 - Q.N2_b05) / 0.005384222   # +0.6%  N2_b05 < 0.3436
        - 0.005529002 * max(0.0, Q.pair_mean_lndelta - -2.336539) * max(0.0, Q.pt2_over_pt0 - 0.1526736) / 0.09008409   # -0.6%  pair_mean_lndelta > -2.337 and pt2_over_pt0 > 0.1527
        - 0.005501592 * max(0.0, Q.lnptrel_29 - -5.376631) / 0.2012241   # -0.6%  lnptrel_29 > -5.377
        - 0.005497186 * max(0.0, Q.psi_0p1 - 0.7386202) / 0.03082667   # -0.5%  psi_0p1 > 0.7386
        - 0.004969831 * max(0.0, 0.8860453 - Q.D3_b05) / 0.4670665   # -0.5%  D3_b05 < 0.886
        + 0.004915606 * max(0.0, Q.n_pairs_kt_above_3 - 144.0) / 11.30914   # +0.5%  n_pairs_kt_above_3 > 144
        + 0.004698278 * max(0.0, 121.0623 - Q.mass_top30) / 23.2224   # +0.5%  mass_top30 < 121.1
        - 0.004353332 * max(0.0, 1.488114 - Q.pair_mean_lnm2) / 0.08573017   # -0.4%  pair_mean_lnm2 < 1.488
        + 0.004207361 * max(0.0, Q.lund_max_lndelta - -0.752531) / 0.05045858   # +0.4%  lund_max_lndelta > -0.7525
        + 0.00412155 * max(0.0, 0.1024943 - Q.dr_0) / 0.02228225   # +0.4%  dr_0 < 0.1025
        + 0.004097617 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +0.4%  n_pairs_kt_above_1 < 80
        + 0.004064787 * max(0.0, Q.m01 - 8.773939) * max(0.0, 0.2586201 - Q.dr_1) / 0.6790036   # +0.4%  m01 > 8.774 and dr_1 < 0.2586
        - 0.004030914 * max(0.0, 0.2243273 - Q.N2_b2) * max(0.0, Q.lne_1 - 4.137663) / 0.03674614   # -0.4%  N2_b2 < 0.2243 and lne_1 > 4.138
        + 0.00389079 * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.3730633   # +0.4%  n_dr_0p1_0p2 < 4
        + 0.003720723 * max(0.0, 0.1702808 - Q.dr_2) * max(0.0, 13.10191 - Q.sj3_mass3) / 0.4818677   # +0.4%  dr_2 < 0.1703 and sj3_mass3 < 13.1
        + 0.003601223 * max(0.0, 0.03469679 - Q.z_dr_0p4_up) * max(0.0, Q.D2 - 0.7107791) / 0.01849723   # +0.4%  z_dr_0p4_up < 0.0347 and D2 > 0.7108
        - 0.003581385 * max(0.0, Q.sj3_pairmax_over_m - 0.8809196) / 0.01122806   # -0.4%  sj3_pairmax_over_m > 0.8809
        + 0.003448917 * max(0.0, 0.2243273 - Q.N2_b2) * max(0.0, Q.n_lund - 9.0) / 0.13782   # +0.3%  N2_b2 < 0.2243 and n_lund > 9
        - 0.003112432 * max(0.0, Q.psi_0p3 - 0.9925964) / 0.0007268581   # -0.3%  psi_0p3 > 0.9926
        - 0.002731484 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.ecf_g42 - 2.701529e-06) / 4.668886e-05   # -0.3%  n_pairs_kt_above_3 < 34 and ecf_g42 > 2.702e-06
        - 0.002611605 * max(0.0, 0.1573433 - Q.M2_b05) * max(0.0, Q.n_pairs_kt_above_10 - 3.0) / 0.08549489   # -0.3%  M2_b05 < 0.1573 and n_pairs_kt_above_10 > 3
        + 0.00257095 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # +0.3%  sd_mass > 175.9
        + 0.00253175 * max(0.0, 0.01545532 - Q.psi_0p1) / 0.001647647   # +0.3%  psi_0p1 < 0.01546
        - 0.002198662 * max(0.0, 29.68311 - Q.mass_top5) * max(0.0, 0.09246155 - Q.dr_34) / 0.2595826   # -0.2%  mass_top5 < 29.68 and dr_34 < 0.09246
        + 0.002176766 * max(0.0, 0.05865627 - Q.sj4_dr_min) / 0.005184347   # +0.2%  sj4_dr_min < 0.05866
        - 0.00193984 * max(0.0, Q.pair_mean_lnkt - 1.030977) / 0.04518922   # -0.2%  pair_mean_lnkt > 1.031
        - 0.001809891 * max(0.0, 0.2243273 - Q.N2_b2) * max(0.0, 0.1455861 - Q.dr_38) / 0.006843614   # -0.2%  N2_b2 < 0.2243 and dr_38 < 0.1456
        - 0.001715096 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, 0.120439 - Q.M2) / 2.252724   # -0.2%  mass_top50 < 161.1 and M2 < 0.1204
        + 0.001713967 * max(0.0, 0.2243273 - Q.N2_b2) * max(0.0, 1.650776e-05 - Q.ecf_g42) / 3.461705e-07   # +0.2%  N2_b2 < 0.2243 and ecf_g42 < 1.651e-05
        - 0.001663508 * max(0.0, 0.01937651 - Q.dr12) / 0.002668782   # -0.2%  dr12 < 0.01938
        + 0.001411528 * max(0.0, 0.2159556 - Q.dr02) * max(0.0, Q.ecf_g41 - 6.216954e-05) / 1.321766e-05   # +0.1%  dr02 < 0.216 and ecf_g41 > 6.217e-05
        - 0.001403327 * max(0.0, 0.03469679 - Q.z_dr_0p4_up) * max(0.0, Q.lund2_lndelta - -0.8575264) / 0.001014848   # -0.1%  z_dr_0p4_up < 0.0347 and lund2_lndelta > -0.8575
        - 0.001398026 * max(0.0, 11.0 - Q.n_for_90pct) / 0.60437   # -0.1%  n_for_90pct < 11
        - 0.001392948 * max(0.0, Q.m01 - 8.773939) * max(0.0, 0.02349873 - Q.C2_b2) / 0.03626023   # -0.1%  m01 > 8.774 and C2_b2 < 0.0235
        - 0.001339431 * max(0.0, 0.09209404 - Q.C2_b2) * max(0.0, Q.mass_top5 - 62.84493) / 0.1399485   # -0.1%  C2_b2 < 0.09209 and mass_top5 > 62.84
        - 0.001326594 * max(0.0, 0.03469679 - Q.z_dr_0p4_up) * max(0.0, Q.C2_b2 - 0.06190784) / 0.0003610303   # -0.1%  z_dr_0p4_up < 0.0347 and C2_b2 > 0.06191
        + 0.001309844 * max(0.0, 5.0 - Q.n_dr_0p4_up) * max(0.0, -3.197773 - Q.lund2_lnz) / 1.849372   # +0.1%  n_dr_0p4_up < 5 and lund2_lnz < -3.198
        + 0.001193048 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.ecf_g41 - 8.500146e-05) / 0.0003863456   # +0.1%  n_pairs_kt_above_3 < 34 and ecf_g41 > 8.5e-05
        - 0.001141821 * max(0.0, Q.jet_abs_eta - 1.465946) / 0.02086851   # -0.1%  jet_abs_eta > 1.466
        + 0.00112064 * max(0.0, Q.n_pairs_kt_above_3 - 54.0) * max(0.0, 0.1204509 - Q.C2_b2) / 1.557801   # +0.1%  n_pairs_kt_above_3 > 54 and C2_b2 < 0.1205
        + 0.001067782 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, Q.sj3_dr23 - 0.3971678) / 4.1992   # +0.1%  mass_top50 < 161.1 and sj3_dr23 > 0.3972
        - 0.001039804 * max(0.0, 0.001184159 - Q.D3_b2) / 0.0001406867   # -0.1%  D3_b2 < 0.001184
        - 0.001039378 * max(0.0, 3.298199 - Q.sj3_mass2) / 0.1692566   # -0.1%  sj3_mass2 < 3.298
        + 0.0009109308 * max(0.0, Q.N2_b2 - 0.255112) / 0.008335535   # +0.1%  N2_b2 > 0.2551
        + 0.0007885298 * max(0.0, Q.mass_top5 - 58.28268) / 4.708762   # +0.1%  mass_top5 > 58.28
        + 0.0007307956 * max(0.0, 0.07373689 - Q.dr02) * max(0.0, 4.026467 - Q.lnpt_4) / 0.01315322   # +0.1%  dr02 < 0.07374 and lnpt_4 < 4.026
        - 0.0007226307 * max(0.0, 0.05613495 - Q.tau5) * max(0.0, Q.lam2 - 0.00972048) / 1.460067e-05   # -0.1%  tau5 < 0.05613 and lam2 > 0.00972
        - 0.0007151254 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) / 9.886423   # -0.1%  n_pairs_kt_above_3 < 47
        - 0.0007127749 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) * max(0.0, Q.dr_4 - 0.06996917) / 3.600404   # -0.1%  sj3_pair_mass_min < 80.03 and dr_4 > 0.06997
        + 0.0006818981 * max(0.0, 0.2243273 - Q.N2_b2) * max(0.0, Q.centroid_offset - 0.007708221) / 0.0003089815   # +0.1%  N2_b2 < 0.2243 and centroid_offset > 0.007708
        - 0.0006625225 * max(0.0, Q.sd_mass - 175.9333) * max(0.0, Q.tau54 - 0.6772033) / 0.2894805   # -0.1%  sd_mass > 175.9 and tau54 > 0.6772
        - 0.0005821469 * max(0.0, Q.C2_b05 - 0.3007515) / 0.01381455   # -0.1%  C2_b05 > 0.3008
        - 0.0005394585 * max(0.0, 0.2159556 - Q.dr02) * max(0.0, Q.dr_3 - 0.02154917) / 0.01148768   # -0.1%  dr02 < 0.216 and dr_3 > 0.02155
        - 0.0005175709 * max(0.0, Q.max_dr - 0.9206713) / 0.06375758   # -0.1%  max_dr > 0.9207
        + 0.0004550479 * max(0.0, Q.sd_mass - 123.2919) * max(0.0, 1.373172e-06 - Q.e4) / 1.450483e-06   # +0.0%  sd_mass > 123.3 and e4 < 1.373e-06
        - 0.0003822235 * max(0.0, 0.2159556 - Q.dr02) * max(0.0, Q.sj3_dr13 - 0.7462286) / 0.001446736   # -0.0%  dr02 < 0.216 and sj3_dr13 > 0.7462
        - 0.0003651405 * max(0.0, Q.n_pairs_kt_above_3 - 144.0) * max(0.0, Q.tau43 - 0.8357534) / 0.2281763   # -0.0%  n_pairs_kt_above_3 > 144 and tau43 > 0.8358
        + 0.0003044173 * max(0.0, Q.m01 - 8.773939) * max(0.0, 0.2567354 - Q.dr_6) / 0.8673451   # +0.0%  m01 > 8.774 and dr_6 < 0.2567
        + 0.0002904429 * max(0.0, 3.298199 - Q.sj3_mass2) * max(0.0, 0.2317874 - Q.sj3_dr13) / 0.00630746   # +0.0%  sj3_mass2 < 3.298 and sj3_dr13 < 0.2318
        - 0.0002342965 * max(0.0, 0.1573433 - Q.M2_b05) * max(0.0, 0.2977652 - Q.dr_25) / 0.004179043   # -0.0%  M2_b05 < 0.1573 and dr_25 < 0.2978
        - 5.88328e-05 * max(0.0, 0.04229114 - Q.C3_b2) * max(0.0, Q.ecf_g42 - 1.874473e-05) / 3.656487e-07   # -0.0%  C3_b2 < 0.04229 and ecf_g42 > 1.874e-05
    )
    return z


def neuron_104(Q):
    # scale S = 6.675; each line: share * term / its average size
    z = 6.675024 * (-0.09091175
        + 0.1380093 * max(0.0, 164.4374 - Q.mass) / 52.54489   # +13.8%  mass < 164.4
        + 0.08489402 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # +8.5%  sd_mass > 78.48
        - 0.07265519 * max(0.0, 182.8592 - Q.mass) / 69.61635   # -7.3%  mass < 182.9
        - 0.07041478 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # -7.0%  sd_mass > 94.56
        + 0.06545837 * max(0.0, Q.mass - 56.49019) / 59.27462   # +6.5%  mass > 56.49
        - 0.04734218 * max(0.0, Q.mass_top50 - 89.68964) / 28.47939   # -4.7%  mass_top50 > 89.69
        - 0.03510365 * max(0.0, Q.sd_mass - 62.03827) / 42.41619   # -3.5%  sd_mass > 62.04
        + 0.0311042 * max(0.0, 0.1335998 - Q.dr01) / 0.06067654   # +3.1%  dr01 < 0.1336
        - 0.02820627 * max(0.0, 0.4349199 - Q.dr02) / 0.2787527   # -2.8%  dr02 < 0.4349
        - 0.02604958 * max(0.0, 0.3701694 - Q.dr01) / 0.2308018   # -2.6%  dr01 < 0.3702
        + 0.02087601 * max(0.0, 0.1058963 - Q.dr02) / 0.04190221   # +2.1%  dr02 < 0.1059
        - 0.01902772 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # -1.9%  n_pairs_kt_above_1 < 366
        + 0.01674564 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # +1.7%  sd_mass > 106.8
        - 0.01638238 * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0001737301   # -1.6%  e3_b2 < 0.0002537
        + 0.01604578 * max(0.0, Q.N2_b05 - 0.302405) * max(0.0, 33.1786 - Q.sj2_mass2) / 2.327225   # +1.6%  N2_b05 > 0.3024 and sj2_mass2 < 33.18
        + 0.01575292 * max(0.0, Q.mass_top50 - 79.27954) / 36.81741   # +1.6%  mass_top50 > 79.28
        + 0.01561699 * max(0.0, Q.pt_entropy - 2.768601) / 0.2459887   # +1.6%  pt_entropy > 2.769
        - 0.01467053 * max(0.0, 0.9212656 - Q.z_top15_slots) / 0.09473096   # -1.5%  z_top15_slots < 0.9213
        + 0.01378254 * max(0.0, Q.N2_b05 - 0.302405) / 0.1383194   # +1.4%  N2_b05 > 0.3024
        + 0.01317949 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, 8.0 - Q.n_lund_kt_above_1) / 0.0005145216   # +1.3%  e3_b2 < 0.0002537 and n_lund_kt_above_1 < 8
        + 0.01123874 * max(0.0, 0.2901133 - Q.sj3_dr_min) * max(0.0, Q.lnpt_4 - 2.976008) / 0.06093991   # +1.1%  sj3_dr_min < 0.2901 and lnpt_4 > 2.976
        - 0.01062273 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.lnpt_34 - 0.05231816) / 19.2026   # -1.1%  mass < 164.4 and lnpt_34 > 0.05232
        + 0.009464737 * max(0.0, 0.2052214 - Q.N2) / 0.006739882   # +0.9%  N2 < 0.2052
        - 0.009380432 * max(0.0, 0.3961587 - Q.N2_b05) / 0.01408272   # -0.9%  N2_b05 < 0.3962
        + 0.008915142 * max(0.0, 117.4867 - Q.mass) / 15.64803   # +0.9%  mass < 117.5
        + 0.00754359 * max(0.0, 0.2901133 - Q.sj3_dr_min) / 0.1104413   # +0.8%  sj3_dr_min < 0.2901
        - 0.007450717 * max(0.0, 2.340672 - Q.pair_mean_lnm2) / 0.3294169   # -0.7%  pair_mean_lnm2 < 2.341
        - 0.00728931 * max(0.0, 2.151069 - Q.sj3_mass3) / 0.3356229   # -0.7%  sj3_mass3 < 2.151
        + 0.006659017 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.sj2_dr - 0.2848657) / 13.01035   # +0.7%  n_pairs_kt_above_1 < 366 and sj2_dr > 0.2849
        - 0.006021281 * max(0.0, Q.sj2_mass1 - 60.04967) / 4.606883   # -0.6%  sj2_mass1 > 60.05
        + 0.00578459 * max(0.0, 0.3835105 - Q.tau32_b2) / 0.03443666   # +0.6%  tau32_b2 < 0.3835
        - 0.0057477 * max(0.0, 4.069088 - Q.pair_mean_lnm2) / 1.587861   # -0.6%  pair_mean_lnm2 < 4.069
        - 0.005681332 * max(0.0, 0.1058963 - Q.dr02) * max(0.0, 0.9376224 - Q.planar_flow) / 0.01921669   # -0.6%  dr02 < 0.1059 and planar_flow < 0.9376
        + 0.00550756 * max(0.0, Q.mass_top50 - 79.27954) * max(0.0, 0.7925455 - Q.sj3_pairmax_over_m) / 1.371316   # +0.6%  mass_top50 > 79.28 and sj3_pairmax_over_m < 0.7925
        + 0.005424635 * max(0.0, Q.sj2_mass1 - 77.42768) / 2.022907   # +0.5%  sj2_mass1 > 77.43
        + 0.005310964 * max(0.0, 0.00761379 - Q.M3_b2) / 0.0008114698   # +0.5%  M3_b2 < 0.007614
        - 0.005226023 * max(0.0, 0.003262024 - Q.e3_b05) / 0.0001880176   # -0.5%  e3_b05 < 0.003262
        - 0.005221105 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) / 59.61773   # -0.5%  n_pairs_kt_above_3 < 126
        + 0.004923644 * max(0.0, 0.02858644 - Q.sj3_z3) / 0.002115303   # +0.5%  sj3_z3 < 0.02859
        + 0.004883607 * max(0.0, 2.151069 - Q.sj3_mass3) * max(0.0, 1.878367e-05 - Q.ecf_g43) / 5.211002e-06   # +0.5%  sj3_mass3 < 2.151 and ecf_g43 < 1.878e-05
        - 0.004755712 * max(0.0, 131.3917 - Q.mass) / 24.784   # -0.5%  mass < 131.4
        - 0.004613959 * max(0.0, Q.sj3_pair_mass_min - 40.45675) / 7.213936   # -0.5%  sj3_pair_mass_min > 40.46
        - 0.004609608 * max(0.0, Q.pair_max_lnkt - 2.837304) * max(0.0, 0.3246647 - Q.sj3_z2) / 0.007368465   # -0.5%  pair_max_lnkt > 2.837 and sj3_z2 < 0.3247
        + 0.004608561 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # +0.5%  sj3_pair_mass_max < 73.25
        + 0.00430262 * max(0.0, 0.04544298 - Q.sum_z_dr2_top5) / 0.0242958   # +0.4%  sum_z_dr2_top5 < 0.04544
        - 0.004036391 * max(0.0, Q.psi_0p2 - 0.6958425) / 0.1174907   # -0.4%  psi_0p2 > 0.6958
        - 0.004033425 * max(0.0, Q.sj4_pair_mass_max - 88.5845) / 7.683986   # -0.4%  sj4_pair_mass_max > 88.58
        + 0.004020085 * max(0.0, Q.mass_top50 - 79.27954) * max(0.0, 5.423297 - Q.lne_0) / 17.05536   # +0.4%  mass_top50 > 79.28 and lne_0 < 5.423
        + 0.003731876 * max(0.0, 0.1058963 - Q.dr02) * max(0.0, Q.mratio_max_012 - 0.7525667) / 0.001895177   # +0.4%  dr02 < 0.1059 and mratio_max_012 > 0.7526
        + 0.003727187 * max(0.0, Q.z_top5 - 0.7768561) / 0.01136195   # +0.4%  z_top5 > 0.7769
        - 0.003603051 * max(0.0, 0.6206221 - Q.tau32) / 0.05684264   # -0.4%  tau32 < 0.6206
        + 0.003305236 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +0.3%  sd_mass > 154.6
        + 0.003202845 * max(0.0, Q.mass_over_sum_pt - 0.2478639) / 0.00727487   # +0.3%  mass_over_sum_pt > 0.2479
        - 0.002913089 * max(0.0, 8.0 - Q.n_lund) / 0.3988733   # -0.3%  n_lund < 8
        + 0.002851755 * max(0.0, Q.mass_top40 - 132.5189) / 6.082227   # +0.3%  mass_top40 > 132.5
        - 0.002355411 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # -0.2%  mass_top10 > 103.5
        + 0.002131887 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # +0.2%  sd_mass > 175.9
        - 0.002034951 * max(0.0, 0.01851479 - Q.sum_z_dr2_top5) / 0.005275115   # -0.2%  sum_z_dr2_top5 < 0.01851
        + 0.002028597 * max(0.0, 21.76123 - Q.m01) / 11.6217   # +0.2%  m01 < 21.76
        - 0.002023902 * max(0.0, Q.psi_0p1 - 0.7386202) / 0.03082667   # -0.2%  psi_0p1 > 0.7386
        - 0.002016424 * max(0.0, Q.sj2_mass1 - 91.2852) / 1.019309   # -0.2%  sj2_mass1 > 91.29
        - 0.001907666 * max(0.0, Q.mass - 56.49019) * max(0.0, 4.039064 - Q.lund3_lnkt) / 169.3052   # -0.2%  mass > 56.49 and lund3_lnkt < 4.039
        + 0.001891619 * max(0.0, 0.01602882 - Q.dr12) / 0.001741445   # +0.2%  dr12 < 0.01603
        - 0.00186683 * max(0.0, 0.07110203 - Q.sj3_dr_min) / 0.003448912   # -0.2%  sj3_dr_min < 0.0711
        + 0.001793748 * max(0.0, Q.psi_0p1 - 0.9356675) / 0.00150838   # +0.2%  psi_0p1 > 0.9357
        - 0.001718343 * max(0.0, Q.mass_top50 - 79.27954) * max(0.0, 0.6363796 - Q.tau43) / 0.4543033   # -0.2%  mass_top50 > 79.28 and tau43 < 0.6364
        + 0.001654429 * max(0.0, 0.3016348 - Q.LHA) / 0.01068231   # +0.2%  LHA < 0.3016
        - 0.001558388 * max(0.0, 2.151069 - Q.sj3_mass3) * max(0.0, Q.sj3_pairmin_over_m - 0.1073315) / 0.04280576   # -0.2%  sj3_mass3 < 2.151 and sj3_pairmin_over_m > 0.1073
        + 0.001516818 * max(0.0, Q.mass_top10 - 103.4976) * max(0.0, 19.43809 - Q.sj2_mass2) / 10.52663   # +0.2%  mass_top10 > 103.5 and sj2_mass2 < 19.44
        - 0.001318677 * max(0.0, Q.min_pair_mass - 10.59762) / 0.3791806   # -0.1%  min_pair_mass > 10.6
        - 0.001178052 * max(0.0, Q.mass_top50 - 89.68964) * max(0.0, Q.sj3_dr13 - 0.3524911) / 4.781425   # -0.1%  mass_top50 > 89.69 and sj3_dr13 > 0.3525
        - 0.001142988 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.pair_max_lnkt - 1.886287) / 36.76563   # -0.1%  mass < 164.4 and pair_max_lnkt > 1.886
        - 0.001140363 * max(0.0, 0.3835105 - Q.tau32_b2) * max(0.0, Q.n_dr_0p05_0p1 - 0.0) / 0.1610759   # -0.1%  tau32_b2 < 0.3835 and n_dr_0p05_0p1 > 0
        + 0.001109113 * max(0.0, 0.07307944 - Q.z_dr_0p2_0p4) / 0.01707726   # +0.1%  z_dr_0p2_0p4 < 0.07308
        - 0.001109006 * max(0.0, Q.pair_mean_lndelta - -1.352792) / 0.01174331   # -0.1%  pair_mean_lndelta > -1.353
        + 0.001107473 * max(0.0, 0.6206221 - Q.tau32) * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.02588067   # +0.1%  tau32 < 0.6206 and n_dr_0p1_0p2 < 4
        - 0.0009756481 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, 0.5779883 - Q.pt1_over_pt0) / 1.517431e-05   # -0.1%  e3_b2 < 0.0002537 and pt1_over_pt0 < 0.578
        + 0.0007432617 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.n_lund - 10.0) / 0.000281433   # +0.1%  e3_b2 < 0.0002537 and n_lund > 10
        - 0.0006818263 * max(0.0, 0.3961587 - Q.N2_b05) * max(0.0, 0.9206713 - Q.max_dr) / 0.004729071   # -0.1%  N2_b05 < 0.3962 and max_dr < 0.9207
        + 0.0006560601 * max(0.0, 0.3245484 - Q.tau21) / 0.03776915   # +0.1%  tau21 < 0.3245
        + 0.0006415733 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.min_pair_mass - 10.59762) / 10.32656   # +0.1%  mass < 164.4 and min_pair_mass > 10.6
        - 0.0006294445 * max(0.0, 164.4374 - Q.mass) * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 679.4175   # -0.1%  mass < 164.4 and n_dr_0p2_0p4 < 21
        - 0.0004979301 * max(0.0, Q.sd_mass - 135.3031) / 5.717996   # -0.0%  sd_mass > 135.3
        + 0.0004791719 * max(0.0, 0.2403736 - Q.sj2_dr) / 0.01094304   # +0.0%  sj2_dr < 0.2404
        - 0.0003335319 * max(0.0, Q.sj2_dr - 0.6647889) / 0.004152297   # -0.0%  sj2_dr > 0.6648
        - 0.0003065546 * max(0.0, Q.sd_mass - 175.9333) * max(0.0, Q.lnerel_77 - -18.42068) / 3.915188   # -0.0%  sd_mass > 175.9 and lnerel_77 > -18.42
        - 0.0003038617 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) / 9.886423   # -0.0%  n_pairs_kt_above_3 < 47
        - 0.0002835539 * max(0.0, Q.sj3_mass1 - 44.1303) / 0.6338316   # -0.0%  sj3_mass1 > 44.13
        + 0.0002574758 * max(0.0, Q.pair_max_lnkt - 2.837304) / 0.1852268   # +0.0%  pair_max_lnkt > 2.837
        + 0.0002146525 * max(0.0, Q.mass_top50 - 89.68964) * max(0.0, Q.e4 - 7.184834e-06) / 7.091494e-05   # +0.0%  mass_top50 > 89.69 and e4 > 7.185e-06
        + 0.0001318686 * max(0.0, 0.6206221 - Q.tau32) * max(0.0, Q.pt1_dr01 - 29.53862) / 0.09540574   # +0.0%  tau32 < 0.6206 and pt1_dr01 > 29.54
        - 0.0001239862 * max(0.0, 55.06039 - Q.sj4_pair_mass_max) / 2.138873   # -0.0%  sj4_pair_mass_max < 55.06
        - 7.85841e-05 * max(0.0, Q.mass_top10 - 103.4976) * max(0.0, Q.n_lund_kt_above_5 - 2.0) / 0.775389   # -0.0%  mass_top10 > 103.5 and n_lund_kt_above_5 > 2
        - 6.277953e-05 * max(0.0, Q.sj2_dr - 0.6647889) * max(0.0, Q.sj2_mass2 - 42.12131) / 0.002978186   # -0.0%  sj2_dr > 0.6648 and sj2_mass2 > 42.12
        - 3.37329e-05 * max(0.0, 2.151069 - Q.sj3_mass3) * max(0.0, Q.dr_10 - 0.2577145) / 0.01224544   # -0.0%  sj3_mass3 < 2.151 and dr_10 > 0.2577
        - 1.067737e-05 * max(0.0, 3.22013 - Q.lund_max_lnkt) / 0.1084675   # -0.0%  lund_max_lnkt < 3.22
        + 8.201752e-06 * max(0.0, 0.6206221 - Q.tau32) * max(0.0, 1.617171 - Q.sj3_mass3) / 0.006881104   # +0.0%  tau32 < 0.6206 and sj3_mass3 < 1.617
        - 4.976494e-06 * max(0.0, Q.sj2_mass1 - 91.2852) * max(0.0, 0.03336182 - Q.phi_58) / 0.08243346   # -0.0%  sj2_mass1 > 91.29 and phi_58 < 0.03336
        - 4.427761e-06 * max(0.0, Q.pair_max_lnkt - 2.837304) * max(0.0, 0.3420532 - Q.phi_20) / 0.06466735   # -0.0%  pair_max_lnkt > 2.837 and phi_20 < 0.3421
        - 1.756592e-06 * max(0.0, Q.sd_mass - 175.9333) * max(0.0, 9.0 - Q.n_dr_0p1_0p2) / 5.10426   # -0.0%  sd_mass > 175.9 and n_dr_0p1_0p2 < 9
    )
    return z


def neuron_105(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.008285e-06
    )
    return z


def neuron_106(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.010549e-05
    )
    return z


def neuron_107(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.028072e-05
    )
    return z


def neuron_108(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.820155e-07
    )
    return z


def neuron_109(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.850338e-06
    )
    return z


def neuron_110(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.695643e-06
    )
    return z


def neuron_111(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.546122e-06
    )
    return z


def neuron_112(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.208627e-05
    )
    return z


def neuron_113(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.073401e-05
    )
    return z


def neuron_114(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.342204e-05
    )
    return z


def neuron_115(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.963158e-06
    )
    return z


def neuron_116(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.341653e-05
    )
    return z


def neuron_117(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.964242e-06
    )
    return z


def neuron_118(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.141795e-06
    )
    return z


def neuron_119(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.952457e-06
    )
    return z


def neuron_120(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.108244e-06
    )
    return z


def neuron_121(Q):
    # scale S = 10.4; each line: share * term / its average size
    z = 10.40191 * (-0.1179133
        - 0.07380514 * max(0.0, 3.303713 - Q.pt_entropy) / 0.5778306   # -7.4%  pt_entropy < 3.304
        + 0.07185581 * max(0.0, 0.5083429 - Q.N2_b05) / 0.07626804   # +7.2%  N2_b05 < 0.5083
        + 0.06595765 * max(0.0, 598.0 - Q.n_pairs_kt_above_1) / 306.1506   # +6.6%  n_pairs_kt_above_1 < 598
        + 0.0465055 * max(0.0, Q.mass_top40 - 55.75601) / 55.51953   # +4.7%  mass_top40 > 55.76
        - 0.04483765 * max(0.0, 0.3701068 - Q.N2) / 0.07587825   # -4.5%  N2 < 0.3701
        + 0.04452268 * max(0.0, 0.8860453 - Q.D3_b05) / 0.4670665   # +4.5%  D3_b05 < 0.886
        + 0.03788134 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) / 59.61773   # +3.8%  n_pairs_kt_above_3 < 126
        + 0.0341702 * max(0.0, 1.549573 - Q.N3_b05) / 0.8053398   # +3.4%  N3_b05 < 1.55
        - 0.02972319 * max(0.0, 598.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.tau32 - 0.456416) / 74.22184   # -3.0%  n_pairs_kt_above_1 < 598 and tau32 > 0.4564
        + 0.0279294 * max(0.0, Q.M2_b05 - 0.1258758) / 0.01827883   # +2.8%  M2_b05 > 0.1259
        - 0.02598276 * max(0.0, 34.83233 - Q.sj4_pair_mass_min) / 19.46247   # -2.6%  sj4_pair_mass_min < 34.83
        + 0.02366775 * max(0.0, 223.0 - Q.n_pairs_kt_above_1) / 49.27858   # +2.4%  n_pairs_kt_above_1 < 223
        + 0.0232281 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # +2.3%  n_pairs_kt_above_3 < 34
        + 0.02281446 * max(0.0, Q.pt_dispersion - 0.2484767) / 0.09718337   # +2.3%  pt_dispersion > 0.2485
        + 0.02128149 * max(0.0, 3.303713 - Q.pt_entropy) * max(0.0, 10.59762 - Q.min_pair_mass) / 4.769952   # +2.1%  pt_entropy < 3.304 and min_pair_mass < 10.6
        - 0.02079026 * max(0.0, 137.452 - Q.mass) / 29.4899   # -2.1%  mass < 137.5
        - 0.01998706 * max(0.0, 598.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.tau43 - 0.568406) / 70.86467   # -2.0%  n_pairs_kt_above_1 < 598 and tau43 > 0.5684
        - 0.01715191 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.02489714 - Q.sum_z_dr2_top2) / 0.08624095   # -1.7%  n_pairs_kt_above_3 < 34 and sum_z_dr2_top2 < 0.0249
        + 0.01548362 * max(0.0, 36.0 - Q.n_real_top40) / 4.48314   # +1.5%  n_real_top40 < 36
        - 0.01388347 * max(0.0, 0.8415274 - Q.tau32) / 0.1805686   # -1.4%  tau32 < 0.8415
        + 0.01319353 * max(0.0, 107.8099 - Q.sj4_pair_mass_max) / 29.80438   # +1.3%  sj4_pair_mass_max < 107.8
        + 0.01222726 * max(0.0, 3.303713 - Q.pt_entropy) * max(0.0, Q.M2_b2 - 0.01262589) / 0.007744308   # +1.2%  pt_entropy < 3.304 and M2_b2 > 0.01263
        - 0.01210927 * max(0.0, Q.mass - 100.4835) / 22.56274   # -1.2%  mass > 100.5
        - 0.01202669 * max(0.0, 0.4536006 - Q.pt_dispersion) / 0.1342552   # -1.2%  pt_dispersion < 0.4536
        - 0.01180093 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.1620436 - Q.dr02) / 4.573172   # -1.2%  n_pairs_kt_above_3 < 126 and dr02 < 0.162
        - 0.01050094 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) * max(0.0, 1.831581 - Q.D2) / 22.19502   # -1.1%  n_pairs_kt_above_3 < 126 and D2 < 1.832
        + 0.01048695 * max(0.0, 23.0 - Q.n_pairs_kt_above_10) / 16.58492   # +1.0%  n_pairs_kt_above_10 < 23
        - 0.009475984 * max(0.0, Q.mass_top40 - 55.75601) * max(0.0, 0.1557839 - Q.dr_min_012) / 6.633597   # -0.9%  mass_top40 > 55.76 and dr_min_012 < 0.1558
        + 0.009271264 * max(0.0, 0.9329343 - Q.tau54) / 0.09160783   # +0.9%  tau54 < 0.9329
        - 0.009171766 * max(0.0, 0.4276838 - Q.tau32_b2) / 0.04662994   # -0.9%  tau32_b2 < 0.4277
        + 0.009013152 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # +0.9%  n_pairs_kt_above_1 < 366
        - 0.008793017 * max(0.0, 23.0 - Q.n_pairs_kt_above_10) * max(0.0, Q.sj2_dr - 0.2083105) / 2.766063   # -0.9%  n_pairs_kt_above_10 < 23 and sj2_dr > 0.2083
        - 0.007626416 * max(0.0, 0.02160244 - Q.M3_b2) / 0.009628213   # -0.8%  M3_b2 < 0.0216
        - 0.007481863 * max(0.0, 0.1528932 - Q.N2_b2) / 0.02866184   # -0.7%  N2_b2 < 0.1529
        + 0.007478768 * max(0.0, 23.0 - Q.n_pairs_kt_above_10) * max(0.0, 0.9539857 - Q.sj3_pairmax_over_m) / 2.380258   # +0.7%  n_pairs_kt_above_10 < 23 and sj3_pairmax_over_m < 0.954
        - 0.007190784 * max(0.0, 3.303713 - Q.pt_entropy) * max(0.0, 6.006437 - Q.sj3_mass2) / 0.6508844   # -0.7%  pt_entropy < 3.304 and sj3_mass2 < 6.006
        + 0.006898798 * max(0.0, Q.mass - 164.4374) / 3.038568   # +0.7%  mass > 164.4
        + 0.006243434 * max(0.0, 5.207416 - Q.sj3_mass2) / 0.4506597   # +0.6%  sj3_mass2 < 5.207
        - 0.00603281 * max(0.0, Q.sj3_pair_mass_min - 44.1643) / 6.009845   # -0.6%  sj3_pair_mass_min > 44.16
        - 0.006012387 * max(0.0, 184.8313 - Q.sum_pt_top2) / 19.05675   # -0.6%  sum_pt_top2 < 184.8
        + 0.005980245 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # +0.6%  n_pairs_kt_above_1 < 58
        + 0.005904517 * max(0.0, 5.0 - Q.n_dr_0p1_0p2) / 0.5919233   # +0.6%  n_dr_0p1_0p2 < 5
        - 0.005895156 * max(0.0, 3.303713 - Q.pt_entropy) * max(0.0, 24.55881 - Q.sj2_mass2) / 7.70719   # -0.6%  pt_entropy < 3.304 and sj2_mass2 < 24.56
        + 0.005816193 * max(0.0, 101.0 - Q.n_pairs_kt_above_1) / 8.83225   # +0.6%  n_pairs_kt_above_1 < 101
        + 0.005527934 * max(0.0, 0.4536006 - Q.pt_dispersion) * max(0.0, 3.217057 - Q.sj3_mass3) / 0.0633349   # +0.6%  pt_dispersion < 0.4536 and sj3_mass3 < 3.217
        + 0.005377649 * max(0.0, 7.0 - Q.n_lund) / 0.2630033   # +0.5%  n_lund < 7
        + 0.005190097 * max(0.0, Q.z_top20_slots - 0.9834667) / 0.002452163   # +0.5%  z_top20_slots > 0.9835
        - 0.004965903 * max(0.0, 3.217057 - Q.sj3_mass3) / 0.6549344   # -0.5%  sj3_mass3 < 3.217
        + 0.004882097 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.sum_z_dr2_top2 - 0.001440167) / 0.8573124   # +0.5%  n_pairs_kt_above_3 < 126 and sum_z_dr2_top2 > 0.00144
        + 0.004761452 * max(0.0, 3.303713 - Q.pt_entropy) * max(0.0, Q.n_lund - 9.0) / 0.8207453   # +0.5%  pt_entropy < 3.304 and n_lund > 9
        - 0.004533325 * max(0.0, 5.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.2116473 - Q.dr12) / 0.08946272   # -0.5%  n_dr_0p1_0p2 < 5 and dr12 < 0.2116
        - 0.004345564 * max(0.0, Q.M2_b05 - 0.1258758) * max(0.0, 0.1302765 - Q.sj4_dr_min) / 0.0005576096   # -0.4%  M2_b05 > 0.1259 and sj4_dr_min < 0.1303
        + 0.004125312 * max(0.0, -2.572052 - Q.pair_mean_lnz) / 0.02771534   # +0.4%  pair_mean_lnz < -2.572
        - 0.004067794 * max(0.0, 91.4679 - Q.mass_top30) / 6.787082   # -0.4%  mass_top30 < 91.47
        - 0.003041144 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.dr_0 - 0.03142385) / 4.086128   # -0.3%  n_pairs_kt_above_3 < 126 and dr_0 > 0.03142
        - 0.002965734 * max(0.0, 29.07349 - Q.sj3_pair_mass_min) / 4.80394   # -0.3%  sj3_pair_mass_min < 29.07
        + 0.002952497 * max(0.0, Q.mass_top40 - 55.75601) * max(0.0, Q.tau21 - 0.5440886) / 2.054452   # +0.3%  mass_top40 > 55.76 and tau21 > 0.5441
        + 0.002910239 * max(0.0, 36.0 - Q.n_real_top40) * max(0.0, Q.n_lund_kt_above_1 - 3.0) / 4.79206   # +0.3%  n_real_top40 < 36 and n_lund_kt_above_1 > 3
        - 0.002890664 * max(0.0, 0.02160244 - Q.M3_b2) * max(0.0, Q.mratio_max_012 - 0.787882) / 0.0001982235   # -0.3%  M3_b2 < 0.0216 and mratio_max_012 > 0.7879
        - 0.002694605 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.dr_1 - 0.01964368) / 0.4649398   # -0.3%  n_pairs_kt_above_3 < 34 and dr_1 > 0.01964
        + 0.002645675 * max(0.0, 0.02677564 - Q.tau2) / 0.0006804542   # +0.3%  tau2 < 0.02678
        - 0.002625065 * max(0.0, 0.2428609 - Q.z_dr_0p1_0p2) / 0.07286283   # -0.3%  z_dr_0p1_0p2 < 0.2429
        - 0.002431084 * max(0.0, Q.mass_top40 - 122.7145) * max(0.0, 0.9433644 - Q.tau43_b2) / 2.394169   # -0.2%  mass_top40 > 122.7 and tau43_b2 < 0.9434
        - 0.00239711 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.sd_zg - 0.2450652) / 3.949142   # -0.2%  n_pairs_kt_above_3 < 126 and sd_zg > 0.2451
        - 0.002193434 * max(0.0, Q.mass_top30 - 126.5007) / 5.381458   # -0.2%  mass_top30 > 126.5
        - 0.002191431 * max(0.0, Q.mass_top40 - 122.7145) / 8.484825   # -0.2%  mass_top40 > 122.7
        + 0.002091295 * max(0.0, 0.2757029 - Q.pt_balance01) * max(0.0, Q.sj2_zsoft - 0.2530865) / 0.0009522205   # +0.2%  pt_balance01 < 0.2757 and sj2_zsoft > 0.2531
        - 0.001963564 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.jet_abs_eta - 1.323111) / 2.444077   # -0.2%  n_pairs_kt_above_3 < 126 and jet_abs_eta > 1.323
        - 0.001855767 * max(0.0, 137.452 - Q.mass) * max(0.0, 0.0004257509 - Q.C3_b2) / 0.00319073   # -0.2%  mass < 137.5 and C3_b2 < 0.0004258
        + 0.001787918 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.02915846 - Q.dr_22) / 0.07696285   # +0.2%  n_pairs_kt_above_3 < 34 and dr_22 < 0.02916
        - 0.001769776 * max(0.0, 8.293431e-05 - Q.C3_b2) / 9.529737e-06   # -0.2%  C3_b2 < 8.293e-05
        - 0.001687977 * max(0.0, Q.mass_top50 - 178.725) / 1.512159   # -0.2%  mass_top50 > 178.7
        - 0.001614491 * max(0.0, 0.00678572 - Q.M3_b2) / 0.0005845828   # -0.2%  M3_b2 < 0.006786
        - 0.001507274 * max(0.0, Q.n_pairs_kt_above_3 - 171.0) / 8.03303   # -0.2%  n_pairs_kt_above_3 > 171
        - 0.001446375 * max(0.0, 0.006735806 - Q.C2_b2) / 0.0001176256   # -0.1%  C2_b2 < 0.006736
        + 0.001436591 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.1403176 - Q.dr_2) / 0.3495694   # +0.1%  n_pairs_kt_above_3 < 34 and dr_2 < 0.1403
        + 0.001336322 * max(0.0, 0.4276838 - Q.tau32_b2) * max(0.0, 7.508521 - Q.sj3_mass2) / 0.05886281   # +0.1%  tau32_b2 < 0.4277 and sj3_mass2 < 7.509
        + 0.001313793 * max(0.0, 598.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.sj3_mass2 - 1.824785) / 2186.199   # +0.1%  n_pairs_kt_above_1 < 598 and sj3_mass2 > 1.825
        + 0.001255847 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # +0.1%  sj2_mass2 < 1.852
        - 0.001235091 * max(0.0, 0.8415274 - Q.tau32) * max(0.0, 0.04063514 - Q.dr_21) / 0.0009863141   # -0.1%  tau32 < 0.8415 and dr_21 < 0.04064
        + 0.001231409 * max(0.0, Q.sj3_pair_mass_min - 44.1643) * max(0.0, 4.821289 - Q.lne_3) / 5.419306   # +0.1%  sj3_pair_mass_min > 44.16 and lne_3 < 4.821
        - 0.001036697 * max(0.0, 598.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.e4_b05 - 7.503722e-05) / 0.001749113   # -0.1%  n_pairs_kt_above_1 < 598 and e4_b05 > 7.504e-05
        - 0.001036014 * max(0.0, 598.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.sj2_zsoft - 0.2089827) / 26.91716   # -0.1%  n_pairs_kt_above_1 < 598 and sj2_zsoft > 0.209
        - 0.0009142145 * max(0.0, 0.2757029 - Q.pt_balance01) / 0.01818717   # -0.1%  pt_balance01 < 0.2757
        + 0.0009039616 * max(0.0, 101.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.n_dr_0p05_0p1 - 6.0) / 6.513577   # +0.1%  n_pairs_kt_above_1 < 101 and n_dr_0p05_0p1 > 6
        - 0.0008834711 * max(0.0, Q.sj3_pair_mass_min - 44.1643) * max(0.0, 4.0 - Q.n_lund_kt_above_5) / 6.782067   # -0.1%  sj3_pair_mass_min > 44.16 and n_lund_kt_above_5 < 4
        + 0.0008694667 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.dr12 - 0.3141459) / 0.05421831   # +0.1%  n_pairs_kt_above_3 < 34 and dr12 > 0.3141
        + 0.0005834647 * max(0.0, Q.mass_top40 - 122.7145) * max(0.0, 0.2116473 - Q.dr12) / 0.7906174   # +0.1%  mass_top40 > 122.7 and dr12 < 0.2116
        - 0.0004937194 * max(0.0, Q.M2_b05 - 0.1258758) * max(0.0, 0.456416 - Q.tau32) / 0.0002559086   # -0.0%  M2_b05 > 0.1259 and tau32 < 0.4564
        - 0.0004261918 * max(0.0, Q.sj3_mass1 - 44.1303) / 0.6338316   # -0.0%  sj3_mass1 > 44.13
        + 0.000408319 * max(0.0, 3.303713 - Q.pt_entropy) * max(0.0, 0.1335998 - Q.dr01) / 0.03364197   # +0.0%  pt_entropy < 3.304 and dr01 < 0.1336
        + 0.0002916814 * max(0.0, 34.83233 - Q.sj4_pair_mass_min) * max(0.0, Q.e4 - 7.184834e-06) / 4.07853e-06   # +0.0%  sj4_pair_mass_min < 34.83 and e4 > 7.185e-06
        - 0.000264469 * max(0.0, Q.mass_top30 - 126.5007) * max(0.0, Q.phi_17 - -0.1699219) / 1.132899   # -0.0%  mass_top30 > 126.5 and phi_17 > -0.1699
        + 0.0001381976 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.dr_1 - 0.03335825) / 5.007184   # +0.0%  n_pairs_kt_above_3 < 126 and dr_1 > 0.03336
        - 0.000121041 * max(0.0, 23.0 - Q.n_pairs_kt_above_10) * max(0.0, Q.eta_2 - 0.126709) / 0.1084651   # -0.0%  n_pairs_kt_above_10 < 23 and eta_2 > 0.1267
        - 0.0001046581 * max(0.0, Q.M2_b05 - 0.1258758) * max(0.0, Q.C3_b2 - 2.435109e-06) / 0.0001018751   # -0.0%  M2_b05 > 0.1259 and C3_b2 > 2.435e-06
        - 9.774658e-05 * max(0.0, 0.9329343 - Q.tau54) * max(0.0, -0.0838623 - Q.phi_2) / 0.001655157   # -0.0%  tau54 < 0.9329 and phi_2 < -0.08386
        + 6.440916e-05 * max(0.0, Q.mass_top40 - 122.7145) * max(0.0, 12.0 - Q.n_dr_0p2_0p4) / 14.9   # +0.0%  mass_top40 > 122.7 and n_dr_0p2_0p4 < 12
        - 1.917757e-05 * max(0.0, 1.851735 - Q.sj2_mass2) * max(0.0, -0.3249512 - Q.phi_31) / 0.0001446519   # -0.0%  sj2_mass2 < 1.852 and phi_31 < -0.325
        + 2.250605e-06 * max(0.0, Q.mass - 164.4374) * max(0.0, 0.1558838 - Q.eta_39) / 0.65571   # +0.0%  mass > 164.4 and eta_39 < 0.1559
    )
    return z


def neuron_122(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.169684e-06
    )
    return z


def neuron_123(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.021101e-05
    )
    return z


def neuron_124(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.958967e-06
    )
    return z


def neuron_125(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.844663e-05
    )
    return z


def neuron_126(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.142643e-05
    )
    return z


def neuron_127(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.445596e-06
    )
    return z


def class_token(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q), neuron_16(Q), neuron_17(Q), neuron_18(Q), neuron_19(Q), neuron_20(Q), neuron_21(Q), neuron_22(Q), neuron_23(Q), neuron_24(Q), neuron_25(Q), neuron_26(Q), neuron_27(Q), neuron_28(Q), neuron_29(Q), neuron_30(Q), neuron_31(Q), neuron_32(Q), neuron_33(Q), neuron_34(Q), neuron_35(Q), neuron_36(Q), neuron_37(Q), neuron_38(Q), neuron_39(Q), neuron_40(Q), neuron_41(Q), neuron_42(Q), neuron_43(Q), neuron_44(Q), neuron_45(Q), neuron_46(Q), neuron_47(Q), neuron_48(Q), neuron_49(Q), neuron_50(Q), neuron_51(Q), neuron_52(Q), neuron_53(Q), neuron_54(Q), neuron_55(Q), neuron_56(Q), neuron_57(Q), neuron_58(Q), neuron_59(Q), neuron_60(Q), neuron_61(Q), neuron_62(Q), neuron_63(Q), neuron_64(Q), neuron_65(Q), neuron_66(Q), neuron_67(Q), neuron_68(Q), neuron_69(Q), neuron_70(Q), neuron_71(Q), neuron_72(Q), neuron_73(Q), neuron_74(Q), neuron_75(Q), neuron_76(Q), neuron_77(Q), neuron_78(Q), neuron_79(Q), neuron_80(Q), neuron_81(Q), neuron_82(Q), neuron_83(Q), neuron_84(Q), neuron_85(Q), neuron_86(Q), neuron_87(Q), neuron_88(Q), neuron_89(Q), neuron_90(Q), neuron_91(Q), neuron_92(Q), neuron_93(Q), neuron_94(Q), neuron_95(Q), neuron_96(Q), neuron_97(Q), neuron_98(Q), neuron_99(Q), neuron_100(Q), neuron_101(Q), neuron_102(Q), neuron_103(Q), neuron_104(Q), neuron_105(Q), neuron_106(Q), neuron_107(Q), neuron_108(Q), neuron_109(Q), neuron_110(Q), neuron_111(Q), neuron_112(Q), neuron_113(Q), neuron_114(Q), neuron_115(Q), neuron_116(Q), neuron_117(Q), neuron_118(Q), neuron_119(Q), neuron_120(Q), neuron_121(Q), neuron_122(Q), neuron_123(Q), neuron_124(Q), neuron_125(Q), neuron_126(Q), neuron_127(Q)]


H_AVG = [2.3223441530717537e-05, 4.7610355977667496e-06, 0.000625310989562422, 4.991911282559158e-06, 0.2754983485520545, 0.0006186546524986625, 2.169769322790671e-05, 1.171534586319467e-05, 1.3417235095403157e-05, 0.2722261810139913, 4.742235432786401e-06, 4.878501385974232e-06, 1.6283067452604882e-05, 6.684680101898266e-06, 9.188985131913796e-06, 1.0382567779743113e-05, 8.221252755902242e-06, 1.2693948519881815e-05, 5.3481559007195756e-05, 1.5537894796580076e-05, 4.223183714202605e-05, 1.0482672223588452e-06, 0.5148767006649316, 1.4006432138558012e-05, 5.022346886107698e-05, 0.00015393634384963661, 4.593832272803411e-06, 0.8786713665014455, 0.05822149890661946, 2.3019743821350858e-05, 0.058898862270419704, 2.8225103960721754e-06, 1.2321979738771915e-06, 0.7005644482168505, 2.4639919615765393e-07, 0.04584319670845008, 1.035493937706633e-06, 3.7011697713751346e-05, 0.6668205247625325, 9.904104445013218e-06, 5.0037660912494175e-06, 8.526474630343728e-06, 4.680510301113827e-06, 3.898100203514332e-06, 2.3328244424192235e-05, 3.218036727048457e-05, 9.101478099182714e-06, 4.134960181545466e-05, 8.639792667963775e-07, 1.9026601876248606e-05, 4.526367411017418e-05, 0.6853954057434439, 2.68820116389179e-07, 8.044334208534565e-06, 1.5292748650438708, 6.958066478546243e-06, 1.8051575807476183e-06, 0.0016884534852579236, 2.7755879273172468e-05, 6.876595307403477e-06, 8.460408935206942e-06, 4.69128463009838e-06, 6.22240122538642e-06, 2.7300025976728648e-05, 1.5831088603590615e-05, 1.7997739632846788e-05, 1.8196469682152383e-05, 6.789744475099724e-06, 5.798125357614481e-07, 0.13137494740696165, 0.20086155801438468, 1.1814277968369424e-05, 2.547756230342202e-05, 2.301758468092885e-05, 1.363608862448018e-05, 1.582183131176862e-06, 2.7397811663831817e-07, 2.0052515537827276e-05, 3.790103392020683e-06, 1.505479900743012e-07, 3.804589869105257e-05, 0.19863842541303983, 0.0004023885412607342, 1.6589836377534084e-05, 2.650188207553583e-06, 6.717486940033268e-06, 0.7706476432246206, 1.6395897546317428e-06, 4.162050026934594e-05, 7.85128304414684e-06, 1.8974318663822487e-05, 7.23861921869684e-06, 8.794084351393394e-06, 2.1000034848839277e-06, 2.3842123482609168e-05, 3.2773445127531886e-05, 4.3241387174930423e-05, 0.4081280369967957, 1.0674192481019418e-06, 1.5804722352186218e-05, 2.5440167519263923e-05, 3.0244464142015204e-05, 0.11063819943904649, 0.8608200280126334, 0.3377621393665819, 4.0082845771394204e-06, 2.010549360420555e-05, 5.028072337154299e-05, 3.8201554275474336e-07, 8.850337508192752e-06, 5.695643267245032e-06, 2.5461217774136458e-06, 3.208626731066033e-05, 3.07340087601915e-05, 2.3422038793796673e-05, 2.963158294733148e-06, 4.341653402661905e-05, 3.9642418414587155e-06, 3.141795332339825e-06, 2.9524569526984124e-06, 1.1082440778409364e-06, 0.9982267013457705, 4.169684416410746e-06, 1.0211010703642387e-05, 2.958966661026352e-06, 3.844662569463253e-05, 1.1426431228755973e-05, 1.4455960126724676e-06]
T = [3.029348672599432, 3.625783940212129, 3.3290349705693387, 4.156554506179571, 5.298959931019401, 6.078472254689889, 2.9927258566909027, 3.9015580534541563, 5.667470394891636, 6.919729285662835]


def logits(h):
    # rounded to a multiple of 2^-20 (the formula's class scores are exact binary fractions): removes floating-point noise
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        0.4440334 + T[0] * (   # class QCD
            - 0.1594641 * h[38] / H_AVG[38]
            - 0.1472635 * h[33] / H_AVG[33]
            + 0.1373764 * h[51] / H_AVG[51]
            - 0.1074385 * h[86] / H_AVG[86]
            - 0.1074319 * h[97] / H_AVG[97]
            + 0.06880034 * h[103] / H_AVG[103]
            - 0.06640316 * h[4] / H_AVG[4]
            + 0.05864869 * h[9] / H_AVG[9]
            + 0.0530207 * h[27] / H_AVG[27]
            + 0.04630463 * h[104] / H_AVG[104]
            + 0.01292573 * h[81] / H_AVG[81]
            + 0.01166342 * h[54] / H_AVG[54]
            + 0.008144524 * h[70] / H_AVG[70]
            - 0.005547067 * h[69] / H_AVG[69]
            + 0.00371146 * h[28] / H_AVG[28]
            - 0.003345352 * h[22] / H_AVG[22]
            + 0.001495897 * h[102] / H_AVG[102]
            + 0.0006163136 * h[35] / H_AVG[35]
            - 0.0002097572 * h[30] / H_AVG[30]
            + 0.0001050612 * h[121] / H_AVG[121]
            + 4.55922e-05 * h[57] / H_AVG[57]
            - 1.064223e-05 * h[82] / H_AVG[82]
            + 7.268985e-06 * h[5] / H_AVG[5]
            - 5.445475e-06 * h[25] / H_AVG[25]
            + 3.651551e-06 * h[2] / H_AVG[2]
            + 9.976597e-07 * h[107] / H_AVG[107]
            + 5.895333e-07 * h[18] / H_AVG[18]
            - 4.942531e-07 * h[20] / H_AVG[20]
            + 4.929101e-07 * h[24] / H_AVG[24]
            - 4.818141e-07 * h[80] / H_AVG[80]
            + 4.575009e-07 * h[50] / H_AVG[50]
            - 4.079739e-07 * h[116] / H_AVG[116]
            - 3.637243e-07 * h[63] / H_AVG[63]
            - 3.546058e-07 * h[47] / H_AVG[47]
            + 3.43305e-07 * h[96] / H_AVG[96]
            - 3.395978e-07 * h[88] / H_AVG[88]
            + 3.375541e-07 * h[125] / H_AVG[125]
            + 2.838774e-07 * h[95] / H_AVG[95]
            + 2.794737e-07 * h[37] / H_AVG[37]
            - 2.690898e-07 * h[101] / H_AVG[101]
            - 2.266217e-07 * h[58] / H_AVG[58]
            - 2.106788e-07 * h[113] / H_AVG[113]
            + 2.032406e-07 * h[45] / H_AVG[45]
            - 1.875924e-07 * h[112] / H_AVG[112]
            + 1.875412e-07 * h[94] / H_AVG[94]
            + 1.463707e-07 * h[126] / H_AVG[126]
            - 1.433472e-07 * h[99] / H_AVG[99]
            + 1.407015e-07 * h[73] / H_AVG[73]
            - 1.39692e-07 * h[0] / H_AVG[0]
            + 1.306889e-07 * h[6] / H_AVG[6]
            + 1.262838e-07 * h[49] / H_AVG[49]
            - 1.234897e-07 * h[114] / H_AVG[114]
            + 1.228769e-07 * h[77] / H_AVG[77]
            - 1.17753e-07 * h[72] / H_AVG[72]
            + 1.169058e-07 * h[29] / H_AVG[29]
            - 1.109577e-07 * h[83] / H_AVG[83]
            - 1.104743e-07 * h[12] / H_AVG[12]
            - 1.046038e-07 * h[44] / H_AVG[44]
            + 1.008684e-07 * h[19] / H_AVG[19]
            - 9.815632e-08 * h[100] / H_AVG[100]
            - 9.564271e-08 * h[65] / H_AVG[65]
            - 9.110872e-08 * h[64] / H_AVG[64]
            - 8.883603e-08 * h[106] / H_AVG[106]
            - 8.54947e-08 * h[74] / H_AVG[74]
            - 8.364126e-08 * h[7] / H_AVG[7]
            + 8.206845e-08 * h[90] / H_AVG[90]
            + 7.981796e-08 * h[66] / H_AVG[66]
            - 7.561197e-08 * h[71] / H_AVG[71]
            - 5.734383e-08 * h[17] / H_AVG[17]
            + 5.339097e-08 * h[53] / H_AVG[53]
            - 4.642924e-08 * h[23] / H_AVG[23]
            - 3.954822e-08 * h[67] / H_AVG[67]
            - 3.622119e-08 * h[55] / H_AVG[55]
            - 3.389198e-08 * h[123] / H_AVG[123]
            - 3.313133e-08 * h[109] / H_AVG[109]
            - 2.997812e-08 * h[92] / H_AVG[92]
            - 2.721381e-08 * h[15] / H_AVG[15]
            - 2.662588e-08 * h[46] / H_AVG[46]
            - 2.563173e-08 * h[8] / H_AVG[8]
            - 2.47448e-08 * h[14] / H_AVG[14]
            + 2.435899e-08 * h[59] / H_AVG[59]
            - 2.423762e-08 * h[41] / H_AVG[41]
            - 2.404938e-08 * h[39] / H_AVG[39]
            + 2.175656e-08 * h[16] / H_AVG[16]
            - 1.726665e-08 * h[110] / H_AVG[110]
            - 1.690078e-08 * h[85] / H_AVG[85]
            + 1.627788e-08 * h[119] / H_AVG[119]
            - 1.38982e-08 * h[89] / H_AVG[89]
            - 1.328136e-08 * h[60] / H_AVG[60]
            + 1.291518e-08 * h[43] / H_AVG[43]
            - 1.284612e-08 * h[105] / H_AVG[105]
            - 1.262023e-08 * h[91] / H_AVG[91]
            - 1.121017e-08 * h[13] / H_AVG[13]
            - 1.045396e-08 * h[10] / H_AVG[10]
            + 9.882961e-09 * h[48] / H_AVG[48]
            - 9.634754e-09 * h[84] / H_AVG[84]
            - 9.311957e-09 * h[75] / H_AVG[75]
            + 8.451206e-09 * h[31] / H_AVG[31]
            + 7.952065e-09 * h[62] / H_AVG[62]
            + 7.270307e-09 * h[42] / H_AVG[42]
            + 6.727692e-09 * h[11] / H_AVG[11]
            - 6.386659e-09 * h[36] / H_AVG[36]
            - 6.03762e-09 * h[122] / H_AVG[122]
            - 5.722778e-09 * h[98] / H_AVG[98]
            + 5.502972e-09 * h[3] / H_AVG[3]
            + 5.329839e-09 * h[26] / H_AVG[26]
            - 5.226126e-09 * h[40] / H_AVG[40]
            - 5.11725e-09 * h[61] / H_AVG[61]
            + 4.881656e-09 * h[68] / H_AVG[68]
            - 4.063503e-09 * h[117] / H_AVG[117]
            - 3.86415e-09 * h[120] / H_AVG[120]
            + 3.402034e-09 * h[93] / H_AVG[93]
            - 3.400234e-09 * h[1] / H_AVG[1]
            - 3.388108e-09 * h[78] / H_AVG[78]
            - 3.378658e-09 * h[111] / H_AVG[111]
            + 2.897188e-09 * h[118] / H_AVG[118]
            + 2.134909e-09 * h[124] / H_AVG[124]
            - 2.07971e-09 * h[21] / H_AVG[21]
            - 1.620235e-09 * h[115] / H_AVG[115]
            - 1.487143e-09 * h[87] / H_AVG[87]
            - 1.365784e-09 * h[108] / H_AVG[108]
            + 1.202376e-09 * h[76] / H_AVG[76]
            - 1.018152e-09 * h[127] / H_AVG[127]
            - 7.677162e-10 * h[56] / H_AVG[56]
            + 7.597684e-10 * h[32] / H_AVG[32]
            - 5.79373e-11 * h[34] / H_AVG[34]
            - 3.918365e-11 * h[79] / H_AVG[79]
            - 1.108087e-11 * h[52] / H_AVG[52]
        ),
        0.2097928 + T[1] * (   # class Hbb
            + 0.2564379 * h[54] / H_AVG[54]
            + 0.1666572 * h[86] / H_AVG[86]
            - 0.1488477 * h[121] / H_AVG[121]
            + 0.09033895 * h[27] / H_AVG[27]
            - 0.07874094 * h[51] / H_AVG[51]
            - 0.05607696 * h[33] / H_AVG[33]
            - 0.05468158 * h[22] / H_AVG[22]
            + 0.04697076 * h[97] / H_AVG[97]
            - 0.0315774 * h[9] / H_AVG[9]
            + 0.0197658 * h[104] / H_AVG[104]
            + 0.01076565 * h[81] / H_AVG[81]
            + 0.009147933 * h[38] / H_AVG[38]
            + 0.007439761 * h[4] / H_AVG[4]
            - 0.005725379 * h[102] / H_AVG[102]
            + 0.005612955 * h[69] / H_AVG[69]
            - 0.005391919 * h[70] / H_AVG[70]
            + 0.002367677 * h[30] / H_AVG[30]
            - 0.001857606 * h[35] / H_AVG[35]
            - 0.001217323 * h[28] / H_AVG[28]
            - 0.0003359729 * h[103] / H_AVG[103]
            + 2.124411e-05 * h[57] / H_AVG[57]
            - 4.549658e-06 * h[25] / H_AVG[25]
            + 3.034422e-06 * h[2] / H_AVG[2]
            + 2.499248e-06 * h[82] / H_AVG[82]
            + 2.285756e-06 * h[5] / H_AVG[5]
            + 8.33531e-07 * h[107] / H_AVG[107]
            + 4.926421e-07 * h[18] / H_AVG[18]
            - 4.131706e-07 * h[20] / H_AVG[20]
            + 4.117548e-07 * h[24] / H_AVG[24]
            - 4.026499e-07 * h[80] / H_AVG[80]
            + 3.82296e-07 * h[50] / H_AVG[50]
            - 3.408823e-07 * h[116] / H_AVG[116]
            - 3.038765e-07 * h[63] / H_AVG[63]
            - 2.963982e-07 * h[47] / H_AVG[47]
            + 2.868613e-07 * h[96] / H_AVG[96]
            - 2.837314e-07 * h[88] / H_AVG[88]
            + 2.820722e-07 * h[125] / H_AVG[125]
            + 2.371858e-07 * h[95] / H_AVG[95]
            + 2.335088e-07 * h[37] / H_AVG[37]
            - 2.248222e-07 * h[101] / H_AVG[101]
            - 1.894011e-07 * h[58] / H_AVG[58]
            - 1.759732e-07 * h[113] / H_AVG[113]
            + 1.698289e-07 * h[45] / H_AVG[45]
            - 1.568185e-07 * h[112] / H_AVG[112]
            + 1.567374e-07 * h[94] / H_AVG[94]
            + 1.222891e-07 * h[126] / H_AVG[126]
            - 1.197435e-07 * h[99] / H_AVG[99]
            + 1.175826e-07 * h[73] / H_AVG[73]
            - 1.167257e-07 * h[0] / H_AVG[0]
            + 1.092446e-07 * h[6] / H_AVG[6]
            + 1.055658e-07 * h[49] / H_AVG[49]
            - 1.031937e-07 * h[114] / H_AVG[114]
            + 1.026949e-07 * h[77] / H_AVG[77]
            - 9.844243e-08 * h[72] / H_AVG[72]
            + 9.767562e-08 * h[29] / H_AVG[29]
            - 9.2712e-08 * h[83] / H_AVG[83]
            - 9.233449e-08 * h[12] / H_AVG[12]
            - 8.739047e-08 * h[44] / H_AVG[44]
            + 8.431613e-08 * h[19] / H_AVG[19]
            - 8.201078e-08 * h[100] / H_AVG[100]
            - 7.986698e-08 * h[65] / H_AVG[65]
            - 7.613693e-08 * h[64] / H_AVG[64]
            - 7.42145e-08 * h[106] / H_AVG[106]
            - 7.142369e-08 * h[74] / H_AVG[74]
            - 6.987315e-08 * h[7] / H_AVG[7]
            + 6.858588e-08 * h[90] / H_AVG[90]
            + 6.669162e-08 * h[66] / H_AVG[66]
            - 6.321521e-08 * h[71] / H_AVG[71]
            - 4.791363e-08 * h[17] / H_AVG[17]
            + 4.460669e-08 * h[53] / H_AVG[53]
            - 3.878696e-08 * h[23] / H_AVG[23]
            - 3.304445e-08 * h[67] / H_AVG[67]
            - 3.026017e-08 * h[55] / H_AVG[55]
            - 2.833444e-08 * h[123] / H_AVG[123]
            - 2.76953e-08 * h[109] / H_AVG[109]
            - 2.504457e-08 * h[92] / H_AVG[92]
            - 2.275167e-08 * h[15] / H_AVG[15]
            - 2.225859e-08 * h[46] / H_AVG[46]
            - 2.141706e-08 * h[8] / H_AVG[8]
            - 2.067743e-08 * h[14] / H_AVG[14]
            + 2.03639e-08 * h[59] / H_AVG[59]
            - 2.02571e-08 * h[41] / H_AVG[41]
            - 2.009339e-08 * h[39] / H_AVG[39]
            + 1.817598e-08 * h[16] / H_AVG[16]
            - 1.443427e-08 * h[110] / H_AVG[110]
            - 1.412288e-08 * h[85] / H_AVG[85]
            + 1.359879e-08 * h[119] / H_AVG[119]
            - 1.161653e-08 * h[89] / H_AVG[89]
            - 1.110334e-08 * h[60] / H_AVG[60]
            + 1.079817e-08 * h[43] / H_AVG[43]
            - 1.073351e-08 * h[105] / H_AVG[105]
            - 1.054652e-08 * h[91] / H_AVG[91]
            - 9.337534e-09 * h[13] / H_AVG[13]
            - 8.737248e-09 * h[10] / H_AVG[10]
            + 8.257504e-09 * h[48] / H_AVG[48]
            - 8.049502e-09 * h[84] / H_AVG[84]
            - 7.780481e-09 * h[75] / H_AVG[75]
            + 7.060851e-09 * h[31] / H_AVG[31]
            + 6.644864e-09 * h[62] / H_AVG[62]
            + 6.074438e-09 * h[42] / H_AVG[42]
            + 5.622498e-09 * h[11] / H_AVG[11]
            - 5.336344e-09 * h[36] / H_AVG[36]
            - 5.046463e-09 * h[122] / H_AVG[122]
            - 4.781181e-09 * h[98] / H_AVG[98]
            + 4.597765e-09 * h[3] / H_AVG[3]
            + 4.448321e-09 * h[26] / H_AVG[26]
            - 4.366046e-09 * h[40] / H_AVG[40]
            - 4.27802e-09 * h[61] / H_AVG[61]
            + 4.078613e-09 * h[68] / H_AVG[68]
            - 3.393267e-09 * h[117] / H_AVG[117]
            - 3.230671e-09 * h[120] / H_AVG[120]
            + 2.842256e-09 * h[93] / H_AVG[93]
            - 2.840761e-09 * h[1] / H_AVG[1]
            - 2.822483e-09 * h[78] / H_AVG[78]
            - 2.822428e-09 * h[111] / H_AVG[111]
            + 2.420665e-09 * h[118] / H_AVG[118]
            + 1.782927e-09 * h[124] / H_AVG[124]
            - 1.737501e-09 * h[21] / H_AVG[21]
            - 1.353781e-09 * h[115] / H_AVG[115]
            - 1.242599e-09 * h[87] / H_AVG[87]
            - 1.141043e-09 * h[108] / H_AVG[108]
            + 1.004565e-09 * h[76] / H_AVG[76]
            - 8.512705e-10 * h[127] / H_AVG[127]
            - 6.410392e-10 * h[56] / H_AVG[56]
            + 6.348183e-10 * h[32] / H_AVG[32]
            - 4.840105e-11 * h[34] / H_AVG[34]
            - 3.273625e-11 * h[79] / H_AVG[79]
            - 9.301151e-12 * h[52] / H_AVG[52]
        ),
        0.3583736 + T[2] * (   # class Hcc
            + 0.1682764 * h[27] / H_AVG[27]
            - 0.1174959 * h[51] / H_AVG[51]
            + 0.1006832 * h[38] / H_AVG[38]
            - 0.09904275 * h[103] / H_AVG[103]
            + 0.09534299 * h[54] / H_AVG[54]
            + 0.08400699 * h[33] / H_AVG[33]
            - 0.06456251 * h[22] / H_AVG[22]
            - 0.05626056 * h[86] / H_AVG[86]
            - 0.04668547 * h[9] / H_AVG[9]
            - 0.04507144 * h[121] / H_AVG[121]
            - 0.04289175 * h[97] / H_AVG[97]
            + 0.02044517 * h[104] / H_AVG[104]
            - 0.01817733 * h[70] / H_AVG[70]
            - 0.01444125 * h[4] / H_AVG[4]
            + 0.01262004 * h[69] / H_AVG[69]
            + 0.003562244 * h[30] / H_AVG[30]
            - 0.003000342 * h[81] / H_AVG[81]
            - 0.002832746 * h[102] / H_AVG[102]
            - 0.002619468 * h[28] / H_AVG[28]
            - 0.001943038 * h[35] / H_AVG[35]
            - 1.481213e-05 * h[57] / H_AVG[57]
            - 4.955111e-06 * h[25] / H_AVG[25]
            + 3.348727e-06 * h[2] / H_AVG[2]
            + 3.176089e-06 * h[5] / H_AVG[5]
            + 2.180618e-06 * h[82] / H_AVG[82]
            + 9.077685e-07 * h[107] / H_AVG[107]
            + 5.364965e-07 * h[18] / H_AVG[18]
            - 4.496916e-07 * h[20] / H_AVG[20]
            + 4.485344e-07 * h[24] / H_AVG[24]
            - 4.384236e-07 * h[80] / H_AVG[80]
            + 4.163147e-07 * h[50] / H_AVG[50]
            - 3.713429e-07 * h[116] / H_AVG[116]
            - 3.309777e-07 * h[63] / H_AVG[63]
            - 3.226275e-07 * h[47] / H_AVG[47]
            + 3.122892e-07 * h[96] / H_AVG[96]
            - 3.089675e-07 * h[88] / H_AVG[88]
            + 3.071722e-07 * h[125] / H_AVG[125]
            + 2.582954e-07 * h[95] / H_AVG[95]
            + 2.54303e-07 * h[37] / H_AVG[37]
            - 2.448394e-07 * h[101] / H_AVG[101]
            - 2.062656e-07 * h[58] / H_AVG[58]
            - 1.916418e-07 * h[113] / H_AVG[113]
            + 1.848983e-07 * h[45] / H_AVG[45]
            - 1.707273e-07 * h[112] / H_AVG[112]
            + 1.706505e-07 * h[94] / H_AVG[94]
            + 1.331882e-07 * h[126] / H_AVG[126]
            - 1.304122e-07 * h[99] / H_AVG[99]
            + 1.280329e-07 * h[73] / H_AVG[73]
            - 1.270405e-07 * h[0] / H_AVG[0]
            + 1.189363e-07 * h[6] / H_AVG[6]
            + 1.149034e-07 * h[49] / H_AVG[49]
            - 1.123719e-07 * h[114] / H_AVG[114]
            + 1.118157e-07 * h[77] / H_AVG[77]
            - 1.071663e-07 * h[72] / H_AVG[72]
            + 1.064014e-07 * h[29] / H_AVG[29]
            - 1.009385e-07 * h[83] / H_AVG[83]
            - 1.005477e-07 * h[12] / H_AVG[12]
            - 9.518751e-08 * h[44] / H_AVG[44]
            + 9.182404e-08 * h[19] / H_AVG[19]
            - 8.930688e-08 * h[100] / H_AVG[100]
            - 8.703795e-08 * h[65] / H_AVG[65]
            - 8.290303e-08 * h[64] / H_AVG[64]
            - 8.080569e-08 * h[106] / H_AVG[106]
            - 7.778362e-08 * h[74] / H_AVG[74]
            - 7.61132e-08 * h[7] / H_AVG[7]
            + 7.469385e-08 * h[90] / H_AVG[90]
            + 7.265158e-08 * h[66] / H_AVG[66]
            - 6.882339e-08 * h[71] / H_AVG[71]
            - 5.216937e-08 * h[17] / H_AVG[17]
            + 4.857684e-08 * h[53] / H_AVG[53]
            - 4.222923e-08 * h[23] / H_AVG[23]
            - 3.598101e-08 * h[67] / H_AVG[67]
            - 3.295311e-08 * h[55] / H_AVG[55]
            - 3.084297e-08 * h[123] / H_AVG[123]
            - 3.015169e-08 * h[109] / H_AVG[109]
            - 2.727825e-08 * h[92] / H_AVG[92]
            - 2.476888e-08 * h[15] / H_AVG[15]
            - 2.423329e-08 * h[46] / H_AVG[46]
            - 2.33181e-08 * h[8] / H_AVG[8]
            - 2.251686e-08 * h[14] / H_AVG[14]
            + 2.216681e-08 * h[59] / H_AVG[59]
            - 2.205606e-08 * h[41] / H_AVG[41]
            - 2.188289e-08 * h[39] / H_AVG[39]
            + 1.979485e-08 * h[16] / H_AVG[16]
            - 1.570999e-08 * h[110] / H_AVG[110]
            - 1.538034e-08 * h[85] / H_AVG[85]
            + 1.481075e-08 * h[119] / H_AVG[119]
            - 1.264723e-08 * h[89] / H_AVG[89]
            - 1.209001e-08 * h[60] / H_AVG[60]
            + 1.175225e-08 * h[43] / H_AVG[43]
            - 1.168956e-08 * h[105] / H_AVG[105]
            - 1.148537e-08 * h[91] / H_AVG[91]
            - 1.019354e-08 * h[13] / H_AVG[13]
            - 9.514289e-09 * h[10] / H_AVG[10]
            + 8.993021e-09 * h[48] / H_AVG[48]
            - 8.767236e-09 * h[84] / H_AVG[84]
            - 8.473759e-09 * h[75] / H_AVG[75]
            + 7.687551e-09 * h[31] / H_AVG[31]
            + 7.236091e-09 * h[62] / H_AVG[62]
            + 6.61721e-09 * h[42] / H_AVG[42]
            + 6.122649e-09 * h[11] / H_AVG[11]
            - 5.811909e-09 * h[36] / H_AVG[36]
            - 5.492847e-09 * h[122] / H_AVG[122]
            - 5.207957e-09 * h[98] / H_AVG[98]
            + 5.010551e-09 * h[3] / H_AVG[3]
            + 4.841754e-09 * h[26] / H_AVG[26]
            - 4.7554e-09 * h[40] / H_AVG[40]
            - 4.654433e-09 * h[61] / H_AVG[61]
            + 4.442203e-09 * h[68] / H_AVG[68]
            - 3.698032e-09 * h[117] / H_AVG[117]
            - 3.516418e-09 * h[120] / H_AVG[120]
            + 3.095696e-09 * h[93] / H_AVG[93]
            - 3.092734e-09 * h[1] / H_AVG[1]
            - 3.083717e-09 * h[78] / H_AVG[78]
            - 3.07386e-09 * h[111] / H_AVG[111]
            + 2.636447e-09 * h[118] / H_AVG[118]
            + 1.942206e-09 * h[124] / H_AVG[124]
            - 1.892444e-09 * h[21] / H_AVG[21]
            - 1.474214e-09 * h[115] / H_AVG[115]
            - 1.353195e-09 * h[87] / H_AVG[87]
            - 1.242663e-09 * h[108] / H_AVG[108]
            + 1.094098e-09 * h[76] / H_AVG[76]
            - 9.264665e-10 * h[127] / H_AVG[127]
            - 6.979505e-10 * h[56] / H_AVG[56]
            + 6.91333e-10 * h[32] / H_AVG[32]
            - 5.270742e-11 * h[34] / H_AVG[34]
            - 3.565191e-11 * h[79] / H_AVG[79]
            - 1.009365e-11 * h[52] / H_AVG[52]
        ),
        0.05534523 + T[3] * (   # class Hgg
            - 0.1631776 * h[86] / H_AVG[86]
            + 0.1601705 * h[54] / H_AVG[54]
            - 0.1360692 * h[121] / H_AVG[121]
            - 0.1291802 * h[33] / H_AVG[33]
            + 0.09374589 * h[38] / H_AVG[38]
            - 0.07133688 * h[103] / H_AVG[103]
            + 0.07017019 * h[97] / H_AVG[97]
            - 0.06957059 * h[27] / H_AVG[27]
            + 0.03817362 * h[4] / H_AVG[4]
            + 0.02553994 * h[9] / H_AVG[9]
            - 0.01573143 * h[22] / H_AVG[22]
            + 0.01464767 * h[51] / H_AVG[51]
            + 0.005973554 * h[69] / H_AVG[69]
            - 0.001989955 * h[102] / H_AVG[102]
            + 0.001410759 * h[28] / H_AVG[28]
            + 0.0009646002 * h[30] / H_AVG[30]
            + 0.0009067158 * h[70] / H_AVG[70]
            + 0.0006577483 * h[81] / H_AVG[81]
            - 0.0005109599 * h[35] / H_AVG[35]
            - 4.568231e-05 * h[104] / H_AVG[104]
            - 7.842671e-06 * h[57] / H_AVG[57]
            - 3.968374e-06 * h[25] / H_AVG[25]
            + 2.680549e-06 * h[2] / H_AVG[2]
            + 2.329748e-06 * h[82] / H_AVG[82]
            + 1.636597e-06 * h[5] / H_AVG[5]
            + 7.271217e-07 * h[107] / H_AVG[107]
            + 4.296712e-07 * h[18] / H_AVG[18]
            - 3.60278e-07 * h[20] / H_AVG[20]
            + 3.59156e-07 * h[24] / H_AVG[24]
            - 3.512427e-07 * h[80] / H_AVG[80]
            + 3.334674e-07 * h[50] / H_AVG[50]
            - 2.974639e-07 * h[116] / H_AVG[116]
            - 2.65086e-07 * h[63] / H_AVG[63]
            - 2.584548e-07 * h[47] / H_AVG[47]
            + 2.501867e-07 * h[96] / H_AVG[96]
            - 2.47469e-07 * h[88] / H_AVG[88]
            + 2.460191e-07 * h[125] / H_AVG[125]
            + 2.068878e-07 * h[95] / H_AVG[95]
            + 2.036547e-07 * h[37] / H_AVG[37]
            - 1.961136e-07 * h[101] / H_AVG[101]
            - 1.6516e-07 * h[58] / H_AVG[58]
            - 1.534789e-07 * h[113] / H_AVG[113]
            + 1.480894e-07 * h[45] / H_AVG[45]
            - 1.367167e-07 * h[112] / H_AVG[112]
            + 1.366718e-07 * h[94] / H_AVG[94]
            + 1.066794e-07 * h[126] / H_AVG[126]
            - 1.044687e-07 * h[99] / H_AVG[99]
            + 1.025882e-07 * h[73] / H_AVG[73]
            - 1.017693e-07 * h[0] / H_AVG[0]
            + 9.528434e-08 * h[6] / H_AVG[6]
            + 9.206802e-08 * h[49] / H_AVG[49]
            - 8.99989e-08 * h[114] / H_AVG[114]
            + 8.959075e-08 * h[77] / H_AVG[77]
            - 8.592698e-08 * h[72] / H_AVG[72]
            + 8.522053e-08 * h[29] / H_AVG[29]
            - 8.084004e-08 * h[83] / H_AVG[83]
            - 8.052991e-08 * h[12] / H_AVG[12]
            - 7.620956e-08 * h[44] / H_AVG[44]
            + 7.353838e-08 * h[19] / H_AVG[19]
            - 7.153809e-08 * h[100] / H_AVG[100]
            - 6.958182e-08 * h[65] / H_AVG[65]
            - 6.642145e-08 * h[64] / H_AVG[64]
            - 6.472378e-08 * h[106] / H_AVG[106]
            - 6.230119e-08 * h[74] / H_AVG[74]
            - 6.095705e-08 * h[7] / H_AVG[7]
            + 5.982394e-08 * h[90] / H_AVG[90]
            + 5.818508e-08 * h[66] / H_AVG[66]
            - 5.513966e-08 * h[71] / H_AVG[71]
            - 4.179044e-08 * h[17] / H_AVG[17]
            + 3.890917e-08 * h[53] / H_AVG[53]
            - 3.38357e-08 * h[23] / H_AVG[23]
            - 2.882564e-08 * h[67] / H_AVG[67]
            - 2.639418e-08 * h[55] / H_AVG[55]
            - 2.469424e-08 * h[123] / H_AVG[123]
            - 2.415182e-08 * h[109] / H_AVG[109]
            - 2.184867e-08 * h[92] / H_AVG[92]
            - 1.984362e-08 * h[15] / H_AVG[15]
            - 1.941342e-08 * h[46] / H_AVG[46]
            - 1.868052e-08 * h[8] / H_AVG[8]
            - 1.802186e-08 * h[14] / H_AVG[14]
            + 1.77572e-08 * h[59] / H_AVG[59]
            - 1.766763e-08 * h[41] / H_AVG[41]
            - 1.752437e-08 * h[39] / H_AVG[39]
            + 1.585436e-08 * h[16] / H_AVG[16]
            - 1.259137e-08 * h[110] / H_AVG[110]
            - 1.231445e-08 * h[85] / H_AVG[85]
            + 1.18628e-08 * h[119] / H_AVG[119]
            - 1.01316e-08 * h[89] / H_AVG[89]
            - 9.683689e-09 * h[60] / H_AVG[60]
            + 9.412947e-09 * h[43] / H_AVG[43]
            - 9.36282e-09 * h[105] / H_AVG[105]
            - 9.198508e-09 * h[91] / H_AVG[91]
            - 8.175909e-09 * h[13] / H_AVG[13]
            - 7.622551e-09 * h[10] / H_AVG[10]
            + 7.202901e-09 * h[48] / H_AVG[48]
            - 7.021954e-09 * h[84] / H_AVG[84]
            - 6.7879e-09 * h[75] / H_AVG[75]
            + 6.157594e-09 * h[31] / H_AVG[31]
            + 5.796795e-09 * h[62] / H_AVG[62]
            + 5.300063e-09 * h[42] / H_AVG[42]
            + 4.903829e-09 * h[11] / H_AVG[11]
            - 4.65504e-09 * h[36] / H_AVG[36]
            - 4.399948e-09 * h[122] / H_AVG[122]
            - 4.171106e-09 * h[98] / H_AVG[98]
            + 4.012106e-09 * h[3] / H_AVG[3]
            + 3.885944e-09 * h[26] / H_AVG[26]
            - 3.809196e-09 * h[40] / H_AVG[40]
            - 3.728038e-09 * h[61] / H_AVG[61]
            + 3.558627e-09 * h[68] / H_AVG[68]
            - 2.960014e-09 * h[117] / H_AVG[117]
            - 2.81652e-09 * h[120] / H_AVG[120]
            + 2.479256e-09 * h[93] / H_AVG[93]
            - 2.477625e-09 * h[1] / H_AVG[1]
            - 2.465278e-09 * h[78] / H_AVG[78]
            - 2.461677e-09 * h[111] / H_AVG[111]
            + 2.111726e-09 * h[118] / H_AVG[118]
            + 1.556021e-09 * h[124] / H_AVG[124]
            - 1.515852e-09 * h[21] / H_AVG[21]
            - 1.180685e-09 * h[115] / H_AVG[115]
            - 1.083844e-09 * h[87] / H_AVG[87]
            - 9.954384e-10 * h[108] / H_AVG[108]
            + 8.763611e-10 * h[76] / H_AVG[76]
            - 7.425551e-10 * h[127] / H_AVG[127]
            - 5.59476e-10 * h[56] / H_AVG[56]
            + 5.53742e-10 * h[32] / H_AVG[32]
            - 4.222125e-11 * h[34] / H_AVG[34]
            - 2.855826e-11 * h[79] / H_AVG[79]
            - 8.061819e-12 * h[52] / H_AVG[52]
        ),
        -0.1126753 + T[4] * (   # class H4q
            - 0.2009723 * h[27] / H_AVG[27]
            + 0.1461102 * h[54] / H_AVG[54]
            + 0.1331938 * h[103] / H_AVG[103]
            + 0.1054517 * h[38] / H_AVG[38]
            - 0.09030262 * h[86] / H_AVG[86]
            + 0.08162882 * h[33] / H_AVG[33]
            + 0.04999201 * h[121] / H_AVG[121]
            - 0.04281483 * h[22] / H_AVG[22]
            + 0.03693696 * h[97] / H_AVG[97]
            + 0.03595659 * h[104] / H_AVG[104]
            + 0.02587477 * h[9] / H_AVG[9]
            - 0.01730728 * h[4] / H_AVG[4]
            - 0.01157268 * h[70] / H_AVG[70]
            - 0.01025829 * h[51] / H_AVG[51]
            - 0.008416496 * h[81] / H_AVG[81]
            + 0.001355468 * h[102] / H_AVG[102]
            + 0.0009518755 * h[30] / H_AVG[30]
            - 0.0004431063 * h[35] / H_AVG[35]
            + 0.0004208945 * h[69] / H_AVG[69]
            - 2.204492e-05 * h[28] / H_AVG[28]
            - 3.112966e-06 * h[25] / H_AVG[25]
            - 2.307004e-06 * h[57] / H_AVG[57]
            + 2.098861e-06 * h[2] / H_AVG[2]
            + 1.945116e-06 * h[5] / H_AVG[5]
            + 1.603504e-06 * h[82] / H_AVG[82]
            + 5.703261e-07 * h[107] / H_AVG[107]
            + 3.370028e-07 * h[18] / H_AVG[18]
            - 2.826126e-07 * h[20] / H_AVG[20]
            + 2.817271e-07 * h[24] / H_AVG[24]
            - 2.754996e-07 * h[80] / H_AVG[80]
            + 2.616075e-07 * h[50] / H_AVG[50]
            - 2.332939e-07 * h[116] / H_AVG[116]
            - 2.079558e-07 * h[63] / H_AVG[63]
            - 2.026486e-07 * h[47] / H_AVG[47]
            + 1.962266e-07 * h[96] / H_AVG[96]
            - 1.940843e-07 * h[88] / H_AVG[88]
            + 1.929867e-07 * h[125] / H_AVG[125]
            + 1.623136e-07 * h[95] / H_AVG[95]
            + 1.597763e-07 * h[37] / H_AVG[37]
            - 1.53816e-07 * h[101] / H_AVG[101]
            - 1.295786e-07 * h[58] / H_AVG[58]
            - 1.203334e-07 * h[113] / H_AVG[113]
            + 1.161686e-07 * h[45] / H_AVG[45]
            - 1.072707e-07 * h[112] / H_AVG[112]
            + 1.072203e-07 * h[94] / H_AVG[94]
            + 8.367492e-08 * h[126] / H_AVG[126]
            - 8.193959e-08 * h[99] / H_AVG[99]
            + 8.045197e-08 * h[73] / H_AVG[73]
            - 7.981205e-08 * h[0] / H_AVG[0]
            + 7.474774e-08 * h[6] / H_AVG[6]
            + 7.222935e-08 * h[49] / H_AVG[49]
            - 7.063187e-08 * h[114] / H_AVG[114]
            + 7.027716e-08 * h[77] / H_AVG[77]
            - 6.736126e-08 * h[72] / H_AVG[72]
            + 6.684648e-08 * h[29] / H_AVG[29]
            - 6.342158e-08 * h[83] / H_AVG[83]
            - 6.317505e-08 * h[12] / H_AVG[12]
            - 5.979034e-08 * h[44] / H_AVG[44]
            + 5.768423e-08 * h[19] / H_AVG[19]
            - 5.609781e-08 * h[100] / H_AVG[100]
            - 5.465805e-08 * h[65] / H_AVG[65]
            - 5.211123e-08 * h[64] / H_AVG[64]
            - 5.078087e-08 * h[106] / H_AVG[106]
            - 4.886735e-08 * h[74] / H_AVG[74]
            - 4.781627e-08 * h[7] / H_AVG[7]
            + 4.692845e-08 * h[90] / H_AVG[90]
            + 4.564116e-08 * h[66] / H_AVG[66]
            - 4.325271e-08 * h[71] / H_AVG[71]
            - 3.277676e-08 * h[17] / H_AVG[17]
            + 3.052336e-08 * h[53] / H_AVG[53]
            - 2.654501e-08 * h[23] / H_AVG[23]
            - 2.26101e-08 * h[67] / H_AVG[67]
            - 2.070597e-08 * h[55] / H_AVG[55]
            - 1.935796e-08 * h[123] / H_AVG[123]
            - 1.895292e-08 * h[109] / H_AVG[109]
            - 1.714444e-08 * h[92] / H_AVG[92]
            - 1.556542e-08 * h[15] / H_AVG[15]
            - 1.522768e-08 * h[46] / H_AVG[46]
            - 1.464985e-08 * h[8] / H_AVG[8]
            - 1.414759e-08 * h[14] / H_AVG[14]
            + 1.393371e-08 * h[59] / H_AVG[59]
            - 1.385443e-08 * h[41] / H_AVG[41]
            - 1.374768e-08 * h[39] / H_AVG[39]
            + 1.244051e-08 * h[16] / H_AVG[16]
            - 9.876137e-09 * h[110] / H_AVG[110]
            - 9.66457e-09 * h[85] / H_AVG[85]
            + 9.307003e-09 * h[119] / H_AVG[119]
            - 7.946306e-09 * h[89] / H_AVG[89]
            - 7.596055e-09 * h[60] / H_AVG[60]
            + 7.384204e-09 * h[43] / H_AVG[43]
            - 7.344882e-09 * h[105] / H_AVG[105]
            - 7.215239e-09 * h[91] / H_AVG[91]
            - 6.399456e-09 * h[13] / H_AVG[13]
            - 5.978522e-09 * h[10] / H_AVG[10]
            + 5.649997e-09 * h[48] / H_AVG[48]
            - 5.509027e-09 * h[84] / H_AVG[84]
            - 5.323722e-09 * h[75] / H_AVG[75]
            + 4.830892e-09 * h[31] / H_AVG[31]
            + 4.546287e-09 * h[62] / H_AVG[62]
            + 4.157739e-09 * h[42] / H_AVG[42]
            + 3.84648e-09 * h[11] / H_AVG[11]
            - 3.652088e-09 * h[36] / H_AVG[36]
            - 3.453177e-09 * h[122] / H_AVG[122]
            - 3.271897e-09 * h[98] / H_AVG[98]
            + 3.145846e-09 * h[3] / H_AVG[3]
            + 3.043605e-09 * h[26] / H_AVG[26]
            - 2.986472e-09 * h[40] / H_AVG[40]
            - 2.924446e-09 * h[61] / H_AVG[61]
            + 2.791193e-09 * h[68] / H_AVG[68]
            - 2.320251e-09 * h[117] / H_AVG[117]
            - 2.210793e-09 * h[120] / H_AVG[120]
            + 1.944794e-09 * h[93] / H_AVG[93]
            - 1.943104e-09 * h[1] / H_AVG[1]
            - 1.936399e-09 * h[78] / H_AVG[78]
            - 1.930749e-09 * h[111] / H_AVG[111]
            + 1.656725e-09 * h[118] / H_AVG[118]
            + 1.219888e-09 * h[124] / H_AVG[124]
            - 1.188989e-09 * h[21] / H_AVG[21]
            - 9.261066e-10 * h[115] / H_AVG[115]
            - 8.503014e-10 * h[87] / H_AVG[87]
            - 7.809369e-10 * h[108] / H_AVG[108]
            + 6.874725e-10 * h[76] / H_AVG[76]
            - 5.822866e-10 * h[127] / H_AVG[127]
            - 4.389908e-10 * h[56] / H_AVG[56]
            + 4.343749e-10 * h[32] / H_AVG[32]
            - 3.311352e-11 * h[34] / H_AVG[34]
            - 2.240062e-11 * h[79] / H_AVG[79]
            - 6.311461e-12 * h[52] / H_AVG[52]
        ),
        -0.4036941 + T[5] * (   # class Hqql
            - 0.2211154 * h[54] / H_AVG[54]
            - 0.1354022 * h[27] / H_AVG[27]
            - 0.1123347 * h[51] / H_AVG[51]
            + 0.0977072 * h[121] / H_AVG[121]
            - 0.07649069 * h[22] / H_AVG[22]
            - 0.05463539 * h[38] / H_AVG[38]
            - 0.05438351 * h[104] / H_AVG[104]
            - 0.04504437 * h[97] / H_AVG[97]
            + 0.04431359 * h[86] / H_AVG[86]
            + 0.03904448 * h[4] / H_AVG[4]
            + 0.03398419 * h[103] / H_AVG[103]
            - 0.03278884 * h[33] / H_AVG[33]
            + 0.02071163 * h[70] / H_AVG[70]
            + 0.01558754 * h[9] / H_AVG[9]
            + 0.007378149 * h[81] / H_AVG[81]
            - 0.003992496 * h[28] / H_AVG[28]
            - 0.003745807 * h[102] / H_AVG[102]
            - 0.0008093133 * h[69] / H_AVG[69]
            - 0.0003426336 * h[35] / H_AVG[35]
            + 8.1963e-05 * h[57] / H_AVG[57]
            - 5.929749e-05 * h[30] / H_AVG[30]
            - 1.884429e-05 * h[82] / H_AVG[82]
            + 1.790478e-05 * h[5] / H_AVG[5]
            - 2.713596e-06 * h[25] / H_AVG[25]
            + 1.810858e-06 * h[2] / H_AVG[2]
            + 4.971525e-07 * h[107] / H_AVG[107]
            + 2.937884e-07 * h[18] / H_AVG[18]
            - 2.464611e-07 * h[20] / H_AVG[20]
            + 2.456701e-07 * h[24] / H_AVG[24]
            - 2.401221e-07 * h[80] / H_AVG[80]
            + 2.280126e-07 * h[50] / H_AVG[50]
            - 2.033195e-07 * h[116] / H_AVG[116]
            - 1.812771e-07 * h[63] / H_AVG[63]
            - 1.766539e-07 * h[47] / H_AVG[47]
            + 1.710915e-07 * h[96] / H_AVG[96]
            - 1.692239e-07 * h[88] / H_AVG[88]
            + 1.682453e-07 * h[125] / H_AVG[125]
            + 1.414665e-07 * h[95] / H_AVG[95]
            + 1.392547e-07 * h[37] / H_AVG[37]
            - 1.340976e-07 * h[101] / H_AVG[101]
            - 1.129455e-07 * h[58] / H_AVG[58]
            - 1.049549e-07 * h[113] / H_AVG[113]
            + 1.012514e-07 * h[45] / H_AVG[45]
            - 9.348787e-08 * h[112] / H_AVG[112]
            + 9.345375e-08 * h[94] / H_AVG[94]
            + 7.294349e-08 * h[126] / H_AVG[126]
            - 7.143362e-08 * h[99] / H_AVG[99]
            + 7.013496e-08 * h[73] / H_AVG[73]
            - 6.957966e-08 * h[0] / H_AVG[0]
            + 6.513096e-08 * h[6] / H_AVG[6]
            + 6.294694e-08 * h[49] / H_AVG[49]
            - 6.15474e-08 * h[114] / H_AVG[114]
            + 6.125072e-08 * h[77] / H_AVG[77]
            - 5.874447e-08 * h[72] / H_AVG[72]
            + 5.82654e-08 * h[29] / H_AVG[29]
            - 5.528105e-08 * h[83] / H_AVG[83]
            - 5.504211e-08 * h[12] / H_AVG[12]
            - 5.211818e-08 * h[44] / H_AVG[44]
            + 5.026727e-08 * h[19] / H_AVG[19]
            - 4.891278e-08 * h[100] / H_AVG[100]
            - 4.763725e-08 * h[65] / H_AVG[65]
            - 4.540585e-08 * h[64] / H_AVG[64]
            - 4.426751e-08 * h[106] / H_AVG[106]
            - 4.259606e-08 * h[74] / H_AVG[74]
            - 4.168274e-08 * h[7] / H_AVG[7]
            + 4.09081e-08 * h[90] / H_AVG[90]
            + 3.97797e-08 * h[66] / H_AVG[66]
            - 3.76833e-08 * h[71] / H_AVG[71]
            - 2.857742e-08 * h[17] / H_AVG[17]
            + 2.660472e-08 * h[53] / H_AVG[53]
            - 2.312957e-08 * h[23] / H_AVG[23]
            - 1.970925e-08 * h[67] / H_AVG[67]
            - 1.80516e-08 * h[55] / H_AVG[55]
            - 1.690521e-08 * h[123] / H_AVG[123]
            - 1.651325e-08 * h[109] / H_AVG[109]
            - 1.494094e-08 * h[92] / H_AVG[92]
            - 1.355935e-08 * h[15] / H_AVG[15]
            - 1.326906e-08 * h[46] / H_AVG[46]
            - 1.277391e-08 * h[8] / H_AVG[8]
            - 1.232767e-08 * h[14] / H_AVG[14]
            + 1.214024e-08 * h[59] / H_AVG[59]
            - 1.208207e-08 * h[41] / H_AVG[41]
            - 1.198143e-08 * h[39] / H_AVG[39]
            + 1.084121e-08 * h[16] / H_AVG[16]
            - 8.607851e-09 * h[110] / H_AVG[110]
            - 8.4203e-09 * h[85] / H_AVG[85]
            + 8.111842e-09 * h[119] / H_AVG[119]
            - 6.925785e-09 * h[89] / H_AVG[89]
            - 6.621318e-09 * h[60] / H_AVG[60]
            + 6.436334e-09 * h[43] / H_AVG[43]
            - 6.40246e-09 * h[105] / H_AVG[105]
            - 6.288799e-09 * h[91] / H_AVG[91]
            - 5.587732e-09 * h[13] / H_AVG[13]
            - 5.21158e-09 * h[10] / H_AVG[10]
            + 4.925177e-09 * h[48] / H_AVG[48]
            - 4.801809e-09 * h[84] / H_AVG[84]
            - 4.641042e-09 * h[75] / H_AVG[75]
            + 4.211855e-09 * h[31] / H_AVG[31]
            + 3.962701e-09 * h[62] / H_AVG[62]
            + 3.623936e-09 * h[42] / H_AVG[42]
            + 3.353245e-09 * h[11] / H_AVG[11]
            - 3.183007e-09 * h[36] / H_AVG[36]
            - 3.010174e-09 * h[122] / H_AVG[122]
            - 2.852306e-09 * h[98] / H_AVG[98]
            + 2.741918e-09 * h[3] / H_AVG[3]
            + 2.659512e-09 * h[26] / H_AVG[26]
            - 2.605215e-09 * h[40] / H_AVG[40]
            - 2.548774e-09 * h[61] / H_AVG[61]
            + 2.432895e-09 * h[68] / H_AVG[68]
            - 2.023579e-09 * h[117] / H_AVG[117]
            - 1.926163e-09 * h[120] / H_AVG[120]
            + 1.695242e-09 * h[93] / H_AVG[93]
            - 1.693529e-09 * h[1] / H_AVG[1]
            - 1.687471e-09 * h[78] / H_AVG[78]
            - 1.683086e-09 * h[111] / H_AVG[111]
            + 1.444403e-09 * h[118] / H_AVG[118]
            + 1.063499e-09 * h[124] / H_AVG[124]
            - 1.036461e-09 * h[21] / H_AVG[21]
            - 8.072986e-10 * h[115] / H_AVG[115]
            - 7.411374e-10 * h[87] / H_AVG[87]
            - 6.806284e-10 * h[108] / H_AVG[108]
            + 5.99212e-10 * h[76] / H_AVG[76]
            - 5.076817e-10 * h[127] / H_AVG[127]
            - 3.824885e-10 * h[56] / H_AVG[56]
            + 3.786242e-10 * h[32] / H_AVG[32]
            - 2.886986e-11 * h[34] / H_AVG[34]
            - 1.952726e-11 * h[79] / H_AVG[79]
            - 5.506547e-12 * h[52] / H_AVG[52]
        ),
        0.3007382 + T[6] * (   # class Zqq
            - 0.2013813 * h[54] / H_AVG[54]
            + 0.12058 * h[86] / H_AVG[86]
            - 0.1191737 * h[103] / H_AVG[103]
            + 0.1105011 * h[22] / H_AVG[22]
            - 0.05519227 * h[38] / H_AVG[38]
            + 0.05276462 * h[33] / H_AVG[33]
            - 0.04701955 * h[81] / H_AVG[81]
            + 0.04441819 * h[104] / H_AVG[104]
            - 0.04167926 * h[27] / H_AVG[27]
            + 0.03744519 * h[121] / H_AVG[121]
            + 0.03597775 * h[70] / H_AVG[70]
            + 0.03442651 * h[51] / H_AVG[51]
            - 0.02869668 * h[69] / H_AVG[69]
            + 0.02479829 * h[9] / H_AVG[9]
            + 0.02159304 * h[102] / H_AVG[102]
            - 0.008034102 * h[30] / H_AVG[30]
            + 0.005688453 * h[35] / H_AVG[35]
            + 0.004506522 * h[4] / H_AVG[4]
            + 0.004324086 * h[28] / H_AVG[28]
            - 0.001722248 * h[97] / H_AVG[97]
            - 3.153371e-05 * h[82] / H_AVG[82]
            - 1.342946e-05 * h[57] / H_AVG[57]
            + 1.198833e-05 * h[5] / H_AVG[5]
            - 5.512163e-06 * h[25] / H_AVG[25]
            + 3.69468e-06 * h[2] / H_AVG[2]
            + 1.009835e-06 * h[107] / H_AVG[107]
            + 5.967346e-07 * h[18] / H_AVG[18]
            - 5.006267e-07 * h[20] / H_AVG[20]
            + 4.99086e-07 * h[24] / H_AVG[24]
            - 4.877058e-07 * h[80] / H_AVG[80]
            + 4.632762e-07 * h[50] / H_AVG[50]
            - 4.130678e-07 * h[116] / H_AVG[116]
            - 3.68301e-07 * h[63] / H_AVG[63]
            - 3.586566e-07 * h[47] / H_AVG[47]
            + 3.475244e-07 * h[96] / H_AVG[96]
            - 3.436799e-07 * h[88] / H_AVG[88]
            + 3.417471e-07 * h[125] / H_AVG[125]
            + 2.874003e-07 * h[95] / H_AVG[95]
            + 2.828431e-07 * h[37] / H_AVG[37]
            - 2.723867e-07 * h[101] / H_AVG[101]
            - 2.293427e-07 * h[58] / H_AVG[58]
            - 2.131079e-07 * h[113] / H_AVG[113]
            + 2.057441e-07 * h[45] / H_AVG[45]
            - 1.899041e-07 * h[112] / H_AVG[112]
            + 1.898292e-07 * h[94] / H_AVG[94]
            + 1.481603e-07 * h[126] / H_AVG[126]
            - 1.451148e-07 * h[99] / H_AVG[99]
            + 1.425161e-07 * h[73] / H_AVG[73]
            - 1.413824e-07 * h[0] / H_AVG[0]
            + 1.32273e-07 * h[6] / H_AVG[6]
            + 1.278907e-07 * h[49] / H_AVG[49]
            - 1.250482e-07 * h[114] / H_AVG[114]
            + 1.24469e-07 * h[77] / H_AVG[77]
            - 1.192212e-07 * h[72] / H_AVG[72]
            + 1.183897e-07 * h[29] / H_AVG[29]
            - 1.123115e-07 * h[83] / H_AVG[83]
            - 1.118023e-07 * h[12] / H_AVG[12]
            - 1.05874e-07 * h[44] / H_AVG[44]
            + 1.020859e-07 * h[19] / H_AVG[19]
            - 9.93956e-08 * h[100] / H_AVG[100]
            - 9.666594e-08 * h[65] / H_AVG[65]
            - 9.229571e-08 * h[64] / H_AVG[64]
            - 8.991768e-08 * h[106] / H_AVG[106]
            - 8.65346e-08 * h[74] / H_AVG[74]
            - 8.467028e-08 * h[7] / H_AVG[7]
            + 8.307958e-08 * h[90] / H_AVG[90]
            + 8.08283e-08 * h[66] / H_AVG[66]
            - 7.658162e-08 * h[71] / H_AVG[71]
            - 5.804626e-08 * h[17] / H_AVG[17]
            + 5.403907e-08 * h[53] / H_AVG[53]
            - 4.698611e-08 * h[23] / H_AVG[23]
            - 4.003055e-08 * h[67] / H_AVG[67]
            - 3.666156e-08 * h[55] / H_AVG[55]
            - 3.433269e-08 * h[123] / H_AVG[123]
            - 3.35391e-08 * h[109] / H_AVG[109]
            - 3.035922e-08 * h[92] / H_AVG[92]
            - 2.754978e-08 * h[15] / H_AVG[15]
            - 2.695284e-08 * h[46] / H_AVG[46]
            - 2.594723e-08 * h[8] / H_AVG[8]
            - 2.505201e-08 * h[14] / H_AVG[14]
            + 2.466393e-08 * h[59] / H_AVG[59]
            - 2.453621e-08 * h[41] / H_AVG[41]
            - 2.43447e-08 * h[39] / H_AVG[39]
            + 2.202013e-08 * h[16] / H_AVG[16]
            - 1.748869e-08 * h[110] / H_AVG[110]
            - 1.710357e-08 * h[85] / H_AVG[85]
            + 1.647629e-08 * h[119] / H_AVG[119]
            - 1.406764e-08 * h[89] / H_AVG[89]
            - 1.344959e-08 * h[60] / H_AVG[60]
            + 1.307383e-08 * h[43] / H_AVG[43]
            - 1.300336e-08 * h[105] / H_AVG[105]
            - 1.278251e-08 * h[91] / H_AVG[91]
            - 1.133431e-08 * h[13] / H_AVG[13]
            - 1.058664e-08 * h[10] / H_AVG[10]
            + 1.000245e-08 * h[48] / H_AVG[48]
            - 9.757511e-09 * h[84] / H_AVG[84]
            - 9.426401e-09 * h[75] / H_AVG[75]
            + 8.551546e-09 * h[31] / H_AVG[31]
            + 8.051052e-09 * h[62] / H_AVG[62]
            + 7.360925e-09 * h[42] / H_AVG[42]
            + 6.809006e-09 * h[11] / H_AVG[11]
            - 6.46488e-09 * h[36] / H_AVG[36]
            - 6.114234e-09 * h[122] / H_AVG[122]
            - 5.792934e-09 * h[98] / H_AVG[98]
            + 5.573215e-09 * h[3] / H_AVG[3]
            + 5.40117e-09 * h[26] / H_AVG[26]
            - 5.291518e-09 * h[40] / H_AVG[40]
            - 5.178636e-09 * h[61] / H_AVG[61]
            + 4.941349e-09 * h[68] / H_AVG[68]
            - 4.110003e-09 * h[117] / H_AVG[117]
            - 3.91156e-09 * h[120] / H_AVG[120]
            + 3.443399e-09 * h[93] / H_AVG[93]
            - 3.441918e-09 * h[1] / H_AVG[1]
            - 3.420128e-09 * h[78] / H_AVG[78]
            - 3.418626e-09 * h[111] / H_AVG[111]
            + 2.932985e-09 * h[118] / H_AVG[118]
            + 2.160919e-09 * h[124] / H_AVG[124]
            - 2.10512e-09 * h[21] / H_AVG[21]
            - 1.640081e-09 * h[115] / H_AVG[115]
            - 1.505388e-09 * h[87] / H_AVG[87]
            - 1.382435e-09 * h[108] / H_AVG[108]
            + 1.217071e-09 * h[76] / H_AVG[76]
            - 1.031194e-09 * h[127] / H_AVG[127]
            - 7.771095e-10 * h[56] / H_AVG[56]
            + 7.690548e-10 * h[32] / H_AVG[32]
            - 5.863583e-11 * h[34] / H_AVG[34]
            - 3.966291e-11 * h[79] / H_AVG[79]
            - 1.11889e-11 * h[52] / H_AVG[52]
        ),
        0.0352555 + T[7] * (   # class Wqq
            - 0.3376064 * h[54] / H_AVG[54]
            + 0.0987931 * h[51] / H_AVG[51]
            + 0.07894107 * h[33] / H_AVG[33]
            - 0.07552532 * h[27] / H_AVG[27]
            + 0.06128613 * h[86] / H_AVG[86]
            - 0.060707 * h[38] / H_AVG[38]
            + 0.0555054 * h[22] / H_AVG[22]
            - 0.04080669 * h[121] / H_AVG[121]
            - 0.03921318 * h[4] / H_AVG[4]
            + 0.03679752 * h[81] / H_AVG[81]
            - 0.02523349 * h[104] / H_AVG[104]
            - 0.02062416 * h[103] / H_AVG[103]
            - 0.01865492 * h[70] / H_AVG[70]
            + 0.01621877 * h[97] / H_AVG[97]
            - 0.01139713 * h[102] / H_AVG[102]
            + 0.009221212 * h[69] / H_AVG[69]
            + 0.00661676 * h[9] / H_AVG[9]
            - 0.002297288 * h[35] / H_AVG[35]
            + 0.00228702 * h[28] / H_AVG[28]
            + 0.002188429 * h[30] / H_AVG[30]
            - 5.111027e-05 * h[57] / H_AVG[57]
            + 9.875127e-06 * h[5] / H_AVG[5]
            - 4.230136e-06 * h[25] / H_AVG[25]
            + 2.807845e-06 * h[2] / H_AVG[2]
            - 2.537652e-06 * h[82] / H_AVG[82]
            + 7.74664e-07 * h[107] / H_AVG[107]
            + 4.577168e-07 * h[18] / H_AVG[18]
            - 3.839314e-07 * h[20] / H_AVG[20]
            + 3.828456e-07 * h[24] / H_AVG[24]
            - 3.742205e-07 * h[80] / H_AVG[80]
            + 3.553457e-07 * h[50] / H_AVG[50]
            - 3.169346e-07 * h[116] / H_AVG[116]
            - 2.82546e-07 * h[63] / H_AVG[63]
            - 2.752227e-07 * h[47] / H_AVG[47]
            + 2.666285e-07 * h[96] / H_AVG[96]
            - 2.637281e-07 * h[88] / H_AVG[88]
            + 2.621545e-07 * h[125] / H_AVG[125]
            + 2.205294e-07 * h[95] / H_AVG[95]
            + 2.170333e-07 * h[37] / H_AVG[37]
            - 2.089537e-07 * h[101] / H_AVG[101]
            - 1.760245e-07 * h[58] / H_AVG[58]
            - 1.634914e-07 * h[113] / H_AVG[113]
            + 1.578827e-07 * h[45] / H_AVG[45]
            - 1.457438e-07 * h[112] / H_AVG[112]
            + 1.456543e-07 * h[94] / H_AVG[94]
            + 1.136719e-07 * h[126] / H_AVG[126]
            - 1.113005e-07 * h[99] / H_AVG[99]
            + 1.093049e-07 * h[73] / H_AVG[73]
            - 1.084872e-07 * h[0] / H_AVG[0]
            + 1.015311e-07 * h[6] / H_AVG[6]
            + 9.810884e-08 * h[49] / H_AVG[49]
            - 9.596212e-08 * h[114] / H_AVG[114]
            + 9.548182e-08 * h[77] / H_AVG[77]
            - 9.145413e-08 * h[72] / H_AVG[72]
            + 9.080967e-08 * h[29] / H_AVG[29]
            - 8.613572e-08 * h[83] / H_AVG[83]
            - 8.581061e-08 * h[12] / H_AVG[12]
            - 8.123031e-08 * h[44] / H_AVG[44]
            + 7.836453e-08 * h[19] / H_AVG[19]
            - 7.623509e-08 * h[100] / H_AVG[100]
            - 7.421109e-08 * h[65] / H_AVG[65]
            - 7.079893e-08 * h[64] / H_AVG[64]
            - 6.898934e-08 * h[106] / H_AVG[106]
            - 6.638344e-08 * h[74] / H_AVG[74]
            - 6.49435e-08 * h[7] / H_AVG[7]
            + 6.37245e-08 * h[90] / H_AVG[90]
            + 6.200579e-08 * h[66] / H_AVG[66]
            - 5.874768e-08 * h[71] / H_AVG[71]
            - 4.451857e-08 * h[17] / H_AVG[17]
            + 4.146422e-08 * h[53] / H_AVG[53]
            - 3.605661e-08 * h[23] / H_AVG[23]
            - 3.070886e-08 * h[67] / H_AVG[67]
            - 2.812173e-08 * h[55] / H_AVG[55]
            - 2.632301e-08 * h[123] / H_AVG[123]
            - 2.574157e-08 * h[109] / H_AVG[109]
            - 2.328357e-08 * h[92] / H_AVG[92]
            - 2.114469e-08 * h[15] / H_AVG[15]
            - 2.06897e-08 * h[46] / H_AVG[46]
            - 1.990725e-08 * h[8] / H_AVG[8]
            - 1.921324e-08 * h[14] / H_AVG[14]
            + 1.89254e-08 * h[59] / H_AVG[59]
            - 1.883059e-08 * h[41] / H_AVG[41]
            - 1.867531e-08 * h[39] / H_AVG[39]
            + 1.689855e-08 * h[16] / H_AVG[16]
            - 1.341377e-08 * h[110] / H_AVG[110]
            - 1.312125e-08 * h[85] / H_AVG[85]
            + 1.263669e-08 * h[119] / H_AVG[119]
            - 1.079143e-08 * h[89] / H_AVG[89]
            - 1.031674e-08 * h[60] / H_AVG[60]
            + 1.002774e-08 * h[43] / H_AVG[43]
            - 9.978915e-09 * h[105] / H_AVG[105]
            - 9.80118e-09 * h[91] / H_AVG[91]
            - 8.688712e-09 * h[13] / H_AVG[13]
            - 8.121101e-09 * h[10] / H_AVG[10]
            + 7.673597e-09 * h[48] / H_AVG[48]
            - 7.483407e-09 * h[84] / H_AVG[84]
            - 7.230319e-09 * h[75] / H_AVG[75]
            + 6.560476e-09 * h[31] / H_AVG[31]
            + 6.176456e-09 * h[62] / H_AVG[62]
            + 5.649938e-09 * h[42] / H_AVG[42]
            + 5.225268e-09 * h[11] / H_AVG[11]
            - 4.960569e-09 * h[36] / H_AVG[36]
            - 4.689985e-09 * h[122] / H_AVG[122]
            - 4.443878e-09 * h[98] / H_AVG[98]
            + 4.275647e-09 * h[3] / H_AVG[3]
            + 4.136585e-09 * h[26] / H_AVG[26]
            - 4.058414e-09 * h[40] / H_AVG[40]
            - 3.972934e-09 * h[61] / H_AVG[61]
            + 3.790426e-09 * h[68] / H_AVG[68]
            - 3.153018e-09 * h[117] / H_AVG[117]
            - 3.00266e-09 * h[120] / H_AVG[120]
            + 2.641578e-09 * h[93] / H_AVG[93]
            - 2.64036e-09 * h[1] / H_AVG[1]
            - 2.627871e-09 * h[78] / H_AVG[78]
            - 2.623143e-09 * h[111] / H_AVG[111]
            + 2.250908e-09 * h[118] / H_AVG[118]
            + 1.657942e-09 * h[124] / H_AVG[124]
            - 1.614874e-09 * h[21] / H_AVG[21]
            - 1.258547e-09 * h[115] / H_AVG[115]
            - 1.155196e-09 * h[87] / H_AVG[87]
            - 1.060389e-09 * h[108] / H_AVG[108]
            + 9.335656e-10 * h[76] / H_AVG[76]
            - 7.906827e-10 * h[127] / H_AVG[127]
            - 5.962847e-10 * h[56] / H_AVG[56]
            + 5.899489e-10 * h[32] / H_AVG[32]
            - 4.497894e-11 * h[34] / H_AVG[34]
            - 3.042391e-11 * h[79] / H_AVG[79]
            - 8.593406e-12 * h[52] / H_AVG[52]
        ),
        -0.5069618 + T[8] * (   # class Tbqq
            + 0.2399979 * h[103] / H_AVG[103]
            - 0.1655514 * h[121] / H_AVG[121]
            - 0.1347066 * h[51] / H_AVG[51]
            - 0.08399312 * h[86] / H_AVG[86]
            + 0.06556269 * h[54] / H_AVG[54]
            + 0.0633135 * h[22] / H_AVG[22]
            - 0.06034543 * h[38] / H_AVG[38]
            + 0.04787757 * h[33] / H_AVG[33]
            - 0.03953543 * h[104] / H_AVG[104]
            + 0.03161574 * h[27] / H_AVG[27]
            + 0.02688799 * h[9] / H_AVG[9]
            + 0.01350528 * h[70] / H_AVG[70]
            - 0.00758606 * h[4] / H_AVG[4]
            + 0.005527348 * h[81] / H_AVG[81]
            - 0.00380699 * h[102] / H_AVG[102]
            - 0.003651608 * h[69] / H_AVG[69]
            + 0.002657103 * h[97] / H_AVG[97]
            - 0.002107856 * h[30] / H_AVG[30]
            + 0.0009974839 * h[35] / H_AVG[35]
            + 0.0007274544 * h[28] / H_AVG[28]
            - 1.78829e-05 * h[57] / H_AVG[57]
            - 9.500147e-06 * h[82] / H_AVG[82]
            + 7.361207e-06 * h[5] / H_AVG[5]
            - 2.910652e-06 * h[25] / H_AVG[25]
            + 1.938621e-06 * h[2] / H_AVG[2]
            + 5.33372e-07 * h[107] / H_AVG[107]
            + 3.150895e-07 * h[18] / H_AVG[18]
            - 2.643739e-07 * h[20] / H_AVG[20]
            + 2.635674e-07 * h[24] / H_AVG[24]
            - 2.576095e-07 * h[80] / H_AVG[80]
            + 2.446434e-07 * h[50] / H_AVG[50]
            - 2.181202e-07 * h[116] / H_AVG[116]
            - 1.945261e-07 * h[63] / H_AVG[63]
            - 1.894517e-07 * h[47] / H_AVG[47]
            + 1.835379e-07 * h[96] / H_AVG[96]
            - 1.815166e-07 * h[88] / H_AVG[88]
            + 1.804571e-07 * h[125] / H_AVG[125]
            + 1.517442e-07 * h[95] / H_AVG[95]
            + 1.494182e-07 * h[37] / H_AVG[37]
            - 1.438408e-07 * h[101] / H_AVG[101]
            - 1.21134e-07 * h[58] / H_AVG[58]
            - 1.125282e-07 * h[113] / H_AVG[113]
            + 1.086384e-07 * h[45] / H_AVG[45]
            - 1.003211e-07 * h[112] / H_AVG[112]
            + 1.002551e-07 * h[94] / H_AVG[94]
            + 7.823759e-08 * h[126] / H_AVG[126]
            - 7.662203e-08 * h[99] / H_AVG[99]
            + 7.526582e-08 * h[73] / H_AVG[73]
            - 7.463994e-08 * h[0] / H_AVG[0]
            + 6.988948e-08 * h[6] / H_AVG[6]
            + 6.75271e-08 * h[49] / H_AVG[49]
            - 6.601399e-08 * h[114] / H_AVG[114]
            + 6.570848e-08 * h[77] / H_AVG[77]
            - 6.28813e-08 * h[72] / H_AVG[72]
            + 6.252503e-08 * h[29] / H_AVG[29]
            - 5.928484e-08 * h[83] / H_AVG[83]
            - 5.904947e-08 * h[12] / H_AVG[12]
            - 5.590604e-08 * h[44] / H_AVG[44]
            + 5.39183e-08 * h[19] / H_AVG[19]
            - 5.24813e-08 * h[100] / H_AVG[100]
            - 5.108764e-08 * h[65] / H_AVG[65]
            - 4.872625e-08 * h[64] / H_AVG[64]
            - 4.751355e-08 * h[106] / H_AVG[106]
            - 4.569043e-08 * h[74] / H_AVG[74]
            - 4.471068e-08 * h[7] / H_AVG[7]
            + 4.387887e-08 * h[90] / H_AVG[90]
            + 4.2668e-08 * h[66] / H_AVG[66]
            - 4.042997e-08 * h[71] / H_AVG[71]
            - 3.065396e-08 * h[17] / H_AVG[17]
            + 2.854123e-08 * h[53] / H_AVG[53]
            - 2.482051e-08 * h[23] / H_AVG[23]
            - 2.113894e-08 * h[67] / H_AVG[67]
            - 1.935457e-08 * h[55] / H_AVG[55]
            - 1.812324e-08 * h[123] / H_AVG[123]
            - 1.77114e-08 * h[109] / H_AVG[109]
            - 1.60272e-08 * h[92] / H_AVG[92]
            - 1.454714e-08 * h[15] / H_AVG[15]
            - 1.423982e-08 * h[46] / H_AVG[46]
            - 1.370425e-08 * h[8] / H_AVG[8]
            - 1.322437e-08 * h[14] / H_AVG[14]
            + 1.302546e-08 * h[59] / H_AVG[59]
            - 1.295249e-08 * h[41] / H_AVG[41]
            - 1.285619e-08 * h[39] / H_AVG[39]
            + 1.162914e-08 * h[16] / H_AVG[16]
            - 9.234378e-09 * h[110] / H_AVG[110]
            - 9.032372e-09 * h[85] / H_AVG[85]
            + 8.698601e-09 * h[119] / H_AVG[119]
            - 7.43259e-09 * h[89] / H_AVG[89]
            - 7.103637e-09 * h[60] / H_AVG[60]
            + 6.903722e-09 * h[43] / H_AVG[43]
            - 6.866523e-09 * h[105] / H_AVG[105]
            - 6.748882e-09 * h[91] / H_AVG[91]
            - 5.986511e-09 * h[13] / H_AVG[13]
            - 5.590681e-09 * h[10] / H_AVG[10]
            + 5.282611e-09 * h[48] / H_AVG[48]
            - 5.149921e-09 * h[84] / H_AVG[84]
            - 4.977716e-09 * h[75] / H_AVG[75]
            + 4.515672e-09 * h[31] / H_AVG[31]
            + 4.250759e-09 * h[62] / H_AVG[62]
            + 3.886397e-09 * h[42] / H_AVG[42]
            + 3.59665e-09 * h[11] / H_AVG[11]
            - 3.414694e-09 * h[36] / H_AVG[36]
            - 3.228627e-09 * h[122] / H_AVG[122]
            - 3.059163e-09 * h[98] / H_AVG[98]
            + 2.942687e-09 * h[3] / H_AVG[3]
            + 2.849661e-09 * h[26] / H_AVG[26]
            - 2.793633e-09 * h[40] / H_AVG[40]
            - 2.734324e-09 * h[61] / H_AVG[61]
            + 2.609295e-09 * h[68] / H_AVG[68]
            - 2.16771e-09 * h[117] / H_AVG[117]
            - 2.065523e-09 * h[120] / H_AVG[120]
            + 1.818415e-09 * h[93] / H_AVG[93]
            - 1.817406e-09 * h[1] / H_AVG[1]
            - 1.813277e-09 * h[78] / H_AVG[78]
            - 1.805371e-09 * h[111] / H_AVG[111]
            + 1.548828e-09 * h[118] / H_AVG[118]
            + 1.140774e-09 * h[124] / H_AVG[124]
            - 1.111674e-09 * h[21] / H_AVG[21]
            - 8.662445e-10 * h[115] / H_AVG[115]
            - 7.949216e-10 * h[87] / H_AVG[87]
            - 7.299951e-10 * h[108] / H_AVG[108]
            + 6.426742e-10 * h[76] / H_AVG[76]
            - 5.445255e-10 * h[127] / H_AVG[127]
            - 4.103907e-10 * h[56] / H_AVG[56]
            + 4.061708e-10 * h[32] / H_AVG[32]
            - 3.096532e-11 * h[34] / H_AVG[34]
            - 2.094444e-11 * h[79] / H_AVG[79]
            - 5.955112e-12 * h[52] / H_AVG[52]
        ),
        -1.389862 + T[9] * (   # class Tbl
            + 0.20178 * h[103] / H_AVG[103]
            - 0.1989773 * h[54] / H_AVG[54]
            - 0.1302708 * h[33] / H_AVG[33]
            + 0.1110266 * h[121] / H_AVG[121]
            + 0.08447495 * h[27] / H_AVG[27]
            - 0.06385452 * h[51] / H_AVG[51]
            - 0.05630741 * h[22] / H_AVG[22]
            + 0.04952348 * h[97] / H_AVG[97]
            - 0.02408308 * h[104] / H_AVG[104]
            - 0.01751068 * h[70] / H_AVG[70]
            + 0.01517092 * h[4] / H_AVG[4]
            - 0.01309115 * h[81] / H_AVG[81]
            + 0.01198425 * h[86] / H_AVG[86]
            - 0.010155 * h[38] / H_AVG[38]
            - 0.00472869 * h[28] / H_AVG[28]
            + 0.00345052 * h[9] / H_AVG[9]
            + 0.002188571 * h[102] / H_AVG[102]
            - 0.0006688404 * h[30] / H_AVG[30]
            + 0.0006539163 * h[69] / H_AVG[69]
            - 3.335447e-05 * h[57] / H_AVG[57]
            + 2.317289e-05 * h[35] / H_AVG[35]
            + 1.962527e-05 * h[5] / H_AVG[5]
            - 1.45006e-05 * h[82] / H_AVG[82]
            - 2.383826e-06 * h[25] / H_AVG[25]
            + 1.595199e-06 * h[2] / H_AVG[2]
            + 4.36676e-07 * h[107] / H_AVG[107]
            + 2.579422e-07 * h[18] / H_AVG[18]
            - 2.163526e-07 * h[20] / H_AVG[20]
            + 2.157775e-07 * h[24] / H_AVG[24]
            - 2.109034e-07 * h[80] / H_AVG[80]
            + 2.002791e-07 * h[50] / H_AVG[50]
            - 1.785866e-07 * h[116] / H_AVG[116]
            - 1.592334e-07 * h[63] / H_AVG[63]
            - 1.551845e-07 * h[47] / H_AVG[47]
            + 1.502649e-07 * h[96] / H_AVG[96]
            - 1.486469e-07 * h[88] / H_AVG[88]
            + 1.477664e-07 * h[125] / H_AVG[125]
            + 1.242845e-07 * h[95] / H_AVG[95]
            + 1.223103e-07 * h[37] / H_AVG[37]
            - 1.177614e-07 * h[101] / H_AVG[101]
            - 9.918808e-08 * h[58] / H_AVG[58]
            - 9.216891e-08 * h[113] / H_AVG[113]
            + 8.89639e-08 * h[45] / H_AVG[45]
            - 8.212228e-08 * h[112] / H_AVG[112]
            + 8.210114e-08 * h[94] / H_AVG[94]
            + 6.408081e-08 * h[126] / H_AVG[126]
            - 6.274148e-08 * h[99] / H_AVG[99]
            + 6.159601e-08 * h[73] / H_AVG[73]
            - 6.112357e-08 * h[0] / H_AVG[0]
            + 5.720896e-08 * h[6] / H_AVG[6]
            + 5.528234e-08 * h[49] / H_AVG[49]
            - 5.405194e-08 * h[114] / H_AVG[114]
            + 5.379798e-08 * h[77] / H_AVG[77]
            - 5.158814e-08 * h[72] / H_AVG[72]
            + 5.118471e-08 * h[29] / H_AVG[29]
            - 4.853392e-08 * h[83] / H_AVG[83]
            - 4.835233e-08 * h[12] / H_AVG[12]
            - 4.577618e-08 * h[44] / H_AVG[44]
            + 4.415327e-08 * h[19] / H_AVG[19]
            - 4.296892e-08 * h[100] / H_AVG[100]
            - 4.186541e-08 * h[65] / H_AVG[65]
            - 3.988916e-08 * h[64] / H_AVG[64]
            - 3.88787e-08 * h[106] / H_AVG[106]
            - 3.741612e-08 * h[74] / H_AVG[74]
            - 3.659431e-08 * h[7] / H_AVG[7]
            + 3.592994e-08 * h[90] / H_AVG[90]
            + 3.49416e-08 * h[66] / H_AVG[66]
            - 3.311477e-08 * h[71] / H_AVG[71]
            - 2.509969e-08 * h[17] / H_AVG[17]
            + 2.337222e-08 * h[53] / H_AVG[53]
            - 2.031913e-08 * h[23] / H_AVG[23]
            - 1.73021e-08 * h[67] / H_AVG[67]
            - 1.584643e-08 * h[55] / H_AVG[55]
            - 1.48382e-08 * h[123] / H_AVG[123]
            - 1.450758e-08 * h[109] / H_AVG[109]
            - 1.312354e-08 * h[92] / H_AVG[92]
            - 1.191278e-08 * h[15] / H_AVG[15]
            - 1.166073e-08 * h[46] / H_AVG[46]
            - 1.122238e-08 * h[8] / H_AVG[8]
            - 1.082691e-08 * h[14] / H_AVG[14]
            + 1.066447e-08 * h[59] / H_AVG[59]
            - 1.060788e-08 * h[41] / H_AVG[41]
            - 1.052578e-08 * h[39] / H_AVG[39]
            + 9.521627e-09 * h[16] / H_AVG[16]
            - 7.558517e-09 * h[110] / H_AVG[110]
            - 7.400107e-09 * h[85] / H_AVG[85]
            + 7.125044e-09 * h[119] / H_AVG[119]
            - 6.083507e-09 * h[89] / H_AVG[89]
            - 5.815255e-09 * h[60] / H_AVG[60]
            + 5.655536e-09 * h[43] / H_AVG[43]
            - 5.624056e-09 * h[105] / H_AVG[105]
            - 5.523664e-09 * h[91] / H_AVG[91]
            - 4.887211e-09 * h[13] / H_AVG[13]
            - 4.57633e-09 * h[10] / H_AVG[10]
            + 4.326657e-09 * h[48] / H_AVG[48]
            - 4.218086e-09 * h[84] / H_AVG[84]
            - 4.075461e-09 * h[75] / H_AVG[75]
            + 3.698294e-09 * h[31] / H_AVG[31]
            + 3.481074e-09 * h[62] / H_AVG[62]
            + 3.183028e-09 * h[42] / H_AVG[42]
            + 2.943984e-09 * h[11] / H_AVG[11]
            - 2.796125e-09 * h[36] / H_AVG[36]
            - 2.642574e-09 * h[122] / H_AVG[122]
            - 2.503957e-09 * h[98] / H_AVG[98]
            + 2.408855e-09 * h[3] / H_AVG[3]
            + 2.331058e-09 * h[26] / H_AVG[26]
            - 2.287362e-09 * h[40] / H_AVG[40]
            - 2.240192e-09 * h[61] / H_AVG[61]
            + 2.137205e-09 * h[68] / H_AVG[68]
            - 1.774308e-09 * h[117] / H_AVG[117]
            - 1.691858e-09 * h[120] / H_AVG[120]
            + 1.48903e-09 * h[93] / H_AVG[93]
            - 1.487763e-09 * h[1] / H_AVG[1]
            - 1.48582e-09 * h[78] / H_AVG[78]
            - 1.478803e-09 * h[111] / H_AVG[111]
            + 1.268322e-09 * h[118] / H_AVG[118]
            + 9.344216e-10 * h[124] / H_AVG[124]
            - 9.103789e-10 * h[21] / H_AVG[21]
            - 7.092427e-10 * h[115] / H_AVG[115]
            - 6.510398e-10 * h[87] / H_AVG[87]
            - 5.978328e-10 * h[108] / H_AVG[108]
            + 5.263541e-10 * h[76] / H_AVG[76]
            - 4.457913e-10 * h[127] / H_AVG[127]
            - 3.359898e-10 * h[56] / H_AVG[56]
            + 3.326053e-10 * h[32] / H_AVG[32]
            - 2.535519e-11 * h[34] / H_AVG[34]
            - 1.714718e-11 * h[79] / H_AVG[79]
            - 4.855399e-12 * h[52] / H_AVG[52]
        ),
    ]]


def classify(pt, eta, phi, energy, jet_pt, jet_eta, jet_energy):
    s = logits(class_token(quantities(pt, eta, phi, energy, jet_pt, jet_eta, jet_energy)))
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
    c, s, p = classify(pt, eta, phi, energy, jet_pt, jet_eta, jet_energy)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
