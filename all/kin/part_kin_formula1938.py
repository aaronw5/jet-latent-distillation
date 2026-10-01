"""Particle Transformer (ParT) jet tagger, JetClass, input features 'kin': tuned on the network's predictions (from 100 if-statements per neuron; the smallest without loss on the validation jets), as if-statements.

Input:  the particles of a jet (up to 128), hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
        energy: each particle's energy [GeV]; jet_pt, jet_eta, jet_energy: the jet's pT [GeV], pseudorapidity, energy [GeV].
Output: the class (QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl), the 10 logits and the 10 probabilities.

1. quantities():  physics quantities of the jet.
2. class_token(): the 128 numbers the network's last layer reads (its class token after the last LayerNorm), each
                  built from if-statements.
3. logits():      the network's own last layer: the 128 numbers multiplied by the weights, plus the biases.
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
    z = 2.322344e-05
    return z


def neuron_1(Q):
    z = 4.761036e-06
    return z


def neuron_2(Q):
    z = -0.000625311
    return z


def neuron_3(Q):
    z = -4.991911e-06
    return z


def neuron_4(Q):
    z = -0.1374526
    if Q.sum_z_dr < 0.04863552:
        z += 35.23424 * Q.sum_z_dr - 1.850291
    if 0.04863552 <= Q.sum_z_dr < 0.07290954:
        z += 5.629695 * Q.sum_z_dr - 0.4104585
    if Q.M3_b2 < 0.02497878:
        z += -6.492488 * Q.M3_b2 + 0.1621744
    if Q.N2 < 0.1824346:
        z += 1.192687 * Q.N2 - 0.2175874
    if Q.m01 < 17.3274:
        z += 0.03301654 * Q.m01 - 0.5720909
    if Q.pt1_dr01 < 15.07073:
        z += -0.01073185 * Q.pt1_dr01 + 0.1617367
    if Q.mass_top40 < 83.57316:
        z += -0.01155849 * Q.mass_top40 + 0.966243
    if 83.57316 <= Q.mass_top40 < 92.68149:
        z += -2.894178e-05 * Q.mass_top40 + 0.002682367
    if Q.mass < 95.14961:
        z += -0.008246507 * Q.mass + 1.418674
    if 95.14961 <= Q.mass < 149.0507:
        z += -0.0117627 * Q.mass + 1.753239
    if Q.sum_z_dr2_top10 < 0.03484839:
        z += 8.069412 * Q.sum_z_dr2_top10 - 0.281206
    if Q.sj2_mass1 >= 67.99453:
        z += 0.002944005 * Q.sj2_mass1 - 0.2001762
    if Q.mass_top50 < 104.3234:
        z += 0.01165431 * Q.mass_top50 - 1.215818
    if Q.n_dr_0p4_up < 4.0:
        z += -0.02793368 * Q.n_dr_0p4_up + 0.1117347
    if Q.m012 < 1.87753:
        z += -0.08838654 * Q.m012 + 0.1659484
    if 57.33311 <= Q.m012 < 69.67489:
        z += -0.003480695 * Q.m012 + 0.1995591
    if Q.m012 >= 69.67489:
        z += -0.009480179 * Q.m012 + 0.6175725
    if Q.M2_b05 < 0.1301769:
        z += 8.499855 * Q.M2_b05 - 1.106485
    if Q.sj3_pair_mass_max < 93.87663:
        z += 0.00245232 * Q.sj3_pair_mass_max - 0.5276108
    if 93.87663 <= Q.sj3_pair_mass_max < 114.7848:
        z += 0.01408057 * Q.sj3_pair_mass_max - 1.619232
    if 114.7848 <= Q.sj3_pair_mass_max < 142.9952:
        z += 0.0001062007 * Q.sj3_pair_mass_max - 0.0151862
    if Q.sj3_pairmax_over_m >= 0.6888277:
        z += -0.592126 * Q.sj3_pairmax_over_m + 0.4078728
    if Q.pair_mean_lnz < -1.332584:
        z += -0.2334779 * Q.pair_mean_lnz - 0.3111289
    if Q.jet_abs_eta >= 0.5325716:
        z += -0.1402618 * Q.jet_abs_eta + 0.07469944
    if Q.dr01 < 0.0259406:
        z += -2.764001 * Q.dr01 + 0.4045403
    if 0.0259406 <= Q.dr01 < 0.3258728:
        z += -1.109719 * Q.dr01 + 0.3616272
    if Q.ecf_g32 < 0.003001458:
        z += 9.057846 * Q.ecf_g32 - 0.05173723
    if 0.003001458 <= Q.ecf_g32 < 0.003965728:
        z += 25.4602 * Q.ecf_g32 - 0.1009682
    if Q.dr02 < 0.02982432:
        z += -2.835688 * Q.dr02 + 0.08457246
    if Q.n_lund >= 10.0:
        z += 0.001844278 * Q.n_lund - 0.01844278
    if 0.5076533 <= Q.sj2_dr < 0.6647889:
        z += 0.2441594 * Q.sj2_dr - 0.1239484
    if Q.sj2_dr >= 0.6647889:
        z += 0.4771779 * Q.sj2_dr - 0.2788564
    if Q.sj3_z1 >= 0.4593088:
        z += -0.01307573 * Q.sj3_z1 + 0.006005798
    if Q.sum_z_dr2_top3 < 0.004077497:
        z += -10.73283 * Q.sum_z_dr2_top3 + 0.04376309
    if Q.pair_mean_lnm2 < 2.09826:
        z += 0.005857996 * Q.pair_mean_lnm2 - 0.0122916
    if Q.pair_mean_lnm2 >= 4.787589:
        z += 0.159582 * Q.pair_mean_lnm2 - 0.7640129
    if Q.lund_max_lndelta >= -0.6897565:
        z += -0.373907 * Q.lund_max_lndelta - 0.2579048
    if Q.z_top3_slots >= 0.7102909:
        z += 1.031529 * Q.z_top3_slots - 0.7326855
    if 0.1179386 <= Q.D3 < 0.3195268:
        z += 0.2923546 * Q.D3 - 0.0344799
    if Q.D3 >= 0.3195268:
        z += -0.02398264 * Q.D3 + 0.0665983
    if Q.n_pairs_kt_above_3 < 16.0:
        z += -0.0207617 * Q.n_pairs_kt_above_3 + 0.3321872
    if Q.lne_29 >= 1.660458:
        z += 0.04273129 * Q.lne_29 - 0.0709535
    if Q.n_pairs_kt_above_1 < 411.0:
        z += 0.0004682557 * Q.n_pairs_kt_above_1 - 0.1924531
    if 68.11898 <= Q.sj3_pair_mass_min < 80.02563:
        z += 0.01148976 * Q.sj3_pair_mass_min - 0.7826705
    if Q.sj3_pair_mass_min >= 80.02563:
        z += -0.002084903 * Q.sj3_pair_mass_min + 0.3036501
    if Q.N2_b05 < 0.4353632:
        z += 0.3055065 * Q.N2_b05 - 0.1330063
    if Q.max_pair_mass >= 45.28191:
        z += 0.01335273 * Q.max_pair_mass - 0.6046371
    if Q.lund_max_lnkt < 2.961008:
        z += 0.3068279 * Q.lund_max_lnkt - 0.9085201
    if Q.M2 < 0.04276413:
        z += 11.79297 * Q.M2 - 0.5043163
    if Q.z_top2_slots >= 0.3197804:
        z += 0.162192 * Q.z_top2_slots - 0.05186584
    if Q.tau32_b2 >= 0.5838518:
        z += -0.4012349 * Q.tau32_b2 + 0.2342617
    if Q.sj3_mass3 < 2.151069:
        z += -0.1181853 * Q.sj3_mass3 + 0.2542247
    if Q.sj3_dr_max < 0.3150493:
        z += 0.8460102 * Q.sj3_dr_max - 0.2665349
    if Q.sum_e < 732.7973:
        z += 0.0003199183 * Q.sum_e - 0.2344352
    if Q.D2_b05 < 1.208316:
        z += -1.174914 * Q.D2_b05 + 1.419668
    if Q.tau21 >= 0.3552591:
        z += -0.05949989 * Q.tau21 + 0.02113788
    if Q.sum_pt_top10 < 428.9828:
        z += 0.0003209946 * Q.sum_pt_top10 - 0.1377012
    if Q.M3 < 0.04495465:
        z += 13.88871 * Q.M3 - 0.6243621
    if Q.N3_b05 < 0.7045747:
        z += -0.8760484 * Q.N3_b05 + 0.6172416
    if Q.tau2 < 0.05997694:
        z += 1.769929 * Q.tau2 - 0.1061549
    if Q.mass < 149.0507 and Q.tau32 < 0.5498426:
        z += 0.02397042 * (149.0507 - Q.mass) * (0.5498426 - Q.tau32)
    if Q.mass < 149.0507 and Q.pt2_over_pt0 < 0.4242331:
        z += 0.001235201 * (149.0507 - Q.mass) * (0.4242331 - Q.pt2_over_pt0)
    if Q.sj3_pair_mass_max < 114.7848 and Q.pair_max_lnkt > 2.687142:
        z += 0.00122302 * (114.7848 - Q.sj3_pair_mass_max) * (Q.pair_max_lnkt - 2.687142)
    if Q.pair_mean_lnz < -1.332584 and Q.ecf_g41 > 0.0001767103:
        z += 655.4116 * (-1.332584 - Q.pair_mean_lnz) * (Q.ecf_g41 - 0.0001767103)
    if Q.M3_b2 < 0.02497878 and Q.e3_b2 < 0.0007909605:
        z += -772.4284 * (0.02497878 - Q.M3_b2) * (0.0007909605 - Q.e3_b2)
    if Q.m01 < 17.3274 and Q.N2_b2 < 0.3166168:
        z += 0.01889073 * (17.3274 - Q.m01) * (0.3166168 - Q.N2_b2)
    if Q.m012 > 69.67489 and Q.sj4_dr_min < 0.09604646:
        z += -0.1793246 * (Q.m012 - 69.67489) * (0.09604646 - Q.sj4_dr_min)
    if Q.sum_z_dr < 0.07290954 and Q.C3_b2 < 0.003536765:
        z += 166.6201 * (0.07290954 - Q.sum_z_dr) * (0.003536765 - Q.C3_b2)
    if Q.mass < 149.0507 and Q.lund_max_lndelta > -1.233804:
        z += 0.00721984 * (149.0507 - Q.mass) * (Q.lund_max_lndelta - -1.233804)
    if Q.m012 > 69.67489 and Q.sj2_mass2 > 1.851735:
        z += -0.0001414827 * (Q.m012 - 69.67489) * (Q.sj2_mass2 - 1.851735)
    if Q.N2 < 0.1824346 and Q.dr_13 < 0.2460229:
        z += 6.046438 * (0.1824346 - Q.N2) * (0.2460229 - Q.dr_13)
    if Q.pt1_dr01 < 15.07073 and Q.n_lund_kt_above_1 < 6.0:
        z += -0.0006103758 * (15.07073 - Q.pt1_dr01) * (6.0 - Q.n_lund_kt_above_1)
    if Q.pair_mean_lnm2 < 2.09826 and Q.lund1_lnz > -3.742431:
        z += 0.001240053 * (2.09826 - Q.pair_mean_lnm2) * (Q.lund1_lnz - -3.742431)
    if Q.mass_top50 < 104.3234 and Q.tau43 < 0.6363796:
        z += 0.0306958 * (104.3234 - Q.mass_top50) * (0.6363796 - Q.tau43)
    if Q.pt1_dr01 < 15.07073 and Q.tau43 < 0.8939856:
        z += -0.01091949 * (15.07073 - Q.pt1_dr01) * (0.8939856 - Q.tau43)
    if Q.lund_max_lndelta > -0.6897565 and Q.e4 > 1.2913e-08:
        z += 5135.206 * (Q.lund_max_lndelta - -0.6897565) * (Q.e4 - 1.2913e-08)
    if Q.m012 > 57.33311 and Q.sj2_mass2 < 8.588303:
        z += -0.003251137 * (Q.m012 - 57.33311) * (8.588303 - Q.sj2_mass2)
    if Q.dr01 < 0.0259406 and Q.lund3_lnkt > 3.765889:
        z += -5.219688 * (0.0259406 - Q.dr01) * (Q.lund3_lnkt - 3.765889)
    if Q.sj3_pair_mass_max < 114.7848 and Q.mratio_max_012 > 0.7784366:
        z += -0.01127912 * (114.7848 - Q.sj3_pair_mass_max) * (Q.mratio_max_012 - 0.7784366)
    if Q.sj3_z1 > 0.4593088 and Q.dr_13 < 0.5444444:
        z += 0.1824652 * (Q.sj3_z1 - 0.4593088) * (0.5444444 - Q.dr_13)
    if Q.z_top2_slots > 0.3197804 and Q.eta_23 < -0.01504822:
        z += -0.03636073 * (Q.z_top2_slots - 0.3197804) * (-0.01504822 - Q.eta_23)
    if Q.sj2_dr > 0.6647889 and Q.phi_51 < 0.0:
        z += -1.697793 * (Q.sj2_dr - 0.6647889) * (0.0 - Q.phi_51)
    if Q.sj3_mass3 < 2.151069 and Q.sj4_zsoft < 0.03350185:
        z += -4.116889 * (2.151069 - Q.sj3_mass3) * (0.03350185 - Q.sj4_zsoft)
    if Q.M2_b05 < 0.1301769 and Q.n_dr_0p1_0p2 < 9.0:
        z += -0.5402483 * (0.1301769 - Q.M2_b05) * (9.0 - Q.n_dr_0p1_0p2)
    if Q.sum_e < 732.7973 and Q.C3_b2 > 2.435109e-06:
        z += 0.001815272 * (732.7973 - Q.sum_e) * (Q.C3_b2 - 2.435109e-06)
    if Q.m01 < 17.3274 and Q.eta_17 < 0.08435059:
        z += -0.006331929 * (17.3274 - Q.m01) * (0.08435059 - Q.eta_17)
    if Q.m012 > 57.33311 and Q.sj3_dr13 < 0.1115643:
        z += 0.1413971 * (Q.m012 - 57.33311) * (0.1115643 - Q.sj3_dr13)
    if Q.lund_max_lndelta > -0.6897565 and Q.lund3_lnz > -6.170656:
        z += 0.08680992 * (Q.lund_max_lndelta - -0.6897565) * (Q.lund3_lnz - -6.170656)
    if Q.n_dr_0p4_up < 4.0 and Q.phi_24 > 0.05783997:
        z += 0.1012258 * (4.0 - Q.n_dr_0p4_up) * (Q.phi_24 - 0.05783997)
    if Q.sum_e < 732.7973 and Q.sj2_mass2 < 1.851735:
        z += -7.318889e-05 * (732.7973 - Q.sum_e) * (1.851735 - Q.sj2_mass2)
    if Q.sj3_mass3 < 2.151069 and Q.sj2_mass2 < 3.928781:
        z += -0.02990078 * (2.151069 - Q.sj3_mass3) * (3.928781 - Q.sj2_mass2)
    if Q.sj3_pairmax_over_m > 0.6888277 and Q.sj2_mass2 > 33.1786:
        z += 0.05199524 * (Q.sj3_pairmax_over_m - 0.6888277) * (Q.sj2_mass2 - 33.1786)
    if Q.sj2_dr > 0.5076533 and Q.lund2_lnz > -4.435851:
        z += 0.1978518 * (Q.sj2_dr - 0.5076533) * (Q.lund2_lnz - -4.435851)
    if Q.max_pair_mass > 45.28191 and Q.lund1_lnkt < 0.7613542:
        z += -0.00456804 * (Q.max_pair_mass - 45.28191) * (0.7613542 - Q.lund1_lnkt)
    if Q.tau21 > 0.3552591 and Q.phi_13 < -0.3098145:
        z += 0.2382828 * (Q.tau21 - 0.3552591) * (-0.3098145 - Q.phi_13)
    if Q.m012 > 69.67489 and Q.lund3_lnz < -3.686417:
        z += -0.001026339 * (Q.m012 - 69.67489) * (-3.686417 - Q.lund3_lnz)
    if Q.max_pair_mass > 45.28191 and Q.lne_4 < 3.175446:
        z += 0.007880338 * (Q.max_pair_mass - 45.28191) * (3.175446 - Q.lne_4)
    if Q.M2 < 0.04276413 and Q.lnpt_6 > 2.807141:
        z += -35.62482 * (0.04276413 - Q.M2) * (Q.lnpt_6 - 2.807141)
    if Q.D3 > 0.3195268 and Q.eta_40 < 0.0:
        z += 0.465304 * (Q.D3 - 0.3195268) * (0.0 - Q.eta_40)
    if Q.D2_b05 < 1.208316 and Q.sj4_dr_min < 0.0657426:
        z += 20.84191 * (1.208316 - Q.D2_b05) * (0.0657426 - Q.sj4_dr_min)
    return z


def neuron_5(Q):
    z = -0.0006186547
    return z


def neuron_6(Q):
    z = -2.169769e-05
    return z


def neuron_7(Q):
    z = 1.171535e-05
    return z


def neuron_8(Q):
    z = 1.341724e-05
    return z


def neuron_9(Q):
    z = 0.2151712
    if Q.N2 < 0.3829918:
        z += 1.559077 * Q.N2 - 0.5971136
    if Q.tau21_b2 < 0.2701529:
        z += -3.453478 * Q.tau21_b2 + 0.932967
    if Q.C2_b05 < 0.1854236:
        z += -3.520655 * Q.C2_b05 + 0.6528127
    if Q.C2_b05 >= 0.3533901:
        z += 2.936972 * Q.C2_b05 - 1.037897
    if 0.01467699 <= Q.tau5 < 0.03167074:
        z += -8.6082 * Q.tau5 + 0.1263424
    if Q.tau5 >= 0.03167074:
        z += 0.4464483 * Q.tau5 - 0.160425
    if Q.mass < 95.14961:
        z += 0.0101036 * Q.mass - 1.215341
    if 95.14961 <= Q.mass < 117.4867:
        z += -0.002003846 * Q.mass - 0.06332293
    if 117.4867 <= Q.mass < 164.4374:
        z += 0.006363007 * Q.mass - 1.046317
    if Q.mass_over_sum_pt_sq >= 0.04296187:
        z += -2.356661 * Q.mass_over_sum_pt_sq + 0.1012466
    if -2.42577 <= Q.pair_mean_lndelta < -2.049856:
        z += -0.1051232 * Q.pair_mean_lndelta - 0.2550047
    if -2.049856 <= Q.pair_mean_lndelta < -1.629715:
        z += -0.1477962 * Q.pair_mean_lndelta - 0.3424783
    if Q.pair_mean_lndelta >= -1.629715:
        z += -0.5630996 * Q.pair_mean_lndelta - 1.019305
    if 0.302405 <= Q.N2_b05 < 0.4265629:
        z += 2.753016 * Q.N2_b05 - 0.8325259
    if Q.N2_b05 >= 0.4265629:
        z += -0.3560166 * Q.N2_b05 + 0.4936724
    if Q.n_particles >= 26.0:
        z += -0.006980962 * Q.n_particles + 0.181505
    if Q.ecf_g42 >= 2.829762e-05:
        z += 1087.131 * Q.ecf_g42 - 0.0307632
    if Q.pt_dispersion < 0.5969141:
        z += -0.02582591 * Q.pt_dispersion + 0.01541585
    if Q.sj3_pair_mass_max < 109.8208:
        z += -0.008501555 * Q.sj3_pair_mass_max + 1.008132
    if 109.8208 <= Q.sj3_pair_mass_max < 120.6471:
        z += -0.006879947 * Q.sj3_pair_mass_max + 0.830046
    if Q.sj3_pairmax_over_m < 0.9148508:
        z += 2.009174 * Q.sj3_pairmax_over_m - 1.838094
    if Q.mass_top30 >= 162.7874:
        z += 0.008884308 * Q.mass_top30 - 1.446253
    if Q.sd_mass < 94.55722:
        z += 0.0002675249 * Q.sd_mass - 0.04706654
    if 94.55722 <= Q.sd_mass < 111.701:
        z += -0.002837348 * Q.sd_mass + 0.2465216
    if 111.701 <= Q.sd_mass < 127.8042:
        z += 0.001711473 * Q.sd_mass - 0.261586
    if 127.8042 <= Q.sd_mass < 135.3031:
        z += 0.01331495 * Q.sd_mass - 1.744559
    if 135.3031 <= Q.sd_mass < 175.9333:
        z += -0.00331849 * Q.sd_mass + 0.5059966
    if Q.sd_mass >= 175.9333:
        z += -0.003586014 * Q.sd_mass + 0.5530631
    if Q.mass_top20 >= 130.7093:
        z += -0.01292006 * Q.mass_top20 + 1.688771
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.005042376 * Q.n_pairs_kt_above_1 + 0.4033901
    if Q.lnpt_6 >= 2.880777:
        z += 0.09976768 * Q.lnpt_6 - 0.2874085
    if Q.n_dr_0p4_up < 5.0:
        z += 0.006748754 * Q.n_dr_0p4_up - 0.03374377
    if Q.max_dr < 0.6079631:
        z += 0.7903226 * Q.max_dr - 0.480487
    if Q.n_lund_kt_above_1 < 2.0:
        z += -0.05308445 * Q.n_lund_kt_above_1 + 0.1061689
    if Q.dr12 < 0.2116473:
        z += 0.4344858 * Q.dr12 - 0.09195775
    if Q.M2 < 0.03465331:
        z += 10.21176 * Q.M2 - 0.3538713
    if Q.tau3 < 0.03086713:
        z += 15.52996 * Q.tau3 - 0.4793654
    if Q.sum_z_dr2_top3 < 0.01520104:
        z += -10.00974 * Q.sum_z_dr2_top3 + 0.1521584
    if Q.sj3_dr_min < 0.1259759:
        z += 3.844886 * Q.sj3_dr_min - 0.4843631
    if 3.22013 <= Q.lund_max_lnkt < 4.323396:
        z += 0.05000716 * Q.lund_max_lnkt - 0.1610296
    if Q.lund_max_lnkt >= 4.323396:
        z += -0.09083004 * Q.lund_max_lnkt + 0.4478655
    if Q.lund_max_lndelta >= -0.4168051:
        z += 0.05075727 * Q.lund_max_lndelta + 0.02115589
    if Q.eta_1 < -0.2019043:
        z += 0.2492996 * Q.eta_1 + 0.05033467
    if Q.pair_max_lnkt < 2.43449:
        z += -0.09753107 * Q.pair_max_lnkt + 0.2374384
    if Q.dr01 < 0.0153855:
        z += -12.17605 * Q.dr01 + 0.1873347
    if Q.pair_mean_lnm2 >= 1.827583:
        z += 0.1677258 * Q.pair_mean_lnm2 - 0.3065328
    if Q.m01 < 0.9123375:
        z += 0.3198194 * Q.m01 - 0.2917832
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.02543163 * Q.n_dr_0p2_0p4 - 0.1780214
    if Q.sj3_dr13 < 0.2317874:
        z += -0.3126799 * Q.sj3_dr13 + 0.07247524
    if Q.tau21 >= 0.4790917:
        z += 0.8412264 * Q.tau21 - 0.4030246
    if Q.sj3_mass2 < 1.824785:
        z += 0.247061 * Q.sj3_mass2 - 0.4508333
    if 92.5542 <= Q.sj4_pair_mass_max < 115.5142:
        z += -0.0001740889 * Q.sj4_pair_mass_max + 0.01611266
    if Q.sj4_pair_mass_max >= 115.5142:
        z += 0.008263772 * Q.sj4_pair_mass_max - 0.9585801
    if Q.sd_rg >= 0.4449101:
        z += 0.02126959 * Q.sd_rg - 0.009463058
    if Q.n_pairs_kt_above_3 < 88.0:
        z += 0.004789151 * Q.n_pairs_kt_above_3 - 0.4214453
    if Q.e3_b05 >= 0.0061892:
        z += 13.56831 * Q.e3_b05 - 0.08397698
    if Q.lne_0 >= 5.157617:
        z += -0.1146692 * Q.lne_0 + 0.59142
    if Q.tau2 < 0.1193588:
        z += -0.8080972 * Q.tau2 + 0.09645353
    if Q.N2 < 0.3829918 and Q.lam1 > 0.05242126:
        z += -30.52884 * (0.3829918 - Q.N2) * (Q.lam1 - 0.05242126)
    if Q.mass < 95.14961 and Q.lnerel_0 > -1.362996:
        z += 0.0104249 * (95.14961 - Q.mass) * (Q.lnerel_0 - -1.362996)
    if Q.tau21_b2 < 0.2701529 and Q.lne_6 > 2.784239:
        z += -0.688181 * (0.2701529 - Q.tau21_b2) * (Q.lne_6 - 2.784239)
    if Q.tau21_b2 < 0.2701529 and Q.max_pair_mass > 45.28191:
        z += -0.006393412 * (0.2701529 - Q.tau21_b2) * (Q.max_pair_mass - 45.28191)
    if Q.pt_dispersion < 0.5969141 and Q.sum_e < 1651.766:
        z += -0.0001897742 * (0.5969141 - Q.pt_dispersion) * (1651.766 - Q.sum_e)
    if Q.pair_mean_lndelta > -2.049856 and Q.lund_max_lndelta < -0.529318:
        z += -0.1482544 * (Q.pair_mean_lndelta - -2.049856) * (-0.529318 - Q.lund_max_lndelta)
    if Q.mass < 164.4374 and Q.sj3_mass1 > 13.49356:
        z += 9.394313e-07 * (164.4374 - Q.mass) * (Q.sj3_mass1 - 13.49356)
    if Q.mass_over_sum_pt_sq > 0.04296187 and Q.lund_max_lndelta < -0.8095462:
        z += 17.20946 * (Q.mass_over_sum_pt_sq - 0.04296187) * (-0.8095462 - Q.lund_max_lndelta)
    if Q.sj3_pairmax_over_m < 0.9148508 and Q.sd_nremoved > 5.0:
        z += 0.3712716 * (0.9148508 - Q.sj3_pairmax_over_m) * (Q.sd_nremoved - 5.0)
    if Q.tau21_b2 < 0.2701529 and Q.D3_b2 < 1.462588:
        z += 0.2255184 * (0.2701529 - Q.tau21_b2) * (1.462588 - Q.D3_b2)
    if Q.N2 < 0.3829918 and Q.dr_47 < 0.09477716:
        z += -13.52642 * (0.3829918 - Q.N2) * (0.09477716 - Q.dr_47)
    if Q.mass < 164.4374 and Q.mass_top2 > 35.44963:
        z += 4.632137e-05 * (164.4374 - Q.mass) * (Q.mass_top2 - 35.44963)
    if Q.mass < 117.4867 and Q.lund_max_lndelta > -1.498181:
        z += -0.009566317 * (117.4867 - Q.mass) * (Q.lund_max_lndelta - -1.498181)
    if Q.mass < 117.4867 and Q.lund1_lndelta < -0.3023635:
        z += 0.004760488 * (117.4867 - Q.mass) * (-0.3023635 - Q.lund1_lndelta)
    if Q.tau21_b2 < 0.2701529 and Q.eta_0 > 0.1077942:
        z += 1.015987 * (0.2701529 - Q.tau21_b2) * (Q.eta_0 - 0.1077942)
    if Q.sj3_pair_mass_max < 120.6471 and Q.n_dr_0p05_0p1 < 12.0:
        z += 0.000329116 * (120.6471 - Q.sj3_pair_mass_max) * (12.0 - Q.n_dr_0p05_0p1)
    if Q.n_particles > 26.0 and Q.lne_49 < 1.070218:
        z += -0.0001435054 * (Q.n_particles - 26.0) * (1.070218 - Q.lne_49)
    if Q.dr12 < 0.2116473 and Q.dr_10 > 0.3349968:
        z += 1.444 * (0.2116473 - Q.dr12) * (Q.dr_10 - 0.3349968)
    if Q.tau3 < 0.03086713 and Q.phi_29 > -0.3354492:
        z += -5.670505 * (0.03086713 - Q.tau3) * (Q.phi_29 - -0.3354492)
    if Q.n_lund_kt_above_1 < 2.0 and Q.phi_17 < -0.3310547:
        z += -1.818636 * (2.0 - Q.n_lund_kt_above_1) * (-0.3310547 - Q.phi_17)
    if Q.tau21_b2 < 0.2701529 and Q.dr_31 < 0.02273044:
        z += -76.17365 * (0.2701529 - Q.tau21_b2) * (0.02273044 - Q.dr_31)
    if Q.M2 < 0.03465331 and Q.phi_27 > 0.0:
        z += 182.3825 * (0.03465331 - Q.M2) * (Q.phi_27 - 0.0)
    if Q.N2 < 0.3829918 and Q.dr_31 > 0.0:
        z += -0.8111559 * (0.3829918 - Q.N2) * (Q.dr_31 - 0.0)
    if Q.lnpt_6 > 2.880777 and Q.phi_17 > 0.07128906:
        z += -0.07252689 * (Q.lnpt_6 - 2.880777) * (Q.phi_17 - 0.07128906)
    if Q.mass_top30 > 162.7874 and Q.D2_b2 < 1.380731:
        z += -0.06632948 * (Q.mass_top30 - 162.7874) * (1.380731 - Q.D2_b2)
    if Q.mass_top20 > 130.7093 and Q.D2_b2 < 1.380731:
        z += 0.05212806 * (Q.mass_top20 - 130.7093) * (1.380731 - Q.D2_b2)
    if Q.sd_mass > 135.3031 and Q.D2_b2 < 1.380731:
        z += 0.005191002 * (Q.sd_mass - 135.3031) * (1.380731 - Q.D2_b2)
    if Q.max_dr < 0.6079631 and Q.lund3_lndelta > -2.817283:
        z += 0.6851614 * (0.6079631 - Q.max_dr) * (Q.lund3_lndelta - -2.817283)
    if Q.pair_mean_lndelta > -2.049856 and Q.mratio_max_012 > 0.8957738:
        z += 15.04742 * (Q.pair_mean_lndelta - -2.049856) * (Q.mratio_max_012 - 0.8957738)
    if Q.tau21_b2 < 0.2701529 and Q.phi_0 > 0.1651611:
        z += 0.4552971 * (0.2701529 - Q.tau21_b2) * (Q.phi_0 - 0.1651611)
    if Q.sj3_pairmax_over_m < 0.9148508 and Q.D2_b2 > 0.5503445:
        z += -0.01645812 * (0.9148508 - Q.sj3_pairmax_over_m) * (Q.D2_b2 - 0.5503445)
    if Q.mass_top20 > 130.7093 and Q.sj3_dr_min < 0.2901133:
        z += 0.09836167 * (Q.mass_top20 - 130.7093) * (0.2901133 - Q.sj3_dr_min)
    if Q.lund_max_lnkt > 4.323396 and Q.sj3_dr_min > 0.07110203:
        z += 0.7789544 * (Q.lund_max_lnkt - 4.323396) * (Q.sj3_dr_min - 0.07110203)
    if Q.tau5 > 0.01467699 and Q.sj2_mass2 < 1.851735:
        z += -0.5375587 * (Q.tau5 - 0.01467699) * (1.851735 - Q.sj2_mass2)
    if Q.tau21_b2 < 0.2701529 and Q.sj2_mass2 < 1.851735:
        z += 3.017988 * (0.2701529 - Q.tau21_b2) * (1.851735 - Q.sj2_mass2)
    if Q.sj3_mass2 < 1.824785 and Q.sj2_mass2 > 5.311506:
        z += -0.003493565 * (1.824785 - Q.sj3_mass2) * (Q.sj2_mass2 - 5.311506)
    if Q.sd_mass > 111.701 and Q.dr01 < 0.191004:
        z += -0.005941632 * (Q.sd_mass - 111.701) * (0.191004 - Q.dr01)
    if Q.dr12 < 0.2116473 and Q.eta_1 > -0.05657959:
        z += -1.691409 * (0.2116473 - Q.dr12) * (Q.eta_1 - -0.05657959)
    if Q.n_lund_kt_above_1 < 2.0 and Q.eta_19 < -0.1276855:
        z += 2.139623 * (2.0 - Q.n_lund_kt_above_1) * (-0.1276855 - Q.eta_19)
    if Q.lund_max_lnkt > 4.323396 and Q.phi_14 < 0.3125:
        z += -0.07112677 * (Q.lund_max_lnkt - 4.323396) * (0.3125 - Q.phi_14)
    if Q.dr12 < 0.2116473 and Q.phi_47 > -0.07501221:
        z += 0.04324381 * (0.2116473 - Q.dr12) * (Q.phi_47 - -0.07501221)
    if Q.M2 < 0.03465331 and Q.eta_19 < 0.0:
        z += -2.83588 * (0.03465331 - Q.M2) * (0.0 - Q.eta_19)
    if Q.lund_max_lnkt > 4.323396 and Q.n_pt_above_50 < 4.0:
        z += 0.07051644 * (Q.lund_max_lnkt - 4.323396) * (4.0 - Q.n_pt_above_50)
    return z


def neuron_10(Q):
    z = 4.742235e-06
    return z


def neuron_11(Q):
    z = -4.878501e-06
    return z


def neuron_12(Q):
    z = 1.628307e-05
    return z


def neuron_13(Q):
    z = 6.68468e-06
    return z


def neuron_14(Q):
    z = 9.188985e-06
    return z


def neuron_15(Q):
    z = 1.038257e-05
    return z


def neuron_16(Q):
    z = -8.221253e-06
    return z


def neuron_17(Q):
    z = 1.269395e-05
    return z


def neuron_18(Q):
    z = -5.348156e-05
    return z


def neuron_19(Q):
    z = -1.553789e-05
    return z


def neuron_20(Q):
    z = 4.223184e-05
    return z


def neuron_21(Q):
    z = 1.048267e-06
    return z


def neuron_22(Q):
    z = 1.486945
    if Q.n_pairs_kt_above_3 < 28.0:
        z += 0.01436178 * Q.n_pairs_kt_above_3 - 0.1372621
    if 28.0 <= Q.n_pairs_kt_above_3 < 47.0:
        z += -0.001379519 * Q.n_pairs_kt_above_3 + 0.3034941
    if 47.0 <= Q.n_pairs_kt_above_3 < 88.0:
        z += 0.004221284 * Q.n_pairs_kt_above_3 + 0.04025641
    if 88.0 <= Q.n_pairs_kt_above_3 < 220.0:
        z += 0.001778031 * Q.n_pairs_kt_above_3 + 0.2552626
    if Q.n_pairs_kt_above_3 >= 220.0:
        z += 0.00315755 * Q.n_pairs_kt_above_3 - 0.04823151
    if Q.M3_b2 < 0.009257989:
        z += 7.851537 * Q.M3_b2 - 0.07268945
    if 44.1643 <= Q.sj3_pair_mass_min < 80.02563:
        z += 0.01980777 * Q.sj3_pair_mass_min - 0.8747964
    if Q.sj3_pair_mass_min >= 80.02563:
        z += 0.0005134363 * Q.sj3_pair_mass_min + 0.6692448
    if Q.pair_mean_lnm2 >= 2.708356:
        z += 0.3883663 * Q.pair_mean_lnm2 - 1.051834
    if Q.sj2_mass2 < 1.851735:
        z += 0.1159565 * Q.sj2_mass2 - 0.2147208
    if Q.e2 >= 0.1081051:
        z += 6.184647 * Q.e2 - 0.6685917
    if Q.mass < 85.02716:
        z += 0.01984857 * Q.mass - 3.329751
    if 85.02716 <= Q.mass < 110.2019:
        z += -0.01875762 * Q.mass - 0.04717652
    if 110.2019 <= Q.mass < 137.452:
        z += 0.005379384 * Q.mass - 2.70712
    if 137.452 <= Q.mass < 164.4374:
        z += 0.007821562 * Q.mass - 3.042802
    if Q.mass >= 164.4374:
        z += -0.01446918 * Q.mass + 0.6226308
    if Q.mass_top50 < 115.614:
        z += -0.005371093 * Q.mass_top50 + 0.6960262
    if 115.614 <= Q.mass_top50 < 129.5874:
        z += -0.01054757 * Q.mass_top50 + 1.2945
    if Q.mass_top50 >= 129.5874:
        z += -0.005176481 * Q.mass_top50 + 0.5984737
    if Q.sum_pt_top2 < 271.1875:
        z += -0.0002420353 * Q.sum_pt_top2 + 0.06563694
    if Q.n_pairs_kt_above_1 < 325.0:
        z += 0.00118306 * Q.n_pairs_kt_above_1 - 0.5441343
    if 325.0 <= Q.n_pairs_kt_above_1 < 702.0:
        z += 0.0004234475 * Q.n_pairs_kt_above_1 - 0.2972601
    if Q.dr02 < 0.07373689:
        z += 0.4529486 * Q.dr02 + 0.03681525
    if 0.07373689 <= Q.dr02 < 0.1058963:
        z += -10.91231 * Q.dr02 + 0.8748544
    if 0.1058963 <= Q.dr02 < 0.3413762:
        z += 1.192114 * Q.dr02 - 0.4069594
    if Q.sj4_pair_mass_max >= 48.76729:
        z += -0.0002119531 * Q.sj4_pair_mass_max + 0.01033638
    if Q.lund_max_lnkt < 2.961008:
        z += -0.2235706 * Q.lund_max_lnkt + 0.6619943
    if 62.03827 <= Q.sd_mass < 88.81751:
        z += 0.01130035 * Q.sd_mass - 0.7010544
    if 88.81751 <= Q.sd_mass < 106.7501:
        z += -0.03163234 * Q.sd_mass + 3.11212
    if 106.7501 <= Q.sd_mass < 123.2919:
        z += -0.01340369 * Q.sd_mass + 1.166211
    if 123.2919 <= Q.sd_mass < 154.5947:
        z += 0.02036236 * Q.sd_mass - 2.996868
    if Q.sd_mass >= 154.5947:
        z += -0.00292071 * Q.sd_mass + 0.6025698
    if Q.N3_b05 < 0.828622:
        z += 1.195686 * Q.N3_b05 - 0.9907719
    if Q.sj3_mass3 < 2.151069:
        z += -0.04472996 * Q.sj3_mass3 + 0.09621722
    if Q.sj3_mass2 < 3.298199:
        z += 0.02456574 * Q.sj3_mass2 - 0.0810227
    if Q.e3_b2 < 0.0001729927:
        z += 3744.189 * Q.e3_b2 - 0.6477172
    if Q.m01 < 12.51777:
        z += -0.08679073 * Q.m01 + 1.422468
    if 12.51777 <= Q.m01 < 35.44963:
        z += -0.01465392 * Q.m01 + 0.5194761
    if Q.dr01 < 0.3701694:
        z += 2.267986 * Q.dr01 - 0.8395389
    if Q.C2_b05 >= 0.1620185:
        z += -1.430103 * Q.C2_b05 + 0.2317031
    if Q.n_particles < 47.0:
        z += 0.005236947 * Q.n_particles - 0.2461365
    if Q.n_pt_above_5 >= 32.0:
        z += 0.0069221 * Q.n_pt_above_5 - 0.2215072
    if Q.n_lund < 5.0:
        z += 0.07990521 * Q.n_lund - 0.3995261
    if Q.N3 < 0.5935014:
        z += -1.380835 * Q.N3 + 0.8195275
    if Q.n_dr_0p1_0p2 < 2.0:
        z += 0.06139266 * Q.n_dr_0p1_0p2 - 0.1227853
    if Q.tau21_b2 < 0.297969:
        z += 4.588227 * Q.tau21_b2 - 1.367149
    if Q.sj3_pair_mass_max < 97.52633:
        z += 0.01086382 * Q.sj3_pair_mass_max - 1.059508
    if Q.pair_max_lnm2 >= 5.908788:
        z += 0.1762545 * Q.pair_max_lnm2 - 1.04145
    if Q.jet_abs_eta >= 1.465946:
        z += 1.097246 * Q.jet_abs_eta - 1.608503
    if Q.lne_38 >= 1.040994:
        z += 0.02227196 * Q.lne_38 - 0.02318498
    if Q.sj3_dr_min < 0.08335692:
        z += 1.204341 * Q.sj3_dr_min + 0.03036047
    if 0.08335692 <= Q.sj3_dr_min < 0.1432984:
        z += -2.181305 * Q.sj3_dr_min + 0.3125775
    if Q.n_pt_above_1 >= 35.0:
        z += 0.003333703 * Q.n_pt_above_1 - 0.1166796
    if Q.C2 < 0.23825:
        z += -5.79832 * Q.C2 + 1.38145
    if Q.tau1 >= 0.2694895:
        z += -1.462625 * Q.tau1 + 0.3941622
    if Q.M2_b2 < 0.05966366:
        z += -12.81347 * Q.M2_b2 + 0.7644987
    if Q.psi_0p2 >= 0.865232:
        z += 1.378077 * Q.psi_0p2 - 1.192356
    if Q.n_dr_0p2_0p4 < 6.0:
        z += 0.0131543 * Q.n_dr_0p2_0p4 - 0.07892578
    if Q.n_pairs_kt_above_3 < 220.0 and Q.dr01 < 0.0856189:
        z += -0.02486716 * (220.0 - Q.n_pairs_kt_above_3) * (0.0856189 - Q.dr01)
    if Q.n_pairs_kt_above_3 < 220.0 and Q.sj3_pairmin_over_m > 0.3081017:
        z += 0.004403179 * (220.0 - Q.n_pairs_kt_above_3) * (Q.sj3_pairmin_over_m - 0.3081017)
    if Q.mass_top50 < 129.5874 and Q.tau21_b2 < 0.297969:
        z += 0.06129561 * (129.5874 - Q.mass_top50) * (0.297969 - Q.tau21_b2)
    if Q.pair_mean_lnm2 > 2.708356 and Q.lne_1 > 4.222261:
        z += 0.09370366 * (Q.pair_mean_lnm2 - 2.708356) * (Q.lne_1 - 4.222261)
    if Q.mass_top50 < 129.5874 and Q.ecf_g42 > 1.451882e-05:
        z += 95.20556 * (129.5874 - Q.mass_top50) * (Q.ecf_g42 - 1.451882e-05)
    if Q.n_pairs_kt_above_3 < 220.0 and Q.psi_0p1 < 0.3850339:
        z += 0.001204078 * (220.0 - Q.n_pairs_kt_above_3) * (0.3850339 - Q.psi_0p1)
    if Q.sj3_pair_mass_min > 44.1643 and Q.tau4 < 0.07749559:
        z += -0.1547839 * (Q.sj3_pair_mass_min - 44.1643) * (0.07749559 - Q.tau4)
    if Q.sj3_pair_mass_min > 44.1643 and Q.n_pairs_kt_above_10 < 13.0:
        z += -0.0003062784 * (Q.sj3_pair_mass_min - 44.1643) * (13.0 - Q.n_pairs_kt_above_10)
    if Q.sj4_pair_mass_max > 48.76729 and Q.dr_max_012 < 0.310246:
        z += 0.001464265 * (Q.sj4_pair_mass_max - 48.76729) * (0.310246 - Q.dr_max_012)
    if Q.sj4_pair_mass_max > 48.76729 and Q.N3 > 0.5935014:
        z += 0.003680925 * (Q.sj4_pair_mass_max - 48.76729) * (Q.N3 - 0.5935014)
    if Q.sj3_mass3 < 2.151069 and Q.n_lund_kt_above_5 < 3.0:
        z += -0.01328544 * (2.151069 - Q.sj3_mass3) * (3.0 - Q.n_lund_kt_above_5)
    if Q.sj3_pair_mass_min > 80.02563 and Q.sj3_mass2 < 19.01265:
        z += -0.003208343 * (Q.sj3_pair_mass_min - 80.02563) * (19.01265 - Q.sj3_mass2)
    if Q.sj3_pair_mass_min > 44.1643 and Q.sj3_mass2 < 21.93246:
        z += 0.0007498492 * (Q.sj3_pair_mass_min - 44.1643) * (21.93246 - Q.sj3_mass2)
    if Q.pair_mean_lnm2 > 2.708356 and Q.sj4_dr_min < 0.1631992:
        z += 0.858026 * (Q.pair_mean_lnm2 - 2.708356) * (0.1631992 - Q.sj4_dr_min)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.sum_z_dr2_top2 < 0.009042742:
        z += 1.530998 * (28.0 - Q.n_pairs_kt_above_3) * (0.009042742 - Q.sum_z_dr2_top2)
    if Q.mass < 164.4374 and Q.sum_z_dr2_top3 > 0.00878048:
        z += -0.1502243 * (164.4374 - Q.mass) * (Q.sum_z_dr2_top3 - 0.00878048)
    if Q.e3_b2 < 0.0001729927 and Q.n_pt_above_10 < 13.0:
        z += 57.99002 * (0.0001729927 - Q.e3_b2) * (13.0 - Q.n_pt_above_10)
    if Q.N3_b05 < 0.828622 and Q.pt_balance01 < 0.2757029:
        z += -1.460822 * (0.828622 - Q.N3_b05) * (0.2757029 - Q.pt_balance01)
    if Q.sj4_pair_mass_max > 48.76729 and Q.n_lund_kt_above_1 < 6.0:
        z += 0.0002722057 * (Q.sj4_pair_mass_max - 48.76729) * (6.0 - Q.n_lund_kt_above_1)
    if Q.m01 < 12.51777 and Q.sj3_dr_min < 0.3956483:
        z += -0.03513017 * (12.51777 - Q.m01) * (0.3956483 - Q.sj3_dr_min)
    if Q.m01 < 35.44963 and Q.z_2nd > 0.07694751:
        z += 0.01795406 * (35.44963 - Q.m01) * (Q.z_2nd - 0.07694751)
    if Q.C2_b05 > 0.1620185 and Q.ecf_g41 < 0.0004392119:
        z += -588.0042 * (Q.C2_b05 - 0.1620185) * (0.0004392119 - Q.ecf_g41)
    if Q.sd_mass > 62.03827 and Q.tau21_b2 > 0.5868564:
        z += -0.01099208 * (Q.sd_mass - 62.03827) * (Q.tau21_b2 - 0.5868564)
    if Q.tau21_b2 < 0.297969 and Q.sj2_mass2 > 7.54918:
        z += -0.03164563 * (0.297969 - Q.tau21_b2) * (Q.sj2_mass2 - 7.54918)
    if Q.sj3_mass3 < 2.151069 and Q.M2_b2 > 0.01934939:
        z += -6.72301 * (2.151069 - Q.sj3_mass3) * (Q.M2_b2 - 0.01934939)
    if Q.sd_mass > 88.81751 and Q.C2_b2 > 0.04664369:
        z += 0.006559005 * (Q.sd_mass - 88.81751) * (Q.C2_b2 - 0.04664369)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.dr_2 < 0.2792084:
        z += -0.02231627 * (28.0 - Q.n_pairs_kt_above_3) * (0.2792084 - Q.dr_2)
    if Q.n_pt_above_5 > 32.0 and Q.e4_b2 > 7.0969e-08:
        z += 3324.456 * (Q.n_pt_above_5 - 32.0) * (Q.e4_b2 - 7.0969e-08)
    if Q.n_pairs_kt_above_3 < 220.0 and Q.sd_zg > 0.2652027:
        z += -0.001943388 * (220.0 - Q.n_pairs_kt_above_3) * (Q.sd_zg - 0.2652027)
    if Q.sj3_pair_mass_max < 97.52633 and Q.tau54 < 0.8610613:
        z += 0.002741258 * (97.52633 - Q.sj3_pair_mass_max) * (0.8610613 - Q.tau54)
    if Q.mass_top50 > 115.614 and Q.max_dr < 0.9206713:
        z += 0.01086574 * (Q.mass_top50 - 115.614) * (0.9206713 - Q.max_dr)
    if Q.N3_b05 < 0.828622 and Q.n_dr_0_0p05 < 6.0:
        z += 0.01064458 * (0.828622 - Q.N3_b05) * (6.0 - Q.n_dr_0_0p05)
    if Q.pair_max_lnm2 > 5.908788 and Q.n_lund_kt_above_5 > 2.0:
        z += -0.03730109 * (Q.pair_max_lnm2 - 5.908788) * (Q.n_lund_kt_above_5 - 2.0)
    if Q.m01 < 12.51777 and Q.eta_12 < 0.005525017:
        z += 0.007382857 * (12.51777 - Q.m01) * (0.005525017 - Q.eta_12)
    if Q.n_dr_0p1_0p2 < 2.0 and Q.eta_38 < 0.2912598:
        z += 0.2670437 * (2.0 - Q.n_dr_0p1_0p2) * (0.2912598 - Q.eta_38)
    if Q.m01 < 12.51777 and Q.e3_b2 > 0.0007909605:
        z += -3.162552 * (12.51777 - Q.m01) * (Q.e3_b2 - 0.0007909605)
    if Q.dr02 < 0.07373689 and Q.D2_b2 < 7.782712:
        z += -0.1036487 * (0.07373689 - Q.dr02) * (7.782712 - Q.D2_b2)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.sj3_dr13 < 0.4383089:
        z += -0.04164694 * (28.0 - Q.n_pairs_kt_above_3) * (0.4383089 - Q.sj3_dr13)
    if Q.sj3_pair_mass_min > 44.1643 and Q.eta_42 < -0.232666:
        z += 0.001572981 * (Q.sj3_pair_mass_min - 44.1643) * (-0.232666 - Q.eta_42)
    if Q.n_pairs_kt_above_3 > 88.0 and Q.jet_abs_eta > 0.3274899:
        z += 0.00417687 * (Q.n_pairs_kt_above_3 - 88.0) * (Q.jet_abs_eta - 0.3274899)
    if Q.pair_mean_lnm2 > 2.708356 and Q.dr_57 < 0.3219863:
        z += -0.8493574 * (Q.pair_mean_lnm2 - 2.708356) * (0.3219863 - Q.dr_57)
    if Q.n_pairs_kt_above_3 > 47.0 and Q.jet_abs_eta > 0.1941339:
        z += -0.00364847 * (Q.n_pairs_kt_above_3 - 47.0) * (Q.jet_abs_eta - 0.1941339)
    if Q.sj3_mass2 < 3.298199 and Q.eta_20 < -0.1281738:
        z += -0.005112199 * (3.298199 - Q.sj3_mass2) * (-0.1281738 - Q.eta_20)
    if Q.C2_b05 > 0.1620185 and Q.dr_2 > 0.1403176:
        z += 0.1484815 * (Q.C2_b05 - 0.1620185) * (Q.dr_2 - 0.1403176)
    if Q.sj3_pair_mass_min > 80.02563 and Q.eta_38 < 0.04125977:
        z += -0.0003402767 * (Q.sj3_pair_mass_min - 80.02563) * (0.04125977 - Q.eta_38)
    return z


def neuron_23(Q):
    z = 1.400643e-05
    return z


def neuron_24(Q):
    z = -5.022347e-05
    return z


def neuron_25(Q):
    z = 0.0001539363
    return z


def neuron_26(Q):
    z = -4.593832e-06
    return z


def neuron_27(Q):
    z = -0.7999794
    if Q.M2 < 0.1066509:
        z += -5.428892 * Q.M2 + 0.5789964
    if Q.mass_top30 < 67.20576:
        z += -0.01637493 * Q.mass_top30 + 1.10049
    if Q.sj4_pair_mass_max < 63.34656:
        z += 0.008674798 * Q.sj4_pair_mass_max - 1.340126
    if 63.34656 <= Q.sj4_pair_mass_max < 115.5142:
        z += 0.01515512 * Q.sj4_pair_mass_max - 1.750632
    if Q.tau2 < 0.07264571:
        z += -2.165483 * Q.tau2 + 0.1573131
    if Q.n_pairs_kt_above_3 < 28.0:
        z += 0.002723727 * Q.n_pairs_kt_above_3 + 0.248554
    if 28.0 <= Q.n_pairs_kt_above_3 < 41.0:
        z += -0.02498603 * Q.n_pairs_kt_above_3 + 1.024427
    if Q.sj4_dr_min < 0.1458275:
        z += -1.785531 * Q.sj4_dr_min + 0.2603796
    if 80.36029 <= Q.sj3_pair_mass_max < 128.6605:
        z += 0.01033765 * Q.sj3_pair_mass_max - 0.8307363
    if Q.sj3_pair_mass_max >= 128.6605:
        z += -0.01258346 * Q.sj3_pair_mass_max + 2.118305
    if 42.31629 <= Q.sd_mass < 83.61981:
        z += -0.003357068 * Q.sd_mass + 0.1420586
    if 83.61981 <= Q.sd_mass < 127.8042:
        z += 0.005811082 * Q.sd_mass - 0.6245803
    if Q.sd_mass >= 127.8042:
        z += -0.003628671 * Q.sd_mass + 0.5818602
    if Q.mass_top10 < 62.96042:
        z += -0.0005222119 * Q.mass_top10 - 0.1687703
    if 62.96042 <= Q.mass_top10 < 116.599:
        z += 0.003759401 * Q.mass_top10 - 0.4383424
    if Q.sum_z_dr2_top2 < 0.072817:
        z += 2.963819 * Q.sum_z_dr2_top2 - 0.2158164
    if Q.sj3_dr_min < 0.08335692:
        z += 6.037359 * Q.sj3_dr_min - 0.5032557
    if Q.sj3_dr_min >= 0.3526989:
        z += 1.12667 * Q.sj3_dr_min - 0.3973754
    if Q.N3_b05 < 0.6240065:
        z += 0.1105311 * Q.N3_b05 + 0.216613
    if 0.6240065 <= Q.N3_b05 < 1.549573:
        z += -0.3085516 * Q.N3_b05 + 0.4781234
    if Q.n_dr_0p1_0p2 < 6.0:
        z += -0.0218797 * Q.n_dr_0p1_0p2 + 0.1312782
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.0137317 * Q.n_pairs_kt_above_1 + 1.199603
    if 80.0 <= Q.n_pairs_kt_above_1 < 223.0:
        z += -0.0007067596 * Q.n_pairs_kt_above_1 + 0.1576074
    if Q.n_dr_0p4_up < 2.0:
        z += 0.001428571 * Q.n_dr_0p4_up - 0.002857142
    if Q.min_pair_mass >= 6.383481:
        z += -0.007347027 * Q.min_pair_mass + 0.04689961
    if Q.pair_mean_lnm2 < 1.488114:
        z += -0.3438476 * Q.pair_mean_lnm2 + 1.162415
    if 1.488114 <= Q.pair_mean_lnm2 < 3.38061:
        z += -0.5005041 * Q.pair_mean_lnm2 + 1.395538
    if Q.pair_mean_lnm2 >= 3.38061:
        z += -0.1566565 * Q.pair_mean_lnm2 + 0.2331227
    if Q.dr_min_012 < 0.01600475:
        z += 18.21674 * Q.dr_min_012 - 0.2915545
    if Q.N2_b05 < 0.4954556:
        z += -6.770232 * Q.N2_b05 + 3.354349
    if Q.sj3_mass2 < 1.824785:
        z += -0.001541931 * Q.sj3_mass2 + 0.002813693
    if Q.sj3_mass2 >= 15.2797:
        z += -0.007257651 * Q.sj3_mass2 + 0.1108947
    if Q.tau32 < 0.3951525:
        z += 0.314943 * Q.tau32 - 0.1244505
    if Q.lund2_lndelta >= -1.411949:
        z += 0.007514949 * Q.lund2_lndelta + 0.01061072
    if Q.n_pairs_kt_above_10 < 6.0:
        z += -0.02879949 * Q.n_pairs_kt_above_10 + 0.1727969
    if Q.pair_mean_lnkt < 0.5586581:
        z += 0.1107135 * Q.pair_mean_lnkt - 0.06185098
    if Q.pair_mean_lnkt >= 0.9367772:
        z += -0.3327227 * Q.pair_mean_lnkt + 0.311687
    if Q.n_dr_0_0p05 < 2.0:
        z += -0.02228109 * Q.n_dr_0_0p05 + 0.04456218
    if Q.tau4 >= 0.05937965:
        z += -7.083476 * Q.tau4 + 0.4206143
    if Q.pair_mean_lndelta >= -1.921258:
        z += 0.02378267 * Q.pair_mean_lndelta + 0.04569265
    if Q.mass_top5 >= 58.28268:
        z += 0.006468811 * Q.mass_top5 - 0.3770196
    if Q.ecf_g42 >= 2.829762e-05:
        z += -2857.068 * Q.ecf_g42 + 0.08084822
    if Q.sj3_dr_max >= 0.8930677:
        z += -0.5315527 * Q.sj3_dr_max + 0.4747125
    if Q.mass_top50 >= 118.8408:
        z += 0.008975833 * Q.mass_top50 - 1.066695
    if Q.sj3_dr12 >= 0.4152525:
        z += 0.6265678 * Q.sj3_dr12 - 0.2601838
    if Q.dr01 < 0.0121584:
        z += 17.40077 * Q.dr01 + 0.3098121
    if 0.0121584 <= Q.dr01 < 0.1335998:
        z += -4.293246 * Q.dr01 + 0.5735767
    if Q.n_dr_0p05_0p1 < 2.0:
        z += -0.06499422 * Q.n_dr_0p05_0p1 + 0.1299884
    if Q.C3_b2 < 0.008597002:
        z += -13.44685 * Q.C3_b2 + 0.1156026
    if Q.dr_11 >= 0.4092361:
        z += 0.1563804 * Q.dr_11 - 0.06399649
    if Q.pt1_dr01 < 6.394276:
        z += 0.08553372 * Q.pt1_dr01 - 0.5469262
    if Q.M2 < 0.1066509 and Q.lam1 < 0.06041764:
        z += -140.6598 * (0.1066509 - Q.M2) * (0.06041764 - Q.lam1)
    if Q.sj4_pair_mass_max < 115.5142 and Q.n_lund > 5.0:
        z += 0.0002599431 * (115.5142 - Q.sj4_pair_mass_max) * (Q.n_lund - 5.0)
    if Q.sj4_pair_mass_max < 115.5142 and Q.lnpt_0 > 3.990719:
        z += 0.002273828 * (115.5142 - Q.sj4_pair_mass_max) * (Q.lnpt_0 - 3.990719)
    if Q.sd_mass > 83.61981 and Q.tau4 < 0.06637116:
        z += 0.05379044 * (Q.sd_mass - 83.61981) * (0.06637116 - Q.tau4)
    if Q.sj4_pair_mass_max < 115.5142 and Q.ecf_g42 > 2.829762e-05:
        z += 12.4646 * (115.5142 - Q.sj4_pair_mass_max) * (Q.ecf_g42 - 2.829762e-05)
    if Q.sum_z_dr2_top2 < 0.072817 and Q.n_dr_0p2_0p4 > 10.0:
        z += 0.1891932 * (0.072817 - Q.sum_z_dr2_top2) * (Q.n_dr_0p2_0p4 - 10.0)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.M3_b05 > 0.05410585:
        z += 0.1652006 * (28.0 - Q.n_pairs_kt_above_3) * (Q.M3_b05 - 0.05410585)
    if Q.sum_z_dr2_top2 < 0.072817 and Q.sj3_dr_min > 0.08335692:
        z += -9.999322 * (0.072817 - Q.sum_z_dr2_top2) * (Q.sj3_dr_min - 0.08335692)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.dr02 < 0.3413762:
        z += 0.01713267 * (28.0 - Q.n_pairs_kt_above_3) * (0.3413762 - Q.dr02)
    if Q.sd_mass > 127.8042 and Q.tau21 < 0.2377332:
        z += -0.08797306 * (Q.sd_mass - 127.8042) * (0.2377332 - Q.tau21)
    if Q.n_pairs_kt_above_1 < 223.0 and Q.dr_5 < 0.4190095:
        z += 0.002068361 * (223.0 - Q.n_pairs_kt_above_1) * (0.4190095 - Q.dr_5)
    if Q.N3_b05 < 1.549573 and Q.sj3_mass1 < 36.36236:
        z += 0.0119087 * (1.549573 - Q.N3_b05) * (36.36236 - Q.sj3_mass1)
    if Q.M2 < 0.1066509 and Q.tau21 < 0.5797033:
        z += 15.62093 * (0.1066509 - Q.M2) * (0.5797033 - Q.tau21)
    if Q.sj3_dr_min > 0.3526989 and Q.sj3_pairmax_over_m < 0.806533:
        z += 12.25331 * (Q.sj3_dr_min - 0.3526989) * (0.806533 - Q.sj3_pairmax_over_m)
    if Q.sj4_pair_mass_max < 63.34656 and Q.z_3rd > 0.06584103:
        z += 0.1865328 * (63.34656 - Q.sj4_pair_mass_max) * (Q.z_3rd - 0.06584103)
    if Q.sd_mass > 42.31629 and Q.mratio_max_012 > 0.7062246:
        z += 0.001147931 * (Q.sd_mass - 42.31629) * (Q.mratio_max_012 - 0.7062246)
    if Q.sj3_pair_mass_max > 128.6605 and Q.lnpt_12 > 1.112834:
        z += 0.003629369 * (Q.sj3_pair_mass_max - 128.6605) * (Q.lnpt_12 - 1.112834)
    if Q.n_dr_0p4_up < 2.0 and Q.sj3_z3 < 0.1719087:
        z += 0.02228307 * (2.0 - Q.n_dr_0p4_up) * (0.1719087 - Q.sj3_z3)
    if Q.tau32 < 0.3951525 and Q.mean_eta2 < 0.0415296:
        z += -41.17744 * (0.3951525 - Q.tau32) * (0.0415296 - Q.mean_eta2)
    if Q.n_dr_0_0p05 < 2.0 and Q.lnerel_12 < -5.280637:
        z += -0.01536495 * (2.0 - Q.n_dr_0_0p05) * (-5.280637 - Q.lnerel_12)
    if Q.n_pairs_kt_above_1 < 223.0 and Q.n_lund_kt_above_1 > 3.0:
        z += 0.000857684 * (223.0 - Q.n_pairs_kt_above_1) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.sum_z_dr2_top2 < 0.072817 and Q.n_dr_0p4_up < 12.0:
        z += -0.3822402 * (0.072817 - Q.sum_z_dr2_top2) * (12.0 - Q.n_dr_0p4_up)
    if Q.sj4_pair_mass_max < 115.5142 and Q.sj3_dr13 > 0.4088773:
        z += 0.01269315 * (115.5142 - Q.sj4_pair_mass_max) * (Q.sj3_dr13 - 0.4088773)
    if Q.lund2_lndelta > -1.411949 and Q.dr_10 > 0.5063302:
        z += 0.2594465 * (Q.lund2_lndelta - -1.411949) * (Q.dr_10 - 0.5063302)
    if Q.N2_b05 < 0.4954556 and Q.dr01 > 0.4232714:
        z += -12.70354 * (0.4954556 - Q.N2_b05) * (Q.dr01 - 0.4232714)
    if Q.sj3_pair_mass_max > 80.36029 and Q.tau3 < 0.05844876:
        z += 0.4838781 * (Q.sj3_pair_mass_max - 80.36029) * (0.05844876 - Q.tau3)
    if Q.sd_mass > 127.8042 and Q.tau4 < 0.05417009:
        z += -0.2141076 * (Q.sd_mass - 127.8042) * (0.05417009 - Q.tau4)
    if Q.sum_z_dr2_top2 < 0.072817 and Q.M3 > 0.02540381:
        z += 101.4649 * (0.072817 - Q.sum_z_dr2_top2) * (Q.M3 - 0.02540381)
    if Q.sj4_dr_min < 0.1458275 and Q.sj3_z3 < 0.1426884:
        z += -11.28527 * (0.1458275 - Q.sj4_dr_min) * (0.1426884 - Q.sj3_z3)
    if Q.sj4_pair_mass_max < 115.5142 and Q.lne_3 < 4.02591:
        z += 0.004672835 * (115.5142 - Q.sj4_pair_mass_max) * (4.02591 - Q.lne_3)
    if Q.N2_b05 < 0.4954556 and Q.phi_47 > 0.0:
        z += -1.95161 * (0.4954556 - Q.N2_b05) * (Q.phi_47 - 0.0)
    if Q.pair_mean_lnm2 > 1.488114 and Q.eta_27 < 0.3334961:
        z += 0.01465179 * (Q.pair_mean_lnm2 - 1.488114) * (0.3334961 - Q.eta_27)
    if Q.sj4_pair_mass_max < 115.5142 and Q.lnerel_50 < -6.393071:
        z += -0.0002388772 * (115.5142 - Q.sj4_pair_mass_max) * (-6.393071 - Q.lnerel_50)
    if Q.sj4_pair_mass_max < 63.34656 and Q.z_2nd < 0.1783885:
        z += -0.002830683 * (63.34656 - Q.sj4_pair_mass_max) * (0.1783885 - Q.z_2nd)
    if Q.dr01 < 0.0121584 and Q.lnpt_18 > 1.844387:
        z += 24.80019 * (0.0121584 - Q.dr01) * (Q.lnpt_18 - 1.844387)
    if Q.n_dr_0_0p05 < 2.0 and Q.e3_b2 > 0.000125186:
        z += 9.866284 * (2.0 - Q.n_dr_0_0p05) * (Q.e3_b2 - 0.000125186)
    if Q.N3_b05 < 1.549573 and Q.n_real_top20 < 20.0:
        z += -0.08630389 * (1.549573 - Q.N3_b05) * (20.0 - Q.n_real_top20)
    if Q.N2_b05 < 0.4954556 and Q.C3_b2 < 0.04229114:
        z += -15.84728 * (0.4954556 - Q.N2_b05) * (0.04229114 - Q.C3_b2)
    if Q.n_dr_0p1_0p2 < 6.0 and Q.dr_4 < 0.1385887:
        z += 0.02004807 * (6.0 - Q.n_dr_0p1_0p2) * (0.1385887 - Q.dr_4)
    if Q.n_pairs_kt_above_1 < 80.0 and Q.dr12 < 0.06632183:
        z += -0.1626139 * (80.0 - Q.n_pairs_kt_above_1) * (0.06632183 - Q.dr12)
    if Q.pair_mean_lnm2 > 1.488114 and Q.dr12 < 0.2677232:
        z += -0.001441318 * (Q.pair_mean_lnm2 - 1.488114) * (0.2677232 - Q.dr12)
    if Q.sj4_pair_mass_max < 63.34656 and Q.pt1_dr01 > 11.84401:
        z += 0.000252639 * (63.34656 - Q.sj4_pair_mass_max) * (Q.pt1_dr01 - 11.84401)
    if Q.dr_min_012 < 0.01600475 and Q.pair_max_lnkt < 3.338396:
        z += 14.80048 * (0.01600475 - Q.dr_min_012) * (3.338396 - Q.pair_max_lnkt)
    if Q.N3_b05 < 1.549573 and Q.dr01 < 0.4232714:
        z += 0.0439811 * (1.549573 - Q.N3_b05) * (0.4232714 - Q.dr01)
    if Q.n_dr_0_0p05 < 2.0 and Q.lnpt_18 < 0.4806885:
        z += -0.003002336 * (2.0 - Q.n_dr_0_0p05) * (0.4806885 - Q.lnpt_18)
    if Q.mass_top50 > 118.8408 and Q.phi_80 < 0.0:
        z += 0.006310272 * (Q.mass_top50 - 118.8408) * (0.0 - Q.phi_80)
    if Q.mass_top10 < 116.599 and Q.C2_b2 < 0.01057938:
        z += -0.549443 * (116.599 - Q.mass_top10) * (0.01057938 - Q.C2_b2)
    if Q.sum_z_dr2_top2 < 0.072817 and Q.C2_b2 < 0.03403084:
        z += 19.39503 * (0.072817 - Q.sum_z_dr2_top2) * (0.03403084 - Q.C2_b2)
    if Q.sum_z_dr2_top2 < 0.072817 and Q.sj3_mass3 > 8.552496:
        z += -0.2264826 * (0.072817 - Q.sum_z_dr2_top2) * (Q.sj3_mass3 - 8.552496)
    if Q.n_pairs_kt_above_3 < 41.0 and Q.planar_flow < 0.4623202:
        z += -0.003481115 * (41.0 - Q.n_pairs_kt_above_3) * (0.4623202 - Q.planar_flow)
    return z


def neuron_28(Q):
    z = -0.0248443
    z += -0.006769287 * Q.pair_mean_lnm2
    if Q.M3_b05 >= 0.03550507:
        z += 0.3677591 * Q.M3_b05 - 0.01305731
    if Q.m01 < 3.294888:
        z += 0.008560186 * Q.m01 - 0.002041859
    if 3.294888 <= Q.m01 < 42.26122:
        z += -0.0006714255 * Q.m01 + 0.02837526
    if 83.57316 <= Q.mass_top40 < 122.7145:
        z += 5.443356e-05 * Q.mass_top40 - 0.004549185
    if Q.mass_top40 >= 122.7145:
        z += -0.001779775 * Q.mass_top40 + 0.2205349
    if Q.sj2_mass2 < 1.851735:
        z += -0.01193275 * Q.sj2_mass2 + 0.02209629
    if Q.LHA < 0.2686235:
        z += -0.1893886 * Q.LHA + 0.05087424
    if Q.dr02 < 0.1620436:
        z += -0.2462822 * Q.dr02 - 0.01109678
    if 0.1620436 <= Q.dr02 < 0.3413762:
        z += 0.2844169 * Q.dr02 - 0.09709317
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.0002171363 * Q.n_pairs_kt_above_3 + 0.01453937
    if 28.0 <= Q.n_pairs_kt_above_3 < 88.0:
        z += -0.0001409925 * Q.n_pairs_kt_above_3 + 0.01240734
    if Q.mass_top50 >= 161.1264:
        z += 0.002660518 * Q.mass_top50 - 0.4286797
    if Q.m012 >= 69.67489:
        z += 0.0009567803 * Q.m012 - 0.06666357
    if Q.lne_1 >= 4.70445:
        z += 0.01624518 * Q.lne_1 - 0.07642465
    if 90.08945 <= Q.mass < 95.14961:
        z += -0.003056143 * Q.mass + 0.2753262
    if 95.14961 <= Q.mass < 114.0172:
        z += -0.003898677 * Q.mass + 0.355493
    if Q.mass >= 114.0172:
        z += -0.001797385 * Q.mass + 0.1159095
    if Q.dr01 >= 0.1335998:
        z += 0.3001153 * Q.dr01 - 0.04009534
    if Q.pair_mean_lndelta >= -1.986684:
        z += -0.05282364 * Q.pair_mean_lndelta - 0.1049439
    if Q.n_pairs_kt_above_1 < 101.0:
        z += 0.0005475452 * Q.n_pairs_kt_above_1 - 0.2476895
    if 101.0 <= Q.n_pairs_kt_above_1 < 325.0:
        z += 0.0004268349 * Q.n_pairs_kt_above_1 - 0.2354977
    if 325.0 <= Q.n_pairs_kt_above_1 < 873.0:
        z += 0.0001765992 * Q.n_pairs_kt_above_1 - 0.1541711
    if Q.pair_mean_lnkt >= 0.9367772:
        z += 0.08695469 * Q.pair_mean_lnkt - 0.08145718
    if Q.sum_z_dr2_top10 < 0.03877522:
        z += -3.369912 * Q.sum_z_dr2_top10 + 0.1306691
    if 3.22013 <= Q.lund_max_lnkt < 4.115614:
        z += 0.1662894 * Q.lund_max_lnkt - 0.5354735
    if Q.lund_max_lnkt >= 4.115614:
        z += 0.07264977 * Q.lund_max_lnkt - 0.1500889
    if Q.tau5 < 0.02098847:
        z += 4.266754 * Q.tau5 - 0.08955262
    if Q.C2 < 0.11544:
        z += -0.3610284 * Q.C2 + 0.04167712
    if Q.mass_top20 >= 121.2735:
        z += -0.0008204046 * Q.mass_top20 + 0.09949334
    if Q.jet_abs_eta >= 1.465946:
        z += 0.02565981 * Q.jet_abs_eta - 0.03761589
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.01772962 * Q.n_dr_0p2_0p4 - 0.1241073
    if Q.e3 < 0.004407991:
        z += 18.02768 * Q.e3 - 0.07946585
    if Q.ecf_g32 < 0.003001458:
        z += 5.80589 * Q.ecf_g32 - 0.01742614
    if Q.ecf_g32 >= 0.006299414:
        z += -6.037929 * Q.ecf_g32 + 0.03803542
    if Q.M2 < 0.06625964:
        z += -1.163075 * Q.M2 + 0.07706496
    if Q.z_top2_slots >= 0.4675914:
        z += -0.08049833 * Q.z_top2_slots + 0.03764033
    if Q.lne_29 >= 1.557322:
        z += -0.003631477 * Q.lne_29 + 0.005655379
    if Q.sj3_pair_mass_min >= 37.19471:
        z += -0.0003527532 * Q.sj3_pair_mass_min + 0.01312055
    if Q.sj4_pair_mass_max >= 115.5142:
        z += 0.001236198 * Q.sj4_pair_mass_max - 0.1427985
    if Q.z_dr_0p2_0p4 < 0.05279362:
        z += -2.17002 * Q.z_dr_0p2_0p4 + 0.1145632
    if Q.n_lund < 5.0:
        z += 0.03336112 * Q.n_lund - 0.1911835
    if 5.0 <= Q.n_lund < 8.0:
        z += 0.008125966 * Q.n_lund - 0.06500773
    if Q.n_lund_kt_above_1 < 4.0:
        z += 0.001495921 * Q.n_lund_kt_above_1 - 0.005983686
    if Q.dr_max_012 < 0.3398637:
        z += 0.05359741 * Q.dr_max_012 - 0.01821581
    if Q.n_dr_0p4_up < 5.0:
        z += 0.006053164 * Q.n_dr_0p4_up - 0.03026582
    if Q.e2_b2 < 0.01374378:
        z += 2.463391 * Q.e2_b2 - 0.03385631
    if Q.sj3_pair_mass_max < 44.2029:
        z += -0.003470607 * Q.sj3_pair_mass_max + 0.2097723
    if 44.2029 <= Q.sj3_pair_mass_max < 73.24742:
        z += -0.003488446 * Q.sj3_pair_mass_max + 0.2105608
    if 73.24742 <= Q.sj3_pair_mass_max < 93.87663:
        z += 0.002636028 * Q.sj3_pair_mass_max - 0.2380411
    if 93.87663 <= Q.sj3_pair_mass_max < 120.6471:
        z += -0.0004028319 * Q.sj3_pair_mass_max + 0.0472368
    if Q.sj3_pair_mass_max >= 120.6471:
        z += -1.783933e-05 * Q.sj3_pair_mass_max + 0.00078855
    if Q.n_pt_above_10 < 13.0:
        z += -0.001943426 * Q.n_pt_above_10 + 0.02526454
    if Q.n_pairs_kt_above_10 < 7.0:
        z += 0.001755259 * Q.n_pairs_kt_above_10 - 0.01228681
    if Q.pair_max_lnkt >= 2.080306:
        z += -0.130344 * Q.pair_max_lnkt + 0.2711553
    if Q.pt_balance01 < 0.3035327:
        z += -0.09232755 * Q.pt_balance01 + 0.02802444
    if Q.pair_max_lnm2 >= 6.411593:
        z += 0.1446564 * Q.pair_max_lnm2 - 0.9274782
    if Q.tau1 >= 0.1866173:
        z += 0.1350238 * Q.tau1 - 0.02519779
    if Q.sum_z_dr2_top3 < 0.07142062:
        z += -1.575037 * Q.sum_z_dr2_top3 + 0.1124901
    if Q.sj3_mass3 < 1.053727:
        z += 0.02438697 * Q.sj3_mass3 - 0.0256972
    if Q.sj3_z3 < 0.0184643:
        z += -3.602717 * Q.sj3_z3 + 0.06652165
    if 94.55722 <= Q.sd_mass < 119.4443:
        z += -0.001087222 * Q.sd_mass + 0.1028047
    if 119.4443 <= Q.sd_mass < 154.5947:
        z += 0.0008656578 * Q.sd_mass - 0.1304557
    if Q.sd_mass >= 154.5947:
        z += -0.0008482926 * Q.sd_mass + 0.1345118
    if Q.M2_b05 < 0.0815014:
        z += -0.7190338 * Q.M2_b05 + 0.05860227
    if Q.e3_b05 >= 0.001630164:
        z += 8.231256 * Q.e3_b05 - 0.0134183
    if Q.sd_zg < 0.1900649:
        z += -0.2941108 * Q.sd_zg + 0.05590013
    if Q.lund3_lndelta >= -2.817283:
        z += -0.01079316 * Q.lund3_lndelta - 0.03040739
    if Q.M3_b05 > 0.03550507 and Q.sj2_zsoft > 0.2089827:
        z += 1.866856 * (Q.M3_b05 - 0.03550507) * (Q.sj2_zsoft - 0.2089827)
    if Q.M3_b05 > 0.03550507 and Q.tau32_b2 < 0.6550361:
        z += -0.6562298 * (Q.M3_b05 - 0.03550507) * (0.6550361 - Q.tau32_b2)
    if Q.sj2_mass2 < 1.851735 and Q.lnptrel_14 > -4.925159:
        z += 0.02766891 * (1.851735 - Q.sj2_mass2) * (Q.lnptrel_14 - -4.925159)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.dr_0 < 0.1589855:
        z += -0.004587993 * (28.0 - Q.n_pairs_kt_above_3) * (0.1589855 - Q.dr_0)
    if Q.n_pairs_kt_above_1 < 325.0 and Q.M2_b05 < 0.1573433:
        z += -0.001014709 * (325.0 - Q.n_pairs_kt_above_1) * (0.1573433 - Q.M2_b05)
    if Q.dr01 > 0.1335998 and Q.C3_b2 < 0.04229114:
        z += 3.507241 * (Q.dr01 - 0.1335998) * (0.04229114 - Q.C3_b2)
    if Q.dr02 < 0.1620436 and Q.n_pt_above_5 > 16.0:
        z += 0.002342965 * (0.1620436 - Q.dr02) * (Q.n_pt_above_5 - 16.0)
    if Q.n_pairs_kt_above_1 < 325.0 and Q.sum_z_dr2_top2 < 0.072817:
        z += 0.00149629 * (325.0 - Q.n_pairs_kt_above_1) * (0.072817 - Q.sum_z_dr2_top2)
    if Q.mass > 114.0172 and Q.e3_b2 < 2.306773e-05:
        z += -18.50776 * (Q.mass - 114.0172) * (2.306773e-05 - Q.e3_b2)
    if Q.mass_top40 > 122.7145 and Q.e3_b2 < 0.0007909605:
        z += 1.943983 * (Q.mass_top40 - 122.7145) * (0.0007909605 - Q.e3_b2)
    if Q.pair_mean_lndelta > -1.986684 and Q.psi_0p3 > 0.6500863:
        z += -0.4717143 * (Q.pair_mean_lndelta - -1.986684) * (Q.psi_0p3 - 0.6500863)
    if Q.m01 < 42.26122 and Q.sj3_pair_mass_min > 29.07349:
        z += 4.794589e-05 * (42.26122 - Q.m01) * (Q.sj3_pair_mass_min - 29.07349)
    if Q.mass_top40 > 83.57316 and Q.sj2_dr > 0.5076533:
        z += -0.001982431 * (Q.mass_top40 - 83.57316) * (Q.sj2_dr - 0.5076533)
    if Q.dr01 > 0.1335998 and Q.n_dr_0p05_0p1 > 6.0:
        z += -0.01558403 * (Q.dr01 - 0.1335998) * (Q.n_dr_0p05_0p1 - 6.0)
    if Q.m012 > 69.67489 and Q.sj4_zsoft > 0.1545914:
        z += 0.00825402 * (Q.m012 - 69.67489) * (Q.sj4_zsoft - 0.1545914)
    if Q.mass_top20 > 121.2735 and Q.e4 < 4.06096e-07:
        z += 13069.37 * (Q.mass_top20 - 121.2735) * (4.06096e-07 - Q.e4)
    if Q.mass_top20 > 121.2735 and Q.phi_17 > -0.02920532:
        z += -4.10905e-05 * (Q.mass_top20 - 121.2735) * (Q.phi_17 - -0.02920532)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.D2_b2 < 2.613544:
        z += -0.005082424 * (7.0 - Q.n_dr_0p2_0p4) * (2.613544 - Q.D2_b2)
    if Q.m01 < 42.26122 and Q.D2_b2 < 15.17086:
        z += 0.0001126381 * (42.26122 - Q.m01) * (15.17086 - Q.D2_b2)
    if Q.M2 < 0.06625964 and Q.dr_6 > 0.08715843:
        z += -0.5484249 * (0.06625964 - Q.M2) * (Q.dr_6 - 0.08715843)
    if Q.M3_b05 > 0.03550507 and Q.dr_5 < 0.4190095:
        z += 0.5576672 * (Q.M3_b05 - 0.03550507) * (0.4190095 - Q.dr_5)
    if Q.n_pairs_kt_above_1 < 325.0 and Q.n_lund < 14.0:
        z += 6.988173e-05 * (325.0 - Q.n_pairs_kt_above_1) * (14.0 - Q.n_lund)
    if Q.sj3_pair_mass_max < 120.6471 and Q.lne_29 < 2.13546:
        z += -1.885275e-05 * (120.6471 - Q.sj3_pair_mass_max) * (2.13546 - Q.lne_29)
    if Q.sj4_pair_mass_max > 115.5142 and Q.e4 > 7.184834e-06:
        z += 31.36629 * (Q.sj4_pair_mass_max - 115.5142) * (Q.e4 - 7.184834e-06)
    if Q.mass_top20 > 121.2735 and Q.sj3_mass1 < 13.49356:
        z += 0.000284205 * (Q.mass_top20 - 121.2735) * (13.49356 - Q.sj3_mass1)
    if Q.dr02 < 0.1620436 and Q.sj3_mass1 > 4.621465:
        z += 0.001474356 * (0.1620436 - Q.dr02) * (Q.sj3_mass1 - 4.621465)
    if Q.lund_max_lnkt > 3.22013 and Q.pt2_over_pt0 < 0.3898758:
        z += -0.511111 * (Q.lund_max_lnkt - 3.22013) * (0.3898758 - Q.pt2_over_pt0)
    if Q.pair_max_lnm2 > 6.411593 and Q.eta_81 > 0.0:
        z += -0.137355 * (Q.pair_max_lnm2 - 6.411593) * (Q.eta_81 - 0.0)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.tau54 > 0.7835935:
        z += 0.01001705 * (7.0 - Q.n_dr_0p2_0p4) * (Q.tau54 - 0.7835935)
    if Q.m01 < 3.294888 and Q.eta_12 < 0.1876221:
        z += 0.009225181 * (3.294888 - Q.m01) * (0.1876221 - Q.eta_12)
    if Q.sj3_mass3 < 1.053727 and Q.eta_33 < 0.0:
        z += 0.008925013 * (1.053727 - Q.sj3_mass3) * (0.0 - Q.eta_33)
    if Q.n_pairs_kt_above_1 < 873.0 and Q.eta_33 < 0.0:
        z += 3.652491e-06 * (873.0 - Q.n_pairs_kt_above_1) * (0.0 - Q.eta_33)
    if Q.pair_max_lnm2 > 6.411593 and Q.lund2_lndelta > -1.20523:
        z += -0.03699504 * (Q.pair_max_lnm2 - 6.411593) * (Q.lund2_lndelta - -1.20523)
    return z


def neuron_29(Q):
    z = -2.301974e-05
    return z


def neuron_30(Q):
    z = -0.07256897
    if 53.29621 <= Q.sj3_pair_mass_min < 80.02563:
        z += -0.0007695663 * Q.sj3_pair_mass_min + 0.04101497
    if Q.sj3_pair_mass_min >= 80.02563:
        z += -0.004870449 * Q.sj3_pair_mass_min + 0.3691907
    if -2.42577 <= Q.pair_mean_lndelta < -1.352792:
        z += 0.001789044 * Q.pair_mean_lndelta + 0.004339809
    if Q.pair_mean_lndelta >= -1.352792:
        z += 0.09712129 * Q.pair_mean_lndelta + 0.1333045
    if Q.mass_over_sum_pt >= 0.2243345:
        z += -1.010597 * Q.mass_over_sum_pt + 0.2267117
    if Q.mass_top5 >= 68.52153:
        z += -0.004131468 * Q.mass_top5 + 0.2830945
    if Q.mass < 79.47361:
        z += -0.004538674 * Q.mass + 0.3993614
    if 79.47361 <= Q.mass < 95.14961:
        z += -0.003808673 * Q.mass + 0.3413455
    if 95.14961 <= Q.mass < 110.2019:
        z += -0.0005769652 * Q.mass + 0.03384985
    if 110.2019 <= Q.mass < 120.653:
        z += 0.005097492 * Q.mass - 0.5914861
    if 120.653 <= Q.mass < 137.452:
        z += -0.0002145136 * Q.mass + 0.0494235
    if 137.452 <= Q.mass < 164.4374:
        z += -0.0007388494 * Q.mass + 0.1214945
    if Q.sj3_pair_mass_max < 73.24742:
        z += 0.002829921 * Q.sj3_pair_mass_max - 0.2348108
    if 73.24742 <= Q.sj3_pair_mass_max < 105.4845:
        z += -0.0009961617 * Q.sj3_pair_mass_max + 0.04543986
    if 105.4845 <= Q.sj3_pair_mass_max < 128.6605:
        z += 0.002573343 * Q.sj3_pair_mass_max - 0.3310876
    if Q.pt_entropy < 1.63996:
        z += 0.2657713 * Q.pt_entropy - 0.4358542
    if Q.tau1 >= 0.1246227:
        z += 0.7055911 * Q.tau1 - 0.08793269
    if Q.n_pairs_kt_above_1 < 101.0:
        z += 0.0003955315 * Q.n_pairs_kt_above_1 - 0.1448716
    if 101.0 <= Q.n_pairs_kt_above_1 < 366.0:
        z += -9.744908e-06 * Q.n_pairs_kt_above_1 - 0.1039387
    if 366.0 <= Q.n_pairs_kt_above_1 < 873.0:
        z += 0.0002120421 * Q.n_pairs_kt_above_1 - 0.1851128
    if Q.tau2 < 0.05997694:
        z += 2.123361 * Q.tau2 - 0.1054717
    if 0.05997694 <= Q.tau2 < 0.08811137:
        z += -0.7777315 * Q.tau2 + 0.06852699
    if Q.e3_b2 < 0.0004126585:
        z += -216.7289 * Q.e3_b2 + 0.08943505
    if Q.sj3_mass3 < 1.617171:
        z += 0.0156099 * Q.sj3_mass3 - 0.02524387
    if Q.n_dr_0_0p05 >= 10.0:
        z += -0.0004739217 * Q.n_dr_0_0p05 + 0.004739217
    if Q.n_pairs_kt_above_3 < 28.0:
        z += 0.0007785306 * Q.n_pairs_kt_above_3 - 0.0489957
    if 28.0 <= Q.n_pairs_kt_above_3 < 88.0:
        z += 0.0004532808 * Q.n_pairs_kt_above_3 - 0.03988871
    if 62.03827 <= Q.sd_mass < 78.4753:
        z += 0.0007513486 * Q.sd_mass - 0.04661237
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += -0.006846062 * Q.sd_mass + 0.5495967
    if 88.81751 <= Q.sd_mass < 94.55722:
        z += -0.0006367057 * Q.sd_mass - 0.001902883
    if 94.55722 <= Q.sd_mass < 106.7501:
        z += 0.005246293 * Q.sd_mass - 0.5581829
    if 106.7501 <= Q.sd_mass < 175.9333:
        z += -0.0007859604 * Q.sd_mass + 0.08576062
    if Q.sd_mass >= 175.9333:
        z += -0.01137399 * Q.sd_mass + 1.948549
    if 32.7897 <= Q.m012 < 49.92073:
        z += -0.001065854 * Q.m012 + 0.03494904
    if Q.m012 >= 49.92073:
        z += 0.004097402 * Q.m012 - 0.2228045
    if Q.N2 < 0.2734424:
        z += 0.2854697 * Q.N2 - 0.07805954
    if Q.z_top15_slots >= 0.8008865:
        z += 0.3381249 * Q.z_top15_slots - 0.2707997
    if Q.sj2_mass2 < 1.851735:
        z += -0.07252767 * Q.sj2_mass2 + 0.134302
    if Q.dr_25 < 0.2613173:
        z += 0.1625842 * Q.dr_25 - 0.04248605
    if Q.tau3 < 0.02808351:
        z += -3.017841 * Q.tau3 + 0.08475155
    if Q.lund_max_lndelta >= -0.615738:
        z += 0.3214642 * Q.lund_max_lndelta + 0.1979377
    if Q.z_1st >= 0.5700296:
        z += 1.125905 * Q.z_1st - 0.6417993
    if Q.sj4_pair_mass_max >= 69.83554:
        z += -0.0007482936 * Q.sj4_pair_mass_max + 0.05225749
    if Q.sum_z_dr2_top2 < 0.0005509685:
        z += 39.37417 * Q.sum_z_dr2_top2 - 0.02169393
    if Q.mass_top50 < 135.5368:
        z += -0.002547235 * Q.mass_top50 + 0.3452441
    if Q.mass_top40 < 97.12186:
        z += 0.01109963 * Q.mass_top40 - 1.078016
    if Q.mass_top30 < 79.54717:
        z += -0.007762353 * Q.mass_top30 + 0.6174732
    if Q.z_top40_slots >= 0.9077314:
        z += 1.011649 * Q.z_top40_slots - 0.9183055
    if Q.sj4_dr_min < 0.1631992:
        z += 0.3110264 * Q.sj4_dr_min - 0.05075926
    if Q.N3_b05 < 0.7591346:
        z += 0.01309252 * Q.N3_b05 - 0.009938986
    if Q.e4_b05 < 9.76395e-05:
        z += -159.7236 * Q.e4_b05 + 0.01559533
    if Q.mass_top5 > 68.52153 and Q.dr_32 < 0.2690436:
        z += 0.007119017 * (Q.mass_top5 - 68.52153) * (0.2690436 - Q.dr_32)
    if Q.mass < 164.4374 and Q.lund3_lndelta > -1.902701:
        z += 0.0003437974 * (164.4374 - Q.mass) * (Q.lund3_lndelta - -1.902701)
    if Q.mass < 164.4374 and Q.n_pt_above_5 > 10.0:
        z += 1.321465e-05 * (164.4374 - Q.mass) * (Q.n_pt_above_5 - 10.0)
    if Q.tau2 < 0.08811137 and Q.dr12 > 0.2677232:
        z += -5.586064 * (0.08811137 - Q.tau2) * (Q.dr12 - 0.2677232)
    if Q.tau2 < 0.08811137 and Q.M3 < 0.0391767:
        z += -43.40176 * (0.08811137 - Q.tau2) * (0.0391767 - Q.M3)
    if Q.sj3_pair_mass_max < 105.4845 and Q.dr_32 < 0.08614914:
        z += 0.01584576 * (105.4845 - Q.sj3_pair_mass_max) * (0.08614914 - Q.dr_32)
    if Q.sj3_pair_mass_min > 53.29621 and Q.sj3_z3 > 0.009630718:
        z += 0.001985227 * (Q.sj3_pair_mass_min - 53.29621) * (Q.sj3_z3 - 0.009630718)
    if Q.sd_mass > 78.4753 and Q.lund3_lnz < -4.934509:
        z += 3.604549e-05 * (Q.sd_mass - 78.4753) * (-4.934509 - Q.lund3_lnz)
    if Q.n_pairs_kt_above_3 < 88.0 and Q.sj3_z2 > 0.1272236:
        z += 0.001385681 * (88.0 - Q.n_pairs_kt_above_3) * (Q.sj3_z2 - 0.1272236)
    if Q.N2 < 0.2734424 and Q.pt1_over_pt0 < 0.5339053:
        z += 0.5125557 * (0.2734424 - Q.N2) * (0.5339053 - Q.pt1_over_pt0)
    if Q.N2 < 0.2734424 and Q.n_dr_0p05_0p1 > 16.0:
        z += -0.6318763 * (0.2734424 - Q.N2) * (Q.n_dr_0p05_0p1 - 16.0)
    if Q.mass < 164.4374 and Q.sj3_dr_max > 0.6446256:
        z += -0.00170777 * (164.4374 - Q.mass) * (Q.sj3_dr_max - 0.6446256)
    if Q.sj2_mass2 < 1.851735 and Q.dr_17 < 0.2325538:
        z += 0.07501473 * (1.851735 - Q.sj2_mass2) * (0.2325538 - Q.dr_17)
    if Q.mass < 120.653 and Q.sj2_mass2 < 1.851735:
        z += -0.002113279 * (120.653 - Q.mass) * (1.851735 - Q.sj2_mass2)
    if Q.mass < 79.47361 and Q.sj2_mass2 < 1.851735:
        z += -2.628768e-05 * (79.47361 - Q.mass) * (1.851735 - Q.sj2_mass2)
    if Q.mass < 95.14961 and Q.sj2_mass2 < 1.851735:
        z += 0.002165796 * (95.14961 - Q.mass) * (1.851735 - Q.sj2_mass2)
    if Q.N2 < 0.2734424 and Q.lne_8 > 2.942793:
        z += -0.5781242 * (0.2734424 - Q.N2) * (Q.lne_8 - 2.942793)
    if Q.sj3_mass3 < 1.617171 and Q.tau32_b2 < 0.6550361:
        z += 0.1077791 * (1.617171 - Q.sj3_mass3) * (0.6550361 - Q.tau32_b2)
    if Q.sd_mass > 106.7501 and Q.tau32 > 0.6513932:
        z += 0.01404612 * (Q.sd_mass - 106.7501) * (Q.tau32 - 0.6513932)
    if Q.tau2 < 0.08811137 and Q.tau32_b2 < 0.5464273:
        z += 0.5082186 * (0.08811137 - Q.tau2) * (0.5464273 - Q.tau32_b2)
    if Q.e3_b2 < 0.0004126585 and Q.sj2_mass2 < 1.851735:
        z += -150.7454 * (0.0004126585 - Q.e3_b2) * (1.851735 - Q.sj2_mass2)
    if Q.sd_mass > 78.4753 and Q.sj2_dr < 0.3039183:
        z += 0.01004805 * (Q.sd_mass - 78.4753) * (0.3039183 - Q.sj2_dr)
    if Q.e3_b2 < 0.0004126585 and Q.n_dr_0p05_0p1 < 4.0:
        z += -6.887254 * (0.0004126585 - Q.e3_b2) * (4.0 - Q.n_dr_0p05_0p1)
    if Q.e3_b2 < 0.0004126585 and Q.sj3_z1 < 0.8720575:
        z += -153.3514 * (0.0004126585 - Q.e3_b2) * (0.8720575 - Q.sj3_z1)
    if Q.m012 > 49.92073 and Q.mratio_max_012 > 0.8335772:
        z += 0.0114981 * (Q.m012 - 49.92073) * (Q.mratio_max_012 - 0.8335772)
    if Q.sd_mass > 62.03827 and Q.sj2_mass2 < 21.7295:
        z += 1.380054e-05 * (Q.sd_mass - 62.03827) * (21.7295 - Q.sj2_mass2)
    if Q.mass < 120.653 and Q.eccentricity > 0.8244523:
        z += 0.007772365 * (120.653 - Q.mass) * (Q.eccentricity - 0.8244523)
    if Q.mass_top5 > 68.52153 and Q.tau54 < 0.7608692:
        z += 0.009486003 * (Q.mass_top5 - 68.52153) * (0.7608692 - Q.tau54)
    if Q.mass < 164.4374 and Q.min_pair_mass > 10.59762:
        z += 0.0001499093 * (164.4374 - Q.mass) * (Q.min_pair_mass - 10.59762)
    if Q.mass < 110.2019 and Q.min_pair_mass > 10.59762:
        z += -0.001146218 * (110.2019 - Q.mass) * (Q.min_pair_mass - 10.59762)
    if Q.m012 > 49.92073 and Q.min_pair_mass > 1.514826:
        z += -0.0001159665 * (Q.m012 - 49.92073) * (Q.min_pair_mass - 1.514826)
    if Q.sj3_pair_mass_max < 105.4845 and Q.dr_9 < 0.1648004:
        z += -0.002877303 * (105.4845 - Q.sj3_pair_mass_max) * (0.1648004 - Q.dr_9)
    if Q.pair_mean_lndelta > -2.42577 and Q.dr_17 < 0.3346349:
        z += 0.05488016 * (Q.pair_mean_lndelta - -2.42577) * (0.3346349 - Q.dr_17)
    if Q.sj3_mass3 < 1.617171 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.0011115 * (1.617171 - Q.sj3_mass3) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.sj3_pair_mass_max < 105.4845 and Q.n_dr_0p4_up > 0.0:
        z += -0.0001092317 * (105.4845 - Q.sj3_pair_mass_max) * (Q.n_dr_0p4_up - 0.0)
    if Q.sd_mass > 175.9333 and Q.tau32_b2 < 0.8301995:
        z += 0.02307712 * (Q.sd_mass - 175.9333) * (0.8301995 - Q.tau32_b2)
    if Q.sd_mass > 106.7501 and Q.tau32_b2 < 0.8673543:
        z += 0.0001290551 * (Q.sd_mass - 106.7501) * (0.8673543 - Q.tau32_b2)
    if Q.sd_mass > 175.9333 and Q.tau43 > 0.568406:
        z += 0.01720257 * (Q.sd_mass - 175.9333) * (Q.tau43 - 0.568406)
    if Q.mass_over_sum_pt > 0.2243345 and Q.sj4_zsoft < 0.01399549:
        z += 22.56624 * (Q.mass_over_sum_pt - 0.2243345) * (0.01399549 - Q.sj4_zsoft)
    if Q.n_pairs_kt_above_1 < 366.0 and Q.phi_0 < -0.008700943:
        z += 0.0002175203 * (366.0 - Q.n_pairs_kt_above_1) * (-0.008700943 - Q.phi_0)
    if Q.m012 > 49.92073 and Q.phi_9 > 0.2814941:
        z += -0.004287643 * (Q.m012 - 49.92073) * (Q.phi_9 - 0.2814941)
    if Q.dr_25 < 0.2613173 and Q.lnpt_5 > 2.515047:
        z += 0.2497417 * (0.2613173 - Q.dr_25) * (Q.lnpt_5 - 2.515047)
    if Q.sd_mass > 78.4753 and Q.pt2_over_pt0 < 0.2816529:
        z += -0.001343288 * (Q.sd_mass - 78.4753) * (0.2816529 - Q.pt2_over_pt0)
    if Q.tau2 < 0.08811137 and Q.ecf_g41 > 6.216954e-05:
        z += 222.59 * (0.08811137 - Q.tau2) * (Q.ecf_g41 - 6.216954e-05)
    if Q.sj3_mass3 < 1.617171 and Q.sj3_dr13 > 0.1942334:
        z += -0.005948224 * (1.617171 - Q.sj3_mass3) * (Q.sj3_dr13 - 0.1942334)
    if Q.e3_b2 < 0.0004126585 and Q.tau43_b2 < 0.8314744:
        z += 3.788879 * (0.0004126585 - Q.e3_b2) * (0.8314744 - Q.tau43_b2)
    if Q.z_top15_slots > 0.8008865 and Q.eta_26 > -0.21521:
        z += -0.256471 * (Q.z_top15_slots - 0.8008865) * (Q.eta_26 - -0.21521)
    if Q.m012 > 49.92073 and Q.dr_34 > 0.2187045:
        z += 7.551599e-05 * (Q.m012 - 49.92073) * (Q.dr_34 - 0.2187045)
    if Q.n_pairs_kt_above_1 < 873.0 and Q.phi_8 < 0.0:
        z += 5.053831e-05 * (873.0 - Q.n_pairs_kt_above_1) * (0.0 - Q.phi_8)
    return z


def neuron_31(Q):
    z = -2.82251e-06
    return z


def neuron_32(Q):
    z = -1.232198e-06
    return z


def neuron_33(Q):
    z = 2.506875
    if Q.n_pairs_kt_above_3 < 28.0:
        z += 0.03466488 * Q.n_pairs_kt_above_3 - 1.565799
    if 28.0 <= Q.n_pairs_kt_above_3 < 111.0:
        z += 0.003189774 * Q.n_pairs_kt_above_3 - 0.684496
    if 111.0 <= Q.n_pairs_kt_above_3 < 220.0:
        z += 0.003031478 * Q.n_pairs_kt_above_3 - 0.6669252
    if Q.n_lund >= 10.0:
        z += -0.09887541 * Q.n_lund + 0.9887541
    if Q.m012 >= 44.62237:
        z += 0.005443068 * Q.m012 - 0.2428826
    if Q.tau32 < 0.6800935:
        z += 0.3759181 * Q.tau32 - 0.03624023
    if 0.6800935 <= Q.tau32 < 0.863865:
        z += -1.193979 * Q.tau32 + 1.031437
    if Q.lne_2 < 4.905275:
        z += 0.2660511 * Q.lne_2 - 1.305054
    if Q.ecf_g31 < 0.00553358:
        z += 145.5397 * Q.ecf_g31 - 1.2252
    if 0.00553358 <= Q.ecf_g31 < 0.006877516:
        z += 272.5226 * Q.ecf_g31 - 1.927869
    if 0.006877516 <= Q.ecf_g31 < 0.01162881:
        z += 174.18 * Q.ecf_g31 - 1.251517
    if Q.ecf_g31 >= 0.01162881:
        z += 126.9828 * Q.ecf_g31 - 0.7026697
    if Q.dr01 < 0.0259406:
        z += -7.235521 * Q.dr01 + 0.1876938
    if Q.sum_z_dr2_top20 >= 0.01911708:
        z += -3.393244 * Q.sum_z_dr2_top20 + 0.06486893
    if Q.mass_top10 < 56.50998:
        z += -0.0001870523 * Q.mass_top10 + 0.01057032
    if Q.sj2_mass2 < 1.851735:
        z += 0.2655337 * Q.sj2_mass2 - 0.1283067
    if 1.851735 <= Q.sj2_mass2 < 11.87755:
        z += -0.03624556 * Q.sj2_mass2 + 0.4305086
    if Q.tau43 < 0.9062492:
        z += -1.202025 * Q.tau43 + 1.089334
    if Q.pair_mean_lndelta >= -2.42577:
        z += -0.01858615 * Q.pair_mean_lndelta - 0.04508572
    if Q.sj2_dr < 0.2083105:
        z += 0.0490392 * Q.sj2_dr - 0.01021538
    if 37.19471 <= Q.sj3_pair_mass_min < 80.02563:
        z += -0.00106964 * Q.sj3_pair_mass_min + 0.03978495
    if Q.sj3_pair_mass_min >= 80.02563:
        z += -0.008797092 * Q.sj3_pair_mass_min + 0.6581791
    z += -0.0261438 * Q.n_pt_above_1
    if Q.M2_b2 < 0.0417856:
        z += -8.424299 * Q.M2_b2 + 0.3520144
    if Q.n_pairs_kt_above_1 < 101.0:
        z += 0.01297606 * Q.n_pairs_kt_above_1 - 1.400299
    if 101.0 <= Q.n_pairs_kt_above_1 < 325.0:
        z += 0.005331279 * Q.n_pairs_kt_above_1 - 0.6281767
    if 325.0 <= Q.n_pairs_kt_above_1 < 462.0:
        z += 0.003123647 * Q.n_pairs_kt_above_1 + 0.08930359
    if Q.n_pairs_kt_above_1 >= 462.0:
        z += 0.001763957 * Q.n_pairs_kt_above_1 + 0.7174803
    if 79.47361 <= Q.mass < 100.4835:
        z += -0.0260833 * Q.mass + 2.072934
    if 100.4835 <= Q.mass < 110.2019:
        z += -0.002045864 * Q.mass - 0.3424306
    if Q.mass >= 110.2019:
        z += -0.01060431 * Q.mass + 0.600726
    if Q.n_pairs_kt_above_10 < 7.0:
        z += 0.02068289 * Q.n_pairs_kt_above_10 - 0.05711008
    if Q.n_pairs_kt_above_10 >= 7.0:
        z += 0.01252431 * Q.n_pairs_kt_above_10
    if Q.sj4_pair_mass_max < 69.83554:
        z += 0.004118856 * Q.sj4_pair_mass_max - 0.1889517
    if 69.83554 <= Q.sj4_pair_mass_max < 101.849:
        z += -0.003082793 * Q.sj4_pair_mass_max + 0.3139794
    if Q.n_lund_kt_above_1 < 8.0:
        z += -0.05262431 * Q.n_lund_kt_above_1 + 0.4209945
    if Q.dr02 < 0.02982432:
        z += -8.637734 * Q.dr02 + 0.2576146
    if Q.sj3_dr_min >= 0.3526989:
        z += -0.4170589 * Q.sj3_dr_min + 0.1470962
    if Q.lne_1 >= 3.745821:
        z += 0.07228645 * Q.lne_1 - 0.2707721
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.004133475 * Q.n_dr_0p2_0p4 - 0.02893432
    if Q.n_dr_0p2_0p4 >= 9.0:
        z += -0.02245983 * Q.n_dr_0p2_0p4 + 0.2021385
    if Q.C2_b05 < 0.1620185:
        z += 7.884616 * Q.C2_b05 - 1.277454
    if Q.mass_top20 >= 114.4658:
        z += -0.004808215 * Q.mass_top20 + 0.5503762
    if Q.sj3_mass2 < 3.298199:
        z += 0.1265834 * Q.sj3_mass2 - 0.4174973
    if Q.n_for_90pct >= 14.0:
        z += -0.03149275 * Q.n_for_90pct + 0.4408985
    if Q.M3 < 0.01517988:
        z += -15.37614 * Q.M3 + 0.5319194
    if 0.01517988 <= Q.M3 < 0.03459382:
        z += -34.47321 * Q.M3 + 0.8218105
    if Q.M3 >= 0.03459382:
        z += -19.09706 * Q.M3 + 0.2898911
    if 42.31629 <= Q.sd_mass < 83.61981:
        z += 0.008851788 * Q.sd_mass - 0.3745748
    if 83.61981 <= Q.sd_mass < 111.701:
        z += -0.0184028 * Q.sd_mass + 1.904449
    if Q.sd_mass >= 111.701:
        z += 0.003787045 * Q.sd_mass - 0.5741784
    if Q.N3_b05 < 1.33105:
        z += 0.8341714 * Q.N3_b05 - 1.110324
    if Q.pt_dispersion >= 0.2706555:
        z += -1.679156 * Q.pt_dispersion + 0.4544728
    if Q.mratio_max_012 >= 0.787882:
        z += 1.3583 * Q.mratio_max_012 - 1.07018
    if Q.sum_z_dr2_top3 < 0.004077497:
        z += -62.47669 * Q.sum_z_dr2_top3 + 0.2547485
    if Q.tau5 < 0.05613495:
        z += -9.112477 * Q.tau5 + 0.5115284
    if Q.D3 < 0.4936772:
        z += -0.3643764 * Q.D3 + 0.1798843
    if Q.N2_b05 < 0.3436326:
        z += 2.030415 * Q.N2_b05 - 0.6977167
    if Q.n_dr_0p1_0p2 >= 7.0:
        z += -0.01346441 * Q.n_dr_0p1_0p2 + 0.09425089
    if Q.n_dr_0p4_up >= 6.0:
        z += -0.02226918 * Q.n_dr_0p4_up + 0.1336151
    if Q.D2 < 1.00791:
        z += -0.3794915 * Q.D2 + 0.3824934
    if Q.C3_b05 < 0.2250047:
        z += 7.908311 * Q.C3_b05 - 1.779407
    if Q.C3 < 0.1271654:
        z += -5.898738 * Q.C3 + 0.7501155
    if Q.pair_max_lnkt >= 3.143912:
        z += -0.1880468 * Q.pair_max_lnkt + 0.5912026
    if Q.M2_b05 < 0.1375297:
        z += 0.7474343 * Q.M2_b05 - 0.1027944
    if Q.ecf_g31 > 0.00553358 and Q.pair_mean_lnz < -1.332584:
        z += -69.1655 * (Q.ecf_g31 - 0.00553358) * (-1.332584 - Q.pair_mean_lnz)
    if Q.M2_b2 < 0.0417856 and Q.n_lund_kt_above_1 > 3.0:
        z += 0.6017888 * (0.0417856 - Q.M2_b2) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.M2_b2 < 0.0417856 and Q.C2 < 0.11544:
        z += -20.65052 * (0.0417856 - Q.M2_b2) * (0.11544 - Q.C2)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.ecf_g41 < 0.0004392119:
        z += 37.03098 * (28.0 - Q.n_pairs_kt_above_3) * (0.0004392119 - Q.ecf_g41)
    if Q.mass > 79.47361 and Q.e3_b2 < 5.85048e-05:
        z += -7.098526 * (Q.mass - 79.47361) * (5.85048e-05 - Q.e3_b2)
    if Q.M2_b2 < 0.0417856 and Q.C3_b05 < 0.1869341:
        z += -155.6706 * (0.0417856 - Q.M2_b2) * (0.1869341 - Q.C3_b05)
    if Q.M2_b2 < 0.0417856 and Q.C3 < 0.07139556:
        z += 244.3465 * (0.0417856 - Q.M2_b2) * (0.07139556 - Q.C3)
    if Q.pair_mean_lndelta > -2.42577 and Q.tau54 > 0.6772033:
        z += -2.089529 * (Q.pair_mean_lndelta - -2.42577) * (Q.tau54 - 0.6772033)
    if Q.n_pairs_kt_above_3 < 111.0 and Q.dr_2 < 0.2108315:
        z += -0.01572815 * (111.0 - Q.n_pairs_kt_above_3) * (0.2108315 - Q.dr_2)
    if Q.mass > 79.47361 and Q.lund3_lndelta > -2.817283:
        z += 0.001899972 * (Q.mass - 79.47361) * (Q.lund3_lndelta - -2.817283)
    if Q.n_pairs_kt_above_3 < 111.0 and Q.ecf_g41 > 6.216954e-05:
        z += -9.293037 * (111.0 - Q.n_pairs_kt_above_3) * (Q.ecf_g41 - 6.216954e-05)
    if Q.lne_1 > 3.745821 and Q.sj2_zsoft > 0.1869576:
        z += 0.3963695 * (Q.lne_1 - 3.745821) * (Q.sj2_zsoft - 0.1869576)
    if Q.sj4_pair_mass_max < 101.849 and Q.lnpt_4 > 3.230063:
        z += 0.001843446 * (101.849 - Q.sj4_pair_mass_max) * (Q.lnpt_4 - 3.230063)
    if Q.n_lund > 10.0 and Q.n_lund_kt_above_5 > 1.0:
        z += 0.02479088 * (Q.n_lund - 10.0) * (Q.n_lund_kt_above_5 - 1.0)
    if Q.m012 > 44.62237 and Q.D3_b2 < 0.01408252:
        z += 0.3827498 * (Q.m012 - 44.62237) * (0.01408252 - Q.D3_b2)
    if Q.sj3_pair_mass_min > 37.19471 and Q.C2_b2 < 0.221369:
        z += 0.03347563 * (Q.sj3_pair_mass_min - 37.19471) * (0.221369 - Q.C2_b2)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.dr_0 < 0.1451741:
        z += 0.1597685 * (28.0 - Q.n_pairs_kt_above_3) * (0.1451741 - Q.dr_0)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.sd_zg > 0.2075336:
        z += 0.2019947 * (7.0 - Q.n_dr_0p2_0p4) * (Q.sd_zg - 0.2075336)
    if Q.sj4_pair_mass_max < 101.849 and Q.dr_35 < 0.07750231:
        z += -0.009943516 * (101.849 - Q.sj4_pair_mass_max) * (0.07750231 - Q.dr_35)
    if Q.n_pairs_kt_above_3 < 111.0 and Q.dr_3 < 0.1619198:
        z += -0.006160563 * (111.0 - Q.n_pairs_kt_above_3) * (0.1619198 - Q.dr_3)
    if Q.M2_b2 < 0.0417856 and Q.D3_b2 < 0.001184159:
        z += 8623.142 * (0.0417856 - Q.M2_b2) * (0.001184159 - Q.D3_b2)
    if Q.sj2_mass2 < 11.87755 and Q.C2_b2 < 0.07094338:
        z += 0.1815378 * (11.87755 - Q.sj2_mass2) * (0.07094338 - Q.C2_b2)
    if Q.M3 > 0.01517988 and Q.dr_37 < 0.2338039:
        z += 23.23216 * (Q.M3 - 0.01517988) * (0.2338039 - Q.dr_37)
    if Q.mass > 110.2019 and Q.D3_b2 < 0.007827335:
        z += -0.0986368 * (Q.mass - 110.2019) * (0.007827335 - Q.D3_b2)
    if Q.tau32 < 0.863865 and Q.eccentricity > 0.3996864:
        z += 0.5413654 * (0.863865 - Q.tau32) * (Q.eccentricity - 0.3996864)
    if Q.pair_mean_lndelta > -2.42577 and Q.n_pt_above_50 > 3.0:
        z += 0.1196935 * (Q.pair_mean_lndelta - -2.42577) * (Q.n_pt_above_50 - 3.0)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.dr_4 < 0.2082728:
        z += -0.2985085 * (7.0 - Q.n_dr_0p2_0p4) * (0.2082728 - Q.dr_4)
    if Q.sj3_dr_min > 0.3526989 and Q.sum_z_dr2_top2 < 0.072817:
        z += 7.66183 * (Q.sj3_dr_min - 0.3526989) * (0.072817 - Q.sum_z_dr2_top2)
    if Q.sj4_pair_mass_max < 101.849 and Q.C2_b2 > 0.02858957:
        z += -0.02529548 * (101.849 - Q.sj4_pair_mass_max) * (Q.C2_b2 - 0.02858957)
    if Q.n_lund > 10.0 and Q.dr_19 > 0.5935318:
        z += -0.09163087 * (Q.n_lund - 10.0) * (Q.dr_19 - 0.5935318)
    if Q.M3 < 0.03459382 and Q.eta_20 < -0.1281738:
        z += -13.21979 * (0.03459382 - Q.M3) * (-0.1281738 - Q.eta_20)
    if Q.tau32 < 0.6800935 and Q.dr_21 < 0.04063514:
        z += 27.02425 * (0.6800935 - Q.tau32) * (0.04063514 - Q.dr_21)
    if Q.sum_z_dr2_top20 > 0.01911708 and Q.phi_68 > 0.0:
        z += 0.516956 * (Q.sum_z_dr2_top20 - 0.01911708) * (Q.phi_68 - 0.0)
    if Q.sd_mass > 83.61981 and Q.dr_6 < 0.4370905:
        z += -0.001112907 * (Q.sd_mass - 83.61981) * (0.4370905 - Q.dr_6)
    if Q.dr02 < 0.02982432 and Q.lund1_lndelta < -0.4705779:
        z += -2.057672 * (0.02982432 - Q.dr02) * (-0.4705779 - Q.lund1_lndelta)
    if Q.ecf_g31 > 0.00553358 and Q.D3_b2 < 1.462588:
        z += -48.9466 * (Q.ecf_g31 - 0.00553358) * (1.462588 - Q.D3_b2)
    if Q.sj4_pair_mass_max < 69.83554 and Q.D3_b2 < 1.892933e-05:
        z += 234.7738 * (69.83554 - Q.sj4_pair_mass_max) * (1.892933e-05 - Q.D3_b2)
    return z


def neuron_34(Q):
    z = 2.463992e-07
    return z


def neuron_35(Q):
    z = 0.0008335161
    if Q.lund_max_lnkt < 3.734077:
        z += 0.01425992 * Q.lund_max_lnkt - 0.05324766
    if Q.M2_b2 < 0.02656308:
        z += -0.6676816 * Q.M2_b2 + 0.04515387
    if 0.02656308 <= Q.M2_b2 < 0.06762785:
        z += -0.2810371 * Q.M2_b2 + 0.0348834
    if Q.M2_b2 >= 0.06762785:
        z += 0.3866445 * Q.M2_b2 - 0.01027047
    if 5.0 <= Q.n_pairs_kt_above_3 < 34.0:
        z += -0.0006770866 * Q.n_pairs_kt_above_3 + 0.003385433
    if 34.0 <= Q.n_pairs_kt_above_3 < 78.0:
        z += -0.0003249526 * Q.n_pairs_kt_above_3 - 0.008587122
    if Q.n_pairs_kt_above_3 >= 78.0:
        z += -1.47316e-05 * Q.n_pairs_kt_above_3 - 0.03278436
    if Q.pair_mean_lndelta >= -1.352792:
        z += -0.1133987 * Q.pair_mean_lndelta - 0.1534048
    if Q.n_pairs_kt_above_1 < 58.0:
        z += -0.0005742078 * Q.n_pairs_kt_above_1 + 0.02042114
    if 58.0 <= Q.n_pairs_kt_above_1 < 325.0:
        z += 4.825059e-05 * Q.n_pairs_kt_above_1 - 0.01568144
    if Q.sd_rg >= 0.4731839:
        z += -0.1121012 * Q.sd_rg + 0.05304446
    if Q.m012 >= 36.38136:
        z += -0.001627848 * Q.m012 + 0.05922333
    if 78.76797 <= Q.sj4_pair_mass_max < 88.5845:
        z += 0.004822325 * Q.sj4_pair_mass_max - 0.3798448
    if Q.sj4_pair_mass_max >= 88.5845:
        z += -0.001330932 * Q.sj4_pair_mass_max + 0.1652384
    if 71.96396 <= Q.mass < 79.47361:
        z += 0.003756194 * Q.mass - 0.2703106
    if 79.47361 <= Q.mass < 95.14961:
        z += 0.001665581 * Q.mass - 0.104162
    if 95.14961 <= Q.mass < 105.7234:
        z += -0.0006593382 * Q.mass + 0.1170531
    if 105.7234 <= Q.mass < 110.2019:
        z += -0.001158397 * Q.mass + 0.1698153
    if Q.mass >= 110.2019:
        z += 0.0001061303 * Q.mass + 0.03046197
    if Q.mass_top50 >= 112.2327:
        z += 0.0003367873 * Q.mass_top50 - 0.03779854
    if Q.N2_b2 >= 0.1528932:
        z += 0.0831705 * Q.N2_b2 - 0.0127162
    if Q.sj3_pairmin_over_m >= 0.2477126:
        z += -0.01415633 * Q.sj3_pairmin_over_m + 0.003506701
    if 56.73313 <= Q.sj3_pair_mass_max < 76.91486:
        z += -0.002993385 * Q.sj3_pair_mass_max + 0.1698241
    if 76.91486 <= Q.sj3_pair_mass_max < 93.87663:
        z += 0.0003206774 * Q.sj3_pair_mass_max - 0.08507654
    if Q.sj3_pair_mass_max >= 93.87663:
        z += 0.0007028845 * Q.sj3_pair_mass_max - 0.1209569
    if Q.sj3_mass3 < 1.617171:
        z += 0.00369342 * Q.sj3_mass3 - 0.00597289
    if 62.03827 <= Q.sd_mass < 78.4753:
        z += -0.002176502 * Q.sd_mass + 0.1350264
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += 0.006251197 * Q.sd_mass - 0.5263398
    if 88.81751 <= Q.sd_mass < 94.55722:
        z += 0.001795862 * Q.sd_mass - 0.130628
    if 94.55722 <= Q.sd_mass < 106.7501:
        z += -0.004512079 * Q.sd_mass + 0.4658334
    if 106.7501 <= Q.sd_mass < 111.701:
        z += -0.004300181 * Q.sd_mass + 0.4432132
    if 111.701 <= Q.sd_mass < 127.8042:
        z += -0.001480776 * Q.sd_mass + 0.1282829
    if 127.8042 <= Q.sd_mass < 175.9333:
        z += 0.0009194033 * Q.sd_mass - 0.1784701
    if Q.sd_mass >= 175.9333:
        z += -0.00180015 * Q.sd_mass + 0.2999899
    if Q.lnptrel_30 >= -5.77735:
        z += 0.04127617 * Q.lnptrel_30 + 0.2384669
    if Q.ecf_g31 >= 0.004728078:
        z += -6.324849 * Q.ecf_g31 + 0.02990438
    if Q.pair_mean_lnm2 >= 4.069088:
        z += -0.04911755 * Q.pair_mean_lnm2 + 0.1998636
    if Q.lne_0 >= 6.304449:
        z += -0.03647844 * Q.lne_0 + 0.2299765
    if Q.mratio_min_012 < 0.01744666:
        z += -4.253861 * Q.mratio_min_012 + 0.07421565
    if Q.pair_max_lnm2 < 5.152094:
        z += 0.009194495 * Q.pair_max_lnm2 - 0.04737091
    if 0.002431555 <= Q.lam2 < 0.01524581:
        z += -0.4565598 * Q.lam2 + 0.00111015
    if Q.lam2 >= 0.01524581:
        z += -0.716735 * Q.lam2 + 0.005076731
    if Q.tau32 < 0.5498426:
        z += 0.1999333 * Q.tau32 - 0.1099318
    if Q.tau21 < 0.5440886:
        z += 0.02131057 * Q.tau21 - 0.01159484
    if Q.mass_top20 >= 97.09528:
        z += 0.0009682794 * Q.mass_top20 - 0.09401536
    if Q.tau32_b2 < 0.6203012:
        z += 0.07807943 * Q.tau32_b2 - 0.04843277
    if Q.n_pairs_kt_above_10 < 5.0:
        z += -0.007558083 * Q.n_pairs_kt_above_10 + 0.02783498
    if 5.0 <= Q.n_pairs_kt_above_10 < 23.0:
        z += 0.0005530795 * Q.n_pairs_kt_above_10 - 0.01272083
    if Q.tau1 < 0.06074238:
        z += -0.9350081 * Q.tau1 + 0.05679462
    if Q.D3_b2 >= 0.2418808:
        z += 0.002815985 * Q.D3_b2 - 0.0006811328
    if Q.mass_top15 >= 83.68425:
        z += 0.001531337 * Q.mass_top15 - 0.1281488
    if Q.N2_b2 > 0.1528932 and Q.lne_20 > 0.1290046:
        z += -0.1333937 * (Q.N2_b2 - 0.1528932) * (Q.lne_20 - 0.1290046)
    if Q.n_pairs_kt_above_1 < 325.0 and Q.tau2 > 0.07264571:
        z += -0.00273067 * (325.0 - Q.n_pairs_kt_above_1) * (Q.tau2 - 0.07264571)
    if Q.mass > 95.14961 and Q.pt_balance01 < 0.3035327:
        z += -0.003324339 * (Q.mass - 95.14961) * (0.3035327 - Q.pt_balance01)
    if Q.mass > 79.47361 and Q.sj4_dr_min < 0.1302765:
        z += 0.006170284 * (Q.mass - 79.47361) * (0.1302765 - Q.sj4_dr_min)
    if Q.n_pairs_kt_above_1 < 325.0 and Q.lnpt_7 > 2.07354:
        z += -3.265234e-05 * (325.0 - Q.n_pairs_kt_above_1) * (Q.lnpt_7 - 2.07354)
    if Q.sj3_mass3 < 1.617171 and Q.tau32_b2 < 0.3377343:
        z += -0.3804596 * (1.617171 - Q.sj3_mass3) * (0.3377343 - Q.tau32_b2)
    if Q.M2_b2 < 0.06762785 and Q.sj3_dr_max > 0.2345694:
        z += -0.4856249 * (0.06762785 - Q.M2_b2) * (Q.sj3_dr_max - 0.2345694)
    if Q.n_pairs_kt_above_1 < 325.0 and Q.tau43 < 0.8709334:
        z += -0.0003631343 * (325.0 - Q.n_pairs_kt_above_1) * (0.8709334 - Q.tau43)
    if Q.mass > 110.2019 and Q.psi_0p3 > 0.6500863:
        z += -0.003684965 * (Q.mass - 110.2019) * (Q.psi_0p3 - 0.6500863)
    if Q.n_pairs_kt_above_1 < 58.0 and Q.tau32_b2 < 0.4680886:
        z += -0.001134709 * (58.0 - Q.n_pairs_kt_above_1) * (0.4680886 - Q.tau32_b2)
    if Q.m012 > 36.38136 and Q.dr12 > 0.08983921:
        z += 0.001659504 * (Q.m012 - 36.38136) * (Q.dr12 - 0.08983921)
    if Q.M2_b2 < 0.06762785 and Q.D2_b05 > 1.102602:
        z += 0.1944985 * (0.06762785 - Q.M2_b2) * (Q.D2_b05 - 1.102602)
    if Q.n_pairs_kt_above_1 < 325.0 and Q.min_pair_mass > 10.59762:
        z += 1.228983e-05 * (325.0 - Q.n_pairs_kt_above_1) * (Q.min_pair_mass - 10.59762)
    if Q.sd_mass > 175.9333 and Q.e4 > 3.53193e-06:
        z += 56.68787 * (Q.sd_mass - 175.9333) * (Q.e4 - 3.53193e-06)
    if Q.sj3_mass3 < 1.617171 and Q.sj3_mass2 < 4.326415:
        z += 0.004370701 * (1.617171 - Q.sj3_mass3) * (4.326415 - Q.sj3_mass2)
    if Q.ecf_g31 > 0.004728078 and Q.dr_17 < 0.002489417:
        z += -8221.177 * (Q.ecf_g31 - 0.004728078) * (0.002489417 - Q.dr_17)
    if Q.n_pairs_kt_above_3 > 5.0 and Q.sj3_mass2 < 3.298199:
        z += -0.0004764818 * (Q.n_pairs_kt_above_3 - 5.0) * (3.298199 - Q.sj3_mass2)
    if Q.M2_b2 < 0.06762785 and Q.sj3_mass2 < 26.98412:
        z += 0.004063847 * (0.06762785 - Q.M2_b2) * (26.98412 - Q.sj3_mass2)
    if Q.pair_mean_lnm2 > 4.069088 and Q.dr_32 < 0.3168775:
        z += 0.1358818 * (Q.pair_mean_lnm2 - 4.069088) * (0.3168775 - Q.dr_32)
    if Q.M2_b2 > 0.02656308 and Q.dr_25 < 0.01566153:
        z += -81.1102 * (Q.M2_b2 - 0.02656308) * (0.01566153 - Q.dr_25)
    if Q.n_pairs_kt_above_3 > 5.0 and Q.dr_32 < 0.05157804:
        z += -0.005898381 * (Q.n_pairs_kt_above_3 - 5.0) * (0.05157804 - Q.dr_32)
    if Q.ecf_g31 > 0.004728078 and Q.e3_b2 > 0.0007909605:
        z += -1988.55 * (Q.ecf_g31 - 0.004728078) * (Q.e3_b2 - 0.0007909605)
    if Q.ecf_g31 > 0.004728078 and Q.sj4_zsoft < 0.01399549:
        z += 679.504 * (Q.ecf_g31 - 0.004728078) * (0.01399549 - Q.sj4_zsoft)
    if Q.M2_b2 > 0.02656308 and Q.jet_pt > 584.4842:
        z += 0.005531329 * (Q.M2_b2 - 0.02656308) * (Q.jet_pt - 584.4842)
    if Q.sd_mass > 78.4753 and Q.tau43_b2 < 0.6509029:
        z += 0.00253632 * (Q.sd_mass - 78.4753) * (0.6509029 - Q.tau43_b2)
    if Q.sj3_pair_mass_max > 56.73313 and Q.sj3_z3 > 0.02858644:
        z += -0.004203016 * (Q.sj3_pair_mass_max - 56.73313) * (Q.sj3_z3 - 0.02858644)
    if Q.sd_mass > 62.03827 and Q.dr_25 < 0.01566153:
        z += 0.02507624 * (Q.sd_mass - 62.03827) * (0.01566153 - Q.dr_25)
    if Q.sj3_pair_mass_max > 76.91486 and Q.ecf_g41 > 0.0001644218:
        z += -1.674611 * (Q.sj3_pair_mass_max - 76.91486) * (Q.ecf_g41 - 0.0001644218)
    if Q.m012 > 36.38136 and Q.phi_30 < -0.3327637:
        z += 0.002396729 * (Q.m012 - 36.38136) * (-0.3327637 - Q.phi_30)
    if Q.m012 > 36.38136 and Q.eta_55 > 0.1072388:
        z += 0.001117897 * (Q.m012 - 36.38136) * (Q.eta_55 - 0.1072388)
    if Q.M2_b2 > 0.02656308 and Q.sj3_mass1 < 36.36236:
        z += 0.02438826 * (Q.M2_b2 - 0.02656308) * (36.36236 - Q.sj3_mass1)
    if Q.tau32 < 0.5498426 and Q.phi_33 < 0.1326904:
        z += -0.07606465 * (0.5498426 - Q.tau32) * (0.1326904 - Q.phi_33)
    if Q.mratio_min_012 < 0.01744666 and Q.phi_39 > -0.282959:
        z += -9.090427 * (0.01744666 - Q.mratio_min_012) * (Q.phi_39 - -0.282959)
    if Q.tau32 < 0.5498426 and Q.eta_77 > 0.0:
        z += 0.5452215 * (0.5498426 - Q.tau32) * (Q.eta_77 - 0.0)
    if Q.N2_b2 > 0.1528932 and Q.eta_1 < -0.2019043:
        z += 0.6884212 * (Q.N2_b2 - 0.1528932) * (-0.2019043 - Q.eta_1)
    if Q.n_pairs_kt_above_3 > 78.0 and Q.lne_0 > 4.992132:
        z += 0.0006658594 * (Q.n_pairs_kt_above_3 - 78.0) * (Q.lne_0 - 4.992132)
    if Q.sj4_pair_mass_max > 78.76797 and Q.eta_26 > -0.05227661:
        z += 0.0005983795 * (Q.sj4_pair_mass_max - 78.76797) * (Q.eta_26 - -0.05227661)
    if Q.mass > 95.14961 and Q.sum_e < 1049.334:
        z += 6.959993e-07 * (Q.mass - 95.14961) * (1049.334 - Q.sum_e)
    if Q.sd_mass > 175.9333 and Q.pt2_over_pt0 < 0.3898758:
        z += -0.001750522 * (Q.sd_mass - 175.9333) * (0.3898758 - Q.pt2_over_pt0)
    if Q.mass > 95.14961 and Q.D3_b2 < 0.0225905:
        z += -0.0115224 * (Q.mass - 95.14961) * (0.0225905 - Q.D3_b2)
    if Q.m012 > 36.38136 and Q.dr_17 < 0.294021:
        z += 0.002006275 * (Q.m012 - 36.38136) * (0.294021 - Q.dr_17)
    if Q.sd_mass > 127.8042 and Q.D3 > 0.611688:
        z += -0.000850414 * (Q.sd_mass - 127.8042) * (Q.D3 - 0.611688)
    if Q.N2_b2 > 0.1528932 and Q.n_dr_0p05_0p1 < 12.0:
        z += 0.02585159 * (Q.N2_b2 - 0.1528932) * (12.0 - Q.n_dr_0p05_0p1)
    if Q.N2_b2 > 0.1528932 and Q.eta_64 > 0.0:
        z += 0.688369 * (Q.N2_b2 - 0.1528932) * (Q.eta_64 - 0.0)
    if Q.lnptrel_30 > -5.77735 and Q.sj2_mass2 < 3.928781:
        z += 0.001667119 * (Q.lnptrel_30 - -5.77735) * (3.928781 - Q.sj2_mass2)
    if Q.tau32_b2 < 0.6203012 and Q.log_sum_pt < 6.750142:
        z += 0.6239684 * (0.6203012 - Q.tau32_b2) * (6.750142 - Q.log_sum_pt)
    if Q.sd_mass > 127.8042 and Q.D3_b2 > 0.7423282:
        z += -0.0002971497 * (Q.sd_mass - 127.8042) * (Q.D3_b2 - 0.7423282)
    if Q.mass_top20 > 97.09528 and Q.z_3rd < 0.1192981:
        z += -0.006705693 * (Q.mass_top20 - 97.09528) * (0.1192981 - Q.z_3rd)
    if Q.lam2 > 0.002431555 and Q.sj2_zsoft < 0.2530865:
        z += -9.255735 * (Q.lam2 - 0.002431555) * (0.2530865 - Q.sj2_zsoft)
    if Q.pair_mean_lnm2 > 4.069088 and Q.eta_19 > -0.01043701:
        z += -0.02825326 * (Q.pair_mean_lnm2 - 4.069088) * (Q.eta_19 - -0.01043701)
    return z


def neuron_36(Q):
    z = 1.035494e-06
    return z


def neuron_37(Q):
    z = -3.70117e-05
    return z


def neuron_38(Q):
    z = -0.06407473
    if Q.mass < 105.7234:
        z += 0.04820956 * Q.mass - 4.634006
    if 105.7234 <= Q.mass < 117.4867:
        z += 0.02676029 * Q.mass - 2.366317
    if 117.4867 <= Q.mass < 164.4374:
        z += -0.01656331 * Q.mass + 2.723629
    if Q.mass_top30 < 162.7874:
        z += 0.002077265 * Q.mass_top30 - 0.3381526
    if Q.mass_top50 < 94.51361:
        z += -0.01653174 * Q.mass_top50 + 1.810144
    if 94.51361 <= Q.mass_top50 < 118.8408:
        z += -0.01525306 * Q.mass_top50 + 1.689292
    if 118.8408 <= Q.mass_top50 < 129.5874:
        z += 0.009464757 * Q.mass_top50 - 1.248192
    if 129.5874 <= Q.mass_top50 < 161.1264:
        z += 0.003388032 * Q.mass_top50 - 0.460725
    if Q.mass_top50 >= 161.1264:
        z += 0.001278679 * Q.mass_top50 - 0.1208525
    if Q.e2_b05 < 0.1894158:
        z += -1.175546 * Q.e2_b05 + 0.222667
    if Q.tau3 < 0.1024935:
        z += 2.133185 * Q.tau3 - 0.2186375
    if Q.e3_b2 < 0.0001729927:
        z += 134.5121 * Q.e3_b2 - 0.02326961
    if 42.31629 <= Q.sd_mass < 78.4753:
        z += -0.004753945 * Q.sd_mass + 0.2011693
    if 78.4753 <= Q.sd_mass < 94.55722:
        z += 0.000670861 * Q.sd_mass - 0.224544
    if 94.55722 <= Q.sd_mass < 127.8042:
        z += 0.01587827 * Q.sd_mass - 1.662514
    if 127.8042 <= Q.sd_mass < 154.5947:
        z += -0.02830407 * Q.sd_mass + 3.984176
    if Q.sd_mass >= 154.5947:
        z += 0.008403229 * Q.sd_mass - 1.690576
    if Q.lund2_lndelta >= -1.411949:
        z += -0.126813 * Q.lund2_lndelta - 0.1790534
    if Q.tau32 >= 0.6800935:
        z += -0.4406081 * Q.tau32 + 0.2996547
    if Q.mass_top20 >= 93.50967:
        z += -0.0009218578 * Q.mass_top20 + 0.08620261
    if Q.max_dr < 0.7634316:
        z += -0.6261491 * Q.max_dr + 0.478022
    if Q.mass_top15 >= 119.5993:
        z += -0.01821104 * Q.mass_top15 + 2.178027
    if Q.psi_0p1 < 0.01545532:
        z += 2.989085 * Q.psi_0p1 - 0.04619725
    if Q.e2 < 0.1240514:
        z += 11.73034 * Q.e2 - 1.455165
    if Q.sum_zz_dr2 < 0.06117886:
        z += -272.3337 * Q.sum_zz_dr2 + 16.66107
    if Q.sum_pt_top3 >= 356.8516:
        z += 0.0003900358 * Q.sum_pt_top3 - 0.1391849
    if Q.z_top30_slots >= 0.9283051:
        z += 0.2014277 * Q.z_top30_slots - 0.1869863
    if Q.sum_z_dr2 < 0.06154656:
        z += 252.3505 * Q.sum_z_dr2 - 15.53131
    if Q.e3 < 0.001589861:
        z += 129.9311 * Q.e3 - 0.2065724
    if Q.M3_b2 < 0.00761379:
        z += 15.2096 * Q.M3_b2 - 0.1158027
    if Q.m01 < 1.868856:
        z += 0.01838958 * Q.m01 - 0.03436748
    if Q.LHA < 0.2686235:
        z += 0.5619143 * Q.LHA - 0.1509434
    z += 0.003982749 * Q.dr_46
    if Q.mass_top40 >= 141.9283:
        z += -0.0007221505 * Q.mass_top40 + 0.1024936
    if Q.n_dr_0p4_up < 19.0:
        z += -0.04563767 * Q.n_dr_0p4_up + 0.8671158
    if Q.pair_max_lnm2 >= 8.097646:
        z += 0.2736648 * Q.pair_max_lnm2 - 2.216041
    if Q.sj4_zsoft < 0.007407379:
        z += -89.48661 * Q.sj4_zsoft + 0.6628613
    if Q.z_dr_0p4_up < 0.004729211:
        z += 7.041488 * Q.z_dr_0p4_up - 0.03330068
    if 60.04967 <= Q.sj2_mass1 < 77.42768:
        z += 0.002455527 * Q.sj2_mass1 - 0.1474536
    if Q.sj2_mass1 >= 77.42768:
        z += 0.001140192 * Q.sj2_mass1 - 0.04561025
    if Q.sum_z_dr2_top30 < 0.05733843:
        z += 8.405212 * Q.sum_z_dr2_top30 - 0.4819417
    if Q.lund_max_lndelta >= -0.4168051:
        z += -0.4566049 * Q.lund_max_lndelta - 0.1903152
    if Q.sj2_zsoft < 0.08996752:
        z += 1.336575 * Q.sj2_zsoft - 0.1202483
    if Q.mass_top30 < 162.7874 and Q.n_dr_0p4_up > 0.0:
        z += 0.0004919512 * (162.7874 - Q.mass_top30) * (Q.n_dr_0p4_up - 0.0)
    if Q.mass_top30 < 162.7874 and Q.lnerel_0 > -1.516067:
        z += -0.0019624 * (162.7874 - Q.mass_top30) * (Q.lnerel_0 - -1.516067)
    if Q.e3_b2 < 0.0001729927 and Q.n_real_top40 < 40.0:
        z += -76.6284 * (0.0001729927 - Q.e3_b2) * (40.0 - Q.n_real_top40)
    if Q.e2_b05 < 0.1894158 and Q.tau21_b2 < 0.5868564:
        z += 4.689711 * (0.1894158 - Q.e2_b05) * (0.5868564 - Q.tau21_b2)
    if Q.tau3 < 0.1024935 and Q.n_dr_0p2_0p4 > 21.0:
        z += -0.540258 * (0.1024935 - Q.tau3) * (Q.n_dr_0p2_0p4 - 21.0)
    if Q.tau3 < 0.1024935 and Q.sj2_dr > 0.5454864:
        z += -8.524223 * (0.1024935 - Q.tau3) * (Q.sj2_dr - 0.5454864)
    if Q.mass_top50 > 94.51361 and Q.mass_top2 > 30.2961:
        z += 6.623687e-05 * (Q.mass_top50 - 94.51361) * (Q.mass_top2 - 30.2961)
    if Q.mass_top20 > 93.50967 and Q.sj3_pairmax_over_m < 0.6627397:
        z += 0.03371088 * (Q.mass_top20 - 93.50967) * (0.6627397 - Q.sj3_pairmax_over_m)
    if Q.mass_top20 > 93.50967 and Q.dr_34 < 0.4587485:
        z += 0.005695574 * (Q.mass_top20 - 93.50967) * (0.4587485 - Q.dr_34)
    if Q.sd_mass > 94.55722 and Q.sd_rg < 0.3402972:
        z += 0.08086345 * (Q.sd_mass - 94.55722) * (0.3402972 - Q.sd_rg)
    if Q.mass_top50 > 94.51361 and Q.sj3_dr13 > 0.2949571:
        z += 0.00220598 * (Q.mass_top50 - 94.51361) * (Q.sj3_dr13 - 0.2949571)
    if Q.mass_top50 > 94.51361 and Q.sj4_pair_mass_min < 25.64898:
        z += -9.921638e-05 * (Q.mass_top50 - 94.51361) * (25.64898 - Q.sj4_pair_mass_min)
    if Q.mass_top50 > 94.51361 and Q.dr12 < 0.2677232:
        z += -0.005184648 * (Q.mass_top50 - 94.51361) * (0.2677232 - Q.dr12)
    if Q.mass_top15 > 119.5993 and Q.z_2nd < 0.09424246:
        z += 0.03334979 * (Q.mass_top15 - 119.5993) * (0.09424246 - Q.z_2nd)
    if Q.lund2_lndelta > -1.411949 and Q.z_dr_0p2_0p4 > 0.4683306:
        z += 0.5200151 * (Q.lund2_lndelta - -1.411949) * (Q.z_dr_0p2_0p4 - 0.4683306)
    if Q.mass_top15 > 119.5993 and Q.e4_b2 > 7.0969e-08:
        z += 12.055 * (Q.mass_top15 - 119.5993) * (Q.e4_b2 - 7.0969e-08)
    if Q.e2_b05 < 0.1894158 and Q.lund2_lndelta < -1.411949:
        z += -1.241348 * (0.1894158 - Q.e2_b05) * (-1.411949 - Q.lund2_lndelta)
    if Q.sd_mass > 127.8042 and Q.D2 < 3.038341:
        z += 0.0009425256 * (Q.sd_mass - 127.8042) * (3.038341 - Q.D2)
    if Q.sd_mass > 42.31629 and Q.lund1_lndelta > -0.4705779:
        z += 0.0006706084 * (Q.sd_mass - 42.31629) * (Q.lund1_lndelta - -0.4705779)
    if Q.sd_mass > 154.5947 and Q.lnptrel_77 > -18.42068:
        z += -0.0003282278 * (Q.sd_mass - 154.5947) * (Q.lnptrel_77 - -18.42068)
    if Q.mass < 164.4374 and Q.sj2_mass1 > 53.51926:
        z += -0.0001642805 * (164.4374 - Q.mass) * (Q.sj2_mass1 - 53.51926)
    if Q.sd_mass > 42.31629 and Q.eta_55 > 0.0:
        z += -0.0007917765 * (Q.sd_mass - 42.31629) * (Q.eta_55 - 0.0)
    if Q.sd_mass > 154.5947 and Q.tau43_b2 < 0.9433644:
        z += -0.007523216 * (Q.sd_mass - 154.5947) * (0.9433644 - Q.tau43_b2)
    if Q.mass_top50 > 94.51361 and Q.tau43_b2 > 0.4866492:
        z += -0.00088496 * (Q.mass_top50 - 94.51361) * (Q.tau43_b2 - 0.4866492)
    if Q.mass < 117.4867 and Q.C3 < 0.02854284:
        z += 0.2430548 * (117.4867 - Q.mass) * (0.02854284 - Q.C3)
    if Q.sd_mass > 154.5947 and Q.N3_b2 < 0.1776883:
        z += 0.01348291 * (Q.sd_mass - 154.5947) * (0.1776883 - Q.N3_b2)
    if Q.mass < 105.7234 and Q.dr_37 < 0.0281821:
        z += 0.1655994 * (105.7234 - Q.mass) * (0.0281821 - Q.dr_37)
    if Q.sd_mass > 42.31629 and Q.sj3_pairmin_over_m > 0.2278414:
        z += -0.003415176 * (Q.sd_mass - 42.31629) * (Q.sj3_pairmin_over_m - 0.2278414)
    if Q.mass_top50 > 94.51361 and Q.sj3_pairmax_over_m < 0.806533:
        z += -0.002269683 * (Q.mass_top50 - 94.51361) * (0.806533 - Q.sj3_pairmax_over_m)
    if Q.tau3 < 0.1024935 and Q.dr_41 > 0.3126638:
        z += -0.06095856 * (0.1024935 - Q.tau3) * (Q.dr_41 - 0.3126638)
    if Q.sd_mass > 127.8042 and Q.C2_b2 < 0.1204509:
        z += -0.1958772 * (Q.sd_mass - 127.8042) * (0.1204509 - Q.C2_b2)
    if Q.mass < 105.7234 and Q.mass_top3 > 29.24995:
        z += 0.000732969 * (105.7234 - Q.mass) * (Q.mass_top3 - 29.24995)
    if Q.sd_mass > 127.8042 and Q.n_lund < 5.0:
        z += 0.002709623 * (Q.sd_mass - 127.8042) * (5.0 - Q.n_lund)
    if Q.mass < 117.4867 and Q.jet_abs_eta > 0.6771968:
        z += -0.00982985 * (117.4867 - Q.mass) * (Q.jet_abs_eta - 0.6771968)
    if Q.mass_top15 > 119.5993 and Q.C2_b2 < 0.221369:
        z += 0.1606026 * (Q.mass_top15 - 119.5993) * (0.221369 - Q.C2_b2)
    if Q.mass_top40 > 141.9283 and Q.dr_12 < 0.09015482:
        z += 0.04863249 * (Q.mass_top40 - 141.9283) * (0.09015482 - Q.dr_12)
    if Q.mass_top30 < 162.7874 and Q.C2_b2 > 0.04016973:
        z += -0.01088003 * (162.7874 - Q.mass_top30) * (Q.C2_b2 - 0.04016973)
    if Q.mass_top30 < 162.7874 and Q.eta_0 > 0.08172607:
        z += -0.01233165 * (162.7874 - Q.mass_top30) * (Q.eta_0 - 0.08172607)
    if Q.sd_mass > 154.5947 and Q.n_lund < 11.0:
        z += 0.0004707065 * (Q.sd_mass - 154.5947) * (11.0 - Q.n_lund)
    if Q.mass_top40 > 141.9283 and Q.n_lund < 10.0:
        z += -0.001195138 * (Q.mass_top40 - 141.9283) * (10.0 - Q.n_lund)
    if Q.mass_top50 > 94.51361 and Q.dr_12 < 0.3533641:
        z += -0.0004601441 * (Q.mass_top50 - 94.51361) * (0.3533641 - Q.dr_12)
    if Q.mass_top50 < 161.1264 and Q.sum_z_dr2_top5 > 0.02100028:
        z += 0.01855916 * (161.1264 - Q.mass_top50) * (Q.sum_z_dr2_top5 - 0.02100028)
    if Q.mass_top15 > 119.5993 and Q.dr_37 < 0.08382604:
        z += -0.0005957588 * (Q.mass_top15 - 119.5993) * (0.08382604 - Q.dr_37)
    if Q.tau3 < 0.1024935 and Q.sj4_zsoft < 0.007407379:
        z += -1372.574 * (0.1024935 - Q.tau3) * (0.007407379 - Q.sj4_zsoft)
    if Q.e3_b2 < 0.0001729927 and Q.n_lund > 11.0:
        z += 43.47092 * (0.0001729927 - Q.e3_b2) * (Q.n_lund - 11.0)
    if Q.e3_b2 < 0.0001729927 and Q.sj3_pairmin_over_m > 0.1383534:
        z += -939.081 * (0.0001729927 - Q.e3_b2) * (Q.sj3_pairmin_over_m - 0.1383534)
    if Q.pair_max_lnm2 > 8.097646 and Q.eta_28 > 0.09893799:
        z += -0.06340946 * (Q.pair_max_lnm2 - 8.097646) * (Q.eta_28 - 0.09893799)
    if Q.mass < 164.4374 and Q.lund_max_lndelta > -1.233804:
        z += 0.002010942 * (164.4374 - Q.mass) * (Q.lund_max_lndelta - -1.233804)
    if Q.z_dr_0p4_up < 0.004729211 and Q.mratio_min_012 < 0.03900679:
        z += 1719.475 * (0.004729211 - Q.z_dr_0p4_up) * (0.03900679 - Q.mratio_min_012)
    return z


def neuron_39(Q):
    z = 9.904104e-06
    return z


def neuron_40(Q):
    z = 5.003766e-06
    return z


def neuron_41(Q):
    z = 8.526475e-06
    return z


def neuron_42(Q):
    z = -4.68051e-06
    return z


def neuron_43(Q):
    z = -3.8981e-06
    return z


def neuron_44(Q):
    z = 2.332824e-05
    return z


def neuron_45(Q):
    z = -3.218037e-05
    return z


def neuron_46(Q):
    z = 9.101478e-06
    return z


def neuron_47(Q):
    z = 4.13496e-05
    return z


def neuron_48(Q):
    z = 8.639793e-07
    return z


def neuron_49(Q):
    z = -1.90266e-05
    return z


def neuron_50(Q):
    z = -4.526367e-05
    return z


def neuron_51(Q):
    z = 0.3771817
    if Q.mass_top15 < 70.93762:
        z += -0.001224056 * Q.mass_top15 + 0.1180334
    if 70.93762 <= Q.mass_top15 < 104.4937:
        z += -0.0009298392 * Q.mass_top15 + 0.09716235
    if Q.tau32 < 0.7544983:
        z += 0.8311422 * Q.tau32 - 0.6270954
    if Q.mass_top40 < 105.3556:
        z += -0.01808647 * Q.mass_top40 + 1.905511
    z += 0.0001251539 * Q.sum_e
    if Q.dr01 < 0.03574519:
        z += -1.967954 * Q.dr01 + 0.07034487
    if Q.tau5 < 0.02938903:
        z += 54.79472 * Q.tau5 - 1.995243
    if 0.02938903 <= Q.tau5 < 0.05045808:
        z += 18.2675 * Q.tau5 - 0.9217431
    if Q.tau5 >= 0.05613495:
        z += 7.915592 * Q.tau5 - 0.4443413
    if 57.33311 <= Q.m012 < 69.67489:
        z += 0.01226815 * Q.m012 - 0.7033714
    if Q.m012 >= 69.67489:
        z += 0.01434112 * Q.m012 - 0.847805
    if Q.e2 < 0.1400425:
        z += -10.97014 * Q.e2 + 1.536287
    if Q.sj2_dr >= 0.5076533:
        z += 0.667045 * Q.sj2_dr - 0.3386276
    if Q.dr02 < 0.02982432:
        z += -7.403408 * Q.dr02 + 0.2208016
    if Q.sj2_mass2 < 8.588303:
        z += 0.04160284 * Q.sj2_mass2 - 0.3572978
    if Q.M2_b05 < 0.1209237:
        z += -0.2738127 * Q.M2_b05 + 0.03311043
    if Q.M2_b05 >= 0.1506455:
        z += 7.46512 * Q.M2_b05 - 1.124587
    z += -11.39127 * Q.M2_b2
    if Q.e4_b05 >= 0.000141779:
        z += -2813.76 * Q.e4_b05 + 0.398932
    if Q.mass_top10 >= 103.4976:
        z += 0.001895132 * Q.mass_top10 - 0.1961417
    if Q.lne_1 >= 5.29456:
        z += 0.2526742 * Q.lne_1 - 1.337799
    if Q.pair_max_lnkt >= 2.43449:
        z += -0.1566077 * Q.pair_max_lnkt + 0.3812598
    if Q.sj3_pair_mass_max < 90.35243:
        z += -0.01017918 * Q.sj3_pair_mass_max + 1.050642
    if 90.35243 <= Q.sj3_pair_mass_max < 128.6605:
        z += -0.003417774 * Q.sj3_pair_mass_max + 0.4397326
    if Q.mass < 100.4835:
        z += -0.008326597 * Q.mass + 0.7770541
    if 100.4835 <= Q.mass < 105.7234:
        z += 0.01138015 * Q.mass - 1.203148
    if 2.458746 <= Q.pair_mean_lnm2 < 4.069088:
        z += -0.0967394 * Q.pair_mean_lnm2 + 0.2378576
    if Q.pair_mean_lnm2 >= 4.069088:
        z += -0.06517412 * Q.pair_mean_lnm2 + 0.1094157
    if Q.dr_max_012 < 0.02981375:
        z += 2.145977 * Q.dr_max_012 - 0.06397964
    if Q.sd_mass < 78.4753:
        z += 0.008879961 * Q.sd_mass - 1.134897
    if 78.4753 <= Q.sd_mass < 106.7501:
        z += -0.01343595 * Q.sd_mass + 0.6163509
    if 106.7501 <= Q.sd_mass < 127.8042:
        z += 0.003985431 * Q.sd_mass - 1.243383
    if Q.sd_mass >= 127.8042:
        z += -0.00489453 * Q.sd_mass - 0.1084859
    if Q.lnpt_6 >= 3.082483:
        z += -0.004121686 * Q.lnpt_6 + 0.01270503
    if Q.pair_max_lnm2 >= 6.840102:
        z += 0.1747777 * Q.pair_max_lnm2 - 1.195497
    if 79.54717 <= Q.mass_top30 < 134.2224:
        z += -0.004218443 * Q.mass_top30 + 0.3355652
    if Q.mass_top30 >= 134.2224:
        z += 0.004189487 * Q.mass_top30 - 0.7929672
    if Q.lund_max_lnkt < 2.961008:
        z += 0.7353294 * Q.lund_max_lnkt - 2.860511
    if 2.961008 <= Q.lund_max_lnkt < 3.9754:
        z += 0.6735015 * Q.lund_max_lnkt - 2.677438
    if Q.sd_rg < 0.2009044:
        z += -3.437018 * Q.sd_rg + 0.6905122
    if Q.n_pairs_kt_above_3 < 16.0:
        z += 0.004335213 * Q.n_pairs_kt_above_3 - 0.03758142
    if 16.0 <= Q.n_pairs_kt_above_3 < 111.0:
        z += -0.0003345472 * Q.n_pairs_kt_above_3 + 0.03713474
    if Q.e3_b2 < 9.621843e-05:
        z += 2805.537 * Q.e3_b2 - 0.2699444
    if Q.n_dr_0p05_0p1 < 3.0:
        z += 0.03988356 * Q.n_dr_0p05_0p1 - 0.1196507
    if Q.sum_z_dr2_top2 < 0.009042742:
        z += 13.41846 * Q.sum_z_dr2_top2 - 0.1213397
    if Q.sj3_pairmax_over_m >= 0.7925455:
        z += 1.191905 * Q.sj3_pairmax_over_m - 0.9446394
    if Q.N2_b05 < 0.4954556:
        z += -0.05780872 * Q.N2_b05 + 0.02864165
    if Q.D2 < 1.953274:
        z += -0.2465711 * Q.D2 + 0.481621
    if Q.z_dr_0p1_0p2 >= 0.5192063:
        z += 0.03772065 * Q.z_dr_0p1_0p2 - 0.0195848
    if Q.lam1 < 0.02605908:
        z += -21.02327 * Q.lam1 + 0.5478471
    if Q.n_pt_above_5 < 35.0:
        z += -0.01189722 * Q.n_pt_above_5 + 0.4164028
    if Q.n_dr_0_0p05 < 3.0:
        z += 0.06081129 * Q.n_dr_0_0p05 - 0.1824339
    if Q.z_dr_0_0p05 < 0.1072997:
        z += -0.7365757 * Q.z_dr_0_0p05 + 0.07903437
    if Q.D3 < 0.2852651:
        z += -0.1337043 * Q.D3 + 0.03814117
    if Q.C3_b2 < 0.01108077:
        z += 4.449572 * Q.C3_b2 - 0.0493047
    if Q.n_lund_kt_above_1 < 5.0:
        z += -0.04651979 * Q.n_lund_kt_above_1 + 0.232599
    if Q.n_dr_0p2_0p4 < 24.0:
        z += -0.003968252 * Q.n_dr_0p2_0p4 + 0.09523804
    if Q.pt_dispersion >= 0.5051136:
        z += -1.300163 * Q.pt_dispersion + 0.65673
    if Q.mass_top15 < 104.4937 and Q.pair_mean_lnm2 > 2.340672:
        z += -0.002260107 * (104.4937 - Q.mass_top15) * (Q.pair_mean_lnm2 - 2.340672)
    if Q.tau5 < 0.05045808 and Q.sd_zg > 0.2652027:
        z += 2.474171 * (0.05045808 - Q.tau5) * (Q.sd_zg - 0.2652027)
    if Q.sj2_dr > 0.5076533 and Q.D2 < 1.325034:
        z += -0.2445365 * (Q.sj2_dr - 0.5076533) * (1.325034 - Q.D2)
    if Q.dr01 < 0.03574519 and Q.D2_b2 < 15.17086:
        z += 0.212964 * (0.03574519 - Q.dr01) * (15.17086 - Q.D2_b2)
    if Q.M2_b05 > 0.1506455 and Q.sj3_mass3 < 0.3886647:
        z += 8.479984 * (Q.M2_b05 - 0.1506455) * (0.3886647 - Q.sj3_mass3)
    if Q.e2 < 0.1400425 and Q.mean_eta < -0.002507949:
        z += -118.0082 * (0.1400425 - Q.e2) * (-0.002507949 - Q.mean_eta)
    if Q.mass_top15 < 104.4937 and Q.M3_b2 > 0.00678572:
        z += -0.2534975 * (104.4937 - Q.mass_top15) * (Q.M3_b2 - 0.00678572)
    if Q.tau32 < 0.7544983 and Q.sj3_mass3 < 3.748488:
        z += -0.1649195 * (0.7544983 - Q.tau32) * (3.748488 - Q.sj3_mass3)
    if Q.pair_max_lnkt > 2.43449 and Q.sj3_z1 > 0.5904161:
        z += -1.098234 * (Q.pair_max_lnkt - 2.43449) * (Q.sj3_z1 - 0.5904161)
    if Q.mass_top10 > 103.4976 and Q.min_pair_mass < 6.383481:
        z += 0.0003304627 * (Q.mass_top10 - 103.4976) * (6.383481 - Q.min_pair_mass)
    if Q.mass_top10 > 103.4976 and Q.n_lund_kt_above_5 < 4.0:
        z += 0.003573601 * (Q.mass_top10 - 103.4976) * (4.0 - Q.n_lund_kt_above_5)
    if Q.pair_max_lnkt > 2.43449 and Q.n_dr_0p1_0p2 < 17.0:
        z += -0.01151246 * (Q.pair_max_lnkt - 2.43449) * (17.0 - Q.n_dr_0p1_0p2)
    if Q.mass_top40 < 105.3556 and Q.pt1_over_pt0 < 0.4358177:
        z += -0.04308959 * (105.3556 - Q.mass_top40) * (0.4358177 - Q.pt1_over_pt0)
    if Q.m012 > 69.67489 and Q.lnpt_29 > 0.2696991:
        z += -0.01120147 * (Q.m012 - 69.67489) * (Q.lnpt_29 - 0.2696991)
    if Q.sj3_pair_mass_max < 128.6605 and Q.lnpt_29 < 1.618769:
        z += -3.726481e-05 * (128.6605 - Q.sj3_pair_mass_max) * (1.618769 - Q.lnpt_29)
    if Q.tau32 < 0.7544983 and Q.sj3_dr_min < 0.1259759:
        z += 23.66292 * (0.7544983 - Q.tau32) * (0.1259759 - Q.sj3_dr_min)
    if Q.tau5 < 0.05045808 and Q.sj3_dr_min > 0.1611011:
        z += 17.84953 * (0.05045808 - Q.tau5) * (Q.sj3_dr_min - 0.1611011)
    if Q.sd_mass > 106.7501 and Q.D2_b2 < 15.17086:
        z += 0.0006625552 * (Q.sd_mass - 106.7501) * (15.17086 - Q.D2_b2)
    if Q.mass_top15 < 104.4937 and Q.C2_b2 > 0.08076625:
        z += 0.003569786 * (104.4937 - Q.mass_top15) * (Q.C2_b2 - 0.08076625)
    if Q.M2_b05 > 0.1506455 and Q.pt2_over_pt0 < 0.4580564:
        z += -15.01553 * (Q.M2_b05 - 0.1506455) * (0.4580564 - Q.pt2_over_pt0)
    if Q.mass_top30 > 79.54717 and Q.lund3_lnz < -1.763117:
        z += 0.0002458018 * (Q.mass_top30 - 79.54717) * (-1.763117 - Q.lund3_lnz)
    if Q.pair_max_lnm2 > 6.840102 and Q.dr_31 < 0.1762573:
        z += 1.160449 * (Q.pair_max_lnm2 - 6.840102) * (0.1762573 - Q.dr_31)
    if Q.dr_max_012 < 0.02981375 and Q.eta_12 < 0.03826904:
        z += 7.355903 * (0.02981375 - Q.dr_max_012) * (0.03826904 - Q.eta_12)
    if Q.pair_max_lnm2 > 6.840102 and Q.eta_25 < 0.1497803:
        z += 0.02642951 * (Q.pair_max_lnm2 - 6.840102) * (0.1497803 - Q.eta_25)
    if Q.M2_b05 > 0.1506455 and Q.tau43_b2 < 0.4240535:
        z += 26.98458 * (Q.M2_b05 - 0.1506455) * (0.4240535 - Q.tau43_b2)
    if Q.dr01 < 0.03574519 and Q.eta_2 < -0.07635498:
        z += -4.243967 * (0.03574519 - Q.dr01) * (-0.07635498 - Q.eta_2)
    if Q.pair_max_lnm2 > 6.840102 and Q.dr_8 > 0.1767313:
        z += -0.1537987 * (Q.pair_max_lnm2 - 6.840102) * (Q.dr_8 - 0.1767313)
    if Q.lund_max_lnkt < 3.9754 and Q.eta_0 < -0.0848999:
        z += -4.541601 * (3.9754 - Q.lund_max_lnkt) * (-0.0848999 - Q.eta_0)
    if Q.sj3_pair_mass_max < 90.35243 and Q.dr01 < 0.191004:
        z += 0.02092287 * (90.35243 - Q.sj3_pair_mass_max) * (0.191004 - Q.dr01)
    if Q.dr01 < 0.03574519 and Q.sj4_dr_min < 0.1458275:
        z += -27.54393 * (0.03574519 - Q.dr01) * (0.1458275 - Q.sj4_dr_min)
    if Q.mass_top30 > 79.54717 and Q.sj3_dr13 < 0.7462286:
        z += -0.007491129 * (Q.mass_top30 - 79.54717) * (0.7462286 - Q.sj3_dr13)
    if Q.sj2_mass2 < 8.588303 and Q.dr01 > 0.2849189:
        z += 0.1706611 * (8.588303 - Q.sj2_mass2) * (Q.dr01 - 0.2849189)
    if Q.N2_b05 < 0.4954556 and Q.n_pt_above_10 > 6.0:
        z += 0.01148201 * (0.4954556 - Q.N2_b05) * (Q.n_pt_above_10 - 6.0)
    if Q.mass_top30 > 79.54717 and Q.D3 > 0.04661539:
        z += 0.0005405663 * (Q.mass_top30 - 79.54717) * (Q.D3 - 0.04661539)
    if Q.mass_top30 > 79.54717 and Q.lund1_lndelta < -0.831543:
        z += -0.01194186 * (Q.mass_top30 - 79.54717) * (-0.831543 - Q.lund1_lndelta)
    if Q.lund_max_lnkt < 3.9754 and Q.n_pairs_kt_above_30 < 1.0:
        z += 0.3703387 * (3.9754 - Q.lund_max_lnkt) * (1.0 - Q.n_pairs_kt_above_30)
    if Q.D2 < 1.953274 and Q.dr01 < 0.3258728:
        z += -0.5339291 * (1.953274 - Q.D2) * (0.3258728 - Q.dr01)
    if Q.N2_b05 < 0.4954556 and Q.eta_41 > -0.1331787:
        z += -0.09275607 * (0.4954556 - Q.N2_b05) * (Q.eta_41 - -0.1331787)
    if Q.e3_b2 < 9.621843e-05 and Q.lund3_lndelta > -2.817283:
        z += 851.3491 * (9.621843e-05 - Q.e3_b2) * (Q.lund3_lndelta - -2.817283)
    if Q.pair_mean_lnm2 > 2.458746 and Q.eta_73 > 0.0:
        z += -0.4374679 * (Q.pair_mean_lnm2 - 2.458746) * (Q.eta_73 - 0.0)
    if Q.sd_mass > 106.7501 and Q.phi_72 < 0.0:
        z += -0.001520385 * (Q.sd_mass - 106.7501) * (0.0 - Q.phi_72)
    if Q.lnpt_6 > 3.082483 and Q.eta_14 < -0.05612183:
        z += 0.2157121 * (Q.lnpt_6 - 3.082483) * (-0.05612183 - Q.eta_14)
    return z


def neuron_52(Q):
    z = 2.688201e-07
    return z


def neuron_53(Q):
    z = -8.044334e-06
    return z


def neuron_54(Q):
    z = -0.1019389
    if Q.n_pairs_kt_above_1 < 195.0:
        z += 0.008022977 * Q.n_pairs_kt_above_1 - 2.975591
    if 195.0 <= Q.n_pairs_kt_above_1 < 366.0:
        z += 0.003928252 * Q.n_pairs_kt_above_1 - 2.177119
    if 366.0 <= Q.n_pairs_kt_above_1 < 873.0:
        z += 0.001458342 * Q.n_pairs_kt_above_1 - 1.273132
    if Q.M2_b2 < 0.02398667:
        z += -15.04834 * Q.M2_b2 + 0.2845952
    if 0.02398667 <= Q.M2_b2 < 0.03513777:
        z += 6.848133 * Q.M2_b2 - 0.2406281
    if Q.mass_top40 >= 126.8853:
        z += -0.01383456 * Q.mass_top40 + 1.755402
    if Q.sd_mass < 42.31629:
        z += 0.01327334 * Q.sd_mass - 0.1709504
    if 42.31629 <= Q.sd_mass < 83.61981:
        z += -0.02084287 * Q.sd_mass + 1.272721
    if 83.61981 <= Q.sd_mass < 115.7091:
        z += 0.0303119 * Q.sd_mass - 3.004831
    if 115.7091 <= Q.sd_mass < 123.2919:
        z += 0.03422187 * Q.sd_mass - 3.457251
    if 123.2919 <= Q.sd_mass < 127.8042:
        z += -0.006891573 * Q.sd_mass + 1.611703
    if 127.8042 <= Q.sd_mass < 175.9333:
        z += -0.02975766 * Q.sd_mass + 4.534086
    if Q.sd_mass >= 175.9333:
        z += -0.001917555 * Q.sd_mass - 0.3639171
    if Q.pair_mean_lnm2 < 4.787589:
        z += -0.254059 * Q.pair_mean_lnm2 + 1.21633
    if Q.n_pairs_kt_above_3 < 34.0:
        z += 0.009280925 * Q.n_pairs_kt_above_3 - 0.9582091
    if 34.0 <= Q.n_pairs_kt_above_3 < 111.0:
        z += 0.008346204 * Q.n_pairs_kt_above_3 - 0.9264286
    if Q.tau2 < 0.1076451:
        z += -0.09814822 * Q.tau2 + 0.01056517
    if 79.47361 <= Q.mass < 117.4867:
        z += 0.05203637 * Q.mass - 4.135518
    if 117.4867 <= Q.mass < 164.4374:
        z += -0.01313811 * Q.mass + 3.521615
    if Q.mass >= 164.4374:
        z += 0.005778698 * Q.mass + 0.4109835
    if 3.294888 <= Q.m01 < 4.577187:
        z += -0.1230624 * Q.m01 + 0.4054767
    if Q.m01 >= 4.577187:
        z += -0.01313487 * Q.m01 - 0.09768204
    if Q.pt_entropy >= 2.282975:
        z += 0.4696915 * Q.pt_entropy - 1.072294
    if Q.n_dr_0p2_0p4 < 19.0:
        z += 0.05799749 * Q.n_dr_0p2_0p4 - 1.101952
    if Q.lne_16 >= 1.463328:
        z += 0.03999879 * Q.lne_16 - 0.05853134
    if Q.e3_b2 < 0.0001729927:
        z += -2995.41 * Q.e3_b2 + 0.518184
    if Q.sj3_mass3 < 0.3886647:
        z += 1.863375 * Q.sj3_mass3 - 0.7242279
    if Q.lund_max_lnkt < 3.734077:
        z += -0.4337159 * Q.lund_max_lnkt + 1.619529
    if Q.z_dr_0p2_0p4 < 0.1724455:
        z += -2.727719 * Q.z_dr_0p2_0p4 + 0.4703828
    if 59.49644 <= Q.sj3_pair_mass_min < 80.02563:
        z += -0.005152842 * Q.sj3_pair_mass_min + 0.3065757
    if Q.sj3_pair_mass_min >= 80.02563:
        z += 0.01891532 * Q.sj3_pair_mass_min - 1.619494
    if Q.pair_mean_lndelta < -2.115543:
        z += 0.6418108 * Q.pair_mean_lndelta + 1.357778
    if Q.min_pair_mass < 1.514826:
        z += 0.1333782 * Q.min_pair_mass - 0.2789387
    if 1.514826 <= Q.min_pair_mass < 2.102317:
        z += 0.1308854 * Q.min_pair_mass - 0.2751626
    if Q.dr02 < 0.05551408:
        z += -1.351074 * Q.dr02 + 0.07500363
    if Q.dr02 >= 0.1620436:
        z += -0.03962623 * Q.dr02 + 0.006421176
    if Q.sj2_dr < 0.2083105:
        z += -6.409076 * Q.sj2_dr + 1.252645
    if 0.2083105 <= Q.sj2_dr < 0.3740528:
        z += -1.014863 * Q.sj2_dr + 0.1289736
    if 0.3740528 <= Q.sj2_dr < 0.6647889:
        z += 0.862083 * Q.sj2_dr - 0.5731032
    if Q.tau32 < 0.5498426:
        z += 2.612784 * Q.tau32 - 1.43662
    if Q.n_lund_kt_above_1 >= 5.0:
        z += -0.06940782 * Q.n_lund_kt_above_1 + 0.3470391
    if Q.C3_b05 >= 0.1094329:
        z += 1.991425 * Q.C3_b05 - 0.2179274
    if Q.max_pair_mass < 1.408071:
        z += 0.2558207 * Q.max_pair_mass - 0.3602138
    if Q.sj3_mass2 < 4.326415:
        z += 0.1212408 * Q.sj3_mass2 - 0.5245381
    if Q.jet_abs_eta >= 0.3274899:
        z += -0.2752409 * Q.jet_abs_eta + 0.0901386
    if Q.sj4_pair_mass_min < 19.01187:
        z += -0.006463838 * Q.sj4_pair_mass_min + 0.1228896
    if Q.z_top10_slots >= 0.7362734:
        z += 2.745502 * Q.z_top10_slots - 2.02144
    if Q.n_particles < 40.0:
        z += 0.03245366 * Q.n_particles - 1.298146
    if Q.M3_b05 < 0.07625067:
        z += 10.03717 * Q.M3_b05 - 0.7653407
    if Q.mass_top5 >= 54.36876:
        z += 0.006154231 * Q.mass_top5 - 0.3345979
    if Q.mass_over_sum_pt_sq < 0.05500758:
        z += -13.43227 * Q.mass_over_sum_pt_sq + 0.7388767
    if Q.e2_b2 < 0.052589:
        z += 13.03085 * Q.e2_b2 - 0.6852795
    if Q.sj4_pair_mass_max >= 128.0079:
        z += -0.00226475 * Q.sj4_pair_mass_max + 0.2899058
    if Q.mass_top15 >= 133.1648:
        z += -0.004267853 * Q.mass_top15 + 0.5683278
    if Q.sum_z_dr2_top15 < 0.007507913:
        z += -41.24245 * Q.sum_z_dr2_top15 + 0.3096448
    if Q.pair_mean_lnkt < 0.2542227:
        z += -0.4315163 * Q.pair_mean_lnkt + 0.1097012
    if Q.pair_mean_lnkt >= 0.7891509:
        z += 0.3051317 * Q.pair_mean_lnkt - 0.2407949
    if Q.D2_b05 < 1.102602:
        z += 1.48584 * Q.D2_b05 - 1.649611
    if 1.102602 <= Q.D2_b05 < 1.281478:
        z += 0.06328747 * Q.D2_b05 - 0.08110149
    if Q.pt_balance01 < 0.1985369:
        z += 1.519135 * Q.pt_balance01 - 0.3016045
    if Q.N3_b05 < 1.166286:
        z += 1.012214 * Q.N3_b05 - 1.180532
    if Q.N3_b2 < 0.4545826:
        z += -0.3688983 * Q.N3_b2 + 0.1676947
    if Q.dr_23 < 0.1464694:
        z += 0.4200705 * Q.dr_23 - 0.06152748
    if 90.35243 <= Q.sj3_pair_mass_max < 128.6605:
        z += 0.01630542 * Q.sj3_pair_mass_max - 1.473235
    if Q.sj3_pair_mass_max >= 128.6605:
        z += -0.0008372497 * Q.sj3_pair_mass_max + 0.7323509
    if Q.sj3_pairmax_over_m < 0.7366642:
        z += -1.765538 * Q.sj3_pairmax_over_m + 1.300608
    if Q.dr01 >= 0.1335998:
        z += 1.333196 * Q.dr01 - 0.1781146
    if Q.sum_z_dr2_top5 < 0.07109528:
        z += -11.88209 * Q.sum_z_dr2_top5 + 0.8447608
    if Q.dr_min_012 < 0.05063855:
        z += -0.9432944 * Q.dr_min_012 + 0.04776706
    if Q.lne_1 >= 5.29456:
        z += 0.03841117 * Q.lne_1 - 0.2033702
    if Q.lund2_lndelta >= -0.737181:
        z += -0.1666841 * Q.lund2_lndelta - 0.1228763
    if Q.sj2_mass1 < 39.95503:
        z += 0.002619703 * Q.sj2_mass1 + 0.09979953
    if 39.95503 <= Q.sj2_mass1 < 77.42768:
        z += -0.005456508 * Q.sj2_mass1 + 0.4224848
    if 0.1991803 <= Q.sj3_dr_min < 0.3526989:
        z += -0.2412854 * Q.sj3_dr_min + 0.04805929
    if 0.3526989 <= Q.sj3_dr_min < 0.462488:
        z += 1.600511 * Q.sj3_dr_min - 0.6015401
    if Q.sj3_dr_min >= 0.462488:
        z += 0.9333637 * Q.sj3_dr_min - 0.2929928
    if Q.M2_b05 < 0.1573433:
        z += -2.667594 * Q.M2_b05 + 0.419728
    if Q.M3 >= 0.04863496:
        z += 4.685095 * Q.M3 - 0.2278594
    z += -3.035408 * Q.sum_z_dr2_top2
    if Q.n_pairs_kt_above_3 < 111.0 and Q.dr02 > 0.017759:
        z += -0.00422792 * (111.0 - Q.n_pairs_kt_above_3) * (Q.dr02 - 0.017759)
    if Q.tau2 < 0.1076451 and Q.N3_b05 < 0.7045747:
        z += 5.236191 * (0.1076451 - Q.tau2) * (0.7045747 - Q.N3_b05)
    if Q.mass > 79.47361 and Q.n_pairs_kt_above_10 < 17.0:
        z += -0.000251373 * (Q.mass - 79.47361) * (17.0 - Q.n_pairs_kt_above_10)
    if Q.sj3_mass3 < 0.3886647 and Q.sj3_z3 < 0.06149175:
        z += 31.21772 * (0.3886647 - Q.sj3_mass3) * (0.06149175 - Q.sj3_z3)
    if Q.m01 > 4.577187 and Q.dr_1 < 0.1975394:
        z += -0.02053255 * (Q.m01 - 4.577187) * (0.1975394 - Q.dr_1)
    if Q.sd_mass > 42.31629 and Q.pt1_over_pt0 < 0.5339053:
        z += -0.002814758 * (Q.sd_mass - 42.31629) * (0.5339053 - Q.pt1_over_pt0)
    if Q.sd_mass < 175.9333 and Q.min_pair_mass > 10.59762:
        z += -0.0001853939 * (175.9333 - Q.sd_mass) * (Q.min_pair_mass - 10.59762)
    if Q.n_dr_0p2_0p4 < 19.0 and Q.M3_b2 < 0.01724773:
        z += -0.1280805 * (19.0 - Q.n_dr_0p2_0p4) * (0.01724773 - Q.M3_b2)
    if Q.tau32 < 0.5498426 and Q.n_lund < 14.0:
        z += 0.2292417 * (0.5498426 - Q.tau32) * (14.0 - Q.n_lund)
    if Q.mass_top40 > 126.8853 and Q.M3_b2 < 0.009257989:
        z += 0.1432034 * (Q.mass_top40 - 126.8853) * (0.009257989 - Q.M3_b2)
    if Q.lne_16 > 1.463328 and Q.tau43_b2 < 0.8531024:
        z += 0.2354184 * (Q.lne_16 - 1.463328) * (0.8531024 - Q.tau43_b2)
    if Q.lne_16 > 1.463328 and Q.n_dr_0p1_0p2 < 9.0:
        z += -0.02843906 * (Q.lne_16 - 1.463328) * (9.0 - Q.n_dr_0p1_0p2)
    if Q.sj2_dr < 0.6647889 and Q.M3_b2 > 0.004141442:
        z += -24.14952 * (0.6647889 - Q.sj2_dr) * (Q.M3_b2 - 0.004141442)
    if Q.sd_mass < 175.9333 and Q.sj4_zsoft < 0.03350185:
        z += -0.04660973 * (175.9333 - Q.sd_mass) * (0.03350185 - Q.sj4_zsoft)
    if Q.m01 > 3.294888 and Q.n_real_top20 < 20.0:
        z += 0.00113559 * (Q.m01 - 3.294888) * (20.0 - Q.n_real_top20)
    if Q.n_pairs_kt_above_1 < 366.0 and Q.phi_18 > 0.04797363:
        z += 0.0007875336 * (366.0 - Q.n_pairs_kt_above_1) * (Q.phi_18 - 0.04797363)
    if Q.n_pairs_kt_above_1 < 366.0 and Q.sj3_mass2 < 6.006437:
        z += 0.000148204 * (366.0 - Q.n_pairs_kt_above_1) * (6.006437 - Q.sj3_mass2)
    if Q.mass > 117.4867 and Q.mratio_max_012 > 0.7696297:
        z += 0.01307859 * (Q.mass - 117.4867) * (Q.mratio_max_012 - 0.7696297)
    if Q.sd_mass < 123.2919 and Q.dr_10 > 0.5063302:
        z += 0.007070553 * (123.2919 - Q.sd_mass) * (Q.dr_10 - 0.5063302)
    if Q.C3_b05 > 0.1094329 and Q.eta_40 < 0.07720947:
        z += -4.656413 * (Q.C3_b05 - 0.1094329) * (0.07720947 - Q.eta_40)
    if Q.sj3_mass3 < 0.3886647 and Q.lund2_lnz > -5.443387:
        z += -0.1713248 * (0.3886647 - Q.sj3_mass3) * (Q.lund2_lnz - -5.443387)
    if Q.sj3_mass3 < 0.3886647 and Q.lund2_lnkt > 3.917562:
        z += 2.030876 * (0.3886647 - Q.sj3_mass3) * (Q.lund2_lnkt - 3.917562)
    return z


def neuron_55(Q):
    z = 6.958066e-06
    return z


def neuron_56(Q):
    z = 1.805158e-06
    return z


def neuron_57(Q):
    z = -0.001688453
    return z


def neuron_58(Q):
    z = 2.775588e-05
    return z


def neuron_59(Q):
    z = -6.876595e-06
    return z


def neuron_60(Q):
    z = 8.460409e-06
    return z


def neuron_61(Q):
    z = 4.691285e-06
    return z


def neuron_62(Q):
    z = -6.222401e-06
    return z


def neuron_63(Q):
    z = 2.730003e-05
    return z


def neuron_64(Q):
    z = 1.583109e-05
    return z


def neuron_65(Q):
    z = 1.799774e-05
    return z


def neuron_66(Q):
    z = -1.819647e-05
    return z


def neuron_67(Q):
    z = 6.789744e-06
    return z


def neuron_68(Q):
    z = 5.798125e-07
    return z


def neuron_69(Q):
    z = -0.07747842
    z += 0.09705882 * Q.pair_mean_lndelta
    if 56.49019 <= Q.mass < 79.47361:
        z += -0.004602365 * Q.mass + 0.2599885
    if 79.47361 <= Q.mass < 95.14961:
        z += -0.006390191 * Q.mass + 0.4020735
    if 95.14961 <= Q.mass < 114.0172:
        z += 0.002940529 * Q.mass - 0.4857408
    if 114.0172 <= Q.mass < 131.3917:
        z += 0.000983695 * Q.mass - 0.2626281
    if 131.3917 <= Q.mass < 182.8592:
        z += 3.676477e-05 * Q.mass - 0.1382093
    if Q.mass >= 182.8592:
        z += 0.003209026 * Q.mass - 0.7182866
    if Q.n_pairs_kt_above_1 < 58.0:
        z += -0.002958516 * Q.n_pairs_kt_above_1 + 0.1715939
    if Q.lnerel_0 >= -1.362996:
        z += -0.3617984 * Q.lnerel_0 - 0.4931298
    if Q.lund_max_lnkt < 3.634108:
        z += -0.1508085 * Q.lund_max_lnkt + 0.5480543
    if Q.n_lund < 8.0:
        z += -0.06214369 * Q.n_lund + 0.4971496
    if Q.tau21 < 0.2077175:
        z += 0.9589424 * Q.tau21 - 0.1991891
    if Q.tau3 < 0.0366743:
        z += -1.083998 * Q.tau3 + 0.03975485
    if Q.sj3_pair_mass_max < 73.24742:
        z += 0.004016216 * Q.sj3_pair_mass_max - 0.1949064
    if 73.24742 <= Q.sj3_pair_mass_max < 76.91486:
        z += 0.008886913 * Q.sj3_pair_mass_max - 0.5516724
    if 76.91486 <= Q.sj3_pair_mass_max < 97.52633:
        z += -0.006397567 * Q.sj3_pair_mass_max + 0.6239312
    if Q.sj3_pair_mass_max >= 128.6605:
        z += -0.01358175 * Q.sj3_pair_mass_max + 1.747436
    if Q.mass_top50 >= 56.41079:
        z += -0.0001791661 * Q.mass_top50 + 0.0101069
    if Q.sj3_pairmin_over_m >= 0.2878167:
        z += 0.4190333 * Q.sj3_pairmin_over_m - 0.1206048
    if Q.tau2 < 0.1193588:
        z += -0.7511919 * Q.tau2 + 0.08966138
    if Q.m01 >= 17.3274:
        z += -6.171904e-05 * Q.m01 + 0.001069431
    if Q.n_pairs_kt_above_3 < 34.0:
        z += -0.0008789414 * Q.n_pairs_kt_above_3 + 0.0606884
    if 34.0 <= Q.n_pairs_kt_above_3 < 88.0:
        z += -0.0005704518 * Q.n_pairs_kt_above_3 + 0.05019976
    if Q.C2 >= 0.2035961:
        z += 0.3598875 * Q.C2 - 0.07327169
    if Q.sd_mass < 42.31629:
        z += 0.0007832528 * Q.sd_mass + 0.3754381
    if 42.31629 <= Q.sd_mass < 78.4753:
        z += 0.004952581 * Q.sd_mass + 0.1990075
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += -0.02150437 * Q.sd_mass + 2.275225
    if 88.81751 <= Q.sd_mass < 94.55722:
        z += -0.01226632 * Q.sd_mass + 1.454724
    if 94.55722 <= Q.sd_mass < 106.7501:
        z += 0.005442691 * Q.sd_mass - 0.2197908
    if 106.7501 <= Q.sd_mass < 115.7091:
        z += 0.009380064 * Q.sd_mass - 0.6401057
    if 115.7091 <= Q.sd_mass < 175.9333:
        z += 0.001857066 * Q.sd_mass + 0.2303736
    if Q.sd_mass >= 175.9333:
        z += 0.004169329 * Q.sd_mass - 0.1764305
    if 92.68149 <= Q.mass_top40 < 101.351:
        z += -0.007332199 * Q.mass_top40 + 0.6795592
    if 101.351 <= Q.mass_top40 < 122.7145:
        z += -0.007573226 * Q.mass_top40 + 0.7039875
    if Q.mass_top40 >= 122.7145:
        z += 0.002032631 * Q.mass_top40 - 0.4747907
    if Q.sj2_mass2 < 1.851735:
        z += 0.05654093 * Q.sj2_mass2 - 0.1046988
    if Q.sj2_mass1 < 33.62116:
        z += 0.007243162 * Q.sj2_mass1 - 0.2435235
    z += -0.002227016 * Q.pt1_dr01
    if Q.n_dr_0_0p05 >= 10.0:
        z += -0.003866368 * Q.n_dr_0_0p05 + 0.03866368
    if Q.mass_top30 < 83.81888:
        z += 0.0031517 * Q.mass_top30 - 0.4022062
    if 83.81888 <= Q.mass_top30 < 95.20959:
        z += 0.01211814 * Q.mass_top30 - 1.153763
    if Q.mass_top30 >= 146.3096:
        z += -0.002937018 * Q.mass_top30 + 0.4297138
    if Q.tau1 < 0.06074238:
        z += 5.605616 * Q.tau1 - 0.3404984
    if Q.dr_9 < 0.1648004:
        z += -0.1597071 * Q.dr_9 + 0.02631979
    if Q.mass > 95.14961 and Q.lnptrel_0 > -1.266303:
        z += -0.006307256 * (Q.mass - 95.14961) * (Q.lnptrel_0 - -1.266303)
    if Q.mass > 79.47361 and Q.M2_b2 < 0.04944303:
        z += 0.002366612 * (Q.mass - 79.47361) * (0.04944303 - Q.M2_b2)
    if Q.lnerel_0 > -1.362996 and Q.dr_0 > 0.1756108:
        z += 4.350904 * (Q.lnerel_0 - -1.362996) * (Q.dr_0 - 0.1756108)
    if Q.mass > 95.14961 and Q.C2_b2 < 0.1049877:
        z += 0.1703382 * (Q.mass - 95.14961) * (0.1049877 - Q.C2_b2)
    if Q.mass_top50 > 56.41079 and Q.sj3_mass3 < 0.3886647:
        z += 0.01141895 * (Q.mass_top50 - 56.41079) * (0.3886647 - Q.sj3_mass3)
    if Q.mass > 95.14961 and Q.mratio_max_012 > 0.8680029:
        z += -0.005410162 * (Q.mass - 95.14961) * (Q.mratio_max_012 - 0.8680029)
    if Q.lnerel_0 > -1.362996 and Q.z_2nd < 0.2273903:
        z += 0.9604841 * (Q.lnerel_0 - -1.362996) * (0.2273903 - Q.z_2nd)
    if Q.tau21 < 0.2077175 and Q.sj3_dr_max < 0.5381046:
        z += 6.806292 * (0.2077175 - Q.tau21) * (0.5381046 - Q.sj3_dr_max)
    if Q.sj3_pairmin_over_m > 0.2878167 and Q.dr_32 < 0.08614914:
        z += 3.782408 * (Q.sj3_pairmin_over_m - 0.2878167) * (0.08614914 - Q.dr_32)
    if Q.mass > 114.0172 and Q.C2_b2 < 0.1204509:
        z += -0.1303125 * (Q.mass - 114.0172) * (0.1204509 - Q.C2_b2)
    if Q.tau3 < 0.0366743 and Q.mratio_min_012 > 0.1991254:
        z += 1.553151 * (0.0366743 - Q.tau3) * (Q.mratio_min_012 - 0.1991254)
    if Q.m01 > 17.3274 and Q.dr_17 < 0.002489417:
        z += -1.406745 * (Q.m01 - 17.3274) * (0.002489417 - Q.dr_17)
    if Q.tau2 < 0.1193588 and Q.pair_max_lnkt < 3.143912:
        z += -0.01134581 * (0.1193588 - Q.tau2) * (3.143912 - Q.pair_max_lnkt)
    if Q.tau21 < 0.2077175 and Q.n_dr_0p05_0p1 > 16.0:
        z += 0.3495209 * (0.2077175 - Q.tau21) * (Q.n_dr_0p05_0p1 - 16.0)
    if Q.m01 > 17.3274 and Q.dr_16 < 0.1869778:
        z += -0.01115166 * (Q.m01 - 17.3274) * (0.1869778 - Q.dr_16)
    if Q.m01 > 17.3274 and Q.eta_1 < 0.09069824:
        z += 0.002606534 * (Q.m01 - 17.3274) * (0.09069824 - Q.eta_1)
    if Q.n_pairs_kt_above_1 < 58.0 and Q.n_dr_0p05_0p1 > 16.0:
        z += -0.001617123 * (58.0 - Q.n_pairs_kt_above_1) * (Q.n_dr_0p05_0p1 - 16.0)
    if Q.m01 > 17.3274 and Q.dr_55 > 0.1893821:
        z += -0.0004165005 * (Q.m01 - 17.3274) * (Q.dr_55 - 0.1893821)
    if Q.sj2_mass1 < 33.62116 and Q.n_dr_0p05_0p1 < 2.0:
        z += 0.001244611 * (33.62116 - Q.sj2_mass1) * (2.0 - Q.n_dr_0p05_0p1)
    if Q.sd_mass < 78.4753 and Q.M3 > 0.04863496:
        z += -0.04206423 * (78.4753 - Q.sd_mass) * (Q.M3 - 0.04863496)
    if Q.lnerel_0 > -1.362996 and Q.jet_abs_eta > 1.098425:
        z += -0.03461977 * (Q.lnerel_0 - -1.362996) * (Q.jet_abs_eta - 1.098425)
    if Q.sj2_mass2 < 1.851735 and Q.n_pt_above_10 < 21.0:
        z += -0.005764548 * (1.851735 - Q.sj2_mass2) * (21.0 - Q.n_pt_above_10)
    if Q.n_lund < 8.0 and Q.pt1_over_pt0 > 0.380649:
        z += -0.06722516 * (8.0 - Q.n_lund) * (Q.pt1_over_pt0 - 0.380649)
    if Q.mass > 114.0172 and Q.pt_balance01 < 0.1361601:
        z += 0.08920915 * (Q.mass - 114.0172) * (0.1361601 - Q.pt_balance01)
    if Q.mass > 182.8592 and Q.pt1_over_pt0 < 0.4868628:
        z += 0.0041726 * (Q.mass - 182.8592) * (0.4868628 - Q.pt1_over_pt0)
    if Q.n_pairs_kt_above_3 < 88.0 and Q.dr_40 < 0.1605483:
        z += 0.0006735709 * (88.0 - Q.n_pairs_kt_above_3) * (0.1605483 - Q.dr_40)
    if Q.sd_mass < 175.9333 and Q.n_pt_above_5 > 14.0:
        z += -4.947868e-05 * (175.9333 - Q.sd_mass) * (Q.n_pt_above_5 - 14.0)
    if Q.mass > 114.0172 and Q.sj3_mass2 < 1.824785:
        z += 0.006274664 * (Q.mass - 114.0172) * (1.824785 - Q.sj3_mass2)
    if Q.mass_top40 > 101.351 and Q.dr12 < 0.3141459:
        z += 0.006516197 * (Q.mass_top40 - 101.351) * (0.3141459 - Q.dr12)
    if Q.sd_mass < 175.9333 and Q.min_pair_mass > 10.59762:
        z += -0.0001557212 * (175.9333 - Q.sd_mass) * (Q.min_pair_mass - 10.59762)
    if Q.n_lund < 8.0 and Q.e3_b2 < 0.000125186:
        z += -297.8477 * (8.0 - Q.n_lund) * (0.000125186 - Q.e3_b2)
    if Q.sd_mass < 94.55722 and Q.C3_b2 < 0.00278761:
        z += -0.1171259 * (94.55722 - Q.sd_mass) * (0.00278761 - Q.C3_b2)
    if Q.m01 > 17.3274 and Q.lund1_lndelta < -0.3948254:
        z += 0.002625358 * (Q.m01 - 17.3274) * (-0.3948254 - Q.lund1_lndelta)
    if Q.n_pairs_kt_above_1 < 58.0 and Q.e3_b2 < 4.696862e-06:
        z += 1129.39 * (58.0 - Q.n_pairs_kt_above_1) * (4.696862e-06 - Q.e3_b2)
    if Q.sd_mass > 42.31629 and Q.e3_b2 > 0.0002536827:
        z += -0.02737022 * (Q.sd_mass - 42.31629) * (Q.e3_b2 - 0.0002536827)
    if Q.mass_top30 > 146.3096 and Q.e4_b2 > 6.5e-11:
        z += 1.609716 * (Q.mass_top30 - 146.3096) * (Q.e4_b2 - 6.5e-11)
    if Q.mass > 114.0172 and Q.tau32_b2 < 0.8673543:
        z += -0.0004796519 * (Q.mass - 114.0172) * (0.8673543 - Q.tau32_b2)
    if Q.mass_top40 > 122.7145 and Q.sj3_z3 < 0.2075244:
        z += 0.01593085 * (Q.mass_top40 - 122.7145) * (0.2075244 - Q.sj3_z3)
    if Q.mass_top30 < 83.81888 and Q.pt1_over_pt0 < 0.4868628:
        z += 0.008953022 * (83.81888 - Q.mass_top30) * (0.4868628 - Q.pt1_over_pt0)
    if Q.sd_mass < 175.9333 and Q.pt1_over_pt0 < 0.6591684:
        z += 0.001340347 * (175.9333 - Q.sd_mass) * (0.6591684 - Q.pt1_over_pt0)
    return z


def neuron_70(Q):
    z = -0.1812737
    if Q.N2_b05 < 0.3667049:
        z += 3.253985 * Q.N2_b05 - 1.193253
    if Q.tau32_b2 < 0.4276838:
        z += -0.2234217 * Q.tau32_b2 + 0.09555387
    if Q.m012 >= 49.92073:
        z += -0.008286118 * Q.m012 + 0.413649
    if Q.mass_top50 < 99.54528:
        z += -0.001057517 * Q.mass_top50 + 0.1703939
    if 99.54528 <= Q.mass_top50 < 129.5874:
        z += 0.001427229 * Q.mass_top50 - 0.07695078
    if 129.5874 <= Q.mass_top50 < 161.1264:
        z += -0.005375138 * Q.mass_top50 + 0.8045505
    if Q.mass_top50 >= 161.1264:
        z += -0.004317621 * Q.mass_top50 + 0.6341565
    if Q.z_top3_slots < 0.654849:
        z += -0.3530797 * Q.z_top3_slots + 0.6563938
    if Q.z_top3_slots >= 0.654849:
        z += 0.6492794 * Q.z_top3_slots
    if Q.mass < 71.96396:
        z += 0.0155307 * Q.mass - 1.865821
    if 71.96396 <= Q.mass < 95.14961:
        z += 0.01657004 * Q.mass - 1.940617
    if 95.14961 <= Q.mass < 117.4867:
        z += 0.008083817 * Q.mass - 1.133156
    if 117.4867 <= Q.mass < 131.3917:
        z += 0.001282123 * Q.mass - 0.3340478
    if 131.3917 <= Q.mass < 164.4374:
        z += 0.007919298 * Q.mass - 1.206117
    if 164.4374 <= Q.mass < 182.8592:
        z += 0.001039343 * Q.mass - 0.07479521
    if Q.mass >= 182.8592:
        z += -0.00322111 * Q.mass + 0.7042678
    if -2.049856 <= Q.pair_mean_lndelta < -1.523797:
        z += -0.0389839 * Q.pair_mean_lndelta - 0.07991137
    if Q.pair_mean_lndelta >= -1.523797:
        z += -0.3601552 * Q.pair_mean_lndelta - 0.5693112
    if Q.e3_b05 >= 0.01046349:
        z += 44.53102 * Q.e3_b05 - 0.4659501
    if Q.n_pairs_kt_above_3 < 47.0:
        z += 0.008188142 * Q.n_pairs_kt_above_3 - 0.3848427
    if Q.sj3_mass2 < 1.824785:
        z += -0.09459542 * Q.sj3_mass2 + 0.1726163
    if Q.sum_z_dr2_top5 < 0.007457486:
        z += 12.45248 * Q.sum_z_dr2_top5 - 0.09286417
    if Q.max_pair_mass >= 55.24488:
        z += 0.02801473 * Q.max_pair_mass - 1.547671
    if -1.303913 <= Q.lund_max_lndelta < -0.9602344:
        z += -0.04375739 * Q.lund_max_lndelta - 0.05705584
    if Q.lund_max_lndelta >= -0.9602344:
        z += -0.261514 * Q.lund_max_lndelta - 0.2661532
    if Q.mass_top40 < 83.57316:
        z += -0.004021259 * Q.mass_top40 + 0.3360693
    if 78.4753 <= Q.sd_mass < 94.55722:
        z += 0.0190381 * Q.sd_mass - 1.49402
    if 94.55722 <= Q.sd_mass < 106.7501:
        z += 0.0003053788 * Q.sd_mass + 0.2772934
    if 106.7501 <= Q.sd_mass < 135.3031:
        z += -0.005971561 * Q.sd_mass + 0.9473572
    if 135.3031 <= Q.sd_mass < 154.5947:
        z += 0.003829686 * Q.sd_mass - 0.3787821
    if Q.sd_mass >= 154.5947:
        z += 0.006812444 * Q.sd_mass - 0.8399005
    if Q.sj3_pair_mass_max < 76.91486:
        z += 0.005600131 * Q.sj3_pair_mass_max - 0.1099337
    if 76.91486 <= Q.sj3_pair_mass_max < 109.8208:
        z += -0.009748987 * Q.sj3_pair_mass_max + 1.070642
    if Q.sj3_pair_mass_max >= 142.9952:
        z += 0.005186867 * Q.sj3_pair_mass_max - 0.7416973
    if Q.pair_mean_lnkt >= 0.6690549:
        z += 0.09689301 * Q.pair_mean_lnkt - 0.06482674
    if Q.n_pairs_kt_above_1 < 146.0:
        z += 0.002224776 * Q.n_pairs_kt_above_1 - 0.3248173
    if Q.ecf_g31 >= 0.001656914:
        z += -54.2482 * Q.ecf_g31 + 0.0898846
    if Q.e4_b05 < 4.363663e-05:
        z += 6103.319 * Q.e4_b05 - 0.2663283
    if Q.e4_b05 >= 7.503722e-05:
        z += -2077.449 * Q.e4_b05 + 0.155886
    if Q.dr_max_012 < 0.2054406:
        z += -1.137901 * Q.dr_max_012 + 0.233771
    if Q.pt1_dr01 >= 29.53862:
        z += 0.0002483999 * Q.pt1_dr01 - 0.007337391
    if Q.M2_b05 < 0.09835839:
        z += 0.5496772 * Q.M2_b05 - 0.05406537
    if Q.tau5 < 0.01467699:
        z += 10.72558 * Q.tau5 - 0.1574192
    if Q.pair_mean_lnm2 >= 2.09826:
        z += -0.05180607 * Q.pair_mean_lnm2 + 0.1087026
    if Q.M3_b2 < 0.00761379:
        z += 17.06267 * Q.M3_b2 - 0.1299116
    if Q.mass_over_sum_pt >= 0.2478639:
        z += 0.6180783 * Q.mass_over_sum_pt - 0.1531993
    if Q.jet_pt < 646.5833:
        z += 0.0001977718 * Q.jet_pt - 0.127876
    if Q.C2_b05 < 0.2234678:
        z += -2.85273 * Q.C2_b05 + 0.6374933
    if Q.sj3_pair_mass_min >= 53.29621:
        z += -2.832885e-05 * Q.sj3_pair_mass_min + 0.00150982
    if Q.sj3_pairmax_over_m < 0.6627397:
        z += 5.018064 * Q.sj3_pairmax_over_m - 3.325671
    if Q.C2 < 0.1411752:
        z += 1.183952 * Q.C2 - 0.1671447
    if Q.mass_top10 >= 103.4976:
        z += -0.004310071 * Q.mass_top10 + 0.4460822
    if Q.m012 > 49.92073 and Q.dr12 < 0.3613291:
        z += 0.01988752 * (Q.m012 - 49.92073) * (0.3613291 - Q.dr12)
    if Q.mass_top50 > 129.5874 and Q.n_pairs_kt_above_1 < 288.0:
        z += 2.371895e-05 * (Q.mass_top50 - 129.5874) * (288.0 - Q.n_pairs_kt_above_1)
    if Q.mass < 117.4867 and Q.z_2nd < 0.2273903:
        z += 0.04557298 * (117.4867 - Q.mass) * (0.2273903 - Q.z_2nd)
    if Q.mass > 182.8592 and Q.log_sum_pt < 6.45188:
        z += -0.08575118 * (Q.mass - 182.8592) * (6.45188 - Q.log_sum_pt)
    if Q.mass > 71.96396 and Q.sum_pt_top40 < 623.2168:
        z += 1.216564e-05 * (Q.mass - 71.96396) * (623.2168 - Q.sum_pt_top40)
    if Q.mass > 71.96396 and Q.n_pairs_kt_above_10 < 23.0:
        z += -6.601703e-05 * (Q.mass - 71.96396) * (23.0 - Q.n_pairs_kt_above_10)
    if Q.m012 > 49.92073 and Q.dr_2 > 0.0205536:
        z += 0.007804516 * (Q.m012 - 49.92073) * (Q.dr_2 - 0.0205536)
    if Q.mass_top50 < 161.1264 and Q.lne_12 > 1.787846:
        z += -0.001183598 * (161.1264 - Q.mass_top50) * (Q.lne_12 - 1.787846)
    if Q.mass_top50 > 99.54528 and Q.sj2_mass2 < 3.928781:
        z += -0.002277458 * (Q.mass_top50 - 99.54528) * (3.928781 - Q.sj2_mass2)
    if Q.tau32_b2 < 0.4276838 and Q.dr_10 > 0.3989771:
        z += -2.77722 * (0.4276838 - Q.tau32_b2) * (Q.dr_10 - 0.3989771)
    if Q.pair_mean_lndelta > -2.049856 and Q.dr_55 > 0.0:
        z += -0.09455572 * (Q.pair_mean_lndelta - -2.049856) * (Q.dr_55 - 0.0)
    if Q.z_top3_slots < 0.654849 and Q.tau43 > 0.568406:
        z += 1.101895 * (0.654849 - Q.z_top3_slots) * (Q.tau43 - 0.568406)
    if Q.mass < 95.14961 and Q.sj3_mass1 < 4.621465:
        z += 0.00413323 * (95.14961 - Q.mass) * (4.621465 - Q.sj3_mass1)
    if Q.z_top3_slots < 0.654849 and Q.C2_b2 < 0.221369:
        z += -2.752719 * (0.654849 - Q.z_top3_slots) * (0.221369 - Q.C2_b2)
    if Q.sd_mass > 154.5947 and Q.C3_b05 > 0.1643977:
        z += -0.04053498 * (Q.sd_mass - 154.5947) * (Q.C3_b05 - 0.1643977)
    if Q.sd_mass > 154.5947 and Q.sd_zg < 0.2652027:
        z += -0.02265419 * (Q.sd_mass - 154.5947) * (0.2652027 - Q.sd_zg)
    if Q.z_top3_slots < 0.654849 and Q.dr_max_012 < 0.4410123:
        z += -1.007641 * (0.654849 - Q.z_top3_slots) * (0.4410123 - Q.dr_max_012)
    if Q.mass < 117.4867 and Q.z_3rd > 0.07296058:
        z += -0.009669465 * (117.4867 - Q.mass) * (Q.z_3rd - 0.07296058)
    if Q.mass_top50 > 99.54528 and Q.sj2_mass1 < 91.2852:
        z += -2.262798e-05 * (Q.mass_top50 - 99.54528) * (91.2852 - Q.sj2_mass1)
    if Q.m012 > 49.92073 and Q.sj3_pairmin_over_m < 0.3713204:
        z += 0.02447451 * (Q.m012 - 49.92073) * (0.3713204 - Q.sj3_pairmin_over_m)
    if Q.n_pairs_kt_above_1 < 146.0 and Q.sj3_pairmin_over_m > 0.1636952:
        z += 0.01482314 * (146.0 - Q.n_pairs_kt_above_1) * (Q.sj3_pairmin_over_m - 0.1636952)
    if Q.n_pairs_kt_above_3 < 47.0 and Q.sum_z_dr2_top2 < 0.0420789:
        z += 0.4407283 * (47.0 - Q.n_pairs_kt_above_3) * (0.0420789 - Q.sum_z_dr2_top2)
    if Q.e4_b05 < 4.363663e-05 and Q.dr_1 > 0.05473942:
        z += -5681.423 * (4.363663e-05 - Q.e4_b05) * (Q.dr_1 - 0.05473942)
    if Q.max_pair_mass > 55.24488 and Q.jet_e > 551.9263:
        z += 2.285079e-06 * (Q.max_pair_mass - 55.24488) * (Q.jet_e - 551.9263)
    if Q.tau32_b2 < 0.4276838 and Q.dr_32 < 0.1686866:
        z += 4.061592 * (0.4276838 - Q.tau32_b2) * (0.1686866 - Q.dr_32)
    if Q.tau32_b2 < 0.4276838 and Q.sum_z_dr2_top2 > 0.0420789:
        z += -12.48973 * (0.4276838 - Q.tau32_b2) * (Q.sum_z_dr2_top2 - 0.0420789)
    if Q.ecf_g31 > 0.001656914 and Q.eta_1 < -0.1473389:
        z += 54.5309 * (Q.ecf_g31 - 0.001656914) * (-0.1473389 - Q.eta_1)
    if Q.mass_top50 > 129.5874 and Q.dr_22 < 0.02915846:
        z += -0.09619988 * (Q.mass_top50 - 129.5874) * (0.02915846 - Q.dr_22)
    if Q.pair_mean_lndelta > -2.049856 and Q.dr_22 < 0.02915846:
        z += 12.94263 * (Q.pair_mean_lndelta - -2.049856) * (0.02915846 - Q.dr_22)
    if Q.e3_b05 > 0.01046349 and Q.sj3_z1 < 0.68373:
        z += 86.59756 * (Q.e3_b05 - 0.01046349) * (0.68373 - Q.sj3_z1)
    if Q.n_pairs_kt_above_1 < 146.0 and Q.sum_e > 1247.266:
        z += -2.624596e-06 * (146.0 - Q.n_pairs_kt_above_1) * (Q.sum_e - 1247.266)
    if Q.max_pair_mass > 55.24488 and Q.eta_23 > -0.3259277:
        z += -0.01069236 * (Q.max_pair_mass - 55.24488) * (Q.eta_23 - -0.3259277)
    if Q.M3_b2 < 0.00761379 and Q.eta_57 > 0.07013245:
        z += 36.54875 * (0.00761379 - Q.M3_b2) * (Q.eta_57 - 0.07013245)
    if Q.sd_mass > 106.7501 and Q.phi_17 > 0.328125:
        z += -0.008528664 * (Q.sd_mass - 106.7501) * (Q.phi_17 - 0.328125)
    if Q.z_top3_slots < 0.654849 and Q.lund3_lnkt > 3.058345:
        z += 0.1308909 * (0.654849 - Q.z_top3_slots) * (Q.lund3_lnkt - 3.058345)
    if Q.pt1_dr01 > 29.53862 and Q.dr_5 < 0.1959828:
        z += -0.023907 * (Q.pt1_dr01 - 29.53862) * (0.1959828 - Q.dr_5)
    if Q.ecf_g31 > 0.001656914 and Q.dr_16 < 0.01827585:
        z += 2774.574 * (Q.ecf_g31 - 0.001656914) * (0.01827585 - Q.dr_16)
    if Q.ecf_g31 > 0.001656914 and Q.sj3_z3 < 0.0184643:
        z += -748.752 * (Q.ecf_g31 - 0.001656914) * (0.0184643 - Q.sj3_z3)
    if Q.mass > 71.96396 and Q.C2_b2 > 0.02858957:
        z += -4.490215e-05 * (Q.mass - 71.96396) * (Q.C2_b2 - 0.02858957)
    if Q.mass < 164.4374 and Q.sj3_mass1 > 11.10125:
        z += -6.039995e-06 * (164.4374 - Q.mass) * (Q.sj3_mass1 - 11.10125)
    if Q.lund_max_lndelta > -0.9602344 and Q.dr_18 > 0.09105897:
        z += 0.06431141 * (Q.lund_max_lndelta - -0.9602344) * (Q.dr_18 - 0.09105897)
    if Q.ecf_g31 > 0.001656914 and Q.tau54 > 0.7835935:
        z += 66.55663 * (Q.ecf_g31 - 0.001656914) * (Q.tau54 - 0.7835935)
    if Q.mass > 182.8592 and Q.eta_23 > 0.3276367:
        z += 3.258575e-05 * (Q.mass - 182.8592) * (Q.eta_23 - 0.3276367)
    return z


def neuron_71(Q):
    z = 1.181428e-05
    return z


def neuron_72(Q):
    z = -2.547756e-05
    return z


def neuron_73(Q):
    z = -2.301758e-05
    return z


def neuron_74(Q):
    z = 1.363609e-05
    return z


def neuron_75(Q):
    z = -1.582183e-06
    return z


def neuron_76(Q):
    z = 2.739781e-07
    return z


def neuron_77(Q):
    z = -2.005252e-05
    return z


def neuron_78(Q):
    z = 3.790103e-06
    return z


def neuron_79(Q):
    z = 1.50548e-07
    return z


def neuron_80(Q):
    z = 3.80459e-05
    return z


def neuron_81(Q):
    z = -0.4214839
    if Q.tau21_b2 < 0.3265797:
        z += 0.1973461 * Q.tau21_b2 - 0.06444923
    if Q.tau32 < 0.5871757:
        z += -0.634006 * Q.tau32 + 0.3722729
    if Q.lne_0 >= 5.423297:
        z += -0.001995399 * Q.lne_0 + 0.01082164
    z += 1.390233 * Q.N2_b05
    if Q.mass < 71.96396:
        z += 0.0111034 * Q.mass - 0.94087
    if 71.96396 <= Q.mass < 79.47361:
        z += -0.02237854 * Q.mass + 1.468623
    if 79.47361 <= Q.mass < 95.14961:
        z += -0.006920113 * Q.mass + 0.2400865
    if 95.14961 <= Q.mass < 110.2019:
        z += 0.00776162 * Q.mass - 1.156875
    if 110.2019 <= Q.mass < 149.0507:
        z += 0.01105334 * Q.mass - 1.519629
    if Q.mass >= 149.0507:
        z += 0.003291722 * Q.mass - 0.362754
    if Q.sj3_pair_mass_max < 73.24742:
        z += -0.004669916 * Q.sj3_pair_mass_max + 0.4150143
    if 73.24742 <= Q.sj3_pair_mass_max < 76.91486:
        z += 0.006132695 * Q.sj3_pair_mass_max - 0.376249
    if 76.91486 <= Q.sj3_pair_mass_max < 105.4845:
        z += -0.003340825 * Q.sj3_pair_mass_max + 0.3524054
    if Q.n_pairs_kt_above_3 < 34.0:
        z += 0.007194459 * Q.n_pairs_kt_above_3 - 0.2446116
    if Q.mass_top50 < 71.80762:
        z += -0.00576771 * Q.mass_top50 + 0.7817368
    if 71.80762 <= Q.mass_top50 < 108.5283:
        z += -0.003075052 * Q.mass_top50 + 0.5883835
    if 108.5283 <= Q.mass_top50 < 135.5368:
        z += -0.01194994 * Q.mass_top50 + 1.55156
    if 135.5368 <= Q.mass_top50 < 146.203:
        z += -0.006182233 * Q.mass_top50 + 0.7698233
    if Q.mass_top50 >= 146.203:
        z += -0.004847793 * Q.mass_top50 + 0.5747241
    if Q.e3_b2 < 0.000125186:
        z += 2644.521 * Q.e3_b2 - 0.3310571
    if -1.498181 <= Q.lund_max_lndelta < -0.8095462:
        z += -0.008444216 * Q.lund_max_lndelta - 0.01265097
    if Q.lund_max_lndelta >= -0.8095462:
        z += -0.1665349 * Q.lund_max_lndelta - 0.1406327
    if Q.N2 >= 0.2734424:
        z += -1.660564 * Q.N2 + 0.4540686
    if Q.lnptrel_0 >= -1.266303:
        z += 0.1983255 * Q.lnptrel_0 + 0.2511401
    if Q.M3_b2 < 0.01202341:
        z += -1.26478 * Q.M3_b2 + 0.01520697
    if 0.1701175 <= Q.pair_mean_lnkt < 0.9367772:
        z += 0.2626141 * Q.pair_mean_lnkt - 0.04467526
    if Q.pair_mean_lnkt >= 0.9367772:
        z += 0.2942526 * Q.pair_mean_lnkt - 0.07431349
    if 62.03827 <= Q.sd_mass < 78.4753:
        z += 0.01093957 * Q.sd_mass - 0.6786722
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += -0.03169149 * Q.sd_mass + 2.666813
    if 88.81751 <= Q.sd_mass < 94.55722:
        z += -0.01101096 * Q.sd_mass + 0.8300197
    if 94.55722 <= Q.sd_mass < 106.7501:
        z += 0.01739424 * Q.sd_mass - 1.855897
    if 106.7501 <= Q.sd_mass < 115.7091:
        z += 0.007790674 * Q.sd_mass - 0.8307153
    if 115.7091 <= Q.sd_mass < 123.2919:
        z += -0.002088722 * Q.sd_mass + 0.3124205
    if 123.2919 <= Q.sd_mass < 154.5947:
        z += 0.004073633 * Q.sd_mass - 0.4473477
    if 154.5947 <= Q.sd_mass < 175.9333:
        z += 0.003288828 * Q.sd_mass - 0.326021
    if Q.sd_mass >= 175.9333:
        z += 0.006574164 * Q.sd_mass - 0.9040211
    if Q.n_pairs_kt_above_1 < 101.0:
        z += 0.0008012205 * Q.n_pairs_kt_above_1 - 0.2847024
    if 101.0 <= Q.n_pairs_kt_above_1 < 411.0:
        z += 0.000657352 * Q.n_pairs_kt_above_1 - 0.2701717
    if Q.mass_top40 < 132.5189:
        z += -0.001353685 * Q.mass_top40 + 0.1793889
    if Q.sd_rg >= 0.4449101:
        z += -0.1280421 * Q.sd_rg + 0.05696725
    if Q.max_pair_mass >= 25.35141:
        z += -0.01095099 * Q.max_pair_mass + 0.277623
    if Q.pair_mean_lndelta >= -1.352792:
        z += 0.09289126 * Q.pair_mean_lndelta + 0.1256625
    if Q.pair_mean_lnm2 >= 3.654683:
        z += -0.07819987 * Q.pair_mean_lnm2 + 0.2857958
    if Q.pair_max_lnkt < 1.555555:
        z += 0.2331178 * Q.pair_max_lnkt - 0.3626276
    if Q.pair_max_lnkt >= 2.98386:
        z += -0.02147615 * Q.pair_max_lnkt + 0.06408184
    if Q.sj3_dr_min < 0.2638636:
        z += -0.5482584 * Q.sj3_dr_min + 0.1446655
    if Q.e2_b2 < 0.06095651:
        z += -4.222671 * Q.e2_b2 + 0.2573993
    if Q.sum_pt_top10 < 470.9609:
        z += 0.0007332414 * Q.sum_pt_top10 - 0.345328
    if Q.e3_b05 < 0.0131933:
        z += 18.71904 * Q.e3_b05 - 0.2469658
    if Q.lnerel_9 >= -3.64491:
        z += 0.0189337 * Q.lnerel_9 + 0.06901163
    if Q.sj2_mass1 < 16.48115:
        z += -0.005406335 * Q.sj2_mass1 + 0.08910262
    if Q.mass_top10 >= 89.15489:
        z += -0.005411189 * Q.mass_top10 + 0.4824339
    if Q.m012 >= 57.33311:
        z += 0.009733979 * Q.m012 - 0.5580793
    if Q.M2_b05 < 0.1651975:
        z += -1.985572 * Q.M2_b05 + 0.3280115
    if Q.pt1_dr01 >= 29.53862:
        z += 0.0007582163 * Q.pt1_dr01 - 0.02239666
    if Q.sj2_mass2 < 1.851735:
        z += -0.2478559 * Q.sj2_mass2 + 0.4589636
    if Q.mass < 149.0507 and Q.sj3_pairmax_over_m < 0.897453:
        z += 0.007219292 * (149.0507 - Q.mass) * (0.897453 - Q.sj3_pairmax_over_m)
    if Q.tau32 < 0.5871757 and Q.sj2_mass1 > 77.42768:
        z += -0.02659499 * (0.5871757 - Q.tau32) * (Q.sj2_mass1 - 77.42768)
    if Q.mass < 149.0507 and Q.min_pair_mass > 10.59762:
        z += 0.0006569155 * (149.0507 - Q.mass) * (Q.min_pair_mass - 10.59762)
    if Q.tau32 < 0.5871757 and Q.sj3_mass3 < 1.053727:
        z += -0.1214281 * (0.5871757 - Q.tau32) * (1.053727 - Q.sj3_mass3)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.pair_max_lnkt < 3.338396:
        z += 0.004784068 * (34.0 - Q.n_pairs_kt_above_3) * (3.338396 - Q.pair_max_lnkt)
    if Q.mass_top50 > 108.5283 and Q.n_lund < 5.0:
        z += 0.002658427 * (Q.mass_top50 - 108.5283) * (5.0 - Q.n_lund)
    if Q.mass < 95.14961 and Q.n_dr_0_0p05 > 2.0:
        z += -0.0004293563 * (95.14961 - Q.mass) * (Q.n_dr_0_0p05 - 2.0)
    if Q.mass < 149.0507 and Q.lnerel_1 < -2.500134:
        z += -0.002842988 * (149.0507 - Q.mass) * (-2.500134 - Q.lnerel_1)
    if Q.tau21_b2 < 0.3265797 and Q.n_dr_0p05_0p1 > 7.0:
        z += -0.04543058 * (0.3265797 - Q.tau21_b2) * (Q.n_dr_0p05_0p1 - 7.0)
    if Q.tau32 < 0.5871757 and Q.sj3_dr_max > 0.6054934:
        z += -0.323541 * (0.5871757 - Q.tau32) * (Q.sj3_dr_max - 0.6054934)
    if Q.tau32 < 0.5871757 and Q.n_real_top20 < 20.0:
        z += -0.1529243 * (0.5871757 - Q.tau32) * (20.0 - Q.n_real_top20)
    if Q.mass_top50 < 135.5368 and Q.dr01 > 0.009409147:
        z += -0.01002728 * (135.5368 - Q.mass_top50) * (Q.dr01 - 0.009409147)
    if Q.mass_top50 < 135.5368 and Q.sd_zg > 0.2450652:
        z += -0.001542388 * (135.5368 - Q.mass_top50) * (Q.sd_zg - 0.2450652)
    if Q.tau21_b2 < 0.3265797 and Q.mass_top2 > 30.2961:
        z += 0.03683316 * (0.3265797 - Q.tau21_b2) * (Q.mass_top2 - 30.2961)
    if Q.tau32 < 0.5871757 and Q.C2_b2 < 0.03403084:
        z += -17.47185 * (0.5871757 - Q.tau32) * (0.03403084 - Q.C2_b2)
    if Q.tau32 < 0.5871757 and Q.lam2 > 0.00972048:
        z += 5.869868 * (0.5871757 - Q.tau32) * (Q.lam2 - 0.00972048)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.z_dr_0p2_0p4 > 0.1986699:
        z += -0.045743 * (34.0 - Q.n_pairs_kt_above_3) * (Q.z_dr_0p2_0p4 - 0.1986699)
    if Q.sj3_pair_mass_max < 73.24742 and Q.ecf_g43 > 2.37474e-07:
        z += -717.9959 * (73.24742 - Q.sj3_pair_mass_max) * (Q.ecf_g43 - 2.37474e-07)
    if Q.mass < 149.0507 and Q.ecf_g43 > 5.642202e-06:
        z += 60.60213 * (149.0507 - Q.mass) * (Q.ecf_g43 - 5.642202e-06)
    if Q.mass_top50 < 135.5368 and Q.sum_e > 798.3291:
        z += 2.65037e-06 * (135.5368 - Q.mass_top50) * (Q.sum_e - 798.3291)
    if Q.N2 > 0.2734424 and Q.sj3_z1 > 0.4064022:
        z += 0.8752167 * (Q.N2 - 0.2734424) * (Q.sj3_z1 - 0.4064022)
    if Q.mass_top50 > 146.203 and Q.tau32_b2 < 0.6203012:
        z += -0.008779586 * (Q.mass_top50 - 146.203) * (0.6203012 - Q.tau32_b2)
    if Q.e3_b2 < 0.000125186 and Q.lne_27 > 0.04953394:
        z += -293.6637 * (0.000125186 - Q.e3_b2) * (Q.lne_27 - 0.04953394)
    if Q.tau32 < 0.5871757 and Q.lne_4 < 3.516199:
        z += 0.679414 * (0.5871757 - Q.tau32) * (3.516199 - Q.lne_4)
    if Q.sd_mass > 123.2919 and Q.dr12 < 0.2116473:
        z += -0.0124094 * (Q.sd_mass - 123.2919) * (0.2116473 - Q.dr12)
    if Q.M3_b2 < 0.01202341 and Q.centroid_offset < 0.00586924:
        z += 1.30893 * (0.01202341 - Q.M3_b2) * (0.00586924 - Q.centroid_offset)
    if Q.e3_b2 < 0.000125186 and Q.eta_37 < -0.09778748:
        z += 982.3208 * (0.000125186 - Q.e3_b2) * (-0.09778748 - Q.eta_37)
    if Q.e3_b2 < 0.000125186 and Q.sj3_z3 < 0.2303838:
        z += 4837.284 * (0.000125186 - Q.e3_b2) * (0.2303838 - Q.sj3_z3)
    if Q.mass_top40 < 132.5189 and Q.tau32_b2 > 0.3377343:
        z += -0.003617188 * (132.5189 - Q.mass_top40) * (Q.tau32_b2 - 0.3377343)
    if Q.sd_mass > 106.7501 and Q.max_dr < 0.5286514:
        z += 0.02640813 * (Q.sd_mass - 106.7501) * (0.5286514 - Q.max_dr)
    if Q.mass_top50 > 146.203 and Q.max_dr < 0.4811724:
        z += 0.2015274 * (Q.mass_top50 - 146.203) * (0.4811724 - Q.max_dr)
    if Q.lnptrel_0 > -1.266303 and Q.max_dr < 0.4811724:
        z += -0.8569774 * (Q.lnptrel_0 - -1.266303) * (0.4811724 - Q.max_dr)
    if Q.mass_top50 > 71.80762 and Q.sj3_dr13 > 0.07631451:
        z += -0.001972918 * (Q.mass_top50 - 71.80762) * (Q.sj3_dr13 - 0.07631451)
    if Q.sj3_dr_min < 0.2638636 and Q.dr_9 < 0.1648004:
        z += -0.3874983 * (0.2638636 - Q.sj3_dr_min) * (0.1648004 - Q.dr_9)
    if Q.sum_pt_top10 < 470.9609 and Q.max_dr > 0.7339085:
        z += 0.0002356879 * (470.9609 - Q.sum_pt_top10) * (Q.max_dr - 0.7339085)
    if Q.n_pairs_kt_above_1 < 101.0 and Q.n_dr_0p05_0p1 > 16.0:
        z += -0.001766173 * (101.0 - Q.n_pairs_kt_above_1) * (Q.n_dr_0p05_0p1 - 16.0)
    if Q.sj2_mass1 < 16.48115 and Q.phi_26 > 0.3427734:
        z += -0.04188012 * (16.48115 - Q.sj2_mass1) * (Q.phi_26 - 0.3427734)
    if Q.M3_b2 < 0.01202341 and Q.tau54 < 0.9036234:
        z += -11.90212 * (0.01202341 - Q.M3_b2) * (0.9036234 - Q.tau54)
    if Q.pair_mean_lnkt > 0.9367772 and Q.dr_17 < 0.2325538:
        z += 1.66042 * (Q.pair_mean_lnkt - 0.9367772) * (0.2325538 - Q.dr_17)
    if Q.tau32 < 0.5871757 and Q.eta_4 < 0.02658081:
        z += -0.4143741 * (0.5871757 - Q.tau32) * (0.02658081 - Q.eta_4)
    if Q.mass_top50 > 71.80762 and Q.n_dr_0p05_0p1 > 5.0:
        z += -9.559331e-05 * (Q.mass_top50 - 71.80762) * (Q.n_dr_0p05_0p1 - 5.0)
    if Q.pair_mean_lnm2 > 3.654683 and Q.dr_23 < 0.4020128:
        z += 0.1504901 * (Q.pair_mean_lnm2 - 3.654683) * (0.4020128 - Q.dr_23)
    if Q.M3_b2 < 0.01202341 and Q.eta_29 < -0.03344727:
        z += 13.83216 * (0.01202341 - Q.M3_b2) * (-0.03344727 - Q.eta_29)
    return z


def neuron_82(Q):
    z = 0.0004023885
    return z


def neuron_83(Q):
    z = 1.658984e-05
    return z


def neuron_84(Q):
    z = 2.650188e-06
    return z


def neuron_85(Q):
    z = 6.717487e-06
    return z


def neuron_86(Q):
    z = 1.268771
    if Q.tau3 < 0.07659457:
        z += -12.34205 * Q.tau3 + 0.945334
    if Q.M3 < 0.02540381:
        z += 19.23483 * Q.M3 - 0.8113207
    if 0.02540381 <= Q.M3 < 0.03259227:
        z += 38.98056 * Q.M3 - 1.312937
    if 0.03259227 <= Q.M3 < 0.04863496:
        z += 31.24092 * Q.M3 - 1.060685
    if Q.M3 >= 0.04863496:
        z += 19.74572 * Q.M3 - 0.5016167
    if Q.mass_top30 < 67.20576:
        z += 0.01514665 * Q.mass_top30 - 1.017942
    if Q.mass < 90.08945:
        z += 0.01975987 * Q.mass - 0.9950711
    if 90.08945 <= Q.mass < 95.14961:
        z += 0.004767169 * Q.mass + 0.3556132
    if 95.14961 <= Q.mass < 123.7928:
        z += -0.02560053 * Q.mass + 3.245087
    if 123.7928 <= Q.mass < 131.3917:
        z += -0.009991888 * Q.mass + 1.312851
    if Q.mass_top40 < 92.68149:
        z += -0.004039939 * Q.mass_top40 + 0.3744276
    if Q.e3_b2 < 0.0007909605:
        z += 434.3631 * Q.e3_b2 - 0.343564
    if Q.n_pairs_kt_above_3 < 54.0:
        z += 0.002472923 * Q.n_pairs_kt_above_3 + 0.1135681
    if 54.0 <= Q.n_pairs_kt_above_3 < 220.0:
        z += -0.00148859 * Q.n_pairs_kt_above_3 + 0.3274898
    if Q.dr_min_012 < 0.01600475:
        z += 8.899566 * Q.dr_min_012 - 0.1424354
    if Q.tau2 < 0.07991731:
        z += 1.96606 * Q.tau2 - 0.1571222
    if Q.n_pairs_kt_above_1 < 123.0:
        z += 0.0004266909 * Q.n_pairs_kt_above_1 - 0.05248299
    z += -13.78949 * Q.tau5
    if Q.M2_b05 >= 0.1258758:
        z += -13.45582 * Q.M2_b05 + 1.693762
    if Q.lund2_lndelta >= -1.124734:
        z += 0.1988141 * Q.lund2_lndelta + 0.2236129
    if Q.lne_22 >= 0.3560888:
        z += 0.1194103 * Q.lne_22 - 0.04252068
    if Q.n_particles >= 23.0:
        z += -0.02547329 * Q.n_particles + 0.5858856
    if Q.z_top15_slots >= 0.6733431:
        z += -8.019785 * Q.z_top15_slots + 5.400067
    if Q.m01 < 1.868856:
        z += 0.2046684 * Q.m01 - 0.3824957
    if Q.n_dr_0p1_0p2 < 5.0:
        z += -0.06704129 * Q.n_dr_0p1_0p2 + 0.3352064
    if Q.n_dr_0_0p05 < 4.0:
        z += -0.02614895 * Q.n_dr_0_0p05 + 0.1045958
    if Q.n_dr_0p4_up < 4.0:
        z += -0.06219222 * Q.n_dr_0p4_up + 0.2487689
    if Q.n_dr_0p4_up >= 9.0:
        z += 0.02322813 * Q.n_dr_0p4_up - 0.2090531
    if Q.lund_max_lndelta >= -0.8095462:
        z += -0.2655657 * Q.lund_max_lndelta - 0.2149877
    if Q.ecf_g43 < 3.375095e-06:
        z += 67451.49 * Q.ecf_g43 - 0.2276552
    if Q.M3_b2 < 0.01919357:
        z += -4.001288 * Q.M3_b2 + 0.076799
    if Q.M2 < 0.06625964:
        z += -10.90859 * Q.M2 + 0.722799
    if Q.D2_b2 < 3.019755:
        z += 0.09390717 * Q.D2_b2 - 0.2835767
    if Q.z_top30_slots >= 0.8695891:
        z += 7.333893 * Q.z_top30_slots - 6.377473
    if Q.pair_max_lnm2 >= 7.752558:
        z += 0.3669612 * Q.pair_max_lnm2 - 2.844888
    if Q.mass_over_sum_pt < 0.1445164:
        z += -0.1411961 * Q.mass_over_sum_pt + 0.02040516
    if Q.sj3_pair_mass_max < 56.73313:
        z += -0.01182336 * Q.sj3_pair_mass_max + 0.670776
    if Q.z_dr_0p05_0p1 < 0.008254376:
        z += -17.35207 * Q.z_dr_0p05_0p1 + 0.1432305
    if Q.tau21_b2 < 0.3265797:
        z += 0.3034036 * Q.tau21_b2 - 0.09908545
    if Q.sj3_mass1 < 18.73268:
        z += -0.01174452 * Q.sj3_mass1 + 0.2200063
    if Q.sj3_mass3 < 1.617171:
        z += -0.1297387 * Q.sj3_mass3 + 0.2098097
    if Q.N3_b2 < 0.4545826:
        z += 0.2802156 * Q.N3_b2 - 0.1273811
    if Q.dr02 < 0.02252173:
        z += 5.229283 * Q.dr02 - 0.1177725
    if Q.lne_20 >= 2.544918:
        z += 0.01464316 * Q.lne_20 - 0.03726564
    if Q.n_lund_kt_above_1 < 5.0:
        z += 0.01638375 * Q.n_lund_kt_above_1 - 0.08191874
    if Q.lnerel_50 >= -5.787293:
        z += -0.2436596 * Q.lnerel_50 - 1.410129
    if Q.z_top10_slots < 0.5546022:
        z += -2.586209 * Q.z_top10_slots + 1.376719
    if 0.5546022 <= Q.z_top10_slots < 0.6922585:
        z += 0.4184175 * Q.z_top10_slots - 0.2896531
    z += 9.142147 * Q.M2_b2
    if Q.sj3_dr_min < 0.1259759:
        z += 2.356142 * Q.sj3_dr_min - 0.2968172
    if Q.D2_b05 < 1.954943:
        z += 0.2223028 * Q.D2_b05 - 0.4345893
    if Q.z_dr_0p4_up < 0.004729211:
        z += 20.85572 * Q.z_dr_0p4_up - 0.09863113
    if Q.lam2 >= 0.02148541:
        z += -18.81157 * Q.lam2 + 0.4041742
    if Q.mass < 131.3917 and Q.N2_b2 < 0.2099278:
        z += 0.1106576 * (131.3917 - Q.mass) * (0.2099278 - Q.N2_b2)
    if Q.tau3 < 0.07659457 and Q.n_lund_kt_above_1 > 6.0:
        z += -0.934362 * (0.07659457 - Q.tau3) * (Q.n_lund_kt_above_1 - 6.0)
    if Q.tau3 < 0.07659457 and Q.lund3_lndelta > -2.817283:
        z += -1.733673 * (0.07659457 - Q.tau3) * (Q.lund3_lndelta - -2.817283)
    if Q.e3_b2 < 0.0007909605 and Q.sj2_mass1 > 60.04967:
        z += 3.742412 * (0.0007909605 - Q.e3_b2) * (Q.sj2_mass1 - 60.04967)
    if Q.tau2 < 0.07991731 and Q.sum_z_dr2_top10 > 0.04905122:
        z += -853.3036 * (0.07991731 - Q.tau2) * (Q.sum_z_dr2_top10 - 0.04905122)
    if Q.tau2 < 0.07991731 and Q.sj3_z1 < 0.7109507:
        z += -13.95011 * (0.07991731 - Q.tau2) * (0.7109507 - Q.sj3_z1)
    if Q.tau3 < 0.07659457 and Q.eta_37 > 0.1695557:
        z += 0.9543089 * (0.07659457 - Q.tau3) * (Q.eta_37 - 0.1695557)
    if Q.M3 < 0.04863496 and Q.n_pairs_kt_above_10 < 10.0:
        z += 0.3486782 * (0.04863496 - Q.M3) * (10.0 - Q.n_pairs_kt_above_10)
    if Q.dr_min_012 < 0.01600475 and Q.dr_35 < 0.2119274:
        z += -12.04868 * (0.01600475 - Q.dr_min_012) * (0.2119274 - Q.dr_35)
    if Q.M3 > 0.02540381 and Q.n_dr_0p05_0p1 > 1.0:
        z += -0.6548805 * (Q.M3 - 0.02540381) * (Q.n_dr_0p05_0p1 - 1.0)
    if Q.D2_b2 < 3.019755 and Q.pt1_dr01 < 0.3544847:
        z += 0.2799361 * (3.019755 - Q.D2_b2) * (0.3544847 - Q.pt1_dr01)
    if Q.lne_22 > 0.3560888 and Q.dr01 < 0.04908137:
        z += 1.416669 * (Q.lne_22 - 0.3560888) * (0.04908137 - Q.dr01)
    if Q.n_particles > 23.0 and Q.lund3_lnz < -1.763117:
        z += -0.0002898009 * (Q.n_particles - 23.0) * (-1.763117 - Q.lund3_lnz)
    if Q.z_top30_slots > 0.8695891 and Q.sj3_dr_min < 0.2410564:
        z += 9.013688 * (Q.z_top30_slots - 0.8695891) * (0.2410564 - Q.sj3_dr_min)
    if Q.z_dr_0p05_0p1 < 0.008254376 and Q.phi_19 < -0.02503967:
        z += 9.075502 * (0.008254376 - Q.z_dr_0p05_0p1) * (-0.02503967 - Q.phi_19)
    if Q.M3 < 0.03259227 and Q.eta_50 > 0.03570557:
        z += 9.819242 * (0.03259227 - Q.M3) * (Q.eta_50 - 0.03570557)
    if Q.N3_b2 < 0.4545826 and Q.dr_19 > 0.3410959:
        z += 0.5342898 * (0.4545826 - Q.N3_b2) * (Q.dr_19 - 0.3410959)
    if Q.sj3_mass1 < 18.73268 and Q.dr_33 < 0.02553503:
        z += -0.3544034 * (18.73268 - Q.sj3_mass1) * (0.02553503 - Q.dr_33)
    if Q.dr02 < 0.02252173 and Q.dr_14 > 0.2250629:
        z += -14.29961 * (0.02252173 - Q.dr02) * (Q.dr_14 - 0.2250629)
    if Q.D2_b2 < 3.019755 and Q.sj3_mass2 > 6.769801:
        z += -0.001945213 * (3.019755 - Q.D2_b2) * (Q.sj3_mass2 - 6.769801)
    if Q.tau21_b2 < 0.3265797 and Q.sj2_mass2 < 5.311506:
        z += -0.3915064 * (0.3265797 - Q.tau21_b2) * (5.311506 - Q.sj2_mass2)
    if Q.D2_b2 < 3.019755 and Q.eta_55 > 0.1072388:
        z += 0.02687938 * (3.019755 - Q.D2_b2) * (Q.eta_55 - 0.1072388)
    if Q.tau3 < 0.07659457 and Q.jet_e < 1650.879:
        z += -0.004659066 * (0.07659457 - Q.tau3) * (1650.879 - Q.jet_e)
    if Q.n_particles > 23.0 and Q.dr_7 > 0.3042435:
        z += -0.009340059 * (Q.n_particles - 23.0) * (Q.dr_7 - 0.3042435)
    if Q.tau2 < 0.07991731 and Q.dr_53 < 0.1189606:
        z += 33.94183 * (0.07991731 - Q.tau2) * (0.1189606 - Q.dr_53)
    if Q.lne_22 > 0.3560888 and Q.dr_34 > 0.3708198:
        z += 0.05738495 * (Q.lne_22 - 0.3560888) * (Q.dr_34 - 0.3708198)
    if Q.M2_b05 > 0.1258758 and Q.dr_53 < 0.3969707:
        z += 1.35698 * (Q.M2_b05 - 0.1258758) * (0.3969707 - Q.dr_53)
    if Q.N3_b2 < 0.4545826 and Q.dr_9 > 0.1346091:
        z += -0.3986843 * (0.4545826 - Q.N3_b2) * (Q.dr_9 - 0.1346091)
    if Q.M3_b2 < 0.01919357 and Q.dr_16 < 0.4540665:
        z += -2.781009 * (0.01919357 - Q.M3_b2) * (0.4540665 - Q.dr_16)
    if Q.pair_max_lnm2 > 7.752558 and Q.dr_13 < 0.3615236:
        z += 0.05472905 * (Q.pair_max_lnm2 - 7.752558) * (0.3615236 - Q.dr_13)
    if Q.tau2 < 0.07991731 and Q.lnpt_10 > 3.138208:
        z += 6.42614 * (0.07991731 - Q.tau2) * (Q.lnpt_10 - 3.138208)
    if Q.D2_b2 < 3.019755 and Q.phi_22 < -0.2314453:
        z += 0.09191325 * (3.019755 - Q.D2_b2) * (-0.2314453 - Q.phi_22)
    if Q.lne_22 > 0.3560888 and Q.planar_flow < 0.8202927:
        z += -0.007752032 * (Q.lne_22 - 0.3560888) * (0.8202927 - Q.planar_flow)
    if Q.lund2_lndelta > -1.124734 and Q.dr_max_012 > 0.5651787:
        z += -0.6698701 * (Q.lund2_lndelta - -1.124734) * (Q.dr_max_012 - 0.5651787)
    if Q.pair_max_lnm2 > 7.752558 and Q.n_pairs_kt_above_30 > 1.0:
        z += 0.0007129572 * (Q.pair_max_lnm2 - 7.752558) * (Q.n_pairs_kt_above_30 - 1.0)
    if Q.M3_b2 < 0.01919357 and Q.sj2_mass2 > 3.928781:
        z += 0.2098589 * (0.01919357 - Q.M3_b2) * (Q.sj2_mass2 - 3.928781)
    if Q.n_dr_0p4_up < 4.0 and Q.sj4_pair_mass_min > 16.05703:
        z += -0.001707053 * (4.0 - Q.n_dr_0p4_up) * (Q.sj4_pair_mass_min - 16.05703)
    if Q.dr_min_012 < 0.01600475 and Q.phi_54 > -0.1107788:
        z += 7.587605 * (0.01600475 - Q.dr_min_012) * (Q.phi_54 - -0.1107788)
    return z


def neuron_87(Q):
    z = 1.63959e-06
    return z


def neuron_88(Q):
    z = 4.16205e-05
    return z


def neuron_89(Q):
    z = 7.851283e-06
    return z


def neuron_90(Q):
    z = -1.897432e-05
    return z


def neuron_91(Q):
    z = 7.238619e-06
    return z


def neuron_92(Q):
    z = 8.794084e-06
    return z


def neuron_93(Q):
    z = -2.100003e-06
    return z


def neuron_94(Q):
    z = -2.384212e-05
    return z


def neuron_95(Q):
    z = -3.277345e-05
    return z


def neuron_96(Q):
    z = -4.324139e-05
    return z


def neuron_97(Q):
    z = -1.077089
    if Q.sum_z_dr < 0.1184723:
        z += 1.479981 * Q.sum_z_dr - 0.1296288
    if 0.1184723 <= Q.sum_z_dr < 0.1927916:
        z += -0.6150212 * Q.sum_z_dr + 0.1185709
    if Q.C3_b05 < 0.2250047:
        z += -2.143301 * Q.C3_b05 + 0.4822528
    if Q.dr_min_012 < 0.02139785:
        z += 4.803408 * Q.dr_min_012 - 0.1027826
    if Q.sj3_pair_mass_max >= 105.4845:
        z += 1.446319e-05 * Q.sj3_pair_mass_max - 0.001525642
    if Q.m01 >= 35.44963:
        z += 0.01408472 * Q.m01 - 0.4992981
    if Q.pt_entropy >= 1.63996:
        z += 0.4964127 * Q.pt_entropy - 0.8140969
    if Q.mass_top30 < 74.51927:
        z += -0.001195168 * Q.mass_top30 + 0.08906306
    if Q.n_pairs_kt_above_3 < 47.0:
        z += -0.007577362 * Q.n_pairs_kt_above_3 + 0.4754871
    if 47.0 <= Q.n_pairs_kt_above_3 < 220.0:
        z += -0.0006898907 * Q.n_pairs_kt_above_3 + 0.151776
    if Q.C3_b2 < 0.006832265:
        z += 15.52978 * Q.C3_b2 - 0.1061035
    if Q.n_pairs_kt_above_1 < 123.0:
        z += -0.008580462 * Q.n_pairs_kt_above_1 + 1.055397
    if Q.n_real_top50 < 44.0:
        z += 0.02023679 * Q.n_real_top50 - 0.8904188
    if Q.lnpt_1 < 4.9989:
        z += -0.103834 * Q.lnpt_1 + 0.5190558
    if Q.n_pairs_kt_above_10 < 10.0:
        z += -0.01902961 * Q.n_pairs_kt_above_10 + 0.07901442
    if 10.0 <= Q.n_pairs_kt_above_10 < 23.0:
        z += 0.008560126 * Q.n_pairs_kt_above_10 - 0.1968829
    if 123.7928 <= Q.mass < 164.4374:
        z += -0.005588321 * Q.mass + 0.6917936
    if 164.4374 <= Q.mass < 182.8592:
        z += -0.002159117 * Q.mass + 0.1279041
    if Q.mass >= 182.8592:
        z += -0.006665202 * Q.mass + 0.9518833
    if 106.7501 <= Q.sd_mass < 127.8042:
        z += -0.003831026 * Q.sd_mass + 0.4089623
    if 127.8042 <= Q.sd_mass < 154.5947:
        z += -0.008197576 * Q.sd_mass + 0.9670259
    if Q.sd_mass >= 154.5947:
        z += 0.01102348 * Q.sd_mass - 2.004446
    if Q.tau2 >= 0.04957212:
        z += 1.73832 * Q.tau2 - 0.08617223
    if Q.sj3_z1 < 0.68373:
        z += 0.151562 * Q.sj3_z1 - 0.1036275
    if Q.N2_b2 >= 0.255112:
        z += -1.736046 * Q.N2_b2 + 0.4428861
    if Q.dr_max_012 < 0.02210827:
        z += 11.00657 * Q.dr_max_012 - 0.2433363
    if Q.tau21_b2 < 0.07980588:
        z += 1.171156 * Q.tau21_b2 - 0.0934651
    if Q.pt2_over_pt0 < 0.2001251:
        z += -1.293738 * Q.pt2_over_pt0 + 0.2589095
    if Q.lund_max_lndelta >= -0.615738:
        z += -1.180737 * Q.lund_max_lndelta - 0.7270245
    if Q.z_top20_slots >= 0.9741142:
        z += -4.469115 * Q.z_top20_slots + 4.353428
    if Q.lund3_lndelta >= -2.4279:
        z += -0.04114842 * Q.lund3_lndelta - 0.09990422
    if Q.D2 < 2.692859:
        z += 0.1111524 * Q.D2 - 0.2993178
    z += 10.54064 * Q.M3
    if 44.18338 <= Q.mass_top5 < 68.52153:
        z += -0.005069187 * Q.mass_top5 + 0.2239738
    if Q.mass_top5 >= 68.52153:
        z += -0.004445797 * Q.mass_top5 + 0.1812582
    if 28.31178 <= Q.max_pair_mass < 55.24488:
        z += -0.01044749 * Q.max_pair_mass + 0.2957872
    if Q.max_pair_mass >= 55.24488:
        z += -0.01403694 * Q.max_pair_mass + 0.4940858
    if Q.mass_top15 < 70.93762:
        z += 0.004206151 * Q.mass_top15 - 0.2230982
    if 70.93762 <= Q.mass_top15 < 110.7667:
        z += -0.00188998 * Q.mass_top15 + 0.2093469
    if Q.C2_b2 < 0.02858957:
        z += -0.5956879 * Q.C2_b2 + 0.01703046
    if Q.pair_mean_lnkt >= 1.370103:
        z += 0.4503514 * Q.pair_mean_lnkt - 0.6170277
    if Q.n_dr_0p4_up < 12.0:
        z += -0.00334706 * Q.n_dr_0p4_up + 0.04016472
    if Q.mass_top40 < 78.33213:
        z += 0.01342241 * Q.mass_top40 - 1.051406
    if Q.m012 >= 32.7897:
        z += 0.01301788 * Q.m012 - 0.4268523
    if Q.pair_mean_lnz < -2.572052:
        z += -0.2377804 * Q.pair_mean_lnz - 0.6115835
    if Q.pair_mean_lnz >= -1.676789:
        z += -0.2853442 * Q.pair_mean_lnz - 0.4784621
    if Q.M2_b05 < 0.1338707:
        z += 1.09469 * Q.M2_b05 - 0.146547
    if Q.M3_b2 < 0.02497878:
        z += -14.36378 * Q.M3_b2 + 0.3587896
    if Q.n_dr_0p1_0p2 < 6.0:
        z += 0.004608278 * Q.n_dr_0p1_0p2 - 0.02764967
    if Q.n_lund >= 10.0:
        z += 0.004023925 * Q.n_lund - 0.04023925
    if Q.sj3_mass1 < 4.621465:
        z += -0.01894197 * Q.sj3_mass1 + 0.08753963
    if Q.D3_b2 < 0.4675349:
        z += 0.135655 * Q.D3_b2 - 0.06342344
    if Q.sum_z_dr2_top5 < 0.005508318:
        z += 43.82495 * Q.sum_z_dr2_top5 - 0.2414017
    if Q.N2_b05 < 0.3436326:
        z += 10.17685 * Q.N2_b05 - 3.497097
    if Q.sum_z_dr < 0.1184723 and Q.lnpt_0 > 4.993828:
        z += -5.191089 * (0.1184723 - Q.sum_z_dr) * (Q.lnpt_0 - 4.993828)
    if Q.n_pairs_kt_above_3 < 220.0 and Q.dr02 > 0.1620436:
        z += 0.004109733 * (220.0 - Q.n_pairs_kt_above_3) * (Q.dr02 - 0.1620436)
    if Q.n_pairs_kt_above_3 < 220.0 and Q.M3 > 0.02886227:
        z += 0.01337315 * (220.0 - Q.n_pairs_kt_above_3) * (Q.M3 - 0.02886227)
    if Q.sj3_pair_mass_max > 105.4845 and Q.sj3_dr_min < 0.3956483:
        z += 0.0054597 * (Q.sj3_pair_mass_max - 105.4845) * (0.3956483 - Q.sj3_dr_min)
    if Q.n_real_top50 < 44.0 and Q.n_lund > 7.0:
        z += 0.002233451 * (44.0 - Q.n_real_top50) * (Q.n_lund - 7.0)
    if Q.C3_b05 < 0.2250047 and Q.lund1_lndelta < -0.4705779:
        z += 1.104272 * (0.2250047 - Q.C3_b05) * (-0.4705779 - Q.lund1_lndelta)
    if Q.n_pairs_kt_above_1 < 123.0 and Q.n_pairs_kt_above_10 < 3.0:
        z += -0.003277282 * (123.0 - Q.n_pairs_kt_above_1) * (3.0 - Q.n_pairs_kt_above_10)
    if Q.lnpt_1 < 4.9989 and Q.tau21 > 0.1757731:
        z += -0.1863042 * (4.9989 - Q.lnpt_1) * (Q.tau21 - 0.1757731)
    if Q.n_pairs_kt_above_3 < 220.0 and Q.e4 < 7.184834e-06:
        z += -122.7128 * (220.0 - Q.n_pairs_kt_above_3) * (7.184834e-06 - Q.e4)
    if Q.m01 > 35.44963 and Q.dr12 < 0.4959595:
        z += -0.001435621 * (Q.m01 - 35.44963) * (0.4959595 - Q.dr12)
    if Q.m01 > 35.44963 and Q.D3_b2 < 1.462588:
        z += -0.00723716 * (Q.m01 - 35.44963) * (1.462588 - Q.D3_b2)
    if Q.pt_entropy > 1.63996 and Q.lund3_lndelta < -0.603253:
        z += 0.01430647 * (Q.pt_entropy - 1.63996) * (-0.603253 - Q.lund3_lndelta)
    if Q.pt_entropy > 1.63996 and Q.n_lund_kt_above_1 < 6.0:
        z += -0.01910607 * (Q.pt_entropy - 1.63996) * (6.0 - Q.n_lund_kt_above_1)
    if Q.n_pairs_kt_above_1 < 123.0 and Q.n_lund_kt_above_5 > 1.0:
        z += -0.002330798 * (123.0 - Q.n_pairs_kt_above_1) * (Q.n_lund_kt_above_5 - 1.0)
    if Q.sj3_pair_mass_max > 105.4845 and Q.dr_31 < 0.1500144:
        z += -0.01179683 * (Q.sj3_pair_mass_max - 105.4845) * (0.1500144 - Q.dr_31)
    if Q.n_pairs_kt_above_3 < 220.0 and Q.dr_2 > 0.04584392:
        z += -0.004874406 * (220.0 - Q.n_pairs_kt_above_3) * (Q.dr_2 - 0.04584392)
    if Q.n_pairs_kt_above_10 < 23.0 and Q.mass_top5 > 34.01347:
        z += 0.000413949 * (23.0 - Q.n_pairs_kt_above_10) * (Q.mass_top5 - 34.01347)
    if Q.sd_mass > 127.8042 and Q.sj3_mass2 < 1.824785:
        z += -0.00937607 * (Q.sd_mass - 127.8042) * (1.824785 - Q.sj3_mass2)
    if Q.lnpt_1 < 4.9989 and Q.lund_max_lndelta < -0.752531:
        z += -0.03934408 * (4.9989 - Q.lnpt_1) * (-0.752531 - Q.lund_max_lndelta)
    if Q.pt_entropy > 1.63996 and Q.n_lund_kt_above_5 > 1.0:
        z += -0.007038287 * (Q.pt_entropy - 1.63996) * (Q.n_lund_kt_above_5 - 1.0)
    if Q.C3_b05 < 0.2250047 and Q.psi_0p3 < 0.8629612:
        z += -1.520354 * (0.2250047 - Q.C3_b05) * (0.8629612 - Q.psi_0p3)
    if Q.pt2_over_pt0 < 0.2001251 and Q.z_dr_0p2_0p4 > 0.1724455:
        z += -3.227846 * (0.2001251 - Q.pt2_over_pt0) * (Q.z_dr_0p2_0p4 - 0.1724455)
    if Q.lnpt_1 < 4.9989 and Q.e3_b2 > 1.78923e-05:
        z += -19.74489 * (4.9989 - Q.lnpt_1) * (Q.e3_b2 - 1.78923e-05)
    if Q.lnpt_1 < 4.9989 and Q.z_dr_0p1_0p2 > 0.4056731:
        z += 0.287208 * (4.9989 - Q.lnpt_1) * (Q.z_dr_0p1_0p2 - 0.4056731)
    if Q.sum_z_dr < 0.1184723 and Q.lnerel_4 < -2.471159:
        z += -0.5221437 * (0.1184723 - Q.sum_z_dr) * (-2.471159 - Q.lnerel_4)
    if Q.lnpt_1 < 4.9989 and Q.dr_19 > 0.1712001:
        z += -0.03456642 * (4.9989 - Q.lnpt_1) * (Q.dr_19 - 0.1712001)
    if Q.mass > 123.7928 and Q.sj3_mass2 < 1.824785:
        z += 0.00640401 * (Q.mass - 123.7928) * (1.824785 - Q.sj3_mass2)
    if Q.mass_top5 > 44.18338 and Q.tau32_b2 < 0.4680886:
        z += 0.01024858 * (Q.mass_top5 - 44.18338) * (0.4680886 - Q.tau32_b2)
    if Q.dr_min_012 < 0.02139785 and Q.dr_47 < 0.09477716:
        z += -10.22846 * (0.02139785 - Q.dr_min_012) * (0.09477716 - Q.dr_47)
    if Q.D2 < 2.692859 and Q.tau43_b2 < 0.5796538:
        z += 0.2658138 * (2.692859 - Q.D2) * (0.5796538 - Q.tau43_b2)
    if Q.C2_b2 < 0.02858957 and Q.eta_31 > 0.0:
        z += 1.385232 * (0.02858957 - Q.C2_b2) * (Q.eta_31 - 0.0)
    if Q.C3_b2 < 0.006832265 and Q.mean_phi2 < 0.00859211:
        z += -2318.906 * (0.006832265 - Q.C3_b2) * (0.00859211 - Q.mean_phi2)
    if Q.n_real_top50 < 44.0 and Q.dr_14 < 0.2250629:
        z += -0.009712423 * (44.0 - Q.n_real_top50) * (0.2250629 - Q.dr_14)
    if Q.D2 < 2.692859 and Q.sum_e > 1247.266:
        z += 0.0001109544 * (2.692859 - Q.D2) * (Q.sum_e - 1247.266)
    if Q.max_pair_mass > 55.24488 and Q.min_pair_mass < 2.102317:
        z += -0.006332127 * (Q.max_pair_mass - 55.24488) * (2.102317 - Q.min_pair_mass)
    if Q.n_pairs_kt_above_1 < 123.0 and Q.log_sum_pt < 6.310762:
        z += 0.006578626 * (123.0 - Q.n_pairs_kt_above_1) * (6.310762 - Q.log_sum_pt)
    if Q.mass_top40 < 78.33213 and Q.min_pair_mass < 2.102317:
        z += -0.0007786124 * (78.33213 - Q.mass_top40) * (2.102317 - Q.min_pair_mass)
    if Q.n_pairs_kt_above_3 < 220.0 and Q.min_pair_mass > 2.606496:
        z += -5.307493e-05 * (220.0 - Q.n_pairs_kt_above_3) * (Q.min_pair_mass - 2.606496)
    if Q.m012 > 32.7897 and Q.D2_b2 < 2.01745:
        z += -0.002962106 * (Q.m012 - 32.7897) * (2.01745 - Q.D2_b2)
    if Q.n_pairs_kt_above_3 < 47.0 and Q.planar_flow < 0.6052136:
        z += 0.02097255 * (47.0 - Q.n_pairs_kt_above_3) * (0.6052136 - Q.planar_flow)
    if Q.pt2_over_pt0 < 0.2001251 and Q.C3_b2 > 0.04229114:
        z += -10.06291 * (0.2001251 - Q.pt2_over_pt0) * (Q.C3_b2 - 0.04229114)
    if Q.mass_top15 < 110.7667 and Q.pt2_over_pt0 < 0.2001251:
        z += -0.06014323 * (110.7667 - Q.mass_top15) * (0.2001251 - Q.pt2_over_pt0)
    if Q.m012 > 32.7897 and Q.dr_1 < 0.3209146:
        z += 0.02504891 * (Q.m012 - 32.7897) * (0.3209146 - Q.dr_1)
    if Q.sd_mass > 154.5947 and Q.pt2_over_pt0 < 0.2412858:
        z += -0.05810595 * (Q.sd_mass - 154.5947) * (0.2412858 - Q.pt2_over_pt0)
    if Q.mass_top30 < 74.51927 and Q.pt2_over_pt0 < 0.09605749:
        z += 0.01233273 * (74.51927 - Q.mass_top30) * (0.09605749 - Q.pt2_over_pt0)
    if Q.M3_b2 < 0.02497878 and Q.pt2_over_pt0 < 0.3179541:
        z += 53.69611 * (0.02497878 - Q.M3_b2) * (0.3179541 - Q.pt2_over_pt0)
    return z


def neuron_98(Q):
    z = 1.067419e-06
    return z


def neuron_99(Q):
    z = 1.580472e-05
    return z


def neuron_100(Q):
    z = 2.544017e-05
    return z


def neuron_101(Q):
    z = 3.024446e-05
    return z


def neuron_102(Q):
    z = 0.1241446
    if Q.tau2 < 0.1526454:
        z += 0.9177745 * Q.tau2 - 0.140094
    if Q.lnpt_0 >= 4.46447:
        z += -0.05059738 * Q.lnpt_0 + 0.2258905
    if Q.M2 < 0.03465331:
        z += -17.27383 * Q.M2 + 0.5985955
    if Q.n_dr_0p4_up < 4.0:
        z += 0.03004411 * Q.n_dr_0p4_up - 0.1201764
    if Q.sj4_pair_mass_max < 48.76729:
        z += 0.0002205418 * Q.sj4_pair_mass_max - 0.1586001
    if 48.76729 <= Q.sj4_pair_mass_max < 55.06039:
        z += -0.008790523 * Q.sj4_pair_mass_max + 0.2808451
    if 55.06039 <= Q.sj4_pair_mass_max < 96.97763:
        z += 0.004846799 * Q.sj4_pair_mass_max - 0.4700311
    if Q.mass < 71.96396:
        z += -0.008794571 * Q.mass + 0.8882347
    if 71.96396 <= Q.mass < 79.47361:
        z += 0.004345533 * Q.mass - 0.05737931
    if 79.47361 <= Q.mass < 90.08945:
        z += -0.003413208 * Q.mass + 0.5592358
    if 90.08945 <= Q.mass < 95.14961:
        z += -0.00473603 * Q.mass + 0.6784082
    if 95.14961 <= Q.mass < 110.2019:
        z += -0.002166858 * Q.mass + 0.4339524
    if 110.2019 <= Q.mass < 131.3917:
        z += -0.006281897 * Q.mass + 0.8874375
    if 131.3917 <= Q.mass < 149.0507:
        z += -0.003513701 * Q.mass + 0.5237195
    if Q.sj3_pair_mass_max < 56.73313:
        z += 0.001170888 * Q.sj3_pair_mass_max + 0.07491978
    if 56.73313 <= Q.sj3_pair_mass_max < 73.24742:
        z += -0.01307922 * Q.sj3_pair_mass_max + 0.8833732
    if 73.24742 <= Q.sj3_pair_mass_max < 76.91486:
        z += -0.01196052 * Q.sj3_pair_mass_max + 0.8014311
    if 76.91486 <= Q.sj3_pair_mass_max < 97.52633:
        z += 0.005749743 * Q.sj3_pair_mass_max - 0.5607513
    if Q.dr_32 < 0.1148075:
        z += 0.250387 * Q.dr_32 - 0.02874631
    if Q.n_pairs_kt_above_3 < 34.0:
        z += 0.002123885 * Q.n_pairs_kt_above_3 + 0.05233834
    if 34.0 <= Q.n_pairs_kt_above_3 < 41.0:
        z += -0.006884167 * Q.n_pairs_kt_above_3 + 0.3586121
    if 41.0 <= Q.n_pairs_kt_above_3 < 99.0:
        z += -0.001316573 * Q.n_pairs_kt_above_3 + 0.1303408
    if Q.sj3_mass3 < 0.3886647:
        z += -0.2176866 * Q.sj3_mass3 + 0.112997
    if 0.3886647 <= Q.sj3_mass3 < 11.07803:
        z += -0.002655901 * Q.sj3_mass3 + 0.02942216
    if Q.tau21_b2 < 0.1242239:
        z += -0.8188257 * Q.tau21_b2 + 0.1017177
    if Q.sj2_mass1 < 33.62116:
        z += 0.002671336 * Q.sj2_mass1 - 0.08981341
    if Q.ecf_g31 < 0.00391037:
        z += -27.19572 * Q.ecf_g31 + 0.1063453
    if Q.lund_max_lnkt >= 2.961008:
        z += -0.01619929 * Q.lund_max_lnkt + 0.04796624
    if Q.tau21 < 0.2949917:
        z += 0.3683573 * Q.tau21 - 0.1086623
    if Q.dr_1 >= 0.1975394:
        z += -0.1699871 * Q.dr_1 + 0.03357915
    if Q.M2_b05 < 0.0815014:
        z += 0.6551895 * Q.M2_b05 + 0.2336224
    if 0.0815014 <= Q.M2_b05 < 0.1209237:
        z += -7.28069 * Q.M2_b05 + 0.8804077
    if Q.lnptrel_0 >= -1.266303:
        z += -0.1824685 * Q.lnptrel_0 - 0.2310603
    if Q.n_pairs_kt_above_1 < 101.0:
        z += 0.001733775 * Q.n_pairs_kt_above_1 - 0.3336974
    if 101.0 <= Q.n_pairs_kt_above_1 < 325.0:
        z += 0.0007079742 * Q.n_pairs_kt_above_1 - 0.2300916
    if Q.n_pt_above_5 >= 14.0:
        z += -0.00349161 * Q.n_pt_above_5 + 0.04888254
    if Q.sj3_pair_mass_min >= 59.49644:
        z += 0.001978209 * Q.sj3_pair_mass_min - 0.1176964
    if Q.tau32 < 0.5068038:
        z += 0.04838077 * Q.tau32 - 0.02451956
    if Q.sj3_dr23 < 0.2109554:
        z += -0.2358523 * Q.sj3_dr23 + 0.04975431
    if Q.mratio_min_012 < 0.01744666:
        z += -5.302689 * Q.mratio_min_012 + 0.09251419
    if Q.sd_mass < 62.03827:
        z += 0.0004513572 * Q.sd_mass - 0.07258946
    if 62.03827 <= Q.sd_mass < 78.4753:
        z += -0.00521817 * Q.sd_mass + 0.2791382
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += 0.01943233 * Q.sd_mass - 1.655317
    if 88.81751 <= Q.sd_mass < 94.55722:
        z += 0.009637409 * Q.sd_mass - 0.7853566
    if 94.55722 <= Q.sd_mass < 106.7501:
        z += -0.009637911 * Q.sd_mass + 1.037264
    if 106.7501 <= Q.sd_mass < 115.7091:
        z += -0.002778325 * Q.sd_mass + 0.3050028
    if 115.7091 <= Q.sd_mass < 127.8042:
        z += 0.001362092 * Q.sd_mass - 0.1740811
    if Q.pt1_dr01 >= 4.704996:
        z += 0.004423136 * Q.pt1_dr01 - 0.02081084
    if Q.pair_mean_lndelta >= -1.352792:
        z += 0.1854254 * Q.pair_mean_lndelta + 0.250842
    if Q.z_dr_0p05_0p1 < 0.008254376:
        z += -12.37992 * Q.z_dr_0p05_0p1 + 0.1021885
    if Q.pair_mean_lnm2 >= 4.069088:
        z += 0.004848838 * Q.pair_mean_lnm2 - 0.01973035
    if Q.dr01 >= 0.2849189:
        z += -0.390044 * Q.dr01 + 0.1111309
    if Q.sj3_dr_min < 0.2410564:
        z += -0.2032371 * Q.sj3_dr_min + 0.0489916
    if Q.lnpt_22 >= 1.005903:
        z += -0.08965036 * Q.lnpt_22 + 0.09017956
    if Q.tau2 < 0.1526454 and Q.sj3_pairmax_over_m < 0.8351741:
        z += 0.8805165 * (0.1526454 - Q.tau2) * (0.8351741 - Q.sj3_pairmax_over_m)
    if Q.n_dr_0p4_up < 4.0 and Q.min_pair_mass > 10.59762:
        z += -0.00344682 * (4.0 - Q.n_dr_0p4_up) * (Q.min_pair_mass - 10.59762)
    if Q.tau2 < 0.1526454 and Q.lnerel_67 > -18.42068:
        z += 0.1311731 * (0.1526454 - Q.tau2) * (Q.lnerel_67 - -18.42068)
    if Q.lnpt_0 > 4.46447 and Q.dr01 > 0.3258728:
        z += -0.5635661 * (Q.lnpt_0 - 4.46447) * (Q.dr01 - 0.3258728)
    if Q.tau21_b2 < 0.1242239 and Q.sj3_dr_max < 0.4372584:
        z += 12.42012 * (0.1242239 - Q.tau21_b2) * (0.4372584 - Q.sj3_dr_max)
    if Q.sj3_mass3 < 11.07803 and Q.n_dr_0p05_0p1 < 4.0:
        z += -0.002086969 * (11.07803 - Q.sj3_mass3) * (4.0 - Q.n_dr_0p05_0p1)
    if Q.mass < 110.2019 and Q.n_dr_0p2_0p4 > 11.0:
        z += 0.001164322 * (110.2019 - Q.mass) * (Q.n_dr_0p2_0p4 - 11.0)
    if Q.sj3_pair_mass_max < 97.52633 and Q.M3_b2 < 0.01312409:
        z += 0.3870551 * (97.52633 - Q.sj3_pair_mass_max) * (0.01312409 - Q.M3_b2)
    if Q.mass < 90.08945 and Q.z_2nd < 0.1783885:
        z += -0.07538031 * (90.08945 - Q.mass) * (0.1783885 - Q.z_2nd)
    if Q.n_pairs_kt_above_3 < 99.0 and Q.eccentricity > 0.8461094:
        z += 0.001962767 * (99.0 - Q.n_pairs_kt_above_3) * (Q.eccentricity - 0.8461094)
    if Q.sj3_pair_mass_max < 97.52633 and Q.lne_1 < 4.855929:
        z += 0.001954296 * (97.52633 - Q.sj3_pair_mass_max) * (4.855929 - Q.lne_1)
    if Q.M2 < 0.03465331 and Q.n_dr_0p05_0p1 > 16.0:
        z += -50.49947 * (0.03465331 - Q.M2) * (Q.n_dr_0p05_0p1 - 16.0)
    if Q.lnpt_0 > 4.46447 and Q.n_dr_0p05_0p1 < 4.0:
        z += 0.002430473 * (Q.lnpt_0 - 4.46447) * (4.0 - Q.n_dr_0p05_0p1)
    if Q.tau21_b2 < 0.1242239 and Q.n_pt_above_10 > 13.0:
        z += -0.09290335 * (0.1242239 - Q.tau21_b2) * (Q.n_pt_above_10 - 13.0)
    if Q.mass < 79.47361 and Q.n_pt_above_10 < 14.0:
        z += -0.0006734077 * (79.47361 - Q.mass) * (14.0 - Q.n_pt_above_10)
    if Q.dr_32 < 0.1148075 and Q.n_pt_above_10 < 13.0:
        z += -0.02083016 * (0.1148075 - Q.dr_32) * (13.0 - Q.n_pt_above_10)
    if Q.mass < 110.2019 and Q.D2 < 0.7107791:
        z += -0.02232998 * (110.2019 - Q.mass) * (0.7107791 - Q.D2)
    if Q.sj3_pair_mass_max < 76.91486 and Q.tau32_b2 < 0.6550361:
        z += 0.002174589 * (76.91486 - Q.sj3_pair_mass_max) * (0.6550361 - Q.tau32_b2)
    if Q.mass < 90.08945 and Q.lund2_lnz > -4.435851:
        z += -0.0006382926 * (90.08945 - Q.mass) * (Q.lund2_lnz - -4.435851)
    if Q.tau32 < 0.5068038 and Q.sj3_dr13 > 0.7462286:
        z += 0.8149401 * (0.5068038 - Q.tau32) * (Q.sj3_dr13 - 0.7462286)
    if Q.sj3_mass3 < 0.3886647 and Q.sj4_pair_mass_min > 4.004863:
        z += 0.001268669 * (0.3886647 - Q.sj3_mass3) * (Q.sj4_pair_mass_min - 4.004863)
    if Q.sj3_mass3 < 11.07803 and Q.sj3_dr13 > 0.4088773:
        z += -0.01572369 * (11.07803 - Q.sj3_mass3) * (Q.sj3_dr13 - 0.4088773)
    if Q.tau21_b2 < 0.1242239 and Q.eta_26 < -0.3178711:
        z += -2.922781 * (0.1242239 - Q.tau21_b2) * (-0.3178711 - Q.eta_26)
    if Q.sj3_mass3 < 0.3886647 and Q.tau32 < 0.731235:
        z += -0.3781875 * (0.3886647 - Q.sj3_mass3) * (0.731235 - Q.tau32)
    if Q.lund_max_lnkt > 2.961008 and Q.tau32 > 0.731235:
        z += 0.1706557 * (Q.lund_max_lnkt - 2.961008) * (Q.tau32 - 0.731235)
    if Q.sd_mass < 94.55722 and Q.pt1_dr01 > 2.525555:
        z += -3.496066e-06 * (94.55722 - Q.sd_mass) * (Q.pt1_dr01 - 2.525555)
    if Q.tau32 < 0.5068038 and Q.pt1_dr01 > 29.53862:
        z += -0.03086402 * (0.5068038 - Q.tau32) * (Q.pt1_dr01 - 29.53862)
    if Q.tau2 < 0.1526454 and Q.sj4_zsoft > 0.03350185:
        z += -0.3326843 * (0.1526454 - Q.tau2) * (Q.sj4_zsoft - 0.03350185)
    if Q.tau2 < 0.1526454 and Q.dr_9 < 0.4951794:
        z += 0.6138631 * (0.1526454 - Q.tau2) * (0.4951794 - Q.dr_9)
    if Q.sj4_pair_mass_max < 48.76729 and Q.z_dr_0p05_0p1 > 0.4355562:
        z += -0.03771665 * (48.76729 - Q.sj4_pair_mass_max) * (Q.z_dr_0p05_0p1 - 0.4355562)
    if Q.sj3_dr_min < 0.2410564 and Q.lnerel_27 < -6.585249:
        z += -0.001991095 * (0.2410564 - Q.sj3_dr_min) * (-6.585249 - Q.lnerel_27)
    return z


def neuron_103(Q):
    z = 1.613216
    if Q.pair_mean_lndelta >= -2.336539:
        z += 0.4199489 * Q.pair_mean_lndelta + 0.9812271
    if Q.C2_b2 < 0.09209404:
        z += -5.649959 * Q.C2_b2 + 0.5203276
    if Q.sj3_pair_mass_min < 68.11898:
        z += 0.003503728 * Q.sj3_pair_mass_min - 0.09824765
    if 68.11898 <= Q.sj3_pair_mass_min < 80.02563:
        z += -0.01179364 * Q.sj3_pair_mass_min + 0.9437938
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.009057947 * Q.n_pairs_kt_above_1 + 1.098141
    if 80.0 <= Q.n_pairs_kt_above_1 < 223.0:
        z += -0.002611928 * Q.n_pairs_kt_above_1 + 0.58246
    if Q.mass_top50 < 161.1264:
        z += 0.007021789 * Q.mass_top50 - 1.131396
    if Q.n_pairs_kt_above_3 < 34.0:
        z += 0.01354958 * Q.n_pairs_kt_above_3 - 0.4683449
    if 34.0 <= Q.n_pairs_kt_above_3 < 47.0:
        z += 0.000589173 * Q.n_pairs_kt_above_3 - 0.02769113
    if 54.0 <= Q.n_pairs_kt_above_3 < 144.0:
        z += -0.003504898 * Q.n_pairs_kt_above_3 + 0.1892645
    if Q.n_pairs_kt_above_3 >= 144.0:
        z += 3.546011e-05 * Q.n_pairs_kt_above_3 - 0.3205471
    if Q.mass_top40 < 92.68149:
        z += -0.01303614 * Q.mass_top40 + 1.155712
    if 92.68149 <= Q.mass_top40 < 97.12186:
        z += 0.01182279 * Q.mass_top40 - 1.148251
    if Q.tau2 < 0.09733903:
        z += 8.615976 * Q.tau2 - 0.8386708
    if Q.sj3_mass2 < 3.298199:
        z += 0.05001819 * Q.sj3_mass2 - 0.1649699
    if Q.n_for_90pct < 11.0:
        z += 0.01884136 * Q.n_for_90pct - 0.207255
    if Q.C2_b05 >= 0.3007515:
        z += -0.3432381 * Q.C2_b05 + 0.1032294
    if Q.N2_b2 < 0.2243273:
        z += 3.801536 * Q.N2_b2 - 0.8527885
    if Q.N2_b2 >= 0.255112:
        z += 0.8901266 * Q.N2_b2 - 0.227082
    if Q.z_dr_0p4_up < 0.03469679:
        z += 6.170343 * Q.z_dr_0p4_up - 0.2140911
    if Q.sj4_pair_mass_max < 96.97763:
        z += -0.002960976 * Q.sj4_pair_mass_max + 0.2871484
    if Q.dr02 < 0.07373689:
        z += -1.10193 * Q.dr02 - 0.173345
    if 0.07373689 <= Q.dr02 < 0.2159556:
        z += 1.790186 * Q.dr02 - 0.3866006
    if Q.dr_2 < 0.1702808:
        z += -0.8778979 * Q.dr_2 + 0.1494892
    if Q.max_dr >= 0.9206713:
        z += -0.06612077 * Q.max_dr + 0.06087549
    if Q.psi_0p1 < 0.01545532:
        z += -12.51574 * Q.psi_0p1 + 0.1934347
    if Q.psi_0p1 >= 0.7386202:
        z += -1.452492 * Q.psi_0p1 + 1.07284
    if Q.pair_mean_lnkt >= 1.030977:
        z += -0.3496478 * Q.pair_mean_lnkt + 0.3604788
    if Q.n_dr_0p1_0p2 < 4.0:
        z += -0.08494837 * Q.n_dr_0p1_0p2 + 0.3397935
    if Q.m01 >= 8.773939:
        z += 0.004891391 * Q.m01 - 0.04291677
    if Q.N2_b05 < 0.3436326:
        z += -8.693158 * Q.N2_b05 + 2.987253
    if Q.dr12 < 0.01937651:
        z += 5.077054 * Q.dr12 - 0.09837556
    z += -0.101592 * Q.lne_1
    if Q.psi_0p3 >= 0.9925964:
        z += -34.87788 * Q.psi_0p3 + 34.61966
    if Q.tau5 < 0.05613495:
        z += -20.25585 * Q.tau5 + 1.137061
    if Q.mass < 164.4374:
        z += 0.01156338 * Q.mass - 1.901452
    if Q.lnptrel_29 >= -5.376631:
        z += -0.2226938 * Q.lnptrel_29 - 1.197342
    if Q.M2_b05 < 0.1573433:
        z += 4.568944 * Q.M2_b05 - 0.7188927
    if Q.e3 < 0.00228569:
        z += 230.5745 * Q.e3 - 0.5270219
    if Q.z_top5 >= 0.5239396:
        z += -1.432238 * Q.z_top5 + 0.7504061
    if Q.n_dr_0p4_up < 5.0:
        z += 0.04942337 * Q.n_dr_0p4_up - 0.2471169
    if Q.pair_mean_lnm2 < 1.488114:
        z += 0.4136071 * Q.pair_mean_lnm2 - 0.6154943
    if Q.dr_0 < 0.1024943:
        z += -1.506612 * Q.dr_0 + 0.1544192
    if Q.mass_top30 < 121.0623:
        z += -0.001647902 * Q.mass_top30 + 0.1994988
    if Q.D3_b2 < 0.001184159:
        z += 60.20024 * Q.D3_b2 - 0.07128668
    if Q.sj3_pairmax_over_m >= 0.8809196:
        z += -2.598043 * Q.sj3_pairmax_over_m + 2.288667
    if Q.mass_top5 < 29.68311:
        z += 0.008564137 * Q.mass_top5 - 0.2542102
    if Q.mass_top5 >= 58.28268:
        z += 0.00136399 * Q.mass_top5 - 0.079497
    if Q.sj3_dr_min < 0.1798005:
        z += 1.311015 * Q.sj3_dr_min - 0.2357212
    if -1.388411 <= Q.lund_max_lndelta < -0.752531:
        z += -0.2077103 * Q.lund_max_lndelta - 0.2883873
    if Q.lund_max_lndelta >= -0.752531:
        z += 0.4714536 * Q.lund_max_lndelta + 0.2227046
    if Q.sj4_dr_min < 0.05865627:
        z += -3.419932 * Q.sj4_dr_min + 0.2006005
    if 123.2919 <= Q.sd_mass < 154.5947:
        z += 0.01438004 * Q.sd_mass - 1.772942
    if 154.5947 <= Q.sd_mass < 175.9333:
        z += -0.01093204 * Q.sd_mass + 2.14017
    if Q.sd_mass >= 175.9333:
        z += 0.001164801 * Q.sd_mass + 0.01193286
    if Q.C3_b2 < 0.04229114:
        z += 2.525733 * Q.C3_b2 - 0.1068161
    if Q.D3_b05 < 0.8860453:
        z += 0.0866688 * Q.D3_b05 - 0.07679248
    if Q.jet_abs_eta >= 1.465946:
        z += -0.4456629 * Q.jet_abs_eta + 0.6533178
    if Q.C2_b2 < 0.09209404 and Q.mass_top5 > 62.84493:
        z += -0.07795643 * (0.09209404 - Q.C2_b2) * (Q.mass_top5 - 62.84493)
    if Q.mass_top50 < 161.1264 and Q.M2 < 0.120439:
        z += -0.006201266 * (161.1264 - Q.mass_top50) * (0.120439 - Q.M2)
    if Q.n_pairs_kt_above_3 < 47.0 and Q.dr01 < 0.4974172:
        z += 0.03617479 * (47.0 - Q.n_pairs_kt_above_3) * (0.4974172 - Q.dr01)
    if Q.N2_b2 < 0.2243273 and Q.lne_1 > 4.137663:
        z += -0.8934941 * (0.2243273 - Q.N2_b2) * (Q.lne_1 - 4.137663)
    if Q.n_pairs_kt_above_3 < 47.0 and Q.pair_max_lnkt > 1.555555:
        z += 0.00915309 * (47.0 - Q.n_pairs_kt_above_3) * (Q.pair_max_lnkt - 1.555555)
    if Q.N2_b2 < 0.2243273 and Q.n_lund > 9.0:
        z += 0.203831 * (0.2243273 - Q.N2_b2) * (Q.n_lund - 9.0)
    if Q.tau2 < 0.09733903 and Q.n_pairs_kt_above_10 < 11.0:
        z += 0.4070318 * (0.09733903 - Q.tau2) * (11.0 - Q.n_pairs_kt_above_10)
    if Q.dr02 < 0.2159556 and Q.ecf_g41 > 6.216954e-05:
        z += 869.8306 * (0.2159556 - Q.dr02) * (Q.ecf_g41 - 6.216954e-05)
    if Q.sj3_pair_mass_min < 80.02563 and Q.dr_1 < 0.1975394:
        z += 0.01736205 * (80.02563 - Q.sj3_pair_mass_min) * (0.1975394 - Q.dr_1)
    if Q.dr02 < 0.2159556 and Q.sj3_dr13 > 0.7462286:
        z += -2.151929 * (0.2159556 - Q.dr02) * (Q.sj3_dr13 - 0.7462286)
    if Q.m01 > 8.773939 and Q.dr_6 < 0.2567354:
        z += 0.002858757 * (Q.m01 - 8.773939) * (0.2567354 - Q.dr_6)
    if Q.N2_b2 < 0.2243273 and Q.dr_38 < 0.1455861:
        z += -2.154104 * (0.2243273 - Q.N2_b2) * (0.1455861 - Q.dr_38)
    if Q.mass_top50 < 161.1264 and Q.sj3_dr23 > 0.3971678:
        z += 0.002071171 * (161.1264 - Q.mass_top50) * (Q.sj3_dr23 - 0.3971678)
    if Q.M2_b05 < 0.1573433 and Q.n_pairs_kt_above_10 > 3.0:
        z += -0.2488096 * (0.1573433 - Q.M2_b05) * (Q.n_pairs_kt_above_10 - 3.0)
    if Q.N2_b2 < 0.2243273 and Q.centroid_offset > 0.007708221:
        z += 17.97575 * (0.2243273 - Q.N2_b2) * (Q.centroid_offset - 0.007708221)
    if Q.pair_mean_lndelta > -2.336539 and Q.pt2_over_pt0 > 0.1526736:
        z += -0.4999177 * (Q.pair_mean_lndelta - -2.336539) * (Q.pt2_over_pt0 - 0.1526736)
    if Q.n_pairs_kt_above_3 > 54.0 and Q.C2_b2 < 0.1204509:
        z += 0.005859416 * (Q.n_pairs_kt_above_3 - 54.0) * (0.1204509 - Q.C2_b2)
    if Q.tau5 < 0.05613495 and Q.lam2 > 0.00972048:
        z += -403.1286 * (0.05613495 - Q.tau5) * (Q.lam2 - 0.00972048)
    if Q.z_dr_0p4_up < 0.03469679 and Q.C2_b2 < 0.1049877:
        z += 58.24279 * (0.03469679 - Q.z_dr_0p4_up) * (0.1049877 - Q.C2_b2)
    if Q.z_dr_0p4_up < 0.03469679 and Q.D2 > 0.7107791:
        z += 1.585781 * (0.03469679 - Q.z_dr_0p4_up) * (Q.D2 - 0.7107791)
    if Q.m01 > 8.773939 and Q.C2_b2 < 0.02349873:
        z += -0.3128991 * (Q.m01 - 8.773939) * (0.02349873 - Q.C2_b2)
    if Q.dr02 < 0.2159556 and Q.pt1_over_pt0 > 0.2477181:
        z += 1.139824 * (0.2159556 - Q.dr02) * (Q.pt1_over_pt0 - 0.2477181)
    if Q.N2_b2 < 0.2243273 and Q.ecf_g42 < 1.650776e-05:
        z += 40328.51 * (0.2243273 - Q.N2_b2) * (1.650776e-05 - Q.ecf_g42)
    if Q.m01 > 8.773939 and Q.dr_1 < 0.2586201:
        z += 0.04876021 * (Q.m01 - 8.773939) * (0.2586201 - Q.dr_1)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.ecf_g42 > 2.701529e-06:
        z += -476.5245 * (34.0 - Q.n_pairs_kt_above_3) * (Q.ecf_g42 - 2.701529e-06)
    if Q.mass < 164.4374 and Q.C2_b2 < 0.1204509:
        z += -0.1360644 * (164.4374 - Q.mass) * (0.1204509 - Q.C2_b2)
    if Q.mass_top40 < 97.12186 and Q.C2_b2 < 0.1049877:
        z += 0.1881197 * (97.12186 - Q.mass_top40) * (0.1049877 - Q.C2_b2)
    if Q.z_dr_0p4_up < 0.03469679 and Q.C2_b2 > 0.06190784:
        z += -29.92914 * (0.03469679 - Q.z_dr_0p4_up) * (Q.C2_b2 - 0.06190784)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.ecf_g41 > 8.500146e-05:
        z += 25.15254 * (34.0 - Q.n_pairs_kt_above_3) * (Q.ecf_g41 - 8.500146e-05)
    if Q.dr02 < 0.07373689 and Q.lnpt_4 < 4.026467:
        z += 0.452547 * (0.07373689 - Q.dr02) * (4.026467 - Q.lnpt_4)
    if Q.dr_2 < 0.1702808 and Q.sj3_mass3 < 13.10191:
        z += 0.06289256 * (0.1702808 - Q.dr_2) * (13.10191 - Q.sj3_mass3)
    if Q.sj3_pair_mass_min < 80.02563 and Q.dr_4 > 0.06996917:
        z += -0.001612505 * (80.02563 - Q.sj3_pair_mass_min) * (Q.dr_4 - 0.06996917)
    if Q.sj3_mass2 < 3.298199 and Q.sj3_dr13 < 0.2317874:
        z += 0.3750646 * (3.298199 - Q.sj3_mass2) * (0.2317874 - Q.sj3_dr13)
    if Q.n_pairs_kt_above_3 > 144.0 and Q.tau43 > 0.8357534:
        z += -0.01303435 * (Q.n_pairs_kt_above_3 - 144.0) * (Q.tau43 - 0.8357534)
    if Q.sd_mass > 123.2919 and Q.e4 < 1.373172e-06:
        z += 2555.313 * (Q.sd_mass - 123.2919) * (1.373172e-06 - Q.e4)
    if Q.sd_mass > 175.9333 and Q.tau54 > 0.6772033:
        z += -0.01864151 * (Q.sd_mass - 175.9333) * (Q.tau54 - 0.6772033)
    if Q.dr02 < 0.2159556 and Q.dr_3 > 0.02154917:
        z += -0.3824949 * (0.2159556 - Q.dr02) * (Q.dr_3 - 0.02154917)
    if Q.mass_top5 < 29.68311 and Q.dr_34 < 0.09246155:
        z += -0.06898946 * (29.68311 - Q.mass_top5) * (0.09246155 - Q.dr_34)
    if Q.M2_b05 < 0.1573433 and Q.dr_25 < 0.2977652:
        z += -0.4566556 * (0.1573433 - Q.M2_b05) * (0.2977652 - Q.dr_25)
    if Q.C3_b2 < 0.04229114 and Q.ecf_g42 > 1.874473e-05:
        z += -1310.555 * (0.04229114 - Q.C3_b2) * (Q.ecf_g42 - 1.874473e-05)
    if Q.z_dr_0p4_up < 0.03469679 and Q.lund2_lndelta > -0.8575264:
        z += -11.26309 * (0.03469679 - Q.z_dr_0p4_up) * (Q.lund2_lndelta - -0.8575264)
    if Q.n_dr_0p4_up < 5.0 and Q.lund2_lnz < -3.197773:
        z += 0.005768928 * (5.0 - Q.n_dr_0p4_up) * (-3.197773 - Q.lund2_lnz)
    return z


def neuron_104(Q):
    z = -0.6068381
    if Q.pair_mean_lnm2 < 2.340672:
        z += 0.1751371 * Q.pair_mean_lnm2 - 0.4517006
    if 2.340672 <= Q.pair_mean_lnm2 < 4.069088:
        z += 0.02416208 * Q.pair_mean_lnm2 - 0.09831765
    if Q.tau32 < 0.6206221:
        z += 0.4231058 * Q.tau32 - 0.2625889
    if Q.N2 < 0.2052214:
        z += -9.373658 * Q.N2 + 1.923675
    if 0.7386202 <= Q.psi_0p1 < 0.9356675:
        z += -0.4382437 * Q.psi_0p1 + 0.3236956
    if Q.psi_0p1 >= 0.9356675:
        z += 7.499618 * Q.psi_0p1 - 7.103503
    if Q.mass < 56.49019:
        z += -0.01308768 * Q.mass + 1.887546
    if 56.49019 <= Q.mass < 117.4867:
        z += -0.005716291 * Q.mass + 1.471135
    if 117.4867 <= Q.mass < 131.3917:
        z += -0.001913334 * Q.mass + 1.024338
    if 131.3917 <= Q.mass < 164.4374:
        z += -0.003194181 * Q.mass + 1.192631
    if 164.4374 <= Q.mass < 182.8592:
        z += 0.01433778 * Q.mass - 1.690281
    if Q.mass >= 182.8592:
        z += 0.007371387 * Q.mass - 0.4164111
    if Q.e3_b2 < 0.0002536827:
        z += 629.4403 * Q.e3_b2 - 0.1596781
    if Q.sj4_pair_mass_max < 55.06039:
        z += 0.0003869379 * Q.sj4_pair_mass_max - 0.02130495
    if Q.sj4_pair_mass_max >= 88.5845:
        z += -0.003503808 * Q.sj4_pair_mass_max + 0.3103831
    if 79.27954 <= Q.mass_top50 < 89.68964:
        z += 0.002856017 * Q.mass_top50 - 0.2264237
    if Q.mass_top50 >= 89.68964:
        z += -0.008240083 * Q.mass_top50 + 0.7687814
    if Q.z_top5 >= 0.7768561:
        z += 2.189683 * Q.z_top5 - 1.701069
    if Q.LHA < 0.3016348:
        z += -1.033799 * Q.LHA + 0.3118296
    if Q.m01 < 21.76123:
        z += -0.001165142 * Q.m01 + 0.02535492
    if Q.N2_b05 < 0.302405:
        z += 4.446201 * Q.N2_b05 - 1.761401
    if 0.302405 <= Q.N2_b05 < 0.3961587:
        z += 5.11132 * Q.N2_b05 - 1.962537
    if Q.N2_b05 >= 0.3961587:
        z += 0.6651188 * Q.N2_b05 - 0.2011352
    if Q.sj3_dr_min < 0.07110203:
        z += 3.157132 * Q.sj3_dr_min - 0.1246244
    if 0.07110203 <= Q.sj3_dr_min < 0.2901133:
        z += -0.4559313 * Q.sj3_dr_min + 0.1322717
    if Q.n_pairs_kt_above_1 < 366.0:
        z += 0.0009841261 * Q.n_pairs_kt_above_1 - 0.3601902
    if Q.sj2_dr < 0.2403736:
        z += -0.2922847 * Q.sj2_dr + 0.07025754
    if Q.sj2_dr >= 0.6647889:
        z += -0.5361692 * Q.sj2_dr + 0.3564393
    if Q.dr02 < 0.1058963:
        z += -2.65012 * Q.dr02 + 0.05840592
    if 0.1058963 <= Q.dr02 < 0.4349199:
        z += 0.6754284 * Q.dr02 - 0.2937572
    if Q.sj3_z3 < 0.02858644:
        z += -15.53699 * Q.sj3_z3 + 0.4441474
    if Q.mass_top10 >= 103.4976:
        z += -0.008418566 * Q.mass_top10 + 0.8713017
    if Q.sj3_mass3 < 2.151069:
        z += 0.1449732 * Q.sj3_mass3 - 0.3118473
    if Q.z_dr_0p2_0p4 < 0.07307944:
        z += -0.4335211 * Q.z_dr_0p2_0p4 + 0.03168148
    if Q.tau32_b2 < 0.3835105:
        z += -1.121255 * Q.tau32_b2 + 0.430013
    if Q.dr01 < 0.1335998:
        z += -2.668392 * Q.dr01 + 0.1782697
    if 0.1335998 <= Q.dr01 < 0.3701694:
        z += 0.7533804 * Q.dr01 - 0.2788783
    if Q.sj3_pair_mass_max < 73.24742:
        z += -0.007187329 * Q.sj3_pair_mass_max + 0.5264533
    if Q.n_lund < 8.0:
        z += 0.04874966 * Q.n_lund - 0.3899972
    if Q.n_pairs_kt_above_3 < 47.0:
        z += 0.000789733 * Q.n_pairs_kt_above_3 - 0.08329884
    if 47.0 <= Q.n_pairs_kt_above_3 < 126.0:
        z += 0.0005845745 * Q.n_pairs_kt_above_3 - 0.07365638
    if Q.e3_b05 < 0.003262024:
        z += 185.5349 * Q.e3_b05 - 0.6052194
    if Q.lund_max_lnkt < 3.22013:
        z += 0.0006570791 * Q.lund_max_lnkt - 0.00211588
    if 62.03827 <= Q.sd_mass < 78.4753:
        z += -0.005524252 * Q.sd_mass + 0.342715
    if 78.4753 <= Q.sd_mass < 94.55722:
        z += 0.01339223 * Q.sd_mass - 1.141761
    if 94.55722 <= Q.sd_mass < 106.7501:
        z += -0.01015341 * Q.sd_mass + 1.084649
    if 106.7501 <= Q.sd_mass < 135.3031:
        z += -0.002099909 * Q.sd_mass + 0.2249369
    if 135.3031 <= Q.sd_mass < 154.5947:
        z += -0.002681178 * Q.sd_mass + 0.3035844
    if 154.5947 <= Q.sd_mass < 175.9333:
        z += 0.003912157 * Q.sd_mass - 0.71571
    if Q.sd_mass >= 175.9333:
        z += 0.01213261 * Q.sd_mass - 2.161961
    if Q.sum_z_dr2_top5 < 0.01851479:
        z += 1.392885 * Q.sum_z_dr2_top5 + 0.006042864
    if 0.01851479 <= Q.sum_z_dr2_top5 < 0.04544298:
        z += -1.182101 * Q.sum_z_dr2_top5 + 0.05371819
    if Q.mass_top40 >= 132.5189:
        z += 0.003129698 * Q.mass_top40 - 0.4147442
    if Q.z_top15_slots < 0.9212656:
        z += 1.033729 * Q.z_top15_slots - 0.9523393
    if Q.mass_over_sum_pt >= 0.2478639:
        z += 2.938756 * Q.mass_over_sum_pt - 0.7284115
    if Q.tau21 < 0.3245484:
        z += -0.1159469 * Q.tau21 + 0.03763039
    if Q.psi_0p2 >= 0.6958425:
        z += -0.2293203 * Q.psi_0p2 + 0.1595708
    if Q.pair_mean_lndelta >= -1.352792:
        z += -0.6303708 * Q.pair_mean_lndelta - 0.8527603
    if Q.sj3_mass1 >= 44.1303:
        z += -0.002986171 * Q.sj3_mass1 + 0.1317806
    if Q.M3_b2 < 0.00761379:
        z += -43.68716 * Q.M3_b2 + 0.3326249
    if Q.pair_max_lnkt >= 2.837304:
        z += 0.009278664 * Q.pair_max_lnkt - 0.02632639
    if Q.sj3_pair_mass_min >= 40.45675:
        z += -0.004269276 * Q.sj3_pair_mass_min + 0.172721
    if 60.04967 <= Q.sj2_mass1 < 77.42768:
        z += -0.008724378 * Q.sj2_mass1 + 0.5238961
    if 77.42768 <= Q.sj2_mass1 < 91.2852:
        z += 0.00917539 * Q.sj2_mass1 - 0.8620416
    if Q.sj2_mass1 >= 91.2852:
        z += -0.00402932 * Q.sj2_mass1 + 0.343353
    if Q.dr12 < 0.01602882:
        z += -7.250646 * Q.dr12 + 0.1162193
    if Q.pt_entropy >= 2.768601:
        z += 0.4237747 * Q.pt_entropy - 1.173263
    if Q.min_pair_mass >= 10.59762:
        z += -0.02321374 * Q.min_pair_mass + 0.2460103
    if Q.mass < 164.4374 and Q.pair_max_lnkt > 1.886287:
        z += -0.0002075165 * (164.4374 - Q.mass) * (Q.pair_max_lnkt - 1.886287)
    if Q.e3_b2 < 0.0002536827 and Q.pt1_over_pt0 < 0.5779883:
        z += -429.1776 * (0.0002536827 - Q.e3_b2) * (0.5779883 - Q.pt1_over_pt0)
    if Q.tau32 < 0.6206221 and Q.sj3_mass3 < 1.617171:
        z += 0.00795612 * (0.6206221 - Q.tau32) * (1.617171 - Q.sj3_mass3)
    if Q.mass < 164.4374 and Q.min_pair_mass > 10.59762:
        z += 0.0004147088 * (164.4374 - Q.mass) * (Q.min_pair_mass - 10.59762)
    if Q.mass_top50 > 79.27954 and Q.tau43 < 0.6363796:
        z += -0.0252474 * (Q.mass_top50 - 79.27954) * (0.6363796 - Q.tau43)
    if Q.mass < 164.4374 and Q.lnpt_34 > 0.05231816:
        z += -0.003692574 * (164.4374 - Q.mass) * (Q.lnpt_34 - 0.05231816)
    if Q.e3_b2 < 0.0002536827 and Q.n_lund > 10.0:
        z += 17.62867 * (0.0002536827 - Q.e3_b2) * (Q.n_lund - 10.0)
    if Q.n_pairs_kt_above_1 < 366.0 and Q.sj2_dr > 0.2848657:
        z += 0.003416443 * (366.0 - Q.n_pairs_kt_above_1) * (Q.sj2_dr - 0.2848657)
    if Q.tau32 < 0.6206221 and Q.pt1_dr01 > 29.53862:
        z += 0.009226135 * (0.6206221 - Q.tau32) * (Q.pt1_dr01 - 29.53862)
    if Q.mass < 164.4374 and Q.n_dr_0p2_0p4 < 21.0:
        z += -6.184058e-06 * (164.4374 - Q.mass) * (21.0 - Q.n_dr_0p2_0p4)
    if Q.mass_top50 > 79.27954 and Q.lne_0 < 5.423297:
        z += 0.001573356 * (Q.mass_top50 - 79.27954) * (5.423297 - Q.lne_0)
    if Q.mass_top10 > 103.4976 and Q.n_lund_kt_above_5 > 2.0:
        z += -0.0006765001 * (Q.mass_top10 - 103.4976) * (Q.n_lund_kt_above_5 - 2.0)
    if Q.sj3_dr_min < 0.2901133 and Q.lnpt_4 > 2.976008:
        z += 1.23103 * (0.2901133 - Q.sj3_dr_min) * (Q.lnpt_4 - 2.976008)
    if Q.mass_top50 > 79.27954 and Q.sj3_pairmax_over_m < 0.7925455:
        z += 0.02680863 * (Q.mass_top50 - 79.27954) * (0.7925455 - Q.sj3_pairmax_over_m)
    if Q.sj3_mass3 < 2.151069 and Q.sj3_pairmin_over_m > 0.1073315:
        z += -0.2430112 * (2.151069 - Q.sj3_mass3) * (Q.sj3_pairmin_over_m - 0.1073315)
    if Q.sj3_mass3 < 2.151069 and Q.ecf_g43 < 1.878367e-05:
        z += 6255.649 * (2.151069 - Q.sj3_mass3) * (1.878367e-05 - Q.ecf_g43)
    if Q.mass_top50 > 89.68964 and Q.sj3_dr13 > 0.3524911:
        z += -0.001644599 * (Q.mass_top50 - 89.68964) * (Q.sj3_dr13 - 0.3524911)
    if Q.mass_top50 > 89.68964 and Q.e4 > 7.184834e-06:
        z += 20.20463 * (Q.mass_top50 - 89.68964) * (Q.e4 - 7.184834e-06)
    if Q.tau32_b2 < 0.3835105 and Q.n_dr_0p05_0p1 > 0.0:
        z += -0.04725692 * (0.3835105 - Q.tau32_b2) * (Q.n_dr_0p05_0p1 - 0.0)
    if Q.e3_b2 < 0.0002536827 and Q.n_lund_kt_above_1 < 8.0:
        z += 170.981 * (0.0002536827 - Q.e3_b2) * (8.0 - Q.n_lund_kt_above_1)
    if Q.sd_mass > 175.9333 and Q.lnerel_77 > -18.42068:
        z += -0.0005226465 * (Q.sd_mass - 175.9333) * (Q.lnerel_77 - -18.42068)
    if Q.mass > 56.49019 and Q.lund3_lnkt < 4.039064:
        z += -7.521163e-05 * (Q.mass - 56.49019) * (4.039064 - Q.lund3_lnkt)
    if Q.tau32 < 0.6206221 and Q.n_dr_0p1_0p2 < 4.0:
        z += 0.2856345 * (0.6206221 - Q.tau32) * (4.0 - Q.n_dr_0p1_0p2)
    if Q.sd_mass > 175.9333 and Q.n_dr_0p1_0p2 < 9.0:
        z += -2.297158e-06 * (Q.sd_mass - 175.9333) * (9.0 - Q.n_dr_0p1_0p2)
    if Q.dr02 < 0.1058963 and Q.mratio_max_012 > 0.7525667:
        z += 13.14409 * (0.1058963 - Q.dr02) * (Q.mratio_max_012 - 0.7525667)
    if Q.dr02 < 0.1058963 and Q.planar_flow < 0.9376224:
        z += -1.973442 * (0.1058963 - Q.dr02) * (0.9376224 - Q.planar_flow)
    if Q.mass_top10 > 103.4976 and Q.sj2_mass2 < 19.43809:
        z += 0.0009618269 * (Q.mass_top10 - 103.4976) * (19.43809 - Q.sj2_mass2)
    if Q.pair_max_lnkt > 2.837304 and Q.sj3_z2 < 0.3246647:
        z += -4.175801 * (Q.pair_max_lnkt - 2.837304) * (0.3246647 - Q.sj3_z2)
    if Q.sj2_mass1 > 91.2852 and Q.phi_58 < 0.03336182:
        z += -0.0004029701 * (Q.sj2_mass1 - 91.2852) * (0.03336182 - Q.phi_58)
    if Q.sj2_dr > 0.6647889 and Q.sj2_mass2 > 42.12131:
        z += -0.1407081 * (Q.sj2_dr - 0.6647889) * (Q.sj2_mass2 - 42.12131)
    if Q.pair_max_lnkt > 2.837304 and Q.phi_20 < 0.3420532:
        z += -0.0004570376 * (Q.pair_max_lnkt - 2.837304) * (0.3420532 - Q.phi_20)
    if Q.sj3_mass3 < 2.151069 and Q.dr_10 > 0.2577145:
        z += -0.0183879 * (2.151069 - Q.sj3_mass3) * (Q.dr_10 - 0.2577145)
    if Q.N2_b05 > 0.302405 and Q.sj2_mass2 < 33.1786:
        z += 0.04602304 * (Q.N2_b05 - 0.302405) * (33.1786 - Q.sj2_mass2)
    if Q.N2_b05 < 0.3961587 and Q.max_dr < 0.9206713:
        z += -0.9623892 * (0.3961587 - Q.N2_b05) * (0.9206713 - Q.max_dr)
    return z


def neuron_105(Q):
    z = 4.008285e-06
    return z


def neuron_106(Q):
    z = 2.010549e-05
    return z


def neuron_107(Q):
    z = -5.028072e-05
    return z


def neuron_108(Q):
    z = 3.820155e-07
    return z


def neuron_109(Q):
    z = 8.850338e-06
    return z


def neuron_110(Q):
    z = 5.695643e-06
    return z


def neuron_111(Q):
    z = 2.546122e-06
    return z


def neuron_112(Q):
    z = 3.208627e-05
    return z


def neuron_113(Q):
    z = 3.073401e-05
    return z


def neuron_114(Q):
    z = 2.342204e-05
    return z


def neuron_115(Q):
    z = 2.963158e-06
    return z


def neuron_116(Q):
    z = 4.341653e-05
    return z


def neuron_117(Q):
    z = 3.964242e-06
    return z


def neuron_118(Q):
    z = -3.141795e-06
    return z


def neuron_119(Q):
    z = -2.952457e-06
    return z


def neuron_120(Q):
    z = -1.108244e-06
    return z


def neuron_121(Q):
    z = -1.226524
    if Q.pt_entropy < 3.303713:
        z += 1.328616 * Q.pt_entropy - 4.389365
    if Q.n_pairs_kt_above_3 < 34.0:
        z += -0.05195316 * Q.n_pairs_kt_above_3 + 2.374474
    if 34.0 <= Q.n_pairs_kt_above_3 < 126.0:
        z += -0.006609417 * Q.n_pairs_kt_above_3 + 0.8327866
    if Q.n_pairs_kt_above_3 >= 171.0:
        z += -0.001951759 * Q.n_pairs_kt_above_3 + 0.3337508
    if Q.sj2_mass2 < 1.851735:
        z += -0.1876963 * Q.sj2_mass2 + 0.3475638
    if Q.M2_b05 >= 0.1258758:
        z += 15.89375 * Q.M2_b05 - 2.000639
    if Q.n_pairs_kt_above_1 < 58.0:
        z += -0.04040821 * Q.n_pairs_kt_above_1 + 4.896428
    if 58.0 <= Q.n_pairs_kt_above_1 < 101.0:
        z += -0.01481317 * Q.n_pairs_kt_above_1 + 3.411916
    if 101.0 <= Q.n_pairs_kt_above_1 < 223.0:
        z += -0.00796333 * Q.n_pairs_kt_above_1 + 2.720082
    if 223.0 <= Q.n_pairs_kt_above_1 < 366.0:
        z += -0.00296745 * Q.n_pairs_kt_above_1 + 1.606
    if 366.0 <= Q.n_pairs_kt_above_1 < 598.0:
        z += -0.002241007 * Q.n_pairs_kt_above_1 + 1.340122
    if Q.sj3_pair_mass_min < 29.07349:
        z += 0.006421668 * Q.sj3_pair_mass_min - 0.1867003
    if Q.sj3_pair_mass_min >= 44.1643:
        z += -0.01044166 * Q.sj3_pair_mass_min + 0.4611488
    if Q.n_pairs_kt_above_10 < 23.0:
        z += -0.006577323 * Q.n_pairs_kt_above_10 + 0.1512784
    if Q.mass_top50 >= 178.725:
        z += -0.01161134 * Q.mass_top50 + 2.075236
    if 55.75601 <= Q.mass_top40 < 122.7145:
        z += 0.008713081 * Q.mass_top40 - 0.4858066
    if Q.mass_top40 >= 122.7145:
        z += 0.006026511 * Q.mass_top40 - 0.1561254
    if Q.pt_dispersion < 0.2484767:
        z += 0.9318121 * Q.pt_dispersion - 0.4226705
    if 0.2484767 <= Q.pt_dispersion < 0.4536006:
        z += 3.373732 * Q.pt_dispersion - 1.029431
    if Q.pt_dispersion >= 0.4536006:
        z += 2.44192 * Q.pt_dispersion - 0.6067603
    if Q.sj3_mass2 < 5.207416:
        z += -0.144108 * Q.sj3_mass2 + 0.7504303
    if Q.sj4_pair_mass_min < 34.83233:
        z += 0.01388675 * Q.sj4_pair_mass_min - 0.4837077
    if Q.mass < 100.4835:
        z += 0.007333306 * Q.mass - 1.007977
    if 100.4835 <= Q.mass < 137.452:
        z += 0.001750669 * Q.mass - 0.4470149
    if 137.452 <= Q.mass < 164.4374:
        z += -0.005582637 * Q.mass + 0.5609626
    if Q.mass >= 164.4374:
        z += 0.01803398 * Q.mass - 3.322494
    if Q.tau54 < 0.9329343:
        z += -1.052736 * Q.tau54 + 0.9821339
    if Q.mass_top30 < 91.4679:
        z += 0.00623432 * Q.mass_top30 - 0.5702401
    if Q.mass_top30 >= 126.5007:
        z += -0.004239727 * Q.mass_top30 + 0.5363285
    if Q.pt_balance01 < 0.2757029:
        z += 0.522873 * Q.pt_balance01 - 0.1441576
    if Q.n_dr_0p1_0p2 < 5.0:
        z += -0.1037605 * Q.n_dr_0p1_0p2 + 0.5188027
    if Q.n_real_top40 < 36.0:
        z += -0.03592555 * Q.n_real_top40 + 1.29332
    if Q.M3_b2 < 0.00678572:
        z += 36.96709 * Q.M3_b2 - 0.3729271
    if 0.00678572 <= Q.M3_b2 < 0.02160244:
        z += 8.239257 * Q.M3_b2 - 0.177988
    if Q.N3_b05 < 1.549573:
        z += -0.4413485 * Q.N3_b05 + 0.6839019
    if Q.tau32_b2 < 0.4276838:
        z += 2.04598 * Q.tau32_b2 - 0.8750326
    if Q.sj3_mass3 < 3.217057:
        z += 0.07887033 * Q.sj3_mass3 - 0.2537304
    if Q.pair_mean_lnz < -2.572052:
        z += -1.548281 * Q.pair_mean_lnz - 3.98226
    if Q.n_lund < 7.0:
        z += -0.2126887 * Q.n_lund + 1.488821
    if Q.tau2 < 0.02677564:
        z += -40.4437 * Q.tau2 + 1.082906
    if Q.tau32 < 0.8415274:
        z += 0.7997773 * Q.tau32 - 0.6730345
    if Q.z_dr_0p1_0p2 < 0.2428609:
        z += 0.3747549 * Q.z_dr_0p1_0p2 - 0.09101332
    if Q.C2_b2 < 0.006735806:
        z += 127.9065 * Q.C2_b2 - 0.8615533
    if Q.N2_b2 < 0.1528932:
        z += 2.715307 * Q.N2_b2 - 0.4151518
    if Q.D3_b05 < 0.8860453:
        z += -0.9915528 * Q.D3_b05 + 0.8785608
    if Q.N2_b05 < 0.5083429:
        z += -9.800146 * Q.N2_b05 + 4.981835
    if Q.N2 < 0.3701068:
        z += 6.146655 * Q.N2 - 2.274919
    if Q.C3_b2 < 8.293431e-05:
        z += 1931.749 * Q.C3_b2 - 0.1602082
    if Q.sj4_pair_mass_max < 107.8099:
        z += -0.004604626 * Q.sj4_pair_mass_max + 0.4964244
    if Q.sum_pt_top2 < 184.8313:
        z += 0.003281794 * Q.sum_pt_top2 - 0.6065781
    if Q.z_top20_slots >= 0.9834667:
        z += 22.01605 * Q.z_top20_slots - 21.65205
    if Q.sj3_mass1 >= 44.1303:
        z += -0.006994304 * Q.sj3_mass1 + 0.3086608
    if Q.pt_entropy < 3.303713 and Q.dr01 < 0.1335998:
        z += 0.12625 * (3.303713 - Q.pt_entropy) * (0.1335998 - Q.dr01)
    if Q.n_pairs_kt_above_3 < 126.0 and Q.dr_0 > 0.03142385:
        z += -0.007741734 * (126.0 - Q.n_pairs_kt_above_3) * (Q.dr_0 - 0.03142385)
    if Q.M2_b05 > 0.1258758 and Q.tau32 < 0.456416:
        z += -20.06821 * (Q.M2_b05 - 0.1258758) * (0.456416 - Q.tau32)
    if Q.n_pairs_kt_above_3 < 126.0 and Q.jet_abs_eta > 1.323111:
        z += -0.008356866 * (126.0 - Q.n_pairs_kt_above_3) * (Q.jet_abs_eta - 1.323111)
    if Q.n_pairs_kt_above_3 < 126.0 and Q.dr02 < 0.1620436:
        z += -0.02684181 * (126.0 - Q.n_pairs_kt_above_3) * (0.1620436 - Q.dr02)
    if Q.n_pairs_kt_above_3 < 126.0 and Q.sd_zg > 0.2450652:
        z += -0.006313912 * (126.0 - Q.n_pairs_kt_above_3) * (Q.sd_zg - 0.2450652)
    if Q.pt_entropy < 3.303713 and Q.sj2_mass2 < 24.55881:
        z += -0.007956325 * (3.303713 - Q.pt_entropy) * (24.55881 - Q.sj2_mass2)
    if Q.pt_dispersion < 0.4536006 and Q.sj3_mass3 < 3.217057:
        z += 0.9078896 * (0.4536006 - Q.pt_dispersion) * (3.217057 - Q.sj3_mass3)
    if Q.n_pairs_kt_above_10 < 23.0 and Q.sj3_pairmax_over_m < 0.9539857:
        z += 0.03268281 * (23.0 - Q.n_pairs_kt_above_10) * (0.9539857 - Q.sj3_pairmax_over_m)
    if Q.pt_entropy < 3.303713 and Q.M2_b2 > 0.01262589:
        z += 16.42327 * (3.303713 - Q.pt_entropy) * (Q.M2_b2 - 0.01262589)
    if Q.M2_b05 > 0.1258758 and Q.sj4_dr_min < 0.1302765:
        z += -81.06422 * (Q.M2_b05 - 0.1258758) * (0.1302765 - Q.sj4_dr_min)
    if Q.pt_entropy < 3.303713 and Q.n_lund > 9.0:
        z += 0.06034542 * (3.303713 - Q.pt_entropy) * (Q.n_lund - 9.0)
    if Q.n_pairs_kt_above_1 < 101.0 and Q.n_dr_0p05_0p1 > 6.0:
        z += 0.00144359 * (101.0 - Q.n_pairs_kt_above_1) * (Q.n_dr_0p05_0p1 - 6.0)
    if Q.n_real_top40 < 36.0 and Q.n_lund_kt_above_1 > 3.0:
        z += 0.006317127 * (36.0 - Q.n_real_top40) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.pt_balance01 < 0.2757029 and Q.sj2_zsoft > 0.2530865:
        z += 22.84499 * (0.2757029 - Q.pt_balance01) * (Q.sj2_zsoft - 0.2530865)
    if Q.mass_top40 > 122.7145 and Q.n_dr_0p2_0p4 < 12.0:
        z += 4.496502e-05 * (Q.mass_top40 - 122.7145) * (12.0 - Q.n_dr_0p2_0p4)
    if Q.pt_entropy < 3.303713 and Q.min_pair_mass < 10.59762:
        z += 0.04640891 * (3.303713 - Q.pt_entropy) * (10.59762 - Q.min_pair_mass)
    if Q.sj3_pair_mass_min > 44.1643 and Q.n_lund_kt_above_5 < 4.0:
        z += -0.001355013 * (Q.sj3_pair_mass_min - 44.1643) * (4.0 - Q.n_lund_kt_above_5)
    if Q.n_pairs_kt_above_1 < 598.0 and Q.sj2_zsoft > 0.2089827:
        z += -0.000400359 * (598.0 - Q.n_pairs_kt_above_1) * (Q.sj2_zsoft - 0.2089827)
    if Q.n_pairs_kt_above_1 < 598.0 and Q.tau43 > 0.568406:
        z += -0.002933812 * (598.0 - Q.n_pairs_kt_above_1) * (Q.tau43 - 0.568406)
    if Q.n_pairs_kt_above_10 < 23.0 and Q.sj2_dr > 0.2083105:
        z += -0.03306656 * (23.0 - Q.n_pairs_kt_above_10) * (Q.sj2_dr - 0.2083105)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.sum_z_dr2_top2 < 0.02489714:
        z += -2.06877 * (34.0 - Q.n_pairs_kt_above_3) * (0.02489714 - Q.sum_z_dr2_top2)
    if Q.n_pairs_kt_above_3 < 126.0 and Q.D2 < 1.831581:
        z += -0.004921369 * (126.0 - Q.n_pairs_kt_above_3) * (1.831581 - Q.D2)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.dr_22 < 0.02915846:
        z += 0.2416461 * (34.0 - Q.n_pairs_kt_above_3) * (0.02915846 - Q.dr_22)
    if Q.mass_top40 > 55.75601 and Q.tau21 > 0.5440886:
        z += 0.01494881 * (Q.mass_top40 - 55.75601) * (Q.tau21 - 0.5440886)
    if Q.sj3_pair_mass_min > 44.1643 and Q.lne_3 < 4.821289:
        z += 0.002363588 * (Q.sj3_pair_mass_min - 44.1643) * (4.821289 - Q.lne_3)
    if Q.tau32_b2 < 0.4276838 and Q.sj3_mass2 < 7.508521:
        z += 0.2361476 * (0.4276838 - Q.tau32_b2) * (7.508521 - Q.sj3_mass2)
    if Q.n_pairs_kt_above_1 < 598.0 and Q.tau32 > 0.456416:
        z += -0.004165594 * (598.0 - Q.n_pairs_kt_above_1) * (Q.tau32 - 0.456416)
    if Q.n_pairs_kt_above_3 < 126.0 and Q.sum_z_dr2_top2 > 0.001440167:
        z += 0.05923529 * (126.0 - Q.n_pairs_kt_above_3) * (Q.sum_z_dr2_top2 - 0.001440167)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.dr_1 > 0.01964368:
        z += -0.06028532 * (34.0 - Q.n_pairs_kt_above_3) * (Q.dr_1 - 0.01964368)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.dr12 > 0.3141459:
        z += 0.1668093 * (34.0 - Q.n_pairs_kt_above_3) * (Q.dr12 - 0.3141459)
    if Q.M2_b05 > 0.1258758 and Q.C3_b2 > 2.435109e-06:
        z += -10.68607 * (Q.M2_b05 - 0.1258758) * (Q.C3_b2 - 2.435109e-06)
    if Q.mass_top40 > 122.7145 and Q.tau43_b2 < 0.9433644:
        z += -0.0105623 * (Q.mass_top40 - 122.7145) * (0.9433644 - Q.tau43_b2)
    if Q.n_pairs_kt_above_1 < 598.0 and Q.e4_b05 > 7.503722e-05:
        z += -6.165198 * (598.0 - Q.n_pairs_kt_above_1) * (Q.e4_b05 - 7.503722e-05)
    if Q.n_pairs_kt_above_10 < 23.0 and Q.eta_2 > 0.126709:
        z += -0.01160796 * (23.0 - Q.n_pairs_kt_above_10) * (Q.eta_2 - 0.126709)
    if Q.tau32 < 0.8415274 and Q.dr_21 < 0.04063514:
        z += -13.02557 * (0.8415274 - Q.tau32) * (0.04063514 - Q.dr_21)
    if Q.mass > 164.4374 and Q.eta_39 < 0.1558838:
        z += 3.570268e-05 * (Q.mass - 164.4374) * (0.1558838 - Q.eta_39)
    if Q.n_dr_0p1_0p2 < 5.0 and Q.dr12 < 0.2116473:
        z += -0.5270939 * (5.0 - Q.n_dr_0p1_0p2) * (0.2116473 - Q.dr12)
    if Q.mass_top40 > 122.7145 and Q.dr12 < 0.2116473:
        z += 0.007676468 * (Q.mass_top40 - 122.7145) * (0.2116473 - Q.dr12)
    if Q.mass_top40 > 55.75601 and Q.dr_min_012 < 0.1557839:
        z += -0.01485896 * (Q.mass_top40 - 55.75601) * (0.1557839 - Q.dr_min_012)
    if Q.n_pairs_kt_above_3 < 126.0 and Q.dr_1 > 0.03335825:
        z += 0.0002870915 * (126.0 - Q.n_pairs_kt_above_3) * (Q.dr_1 - 0.03335825)
    if Q.mass < 137.452 and Q.C3_b2 < 0.0004257509:
        z += -6.049878 * (137.452 - Q.mass) * (0.0004257509 - Q.C3_b2)
    if Q.pt_entropy < 3.303713 and Q.sj3_mass2 < 6.006437:
        z += -0.1149174 * (3.303713 - Q.pt_entropy) * (6.006437 - Q.sj3_mass2)
    if Q.sj2_mass2 < 1.851735 and Q.phi_31 < -0.3249512:
        z += -1.379058 * (1.851735 - Q.sj2_mass2) * (-0.3249512 - Q.phi_31)
    if Q.mass_top30 > 126.5007 and Q.phi_17 > -0.1699219:
        z += -0.00242827 * (Q.mass_top30 - 126.5007) * (Q.phi_17 - -0.1699219)
    if Q.sj4_pair_mass_min < 34.83233 and Q.e4 > 7.184834e-06:
        z += 743.9065 * (34.83233 - Q.sj4_pair_mass_min) * (Q.e4 - 7.184834e-06)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.dr_2 < 0.1403176:
        z += 0.04274773 * (34.0 - Q.n_pairs_kt_above_3) * (0.1403176 - Q.dr_2)
    if Q.n_pairs_kt_above_1 < 598.0 and Q.sj3_mass2 > 1.824785:
        z += 6.251013e-06 * (598.0 - Q.n_pairs_kt_above_1) * (Q.sj3_mass2 - 1.824785)
    if Q.M3_b2 < 0.02160244 and Q.mratio_max_012 > 0.787882:
        z += -151.6896 * (0.02160244 - Q.M3_b2) * (Q.mratio_max_012 - 0.787882)
    if Q.tau54 < 0.9329343 and Q.phi_2 < -0.0838623:
        z += -0.6142933 * (0.9329343 - Q.tau54) * (-0.0838623 - Q.phi_2)
    return z


def neuron_122(Q):
    z = 4.169684e-06
    return z


def neuron_123(Q):
    z = 1.021101e-05
    return z


def neuron_124(Q):
    z = -2.958967e-06
    return z


def neuron_125(Q):
    z = -3.844663e-05
    return z


def neuron_126(Q):
    z = -1.142643e-05
    return z


def neuron_127(Q):
    z = 1.445596e-06
    return z


def class_token(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q), neuron_16(Q), neuron_17(Q), neuron_18(Q), neuron_19(Q), neuron_20(Q), neuron_21(Q), neuron_22(Q), neuron_23(Q), neuron_24(Q), neuron_25(Q), neuron_26(Q), neuron_27(Q), neuron_28(Q), neuron_29(Q), neuron_30(Q), neuron_31(Q), neuron_32(Q), neuron_33(Q), neuron_34(Q), neuron_35(Q), neuron_36(Q), neuron_37(Q), neuron_38(Q), neuron_39(Q), neuron_40(Q), neuron_41(Q), neuron_42(Q), neuron_43(Q), neuron_44(Q), neuron_45(Q), neuron_46(Q), neuron_47(Q), neuron_48(Q), neuron_49(Q), neuron_50(Q), neuron_51(Q), neuron_52(Q), neuron_53(Q), neuron_54(Q), neuron_55(Q), neuron_56(Q), neuron_57(Q), neuron_58(Q), neuron_59(Q), neuron_60(Q), neuron_61(Q), neuron_62(Q), neuron_63(Q), neuron_64(Q), neuron_65(Q), neuron_66(Q), neuron_67(Q), neuron_68(Q), neuron_69(Q), neuron_70(Q), neuron_71(Q), neuron_72(Q), neuron_73(Q), neuron_74(Q), neuron_75(Q), neuron_76(Q), neuron_77(Q), neuron_78(Q), neuron_79(Q), neuron_80(Q), neuron_81(Q), neuron_82(Q), neuron_83(Q), neuron_84(Q), neuron_85(Q), neuron_86(Q), neuron_87(Q), neuron_88(Q), neuron_89(Q), neuron_90(Q), neuron_91(Q), neuron_92(Q), neuron_93(Q), neuron_94(Q), neuron_95(Q), neuron_96(Q), neuron_97(Q), neuron_98(Q), neuron_99(Q), neuron_100(Q), neuron_101(Q), neuron_102(Q), neuron_103(Q), neuron_104(Q), neuron_105(Q), neuron_106(Q), neuron_107(Q), neuron_108(Q), neuron_109(Q), neuron_110(Q), neuron_111(Q), neuron_112(Q), neuron_113(Q), neuron_114(Q), neuron_115(Q), neuron_116(Q), neuron_117(Q), neuron_118(Q), neuron_119(Q), neuron_120(Q), neuron_121(Q), neuron_122(Q), neuron_123(Q), neuron_124(Q), neuron_125(Q), neuron_126(Q), neuron_127(Q)]


def logits(h):
    return [B[c] + sum(h[j] * W[j][c] for j in range(len(h))) for c in range(10)]


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
