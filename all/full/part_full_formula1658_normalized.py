"""Particle Transformer (ParT) jet tagger, JetClass, input features 'full': tuned on the network's predictions (from 100 if-statements per neuron; the 17 neurons of the first neuron cut, terms not pruned), with normalized weights (how much each one matters), as if-statements.

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
  neuron 104:  10.3%
  neuron 24:   9.8%
  neuron 120:   9.0%
  neuron 97:   8.9%
  neuron 68:   7.9%
  neuron 18:   7.5%
  neuron 115:   6.8%
  neuron 52:   5.9%
  neuron 83:   5.9%
  neuron 124:   5.0%
  neuron 78:   3.9%
  neuron 81:   3.4%
  neuron 70:   1.4%
  neuron 90:   0.8%
  neuron 25:   0.7%
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

Whole test file (2,000,000 jets): accuracy 74.77% (the network: 86.03%); same class as the network for 79.48% of jets.

Quantities:
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b05                 e3/e2² with β = 0.5
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3_b05                 e4·e2/e3² with β = 0.5
  Q.C3_b2                  e4·e2/e3² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b05                 e3/e2³ with β = 0.5
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
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
  Q.charge_0               charge of particle 0 (ParT input; empty slot: 0)
  Q.charge_25              charge of particle 25 (ParT input; empty slot: 0)
  Q.charge_27              charge of particle 27 (ParT input; empty slot: 0)
  Q.charge_32              charge of particle 32 (ParT input; empty slot: 0)
  Q.charge_42              charge of particle 42 (ParT input; empty slot: 0)
  Q.charge_43              charge of particle 43 (ParT input; empty slot: 0)
  Q.charge_6               charge of particle 6 (ParT input; empty slot: 0)
  Q.d0err_10               σ(d0), clipped to [0, 1] of particle 10 (ParT input; empty slot: 0)
  Q.d0err_12               σ(d0), clipped to [0, 1] of particle 12 (ParT input; empty slot: 0)
  Q.d0err_14               σ(d0), clipped to [0, 1] of particle 14 (ParT input; empty slot: 0)
  Q.d0err_16               σ(d0), clipped to [0, 1] of particle 16 (ParT input; empty slot: 0)
  Q.d0err_3                σ(d0), clipped to [0, 1] of particle 3 (ParT input; empty slot: 0)
  Q.d0err_42               σ(d0), clipped to [0, 1] of particle 42 (ParT input; empty slot: 0)
  Q.d0err_52               σ(d0), clipped to [0, 1] of particle 52 (ParT input; empty slot: 0)
  Q.d0err_75               σ(d0), clipped to [0, 1] of particle 75 (ParT input; empty slot: 0)
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr02                   ΔR between particles 0 and 2
  Q.dr12                   ΔR between particles 1 and 2
  Q.dr_0                   ΔR from the jet axis of particle 0 (ParT input; empty slot: 0)
  Q.dr_1                   ΔR from the jet axis of particle 1 (ParT input; empty slot: 0)
  Q.dr_10                  ΔR from the jet axis of particle 10 (ParT input; empty slot: 0)
  Q.dr_16                  ΔR from the jet axis of particle 16 (ParT input; empty slot: 0)
  Q.dr_19                  ΔR from the jet axis of particle 19 (ParT input; empty slot: 0)
  Q.dr_21                  ΔR from the jet axis of particle 21 (ParT input; empty slot: 0)
  Q.dr_22                  ΔR from the jet axis of particle 22 (ParT input; empty slot: 0)
  Q.dr_25                  ΔR from the jet axis of particle 25 (ParT input; empty slot: 0)
  Q.dr_28                  ΔR from the jet axis of particle 28 (ParT input; empty slot: 0)
  Q.dr_3                   ΔR from the jet axis of particle 3 (ParT input; empty slot: 0)
  Q.dr_31                  ΔR from the jet axis of particle 31 (ParT input; empty slot: 0)
  Q.dr_4                   ΔR from the jet axis of particle 4 (ParT input; empty slot: 0)
  Q.dr_53                  ΔR from the jet axis of particle 53 (ParT input; empty slot: 0)
  Q.dr_55                  ΔR from the jet axis of particle 55 (ParT input; empty slot: 0)
  Q.dr_77                  ΔR from the jet axis of particle 77 (ParT input; empty slot: 0)
  Q.dr_8                   ΔR from the jet axis of particle 8 (ParT input; empty slot: 0)
  Q.dr_9                   ΔR from the jet axis of particle 9 (ParT input; empty slot: 0)
  Q.dr_max_012             largest distance among the 3 hardest
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.dzerr_0                σ(dz), clipped to [0, 1] of particle 0 (ParT input; empty slot: 0)
  Q.dzerr_1                σ(dz), clipped to [0, 1] of particle 1 (ParT input; empty slot: 0)
  Q.dzerr_43               σ(dz), clipped to [0, 1] of particle 43 (ParT input; empty slot: 0)
  Q.dzerr_44               σ(dz), clipped to [0, 1] of particle 44 (ParT input; empty slot: 0)
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_b05                 energy correlation e2 with β = 0.5
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
  Q.eta_14                 Δη of particle 14 (ParT input; empty slot: 0)
  Q.eta_19                 Δη of particle 19 (ParT input; empty slot: 0)
  Q.eta_2                  Δη of particle 2 (ParT input; empty slot: 0)
  Q.eta_28                 Δη of particle 28 (ParT input; empty slot: 0)
  Q.eta_29                 Δη of particle 29 (ParT input; empty slot: 0)
  Q.eta_36                 Δη of particle 36 (ParT input; empty slot: 0)
  Q.eta_37                 Δη of particle 37 (ParT input; empty slot: 0)
  Q.eta_4                  Δη of particle 4 (ParT input; empty slot: 0)
  Q.eta_48                 Δη of particle 48 (ParT input; empty slot: 0)
  Q.eta_7                  Δη of particle 7 (ParT input; empty slot: 0)
  Q.eta_76                 Δη of particle 76 (ParT input; empty slot: 0)
  Q.eta_8                  Δη of particle 8 (ParT input; empty slot: 0)
  Q.eta_9                  Δη of particle 9 (ParT input; empty slot: 0)
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.ischhad_10             1 if a charged hadron of particle 10 (ParT input; empty slot: 0)
  Q.ischhad_26             1 if a charged hadron of particle 26 (ParT input; empty slot: 0)
  Q.ischhad_4              1 if a charged hadron of particle 4 (ParT input; empty slot: 0)
  Q.ischhad_57             1 if a charged hadron of particle 57 (ParT input; empty slot: 0)
  Q.iselectron_0           1 if an electron of particle 0 (ParT input; empty slot: 0)
  Q.iselectron_1           1 if an electron of particle 1 (ParT input; empty slot: 0)
  Q.iselectron_11          1 if an electron of particle 11 (ParT input; empty slot: 0)
  Q.iselectron_12          1 if an electron of particle 12 (ParT input; empty slot: 0)
  Q.iselectron_14          1 if an electron of particle 14 (ParT input; empty slot: 0)
  Q.iselectron_24          1 if an electron of particle 24 (ParT input; empty slot: 0)
  Q.iselectron_27          1 if an electron of particle 27 (ParT input; empty slot: 0)
  Q.iselectron_29          1 if an electron of particle 29 (ParT input; empty slot: 0)
  Q.iselectron_30          1 if an electron of particle 30 (ParT input; empty slot: 0)
  Q.iselectron_34          1 if an electron of particle 34 (ParT input; empty slot: 0)
  Q.ismuon_0               1 if a muon of particle 0 (ParT input; empty slot: 0)
  Q.ismuon_11              1 if a muon of particle 11 (ParT input; empty slot: 0)
  Q.ismuon_12              1 if a muon of particle 12 (ParT input; empty slot: 0)
  Q.ismuon_14              1 if a muon of particle 14 (ParT input; empty slot: 0)
  Q.ismuon_2               1 if a muon of particle 2 (ParT input; empty slot: 0)
  Q.ismuon_24              1 if a muon of particle 24 (ParT input; empty slot: 0)
  Q.ismuon_27              1 if a muon of particle 27 (ParT input; empty slot: 0)
  Q.ismuon_29              1 if a muon of particle 29 (ParT input; empty slot: 0)
  Q.ismuon_37              1 if a muon of particle 37 (ParT input; empty slot: 0)
  Q.ismuon_4               1 if a muon of particle 4 (ParT input; empty slot: 0)
  Q.ismuon_49              1 if a muon of particle 49 (ParT input; empty slot: 0)
  Q.ismuon_73              1 if a muon of particle 73 (ParT input; empty slot: 0)
  Q.isnhad_1               1 if a neutral hadron of particle 1 (ParT input; empty slot: 0)
  Q.isnhad_27              1 if a neutral hadron of particle 27 (ParT input; empty slot: 0)
  Q.isnhad_34              1 if a neutral hadron of particle 34 (ParT input; empty slot: 0)
  Q.isnhad_8               1 if a neutral hadron of particle 8 (ParT input; empty slot: 0)
  Q.isphoton_0             1 if a photon of particle 0 (ParT input; empty slot: 0)
  Q.isphoton_12            1 if a photon of particle 12 (ParT input; empty slot: 0)
  Q.isphoton_14            1 if a photon of particle 14 (ParT input; empty slot: 0)
  Q.isphoton_32            1 if a photon of particle 32 (ParT input; empty slot: 0)
  Q.isphoton_41            1 if a photon of particle 41 (ParT input; empty slot: 0)
  Q.isphoton_42            1 if a photon of particle 42 (ParT input; empty slot: 0)
  Q.isphoton_53            1 if a photon of particle 53 (ParT input; empty slot: 0)
  Q.jet_abs_eta            absolute pseudorapidity of the jet axis
  Q.jet_charge             pT-weighted jet charge (κ = 1)
  Q.jet_charge_k03         jet charge with κ = 0.3
  Q.jet_charge_k05         jet charge with κ = 0.5
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lead_ch_sdz            signed dz/σ of the hardest charged particle (0 if none)
  Q.lep_dr                 ΔR of the hardest lepton from the jet axis (0 if none)
  Q.lep_iso                Σ pT of the other particles within ΔR < 0.2 of the hardest lepton / its pT (0 if none)
  Q.lep_ptrel              pT × ΔR from the jet axis of the hardest lepton [GeV] (0 if none)
  Q.lep_z                  pT share of the hardest electron or muon (0 if none)
  Q.lne_0                  ln E [GeV] of particle 0 (ParT input; empty slot: ln 1e-8)
  Q.lne_10                 ln E [GeV] of particle 10 (ParT input; empty slot: ln 1e-8)
  Q.lne_2                  ln E [GeV] of particle 2 (ParT input; empty slot: ln 1e-8)
  Q.lne_4                  ln E [GeV] of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lne_5                  ln E [GeV] of particle 5 (ParT input; empty slot: ln 1e-8)
  Q.lne_7                  ln E [GeV] of particle 7 (ParT input; empty slot: ln 1e-8)
  Q.lne_8                  ln E [GeV] of particle 8 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_17              ln(E / E of the jet) of particle 17 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_4               ln(E / E of the jet) of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_49              ln(E / E of the jet) of particle 49 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_82              ln(E / E of the jet) of particle 82 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_11                ln pT [GeV] of particle 11 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_41                ln pT [GeV] of particle 41 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_44                ln pT [GeV] of particle 44 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_5                 ln pT [GeV] of particle 5 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_12             ln(pT / pT of the jet) of particle 12 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_3              ln(pT / pT of the jet) of particle 3 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_40             ln(pT / pT of the jet) of particle 40 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_45             ln(pT / pT of the jet) of particle 45 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_77             ln(pT / pT of the jet) of particle 77 (ParT input; empty slot: ln 1e-8)
  Q.log_sum_pt             natural log of the total pT
  Q.lund1_lndelta          ln Δ of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund1_lnz              ln z of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund2_lndelta          ln Δ of the 2. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lndelta          ln Δ of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lnkt             ln kT of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lnz              ln z of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund_max_lndelta       ln Δ of the primary splitting with the largest kT
  Q.lund_max_lnkt          largest ln kT among the primary splittings
  Q.m01                    mass of particles 0 and 1 [GeV]
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_2charged          invariant mass of the 2 hardest charged particles [GeV]
  Q.mass_2photon           invariant mass of the 2 hardest photons [GeV] (0 if fewer)
  Q.mass_charged           invariant mass of all charged particles [GeV]
  Q.mass_displaced3        invariant mass of the charged particles with |d0|/σ > 3 [GeV]
  Q.mass_displaced5        invariant mass of the charged particles with |d0|/σ > 5 [GeV]
  Q.mass_neutral           invariant mass of all neutral particles [GeV]
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
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
  Q.min_pair_mass          smallest pair mass among particles 0, 1, 2 [GeV]
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.n_charged              number of charged particles
  Q.n_charged_had          number of charged hadrons
  Q.n_charged_pt_above_1   number of charged particles with pT > 1 GeV
  Q.n_charged_pt_above_10  number of charged particles with pT > 10 GeV
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
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
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.n_real_top15           number of real particles among the 15 hardest
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.n_s3d_above_10         number of charged particles with 3d significance > 10
  Q.n_s3d_above_3          number of charged particles with 3d significance > 3
  Q.n_sd0_above_10         number of charged particles with d0 significance > 10
  Q.n_sd0_above_2          number of charged particles with d0 significance > 2
  Q.n_sd0_above_3          number of charged particles with d0 significance > 3
  Q.n_sd0_above_5          number of charged particles with d0 significance > 5
  Q.n_sdz_above_2          number of charged particles with dz significance > 2
  Q.n_sdz_above_5          number of charged particles with dz significance > 5
  Q.pair_max_lnkt          largest ln kT among all pairs
  Q.pair_max_lnm2          largest ln m² among all pairs
  Q.pair_mean_lndelta      zᵢzⱼ-weighted mean of ln ΔRᵢⱼ over all pairs
  Q.pair_mean_lnkt         zᵢzⱼ-weighted mean of ln kT = ln(min(pTᵢ, pTⱼ)·ΔRᵢⱼ) over all pairs
  Q.pair_mean_lnm2         zᵢzⱼ-weighted mean of ln mᵢⱼ² (massless) over all pairs
  Q.pair_mean_lnz          zᵢzⱼ-weighted mean of ln z = ln(min(pTᵢ, pTⱼ)/(pTᵢ + pTⱼ)) over all pairs
  Q.phi_1                  Δφ of particle 1 (ParT input; empty slot: 0)
  Q.phi_21                 Δφ of particle 21 (ParT input; empty slot: 0)
  Q.phi_22                 Δφ of particle 22 (ParT input; empty slot: 0)
  Q.phi_26                 Δφ of particle 26 (ParT input; empty slot: 0)
  Q.phi_29                 Δφ of particle 29 (ParT input; empty slot: 0)
  Q.phi_3                  Δφ of particle 3 (ParT input; empty slot: 0)
  Q.phi_31                 Δφ of particle 31 (ParT input; empty slot: 0)
  Q.phi_79                 Δφ of particle 79 (ParT input; empty slot: 0)
  Q.phi_84                 Δφ of particle 84 (ParT input; empty slot: 0)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.pt1_over_pt0           pT1 / pT0
  Q.pt2_over_pt0           pT2 / pT0
  Q.pt_balance01           min(pT0, pT1) / (pT0 + pT1)
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.sd_mass                soft-drop groomed mass, C/A on the 128 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_nremoved            number of branches removed by soft drop
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sip_3d_1               the 1. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sip_3d_2               the 2. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sip_3d_3               the 3. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
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
  Q.sj3_z2                 pT share of subjet 2 of 3 (by pT)
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.sj4_dr_min             smallest distance among the 4 subjet axes
  Q.sj4_pair_mass_max      largest mass of two of the 4 subjets [GeV]
  Q.sj4_pair_mass_min      smallest mass of two of the 4 subjets [GeV]
  Q.sj4_zsoft              pT share of the softest of 4 subjets
  Q.sum_charge             total charge of the particles
  Q.sum_e                  total energy of the particles [GeV]
  Q.sum_pt                 total pT of the particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
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
  Q.td0_13                 tanh(d0 [mm]) of particle 13 (ParT input; empty slot: 0)
  Q.td0_14                 tanh(d0 [mm]) of particle 14 (ParT input; empty slot: 0)
  Q.td0_17                 tanh(d0 [mm]) of particle 17 (ParT input; empty slot: 0)
  Q.td0_2                  tanh(d0 [mm]) of particle 2 (ParT input; empty slot: 0)
  Q.td0_23                 tanh(d0 [mm]) of particle 23 (ParT input; empty slot: 0)
  Q.td0_3                  tanh(d0 [mm]) of particle 3 (ParT input; empty slot: 0)
  Q.td0_30                 tanh(d0 [mm]) of particle 30 (ParT input; empty slot: 0)
  Q.td0_33                 tanh(d0 [mm]) of particle 33 (ParT input; empty slot: 0)
  Q.td0_38                 tanh(d0 [mm]) of particle 38 (ParT input; empty slot: 0)
  Q.td0_44                 tanh(d0 [mm]) of particle 44 (ParT input; empty slot: 0)
  Q.td0_7                  tanh(d0 [mm]) of particle 7 (ParT input; empty slot: 0)
  Q.td0_9                  tanh(d0 [mm]) of particle 9 (ParT input; empty slot: 0)
  Q.tdz_0                  tanh(dz [mm]) of particle 0 (ParT input; empty slot: 0)
  Q.tdz_1                  tanh(dz [mm]) of particle 1 (ParT input; empty slot: 0)
  Q.tdz_13                 tanh(dz [mm]) of particle 13 (ParT input; empty slot: 0)
  Q.tdz_2                  tanh(dz [mm]) of particle 2 (ParT input; empty slot: 0)
  Q.tdz_25                 tanh(dz [mm]) of particle 25 (ParT input; empty slot: 0)
  Q.tdz_29                 tanh(dz [mm]) of particle 29 (ParT input; empty slot: 0)
  Q.tdz_33                 tanh(dz [mm]) of particle 33 (ParT input; empty slot: 0)
  Q.tdz_41                 tanh(dz [mm]) of particle 41 (ParT input; empty slot: 0)
  Q.tdz_44                 tanh(dz [mm]) of particle 44 (ParT input; empty slot: 0)
  Q.tdz_47                 tanh(dz [mm]) of particle 47 (ParT input; empty slot: 0)
  Q.tdz_54                 tanh(dz [mm]) of particle 54 (ParT input; empty slot: 0)
  Q.tdz_6                  tanh(dz [mm]) of particle 6 (ParT input; empty slot: 0)
  Q.tdz_68                 tanh(dz [mm]) of particle 68 (ParT input; empty slot: 0)
  Q.tdz_74                 tanh(dz [mm]) of particle 74 (ParT input; empty slot: 0)
  Q.tdz_9                  tanh(dz [mm]) of particle 9 (ParT input; empty slot: 0)
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
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top5                 pT share of the 5 largest
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

    return SimpleNamespace(
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b05=ecfb('e3', 0.5) / max(ecfb('e2', 0.5) ** 2, 1e-30),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3_b05=ecfb('e4', 0.5) * ecfb('e2', 0.5) / max(ecfb('e3', 0.5) ** 2, 1e-30),
        C3_b2=ecfb('e4', 2) * ecfb('e2', 2) / max(ecfb('e3', 2) ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b05=ecfb('e3', 0.5) / max(ecfb('e2', 0.5) ** 3, 1e-30),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
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
        charge_0=pfeat(0, 'charge'),
        charge_25=pfeat(25, 'charge'),
        charge_27=pfeat(27, 'charge'),
        charge_32=pfeat(32, 'charge'),
        charge_42=pfeat(42, 'charge'),
        charge_43=pfeat(43, 'charge'),
        charge_6=pfeat(6, 'charge'),
        d0err_10=pfeat(10, 'd0err'),
        d0err_12=pfeat(12, 'd0err'),
        d0err_14=pfeat(14, 'd0err'),
        d0err_16=pfeat(16, 'd0err'),
        d0err_3=pfeat(3, 'd0err'),
        d0err_42=pfeat(42, 'd0err'),
        d0err_52=pfeat(52, 'd0err'),
        d0err_75=pfeat(75, 'd0err'),
        dr01=math.sqrt(dist2(0, 1)),
        dr02=math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        dr_0=pfeat(0, 'dr'),
        dr_1=pfeat(1, 'dr'),
        dr_10=pfeat(10, 'dr'),
        dr_16=pfeat(16, 'dr'),
        dr_19=pfeat(19, 'dr'),
        dr_21=pfeat(21, 'dr'),
        dr_22=pfeat(22, 'dr'),
        dr_25=pfeat(25, 'dr'),
        dr_28=pfeat(28, 'dr'),
        dr_3=pfeat(3, 'dr'),
        dr_31=pfeat(31, 'dr'),
        dr_4=pfeat(4, 'dr'),
        dr_53=pfeat(53, 'dr'),
        dr_55=pfeat(55, 'dr'),
        dr_77=pfeat(77, 'dr'),
        dr_8=pfeat(8, 'dr'),
        dr_9=pfeat(9, 'dr'),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        dzerr_0=pfeat(0, 'dzerr'),
        dzerr_1=pfeat(1, 'dzerr'),
        dzerr_43=pfeat(43, 'dzerr'),
        dzerr_44=pfeat(44, 'dzerr'),
        e2=e2,
        e2_b05=ecfb('e2', 0.5),
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
        eta_14=pfeat(14, 'eta'),
        eta_19=pfeat(19, 'eta'),
        eta_2=pfeat(2, 'eta'),
        eta_28=pfeat(28, 'eta'),
        eta_29=pfeat(29, 'eta'),
        eta_36=pfeat(36, 'eta'),
        eta_37=pfeat(37, 'eta'),
        eta_4=pfeat(4, 'eta'),
        eta_48=pfeat(48, 'eta'),
        eta_7=pfeat(7, 'eta'),
        eta_76=pfeat(76, 'eta'),
        eta_8=pfeat(8, 'eta'),
        eta_9=pfeat(9, 'eta'),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        ischhad_10=pfeat(10, 'ischhad'),
        ischhad_26=pfeat(26, 'ischhad'),
        ischhad_4=pfeat(4, 'ischhad'),
        ischhad_57=pfeat(57, 'ischhad'),
        iselectron_0=pfeat(0, 'iselectron'),
        iselectron_1=pfeat(1, 'iselectron'),
        iselectron_11=pfeat(11, 'iselectron'),
        iselectron_12=pfeat(12, 'iselectron'),
        iselectron_14=pfeat(14, 'iselectron'),
        iselectron_24=pfeat(24, 'iselectron'),
        iselectron_27=pfeat(27, 'iselectron'),
        iselectron_29=pfeat(29, 'iselectron'),
        iselectron_30=pfeat(30, 'iselectron'),
        iselectron_34=pfeat(34, 'iselectron'),
        ismuon_0=pfeat(0, 'ismuon'),
        ismuon_11=pfeat(11, 'ismuon'),
        ismuon_12=pfeat(12, 'ismuon'),
        ismuon_14=pfeat(14, 'ismuon'),
        ismuon_2=pfeat(2, 'ismuon'),
        ismuon_24=pfeat(24, 'ismuon'),
        ismuon_27=pfeat(27, 'ismuon'),
        ismuon_29=pfeat(29, 'ismuon'),
        ismuon_37=pfeat(37, 'ismuon'),
        ismuon_4=pfeat(4, 'ismuon'),
        ismuon_49=pfeat(49, 'ismuon'),
        ismuon_73=pfeat(73, 'ismuon'),
        isnhad_1=pfeat(1, 'isnhad'),
        isnhad_27=pfeat(27, 'isnhad'),
        isnhad_34=pfeat(34, 'isnhad'),
        isnhad_8=pfeat(8, 'isnhad'),
        isphoton_0=pfeat(0, 'isphoton'),
        isphoton_12=pfeat(12, 'isphoton'),
        isphoton_14=pfeat(14, 'isphoton'),
        isphoton_32=pfeat(32, 'isphoton'),
        isphoton_41=pfeat(41, 'isphoton'),
        isphoton_42=pfeat(42, 'isphoton'),
        isphoton_53=pfeat(53, 'isphoton'),
        jet_abs_eta=abs(jet_eta),
        jet_charge=sum(charge[i] * z[i] for i in real),
        jet_charge_k03=sum(charge[i] * z[i] ** 0.3 for i in real),
        jet_charge_k05=sum(charge[i] * z[i] ** 0.5 for i in real),
        lam1=lam1,
        lam2=lam2,
        lead_ch_sdz=leadtrack('dz'),
        lep_dr=lepton('dr'),
        lep_iso=lepton('iso'),
        lep_ptrel=lepton('ptrel'),
        lep_z=lepton('z'),
        lne_0=pfeat(0, 'lne'),
        lne_10=pfeat(10, 'lne'),
        lne_2=pfeat(2, 'lne'),
        lne_4=pfeat(4, 'lne'),
        lne_5=pfeat(5, 'lne'),
        lne_7=pfeat(7, 'lne'),
        lne_8=pfeat(8, 'lne'),
        lnerel_17=pfeat(17, 'lnerel'),
        lnerel_4=pfeat(4, 'lnerel'),
        lnerel_49=pfeat(49, 'lnerel'),
        lnerel_82=pfeat(82, 'lnerel'),
        lnpt_11=pfeat(11, 'lnpt'),
        lnpt_41=pfeat(41, 'lnpt'),
        lnpt_44=pfeat(44, 'lnpt'),
        lnpt_5=pfeat(5, 'lnpt'),
        lnptrel_12=pfeat(12, 'lnptrel'),
        lnptrel_3=pfeat(3, 'lnptrel'),
        lnptrel_40=pfeat(40, 'lnptrel'),
        lnptrel_45=pfeat(45, 'lnptrel'),
        lnptrel_77=pfeat(77, 'lnptrel'),
        log_sum_pt=math.log(tot),
        lund1_lndelta=lund(1, 'lndelta'),
        lund1_lnz=lund(1, 'lnz'),
        lund2_lndelta=lund(2, 'lndelta'),
        lund3_lndelta=lund(3, 'lndelta'),
        lund3_lnkt=lund(3, 'lnkt'),
        lund3_lnz=lund(3, 'lnz'),
        lund_max_lndelta=lund(0, 'maxdelta'),
        lund_max_lnkt=lund(0, 'maxkt'),
        m01=pair_mass(0, 1),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        mass=mass_of(n),
        mass_2charged=subset_mass('charged2'),
        mass_2photon=subset_mass('photon2'),
        mass_charged=subset_mass('charged'),
        mass_displaced3=displaced(3, 'mass'),
        mass_displaced5=displaced(5, 'mass'),
        mass_neutral=subset_mass('neutral'),
        mass_over_sum_pt=mass_of(n) / tot,
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
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
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        n_charged=sum(1 for i in real if charge[i] != 0),
        n_charged_had=sum(1 for i in real if ptype[i] == 1),
        n_charged_pt_above_1=sum(1 for i in real if charge[i] != 0 and pt[i] > 1),
        n_charged_pt_above_10=sum(1 for i in real if charge[i] != 0 and pt[i] > 10),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
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
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        n_real_top15=sum(1 for x in pt[:15] if x > 0),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        n_s3d_above_10=nsig('3d', 10),
        n_s3d_above_3=nsig('3d', 3),
        n_sd0_above_10=nsig('d0', 10),
        n_sd0_above_2=nsig('d0', 2),
        n_sd0_above_3=nsig('d0', 3),
        n_sd0_above_5=nsig('d0', 5),
        n_sdz_above_2=nsig('dz', 2),
        n_sdz_above_5=nsig('dz', 5),
        pair_max_lnkt=pairmax('lnkt'),
        pair_max_lnm2=pairmax('lnm2'),
        pair_mean_lndelta=pairsum('lndelta'),
        pair_mean_lnkt=pairsum('lnkt'),
        pair_mean_lnm2=pairsum('lnm2'),
        pair_mean_lnz=pairsum('lnz'),
        phi_1=pfeat(1, 'phi'),
        phi_21=pfeat(21, 'phi'),
        phi_22=pfeat(22, 'phi'),
        phi_26=pfeat(26, 'phi'),
        phi_29=pfeat(29, 'phi'),
        phi_3=pfeat(3, 'phi'),
        phi_31=pfeat(31, 'phi'),
        phi_79=pfeat(79, 'phi'),
        phi_84=pfeat(84, 'phi'),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        pt_balance01=min(pt[0], pt[1]) / max(pt[0] + pt[1], 1e-9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        sd_mass=softdrop("mass"),
        sd_nremoved=softdrop("removed"),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        sip_3d_1=sip('3d', 1),
        sip_3d_2=sip('3d', 2),
        sip_3d_3=sip('3d', 3),
        sj2_dr=subjets(2)["dr"][0],
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
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
        sj3_z2=subjets(3)["z"][1],
        sj3_z3=subjets(3)["z"][2],
        sj4_dr_min=min(subjets(4)["dr"]),
        sj4_pair_mass_max=max(subjets(4)["mpair"]),
        sj4_pair_mass_min=min(subjets(4)["mpair"]),
        sj4_zsoft=subjets(4)["z"][3],
        sum_charge=sum(charge[i] for i in real),
        sum_e=sum(energy[i] for i in real),
        sum_pt=tot,
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top5=sum(pt[:5]),
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
        td0_13=pfeat(13, 'td0'),
        td0_14=pfeat(14, 'td0'),
        td0_17=pfeat(17, 'td0'),
        td0_2=pfeat(2, 'td0'),
        td0_23=pfeat(23, 'td0'),
        td0_3=pfeat(3, 'td0'),
        td0_30=pfeat(30, 'td0'),
        td0_33=pfeat(33, 'td0'),
        td0_38=pfeat(38, 'td0'),
        td0_44=pfeat(44, 'td0'),
        td0_7=pfeat(7, 'td0'),
        td0_9=pfeat(9, 'td0'),
        tdz_0=pfeat(0, 'tdz'),
        tdz_1=pfeat(1, 'tdz'),
        tdz_13=pfeat(13, 'tdz'),
        tdz_2=pfeat(2, 'tdz'),
        tdz_25=pfeat(25, 'tdz'),
        tdz_29=pfeat(29, 'tdz'),
        tdz_33=pfeat(33, 'tdz'),
        tdz_41=pfeat(41, 'tdz'),
        tdz_44=pfeat(44, 'tdz'),
        tdz_47=pfeat(47, 'tdz'),
        tdz_54=pfeat(54, 'tdz'),
        tdz_6=pfeat(6, 'tdz'),
        tdz_68=pfeat(68, 'tdz'),
        tdz_74=pfeat(74, 'tdz'),
        tdz_9=pfeat(9, 'tdz'),
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
        z_top10_slots=sum(pt[:10]) / tot,
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top3_slots=sum(pt[:3]) / tot,
        z_top5=sum(zs[:5]),
        z_top50_slots=sum(pt[:50]) / tot,
    )


def neuron_0(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.005192869
    )
    return z


def neuron_1(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.32527e-06
    )
    return z


def neuron_2(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.44731e-06
    )
    return z


def neuron_3(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.399862e-06
    )
    return z


def neuron_4(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.792155e-06
    )
    return z


def neuron_5(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.156869e-06
    )
    return z


def neuron_6(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.770101e-05
    )
    return z


def neuron_7(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.430821e-06
    )
    return z


def neuron_8(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.926455e-06
    )
    return z


def neuron_9(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.443661e-06
    )
    return z


def neuron_10(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.570106e-05
    )
    return z


def neuron_11(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.887716e-05
    )
    return z


def neuron_12(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.099316e-06
    )
    return z


def neuron_13(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.422903e-06
    )
    return z


def neuron_14(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.845306e-06
    )
    return z


def neuron_15(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.459517e-06
    )
    return z


def neuron_16(Q):
    # scale S = 18.69; each line: share * term / its average size
    z = 18.69182 * (0.2456079
        - 0.2021122 * Q.z_charged_had / 0.5066138   # -20.2%  z_charged_had
        - 0.1465522 * Q.z_neutral / 0.4037153   # -14.7%  z_neutral
        + 0.07026758 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # +7.0%  lep_ptrel < 27.32
        - 0.04400108 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # -4.4%  n_s3d_above_3 < 10
        - 0.04093184 * max(0.0, 0.2801368 - Q.z_displaced5) / 0.1998239   # -4.1%  z_displaced5 < 0.2801
        + 0.03545284 * Q.lep_ptrel / 7.315413   # +3.5%  lep_ptrel
        - 0.03019169 * max(0.0, 33.08364 - Q.sip_3d_3) / 19.9074   # -3.0%  sip_3d_3 < 33.08
        + 0.02551996 * max(0.0, 0.2801368 - Q.z_displaced5) * max(0.0, 33.08364 - Q.sip_3d_3) / 4.739402   # +2.6%  z_displaced5 < 0.2801 and sip_3d_3 < 33.08
        - 0.02443172 * max(0.0, Q.lep_z - 0.01245833) / 0.08062103   # -2.4%  lep_z > 0.01246
        + 0.02379798 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # +2.4%  sip_3d_2 < 226.3
        + 0.02318736 * max(0.0, 577.991 - Q.sip_3d_3) / 516.7039   # +2.3%  sip_3d_3 < 578
        + 0.0199897 * max(0.0, 0.2801368 - Q.z_displaced5) * max(0.0, 175.9957 - Q.sip_3d_3) / 30.21447   # +2.0%  z_displaced5 < 0.2801 and sip_3d_3 < 176
        + 0.01970019 * max(0.0, 29.0 - Q.n_charged_had) / 10.94929   # +2.0%  n_charged_had < 29
        + 0.01754541 * Q.n_sd0_above_2 / 4.174103   # +1.8%  n_sd0_above_2
        - 0.01578602 * Q.mass_charged / 64.25857   # -1.6%  mass_charged
        - 0.01310277 * max(0.0, 1.777286 - Q.mass_displaced3) / 0.7377974   # -1.3%  mass_displaced3 < 1.777
        - 0.0128742 * max(0.0, 87.9162 - Q.mass_neutral) * max(0.0, 0.06762785 - Q.M2_b2) / 1.669202   # -1.3%  mass_neutral < 87.92 and M2_b2 < 0.06763
        + 0.01234869 * Q.n_photon / 16.03902   # +1.2%  n_photon
        - 0.01225726 * max(0.0, 0.06245248 - Q.z_displaced5) / 0.02875704   # -1.2%  z_displaced5 < 0.06245
        - 0.01205075 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, 7.782712 - Q.D2_b2) / 31.51573   # -1.2%  n_s3d_above_3 < 10 and D2_b2 < 7.783
        + 0.01139437 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) / 7.619143   # +1.1%  n_pairs_kt_above_3 < 41
        + 0.01137054 * max(0.0, 0.09740996 - Q.M2) / 0.02324748   # +1.1%  M2 < 0.09741
        + 0.01078753 * max(0.0, 87.9162 - Q.mass_neutral) / 43.67158   # +1.1%  mass_neutral < 87.92
        + 0.009651542 * max(0.0, 4.41005 - Q.mass_displaced3) / 2.286034   # +1.0%  mass_displaced3 < 4.41
        - 0.008798737 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # -0.9%  sip_3d_2 < 447.1
        + 0.008718617 * max(0.0, 5.8125 - Q.max_abs_d0) * max(0.0, 49.03695 - Q.sip_3d_3) / 122.3704   # +0.9%  max_abs_d0 < 5.812 and sip_3d_3 < 49.04
        - 0.007878 * max(0.0, 5.0 - Q.n_sdz_above_2) / 1.969117   # -0.8%  n_sdz_above_2 < 5
        + 0.007051914 * max(0.0, Q.mass_neutral - 19.57354) / 26.6998   # +0.7%  mass_neutral > 19.57
        + 0.006866255 * max(0.0, Q.M3 - 0.02540381) / 0.01242824   # +0.7%  M3 > 0.0254
        - 0.006646688 * max(0.0, 16.23838 - Q.sip_3d_3) / 8.003297   # -0.7%  sip_3d_3 < 16.24
        + 0.006476197 * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0001737301   # +0.6%  e3_b2 < 0.0002537
        + 0.006370119 * max(0.0, 2.957031 - Q.max_abs_dz) / 1.247099   # +0.6%  max_abs_dz < 2.957
        + 0.005952817 * max(0.0, Q.sd_mass - 119.4443) / 9.070113   # +0.6%  sd_mass > 119.4
        + 0.005825989 * max(0.0, Q.sj3_pair_mass_max - 83.63398) / 16.36149   # +0.6%  sj3_pair_mass_max > 83.63
        - 0.004262387 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.4%  lep_ptrel > 43.21
        - 0.004230031 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # -0.4%  sd_mass > 154.6
        + 0.004203054 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +0.4%  n_pairs_kt_above_1 < 80
        + 0.003983339 * max(0.0, 87.9162 - Q.mass_neutral) * max(0.0, Q.n_lund - 5.0) / 240.7844   # +0.4%  mass_neutral < 87.92 and n_lund > 5
        - 0.003808763 * max(0.0, Q.lep_z - 0.221436) / 0.03510013   # -0.4%  lep_z > 0.2214
        - 0.003549936 * max(0.0, 1.763283 - Q.min_pair_mass) / 0.6216059   # -0.4%  min_pair_mass < 1.763
        + 0.003400191 * max(0.0, 13.03663 - Q.mass_displaced3) / 8.661134   # +0.3%  mass_displaced3 < 13.04
        - 0.003303611 * max(0.0, 0.6455078 - Q.max_abs_dz) / 0.1467826   # -0.3%  max_abs_dz < 0.6455
        + 0.003129732 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 1.43511   # +0.3%  n_dr_0p2_0p4 < 7
        + 0.003015943 * max(0.0, Q.C2 - 0.1323775) / 0.03527464   # +0.3%  C2 > 0.1324
        + 0.002820972 * max(0.0, 0.2490748 - Q.mass_displaced3) / 0.08578912   # +0.3%  mass_displaced3 < 0.2491
        + 0.002602141 * max(0.0, 0.2413007 - Q.C2_b05) / 0.01461377   # +0.3%  C2_b05 < 0.2413
        - 0.002470661 * max(0.0, 33.0 - Q.n_pt_above_1) / 3.50532   # -0.2%  n_pt_above_1 < 33
        + 0.002459219 * max(0.0, Q.z_top15_slots - 0.8008865) / 0.07132828   # +0.2%  z_top15_slots > 0.8009
        + 0.002381307 * max(0.0, Q.mass_top40 - 122.7145) / 8.484825   # +0.2%  mass_top40 > 122.7
        + 0.002337758 * max(0.0, Q.mass_top50 - 125.4927) / 8.989867   # +0.2%  mass_top50 > 125.5
        - 0.002333526 * max(0.0, Q.lund_max_lndelta - -1.062087) * max(0.0, 4.606241 - Q.sip_3d_3) / 0.160478   # -0.2%  lund_max_lndelta > -1.062 and sip_3d_3 < 4.606
        - 0.002308781 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # -0.2%  n_pairs_kt_above_3 < 28
        - 0.002278582 * max(0.0, 29.0 - Q.n_charged_had) * max(0.0, Q.ecf_g42 - 2.701529e-06) / 0.0002316724   # -0.2%  n_charged_had < 29 and ecf_g42 > 2.702e-06
        + 0.001886208 * max(0.0, Q.lund_max_lndelta - -1.062087) * max(0.0, 34.83233 - Q.sj4_pair_mass_min) / 3.204401   # +0.2%  lund_max_lndelta > -1.062 and sj4_pair_mass_min < 34.83
        + 0.001799626 * max(0.0, 87.9162 - Q.mass_neutral) * max(0.0, 6.0 - Q.n_dr_0p1_0p2) / 46.21754   # +0.2%  mass_neutral < 87.92 and n_dr_0p1_0p2 < 6
        - 0.00141507 * max(0.0, 0.02720087 - Q.tau4) / 0.003105367   # -0.1%  tau4 < 0.0272
        + 0.001342913 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, 15.0 - Q.n_dr_0p4_up) / 63.11649   # +0.1%  n_s3d_above_3 < 10 and n_dr_0p4_up < 15
        - 0.001242855 * max(0.0, Q.psi_0p3 - 0.9828705) / 0.002731628   # -0.1%  psi_0p3 > 0.9829
        - 0.00118297 * max(0.0, 0.06245248 - Q.z_displaced5) * max(0.0, 0.8786609 - Q.tau54) / 0.001586542   # -0.1%  z_displaced5 < 0.06245 and tau54 < 0.8787
        - 0.0011538 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # -0.1%  mass_top50 > 161.1
        - 0.001136323 * max(0.0, Q.tau3 - 0.06348273) / 0.006109561   # -0.1%  tau3 > 0.06348
        - 0.001070567 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.516502 - Q.D2) / 0.2767988   # -0.1%  n_dr_0p2_0p4 < 7 and D2 < 1.517
        + 0.001041364 * max(0.0, 87.9162 - Q.mass_neutral) * max(0.0, Q.mass_2charged - 17.49066) / 266.6911   # +0.1%  mass_neutral < 87.92 and mass_2charged > 17.49
        + 0.001038746 * max(0.0, Q.lund_max_lndelta - -1.062087) * max(0.0, 15.0 - Q.n_charged_had) / 0.2371327   # +0.1%  lund_max_lndelta > -1.062 and n_charged_had < 15
        + 0.0009180943 * max(0.0, 0.01556887 - Q.z_dr_0p05_0p1) / 0.002231501   # +0.1%  z_dr_0p05_0p1 < 0.01557
        - 0.0008641191 * max(0.0, 0.0265067 - Q.z_dr_0p4_up) / 0.01031536   # -0.1%  z_dr_0p4_up < 0.02651
        + 0.0008525399 * max(0.0, Q.lund_max_lndelta - -1.062087) / 0.1730972   # +0.1%  lund_max_lndelta > -1.062
        + 0.0008492486 * max(0.0, Q.mass_top5 - 62.84493) / 3.692784   # +0.1%  mass_top5 > 62.84
        + 0.0007664119 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.M3_b05 - 0.06007167) / 0.06230838   # +0.1%  n_pairs_kt_above_3 < 28 and M3_b05 > 0.06007
        + 0.0006154918 * max(0.0, -0.3599791 - Q.jet_charge_k05) / 0.04881883   # +0.1%  jet_charge_k05 < -0.36
        - 0.0005972177 * max(0.0, 92.68149 - Q.mass_top40) / 5.990263   # -0.1%  mass_top40 < 92.68
        - 0.0005402367 * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 1.107657   # -0.1%  n_lund_kt_above_5 > 1
        - 0.0004795532 * max(0.0, 87.9162 - Q.mass_neutral) * max(0.0, 0.02981375 - Q.dr_max_012) / 0.1323256   # -0.0%  mass_neutral < 87.92 and dr_max_012 < 0.02981
        - 0.0004646022 * max(0.0, Q.n_electron - 1.0) / 0.08167   # -0.0%  n_electron > 1
        + 0.0004457667 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # +0.0%  sd_mass > 175.9
        - 0.0004036405 * max(0.0, Q.max_pair_mass - 55.24488) / 0.6579942   # -0.0%  max_pair_mass > 55.24
        - 0.0003828522 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.2338497 - Q.lep_dr) / 0.6071668   # -0.0%  n_pairs_kt_above_3 < 28 and lep_dr < 0.2338
        + 0.0003599633 * max(0.0, Q.jet_charge_k05 - 0.5864519) / 0.01952642   # +0.0%  jet_charge_k05 > 0.5865
        + 0.0003355308 * max(0.0, 1.824785 - Q.sj3_mass2) / 0.06171879   # +0.0%  sj3_mass2 < 1.825
        + 0.0003189718 * max(0.0, Q.lep_z - 0.221436) * max(0.0, 0.7544983 - Q.tau32) / 0.006468695   # +0.0%  lep_z > 0.2214 and tau32 < 0.7545
        + 0.0003023013 * max(0.0, 3.452465 - Q.lne_4) / 0.07948689   # +0.0%  lne_4 < 3.452
        + 0.0002621785 * max(0.0, Q.n_pairs_kt_above_10 - 17.0) / 0.94113   # +0.0%  n_pairs_kt_above_10 > 17
        - 0.0001979646 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # -0.0%  max_abs_d0 < 5.812
        - 0.0001593376 * max(0.0, 0.302405 - Q.N2_b05) / 0.002473216   # -0.0%  N2_b05 < 0.3024
        + 0.0001502332 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, Q.d0err_12 - 0.0236969) / 0.5206052   # +0.0%  sip_3d_2 < 447.1 and d0err_12 > 0.0237
        + 0.0001190362 * max(0.0, 0.01556887 - Q.z_dr_0p05_0p1) * max(0.0, 0.1083984 - Q.eta_36) / 0.0003041121   # +0.0%  z_dr_0p05_0p1 < 0.01557 and eta_36 < 0.1084
        - 0.0001155105 * max(0.0, Q.mass_2photon - 22.18431) / 0.4406114   # -0.0%  mass_2photon > 22.18
        - 0.0001142122 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, Q.sip_3d_3 - 49.03695) / 133.8744   # -0.0%  n_lund_kt_above_5 > 1 and sip_3d_3 > 49.04
        + 9.023249e-05 * max(0.0, 29.0 - Q.n_charged_had) * max(0.0, Q.n_lepton - 1.0) / 1.919187   # +0.0%  n_charged_had < 29 and n_lepton > 1
        + 8.999808e-05 * max(0.0, -0.1170754 - Q.tdz_1) / 0.01260556   # +0.0%  tdz_1 < -0.1171
        - 4.5611e-05 * max(0.0, 13.03663 - Q.mass_displaced3) * max(0.0, Q.iselectron_1 - 0.0) / 0.1996362   # -0.0%  mass_displaced3 < 13.04 and iselectron_1 > 0
        - 4.155069e-05 * max(0.0, 0.7107791 - Q.D2) / 0.007444256   # -0.0%  D2 < 0.7108
        + 3.172083e-05 * max(0.0, 0.01556887 - Q.z_dr_0p05_0p1) * max(0.0, 0.3377343 - Q.tau32_b2) / 9.465289e-05   # +0.0%  z_dr_0p05_0p1 < 0.01557 and tau32_b2 < 0.3377
        + 3.368022e-06 * max(0.0, Q.tau3 - 0.06348273) * max(0.0, 0.0 - Q.phi_84) / 4.04563e-05   # +0.0%  tau3 > 0.06348 and phi_84 < 0
        - 1.440529e-06 * max(0.0, Q.mass_top50 - 125.4927) * max(0.0, 0.0 - Q.tdz_68) / 0.03907777   # -0.0%  mass_top50 > 125.5 and tdz_68 < 0
    )
    return z


def neuron_17(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.035355e-05
    )
    return z


def neuron_18(Q):
    # scale S = 11.15; each line: share * term / its average size
    z = 11.1464 * (0.05473269
        + 0.06838024 * Q.n_for_90pct / 20.33681   # +6.8%  n_for_90pct
        - 0.04784254 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -4.8%  lep_z < 0.2214
        - 0.04716119 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # -4.7%  n_s3d_above_3 < 3
        - 0.04011533 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # -4.0%  e3_b2 < 0.000791
        + 0.04000723 * max(0.0, 3.0 - Q.n_s3d_above_3) * max(0.0, 117.0891 - Q.sip_3d_2) / 97.96476   # +4.0%  n_s3d_above_3 < 3 and sip_3d_2 < 117.1
        + 0.03985906 * max(0.0, 9.0 - Q.n_sd0_above_3) / 5.844283   # +4.0%  n_sd0_above_3 < 9
        - 0.03934041 * max(0.0, 0.004135872 - Q.lep_z) / 0.002390373   # -3.9%  lep_z < 0.004136
        + 0.03195005 * max(0.0, 155.8928 - Q.mass_top40) / 48.12072   # +3.2%  mass_top40 < 155.9
        - 0.0285918 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.1046203 - Q.z_displaced5) / 0.01009417   # -2.9%  lep_z < 0.2214 and z_displaced5 < 0.1046
        + 0.02748789 * max(0.0, 27.0 - Q.n_charged_pt_above_1) / 8.874957   # +2.7%  n_charged_pt_above_1 < 27
        - 0.02597974 * max(0.0, 132.5189 - Q.mass_top40) / 28.13525   # -2.6%  mass_top40 < 132.5
        - 0.0236405 * max(0.0, Q.sd_mass - 123.2919) / 8.015693   # -2.4%  sd_mass > 123.3
        - 0.02209757 * max(0.0, Q.n_neutral - 12.0) / 8.788217   # -2.2%  n_neutral > 12
        + 0.0212654 * max(0.0, 2.097213e-06 - Q.e4) / 1.474625e-06   # +2.1%  e4 < 2.097e-06
        - 0.02097856 * max(0.0, 0.3829918 - Q.N2) / 0.08586977   # -2.1%  N2 < 0.383
        - 0.01898749 * max(0.0, 0.06416437 - Q.z_displaced3) / 0.02655242   # -1.9%  z_displaced3 < 0.06416
        + 0.01895261 * max(0.0, 2.0 - Q.n_lepton) / 1.440613   # +1.9%  n_lepton < 2
        - 0.01594768 * max(0.0, 6.0 - Q.n_sd0_above_2) / 2.514897   # -1.6%  n_sd0_above_2 < 6
        + 0.0151399 * max(0.0, 0.06245248 - Q.z_displaced5) / 0.02875704   # +1.5%  z_displaced5 < 0.06245
        - 0.01397744 * max(0.0, 114.4658 - Q.mass_top20) / 26.77686   # -1.4%  mass_top20 < 114.5
        + 0.01372365 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.3829918 - Q.N2) / 0.01292108   # +1.4%  lep_z < 0.2214 and N2 < 0.383
        + 0.01368784 * max(0.0, Q.mass_neutral - 14.92637) / 30.76836   # +1.4%  mass_neutral > 14.93
        + 0.01365953 * max(0.0, 49.03695 - Q.sip_3d_3) / 32.30721   # +1.4%  sip_3d_3 < 49.04
        - 0.0129959 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -1.3%  sip_3d_2 < 226.3
        + 0.01229269 * max(0.0, 6.0 - Q.n_s3d_above_10) * max(0.0, Q.lund_max_lndelta - -1.670992) / 2.1779   # +1.2%  n_s3d_above_10 < 6 and lund_max_lndelta > -1.671
        - 0.01222427 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, Q.N2_b2 - 0.0395449) / 2.959526   # -1.2%  lep_ptrel < 27.32 and N2_b2 > 0.03954
        + 0.01215693 * max(0.0, 0.0001729927 - Q.e3_b2) / 0.0001069897   # +1.2%  e3_b2 < 0.000173
        + 0.01211968 * max(0.0, 9.090532 - Q.mass_displaced3) * max(0.0, 29.0 - Q.n_charged_pt_above_1) / 66.69358   # +1.2%  mass_displaced3 < 9.091 and n_charged_pt_above_1 < 29
        + 0.01207547 * max(0.0, Q.sd_mass - 111.701) / 11.78149   # +1.2%  sd_mass > 111.7
        - 0.0119679 * max(0.0, Q.mass - 110.2019) / 16.73354   # -1.2%  mass > 110.2
        + 0.01171404 * max(0.0, 0.3829918 - Q.N2) * max(0.0, -5.306837 - Q.lnptrel_45) / 0.9709755   # +1.2%  N2 < 0.383 and lnptrel_45 < -5.307
        + 0.01109887 * max(0.0, 0.6289062 - Q.max_abs_d0) / 0.1808136   # +1.1%  max_abs_d0 < 0.6289
        - 0.01087484 * max(0.0, Q.tau43 - 0.6808537) / 0.1315878   # -1.1%  tau43 > 0.6809
        - 0.0104523 * max(0.0, 0.06245248 - Q.z_displaced5) * max(0.0, 4.636903 - Q.sip_3d_2) / 0.04573141   # -1.0%  z_displaced5 < 0.06245 and sip_3d_2 < 4.637
        + 0.01009758 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +1.0%  lep_iso < 0.4382
        - 0.01008372 * max(0.0, 6.0 - Q.n_s3d_above_10) / 3.631173   # -1.0%  n_s3d_above_10 < 6
        + 0.009969938 * max(0.0, 14.38858 - Q.lep_iso) / 12.08589   # +1.0%  lep_iso < 14.39
        - 0.009823417 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # -1.0%  lep_ptrel < 27.32
        + 0.009167845 * max(0.0, Q.mass - 164.4374) / 3.038568   # +0.9%  mass > 164.4
        + 0.008955623 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +0.9%  sd_mass > 154.6
        - 0.008616268 * max(0.0, Q.lund3_lndelta - -2.817283) * max(0.0, Q.tau54 - 0.7608692) / 0.1294652   # -0.9%  lund3_lndelta > -2.817 and tau54 > 0.7609
        + 0.00846284 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +0.8%  lep_iso < 1.362
        + 0.008365282 * max(0.0, 111.0 - Q.n_pairs_kt_above_3) / 47.95244   # +0.8%  n_pairs_kt_above_3 < 111
        + 0.008210099 * max(0.0, 3.0 - Q.n_sdz_above_5) / 1.349857   # +0.8%  n_sdz_above_5 < 3
        - 0.008146201 * max(0.0, Q.lund_max_lnkt - 3.22013) / 0.7220204   # -0.8%  lund_max_lnkt > 3.22
        + 0.008094978 * max(0.0, 0.06245248 - Q.z_displaced5) * max(0.0, Q.n_pt_above_5 - 10.0) / 0.3067365   # +0.8%  z_displaced5 < 0.06245 and n_pt_above_5 > 10
        - 0.007374938 * max(0.0, Q.n_pairs_kt_above_1 - 146.0) / 205.605   # -0.7%  n_pairs_kt_above_1 > 146
        - 0.006772153 * max(0.0, Q.tau32 - 0.5871757) / 0.1304199   # -0.7%  tau32 > 0.5872
        - 0.00598675 * max(0.0, Q.z_displaced5 - 0.006242101) / 0.08840268   # -0.6%  z_displaced5 > 0.006242
        - 0.00562276 * max(0.0, 9.090532 - Q.mass_displaced3) * max(0.0, 0.114659 - Q.lep_dr) / 0.430453   # -0.6%  mass_displaced3 < 9.091 and lep_dr < 0.1147
        - 0.004879205 * max(0.0, 3.373037 - Q.sip_3d_1) / 0.1802381   # -0.5%  sip_3d_1 < 3.373
        - 0.004855628 * max(0.0, 0.01091199 - Q.sum_z_dr2_top2) / 0.002905672   # -0.5%  sum_z_dr2_top2 < 0.01091
        + 0.004798529 * max(0.0, Q.N2_b05 - 0.4265629) / 0.03557833   # +0.5%  N2_b05 > 0.4266
        + 0.004379075 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +0.4%  n_dr_0p4_up < 4
        + 0.003979153 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # +0.4%  n_pairs_kt_above_1 < 366
        - 0.003811852 * max(0.0, 9.046875 - Q.max_abs_dz) / 5.120361   # -0.4%  max_abs_dz < 9.047
        + 0.003542474 * max(0.0, 6.0 - Q.n_s3d_above_10) * max(0.0, Q.lne_7 - 2.313463) / 3.324212   # +0.4%  n_s3d_above_10 < 6 and lne_7 > 2.313
        + 0.003495774 * max(0.0, 6.0 - Q.n_s3d_above_10) * max(0.0, 0.212136 - Q.z_neutral_had) / 0.3191128   # +0.3%  n_s3d_above_10 < 6 and z_neutral_had < 0.2121
        - 0.003345255 * max(0.0, Q.n_charged_had - 27.0) / 0.71757   # -0.3%  n_charged_had > 27
        - 0.00328026 * max(0.0, 0.4545826 - Q.N3_b2) / 0.1894157   # -0.3%  N3_b2 < 0.4546
        + 0.00324028 * max(0.0, Q.lund2_lndelta - -1.124734) / 0.274477   # +0.3%  lund2_lndelta > -1.125
        + 0.003120842 * max(0.0, 5.311506 - Q.sj2_mass2) / 0.3937533   # +0.3%  sj2_mass2 < 5.312
        + 0.002967389 * max(0.0, 0.004135872 - Q.lep_z) * max(0.0, Q.jet_abs_eta - 0.8317778) / 0.0004322819   # +0.3%  lep_z < 0.004136 and jet_abs_eta > 0.8318
        - 0.002909058 * max(0.0, Q.mass - 182.8592) / 1.688248   # -0.3%  mass > 182.9
        - 0.002779668 * max(0.0, 0.03624058 - Q.z_displaced3) / 0.01247855   # -0.3%  z_displaced3 < 0.03624
        - 0.002636459 * max(0.0, Q.sj3_pair_mass_max - 128.6605) / 2.418855   # -0.3%  sj3_pair_mass_max > 128.7
        - 0.002563083 * max(0.0, Q.lund_max_lndelta - -0.6897565) / 0.03634831   # -0.3%  lund_max_lndelta > -0.6898
        - 0.002553367 * max(0.0, Q.tau32 - 0.5871757) * max(0.0, 4.0 - Q.n_pt_above_50) / 0.1810898   # -0.3%  tau32 > 0.5872 and n_pt_above_50 < 4
        + 0.002380938 * max(0.0, 0.3206143 - Q.pt1_over_pt0) / 0.01867984   # +0.2%  pt1_over_pt0 < 0.3206
        - 0.002259181 * max(0.0, 0.03624058 - Q.z_displaced3) * max(0.0, Q.z_charged_had - 0.1891627) / 0.003966365   # -0.2%  z_displaced3 < 0.03624 and z_charged_had > 0.1892
        - 0.002201208 * max(0.0, 2.097213e-06 - Q.e4) * max(0.0, 1.362094 - Q.lep_iso) / 1.442174e-06   # -0.2%  e4 < 2.097e-06 and lep_iso < 1.362
        - 0.002086841 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.jet_charge_k05 - 0.1901233) / 0.01534374   # -0.2%  lep_z < 0.2214 and jet_charge_k05 > 0.1901
        - 0.001917174 * max(0.0, 0.07290954 - Q.sum_z_dr) / 0.002660742   # -0.2%  sum_z_dr < 0.07291
        - 0.00184659 * max(0.0, Q.sj3_pair_mass_min - 44.1643) / 6.009845   # -0.2%  sj3_pair_mass_min > 44.16
        - 0.001606701 * max(0.0, 0.3084865 - Q.sj3_dr23) / 0.04356127   # -0.2%  sj3_dr23 < 0.3085
        - 0.001268971 * max(0.0, 9.090532 - Q.mass_displaced3) / 5.595081   # -0.1%  mass_displaced3 < 9.091
        + 0.001021013 * max(0.0, 0.3887012 - Q.z_charged) / 0.008149801   # +0.1%  z_charged < 0.3887
        - 0.0007858852 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.380649 - Q.pt1_over_pt0) / 0.002473162   # -0.1%  lep_z < 0.2214 and pt1_over_pt0 < 0.3806
        - 0.000748003 * max(0.0, Q.mass_top5 - 76.4159) / 1.748522   # -0.1%  mass_top5 > 76.42
        - 0.0007215618 * max(0.0, Q.n_lund - 13.0) / 0.37363   # -0.1%  n_lund > 13
        - 0.0006843399 * max(0.0, Q.sj3_pair_mass_max - 128.6605) * max(0.0, Q.eccentricity - 0.9132117) / 0.02034252   # -0.1%  sj3_pair_mass_max > 128.7 and eccentricity > 0.9132
        - 0.0006758788 * max(0.0, Q.sj3_pairmax_over_m - 0.897453) / 0.007530885   # -0.1%  sj3_pairmax_over_m > 0.8975
        + 0.0005885552 * max(0.0, Q.mass - 182.8592) * max(0.0, 0.4623202 - Q.planar_flow) / 0.1098624   # +0.1%  mass > 182.9 and planar_flow < 0.4623
        + 0.0004974783 * max(0.0, Q.mass_charged - 113.5019) / 1.042604   # +0.0%  mass_charged > 113.5
        + 0.000494179 * max(0.0, Q.lund3_lndelta - -2.817283) / 1.325774   # +0.0%  lund3_lndelta > -2.817
        + 0.0004465427 * max(0.0, Q.z_displaced5 - 0.006242101) * max(0.0, 447.0873 - Q.sip_3d_2) / 24.18806   # +0.0%  z_displaced5 > 0.006242 and sip_3d_2 < 447.1
        + 0.0004317059 * max(0.0, 0.2598003 - Q.psi_0p2) / 0.007656233   # +0.0%  psi_0p2 < 0.2598
        + 0.0002837853 * max(0.0, Q.tau3 - 0.05398263) / 0.008920324   # +0.0%  tau3 > 0.05398
        - 0.0002519552 * max(0.0, Q.sj3_pair_mass_max - 128.6605) * max(0.0, -0.3624079 - Q.lund1_lndelta) / 0.1198998   # -0.0%  sj3_pair_mass_max > 128.7 and lund1_lndelta < -0.3624
        + 0.0002419085 * max(0.0, Q.z_displaced5 - 0.006242101) * max(0.0, 0.2116473 - Q.dr12) / 0.01073306   # +0.0%  z_displaced5 > 0.006242 and dr12 < 0.2116
        - 0.0002210445 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.e4_b05 - 4.363663e-05) / 2.737046e-06   # -0.0%  lep_z < 0.2214 and e4_b05 > 4.364e-05
        + 0.0001714881 * max(0.0, Q.z_displaced5 - 0.006242101) * max(0.0, 29.0 - Q.n_dr_0p2_0p4) / 1.555704   # +0.0%  z_displaced5 > 0.006242 and n_dr_0p2_0p4 < 29
        + 0.0001631549 * max(0.0, Q.n_neutral - 12.0) * max(0.0, Q.eta_28 - -0.1078491) / 1.228715   # +0.0%  n_neutral > 12 and eta_28 > -0.1078
        - 0.0001330299 * max(0.0, 2.409613 - Q.mass_displaced3) / 1.070706   # -0.0%  mass_displaced3 < 2.41
        + 9.075922e-05 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, Q.eta_9 - 0.09516602) / 0.4726864   # +0.0%  lep_ptrel < 27.32 and eta_9 > 0.09517
        - 6.4131e-05 * max(0.0, Q.sj2_mass2 - 42.12131) / 0.6609289   # -0.0%  sj2_mass2 > 42.12
        + 5.686104e-05 * max(0.0, 0.3887012 - Q.z_charged) * max(0.0, Q.mass_2photon - 1.081989) / 0.06983535   # +0.0%  z_charged < 0.3887 and mass_2photon > 1.082
        + 1.661431e-05 * max(0.0, -0.2019043 - Q.eta_1) / 0.004072277   # +0.0%  eta_1 < -0.2019
        + 7.347349e-06 * max(0.0, Q.mass - 110.2019) * max(0.0, 8.0 - Q.n_charged_pt_above_10) / 11.56508   # +0.0%  mass > 110.2 and n_charged_pt_above_10 < 8
        - 8.874643e-07 * max(0.0, Q.sj3_pairmax_over_m - 0.897453) * max(0.0, Q.iselectron_29 - 0.0) / 2.406724e-05   # -0.0%  sj3_pairmax_over_m > 0.8975 and iselectron_29 > 0
    )
    return z


def neuron_19(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.407883e-06
    )
    return z


def neuron_20(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.253655e-07
    )
    return z


def neuron_21(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.001686459
    )
    return z


def neuron_22(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.044882e-06
    )
    return z


def neuron_23(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.979496e-07
    )
    return z


def neuron_24(Q):
    # scale S = 14.81; each line: share * term / its average size
    z = 14.80914 * (-0.03577832
        - 0.07253247 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # -7.3%  sd_mass > 88.82
        + 0.06001335 * max(0.0, Q.sd_mass - 62.03827) / 42.41619   # +6.0%  sd_mass > 62.04
        - 0.05502601 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -5.5%  lep_z < 0.2214
        - 0.04526759 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -4.5%  mass < 164.4
        - 0.04167463 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # -4.2%  e3_b2 < 0.0004127
        + 0.03852929 * Q.pair_max_lnkt / 2.706794   # +3.9%  pair_max_lnkt
        + 0.03556567 * max(0.0, 1.33105 - Q.N3_b05) / 0.6024977   # +3.6%  N3_b05 < 1.331
        + 0.03191699 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +3.2%  lep_iso < 1.362
        - 0.03180365 * max(0.0, 6.0 - Q.n_sd0_above_3) / 3.16582   # -3.2%  n_sd0_above_3 < 6
        - 0.02973573 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -3.0%  n_lepton < 1
        + 0.02837328 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +2.8%  lep_iso < 0.4382
        + 0.02810244 * max(0.0, 114.0172 - Q.mass) / 13.82223   # +2.8%  mass < 114
        + 0.02307661 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) / 36.55147   # +2.3%  sj4_pair_mass_max < 115.5
        + 0.02292297 * max(0.0, 0.5944498 - Q.sj2_dr) * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 2.50037   # +2.3%  sj2_dr < 0.5944 and n_dr_0p2_0p4 < 21
        + 0.02125869 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.lund2_lndelta - -1.411949) / 21.06341   # +2.1%  mass < 164.4 and lund2_lndelta > -1.412
        + 0.02056007 * max(0.0, 26.78691 - Q.mass_displaced3) / 20.43083   # +2.1%  mass_displaced3 < 26.79
        + 0.0194972 * max(0.0, Q.sd_mass - 119.4443) / 9.070113   # +1.9%  sd_mass > 119.4
        - 0.01904915 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # -1.9%  lep_z < 0.3397
        + 0.01893248 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 127.2317 - Q.mass) / 5.805704   # +1.9%  lep_z < 0.3397 and mass < 127.2
        - 0.01769842 * max(0.0, Q.n_real_top30 - 21.0) / 7.215303   # -1.8%  n_real_top30 > 21
        + 0.0169463 * max(0.0, 134.2224 - Q.mass_top30) / 33.8221   # +1.7%  mass_top30 < 134.2
        + 0.01634802 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) / 59.61773   # +1.6%  n_pairs_kt_above_3 < 126
        + 0.01580047 * Q.sum_e / 905.8561   # +1.6%  sum_e
        - 0.01529381 * max(0.0, 0.3923158 - Q.sj2_dr) / 0.06287263   # -1.5%  sj2_dr < 0.3923
        - 0.01529311 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.lund2_lndelta - -1.755318) / 9.158987   # -1.5%  mass < 117.5 and lund2_lndelta > -1.755
        - 0.01449443 * Q.n_sd0_above_2 / 4.174103   # -1.4%  n_sd0_above_2
        + 0.01440783 * max(0.0, 0.3995332 - Q.sd_rg) / 0.1037053   # +1.4%  sd_rg < 0.3995
        + 0.01262295 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) / 103.3282   # +1.3%  n_pairs_kt_above_1 < 325
        - 0.0126004 * max(0.0, 0.9644868 - Q.N3_b2) / 0.6211118   # -1.3%  N3_b2 < 0.9645
        + 0.01187945 * max(0.0, 0.1061578 - Q.z_displaced3) / 0.05189409   # +1.2%  z_displaced3 < 0.1062
        + 0.01027461 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +1.0%  lep_ptrel < 43.21
        - 0.009645454 * max(0.0, Q.C2 - 0.07037463) / 0.0805072   # -1.0%  C2 > 0.07037
        - 0.008893869 * max(0.0, Q.n_s3d_above_3 - 2.0) / 2.23187   # -0.9%  n_s3d_above_3 > 2
        + 0.008116607 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +0.8%  n_dr_0p4_up < 4
        + 0.007794752 * max(0.0, 2.0 - Q.n_s3d_above_3) / 0.4717633   # +0.8%  n_s3d_above_3 < 2
        - 0.00764532 * max(0.0, 117.4867 - Q.mass) / 15.64803   # -0.8%  mass < 117.5
        + 0.006552089 * max(0.0, 0.04544279 - Q.M2_b2) / 0.01635302   # +0.7%  M2_b2 < 0.04544
        - 0.006164887 * max(0.0, Q.z_top15_slots - 0.8558319) / 0.03951754   # -0.6%  z_top15_slots > 0.8558
        + 0.006111968 * max(0.0, 117.4867 - Q.mass) * max(0.0, 3.038341 - Q.D2) / 14.74609   # +0.6%  mass < 117.5 and D2 < 3.038
        - 0.005964599 * max(0.0, 75.78208 - Q.sj4_pair_mass_max) / 8.111726   # -0.6%  sj4_pair_mass_max < 75.78
        - 0.005885558 * max(0.0, 0.258375 - Q.N2) / 0.01851814   # -0.6%  N2 < 0.2584
        - 0.0058802 * max(0.0, Q.n_s3d_above_10 - 1.0) / 1.904323   # -0.6%  n_s3d_above_10 > 1
        - 0.005872209 * max(0.0, Q.lund2_lndelta - -1.299999) / 0.3974116   # -0.6%  lund2_lndelta > -1.3
        + 0.005635978 * max(0.0, Q.n_sd0_above_3 - 3.0) / 1.31993   # +0.6%  n_sd0_above_3 > 3
        - 0.005467088 * max(0.0, 79.47361 - Q.mass) / 2.857843   # -0.5%  mass < 79.47
        - 0.005454436 * max(0.0, Q.psi_0p2 - 0.8857951) / 0.01931927   # -0.5%  psi_0p2 > 0.8858
        + 0.005103836 * max(0.0, 0.5944498 - Q.sj2_dr) / 0.2163159   # +0.5%  sj2_dr < 0.5944
        - 0.005096375 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # -0.5%  max_abs_d0 < 10.52
        + 0.00482623 * max(0.0, 0.2403736 - Q.sj2_dr) / 0.01094304   # +0.5%  sj2_dr < 0.2404
        - 0.004782635 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.2403736 - Q.sj2_dr) / 0.001837088   # -0.5%  lep_z < 0.2214 and sj2_dr < 0.2404
        - 0.004451353 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.4%  mass < 90.09
        + 0.004336528 * max(0.0, 6.0 - Q.n_sd0_above_3) * max(0.0, Q.n_lund_kt_above_1 - 4.0) / 4.546673   # +0.4%  n_sd0_above_3 < 6 and n_lund_kt_above_1 > 4
        - 0.003986384 * max(0.0, 0.07094338 - Q.C2_b2) / 0.02378602   # -0.4%  C2_b2 < 0.07094
        + 0.003797747 * max(0.0, Q.n_s3d_above_3 - 5.0) / 0.8834567   # +0.4%  n_s3d_above_3 > 5
        - 0.003401435 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 5.969698e-05 - Q.ecf_g42) / 1.977438e-06   # -0.3%  z_displaced3 < 0.1062 and ecf_g42 < 5.97e-05
        + 0.003287114 * max(0.0, 149.0507 - Q.mass) / 39.08704   # +0.3%  mass < 149.1
        - 0.003030059 * max(0.0, 74.51927 - Q.mass_top30) / 2.669461   # -0.3%  mass_top30 < 74.52
        + 0.002777483 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, 0.3972884 - Q.pt_balance01) / 1.831735e-05   # +0.3%  e3_b2 < 0.0004127 and pt_balance01 < 0.3973
        - 0.002727698 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, Q.D2 - 2.446407) / 0.03497164   # -0.3%  z_displaced3 < 0.1062 and D2 > 2.446
        - 0.002709246 * max(0.0, Q.N2_b05 - 0.4599592) / 0.01884027   # -0.3%  N2_b05 > 0.46
        - 0.002333557 * max(0.0, 26.78691 - Q.mass_displaced3) * max(0.0, Q.z_charged - 0.4622094) / 3.130524   # -0.2%  mass_displaced3 < 26.79 and z_charged > 0.4622
        + 0.002188888 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.M3 - 0.03259227) / 0.2883986   # +0.2%  lep_ptrel < 43.21 and M3 > 0.03259
        - 0.002063998 * max(0.0, 0.8799072 - Q.sj3_dr23) / 0.4603866   # -0.2%  sj3_dr23 < 0.8799
        + 0.002045529 * max(0.0, 26.78691 - Q.mass_displaced3) * max(0.0, 0.06777369 - Q.dr_31) / 0.5598919   # +0.2%  mass_displaced3 < 26.79 and dr_31 < 0.06777
        + 0.001835074 * max(0.0, 0.3886647 - Q.sj3_mass3) / 0.03562942   # +0.2%  sj3_mass3 < 0.3887
        + 0.001766222 * max(0.0, 0.5068038 - Q.tau32) / 0.02583179   # +0.2%  tau32 < 0.5068
        - 0.001448681 * max(0.0, 74.51927 - Q.mass_top30) * max(0.0, 3.038341 - Q.D2) / 1.107918   # -0.1%  mass_top30 < 74.52 and D2 < 3.038
        - 0.001432068 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.n_muon - 0.0) / 5.859253   # -0.1%  lep_ptrel < 43.21 and n_muon > 0
        - 0.001334899 * max(0.0, Q.mass_displaced3 - 39.09615) / 0.6617151   # -0.1%  mass_displaced3 > 39.1
        - 0.001076498 * max(0.0, 0.3945478 - Q.z_charged_had) / 0.03037343   # -0.1%  z_charged_had < 0.3945
        + 0.001016092 * max(0.0, 90.08945 - Q.mass) * max(0.0, Q.lund2_lndelta - -2.073049) / 3.81955   # +0.1%  mass < 90.09 and lund2_lndelta > -2.073
        - 0.0008293614 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # -0.1%  sd_mass > 175.9
        + 0.0007512929 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.mass_charged - 90.44234) / 0.0003362549   # +0.1%  e3_b2 < 0.0004127 and mass_charged > 90.44
        + 0.0006402715 * max(0.0, Q.sj3_pair_mass_max - 142.9952) / 1.384637   # +0.1%  sj3_pair_mass_max > 143
        - 0.0005486127 * max(0.0, Q.sj3_dr_min - 0.3526989) / 0.01394539   # -0.1%  sj3_dr_min > 0.3527
        + 0.0005411652 * max(0.0, 74.51927 - Q.mass_top30) * max(0.0, Q.n_muon - 0.0) / 0.3416725   # +0.1%  mass_top30 < 74.52 and n_muon > 0
        + 0.0004616498 * max(0.0, Q.mass_charged - 99.20396) / 2.066138   # +0.0%  mass_charged > 99.2
        + 0.0004281889 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.e4 - 5.29882e-07) / 3.445168e-07   # +0.0%  lep_z < 0.3397 and e4 > 5.299e-07
        - 0.0004135638 * max(0.0, 0.4044054 - Q.D2_b2) / 0.01564934   # -0.0%  D2_b2 < 0.4044
        + 0.0004043178 * max(0.0, Q.mass_top5 - 68.52153) / 2.713876   # +0.0%  mass_top5 > 68.52
        - 0.000392612 * max(0.0, 1.0 - Q.isnhad_1) / 0.82048   # -0.0%  isnhad_1 < 1
        + 0.0003897036 * max(0.0, Q.n_sd0_above_10 - 6.0) / 0.15998   # +0.0%  n_sd0_above_10 > 6
        + 0.0003473757 * max(0.0, 1.262841 - Q.mass_displaced5) / 0.5499328   # +0.0%  mass_displaced5 < 1.263
        + 0.0003413326 * max(0.0, 2.0 - Q.n_sdz_above_5) / 0.7314567   # +0.0%  n_sdz_above_5 < 2
        + 0.000281786 * max(0.0, 0.4680886 - Q.tau32_b2) / 0.05978538   # +0.0%  tau32_b2 < 0.4681
        - 0.0002699923 * max(0.0, 0.2083105 - Q.sj2_dr) / 0.00695417   # -0.0%  sj2_dr < 0.2083
        + 0.0002659467 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.min_pair_mass - 10.59762) / 7.086748e-05   # +0.0%  e3_b2 < 0.0004127 and min_pair_mass > 10.6
        + 0.0002559215 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, Q.mass_2photon - 4.18513) / 0.1670932   # +0.0%  z_displaced3 < 0.1062 and mass_2photon > 4.185
        - 0.0001938154 * max(0.0, Q.sj3_dr_min - 0.2410564) / 0.04091582   # -0.0%  sj3_dr_min > 0.2411
        + 0.0001851754 * max(0.0, 2.0 - Q.n_sd0_above_10) / 0.7503067   # +0.0%  n_sd0_above_10 < 2
        + 0.0001512323 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.tdz_6 - 0.03675711) / 1.407383   # +0.0%  mass < 164.4 and tdz_6 > 0.03676
        + 0.0001178808 * max(0.0, 0.4044054 - Q.D2_b2) * max(0.0, Q.dr_31 - 0.02273044) / 0.001130568   # +0.0%  D2_b2 < 0.4044 and dr_31 > 0.02273
        + 0.000112629 * max(0.0, Q.tdz_2 - -0.03532465) / 0.05855813   # +0.0%  tdz_2 > -0.03532
        - 8.82053e-05 * max(0.0, 0.01040452 - Q.sum_z_dr2_top15) / 0.0007317661   # -0.0%  sum_z_dr2_top15 < 0.0104
        - 7.187271e-05 * max(0.0, Q.sd_mass - 88.81751) * max(0.0, 0.0 - Q.charge_0) / 6.688685   # -0.0%  sd_mass > 88.82 and charge_0 < 0
        - 3.10796e-05 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.dr_10 - 0.3349968) / 0.7314664   # -0.0%  lep_ptrel < 43.21 and dr_10 > 0.335
        + 2.766412e-05 * max(0.0, Q.n_sd0_above_3 - 3.0) * max(0.0, Q.d0err_42 - 0.0) / 0.01390804   # +0.0%  n_sd0_above_3 > 3 and d0err_42 > 0
        - 1.340517e-05 * max(0.0, Q.mass_charged - 99.20396) * max(0.0, Q.lnerel_82 - -18.42068) / 1.5218   # -0.0%  mass_charged > 99.2 and lnerel_82 > -18.42
        - 4.538937e-06 * max(0.0, 75.78208 - Q.sj4_pair_mass_max) * max(0.0, Q.iselectron_24 - 0.0) / 0.02893548   # -0.0%  sj4_pair_mass_max < 75.78 and iselectron_24 > 0
        + 6.731305e-07 * max(0.0, Q.lund2_lndelta - -1.299999) * max(0.0, -0.02698624 - Q.tdz_0) / 0.005244564   # +0.0%  lund2_lndelta > -1.3 and tdz_0 < -0.02699
    )
    return z


def neuron_25(Q):
    # scale S = 4.258; each line: share * term / its average size
    z = 4.257951 * (0.0620604
        + 0.08986669 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # +9.0%  sd_mass > 78.48
        - 0.0621988 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # -6.2%  lep_z < 0.3397
        - 0.05786344 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # -5.8%  sd_mass > 94.56
        - 0.05044647 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # -5.0%  lep_ptrel < 27.32
        - 0.04009409 * max(0.0, Q.mass_top5 - 29.68311) / 17.64916   # -4.0%  mass_top5 > 29.68
        - 0.03120969 * max(0.0, Q.sd_mass - 42.31629) / 58.72614   # -3.1%  sd_mass > 42.32
        - 0.03106527 * max(0.0, Q.lund_max_lndelta - -2.071997) / 1.011402   # -3.1%  lund_max_lndelta > -2.072
        - 0.03062198 * max(0.0, 2.907433 - Q.lep_iso) / 2.168962   # -3.1%  lep_iso < 2.907
        + 0.0305687 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +3.1%  lep_iso < 1.362
        + 0.0292584 * max(0.0, Q.sd_mass - 115.7091) / 10.28199   # +2.9%  sd_mass > 115.7
        - 0.02613267 * max(0.0, 3.0 - Q.n_sd0_above_3) / 1.04769   # -2.6%  n_sd0_above_3 < 3
        + 0.02339527 * max(0.0, 182.8592 - Q.mass) / 69.61635   # +2.3%  mass < 182.9
        + 0.02107677 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.mass_top5 - 37.66803) / 2.651954   # +2.1%  lep_z < 0.3397 and mass_top5 > 37.67
        + 0.02068449 * max(0.0, 0.1336727 - Q.tau2) / 0.06026045   # +2.1%  tau2 < 0.1337
        - 0.020422 * max(0.0, Q.sd_mass - 100.8525) / 16.67179   # -2.0%  sd_mass > 100.9
        + 0.02031929 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) / 26.86577   # +2.0%  sj3_pair_mass_max < 114.8
        + 0.01994604 * Q.n_dr_0p2_0p4 / 11.50196   # +2.0%  n_dr_0p2_0p4
        - 0.01859829 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # -1.9%  sd_mass > 88.82
        + 0.01843053 * max(0.0, Q.lund_max_lndelta - -1.303913) / 0.3311482   # +1.8%  lund_max_lndelta > -1.304
        + 0.01832838 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 19.0 - Q.n_dr_0p4_up) / 3.559354   # +1.8%  lep_z < 0.3397 and n_dr_0p4_up < 19
        + 0.01735448 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 13.10191 - Q.sj3_mass3) / 2.01323   # +1.7%  lep_z < 0.3397 and sj3_mass3 < 13.1
        + 0.01547281 * max(0.0, 0.04607888 - Q.lep_dr) * max(0.0, 3.157646 - Q.sip_3d_3) / 0.01499378   # +1.5%  lep_dr < 0.04608 and sip_3d_3 < 3.158
        - 0.01372393 * max(0.0, 0.6200816 - Q.tau21) / 0.2103839   # -1.4%  tau21 < 0.6201
        + 0.01351537 * max(0.0, 9.0 - Q.n_sd0_above_2) / 5.032627   # +1.4%  n_sd0_above_2 < 9
        + 0.01180678 * max(0.0, 7.075642 - Q.pair_max_lnm2) / 0.6225895   # +1.2%  pair_max_lnm2 < 7.076
        + 0.01163829 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +1.2%  mass < 95.15
        - 0.01137135 * max(0.0, 2.097213e-06 - Q.e4) / 1.474625e-06   # -1.1%  e4 < 2.097e-06
        + 0.01131312 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # +1.1%  n_pairs_kt_above_3 < 28
        - 0.01105656 * max(0.0, 2.0 - Q.n_sdz_above_5) / 0.7314567   # -1.1%  n_sdz_above_5 < 2
        - 0.01094546 * max(0.0, 0.2601475 - Q.sd_rg) / 0.03754888   # -1.1%  sd_rg < 0.2601
        - 0.01094087 * max(0.0, 105.7234 - Q.mass) * max(0.0, 1.0 - Q.n_electron) / 7.831057   # -1.1%  mass < 105.7 and n_electron < 1
        - 0.01039934 * max(0.0, 105.7234 - Q.mass) / 10.09077   # -1.0%  mass < 105.7
        + 0.01033708 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.sj3_pairmax_over_m - 0.6627397) / 0.0401834   # +1.0%  lep_z < 0.3397 and sj3_pairmax_over_m > 0.6627
        - 0.01027384 * max(0.0, 0.3237313 - Q.z_dr_0_0p05) / 0.2281116   # -1.0%  z_dr_0_0p05 < 0.3237
        - 0.00917293 * max(0.0, 0.0417856 - Q.M2_b2) * max(0.0, 23.0 - Q.n_pairs_kt_above_10) / 0.2397464   # -0.9%  M2_b2 < 0.04179 and n_pairs_kt_above_10 < 23
        - 0.009120811 * max(0.0, 101.379 - Q.sj3_pair_mass_max) / 17.08376   # -0.9%  sj3_pair_mass_max < 101.4
        + 0.009054617 * max(0.0, 8.0 - Q.n_s3d_above_10) / 5.470973   # +0.9%  n_s3d_above_10 < 8
        + 0.008058936 * max(0.0, Q.sd_mass - 119.4443) / 9.070113   # +0.8%  sd_mass > 119.4
        - 0.007830829 * max(0.0, 0.0417856 - Q.M2_b2) * max(0.0, 0.4232894 - Q.z_displaced3) / 0.00410717   # -0.8%  M2_b2 < 0.04179 and z_displaced3 < 0.4233
        + 0.007201804 * max(0.0, Q.sj4_pair_mass_max - 69.83554) / 16.93977   # +0.7%  sj4_pair_mass_max > 69.84
        + 0.006832348 * max(0.0, 0.02600452 - Q.z_displaced3) / 0.00810998   # +0.7%  z_displaced3 < 0.026
        + 0.006449303 * max(0.0, 18.0 - Q.n_neutral) / 2.453357   # +0.6%  n_neutral < 18
        + 0.006356183 * max(0.0, 3.373037 - Q.sip_3d_1) / 0.1802381   # +0.6%  sip_3d_1 < 3.373
        - 0.005531883 * max(0.0, Q.sd_mass - 127.8042) / 7.009137   # -0.6%  sd_mass > 127.8
        - 0.005335599 * max(0.0, 0.5010328 - Q.z_top5) / 0.03266783   # -0.5%  z_top5 < 0.501
        + 0.005193944 * max(0.0, Q.lund_max_lndelta - -1.303913) * max(0.0, 0.02931273 - Q.dr_55) / 0.008045868   # +0.5%  lund_max_lndelta > -1.304 and dr_55 < 0.02931
        - 0.005134782 * max(0.0, 0.1088678 - Q.D3_b2) / 0.04873824   # -0.5%  D3_b2 < 0.1089
        - 0.005059696 * Q.sum_z_dr2_top20 / 0.03374663   # -0.5%  sum_z_dr2_top20
        + 0.004488203 * max(0.0, Q.pair_mean_lnm2 - 3.654683) / 0.143116   # +0.4%  pair_mean_lnm2 > 3.655
        - 0.004459505 * max(0.0, 79.47361 - Q.mass) / 2.857843   # -0.4%  mass < 79.47
        - 0.00445588 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # -0.4%  sj3_pair_mass_max < 73.25
        - 0.004310706 * max(0.0, 0.5068038 - Q.tau32) / 0.02583179   # -0.4%  tau32 < 0.5068
        + 0.004183232 * max(0.0, 0.6200816 - Q.tau21) * max(0.0, 1343.246 - Q.sip_3d_1) / 204.4871   # +0.4%  tau21 < 0.6201 and sip_3d_1 < 1343
        + 0.004086369 * max(0.0, Q.lund_max_lndelta - -1.303913) * max(0.0, 0.3112717 - Q.sj4_dr_min) / 0.05864771   # +0.4%  lund_max_lndelta > -1.304 and sj4_dr_min < 0.3113
        + 0.003996873 * max(0.0, Q.n_charged_pt_above_1 - 10.0) / 9.07124   # +0.4%  n_charged_pt_above_1 > 10
        - 0.003587617 * max(0.0, 0.6200816 - Q.tau21) * max(0.0, 65.0404 - Q.mass_charged) / 2.021854   # -0.4%  tau21 < 0.6201 and mass_charged < 65.04
        + 0.003121667 * max(0.0, Q.sj2_mass1 - 77.42768) / 2.022907   # +0.3%  sj2_mass1 > 77.43
        + 0.002253248 * max(0.0, 0.2116473 - Q.dr12) / 0.117423   # +0.2%  dr12 < 0.2116
        + 0.002227164 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # +0.2%  lep_ptrel > 43.21
        + 0.001960334 * max(0.0, 2.097213e-06 - Q.e4) * max(0.0, Q.mass_2photon - 12.47889) / 1.519147e-06   # +0.2%  e4 < 2.097e-06 and mass_2photon > 12.48
        + 0.001947346 * max(0.0, 0.0417856 - Q.M2_b2) / 0.01370017   # +0.2%  M2_b2 < 0.04179
        - 0.001816883 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) * max(0.0, Q.lne_8 - 3.172177) / 4.06093   # -0.2%  sj3_pair_mass_max < 114.8 and lne_8 > 3.172
        - 0.001783031 * max(0.0, 0.6200816 - Q.tau21) * max(0.0, 7.0 - Q.n_lund_kt_above_1) / 0.5148103   # -0.2%  tau21 < 0.6201 and n_lund_kt_above_1 < 7
        - 0.001756023 * max(0.0, 0.01777495 - Q.sum_z_dr2_top3) / 0.00589586   # -0.2%  sum_z_dr2_top3 < 0.01777
        - 0.00175232 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # -0.2%  sd_mass > 175.9
        + 0.001617329 * max(0.0, 6.341631 - Q.mass_displaced3) / 3.595234   # +0.2%  mass_displaced3 < 6.342
        - 0.001475674 * max(0.0, Q.sj3_pair_mass_max - 128.6605) / 2.418855   # -0.1%  sj3_pair_mass_max > 128.7
        + 0.001343613 * max(0.0, Q.sj2_mass2 - 42.12131) / 0.6609289   # +0.1%  sj2_mass2 > 42.12
        + 0.001253004 * max(0.0, 101.379 - Q.sj3_pair_mass_max) * max(0.0, Q.lne_4 - 3.378014) / 7.70898   # +0.1%  sj3_pair_mass_max < 101.4 and lne_4 > 3.378
        - 0.001198038 * max(0.0, 101.379 - Q.sj3_pair_mass_max) * max(0.0, Q.sd_zg - 0.1728409) / 1.820907   # -0.1%  sj3_pair_mass_max < 101.4 and sd_zg > 0.1728
        - 0.001136664 * max(0.0, 0.6200816 - Q.tau21) * max(0.0, Q.z_photon - 0.3090294) / 0.006821502   # -0.1%  tau21 < 0.6201 and z_photon > 0.309
        + 0.001092306 * max(0.0, Q.C2 - 0.2687133) / 0.002106492   # +0.1%  C2 > 0.2687
        + 0.001052594 * max(0.0, 90.08945 - Q.mass) / 4.972097   # +0.1%  mass < 90.09
        + 0.0009369045 * max(0.0, 21.21062 - Q.sj2_mass1) / 1.735779   # +0.1%  sj2_mass1 < 21.21
        - 0.0009253717 * max(0.0, Q.lep_ptrel - 43.20788) * max(0.0, 3.53193e-06 - Q.e4) / 1.344298e-06   # -0.1%  lep_ptrel > 43.21 and e4 < 3.532e-06
        + 0.0009020602 * max(0.0, Q.z_displaced3 - 0.4232894) / 0.005188716   # +0.1%  z_displaced3 > 0.4233
        + 0.0008601338 * max(0.0, Q.mass_top40 - 115.7429) / 10.91182   # +0.1%  mass_top40 > 115.7
        - 0.0008205527 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.ischhad_10 - 0.0) / 0.1598293   # -0.1%  lep_z < 0.3397 and ischhad_10 > 0
        - 0.0007395972 * max(0.0, Q.mass_displaced3 - 13.03663) / 3.535402   # -0.1%  mass_displaced3 > 13.04
        - 0.0006995007 * max(0.0, Q.pair_mean_lndelta - -1.352792) / 0.01174331   # -0.1%  pair_mean_lndelta > -1.353
        + 0.0006990517 * max(0.0, 0.04607888 - Q.lep_dr) / 0.02662022   # +0.1%  lep_dr < 0.04608
        - 0.0005564967 * Q.sj3_pairmax_over_m / 0.807958   # -0.1%  sj3_pairmax_over_m
        + 0.0005448872 * max(0.0, Q.tdz_9 - 0.0) / 0.03812136   # +0.1%  tdz_9 > 0
        - 0.0004783536 * max(0.0, 0.007018285 - Q.sum_z_dr2_top3) / 0.001205932   # -0.0%  sum_z_dr2_top3 < 0.007018
        - 0.0004630664 * max(0.0, Q.C2 - 0.191059) / 0.01221273   # -0.0%  C2 > 0.1911
        - 0.0004243767 * max(0.0, Q.z_photon - 0.4982257) / 0.004464093   # -0.0%  z_photon > 0.4982
        + 0.0003825997 * max(0.0, Q.lund_max_lndelta - -0.8095462) / 0.06614362   # +0.0%  lund_max_lndelta > -0.8095
        - 0.0002949812 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 15.0 - Q.n_real_top15) / 0.009438413   # -0.0%  lep_z < 0.3397 and n_real_top15 < 15
        + 0.0002768944 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) * max(0.0, Q.td0_44 - -0.009316366) / 0.3796189   # +0.0%  sj3_pair_mass_max < 114.8 and td0_44 > -0.009316
        - 0.0002154606 * max(0.0, Q.lep_dr - 0.3014662) * max(0.0, 0.1048749 - Q.dr_8) / 0.0002533438   # -0.0%  lep_dr > 0.3015 and dr_8 < 0.1049
        - 0.0001211337 * max(0.0, 2.17433 - Q.sip_3d_1) / 0.01393752   # -0.0%  sip_3d_1 < 2.174
        + 9.966691e-05 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.212136 - Q.z_neutral_had) / 0.02151948   # +0.0%  lep_z < 0.3397 and z_neutral_had < 0.2121
        - 8.549017e-05 * max(0.0, Q.mass_top5 - 29.68311) * max(0.0, Q.ismuon_29 - 0.0) / 0.01551335   # -0.0%  mass_top5 > 29.68 and ismuon_29 > 0
        + 4.562573e-05 * max(0.0, Q.sj3_pair_mass_max - 128.6605) * max(0.0, Q.tdz_54 - 0.0) / 0.04606238   # +0.0%  sj3_pair_mass_max > 128.7 and tdz_54 > 0
        + 2.950904e-05 * max(0.0, Q.sj2_mass2 - 42.12131) * max(0.0, Q.eta_76 - 0.0) / 0.01028937   # +0.0%  sj2_mass2 > 42.12 and eta_76 > 0
        - 2.6783e-05 * max(0.0, Q.lep_dr - 0.3014662) / 0.01598628   # -0.0%  lep_dr > 0.3015
        - 3.91422e-06 * max(0.0, 56.49019 - Q.mass) / 0.8336837   # -0.0%  mass < 56.49
        + 1.750633e-06 * max(0.0, 114.7848 - Q.sj3_pair_mass_max) * max(0.0, Q.ismuon_12 - 0.0) / 0.07401273   # +0.0%  sj3_pair_mass_max < 114.8 and ismuon_12 > 0
    )
    return z


def neuron_26(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.231297e-06
    )
    return z


def neuron_27(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.308482e-06
    )
    return z


def neuron_28(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.617223e-06
    )
    return z


def neuron_29(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.183854e-06
    )
    return z


def neuron_30(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.603079e-06
    )
    return z


def neuron_31(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.509509e-06
    )
    return z


def neuron_32(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.441352e-05
    )
    return z


def neuron_33(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.567321e-06
    )
    return z


def neuron_34(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.388482e-05
    )
    return z


def neuron_35(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.743304e-06
    )
    return z


def neuron_36(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.831943e-06
    )
    return z


def neuron_37(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.357445e-06
    )
    return z


def neuron_38(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.41109e-06
    )
    return z


def neuron_39(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.27427e-06
    )
    return z


def neuron_40(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.775538e-07
    )
    return z


def neuron_41(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.814238e-08
    )
    return z


def neuron_42(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.779079e-06
    )
    return z


def neuron_43(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.407695e-06
    )
    return z


def neuron_44(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.324356e-06
    )
    return z


def neuron_45(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.849297e-06
    )
    return z


def neuron_46(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.368882e-07
    )
    return z


def neuron_47(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.49205e-07
    )
    return z


def neuron_48(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.791764e-06
    )
    return z


def neuron_49(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.243814e-06
    )
    return z


def neuron_50(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.175113e-06
    )
    return z


def neuron_51(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.01221e-06
    )
    return z


def neuron_52(Q):
    # scale S = 12.09; each line: share * term / its average size
    z = 12.08771 * (-0.08421497
        + 0.07896331 * max(0.0, 7.0 - Q.n_s3d_above_3) / 3.67411   # +7.9%  n_s3d_above_3 < 7
        + 0.06221267 * max(0.0, Q.mass - 71.96396) / 44.90212   # +6.2%  mass > 71.96
        - 0.0575002 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -5.8%  n_lepton < 1
        - 0.04459822 * max(0.0, 8.0 - Q.n_sd0_above_3) / 4.91424   # -4.5%  n_sd0_above_3 < 8
        - 0.04177245 * max(0.0, 123.2919 - Q.sd_mass) / 34.26699   # -4.2%  sd_mass < 123.3
        + 0.04122193 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +4.1%  lep_iso < 0.4382
        - 0.03416929 * max(0.0, 7.0 - Q.n_s3d_above_3) * max(0.0, 175.9957 - Q.sip_3d_3) / 598.947   # -3.4%  n_s3d_above_3 < 7 and sip_3d_3 < 176
        - 0.03363893 * max(0.0, 7.0 - Q.n_s3d_above_3) * max(0.0, 70.05844 - Q.sip_3d_2) / 185.9687   # -3.4%  n_s3d_above_3 < 7 and sip_3d_2 < 70.06
        + 0.03167548 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 18.80005 - Q.mass_displaced3) / 190.788   # +3.2%  lep_ptrel < 18.77 and mass_displaced3 < 18.8
        - 0.03159561 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -3.2%  lep_z < 0.2214
        + 0.03125292 * max(0.0, 0.5626523 - Q.LHA) / 0.1600511   # +3.1%  LHA < 0.5627
        + 0.02573392 * max(0.0, 154.5947 - Q.sd_mass) / 60.90029   # +2.6%  sd_mass < 154.6
        + 0.02514892 * max(0.0, 44.14912 - Q.lep_iso) / 39.77068   # +2.5%  lep_iso < 44.15
        + 0.02451775 * max(0.0, 83.61981 - Q.sd_mass) / 13.06909   # +2.5%  sd_mass < 83.62
        + 0.02142717 * max(0.0, 161.1264 - Q.mass_top50) / 50.68933   # +2.1%  mass_top50 < 161.1
        + 0.01862375 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # +1.9%  lep_ptrel < 27.32
        - 0.01822385 * max(0.0, 0.04229114 - Q.C3_b2) / 0.03492918   # -1.8%  C3_b2 < 0.04229
        + 0.01583097 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 4.606241 - Q.sip_3d_3) / 0.1354168   # +1.6%  z_displaced3 < 0.1343 and sip_3d_3 < 4.606
        + 0.01547662 * max(0.0, 0.07687439 - Q.tdz_0) / 0.08526919   # +1.5%  tdz_0 < 0.07687
        - 0.01418616 * max(0.0, 18.7678 - Q.lep_ptrel) / 14.28156   # -1.4%  lep_ptrel < 18.77
        - 0.01332852 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, 0.08643515 - Q.tau3) / 0.8197384   # -1.3%  lep_ptrel < 27.32 and tau3 < 0.08644
        - 0.01283578 * Q.n_pairs_kt_above_1 / 331.4992   # -1.3%  n_pairs_kt_above_1
        + 0.01185837 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +1.2%  n_pairs_kt_above_1 < 80
        - 0.0117332 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # -1.2%  lep_z < 0.3397
        - 0.01115762 * max(0.0, 1.362094 - Q.lep_iso) * max(0.0, 49.03695 - Q.sip_3d_3) / 32.75876   # -1.1%  lep_iso < 1.362 and sip_3d_3 < 49.04
        + 0.01043469 * max(0.0, 3.373037 - Q.sip_3d_1) / 0.1802381   # +1.0%  sip_3d_1 < 3.373
        + 0.01025571 * max(0.0, 8.0 - Q.n_dr_0_0p05) / 5.25296   # +1.0%  n_dr_0_0p05 < 8
        - 0.0101569 * max(0.0, Q.mass - 90.08945) * max(0.0, 0.7269473 - Q.tau32_b2) / 7.667197   # -1.0%  mass > 90.09 and tau32_b2 < 0.7269
        - 0.009672659 * max(0.0, 0.3788785 - Q.z_neutral_had) / 0.2321329   # -1.0%  z_neutral_had < 0.3789
        - 0.009509697 * max(0.0, 117.4867 - Q.mass) * max(0.0, 0.3565533 - Q.tau21_b2) / 1.536552   # -1.0%  mass < 117.5 and tau21_b2 < 0.3566
        - 0.008759705 * max(0.0, 0.06603871 - Q.tau2) / 0.01287286   # -0.9%  tau2 < 0.06604
        + 0.008703786 * max(0.0, 97.52633 - Q.sj3_pair_mass_max) / 14.66667   # +0.9%  sj3_pair_mass_max < 97.53
        + 0.008591446 * max(0.0, 45.26581 - Q.sip_3d_2) / 21.48394   # +0.9%  sip_3d_2 < 45.27
        - 0.008185231 * max(0.0, 42.31629 - Q.sd_mass) / 4.001873   # -0.8%  sd_mass < 42.32
        + 0.007922074 * max(0.0, Q.z_top10_slots - 0.7362734) / 0.06667364   # +0.8%  z_top10_slots > 0.7363
        + 0.00779369 * max(0.0, 129.5874 - Q.mass_top50) * max(0.0, 0.114659 - Q.lep_dr) / 1.914075   # +0.8%  mass_top50 < 129.6 and lep_dr < 0.1147
        - 0.007749081 * max(0.0, 0.0 - Q.tdz_0) / 0.01858885   # -0.8%  tdz_0 < 0
        + 0.00747882 * max(0.0, 0.3788785 - Q.z_neutral_had) * max(0.0, Q.ecf_g41 - 6.216954e-05) / 3.09401e-05   # +0.7%  z_neutral_had < 0.3789 and ecf_g41 > 6.217e-05
        + 0.00736889 * max(0.0, 26.78691 - Q.mass_displaced3) / 20.43083   # +0.7%  mass_displaced3 < 26.79
        - 0.006546272 * max(0.0, 129.5874 - Q.mass_top50) / 24.20267   # -0.7%  mass_top50 < 129.6
        + 0.006508547 * max(0.0, 0.09209404 - Q.C2_b2) * max(0.0, 1.462588 - Q.D3_b2) / 0.04100572   # +0.7%  C2_b2 < 0.09209 and D3_b2 < 1.463
        - 0.006072962 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 3.113281 - Q.max_abs_d0) / 0.1236147   # -0.6%  z_displaced3 < 0.1343 and max_abs_d0 < 3.113
        + 0.00607011 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 4.636903 - Q.sip_3d_2) / 0.2196467   # +0.6%  lep_z < 0.3397 and sip_3d_2 < 4.637
        - 0.006050286 * max(0.0, 2.802481 - Q.sip_3d_1) / 0.07766626   # -0.6%  sip_3d_1 < 2.802
        + 0.006040564 * max(0.0, 0.09209404 - Q.C2_b2) / 0.0375889   # +0.6%  C2_b2 < 0.09209
        + 0.005623558 * max(0.0, 0.5626523 - Q.LHA) * max(0.0, 26.78691 - Q.mass_displaced3) / 3.494642   # +0.6%  LHA < 0.5627 and mass_displaced3 < 26.79
        - 0.005443605 * max(0.0, Q.mass - 90.08945) / 29.81377   # -0.5%  mass > 90.09
        - 0.005083632 * max(0.0, Q.lep_ptrel - 12.15228) / 3.983062   # -0.5%  lep_ptrel > 12.15
        + 0.005063074 * max(0.0, 1.0 - Q.n_lepton) * max(0.0, 11.13866 - Q.sip_3d_3) / 3.098729   # +0.5%  n_lepton < 1 and sip_3d_3 < 11.14
        + 0.004797901 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +0.5%  lep_iso < 1.362
        + 0.004693885 * max(0.0, 0.8035262 - Q.mass_displaced5) / 0.3308315   # +0.5%  mass_displaced5 < 0.8035
        + 0.004478878 * Q.n_charged_had / 18.52697   # +0.4%  n_charged_had
        + 0.004178634 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, Q.N2 - 0.1824346) / 0.009842026   # +0.4%  z_displaced3 < 0.1343 and N2 > 0.1824
        - 0.004124038 * max(0.0, 1.008706e-05 - Q.e3_b2) / 1.503587e-06   # -0.4%  e3_b2 < 1.009e-05
        - 0.004006391 * max(0.0, 4.0 - Q.n_sd0_above_3) / 1.660297   # -0.4%  n_sd0_above_3 < 4
        - 0.003986296 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, 0.3201841 - Q.mass_displaced5) / 0.04106726   # -0.4%  lep_iso < 0.4382 and mass_displaced5 < 0.3202
        - 0.003774196 * max(0.0, Q.jet_charge_k03 - -0.1486714) / 0.3723647   # -0.4%  jet_charge_k03 > -0.1487
        - 0.003738107 * max(0.0, Q.jet_abs_eta - 0.8317778) / 0.1684788   # -0.4%  jet_abs_eta > 0.8318
        + 0.003696435 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # +0.4%  n_pairs_kt_above_3 < 28
        - 0.003545223 * max(0.0, 101.849 - Q.sj4_pair_mass_max) / 24.8879   # -0.4%  sj4_pair_mass_max < 101.8
        - 0.00333651 * max(0.0, 117.4867 - Q.mass) / 15.64803   # -0.3%  mass < 117.5
        - 0.003097435 * max(0.0, 0.1342762 - Q.z_displaced3) / 0.07089169   # -0.3%  z_displaced3 < 0.1343
        + 0.002955683 * max(0.0, 26.78691 - Q.mass_displaced3) * max(0.0, Q.n_lund_kt_above_1 - 4.0) / 31.51463   # +0.3%  mass_displaced3 < 26.79 and n_lund_kt_above_1 > 4
        + 0.002874949 * max(0.0, Q.N2_b2 - 0.2392833) / 0.0118944   # +0.3%  N2_b2 > 0.2393
        + 0.002316103 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, -0.01644749 - Q.tdz_0) / 0.003896084   # +0.2%  lep_z < 0.3397 and tdz_0 < -0.01645
        + 0.002132032 * max(0.0, Q.n_particles - 38.0) / 6.78515   # +0.2%  n_particles > 38
        + 0.002072031 * max(0.0, Q.lund_max_lndelta - -0.615738) / 0.02343634   # +0.2%  lund_max_lndelta > -0.6157
        - 0.00195905 * max(0.0, 0.2848657 - Q.sj2_dr) / 0.01978141   # -0.2%  sj2_dr < 0.2849
        - 0.001875483 * max(0.0, Q.mass - 90.08945) * max(0.0, 0.6818924 - Q.tau43_b2) / 2.346852   # -0.2%  mass > 90.09 and tau43_b2 < 0.6819
        + 0.001816009 * max(0.0, 79.27954 - Q.mass_top50) / 2.845865   # +0.2%  mass_top50 < 79.28
        - 0.001690516 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.sj3_pair_mass_min - 48.40547) / 1.423551   # -0.2%  lep_z < 0.3397 and sj3_pair_mass_min > 48.41
        - 0.001618858 * max(0.0, Q.N3 - 0.2681212) / 0.3845471   # -0.2%  N3 > 0.2681
        + 0.001528988 * max(0.0, 0.04229114 - Q.C3_b2) * max(0.0, 0.2816529 - Q.pt2_over_pt0) / 0.0008006262   # +0.2%  C3_b2 < 0.04229 and pt2_over_pt0 < 0.2817
        + 0.001516202 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # +0.2%  sj2_mass2 < 1.852
        - 0.001434791 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 0.1905794 - Q.dr_21) / 0.7483514   # -0.1%  lep_ptrel < 18.77 and dr_21 < 0.1906
        - 0.001217383 * max(0.0, 0.3951525 - Q.tau32) / 0.009206105   # -0.1%  tau32 < 0.3952
        + 0.001205732 * max(0.0, 65.0404 - Q.mass_charged) / 11.01768   # +0.1%  mass_charged < 65.04
        - 0.001149921 * max(0.0, 56.73313 - Q.sj3_pair_mass_max) / 1.542034   # -0.1%  sj3_pair_mass_max < 56.73
        - 0.001095411 * max(0.0, Q.sj3_mass1 - 36.36236) / 1.189826   # -0.1%  sj3_mass1 > 36.36
        - 0.001018929 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, Q.lep_dr - 0.114659) / 0.002466724   # -0.1%  z_displaced3 < 0.1343 and lep_dr > 0.1147
        - 0.0009959772 * max(0.0, 0.3788785 - Q.z_neutral_had) * max(0.0, 0.01100159 - Q.d0err_3) / 0.0009171798   # -0.1%  z_neutral_had < 0.3789 and d0err_3 < 0.011
        + 0.0009852878 * max(0.0, Q.mass_top15 - 94.94813) / 6.576712   # +0.1%  mass_top15 > 94.95
        - 0.0006604255 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 0.04063514 - Q.dr_21) / 0.0004216547   # -0.1%  z_displaced3 < 0.1343 and dr_21 < 0.04064
        - 0.000603605 * max(0.0, 5.511295 - Q.pair_max_lnm2) / 0.0518722   # -0.1%  pair_max_lnm2 < 5.511
        + 0.0005921082 * max(0.0, 129.5874 - Q.mass_top50) * max(0.0, 0.0632362 - Q.dr_21) / 0.40919   # +0.1%  mass_top50 < 129.6 and dr_21 < 0.06324
        - 0.0003907604 * max(0.0, 0.09209404 - Q.C2_b2) * max(0.0, Q.sip_3d_2 - 3.041925) / 13.63278   # -0.0%  C2_b2 < 0.09209 and sip_3d_2 > 3.042
        + 0.0003619972 * max(0.0, 0.02600452 - Q.z_displaced3) / 0.00810998   # +0.0%  z_displaced3 < 0.026
        + 0.0003538674 * max(0.0, Q.sj2_mass1 - 91.2852) / 1.019309   # +0.0%  sj2_mass1 > 91.29
        + 0.0003310141 * max(0.0, Q.sj3_pair_mass_min - 80.02563) / 0.702582   # +0.0%  sj3_pair_mass_min > 80.03
        - 0.0003243948 * max(0.0, Q.mass_top15 - 94.94813) * max(0.0, Q.sj3_dr13 - 0.2640447) / 1.729593   # -0.0%  mass_top15 > 94.95 and sj3_dr13 > 0.264
        + 0.0003240033 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.mass_2photon - 3.013) / 4.662811   # +0.0%  n_pairs_kt_above_3 < 28 and mass_2photon > 3.013
        + 0.0003013959 * max(0.0, Q.n_neutral_had - 7.0) / 0.21307   # +0.0%  n_neutral_had > 7
        + 0.0002620622 * max(0.0, Q.mass - 90.08945) * max(0.0, 5.0 - Q.n_s3d_above_3) / 43.00566   # +0.0%  mass > 90.09 and n_s3d_above_3 < 5
        - 0.0002157064 * max(0.0, 0.04276413 - Q.M2) / 0.0009967452   # -0.0%  M2 < 0.04276
        - 0.000165401 * max(0.0, 0.09209404 - Q.C2_b2) * max(0.0, Q.n_muon - 0.0) / 0.008709924   # -0.0%  C2_b2 < 0.09209 and n_muon > 0
        + 0.0001618674 * max(0.0, Q.mass - 90.08945) * max(0.0, Q.td0_13 - -0.06278656) / 2.625608   # +0.0%  mass > 90.09 and td0_13 > -0.06279
        - 0.00012975 * max(0.0, 0.1238959 - Q.C2) / 0.01477465   # -0.0%  C2 < 0.1239
        + 7.389911e-05 * max(0.0, Q.sj3_pair_mass_min - 80.02563) * max(0.0, 20.75111 - Q.sip_3d_2) / 2.43357   # +0.0%  sj3_pair_mass_min > 80.03 and sip_3d_2 < 20.75
        - 5.023712e-05 * max(0.0, 0.3951525 - Q.tau32) * max(0.0, Q.d0err_14 - 0.01499939) / 4.887181e-05   # -0.0%  tau32 < 0.3952 and d0err_14 > 0.015
        - 1.563737e-05 * max(0.0, Q.mass - 90.08945) * max(0.0, Q.d0err_75 - 0.0) / 0.03680061   # -0.0%  mass > 90.09 and d0err_75 > 0
    )
    return z


def neuron_53(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.889392e-06
    )
    return z


def neuron_54(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.190656e-05
    )
    return z


def neuron_55(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.066932e-05
    )
    return z


def neuron_56(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.072387e-07
    )
    return z


def neuron_57(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.959676e-06
    )
    return z


def neuron_58(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.61715e-06
    )
    return z


def neuron_59(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.029272e-06
    )
    return z


def neuron_60(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.666768e-06
    )
    return z


def neuron_61(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.840546e-06
    )
    return z


def neuron_62(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.422871e-07
    )
    return z


def neuron_63(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.308531e-06
    )
    return z


def neuron_64(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.258629e-06
    )
    return z


def neuron_65(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.791076e-06
    )
    return z


def neuron_66(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.709755e-06
    )
    return z


def neuron_67(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.219533e-06
    )
    return z


def neuron_68(Q):
    # scale S = 14.88; each line: share * term / its average size
    z = 14.8843 * (0.2540838
        - 0.1300536 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -13.0%  mass < 164.4
        - 0.09676826 * max(0.0, Q.sd_mass - 83.61981) / 26.48984   # -9.7%  sd_mass > 83.62
        - 0.07144583 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) / 44.19386   # -7.1%  sj3_pair_mass_min < 80.03
        + 0.0699553 * max(0.0, Q.sd_mass - 42.31629) / 58.72614   # +7.0%  sd_mass > 42.32
        - 0.05279381 * max(0.0, Q.n_pt_above_1 - 17.0) / 21.33585   # -5.3%  n_pt_above_1 > 17
        - 0.04564649 * max(0.0, Q.mass - 79.47361) / 38.31536   # -4.6%  mass > 79.47
        + 0.04125925 * max(0.0, 120.6471 - Q.sj3_pair_mass_max) / 31.70826   # +4.1%  sj3_pair_mass_max < 120.6
        + 0.03639236 * max(0.0, 114.0172 - Q.mass) / 13.82223   # +3.6%  mass < 114
        - 0.03476581 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # -3.5%  lep_iso < 0.4382
        + 0.03149059 * max(0.0, 2.0 - Q.n_lepton) / 1.440613   # +3.1%  n_lepton < 2
        + 0.02923588 * max(0.0, Q.sj3_pair_mass_max - 44.2029) / 48.7862   # +2.9%  sj3_pair_mass_max > 44.2
        + 0.02733574 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # +2.7%  sd_mass > 88.82
        - 0.02564884 * Q.sj3_pairmin_over_m / 0.3112428   # -2.6%  sj3_pairmin_over_m
        - 0.02561776 * max(0.0, Q.n_charged_had - 13.0) / 6.4188   # -2.6%  n_charged_had > 13
        + 0.02213209 * max(0.0, Q.sd_mass - 127.8042) / 7.009137   # +2.2%  sd_mass > 127.8
        + 0.01980709 * max(0.0, 0.7981752 - Q.tau32) / 0.1480334   # +2.0%  tau32 < 0.7982
        - 0.01465228 * max(0.0, 0.04544279 - Q.M2_b2) / 0.01635302   # -1.5%  M2_b2 < 0.04544
        - 0.01435077 * max(0.0, 0.3261071 - Q.z_displaced3) * max(0.0, 0.4191372 - Q.lep_dr) / 0.0789782   # -1.4%  z_displaced3 < 0.3261 and lep_dr < 0.4191
        + 0.01336433 * max(0.0, Q.n_pairs_kt_above_1 - 101.0) / 239.3315   # +1.3%  n_pairs_kt_above_1 > 101
        - 0.01234026 * max(0.0, 127.2317 - Q.mass) / 21.76326   # -1.2%  mass < 127.2
        + 0.01217409 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # +1.2%  n_lepton < 1
        - 0.008996384 * max(0.0, 1.323111 - Q.jet_abs_eta) / 0.6200087   # -0.9%  jet_abs_eta < 1.323
        - 0.008220036 * max(0.0, Q.n_pt_above_1 - 17.0) * max(0.0, 0.1061578 - Q.z_displaced3) / 1.056777   # -0.8%  n_pt_above_1 > 17 and z_displaced3 < 0.1062
        - 0.008131297 * max(0.0, 3.38061 - Q.pair_mean_lnm2) / 0.9985192   # -0.8%  pair_mean_lnm2 < 3.381
        + 0.007102289 * max(0.0, Q.n_pt_above_1 - 17.0) * max(0.0, 0.8939856 - Q.tau43) / 2.073009   # +0.7%  n_pt_above_1 > 17 and tau43 < 0.894
        - 0.00672041 * max(0.0, 90.08945 - Q.mass) * max(0.0, 1.0 - Q.n_lepton) / 3.361761   # -0.7%  mass < 90.09 and n_lepton < 1
        + 0.006618344 * max(0.0, Q.mass_top10 - 27.42853) / 40.59191   # +0.7%  mass_top10 > 27.43
        + 0.005845373 * max(0.0, 3.38061 - Q.pair_mean_lnm2) * max(0.0, 0.3788785 - Q.z_neutral_had) / 0.2490716   # +0.6%  pair_mean_lnm2 < 3.381 and z_neutral_had < 0.3789
        - 0.005760065 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.6%  mass < 90.09
        + 0.005582597 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +0.6%  n_pairs_kt_above_1 < 80
        - 0.00545927 * max(0.0, 2.286291 - Q.D2_b2) / 0.7248948   # -0.5%  D2_b2 < 2.286
        - 0.005366437 * max(0.0, Q.lep_ptrel - 6.983043) / 5.140178   # -0.5%  lep_ptrel > 6.983
        - 0.005219586 * max(0.0, 1.0 - Q.n_lepton) * max(0.0, 4.636903 - Q.sip_3d_2) / 0.5344861   # -0.5%  n_lepton < 1 and sip_3d_2 < 4.637
        + 0.005203685 * max(0.0, 0.3261071 - Q.z_displaced3) * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 0.2634324   # +0.5%  z_displaced3 < 0.3261 and n_lund_kt_above_5 > 1
        + 0.005173752 * max(0.0, 0.04544279 - Q.M2_b2) * max(0.0, 76.15079 - Q.mass_neutral) / 0.6214169   # +0.5%  M2_b2 < 0.04544 and mass_neutral < 76.15
        - 0.004818127 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # -0.5%  sd_mass > 154.6
        + 0.004770712 * max(0.0, 114.0172 - Q.mass) * max(0.0, 0.114659 - Q.lep_dr) / 1.144546   # +0.5%  mass < 114 and lep_dr < 0.1147
        - 0.004417911 * max(0.0, 0.3261071 - Q.z_displaced3) * max(0.0, Q.n_lund - 9.0) / 0.5293194   # -0.4%  z_displaced3 < 0.3261 and n_lund > 9
        + 0.004185028 * max(0.0, 3.0 - Q.n_sdz_above_5) / 1.349857   # +0.4%  n_sdz_above_5 < 3
        + 0.003532787 * max(0.0, 2.382955 - Q.mass_displaced5) / 1.169903   # +0.4%  mass_displaced5 < 2.383
        + 0.00321349 * max(0.0, Q.n_pt_above_1 - 17.0) * max(0.0, 0.8786609 - Q.tau54) / 0.8588641   # +0.3%  n_pt_above_1 > 17 and tau54 < 0.8787
        - 0.003031004 * max(0.0, 0.7223231 - Q.z_charged_had) / 0.2225062   # -0.3%  z_charged_had < 0.7223
        + 0.002926216 * max(0.0, 0.04544279 - Q.M2_b2) * max(0.0, Q.mass_displaced3 - 0.0) / 0.1412916   # +0.3%  M2_b2 < 0.04544 and mass_displaced3 > 0
        - 0.002701767 * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 4.79296   # -0.3%  n_dr_0p2_0p4 > 9
        - 0.002415555 * max(0.0, 80.36029 - Q.sj3_pair_mass_max) / 6.418891   # -0.2%  sj3_pair_mass_max < 80.36
        - 0.002396715 * max(0.0, 0.7981752 - Q.tau32) * max(0.0, 0.7223231 - Q.z_charged_had) / 0.03607804   # -0.2%  tau32 < 0.7982 and z_charged_had < 0.7223
        + 0.002386113 * max(0.0, Q.lep_ptrel - 6.983043) * max(0.0, 0.1404188 - Q.C2_b2) / 0.4062449   # +0.2%  lep_ptrel > 6.983 and C2_b2 < 0.1404
        + 0.002144525 * max(0.0, 2.317124 - Q.sip_3d_3) / 0.1434495   # +0.2%  sip_3d_3 < 2.317
        - 0.002038448 * max(0.0, 0.7981752 - Q.tau32) * max(0.0, Q.sj3_dr_min - 0.3526989) / 0.003071098   # -0.2%  tau32 < 0.7982 and sj3_dr_min > 0.3527
        - 0.001966553 * max(0.0, 47.41085 - Q.mass_top5) / 14.28312   # -0.2%  mass_top5 < 47.41
        - 0.001880108 * max(0.0, Q.psi_0p1 - 0.835544) / 0.01134109   # -0.2%  psi_0p1 > 0.8355
        + 0.001866696 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) * max(0.0, Q.sj3_mass1 - 18.73268) / 180.6276   # +0.2%  sj3_pair_mass_min < 80.03 and sj3_mass1 > 18.73
        - 0.001794653 * max(0.0, 56.73313 - Q.sj3_pair_mass_max) / 1.542034   # -0.2%  sj3_pair_mass_max < 56.73
        - 0.001749262 * max(0.0, 114.0172 - Q.mass) * max(0.0, Q.n_lund_kt_above_1 - 3.0) / 23.65019   # -0.2%  mass < 114 and n_lund_kt_above_1 > 3
        + 0.001739759 * max(0.0, Q.e4_b05 - 2.879843e-05) / 2.220464e-05   # +0.2%  e4_b05 > 2.88e-05
        + 0.001622466 * max(0.0, 127.2317 - Q.mass) * max(0.0, Q.sd_zg - 0.1900649) / 2.080143   # +0.2%  mass < 127.2 and sd_zg > 0.1901
        - 0.001615143 * max(0.0, 0.7981752 - Q.tau32) * max(0.0, 0.3190414 - Q.sj3_dr_min) / 0.01394611   # -0.2%  tau32 < 0.7982 and sj3_dr_min < 0.319
        + 0.001611324 * max(0.0, 0.9506259 - Q.mass_2charged) / 0.1206772   # +0.2%  mass_2charged < 0.9506
        + 0.001547303 * max(0.0, 0.1036778 - Q.z_dr_0p1_0p2) / 0.02011467   # +0.2%  z_dr_0p1_0p2 < 0.1037
        + 0.001415527 * max(0.0, 0.002792418 - Q.sum_z_dr2_top3) / 0.0002360284   # +0.1%  sum_z_dr2_top3 < 0.002792
        - 0.001265833 * max(0.0, Q.z_displaced5 - 0.2801368) / 0.0126536   # -0.1%  z_displaced5 > 0.2801
        - 0.001224408 * max(0.0, Q.tau21_b2 - 0.5211946) / 0.02080308   # -0.1%  tau21_b2 > 0.5212
        + 0.001181618 * max(0.0, 0.3261071 - Q.z_displaced3) / 0.2272573   # +0.1%  z_displaced3 < 0.3261
        - 0.001168184 * max(0.0, 3.0 - Q.n_sdz_above_5) * max(0.0, Q.sj2_dr - 0.3220633) / 0.1138968   # -0.1%  n_sdz_above_5 < 3 and sj2_dr > 0.3221
        - 0.001131239 * max(0.0, 0.03259227 - Q.M3) / 0.00464829   # -0.1%  M3 < 0.03259
        + 0.001083621 * max(0.0, Q.n_pt_above_5 - 25.0) / 1.700337   # +0.1%  n_pt_above_5 > 25
        - 0.001015619 * max(0.0, 0.01040452 - Q.sum_z_dr2_top15) / 0.0007317661   # -0.1%  sum_z_dr2_top15 < 0.0104
        - 0.0009007323 * max(0.0, Q.n_s3d_above_3 - 5.0) / 0.8834567   # -0.1%  n_s3d_above_3 > 5
        - 0.0009003568 * max(0.0, Q.sd_mass - 88.81751) * max(0.0, Q.M3 - 0.02540381) / 0.1244951   # -0.1%  sd_mass > 88.82 and M3 > 0.0254
        - 0.00087786 * max(0.0, Q.mass_top15 - 119.5993) / 2.048216   # -0.1%  mass_top15 > 119.6
        + 0.0008444308 * max(0.0, Q.n_electron - 1.0) / 0.08167   # +0.1%  n_electron > 1
        - 0.0007174187 * max(0.0, 114.0172 - Q.mass) * max(0.0, 2.255345 - Q.D2) / 6.249005   # -0.1%  mass < 114 and D2 < 2.255
        + 0.0006126855 * max(0.0, 0.3261071 - Q.z_displaced3) * max(0.0, Q.mass_2photon - 9.82024) / 0.4109246   # +0.1%  z_displaced3 < 0.3261 and mass_2photon > 9.82
        + 0.000588027 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # +0.1%  n_pairs_kt_above_3 < 28
        - 0.000540345 * max(0.0, Q.mass_over_sum_pt_sq - 0.0729277) / 0.002813812   # -0.1%  mass_over_sum_pt_sq > 0.07293
        - 0.0005165642 * max(0.0, 1.323111 - Q.jet_abs_eta) * max(0.0, Q.n_neutral_had - 6.0) / 0.1578672   # -0.1%  jet_abs_eta < 1.323 and n_neutral_had > 6
        - 0.0004353558 * max(0.0, Q.sd_mass - 42.31629) * max(0.0, 0.4683306 - Q.z_dr_0p2_0p4) / 13.07579   # -0.0%  sd_mass > 42.32 and z_dr_0p2_0p4 < 0.4683
        + 0.0004188752 * max(0.0, Q.mass_top30 - 162.7874) / 1.306268   # +0.0%  mass_top30 > 162.8
        - 0.0004171858 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -0.0%  sip_3d_2 < 226.3
        - 0.0003994197 * max(0.0, Q.sd_mass - 42.31629) * max(0.0, Q.sj3_dr13 - 0.6229991) / 1.427562   # -0.0%  sd_mass > 42.32 and sj3_dr13 > 0.623
        - 0.0003994127 * max(0.0, 3.38061 - Q.pair_mean_lnm2) * max(0.0, Q.z_top30_slots - 0.8316085) / 0.1174666   # -0.0%  pair_mean_lnm2 < 3.381 and z_top30_slots > 0.8316
        + 0.0003639822 * max(0.0, Q.lep_z - 0.5187302) / 0.007316078   # +0.0%  lep_z > 0.5187
        - 0.0003426189 * max(0.0, Q.n_s3d_above_3 - 5.0) * max(0.0, 226.3008 - Q.sip_3d_2) / 73.04149   # -0.0%  n_s3d_above_3 > 5 and sip_3d_2 < 226.3
        - 0.0003400346 * max(0.0, 7.104488e-06 - Q.e3_b2) / 8.290971e-07   # -0.0%  e3_b2 < 7.104e-06
        - 0.0003288206 * max(0.0, Q.mass_2photon - 22.18431) / 0.4406114   # -0.0%  mass_2photon > 22.18
        - 0.0002743225 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.0%  lep_ptrel > 43.21
        + 0.0002697388 * max(0.0, 3.298199 - Q.sj3_mass2) / 0.1692566   # +0.0%  sj3_mass2 < 3.298
        - 0.0002421112 * max(0.0, Q.C2_b05 - 0.3533901) / 0.003974948   # -0.0%  C2_b05 > 0.3534
        - 0.0002320952 * max(0.0, Q.mass_top30 - 162.7874) * max(0.0, 1049.334 - Q.sum_e) / 214.2391   # -0.0%  mass_top30 > 162.8 and sum_e < 1049
        + 0.0001850603 * max(0.0, Q.sj3_dr_min - 0.3526989) / 0.01394539   # +0.0%  sj3_dr_min > 0.3527
        - 0.0001700732 * max(0.0, Q.sd_mass - 154.5947) * max(0.0, 0.2450652 - Q.sd_zg) / 0.1089994   # -0.0%  sd_mass > 154.6 and sd_zg < 0.2451
        - 5.98929e-05 * max(0.0, Q.sj3_dr_min - 0.3526989) * max(0.0, Q.charge_27 - 0.0) / 0.002673855   # -0.0%  sj3_dr_min > 0.3527 and charge_27 > 0
        + 4.855806e-05 * max(0.0, 0.03378303 - Q.tau3) / 0.003783449   # +0.0%  tau3 < 0.03378
        + 4.144076e-05 * max(0.0, Q.mass_top10 - 27.42853) * max(0.0, Q.dr_77 - 0.0) / 0.1917396   # +0.0%  mass_top10 > 27.43 and dr_77 > 0
        - 1.658452e-05 * max(0.0, Q.n_s3d_above_3 - 5.0) * max(0.0, Q.eta_48 - 0.2019104) / 0.01429746   # -0.0%  n_s3d_above_3 > 5 and eta_48 > 0.2019
        - 1.63373e-05 * max(0.0, Q.mass_2photon - 22.18431) * max(0.0, Q.iselectron_1 - 0.0) / 0.01053741   # -0.0%  mass_2photon > 22.18 and iselectron_1 > 0
        - 6.857211e-06 * max(0.0, 0.03259227 - Q.M3) * max(0.0, Q.ismuon_4 - 0.0) / 3.402682e-05   # -0.0%  M3 < 0.03259 and ismuon_4 > 0
        + 5.144059e-06 * max(0.0, Q.mass_top30 - 162.7874) * max(0.0, Q.e4_b2 - 1.13e-10) / 2.583592e-06   # +0.0%  mass_top30 > 162.8 and e4_b2 > 1.13e-10
    )
    return z


def neuron_69(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.926153e-06
    )
    return z


def neuron_70(Q):
    # scale S = 5.848; each line: share * term / its average size
    z = 5.847622 * (0.01712203
        + 0.0772598 * max(0.0, 0.3258728 - Q.dr01) / 0.1941327   # +7.7%  dr01 < 0.3259
        - 0.05740758 * max(0.0, 17.3274 - Q.m01) / 8.627026   # -5.7%  m01 < 17.33
        + 0.05614167 * max(0.0, 164.4374 - Q.mass) / 52.54489   # +5.6%  mass < 164.4
        - 0.04348113 * max(0.0, 32.61679 - Q.mass_displaced5) / 27.02182   # -4.3%  mass_displaced5 < 32.62
        - 0.04199093 * max(0.0, 182.8592 - Q.mass) / 69.61635   # -4.2%  mass < 182.9
        + 0.0404638 * max(0.0, 0.2392833 - Q.N2_b2) / 0.0805745   # +4.0%  N2_b2 < 0.2393
        - 0.03878005 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # -3.9%  lep_iso < 1.362
        + 0.03698653 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +3.7%  lep_iso < 0.4382
        + 0.03403742 * max(0.0, 24.0 - Q.n_dr_0p2_0p4) / 13.19173   # +3.4%  n_dr_0p2_0p4 < 24
        - 0.02859983 * max(0.0, 117.4867 - Q.mass) * max(0.0, 0.221369 - Q.C2_b2) / 2.416124   # -2.9%  mass < 117.5 and C2_b2 < 0.2214
        - 0.02636365 * max(0.0, 577.991 - Q.sip_3d_3) / 516.7039   # -2.6%  sip_3d_3 < 578
        - 0.02433738 * max(0.0, Q.lund_max_lndelta - -2.071997) / 1.011402   # -2.4%  lund_max_lndelta > -2.072
        - 0.02385808 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -2.4%  n_lepton < 1
        + 0.02320227 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # +2.3%  sd_mass > 88.82
        - 0.02031341 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, 24.0 - Q.n_dr_0p2_0p4) / 276.6904   # -2.0%  lep_ptrel < 27.32 and n_dr_0p2_0p4 < 24
        + 0.01838402 * Q.n_photon / 16.03902   # +1.8%  n_photon
        - 0.01794283 * max(0.0, 0.0856189 - Q.dr01) / 0.03291804   # -1.8%  dr01 < 0.08562
        - 0.01680781 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -1.7%  lep_z < 0.2214
        - 0.01651461 * max(0.0, 161.1264 - Q.mass_top50) / 50.68933   # -1.7%  mass_top50 < 161.1
        + 0.01598406 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # +1.6%  n_s3d_above_3 < 10
        + 0.01589525 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, 0.114659 - Q.lep_dr) / 3.844982   # +1.6%  mass_top50 < 161.1 and lep_dr < 0.1147
        - 0.0152679 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, 0.2392833 - Q.N2_b2) / 1.686196   # -1.5%  lep_ptrel < 27.32 and N2_b2 < 0.2393
        + 0.01463655 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # +1.5%  sip_3d_2 < 447.1
        + 0.01419346 * max(0.0, Q.z_displaced5 - 0.06245248) * max(0.0, Q.z_charged_had - 0.1891627) / 0.02129547   # +1.4%  z_displaced5 > 0.06245 and z_charged_had > 0.1892
        + 0.01321209 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, 0.6651974 - Q.tau21) / 12.34031   # +1.3%  mass_top50 < 161.1 and tau21 < 0.6652
        - 0.01307845 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # -1.3%  max_abs_d0 < 10.52
        + 0.01213876 * max(0.0, Q.lund_max_lndelta - -1.388411) / 0.3966091   # +1.2%  lund_max_lndelta > -1.388
        + 0.01165341 * max(0.0, 68.20711 - Q.mass_charged) / 12.83892   # +1.2%  mass_charged < 68.21
        - 0.01161732 * max(0.0, Q.z_displaced5 - 0.06245248) / 0.0592711   # -1.2%  z_displaced5 > 0.06245
        - 0.01042908 * max(0.0, 146.3096 - Q.mass_top30) / 44.39811   # -1.0%  mass_top30 < 146.3
        - 0.009413422 * max(0.0, Q.z_displaced5 - 0.06245248) * max(0.0, Q.lnpt_11 - 1.305478) / 0.06759248   # -0.9%  z_displaced5 > 0.06245 and lnpt_11 > 1.305
        + 0.008556083 * max(0.0, 121.0623 - Q.mass_top30) / 23.2224   # +0.9%  mass_top30 < 121.1
        - 0.008002151 * max(0.0, Q.sd_mass - 62.03827) / 42.41619   # -0.8%  sd_mass > 62.04
        + 0.007690148 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, 1.0 - Q.z_top50_slots) / 0.1624311   # +0.8%  lep_ptrel < 27.32 and z_top50_slots < 1
        + 0.0072865 * max(0.0, 117.4867 - Q.mass) / 15.64803   # +0.7%  mass < 117.5
        - 0.007026421 * max(0.0, Q.sj3_pair_mass_min - 44.1643) / 6.009845   # -0.7%  sj3_pair_mass_min > 44.16
        + 0.007012701 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, 0.4381892 - Q.lep_iso) / 2.011766   # +0.7%  n_s3d_above_3 < 10 and lep_iso < 0.4382
        - 0.006854887 * max(0.0, Q.sd_mass - 123.2919) / 8.015693   # -0.7%  sd_mass > 123.3
        + 0.006280402 * max(0.0, 2.43362 - Q.m01) / 0.4606572   # +0.6%  m01 < 2.434
        + 0.006163477 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, -5.474085 - Q.lnerel_49) / 1.588383   # +0.6%  lep_z < 0.2214 and lnerel_49 < -5.474
        + 0.006058394 * max(0.0, Q.pair_mean_lnz - -1.676789) / 0.09958353   # +0.6%  pair_mean_lnz > -1.677
        - 0.005703417 * max(0.0, 0.1049877 - Q.C2_b2) / 0.04694282   # -0.6%  C2_b2 < 0.105
        - 0.005688519 * max(0.0, Q.tdz_0 - 0.04233308) / 0.01087807   # -0.6%  tdz_0 > 0.04233
        - 0.005585447 * max(0.0, 2.0 - Q.n_s3d_above_10) / 0.6939567   # -0.6%  n_s3d_above_10 < 2
        + 0.005513726 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 162.7874 - Q.mass_top30) / 16.52391   # +0.6%  lep_z < 0.3397 and mass_top30 < 162.8
        - 0.004777796 * max(0.0, 2.613544 - Q.D2_b2) / 0.9296486   # -0.5%  D2_b2 < 2.614
        + 0.004724136 * max(0.0, 4.326415 - Q.sj3_mass2) / 0.2966808   # +0.5%  sj3_mass2 < 4.326
        + 0.004685267 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.z_photon - 0.05998812) / 1.25908   # +0.5%  n_s3d_above_3 < 10 and z_photon > 0.05999
        - 0.004570091 * max(0.0, 0.02574154 - Q.mass_over_sum_pt_sq) / 0.003415121   # -0.5%  mass_over_sum_pt_sq < 0.02574
        + 0.004341594 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.tdz_0 - 0.04233308) / 0.0029958   # +0.4%  lep_z < 0.3397 and tdz_0 > 0.04233
        - 0.004247003 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # -0.4%  lep_ptrel < 27.32
        + 0.004146075 * max(0.0, Q.pair_mean_lndelta - -1.714682) / 0.05165279   # +0.4%  pair_mean_lndelta > -1.715
        - 0.004109218 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, Q.n_neutral - 8.0) / 506.9335   # -0.4%  mass_top50 < 161.1 and n_neutral > 8
        - 0.003819329 * max(0.0, 0.072817 - Q.sum_z_dr2_top2) / 0.05267127   # -0.4%  sum_z_dr2_top2 < 0.07282
        + 0.00378761 * max(0.0, Q.mass - 127.2317) / 9.462689   # +0.4%  mass > 127.2
        + 0.003736216 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, Q.M2 - 0.05353765) / 1.308851   # +0.4%  mass_top50 < 161.1 and M2 > 0.05354
        - 0.003579898 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # -0.4%  n_pairs_kt_above_1 < 58
        - 0.002980018 * max(0.0, Q.z_displaced3 - 0.4232894) / 0.005188716   # -0.3%  z_displaced3 > 0.4233
        + 0.002733007 * max(0.0, 22.56857 - Q.mass_charged) / 0.407248   # +0.3%  mass_charged < 22.57
        + 0.00253334 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.5871757 - Q.tau32) / 0.0105391   # +0.3%  lep_z < 0.3397 and tau32 < 0.5872
        - 0.002487465 * max(0.0, Q.n_pairs_kt_above_1 - 462.0) / 66.16959   # -0.2%  n_pairs_kt_above_1 > 462
        + 0.002461617 * max(0.0, 2.151069 - Q.sj3_mass3) / 0.3356229   # +0.2%  sj3_mass3 < 2.151
        + 0.002408592 * max(0.0, Q.sj3_pairmin_over_m - 0.4866692) / 0.003811804   # +0.2%  sj3_pairmin_over_m > 0.4867
        - 0.002351956 * max(0.0, Q.n_pt_above_1 - 54.0) / 1.37863   # -0.2%  n_pt_above_1 > 54
        - 0.002294325 * max(0.0, Q.z_top3_slots - 0.6094828) / 0.02331941   # -0.2%  z_top3_slots > 0.6095
        + 0.002145883 * max(0.0, Q.sj3_pair_mass_min - 44.1643) * max(0.0, 4.0 - Q.n_lund_kt_above_5) / 6.782067   # +0.2%  sj3_pair_mass_min > 44.16 and n_lund_kt_above_5 < 4
        + 0.002111563 * max(0.0, 0.2317874 - Q.sj3_dr13) / 0.02459729   # +0.2%  sj3_dr13 < 0.2318
        + 0.002019237 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.lne_0 - 4.727941) / 3.815164   # +0.2%  n_s3d_above_3 < 10 and lne_0 > 4.728
        - 0.001996308 * max(0.0, Q.lep_dr - 0.4191372) / 0.007569263   # -0.2%  lep_dr > 0.4191
        - 0.001943963 * max(0.0, 0.1049877 - Q.C2_b2) * max(0.0, 0.02747645 - Q.tdz_0) / 0.001924279   # -0.2%  C2_b2 < 0.105 and tdz_0 < 0.02748
        + 0.001769997 * max(0.0, 0.3273298 - Q.sj3_dr12) / 0.07555339   # +0.2%  sj3_dr12 < 0.3273
        + 0.001739523 * max(0.0, 32.61679 - Q.mass_displaced5) * max(0.0, Q.lep_dr - 0.1864963) / 0.7933211   # +0.2%  mass_displaced5 < 32.62 and lep_dr > 0.1865
        + 0.001591885 * max(0.0, Q.n_sd0_above_5 - 4.0) / 0.7095433   # +0.2%  n_sd0_above_5 > 4
        - 0.001547069 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, Q.D3_b2 - 0.4675349) / 2.889606   # -0.2%  lep_ptrel < 27.32 and D3_b2 > 0.4675
        - 0.00147373 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, 3.058345 - Q.lund3_lnkt) / 0.7271861   # -0.1%  lep_iso < 0.4382 and lund3_lnkt < 3.058
        - 0.00144644 * max(0.0, Q.mass_2charged - 24.4079) / 2.942332   # -0.1%  mass_2charged > 24.41
        + 0.001354257 * max(0.0, Q.mass - 90.08945) / 29.81377   # +0.1%  mass > 90.09
        + 0.001344328 * max(0.0, 0.01470468 - Q.tau3) / 0.0001776565   # +0.1%  tau3 < 0.0147
        + 0.00122652 * max(0.0, 48.76729 - Q.sj4_pair_mass_max) / 1.362821   # +0.1%  sj4_pair_mass_max < 48.77
        - 0.001182488 * max(0.0, Q.sj3_pair_mass_max - 128.6605) / 2.418855   # -0.1%  sj3_pair_mass_max > 128.7
        - 0.001147739 * max(0.0, 0.0856189 - Q.dr01) * max(0.0, Q.n_muon - 0.0) / 0.005054636   # -0.1%  dr01 < 0.08562 and n_muon > 0
        - 0.001120738 * max(0.0, Q.n_s3d_above_3 - 4.0) / 1.224997   # -0.1%  n_s3d_above_3 > 4
        - 0.001087119 * max(0.0, 4.326415 - Q.sj3_mass2) * max(0.0, 0.1405311 - Q.z_dr_0p05_0p1) / 0.01601388   # -0.1%  sj3_mass2 < 4.326 and z_dr_0p05_0p1 < 0.1405
        - 0.001056727 * max(0.0, Q.sj3_pair_mass_min - 44.1643) * max(0.0, 3.707456 - Q.lne_10) / 4.223242   # -0.1%  sj3_pair_mass_min > 44.16 and lne_10 < 3.707
        - 0.0009376249 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # -0.1%  sd_mass > 154.6
        - 0.0006810098 * max(0.0, Q.tau4 - 0.05417009) / 0.003306893   # -0.1%  tau4 > 0.05417
        + 0.0006557262 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # +0.1%  lep_z < 0.3397
        + 0.0006511741 * max(0.0, 0.02098847 - Q.tau5) / 0.002077107   # +0.1%  tau5 < 0.02099
        - 0.0005860776 * max(0.0, Q.sj3_mass1 - 44.1303) / 0.6338316   # -0.1%  sj3_mass1 > 44.13
        + 0.000569886 * max(0.0, 79.47361 - Q.mass) / 2.857843   # +0.1%  mass < 79.47
        + 0.0004768382 * max(0.0, Q.z_displaced5 - 0.06245248) * max(0.0, Q.dr_3 - 0.1990664) / 0.001456479   # +0.0%  z_displaced5 > 0.06245 and dr_3 > 0.1991
        + 0.0003456214 * max(0.0, 2.0 - Q.n_s3d_above_3) / 0.4717633   # +0.0%  n_s3d_above_3 < 2
        - 0.0003251259 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 2.246622e-05 - Q.C3_b2) / 1.757019e-06   # -0.0%  n_s3d_above_3 > 4 and C3_b2 < 2.247e-05
        + 0.0002839337 * max(0.0, Q.sj2_mass1 - 91.2852) / 1.019309   # +0.0%  sj2_mass1 > 91.29
        - 0.0002668272 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, Q.eta_14 - 0.1429443) / 0.4122901   # -0.0%  lep_ptrel < 27.32 and eta_14 > 0.1429
        + 0.0002663656 * max(0.0, 0.3273298 - Q.sj3_dr12) * max(0.0, Q.ecf_g42 - 1.874473e-05) / 4.076452e-07   # +0.0%  sj3_dr12 < 0.3273 and ecf_g42 > 1.874e-05
        + 3.459491e-05 * max(0.0, 68.20711 - Q.mass_charged) * max(0.0, Q.mass_2photon - 5.763861) / 40.17326   # +0.0%  mass_charged < 68.21 and mass_2photon > 5.764
        + 2.314198e-05 * max(0.0, Q.z_displaced5 - 0.06245248) * max(0.0, Q.iselectron_29 - 0.0) / 0.0002943135   # +0.0%  z_displaced5 > 0.06245 and iselectron_29 > 0
        + 1.163017e-06 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, Q.tdz_41 - 0.03879887) / 0.03419011   # +0.0%  max_abs_d0 < 10.52 and tdz_41 > 0.0388
    )
    return z


def neuron_71(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.029693e-06
    )
    return z


def neuron_72(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.164101e-06
    )
    return z


def neuron_73(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.699636e-05
    )
    return z


def neuron_74(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.493413e-06
    )
    return z


def neuron_75(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.973831e-06
    )
    return z


def neuron_76(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.542784e-06
    )
    return z


def neuron_77(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.087468e-06
    )
    return z


def neuron_78(Q):
    # scale S = 6.831; each line: share * term / its average size
    z = 6.830853 * (-0.165185
        + 0.103083 * Q.n_charged_had / 18.52697   # +10.3%  n_charged_had
        + 0.07584937 * max(0.0, 6.0 - Q.n_s3d_above_3) / 2.86534   # +7.6%  n_s3d_above_3 < 6
        + 0.05543445 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +5.5%  lep_iso < 0.4382
        - 0.05163373 * max(0.0, 12.15228 - Q.lep_ptrel) / 8.819929   # -5.2%  lep_ptrel < 12.15
        - 0.05023332 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # -5.0%  lep_ptrel < 27.32
        + 0.04836567 * max(0.0, 0.1024935 - Q.tau3) / 0.05489442   # +4.8%  tau3 < 0.1025
        - 0.04804041 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # -4.8%  lep_z < 0.3397
        + 0.0430452 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +4.3%  lep_ptrel < 43.21
        - 0.03909258 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # -3.9%  lep_iso < 1.362
        - 0.03307141 * Q.lne_0 / 5.19469   # -3.3%  lne_0
        + 0.02954007 * max(0.0, Q.M2 - 0.04276413) / 0.03598257   # +3.0%  M2 > 0.04276
        + 0.02691829 * max(0.0, 411.0 - Q.n_pairs_kt_above_1) / 159.5699   # +2.7%  n_pairs_kt_above_1 < 411
        + 0.02609778 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.177312 - Q.M2_b05) / 0.009855904   # +2.6%  lep_z < 0.3397 and M2_b05 < 0.1773
        - 0.02520855 * max(0.0, 18.7678 - Q.lep_ptrel) / 14.28156   # -2.5%  lep_ptrel < 18.77
        + 0.02444348 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # +2.4%  n_s3d_above_3 < 10
        - 0.01711384 * max(0.0, 0.1024935 - Q.tau3) * max(0.0, 577.991 - Q.sip_3d_3) / 29.11377   # -1.7%  tau3 < 0.1025 and sip_3d_3 < 578
        + 0.01579251 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # +1.6%  lep_z < 0.2214
        - 0.01489223 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.pair_mean_lnkt - -0.0709631) / 0.1542639   # -1.5%  lep_z < 0.3397 and pair_mean_lnkt > -0.07096
        - 0.01386182 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # -1.4%  max_abs_d0 < 10.52
        + 0.01209374 * max(0.0, Q.mass - 127.2317) / 9.462689   # +1.2%  mass > 127.2
        - 0.0116565 * max(0.0, 2.0 - Q.n_lepton) * max(0.0, 0.9333327 - Q.sj3_pairmax_over_m) / 0.1821352   # -1.2%  n_lepton < 2 and sj3_pairmax_over_m < 0.9333
        + 0.01080435 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 175.9957 - Q.sip_3d_3) / 116.3453   # +1.1%  n_s3d_above_3 > 4 and sip_3d_3 < 176
        + 0.009670581 * max(0.0, 0.3264446 - Q.D3_b2) / 0.2054222   # +1.0%  D3_b2 < 0.3264
        - 0.009569443 * max(0.0, Q.z_displaced3 - 0.02600452) / 0.09324453   # -1.0%  z_displaced3 > 0.026
        + 0.00849466 * max(0.0, Q.sj4_pair_mass_max - 78.76797) / 11.81635   # +0.8%  sj4_pair_mass_max > 78.77
        - 0.008412949 * max(0.0, 32.61679 - Q.mass_displaced5) / 27.02182   # -0.8%  mass_displaced5 < 32.62
        + 0.008019587 * max(0.0, 99.0 - Q.n_pairs_kt_above_3) / 39.22243   # +0.8%  n_pairs_kt_above_3 < 99
        - 0.00792833 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, 11.13866 - Q.sip_3d_3) / 1.560811   # -0.8%  lep_iso < 0.4382 and sip_3d_3 < 11.14
        - 0.007912159 * max(0.0, Q.sum_e - 550.9555) / 355.726   # -0.8%  sum_e > 551
        + 0.007874406 * max(0.0, Q.mass_charged - 37.42054) / 28.62477   # +0.8%  mass_charged > 37.42
        - 0.006958998 * max(0.0, Q.mass - 164.4374) / 3.038568   # -0.7%  mass > 164.4
        + 0.006829687 * max(0.0, 11.0 - Q.n_dr_0p1_0p2) / 2.841407   # +0.7%  n_dr_0p1_0p2 < 11
        - 0.006506444 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 15.0 - Q.n_dr_0p4_up) / 132.2245   # -0.7%  lep_ptrel < 18.77 and n_dr_0p4_up < 15
        - 0.006340386 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 0.5440886 - Q.tau21) / 0.2104196   # -0.6%  n_s3d_above_3 > 4 and tau21 < 0.5441
        - 0.006056816 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.sj3_dr_min - 0.08335692) / 0.7526776   # -0.6%  n_s3d_above_3 < 10 and sj3_dr_min > 0.08336
        + 0.006027922 * max(0.0, 4.606241 - Q.sip_3d_3) / 1.172198   # +0.6%  sip_3d_3 < 4.606
        - 0.005653794 * max(0.0, Q.pair_max_lnm2 - 7.347625) / 0.1021051   # -0.6%  pair_max_lnm2 > 7.348
        + 0.00523159 * max(0.0, Q.mass_top5 - 54.36876) / 5.778115   # +0.5%  mass_top5 > 54.37
        + 0.004754504 * max(0.0, Q.n_sd0_above_3 - 2.0) / 1.826197   # +0.5%  n_sd0_above_3 > 2
        + 0.004746837 * max(0.0, Q.n_s3d_above_3 - 4.0) / 1.224997   # +0.5%  n_s3d_above_3 > 4
        + 0.004694505 * max(0.0, Q.n_sd0_above_5 - 4.0) / 0.7095433   # +0.5%  n_sd0_above_5 > 4
        + 0.004382492 * max(0.0, 71.96396 - Q.mass) / 1.934957   # +0.4%  mass < 71.96
        + 0.004251839 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.1631992 - Q.sj4_dr_min) / 0.0170385   # +0.4%  lep_z < 0.3397 and sj4_dr_min < 0.1632
        - 0.004061393 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -0.4%  n_lepton < 1
        + 0.003982255 * max(0.0, 13.03663 - Q.mass_displaced3) / 8.661134   # +0.4%  mass_displaced3 < 13.04
        - 0.003971616 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, -1.332584 - Q.pair_mean_lnz) / 0.09084841   # -0.4%  lep_z < 0.3397 and pair_mean_lnz < -1.333
        - 0.003886838 * max(0.0, 67.20576 - Q.mass_top30) / 1.774095   # -0.4%  mass_top30 < 67.21
        + 0.003801977 * max(0.0, Q.z_photon - 0.3318968) / 0.02575489   # +0.4%  z_photon > 0.3319
        + 0.003739174 * max(0.0, Q.mass_top50 - 122.0585) / 10.10312   # +0.4%  mass_top50 > 122.1
        + 0.00362412 * max(0.0, 0.6206221 - Q.tau32) / 0.05684264   # +0.4%  tau32 < 0.6206
        - 0.003449168 * max(0.0, 2.0 - Q.n_lepton) / 1.440613   # -0.3%  n_lepton < 2
        - 0.003260263 * max(0.0, 0.6378426 - Q.z_charged) / 0.08720331   # -0.3%  z_charged < 0.6378
        + 0.003244058 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.mass_top20 - 86.78877) / 3.685704   # +0.3%  lep_z < 0.3397 and mass_top20 > 86.79
        + 0.003163038 * max(0.0, 21.21062 - Q.sj2_mass1) / 1.735779   # +0.3%  sj2_mass1 < 21.21
        + 0.002828348 * max(0.0, 2.0 - Q.n_dr_0_0p05) / 0.7938667   # +0.3%  n_dr_0_0p05 < 2
        + 0.002816318 * max(0.0, Q.mass_top15 - 80.3877) / 12.32501   # +0.3%  mass_top15 > 80.39
        + 0.002702824 * max(0.0, 0.8009208 - Q.psi_0p2) / 0.1053387   # +0.3%  psi_0p2 < 0.8009
        + 0.002601421 * max(0.0, 2.255345 - Q.D2) / 0.5917601   # +0.3%  D2 < 2.255
        - 0.002512228 * max(0.0, Q.mass_displaced5 - 6.387683) / 3.821495   # -0.3%  mass_displaced5 > 6.388
        + 0.002450789 * max(0.0, 1.102602 - Q.D2_b05) / 0.006783992   # +0.2%  D2_b05 < 1.103
        - 0.00239023 * max(0.0, 23.0 - Q.n_neutral) / 5.227467   # -0.2%  n_neutral < 23
        + 0.002223109 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 0.5699805 - Q.z_dr_0p1_0p2) / 3.961803   # +0.2%  lep_ptrel < 18.77 and z_dr_0p1_0p2 < 0.57
        - 0.002169864 * max(0.0, 2.0 - Q.n_dr_0_0p05) * max(0.0, 1.0 - Q.isnhad_27) / 0.74134   # -0.2%  n_dr_0_0p05 < 2 and isnhad_27 < 1
        - 0.002107353 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, 0.8436463 - Q.psi_0p2) / 2.692385   # -0.2%  lep_ptrel < 27.32 and psi_0p2 < 0.8436
        + 0.002074363 * max(0.0, Q.mass_displaced5 - 6.387683) * max(0.0, 14.38858 - Q.lep_iso) / 40.3921   # +0.2%  mass_displaced5 > 6.388 and lep_iso < 14.39
        - 0.001414636 * max(0.0, Q.sj4_pair_mass_max - 115.5142) / 2.08655   # -0.1%  sj4_pair_mass_max > 115.5
        + 0.00131625 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 0.3190414 - Q.sj3_dr_min) / 0.14046   # +0.1%  n_s3d_above_3 > 4 and sj3_dr_min < 0.319
        - 0.001185555 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 23.30856 - Q.sip_3d_3) / 97.09082   # -0.1%  max_abs_d0 < 10.52 and sip_3d_3 < 23.31
        + 0.001002469 * max(0.0, Q.n_pairs_kt_above_3 - 220.0) / 4.57629   # +0.1%  n_pairs_kt_above_3 > 220
        - 0.0009908452 * max(0.0, 0.6206221 - Q.tau32) * max(0.0, Q.jet_charge_k03 - 0.02942741) / 0.01536299   # -0.1%  tau32 < 0.6206 and jet_charge_k03 > 0.02943
        - 0.0009685837 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # -0.1%  mass_top50 > 161.1
        + 0.0008997768 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 8.17233e-05 - Q.ecf_g42) / 1.595489e-05   # +0.1%  lep_z < 0.3397 and ecf_g42 < 8.172e-05
        - 0.0008543017 * max(0.0, Q.n_electron - 1.0) / 0.08167   # -0.1%  n_electron > 1
        + 0.0008319669 * max(0.0, Q.lund2_lndelta - -1.556018) / 0.6029678   # +0.1%  lund2_lndelta > -1.556
        + 0.0007035239 * max(0.0, Q.z_displaced5 - 0.1709091) / 0.02847701   # +0.1%  z_displaced5 > 0.1709
        - 0.0006753797 * max(0.0, Q.psi_0p1 - 0.9356675) / 0.00150838   # -0.1%  psi_0p1 > 0.9357
        - 0.0006081047 * max(0.0, Q.mass_displaced5 - 6.387683) * max(0.0, 1.0 - Q.n_lepton) / 1.715768   # -0.1%  mass_displaced5 > 6.388 and n_lepton < 1
        - 0.0005913108 * max(0.0, 0.07985021 - Q.sj4_dr_min) / 0.01211577   # -0.1%  sj4_dr_min < 0.07985
        + 0.0005528249 * max(0.0, 0.2458451 - Q.sj3_dr23) / 0.02482854   # +0.1%  sj3_dr23 < 0.2458
        - 0.0005366946 * max(0.0, Q.z_neutral_had - 0.3788785) / 0.004557199   # -0.1%  z_neutral_had > 0.3789
        - 0.0005043683 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, 0.03430176 - Q.dzerr_0) / 0.1014645   # -0.1%  n_s3d_above_3 < 10 and dzerr_0 < 0.0343
        + 0.00045803 * max(0.0, Q.mass_displaced3 - 39.09615) / 0.6617151   # +0.0%  mass_displaced3 > 39.1
        - 0.0004395539 * max(0.0, 21.21062 - Q.sj2_mass1) * max(0.0, 0.04891968 - Q.phi_26) / 0.1248013   # -0.0%  sj2_mass1 < 21.21 and phi_26 < 0.04892
        + 0.0004283345 * max(0.0, 0.0007393085 - Q.e3) / 0.000161292   # +0.0%  e3 < 0.0007393
        + 0.0002932967 * max(0.0, Q.mass_displaced5 - 6.387683) * max(0.0, 11.13866 - Q.sip_3d_3) / 0.8594348   # +0.0%  mass_displaced5 > 6.388 and sip_3d_3 < 11.14
        + 0.0001627596 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, -0.210083 - Q.eta_28) / 0.3467154   # +0.0%  lep_ptrel < 27.32 and eta_28 < -0.2101
        + 0.0001409656 * max(0.0, Q.mass_displaced3 - 39.09615) * max(0.0, 33.08364 - Q.sip_3d_3) / 2.849973   # +0.0%  mass_displaced3 > 39.1 and sip_3d_3 < 33.08
        + 0.0001370998 * max(0.0, Q.z_displaced3 - 0.02600452) * max(0.0, Q.isphoton_14 - 0.0) / 0.02961916   # +0.0%  z_displaced3 > 0.026 and isphoton_14 > 0
        - 0.0001318719 * max(0.0, 0.1024935 - Q.tau3) * max(0.0, Q.tdz_33 - -0.01794241) / 0.001834404   # -0.0%  tau3 < 0.1025 and tdz_33 > -0.01794
        - 0.0001176469 * max(0.0, Q.n_sd0_above_5 - 4.0) * max(0.0, 11.13866 - Q.sip_3d_3) / 0.01232867   # -0.0%  n_sd0_above_5 > 4 and sip_3d_3 < 11.14
        + 0.0001153197 * max(0.0, Q.mass_top40 - 173.1022) / 1.415364   # +0.0%  mass_top40 > 173.1
        + 6.647647e-05 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # +0.0%  mass_top10 > 103.5
        - 6.424425e-05 * max(0.0, 1.104348 - Q.lead_ch_sdz) * max(0.0, Q.iselectron_1 - 0.0) / 0.05540389   # -0.0%  lead_ch_sdz < 1.104 and iselectron_1 > 0
        - 6.341306e-05 * max(0.0, 1.104348 - Q.lead_ch_sdz) / 1.96298   # -0.0%  lead_ch_sdz < 1.104
        - 3.74147e-05 * max(0.0, Q.mass_displaced5 - 6.387683) * max(0.0, 0.8156512 - Q.psi_0p3) / 0.1625677   # -0.0%  mass_displaced5 > 6.388 and psi_0p3 < 0.8157
        + 2.866321e-05 * max(0.0, Q.sum_e - 550.9555) * max(0.0, Q.ismuon_4 - 0.0) / 2.93247   # +0.0%  sum_e > 551 and ismuon_4 > 0
        - 1.47658e-05 * max(0.0, Q.n_electron - 1.0) * max(0.0, Q.iselectron_30 - 0.0) / 0.003523333   # -0.0%  n_electron > 1 and iselectron_30 > 0
        - 1.253878e-05 * max(0.0, Q.mass_top40 - 173.1022) * max(0.0, 1.0 - Q.n_muon) / 1.026809   # -0.0%  mass_top40 > 173.1 and n_muon < 1
    )
    return z


def neuron_79(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.602393e-05
    )
    return z


def neuron_80(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.707704e-06
    )
    return z


def neuron_81(Q):
    # scale S = 4.768; each line: share * term / its average size
    z = 4.767506 * (-0.1024316
        + 0.0688396 * max(0.0, Q.mass_top40 - 97.12186) / 20.78291   # +6.9%  mass_top40 > 97.12
        - 0.06680779 * max(0.0, Q.mass_top50 - 99.54528) / 21.60123   # -6.7%  mass_top50 > 99.55
        - 0.05519794 * Q.M2 / 0.07774996   # -5.5%  M2
        + 0.05076391 * max(0.0, 13.97388 - Q.mass_displaced5) / 10.07648   # +5.1%  mass_displaced5 < 13.97
        + 0.04987177 * max(0.0, Q.mass_top50 - 115.614) / 12.67421   # +5.0%  mass_top50 > 115.6
        + 0.044948 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 0.03021637 - Q.lep_z) / 0.001161425   # +4.5%  z_displaced3 < 0.1062 and lep_z < 0.03022
        - 0.04490662 * max(0.0, 0.1076451 - Q.tau2) / 0.0393622   # -4.5%  tau2 < 0.1076
        + 0.03605726 * max(0.0, 0.4242439 - Q.LHA) / 0.05026985   # +3.6%  LHA < 0.4242
        + 0.03441333 * max(0.0, 5.0 - Q.n_s3d_above_3) / 2.12335   # +3.4%  n_s3d_above_3 < 5
        - 0.0332621 * max(0.0, Q.mass_top40 - 119.1279) / 9.647038   # -3.3%  mass_top40 > 119.1
        + 0.0316117 * max(0.0, Q.n_s3d_above_3 - 3.0) / 1.66923   # +3.2%  n_s3d_above_3 > 3
        - 0.02654277 * max(0.0, 0.1061578 - Q.z_displaced3) / 0.05189409   # -2.7%  z_displaced3 < 0.1062
        - 0.02487318 * max(0.0, 15.0 - Q.n_dr_0p4_up) / 9.48698   # -2.5%  n_dr_0p4_up < 15
        - 0.02431899 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 7.345216 - Q.sip_3d_3) / 20.68407   # -2.4%  max_abs_d0 < 10.52 and sip_3d_3 < 7.345
        + 0.02151002 * max(0.0, 6.983043 - Q.lep_ptrel) / 4.807808   # +2.2%  lep_ptrel < 6.983
        - 0.02122682 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.z_charged_had - 0.265564) / 0.02462507   # -2.1%  z_displaced3 > 0.03624 and z_charged_had > 0.2656
        + 0.02096264 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # +2.1%  lep_z < 0.2214
        + 0.02043213 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # +2.0%  max_abs_d0 < 5.812
        + 0.01864661 * max(0.0, Q.z_displaced3 - 0.03624058) / 0.08737704   # +1.9%  z_displaced3 > 0.03624
        - 0.01831272 * max(0.0, 1.777286 - Q.mass_displaced3) / 0.7377974   # -1.8%  mass_displaced3 < 1.777
        - 0.01810378 * max(0.0, 0.01329067 - Q.z_electron) / 0.01018622   # -1.8%  z_electron < 0.01329
        - 0.01644164 * max(0.0, Q.sj4_pair_mass_max - 63.34656) / 21.48011   # -1.6%  sj4_pair_mass_max > 63.35
        + 0.01606708 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 33.0 - Q.n_charged_had) / 19.17455   # +1.6%  n_s3d_above_3 > 3 and n_charged_had < 33
        + 0.01500242 * max(0.0, Q.n_s3d_above_10 - 3.0) / 0.9107233   # +1.5%  n_s3d_above_10 > 3
        + 0.01257319 * max(0.0, Q.mass_top50 - 125.4927) / 8.989867   # +1.3%  mass_top50 > 125.5
        - 0.01188201 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 21.12768 - Q.mass_displaced5) / 13.37539   # -1.2%  n_s3d_above_3 > 3 and mass_displaced5 < 21.13
        + 0.01177345 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 447.0873 - Q.sip_3d_2) / 370.2766   # +1.2%  n_s3d_above_3 > 3 and sip_3d_2 < 447.1
        - 0.01097642 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 0.3014662 - Q.lep_dr) / 0.01653138   # -1.1%  z_displaced3 > 0.03624 and lep_dr < 0.3015
        - 0.01004947 * max(0.0, 0.7634316 - Q.max_dr) / 0.1364426   # -1.0%  max_dr < 0.7634
        - 0.009733571 * max(0.0, Q.n_sd0_above_5 - 5.0) / 0.4736133   # -1.0%  n_sd0_above_5 > 5
        - 0.008568669 * max(0.0, 6.983043 - Q.lep_ptrel) * max(0.0, Q.pt_balance01 - 0.1985369) / 0.9136075   # -0.9%  lep_ptrel < 6.983 and pt_balance01 > 0.1985
        + 0.008109897 * max(0.0, Q.sj4_pair_mass_max - 63.34656) * max(0.0, 0.3788785 - Q.z_neutral_had) / 4.99524   # +0.8%  sj4_pair_mass_max > 63.35 and z_neutral_had < 0.3789
        + 0.008056992 * max(0.0, 1.203438 - Q.jet_abs_eta) / 0.5210607   # +0.8%  jet_abs_eta < 1.203
        + 0.007890264 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # +0.8%  lep_ptrel > 27.32
        - 0.007222401 * max(0.0, 22.0 - Q.n_pairs_kt_above_3) * max(0.0, 1.0 - Q.n_pairs_kt_above_30) / 1.994697   # -0.7%  n_pairs_kt_above_3 < 22 and n_pairs_kt_above_30 < 1
        + 0.006789505 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.n_lund - 5.0) / 0.5382206   # +0.7%  z_displaced3 > 0.03624 and n_lund > 5
        - 0.006514239 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -0.7%  sip_3d_2 < 226.3
        - 0.006464868 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 1.839882 - Q.mass_2charged) / 0.03997451   # -0.6%  z_displaced3 > 0.03624 and mass_2charged < 1.84
        + 0.006080275 * max(0.0, 0.2377332 - Q.tau21) / 0.01385845   # +0.6%  tau21 < 0.2377
        - 0.00606804 * max(0.0, 8.0 - Q.n_sd0_above_3) / 4.91424   # -0.6%  n_sd0_above_3 < 8
        - 0.005941609 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 0.8882532 - Q.tau32) / 0.3641546   # -0.6%  n_s3d_above_3 > 3 and tau32 < 0.8883
        - 0.005503769 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # -0.6%  n_pairs_kt_above_1 < 80
        - 0.005034424 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # -0.5%  max_abs_d0 < 10.52
        + 0.004761364 * max(0.0, 2.723325 - Q.sip_3d_3) / 0.2881626   # +0.5%  sip_3d_3 < 2.723
        + 0.004512466 * max(0.0, 577.991 - Q.sip_3d_3) / 516.7039   # +0.5%  sip_3d_3 < 578
        - 0.004198013 * max(0.0, 0.0009010251 - Q.ecf_g32) / 7.436273e-05   # -0.4%  ecf_g32 < 0.000901
        - 0.003921556 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # -0.4%  mass_top50 > 161.1
        + 0.00389928 * max(0.0, 52.72207 - Q.mass_top30) / 0.741523   # +0.4%  mass_top30 < 52.72
        + 0.003705497 * max(0.0, 0.07290954 - Q.sum_z_dr) / 0.002660742   # +0.4%  sum_z_dr < 0.07291
        - 0.003021044 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 0.8597199 - Q.tau43) / 0.1092297   # -0.3%  n_s3d_above_3 > 3 and tau43 < 0.8597
        + 0.002960454 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 16.23838 - Q.sip_3d_3) / 0.2292165   # +0.3%  z_displaced3 > 0.03624 and sip_3d_3 < 16.24
        + 0.002841098 * max(0.0, Q.pair_mean_lndelta - -1.352792) / 0.01174331   # +0.3%  pair_mean_lndelta > -1.353
        - 0.002548554 * max(0.0, 0.02949822 - Q.tau4) / 0.003969373   # -0.3%  tau4 < 0.0295
        - 0.002388041 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 0.9036234 - Q.tau54) / 0.08976281   # -0.2%  n_s3d_above_3 > 3 and tau54 < 0.9036
        + 0.002338444 * max(0.0, 1.763283 - Q.min_pair_mass) / 0.6216059   # +0.2%  min_pair_mass < 1.763
        + 0.00181239 * max(0.0, Q.n_s3d_above_3 - 10.0) / 0.12486   # +0.2%  n_s3d_above_3 > 10
        + 0.001640841 * max(0.0, Q.mass_2charged - 20.76537) / 3.761489   # +0.2%  mass_2charged > 20.77
        - 0.001618883 * max(0.0, Q.sj3_pairmin_over_m - 0.2477126) / 0.09094898   # -0.2%  sj3_pairmin_over_m > 0.2477
        + 0.001427734 * max(0.0, Q.n_s3d_above_3 - 10.0) * max(0.0, 1.0 - Q.isphoton_0) / 0.09957333   # +0.1%  n_s3d_above_3 > 10 and isphoton_0 < 1
        - 0.001426749 * max(0.0, 0.01181938 - Q.z_dr_0p4_up) / 0.003195022   # -0.1%  z_dr_0p4_up < 0.01182
        + 0.001417312 * max(0.0, 0.2377332 - Q.tau21) * max(0.0, 0.1985369 - Q.pt_balance01) / 0.0002543562   # +0.1%  tau21 < 0.2377 and pt_balance01 < 0.1985
        - 0.001234387 * max(0.0, Q.n_sd0_above_3 - 9.0) / 0.1165233   # -0.1%  n_sd0_above_3 > 9
        + 0.001216798 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.sj2_dr - 0.2403736) / 0.015899   # +0.1%  z_displaced3 > 0.03624 and sj2_dr > 0.2404
        + 0.00120864 * max(0.0, Q.mass_charged - 99.20396) / 2.066138   # +0.1%  mass_charged > 99.2
        + 0.00119705 * max(0.0, Q.tdz_2 - -0.02221619) / 0.04771882   # +0.1%  tdz_2 > -0.02222
        + 0.001057059 * max(0.0, 0.1076451 - Q.tau2) * max(0.0, Q.n_muon - 0.0) / 0.009639314   # +0.1%  tau2 < 0.1076 and n_muon > 0
        + 0.0008465239 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.z_photon - 0.3887278) / 0.0005662707   # +0.1%  z_displaced3 > 0.03624 and z_photon > 0.3887
        - 0.0008283244 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.1%  lep_ptrel > 43.21
        - 0.0008066193 * max(0.0, 22.0 - Q.n_pairs_kt_above_3) / 2.320537   # -0.1%  n_pairs_kt_above_3 < 22
        - 0.0007747114 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 0.04032236 - Q.td0_33) / 0.004915466   # -0.1%  z_displaced3 > 0.03624 and td0_33 < 0.04032
        - 0.0007239514 * max(0.0, Q.mass_top40 - 119.1279) * max(0.0, 1.0 - Q.isphoton_32) / 5.648897   # -0.1%  mass_top40 > 119.1 and isphoton_32 < 1
        + 0.0007017048 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.z_muon - 0.03898651) / 0.003551335   # +0.1%  z_displaced3 > 0.03624 and z_muon > 0.03899
        - 0.0005632175 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, Q.n_lund_kt_above_5 - 2.0) / 0.7653267   # -0.1%  n_s3d_above_3 > 3 and n_lund_kt_above_5 > 2
        + 0.0005309355 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 0.2477181 - Q.pt1_over_pt0) / 0.01004037   # +0.1%  n_s3d_above_3 > 3 and pt1_over_pt0 < 0.2477
        + 0.0005169772 * max(0.0, -0.1170754 - Q.tdz_1) / 0.01260556   # +0.1%  tdz_1 < -0.1171
        - 0.0005087502 * max(0.0, Q.mass_top40 - 97.12186) * max(0.0, Q.tau43_b2 - 0.6509029) / 2.050221   # -0.1%  mass_top40 > 97.12 and tau43_b2 > 0.6509
        + 0.000506642 * max(0.0, Q.sj4_pair_mass_max - 63.34656) * max(0.0, 0.1695557 - Q.eta_37) / 4.215693   # +0.1%  sj4_pair_mass_max > 63.35 and eta_37 < 0.1696
        - 0.0003775923 * max(0.0, Q.z_displaced5 - 0.01810676) / 0.08074326   # -0.0%  z_displaced5 > 0.01811
        - 0.0003106117 * max(0.0, 0.1076451 - Q.tau2) * max(0.0, Q.jet_charge_k03 - 0.2021837) / 0.007468288   # -0.0%  tau2 < 0.1076 and jet_charge_k03 > 0.2022
        + 0.0002933711 * max(0.0, 70.05844 - Q.sip_3d_2) / 37.02037   # +0.0%  sip_3d_2 < 70.06
        - 0.0002496014 * max(0.0, 1.763283 - Q.min_pair_mass) * max(0.0, 1.0 - Q.isnhad_8) / 0.5334306   # -0.0%  min_pair_mass < 1.763 and isnhad_8 < 1
        - 0.0002446505 * max(0.0, 5.0 - Q.n_s3d_above_3) * max(0.0, Q.eccentricity - 0.5174679) / 0.5955715   # -0.0%  n_s3d_above_3 < 5 and eccentricity > 0.5175
        - 0.0001919352 * max(0.0, Q.tdz_2 - -0.02221619) * max(0.0, 0.0881958 - Q.dzerr_44) / 0.003828339   # -0.0%  tdz_2 > -0.02222 and dzerr_44 < 0.0882
        + 0.0001200286 * max(0.0, Q.sj3_pairmin_over_m - 0.2477126) * max(0.0, 0.2877534 - Q.dr_16) / 0.01048531   # +0.0%  sj3_pairmin_over_m > 0.2477 and dr_16 < 0.2878
        - 0.000106042 * max(0.0, 0.4242439 - Q.LHA) * max(0.0, 8.0 - Q.n_lund) / 0.01975455   # -0.0%  LHA < 0.4242 and n_lund < 8
        - 2.740523e-05 * max(0.0, Q.tdz_2 - -0.02221619) * max(0.0, Q.ismuon_12 - 0.0) / 0.0002048848   # -0.0%  tdz_2 > -0.02222 and ismuon_12 > 0
        - 2.120727e-05 * max(0.0, 6.983043 - Q.lep_ptrel) * max(0.0, Q.ismuon_11 - 0.0) / 0.009388841   # -0.0%  lep_ptrel < 6.983 and ismuon_11 > 0
        - 1.379998e-05 * max(0.0, Q.n_sd0_above_3 - 9.0) * max(0.0, 0.0 - Q.tdz_47) / 0.003538555   # -0.0%  n_sd0_above_3 > 9 and tdz_47 < 0
        - 1.162576e-05 * max(0.0, 0.7634316 - Q.max_dr) * max(0.0, Q.ismuon_49 - 0.0) / 1.799864e-05   # -0.0%  max_dr < 0.7634 and ismuon_49 > 0
        + 9.036533e-06 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, Q.isnhad_34 - 0.0) / 0.08861667   # +0.0%  n_s3d_above_3 > 3 and isnhad_34 > 0
        - 6.852504e-06 * max(0.0, 6.983043 - Q.lep_ptrel) * max(0.0, Q.ismuon_73 - 0.0) / 0.0001851159   # -0.0%  lep_ptrel < 6.983 and ismuon_73 > 0
        - 3.704964e-07 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.phi_79 - 0.0) / 4.961935e-05   # -0.0%  z_displaced3 > 0.03624 and phi_79 > 0
    )
    return z


def neuron_82(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.942086e-06
    )
    return z


def neuron_83(Q):
    # scale S = 12.52; each line: share * term / its average size
    z = 12.52067 * (0.07990008
        - 0.1167994 * max(0.0, 173.1022 - Q.mass_top40) / 64.05167   # -11.7%  mass_top40 < 173.1
        + 0.08998288 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # +9.0%  n_s3d_above_3 < 10
        - 0.08619591 * max(0.0, Q.sd_mass - 83.61981) / 26.48984   # -8.6%  sd_mass > 83.62
        + 0.06683867 * max(0.0, Q.sd_mass - 62.03827) / 42.41619   # +6.7%  sd_mass > 62.04
        - 0.04811609 * max(0.0, 0.000141779 - Q.e4_b05) / 0.0001033822   # -4.8%  e4_b05 < 0.0001418
        + 0.04071252 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # +4.1%  lep_ptrel < 27.32
        - 0.03003184 * max(0.0, Q.mass_top40 - 78.33213) * max(0.0, 15.17086 - Q.D2_b2) / 454.9867   # -3.0%  mass_top40 > 78.33 and D2_b2 < 15.17
        - 0.02829587 * max(0.0, Q.mass_top40 - 78.33213) / 34.93312   # -2.8%  mass_top40 > 78.33
        + 0.02777201 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # +2.8%  sd_mass > 88.82
        + 0.02532642 * max(0.0, 128.0079 - Q.sj4_pair_mass_max) / 48.13337   # +2.5%  sj4_pair_mass_max < 128
        - 0.02403202 * max(0.0, Q.mass_top40 - 78.33213) * max(0.0, 18.7678 - Q.lep_ptrel) / 484.8844   # -2.4%  mass_top40 > 78.33 and lep_ptrel < 18.77
        + 0.0223171 * max(0.0, 117.4867 - Q.mass) / 15.64803   # +2.2%  mass < 117.5
        - 0.02170571 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.z_neutral - 0.1384639) / 1.740792   # -2.2%  n_s3d_above_3 < 10 and z_neutral > 0.1385
        - 0.02030213 * max(0.0, 7.0 - Q.n_sd0_above_10) / 4.781883   # -2.0%  n_sd0_above_10 < 7
        - 0.01680611 * max(0.0, Q.lund3_lndelta - -2.817283) / 1.325774   # -1.7%  lund3_lndelta > -2.817
        + 0.01672142 * max(0.0, 0.05018249 - Q.sum_zz_dr2) / 0.01730417   # +1.7%  sum_zz_dr2 < 0.05018
        + 0.01428844 * max(0.0, 0.1681173 - Q.z_displaced3) / 0.09545189   # +1.4%  z_displaced3 < 0.1681
        + 0.01382429 * max(0.0, 13.03663 - Q.mass_displaced3) / 8.661134   # +1.4%  mass_displaced3 < 13.04
        + 0.01350428 * Q.ecf_g41 / 0.0002120905   # +1.4%  ecf_g41
        - 0.01310597 * max(0.0, Q.mass_top15 - 80.3877) / 12.32501   # -1.3%  mass_top15 > 80.39
        + 0.01021122 * max(0.0, 33.0 - Q.n_charged) / 14.10218   # +1.0%  n_charged < 33
        - 0.01008448 * max(0.0, 30.49226 - Q.sip_3d_2) / 12.97731   # -1.0%  sip_3d_2 < 30.49
        - 0.009979514 * max(0.0, 30.0 - Q.n_pt_above_5) / 9.011497   # -1.0%  n_pt_above_5 < 30
        - 0.009560927 * max(0.0, Q.e2 - 0.04393457) / 0.04061416   # -1.0%  e2 > 0.04393
        - 0.009408022 * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0001737301   # -0.9%  e3_b2 < 0.0002537
        + 0.009246749 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # +0.9%  n_s3d_above_3 < 3
        - 0.008847468 * max(0.0, 1.251841 - Q.mass_displaced3) / 0.4870081   # -0.9%  mass_displaced3 < 1.252
        - 0.00877362 * max(0.0, Q.mass_top30 - 102.3652) / 13.48953   # -0.9%  mass_top30 > 102.4
        + 0.008752136 * max(0.0, 0.06416437 - Q.z_displaced3) / 0.02655242   # +0.9%  z_displaced3 < 0.06416
        + 0.008674978 * max(0.0, Q.pair_mean_lnm2 - 1.96906) / 0.7951971   # +0.9%  pair_mean_lnm2 > 1.969
        + 0.008392379 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 175.9957 - Q.sip_3d_3) / 54.70527   # +0.8%  n_s3d_above_3 > 6 and sip_3d_3 < 176
        + 0.008020265 * max(0.0, 4.636903 - Q.sip_3d_2) / 0.784109   # +0.8%  sip_3d_2 < 4.637
        + 0.007727354 * max(0.0, 173.1022 - Q.mass_top40) * max(0.0, 15.17086 - Q.D2_b2) / 720.4267   # +0.8%  mass_top40 < 173.1 and D2_b2 < 15.17
        - 0.007706662 * max(0.0, 0.4265629 - Q.N2_b05) / 0.02389014   # -0.8%  N2_b05 < 0.4266
        + 0.007537987 * max(0.0, Q.tau5 - 0.01222366) / 0.0203374   # +0.8%  tau5 > 0.01222
        - 0.007272915 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -0.7%  sip_3d_2 < 226.3
        - 0.00709801 * max(0.0, 6.661019 - Q.log_sum_pt) / 0.2482436   # -0.7%  log_sum_pt < 6.661
        + 0.006661956 * max(0.0, Q.n_s3d_above_3 - 6.0) / 0.6254467   # +0.7%  n_s3d_above_3 > 6
        - 0.006103689 * max(0.0, 0.3112717 - Q.sj4_dr_min) / 0.1919254   # -0.6%  sj4_dr_min < 0.3113
        - 0.006048874 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # -0.6%  max_abs_d0 < 10.52
        + 0.005981922 * max(0.0, 0.8799072 - Q.sj3_dr23) / 0.4603866   # +0.6%  sj3_dr23 < 0.8799
        + 0.005934847 * max(0.0, Q.N2_b05 - 0.4518419) / 0.02228442   # +0.6%  N2_b05 > 0.4518
        - 0.005804574 * max(0.0, 2.448333e-05 - Q.ecf_g42) / 8.669871e-06   # -0.6%  ecf_g42 < 2.448e-05
        + 0.005594744 * max(0.0, Q.mass_top40 - 155.8928) / 2.693823   # +0.6%  mass_top40 > 155.9
        - 0.005447292 * max(0.0, 49.25971 - Q.mass_neutral) / 11.9268   # -0.5%  mass_neutral < 49.26
        + 0.005249656 * max(0.0, Q.mass_top40 - 119.1279) / 9.647038   # +0.5%  mass_top40 > 119.1
        + 0.004574793 * max(0.0, 2.448333e-05 - Q.ecf_g42) * max(0.0, 6.0 - Q.n_s3d_above_10) / 3.287542e-05   # +0.5%  ecf_g42 < 2.448e-05 and n_s3d_above_10 < 6
        - 0.00443957 * max(0.0, Q.e2 - 0.07745967) / 0.01550545   # -0.4%  e2 > 0.07746
        - 0.004352194 * max(0.0, 0.2250047 - Q.C3_b05) / 0.1186192   # -0.4%  C3_b05 < 0.225
        + 0.00433632 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +0.4%  sd_mass > 154.6
        - 0.004110794 * max(0.0, 8.0 - Q.n_sd0_above_3) / 4.91424   # -0.4%  n_sd0_above_3 < 8
        - 0.003738564 * max(0.0, 72.862 - Q.sj4_pair_mass_max) / 6.86631   # -0.4%  sj4_pair_mass_max < 72.86
        + 0.003105758 * max(0.0, Q.n_s3d_above_10 - 1.0) * max(0.0, 351.3271 - Q.sip_3d_1) / 195.5765   # +0.3%  n_s3d_above_10 > 1 and sip_3d_1 < 351.3
        - 0.003002588 * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 1.107657   # -0.3%  n_lund_kt_above_5 > 1
        - 0.002607657 * max(0.0, 0.06228948 - Q.M2) / 0.004761627   # -0.3%  M2 < 0.06229
        + 0.002463287 * max(0.0, Q.n_s3d_above_10 - 1.0) / 1.904323   # +0.2%  n_s3d_above_10 > 1
        + 0.002169311 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 114.8082   # +0.2%  mass < 117.5 and n_dr_0p1_0p2 > 1
        - 0.002116495 * max(0.0, 2.0 - Q.n_lepton) / 1.440613   # -0.2%  n_lepton < 2
        - 0.002074943 * max(0.0, 0.004294711 - Q.z_dr_0_0p05) / 0.00136489   # -0.2%  z_dr_0_0p05 < 0.004295
        + 0.001864136 * max(0.0, Q.sj4_pair_mass_max - 115.5142) / 2.08655   # +0.2%  sj4_pair_mass_max > 115.5
        + 0.00152352 * max(0.0, Q.pair_max_lnm2 - 7.347625) / 0.1021051   # +0.2%  pair_max_lnm2 > 7.348
        + 0.001377014 * max(0.0, 0.00228569 - Q.e3) * max(0.0, Q.jet_abs_eta - 0.9149342) / 0.0001688642   # +0.1%  e3 < 0.002286 and jet_abs_eta > 0.9149
        + 0.001344983 * max(0.0, Q.n_s3d_above_10 - 1.0) * max(0.0, 0.3264446 - Q.D3_b2) / 0.409072   # +0.1%  n_s3d_above_10 > 1 and D3_b2 < 0.3264
        + 0.001328093 * max(0.0, Q.n_sd0_above_5 - 7.0) / 0.1953867   # +0.1%  n_sd0_above_5 > 7
        - 0.001141982 * max(0.0, Q.pair_mean_lndelta - -2.115543) / 0.1901107   # -0.1%  pair_mean_lndelta > -2.116
        + 0.001031275 * max(0.0, Q.z_photon - 0.4305934) / 0.009360866   # +0.1%  z_photon > 0.4306
        + 0.001021007 * max(0.0, 0.3951525 - Q.tau32) / 0.009206105   # +0.1%  tau32 < 0.3952
        + 0.0009387608 * max(0.0, 0.3603262 - Q.z_charged_had) / 0.02270075   # +0.1%  z_charged_had < 0.3603
        - 0.0009373692 * max(0.0, Q.psi_0p2 - 0.8857951) / 0.01931927   # -0.1%  psi_0p2 > 0.8858
        - 0.0008091843 * max(0.0, 0.3977051 - Q.max_abs_d0) / 0.09373534   # -0.1%  max_abs_d0 < 0.3977
        - 0.0008008129 * max(0.0, 0.02210827 - Q.dr_max_012) / 0.001265414   # -0.1%  dr_max_012 < 0.02211
        - 0.0007670527 * max(0.0, 0.05018249 - Q.sum_zz_dr2) * max(0.0, 16.0 - Q.n_lund) / 0.09623434   # -0.1%  sum_zz_dr2 < 0.05018 and n_lund < 16
        - 0.0007635565 * max(0.0, Q.n_s3d_above_10 - 1.0) * max(0.0, 0.3788372 - Q.dr_16) / 0.3385076   # -0.1%  n_s3d_above_10 > 1 and dr_16 < 0.3788
        - 0.0006879619 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, Q.sj3_pairmin_over_m - 0.1636952) / 0.09021261   # -0.1%  n_s3d_above_3 > 6 and sj3_pairmin_over_m > 0.1637
        - 0.0006844733 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.1%  mass < 90.09
        + 0.0004326376 * max(0.0, Q.z_neutral_had - 0.3788785) / 0.004557199   # +0.0%  z_neutral_had > 0.3789
        + 0.0003918863 * max(0.0, 6.185635 - Q.lep_iso) / 4.883283   # +0.0%  lep_iso < 6.186
        + 0.0002883855 * max(0.0, Q.td0_3 - 0.1091249) / 0.01632519   # +0.0%  td0_3 > 0.1091
        + 0.0002871542 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 0.2949288 - Q.z_dr_0p1_0p2) / 0.04689436   # +0.0%  n_s3d_above_3 > 6 and z_dr_0p1_0p2 < 0.2949
        + 0.0001841903 * max(0.0, 0.5779883 - Q.pt1_over_pt0) / 0.08763774   # +0.0%  pt1_over_pt0 < 0.578
        + 0.0001740623 * max(0.0, 173.1022 - Q.mass_top40) * max(0.0, Q.mass_displaced5 - 0.0) / 311.5677   # +0.0%  mass_top40 < 173.1 and mass_displaced5 > 0
        - 0.0001651802 * max(0.0, 0.01517988 - Q.M3) / 0.0001250003   # -0.0%  M3 < 0.01518
        + 0.0001589654 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # +0.0%  mass_top10 > 103.5
        - 0.0001528104 * max(0.0, Q.mass_top10 - 103.4976) * max(0.0, Q.mratio_min_012 - 0.0111128) / 0.1811386   # -0.0%  mass_top10 > 103.5 and mratio_min_012 > 0.01111
        + 0.0001379911 * max(0.0, 0.06416437 - Q.z_displaced3) * max(0.0, 0.2370407 - Q.z_neutral_had) / 0.002830821   # +0.0%  z_displaced3 < 0.06416 and z_neutral_had < 0.237
        - 0.0001196541 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, Q.phi_3 - 0.06884766) / 0.01587558   # -0.0%  n_s3d_above_3 > 6 and phi_3 > 0.06885
        - 0.0001065824 * max(0.0, 0.00228569 - Q.e3) / 0.001253813   # -0.0%  e3 < 0.002286
        + 8.989731e-05 * max(0.0, 0.06228948 - Q.M2) * max(0.0, Q.eta_37 - 0.0) / 5.820814e-05   # +0.0%  M2 < 0.06229 and eta_37 > 0
        - 8.955891e-05 * max(0.0, 0.3112717 - Q.sj4_dr_min) * max(0.0, 0.04589821 - Q.td0_30) / 0.01139648   # -0.0%  sj4_dr_min < 0.3113 and td0_30 < 0.0459
        - 8.003636e-05 * max(0.0, 0.3886647 - Q.sj3_mass3) / 0.03562942   # -0.0%  sj3_mass3 < 0.3887
        + 6.876016e-05 * max(0.0, Q.n_electron - 1.0) / 0.08167   # +0.0%  n_electron > 1
        + 4.60402e-05 * max(0.0, -0.1170754 - Q.tdz_1) / 0.01260556   # +0.0%  tdz_1 < -0.1171
        - 3.968434e-05 * max(0.0, 0.1681173 - Q.z_displaced3) * max(0.0, Q.ismuon_2 - 0.0) / 0.0008027911   # -0.0%  z_displaced3 < 0.1681 and ismuon_2 > 0
        + 2.235098e-05 * max(0.0, Q.mass_top15 - 80.3877) * max(0.0, Q.tdz_29 - -0.09486766) / 1.402767   # +0.0%  mass_top15 > 80.39 and tdz_29 > -0.09487
        - 1.645164e-05 * max(0.0, Q.e2 - 0.04393457) * max(0.0, Q.iselectron_11 - 0.0) / 0.0002408766   # -0.0%  e2 > 0.04393 and iselectron_11 > 0
        + 1.416749e-05 * max(0.0, Q.mass_top40 - 155.8928) * max(0.0, -1.411949 - Q.lund2_lndelta) / 0.1276465   # +0.0%  mass_top40 > 155.9 and lund2_lndelta < -1.412
        - 1.396326e-05 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.iselectron_12 - 0.0) / 9.674675e-07   # -0.0%  e3_b2 < 0.0002537 and iselectron_12 > 0
        - 3.271029e-06 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, Q.ismuon_24 - 0.0) / 0.002623333   # -0.0%  n_s3d_above_3 > 6 and ismuon_24 > 0
        + 1.584226e-06 * max(0.0, Q.n_s3d_above_10 - 1.0) * max(0.0, Q.ismuon_27 - 0.0) / 0.004473333   # +0.0%  n_s3d_above_10 > 1 and ismuon_27 > 0
    )
    return z


def neuron_84(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.427524e-05
    )
    return z


def neuron_85(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.185613e-07
    )
    return z


def neuron_86(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.709903e-05
    )
    return z


def neuron_87(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.61916e-06
    )
    return z


def neuron_88(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.996807e-07
    )
    return z


def neuron_89(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.735346e-06
    )
    return z


def neuron_90(Q):
    # scale S = 5.092; each line: share * term / its average size
    z = 5.091795 * (0.0174171
        + 0.08873373 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +8.9%  lep_iso < 1.362
        - 0.06834626 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 0.02781886 - Q.lep_iso) / 0.2987542   # -6.8%  lep_ptrel < 18.77 and lep_iso < 0.02782
        - 0.06653097 * max(0.0, 0.08643515 - Q.tau3) / 0.03999044   # -6.7%  tau3 < 0.08644
        + 0.0613918 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # +6.1%  n_s3d_above_3 < 10
        + 0.04735656 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # +4.7%  sd_mass > 78.48
        + 0.04663957 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +4.7%  lep_ptrel < 43.21
        - 0.0398325 * max(0.0, 12.15228 - Q.lep_ptrel) / 8.819929   # -4.0%  lep_ptrel < 12.15
        - 0.03372983 * max(0.0, Q.n_pairs_kt_above_1 - 80.0) / 256.677   # -3.4%  n_pairs_kt_above_1 > 80
        - 0.02812051 * max(0.0, Q.sd_mass - 123.2919) / 8.015693   # -2.8%  sd_mass > 123.3
        + 0.02573749 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # +2.6%  sd_mass > 94.56
        - 0.02356487 * max(0.0, Q.z_displaced3 - 0.03624058) / 0.08737704   # -2.4%  z_displaced3 > 0.03624
        + 0.02342063 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.z_photon - 0.05998812) / 0.03655262   # +2.3%  lep_z < 0.2214 and z_photon > 0.05999
        - 0.02305518 * max(0.0, 0.1709091 - Q.z_displaced5) / 0.1064196   # -2.3%  z_displaced5 < 0.1709
        + 0.02009538 * max(0.0, Q.mass_top40 - 122.7145) / 8.484825   # +2.0%  mass_top40 > 122.7
        - 0.01922858 * max(0.0, 0.5187302 - Q.lep_z) / 0.4405854   # -1.9%  lep_z < 0.5187
        + 0.0160064 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 226.3008 - Q.sip_3d_2) / 2081.584   # +1.6%  lep_ptrel < 18.77 and sip_3d_2 < 226.3
        - 0.01568557 * max(0.0, Q.mass_top40 - 97.12186) / 20.78291   # -1.6%  mass_top40 > 97.12
        + 0.01416439 * max(0.0, Q.mass - 149.0507) / 4.967491   # +1.4%  mass > 149.1
        + 0.01381068 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 0.2184442 - Q.z_displaced5) / 2.17565   # +1.4%  lep_ptrel < 18.77 and z_displaced5 < 0.2184
        + 0.01337985 * Q.mass_displaced3 / 7.9109   # +1.3%  mass_displaced3
        - 0.01212574 * max(0.0, Q.sd_mass - 100.8525) / 16.67179   # -1.2%  sd_mass > 100.9
        - 0.011819 * max(0.0, Q.mass_top30 - 105.8151) / 11.8528   # -1.2%  mass_top30 > 105.8
        + 0.01150204 * max(0.0, Q.n_sd0_above_5 - 1.0) / 2.107647   # +1.2%  n_sd0_above_5 > 1
        - 0.01115795 * max(0.0, 2.907433 - Q.lep_iso) / 2.168962   # -1.1%  lep_iso < 2.907
        - 0.01102655 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # -1.1%  max_abs_d0 < 5.812
        + 0.01094868 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.z_charged_had - 0.265564) / 0.02462507   # +1.1%  z_displaced3 > 0.03624 and z_charged_had > 0.2656
        - 0.0109357 * max(0.0, Q.sj4_pair_mass_max - 63.34656) / 21.48011   # -1.1%  sj4_pair_mass_max > 63.35
        + 0.01056225 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # +1.1%  lep_z < 0.3397
        - 0.01044124 * max(0.0, Q.sj3_pair_mass_max - 44.2029) / 48.7862   # -1.0%  sj3_pair_mass_max > 44.2
        - 0.01016285 * max(0.0, 49.03695 - Q.sip_3d_3) / 32.30721   # -1.0%  sip_3d_3 < 49.04
        + 0.01005981 * max(0.0, 0.2801368 - Q.z_displaced5) / 0.1998239   # +1.0%  z_displaced5 < 0.2801
        - 0.009899104 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.sj3_pairmin_over_m - 0.2278414) / 0.02903772   # -1.0%  lep_z < 0.3397 and sj3_pairmin_over_m > 0.2278
        + 0.009794186 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # +1.0%  lep_z < 0.2214
        - 0.009576207 * max(0.0, Q.M2_b05 - 0.1080247) / 0.03225212   # -1.0%  M2_b05 > 0.108
        - 0.009152065 * max(0.0, 0.3261071 - Q.z_displaced3) / 0.2272573   # -0.9%  z_displaced3 < 0.3261
        + 0.008915692 * max(0.0, 0.820003 - Q.tau32) / 0.16387   # +0.9%  tau32 < 0.82
        - 0.008199026 * max(0.0, 1.362094 - Q.lep_iso) * max(0.0, 6.0 - Q.n_neutral_had) / 2.161174   # -0.8%  lep_iso < 1.362 and n_neutral_had < 6
        + 0.008102065 * max(0.0, 0.05697681 - Q.e2) / 0.003290231   # +0.8%  e2 < 0.05698
        - 0.00713328 * max(0.0, 0.06366826 - Q.z_dr_0_0p05) / 0.03452434   # -0.7%  z_dr_0_0p05 < 0.06367
        - 0.007037712 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # -0.7%  mass_top50 > 161.1
        + 0.00635732 * max(0.0, Q.pair_mean_lnm2 - 2.843705) / 0.3360502   # +0.6%  pair_mean_lnm2 > 2.844
        - 0.006240282 * max(0.0, 18.7678 - Q.lep_ptrel) / 14.28156   # -0.6%  lep_ptrel < 18.77
        + 0.006126073 * max(0.0, 0.07298691 - Q.sum_z_dr2) / 0.03643549   # +0.6%  sum_z_dr2 < 0.07299
        + 0.005314708 * max(0.0, 0.08643515 - Q.tau3) * max(0.0, Q.lne_7 - 2.313463) / 0.03612434   # +0.5%  tau3 < 0.08644 and lne_7 > 2.313
        - 0.005274087 * max(0.0, 0.02160244 - Q.M3_b2) / 0.009628213   # -0.5%  M3_b2 < 0.0216
        + 0.004947699 * max(0.0, 3.217057 - Q.sj3_mass3) / 0.6549344   # +0.5%  sj3_mass3 < 3.217
        - 0.004574312 * max(0.0, Q.mass_displaced5 - 13.97388) / 2.367218   # -0.5%  mass_displaced5 > 13.97
        - 0.004046761 * max(0.0, 6.0 - Q.n_s3d_above_3) / 2.86534   # -0.4%  n_s3d_above_3 < 6
        + 0.003818467 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # +0.4%  sd_mass > 175.9
        - 0.003604481 * max(0.0, 0.0004188921 - Q.e3) / 4.735812e-05   # -0.4%  e3 < 0.0004189
        - 0.003563646 * max(0.0, 0.02160244 - Q.M3_b2) * max(0.0, Q.sj3_dr23 - 0.3377973) / 0.001813751   # -0.4%  M3_b2 < 0.0216 and sj3_dr23 > 0.3378
        + 0.003469643 * max(0.0, 6.0 - Q.n_s3d_above_3) * max(0.0, Q.eccentricity - 0.5954257) / 0.6095166   # +0.3%  n_s3d_above_3 < 6 and eccentricity > 0.5954
        - 0.003344823 * max(0.0, Q.sj4_pair_mass_max - 48.76729) / 33.64482   # -0.3%  sj4_pair_mass_max > 48.77
        - 0.003150745 * max(0.0, 0.3191471 - Q.z_charged_had) / 0.0155244   # -0.3%  z_charged_had < 0.3191
        - 0.002957725 * max(0.0, Q.mass_2charged - 24.4079) / 2.942332   # -0.3%  mass_2charged > 24.41
        + 0.002663941 * max(0.0, 0.07298691 - Q.sum_z_dr2) * max(0.0, 0.2377815 - Q.dr_31) / 0.005211769   # +0.3%  sum_z_dr2 < 0.07299 and dr_31 < 0.2378
        - 0.002444496 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, Q.sd_rg - 0.1450854) / 2.854918   # -0.2%  lep_ptrel < 18.77 and sd_rg > 0.1451
        - 0.002345457 * max(0.0, 0.06074238 - Q.tau1) / 0.001113326   # -0.2%  tau1 < 0.06074
        - 0.002314003 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.jet_charge - 0.2072767) / 0.005215024   # -0.2%  lep_z < 0.5187 and jet_charge > 0.2073
        - 0.002149826 * max(0.0, Q.sj3_mass2 - 9.068761) / 4.236228   # -0.2%  sj3_mass2 > 9.069
        - 0.001982663 * max(0.0, Q.mass_top50 - 178.725) / 1.512159   # -0.2%  mass_top50 > 178.7
        + 0.001932885 * max(0.0, Q.n_pairs_kt_above_1 - 80.0) * max(0.0, 5.207416 - Q.sj3_mass2) / 22.17356   # +0.2%  n_pairs_kt_above_1 > 80 and sj3_mass2 < 5.207
        - 0.001926446 * max(0.0, Q.mass_top30 - 105.8151) * max(0.0, 3.157646 - Q.sip_3d_3) / 2.606556   # -0.2%  mass_top30 > 105.8 and sip_3d_3 < 3.158
        + 0.001727378 * max(0.0, Q.tau4 - 0.05417009) / 0.003306893   # +0.2%  tau4 > 0.05417
        + 0.001651742 * max(0.0, 17.0 - Q.n_photon) / 3.570917   # +0.2%  n_photon < 17
        + 0.001639613 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, -0.603253 - Q.lund3_lndelta) / 0.09792469   # +0.2%  z_displaced3 > 0.03624 and lund3_lndelta < -0.6033
        - 0.001580095 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, Q.M2 - 0.03465331) / 0.03689632   # -0.2%  lep_ptrel > 27.32 and M2 > 0.03465
        + 0.001501 * max(0.0, 0.09740996 - Q.M2) / 0.02324748   # +0.2%  M2 < 0.09741
        - 0.001328535 * max(0.0, 0.3261071 - Q.z_displaced3) * max(0.0, Q.ecf_g41 - 0.0001893721) / 1.305096e-05   # -0.1%  z_displaced3 < 0.3261 and ecf_g41 > 0.0001894
        - 0.001277586 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # -0.1%  n_dr_0p4_up < 4
        - 0.00121672 * max(0.0, 0.2665611 - Q.tau21) * max(0.0, 0.6056728 - Q.z_charged_had) / 0.003641803   # -0.1%  tau21 < 0.2666 and z_charged_had < 0.6057
        - 0.001186134 * max(0.0, 4.0 - Q.n_dr_0p4_up) * max(0.0, 0.6206221 - Q.tau32) / 0.05824379   # -0.1%  n_dr_0p4_up < 4 and tau32 < 0.6206
        - 0.001165749 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 5.207416 - Q.sj3_mass2) / 5.014232   # -0.1%  lep_ptrel < 18.77 and sj3_mass2 < 5.207
        - 0.00112946 * max(0.0, 0.3265243 - Q.z_charged) / 0.003638213   # -0.1%  z_charged < 0.3265
        + 0.001083254 * max(0.0, Q.sj3_pairmin_over_m - 0.1860959) / 0.135716   # +0.1%  sj3_pairmin_over_m > 0.1861
        + 0.00098125 * max(0.0, 0.2665611 - Q.tau21) / 0.02036837   # +0.1%  tau21 < 0.2666
        + 0.0008311396 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # +0.1%  lep_ptrel > 27.32
        + 0.0008219048 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +0.1%  sd_mass > 154.6
        - 0.0007890123 * max(0.0, 0.2665611 - Q.tau21) * max(0.0, 2.0 - Q.n_lepton) / 0.0254774   # -0.1%  tau21 < 0.2666 and n_lepton < 2
        + 0.0007581845 * max(0.0, 0.4265629 - Q.N2_b05) / 0.02389014   # +0.1%  N2_b05 < 0.4266
        + 0.0007136727 * max(0.0, 0.2801368 - Q.z_displaced5) * max(0.0, 175.9957 - Q.sip_3d_3) / 30.21447   # +0.1%  z_displaced5 < 0.2801 and sip_3d_3 < 176
        + 0.0006872403 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, 23.30856 - Q.sip_3d_3) / 16.823   # +0.1%  lep_ptrel > 27.32 and sip_3d_3 < 23.31
        - 0.0006372142 * max(0.0, 0.002975626 - Q.ecf_g31) / 0.0001549193   # -0.1%  ecf_g31 < 0.002976
        + 0.0006019947 * max(0.0, Q.sj4_pair_mass_max - 63.34656) * max(0.0, 0.0 - Q.charge_32) / 3.593516   # +0.1%  sj4_pair_mass_max > 63.35 and charge_32 < 0
        + 0.0005969685 * max(0.0, 0.07298691 - Q.sum_z_dr2) * max(0.0, -0.02072144 - Q.phi_1) / 0.0008939999   # +0.1%  sum_z_dr2 < 0.07299 and phi_1 < -0.02072
        - 0.0005632628 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, Q.z_neutral - 0.2647267) / 0.06649487   # -0.1%  lep_ptrel > 27.32 and z_neutral > 0.2647
        - 0.0004857398 * max(0.0, 5.8125 - Q.max_abs_d0) * max(0.0, 4.606241 - Q.sip_3d_3) / 5.011219   # -0.0%  max_abs_d0 < 5.812 and sip_3d_3 < 4.606
        + 0.0004761091 * max(0.0, Q.psi_0p3 - 0.9925964) / 0.0007268581   # +0.0%  psi_0p3 > 0.9926
        + 0.0004690503 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 1.0 - Q.z_top50_slots) / 0.1093734   # +0.0%  lep_ptrel < 18.77 and z_top50_slots < 1
        + 0.0004425358 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 3.018136 - Q.lne_7) / 0.007381068   # +0.0%  z_displaced3 > 0.03624 and lne_7 < 3.018
        + 0.0004188389 * max(0.0, Q.sj3_pair_mass_max - 142.9952) / 1.384637   # +0.0%  sj3_pair_mass_max > 143
        + 0.0002700252 * max(0.0, Q.mass_top40 - 97.12186) * max(0.0, Q.ismuon_2 - 0.0) / 0.3270855   # +0.0%  mass_top40 > 97.12 and ismuon_2 > 0
        - 0.0002233933 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, 0.03403084 - Q.C2_b2) / 0.01443371   # -0.0%  lep_ptrel > 27.32 and C2_b2 < 0.03403
        + 0.0001693562 * max(0.0, Q.sj3_pairmin_over_m - 0.1860959) * max(0.0, -0.02151157 - Q.td0_2) / 0.002147049   # +0.0%  sj3_pairmin_over_m > 0.1861 and td0_2 < -0.02151
        - 0.0001676095 * max(0.0, Q.n_charged_had - 21.0) / 2.12364   # -0.0%  n_charged_had > 21
        + 0.0001457158 * max(0.0, 1.362094 - Q.lep_iso) * max(0.0, 3.157646 - Q.sip_3d_3) / 0.5211504   # +0.0%  lep_iso < 1.362 and sip_3d_3 < 3.158
        - 0.00014031 * max(0.0, Q.sj3_pair_mass_max - 142.9952) * max(0.0, Q.ischhad_57 - 0.0) / 0.2570795   # -0.0%  sj3_pair_mass_max > 143 and ischhad_57 > 0
        + 9.633444e-05 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.C2_b2 - 0.221369) / 0.3704885   # +0.0%  lep_ptrel < 43.21 and C2_b2 > 0.2214
        + 7.917432e-05 * max(0.0, Q.sj3_pairmin_over_m - 0.1860959) * max(0.0, 0.1138902 - Q.z_neutral_had) / 0.002815326   # +0.0%  sj3_pairmin_over_m > 0.1861 and z_neutral_had < 0.1139
        + 1.430803e-05 * max(0.0, Q.mass_top50 - 178.725) * max(0.0, -0.1329965 - Q.tdz_13) / 0.0320315   # +0.0%  mass_top50 > 178.7 and tdz_13 < -0.133
    )
    return z


def neuron_91(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.99263e-06
    )
    return z


def neuron_92(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.724725e-06
    )
    return z


def neuron_93(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.464321e-07
    )
    return z


def neuron_94(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.322211e-06
    )
    return z


def neuron_95(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.369356e-07
    )
    return z


def neuron_96(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.277726e-07
    )
    return z


def neuron_97(Q):
    # scale S = 21.17; each line: share * term / its average size
    z = 21.17486 * (-0.02733362
        + 0.1368057 * max(0.0, 119.4443 - Q.sd_mass) / 31.47389   # +13.7%  sd_mass < 119.4
        - 0.09262104 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -9.3%  mass < 164.4
        + 0.08848995 * max(0.0, 39.09615 - Q.mass_displaced3) / 31.84697   # +8.8%  mass_displaced3 < 39.1
        - 0.08811227 * max(0.0, 154.5947 - Q.sd_mass) / 60.90029   # -8.8%  sd_mass < 154.6
        - 0.07542986 * max(0.0, 115.7091 - Q.sd_mass) / 28.95052   # -7.5%  sd_mass < 115.7
        + 0.05985217 * max(0.0, 88.81751 - Q.sd_mass) / 15.02663   # +6.0%  sd_mass < 88.82
        - 0.0505162 * max(0.0, 78.4753 - Q.sd_mass) / 11.39114   # -5.1%  sd_mass < 78.48
        + 0.04209468 * max(0.0, 18.80005 - Q.mass_displaced3) * max(0.0, 175.9957 - Q.sip_3d_3) / 2083.071   # +4.2%  mass_displaced3 < 18.8 and sip_3d_3 < 176
        - 0.04169929 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 175.9957 - Q.sip_3d_3) / 4770.711   # -4.2%  mass_displaced3 < 39.1 and sip_3d_3 < 176
        + 0.01998777 * max(0.0, Q.n_sd0_above_3 - 1.0) / 2.484937   # +2.0%  n_sd0_above_3 > 1
        + 0.01885759 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # +1.9%  e3_b2 < 0.000791
        - 0.01877558 * max(0.0, 18.80005 - Q.mass_displaced3) / 13.42744   # -1.9%  mass_displaced3 < 18.8
        - 0.01564884 * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0001737301   # -1.6%  e3_b2 < 0.0002537
        + 0.01534917 * max(0.0, 123.7928 - Q.mass) / 19.43787   # +1.5%  mass < 123.8
        + 0.01517992 * max(0.0, 120.6471 - Q.sj3_pair_mass_max) / 31.70826   # +1.5%  sj3_pair_mass_max < 120.6
        - 0.01302575 * max(0.0, Q.n_s3d_above_3 - 3.0) / 1.66923   # -1.3%  n_s3d_above_3 > 3
        + 0.01277481 * max(0.0, 161.1264 - Q.mass_top50) / 50.68933   # +1.3%  mass_top50 < 161.1
        + 0.01189341 * max(0.0, 62.03827 - Q.sd_mass) / 7.413901   # +1.2%  sd_mass < 62.04
        - 0.01166345 * max(0.0, 100.8525 - Q.sd_mass) / 20.48378   # -1.2%  sd_mass < 100.9
        - 0.01068824 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 27.3236 - Q.lep_ptrel) / 689.887   # -1.1%  mass_displaced3 < 39.1 and lep_ptrel < 27.32
        + 0.01035066 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +1.0%  mass < 95.15
        - 0.009194395 * max(0.0, 72.27436 - Q.sd_mass) / 9.692364   # -0.9%  sd_mass < 72.27
        - 0.008987063 * max(0.0, 0.1403354 - Q.z_muon) / 0.1213357   # -0.9%  z_muon < 0.1403
        + 0.007492849 * max(0.0, 126.8853 - Q.mass_top40) / 23.76162   # +0.7%  mass_top40 < 126.9
        - 0.007340436 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # -0.7%  max_abs_d0 < 5.812
        + 0.007116306 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) / 59.61773   # +0.7%  n_pairs_kt_above_3 < 126
        + 0.006926458 * max(0.0, 100.4835 - Q.mass) / 8.115061   # +0.7%  mass < 100.5
        - 0.005726553 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # -0.6%  sip_3d_2 < 447.1
        + 0.005630992 * max(0.0, 5.8125 - Q.max_abs_d0) * max(0.0, 7.345216 - Q.sip_3d_3) / 10.83619   # +0.6%  max_abs_d0 < 5.812 and sip_3d_3 < 7.345
        + 0.005027598 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) / 103.3282   # +0.5%  n_pairs_kt_above_1 < 325
        + 0.004330638 * max(0.0, Q.sj2_mass1 - 53.51926) / 6.081546   # +0.4%  sj2_mass1 > 53.52
        + 0.004129457 * max(0.0, 19.0 - Q.n_photon) / 4.85136   # +0.4%  n_photon < 19
        - 0.004073194 * max(0.0, Q.n_sd0_above_2 - 5.0) / 0.9822033   # -0.4%  n_sd0_above_2 > 5
        - 0.003919212 * max(0.0, Q.sj2_mass1 - 77.42768) / 2.022907   # -0.4%  sj2_mass1 > 77.43
        - 0.003764468 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.n_pairs_kt_above_10 - 2.0) / 152.7677   # -0.4%  mass_displaced3 < 39.1 and n_pairs_kt_above_10 > 2
        + 0.003725685 * Q.n_pairs_kt_above_10 / 6.942877   # +0.4%  n_pairs_kt_above_10
        - 0.003402351 * max(0.0, 0.8501496 - Q.sj3_pairmax_over_m) / 0.06258068   # -0.3%  sj3_pairmax_over_m < 0.8501
        - 0.003350891 * max(0.0, Q.lund_max_lndelta - -1.388411) / 0.3966091   # -0.3%  lund_max_lndelta > -1.388
        + 0.003268935 * max(0.0, Q.lund3_lndelta - -1.902701) / 0.5262124   # +0.3%  lund3_lndelta > -1.903
        - 0.002884654 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, Q.n_s3d_above_3 - 1.0) / 116.8893   # -0.3%  mass_top50 < 161.1 and n_s3d_above_3 > 1
        - 0.002769076 * max(0.0, 79.27954 - Q.mass_top50) / 2.845865   # -0.3%  mass_top50 < 79.28
        + 0.002598256 * max(0.0, 7.203613 - Q.pair_max_lnm2) / 0.7155444   # +0.3%  pair_max_lnm2 < 7.204
        + 0.002562275 * max(0.0, Q.lne_0 - 5.157617) / 0.277972   # +0.3%  lne_0 > 5.158
        + 0.00249719 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.jet_charge_k03 - -0.05990128) / 5.503613e-05   # +0.2%  e3_b2 < 0.0002537 and jet_charge_k03 > -0.0599
        + 0.002262863 * max(0.0, 109.8208 - Q.sj3_pair_mass_max) / 23.01361   # +0.2%  sj3_pair_mass_max < 109.8
        - 0.00209027 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, 175.9957 - Q.sip_3d_3) / 7530.903   # -0.2%  mass_top50 < 161.1 and sip_3d_3 < 176
        - 0.00194598 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, 0.3565533 - Q.tau21_b2) / 2.214781e-05   # -0.2%  e3_b2 < 0.0002537 and tau21_b2 < 0.3566
        - 0.001909386 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # -0.2%  n_pairs_kt_above_3 < 34
        - 0.001877949 * max(0.0, 0.6079631 - Q.max_dr) / 0.05926836   # -0.2%  max_dr < 0.608
        + 0.001830165 * max(0.0, Q.jet_charge_k05 - 0.07435708) / 0.1495371   # +0.2%  jet_charge_k05 > 0.07436
        - 0.00172334 * max(0.0, Q.z_neutral_had - 0.07573803) / 0.08719919   # -0.2%  z_neutral_had > 0.07574
        + 0.001600478 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 586.125 - Q.sum_pt_top5) / 6983.553   # +0.2%  mass_displaced3 < 39.1 and sum_pt_top5 < 586.1
        - 0.00158719 * max(0.0, 126.8853 - Q.mass_top40) * max(0.0, Q.n_muon - 0.0) / 4.915939   # -0.2%  mass_top40 < 126.9 and n_muon > 0
        + 0.001536353 * max(0.0, 126.8853 - Q.mass_top40) * max(0.0, Q.lne_5 - 2.737811) / 19.53995   # +0.2%  mass_top40 < 126.9 and lne_5 > 2.738
        - 0.00150123 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # -0.2%  sj3_pair_mass_max < 73.25
        + 0.001459285 * max(0.0, 2.68613 - Q.sj3_mass3) / 0.4824158   # +0.1%  sj3_mass3 < 2.686
        + 0.001444651 * max(0.0, Q.sj2_mass1 - 91.2852) / 1.019309   # +0.1%  sj2_mass1 > 91.29
        - 0.001407564 * max(0.0, Q.sj3_pair_mass_min - 59.49644) / 2.657693   # -0.1%  sj3_pair_mass_min > 59.5
        - 0.001379979 * max(0.0, 115.614 - Q.mass_top50) / 15.03711   # -0.1%  mass_top50 < 115.6
        - 0.001137732 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, 0.3261071 - Q.z_displaced3) / 4.035741e-05   # -0.1%  e3_b2 < 0.0002537 and z_displaced3 < 0.3261
        - 0.001135383 * max(0.0, 0.6206221 - Q.tau32) / 0.05684264   # -0.1%  tau32 < 0.6206
        - 0.001069331 * max(0.0, Q.z_displaced5 - 0.1339658) / 0.03671738   # -0.1%  z_displaced5 > 0.134
        - 0.000917915 * max(0.0, 9.621843e-05 - Q.e3_b2) / 4.881439e-05   # -0.1%  e3_b2 < 9.622e-05
        + 0.0008459782 * max(0.0, 0.6206221 - Q.tau32) * max(0.0, 0.6056728 - Q.z_charged_had) / 0.008850292   # +0.1%  tau32 < 0.6206 and z_charged_had < 0.6057
        + 0.0008398464 * max(0.0, Q.lund_max_lndelta - -0.6897565) / 0.03634831   # +0.1%  lund_max_lndelta > -0.6898
        - 0.000792985 * max(0.0, Q.e2_b05 - 0.1510354) / 0.01899022   # -0.1%  e2_b05 > 0.151
        - 0.0006509929 * max(0.0, Q.lund_max_lndelta - -1.388411) * max(0.0, Q.lnerel_4 - -3.310904) / 0.1430312   # -0.1%  lund_max_lndelta > -1.388 and lnerel_4 > -3.311
        + 0.0006264647 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.n_dr_0p4_up - 8.0) / 9.895202e-05   # +0.1%  e3_b2 < 0.0002537 and n_dr_0p4_up > 8
        - 0.0005714337 * max(0.0, Q.psi_0p1 - 0.7386202) / 0.03082667   # -0.1%  psi_0p1 > 0.7386
        - 0.0005561665 * max(0.0, 3.521178 - Q.lund_max_lnkt) / 0.1676122   # -0.1%  lund_max_lnkt < 3.521
        + 0.0005300153 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.mass_charged - 84.24838) / 0.0002363463   # +0.1%  e3_b2 < 0.0002537 and mass_charged > 84.25
        + 0.0005165831 * max(0.0, 18.80005 - Q.mass_displaced3) * max(0.0, Q.mass_2photon - 9.82024) / 23.24627   # +0.1%  mass_displaced3 < 18.8 and mass_2photon > 9.82
        + 0.0004961708 * max(0.0, Q.D2 - 3.597891) / 0.2539167   # +0.0%  D2 > 3.598
        + 0.0004808937 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, 2.286291 - Q.D2_b2) / 2.924988   # +0.0%  n_pairs_kt_above_3 < 34 and D2_b2 < 2.286
        + 0.0004579707 * max(0.0, Q.e2 - 0.09764648) / 0.007060655   # +0.0%  e2 > 0.09765
        - 0.0002705045 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, -0.6526285 - Q.lead_ch_sdz) / 232.1779   # -0.0%  sip_3d_2 < 447.1 and lead_ch_sdz < -0.6526
        - 0.0002408621 * max(0.0, Q.pair_mean_lnm2 - 2.09826) / 0.7077607   # -0.0%  pair_mean_lnm2 > 2.098
        - 0.0002266808 * max(0.0, Q.sj3_dr_min - 0.3190414) / 0.01982359   # -0.0%  sj3_dr_min > 0.319
        + 0.0002173113 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, Q.td0_33 - -0.04023096) / 15.38564   # +0.0%  sip_3d_2 < 447.1 and td0_33 > -0.04023
        + 0.0002057467 * max(0.0, 115.614 - Q.mass_top50) * max(0.0, 1.0 - Q.n_lepton) / 9.314921   # +0.0%  mass_top50 < 115.6 and n_lepton < 1
        + 0.0001509647 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.iselectron_0 - 0.0) / 2.911082   # +0.0%  mass_displaced3 < 39.1 and iselectron_0 > 0
        - 0.0001298891 * max(0.0, 100.4835 - Q.mass) * max(0.0, Q.iselectron_0 - 0.0) / 0.7641534   # -0.0%  mass < 100.5 and iselectron_0 > 0
        - 0.0001213359 * max(0.0, 0.01015545 - Q.M2_b2) / 0.0003136992   # -0.0%  M2_b2 < 0.01016
        - 0.0001172137 * max(0.0, Q.n_pt_above_5 - 27.0) / 1.148523   # -0.0%  n_pt_above_5 > 27
        - 0.0001108775 * max(0.0, 0.8501496 - Q.sj3_pairmax_over_m) * max(0.0, Q.dr_53 - 0.0) / 0.004664772   # -0.0%  sj3_pairmax_over_m < 0.8501 and dr_53 > 0
        - 8.551547e-05 * max(0.0, Q.n_sd0_above_3 - 1.0) * max(0.0, Q.sj3_dr13 - 0.6229991) / 0.06211718   # -0.0%  n_sd0_above_3 > 1 and sj3_dr13 > 0.623
        + 6.962266e-05 * max(0.0, 0.1244374 - Q.N2_b2) * max(0.0, -0.1424561 - Q.eta_29) / 0.0002260608   # +0.0%  N2_b2 < 0.1244 and eta_29 < -0.1425
        + 6.711561e-05 * max(0.0, Q.lund_max_lndelta - -1.388411) * max(0.0, 0.09541201 - Q.td0_7) / 0.04689523   # +0.0%  lund_max_lndelta > -1.388 and td0_7 < 0.09541
        + 6.704585e-05 * max(0.0, 0.1244374 - Q.N2_b2) / 0.01729494   # +0.0%  N2_b2 < 0.1244
        + 4.535992e-05 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, Q.min_pair_mass - 3.369962) / 40.0642   # +0.0%  mass_top50 < 161.1 and min_pair_mass > 3.37
        + 4.345612e-05 * max(0.0, 0.0007909605 - Q.e3_b2) * max(0.0, Q.dr_28 - 0.3368505) / 1.879537e-05   # +0.0%  e3_b2 < 0.000791 and dr_28 > 0.3369
        - 4.03799e-05 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.sj3_dr13 - 0.7462286) / 9.554771e-07   # -0.0%  e3_b2 < 0.0002537 and sj3_dr13 > 0.7462
        - 2.509192e-05 * max(0.0, Q.z_neutral_had - 0.07573803) * max(0.0, Q.td0_23 - 0.0) / 0.002247624   # -0.0%  z_neutral_had > 0.07574 and td0_23 > 0
        - 2.10967e-05 * max(0.0, Q.sj2_mass1 - 77.42768) * max(0.0, 0.8678264 - Q.D2_b2) / 0.003578708   # -0.0%  sj2_mass1 > 77.43 and D2_b2 < 0.8678
        + 1.319831e-05 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, Q.d0err_10 - 0.02780151) / 0.0002782402   # +0.0%  n_s3d_above_3 > 3 and d0err_10 > 0.0278
        - 9.473688e-06 * max(0.0, Q.z_displaced5 - 0.1339658) * max(0.0, Q.C3_b2 - 0.04229114) / 9.919964e-05   # -0.0%  z_displaced5 > 0.134 and C3_b2 > 0.04229
        - 2.732897e-06 * max(0.0, 109.8208 - Q.sj3_pair_mass_max) * max(0.0, Q.lnpt_44 - -18.42068) / 82.7679   # -0.0%  sj3_pair_mass_max < 109.8 and lnpt_44 > -18.42
        + 6.546412e-07 * max(0.0, 0.8501496 - Q.sj3_pairmax_over_m) * max(0.0, Q.ismuon_37 - 0.0) / 5.35795e-05   # +0.0%  sj3_pairmax_over_m < 0.8501 and ismuon_37 > 0
    )
    return z


def neuron_98(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.990151e-07
    )
    return z


def neuron_99(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.001934279
    )
    return z


def neuron_100(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.26407e-06
    )
    return z


def neuron_101(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.878267e-06
    )
    return z


def neuron_102(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.490487e-06
    )
    return z


def neuron_103(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.414348e-06
    )
    return z


def neuron_104(Q):
    # scale S = 7.016; each line: share * term / its average size
    z = 7.016036 * (-0.2566057
        + 0.1155961 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # +11.6%  sd_mass > 78.48
        - 0.1052643 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # -10.5%  sd_mass > 88.82
        + 0.0730673 * max(0.0, Q.mass_displaced3 - 3.208089) / 6.234439   # +7.3%  mass_displaced3 > 3.208
        + 0.05125484 * max(0.0, -1.352792 - Q.pair_mean_lndelta) / 0.8695512   # +5.1%  pair_mean_lndelta < -1.353
        + 0.04341153 * max(0.0, 155.8928 - Q.mass_top40) / 48.12072   # +4.3%  mass_top40 < 155.9
        + 0.03982818 * max(0.0, 120.653 - Q.mass) / 17.47277   # +4.0%  mass < 120.7
        - 0.03511766 * max(0.0, Q.mass_displaced3 - 6.341631) / 5.164503   # -3.5%  mass_displaced3 > 6.342
        + 0.03278979 * max(0.0, 67.0 - Q.n_particles) / 28.06102   # +3.3%  n_particles < 67
        + 0.03154709 * max(0.0, 149.0507 - Q.mass) / 39.08704   # +3.2%  mass < 149.1
        + 0.02811757 * max(0.0, 8.0 - Q.n_s3d_above_10) / 5.470973   # +2.8%  n_s3d_above_10 < 8
        + 0.02534358 * max(0.0, 0.5562194 - Q.sd_rg) / 0.2284833   # +2.5%  sd_rg < 0.5562
        - 0.02248147 * max(0.0, Q.N2_b05 - 0.3667049) / 0.07978604   # -2.2%  N2_b05 > 0.3667
        + 0.02019135 * max(0.0, 6.0 - Q.n_sd0_above_3) / 3.16582   # +2.0%  n_sd0_above_3 < 6
        + 0.01977144 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, 49.0 - Q.n_real_top50) / 74.79127   # +2.0%  n_s3d_above_10 < 8 and n_real_top50 < 49
        - 0.0185868 * Q.n_charged_pt_above_1 / 18.83947   # -1.9%  n_charged_pt_above_1
        + 0.01700405 * max(0.0, Q.z_top15_slots - 0.9212656) / 0.01314087   # +1.7%  z_top15_slots > 0.9213
        - 0.01621249 * max(0.0, 4.0 - Q.sd_nremoved) / 2.43384   # -1.6%  sd_nremoved < 4
        - 0.01420097 * max(0.0, 146.0 - Q.n_pairs_kt_above_1) / 20.10573   # -1.4%  n_pairs_kt_above_1 < 146
        + 0.01258195 * Q.mass_neutral / 45.13363   # +1.3%  mass_neutral
        + 0.01155986 * max(0.0, 67.0 - Q.n_particles) * max(0.0, 2.0 - Q.n_lepton) / 39.27554   # +1.2%  n_particles < 67 and n_lepton < 2
        + 0.01111824 * max(0.0, 0.191059 - Q.C2) / 0.05400117   # +1.1%  C2 < 0.1911
        + 0.01093601 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # +1.1%  sd_mass > 106.8
        + 0.01014998 * max(0.0, 0.02949822 - Q.tau4) / 0.003969373   # +1.0%  tau4 < 0.0295
        + 0.009980878 * max(0.0, 132.5189 - Q.mass_top40) / 28.13525   # +1.0%  mass_top40 < 132.5
        + 0.009932897 * max(0.0, 10.0 - Q.n_dr_0p4_up) / 5.21002   # +1.0%  n_dr_0p4_up < 10
        - 0.009512833 * max(0.0, 6.185635 - Q.lep_iso) / 4.883283   # -1.0%  lep_iso < 6.186
        - 0.009342895 * max(0.0, 7.0 - Q.n_sd0_above_2) / 3.29892   # -0.9%  n_sd0_above_2 < 7
        + 0.009020705 * max(0.0, 149.0507 - Q.mass) * max(0.0, Q.lne_0 - 5.157617) / 13.38687   # +0.9%  mass < 149.1 and lne_0 > 5.158
        + 0.008640972 * max(0.0, Q.sd_rg - 0.3025746) / 0.08639404   # +0.9%  sd_rg > 0.3026
        - 0.008528853 * max(0.0, 0.4680886 - Q.tau32_b2) / 0.05978538   # -0.9%  tau32_b2 < 0.4681
        - 0.008106076 * max(0.0, 4.0 - Q.n_neutral_had) / 0.8492867   # -0.8%  n_neutral_had < 4
        - 0.008089382 * max(0.0, 4.046329 - Q.lund_max_lnkt) / 0.370868   # -0.8%  lund_max_lnkt < 4.046
        + 0.007571209 * max(0.0, 0.2326317 - Q.C2_b05) / 0.01179395   # +0.8%  C2_b05 < 0.2326
        + 0.007314199 * max(0.0, Q.pair_max_lnm2 - 6.052324) / 0.7112379   # +0.7%  pair_max_lnm2 > 6.052
        - 0.007046736 * max(0.0, Q.mass_top5 - 17.63354) / 26.12147   # -0.7%  mass_top5 > 17.63
        - 0.00646951 * max(0.0, Q.e3 - 0.001589861) / 0.0004282381   # -0.6%  e3 > 0.00159
        - 0.005848774 * max(0.0, 5.0 - Q.n_s3d_above_3) * max(0.0, 0.1404188 - Q.C2_b2) / 0.1613146   # -0.6%  n_s3d_above_3 < 5 and C2_b2 < 0.1404
        - 0.00572004 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.221436 - Q.lep_z) / 1.248772   # -0.6%  n_pairs_kt_above_3 < 47 and lep_z < 0.2214
        + 0.005582439 * max(0.0, 0.3190414 - Q.sj3_dr_min) / 0.1328834   # +0.6%  sj3_dr_min < 0.319
        + 0.005500253 * max(0.0, -1.352792 - Q.pair_mean_lndelta) * max(0.0, Q.lund1_lndelta - -0.6177752) / 0.1862499   # +0.6%  pair_mean_lndelta < -1.353 and lund1_lndelta > -0.6178
        - 0.005285267 * max(0.0, Q.mass_displaced3 - 26.78691) / 1.554829   # -0.5%  mass_displaced3 > 26.79
        + 0.005215319 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +0.5%  sd_mass > 154.6
        + 0.005005014 * max(0.0, Q.jet_abs_eta - 0.6771968) / 0.2378129   # +0.5%  jet_abs_eta > 0.6772
        + 0.004975048 * max(0.0, Q.mass_over_sum_pt - 0.215596) / 0.01424531   # +0.5%  mass_over_sum_pt > 0.2156
        + 0.004760167 * max(0.0, Q.mass_displaced3 - 3.208089) * max(0.0, 2.885768e-05 - Q.ecf_g43) / 0.000134714   # +0.5%  mass_displaced3 > 3.208 and ecf_g43 < 2.886e-05
        + 0.004386909 * max(0.0, 99.20396 - Q.mass_charged) / 37.01153   # +0.4%  mass_charged < 99.2
        - 0.00434742 * max(0.0, Q.n_lund - 11.0) / 1.107333   # -0.4%  n_lund > 11
        + 0.004339698 * max(0.0, Q.tau21 - 0.5797033) / 0.03370208   # +0.4%  tau21 > 0.5797
        + 0.003857915 * max(0.0, 6.0 - Q.n_sd0_above_3) * max(0.0, Q.n_dr_0p4_up - 0.0) / 16.68924   # +0.4%  n_sd0_above_3 < 6 and n_dr_0p4_up > 0
        - 0.00373502 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.4%  lep_ptrel > 43.21
        + 0.00334727 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # +0.3%  lep_ptrel > 27.32
        + 0.003264774 * max(0.0, Q.N2 - 0.3245983) / 0.02636523   # +0.3%  N2 > 0.3246
        - 0.003239197 * max(0.0, 5.0 - Q.n_s3d_above_3) / 2.12335   # -0.3%  n_s3d_above_3 < 5
        + 0.003235399 * max(0.0, Q.D2 - 3.597891) / 0.2539167   # +0.3%  D2 > 3.598
        + 0.00315502 * max(0.0, Q.n_lund_kt_above_5 - 2.0) / 0.4166533   # +0.3%  n_lund_kt_above_5 > 2
        - 0.002836447 * max(0.0, -1.352792 - Q.pair_mean_lndelta) * max(0.0, Q.lep_ptrel - 12.15228) / 1.748073   # -0.3%  pair_mean_lndelta < -1.353 and lep_ptrel > 12.15
        + 0.002733232 * max(0.0, Q.lep_ptrel - 18.7678) / 2.829169   # +0.3%  lep_ptrel > 18.77
        + 0.002574165 * max(0.0, Q.M3 - 0.05919662) / 0.001232267   # +0.3%  M3 > 0.0592
        + 0.002566236 * max(0.0, 0.3886647 - Q.sj3_mass3) / 0.03562942   # +0.3%  sj3_mass3 < 0.3887
        + 0.002412843 * max(0.0, 3.0 - Q.n_sdz_above_5) / 1.349857   # +0.2%  n_sdz_above_5 < 3
        - 0.00210721 * max(0.0, 67.0 - Q.n_particles) * max(0.0, Q.dr01 - 0.06433539) / 3.156584   # -0.2%  n_particles < 67 and dr01 > 0.06434
        - 0.002001649 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -0.2%  sip_3d_2 < 226.3
        - 0.001931925 * max(0.0, 0.2377332 - Q.tau21) / 0.01385845   # -0.2%  tau21 < 0.2377
        - 0.001733949 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.dr02 - 0.2159556) / 0.5243339   # -0.2%  n_pairs_kt_above_3 < 47 and dr02 > 0.216
        - 0.001607549 * max(0.0, Q.sj2_mass1 - 53.51926) / 6.081546   # -0.2%  sj2_mass1 > 53.52
        - 0.001598431 * max(0.0, 146.0 - Q.n_pairs_kt_above_1) * max(0.0, 4.878859 - Q.D2) / 53.98141   # -0.2%  n_pairs_kt_above_1 < 146 and D2 < 4.879
        - 0.001443818 * max(0.0, -0.5570337 - Q.jet_charge_k03) / 0.07719303   # -0.1%  jet_charge_k03 < -0.557
        - 0.001343417 * max(0.0, Q.mass_displaced3 - 3.208089) * max(0.0, 0.02148541 - Q.lam2) / 0.09278072   # -0.1%  mass_displaced3 > 3.208 and lam2 < 0.02149
        + 0.001342673 * max(0.0, 0.135772 - Q.tau21) / 0.001647604   # +0.1%  tau21 < 0.1358
        - 0.001086451 * max(0.0, 0.01782783 - Q.z_displaced3) / 0.005025534   # -0.1%  z_displaced3 < 0.01783
        + 0.001046219 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, Q.lep_ptrel - 18.7678) / 13.83744   # +0.1%  n_s3d_above_10 < 8 and lep_ptrel > 18.77
        + 0.0009458121 * max(0.0, 146.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 7.535107   # +0.1%  n_pairs_kt_above_1 < 146 and n_lund_kt_above_5 > 1
        + 0.0008760443 * max(0.0, 0.4680886 - Q.tau32_b2) * max(0.0, 0.003965728 - Q.ecf_g32) / 7.486955e-05   # +0.1%  tau32_b2 < 0.4681 and ecf_g32 < 0.003966
        - 0.0008436977 * max(0.0, 0.07264571 - Q.tau2) / 0.01634431   # -0.1%  tau2 < 0.07265
        - 0.0007266057 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # -0.1%  sd_mass > 175.9
        + 0.000717413 * max(0.0, 6.216954e-05 - Q.ecf_g41) / 1.118908e-06   # +0.1%  ecf_g41 < 6.217e-05
        - 0.0006742376 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.2601259 - Q.z_displaced3) / 1.851029   # -0.1%  n_pairs_kt_above_3 < 47 and z_displaced3 < 0.2601
        + 0.0006656542 * max(0.0, Q.z_top15_slots - 0.9212656) * max(0.0, 0.03430176 - Q.dzerr_1) / 0.0001974266   # +0.1%  z_top15_slots > 0.9213 and dzerr_1 < 0.0343
        - 0.0006032676 * max(0.0, 3.0 - Q.n_pairs_kt_above_10) / 0.88044   # -0.1%  n_pairs_kt_above_10 < 3
        + 0.000567815 * max(0.0, Q.lund_max_lndelta - -0.529318) / 0.01270035   # +0.1%  lund_max_lndelta > -0.5293
        - 0.0004733101 * max(0.0, Q.n_for_90pct - 39.0) / 0.30656   # -0.0%  n_for_90pct > 39
        + 0.0003286801 * max(0.0, 4.0 - Q.sd_nremoved) * max(0.0, Q.eta_4 - 0.05764771) / 0.06679296   # +0.0%  sd_nremoved < 4 and eta_4 > 0.05765
        + 0.0002461338 * max(0.0, 0.4680886 - Q.tau32_b2) * max(0.0, Q.eta_1 - 0.01268768) / 0.002230802   # +0.0%  tau32_b2 < 0.4681 and eta_1 > 0.01269
        + 0.0002460812 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, Q.sj4_pair_mass_max - 101.849) / 18.13026   # +0.0%  n_s3d_above_10 < 8 and sj4_pair_mass_max > 101.8
        - 0.0002393869 * max(0.0, Q.sd_mass - 175.9333) * max(0.0, Q.lnptrel_77 - -18.42068) / 3.917776   # -0.0%  sd_mass > 175.9 and lnptrel_77 > -18.42
        - 0.000220492 * max(0.0, 0.04483276 - Q.tau2) / 0.004264343   # -0.0%  tau2 < 0.04483
        + 0.0001357211 * max(0.0, 4.0 - Q.sd_nremoved) * max(0.0, Q.eta_28 - -0.3166626) / 0.7833793   # +0.0%  sd_nremoved < 4 and eta_28 > -0.3167
        - 0.0001171733 * max(0.0, Q.sum_charge - 4.0) / 0.03763333   # -0.0%  sum_charge > 4
        + 8.944245e-05 * max(0.0, 0.0318986 - Q.e2) / 0.0005225451   # +0.0%  e2 < 0.0319
        + 8.700081e-05 * max(0.0, Q.C2_b2 - 0.221369) / 0.0106764   # +0.0%  C2_b2 > 0.2214
        + 7.403631e-05 * max(0.0, Q.sd_mass - 78.4753) * max(0.0, 1.0 - Q.charge_25) / 29.99101   # +0.0%  sd_mass > 78.48 and charge_25 < 1
        + 6.292138e-05 * max(0.0, -0.02912079 - Q.td0_38) / 0.01076472   # +0.0%  td0_38 < -0.02912
        + 4.667023e-05 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) / 9.886423   # +0.0%  n_pairs_kt_above_3 < 47
        + 4.44555e-05 * max(0.0, 3.0 - Q.n_pairs_kt_above_10) * max(0.0, 0.0328064 - Q.phi_29) / 0.06656675   # +0.0%  n_pairs_kt_above_10 < 3 and phi_29 < 0.03281
        + 3.829533e-05 * max(0.0, Q.n_lund - 11.0) * max(0.0, 0.1278076 - Q.phi_21) / 0.176051   # +0.0%  n_lund > 11 and phi_21 < 0.1278
        + 2.496153e-05 * max(0.0, Q.sd_mass - 175.9333) * max(0.0, 0.0 - Q.charge_6) / 0.5226794   # +0.0%  sd_mass > 175.9 and charge_6 < 0
        + 2.183183e-05 * max(0.0, Q.mass_over_sum_pt - 0.215596) * max(0.0, 0.1211197 - Q.dr_9) / 8.201701e-05   # +0.0%  mass_over_sum_pt > 0.2156 and dr_9 < 0.1211
        + 1.308458e-05 * max(0.0, 6.185635 - Q.lep_iso) * max(0.0, Q.iselectron_27 - 0.0) / 0.004863385   # +0.0%  lep_iso < 6.186 and iselectron_27 > 0
        + 1.292724e-05 * max(0.0, Q.sd_mass - 78.4753) * max(0.0, 0.0 - Q.tdz_74) / 0.05707131   # +0.0%  sd_mass > 78.48 and tdz_74 < 0
    )
    return z


def neuron_105(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.732385e-05
    )
    return z


def neuron_106(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.160924e-05
    )
    return z


def neuron_107(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.15856e-05
    )
    return z


def neuron_108(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.470689e-06
    )
    return z


def neuron_109(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.200097e-07
    )
    return z


def neuron_110(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.579629e-06
    )
    return z


def neuron_111(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.669613e-06
    )
    return z


def neuron_112(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.237812e-06
    )
    return z


def neuron_113(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.665935e-05
    )
    return z


def neuron_114(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.677423e-06
    )
    return z


def neuron_115(Q):
    # scale S = 13.97; each line: share * term / its average size
    z = 13.96853 * (0.04043916
        + 0.07554878 * max(0.0, 182.8592 - Q.mass) / 69.61635   # +7.6%  mass < 182.9
        - 0.06316752 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -6.3%  mass < 164.4
        + 0.06296864 * max(0.0, 129.5874 - Q.mass_top50) / 24.20267   # +6.3%  mass_top50 < 129.6
        + 0.05542621 * max(0.0, 39.09615 - Q.mass_displaced3) / 31.84697   # +5.5%  mass_displaced3 < 39.1
        - 0.04433517 * max(0.0, 125.4927 - Q.mass_top50) / 21.23144   # -4.4%  mass_top50 < 125.5
        - 0.04366067 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 0.3788785 - Q.z_neutral_had) / 7.382889   # -4.4%  mass_displaced3 < 39.1 and z_neutral_had < 0.3789
        - 0.04362674 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) / 44.19386   # -4.4%  sj3_pair_mass_min < 80.03
        + 0.03779785 * max(0.0, 0.0131933 - Q.e3_b05) / 0.006147996   # +3.8%  e3_b05 < 0.01319
        - 0.03661106 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.z_top20_slots - 0.7122459) / 11.06965   # -3.7%  mass < 164.4 and z_top20_slots > 0.7122
        + 0.03590128 * max(0.0, 120.653 - Q.mass) / 17.47277   # +3.6%  mass < 120.7
        - 0.03589572 * max(0.0, Q.pair_mean_lndelta - -3.421602) / 1.23252   # -3.6%  pair_mean_lndelta > -3.422
        - 0.0350257 * max(0.0, 107.8099 - Q.sj4_pair_mass_max) / 29.80438   # -3.5%  sj4_pair_mass_max < 107.8
        - 0.03244446 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 0.0007909605 - Q.e3_b2) / 0.02145527   # -3.2%  mass_displaced3 < 39.1 and e3_b2 < 0.000791
        + 0.03169689 * max(0.0, 0.120439 - Q.M2) / 0.04313262   # +3.2%  M2 < 0.1204
        - 0.02111129 * max(0.0, 0.4888886 - Q.LHA) / 0.09435016   # -2.1%  LHA < 0.4889
        - 0.02002265 * max(0.0, 0.05919662 - Q.M3) / 0.02436955   # -2.0%  M3 < 0.0592
        - 0.01873457 * max(0.0, 109.8208 - Q.sj3_pair_mass_max) / 23.01361   # -1.9%  sj3_pair_mass_max < 109.8
        + 0.01762961 * max(0.0, Q.tau1 - 0.1246227) / 0.07463455   # +1.8%  tau1 > 0.1246
        - 0.01728811 * max(0.0, 28.89659 - Q.mass_2charged) / 18.56832   # -1.7%  mass_2charged < 28.9
        + 0.01542125 * max(0.0, 6.767937 - Q.mass_2charged) / 3.01163   # +1.5%  mass_2charged < 6.768
        + 0.01501193 * max(0.0, 36.36236 - Q.sj3_mass1) / 17.59922   # +1.5%  sj3_mass1 < 36.36
        + 0.01433117 * max(0.0, 0.05966366 - Q.M2_b2) / 0.0281728   # +1.4%  M2_b2 < 0.05966
        - 0.01327215 * max(0.0, 0.3190414 - Q.sj3_dr_min) / 0.1328834   # -1.3%  sj3_dr_min < 0.319
        + 0.01181407 * max(0.0, 0.2099278 - Q.N2_b2) / 0.06000895   # +1.2%  N2_b2 < 0.2099
        + 0.00895305 * max(0.0, 0.1845735 - Q.sj4_dr_min) / 0.07908106   # +0.9%  sj4_dr_min < 0.1846
        + 0.008485527 * max(0.0, 0.09733903 - Q.tau2) * max(0.0, 27.3236 - Q.lep_ptrel) / 0.6620054   # +0.8%  tau2 < 0.09734 and lep_ptrel < 27.32
        - 0.007157501 * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 1.107657   # -0.7%  n_lund_kt_above_5 > 1
        + 0.007077816 * max(0.0, 72.862 - Q.sj4_pair_mass_max) / 6.86631   # +0.7%  sj4_pair_mass_max < 72.86
        - 0.006872967 * max(0.0, 94.51361 - Q.mass_top50) / 6.251054   # -0.7%  mass_top50 < 94.51
        + 0.006604046 * max(0.0, 2.911067 - Q.pair_max_lnkt) / 0.3583041   # +0.7%  pair_max_lnkt < 2.911
        - 0.006399723 * max(0.0, 0.05966366 - Q.M2_b2) * max(0.0, Q.mass_charged - 37.42054) / 0.741033   # -0.6%  M2_b2 < 0.05966 and mass_charged > 37.42
        + 0.006345651 * max(0.0, Q.n_s3d_above_3 - 4.0) / 1.224997   # +0.6%  n_s3d_above_3 > 4
        + 0.005900498 * max(0.0, 0.09733903 - Q.tau2) * max(0.0, 1.0 - Q.isphoton_41) / 0.02865177   # +0.6%  tau2 < 0.09734 and isphoton_41 < 1
        + 0.005847174 * max(0.0, 125.4927 - Q.mass_top50) * max(0.0, 1.0 - Q.n_lepton) / 12.84432   # +0.6%  mass_top50 < 125.5 and n_lepton < 1
        - 0.005740179 * max(0.0, Q.z_top20_slots - 0.9417195) / 0.01519974   # -0.6%  z_top20_slots > 0.9417
        - 0.005551622 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 0.8709334 - Q.tau43) / 2.811381   # -0.6%  mass_displaced3 < 39.1 and tau43 < 0.8709
        + 0.005139897 * max(0.0, Q.tau32_b2 - 0.6908801) / 0.04407883   # +0.5%  tau32_b2 > 0.6909
        + 0.004683995 * max(0.0, 0.2099278 - Q.N2_b2) * max(0.0, 0.05511475 - Q.dzerr_43) / 0.003066145   # +0.5%  N2_b2 < 0.2099 and dzerr_43 < 0.05511
        + 0.004397957 * max(0.0, 109.3251 - Q.mass_top30) * max(0.0, Q.lnptrel_12 - -5.283873) / 15.29061   # +0.4%  mass_top30 < 109.3 and lnptrel_12 > -5.284
        + 0.00438677 * max(0.0, 65.0404 - Q.mass_charged) / 11.01768   # +0.4%  mass_charged < 65.04
        + 0.004288433 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 175.9957 - Q.sip_3d_3) / 116.3453   # +0.4%  n_s3d_above_3 > 4 and sip_3d_3 < 176
        - 0.004130928 * max(0.0, 2.0 - Q.n_sdz_above_5) / 0.7314567   # -0.4%  n_sdz_above_5 < 2
        - 0.004063053 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # -0.4%  lep_iso < 0.4382
        - 0.003927211 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # -0.4%  sip_3d_2 < 447.1
        - 0.003811695 * max(0.0, Q.lep_z - 0.03021637) * max(0.0, 2.0 - Q.n_lepton) / 0.0540952   # -0.4%  lep_z > 0.03022 and n_lepton < 2
        + 0.003736333 * max(0.0, 303.75 - Q.sum_pt_top5) / 21.29329   # +0.4%  sum_pt_top5 < 303.8
        + 0.003736198 * max(0.0, 0.1991803 - Q.sj3_dr_min) / 0.05079997   # +0.4%  sj3_dr_min < 0.1992
        - 0.003656896 * max(0.0, Q.m012 - 11.86162) / 16.74842   # -0.4%  m012 > 11.86
        - 0.00357243 * max(0.0, 12.0 - Q.n_dr_0_0p05) / 8.882307   # -0.4%  n_dr_0_0p05 < 12
        - 0.00355003 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, Q.z_charged_had - 0.3603262) / 0.2342076   # -0.4%  n_s3d_above_3 > 4 and z_charged_had > 0.3603
        + 0.003536567 * max(0.0, 59.49644 - Q.sj3_pair_mass_min) / 25.61978   # +0.4%  sj3_pair_mass_min < 59.5
        + 0.003504344 * max(0.0, 37.19471 - Q.sj3_pair_mass_min) * max(0.0, 0.3118983 - Q.sj3_z2) / 0.8244468   # +0.4%  sj3_pair_mass_min < 37.19 and sj3_z2 < 0.3119
        - 0.003352691 * max(0.0, Q.tau4 - 0.04996) / 0.004242161   # -0.3%  tau4 > 0.04996
        + 0.003331437 * max(0.0, 5.763861 - Q.mass_2photon) / 2.954304   # +0.3%  mass_2photon < 5.764
        + 0.003308933 * max(0.0, Q.lep_z - 0.03021637) / 0.07491017   # +0.3%  lep_z > 0.03022
        + 0.00326178 * max(0.0, 0.8207647 - Q.sj3_pairmax_over_m) / 0.04486605   # +0.3%  sj3_pairmax_over_m < 0.8208
        - 0.003072044 * max(0.0, 1.398635 - Q.mass_2charged) / 0.2801595   # -0.3%  mass_2charged < 1.399
        + 0.003049799 * max(0.0, Q.mass_top5 - 37.66803) / 12.83747   # +0.3%  mass_top5 > 37.67
        + 0.002780033 * max(0.0, 3.692778 - Q.lne_4) / 0.1626312   # +0.3%  lne_4 < 3.693
        - 0.002395209 * max(0.0, Q.n_sd0_above_3 - 3.0) / 1.31993   # -0.2%  n_sd0_above_3 > 3
        + 0.002309876 * max(0.0, 109.3251 - Q.mass_top30) / 15.25064   # +0.2%  mass_top30 < 109.3
        - 0.002287921 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.2%  lep_ptrel > 43.21
        + 0.002239324 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 0.01013989 - Q.M3_b2) / 0.05136574   # +0.2%  mass_displaced3 < 39.1 and M3_b2 < 0.01014
        - 0.002224332 * max(0.0, 2.852309e-06 - Q.ecf_g43) / 7.166198e-07   # -0.2%  ecf_g43 < 2.852e-06
        - 0.00208922 * max(0.0, 36.36236 - Q.sj3_mass1) * max(0.0, Q.sj3_mass2 - 9.068761) / 63.18552   # -0.2%  sj3_mass1 < 36.36 and sj3_mass2 > 9.069
        - 0.002037278 * max(0.0, 0.0006496195 - Q.e3) / 0.000123029   # -0.2%  e3 < 0.0006496
        + 0.002006643 * max(0.0, 37.19471 - Q.sj3_pair_mass_min) / 9.096396   # +0.2%  sj3_pair_mass_min < 37.19
        + 0.001821315 * max(0.0, 0.09733903 - Q.tau2) / 0.03188557   # +0.2%  tau2 < 0.09734
        - 0.001627258 * max(0.0, 0.07985021 - Q.sj4_dr_min) / 0.01211577   # -0.2%  sj4_dr_min < 0.07985
        - 0.00146295 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, 0.8953628 - Q.tau54) / 0.07410258   # -0.1%  n_lund_kt_above_5 > 1 and tau54 < 0.8954
        + 0.001206785 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, 0.06029776 - Q.sj4_zsoft) / 0.01631242   # +0.1%  n_lund_kt_above_5 > 1 and sj4_zsoft < 0.0603
        + 0.001046871 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, Q.z_displaced5 - 0.1046203) / 0.03484583   # +0.1%  n_lund_kt_above_5 > 1 and z_displaced5 > 0.1046
        - 0.001041127 * max(0.0, 0.1117947 - Q.dr_0) / 0.02716844   # -0.1%  dr_0 < 0.1118
        + 0.0009115402 * max(0.0, 0.0225905 - Q.D3_b2) / 0.006016147   # +0.1%  D3_b2 < 0.02259
        - 0.000728966 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.isphoton_12 - 0.0) / 9.753263   # -0.1%  mass_displaced3 < 39.1 and isphoton_12 > 0
        - 0.0007252308 * max(0.0, 101.0 - Q.n_pairs_kt_above_1) / 8.83225   # -0.1%  n_pairs_kt_above_1 < 101
        - 0.0006279238 * max(0.0, 0.530706 - Q.sj3_dr23) / 0.1618797   # -0.1%  sj3_dr23 < 0.5307
        + 0.0005824087 * max(0.0, 0.8207647 - Q.sj3_pairmax_over_m) * max(0.0, Q.isphoton_53 - 0.0) / 0.007175802   # +0.1%  sj3_pairmax_over_m < 0.8208 and isphoton_53 > 0
        - 0.0005738569 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.sj4_pair_mass_min - 13.57513) / 133.7659   # -0.1%  mass < 164.4 and sj4_pair_mass_min > 13.58
        - 0.0005217738 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.sum_z_dr2_top20 - 0.0399789) / 0.1938807   # -0.1%  mass_displaced3 < 39.1 and sum_z_dr2_top20 > 0.03998
        + 0.0004896141 * max(0.0, Q.jet_abs_eta - 1.323111) / 0.03846033   # +0.0%  jet_abs_eta > 1.323
        - 0.0004722413 * max(0.0, 8.461136 - Q.sj3_mass1) / 0.4935496   # -0.0%  sj3_mass1 < 8.461
        - 0.0004247915 * max(0.0, 0.006220408 - Q.psi_0p1) / 0.0004886818   # -0.0%  psi_0p1 < 0.00622
        + 0.0003717292 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, Q.n_electron - 0.0) / 0.40422   # +0.0%  n_lund_kt_above_5 > 1 and n_electron > 0
        + 0.0003231121 * max(0.0, 107.8099 - Q.sj4_pair_mass_max) * max(0.0, 515.5711 - Q.sum_pt_top20) / 513.304   # +0.0%  sj4_pair_mass_max < 107.8 and sum_pt_top20 < 515.6
        + 0.000318685 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # +0.0%  max_abs_d0 < 5.812
        + 0.0003117436 * max(0.0, -2.0 - Q.sum_charge) / 0.1953233   # +0.0%  sum_charge < -2
        - 0.0001855748 * max(0.0, 107.8099 - Q.sj4_pair_mass_max) * max(0.0, Q.td0_9 - -0.01537965) / 1.165896   # -0.0%  sj4_pair_mass_max < 107.8 and td0_9 > -0.01538
        + 0.0001412062 * max(0.0, 107.8099 - Q.sj4_pair_mass_max) * max(0.0, Q.charge_43 - -1.0) / 29.80574   # +0.0%  sj4_pair_mass_max < 107.8 and charge_43 > -1
        - 0.0001243587 * max(0.0, Q.dr_0 - 0.2254414) / 0.007241025   # -0.0%  dr_0 > 0.2254
        - 0.0001100576 * max(0.0, 0.05966366 - Q.M2_b2) * max(0.0, Q.d0err_52 - 0.0) / 7.007326e-05   # -0.0%  M2_b2 < 0.05966 and d0err_52 > 0
        - 9.2949e-05 * max(0.0, Q.phi_26 - 0.1617432) / 0.02351405   # -0.0%  phi_26 > 0.1617
        - 5.115244e-05 * max(0.0, Q.lam1 - 0.05242126) / 0.003271758   # -0.0%  lam1 > 0.05242
        - 3.372738e-05 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, Q.tdz_44 - 0.01757784) / 0.02829223   # -0.0%  n_s3d_above_3 > 4 and tdz_44 > 0.01758
        - 2.858352e-05 * max(0.0, 107.8099 - Q.sj4_pair_mass_max) * max(0.0, 0.0 - Q.tdz_25) / 0.7807076   # -0.0%  sj4_pair_mass_max < 107.8 and tdz_25 < 0
        + 2.662142e-05 * max(0.0, 6.767937 - Q.mass_2charged) * max(0.0, 15.0 - Q.n_real_top15) / 0.1444146   # +0.0%  mass_2charged < 6.768 and n_real_top15 < 15
        + 2.325525e-05 * max(0.0, 28.89659 - Q.mass_2charged) * max(0.0, Q.phi_31 - 0.326416) / 0.1423609   # +0.0%  mass_2charged < 28.9 and phi_31 > 0.3264
        + 2.247875e-05 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, Q.iselectron_34 - 0.0) / 0.00481   # +0.0%  n_lund_kt_above_5 > 1 and iselectron_34 > 0
        - 2.163056e-05 * max(0.0, 4.326415 - Q.sj3_mass2) / 0.2966808   # -0.0%  sj3_mass2 < 4.326
    )
    return z


def neuron_116(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.39926e-07
    )
    return z


def neuron_117(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.86214e-06
    )
    return z


def neuron_118(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.23318e-06
    )
    return z


def neuron_119(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.337149e-07
    )
    return z


def neuron_120(Q):
    # scale S = 7.154; each line: share * term / its average size
    z = 7.154208 * (0.2689496
        - 0.1017313 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 10.0 - Q.n_s3d_above_3) / 15.42709   # -10.2%  lep_ptrel < 3.535 and n_s3d_above_3 < 10
        - 0.0941499 * max(0.0, 0.5187302 - Q.lep_z) / 0.4405854   # -9.4%  lep_z < 0.5187
        - 0.05853354 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 0.02781886 - Q.lep_iso) / 0.05591184   # -5.9%  lep_ptrel < 3.535 and lep_iso < 0.02782
        + 0.05674892 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 577.991 - Q.sip_3d_3) / 1181.582   # +5.7%  lep_ptrel < 3.535 and sip_3d_3 < 578
        - 0.05621477 * max(0.0, 12.15228 - Q.lep_ptrel) / 8.819929   # -5.6%  lep_ptrel < 12.15
        - 0.04940457 * max(0.0, 3.53503 - Q.lep_ptrel) / 2.299439   # -4.9%  lep_ptrel < 3.535
        - 0.04833893 * max(0.0, 114.0172 - Q.mass) / 13.82223   # -4.8%  mass < 114
        + 0.0406939 * max(0.0, 2.907433 - Q.lep_iso) / 2.168962   # +4.1%  lep_iso < 2.907
        - 0.02752645 * max(0.0, 105.7234 - Q.mass) / 10.09077   # -2.8%  mass < 105.7
        - 0.02706206 * max(0.0, 0.8860453 - Q.D3_b05) / 0.4670665   # -2.7%  D3_b05 < 0.886
        - 0.0245352 * max(0.0, 577.991 - Q.sip_3d_3) / 516.7039   # -2.5%  sip_3d_3 < 578
        + 0.02164009 * max(0.0, 2.0 - Q.n_electron) / 1.6571   # +2.2%  n_electron < 2
        + 0.02089685 * max(0.0, 0.02781886 - Q.lep_iso) / 0.01776032   # +2.1%  lep_iso < 0.02782
        + 0.02018542 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, Q.n_photon - 5.0) / 28.0407   # +2.0%  lep_ptrel < 3.535 and n_photon > 5
        - 0.02005776 * max(0.0, Q.lund3_lndelta - -2.817283) / 1.325774   # -2.0%  lund3_lndelta > -2.817
        + 0.01700431 * max(0.0, 90.08945 - Q.mass) / 4.972097   # +1.7%  mass < 90.09
        + 0.01670262 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.mass_displaced3 - 0.0) / 3.662943   # +1.7%  lep_z < 0.5187 and mass_displaced3 > 0
        + 0.01619608 * max(0.0, 25.0 - Q.n_charged) / 7.039403   # +1.6%  n_charged < 25
        - 0.01600478 * max(0.0, 0.1719087 - Q.sj3_z3) / 0.06883262   # -1.6%  sj3_z3 < 0.1719
        - 0.01344333 * max(0.0, Q.mass_top30 - 109.3251) / 10.36391   # -1.3%  mass_top30 > 109.3
        - 0.01280244 * max(0.0, Q.lep_ptrel - 18.7678) * max(0.0, 0.4381892 - Q.lep_iso) / 1.037293   # -1.3%  lep_ptrel > 18.77 and lep_iso < 0.4382
        + 0.01103962 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.2601259 - Q.z_displaced3) / 3.087204   # +1.1%  n_pairs_kt_above_3 < 62 and z_displaced3 < 0.2601
        + 0.01090373 * max(0.0, Q.e4_b05 - 1.4104e-05) / 3.051148e-05   # +1.1%  e4_b05 > 1.41e-05
        + 0.01037293 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 0.1889947 - Q.sj3_z3) / 0.7306739   # +1.0%  lep_ptrel < 12.15 and sj3_z3 < 0.189
        - 0.01006327 * max(0.0, 0.4174214 - Q.N2_b05) / 0.02045623   # -1.0%  N2_b05 < 0.4174
        - 0.009792055 * max(0.0, Q.mass_top10 - 48.96085) / 22.52231   # -1.0%  mass_top10 > 48.96
        + 0.009775755 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.mass_top10 - 48.96085) / 8.959233   # +1.0%  lep_z < 0.5187 and mass_top10 > 48.96
        - 0.009506311 * max(0.0, 4.0 - Q.n_sd0_above_2) / 1.197917   # -1.0%  n_sd0_above_2 < 4
        - 0.008972043 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) / 16.68515   # -0.9%  n_pairs_kt_above_3 < 62
        - 0.008866398 * max(0.0, 195.0 - Q.n_pairs_kt_above_1) / 37.35244   # -0.9%  n_pairs_kt_above_1 < 195
        - 0.00759675 * max(0.0, 68.20711 - Q.mass_charged) / 12.83892   # -0.8%  mass_charged < 68.21
        + 0.007481594 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 69.1565 - Q.mass_neutral) / 10.38924   # +0.7%  lep_z < 0.5187 and mass_neutral < 69.16
        + 0.00739156 * max(0.0, 0.6800935 - Q.tau32) / 0.0806267   # +0.7%  tau32 < 0.6801
        + 0.007357406 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 0.7095008 - Q.planar_flow) / 2.189065   # +0.7%  lep_ptrel < 12.15 and planar_flow < 0.7095
        + 0.007146821 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # +0.7%  lep_ptrel > 27.32
        - 0.006213488 * max(0.0, Q.mass_top15 - 104.4937) / 4.229149   # -0.6%  mass_top15 > 104.5
        - 0.006139269 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.z_neutral_had - 0.02425618) / 1.870834   # -0.6%  n_pairs_kt_above_3 < 62 and z_neutral_had > 0.02426
        + 0.005470875 * max(0.0, 3.0 - Q.n_neutral_had) / 0.4082667   # +0.5%  n_neutral_had < 3
        + 0.005447911 * max(0.0, Q.pair_mean_lnkt - 0.5586581) / 0.1900469   # +0.5%  pair_mean_lnkt > 0.5587
        + 0.005260844 * max(0.0, 5.8125 - Q.max_abs_d0) * max(0.0, 1.0 - Q.ismuon_0) / 2.671766   # +0.5%  max_abs_d0 < 5.812 and ismuon_0 < 1
        - 0.004419742 * max(0.0, Q.mass_top50 - 135.5368) / 6.535045   # -0.4%  mass_top50 > 135.5
        + 0.004233151 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # +0.4%  mass_top50 > 161.1
        + 0.004021522 * max(0.0, 0.3052386 - Q.D3_b05) / 0.03985092   # +0.4%  D3_b05 < 0.3052
        + 0.00389239 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # +0.4%  max_abs_d0 < 5.812
        + 0.00388204 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # +0.4%  sip_3d_2 < 226.3
        - 0.003722261 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 0.5715866 - Q.dr_16) / 3.224727   # -0.4%  lep_ptrel < 12.15 and dr_16 < 0.5716
        - 0.003539516 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, 0.1991254 - Q.mratio_min_012) / 0.2743283   # -0.4%  lep_ptrel > 27.32 and mratio_min_012 < 0.1991
        + 0.003410378 * max(0.0, 5.0 - Q.n_s3d_above_10) / 2.774583   # +0.3%  n_s3d_above_10 < 5
        - 0.003325578 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.n_lund - 5.0) / 78.27875   # -0.3%  n_pairs_kt_above_3 < 62 and n_lund > 5
        - 0.003101213 * max(0.0, 0.1258758 - Q.M2_b05) / 0.007276394   # -0.3%  M2_b05 < 0.1259
        + 0.003037595 * max(0.0, 1.938659 - Q.pt_entropy) / 0.03894921   # +0.3%  pt_entropy < 1.939
        + 0.002999946 * max(0.0, 0.1719087 - Q.sj3_z3) * max(0.0, 24.55881 - Q.sj2_mass2) / 0.8177711   # +0.3%  sj3_z3 < 0.1719 and sj2_mass2 < 24.56
        + 0.002578603 * max(0.0, Q.e4_b05 - 1.4104e-05) * max(0.0, Q.sum_pt - 550.4644) / 0.001444541   # +0.3%  e4_b05 > 1.41e-05 and sum_pt > 550.5
        + 0.002541046 * max(0.0, 0.1719087 - Q.sj3_z3) * max(0.0, 1.0 - Q.isphoton_42) / 0.05631444   # +0.3%  sj3_z3 < 0.1719 and isphoton_42 < 1
        - 0.002371229 * max(0.0, Q.M3_b05 - 0.08175231) / 0.002201961   # -0.2%  M3_b05 > 0.08175
        - 0.002348983 * Q.z_displaced5 / 0.09296654   # -0.2%  z_displaced5
        - 0.002040279 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # -0.2%  n_pairs_kt_above_1 < 80
        - 0.002006674 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # -0.2%  mass_top10 > 103.5
        - 0.001888614 * max(0.0, Q.lep_ptrel - 6.983043) / 5.140178   # -0.2%  lep_ptrel > 6.983
        - 0.001666585 * max(0.0, 0.01394245 - Q.dr_min_012) / 0.002973681   # -0.2%  dr_min_012 < 0.01394
        + 0.001602054 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # +0.2%  lep_ptrel > 43.21
        + 0.001595341 * max(0.0, Q.lep_ptrel - 18.7678) / 2.829169   # +0.2%  lep_ptrel > 18.77
        - 0.001484931 * max(0.0, 5.0 - Q.n_dr_0p1_0p2) / 0.5919233   # -0.1%  n_dr_0p1_0p2 < 5
        + 0.001447172 * max(0.0, Q.mass_top10 - 48.96085) * max(0.0, Q.eta_7 - -0.1883545) / 4.333651   # +0.1%  mass_top10 > 48.96 and eta_7 > -0.1884
        + 0.001427111 * max(0.0, 3.0 - Q.n_sd0_above_5) / 1.183617   # +0.1%  n_sd0_above_5 < 3
        - 0.001326489 * max(0.0, Q.sd_rg - 0.5562194) / 0.008836565   # -0.1%  sd_rg > 0.5562
        - 0.001276375 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, 4.606241 - Q.sip_3d_3) / 27.36421   # -0.1%  n_pairs_kt_above_3 < 62 and sip_3d_3 < 4.606
        - 0.001173439 * max(0.0, Q.mass_top30 - 162.7874) / 1.306268   # -0.1%  mass_top30 > 162.8
        - 0.001127138 * max(0.0, 0.3191471 - Q.z_charged_had) / 0.0155244   # -0.1%  z_charged_had < 0.3191
        + 0.001056459 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.lep_dr - 0.114659) / 0.02162773   # +0.1%  lep_z < 0.5187 and lep_dr > 0.1147
        - 0.0009977951 * max(0.0, Q.lund3_lndelta - -2.817283) * max(0.0, Q.sj3_dr13 - 0.577548) / 0.05077784   # -0.1%  lund3_lndelta > -2.817 and sj3_dr13 > 0.5775
        - 0.000956792 * max(0.0, 16.0 - Q.n_pt_above_5) / 0.9330767   # -0.1%  n_pt_above_5 < 16
        + 0.0008837871 * max(0.0, 0.4174214 - Q.N2_b05) * max(0.0, 0.8930677 - Q.sj3_dr_max) / 0.008151529   # +0.1%  N2_b05 < 0.4174 and sj3_dr_max < 0.8931
        + 0.0007987079 * max(0.0, Q.mass_top10 - 103.4976) * max(0.0, Q.log_sum_pt - 6.426552) / 0.1902225   # +0.1%  mass_top10 > 103.5 and log_sum_pt > 6.427
        + 0.000677276 * max(0.0, Q.z_neutral - 0.6734757) / 0.003638212   # +0.1%  z_neutral > 0.6735
        - 0.0006376453 * max(0.0, 4.0 - Q.n_sd0_above_2) * max(0.0, 0.3967994 - Q.dr_19) / 0.2672384   # -0.1%  n_sd0_above_2 < 4 and dr_19 < 0.3968
        - 0.0006133275 * max(0.0, 0.08995834 - Q.e2_b05) / 0.0009622265   # -0.1%  e2_b05 < 0.08996
        + 0.0006056863 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +0.1%  n_dr_0p4_up < 4
        - 0.0005898573 * max(0.0, 105.7234 - Q.mass) * max(0.0, Q.eta_0 - 0.04705811) / 0.05007309   # -0.1%  mass < 105.7 and eta_0 > 0.04706
        - 0.0005375757 * max(0.0, Q.pair_mean_lndelta - -1.523797) / 0.02402574   # -0.1%  pair_mean_lndelta > -1.524
        - 0.0005056986 * max(0.0, Q.z_displaced3 - 0.2092108) / 0.02926899   # -0.1%  z_displaced3 > 0.2092
        - 0.0004013393 * max(0.0, -3.360241 - Q.lnptrel_3) / 0.01767913   # -0.0%  lnptrel_3 < -3.36
        - 0.0003703104 * max(0.0, Q.lep_ptrel - 6.983043) * max(0.0, Q.jet_charge_k03 - 0.7410651) / 0.319818   # -0.0%  lep_ptrel > 6.983 and jet_charge_k03 > 0.7411
        + 0.0003154822 * max(0.0, 5.8125 - Q.max_abs_d0) * max(0.0, Q.d0err_12 - 0.01499939) / 0.01282045   # +0.0%  max_abs_d0 < 5.812 and d0err_12 > 0.015
        + 0.0002885698 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, -0.2314453 - Q.phi_22) / 0.1307046   # +0.0%  lep_ptrel < 12.15 and phi_22 < -0.2314
        + 0.00028218 * max(0.0, 1.824785 - Q.sj3_mass2) / 0.06171879   # +0.0%  sj3_mass2 < 1.825
        - 0.0002711408 * max(0.0, 5.0 - Q.n_dr_0p1_0p2) * max(0.0, -0.217041 - Q.eta_2) / 0.005386908   # -0.0%  n_dr_0p1_0p2 < 5 and eta_2 < -0.217
        - 0.0002526586 * max(0.0, 0.1132155 - Q.dr_4) / 0.02246667   # -0.0%  dr_4 < 0.1132
        - 0.0002429736 * max(0.0, Q.e3 - 0.00228569) / 0.000293828   # -0.0%  e3 > 0.002286
        - 0.0001511011 * max(0.0, Q.pair_mean_lnkt - 0.5586581) * max(0.0, Q.lund3_lnz - -5.757953) / 0.4705559   # -0.0%  pair_mean_lnkt > 0.5587 and lund3_lnz > -5.758
        - 0.0001258225 * max(0.0, 90.08945 - Q.mass) * max(0.0, Q.iselectron_1 - 0.0) / 0.1007217   # -0.0%  mass < 90.09 and iselectron_1 > 0
        + 8.614526e-05 * max(0.0, Q.M3_b05 - 0.08175231) * max(0.0, Q.td0_13 - 0.01621867) / 5.874042e-05   # +0.0%  M3_b05 > 0.08175 and td0_13 > 0.01622
        + 6.874013e-05 * Q.ecf_g42 / 2.611138e-05   # +0.0%  ecf_g42
        - 5.479318e-05 * max(0.0, 0.01394245 - Q.dr_min_012) * max(0.0, Q.ischhad_26 - 0.0) / 0.0009175735   # -0.0%  dr_min_012 < 0.01394 and ischhad_26 > 0
        - 2.893013e-05 * max(0.0, 2.907433 - Q.lep_iso) * max(0.0, Q.iselectron_12 - 0.0) / 0.003127301   # -0.0%  lep_iso < 2.907 and iselectron_12 > 0
        - 2.138293e-05 * max(0.0, 0.1719087 - Q.sj3_z3) * max(0.0, Q.ismuon_11 - 0.0) / 0.0002559885   # -0.0%  sj3_z3 < 0.1719 and ismuon_11 > 0
        + 1.853497e-05 * max(0.0, Q.mass_top10 - 48.96085) * max(0.0, Q.iselectron_14 - 0.0) / 0.1223086   # +0.0%  mass_top10 > 48.96 and iselectron_14 > 0
    )
    return z


def neuron_121(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.730505e-06
    )
    return z


def neuron_122(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.395913e-06
    )
    return z


def neuron_123(Q):
    # scale S = 3.738; each line: share * term / its average size
    z = 3.738156 * (0.01786827
        - 0.1060714 * max(0.0, 0.4232894 - Q.z_displaced3) / 0.3173391   # -10.6%  z_displaced3 < 0.4233
        + 0.08941067 * max(0.0, 0.3803178 - Q.z_displaced5) * max(0.0, 0.06457187 - Q.lep_z) / 0.01371315   # +8.9%  z_displaced5 < 0.3803 and lep_z < 0.06457
        - 0.07716269 * max(0.0, 0.3803178 - Q.z_displaced5) * max(0.0, 0.5187302 - Q.lep_z) / 0.1298273   # -7.7%  z_displaced5 < 0.3803 and lep_z < 0.5187
        + 0.07238466 * max(0.0, 0.3803178 - Q.z_displaced5) / 0.2926897   # +7.2%  z_displaced5 < 0.3803
        - 0.05288713 * max(0.0, 0.06457187 - Q.lep_z) / 0.044696   # -5.3%  lep_z < 0.06457
        + 0.03675002 * max(0.0, Q.n_s3d_above_3 - 1.0) / 2.933937   # +3.7%  n_s3d_above_3 > 1
        + 0.03630609 * max(0.0, Q.mass_top50 - 79.27954) / 36.81741   # +3.6%  mass_top50 > 79.28
        - 0.034788 * max(0.0, Q.mass_top50 - 84.74733) / 32.30684   # -3.5%  mass_top50 > 84.75
        + 0.03465753 * max(0.0, 4.606241 - Q.sip_3d_3) / 1.172198   # +3.5%  sip_3d_3 < 4.606
        - 0.02724021 * max(0.0, Q.n_s3d_above_3 - 1.0) * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0004668424   # -2.7%  n_s3d_above_3 > 1 and e3_b2 < 0.0002537
        + 0.0240067 * max(0.0, Q.M2_b05 - 0.1080247) / 0.03225212   # +2.4%  M2_b05 > 0.108
        - 0.01945723 * max(0.0, Q.lep_ptrel - 18.7678) / 2.829169   # -1.9%  lep_ptrel > 18.77
        + 0.01916707 * max(0.0, 0.6800935 - Q.tau32) / 0.0806267   # +1.9%  tau32 < 0.6801
        - 0.01904835 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) * max(0.0, 12.0 - Q.n_lund) / 25.01688   # -1.9%  n_pairs_kt_above_3 < 41 and n_lund < 12
        + 0.01724939 * max(0.0, 18.80005 - Q.mass_displaced3) / 13.42744   # +1.7%  mass_displaced3 < 18.8
        + 0.01538652 * max(0.0, 123.0 - Q.n_pairs_kt_above_1) / 13.76892   # +1.5%  n_pairs_kt_above_1 < 123
        - 0.01490802 * max(0.0, 6.983043 - Q.lep_ptrel) / 4.807808   # -1.5%  lep_ptrel < 6.983
        - 0.01427808 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # -1.4%  lep_ptrel > 27.32
        + 0.01380982 * max(0.0, Q.lep_ptrel - 18.7678) * max(0.0, 0.0004126585 - Q.e3_b2) / 0.000747565   # +1.4%  lep_ptrel > 18.77 and e3_b2 < 0.0004127
        - 0.01356366 * max(0.0, 0.6800935 - Q.tau32) * max(0.0, 0.7462286 - Q.sj3_dr13) / 0.0242097   # -1.4%  tau32 < 0.6801 and sj3_dr13 < 0.7462
        + 0.01212663 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) * max(0.0, 226.3008 - Q.sip_3d_2) / 1343.727   # +1.2%  n_pairs_kt_above_3 < 41 and sip_3d_2 < 226.3
        - 0.01199696 * max(0.0, 6.983043 - Q.lep_ptrel) * max(0.0, 4.606241 - Q.sip_3d_3) / 6.130914   # -1.2%  lep_ptrel < 6.983 and sip_3d_3 < 4.606
        + 0.01195613 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # +1.2%  lep_ptrel > 43.21
        + 0.01134557 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) / 7.619143   # +1.1%  n_pairs_kt_above_3 < 41
        - 0.01065946 * max(0.0, 0.002792418 - Q.sum_z_dr2_top3) / 0.0002360284   # -1.1%  sum_z_dr2_top3 < 0.002792
        - 0.01061261 * max(0.0, 4.0 - Q.n_sdz_above_5) / 2.084233   # -1.1%  n_sdz_above_5 < 4
        - 0.01002155 * max(0.0, Q.lep_z - 0.221436) / 0.03510013   # -1.0%  lep_z > 0.2214
        - 0.009285984 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) * max(0.0, 7.345216 - Q.sip_3d_3) / 27.14726   # -0.9%  n_pairs_kt_above_3 < 41 and sip_3d_3 < 7.345
        + 0.008279679 * max(0.0, Q.n_s3d_above_3 - 1.0) * max(0.0, 4.0 - Q.n_lund_kt_above_5) / 5.43464   # +0.8%  n_s3d_above_3 > 1 and n_lund_kt_above_5 < 4
        + 0.007459415 * max(0.0, 0.0004257509 - Q.C3_b2) / 8.06916e-05   # +0.7%  C3_b2 < 0.0004258
        + 0.007398756 * max(0.0, Q.mass_top5 - 68.52153) / 2.713876   # +0.7%  mass_top5 > 68.52
        + 0.007139031 * max(0.0, 0.4239562 - Q.z_charged_had) / 0.03843152   # +0.7%  z_charged_had < 0.424
        - 0.006721699 * max(0.0, Q.mass - 137.452) / 6.969039   # -0.7%  mass > 137.5
        - 0.006651909 * max(0.0, Q.n_s3d_above_3 - 1.0) * max(0.0, 578.2954 - Q.sip_3d_1) / 710.1849   # -0.7%  n_s3d_above_3 > 1 and sip_3d_1 < 578.3
        - 0.006434985 * max(0.0, 0.6800935 - Q.tau32) * max(0.0, 0.01108077 - Q.C3_b2) / 0.0004661421   # -0.6%  tau32 < 0.6801 and C3_b2 < 0.01108
        + 0.006330317 * max(0.0, Q.sj3_pair_mass_min - 37.19471) / 8.436037   # +0.6%  sj3_pair_mass_min > 37.19
        - 0.005917066 * max(0.0, 6.983043 - Q.lep_ptrel) * max(0.0, 0.0004257509 - Q.C3_b2) / 0.0004214202   # -0.6%  lep_ptrel < 6.983 and C3_b2 < 0.0004258
        - 0.005449803 * max(0.0, Q.lep_z - 0.221436) * max(0.0, 0.1591656 - Q.dr_25) / 0.003703423   # -0.5%  lep_z > 0.2214 and dr_25 < 0.1592
        + 0.004974752 * max(0.0, Q.n_s3d_above_3 - 1.0) * max(0.0, 0.1253926 - Q.dr_4) / 0.06548268   # +0.5%  n_s3d_above_3 > 1 and dr_4 < 0.1254
        - 0.004712656 * max(0.0, 0.3803178 - Q.z_displaced5) * max(0.0, Q.lnptrel_40 - -18.42068) / 1.631531   # -0.5%  z_displaced5 < 0.3803 and lnptrel_40 > -18.42
        + 0.004257777 * max(0.0, -2.572052 - Q.pair_mean_lnz) / 0.02771534   # +0.4%  pair_mean_lnz < -2.572
        - 0.003964092 * max(0.0, 0.6800935 - Q.tau32) * max(0.0, 0.02915846 - Q.dr_22) / 0.0003639338   # -0.4%  tau32 < 0.6801 and dr_22 < 0.02916
        + 0.00372875 * max(0.0, 4.696862e-06 - Q.e3_b2) / 4.050596e-07   # +0.4%  e3_b2 < 4.697e-06
        + 0.003472713 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # +0.3%  n_pairs_kt_above_1 < 58
        - 0.003371908 * max(0.0, Q.mass_top50 - 129.5874) * max(0.0, Q.sj3_dr13 - 0.2640447) / 2.414022   # -0.3%  mass_top50 > 129.6 and sj3_dr13 > 0.264
        - 0.00337127 * max(0.0, 14.92637 - Q.mass_neutral) / 0.561096   # -0.3%  mass_neutral < 14.93
        + 0.003357646 * max(0.0, 0.02926638 - Q.M2_b2) / 0.006137723   # +0.3%  M2_b2 < 0.02927
        + 0.003335426 * max(0.0, 5.0 - Q.n_lund) / 0.1017067   # +0.3%  n_lund < 5
        - 0.003279595 * max(0.0, 0.3265243 - Q.z_charged) / 0.003638213   # -0.3%  z_charged < 0.3265
        + 0.003197013 * max(0.0, Q.sj3_dr12 - 0.3470463) / 0.04884439   # +0.3%  sj3_dr12 > 0.347
        + 0.003034859 * max(0.0, Q.jet_charge_k05 - 0.5864519) / 0.01952642   # +0.3%  jet_charge_k05 > 0.5865
        + 0.003011092 * max(0.0, Q.pair_mean_lnm2 - 4.069088) / 0.0918484   # +0.3%  pair_mean_lnm2 > 4.069
        - 0.003000266 * max(0.0, Q.pair_max_lnm2 - 8.097646) / 0.0185767   # -0.3%  pair_max_lnm2 > 8.098
        + 0.002711827 * max(0.0, Q.mass_top50 - 79.27954) * max(0.0, 0.3603262 - Q.z_charged_had) / 0.6799953   # +0.3%  mass_top50 > 79.28 and z_charged_had < 0.3603
        + 0.002465982 * max(0.0, 4.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.4239562 - Q.z_charged_had) / 0.0296802   # +0.2%  n_dr_0p1_0p2 < 4 and z_charged_had < 0.424
        + 0.002403448 * max(0.0, 0.3803178 - Q.z_displaced5) * max(0.0, Q.M2_b05 - 0.1080247) / 0.009728268   # +0.2%  z_displaced5 < 0.3803 and M2_b05 > 0.108
        + 0.002307144 * max(0.0, Q.mass - 164.4374) / 3.038568   # +0.2%  mass > 164.4
        - 0.002220277 * max(0.0, 123.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.ischhad_4 - 0.0) / 8.454083   # -0.2%  n_pairs_kt_above_1 < 123 and ischhad_4 > 0
        + 0.002141223 * max(0.0, 4.606241 - Q.sip_3d_3) * max(0.0, 3.824557 - Q.lnpt_5) / 0.7553007   # +0.2%  sip_3d_3 < 4.606 and lnpt_5 < 3.825
        - 0.00211744 * max(0.0, Q.M3_b05 - 0.08175231) * max(0.0, Q.sd_zg - 0.2450652) / 0.0001306843   # -0.2%  M3_b05 > 0.08175 and sd_zg > 0.2451
        - 0.002047826 * max(0.0, Q.mass_top50 - 129.5874) / 7.866327   # -0.2%  mass_top50 > 129.6
        + 0.002003939 * max(0.0, Q.M3_b05 - 0.08175231) / 0.002201961   # +0.2%  M3_b05 > 0.08175
        - 0.001880383 * max(0.0, Q.lep_z - 0.221436) * max(0.0, 0.676889 - Q.sj3_dr13) / 0.009500555   # -0.2%  lep_z > 0.2214 and sj3_dr13 < 0.6769
        - 0.001874995 * max(0.0, 76.91486 - Q.sj3_pair_mass_max) * max(0.0, 175.9957 - Q.sip_3d_3) / 841.368   # -0.2%  sj3_pair_mass_max < 76.91 and sip_3d_3 < 176
        + 0.001790595 * max(0.0, 0.06457187 - Q.lep_z) * max(0.0, 4.606241 - Q.sip_3d_3) / 0.05651432   # +0.2%  lep_z < 0.06457 and sip_3d_3 < 4.606
        - 0.001768432 * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.3730633   # -0.2%  n_dr_0p1_0p2 < 4
        + 0.001766196 * max(0.0, Q.mass_2photon - 22.18431) / 0.4406114   # +0.2%  mass_2photon > 22.18
        - 0.001593372 * max(0.0, Q.mass_top5 - 68.52153) * max(0.0, 0.08983921 - Q.dr12) / 0.08429505   # -0.2%  mass_top5 > 68.52 and dr12 < 0.08984
        - 0.00158445 * max(0.0, Q.ecf_g41 - 0.0002327102) / 3.737478e-05   # -0.2%  ecf_g41 > 0.0002327
        + 0.001482263 * max(0.0, 76.91486 - Q.sj3_pair_mass_max) / 5.293372   # +0.1%  sj3_pair_mass_max < 76.91
        + 0.001464825 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.3078028 - Q.mratio_min_012) / 1.30982   # +0.1%  n_pairs_kt_above_3 < 41 and mratio_min_012 < 0.3078
        + 0.001423869 * max(0.0, 0.3803178 - Q.z_displaced5) * max(0.0, Q.n_lepton - 1.0) / 0.03844434   # +0.1%  z_displaced5 < 0.3803 and n_lepton > 1
        - 0.001348376 * max(0.0, Q.lep_ptrel - 18.7678) * max(0.0, 2.0 - Q.n_s3d_above_3) / 0.5512537   # -0.1%  lep_ptrel > 18.77 and n_s3d_above_3 < 2
        + 0.001252694 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.n_lund - 7.0) / 4.73168   # +0.1%  n_pairs_kt_above_1 < 58 and n_lund > 7
        + 0.001143533 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # +0.1%  sj2_mass2 < 1.852
        - 0.001126245 * max(0.0, 1.824785 - Q.sj3_mass2) / 0.06171879   # -0.1%  sj3_mass2 < 1.825
        - 0.001097246 * max(0.0, 76.91486 - Q.sj3_pair_mass_max) * max(0.0, 11.13866 - Q.sip_3d_3) / 37.84876   # -0.1%  sj3_pair_mass_max < 76.91 and sip_3d_3 < 11.14
        + 0.00106304 * max(0.0, -0.4474182 - Q.jet_charge_k05) / 0.03365211   # +0.1%  jet_charge_k05 < -0.4474
        - 0.0009375774 * max(0.0, 0.02926638 - Q.M2_b2) * max(0.0, 10.0 - Q.n_s3d_above_3) / 0.03986659   # -0.1%  M2_b2 < 0.02927 and n_s3d_above_3 < 10
        + 0.0008943017 * max(0.0, Q.lep_ptrel - 18.7678) * max(0.0, 82.17305 - Q.sip_3d_1) / 88.11245   # +0.1%  lep_ptrel > 18.77 and sip_3d_1 < 82.17
        + 0.0007413676 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, 0.7366642 - Q.sj3_pairmax_over_m) / 0.01198887   # +0.1%  lep_ptrel > 27.32 and sj3_pairmax_over_m < 0.7367
        - 0.0006669963 * max(0.0, Q.lep_ptrel - 43.20788) * max(0.0, Q.dr_4 - 0.2339164) / 0.0365315   # -0.1%  lep_ptrel > 43.21 and dr_4 > 0.2339
        + 0.0005906065 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, 2.0 - Q.n_s3d_above_3) / 0.2299827   # +0.1%  lep_ptrel > 27.32 and n_s3d_above_3 < 2
        + 0.0005463695 * max(0.0, Q.lep_ptrel - 18.7678) * max(0.0, 0.2429199 - Q.eta_8) / 0.7203425   # +0.1%  lep_ptrel > 18.77 and eta_8 < 0.2429
        + 0.0004734224 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.iselectron_1 - 0.0) / 0.2964133   # +0.0%  n_pairs_kt_above_3 < 41 and iselectron_1 > 0
        + 0.0004030545 * max(0.0, 0.6800935 - Q.tau32) * max(0.0, Q.eta_29 - 0.3234863) / 0.0008195142   # +0.0%  tau32 < 0.6801 and eta_29 > 0.3235
        + 0.000398493 * max(0.0, Q.lep_ptrel - 18.7678) * max(0.0, Q.ecf_g42 - 2.448333e-05) / 4.361725e-05   # +0.0%  lep_ptrel > 18.77 and ecf_g42 > 2.448e-05
        - 0.0003440744 * max(0.0, Q.n_s3d_above_3 - 1.0) * max(0.0, Q.ismuon_14 - 0.0) / 0.01448667   # -0.0%  n_s3d_above_3 > 1 and ismuon_14 > 0
        + 0.0002590126 * max(0.0, Q.M3_b05 - 0.08175231) * max(0.0, 7.345216 - Q.sip_3d_3) / 0.007548354   # +0.0%  M3_b05 > 0.08175 and sip_3d_3 < 7.345
        - 0.0002511393 * max(0.0, 0.3803178 - Q.z_displaced5) * max(0.0, Q.lep_dr - 0.1864963) / 0.008369401   # -0.0%  z_displaced5 < 0.3803 and lep_dr > 0.1865
        + 0.0002351304 * max(0.0, Q.n_s3d_above_3 - 1.0) * max(0.0, -0.1665039 - Q.eta_19) / 0.06823572   # +0.0%  n_s3d_above_3 > 1 and eta_19 < -0.1665
        - 0.0002183978 * max(0.0, 0.02926638 - Q.M2_b2) * max(0.0, Q.dr_1 - 0.1779524) / 0.000133946   # -0.0%  M2_b2 < 0.02927 and dr_1 > 0.178
        - 0.0002067993 * max(0.0, 0.06457187 - Q.lep_z) * max(0.0, Q.lund1_lnz - -4.349248) / 0.04345521   # -0.0%  lep_z < 0.06457 and lund1_lnz > -4.349
        + 0.0001921938 * max(0.0, 0.6800935 - Q.tau32) * max(0.0, Q.dr_10 - 0.5063302) / 0.0007099037   # +0.0%  tau32 < 0.6801 and dr_10 > 0.5063
        + 0.0001101123 * max(0.0, 0.4239562 - Q.z_charged_had) * max(0.0, 0.0 - Q.charge_27) / 0.003091592   # +0.0%  z_charged_had < 0.424 and charge_27 < 0
        + 3.368607e-05 * max(0.0, Q.lep_ptrel - 18.7678) * max(0.0, -6.508771 - Q.lund1_lnz) / 0.1190452   # +0.0%  lep_ptrel > 18.77 and lund1_lnz < -6.509
        - 2.700221e-05 * max(0.0, Q.lep_ptrel - 43.20788) * max(0.0, Q.lnpt_41 - -18.42068) / 1.930561   # -0.0%  lep_ptrel > 43.21 and lnpt_41 > -18.42
    )
    return z


def neuron_124(Q):
    # scale S = 10.5; each line: share * term / its average size
    z = 10.49663 * (0.003187171
        - 0.1120175 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # -11.2%  sd_mass > 78.48
        + 0.08403375 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # +8.4%  sd_mass > 88.82
        + 0.0585655 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # +5.9%  sd_mass > 94.56
        + 0.04263638 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # +4.3%  n_lepton < 1
        - 0.04037548 * max(0.0, Q.mass_top40 - 70.88236) / 41.46808   # -4.0%  mass_top40 > 70.88
        + 0.03818855 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +3.8%  lep_ptrel < 43.21
        + 0.03756234 * max(0.0, 8.0 - Q.n_s3d_above_10) / 5.470973   # +3.8%  n_s3d_above_10 < 8
        - 0.03575388 * max(0.0, 0.09942631 - Q.M3_b05) / 0.0421804   # -3.6%  M3_b05 < 0.09943
        - 0.0356634 * max(0.0, Q.sd_mass - 127.8042) / 7.009137   # -3.6%  sd_mass > 127.8
        - 0.03145408 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # -3.1%  lep_iso < 1.362
        + 0.02848633 * max(0.0, 0.0729277 - Q.mass_over_sum_pt_sq) / 0.03640267   # +2.8%  mass_over_sum_pt_sq < 0.07293
        + 0.02759625 * max(0.0, 0.0002944259 - Q.ecf_g41) / 0.0001030636   # +2.8%  ecf_g41 < 0.0002944
        + 0.02520568 * max(0.0, Q.mass_top40 - 70.88236) * max(0.0, Q.lnerel_17 - -6.957861) / 100.3377   # +2.5%  mass_top40 > 70.88 and lnerel_17 > -6.958
        - 0.023724 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, 0.2675458 - Q.z_neutral_had) / 0.7200105   # -2.4%  n_s3d_above_10 < 8 and z_neutral_had < 0.2675
        + 0.02322827 * max(0.0, 577.991 - Q.sip_3d_3) / 516.7039   # +2.3%  sip_3d_3 < 578
        - 0.02182616 * max(0.0, 117.4867 - Q.mass) / 15.64803   # -2.2%  mass < 117.5
        - 0.01670921 * max(0.0, 702.0 - Q.n_pairs_kt_above_1) / 397.4307   # -1.7%  n_pairs_kt_above_1 < 702
        - 0.01663931 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # -1.7%  sd_mass > 106.8
        + 0.01647078 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # +1.6%  lep_z < 0.2214
        + 0.01564931 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +1.6%  mass < 95.15
        + 0.01405092 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +1.4%  sd_mass > 154.6
        - 0.01212207 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, 142.9952 - Q.sj3_pair_mass_max) / 1932.344   # -1.2%  lep_ptrel < 43.21 and sj3_pair_mass_max < 143
        - 0.01200079 * max(0.0, 97.12186 - Q.mass_top40) / 7.43888   # -1.2%  mass_top40 < 97.12
        + 0.0115577 * max(0.0, 2.0 - Q.n_s3d_above_3) / 0.4717633   # +1.2%  n_s3d_above_3 < 2
        - 0.01119532 * max(0.0, 0.5076533 - Q.sj2_dr) / 0.1421645   # -1.1%  sj2_dr < 0.5077
        + 0.01091077 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # +1.1%  lep_z < 0.3397
        - 0.0108068 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # -1.1%  lep_iso < 0.4382
        - 0.0102079 * max(0.0, 0.06041764 - Q.lam1) / 0.02970598   # -1.0%  lam1 < 0.06042
        + 0.01005746 * max(0.0, 93.87663 - Q.sj3_pair_mass_max) / 12.55772   # +1.0%  sj3_pair_mass_max < 93.88
        + 0.009984678 * max(0.0, 15.0 - Q.n_dr_0p4_up) / 9.48698   # +1.0%  n_dr_0p4_up < 15
        - 0.009704511 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 10.52344 - Q.max_abs_d0) / 6.106458   # -1.0%  n_s3d_above_3 > 3 and max_abs_d0 < 10.52
        - 0.00965416 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # -1.0%  n_pairs_kt_above_1 < 366
        - 0.008996315 * max(0.0, 0.03624058 - Q.z_displaced3) / 0.01247855   # -0.9%  z_displaced3 < 0.03624
        - 0.00788498 * max(0.0, 4.41005 - Q.mass_displaced3) / 2.286034   # -0.8%  mass_displaced3 < 4.41
        - 0.007678344 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) * max(0.0, 15.17086 - Q.D2_b2) / 1437.013   # -0.8%  n_pairs_kt_above_1 < 366 and D2_b2 < 15.17
        - 0.007446322 * max(0.0, Q.mass_top40 - 70.88236) * max(0.0, 4.726476e-05 - Q.ecf_g42) / 0.0008572976   # -0.7%  mass_top40 > 70.88 and ecf_g42 < 4.726e-05
        + 0.006435528 * max(0.0, Q.n_s3d_above_3 - 3.0) / 1.66923   # +0.6%  n_s3d_above_3 > 3
        - 0.006347941 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 0.06762785 - Q.M2_b2) / 0.05887994   # -0.6%  n_s3d_above_3 > 3 and M2_b2 < 0.06763
        - 0.006209464 * max(0.0, 100.4835 - Q.mass) / 8.115061   # -0.6%  mass < 100.5
        - 0.005111121 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, Q.z_photon - 0.05998812) / 0.3053896   # -0.5%  n_s3d_above_3 > 3 and z_photon > 0.05999
        - 0.005065553 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # -0.5%  sj3_pair_mass_max < 73.25
        - 0.004964584 * max(0.0, Q.n_lund - 7.0) / 4.044803   # -0.5%  n_lund > 7
        - 0.004546654 * max(0.0, 1.0 - Q.n_s3d_above_3) / 0.17383   # -0.5%  n_s3d_above_3 < 1
        + 0.004297948 * max(0.0, 15.0 - Q.n_dr_0p4_up) * max(0.0, Q.jet_charge_k03 - 0.1164034) / 2.156297   # +0.4%  n_dr_0p4_up < 15 and jet_charge_k03 > 0.1164
        + 0.004130058 * max(0.0, 0.5076533 - Q.sj2_dr) * max(0.0, 0.1403354 - Q.z_muon) / 0.0170514   # +0.4%  sj2_dr < 0.5077 and z_muon < 0.1403
        - 0.003895739 * max(0.0, Q.sd_mass - 42.31629) / 58.72614   # -0.4%  sd_mass > 42.32
        - 0.003747447 * max(0.0, 0.08304558 - Q.z_displaced3) / 0.03742531   # -0.4%  z_displaced3 < 0.08305
        - 0.003316643 * max(0.0, 0.01365334 - Q.ecf_g31) / 0.007190153   # -0.3%  ecf_g31 < 0.01365
        + 0.002554684 * max(0.0, 15.0 - Q.n_charged_pt_above_1) / 1.32672   # +0.3%  n_charged_pt_above_1 < 15
        + 0.002550152 * max(0.0, 0.007018285 - Q.sum_z_dr2_top3) / 0.001205932   # +0.3%  sum_z_dr2_top3 < 0.007018
        - 0.002524283 * max(0.0, 0.5076533 - Q.sj2_dr) * max(0.0, 5.114244 - Q.lne_2) / 0.1161855   # -0.3%  sj2_dr < 0.5077 and lne_2 < 5.114
        + 0.002494677 * Q.M3 / 0.03605934   # +0.2%  M3
        - 0.002454469 * max(0.0, 6.0 - Q.n_dr_0p1_0p2) / 0.8571967   # -0.2%  n_dr_0p1_0p2 < 6
        + 0.002317888 * max(0.0, 0.4008517 - Q.LHA) / 0.03854908   # +0.2%  LHA < 0.4009
        + 0.002292502 * max(0.0, 0.2428609 - Q.z_dr_0p1_0p2) / 0.07286283   # +0.2%  z_dr_0p1_0p2 < 0.2429
        - 0.002042353 * max(0.0, 100.4835 - Q.mass) * max(0.0, 30.0 - Q.n_real_top30) / 49.03597   # -0.2%  mass < 100.5 and n_real_top30 < 30
        + 0.002022822 * max(0.0, Q.mass - 182.8592) / 1.688248   # +0.2%  mass > 182.9
        + 0.001853241 * max(0.0, 0.5076533 - Q.sj2_dr) * max(0.0, Q.lund1_lnz - -4.349248) / 0.1067436   # +0.2%  sj2_dr < 0.5077 and lund1_lnz > -4.349
        - 0.001770377 * max(0.0, Q.N3_b05 - 0.4404318) / 0.3441255   # -0.2%  N3_b05 > 0.4404
        - 0.001748814 * max(0.0, 1.777286 - Q.mass_displaced3) / 0.7377974   # -0.2%  mass_displaced3 < 1.777
        - 0.001663228 * max(0.0, Q.z_displaced5 - 0.2801368) / 0.0126536   # -0.2%  z_displaced5 > 0.2801
        + 0.001597691 * max(0.0, Q.pair_max_lnm2 - 6.520267) / 0.4040215   # +0.2%  pair_max_lnm2 > 6.52
        - 0.001590108 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # -0.2%  n_pairs_kt_above_1 < 58
        - 0.001495793 * max(0.0, Q.mass_top40 - 70.88236) * max(0.0, 3.808744 - Q.lne_4) / 8.721589   # -0.1%  mass_top40 > 70.88 and lne_4 < 3.809
        + 0.001453894 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, Q.n_lund_kt_above_1 - 3.0) / 12.43708   # +0.1%  n_s3d_above_10 < 8 and n_lund_kt_above_1 > 3
        - 0.001443209 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.z_displaced5 - 0.1046203) / 0.01140989   # -0.1%  lep_z < 0.3397 and z_displaced5 > 0.1046
        + 0.001297156 * max(0.0, 1.777286 - Q.mass_displaced3) * max(0.0, 9.0 - Q.n_neutral_had) / 3.847284   # +0.1%  mass_displaced3 < 1.777 and n_neutral_had < 9
        - 0.001060714 * max(0.0, 0.006220408 - Q.psi_0p1) / 0.0004886818   # -0.1%  psi_0p1 < 0.00622
        + 0.001057777 * max(0.0, Q.lep_dr - 0.4191372) / 0.007569263   # +0.1%  lep_dr > 0.4191
        + 0.001014217 * max(0.0, 63.7508 - Q.sj3_pair_mass_max) / 2.405567   # +0.1%  sj3_pair_mass_max < 63.75
        - 0.0009541334 * max(0.0, 1.824785 - Q.sj3_mass2) / 0.06171879   # -0.1%  sj3_mass2 < 1.825
        - 0.0009166375 * max(0.0, 0.5076533 - Q.sj2_dr) * max(0.0, Q.iselectron_0 - 0.0) / 0.01382147   # -0.1%  sj2_dr < 0.5077 and iselectron_0 > 0
        - 0.0006588918 * max(0.0, Q.tdz_2 - -0.02221619) / 0.04771882   # -0.1%  tdz_2 > -0.02222
        - 0.0005557735 * max(0.0, 25.0 - Q.n_charged_had) / 7.529693   # -0.1%  n_charged_had < 25
        + 0.0004731308 * max(0.0, 0.3164143 - Q.tau32) * max(0.0, 0.02580261 - Q.d0err_14) / 5.010672e-05   # +0.0%  tau32 < 0.3164 and d0err_14 < 0.0258
        + 0.0004109759 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, 0.019104 - Q.d0err_16) / 0.05526866   # +0.0%  n_s3d_above_10 < 8 and d0err_16 < 0.0191
        - 0.0003816267 * max(0.0, 0.5076533 - Q.sj2_dr) * max(0.0, 0.4381892 - Q.lep_iso) / 0.04132887   # -0.0%  sj2_dr < 0.5077 and lep_iso < 0.4382
        - 0.0003215374 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, -0.1462819 - Q.tdz_2) / 0.4585186   # -0.0%  lep_ptrel < 43.21 and tdz_2 < -0.1463
        + 0.000292572 * max(0.0, 45.8749 - Q.mass_top15) / 1.516652   # +0.0%  mass_top15 < 45.87
        + 0.0001556548 * max(0.0, -0.05792002 - Q.td0_17) / 0.02153752   # +0.0%  td0_17 < -0.05792
        + 0.0001435117 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, Q.charge_42 - 0.0) / 0.1636467   # +0.0%  n_s3d_above_3 > 3 and charge_42 > 0
        - 0.0001041828 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.sj3_z3 - 0.09421497) / 1.597426   # -0.0%  lep_ptrel < 43.21 and sj3_z3 > 0.09421
        + 6.934039e-05 * max(0.0, Q.pair_max_lnm2 - 6.520267) * max(0.0, -0.06205547 - Q.td0_14) / 0.009653046   # +0.0%  pair_max_lnm2 > 6.52 and td0_14 < -0.06206
        - 5.684076e-05 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.iselectron_1 - 0.0) / 4.67319   # -0.0%  n_pairs_kt_above_1 < 366 and iselectron_1 > 0
        - 3.838555e-05 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, Q.eta_76 - 0.0) / 0.008868921   # -0.0%  n_s3d_above_10 < 8 and eta_76 > 0
        - 2.665876e-05 * max(0.0, 0.3164143 - Q.tau32) / 0.003288023   # -0.0%  tau32 < 0.3164
        + 2.244936e-05 * max(0.0, 702.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.ismuon_37 - 0.0) / 0.17736   # +0.0%  n_pairs_kt_above_1 < 702 and ismuon_37 > 0
        - 3.452164e-06 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.3329206 - Q.tau43_b2) / 0.0009626607   # -0.0%  lep_z < 0.3397 and tau43_b2 < 0.3329
        - 3.340029e-07 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.ismuon_49 - 0.0) / 0.0009600611   # -0.0%  mass < 117.5 and ismuon_49 > 0
    )
    return z


def neuron_125(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.190753e-06
    )
    return z


def neuron_126(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.6326e-06
    )
    return z


def neuron_127(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.966841e-07
    )
    return z


def class_token(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q), neuron_16(Q), neuron_17(Q), neuron_18(Q), neuron_19(Q), neuron_20(Q), neuron_21(Q), neuron_22(Q), neuron_23(Q), neuron_24(Q), neuron_25(Q), neuron_26(Q), neuron_27(Q), neuron_28(Q), neuron_29(Q), neuron_30(Q), neuron_31(Q), neuron_32(Q), neuron_33(Q), neuron_34(Q), neuron_35(Q), neuron_36(Q), neuron_37(Q), neuron_38(Q), neuron_39(Q), neuron_40(Q), neuron_41(Q), neuron_42(Q), neuron_43(Q), neuron_44(Q), neuron_45(Q), neuron_46(Q), neuron_47(Q), neuron_48(Q), neuron_49(Q), neuron_50(Q), neuron_51(Q), neuron_52(Q), neuron_53(Q), neuron_54(Q), neuron_55(Q), neuron_56(Q), neuron_57(Q), neuron_58(Q), neuron_59(Q), neuron_60(Q), neuron_61(Q), neuron_62(Q), neuron_63(Q), neuron_64(Q), neuron_65(Q), neuron_66(Q), neuron_67(Q), neuron_68(Q), neuron_69(Q), neuron_70(Q), neuron_71(Q), neuron_72(Q), neuron_73(Q), neuron_74(Q), neuron_75(Q), neuron_76(Q), neuron_77(Q), neuron_78(Q), neuron_79(Q), neuron_80(Q), neuron_81(Q), neuron_82(Q), neuron_83(Q), neuron_84(Q), neuron_85(Q), neuron_86(Q), neuron_87(Q), neuron_88(Q), neuron_89(Q), neuron_90(Q), neuron_91(Q), neuron_92(Q), neuron_93(Q), neuron_94(Q), neuron_95(Q), neuron_96(Q), neuron_97(Q), neuron_98(Q), neuron_99(Q), neuron_100(Q), neuron_101(Q), neuron_102(Q), neuron_103(Q), neuron_104(Q), neuron_105(Q), neuron_106(Q), neuron_107(Q), neuron_108(Q), neuron_109(Q), neuron_110(Q), neuron_111(Q), neuron_112(Q), neuron_113(Q), neuron_114(Q), neuron_115(Q), neuron_116(Q), neuron_117(Q), neuron_118(Q), neuron_119(Q), neuron_120(Q), neuron_121(Q), neuron_122(Q), neuron_123(Q), neuron_124(Q), neuron_125(Q), neuron_126(Q), neuron_127(Q)]


H_AVG = [0.00519286934286356, 5.325269739842042e-06, 5.447309831652092e-06, 8.399862053920515e-06, 2.7921546461584512e-06, 2.156869186364929e-06, 1.770100607245695e-05, 6.4308210312447045e-06, 2.9264551812957507e-06, 2.4436606054223375e-06, 1.570105814607814e-05, 1.8877162801800296e-05, 5.099316240375629e-06, 4.422903202794259e-06, 2.845305516530061e-06, 6.459516953327693e-06, 1.3124225750408958, 1.0353551260777749e-05, 0.8873173405263857, 1.4078832464292645e-06, 8.253655323642306e-07, 0.001686458708718419, 1.0448824241393595e-06, 6.979495879022579e-07, 1.0646936907496725, 0.1545660074478739, 3.23129665957822e-06, 6.308482170425123e-06, 9.617222531232983e-06, 9.183853762806393e-06, 8.603078640589956e-06, 3.509509269861155e-06, 1.4413515600608662e-05, 2.5673205072962446e-06, 1.3884820873499848e-05, 6.7433038566377945e-06, 4.831943442695774e-06, 4.357445050118258e-06, 9.411090104549658e-06, 8.274269930552691e-06, 7.775537937959598e-07, 3.814237814481203e-08, 8.779078598308843e-06, 9.40769496082794e-06, 3.3243563848373014e-06, 6.849297278677113e-06, 1.3688823230495473e-07, 2.492049873126234e-07, 8.791764230409171e-06, 2.243813696622965e-06, 5.175112619326683e-06, 2.012209961321787e-06, 0.6927176819368068, 1.889392137854884e-06, 1.1906564395758323e-05, 2.066932211164385e-05, 7.072387120388157e-07, 1.9596764104790054e-06, 3.6171500141790602e-06, 6.0292718444543425e-06, 4.666768290917389e-06, 5.840545782120898e-06, 9.422871016795398e-07, 8.308530595968477e-06, 1.258629481526441e-06, 5.791076091554714e-06, 2.709754880925175e-06, 5.219533250055974e-06, 0.9619812672402803, 6.926153218955733e-06, 0.24724315337120056, 3.029693061762373e-06, 1.164101036010834e-06, 1.699636413832195e-05, 4.4934126890439074e-06, 1.9738313312700484e-06, 5.542784492718056e-06, 3.087468030571472e-06, 0.46346657009452474, 3.602392825996503e-05, 9.707703611638863e-06, 0.43967644800267747, 2.9420859846140957e-06, 0.9315603441960337, 7.427523814840242e-05, 8.185612614397542e-07, 1.7099026081268676e-05, 1.6191603435800062e-06, 2.99680692705806e-07, 1.7353464727420942e-06, 0.18260922862322762, 3.992630354332505e-06, 6.724725153617328e-06, 8.464321012979781e-07, 2.3222107756737387e-06, 4.369355508515582e-07, 4.2777256226145255e-07, 0.9965426460197979, 4.99015129662439e-07, 0.0019342793384566903, 6.264069725148147e-06, 8.878267180989496e-06, 5.490486728376709e-06, 3.4143481570936274e-06, 1.1695790780984914, 2.732384564296808e-05, 2.1609239411191083e-05, 1.1585599168029148e-05, 4.470689418667462e-06, 5.200096779844898e-07, 3.579629265004769e-06, 3.6696130791824544e-06, 5.237811819824856e-06, 2.6659345166990533e-05, 4.677422566601308e-06, 0.9304932984986668, 3.399259753678052e-07, 2.8621400360862026e-06, 8.233179869421292e-06, 6.337148761303979e-07, 1.036848848081272, 5.730504653911339e-06, 2.395912815700285e-06, 0.20248684538492273, 0.7000283409351076, 1.1907533234989387e-06, 7.632599590579048e-06, 1.966840841305384e-07]
T = [4.191437131027967, 6.014148196101935, 4.249994994983624, 4.814083594428125, 5.10357334632402, 9.125427673037036, 4.181306226133169, 4.58992424337425, 7.680802570508697, 11.09934039050986]


def logits(h):
    # rounded to a multiple of 2^-20 (the formula's class scores are exact binary fractions): removes floating-point noise
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        0.4043572 + T[0] * (   # class QCD
            - 0.1482775 * h[18] / H_AVG[18]
            + 0.1418028 * h[97] / H_AVG[97]
            - 0.1133493 * h[68] / H_AVG[68]
            - 0.09770477 * h[120] / H_AVG[120]
            + 0.09135177 * h[104] / H_AVG[104]
            + 0.08497759 * h[52] / H_AVG[52]
            - 0.08394071 * h[24] / H_AVG[24]
            + 0.06781785 * h[81] / H_AVG[81]
            - 0.03338061 * h[83] / H_AVG[83]
            + 0.03188151 * h[16] / H_AVG[16]
            - 0.03045335 * h[124] / H_AVG[124]
            + 0.02940266 * h[78] / H_AVG[78]
            + 0.02827783 * h[115] / H_AVG[115]
            - 0.009940903 * h[90] / H_AVG[90]
            + 0.003373723 * h[123] / H_AVG[123]
            + 0.002877412 * h[70] / H_AVG[70]
            + 0.001048375 * h[25] / H_AVG[25]
            + 6.303819e-05 * h[0] / H_AVG[0]
            - 4.077889e-05 * h[99] / H_AVG[99]
            - 3.150919e-05 * h[21] / H_AVG[21]
            + 1.926338e-06 * h[84] / H_AVG[84]
            + 4.544602e-07 * h[55] / H_AVG[55]
            + 3.050143e-07 * h[79] / H_AVG[79]
            - 2.911843e-07 * h[105] / H_AVG[105]
            - 2.362952e-07 * h[106] / H_AVG[106]
            + 1.920742e-07 * h[113] / H_AVG[113]
            - 1.824048e-07 * h[34] / H_AVG[34]
            + 1.637878e-07 * h[73] / H_AVG[73]
            + 1.521117e-07 * h[10] / H_AVG[10]
            + 1.449549e-07 * h[11] / H_AVG[11]
            - 1.234562e-07 * h[6] / H_AVG[6]
            + 1.1781e-07 * h[86] / H_AVG[86]
            - 1.12029e-07 * h[32] / H_AVG[32]
            + 1.029685e-07 * h[17] / H_AVG[17]
            + 9.305252e-08 * h[38] / H_AVG[38]
            - 7.641626e-08 * h[42] / H_AVG[42]
            - 6.810219e-08 * h[54] / H_AVG[54]
            - 6.472885e-08 * h[29] / H_AVG[29]
            + 5.868483e-08 * h[118] / H_AVG[118]
            + 4.580949e-08 * h[48] / H_AVG[48]
            - 4.371777e-08 * h[45] / H_AVG[45]
            - 4.222116e-08 * h[39] / H_AVG[39]
            + 3.892664e-08 * h[27] / H_AVG[27]
            + 3.876631e-08 * h[63] / H_AVG[63]
            + 3.835278e-08 * h[30] / H_AVG[30]
            + 3.755692e-08 * h[43] / H_AVG[43]
            + 3.493542e-08 * h[121] / H_AVG[121]
            - 3.431546e-08 * h[80] / H_AVG[80]
            - 3.140844e-08 * h[126] / H_AVG[126]
            + 3.03202e-08 * h[60] / H_AVG[60]
            - 2.986267e-08 * h[61] / H_AVG[61]
            - 2.835588e-08 * h[102] / H_AVG[102]
            - 2.83135e-08 * h[107] / H_AVG[107]
            - 2.787332e-08 * h[7] / H_AVG[7]
            + 2.510882e-08 * h[92] / H_AVG[92]
            + 2.45554e-08 * h[35] / H_AVG[35]
            + 2.387983e-08 * h[15] / H_AVG[15]
            - 2.306146e-08 * h[59] / H_AVG[59]
            - 2.224782e-08 * h[101] / H_AVG[101]
            - 2.126908e-08 * h[28] / H_AVG[28]
            + 2.062493e-08 * h[103] / H_AVG[103]
            + 1.941238e-08 * h[69] / H_AVG[69]
            + 1.884023e-08 * h[67] / H_AVG[67]
            + 1.82492e-08 * h[1] / H_AVG[1]
            + 1.811407e-08 * h[100] / H_AVG[100]
            - 1.663672e-08 * h[110] / H_AVG[110]
            - 1.609764e-08 * h[3] / H_AVG[3]
            + 1.546312e-08 * h[37] / H_AVG[37]
            - 1.514772e-08 * h[50] / H_AVG[50]
            + 1.423864e-08 * h[44] / H_AVG[44]
            + 1.42273e-08 * h[36] / H_AVG[36]
            - 1.398545e-08 * h[12] / H_AVG[12]
            - 1.330441e-08 * h[114] / H_AVG[114]
            - 1.319038e-08 * h[2] / H_AVG[2]
            + 1.27125e-08 * h[58] / H_AVG[58]
            - 1.265923e-08 * h[112] / H_AVG[112]
            + 1.239038e-08 * h[76] / H_AVG[76]
            - 1.194406e-08 * h[13] / H_AVG[13]
            - 1.159975e-08 * h[19] / H_AVG[19]
            - 1.138503e-08 * h[111] / H_AVG[111]
            + 1.085076e-08 * h[74] / H_AVG[74]
            + 1.029643e-08 * h[108] / H_AVG[108]
            + 9.678681e-09 * h[5] / H_AVG[5]
            - 9.251557e-09 * h[8] / H_AVG[8]
            - 8.871257e-09 * h[91] / H_AVG[91]
            - 8.270439e-09 * h[4] / H_AVG[4]
            - 7.922597e-09 * h[117] / H_AVG[117]
            + 6.694825e-09 * h[53] / H_AVG[53]
            - 6.620429e-09 * h[66] / H_AVG[66]
            - 6.025521e-09 * h[33] / H_AVG[33]
            - 5.854273e-09 * h[65] / H_AVG[65]
            + 5.413608e-09 * h[82] / H_AVG[82]
            - 5.087307e-09 * h[31] / H_AVG[31]
            - 4.674894e-09 * h[71] / H_AVG[71]
            + 4.342171e-09 * h[94] / H_AVG[94]
            + 4.193646e-09 * h[9] / H_AVG[9]
            + 3.686232e-09 * h[26] / H_AVG[26]
            - 3.303444e-09 * h[14] / H_AVG[14]
            + 3.272452e-09 * h[57] / H_AVG[57]
            + 3.092872e-09 * h[49] / H_AVG[49]
            + 3.021391e-09 * h[77] / H_AVG[77]
            - 3.015351e-09 * h[22] / H_AVG[22]
            + 2.797286e-09 * h[122] / H_AVG[122]
            + 2.261291e-09 * h[72] / H_AVG[72]
            + 2.16584e-09 * h[51] / H_AVG[51]
            - 2.132415e-09 * h[23] / H_AVG[23]
            - 1.472475e-09 * h[89] / H_AVG[89]
            + 1.438722e-09 * h[125] / H_AVG[125]
            - 1.350709e-09 * h[75] / H_AVG[75]
            - 1.168851e-09 * h[87] / H_AVG[87]
            + 1.164961e-09 * h[93] / H_AVG[93]
            - 1.04768e-09 * h[40] / H_AVG[40]
            - 9.734997e-10 * h[20] / H_AVG[20]
            + 8.69878e-10 * h[119] / H_AVG[119]
            - 7.956827e-10 * h[62] / H_AVG[62]
            - 7.154384e-10 * h[95] / H_AVG[95]
            - 6.630277e-10 * h[96] / H_AVG[96]
            + 3.908853e-10 * h[116] / H_AVG[116]
            + 3.732926e-10 * h[56] / H_AVG[56]
            + 2.595261e-10 * h[85] / H_AVG[85]
            + 1.54477e-10 * h[109] / H_AVG[109]
            - 1.344965e-10 * h[98] / H_AVG[98]
            - 1.074212e-10 * h[88] / H_AVG[88]
            - 7.241889e-11 * h[46] / H_AVG[46]
            - 6.835692e-11 * h[47] / H_AVG[47]
            - 4.215509e-11 * h[64] / H_AVG[64]
            + 1.642662e-11 * h[127] / H_AVG[127]
            + 1.450663e-12 * h[41] / H_AVG[41]
        ),
        -0.4064285 + T[1] * (   # class Hbb
            + 0.2912101 * h[16] / H_AVG[16]
            + 0.2040233 * h[120] / H_AVG[120]
            - 0.0977923 * h[68] / H_AVG[68]
            - 0.07320116 * h[24] / H_AVG[24]
            + 0.06412161 * h[18] / H_AVG[18]
            - 0.05854117 * h[97] / H_AVG[97]
            - 0.05764605 * h[78] / H_AVG[78]
            - 0.05417084 * h[52] / H_AVG[52]
            + 0.0374222 * h[115] / H_AVG[115]
            + 0.02299294 * h[81] / H_AVG[81]
            - 0.01011501 * h[70] / H_AVG[70]
            + 0.007731729 * h[83] / H_AVG[83]
            - 0.007449153 * h[25] / H_AVG[25]
            - 0.005171539 * h[104] / H_AVG[104]
            - 0.005038484 * h[90] / H_AVG[90]
            + 0.002917625 * h[124] / H_AVG[124]
            + 0.0002977082 * h[123] / H_AVG[123]
            + 9.973934e-05 * h[0] / H_AVG[0]
            - 2.878784e-05 * h[99] / H_AVG[99]
            - 2.440986e-05 * h[21] / H_AVG[21]
            + 1.342021e-06 * h[84] / H_AVG[84]
            + 3.16682e-07 * h[55] / H_AVG[55]
            + 2.12477e-07 * h[79] / H_AVG[79]
            - 2.029116e-07 * h[105] / H_AVG[105]
            - 1.6464e-07 * h[106] / H_AVG[106]
            + 1.33788e-07 * h[113] / H_AVG[113]
            - 1.271012e-07 * h[34] / H_AVG[34]
            + 1.141047e-07 * h[73] / H_AVG[73]
            + 1.060077e-07 * h[10] / H_AVG[10]
            + 1.0105e-07 * h[11] / H_AVG[11]
            - 8.604865e-08 * h[6] / H_AVG[6]
            + 8.207749e-08 * h[86] / H_AVG[86]
            - 7.80423e-08 * h[32] / H_AVG[32]
            + 7.174067e-08 * h[17] / H_AVG[17]
            + 6.486062e-08 * h[38] / H_AVG[38]
            - 5.325373e-08 * h[42] / H_AVG[42]
            - 4.745212e-08 * h[54] / H_AVG[54]
            - 4.509943e-08 * h[29] / H_AVG[29]
            + 4.089086e-08 * h[118] / H_AVG[118]
            + 3.192079e-08 * h[48] / H_AVG[48]
            - 3.045695e-08 * h[45] / H_AVG[45]
            - 2.942366e-08 * h[39] / H_AVG[39]
            + 2.713046e-08 * h[27] / H_AVG[27]
            + 2.701357e-08 * h[63] / H_AVG[63]
            + 2.672344e-08 * h[30] / H_AVG[30]
            + 2.617263e-08 * h[43] / H_AVG[43]
            + 2.434705e-08 * h[121] / H_AVG[121]
            - 2.390755e-08 * h[80] / H_AVG[80]
            - 2.188422e-08 * h[126] / H_AVG[126]
            + 2.113168e-08 * h[60] / H_AVG[60]
            - 2.081072e-08 * h[61] / H_AVG[61]
            - 1.975994e-08 * h[102] / H_AVG[102]
            - 1.971487e-08 * h[107] / H_AVG[107]
            - 1.942485e-08 * h[7] / H_AVG[7]
            + 1.749772e-08 * h[92] / H_AVG[92]
            + 1.711142e-08 * h[35] / H_AVG[35]
            + 1.663874e-08 * h[15] / H_AVG[15]
            - 1.607177e-08 * h[59] / H_AVG[59]
            - 1.550507e-08 * h[101] / H_AVG[101]
            - 1.481761e-08 * h[28] / H_AVG[28]
            + 1.437124e-08 * h[103] / H_AVG[103]
            + 1.352467e-08 * h[69] / H_AVG[69]
            + 1.312655e-08 * h[67] / H_AVG[67]
            + 1.271787e-08 * h[1] / H_AVG[1]
            + 1.26253e-08 * h[100] / H_AVG[100]
            - 1.159579e-08 * h[110] / H_AVG[110]
            - 1.121889e-08 * h[3] / H_AVG[3]
            + 1.077394e-08 * h[37] / H_AVG[37]
            - 1.054774e-08 * h[50] / H_AVG[50]
            + 9.923761e-09 * h[44] / H_AVG[44]
            + 9.915026e-09 * h[36] / H_AVG[36]
            - 9.742844e-09 * h[12] / H_AVG[12]
            - 9.266952e-09 * h[114] / H_AVG[114]
            - 9.185559e-09 * h[2] / H_AVG[2]
            + 8.859284e-09 * h[58] / H_AVG[58]
            - 8.820687e-09 * h[112] / H_AVG[112]
            + 8.633553e-09 * h[76] / H_AVG[76]
            - 8.323635e-09 * h[13] / H_AVG[13]
            - 8.091043e-09 * h[19] / H_AVG[19]
            - 7.933717e-09 * h[111] / H_AVG[111]
            + 7.561317e-09 * h[74] / H_AVG[74]
            + 7.17808e-09 * h[108] / H_AVG[108]
            + 6.745042e-09 * h[5] / H_AVG[5]
            - 6.4469e-09 * h[8] / H_AVG[8]
            - 6.181964e-09 * h[91] / H_AVG[91]
            - 5.763569e-09 * h[4] / H_AVG[4]
            - 5.520772e-09 * h[117] / H_AVG[117]
            + 4.66265e-09 * h[53] / H_AVG[53]
            - 4.61011e-09 * h[66] / H_AVG[66]
            - 4.199808e-09 * h[33] / H_AVG[33]
            - 4.078838e-09 * h[65] / H_AVG[65]
            + 3.772065e-09 * h[82] / H_AVG[82]
            - 3.546129e-09 * h[31] / H_AVG[31]
            - 3.256228e-09 * h[71] / H_AVG[71]
            + 3.024766e-09 * h[94] / H_AVG[94]
            + 2.921939e-09 * h[9] / H_AVG[9]
            + 2.566993e-09 * h[26] / H_AVG[26]
            - 2.301742e-09 * h[14] / H_AVG[14]
            + 2.280058e-09 * h[57] / H_AVG[57]
            + 2.155397e-09 * h[49] / H_AVG[49]
            + 2.106412e-09 * h[77] / H_AVG[77]
            - 2.101607e-09 * h[22] / H_AVG[22]
            + 1.948946e-09 * h[122] / H_AVG[122]
            + 1.575869e-09 * h[72] / H_AVG[72]
            + 1.509142e-09 * h[51] / H_AVG[51]
            - 1.485835e-09 * h[23] / H_AVG[23]
            - 1.026319e-09 * h[89] / H_AVG[89]
            + 1.003035e-09 * h[125] / H_AVG[125]
            - 9.409811e-10 * h[75] / H_AVG[75]
            - 8.141483e-10 * h[87] / H_AVG[87]
            + 8.113661e-10 * h[93] / H_AVG[93]
            - 7.300298e-10 * h[40] / H_AVG[40]
            - 6.785583e-10 * h[20] / H_AVG[20]
            + 6.055088e-10 * h[119] / H_AVG[119]
            - 5.545424e-10 * h[62] / H_AVG[62]
            - 4.984963e-10 * h[95] / H_AVG[95]
            - 4.61912e-10 * h[96] / H_AVG[96]
            + 2.724481e-10 * h[116] / H_AVG[116]
            + 2.600357e-10 * h[56] / H_AVG[56]
            + 1.807404e-10 * h[85] / H_AVG[85]
            + 1.076703e-10 * h[109] / H_AVG[109]
            - 9.373963e-11 * h[98] / H_AVG[98]
            - 7.490166e-11 * h[88] / H_AVG[88]
            - 5.043787e-11 * h[46] / H_AVG[46]
            - 4.764963e-11 * h[47] / H_AVG[47]
            - 2.587135e-11 * h[64] / H_AVG[64]
            + 1.141398e-11 * h[127] / H_AVG[127]
            + 1.009055e-12 * h[41] / H_AVG[41]
        ),
        0.1415668 + T[2] * (   # class Hcc
            - 0.208841 * h[83] / H_AVG[83]
            + 0.1570419 * h[18] / H_AVG[18]
            - 0.1500231 * h[24] / H_AVG[24]
            - 0.07920443 * h[81] / H_AVG[81]
            + 0.06528993 * h[124] / H_AVG[124]
            + 0.06092035 * h[104] / H_AVG[104]
            - 0.06043846 * h[16] / H_AVG[16]
            - 0.04804863 * h[97] / H_AVG[97]
            + 0.04311059 * h[78] / H_AVG[78]
            + 0.03467517 * h[120] / H_AVG[120]
            - 0.02547188 * h[68] / H_AVG[68]
            - 0.02166964 * h[52] / H_AVG[52]
            + 0.01942958 * h[115] / H_AVG[115]
            + 0.01343451 * h[70] / H_AVG[70]
            - 0.006991119 * h[25] / H_AVG[25]
            - 0.004914559 * h[123] / H_AVG[123]
            + 0.000219046 * h[90] / H_AVG[90]
            + 0.0002028343 * h[0] / H_AVG[0]
            - 4.042266e-05 * h[99] / H_AVG[99]
            - 2.69274e-05 * h[21] / H_AVG[21]
            + 1.899689e-06 * h[84] / H_AVG[84]
            + 4.483687e-07 * h[55] / H_AVG[55]
            + 3.008526e-07 * h[79] / H_AVG[79]
            - 2.871569e-07 * h[105] / H_AVG[105]
            - 2.330954e-07 * h[106] / H_AVG[106]
            + 1.894373e-07 * h[113] / H_AVG[113]
            - 1.798727e-07 * h[34] / H_AVG[34]
            + 1.615268e-07 * h[73] / H_AVG[73]
            + 1.500703e-07 * h[10] / H_AVG[10]
            + 1.430009e-07 * h[11] / H_AVG[11]
            - 1.217662e-07 * h[6] / H_AVG[6]
            + 1.161688e-07 * h[86] / H_AVG[86]
            - 1.104896e-07 * h[32] / H_AVG[32]
            + 1.015906e-07 * h[17] / H_AVG[17]
            + 9.175863e-08 * h[38] / H_AVG[38]
            - 7.536323e-08 * h[42] / H_AVG[42]
            - 6.717658e-08 * h[54] / H_AVG[54]
            - 6.385339e-08 * h[29] / H_AVG[29]
            + 5.78854e-08 * h[118] / H_AVG[118]
            + 4.518088e-08 * h[48] / H_AVG[48]
            - 4.311331e-08 * h[45] / H_AVG[45]
            - 4.16483e-08 * h[39] / H_AVG[39]
            + 3.840454e-08 * h[27] / H_AVG[27]
            + 3.824383e-08 * h[63] / H_AVG[63]
            + 3.782206e-08 * h[30] / H_AVG[30]
            + 3.703397e-08 * h[43] / H_AVG[43]
            + 3.447249e-08 * h[121] / H_AVG[121]
            - 3.384937e-08 * h[80] / H_AVG[80]
            - 3.097271e-08 * h[126] / H_AVG[126]
            + 2.991115e-08 * h[60] / H_AVG[60]
            - 2.945409e-08 * h[61] / H_AVG[61]
            - 2.796557e-08 * h[102] / H_AVG[102]
            - 2.789596e-08 * h[107] / H_AVG[107]
            - 2.750838e-08 * h[7] / H_AVG[7]
            + 2.476451e-08 * h[92] / H_AVG[92]
            + 2.422179e-08 * h[35] / H_AVG[35]
            + 2.355395e-08 * h[15] / H_AVG[15]
            - 2.274749e-08 * h[59] / H_AVG[59]
            - 2.193614e-08 * h[101] / H_AVG[101]
            - 2.097816e-08 * h[28] / H_AVG[28]
            + 2.034465e-08 * h[103] / H_AVG[103]
            + 1.914664e-08 * h[69] / H_AVG[69]
            + 1.858424e-08 * h[67] / H_AVG[67]
            + 1.800513e-08 * h[1] / H_AVG[1]
            + 1.787057e-08 * h[100] / H_AVG[100]
            - 1.641371e-08 * h[110] / H_AVG[110]
            - 1.588063e-08 * h[3] / H_AVG[3]
            + 1.524965e-08 * h[37] / H_AVG[37]
            - 1.493576e-08 * h[50] / H_AVG[50]
            + 1.404644e-08 * h[44] / H_AVG[44]
            + 1.40368e-08 * h[36] / H_AVG[36]
            - 1.379023e-08 * h[12] / H_AVG[12]
            - 1.311916e-08 * h[114] / H_AVG[114]
            - 1.300945e-08 * h[2] / H_AVG[2]
            + 1.254294e-08 * h[58] / H_AVG[58]
            - 1.249041e-08 * h[112] / H_AVG[112]
            + 1.222277e-08 * h[76] / H_AVG[76]
            - 1.178162e-08 * h[13] / H_AVG[13]
            - 1.14477e-08 * h[19] / H_AVG[19]
            - 1.12273e-08 * h[111] / H_AVG[111]
            + 1.070263e-08 * h[74] / H_AVG[74]
            + 1.015761e-08 * h[108] / H_AVG[108]
            + 9.551875e-09 * h[5] / H_AVG[5]
            - 9.127862e-09 * h[8] / H_AVG[8]
            - 8.753749e-09 * h[91] / H_AVG[91]
            - 8.157725e-09 * h[4] / H_AVG[4]
            - 7.814119e-09 * h[117] / H_AVG[117]
            + 6.604738e-09 * h[53] / H_AVG[53]
            - 6.530569e-09 * h[66] / H_AVG[66]
            - 5.943351e-09 * h[33] / H_AVG[33]
            - 5.775863e-09 * h[65] / H_AVG[65]
            + 5.340639e-09 * h[82] / H_AVG[82]
            - 5.015453e-09 * h[31] / H_AVG[31]
            - 4.608971e-09 * h[71] / H_AVG[71]
            + 4.283331e-09 * h[94] / H_AVG[94]
            + 4.135706e-09 * h[9] / H_AVG[9]
            + 3.631576e-09 * h[26] / H_AVG[26]
            - 3.259856e-09 * h[14] / H_AVG[14]
            + 3.22865e-09 * h[57] / H_AVG[57]
            + 3.050005e-09 * h[49] / H_AVG[49]
            + 2.979694e-09 * h[77] / H_AVG[77]
            - 2.975854e-09 * h[22] / H_AVG[22]
            + 2.760047e-09 * h[122] / H_AVG[122]
            + 2.229653e-09 * h[72] / H_AVG[72]
            + 2.137506e-09 * h[51] / H_AVG[51]
            - 2.104079e-09 * h[23] / H_AVG[23]
            - 1.452165e-09 * h[89] / H_AVG[89]
            + 1.419089e-09 * h[125] / H_AVG[125]
            - 1.332082e-09 * h[75] / H_AVG[75]
            - 1.152596e-09 * h[87] / H_AVG[87]
            + 1.148397e-09 * h[93] / H_AVG[93]
            - 1.033197e-09 * h[40] / H_AVG[40]
            - 9.607077e-10 * h[20] / H_AVG[20]
            + 8.577916e-10 * h[119] / H_AVG[119]
            - 7.848907e-10 * h[62] / H_AVG[62]
            - 7.05441e-10 * h[95] / H_AVG[95]
            - 6.537413e-10 * h[96] / H_AVG[96]
            + 3.857624e-10 * h[116] / H_AVG[116]
            + 3.683119e-10 * h[56] / H_AVG[56]
            + 2.561324e-10 * h[85] / H_AVG[85]
            + 1.523529e-10 * h[109] / H_AVG[109]
            - 1.326625e-10 * h[98] / H_AVG[98]
            - 1.058793e-10 * h[88] / H_AVG[88]
            - 7.142572e-11 * h[46] / H_AVG[46]
            - 6.739793e-11 * h[47] / H_AVG[47]
            - 4.255656e-11 * h[64] / H_AVG[64]
            + 1.619399e-11 * h[127] / H_AVG[127]
            + 1.432403e-12 * h[41] / H_AVG[41]
        ),
        0.1286925 + T[3] * (   # class Hgg
            - 0.243305 * h[104] / H_AVG[104]
            - 0.162841 * h[68] / H_AVG[68]
            - 0.1000287 * h[16] / H_AVG[16]
            - 0.08343364 * h[18] / H_AVG[18]
            + 0.07649792 * h[52] / H_AVG[52]
            - 0.06925706 * h[97] / H_AVG[97]
            + 0.05677391 * h[124] / H_AVG[124]
            - 0.05079353 * h[24] / H_AVG[24]
            - 0.04795251 * h[81] / H_AVG[81]
            + 0.04603507 * h[115] / H_AVG[115]
            - 0.01752647 * h[78] / H_AVG[78]
            + 0.01616768 * h[83] / H_AVG[83]
            + 0.01229776 * h[120] / H_AVG[120]
            + 0.007718348 * h[70] / H_AVG[70]
            - 0.004297931 * h[25] / H_AVG[25]
            + 0.003967788 * h[90] / H_AVG[90]
            - 0.0008204931 * h[123] / H_AVG[123]
            + 0.0002189037 * h[0] / H_AVG[0]
            - 3.548413e-05 * h[99] / H_AVG[99]
            - 2.553747e-05 * h[21] / H_AVG[21]
            + 1.676868e-06 * h[84] / H_AVG[84]
            + 3.956703e-07 * h[55] / H_AVG[55]
            + 2.655347e-07 * h[79] / H_AVG[79]
            - 2.535026e-07 * h[105] / H_AVG[105]
            - 2.057123e-07 * h[106] / H_AVG[106]
            + 1.672169e-07 * h[113] / H_AVG[113]
            - 1.588103e-07 * h[34] / H_AVG[34]
            + 1.425828e-07 * h[73] / H_AVG[73]
            + 1.32425e-07 * h[10] / H_AVG[10]
            + 1.261967e-07 * h[11] / H_AVG[11]
            - 1.074819e-07 * h[6] / H_AVG[6]
            + 1.025539e-07 * h[86] / H_AVG[86]
            - 9.754933e-08 * h[32] / H_AVG[32]
            + 8.966988e-08 * h[17] / H_AVG[17]
            + 8.103871e-08 * h[38] / H_AVG[38]
            - 6.652219e-08 * h[42] / H_AVG[42]
            - 5.929056e-08 * h[54] / H_AVG[54]
            - 5.6348e-08 * h[29] / H_AVG[29]
            + 5.110084e-08 * h[118] / H_AVG[118]
            + 3.988355e-08 * h[48] / H_AVG[48]
            - 3.806159e-08 * h[45] / H_AVG[45]
            - 3.675617e-08 * h[39] / H_AVG[39]
            + 3.389599e-08 * h[27] / H_AVG[27]
            + 3.374782e-08 * h[63] / H_AVG[63]
            + 3.338448e-08 * h[30] / H_AVG[30]
            + 3.269548e-08 * h[43] / H_AVG[43]
            + 3.042176e-08 * h[121] / H_AVG[121]
            - 2.987262e-08 * h[80] / H_AVG[80]
            - 2.734349e-08 * h[126] / H_AVG[126]
            + 2.639706e-08 * h[60] / H_AVG[60]
            - 2.600403e-08 * h[61] / H_AVG[61]
            - 2.468706e-08 * h[102] / H_AVG[102]
            - 2.464111e-08 * h[107] / H_AVG[107]
            - 2.426669e-08 * h[7] / H_AVG[7]
            + 2.185835e-08 * h[92] / H_AVG[92]
            + 2.137922e-08 * h[35] / H_AVG[35]
            + 2.078628e-08 * h[15] / H_AVG[15]
            - 2.008086e-08 * h[59] / H_AVG[59]
            - 1.936639e-08 * h[101] / H_AVG[101]
            - 1.851677e-08 * h[28] / H_AVG[28]
            + 1.795676e-08 * h[103] / H_AVG[103]
            + 1.68988e-08 * h[69] / H_AVG[69]
            + 1.639887e-08 * h[67] / H_AVG[67]
            + 1.588797e-08 * h[1] / H_AVG[1]
            + 1.576979e-08 * h[100] / H_AVG[100]
            - 1.448321e-08 * h[110] / H_AVG[110]
            - 1.401528e-08 * h[3] / H_AVG[3]
            + 1.346075e-08 * h[37] / H_AVG[37]
            - 1.318931e-08 * h[50] / H_AVG[50]
            + 1.239687e-08 * h[44] / H_AVG[44]
            + 1.238613e-08 * h[36] / H_AVG[36]
            - 1.217624e-08 * h[12] / H_AVG[12]
            - 1.158118e-08 * h[114] / H_AVG[114]
            - 1.148199e-08 * h[2] / H_AVG[2]
            + 1.107178e-08 * h[58] / H_AVG[58]
            - 1.102663e-08 * h[112] / H_AVG[112]
            + 1.079095e-08 * h[76] / H_AVG[76]
            - 1.039912e-08 * h[13] / H_AVG[13]
            - 1.01057e-08 * h[19] / H_AVG[19]
            - 9.911801e-09 * h[111] / H_AVG[111]
            + 9.447018e-09 * h[74] / H_AVG[74]
            + 8.965151e-09 * h[108] / H_AVG[108]
            + 8.430243e-09 * h[5] / H_AVG[5]
            - 8.056961e-09 * h[8] / H_AVG[8]
            - 7.722685e-09 * h[91] / H_AVG[91]
            - 7.201094e-09 * h[4] / H_AVG[4]
            - 6.896956e-09 * h[117] / H_AVG[117]
            + 5.82687e-09 * h[53] / H_AVG[53]
            - 5.762586e-09 * h[66] / H_AVG[66]
            - 5.246498e-09 * h[33] / H_AVG[33]
            - 5.090702e-09 * h[65] / H_AVG[65]
            + 4.712809e-09 * h[82] / H_AVG[82]
            - 4.428093e-09 * h[31] / H_AVG[31]
            - 4.069862e-09 * h[71] / H_AVG[71]
            + 3.779528e-09 * h[94] / H_AVG[94]
            + 3.650896e-09 * h[9] / H_AVG[9]
            + 3.207461e-09 * h[26] / H_AVG[26]
            - 2.875826e-09 * h[14] / H_AVG[14]
            + 2.849589e-09 * h[57] / H_AVG[57]
            + 2.692585e-09 * h[49] / H_AVG[49]
            + 2.631088e-09 * h[77] / H_AVG[77]
            - 2.624708e-09 * h[22] / H_AVG[22]
            + 2.434962e-09 * h[122] / H_AVG[122]
            + 1.968906e-09 * h[72] / H_AVG[72]
            + 1.88682e-09 * h[51] / H_AVG[51]
            - 1.85624e-09 * h[23] / H_AVG[23]
            - 1.282056e-09 * h[89] / H_AVG[89]
            + 1.252524e-09 * h[125] / H_AVG[125]
            - 1.17602e-09 * h[75] / H_AVG[75]
            - 1.017198e-09 * h[87] / H_AVG[87]
            + 1.01369e-09 * h[93] / H_AVG[93]
            - 9.120827e-10 * h[40] / H_AVG[40]
            - 8.479058e-10 * h[20] / H_AVG[20]
            + 7.570229e-10 * h[119] / H_AVG[119]
            - 6.927384e-10 * h[62] / H_AVG[62]
            - 6.227213e-10 * h[95] / H_AVG[95]
            - 5.771272e-10 * h[96] / H_AVG[96]
            + 3.403357e-10 * h[116] / H_AVG[116]
            + 3.248893e-10 * h[56] / H_AVG[56]
            + 2.260535e-10 * h[85] / H_AVG[85]
            + 1.344987e-10 * h[109] / H_AVG[109]
            - 1.170925e-10 * h[98] / H_AVG[98]
            - 9.360819e-11 * h[88] / H_AVG[88]
            - 6.303602e-11 * h[46] / H_AVG[46]
            - 5.950676e-11 * h[47] / H_AVG[47]
            - 3.601984e-11 * h[64] / H_AVG[64]
            + 1.429588e-11 * h[127] / H_AVG[127]
            + 1.26254e-12 * h[41] / H_AVG[41]
        ),
        -0.07702489 + T[4] * (   # class H4q
            - 0.2729641 * h[115] / H_AVG[115]
            - 0.1163176 * h[104] / H_AVG[104]
            - 0.09161914 * h[16] / H_AVG[16]
            - 0.08129721 * h[24] / H_AVG[24]
            + 0.07494726 * h[52] / H_AVG[52]
            - 0.068192 * h[97] / H_AVG[97]
            - 0.06404141 * h[78] / H_AVG[78]
            + 0.05449608 * h[124] / H_AVG[124]
            + 0.04951943 * h[68] / H_AVG[68]
            - 0.04873106 * h[81] / H_AVG[81]
            - 0.03859831 * h[18] / H_AVG[18]
            + 0.01317682 * h[120] / H_AVG[120]
            - 0.01126612 * h[83] / H_AVG[83]
            + 0.009265605 * h[70] / H_AVG[70]
            - 0.004233554 * h[25] / H_AVG[25]
            - 0.0007795787 * h[90] / H_AVG[90]
            + 0.0002605482 * h[123] / H_AVG[123]
            + 0.0002288482 * h[0] / H_AVG[0]
            - 3.342108e-05 * h[99] / H_AVG[99]
            - 2.698045e-05 * h[21] / H_AVG[21]
            + 1.581612e-06 * h[84] / H_AVG[84]
            + 3.733622e-07 * h[55] / H_AVG[55]
            + 2.50492e-07 * h[79] / H_AVG[79]
            - 2.391335e-07 * h[105] / H_AVG[105]
            - 1.941218e-07 * h[106] / H_AVG[106]
            + 1.577009e-07 * h[113] / H_AVG[113]
            - 1.498094e-07 * h[34] / H_AVG[34]
            + 1.345078e-07 * h[73] / H_AVG[73]
            + 1.249376e-07 * h[10] / H_AVG[10]
            + 1.190379e-07 * h[11] / H_AVG[11]
            - 1.01403e-07 * h[6] / H_AVG[6]
            + 9.674113e-08 * h[86] / H_AVG[86]
            - 9.200728e-08 * h[32] / H_AVG[32]
            + 8.45943e-08 * h[17] / H_AVG[17]
            + 7.644628e-08 * h[38] / H_AVG[38]
            - 6.275585e-08 * h[42] / H_AVG[42]
            - 5.594243e-08 * h[54] / H_AVG[54]
            - 5.316774e-08 * h[29] / H_AVG[29]
            + 4.820838e-08 * h[118] / H_AVG[118]
            + 3.761994e-08 * h[48] / H_AVG[48]
            - 3.590409e-08 * h[45] / H_AVG[45]
            - 3.467419e-08 * h[39] / H_AVG[39]
            + 3.197673e-08 * h[27] / H_AVG[27]
            + 3.183543e-08 * h[63] / H_AVG[63]
            + 3.149603e-08 * h[30] / H_AVG[30]
            + 3.084354e-08 * h[43] / H_AVG[43]
            + 2.869794e-08 * h[121] / H_AVG[121]
            - 2.818419e-08 * h[80] / H_AVG[80]
            - 2.579773e-08 * h[126] / H_AVG[126]
            + 2.491102e-08 * h[60] / H_AVG[60]
            - 2.452446e-08 * h[61] / H_AVG[61]
            - 2.328683e-08 * h[102] / H_AVG[102]
            - 2.324562e-08 * h[107] / H_AVG[107]
            - 2.289465e-08 * h[7] / H_AVG[7]
            + 2.06187e-08 * h[92] / H_AVG[92]
            + 2.017052e-08 * h[35] / H_AVG[35]
            + 1.961159e-08 * h[15] / H_AVG[15]
            - 1.893715e-08 * h[59] / H_AVG[59]
            - 1.826938e-08 * h[101] / H_AVG[101]
            - 1.746905e-08 * h[28] / H_AVG[28]
            + 1.693893e-08 * h[103] / H_AVG[103]
            + 1.594074e-08 * h[69] / H_AVG[69]
            + 1.547311e-08 * h[67] / H_AVG[67]
            + 1.498791e-08 * h[1] / H_AVG[1]
            + 1.487568e-08 * h[100] / H_AVG[100]
            - 1.366547e-08 * h[110] / H_AVG[110]
            - 1.321896e-08 * h[3] / H_AVG[3]
            + 1.269836e-08 * h[37] / H_AVG[37]
            - 1.243236e-08 * h[50] / H_AVG[50]
            + 1.169434e-08 * h[44] / H_AVG[44]
            + 1.168428e-08 * h[36] / H_AVG[36]
            - 1.148348e-08 * h[12] / H_AVG[12]
            - 1.092628e-08 * h[114] / H_AVG[114]
            - 1.083369e-08 * h[2] / H_AVG[2]
            + 1.044374e-08 * h[58] / H_AVG[58]
            - 1.040007e-08 * h[112] / H_AVG[112]
            + 1.017908e-08 * h[76] / H_AVG[76]
            - 9.808784e-09 * h[13] / H_AVG[13]
            - 9.530001e-09 * h[19] / H_AVG[19]
            - 9.34971e-09 * h[111] / H_AVG[111]
            + 8.910742e-09 * h[74] / H_AVG[74]
            + 8.456021e-09 * h[108] / H_AVG[108]
            + 7.953273e-09 * h[5] / H_AVG[5]
            - 7.600685e-09 * h[8] / H_AVG[8]
            - 7.289209e-09 * h[91] / H_AVG[91]
            - 6.792778e-09 * h[4] / H_AVG[4]
            - 6.507382e-09 * h[117] / H_AVG[117]
            + 5.496034e-09 * h[53] / H_AVG[53]
            - 5.43635e-09 * h[66] / H_AVG[66]
            - 4.948469e-09 * h[33] / H_AVG[33]
            - 4.803351e-09 * h[65] / H_AVG[65]
            + 4.447147e-09 * h[82] / H_AVG[82]
            - 4.177477e-09 * h[31] / H_AVG[31]
            - 3.839222e-09 * h[71] / H_AVG[71]
            + 3.565809e-09 * h[94] / H_AVG[94]
            + 3.443755e-09 * h[9] / H_AVG[9]
            + 3.025658e-09 * h[26] / H_AVG[26]
            - 2.712483e-09 * h[14] / H_AVG[14]
            + 2.688079e-09 * h[57] / H_AVG[57]
            + 2.539971e-09 * h[49] / H_AVG[49]
            + 2.484041e-09 * h[77] / H_AVG[77]
            - 2.477727e-09 * h[22] / H_AVG[22]
            + 2.29752e-09 * h[122] / H_AVG[122]
            + 1.857343e-09 * h[72] / H_AVG[72]
            + 1.779599e-09 * h[51] / H_AVG[51]
            - 1.750841e-09 * h[23] / H_AVG[23]
            - 1.20932e-09 * h[89] / H_AVG[89]
            + 1.181787e-09 * h[125] / H_AVG[125]
            - 1.109323e-09 * h[75] / H_AVG[75]
            - 9.59684e-10 * h[87] / H_AVG[87]
            + 9.568314e-10 * h[93] / H_AVG[93]
            - 8.604492e-10 * h[40] / H_AVG[40]
            - 7.998139e-10 * h[20] / H_AVG[20]
            + 7.145195e-10 * h[119] / H_AVG[119]
            - 6.536847e-10 * h[62] / H_AVG[62]
            - 5.876768e-10 * h[95] / H_AVG[95]
            - 5.445512e-10 * h[96] / H_AVG[96]
            + 3.211122e-10 * h[116] / H_AVG[116]
            + 3.067071e-10 * h[56] / H_AVG[56]
            + 2.131402e-10 * h[85] / H_AVG[85]
            + 1.269129e-10 * h[109] / H_AVG[109]
            - 1.104631e-10 * h[98] / H_AVG[98]
            - 8.825713e-11 * h[88] / H_AVG[88]
            - 5.947356e-11 * h[46] / H_AVG[46]
            - 5.614131e-11 * h[47] / H_AVG[47]
            - 3.46929e-11 * h[64] / H_AVG[64]
            + 1.34695e-11 * h[127] / H_AVG[127]
            + 1.192149e-12 * h[41] / H_AVG[41]
        ),
        -0.9789527 + T[5] * (   # class Hqql
            + 0.1862117 * h[120] / H_AVG[120]
            + 0.143305 * h[24] / H_AVG[24]
            + 0.1162688 * h[97] / H_AVG[97]
            + 0.08749075 * h[18] / H_AVG[18]
            - 0.08050761 * h[124] / H_AVG[124]
            + 0.07964369 * h[52] / H_AVG[52]
            + 0.07356867 * h[83] / H_AVG[83]
            - 0.06378592 * h[115] / H_AVG[115]
            + 0.05128744 * h[78] / H_AVG[78]
            + 0.02173394 * h[70] / H_AVG[70]
            + 0.0182797 * h[90] / H_AVG[90]
            + 0.0167395 * h[68] / H_AVG[68]
            + 0.01673325 * h[104] / H_AVG[104]
            - 0.01637284 * h[81] / H_AVG[81]
            + 0.01179368 * h[25] / H_AVG[25]
            - 0.01054948 * h[123] / H_AVG[123]
            - 0.005320471 * h[16] / H_AVG[16]
            - 0.0003193394 * h[0] / H_AVG[0]
            + 6.653937e-05 * h[21] / H_AVG[21]
            - 1.89403e-05 * h[99] / H_AVG[99]
            + 8.849405e-07 * h[84] / H_AVG[84]
            + 2.087892e-07 * h[55] / H_AVG[55]
            + 1.400464e-07 * h[79] / H_AVG[79]
            - 1.337187e-07 * h[105] / H_AVG[105]
            - 1.085738e-07 * h[106] / H_AVG[106]
            + 8.825016e-08 * h[113] / H_AVG[113]
            - 8.378918e-08 * h[34] / H_AVG[34]
            + 7.520769e-08 * h[73] / H_AVG[73]
            + 6.983333e-08 * h[10] / H_AVG[10]
            + 6.662909e-08 * h[11] / H_AVG[11]
            - 5.666605e-08 * h[6] / H_AVG[6]
            + 5.408757e-08 * h[86] / H_AVG[86]
            - 5.14476e-08 * h[32] / H_AVG[32]
            + 4.731526e-08 * h[17] / H_AVG[17]
            + 4.272795e-08 * h[38] / H_AVG[38]
            - 3.5085e-08 * h[42] / H_AVG[42]
            - 3.130393e-08 * h[54] / H_AVG[54]
            - 2.972507e-08 * h[29] / H_AVG[29]
            + 2.696592e-08 * h[118] / H_AVG[118]
            + 2.1027e-08 * h[48] / H_AVG[48]
            - 2.008128e-08 * h[45] / H_AVG[45]
            - 1.937889e-08 * h[39] / H_AVG[39]
            + 1.788472e-08 * h[27] / H_AVG[27]
            + 1.780578e-08 * h[63] / H_AVG[63]
            + 1.761607e-08 * h[30] / H_AVG[30]
            + 1.723629e-08 * h[43] / H_AVG[43]
            + 1.604442e-08 * h[121] / H_AVG[121]
            - 1.576053e-08 * h[80] / H_AVG[80]
            - 1.444549e-08 * h[126] / H_AVG[126]
            + 1.392389e-08 * h[60] / H_AVG[60]
            - 1.370733e-08 * h[61] / H_AVG[61]
            - 1.301786e-08 * h[102] / H_AVG[102]
            - 1.297903e-08 * h[107] / H_AVG[107]
            - 1.280486e-08 * h[7] / H_AVG[7]
            + 1.152463e-08 * h[92] / H_AVG[92]
            + 1.12684e-08 * h[35] / H_AVG[35]
            + 1.096476e-08 * h[15] / H_AVG[15]
            - 1.060055e-08 * h[59] / H_AVG[59]
            - 1.022516e-08 * h[101] / H_AVG[101]
            - 9.74908e-09 * h[28] / H_AVG[28]
            + 9.476666e-09 * h[103] / H_AVG[103]
            + 8.925928e-09 * h[69] / H_AVG[69]
            + 8.667857e-09 * h[67] / H_AVG[67]
            + 8.377472e-09 * h[1] / H_AVG[1]
            + 8.313736e-09 * h[100] / H_AVG[100]
            - 7.635663e-09 * h[110] / H_AVG[110]
            - 7.398327e-09 * h[3] / H_AVG[3]
            + 7.102265e-09 * h[37] / H_AVG[37]
            - 6.948657e-09 * h[50] / H_AVG[50]
            + 6.53841e-09 * h[36] / H_AVG[36]
            + 6.535154e-09 * h[44] / H_AVG[44]
            - 6.429838e-09 * h[12] / H_AVG[12]
            - 6.110027e-09 * h[114] / H_AVG[114]
            - 6.064805e-09 * h[2] / H_AVG[2]
            + 5.84583e-09 * h[58] / H_AVG[58]
            - 5.827386e-09 * h[112] / H_AVG[112]
            + 5.696753e-09 * h[76] / H_AVG[76]
            - 5.479521e-09 * h[13] / H_AVG[13]
            - 5.33157e-09 * h[19] / H_AVG[19]
            - 5.224162e-09 * h[111] / H_AVG[111]
            + 4.981915e-09 * h[74] / H_AVG[74]
            + 4.719765e-09 * h[108] / H_AVG[108]
            + 4.443099e-09 * h[5] / H_AVG[5]
            - 4.25421e-09 * h[8] / H_AVG[8]
            - 4.083288e-09 * h[91] / H_AVG[91]
            - 3.80127e-09 * h[4] / H_AVG[4]
            - 3.64031e-09 * h[117] / H_AVG[117]
            + 3.075927e-09 * h[53] / H_AVG[53]
            - 3.039355e-09 * h[66] / H_AVG[66]
            - 2.760091e-09 * h[33] / H_AVG[33]
            - 2.695382e-09 * h[65] / H_AVG[65]
            + 2.491285e-09 * h[82] / H_AVG[82]
            - 2.328908e-09 * h[31] / H_AVG[31]
            - 2.144792e-09 * h[71] / H_AVG[71]
            + 1.99311e-09 * h[94] / H_AVG[94]
            + 1.92556e-09 * h[9] / H_AVG[9]
            + 1.699764e-09 * h[26] / H_AVG[26]
            - 1.518239e-09 * h[14] / H_AVG[14]
            + 1.508301e-09 * h[57] / H_AVG[57]
            + 1.418841e-09 * h[49] / H_AVG[49]
            + 1.388392e-09 * h[77] / H_AVG[77]
            - 1.384677e-09 * h[22] / H_AVG[22]
            + 1.288435e-09 * h[122] / H_AVG[122]
            + 1.037578e-09 * h[72] / H_AVG[72]
            + 9.987658e-10 * h[51] / H_AVG[51]
            - 9.809028e-10 * h[23] / H_AVG[23]
            - 6.726359e-10 * h[89] / H_AVG[89]
            + 6.596329e-10 * h[125] / H_AVG[125]
            - 6.190767e-10 * h[75] / H_AVG[75]
            + 5.352167e-10 * h[93] / H_AVG[93]
            - 5.342229e-10 * h[87] / H_AVG[87]
            - 4.80916e-10 * h[40] / H_AVG[40]
            - 4.471755e-10 * h[20] / H_AVG[20]
            + 4.007711e-10 * h[119] / H_AVG[119]
            - 3.645468e-10 * h[62] / H_AVG[62]
            - 3.284627e-10 * h[95] / H_AVG[95]
            - 3.037583e-10 * h[96] / H_AVG[96]
            + 1.792706e-10 * h[116] / H_AVG[116]
            + 1.720954e-10 * h[56] / H_AVG[56]
            + 1.200066e-10 * h[85] / H_AVG[85]
            + 7.072527e-11 * h[109] / H_AVG[109]
            - 6.159046e-11 * h[98] / H_AVG[98]
            - 4.902833e-11 * h[88] / H_AVG[88]
            - 3.346738e-11 * h[46] / H_AVG[46]
            - 3.131214e-11 * h[47] / H_AVG[47]
            - 1.497961e-11 * h[64] / H_AVG[64]
            + 7.797592e-12 * h[127] / H_AVG[127]
            + 6.936347e-13 * h[41] / H_AVG[41]
        ),
        0.3035426 + T[6] * (   # class Zqq
            + 0.1190547 * h[104] / H_AVG[104]
            + 0.1136618 * h[24] / H_AVG[24]
            - 0.1074233 * h[124] / H_AVG[124]
            + 0.09783696 * h[115] / H_AVG[115]
            - 0.09056984 * h[120] / H_AVG[120]
            - 0.09047722 * h[97] / H_AVG[97]
            + 0.08489026 * h[83] / H_AVG[83]
            + 0.0824024 * h[68] / H_AVG[68]
            + 0.05917879 * h[81] / H_AVG[81]
            - 0.0469356 * h[52] / H_AVG[52]
            - 0.04491699 * h[16] / H_AVG[16]
            - 0.01935416 * h[70] / H_AVG[70]
            + 0.014051 * h[18] / H_AVG[18]
            + 0.008974161 * h[25] / H_AVG[25]
            - 0.008425549 * h[90] / H_AVG[90]
            - 0.007159583 * h[78] / H_AVG[78]
            + 0.004550148 * h[123] / H_AVG[123]
            - 5.768396e-05 * h[0] / H_AVG[0]
            - 4.099422e-05 * h[99] / H_AVG[99]
            - 3.283022e-05 * h[21] / H_AVG[21]
            + 1.930479e-06 * h[84] / H_AVG[84]
            + 4.556982e-07 * h[55] / H_AVG[55]
            + 3.057089e-07 * h[79] / H_AVG[79]
            - 2.918856e-07 * h[105] / H_AVG[105]
            - 2.369255e-07 * h[106] / H_AVG[106]
            + 1.924782e-07 * h[113] / H_AVG[113]
            - 1.828551e-07 * h[34] / H_AVG[34]
            + 1.641692e-07 * h[73] / H_AVG[73]
            + 1.525207e-07 * h[10] / H_AVG[10]
            + 1.453171e-07 * h[11] / H_AVG[11]
            - 1.237762e-07 * h[6] / H_AVG[6]
            + 1.18075e-07 * h[86] / H_AVG[86]
            - 1.122947e-07 * h[32] / H_AVG[32]
            + 1.032133e-07 * h[17] / H_AVG[17]
            + 9.329617e-08 * h[38] / H_AVG[38]
            - 7.659617e-08 * h[42] / H_AVG[42]
            - 6.826803e-08 * h[54] / H_AVG[54]
            - 6.489691e-08 * h[29] / H_AVG[29]
            + 5.883683e-08 * h[118] / H_AVG[118]
            + 4.592109e-08 * h[48] / H_AVG[48]
            - 4.382225e-08 * h[45] / H_AVG[45]
            - 4.233367e-08 * h[39] / H_AVG[39]
            + 3.90304e-08 * h[27] / H_AVG[27]
            + 3.885988e-08 * h[63] / H_AVG[63]
            + 3.843829e-08 * h[30] / H_AVG[30]
            + 3.764588e-08 * h[43] / H_AVG[43]
            + 3.503412e-08 * h[121] / H_AVG[121]
            - 3.439988e-08 * h[80] / H_AVG[80]
            - 3.146771e-08 * h[126] / H_AVG[126]
            + 3.040058e-08 * h[60] / H_AVG[60]
            - 2.994351e-08 * h[61] / H_AVG[61]
            - 2.842905e-08 * h[102] / H_AVG[102]
            - 2.836055e-08 * h[107] / H_AVG[107]
            - 2.795816e-08 * h[7] / H_AVG[7]
            + 2.517169e-08 * h[92] / H_AVG[92]
            + 2.459855e-08 * h[35] / H_AVG[35]
            + 2.393459e-08 * h[15] / H_AVG[15]
            - 2.312312e-08 * h[59] / H_AVG[59]
            - 2.230132e-08 * h[101] / H_AVG[101]
            - 2.133094e-08 * h[28] / H_AVG[28]
            + 2.06752e-08 * h[103] / H_AVG[103]
            + 1.945996e-08 * h[69] / H_AVG[69]
            + 1.888428e-08 * h[67] / H_AVG[67]
            + 1.829638e-08 * h[1] / H_AVG[1]
            + 1.816106e-08 * h[100] / H_AVG[100]
            - 1.66807e-08 * h[110] / H_AVG[110]
            - 1.61329e-08 * h[3] / H_AVG[3]
            + 1.549955e-08 * h[37] / H_AVG[37]
            - 1.51775e-08 * h[50] / H_AVG[50]
            + 1.427424e-08 * h[44] / H_AVG[44]
            + 1.426228e-08 * h[36] / H_AVG[36]
            - 1.402274e-08 * h[12] / H_AVG[12]
            - 1.333446e-08 * h[114] / H_AVG[114]
            - 1.321397e-08 * h[2] / H_AVG[2]
            + 1.274685e-08 * h[58] / H_AVG[58]
            - 1.269536e-08 * h[112] / H_AVG[112]
            + 1.242331e-08 * h[76] / H_AVG[76]
            - 1.197405e-08 * h[13] / H_AVG[13]
            - 1.163608e-08 * h[19] / H_AVG[19]
            - 1.141274e-08 * h[111] / H_AVG[111]
            + 1.087661e-08 * h[74] / H_AVG[74]
            + 1.032285e-08 * h[108] / H_AVG[108]
            + 9.70744e-09 * h[5] / H_AVG[5]
            - 9.276872e-09 * h[8] / H_AVG[8]
            - 8.896409e-09 * h[91] / H_AVG[91]
            - 8.291353e-09 * h[4] / H_AVG[4]
            - 7.941669e-09 * h[117] / H_AVG[117]
            + 6.709033e-09 * h[53] / H_AVG[53]
            - 6.632244e-09 * h[66] / H_AVG[66]
            - 6.039758e-09 * h[33] / H_AVG[33]
            - 5.865292e-09 * h[65] / H_AVG[65]
            + 5.427282e-09 * h[82] / H_AVG[82]
            - 5.098376e-09 * h[31] / H_AVG[31]
            - 4.684921e-09 * h[71] / H_AVG[71]
            + 4.351876e-09 * h[94] / H_AVG[94]
            + 4.203384e-09 * h[9] / H_AVG[9]
            + 3.692678e-09 * h[26] / H_AVG[26]
            - 3.311183e-09 * h[14] / H_AVG[14]
            + 3.27996e-09 * h[57] / H_AVG[57]
            + 3.100105e-09 * h[49] / H_AVG[49]
            + 3.031766e-09 * h[77] / H_AVG[77]
            - 3.023327e-09 * h[22] / H_AVG[22]
            + 2.803799e-09 * h[122] / H_AVG[122]
            + 2.267201e-09 * h[72] / H_AVG[72]
            + 2.170851e-09 * h[51] / H_AVG[51]
            - 2.138474e-09 * h[23] / H_AVG[23]
            - 1.476481e-09 * h[89] / H_AVG[89]
            + 1.442454e-09 * h[125] / H_AVG[125]
            - 1.354116e-09 * h[75] / H_AVG[75]
            - 1.171831e-09 * h[87] / H_AVG[87]
            + 1.167839e-09 * h[93] / H_AVG[93]
            - 1.050197e-09 * h[40] / H_AVG[40]
            - 9.762165e-10 * h[20] / H_AVG[20]
            + 8.719877e-10 * h[119] / H_AVG[119]
            - 7.97769e-10 * h[62] / H_AVG[62]
            - 7.171382e-10 * h[95] / H_AVG[95]
            - 6.646328e-10 * h[96] / H_AVG[96]
            + 3.919807e-10 * h[116] / H_AVG[116]
            + 3.740995e-10 * h[56] / H_AVG[56]
            + 2.600578e-10 * h[85] / H_AVG[85]
            + 1.549173e-10 * h[109] / H_AVG[109]
            - 1.348531e-10 * h[98] / H_AVG[98]
            - 1.077754e-10 * h[88] / H_AVG[88]
            - 7.258531e-11 * h[46] / H_AVG[46]
            - 6.851946e-11 * h[47] / H_AVG[47]
            - 3.896824e-11 * h[64] / H_AVG[64]
            + 1.642186e-11 * h[127] / H_AVG[127]
            + 1.451649e-12 * h[41] / H_AVG[41]
        ),
        0.09887857 + T[7] * (   # class Wqq
            + 0.2543648 * h[97] / H_AVG[97]
            + 0.1110388 * h[24] / H_AVG[24]
            + 0.1047955 * h[68] / H_AVG[68]
            + 0.09863661 * h[83] / H_AVG[83]
            - 0.08104524 * h[120] / H_AVG[120]
            - 0.078265 * h[52] / H_AVG[52]
            + 0.06246051 * h[104] / H_AVG[104]
            + 0.048202 * h[115] / H_AVG[115]
            - 0.04286503 * h[16] / H_AVG[16]
            - 0.03965644 * h[81] / H_AVG[81]
            - 0.03027934 * h[78] / H_AVG[78]
            - 0.01582261 * h[70] / H_AVG[70]
            + 0.0129345 * h[124] / H_AVG[124]
            - 0.01023031 * h[25] / H_AVG[25]
            - 0.006279688 * h[90] / H_AVG[90]
            + 0.001497834 * h[123] / H_AVG[123]
            + 0.001391092 * h[18] / H_AVG[18]
            - 0.0001657629 * h[0] / H_AVG[0]
            - 3.739904e-05 * h[99] / H_AVG[99]
            - 2.607392e-05 * h[21] / H_AVG[21]
            + 1.758918e-06 * h[84] / H_AVG[84]
            + 4.150729e-07 * h[55] / H_AVG[55]
            + 2.785286e-07 * h[79] / H_AVG[79]
            - 2.659282e-07 * h[105] / H_AVG[105]
            - 2.158386e-07 * h[106] / H_AVG[106]
            + 1.753868e-07 * h[113] / H_AVG[113]
            - 1.665885e-07 * h[34] / H_AVG[34]
            + 1.495817e-07 * h[73] / H_AVG[73]
            + 1.389089e-07 * h[10] / H_AVG[10]
            + 1.323928e-07 * h[11] / H_AVG[11]
            - 1.127717e-07 * h[6] / H_AVG[6]
            + 1.075865e-07 * h[86] / H_AVG[86]
            - 1.023055e-07 * h[32] / H_AVG[32]
            + 9.402211e-08 * h[17] / H_AVG[17]
            + 8.50006e-08 * h[38] / H_AVG[38]
            - 6.979067e-08 * h[42] / H_AVG[42]
            - 6.220579e-08 * h[54] / H_AVG[54]
            - 5.911994e-08 * h[29] / H_AVG[29]
            + 5.360219e-08 * h[118] / H_AVG[118]
            + 4.183895e-08 * h[48] / H_AVG[48]
            - 3.992645e-08 * h[45] / H_AVG[45]
            - 3.855853e-08 * h[39] / H_AVG[39]
            + 3.55603e-08 * h[27] / H_AVG[27]
            + 3.540144e-08 * h[63] / H_AVG[63]
            + 3.502223e-08 * h[30] / H_AVG[30]
            + 3.429627e-08 * h[43] / H_AVG[43]
            + 3.1915e-08 * h[121] / H_AVG[121]
            - 3.133761e-08 * h[80] / H_AVG[80]
            - 2.868336e-08 * h[126] / H_AVG[126]
            + 2.768928e-08 * h[60] / H_AVG[60]
            - 2.727227e-08 * h[61] / H_AVG[61]
            - 2.589938e-08 * h[102] / H_AVG[102]
            - 2.584016e-08 * h[107] / H_AVG[107]
            - 2.54601e-08 * h[7] / H_AVG[7]
            + 2.293115e-08 * h[92] / H_AVG[92]
            + 2.241411e-08 * h[35] / H_AVG[35]
            + 2.180584e-08 * h[15] / H_AVG[15]
            - 2.10744e-08 * h[59] / H_AVG[59]
            - 2.031611e-08 * h[101] / H_AVG[101]
            - 1.942182e-08 * h[28] / H_AVG[28]
            + 1.883781e-08 * h[103] / H_AVG[103]
            + 1.772726e-08 * h[69] / H_AVG[69]
            + 1.720207e-08 * h[67] / H_AVG[67]
            + 1.666656e-08 * h[1] / H_AVG[1]
            + 1.654427e-08 * h[100] / H_AVG[100]
            - 1.51925e-08 * h[110] / H_AVG[110]
            - 1.470147e-08 * h[3] / H_AVG[3]
            + 1.412057e-08 * h[37] / H_AVG[37]
            - 1.382951e-08 * h[50] / H_AVG[50]
            + 1.300359e-08 * h[44] / H_AVG[44]
            + 1.299422e-08 * h[36] / H_AVG[36]
            - 1.277028e-08 * h[12] / H_AVG[12]
            - 1.214704e-08 * h[114] / H_AVG[114]
            - 1.204704e-08 * h[2] / H_AVG[2]
            + 1.161301e-08 * h[58] / H_AVG[58]
            - 1.156702e-08 * h[112] / H_AVG[112]
            + 1.131808e-08 * h[76] / H_AVG[76]
            - 1.09077e-08 * h[13] / H_AVG[13]
            - 1.060076e-08 * h[19] / H_AVG[19]
            - 1.039663e-08 * h[111] / H_AVG[111]
            + 9.908575e-09 * h[74] / H_AVG[74]
            + 9.405378e-09 * h[108] / H_AVG[108]
            + 8.843667e-09 * h[5] / H_AVG[5]
            - 8.451446e-09 * h[8] / H_AVG[8]
            - 8.102411e-09 * h[91] / H_AVG[91]
            - 7.553624e-09 * h[4] / H_AVG[4]
            - 7.235262e-09 * h[117] / H_AVG[117]
            + 6.111024e-09 * h[53] / H_AVG[53]
            - 6.046993e-09 * h[66] / H_AVG[66]
            - 5.502963e-09 * h[33] / H_AVG[33]
            - 5.344923e-09 * h[65] / H_AVG[65]
            + 4.94351e-09 * h[82] / H_AVG[82]
            - 4.646647e-09 * h[31] / H_AVG[31]
            - 4.26912e-09 * h[71] / H_AVG[71]
            + 3.96306e-09 * h[94] / H_AVG[94]
            + 3.829456e-09 * h[9] / H_AVG[9]
            + 3.362517e-09 * h[26] / H_AVG[26]
            - 3.017143e-09 * h[14] / H_AVG[14]
            + 2.988324e-09 * h[57] / H_AVG[57]
            + 2.824353e-09 * h[49] / H_AVG[49]
            + 2.759817e-09 * h[77] / H_AVG[77]
            - 2.754805e-09 * h[22] / H_AVG[22]
            + 2.554426e-09 * h[122] / H_AVG[122]
            + 2.065491e-09 * h[72] / H_AVG[72]
            + 1.978633e-09 * h[51] / H_AVG[51]
            - 1.948958e-09 * h[23] / H_AVG[23]
            - 1.345175e-09 * h[89] / H_AVG[89]
            + 1.313963e-09 * h[125] / H_AVG[125]
            - 1.233809e-09 * h[75] / H_AVG[75]
            - 1.067007e-09 * h[87] / H_AVG[87]
            + 1.063123e-09 * h[93] / H_AVG[93]
            - 9.567296e-10 * h[40] / H_AVG[40]
            - 8.893859e-10 * h[20] / H_AVG[20]
            + 7.937008e-10 * h[119] / H_AVG[119]
            - 7.267685e-10 * h[62] / H_AVG[62]
            - 6.532713e-10 * h[95] / H_AVG[95]
            - 6.055617e-10 * h[96] / H_AVG[96]
            + 3.570454e-10 * h[116] / H_AVG[116]
            + 3.408345e-10 * h[56] / H_AVG[56]
            + 2.369997e-10 * h[85] / H_AVG[85]
            + 1.410954e-10 * h[109] / H_AVG[109]
            - 1.228428e-10 * h[98] / H_AVG[98]
            - 9.813892e-11 * h[88] / H_AVG[88]
            - 6.6108e-11 * h[46] / H_AVG[46]
            - 6.242471e-11 * h[47] / H_AVG[47]
            - 3.686281e-11 * h[64] / H_AVG[64]
            + 1.495396e-11 * h[127] / H_AVG[127]
            + 1.323966e-12 * h[41] / H_AVG[41]
        ),
        -0.4320669 + T[8] * (   # class Tbqq
            - 0.216396 * h[104] / H_AVG[104]
            + 0.1461497 * h[16] / H_AVG[16]
            + 0.1259192 * h[120] / H_AVG[120]
            + 0.1248868 * h[68] / H_AVG[68]
            - 0.104323 * h[83] / H_AVG[83]
            - 0.06474367 * h[18] / H_AVG[18]
            + 0.03999315 * h[78] / H_AVG[78]
            - 0.03789333 * h[24] / H_AVG[24]
            + 0.02774753 * h[81] / H_AVG[81]
            + 0.02650483 * h[97] / H_AVG[97]
            - 0.02304937 * h[52] / H_AVG[52]
            - 0.02205636 * h[115] / H_AVG[115]
            - 0.01635702 * h[70] / H_AVG[70]
            + 0.008379549 * h[124] / H_AVG[124]
            - 0.007683996 * h[90] / H_AVG[90]
            - 0.006915622 * h[123] / H_AVG[123]
            + 0.0008494612 * h[25] / H_AVG[25]
            - 0.0001158724 * h[0] / H_AVG[0]
            - 2.218373e-05 * h[99] / H_AVG[99]
            - 1.012116e-05 * h[21] / H_AVG[21]
            + 1.051048e-06 * h[84] / H_AVG[84]
            + 2.480868e-07 * h[55] / H_AVG[55]
            + 1.665045e-07 * h[79] / H_AVG[79]
            - 1.588845e-07 * h[105] / H_AVG[105]
            - 1.289519e-07 * h[106] / H_AVG[106]
            + 1.048101e-07 * h[113] / H_AVG[113]
            - 9.95414e-08 * h[34] / H_AVG[34]
            + 8.935967e-08 * h[73] / H_AVG[73]
            + 8.301195e-08 * h[10] / H_AVG[10]
            + 7.912101e-08 * h[11] / H_AVG[11]
            - 6.737494e-08 * h[6] / H_AVG[6]
            + 6.428171e-08 * h[86] / H_AVG[86]
            - 6.113324e-08 * h[32] / H_AVG[32]
            + 5.620923e-08 * h[17] / H_AVG[17]
            + 5.079399e-08 * h[38] / H_AVG[38]
            - 4.170478e-08 * h[42] / H_AVG[42]
            - 3.716745e-08 * h[54] / H_AVG[54]
            - 3.53299e-08 * h[29] / H_AVG[29]
            + 3.202824e-08 * h[118] / H_AVG[118]
            + 2.500193e-08 * h[48] / H_AVG[48]
            - 2.385204e-08 * h[45] / H_AVG[45]
            - 2.303929e-08 * h[39] / H_AVG[39]
            + 2.124873e-08 * h[27] / H_AVG[27]
            + 2.115645e-08 * h[63] / H_AVG[63]
            + 2.092709e-08 * h[30] / H_AVG[30]
            + 2.049086e-08 * h[43] / H_AVG[43]
            + 1.907377e-08 * h[121] / H_AVG[121]
            - 1.872774e-08 * h[80] / H_AVG[80]
            - 1.713357e-08 * h[126] / H_AVG[126]
            + 1.65508e-08 * h[60] / H_AVG[60]
            - 1.629662e-08 * h[61] / H_AVG[61]
            - 1.54719e-08 * h[102] / H_AVG[102]
            - 1.54396e-08 * h[107] / H_AVG[107]
            - 1.521504e-08 * h[7] / H_AVG[7]
            + 1.370039e-08 * h[92] / H_AVG[92]
            + 1.338489e-08 * h[35] / H_AVG[35]
            + 1.303006e-08 * h[15] / H_AVG[15]
            - 1.258506e-08 * h[59] / H_AVG[59]
            - 1.21406e-08 * h[101] / H_AVG[101]
            - 1.161473e-08 * h[28] / H_AVG[28]
            + 1.125285e-08 * h[103] / H_AVG[103]
            + 1.059132e-08 * h[69] / H_AVG[69]
            + 1.028032e-08 * h[67] / H_AVG[67]
            + 9.961244e-09 * h[1] / H_AVG[1]
            + 9.886484e-09 * h[100] / H_AVG[100]
            - 9.081164e-09 * h[110] / H_AVG[110]
            - 8.785521e-09 * h[3] / H_AVG[3]
            + 8.43795e-09 * h[37] / H_AVG[37]
            - 8.260071e-09 * h[50] / H_AVG[50]
            + 7.770545e-09 * h[44] / H_AVG[44]
            + 7.763344e-09 * h[36] / H_AVG[36]
            - 7.632874e-09 * h[12] / H_AVG[12]
            - 7.259011e-09 * h[114] / H_AVG[114]
            - 7.198073e-09 * h[2] / H_AVG[2]
            + 6.939777e-09 * h[58] / H_AVG[58]
            - 6.911125e-09 * h[112] / H_AVG[112]
            + 6.763414e-09 * h[76] / H_AVG[76]
            - 6.517404e-09 * h[13] / H_AVG[13]
            - 6.333613e-09 * h[19] / H_AVG[19]
            - 6.213286e-09 * h[111] / H_AVG[111]
            + 5.921882e-09 * h[74] / H_AVG[74]
            + 5.619867e-09 * h[108] / H_AVG[108]
            + 5.283616e-09 * h[5] / H_AVG[5]
            - 5.04993e-09 * h[8] / H_AVG[8]
            - 4.841042e-09 * h[91] / H_AVG[91]
            - 4.514002e-09 * h[4] / H_AVG[4]
            - 4.322716e-09 * h[117] / H_AVG[117]
            + 3.652302e-09 * h[53] / H_AVG[53]
            - 3.610208e-09 * h[66] / H_AVG[66]
            - 3.288216e-09 * h[33] / H_AVG[33]
            - 3.194149e-09 * h[65] / H_AVG[65]
            + 2.954201e-09 * h[82] / H_AVG[82]
            - 2.774547e-09 * h[31] / H_AVG[31]
            - 2.55015e-09 * h[71] / H_AVG[71]
            + 2.370102e-09 * h[94] / H_AVG[94]
            + 2.288212e-09 * h[9] / H_AVG[9]
            + 2.011367e-09 * h[26] / H_AVG[26]
            - 1.802457e-09 * h[14] / H_AVG[14]
            + 1.786435e-09 * h[57] / H_AVG[57]
            + 1.687722e-09 * h[49] / H_AVG[49]
            + 1.650459e-09 * h[77] / H_AVG[77]
            - 1.645837e-09 * h[22] / H_AVG[22]
            + 1.526346e-09 * h[122] / H_AVG[122]
            + 1.234196e-09 * h[72] / H_AVG[72]
            + 1.182174e-09 * h[51] / H_AVG[51]
            - 1.162446e-09 * h[23] / H_AVG[23]
            - 8.036801e-10 * h[89] / H_AVG[89]
            + 7.852294e-10 * h[125] / H_AVG[125]
            - 7.370303e-10 * h[75] / H_AVG[75]
            - 6.377581e-10 * h[87] / H_AVG[87]
            + 6.35595e-10 * h[93] / H_AVG[93]
            - 5.71722e-10 * h[40] / H_AVG[40]
            - 5.314202e-10 * h[20] / H_AVG[20]
            + 4.747292e-10 * h[119] / H_AVG[119]
            - 4.342071e-10 * h[62] / H_AVG[62]
            - 3.903044e-10 * h[95] / H_AVG[95]
            - 3.61696e-10 * h[96] / H_AVG[96]
            + 2.133573e-10 * h[116] / H_AVG[116]
            + 2.037058e-10 * h[56] / H_AVG[56]
            + 1.416747e-10 * h[85] / H_AVG[85]
            + 8.429616e-11 * h[109] / H_AVG[109]
            - 7.34229e-11 * h[98] / H_AVG[98]
            - 5.857187e-11 * h[88] / H_AVG[88]
            - 3.951808e-11 * h[46] / H_AVG[46]
            - 3.729551e-11 * h[47] / H_AVG[47]
            - 2.041348e-11 * h[64] / H_AVG[64]
            + 8.956992e-12 * h[127] / H_AVG[127]
            + 7.918312e-13 * h[41] / H_AVG[41]
        ),
        -1.495433 + T[9] * (   # class Tbl
            + 0.23865 * h[16] / H_AVG[16]
            + 0.1194219 * h[24] / H_AVG[24]
            - 0.1147906 * h[104] / H_AVG[104]
            + 0.0867944 * h[18] / H_AVG[18]
            + 0.06898835 * h[97] / H_AVG[97]
            - 0.06889069 * h[124] / H_AVG[124]
            - 0.06449341 * h[115] / H_AVG[115]
            - 0.05894783 * h[68] / H_AVG[68]
            + 0.05572059 * h[52] / H_AVG[52]
            + 0.03604426 * h[78] / H_AVG[78]
            + 0.01776882 * h[120] / H_AVG[120]
            + 0.01680435 * h[70] / H_AVG[70]
            + 0.01592608 * h[123] / H_AVG[123]
            + 0.01186613 * h[25] / H_AVG[25]
            + 0.01030092 * h[90] / H_AVG[90]
            - 0.009021226 * h[83] / H_AVG[83]
            - 0.00519104 * h[81] / H_AVG[81]
            - 0.0003099982 * h[0] / H_AVG[0]
            + 5.189571e-05 * h[21] / H_AVG[21]
            - 1.523791e-05 * h[99] / H_AVG[99]
            + 7.276351e-07 * h[84] / H_AVG[84]
            + 1.716192e-07 * h[55] / H_AVG[55]
            + 1.15137e-07 * h[79] / H_AVG[79]
            - 1.100266e-07 * h[105] / H_AVG[105]
            - 8.927033e-08 * h[106] / H_AVG[106]
            + 7.259151e-08 * h[113] / H_AVG[113]
            - 6.890068e-08 * h[34] / H_AVG[34]
            + 6.185319e-08 * h[73] / H_AVG[73]
            + 5.742095e-08 * h[10] / H_AVG[10]
            + 5.478345e-08 * h[11] / H_AVG[11]
            - 4.65978e-08 * h[6] / H_AVG[6]
            + 4.448741e-08 * h[86] / H_AVG[86]
            - 4.232995e-08 * h[32] / H_AVG[32]
            + 3.890273e-08 * h[17] / H_AVG[17]
            + 3.513121e-08 * h[38] / H_AVG[38]
            - 2.886561e-08 * h[42] / H_AVG[42]
            - 2.574145e-08 * h[54] / H_AVG[54]
            - 2.444558e-08 * h[29] / H_AVG[29]
            + 2.217126e-08 * h[118] / H_AVG[118]
            + 1.728995e-08 * h[48] / H_AVG[48]
            - 1.651192e-08 * h[45] / H_AVG[45]
            - 1.593618e-08 * h[39] / H_AVG[39]
            + 1.470862e-08 * h[27] / H_AVG[27]
            + 1.464022e-08 * h[63] / H_AVG[63]
            + 1.449569e-08 * h[30] / H_AVG[30]
            + 1.417443e-08 * h[43] / H_AVG[43]
            + 1.319171e-08 * h[121] / H_AVG[121]
            - 1.295769e-08 * h[80] / H_AVG[80]
            - 1.187265e-08 * h[126] / H_AVG[126]
            + 1.144964e-08 * h[60] / H_AVG[60]
            - 1.126924e-08 * h[61] / H_AVG[61]
            - 1.070725e-08 * h[102] / H_AVG[102]
            - 1.067202e-08 * h[107] / H_AVG[107]
            - 1.052894e-08 * h[7] / H_AVG[7]
            + 9.476996e-09 * h[92] / H_AVG[92]
            + 9.269694e-09 * h[35] / H_AVG[35]
            + 9.012389e-09 * h[15] / H_AVG[15]
            - 8.715062e-09 * h[59] / H_AVG[59]
            - 8.408864e-09 * h[101] / H_AVG[101]
            - 8.018419e-09 * h[28] / H_AVG[28]
            + 7.792578e-09 * h[103] / H_AVG[103]
            + 7.342777e-09 * h[69] / H_AVG[69]
            + 7.127533e-09 * h[67] / H_AVG[67]
            + 6.886092e-09 * h[1] / H_AVG[1]
            + 6.833535e-09 * h[100] / H_AVG[100]
            - 6.279046e-09 * h[110] / H_AVG[110]
            - 6.088056e-09 * h[3] / H_AVG[3]
            + 5.841545e-09 * h[37] / H_AVG[37]
            - 5.711841e-09 * h[50] / H_AVG[50]
            + 5.375833e-09 * h[36] / H_AVG[36]
            + 5.374588e-09 * h[44] / H_AVG[44]
            - 5.288292e-09 * h[12] / H_AVG[12]
            - 5.02465e-09 * h[114] / H_AVG[114]
            - 4.989558e-09 * h[2] / H_AVG[2]
            + 4.807739e-09 * h[58] / H_AVG[58]
            - 4.793116e-09 * h[112] / H_AVG[112]
            + 4.686299e-09 * h[76] / H_AVG[76]
            - 4.504638e-09 * h[13] / H_AVG[13]
            - 4.383669e-09 * h[19] / H_AVG[19]
            - 4.294921e-09 * h[111] / H_AVG[111]
            + 4.097204e-09 * h[74] / H_AVG[74]
            + 3.880133e-09 * h[108] / H_AVG[108]
            + 3.653337e-09 * h[5] / H_AVG[5]
            - 3.497916e-09 * h[8] / H_AVG[8]
            - 3.358108e-09 * h[91] / H_AVG[91]
            - 3.127363e-09 * h[4] / H_AVG[4]
            - 2.995199e-09 * h[117] / H_AVG[117]
            + 2.527179e-09 * h[53] / H_AVG[53]
            - 2.499787e-09 * h[66] / H_AVG[66]
            - 2.269522e-09 * h[33] / H_AVG[33]
            - 2.21719e-09 * h[65] / H_AVG[65]
            + 2.049685e-09 * h[82] / H_AVG[82]
            - 1.917144e-09 * h[31] / H_AVG[31]
            - 1.76326e-09 * h[71] / H_AVG[71]
            + 1.639088e-09 * h[94] / H_AVG[94]
            + 1.583166e-09 * h[9] / H_AVG[9]
            + 1.398057e-09 * h[26] / H_AVG[26]
            - 1.248159e-09 * h[14] / H_AVG[14]
            + 1.240946e-09 * h[57] / H_AVG[57]
            + 1.166727e-09 * h[49] / H_AVG[49]
            + 1.139991e-09 * h[77] / H_AVG[77]
            - 1.13828e-09 * h[22] / H_AVG[22]
            + 1.060055e-09 * h[122] / H_AVG[122]
            + 8.530463e-10 * h[72] / H_AVG[72]
            + 8.21829e-10 * h[51] / H_AVG[51]
            - 8.074659e-10 * h[23] / H_AVG[23]
            - 5.526836e-10 * h[89] / H_AVG[89]
            + 5.421104e-10 * h[125] / H_AVG[125]
            - 5.090332e-10 * h[75] / H_AVG[75]
            + 4.400482e-10 * h[93] / H_AVG[93]
            - 4.391388e-10 * h[87] / H_AVG[87]
            - 3.955096e-10 * h[40] / H_AVG[40]
            - 3.677953e-10 * h[20] / H_AVG[20]
            + 3.295913e-10 * h[119] / H_AVG[119]
            - 2.997156e-10 * h[62] / H_AVG[62]
            - 2.699939e-10 * h[95] / H_AVG[95]
            - 2.496533e-10 * h[96] / H_AVG[96]
            + 1.473802e-10 * h[116] / H_AVG[116]
            + 1.415376e-10 * h[56] / H_AVG[56]
            + 9.873725e-11 * h[85] / H_AVG[85]
            + 5.812033e-11 * h[109] / H_AVG[109]
            - 5.061993e-11 * h[98] / H_AVG[98]
            - 4.02714e-11 * h[88] / H_AVG[88]
            - 2.754308e-11 * h[46] / H_AVG[46]
            - 2.574581e-11 * h[47] / H_AVG[47]
            - 1.348451e-11 * h[64] / H_AVG[64]
            + 6.441921e-12 * h[127] / H_AVG[127]
            + 5.734483e-13 * h[41] / H_AVG[41]
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
