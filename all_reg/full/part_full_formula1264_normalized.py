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
  neuron 16:  12.3%
  neuron 24:   9.5%
  neuron 97:   9.3%
  neuron 120:   9.2%
  neuron 104:   9.2%
  neuron 18:   9.1%
  neuron 68:   8.2%
  neuron 115:   7.2%
  neuron 52:   6.5%
  neuron 83:   5.6%
  neuron 78:   4.9%
  neuron 124:   4.7%
  neuron 81:   3.8%
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
  neuron 22:   0.0%
  neuron 77:   0.0%
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

Whole test file (2,000,000 jets): accuracy 74.65% (the network: 86.03%); same class as the network for 79.33% of jets.

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
  Q.dr_10                  ΔR from the jet axis of particle 10 (ParT input; empty slot: 0)
  Q.dr_16                  ΔR from the jet axis of particle 16 (ParT input; empty slot: 0)
  Q.dr_19                  ΔR from the jet axis of particle 19 (ParT input; empty slot: 0)
  Q.dr_21                  ΔR from the jet axis of particle 21 (ParT input; empty slot: 0)
  Q.dr_28                  ΔR from the jet axis of particle 28 (ParT input; empty slot: 0)
  Q.dr_31                  ΔR from the jet axis of particle 31 (ParT input; empty slot: 0)
  Q.dr_4                   ΔR from the jet axis of particle 4 (ParT input; empty slot: 0)
  Q.dr_53                  ΔR from the jet axis of particle 53 (ParT input; empty slot: 0)
  Q.dr_77                  ΔR from the jet axis of particle 77 (ParT input; empty slot: 0)
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
  Q.eta_2                  Δη of particle 2 (ParT input; empty slot: 0)
  Q.eta_28                 Δη of particle 28 (ParT input; empty slot: 0)
  Q.eta_29                 Δη of particle 29 (ParT input; empty slot: 0)
  Q.eta_36                 Δη of particle 36 (ParT input; empty slot: 0)
  Q.eta_37                 Δη of particle 37 (ParT input; empty slot: 0)
  Q.eta_4                  Δη of particle 4 (ParT input; empty slot: 0)
  Q.eta_48                 Δη of particle 48 (ParT input; empty slot: 0)
  Q.eta_7                  Δη of particle 7 (ParT input; empty slot: 0)
  Q.eta_76                 Δη of particle 76 (ParT input; empty slot: 0)
  Q.eta_9                  Δη of particle 9 (ParT input; empty slot: 0)
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.ischhad_26             1 if a charged hadron of particle 26 (ParT input; empty slot: 0)
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
  Q.ismuon_2               1 if a muon of particle 2 (ParT input; empty slot: 0)
  Q.ismuon_24              1 if a muon of particle 24 (ParT input; empty slot: 0)
  Q.ismuon_27              1 if a muon of particle 27 (ParT input; empty slot: 0)
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
  Q.lne_2                  ln E [GeV] of particle 2 (ParT input; empty slot: ln 1e-8)
  Q.lne_4                  ln E [GeV] of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lne_5                  ln E [GeV] of particle 5 (ParT input; empty slot: ln 1e-8)
  Q.lne_7                  ln E [GeV] of particle 7 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_17              ln(E / E of the jet) of particle 17 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_4               ln(E / E of the jet) of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_82              ln(E / E of the jet) of particle 82 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_44                ln pT [GeV] of particle 44 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_12             ln(pT / pT of the jet) of particle 12 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_3              ln(pT / pT of the jet) of particle 3 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_45             ln(pT / pT of the jet) of particle 45 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_77             ln(pT / pT of the jet) of particle 77 (ParT input; empty slot: ln 1e-8)
  Q.log_sum_pt             natural log of the total pT
  Q.lund1_lndelta          ln Δ of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund1_lnz              ln z of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund2_lndelta          ln Δ of the 2. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lndelta          ln Δ of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lnz              ln z of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund_max_lndelta       ln Δ of the primary splitting with the largest kT
  Q.lund_max_lnkt          largest ln kT among the primary splittings
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
  Q.td0_23                 tanh(d0 [mm]) of particle 23 (ParT input; empty slot: 0)
  Q.td0_3                  tanh(d0 [mm]) of particle 3 (ParT input; empty slot: 0)
  Q.td0_30                 tanh(d0 [mm]) of particle 30 (ParT input; empty slot: 0)
  Q.td0_33                 tanh(d0 [mm]) of particle 33 (ParT input; empty slot: 0)
  Q.td0_38                 tanh(d0 [mm]) of particle 38 (ParT input; empty slot: 0)
  Q.td0_7                  tanh(d0 [mm]) of particle 7 (ParT input; empty slot: 0)
  Q.td0_9                  tanh(d0 [mm]) of particle 9 (ParT input; empty slot: 0)
  Q.tdz_0                  tanh(dz [mm]) of particle 0 (ParT input; empty slot: 0)
  Q.tdz_1                  tanh(dz [mm]) of particle 1 (ParT input; empty slot: 0)
  Q.tdz_2                  tanh(dz [mm]) of particle 2 (ParT input; empty slot: 0)
  Q.tdz_25                 tanh(dz [mm]) of particle 25 (ParT input; empty slot: 0)
  Q.tdz_29                 tanh(dz [mm]) of particle 29 (ParT input; empty slot: 0)
  Q.tdz_33                 tanh(dz [mm]) of particle 33 (ParT input; empty slot: 0)
  Q.tdz_44                 tanh(dz [mm]) of particle 44 (ParT input; empty slot: 0)
  Q.tdz_47                 tanh(dz [mm]) of particle 47 (ParT input; empty slot: 0)
  Q.tdz_6                  tanh(dz [mm]) of particle 6 (ParT input; empty slot: 0)
  Q.tdz_68                 tanh(dz [mm]) of particle 68 (ParT input; empty slot: 0)
  Q.tdz_74                 tanh(dz [mm]) of particle 74 (ParT input; empty slot: 0)
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
        dr_10=pfeat(10, 'dr'),
        dr_16=pfeat(16, 'dr'),
        dr_19=pfeat(19, 'dr'),
        dr_21=pfeat(21, 'dr'),
        dr_28=pfeat(28, 'dr'),
        dr_31=pfeat(31, 'dr'),
        dr_4=pfeat(4, 'dr'),
        dr_53=pfeat(53, 'dr'),
        dr_77=pfeat(77, 'dr'),
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
        eta_2=pfeat(2, 'eta'),
        eta_28=pfeat(28, 'eta'),
        eta_29=pfeat(29, 'eta'),
        eta_36=pfeat(36, 'eta'),
        eta_37=pfeat(37, 'eta'),
        eta_4=pfeat(4, 'eta'),
        eta_48=pfeat(48, 'eta'),
        eta_7=pfeat(7, 'eta'),
        eta_76=pfeat(76, 'eta'),
        eta_9=pfeat(9, 'eta'),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        ischhad_26=pfeat(26, 'ischhad'),
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
        ismuon_2=pfeat(2, 'ismuon'),
        ismuon_24=pfeat(24, 'ismuon'),
        ismuon_27=pfeat(27, 'ismuon'),
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
        lne_2=pfeat(2, 'lne'),
        lne_4=pfeat(4, 'lne'),
        lne_5=pfeat(5, 'lne'),
        lne_7=pfeat(7, 'lne'),
        lnerel_17=pfeat(17, 'lnerel'),
        lnerel_4=pfeat(4, 'lnerel'),
        lnerel_82=pfeat(82, 'lnerel'),
        lnpt_44=pfeat(44, 'lnpt'),
        lnptrel_12=pfeat(12, 'lnptrel'),
        lnptrel_3=pfeat(3, 'lnptrel'),
        lnptrel_45=pfeat(45, 'lnptrel'),
        lnptrel_77=pfeat(77, 'lnptrel'),
        log_sum_pt=math.log(tot),
        lund1_lndelta=lund(1, 'lndelta'),
        lund1_lnz=lund(1, 'lnz'),
        lund2_lndelta=lund(2, 'lndelta'),
        lund3_lndelta=lund(3, 'lndelta'),
        lund3_lnz=lund(3, 'lnz'),
        lund_max_lndelta=lund(0, 'maxdelta'),
        lund_max_lnkt=lund(0, 'maxkt'),
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
        td0_23=pfeat(23, 'td0'),
        td0_3=pfeat(3, 'td0'),
        td0_30=pfeat(30, 'td0'),
        td0_33=pfeat(33, 'td0'),
        td0_38=pfeat(38, 'td0'),
        td0_7=pfeat(7, 'td0'),
        td0_9=pfeat(9, 'td0'),
        tdz_0=pfeat(0, 'tdz'),
        tdz_1=pfeat(1, 'tdz'),
        tdz_2=pfeat(2, 'tdz'),
        tdz_25=pfeat(25, 'tdz'),
        tdz_29=pfeat(29, 'tdz'),
        tdz_33=pfeat(33, 'tdz'),
        tdz_44=pfeat(44, 'tdz'),
        tdz_47=pfeat(47, 'tdz'),
        tdz_6=pfeat(6, 'tdz'),
        tdz_68=pfeat(68, 'tdz'),
        tdz_74=pfeat(74, 'tdz'),
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
    )


def neuron_0(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.005194661
    )
    return z


def neuron_1(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.328472e-06
    )
    return z


def neuron_2(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.442857e-06
    )
    return z


def neuron_3(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.399042e-06
    )
    return z


def neuron_4(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.789399e-06
    )
    return z


def neuron_5(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.155747e-06
    )
    return z


def neuron_6(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.769729e-05
    )
    return z


def neuron_7(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.433383e-06
    )
    return z


def neuron_8(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.926356e-06
    )
    return z


def neuron_9(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.440223e-06
    )
    return z


def neuron_10(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.570377e-05
    )
    return z


def neuron_11(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.888238e-05
    )
    return z


def neuron_12(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.094681e-06
    )
    return z


def neuron_13(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.427915e-06
    )
    return z


def neuron_14(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.850953e-06
    )
    return z


def neuron_15(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.461601e-06
    )
    return z


def neuron_16(Q):
    # scale S = 18.24; each line: share * term / its average size
    z = 18.23683 * (0.2538843
        - 0.2034685 * Q.z_charged_had / 0.5066138   # -20.3%  z_charged_had
        - 0.1479156 * Q.z_neutral / 0.4037153   # -14.8%  z_neutral
        + 0.0571181 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # +5.7%  lep_ptrel < 27.32
        - 0.04125783 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # -4.1%  n_s3d_above_3 < 10
        - 0.03954884 * max(0.0, 0.2801368 - Q.z_displaced5) / 0.1998239   # -4.0%  z_displaced5 < 0.2801
        - 0.0312273 * max(0.0, 33.08364 - Q.sip_3d_3) / 19.9074   # -3.1%  sip_3d_3 < 33.08
        + 0.03047588 * Q.lep_ptrel / 7.315413   # +3.0%  lep_ptrel
        + 0.02573808 * max(0.0, 0.2801368 - Q.z_displaced5) * max(0.0, 33.08364 - Q.sip_3d_3) / 4.739402   # +2.6%  z_displaced5 < 0.2801 and sip_3d_3 < 33.08
        + 0.02443462 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # +2.4%  sip_3d_2 < 226.3
        + 0.02389377 * max(0.0, 577.991 - Q.sip_3d_3) / 516.7039   # +2.4%  sip_3d_3 < 578
        - 0.02322569 * max(0.0, Q.lep_z - 0.01245833) / 0.08062103   # -2.3%  lep_z > 0.01246
        + 0.02018033 * max(0.0, 0.2801368 - Q.z_displaced5) * max(0.0, 175.9957 - Q.sip_3d_3) / 30.21447   # +2.0%  z_displaced5 < 0.2801 and sip_3d_3 < 176
        + 0.02008338 * max(0.0, 29.0 - Q.n_charged_had) / 10.94929   # +2.0%  n_charged_had < 29
        + 0.0175859 * Q.n_sd0_above_2 / 4.174103   # +1.8%  n_sd0_above_2
        - 0.0166727 * Q.mass_charged / 64.25857   # -1.7%  mass_charged
        - 0.01578174 * max(0.0, 87.9162 - Q.mass_neutral) * max(0.0, 0.06762785 - Q.M2_b2) / 1.669202   # -1.6%  mass_neutral < 87.92 and M2_b2 < 0.06763
        - 0.01306614 * max(0.0, 1.777286 - Q.mass_displaced3) / 0.7377974   # -1.3%  mass_displaced3 < 1.777
        + 0.0127083 * Q.n_photon / 16.03902   # +1.3%  n_photon
        - 0.0125981 * max(0.0, 0.06245248 - Q.z_displaced5) / 0.02875704   # -1.3%  z_displaced5 < 0.06245
        - 0.01243066 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, 7.782712 - Q.D2_b2) / 31.51573   # -1.2%  n_s3d_above_3 < 10 and D2_b2 < 7.783
        + 0.01229662 * max(0.0, 41.0 - Q.n_pairs_kt_above_3) / 7.619143   # +1.2%  n_pairs_kt_above_3 < 41
        + 0.0116595 * max(0.0, 0.09740996 - Q.M2) / 0.02324748   # +1.2%  M2 < 0.09741
        + 0.01165927 * max(0.0, 87.9162 - Q.mass_neutral) / 43.67158   # +1.2%  mass_neutral < 87.92
        - 0.01021693 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # -1.0%  sip_3d_2 < 447.1
        + 0.01018128 * max(0.0, 4.41005 - Q.mass_displaced3) / 2.286034   # +1.0%  mass_displaced3 < 4.41
        + 0.009321081 * max(0.0, 5.8125 - Q.max_abs_d0) * max(0.0, 49.03695 - Q.sip_3d_3) / 122.3704   # +0.9%  max_abs_d0 < 5.812 and sip_3d_3 < 49.04
        - 0.008251165 * max(0.0, 5.0 - Q.n_sdz_above_2) / 1.969117   # -0.8%  n_sdz_above_2 < 5
        + 0.006608059 * max(0.0, Q.M3 - 0.02540381) / 0.01242824   # +0.7%  M3 > 0.0254
        + 0.006568553 * max(0.0, Q.sd_mass - 119.4443) / 9.070113   # +0.7%  sd_mass > 119.4
        + 0.006509751 * max(0.0, 2.957031 - Q.max_abs_dz) / 1.247099   # +0.7%  max_abs_dz < 2.957
        + 0.006050185 * max(0.0, Q.mass_neutral - 19.57354) / 26.6998   # +0.6%  mass_neutral > 19.57
        + 0.00593234 * max(0.0, 87.9162 - Q.mass_neutral) * max(0.0, Q.n_lund - 5.0) / 240.7844   # +0.6%  mass_neutral < 87.92 and n_lund > 5
        + 0.0058829 * max(0.0, Q.sj3_pair_mass_max - 83.63398) / 16.36149   # +0.6%  sj3_pair_mass_max > 83.63
        - 0.005618085 * max(0.0, 16.23838 - Q.sip_3d_3) / 8.003297   # -0.6%  sip_3d_3 < 16.24
        + 0.005305925 * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0001737301   # +0.5%  e3_b2 < 0.0002537
        - 0.005141972 * max(0.0, Q.lep_z - 0.221436) / 0.03510013   # -0.5%  lep_z > 0.2214
        - 0.0044497 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # -0.4%  sd_mass > 154.6
        - 0.004095544 * max(0.0, 33.0 - Q.n_pt_above_1) / 3.50532   # -0.4%  n_pt_above_1 < 33
        - 0.00395165 * max(0.0, 1.763283 - Q.min_pair_mass) / 0.6216059   # -0.4%  min_pair_mass < 1.763
        + 0.00386936 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +0.4%  n_pairs_kt_above_1 < 80
        - 0.003675667 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.4%  lep_ptrel > 43.21
        + 0.003581373 * max(0.0, Q.C2 - 0.1323775) / 0.03527464   # +0.4%  C2 > 0.1324
        + 0.003538012 * max(0.0, 13.03663 - Q.mass_displaced3) / 8.661134   # +0.4%  mass_displaced3 < 13.04
        - 0.003438911 * max(0.0, 0.6455078 - Q.max_abs_dz) / 0.1467826   # -0.3%  max_abs_dz < 0.6455
        + 0.003332164 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 1.43511   # +0.3%  n_dr_0p2_0p4 < 7
        + 0.003180464 * max(0.0, Q.z_top15_slots - 0.8008865) / 0.07132828   # +0.3%  z_top15_slots > 0.8009
        + 0.002627237 * max(0.0, 0.2413007 - Q.C2_b05) / 0.01461377   # +0.3%  C2_b05 < 0.2413
        + 0.002602455 * max(0.0, Q.mass_top50 - 125.4927) / 8.989867   # +0.3%  mass_top50 > 125.5
        + 0.002585015 * max(0.0, Q.mass_top40 - 122.7145) / 8.484825   # +0.3%  mass_top40 > 122.7
        - 0.00255825 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # -0.3%  n_pairs_kt_above_3 < 28
        - 0.002515391 * max(0.0, Q.lund_max_lndelta - -1.062087) * max(0.0, 4.606241 - Q.sip_3d_3) / 0.160478   # -0.3%  lund_max_lndelta > -1.062 and sip_3d_3 < 4.606
        - 0.002381506 * max(0.0, 0.02720087 - Q.tau4) / 0.003105367   # -0.2%  tau4 < 0.0272
        + 0.002324877 * max(0.0, 0.2490748 - Q.mass_displaced3) / 0.08578912   # +0.2%  mass_displaced3 < 0.2491
        + 0.002063816 * max(0.0, 87.9162 - Q.mass_neutral) * max(0.0, 6.0 - Q.n_dr_0p1_0p2) / 46.21754   # +0.2%  mass_neutral < 87.92 and n_dr_0p1_0p2 < 6
        - 0.002028188 * max(0.0, 29.0 - Q.n_charged_had) * max(0.0, Q.ecf_g42 - 2.701529e-06) / 0.0002316724   # -0.2%  n_charged_had < 29 and ecf_g42 > 2.702e-06
        + 0.002019893 * max(0.0, Q.lund_max_lndelta - -1.062087) * max(0.0, 34.83233 - Q.sj4_pair_mass_min) / 3.204401   # +0.2%  lund_max_lndelta > -1.062 and sj4_pair_mass_min < 34.83
        - 0.001520984 * max(0.0, 0.0265067 - Q.z_dr_0p4_up) / 0.01031536   # -0.2%  z_dr_0p4_up < 0.02651
        - 0.001426737 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # -0.1%  mass_top50 > 161.1
        - 0.001391006 * max(0.0, Q.psi_0p3 - 0.9828705) / 0.002731628   # -0.1%  psi_0p3 > 0.9829
        + 0.001234083 * max(0.0, Q.mass_top5 - 62.84493) / 3.692784   # +0.1%  mass_top5 > 62.84
        - 0.001203821 * max(0.0, 92.68149 - Q.mass_top40) / 5.990263   # -0.1%  mass_top40 < 92.68
        - 0.001177871 * max(0.0, 0.06245248 - Q.z_displaced5) * max(0.0, 0.8786609 - Q.tau54) / 0.001586542   # -0.1%  z_displaced5 < 0.06245 and tau54 < 0.8787
        - 0.001157911 * max(0.0, Q.tau3 - 0.06348273) / 0.006109561   # -0.1%  tau3 > 0.06348
        + 0.001155462 * max(0.0, Q.lund_max_lndelta - -1.062087) * max(0.0, 15.0 - Q.n_charged_had) / 0.2371327   # +0.1%  lund_max_lndelta > -1.062 and n_charged_had < 15
        + 0.001140152 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.M3_b05 - 0.06007167) / 0.06230838   # +0.1%  n_pairs_kt_above_3 < 28 and M3_b05 > 0.06007
        - 0.001072287 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, 1.516502 - Q.D2) / 0.2767988   # -0.1%  n_dr_0p2_0p4 < 7 and D2 < 1.517
        + 0.0009750922 * max(0.0, 0.01556887 - Q.z_dr_0p05_0p1) / 0.002231501   # +0.1%  z_dr_0p05_0p1 < 0.01557
        - 0.0009548542 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.2338497 - Q.lep_dr) / 0.6071668   # -0.1%  n_pairs_kt_above_3 < 28 and lep_dr < 0.2338
        + 0.0007957453 * max(0.0, 87.9162 - Q.mass_neutral) * max(0.0, Q.mass_2charged - 17.49066) / 266.6911   # +0.1%  mass_neutral < 87.92 and mass_2charged > 17.49
        + 0.0007057162 * max(0.0, -0.3599791 - Q.jet_charge_k05) / 0.04881883   # +0.1%  jet_charge_k05 < -0.36
        + 0.0006576914 * max(0.0, Q.lund_max_lndelta - -1.062087) / 0.1730972   # +0.1%  lund_max_lndelta > -1.062
        - 0.0006503467 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # -0.1%  max_abs_d0 < 5.812
        - 0.0006078652 * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 1.107657   # -0.1%  n_lund_kt_above_5 > 1
        + 0.0005352294 * max(0.0, Q.jet_charge_k05 - 0.5864519) / 0.01952642   # +0.1%  jet_charge_k05 > 0.5865
        - 0.0005058411 * max(0.0, Q.max_pair_mass - 55.24488) / 0.6579942   # -0.1%  max_pair_mass > 55.24
        - 0.0004946651 * max(0.0, 87.9162 - Q.mass_neutral) * max(0.0, 0.02981375 - Q.dr_max_012) / 0.1323256   # -0.0%  mass_neutral < 87.92 and dr_max_012 < 0.02981
        - 0.0004802848 * max(0.0, Q.n_electron - 1.0) / 0.08167   # -0.0%  n_electron > 1
        + 0.0004068095 * max(0.0, 3.452465 - Q.lne_4) / 0.07948689   # +0.0%  lne_4 < 3.452
        + 0.0003776132 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # +0.0%  sd_mass > 175.9
        + 0.0002740075 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, 15.0 - Q.n_dr_0p4_up) / 63.11649   # +0.0%  n_s3d_above_3 < 10 and n_dr_0p4_up < 15
        + 0.0002684816 * max(0.0, Q.n_pairs_kt_above_10 - 17.0) / 0.94113   # +0.0%  n_pairs_kt_above_10 > 17
        + 0.0002465729 * max(0.0, Q.lep_z - 0.221436) * max(0.0, 0.7544983 - Q.tau32) / 0.006468695   # +0.0%  lep_z > 0.2214 and tau32 < 0.7545
        + 0.0001822385 * max(0.0, 29.0 - Q.n_charged_had) * max(0.0, Q.n_lepton - 1.0) / 1.919187   # +0.0%  n_charged_had < 29 and n_lepton > 1
        - 0.000181328 * max(0.0, 0.302405 - Q.N2_b05) / 0.002473216   # -0.0%  N2_b05 < 0.3024
        + 0.0001515079 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, Q.d0err_12 - 0.0236969) / 0.5206052   # +0.0%  sip_3d_2 < 447.1 and d0err_12 > 0.0237
        + 0.0001289557 * max(0.0, 1.824785 - Q.sj3_mass2) / 0.06171879   # +0.0%  sj3_mass2 < 1.825
        - 0.0001156747 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, Q.sip_3d_3 - 49.03695) / 133.8744   # -0.0%  n_lund_kt_above_5 > 1 and sip_3d_3 > 49.04
        + 0.0001119214 * max(0.0, 0.01556887 - Q.z_dr_0p05_0p1) * max(0.0, 0.1083984 - Q.eta_36) / 0.0003041121   # +0.0%  z_dr_0p05_0p1 < 0.01557 and eta_36 < 0.1084
        + 0.0001010245 * max(0.0, 0.01556887 - Q.z_dr_0p05_0p1) * max(0.0, 0.3377343 - Q.tau32_b2) / 9.465289e-05   # +0.0%  z_dr_0p05_0p1 < 0.01557 and tau32_b2 < 0.3377
        + 9.2738e-05 * max(0.0, -0.1170754 - Q.tdz_1) / 0.01260556   # +0.0%  tdz_1 < -0.1171
        - 5.590698e-05 * max(0.0, 0.7107791 - Q.D2) / 0.007444256   # -0.0%  D2 < 0.7108
        - 1.917812e-05 * max(0.0, Q.mass_2photon - 22.18431) / 0.4406114   # -0.0%  mass_2photon > 22.18
        + 4.857487e-06 * max(0.0, 13.03663 - Q.mass_displaced3) * max(0.0, Q.iselectron_1 - 0.0) / 0.1996362   # +0.0%  mass_displaced3 < 13.04 and iselectron_1 > 0
        + 1.627771e-06 * max(0.0, Q.tau3 - 0.06348273) * max(0.0, 0.0 - Q.phi_84) / 4.04563e-05   # +0.0%  tau3 > 0.06348 and phi_84 < 0
        - 1.35188e-06 * max(0.0, Q.mass_top50 - 125.4927) * max(0.0, 0.0 - Q.tdz_68) / 0.03907777   # -0.0%  mass_top50 > 125.5 and tdz_68 < 0
    )
    return z


def neuron_17(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.035383e-05
    )
    return z


def neuron_18(Q):
    # scale S = 11.38; each line: share * term / its average size
    z = 11.37574 * (0.06277493
        + 0.0741371 * Q.n_for_90pct / 20.33681   # +7.4%  n_for_90pct
        - 0.0458432 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # -4.6%  n_s3d_above_3 < 3
        - 0.04194018 * max(0.0, 0.004135872 - Q.lep_z) / 0.002390373   # -4.2%  lep_z < 0.004136
        + 0.03816007 * max(0.0, 3.0 - Q.n_s3d_above_3) * max(0.0, 117.0891 - Q.sip_3d_2) / 97.96476   # +3.8%  n_s3d_above_3 < 3 and sip_3d_2 < 117.1
        - 0.03571008 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -3.6%  lep_z < 0.2214
        - 0.03484076 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # -3.5%  e3_b2 < 0.000791
        + 0.0313055 * max(0.0, 9.0 - Q.n_sd0_above_3) / 5.844283   # +3.1%  n_sd0_above_3 < 9
        - 0.03053781 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.1046203 - Q.z_displaced5) / 0.01009417   # -3.1%  lep_z < 0.2214 and z_displaced5 < 0.1046
        + 0.03021188 * max(0.0, 27.0 - Q.n_charged_pt_above_1) / 8.874957   # +3.0%  n_charged_pt_above_1 < 27
        + 0.02924271 * max(0.0, 155.8928 - Q.mass_top40) / 48.12072   # +2.9%  mass_top40 < 155.9
        - 0.0287893 * max(0.0, 132.5189 - Q.mass_top40) / 28.13525   # -2.9%  mass_top40 < 132.5
        - 0.02442501 * max(0.0, Q.sd_mass - 123.2919) / 8.015693   # -2.4%  sd_mass > 123.3
        + 0.02227476 * max(0.0, 2.097213e-06 - Q.e4) / 1.474625e-06   # +2.2%  e4 < 2.097e-06
        - 0.02196067 * max(0.0, Q.n_neutral - 12.0) / 8.788217   # -2.2%  n_neutral > 12
        + 0.01903448 * max(0.0, 2.0 - Q.n_lepton) / 1.440613   # +1.9%  n_lepton < 2
        - 0.01853427 * max(0.0, 0.06416437 - Q.z_displaced3) / 0.02655242   # -1.9%  z_displaced3 < 0.06416
        - 0.01598119 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # -1.6%  lep_ptrel < 27.32
        + 0.01595644 * max(0.0, Q.mass_neutral - 14.92637) / 30.76836   # +1.6%  mass_neutral > 14.93
        + 0.01580991 * max(0.0, 0.06245248 - Q.z_displaced5) / 0.02875704   # +1.6%  z_displaced5 < 0.06245
        - 0.01564116 * max(0.0, 6.0 - Q.n_sd0_above_2) / 2.514897   # -1.6%  n_sd0_above_2 < 6
        - 0.01555998 * max(0.0, 0.3829918 - Q.N2) / 0.08586977   # -1.6%  N2 < 0.383
        - 0.01506996 * max(0.0, Q.mass - 110.2019) / 16.73354   # -1.5%  mass > 110.2
        - 0.01421045 * max(0.0, 114.4658 - Q.mass_top20) / 26.77686   # -1.4%  mass_top20 < 114.5
        - 0.01381088 * max(0.0, 6.0 - Q.n_s3d_above_10) / 3.631173   # -1.4%  n_s3d_above_10 < 6
        - 0.01325357 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, Q.N2_b2 - 0.0395449) / 2.959526   # -1.3%  lep_ptrel < 27.32 and N2_b2 > 0.03954
        + 0.01299165 * max(0.0, 0.0001729927 - Q.e3_b2) / 0.0001069897   # +1.3%  e3_b2 < 0.000173
        + 0.01266126 * max(0.0, 49.03695 - Q.sip_3d_3) / 32.30721   # +1.3%  sip_3d_3 < 49.04
        + 0.01237347 * max(0.0, Q.sd_mass - 111.701) / 11.78149   # +1.2%  sd_mass > 111.7
        + 0.01235793 * max(0.0, 14.38858 - Q.lep_iso) / 12.08589   # +1.2%  lep_iso < 14.39
        + 0.011833 * max(0.0, 0.3829918 - Q.N2) * max(0.0, -5.306837 - Q.lnptrel_45) / 0.9709755   # +1.2%  N2 < 0.383 and lnptrel_45 < -5.307
        + 0.01164208 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.3829918 - Q.N2) / 0.01292108   # +1.2%  lep_z < 0.2214 and N2 < 0.383
        - 0.0114165 * max(0.0, Q.tau43 - 0.6808537) / 0.1315878   # -1.1%  tau43 > 0.6809
        + 0.01140817 * max(0.0, 9.090532 - Q.mass_displaced3) * max(0.0, 29.0 - Q.n_charged_pt_above_1) / 66.69358   # +1.1%  mass_displaced3 < 9.091 and n_charged_pt_above_1 < 29
        + 0.01131125 * max(0.0, 111.0 - Q.n_pairs_kt_above_3) / 47.95244   # +1.1%  n_pairs_kt_above_3 < 111
        + 0.01111407 * max(0.0, 6.0 - Q.n_s3d_above_10) * max(0.0, Q.lund_max_lndelta - -1.670992) / 2.1779   # +1.1%  n_s3d_above_10 < 6 and lund_max_lndelta > -1.671
        + 0.01072887 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +1.1%  lep_iso < 0.4382
        - 0.01015067 * max(0.0, Q.n_pairs_kt_above_1 - 146.0) / 205.605   # -1.0%  n_pairs_kt_above_1 > 146
        + 0.01012604 * max(0.0, 0.6289062 - Q.max_abs_d0) / 0.1808136   # +1.0%  max_abs_d0 < 0.6289
        - 0.009879558 * max(0.0, 0.06245248 - Q.z_displaced5) * max(0.0, 4.636903 - Q.sip_3d_2) / 0.04573141   # -1.0%  z_displaced5 < 0.06245 and sip_3d_2 < 4.637
        + 0.00918924 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +0.9%  sd_mass > 154.6
        + 0.009170715 * max(0.0, Q.mass - 164.4374) / 3.038568   # +0.9%  mass > 164.4
        - 0.008779492 * max(0.0, Q.lund3_lndelta - -2.817283) * max(0.0, Q.tau54 - 0.7608692) / 0.1294652   # -0.9%  lund3_lndelta > -2.817 and tau54 > 0.7609
        + 0.008017634 * max(0.0, 3.0 - Q.n_sdz_above_5) / 1.349857   # +0.8%  n_sdz_above_5 < 3
        + 0.007988928 * max(0.0, 0.06245248 - Q.z_displaced5) * max(0.0, Q.n_pt_above_5 - 10.0) / 0.3067365   # +0.8%  z_displaced5 < 0.06245 and n_pt_above_5 > 10
        - 0.007089906 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -0.7%  sip_3d_2 < 226.3
        - 0.006683321 * max(0.0, Q.lund_max_lnkt - 3.22013) / 0.7220204   # -0.7%  lund_max_lnkt > 3.22
        - 0.006319706 * max(0.0, Q.tau32 - 0.5871757) / 0.1304199   # -0.6%  tau32 > 0.5872
        + 0.006158733 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +0.6%  lep_iso < 1.362
        + 0.00545241 * max(0.0, Q.N2_b05 - 0.4265629) / 0.03557833   # +0.5%  N2_b05 > 0.4266
        - 0.005382355 * max(0.0, 0.01091199 - Q.sum_z_dr2_top2) / 0.002905672   # -0.5%  sum_z_dr2_top2 < 0.01091
        - 0.00529718 * max(0.0, 9.090532 - Q.mass_displaced3) / 5.595081   # -0.5%  mass_displaced3 < 9.091
        + 0.005282759 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # +0.5%  n_pairs_kt_above_1 < 366
        - 0.004929242 * max(0.0, Q.z_displaced5 - 0.006242101) / 0.08840268   # -0.5%  z_displaced5 > 0.006242
        + 0.004586091 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +0.5%  n_dr_0p4_up < 4
        - 0.004386112 * max(0.0, 9.046875 - Q.max_abs_dz) / 5.120361   # -0.4%  max_abs_dz < 9.047
        + 0.004240004 * max(0.0, 6.0 - Q.n_s3d_above_10) * max(0.0, 0.212136 - Q.z_neutral_had) / 0.3191128   # +0.4%  n_s3d_above_10 < 6 and z_neutral_had < 0.2121
        - 0.003797915 * max(0.0, 3.373037 - Q.sip_3d_1) / 0.1802381   # -0.4%  sip_3d_1 < 3.373
        + 0.003788015 * max(0.0, 6.0 - Q.n_s3d_above_10) * max(0.0, Q.lne_7 - 2.313463) / 3.324212   # +0.4%  n_s3d_above_10 < 6 and lne_7 > 2.313
        - 0.003715551 * max(0.0, 0.03624058 - Q.z_displaced3) * max(0.0, Q.z_charged_had - 0.1891627) / 0.003966365   # -0.4%  z_displaced3 < 0.03624 and z_charged_had > 0.1892
        - 0.003702652 * max(0.0, 9.090532 - Q.mass_displaced3) * max(0.0, 0.114659 - Q.lep_dr) / 0.430453   # -0.4%  mass_displaced3 < 9.091 and lep_dr < 0.1147
        - 0.003385672 * max(0.0, Q.n_charged_had - 27.0) / 0.71757   # -0.3%  n_charged_had > 27
        + 0.003363364 * max(0.0, 5.311506 - Q.sj2_mass2) / 0.3937533   # +0.3%  sj2_mass2 < 5.312
        + 0.003175502 * max(0.0, Q.lund2_lndelta - -1.124734) / 0.274477   # +0.3%  lund2_lndelta > -1.125
        + 0.003143705 * max(0.0, 0.004135872 - Q.lep_z) * max(0.0, Q.jet_abs_eta - 0.8317778) / 0.0004322819   # +0.3%  lep_z < 0.004136 and jet_abs_eta > 0.8318
        + 0.003140566 * max(0.0, 0.3206143 - Q.pt1_over_pt0) / 0.01867984   # +0.3%  pt1_over_pt0 < 0.3206
        - 0.00299094 * max(0.0, 0.4545826 - Q.N3_b2) / 0.1894157   # -0.3%  N3_b2 < 0.4546
        - 0.00298086 * max(0.0, Q.mass - 182.8592) / 1.688248   # -0.3%  mass > 182.9
        - 0.0028302 * max(0.0, Q.sj3_pair_mass_max - 128.6605) / 2.418855   # -0.3%  sj3_pair_mass_max > 128.7
        - 0.002711561 * max(0.0, 2.097213e-06 - Q.e4) * max(0.0, 1.362094 - Q.lep_iso) / 1.442174e-06   # -0.3%  e4 < 2.097e-06 and lep_iso < 1.362
        - 0.002665596 * max(0.0, Q.tau32 - 0.5871757) * max(0.0, 4.0 - Q.n_pt_above_50) / 0.1810898   # -0.3%  tau32 > 0.5872 and n_pt_above_50 < 4
        - 0.002387439 * max(0.0, Q.lund_max_lndelta - -0.6897565) / 0.03634831   # -0.2%  lund_max_lndelta > -0.6898
        - 0.002301942 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.jet_charge_k05 - 0.1901233) / 0.01534374   # -0.2%  lep_z < 0.2214 and jet_charge_k05 > 0.1901
        - 0.002031234 * max(0.0, 0.07290954 - Q.sum_z_dr) / 0.002660742   # -0.2%  sum_z_dr < 0.07291
        - 0.001945543 * max(0.0, Q.sj3_pair_mass_min - 44.1643) / 6.009845   # -0.2%  sj3_pair_mass_min > 44.16
        - 0.001614111 * max(0.0, 0.3084865 - Q.sj3_dr23) / 0.04356127   # -0.2%  sj3_dr23 < 0.3085
        - 0.001533629 * max(0.0, 2.409613 - Q.mass_displaced3) / 1.070706   # -0.2%  mass_displaced3 < 2.41
        + 0.001440421 * max(0.0, Q.lund3_lndelta - -2.817283) / 1.325774   # +0.1%  lund3_lndelta > -2.817
        - 0.001051413 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.380649 - Q.pt1_over_pt0) / 0.002473162   # -0.1%  lep_z < 0.2214 and pt1_over_pt0 < 0.3806
        - 0.001021295 * max(0.0, 0.03624058 - Q.z_displaced3) / 0.01247855   # -0.1%  z_displaced3 < 0.03624
        + 0.00101874 * max(0.0, 0.3887012 - Q.z_charged) / 0.008149801   # +0.1%  z_charged < 0.3887
        + 0.0009326421 * max(0.0, Q.z_displaced5 - 0.006242101) * max(0.0, 29.0 - Q.n_dr_0p2_0p4) / 1.555704   # +0.1%  z_displaced5 > 0.006242 and n_dr_0p2_0p4 < 29
        - 0.0007641402 * max(0.0, Q.mass_top5 - 76.4159) / 1.748522   # -0.1%  mass_top5 > 76.42
        - 0.0006951862 * max(0.0, Q.n_lund - 13.0) / 0.37363   # -0.1%  n_lund > 13
        - 0.0006795159 * max(0.0, Q.sj3_pair_mass_max - 128.6605) * max(0.0, Q.eccentricity - 0.9132117) / 0.02034252   # -0.1%  sj3_pair_mass_max > 128.7 and eccentricity > 0.9132
        - 0.0006690201 * max(0.0, Q.sj3_pairmax_over_m - 0.897453) / 0.007530885   # -0.1%  sj3_pairmax_over_m > 0.8975
        + 0.0006131291 * max(0.0, Q.mass - 182.8592) * max(0.0, 0.4623202 - Q.planar_flow) / 0.1098624   # +0.1%  mass > 182.9 and planar_flow < 0.4623
        + 0.0005098848 * max(0.0, Q.z_displaced5 - 0.006242101) * max(0.0, 0.2116473 - Q.dr12) / 0.01073306   # +0.1%  z_displaced5 > 0.006242 and dr12 < 0.2116
        + 0.0004239691 * max(0.0, Q.mass_charged - 113.5019) / 1.042604   # +0.0%  mass_charged > 113.5
        + 0.0004080173 * max(0.0, 0.2598003 - Q.psi_0p2) / 0.007656233   # +0.0%  psi_0p2 < 0.2598
        + 0.0003043049 * max(0.0, Q.tau3 - 0.05398263) / 0.008920324   # +0.0%  tau3 > 0.05398
        - 0.0002531154 * max(0.0, Q.sj3_pair_mass_max - 128.6605) * max(0.0, -0.3624079 - Q.lund1_lndelta) / 0.1198998   # -0.0%  sj3_pair_mass_max > 128.7 and lund1_lndelta < -0.3624
        + 0.0002273057 * max(0.0, Q.z_displaced5 - 0.006242101) * max(0.0, 447.0873 - Q.sip_3d_2) / 24.18806   # +0.0%  z_displaced5 > 0.006242 and sip_3d_2 < 447.1
        + 0.000165993 * max(0.0, Q.n_neutral - 12.0) * max(0.0, Q.eta_28 - -0.1078491) / 1.228715   # +0.0%  n_neutral > 12 and eta_28 > -0.1078
        - 0.0001102866 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, Q.e4_b05 - 4.363663e-05) / 2.737046e-06   # -0.0%  lep_z < 0.2214 and e4_b05 > 4.364e-05
        + 9.680966e-05 * max(0.0, 0.3887012 - Q.z_charged) * max(0.0, Q.mass_2photon - 1.081989) / 0.06983535   # +0.0%  z_charged < 0.3887 and mass_2photon > 1.082
        - 9.675768e-05 * max(0.0, Q.sj2_mass2 - 42.12131) / 0.6609289   # -0.0%  sj2_mass2 > 42.12
        + 7.654875e-05 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, Q.eta_9 - 0.09516602) / 0.4726864   # +0.0%  lep_ptrel < 27.32 and eta_9 > 0.09517
        - 2.781782e-05 * max(0.0, Q.mass - 110.2019) * max(0.0, 8.0 - Q.n_charged_pt_above_10) / 11.56508   # -0.0%  mass > 110.2 and n_charged_pt_above_10 < 8
        + 1.784829e-05 * max(0.0, -0.2019043 - Q.eta_1) / 0.004072277   # +0.0%  eta_1 < -0.2019
        + 1.695059e-07 * max(0.0, Q.sj3_pairmax_over_m - 0.897453) * max(0.0, Q.iselectron_29 - 0.0) / 2.406724e-05   # +0.0%  sj3_pairmax_over_m > 0.8975 and iselectron_29 > 0
    )
    return z


def neuron_19(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.403498e-06
    )
    return z


def neuron_20(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.213595e-07
    )
    return z


def neuron_21(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.001682882
    )
    return z


def neuron_22(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.046941e-06
    )
    return z


def neuron_23(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.014224e-07
    )
    return z


def neuron_24(Q):
    # scale S = 15.1; each line: share * term / its average size
    z = 15.09688 * (-0.03797198
        - 0.07082732 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # -7.1%  sd_mass > 88.82
        + 0.0641594 * max(0.0, Q.sd_mass - 62.03827) / 42.41619   # +6.4%  sd_mass > 62.04
        - 0.05446941 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -5.4%  lep_z < 0.2214
        - 0.04726367 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -4.7%  mass < 164.4
        - 0.04177196 * max(0.0, 0.0004126585 - Q.e3_b2) / 0.0003132599   # -4.2%  e3_b2 < 0.0004127
        + 0.03689391 * max(0.0, 1.33105 - Q.N3_b05) / 0.6024977   # +3.7%  N3_b05 < 1.331
        - 0.03310553 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -3.3%  n_lepton < 1
        - 0.03215219 * max(0.0, 6.0 - Q.n_sd0_above_3) / 3.16582   # -3.2%  n_sd0_above_3 < 6
        + 0.0315199 * Q.pair_max_lnkt / 2.706794   # +3.2%  pair_max_lnkt
        + 0.02864925 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +2.9%  lep_iso < 0.4382
        + 0.02764486 * max(0.0, 114.0172 - Q.mass) / 13.82223   # +2.8%  mass < 114
        + 0.02704495 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +2.7%  lep_iso < 1.362
        + 0.02589207 * max(0.0, 115.5142 - Q.sj4_pair_mass_max) / 36.55147   # +2.6%  sj4_pair_mass_max < 115.5
        + 0.02422655 * max(0.0, 26.78691 - Q.mass_displaced3) / 20.43083   # +2.4%  mass_displaced3 < 26.79
        + 0.02134962 * max(0.0, 0.5944498 - Q.sj2_dr) * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 2.50037   # +2.1%  sj2_dr < 0.5944 and n_dr_0p2_0p4 < 21
        - 0.02026635 * max(0.0, Q.n_real_top30 - 21.0) / 7.215303   # -2.0%  n_real_top30 > 21
        + 0.02018963 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.lund2_lndelta - -1.411949) / 21.06341   # +2.0%  mass < 164.4 and lund2_lndelta > -1.412
        + 0.01983843 * max(0.0, Q.sd_mass - 119.4443) / 9.070113   # +2.0%  sd_mass > 119.4
        + 0.01786588 * max(0.0, 134.2224 - Q.mass_top30) / 33.8221   # +1.8%  mass_top30 < 134.2
        + 0.01700729 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 127.2317 - Q.mass) / 5.805704   # +1.7%  lep_z < 0.3397 and mass < 127.2
        + 0.01691881 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +1.7%  lep_ptrel < 43.21
        - 0.01666614 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # -1.7%  lep_z < 0.3397
        + 0.01601672 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) / 59.61773   # +1.6%  n_pairs_kt_above_3 < 126
        - 0.01598826 * max(0.0, 0.9644868 - Q.N3_b2) / 0.6211118   # -1.6%  N3_b2 < 0.9645
        - 0.01503416 * Q.n_sd0_above_2 / 4.174103   # -1.5%  n_sd0_above_2
        + 0.01491967 * Q.sum_e / 905.8561   # +1.5%  sum_e
        - 0.014392 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.lund2_lndelta - -1.755318) / 9.158987   # -1.4%  mass < 117.5 and lund2_lndelta > -1.755
        - 0.01371469 * max(0.0, 0.3923158 - Q.sj2_dr) / 0.06287263   # -1.4%  sj2_dr < 0.3923
        + 0.01177825 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) / 103.3282   # +1.2%  n_pairs_kt_above_1 < 325
        + 0.0109189 * max(0.0, 0.3995332 - Q.sd_rg) / 0.1037053   # +1.1%  sd_rg < 0.3995
        + 0.01071832 * max(0.0, 0.1061578 - Q.z_displaced3) / 0.05189409   # +1.1%  z_displaced3 < 0.1062
        + 0.009177883 * max(0.0, 2.0 - Q.n_s3d_above_3) / 0.4717633   # +0.9%  n_s3d_above_3 < 2
        - 0.008648619 * max(0.0, Q.n_s3d_above_3 - 2.0) / 2.23187   # -0.9%  n_s3d_above_3 > 2
        + 0.007311837 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +0.7%  n_dr_0p4_up < 4
        - 0.007175169 * max(0.0, Q.z_top15_slots - 0.8558319) / 0.03951754   # -0.7%  z_top15_slots > 0.8558
        - 0.007052246 * max(0.0, Q.C2 - 0.07037463) / 0.0805072   # -0.7%  C2 > 0.07037
        + 0.006940549 * max(0.0, Q.n_s3d_above_3 - 5.0) / 0.8834567   # +0.7%  n_s3d_above_3 > 5
        - 0.006400569 * max(0.0, Q.lund2_lndelta - -1.299999) / 0.3974116   # -0.6%  lund2_lndelta > -1.3
        - 0.00635004 * max(0.0, 117.4867 - Q.mass) / 15.64803   # -0.6%  mass < 117.5
        - 0.006302106 * max(0.0, Q.n_s3d_above_10 - 1.0) / 1.904323   # -0.6%  n_s3d_above_10 > 1
        - 0.006168866 * max(0.0, 75.78208 - Q.sj4_pair_mass_max) / 8.111726   # -0.6%  sj4_pair_mass_max < 75.78
        + 0.006157532 * max(0.0, Q.n_sd0_above_3 - 3.0) / 1.31993   # +0.6%  n_sd0_above_3 > 3
        - 0.00614206 * max(0.0, 0.258375 - Q.N2) / 0.01851814   # -0.6%  N2 < 0.2584
        + 0.005671577 * max(0.0, 0.2403736 - Q.sj2_dr) / 0.01094304   # +0.6%  sj2_dr < 0.2404
        - 0.005309058 * max(0.0, 0.221436 - Q.lep_z) * max(0.0, 0.2403736 - Q.sj2_dr) / 0.001837088   # -0.5%  lep_z < 0.2214 and sj2_dr < 0.2404
        - 0.005218656 * max(0.0, Q.psi_0p2 - 0.8857951) / 0.01931927   # -0.5%  psi_0p2 > 0.8858
        + 0.005209664 * max(0.0, 0.5944498 - Q.sj2_dr) / 0.2163159   # +0.5%  sj2_dr < 0.5944
        - 0.005020235 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # -0.5%  max_abs_d0 < 10.52
        + 0.004780088 * max(0.0, 117.4867 - Q.mass) * max(0.0, 3.038341 - Q.D2) / 14.74609   # +0.5%  mass < 117.5 and D2 < 3.038
        + 0.004699357 * max(0.0, 6.0 - Q.n_sd0_above_3) * max(0.0, Q.n_lund_kt_above_1 - 4.0) / 4.546673   # +0.5%  n_sd0_above_3 < 6 and n_lund_kt_above_1 > 4
        + 0.004621963 * max(0.0, 0.04544279 - Q.M2_b2) / 0.01635302   # +0.5%  M2_b2 < 0.04544
        - 0.004552679 * max(0.0, 79.47361 - Q.mass) / 2.857843   # -0.5%  mass < 79.47
        - 0.003740278 * max(0.0, 0.07094338 - Q.C2_b2) / 0.02378602   # -0.4%  C2_b2 < 0.07094
        - 0.003575598 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.4%  mass < 90.09
        - 0.00312935 * max(0.0, 74.51927 - Q.mass_top30) / 2.669461   # -0.3%  mass_top30 < 74.52
        - 0.00295356 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 5.969698e-05 - Q.ecf_g42) / 1.977438e-06   # -0.3%  z_displaced3 < 0.1062 and ecf_g42 < 5.97e-05
        + 0.002807691 * max(0.0, 149.0507 - Q.mass) / 39.08704   # +0.3%  mass < 149.1
        - 0.002777626 * max(0.0, 0.8799072 - Q.sj3_dr23) / 0.4603866   # -0.3%  sj3_dr23 < 0.8799
        - 0.002476359 * max(0.0, 26.78691 - Q.mass_displaced3) * max(0.0, Q.z_charged - 0.4622094) / 3.130524   # -0.2%  mass_displaced3 < 26.79 and z_charged > 0.4622
        - 0.002475551 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, Q.D2 - 2.446407) / 0.03497164   # -0.2%  z_displaced3 < 0.1062 and D2 > 2.446
        + 0.002409151 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, 0.3972884 - Q.pt_balance01) / 1.831735e-05   # +0.2%  e3_b2 < 0.0004127 and pt_balance01 < 0.3973
        + 0.002152554 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.M3 - 0.03259227) / 0.2883986   # +0.2%  lep_ptrel < 43.21 and M3 > 0.03259
        + 0.002105282 * max(0.0, 26.78691 - Q.mass_displaced3) * max(0.0, 0.06777369 - Q.dr_31) / 0.5598919   # +0.2%  mass_displaced3 < 26.79 and dr_31 < 0.06777
        - 0.00206077 * max(0.0, Q.N2_b05 - 0.4599592) / 0.01884027   # -0.2%  N2_b05 > 0.46
        + 0.001998152 * max(0.0, 0.3886647 - Q.sj3_mass3) / 0.03562942   # +0.2%  sj3_mass3 < 0.3887
        + 0.001503929 * max(0.0, 0.5068038 - Q.tau32) / 0.02583179   # +0.2%  tau32 < 0.5068
        - 0.001323534 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # -0.1%  sd_mass > 175.9
        - 0.001319613 * max(0.0, 74.51927 - Q.mass_top30) * max(0.0, 3.038341 - Q.D2) / 1.107918   # -0.1%  mass_top30 < 74.52 and D2 < 3.038
        - 0.001314708 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.n_muon - 0.0) / 5.859253   # -0.1%  lep_ptrel < 43.21 and n_muon > 0
        - 0.00114086 * max(0.0, 0.3945478 - Q.z_charged_had) / 0.03037343   # -0.1%  z_charged_had < 0.3945
        - 0.001135288 * max(0.0, Q.mass_displaced3 - 39.09615) / 0.6617151   # -0.1%  mass_displaced3 > 39.1
        + 0.001085048 * max(0.0, 90.08945 - Q.mass) * max(0.0, Q.lund2_lndelta - -2.073049) / 3.81955   # +0.1%  mass < 90.09 and lund2_lndelta > -2.073
        - 0.0009208646 * max(0.0, 2.0 - Q.n_sdz_above_5) / 0.7314567   # -0.1%  n_sdz_above_5 < 2
        - 0.0009135671 * max(0.0, Q.sj3_dr_min - 0.2410564) / 0.04091582   # -0.1%  sj3_dr_min > 0.2411
        + 0.0007312018 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.mass_charged - 90.44234) / 0.0003362549   # +0.1%  e3_b2 < 0.0004127 and mass_charged > 90.44
        + 0.0005343689 * max(0.0, Q.sj3_pair_mass_max - 142.9952) / 1.384637   # +0.1%  sj3_pair_mass_max > 143
        + 0.000522088 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, Q.mass_2photon - 4.18513) / 0.1670932   # +0.1%  z_displaced3 < 0.1062 and mass_2photon > 4.185
        + 0.0004815463 * max(0.0, 74.51927 - Q.mass_top30) * max(0.0, Q.n_muon - 0.0) / 0.3416725   # +0.0%  mass_top30 < 74.52 and n_muon > 0
        + 0.0004730643 * max(0.0, Q.n_sd0_above_10 - 6.0) / 0.15998   # +0.0%  n_sd0_above_10 > 6
        + 0.0004074858 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.e4 - 5.29882e-07) / 3.445168e-07   # +0.0%  lep_z < 0.3397 and e4 > 5.299e-07
        - 0.0004012362 * max(0.0, 0.2083105 - Q.sj2_dr) / 0.00695417   # -0.0%  sj2_dr < 0.2083
        + 0.0004010763 * max(0.0, Q.mass_top5 - 68.52153) / 2.713876   # +0.0%  mass_top5 > 68.52
        + 0.0003339119 * max(0.0, Q.mass_charged - 99.20396) / 2.066138   # +0.0%  mass_charged > 99.2
        + 0.0003115382 * max(0.0, 0.0004126585 - Q.e3_b2) * max(0.0, Q.min_pair_mass - 10.59762) / 7.086748e-05   # +0.0%  e3_b2 < 0.0004127 and min_pair_mass > 10.6
        - 0.0002989601 * max(0.0, Q.sj3_dr_min - 0.3526989) / 0.01394539   # -0.0%  sj3_dr_min > 0.3527
        - 0.0002835675 * max(0.0, 0.4044054 - Q.D2_b2) / 0.01564934   # -0.0%  D2_b2 < 0.4044
        - 0.0002672443 * max(0.0, 1.0 - Q.isnhad_1) / 0.82048   # -0.0%  isnhad_1 < 1
        + 0.0002124156 * max(0.0, Q.tdz_2 - -0.03532465) / 0.05855813   # +0.0%  tdz_2 > -0.03532
        + 0.0001976801 * max(0.0, 0.4680886 - Q.tau32_b2) / 0.05978538   # +0.0%  tau32_b2 < 0.4681
        + 0.0001559271 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.tdz_6 - 0.03675711) / 1.407383   # +0.0%  mass < 164.4 and tdz_6 > 0.03676
        + 0.0001176384 * max(0.0, 0.01040452 - Q.sum_z_dr2_top15) / 0.0007317661   # +0.0%  sum_z_dr2_top15 < 0.0104
        - 0.000100461 * max(0.0, Q.sd_mass - 88.81751) * max(0.0, 0.0 - Q.charge_0) / 6.688685   # -0.0%  sd_mass > 88.82 and charge_0 < 0
        + 9.392922e-05 * max(0.0, 1.262841 - Q.mass_displaced5) / 0.5499328   # +0.0%  mass_displaced5 < 1.263
        + 8.965694e-05 * max(0.0, 0.4044054 - Q.D2_b2) * max(0.0, Q.dr_31 - 0.02273044) / 0.001130568   # +0.0%  D2_b2 < 0.4044 and dr_31 > 0.02273
        - 4.82949e-05 * max(0.0, 2.0 - Q.n_sd0_above_10) / 0.7503067   # -0.0%  n_sd0_above_10 < 2
        + 3.836677e-05 * max(0.0, Q.lund2_lndelta - -1.299999) * max(0.0, -0.02698624 - Q.tdz_0) / 0.005244564   # +0.0%  lund2_lndelta > -1.3 and tdz_0 < -0.02699
        - 3.675034e-05 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.dr_10 - 0.3349968) / 0.7314664   # -0.0%  lep_ptrel < 43.21 and dr_10 > 0.335
        - 1.749936e-05 * max(0.0, Q.mass_charged - 99.20396) * max(0.0, Q.lnerel_82 - -18.42068) / 1.5218   # -0.0%  mass_charged > 99.2 and lnerel_82 > -18.42
        + 5.907744e-06 * max(0.0, Q.n_sd0_above_3 - 3.0) * max(0.0, Q.d0err_42 - 0.0) / 0.01390804   # +0.0%  n_sd0_above_3 > 3 and d0err_42 > 0
        - 4.002609e-06 * max(0.0, 75.78208 - Q.sj4_pair_mass_max) * max(0.0, Q.iselectron_24 - 0.0) / 0.02893548   # -0.0%  sj4_pair_mass_max < 75.78 and iselectron_24 > 0
    )
    return z


def neuron_25(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.03103402
    )
    return z


def neuron_26(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.228471e-06
    )
    return z


def neuron_27(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.307112e-06
    )
    return z


def neuron_28(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.621167e-06
    )
    return z


def neuron_29(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.182457e-06
    )
    return z


def neuron_30(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.59933e-06
    )
    return z


def neuron_31(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.507126e-06
    )
    return z


def neuron_32(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.44091e-05
    )
    return z


def neuron_33(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.569357e-06
    )
    return z


def neuron_34(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.388889e-05
    )
    return z


def neuron_35(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.74948e-06
    )
    return z


def neuron_36(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.829923e-06
    )
    return z


def neuron_37(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.35341e-06
    )
    return z


def neuron_38(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.413764e-06
    )
    return z


def neuron_39(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.275775e-06
    )
    return z


def neuron_40(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.81274e-07
    )
    return z


def neuron_41(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.064824e-08
    )
    return z


def neuron_42(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.777584e-06
    )
    return z


def neuron_43(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-9.412546e-06
    )
    return z


def neuron_44(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.328536e-06
    )
    return z


def neuron_45(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.852588e-06
    )
    return z


def neuron_46(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.357071e-07
    )
    return z


def neuron_47(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.525433e-07
    )
    return z


def neuron_48(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.789037e-06
    )
    return z


def neuron_49(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.241616e-06
    )
    return z


def neuron_50(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.17425e-06
    )
    return z


def neuron_51(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.007586e-06
    )
    return z


def neuron_52(Q):
    # scale S = 12.92; each line: share * term / its average size
    z = 12.91895 * (-0.07118707
        + 0.08453676 * max(0.0, 7.0 - Q.n_s3d_above_3) / 3.67411   # +8.5%  n_s3d_above_3 < 7
        - 0.06042522 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -6.0%  n_lepton < 1
        + 0.05885875 * max(0.0, Q.mass - 71.96396) / 44.90212   # +5.9%  mass > 71.96
        - 0.04902382 * max(0.0, 8.0 - Q.n_sd0_above_3) / 4.91424   # -4.9%  n_sd0_above_3 < 8
        - 0.0469919 * max(0.0, 123.2919 - Q.sd_mass) / 34.26699   # -4.7%  sd_mass < 123.3
        + 0.04547464 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +4.5%  lep_iso < 0.4382
        - 0.03649272 * max(0.0, 7.0 - Q.n_s3d_above_3) * max(0.0, 175.9957 - Q.sip_3d_3) / 598.947   # -3.6%  n_s3d_above_3 < 7 and sip_3d_3 < 176
        + 0.03251334 * max(0.0, 0.5626523 - Q.LHA) / 0.1600511   # +3.3%  LHA < 0.5627
        - 0.02995763 * max(0.0, 7.0 - Q.n_s3d_above_3) * max(0.0, 70.05844 - Q.sip_3d_2) / 185.9687   # -3.0%  n_s3d_above_3 < 7 and sip_3d_2 < 70.06
        + 0.0294507 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 18.80005 - Q.mass_displaced3) / 190.788   # +2.9%  lep_ptrel < 18.77 and mass_displaced3 < 18.8
        + 0.02657898 * max(0.0, 83.61981 - Q.sd_mass) / 13.06909   # +2.7%  sd_mass < 83.62
        - 0.02573571 * max(0.0, 18.7678 - Q.lep_ptrel) / 14.28156   # -2.6%  lep_ptrel < 18.77
        - 0.02551959 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # -2.6%  lep_z < 0.2214
        + 0.02315151 * max(0.0, 44.14912 - Q.lep_iso) / 39.77068   # +2.3%  lep_iso < 44.15
        + 0.02139298 * max(0.0, 161.1264 - Q.mass_top50) / 50.68933   # +2.1%  mass_top50 < 161.1
        + 0.02018126 * max(0.0, 154.5947 - Q.sd_mass) / 60.90029   # +2.0%  sd_mass < 154.6
        + 0.02008861 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # +2.0%  lep_ptrel < 27.32
        + 0.0190534 * max(0.0, 0.07687439 - Q.tdz_0) / 0.08526919   # +1.9%  tdz_0 < 0.07687
        - 0.01808775 * max(0.0, 0.04229114 - Q.C3_b2) / 0.03492918   # -1.8%  C3_b2 < 0.04229
        - 0.01754317 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, 0.08643515 - Q.tau3) / 0.8197384   # -1.8%  lep_ptrel < 27.32 and tau3 < 0.08644
        + 0.01632614 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 4.606241 - Q.sip_3d_3) / 0.1354168   # +1.6%  z_displaced3 < 0.1343 and sip_3d_3 < 4.606
        - 0.015262 * Q.n_pairs_kt_above_1 / 331.4992   # -1.5%  n_pairs_kt_above_1
        - 0.01127151 * max(0.0, Q.mass - 90.08945) / 29.81377   # -1.1%  mass > 90.09
        + 0.01071807 * max(0.0, 3.373037 - Q.sip_3d_1) / 0.1802381   # +1.1%  sip_3d_1 < 3.373
        - 0.009398471 * max(0.0, 1.362094 - Q.lep_iso) * max(0.0, 49.03695 - Q.sip_3d_3) / 32.75876   # -0.9%  lep_iso < 1.362 and sip_3d_3 < 49.04
        - 0.009381917 * max(0.0, 0.06603871 - Q.tau2) / 0.01287286   # -0.9%  tau2 < 0.06604
        + 0.009300064 * max(0.0, 129.5874 - Q.mass_top50) * max(0.0, 0.114659 - Q.lep_dr) / 1.914075   # +0.9%  mass_top50 < 129.6 and lep_dr < 0.1147
        - 0.009125019 * max(0.0, Q.mass - 90.08945) * max(0.0, 0.7269473 - Q.tau32_b2) / 7.667197   # -0.9%  mass > 90.09 and tau32_b2 < 0.7269
        + 0.009038984 * max(0.0, 0.3788785 - Q.z_neutral_had) * max(0.0, Q.ecf_g41 - 6.216954e-05) / 3.09401e-05   # +0.9%  z_neutral_had < 0.3789 and ecf_g41 > 6.217e-05
        - 0.009032422 * max(0.0, 117.4867 - Q.mass) * max(0.0, 0.3565533 - Q.tau21_b2) / 1.536552   # -0.9%  mass < 117.5 and tau21_b2 < 0.3566
        - 0.008706396 * max(0.0, 0.0 - Q.tdz_0) / 0.01858885   # -0.9%  tdz_0 < 0
        - 0.008429943 * max(0.0, 42.31629 - Q.sd_mass) / 4.001873   # -0.8%  sd_mass < 42.32
        + 0.008270733 * max(0.0, 8.0 - Q.n_dr_0_0p05) / 5.25296   # +0.8%  n_dr_0_0p05 < 8
        + 0.008065707 * max(0.0, 45.26581 - Q.sip_3d_2) / 21.48394   # +0.8%  sip_3d_2 < 45.27
        - 0.007606547 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 3.113281 - Q.max_abs_d0) / 0.1236147   # -0.8%  z_displaced3 < 0.1343 and max_abs_d0 < 3.113
        + 0.007589362 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +0.8%  n_pairs_kt_above_1 < 80
        - 0.007519444 * max(0.0, 117.4867 - Q.mass) / 15.64803   # -0.8%  mass < 117.5
        + 0.00737732 * max(0.0, 0.09209404 - Q.C2_b2) * max(0.0, 1.462588 - Q.D3_b2) / 0.04100572   # +0.7%  C2_b2 < 0.09209 and D3_b2 < 1.463
        + 0.006841052 * max(0.0, Q.z_top10_slots - 0.7362734) / 0.06667364   # +0.7%  z_top10_slots > 0.7363
        + 0.006220477 * max(0.0, 97.52633 - Q.sj3_pair_mass_max) / 14.66667   # +0.6%  sj3_pair_mass_max < 97.53
        + 0.005945124 * Q.n_charged_had / 18.52697   # +0.6%  n_charged_had
        - 0.005678825 * max(0.0, 2.802481 - Q.sip_3d_1) / 0.07766626   # -0.6%  sip_3d_1 < 2.802
        - 0.005436916 * max(0.0, 0.3788785 - Q.z_neutral_had) / 0.2321329   # -0.5%  z_neutral_had < 0.3789
        + 0.005312006 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 4.636903 - Q.sip_3d_2) / 0.2196467   # +0.5%  lep_z < 0.3397 and sip_3d_2 < 4.637
        + 0.005119879 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # +0.5%  lep_iso < 1.362
        + 0.004570579 * max(0.0, 0.5626523 - Q.LHA) * max(0.0, 26.78691 - Q.mass_displaced3) / 3.494642   # +0.5%  LHA < 0.5627 and mass_displaced3 < 26.79
        - 0.004488875 * max(0.0, 4.0 - Q.n_sd0_above_3) / 1.660297   # -0.4%  n_sd0_above_3 < 4
        + 0.004440043 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, Q.N2 - 0.1824346) / 0.009842026   # +0.4%  z_displaced3 < 0.1343 and N2 > 0.1824
        + 0.004402366 * max(0.0, 0.09209404 - Q.C2_b2) / 0.0375889   # +0.4%  C2_b2 < 0.09209
        - 0.00425354 * max(0.0, 1.008706e-05 - Q.e3_b2) / 1.503587e-06   # -0.4%  e3_b2 < 1.009e-05
        + 0.004203671 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # +0.4%  n_pairs_kt_above_3 < 28
        - 0.003735421 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # -0.4%  lep_z < 0.3397
        - 0.003726643 * max(0.0, Q.jet_charge_k03 - -0.1486714) / 0.3723647   # -0.4%  jet_charge_k03 > -0.1487
        + 0.003561767 * max(0.0, 1.0 - Q.n_lepton) * max(0.0, 11.13866 - Q.sip_3d_3) / 3.098729   # +0.4%  n_lepton < 1 and sip_3d_3 < 11.14
        - 0.003176281 * max(0.0, Q.jet_abs_eta - 0.8317778) / 0.1684788   # -0.3%  jet_abs_eta > 0.8318
        + 0.00312833 * max(0.0, Q.n_particles - 38.0) / 6.78515   # +0.3%  n_particles > 38
        + 0.00310108 * max(0.0, 0.8035262 - Q.mass_displaced5) / 0.3308315   # +0.3%  mass_displaced5 < 0.8035
        - 0.002931054 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, 0.3201841 - Q.mass_displaced5) / 0.04106726   # -0.3%  lep_iso < 0.4382 and mass_displaced5 < 0.3202
        + 0.002861844 * max(0.0, Q.N2_b2 - 0.2392833) / 0.0118944   # +0.3%  N2_b2 > 0.2393
        - 0.002664108 * max(0.0, Q.lep_ptrel - 12.15228) / 3.983062   # -0.3%  lep_ptrel > 12.15
        - 0.002654328 * max(0.0, Q.N3 - 0.2681212) / 0.3845471   # -0.3%  N3 > 0.2681
        + 0.002611033 * max(0.0, 26.78691 - Q.mass_displaced3) * max(0.0, Q.n_lund_kt_above_1 - 4.0) / 31.51463   # +0.3%  mass_displaced3 < 26.79 and n_lund_kt_above_1 > 4
        + 0.002580004 * max(0.0, 65.0404 - Q.mass_charged) / 11.01768   # +0.3%  mass_charged < 65.04
        + 0.002396542 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, -0.01644749 - Q.tdz_0) / 0.003896084   # +0.2%  lep_z < 0.3397 and tdz_0 < -0.01645
        + 0.002180388 * max(0.0, Q.lund_max_lndelta - -0.615738) / 0.02343634   # +0.2%  lund_max_lndelta > -0.6157
        + 0.002177039 * max(0.0, 26.78691 - Q.mass_displaced3) / 20.43083   # +0.2%  mass_displaced3 < 26.79
        - 0.002061669 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.sj3_pair_mass_min - 48.40547) / 1.423551   # -0.2%  lep_z < 0.3397 and sj3_pair_mass_min > 48.41
        - 0.001768618 * max(0.0, 0.1342762 - Q.z_displaced3) / 0.07089169   # -0.2%  z_displaced3 < 0.1343
        - 0.001692573 * max(0.0, Q.mass - 90.08945) * max(0.0, 0.6818924 - Q.tau43_b2) / 2.346852   # -0.2%  mass > 90.09 and tau43_b2 < 0.6819
        - 0.001605767 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, Q.lep_dr - 0.114659) / 0.002466724   # -0.2%  z_displaced3 < 0.1343 and lep_dr > 0.1147
        + 0.001599243 * max(0.0, 1.851735 - Q.sj2_mass2) / 0.06959762   # +0.2%  sj2_mass2 < 1.852
        + 0.00151898 * max(0.0, 0.04229114 - Q.C3_b2) * max(0.0, 0.2816529 - Q.pt2_over_pt0) / 0.0008006262   # +0.2%  C3_b2 < 0.04229 and pt2_over_pt0 < 0.2817
        - 0.001393579 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 0.1905794 - Q.dr_21) / 0.7483514   # -0.1%  lep_ptrel < 18.77 and dr_21 < 0.1906
        - 0.001356497 * max(0.0, 129.5874 - Q.mass_top50) / 24.20267   # -0.1%  mass_top50 < 129.6
        - 0.001283989 * max(0.0, 56.73313 - Q.sj3_pair_mass_max) / 1.542034   # -0.1%  sj3_pair_mass_max < 56.73
        - 0.001183123 * max(0.0, Q.sj3_mass1 - 36.36236) / 1.189826   # -0.1%  sj3_mass1 > 36.36
        - 0.001085528 * max(0.0, 0.2848657 - Q.sj2_dr) / 0.01978141   # -0.1%  sj2_dr < 0.2849
        - 0.00105793 * max(0.0, 0.3788785 - Q.z_neutral_had) * max(0.0, 0.01100159 - Q.d0err_3) / 0.0009171798   # -0.1%  z_neutral_had < 0.3789 and d0err_3 < 0.011
        - 0.0009124127 * max(0.0, 101.849 - Q.sj4_pair_mass_max) / 24.8879   # -0.1%  sj4_pair_mass_max < 101.8
        - 0.0008178951 * max(0.0, 0.3951525 - Q.tau32) / 0.009206105   # -0.1%  tau32 < 0.3952
        + 0.0008168921 * max(0.0, 129.5874 - Q.mass_top50) * max(0.0, 0.0632362 - Q.dr_21) / 0.40919   # +0.1%  mass_top50 < 129.6 and dr_21 < 0.06324
        - 0.0007688638 * max(0.0, 0.1342762 - Q.z_displaced3) * max(0.0, 0.04063514 - Q.dr_21) / 0.0004216547   # -0.1%  z_displaced3 < 0.1343 and dr_21 < 0.04064
        + 0.0006375628 * max(0.0, 0.02600452 - Q.z_displaced3) / 0.00810998   # +0.1%  z_displaced3 < 0.026
        + 0.0006182925 * max(0.0, 79.27954 - Q.mass_top50) / 2.845865   # +0.1%  mass_top50 < 79.28
        - 0.0005486069 * max(0.0, 5.511295 - Q.pair_max_lnm2) / 0.0518722   # -0.1%  pair_max_lnm2 < 5.511
        - 0.0005419671 * max(0.0, 0.09209404 - Q.C2_b2) * max(0.0, Q.n_muon - 0.0) / 0.008709924   # -0.1%  C2_b2 < 0.09209 and n_muon > 0
        + 0.0004729623 * max(0.0, Q.sj2_mass1 - 91.2852) / 1.019309   # +0.0%  sj2_mass1 > 91.29
        - 0.0004039793 * max(0.0, 0.09209404 - Q.C2_b2) * max(0.0, Q.sip_3d_2 - 3.041925) / 13.63278   # -0.0%  C2_b2 < 0.09209 and sip_3d_2 > 3.042
        + 0.0003954125 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.mass_2photon - 3.013) / 4.662811   # +0.0%  n_pairs_kt_above_3 < 28 and mass_2photon > 3.013
        - 0.0003364017 * max(0.0, 0.1238959 - Q.C2) / 0.01477465   # -0.0%  C2 < 0.1239
        + 0.0002858931 * max(0.0, Q.sj3_pair_mass_min - 80.02563) / 0.702582   # +0.0%  sj3_pair_mass_min > 80.03
        + 0.0002447471 * max(0.0, Q.n_neutral_had - 7.0) / 0.21307   # +0.0%  n_neutral_had > 7
        - 0.0001729204 * max(0.0, Q.mass_top15 - 94.94813) * max(0.0, Q.sj3_dr13 - 0.2640447) / 1.729593   # -0.0%  mass_top15 > 94.95 and sj3_dr13 > 0.264
        + 0.0001558971 * max(0.0, Q.mass - 90.08945) * max(0.0, Q.td0_13 - -0.06278656) / 2.625608   # +0.0%  mass > 90.09 and td0_13 > -0.06279
        - 0.0001228066 * max(0.0, Q.mass - 90.08945) * max(0.0, 5.0 - Q.n_s3d_above_3) / 43.00566   # -0.0%  mass > 90.09 and n_s3d_above_3 < 5
        - 0.0001046 * max(0.0, Q.mass_top15 - 94.94813) / 6.576712   # -0.0%  mass_top15 > 94.95
        + 5.839552e-05 * max(0.0, Q.sj3_pair_mass_min - 80.02563) * max(0.0, 20.75111 - Q.sip_3d_2) / 2.43357   # +0.0%  sj3_pair_mass_min > 80.03 and sip_3d_2 < 20.75
        - 5.719722e-05 * max(0.0, 0.04276413 - Q.M2) / 0.0009967452   # -0.0%  M2 < 0.04276
        - 2.915145e-05 * max(0.0, 0.3951525 - Q.tau32) * max(0.0, Q.d0err_14 - 0.01499939) / 4.887181e-05   # -0.0%  tau32 < 0.3952 and d0err_14 > 0.015
        - 1.19409e-05 * max(0.0, Q.mass - 90.08945) * max(0.0, Q.d0err_75 - 0.0) / 0.03680061   # -0.0%  mass > 90.09 and d0err_75 > 0
    )
    return z


def neuron_53(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.888975e-06
    )
    return z


def neuron_54(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.190794e-05
    )
    return z


def neuron_55(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.067479e-05
    )
    return z


def neuron_56(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.043094e-07
    )
    return z


def neuron_57(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.964853e-06
    )
    return z


def neuron_58(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.616062e-06
    )
    return z


def neuron_59(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (6.030878e-06
    )
    return z


def neuron_60(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.66871e-06
    )
    return z


def neuron_61(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.836067e-06
    )
    return z


def neuron_62(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.460825e-07
    )
    return z


def neuron_63(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.312421e-06
    )
    return z


def neuron_64(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.252446e-06
    )
    return z


def neuron_65(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.790296e-06
    )
    return z


def neuron_66(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.71163e-06
    )
    return z


def neuron_67(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.214326e-06
    )
    return z


def neuron_68(Q):
    # scale S = 15.19; each line: share * term / its average size
    z = 15.19016 * (0.2460062
        - 0.1259871 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -12.6%  mass < 164.4
        - 0.09460128 * max(0.0, Q.sd_mass - 83.61981) / 26.48984   # -9.5%  sd_mass > 83.62
        - 0.07148003 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) / 44.19386   # -7.1%  sj3_pair_mass_min < 80.03
        + 0.06721176 * max(0.0, Q.sd_mass - 42.31629) / 58.72614   # +6.7%  sd_mass > 42.32
        - 0.05032446 * max(0.0, Q.n_pt_above_1 - 17.0) / 21.33585   # -5.0%  n_pt_above_1 > 17
        - 0.04921958 * max(0.0, Q.mass - 79.47361) / 38.31536   # -4.9%  mass > 79.47
        + 0.0414893 * max(0.0, 120.6471 - Q.sj3_pair_mass_max) / 31.70826   # +4.1%  sj3_pair_mass_max < 120.6
        + 0.03687256 * max(0.0, 114.0172 - Q.mass) / 13.82223   # +3.7%  mass < 114
        + 0.0331057 * max(0.0, Q.sj3_pair_mass_max - 44.2029) / 48.7862   # +3.3%  sj3_pair_mass_max > 44.2
        - 0.03159418 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # -3.2%  lep_iso < 0.4382
        + 0.03040155 * max(0.0, 2.0 - Q.n_lepton) / 1.440613   # +3.0%  n_lepton < 2
        - 0.02695195 * Q.sj3_pairmin_over_m / 0.3112428   # -2.7%  sj3_pairmin_over_m
        + 0.02589806 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # +2.6%  sd_mass > 88.82
        - 0.02430054 * max(0.0, Q.n_charged_had - 13.0) / 6.4188   # -2.4%  n_charged_had > 13
        + 0.02164124 * max(0.0, Q.sd_mass - 127.8042) / 7.009137   # +2.2%  sd_mass > 127.8
        + 0.01972315 * max(0.0, 0.7981752 - Q.tau32) / 0.1480334   # +2.0%  tau32 < 0.7982
        - 0.01680924 * max(0.0, 0.04544279 - Q.M2_b2) / 0.01635302   # -1.7%  M2_b2 < 0.04544
        - 0.01510988 * max(0.0, 0.3261071 - Q.z_displaced3) * max(0.0, 0.4191372 - Q.lep_dr) / 0.0789782   # -1.5%  z_displaced3 < 0.3261 and lep_dr < 0.4191
        + 0.01438368 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # +1.4%  n_lepton < 1
        + 0.01365746 * max(0.0, Q.n_pairs_kt_above_1 - 101.0) / 239.3315   # +1.4%  n_pairs_kt_above_1 > 101
        - 0.01192957 * max(0.0, 127.2317 - Q.mass) / 21.76326   # -1.2%  mass < 127.2
        - 0.009500557 * max(0.0, 3.38061 - Q.pair_mean_lnm2) / 0.9985192   # -1.0%  pair_mean_lnm2 < 3.381
        - 0.008741437 * max(0.0, 1.323111 - Q.jet_abs_eta) / 0.6200087   # -0.9%  jet_abs_eta < 1.323
        - 0.008552166 * max(0.0, Q.n_pt_above_1 - 17.0) * max(0.0, 0.1061578 - Q.z_displaced3) / 1.056777   # -0.9%  n_pt_above_1 > 17 and z_displaced3 < 0.1062
        - 0.007087217 * max(0.0, 90.08945 - Q.mass) * max(0.0, 1.0 - Q.n_lepton) / 3.361761   # -0.7%  mass < 90.09 and n_lepton < 1
        + 0.006878643 * max(0.0, Q.n_pt_above_1 - 17.0) * max(0.0, 0.8939856 - Q.tau43) / 2.073009   # +0.7%  n_pt_above_1 > 17 and tau43 < 0.894
        + 0.006407254 * max(0.0, Q.mass_top10 - 27.42853) / 40.59191   # +0.6%  mass_top10 > 27.43
        + 0.006169614 * max(0.0, 3.38061 - Q.pair_mean_lnm2) * max(0.0, 0.3788785 - Q.z_neutral_had) / 0.2490716   # +0.6%  pair_mean_lnm2 < 3.381 and z_neutral_had < 0.3789
        + 0.0059822 * max(0.0, 114.0172 - Q.mass) * max(0.0, 0.114659 - Q.lep_dr) / 1.144546   # +0.6%  mass < 114 and lep_dr < 0.1147
        - 0.005980817 * max(0.0, 0.7223231 - Q.z_charged_had) / 0.2225062   # -0.6%  z_charged_had < 0.7223
        - 0.005877136 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.6%  mass < 90.09
        - 0.005488295 * max(0.0, 2.286291 - Q.D2_b2) / 0.7248948   # -0.5%  D2_b2 < 2.286
        + 0.005405092 * max(0.0, 0.3261071 - Q.z_displaced3) * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 0.2634324   # +0.5%  z_displaced3 < 0.3261 and n_lund_kt_above_5 > 1
        + 0.005379634 * max(0.0, 3.0 - Q.n_sdz_above_5) / 1.349857   # +0.5%  n_sdz_above_5 < 3
        - 0.004879015 * max(0.0, 1.0 - Q.n_lepton) * max(0.0, 4.636903 - Q.sip_3d_2) / 0.5344861   # -0.5%  n_lepton < 1 and sip_3d_2 < 4.637
        - 0.004604506 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # -0.5%  sd_mass > 154.6
        + 0.00454157 * max(0.0, 0.04544279 - Q.M2_b2) * max(0.0, 76.15079 - Q.mass_neutral) / 0.6214169   # +0.5%  M2_b2 < 0.04544 and mass_neutral < 76.15
        - 0.004281168 * max(0.0, Q.lep_ptrel - 6.983043) / 5.140178   # -0.4%  lep_ptrel > 6.983
        - 0.004255224 * max(0.0, 0.3261071 - Q.z_displaced3) * max(0.0, Q.n_lund - 9.0) / 0.5293194   # -0.4%  z_displaced3 < 0.3261 and n_lund > 9
        + 0.003606445 * max(0.0, 2.382955 - Q.mass_displaced5) / 1.169903   # +0.4%  mass_displaced5 < 2.383
        + 0.003365286 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # +0.3%  n_pairs_kt_above_1 < 80
        + 0.003141656 * max(0.0, Q.n_pt_above_1 - 17.0) * max(0.0, 0.8786609 - Q.tau54) / 0.8588641   # +0.3%  n_pt_above_1 > 17 and tau54 < 0.8787
        + 0.00290977 * max(0.0, 0.3261071 - Q.z_displaced3) / 0.2272573   # +0.3%  z_displaced3 < 0.3261
        - 0.002795972 * max(0.0, 0.7981752 - Q.tau32) * max(0.0, 0.7223231 - Q.z_charged_had) / 0.03607804   # -0.3%  tau32 < 0.7982 and z_charged_had < 0.7223
        + 0.002613086 * max(0.0, 0.04544279 - Q.M2_b2) * max(0.0, Q.mass_displaced3 - 0.0) / 0.1412916   # +0.3%  M2_b2 < 0.04544 and mass_displaced3 > 0
        - 0.002541823 * max(0.0, Q.n_dr_0p2_0p4 - 9.0) / 4.79296   # -0.3%  n_dr_0p2_0p4 > 9
        - 0.002154188 * max(0.0, 80.36029 - Q.sj3_pair_mass_max) / 6.418891   # -0.2%  sj3_pair_mass_max < 80.36
        - 0.002106877 * max(0.0, 114.0172 - Q.mass) * max(0.0, Q.n_lund_kt_above_1 - 3.0) / 23.65019   # -0.2%  mass < 114 and n_lund_kt_above_1 > 3
        - 0.00209486 * max(0.0, 0.7981752 - Q.tau32) * max(0.0, Q.sj3_dr_min - 0.3526989) / 0.003071098   # -0.2%  tau32 < 0.7982 and sj3_dr_min > 0.3527
        - 0.001990448 * max(0.0, Q.psi_0p1 - 0.835544) / 0.01134109   # -0.2%  psi_0p1 > 0.8355
        + 0.001921847 * max(0.0, 2.317124 - Q.sip_3d_3) / 0.1434495   # +0.2%  sip_3d_3 < 2.317
        - 0.00191575 * max(0.0, 56.73313 - Q.sj3_pair_mass_max) / 1.542034   # -0.2%  sj3_pair_mass_max < 56.73
        + 0.001839517 * max(0.0, 0.002792418 - Q.sum_z_dr2_top3) / 0.0002360284   # +0.2%  sum_z_dr2_top3 < 0.002792
        + 0.001790705 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) * max(0.0, Q.sj3_mass1 - 18.73268) / 180.6276   # +0.2%  sj3_pair_mass_min < 80.03 and sj3_mass1 > 18.73
        - 0.001731023 * max(0.0, Q.n_s3d_above_3 - 5.0) / 0.8834567   # -0.2%  n_s3d_above_3 > 5
        + 0.001656469 * max(0.0, Q.e4_b05 - 2.879843e-05) / 2.220464e-05   # +0.2%  e4_b05 > 2.88e-05
        - 0.001639061 * max(0.0, 0.7981752 - Q.tau32) * max(0.0, 0.3190414 - Q.sj3_dr_min) / 0.01394611   # -0.2%  tau32 < 0.7982 and sj3_dr_min < 0.319
        + 0.001537937 * max(0.0, 0.9506259 - Q.mass_2charged) / 0.1206772   # +0.2%  mass_2charged < 0.9506
        + 0.001509271 * max(0.0, 127.2317 - Q.mass) * max(0.0, Q.sd_zg - 0.1900649) / 2.080143   # +0.2%  mass < 127.2 and sd_zg > 0.1901
        - 0.001485195 * max(0.0, 47.41085 - Q.mass_top5) / 14.28312   # -0.1%  mass_top5 < 47.41
        + 0.00147511 * max(0.0, 0.1036778 - Q.z_dr_0p1_0p2) / 0.02011467   # +0.1%  z_dr_0p1_0p2 < 0.1037
        - 0.001329399 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -0.1%  sip_3d_2 < 226.3
        - 0.001247632 * max(0.0, 3.0 - Q.n_sdz_above_5) * max(0.0, Q.sj2_dr - 0.3220633) / 0.1138968   # -0.1%  n_sdz_above_5 < 3 and sj2_dr > 0.3221
        - 0.001190562 * max(0.0, Q.tau21_b2 - 0.5211946) / 0.02080308   # -0.1%  tau21_b2 > 0.5212
        - 0.001183243 * max(0.0, Q.sd_mass - 88.81751) * max(0.0, Q.M3 - 0.02540381) / 0.1244951   # -0.1%  sd_mass > 88.82 and M3 > 0.0254
        + 0.001144094 * max(0.0, Q.lep_ptrel - 6.983043) * max(0.0, 0.1404188 - Q.C2_b2) / 0.4062449   # +0.1%  lep_ptrel > 6.983 and C2_b2 < 0.1404
        - 0.001023812 * max(0.0, 0.01040452 - Q.sum_z_dr2_top15) / 0.0007317661   # -0.1%  sum_z_dr2_top15 < 0.0104
        + 0.0010015 * max(0.0, 3.38061 - Q.pair_mean_lnm2) * max(0.0, Q.z_top30_slots - 0.8316085) / 0.1174666   # +0.1%  pair_mean_lnm2 < 3.381 and z_top30_slots > 0.8316
        + 0.0009328005 * max(0.0, Q.n_pt_above_5 - 25.0) / 1.700337   # +0.1%  n_pt_above_5 > 25
        - 0.0009006257 * max(0.0, Q.mass_top15 - 119.5993) / 2.048216   # -0.1%  mass_top15 > 119.6
        + 0.0008974505 * max(0.0, 28.0 - Q.n_pairs_kt_above_3) / 3.678273   # +0.1%  n_pairs_kt_above_3 < 28
        + 0.0007902724 * max(0.0, Q.n_electron - 1.0) / 0.08167   # +0.1%  n_electron > 1
        - 0.0007883871 * max(0.0, 0.03259227 - Q.M3) / 0.00464829   # -0.1%  M3 < 0.03259
        - 0.0007602234 * max(0.0, Q.z_displaced5 - 0.2801368) / 0.0126536   # -0.1%  z_displaced5 > 0.2801
        - 0.0006725046 * max(0.0, Q.lep_z - 0.5187302) / 0.007316078   # -0.1%  lep_z > 0.5187
        + 0.000576512 * max(0.0, 0.3261071 - Q.z_displaced3) * max(0.0, Q.mass_2photon - 9.82024) / 0.4109246   # +0.1%  z_displaced3 < 0.3261 and mass_2photon > 9.82
        - 0.000490271 * max(0.0, Q.mass_over_sum_pt_sq - 0.0729277) / 0.002813812   # -0.0%  mass_over_sum_pt_sq > 0.07293
        - 0.0004832276 * max(0.0, 1.323111 - Q.jet_abs_eta) * max(0.0, Q.n_neutral_had - 6.0) / 0.1578672   # -0.0%  jet_abs_eta < 1.323 and n_neutral_had > 6
        + 0.0004575117 * max(0.0, Q.mass_top30 - 162.7874) / 1.306268   # +0.0%  mass_top30 > 162.8
        - 0.0004395975 * max(0.0, Q.mass_2photon - 22.18431) / 0.4406114   # -0.0%  mass_2photon > 22.18
        - 0.0003811399 * max(0.0, Q.C2_b05 - 0.3533901) / 0.003974948   # -0.0%  C2_b05 > 0.3534
        - 0.0003439425 * max(0.0, Q.sd_mass - 42.31629) * max(0.0, Q.sj3_dr13 - 0.6229991) / 1.427562   # -0.0%  sd_mass > 42.32 and sj3_dr13 > 0.623
        - 0.0003257403 * max(0.0, 0.03378303 - Q.tau3) / 0.003783449   # -0.0%  tau3 < 0.03378
        - 0.0003160017 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.0%  lep_ptrel > 43.21
        - 0.0003085388 * max(0.0, 3.298199 - Q.sj3_mass2) / 0.1692566   # -0.0%  sj3_mass2 < 3.298
        - 0.000284828 * max(0.0, 114.0172 - Q.mass) * max(0.0, 2.255345 - Q.D2) / 6.249005   # -0.0%  mass < 114 and D2 < 2.255
        - 0.0002537478 * max(0.0, Q.sd_mass - 42.31629) * max(0.0, 0.4683306 - Q.z_dr_0p2_0p4) / 13.07579   # -0.0%  sd_mass > 42.32 and z_dr_0p2_0p4 < 0.4683
        + 0.0002467055 * max(0.0, Q.sj3_dr_min - 0.3526989) / 0.01394539   # +0.0%  sj3_dr_min > 0.3527
        - 0.0002294735 * max(0.0, Q.mass_top30 - 162.7874) * max(0.0, 1049.334 - Q.sum_e) / 214.2391   # -0.0%  mass_top30 > 162.8 and sum_e < 1049
        - 0.0001371102 * max(0.0, Q.sd_mass - 154.5947) * max(0.0, 0.2450652 - Q.sd_zg) / 0.1089994   # -0.0%  sd_mass > 154.6 and sd_zg < 0.2451
        + 0.0001160219 * max(0.0, 7.104488e-06 - Q.e3_b2) / 8.290971e-07   # +0.0%  e3_b2 < 7.104e-06
        - 8.190495e-05 * max(0.0, Q.n_s3d_above_3 - 5.0) * max(0.0, 226.3008 - Q.sip_3d_2) / 73.04149   # -0.0%  n_s3d_above_3 > 5 and sip_3d_2 < 226.3
        - 5.316844e-05 * max(0.0, Q.sj3_dr_min - 0.3526989) * max(0.0, Q.charge_27 - 0.0) / 0.002673855   # -0.0%  sj3_dr_min > 0.3527 and charge_27 > 0
        + 3.808343e-05 * max(0.0, Q.mass_top10 - 27.42853) * max(0.0, Q.dr_77 - 0.0) / 0.1917396   # +0.0%  mass_top10 > 27.43 and dr_77 > 0
        - 1.887927e-05 * max(0.0, Q.mass_2photon - 22.18431) * max(0.0, Q.iselectron_1 - 0.0) / 0.01053741   # -0.0%  mass_2photon > 22.18 and iselectron_1 > 0
        - 1.708767e-05 * max(0.0, Q.n_s3d_above_3 - 5.0) * max(0.0, Q.eta_48 - 0.2019104) / 0.01429746   # -0.0%  n_s3d_above_3 > 5 and eta_48 > 0.2019
        + 6.106964e-06 * max(0.0, Q.mass_top30 - 162.7874) * max(0.0, Q.e4_b2 - 1.13e-10) / 2.583592e-06   # +0.0%  mass_top30 > 162.8 and e4_b2 > 1.13e-10
        - 5.808886e-06 * max(0.0, 0.03259227 - Q.M3) * max(0.0, Q.ismuon_4 - 0.0) / 3.402682e-05   # -0.0%  M3 < 0.03259 and ismuon_4 > 0
    )
    return z


def neuron_69(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.928535e-06
    )
    return z


def neuron_70(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.01607455
    )
    return z


def neuron_71(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.030836e-06
    )
    return z


def neuron_72(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.164686e-06
    )
    return z


def neuron_73(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.699241e-05
    )
    return z


def neuron_74(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.49692e-06
    )
    return z


def neuron_75(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.971832e-06
    )
    return z


def neuron_76(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.5435e-06
    )
    return z


def neuron_77(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.08463e-06
    )
    return z


def neuron_78(Q):
    # scale S = 6.634; each line: share * term / its average size
    z = 6.634183 * (-0.1656804
        + 0.09714664 * Q.n_charged_had / 18.52697   # +9.7%  n_charged_had
        + 0.06582654 * max(0.0, 6.0 - Q.n_s3d_above_3) / 2.86534   # +6.6%  n_s3d_above_3 < 6
        - 0.05819362 * max(0.0, 12.15228 - Q.lep_ptrel) / 8.819929   # -5.8%  lep_ptrel < 12.15
        - 0.05493984 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # -5.5%  lep_ptrel < 27.32
        + 0.05368959 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +5.4%  lep_iso < 0.4382
        + 0.05162163 * max(0.0, 0.1024935 - Q.tau3) / 0.05489442   # +5.2%  tau3 < 0.1025
        - 0.05013891 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # -5.0%  lep_z < 0.3397
        + 0.04303013 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +4.3%  lep_ptrel < 43.21
        + 0.04070512 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # +4.1%  n_s3d_above_3 < 10
        - 0.03485288 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # -3.5%  lep_iso < 1.362
        + 0.03147582 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.177312 - Q.M2_b05) / 0.009855904   # +3.1%  lep_z < 0.3397 and M2_b05 < 0.1773
        + 0.02798099 * max(0.0, 411.0 - Q.n_pairs_kt_above_1) / 159.5699   # +2.8%  n_pairs_kt_above_1 < 411
        - 0.02705226 * max(0.0, 18.7678 - Q.lep_ptrel) / 14.28156   # -2.7%  lep_ptrel < 18.77
        + 0.02377446 * max(0.0, Q.M2 - 0.04276413) / 0.03598257   # +2.4%  M2 > 0.04276
        - 0.01897517 * max(0.0, 0.1024935 - Q.tau3) * max(0.0, 577.991 - Q.sip_3d_3) / 29.11377   # -1.9%  tau3 < 0.1025 and sip_3d_3 < 578
        - 0.01411256 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # -1.4%  max_abs_d0 < 10.52
        + 0.01290115 * max(0.0, Q.mass - 127.2317) / 9.462689   # +1.3%  mass > 127.2
        - 0.01280138 * max(0.0, 2.0 - Q.n_lepton) * max(0.0, 0.9333327 - Q.sj3_pairmax_over_m) / 0.1821352   # -1.3%  n_lepton < 2 and sj3_pairmax_over_m < 0.9333
        - 0.01194013 * max(0.0, Q.z_displaced3 - 0.02600452) / 0.09324453   # -1.2%  z_displaced3 > 0.026
        + 0.01174056 * max(0.0, Q.sj4_pair_mass_max - 78.76797) / 11.81635   # +1.2%  sj4_pair_mass_max > 78.77
        - 0.01093353 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.pair_mean_lnkt - -0.0709631) / 0.1542639   # -1.1%  lep_z < 0.3397 and pair_mean_lnkt > -0.07096
        + 0.01054758 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 175.9957 - Q.sip_3d_3) / 116.3453   # +1.1%  n_s3d_above_3 > 4 and sip_3d_3 < 176
        - 0.01010319 * max(0.0, Q.sum_e - 550.9555) / 355.726   # -1.0%  sum_e > 551
        + 0.009678845 * max(0.0, Q.mass_charged - 37.42054) / 28.62477   # +1.0%  mass_charged > 37.42
        - 0.008960502 * Q.lne_0 / 5.19469   # -0.9%  lne_0
        - 0.008853367 * max(0.0, Q.mass - 164.4374) / 3.038568   # -0.9%  mass > 164.4
        - 0.00851125 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # -0.9%  n_lepton < 1
        + 0.008378865 * max(0.0, 11.0 - Q.n_dr_0p1_0p2) / 2.841407   # +0.8%  n_dr_0p1_0p2 < 11
        + 0.00835552 * max(0.0, 71.96396 - Q.mass) / 1.934957   # +0.8%  mass < 71.96
        + 0.00797019 * max(0.0, Q.mass_top50 - 122.0585) / 10.10312   # +0.8%  mass_top50 > 122.1
        + 0.007952626 * max(0.0, 0.3264446 - Q.D3_b2) / 0.2054222   # +0.8%  D3_b2 < 0.3264
        + 0.007461421 * max(0.0, 99.0 - Q.n_pairs_kt_above_3) / 39.22243   # +0.7%  n_pairs_kt_above_3 < 99
        - 0.007332971 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 15.0 - Q.n_dr_0p4_up) / 132.2245   # -0.7%  lep_ptrel < 18.77 and n_dr_0p4_up < 15
        - 0.007043419 * max(0.0, 0.4381892 - Q.lep_iso) * max(0.0, 11.13866 - Q.sip_3d_3) / 1.560811   # -0.7%  lep_iso < 0.4382 and sip_3d_3 < 11.14
        - 0.007007987 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.sj3_dr_min - 0.08335692) / 0.7526776   # -0.7%  n_s3d_above_3 < 10 and sj3_dr_min > 0.08336
        + 0.006845614 * max(0.0, 2.255345 - Q.D2) / 0.5917601   # +0.7%  D2 < 2.255
        + 0.006383347 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.1631992 - Q.sj4_dr_min) / 0.0170385   # +0.6%  lep_z < 0.3397 and sj4_dr_min < 0.1632
        - 0.006018256 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 0.5440886 - Q.tau21) / 0.2104196   # -0.6%  n_s3d_above_3 > 4 and tau21 < 0.5441
        - 0.005962465 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, -1.332584 - Q.pair_mean_lnz) / 0.09084841   # -0.6%  lep_z < 0.3397 and pair_mean_lnz < -1.333
        - 0.005936895 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 23.30856 - Q.sip_3d_3) / 97.09082   # -0.6%  max_abs_d0 < 10.52 and sip_3d_3 < 23.31
        + 0.005492534 * max(0.0, Q.mass_top5 - 54.36876) / 5.778115   # +0.5%  mass_top5 > 54.37
        - 0.005289728 * max(0.0, Q.pair_max_lnm2 - 7.347625) / 0.1021051   # -0.5%  pair_max_lnm2 > 7.348
        + 0.004855489 * max(0.0, Q.n_sd0_above_5 - 4.0) / 0.7095433   # +0.5%  n_sd0_above_5 > 4
        + 0.004728213 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # +0.5%  lep_z < 0.2214
        + 0.004270875 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.mass_top20 - 86.78877) / 3.685704   # +0.4%  lep_z < 0.3397 and mass_top20 > 86.79
        + 0.004268894 * max(0.0, Q.mass_top15 - 80.3877) / 12.32501   # +0.4%  mass_top15 > 80.39
        - 0.004219187 * max(0.0, 32.61679 - Q.mass_displaced5) / 27.02182   # -0.4%  mass_displaced5 < 32.62
        + 0.004194032 * max(0.0, Q.n_sd0_above_3 - 2.0) / 1.826197   # +0.4%  n_sd0_above_3 > 2
        - 0.004111698 * max(0.0, 67.20576 - Q.mass_top30) / 1.774095   # -0.4%  mass_top30 < 67.21
        + 0.003956481 * max(0.0, 4.606241 - Q.sip_3d_3) / 1.172198   # +0.4%  sip_3d_3 < 4.606
        + 0.003695933 * max(0.0, Q.z_photon - 0.3318968) / 0.02575489   # +0.4%  z_photon > 0.3319
        + 0.003558621 * max(0.0, 0.6206221 - Q.tau32) / 0.05684264   # +0.4%  tau32 < 0.6206
        - 0.00335262 * max(0.0, 0.0007393085 - Q.e3) / 0.000161292   # -0.3%  e3 < 0.0007393
        + 0.00309575 * max(0.0, 21.21062 - Q.sj2_mass1) / 1.735779   # +0.3%  sj2_mass1 < 21.21
        + 0.002957492 * max(0.0, 2.0 - Q.n_dr_0_0p05) / 0.7938667   # +0.3%  n_dr_0_0p05 < 2
        + 0.0026941 * max(0.0, 1.102602 - Q.D2_b05) / 0.006783992   # +0.3%  D2_b05 < 1.103
        - 0.002396758 * max(0.0, 2.0 - Q.n_dr_0_0p05) * max(0.0, 1.0 - Q.isnhad_27) / 0.74134   # -0.2%  n_dr_0_0p05 < 2 and isnhad_27 < 1
        - 0.00237246 * max(0.0, Q.sj4_pair_mass_max - 115.5142) / 2.08655   # -0.2%  sj4_pair_mass_max > 115.5
        - 0.002189767 * max(0.0, 0.6378426 - Q.z_charged) / 0.08720331   # -0.2%  z_charged < 0.6378
        + 0.0020054 * max(0.0, 0.8009208 - Q.psi_0p2) / 0.1053387   # +0.2%  psi_0p2 < 0.8009
        + 0.001938465 * max(0.0, 18.7678 - Q.lep_ptrel) * max(0.0, 0.5699805 - Q.z_dr_0p1_0p2) / 3.961803   # +0.2%  lep_ptrel < 18.77 and z_dr_0p1_0p2 < 0.57
        + 0.001928594 * max(0.0, Q.mass_displaced5 - 6.387683) * max(0.0, 14.38858 - Q.lep_iso) / 40.3921   # +0.2%  mass_displaced5 > 6.388 and lep_iso < 14.39
        - 0.00181072 * max(0.0, Q.mass_displaced5 - 6.387683) / 3.821495   # -0.2%  mass_displaced5 > 6.388
        - 0.001737905 * max(0.0, 0.07985021 - Q.sj4_dr_min) / 0.01211577   # -0.2%  sj4_dr_min < 0.07985
        + 0.001588797 * max(0.0, 13.03663 - Q.mass_displaced3) / 8.661134   # +0.2%  mass_displaced3 < 13.04
        + 0.001518672 * max(0.0, Q.lund2_lndelta - -1.556018) / 0.6029678   # +0.2%  lund2_lndelta > -1.556
        + 0.001487452 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 0.3190414 - Q.sj3_dr_min) / 0.14046   # +0.1%  n_s3d_above_3 > 4 and sj3_dr_min < 0.319
        - 0.001331936 * max(0.0, 2.0 - Q.n_lepton) / 1.440613   # -0.1%  n_lepton < 2
        - 0.001226776 * max(0.0, 0.6206221 - Q.tau32) * max(0.0, Q.jet_charge_k03 - 0.02942741) / 0.01536299   # -0.1%  tau32 < 0.6206 and jet_charge_k03 > 0.02943
        - 0.001033626 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, 0.8436463 - Q.psi_0p2) / 2.692385   # -0.1%  lep_ptrel < 27.32 and psi_0p2 < 0.8436
        - 0.001007302 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 8.17233e-05 - Q.ecf_g42) / 1.595489e-05   # -0.1%  lep_z < 0.3397 and ecf_g42 < 8.172e-05
        + 0.0009999307 * max(0.0, Q.n_pairs_kt_above_3 - 220.0) / 4.57629   # +0.1%  n_pairs_kt_above_3 > 220
        - 0.0008507886 * max(0.0, Q.n_electron - 1.0) / 0.08167   # -0.1%  n_electron > 1
        - 0.0007398878 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # -0.1%  mass_top50 > 161.1
        - 0.0004921485 * max(0.0, Q.z_neutral_had - 0.3788785) / 0.004557199   # -0.0%  z_neutral_had > 0.3789
        - 0.0004153646 * max(0.0, Q.mass_displaced5 - 6.387683) * max(0.0, 1.0 - Q.n_lepton) / 1.715768   # -0.0%  mass_displaced5 > 6.388 and n_lepton < 1
        - 0.0003996365 * max(0.0, 21.21062 - Q.sj2_mass1) * max(0.0, 0.04891968 - Q.phi_26) / 0.1248013   # -0.0%  sj2_mass1 < 21.21 and phi_26 < 0.04892
        + 0.000344059 * max(0.0, Q.mass_displaced5 - 6.387683) * max(0.0, 11.13866 - Q.sip_3d_3) / 0.8594348   # +0.0%  mass_displaced5 > 6.388 and sip_3d_3 < 11.14
        + 0.0002702928 * max(0.0, 0.2458451 - Q.sj3_dr23) / 0.02482854   # +0.0%  sj3_dr23 < 0.2458
        - 0.0002568221 * max(0.0, Q.mass_top40 - 173.1022) / 1.415364   # -0.0%  mass_top40 > 173.1
        + 0.0001972581 * max(0.0, Q.z_displaced3 - 0.02600452) * max(0.0, Q.isphoton_14 - 0.0) / 0.02961916   # +0.0%  z_displaced3 > 0.026 and isphoton_14 > 0
        + 0.0001888547 * max(0.0, Q.mass_displaced3 - 39.09615) * max(0.0, 33.08364 - Q.sip_3d_3) / 2.849973   # +0.0%  mass_displaced3 > 39.1 and sip_3d_3 < 33.08
        + 0.0001733005 * max(0.0, 27.3236 - Q.lep_ptrel) * max(0.0, -0.210083 - Q.eta_28) / 0.3467154   # +0.0%  lep_ptrel < 27.32 and eta_28 < -0.2101
        - 0.0001616237 * max(0.0, Q.n_s3d_above_3 - 4.0) / 1.224997   # -0.0%  n_s3d_above_3 > 4
        - 0.0001413367 * max(0.0, 0.1024935 - Q.tau3) * max(0.0, Q.tdz_33 - -0.01794241) / 0.001834404   # -0.0%  tau3 < 0.1025 and tdz_33 > -0.01794
        - 0.0001243902 * max(0.0, 23.0 - Q.n_neutral) / 5.227467   # -0.0%  n_neutral < 23
        - 0.0001167194 * max(0.0, Q.n_sd0_above_5 - 4.0) * max(0.0, 11.13866 - Q.sip_3d_3) / 0.01232867   # -0.0%  n_sd0_above_5 > 4 and sip_3d_3 < 11.14
        + 9.857214e-05 * max(0.0, Q.mass_displaced3 - 39.09615) / 0.6617151   # +0.0%  mass_displaced3 > 39.1
        - 9.590607e-05 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, 0.03430176 - Q.dzerr_0) / 0.1014645   # -0.0%  n_s3d_above_3 < 10 and dzerr_0 < 0.0343
        + 8.468553e-05 * max(0.0, Q.mass_top40 - 173.1022) * max(0.0, 1.0 - Q.n_muon) / 1.026809   # +0.0%  mass_top40 > 173.1 and n_muon < 1
        - 7.529163e-05 * max(0.0, Q.z_displaced5 - 0.1709091) / 0.02847701   # -0.0%  z_displaced5 > 0.1709
        - 6.507609e-05 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # -0.0%  mass_top10 > 103.5
        - 6.27673e-05 * max(0.0, 1.104348 - Q.lead_ch_sdz) * max(0.0, Q.iselectron_1 - 0.0) / 0.05540389   # -0.0%  lead_ch_sdz < 1.104 and iselectron_1 > 0
        - 6.219792e-05 * max(0.0, 1.104348 - Q.lead_ch_sdz) / 1.96298   # -0.0%  lead_ch_sdz < 1.104
        + 5.255533e-05 * max(0.0, Q.psi_0p1 - 0.9356675) / 0.00150838   # +0.0%  psi_0p1 > 0.9357
        - 4.300382e-05 * max(0.0, Q.mass_displaced5 - 6.387683) * max(0.0, 0.8156512 - Q.psi_0p3) / 0.1625677   # -0.0%  mass_displaced5 > 6.388 and psi_0p3 < 0.8157
        + 1.980693e-05 * max(0.0, Q.sum_e - 550.9555) * max(0.0, Q.ismuon_4 - 0.0) / 2.93247   # +0.0%  sum_e > 551 and ismuon_4 > 0
        - 1.420262e-05 * max(0.0, Q.n_electron - 1.0) * max(0.0, Q.iselectron_30 - 0.0) / 0.003523333   # -0.0%  n_electron > 1 and iselectron_30 > 0
    )
    return z


def neuron_79(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.602822e-05
    )
    return z


def neuron_80(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (9.706954e-06
    )
    return z


def neuron_81(Q):
    # scale S = 5.064; each line: share * term / its average size
    z = 5.06355 * (-0.09414857
        + 0.06505177 * max(0.0, Q.mass_top40 - 97.12186) / 20.78291   # +6.5%  mass_top40 > 97.12
        + 0.05738653 * max(0.0, 13.97388 - Q.mass_displaced5) / 10.07648   # +5.7%  mass_displaced5 < 13.97
        - 0.05726098 * max(0.0, Q.mass_top50 - 99.54528) / 21.60123   # -5.7%  mass_top50 > 99.55
        + 0.05060124 * max(0.0, Q.mass_top50 - 115.614) / 12.67421   # +5.1%  mass_top50 > 115.6
        - 0.05051334 * Q.M2 / 0.07774996   # -5.1%  M2
        + 0.04383261 * max(0.0, 0.1061578 - Q.z_displaced3) * max(0.0, 0.03021637 - Q.lep_z) / 0.001161425   # +4.4%  z_displaced3 < 0.1062 and lep_z < 0.03022
        + 0.03743504 * max(0.0, Q.n_s3d_above_3 - 3.0) / 1.66923   # +3.7%  n_s3d_above_3 > 3
        - 0.03353847 * max(0.0, Q.mass_top40 - 119.1279) / 9.647038   # -3.4%  mass_top40 > 119.1
        - 0.03351563 * max(0.0, 0.1076451 - Q.tau2) / 0.0393622   # -3.4%  tau2 < 0.1076
        + 0.03006587 * max(0.0, 0.4242439 - Q.LHA) / 0.05026985   # +3.0%  LHA < 0.4242
        + 0.02943451 * max(0.0, 5.0 - Q.n_s3d_above_3) / 2.12335   # +2.9%  n_s3d_above_3 < 5
        + 0.02834455 * max(0.0, Q.z_displaced3 - 0.03624058) / 0.08737704   # +2.8%  z_displaced3 > 0.03624
        - 0.02618821 * max(0.0, 0.1061578 - Q.z_displaced3) / 0.05189409   # -2.6%  z_displaced3 < 0.1062
        - 0.02558594 * max(0.0, 10.52344 - Q.max_abs_d0) * max(0.0, 7.345216 - Q.sip_3d_3) / 20.68407   # -2.6%  max_abs_d0 < 10.52 and sip_3d_3 < 7.345
        - 0.02280357 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.z_charged_had - 0.265564) / 0.02462507   # -2.3%  z_displaced3 > 0.03624 and z_charged_had > 0.2656
        - 0.0218676 * max(0.0, 1.777286 - Q.mass_displaced3) / 0.7377974   # -2.2%  mass_displaced3 < 1.777
        + 0.02092018 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # +2.1%  max_abs_d0 < 5.812
        - 0.01947449 * max(0.0, 0.01329067 - Q.z_electron) / 0.01018622   # -1.9%  z_electron < 0.01329
        + 0.01917134 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # +1.9%  lep_ptrel > 27.32
        + 0.0180618 * max(0.0, 6.983043 - Q.lep_ptrel) / 4.807808   # +1.8%  lep_ptrel < 6.983
        - 0.01626455 * max(0.0, 8.0 - Q.n_sd0_above_3) / 4.91424   # -1.6%  n_sd0_above_3 < 8
        + 0.01619183 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 33.0 - Q.n_charged_had) / 19.17455   # +1.6%  n_s3d_above_3 > 3 and n_charged_had < 33
        + 0.01488721 * max(0.0, 577.991 - Q.sip_3d_3) / 516.7039   # +1.5%  sip_3d_3 < 578
        + 0.01409107 * max(0.0, Q.n_s3d_above_10 - 3.0) / 0.9107233   # +1.4%  n_s3d_above_10 > 3
        - 0.01356898 * max(0.0, Q.sj4_pair_mass_max - 63.34656) / 21.48011   # -1.4%  sj4_pair_mass_max > 63.35
        - 0.01249012 * max(0.0, 15.0 - Q.n_dr_0p4_up) / 9.48698   # -1.2%  n_dr_0p4_up < 15
        + 0.01228769 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 447.0873 - Q.sip_3d_2) / 370.2766   # +1.2%  n_s3d_above_3 > 3 and sip_3d_2 < 447.1
        - 0.01124138 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 0.3014662 - Q.lep_dr) / 0.01653138   # -1.1%  z_displaced3 > 0.03624 and lep_dr < 0.3015
        - 0.01037521 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 21.12768 - Q.mass_displaced5) / 13.37539   # -1.0%  n_s3d_above_3 > 3 and mass_displaced5 < 21.13
        + 0.01009018 * max(0.0, 2.723325 - Q.sip_3d_3) / 0.2881626   # +1.0%  sip_3d_3 < 2.723
        + 0.009725916 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # +1.0%  lep_z < 0.2214
        - 0.009490333 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.9%  lep_ptrel > 43.21
        + 0.009159621 * max(0.0, Q.mass_top50 - 125.4927) / 8.989867   # +0.9%  mass_top50 > 125.5
        - 0.008718537 * max(0.0, Q.n_sd0_above_5 - 5.0) / 0.4736133   # -0.9%  n_sd0_above_5 > 5
        - 0.008659816 * max(0.0, 6.983043 - Q.lep_ptrel) * max(0.0, Q.pt_balance01 - 0.1985369) / 0.9136075   # -0.9%  lep_ptrel < 6.983 and pt_balance01 > 0.1985
        + 0.007993596 * max(0.0, Q.z_displaced5 - 0.01810676) / 0.08074326   # +0.8%  z_displaced5 > 0.01811
        - 0.00783817 * max(0.0, 22.0 - Q.n_pairs_kt_above_3) * max(0.0, 1.0 - Q.n_pairs_kt_above_30) / 1.994697   # -0.8%  n_pairs_kt_above_3 < 22 and n_pairs_kt_above_30 < 1
        - 0.007718729 * max(0.0, 0.7634316 - Q.max_dr) / 0.1364426   # -0.8%  max_dr < 0.7634
        + 0.007709015 * max(0.0, 1.203438 - Q.jet_abs_eta) / 0.5210607   # +0.8%  jet_abs_eta < 1.203
        + 0.007247156 * max(0.0, Q.sj4_pair_mass_max - 63.34656) * max(0.0, 0.3788785 - Q.z_neutral_had) / 4.99524   # +0.7%  sj4_pair_mass_max > 63.35 and z_neutral_had < 0.3789
        + 0.006469371 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.n_lund - 5.0) / 0.5382206   # +0.6%  z_displaced3 > 0.03624 and n_lund > 5
        - 0.006346194 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -0.6%  sip_3d_2 < 226.3
        - 0.006182761 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 0.8882532 - Q.tau32) / 0.3641546   # -0.6%  n_s3d_above_3 > 3 and tau32 < 0.8883
        + 0.006011516 * max(0.0, 0.2377332 - Q.tau21) / 0.01385845   # +0.6%  tau21 < 0.2377
        - 0.005944878 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 1.839882 - Q.mass_2charged) / 0.03997451   # -0.6%  z_displaced3 > 0.03624 and mass_2charged < 1.84
        - 0.005157279 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # -0.5%  mass_top50 > 161.1
        + 0.004488543 * max(0.0, 52.72207 - Q.mass_top30) / 0.741523   # +0.4%  mass_top30 < 52.72
        + 0.004046718 * max(0.0, 1.763283 - Q.min_pair_mass) / 0.6216059   # +0.4%  min_pair_mass < 1.763
        - 0.003813823 * max(0.0, 0.0009010251 - Q.ecf_g32) / 7.436273e-05   # -0.4%  ecf_g32 < 0.000901
        + 0.003647411 * max(0.0, Q.mass_2charged - 20.76537) / 3.761489   # +0.4%  mass_2charged > 20.77
        + 0.003570353 * max(0.0, 22.0 - Q.n_pairs_kt_above_3) / 2.320537   # +0.4%  n_pairs_kt_above_3 < 22
        + 0.003176446 * max(0.0, 0.07290954 - Q.sum_z_dr) / 0.002660742   # +0.3%  sum_z_dr < 0.07291
        - 0.003038334 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 0.8597199 - Q.tau43) / 0.1092297   # -0.3%  n_s3d_above_3 > 3 and tau43 < 0.8597
        + 0.002999191 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # +0.3%  max_abs_d0 < 10.52
        - 0.002487128 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 0.9036234 - Q.tau54) / 0.08976281   # -0.2%  n_s3d_above_3 > 3 and tau54 < 0.9036
        + 0.002462144 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 16.23838 - Q.sip_3d_3) / 0.2292165   # +0.2%  z_displaced3 > 0.03624 and sip_3d_3 < 16.24
        + 0.002026835 * max(0.0, Q.pair_mean_lndelta - -1.352792) / 0.01174331   # +0.2%  pair_mean_lndelta > -1.353
        + 0.001684377 * max(0.0, Q.n_s3d_above_3 - 10.0) / 0.12486   # +0.2%  n_s3d_above_3 > 10
        + 0.001536877 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.sj2_dr - 0.2403736) / 0.015899   # +0.2%  z_displaced3 > 0.03624 and sj2_dr > 0.2404
        + 0.001462464 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.z_muon - 0.03898651) / 0.003551335   # +0.1%  z_displaced3 > 0.03624 and z_muon > 0.03899
        + 0.001420582 * max(0.0, 0.1076451 - Q.tau2) * max(0.0, Q.n_muon - 0.0) / 0.009639314   # +0.1%  tau2 < 0.1076 and n_muon > 0
        - 0.0013221 * max(0.0, Q.n_sd0_above_3 - 9.0) / 0.1165233   # -0.1%  n_sd0_above_3 > 9
        + 0.00121771 * max(0.0, Q.tdz_2 - -0.02221619) / 0.04771882   # +0.1%  tdz_2 > -0.02222
        + 0.001198652 * max(0.0, Q.n_s3d_above_3 - 10.0) * max(0.0, 1.0 - Q.isphoton_0) / 0.09957333   # +0.1%  n_s3d_above_3 > 10 and isphoton_0 < 1
        + 0.001190367 * max(0.0, 0.2377332 - Q.tau21) * max(0.0, 0.1985369 - Q.pt_balance01) / 0.0002543562   # +0.1%  tau21 < 0.2377 and pt_balance01 < 0.1985
        - 0.001045362 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # -0.1%  n_pairs_kt_above_1 < 80
        - 0.001003547 * max(0.0, 0.02949822 - Q.tau4) / 0.003969373   # -0.1%  tau4 < 0.0295
        - 0.0009138297 * max(0.0, 5.0 - Q.n_s3d_above_3) * max(0.0, Q.eccentricity - 0.5174679) / 0.5955715   # -0.1%  n_s3d_above_3 < 5 and eccentricity > 0.5175
        - 0.0006968622 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, 0.04032236 - Q.td0_33) / 0.004915466   # -0.1%  z_displaced3 > 0.03624 and td0_33 < 0.04032
        + 0.0006821672 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.z_photon - 0.3887278) / 0.0005662707   # +0.1%  z_displaced3 > 0.03624 and z_photon > 0.3887
        - 0.000669039 * max(0.0, 70.05844 - Q.sip_3d_2) / 37.02037   # -0.1%  sip_3d_2 < 70.06
        - 0.0006514613 * max(0.0, Q.mass_top40 - 119.1279) * max(0.0, 1.0 - Q.isphoton_32) / 5.648897   # -0.1%  mass_top40 > 119.1 and isphoton_32 < 1
        + 0.0005645294 * max(0.0, -0.1170754 - Q.tdz_1) / 0.01260556   # +0.1%  tdz_1 < -0.1171
        - 0.0005368006 * max(0.0, Q.mass_top40 - 97.12186) * max(0.0, Q.tau43_b2 - 0.6509029) / 2.050221   # -0.1%  mass_top40 > 97.12 and tau43_b2 > 0.6509
        + 0.0005130644 * max(0.0, Q.mass_charged - 99.20396) / 2.066138   # +0.1%  mass_charged > 99.2
        + 0.0004952322 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 0.2477181 - Q.pt1_over_pt0) / 0.01004037   # +0.0%  n_s3d_above_3 > 3 and pt1_over_pt0 < 0.2477
        + 0.0004642656 * max(0.0, Q.sj4_pair_mass_max - 63.34656) * max(0.0, 0.1695557 - Q.eta_37) / 4.215693   # +0.0%  sj4_pair_mass_max > 63.35 and eta_37 < 0.1696
        - 0.0003904574 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, Q.n_lund_kt_above_5 - 2.0) / 0.7653267   # -0.0%  n_s3d_above_3 > 3 and n_lund_kt_above_5 > 2
        - 0.0003497491 * max(0.0, 0.1076451 - Q.tau2) * max(0.0, Q.jet_charge_k03 - 0.2021837) / 0.007468288   # -0.0%  tau2 < 0.1076 and jet_charge_k03 > 0.2022
        - 0.0003379088 * max(0.0, 1.763283 - Q.min_pair_mass) * max(0.0, 1.0 - Q.isnhad_8) / 0.5334306   # -0.0%  min_pair_mass < 1.763 and isnhad_8 < 1
        - 0.000264756 * max(0.0, Q.tdz_2 - -0.02221619) * max(0.0, 0.0881958 - Q.dzerr_44) / 0.003828339   # -0.0%  tdz_2 > -0.02222 and dzerr_44 < 0.0882
        + 0.0002423617 * max(0.0, 0.01181938 - Q.z_dr_0p4_up) / 0.003195022   # +0.0%  z_dr_0p4_up < 0.01182
        - 0.0001598875 * max(0.0, 0.4242439 - Q.LHA) * max(0.0, 8.0 - Q.n_lund) / 0.01975455   # -0.0%  LHA < 0.4242 and n_lund < 8
        + 0.0001453115 * max(0.0, Q.sj3_pairmin_over_m - 0.2477126) / 0.09094898   # +0.0%  sj3_pairmin_over_m > 0.2477
        + 6.25288e-05 * max(0.0, Q.sj3_pairmin_over_m - 0.2477126) * max(0.0, 0.2877534 - Q.dr_16) / 0.01048531   # +0.0%  sj3_pairmin_over_m > 0.2477 and dr_16 < 0.2878
        + 2.509357e-05 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, Q.isnhad_34 - 0.0) / 0.08861667   # +0.0%  n_s3d_above_3 > 3 and isnhad_34 > 0
        - 2.356542e-05 * max(0.0, Q.tdz_2 - -0.02221619) * max(0.0, Q.ismuon_12 - 0.0) / 0.0002048848   # -0.0%  tdz_2 > -0.02222 and ismuon_12 > 0
        - 1.780471e-05 * max(0.0, 6.983043 - Q.lep_ptrel) * max(0.0, Q.ismuon_11 - 0.0) / 0.009388841   # -0.0%  lep_ptrel < 6.983 and ismuon_11 > 0
        - 1.367556e-05 * max(0.0, Q.n_sd0_above_3 - 9.0) * max(0.0, 0.0 - Q.tdz_47) / 0.003538555   # -0.0%  n_sd0_above_3 > 9 and tdz_47 < 0
        - 1.16315e-05 * max(0.0, 0.7634316 - Q.max_dr) * max(0.0, Q.ismuon_49 - 0.0) / 1.799864e-05   # -0.0%  max_dr < 0.7634 and ismuon_49 > 0
        - 8.211343e-06 * max(0.0, Q.z_displaced3 - 0.03624058) * max(0.0, Q.phi_79 - 0.0) / 4.961935e-05   # -0.0%  z_displaced3 > 0.03624 and phi_79 > 0
        - 6.071091e-06 * max(0.0, 6.983043 - Q.lep_ptrel) * max(0.0, Q.ismuon_73 - 0.0) / 0.0001851159   # -0.0%  lep_ptrel < 6.983 and ismuon_73 > 0
    )
    return z


def neuron_82(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.943195e-06
    )
    return z


def neuron_83(Q):
    # scale S = 12.32; each line: share * term / its average size
    z = 12.31658 * (0.07829922
        - 0.122621 * max(0.0, 173.1022 - Q.mass_top40) / 64.05167   # -12.3%  mass_top40 < 173.1
        + 0.09747908 * max(0.0, 10.0 - Q.n_s3d_above_3) / 6.364753   # +9.7%  n_s3d_above_3 < 10
        - 0.08758502 * max(0.0, Q.sd_mass - 83.61981) / 26.48984   # -8.8%  sd_mass > 83.62
        + 0.06485441 * max(0.0, Q.sd_mass - 62.03827) / 42.41619   # +6.5%  sd_mass > 62.04
        - 0.04804863 * max(0.0, 0.000141779 - Q.e4_b05) / 0.0001033822   # -4.8%  e4_b05 < 0.0001418
        + 0.03162285 * max(0.0, 27.3236 - Q.lep_ptrel) / 21.78277   # +3.2%  lep_ptrel < 27.32
        - 0.03069069 * max(0.0, Q.mass_top40 - 78.33213) * max(0.0, 15.17086 - Q.D2_b2) / 454.9867   # -3.1%  mass_top40 > 78.33 and D2_b2 < 15.17
        - 0.02740884 * max(0.0, Q.mass_top40 - 78.33213) / 34.93312   # -2.7%  mass_top40 > 78.33
        + 0.02715499 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # +2.7%  sd_mass > 88.82
        - 0.02295344 * max(0.0, Q.mass_top40 - 78.33213) * max(0.0, 18.7678 - Q.lep_ptrel) / 484.8844   # -2.3%  mass_top40 > 78.33 and lep_ptrel < 18.77
        - 0.02188173 * max(0.0, 10.0 - Q.n_s3d_above_3) * max(0.0, Q.z_neutral - 0.1384639) / 1.740792   # -2.2%  n_s3d_above_3 < 10 and z_neutral > 0.1385
        + 0.02151823 * max(0.0, 117.4867 - Q.mass) / 15.64803   # +2.2%  mass < 117.5
        - 0.02021735 * max(0.0, 7.0 - Q.n_sd0_above_10) / 4.781883   # -2.0%  n_sd0_above_10 < 7
        + 0.02000764 * max(0.0, 128.0079 - Q.sj4_pair_mass_max) / 48.13337   # +2.0%  sj4_pair_mass_max < 128
        + 0.01612468 * max(0.0, 0.1681173 - Q.z_displaced3) / 0.09545189   # +1.6%  z_displaced3 < 0.1681
        - 0.01576997 * max(0.0, Q.lund3_lndelta - -2.817283) / 1.325774   # -1.6%  lund3_lndelta > -2.817
        + 0.01423861 * max(0.0, 0.05018249 - Q.sum_zz_dr2) / 0.01730417   # +1.4%  sum_zz_dr2 < 0.05018
        + 0.0139496 * max(0.0, 33.0 - Q.n_charged) / 14.10218   # +1.4%  n_charged < 33
        + 0.01333991 * Q.ecf_g41 / 0.0002120905   # +1.3%  ecf_g41
        - 0.01279365 * max(0.0, Q.mass_top15 - 80.3877) / 12.32501   # -1.3%  mass_top15 > 80.39
        + 0.01216467 * max(0.0, 173.1022 - Q.mass_top40) * max(0.0, 15.17086 - Q.D2_b2) / 720.4267   # +1.2%  mass_top40 < 173.1 and D2_b2 < 15.17
        + 0.01121844 * max(0.0, 13.03663 - Q.mass_displaced3) / 8.661134   # +1.1%  mass_displaced3 < 13.04
        - 0.01036429 * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0001737301   # -1.0%  e3_b2 < 0.0002537
        - 0.009762411 * max(0.0, 1.251841 - Q.mass_displaced3) / 0.4870081   # -1.0%  mass_displaced3 < 1.252
        - 0.009650959 * max(0.0, 30.49226 - Q.sip_3d_2) / 12.97731   # -1.0%  sip_3d_2 < 30.49
        + 0.009381034 * max(0.0, Q.pair_mean_lnm2 - 1.96906) / 0.7951971   # +0.9%  pair_mean_lnm2 > 1.969
        - 0.00883179 * max(0.0, Q.mass_top30 - 102.3652) / 13.48953   # -0.9%  mass_top30 > 102.4
        - 0.008770933 * max(0.0, 30.0 - Q.n_pt_above_5) / 9.011497   # -0.9%  n_pt_above_5 < 30
        + 0.008369644 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 175.9957 - Q.sip_3d_3) / 54.70527   # +0.8%  n_s3d_above_3 > 6 and sip_3d_3 < 176
        + 0.007910545 * max(0.0, 4.636903 - Q.sip_3d_2) / 0.784109   # +0.8%  sip_3d_2 < 4.637
        + 0.007871134 * max(0.0, 0.06416437 - Q.z_displaced3) / 0.02655242   # +0.8%  z_displaced3 < 0.06416
        + 0.007500863 * max(0.0, 0.8799072 - Q.sj3_dr23) / 0.4603866   # +0.8%  sj3_dr23 < 0.8799
        - 0.007394829 * max(0.0, 10.52344 - Q.max_abs_d0) / 5.916931   # -0.7%  max_abs_d0 < 10.52
        + 0.007390664 * max(0.0, Q.tau5 - 0.01222366) / 0.0203374   # +0.7%  tau5 > 0.01222
        - 0.006899415 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -0.7%  sip_3d_2 < 226.3
        - 0.006844098 * max(0.0, 6.661019 - Q.log_sum_pt) / 0.2482436   # -0.7%  log_sum_pt < 6.661
        + 0.006724276 * max(0.0, Q.mass_top40 - 119.1279) / 9.647038   # +0.7%  mass_top40 > 119.1
        + 0.006147169 * max(0.0, 3.0 - Q.n_s3d_above_3) / 0.9091233   # +0.6%  n_s3d_above_3 < 3
        - 0.005686163 * max(0.0, 0.4265629 - Q.N2_b05) / 0.02389014   # -0.6%  N2_b05 < 0.4266
        - 0.005597649 * max(0.0, 0.2250047 - Q.C3_b05) / 0.1186192   # -0.6%  C3_b05 < 0.225
        - 0.005437298 * max(0.0, 2.448333e-05 - Q.ecf_g42) / 8.669871e-06   # -0.5%  ecf_g42 < 2.448e-05
        + 0.005150481 * max(0.0, Q.mass_top40 - 155.8928) / 2.693823   # +0.5%  mass_top40 > 155.9
        - 0.004988612 * max(0.0, Q.e2 - 0.07745967) / 0.01550545   # -0.5%  e2 > 0.07746
        + 0.004916132 * max(0.0, Q.n_s3d_above_3 - 6.0) / 0.6254467   # +0.5%  n_s3d_above_3 > 6
        - 0.004879996 * max(0.0, Q.e2 - 0.04393457) / 0.04061416   # -0.5%  e2 > 0.04393
        + 0.004662487 * max(0.0, 2.448333e-05 - Q.ecf_g42) * max(0.0, 6.0 - Q.n_s3d_above_10) / 3.287542e-05   # +0.5%  ecf_g42 < 2.448e-05 and n_s3d_above_10 < 6
        - 0.004629065 * max(0.0, 0.3112717 - Q.sj4_dr_min) / 0.1919254   # -0.5%  sj4_dr_min < 0.3113
        + 0.004399431 * max(0.0, Q.N2_b05 - 0.4518419) / 0.02228442   # +0.4%  N2_b05 > 0.4518
        + 0.004341917 * max(0.0, 6.185635 - Q.lep_iso) / 4.883283   # +0.4%  lep_iso < 6.186
        - 0.004317436 * max(0.0, 49.25971 - Q.mass_neutral) / 11.9268   # -0.4%  mass_neutral < 49.26
        + 0.004307928 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +0.4%  sd_mass > 154.6
        + 0.003712195 * max(0.0, Q.n_s3d_above_10 - 1.0) / 1.904323   # +0.4%  n_s3d_above_10 > 1
        + 0.003585013 * max(0.0, Q.n_s3d_above_10 - 1.0) * max(0.0, 351.3271 - Q.sip_3d_1) / 195.5765   # +0.4%  n_s3d_above_10 > 1 and sip_3d_1 < 351.3
        - 0.003298117 * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 1.107657   # -0.3%  n_lund_kt_above_5 > 1
        - 0.003230594 * max(0.0, 90.08945 - Q.mass) / 4.972097   # -0.3%  mass < 90.09
        - 0.003218424 * max(0.0, 72.862 - Q.sj4_pair_mass_max) / 6.86631   # -0.3%  sj4_pair_mass_max < 72.86
        - 0.002941068 * max(0.0, 8.0 - Q.n_sd0_above_3) / 4.91424   # -0.3%  n_sd0_above_3 < 8
        - 0.002745185 * max(0.0, 2.0 - Q.n_lepton) / 1.440613   # -0.3%  n_lepton < 2
        - 0.002711572 * max(0.0, 0.06228948 - Q.M2) / 0.004761627   # -0.3%  M2 < 0.06229
        + 0.002682031 * max(0.0, 0.00228569 - Q.e3) / 0.001253813   # +0.3%  e3 < 0.002286
        + 0.00261664 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 114.8082   # +0.3%  mass < 117.5 and n_dr_0p1_0p2 > 1
        - 0.002102678 * max(0.0, 0.004294711 - Q.z_dr_0_0p05) / 0.00136489   # -0.2%  z_dr_0_0p05 < 0.004295
        + 0.001911035 * max(0.0, Q.pair_max_lnm2 - 7.347625) / 0.1021051   # +0.2%  pair_max_lnm2 > 7.348
        + 0.001752552 * max(0.0, Q.sj4_pair_mass_max - 115.5142) / 2.08655   # +0.2%  sj4_pair_mass_max > 115.5
        - 0.001446196 * max(0.0, Q.psi_0p2 - 0.8857951) / 0.01931927   # -0.1%  psi_0p2 > 0.8858
        + 0.001386684 * max(0.0, 0.00228569 - Q.e3) * max(0.0, Q.jet_abs_eta - 0.9149342) / 0.0001688642   # +0.1%  e3 < 0.002286 and jet_abs_eta > 0.9149
        + 0.001279345 * max(0.0, Q.n_sd0_above_5 - 7.0) / 0.1953867   # +0.1%  n_sd0_above_5 > 7
        - 0.001215337 * max(0.0, Q.pair_mean_lndelta - -2.115543) / 0.1901107   # -0.1%  pair_mean_lndelta > -2.116
        + 0.001154074 * max(0.0, Q.n_s3d_above_10 - 1.0) * max(0.0, 0.3264446 - Q.D3_b2) / 0.409072   # +0.1%  n_s3d_above_10 > 1 and D3_b2 < 0.3264
        + 0.001131849 * max(0.0, Q.z_photon - 0.4305934) / 0.009360866   # +0.1%  z_photon > 0.4306
        + 0.001080482 * max(0.0, 0.3951525 - Q.tau32) / 0.009206105   # +0.1%  tau32 < 0.3952
        + 0.001006675 * max(0.0, 0.05018249 - Q.sum_zz_dr2) * max(0.0, 16.0 - Q.n_lund) / 0.09623434   # +0.1%  sum_zz_dr2 < 0.05018 and n_lund < 16
        - 0.0009854779 * max(0.0, 0.3977051 - Q.max_abs_d0) / 0.09373534   # -0.1%  max_abs_d0 < 0.3977
        - 0.00088636 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, Q.sj3_pairmin_over_m - 0.1636952) / 0.09021261   # -0.1%  n_s3d_above_3 > 6 and sj3_pairmin_over_m > 0.1637
        - 0.0008333567 * max(0.0, 0.02210827 - Q.dr_max_012) / 0.001265414   # -0.1%  dr_max_012 < 0.02211
        - 0.0007664877 * max(0.0, Q.n_s3d_above_10 - 1.0) * max(0.0, 0.3788372 - Q.dr_16) / 0.3385076   # -0.1%  n_s3d_above_10 > 1 and dr_16 < 0.3788
        + 0.0006152492 * max(0.0, 0.5779883 - Q.pt1_over_pt0) / 0.08763774   # +0.1%  pt1_over_pt0 < 0.578
        + 0.0004679645 * max(0.0, Q.z_neutral_had - 0.3788785) / 0.004557199   # +0.0%  z_neutral_had > 0.3789
        + 0.000403663 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, 0.2949288 - Q.z_dr_0p1_0p2) / 0.04689436   # +0.0%  n_s3d_above_3 > 6 and z_dr_0p1_0p2 < 0.2949
        + 0.0003221517 * max(0.0, 0.3603262 - Q.z_charged_had) / 0.02270075   # +0.0%  z_charged_had < 0.3603
        + 0.0003076914 * max(0.0, Q.td0_3 - 0.1091249) / 0.01632519   # +0.0%  td0_3 > 0.1091
        - 0.0002134405 * max(0.0, 0.3886647 - Q.sj3_mass3) / 0.03562942   # -0.0%  sj3_mass3 < 0.3887
        - 0.0001906274 * max(0.0, 0.01517988 - Q.M3) / 0.0001250003   # -0.0%  M3 < 0.01518
        - 0.000181622 * max(0.0, Q.mass_top10 - 103.4976) * max(0.0, Q.mratio_min_012 - 0.0111128) / 0.1811386   # -0.0%  mass_top10 > 103.5 and mratio_min_012 > 0.01111
        + 0.0001670582 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # +0.0%  mass_top10 > 103.5
        - 0.000124881 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, Q.phi_3 - 0.06884766) / 0.01587558   # -0.0%  n_s3d_above_3 > 6 and phi_3 > 0.06885
        + 0.0001242229 * max(0.0, 173.1022 - Q.mass_top40) * max(0.0, Q.mass_displaced5 - 0.0) / 311.5677   # +0.0%  mass_top40 < 173.1 and mass_displaced5 > 0
        - 0.0001047902 * max(0.0, 0.3112717 - Q.sj4_dr_min) * max(0.0, 0.04589821 - Q.td0_30) / 0.01139648   # -0.0%  sj4_dr_min < 0.3113 and td0_30 < 0.0459
        + 8.606192e-05 * max(0.0, 0.06228948 - Q.M2) * max(0.0, Q.eta_37 - 0.0) / 5.820814e-05   # +0.0%  M2 < 0.06229 and eta_37 > 0
        + 6.379315e-05 * max(0.0, 0.06416437 - Q.z_displaced3) * max(0.0, 0.2370407 - Q.z_neutral_had) / 0.002830821   # +0.0%  z_displaced3 < 0.06416 and z_neutral_had < 0.237
        + 4.643644e-05 * max(0.0, -0.1170754 - Q.tdz_1) / 0.01260556   # +0.0%  tdz_1 < -0.1171
        + 4.265721e-05 * max(0.0, Q.n_electron - 1.0) / 0.08167   # +0.0%  n_electron > 1
        - 2.949081e-05 * max(0.0, 0.1681173 - Q.z_displaced3) * max(0.0, Q.ismuon_2 - 0.0) / 0.0008027911   # -0.0%  z_displaced3 < 0.1681 and ismuon_2 > 0
        - 1.914998e-05 * max(0.0, Q.e2 - 0.04393457) * max(0.0, Q.iselectron_11 - 0.0) / 0.0002408766   # -0.0%  e2 > 0.04393 and iselectron_11 > 0
        - 1.26465e-05 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.iselectron_12 - 0.0) / 9.674675e-07   # -0.0%  e3_b2 < 0.0002537 and iselectron_12 > 0
        + 9.768817e-06 * max(0.0, Q.mass_top40 - 155.8928) * max(0.0, -1.411949 - Q.lund2_lndelta) / 0.1276465   # +0.0%  mass_top40 > 155.9 and lund2_lndelta < -1.412
        + 7.961718e-06 * max(0.0, Q.mass_top15 - 80.3877) * max(0.0, Q.tdz_29 - -0.09486766) / 1.402767   # +0.0%  mass_top15 > 80.39 and tdz_29 > -0.09487
        - 5.84732e-06 * max(0.0, Q.n_s3d_above_3 - 6.0) * max(0.0, Q.ismuon_24 - 0.0) / 0.002623333   # -0.0%  n_s3d_above_3 > 6 and ismuon_24 > 0
        + 1.267283e-06 * max(0.0, Q.n_s3d_above_10 - 1.0) * max(0.0, Q.ismuon_27 - 0.0) / 0.004473333   # +0.0%  n_s3d_above_10 > 1 and ismuon_27 > 0
    )
    return z


def neuron_84(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-7.426966e-05
    )
    return z


def neuron_85(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.161996e-07
    )
    return z


def neuron_86(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.710251e-05
    )
    return z


def neuron_87(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.615846e-06
    )
    return z


def neuron_88(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.031722e-07
    )
    return z


def neuron_89(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.739162e-06
    )
    return z


def neuron_90(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.0009407948
    )
    return z


def neuron_91(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.992527e-06
    )
    return z


def neuron_92(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.723402e-06
    )
    return z


def neuron_93(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.443295e-07
    )
    return z


def neuron_94(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.322102e-06
    )
    return z


def neuron_95(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.400555e-07
    )
    return z


def neuron_96(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.32597e-07
    )
    return z


def neuron_97(Q):
    # scale S = 21.24; each line: share * term / its average size
    z = 21.23923 * (-0.026621
        + 0.1368489 * max(0.0, 119.4443 - Q.sd_mass) / 31.47389   # +13.7%  sd_mass < 119.4
        - 0.09100187 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -9.1%  mass < 164.4
        + 0.08941886 * max(0.0, 39.09615 - Q.mass_displaced3) / 31.84697   # +8.9%  mass_displaced3 < 39.1
        - 0.08672292 * max(0.0, 154.5947 - Q.sd_mass) / 60.90029   # -8.7%  sd_mass < 154.6
        - 0.07493547 * max(0.0, 115.7091 - Q.sd_mass) / 28.95052   # -7.5%  sd_mass < 115.7
        + 0.05993405 * max(0.0, 88.81751 - Q.sd_mass) / 15.02663   # +6.0%  sd_mass < 88.82
        - 0.05065472 * max(0.0, 78.4753 - Q.sd_mass) / 11.39114   # -5.1%  sd_mass < 78.48
        + 0.04138635 * max(0.0, 18.80005 - Q.mass_displaced3) * max(0.0, 175.9957 - Q.sip_3d_3) / 2083.071   # +4.1%  mass_displaced3 < 18.8 and sip_3d_3 < 176
        - 0.04120449 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 175.9957 - Q.sip_3d_3) / 4770.711   # -4.1%  mass_displaced3 < 39.1 and sip_3d_3 < 176
        + 0.02026662 * max(0.0, Q.n_sd0_above_3 - 1.0) / 2.484937   # +2.0%  n_sd0_above_3 > 1
        + 0.01935039 * max(0.0, 0.0007909605 - Q.e3_b2) / 0.0006645338   # +1.9%  e3_b2 < 0.000791
        - 0.01927926 * max(0.0, 18.80005 - Q.mass_displaced3) / 13.42744   # -1.9%  mass_displaced3 < 18.8
        + 0.01611188 * max(0.0, 120.6471 - Q.sj3_pair_mass_max) / 31.70826   # +1.6%  sj3_pair_mass_max < 120.6
        + 0.0155299 * max(0.0, 123.7928 - Q.mass) / 19.43787   # +1.6%  mass < 123.8
        - 0.01483576 * max(0.0, 0.0002536827 - Q.e3_b2) / 0.0001737301   # -1.5%  e3_b2 < 0.0002537
        + 0.01386651 * max(0.0, 161.1264 - Q.mass_top50) / 50.68933   # +1.4%  mass_top50 < 161.1
        - 0.01241698 * max(0.0, Q.n_s3d_above_3 - 3.0) / 1.66923   # -1.2%  n_s3d_above_3 > 3
        + 0.01224499 * max(0.0, 62.03827 - Q.sd_mass) / 7.413901   # +1.2%  sd_mass < 62.04
        - 0.01190243 * max(0.0, 100.8525 - Q.sd_mass) / 20.48378   # -1.2%  sd_mass < 100.9
        + 0.01029553 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +1.0%  mass < 95.15
        - 0.0102323 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 27.3236 - Q.lep_ptrel) / 689.887   # -1.0%  mass_displaced3 < 39.1 and lep_ptrel < 27.32
        - 0.009767138 * max(0.0, 0.1403354 - Q.z_muon) / 0.1213357   # -1.0%  z_muon < 0.1403
        - 0.009341571 * max(0.0, 72.27436 - Q.sd_mass) / 9.692364   # -0.9%  sd_mass < 72.27
        + 0.007213837 * max(0.0, 126.8853 - Q.mass_top40) / 23.76162   # +0.7%  mass_top40 < 126.9
        + 0.007077876 * max(0.0, 100.4835 - Q.mass) / 8.115061   # +0.7%  mass < 100.5
        - 0.006856023 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # -0.7%  max_abs_d0 < 5.812
        + 0.006415935 * max(0.0, 126.0 - Q.n_pairs_kt_above_3) / 59.61773   # +0.6%  n_pairs_kt_above_3 < 126
        + 0.005656221 * max(0.0, 325.0 - Q.n_pairs_kt_above_1) / 103.3282   # +0.6%  n_pairs_kt_above_1 < 325
        + 0.005404783 * max(0.0, 5.8125 - Q.max_abs_d0) * max(0.0, 7.345216 - Q.sip_3d_3) / 10.83619   # +0.5%  max_abs_d0 < 5.812 and sip_3d_3 < 7.345
        - 0.005132813 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # -0.5%  sip_3d_2 < 447.1
        - 0.004262894 * max(0.0, Q.lund_max_lndelta - -1.388411) / 0.3966091   # -0.4%  lund_max_lndelta > -1.388
        + 0.004241609 * max(0.0, Q.sj2_mass1 - 53.51926) / 6.081546   # +0.4%  sj2_mass1 > 53.52
        + 0.004232945 * Q.n_pairs_kt_above_10 / 6.942877   # +0.4%  n_pairs_kt_above_10
        - 0.004093524 * max(0.0, Q.n_sd0_above_2 - 5.0) / 0.9822033   # -0.4%  n_sd0_above_2 > 5
        - 0.003931257 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.n_pairs_kt_above_10 - 2.0) / 152.7677   # -0.4%  mass_displaced3 < 39.1 and n_pairs_kt_above_10 > 2
        - 0.003804475 * max(0.0, Q.sj2_mass1 - 77.42768) / 2.022907   # -0.4%  sj2_mass1 > 77.43
        + 0.003702807 * max(0.0, 19.0 - Q.n_photon) / 4.85136   # +0.4%  n_photon < 19
        - 0.003504415 * max(0.0, 0.8501496 - Q.sj3_pairmax_over_m) / 0.06258068   # -0.4%  sj3_pairmax_over_m < 0.8501
        + 0.003223909 * max(0.0, Q.lund3_lndelta - -1.902701) / 0.5262124   # +0.3%  lund3_lndelta > -1.903
        - 0.002972334 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, Q.n_s3d_above_3 - 1.0) / 116.8893   # -0.3%  mass_top50 < 161.1 and n_s3d_above_3 > 1
        + 0.002634029 * max(0.0, 109.8208 - Q.sj3_pair_mass_max) / 23.01361   # +0.3%  sj3_pair_mass_max < 109.8
        + 0.00258329 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.jet_charge_k03 - -0.05990128) / 5.503613e-05   # +0.3%  e3_b2 < 0.0002537 and jet_charge_k03 > -0.0599
        - 0.002581187 * max(0.0, 79.27954 - Q.mass_top50) / 2.845865   # -0.3%  mass_top50 < 79.28
        + 0.00257176 * max(0.0, Q.lne_0 - 5.157617) / 0.277972   # +0.3%  lne_0 > 5.158
        + 0.00214621 * max(0.0, 7.203613 - Q.pair_max_lnm2) / 0.7155444   # +0.2%  pair_max_lnm2 < 7.204
        - 0.002124372 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) / 5.328557   # -0.2%  n_pairs_kt_above_3 < 34
        - 0.0020462 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, 175.9957 - Q.sip_3d_3) / 7530.903   # -0.2%  mass_top50 < 161.1 and sip_3d_3 < 176
        - 0.00192549 * max(0.0, 0.6079631 - Q.max_dr) / 0.05926836   # -0.2%  max_dr < 0.608
        - 0.001897257 * max(0.0, 115.614 - Q.mass_top50) / 15.03711   # -0.2%  mass_top50 < 115.6
        + 0.00180915 * max(0.0, 126.8853 - Q.mass_top40) * max(0.0, Q.lne_5 - 2.737811) / 19.53995   # +0.2%  mass_top40 < 126.9 and lne_5 > 2.738
        - 0.001791589 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, 0.3261071 - Q.z_displaced3) / 4.035741e-05   # -0.2%  e3_b2 < 0.0002537 and z_displaced3 < 0.3261
        - 0.001638921 * max(0.0, Q.z_neutral_had - 0.07573803) / 0.08719919   # -0.2%  z_neutral_had > 0.07574
        + 0.001637539 * max(0.0, Q.jet_charge_k05 - 0.07435708) / 0.1495371   # +0.2%  jet_charge_k05 > 0.07436
        - 0.001576405 * max(0.0, 126.8853 - Q.mass_top40) * max(0.0, Q.n_muon - 0.0) / 4.915939   # -0.2%  mass_top40 < 126.9 and n_muon > 0
        - 0.001510326 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # -0.2%  sj3_pair_mass_max < 73.25
        - 0.001497412 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, 0.3565533 - Q.tau21_b2) / 2.214781e-05   # -0.1%  e3_b2 < 0.0002537 and tau21_b2 < 0.3566
        - 0.001466526 * max(0.0, Q.sj3_pair_mass_min - 59.49644) / 2.657693   # -0.1%  sj3_pair_mass_min > 59.5
        + 0.001456552 * max(0.0, 2.68613 - Q.sj3_mass3) / 0.4824158   # +0.1%  sj3_mass3 < 2.686
        + 0.001436632 * max(0.0, Q.sj2_mass1 - 91.2852) / 1.019309   # +0.1%  sj2_mass1 > 91.29
        - 0.001296126 * max(0.0, Q.z_displaced5 - 0.1339658) / 0.03671738   # -0.1%  z_displaced5 > 0.134
        - 0.001130908 * max(0.0, 0.6206221 - Q.tau32) / 0.05684264   # -0.1%  tau32 < 0.6206
        + 0.0009929116 * max(0.0, Q.lund_max_lndelta - -0.6897565) / 0.03634831   # +0.1%  lund_max_lndelta > -0.6898
        + 0.0009606996 * max(0.0, 0.6206221 - Q.tau32) * max(0.0, 0.6056728 - Q.z_charged_had) / 0.008850292   # +0.1%  tau32 < 0.6206 and z_charged_had < 0.6057
        - 0.0009207337 * max(0.0, 9.621843e-05 - Q.e3_b2) / 4.881439e-05   # -0.1%  e3_b2 < 9.622e-05
        - 0.0008155241 * max(0.0, 3.521178 - Q.lund_max_lnkt) / 0.1676122   # -0.1%  lund_max_lnkt < 3.521
        + 0.0007413711 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 586.125 - Q.sum_pt_top5) / 6983.553   # +0.1%  mass_displaced3 < 39.1 and sum_pt_top5 < 586.1
        - 0.0006768005 * max(0.0, Q.e2_b05 - 0.1510354) / 0.01899022   # -0.1%  e2_b05 > 0.151
        + 0.0006684789 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.n_dr_0p4_up - 8.0) / 9.895202e-05   # +0.1%  e3_b2 < 0.0002537 and n_dr_0p4_up > 8
        - 0.0005671216 * max(0.0, Q.lund_max_lndelta - -1.388411) * max(0.0, Q.lnerel_4 - -3.310904) / 0.1430312   # -0.1%  lund_max_lndelta > -1.388 and lnerel_4 > -3.311
        + 0.0005309777 * max(0.0, 115.614 - Q.mass_top50) * max(0.0, 1.0 - Q.n_lepton) / 9.314921   # +0.1%  mass_top50 < 115.6 and n_lepton < 1
        + 0.0005137207 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.mass_charged - 84.24838) / 0.0002363463   # +0.1%  e3_b2 < 0.0002537 and mass_charged > 84.25
        + 0.0004986376 * max(0.0, Q.D2 - 3.597891) / 0.2539167   # +0.0%  D2 > 3.598
        + 0.0004917515 * max(0.0, 18.80005 - Q.mass_displaced3) * max(0.0, Q.mass_2photon - 9.82024) / 23.24627   # +0.0%  mass_displaced3 < 18.8 and mass_2photon > 9.82
        + 0.0004653242 * max(0.0, 34.0 - Q.n_pairs_kt_above_3) * max(0.0, 2.286291 - Q.D2_b2) / 2.924988   # +0.0%  n_pairs_kt_above_3 < 34 and D2_b2 < 2.286
        - 0.0004382923 * max(0.0, Q.psi_0p1 - 0.7386202) / 0.03082667   # -0.0%  psi_0p1 > 0.7386
        - 0.000366278 * max(0.0, Q.pair_mean_lnm2 - 2.09826) / 0.7077607   # -0.0%  pair_mean_lnm2 > 2.098
        + 0.0003613594 * max(0.0, Q.e2 - 0.09764648) / 0.007060655   # +0.0%  e2 > 0.09765
        - 0.0002593699 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, -0.6526285 - Q.lead_ch_sdz) / 232.1779   # -0.0%  sip_3d_2 < 447.1 and lead_ch_sdz < -0.6526
        - 0.0002586552 * max(0.0, Q.sj3_dr_min - 0.3190414) / 0.01982359   # -0.0%  sj3_dr_min > 0.319
        + 0.0002130429 * max(0.0, 447.0873 - Q.sip_3d_2) * max(0.0, Q.td0_33 - -0.04023096) / 15.38564   # +0.0%  sip_3d_2 < 447.1 and td0_33 > -0.04023
        + 0.0001849267 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.iselectron_0 - 0.0) / 2.911082   # +0.0%  mass_displaced3 < 39.1 and iselectron_0 > 0
        - 0.0001653578 * max(0.0, 100.4835 - Q.mass) * max(0.0, Q.iselectron_0 - 0.0) / 0.7641534   # -0.0%  mass < 100.5 and iselectron_0 > 0
        - 0.0001157906 * max(0.0, 0.01015545 - Q.M2_b2) / 0.0003136992   # -0.0%  M2_b2 < 0.01016
        - 0.0001115491 * max(0.0, Q.n_pt_above_5 - 27.0) / 1.148523   # -0.0%  n_pt_above_5 > 27
        - 9.509824e-05 * max(0.0, 0.8501496 - Q.sj3_pairmax_over_m) * max(0.0, Q.dr_53 - 0.0) / 0.004664772   # -0.0%  sj3_pairmax_over_m < 0.8501 and dr_53 > 0
        + 8.457433e-05 * max(0.0, 161.1264 - Q.mass_top50) * max(0.0, Q.min_pair_mass - 3.369962) / 40.0642   # +0.0%  mass_top50 < 161.1 and min_pair_mass > 3.37
        - 8.443553e-05 * max(0.0, Q.n_sd0_above_3 - 1.0) * max(0.0, Q.sj3_dr13 - 0.6229991) / 0.06211718   # -0.0%  n_sd0_above_3 > 1 and sj3_dr13 > 0.623
        + 7.284725e-05 * max(0.0, 0.1244374 - Q.N2_b2) / 0.01729494   # +0.0%  N2_b2 < 0.1244
        + 7.060458e-05 * max(0.0, 0.0007909605 - Q.e3_b2) * max(0.0, Q.dr_28 - 0.3368505) / 1.879537e-05   # +0.0%  e3_b2 < 0.000791 and dr_28 > 0.3369
        + 7.026868e-05 * max(0.0, 0.1244374 - Q.N2_b2) * max(0.0, -0.1424561 - Q.eta_29) / 0.0002260608   # +0.0%  N2_b2 < 0.1244 and eta_29 < -0.1425
        + 6.154933e-05 * max(0.0, Q.lund_max_lndelta - -1.388411) * max(0.0, 0.09541201 - Q.td0_7) / 0.04689523   # +0.0%  lund_max_lndelta > -1.388 and td0_7 < 0.09541
        - 3.929191e-05 * max(0.0, 0.0002536827 - Q.e3_b2) * max(0.0, Q.sj3_dr13 - 0.7462286) / 9.554771e-07   # -0.0%  e3_b2 < 0.0002537 and sj3_dr13 > 0.7462
        - 2.512192e-05 * max(0.0, Q.z_neutral_had - 0.07573803) * max(0.0, Q.td0_23 - 0.0) / 0.002247624   # -0.0%  z_neutral_had > 0.07574 and td0_23 > 0
        - 2.008603e-05 * max(0.0, Q.sj2_mass1 - 77.42768) * max(0.0, 0.8678264 - Q.D2_b2) / 0.003578708   # -0.0%  sj2_mass1 > 77.43 and D2_b2 < 0.8678
        - 1.140239e-05 * max(0.0, Q.z_displaced5 - 0.1339658) * max(0.0, Q.C3_b2 - 0.04229114) / 9.919964e-05   # -0.0%  z_displaced5 > 0.134 and C3_b2 > 0.04229
        + 7.967624e-06 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, Q.d0err_10 - 0.02780151) / 0.0002782402   # +0.0%  n_s3d_above_3 > 3 and d0err_10 > 0.0278
        + 2.61192e-06 * max(0.0, 109.8208 - Q.sj3_pair_mass_max) * max(0.0, Q.lnpt_44 - -18.42068) / 82.7679   # +0.0%  sj3_pair_mass_max < 109.8 and lnpt_44 > -18.42
        + 1.017587e-06 * max(0.0, 0.8501496 - Q.sj3_pairmax_over_m) * max(0.0, Q.ismuon_37 - 0.0) / 5.35795e-05   # +0.0%  sj3_pairmax_over_m < 0.8501 and ismuon_37 > 0
    )
    return z


def neuron_98(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.96437e-07
    )
    return z


def neuron_99(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-0.001934247
    )
    return z


def neuron_100(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.267939e-06
    )
    return z


def neuron_101(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (8.878886e-06
    )
    return z


def neuron_102(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.486108e-06
    )
    return z


def neuron_103(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.416626e-06
    )
    return z


def neuron_104(Q):
    # scale S = 6.998; each line: share * term / its average size
    z = 6.998282 * (-0.2650225
        + 0.1159611 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # +11.6%  sd_mass > 78.48
        - 0.1053762 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # -10.5%  sd_mass > 88.82
        + 0.07147174 * max(0.0, Q.mass_displaced3 - 3.208089) / 6.234439   # +7.1%  mass_displaced3 > 3.208
        + 0.04815409 * max(0.0, -1.352792 - Q.pair_mean_lndelta) / 0.8695512   # +4.8%  pair_mean_lndelta < -1.353
        + 0.03668201 * max(0.0, 8.0 - Q.n_s3d_above_10) / 5.470973   # +3.7%  n_s3d_above_10 < 8
        + 0.03366319 * max(0.0, 120.653 - Q.mass) / 17.47277   # +3.4%  mass < 120.7
        - 0.03324332 * max(0.0, Q.mass_displaced3 - 6.341631) / 5.164503   # -3.3%  mass_displaced3 > 6.342
        + 0.03223617 * max(0.0, 149.0507 - Q.mass) / 39.08704   # +3.2%  mass < 149.1
        + 0.02953625 * max(0.0, 67.0 - Q.n_particles) / 28.06102   # +3.0%  n_particles < 67
        + 0.02778208 * max(0.0, 155.8928 - Q.mass_top40) / 48.12072   # +2.8%  mass_top40 < 155.9
        + 0.02323014 * max(0.0, 6.0 - Q.n_sd0_above_3) / 3.16582   # +2.3%  n_sd0_above_3 < 6
        + 0.02068789 * max(0.0, 132.5189 - Q.mass_top40) / 28.13525   # +2.1%  mass_top40 < 132.5
        + 0.0201082 * max(0.0, 0.5562194 - Q.sd_rg) / 0.2284833   # +2.0%  sd_rg < 0.5562
        + 0.01926834 * max(0.0, Q.z_top15_slots - 0.9212656) / 0.01314087   # +1.9%  z_top15_slots > 0.9213
        - 0.01921544 * max(0.0, Q.N2_b05 - 0.3667049) / 0.07978604   # -1.9%  N2_b05 > 0.3667
        + 0.01863691 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, 49.0 - Q.n_real_top50) / 74.79127   # +1.9%  n_s3d_above_10 < 8 and n_real_top50 < 49
        - 0.01850066 * max(0.0, 146.0 - Q.n_pairs_kt_above_1) / 20.10573   # -1.9%  n_pairs_kt_above_1 < 146
        - 0.01634827 * Q.n_charged_pt_above_1 / 18.83947   # -1.6%  n_charged_pt_above_1
        - 0.01626092 * max(0.0, 4.0 - Q.sd_nremoved) / 2.43384   # -1.6%  sd_nremoved < 4
        + 0.01258305 * max(0.0, 67.0 - Q.n_particles) * max(0.0, 2.0 - Q.n_lepton) / 39.27554   # +1.3%  n_particles < 67 and n_lepton < 2
        - 0.01179458 * max(0.0, 6.185635 - Q.lep_iso) / 4.883283   # -1.2%  lep_iso < 6.186
        + 0.01113685 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # +1.1%  sd_mass > 106.8
        + 0.009989355 * max(0.0, 0.191059 - Q.C2) / 0.05400117   # +1.0%  C2 < 0.1911
        + 0.009597364 * max(0.0, 149.0507 - Q.mass) * max(0.0, Q.lne_0 - 5.157617) / 13.38687   # +1.0%  mass < 149.1 and lne_0 > 5.158
        + 0.009457342 * max(0.0, 10.0 - Q.n_dr_0p4_up) / 5.21002   # +0.9%  n_dr_0p4_up < 10
        - 0.009420706 * max(0.0, 4.046329 - Q.lund_max_lnkt) / 0.370868   # -0.9%  lund_max_lnkt < 4.046
        - 0.009311178 * max(0.0, 7.0 - Q.n_sd0_above_2) / 3.29892   # -0.9%  n_sd0_above_2 < 7
        + 0.009242297 * max(0.0, 0.02949822 - Q.tau4) / 0.003969373   # +0.9%  tau4 < 0.0295
        + 0.009197982 * Q.mass_neutral / 45.13363   # +0.9%  mass_neutral
        + 0.008648628 * max(0.0, Q.sd_rg - 0.3025746) / 0.08639404   # +0.9%  sd_rg > 0.3026
        - 0.008046026 * max(0.0, 0.4680886 - Q.tau32_b2) / 0.05978538   # -0.8%  tau32_b2 < 0.4681
        - 0.00777013 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) / 9.886423   # -0.8%  n_pairs_kt_above_3 < 47
        - 0.007650669 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # -0.8%  sip_3d_2 < 226.3
        - 0.007649189 * max(0.0, 4.0 - Q.n_neutral_had) / 0.8492867   # -0.8%  n_neutral_had < 4
        + 0.00744029 * max(0.0, Q.pair_max_lnm2 - 6.052324) / 0.7112379   # +0.7%  pair_max_lnm2 > 6.052
        - 0.006680475 * max(0.0, Q.mass_top5 - 17.63354) / 26.12147   # -0.7%  mass_top5 > 17.63
        - 0.006546836 * max(0.0, Q.e3 - 0.001589861) / 0.0004282381   # -0.7%  e3 > 0.00159
        - 0.005749941 * max(0.0, 5.0 - Q.n_s3d_above_3) * max(0.0, 0.1404188 - Q.C2_b2) / 0.1613146   # -0.6%  n_s3d_above_3 < 5 and C2_b2 < 0.1404
        + 0.005727077 * max(0.0, 0.3190414 - Q.sj3_dr_min) / 0.1328834   # +0.6%  sj3_dr_min < 0.319
        - 0.005640008 * max(0.0, Q.mass_displaced3 - 26.78691) / 1.554829   # -0.6%  mass_displaced3 > 26.79
        + 0.00555197 * max(0.0, Q.mass_over_sum_pt - 0.215596) / 0.01424531   # +0.6%  mass_over_sum_pt > 0.2156
        + 0.005407338 * max(0.0, -1.352792 - Q.pair_mean_lndelta) * max(0.0, Q.lund1_lndelta - -0.6177752) / 0.1862499   # +0.5%  pair_mean_lndelta < -1.353 and lund1_lndelta > -0.6178
        + 0.005359592 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +0.5%  sd_mass > 154.6
        + 0.005050071 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # +0.5%  lep_ptrel > 27.32
        - 0.004942986 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.5%  lep_ptrel > 43.21
        + 0.00488139 * max(0.0, 0.2326317 - Q.C2_b05) / 0.01179395   # +0.5%  C2_b05 < 0.2326
        - 0.004651809 * max(0.0, -1.352792 - Q.pair_mean_lndelta) * max(0.0, Q.lep_ptrel - 12.15228) / 1.748073   # -0.5%  pair_mean_lndelta < -1.353 and lep_ptrel > 12.15
        + 0.004554237 * max(0.0, Q.tau21 - 0.5797033) / 0.03370208   # +0.5%  tau21 > 0.5797
        + 0.004530349 * max(0.0, Q.jet_abs_eta - 0.6771968) / 0.2378129   # +0.5%  jet_abs_eta > 0.6772
        + 0.00435368 * max(0.0, Q.mass_displaced3 - 3.208089) * max(0.0, 2.885768e-05 - Q.ecf_g43) / 0.000134714   # +0.4%  mass_displaced3 > 3.208 and ecf_g43 < 2.886e-05
        - 0.004346545 * max(0.0, Q.n_lund - 11.0) / 1.107333   # -0.4%  n_lund > 11
        - 0.00409432 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.221436 - Q.lep_z) / 1.248772   # -0.4%  n_pairs_kt_above_3 < 47 and lep_z < 0.2214
        + 0.003854761 * max(0.0, 6.0 - Q.n_sd0_above_3) * max(0.0, Q.n_dr_0p4_up - 0.0) / 16.68924   # +0.4%  n_sd0_above_3 < 6 and n_dr_0p4_up > 0
        + 0.003224536 * max(0.0, 99.20396 - Q.mass_charged) / 37.01153   # +0.3%  mass_charged < 99.2
        + 0.003190887 * max(0.0, Q.N2 - 0.3245983) / 0.02636523   # +0.3%  N2 > 0.3246
        - 0.003029867 * max(0.0, 146.0 - Q.n_pairs_kt_above_1) * max(0.0, 4.878859 - Q.D2) / 53.98141   # -0.3%  n_pairs_kt_above_1 < 146 and D2 < 4.879
        + 0.003010264 * max(0.0, Q.n_lund_kt_above_5 - 2.0) / 0.4166533   # +0.3%  n_lund_kt_above_5 > 2
        + 0.002711588 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.2601259 - Q.z_displaced3) / 1.851029   # +0.3%  n_pairs_kt_above_3 < 47 and z_displaced3 < 0.2601
        + 0.002640255 * max(0.0, Q.D2 - 3.597891) / 0.2539167   # +0.3%  D2 > 3.598
        + 0.002532606 * max(0.0, Q.lep_ptrel - 18.7678) / 2.829169   # +0.3%  lep_ptrel > 18.77
        - 0.002440138 * max(0.0, 67.0 - Q.n_particles) * max(0.0, Q.dr01 - 0.06433539) / 3.156584   # -0.2%  n_particles < 67 and dr01 > 0.06434
        - 0.002390187 * max(0.0, 0.07264571 - Q.tau2) / 0.01634431   # -0.2%  tau2 < 0.07265
        + 0.002373153 * max(0.0, 0.3886647 - Q.sj3_mass3) / 0.03562942   # +0.2%  sj3_mass3 < 0.3887
        + 0.001989416 * max(0.0, Q.M3 - 0.05919662) / 0.001232267   # +0.2%  M3 > 0.0592
        - 0.001965128 * max(0.0, 5.0 - Q.n_s3d_above_3) / 2.12335   # -0.2%  n_s3d_above_3 < 5
        - 0.001865511 * max(0.0, Q.sj2_mass1 - 53.51926) / 6.081546   # -0.2%  sj2_mass1 > 53.52
        - 0.001825387 * max(0.0, 47.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.dr02 - 0.2159556) / 0.5243339   # -0.2%  n_pairs_kt_above_3 < 47 and dr02 > 0.216
        - 0.001574506 * max(0.0, 0.2377332 - Q.tau21) / 0.01385845   # -0.2%  tau21 < 0.2377
        + 0.001557671 * max(0.0, 3.0 - Q.n_sdz_above_5) / 1.349857   # +0.2%  n_sdz_above_5 < 3
        - 0.001531933 * max(0.0, Q.mass_displaced3 - 3.208089) * max(0.0, 0.02148541 - Q.lam2) / 0.09278072   # -0.2%  mass_displaced3 > 3.208 and lam2 < 0.02149
        - 0.001481185 * max(0.0, -0.5570337 - Q.jet_charge_k03) / 0.07719303   # -0.1%  jet_charge_k03 < -0.557
        - 0.001374583 * max(0.0, 0.01782783 - Q.z_displaced3) / 0.005025534   # -0.1%  z_displaced3 < 0.01783
        - 0.001218823 * max(0.0, 0.04483276 - Q.tau2) / 0.004264343   # -0.1%  tau2 < 0.04483
        + 0.001092012 * max(0.0, 146.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 7.535107   # +0.1%  n_pairs_kt_above_1 < 146 and n_lund_kt_above_5 > 1
        + 0.001059711 * max(0.0, 0.4680886 - Q.tau32_b2) * max(0.0, 0.003965728 - Q.ecf_g32) / 7.486955e-05   # +0.1%  tau32_b2 < 0.4681 and ecf_g32 < 0.003966
        + 0.001030921 * max(0.0, 0.135772 - Q.tau21) / 0.001647604   # +0.1%  tau21 < 0.1358
        + 0.0009415005 * max(0.0, Q.z_top15_slots - 0.9212656) * max(0.0, 0.03430176 - Q.dzerr_1) / 0.0001974266   # +0.1%  z_top15_slots > 0.9213 and dzerr_1 < 0.0343
        - 0.0008823024 * max(0.0, Q.sd_mass - 175.9333) / 1.731098   # -0.1%  sd_mass > 175.9
        + 0.0005846436 * max(0.0, Q.lund_max_lndelta - -0.529318) / 0.01270035   # +0.1%  lund_max_lndelta > -0.5293
        - 0.0004634646 * max(0.0, Q.n_for_90pct - 39.0) / 0.30656   # -0.0%  n_for_90pct > 39
        - 0.0004304664 * max(0.0, 3.0 - Q.n_pairs_kt_above_10) / 0.88044   # -0.0%  n_pairs_kt_above_10 < 3
        + 0.0003605378 * max(0.0, 6.216954e-05 - Q.ecf_g41) / 1.118908e-06   # +0.0%  ecf_g41 < 6.217e-05
        + 0.0003258926 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, Q.sj4_pair_mass_max - 101.849) / 18.13026   # +0.0%  n_s3d_above_10 < 8 and sj4_pair_mass_max > 101.8
        + 0.0002689848 * max(0.0, 4.0 - Q.sd_nremoved) * max(0.0, Q.eta_4 - 0.05764771) / 0.06679296   # +0.0%  sd_nremoved < 4 and eta_4 > 0.05765
        - 0.0002584264 * max(0.0, Q.sd_mass - 175.9333) * max(0.0, Q.lnptrel_77 - -18.42068) / 3.917776   # -0.0%  sd_mass > 175.9 and lnptrel_77 > -18.42
        + 0.0002465374 * max(0.0, 0.4680886 - Q.tau32_b2) * max(0.0, Q.eta_1 - 0.01268768) / 0.002230802   # +0.0%  tau32_b2 < 0.4681 and eta_1 > 0.01269
        + 0.0002399648 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, Q.lep_ptrel - 18.7678) / 13.83744   # +0.0%  n_s3d_above_10 < 8 and lep_ptrel > 18.77
        + 0.0001203244 * max(0.0, Q.C2_b2 - 0.221369) / 0.0106764   # +0.0%  C2_b2 > 0.2214
        - 9.661813e-05 * max(0.0, Q.sum_charge - 4.0) / 0.03763333   # -0.0%  sum_charge > 4
        + 8.147033e-05 * max(0.0, 0.0318986 - Q.e2) / 0.0005225451   # +0.0%  e2 < 0.0319
        + 7.283345e-05 * max(0.0, Q.sd_mass - 78.4753) * max(0.0, 1.0 - Q.charge_25) / 29.99101   # +0.0%  sd_mass > 78.48 and charge_25 < 1
        + 7.021483e-05 * max(0.0, -0.02912079 - Q.td0_38) / 0.01076472   # +0.0%  td0_38 < -0.02912
        + 5.76181e-05 * max(0.0, Q.n_lund - 11.0) * max(0.0, 0.1278076 - Q.phi_21) / 0.176051   # +0.0%  n_lund > 11 and phi_21 < 0.1278
        + 4.118475e-05 * max(0.0, 3.0 - Q.n_pairs_kt_above_10) * max(0.0, 0.0328064 - Q.phi_29) / 0.06656675   # +0.0%  n_pairs_kt_above_10 < 3 and phi_29 < 0.03281
        + 3.716702e-05 * max(0.0, 4.0 - Q.sd_nremoved) * max(0.0, Q.eta_28 - -0.3166626) / 0.7833793   # +0.0%  sd_nremoved < 4 and eta_28 > -0.3167
        + 3.219771e-05 * max(0.0, Q.mass_over_sum_pt - 0.215596) * max(0.0, 0.1211197 - Q.dr_9) / 8.201701e-05   # +0.0%  mass_over_sum_pt > 0.2156 and dr_9 < 0.1211
        + 2.99868e-05 * max(0.0, Q.sd_mass - 175.9333) * max(0.0, 0.0 - Q.charge_6) / 0.5226794   # +0.0%  sd_mass > 175.9 and charge_6 < 0
        + 1.280761e-05 * max(0.0, 6.185635 - Q.lep_iso) * max(0.0, Q.iselectron_27 - 0.0) / 0.004863385   # +0.0%  lep_iso < 6.186 and iselectron_27 > 0
        + 1.266774e-05 * max(0.0, Q.sd_mass - 78.4753) * max(0.0, 0.0 - Q.tdz_74) / 0.05707131   # +0.0%  sd_mass > 78.48 and tdz_74 < 0
    )
    return z


def neuron_105(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.732283e-05
    )
    return z


def neuron_106(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.161276e-05
    )
    return z


def neuron_107(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (1.158186e-05
    )
    return z


def neuron_108(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-4.469853e-06
    )
    return z


def neuron_109(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.224127e-07
    )
    return z


def neuron_110(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.577024e-06
    )
    return z


def neuron_111(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (3.66789e-06
    )
    return z


def neuron_112(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (5.232593e-06
    )
    return z


def neuron_113(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.665687e-05
    )
    return z


def neuron_114(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (4.677441e-06
    )
    return z


def neuron_115(Q):
    # scale S = 13.96; each line: share * term / its average size
    z = 13.96204 * (0.0371346
        + 0.07854498 * max(0.0, 182.8592 - Q.mass) / 69.61635   # +7.9%  mass < 182.9
        + 0.0640445 * max(0.0, 129.5874 - Q.mass_top50) / 24.20267   # +6.4%  mass_top50 < 129.6
        - 0.05636038 * max(0.0, 164.4374 - Q.mass) / 52.54489   # -5.6%  mass < 164.4
        + 0.0557195 * max(0.0, 39.09615 - Q.mass_displaced3) / 31.84697   # +5.6%  mass_displaced3 < 39.1
        - 0.04546666 * max(0.0, 125.4927 - Q.mass_top50) / 21.23144   # -4.5%  mass_top50 < 125.5
        - 0.0440658 * max(0.0, 80.02563 - Q.sj3_pair_mass_min) / 44.19386   # -4.4%  sj3_pair_mass_min < 80.03
        - 0.04225988 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 0.3788785 - Q.z_neutral_had) / 7.382889   # -4.2%  mass_displaced3 < 39.1 and z_neutral_had < 0.3789
        - 0.04210596 * max(0.0, Q.pair_mean_lndelta - -3.421602) / 1.23252   # -4.2%  pair_mean_lndelta > -3.422
        + 0.04030191 * max(0.0, 0.0131933 - Q.e3_b05) / 0.006147996   # +4.0%  e3_b05 < 0.01319
        - 0.0379277 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.z_top20_slots - 0.7122459) / 11.06965   # -3.8%  mass < 164.4 and z_top20_slots > 0.7122
        + 0.03586271 * max(0.0, 120.653 - Q.mass) / 17.47277   # +3.6%  mass < 120.7
        - 0.03361054 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 0.0007909605 - Q.e3_b2) / 0.02145527   # -3.4%  mass_displaced3 < 39.1 and e3_b2 < 0.000791
        - 0.03349637 * max(0.0, 107.8099 - Q.sj4_pair_mass_max) / 29.80438   # -3.3%  sj4_pair_mass_max < 107.8
        + 0.02840399 * max(0.0, 0.120439 - Q.M2) / 0.04313262   # +2.8%  M2 < 0.1204
        - 0.02380463 * max(0.0, 0.4888886 - Q.LHA) / 0.09435016   # -2.4%  LHA < 0.4889
        + 0.01791571 * max(0.0, Q.tau1 - 0.1246227) / 0.07463455   # +1.8%  tau1 > 0.1246
        - 0.01784418 * max(0.0, 0.05919662 - Q.M3) / 0.02436955   # -1.8%  M3 < 0.0592
        - 0.01780536 * max(0.0, 28.89659 - Q.mass_2charged) / 18.56832   # -1.8%  mass_2charged < 28.9
        - 0.01770692 * max(0.0, 109.8208 - Q.sj3_pair_mass_max) / 23.01361   # -1.8%  sj3_pair_mass_max < 109.8
        + 0.01617424 * max(0.0, 6.767937 - Q.mass_2charged) / 3.01163   # +1.6%  mass_2charged < 6.768
        + 0.01518113 * max(0.0, 36.36236 - Q.sj3_mass1) / 17.59922   # +1.5%  sj3_mass1 < 36.36
        - 0.01390043 * max(0.0, 0.3190414 - Q.sj3_dr_min) / 0.1328834   # -1.4%  sj3_dr_min < 0.319
        + 0.01374334 * max(0.0, 0.05966366 - Q.M2_b2) / 0.0281728   # +1.4%  M2_b2 < 0.05966
        + 0.01148349 * max(0.0, 0.2099278 - Q.N2_b2) / 0.06000895   # +1.1%  N2_b2 < 0.2099
        + 0.008296244 * max(0.0, 0.1845735 - Q.sj4_dr_min) / 0.07908106   # +0.8%  sj4_dr_min < 0.1846
        + 0.006802405 * max(0.0, 2.911067 - Q.pair_max_lnkt) / 0.3583041   # +0.7%  pair_max_lnkt < 2.911
        - 0.00671894 * max(0.0, Q.n_lund_kt_above_5 - 1.0) / 1.107657   # -0.7%  n_lund_kt_above_5 > 1
        + 0.006683921 * max(0.0, 72.862 - Q.sj4_pair_mass_max) / 6.86631   # +0.7%  sj4_pair_mass_max < 72.86
        + 0.006400117 * max(0.0, Q.n_s3d_above_3 - 4.0) / 1.224997   # +0.6%  n_s3d_above_3 > 4
        - 0.00621509 * max(0.0, Q.z_top20_slots - 0.9417195) / 0.01519974   # -0.6%  z_top20_slots > 0.9417
        + 0.006201737 * max(0.0, 125.4927 - Q.mass_top50) * max(0.0, 1.0 - Q.n_lepton) / 12.84432   # +0.6%  mass_top50 < 125.5 and n_lepton < 1
        - 0.006182817 * max(0.0, 0.05966366 - Q.M2_b2) * max(0.0, Q.mass_charged - 37.42054) / 0.741033   # -0.6%  M2_b2 < 0.05966 and mass_charged > 37.42
        + 0.005821964 * max(0.0, 0.09733903 - Q.tau2) * max(0.0, 1.0 - Q.isphoton_41) / 0.02865177   # +0.6%  tau2 < 0.09734 and isphoton_41 < 1
        - 0.005615571 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 0.8709334 - Q.tau43) / 2.811381   # -0.6%  mass_displaced3 < 39.1 and tau43 < 0.8709
        - 0.005239182 * max(0.0, 94.51361 - Q.mass_top50) / 6.251054   # -0.5%  mass_top50 < 94.51
        + 0.005181542 * max(0.0, Q.tau32_b2 - 0.6908801) / 0.04407883   # +0.5%  tau32_b2 > 0.6909
        + 0.005033455 * max(0.0, 0.09733903 - Q.tau2) / 0.03188557   # +0.5%  tau2 < 0.09734
        + 0.004723597 * max(0.0, 0.09733903 - Q.tau2) * max(0.0, 27.3236 - Q.lep_ptrel) / 0.6620054   # +0.5%  tau2 < 0.09734 and lep_ptrel < 27.32
        - 0.004670207 * max(0.0, 12.0 - Q.n_dr_0_0p05) / 8.882307   # -0.5%  n_dr_0_0p05 < 12
        + 0.004642246 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, 175.9957 - Q.sip_3d_3) / 116.3453   # +0.5%  n_s3d_above_3 > 4 and sip_3d_3 < 176
        + 0.004620706 * max(0.0, Q.lep_z - 0.03021637) / 0.07491017   # +0.5%  lep_z > 0.03022
        - 0.004309602 * max(0.0, 447.0873 - Q.sip_3d_2) / 319.6838   # -0.4%  sip_3d_2 < 447.1
        + 0.004262748 * max(0.0, 0.2099278 - Q.N2_b2) * max(0.0, 0.05511475 - Q.dzerr_43) / 0.003066145   # +0.4%  N2_b2 < 0.2099 and dzerr_43 < 0.05511
        + 0.004137161 * max(0.0, 109.3251 - Q.mass_top30) * max(0.0, Q.lnptrel_12 - -5.283873) / 15.29061   # +0.4%  mass_top30 < 109.3 and lnptrel_12 > -5.284
        - 0.00413634 * max(0.0, Q.lep_z - 0.03021637) * max(0.0, 2.0 - Q.n_lepton) / 0.0540952   # -0.4%  lep_z > 0.03022 and n_lepton < 2
        + 0.003774963 * max(0.0, 303.75 - Q.sum_pt_top5) / 21.29329   # +0.4%  sum_pt_top5 < 303.8
        + 0.003704892 * max(0.0, 0.1991803 - Q.sj3_dr_min) / 0.05079997   # +0.4%  sj3_dr_min < 0.1992
        - 0.003569066 * max(0.0, Q.m012 - 11.86162) / 16.74842   # -0.4%  m012 > 11.86
        + 0.003500259 * max(0.0, 37.19471 - Q.sj3_pair_mass_min) * max(0.0, 0.3118983 - Q.sj3_z2) / 0.8244468   # +0.4%  sj3_pair_mass_min < 37.19 and sj3_z2 < 0.3119
        - 0.003484998 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, Q.z_charged_had - 0.3603262) / 0.2342076   # -0.3%  n_s3d_above_3 > 4 and z_charged_had > 0.3603
        - 0.003418992 * max(0.0, 1.398635 - Q.mass_2charged) / 0.2801595   # -0.3%  mass_2charged < 1.399
        + 0.003375681 * max(0.0, 5.763861 - Q.mass_2photon) / 2.954304   # +0.3%  mass_2photon < 5.764
        - 0.003229372 * max(0.0, 2.0 - Q.n_sdz_above_5) / 0.7314567   # -0.3%  n_sdz_above_5 < 2
        - 0.003189354 * max(0.0, 101.0 - Q.n_pairs_kt_above_1) / 8.83225   # -0.3%  n_pairs_kt_above_1 < 101
        + 0.003148377 * max(0.0, 65.0404 - Q.mass_charged) / 11.01768   # +0.3%  mass_charged < 65.04
        - 0.003144175 * max(0.0, Q.tau4 - 0.04996) / 0.004242161   # -0.3%  tau4 > 0.04996
        + 0.003092729 * max(0.0, 59.49644 - Q.sj3_pair_mass_min) / 25.61978   # +0.3%  sj3_pair_mass_min < 59.5
        + 0.002804179 * max(0.0, 37.19471 - Q.sj3_pair_mass_min) / 9.096396   # +0.3%  sj3_pair_mass_min < 37.19
        + 0.002657357 * max(0.0, Q.mass_top5 - 37.66803) / 12.83747   # +0.3%  mass_top5 > 37.67
        + 0.002637905 * max(0.0, 109.3251 - Q.mass_top30) / 15.25064   # +0.3%  mass_top30 < 109.3
        + 0.002627131 * max(0.0, 3.692778 - Q.lne_4) / 0.1626312   # +0.3%  lne_4 < 3.693
        - 0.00256055 * max(0.0, 2.852309e-06 - Q.ecf_g43) / 7.166198e-07   # -0.3%  ecf_g43 < 2.852e-06
        - 0.002522681 * max(0.0, Q.n_sd0_above_3 - 3.0) / 1.31993   # -0.3%  n_sd0_above_3 > 3
        + 0.002471799 * max(0.0, 0.8207647 - Q.sj3_pairmax_over_m) / 0.04486605   # +0.2%  sj3_pairmax_over_m < 0.8208
        + 0.002219098 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, 0.01013989 - Q.M3_b2) / 0.05136574   # +0.2%  mass_displaced3 < 39.1 and M3_b2 < 0.01014
        - 0.002124853 * max(0.0, 36.36236 - Q.sj3_mass1) * max(0.0, Q.sj3_mass2 - 9.068761) / 63.18552   # -0.2%  sj3_mass1 < 36.36 and sj3_mass2 > 9.069
        - 0.001547727 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, 0.8953628 - Q.tau54) / 0.07410258   # -0.2%  n_lund_kt_above_5 > 1 and tau54 < 0.8954
        - 0.001227199 * max(0.0, 0.07985021 - Q.sj4_dr_min) / 0.01211577   # -0.1%  sj4_dr_min < 0.07985
        + 0.001135677 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, 0.06029776 - Q.sj4_zsoft) / 0.01631242   # +0.1%  n_lund_kt_above_5 > 1 and sj4_zsoft < 0.0603
        + 0.001111893 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, Q.z_displaced5 - 0.1046203) / 0.03484583   # +0.1%  n_lund_kt_above_5 > 1 and z_displaced5 > 0.1046
        + 0.0010879 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # +0.1%  max_abs_d0 < 5.812
        - 0.001085225 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # -0.1%  lep_ptrel > 43.21
        - 0.001047041 * max(0.0, 0.0006496195 - Q.e3) / 0.000123029   # -0.1%  e3 < 0.0006496
        + 0.0009354337 * max(0.0, 0.0225905 - Q.D3_b2) / 0.006016147   # +0.1%  D3_b2 < 0.02259
        - 0.0008989551 * max(0.0, 4.326415 - Q.sj3_mass2) / 0.2966808   # -0.1%  sj3_mass2 < 4.326
        - 0.0007537 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.isphoton_12 - 0.0) / 9.753263   # -0.1%  mass_displaced3 < 39.1 and isphoton_12 > 0
        - 0.000609573 * max(0.0, 39.09615 - Q.mass_displaced3) * max(0.0, Q.sum_z_dr2_top20 - 0.0399789) / 0.1938807   # -0.1%  mass_displaced3 < 39.1 and sum_z_dr2_top20 > 0.03998
        + 0.0006010058 * max(0.0, 0.8207647 - Q.sj3_pairmax_over_m) * max(0.0, Q.isphoton_53 - 0.0) / 0.007175802   # +0.1%  sj3_pairmax_over_m < 0.8208 and isphoton_53 > 0
        - 0.0005987315 * max(0.0, 0.1117947 - Q.dr_0) / 0.02716844   # -0.1%  dr_0 < 0.1118
        + 0.0005349359 * max(0.0, Q.jet_abs_eta - 1.323111) / 0.03846033   # +0.1%  jet_abs_eta > 1.323
        - 0.0005343103 * max(0.0, 8.461136 - Q.sj3_mass1) / 0.4935496   # -0.1%  sj3_mass1 < 8.461
        - 0.0005185303 * max(0.0, 164.4374 - Q.mass) * max(0.0, Q.sj4_pair_mass_min - 13.57513) / 133.7659   # -0.1%  mass < 164.4 and sj4_pair_mass_min > 13.58
        - 0.0004220151 * max(0.0, 0.006220408 - Q.psi_0p1) / 0.0004886818   # -0.0%  psi_0p1 < 0.00622
        - 0.0003907388 * max(0.0, 0.530706 - Q.sj3_dr23) / 0.1618797   # -0.0%  sj3_dr23 < 0.5307
        + 0.0003374659 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, Q.n_electron - 0.0) / 0.40422   # +0.0%  n_lund_kt_above_5 > 1 and n_electron > 0
        + 0.0003125724 * max(0.0, -2.0 - Q.sum_charge) / 0.1953233   # +0.0%  sum_charge < -2
        + 0.0002225023 * max(0.0, 107.8099 - Q.sj4_pair_mass_max) * max(0.0, 515.5711 - Q.sum_pt_top20) / 513.304   # +0.0%  sj4_pair_mass_max < 107.8 and sum_pt_top20 < 515.6
        + 0.0002037146 * max(0.0, 107.8099 - Q.sj4_pair_mass_max) * max(0.0, Q.charge_43 - -1.0) / 29.80574   # +0.0%  sj4_pair_mass_max < 107.8 and charge_43 > -1
        + 0.0001984789 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # +0.0%  lep_iso < 0.4382
        - 0.0001936385 * max(0.0, 107.8099 - Q.sj4_pair_mass_max) * max(0.0, Q.td0_9 - -0.01537965) / 1.165896   # -0.0%  sj4_pair_mass_max < 107.8 and td0_9 > -0.01538
        - 0.0001416574 * max(0.0, Q.lam1 - 0.05242126) / 0.003271758   # -0.0%  lam1 > 0.05242
        - 0.0001130567 * max(0.0, Q.dr_0 - 0.2254414) / 0.007241025   # -0.0%  dr_0 > 0.2254
        - 0.0001086255 * max(0.0, Q.phi_26 - 0.1617432) / 0.02351405   # -0.0%  phi_26 > 0.1617
        - 9.651119e-05 * max(0.0, 0.05966366 - Q.M2_b2) * max(0.0, Q.d0err_52 - 0.0) / 7.007326e-05   # -0.0%  M2_b2 < 0.05966 and d0err_52 > 0
        - 3.488979e-05 * max(0.0, 107.8099 - Q.sj4_pair_mass_max) * max(0.0, 0.0 - Q.tdz_25) / 0.7807076   # -0.0%  sj4_pair_mass_max < 107.8 and tdz_25 < 0
        - 3.233249e-05 * max(0.0, 6.767937 - Q.mass_2charged) * max(0.0, 15.0 - Q.n_real_top15) / 0.1444146   # -0.0%  mass_2charged < 6.768 and n_real_top15 < 15
        - 3.16434e-05 * max(0.0, Q.n_s3d_above_3 - 4.0) * max(0.0, Q.tdz_44 - 0.01757784) / 0.02829223   # -0.0%  n_s3d_above_3 > 4 and tdz_44 > 0.01758
        + 2.536623e-05 * max(0.0, 28.89659 - Q.mass_2charged) * max(0.0, Q.phi_31 - 0.326416) / 0.1423609   # +0.0%  mass_2charged < 28.9 and phi_31 > 0.3264
        + 2.264224e-05 * max(0.0, Q.n_lund_kt_above_5 - 1.0) * max(0.0, Q.iselectron_34 - 0.0) / 0.00481   # +0.0%  n_lund_kt_above_5 > 1 and iselectron_34 > 0
    )
    return z


def neuron_116(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-3.362467e-07
    )
    return z


def neuron_117(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (2.856538e-06
    )
    return z


def neuron_118(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-8.231617e-06
    )
    return z


def neuron_119(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-6.299155e-07
    )
    return z


def neuron_120(Q):
    # scale S = 7.253; each line: share * term / its average size
    z = 7.253176 * (0.2637914
        - 0.1000082 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 10.0 - Q.n_s3d_above_3) / 15.42709   # -10.0%  lep_ptrel < 3.535 and n_s3d_above_3 < 10
        - 0.09166338 * max(0.0, 0.5187302 - Q.lep_z) / 0.4405854   # -9.2%  lep_z < 0.5187
        + 0.05848245 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 577.991 - Q.sip_3d_3) / 1181.582   # +5.8%  lep_ptrel < 3.535 and sip_3d_3 < 578
        - 0.05476368 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, 0.02781886 - Q.lep_iso) / 0.05591184   # -5.5%  lep_ptrel < 3.535 and lep_iso < 0.02782
        - 0.05260955 * max(0.0, 12.15228 - Q.lep_ptrel) / 8.819929   # -5.3%  lep_ptrel < 12.15
        - 0.05062056 * max(0.0, 3.53503 - Q.lep_ptrel) / 2.299439   # -5.1%  lep_ptrel < 3.535
        + 0.04384764 * max(0.0, 2.907433 - Q.lep_iso) / 2.168962   # +4.4%  lep_iso < 2.907
        - 0.04316758 * max(0.0, 114.0172 - Q.mass) / 13.82223   # -4.3%  mass < 114
        - 0.03080241 * max(0.0, 0.8860453 - Q.D3_b05) / 0.4670665   # -3.1%  D3_b05 < 0.886
        - 0.02563328 * max(0.0, 105.7234 - Q.mass) / 10.09077   # -2.6%  mass < 105.7
        - 0.02413884 * max(0.0, 577.991 - Q.sip_3d_3) / 516.7039   # -2.4%  sip_3d_3 < 578
        + 0.02325023 * max(0.0, 2.0 - Q.n_electron) / 1.6571   # +2.3%  n_electron < 2
        + 0.01934121 * max(0.0, 3.53503 - Q.lep_ptrel) * max(0.0, Q.n_photon - 5.0) / 28.0407   # +1.9%  lep_ptrel < 3.535 and n_photon > 5
        - 0.01912228 * max(0.0, Q.lund3_lndelta - -2.817283) / 1.325774   # -1.9%  lund3_lndelta > -2.817
        + 0.01870488 * max(0.0, 90.08945 - Q.mass) / 4.972097   # +1.9%  mass < 90.09
        + 0.01688434 * max(0.0, 0.02781886 - Q.lep_iso) / 0.01776032   # +1.7%  lep_iso < 0.02782
        + 0.01594962 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.mass_displaced3 - 0.0) / 3.662943   # +1.6%  lep_z < 0.5187 and mass_displaced3 > 0
        - 0.01562738 * max(0.0, Q.mass_top30 - 109.3251) / 10.36391   # -1.6%  mass_top30 > 109.3
        - 0.01503927 * max(0.0, 0.1719087 - Q.sj3_z3) / 0.06883262   # -1.5%  sj3_z3 < 0.1719
        + 0.01462741 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, 0.2601259 - Q.z_displaced3) / 3.087204   # +1.5%  n_pairs_kt_above_3 < 62 and z_displaced3 < 0.2601
        - 0.01383041 * max(0.0, Q.lep_ptrel - 18.7678) * max(0.0, 0.4381892 - Q.lep_iso) / 1.037293   # -1.4%  lep_ptrel > 18.77 and lep_iso < 0.4382
        + 0.01341576 * max(0.0, 25.0 - Q.n_charged) / 7.039403   # +1.3%  n_charged < 25
        + 0.01173172 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.mass_top10 - 48.96085) / 8.959233   # +1.2%  lep_z < 0.5187 and mass_top10 > 48.96
        - 0.01157845 * max(0.0, Q.mass_top10 - 48.96085) / 22.52231   # -1.2%  mass_top10 > 48.96
        - 0.01127037 * max(0.0, 0.4174214 - Q.N2_b05) / 0.02045623   # -1.1%  N2_b05 < 0.4174
        + 0.01012362 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 0.1889947 - Q.sj3_z3) / 0.7306739   # +1.0%  lep_ptrel < 12.15 and sj3_z3 < 0.189
        + 0.009973621 * max(0.0, Q.e4_b05 - 1.4104e-05) / 3.051148e-05   # +1.0%  e4_b05 > 1.41e-05
        - 0.008883036 * max(0.0, 4.0 - Q.n_sd0_above_2) / 1.197917   # -0.9%  n_sd0_above_2 < 4
        + 0.008840591 * max(0.0, Q.lep_ptrel - 27.3236) / 1.774579   # +0.9%  lep_ptrel > 27.32
        - 0.00821128 * max(0.0, 68.20711 - Q.mass_charged) / 12.83892   # -0.8%  mass_charged < 68.21
        - 0.008079257 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.n_lund - 5.0) / 78.27875   # -0.8%  n_pairs_kt_above_3 < 62 and n_lund > 5
        - 0.007638116 * max(0.0, 195.0 - Q.n_pairs_kt_above_1) / 37.35244   # -0.8%  n_pairs_kt_above_1 < 195
        + 0.007523376 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 0.7095008 - Q.planar_flow) / 2.189065   # +0.8%  lep_ptrel < 12.15 and planar_flow < 0.7095
        - 0.007212409 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) / 16.68515   # -0.7%  n_pairs_kt_above_3 < 62
        + 0.00716571 * max(0.0, 0.6800935 - Q.tau32) / 0.0806267   # +0.7%  tau32 < 0.6801
        - 0.006816655 * max(0.0, Q.mass_top15 - 104.4937) / 4.229149   # -0.7%  mass_top15 > 104.5
        - 0.006176222 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, Q.z_neutral_had - 0.02425618) / 1.870834   # -0.6%  n_pairs_kt_above_3 < 62 and z_neutral_had > 0.02426
        + 0.006036078 * max(0.0, 5.8125 - Q.max_abs_d0) * max(0.0, 1.0 - Q.ismuon_0) / 2.671766   # +0.6%  max_abs_d0 < 5.812 and ismuon_0 < 1
        + 0.005785535 * max(0.0, 5.0 - Q.n_s3d_above_10) / 2.774583   # +0.6%  n_s3d_above_10 < 5
        + 0.005495923 * max(0.0, 3.0 - Q.n_neutral_had) / 0.4082667   # +0.5%  n_neutral_had < 3
        + 0.005278132 * max(0.0, Q.pair_mean_lnkt - 0.5586581) / 0.1900469   # +0.5%  pair_mean_lnkt > 0.5587
        + 0.004988288 * max(0.0, Q.mass_top50 - 161.1264) / 2.814022   # +0.5%  mass_top50 > 161.1
        + 0.00419167 * max(0.0, 0.3052386 - Q.D3_b05) / 0.03985092   # +0.4%  D3_b05 < 0.3052
        - 0.004170035 * max(0.0, 0.1258758 - Q.M2_b05) / 0.007276394   # -0.4%  M2_b05 < 0.1259
        - 0.004045283 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, 0.5715866 - Q.dr_16) / 3.224727   # -0.4%  lep_ptrel < 12.15 and dr_16 < 0.5716
        + 0.003799901 * max(0.0, 1.938659 - Q.pt_entropy) / 0.03894921   # +0.4%  pt_entropy < 1.939
        + 0.003537321 * max(0.0, Q.lep_ptrel - 18.7678) / 2.829169   # +0.4%  lep_ptrel > 18.77
        + 0.003254531 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, 69.1565 - Q.mass_neutral) / 10.38924   # +0.3%  lep_z < 0.5187 and mass_neutral < 69.16
        - 0.003120742 * max(0.0, 0.3191471 - Q.z_charged_had) / 0.0155244   # -0.3%  z_charged_had < 0.3191
        - 0.002939912 * max(0.0, Q.lep_ptrel - 6.983043) / 5.140178   # -0.3%  lep_ptrel > 6.983
        + 0.002867137 * max(0.0, 5.8125 - Q.max_abs_d0) / 2.970994   # +0.3%  max_abs_d0 < 5.812
        - 0.002839573 * max(0.0, Q.lep_ptrel - 27.3236) * max(0.0, 0.1991254 - Q.mratio_min_012) / 0.2743283   # -0.3%  lep_ptrel > 27.32 and mratio_min_012 < 0.1991
        + 0.002703305 * max(0.0, 0.1719087 - Q.sj3_z3) * max(0.0, 24.55881 - Q.sj2_mass2) / 0.8177711   # +0.3%  sj3_z3 < 0.1719 and sj2_mass2 < 24.56
        - 0.002399207 * Q.z_displaced5 / 0.09296654   # -0.2%  z_displaced5
        - 0.002380067 * max(0.0, Q.M3_b05 - 0.08175231) / 0.002201961   # -0.2%  M3_b05 > 0.08175
        + 0.002364297 * max(0.0, Q.e4_b05 - 1.4104e-05) * max(0.0, Q.sum_pt - 550.4644) / 0.001444541   # +0.2%  e4_b05 > 1.41e-05 and sum_pt > 550.5
        - 0.002286603 * max(0.0, 62.0 - Q.n_pairs_kt_above_3) * max(0.0, 4.606241 - Q.sip_3d_3) / 27.36421   # -0.2%  n_pairs_kt_above_3 < 62 and sip_3d_3 < 4.606
        + 0.002215228 * max(0.0, 0.1719087 - Q.sj3_z3) * max(0.0, 1.0 - Q.isphoton_42) / 0.05631444   # +0.2%  sj3_z3 < 0.1719 and isphoton_42 < 1
        - 0.002101027 * max(0.0, Q.mass_top10 - 103.4976) / 1.86759   # -0.2%  mass_top10 > 103.5
        - 0.002045281 * max(0.0, 80.0 - Q.n_pairs_kt_above_1) / 5.177733   # -0.2%  n_pairs_kt_above_1 < 80
        - 0.001978758 * max(0.0, Q.mass_top50 - 135.5368) / 6.535045   # -0.2%  mass_top50 > 135.5
        + 0.001874197 * max(0.0, 226.3008 - Q.sip_3d_2) / 148.3364   # +0.2%  sip_3d_2 < 226.3
        - 0.001810571 * max(0.0, Q.pair_mean_lndelta - -1.523797) / 0.02402574   # -0.2%  pair_mean_lndelta > -1.524
        - 0.001585496 * max(0.0, Q.z_displaced3 - 0.2092108) / 0.02926899   # -0.2%  z_displaced3 > 0.2092
        - 0.001499174 * max(0.0, 5.0 - Q.n_dr_0p1_0p2) / 0.5919233   # -0.1%  n_dr_0p1_0p2 < 5
        + 0.001344406 * max(0.0, Q.mass_top10 - 48.96085) * max(0.0, Q.eta_7 - -0.1883545) / 4.333651   # +0.1%  mass_top10 > 48.96 and eta_7 > -0.1884
        + 0.001340234 * max(0.0, 0.5187302 - Q.lep_z) * max(0.0, Q.lep_dr - 0.114659) / 0.02162773   # +0.1%  lep_z < 0.5187 and lep_dr > 0.1147
        + 0.001307843 * max(0.0, 0.4174214 - Q.N2_b05) * max(0.0, 0.8930677 - Q.sj3_dr_max) / 0.008151529   # +0.1%  N2_b05 < 0.4174 and sj3_dr_max < 0.8931
        - 0.001237715 * max(0.0, 0.01394245 - Q.dr_min_012) / 0.002973681   # -0.1%  dr_min_012 < 0.01394
        - 0.001226998 * max(0.0, Q.sd_rg - 0.5562194) / 0.008836565   # -0.1%  sd_rg > 0.5562
        + 0.001122238 * max(0.0, Q.z_neutral - 0.6734757) / 0.003638212   # +0.1%  z_neutral > 0.6735
        - 0.001072222 * max(0.0, Q.mass_top30 - 162.7874) / 1.306268   # -0.1%  mass_top30 > 162.8
        - 0.0009019386 * max(0.0, Q.lund3_lndelta - -2.817283) * max(0.0, Q.sj3_dr13 - 0.577548) / 0.05077784   # -0.1%  lund3_lndelta > -2.817 and sj3_dr13 > 0.5775
        - 0.0007405597 * max(0.0, 0.08995834 - Q.e2_b05) / 0.0009622265   # -0.1%  e2_b05 < 0.08996
        + 0.0007257446 * max(0.0, Q.mass_top10 - 103.4976) * max(0.0, Q.log_sum_pt - 6.426552) / 0.1902225   # +0.1%  mass_top10 > 103.5 and log_sum_pt > 6.427
        + 0.0006995758 * Q.ecf_g42 / 2.611138e-05   # +0.1%  ecf_g42
        - 0.0006728059 * max(0.0, 105.7234 - Q.mass) * max(0.0, Q.eta_0 - 0.04705811) / 0.05007309   # -0.1%  mass < 105.7 and eta_0 > 0.04706
        - 0.0006550451 * max(0.0, 4.0 - Q.n_sd0_above_2) * max(0.0, 0.3967994 - Q.dr_19) / 0.2672384   # -0.1%  n_sd0_above_2 < 4 and dr_19 < 0.3968
        - 0.0006535462 * max(0.0, Q.lep_ptrel - 6.983043) * max(0.0, Q.jet_charge_k03 - 0.7410651) / 0.319818   # -0.1%  lep_ptrel > 6.983 and jet_charge_k03 > 0.7411
        - 0.0006403043 * max(0.0, 16.0 - Q.n_pt_above_5) / 0.9330767   # -0.1%  n_pt_above_5 < 16
        + 0.0004829679 * max(0.0, 1.824785 - Q.sj3_mass2) / 0.06171879   # +0.0%  sj3_mass2 < 1.825
        - 0.0004654125 * max(0.0, -3.360241 - Q.lnptrel_3) / 0.01767913   # -0.0%  lnptrel_3 < -3.36
        + 0.0004294077 * max(0.0, 4.0 - Q.n_dr_0p4_up) / 1.222967   # +0.0%  n_dr_0p4_up < 4
        - 0.0003106102 * max(0.0, 5.0 - Q.n_dr_0p1_0p2) * max(0.0, -0.217041 - Q.eta_2) / 0.005386908   # -0.0%  n_dr_0p1_0p2 < 5 and eta_2 < -0.217
        + 0.0002874894 * max(0.0, 12.15228 - Q.lep_ptrel) * max(0.0, -0.2314453 - Q.phi_22) / 0.1307046   # +0.0%  lep_ptrel < 12.15 and phi_22 < -0.2314
        + 0.0002557072 * max(0.0, 3.0 - Q.n_sd0_above_5) / 1.183617   # +0.0%  n_sd0_above_5 < 3
        + 0.0002226636 * max(0.0, 5.8125 - Q.max_abs_d0) * max(0.0, Q.d0err_12 - 0.01499939) / 0.01282045   # +0.0%  max_abs_d0 < 5.812 and d0err_12 > 0.015
        + 0.000213833 * max(0.0, Q.e3 - 0.00228569) / 0.000293828   # +0.0%  e3 > 0.002286
        - 0.0001694668 * max(0.0, 90.08945 - Q.mass) * max(0.0, Q.iselectron_1 - 0.0) / 0.1007217   # -0.0%  mass < 90.09 and iselectron_1 > 0
        - 0.0001555851 * max(0.0, 0.1132155 - Q.dr_4) / 0.02246667   # -0.0%  dr_4 < 0.1132
        - 9.287489e-05 * max(0.0, 0.01394245 - Q.dr_min_012) * max(0.0, Q.ischhad_26 - 0.0) / 0.0009175735   # -0.0%  dr_min_012 < 0.01394 and ischhad_26 > 0
        + 7.941119e-05 * max(0.0, Q.M3_b05 - 0.08175231) * max(0.0, Q.td0_13 - 0.01621867) / 5.874042e-05   # +0.0%  M3_b05 > 0.08175 and td0_13 > 0.01622
        - 7.440808e-05 * max(0.0, Q.pair_mean_lnkt - 0.5586581) * max(0.0, Q.lund3_lnz - -5.757953) / 0.4705559   # -0.0%  pair_mean_lnkt > 0.5587 and lund3_lnz > -5.758
        - 2.83647e-05 * max(0.0, 2.907433 - Q.lep_iso) * max(0.0, Q.iselectron_12 - 0.0) / 0.003127301   # -0.0%  lep_iso < 2.907 and iselectron_12 > 0
        - 2.298734e-05 * max(0.0, 0.1719087 - Q.sj3_z3) * max(0.0, Q.ismuon_11 - 0.0) / 0.0002559885   # -0.0%  sj3_z3 < 0.1719 and ismuon_11 > 0
        + 1.88546e-05 * max(0.0, Q.mass_top10 - 48.96085) * max(0.0, Q.iselectron_14 - 0.0) / 0.1223086   # +0.0%  mass_top10 > 48.96 and iselectron_14 > 0
        + 6.644792e-07 * max(0.0, Q.lep_ptrel - 43.20788) / 0.6406761   # +0.0%  lep_ptrel > 43.21
    )
    return z


def neuron_121(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-5.735425e-06
    )
    return z


def neuron_122(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-2.399869e-06
    )
    return z


def neuron_123(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (0.02055523
    )
    return z


def neuron_124(Q):
    # scale S = 10.31; each line: share * term / its average size
    z = 10.31131 * (0.005712303
        - 0.1174695 * max(0.0, Q.sd_mass - 78.4753) / 29.9564   # -11.7%  sd_mass > 78.48
        + 0.08582215 * max(0.0, Q.sd_mass - 88.81751) / 23.24968   # +8.6%  sd_mass > 88.82
        + 0.06383957 * max(0.0, Q.sd_mass - 94.55722) / 19.9621   # +6.4%  sd_mass > 94.56
        + 0.04279353 * max(0.0, 1.0 - Q.n_lepton) / 0.56532   # +4.3%  n_lepton < 1
        - 0.03812712 * max(0.0, Q.sd_mass - 127.8042) / 7.009137   # -3.8%  sd_mass > 127.8
        - 0.03765636 * max(0.0, Q.mass_top40 - 70.88236) / 41.46808   # -3.8%  mass_top40 > 70.88
        + 0.03602154 * max(0.0, 8.0 - Q.n_s3d_above_10) / 5.470973   # +3.6%  n_s3d_above_10 < 8
        - 0.03602084 * max(0.0, 1.362094 - Q.lep_iso) / 0.968642   # -3.6%  lep_iso < 1.362
        - 0.03366355 * max(0.0, 0.09942631 - Q.M3_b05) / 0.0421804   # -3.4%  M3_b05 < 0.09943
        + 0.03048582 * max(0.0, 43.20788 - Q.lep_ptrel) / 36.53314   # +3.0%  lep_ptrel < 43.21
        + 0.02890259 * max(0.0, Q.mass_top40 - 70.88236) * max(0.0, Q.lnerel_17 - -6.957861) / 100.3377   # +2.9%  mass_top40 > 70.88 and lnerel_17 > -6.958
        + 0.02756145 * max(0.0, 0.0002944259 - Q.ecf_g41) / 0.0001030636   # +2.8%  ecf_g41 < 0.0002944
        + 0.02742195 * max(0.0, 0.0729277 - Q.mass_over_sum_pt_sq) / 0.03640267   # +2.7%  mass_over_sum_pt_sq < 0.07293
        - 0.0249439 * max(0.0, 117.4867 - Q.mass) / 15.64803   # -2.5%  mass < 117.5
        + 0.02473737 * max(0.0, 577.991 - Q.sip_3d_3) / 516.7039   # +2.5%  sip_3d_3 < 578
        - 0.02206689 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, 0.2675458 - Q.z_neutral_had) / 0.7200105   # -2.2%  n_s3d_above_10 < 8 and z_neutral_had < 0.2675
        - 0.01635861 * max(0.0, 702.0 - Q.n_pairs_kt_above_1) / 397.4307   # -1.6%  n_pairs_kt_above_1 < 702
        - 0.01467028 * max(0.0, 0.4381892 - Q.lep_iso) / 0.2976879   # -1.5%  lep_iso < 0.4382
        + 0.01361663 * max(0.0, 95.14961 - Q.mass) / 6.371707   # +1.4%  mass < 95.15
        + 0.01353042 * max(0.0, Q.sd_mass - 154.5947) / 3.346187   # +1.4%  sd_mass > 154.6
        + 0.01341694 * max(0.0, 0.221436 - Q.lep_z) / 0.1710752   # +1.3%  lep_z < 0.2214
        - 0.01321972 * max(0.0, Q.sd_mass - 106.7501) / 13.87938   # -1.3%  sd_mass > 106.8
        + 0.01232471 * max(0.0, 2.0 - Q.n_s3d_above_3) / 0.4717633   # +1.2%  n_s3d_above_3 < 2
        - 0.0121122 * max(0.0, 0.06041764 - Q.lam1) / 0.02970598   # -1.2%  lam1 < 0.06042
        + 0.01190584 * max(0.0, 15.0 - Q.n_dr_0p4_up) / 9.48698   # +1.2%  n_dr_0p4_up < 15
        - 0.01171967 * max(0.0, 97.12186 - Q.mass_top40) / 7.43888   # -1.2%  mass_top40 < 97.12
        + 0.01116117 * max(0.0, 93.87663 - Q.sj3_pair_mass_max) / 12.55772   # +1.1%  sj3_pair_mass_max < 93.88
        - 0.01012461 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) / 129.0592   # -1.0%  n_pairs_kt_above_1 < 366
        + 0.009800325 * max(0.0, Q.n_s3d_above_3 - 3.0) / 1.66923   # +1.0%  n_s3d_above_3 > 3
        - 0.009040739 * max(0.0, 0.5076533 - Q.sj2_dr) / 0.1421645   # -0.9%  sj2_dr < 0.5077
        - 0.008764251 * max(0.0, 0.03624058 - Q.z_displaced3) / 0.01247855   # -0.9%  z_displaced3 < 0.03624
        - 0.008546869 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) * max(0.0, 15.17086 - Q.D2_b2) / 1437.013   # -0.9%  n_pairs_kt_above_1 < 366 and D2_b2 < 15.17
        - 0.00841158 * max(0.0, 0.08304558 - Q.z_displaced3) / 0.03742531   # -0.8%  z_displaced3 < 0.08305
        - 0.008328338 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 10.52344 - Q.max_abs_d0) / 6.106458   # -0.8%  n_s3d_above_3 > 3 and max_abs_d0 < 10.52
        - 0.007905755 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, 142.9952 - Q.sj3_pair_mass_max) / 1932.344   # -0.8%  lep_ptrel < 43.21 and sj3_pair_mass_max < 143
        - 0.00787701 * max(0.0, 4.41005 - Q.mass_displaced3) / 2.286034   # -0.8%  mass_displaced3 < 4.41
        - 0.007056711 * max(0.0, Q.mass_top40 - 70.88236) * max(0.0, 4.726476e-05 - Q.ecf_g42) / 0.0008572976   # -0.7%  mass_top40 > 70.88 and ecf_g42 < 4.726e-05
        + 0.006367037 * max(0.0, 0.3396572 - Q.lep_z) / 0.2746053   # +0.6%  lep_z < 0.3397
        - 0.006234249 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, 0.06762785 - Q.M2_b2) / 0.05887994   # -0.6%  n_s3d_above_3 > 3 and M2_b2 < 0.06763
        - 0.005339481 * max(0.0, 1.0 - Q.n_s3d_above_3) / 0.17383   # -0.5%  n_s3d_above_3 < 1
        - 0.004933971 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, Q.z_photon - 0.05998812) / 0.3053896   # -0.5%  n_s3d_above_3 > 3 and z_photon > 0.05999
        - 0.004706418 * max(0.0, Q.n_lund - 7.0) / 4.044803   # -0.5%  n_lund > 7
        + 0.004240639 * max(0.0, 15.0 - Q.n_dr_0p4_up) * max(0.0, Q.jet_charge_k03 - 0.1164034) / 2.156297   # +0.4%  n_dr_0p4_up < 15 and jet_charge_k03 > 0.1164
        - 0.004087573 * max(0.0, 73.24742 - Q.sj3_pair_mass_max) / 4.280067   # -0.4%  sj3_pair_mass_max < 73.25
        - 0.004026561 * max(0.0, 100.4835 - Q.mass) / 8.115061   # -0.4%  mass < 100.5
        - 0.003464396 * max(0.0, 1.777286 - Q.mass_displaced3) / 0.7377974   # -0.3%  mass_displaced3 < 1.777
        + 0.003349345 * max(0.0, 25.0 - Q.n_charged_had) / 7.529693   # +0.3%  n_charged_had < 25
        + 0.002803864 * max(0.0, Q.pair_max_lnm2 - 6.520267) / 0.4040215   # +0.3%  pair_max_lnm2 > 6.52
        + 0.002526645 * max(0.0, 15.0 - Q.n_charged_pt_above_1) / 1.32672   # +0.3%  n_charged_pt_above_1 < 15
        - 0.002495975 * max(0.0, Q.N3_b05 - 0.4404318) / 0.3441255   # -0.2%  N3_b05 > 0.4404
        + 0.002466489 * max(0.0, 0.007018285 - Q.sum_z_dr2_top3) / 0.001205932   # +0.2%  sum_z_dr2_top3 < 0.007018
        - 0.002447834 * max(0.0, 6.0 - Q.n_dr_0p1_0p2) / 0.8571967   # -0.2%  n_dr_0p1_0p2 < 6
        - 0.002419781 * max(0.0, 0.5076533 - Q.sj2_dr) * max(0.0, 5.114244 - Q.lne_2) / 0.1161855   # -0.2%  sj2_dr < 0.5077 and lne_2 < 5.114
        + 0.002150214 * max(0.0, 0.5076533 - Q.sj2_dr) * max(0.0, 0.1403354 - Q.z_muon) / 0.0170514   # +0.2%  sj2_dr < 0.5077 and z_muon < 0.1403
        - 0.002093323 * max(0.0, 58.0 - Q.n_pairs_kt_above_1) / 2.430393   # -0.2%  n_pairs_kt_above_1 < 58
        + 0.001852619 * max(0.0, 0.5076533 - Q.sj2_dr) * max(0.0, Q.lund1_lnz - -4.349248) / 0.1067436   # +0.2%  sj2_dr < 0.5077 and lund1_lnz > -4.349
        - 0.001822359 * max(0.0, 0.01365334 - Q.ecf_g31) / 0.007190153   # -0.2%  ecf_g31 < 0.01365
        + 0.001711689 * max(0.0, 0.2428609 - Q.z_dr_0p1_0p2) / 0.07286283   # +0.2%  z_dr_0p1_0p2 < 0.2429
        + 0.001571772 * max(0.0, Q.sd_mass - 42.31629) / 58.72614   # +0.2%  sd_mass > 42.32
        + 0.001557972 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, Q.n_lund_kt_above_1 - 3.0) / 12.43708   # +0.2%  n_s3d_above_10 < 8 and n_lund_kt_above_1 > 3
        + 0.001520967 * max(0.0, 1.777286 - Q.mass_displaced3) * max(0.0, 9.0 - Q.n_neutral_had) / 3.847284   # +0.2%  mass_displaced3 < 1.777 and n_neutral_had < 9
        - 0.001514028 * max(0.0, Q.mass_top40 - 70.88236) * max(0.0, 3.808744 - Q.lne_4) / 8.721589   # -0.2%  mass_top40 > 70.88 and lne_4 < 3.809
        - 0.001371231 * max(0.0, 100.4835 - Q.mass) * max(0.0, 30.0 - Q.n_real_top30) / 49.03597   # -0.1%  mass < 100.5 and n_real_top30 < 30
        - 0.001203203 * max(0.0, 1.824785 - Q.sj3_mass2) / 0.06171879   # -0.1%  sj3_mass2 < 1.825
        + 0.001147992 * max(0.0, 63.7508 - Q.sj3_pair_mass_max) / 2.405567   # +0.1%  sj3_pair_mass_max < 63.75
        + 0.001117164 * max(0.0, Q.lep_dr - 0.4191372) / 0.007569263   # +0.1%  lep_dr > 0.4191
        - 0.001066241 * max(0.0, 0.4008517 - Q.LHA) / 0.03854908   # -0.1%  LHA < 0.4009
        + 0.001048008 * max(0.0, 0.5076533 - Q.sj2_dr) * max(0.0, 0.4381892 - Q.lep_iso) / 0.04132887   # +0.1%  sj2_dr < 0.5077 and lep_iso < 0.4382
        - 0.0009261232 * max(0.0, 0.5076533 - Q.sj2_dr) * max(0.0, Q.iselectron_0 - 0.0) / 0.01382147   # -0.1%  sj2_dr < 0.5077 and iselectron_0 > 0
        - 0.0009182604 * Q.M3 / 0.03605934   # -0.1%  M3
        - 0.0008448744 * max(0.0, 0.006220408 - Q.psi_0p1) / 0.0004886818   # -0.1%  psi_0p1 < 0.00622
        - 0.0007634251 * max(0.0, Q.z_displaced5 - 0.2801368) / 0.0126536   # -0.1%  z_displaced5 > 0.2801
        + 0.0007581471 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, Q.sj3_z3 - 0.09421497) / 1.597426   # +0.1%  lep_ptrel < 43.21 and sj3_z3 > 0.09421
        + 0.0005850563 * max(0.0, Q.mass - 182.8592) / 1.688248   # +0.1%  mass > 182.9
        - 0.0005788135 * max(0.0, Q.tdz_2 - -0.02221619) / 0.04771882   # -0.1%  tdz_2 > -0.02222
        + 0.0004886769 * max(0.0, 0.3164143 - Q.tau32) * max(0.0, 0.02580261 - Q.d0err_14) / 5.010672e-05   # +0.0%  tau32 < 0.3164 and d0err_14 < 0.0258
        + 0.0004886249 * max(0.0, 45.8749 - Q.mass_top15) / 1.516652   # +0.0%  mass_top15 < 45.87
        + 0.0003912091 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, 0.019104 - Q.d0err_16) / 0.05526866   # +0.0%  n_s3d_above_10 < 8 and d0err_16 < 0.0191
        - 0.0003203667 * max(0.0, 43.20788 - Q.lep_ptrel) * max(0.0, -0.1462819 - Q.tdz_2) / 0.4585186   # -0.0%  lep_ptrel < 43.21 and tdz_2 < -0.1463
        - 0.000243684 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, Q.z_displaced5 - 0.1046203) / 0.01140989   # -0.0%  lep_z < 0.3397 and z_displaced5 > 0.1046
        + 0.0001610623 * max(0.0, -0.05792002 - Q.td0_17) / 0.02153752   # +0.0%  td0_17 < -0.05792
        + 0.0001163552 * max(0.0, Q.n_s3d_above_3 - 3.0) * max(0.0, Q.charge_42 - 0.0) / 0.1636467   # +0.0%  n_s3d_above_3 > 3 and charge_42 > 0
        - 0.0001069228 * max(0.0, 366.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.iselectron_1 - 0.0) / 4.67319   # -0.0%  n_pairs_kt_above_1 < 366 and iselectron_1 > 0
        + 8.810354e-05 * max(0.0, Q.pair_max_lnm2 - 6.520267) * max(0.0, -0.06205547 - Q.td0_14) / 0.009653046   # +0.0%  pair_max_lnm2 > 6.52 and td0_14 < -0.06206
        + 6.17436e-05 * max(0.0, 0.3164143 - Q.tau32) / 0.003288023   # +0.0%  tau32 < 0.3164
        - 3.658861e-05 * max(0.0, 8.0 - Q.n_s3d_above_10) * max(0.0, Q.eta_76 - 0.0) / 0.008868921   # -0.0%  n_s3d_above_10 < 8 and eta_76 > 0
        + 2.400771e-05 * max(0.0, 702.0 - Q.n_pairs_kt_above_1) * max(0.0, Q.ismuon_37 - 0.0) / 0.17736   # +0.0%  n_pairs_kt_above_1 < 702 and ismuon_37 > 0
        + 1.184346e-05 * max(0.0, 0.3396572 - Q.lep_z) * max(0.0, 0.3329206 - Q.tau43_b2) / 0.0009626607   # +0.0%  lep_z < 0.3397 and tau43_b2 < 0.3329
        - 6.030546e-07 * max(0.0, 117.4867 - Q.mass) * max(0.0, Q.ismuon_49 - 0.0) / 0.0009600611   # -0.0%  mass < 117.5 and ismuon_49 > 0
    )
    return z


def neuron_125(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.191802e-06
    )
    return z


def neuron_126(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (7.633821e-06
    )
    return z


def neuron_127(Q):
    # scale S = 1; each line: share * term / its average size
    z = 1.0 * (-1.922015e-07
    )
    return z


def class_token(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q), neuron_16(Q), neuron_17(Q), neuron_18(Q), neuron_19(Q), neuron_20(Q), neuron_21(Q), neuron_22(Q), neuron_23(Q), neuron_24(Q), neuron_25(Q), neuron_26(Q), neuron_27(Q), neuron_28(Q), neuron_29(Q), neuron_30(Q), neuron_31(Q), neuron_32(Q), neuron_33(Q), neuron_34(Q), neuron_35(Q), neuron_36(Q), neuron_37(Q), neuron_38(Q), neuron_39(Q), neuron_40(Q), neuron_41(Q), neuron_42(Q), neuron_43(Q), neuron_44(Q), neuron_45(Q), neuron_46(Q), neuron_47(Q), neuron_48(Q), neuron_49(Q), neuron_50(Q), neuron_51(Q), neuron_52(Q), neuron_53(Q), neuron_54(Q), neuron_55(Q), neuron_56(Q), neuron_57(Q), neuron_58(Q), neuron_59(Q), neuron_60(Q), neuron_61(Q), neuron_62(Q), neuron_63(Q), neuron_64(Q), neuron_65(Q), neuron_66(Q), neuron_67(Q), neuron_68(Q), neuron_69(Q), neuron_70(Q), neuron_71(Q), neuron_72(Q), neuron_73(Q), neuron_74(Q), neuron_75(Q), neuron_76(Q), neuron_77(Q), neuron_78(Q), neuron_79(Q), neuron_80(Q), neuron_81(Q), neuron_82(Q), neuron_83(Q), neuron_84(Q), neuron_85(Q), neuron_86(Q), neuron_87(Q), neuron_88(Q), neuron_89(Q), neuron_90(Q), neuron_91(Q), neuron_92(Q), neuron_93(Q), neuron_94(Q), neuron_95(Q), neuron_96(Q), neuron_97(Q), neuron_98(Q), neuron_99(Q), neuron_100(Q), neuron_101(Q), neuron_102(Q), neuron_103(Q), neuron_104(Q), neuron_105(Q), neuron_106(Q), neuron_107(Q), neuron_108(Q), neuron_109(Q), neuron_110(Q), neuron_111(Q), neuron_112(Q), neuron_113(Q), neuron_114(Q), neuron_115(Q), neuron_116(Q), neuron_117(Q), neuron_118(Q), neuron_119(Q), neuron_120(Q), neuron_121(Q), neuron_122(Q), neuron_123(Q), neuron_124(Q), neuron_125(Q), neuron_126(Q), neuron_127(Q)]


H_AVG = [0.00519466120749712, 5.328471615939634e-06, 5.4428569455922116e-06, 8.399041689699516e-06, 2.789398649838404e-06, 2.1557473246502923e-06, 1.7697291696094908e-05, 6.433383077819599e-06, 2.9263555916259065e-06, 2.4402231701969868e-06, 1.5703770259278826e-05, 1.8882376025430858e-05, 5.094681000628043e-06, 4.42791542809573e-06, 2.850953478628071e-06, 6.461600605689455e-06, 1.2632913684553615, 1.0353830475651193e-05, 1.0069860054960655, 1.4034981177246664e-06, 8.213594924200152e-07, 0.0016828823136165738, 1.0469406106494716e-06, 7.014223797341401e-07, 0.9669522649820999, 0.031034015119075775, 3.2284713142871624e-06, 6.307112016656902e-06, 9.621167009754572e-06, 9.18245677894447e-06, 8.599329703429248e-06, 3.50712639374251e-06, 1.4409097275347449e-05, 2.5693570933071896e-06, 1.3888894500269089e-05, 6.7494802351575345e-06, 4.829922545468435e-06, 4.353409622126492e-06, 9.41376401897287e-06, 8.275775144284125e-06, 7.812739681867242e-07, 4.064823855287614e-08, 8.77758429851383e-06, 9.412546205567196e-06, 3.3285361951129744e-06, 6.852587830508128e-06, 1.3570706869359128e-07, 2.525432591937715e-07, 8.789036655798554e-06, 2.241615675302455e-06, 5.174249963602051e-06, 2.0075860902579734e-06, 0.7138985967257327, 1.8889745660999324e-06, 1.1907937732758e-05, 2.0674791812780313e-05, 7.043094001346617e-07, 1.9648525722004706e-06, 3.6160620311420644e-06, 6.0308780120976735e-06, 4.668709607358323e-06, 5.8360674302093685e-06, 9.460824799134571e-07, 8.31242050480796e-06, 1.2524461681096e-06, 5.790296199847944e-06, 2.711630031626555e-06, 5.214326392888324e-06, 0.936354245550556, 6.928535185579676e-06, 0.01607455126941204, 3.0308362966025015e-06, 1.1646856137303985e-06, 1.699241329333745e-05, 4.496920155361295e-06, 1.9718322619155515e-06, 5.543500265048351e-06, 3.0846304071019404e-06, 0.5427997386537919, 3.602822471293621e-05, 9.706954188004602e-06, 0.4626189144783929, 2.943194658655557e-06, 0.8274226559920113, 7.426965748891234e-05, 8.161995879163442e-07, 1.7102513083955273e-05, 1.6158464859472588e-06, 3.0317224286591227e-07, 1.739162485137058e-06, 0.000940794765483588, 3.9925271266838536e-06, 6.723401838826248e-06, 8.443295200777357e-07, 2.3221018636832014e-06, 4.400555440042808e-07, 4.325970337504259e-07, 0.9839994808256386, 4.96436996400007e-07, 0.0019342465093359351, 6.26793917035684e-06, 8.878885637386702e-06, 5.486107966135023e-06, 3.4166259865742177e-06, 0.9701215452206622, 2.73228288278915e-05, 2.161275733669754e-05, 1.1581857506826054e-05, 4.469852683541831e-06, 5.224126766734116e-07, 3.5770240174315404e-06, 3.6678904962172965e-06, 5.232592684478732e-06, 2.6656874979380518e-05, 4.677441211242694e-06, 0.9192100034900258, 3.362466713952017e-07, 2.856538458217983e-06, 8.231617357523646e-06, 6.299155188571604e-07, 0.9912917619435098, 5.7354254749952815e-06, 2.3998686629056465e-06, 0.020555227994918823, 0.6185101030264014, 1.191802311950596e-06, 7.633821041963529e-06, 1.922015400168675e-07]
T = [4.080455638317645, 5.822974457159112, 4.038160866271293, 4.521051476037063, 4.917518725797075, 8.411681822244208, 3.792389373785767, 4.287668868548693, 7.065935013557428, 10.184663187592268]


def logits(h):
    # rounded to a multiple of 2^-20 (the formula's class scores are exact binary fractions): removes floating-point noise
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        0.4043572 + T[0] * (   # class QCD
            - 0.1728519 * h[18] / H_AVG[18]
            + 0.1438262 * h[97] / H_AVG[97]
            - 0.1133305 * h[68] / H_AVG[68]
            - 0.09595246 * h[120] / H_AVG[120]
            + 0.08995783 * h[52] / H_AVG[52]
            - 0.07830821 * h[24] / H_AVG[24]
            + 0.07783373 * h[104] / H_AVG[104]
            + 0.07329739 * h[81] / H_AVG[81]
            + 0.03537221 * h[78] / H_AVG[78]
            + 0.03152267 * h[16] / H_AVG[16]
            - 0.03045544 * h[83] / H_AVG[83]
            + 0.02869471 * h[115] / H_AVG[115]
            - 0.02763888 * h[124] / H_AVG[124]
            + 0.0003517946 * h[123] / H_AVG[123]
            + 0.0002162196 * h[25] / H_AVG[25]
            + 0.0001921635 * h[70] / H_AVG[70]
            + 6.477507e-05 * h[0] / H_AVG[0]
            - 5.260806e-05 * h[90] / H_AVG[90]
            - 4.188729e-05 * h[99] / H_AVG[99]
            - 3.229755e-05 * h[21] / H_AVG[21]
            + 1.978583e-06 * h[84] / H_AVG[84]
            + 4.669443e-07 * h[55] / H_AVG[55]
            + 3.133475e-07 * h[79] / H_AVG[79]
            - 2.990929e-07 * h[105] / H_AVG[105]
            - 2.427615e-07 * h[106] / H_AVG[106]
            + 1.9728e-07 * h[113] / H_AVG[113]
            - 1.874209e-07 * h[34] / H_AVG[34]
            + 1.682034e-07 * h[73] / H_AVG[73]
            + 1.562759e-07 * h[10] / H_AVG[10]
            + 1.489386e-07 * h[11] / H_AVG[11]
            - 1.267874e-07 * h[6] / H_AVG[6]
            + 1.210389e-07 * h[86] / H_AVG[86]
            - 1.150407e-07 * h[32] / H_AVG[32]
            + 1.05772e-07 * h[17] / H_AVG[17]
            + 9.561055e-08 * h[38] / H_AVG[38]
            - 7.848129e-08 * h[42] / H_AVG[42]
            - 6.996253e-08 * h[54] / H_AVG[54]
            - 6.647925e-08 * h[29] / H_AVG[29]
            + 6.026952e-08 * h[118] / H_AVG[118]
            + 4.704083e-08 * h[48] / H_AVG[48]
            - 4.49284e-08 * h[45] / H_AVG[45]
            - 4.337739e-08 * h[39] / H_AVG[39]
            + 3.99767e-08 * h[27] / H_AVG[27]
            + 3.983934e-08 * h[63] / H_AVG[63]
            + 3.937874e-08 * h[30] / H_AVG[30]
            + 3.85983e-08 * h[43] / H_AVG[43]
            + 3.591641e-08 * h[121] / H_AVG[121]
            - 3.524606e-08 * h[80] / H_AVG[80]
            - 3.226786e-08 * h[126] / H_AVG[126]
            + 3.115782e-08 * h[60] / H_AVG[60]
            - 3.065136e-08 * h[61] / H_AVG[61]
            - 2.910389e-08 * h[102] / H_AVG[102]
            - 2.907419e-08 * h[107] / H_AVG[107]
            - 2.864284e-08 * h[7] / H_AVG[7]
            + 2.578666e-08 * h[92] / H_AVG[92]
            + 2.524637e-08 * h[35] / H_AVG[35]
            + 2.453723e-08 * h[15] / H_AVG[15]
            - 2.3695e-08 * h[59] / H_AVG[59]
            - 2.285452e-08 * h[101] / H_AVG[101]
            - 2.185652e-08 * h[28] / H_AVG[28]
            + 2.120002e-08 * h[103] / H_AVG[103]
            + 1.994722e-08 * h[69] / H_AVG[69]
            + 1.933334e-08 * h[67] / H_AVG[67]
            + 1.875682e-08 * h[1] / H_AVG[1]
            + 1.861823e-08 * h[100] / H_AVG[100]
            - 1.707677e-08 * h[110] / H_AVG[110]
            - 1.653385e-08 * h[3] / H_AVG[3]
            + 1.586898e-08 * h[37] / H_AVG[37]
            - 1.555712e-08 * h[50] / H_AVG[50]
            + 1.464429e-08 * h[44] / H_AVG[44]
            + 1.460814e-08 * h[36] / H_AVG[36]
            - 1.435277e-08 * h[12] / H_AVG[12]
            - 1.366633e-08 * h[114] / H_AVG[114]
            - 1.353806e-08 * h[2] / H_AVG[2]
            + 1.305433e-08 * h[58] / H_AVG[58]
            - 1.299058e-08 * h[112] / H_AVG[112]
            + 1.272903e-08 * h[76] / H_AVG[76]
            - 1.228282e-08 * h[13] / H_AVG[13]
            - 1.187813e-08 * h[19] / H_AVG[19]
            - 1.16892e-08 * h[111] / H_AVG[111]
            + 1.115459e-08 * h[74] / H_AVG[74]
            + 1.05745e-08 * h[108] / H_AVG[108]
            + 9.936754e-09 * h[5] / H_AVG[5]
            - 9.502861e-09 * h[8] / H_AVG[8]
            - 9.112305e-09 * h[91] / H_AVG[91]
            - 8.486995e-09 * h[4] / H_AVG[4]
            - 8.122151e-09 * h[117] / H_AVG[117]
            + 6.875393e-09 * h[53] / H_AVG[53]
            - 6.8052e-09 * h[66] / H_AVG[66]
            - 6.194315e-09 * h[33] / H_AVG[33]
            - 6.01269e-09 * h[65] / H_AVG[65]
            + 5.562944e-09 * h[82] / H_AVG[82]
            - 5.222125e-09 * h[31] / H_AVG[31]
            - 4.803855e-09 * h[71] / H_AVG[71]
            + 4.460061e-09 * h[94] / H_AVG[94]
            + 4.301646e-09 * h[9] / H_AVG[9]
            + 3.78318e-09 * h[26] / H_AVG[26]
            - 3.400028e-09 * h[14] / H_AVG[14]
            + 3.370336e-09 * h[57] / H_AVG[57]
            + 3.173881e-09 * h[49] / H_AVG[49]
            - 3.103464e-09 * h[22] / H_AVG[22]
            + 3.100715e-09 * h[77] / H_AVG[77]
            + 2.878112e-09 * h[122] / H_AVG[122]
            + 2.323961e-09 * h[72] / H_AVG[72]
            + 2.219635e-09 * h[51] / H_AVG[51]
            - 2.201312e-09 * h[23] / H_AVG[23]
            - 1.515849e-09 * h[89] / H_AVG[89]
            + 1.479155e-09 * h[125] / H_AVG[125]
            - 1.386041e-09 * h[75] / H_AVG[75]
            - 1.198185e-09 * h[87] / H_AVG[87]
            + 1.193673e-09 * h[93] / H_AVG[93]
            - 1.081325e-09 * h[40] / H_AVG[40]
            - 9.951237e-10 * h[20] / H_AVG[20]
            + 8.881801e-10 * h[119] / H_AVG[119]
            - 8.20616e-10 * h[62] / H_AVG[62]
            - 7.401448e-10 * h[95] / H_AVG[95]
            - 6.88742e-10 * h[96] / H_AVG[96]
            + 3.971708e-10 * h[116] / H_AVG[116]
            + 3.818574e-10 * h[56] / H_AVG[56]
            + 2.658157e-10 * h[85] / H_AVG[85]
            + 1.594118e-10 * h[109] / H_AVG[109]
            - 1.374408e-10 * h[98] / H_AVG[98]
            - 1.116285e-10 * h[88] / H_AVG[88]
            - 7.374669e-11 * h[46] / H_AVG[46]
            - 7.115671e-11 * h[47] / H_AVG[47]
            - 4.308891e-11 * h[64] / H_AVG[64]
            + 1.648884e-11 * h[127] / H_AVG[127]
            + 1.588016e-12 * h[41] / H_AVG[41]
        ),
        -0.4064285 + T[1] * (   # class Hbb
            + 0.2895113 * h[16] / H_AVG[16]
            + 0.2014629 * h[120] / H_AVG[120]
            - 0.09831222 * h[68] / H_AVG[68]
            + 0.0751585 * h[18] / H_AVG[18]
            - 0.06973006 * h[78] / H_AVG[78]
            - 0.06866375 * h[24] / H_AVG[24]
            - 0.0597021 * h[97] / H_AVG[97]
            - 0.05766006 * h[52] / H_AVG[52]
            + 0.03818212 * h[115] / H_AVG[115]
            + 0.02498699 * h[81] / H_AVG[81]
            + 0.007092874 * h[83] / H_AVG[83]
            - 0.004430428 * h[104] / H_AVG[104]
            + 0.002662502 * h[124] / H_AVG[124]
            - 0.001544757 * h[25] / H_AVG[25]
            - 0.0006792197 * h[70] / H_AVG[70]
            + 0.0001030494 * h[0] / H_AVG[0]
            + 3.121372e-05 * h[123] / H_AVG[123]
            - 2.973247e-05 * h[99] / H_AVG[99]
            - 2.681028e-05 * h[90] / H_AVG[90]
            - 2.515779e-05 * h[21] / H_AVG[21]
            + 1.385977e-06 * h[84] / H_AVG[84]
            + 3.271655e-07 * h[55] / H_AVG[55]
            + 2.19479e-07 * h[79] / H_AVG[79]
            - 2.095656e-07 * h[105] / H_AVG[105]
            - 1.70073e-07 * h[106] / H_AVG[106]
            + 1.381676e-07 * h[113] / H_AVG[113]
            - 1.313126e-07 * h[34] / H_AVG[34]
            + 1.178235e-07 * h[73] / H_AVG[73]
            + 1.095069e-07 * h[10] / H_AVG[10]
            + 1.043964e-07 * h[11] / H_AVG[11]
            - 8.885506e-08 * h[6] / H_AVG[6]
            + 8.478946e-08 * h[86] / H_AVG[86]
            - 8.057979e-08 * h[32] / H_AVG[32]
            + 7.409798e-08 * h[17] / H_AVG[17]
            + 6.700908e-08 * h[38] / H_AVG[38]
            - 5.499274e-08 * h[42] / H_AVG[42]
            - 4.901567e-08 * h[54] / H_AVG[54]
            - 4.6573e-08 * h[29] / H_AVG[29]
            + 4.222533e-08 * h[118] / H_AVG[118]
            + 3.295855e-08 * h[48] / H_AVG[48]
            - 3.147199e-08 * h[45] / H_AVG[45]
            - 3.039519e-08 * h[39] / H_AVG[39]
            + 2.801509e-08 * h[27] / H_AVG[27]
            + 2.791351e-08 * h[63] / H_AVG[63]
            + 2.758877e-08 * h[30] / H_AVG[30]
            + 2.704584e-08 * h[43] / H_AVG[43]
            + 2.516798e-08 * h[121] / H_AVG[121]
            - 2.469055e-08 * h[80] / H_AVG[80]
            - 2.260632e-08 * h[126] / H_AVG[126]
            + 2.183453e-08 * h[60] / H_AVG[60]
            - 2.147748e-08 * h[61] / H_AVG[61]
            - 2.039241e-08 * h[102] / H_AVG[102]
            - 2.035555e-08 * h[107] / H_AVG[107]
            - 2.007058e-08 * h[7] / H_AVG[7]
            + 1.806864e-08 * h[92] / H_AVG[92]
            + 1.768939e-08 * h[35] / H_AVG[35]
            + 1.719055e-08 * h[15] / H_AVG[15]
            - 1.660384e-08 * h[59] / H_AVG[59]
            - 1.601523e-08 * h[101] / H_AVG[101]
            - 1.531036e-08 * h[28] / H_AVG[28]
            + 1.485297e-08 * h[103] / H_AVG[103]
            + 1.39735e-08 * h[69] / H_AVG[69]
            + 1.354398e-08 * h[67] / H_AVG[67]
            + 1.314331e-08 * h[1] / H_AVG[1]
            + 1.304785e-08 * h[100] / H_AVG[100]
            - 1.196777e-08 * h[110] / H_AVG[110]
            - 1.158609e-08 * h[3] / H_AVG[3]
            + 1.111735e-08 * h[37] / H_AVG[37]
            - 1.089222e-08 * h[50] / H_AVG[50]
            + 1.026245e-08 * h[44] / H_AVG[44]
            + 1.023626e-08 * h[36] / H_AVG[36]
            - 1.005356e-08 * h[12] / H_AVG[12]
            - 9.571233e-09 * h[114] / H_AVG[114]
            - 9.479374e-09 * h[2] / H_AVG[2]
            + 9.14739e-09 * h[58] / H_AVG[58]
            - 9.101201e-09 * h[112] / H_AVG[112]
            + 8.918152e-09 * h[76] / H_AVG[76]
            - 8.60665e-09 * h[13] / H_AVG[13]
            - 8.330651e-09 * h[19] / H_AVG[19]
            - 8.190342e-09 * h[111] / H_AVG[111]
            + 7.815659e-09 * h[74] / H_AVG[74]
            + 7.412356e-09 * h[108] / H_AVG[108]
            + 6.962865e-09 * h[5] / H_AVG[5]
            - 6.658331e-09 * h[8] / H_AVG[8]
            - 6.384759e-09 * h[91] / H_AVG[91]
            - 5.946917e-09 * h[4] / H_AVG[4]
            - 5.690864e-09 * h[117] / H_AVG[117]
            + 4.814665e-09 * h[53] / H_AVG[53]
            - 4.764759e-09 * h[66] / H_AVG[66]
            - 4.341132e-09 * h[33] / H_AVG[33]
            - 4.212183e-09 * h[65] / H_AVG[65]
            + 3.897373e-09 * h[82] / H_AVG[82]
            - 3.660065e-09 * h[31] / H_AVG[31]
            - 3.364402e-09 * h[71] / H_AVG[71]
            + 3.123926e-09 * h[94] / H_AVG[94]
            + 3.013623e-09 * h[9] / H_AVG[9]
            + 2.648951e-09 * h[26] / H_AVG[26]
            - 2.382029e-09 * h[14] / H_AVG[14]
            + 2.361135e-09 * h[57] / H_AVG[57]
            + 2.22398e-09 * h[49] / H_AVG[49]
            - 2.17488e-09 * h[22] / H_AVG[22]
            + 2.173568e-09 * h[77] / H_AVG[77]
            + 2.016256e-09 * h[122] / H_AVG[122]
            + 1.628424e-09 * h[72] / H_AVG[72]
            + 1.555106e-09 * h[51] / H_AVG[51]
            - 1.542252e-09 * h[23] / H_AVG[23]
            - 1.062345e-09 * h[89] / H_AVG[89]
            + 1.036878e-09 * h[125] / H_AVG[125]
            - 9.7089e-10 * h[75] / H_AVG[75]
            - 8.391565e-10 * h[87] / H_AVG[87]
            + 8.359224e-10 * h[93] / H_AVG[93]
            - 7.576049e-10 * h[40] / H_AVG[40]
            - 6.974344e-10 * h[20] / H_AVG[20]
            + 6.216388e-10 * h[119] / H_AVG[119]
            - 5.750555e-10 * h[62] / H_AVG[62]
            - 5.185388e-10 * h[95] / H_AVG[95]
            - 4.824575e-10 * h[96] / H_AVG[96]
            + 2.783471e-10 * h[116] / H_AVG[116]
            + 2.674605e-10 * h[56] / H_AVG[56]
            + 1.861357e-10 * h[85] / H_AVG[85]
            + 1.117192e-10 * h[109] / H_AVG[109]
            - 9.631699e-11 * h[98] / H_AVG[98]
            - 7.826208e-11 * h[88] / H_AVG[88]
            - 5.164429e-11 * h[46] / H_AVG[46]
            - 4.987327e-11 * h[47] / H_AVG[47]
            - 2.658945e-11 * h[64] / H_AVG[64]
            + 1.152004e-11 * h[127] / H_AVG[127]
            + 1.110653e-12 * h[41] / H_AVG[41]
        ),
        0.1415668 + T[2] * (   # class Hcc
            - 0.1952257 * h[83] / H_AVG[83]
            + 0.1875706 * h[18] / H_AVG[18]
            - 0.1433981 * h[24] / H_AVG[24]
            - 0.08770906 * h[81] / H_AVG[81]
            - 0.06122771 * h[16] / H_AVG[16]
            + 0.06071307 * h[124] / H_AVG[124]
            + 0.05318188 * h[104] / H_AVG[104]
            + 0.05313858 * h[78] / H_AVG[78]
            - 0.04993267 * h[97] / H_AVG[97]
            + 0.03489068 * h[120] / H_AVG[120]
            - 0.02609392 * h[68] / H_AVG[68]
            - 0.02350373 * h[52] / H_AVG[52]
            + 0.02020085 * h[115] / H_AVG[115]
            - 0.001477323 * h[25] / H_AVG[25]
            + 0.0009192659 * h[70] / H_AVG[70]
            - 0.0005250671 * h[123] / H_AVG[123]
            + 0.0002135483 * h[0] / H_AVG[0]
            - 4.254243e-05 * h[99] / H_AVG[99]
            - 2.827986e-05 * h[21] / H_AVG[21]
            + 1.999193e-06 * h[84] / H_AVG[84]
            + 1.187715e-06 * h[90] / H_AVG[90]
            + 4.720142e-07 * h[55] / H_AVG[55]
            + 3.166725e-07 * h[79] / H_AVG[79]
            - 3.022094e-07 * h[105] / H_AVG[105]
            - 2.453631e-07 * h[106] / H_AVG[106]
            + 1.993564e-07 * h[113] / H_AVG[113]
            - 1.89364e-07 * h[34] / H_AVG[34]
            + 1.699606e-07 * h[73] / H_AVG[73]
            + 1.579699e-07 * h[10] / H_AVG[10]
            + 1.50544e-07 * h[11] / H_AVG[11]
            - 1.281269e-07 * h[6] / H_AVG[6]
            + 1.222877e-07 * h[86] / H_AVG[86]
            - 1.1625e-07 * h[32] / H_AVG[32]
            + 1.069227e-07 * h[17] / H_AVG[17]
            + 9.659955e-08 * h[38] / H_AVG[38]
            - 7.930314e-08 * h[42] / H_AVG[42]
            - 7.070869e-08 * h[54] / H_AVG[54]
            - 6.719279e-08 * h[29] / H_AVG[29]
            + 6.091039e-08 * h[118] / H_AVG[118]
            + 4.753623e-08 * h[48] / H_AVG[48]
            - 4.539675e-08 * h[45] / H_AVG[45]
            - 4.384106e-08 * h[39] / H_AVG[39]
            + 4.041039e-08 * h[27] / H_AVG[27]
            + 4.026887e-08 * h[63] / H_AVG[63]
            + 3.978879e-08 * h[30] / H_AVG[30]
            + 3.89968e-08 * h[43] / H_AVG[43]
            + 3.6312e-08 * h[121] / H_AVG[121]
            - 3.56223e-08 * h[80] / H_AVG[80]
            - 3.260269e-08 * h[126] / H_AVG[126]
            + 3.149333e-08 * h[60] / H_AVG[60]
            - 3.097542e-08 * h[61] / H_AVG[61]
            - 2.940911e-08 * h[102] / H_AVG[102]
            - 2.934985e-08 * h[107] / H_AVG[107]
            - 2.896295e-08 * h[7] / H_AVG[7]
            + 2.605848e-08 * h[92] / H_AVG[92]
            + 2.551577e-08 * h[35] / H_AVG[35]
            + 2.479754e-08 * h[15] / H_AVG[15]
            - 2.394716e-08 * h[59] / H_AVG[59]
            - 2.308848e-08 * h[101] / H_AVG[101]
            - 2.208768e-08 * h[28] / H_AVG[28]
            + 2.142618e-08 * h[103] / H_AVG[103]
            + 2.015796e-08 * h[69] / H_AVG[69]
            + 1.953962e-08 * h[67] / H_AVG[67]
            + 1.896104e-08 * h[1] / H_AVG[1]
            + 1.881964e-08 * h[100] / H_AVG[100]
            - 1.726216e-08 * h[110] / H_AVG[110]
            - 1.671206e-08 * h[3] / H_AVG[3]
            + 1.603476e-08 * h[37] / H_AVG[37]
            - 1.571665e-08 * h[50] / H_AVG[50]
            + 1.480188e-08 * h[44] / H_AVG[44]
            + 1.476696e-08 * h[36] / H_AVG[36]
            - 1.450045e-08 * h[12] / H_AVG[12]
            - 1.380742e-08 * h[114] / H_AVG[114]
            - 1.36807e-08 * h[2] / H_AVG[2]
            + 1.319695e-08 * h[58] / H_AVG[58]
            - 1.313253e-08 * h[112] / H_AVG[112]
            + 1.286561e-08 * h[76] / H_AVG[76]
            - 1.241371e-08 * h[13] / H_AVG[13]
            - 1.201069e-08 * h[19] / H_AVG[19]
            - 1.181072e-08 * h[111] / H_AVG[111]
            + 1.127286e-08 * h[74] / H_AVG[74]
            + 1.068846e-08 * h[108] / H_AVG[108]
            + 1.004772e-08 * h[5] / H_AVG[5]
            - 9.606365e-09 * h[8] / H_AVG[8]
            - 9.212715e-09 * h[91] / H_AVG[91]
            - 8.577189e-09 * h[4] / H_AVG[4]
            - 8.207937e-09 * h[117] / H_AVG[117]
            + 6.949673e-09 * h[53] / H_AVG[53]
            - 6.877906e-09 * h[66] / H_AVG[66]
            - 6.26009e-09 * h[33] / H_AVG[33]
            - 6.078035e-09 * h[65] / H_AVG[65]
            + 5.622917e-09 * h[82] / H_AVG[82]
            - 5.27497e-09 * h[31] / H_AVG[31]
            - 4.852579e-09 * h[71] / H_AVG[71]
            + 4.507815e-09 * h[94] / H_AVG[94]
            + 4.346535e-09 * h[9] / H_AVG[9]
            + 3.81874e-09 * h[26] / H_AVG[26]
            - 3.437672e-09 * h[14] / H_AVG[14]
            + 3.406994e-09 * h[57] / H_AVG[57]
            + 3.206858e-09 * h[49] / H_AVG[49]
            - 3.138131e-09 * h[22] / H_AVG[22]
            + 3.133121e-09 * h[77] / H_AVG[77]
            + 2.90963e-09 * h[122] / H_AVG[122]
            + 2.347794e-09 * h[72] / H_AVG[72]
            + 2.244466e-09 * h[51] / H_AVG[51]
            - 2.225473e-09 * h[23] / H_AVG[23]
            - 1.531703e-09 * h[89] / H_AVG[89]
            + 1.494847e-09 * h[125] / H_AVG[125]
            - 1.40054e-09 * h[75] / H_AVG[75]
            - 1.210576e-09 * h[87] / H_AVG[87]
            + 1.205638e-09 * h[93] / H_AVG[93]
            - 1.0926e-09 * h[40] / H_AVG[40]
            - 1.006197e-09 * h[20] / H_AVG[20]
            + 8.973772e-10 * h[119] / H_AVG[119]
            - 8.293918e-10 * h[62] / H_AVG[62]
            - 7.477486e-10 * h[95] / H_AVG[95]
            - 6.95795e-10 * h[96] / H_AVG[96]
            + 4.016043e-10 * h[116] / H_AVG[116]
            + 3.860273e-10 * h[56] / H_AVG[56]
            + 2.687909e-10 * h[85] / H_AVG[85]
            + 1.61086e-10 * h[109] / H_AVG[109]
            - 1.389004e-10 * h[98] / H_AVG[98]
            - 1.127318e-10 * h[88] / H_AVG[88]
            - 7.452394e-11 * h[46] / H_AVG[46]
            - 7.18837e-11 * h[47] / H_AVG[47]
            - 4.456895e-11 * h[64] / H_AVG[64]
            + 1.665506e-11 * h[127] / H_AVG[127]
            + 1.606586e-12 * h[41] / H_AVG[41]
        ),
        0.1286925 + T[3] * (   # class Hgg
            - 0.2148928 * h[104] / H_AVG[104]
            - 0.1687763 * h[68] / H_AVG[68]
            - 0.1025248 * h[16] / H_AVG[16]
            - 0.100823 * h[18] / H_AVG[18]
            + 0.08394678 * h[52] / H_AVG[52]
            - 0.07281774 * h[97] / H_AVG[97]
            - 0.05372491 * h[81] / H_AVG[81]
            + 0.05341389 * h[124] / H_AVG[124]
            - 0.04912052 * h[24] / H_AVG[24]
            + 0.04842442 * h[115] / H_AVG[115]
            - 0.02185696 * h[78] / H_AVG[78]
            + 0.01529108 * h[83] / H_AVG[83]
            + 0.01251948 * h[120] / H_AVG[120]
            - 0.0009188774 * h[25] / H_AVG[25]
            + 0.0005343344 * h[70] / H_AVG[70]
            + 0.0002331724 * h[0] / H_AVG[0]
            - 8.868998e-05 * h[123] / H_AVG[123]
            - 3.77834e-05 * h[99] / H_AVG[99]
            - 2.713501e-05 * h[21] / H_AVG[21]
            + 2.176681e-05 * h[90] / H_AVG[90]
            + 1.78542e-06 * h[84] / H_AVG[84]
            + 4.214271e-07 * h[55] / H_AVG[55]
            + 2.827791e-07 * h[79] / H_AVG[79]
            - 2.699233e-07 * h[105] / H_AVG[105]
            - 2.190812e-07 * h[106] / H_AVG[106]
            + 1.780386e-07 * h[113] / H_AVG[113]
            - 1.691532e-07 * h[34] / H_AVG[34]
            + 1.51789e-07 * h[73] / H_AVG[73]
            + 1.410325e-07 * h[10] / H_AVG[10]
            + 1.344133e-07 * h[11] / H_AVG[11]
            - 1.144243e-07 * h[6] / H_AVG[6]
            + 1.092232e-07 * h[86] / H_AVG[86]
            - 1.038402e-07 * h[32] / H_AVG[32]
            + 9.548441e-08 * h[17] / H_AVG[17]
            + 8.631575e-08 * h[38] / H_AVG[38]
            - 7.082177e-08 * h[42] / H_AVG[42]
            - 6.314076e-08 * h[54] / H_AVG[54]
            - 5.999107e-08 * h[29] / H_AVG[29]
            + 5.440262e-08 * h[118] / H_AVG[118]
            + 4.245543e-08 * h[48] / H_AVG[48]
            - 4.054802e-08 * h[45] / H_AVG[45]
            - 3.914564e-08 * h[39] / H_AVG[39]
            + 3.608513e-08 * h[27] / H_AVG[27]
            + 3.595201e-08 * h[63] / H_AVG[63]
            + 3.55328e-08 * h[30] / H_AVG[30]
            + 3.483259e-08 * h[43] / H_AVG[43]
            + 3.242136e-08 * h[121] / H_AVG[121]
            - 3.180636e-08 * h[80] / H_AVG[80]
            - 2.912042e-08 * h[126] / H_AVG[126]
            + 2.811968e-08 * h[60] / H_AVG[60]
            - 2.766825e-08 * h[61] / H_AVG[61]
            - 2.626619e-08 * h[102] / H_AVG[102]
            - 2.622975e-08 * h[107] / H_AVG[107]
            - 2.584983e-08 * h[7] / H_AVG[7]
            + 2.327052e-08 * h[92] / H_AVG[92]
            + 2.278577e-08 * h[35] / H_AVG[35]
            + 2.214068e-08 * h[15] / H_AVG[15]
            - 2.13881e-08 * h[59] / H_AVG[59]
            - 2.062306e-08 * h[101] / H_AVG[101]
            - 1.972502e-08 * h[28] / H_AVG[28]
            + 1.913338e-08 * h[103] / H_AVG[103]
            + 1.800028e-08 * h[69] / H_AVG[69]
            + 1.744434e-08 * h[67] / H_AVG[67]
            + 1.692792e-08 * h[1] / H_AVG[1]
            + 1.680228e-08 * h[100] / H_AVG[100]
            - 1.541071e-08 * h[110] / H_AVG[110]
            - 1.492222e-08 * h[3] / H_AVG[3]
            + 1.431993e-08 * h[37] / H_AVG[37]
            - 1.404184e-08 * h[50] / H_AVG[50]
            + 1.321698e-08 * h[44] / H_AVG[44]
            + 1.318342e-08 * h[36] / H_AVG[36]
            - 1.295366e-08 * h[12] / H_AVG[12]
            - 1.233186e-08 * h[114] / H_AVG[114]
            - 1.22162e-08 * h[2] / H_AVG[2]
            + 1.178585e-08 * h[58] / H_AVG[58]
            - 1.172962e-08 * h[112] / H_AVG[112]
            + 1.149185e-08 * h[76] / H_AVG[76]
            - 1.108569e-08 * h[13] / H_AVG[13]
            - 1.072719e-08 * h[19] / H_AVG[19]
            - 1.054928e-08 * h[111] / H_AVG[111]
            + 1.006718e-08 * h[74] / H_AVG[74]
            + 9.544441e-09 * h[108] / H_AVG[108]
            + 8.971981e-09 * h[5] / H_AVG[5]
            - 8.578882e-09 * h[8] / H_AVG[8]
            - 8.223018e-09 * h[91] / H_AVG[91]
            - 7.660265e-09 * h[4] / H_AVG[4]
            - 7.329609e-09 * h[117] / H_AVG[117]
            + 6.203168e-09 * h[53] / H_AVG[53]
            - 6.140334e-09 * h[66] / H_AVG[66]
            - 5.590982e-09 * h[33] / H_AVG[33]
            - 5.419926e-09 * h[65] / H_AVG[65]
            + 5.020161e-09 * h[82] / H_AVG[82]
            - 4.711899e-09 * h[31] / H_AVG[31]
            - 4.335285e-09 * h[71] / H_AVG[71]
            + 4.02431e-09 * h[94] / H_AVG[94]
            + 3.882061e-09 * h[9] / H_AVG[9]
            + 3.412366e-09 * h[26] / H_AVG[26]
            - 3.068302e-09 * h[14] / H_AVG[14]
            + 3.0423e-09 * h[57] / H_AVG[57]
            + 2.864296e-09 * h[49] / H_AVG[49]
            - 2.800334e-09 * h[22] / H_AVG[22]
            + 2.799047e-09 * h[77] / H_AVG[77]
            + 2.597065e-09 * h[122] / H_AVG[122]
            + 2.097573e-09 * h[72] / H_AVG[72]
            + 2.004498e-09 * h[51] / H_AVG[51]
            - 1.986387e-09 * h[23] / H_AVG[23]
            - 1.368155e-09 * h[89] / H_AVG[89]
            + 1.334882e-09 * h[125] / H_AVG[125]
            - 1.250975e-09 * h[75] / H_AVG[75]
            - 1.080911e-09 * h[87] / H_AVG[87]
            + 1.076711e-09 * h[93] / H_AVG[93]
            - 9.75846e-10 * h[40] / H_AVG[40]
            - 8.984806e-10 * h[20] / H_AVG[20]
            + 8.012565e-10 * h[119] / H_AVG[119]
            - 7.406093e-10 * h[62] / H_AVG[62]
            - 6.678178e-10 * h[95] / H_AVG[95]
            - 6.214645e-10 * h[96] / H_AVG[96]
            + 3.58472e-10 * h[116] / H_AVG[116]
            + 3.445141e-10 * h[56] / H_AVG[56]
            + 2.400106e-10 * h[85] / H_AVG[85]
            + 1.438781e-10 * h[109] / H_AVG[109]
            - 1.240377e-10 * h[98] / H_AVG[98]
            - 1.008367e-10 * h[88] / H_AVG[88]
            - 6.654253e-11 * h[46] / H_AVG[46]
            - 6.421249e-11 * h[47] / H_AVG[47]
            - 3.816604e-11 * h[64] / H_AVG[64]
            + 1.487554e-11 * h[127] / H_AVG[127]
            + 1.432693e-12 * h[41] / H_AVG[41]
        ),
        -0.07702489 + T[4] * (   # class H4q
            - 0.2798565 * h[115] / H_AVG[115]
            - 0.1001314 * h[104] / H_AVG[104]
            - 0.09152598 * h[16] / H_AVG[16]
            + 0.08016123 * h[52] / H_AVG[52]
            - 0.07784136 * h[78] / H_AVG[78]
            - 0.07662744 * h[24] / H_AVG[24]
            - 0.06988126 * h[97] / H_AVG[97]
            - 0.05321381 * h[81] / H_AVG[81]
            + 0.0500239 * h[68] / H_AVG[68]
            + 0.04997177 * h[124] / H_AVG[124]
            - 0.04546122 * h[18] / H_AVG[18]
            + 0.0130745 * h[120] / H_AVG[120]
            - 0.0103853 * h[83] / H_AVG[83]
            - 0.0008821805 * h[25] / H_AVG[25]
            + 0.0006251967 * h[70] / H_AVG[70]
            + 0.0002375887 * h[0] / H_AVG[0]
            - 3.468498e-05 * h[99] / H_AVG[99]
            - 2.794187e-05 * h[21] / H_AVG[21]
            + 2.744998e-05 * h[123] / H_AVG[123]
            - 4.168314e-06 * h[90] / H_AVG[90]
            + 1.641329e-06 * h[84] / H_AVG[84]
            + 3.875909e-07 * h[55] / H_AVG[55]
            + 2.600004e-07 * h[79] / H_AVG[79]
            - 2.481719e-07 * h[105] / H_AVG[105]
            - 2.014992e-07 * h[106] / H_AVG[106]
            + 1.636523e-07 * h[113] / H_AVG[113]
            - 1.55523e-07 * h[34] / H_AVG[34]
            + 1.395645e-07 * h[73] / H_AVG[73]
            + 1.296871e-07 * h[10] / H_AVG[10]
            + 1.235758e-07 * h[11] / H_AVG[11]
            - 1.052175e-07 * h[6] / H_AVG[6]
            + 1.004218e-07 * h[86] / H_AVG[86]
            - 9.545911e-08 * h[32] / H_AVG[32]
            + 8.779729e-08 * h[17] / H_AVG[17]
            + 7.936117e-08 * h[38] / H_AVG[38]
            - 6.511914e-08 * h[42] / H_AVG[42]
            - 5.806571e-08 * h[54] / H_AVG[54]
            - 5.517095e-08 * h[29] / H_AVG[29]
            + 5.002285e-08 * h[118] / H_AVG[118]
            + 3.903118e-08 * h[48] / H_AVG[48]
            - 3.728043e-08 * h[45] / H_AVG[45]
            - 3.599264e-08 * h[39] / H_AVG[39]
            + 3.317937e-08 * h[27] / H_AVG[27]
            + 3.305539e-08 * h[63] / H_AVG[63]
            + 3.267344e-08 * h[30] / H_AVG[30]
            + 3.202701e-08 * h[43] / H_AVG[43]
            + 2.980931e-08 * h[121] / H_AVG[121]
            - 2.924828e-08 * h[80] / H_AVG[80]
            - 2.677808e-08 * h[126] / H_AVG[126]
            + 2.586429e-08 * h[60] / H_AVG[60]
            - 2.543283e-08 * h[61] / H_AVG[61]
            - 2.414862e-08 * h[102] / H_AVG[102]
            - 2.411733e-08 * h[107] / H_AVG[107]
            - 2.377033e-08 * h[7] / H_AVG[7]
            + 2.13946e-08 * h[92] / H_AVG[92]
            + 2.095284e-08 * h[35] / H_AVG[35]
            + 2.036016e-08 * h[15] / H_AVG[15]
            - 1.965888e-08 * h[59] / H_AVG[59]
            - 1.896192e-08 * h[101] / H_AVG[101]
            - 1.813743e-08 * h[28] / H_AVG[28]
            + 1.759154e-08 * h[103] / H_AVG[103]
            + 1.654955e-08 * h[69] / H_AVG[69]
            + 1.604252e-08 * h[67] / H_AVG[67]
            + 1.556433e-08 * h[1] / H_AVG[1]
            + 1.544804e-08 * h[100] / H_AVG[100]
            - 1.417218e-08 * h[110] / H_AVG[110]
            - 1.371776e-08 * h[3] / H_AVG[3]
            + 1.31666e-08 * h[37] / H_AVG[37]
            - 1.290059e-08 * h[50] / H_AVG[50]
            + 1.215205e-08 * h[44] / H_AVG[44]
            + 1.212128e-08 * h[36] / H_AVG[36]
            - 1.190713e-08 * h[12] / H_AVG[12]
            - 1.133972e-08 * h[114] / H_AVG[114]
            - 1.123439e-08 * h[2] / H_AVG[2]
            + 1.083562e-08 * h[58] / H_AVG[58]
            - 1.07828e-08 * h[112] / H_AVG[112]
            + 1.056557e-08 * h[76] / H_AVG[76]
            - 1.019144e-08 * h[13] / H_AVG[13]
            - 9.859763e-09 * h[19] / H_AVG[19]
            - 9.698902e-09 * h[111] / H_AVG[111]
            + 9.255099e-09 * h[74] / H_AVG[74]
            + 8.774313e-09 * h[108] / H_AVG[108]
            + 8.249892e-09 * h[5] / H_AVG[5]
            - 7.887989e-09 * h[8] / H_AVG[8]
            - 7.564801e-09 * h[91] / H_AVG[91]
            - 7.042825e-09 * h[4] / H_AVG[4]
            - 6.740372e-09 * h[117] / H_AVG[117]
            + 5.702716e-09 * h[53] / H_AVG[53]
            - 5.645939e-09 * h[66] / H_AVG[66]
            - 5.139769e-09 * h[33] / H_AVG[33]
            - 4.984415e-09 * h[65] / H_AVG[65]
            + 4.617145e-09 * h[82] / H_AVG[82]
            - 4.332588e-09 * h[31] / H_AVG[31]
            - 3.985983e-09 * h[71] / H_AVG[71]
            + 3.700548e-09 * h[94] / H_AVG[94]
            + 3.569022e-09 * h[9] / H_AVG[9]
            + 3.137388e-09 * h[26] / H_AVG[26]
            - 2.820698e-09 * h[14] / H_AVG[14]
            + 2.797152e-09 * h[57] / H_AVG[57]
            + 2.633488e-09 * h[49] / H_AVG[49]
            - 2.576538e-09 * h[22] / H_AVG[22]
            + 2.575655e-09 * h[77] / H_AVG[77]
            + 2.388384e-09 * h[122] / H_AVG[122]
            + 1.928584e-09 * h[72] / H_AVG[72]
            + 1.842686e-09 * h[51] / H_AVG[51]
            - 1.826125e-09 * h[23] / H_AVG[23]
            - 1.257834e-09 * h[89] / H_AVG[89]
            + 1.227581e-09 * h[125] / H_AVG[125]
            - 1.150128e-09 * h[75] / H_AVG[75]
            - 9.939552e-10 * h[87] / H_AVG[87]
            + 9.905664e-10 * h[93] / H_AVG[93]
            - 8.972769e-10 * h[40] / H_AVG[40]
            - 8.26046e-10 * h[20] / H_AVG[20]
            + 7.371075e-10 * h[119] / H_AVG[119]
            - 6.811494e-10 * h[62] / H_AVG[62]
            - 6.142667e-10 * h[95] / H_AVG[95]
            - 5.715282e-10 * h[96] / H_AVG[96]
            + 3.296544e-10 * h[116] / H_AVG[116]
            + 3.16993e-10 * h[56] / H_AVG[56]
            + 2.205661e-10 * h[85] / H_AVG[85]
            + 1.323233e-10 * h[109] / H_AVG[109]
            - 1.140501e-10 * h[98] / H_AVG[98]
            - 9.266352e-11 * h[88] / H_AVG[88]
            - 6.119116e-11 * h[46] / H_AVG[46]
            - 5.904592e-11 * h[47] / H_AVG[47]
            - 3.582862e-11 * h[64] / H_AVG[64]
            + 1.366053e-11 * h[127] / H_AVG[127]
            + 1.318539e-12 * h[41] / H_AVG[41]
        ),
        -0.9789527 + T[5] * (   # class Hqql
            + 0.1931361 * h[120] / H_AVG[120]
            + 0.1411926 * h[24] / H_AVG[24]
            + 0.1245468 * h[97] / H_AVG[97]
            + 0.1077152 * h[18] / H_AVG[18]
            + 0.08904346 * h[52] / H_AVG[52]
            - 0.07716822 * h[124] / H_AVG[124]
            + 0.07088914 * h[83] / H_AVG[83]
            - 0.06835916 * h[115] / H_AVG[115]
            + 0.06516324 * h[78] / H_AVG[78]
            - 0.01868894 * h[81] / H_AVG[81]
            + 0.0176761 * h[68] / H_AVG[68]
            + 0.01505731 * h[104] / H_AVG[104]
            - 0.005555847 * h[16] / H_AVG[16]
            + 0.00256888 * h[25] / H_AVG[25]
            + 0.001532934 * h[70] / H_AVG[70]
            - 0.001161788 * h[123] / H_AVG[123]
            - 0.0003465555 * h[0] / H_AVG[0]
            + 0.0001021673 * h[90] / H_AVG[90]
            + 7.203227e-05 * h[21] / H_AVG[21]
            - 2.054707e-05 * h[99] / H_AVG[99]
            + 9.599571e-07 * h[84] / H_AVG[84]
            + 2.265653e-07 * h[55] / H_AVG[55]
            + 1.519477e-07 * h[79] / H_AVG[79]
            - 1.450595e-07 * h[105] / H_AVG[105]
            - 1.178056e-07 * h[106] / H_AVG[106]
            + 9.572947e-08 * h[113] / H_AVG[113]
            - 9.092551e-08 * h[34] / H_AVG[34]
            + 8.157023e-08 * h[73] / H_AVG[73]
            + 7.577189e-08 * h[10] / H_AVG[10]
            + 7.230264e-08 * h[11] / H_AVG[11]
            - 6.146137e-08 * h[6] / H_AVG[6]
            + 5.868896e-08 * h[86] / H_AVG[86]
            - 5.579591e-08 * h[32] / H_AVG[32]
            + 5.133143e-08 * h[17] / H_AVG[17]
            + 4.636666e-08 * h[38] / H_AVG[38]
            - 3.805554e-08 * h[42] / H_AVG[42]
            - 3.396404e-08 * h[54] / H_AVG[54]
            - 3.224239e-08 * h[29] / H_AVG[29]
            + 2.924848e-08 * h[118] / H_AVG[118]
            + 2.28041e-08 * h[48] / H_AVG[48]
            - 2.179568e-08 * h[45] / H_AVG[45]
            - 2.102705e-08 * h[39] / H_AVG[39]
            + 1.939806e-08 * h[27] / H_AVG[27]
            + 1.932567e-08 * h[63] / H_AVG[63]
            + 1.91025e-08 * h[30] / H_AVG[30]
            + 1.870846e-08 * h[43] / H_AVG[43]
            + 1.742076e-08 * h[121] / H_AVG[121]
            - 1.709652e-08 * h[80] / H_AVG[80]
            - 1.567372e-08 * h[126] / H_AVG[126]
            + 1.511164e-08 * h[60] / H_AVG[60]
            - 1.485902e-08 * h[61] / H_AVG[61]
            - 1.411119e-08 * h[102] / H_AVG[102]
            - 1.407577e-08 * h[107] / H_AVG[107]
            - 1.389691e-08 * h[7] / H_AVG[7]
            + 1.250006e-08 * h[92] / H_AVG[92]
            + 1.223574e-08 * h[35] / H_AVG[35]
            + 1.189898e-08 * h[15] / H_AVG[15]
            - 1.150309e-08 * h[59] / H_AVG[59]
            - 1.109356e-08 * h[101] / H_AVG[101]
            - 1.058064e-08 * h[28] / H_AVG[28]
            + 1.028764e-08 * h[103] / H_AVG[103]
            + 9.686639e-09 * h[69] / H_AVG[69]
            + 9.393959e-09 * h[67] / H_AVG[67]
            + 9.093779e-09 * h[1] / H_AVG[1]
            + 9.024742e-09 * h[100] / H_AVG[100]
            - 8.277533e-09 * h[110] / H_AVG[110]
            - 8.025304e-09 * h[3] / H_AVG[3]
            + 7.697769e-09 * h[37] / H_AVG[37]
            - 7.537007e-09 * h[50] / H_AVG[50]
            + 7.098588e-09 * h[44] / H_AVG[44]
            + 7.090239e-09 * h[36] / H_AVG[36]
            - 6.969081e-09 * h[12] / H_AVG[12]
            - 6.6285e-09 * h[114] / H_AVG[114]
            - 6.574036e-09 * h[2] / H_AVG[2]
            + 6.339952e-09 * h[58] / H_AVG[58]
            - 6.31555e-09 * h[112] / H_AVG[112]
            + 6.180931e-09 * h[76] / H_AVG[76]
            - 5.951204e-09 * h[13] / H_AVG[13]
            - 5.765948e-09 * h[19] / H_AVG[19]
            - 5.664781e-09 * h[111] / H_AVG[111]
            + 5.408858e-09 * h[74] / H_AVG[74]
            + 5.119287e-09 * h[108] / H_AVG[108]
            + 4.817596e-09 * h[5] / H_AVG[5]
            - 4.61503e-09 * h[8] / H_AVG[8]
            - 4.429647e-09 * h[91] / H_AVG[91]
            - 4.119744e-09 * h[4] / H_AVG[4]
            - 3.941467e-09 * h[117] / H_AVG[117]
            + 3.336187e-09 * h[53] / H_AVG[53]
            - 3.299531e-09 * h[66] / H_AVG[66]
            - 2.996665e-09 * h[33] / H_AVG[33]
            - 2.923696e-09 * h[65] / H_AVG[65]
            + 2.703693e-09 * h[82] / H_AVG[82]
            - 2.524805e-09 * h[31] / H_AVG[31]
            - 2.327659e-09 * h[71] / H_AVG[71]
            + 2.162128e-09 * h[94] / H_AVG[94]
            + 2.086008e-09 * h[9] / H_AVG[9]
            + 1.842379e-09 * h[26] / H_AVG[26]
            - 1.650334e-09 * h[14] / H_AVG[14]
            + 1.640605e-09 * h[57] / H_AVG[57]
            + 1.537725e-09 * h[49] / H_AVG[49]
            - 1.505128e-09 * h[22] / H_AVG[22]
            + 1.504816e-09 * h[77] / H_AVG[77]
            + 1.400068e-09 * h[122] / H_AVG[122]
            + 1.126183e-09 * h[72] / H_AVG[72]
            + 1.081023e-09 * h[51] / H_AVG[51]
            - 1.069429e-09 * h[23] / H_AVG[23]
            - 7.313149e-10 * h[89] / H_AVG[89]
            + 7.162343e-10 * h[125] / H_AVG[125]
            - 6.709262e-10 * h[75] / H_AVG[75]
            + 5.791885e-10 * h[93] / H_AVG[93]
            - 5.783665e-10 * h[87] / H_AVG[87]
            - 5.242187e-10 * h[40] / H_AVG[40]
            - 4.827645e-10 * h[20] / H_AVG[20]
            + 4.321706e-10 * h[119] / H_AVG[119]
            - 3.970722e-10 * h[62] / H_AVG[62]
            - 3.588778e-10 * h[95] / H_AVG[95]
            - 3.332493e-10 * h[96] / H_AVG[96]
            + 1.92377e-10 * h[116] / H_AVG[116]
            + 1.859248e-10 * h[56] / H_AVG[56]
            + 1.298137e-10 * h[85] / H_AVG[85]
            + 7.708099e-11 * h[109] / H_AVG[109]
            - 6.647132e-11 * h[98] / H_AVG[98]
            - 5.380816e-11 * h[88] / H_AVG[88]
            - 3.599386e-11 * h[46] / H_AVG[46]
            - 3.442406e-11 * h[47] / H_AVG[47]
            - 1.617082e-11 * h[64] / H_AVG[64]
            + 8.26644e-12 * h[127] / H_AVG[127]
            + 8.019276e-13 * h[41] / H_AVG[41]
        ),
        0.3035426 + T[6] * (   # class Zqq
            + 0.1138135 * h[24] / H_AVG[24]
            + 0.1088785 * h[104] / H_AVG[104]
            + 0.1065623 * h[115] / H_AVG[115]
            - 0.1046475 * h[124] / H_AVG[124]
            - 0.09850024 * h[97] / H_AVG[97]
            - 0.09547039 * h[120] / H_AVG[120]
            + 0.08843262 * h[68] / H_AVG[68]
            + 0.08313297 * h[83] / H_AVG[83]
            + 0.06865234 * h[81] / H_AVG[81]
            - 0.05333124 * h[52] / H_AVG[52]
            - 0.04766938 * h[16] / H_AVG[16]
            + 0.01758129 * h[18] / H_AVG[18]
            - 0.009245022 * h[78] / H_AVG[78]
            + 0.00198663 * h[25] / H_AVG[25]
            - 0.001387356 * h[70] / H_AVG[70]
            + 0.0005092723 * h[123] / H_AVG[123]
            - 6.362151e-05 * h[0] / H_AVG[0]
            - 4.785964e-05 * h[90] / H_AVG[90]
            - 4.519749e-05 * h[99] / H_AVG[99]
            - 3.612026e-05 * h[21] / H_AVG[21]
            + 2.128293e-06 * h[84] / H_AVG[84]
            + 5.025639e-07 * h[55] / H_AVG[55]
            + 3.371001e-07 * h[79] / H_AVG[79]
            - 3.218071e-07 * h[105] / H_AVG[105]
            - 2.612652e-07 * h[106] / H_AVG[106]
            + 2.121976e-07 * h[113] / H_AVG[113]
            - 2.016664e-07 * h[34] / H_AVG[34]
            + 1.80963e-07 * h[73] / H_AVG[73]
            + 1.681911e-07 * h[10] / H_AVG[10]
            + 1.602639e-07 * h[11] / H_AVG[11]
            - 1.36441e-07 * h[6] / H_AVG[6]
            + 1.302103e-07 * h[86] / H_AVG[86]
            - 1.237728e-07 * h[32] / H_AVG[32]
            + 1.138011e-07 * h[17] / H_AVG[17]
            + 1.028931e-07 * h[38] / H_AVG[38]
            - 8.443688e-08 * h[42] / H_AVG[42]
            - 7.527773e-08 * h[54] / H_AVG[54]
            - 7.154133e-08 * h[29] / H_AVG[29]
            + 6.485835e-08 * h[118] / H_AVG[118]
            + 5.061468e-08 * h[48] / H_AVG[48]
            - 4.833952e-08 * h[45] / H_AVG[45]
            - 4.668356e-08 * h[39] / H_AVG[39]
            + 4.30237e-08 * h[27] / H_AVG[27]
            + 4.28651e-08 * h[63] / H_AVG[63]
            + 4.236175e-08 * h[30] / H_AVG[30]
            + 4.152795e-08 * h[43] / H_AVG[43]
            + 3.866011e-08 * h[121] / H_AVG[121]
            - 3.792472e-08 * h[80] / H_AVG[80]
            - 3.470034e-08 * h[126] / H_AVG[126]
            + 3.353216e-08 * h[60] / H_AVG[60]
            - 3.298896e-08 * h[61] / H_AVG[61]
            - 3.131951e-08 * h[102] / H_AVG[102]
            - 3.125888e-08 * h[107] / H_AVG[107]
            - 3.083761e-08 * h[7] / H_AVG[7]
            + 2.774764e-08 * h[92] / H_AVG[92]
            + 2.714602e-08 * h[35] / H_AVG[35]
            + 2.639764e-08 * h[15] / H_AVG[15]
            - 2.550124e-08 * h[59] / H_AVG[59]
            - 2.459008e-08 * h[101] / H_AVG[101]
            - 2.352811e-08 * h[28] / H_AVG[28]
            + 2.281069e-08 * h[103] / H_AVG[103]
            + 2.146299e-08 * h[69] / H_AVG[69]
            + 2.080013e-08 * h[67] / H_AVG[67]
            + 2.018484e-08 * h[1] / H_AVG[1]
            + 2.003588e-08 * h[100] / H_AVG[100]
            - 1.837795e-08 * h[110] / H_AVG[110]
            - 1.778562e-08 * h[3] / H_AVG[3]
            + 1.707323e-08 * h[37] / H_AVG[37]
            - 1.673119e-08 * h[50] / H_AVG[50]
            + 1.575787e-08 * h[44] / H_AVG[44]
            + 1.571833e-08 * h[36] / H_AVG[36]
            - 1.544674e-08 * h[12] / H_AVG[12]
            - 1.4702e-08 * h[114] / H_AVG[114]
            - 1.455718e-08 * h[2] / H_AVG[2]
            + 1.404984e-08 * h[58] / H_AVG[58]
            - 1.398335e-08 * h[112] / H_AVG[112]
            + 1.369912e-08 * h[76] / H_AVG[76]
            - 1.321697e-08 * h[13] / H_AVG[13]
            - 1.278942e-08 * h[19] / H_AVG[19]
            - 1.257724e-08 * h[111] / H_AVG[111]
            + 1.200138e-08 * h[74] / H_AVG[74]
            + 1.137934e-08 * h[108] / H_AVG[108]
            + 1.069739e-08 * h[5] / H_AVG[5]
            - 1.022789e-08 * h[8] / H_AVG[8]
            - 9.808499e-09 * h[91] / H_AVG[91]
            - 9.132624e-09 * h[4] / H_AVG[4]
            - 8.738966e-09 * h[117] / H_AVG[117]
            + 7.395422e-09 * h[53] / H_AVG[53]
            - 7.317453e-09 * h[66] / H_AVG[66]
            - 6.66443e-09 * h[33] / H_AVG[33]
            - 6.465918e-09 * h[65] / H_AVG[65]
            + 5.986115e-09 * h[82] / H_AVG[82]
            - 5.617407e-09 * h[31] / H_AVG[31]
            - 5.167318e-09 * h[71] / H_AVG[71]
            + 4.797945e-09 * h[94] / H_AVG[94]
            + 4.62793e-09 * h[9] / H_AVG[9]
            + 4.067809e-09 * h[26] / H_AVG[26]
            - 3.657998e-09 * h[14] / H_AVG[14]
            + 3.625878e-09 * h[57] / H_AVG[57]
            + 3.414678e-09 * h[49] / H_AVG[49]
            - 3.339941e-09 * h[22] / H_AVG[22]
            + 3.339607e-09 * h[77] / H_AVG[77]
            + 3.096438e-09 * h[122] / H_AVG[122]
            + 2.500962e-09 * h[72] / H_AVG[72]
            + 2.387976e-09 * h[51] / H_AVG[51]
            - 2.36951e-09 * h[23] / H_AVG[23]
            - 1.631477e-09 * h[89] / H_AVG[89]
            + 1.591782e-09 * h[125] / H_AVG[125]
            - 1.491471e-09 * h[75] / H_AVG[75]
            - 1.28936e-09 * h[87] / H_AVG[87]
            + 1.284405e-09 * h[93] / H_AVG[93]
            - 1.163437e-09 * h[40] / H_AVG[40]
            - 1.071105e-09 * h[20] / H_AVG[20]
            + 9.556477e-10 * h[119] / H_AVG[119]
            - 8.831246e-10 * h[62] / H_AVG[62]
            - 7.963281e-10 * h[95] / H_AVG[95]
            - 7.410567e-10 * h[96] / H_AVG[96]
            + 4.275013e-10 * h[116] / H_AVG[116]
            + 4.107557e-10 * h[56] / H_AVG[56]
            + 2.859e-10 * h[85] / H_AVG[85]
            + 1.715937e-10 * h[109] / H_AVG[109]
            - 1.479144e-10 * h[98] / H_AVG[98]
            - 1.202124e-10 * h[88] / H_AVG[88]
            - 7.933853e-11 * h[46] / H_AVG[46]
            - 7.655826e-11 * h[47] / H_AVG[47]
            - 4.275344e-11 * h[64] / H_AVG[64]
            + 1.769331e-11 * h[127] / H_AVG[127]
            + 1.705668e-12 * h[41] / H_AVG[41]
        ),
        0.09887857 + T[7] * (   # class Wqq
            + 0.2688687 * h[97] / H_AVG[97]
            + 0.1091944 * h[68] / H_AVG[68]
            + 0.1079542 * h[24] / H_AVG[24]
            + 0.09378617 * h[83] / H_AVG[83]
            - 0.08634399 * h[52] / H_AVG[52]
            - 0.08294646 * h[120] / H_AVG[120]
            + 0.05546083 * h[104] / H_AVG[104]
            + 0.05097424 * h[115] / H_AVG[115]
            - 0.04466714 * h[81] / H_AVG[81]
            - 0.04416897 * h[16] / H_AVG[16]
            - 0.03796225 * h[78] / H_AVG[78]
            + 0.0122339 * h[124] / H_AVG[124]
            - 0.002198857 * h[25] / H_AVG[25]
            + 0.001689992 * h[18] / H_AVG[18]
            - 0.001101227 * h[70] / H_AVG[70]
            - 0.0001775095 * h[0] / H_AVG[0]
            + 0.0001627696 * h[123] / H_AVG[123]
            - 4.003477e-05 * h[99] / H_AVG[99]
            - 3.463335e-05 * h[90] / H_AVG[90]
            - 2.785279e-05 * h[21] / H_AVG[21]
            + 1.88277e-06 * h[84] / H_AVG[84]
            + 4.444506e-07 * h[55] / H_AVG[55]
            + 2.981987e-07 * h[79] / H_AVG[79]
            - 2.84664e-07 * h[105] / H_AVG[105]
            - 2.310916e-07 * h[106] / H_AVG[106]
            + 1.877331e-07 * h[113] / H_AVG[113]
            - 1.783843e-07 * h[34] / H_AVG[34]
            + 1.600891e-07 * h[73] / H_AVG[73]
            + 1.487269e-07 * h[10] / H_AVG[10]
            + 1.417649e-07 * h[11] / H_AVG[11]
            - 1.206961e-07 * h[6] / H_AVG[6]
            + 1.151943e-07 * h[86] / H_AVG[86]
            - 1.094838e-07 * h[32] / H_AVG[32]
            + 1.006528e-07 * h[17] / H_AVG[17]
            + 9.101849e-08 * h[38] / H_AVG[38]
            - 7.469778e-08 * h[42] / H_AVG[42]
            - 6.659861e-08 * h[54] / H_AVG[54]
            - 6.327792e-08 * h[29] / H_AVG[29]
            + 5.736994e-08 * h[118] / H_AVG[118]
            + 4.477446e-08 * h[48] / H_AVG[48]
            - 4.276156e-08 * h[45] / H_AVG[45]
            - 4.128418e-08 * h[39] / H_AVG[39]
            + 3.805883e-08 * h[27] / H_AVG[27]
            + 3.791478e-08 * h[63] / H_AVG[63]
            + 3.747476e-08 * h[30] / H_AVG[30]
            + 3.673289e-08 * h[43] / H_AVG[43]
            + 3.419416e-08 * h[121] / H_AVG[121]
            - 3.354414e-08 * h[80] / H_AVG[80]
            - 3.071028e-08 * h[126] / H_AVG[126]
            + 2.965354e-08 * h[60] / H_AVG[60]
            - 2.917242e-08 * h[61] / H_AVG[61]
            - 2.770303e-08 * h[102] / H_AVG[102]
            - 2.765281e-08 * h[107] / H_AVG[107]
            - 2.726574e-08 * h[7] / H_AVG[7]
            + 2.454283e-08 * h[92] / H_AVG[92]
            + 2.401615e-08 * h[35] / H_AVG[35]
            + 2.335056e-08 * h[15] / H_AVG[15]
            - 2.256603e-08 * h[59] / H_AVG[59]
            - 2.174979e-08 * h[101] / H_AVG[101]
            - 2.079947e-08 * h[28] / H_AVG[28]
            + 2.017922e-08 * h[103] / H_AVG[103]
            + 1.898345e-08 * h[69] / H_AVG[69]
            + 1.839634e-08 * h[67] / H_AVG[67]
            + 1.785218e-08 * h[1] / H_AVG[1]
            + 1.772149e-08 * h[100] / H_AVG[100]
            - 1.625164e-08 * h[110] / H_AVG[110]
            - 1.57363e-08 * h[3] / H_AVG[3]
            + 1.510199e-08 * h[37] / H_AVG[37]
            - 1.480194e-08 * h[50] / H_AVG[50]
            + 1.393777e-08 * h[44] / H_AVG[44]
            + 1.390441e-08 * h[36] / H_AVG[36]
            - 1.365808e-08 * h[12] / H_AVG[12]
            - 1.300339e-08 * h[114] / H_AVG[114]
            - 1.288574e-08 * h[2] / H_AVG[2]
            + 1.242792e-08 * h[58] / H_AVG[58]
            - 1.237008e-08 * h[112] / H_AVG[112]
            + 1.21175e-08 * h[76] / H_AVG[76]
            - 1.168987e-08 * h[13] / H_AVG[13]
            - 1.131271e-08 * h[19] / H_AVG[19]
            - 1.112431e-08 * h[111] / H_AVG[111]
            + 1.061535e-08 * h[74] / H_AVG[74]
            + 1.006652e-08 * h[108] / H_AVG[108]
            + 9.462169e-09 * h[5] / H_AVG[5]
            - 9.046915e-09 * h[8] / H_AVG[8]
            - 8.673359e-09 * h[91] / H_AVG[91]
            - 8.078129e-09 * h[4] / H_AVG[4]
            - 7.730147e-09 * h[117] / H_AVG[117]
            + 6.540369e-09 * h[53] / H_AVG[53]
            - 6.477749e-09 * h[66] / H_AVG[66]
            - 5.895563e-09 * h[33] / H_AVG[33]
            - 5.720938e-09 * h[65] / H_AVG[65]
            + 5.293993e-09 * h[82] / H_AVG[82]
            - 4.970831e-09 * h[31] / H_AVG[31]
            - 4.571793e-09 * h[71] / H_AVG[71]
            + 4.242233e-09 * h[94] / H_AVG[94]
            + 4.093643e-09 * h[9] / H_AVG[9]
            + 3.596407e-09 * h[26] / H_AVG[26]
            - 3.236245e-09 * h[14] / H_AVG[14]
            + 3.207433e-09 * h[57] / H_AVG[57]
            + 3.020492e-09 * h[49] / H_AVG[49]
            - 2.954812e-09 * h[22] / H_AVG[22]
            + 2.951652e-09 * h[77] / H_AVG[77]
            + 2.739013e-09 * h[122] / H_AVG[122]
            + 2.212206e-09 * h[72] / H_AVG[72]
            + 2.113247e-09 * h[51] / H_AVG[51]
            - 2.09673e-09 * h[23] / H_AVG[23]
            - 1.443169e-09 * h[89] / H_AVG[89]
            + 1.407829e-09 * h[125] / H_AVG[125]
            - 1.319448e-09 * h[75] / H_AVG[75]
            - 1.139887e-09 * h[87] / H_AVG[87]
            + 1.13524e-09 * h[93] / H_AVG[93]
            - 1.029073e-09 * h[40] / H_AVG[40]
            - 9.474612e-10 * h[20] / H_AVG[20]
            + 8.445581e-10 * h[119] / H_AVG[119]
            - 7.81135e-10 * h[62] / H_AVG[62]
            - 7.043166e-10 * h[95] / H_AVG[95]
            - 6.555613e-10 * h[96] / H_AVG[96]
            + 3.78078e-10 * h[116] / H_AVG[116]
            + 3.633501e-10 * h[56] / H_AVG[56]
            + 2.529748e-10 * h[85] / H_AVG[85]
            + 1.517397e-10 * h[109] / H_AVG[109]
            - 1.308231e-10 * h[98] / H_AVG[98]
            - 1.062811e-10 * h[88] / H_AVG[88]
            - 7.015759e-11 * h[46] / H_AVG[46]
            - 6.772045e-11 * h[47] / H_AVG[47]
            - 3.926755e-11 * h[64] / H_AVG[64]
            + 1.564329e-11 * h[127] / H_AVG[127]
            + 1.51041e-12 * h[41] / H_AVG[41]
        ),
        -0.4320669 + T[8] * (   # class Tbqq
            - 0.1951115 * h[104] / H_AVG[104]
            + 0.1529202 * h[16] / H_AVG[16]
            + 0.1321378 * h[68] / H_AVG[68]
            + 0.1308624 * h[120] / H_AVG[120]
            - 0.1007241 * h[83] / H_AVG[83]
            - 0.0798691 * h[18] / H_AVG[18]
            + 0.05091478 * h[78] / H_AVG[78]
            - 0.03740935 * h[24] / H_AVG[24]
            + 0.03173595 * h[81] / H_AVG[81]
            + 0.02844861 * h[97] / H_AVG[97]
            - 0.02582119 * h[52] / H_AVG[52]
            - 0.02368494 * h[115] / H_AVG[115]
            + 0.008048015 * h[124] / H_AVG[124]
            - 0.001155994 * h[70] / H_AVG[70]
            - 0.0007631215 * h[123] / H_AVG[123]
            + 0.0001853978 * h[25] / H_AVG[25]
            - 0.0001259989 * h[0] / H_AVG[0]
            - 4.303247e-05 * h[90] / H_AVG[90]
            - 2.411371e-05 * h[99] / H_AVG[99]
            - 1.097855e-05 * h[21] / H_AVG[21]
            + 1.142423e-06 * h[84] / H_AVG[84]
            + 2.697463e-07 * h[55] / H_AVG[55]
            + 1.810151e-07 * h[79] / H_AVG[79]
            - 1.72704e-07 * h[105] / H_AVG[105]
            - 1.401959e-07 * h[106] / H_AVG[106]
            + 1.139199e-07 * h[113] / H_AVG[113]
            - 1.082351e-07 * h[34] / H_AVG[34]
            + 9.711304e-08 * h[73] / H_AVG[73]
            + 9.025112e-08 * h[10] / H_AVG[10]
            + 8.602976e-08 * h[11] / H_AVG[11]
            - 7.322244e-08 * h[6] / H_AVG[6]
            + 6.988967e-08 * h[86] / H_AVG[86]
            - 6.643259e-08 * h[32] / H_AVG[32]
            + 6.110212e-08 * h[17] / H_AVG[17]
            + 5.52297e-08 * h[38] / H_AVG[38]
            - 4.532616e-08 * h[42] / H_AVG[42]
            - 4.040637e-08 * h[54] / H_AVG[54]
            - 3.839841e-08 * h[29] / H_AVG[29]
            + 3.480869e-08 * h[118] / H_AVG[118]
            + 2.716913e-08 * h[48] / H_AVG[48]
            - 2.594007e-08 * h[45] / H_AVG[45]
            - 2.50487e-08 * h[39] / H_AVG[39]
            + 2.309275e-08 * h[27] / H_AVG[27]
            + 2.300822e-08 * h[63] / H_AVG[63]
            + 2.273822e-08 * h[30] / H_AVG[30]
            + 2.228543e-08 * h[43] / H_AVG[43]
            + 2.075135e-08 * h[121] / H_AVG[121]
            - 2.035583e-08 * h[80] / H_AVG[80]
            - 1.862749e-08 * h[126] / H_AVG[126]
            + 1.799852e-08 * h[60] / H_AVG[60]
            - 1.770115e-08 * h[61] / H_AVG[61]
            - 1.680483e-08 * h[102] / H_AVG[102]
            - 1.677771e-08 * h[107] / H_AVG[107]
            - 1.654562e-08 * h[7] / H_AVG[7]
            + 1.488965e-08 * h[92] / H_AVG[92]
            + 1.456295e-08 * h[35] / H_AVG[35]
            + 1.416849e-08 * h[15] / H_AVG[15]
            - 1.368384e-08 * h[59] / H_AVG[59]
            - 1.319798e-08 * h[101] / H_AVG[101]
            - 1.26306e-08 * h[28] / H_AVG[28]
            + 1.224022e-08 * h[103] / H_AVG[103]
            + 1.151692e-08 * h[69] / H_AVG[69]
            + 1.116375e-08 * h[67] / H_AVG[67]
            + 1.083457e-08 * h[1] / H_AVG[1]
            + 1.075343e-08 * h[100] / H_AVG[100]
            - 9.864209e-09 * h[110] / H_AVG[110]
            - 9.549092e-09 * h[3] / H_AVG[3]
            + 9.163714e-09 * h[37] / H_AVG[37]
            - 8.977354e-09 * h[50] / H_AVG[50]
            + 8.457347e-09 * h[44] / H_AVG[44]
            + 8.43537e-09 * h[36] / H_AVG[36]
            - 8.289534e-09 * h[12] / H_AVG[12]
            - 7.890711e-09 * h[114] / H_AVG[114]
            - 7.818043e-09 * h[2] / H_AVG[2]
            + 7.541397e-09 * h[58] / H_AVG[58]
            - 7.505035e-09 * h[112] / H_AVG[112]
            + 7.352906e-09 * h[76] / H_AVG[76]
            - 7.092568e-09 * h[13] / H_AVG[13]
            - 6.863311e-09 * h[19] / H_AVG[19]
            - 6.750786e-09 * h[111] / H_AVG[111]
            + 6.44222e-09 * h[74] / H_AVG[74]
            + 6.107757e-09 * h[108] / H_AVG[108]
            + 5.740402e-09 * h[5] / H_AVG[5]
            - 5.489181e-09 * h[8] / H_AVG[8]
            - 5.262167e-09 * h[91] / H_AVG[91]
            - 4.901961e-09 * h[4] / H_AVG[4]
            - 4.689676e-09 * h[117] / H_AVG[117]
            + 3.969242e-09 * h[53] / H_AVG[53]
            - 3.927078e-09 * h[66] / H_AVG[66]
            - 3.577187e-09 * h[33] / H_AVG[33]
            - 3.471631e-09 * h[65] / H_AVG[65]
            + 3.212482e-09 * h[82] / H_AVG[82]
            - 3.013936e-09 * h[31] / H_AVG[31]
            - 2.773106e-09 * h[71] / H_AVG[71]
            + 2.576224e-09 * h[94] / H_AVG[94]
            + 2.48383e-09 * h[9] / H_AVG[9]
            + 2.184481e-09 * h[26] / H_AVG[26]
            - 1.963194e-09 * h[14] / H_AVG[14]
            + 1.947018e-09 * h[57] / H_AVG[57]
            + 1.832788e-09 * h[49] / H_AVG[49]
            - 1.79258e-09 * h[22] / H_AVG[22]
            + 1.792431e-09 * h[77] / H_AVG[77]
            + 1.661906e-09 * h[122] / H_AVG[122]
            + 1.342267e-09 * h[72] / H_AVG[72]
            + 1.282092e-09 * h[51] / H_AVG[51]
            - 1.269888e-09 * h[23] / H_AVG[23]
            - 8.755363e-10 * h[89] / H_AVG[89]
            + 8.543109e-10 * h[125] / H_AVG[125]
            - 8.003542e-10 * h[75] / H_AVG[75]
            - 6.918361e-10 * h[87] / H_AVG[87]
            + 6.891873e-10 * h[93] / H_AVG[93]
            - 6.244459e-10 * h[40] / H_AVG[40]
            - 5.748598e-10 * h[20] / H_AVG[20]
            + 5.129456e-10 * h[119] / H_AVG[119]
            - 4.738923e-10 * h[62] / H_AVG[62]
            - 4.272976e-10 * h[95] / H_AVG[95]
            - 3.976045e-10 * h[96] / H_AVG[96]
            + 2.29413e-10 * h[116] / H_AVG[116]
            + 2.205148e-10 * h[56] / H_AVG[56]
            + 1.535587e-10 * h[85] / H_AVG[85]
            + 9.205492e-11 * h[109] / H_AVG[109]
            - 7.939971e-11 * h[98] / H_AVG[98]
            - 6.441051e-11 * h[88] / H_AVG[88]
            - 4.258623e-11 * h[46] / H_AVG[46]
            - 4.108399e-11 * h[47] / H_AVG[47]
            - 2.208082e-11 * h[64] / H_AVG[64]
            + 9.514518e-12 * h[127] / H_AVG[127]
            + 9.172834e-13 * h[41] / H_AVG[41]
        ),
        -1.495433 + T[9] * (   # class Tbl
            + 0.2503466 * h[16] / H_AVG[16]
            + 0.1181993 * h[24] / H_AVG[24]
            + 0.1073462 * h[18] / H_AVG[18]
            - 0.1037656 * h[104] / H_AVG[104]
            + 0.07423782 * h[97] / H_AVG[97]
            - 0.06943323 * h[115] / H_AVG[115]
            - 0.06633492 * h[124] / H_AVG[124]
            + 0.06258157 * h[52] / H_AVG[52]
            - 0.0625305 * h[68] / H_AVG[68]
            + 0.0460053 * h[78] / H_AVG[78]
            + 0.01851378 * h[120] / H_AVG[120]
            - 0.008732377 * h[83] / H_AVG[83]
            - 0.005952441 * h[81] / H_AVG[81]
            + 0.002596472 * h[25] / H_AVG[25]
            + 0.001761915 * h[123] / H_AVG[123]
            + 0.001190657 * h[70] / H_AVG[70]
            - 0.0003379554 * h[0] / H_AVG[0]
            + 5.783608e-05 * h[90] / H_AVG[90]
            + 5.643649e-05 * h[21] / H_AVG[21]
            - 1.660613e-05 * h[99] / H_AVG[99]
            + 7.929239e-07 * h[84] / H_AVG[84]
            + 1.870817e-07 * h[55] / H_AVG[55]
            + 1.254924e-07 * h[79] / H_AVG[79]
            - 1.199036e-07 * h[105] / H_AVG[105]
            - 9.730347e-08 * h[106] / H_AVG[106]
            + 7.910357e-08 * h[113] / H_AVG[113]
            - 7.511063e-08 * h[34] / H_AVG[34]
            + 6.739251e-08 * h[73] / H_AVG[73]
            + 6.25887e-08 * h[10] / H_AVG[10]
            + 5.972e-08 * h[11] / H_AVG[11]
            - 5.077206e-08 * h[6] / H_AVG[6]
            + 4.849267e-08 * h[86] / H_AVG[86]
            - 4.611743e-08 * h[32] / H_AVG[32]
            + 4.239769e-08 * h[17] / H_AVG[17]
            + 3.829719e-08 * h[38] / H_AVG[38]
            - 3.145266e-08 * h[42] / H_AVG[42]
            - 2.805651e-08 * h[54] / H_AVG[54]
            - 2.663697e-08 * h[29] / H_AVG[29]
            + 2.415786e-08 * h[118] / H_AVG[118]
            + 1.88369e-08 * h[48] / H_AVG[48]
            - 1.800349e-08 * h[45] / H_AVG[45]
            - 1.737055e-08 * h[39] / H_AVG[39]
            + 1.602611e-08 * h[27] / H_AVG[27]
            + 1.596252e-08 * h[63] / H_AVG[63]
            + 1.579065e-08 * h[30] / H_AVG[30]
            + 1.545539e-08 * h[43] / H_AVG[43]
            + 1.438879e-08 * h[121] / H_AVG[121]
            - 1.412032e-08 * h[80] / H_AVG[80]
            - 1.294099e-08 * h[126] / H_AVG[126]
            + 1.248311e-08 * h[60] / H_AVG[60]
            - 1.227191e-08 * h[61] / H_AVG[61]
            - 1.165956e-08 * h[102] / H_AVG[102]
            - 1.162671e-08 * h[107] / H_AVG[107]
            - 1.14791e-08 * h[7] / H_AVG[7]
            + 1.032609e-08 * h[92] / H_AVG[92]
            + 1.011145e-08 * h[35] / H_AVG[35]
            + 9.824953e-09 * h[15] / H_AVG[15]
            - 9.500285e-09 * h[59] / H_AVG[59]
            - 9.164696e-09 * h[101] / H_AVG[101]
            - 8.742131e-09 * h[28] / H_AVG[28]
            + 8.498089e-09 * h[103] / H_AVG[103]
            + 8.004979e-09 * h[69] / H_AVG[69]
            + 7.759902e-09 * h[67] / H_AVG[67]
            + 7.509039e-09 * h[1] / H_AVG[1]
            + 7.451851e-09 * h[100] / H_AVG[100]
            - 6.837982e-09 * h[110] / H_AVG[110]
            - 6.634172e-09 * h[3] / H_AVG[3]
            + 6.360274e-09 * h[37] / H_AVG[37]
            - 6.223779e-09 * h[50] / H_AVG[50]
            + 5.864641e-09 * h[44] / H_AVG[44]
            + 5.856183e-09 * h[36] / H_AVG[36]
            - 5.757991e-09 * h[12] / H_AVG[12]
            - 5.475932e-09 * h[114] / H_AVG[114]
            - 5.433222e-09 * h[2] / H_AVG[2]
            + 5.237942e-09 * h[58] / H_AVG[58]
            - 5.218377e-09 * h[112] / H_AVG[112]
            + 5.107832e-09 * h[76] / H_AVG[76]
            - 4.914759e-09 * h[13] / H_AVG[13]
            - 4.762483e-09 * h[19] / H_AVG[19]
            - 4.678448e-09 * h[111] / H_AVG[111]
            + 4.468656e-09 * h[74] / H_AVG[74]
            + 4.227814e-09 * h[108] / H_AVG[108]
            + 3.979369e-09 * h[5] / H_AVG[5]
            - 3.811932e-09 * h[8] / H_AVG[8]
            - 3.659603e-09 * h[91] / H_AVG[91]
            - 3.404865e-09 * h[4] / H_AVG[4]
            - 3.257807e-09 * h[117] / H_AVG[117]
            + 2.753534e-09 * h[53] / H_AVG[53]
            - 2.726176e-09 * h[66] / H_AVG[66]
            - 2.475308e-09 * h[33] / H_AVG[33]
            - 2.415988e-09 * h[65] / H_AVG[65]
            + 2.234608e-09 * h[82] / H_AVG[82]
            - 2.087903e-09 * h[31] / H_AVG[31]
            - 1.922343e-09 * h[71] / H_AVG[71]
            + 1.786209e-09 * h[94] / H_AVG[94]
            + 1.722922e-09 * h[9] / H_AVG[9]
            + 1.522283e-09 * h[26] / H_AVG[26]
            - 1.362955e-09 * h[14] / H_AVG[14]
            + 1.355966e-09 * h[57] / H_AVG[57]
            + 1.270264e-09 * h[49] / H_AVG[49]
            - 1.242952e-09 * h[22] / H_AVG[22]
            + 1.241231e-09 * h[77] / H_AVG[77]
            + 1.157165e-09 * h[122] / H_AVG[122]
            + 9.301246e-10 * h[72] / H_AVG[72]
            + 8.935788e-10 * h[51] / H_AVG[51]
            - 8.843624e-10 * h[23] / H_AVG[23]
            - 6.036442e-10 * h[89] / H_AVG[89]
            + 5.913174e-10 * h[125] / H_AVG[125]
            - 5.541872e-10 * h[75] / H_AVG[75]
            + 4.783773e-10 * h[93] / H_AVG[93]
            - 4.775981e-10 * h[87] / H_AVG[87]
            - 4.330923e-10 * h[40] / H_AVG[40]
            - 3.988813e-10 * h[20] / H_AVG[20]
            + 3.570381e-10 * h[119] / H_AVG[119]
            - 3.279485e-10 * h[62] / H_AVG[62]
            - 2.963429e-10 * h[95] / H_AVG[95]
            - 2.751429e-10 * h[96] / H_AVG[96]
            + 1.588778e-10 * h[116] / H_AVG[116]
            + 1.536101e-10 * h[56] / H_AVG[56]
            + 1.072943e-10 * h[85] / H_AVG[85]
            + 6.363278e-11 * h[109] / H_AVG[109]
            - 5.488105e-11 * h[98] / H_AVG[98]
            - 4.439948e-11 * h[88] / H_AVG[88]
            - 2.97577e-11 * h[46] / H_AVG[46]
            - 2.843388e-11 * h[47] / H_AVG[47]
            - 1.462335e-11 * h[64] / H_AVG[64]
            + 6.860464e-12 * h[127] / H_AVG[127]
            + 6.660069e-13 * h[41] / H_AVG[41]
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
