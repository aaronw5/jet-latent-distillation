"""Particle Transformer (ParT) jet tagger, JetClass, input features 'full': tuned on the network's predictions (from 100 if-statements per neuron; the 17 neurons of the first neuron cut, terms not pruned), as if-statements.

Input:  the particles of a jet (up to 128), hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
        energy: each particle's energy [GeV]; jet_pt, jet_eta, jet_energy: the jet's pT [GeV], pseudorapidity, energy [GeV].
        charge, ptype (1 charged hadron, 2 neutral hadron, 3 photon, 4 electron, 5 muon), d0, d0err, dz, dzerr:
        per particle, as JetClass stores them (impact parameters in mm).
Output: the class (QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl), the 10 logits and the 10 probabilities.

1. quantities():  physics quantities of the jet.
2. class_token(): the 128 numbers the network's last layer reads (its class token after the last LayerNorm), each
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
    z = 0.005192869
    return z


def neuron_1(Q):
    z = -5.32527e-06
    return z


def neuron_2(Q):
    z = 5.44731e-06
    return z


def neuron_3(Q):
    z = 8.399862e-06
    return z


def neuron_4(Q):
    z = 2.792155e-06
    return z


def neuron_5(Q):
    z = -2.156869e-06
    return z


def neuron_6(Q):
    z = 1.770101e-05
    return z


def neuron_7(Q):
    z = 6.430821e-06
    return z


def neuron_8(Q):
    z = 2.926455e-06
    return z


def neuron_9(Q):
    z = -2.443661e-06
    return z


def neuron_10(Q):
    z = -1.570106e-05
    return z


def neuron_11(Q):
    z = -1.887716e-05
    return z


def neuron_12(Q):
    z = 5.099316e-06
    return z


def neuron_13(Q):
    z = 4.422903e-06
    return z


def neuron_14(Q):
    z = 2.845306e-06
    return z


def neuron_15(Q):
    z = -6.459517e-06
    return z


def neuron_16(Q):
    z = 4.590858
    if Q.lep_ptrel < 27.3236:
        z += 0.03028987 * Q.lep_ptrel + 1.647523
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.09058655 * Q.lep_ptrel
    if Q.lep_ptrel >= 43.20788:
        z += -0.03376921 * Q.lep_ptrel + 5.373149
    if Q.n_s3d_above_3 < 10.0:
        z += 0.1292211 * Q.n_s3d_above_3 - 1.292211
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.01622097 * Q.n_pairs_kt_above_3 + 0.8175823
    if 28.0 <= Q.n_pairs_kt_above_3 < 41.0:
        z += -0.02795346 * Q.n_pairs_kt_above_3 + 1.146092
    if Q.sip_3d_2 < 226.3008:
        z += -0.002484315 * Q.sip_3d_2 + 0.4486167
    if 226.3008 <= Q.sip_3d_2 < 447.0873:
        z += 0.0005144596 * Q.sip_3d_2 - 0.2300084
    if Q.lund_max_lndelta >= -1.062087:
        z += 0.09206115 * Q.lund_max_lndelta + 0.09777697
    if Q.z_displaced5 < 0.06245248:
        z += 11.79594 * Q.z_displaced5 - 1.570161
    if 0.06245248 <= Q.z_displaced5 < 0.2801368:
        z += 3.828825 * Q.z_displaced5 - 1.072595
    if Q.mass_neutral < 19.57354:
        z += -0.004617158 * Q.mass_neutral + 0.405923
    if 19.57354 <= Q.mass_neutral < 87.9162:
        z += 0.0003196993 * Q.mass_neutral + 0.3092912
    if Q.mass_neutral >= 87.9162:
        z += 0.004936857 * Q.mass_neutral - 0.09663177
    if Q.mass_top40 < 92.68149:
        z += 0.001863539 * Q.mass_top40 - 0.1727155
    if Q.mass_top40 >= 122.7145:
        z += 0.005245949 * Q.mass_top40 - 0.6437541
    if Q.e3_b2 < 0.0002536827:
        z += -696.7812 * Q.e3_b2 + 0.1767613
    if 119.4443 <= Q.sd_mass < 154.5947:
        z += 0.01226765 * Q.sd_mass - 1.465302
    if 154.5947 <= Q.sd_mass < 175.9333:
        z += -0.01136133 * Q.sd_mass + 2.187613
    if Q.sd_mass >= 175.9333:
        z += -0.006548089 * Q.sd_mass + 1.340803
    if Q.max_abs_d0 < 5.8125:
        z += 0.001245481 * Q.max_abs_d0 - 0.00723936
    if Q.sip_3d_3 < 16.23838:
        z += 0.04303276 * Q.sip_3d_3 - 0.7051128
    if 16.23838 <= Q.sip_3d_3 < 33.08364:
        z += 0.02750932 * Q.sip_3d_3 - 0.4530373
    if 33.08364 <= Q.sip_3d_3 < 577.991:
        z += -0.0008388053 * Q.sip_3d_3 + 0.4848219
    z += -7.457049 * Q.z_charged_had
    if 125.4927 <= Q.mass_top50 < 161.1264:
        z += 0.00486069 * Q.mass_top50 - 0.6099809
    if Q.mass_top50 >= 161.1264:
        z += -0.002803296 * Q.mass_top50 + 0.6248896
    if Q.n_charged_had < 29.0:
        z += -0.03363072 * Q.n_charged_had + 0.975291
    z += 0.01439112 * Q.n_photon
    if Q.mass_top5 >= 62.84493:
        z += 0.004298654 * Q.mass_top5 - 0.2701486
    if Q.n_electron >= 1.0:
        z += -0.1063336 * Q.n_electron + 0.1063336
    if Q.n_lund_kt_above_5 >= 1.0:
        z += -0.009116549 * Q.n_lund_kt_above_5 + 0.009116549
    z += 0.07856911 * Q.n_sd0_above_2
    if Q.mass_displaced3 < 0.2490748:
        z += -0.3689366 * Q.mass_displaced3 + 0.00680117
    if 0.2490748 <= Q.mass_displaced3 < 1.777286:
        z += 0.2456995 * Q.mass_displaced3 - 0.1462892
    if 1.777286 <= Q.mass_displaced3 < 4.41005:
        z += -0.08625414 * Q.mass_displaced3 + 0.4436873
    if 4.41005 <= Q.mass_displaced3 < 13.03663:
        z += -0.007338041 * Q.mass_displaced3 + 0.09566334
    if Q.z_dr_0p05_0p1 < 0.01556887:
        z += -7.690272 * Q.z_dr_0p05_0p1 + 0.1197289
    if 0.01245833 <= Q.lep_z < 0.221436:
        z += -5.664444 * Q.lep_z + 0.07056952
    if Q.lep_z >= 0.221436:
        z += -7.692719 * Q.lep_z + 0.5197026
    z += -0.004591907 * Q.mass_charged
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.01517319 * Q.n_pairs_kt_above_1 + 1.213855
    if Q.tau4 < 0.02720087:
        z += 8.517589 * Q.tau4 - 0.2316858
    if Q.n_pairs_kt_above_10 >= 17.0:
        z += 0.005207138 * Q.n_pairs_kt_above_10 - 0.08852135
    z += -6.785293 * Q.z_neutral
    if Q.n_dr_0p2_0p4 < 7.0:
        z += -0.0407637 * Q.n_dr_0p2_0p4 + 0.2853459
    if Q.lne_4 < 3.452465:
        z += -0.07108796 * Q.lne_4 + 0.2454287
    if Q.n_pt_above_1 < 33.0:
        z += 0.01317459 * Q.n_pt_above_1 - 0.4347615
    if Q.M3 >= 0.02540381:
        z += 10.32671 * Q.M3 - 0.2623378
    if Q.psi_0p3 >= 0.9828705:
        z += -8.50453 * Q.psi_0p3 + 8.358852
    if Q.M2 < 0.09740996:
        z += -9.142333 * Q.M2 + 0.8905543
    if Q.z_top15_slots >= 0.8008865:
        z += 0.6444468 * Q.z_top15_slots - 0.5161288
    if Q.C2 >= 0.1323775:
        z += 1.59813 * Q.C2 - 0.2115565
    if Q.tau3 >= 0.06348273:
        z += -3.47651 * Q.tau3 + 0.2206983
    if Q.sj3_pair_mass_max >= 83.63398:
        z += 0.006655771 * Q.sj3_pair_mass_max - 0.5566486
    if Q.jet_charge_k05 < -0.3599791:
        z += -0.2356603 * Q.jet_charge_k05 - 0.08483279
    if Q.jet_charge_k05 >= 0.5864519:
        z += 0.3445777 * Q.jet_charge_k05 - 0.2020782
    if Q.mass_2photon >= 22.18431:
        z += -0.004900238 * Q.mass_2photon + 0.1087084
    if Q.min_pair_mass < 1.763283:
        z += 0.1067473 * Q.min_pair_mass - 0.1882257
    if Q.n_sdz_above_2 < 5.0:
        z += 0.07478184 * Q.n_sdz_above_2 - 0.3739092
    if Q.z_dr_0p4_up < 0.0265067:
        z += 1.565817 * Q.z_dr_0p4_up - 0.04150463
    if Q.C2_b05 < 0.2413007:
        z += -3.328282 * Q.C2_b05 + 0.803117
    if Q.D2 < 0.7107791:
        z += 0.1043298 * Q.D2 - 0.07415546
    if Q.max_pair_mass >= 55.24488:
        z += -0.01146632 * Q.max_pair_mass + 0.6334557
    if Q.N2_b05 < 0.302405:
        z += 1.204226 * Q.N2_b05 - 0.3641638
    if Q.tdz_1 < -0.1170754:
        z += -0.1334512 * Q.tdz_1 - 0.01562386
    if Q.sj3_mass2 < 1.824785:
        z += -0.1016171 * Q.sj3_mass2 + 0.1854293
    if Q.max_abs_dz < 0.6455078:
        z += 0.3252168 * Q.max_abs_dz + 0.01076705
    if 0.6455078 <= Q.max_abs_dz < 2.957031:
        z += -0.09547688 * Q.max_abs_dz + 0.2823281
    if Q.mass_neutral < 87.9162 and Q.M2_b2 < 0.06762785:
        z += -0.144166 * (87.9162 - Q.mass_neutral) * (0.06762785 - Q.M2_b2)
    if Q.mass_neutral < 87.9162 and Q.mass_2charged > 17.49066:
        z += 7.298701e-05 * (87.9162 - Q.mass_neutral) * (Q.mass_2charged - 17.49066)
    if Q.z_displaced5 < 0.2801368 and Q.sip_3d_3 < 175.9957:
        z += 0.01236639 * (0.2801368 - Q.z_displaced5) * (175.9957 - Q.sip_3d_3)
    if Q.mass_neutral < 87.9162 and Q.n_lund > 5.0:
        z += 0.0003092221 * (87.9162 - Q.mass_neutral) * (Q.n_lund - 5.0)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.lep_dr < 0.2338497:
        z += -0.01178623 * (28.0 - Q.n_pairs_kt_above_3) * (0.2338497 - Q.lep_dr)
    if Q.lund_max_lndelta > -1.062087 and Q.n_charged_had < 15.0:
        z += 0.08187842 * (Q.lund_max_lndelta - -1.062087) * (15.0 - Q.n_charged_had)
    if Q.n_s3d_above_3 < 10.0 and Q.n_dr_0p4_up < 15.0:
        z += 0.0003977008 * (10.0 - Q.n_s3d_above_3) * (15.0 - Q.n_dr_0p4_up)
    if Q.n_charged_had < 29.0 and Q.n_lepton > 1.0:
        z += 0.0008788147 * (29.0 - Q.n_charged_had) * (Q.n_lepton - 1.0)
    if Q.n_charged_had < 29.0 and Q.ecf_g42 > 2.701529e-06:
        z += -183.8409 * (29.0 - Q.n_charged_had) * (Q.ecf_g42 - 2.701529e-06)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.M3_b05 > 0.06007167:
        z += 0.229915 * (28.0 - Q.n_pairs_kt_above_3) * (Q.M3_b05 - 0.06007167)
    if Q.max_abs_d0 < 5.8125 and Q.sip_3d_3 < 49.03695:
        z += 0.001331751 * (5.8125 - Q.max_abs_d0) * (49.03695 - Q.sip_3d_3)
    if Q.n_s3d_above_3 < 10.0 and Q.D2_b2 < 7.782712:
        z += -0.007147237 * (10.0 - Q.n_s3d_above_3) * (7.782712 - Q.D2_b2)
    if Q.mass_neutral < 87.9162 and Q.dr_max_012 < 0.02981375:
        z += -0.0677399 * (87.9162 - Q.mass_neutral) * (0.02981375 - Q.dr_max_012)
    if Q.mass_neutral < 87.9162 and Q.n_dr_0p1_0p2 < 6.0:
        z += 0.000727825 * (87.9162 - Q.mass_neutral) * (6.0 - Q.n_dr_0p1_0p2)
    if Q.z_displaced5 < 0.2801368 and Q.sip_3d_3 < 33.08364:
        z += 0.1006487 * (0.2801368 - Q.z_displaced5) * (33.08364 - Q.sip_3d_3)
    if Q.lund_max_lndelta > -1.062087 and Q.sip_3d_3 < 4.606241:
        z += -0.2717997 * (Q.lund_max_lndelta - -1.062087) * (4.606241 - Q.sip_3d_3)
    if Q.lep_z > 0.221436 and Q.tau32 < 0.7544983:
        z += 0.9216949 * (Q.lep_z - 0.221436) * (0.7544983 - Q.tau32)
    if Q.mass_displaced3 < 13.03663 and Q.iselectron_1 > 0.0:
        z += -0.004270532 * (13.03663 - Q.mass_displaced3) * (Q.iselectron_1 - 0.0)
    if Q.lund_max_lndelta > -1.062087 and Q.sj4_pair_mass_min < 34.83233:
        z += 0.01100257 * (Q.lund_max_lndelta - -1.062087) * (34.83233 - Q.sj4_pair_mass_min)
    if Q.n_lund_kt_above_5 > 1.0 and Q.sip_3d_3 > 49.03695:
        z += -1.594654e-05 * (Q.n_lund_kt_above_5 - 1.0) * (Q.sip_3d_3 - 49.03695)
    if Q.z_dr_0p05_0p1 < 0.01556887 and Q.tau32_b2 < 0.3377343:
        z += 6.264152 * (0.01556887 - Q.z_dr_0p05_0p1) * (0.3377343 - Q.tau32_b2)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.D2 < 1.516502:
        z += -0.0722938 * (7.0 - Q.n_dr_0p2_0p4) * (1.516502 - Q.D2)
    if Q.tau3 > 0.06348273 and Q.phi_84 < 0.0:
        z += 1.55611 * (Q.tau3 - 0.06348273) * (0.0 - Q.phi_84)
    if Q.sip_3d_2 < 447.0873 and Q.d0err_12 > 0.0236969:
        z += 0.005393974 * (447.0873 - Q.sip_3d_2) * (Q.d0err_12 - 0.0236969)
    if Q.mass_top50 > 125.4927 and Q.tdz_68 < 0.0:
        z += -0.0006890391 * (Q.mass_top50 - 125.4927) * (0.0 - Q.tdz_68)
    if Q.z_dr_0p05_0p1 < 0.01556887 and Q.eta_36 < 0.1083984:
        z += 7.316391 * (0.01556887 - Q.z_dr_0p05_0p1) * (0.1083984 - Q.eta_36)
    if Q.z_displaced5 < 0.06245248 and Q.tau54 < 0.8786609:
        z += -13.93715 * (0.06245248 - Q.z_displaced5) * (0.8786609 - Q.tau54)
    return z


def neuron_17(Q):
    z = -1.035355e-05
    return z


def neuron_18(Q):
    z = 0.6100723
    if Q.lep_z < 0.004135872:
        z += 186.5629 * Q.lep_z - 1.448964
    if 0.004135872 <= Q.lep_z < 0.221436:
        z += 3.117178 * Q.lep_z - 0.6902556
    if 110.2019 <= Q.mass < 164.4374:
        z += -0.007971954 * Q.mass + 0.8785244
    if 164.4374 <= Q.mass < 182.8592:
        z += 0.02565851 * Q.mass - 4.651583
    if Q.mass >= 182.8592:
        z += 0.006451899 * Q.mass - 1.139477
    if Q.mass_top20 < 114.4658:
        z += 0.005818386 * Q.mass_top20 - 0.6660062
    if Q.z_displaced3 < 0.03624058:
        z += 10.45365 * Q.z_displaced3 - 0.6014194
    if 0.03624058 <= Q.z_displaced3 < 0.06416437:
        z += 7.970729 * Q.z_displaced3 - 0.5114368
    if Q.n_s3d_above_10 < 6.0:
        z += 0.03095341 * Q.n_s3d_above_10 - 0.1857205
    if Q.lep_ptrel < 27.3236:
        z += 0.005026712 * Q.lep_ptrel - 0.1373479
    if Q.lund3_lndelta >= -2.817283:
        z += 0.004154794 * Q.lund3_lndelta + 0.01170523
    if Q.mass_displaced3 < 2.409613:
        z += 0.003912901 * Q.mass_displaced3 - 0.02631805
    if 2.409613 <= Q.mass_displaced3 < 9.090532:
        z += 0.002528016 * Q.mass_displaced3 - 0.02298101
    if Q.z_displaced5 < 0.006242101:
        z += -5.868313 * Q.z_displaced5 + 0.3664907
    if 0.006242101 <= Q.z_displaced5 < 0.06245248:
        z += -6.623162 * Q.z_displaced5 + 0.3712025
    if Q.z_displaced5 >= 0.06245248:
        z += -0.7548494 * Q.z_displaced5 + 0.004711846
    if Q.sip_3d_1 < 3.373037:
        z += 0.3017429 * Q.sip_3d_1 - 1.01779
    if Q.n_pairs_kt_above_1 < 146.0:
        z += -0.0003436658 * Q.n_pairs_kt_above_1 + 0.1257817
    if 146.0 <= Q.n_pairs_kt_above_1 < 366.0:
        z += -0.000743481 * Q.n_pairs_kt_above_1 + 0.1841547
    if Q.n_pairs_kt_above_1 >= 366.0:
        z += -0.0003998152 * Q.n_pairs_kt_above_1 + 0.05837302
    if Q.sum_z_dr < 0.07290954:
        z += 8.031436 * Q.sum_z_dr - 0.5855683
    if 111.701 <= Q.sd_mass < 123.2919:
        z += 0.01142453 * Q.sd_mass - 1.276131
    if 123.2919 <= Q.sd_mass < 154.5947:
        z += -0.02144928 * Q.sd_mass + 2.776943
    if Q.sd_mass >= 154.5947:
        z += 0.008382566 * Q.sd_mass - 1.834902
    if Q.sj2_mass2 < 5.311506:
        z += -0.08834504 * Q.sj2_mass2 + 0.4692452
    if Q.sj2_mass2 >= 42.12131:
        z += -0.001081553 * Q.sj2_mass2 + 0.04555643
    if Q.tau3 >= 0.05398263:
        z += 0.3546041 * Q.tau3 - 0.01914246
    if Q.max_abs_d0 < 0.6289062:
        z += -0.6841984 * Q.max_abs_d0 + 0.4302967
    if Q.n_s3d_above_3 < 3.0:
        z += 0.5782245 * Q.n_s3d_above_3 - 1.734674
    if Q.sj3_pair_mass_max >= 128.6605:
        z += -0.01214914 * Q.sj3_pair_mass_max + 1.563116
    if Q.n_sd0_above_3 < 9.0:
        z += -0.07602043 * Q.n_sd0_above_3 + 0.6841839
    if Q.n_sdz_above_5 < 3.0:
        z += -0.06779463 * Q.n_sdz_above_5 + 0.2033839
    if Q.n_dr_0p4_up < 4.0:
        z += -0.0399119 * Q.n_dr_0p4_up + 0.1596476
    if Q.sj3_dr23 < 0.3084865:
        z += 0.4111205 * Q.sj3_dr23 - 0.1268251
    if Q.tau43 >= 0.6808537:
        z += -0.9211739 * Q.tau43 + 0.6271847
    if Q.lep_iso < 0.4381892:
        z += -0.4846648 * Q.lep_iso + 0.4306213
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += -0.1065789 * Q.lep_iso + 0.2649481
    if 1.362094 <= Q.lep_iso < 14.38858:
        z += -0.009194926 * Q.lep_iso + 0.132302
    if Q.N2 < 0.3829918:
        z += 2.723139 * Q.N2 - 1.04294
    if Q.e4 < 2.097213e-06:
        z += -160741.0 * Q.e4 + 0.337108
    if Q.n_lepton < 2.0:
        z += -0.1466413 * Q.n_lepton + 0.2932826
    if Q.max_abs_dz < 9.046875:
        z += 0.008297934 * Q.max_abs_dz - 0.07507037
    if Q.n_charged_had >= 27.0:
        z += -0.05196363 * Q.n_charged_had + 1.403018
    if Q.N2_b05 >= 0.4265629:
        z += 1.50334 * Q.N2_b05 - 0.641269
    if Q.psi_0p2 < 0.2598003:
        z += -0.6285031 * Q.psi_0p2 + 0.1632853
    if Q.lund2_lndelta >= -1.124734:
        z += 0.1315864 * Q.lund2_lndelta + 0.1479997
    if Q.lund_max_lndelta >= -0.6897565:
        z += -0.7859826 * Q.lund_max_lndelta - 0.5421366
    if Q.n_pairs_kt_above_3 < 111.0:
        z += -0.001944484 * Q.n_pairs_kt_above_3 + 0.2158377
    if Q.tau32 >= 0.5871757:
        z += -0.5787855 * Q.tau32 + 0.3398487
    if Q.mass_top5 >= 76.4159:
        z += -0.004768336 * Q.mass_top5 + 0.3643767
    if Q.sj3_pair_mass_min >= 44.1643:
        z += -0.003424852 * Q.sj3_pair_mass_min + 0.1512562
    if Q.mass_charged >= 113.5019:
        z += 0.005318504 * Q.mass_charged - 0.6036605
    if Q.z_charged < 0.3887012:
        z += -1.396429 * Q.z_charged + 0.5427935
    if Q.pt1_over_pt0 < 0.3206143:
        z += -1.420723 * Q.pt1_over_pt0 + 0.4555041
    if Q.sj3_pairmax_over_m >= 0.897453:
        z += -1.000362 * Q.sj3_pairmax_over_m + 0.8977782
    if Q.mass_top40 < 132.5189:
        z += 0.002891723 * Q.mass_top40 - 0.2102245
    if 132.5189 <= Q.mass_top40 < 155.8928:
        z += -0.007400721 * Q.mass_top40 + 1.153719
    if Q.n_neutral >= 12.0:
        z += -0.02802711 * Q.n_neutral + 0.3363253
    z += 0.03747851 * Q.n_for_90pct
    if Q.mass_neutral >= 14.92637:
        z += 0.004958668 * Q.mass_neutral - 0.07401491
    if Q.lund_max_lnkt >= 3.22013:
        z += -0.1257593 * Q.lund_max_lnkt + 0.4049613
    if Q.sum_z_dr2_top2 < 0.01091199:
        z += 18.62659 * Q.sum_z_dr2_top2 - 0.2032532
    if Q.N3_b2 < 0.4545826:
        z += 0.1930309 * Q.N3_b2 - 0.08774851
    if Q.e3_b2 < 0.0001729927:
        z += -593.6683 * Q.e3_b2 - 0.3131086
    if 0.0001729927 <= Q.e3_b2 < 0.0007909605:
        z += 672.8649 * Q.e3_b2 - 0.5322095
    if Q.eta_1 < -0.2019043:
        z += -0.0454757 * Q.eta_1 - 0.00918174
    if Q.n_lund >= 13.0:
        z += -0.02152615 * Q.n_lund + 0.27984
    if Q.sip_3d_3 < 49.03695:
        z += -0.004712712 * Q.sip_3d_3 + 0.231097
    if Q.sip_3d_2 < 226.3008:
        z += 0.0009765466 * Q.sip_3d_2 - 0.2209933
    if Q.n_sd0_above_2 < 6.0:
        z += 0.0706825 * Q.n_sd0_above_2 - 0.424095
    if Q.n_charged_pt_above_1 < 27.0:
        z += -0.03452309 * Q.n_charged_pt_above_1 + 0.9321235
    if Q.lep_z < 0.221436 and Q.N2 < 0.3829918:
        z += 11.83874 * (0.221436 - Q.lep_z) * (0.3829918 - Q.N2)
    if Q.z_displaced3 < 0.03624058 and Q.z_charged_had > 0.1891627:
        z += -6.348819 * (0.03624058 - Q.z_displaced3) * (Q.z_charged_had - 0.1891627)
    if Q.lep_z < 0.221436 and Q.z_displaced5 < 0.1046203:
        z += -31.57223 * (0.221436 - Q.lep_z) * (0.1046203 - Q.z_displaced5)
    if Q.n_s3d_above_10 < 6.0 and Q.lne_7 > 2.313463:
        z += 0.01187825 * (6.0 - Q.n_s3d_above_10) * (Q.lne_7 - 2.313463)
    if Q.mass > 110.2019 and Q.n_charged_pt_above_10 < 8.0:
        z += 7.081359e-06 * (Q.mass - 110.2019) * (8.0 - Q.n_charged_pt_above_10)
    if Q.n_s3d_above_10 < 6.0 and Q.z_neutral_had < 0.212136:
        z += 0.1221051 * (6.0 - Q.n_s3d_above_10) * (0.212136 - Q.z_neutral_had)
    if Q.n_s3d_above_3 < 3.0 and Q.sip_3d_2 < 117.0891:
        z += 0.00455201 * (3.0 - Q.n_s3d_above_3) * (117.0891 - Q.sip_3d_2)
    if Q.z_displaced5 < 0.06245248 and Q.sip_3d_2 < 4.636903:
        z += -2.547604 * (0.06245248 - Q.z_displaced5) * (4.636903 - Q.sip_3d_2)
    if Q.z_displaced5 > 0.006242101 and Q.sip_3d_2 < 447.0873:
        z += 0.0002057768 * (Q.z_displaced5 - 0.006242101) * (447.0873 - Q.sip_3d_2)
    if Q.mass_displaced3 < 9.090532 and Q.n_charged_pt_above_1 < 29.0:
        z += 0.002025543 * (9.090532 - Q.mass_displaced3) * (29.0 - Q.n_charged_pt_above_1)
    if Q.z_displaced5 > 0.006242101 and Q.n_dr_0p2_0p4 < 29.0:
        z += 0.001228687 * (Q.z_displaced5 - 0.006242101) * (29.0 - Q.n_dr_0p2_0p4)
    if Q.z_displaced5 < 0.06245248 and Q.n_pt_above_5 > 10.0:
        z += 0.2941608 * (0.06245248 - Q.z_displaced5) * (Q.n_pt_above_5 - 10.0)
    if Q.n_s3d_above_10 < 6.0 and Q.lund_max_lndelta > -1.670992:
        z += 0.06291346 * (6.0 - Q.n_s3d_above_10) * (Q.lund_max_lndelta - -1.670992)
    if Q.lep_z < 0.221436 and Q.e4_b05 > 4.363663e-05:
        z += -900.1859 * (0.221436 - Q.lep_z) * (Q.e4_b05 - 4.363663e-05)
    if Q.z_displaced5 > 0.006242101 and Q.dr12 < 0.2116473:
        z += 0.2512246 * (Q.z_displaced5 - 0.006242101) * (0.2116473 - Q.dr12)
    if Q.lund3_lndelta > -2.817283 and Q.tau54 > 0.7608692:
        z += -0.7418237 * (Q.lund3_lndelta - -2.817283) * (Q.tau54 - 0.7608692)
    if Q.mass_displaced3 < 9.090532 and Q.lep_dr < 0.114659:
        z += -0.145599 * (9.090532 - Q.mass_displaced3) * (0.114659 - Q.lep_dr)
    if Q.lep_z < 0.004135872 and Q.jet_abs_eta > 0.8317778:
        z += 76.51419 * (0.004135872 - Q.lep_z) * (Q.jet_abs_eta - 0.8317778)
    if Q.lep_z < 0.221436 and Q.jet_charge_k05 > 0.1901233:
        z += -1.515977 * (0.221436 - Q.lep_z) * (Q.jet_charge_k05 - 0.1901233)
    if Q.sj3_pair_mass_max > 128.6605 and Q.eccentricity > 0.9132117:
        z += -0.3749743 * (Q.sj3_pair_mass_max - 128.6605) * (Q.eccentricity - 0.9132117)
    if Q.mass > 182.8592 and Q.planar_flow < 0.4623202:
        z += 0.05971354 * (Q.mass - 182.8592) * (0.4623202 - Q.planar_flow)
    if Q.e4 < 2.097213e-06 and Q.lep_iso < 1.362094:
        z += -17012.89 * (2.097213e-06 - Q.e4) * (1.362094 - Q.lep_iso)
    if Q.sj3_pair_mass_max > 128.6605 and Q.lund1_lndelta < -0.3624079:
        z += -0.02342284 * (Q.sj3_pair_mass_max - 128.6605) * (-0.3624079 - Q.lund1_lndelta)
    if Q.N2 < 0.3829918 and Q.lnptrel_45 < -5.306837:
        z += 0.1344724 * (0.3829918 - Q.N2) * (-5.306837 - Q.lnptrel_45)
    if Q.lep_ptrel < 27.3236 and Q.N2_b2 > 0.0395449:
        z += -0.04603999 * (27.3236 - Q.lep_ptrel) * (Q.N2_b2 - 0.0395449)
    if Q.lep_ptrel < 27.3236 and Q.eta_9 > 0.09516602:
        z += 0.00214019 * (27.3236 - Q.lep_ptrel) * (Q.eta_9 - 0.09516602)
    if Q.lep_z < 0.221436 and Q.pt1_over_pt0 < 0.380649:
        z += -3.54194 * (0.221436 - Q.lep_z) * (0.380649 - Q.pt1_over_pt0)
    if Q.z_charged < 0.3887012 and Q.mass_2photon > 1.081989:
        z += 0.009075574 * (0.3887012 - Q.z_charged) * (Q.mass_2photon - 1.081989)
    if Q.n_neutral > 12.0 and Q.eta_28 > -0.1078491:
        z += 0.001480073 * (Q.n_neutral - 12.0) * (Q.eta_28 - -0.1078491)
    if Q.sj3_pairmax_over_m > 0.897453 and Q.iselectron_29 > 0.0:
        z += -0.4110164 * (Q.sj3_pairmax_over_m - 0.897453) * (Q.iselectron_29 - 0.0)
    if Q.tau32 > 0.5871757 and Q.n_pt_above_50 < 4.0:
        z += -0.1571643 * (Q.tau32 - 0.5871757) * (4.0 - Q.n_pt_above_50)
    return z


def neuron_19(Q):
    z = -1.407883e-06
    return z


def neuron_20(Q):
    z = 8.253655e-07
    return z


def neuron_21(Q):
    z = -0.001686459
    return z


def neuron_22(Q):
    z = -1.044882e-06
    return z


def neuron_23(Q):
    z = -6.979496e-07
    return z


def neuron_24(Q):
    z = -0.5298461
    if Q.lep_z < 0.221436:
        z += 5.790629 * Q.lep_z - 1.403702
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += 1.027298 * Q.lep_z - 0.3489294
    if Q.mass_top30 < 74.51927:
        z += 0.00938959 * Q.mass_top30 - 0.2567081
    if 74.51927 <= Q.mass_top30 < 134.2224:
        z += -0.007420005 * Q.mass_top30 + 0.9959308
    if Q.mass < 79.47361:
        z += 0.03022741 * Q.mass - 2.775324
    if 79.47361 <= Q.mass < 90.08945:
        z += 0.001897343 * Q.mass - 0.5238311
    if 90.08945 <= Q.mass < 114.0172:
        z += -0.01136079 * Q.mass + 0.6705865
    if 114.0172 <= Q.mass < 117.4867:
        z += 0.01874817 * Q.mass - 2.762354
    if 117.4867 <= Q.mass < 149.0507:
        z += 0.01151271 * Q.mass - 1.912284
    if 149.0507 <= Q.mass < 164.4374:
        z += 0.01275812 * Q.mass - 2.097913
    z += 0.2107976 * Q.pair_max_lnkt
    if Q.z_displaced3 < 0.1061578:
        z += -3.390066 * Q.z_displaced3 + 0.3598819
    if Q.e3_b2 < 0.0004126585:
        z += 1970.138 * Q.e3_b2 - 0.8129944
    if Q.lep_ptrel < 43.20788:
        z += -0.004164935 * Q.lep_ptrel + 0.179958
    if Q.n_s3d_above_10 >= 1.0:
        z += -0.0457279 * Q.n_s3d_above_10 + 0.0457279
    if Q.mass_displaced3 < 26.78691:
        z += -0.01490282 * Q.mass_displaced3 + 0.3992004
    if Q.mass_displaced3 >= 39.09615:
        z += -0.02987495 * Q.mass_displaced3 + 1.167995
    if Q.tau32 < 0.5068038:
        z += -1.01256 * Q.tau32 + 0.513169
    if Q.lep_iso < 0.4381892:
        z += -1.899456 * Q.lep_iso + 1.283154
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += -0.4879648 * Q.lep_iso + 0.6646541
    if Q.n_lepton < 1.0:
        z += 0.7789581 * Q.n_lepton - 0.7789581
    if Q.n_s3d_above_3 < 2.0:
        z += -0.2446853 * Q.n_s3d_above_3 + 0.4893707
    if 2.0 <= Q.n_s3d_above_3 < 5.0:
        z += -0.05901354 * Q.n_s3d_above_3 + 0.1180271
    if Q.n_s3d_above_3 >= 5.0:
        z += 0.004647035 * Q.n_s3d_above_3 - 0.2002758
    z += 0.0002583096 * Q.sum_e
    if Q.n_sd0_above_10 < 2.0:
        z += -0.00365489 * Q.n_sd0_above_10 + 0.00730978
    if Q.n_sd0_above_10 >= 6.0:
        z += 0.03607435 * Q.n_sd0_above_10 - 0.2164461
    if Q.sj2_dr < 0.2083105:
        z += -2.703424 * Q.sj2_dr + 0.244639
    if 0.2083105 <= Q.sj2_dr < 0.2403736:
        z += -3.278382 * Q.sj2_dr + 0.3644087
    if 0.2403736 <= Q.sj2_dr < 0.3923158:
        z += 3.25292 * Q.sj2_dr - 1.205544
    if 0.3923158 <= Q.sj2_dr < 0.5944498:
        z += -0.3494122 * Q.sj2_dr + 0.207708
    if Q.n_sd0_above_3 < 3.0:
        z += 0.1487718 * Q.n_sd0_above_3 - 0.8926308
    if 3.0 <= Q.n_sd0_above_3 < 6.0:
        z += 0.2120054 * Q.n_sd0_above_3 - 1.082332
    if Q.n_sd0_above_3 >= 6.0:
        z += 0.06323365 * Q.n_sd0_above_3 - 0.189701
    if Q.psi_0p2 >= 0.8857951:
        z += -4.181084 * Q.psi_0p2 + 3.703584
    if Q.N2 < 0.258375:
        z += 4.706738 * Q.N2 - 1.216103
    if Q.n_pairs_kt_above_1 < 325.0:
        z += -0.001809139 * Q.n_pairs_kt_above_1 + 0.5879703
    if Q.N2_b05 >= 0.4599592:
        z += -2.129567 * Q.N2_b05 + 0.979514
    z += -0.05142423 * Q.n_sd0_above_2
    if Q.sj3_mass3 < 0.3886647:
        z += -0.7627366 * Q.sj3_mass3 + 0.2964488
    if 0.2410564 <= Q.sj3_dr_min < 0.3526989:
        z += -0.07014988 * Q.sj3_dr_min + 0.01691007
    if Q.sj3_dr_min >= 0.3526989:
        z += -0.6527424 * Q.sj3_dr_min + 0.2223898
    if Q.mass_displaced5 < 1.262841:
        z += -0.009354479 * Q.mass_displaced5 + 0.01181322
    if Q.sj4_pair_mass_max < 75.78208:
        z += 0.00153956 * Q.sj4_pair_mass_max + 0.2548118
    if 75.78208 <= Q.sj4_pair_mass_max < 115.5142:
        z += -0.009349685 * Q.sj4_pair_mass_max + 1.080021
    if Q.z_charged_had < 0.3945478:
        z += 0.5248669 * Q.z_charged_had - 0.2070851
    if Q.lund2_lndelta >= -1.299999:
        z += -0.2188219 * Q.lund2_lndelta - 0.2844683
    if Q.D2_b2 < 0.4044054:
        z += 0.3913599 * Q.D2_b2 - 0.1582681
    if Q.max_abs_d0 < 10.52344:
        z += 0.01275542 * Q.max_abs_d0 - 0.1342309
    if Q.n_sdz_above_5 < 2.0:
        z += -0.006910651 * Q.n_sdz_above_5 + 0.0138213
    if Q.sum_z_dr2_top15 < 0.01040452:
        z += 1.785058 * Q.sum_z_dr2_top15 - 0.01857266
    if Q.mass_top5 >= 68.52153:
        z += 0.002206291 * Q.mass_top5 - 0.1511784
    if Q.z_top15_slots >= 0.8558319:
        z += -2.310282 * Q.z_top15_slots + 1.977213
    if Q.N3_b2 < 0.9644868:
        z += 0.3004308 * Q.N3_b2 - 0.2897616
    if Q.M2_b2 < 0.04544279:
        z += -5.933509 * Q.M2_b2 + 0.2696352
    if Q.C2_b2 < 0.07094338:
        z += 2.481917 * Q.C2_b2 - 0.1760756
    if Q.sj3_dr23 < 0.8799072:
        z += 0.06639211 * Q.sj3_dr23 - 0.05841889
    z += 0.007086397 * Q.isnhad_1 - 0.007086397
    if Q.tdz_2 >= -0.03532465:
        z += 0.02848348 * Q.tdz_2 + 0.001006169
    if Q.mass_charged >= 99.20396:
        z += 0.003308897 * Q.mass_charged - 0.3282557
    if Q.N3_b05 < 1.33105:
        z += -0.8741893 * Q.N3_b05 + 1.16359
    if Q.n_pairs_kt_above_3 < 126.0:
        z += -0.004060875 * Q.n_pairs_kt_above_3 + 0.5116702
    if Q.n_real_top30 >= 21.0:
        z += -0.03632534 * Q.n_real_top30 + 0.762832
    if 62.03827 <= Q.sd_mass < 88.81751:
        z += 0.020953 * Q.sd_mass - 1.299888
    if 88.81751 <= Q.sd_mass < 119.4443:
        z += -0.02524735 * Q.sd_mass + 2.803512
    if 119.4443 <= Q.sd_mass < 175.9333:
        z += 0.006586518 * Q.sd_mass - 0.9988635
    if Q.sd_mass >= 175.9333:
        z += -0.0005084765 * Q.sd_mass + 0.2493826
    if Q.sj3_pair_mass_max >= 142.9952:
        z += 0.006847911 * Q.sj3_pair_mass_max - 0.9792187
    if Q.tau32_b2 < 0.4680886:
        z += -0.06979982 * Q.tau32_b2 + 0.0326725
    if Q.sd_rg < 0.3995332:
        z += -2.057442 * Q.sd_rg + 0.8220164
    if Q.C2 >= 0.07037463:
        z += -1.774262 * Q.C2 + 0.124863
    if Q.n_dr_0p4_up < 4.0:
        z += -0.09828556 * Q.n_dr_0p4_up + 0.3931423
    if Q.lep_z < 0.3396572 and Q.mass < 127.2317:
        z += 0.04829281 * (0.3396572 - Q.lep_z) * (127.2317 - Q.mass)
    if Q.mass < 117.4867 and Q.D2 < 3.038341:
        z += 0.006138101 * (117.4867 - Q.mass) * (3.038341 - Q.D2)
    if Q.lep_ptrel < 43.20788 and Q.M3 > 0.03259227:
        z += 0.1123984 * (43.20788 - Q.lep_ptrel) * (Q.M3 - 0.03259227)
    if Q.e3_b2 < 0.0004126585 and Q.mass_charged > 90.44234:
        z += 33.088 * (0.0004126585 - Q.e3_b2) * (Q.mass_charged - 90.44234)
    if Q.lep_z < 0.221436 and Q.sj2_dr < 0.2403736:
        z += -38.5538 * (0.221436 - Q.lep_z) * (0.2403736 - Q.sj2_dr)
    if Q.sj2_dr < 0.5944498 and Q.n_dr_0p2_0p4 < 21.0:
        z += 0.1357677 * (0.5944498 - Q.sj2_dr) * (21.0 - Q.n_dr_0p2_0p4)
    if Q.z_displaced3 < 0.1061578 and Q.ecf_g42 < 5.969698e-05:
        z += -25473.53 * (0.1061578 - Q.z_displaced3) * (5.969698e-05 - Q.ecf_g42)
    if Q.mass_displaced3 < 26.78691 and Q.z_charged > 0.4622094:
        z += -0.01103904 * (26.78691 - Q.mass_displaced3) * (Q.z_charged - 0.4622094)
    if Q.mass_top30 < 74.51927 and Q.D2 < 3.038341:
        z += -0.019364 * (74.51927 - Q.mass_top30) * (3.038341 - Q.D2)
    if Q.z_displaced3 < 0.1061578 and Q.D2 > 2.446407:
        z += -1.155075 * (0.1061578 - Q.z_displaced3) * (Q.D2 - 2.446407)
    if Q.z_displaced3 < 0.1061578 and Q.mass_2photon > 4.18513:
        z += 0.02268182 * (0.1061578 - Q.z_displaced3) * (Q.mass_2photon - 4.18513)
    if Q.mass < 117.4867 and Q.lund2_lndelta > -1.755318:
        z += -0.02472739 * (117.4867 - Q.mass) * (Q.lund2_lndelta - -1.755318)
    if Q.mass_top30 < 74.51927 and Q.n_muon > 0.0:
        z += 0.02345577 * (74.51927 - Q.mass_top30) * (Q.n_muon - 0.0)
    if Q.lep_ptrel < 43.20788 and Q.n_muon > 0.0:
        z += -0.003619523 * (43.20788 - Q.lep_ptrel) * (Q.n_muon - 0.0)
    if Q.e3_b2 < 0.0004126585 and Q.pt_balance01 < 0.3972884:
        z += 2245.529 * (0.0004126585 - Q.e3_b2) * (0.3972884 - Q.pt_balance01)
    if Q.n_sd0_above_3 > 3.0 and Q.d0err_42 > 0.0:
        z += 0.02945647 * (Q.n_sd0_above_3 - 3.0) * (Q.d0err_42 - 0.0)
    if Q.e3_b2 < 0.0004126585 and Q.min_pair_mass > 10.59762:
        z += 55.57474 * (0.0004126585 - Q.e3_b2) * (Q.min_pair_mass - 10.59762)
    if Q.lund2_lndelta > -1.299999 and Q.tdz_0 < -0.02698624:
        z += 0.001900727 * (Q.lund2_lndelta - -1.299999) * (-0.02698624 - Q.tdz_0)
    if Q.lep_ptrel < 43.20788 and Q.dr_10 > 0.3349968:
        z += -0.0006292321 * (43.20788 - Q.lep_ptrel) * (Q.dr_10 - 0.3349968)
    if Q.sj4_pair_mass_max < 75.78208 and Q.iselectron_24 > 0.0:
        z += -0.002323022 * (75.78208 - Q.sj4_pair_mass_max) * (Q.iselectron_24 - 0.0)
    if Q.mass < 164.4374 and Q.tdz_6 > 0.03675711:
        z += 0.001591336 * (164.4374 - Q.mass) * (Q.tdz_6 - 0.03675711)
    if Q.lep_z < 0.3396572 and Q.e4 > 5.29882e-07:
        z += 18405.81 * (0.3396572 - Q.lep_z) * (Q.e4 - 5.29882e-07)
    if Q.mass_charged > 99.20396 and Q.lnerel_82 > -18.42068:
        z += -0.0001304502 * (Q.mass_charged - 99.20396) * (Q.lnerel_82 - -18.42068)
    if Q.D2_b2 < 0.4044054 and Q.dr_31 > 0.02273044:
        z += 1.544103 * (0.4044054 - Q.D2_b2) * (Q.dr_31 - 0.02273044)
    if Q.mass < 164.4374 and Q.lund2_lndelta > -1.411949:
        z += 0.01494644 * (164.4374 - Q.mass) * (Q.lund2_lndelta - -1.411949)
    if Q.n_sd0_above_3 < 6.0 and Q.n_lund_kt_above_1 > 4.0:
        z += 0.01412467 * (6.0 - Q.n_sd0_above_3) * (Q.n_lund_kt_above_1 - 4.0)
    if Q.mass_displaced3 < 26.78691 and Q.dr_31 < 0.06777369:
        z += 0.05410423 * (26.78691 - Q.mass_displaced3) * (0.06777369 - Q.dr_31)
    if Q.mass < 90.08945 and Q.lund2_lndelta > -2.073049:
        z += 0.003939589 * (90.08945 - Q.mass) * (Q.lund2_lndelta - -2.073049)
    if Q.sd_mass > 88.81751 and Q.charge_0 < 0.0:
        z += -0.0001591304 * (Q.sd_mass - 88.81751) * (0.0 - Q.charge_0)
    return z


def neuron_25(Q):
    z = 0.2642502
    if Q.lep_z < 0.3396572:
        z += 0.9644369 * Q.lep_z - 0.327578
    if Q.M2_b2 < 0.0417856:
        z += -0.6052263 * Q.M2_b2 + 0.02528975
    if Q.mass < 56.49019:
        z += 0.000942713 * Q.mass + 0.08977555
    if 56.49019 <= Q.mass < 79.47361:
        z += 0.0009227216 * Q.mass + 0.09090487
    if 79.47361 <= Q.mass < 90.08945:
        z += -0.005721574 * Q.mass + 0.618951
    if 90.08945 <= Q.mass < 95.14961:
        z += -0.004820165 * Q.mass + 0.5377436
    if 95.14961 <= Q.mass < 105.7234:
        z += 0.00295723 * Q.mass - 0.2022725
    if 105.7234 <= Q.mass < 182.8592:
        z += -0.001430927 * Q.mass + 0.2616582
    if Q.z_displaced3 < 0.02600452:
        z += -3.587161 * Q.z_displaced3 + 0.09328239
    if Q.z_displaced3 >= 0.4232894:
        z += 0.7402465 * Q.z_displaced3 - 0.3133385
    if Q.tau21 < 0.6200816:
        z += 0.277758 * Q.tau21 - 0.1722326
    if Q.n_neutral < 18.0:
        z += -0.01119316 * Q.n_neutral + 0.2014769
    if Q.lep_iso < 1.362094:
        z += -0.07425884 * Q.lep_iso + 0.008249687
    if 1.362094 <= Q.lep_iso < 2.907433:
        z += 0.06011489 * Q.lep_iso - 0.17478
    if Q.n_s3d_above_10 < 8.0:
        z += -0.007047031 * Q.n_s3d_above_10 + 0.05637624
    if Q.lep_ptrel < 27.3236:
        z += 0.00986094 * Q.lep_ptrel - 0.2694364
    if Q.lep_ptrel >= 43.20788:
        z += 0.01480179 * Q.lep_ptrel - 0.6395541
    if Q.lep_dr < 0.04607888:
        z += -0.1118146 * Q.lep_dr + 0.00515229
    if Q.lep_dr >= 0.3014662:
        z += -0.007133661 * Q.lep_dr + 0.002150558
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.01309601 * Q.n_pairs_kt_above_3 + 0.3666882
    if Q.sj2_mass1 < 21.21062:
        z += -0.002298273 * Q.sj2_mass1 + 0.04874779
    if Q.sj2_mass1 >= 77.42768:
        z += 0.006570696 * Q.sj2_mass1 - 0.5087538
    if Q.sip_3d_1 < 2.17433:
        z += -0.113152 * Q.sip_3d_1 + 0.426026
    if 2.17433 <= Q.sip_3d_1 < 3.373037:
        z += -0.1501587 * Q.sip_3d_1 + 0.5064908
    if Q.mass_top5 >= 29.68311:
        z += -0.009672905 * Q.mass_top5 + 0.2871219
    if Q.sj3_pair_mass_max < 73.24742:
        z += 0.003485724 * Q.sj3_pair_mass_max - 0.1855038
    if 73.24742 <= Q.sj3_pair_mass_max < 101.379:
        z += -0.0009471318 * Q.sj3_pair_mass_max + 0.1391914
    if 101.379 <= Q.sj3_pair_mass_max < 114.7848:
        z += -0.0032204 * Q.sj3_pair_mass_max + 0.3696531
    if Q.sj3_pair_mass_max >= 128.6605:
        z += -0.002597653 * Q.sj3_pair_mass_max + 0.3342155
    if -2.071997 <= Q.lund_max_lndelta < -1.303913:
        z += -0.1307833 * Q.lund_max_lndelta - 0.2709825
    if -1.303913 <= Q.lund_max_lndelta < -0.8095462:
        z += 0.1061992 * Q.lund_max_lndelta + 0.03802204
    if Q.lund_max_lndelta >= -0.8095462:
        z += 0.1308288 * Q.lund_max_lndelta + 0.05796084
    if Q.tau32 < 0.5068038:
        z += 0.7105498 * Q.tau32 - 0.3601093
    z += 0.007383891 * Q.n_dr_0p2_0p4
    if Q.sum_z_dr2_top3 < 0.007018285:
        z += 2.957177 * Q.sum_z_dr2_top3 - 0.03439579
    if 0.007018285 <= Q.sum_z_dr2_top3 < 0.01777495:
        z += 1.268189 * Q.sum_z_dr2_top3 - 0.02254199
    if Q.n_sdz_above_5 < 2.0:
        z += 0.06436236 * Q.n_sdz_above_5 - 0.1287247
    if 0.191059 <= Q.C2 < 0.2687133:
        z += -0.1614475 * Q.C2 + 0.03084599
    if Q.C2 >= 0.2687133:
        z += 2.046481 * Q.C2 - 0.5624537
    if Q.e4 < 2.097213e-06:
        z += 32834.57 * Q.e4 - 0.06886109
    if Q.pair_mean_lnm2 >= 3.654683:
        z += 0.1335319 * Q.pair_mean_lnm2 - 0.4880169
    if Q.pair_max_lnm2 < 7.075642:
        z += -0.08074774 * Q.pair_max_lnm2 + 0.5713421
    if Q.dr12 < 0.2116473:
        z += -0.08170646 * Q.dr12 + 0.01729295
    if Q.z_top5 < 0.5010328:
        z += 0.6954463 * Q.z_top5 - 0.3484414
    z += -0.6384028 * Q.sum_z_dr2_top20
    if Q.z_photon >= 0.4982257:
        z += -0.4047799 * Q.z_photon + 0.2016718
    if Q.sj4_pair_mass_max >= 69.83554:
        z += 0.001810233 * Q.sj4_pair_mass_max - 0.1264186
    z += -0.002932746 * Q.sj3_pairmax_over_m
    if 42.31629 <= Q.sd_mass < 78.4753:
        z += -0.002262865 * Q.sd_mass + 0.09575605
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += 0.01051063 * Q.sd_mass - 0.9066479
    if 88.81751 <= Q.sd_mass < 94.55722:
        z += 0.007104538 * Q.sd_mass - 0.6041272
    if 94.55722 <= Q.sd_mass < 100.8525:
        z += -0.005237835 * Q.sd_mass + 0.5629334
    if 100.8525 <= Q.sd_mass < 115.7091:
        z += -0.01045359 * Q.sd_mass + 1.088955
    if 115.7091 <= Q.sd_mass < 119.4443:
        z += 0.001662834 * Q.sd_mass - 0.3130248
    if 119.4443 <= Q.sd_mass < 127.8042:
        z += 0.00544609 * Q.sd_mass - 0.7649133
    if 127.8042 <= Q.sd_mass < 175.9333:
        z += 0.00208555 * Q.sd_mass - 0.335422
    if Q.sd_mass >= 175.9333:
        z += -0.002224603 * Q.sd_mass + 0.4228774
    if Q.mass_top40 >= 115.7429:
        z += 0.0003356366 * Q.mass_top40 - 0.03884756
    if Q.n_charged_pt_above_1 >= 10.0:
        z += 0.001876093 * Q.n_charged_pt_above_1 - 0.01876093
    if Q.n_sd0_above_2 < 9.0:
        z += -0.01143494 * Q.n_sd0_above_2 + 0.1029145
    if Q.sd_rg < 0.2601475:
        z += 1.241188 * Q.sd_rg - 0.322892
    if Q.D3_b2 < 0.1088678:
        z += 0.4485933 * Q.D3_b2 - 0.04883735
    if Q.mass_displaced3 < 6.341631:
        z += -0.001915455 * Q.mass_displaced3 + 0.01214711
    if Q.mass_displaced3 >= 13.03663:
        z += -0.0008907527 * Q.mass_displaced3 + 0.01161242
    if Q.n_sd0_above_3 < 3.0:
        z += 0.1062066 * Q.n_sd0_above_3 - 0.3186199
    if Q.tdz_9 >= 0.0:
        z += 0.06086097 * Q.tdz_9
    if Q.z_dr_0_0p05 < 0.3237313:
        z += 0.1917724 * Q.z_dr_0_0p05 - 0.06208275
    if Q.sj2_mass2 >= 42.12131:
        z += 0.008656057 * Q.sj2_mass2 - 0.3646045
    if Q.pair_mean_lndelta >= -1.352792:
        z += -0.2536286 * Q.pair_mean_lndelta - 0.3431067
    if Q.tau2 < 0.1336727:
        z += -1.461548 * Q.tau2 + 0.1953691
    if Q.lep_z < 0.3396572 and Q.mass_top5 > 37.66803:
        z += 0.03384066 * (0.3396572 - Q.lep_z) * (Q.mass_top5 - 37.66803)
    if Q.M2_b2 < 0.0417856 and Q.z_displaced3 < 0.4232894:
        z += -8.118312 * (0.0417856 - Q.M2_b2) * (0.4232894 - Q.z_displaced3)
    if Q.tau21 < 0.6200816 and Q.mass_charged < 65.0404:
        z += -0.007555393 * (0.6200816 - Q.tau21) * (65.0404 - Q.mass_charged)
    if Q.lep_z < 0.3396572 and Q.sj3_pairmax_over_m > 0.6627397:
        z += 1.095347 * (0.3396572 - Q.lep_z) * (Q.sj3_pairmax_over_m - 0.6627397)
    if Q.M2_b2 < 0.0417856 and Q.n_pairs_kt_above_10 < 23.0:
        z += -0.1629134 * (0.0417856 - Q.M2_b2) * (23.0 - Q.n_pairs_kt_above_10)
    if Q.lep_ptrel > 43.20788 and Q.e4 < 3.53193e-06:
        z += -2931.036 * (Q.lep_ptrel - 43.20788) * (3.53193e-06 - Q.e4)
    if Q.tau21 < 0.6200816 and Q.n_lund_kt_above_1 < 7.0:
        z += -0.0147473 * (0.6200816 - Q.tau21) * (7.0 - Q.n_lund_kt_above_1)
    if Q.sj3_pair_mass_max < 101.379 and Q.lne_4 > 3.378014:
        z += 0.0006920801 * (101.379 - Q.sj3_pair_mass_max) * (Q.lne_4 - 3.378014)
    if Q.lep_z < 0.3396572 and Q.n_dr_0p4_up < 19.0:
        z += 0.02192571 * (0.3396572 - Q.lep_z) * (19.0 - Q.n_dr_0p4_up)
    if Q.tau21 < 0.6200816 and Q.z_photon > 0.3090294:
        z += -0.7095004 * (0.6200816 - Q.tau21) * (Q.z_photon - 0.3090294)
    if Q.lep_z < 0.3396572 and Q.sj3_mass3 < 13.10191:
        z += 0.03670446 * (0.3396572 - Q.lep_z) * (13.10191 - Q.sj3_mass3)
    if Q.lep_dr < 0.04607888 and Q.sip_3d_3 < 3.157646:
        z += 4.393987 * (0.04607888 - Q.lep_dr) * (3.157646 - Q.sip_3d_3)
    if Q.sj3_pair_mass_max < 101.379 and Q.sd_zg > 0.1728409:
        z += -0.002801453 * (101.379 - Q.sj3_pair_mass_max) * (Q.sd_zg - 0.1728409)
    if Q.tau21 < 0.6200816 and Q.sip_3d_1 < 1343.246:
        z += 8.710575e-05 * (0.6200816 - Q.tau21) * (1343.246 - Q.sip_3d_1)
    if Q.lep_z < 0.3396572 and Q.z_neutral_had < 0.212136:
        z += 0.01972059 * (0.3396572 - Q.lep_z) * (0.212136 - Q.z_neutral_had)
    if Q.mass < 105.7234 and Q.n_electron < 1.0:
        z += -0.005948841 * (105.7234 - Q.mass) * (1.0 - Q.n_electron)
    if Q.lund_max_lndelta > -1.303913 and Q.sj4_dr_min < 0.3112717:
        z += 0.2966793 * (Q.lund_max_lndelta - -1.303913) * (0.3112717 - Q.sj4_dr_min)
    if Q.lund_max_lndelta > -1.303913 and Q.dr_55 < 0.02931273:
        z += 2.748685 * (Q.lund_max_lndelta - -1.303913) * (0.02931273 - Q.dr_55)
    if Q.e4 < 2.097213e-06 and Q.mass_2photon > 12.47889:
        z += 5494.536 * (2.097213e-06 - Q.e4) * (Q.mass_2photon - 12.47889)
    if Q.lep_z < 0.3396572 and Q.ischhad_10 > 0.0:
        z += -0.02186003 * (0.3396572 - Q.lep_z) * (Q.ischhad_10 - 0.0)
    if Q.mass_top5 > 29.68311 and Q.ismuon_29 > 0.0:
        z += -0.0234645 * (Q.mass_top5 - 29.68311) * (Q.ismuon_29 - 0.0)
    if Q.lep_dr > 0.3014662 and Q.dr_8 < 0.1048749:
        z += -3.621247 * (Q.lep_dr - 0.3014662) * (0.1048749 - Q.dr_8)
    if Q.sj3_pair_mass_max < 114.7848 and Q.ismuon_12 > 0.0:
        z += 0.0001007139 * (114.7848 - Q.sj3_pair_mass_max) * (Q.ismuon_12 - 0.0)
    if Q.sj3_pair_mass_max > 128.6605 and Q.tdz_54 > 0.0:
        z += 0.004217588 * (Q.sj3_pair_mass_max - 128.6605) * (Q.tdz_54 - 0.0)
    if Q.sj3_pair_mass_max < 114.7848 and Q.td0_44 > -0.009316366:
        z += 0.003105754 * (114.7848 - Q.sj3_pair_mass_max) * (Q.td0_44 - -0.009316366)
    if Q.sj2_mass2 > 42.12131 and Q.eta_76 > 0.0:
        z += 0.01221145 * (Q.sj2_mass2 - 42.12131) * (Q.eta_76 - 0.0)
    if Q.sj3_pair_mass_max < 114.7848 and Q.lne_8 > 3.172177:
        z += -0.001905032 * (114.7848 - Q.sj3_pair_mass_max) * (Q.lne_8 - 3.172177)
    if Q.lep_z < 0.3396572 and Q.n_real_top15 < 15.0:
        z += -0.1330749 * (0.3396572 - Q.lep_z) * (15.0 - Q.n_real_top15)
    return z


def neuron_26(Q):
    z = -3.231297e-06
    return z


def neuron_27(Q):
    z = -6.308482e-06
    return z


def neuron_28(Q):
    z = 9.617223e-06
    return z


def neuron_29(Q):
    z = 9.183854e-06
    return z


def neuron_30(Q):
    z = -8.603079e-06
    return z


def neuron_31(Q):
    z = 3.509509e-06
    return z


def neuron_32(Q):
    z = 1.441352e-05
    return z


def neuron_33(Q):
    z = 2.567321e-06
    return z


def neuron_34(Q):
    z = 1.388482e-05
    return z


def neuron_35(Q):
    z = -6.743304e-06
    return z


def neuron_36(Q):
    z = -4.831943e-06
    return z


def neuron_37(Q):
    z = -4.357445e-06
    return z


def neuron_38(Q):
    z = -9.41109e-06
    return z


def neuron_39(Q):
    z = 8.27427e-06
    return z


def neuron_40(Q):
    z = 7.775538e-07
    return z


def neuron_41(Q):
    z = 3.814238e-08
    return z


def neuron_42(Q):
    z = 8.779079e-06
    return z


def neuron_43(Q):
    z = -9.407695e-06
    return z


def neuron_44(Q):
    z = -3.324356e-06
    return z


def neuron_45(Q):
    z = 6.849297e-06
    return z


def neuron_46(Q):
    z = 1.368882e-07
    return z


def neuron_47(Q):
    z = 2.49205e-07
    return z


def neuron_48(Q):
    z = -8.791764e-06
    return z


def neuron_49(Q):
    z = -2.243814e-06
    return z


def neuron_50(Q):
    z = 5.175113e-06
    return z


def neuron_51(Q):
    z = -2.01221e-06
    return z


def neuron_52(Q):
    z = -1.017966
    if Q.lep_z < 0.221436:
        z += 2.748936 * Q.lep_z - 0.6697721
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += 0.5164775 * Q.lep_z - 0.1754253
    if Q.z_displaced3 < 0.02600452:
        z += -0.01140499 * Q.z_displaced3 - 0.05688625
    if 0.02600452 <= Q.z_displaced3 < 0.1342762:
        z += 0.5281421 * Q.z_displaced3 - 0.07091691
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.02815206 * Q.n_pairs_kt_above_1 + 2.214721
    if Q.n_pairs_kt_above_1 >= 80.0:
        z += -0.0004680405 * Q.n_pairs_kt_above_1
    z += 0.002922191 * Q.n_charged_had
    if Q.lep_ptrel < 12.15228:
        z += 0.001672263 * Q.lep_ptrel + 0.05703699
    if 12.15228 <= Q.lep_ptrel < 18.7678:
        z += -0.01375543 * Q.lep_ptrel + 0.2445186
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += -0.02576239 * Q.lep_ptrel + 0.4698629
    if Q.lep_ptrel >= 27.3236:
        z += -0.01542769 * Q.lep_ptrel + 0.1874816
    if Q.sj3_pair_mass_max < 56.73313:
        z += 0.001840682 * Q.sj3_pair_mass_max + 0.1881954
    if 56.73313 <= Q.sj3_pair_mass_max < 97.52633:
        z += -0.007173328 * Q.sj3_pair_mass_max + 0.6995884
    if Q.lep_iso < 0.4381892:
        z += -1.741346 * Q.lep_iso + 1.152466
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += -0.06751676 * Q.lep_iso + 0.4190127
    if 1.362094 <= Q.lep_iso < 44.14912:
        z += -0.007643638 * Q.lep_iso + 0.3374599
    if Q.mass < 71.96396:
        z += 0.002577369 * Q.mass - 0.3028065
    if 71.96396 <= Q.mass < 90.08945:
        z += 0.0193251 * Q.mass - 1.508039
    if 90.08945 <= Q.mass < 117.4867:
        z += 0.01711804 * Q.mass - 1.309207
    if Q.mass >= 117.4867:
        z += 0.01454067 * Q.mass - 1.0064
    if Q.jet_abs_eta >= 0.8317778:
        z += -0.2681949 * Q.jet_abs_eta + 0.2230786
    if Q.mass_top50 < 79.27954:
        z += -0.009553643 * Q.mass_top50 + 1.011139
    if 79.27954 <= Q.mass_top50 < 129.5874:
        z += -0.001840212 * Q.mass_top50 + 0.3996217
    if 129.5874 <= Q.mass_top50 < 161.1264:
        z += -0.005109661 * Q.mass_top50 + 0.8233012
    if Q.n_lepton < 1.0:
        z += 1.229473 * Q.n_lepton - 1.229473
    if Q.LHA < 0.5626523:
        z += -2.360347 * Q.LHA + 1.328055
    if Q.z_neutral_had < 0.3788785:
        z += 0.503678 * Q.z_neutral_had - 0.1908327
    if Q.mass_top15 >= 94.94813:
        z += 0.001810915 * Q.mass_top15 - 0.171943
    if Q.C2 < 0.1238959:
        z += 0.1061534 * Q.C2 - 0.01315197
    if Q.sip_3d_1 < 2.802481:
        z += 0.2418409 * Q.sip_3d_1 - 0.278477
    if 2.802481 <= Q.sip_3d_1 < 3.373037:
        z += -0.6998045 * Q.sip_3d_1 + 2.360467
    if Q.sj3_pair_mass_min >= 80.02563:
        z += 0.005694996 * Q.sj3_pair_mass_min - 0.4557456
    if Q.sj2_dr < 0.2848657:
        z += 1.197105 * Q.sj2_dr - 0.3410141
    if Q.sj2_mass2 < 1.851735:
        z += -0.2633338 * Q.sj2_mass2 + 0.4876244
    if Q.z_top10_slots >= 0.7362734:
        z += 1.436245 * Q.z_top10_slots - 1.057469
    if Q.jet_charge_k03 >= -0.1486714:
        z += -0.122518 * Q.jet_charge_k03 - 0.01821492
    if Q.C2_b2 < 0.09209404:
        z += -1.942503 * Q.C2_b2 + 0.178893
    if Q.sj2_mass1 >= 91.2852:
        z += 0.004196416 * Q.sj2_mass1 - 0.3830706
    if Q.sj3_mass1 >= 36.36236:
        z += -0.01112852 * Q.sj3_mass1 + 0.4046591
    if Q.e3_b2 < 1.008706e-05:
        z += 33154.16 * Q.e3_b2 - 0.3344281
    if Q.mass_displaced3 < 26.78691:
        z += -0.004359733 * Q.mass_displaced3 + 0.1167838
    if Q.sip_3d_2 < 45.26581:
        z += -0.004833884 * Q.sip_3d_2 + 0.2188096
    if Q.C3_b2 < 0.04229114:
        z += 6.306606 * Q.C3_b2 - 0.2667135
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.01214739 * Q.n_pairs_kt_above_3 + 0.3401269
    if Q.tdz_0 < 0.0:
        z += 2.845012 * Q.tdz_0 + 0.1686591
    if 0.0 <= Q.tdz_0 < 0.07687439:
        z += -2.193956 * Q.tdz_0 + 0.1686591
    if Q.mass_charged < 65.0404:
        z += -0.001322832 * Q.mass_charged + 0.08603755
    if Q.n_neutral_had >= 7.0:
        z += 0.01709853 * Q.n_neutral_had - 0.1196897
    if Q.N3 >= 0.2681212:
        z += -0.05088655 * Q.N3 + 0.01364376
    if Q.M2 < 0.04276413:
        z += 2.61591 * Q.M2 - 0.1118671
    if Q.tau32 < 0.3951525:
        z += 1.598436 * Q.tau32 - 0.631626
    if Q.tau2 < 0.06603871:
        z += 8.225422 * Q.tau2 - 0.5431963
    if Q.pair_max_lnm2 < 5.511295:
        z += 0.1406572 * Q.pair_max_lnm2 - 0.7752035
    if Q.N2_b2 >= 0.2392833:
        z += 2.921672 * Q.N2_b2 - 0.6991073
    if Q.lund_max_lndelta >= -0.615738:
        z += 1.068687 * Q.lund_max_lndelta + 0.6580309
    if Q.sd_mass < 42.31629:
        z += 0.01167443 * Q.sd_mass - 0.1770975
    if 42.31629 <= Q.sd_mass < 83.61981:
        z += -0.01304916 * Q.sd_mass + 0.8691131
    if 83.61981 <= Q.sd_mass < 123.2919:
        z += 0.009627499 * Q.sd_mass - 1.027105
    if 123.2919 <= Q.sd_mass < 154.5947:
        z += -0.00510776 * Q.sd_mass + 0.7896325
    if Q.sj4_pair_mass_max < 101.849:
        z += 0.001721865 * Q.sj4_pair_mass_max - 0.1753703
    if Q.n_dr_0_0p05 < 8.0:
        z += -0.02359965 * Q.n_dr_0_0p05 + 0.1887972
    if Q.n_sd0_above_3 < 4.0:
        z += 0.1388679 * Q.n_sd0_above_3 - 0.9942701
    if 4.0 <= Q.n_sd0_above_3 < 8.0:
        z += 0.1096996 * Q.n_sd0_above_3 - 0.8775968
    if Q.n_s3d_above_3 < 7.0:
        z += -0.2597868 * Q.n_s3d_above_3 + 1.818507
    if Q.mass_displaced5 < 0.8035262:
        z += -0.1715021 * Q.mass_displaced5 + 0.1378065
    if Q.n_particles >= 38.0:
        z += 0.003798202 * Q.n_particles - 0.1443317
    if Q.z_displaced3 < 0.1342762 and Q.N2 > 0.1824346:
        z += 5.132083 * (0.1342762 - Q.z_displaced3) * (Q.N2 - 0.1824346)
    if Q.lep_z < 0.3396572 and Q.sj3_pair_mass_min > 48.40547:
        z += -0.01435457 * (0.3396572 - Q.lep_z) * (Q.sj3_pair_mass_min - 48.40547)
    if Q.mass < 117.4867 and Q.tau21_b2 < 0.3565533:
        z += -0.07481062 * (117.4867 - Q.mass) * (0.3565533 - Q.tau21_b2)
    if Q.mass > 90.08945 and Q.n_s3d_above_3 < 5.0:
        z += 7.365845e-05 * (Q.mass - 90.08945) * (5.0 - Q.n_s3d_above_3)
    if Q.lep_ptrel < 27.3236 and Q.tau3 < 0.08643515:
        z += -0.1965399 * (27.3236 - Q.lep_ptrel) * (0.08643515 - Q.tau3)
    if Q.z_neutral_had < 0.3788785 and Q.ecf_g41 > 6.216954e-05:
        z += 2921.832 * (0.3788785 - Q.z_neutral_had) * (Q.ecf_g41 - 6.216954e-05)
    if Q.lep_iso < 1.362094 and Q.sip_3d_3 < 49.03695:
        z += -0.004117067 * (1.362094 - Q.lep_iso) * (49.03695 - Q.sip_3d_3)
    if Q.LHA < 0.5626523 and Q.mass_displaced3 < 26.78691:
        z += 0.01945147 * (0.5626523 - Q.LHA) * (26.78691 - Q.mass_displaced3)
    if Q.mass_top50 < 129.5874 and Q.lep_dr < 0.114659:
        z += 0.04921846 * (129.5874 - Q.mass_top50) * (0.114659 - Q.lep_dr)
    if Q.mass > 90.08945 and Q.tau32_b2 < 0.7269473:
        z += -0.01601285 * (Q.mass - 90.08945) * (0.7269473 - Q.tau32_b2)
    if Q.z_displaced3 < 0.1342762 and Q.lep_dr > 0.114659:
        z += -4.993064 * (0.1342762 - Q.z_displaced3) * (Q.lep_dr - 0.114659)
    if Q.z_displaced3 < 0.1342762 and Q.max_abs_d0 < 3.113281:
        z += -0.5938468 * (0.1342762 - Q.z_displaced3) * (3.113281 - Q.max_abs_d0)
    if Q.mass > 90.08945 and Q.tau43_b2 < 0.6818924:
        z += -0.009659874 * (Q.mass - 90.08945) * (0.6818924 - Q.tau43_b2)
    if Q.C2_b2 < 0.09209404 and Q.D3_b2 < 1.462588:
        z += 1.918596 * (0.09209404 - Q.C2_b2) * (1.462588 - Q.D3_b2)
    if Q.C2_b2 < 0.09209404 and Q.n_muon > 0.0:
        z += -0.2295449 * (0.09209404 - Q.C2_b2) * (Q.n_muon - 0.0)
    if Q.lep_iso < 0.4381892 and Q.mass_displaced5 < 0.3201841:
        z += -1.173323 * (0.4381892 - Q.lep_iso) * (0.3201841 - Q.mass_displaced5)
    if Q.z_displaced3 < 0.1342762 and Q.sip_3d_3 < 4.606241:
        z += 1.41312 * (0.1342762 - Q.z_displaced3) * (4.606241 - Q.sip_3d_3)
    if Q.lep_ptrel < 18.7678 and Q.mass_displaced3 < 18.80005:
        z += 0.002006855 * (18.7678 - Q.lep_ptrel) * (18.80005 - Q.mass_displaced3)
    if Q.z_displaced3 < 0.1342762 and Q.dr_21 < 0.04063514:
        z += -18.93262 * (0.1342762 - Q.z_displaced3) * (0.04063514 - Q.dr_21)
    if Q.mass_displaced3 < 26.78691 and Q.n_lund_kt_above_1 > 4.0:
        z += 0.001133677 * (26.78691 - Q.mass_displaced3) * (Q.n_lund_kt_above_1 - 4.0)
    if Q.mass_top50 < 129.5874 and Q.dr_21 < 0.0632362:
        z += 0.01749121 * (129.5874 - Q.mass_top50) * (0.0632362 - Q.dr_21)
    if Q.mass_top15 > 94.94813 and Q.sj3_dr13 > 0.2640447:
        z += -0.002267117 * (Q.mass_top15 - 94.94813) * (Q.sj3_dr13 - 0.2640447)
    if Q.lep_ptrel < 18.7678 and Q.dr_21 < 0.1905794:
        z += -0.02317539 * (18.7678 - Q.lep_ptrel) * (0.1905794 - Q.dr_21)
    if Q.mass > 90.08945 and Q.td0_13 > -0.06278656:
        z += 0.0007452012 * (Q.mass - 90.08945) * (Q.td0_13 - -0.06278656)
    if Q.lep_z < 0.3396572 and Q.tdz_0 < -0.01644749:
        z += 7.185773 * (0.3396572 - Q.lep_z) * (-0.01644749 - Q.tdz_0)
    if Q.C3_b2 < 0.04229114 and Q.pt2_over_pt0 < 0.2816529:
        z += 23.08438 * (0.04229114 - Q.C3_b2) * (0.2816529 - Q.pt2_over_pt0)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.mass_2photon > 3.013:
        z += 0.0008399347 * (28.0 - Q.n_pairs_kt_above_3) * (Q.mass_2photon - 3.013)
    if Q.mass > 90.08945 and Q.d0err_75 > 0.0:
        z += -0.005136326 * (Q.mass - 90.08945) * (Q.d0err_75 - 0.0)
    if Q.z_neutral_had < 0.3788785 and Q.d0err_3 < 0.01100159:
        z += -13.12619 * (0.3788785 - Q.z_neutral_had) * (0.01100159 - Q.d0err_3)
    if Q.n_s3d_above_3 < 7.0 and Q.sip_3d_3 < 175.9957:
        z += -0.0006895907 * (7.0 - Q.n_s3d_above_3) * (175.9957 - Q.sip_3d_3)
    if Q.n_s3d_above_3 < 7.0 and Q.sip_3d_2 < 70.05844:
        z += -0.002186483 * (7.0 - Q.n_s3d_above_3) * (70.05844 - Q.sip_3d_2)
    if Q.lep_z < 0.3396572 and Q.sip_3d_2 < 4.636903:
        z += 0.3340534 * (0.3396572 - Q.lep_z) * (4.636903 - Q.sip_3d_2)
    if Q.sj3_pair_mass_min > 80.02563 and Q.sip_3d_2 < 20.75111:
        z += 0.0003670619 * (Q.sj3_pair_mass_min - 80.02563) * (20.75111 - Q.sip_3d_2)
    if Q.C2_b2 < 0.09209404 and Q.sip_3d_2 > 3.041925:
        z += -0.0003464736 * (0.09209404 - Q.C2_b2) * (Q.sip_3d_2 - 3.041925)
    if Q.tau32 < 0.3951525 and Q.d0err_14 > 0.01499939:
        z += -12.4254 * (0.3951525 - Q.tau32) * (Q.d0err_14 - 0.01499939)
    if Q.n_lepton < 1.0 and Q.sip_3d_3 < 11.13866:
        z += 0.01975034 * (1.0 - Q.n_lepton) * (11.13866 - Q.sip_3d_3)
    return z


def neuron_53(Q):
    z = -1.889392e-06
    return z


def neuron_54(Q):
    z = 1.190656e-05
    return z


def neuron_55(Q):
    z = -2.066932e-05
    return z


def neuron_56(Q):
    z = -7.072387e-07
    return z


def neuron_57(Q):
    z = -1.959676e-06
    return z


def neuron_58(Q):
    z = -3.61715e-06
    return z


def neuron_59(Q):
    z = 6.029272e-06
    return z


def neuron_60(Q):
    z = -4.666768e-06
    return z


def neuron_61(Q):
    z = 5.840546e-06
    return z


def neuron_62(Q):
    z = 9.422871e-07
    return z


def neuron_63(Q):
    z = -8.308531e-06
    return z


def neuron_64(Q):
    z = 1.258629e-06
    return z


def neuron_65(Q):
    z = 5.791076e-06
    return z


def neuron_66(Q):
    z = 2.709755e-06
    return z


def neuron_67(Q):
    z = -5.219533e-06
    return z


def neuron_68(Q):
    z = 3.781859
    if Q.tau32 < 0.7981752:
        z += -1.991541 * Q.tau32 + 1.589599
    if 6.983043 <= Q.lep_ptrel < 43.20788:
        z += -0.01553947 * Q.lep_ptrel + 0.1085128
    if Q.lep_ptrel >= 43.20788:
        z += -0.02191258 * Q.lep_ptrel + 0.3838813
    if Q.pair_mean_lnm2 < 3.38061:
        z += 0.1212081 * Q.pair_mean_lnm2 - 0.4097575
    if Q.n_s3d_above_3 >= 5.0:
        z += -0.01517536 * Q.n_s3d_above_3 + 0.07587678
    if Q.sj3_pair_mass_max < 44.2029:
        z += 0.003556264 * Q.sj3_pair_mass_max + 0.9037657
    if 44.2029 <= Q.sj3_pair_mass_max < 56.73313:
        z += 0.01247591 * Q.sj3_pair_mass_max + 0.5094915
    if 56.73313 <= Q.sj3_pair_mass_max < 80.36029:
        z += -0.004846768 * Q.sj3_pair_mass_max + 1.492261
    if 80.36029 <= Q.sj3_pair_mass_max < 120.6471:
        z += -0.01044802 * Q.sj3_pair_mass_max + 1.94238
    if Q.sj3_pair_mass_max >= 120.6471:
        z += 0.008919645 * Q.sj3_pair_mass_max - 0.3942742
    if Q.M2_b2 < 0.04544279:
        z += 13.33631 * Q.M2_b2 - 0.606039
    if Q.mass < 79.47361:
        z += 0.02333425 * Q.mass - 4.216927
    if 79.47361 <= Q.mass < 90.08945:
        z += 0.005602043 * Q.mass - 2.807684
    if 90.08945 <= Q.mass < 114.0172:
        z += -0.01164109 * Q.mass - 1.254259
    if 114.0172 <= Q.mass < 127.2317:
        z += 0.02754757 * Q.mass - 5.722443
    if 127.2317 <= Q.mass < 164.4374:
        z += 0.01910784 * Q.mass - 4.648641
    if Q.mass >= 164.4374:
        z += -0.01773221 * Q.mass + 1.409243
    if Q.n_pt_above_1 >= 17.0:
        z += -0.03682998 * Q.n_pt_above_1 + 0.6261097
    if Q.sj3_pair_mass_min < 80.02563:
        z += 0.02406265 * Q.sj3_pair_mass_min - 1.925628
    if Q.mass_top30 >= 162.7874:
        z += 0.004772884 * Q.mass_top30 - 0.7769654
    if Q.sj3_dr_min >= 0.3526989:
        z += 0.1975199 * Q.sj3_dr_min - 0.06966506
    if Q.z_displaced3 < 0.3261071:
        z += -0.07739051 * Q.z_displaced3 + 0.02523759
    if Q.n_sdz_above_5 < 3.0:
        z += -0.04614654 * Q.n_sdz_above_5 + 0.1384396
    if Q.psi_0p1 >= 0.835544:
        z += -2.467494 * Q.psi_0p1 + 2.0617
    if Q.jet_abs_eta < 1.323111:
        z += 0.2159726 * Q.jet_abs_eta - 0.2857557
    z += -1.226583 * Q.sj3_pairmin_over_m
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.002379478 * Q.n_pairs_kt_above_3 + 0.06662538
    if 42.31629 <= Q.sd_mass < 83.61981:
        z += 0.01773036 * Q.sd_mass - 0.750283
    if 83.61981 <= Q.sd_mass < 88.81751:
        z += -0.03664248 * Q.sd_mass + 3.796364
    if 88.81751 <= Q.sd_mass < 127.8042:
        z += -0.01914231 * Q.sd_mass + 2.242043
    if 127.8042 <= Q.sd_mass < 154.5947:
        z += 0.02785644 * Q.sd_mass - 3.764598
    if Q.sd_mass >= 154.5947:
        z += 0.006424747 * Q.sd_mass - 0.4513726
    if Q.n_lepton < 1.0:
        z += -0.6458898 * Q.n_lepton + 0.971248
    if 1.0 <= Q.n_lepton < 2.0:
        z += -0.3253582 * Q.n_lepton + 0.6507165
    if Q.lep_iso < 0.4381892:
        z += 1.73828 * Q.lep_iso - 0.7616954
    if Q.tau21_b2 >= 0.5211946:
        z += -0.8760463 * Q.tau21_b2 + 0.4565906
    if Q.mass_top15 >= 119.5993:
        z += -0.006379372 * Q.mass_top15 + 0.7629682
    if Q.n_charged_had >= 13.0:
        z += -0.059404 * Q.n_charged_had + 0.7722521
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.01604815 * Q.n_pairs_kt_above_1 + 1.283852
    if Q.n_pairs_kt_above_1 >= 101.0:
        z += 0.0008311433 * Q.n_pairs_kt_above_1 - 0.08394547
    if Q.tau3 < 0.03378303:
        z += -0.1910301 * Q.tau3 + 0.006453576
    if Q.sip_3d_2 < 226.3008:
        z += 4.186104e-05 * Q.sip_3d_2 - 0.009473187
    if Q.mass_top10 >= 27.42853:
        z += 0.002426824 * Q.mass_top10 - 0.0665642
    if Q.sum_z_dr2_top3 < 0.002792418:
        z += -89.26525 * Q.sum_z_dr2_top3 + 0.2492659
    if Q.M3 < 0.03259227:
        z += 3.622344 * Q.M3 - 0.1180604
    if Q.sj3_mass2 < 3.298199:
        z += -0.02372063 * Q.sj3_mass2 + 0.07823535
    if Q.n_electron >= 1.0:
        z += 0.1538969 * Q.n_electron - 0.1538969
    if Q.sum_z_dr2_top15 < 0.01040452:
        z += 20.65794 * Q.sum_z_dr2_top15 - 0.2149358
    if Q.z_displaced5 >= 0.2801368:
        z += -1.488986 * Q.z_displaced5 + 0.41712
    if Q.n_dr_0p2_0p4 >= 9.0:
        z += -0.008390204 * Q.n_dr_0p2_0p4 + 0.07551184
    if Q.D2_b2 < 2.286291:
        z += 0.1120954 * Q.D2_b2 - 0.2562827
    if Q.mass_2photon >= 22.18431:
        z += -0.01110789 * Q.mass_2photon + 0.2464209
    if Q.mass_over_sum_pt_sq >= 0.0729277:
        z += -2.858279 * Q.mass_over_sum_pt_sq + 0.2084477
    if Q.e4_b05 >= 2.879843e-05:
        z += 1166.202 * Q.e4_b05 - 0.03358477
    if Q.e3_b2 < 7.104488e-06:
        z += 6104.444 * Q.e3_b2 - 0.04336895
    if Q.sip_3d_3 < 2.317124:
        z += -0.2225156 * Q.sip_3d_3 + 0.5155961
    if Q.mass_2charged < 0.9506259:
        z += -0.1987404 * Q.mass_2charged + 0.1889278
    if Q.mass_top5 < 47.41085:
        z += 0.002049326 * Q.mass_top5 - 0.09716028
    if Q.z_dr_0p1_0p2 < 0.1036778:
        z += -1.144961 * Q.z_dr_0p1_0p2 + 0.118707
    if Q.z_charged_had < 0.7223231:
        z += 0.2027556 * Q.z_charged_had - 0.146455
    if Q.lep_z >= 0.5187302:
        z += 0.7405088 * Q.lep_z - 0.3841243
    if Q.C2_b05 >= 0.3533901:
        z += -0.9065918 * Q.C2_b05 + 0.3203806
    if Q.n_pt_above_5 >= 25.0:
        z += 0.009485736 * Q.n_pt_above_5 - 0.2371434
    if Q.mass_displaced5 < 2.382955:
        z += -0.04494651 * Q.mass_displaced5 + 0.1071055
    if Q.tau32 < 0.7981752 and Q.sj3_dr_min < 0.3190414:
        z += -1.723798 * (0.7981752 - Q.tau32) * (0.3190414 - Q.sj3_dr_min)
    if Q.pair_mean_lnm2 < 3.38061 and Q.z_top30_slots > 0.8316085:
        z += -0.05060993 * (3.38061 - Q.pair_mean_lnm2) * (Q.z_top30_slots - 0.8316085)
    if Q.mass < 114.0172 and Q.n_lund_kt_above_1 > 3.0:
        z += -0.001100902 * (114.0172 - Q.mass) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.M2_b2 < 0.04544279 and Q.mass_displaced3 > 0.0:
        z += 0.3082609 * (0.04544279 - Q.M2_b2) * (Q.mass_displaced3 - 0.0)
    if Q.tau32 < 0.7981752 and Q.z_charged_had < 0.7223231:
        z += -0.9887849 * (0.7981752 - Q.tau32) * (0.7223231 - Q.z_charged_had)
    if Q.n_pt_above_1 > 17.0 and Q.tau43 < 0.8939856:
        z += 0.05099476 * (Q.n_pt_above_1 - 17.0) * (0.8939856 - Q.tau43)
    if Q.mass < 114.0172 and Q.D2 < 2.255345:
        z += -0.001708796 * (114.0172 - Q.mass) * (2.255345 - Q.D2)
    if Q.pair_mean_lnm2 < 3.38061 and Q.z_neutral_had < 0.3788785:
        z += 0.3493143 * (3.38061 - Q.pair_mean_lnm2) * (0.3788785 - Q.z_neutral_had)
    if Q.n_pt_above_1 > 17.0 and Q.z_displaced3 < 0.1061578:
        z += -0.115776 * (Q.n_pt_above_1 - 17.0) * (0.1061578 - Q.z_displaced3)
    if Q.n_pt_above_1 > 17.0 and Q.tau54 < 0.8786609:
        z += 0.05569046 * (Q.n_pt_above_1 - 17.0) * (0.8786609 - Q.tau54)
    if Q.n_s3d_above_3 > 5.0 and Q.sip_3d_2 < 226.3008:
        z += -6.981844e-05 * (Q.n_s3d_above_3 - 5.0) * (226.3008 - Q.sip_3d_2)
    if Q.mass_top30 > 162.7874 and Q.sum_e < 1049.334:
        z += -1.612486e-05 * (Q.mass_top30 - 162.7874) * (1049.334 - Q.sum_e)
    if Q.sj3_pair_mass_min < 80.02563 and Q.sj3_mass1 > 18.73268:
        z += 0.0001538218 * (80.02563 - Q.sj3_pair_mass_min) * (Q.sj3_mass1 - 18.73268)
    if Q.n_sdz_above_5 < 3.0 and Q.sj2_dr > 0.3220633:
        z += -0.1526609 * (3.0 - Q.n_sdz_above_5) * (Q.sj2_dr - 0.3220633)
    if Q.tau32 < 0.7981752 and Q.sj3_dr_min > 0.3526989:
        z += -9.879487 * (0.7981752 - Q.tau32) * (Q.sj3_dr_min - 0.3526989)
    if Q.mass < 114.0172 and Q.lep_dr < 0.114659:
        z += 0.06204097 * (114.0172 - Q.mass) * (0.114659 - Q.lep_dr)
    if Q.z_displaced3 < 0.3261071 and Q.lep_dr < 0.4191372:
        z += -2.704558 * (0.3261071 - Q.z_displaced3) * (0.4191372 - Q.lep_dr)
    if Q.z_displaced3 < 0.3261071 and Q.n_lund > 9.0:
        z += -0.1242303 * (0.3261071 - Q.z_displaced3) * (Q.n_lund - 9.0)
    if Q.mass < 90.08945 and Q.n_lepton < 1.0:
        z += -0.02975482 * (90.08945 - Q.mass) * (1.0 - Q.n_lepton)
    if Q.lep_ptrel > 6.983043 and Q.C2_b2 < 0.1404188:
        z += 0.08742416 * (Q.lep_ptrel - 6.983043) * (0.1404188 - Q.C2_b2)
    if Q.sd_mass > 154.5947 and Q.sd_zg < 0.2450652:
        z += -0.02322416 * (Q.sd_mass - 154.5947) * (0.2450652 - Q.sd_zg)
    if Q.sd_mass > 42.31629 and Q.sj3_dr13 > 0.6229991:
        z += -0.0041645 * (Q.sd_mass - 42.31629) * (Q.sj3_dr13 - 0.6229991)
    if Q.n_lepton < 1.0 and Q.sip_3d_2 < 4.636903:
        z += -0.1453544 * (1.0 - Q.n_lepton) * (4.636903 - Q.sip_3d_2)
    if Q.M2_b2 < 0.04544279 and Q.mass_neutral < 76.15079:
        z += 0.1239227 * (0.04544279 - Q.M2_b2) * (76.15079 - Q.mass_neutral)
    if Q.sd_mass > 88.81751 and Q.M3 > 0.02540381:
        z += -0.1076442 * (Q.sd_mass - 88.81751) * (Q.M3 - 0.02540381)
    if Q.z_displaced3 < 0.3261071 and Q.n_lund_kt_above_5 > 1.0:
        z += 0.2940155 * (0.3261071 - Q.z_displaced3) * (Q.n_lund_kt_above_5 - 1.0)
    if Q.jet_abs_eta < 1.323111 and Q.n_neutral_had > 6.0:
        z += -0.04870357 * (1.323111 - Q.jet_abs_eta) * (Q.n_neutral_had - 6.0)
    if Q.n_s3d_above_3 > 5.0 and Q.eta_48 > 0.2019104:
        z += -0.01726523 * (Q.n_s3d_above_3 - 5.0) * (Q.eta_48 - 0.2019104)
    if Q.sd_mass > 42.31629 and Q.z_dr_0p2_0p4 < 0.4683306:
        z += -0.0004955695 * (Q.sd_mass - 42.31629) * (0.4683306 - Q.z_dr_0p2_0p4)
    if Q.z_displaced3 < 0.3261071 and Q.mass_2photon > 9.82024:
        z += 0.02219238 * (0.3261071 - Q.z_displaced3) * (Q.mass_2photon - 9.82024)
    if Q.mass_top10 > 27.42853 and Q.dr_77 > 0.0:
        z += 0.003216949 * (Q.mass_top10 - 27.42853) * (Q.dr_77 - 0.0)
    if Q.mass_top30 > 162.7874 and Q.e4_b2 > 1.13e-10:
        z += 29.63538 * (Q.mass_top30 - 162.7874) * (Q.e4_b2 - 1.13e-10)
    if Q.mass_2photon > 22.18431 and Q.iselectron_1 > 0.0:
        z += -0.02307676 * (Q.mass_2photon - 22.18431) * (Q.iselectron_1 - 0.0)
    if Q.mass < 127.2317 and Q.sd_zg > 0.1900649:
        z += 0.01160943 * (127.2317 - Q.mass) * (Q.sd_zg - 0.1900649)
    if Q.sj3_dr_min > 0.3526989 and Q.charge_27 > 0.0:
        z += -0.3334003 * (Q.sj3_dr_min - 0.3526989) * (Q.charge_27 - 0.0)
    if Q.M3 < 0.03259227 and Q.ismuon_4 > 0.0:
        z += -2.99954 * (0.03259227 - Q.M3) * (Q.ismuon_4 - 0.0)
    return z


def neuron_69(Q):
    z = -6.926153e-06
    return z


def neuron_70(Q):
    z = 0.1001232
    if Q.lep_z < 0.221436:
        z += 0.5605539 * Q.lep_z - 0.122476
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -0.01396346 * Q.lep_z + 0.004742789
    if Q.n_s3d_above_3 < 2.0:
        z += -0.01896943 * Q.n_s3d_above_3 + 0.1554218
    if 2.0 <= Q.n_s3d_above_3 < 4.0:
        z += -0.01468537 * Q.n_s3d_above_3 + 0.1468537
    if 4.0 <= Q.n_s3d_above_3 < 10.0:
        z += -0.0200353 * Q.n_s3d_above_3 + 0.1682534
    if Q.n_s3d_above_3 >= 10.0:
        z += -0.005349935 * Q.n_s3d_above_3 + 0.02139974
    if Q.n_pairs_kt_above_1 < 58.0:
        z += 0.008613376 * Q.n_pairs_kt_above_1 - 0.4995758
    if Q.n_pairs_kt_above_1 >= 462.0:
        z += -0.0002198254 * Q.n_pairs_kt_above_1 + 0.1015593
    if Q.sip_3d_2 < 447.0873:
        z += -0.0002677303 * Q.sip_3d_2 + 0.1196988
    if Q.lep_ptrel < 27.3236:
        z += 0.001140115 * Q.lep_ptrel - 0.03115205
    if Q.mass_displaced5 < 32.61679:
        z += 0.009409478 * Q.mass_displaced5 - 0.3069069
    if Q.sj4_pair_mass_max < 48.76729:
        z += -0.005262778 * Q.sj4_pair_mass_max + 0.2566514
    if Q.mass < 79.47361:
        z += -0.006609779 * Q.mass + 0.7949998
    if 79.47361 <= Q.mass < 90.08945:
        z += -0.005443697 * Q.mass + 0.702327
    if 90.08945 <= Q.mass < 117.4867:
        z += -0.005178075 * Q.mass + 0.6783973
    if 117.4867 <= Q.mass < 127.2317:
        z += -0.002455132 * Q.mass + 0.3584878
    if 127.2317 <= Q.mass < 164.4374:
        z += -0.0001145174 * Q.mass + 0.06068737
    if 164.4374 <= Q.mass < 182.8592:
        z += 0.006133383 * Q.mass - 0.9667015
    if Q.mass >= 182.8592:
        z += 0.002606236 * Q.mass - 0.3217301
    if Q.mass_top50 < 161.1264:
        z += 0.001905158 * Q.mass_top50 - 0.3069713
    if Q.lep_iso < 0.4381892:
        z += -0.4924313 * Q.lep_iso - 0.0005195034
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += 0.2341124 * Q.lep_iso - 0.3188831
    if Q.n_lepton < 1.0:
        z += 0.2467859 * Q.n_lepton - 0.2467859
    if Q.z_displaced5 >= 0.06245248:
        z += -1.146152 * Q.z_displaced5 + 0.07158006
    if Q.mass_2charged >= 24.4079:
        z += -0.00287467 * Q.mass_2charged + 0.07016467
    if Q.sj3_mass2 < 4.326415:
        z += -0.09311341 * Q.sj3_mass2 + 0.4028472
    if Q.sj3_pair_mass_min >= 44.1643:
        z += -0.006836758 * Q.sj3_pair_mass_min + 0.3019406
    if Q.sj2_mass1 >= 91.2852:
        z += 0.001628884 * Q.sj2_mass1 - 0.148693
    if Q.sj3_dr12 < 0.3273298:
        z += -0.1369928 * Q.sj3_dr12 + 0.04484183
    if Q.N2_b2 < 0.2392833:
        z += -2.936624 * Q.N2_b2 + 0.7026849
    if Q.C2_b2 < 0.1049877:
        z += 0.7104692 * Q.C2_b2 - 0.07459055
    if Q.sj3_mass3 < 2.151069:
        z += -0.04288922 * Q.sj3_mass3 + 0.09225767
    if Q.sip_3d_3 < 577.991:
        z += 0.0002983617 * Q.sip_3d_3 - 0.1724504
    z += 0.00670258 * Q.n_photon
    if Q.mass_top30 < 121.0623:
        z += -0.0007809016 * Q.mass_top30 + 0.0598581
    if 121.0623 <= Q.mass_top30 < 146.3096:
        z += 0.001373601 * Q.mass_top30 - 0.200971
    if Q.z_displaced3 >= 0.4232894:
        z += -3.358446 * Q.z_displaced3 + 1.421594
    if Q.sj3_pair_mass_max >= 128.6605:
        z += -0.002858685 * Q.sj3_pair_mass_max + 0.3677999
    if Q.tau5 < 0.02098847:
        z += -1.833232 * Q.tau5 + 0.03847674
    if Q.sj3_pairmin_over_m >= 0.4866692:
        z += 3.694979 * Q.sj3_pairmin_over_m - 1.798232
    if Q.sj3_dr13 < 0.2317874:
        z += -0.5019911 * Q.sj3_dr13 + 0.1163552
    if Q.max_abs_d0 < 10.52344:
        z += 0.01292526 * Q.max_abs_d0 - 0.1360181
    if Q.n_s3d_above_10 < 2.0:
        z += 0.04706574 * Q.n_s3d_above_10 - 0.09413148
    if Q.dr01 < 0.0856189:
        z += 0.8601964 * Q.dr01 + 0.4854703
    if 0.0856189 <= Q.dr01 < 0.3258728:
        z += -2.327202 * Q.dr01 + 0.7583719
    if Q.m01 < 2.43362:
        z += -0.04081161 * Q.m01 - 0.480232
    if 2.43362 <= Q.m01 < 17.3274:
        z += 0.03891235 * Q.m01 - 0.6742499
    if -2.071997 <= Q.lund_max_lndelta < -1.388411:
        z += -0.1407115 * Q.lund_max_lndelta - 0.2915538
    if Q.lund_max_lndelta >= -1.388411:
        z += 0.03826295 * Q.lund_max_lndelta - 0.0430637
    if Q.D2_b2 < 2.613544:
        z += 0.03005302 * Q.D2_b2 - 0.07854488
    if Q.n_dr_0p2_0p4 < 24.0:
        z += -0.01508808 * Q.n_dr_0p2_0p4 + 0.3621139
    if Q.n_pt_above_1 >= 54.0:
        z += -0.009976099 * Q.n_pt_above_1 + 0.5387094
    if Q.lep_dr >= 0.4191372:
        z += -1.542244 * Q.lep_dr + 0.646412
    if Q.n_sd0_above_5 >= 4.0:
        z += 0.01311934 * Q.n_sd0_above_5 - 0.05247736
    if Q.mass_charged < 22.56857:
        z += -0.04455056 * Q.mass_charged + 1.247677
    if 22.56857 <= Q.mass_charged < 68.20711:
        z += -0.005307666 * Q.mass_charged + 0.3620206
    if Q.tau4 >= 0.05417009:
        z += -1.204238 * Q.tau4 + 0.06523371
    if Q.pair_mean_lnz >= -1.676789:
        z += 0.3557535 * Q.pair_mean_lnz + 0.5965237
    if Q.tdz_0 >= 0.04233308:
        z += -3.057923 * Q.tdz_0 + 0.1294513
    if Q.tau3 < 0.01470468:
        z += -44.24899 * Q.tau3 + 0.6506674
    if Q.mass_over_sum_pt_sq < 0.02574154:
        z += 7.825246 * Q.mass_over_sum_pt_sq - 0.2014339
    if Q.z_top3_slots >= 0.6094828:
        z += -0.5753295 * Q.z_top3_slots + 0.3506534
    if Q.pair_mean_lndelta >= -1.714682:
        z += 0.4693779 * Q.pair_mean_lndelta + 0.804834
    if 62.03827 <= Q.sd_mass < 88.81751:
        z += -0.0011032 * Q.sd_mass + 0.06844065
    if 88.81751 <= Q.sd_mass < 123.2919:
        z += 0.004732496 * Q.sd_mass - 0.4498713
    if 123.2919 <= Q.sd_mass < 154.5947:
        z += -0.0002682934 * Q.sd_mass + 0.1666852
    if Q.sd_mass >= 154.5947:
        z += -0.001906838 * Q.sd_mass + 0.4199955
    if Q.sum_z_dr2_top2 < 0.072817:
        z += 0.4240261 * Q.sum_z_dr2_top2 - 0.03087631
    if Q.sj3_mass1 >= 44.1303:
        z += -0.005407052 * Q.sj3_mass1 + 0.2386148
    if Q.lep_z < 0.3396572 and Q.mass_top30 < 162.7874:
        z += 0.001951245 * (0.3396572 - Q.lep_z) * (162.7874 - Q.mass_top30)
    if Q.lep_z < 0.3396572 and Q.tau32 < 0.5871757:
        z += 1.405624 * (0.3396572 - Q.lep_z) * (0.5871757 - Q.tau32)
    if Q.n_s3d_above_3 < 10.0 and Q.lep_iso < 0.4381892:
        z += 0.02038389 * (10.0 - Q.n_s3d_above_3) * (0.4381892 - Q.lep_iso)
    if Q.mass_top50 < 161.1264 and Q.tau21 < 0.6651974:
        z += 0.006260723 * (161.1264 - Q.mass_top50) * (0.6651974 - Q.tau21)
    if Q.lep_ptrel < 27.3236 and Q.N2_b2 < 0.2392833:
        z += -0.05294814 * (27.3236 - Q.lep_ptrel) * (0.2392833 - Q.N2_b2)
    if Q.mass_top50 < 161.1264 and Q.lep_dr < 0.114659:
        z += 0.02417421 * (161.1264 - Q.mass_top50) * (0.114659 - Q.lep_dr)
    if Q.mass_top50 < 161.1264 and Q.n_neutral > 8.0:
        z += -4.7401e-05 * (161.1264 - Q.mass_top50) * (Q.n_neutral - 8.0)
    if Q.mass_top50 < 161.1264 and Q.M2 > 0.05353765:
        z += 0.01669248 * (161.1264 - Q.mass_top50) * (Q.M2 - 0.05353765)
    if Q.mass_displaced5 < 32.61679 and Q.lep_dr > 0.1864963:
        z += 0.01282214 * (32.61679 - Q.mass_displaced5) * (Q.lep_dr - 0.1864963)
    if Q.mass < 117.4867 and Q.C2_b2 < 0.221369:
        z += -0.06921872 * (117.4867 - Q.mass) * (0.221369 - Q.C2_b2)
    if Q.lep_ptrel < 27.3236 and Q.z_top50_slots < 1.0:
        z += 0.2768502 * (27.3236 - Q.lep_ptrel) * (1.0 - Q.z_top50_slots)
    if Q.n_s3d_above_3 < 10.0 and Q.z_photon > 0.05998812:
        z += 0.02176008 * (10.0 - Q.n_s3d_above_3) * (Q.z_photon - 0.05998812)
    if Q.z_displaced5 > 0.06245248 and Q.z_charged_had > 0.1891627:
        z += 3.897449 * (Q.z_displaced5 - 0.06245248) * (Q.z_charged_had - 0.1891627)
    if Q.sj3_dr12 < 0.3273298 and Q.ecf_g42 > 1.874473e-05:
        z += 3820.984 * (0.3273298 - Q.sj3_dr12) * (Q.ecf_g42 - 1.874473e-05)
    if Q.sj3_pair_mass_min > 44.1643 and Q.n_lund_kt_above_5 < 4.0:
        z += 0.001850219 * (Q.sj3_pair_mass_min - 44.1643) * (4.0 - Q.n_lund_kt_above_5)
    if Q.sj3_pair_mass_min > 44.1643 and Q.lne_10 < 3.707456:
        z += -0.001463175 * (Q.sj3_pair_mass_min - 44.1643) * (3.707456 - Q.lne_10)
    if Q.lep_z < 0.221436 and Q.lnerel_49 < -5.474085:
        z += 0.02269081 * (0.221436 - Q.lep_z) * (-5.474085 - Q.lnerel_49)
    if Q.n_s3d_above_3 < 10.0 and Q.lne_0 > 4.727941:
        z += 0.003094949 * (10.0 - Q.n_s3d_above_3) * (Q.lne_0 - 4.727941)
    if Q.sj3_mass2 < 4.326415 and Q.z_dr_0p05_0p1 < 0.1405311:
        z += -0.3969719 * (4.326415 - Q.sj3_mass2) * (0.1405311 - Q.z_dr_0p05_0p1)
    if Q.z_displaced5 > 0.06245248 and Q.lnpt_11 > 1.305478:
        z += -0.8143826 * (Q.z_displaced5 - 0.06245248) * (Q.lnpt_11 - 1.305478)
    if Q.lep_ptrel < 27.3236 and Q.D3_b2 > 0.4675349:
        z += -0.003130764 * (27.3236 - Q.lep_ptrel) * (Q.D3_b2 - 0.4675349)
    if Q.z_displaced5 > 0.06245248 and Q.dr_3 > 0.1990664:
        z += 1.914459 * (Q.z_displaced5 - 0.06245248) * (Q.dr_3 - 0.1990664)
    if Q.lep_iso < 0.4381892 and Q.lund3_lnkt < 3.058345:
        z += -0.01185091 * (0.4381892 - Q.lep_iso) * (3.058345 - Q.lund3_lnkt)
    if Q.dr01 < 0.0856189 and Q.n_muon > 0.0:
        z += -1.327799 * (0.0856189 - Q.dr01) * (Q.n_muon - 0.0)
    if Q.lep_ptrel < 27.3236 and Q.n_dr_0p2_0p4 < 24.0:
        z += -0.000429307 * (27.3236 - Q.lep_ptrel) * (24.0 - Q.n_dr_0p2_0p4)
    if Q.mass_charged < 68.20711 and Q.mass_2photon > 5.763861:
        z += 5.035637e-06 * (68.20711 - Q.mass_charged) * (Q.mass_2photon - 5.763861)
    if Q.max_abs_d0 < 10.52344 and Q.tdz_41 > 0.03879887:
        z += 0.0001989138 * (10.52344 - Q.max_abs_d0) * (Q.tdz_41 - 0.03879887)
    if Q.lep_z < 0.3396572 and Q.tdz_0 > 0.04233308:
        z += 8.474531 * (0.3396572 - Q.lep_z) * (Q.tdz_0 - 0.04233308)
    if Q.C2_b2 < 0.1049877 and Q.tdz_0 < 0.02747645:
        z += -5.907439 * (0.1049877 - Q.C2_b2) * (0.02747645 - Q.tdz_0)
    if Q.z_displaced5 > 0.06245248 and Q.iselectron_29 > 0.0:
        z += 0.4598007 * (Q.z_displaced5 - 0.06245248) * (Q.iselectron_29 - 0.0)
    if Q.n_s3d_above_3 > 4.0 and Q.C3_b2 < 2.246622e-05:
        z += -1082.068 * (Q.n_s3d_above_3 - 4.0) * (2.246622e-05 - Q.C3_b2)
    if Q.lep_ptrel < 27.3236 and Q.eta_14 > 0.1429443:
        z += -0.003784482 * (27.3236 - Q.lep_ptrel) * (Q.eta_14 - 0.1429443)
    return z


def neuron_71(Q):
    z = 3.029693e-06
    return z


def neuron_72(Q):
    z = -1.164101e-06
    return z


def neuron_73(Q):
    z = -1.699636e-05
    return z


def neuron_74(Q):
    z = -4.493413e-06
    return z


def neuron_75(Q):
    z = 1.973831e-06
    return z


def neuron_76(Q):
    z = -5.542784e-06
    return z


def neuron_77(Q):
    z = -3.087468e-06
    return z


def neuron_78(Q):
    z = -1.128354
    if Q.lep_z < 0.221436:
        z += 0.5644352 * Q.lep_z - 0.2662622
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += 1.195013 * Q.lep_z - 0.4058949
    if Q.n_s3d_above_3 < 4.0:
        z += -0.2070553 * Q.n_s3d_above_3 + 1.347266
    if 4.0 <= Q.n_s3d_above_3 < 6.0:
        z += -0.1805859 * Q.n_s3d_above_3 + 1.241388
    if 6.0 <= Q.n_s3d_above_3 < 10.0:
        z += 0.000235904 * Q.n_s3d_above_3 + 0.1564574
    if Q.n_s3d_above_3 >= 10.0:
        z += 0.02646941 * Q.n_s3d_above_3 - 0.1058777
    if Q.mass_top30 < 67.20576:
        z += 0.01496561 * Q.mass_top30 - 1.005775
    if Q.mass_top40 >= 173.1022:
        z += 0.000556558 * Q.mass_top40 - 0.0963414
    if Q.lep_ptrel < 12.15228:
        z += 0.05975067 * Q.lep_ptrel - 0.7949107
    if 12.15228 <= Q.lep_ptrel < 18.7678:
        z += 0.01976142 * Q.lep_ptrel - 0.30895
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += 0.007704196 * Q.lep_ptrel - 0.08266251
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += -0.008048456 * Q.lep_ptrel + 0.3477567
    if Q.mass_displaced5 < 6.387683:
        z += 0.002126712 * Q.mass_displaced5 - 0.0693665
    if 6.387683 <= Q.mass_displaced5 < 32.61679:
        z += -0.00236385 * Q.mass_displaced5 - 0.04068221
    if Q.mass_displaced5 >= 32.61679:
        z += -0.004490562 * Q.mass_displaced5 + 0.02868428
    if 78.76797 <= Q.sj4_pair_mass_max < 115.5142:
        z += 0.004910634 * Q.sj4_pair_mass_max - 0.3868007
    if Q.sj4_pair_mass_max >= 115.5142:
        z += 0.0002794634 * Q.sj4_pair_mass_max + 0.1481653
    if Q.D3_b2 < 0.3264446:
        z += -0.3215734 * Q.D3_b2 + 0.1049759
    if Q.n_neutral < 23.0:
        z += 0.00312337 * Q.n_neutral - 0.0718375
    if Q.tau32 < 0.6206221:
        z += -0.4355151 * Q.tau32 + 0.2702903
    if Q.z_displaced5 >= 0.1709091:
        z += 0.1687561 * Q.z_displaced5 - 0.02884195
    z += -0.04348786 * Q.lne_0
    if Q.max_abs_d0 < 10.52344:
        z += 0.0160029 * Q.max_abs_d0 - 0.1684056
    if Q.pair_max_lnm2 >= 7.347625:
        z += -0.3782401 * Q.pair_max_lnm2 + 2.779166
    if Q.lund2_lndelta >= -1.556018:
        z += 0.009425119 * Q.lund2_lndelta + 0.01466566
    if Q.n_sd0_above_3 >= 2.0:
        z += 0.01778413 * Q.n_sd0_above_3 - 0.03556826
    if Q.mass_top15 >= 80.3877:
        z += 0.00156088 * Q.mass_top15 - 0.1254756
    if Q.lep_iso < 0.4381892:
        z += -0.9963383 * Q.lep_iso + 0.1818821
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += 0.2756805 * Q.lep_iso - 0.3755028
    if Q.n_lepton < 1.0:
        z += 0.06542914 * Q.n_lepton - 0.08178381
    if 1.0 <= Q.n_lepton < 2.0:
        z += 0.01635467 * Q.n_lepton - 0.03270935
    if 122.0585 <= Q.mass_top50 < 161.1264:
        z += 0.002528104 * Q.mass_top50 - 0.3085767
    if Q.mass_top50 >= 161.1264:
        z += 0.0001769315 * Q.mass_top50 + 0.07025926
    if Q.mass_top10 >= 103.4976:
        z += 0.0002431428 * Q.mass_top10 - 0.02516471
    if Q.n_pairs_kt_above_3 < 99.0:
        z += -0.001396666 * Q.n_pairs_kt_above_3 + 0.1382699
    if Q.n_pairs_kt_above_3 >= 220.0:
        z += 0.001496347 * Q.n_pairs_kt_above_3 - 0.3291964
    if Q.n_electron >= 1.0:
        z += -0.07145352 * Q.n_electron + 0.07145352
    if Q.mass_charged >= 37.42054:
        z += 0.001879103 * Q.mass_charged - 0.07031705
    if Q.mass < 71.96396:
        z += -0.01547123 * Q.mass + 1.113371
    if 127.2317 <= Q.mass < 164.4374:
        z += 0.008730137 * Q.mass - 1.11075
    if Q.mass >= 164.4374:
        z += -0.006914037 * Q.mass + 1.461738
    if Q.psi_0p1 >= 0.9356675:
        z += -3.058526 * Q.psi_0p1 + 2.861764
    if Q.n_pairs_kt_above_1 < 411.0:
        z += -0.001152316 * Q.n_pairs_kt_above_1 + 0.4736018
    z += 0.03800647 * Q.n_charged_had
    if Q.sum_e >= 550.9555:
        z += -0.0001519338 * Q.sum_e + 0.08370874
    if Q.D2_b05 < 1.102602:
        z += -2.467718 * Q.D2_b05 + 2.720911
    if Q.e3 < 0.0007393085:
        z += -18.14033 * Q.e3 + 0.0134113
    if Q.tau3 < 0.1024935:
        z += -6.01844 * Q.tau3 + 0.6168508
    if Q.mass_top5 >= 54.36876:
        z += 0.006184754 * Q.mass_top5 - 0.3362574
    if Q.sj2_mass1 < 21.21062:
        z += -0.01244758 * Q.sj2_mass1 + 0.2640209
    if Q.mass_displaced3 < 13.03663:
        z += -0.00314072 * Q.mass_displaced3 + 0.04094441
    if Q.mass_displaced3 >= 39.09615:
        z += 0.004728221 * Q.mass_displaced3 - 0.1848553
    if Q.z_displaced3 >= 0.02600452:
        z += -0.7010326 * Q.z_displaced3 + 0.01823002
    if Q.sip_3d_3 < 4.606241:
        z += -0.03512703 * Q.sip_3d_3 + 0.1618035
    if Q.sj4_dr_min < 0.07985021:
        z += 0.3333802 * Q.sj4_dr_min - 0.02662048
    if Q.z_charged < 0.6378426:
        z += 0.2553846 * Q.z_charged - 0.1628951
    if Q.D2 < 2.255345:
        z += -0.03002893 * Q.D2 + 0.0677256
    if Q.n_dr_0p1_0p2 < 11.0:
        z += -0.01641883 * Q.n_dr_0p1_0p2 + 0.1806072
    if Q.sj3_dr23 < 0.2458451:
        z += -0.1520937 * Q.sj3_dr23 + 0.0373915
    if Q.n_dr_0_0p05 < 2.0:
        z += -0.02433662 * Q.n_dr_0_0p05 + 0.04867324
    if Q.z_neutral_had >= 0.3788785:
        z += -0.8044595 * Q.z_neutral_had + 0.3047924
    if Q.M2 >= 0.04276413:
        z += 5.607822 * Q.M2 - 0.2398136
    if Q.z_photon >= 0.3318968:
        z += 1.008381 * Q.z_photon - 0.3346785
    if Q.n_sd0_above_5 >= 4.0:
        z += 0.04519453 * Q.n_sd0_above_5 - 0.1807781
    if Q.lead_ch_sdz < 1.104348:
        z += 0.0002206671 * Q.lead_ch_sdz - 0.0002436933
    if Q.psi_0p2 < 0.8009208:
        z += -0.1752689 * Q.psi_0p2 + 0.1403765
    if Q.lep_z < 0.3396572 and Q.mass_top20 > 86.78877:
        z += 0.006012334 * (0.3396572 - Q.lep_z) * (Q.mass_top20 - 86.78877)
    if Q.lep_z < 0.3396572 and Q.M2_b05 < 0.177312:
        z += 18.08765 * (0.3396572 - Q.lep_z) * (0.177312 - Q.M2_b05)
    if Q.n_s3d_above_3 > 4.0 and Q.sj3_dr_min < 0.3190414:
        z += 0.06401186 * (Q.n_s3d_above_3 - 4.0) * (0.3190414 - Q.sj3_dr_min)
    if Q.lep_z < 0.3396572 and Q.pair_mean_lnz < -1.332584:
        z += -0.2986241 * (0.3396572 - Q.lep_z) * (-1.332584 - Q.pair_mean_lnz)
    if Q.n_s3d_above_3 > 4.0 and Q.tau21 < 0.5440886:
        z += -0.205828 * (Q.n_s3d_above_3 - 4.0) * (0.5440886 - Q.tau21)
    if Q.mass_displaced5 > 6.387683 and Q.n_lepton < 1.0:
        z += -0.002421 * (Q.mass_displaced5 - 6.387683) * (1.0 - Q.n_lepton)
    if Q.n_s3d_above_3 > 4.0 and Q.sip_3d_3 < 175.9957:
        z += 0.0006343441 * (Q.n_s3d_above_3 - 4.0) * (175.9957 - Q.sip_3d_3)
    if Q.lep_z < 0.3396572 and Q.pair_mean_lnkt > -0.0709631:
        z += -0.6594323 * (0.3396572 - Q.lep_z) * (Q.pair_mean_lnkt - -0.0709631)
    if Q.lep_z < 0.3396572 and Q.sj4_dr_min < 0.1631992:
        z += 1.704591 * (0.3396572 - Q.lep_z) * (0.1631992 - Q.sj4_dr_min)
    if Q.lep_ptrel < 27.3236 and Q.psi_0p2 < 0.8436463:
        z += -0.005346568 * (27.3236 - Q.lep_ptrel) * (0.8436463 - Q.psi_0p2)
    if Q.mass_displaced5 > 6.387683 and Q.sip_3d_3 < 11.13866:
        z += 0.002331144 * (Q.mass_displaced5 - 6.387683) * (11.13866 - Q.sip_3d_3)
    if Q.lep_z < 0.3396572 and Q.ecf_g42 < 8.17233e-05:
        z += 385.2264 * (0.3396572 - Q.lep_z) * (8.17233e-05 - Q.ecf_g42)
    if Q.mass_displaced5 > 6.387683 and Q.psi_0p3 < 0.8156512:
        z += -0.00157211 * (Q.mass_displaced5 - 6.387683) * (0.8156512 - Q.psi_0p3)
    if Q.max_abs_d0 < 10.52344 and Q.sip_3d_3 < 23.30856:
        z += -8.341004e-05 * (10.52344 - Q.max_abs_d0) * (23.30856 - Q.sip_3d_3)
    if Q.mass_displaced5 > 6.387683 and Q.lep_iso < 14.38858:
        z += 0.000350803 * (Q.mass_displaced5 - 6.387683) * (14.38858 - Q.lep_iso)
    if Q.n_s3d_above_3 < 10.0 and Q.sj3_dr_min > 0.08335692:
        z += -0.05496804 * (10.0 - Q.n_s3d_above_3) * (Q.sj3_dr_min - 0.08335692)
    if Q.lep_ptrel < 18.7678 and Q.z_dr_0p1_0p2 < 0.5699805:
        z += 0.003833035 * (18.7678 - Q.lep_ptrel) * (0.5699805 - Q.z_dr_0p1_0p2)
    if Q.mass_top40 > 173.1022 and Q.n_muon < 1.0:
        z += -8.341431e-05 * (Q.mass_top40 - 173.1022) * (1.0 - Q.n_muon)
    if Q.n_s3d_above_3 < 10.0 and Q.dzerr_0 < 0.03430176:
        z += -0.03395537 * (10.0 - Q.n_s3d_above_3) * (0.03430176 - Q.dzerr_0)
    if Q.lep_ptrel < 27.3236 and Q.eta_28 < -0.210083:
        z += 0.003206628 * (27.3236 - Q.lep_ptrel) * (-0.210083 - Q.eta_28)
    if Q.lep_ptrel < 18.7678 and Q.n_dr_0p4_up < 15.0:
        z += -0.0003361296 * (18.7678 - Q.lep_ptrel) * (15.0 - Q.n_dr_0p4_up)
    if Q.n_electron > 1.0 and Q.iselectron_30 > 0.0:
        z += -0.02862715 * (Q.n_electron - 1.0) * (Q.iselectron_30 - 0.0)
    if Q.lep_iso < 0.4381892 and Q.sip_3d_3 < 11.13866:
        z += -0.03469815 * (0.4381892 - Q.lep_iso) * (11.13866 - Q.sip_3d_3)
    if Q.z_displaced3 > 0.02600452 and Q.isphoton_14 > 0.0:
        z += 0.03161833 * (Q.z_displaced3 - 0.02600452) * (Q.isphoton_14 - 0.0)
    if Q.mass_displaced3 > 39.09615 and Q.sip_3d_3 < 33.08364:
        z += 0.0003378683 * (Q.mass_displaced3 - 39.09615) * (33.08364 - Q.sip_3d_3)
    if Q.tau3 < 0.1024935 and Q.tdz_33 > -0.01794241:
        z += -0.4910573 * (0.1024935 - Q.tau3) * (Q.tdz_33 - -0.01794241)
    if Q.lead_ch_sdz < 1.104348 and Q.iselectron_1 > 0.0:
        z += -0.007920797 * (1.104348 - Q.lead_ch_sdz) * (Q.iselectron_1 - 0.0)
    if Q.n_dr_0_0p05 < 2.0 and Q.isnhad_27 < 1.0:
        z += -0.01999355 * (2.0 - Q.n_dr_0_0p05) * (1.0 - Q.isnhad_27)
    if Q.n_lepton < 2.0 and Q.sj3_pairmax_over_m < 0.9333327:
        z += -0.437169 * (2.0 - Q.n_lepton) * (0.9333327 - Q.sj3_pairmax_over_m)
    if Q.tau32 < 0.6206221 and Q.jet_charge_k03 > 0.02942741:
        z += -0.4405599 * (0.6206221 - Q.tau32) * (Q.jet_charge_k03 - 0.02942741)
    if Q.n_sd0_above_5 > 4.0 and Q.sip_3d_3 < 11.13866:
        z += -0.06518377 * (Q.n_sd0_above_5 - 4.0) * (11.13866 - Q.sip_3d_3)
    if Q.tau3 < 0.1024935 and Q.sip_3d_3 < 577.991:
        z += -0.004015356 * (0.1024935 - Q.tau3) * (577.991 - Q.sip_3d_3)
    if Q.sj2_mass1 < 21.21062 and Q.phi_26 < 0.04891968:
        z += -0.02405847 * (21.21062 - Q.sj2_mass1) * (0.04891968 - Q.phi_26)
    if Q.sum_e > 550.9555 and Q.ismuon_4 > 0.0:
        z += 6.676768e-05 * (Q.sum_e - 550.9555) * (Q.ismuon_4 - 0.0)
    return z


def neuron_79(Q):
    z = -3.602393e-05
    return z


def neuron_80(Q):
    z = 9.707704e-06
    return z


def neuron_81(Q):
    z = -0.4883434
    if Q.z_displaced3 < 0.03624058:
        z += 2.438482 * Q.z_displaced3 - 0.2588638
    if 0.03624058 <= Q.z_displaced3 < 0.1061578:
        z += 3.455886 * Q.z_displaced3 - 0.2957352
    if Q.z_displaced3 >= 0.1061578:
        z += 1.017405 * Q.z_displaced3 - 0.03687133
    if Q.n_pairs_kt_above_3 < 22.0:
        z += 0.001657186 * Q.n_pairs_kt_above_3 - 0.0364581
    if Q.n_s3d_above_3 < 3.0:
        z += -0.07726739 * Q.n_s3d_above_3 + 0.386337
    if 3.0 <= Q.n_s3d_above_3 < 5.0:
        z += 0.01301911 * Q.n_s3d_above_3 + 0.1154775
    if 5.0 <= Q.n_s3d_above_3 < 10.0:
        z += 0.0902865 * Q.n_s3d_above_3 - 0.2708595
    if Q.n_s3d_above_3 >= 10.0:
        z += 0.1594886 * Q.n_s3d_above_3 - 0.962881
    z += -3.384651 * Q.M2
    if 97.12186 <= Q.mass_top40 < 119.1279:
        z += 0.01579149 * Q.mass_top40 - 1.533699
    if Q.mass_top40 >= 119.1279:
        z += -0.0006464273 * Q.mass_top40 + 0.4245154
    if 99.54528 <= Q.mass_top50 < 115.614:
        z += -0.01474483 * Q.mass_top50 + 1.467779
    if 115.614 <= Q.mass_top50 < 125.4927:
        z += 0.004014832 * Q.mass_top50 - 0.7011013
    if 125.4927 <= Q.mass_top50 < 161.1264:
        z += 0.01068264 * Q.mass_top50 - 1.537863
    if Q.mass_top50 >= 161.1264:
        z += 0.004038759 * Q.mass_top50 - 0.4673575
    if Q.LHA < 0.4242439:
        z += -3.419608 * Q.LHA + 1.450748
    if Q.max_abs_d0 < 5.8125:
        z += -0.02873067 * Q.max_abs_d0 + 0.1478874
    if 5.8125 <= Q.max_abs_d0 < 10.52344:
        z += 0.004056435 * Q.max_abs_d0 - 0.04268764
    if Q.mass_top30 < 52.72207:
        z += -0.02506981 * Q.mass_top30 + 1.321732
    if Q.max_dr < 0.7634316:
        z += 0.3511431 * Q.max_dr - 0.2680738
    if Q.sip_3d_3 < 2.723325:
        z += -0.07881601 * Q.sip_3d_3 + 0.2385931
    if 2.723325 <= Q.sip_3d_3 < 577.991:
        z += -4.163546e-05 * Q.sip_3d_3 + 0.02406492
    if Q.n_sd0_above_3 < 8.0:
        z += 0.005886855 * Q.n_sd0_above_3 - 0.04709484
    if Q.n_sd0_above_3 >= 9.0:
        z += -0.05050447 * Q.n_sd0_above_3 + 0.4545402
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.00506771 * Q.n_pairs_kt_above_1 - 0.4054168
    if Q.z_dr_0p4_up < 0.01181938:
        z += 2.128947 * Q.z_dr_0p4_up - 0.02516283
    if Q.jet_abs_eta < 1.203438:
        z += -0.07371838 * Q.jet_abs_eta + 0.08871552
    if Q.tdz_1 < -0.1170754:
        z += -0.1955241 * Q.tdz_1 - 0.02289107
    if Q.lep_ptrel < 6.983043:
        z += -0.0213297 * Q.lep_ptrel + 0.1489462
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.02119763 * Q.lep_ptrel - 0.5791956
    if Q.lep_ptrel >= 43.20788:
        z += 0.01503376 * Q.lep_ptrel - 0.3128681
    if Q.sum_z_dr < 0.07290954:
        z += -6.639492 * Q.sum_z_dr + 0.4840823
    if Q.n_dr_0p4_up < 15.0:
        z += 0.01249955 * Q.n_dr_0p4_up - 0.1874933
    if Q.tau2 < 0.1076451:
        z += 5.439039 * Q.tau2 - 0.5854859
    if Q.ecf_g32 < 0.0009010251:
        z += 269.1409 * Q.ecf_g32 - 0.2425027
    if Q.mass_displaced3 < 1.777286:
        z += 0.1183333 * Q.mass_displaced3 - 0.210312
    if Q.mass_displaced5 < 13.97388:
        z += -0.02401802 * Q.mass_displaced5 + 0.3356249
    if Q.tau21 < 0.2377332:
        z += -2.091701 * Q.tau21 + 0.4972668
    if Q.lep_z < 0.221436:
        z += -0.5841844 * Q.lep_z + 0.1293595
    if Q.pair_mean_lndelta >= -1.352792:
        z += 1.153418 * Q.pair_mean_lndelta + 1.560335
    if Q.mass_2charged >= 20.76537:
        z += 0.002079686 * Q.mass_2charged - 0.04318546
    if Q.sj4_pair_mass_max >= 63.34656:
        z += -0.003649218 * Q.sj4_pair_mass_max + 0.2311654
    if Q.n_s3d_above_10 >= 3.0:
        z += 0.0785355 * Q.n_s3d_above_10 - 0.2356065
    if Q.n_sd0_above_5 >= 5.0:
        z += -0.09798046 * Q.n_sd0_above_5 + 0.4899023
    if Q.mass_charged >= 99.20396:
        z += 0.002788874 * Q.mass_charged - 0.2766674
    if Q.z_electron < 0.01329067:
        z += 8.473199 * Q.z_electron - 0.1126145
    if Q.sj3_pairmin_over_m >= 0.2477126:
        z += -0.08486115 * Q.sj3_pairmin_over_m + 0.02102118
    if Q.z_displaced5 >= 0.01810676:
        z += -0.02229503 * Q.z_displaced5 + 0.0004036909
    if Q.min_pair_mass < 1.763283:
        z += -0.01793507 * Q.min_pair_mass + 0.03162461
    if Q.sip_3d_2 < 70.05844:
        z += 0.0001715859 * Q.sip_3d_2 - 0.04473294
    if 70.05844 <= Q.sip_3d_2 < 226.3008:
        z += 0.0002093664 * Q.sip_3d_2 - 0.04737979
    if Q.tdz_2 >= -0.02221619:
        z += 0.1195952 * Q.tdz_2 + 0.002656951
    if Q.tau4 < 0.02949822:
        z += 3.060999 * Q.tau4 - 0.09029404
    if Q.n_s3d_above_3 > 3.0 and Q.sip_3d_2 < 447.0873:
        z += 0.0001515893 * (Q.n_s3d_above_3 - 3.0) * (447.0873 - Q.sip_3d_2)
    if Q.z_displaced3 > 0.03624058 and Q.z_charged_had > 0.265564:
        z += -4.109592 * (Q.z_displaced3 - 0.03624058) * (Q.z_charged_had - 0.265564)
    if Q.z_displaced3 > 0.03624058 and Q.sip_3d_3 < 16.23838:
        z += 0.06157491 * (Q.z_displaced3 - 0.03624058) * (16.23838 - Q.sip_3d_3)
    if Q.n_s3d_above_3 > 3.0 and Q.tau32 < 0.8882532:
        z += -0.07778744 * (Q.n_s3d_above_3 - 3.0) * (0.8882532 - Q.tau32)
    if Q.max_abs_d0 < 10.52344 and Q.sip_3d_3 < 7.345216:
        z += -0.005605325 * (10.52344 - Q.max_abs_d0) * (7.345216 - Q.sip_3d_3)
    if Q.n_s3d_above_3 > 3.0 and Q.n_charged_had < 33.0:
        z += 0.003994873 * (Q.n_s3d_above_3 - 3.0) * (33.0 - Q.n_charged_had)
    if Q.n_pairs_kt_above_3 < 22.0 and Q.n_pairs_kt_above_30 < 1.0:
        z += -0.01726219 * (22.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.n_pairs_kt_above_30)
    if Q.n_s3d_above_3 > 3.0 and Q.tau43 < 0.8597199:
        z += -0.1318583 * (Q.n_s3d_above_3 - 3.0) * (0.8597199 - Q.tau43)
    if Q.z_displaced3 > 0.03624058 and Q.lep_dr < 0.3014662:
        z += -3.165504 * (Q.z_displaced3 - 0.03624058) * (0.3014662 - Q.lep_dr)
    if Q.n_s3d_above_3 > 3.0 and Q.mass_displaced5 < 21.12768:
        z += -0.004235206 * (Q.n_s3d_above_3 - 3.0) * (21.12768 - Q.mass_displaced5)
    if Q.n_s3d_above_3 > 3.0 and Q.tau54 < 0.9036234:
        z += -0.1268343 * (Q.n_s3d_above_3 - 3.0) * (0.9036234 - Q.tau54)
    if Q.LHA < 0.4242439 and Q.n_lund < 8.0:
        z += -0.02559186 * (0.4242439 - Q.LHA) * (8.0 - Q.n_lund)
    if Q.z_displaced3 > 0.03624058 and Q.mass_2charged < 1.839882:
        z += -0.7710237 * (Q.z_displaced3 - 0.03624058) * (1.839882 - Q.mass_2charged)
    if Q.n_s3d_above_3 < 5.0 and Q.eccentricity > 0.5174679:
        z += -0.001958409 * (5.0 - Q.n_s3d_above_3) * (Q.eccentricity - 0.5174679)
    if Q.z_displaced3 < 0.1061578 and Q.lep_z < 0.03021637:
        z += 184.5059 * (0.1061578 - Q.z_displaced3) * (0.03021637 - Q.lep_z)
    if Q.z_displaced3 > 0.03624058 and Q.z_photon > 0.3887278:
        z += 7.126993 * (Q.z_displaced3 - 0.03624058) * (Q.z_photon - 0.3887278)
    if Q.z_displaced3 > 0.03624058 and Q.sj2_dr > 0.2403736:
        z += 0.3648713 * (Q.z_displaced3 - 0.03624058) * (Q.sj2_dr - 0.2403736)
    if Q.n_s3d_above_3 > 3.0 and Q.n_lund_kt_above_5 > 2.0:
        z += -0.003508492 * (Q.n_s3d_above_3 - 3.0) * (Q.n_lund_kt_above_5 - 2.0)
    if Q.tau2 < 0.1076451 and Q.n_muon > 0.0:
        z += 0.5228104 * (0.1076451 - Q.tau2) * (Q.n_muon - 0.0)
    if Q.z_displaced3 > 0.03624058 and Q.z_muon > 0.03898651:
        z += 0.9420068 * (Q.z_displaced3 - 0.03624058) * (Q.z_muon - 0.03898651)
    if Q.z_displaced3 > 0.03624058 and Q.n_lund > 5.0:
        z += 0.06014077 * (Q.z_displaced3 - 0.03624058) * (Q.n_lund - 5.0)
    if Q.mass_top40 > 97.12186 and Q.tau43_b2 > 0.6509029:
        z += -0.001183028 * (Q.mass_top40 - 97.12186) * (Q.tau43_b2 - 0.6509029)
    if Q.n_s3d_above_3 > 3.0 and Q.isnhad_34 > 0.0:
        z += 0.0004861582 * (Q.n_s3d_above_3 - 3.0) * (Q.isnhad_34 - 0.0)
    if Q.n_s3d_above_3 > 3.0 and Q.pt1_over_pt0 < 0.2477181:
        z += 0.252106 * (Q.n_s3d_above_3 - 3.0) * (0.2477181 - Q.pt1_over_pt0)
    if Q.tau21 < 0.2377332 and Q.pt_balance01 < 0.1985369:
        z += 26.56529 * (0.2377332 - Q.tau21) * (0.1985369 - Q.pt_balance01)
    if Q.n_sd0_above_3 > 9.0 and Q.tdz_47 < 0.0:
        z += -0.01859276 * (Q.n_sd0_above_3 - 9.0) * (0.0 - Q.tdz_47)
    if Q.lep_ptrel < 6.983043 and Q.ismuon_73 > 0.0:
        z += -0.1764805 * (6.983043 - Q.lep_ptrel) * (Q.ismuon_73 - 0.0)
    if Q.lep_ptrel < 6.983043 and Q.ismuon_11 > 0.0:
        z += -0.01076872 * (6.983043 - Q.lep_ptrel) * (Q.ismuon_11 - 0.0)
    if Q.z_displaced3 > 0.03624058 and Q.phi_79 > 0.0:
        z += -0.03559788 * (Q.z_displaced3 - 0.03624058) * (Q.phi_79 - 0.0)
    if Q.tau2 < 0.1076451 and Q.jet_charge_k03 > 0.2021837:
        z += -0.1982842 * (0.1076451 - Q.tau2) * (Q.jet_charge_k03 - 0.2021837)
    if Q.mass_top40 > 119.1279 and Q.isphoton_32 < 1.0:
        z += -0.000610994 * (Q.mass_top40 - 119.1279) * (1.0 - Q.isphoton_32)
    if Q.sj4_pair_mass_max > 63.34656 and Q.z_neutral_had < 0.3788785:
        z += 0.007740165 * (Q.sj4_pair_mass_max - 63.34656) * (0.3788785 - Q.z_neutral_had)
    if Q.sj3_pairmin_over_m > 0.2477126 and Q.dr_16 < 0.2877534:
        z += 0.05457512 * (Q.sj3_pairmin_over_m - 0.2477126) * (0.2877534 - Q.dr_16)
    if Q.min_pair_mass < 1.763283 and Q.isnhad_8 < 1.0:
        z += -0.002230798 * (1.763283 - Q.min_pair_mass) * (1.0 - Q.isnhad_8)
    if Q.n_s3d_above_3 > 10.0 and Q.isphoton_0 < 1.0:
        z += 0.06835898 * (Q.n_s3d_above_3 - 10.0) * (1.0 - Q.isphoton_0)
    if Q.sj4_pair_mass_max > 63.34656 and Q.eta_37 < 0.1695557:
        z += 0.0005729588 * (Q.sj4_pair_mass_max - 63.34656) * (0.1695557 - Q.eta_37)
    if Q.tdz_2 > -0.02221619 and Q.ismuon_12 > 0.0:
        z += -0.6376978 * (Q.tdz_2 - -0.02221619) * (Q.ismuon_12 - 0.0)
    if Q.max_dr < 0.7634316 and Q.ismuon_49 > 0.0:
        z += -3.079449 * (0.7634316 - Q.max_dr) * (Q.ismuon_49 - 0.0)
    if Q.lep_ptrel < 6.983043 and Q.pt_balance01 > 0.1985369:
        z += -0.04471415 * (6.983043 - Q.lep_ptrel) * (Q.pt_balance01 - 0.1985369)
    if Q.z_displaced3 > 0.03624058 and Q.td0_33 < 0.04032236:
        z += -0.7513919 * (Q.z_displaced3 - 0.03624058) * (0.04032236 - Q.td0_33)
    if Q.tdz_2 > -0.02221619 and Q.dzerr_44 < 0.0881958:
        z += -0.2390207 * (Q.tdz_2 - -0.02221619) * (0.0881958 - Q.dzerr_44)
    return z


def neuron_82(Q):
    z = -2.942086e-06
    return z


def neuron_83(Q):
    z = 1.000403
    if Q.mass_top40 < 78.33213:
        z += 0.02283167 * Q.mass_top40 - 3.952213
    if 78.33213 <= Q.mass_top40 < 119.1279:
        z += 0.01268991 * Q.mass_top40 - 3.157787
    if 119.1279 <= Q.mass_top40 < 155.8928:
        z += 0.01950332 * Q.mass_top40 - 3.969454
    if 155.8928 <= Q.mass_top40 < 173.1022:
        z += 0.04550724 * Q.mass_top40 - 8.023278
    if Q.mass_top40 >= 173.1022:
        z += 0.02267557 * Q.mass_top40 - 4.071065
    if Q.n_s3d_above_3 < 3.0:
        z += -0.3043619 * Q.n_s3d_above_3 + 2.152179
    if 3.0 <= Q.n_s3d_above_3 < 6.0:
        z += -0.1770133 * Q.n_s3d_above_3 + 1.770133
    if 6.0 <= Q.n_s3d_above_3 < 10.0:
        z += -0.04364917 * Q.n_s3d_above_3 + 0.9699484
    if Q.n_s3d_above_3 >= 10.0:
        z += 0.1333642 * Q.n_s3d_above_3 - 0.800185
    if Q.ecf_g42 < 2.448333e-05:
        z += 8382.729 * Q.ecf_g42 - 0.2052371
    if Q.z_displaced3 < 0.06416437:
        z += -6.001282 * Q.z_displaced3 + 0.5799025
    if 0.06416437 <= Q.z_displaced3 < 0.1681173:
        z += -1.874252 * Q.z_displaced3 + 0.3150942
    if Q.sum_zz_dr2 < 0.05018249:
        z += -12.09902 * Q.sum_zz_dr2 + 0.6071587
    if Q.e3 < 0.00228569:
        z += 1.06434 * Q.e3 - 0.002432751
    if Q.mass_top15 >= 80.3877:
        z += -0.01331404 * Q.mass_top15 + 1.070285
    z += 797.2196 * Q.ecf_g41
    if Q.pair_max_lnm2 >= 7.347625:
        z += 0.1868222 * Q.pair_max_lnm2 - 1.3727
    if Q.n_s3d_above_10 >= 1.0:
        z += 0.01619579 * Q.n_s3d_above_10 - 0.01619579
    if Q.sip_3d_2 < 4.636903:
        z += -0.1177243 * Q.sip_3d_2 + 0.1582363
    if 4.636903 <= Q.sip_3d_2 < 30.49226:
        z += 0.01034353 * Q.sip_3d_2 - 0.4356018
    if 30.49226 <= Q.sip_3d_2 < 226.3008:
        z += 0.0006138869 * Q.sip_3d_2 - 0.1389231
    if 0.04393457 <= Q.e2 < 0.07745967:
        z += -2.947475 * Q.e2 + 0.1294961
    if Q.e2 >= 0.07745967:
        z += -6.532436 * Q.e2 + 0.4071859
    if Q.mass_displaced3 < 1.251841:
        z += 0.2074783 * Q.mass_displaced3 - 0.0242153
    if 1.251841 <= Q.mass_displaced3 < 13.03663:
        z += -0.01998461 * Q.mass_displaced3 + 0.260532
    if Q.mass_neutral < 49.25971:
        z += 0.005718532 * Q.mass_neutral - 0.2816932
    if Q.mass_top30 >= 102.3652:
        z += -0.008143472 * Q.mass_top30 + 0.8336085
    if Q.N2_b05 < 0.4265629:
        z += 4.039013 * Q.N2_b05 - 1.722893
    if Q.N2_b05 >= 0.4518419:
        z += 3.33454 * Q.N2_b05 - 1.506685
    if Q.sj4_pair_mass_max < 72.862:
        z += 0.0002292236 * Q.sj4_pair_mass_max + 0.3466005
    if 72.862 <= Q.sj4_pair_mass_max < 115.5142:
        z += -0.006588024 * Q.sj4_pair_mass_max + 0.8433188
    if 115.5142 <= Q.sj4_pair_mass_max < 128.0079:
        z += 0.00459802 * Q.sj4_pair_mass_max - 0.4488281
    if Q.sj4_pair_mass_max >= 128.0079:
        z += 0.01118604 * Q.sj4_pair_mass_max - 1.292147
    if Q.M2 < 0.06228948:
        z += 6.85682 * Q.M2 - 0.4271078
    if 62.03827 <= Q.sd_mass < 83.61981:
        z += 0.01972985 * Q.sd_mass - 1.224006
    if 83.61981 <= Q.sd_mass < 88.81751:
        z += -0.02101146 * Q.sd_mass + 2.182775
    if 88.81751 <= Q.sd_mass < 154.5947:
        z += -0.006055376 * Q.sd_mass + 0.8544129
    if Q.sd_mass >= 154.5947:
        z += 0.01017015 * Q.sd_mass - 1.653967
    if Q.pt1_over_pt0 < 0.5779883:
        z += -0.02631499 * Q.pt1_over_pt0 + 0.01520976
    if Q.mass_top10 >= 103.4976:
        z += 0.001065734 * Q.mass_top10 - 0.110301
    if Q.mass < 90.08945:
        z += -0.01613326 * Q.mass + 1.942665
    if 90.08945 <= Q.mass < 117.4867:
        z += -0.01785689 * Q.mass + 2.097946
    if Q.tdz_1 < -0.1170754:
        z += -0.04573015 * Q.tdz_1 - 0.005353876
    if Q.e4_b05 < 0.000141779:
        z += 5827.365 * Q.e4_b05 - 0.8261981
    if Q.log_sum_pt < 6.661019:
        z += 0.3580026 * Q.log_sum_pt - 2.384662
    if Q.n_charged < 33.0:
        z += -0.00906607 * Q.n_charged + 0.2991803
    if Q.tau5 >= 0.01222366:
        z += 4.640745 * Q.tau5 - 0.05672691
    if Q.n_pt_above_5 < 30.0:
        z += 0.01386565 * Q.n_pt_above_5 - 0.4159694
    if Q.tau32 < 0.3951525:
        z += -1.388611 * Q.tau32 + 0.5487132
    if Q.lep_iso < 6.185635:
        z += -0.001004791 * Q.lep_iso + 0.006215272
    if Q.z_photon >= 0.4305934:
        z += 1.379388 * Q.z_photon - 0.5939552
    if Q.z_neutral_had >= 0.3788785:
        z += 1.18865 * Q.z_neutral_had - 0.4503539
    if Q.sj3_mass3 < 0.3886647:
        z += 0.02812589 * Q.sj3_mass3 - 0.01093154
    if Q.M3 < 0.01517988:
        z += 16.5453 * Q.M3 - 0.2511556
    if Q.n_sd0_above_5 >= 7.0:
        z += 0.08510623 * Q.n_sd0_above_5 - 0.5957436
    if Q.sj3_dr23 < 0.8799072:
        z += -0.1626843 * Q.sj3_dr23 + 0.1431471
    if Q.e3_b2 < 0.0002536827:
        z += 678.033 * Q.e3_b2 - 0.1720052
    if Q.n_lepton < 2.0:
        z += 0.0183949 * Q.n_lepton - 0.0367898
    if Q.z_charged_had < 0.3603262:
        z += -0.5177767 * Q.z_charged_had + 0.1865685
    if Q.C3_b05 < 0.2250047:
        z += 0.4593895 * Q.C3_b05 - 0.1033648
    if Q.psi_0p2 >= 0.8857951:
        z += -0.6075018 * Q.psi_0p2 + 0.5381221
    if Q.td0_3 >= 0.1091249:
        z += 0.2211785 * Q.td0_3 - 0.02413608
    if Q.dr_max_012 < 0.02210827:
        z += 7.923666 * Q.dr_max_012 - 0.1751785
    if Q.max_abs_d0 < 0.3977051:
        z += 0.1208865 * Q.max_abs_d0 - 0.1776853
    if 0.3977051 <= Q.max_abs_d0 < 10.52344:
        z += 0.01279988 * Q.max_abs_d0 - 0.1346987
    if Q.lund3_lndelta >= -2.817283:
        z += -0.1587178 * Q.lund3_lndelta - 0.4471529
    if Q.sj4_dr_min < 0.3112717:
        z += 0.3981875 * Q.sj4_dr_min - 0.1239445
    if Q.n_electron >= 1.0:
        z += 0.01054149 * Q.n_electron - 0.01054149
    if Q.n_sd0_above_10 < 7.0:
        z += 0.05315821 * Q.n_sd0_above_10 - 0.3721074
    if Q.lep_ptrel < 27.3236:
        z += -0.02340144 * Q.lep_ptrel + 0.6394117
    if Q.z_dr_0_0p05 < 0.004294711:
        z += 19.03427 * Q.z_dr_0_0p05 - 0.08174668
    if Q.pair_mean_lnm2 >= 1.96906:
        z += 0.1365908 * Q.pair_mean_lnm2 - 0.2689554
    if Q.pair_mean_lndelta >= -2.115543:
        z += -0.07521082 * Q.pair_mean_lndelta - 0.1591117
    if Q.n_lund_kt_above_5 >= 1.0:
        z += -0.03394051 * Q.n_lund_kt_above_5 + 0.03394051
    if Q.n_sd0_above_3 < 8.0:
        z += 0.01047362 * Q.n_sd0_above_3 - 0.083789
    if Q.mass_top40 < 173.1022 and Q.mass_displaced5 > 0.0:
        z += 6.994875e-06 * (173.1022 - Q.mass_top40) * (Q.mass_displaced5 - 0.0)
    if Q.mass_top40 > 78.33213 and Q.lep_ptrel < 18.7678:
        z += -0.0006205542 * (Q.mass_top40 - 78.33213) * (18.7678 - Q.lep_ptrel)
    if Q.ecf_g42 < 2.448333e-05 and Q.n_s3d_above_10 < 6.0:
        z += 1742.32 * (2.448333e-05 - Q.ecf_g42) * (6.0 - Q.n_s3d_above_10)
    if Q.n_s3d_above_3 > 6.0 and Q.sip_3d_3 < 175.9957:
        z += 0.001920806 * (Q.n_s3d_above_3 - 6.0) * (175.9957 - Q.sip_3d_3)
    if Q.sum_zz_dr2 < 0.05018249 and Q.n_lund < 16.0:
        z += -0.09979822 * (0.05018249 - Q.sum_zz_dr2) * (16.0 - Q.n_lund)
    if Q.z_displaced3 < 0.06416437 and Q.z_neutral_had < 0.2370407:
        z += 0.6103322 * (0.06416437 - Q.z_displaced3) * (0.2370407 - Q.z_neutral_had)
    if Q.n_s3d_above_10 > 1.0 and Q.sip_3d_1 < 351.3271:
        z += 0.0001988285 * (Q.n_s3d_above_10 - 1.0) * (351.3271 - Q.sip_3d_1)
    if Q.e3 < 0.00228569 and Q.jet_abs_eta > 0.9149342:
        z += 102.1006 * (0.00228569 - Q.e3) * (Q.jet_abs_eta - 0.9149342)
    if Q.n_s3d_above_3 > 6.0 and Q.sj3_pairmin_over_m > 0.1636952:
        z += -0.09548273 * (Q.n_s3d_above_3 - 6.0) * (Q.sj3_pairmin_over_m - 0.1636952)
    if Q.mass_top40 > 78.33213 and Q.D2_b2 < 15.17086:
        z += -0.0008264392 * (Q.mass_top40 - 78.33213) * (15.17086 - Q.D2_b2)
    if Q.mass_top40 < 173.1022 and Q.D2_b2 < 15.17086:
        z += 0.0001342977 * (173.1022 - Q.mass_top40) * (15.17086 - Q.D2_b2)
    if Q.n_s3d_above_3 < 10.0 and Q.z_neutral > 0.1384639:
        z += -0.1561186 * (10.0 - Q.n_s3d_above_3) * (Q.z_neutral - 0.1384639)
    if Q.n_s3d_above_3 > 6.0 and Q.z_dr_0p1_0p2 < 0.2949288:
        z += 0.07666942 * (Q.n_s3d_above_3 - 6.0) * (0.2949288 - Q.z_dr_0p1_0p2)
    if Q.n_s3d_above_10 > 1.0 and Q.D3_b2 < 0.3264446:
        z += 0.04116658 * (Q.n_s3d_above_10 - 1.0) * (0.3264446 - Q.D3_b2)
    if Q.mass < 117.4867 and Q.n_dr_0p1_0p2 > 1.0:
        z += 0.0002365793 * (117.4867 - Q.mass) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.mass_top15 > 80.3877 and Q.tdz_29 > -0.09486766:
        z += 0.0001994981 * (Q.mass_top15 - 80.3877) * (Q.tdz_29 - -0.09486766)
    if Q.n_s3d_above_3 > 6.0 and Q.ismuon_24 > 0.0:
        z += -0.015612 * (Q.n_s3d_above_3 - 6.0) * (Q.ismuon_24 - 0.0)
    if Q.n_s3d_above_3 > 6.0 and Q.phi_3 > 0.06884766:
        z += -0.09436817 * (Q.n_s3d_above_3 - 6.0) * (Q.phi_3 - 0.06884766)
    if Q.mass_top40 > 155.8928 and Q.lund2_lndelta < -1.411949:
        z += 0.00138967 * (Q.mass_top40 - 155.8928) * (-1.411949 - Q.lund2_lndelta)
    if Q.M2 < 0.06228948 and Q.eta_37 > 0.0:
        z += 19.33707 * (0.06228948 - Q.M2) * (Q.eta_37 - 0.0)
    if Q.n_s3d_above_10 > 1.0 and Q.dr_16 < 0.3788372:
        z += -0.02824233 * (Q.n_s3d_above_10 - 1.0) * (0.3788372 - Q.dr_16)
    if Q.z_displaced3 < 0.1681173 and Q.ismuon_2 > 0.0:
        z += -0.6189339 * (0.1681173 - Q.z_displaced3) * (Q.ismuon_2 - 0.0)
    if Q.e3_b2 < 0.0002536827 and Q.iselectron_12 > 0.0:
        z += -180.7083 * (0.0002536827 - Q.e3_b2) * (Q.iselectron_12 - 0.0)
    if Q.mass_top10 > 103.4976 and Q.mratio_min_012 > 0.0111128:
        z += -0.01056257 * (Q.mass_top10 - 103.4976) * (Q.mratio_min_012 - 0.0111128)
    if Q.e2 > 0.04393457 and Q.iselectron_11 > 0.0:
        z += -0.85515 * (Q.e2 - 0.04393457) * (Q.iselectron_11 - 0.0)
    if Q.sj4_dr_min < 0.3112717 and Q.td0_30 < 0.04589821:
        z += -0.09839339 * (0.3112717 - Q.sj4_dr_min) * (0.04589821 - Q.td0_30)
    if Q.n_s3d_above_10 > 1.0 and Q.ismuon_27 > 0.0:
        z += 0.004434182 * (Q.n_s3d_above_10 - 1.0) * (Q.ismuon_27 - 0.0)
    return z


def neuron_84(Q):
    z = -7.427524e-05
    return z


def neuron_85(Q):
    z = -8.185613e-07
    return z


def neuron_86(Q):
    z = -1.709903e-05
    return z


def neuron_87(Q):
    z = 1.61916e-06
    return z


def neuron_88(Q):
    z = 2.996807e-07
    return z


def neuron_89(Q):
    z = 1.735346e-06
    return z


def neuron_90(Q):
    z = 0.08868433
    if Q.lep_z < 0.221436:
        z += -0.2651343 * Q.lep_z + 0.01579818
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += 0.0263748 * Q.lep_z - 0.04875243
    if 0.3396572 <= Q.lep_z < 0.5187302:
        z += 0.2222225 * Q.lep_z - 0.1152735
    if Q.z_displaced3 < 0.03624058:
        z += 0.2050558 * Q.z_displaced3 - 0.06687015
    if 0.03624058 <= Q.z_displaced3 < 0.3261071:
        z += -1.168159 * Q.z_displaced3 - 0.01710404
    if Q.z_displaced3 >= 0.3261071:
        z += -1.373215 * Q.z_displaced3 + 0.04976612
    if 48.76729 <= Q.sj4_pair_mass_max < 63.34656:
        z += -0.0005062044 * Q.sj4_pair_mass_max + 0.02468622
    if Q.sj4_pair_mass_max >= 63.34656:
        z += -0.003098479 * Q.sj4_pair_mass_max + 0.1888979
    if Q.ecf_g31 < 0.002975626:
        z += 20.94357 * Q.ecf_g31 - 0.06232024
    if 44.2029 <= Q.sj3_pair_mass_max < 142.9952:
        z += -0.001089748 * Q.sj3_pair_mass_max + 0.04817003
    if Q.sj3_pair_mass_max >= 142.9952:
        z += 0.0004504693 * Q.sj3_pair_mass_max - 0.1720737
    if Q.lep_ptrel < 12.15228:
        z += 0.01872 * Q.lep_ptrel - 0.04033612
    if 12.15228 <= Q.lep_ptrel < 18.7678:
        z += -0.00427553 * Q.lep_ptrel + 0.239112
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += -0.006500375 * Q.lep_ptrel + 0.2808674
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += -0.004115588 * Q.lep_ptrel + 0.2157065
    if Q.lep_ptrel >= 43.20788:
        z += 0.002384786 * Q.lep_ptrel - 0.06516096
    if Q.mass_top30 >= 105.8151:
        z += -0.005077272 * Q.mass_top30 + 0.537252
    if Q.n_dr_0p4_up < 4.0:
        z += 0.0053192 * Q.n_dr_0p4_up - 0.0212768
    if 161.1264 <= Q.mass_top50 < 178.725:
        z += -0.01273429 * Q.mass_top50 + 2.051831
    if Q.mass_top50 >= 178.725:
        z += -0.01941038 * Q.mass_top50 + 3.245015
    if Q.tau1 < 0.06074238:
        z += 10.72695 * Q.tau1 - 0.6515805
    if Q.tau21 < 0.2665611:
        z += -0.2452982 * Q.tau21 + 0.06538696
    z += 0.008611848 * Q.mass_displaced3
    if 97.12186 <= Q.mass_top40 < 122.7145:
        z += -0.003842949 * Q.mass_top40 + 0.3732344
    if Q.mass_top40 >= 122.7145:
        z += 0.008216409 * Q.mass_top40 - 1.106624
    if Q.n_s3d_above_3 < 6.0:
        z += -0.04192215 * Q.n_s3d_above_3 + 0.4479864
    if 6.0 <= Q.n_s3d_above_3 < 10.0:
        z += -0.04911337 * Q.n_s3d_above_3 + 0.4911337
    if Q.n_charged_had >= 21.0:
        z += -0.0004018729 * Q.n_charged_had + 0.008439332
    if Q.z_displaced5 < 0.1709091:
        z += 0.8467698 * Q.z_displaced5 - 0.1167214
    if 0.1709091 <= Q.z_displaced5 < 0.2801368:
        z += -0.2563382 * Q.z_displaced5 + 0.07180976
    if Q.sj3_pairmin_over_m >= 0.1860959:
        z += 0.04064155 * Q.sj3_pairmin_over_m - 0.007563223
    if Q.M3_b2 < 0.02160244:
        z += 2.789154 * Q.M3_b2 - 0.06025253
    if Q.lep_iso < 1.362094:
        z += -0.4402465 * Q.lep_iso + 0.5591785
    if 1.362094 <= Q.lep_iso < 2.907433:
        z += 0.0261941 * Q.lep_iso - 0.07615758
    if Q.M2 < 0.09740996:
        z += -0.3287576 * Q.M2 + 0.03202427
    if Q.max_abs_d0 < 5.8125:
        z += 0.0188977 * Q.max_abs_d0 - 0.1098429
    if Q.e3 < 0.0004188921:
        z += 387.5424 * Q.e3 - 0.1623384
    if Q.e2 < 0.05697681:
        z += -12.53835 * Q.e2 + 0.714395
    if Q.mass_2charged >= 24.4079:
        z += -0.005118434 * Q.mass_2charged + 0.1249302
    if Q.n_photon < 17.0:
        z += -0.002355231 * Q.n_photon + 0.04003893
    if Q.n_pairs_kt_above_1 >= 80.0:
        z += -0.000669111 * Q.n_pairs_kt_above_1 + 0.05352888
    if Q.z_charged_had < 0.3191471:
        z += 1.033402 * Q.z_charged_had - 0.3298073
    if Q.tau3 < 0.08643515:
        z += 8.471077 * Q.tau3 - 0.7321988
    if 78.4753 <= Q.sd_mass < 94.55722:
        z += 0.008049361 * Q.sd_mass - 0.631676
    if 94.55722 <= Q.sd_mass < 100.8525:
        z += 0.0146143 * Q.sd_mass - 1.252439
    if 100.8525 <= Q.sd_mass < 123.2919:
        z += 0.01091093 * Q.sd_mass - 0.8789445
    if 123.2919 <= Q.sd_mass < 154.5947:
        z += -0.006952013 * Q.sd_mass + 1.323411
    if 154.5947 <= Q.sd_mass < 175.9333:
        z += -0.005701344 * Q.sd_mass + 1.130065
    if Q.sd_mass >= 175.9333:
        z += 0.00553017 * Q.sd_mass - 0.845933
    if Q.pair_mean_lnm2 >= 2.843705:
        z += 0.09632539 * Q.pair_mean_lnm2 - 0.273921
    if Q.sum_z_dr2 < 0.07298691:
        z += -0.8561078 * Q.sum_z_dr2 + 0.06248466
    if Q.sip_3d_3 < 49.03695:
        z += 0.001601722 * Q.sip_3d_3 - 0.07854355
    if Q.sj3_mass3 < 3.217057:
        z += -0.03846594 * Q.sj3_mass3 + 0.1237471
    if Q.mass >= 149.0507:
        z += 0.01451884 * Q.mass - 2.164043
    if Q.tau32 < 0.820003:
        z += -0.2770299 * Q.tau32 + 0.2271653
    if Q.psi_0p3 >= 0.9925964:
        z += 3.335245 * Q.psi_0p3 - 3.310552
    if Q.N2_b05 < 0.4265629:
        z += -0.1615947 * Q.N2_b05 + 0.0689303
    if Q.sj3_mass2 >= 9.068761:
        z += -0.002584015 * Q.sj3_mass2 + 0.02343381
    if Q.n_sd0_above_5 >= 1.0:
        z += 0.02778741 * Q.n_sd0_above_5 - 0.02778741
    if Q.mass_displaced5 >= 13.97388:
        z += -0.009839169 * Q.mass_displaced5 + 0.1374913
    if Q.z_dr_0_0p05 < 0.06366826:
        z += 1.052046 * Q.z_dr_0_0p05 - 0.06698195
    if Q.M2_b05 >= 0.1080247:
        z += -1.511841 * Q.M2_b05 + 0.1633163
    if Q.z_charged < 0.3265243:
        z += 1.580716 * Q.z_charged - 0.5161421
    if Q.tau4 >= 0.05417009:
        z += 2.659734 * Q.tau4 - 0.144078
    if Q.lep_ptrel > 27.3236 and Q.M2 > 0.03465331:
        z += -0.2180575 * (Q.lep_ptrel - 27.3236) * (Q.M2 - 0.03465331)
    if Q.n_dr_0p4_up < 4.0 and Q.tau32 < 0.6206221:
        z += -0.1036944 * (4.0 - Q.n_dr_0p4_up) * (0.6206221 - Q.tau32)
    if Q.z_displaced3 < 0.3261071 and Q.ecf_g41 > 0.0001893721:
        z += -518.3243 * (0.3261071 - Q.z_displaced3) * (Q.ecf_g41 - 0.0001893721)
    if Q.tau21 < 0.2665611 and Q.n_lepton < 2.0:
        z += -0.1576883 * (0.2665611 - Q.tau21) * (2.0 - Q.n_lepton)
    if Q.lep_ptrel < 18.7678 and Q.sip_3d_2 < 226.3008:
        z += 3.915351e-05 * (18.7678 - Q.lep_ptrel) * (226.3008 - Q.sip_3d_2)
    if Q.lep_ptrel < 18.7678 and Q.z_displaced5 < 0.2184442:
        z += 0.0323219 * (18.7678 - Q.lep_ptrel) * (0.2184442 - Q.z_displaced5)
    if Q.tau21 < 0.2665611 and Q.z_charged_had < 0.6056728:
        z += -1.70116 * (0.2665611 - Q.tau21) * (0.6056728 - Q.z_charged_had)
    if Q.lep_z < 0.3396572 and Q.sj3_pairmin_over_m > 0.2278414:
        z += -1.735818 * (0.3396572 - Q.lep_z) * (Q.sj3_pairmin_over_m - 0.2278414)
    if Q.lep_ptrel > 27.3236 and Q.sip_3d_3 < 23.30856:
        z += 0.0002080062 * (Q.lep_ptrel - 27.3236) * (23.30856 - Q.sip_3d_3)
    if Q.lep_ptrel < 18.7678 and Q.lep_iso < 0.02781886:
        z += -1.164855 * (18.7678 - Q.lep_ptrel) * (0.02781886 - Q.lep_iso)
    if Q.mass_top30 > 105.8151 and Q.sip_3d_3 < 3.157646:
        z += -0.00376323 * (Q.mass_top30 - 105.8151) * (3.157646 - Q.sip_3d_3)
    if Q.lep_ptrel > 27.3236 and Q.z_neutral > 0.2647267:
        z += -0.04313143 * (Q.lep_ptrel - 27.3236) * (Q.z_neutral - 0.2647267)
    if Q.lep_iso < 1.362094 and Q.sip_3d_3 < 3.157646:
        z += 0.001423687 * (1.362094 - Q.lep_iso) * (3.157646 - Q.sip_3d_3)
    if Q.max_abs_d0 < 5.8125 and Q.sip_3d_3 < 4.606241:
        z += -0.00049355 * (5.8125 - Q.max_abs_d0) * (4.606241 - Q.sip_3d_3)
    if Q.mass_top50 > 178.725 and Q.tdz_13 < -0.1329965:
        z += 0.002274435 * (Q.mass_top50 - 178.725) * (-0.1329965 - Q.tdz_13)
    if Q.sj3_pairmin_over_m > 0.1860959 and Q.z_neutral_had < 0.1138902:
        z += 0.1431946 * (Q.sj3_pairmin_over_m - 0.1860959) * (0.1138902 - Q.z_neutral_had)
    if Q.z_displaced5 < 0.2801368 and Q.sip_3d_3 < 175.9957:
        z += 0.0001202694 * (0.2801368 - Q.z_displaced5) * (175.9957 - Q.sip_3d_3)
    if Q.lep_z < 0.5187302 and Q.jet_charge > 0.2072767:
        z += -2.259324 * (0.5187302 - Q.lep_z) * (Q.jet_charge - 0.2072767)
    if Q.lep_ptrel < 18.7678 and Q.z_top50_slots < 1.0:
        z += 0.02183628 * (18.7678 - Q.lep_ptrel) * (1.0 - Q.z_top50_slots)
    if Q.z_displaced3 > 0.03624058 and Q.z_charged_had > 0.265564:
        z += 2.26389 * (Q.z_displaced3 - 0.03624058) * (Q.z_charged_had - 0.265564)
    if Q.z_displaced3 > 0.03624058 and Q.lund3_lndelta < -0.603253:
        z += 0.08525503 * (Q.z_displaced3 - 0.03624058) * (-0.603253 - Q.lund3_lndelta)
    if Q.lep_ptrel > 27.3236 and Q.C2_b2 < 0.03403084:
        z += -0.07880668 * (Q.lep_ptrel - 27.3236) * (0.03403084 - Q.C2_b2)
    if Q.lep_z < 0.221436 and Q.z_photon > 0.05998812:
        z += 3.262503 * (0.221436 - Q.lep_z) * (Q.z_photon - 0.05998812)
    if Q.n_s3d_above_3 < 6.0 and Q.eccentricity > 0.5954257:
        z += 0.02898479 * (6.0 - Q.n_s3d_above_3) * (Q.eccentricity - 0.5954257)
    if Q.sum_z_dr2 < 0.07298691 and Q.phi_1 < -0.02072144:
        z += 3.400046 * (0.07298691 - Q.sum_z_dr2) * (-0.02072144 - Q.phi_1)
    if Q.sj4_pair_mass_max > 63.34656 and Q.charge_32 < 0.0:
        z += 0.0008529901 * (Q.sj4_pair_mass_max - 63.34656) * (0.0 - Q.charge_32)
    if Q.z_displaced3 > 0.03624058 and Q.lne_7 < 3.018136:
        z += 0.3052813 * (Q.z_displaced3 - 0.03624058) * (3.018136 - Q.lne_7)
    if Q.tau3 < 0.08643515 and Q.lne_7 > 2.313463:
        z += 0.7491184 * (0.08643515 - Q.tau3) * (Q.lne_7 - 2.313463)
    if Q.sum_z_dr2 < 0.07298691 and Q.dr_31 < 0.2377815:
        z += 2.602618 * (0.07298691 - Q.sum_z_dr2) * (0.2377815 - Q.dr_31)
    if Q.lep_iso < 1.362094 and Q.n_neutral_had < 6.0:
        z += -0.01931717 * (1.362094 - Q.lep_iso) * (6.0 - Q.n_neutral_had)
    if Q.lep_ptrel < 43.20788 and Q.C2_b2 > 0.221369:
        z += 0.001323969 * (43.20788 - Q.lep_ptrel) * (Q.C2_b2 - 0.221369)
    if Q.lep_ptrel < 18.7678 and Q.sd_rg > 0.1450854:
        z += -0.004359799 * (18.7678 - Q.lep_ptrel) * (Q.sd_rg - 0.1450854)
    if Q.M3_b2 < 0.02160244 and Q.sj3_dr23 > 0.3377973:
        z += -10.00433 * (0.02160244 - Q.M3_b2) * (Q.sj3_dr23 - 0.3377973)
    if Q.n_pairs_kt_above_1 > 80.0 and Q.sj3_mass2 < 5.207416:
        z += 0.0004438555 * (Q.n_pairs_kt_above_1 - 80.0) * (5.207416 - Q.sj3_mass2)
    if Q.lep_ptrel < 18.7678 and Q.sj3_mass2 < 5.207416:
        z += -0.001183782 * (18.7678 - Q.lep_ptrel) * (5.207416 - Q.sj3_mass2)
    if Q.sj3_pairmin_over_m > 0.1860959 and Q.td0_2 < -0.02151157:
        z += 0.4016336 * (Q.sj3_pairmin_over_m - 0.1860959) * (-0.02151157 - Q.td0_2)
    if Q.mass_top40 > 97.12186 and Q.ismuon_2 > 0.0:
        z += 0.004203528 * (Q.mass_top40 - 97.12186) * (Q.ismuon_2 - 0.0)
    if Q.sj3_pair_mass_max > 142.9952 and Q.ischhad_57 > 0.0:
        z += -0.002779023 * (Q.sj3_pair_mass_max - 142.9952) * (Q.ischhad_57 - 0.0)
    return z


def neuron_91(Q):
    z = 3.99263e-06
    return z


def neuron_92(Q):
    z = -6.724725e-06
    return z


def neuron_93(Q):
    z = 8.464321e-07
    return z


def neuron_94(Q):
    z = -2.322211e-06
    return z


def neuron_95(Q):
    z = 4.369356e-07
    return z


def neuron_96(Q):
    z = 4.277726e-07
    return z


def neuron_97(Q):
    z = -0.5787856
    if Q.mass_displaced3 < 18.80005:
        z += -0.02922767 * Q.mass_displaced3 + 1.743632
    if 18.80005 <= Q.mass_displaced3 < 39.09615:
        z += -0.05883645 * Q.mass_displaced3 + 2.300279
    if Q.mass < 95.14961:
        z += -0.03186718 * Q.mass + 1.021318
    if 95.14961 <= Q.mass < 100.4835:
        z += 0.002530802 * Q.mass - 2.251637
    if 100.4835 <= Q.mass < 123.7928:
        z += 0.02060421 * Q.mass - 4.067715
    if 123.7928 <= Q.mass < 164.4374:
        z += 0.037325 * Q.mass - 6.137628
    if Q.e3_b2 < 9.621843e-05:
        z += 1704.632 * Q.e3_b2 - 0.04689601
    if 9.621843e-05 <= Q.e3_b2 < 0.0002536827:
        z += 1306.456 * Q.e3_b2 - 0.008584121
    if 0.0002536827 <= Q.e3_b2 < 0.0007909605:
        z += -600.8829 * Q.e3_b2 + 0.4752746
    if Q.pair_mean_lnm2 >= 2.09826:
        z += -0.007206139 * Q.pair_mean_lnm2 + 0.01512036
    if Q.mass_top50 < 79.27954:
        z += 0.01721024 * Q.mass_top50 - 0.9982488
    if 79.27954 <= Q.mass_top50 < 115.614:
        z += -0.003393275 * Q.mass_top50 + 0.6351882
    if 115.614 <= Q.mass_top50 < 161.1264:
        z += -0.005336525 * Q.mass_top50 + 0.859855
    if Q.sip_3d_2 < 447.0873:
        z += 0.0003793091 * Q.sip_3d_2 - 0.1695843
    if Q.tau32 < 0.6206221:
        z += 0.4229496 * Q.tau32 - 0.2624919
    if Q.mass_top40 < 126.8853:
        z += -0.006677156 * Q.mass_top40 + 0.8472327
    if 53.51926 <= Q.sj2_mass1 < 77.42768:
        z += 0.01507851 * Q.sj2_mass1 - 0.8069908
    if 77.42768 <= Q.sj2_mass1 < 91.2852:
        z += -0.02594601 * Q.sj2_mass1 + 2.369443
    if Q.sj2_mass1 >= 91.2852:
        z += 0.004064807 * Q.sj2_mass1 - 0.3701003
    if Q.n_pairs_kt_above_1 < 325.0:
        z += -0.001030297 * Q.n_pairs_kt_above_1 + 0.3348465
    if Q.M2_b2 < 0.01015545:
        z += 8.190239 * Q.M2_b2 - 0.08317559
    if Q.psi_0p1 >= 0.7386202:
        z += -0.3925182 * Q.psi_0p1 + 0.2899219
    if Q.sj3_pair_mass_max < 73.24742:
        z += -0.004792192 * Q.sj3_pair_mass_max + 0.9076641
    if 73.24742 <= Q.sj3_pair_mass_max < 109.8208:
        z += -0.01221926 * Q.sj3_pair_mass_max + 1.451678
    if 109.8208 <= Q.sj3_pair_mass_max < 120.6471:
        z += -0.01013719 * Q.sj3_pair_mass_max + 1.223024
    if Q.max_abs_d0 < 5.8125:
        z += 0.05231674 * Q.max_abs_d0 - 0.304091
    if Q.sj3_mass3 < 2.68613:
        z += -0.06405294 * Q.sj3_mass3 + 0.1720545
    if Q.lund3_lndelta >= -1.902701:
        z += 0.1315424 * Q.lund3_lndelta + 0.2502859
    if -1.388411 <= Q.lund_max_lndelta < -0.6897565:
        z += -0.1789033 * Q.lund_max_lndelta - 0.2483913
    if Q.lund_max_lndelta >= -0.6897565:
        z += 0.3103528 * Q.lund_max_lndelta + 0.0890763
    if Q.e2 >= 0.09764648:
        z += 1.373451 * Q.e2 - 0.1341127
    if Q.z_neutral_had >= 0.07573803:
        z += -0.4184843 * Q.z_neutral_had + 0.03169518
    if Q.z_muon < 0.1403354:
        z += 1.568375 * Q.z_muon - 0.2200984
    if Q.n_sd0_above_3 >= 1.0:
        z += 0.1703216 * Q.n_sd0_above_3 - 0.1703216
    if Q.n_s3d_above_3 >= 3.0:
        z += -0.165237 * Q.n_s3d_above_3 + 0.495711
    if Q.z_displaced5 >= 0.1339658:
        z += -0.6166818 * Q.z_displaced5 + 0.08261426
    if Q.lne_0 >= 5.157617:
        z += 0.1951845 * Q.lne_0 - 1.006687
    if Q.n_pairs_kt_above_3 < 34.0:
        z += 0.005060057 * Q.n_pairs_kt_above_3 + 0.06049269
    if 34.0 <= Q.n_pairs_kt_above_3 < 126.0:
        z += -0.00252755 * Q.n_pairs_kt_above_3 + 0.3184713
    if Q.n_photon < 19.0:
        z += -0.01802395 * Q.n_photon + 0.3424551
    if Q.sj3_pairmax_over_m < 0.8501496:
        z += 1.151223 * Q.sj3_pairmax_over_m - 0.9787117
    if Q.n_sd0_above_2 >= 5.0:
        z += -0.08781209 * Q.n_sd0_above_2 + 0.4390604
    if Q.pair_max_lnm2 < 7.203613:
        z += -0.07688931 * Q.pair_max_lnm2 + 0.5538808
    if Q.max_dr < 0.6079631:
        z += 0.6709365 * Q.max_dr - 0.4079047
    z += 0.01136285 * Q.n_pairs_kt_above_10
    if Q.D2 >= 3.597891:
        z += 0.04137716 * Q.D2 - 0.1488705
    if Q.lund_max_lnkt < 3.521178:
        z += 0.07026187 * Q.lund_max_lnkt - 0.2474046
    if Q.n_pt_above_5 >= 27.0:
        z += -0.002161021 * Q.n_pt_above_5 + 0.05834757
    if Q.sj3_dr_min >= 0.3190414:
        z += -0.2421324 * Q.sj3_dr_min + 0.07725027
    if Q.sd_mass < 62.03827:
        z += 0.001505531 * Q.sd_mass - 0.564925
    if 62.03827 <= Q.sd_mass < 72.27436:
        z += 0.03547435 * Q.sd_mass - 2.672292
    if 72.27436 <= Q.sd_mass < 78.4753:
        z += 0.01538739 * Q.sd_mass - 1.22052
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += -0.07851662 * Q.sd_mass + 6.148625
    if 88.81751 <= Q.sd_mass < 100.8525:
        z += 0.005824393 * Q.sd_mass - 1.342333
    if 100.8525 <= Q.sd_mass < 115.7091:
        z += -0.006232565 * Q.sd_mass - 0.1263585
    if 115.7091 <= Q.sd_mass < 119.4443:
        z += -0.06140315 * Q.sd_mass + 6.25738
    if 119.4443 <= Q.sd_mass < 154.5947:
        z += 0.03063639 * Q.sd_mass - 4.736223
    if Q.N2_b2 < 0.1244374:
        z += -0.08208684 * Q.N2_b2 + 0.01021468
    if Q.sj3_pair_mass_min >= 59.49644:
        z += -0.0112146 * Q.sj3_pair_mass_min + 0.6672289
    if Q.e2_b05 >= 0.1510354:
        z += -0.8842105 * Q.e2_b05 + 0.1335471
    if Q.jet_charge_k05 >= 0.07435708:
        z += 0.2591563 * Q.jet_charge_k05 - 0.01927011
    if Q.e3_b2 < 0.0002536827 and Q.mass_charged > 84.24838:
        z += 47.4854 * (0.0002536827 - Q.e3_b2) * (Q.mass_charged - 84.24838)
    if Q.e3_b2 < 0.0002536827 and Q.z_displaced3 < 0.3261071:
        z += -596.949 * (0.0002536827 - Q.e3_b2) * (0.3261071 - Q.z_displaced3)
    if Q.mass_displaced3 < 39.09615 and Q.lep_ptrel < 27.3236:
        z += -0.0003280566 * (39.09615 - Q.mass_displaced3) * (27.3236 - Q.lep_ptrel)
    if Q.mass_displaced3 < 39.09615 and Q.sum_pt_top5 < 586.125:
        z += 4.852818e-06 * (39.09615 - Q.mass_displaced3) * (586.125 - Q.sum_pt_top5)
    if Q.mass_top50 < 161.1264 and Q.n_s3d_above_3 > 1.0:
        z += -0.0005225642 * (161.1264 - Q.mass_top50) * (Q.n_s3d_above_3 - 1.0)
    if Q.e3_b2 < 0.0002536827 and Q.jet_charge_k03 > -0.05990128:
        z += 960.7808 * (0.0002536827 - Q.e3_b2) * (Q.jet_charge_k03 - -0.05990128)
    if Q.mass_top50 < 115.614 and Q.n_lepton < 1.0:
        z += 0.0004677075 * (115.614 - Q.mass_top50) * (1.0 - Q.n_lepton)
    if Q.mass_top50 < 161.1264 and Q.sip_3d_3 < 175.9957:
        z += -5.877273e-06 * (161.1264 - Q.mass_top50) * (175.9957 - Q.sip_3d_3)
    if Q.mass_top40 < 126.8853 and Q.lne_5 > 2.737811:
        z += 0.0016649 * (126.8853 - Q.mass_top40) * (Q.lne_5 - 2.737811)
    if Q.mass_top40 < 126.8853 and Q.n_muon > 0.0:
        z += -0.006836648 * (126.8853 - Q.mass_top40) * (Q.n_muon - 0.0)
    if Q.e3_b2 < 0.0002536827 and Q.tau21_b2 < 0.3565533:
        z += -1860.493 * (0.0002536827 - Q.e3_b2) * (0.3565533 - Q.tau21_b2)
    if Q.tau32 < 0.6206221 and Q.z_charged_had < 0.6056728:
        z += 2.024055 * (0.6206221 - Q.tau32) * (0.6056728 - Q.z_charged_had)
    if Q.e3_b2 < 0.0002536827 and Q.n_dr_0p4_up > 8.0:
        z += 134.058 * (0.0002536827 - Q.e3_b2) * (Q.n_dr_0p4_up - 8.0)
    if Q.max_abs_d0 < 5.8125 and Q.sip_3d_3 < 7.345216:
        z += 0.01100345 * (5.8125 - Q.max_abs_d0) * (7.345216 - Q.sip_3d_3)
    if Q.sj3_pair_mass_max < 109.8208 and Q.lnpt_44 > -18.42068:
        z += -6.991685e-07 * (109.8208 - Q.sj3_pair_mass_max) * (Q.lnpt_44 - -18.42068)
    if Q.e3_b2 < 0.0002536827 and Q.sj3_dr13 > 0.7462286:
        z += -894.8817 * (0.0002536827 - Q.e3_b2) * (Q.sj3_dr13 - 0.7462286)
    if Q.mass_displaced3 < 39.09615 and Q.n_pairs_kt_above_10 > 2.0:
        z += -0.0005217863 * (39.09615 - Q.mass_displaced3) * (Q.n_pairs_kt_above_10 - 2.0)
    if Q.mass_displaced3 < 18.80005 and Q.sip_3d_3 < 175.9957:
        z += 0.0004279015 * (18.80005 - Q.mass_displaced3) * (175.9957 - Q.sip_3d_3)
    if Q.mass_displaced3 < 39.09615 and Q.sip_3d_3 < 175.9957:
        z += -0.0001850828 * (39.09615 - Q.mass_displaced3) * (175.9957 - Q.sip_3d_3)
    if Q.n_sd0_above_3 > 1.0 and Q.sj3_dr13 > 0.6229991:
        z += -0.02915101 * (Q.n_sd0_above_3 - 1.0) * (Q.sj3_dr13 - 0.6229991)
    if Q.mass_top50 < 161.1264 and Q.min_pair_mass > 3.369962:
        z += 2.397378e-05 * (161.1264 - Q.mass_top50) * (Q.min_pair_mass - 3.369962)
    if Q.sj3_pairmax_over_m < 0.8501496 and Q.ismuon_37 > 0.0:
        z += 0.2587172 * (0.8501496 - Q.sj3_pairmax_over_m) * (Q.ismuon_37 - 0.0)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.D2_b2 < 2.286291:
        z += 0.003481333 * (34.0 - Q.n_pairs_kt_above_3) * (2.286291 - Q.D2_b2)
    if Q.sj2_mass1 > 77.42768 and Q.D2_b2 < 0.8678264:
        z += -0.1248271 * (Q.sj2_mass1 - 77.42768) * (0.8678264 - Q.D2_b2)
    if Q.mass_displaced3 < 18.80005 and Q.mass_2photon > 9.82024:
        z += 0.000470552 * (18.80005 - Q.mass_displaced3) * (Q.mass_2photon - 9.82024)
    if Q.lund_max_lndelta > -1.388411 and Q.lnerel_4 > -3.310904:
        z += -0.09637537 * (Q.lund_max_lndelta - -1.388411) * (Q.lnerel_4 - -3.310904)
    if Q.sip_3d_2 < 447.0873 and Q.td0_33 > -0.04023096:
        z += 0.0002990801 * (447.0873 - Q.sip_3d_2) * (Q.td0_33 - -0.04023096)
    if Q.n_s3d_above_3 > 3.0 and Q.d0err_10 > 0.02780151:
        z += 1.004428 * (Q.n_s3d_above_3 - 3.0) * (Q.d0err_10 - 0.02780151)
    if Q.mass < 100.4835 and Q.iselectron_0 > 0.0:
        z += -0.003599257 * (100.4835 - Q.mass) * (Q.iselectron_0 - 0.0)
    if Q.mass_displaced3 < 39.09615 and Q.iselectron_0 > 0.0:
        z += 0.0010981 * (39.09615 - Q.mass_displaced3) * (Q.iselectron_0 - 0.0)
    if Q.sip_3d_2 < 447.0873 and Q.lead_ch_sdz < -0.6526285:
        z += -2.467029e-05 * (447.0873 - Q.sip_3d_2) * (-0.6526285 - Q.lead_ch_sdz)
    if Q.sj3_pairmax_over_m < 0.8501496 and Q.dr_53 > 0.0:
        z += -0.5033078 * (0.8501496 - Q.sj3_pairmax_over_m) * (Q.dr_53 - 0.0)
    if Q.N2_b2 < 0.1244374 and Q.eta_29 < -0.1424561:
        z += 6.521479 * (0.1244374 - Q.N2_b2) * (-0.1424561 - Q.eta_29)
    if Q.e3_b2 < 0.0007909605 and Q.dr_28 > 0.3368505:
        z += 48.95766 * (0.0007909605 - Q.e3_b2) * (Q.dr_28 - 0.3368505)
    if Q.lund_max_lndelta > -1.388411 and Q.td0_7 < 0.09541201:
        z += 0.03030508 * (Q.lund_max_lndelta - -1.388411) * (0.09541201 - Q.td0_7)
    if Q.z_displaced5 > 0.1339658 and Q.C3_b2 > 0.04229114:
        z += -2.022226 * (Q.z_displaced5 - 0.1339658) * (Q.C3_b2 - 0.04229114)
    if Q.z_neutral_had > 0.07573803 and Q.td0_23 > 0.0:
        z += -0.236391 * (Q.z_neutral_had - 0.07573803) * (Q.td0_23 - 0.0)
    return z


def neuron_98(Q):
    z = 4.990151e-07
    return z


def neuron_99(Q):
    z = -0.001934279
    return z


def neuron_100(Q):
    z = -6.26407e-06
    return z


def neuron_101(Q):
    z = 8.878267e-06
    return z


def neuron_102(Q):
    z = 5.490487e-06
    return z


def neuron_103(Q):
    z = -3.414348e-06
    return z


def neuron_104(Q):
    z = -1.800354
    if Q.mass < 120.653:
        z += -0.02165528 * Q.mass + 2.773581
    if 120.653 <= Q.mass < 149.0507:
        z += -0.005662631 * Q.mass + 0.844019
    if Q.pair_mean_lndelta < -1.352792:
        z += -0.4135533 * Q.pair_mean_lndelta - 0.5594515
    if Q.N2_b05 >= 0.3667049:
        z += -1.976923 * Q.N2_b05 + 0.7249473
    if 3.208089 <= Q.mass_displaced3 < 6.341631:
        z += 0.08222757 * Q.mass_displaced3 - 0.2637934
    if 6.341631 <= Q.mass_displaced3 < 26.78691:
        z += 0.03451983 * Q.mass_displaced3 + 0.03875157
    if Q.mass_displaced3 >= 26.78691:
        z += 0.0106705 * Q.mass_displaced3 + 0.6776012
    if Q.n_s3d_above_10 < 8.0:
        z += -0.03605828 * Q.n_s3d_above_10 + 0.2884662
    z += 0.001955867 * Q.mass_neutral
    if Q.tau32_b2 < 0.4680886:
        z += 1.000892 * Q.tau32_b2 - 0.4685063
    if Q.N2 >= 0.3245983:
        z += 0.8687871 * Q.N2 - 0.2820068
    if Q.n_pairs_kt_above_3 < 47.0:
        z += -3.312017e-05 * Q.n_pairs_kt_above_3 + 0.001556648
    if Q.z_displaced3 < 0.01782783:
        z += 1.516771 * Q.z_displaced3 - 0.02704072
    if Q.sj3_dr_min < 0.3190414:
        z += -0.2947441 * Q.sj3_dr_min + 0.09403557
    if Q.mass_top40 < 132.5189:
        z += -0.008818347 * Q.mass_top40 + 1.316541
    if 132.5189 <= Q.mass_top40 < 155.8928:
        z += -0.006329434 * Q.mass_top40 + 0.986713
    if Q.tau21 < 0.135772:
        z += -4.739475 * Q.tau21 + 0.5437632
    if 0.135772 <= Q.tau21 < 0.2377332:
        z += 0.9780639 * Q.tau21 - 0.2325182
    if Q.tau21 >= 0.5797033:
        z += 0.9034303 * Q.tau21 - 0.5237215
    if Q.n_sd0_above_3 < 6.0:
        z += -0.04474772 * Q.n_sd0_above_3 + 0.2684863
    if Q.n_particles < 67.0:
        z += -0.008198361 * Q.n_particles + 0.5492902
    if Q.n_pairs_kt_above_1 < 146.0:
        z += 0.004955527 * Q.n_pairs_kt_above_1 - 0.723507
    if Q.C2 < 0.191059:
        z += -1.444524 * Q.C2 + 0.2759892
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += 0.02707356 * Q.sd_mass - 2.124606
    if 88.81751 <= Q.sd_mass < 106.7501:
        z += -0.004691942 * Q.sd_mass + 0.6967271
    if 106.7501 <= Q.sd_mass < 154.5947:
        z += 0.0008362192 * Q.sd_mass + 0.1065956
    if 154.5947 <= Q.sd_mass < 175.9333:
        z += 0.01177131 * Q.sd_mass - 1.583911
    if Q.sd_mass >= 175.9333:
        z += 0.008826422 * Q.sd_mass - 1.065807
    if Q.mass_charged < 99.20396:
        z += -0.0008315978 * Q.mass_charged + 0.08249779
    if Q.jet_abs_eta >= 0.6771968:
        z += 0.1476596 * Q.jet_abs_eta - 0.09999459
    if Q.z_top15_slots >= 0.9212656:
        z += 9.078622 * Q.z_top15_slots - 8.363822
    if Q.lund_max_lndelta >= -0.529318:
        z += 0.3136771 * Q.lund_max_lndelta + 0.166035
    if Q.n_lund >= 11.0:
        z += -0.02754514 * Q.n_lund + 0.3029966
    if Q.e3 >= 0.001589861:
        z += -105.9932 * Q.e3 + 0.1685144
    if Q.mass_over_sum_pt >= 0.215596:
        z += 2.450289 * Q.mass_over_sum_pt - 0.5282725
    if Q.n_pairs_kt_above_10 < 3.0:
        z += 0.004807309 * Q.n_pairs_kt_above_10 - 0.01442193
    if Q.n_sd0_above_2 < 7.0:
        z += 0.01987016 * Q.n_sd0_above_2 - 0.1390911
    if Q.lep_iso < 6.185635:
        z += 0.01366752 * Q.lep_iso - 0.0845423
    if Q.sj3_mass3 < 0.3886647:
        z += -0.5053353 * Q.sj3_mass3 + 0.196406
    if Q.jet_charge_k03 < -0.5570337:
        z += 0.1312279 * Q.jet_charge_k03 + 0.07309838
    if Q.sum_charge >= 4.0:
        z += -0.02184478 * Q.sum_charge + 0.08737912
    if Q.sip_3d_2 < 226.3008:
        z += 9.467423e-05 * Q.sip_3d_2 - 0.02142485
    z += -0.006921937 * Q.n_charged_pt_above_1
    if Q.tau4 < 0.02949822:
        z += -17.94051 * Q.tau4 + 0.5292133
    if Q.n_neutral_had < 4.0:
        z += 0.06696504 * Q.n_neutral_had - 0.2678602
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += 0.006778121 * Q.lep_ptrel - 0.1272104
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.020012 * Q.lep_ptrel - 0.4888076
    if Q.lep_ptrel >= 43.20788:
        z += -0.02089016 * Q.lep_ptrel + 1.278488
    if Q.D2 >= 3.597891:
        z += 0.08939813 * Q.D2 - 0.3216447
    if Q.n_sdz_above_5 < 3.0:
        z += -0.01254103 * Q.n_sdz_above_5 + 0.03762308
    if Q.n_s3d_above_3 < 5.0:
        z += 0.01070305 * Q.n_s3d_above_3 - 0.05351524
    if Q.sd_nremoved < 4.0:
        z += 0.04673579 * Q.sd_nremoved - 0.1869431
    if Q.sd_rg < 0.3025746:
        z += -0.7782252 * Q.sd_rg + 0.432864
    if 0.3025746 <= Q.sd_rg < 0.5562194:
        z += -0.07649434 * Q.sd_rg + 0.220538
    if Q.sd_rg >= 0.5562194:
        z += 0.7017309 * Q.sd_rg - 0.212326
    if Q.n_dr_0p4_up < 10.0:
        z += -0.01337606 * Q.n_dr_0p4_up + 0.1337606
    if Q.M3 >= 0.05919662:
        z += 14.65626 * Q.M3 - 0.8676012
    if Q.ecf_g41 < 6.216954e-05:
        z += -4498.49 * Q.ecf_g41 + 0.2796691
    if Q.tau2 < 0.04483276:
        z += 0.7249404 * Q.tau2 - 0.04257409
    if 0.04483276 <= Q.tau2 < 0.07264571:
        z += 0.3621696 * Q.tau2 - 0.02631007
    if Q.n_for_90pct >= 39.0:
        z += -0.01083233 * Q.n_for_90pct + 0.4224611
    if Q.n_lund_kt_above_5 >= 2.0:
        z += 0.05312746 * Q.n_lund_kt_above_5 - 0.1062549
    if Q.td0_38 < -0.02912079:
        z += -0.04100977 * Q.td0_38 - 0.001194237
    if Q.C2_b05 < 0.2326317:
        z += -4.503993 * Q.C2_b05 + 1.047772
    if Q.e2 < 0.0318986:
        z += -1.200913 * Q.e2 + 0.03830745
    if Q.C2_b2 >= 0.221369:
        z += 0.0571729 * Q.C2_b2 - 0.01265631
    if Q.lund_max_lnkt < 4.046329:
        z += 0.1530339 * Q.lund_max_lnkt - 0.6192257
    if Q.pair_max_lnm2 >= 6.052324:
        z += 0.07215122 * Q.pair_max_lnm2 - 0.4366825
    if Q.mass_top5 >= 17.63354:
        z += -0.001892701 * Q.mass_top5 + 0.03337503
    if Q.sj2_mass1 >= 53.51926:
        z += -0.001854565 * Q.sj2_mass1 + 0.09925494
    if Q.pair_mean_lndelta < -1.352792 and Q.lep_ptrel > 12.15228:
        z += -0.01138432 * (-1.352792 - Q.pair_mean_lndelta) * (Q.lep_ptrel - 12.15228)
    if Q.n_s3d_above_10 < 8.0 and Q.n_real_top50 < 49.0:
        z += 0.001854723 * (8.0 - Q.n_s3d_above_10) * (49.0 - Q.n_real_top50)
    if Q.mass < 149.0507 and Q.lne_0 > 5.157617:
        z += 0.004727736 * (149.0507 - Q.mass) * (Q.lne_0 - 5.157617)
    if Q.mass_displaced3 > 3.208089 and Q.lam2 < 0.02148541:
        z += -0.1015886 * (Q.mass_displaced3 - 3.208089) * (0.02148541 - Q.lam2)
    if Q.n_s3d_above_10 < 8.0 and Q.sj4_pair_mass_max > 101.849:
        z += 9.522836e-05 * (8.0 - Q.n_s3d_above_10) * (Q.sj4_pair_mass_max - 101.849)
    if Q.n_pairs_kt_above_3 < 47.0 and Q.lep_z < 0.221436:
        z += -0.03213717 * (47.0 - Q.n_pairs_kt_above_3) * (0.221436 - Q.lep_z)
    if Q.tau32_b2 < 0.4680886 and Q.ecf_g32 < 0.003965728:
        z += 82.09423 * (0.4680886 - Q.tau32_b2) * (0.003965728 - Q.ecf_g32)
    if Q.pair_mean_lndelta < -1.352792 and Q.lund1_lndelta > -0.6177752:
        z += 0.2071947 * (-1.352792 - Q.pair_mean_lndelta) * (Q.lund1_lndelta - -0.6177752)
    if Q.n_pairs_kt_above_3 < 47.0 and Q.z_displaced3 < 0.2601259:
        z += -0.002555592 * (47.0 - Q.n_pairs_kt_above_3) * (0.2601259 - Q.z_displaced3)
    if Q.n_s3d_above_10 < 8.0 and Q.lep_ptrel > 18.7678:
        z += 0.0005304674 * (8.0 - Q.n_s3d_above_10) * (Q.lep_ptrel - 18.7678)
    if Q.n_sd0_above_3 < 6.0 and Q.n_dr_0p4_up > 0.0:
        z += 0.00162184 * (6.0 - Q.n_sd0_above_3) * (Q.n_dr_0p4_up - 0.0)
    if Q.n_pairs_kt_above_1 < 146.0 and Q.D2 < 4.878859:
        z += -0.0002077502 * (146.0 - Q.n_pairs_kt_above_1) * (4.878859 - Q.D2)
    if Q.n_pairs_kt_above_1 < 146.0 and Q.n_lund_kt_above_5 > 1.0:
        z += 0.0008806579 * (146.0 - Q.n_pairs_kt_above_1) * (Q.n_lund_kt_above_5 - 1.0)
    if Q.mass_displaced3 > 3.208089 and Q.ecf_g43 < 2.885768e-05:
        z += 247.9141 * (Q.mass_displaced3 - 3.208089) * (2.885768e-05 - Q.ecf_g43)
    if Q.n_particles < 67.0 and Q.dr01 > 0.06433539:
        z += -0.004683626 * (67.0 - Q.n_particles) * (Q.dr01 - 0.06433539)
    if Q.sd_mass > 175.9333 and Q.lnptrel_77 > -18.42068:
        z += -0.000428699 * (Q.sd_mass - 175.9333) * (Q.lnptrel_77 - -18.42068)
    if Q.n_particles < 67.0 and Q.n_lepton < 2.0:
        z += 0.00206501 * (67.0 - Q.n_particles) * (2.0 - Q.n_lepton)
    if Q.tau32_b2 < 0.4680886 and Q.eta_1 > 0.01268768:
        z += 0.7741089 * (0.4680886 - Q.tau32_b2) * (Q.eta_1 - 0.01268768)
    if Q.z_top15_slots > 0.9212656 and Q.dzerr_1 < 0.03430176:
        z += 23.65565 * (Q.z_top15_slots - 0.9212656) * (0.03430176 - Q.dzerr_1)
    if Q.sd_mass > 175.9333 and Q.charge_6 < 0.0:
        z += 0.0003350639 * (Q.sd_mass - 175.9333) * (0.0 - Q.charge_6)
    if Q.sd_mass > 78.4753 and Q.charge_25 < 1.0:
        z += 1.731991e-05 * (Q.sd_mass - 78.4753) * (1.0 - Q.charge_25)
    if Q.sd_nremoved < 4.0 and Q.eta_28 > -0.3166626:
        z += 0.001215534 * (4.0 - Q.sd_nremoved) * (Q.eta_28 - -0.3166626)
    if Q.lep_iso < 6.185635 and Q.iselectron_27 > 0.0:
        z += 0.01887613 * (6.185635 - Q.lep_iso) * (Q.iselectron_27 - 0.0)
    if Q.sd_nremoved < 4.0 and Q.eta_4 > 0.05764771:
        z += 0.03452506 * (4.0 - Q.sd_nremoved) * (Q.eta_4 - 0.05764771)
    if Q.sd_mass > 78.4753 and Q.tdz_74 < 0.0:
        z += 0.001589204 * (Q.sd_mass - 78.4753) * (0.0 - Q.tdz_74)
    if Q.n_lund > 11.0 and Q.phi_21 < 0.1278076:
        z += 0.001526157 * (Q.n_lund - 11.0) * (0.1278076 - Q.phi_21)
    if Q.n_pairs_kt_above_3 < 47.0 and Q.dr02 > 0.2159556:
        z += -0.02320171 * (47.0 - Q.n_pairs_kt_above_3) * (Q.dr02 - 0.2159556)
    if Q.n_s3d_above_3 < 5.0 and Q.C2_b2 < 0.1404188:
        z += -0.25438 * (5.0 - Q.n_s3d_above_3) * (0.1404188 - Q.C2_b2)
    if Q.mass_over_sum_pt > 0.215596 and Q.dr_9 < 0.1211197:
        z += 1.867574 * (Q.mass_over_sum_pt - 0.215596) * (0.1211197 - Q.dr_9)
    if Q.n_pairs_kt_above_10 < 3.0 and Q.phi_29 < 0.0328064:
        z += 0.004685543 * (3.0 - Q.n_pairs_kt_above_10) * (0.0328064 - Q.phi_29)
    return z


def neuron_105(Q):
    z = 2.732385e-05
    return z


def neuron_106(Q):
    z = 2.160924e-05
    return z


def neuron_107(Q):
    z = 1.15856e-05
    return z


def neuron_108(Q):
    z = -4.470689e-06
    return z


def neuron_109(Q):
    z = -5.200097e-07
    return z


def neuron_110(Q):
    z = 3.579629e-06
    return z


def neuron_111(Q):
    z = 3.669613e-06
    return z


def neuron_112(Q):
    z = 5.237812e-06
    return z


def neuron_113(Q):
    z = -2.665935e-05
    return z


def neuron_114(Q):
    z = 4.677423e-06
    return z


def neuron_115(Q):
    z = 0.5648758
    if Q.M2_b2 < 0.05966366:
        z += -7.105628 * Q.M2_b2 + 0.4239478
    if Q.lep_z >= 0.03021637:
        z += 0.6170183 * Q.lep_z - 0.01864405
    if Q.mass_displaced3 < 39.09615:
        z += -0.02431072 * Q.mass_displaced3 + 0.9504558
    if Q.n_lund_kt_above_5 >= 1.0:
        z += -0.09026244 * Q.n_lund_kt_above_5 + 0.09026244
    if Q.tau2 < 0.09733903:
        z += -0.7978874 * Q.tau2 + 0.07766559
    if Q.n_pairs_kt_above_1 < 101.0:
        z += 0.00114698 * Q.n_pairs_kt_above_1 - 0.1158449
    if Q.mass_top30 < 109.3251:
        z += -0.002115688 * Q.mass_top30 + 0.2312979
    if Q.sj4_pair_mass_max < 72.862:
        z += 0.002016817 * Q.sj4_pair_mass_max - 0.7206414
    if 72.862 <= Q.sj4_pair_mass_max < 107.8099:
        z += 0.01641563 * Q.sj4_pair_mass_max - 1.769768
    if Q.n_s3d_above_3 >= 4.0:
        z += 0.07235892 * Q.n_s3d_above_3 - 0.2894357
    if Q.sj3_mass1 < 8.461136:
        z += 0.001450462 * Q.sj3_mass1 + 0.3201706
    if 8.461136 <= Q.sj3_mass1 < 36.36236:
        z += -0.011915 * Q.sj3_mass1 + 0.4332575
    if Q.sip_3d_2 < 447.0873:
        z += 0.0001715989 * Q.sip_3d_2 - 0.07671967
    if Q.sj3_pair_mass_min < 37.19471:
        z += 0.008779635 * Q.sj3_pair_mass_min - 0.8741609
    if 37.19471 <= Q.sj3_pair_mass_min < 59.49644:
        z += 0.01186106 * Q.sj3_pair_mass_min - 0.9887735
    if 59.49644 <= Q.sj3_pair_mass_min < 80.02563:
        z += 0.01378928 * Q.sj3_pair_mass_min - 1.103496
    if Q.sj4_dr_min < 0.07985021:
        z += 0.2946738 * Q.sj4_dr_min + 0.1420826
    if 0.07985021 <= Q.sj4_dr_min < 0.1845735:
        z += -1.581428 * Q.sj4_dr_min + 0.2918897
    if Q.M2 < 0.120439:
        z += -10.26506 * Q.M2 + 1.236314
    if Q.jet_abs_eta >= 1.323111:
        z += 0.1778245 * Q.jet_abs_eta - 0.2352817
    if Q.mass_top50 < 94.51361:
        z += 0.008184892 * Q.mass_top50 - 0.4025463
    if 94.51361 <= Q.mass_top50 < 125.4927:
        z += -0.007173363 * Q.mass_top50 + 1.049018
    if 125.4927 <= Q.mass_top50 < 129.5874:
        z += -0.03634225 * Q.mass_top50 + 4.709499
    if Q.mass < 120.653:
        z += -0.02706755 * Q.mass + 3.47351
    if 120.653 <= Q.mass < 164.4374:
        z += 0.001633579 * Q.mass + 0.01063197
    if 164.4374 <= Q.mass < 182.8592:
        z += -0.01515888 * Q.mass + 2.771941
    if Q.lam1 >= 0.05242126:
        z += -0.2183917 * Q.lam1 + 0.01144837
    if Q.ecf_g43 < 2.852309e-06:
        z += 43357.25 * Q.ecf_g43 - 0.1236683
    if Q.lep_iso < 0.4381892:
        z += 0.1906524 * Q.lep_iso - 0.08354181
    if Q.mass_2charged < 1.398635:
        z += 0.09464842 * Q.mass_2charged - 0.1059532
    if 1.398635 <= Q.mass_2charged < 6.767937:
        z += -0.05852133 * Q.mass_2charged + 0.1082754
    if 6.767937 <= Q.mass_2charged < 28.89659:
        z += 0.01300546 * Q.mass_2charged - 0.3758135
    if Q.lep_ptrel >= 43.20788:
        z += -0.04988309 * Q.lep_ptrel + 2.155343
    if Q.sj3_pair_mass_max < 109.8208:
        z += 0.01137129 * Q.sj3_pair_mass_max - 1.248805
    if Q.sj3_dr_min < 0.1991803:
        z += 0.3678051 * Q.sj3_dr_min - 0.240484
    if 0.1991803 <= Q.sj3_dr_min < 0.3190414:
        z += 1.395152 * Q.sj3_dr_min - 0.4451113
    if Q.sum_pt_top5 < 303.75:
        z += -0.002451058 * Q.sum_pt_top5 + 0.744509
    if Q.tau4 >= 0.04996:
        z += -11.0397 * Q.tau4 + 0.5515434
    if Q.e3_b05 < 0.0131933:
        z += -85.87848 * Q.e3_b05 + 1.13302
    if Q.tau1 >= 0.1246227:
        z += 3.299542 * Q.tau1 - 0.4111979
    if Q.sj3_pairmax_over_m < 0.8207647:
        z += -1.015518 * Q.sj3_pairmax_over_m + 0.8335015
    if Q.psi_0p1 < 0.006220408:
        z += 12.14229 * Q.psi_0p1 - 0.07552997
    if Q.max_abs_d0 < 5.8125:
        z += -0.001498341 * Q.max_abs_d0 + 0.008709107
    if Q.n_sdz_above_5 < 2.0:
        z += 0.07888781 * Q.n_sdz_above_5 - 0.1577756
    if Q.dr_0 < 0.1117947:
        z += 0.5352909 * Q.dr_0 - 0.05984268
    if Q.dr_0 >= 0.2254414:
        z += -0.2398981 * Q.dr_0 + 0.05408296
    if Q.N2_b2 < 0.2099278:
        z += -2.75001 * Q.N2_b2 + 0.5773036
    if Q.sj3_mass2 < 4.326415:
        z += 0.001018425 * Q.sj3_mass2 - 0.00440613
    if Q.n_sd0_above_3 >= 3.0:
        z += -0.02534798 * Q.n_sd0_above_3 + 0.07604393
    if Q.D3_b2 < 0.0225905:
        z += -2.116451 * Q.D3_b2 + 0.04781168
    if Q.sum_charge < -2.0:
        z += -0.02229432 * Q.sum_charge - 0.04458865
    if Q.mass_charged < 65.0404:
        z += -0.005561676 * Q.mass_charged + 0.3617337
    if Q.pair_mean_lndelta >= -3.421602:
        z += -0.4068173 * Q.pair_mean_lndelta - 1.391967
    if Q.sj3_dr23 < 0.530706:
        z += 0.05418329 * Q.sj3_dr23 - 0.0287554
    if Q.mass_2photon < 5.763861:
        z += -0.01575169 * Q.mass_2photon + 0.09079058
    if Q.pair_max_lnkt < 2.911067:
        z += -0.2574596 * Q.pair_max_lnkt + 0.7494821
    if Q.M3 < 0.05919662:
        z += 11.47691 * Q.M3 - 0.6793941
    if Q.lne_4 < 3.692778:
        z += -0.2387795 * Q.lne_4 + 0.8817598
    if Q.z_top20_slots >= 0.9417195:
        z += -5.275215 * Q.z_top20_slots + 4.967773
    if Q.e3 < 0.0006496195:
        z += 231.3096 * Q.e3 - 0.1502633
    if Q.phi_26 >= 0.1617432:
        z += -0.0552164 * Q.phi_26 + 0.008930875
    if Q.tau32_b2 >= 0.6908801:
        z += 1.628828 * Q.tau32_b2 - 1.125325
    if Q.mass_top5 >= 37.66803:
        z += 0.003318506 * Q.mass_top5 - 0.1250016
    if Q.LHA < 0.4888886:
        z += 3.125525 * Q.LHA - 1.528033
    if Q.n_dr_0_0p05 < 12.0:
        z += 0.005618092 * Q.n_dr_0_0p05 - 0.06741711
    if Q.m012 >= 11.86162:
        z += -0.003049929 * Q.m012 + 0.03617709
    if Q.mass_displaced3 < 39.09615 and Q.tau43 < 0.8709334:
        z += -0.02758361 * (39.09615 - Q.mass_displaced3) * (0.8709334 - Q.tau43)
    if Q.mass_displaced3 < 39.09615 and Q.sum_z_dr2_top20 > 0.0399789:
        z += -0.03759228 * (39.09615 - Q.mass_displaced3) * (Q.sum_z_dr2_top20 - 0.0399789)
    if Q.mass_displaced3 < 39.09615 and Q.z_neutral_had < 0.3788785:
        z += -0.08260664 * (39.09615 - Q.mass_displaced3) * (0.3788785 - Q.z_neutral_had)
    if Q.mass_displaced3 < 39.09615 and Q.M3_b2 < 0.01013989:
        z += 0.6089678 * (39.09615 - Q.mass_displaced3) * (0.01013989 - Q.M3_b2)
    if Q.M2_b2 < 0.05966366 and Q.mass_charged > 37.42054:
        z += -0.1206353 * (0.05966366 - Q.M2_b2) * (Q.mass_charged - 37.42054)
    if Q.n_lund_kt_above_5 > 1.0 and Q.tau54 < 0.8953628:
        z += -0.2757699 * (Q.n_lund_kt_above_5 - 1.0) * (0.8953628 - Q.tau54)
    if Q.lep_z > 0.03021637 and Q.n_lepton < 2.0:
        z += -0.9842609 * (Q.lep_z - 0.03021637) * (2.0 - Q.n_lepton)
    if Q.sj4_pair_mass_max < 107.8099 and Q.sum_pt_top20 < 515.5711:
        z += 8.792845e-06 * (107.8099 - Q.sj4_pair_mass_max) * (515.5711 - Q.sum_pt_top20)
    if Q.tau2 < 0.09733903 and Q.lep_ptrel < 27.3236:
        z += 0.1790475 * (0.09733903 - Q.tau2) * (27.3236 - Q.lep_ptrel)
    if Q.n_lund_kt_above_5 > 1.0 and Q.z_displaced5 > 0.1046203:
        z += 0.4196558 * (Q.n_lund_kt_above_5 - 1.0) * (Q.z_displaced5 - 0.1046203)
    if Q.mass_top30 < 109.3251 and Q.lnptrel_12 > -5.283873:
        z += 0.004017694 * (109.3251 - Q.mass_top30) * (Q.lnptrel_12 - -5.283873)
    if Q.n_s3d_above_3 > 4.0 and Q.sip_3d_3 < 175.9957:
        z += 0.0005148737 * (Q.n_s3d_above_3 - 4.0) * (175.9957 - Q.sip_3d_3)
    if Q.mass_displaced3 < 39.09615 and Q.e3_b2 < 0.0007909605:
        z += -21.12309 * (39.09615 - Q.mass_displaced3) * (0.0007909605 - Q.e3_b2)
    if Q.mass < 164.4374 and Q.z_top20_slots > 0.7122459:
        z += -0.04619865 * (164.4374 - Q.mass) * (Q.z_top20_slots - 0.7122459)
    if Q.mass < 164.4374 and Q.sj4_pair_mass_min > 13.57513:
        z += -5.992511e-05 * (164.4374 - Q.mass) * (Q.sj4_pair_mass_min - 13.57513)
    if Q.mass_top50 < 125.4927 and Q.n_lepton < 1.0:
        z += 0.006358953 * (125.4927 - Q.mass_top50) * (1.0 - Q.n_lepton)
    if Q.mass_2charged < 6.767937 and Q.n_real_top15 < 15.0:
        z += 0.002574962 * (6.767937 - Q.mass_2charged) * (15.0 - Q.n_real_top15)
    if Q.n_s3d_above_3 > 4.0 and Q.z_charged_had > 0.3603262:
        z += -0.2117297 * (Q.n_s3d_above_3 - 4.0) * (Q.z_charged_had - 0.3603262)
    if Q.sj3_mass1 < 36.36236 and Q.sj3_mass2 > 9.068761:
        z += -0.0004618676 * (36.36236 - Q.sj3_mass1) * (Q.sj3_mass2 - 9.068761)
    if Q.n_lund_kt_above_5 > 1.0 and Q.n_electron > 0.0:
        z += 0.01284576 * (Q.n_lund_kt_above_5 - 1.0) * (Q.n_electron - 0.0)
    if Q.sj3_pair_mass_min < 37.19471 and Q.sj3_z2 < 0.3118983:
        z += 0.0593738 * (37.19471 - Q.sj3_pair_mass_min) * (0.3118983 - Q.sj3_z2)
    if Q.M2_b2 < 0.05966366 and Q.d0err_52 > 0.0:
        z += -21.9391 * (0.05966366 - Q.M2_b2) * (Q.d0err_52 - 0.0)
    if Q.sj3_pairmax_over_m < 0.8207647 and Q.isphoton_53 > 0.0:
        z += 1.133726 * (0.8207647 - Q.sj3_pairmax_over_m) * (Q.isphoton_53 - 0.0)
    if Q.n_lund_kt_above_5 > 1.0 and Q.sj4_zsoft < 0.06029776:
        z += 1.033385 * (Q.n_lund_kt_above_5 - 1.0) * (0.06029776 - Q.sj4_zsoft)
    if Q.mass_displaced3 < 39.09615 and Q.isphoton_12 > 0.0:
        z += -0.001044018 * (39.09615 - Q.mass_displaced3) * (Q.isphoton_12 - 0.0)
    if Q.n_s3d_above_3 > 4.0 and Q.tdz_44 > 0.01757784:
        z += -0.016652 * (Q.n_s3d_above_3 - 4.0) * (Q.tdz_44 - 0.01757784)
    if Q.n_lund_kt_above_5 > 1.0 and Q.iselectron_34 > 0.0:
        z += 0.06527967 * (Q.n_lund_kt_above_5 - 1.0) * (Q.iselectron_34 - 0.0)
    if Q.sj4_pair_mass_max < 107.8099 and Q.td0_9 > -0.01537965:
        z += -0.002223362 * (107.8099 - Q.sj4_pair_mass_max) * (Q.td0_9 - -0.01537965)
    if Q.sj4_pair_mass_max < 107.8099 and Q.tdz_25 < 0.0:
        z += -0.0005114205 * (107.8099 - Q.sj4_pair_mass_max) * (0.0 - Q.tdz_25)
    if Q.mass_2charged < 28.89659 and Q.phi_31 > 0.326416:
        z += 0.002281819 * (28.89659 - Q.mass_2charged) * (Q.phi_31 - 0.326416)
    if Q.sj4_pair_mass_max < 107.8099 and Q.charge_43 > -1.0:
        z += 6.617662e-05 * (107.8099 - Q.sj4_pair_mass_max) * (Q.charge_43 - -1.0)
    if Q.N2_b2 < 0.2099278 and Q.dzerr_43 < 0.05511475:
        z += 21.33903 * (0.2099278 - Q.N2_b2) * (0.05511475 - Q.dzerr_43)
    if Q.tau2 < 0.09733903 and Q.isphoton_41 < 1.0:
        z += 2.876657 * (0.09733903 - Q.tau2) * (1.0 - Q.isphoton_41)
    return z


def neuron_116(Q):
    z = -3.39926e-07
    return z


def neuron_117(Q):
    z = 2.86214e-06
    return z


def neuron_118(Q):
    z = -8.23318e-06
    return z


def neuron_119(Q):
    z = -6.337149e-07
    return z


def neuron_120(Q):
    z = 1.924121
    if Q.lep_ptrel < 3.53503:
        z += 0.1993098 * Q.lep_ptrel - 1.097496
    if 3.53503 <= Q.lep_ptrel < 6.983043:
        z += 0.04559812 * Q.lep_ptrel - 0.5541211
    if 6.983043 <= Q.lep_ptrel < 12.15228:
        z += 0.04296951 * Q.lep_ptrel - 0.5357654
    if 12.15228 <= Q.lep_ptrel < 18.7678:
        z += -0.002628613 * Q.lep_ptrel + 0.01835572
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += 0.001405575 * Q.lep_ptrel - 0.05735711
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.03021795 * Q.lep_ptrel - 0.844615
    if Q.lep_ptrel >= 43.20788:
        z += 0.04810753 * Q.lep_ptrel - 1.617586
    if Q.n_pairs_kt_above_3 < 62.0:
        z += 0.003847005 * Q.n_pairs_kt_above_3 - 0.2385143
    if Q.lep_z < 0.5187302:
        z += 1.528802 * Q.lep_z - 0.793036
    if Q.tau32 < 0.6800935:
        z += -0.6558715 * Q.tau32 + 0.446054
    if Q.n_sd0_above_5 < 3.0:
        z += -0.008625976 * Q.n_sd0_above_5 + 0.02587793
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.00451731 * Q.n_pairs_kt_above_1 - 0.5566782
    if 80.0 <= Q.n_pairs_kt_above_1 < 195.0:
        z += 0.001698204 * Q.n_pairs_kt_above_1 - 0.3311498
    if 48.96085 <= Q.mass_top10 < 103.4976:
        z += -0.003110445 * Q.mass_top10 + 0.15229
    if Q.mass_top10 >= 103.4976:
        z += -0.01079745 * Q.mass_top10 + 0.9478767
    z += 18.83398 * Q.ecf_g42
    if Q.pair_mean_lndelta >= -1.523797:
        z += -0.1600753 * Q.pair_mean_lndelta - 0.2439222
    if Q.mass_top15 >= 104.4937:
        z += -0.010511 * Q.mass_top15 + 1.098334
    if Q.N2_b05 < 0.4174214:
        z += 3.519451 * Q.N2_b05 - 1.469094
    if Q.n_electron < 2.0:
        z += -0.09342691 * Q.n_electron + 0.1868538
    if Q.pair_mean_lnkt >= 0.5586581:
        z += 0.2050835 * Q.pair_mean_lnkt - 0.1145716
    if Q.e4_b05 >= 1.4104e-05:
        z += 2556.663 * Q.e4_b05 - 0.03605917
    if Q.max_abs_d0 < 5.8125:
        z += -0.009372944 * Q.max_abs_d0 + 0.05448024
    if Q.z_charged_had < 0.3191471:
        z += 0.5194261 * Q.z_charged_had - 0.1657733
    if Q.lep_iso < 0.02781886:
        z += -8.551889 * Q.lep_iso + 0.6244249
    if 0.02781886 <= Q.lep_iso < 2.907433:
        z += -0.1342267 * Q.lep_iso + 0.3902552
    if Q.lund3_lndelta >= -2.817283:
        z += -0.1082367 * Q.lund3_lndelta - 0.3049334
    if Q.z_neutral >= 0.6734757:
        z += 1.331801 * Q.z_neutral - 0.8969355
    if Q.mass < 90.08945:
        z += 0.02006844 * Q.mass - 2.711728
    if 90.08945 <= Q.mass < 105.7234:
        z += 0.04453545 * Q.mass - 4.915948
    if 105.7234 <= Q.mass < 114.0172:
        z += 0.0250196 * Q.mass - 2.852666
    if Q.z_displaced3 >= 0.2092108:
        z += -0.1236077 * Q.z_displaced3 + 0.02586007
    if Q.n_dr_0p1_0p2 < 5.0:
        z += 0.01794743 * Q.n_dr_0p1_0p2 - 0.08973716
    if Q.n_charged < 25.0:
        z += -0.01646022 * Q.n_charged + 0.4115054
    if Q.dr_min_012 < 0.01394245:
        z += 4.009541 * Q.dr_min_012 - 0.05590282
    if Q.mass_charged < 68.20711:
        z += 0.004233123 * Q.mass_charged - 0.2887291
    z += -0.1807652 * Q.z_displaced5
    if Q.n_sd0_above_2 < 4.0:
        z += 0.05677367 * Q.n_sd0_above_2 - 0.2270947
    if Q.sd_rg >= 0.5562194:
        z += -1.073944 * Q.sd_rg + 0.5973484
    if Q.e3 >= 0.00228569:
        z += -5.91599 * Q.e3 + 0.01352212
    if 109.3251 <= Q.mass_top30 < 162.7874:
        z += -0.009279935 * Q.mass_top30 + 1.01453
    if Q.mass_top30 >= 162.7874:
        z += -0.01570666 * Q.mass_top30 + 2.06072
    if Q.D3_b05 < 0.3052386:
        z += -0.3074425 * Q.D3_b05 - 0.1469117
    if 0.3052386 <= Q.D3_b05 < 0.8860453:
        z += 0.4145182 * Q.D3_b05 - 0.367282
    if 135.5368 <= Q.mass_top50 < 161.1264:
        z += -0.00483849 * Q.mass_top50 + 0.6557934
    if Q.mass_top50 >= 161.1264:
        z += 0.005923627 * Q.mass_top50 - 1.078268
    if Q.e2_b05 < 0.08995834:
        z += 4.560124 * Q.e2_b05 - 0.4102212
    if Q.pt_entropy < 1.938659:
        z += -0.5579469 * Q.pt_entropy + 1.081669
    if Q.sj3_z3 < 0.1719087:
        z += 1.663477 * Q.sj3_z3 - 0.2859663
    if Q.n_neutral_had < 3.0:
        z += -0.09586816 * Q.n_neutral_had + 0.2876045
    if Q.sip_3d_2 < 226.3008:
        z += -0.0001872293 * Q.sip_3d_2 + 0.04237013
    if Q.n_dr_0p4_up < 4.0:
        z += -0.003543192 * Q.n_dr_0p4_up + 0.01417277
    if Q.lnptrel_3 < -3.360241:
        z += 0.1624099 * Q.lnptrel_3 + 0.5457362
    if Q.n_pt_above_5 < 16.0:
        z += 0.007336041 * Q.n_pt_above_5 - 0.1173767
    if Q.M3_b05 >= 0.08175231:
        z += -7.704159 * Q.M3_b05 + 0.6298328
    if Q.sip_3d_3 < 577.991:
        z += 0.0003397109 * Q.sip_3d_3 - 0.1963498
    if Q.M2_b05 < 0.1258758:
        z += 3.049137 * Q.M2_b05 - 0.3838126
    if Q.sj3_mass2 < 1.824785:
        z += -0.03270923 * Q.sj3_mass2 + 0.05968733
    if Q.dr_4 < 0.1132155:
        z += 0.08045574 * Q.dr_4 - 0.009108834
    if Q.n_s3d_above_10 < 5.0:
        z += -0.008793592 * Q.n_s3d_above_10 + 0.04396796
    if Q.lep_ptrel < 3.53503 and Q.n_s3d_above_3 < 10.0:
        z += -0.04717723 * (3.53503 - Q.lep_ptrel) * (10.0 - Q.n_s3d_above_3)
    if Q.lep_z < 0.5187302 and Q.mass_neutral < 69.1565:
        z += 0.005151952 * (0.5187302 - Q.lep_z) * (69.1565 - Q.mass_neutral)
    if Q.lep_z < 0.5187302 and Q.mass_displaced3 > 0.0:
        z += 0.03262241 * (0.5187302 - Q.lep_z) * (Q.mass_displaced3 - 0.0)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.sip_3d_3 < 4.606241:
        z += -0.0003337005 * (62.0 - Q.n_pairs_kt_above_3) * (4.606241 - Q.sip_3d_3)
    if Q.lep_ptrel < 3.53503 and Q.sip_3d_3 < 577.991:
        z += 0.0003436018 * (3.53503 - Q.lep_ptrel) * (577.991 - Q.sip_3d_3)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.n_lund > 5.0:
        z += -0.0003039378 * (62.0 - Q.n_pairs_kt_above_3) * (Q.n_lund - 5.0)
    if Q.lep_ptrel < 3.53503 and Q.n_photon > 5.0:
        z += 0.005150038 * (3.53503 - Q.lep_ptrel) * (Q.n_photon - 5.0)
    if Q.lep_z < 0.5187302 and Q.mass_top10 > 48.96085:
        z += 0.007806225 * (0.5187302 - Q.lep_z) * (Q.mass_top10 - 48.96085)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.z_displaced3 < 0.2601259:
        z += 0.02558294 * (62.0 - Q.n_pairs_kt_above_3) * (0.2601259 - Q.z_displaced3)
    if Q.lep_ptrel > 27.3236 and Q.mratio_min_012 < 0.1991254:
        z += -0.09230705 * (Q.lep_ptrel - 27.3236) * (0.1991254 - Q.mratio_min_012)
    if Q.lep_z < 0.5187302 and Q.lep_dr > 0.114659:
        z += 0.3494648 * (0.5187302 - Q.lep_z) * (Q.lep_dr - 0.114659)
    if Q.lep_ptrel < 12.15228 and Q.planar_flow < 0.7095008:
        z += 0.02404515 * (12.15228 - Q.lep_ptrel) * (0.7095008 - Q.planar_flow)
    if Q.mass_top10 > 103.4976 and Q.log_sum_pt > 6.426552:
        z += 0.03003915 * (Q.mass_top10 - 103.4976) * (Q.log_sum_pt - 6.426552)
    if Q.e4_b05 > 1.4104e-05 and Q.sum_pt > 550.4644:
        z += 12.77075 * (Q.e4_b05 - 1.4104e-05) * (Q.sum_pt - 550.4644)
    if Q.lep_ptrel > 18.7678 and Q.lep_iso < 0.4381892:
        z += -0.08829837 * (Q.lep_ptrel - 18.7678) * (0.4381892 - Q.lep_iso)
    if Q.lep_ptrel < 3.53503 and Q.lep_iso < 0.02781886:
        z += -7.489669 * (3.53503 - Q.lep_ptrel) * (0.02781886 - Q.lep_iso)
    if Q.N2_b05 < 0.4174214 and Q.sj3_dr_max < 0.8930677:
        z += 0.7756577 * (0.4174214 - Q.N2_b05) * (0.8930677 - Q.sj3_dr_max)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.z_neutral_had > 0.02425618:
        z += -0.02347702 * (62.0 - Q.n_pairs_kt_above_3) * (Q.z_neutral_had - 0.02425618)
    if Q.n_dr_0p1_0p2 < 5.0 and Q.eta_2 < -0.217041:
        z += -0.3600948 * (5.0 - Q.n_dr_0p1_0p2) * (-0.217041 - Q.eta_2)
    if Q.lund3_lndelta > -2.817283 and Q.sj3_dr13 > 0.577548:
        z += -0.1405817 * (Q.lund3_lndelta - -2.817283) * (Q.sj3_dr13 - 0.577548)
    if Q.lep_ptrel < 12.15228 and Q.sj3_z3 < 0.1889947:
        z += 0.1015639 * (12.15228 - Q.lep_ptrel) * (0.1889947 - Q.sj3_z3)
    if Q.mass < 90.08945 and Q.iselectron_1 > 0.0:
        z += -0.008937104 * (90.08945 - Q.mass) * (Q.iselectron_1 - 0.0)
    if Q.max_abs_d0 < 5.8125 and Q.ismuon_0 < 1.0:
        z += 0.014087 * (5.8125 - Q.max_abs_d0) * (1.0 - Q.ismuon_0)
    if Q.n_sd0_above_2 < 4.0 and Q.dr_19 < 0.3967994:
        z += -0.01707033 * (4.0 - Q.n_sd0_above_2) * (0.3967994 - Q.dr_19)
    if Q.lep_iso < 2.907433 and Q.iselectron_12 > 0.0:
        z += -0.06618237 * (2.907433 - Q.lep_iso) * (Q.iselectron_12 - 0.0)
    if Q.mass < 105.7234 and Q.eta_0 > 0.04705811:
        z += -0.08427604 * (105.7234 - Q.mass) * (Q.eta_0 - 0.04705811)
    if Q.sj3_z3 < 0.1719087 and Q.ismuon_11 > 0.0:
        z += -0.5975968 * (0.1719087 - Q.sj3_z3) * (Q.ismuon_11 - 0.0)
    if Q.sj3_z3 < 0.1719087 and Q.isphoton_42 < 1.0:
        z += 0.3228154 * (0.1719087 - Q.sj3_z3) * (1.0 - Q.isphoton_42)
    if Q.dr_min_012 < 0.01394245 and Q.ischhad_26 > 0.0:
        z += -0.4272157 * (0.01394245 - Q.dr_min_012) * (Q.ischhad_26 - 0.0)
    if Q.lep_ptrel < 12.15228 and Q.phi_22 < -0.2314453:
        z += 0.01579508 * (12.15228 - Q.lep_ptrel) * (-0.2314453 - Q.phi_22)
    if Q.mass_top10 > 48.96085 and Q.eta_7 > -0.1883545:
        z += 0.002389064 * (Q.mass_top10 - 48.96085) * (Q.eta_7 - -0.1883545)
    if Q.lep_ptrel < 12.15228 and Q.dr_16 < 0.5715866:
        z += -0.008258008 * (12.15228 - Q.lep_ptrel) * (0.5715866 - Q.dr_16)
    if Q.lep_ptrel > 6.983043 and Q.jet_charge_k03 > 0.7410651:
        z += -0.008283705 * (Q.lep_ptrel - 6.983043) * (Q.jet_charge_k03 - 0.7410651)
    if Q.mass_top10 > 48.96085 and Q.iselectron_14 > 0.0:
        z += 0.001084168 * (Q.mass_top10 - 48.96085) * (Q.iselectron_14 - 0.0)
    if Q.pair_mean_lnkt > 0.5586581 and Q.lund3_lnz > -5.757953:
        z += -0.002297302 * (Q.pair_mean_lnkt - 0.5586581) * (Q.lund3_lnz - -5.757953)
    if Q.max_abs_d0 < 5.8125 and Q.d0err_12 > 0.01499939:
        z += 0.1760488 * (5.8125 - Q.max_abs_d0) * (Q.d0err_12 - 0.01499939)
    if Q.M3_b05 > 0.08175231 and Q.td0_13 > 0.01621867:
        z += 10.49194 * (Q.M3_b05 - 0.08175231) * (Q.td0_13 - 0.01621867)
    if Q.sj3_z3 < 0.1719087 and Q.sj2_mass2 < 24.55881:
        z += 0.0262448 * (0.1719087 - Q.sj3_z3) * (24.55881 - Q.sj2_mass2)
    return z


def neuron_121(Q):
    z = -5.730505e-06
    return z


def neuron_122(Q):
    z = -2.395913e-06
    return z


def neuron_123(Q):
    z = 0.06679438
    if Q.lep_ptrel < 6.983043:
        z += 0.01159125 * Q.lep_ptrel - 0.08094222
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += -0.02570867 * Q.lep_ptrel + 0.4824951
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += -0.05578548 * Q.lep_ptrel + 1.304302
    if Q.lep_ptrel >= 43.20788:
        z += 0.01397499 * Q.lep_ptrel - 1.7099
    if Q.M2_b2 < 0.02926638:
        z += -2.044961 * Q.M2_b2 + 0.05984861
    if Q.z_displaced5 < 0.3803178:
        z += -0.9244781 * Q.z_displaced5 + 0.3515955
    if Q.tau32 < 0.6800935:
        z += -0.8886574 * Q.tau32 + 0.6043702
    if Q.n_pairs_kt_above_1 < 58.0:
        z += -0.009518657 * Q.n_pairs_kt_above_1 + 0.8236081
    if 58.0 <= Q.n_pairs_kt_above_1 < 123.0:
        z += -0.004177323 * Q.n_pairs_kt_above_1 + 0.5138107
    if Q.lep_z < 0.06457187:
        z += 4.423222 * Q.lep_z - 0.2856157
    if Q.lep_z >= 0.221436:
        z += -1.067293 * Q.lep_z + 0.2363372
    if Q.mass_top5 >= 68.52153:
        z += 0.01019122 * Q.mass_top5 - 0.698318
    if 79.27954 <= Q.mass_top50 < 84.74733:
        z += 0.00368624 * Q.mass_top50 - 0.2922435
    if 84.74733 <= Q.mass_top50 < 129.5874:
        z += -0.0003390061 * Q.mass_top50 + 0.04888543
    if Q.mass_top50 >= 129.5874:
        z += -0.001312153 * Q.mass_top50 + 0.1749931
    if Q.n_dr_0p1_0p2 < 4.0:
        z += 0.01771998 * Q.n_dr_0p1_0p2 - 0.07087993
    if 137.452 <= Q.mass < 164.4374:
        z += -0.003605484 * Q.mass + 0.495581
    if Q.mass >= 164.4374:
        z += -0.0007671525 * Q.mass + 0.02885298
    if Q.M2_b05 >= 0.1080247:
        z += 2.782478 * Q.M2_b05 - 0.3005764
    if Q.n_pairs_kt_above_3 < 41.0:
        z += -0.005566443 * Q.n_pairs_kt_above_3 + 0.2282242
    if Q.n_s3d_above_3 >= 1.0:
        z += 0.04682355 * Q.n_s3d_above_3 - 0.04682355
    if Q.sj3_dr12 >= 0.3470463:
        z += 0.2446736 * Q.sj3_dr12 - 0.08491306
    if Q.sip_3d_3 < 4.606241:
        z += -0.1105233 * Q.sip_3d_3 + 0.509097
    if Q.mass_displaced3 < 18.80005:
        z += -0.004802174 * Q.mass_displaced3 + 0.09028113
    if Q.sum_z_dr2_top3 < 0.002792418:
        z += 168.8217 * Q.sum_z_dr2_top3 - 0.4714208
    if Q.n_lund < 5.0:
        z += -0.1225912 * Q.n_lund + 0.612956
    if Q.mass_neutral < 14.92637:
        z += 0.02246021 * Q.mass_neutral - 0.3352494
    if Q.z_charged < 0.3265243:
        z += 3.369686 * Q.z_charged - 1.100285
    if Q.sj3_mass2 < 1.824785:
        z += 0.06821388 * Q.sj3_mass2 - 0.1244757
    if Q.sj3_pair_mass_max < 76.91486:
        z += -0.001046768 * Q.sj3_pair_mass_max + 0.08051199
    if Q.M3_b05 >= 0.08175231:
        z += 3.401983 * Q.M3_b05 - 0.27812
    if Q.pair_mean_lnz < -2.572052:
        z += -0.5742754 * Q.pair_mean_lnz - 1.477066
    if Q.z_charged_had < 0.4239562:
        z += -0.694399 * Q.z_charged_had + 0.2943947
    if Q.e3_b2 < 4.696862e-06:
        z += -34411.36 * Q.e3_b2 + 0.1616254
    if Q.ecf_g41 >= 0.0002327102:
        z += -158.4738 * Q.ecf_g41 + 0.03687846
    if Q.z_displaced3 < 0.4232894:
        z += 1.249488 * Q.z_displaced3 - 0.5288951
    if Q.n_sdz_above_5 < 4.0:
        z += 0.01903415 * Q.n_sdz_above_5 - 0.07613658
    if Q.sj2_mass2 < 1.851735:
        z += -0.06142026 * Q.sj2_mass2 + 0.1137341
    if Q.jet_charge_k05 < -0.4474182:
        z += -0.118085 * Q.jet_charge_k05 - 0.05283341
    if Q.jet_charge_k05 >= 0.5864519:
        z += 0.5809962 * Q.jet_charge_k05 - 0.3407263
    if Q.mass_2photon >= 22.18431:
        z += 0.01498444 * Q.mass_2photon - 0.3324196
    if Q.pair_mean_lnm2 >= 4.069088:
        z += 0.122549 * Q.pair_mean_lnm2 - 0.4986628
    if Q.pair_max_lnm2 >= 8.097646:
        z += -0.6037381 * Q.pair_max_lnm2 + 4.888857
    if Q.C3_b2 < 0.0004257509:
        z += -345.5683 * Q.C3_b2 + 0.147126
    if Q.sj3_pair_mass_min >= 37.19471:
        z += 0.002805075 * Q.sj3_pair_mass_min - 0.1043339
    if Q.z_displaced5 < 0.3803178 and Q.lep_z < 0.06457187:
        z += 24.37303 * (0.3803178 - Q.z_displaced5) * (0.06457187 - Q.lep_z)
    if Q.M2_b2 < 0.02926638 and Q.dr_1 > 0.1779524:
        z += -6.095033 * (0.02926638 - Q.M2_b2) * (Q.dr_1 - 0.1779524)
    if Q.z_displaced5 < 0.3803178 and Q.M2_b05 > 0.1080247:
        z += 0.9235422 * (0.3803178 - Q.z_displaced5) * (Q.M2_b05 - 0.1080247)
    if Q.M2_b2 < 0.02926638 and Q.n_s3d_above_3 < 10.0:
        z += -0.08791349 * (0.02926638 - Q.M2_b2) * (10.0 - Q.n_s3d_above_3)
    if Q.z_displaced5 < 0.3803178 and Q.lep_z < 0.5187302:
        z += -2.221768 * (0.3803178 - Q.z_displaced5) * (0.5187302 - Q.lep_z)
    if Q.n_pairs_kt_above_1 < 58.0 and Q.n_lund > 7.0:
        z += 0.0009896625 * (58.0 - Q.n_pairs_kt_above_1) * (Q.n_lund - 7.0)
    if Q.n_dr_0p1_0p2 < 4.0 and Q.z_charged_had < 0.4239562:
        z += 0.3105851 * (4.0 - Q.n_dr_0p1_0p2) * (0.4239562 - Q.z_charged_had)
    if Q.z_displaced5 < 0.3803178 and Q.lep_dr > 0.1864963:
        z += -0.1121702 * (0.3803178 - Q.z_displaced5) * (Q.lep_dr - 0.1864963)
    if Q.z_displaced5 < 0.3803178 and Q.n_lepton > 1.0:
        z += 0.1384507 * (0.3803178 - Q.z_displaced5) * (Q.n_lepton - 1.0)
    if Q.lep_ptrel > 18.7678 and Q.n_s3d_above_3 < 2.0:
        z += -0.009143594 * (Q.lep_ptrel - 18.7678) * (2.0 - Q.n_s3d_above_3)
    if Q.tau32 < 0.6800935 and Q.sj3_dr13 < 0.7462286:
        z += -2.09433 * (0.6800935 - Q.tau32) * (0.7462286 - Q.sj3_dr13)
    if Q.lep_ptrel > 18.7678 and Q.sip_3d_1 < 82.17305:
        z += 3.79406e-05 * (Q.lep_ptrel - 18.7678) * (82.17305 - Q.sip_3d_1)
    if Q.lep_ptrel > 27.3236 and Q.n_s3d_above_3 < 2.0:
        z += 0.009599762 * (Q.lep_ptrel - 27.3236) * (2.0 - Q.n_s3d_above_3)
    if Q.lep_ptrel > 18.7678 and Q.ecf_g42 > 2.448333e-05:
        z += 34.1523 * (Q.lep_ptrel - 18.7678) * (Q.ecf_g42 - 2.448333e-05)
    if Q.n_pairs_kt_above_3 < 41.0 and Q.sip_3d_3 < 7.345216:
        z += -0.001278673 * (41.0 - Q.n_pairs_kt_above_3) * (7.345216 - Q.sip_3d_3)
    if Q.n_pairs_kt_above_3 < 41.0 and Q.sip_3d_2 < 226.3008:
        z += 3.373544e-05 * (41.0 - Q.n_pairs_kt_above_3) * (226.3008 - Q.sip_3d_2)
    if Q.lep_ptrel > 18.7678 and Q.e3_b2 < 0.0004126585:
        z += 69.05524 * (Q.lep_ptrel - 18.7678) * (0.0004126585 - Q.e3_b2)
    if Q.n_pairs_kt_above_3 < 41.0 and Q.iselectron_1 > 0.0:
        z += 0.00597047 * (41.0 - Q.n_pairs_kt_above_3) * (Q.iselectron_1 - 0.0)
    if Q.lep_z < 0.06457187 and Q.sip_3d_3 < 4.606241:
        z += 0.1184394 * (0.06457187 - Q.lep_z) * (4.606241 - Q.sip_3d_3)
    if Q.n_s3d_above_3 > 1.0 and Q.sip_3d_1 < 578.2954:
        z += -3.501324e-05 * (Q.n_s3d_above_3 - 1.0) * (578.2954 - Q.sip_3d_1)
    if Q.n_s3d_above_3 > 1.0 and Q.n_lund_kt_above_5 < 4.0:
        z += 0.005695084 * (Q.n_s3d_above_3 - 1.0) * (4.0 - Q.n_lund_kt_above_5)
    if Q.n_pairs_kt_above_3 < 41.0 and Q.n_lund < 12.0:
        z += -0.002846306 * (41.0 - Q.n_pairs_kt_above_3) * (12.0 - Q.n_lund)
    if Q.n_s3d_above_3 > 1.0 and Q.e3_b2 < 0.0002536827:
        z += -218.1211 * (Q.n_s3d_above_3 - 1.0) * (0.0002536827 - Q.e3_b2)
    if Q.mass_top50 > 79.27954 and Q.z_charged_had < 0.3603262:
        z += 0.0149078 * (Q.mass_top50 - 79.27954) * (0.3603262 - Q.z_charged_had)
    if Q.mass_top5 > 68.52153 and Q.dr12 < 0.08983921:
        z += -0.07065984 * (Q.mass_top5 - 68.52153) * (0.08983921 - Q.dr12)
    if Q.lep_ptrel > 43.20788 and Q.dr_4 > 0.2339164:
        z += -0.06825168 * (Q.lep_ptrel - 43.20788) * (Q.dr_4 - 0.2339164)
    if Q.n_s3d_above_3 > 1.0 and Q.dr_4 < 0.1253926:
        z += 0.2839896 * (Q.n_s3d_above_3 - 1.0) * (0.1253926 - Q.dr_4)
    if Q.mass_top50 > 129.5874 and Q.sj3_dr13 > 0.2640447:
        z += -0.00522146 * (Q.mass_top50 - 129.5874) * (Q.sj3_dr13 - 0.2640447)
    if Q.tau32 < 0.6800935 and Q.dr_22 < 0.02915846:
        z += -40.71728 * (0.6800935 - Q.tau32) * (0.02915846 - Q.dr_22)
    if Q.sip_3d_3 < 4.606241 and Q.lnpt_5 < 3.824557:
        z += 0.0105974 * (4.606241 - Q.sip_3d_3) * (3.824557 - Q.lnpt_5)
    if Q.lep_ptrel > 27.3236 and Q.sj3_pairmax_over_m < 0.7366642:
        z += 0.2311601 * (Q.lep_ptrel - 27.3236) * (0.7366642 - Q.sj3_pairmax_over_m)
    if Q.lep_ptrel > 18.7678 and Q.eta_8 < 0.2429199:
        z += 0.002835338 * (Q.lep_ptrel - 18.7678) * (0.2429199 - Q.eta_8)
    if Q.lep_ptrel > 43.20788 and Q.lnpt_41 > -18.42068:
        z += -5.228454e-05 * (Q.lep_ptrel - 43.20788) * (Q.lnpt_41 - -18.42068)
    if Q.z_displaced5 < 0.3803178 and Q.lnptrel_40 > -18.42068:
        z += -0.01079762 * (0.3803178 - Q.z_displaced5) * (Q.lnptrel_40 - -18.42068)
    if Q.n_pairs_kt_above_3 < 41.0 and Q.mratio_min_012 < 0.3078028:
        z += 0.004180532 * (41.0 - Q.n_pairs_kt_above_3) * (0.3078028 - Q.mratio_min_012)
    if Q.M3_b05 > 0.08175231 and Q.sd_zg > 0.2450652:
        z += -60.56827 * (Q.M3_b05 - 0.08175231) * (Q.sd_zg - 0.2450652)
    if Q.tau32 < 0.6800935 and Q.eta_29 > 0.3234863:
        z += 1.838504 * (0.6800935 - Q.tau32) * (Q.eta_29 - 0.3234863)
    if Q.n_s3d_above_3 > 1.0 and Q.eta_19 < -0.1665039:
        z += 0.01288114 * (Q.n_s3d_above_3 - 1.0) * (-0.1665039 - Q.eta_19)
    if Q.lep_z > 0.221436 and Q.dr_25 < 0.1591656:
        z += -5.500914 * (Q.lep_z - 0.221436) * (0.1591656 - Q.dr_25)
    if Q.n_pairs_kt_above_1 < 123.0 and Q.ischhad_4 > 0.0:
        z += -0.0009817437 * (123.0 - Q.n_pairs_kt_above_1) * (Q.ischhad_4 - 0.0)
    if Q.lep_z > 0.221436 and Q.sj3_dr13 < 0.676889:
        z += -0.7398689 * (Q.lep_z - 0.221436) * (0.676889 - Q.sj3_dr13)
    if Q.tau32 < 0.6800935 and Q.dr_10 > 0.5063302:
        z += 1.012039 * (0.6800935 - Q.tau32) * (Q.dr_10 - 0.5063302)
    if Q.lep_ptrel > 18.7678 and Q.lund1_lnz < -6.508771:
        z += 0.001057781 * (Q.lep_ptrel - 18.7678) * (-6.508771 - Q.lund1_lnz)
    if Q.lep_z < 0.06457187 and Q.lund1_lnz > -4.349248:
        z += -0.01778954 * (0.06457187 - Q.lep_z) * (Q.lund1_lnz - -4.349248)
    if Q.lep_ptrel < 6.983043 and Q.sip_3d_3 < 4.606241:
        z += -0.007314817 * (6.983043 - Q.lep_ptrel) * (4.606241 - Q.sip_3d_3)
    if Q.lep_ptrel < 6.983043 and Q.C3_b2 < 0.0004257509:
        z += -52.48661 * (6.983043 - Q.lep_ptrel) * (0.0004257509 - Q.C3_b2)
    if Q.z_charged_had < 0.4239562 and Q.charge_27 < 0.0:
        z += 0.1331408 * (0.4239562 - Q.z_charged_had) * (0.0 - Q.charge_27)
    if Q.n_s3d_above_3 > 1.0 and Q.ismuon_14 > 0.0:
        z += -0.08878534 * (Q.n_s3d_above_3 - 1.0) * (Q.ismuon_14 - 0.0)
    if Q.tau32 < 0.6800935 and Q.C3_b2 < 0.01108077:
        z += -51.6044 * (0.6800935 - Q.tau32) * (0.01108077 - Q.C3_b2)
    if Q.M3_b05 > 0.08175231 and Q.sip_3d_3 < 7.345216:
        z += 0.1282703 * (Q.M3_b05 - 0.08175231) * (7.345216 - Q.sip_3d_3)
    if Q.sj3_pair_mass_max < 76.91486 and Q.sip_3d_3 < 11.13866:
        z += -0.0001083702 * (76.91486 - Q.sj3_pair_mass_max) * (11.13866 - Q.sip_3d_3)
    if Q.sj3_pair_mass_max < 76.91486 and Q.sip_3d_3 < 175.9957:
        z += -8.33051e-06 * (76.91486 - Q.sj3_pair_mass_max) * (175.9957 - Q.sip_3d_3)
    return z


def neuron_124(Q):
    z = 0.03345455
    if Q.lep_z < 0.221436:
        z += -1.427652 * Q.lep_z + 0.3654385
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -0.4170577 * Q.lep_z + 0.1416567
    if Q.n_pairs_kt_above_1 < 58.0:
        z += 0.00809402 * Q.n_pairs_kt_above_1 - 0.9954961
    if 58.0 <= Q.n_pairs_kt_above_1 < 366.0:
        z += 0.001226502 * Q.n_pairs_kt_above_1 - 0.59718
    if 366.0 <= Q.n_pairs_kt_above_1 < 702.0:
        z += 0.0004413105 * Q.n_pairs_kt_above_1 - 0.3098
    if Q.n_s3d_above_3 < 1.0:
        z += 0.01739091 * Q.n_s3d_above_3 + 0.2397653
    if 1.0 <= Q.n_s3d_above_3 < 2.0:
        z += -0.2571562 * Q.n_s3d_above_3 + 0.5143124
    if Q.n_s3d_above_3 >= 3.0:
        z += 0.04046856 * Q.n_s3d_above_3 - 0.1214057
    if Q.mass < 95.14961:
        z += -0.003107693 * Q.mass - 0.07417896
    if 95.14961 <= Q.mass < 100.4835:
        z += 0.02267267 * Q.mass - 2.52717
    if 100.4835 <= Q.mass < 117.4867:
        z += 0.01464089 * Q.mass - 1.720109
    if Q.mass >= 182.8592:
        z += 0.01257683 * Q.mass - 2.299789
    if Q.lep_ptrel < 43.20788:
        z += -0.01097225 * Q.lep_ptrel + 0.4740879
    if Q.n_dr_0p4_up < 15.0:
        z += -0.01104729 * Q.n_dr_0p4_up + 0.1657094
    if Q.lep_iso < 0.4381892:
        z += 0.7219034 * Q.lep_iso - 0.6312434
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += 0.3408501 * Q.lep_iso - 0.46427
    if Q.n_lepton < 1.0:
        z += -0.7916545 * Q.n_lepton + 0.7916545
    if Q.mass_top40 < 70.88236:
        z += 0.01693371 * Q.mass_top40 - 1.644633
    if 70.88236 <= Q.mass_top40 < 97.12186:
        z += 0.006713643 * Q.mass_top40 - 0.9202107
    if Q.mass_top40 >= 97.12186:
        z += -0.01022006 * Q.mass_top40 + 0.7244222
    if Q.z_displaced3 < 0.03624058:
        z += 8.618506 * Q.z_displaced3 - 0.3615337
    if 0.03624058 <= Q.z_displaced3 < 0.08304558:
        z += 1.051042 * Q.z_displaced3 - 0.08728435
    if Q.sj3_pair_mass_max < 63.7508:
        z += -0.0004092581 * Q.sj3_pair_mass_max + 0.1613741
    if 63.7508 <= Q.sj3_pair_mass_max < 73.24742:
        z += 0.004016249 * Q.sj3_pair_mass_max - 0.1207555
    if 73.24742 <= Q.sj3_pair_mass_max < 93.87663:
        z += -0.008406737 * Q.sj3_pair_mass_max + 0.7891961
    if Q.n_s3d_above_10 < 8.0:
        z += -0.07206722 * Q.n_s3d_above_10 + 0.5765377
    if Q.pair_max_lnm2 >= 6.520267:
        z += 0.04150859 * Q.pair_max_lnm2 - 0.2706471
    if Q.sip_3d_3 < 577.991:
        z += -0.0004718728 * Q.sip_3d_3 + 0.2727383
    z += 0.7261834 * Q.M3
    if Q.sj2_dr < 0.5076533:
        z += 0.8265992 * Q.sj2_dr - 0.4196259
    if 42.31629 <= Q.sd_mass < 78.4753:
        z += -0.0006963187 * Q.sd_mass + 0.02946562
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += -0.03994689 * Q.sd_mass + 3.109666
    if 88.81751 <= Q.sd_mass < 94.55722:
        z += -0.002007837 * Q.sd_mass - 0.2599863
    if 94.55722 <= Q.sd_mass < 106.7501:
        z += 0.02878753 * Q.sd_mass - 3.17191
    if 106.7501 <= Q.sd_mass < 127.8042:
        z += 0.01620363 * Q.sd_mass - 1.828579
    if 127.8042 <= Q.sd_mass < 154.5947:
        z += -0.03720457 * Q.sd_mass + 4.997217
    if Q.sd_mass >= 154.5947:
        z += 0.006871662 * Q.sd_mass - 1.816734
    if Q.mass_over_sum_pt_sq < 0.0729277:
        z += -8.213966 * Q.mass_over_sum_pt_sq + 0.5990257
    if Q.z_displaced5 >= 0.2801368:
        z += -1.379708 * Q.z_displaced5 + 0.3865071
    if Q.ecf_g31 < 0.01365334:
        z += 4.84184 * Q.ecf_g31 - 0.06610728
    if Q.psi_0p1 < 0.006220408:
        z += 22.78358 * Q.psi_0p1 - 0.1417232
    if Q.sj3_mass2 < 1.824785:
        z += 0.1622712 * Q.sj3_mass2 - 0.2961101
    if Q.sum_z_dr2_top3 < 0.007018285:
        z += -22.19693 * Q.sum_z_dr2_top3 + 0.1557843
    if Q.LHA < 0.4008517:
        z += -0.6311436 * Q.LHA + 0.252995
    if Q.mass_displaced3 < 1.777286:
        z += 0.06108526 * Q.mass_displaced3 - 0.203885
    if 1.777286 <= Q.mass_displaced3 < 4.41005:
        z += 0.03620493 * Q.mass_displaced3 - 0.1596655
    if Q.mass_top15 < 45.8749:
        z += -0.002024867 * Q.mass_top15 + 0.09289057
    if Q.N3_b05 >= 0.4404318:
        z += -0.0540006 * Q.N3_b05 + 0.02378358
    if Q.lep_dr >= 0.4191372:
        z += 1.466866 * Q.lep_dr - 0.614818
    if Q.n_dr_0p1_0p2 < 6.0:
        z += 0.0300557 * Q.n_dr_0p1_0p2 - 0.1803342
    if Q.ecf_g41 < 0.0002944259:
        z += -2810.57 * Q.ecf_g41 + 0.8275044
    if Q.lam1 < 0.06041764:
        z += 3.606967 * Q.lam1 - 0.2179244
    if Q.tdz_2 >= -0.02221619:
        z += -0.1449353 * Q.tdz_2 - 0.00321991
    if Q.td0_17 < -0.05792002:
        z += -0.07586066 * Q.td0_17 - 0.004393851
    if Q.z_dr_0p1_0p2 < 0.2428609:
        z += -0.330258 * Q.z_dr_0p1_0p2 + 0.08020677
    if Q.tau32 < 0.3164143:
        z += 0.08510495 * Q.tau32 - 0.02692842
    if Q.n_charged_had < 25.0:
        z += 0.0007747655 * Q.n_charged_had - 0.01936914
    if Q.M3_b05 < 0.09942631:
        z += 8.897381 * Q.M3_b05 - 0.8846337
    if Q.n_lund >= 7.0:
        z += -0.01288354 * Q.n_lund + 0.09018478
    if Q.n_charged_pt_above_1 < 15.0:
        z += -0.02021192 * Q.n_charged_pt_above_1 + 0.3031789
    if Q.lep_ptrel < 43.20788 and Q.sj3_pair_mass_max < 142.9952:
        z += -6.584794e-05 * (43.20788 - Q.lep_ptrel) * (142.9952 - Q.sj3_pair_mass_max)
    if Q.n_s3d_above_3 > 3.0 and Q.M2_b2 < 0.06762785:
        z += -1.131658 * (Q.n_s3d_above_3 - 3.0) * (0.06762785 - Q.M2_b2)
    if Q.n_s3d_above_3 > 3.0 and Q.max_abs_d0 < 10.52344:
        z += -0.01668146 * (Q.n_s3d_above_3 - 3.0) * (10.52344 - Q.max_abs_d0)
    if Q.n_pairs_kt_above_1 < 366.0 and Q.D2_b2 < 15.17086:
        z += -5.608626e-05 * (366.0 - Q.n_pairs_kt_above_1) * (15.17086 - Q.D2_b2)
    if Q.lep_z < 0.3396572 and Q.z_displaced5 > 0.1046203:
        z += -1.327693 * (0.3396572 - Q.lep_z) * (Q.z_displaced5 - 0.1046203)
    if Q.mass_top40 > 70.88236 and Q.lnerel_17 > -6.957861:
        z += 0.002636841 * (Q.mass_top40 - 70.88236) * (Q.lnerel_17 - -6.957861)
    if Q.mass_top40 > 70.88236 and Q.ecf_g42 < 4.726476e-05:
        z += -91.17167 * (Q.mass_top40 - 70.88236) * (4.726476e-05 - Q.ecf_g42)
    if Q.mass < 100.4835 and Q.n_real_top30 < 30.0:
        z += -0.0004371855 * (100.4835 - Q.mass) * (30.0 - Q.n_real_top30)
    if Q.n_s3d_above_3 > 3.0 and Q.z_photon > 0.05998812:
        z += -0.1756757 * (Q.n_s3d_above_3 - 3.0) * (Q.z_photon - 0.05998812)
    if Q.n_s3d_above_10 < 8.0 and Q.n_lund_kt_above_1 > 3.0:
        z += 0.001227055 * (8.0 - Q.n_s3d_above_10) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.n_dr_0p4_up < 15.0 and Q.jet_charge_k03 > 0.1164034:
        z += 0.02092195 * (15.0 - Q.n_dr_0p4_up) * (Q.jet_charge_k03 - 0.1164034)
    if Q.mass_top40 > 70.88236 and Q.lne_4 < 3.808744:
        z += -0.00180022 * (Q.mass_top40 - 70.88236) * (3.808744 - Q.lne_4)
    if Q.n_s3d_above_10 < 8.0 and Q.z_neutral_had < 0.2675458:
        z += -0.3458587 * (8.0 - Q.n_s3d_above_10) * (0.2675458 - Q.z_neutral_had)
    if Q.sj2_dr < 0.5076533 and Q.lne_2 < 5.114244:
        z += -0.228053 * (0.5076533 - Q.sj2_dr) * (5.114244 - Q.lne_2)
    if Q.sj2_dr < 0.5076533 and Q.lep_iso < 0.4381892:
        z += -0.09692482 * (0.5076533 - Q.sj2_dr) * (0.4381892 - Q.lep_iso)
    if Q.sj2_dr < 0.5076533 and Q.lund1_lnz > -4.349248:
        z += 0.1822384 * (0.5076533 - Q.sj2_dr) * (Q.lund1_lnz - -4.349248)
    if Q.n_pairs_kt_above_1 < 702.0 and Q.ismuon_37 > 0.0:
        z += 0.001328612 * (702.0 - Q.n_pairs_kt_above_1) * (Q.ismuon_37 - 0.0)
    if Q.n_s3d_above_10 < 8.0 and Q.eta_76 > 0.0:
        z += -0.04543041 * (8.0 - Q.n_s3d_above_10) * (Q.eta_76 - 0.0)
    if Q.mass_displaced3 < 1.777286 and Q.n_neutral_had < 9.0:
        z += 0.003539058 * (1.777286 - Q.mass_displaced3) * (9.0 - Q.n_neutral_had)
    if Q.lep_ptrel < 43.20788 and Q.sj3_z3 > 0.09421497:
        z += -0.0006845809 * (43.20788 - Q.lep_ptrel) * (Q.sj3_z3 - 0.09421497)
    if Q.sj2_dr < 0.5076533 and Q.iselectron_0 > 0.0:
        z += -0.6961346 * (0.5076533 - Q.sj2_dr) * (Q.iselectron_0 - 0.0)
    if Q.sj2_dr < 0.5076533 and Q.z_muon < 0.1403354:
        z += 2.542412 * (0.5076533 - Q.sj2_dr) * (0.1403354 - Q.z_muon)
    if Q.n_pairs_kt_above_1 < 366.0 and Q.iselectron_1 > 0.0:
        z += -0.0001276721 * (366.0 - Q.n_pairs_kt_above_1) * (Q.iselectron_1 - 0.0)
    if Q.pair_max_lnm2 > 6.520267 and Q.td0_14 < -0.06205547:
        z += 0.07540005 * (Q.pair_max_lnm2 - 6.520267) * (-0.06205547 - Q.td0_14)
    if Q.mass < 117.4867 and Q.ismuon_49 > 0.0:
        z += -0.003651751 * (117.4867 - Q.mass) * (Q.ismuon_49 - 0.0)
    if Q.n_s3d_above_3 > 3.0 and Q.charge_42 > 0.0:
        z += 0.009205126 * (Q.n_s3d_above_3 - 3.0) * (Q.charge_42 - 0.0)
    if Q.lep_ptrel < 43.20788 and Q.tdz_2 < -0.1462819:
        z += -0.007360788 * (43.20788 - Q.lep_ptrel) * (-0.1462819 - Q.tdz_2)
    if Q.tau32 < 0.3164143 and Q.d0err_14 < 0.02580261:
        z += 99.114 * (0.3164143 - Q.tau32) * (0.02580261 - Q.d0err_14)
    if Q.lep_z < 0.3396572 and Q.tau43_b2 < 0.3329206:
        z += -0.03764158 * (0.3396572 - Q.lep_z) * (0.3329206 - Q.tau43_b2)
    if Q.n_s3d_above_10 < 8.0 and Q.d0err_16 < 0.019104:
        z += 0.07805255 * (8.0 - Q.n_s3d_above_10) * (0.019104 - Q.d0err_16)
    return z


def neuron_125(Q):
    z = -1.190753e-06
    return z


def neuron_126(Q):
    z = 7.6326e-06
    return z


def neuron_127(Q):
    z = -1.966841e-07
    return z


def class_token(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q), neuron_16(Q), neuron_17(Q), neuron_18(Q), neuron_19(Q), neuron_20(Q), neuron_21(Q), neuron_22(Q), neuron_23(Q), neuron_24(Q), neuron_25(Q), neuron_26(Q), neuron_27(Q), neuron_28(Q), neuron_29(Q), neuron_30(Q), neuron_31(Q), neuron_32(Q), neuron_33(Q), neuron_34(Q), neuron_35(Q), neuron_36(Q), neuron_37(Q), neuron_38(Q), neuron_39(Q), neuron_40(Q), neuron_41(Q), neuron_42(Q), neuron_43(Q), neuron_44(Q), neuron_45(Q), neuron_46(Q), neuron_47(Q), neuron_48(Q), neuron_49(Q), neuron_50(Q), neuron_51(Q), neuron_52(Q), neuron_53(Q), neuron_54(Q), neuron_55(Q), neuron_56(Q), neuron_57(Q), neuron_58(Q), neuron_59(Q), neuron_60(Q), neuron_61(Q), neuron_62(Q), neuron_63(Q), neuron_64(Q), neuron_65(Q), neuron_66(Q), neuron_67(Q), neuron_68(Q), neuron_69(Q), neuron_70(Q), neuron_71(Q), neuron_72(Q), neuron_73(Q), neuron_74(Q), neuron_75(Q), neuron_76(Q), neuron_77(Q), neuron_78(Q), neuron_79(Q), neuron_80(Q), neuron_81(Q), neuron_82(Q), neuron_83(Q), neuron_84(Q), neuron_85(Q), neuron_86(Q), neuron_87(Q), neuron_88(Q), neuron_89(Q), neuron_90(Q), neuron_91(Q), neuron_92(Q), neuron_93(Q), neuron_94(Q), neuron_95(Q), neuron_96(Q), neuron_97(Q), neuron_98(Q), neuron_99(Q), neuron_100(Q), neuron_101(Q), neuron_102(Q), neuron_103(Q), neuron_104(Q), neuron_105(Q), neuron_106(Q), neuron_107(Q), neuron_108(Q), neuron_109(Q), neuron_110(Q), neuron_111(Q), neuron_112(Q), neuron_113(Q), neuron_114(Q), neuron_115(Q), neuron_116(Q), neuron_117(Q), neuron_118(Q), neuron_119(Q), neuron_120(Q), neuron_121(Q), neuron_122(Q), neuron_123(Q), neuron_124(Q), neuron_125(Q), neuron_126(Q), neuron_127(Q)]


def logits(h):
    return [B[c] + sum(h[j] * W[j][c] for j in range(len(h))) for c in range(10)]


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
