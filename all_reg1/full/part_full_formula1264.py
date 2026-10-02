"""Particle Transformer (ParT) jet tagger, JetClass, input features 'full': tuned on the network's predictions (from 100 if-statements per neuron; the smallest without loss on the validation jets), as if-statements.

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

Whole test file (2,000,000 jets): accuracy 74.52% (the network: 86.03%); same class as the network for 79.16% of jets.

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
    z = 0.005458094
    return z


def neuron_1(Q):
    z = -9.452106e-05
    return z


def neuron_2(Q):
    z = 2.702e-05
    return z


def neuron_3(Q):
    z = 0.0001025292
    return z


def neuron_4(Q):
    z = 6.796698e-05
    return z


def neuron_5(Q):
    z = 8.155766e-05
    return z


def neuron_6(Q):
    z = 4.519576e-05
    return z


def neuron_7(Q):
    z = -8.874838e-05
    return z


def neuron_8(Q):
    z = -8.492194e-05
    return z


def neuron_9(Q):
    z = 8.877052e-05
    return z


def neuron_10(Q):
    z = -0.0001938225
    return z


def neuron_11(Q):
    z = -0.0001444988
    return z


def neuron_12(Q):
    z = 0.0001555606
    return z


def neuron_13(Q):
    z = -0.0001455579
    return z


def neuron_14(Q):
    z = -6.701636e-05
    return z


def neuron_15(Q):
    z = 5.399063e-05
    return z


def neuron_16(Q):
    z = 4.56704
    if Q.lep_ptrel < 27.3236:
        z += 0.02550307 * Q.lep_ptrel + 1.364792
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.07545225 * Q.lep_ptrel
    if Q.lep_ptrel >= 43.20788:
        z += -0.01730303 * Q.lep_ptrel + 4.007759
    if Q.n_s3d_above_3 < 10.0:
        z += 0.1189977 * Q.n_s3d_above_3 - 1.189977
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.01075585 * Q.n_pairs_kt_above_3 + 0.647381
    if 28.0 <= Q.n_pairs_kt_above_3 < 41.0:
        z += -0.02663209 * Q.n_pairs_kt_above_3 + 1.091916
    if Q.sip_3d_2 < 226.3008:
        z += -0.002508773 * Q.sip_3d_2 + 0.4244133
    if 226.3008 <= Q.sip_3d_2 < 447.0873:
        z += 0.0006491523 * Q.sip_3d_2 - 0.2902278
    if Q.lund_max_lndelta >= -1.062087:
        z += 0.05649639 * Q.lund_max_lndelta + 0.0600041
    if Q.z_displaced5 < 0.06245248:
        z += 12.1203 * Q.z_displaced5 - 1.454055
    if 0.06245248 <= Q.z_displaced5 < 0.2801368:
        z += 3.202398 * Q.z_displaced5 - 0.8971096
    if Q.mass_neutral < 19.57354:
        z += -0.003026132 * Q.mass_neutral + 0.266046
    if 19.57354 <= Q.mass_neutral < 87.9162:
        z += -0.0001391878 * Q.mass_neutral + 0.2095383
    if Q.mass_neutral >= 87.9162:
        z += 0.002886944 * Q.mass_neutral - 0.05650771
    if Q.mass_top40 < 92.68149:
        z += -0.0003536567 * Q.mass_top40 + 0.03277744
    if Q.mass_top40 >= 122.7145:
        z += 0.002378782 * Q.mass_top40 - 0.2919112
    if Q.e3_b2 < 0.0002536827:
        z += -240.3681 * Q.e3_b2 + 0.06097723
    if 119.4443 <= Q.sd_mass < 154.5947:
        z += 0.01412218 * Q.sd_mass - 1.686815
    if 154.5947 <= Q.sd_mass < 175.9333:
        z += -0.01059355 * Q.sd_mass + 2.134105
    if Q.sd_mass >= 175.9333:
        z += -0.008448967 * Q.sd_mass + 1.756802
    if Q.max_abs_d0 < 5.8125:
        z += 0.0006974512 * Q.max_abs_d0 - 0.004053935
    if Q.sip_3d_3 < 16.23838:
        z += 0.03972099 * Q.sip_3d_3 - 0.6606463
    if 16.23838 <= Q.sip_3d_3 < 33.08364:
        z += 0.02782697 * Q.sip_3d_3 - 0.4675066
    if 33.08364 <= Q.sip_3d_3 < 577.991:
        z += -0.0008315372 * Q.sip_3d_3 + 0.480621
    z += -7.318044 * Q.z_charged_had
    if 125.4927 <= Q.mass_top50 < 161.1264:
        z += 0.005679581 * Q.mass_top50 - 0.7127458
    if Q.mass_top50 >= 161.1264:
        z += -0.002511121 * Q.mass_top50 + 0.6069927
    if Q.n_charged_had < 29.0:
        z += -0.02959519 * Q.n_charged_had + 0.8582606
    z += 0.01725372 * Q.n_photon
    if Q.mass_top5 >= 62.84493:
        z += 0.006927385 * Q.mass_top5 - 0.4353511
    if Q.n_electron >= 1.0:
        z += -0.1023717 * Q.n_electron + 0.1023717
    if Q.n_lund_kt_above_5 >= 1.0:
        z += -0.01176413 * Q.n_lund_kt_above_5 + 0.01176413
    z += 0.07194 * Q.n_sd0_above_2
    if Q.mass_displaced3 < 0.2490748:
        z += -0.1020951 * Q.mass_displaced3 - 0.05250008
    if 0.2490748 <= Q.mass_displaced3 < 1.777286:
        z += 0.2580518 * Q.mass_displaced3 - 0.1422036
    if 1.777286 <= Q.mass_displaced3 < 4.41005:
        z += -0.09094786 * Q.mass_displaced3 + 0.4780685
    if 4.41005 <= Q.mass_displaced3 < 13.03663:
        z += -0.008924033 * Q.mass_displaced3 + 0.1163393
    if Q.z_dr_0p05_0p1 < 0.01556887:
        z += -8.828995 * Q.z_dr_0p05_0p1 + 0.1374575
    if 0.01245833 <= Q.lep_z < 0.221436:
        z += -5.035411 * Q.lep_z + 0.06273281
    if Q.lep_z >= 0.221436:
        z += -7.952167 * Q.lep_z + 0.7086077
    z += -0.004165521 * Q.mass_charged
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.01073481 * Q.n_pairs_kt_above_1 + 0.858785
    if Q.tau4 < 0.02720087:
        z += 15.81914 * Q.tau4 - 0.4302944
    if Q.n_pairs_kt_above_10 >= 17.0:
        z += 0.005942529 * Q.n_pairs_kt_above_10 - 0.101023
    z += -6.706351 * Q.z_neutral
    if Q.n_dr_0p2_0p4 < 7.0:
        z += -0.04395214 * Q.n_dr_0p2_0p4 + 0.307665
    if Q.lne_4 < 3.452465:
        z += -0.1005562 * Q.lne_4 + 0.3471666
    if Q.n_pt_above_1 < 33.0:
        z += 0.02728635 * Q.n_pt_above_1 - 0.9004496
    if Q.M3 >= 0.02540381:
        z += 9.058324 * Q.M3 - 0.230116
    if Q.psi_0p3 >= 0.9828705:
        z += -9.446654 * Q.psi_0p3 + 9.284838
    if Q.M2 < 0.09740996:
        z += -9.950003 * Q.M2 + 0.9692294
    if Q.z_top15_slots >= 0.8008865:
        z += 0.7387123 * Q.z_top15_slots - 0.5916247
    if Q.C2 >= 0.1323775:
        z += 1.913784 * Q.C2 - 0.253342
    if Q.tau3 >= 0.06348273:
        z += -3.407491 * Q.tau3 + 0.2163168
    if Q.sj3_pair_mass_max >= 83.63398:
        z += 0.006809203 * Q.sj3_pair_mass_max - 0.5694808
    if Q.jet_charge_k05 < -0.3599791:
        z += -0.2724244 * Q.jet_charge_k05 - 0.0980671
    if Q.jet_charge_k05 >= 0.5864519:
        z += 0.5111696 * Q.jet_charge_k05 - 0.2997764
    if Q.mass_2photon >= 22.18431:
        z += 0.00214121 * Q.mass_2photon - 0.04750127
    if Q.min_pair_mass < 1.763283:
        z += 0.1165871 * Q.min_pair_mass - 0.205576
    if Q.n_sdz_above_2 < 5.0:
        z += 0.08615635 * Q.n_sdz_above_2 - 0.4307817
    if Q.z_dr_0p4_up < 0.0265067:
        z += 3.205348 * Q.z_dr_0p4_up - 0.0849632
    if Q.C2_b05 < 0.2413007:
        z += -3.238902 * Q.C2_b05 + 0.7815494
    if Q.D2 < 0.7107791:
        z += 0.1912808 * Q.D2 - 0.1359584
    if Q.max_pair_mass >= 55.24488:
        z += -0.01374509 * Q.max_pair_mass + 0.7593462
    if Q.N2_b05 < 0.302405:
        z += 0.03388828 * Q.N2_b05 - 0.01024798
    if Q.tdz_1 < -0.1170754:
        z += -0.1602963 * Q.tdz_1 - 0.01876675
    if Q.sj3_mass2 < 1.824785:
        z += -0.02853742 * Q.sj3_mass2 + 0.05207467
    if Q.max_abs_dz < 0.6455078:
        z += 0.3445937 * Q.max_abs_dz + 0.005532502
    if 0.6455078 <= Q.max_abs_dz < 2.957031:
        z += -0.09862346 * Q.max_abs_dz + 0.2916327
    if Q.mass_neutral < 87.9162 and Q.M2_b2 < 0.06762785:
        z += -0.1786848 * (87.9162 - Q.mass_neutral) * (0.06762785 - Q.M2_b2)
    if Q.mass_neutral < 87.9162 and Q.mass_2charged > 17.49066:
        z += 5.687904e-05 * (87.9162 - Q.mass_neutral) * (Q.mass_2charged - 17.49066)
    if Q.z_displaced5 < 0.2801368 and Q.sip_3d_3 < 175.9957:
        z += 0.01219098 * (0.2801368 - Q.z_displaced5) * (175.9957 - Q.sip_3d_3)
    if Q.mass_neutral < 87.9162 and Q.n_lund > 5.0:
        z += 0.0004955261 * (87.9162 - Q.mass_neutral) * (Q.n_lund - 5.0)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.lep_dr < 0.2338497:
        z += -0.02802332 * (28.0 - Q.n_pairs_kt_above_3) * (0.2338497 - Q.lep_dr)
    if Q.lund_max_lndelta > -1.062087 and Q.n_charged_had < 15.0:
        z += 0.1004448 * (Q.lund_max_lndelta - -1.062087) * (15.0 - Q.n_charged_had)
    if Q.n_s3d_above_3 < 10.0 and Q.n_dr_0p4_up < 15.0:
        z += 2.43536e-05 * (10.0 - Q.n_s3d_above_3) * (15.0 - Q.n_dr_0p4_up)
    if Q.n_charged_had < 29.0 and Q.n_lepton > 1.0:
        z += 0.001357098 * (29.0 - Q.n_charged_had) * (Q.n_lepton - 1.0)
    if Q.n_charged_had < 29.0 and Q.ecf_g42 > 2.701529e-06:
        z += -167.906 * (29.0 - Q.n_charged_had) * (Q.ecf_g42 - 2.701529e-06)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.M3_b05 > 0.06007167:
        z += 0.3515005 * (28.0 - Q.n_pairs_kt_above_3) * (Q.M3_b05 - 0.06007167)
    if Q.max_abs_d0 < 5.8125 and Q.sip_3d_3 < 49.03695:
        z += 0.001451665 * (5.8125 - Q.max_abs_d0) * (49.03695 - Q.sip_3d_3)
    if Q.n_s3d_above_3 < 10.0 and Q.D2_b2 < 7.782712:
        z += -0.006722427 * (10.0 - Q.n_s3d_above_3) * (7.782712 - Q.D2_b2)
    if Q.mass_neutral < 87.9162 and Q.dr_max_012 < 0.02981375:
        z += -0.06667657 * (87.9162 - Q.mass_neutral) * (0.02981375 - Q.dr_max_012)
    if Q.mass_neutral < 87.9162 and Q.n_dr_0p1_0p2 < 6.0:
        z += 0.0008992195 * (87.9162 - Q.mass_neutral) * (6.0 - Q.n_dr_0p1_0p2)
    if Q.z_displaced5 < 0.2801368 and Q.sip_3d_3 < 33.08364:
        z += 0.1006625 * (0.2801368 - Q.z_displaced5) * (33.08364 - Q.sip_3d_3)
    if Q.lund_max_lndelta > -1.062087 and Q.sip_3d_3 < 4.606241:
        z += -0.2983956 * (Q.lund_max_lndelta - -1.062087) * (4.606241 - Q.sip_3d_3)
    if Q.lep_z > 0.221436 and Q.tau32 < 0.7544983:
        z += 0.3912866 * (Q.lep_z - 0.221436) * (0.7544983 - Q.tau32)
    if Q.mass_displaced3 < 13.03663 and Q.iselectron_1 > 0.0:
        z += 0.002084483 * (13.03663 - Q.mass_displaced3) * (Q.iselectron_1 - 0.0)
    if Q.lund_max_lndelta > -1.062087 and Q.sj4_pair_mass_min < 34.83233:
        z += 0.0118625 * (Q.lund_max_lndelta - -1.062087) * (34.83233 - Q.sj4_pair_mass_min)
    if Q.n_lund_kt_above_5 > 1.0 and Q.sip_3d_3 > 49.03695:
        z += -1.515731e-05 * (Q.n_lund_kt_above_5 - 1.0) * (Q.sip_3d_3 - 49.03695)
    if Q.z_dr_0p05_0p1 < 0.01556887 and Q.tau32_b2 < 0.3377343:
        z += 19.79788 * (0.01556887 - Q.z_dr_0p05_0p1) * (0.3377343 - Q.tau32_b2)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.D2 < 1.516502:
        z += -0.07254394 * (7.0 - Q.n_dr_0p2_0p4) * (1.516502 - Q.D2)
    if Q.tau3 > 0.06348273 and Q.phi_84 < 0.0:
        z += 0.4053883 * (Q.tau3 - 0.06348273) * (0.0 - Q.phi_84)
    if Q.sip_3d_2 < 447.0873 and Q.d0err_12 > 0.0236969:
        z += 0.005161783 * (447.0873 - Q.sip_3d_2) * (Q.d0err_12 - 0.0236969)
    if Q.mass_top50 > 125.4927 and Q.tdz_68 < 0.0:
        z += -0.0004443056 * (Q.mass_top50 - 125.4927) * (0.0 - Q.tdz_68)
    if Q.z_dr_0p05_0p1 < 0.01556887 and Q.eta_36 < 0.1083984:
        z += 7.156703 * (0.01556887 - Q.z_dr_0p05_0p1) * (0.1083984 - Q.eta_36)
    if Q.z_displaced5 < 0.06245248 and Q.tau54 < 0.8786609:
        z += -14.40912 * (0.06245248 - Q.z_displaced5) * (0.8786609 - Q.tau54)
    return z


def neuron_17(Q):
    z = 8.906888e-05
    return z


def neuron_18(Q):
    z = 0.6625802
    if Q.lep_z < 0.004135872:
        z += 204.2414 * Q.lep_z - 1.388823
    if 0.004135872 <= Q.lep_z < 0.221436:
        z += 2.50394 * Q.lep_z - 0.5544625
    if 110.2019 <= Q.mass < 164.4374:
        z += -0.0117719 * Q.mass + 1.297285
    if 164.4374 <= Q.mass < 182.8592:
        z += 0.02579644 * Q.mass - 4.880356
    if Q.mass >= 182.8592:
        z += 0.004389573 * Q.mass - 0.9659129
    if Q.mass_top20 < 114.4658:
        z += 0.005428988 * Q.mass_top20 - 0.6214336
    if Q.z_displaced3 < 0.03624058:
        z += 6.490083 * Q.z_displaced3 - 0.4424994
    if 0.03624058 <= Q.z_displaced3 < 0.06416437:
        z += 7.423598 * Q.z_displaced3 - 0.4763305
    if Q.n_s3d_above_10 < 6.0:
        z += 0.05220827 * Q.n_s3d_above_10 - 0.3132496
    if Q.lep_ptrel < 27.3236:
        z += 0.001215854 * Q.lep_ptrel - 0.0332215
    if Q.lund3_lndelta >= -2.817283:
        z += 0.005694311 * Q.lund3_lndelta + 0.01604248
    if Q.mass_displaced3 < 2.409613:
        z += -0.0005432591 * Q.mass_displaced3 - 0.1809839
    if 2.409613 <= Q.mass_displaced3 < 9.090532:
        z += 0.0272856 * Q.mass_displaced3 - 0.2480406
    if Q.z_displaced5 < 0.006242101:
        z += -7.319046 * Q.z_displaced5 + 0.4570926
    if 0.006242101 <= Q.z_displaced5 < 0.06245248:
        z += -7.742915 * Q.z_displaced5 + 0.4597384
    if Q.z_displaced5 >= 0.06245248:
        z += -0.4238686 * Q.z_displaced5 + 0.002645831
    if Q.sip_3d_1 < 3.373037:
        z += 0.2209375 * Q.sip_3d_1 - 0.7452302
    if Q.n_pairs_kt_above_1 < 146.0:
        z += -0.000486596 * Q.n_pairs_kt_above_1 + 0.1780941
    if 146.0 <= Q.n_pairs_kt_above_1 < 366.0:
        z += -0.001030652 * Q.n_pairs_kt_above_1 + 0.2575263
    if Q.n_pairs_kt_above_1 >= 366.0:
        z += -0.0005440558 * Q.n_pairs_kt_above_1 + 0.07943215
    if Q.sum_z_dr < 0.07290954:
        z += 7.417527 * Q.sum_z_dr - 0.5408084
    if 111.701 <= Q.sd_mass < 123.2919:
        z += 0.00991827 * Q.sd_mass - 1.10788
    if 123.2919 <= Q.sd_mass < 154.5947:
        z += -0.02415001 * Q.sd_mass + 3.092461
    if Q.sd_mass >= 154.5947:
        z += 0.008700096 * Q.sd_mass - 1.98599
    if Q.sj2_mass2 < 5.311506:
        z += -0.08739786 * Q.sj2_mass2 + 0.4642142
    if Q.sj2_mass2 >= 42.12131:
        z += -0.001504049 * Q.sj2_mass2 + 0.06335252
    if Q.tau3 >= 0.05398263:
        z += 0.4083923 * Q.tau3 - 0.02204609
    if Q.max_abs_d0 < 0.6289062:
        z += -0.6779399 * Q.max_abs_d0 + 0.4263606
    if Q.n_s3d_above_3 < 3.0:
        z += 0.6045177 * Q.n_s3d_above_3 - 1.813553
    if Q.sj3_pair_mass_max >= 128.6605:
        z += -0.01356335 * Q.sj3_pair_mass_max + 1.745068
    if Q.n_sd0_above_3 < 9.0:
        z += -0.04559682 * Q.n_sd0_above_3 + 0.4103714
    if Q.n_sdz_above_5 < 3.0:
        z += -0.07340014 * Q.n_sdz_above_5 + 0.2202004
    if Q.n_dr_0p4_up < 4.0:
        z += -0.0452266 * Q.n_dr_0p4_up + 0.1809064
    if Q.sj3_dr23 < 0.3084865:
        z += 0.3675505 * Q.sj3_dr23 - 0.1133844
    if Q.tau43 >= 0.6808537:
        z += -0.9561891 * Q.tau43 + 0.6510249
    if Q.lep_iso < 0.4381892:
        z += -0.5306327 * Q.lep_iso + 0.5349142
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += -0.07979843 * Q.lep_iso + 0.3373635
    if 1.362094 <= Q.lep_iso < 14.38858:
        z += -0.01755427 * Q.lep_iso + 0.252581
    if Q.N2 < 0.3829918:
        z += 2.427877 * Q.N2 - 0.9298572
    if Q.e4 < 2.097213e-06:
        z += -195982.5 * Q.e4 + 0.4110169
    if Q.n_lepton < 2.0:
        z += -0.1421771 * Q.n_lepton + 0.2843542
    if Q.max_abs_dz < 9.046875:
        z += 0.009908078 * Q.max_abs_dz - 0.08963715
    if Q.n_charged_had >= 27.0:
        z += -0.04927378 * Q.n_charged_had + 1.330392
    if Q.N2_b05 >= 0.4265629:
        z += 1.982572 * Q.N2_b05 - 0.8456918
    if Q.psi_0p2 < 0.2598003:
        z += -0.5678625 * Q.psi_0p2 + 0.1475308
    if Q.lund2_lndelta >= -1.124734:
        z += 0.1561534 * Q.lund2_lndelta + 0.175631
    if Q.lund_max_lndelta >= -0.6897565:
        z += -0.7506256 * Q.lund_max_lndelta - 0.5177489
    if Q.n_pairs_kt_above_3 < 111.0:
        z += -0.002701927 * Q.n_pairs_kt_above_3 + 0.2999139
    if Q.tau32 >= 0.5871757:
        z += -0.5136233 * Q.tau32 + 0.3015871
    if Q.mass_top5 >= 76.4159:
        z += -0.005432988 * Q.mass_top5 + 0.4151666
    if Q.sj3_pair_mass_min >= 44.1643:
        z += -0.004255142 * Q.sj3_pair_mass_min + 0.1879254
    if Q.mass_charged >= 113.5019:
        z += 0.003546997 * Q.mass_charged - 0.402591
    if Q.z_charged < 0.3887012:
        z += -1.178287 * Q.z_charged + 0.4580013
    if Q.pt1_over_pt0 < 0.3206143:
        z += -1.753839 * Q.pt1_over_pt0 + 0.5623058
    if Q.sj3_pairmax_over_m >= 0.897453:
        z += -0.7626414 * Q.sj3_pairmax_over_m + 0.6844347
    if Q.mass_top40 < 132.5189:
        z += 0.005414137 * Q.mass_top40 - 0.5393276
    if 132.5189 <= Q.mass_top40 < 155.8928:
        z += -0.007621669 * Q.mass_top40 + 1.188163
    if Q.n_neutral >= 12.0:
        z += -0.02569798 * Q.n_neutral + 0.3083758
    z += 0.03758172 * Q.n_for_90pct
    if Q.mass_neutral >= 14.92637:
        z += 0.005470541 * Q.mass_neutral - 0.08165531
    if Q.lund_max_lnkt >= 3.22013:
        z += -0.1031484 * Q.lund_max_lnkt + 0.3321513
    if Q.sum_z_dr2_top2 < 0.01091199:
        z += 22.99147 * Q.sum_z_dr2_top2 - 0.2508826
    if Q.N3_b2 < 0.4545826:
        z += 0.183417 * Q.N3_b2 - 0.08337817
    if Q.e3_b2 < 0.0001729927:
        z += -596.2822 * Q.e3_b2 - 0.3103811
    if 0.0001729927 <= Q.e3_b2 < 0.0007909605:
        z += 669.183 * Q.e3_b2 - 0.5292973
    if Q.eta_1 < -0.2019043:
        z += 0.006597539 * Q.eta_1 + 0.001332071
    if Q.n_lund >= 13.0:
        z += -0.01918769 * Q.n_lund + 0.24944
    if Q.sip_3d_3 < 49.03695:
        z += -0.00489206 * Q.sip_3d_3 + 0.2398917
    if Q.sip_3d_2 < 226.3008:
        z += 0.0001633952 * Q.sip_3d_2 - 0.03697647
    if Q.n_sd0_above_2 < 6.0:
        z += 0.07822365 * Q.n_sd0_above_2 - 0.4693419
    if Q.n_charged_pt_above_1 < 27.0:
        z += -0.03023186 * Q.n_charged_pt_above_1 + 0.8162601
    if Q.lep_z < 0.221436 and Q.N2 < 0.3829918:
        z += 16.01175 * (0.221436 - Q.lep_z) * (0.3829918 - Q.N2)
    if Q.z_displaced3 < 0.03624058 and Q.z_charged_had > 0.1891627:
        z += -14.62767 * (0.03624058 - Q.z_displaced3) * (Q.z_charged_had - 0.1891627)
    if Q.lep_z < 0.221436 and Q.z_displaced5 < 0.1046203:
        z += -40.81451 * (0.221436 - Q.lep_z) * (0.1046203 - Q.z_displaced5)
    if Q.n_s3d_above_10 < 6.0 and Q.lne_7 > 2.313463:
        z += 0.01395325 * (6.0 - Q.n_s3d_above_10) * (Q.lne_7 - 2.313463)
    if Q.mass > 110.2019 and Q.n_charged_pt_above_10 < 8.0:
        z += 5.911439e-06 * (Q.mass - 110.2019) * (8.0 - Q.n_charged_pt_above_10)
    if Q.n_s3d_above_10 < 6.0 and Q.z_neutral_had < 0.212136:
        z += 0.1926095 * (6.0 - Q.n_s3d_above_10) * (0.212136 - Q.z_neutral_had)
    if Q.n_s3d_above_3 < 3.0 and Q.sip_3d_2 < 117.0891:
        z += 0.00461262 * (3.0 - Q.n_s3d_above_3) * (117.0891 - Q.sip_3d_2)
    if Q.z_displaced5 < 0.06245248 and Q.sip_3d_2 < 4.636903:
        z += -2.733962 * (0.06245248 - Q.z_displaced5) * (4.636903 - Q.sip_3d_2)
    if Q.z_displaced5 > 0.006242101 and Q.sip_3d_2 < 447.0873:
        z += -0.0005375526 * (Q.z_displaced5 - 0.006242101) * (447.0873 - Q.sip_3d_2)
    if Q.mass_displaced3 < 9.090532 and Q.n_charged_pt_above_1 < 29.0:
        z += 0.002284837 * (9.090532 - Q.mass_displaced3) * (29.0 - Q.n_charged_pt_above_1)
    if Q.z_displaced5 > 0.006242101 and Q.n_dr_0p2_0p4 < 29.0:
        z += 0.01110506 * (Q.z_displaced5 - 0.006242101) * (29.0 - Q.n_dr_0p2_0p4)
    if Q.z_displaced5 < 0.06245248 and Q.n_pt_above_5 > 10.0:
        z += 0.3293965 * (0.06245248 - Q.z_displaced5) * (Q.n_pt_above_5 - 10.0)
    if Q.n_s3d_above_10 < 6.0 and Q.lund_max_lndelta > -1.670992:
        z += 0.05525769 * (6.0 - Q.n_s3d_above_10) * (Q.lund_max_lndelta - -1.670992)
    if Q.lep_z < 0.221436 and Q.e4_b05 > 4.363663e-05:
        z += -2674.419 * (0.221436 - Q.lep_z) * (Q.e4_b05 - 4.363663e-05)
    if Q.z_displaced5 > 0.006242101 and Q.dr12 < 0.2116473:
        z += 0.847868 * (Q.z_displaced5 - 0.006242101) * (0.2116473 - Q.dr12)
    if Q.lund3_lndelta > -2.817283 and Q.tau54 > 0.7608692:
        z += -0.7422594 * (Q.lund3_lndelta - -2.817283) * (Q.tau54 - 0.7608692)
    if Q.mass_displaced3 < 9.090532 and Q.lep_dr < 0.114659:
        z += -0.1737294 * (9.090532 - Q.mass_displaced3) * (0.114659 - Q.lep_dr)
    if Q.lep_z < 0.004135872 and Q.jet_abs_eta > 0.8317778:
        z += 85.42272 * (0.004135872 - Q.lep_z) * (Q.jet_abs_eta - 0.8317778)
    if Q.lep_z < 0.221436 and Q.jet_charge_k05 > 0.1901233:
        z += -1.771624 * (0.221436 - Q.lep_z) * (Q.jet_charge_k05 - 0.1901233)
    if Q.sj3_pair_mass_max > 128.6605 and Q.eccentricity > 0.9132117:
        z += -0.3724083 * (Q.sj3_pair_mass_max - 128.6605) * (Q.eccentricity - 0.9132117)
    if Q.mass > 182.8592 and Q.planar_flow < 0.4623202:
        z += 0.06182081 * (Q.mass - 182.8592) * (0.4623202 - Q.planar_flow)
    if Q.e4 < 2.097213e-06 and Q.lep_iso < 1.362094:
        z += -43289.03 * (2.097213e-06 - Q.e4) * (1.362094 - Q.lep_iso)
    if Q.sj3_pair_mass_max > 128.6605 and Q.lund1_lndelta < -0.3624079:
        z += -0.02329085 * (Q.sj3_pair_mass_max - 128.6605) * (-0.3624079 - Q.lund1_lndelta)
    if Q.N2 < 0.3829918 and Q.lnptrel_45 < -5.306837:
        z += 0.141487 * (0.3829918 - Q.N2) * (-5.306837 - Q.lnptrel_45)
    if Q.lep_ptrel < 27.3236 and Q.N2_b2 > 0.0395449:
        z += -0.05416097 * (27.3236 - Q.lep_ptrel) * (Q.N2_b2 - 0.0395449)
    if Q.lep_ptrel < 27.3236 and Q.eta_9 > 0.09516602:
        z += 0.002248861 * (27.3236 - Q.lep_ptrel) * (Q.eta_9 - 0.09516602)
    if Q.lep_z < 0.221436 and Q.pt1_over_pt0 < 0.380649:
        z += -5.176902 * (0.221436 - Q.lep_z) * (0.380649 - Q.pt1_over_pt0)
    if Q.z_charged < 0.3887012 and Q.mass_2photon > 1.081989:
        z += 0.02120249 * (0.3887012 - Q.z_charged) * (Q.mass_2photon - 1.081989)
    if Q.n_neutral > 12.0 and Q.eta_28 > -0.1078491:
        z += 0.001499048 * (Q.n_neutral - 12.0) * (Q.eta_28 - -0.1078491)
    if Q.sj3_pairmax_over_m > 0.897453 and Q.iselectron_29 > 0.0:
        z += 0.02548679 * (Q.sj3_pairmax_over_m - 0.897453) * (Q.iselectron_29 - 0.0)
    if Q.tau32 > 0.5871757 and Q.n_pt_above_50 < 4.0:
        z += -0.1797975 * (Q.tau32 - 0.5871757) * (4.0 - Q.n_pt_above_50)
    return z


def neuron_19(Q):
    z = 0.0001396965
    return z


def neuron_20(Q):
    z = 7.807752e-05
    return z


def neuron_21(Q):
    z = -0.001876607
    return z


def neuron_22(Q):
    z = 0.0001262092
    return z


def neuron_23(Q):
    z = 0.0001479703
    return z


def neuron_24(Q):
    z = -0.5903357
    if Q.lep_z < 0.221436:
        z += 5.006045 * Q.lep_z - 1.196215
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += 0.7418007 * Q.lep_z - 0.251958
    if Q.mass_top30 < 74.51927:
        z += 0.008948686 * Q.mass_top30 - 0.1867809
    if 74.51927 <= Q.mass_top30 < 134.2224:
        z += -0.008040933 * Q.mass_top30 + 1.079273
    if Q.mass < 79.47361:
        z += 0.02231542 * Q.mass - 2.308081
    if 79.47361 <= Q.mass < 90.08945:
        z += 0.001061923 * Q.mass - 0.6189891
    if 90.08945 <= Q.mass < 114.0172:
        z += -0.01076902 * Q.mass + 0.446854
    if 114.0172 <= Q.mass < 117.4867:
        z += 0.02041915 * Q.mass - 3.109135
    if 117.4867 <= Q.mass < 149.0507:
        z += 0.01502323 * Q.mass - 2.475187
    if 149.0507 <= Q.mass < 164.4374:
        z += 0.0153355 * Q.mass - 2.52173
    z += 0.1277587 * Q.pair_max_lnkt
    if Q.z_displaced3 < 0.1061578:
        z += -4.330916 * Q.z_displaced3 + 0.4597605
    if Q.e3_b2 < 0.0004126585:
        z += 1834.881 * Q.e3_b2 - 0.7571792
    if Q.lep_ptrel < 43.20788:
        z += -0.006496419 * Q.lep_ptrel + 0.2806965
    if Q.n_s3d_above_10 >= 1.0:
        z += -0.02568574 * Q.n_s3d_above_10 + 0.02568574
    if Q.mass_displaced3 < 26.78691:
        z += -0.02334223 * Q.mass_displaced3 + 0.6252662
    if Q.mass_displaced3 >= 39.09615:
        z += -0.02127162 * Q.mass_displaced3 + 0.8316386
    if Q.tau32 < 0.5068038:
        z += -0.9186386 * Q.tau32 + 0.4655695
    if Q.lep_iso < 0.4381892:
        z += -2.016884 * Q.lep_iso + 1.127327
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += -0.2636102 * Q.lep_iso + 0.3590619
    if Q.n_lepton < 1.0:
        z += 0.9914343 * Q.n_lepton - 0.9914343
    if Q.n_s3d_above_3 < 2.0:
        z += -0.2847055 * Q.n_s3d_above_3 + 0.569411
    if 2.0 <= Q.n_s3d_above_3 < 5.0:
        z += -0.09696025 * Q.n_s3d_above_3 + 0.1939205
    if Q.n_s3d_above_3 >= 5.0:
        z += 0.07425532 * Q.n_s3d_above_3 - 0.6621573
    z += 0.0002036169 * Q.sum_e
    if Q.n_sd0_above_10 < 2.0:
        z += 0.03169534 * Q.n_sd0_above_10 - 0.06339069
    if Q.n_sd0_above_10 >= 6.0:
        z += 0.02696203 * Q.n_sd0_above_10 - 0.1617722
    if Q.sj2_dr < 0.2083105:
        z += -3.718467 * Q.sj2_dr + 0.5834717
    if 0.2083105 <= Q.sj2_dr < 0.2403736:
        z += -5.35117 * Q.sj2_dr + 0.9235809
    if 0.2403736 <= Q.sj2_dr < 0.3923158:
        z += 2.703975 * Q.sj2_dr - 1.012664
    if 0.3923158 <= Q.sj2_dr < 0.5944498:
        z += -0.2382018 * Q.sj2_dr + 0.141599
    if Q.n_sd0_above_3 < 3.0:
        z += 0.1267939 * Q.n_sd0_above_3 - 0.7607633
    if 3.0 <= Q.n_sd0_above_3 < 6.0:
        z += 0.2081063 * Q.n_sd0_above_3 - 1.0047
    if Q.n_sd0_above_3 >= 6.0:
        z += 0.08131238 * Q.n_sd0_above_3 - 0.2439371
    if Q.psi_0p2 >= 0.8857951:
        z += -3.862717 * Q.psi_0p2 + 3.421576
    if Q.N2 < 0.258375:
        z += 4.206879 * Q.N2 - 1.086952
    if Q.n_pairs_kt_above_1 < 325.0:
        z += -0.001875637 * Q.n_pairs_kt_above_1 + 0.6095821
    if Q.N2_b05 >= 0.4599592:
        z += -1.009653 * Q.N2_b05 + 0.4643994
    z += -0.05258763 * Q.n_sd0_above_2
    if Q.sj3_mass3 < 0.3886647:
        z += -0.6639299 * Q.sj3_mass3 + 0.2580461
    if 0.2410564 <= Q.sj3_dr_min < 0.3526989:
        z += -0.4544241 * Q.sj3_dr_min + 0.1095418
    if Q.sj3_dr_min >= 0.3526989:
        z += -0.7105736 * Q.sj3_dr_min + 0.1998855
    if Q.mass_displaced5 < 1.262841:
        z += 0.03864625 * Q.mass_displaced5 - 0.04880407
    if Q.sj4_pair_mass_max < 75.78208:
        z += 0.0001395838 * Q.sj4_pair_mass_max + 0.3938242
    if 75.78208 <= Q.sj4_pair_mass_max < 115.5142:
        z += -0.01017822 * Q.sj4_pair_mass_max + 1.175729
    if Q.z_charged_had < 0.3945478:
        z += 0.2308723 * Q.z_charged_had - 0.09109016
    if Q.lund2_lndelta >= -1.299999:
        z += -0.2188966 * Q.lund2_lndelta - 0.2845654
    if Q.D2_b2 < 0.4044054:
        z += 0.262051 * Q.D2_b2 - 0.1059749
    if Q.max_abs_d0 < 10.52344:
        z += 0.01119101 * Q.max_abs_d0 - 0.1177679
    if Q.n_sdz_above_5 < 2.0:
        z += 0.04957834 * Q.n_sdz_above_5 - 0.09915667
    if Q.sum_z_dr2_top15 < 0.01040452:
        z += 11.03526 * Q.sum_z_dr2_top15 - 0.1148165
    if Q.mass_top5 >= 68.52153:
        z += 0.002376466 * Q.mass_top5 - 0.1628391
    if Q.z_top15_slots >= 0.8558319:
        z += -3.033632 * Q.z_top15_slots + 2.596279
    if Q.N3_b2 < 0.9644868:
        z += 0.4281361 * Q.N3_b2 - 0.4129317
    if Q.M2_b2 < 0.04544279:
        z += -2.159193 * Q.M2_b2 + 0.09811974
    if Q.C2_b2 < 0.07094338:
        z += 2.408937 * Q.C2_b2 - 0.1708981
    if Q.sj3_dr23 < 0.8799072:
        z += 0.0403473 * Q.sj3_dr23 - 0.03550188
    z += -0.007346788 * Q.isnhad_1 + 0.007346788
    if Q.tdz_2 >= -0.03532465:
        z += 0.08740193 * Q.tdz_2 + 0.003087443
    if Q.mass_charged >= 99.20396:
        z += 0.002872046 * Q.mass_charged - 0.2849183
    if Q.N3_b05 < 1.33105:
        z += -0.8566406 * Q.N3_b05 + 1.140232
    if Q.n_pairs_kt_above_3 < 126.0:
        z += -0.004013906 * Q.n_pairs_kt_above_3 + 0.5057521
    if Q.n_real_top30 >= 21.0:
        z += -0.04501532 * Q.n_real_top30 + 0.9453216
    if 62.03827 <= Q.sd_mass < 88.81751:
        z += 0.0232743 * Q.sd_mass - 1.443897
    if 88.81751 <= Q.sd_mass < 119.4443:
        z += -0.02358519 * Q.sd_mass + 2.718045
    if 119.4443 <= Q.sd_mass < 175.9333:
        z += 0.01386127 * Q.sd_mass - 1.754722
    if Q.sd_mass >= 175.9333:
        z += -0.004869096 * Q.sd_mass + 1.540574
    if Q.sj3_pair_mass_max >= 142.9952:
        z += 0.005670689 * Q.sj3_pair_mass_max - 0.8108816
    if Q.tau32_b2 < 0.4680886:
        z += -0.114908 * Q.tau32_b2 + 0.0537871
    if Q.sd_rg < 0.3995332:
        z += -1.334907 * Q.sd_rg + 0.5333396
    if Q.C2 >= 0.07037463:
        z += -1.094863 * Q.C2 + 0.07705056
    if Q.n_dr_0p4_up < 4.0:
        z += -0.07343882 * Q.n_dr_0p4_up + 0.2937553
    if Q.lep_z < 0.3396572 and Q.mass < 127.2317:
        z += 0.0499635 * (0.3396572 - Q.lep_z) * (127.2317 - Q.mass)
    if Q.mass < 117.4867 and Q.D2 < 3.038341:
        z += 0.005531611 * (117.4867 - Q.mass) * (3.038341 - Q.D2)
    if Q.lep_ptrel < 43.20788 and Q.M3 > 0.03259227:
        z += 0.09125975 * (43.20788 - Q.lep_ptrel) * (Q.M3 - 0.03259227)
    if Q.e3_b2 < 0.0004126585 and Q.mass_charged > 90.44234:
        z += 29.08957 * (0.0004126585 - Q.e3_b2) * (Q.mass_charged - 90.44234)
    if Q.lep_z < 0.221436 and Q.sj2_dr < 0.2403736:
        z += -42.46636 * (0.221436 - Q.lep_z) * (0.2403736 - Q.sj2_dr)
    if Q.sj2_dr < 0.5944498 and Q.n_dr_0p2_0p4 < 21.0:
        z += 0.1245345 * (0.5944498 - Q.sj2_dr) * (21.0 - Q.n_dr_0p2_0p4)
    if Q.z_displaced3 < 0.1061578 and Q.ecf_g42 < 5.969698e-05:
        z += -31314.45 * (0.1061578 - Q.z_displaced3) * (5.969698e-05 - Q.ecf_g42)
    if Q.mass_displaced3 < 26.78691 and Q.z_charged > 0.4622094:
        z += -0.0007096457 * (26.78691 - Q.mass_displaced3) * (Q.z_charged - 0.4622094)
    if Q.mass_top30 < 74.51927 and Q.D2 < 3.038341:
        z += -0.01687756 * (74.51927 - Q.mass_top30) * (3.038341 - Q.D2)
    if Q.z_displaced3 < 0.1061578 and Q.D2 > 2.446407:
        z += -0.8598836 * (0.1061578 - Q.z_displaced3) * (Q.D2 - 2.446407)
    if Q.z_displaced3 < 0.1061578 and Q.mass_2photon > 4.18513:
        z += 0.05332417 * (0.1061578 - Q.z_displaced3) * (Q.mass_2photon - 4.18513)
    if Q.mass < 117.4867 and Q.lund2_lndelta > -1.755318:
        z += -0.02120166 * (117.4867 - Q.mass) * (Q.lund2_lndelta - -1.755318)
    if Q.mass_top30 < 74.51927 and Q.n_muon > 0.0:
        z += 0.0207721 * (74.51927 - Q.mass_top30) * (Q.n_muon - 0.0)
    if Q.lep_ptrel < 43.20788 and Q.n_muon > 0.0:
        z += -0.003551564 * (43.20788 - Q.lep_ptrel) * (Q.n_muon - 0.0)
    if Q.e3_b2 < 0.0004126585 and Q.pt_balance01 < 0.3972884:
        z += 2110.273 * (0.0004126585 - Q.e3_b2) * (0.3972884 - Q.pt_balance01)
    if Q.n_sd0_above_3 > 3.0 and Q.d0err_42 > 0.0:
        z += -0.05434385 * (Q.n_sd0_above_3 - 3.0) * (Q.d0err_42 - 0.0)
    if Q.e3_b2 < 0.0004126585 and Q.min_pair_mass > 10.59762:
        z += 64.26622 * (0.0004126585 - Q.e3_b2) * (Q.min_pair_mass - 10.59762)
    if Q.lund2_lndelta > -1.299999 and Q.tdz_0 < -0.02698624:
        z += -0.00291803 * (Q.lund2_lndelta - -1.299999) * (-0.02698624 - Q.tdz_0)
    if Q.lep_ptrel < 43.20788 and Q.dr_10 > 0.3349968:
        z += -0.001334366 * (43.20788 - Q.lep_ptrel) * (Q.dr_10 - 0.3349968)
    if Q.sj4_pair_mass_max < 75.78208 and Q.iselectron_24 > 0.0:
        z += -0.002058561 * (75.78208 - Q.sj4_pair_mass_max) * (Q.iselectron_24 - 0.0)
    if Q.mass < 164.4374 and Q.tdz_6 > 0.03675711:
        z += 0.001953991 * (164.4374 - Q.mass) * (Q.tdz_6 - 0.03675711)
    if Q.lep_z < 0.3396572 and Q.e4 > 5.29882e-07:
        z += 17429.06 * (0.3396572 - Q.lep_z) * (Q.e4 - 5.29882e-07)
    if Q.mass_charged > 99.20396 and Q.lnerel_82 > -18.42068:
        z += -0.0001946759 * (Q.mass_charged - 99.20396) * (Q.lnerel_82 - -18.42068)
    if Q.D2_b2 < 0.4044054 and Q.dr_31 > 0.02273044:
        z += 1.793293 * (0.4044054 - Q.D2_b2) * (Q.dr_31 - 0.02273044)
    if Q.mass < 164.4374 and Q.lund2_lndelta > -1.411949:
        z += 0.0127018 * (164.4374 - Q.mass) * (Q.lund2_lndelta - -1.411949)
    if Q.n_sd0_above_3 < 6.0 and Q.n_lund_kt_above_1 > 4.0:
        z += 0.01352936 * (6.0 - Q.n_sd0_above_3) * (Q.n_lund_kt_above_1 - 4.0)
    if Q.mass_displaced3 < 26.78691 and Q.dr_31 < 0.06777369:
        z += 0.04733497 * (26.78691 - Q.mass_displaced3) * (0.06777369 - Q.dr_31)
    if Q.mass < 90.08945 and Q.lund2_lndelta > -2.073049:
        z += 0.003079313 * (90.08945 - Q.mass) * (Q.lund2_lndelta - -2.073049)
    if Q.sd_mass > 88.81751 and Q.charge_0 < 0.0:
        z += -0.0001087737 * (Q.sd_mass - 88.81751) * (0.0 - Q.charge_0)
    return z


def neuron_25(Q):
    z = 0.03127421
    return z


def neuron_26(Q):
    z = -4.927568e-05
    return z


def neuron_27(Q):
    z = -0.0001142129
    return z


def neuron_28(Q):
    z = -9.774775e-05
    return z


def neuron_29(Q):
    z = -5.716836e-05
    return z


def neuron_30(Q):
    z = -1.286882e-05
    return z


def neuron_31(Q):
    z = -0.0001454479
    return z


def neuron_32(Q):
    z = 8.885355e-05
    return z


def neuron_33(Q):
    z = -6.72559e-05
    return z


def neuron_34(Q):
    z = 5.909598e-05
    return z


def neuron_35(Q):
    z = -0.0001787317
    return z


def neuron_36(Q):
    z = 6.448542e-05
    return z


def neuron_37(Q):
    z = 1.87593e-05
    return z


def neuron_38(Q):
    z = -5.47689e-06
    return z


def neuron_39(Q):
    z = -7.429524e-05
    return z


def neuron_40(Q):
    z = 9.897107e-05
    return z


def neuron_41(Q):
    z = 3.86113e-05
    return z


def neuron_42(Q):
    z = 0.0001351143
    return z


def neuron_43(Q):
    z = -8.345711e-05
    return z


def neuron_44(Q):
    z = 0.0001459569
    return z


def neuron_45(Q):
    z = 0.0001019341
    return z


def neuron_46(Q):
    z = -5.265668e-05
    return z


def neuron_47(Q):
    z = 2.294383e-05
    return z


def neuron_48(Q):
    z = -0.000102278
    return z


def neuron_49(Q):
    z = -4.990678e-05
    return z


def neuron_50(Q):
    z = 0.0001310781
    return z


def neuron_51(Q):
    z = 4.271572e-05
    return z


def neuron_52(Q):
    z = -0.9404504
    if Q.lep_z < 0.221436:
        z += 2.493797 * Q.lep_z - 0.664156
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += 0.9468656 * Q.lep_z - 0.3216098
    if Q.z_displaced3 < 0.02600452:
        z += -1.555675 * Q.z_displaced3 + 0.06505955
    if 0.02600452 <= Q.z_displaced3 < 0.1342762:
        z += -0.2272521 * Q.z_displaced3 + 0.03051455
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.01174553 * Q.n_pairs_kt_above_1 + 0.8941422
    if Q.n_pairs_kt_above_1 >= 80.0:
        z += -0.000568751 * Q.n_pairs_kt_above_1
    z += 0.01161732 * Q.n_charged_had
    if Q.lep_ptrel < 12.15228:
        z += 0.01073192 * Q.lep_ptrel - 0.1411518
    if 12.15228 <= Q.lep_ptrel < 18.7678:
        z += 0.005614669 * Q.lep_ptrel - 0.07896555
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += -0.01216074 * Q.lep_ptrel + 0.2546398
    if Q.lep_ptrel >= 27.3236:
        z += -0.005117252 * Q.lep_ptrel + 0.06218629
    if Q.sj3_pair_mass_max < 56.73313:
        z += 0.005050076 * Q.sj3_pair_mass_max - 0.09708923
    if 56.73313 <= Q.sj3_pair_mass_max < 97.52633:
        z += -0.004643356 * Q.sj3_pair_mass_max + 0.4528495
    if Q.lep_iso < 0.4381892:
        z += -2.063659 * Q.lep_iso + 1.195891
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += -0.1038055 * Q.lep_iso + 0.3371047
    if 1.362094 <= Q.lep_iso < 44.14912:
        z += -0.004574094 * Q.lep_iso + 0.2019422
    if Q.mass < 71.96396:
        z += 0.005422961 * Q.mass - 0.6371256
    if 71.96396 <= Q.mass < 90.08945:
        z += 0.02110806 * Q.mass - 1.765888
    if 90.08945 <= Q.mass < 117.4867:
        z += 0.01720641 * Q.mass - 1.41439
    if Q.mass >= 117.4867:
        z += 0.01178345 * Q.mass - 0.777264
    if Q.jet_abs_eta >= 0.8317778:
        z += -0.231906 * Q.jet_abs_eta + 0.1928942
    if Q.mass_top50 < 79.27954:
        z += -0.01602348 * Q.mass_top50 + 1.811808
    if 79.27954 <= Q.mass_top50 < 129.5874:
        z += -0.006689802 * Q.mass_top50 + 1.071838
    if 129.5874 <= Q.mass_top50 < 161.1264:
        z += -0.00649748 * Q.mass_top50 + 1.046916
    if Q.n_lepton < 1.0:
        z += 1.186257 * Q.n_lepton - 1.186257
    if Q.LHA < 0.5626523:
        z += -2.109583 * Q.LHA + 1.186962
    if Q.z_neutral_had < 0.3788785:
        z += -0.3440697 * Q.z_neutral_had + 0.1303606
    if Q.mass_top15 >= 94.94813:
        z += -0.003361974 * Q.mass_top15 + 0.3192131
    if Q.C2 < 0.1238959:
        z += -0.8210788 * Q.C2 + 0.1017283
    if Q.sip_3d_1 < 2.802481:
        z += 0.1396568 * Q.sip_3d_1 + 0.02039042
    if 2.802481 <= Q.sip_3d_1 < 3.373037:
        z += -0.7217105 * Q.sip_3d_1 + 2.434356
    if Q.sj3_pair_mass_min >= 80.02563:
        z += 0.005658069 * Q.sj3_pair_mass_min - 0.4527905
    if Q.sj2_dr < 0.2848657:
        z += 0.2891403 * Q.sj2_dr - 0.08236616
    if Q.sj2_mass2 < 1.851735:
        z += -0.2431863 * Q.sj2_mass2 + 0.4503166
    if Q.z_top10_slots >= 0.7362734:
        z += 1.121703 * Q.z_top10_slots - 0.8258801
    if Q.jet_charge_k03 >= -0.1486714:
        z += -0.1273139 * Q.jet_charge_k03 - 0.01892794
    if Q.C2_b2 < 0.09209404:
        z += -0.7643431 * Q.C2_b2 + 0.07039144
    if Q.sj2_mass1 >= 91.2852:
        z += 0.005229567 * Q.sj2_mass1 - 0.477382
    if Q.sj3_mass1 >= 36.36236:
        z += -0.01167205 * Q.sj3_mass1 + 0.4244233
    if Q.e3_b2 < 1.008706e-05:
        z += 22421.83 * Q.e3_b2 - 0.2261704
    if Q.mass_displaced3 < 26.78691:
        z += -0.006237697 * Q.mass_displaced3 + 0.1670886
    if Q.sip_3d_2 < 45.26581:
        z += -0.005963716 * Q.sip_3d_2 + 0.2699524
    if Q.C3_b2 < 0.04229114:
        z += 6.733991 * Q.C3_b2 - 0.2847881
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.007399841 * Q.n_pairs_kt_above_3 + 0.2071956
    if Q.tdz_0 < 0.0:
        z += 2.87317 * Q.tdz_0 + 0.1947862
    if 0.0 <= Q.tdz_0 < 0.07687439:
        z += -2.533824 * Q.tdz_0 + 0.1947862
    if Q.mass_charged < 65.0404:
        z += 0.0006619531 * Q.mass_charged - 0.0430537
    if Q.n_neutral_had >= 7.0:
        z += 0.02110771 * Q.n_neutral_had - 0.147754
    if Q.N3 >= 0.2681212:
        z += -0.1138677 * Q.N3 + 0.03053034
    if Q.M2 < 0.04276413:
        z += -2.959467 * Q.M2 + 0.126559
    if Q.tau32 < 0.3951525:
        z += 0.9528028 * Q.tau32 - 0.3765024
    if Q.tau2 < 0.06603871:
        z += 9.521288 * Q.tau2 - 0.6287736
    if Q.pair_max_lnm2 < 5.511295:
        z += 0.1196785 * Q.pair_max_lnm2 - 0.6595835
    if Q.N2_b2 >= 0.2392833:
        z += 2.845118 * Q.N2_b2 - 0.6807891
    if Q.lund_max_lndelta >= -0.615738:
        z += 0.9720587 * Q.lund_max_lndelta + 0.5985334
    if Q.sd_mass < 42.31629:
        z += 0.01407574 * Q.sd_mass - 0.4321386
    if 42.31629 <= Q.sd_mass < 83.61981:
        z += -0.01145816 * Q.sd_mass + 0.648361
    if 83.61981 <= Q.sd_mass < 123.2919:
        z += 0.01235418 * Q.sd_mass - 1.342822
    if 123.2919 <= Q.sd_mass < 154.5947:
        z += -0.005761392 * Q.sd_mass + 0.8906804
    if Q.sj4_pair_mass_max < 101.849:
        z += -0.0007460335 * Q.sj4_pair_mass_max + 0.07598277
    if Q.n_dr_0_0p05 < 8.0:
        z += -0.01823214 * Q.n_dr_0_0p05 + 0.1458571
    if Q.n_sd0_above_3 < 4.0:
        z += 0.1358261 * Q.n_sd0_above_3 - 1.089871
    if 4.0 <= Q.n_sd0_above_3 < 8.0:
        z += 0.1366417 * Q.n_sd0_above_3 - 1.093134
    if Q.n_s3d_above_3 < 7.0:
        z += -0.2845728 * Q.n_s3d_above_3 + 1.992009
    if Q.mass_displaced5 < 0.8035262:
        z += -0.1454497 * Q.mass_displaced5 + 0.1168726
    if Q.n_particles >= 38.0:
        z += 0.006357177 * Q.n_particles - 0.2415727
    if Q.z_displaced3 < 0.1342762 and Q.N2 > 0.1824346:
        z += 5.229391 * (0.1342762 - Q.z_displaced3) * (Q.N2 - 0.1824346)
    if Q.lep_z < 0.3396572 and Q.sj3_pair_mass_min > 48.40547:
        z += -0.01875916 * (0.3396572 - Q.lep_z) * (Q.sj3_pair_mass_min - 48.40547)
    if Q.mass < 117.4867 and Q.tau21_b2 < 0.3565533:
        z += -0.06447674 * (117.4867 - Q.mass) * (0.3565533 - Q.tau21_b2)
    if Q.mass > 90.08945 and Q.n_s3d_above_3 < 5.0:
        z += 0.0001515377 * (Q.mass - 90.08945) * (5.0 - Q.n_s3d_above_3)
    if Q.lep_ptrel < 27.3236 and Q.tau3 < 0.08643515:
        z += -0.2677105 * (27.3236 - Q.lep_ptrel) * (0.08643515 - Q.tau3)
    if Q.z_neutral_had < 0.3788785 and Q.ecf_g41 > 6.216954e-05:
        z += 3723.562 * (0.3788785 - Q.z_neutral_had) * (Q.ecf_g41 - 6.216954e-05)
    if Q.lep_iso < 1.362094 and Q.sip_3d_3 < 49.03695:
        z += -0.001937227 * (1.362094 - Q.lep_iso) * (49.03695 - Q.sip_3d_3)
    if Q.LHA < 0.5626523 and Q.mass_displaced3 < 26.78691:
        z += 0.01390563 * (0.5626523 - Q.LHA) * (26.78691 - Q.mass_displaced3)
    if Q.mass_top50 < 129.5874 and Q.lep_dr < 0.114659:
        z += 0.04064662 * (129.5874 - Q.mass_top50) * (0.114659 - Q.lep_dr)
    if Q.mass > 90.08945 and Q.tau32_b2 < 0.7269473:
        z += -0.01410195 * (Q.mass - 90.08945) * (0.7269473 - Q.tau32_b2)
    if Q.z_displaced3 < 0.1342762 and Q.lep_dr > 0.114659:
        z += -8.078782 * (0.1342762 - Q.z_displaced3) * (Q.lep_dr - 0.114659)
    if Q.z_displaced3 < 0.1342762 and Q.max_abs_d0 < 3.113281:
        z += -0.6930291 * (0.1342762 - Q.z_displaced3) * (3.113281 - Q.max_abs_d0)
    if Q.mass > 90.08945 and Q.tau43_b2 < 0.6818924:
        z += -0.008337117 * (Q.mass - 90.08945) * (0.6818924 - Q.tau43_b2)
    if Q.C2_b2 < 0.09209404 and Q.D3_b2 < 1.462588:
        z += 2.511649 * (0.09209404 - Q.C2_b2) * (1.462588 - Q.D3_b2)
    if Q.C2_b2 < 0.09209404 and Q.n_muon > 0.0:
        z += -0.9215971 * (0.09209404 - Q.C2_b2) * (Q.n_muon - 0.0)
    if Q.lep_iso < 0.4381892 and Q.mass_displaced5 < 0.3201841:
        z += -0.9573892 * (0.4381892 - Q.lep_iso) * (0.3201841 - Q.mass_displaced5)
    if Q.z_displaced3 < 0.1342762 and Q.sip_3d_3 < 4.606241:
        z += 1.576492 * (0.1342762 - Q.z_displaced3) * (4.606241 - Q.sip_3d_3)
    if Q.lep_ptrel < 18.7678 and Q.mass_displaced3 < 18.80005:
        z += 0.001092963 * (18.7678 - Q.lep_ptrel) * (18.80005 - Q.mass_displaced3)
    if Q.z_displaced3 < 0.1342762 and Q.dr_21 < 0.04063514:
        z += -23.44034 * (0.1342762 - Q.z_displaced3) * (0.04063514 - Q.dr_21)
    if Q.mass_displaced3 < 26.78691 and Q.n_lund_kt_above_1 > 4.0:
        z += 0.000936538 * (26.78691 - Q.mass_displaced3) * (Q.n_lund_kt_above_1 - 4.0)
    if Q.mass_top50 < 129.5874 and Q.dr_21 < 0.0632362:
        z += 0.03299836 * (129.5874 - Q.mass_top50) * (0.0632362 - Q.dr_21)
    if Q.mass_top15 > 94.94813 and Q.sj3_dr13 > 0.2640447:
        z += 0.000212837 * (Q.mass_top15 - 94.94813) * (Q.sj3_dr13 - 0.2640447)
    if Q.lep_ptrel < 18.7678 and Q.dr_21 < 0.1905794:
        z += -0.02786375 * (18.7678 - Q.lep_ptrel) * (0.1905794 - Q.dr_21)
    if Q.mass > 90.08945 and Q.td0_13 > -0.06278656:
        z += 0.0006809887 * (Q.mass - 90.08945) * (Q.td0_13 - -0.06278656)
    if Q.lep_z < 0.3396572 and Q.tdz_0 < -0.01644749:
        z += 7.405491 * (0.3396572 - Q.lep_z) * (-0.01644749 - Q.tdz_0)
    if Q.C3_b2 < 0.04229114 and Q.pt2_over_pt0 < 0.2816529:
        z += 23.43076 * (0.04229114 - Q.C3_b2) * (0.2816529 - Q.pt2_over_pt0)
    if Q.n_pairs_kt_above_3 < 28.0 and Q.mass_2photon > 3.013:
        z += 0.001175552 * (28.0 - Q.n_pairs_kt_above_3) * (Q.mass_2photon - 3.013)
    if Q.mass > 90.08945 and Q.d0err_75 > 0.0:
        z += -0.004345015 * (Q.mass - 90.08945) * (Q.d0err_75 - 0.0)
    if Q.z_neutral_had < 0.3788785 and Q.d0err_3 < 0.01100159:
        z += -14.65554 * (0.3788785 - Q.z_neutral_had) * (0.01100159 - Q.d0err_3)
    if Q.n_s3d_above_3 < 7.0 and Q.sip_3d_3 < 175.9957:
        z += -0.0007898631 * (7.0 - Q.n_s3d_above_3) * (175.9957 - Q.sip_3d_3)
    if Q.n_s3d_above_3 < 7.0 and Q.sip_3d_2 < 70.05844:
        z += -0.00186686 * (7.0 - Q.n_s3d_above_3) * (70.05844 - Q.sip_3d_2)
    if Q.lep_z < 0.3396572 and Q.sip_3d_2 < 4.636903:
        z += 0.1810955 * (0.3396572 - Q.lep_z) * (4.636903 - Q.sip_3d_2)
    if Q.sj3_pair_mass_min > 80.02563 and Q.sip_3d_2 < 20.75111:
        z += 0.0001227309 * (Q.sj3_pair_mass_min - 80.02563) * (20.75111 - Q.sip_3d_2)
    if Q.C2_b2 < 0.09209404 and Q.sip_3d_2 > 3.041925:
        z += -0.0005097124 * (0.09209404 - Q.C2_b2) * (Q.sip_3d_2 - 3.041925)
    if Q.tau32 < 0.3951525 and Q.d0err_14 > 0.01499939:
        z += -2.636605 * (0.3951525 - Q.tau32) * (Q.d0err_14 - 0.01499939)
    if Q.n_lepton < 1.0 and Q.sip_3d_3 < 11.13866:
        z += 0.005724977 * (1.0 - Q.n_lepton) * (11.13866 - Q.sip_3d_3)
    return z


def neuron_53(Q):
    z = -9.949874e-05
    return z


def neuron_54(Q):
    z = -9.043362e-05
    return z


def neuron_55(Q):
    z = 6.686006e-05
    return z


def neuron_56(Q):
    z = -9.207841e-05
    return z


def neuron_57(Q):
    z = -8.377403e-05
    return z


def neuron_58(Q):
    z = -8.914886e-05
    return z


def neuron_59(Q):
    z = 4.836034e-05
    return z


def neuron_60(Q):
    z = 2.535067e-05
    return z


def neuron_61(Q):
    z = -0.0001436141
    return z


def neuron_62(Q):
    z = 2.824284e-05
    return z


def neuron_63(Q):
    z = -2.344023e-05
    return z


def neuron_64(Q):
    z = 0.0001321867
    return z


def neuron_65(Q):
    z = -6.677285e-05
    return z


def neuron_66(Q):
    z = -0.0001476023
    return z


def neuron_67(Q):
    z = -0.0001149744
    return z


def neuron_68(Q):
    z = 3.739478
    if Q.tau32 < 0.7981752:
        z += -2.064176 * Q.tau32 + 1.647574
    if 6.983043 <= Q.lep_ptrel < 43.20788:
        z += -0.01670069 * Q.lep_ptrel + 0.1166216
    if Q.lep_ptrel >= 43.20788:
        z += -0.008492163 * Q.lep_ptrel - 0.2380514
    if Q.pair_mean_lnm2 < 3.38061:
        z += 0.196527 * Q.pair_mean_lnm2 - 0.6643811
    if Q.n_s3d_above_3 >= 5.0:
        z += -0.03318948 * Q.n_s3d_above_3 + 0.1659474
    if Q.sj3_pair_mass_max < 44.2029:
        z += 0.002565273 * Q.sj3_pair_mass_max + 1.019473
    if 44.2029 <= Q.sj3_pair_mass_max < 56.73313:
        z += 0.01349192 * Q.sj3_pair_mass_max + 0.5364833
    if 56.73313 <= Q.sj3_pair_mass_max < 80.36029:
        z += -0.002307301 * Q.sj3_pair_mass_max + 1.432822
    if 80.36029 <= Q.sj3_pair_mass_max < 120.6471:
        z += -0.01022984 * Q.sj3_pair_mass_max + 2.06948
    if Q.sj3_pair_mass_max >= 120.6471:
        z += 0.01092664 * Q.sj3_pair_mass_max - 0.4829894
    if Q.M2_b2 < 0.04544279:
        z += 15.91721 * Q.M2_b2 - 0.7233226
    if Q.mass < 79.47361:
        z += 0.01685364 * Q.mass - 3.752553
    if 79.47361 <= Q.mass < 90.08945:
        z += -0.00516505 * Q.mass - 2.002648
    if 90.08945 <= Q.mass < 114.0172:
        z += -0.01286186 * Q.mass - 1.309247
    if 114.0172 <= Q.mass < 127.2317:
        z += 0.02673232 * Q.mass - 5.823666
    if 127.2317 <= Q.mass < 164.4374:
        z += 0.01482768 * Q.mass - 4.309018
    if Q.mass >= 164.4374:
        z += -0.02201869 * Q.mass + 1.749905
    if Q.n_pt_above_1 >= 17.0:
        z += -0.032926 * Q.n_pt_above_1 + 0.5597419
    if Q.sj3_pair_mass_min < 80.02563:
        z += 0.02627098 * Q.sj3_pair_mass_min - 2.102352
    if Q.mass_top30 >= 162.7874:
        z += 0.008407423 * Q.mass_top30 - 1.368623
    if Q.sj3_dr_min >= 0.3526989:
        z += 0.3344636 * Q.sj3_dr_min - 0.1179649
    if Q.z_displaced3 < 0.3261071:
        z += -0.7897661 * Q.z_displaced3 + 0.2575483
    if Q.n_sdz_above_5 < 3.0:
        z += -0.08771542 * Q.n_sdz_above_5 + 0.2631463
    if Q.psi_0p1 >= 0.835544:
        z += -2.357068 * Q.psi_0p1 + 1.969434
    if Q.jet_abs_eta < 1.323111:
        z += 0.2260523 * Q.jet_abs_eta - 0.2990923
    z += -1.503458 * Q.sj3_pairmin_over_m
    if Q.n_pairs_kt_above_3 < 28.0:
        z += 0.005877771 * Q.n_pairs_kt_above_3 - 0.1645776
    if 42.31629 <= Q.sd_mass < 83.61981:
        z += 0.01887946 * Q.sd_mass - 0.7989088
    if 83.61981 <= Q.sd_mass < 88.81751:
        z += -0.0379496 * Q.sd_mass + 3.953127
    if 88.81751 <= Q.sd_mass < 127.8042:
        z += -0.0216349 * Q.sd_mass + 2.504096
    if 127.8042 <= Q.sd_mass < 154.5947:
        z += 0.02896186 * Q.sd_mass - 3.962385
    if Q.sd_mass >= 154.5947:
        z += 0.005848819 * Q.sd_mass - 0.3892323
    if Q.n_lepton < 1.0:
        z += -0.8231196 * Q.n_lepton + 1.127824
    if 1.0 <= Q.n_lepton < 2.0:
        z += -0.3047049 * Q.n_lepton + 0.6094099
    if Q.lep_iso < 0.4381892:
        z += 1.650641 * Q.lep_iso - 0.7232932
    if Q.tau21_b2 >= 0.5211946:
        z += -0.850493 * Q.tau21_b2 + 0.4432724
    if Q.mass_top15 >= 119.5993:
        z += -0.008898212 * Q.mass_top15 + 1.06422
    if Q.n_charged_had >= 13.0:
        z += -0.05295726 * Q.n_charged_had + 0.6884444
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.002253914 * Q.n_pairs_kt_above_1 + 0.1803132
    if Q.n_pairs_kt_above_1 >= 101.0:
        z += 0.0009803951 * Q.n_pairs_kt_above_1 - 0.09901991
    if Q.tau3 < 0.03378303:
        z += 4.315483 * Q.tau3 - 0.1457901
    if Q.sip_3d_2 < 226.3008:
        z += 8.127213e-05 * Q.sip_3d_2 - 0.01839195
    if Q.mass_top10 >= 27.42853:
        z += 0.00225737 * Q.mass_top10 - 0.06191634
    if Q.sum_z_dr2_top3 < 0.002792418:
        z += -115.0421 * Q.sum_z_dr2_top3 + 0.3212456
    if Q.M3 < 0.03259227:
        z += 1.102242 * Q.M3 - 0.03592458
    if Q.sj3_mass2 < 3.298199:
        z += 0.04686673 * Q.sj3_mass2 - 0.1545758
    if Q.n_electron >= 1.0:
        z += 0.1413446 * Q.n_electron - 0.1413446
    if Q.sum_z_dr2_top15 < 0.01040452:
        z += 15.47213 * Q.sum_z_dr2_top15 - 0.16098
    if Q.z_displaced5 >= 0.2801368:
        z += -1.18334 * Q.z_displaced5 + 0.331497
    if Q.n_dr_0p2_0p4 >= 9.0:
        z += -0.008887017 * Q.n_dr_0p2_0p4 + 0.07998315
    if Q.D2_b2 < 2.286291:
        z += 0.1147459 * Q.D2_b2 - 0.2623425
    if Q.mass_2photon >= 22.18431:
        z += -0.01735952 * Q.mass_2photon + 0.385109
    if Q.mass_over_sum_pt_sq >= 0.0729277:
        z += -2.796599 * Q.mass_over_sum_pt_sq + 0.2039495
    if Q.e4_b05 >= 2.879843e-05:
        z += 1151.428 * Q.e4_b05 - 0.03315932
    if Q.e3_b2 < 7.104488e-06:
        z += -21518.5 * Q.e3_b2 + 0.1528779
    if Q.sip_3d_3 < 2.317124:
        z += -0.1941952 * Q.sip_3d_3 + 0.4499743
    if Q.mass_2charged < 0.9506259:
        z += -0.1896136 * Q.mass_2charged + 0.1802516
    if Q.mass_top5 < 47.41085:
        z += 0.001926354 * Q.mass_top5 - 0.09133009
    if Q.z_dr_0p1_0p2 < 0.1036778:
        z += -0.9838343 * Q.z_dr_0p1_0p2 + 0.1020017
    if Q.z_charged_had < 0.7223231:
        z += 0.6145211 * Q.z_charged_had - 0.4438828
    if Q.lep_z >= 0.5187302:
        z += -1.016587 * Q.lep_z + 0.5273343
    if Q.C2_b05 >= 0.3533901:
        z += -1.792034 * Q.C2_b05 + 0.6332869
    if Q.n_pt_above_5 >= 25.0:
        z += 0.005990111 * Q.n_pt_above_5 - 0.1497528
    if Q.mass_displaced5 < 2.382955:
        z += -0.05053492 * Q.mass_displaced5 + 0.1204224
    if Q.tau32 < 0.7981752 and Q.sj3_dr_min < 0.3190414:
        z += -1.699529 * (0.7981752 - Q.tau32) * (0.3190414 - Q.sj3_dr_min)
    if Q.pair_mean_lnm2 < 3.38061 and Q.z_top30_slots > 0.8316085:
        z += 0.2868389 * (3.38061 - Q.pair_mean_lnm2) * (Q.z_top30_slots - 0.8316085)
    if Q.mass < 114.0172 and Q.n_lund_kt_above_1 > 3.0:
        z += -0.001378107 * (114.0172 - Q.mass) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.M2_b2 < 0.04544279 and Q.mass_displaced3 > 0.0:
        z += 0.1816569 * (0.04544279 - Q.M2_b2) * (Q.mass_displaced3 - 0.0)
    if Q.tau32 < 0.7981752 and Q.z_charged_had < 0.7223231:
        z += -1.419014 * (0.7981752 - Q.tau32) * (0.7223231 - Q.z_charged_had)
    if Q.n_pt_above_1 > 17.0 and Q.tau43 < 0.8939856:
        z += 0.05144835 * (Q.n_pt_above_1 - 17.0) * (0.8939856 - Q.tau43)
    if Q.mass < 114.0172 and Q.D2 < 2.255345:
        z += -0.0005425712 * (114.0172 - Q.mass) * (2.255345 - Q.D2)
    if Q.pair_mean_lnm2 < 3.38061 and Q.z_neutral_had < 0.3788785:
        z += 0.5436765 * (3.38061 - Q.pair_mean_lnm2) * (0.3788785 - Q.z_neutral_had)
    if Q.n_pt_above_1 > 17.0 and Q.z_displaced3 < 0.1061578:
        z += -0.1466522 * (Q.n_pt_above_1 - 17.0) * (0.1061578 - Q.z_displaced3)
    if Q.n_pt_above_1 > 17.0 and Q.tau54 < 0.8786609:
        z += 0.05504871 * (Q.n_pt_above_1 - 17.0) * (0.8786609 - Q.tau54)
    if Q.n_s3d_above_3 > 5.0 and Q.sip_3d_2 < 226.3008:
        z += -3.863136e-05 * (Q.n_s3d_above_3 - 5.0) * (226.3008 - Q.sip_3d_2)
    if Q.mass_top30 > 162.7874 and Q.sum_e < 1049.334:
        z += -1.795114e-05 * (Q.mass_top30 - 162.7874) * (1049.334 - Q.sum_e)
    if Q.sj3_pair_mass_min < 80.02563 and Q.sj3_mass1 > 18.73268:
        z += 0.0001572344 * (80.02563 - Q.sj3_pair_mass_min) * (Q.sj3_mass1 - 18.73268)
    if Q.n_sdz_above_5 < 3.0 and Q.sj2_dr > 0.3220633:
        z += -0.2072974 * (3.0 - Q.n_sdz_above_5) * (Q.sj2_dr - 0.3220633)
    if Q.tau32 < 0.7981752 and Q.sj3_dr_min > 0.3526989:
        z += -10.91544 * (0.7981752 - Q.tau32) * (Q.sj3_dr_min - 0.3526989)
    if Q.mass < 114.0172 and Q.lep_dr < 0.114659:
        z += 0.1181917 * (114.0172 - Q.mass) * (0.114659 - Q.lep_dr)
    if Q.z_displaced3 < 0.3261071 and Q.lep_dr < 0.4191372:
        z += -3.281018 * (0.3261071 - Q.z_displaced3) * (0.4191372 - Q.lep_dr)
    if Q.z_displaced3 < 0.3261071 and Q.n_lund > 9.0:
        z += -0.1294739 * (0.3261071 - Q.z_displaced3) * (Q.n_lund - 9.0)
    if Q.mass < 90.08945 and Q.n_lepton < 1.0:
        z += -0.03440325 * (90.08945 - Q.mass) * (1.0 - Q.n_lepton)
    if Q.lep_ptrel > 6.983043 and Q.C2_b2 < 0.1404188:
        z += 0.004395116 * (Q.lep_ptrel - 6.983043) * (0.1404188 - Q.C2_b2)
    if Q.sd_mass > 154.5947 and Q.sd_zg < 0.2450652:
        z += -0.02098442 * (Q.sd_mass - 154.5947) * (0.2450652 - Q.sd_zg)
    if Q.sd_mass > 42.31629 and Q.sj3_dr13 > 0.6229991:
        z += -0.003241214 * (Q.sd_mass - 42.31629) * (Q.sj3_dr13 - 0.6229991)
    if Q.n_lepton < 1.0 and Q.sip_3d_2 < 4.636903:
        z += -0.1247134 * (1.0 - Q.n_lepton) * (4.636903 - Q.sip_3d_2)
    if Q.M2_b2 < 0.04544279 and Q.mass_neutral < 76.15079:
        z += 0.117588 * (0.04544279 - Q.M2_b2) * (76.15079 - Q.mass_neutral)
    if Q.sd_mass > 88.81751 and Q.M3 > 0.02540381:
        z += -0.1828162 * (Q.sd_mass - 88.81751) * (Q.M3 - 0.02540381)
    if Q.z_displaced3 < 0.3261071 and Q.n_lund_kt_above_5 > 1.0:
        z += 0.2845699 * (0.3261071 - Q.z_displaced3) * (Q.n_lund_kt_above_5 - 1.0)
    if Q.jet_abs_eta < 1.323111 and Q.n_neutral_had > 6.0:
        z += -0.04947434 * (1.323111 - Q.jet_abs_eta) * (Q.n_neutral_had - 6.0)
    if Q.n_s3d_above_3 > 5.0 and Q.eta_48 > 0.2019104:
        z += -0.01397339 * (Q.n_s3d_above_3 - 5.0) * (Q.eta_48 - 0.2019104)
    if Q.sd_mass > 42.31629 and Q.z_dr_0p2_0p4 < 0.4683306:
        z += -0.0007893534 * (Q.sd_mass - 42.31629) * (0.4683306 - Q.z_dr_0p2_0p4)
    if Q.z_displaced3 < 0.3261071 and Q.mass_2photon > 9.82024:
        z += 0.02819631 * (0.3261071 - Q.z_displaced3) * (Q.mass_2photon - 9.82024)
    if Q.mass_top10 > 27.42853 and Q.dr_77 > 0.0:
        z += 0.003096144 * (Q.mass_top10 - 27.42853) * (Q.dr_77 - 0.0)
    if Q.mass_top30 > 162.7874 and Q.e4_b2 > 1.13e-10:
        z += 38.96002 * (Q.mass_top30 - 162.7874) * (Q.e4_b2 - 1.13e-10)
    if Q.mass_2photon > 22.18431 and Q.iselectron_1 > 0.0:
        z += -0.02479531 * (Q.mass_2photon - 22.18431) * (Q.iselectron_1 - 0.0)
    if Q.mass < 127.2317 and Q.sd_zg > 0.1900649:
        z += 0.01274474 * (127.2317 - Q.mass) * (Q.sd_zg - 0.1900649)
    if Q.sj3_dr_min > 0.3526989 and Q.charge_27 > 0.0:
        z += -0.2755346 * (Q.sj3_dr_min - 0.3526989) * (Q.charge_27 - 0.0)
    if Q.M3 < 0.03259227 and Q.ismuon_4 > 0.0:
        z += -2.306349 * (0.03259227 - Q.M3) * (Q.ismuon_4 - 0.0)
    return z


def neuron_69(Q):
    z = -0.0001536302
    return z


def neuron_70(Q):
    z = 0.01533753
    return z


def neuron_71(Q):
    z = 7.887963e-05
    return z


def neuron_72(Q):
    z = 9.959494e-05
    return z


def neuron_73(Q):
    z = -6.37936e-05
    return z


def neuron_74(Q):
    z = -0.0001058701
    return z


def neuron_75(Q):
    z = 6.376617e-05
    return z


def neuron_76(Q):
    z = -0.0001276191
    return z


def neuron_77(Q):
    z = 8.881019e-05
    return z


def neuron_78(Q):
    z = -1.068199
    if Q.lep_z < 0.221436:
        z += 1.870968 * Q.lep_z - 0.5923935
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += 1.506445 * Q.lep_z - 0.5116749
    if Q.n_s3d_above_3 < 4.0:
        z += -0.1245126 * Q.n_s3d_above_3 + 0.8924845
    if 4.0 <= Q.n_s3d_above_3 < 6.0:
        z += -0.1570036 * Q.n_s3d_above_3 + 1.022449
    if 6.0 <= Q.n_s3d_above_3 < 10.0:
        z += -0.06884325 * Q.n_s3d_above_3 + 0.4934864
    if Q.n_s3d_above_3 >= 10.0:
        z += -0.03249102 * Q.n_s3d_above_3 + 0.1299641
    if Q.mass_top30 < 67.20576:
        z += 0.007781424 * Q.mass_top30 - 0.5229565
    if Q.mass_top40 >= 173.1022:
        z += -0.001082577 * Q.mass_top40 + 0.1873964
    if Q.lep_ptrel < 12.15228:
        z += 0.06160243 * Q.lep_ptrel - 0.7706042
    if 12.15228 <= Q.lep_ptrel < 18.7678:
        z += 0.01780368 * Q.lep_ptrel - 0.2383495
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += 0.006605742 * Q.lep_ptrel - 0.0281889
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += -0.009588337 * Q.lep_ptrel + 0.4142917
    if Q.mass_displaced5 < 6.387683:
        z += -0.0006479103 * Q.mass_displaced5 + 0.02113275
    if 6.387683 <= Q.mass_displaced5 < 32.61679:
        z += -0.002009058 * Q.mass_displaced5 + 0.02982733
    if Q.mass_displaced5 >= 32.61679:
        z += -0.001361147 * Q.mass_displaced5 + 0.008694577
    if 78.76797 <= Q.sj4_pair_mass_max < 115.5142:
        z += 0.007823552 * Q.sj4_pair_mass_max - 0.6162453
    if Q.sj4_pair_mass_max >= 115.5142:
        z += -0.0009012548 * Q.sj4_pair_mass_max + 0.3915938
    if Q.D3_b2 < 0.3264446:
        z += -0.2370939 * Q.D3_b2 + 0.07739801
    if Q.n_neutral < 23.0:
        z += -0.003113549 * Q.n_neutral + 0.07161163
    if Q.tau32 < 0.6206221:
        z += -0.473801 * Q.tau32 + 0.2940514
    if Q.z_displaced5 >= 0.1709091:
        z += -0.1926181 * Q.z_displaced5 + 0.03292018
    z += 0.02840478 * Q.lne_0
    if Q.max_abs_d0 < 10.52344:
        z += 0.01175225 * Q.max_abs_d0 - 0.1236741
    if Q.pair_max_lnm2 >= 7.347625:
        z += -0.3310742 * Q.pair_max_lnm2 + 2.432609
    if Q.lund2_lndelta >= -1.556018:
        z += 0.06227464 * Q.lund2_lndelta + 0.09690048
    if Q.n_sd0_above_3 >= 2.0:
        z += 0.01531119 * Q.n_sd0_above_3 - 0.03062238
    if Q.mass_top15 >= 80.3877:
        z += 0.003334087 * Q.mass_top15 - 0.2680196
    if Q.lep_iso < 0.4381892:
        z += -1.066097 * Q.lep_iso + 0.3181732
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += 0.1612495 * Q.lep_iso - 0.2196371
    if Q.n_lepton < 1.0:
        z += 0.2005383 * Q.n_lepton - 0.1747367
    if 1.0 <= Q.n_lepton < 2.0:
        z += -0.02580154 * Q.n_lepton + 0.05160309
    if 122.0585 <= Q.mass_top50 < 161.1264:
        z += 0.007176785 * Q.mass_top50 - 0.8759879
    if Q.mass_top50 >= 161.1264:
        z += 0.004943254 * Q.mass_top50 - 0.5161072
    if Q.mass_top10 >= 103.4976:
        z += -0.001163324 * Q.mass_top10 + 0.1204013
    if Q.n_pairs_kt_above_3 < 99.0:
        z += -0.001304904 * Q.n_pairs_kt_above_3 + 0.1291855
    if Q.n_pairs_kt_above_3 >= 220.0:
        z += 0.001469443 * Q.n_pairs_kt_above_3 - 0.3232774
    if Q.n_electron >= 1.0:
        z += -0.05180363 * Q.n_electron + 0.05180363
    if Q.mass_charged >= 37.42054:
        z += 0.001750913 * Q.mass_charged - 0.06552009
    if Q.mass < 71.96396:
        z += -0.02795702 * Q.mass + 2.011898
    if 127.2317 <= Q.mass < 164.4374:
        z += 0.007201157 * Q.mass - 0.9162155
    if Q.mass >= 164.4374:
        z += -0.01545352 * Q.mass + 2.809062
    if Q.psi_0p1 >= 0.9356675:
        z += 3.12316 * Q.psi_0p1 - 2.922239
    if Q.n_pairs_kt_above_1 < 411.0:
        z += -0.001285499 * Q.n_pairs_kt_above_1 + 0.52834
    z += 0.02974987 * Q.n_charged_had
    if Q.sum_e >= 550.9555:
        z += -0.0002159008 * Q.sum_e + 0.1189517
    if Q.D2_b05 < 1.102602:
        z += -1.706664 * Q.D2_b05 + 1.881771
    if Q.e3 < 0.0007393085:
        z += 313.3906 * Q.e3 - 0.2316923
    if Q.tau3 < 0.1024935:
        z += -6.626597 * Q.tau3 + 0.6791829
    if Q.mass_top5 >= 54.36876:
        z += 0.006986528 * Q.mass_top5 - 0.3798489
    if Q.sj2_mass1 < 21.21062:
        z += -0.005982138 * Q.sj2_mass1 + 0.1268848
    if Q.mass_displaced3 < 13.03663:
        z += -0.003833457 * Q.mass_displaced3 + 0.04997537
    if Q.mass_displaced3 >= 39.09615:
        z += -0.004464393 * Q.mass_displaced3 + 0.1745406
    if Q.z_displaced3 >= 0.02600452:
        z += -0.525836 * Q.z_displaced3 + 0.01367411
    if Q.sip_3d_3 < 4.606241:
        z += -0.03945035 * Q.sip_3d_3 + 0.1817178
    if Q.sj4_dr_min < 0.07985021:
        z += 1.635439 * Q.sj4_dr_min - 0.1305901
    if Q.z_charged < 0.6378426:
        z += -0.09310587 * Q.z_charged + 0.05938688
    if Q.D2 < 2.255345:
        z += -0.08948755 * Q.D2 + 0.2018253
    if Q.n_dr_0p1_0p2 < 11.0:
        z += -0.01849463 * Q.n_dr_0p1_0p2 + 0.2034409
    if Q.sj3_dr23 < 0.2458451:
        z += 0.171891 * Q.sj3_dr23 - 0.04225855
    if Q.n_dr_0_0p05 < 2.0:
        z += -0.03144884 * Q.n_dr_0_0p05 + 0.06289769
    if Q.z_neutral_had >= 0.3788785:
        z += -0.813867 * Q.z_neutral_had + 0.3083567
    if Q.M2 >= 0.04276413:
        z += 4.070011 * Q.M2 - 0.1740505
    if Q.z_photon >= 0.3318968:
        z += 0.653415 * Q.z_photon - 0.2168664
    if Q.n_sd0_above_5 >= 4.0:
        z += 0.02716178 * Q.n_sd0_above_5 - 0.1086471
    if Q.lead_ch_sdz < 1.104348:
        z += 0.0003767179 * Q.lead_ch_sdz - 0.0004160275
    if Q.psi_0p2 < 0.8009208:
        z += 0.06744045 * Q.psi_0p2 - 0.05401446
    if Q.lep_z < 0.3396572 and Q.mass_top20 > 86.78877:
        z += 0.01468468 * (0.3396572 - Q.lep_z) * (Q.mass_top20 - 86.78877)
    if Q.lep_z < 0.3396572 and Q.M2_b05 < 0.177312:
        z += 25.31502 * (0.3396572 - Q.lep_z) * (0.177312 - Q.M2_b05)
    if Q.n_s3d_above_3 > 4.0 and Q.sj3_dr_min < 0.3190414:
        z += 0.03630912 * (Q.n_s3d_above_3 - 4.0) * (0.3190414 - Q.sj3_dr_min)
    if Q.lep_z < 0.3396572 and Q.pair_mean_lnz < -1.332584:
        z += -0.6083541 * (0.3396572 - Q.lep_z) * (-1.332584 - Q.pair_mean_lnz)
    if Q.n_s3d_above_3 > 4.0 and Q.tau21 < 0.5440886:
        z += -0.1873207 * (Q.n_s3d_above_3 - 4.0) * (0.5440886 - Q.tau21)
    if Q.mass_displaced5 > 6.387683 and Q.n_lepton < 1.0:
        z += 0.001069194 * (Q.mass_displaced5 - 6.387683) * (1.0 - Q.n_lepton)
    if Q.n_s3d_above_3 > 4.0 and Q.sip_3d_3 < 175.9957:
        z += 0.0004302882 * (Q.n_s3d_above_3 - 4.0) * (175.9957 - Q.sip_3d_3)
    if Q.lep_z < 0.3396572 and Q.pair_mean_lnkt > -0.0709631:
        z += -0.5976233 * (0.3396572 - Q.lep_z) * (Q.pair_mean_lnkt - -0.0709631)
    if Q.lep_z < 0.3396572 and Q.sj4_dr_min < 0.1631992:
        z += 3.330745 * (0.3396572 - Q.lep_z) * (0.1631992 - Q.sj4_dr_min)
    if Q.lep_ptrel < 27.3236 and Q.psi_0p2 < 0.8436463:
        z += 0.007617707 * (27.3236 - Q.lep_ptrel) * (0.8436463 - Q.psi_0p2)
    if Q.mass_displaced5 > 6.387683 and Q.sip_3d_3 < 11.13866:
        z += 0.003173382 * (Q.mass_displaced5 - 6.387683) * (11.13866 - Q.sip_3d_3)
    if Q.lep_z < 0.3396572 and Q.ecf_g42 < 8.17233e-05:
        z += 3158.442 * (0.3396572 - Q.lep_z) * (8.17233e-05 - Q.ecf_g42)
    if Q.mass_displaced5 > 6.387683 and Q.psi_0p3 < 0.8156512:
        z += -0.0002987072 * (Q.mass_displaced5 - 6.387683) * (0.8156512 - Q.psi_0p3)
    if Q.max_abs_d0 < 10.52344 and Q.sip_3d_3 < 23.30856:
        z += -0.0003977 * (10.52344 - Q.max_abs_d0) * (23.30856 - Q.sip_3d_3)
    if Q.mass_displaced5 > 6.387683 and Q.lep_iso < 14.38858:
        z += 0.0001695848 * (Q.mass_displaced5 - 6.387683) * (14.38858 - Q.lep_iso)
    if Q.n_s3d_above_3 < 10.0 and Q.sj3_dr_min > 0.08335692:
        z += -0.05438256 * (10.0 - Q.n_s3d_above_3) * (Q.sj3_dr_min - 0.08335692)
    if Q.lep_ptrel < 18.7678 and Q.z_dr_0p1_0p2 < 0.5699805:
        z += 0.0004669224 * (18.7678 - Q.lep_ptrel) * (0.5699805 - Q.z_dr_0p1_0p2)
    if Q.mass_top40 > 173.1022 and Q.n_muon < 1.0:
        z += -1.311414e-06 * (Q.mass_top40 - 173.1022) * (1.0 - Q.n_muon)
    if Q.n_s3d_above_3 < 10.0 and Q.dzerr_0 < 0.03430176:
        z += 0.01873322 * (10.0 - Q.n_s3d_above_3) * (0.03430176 - Q.dzerr_0)
    if Q.lep_ptrel < 27.3236 and Q.eta_28 < -0.210083:
        z += 0.003135605 * (27.3236 - Q.lep_ptrel) * (-0.210083 - Q.eta_28)
    if Q.lep_ptrel < 18.7678 and Q.n_dr_0p4_up < 15.0:
        z += -0.0005741069 * (18.7678 - Q.lep_ptrel) * (15.0 - Q.n_dr_0p4_up)
    if Q.n_electron > 1.0 and Q.iselectron_30 > 0.0:
        z += -0.01038305 * (Q.n_electron - 1.0) * (Q.iselectron_30 - 0.0)
    if Q.lep_iso < 0.4381892 and Q.sip_3d_3 < 11.13866:
        z += -0.0300551 * (0.4381892 - Q.lep_iso) * (11.13866 - Q.sip_3d_3)
    if Q.z_displaced3 > 0.02600452 and Q.isphoton_14 > 0.0:
        z += 0.04639162 * (Q.z_displaced3 - 0.02600452) * (Q.isphoton_14 - 0.0)
    if Q.mass_displaced3 > 39.09615 and Q.sip_3d_3 < 33.08364:
        z += 0.0006573448 * (Q.mass_displaced3 - 39.09615) * (33.08364 - Q.sip_3d_3)
    if Q.tau3 < 0.1024935 and Q.tdz_33 > -0.01794241:
        z += -0.4400312 * (0.1024935 - Q.tau3) * (Q.tdz_33 - -0.01794241)
    if Q.lead_ch_sdz < 1.104348 and Q.iselectron_1 > 0.0:
        z += -0.007165492 * (1.104348 - Q.lead_ch_sdz) * (Q.iselectron_1 - 0.0)
    if Q.n_dr_0_0p05 < 2.0 and Q.isnhad_27 < 1.0:
        z += -0.01999838 * (2.0 - Q.n_dr_0_0p05) * (1.0 - Q.isnhad_27)
    if Q.n_lepton < 2.0 and Q.sj3_pairmax_over_m < 0.9333327:
        z += -0.4381617 * (2.0 - Q.n_lepton) * (0.9333327 - Q.sj3_pairmax_over_m)
    if Q.tau32 < 0.6206221 and Q.jet_charge_k03 > 0.02942741:
        z += -0.5736384 * (0.6206221 - Q.tau32) * (Q.jet_charge_k03 - 0.02942741)
    if Q.n_sd0_above_5 > 4.0 and Q.sip_3d_3 < 11.13866:
        z += -0.04963754 * (Q.n_sd0_above_5 - 4.0) * (11.13866 - Q.sip_3d_3)
    if Q.tau3 < 0.1024935 and Q.sip_3d_3 < 577.991:
        z += -0.005382729 * (0.1024935 - Q.tau3) * (577.991 - Q.sip_3d_3)
    if Q.sj2_mass1 < 21.21062 and Q.phi_26 < 0.04891968:
        z += -0.0141898 * (21.21062 - Q.sj2_mass1) * (0.04891968 - Q.phi_26)
    if Q.sum_e > 550.9555 and Q.ismuon_4 > 0.0:
        z += 3.55468e-05 * (Q.sum_e - 550.9555) * (Q.ismuon_4 - 0.0)
    return z


def neuron_79(Q):
    z = -0.0001469088
    return z


def neuron_80(Q):
    z = -8.667279e-05
    return z


def neuron_81(Q):
    z = -0.4606702
    if Q.z_displaced3 < 0.03624058:
        z += 1.348939 * Q.z_displaced3 - 0.1432004
    if 0.03624058 <= Q.z_displaced3 < 0.1061578:
        z += 3.434444 * Q.z_displaced3 - 0.2187803
    if Q.z_displaced3 >= 0.1061578:
        z += 2.085505 * Q.z_displaced3 - 0.07557992
    if Q.n_pairs_kt_above_3 < 22.0:
        z += -0.01639684 * Q.n_pairs_kt_above_3 + 0.3607306
    if Q.n_s3d_above_3 < 3.0:
        z += -0.06539898 * Q.n_s3d_above_3 + 0.3269949
    if 3.0 <= Q.n_s3d_above_3 < 5.0:
        z += 0.07272961 * Q.n_s3d_above_3 - 0.08739086
    if 5.0 <= Q.n_s3d_above_3 < 10.0:
        z += 0.1381286 * Q.n_s3d_above_3 - 0.4143858
    if Q.n_s3d_above_3 >= 10.0:
        z += 0.1837974 * Q.n_s3d_above_3 - 0.8710735
    z += -2.624149 * Q.M2
    if 97.12186 <= Q.mass_top40 < 119.1279:
        z += 0.01419142 * Q.mass_top40 - 1.378297
    if Q.mass_top40 >= 119.1279:
        z += -0.001312908 * Q.mass_top40 + 0.4687007
    if 99.54528 <= Q.mass_top50 < 115.614:
        z += -0.01683564 * Q.mass_top50 + 1.675908
    if 115.614 <= Q.mass_top50 < 125.4927:
        z += 0.004085016 * Q.mass_top50 - 0.742812
    if 125.4927 <= Q.mass_top50 < 161.1264:
        z += 0.01317425 * Q.mass_top50 - 1.883445
    if Q.mass_top50 >= 161.1264:
        z += 0.001534011 * Q.mass_top50 - 0.00789433
    if Q.LHA < 0.4242439:
        z += -2.197335 * Q.LHA + 0.9322057
    if Q.max_abs_d0 < 5.8125:
        z += -0.04205333 * Q.max_abs_d0 + 0.2698447
    if 5.8125 <= Q.max_abs_d0 < 10.52344:
        z += -0.005393779 * Q.max_abs_d0 + 0.0567611
    if Q.mass_top30 < 52.72207:
        z += -0.02858623 * Q.mass_top30 + 1.507125
    if Q.max_dr < 0.7634316:
        z += 0.2520806 * Q.max_dr - 0.1924463
    if Q.sip_3d_3 < 2.723325:
        z += -0.1713988 * Q.sip_3d_3 + 0.5294479
    if 2.723325 <= Q.sip_3d_3 < 577.991:
        z += -0.0001089463 * Q.sip_3d_3 + 0.06296997
    if Q.n_sd0_above_3 < 8.0:
        z += 0.01221131 * Q.n_sd0_above_3 - 0.0976905
    if Q.n_sd0_above_3 >= 9.0:
        z += -0.05477631 * Q.n_sd0_above_3 + 0.4929868
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.003328864 * Q.n_pairs_kt_above_1 + 0.2663091
    if Q.z_dr_0p4_up < 0.01181938:
        z += -0.5584928 * Q.z_dr_0p4_up + 0.006601036
    if Q.jet_abs_eta < 1.203438:
        z += -0.07424232 * Q.jet_abs_eta + 0.08934604
    if Q.tdz_1 < -0.1170754:
        z += -0.1973566 * Q.tdz_1 - 0.0231056
    if Q.lep_ptrel < 6.983043:
        z += -0.01043538 * Q.lep_ptrel + 0.07287073
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.04268198 * Q.lep_ptrel - 1.166226
    if Q.lep_ptrel >= 43.20788:
        z += -0.02368131 * Q.lep_ptrel + 1.701192
    if Q.sum_z_dr < 0.07290954:
        z += -8.625408 * Q.sum_z_dr + 0.6288745
    if Q.n_dr_0p4_up < 15.0:
        z += 0.006960862 * Q.n_dr_0p4_up - 0.1044129
    if Q.tau2 < 0.1076451:
        z += 2.451421 * Q.tau2 - 0.2638834
    if Q.ecf_g32 < 0.0009010251:
        z += 291.3597 * Q.ecf_g32 - 0.2625224
    if Q.mass_displaced3 < 1.777286:
        z += 0.1183672 * Q.mass_displaced3 - 0.2103723
    if Q.mass_displaced5 < 13.97388:
        z += -0.02957484 * Q.mass_displaced5 + 0.4132752
    if Q.tau21 < 0.2377332:
        z += -1.743763 * Q.tau21 + 0.4145504
    if Q.lep_z < 0.221436:
        z += 0.2012066 * Q.lep_z - 0.04455439
    if Q.pair_mean_lndelta >= -1.352792:
        z += 0.3332291 * Q.pair_mean_lndelta + 0.4507896
    if Q.mass_2charged >= 20.76537:
        z += 0.004485071 * Q.mass_2charged - 0.09313416
    if Q.sj4_pair_mass_max >= 63.34656:
        z += -0.003976265 * Q.sj4_pair_mass_max + 0.2518827
    if Q.n_s3d_above_10 >= 3.0:
        z += 0.08771872 * Q.n_s3d_above_10 - 0.2631562
    if Q.n_sd0_above_5 >= 5.0:
        z += -0.08001766 * Q.n_sd0_above_5 + 0.4000883
    if Q.mass_charged >= 99.20396:
        z += 5.603676e-05 * Q.mass_charged - 0.005559069
    if Q.z_electron < 0.01329067:
        z += 8.510119 * Q.z_electron - 0.1131052
    if Q.sj3_pairmin_over_m >= 0.2477126:
        z += 0.006417762 * Q.sj3_pairmin_over_m - 0.00158976
    if Q.z_displaced5 >= 0.01810676:
        z += 0.684915 * Q.z_displaced5 - 0.01240159
    if Q.min_pair_mass < 1.763283:
        z += -0.02801894 * Q.min_pair_mass + 0.04940532
    if Q.sip_3d_2 < 70.05844:
        z += 0.0003975468 * Q.sip_3d_2 - 0.04328976
    if 70.05844 <= Q.sip_3d_2 < 226.3008:
        z += 9.880969e-05 * Q.sip_3d_2 - 0.02236071
    if Q.tdz_2 >= -0.02221619:
        z += 0.135654 * Q.tdz_2 + 0.003013714
    if Q.tau4 < 0.02949822:
        z += -0.3449495 * Q.tau4 + 0.0101754
    if Q.n_s3d_above_3 > 3.0 and Q.sip_3d_2 < 447.0873:
        z += 0.0002190081 * (Q.n_s3d_above_3 - 3.0) * (447.0873 - Q.sip_3d_2)
    if Q.z_displaced3 > 0.03624058 and Q.z_charged_had > 0.265564:
        z += -5.153242 * (Q.z_displaced3 - 0.03624058) * (Q.z_charged_had - 0.265564)
    if Q.z_displaced3 > 0.03624058 and Q.sip_3d_3 < 16.23838:
        z += 0.03608472 * (Q.z_displaced3 - 0.03624058) * (16.23838 - Q.sip_3d_3)
    if Q.n_s3d_above_3 > 3.0 and Q.tau32 < 0.8882532:
        z += -0.07957986 * (Q.n_s3d_above_3 - 3.0) * (0.8882532 - Q.tau32)
    if Q.max_abs_d0 < 10.52344 and Q.sip_3d_3 < 7.345216:
        z += -0.007161513 * (10.52344 - Q.max_abs_d0) * (7.345216 - Q.sip_3d_3)
    if Q.n_s3d_above_3 > 3.0 and Q.n_charged_had < 33.0:
        z += 0.00454412 * (Q.n_s3d_above_3 - 3.0) * (33.0 - Q.n_charged_had)
    if Q.n_pairs_kt_above_3 < 22.0 and Q.n_pairs_kt_above_30 < 1.0:
        z += -0.01886384 * (22.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.n_pairs_kt_above_30)
    if Q.n_s3d_above_3 > 3.0 and Q.tau43 < 0.8597199:
        z += -0.1585778 * (Q.n_s3d_above_3 - 3.0) * (0.8597199 - Q.tau43)
    if Q.z_displaced3 > 0.03624058 and Q.lep_dr < 0.3014662:
        z += -3.305063 * (Q.z_displaced3 - 0.03624058) * (0.3014662 - Q.lep_dr)
    if Q.n_s3d_above_3 > 3.0 and Q.mass_displaced5 < 21.12768:
        z += -0.005096796 * (Q.n_s3d_above_3 - 3.0) * (21.12768 - Q.mass_displaced5)
    if Q.n_s3d_above_3 > 3.0 and Q.tau54 < 0.9036234:
        z += -0.1544608 * (Q.n_s3d_above_3 - 3.0) * (0.9036234 - Q.tau54)
    if Q.LHA < 0.4242439 and Q.n_lund < 8.0:
        z += 0.04449634 * (0.4242439 - Q.LHA) * (8.0 - Q.n_lund)
    if Q.z_displaced3 > 0.03624058 and Q.mass_2charged < 1.839882:
        z += -0.8024067 * (Q.z_displaced3 - 0.03624058) * (1.839882 - Q.mass_2charged)
    if Q.n_s3d_above_3 < 5.0 and Q.eccentricity > 0.5174679:
        z += -0.01756949 * (5.0 - Q.n_s3d_above_3) * (Q.eccentricity - 0.5174679)
    if Q.z_displaced3 < 0.1061578 and Q.lep_z < 0.03021637:
        z += 163.7996 * (0.1061578 - Q.z_displaced3) * (0.03021637 - Q.lep_z)
    if Q.z_displaced3 > 0.03624058 and Q.z_photon > 0.3887278:
        z += 6.425062 * (Q.z_displaced3 - 0.03624058) * (Q.z_photon - 0.3887278)
    if Q.z_displaced3 > 0.03624058 and Q.sj2_dr > 0.2403736:
        z += 0.8955875 * (Q.z_displaced3 - 0.03624058) * (Q.sj2_dr - 0.2403736)
    if Q.n_s3d_above_3 > 3.0 and Q.n_lund_kt_above_5 > 2.0:
        z += -0.005200856 * (Q.n_s3d_above_3 - 3.0) * (Q.n_lund_kt_above_5 - 2.0)
    if Q.tau2 < 0.1076451 and Q.n_muon > 0.0:
        z += 0.7906942 * (0.1076451 - Q.tau2) * (Q.n_muon - 0.0)
    if Q.z_displaced3 > 0.03624058 and Q.z_muon > 0.03898651:
        z += 1.336446 * (Q.z_displaced3 - 0.03624058) * (Q.z_muon - 0.03898651)
    if Q.z_displaced3 > 0.03624058 and Q.n_lund > 5.0:
        z += 0.07405369 * (Q.z_displaced3 - 0.03624058) * (Q.n_lund - 5.0)
    if Q.mass_top40 > 97.12186 and Q.tau43_b2 > 0.6509029:
        z += -0.0005334379 * (Q.mass_top40 - 97.12186) * (Q.tau43_b2 - 0.6509029)
    if Q.n_s3d_above_3 > 3.0 and Q.isnhad_34 > 0.0:
        z += 0.0007722852 * (Q.n_s3d_above_3 - 3.0) * (Q.isnhad_34 - 0.0)
    if Q.n_s3d_above_3 > 3.0 and Q.pt1_over_pt0 < 0.2477181:
        z += 0.02174963 * (Q.n_s3d_above_3 - 3.0) * (0.2477181 - Q.pt1_over_pt0)
    if Q.tau21 < 0.2377332 and Q.pt_balance01 < 0.1985369:
        z += 13.03935 * (0.2377332 - Q.tau21) * (0.1985369 - Q.pt_balance01)
    if Q.n_sd0_above_3 > 9.0 and Q.tdz_47 < 0.0:
        z += -0.02980991 * (Q.n_sd0_above_3 - 9.0) * (0.0 - Q.tdz_47)
    if Q.lep_ptrel < 6.983043 and Q.ismuon_73 > 0.0:
        z += -0.1596422 * (6.983043 - Q.lep_ptrel) * (Q.ismuon_73 - 0.0)
    if Q.lep_ptrel < 6.983043 and Q.ismuon_11 > 0.0:
        z += -0.011309 * (6.983043 - Q.lep_ptrel) * (Q.ismuon_11 - 0.0)
    if Q.z_displaced3 > 0.03624058 and Q.phi_79 > 0.0:
        z += -0.2065048 * (Q.z_displaced3 - 0.03624058) * (Q.phi_79 - 0.0)
    if Q.tau2 < 0.1076451 and Q.jet_charge_k03 > 0.2021837:
        z += -0.9324871 * (0.1076451 - Q.tau2) * (Q.jet_charge_k03 - 0.2021837)
    if Q.mass_top40 > 119.1279 and Q.isphoton_32 < 1.0:
        z += -0.0005394866 * (Q.mass_top40 - 119.1279) * (1.0 - Q.isphoton_32)
    if Q.sj4_pair_mass_max > 63.34656 and Q.z_neutral_had < 0.3788785:
        z += 0.01056916 * (Q.sj4_pair_mass_max - 63.34656) * (0.3788785 - Q.z_neutral_had)
    if Q.sj3_pairmin_over_m > 0.2477126 and Q.dr_16 < 0.2877534:
        z += -0.05801783 * (Q.sj3_pairmin_over_m - 0.2477126) * (0.2877534 - Q.dr_16)
    if Q.min_pair_mass < 1.763283 and Q.isnhad_8 < 1.0:
        z += -0.003045225 * (1.763283 - Q.min_pair_mass) * (1.0 - Q.isnhad_8)
    if Q.n_s3d_above_3 > 10.0 and Q.isphoton_0 < 1.0:
        z += 0.04857351 * (Q.n_s3d_above_3 - 10.0) * (1.0 - Q.isphoton_0)
    if Q.sj4_pair_mass_max > 63.34656 and Q.eta_37 < 0.1695557:
        z += 0.0005473263 * (Q.sj4_pair_mass_max - 63.34656) * (0.1695557 - Q.eta_37)
    if Q.tdz_2 > -0.02221619 and Q.ismuon_12 > 0.0:
        z += -0.4527213 * (Q.tdz_2 - -0.02221619) * (Q.ismuon_12 - 0.0)
    if Q.max_dr < 0.7634316 and Q.ismuon_49 > 0.0:
        z += -3.104466 * (0.7634316 - Q.max_dr) * (Q.ismuon_49 - 0.0)
    if Q.lep_ptrel < 6.983043 and Q.pt_balance01 > 0.1985369:
        z += -0.04578965 * (6.983043 - Q.lep_ptrel) * (Q.pt_balance01 - 0.1985369)
    if Q.z_displaced3 > 0.03624058 and Q.td0_33 < 0.04032236:
        z += -0.6921025 * (Q.z_displaced3 - 0.03624058) * (0.04032236 - Q.td0_33)
    if Q.tdz_2 > -0.02221619 and Q.dzerr_44 < 0.0881958:
        z += -0.2380083 * (Q.tdz_2 - -0.02221619) * (0.0881958 - Q.dzerr_44)
    return z


def neuron_82(Q):
    z = 0.0001161494
    return z


def neuron_83(Q):
    z = 1.014435
    if Q.mass_top40 < 78.33213:
        z += 0.0212706 * Q.mass_top40 - 3.681987
    if 78.33213 <= Q.mass_top40 < 119.1279:
        z += 0.01070905 * Q.mass_top40 - 2.854678
    if 119.1279 <= Q.mass_top40 < 155.8928:
        z += 0.01655406 * Q.mass_top40 - 3.550982
    if 155.8928 <= Q.mass_top40 < 173.1022:
        z += 0.03679995 * Q.mass_top40 - 6.70717
    if Q.mass_top40 >= 173.1022:
        z += 0.01552936 * Q.mass_top40 - 3.025184
    if Q.n_s3d_above_3 < 3.0:
        z += -0.2014113 * Q.n_s3d_above_3 + 1.625597
    if 3.0 <= Q.n_s3d_above_3 < 6.0:
        z += -0.145909 * Q.n_s3d_above_3 + 1.45909
    if 6.0 <= Q.n_s3d_above_3 < 10.0:
        z += -0.08285365 * Q.n_s3d_above_3 + 1.080758
    if Q.n_s3d_above_3 >= 10.0:
        z += 0.06305536 * Q.n_s3d_above_3 - 0.3783322
    if Q.ecf_g42 < 2.448333e-05:
        z += 7156.661 * Q.ecf_g42 - 0.1752189
    if Q.z_displaced3 < 0.06416437:
        z += -4.968291 * Q.z_displaced3 + 0.4731829
    if 0.06416437 <= Q.z_displaced3 < 0.1681173:
        z += -1.485246 * Q.z_displaced3 + 0.2496956
    if Q.sum_zz_dr2 < 0.05018249:
        z += -7.909434 * Q.sum_zz_dr2 + 0.3969151
    if Q.e3 < 0.00228569:
        z += -47.44535 * Q.e3 + 0.1084454
    if Q.mass_top15 >= 80.3877:
        z += -0.01020063 * Q.mass_top15 + 0.8200048
    z += 688.4406 * Q.ecf_g41
    if Q.pair_max_lnm2 >= 7.347625:
        z += 0.260536 * Q.pair_max_lnm2 - 1.914321
    if Q.n_s3d_above_10 >= 1.0:
        z += 0.02894661 * Q.n_s3d_above_10 - 0.02894661
    if Q.sip_3d_2 < 4.636903:
        z += -0.1055361 * Q.sip_3d_2 + 0.187296
    if 4.636903 <= Q.sip_3d_2 < 30.49226:
        z += 0.007245913 * Q.sip_3d_2 - 0.3356631
    if 30.49226 <= Q.sip_3d_2 < 226.3008:
        z += 0.0005858726 * Q.sip_3d_2 - 0.1325834
    if 0.04393457 <= Q.e2 < 0.07745967:
        z += -1.305046 * Q.e2 + 0.05733665
    if Q.e2 >= 0.07745967:
        z += -5.566639 * Q.e2 + 0.3874382
    if Q.mass_displaced3 < 1.251841:
        z += 0.2194555 * Q.mass_displaced3 - 0.09141202
    if 1.251841 <= Q.mass_displaced3 < 13.03663:
        z += -0.01555491 * Q.mass_displaced3 + 0.2027837
    if Q.mass_neutral < 49.25971:
        z += 0.00286733 * Q.mass_neutral - 0.1412439
    if Q.mass_top30 >= 102.3652:
        z += -0.007126503 * Q.mass_top30 + 0.7295063
    if Q.N2_b05 < 0.4265629:
        z += 1.913355 * Q.N2_b05 - 0.8161665
    if Q.N2_b05 >= 0.4518419:
        z += 0.9544522 * Q.N2_b05 - 0.4312615
    if Q.sj4_pair_mass_max < 72.862:
        z += 0.001715788 * Q.sj4_pair_mass_max + 0.1798414
    if 72.862 <= Q.sj4_pair_mass_max < 115.5142:
        z += -0.005528197 * Q.sj4_pair_mass_max + 0.7076526
    if 115.5142 <= Q.sj4_pair_mass_max < 128.0079:
        z += 0.004153401 * Q.sj4_pair_mass_max - 0.4107095
    if Q.sj4_pair_mass_max >= 128.0079:
        z += 0.009681598 * Q.sj4_pair_mass_max - 1.118362
    if Q.M2 < 0.06228948:
        z += 7.295026 * Q.M2 - 0.4544034
    if 62.03827 <= Q.sd_mass < 83.61981:
        z += 0.01855601 * Q.sd_mass - 1.151183
    if 83.61981 <= Q.sd_mass < 88.81751:
        z += -0.02316971 * Q.sd_mass + 2.337914
    if 88.81751 <= Q.sd_mass < 154.5947:
        z += -0.008069057 * Q.sd_mass + 0.9967118
    if Q.sd_mass >= 154.5947:
        z += 0.007208657 * Q.sd_mass - 1.365141
    if Q.pt1_over_pt0 < 0.5779883:
        z += -0.1093316 * Q.pt1_over_pt0 + 0.06319238
    if Q.mass_top10 >= 103.4976:
        z += 0.001042661 * Q.mass_top10 - 0.1079129
    if Q.mass < 90.08945:
        z += -0.007644336 * Q.mass + 1.094241
    if 90.08945 <= Q.mass < 117.4867:
        z += -0.01480322 * Q.mass + 1.739181
    if Q.tdz_1 < -0.1170754:
        z += -0.1003485 * Q.tdz_1 - 0.01174834
    if Q.e4_b05 < 0.000141779:
        z += 5178.259 * Q.e4_b05 - 0.7341684
    if Q.log_sum_pt < 6.661019:
        z += 0.3102992 * Q.log_sum_pt - 2.066909
    if Q.n_charged < 33.0:
        z += -0.01343091 * Q.n_charged + 0.4432201
    if Q.tau5 >= 0.01222366:
        z += 5.159668 * Q.tau5 - 0.06307005
    if Q.n_pt_above_5 < 30.0:
        z += 0.01095916 * Q.n_pt_above_5 - 0.3287748
    if Q.tau32 < 0.3951525:
        z += -1.360766 * Q.tau32 + 0.5377102
    if Q.lep_iso < 6.185635:
        z += -0.02556612 * Q.lep_iso + 0.1581427
    if Q.z_photon >= 0.4305934:
        z += 1.216712 * Q.z_photon - 0.5239082
    if Q.z_neutral_had >= 0.3788785:
        z += 1.007744 * Q.z_neutral_had - 0.3818125
    if Q.sj3_mass3 < 0.3886647:
        z += -0.08694141 * Q.sj3_mass3 + 0.03379106
    if Q.M3 < 0.01517988:
        z += 16.6354 * Q.M3 - 0.2525233
    if Q.n_sd0_above_5 >= 7.0:
        z += 0.07072324 * Q.n_sd0_above_5 - 0.4950627
    if Q.sj3_dr23 < 0.8799072:
        z += -0.1419352 * Q.sj3_dr23 + 0.1248898
    if Q.e3_b2 < 0.0002536827:
        z += 1025.314 * Q.e3_b2 - 0.2601045
    if Q.n_lepton < 2.0:
        z += 0.03066701 * Q.n_lepton - 0.06133401
    if Q.z_charged_had < 0.3603262:
        z += 0.2509025 * Q.z_charged_had - 0.09040676
    if Q.C3_b05 < 0.2250047:
        z += 0.07345217 * Q.C3_b05 - 0.01652708
    if Q.psi_0p2 >= 0.8857951:
        z += -0.9663832 * Q.psi_0p2 + 0.8560175
    if Q.td0_3 >= 0.1091249:
        z += 0.2443821 * Q.td0_3 - 0.02666816
    if Q.dr_max_012 < 0.02210827:
        z += 8.5439 * Q.dr_max_012 - 0.1888908
    if Q.max_abs_d0 < 0.3977051:
        z += 0.1235649 * Q.max_abs_d0 - 0.1473794
    if 0.3977051 <= Q.max_abs_d0 < 10.52344:
        z += 0.00970172 * Q.max_abs_d0 - 0.1020954
    if Q.lund3_lndelta >= -2.817283:
        z += -0.1223431 * Q.lund3_lndelta - 0.3446751
    if Q.sj4_dr_min < 0.3112717:
        z += 0.2886183 * Q.sj4_dr_min - 0.08983871
    if Q.n_electron >= 1.0:
        z += 0.004793817 * Q.n_electron - 0.004793817
    if Q.n_sd0_above_10 < 7.0:
        z += 0.0396464 * Q.n_sd0_above_10 - 0.2775248
    if Q.lep_ptrel < 27.3236:
        z += -0.01175687 * Q.lep_ptrel + 0.32124
    if Q.z_dr_0_0p05 < 0.004294711:
        z += 15.87575 * Q.z_dr_0_0p05 - 0.06818173
    if Q.pair_mean_lnm2 >= 1.96906:
        z += 0.1527478 * Q.pair_mean_lnm2 - 0.3007697
    if Q.pair_mean_lndelta >= -2.115543:
        z += -0.08034225 * Q.pair_mean_lndelta - 0.1699675
    if Q.n_lund_kt_above_5 >= 1.0:
        z += -0.03677292 * Q.n_lund_kt_above_5 + 0.03677292
    if Q.n_sd0_above_3 < 8.0:
        z += 0.02240339 * Q.n_sd0_above_3 - 0.1792271
    if Q.mass_top40 < 173.1022 and Q.mass_displaced5 > 0.0:
        z += 3.338495e-05 * (173.1022 - Q.mass_top40) * (Q.mass_displaced5 - 0.0)
    if Q.mass_top40 > 78.33213 and Q.lep_ptrel < 18.7678:
        z += -0.0004032814 * (Q.mass_top40 - 78.33213) * (18.7678 - Q.lep_ptrel)
    if Q.ecf_g42 < 2.448333e-05 and Q.n_s3d_above_10 < 6.0:
        z += 1596.876 * (2.448333e-05 - Q.ecf_g42) * (6.0 - Q.n_s3d_above_10)
    if Q.n_s3d_above_3 > 6.0 and Q.sip_3d_3 < 175.9957:
        z += 0.001611056 * (Q.n_s3d_above_3 - 6.0) * (175.9957 - Q.sip_3d_3)
    if Q.sum_zz_dr2 < 0.05018249 and Q.n_lund < 16.0:
        z += 0.02778125 * (0.05018249 - Q.sum_zz_dr2) * (16.0 - Q.n_lund)
    if Q.z_displaced3 < 0.06416437 and Q.z_neutral_had < 0.2370407:
        z += 3.527 * (0.06416437 - Q.z_displaced3) * (0.2370407 - Q.z_neutral_had)
    if Q.n_s3d_above_10 > 1.0 and Q.sip_3d_1 < 351.3271:
        z += 0.0002770173 * (Q.n_s3d_above_10 - 1.0) * (351.3271 - Q.sip_3d_1)
    if Q.e3 < 0.00228569 and Q.jet_abs_eta > 0.9149342:
        z += 138.3551 * (0.00228569 - Q.e3) * (Q.jet_abs_eta - 0.9149342)
    if Q.n_s3d_above_3 > 6.0 and Q.sj3_pairmin_over_m > 0.1636952:
        z += -0.1829269 * (Q.n_s3d_above_3 - 6.0) * (Q.sj3_pairmin_over_m - 0.1636952)
    if Q.mass_top40 > 78.33213 and Q.D2_b2 < 15.17086:
        z += -0.0007345444 * (Q.mass_top40 - 78.33213) * (15.17086 - Q.D2_b2)
    if Q.mass_top40 < 173.1022 and Q.D2_b2 < 15.17086:
        z += 0.0002094666 * (173.1022 - Q.mass_top40) * (15.17086 - Q.D2_b2)
    if Q.n_s3d_above_3 < 10.0 and Q.z_neutral > 0.1384639:
        z += -0.1197424 * (10.0 - Q.n_s3d_above_3) * (Q.z_neutral - 0.1384639)
    if Q.n_s3d_above_3 > 6.0 and Q.z_dr_0p1_0p2 < 0.2949288:
        z += 0.08816076 * (Q.n_s3d_above_3 - 6.0) * (0.2949288 - Q.z_dr_0p1_0p2)
    if Q.n_s3d_above_10 > 1.0 and Q.D3_b2 < 0.3264446:
        z += 0.001249586 * (Q.n_s3d_above_10 - 1.0) * (0.3264446 - Q.D3_b2)
    if Q.mass < 117.4867 and Q.n_dr_0p1_0p2 > 1.0:
        z += 0.0002434814 * (117.4867 - Q.mass) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.mass_top15 > 80.3877 and Q.tdz_29 > -0.09486766:
        z += 0.0001533914 * (Q.mass_top15 - 80.3877) * (Q.tdz_29 - -0.09486766)
    if Q.n_s3d_above_3 > 6.0 and Q.ismuon_24 > 0.0:
        z += -0.0171157 * (Q.n_s3d_above_3 - 6.0) * (Q.ismuon_24 - 0.0)
    if Q.n_s3d_above_3 > 6.0 and Q.phi_3 > 0.06884766:
        z += -0.09898119 * (Q.n_s3d_above_3 - 6.0) * (Q.phi_3 - 0.06884766)
    if Q.mass_top40 > 155.8928 and Q.lund2_lndelta < -1.411949:
        z += 0.0005424063 * (Q.mass_top40 - 155.8928) * (-1.411949 - Q.lund2_lndelta)
    if Q.M2 < 0.06228948 and Q.eta_37 > 0.0:
        z += 17.66562 * (0.06228948 - Q.M2) * (Q.eta_37 - 0.0)
    if Q.n_s3d_above_10 > 1.0 and Q.dr_16 < 0.3788372:
        z += -0.02710982 * (Q.n_s3d_above_10 - 1.0) * (0.3788372 - Q.dr_16)
    if Q.z_displaced3 < 0.1681173 and Q.ismuon_2 > 0.0:
        z += -0.6694596 * (0.1681173 - Q.z_displaced3) * (Q.ismuon_2 - 0.0)
    if Q.e3_b2 < 0.0002536827 and Q.iselectron_12 > 0.0:
        z += -138.11 * (0.0002536827 - Q.e3_b2) * (Q.iselectron_12 - 0.0)
    if Q.mass_top10 > 103.4976 and Q.mratio_min_012 > 0.0111128:
        z += -0.01230573 * (Q.mass_top10 - 103.4976) * (Q.mratio_min_012 - 0.0111128)
    if Q.e2 > 0.04393457 and Q.iselectron_11 > 0.0:
        z += -0.7449188 * (Q.e2 - 0.04393457) * (Q.iselectron_11 - 0.0)
    if Q.sj4_dr_min < 0.3112717 and Q.td0_30 < 0.04589821:
        z += -0.1124673 * (0.3112717 - Q.sj4_dr_min) * (0.04589821 - Q.td0_30)
    if Q.n_s3d_above_10 > 1.0 and Q.ismuon_27 > 0.0:
        z += 0.007710979 * (Q.n_s3d_above_10 - 1.0) * (Q.ismuon_27 - 0.0)
    return z


def neuron_84(Q):
    z = 2.036519e-06
    return z


def neuron_85(Q):
    z = -9.651125e-05
    return z


def neuron_86(Q):
    z = 6.858682e-05
    return z


def neuron_87(Q):
    z = 4.752037e-06
    return z


def neuron_88(Q):
    z = 0.0001273689
    return z


def neuron_89(Q):
    z = -0.0001347911
    return z


def neuron_90(Q):
    z = 0.0006315595
    return z


def neuron_91(Q):
    z = -6.273416e-05
    return z


def neuron_92(Q):
    z = -7.732997e-06
    return z


def neuron_93(Q):
    z = 7.897627e-05
    return z


def neuron_94(Q):
    z = -9.460902e-05
    return z


def neuron_95(Q):
    z = 9.859636e-05
    return z


def neuron_96(Q):
    z = -0.0001331312
    return z


def neuron_97(Q):
    z = -0.6389725
    if Q.mass_displaced3 < 18.80005:
        z += -0.02729382 * Q.mass_displaced3 + 1.767064
    if 18.80005 <= Q.mass_displaced3 < 39.09615:
        z += -0.06178223 * Q.mass_displaced3 + 2.415447
    if Q.mass < 95.14961:
        z += -0.03062213 * Q.mass + 0.9431633
    if 95.14961 <= Q.mass < 100.4835:
        z += 0.004510609 * Q.mass - 2.399703
    if 100.4835 <= Q.mass < 123.7928:
        z += 0.01906846 * Q.mass - 3.862526
    if 123.7928 <= Q.mass < 164.4374:
        z += 0.03695412 * Q.mass - 6.076641
    if Q.e3_b2 < 9.621843e-05:
        z += 1707.394 * Q.e3_b2 - 0.07279328
    if 9.621843e-05 <= Q.e3_b2 < 0.0002536827:
        z += 1361.4 * Q.e3_b2 - 0.03950227
    if 0.0002536827 <= Q.e3_b2 < 0.0007909605:
        z += -569.2797 * Q.e3_b2 + 0.4502777
    if Q.pair_mean_lnm2 >= 2.09826:
        z += -0.008546312 * Q.pair_mean_lnm2 + 0.01793239
    if Q.mass_top50 < 79.27954:
        z += 0.01781419 * Q.mass_top50 - 1.060977
    if 79.27954 <= Q.mass_top50 < 115.614:
        z += -0.002530163 * Q.mass_top50 + 0.5519139
    if 115.614 <= Q.mass_top50 < 161.1264:
        z += -0.005699362 * Q.mass_top50 + 0.9183177
    if Q.sip_3d_2 < 447.0873:
        z += 0.0003027244 * Q.sip_3d_2 - 0.1353442
    if Q.tau32 < 0.6206221:
        z += 0.4000998 * Q.tau32 - 0.2483108
    if Q.mass_top40 < 126.8853:
        z += -0.008949711 * Q.mass_top40 + 1.135586
    if 53.51926 <= Q.sj2_mass1 < 77.42768:
        z += 0.01472281 * Q.sj2_mass1 - 0.7879536
    if 77.42768 <= Q.sj2_mass1 < 91.2852:
        z += -0.02488802 * Q.sj2_mass1 + 2.279021
    if Q.sj2_mass1 >= 91.2852:
        z += 0.004981581 * Q.sj2_mass1 - 0.4476316
    if Q.n_pairs_kt_above_1 < 325.0:
        z += -0.001249587 * Q.n_pairs_kt_above_1 + 0.4061159
    if Q.M2_b2 < 0.01015545:
        z += 4.478571 * Q.M2_b2 - 0.04548193
    if Q.psi_0p1 >= 0.7386202:
        z += -0.1613617 * Q.psi_0p1 + 0.119185
    if Q.sj3_pair_mass_max < 73.24742:
        z += -0.007756706 * Q.sj3_pair_mass_max + 1.279363
    if 73.24742 <= Q.sj3_pair_mass_max < 109.8208:
        z += -0.0160053 * Q.sj3_pair_mass_max + 1.883551
    if 109.8208 <= Q.sj3_pair_mass_max < 120.6471:
        z += -0.01162315 * Q.sj3_pair_mass_max + 1.4023
    if Q.max_abs_d0 < 5.8125:
        z += 0.05384256 * Q.max_abs_d0 - 0.3129599
    if Q.sj3_mass3 < 2.68613:
        z += -0.07025317 * Q.sj3_mass3 + 0.1887091
    if Q.lund3_lndelta >= -1.902701:
        z += 0.1259943 * Q.lund3_lndelta + 0.2397295
    if -1.388411 <= Q.lund_max_lndelta < -0.6897565:
        z += -0.2683599 * Q.lund_max_lndelta - 0.3725938
    if Q.lund_max_lndelta >= -0.6897565:
        z += 0.390406 * Q.lund_max_lndelta + 0.0817942
    if Q.e2 >= 0.09764648:
        z += 1.866033 * Q.e2 - 0.1822115
    if Q.z_neutral_had >= 0.07573803:
        z += -0.3779716 * Q.z_neutral_had + 0.02862682
    if Q.z_muon < 0.1403354:
        z += 1.917277 * Q.z_muon - 0.2690618
    if Q.n_sd0_above_3 >= 1.0:
        z += 0.1712955 * Q.n_sd0_above_3 - 0.1712955
    if Q.n_s3d_above_3 >= 3.0:
        z += -0.1434952 * Q.n_s3d_above_3 + 0.4304857
    if Q.z_displaced5 >= 0.1339658:
        z += -0.6810713 * Q.z_displaced5 + 0.09124025
    if Q.lne_0 >= 5.157617:
        z += 0.1801464 * Q.lne_0 - 0.929126
    if Q.n_pairs_kt_above_3 < 34.0:
        z += 0.004386411 * Q.n_pairs_kt_above_3 + 0.04035434
    if 34.0 <= Q.n_pairs_kt_above_3 < 126.0:
        z += -0.002059699 * Q.n_pairs_kt_above_3 + 0.2595221
    if Q.n_photon < 19.0:
        z += -0.01693508 * Q.n_photon + 0.3217665
    if Q.sj3_pairmax_over_m < 0.8501496:
        z += 1.387736 * Q.sj3_pairmax_over_m - 1.179784
    if Q.n_sd0_above_2 >= 5.0:
        z += -0.08601003 * Q.n_sd0_above_2 + 0.4300502
    if Q.pair_max_lnm2 < 7.203613:
        z += -0.06872277 * Q.pair_max_lnm2 + 0.4950522
    if Q.max_dr < 0.6079631:
        z += 0.6672879 * Q.max_dr - 0.4056865
    z += 0.01551348 * Q.n_pairs_kt_above_10
    if Q.D2 >= 3.597891:
        z += 0.04924971 * Q.D2 - 0.1771951
    if Q.lund_max_lnkt < 3.521178:
        z += 0.1046477 * Q.lund_max_lnkt - 0.3684833
    if Q.n_pt_above_5 >= 27.0:
        z += -0.001907582 * Q.n_pt_above_5 + 0.05150471
    if Q.sj3_dr_min >= 0.3190414:
        z += -0.4302404 * Q.sj3_dr_min + 0.1372645
    if Q.sd_mass < 62.03827:
        z += -0.0005923817 * Q.sd_mass - 0.3324636
    if 62.03827 <= Q.sd_mass < 72.27436:
        z += 0.0376958 * Q.sd_mass - 2.707796
    if 72.27436 <= Q.sd_mass < 78.4753:
        z += 0.01699077 * Q.sd_mass - 1.211354
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += -0.07911175 * Q.sd_mass + 6.33032
    if 88.81751 <= Q.sd_mass < 100.8525:
        z += 0.004747375 * Q.sd_mass - 1.117838
    if 100.8525 <= Q.sd_mass < 115.7091:
        z += -0.01009122 * Q.sd_mass + 0.3786718
    if 115.7091 <= Q.sd_mass < 119.4443:
        z += -0.06433784 * Q.sd_mass + 6.655499
    if 119.4443 <= Q.sd_mass < 154.5947:
        z += 0.02928258 * Q.sd_mass - 4.526931
    if Q.N2_b2 < 0.1244374:
        z += 0.2399726 * Q.N2_b2 - 0.02986158
    if Q.sj3_pair_mass_min >= 59.49644:
        z += -0.01057466 * Q.sj3_pair_mass_min + 0.6291548
    if Q.e2_b05 >= 0.1510354:
        z += -0.3254477 * Q.e2_b05 + 0.04915414
    if Q.jet_charge_k05 >= 0.07435708:
        z += 0.2177952 * Q.jet_charge_k05 - 0.01619461
    if Q.e3_b2 < 0.0002536827 and Q.mass_charged > 84.24838:
        z += 55.48453 * (0.0002536827 - Q.e3_b2) * (Q.mass_charged - 84.24838)
    if Q.e3_b2 < 0.0002536827 and Q.z_displaced3 < 0.3261071:
        z += -1147.794 * (0.0002536827 - Q.e3_b2) * (0.3261071 - Q.z_displaced3)
    if Q.mass_displaced3 < 39.09615 and Q.lep_ptrel < 27.3236:
        z += -0.0003369282 * (39.09615 - Q.mass_displaced3) * (27.3236 - Q.lep_ptrel)
    if Q.mass_displaced3 < 39.09615 and Q.sum_pt_top5 < 586.125:
        z += -2.444174e-06 * (39.09615 - Q.mass_displaced3) * (586.125 - Q.sum_pt_top5)
    if Q.mass_top50 < 161.1264 and Q.n_s3d_above_3 > 1.0:
        z += -0.0006966548 * (161.1264 - Q.mass_top50) * (Q.n_s3d_above_3 - 1.0)
    if Q.e3_b2 < 0.0002536827 and Q.jet_charge_k03 > -0.05990128:
        z += 1034.651 * (0.0002536827 - Q.e3_b2) * (Q.jet_charge_k03 - -0.05990128)
    if Q.mass_top50 < 115.614 and Q.n_lepton < 1.0:
        z += 0.001196775 * (115.614 - Q.mass_top50) * (1.0 - Q.n_lepton)
    if Q.mass_top50 < 161.1264 and Q.sip_3d_3 < 175.9957:
        z += -8.2743e-06 * (161.1264 - Q.mass_top50) * (175.9957 - Q.sip_3d_3)
    if Q.mass_top40 < 126.8853 and Q.lne_5 > 2.737811:
        z += 0.002423866 * (126.8853 - Q.mass_top40) * (Q.lne_5 - 2.737811)
    if Q.mass_top40 < 126.8853 and Q.n_muon > 0.0:
        z += -0.007114009 * (126.8853 - Q.mass_top40) * (Q.n_muon - 0.0)
    if Q.e3_b2 < 0.0002536827 and Q.tau21_b2 < 0.3565533:
        z += -1133.177 * (0.0002536827 - Q.e3_b2) * (0.3565533 - Q.tau21_b2)
    if Q.tau32 < 0.6206221 and Q.z_charged_had < 0.6056728:
        z += 2.22111 * (0.6206221 - Q.tau32) * (0.6056728 - Q.z_charged_had)
    if Q.e3_b2 < 0.0002536827 and Q.n_dr_0p4_up > 8.0:
        z += 141.6232 * (0.0002536827 - Q.e3_b2) * (Q.n_dr_0p4_up - 8.0)
    if Q.max_abs_d0 < 5.8125 and Q.sip_3d_3 < 7.345216:
        z += 0.01171596 * (5.8125 - Q.max_abs_d0) * (7.345216 - Q.sip_3d_3)
    if Q.sj3_pair_mass_max < 109.8208 and Q.lnpt_44 > -18.42068:
        z += -1.00392e-05 * (109.8208 - Q.sj3_pair_mass_max) * (Q.lnpt_44 - -18.42068)
    if Q.e3_b2 < 0.0002536827 and Q.sj3_dr13 > 0.7462286:
        z += -810.6718 * (0.0002536827 - Q.e3_b2) * (Q.sj3_dr13 - 0.7462286)
    if Q.mass_displaced3 < 39.09615 and Q.n_pairs_kt_above_10 > 2.0:
        z += -0.0005836756 * (39.09615 - Q.mass_displaced3) * (Q.n_pairs_kt_above_10 - 2.0)
    if Q.mass_displaced3 < 18.80005 and Q.sip_3d_3 < 175.9957:
        z += 0.0004136921 * (18.80005 - Q.mass_displaced3) * (175.9957 - Q.sip_3d_3)
    if Q.mass_displaced3 < 39.09615 and Q.sip_3d_3 < 175.9957:
        z += -0.0001744619 * (39.09615 - Q.mass_displaced3) * (175.9957 - Q.sip_3d_3)
    if Q.n_sd0_above_3 > 1.0 and Q.sj3_dr13 > 0.6229991:
        z += -0.03488431 * (Q.n_sd0_above_3 - 1.0) * (Q.sj3_dr13 - 0.6229991)
    if Q.mass_top50 < 161.1264 and Q.min_pair_mass > 3.369962:
        z += 7.645137e-05 * (161.1264 - Q.mass_top50) * (Q.min_pair_mass - 3.369962)
    if Q.sj3_pairmax_over_m < 0.8501496 and Q.ismuon_37 > 0.0:
        z += 0.446768 * (0.8501496 - Q.sj3_pairmax_over_m) * (Q.ismuon_37 - 0.0)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.D2_b2 < 2.286291:
        z += 0.004060416 * (34.0 - Q.n_pairs_kt_above_3) * (2.286291 - Q.D2_b2)
    if Q.sj2_mass1 > 77.42768 and Q.D2_b2 < 0.8678264:
        z += -0.1116737 * (Q.sj2_mass1 - 77.42768) * (0.8678264 - Q.D2_b2)
    if Q.mass_displaced3 < 18.80005 and Q.mass_2photon > 9.82024:
        z += 0.0004448817 * (18.80005 - Q.mass_displaced3) * (Q.mass_2photon - 9.82024)
    if Q.lund_max_lndelta > -1.388411 and Q.lnerel_4 > -3.310904:
        z += -0.07871542 * (Q.lund_max_lndelta - -1.388411) * (Q.lnerel_4 - -3.310904)
    if Q.sip_3d_2 < 447.0873 and Q.td0_33 > -0.04023096:
        z += 0.0002918438 * (447.0873 - Q.sip_3d_2) * (Q.td0_33 - -0.04023096)
    if Q.n_s3d_above_3 > 3.0 and Q.d0err_10 > 0.02780151:
        z += 0.8065631 * (Q.n_s3d_above_3 - 3.0) * (Q.d0err_10 - 0.02780151)
    if Q.mass < 100.4835 and Q.iselectron_0 > 0.0:
        z += -0.009227603 * (100.4835 - Q.mass) * (Q.iselectron_0 - 0.0)
    if Q.mass_displaced3 < 39.09615 and Q.iselectron_0 > 0.0:
        z += 0.002601276 * (39.09615 - Q.mass_displaced3) * (Q.iselectron_0 - 0.0)
    if Q.sip_3d_2 < 447.0873 and Q.lead_ch_sdz < -0.6526285:
        z += -2.444245e-05 * (447.0873 - Q.sip_3d_2) * (-0.6526285 - Q.lead_ch_sdz)
    if Q.sj3_pairmax_over_m < 0.8501496 and Q.dr_53 > 0.0:
        z += -0.5311584 * (0.8501496 - Q.sj3_pairmax_over_m) * (Q.dr_53 - 0.0)
    if Q.N2_b2 < 0.1244374 and Q.eta_29 < -0.1424561:
        z += 6.086799 * (0.1244374 - Q.N2_b2) * (-0.1424561 - Q.eta_29)
    if Q.e3_b2 < 0.0007909605 and Q.dr_28 > 0.3368505:
        z += 88.35863 * (0.0007909605 - Q.e3_b2) * (Q.dr_28 - 0.3368505)
    if Q.lund_max_lndelta > -1.388411 and Q.td0_7 < 0.09541201:
        z += 0.02675582 * (Q.lund_max_lndelta - -1.388411) * (0.09541201 - Q.td0_7)
    if Q.z_displaced5 > 0.1339658 and Q.C3_b2 > 0.04229114:
        z += -2.330513 * (Q.z_displaced5 - 0.1339658) * (Q.C3_b2 - 0.04229114)
    if Q.z_neutral_had > 0.07573803 and Q.td0_23 > 0.0:
        z += -0.2270752 * (Q.z_neutral_had - 0.07573803) * (Q.td0_23 - 0.0)
    return z


def neuron_98(Q):
    z = -2.277138e-05
    return z


def neuron_99(Q):
    z = -0.001795841
    return z


def neuron_100(Q):
    z = 4.08328e-05
    return z


def neuron_101(Q):
    z = 0.0001346117
    return z


def neuron_102(Q):
    z = -3.319483e-05
    return z


def neuron_103(Q):
    z = -9.805863e-05
    return z


def neuron_104(Q):
    z = -1.853376
    if Q.mass < 120.653:
        z += -0.01993701 * Q.mass + 2.565231
    if 120.653 <= Q.mass < 149.0507:
        z += -0.005626177 * Q.mass + 0.8385855
    if Q.pair_mean_lndelta < -1.352792:
        z += -0.36898 * Q.pair_mean_lndelta - 0.499153
    if Q.N2_b05 >= 0.3667049:
        z += -1.313322 * Q.N2_b05 + 0.4816017
    if 3.208089 <= Q.mass_displaced3 < 6.341631:
        z += 0.0762373 * Q.mass_displaced3 - 0.244576
    if 6.341631 <= Q.mass_displaced3 < 26.78691:
        z += 0.03181698 * Q.mass_displaced3 + 0.03712125
    if Q.mass_displaced3 >= 26.78691:
        z += 0.006621055 * Q.mass_displaced3 + 0.7120422
    if Q.n_s3d_above_10 < 8.0:
        z += -0.05948851 * Q.n_s3d_above_10 + 0.4759081
    z += 0.001483562 * Q.mass_neutral
    if Q.tau32_b2 < 0.4680886:
        z += 0.8767366 * Q.tau32_b2 - 0.4103904
    if Q.N2 >= 0.3245983:
        z += 0.9456372 * Q.N2 - 0.3069522
    if Q.n_pairs_kt_above_3 < 47.0:
        z += 0.01383428 * Q.n_pairs_kt_above_3 - 0.6502112
    if Q.z_displaced3 < 0.01782783:
        z += 3.480104 * Q.z_displaced3 - 0.06204269
    if Q.sj3_dr_min < 0.3190414:
        z += -0.2282756 * Q.sj3_dr_min + 0.07282938
    if Q.mass_top40 < 132.5189:
        z += -0.006815838 * Q.mass_top40 + 1.005869
    if 132.5189 <= Q.mass_top40 < 155.8928:
        z += -0.004391277 * Q.mass_top40 + 0.6845684
    if Q.tau21 < 0.135772:
        z += -5.371422 * Q.tau21 + 0.734723
    if 0.135772 <= Q.tau21 < 0.2377332:
        z += -0.0532994 * Q.tau21 + 0.01267104
    if Q.tau21 >= 0.5797033:
        z += 0.9424523 * Q.tau21 - 0.5463427
    if Q.n_sd0_above_3 < 6.0:
        z += -0.06607544 * Q.n_sd0_above_3 + 0.3964526
    if Q.n_particles < 67.0:
        z += -0.006141806 * Q.n_particles + 0.411501
    if Q.n_pairs_kt_above_1 < 146.0:
        z += 0.006502738 * Q.n_pairs_kt_above_1 - 0.9493997
    if Q.C2 < 0.191059:
        z += -1.473094 * Q.C2 + 0.2814478
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += 0.02645096 * Q.sd_mass - 2.075747
    if 88.81751 <= Q.sd_mass < 106.7501:
        z += -0.004852792 * Q.sd_mass + 0.7045744
    if 106.7501 <= Q.sd_mass < 154.5947:
        z += 0.001099235 * Q.sd_mass + 0.06919504
    if 154.5947 <= Q.sd_mass < 175.9333:
        z += 0.01263625 * Q.sd_mass - 1.714366
    if Q.sd_mass >= 175.9333:
        z += 0.007524306 * Q.sd_mass - 0.8150046
    if Q.mass_charged < 99.20396:
        z += -0.0003910241 * Q.mass_charged + 0.03879114
    if Q.jet_abs_eta >= 0.6771968:
        z += 0.1471715 * Q.jet_abs_eta - 0.09966408
    if Q.z_top15_slots >= 0.9212656:
        z += 8.255076 * Q.z_top15_slots - 7.605118
    if Q.lund_max_lndelta >= -0.529318:
        z += 0.2963241 * Q.lund_max_lndelta + 0.1568497
    if Q.n_lund >= 11.0:
        z += -0.02813197 * Q.n_lund + 0.3094517
    if Q.e3 >= 0.001589861:
        z += -100.9218 * Q.e3 + 0.1604516
    if Q.mass_over_sum_pt >= 0.215596:
        z += 2.619448 * Q.mass_over_sum_pt - 0.5647425
    if Q.n_pairs_kt_above_10 < 3.0:
        z += 0.001047396 * Q.n_pairs_kt_above_10 - 0.003142188
    if Q.n_sd0_above_2 < 7.0:
        z += 0.01604203 * Q.n_sd0_above_2 - 0.1122942
    if Q.lep_iso < 6.185635:
        z += 0.01854826 * Q.lep_iso - 0.1147328
    if Q.sj3_mass3 < 0.3886647:
        z += -0.400124 * Q.sj3_mass3 + 0.1555141
    if Q.jet_charge_k03 < -0.5570337:
        z += 0.132184 * Q.jet_charge_k03 + 0.07363093
    if Q.sum_charge >= 4.0:
        z += -0.02028002 * Q.sum_charge + 0.0811201
    if Q.sip_3d_2 < 226.3008:
        z += 0.0005738236 * Q.sip_3d_2 - 0.1298567
    z += -0.005015005 * Q.n_charged_pt_above_1
    if Q.tau4 < 0.02949822:
        z += -16.0258 * Q.tau4 + 0.4727326
    if Q.n_neutral_had < 4.0:
        z += 0.05562923 * Q.n_neutral_had - 0.2225169
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += -0.002690159 * Q.lep_ptrel + 0.05048837
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.02447237 * Q.lep_ptrel - 0.6916898
    if Q.lep_ptrel >= 43.20788:
        z += -0.01934368 * Q.lep_ptrel + 1.201509
    if Q.D2 >= 3.597891:
        z += 0.04628823 * Q.D2 - 0.16654
    if Q.n_sdz_above_5 < 3.0:
        z += 0.006576692 * Q.n_sdz_above_5 - 0.01973008
    if Q.n_s3d_above_3 < 5.0:
        z += 0.009071273 * Q.n_s3d_above_3 - 0.04535636
    if Q.sd_nremoved < 4.0:
        z += 0.04088378 * Q.sd_nremoved - 0.1635351
    if Q.sd_rg < 0.3025746:
        z += -0.5208689 * Q.sd_rg + 0.2897174
    if 0.3025746 <= Q.sd_rg < 0.5562194:
        z += 0.1196584 * Q.sd_rg + 0.09591005
    if Q.sd_rg >= 0.5562194:
        z += 0.6405273 * Q.sd_rg - 0.1938073
    if Q.n_dr_0p4_up < 10.0:
        z += -0.01268385 * Q.n_dr_0p4_up + 0.1268385
    if Q.M3 >= 0.05919662:
        z += 5.139923 * Q.M3 - 0.3042661
    if Q.ecf_g41 < 6.216954e-05:
        z += -1862.686 * Q.ecf_g41 + 0.1158023
    if Q.tau2 < 0.04483276:
        z += 0.0095433 * Q.tau2 - 0.01798722
    if 0.04483276 <= Q.tau2 < 0.07264571:
        z += 0.6313378 * Q.tau2 - 0.04586398
    if Q.n_for_90pct >= 39.0:
        z += -0.01221071 * Q.n_for_90pct + 0.4762177
    if Q.n_lund_kt_above_5 >= 2.0:
        z += 0.04979105 * Q.n_lund_kt_above_5 - 0.09958211
    if Q.td0_38 < -0.02912079:
        z += -0.0527184 * Q.td0_38 - 0.001535202
    if Q.C2_b05 < 0.2326317:
        z += -1.207306 * Q.C2_b05 + 0.2808576
    if Q.e2 < 0.0318986:
        z += 1.562401 * Q.e2 - 0.04983842
    if Q.C2_b2 >= 0.221369:
        z += 0.09750523 * Q.C2_b2 - 0.02158464
    if Q.lund_max_lnkt < 4.046329:
        z += 0.1989018 * Q.lund_max_lnkt - 0.8048222
    if Q.pair_max_lnm2 >= 6.052324:
        z += 0.07472835 * Q.pair_max_lnm2 - 0.4522801
    if Q.mass_top5 >= 17.63354:
        z += -0.001706125 * Q.mass_top5 + 0.03008503
    if Q.sj2_mass1 >= 53.51926:
        z += -0.002086554 * Q.sj2_mass1 + 0.1116708
    if Q.pair_mean_lndelta < -1.352792 and Q.lep_ptrel > 12.15228:
        z += -0.01796822 * (-1.352792 - Q.pair_mean_lndelta) * (Q.lep_ptrel - 12.15228)
    if Q.n_s3d_above_10 < 8.0 and Q.n_real_top50 < 49.0:
        z += 0.001594408 * (8.0 - Q.n_s3d_above_10) * (49.0 - Q.n_real_top50)
    if Q.mass < 149.0507 and Q.lne_0 > 5.157617:
        z += 0.004282358 * (149.0507 - Q.mass) * (Q.lne_0 - 5.157617)
    if Q.mass_displaced3 > 3.208089 and Q.lam2 < 0.02148541:
        z += -0.1356184 * (Q.mass_displaced3 - 3.208089) * (0.02148541 - Q.lam2)
    if Q.n_s3d_above_10 < 8.0 and Q.sj4_pair_mass_max > 101.849:
        z += 0.0001694255 * (8.0 - Q.n_s3d_above_10) * (Q.sj4_pair_mass_max - 101.849)
    if Q.n_pairs_kt_above_3 < 47.0 and Q.lep_z < 0.221436:
        z += 0.004950793 * (47.0 - Q.n_pairs_kt_above_3) * (0.221436 - Q.lep_z)
    if Q.tau32_b2 < 0.4680886 and Q.ecf_g32 < 0.003965728:
        z += 88.96476 * (0.4680886 - Q.tau32_b2) * (0.003965728 - Q.ecf_g32)
    if Q.pair_mean_lndelta < -1.352792 and Q.lund1_lndelta > -0.6177752:
        z += 0.2021685 * (-1.352792 - Q.pair_mean_lndelta) * (Q.lund1_lndelta - -0.6177752)
    if Q.n_pairs_kt_above_3 < 47.0 and Q.z_displaced3 < 0.2601259:
        z += 0.02343112 * (47.0 - Q.n_pairs_kt_above_3) * (0.2601259 - Q.z_displaced3)
    if Q.n_s3d_above_10 < 8.0 and Q.lep_ptrel > 18.7678:
        z += 0.0005941553 * (8.0 - Q.n_s3d_above_10) * (Q.lep_ptrel - 18.7678)
    if Q.n_sd0_above_3 < 6.0 and Q.n_dr_0p4_up > 0.0:
        z += 0.001814852 * (6.0 - Q.n_sd0_above_3) * (Q.n_dr_0p4_up - 0.0)
    if Q.n_pairs_kt_above_1 < 146.0 and Q.D2 < 4.878859:
        z += -0.0003808979 * (146.0 - Q.n_pairs_kt_above_1) * (4.878859 - Q.D2)
    if Q.n_pairs_kt_above_1 < 146.0 and Q.n_lund_kt_above_5 > 1.0:
        z += 0.001097979 * (146.0 - Q.n_pairs_kt_above_1) * (Q.n_lund_kt_above_5 - 1.0)
    if Q.mass_displaced3 > 3.208089 and Q.ecf_g43 < 2.885768e-05:
        z += 215.788 * (Q.mass_displaced3 - 3.208089) * (2.885768e-05 - Q.ecf_g43)
    if Q.n_particles < 67.0 and Q.dr01 > 0.06433539:
        z += -0.006486141 * (67.0 - Q.n_particles) * (Q.dr01 - 0.06433539)
    if Q.sd_mass > 175.9333 and Q.lnptrel_77 > -18.42068:
        z += -0.0004662335 * (Q.sd_mass - 175.9333) * (Q.lnptrel_77 - -18.42068)
    if Q.n_particles < 67.0 and Q.n_lepton < 2.0:
        z += 0.003613295 * (67.0 - Q.n_particles) * (2.0 - Q.n_lepton)
    if Q.tau32_b2 < 0.4680886 and Q.eta_1 > 0.01268768:
        z += 0.8189809 * (0.4680886 - Q.tau32_b2) * (Q.eta_1 - 0.01268768)
    if Q.z_top15_slots > 0.9212656 and Q.dzerr_1 < 0.03430176:
        z += 33.67311 * (Q.z_top15_slots - 0.9212656) * (0.03430176 - Q.dzerr_1)
    if Q.sd_mass > 175.9333 and Q.charge_6 < 0.0:
        z += 0.0003412899 * (Q.sd_mass - 175.9333) * (0.0 - Q.charge_6)
    if Q.sd_mass > 78.4753 and Q.charge_25 < 1.0:
        z += 1.465607e-05 * (Q.sd_mass - 78.4753) * (1.0 - Q.charge_25)
    if Q.sd_nremoved < 4.0 and Q.eta_28 > -0.3166626:
        z += 0.001178933 * (4.0 - Q.sd_nremoved) * (Q.eta_28 - -0.3166626)
    if Q.lep_iso < 6.185635 and Q.iselectron_27 > 0.0:
        z += 0.02088817 * (6.185635 - Q.lep_iso) * (Q.iselectron_27 - 0.0)
    if Q.sd_nremoved < 4.0 and Q.eta_4 > 0.05764771:
        z += 0.03130918 * (4.0 - Q.sd_nremoved) * (Q.eta_4 - 0.05764771)
    if Q.sd_mass > 78.4753 and Q.tdz_74 < 0.0:
        z += 0.001172664 * (Q.sd_mass - 78.4753) * (0.0 - Q.tdz_74)
    if Q.n_lund > 11.0 and Q.phi_21 < 0.1278076:
        z += 0.00358732 * (Q.n_lund - 11.0) * (0.1278076 - Q.phi_21)
    if Q.n_pairs_kt_above_3 < 47.0 and Q.dr02 > 0.2159556:
        z += -0.02551562 * (47.0 - Q.n_pairs_kt_above_3) * (Q.dr02 - 0.2159556)
    if Q.n_s3d_above_3 < 5.0 and Q.C2_b2 < 0.1404188:
        z += -0.2461881 * (5.0 - Q.n_s3d_above_3) * (0.1404188 - Q.C2_b2)
    if Q.mass_over_sum_pt > 0.215596 and Q.dr_9 < 0.1211197:
        z += 1.194173 * (Q.mass_over_sum_pt - 0.215596) * (0.1211197 - Q.dr_9)
    if Q.n_pairs_kt_above_10 < 3.0 and Q.phi_29 < 0.0328064:
        z += 0.005014544 * (3.0 - Q.n_pairs_kt_above_10) * (0.0328064 - Q.phi_29)
    return z


def neuron_105(Q):
    z = 3.419518e-06
    return z


def neuron_106(Q):
    z = 4.732984e-05
    return z


def neuron_107(Q):
    z = -3.565694e-05
    return z


def neuron_108(Q):
    z = 5.938025e-06
    return z


def neuron_109(Q):
    z = -9.53234e-05
    return z


def neuron_110(Q):
    z = -0.0001436601
    return z


def neuron_111(Q):
    z = 0.0001309962
    return z


def neuron_112(Q):
    z = 0.0001557103
    return z


def neuron_113(Q):
    z = 7.091409e-05
    return z


def neuron_114(Q):
    z = -0.0001167055
    return z


def neuron_115(Q):
    z = 0.5586337
    if Q.M2_b2 < 0.05966366:
        z += -6.844577 * Q.M2_b2 + 0.4083725
    if Q.lep_z >= 0.03021637:
        z += 0.7689591 * Q.lep_z - 0.02323515
    if Q.mass_displaced3 < 39.09615:
        z += -0.01865414 * Q.mass_displaced3 + 0.7293051
    if Q.n_lund_kt_above_5 >= 1.0:
        z += -0.08337314 * Q.n_lund_kt_above_5 + 0.08337314
    if Q.tau2 < 0.09733903:
        z += -1.937226 * Q.tau2 + 0.1885677
    if Q.n_pairs_kt_above_1 < 101.0:
        z += 0.009703463 * Q.n_pairs_kt_above_1 - 0.9800498
    if Q.mass_top30 < 109.3251:
        z += -0.00373083 * Q.mass_top30 + 0.4078735
    if Q.sj4_pair_mass_max < 72.862:
        z += 0.002828949 * Q.sj4_pair_mass_max - 0.7073968
    if 72.862 <= Q.sj4_pair_mass_max < 107.8099:
        z += 0.01434346 * Q.sj4_pair_mass_max - 1.546367
    if Q.n_s3d_above_3 >= 4.0:
        z += 0.08693101 * Q.n_s3d_above_3 - 0.347724
    if Q.sj3_mass1 < 8.461136:
        z += 0.009050628 * Q.sj3_mass1 + 0.2759888
    if 8.461136 <= Q.sj3_mass1 < 36.36236:
        z += -0.01263627 * Q.sj3_mass1 + 0.4594846
    if Q.sip_3d_2 < 447.0873:
        z += 0.000113943 * Q.sip_3d_2 - 0.05094247
    if Q.sj3_pair_mass_min < 37.19471:
        z += 0.004443458 * Q.sj3_pair_mass_min - 0.7713654
    if 37.19471 <= Q.sj3_pair_mass_min < 59.49644:
        z += 0.01351059 * Q.sj3_pair_mass_min - 1.108615
    if 59.49644 <= Q.sj3_pair_mass_min < 80.02563:
        z += 0.01484631 * Q.sj3_pair_mass_min - 1.188085
    if Q.sj4_dr_min < 0.07985021:
        z += -0.3209293 * Q.sj4_dr_min + 0.1609018
    if 0.07985021 <= Q.sj4_dr_min < 0.1845735:
        z += -1.291742 * Q.sj4_dr_min + 0.2384214
    if Q.M2 < 0.120439:
        z += -8.389308 * Q.M2 + 1.0104
    if Q.jet_abs_eta >= 1.323111:
        z += 0.2368751 * Q.jet_abs_eta - 0.3134122
    if Q.mass_top50 < 94.51361:
        z += -0.000751581 * Q.mass_top50 + 0.472802
    if 94.51361 <= Q.mass_top50 < 125.4927:
        z += -0.008031679 * Q.mass_top50 + 1.16087
    if 125.4927 <= Q.mass_top50 < 129.5874:
        z += -0.03735332 * Q.mass_top50 + 4.840521
    if Q.mass < 120.653:
        z += -0.03016923 * Q.mass + 3.992503
    if 120.653 <= Q.mass < 164.4374:
        z += -0.001175995 * Q.mass + 0.494381
    if 164.4374 <= Q.mass < 182.8592:
        z += -0.01633954 * Q.mass + 2.987835
    if Q.lam1 >= 0.05242126:
        z += -0.912349 * Q.lam1 + 0.04782649
    if Q.ecf_g43 < 2.852309e-06:
        z += 57662.61 * Q.ecf_g43 - 0.1644716
    if Q.lep_iso < 0.4381892:
        z += -0.135814 * Q.lep_iso + 0.05951225
    if Q.mass_2charged < 1.398635:
        z += 0.1098818 * Q.mass_2charged - 0.1184802
    if 1.398635 <= Q.mass_2charged < 6.767937:
        z += -0.06138219 * Q.mass_2charged + 0.1210555
    if 6.767937 <= Q.mass_2charged < 28.89659:
        z += 0.0133029 * Q.mass_2charged - 0.3844085
    if Q.lep_ptrel >= 43.20788:
        z += -0.02304598 * Q.lep_ptrel + 0.9957681
    if Q.sj3_pair_mass_max < 109.8208:
        z += 0.011546 * Q.sj3_pair_mass_max - 1.267991
    if Q.sj3_dr_min < 0.1991803:
        z += 0.1917425 * Q.sj3_dr_min - 0.2241585
    if 0.1991803 <= Q.sj3_dr_min < 0.3190414:
        z += 1.551522 * Q.sj3_dr_min - 0.4949997
    if Q.sum_pt_top5 < 303.75:
        z += -0.002520822 * Q.sum_pt_top5 + 0.7656997
    if Q.tau4 >= 0.04996:
        z += -9.686607 * Q.tau4 + 0.4839429
    if Q.e3_b05 < 0.0131933:
        z += -100.8075 * Q.e3_b05 + 1.329983
    if Q.tau1 >= 0.1246227:
        z += 3.537878 * Q.tau1 - 0.4409
    if Q.sj3_pairmax_over_m < 0.8207647:
        z += -0.7303328 * Q.sj3_pairmax_over_m + 0.5994314
    if Q.psi_0p1 < 0.006220408:
        z += 13.19336 * Q.psi_0p1 - 0.08206808
    if Q.max_abs_d0 < 5.8125:
        z += -0.005762965 * Q.max_abs_d0 + 0.03349724
    if Q.n_sdz_above_5 < 2.0:
        z += 0.02219597 * Q.n_sdz_above_5 - 0.04439193
    if Q.dr_0 < 0.1117947:
        z += 0.3495034 * Q.dr_0 - 0.03907262
    if Q.dr_0 >= 0.2254414:
        z += -0.1729557 * Q.dr_0 + 0.03899138
    if Q.N2_b2 < 0.2099278:
        z += -2.688714 * Q.N2_b2 + 0.5644358
    if Q.sj3_mass2 < 4.326415:
        z += 0.0435935 * Q.sj3_mass2 - 0.1886035
    if Q.n_sd0_above_3 >= 3.0:
        z += -0.04025544 * Q.n_sd0_above_3 + 0.1207663
    if Q.D3_b2 < 0.0225905:
        z += -2.644113 * Q.D3_b2 + 0.05973183
    if Q.sum_charge < -2.0:
        z += -0.02240876 * Q.sum_charge - 0.04481752
    if Q.mass_charged < 65.0404:
        z += -0.001355641 * Q.mass_charged + 0.08817145
    if Q.pair_mean_lndelta >= -3.421602:
        z += -0.4622787 * Q.pair_mean_lndelta - 1.581734
    if Q.sj3_dr23 < 0.530706:
        z += 0.007635183 * Q.sj3_dr23 - 0.004052037
    if Q.mass_2photon < 5.763861:
        z += -0.01543014 * Q.mass_2photon + 0.08893718
    if Q.pair_max_lnkt < 2.911067:
        z += -0.2344651 * Q.pair_max_lnkt + 0.6825437
    if Q.M3 < 0.05919662:
        z += 9.256211 * Q.M3 - 0.5479364
    if Q.lne_4 < 3.692778:
        z += -0.2002485 * Q.lne_4 + 0.7394733
    if Q.z_top20_slots >= 0.9417195:
        z += -3.854054 * Q.z_top20_slots + 3.629438
    if Q.e3 < 0.0006496195:
        z += -145.0548 * Q.e3 + 0.09423043
    if Q.phi_26 >= 0.1617432:
        z += -0.07414152 * Q.phi_26 + 0.01199188
    if Q.tau32_b2 >= 0.6908801:
        z += 1.517275 * Q.tau32_b2 - 1.048255
    if Q.mass_top5 >= 37.66803:
        z += 0.001823791 * Q.mass_top5 - 0.06869863
    if Q.LHA < 0.4888886:
        z += 3.784638 * Q.LHA - 1.850267
    if Q.n_dr_0_0p05 < 12.0:
        z += 0.008884479 * Q.n_dr_0_0p05 - 0.1066138
    if Q.m012 >= 11.86162:
        z += -0.002836713 * Q.m012 + 0.033648
    if Q.mass_displaced3 < 39.09615 and Q.tau43 < 0.8709334:
        z += -0.0276891 * (39.09615 - Q.mass_displaced3) * (0.8709334 - Q.tau43)
    if Q.mass_displaced3 < 39.09615 and Q.sum_z_dr2_top20 > 0.0399789:
        z += -0.0414023 * (39.09615 - Q.mass_displaced3) * (Q.sum_z_dr2_top20 - 0.0399789)
    if Q.mass_displaced3 < 39.09615 and Q.z_neutral_had < 0.3788785:
        z += -0.07073271 * (39.09615 - Q.mass_displaced3) * (0.3788785 - Q.z_neutral_had)
    if Q.mass_displaced3 < 39.09615 and Q.M3_b2 < 0.01013989:
        z += 0.6424735 * (39.09615 - Q.mass_displaced3) * (0.01013989 - Q.M3_b2)
    if Q.M2_b2 < 0.05966366 and Q.mass_charged > 37.42054:
        z += -0.1136074 * (0.05966366 - Q.M2_b2) * (Q.mass_charged - 37.42054)
    if Q.n_lund_kt_above_5 > 1.0 and Q.tau54 < 0.8953628:
        z += -0.3064434 * (Q.n_lund_kt_above_5 - 1.0) * (0.8953628 - Q.tau54)
    if Q.lep_z > 0.03021637 and Q.n_lepton < 2.0:
        z += -1.016686 * (Q.lep_z - 0.03021637) * (2.0 - Q.n_lepton)
    if Q.sj4_pair_mass_max < 107.8099 and Q.sum_pt_top20 < 515.5711:
        z += 4.046265e-06 * (107.8099 - Q.sj4_pair_mass_max) * (515.5711 - Q.sum_pt_top20)
    if Q.tau2 < 0.09733903 and Q.lep_ptrel < 27.3236:
        z += 0.09377258 * (0.09733903 - Q.tau2) * (27.3236 - Q.lep_ptrel)
    if Q.n_lund_kt_above_5 > 1.0 and Q.z_displaced5 > 0.1046203:
        z += 0.3693249 * (Q.n_lund_kt_above_5 - 1.0) * (Q.z_displaced5 - 0.1046203)
    if Q.mass_top30 < 109.3251 and Q.lnptrel_12 > -5.283873:
        z += 0.002665459 * (109.3251 - Q.mass_top30) * (Q.lnptrel_12 - -5.283873)
    if Q.n_s3d_above_3 > 4.0 and Q.sip_3d_3 < 175.9957:
        z += 0.0005783388 * (Q.n_s3d_above_3 - 4.0) * (175.9957 - Q.sip_3d_3)
    if Q.mass_displaced3 < 39.09615 and Q.e3_b2 < 0.0007909605:
        z += -21.08589 * (39.09615 - Q.mass_displaced3) * (0.0007909605 - Q.e3_b2)
    if Q.mass < 164.4374 and Q.z_top20_slots > 0.7122459:
        z += -0.04734022 * (164.4374 - Q.mass) * (Q.z_top20_slots - 0.7122459)
    if Q.mass < 164.4374 and Q.sj4_pair_mass_min > 13.57513:
        z += -3.87716e-05 * (164.4374 - Q.mass) * (Q.sj4_pair_mass_min - 13.57513)
    if Q.mass_top50 < 125.4927 and Q.n_lepton < 1.0:
        z += 0.006672728 * (125.4927 - Q.mass_top50) * (1.0 - Q.n_lepton)
    if Q.mass_2charged < 6.767937 and Q.n_real_top15 < 15.0:
        z += -0.01025208 * (6.767937 - Q.mass_2charged) * (15.0 - Q.n_real_top15)
    if Q.n_s3d_above_3 > 4.0 and Q.z_charged_had > 0.3603262:
        z += -0.1891452 * (Q.n_s3d_above_3 - 4.0) * (Q.z_charged_had - 0.3603262)
    if Q.sj3_mass1 < 36.36236 and Q.sj3_mass2 > 9.068761:
        z += -0.0005004627 * (36.36236 - Q.sj3_mass1) * (Q.sj3_mass2 - 9.068761)
    if Q.n_lund_kt_above_5 > 1.0 and Q.n_electron > 0.0:
        z += 0.008772759 * (Q.n_lund_kt_above_5 - 1.0) * (Q.n_electron - 0.0)
    if Q.sj3_pair_mass_min < 37.19471 and Q.sj3_z2 < 0.3118983:
        z += 0.04654004 * (37.19471 - Q.sj3_pair_mass_min) * (0.3118983 - Q.sj3_z2)
    if Q.M2_b2 < 0.05966366 and Q.d0err_52 > 0.0:
        z += -18.19975 * (0.05966366 - Q.M2_b2) * (Q.d0err_52 - 0.0)
    if Q.sj3_pairmax_over_m < 0.8207647 and Q.isphoton_53 > 0.0:
        z += 1.137661 * (0.8207647 - Q.sj3_pairmax_over_m) * (Q.isphoton_53 - 0.0)
    if Q.n_lund_kt_above_5 > 1.0 and Q.sj4_zsoft < 0.06029776:
        z += 0.935477 * (Q.n_lund_kt_above_5 - 1.0) * (0.06029776 - Q.sj4_zsoft)
    if Q.mass_displaced3 < 39.09615 and Q.isphoton_12 > 0.0:
        z += -0.001162846 * (39.09615 - Q.mass_displaced3) * (Q.isphoton_12 - 0.0)
    if Q.n_s3d_above_3 > 4.0 and Q.tdz_44 > 0.01757784:
        z += -0.01292237 * (Q.n_s3d_above_3 - 4.0) * (Q.tdz_44 - 0.01757784)
    if Q.n_lund_kt_above_5 > 1.0 and Q.iselectron_34 > 0.0:
        z += 0.06648717 * (Q.n_lund_kt_above_5 - 1.0) * (Q.iselectron_34 - 0.0)
    if Q.sj4_pair_mass_max < 107.8099 and Q.td0_9 > -0.01537965:
        z += -0.002460497 * (107.8099 - Q.sj4_pair_mass_max) * (Q.td0_9 - -0.01537965)
    if Q.sj4_pair_mass_max < 107.8099 and Q.tdz_25 < 0.0:
        z += -0.0007397701 * (107.8099 - Q.sj4_pair_mass_max) * (0.0 - Q.tdz_25)
    if Q.mass_2charged < 28.89659 and Q.phi_31 > 0.326416:
        z += 0.002634509 * (28.89659 - Q.mass_2charged) * (Q.phi_31 - 0.326416)
    if Q.sj4_pair_mass_max < 107.8099 and Q.charge_43 > -1.0:
        z += 0.0001107032 * (107.8099 - Q.sj4_pair_mass_max) * (Q.charge_43 - -1.0)
    if Q.N2_b2 < 0.2099278 and Q.dzerr_43 < 0.05511475:
        z += 19.50066 * (0.2099278 - Q.N2_b2) * (0.05511475 - Q.dzerr_43)
    if Q.tau2 < 0.09733903 and Q.isphoton_41 < 1.0:
        z += 3.159208 * (0.09733903 - Q.tau2) * (1.0 - Q.isphoton_41)
    return z


def neuron_116(Q):
    z = -5.958923e-05
    return z


def neuron_117(Q):
    z = -0.0001184036
    return z


def neuron_118(Q):
    z = 7.045374e-05
    return z


def neuron_119(Q):
    z = -0.000129661
    return z


def neuron_120(Q):
    z = 1.931404
    if Q.lep_ptrel < 3.53503:
        z += 0.2009397 * Q.lep_ptrel - 1.089864
    if 3.53503 <= Q.lep_ptrel < 6.983043:
        z += 0.0440438 * Q.lep_ptrel - 0.5352326
    if 6.983043 <= Q.lep_ptrel < 12.15228:
        z += 0.03488798 * Q.lep_ptrel - 0.4712971
    if 12.15228 <= Q.lep_ptrel < 18.7678:
        z += -0.009155814 * Q.lep_ptrel + 0.06393544
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += 0.002073807 * Q.lep_ptrel - 0.1468198
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.04562107 * Q.lep_ptrel - 1.336688
    if Q.lep_ptrel >= 43.20788:
        z += 0.05033603 * Q.lep_ptrel - 1.540411
    if Q.n_pairs_kt_above_3 < 62.0:
        z += 0.003816548 * Q.n_pairs_kt_above_3 - 0.236626
    if Q.lep_z < 0.5187302:
        z += 1.497268 * Q.lep_z - 0.7766781
    if Q.tau32 < 0.6800935:
        z += -0.6947629 * Q.tau32 + 0.4725038
    if Q.n_sd0_above_5 < 3.0:
        z += -0.001085254 * Q.n_sd0_above_5 + 0.003255762
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.005144097 * Q.n_pairs_kt_above_1 - 0.6313742
    if 80.0 <= Q.n_pairs_kt_above_1 < 195.0:
        z += 0.001911709 * Q.n_pairs_kt_above_1 - 0.3727832
    if 48.96085 <= Q.mass_top10 < 103.4976:
        z += -0.0044497 * Q.mass_top10 + 0.2178611
    if Q.mass_top10 >= 103.4976:
        z += -0.01210015 * Q.mass_top10 + 1.009664
    z += 455.6021 * Q.ecf_g42
    if Q.pair_mean_lndelta >= -1.523797:
        z += -0.5522379 * Q.pair_mean_lndelta - 0.8414982
    if Q.mass_top15 >= 104.4937:
        z += -0.01139447 * Q.mass_top15 + 1.190651
    if Q.N2_b05 < 0.4174214:
        z += 4.279158 * Q.N2_b05 - 1.786212
    if Q.n_electron < 2.0:
        z += -0.09820902 * Q.n_electron + 0.196418
    if Q.pair_mean_lnkt >= 0.5586581:
        z += 0.2269675 * Q.pair_mean_lnkt - 0.1267972
    if Q.e4_b05 >= 1.4104e-05:
        z += 1931.709 * Q.e4_b05 - 0.02724482
    if Q.max_abs_d0 < 5.8125:
        z += 0.001736501 * Q.max_abs_d0 - 0.01009341
    if Q.z_charged_had < 0.3191471:
        z += 1.327821 * Q.z_charged_had - 0.4237701
    if Q.lep_iso < 0.02781886:
        z += -5.31649 * Q.lep_iso + 0.5607763
    if 0.02781886 <= Q.lep_iso < 2.907433:
        z += -0.1433795 * Q.lep_iso + 0.4168663
    if Q.lund3_lndelta >= -2.817283:
        z += -0.1006848 * Q.lund3_lndelta - 0.2836574
    if Q.z_neutral >= 0.6734757:
        z += 2.009658 * Q.z_neutral - 1.353456
    if Q.mass < 90.08945:
        z += 0.009121854 * Q.mass - 1.546319
    if 90.08945 <= Q.mass < 105.7234:
        z += 0.03755792 * Q.mass - 4.108109
    if 105.7234 <= Q.mass < 114.0172:
        z += 0.01656146 * Q.mass - 1.888292
    if Q.z_displaced3 >= 0.2092108:
        z += -0.4854292 * Q.z_displaced3 + 0.101557
    if Q.n_dr_0p1_0p2 < 5.0:
        z += 0.01904862 * Q.n_dr_0p1_0p2 - 0.09524308
    if Q.n_charged < 25.0:
        z += -0.01690497 * Q.n_charged + 0.4226244
    if Q.dr_min_012 < 0.01394245:
        z += 3.098557 * Q.dr_min_012 - 0.04320148
    if Q.mass_charged < 68.20711:
        z += 0.004121652 * Q.mass_charged - 0.281126
    z += -0.1059322 * Q.z_displaced5
    if Q.n_sd0_above_2 < 4.0:
        z += 0.05004624 * Q.n_sd0_above_2 - 0.200185
    if Q.sd_rg >= 0.5562194:
        z += -1.097949 * Q.sd_rg + 0.6107003
    if Q.e3 >= 0.00228569:
        z += 17.66248 * Q.e3 - 0.04037096
    if 109.3251 <= Q.mass_top30 < 162.7874:
        z += -0.01144495 * Q.mass_top30 + 1.251221
    if Q.mass_top30 >= 162.7874:
        z += -0.01653306 * Q.mass_top30 + 2.079501
    if Q.D3_b05 < 0.3052386:
        z += -0.2399084 * Q.D3_b05 - 0.2453945
    if 0.3052386 <= Q.D3_b05 < 0.8860453:
        z += 0.5485883 * Q.D3_b05 - 0.4860741
    if 135.5368 <= Q.mass_top50 < 161.1264:
        z += 0.002383857 * Q.mass_top50 - 0.3231004
    if Q.mass_top50 >= 161.1264:
        z += 0.01231143 * Q.mass_top50 - 1.922695
    if Q.e2_b05 < 0.08995834:
        z += 6.077832 * Q.e2_b05 - 0.5467517
    if Q.pt_entropy < 1.938659:
        z += -0.8205318 * Q.pt_entropy + 1.590731
    if Q.sj3_z3 < 0.1719087:
        z += 1.554528 * Q.sj3_z3 - 0.267237
    if Q.n_neutral_had < 3.0:
        z += -0.09545511 * Q.n_neutral_had + 0.2863653
    if Q.sip_3d_2 < 226.3008:
        z += 2.42478e-05 * Q.sip_3d_2 - 0.005487295
    if Q.n_dr_0p4_up < 4.0:
        z += -0.001087469 * Q.n_dr_0p4_up + 0.004349875
    if Q.lnptrel_3 < -3.360241:
        z += 0.2042831 * Q.lnptrel_3 + 0.6864404
    if Q.n_pt_above_5 < 16.0:
        z += 0.004315604 * Q.n_pt_above_5 - 0.06904966
    if Q.M3_b05 >= 0.08175231:
        z += -7.468635 * Q.M3_b05 + 0.6105782
    if Q.sip_3d_3 < 577.991:
        z += 0.0004083445 * Q.sip_3d_3 - 0.2360194
    if Q.M2_b05 < 0.1258758:
        z += 4.793348 * Q.M2_b05 - 0.6033666
    if Q.sj3_mass2 < 1.824785:
        z += -0.03879704 * Q.sj3_mass2 + 0.07079626
    if Q.dr_4 < 0.1132155:
        z += 0.01478812 * Q.dr_4 - 0.001674245
    if Q.n_s3d_above_10 < 5.0:
        z += -0.02021023 * Q.n_s3d_above_10 + 0.1010511
    if Q.lep_ptrel < 3.53503 and Q.n_s3d_above_3 < 10.0:
        z += -0.04616613 * (3.53503 - Q.lep_ptrel) * (10.0 - Q.n_s3d_above_3)
    if Q.lep_z < 0.5187302 and Q.mass_neutral < 69.1565:
        z += 0.00150489 * (0.5187302 - Q.lep_z) * (69.1565 - Q.mass_neutral)
    if Q.lep_z < 0.5187302 and Q.mass_displaced3 > 0.0:
        z += 0.02616787 * (0.5187302 - Q.lep_z) * (Q.mass_displaced3 - 0.0)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.sip_3d_3 < 4.606241:
        z += -0.0002349147 * (62.0 - Q.n_pairs_kt_above_3) * (4.606241 - Q.sip_3d_3)
    if Q.lep_ptrel < 3.53503 and Q.sip_3d_3 < 577.991:
        z += 0.0003878179 * (3.53503 - Q.lep_ptrel) * (577.991 - Q.sip_3d_3)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.n_lund > 5.0:
        z += -0.0007447959 * (62.0 - Q.n_pairs_kt_above_3) * (Q.n_lund - 5.0)
    if Q.lep_ptrel < 3.53503 and Q.n_photon > 5.0:
        z += 0.004300655 * (3.53503 - Q.lep_ptrel) * (Q.n_photon - 5.0)
    if Q.lep_z < 0.5187302 and Q.mass_top10 > 48.96085:
        z += 0.01126515 * (0.5187302 - Q.lep_z) * (Q.mass_top10 - 48.96085)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.z_displaced3 < 0.2601259:
        z += 0.03880277 * (62.0 - Q.n_pairs_kt_above_3) * (0.2601259 - Q.z_displaced3)
    if Q.lep_ptrel > 27.3236 and Q.mratio_min_012 < 0.1991254:
        z += -0.1082158 * (Q.lep_ptrel - 27.3236) * (0.1991254 - Q.mratio_min_012)
    if Q.lep_z < 0.5187302 and Q.lep_dr > 0.114659:
        z += 0.4104457 * (0.5187302 - Q.lep_z) * (Q.lep_dr - 0.114659)
    if Q.lep_ptrel < 12.15228 and Q.planar_flow < 0.7095008:
        z += 0.02279625 * (12.15228 - Q.lep_ptrel) * (0.7095008 - Q.planar_flow)
    if Q.mass_top10 > 103.4976 and Q.log_sum_pt > 6.426552:
        z += 0.02556816 * (Q.mass_top10 - 103.4976) * (Q.log_sum_pt - 6.426552)
    if Q.e4_b05 > 1.4104e-05 and Q.sum_pt > 550.4644:
        z += 11.05343 * (Q.e4_b05 - 1.4104e-05) * (Q.sum_pt - 550.4644)
    if Q.lep_ptrel > 18.7678 and Q.lep_iso < 0.4381892:
        z += -0.1146231 * (Q.lep_ptrel - 18.7678) * (0.4381892 - Q.lep_iso)
    if Q.lep_ptrel < 3.53503 and Q.lep_iso < 0.02781886:
        z += -6.026032 * (3.53503 - Q.lep_ptrel) * (0.02781886 - Q.lep_iso)
    if Q.N2_b05 < 0.4174214 and Q.sj3_dr_max < 0.8930677:
        z += 1.473888 * (0.4174214 - Q.N2_b05) * (0.8930677 - Q.sj3_dr_max)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.z_neutral_had > 0.02425618:
        z += -0.02109329 * (62.0 - Q.n_pairs_kt_above_3) * (Q.z_neutral_had - 0.02425618)
    if Q.n_dr_0p1_0p2 < 5.0 and Q.eta_2 < -0.217041:
        z += -0.4060793 * (5.0 - Q.n_dr_0p1_0p2) * (-0.217041 - Q.eta_2)
    if Q.lund3_lndelta > -2.817283 and Q.sj3_dr13 > 0.577548:
        z += -0.1290122 * (Q.lund3_lndelta - -2.817283) * (Q.sj3_dr13 - 0.577548)
    if Q.lep_ptrel < 12.15228 and Q.sj3_z3 < 0.1889947:
        z += 0.1119502 * (12.15228 - Q.lep_ptrel) * (0.1889947 - Q.sj3_z3)
    if Q.mass < 90.08945 and Q.iselectron_1 > 0.0:
        z += -0.01481147 * (90.08945 - Q.mass) * (Q.iselectron_1 - 0.0)
    if Q.max_abs_d0 < 5.8125 and Q.ismuon_0 < 1.0:
        z += 0.02168467 * (5.8125 - Q.max_abs_d0) * (1.0 - Q.ismuon_0)
    if Q.n_sd0_above_2 < 4.0 and Q.dr_19 < 0.3967994:
        z += 0.006111266 * (4.0 - Q.n_sd0_above_2) * (0.3967994 - Q.dr_19)
    if Q.lep_iso < 2.907433 and Q.iselectron_12 > 0.0:
        z += -0.05270077 * (2.907433 - Q.lep_iso) * (Q.iselectron_12 - 0.0)
    if Q.mass < 105.7234 and Q.eta_0 > 0.04705811:
        z += -0.1001804 * (105.7234 - Q.mass) * (Q.eta_0 - 0.04705811)
    if Q.sj3_z3 < 0.1719087 and Q.ismuon_11 > 0.0:
        z += -0.4873475 * (0.1719087 - Q.sj3_z3) * (Q.ismuon_11 - 0.0)
    if Q.sj3_z3 < 0.1719087 and Q.isphoton_42 < 1.0:
        z += 0.299937 * (0.1719087 - Q.sj3_z3) * (1.0 - Q.isphoton_42)
    if Q.dr_min_012 < 0.01394245 and Q.ischhad_26 > 0.0:
        z += -1.013051 * (0.01394245 - Q.dr_min_012) * (Q.ischhad_26 - 0.0)
    if Q.lep_ptrel < 12.15228 and Q.phi_22 < -0.2314453:
        z += 0.01523623 * (12.15228 - Q.lep_ptrel) * (-0.2314453 - Q.phi_22)
    if Q.mass_top10 > 48.96085 and Q.eta_7 > -0.1883545:
        z += 0.002270394 * (Q.mass_top10 - 48.96085) * (Q.eta_7 - -0.1883545)
    if Q.lep_ptrel < 12.15228 and Q.dr_16 < 0.5715866:
        z += -0.008332967 * (12.15228 - Q.lep_ptrel) * (0.5715866 - Q.dr_16)
    if Q.lep_ptrel > 6.983043 and Q.jet_charge_k03 > 0.7410651:
        z += -0.01407971 * (Q.lep_ptrel - 6.983043) * (Q.jet_charge_k03 - 0.7410651)
    if Q.mass_top10 > 48.96085 and Q.iselectron_14 > 0.0:
        z += 0.00163582 * (Q.mass_top10 - 48.96085) * (Q.iselectron_14 - 0.0)
    if Q.pair_mean_lnkt > 0.5586581 and Q.lund3_lnz > -5.757953:
        z += -0.0001400392 * (Q.pair_mean_lnkt - 0.5586581) * (Q.lund3_lnz - -5.757953)
    if Q.max_abs_d0 < 5.8125 and Q.d0err_12 > 0.01499939:
        z += 0.05461959 * (5.8125 - Q.max_abs_d0) * (Q.d0err_12 - 0.01499939)
    if Q.M3_b05 > 0.08175231 and Q.td0_13 > 0.01621867:
        z += 8.250999 * (Q.M3_b05 - 0.08175231) * (Q.td0_13 - 0.01621867)
    if Q.sj3_z3 < 0.1719087 and Q.sj2_mass2 < 24.55881:
        z += 0.02060498 * (0.1719087 - Q.sj3_z3) * (24.55881 - Q.sj2_mass2)
    return z


def neuron_121(Q):
    z = 4.99221e-05
    return z


def neuron_122(Q):
    z = -4.924992e-05
    return z


def neuron_123(Q):
    z = 0.01948761
    return z


def neuron_124(Q):
    z = 0.04815712
    if Q.lep_z < 0.221436:
        z += -0.9677717 * Q.lep_z + 0.2388592
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -0.2077439 * Q.lep_z + 0.0705617
    if Q.n_pairs_kt_above_1 < 58.0:
        z += 0.00919579 * Q.n_pairs_kt_above_1 - 1.128534
    if 58.0 <= Q.n_pairs_kt_above_1 < 366.0:
        z += 0.0014474 * Q.n_pairs_kt_above_1 - 0.6791277
    if 366.0 <= Q.n_pairs_kt_above_1 < 702.0:
        z += 0.0004445814 * Q.n_pairs_kt_above_1 - 0.3120961
    if Q.n_s3d_above_3 < 1.0:
        z += 0.08168302 * Q.n_s3d_above_3 + 0.05674368
    if 1.0 <= Q.n_s3d_above_3 < 2.0:
        z += -0.1384267 * Q.n_s3d_above_3 + 0.2768534
    if Q.n_s3d_above_3 >= 3.0:
        z += 0.04406479 * Q.n_s3d_above_3 - 0.1321944
    if Q.mass < 95.14961:
        z += -0.003502618 * Q.mass - 0.006122372
    if 95.14961 <= Q.mass < 100.4835:
        z += 0.02262744 * Q.mass - 2.492387
    if 100.4835 <= Q.mass < 117.4867:
        z += 0.01286249 * Q.mass - 1.511171
    if Q.mass >= 182.8592:
        z += -0.002135579 * Q.mass + 0.3905104
    if Q.lep_ptrel < 43.20788:
        z += -0.009420199 * Q.lep_ptrel + 0.4070269
    if Q.n_dr_0p4_up < 15.0:
        z += -0.01107124 * Q.n_dr_0p4_up + 0.1660686
    if Q.lep_iso < 0.4381892:
        z += 0.9841672 * Q.lep_iso - 0.7107185
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += 0.3024846 * Q.lep_iso - 0.4120126
    if Q.n_lepton < 1.0:
        z += -0.6869392 * Q.n_lepton + 0.6869392
    if Q.mass_top40 < 70.88236:
        z += 0.01164042 * Q.mass_top40 - 1.130539
    if 70.88236 <= Q.mass_top40 < 97.12186:
        z += 0.00541191 * Q.mass_top40 - 0.6890477
    if Q.mass_top40 >= 97.12186:
        z += -0.006228511 * Q.mass_top40 + 0.4414916
    if Q.z_displaced3 < 0.03624058:
        z += 7.861836 * Q.z_displaced3 - 0.3732754
    if 0.03624058 <= Q.z_displaced3 < 0.08304558:
        z += 1.887789 * Q.z_displaced3 - 0.1567725
    if Q.sj3_pair_mass_max < 63.7508:
        z += -0.002538672 * Q.sj3_pair_mass_max + 0.2273639
    if 63.7508 <= Q.sj3_pair_mass_max < 73.24742:
        z += 0.006766005 * Q.sj3_pair_mass_max - 0.3658167
    if 73.24742 <= Q.sj3_pair_mass_max < 93.87663:
        z += -0.006290872 * Q.sj3_pair_mass_max + 0.5905658
    if Q.n_s3d_above_10 < 8.0:
        z += -0.05320953 * Q.n_s3d_above_10 + 0.4256763
    if Q.pair_max_lnm2 >= 6.520267:
        z += 0.03849594 * Q.pair_max_lnm2 - 0.2510038
    if Q.sip_3d_3 < 577.991:
        z += -0.0002448322 * Q.sip_3d_3 + 0.1415108
    z += -1.988855 * Q.M3
    if Q.sj2_dr < 0.5076533:
        z += 0.7133689 * Q.sj2_dr - 0.3621441
    if 42.31629 <= Q.sd_mass < 78.4753:
        z += 0.001702793 * Q.sd_mass - 0.07205587
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += -0.03580743 * Q.sd_mass + 2.87157
    if 88.81751 <= Q.sd_mass < 94.55722:
        z += 0.001497037 * Q.sd_mass - 0.4417199
    if 94.55722 <= Q.sd_mass < 106.7501:
        z += 0.02848501 * Q.sd_mass - 2.993627
    if 106.7501 <= Q.sd_mass < 127.8042:
        z += 0.01515798 * Q.sd_mass - 1.570966
    if 127.8042 <= Q.sd_mass < 154.5947:
        z += -0.02370474 * Q.sd_mass + 3.395855
    if Q.sd_mass >= 154.5947:
        z += 0.003682813 * Q.sd_mass - 0.838115
    if Q.mass_over_sum_pt_sq < 0.0729277:
        z += -5.500487 * Q.mass_over_sum_pt_sq + 0.4011378
    if Q.z_displaced5 >= 0.2801368:
        z += -0.4458978 * Q.z_displaced5 + 0.1249124
    if Q.ecf_g31 < 0.01365334:
        z += 2.569021 * Q.ecf_g31 - 0.03507572
    if Q.psi_0p1 < 0.006220408:
        z += 10.4927 * Q.psi_0p1 - 0.06526888
    if Q.sj3_mass2 < 1.824785:
        z += 0.1565125 * Q.sj3_mass2 - 0.2856017
    if Q.sum_z_dr2_top3 < 0.007018285:
        z += -21.1088 * Q.sum_z_dr2_top3 + 0.1481476
    if Q.LHA < 0.4008517:
        z += 1.370872 * Q.LHA - 0.5495164
    if Q.mass_displaced3 < 1.777286:
        z += 0.02385135 * Q.mass_displaced3 - 0.1237296
    if 1.777286 <= Q.mass_displaced3 < 4.41005:
        z += 0.03089488 * Q.mass_displaced3 - 0.136248
    if Q.mass_top15 < 45.8749:
        z += -0.00339291 * Q.mass_top15 + 0.1556494
    if Q.N3_b05 >= 0.4404318:
        z += -0.00475485 * Q.N3_b05 + 0.002094187
    if Q.lep_dr >= 0.4191372:
        z += 1.321299 * Q.lep_dr - 0.5538055
    if Q.n_dr_0p1_0p2 < 6.0:
        z += 0.02619977 * Q.n_dr_0p1_0p2 - 0.1571986
    if Q.ecf_g41 < 0.0002944259:
        z += -2261.809 * Q.ecf_g41 + 0.6659349
    if Q.lam1 < 0.06041764:
        z += 3.425455 * Q.lam1 - 0.2069579
    if Q.tdz_2 >= -0.02221619:
        z += -0.08431645 * Q.tdz_2 - 0.001873191
    if Q.td0_17 < -0.05792002:
        z += -0.0617508 * Q.td0_17 - 0.003576608
    if Q.z_dr_0p1_0p2 < 0.2428609:
        z += -0.1491501 * Q.z_dr_0p1_0p2 + 0.03622274
    if Q.tau32 < 0.3164143:
        z += -0.3764663 * Q.tau32 + 0.1191193
    if Q.n_charged_had < 25.0:
        z += -0.0004777896 * Q.n_charged_had + 0.01194474
    if Q.M3_b05 < 0.09942631:
        z += 6.974358 * Q.M3_b05 - 0.6934346
    if Q.n_lund >= 7.0:
        z += -0.009668479 * Q.n_lund + 0.06767935
    if Q.n_charged_pt_above_1 < 15.0:
        z += -0.008685963 * Q.n_charged_pt_above_1 + 0.1302894
    if Q.lep_ptrel < 43.20788 and Q.sj3_pair_mass_max < 142.9952:
        z += 1.686996e-05 * (43.20788 - Q.lep_ptrel) * (142.9952 - Q.sj3_pair_mass_max)
    if Q.n_s3d_above_3 > 3.0 and Q.M2_b2 < 0.06762785:
        z += -1.208086 * (Q.n_s3d_above_3 - 3.0) * (0.06762785 - Q.M2_b2)
    if Q.n_s3d_above_3 > 3.0 and Q.max_abs_d0 < 10.52344:
        z += -0.009131013 * (Q.n_s3d_above_3 - 3.0) * (10.52344 - Q.max_abs_d0)
    if Q.n_pairs_kt_above_1 < 366.0 and Q.D2_b2 < 15.17086:
        z += -1.672492e-05 * (366.0 - Q.n_pairs_kt_above_1) * (15.17086 - Q.D2_b2)
    if Q.lep_z < 0.3396572 and Q.z_displaced5 > 0.1046203:
        z += 0.5031575 * (0.3396572 - Q.lep_z) * (Q.z_displaced5 - 0.1046203)
    if Q.mass_top40 > 70.88236 and Q.lnerel_17 > -6.957861:
        z += 0.002254335 * (Q.mass_top40 - 70.88236) * (Q.lnerel_17 - -6.957861)
    if Q.mass_top40 > 70.88236 and Q.ecf_g42 < 4.726476e-05:
        z += -75.92397 * (Q.mass_top40 - 70.88236) * (4.726476e-05 - Q.ecf_g42)
    if Q.mass < 100.4835 and Q.n_real_top30 < 30.0:
        z += -0.0002066923 * (100.4835 - Q.mass) * (30.0 - Q.n_real_top30)
    if Q.n_s3d_above_3 > 3.0 and Q.z_photon > 0.05998812:
        z += -0.1490649 * (Q.n_s3d_above_3 - 3.0) * (Q.z_photon - 0.05998812)
    if Q.n_s3d_above_10 < 8.0 and Q.n_lund_kt_above_1 > 3.0:
        z += -3.564202e-05 * (8.0 - Q.n_s3d_above_10) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.n_dr_0p4_up < 15.0 and Q.jet_charge_k03 > 0.1164034:
        z += 0.01481263 * (15.0 - Q.n_dr_0p4_up) * (Q.jet_charge_k03 - 0.1164034)
    if Q.mass_top40 > 70.88236 and Q.lne_4 < 3.808744:
        z += -0.001354009 * (Q.mass_top40 - 70.88236) * (3.808744 - Q.lne_4)
    if Q.n_s3d_above_10 < 8.0 and Q.z_neutral_had < 0.2675458:
        z += -0.2114473 * (8.0 - Q.n_s3d_above_10) * (0.2675458 - Q.z_neutral_had)
    if Q.sj2_dr < 0.5076533 and Q.lne_2 < 5.114244:
        z += -0.1622951 * (0.5076533 - Q.sj2_dr) * (5.114244 - Q.lne_2)
    if Q.sj2_dr < 0.5076533 and Q.lep_iso < 0.4381892:
        z += 0.1778644 * (0.5076533 - Q.sj2_dr) * (0.4381892 - Q.lep_iso)
    if Q.sj2_dr < 0.5076533 and Q.lund1_lnz > -4.349248:
        z += 0.1606159 * (0.5076533 - Q.sj2_dr) * (Q.lund1_lnz - -4.349248)
    if Q.n_pairs_kt_above_1 < 702.0 and Q.ismuon_37 > 0.0:
        z += 0.001103022 * (702.0 - Q.n_pairs_kt_above_1) * (Q.ismuon_37 - 0.0)
    if Q.n_s3d_above_10 < 8.0 and Q.eta_76 > 0.0:
        z += -0.02520691 * (8.0 - Q.n_s3d_above_10) * (Q.eta_76 - 0.0)
    if Q.mass_displaced3 < 1.777286 and Q.n_neutral_had < 9.0:
        z += 0.005380417 * (1.777286 - Q.mass_displaced3) * (9.0 - Q.n_neutral_had)
    if Q.lep_ptrel < 43.20788 and Q.sj3_z3 > 0.09421497:
        z += 0.002695502 * (43.20788 - Q.lep_ptrel) * (Q.sj3_z3 - 0.09421497)
    if Q.sj2_dr < 0.5076533 and Q.iselectron_0 > 0.0:
        z += -0.8211337 * (0.5076533 - Q.sj2_dr) * (Q.iselectron_0 - 0.0)
    if Q.sj2_dr < 0.5076533 and Q.z_muon < 0.1403354:
        z += 2.922208 * (0.5076533 - Q.sj2_dr) * (0.1403354 - Q.z_muon)
    if Q.n_pairs_kt_above_1 < 366.0 and Q.iselectron_1 > 0.0:
        z += -0.0003063005 * (366.0 - Q.n_pairs_kt_above_1) * (Q.iselectron_1 - 0.0)
    if Q.pair_max_lnm2 > 6.520267 and Q.td0_14 < -0.06205547:
        z += 0.07114344 * (Q.pair_max_lnm2 - 6.520267) * (-0.06205547 - Q.td0_14)
    if Q.mass < 117.4867 and Q.ismuon_49 > 0.0:
        z += -0.006062278 * (117.4867 - Q.mass) * (Q.ismuon_49 - 0.0)
    if Q.n_s3d_above_3 > 3.0 and Q.charge_42 > 0.0:
        z += 0.003978679 * (Q.n_s3d_above_3 - 3.0) * (Q.charge_42 - 0.0)
    if Q.lep_ptrel < 43.20788 and Q.tdz_2 < -0.1462819:
        z += -0.00578003 * (43.20788 - Q.lep_ptrel) * (-0.1462819 - Q.tdz_2)
    if Q.tau32 < 0.3164143 and Q.d0err_14 < 0.02580261:
        z += 61.97602 * (0.3164143 - Q.tau32) * (0.02580261 - Q.d0err_14)
    if Q.lep_z < 0.3396572 and Q.tau43_b2 < 0.3329206:
        z += 0.4454081 * (0.3396572 - Q.lep_z) * (0.3329206 - Q.tau43_b2)
    if Q.n_s3d_above_10 < 8.0 and Q.d0err_16 < 0.019104:
        z += 0.07376242 * (8.0 - Q.n_s3d_above_10) * (0.019104 - Q.d0err_16)
    return z


def neuron_125(Q):
    z = -0.0001207049
    return z


def neuron_126(Q):
    z = -0.0001121856
    return z


def neuron_127(Q):
    z = 0.0001268418
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
