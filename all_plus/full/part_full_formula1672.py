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
    z = 0.005196407
    return z


def neuron_1(Q):
    z = -5.325267e-06
    return z


def neuron_2(Q):
    z = 5.4473e-06
    return z


def neuron_3(Q):
    z = 8.399837e-06
    return z


def neuron_4(Q):
    z = 2.792139e-06
    return z


def neuron_5(Q):
    z = -2.156926e-06
    return z


def neuron_6(Q):
    z = 1.770106e-05
    return z


def neuron_7(Q):
    z = 6.430826e-06
    return z


def neuron_8(Q):
    z = 2.926471e-06
    return z


def neuron_9(Q):
    z = -2.443662e-06
    return z


def neuron_10(Q):
    z = -1.570105e-05
    return z


def neuron_11(Q):
    z = -1.887713e-05
    return z


def neuron_12(Q):
    z = 5.0993e-06
    return z


def neuron_13(Q):
    z = 4.422913e-06
    return z


def neuron_14(Q):
    z = 2.845304e-06
    return z


def neuron_15(Q):
    z = -6.459526e-06
    return z


def neuron_16(Q):
    z = -0.45726
    if Q.lep_ptrel < 27.3236:
        z += 0.02884754 * Q.lep_ptrel + 2.048302
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.1038121 * Q.lep_ptrel
    if Q.lep_ptrel >= 43.20788:
        z += 0.001479037 * Q.lep_ptrel + 4.421595
    if Q.n_s3d_above_3 < 10.0:
        z += 0.2148456 * Q.n_s3d_above_3 - 2.148456
    if Q.pz_lnd2 < 0.08321691:
        z += -3.119354 * Q.pz_lnd2 + 0.259583
    if Q.sip_3d_2 < 226.3008:
        z += -0.00201866 * Q.sip_3d_2 + 0.4811167
    if 226.3008 <= Q.sip_3d_2 < 447.0873:
        z += -0.000110027 * Q.sip_3d_2 + 0.04919166
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.005120043 * Q.n_pairs_kt_above_3 + 0.3834136
    if 28.0 <= Q.n_pairs_kt_above_3 < 34.0:
        z += -0.04000874 * Q.n_pairs_kt_above_3 + 1.360297
    if Q.mass_neutral >= 34.02385:
        z += 0.005988545 * Q.mass_neutral - 0.2037533
    if Q.z_displaced5 < 0.2801368:
        z += 3.856106 * Q.z_displaced5 - 1.080237
    if Q.e3_b2 < 0.0004126585:
        z += -173.1459 * Q.e3_b2 + 0.2463768
    if 0.0004126585 <= Q.e3_b2 < 0.0007909605:
        z += -462.3996 * Q.e3_b2 + 0.3657398
    if Q.pz_lnd0 < 0.156893:
        z += 1.335459 * Q.pz_lnd0 - 0.2095241
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.004690775 * Q.mres_sd_mass_b0z005 - 0.5727436
    if 159.9242 <= Q.mres_sd_mass_b0z005 < 178.8957:
        z += -0.02918272 * Q.mres_sd_mass_b0z005 + 4.844449
    if Q.mres_sd_mass_b0z005 >= 178.8957:
        z += -0.01386325 * Q.mres_sd_mass_b0z005 + 2.103861
    z += -3.958809 * Q.z_charged_had
    if Q.mres_sd_mass_b2z01 < 91.15481:
        z += -0.004742844 * Q.mres_sd_mass_b2z01 + 0.4323331
    if 20.76537 <= Q.mass_2charged < 45.73288:
        z += 0.005060782 * Q.mass_2charged - 0.105089
    if Q.mass_2charged >= 45.73288:
        z += -0.008107075 * Q.mass_2charged + 0.497115
    if Q.max_abs_d0 < 5.8125:
        z += -0.04000496 * Q.max_abs_d0 + 0.2325288
    if Q.sdb_2_n < 16.0:
        z += -0.02198959 * Q.sdb_2_n + 0.3518334
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.01165348 * Q.n_pairs_kt_above_1 + 0.9322787
    if Q.z_dr_0p4_up < 0.01553665:
        z += 5.360673 * Q.z_dr_0p4_up - 0.0832869
    if Q.mass_top5 >= 62.84493:
        z += 0.002341773 * Q.mass_top5 - 0.1471686
    if Q.mass_charged < 62.00562:
        z += -0.01407697 * Q.mass_charged + 0.8728512
    if Q.sjq_2_sumabs_k1 >= 0.2432922:
        z += 0.8821595 * Q.sjq_2_sumabs_k1 - 0.2146225
    if Q.ak02_min12_n_disp3 >= 1.0:
        z += 0.08240318 * Q.ak02_min12_n_disp3 - 0.08240318
    if Q.sjq_2_prod_k1 < 0.09802954:
        z += -2.237022 * Q.sjq_2_prod_k1 + 0.2192942
    if Q.mres_sd_prong_mass1 < 46.61784:
        z += -0.003167244 * Q.mres_sd_prong_mass1 + 0.1476501
    if Q.dr_min_012 < 0.01841101:
        z += 8.349607 * Q.dr_min_012 - 0.1537247
    if Q.jd_3d_6 >= 30.57088:
        z += -0.0007068089 * Q.jd_3d_6 + 0.02160777
    if Q.mass_top40 >= 112.406:
        z += 0.01519601 * Q.mass_top40 - 1.708123
    if Q.sjf_3_2_max3d < 3.101398:
        z += -0.04372728 * Q.sjf_3_2_max3d + 0.1356157
    if Q.n_photon < 18.0:
        z += 0.0232117 * Q.n_photon - 0.4178105
    if Q.n_dr_0p1_0p2 < 6.0:
        z += -0.03461901 * Q.n_dr_0p1_0p2 + 0.207714
    z += 0.04627444 * Q.n_sdz_above_2
    if Q.sip_3d_3 < 175.9957:
        z += -0.00182424 * Q.sip_3d_3 + 0.3210583
    if Q.jd_3d_4 < 14.85973:
        z += 0.02927908 * Q.jd_3d_4 - 0.4350791
    if Q.z_dr_0p05_0p1 < 0.01556887:
        z += -9.788151 * Q.z_dr_0p05_0p1 + 0.1523905
    if Q.M3 >= 0.02716047:
        z += 7.657534 * Q.M3 - 0.2079822
    if Q.lne_4 < 3.577424:
        z += -0.07623376 * Q.lne_4 + 0.2727205
    if Q.sj2_mass2 < 7.54918:
        z += 0.03598531 * Q.sj2_mass2 - 0.2716595
    if Q.sjq_2_2_nch < 3.0:
        z += -0.1041158 * Q.sjq_2_2_nch + 0.3123475
    if Q.D3_b05 < 0.8860453:
        z += 0.09148507 * Q.D3_b05 - 0.08105992
    if Q.z_electron < 0.1373477:
        z += 1.79734 * Q.z_electron - 0.2468604
    if Q.max_abs_dz < 0.4436035:
        z += 0.5154854 * Q.max_abs_dz - 0.01975428
    if 0.4436035 <= Q.max_abs_dz < 2.957031:
        z += -0.08312029 * Q.max_abs_dz + 0.2457893
    if Q.sdb_0_n >= 3.0:
        z += 0.03827105 * Q.sdb_0_n - 0.1148131
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += -0.009633956 * Q.sjf_2_1_n_d3 + 0.04816978
    if Q.sjf_4_1_n_d3 >= 2.0:
        z += 0.03494834 * Q.sjf_4_1_n_d3 - 0.06989668
    if Q.mass_displaced3 < 1.777286:
        z += 0.2456001 * Q.mass_displaced3 - 0.4365016
    if Q.mres_sd_rg_b2z01 >= 0.457968:
        z += 0.760975 * Q.mres_sd_rg_b2z01 - 0.3485022
    if Q.tdz_1 < -0.1170754:
        z += -0.2367386 * Q.tdz_1 - 0.02771626
    if Q.sdb_2_z < 0.1530389:
        z += -0.1286896 * Q.sdb_2_z + 0.01969451
    if Q.D2 < 1.00791:
        z += 0.07826547 * Q.D2 - 0.07888458
    if Q.jd_sum_abs_sd0_top5 < 9.408315:
        z += 0.125396 * Q.jd_sum_abs_sd0_top5 - 1.179765
    if Q.mres_sd_mass_b0z02 >= 93.75785:
        z += 0.001737728 * Q.mres_sd_mass_b0z02 - 0.1629256
    if Q.ak02_dr23 >= 0.3194837:
        z += 0.2463152 * Q.ak02_dr23 - 0.0786937
    if Q.sjf_2_2_maxsd0 < 1.141642:
        z += -0.1225785 * Q.sjf_2_2_maxsd0 + 0.1399407
    if Q.sjf_2_1_maxsd0 < 1.113424:
        z += -0.09871136 * Q.sjf_2_1_maxsd0 + 0.1099076
    if Q.ecf_g41 < 0.0002703514:
        z += 519.9277 * Q.ecf_g41 - 0.1405631
    if Q.n_pairs_kt_above_10 < 7.0:
        z += -0.03469909 * Q.n_pairs_kt_above_10 + 0.2428936
    if Q.N2_b05 < 0.302405:
        z += 0.3063491 * Q.N2_b05 - 0.09264149
    if Q.N2 < 0.1506626:
        z += 4.346356 * Q.N2 - 0.6548331
    if Q.sjf_4_n2disp < 2.0:
        z += -0.04508628 * Q.sjf_4_n2disp + 0.09017255
    if Q.z_muon < 0.03898651:
        z += 4.09336 * Q.z_muon - 0.1595858
    if Q.lep_z < 0.221436:
        z += -2.744542 * Q.lep_z + 1.423677
    if 0.221436 <= Q.lep_z < 0.5187302:
        z += -4.713763 * Q.lep_z + 1.859733
    if Q.lep_z >= 0.5187302:
        z += -1.96922 * Q.lep_z + 0.4360563
    if Q.z_charged < 0.8615361:
        z += 3.481381 * Q.z_charged - 2.999336
    if Q.n_pairs_kt_above_3 < 28.0 and Q.kt2_2_n_lep < 1.0:
        z += 0.0007387499 * (28.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.kt2_2_n_lep)
    if Q.z_displaced5 < 0.2801368 and Q.jd_3d_4 < 172.888:
        z += 0.02064821 * (0.2801368 - Q.z_displaced5) * (172.888 - Q.jd_3d_4)
    if Q.n_s3d_above_3 < 10.0 and Q.n_neutral > 10.0:
        z += 0.001396851 * (10.0 - Q.n_s3d_above_3) * (Q.n_neutral - 10.0)
    if Q.mass_2charged > 20.76537 and Q.M2_b2 > 0.01015545:
        z += 0.02593845 * (Q.mass_2charged - 20.76537) * (Q.M2_b2 - 0.01015545)
    if Q.n_s3d_above_3 < 10.0 and Q.sv_1_n > 1.0:
        z += 0.05754017 * (10.0 - Q.n_s3d_above_3) * (Q.sv_1_n - 1.0)
    if Q.e3_b2 < 0.0004126585 and Q.n_lund > 5.0:
        z += 22.21265 * (0.0004126585 - Q.e3_b2) * (Q.n_lund - 5.0)
    if Q.n_s3d_above_3 < 10.0 and Q.sdb_4_z < 0.1270192:
        z += 0.1667648 * (10.0 - Q.n_s3d_above_3) * (0.1270192 - Q.sdb_4_z)
    if Q.n_pairs_kt_above_1 < 80.0 and Q.tau32 < 0.6513932:
        z += -0.0119214 * (80.0 - Q.n_pairs_kt_above_1) * (0.6513932 - Q.tau32)
    if Q.mass_neutral > 34.02385 and Q.jd_3d_5 < 2.912651:
        z += -0.005458024 * (Q.mass_neutral - 34.02385) * (2.912651 - Q.jd_3d_5)
    if Q.sjq_2_prod_k1 < 0.09802954 and Q.ak02_2_n_lep < 1.0:
        z += -0.7283484 * (0.09802954 - Q.sjq_2_prod_k1) * (1.0 - Q.ak02_2_n_lep)
    if Q.sdb_2_n < 16.0 and Q.tau43 < 0.8479222:
        z += -0.04438978 * (16.0 - Q.sdb_2_n) * (0.8479222 - Q.tau43)
    if Q.sjq_2_sumabs_k1 > 0.2432922 and Q.n_lepton < 2.0:
        z += -0.157525 * (Q.sjq_2_sumabs_k1 - 0.2432922) * (2.0 - Q.n_lepton)
    if Q.sdb_2_n < 16.0 and Q.dc_pair_mass_min < 105.4151:
        z += -5.871361e-05 * (16.0 - Q.sdb_2_n) * (105.4151 - Q.dc_pair_mass_min)
    if Q.D3_b05 < 0.8860453 and Q.n_muon > 0.0:
        z += 0.03184402 * (0.8860453 - Q.D3_b05) * (Q.n_muon - 0.0)
    if Q.sip_3d_2 < 447.0873 and Q.lepsj_2_n_d3 > 2.0:
        z += -0.0002790517 * (447.0873 - Q.sip_3d_2) * (Q.lepsj_2_n_d3 - 2.0)
    if Q.sip_3d_3 < 175.9957 and Q.mres_sd_prong_mass2 > 1.685874e-06:
        z += -3.861278e-05 * (175.9957 - Q.sip_3d_3) * (Q.mres_sd_prong_mass2 - 1.685874e-06)
    if Q.sjq_2_prod_k1 < 0.09802954 and Q.dc_1_z_disp3 < 0.200234:
        z += -2.126716 * (0.09802954 - Q.sjq_2_prod_k1) * (0.200234 - Q.dc_1_z_disp3)
    if Q.lep_ptrel > 43.20788 and Q.kt2_1_sd0_1 < 13.96555:
        z += -0.000108168 * (Q.lep_ptrel - 43.20788) * (13.96555 - Q.kt2_1_sd0_1)
    if Q.z_displaced5 < 0.2801368 and Q.n_lund_kt_above_5 > 1.0:
        z += -0.334499 * (0.2801368 - Q.z_displaced5) * (Q.n_lund_kt_above_5 - 1.0)
    if Q.mres_sd_prong_mass1 < 46.61784 and Q.dc_mass_disp_2nd > 0.0:
        z += -0.0007860904 * (46.61784 - Q.mres_sd_prong_mass1) * (Q.dc_mass_disp_2nd - 0.0)
    if Q.n_s3d_above_3 < 10.0 and Q.sjf_4_3_z_d3 > 0.07303924:
        z += 0.1832544 * (10.0 - Q.n_s3d_above_3) * (Q.sjf_4_3_z_d3 - 0.07303924)
    if Q.e3_b2 < 0.0004126585 and Q.mass_2photon > 22.18431:
        z += 32.08265 * (0.0004126585 - Q.e3_b2) * (Q.mass_2photon - 22.18431)
    if Q.sjq_2_sumabs_k1 > 0.2432922 and Q.ischhad_4 < 1.0:
        z += 0.02449648 * (Q.sjq_2_sumabs_k1 - 0.2432922) * (1.0 - Q.ischhad_4)
    if Q.sdb_2_n < 16.0 and Q.sjq_3_sumabs_k03 > 0.5318777:
        z += 0.005834167 * (16.0 - Q.sdb_2_n) * (Q.sjq_3_sumabs_k03 - 0.5318777)
    if Q.dr_min_012 < 0.01841101 and Q.sjf_4_1_max3d < 3.652671:
        z += 3.264803 * (0.01841101 - Q.dr_min_012) * (3.652671 - Q.sjf_4_1_max3d)
    if Q.sjq_2_sumabs_k1 > 0.2432922 and Q.iselectron_29 > 0.0:
        z += -0.3505685 * (Q.sjq_2_sumabs_k1 - 0.2432922) * (Q.iselectron_29 - 0.0)
    if Q.sjf_2_2_maxsd0 < 1.141642 and Q.iselectron_1 > 0.0:
        z += -0.2810674 * (1.141642 - Q.sjf_2_2_maxsd0) * (Q.iselectron_1 - 0.0)
    if Q.M3 > 0.02716047 and Q.jd_mass_d3_over_z < 486.2148:
        z += 0.0005213226 * (Q.M3 - 0.02716047) * (486.2148 - Q.jd_mass_d3_over_z)
    return z


def neuron_17(Q):
    z = -1.035347e-05
    return z


def neuron_18(Q):
    z = 1.983111
    if Q.lep_z < 0.004135872:
        z += 134.7788 * Q.lep_z - 1.504808
    if 0.004135872 <= Q.lep_z < 0.221436:
        z += 4.359778 * Q.lep_z - 0.965412
    if 100.4835 <= Q.mass < 110.2019:
        z += -0.01304894 * Q.mass + 1.311203
    if 110.2019 <= Q.mass < 164.4374:
        z += -0.01985263 * Q.mass + 2.060982
    if 164.4374 <= Q.mass < 182.8592:
        z += 0.0186173 * Q.mass - 4.264915
    if Q.mass >= 182.8592:
        z += -0.007473408 * Q.mass + 0.506012
    if Q.mass_top20 < 114.4658:
        z += 0.003997082 * Q.mass_top20 - 0.4575293
    if Q.lepsj_3_n_d3 < 2.0:
        z += -0.431387 * Q.lepsj_3_n_d3 + 1.133843
    if 2.0 <= Q.lepsj_3_n_d3 < 3.0:
        z += -0.271069 * Q.lepsj_3_n_d3 + 0.813207
    if Q.z_displaced3 < 0.03624058:
        z += 12.68904 * Q.z_displaced3 - 0.4598583
    if Q.z_displaced3 >= 0.2092108:
        z += -1.845764 * Q.z_displaced3 + 0.3861536
    if Q.n_s3d_above_10 < 6.0:
        z += -0.04646158 * Q.n_s3d_above_10 + 0.2787695
    if Q.mass_displaced3 < 2.409613:
        z += -0.13693 * Q.mass_displaced3 + 0.4100576
    if 2.409613 <= Q.mass_displaced3 < 9.090532:
        z += -0.01199075 * Q.mass_displaced3 + 0.1090023
    if Q.lep_iso < 1.362094:
        z += -0.221585 * Q.lep_iso + 0.03973725
    if 1.362094 <= Q.lep_iso < 6.185635:
        z += 0.05433404 * Q.lep_iso - 0.3360905
    if Q.z_displaced5 < 0.08090366:
        z += -9.094439 * Q.z_displaced5 + 0.7357734
    if Q.lund3_lndelta >= -2.4279:
        z += -0.07551424 * Q.lund3_lndelta - 0.183341
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.1630371 * Q.lepsj_3_maxsd0 - 0.3318214
    if 2.413264 <= Q.lepsj_3_maxsd0 < 46.74905:
        z += 0.009487948 * Q.lepsj_3_maxsd0 - 0.7481701
    if 46.74905 <= Q.lepsj_3_maxsd0 < 586.9572:
        z += 0.0005638892 * Q.lepsj_3_maxsd0 - 0.3309788
    if 111.4443 <= Q.mres_sd_mass_b2z01 < 121.6067:
        z += 0.01393555 * Q.mres_sd_mass_b2z01 - 1.553037
    if 121.6067 <= Q.mres_sd_mass_b2z01 < 177.5398:
        z += -0.007803513 * Q.mres_sd_mass_b2z01 + 1.090579
    if Q.mres_sd_mass_b2z01 >= 177.5398:
        z += 0.01564286 * Q.mres_sd_mass_b2z01 - 3.072087
    if Q.C2_b05 >= 0.3223395:
        z += -0.2954381 * Q.C2_b05 + 0.09523137
    if Q.n_s3d_above_3 < 3.0:
        z += 0.1037354 * Q.n_s3d_above_3 - 0.5203725
    if 3.0 <= Q.n_s3d_above_3 < 10.0:
        z += 0.0298809 * Q.n_s3d_above_3 - 0.298809
    if Q.mass_displaced5 < 1.262841:
        z += 0.03288755 * Q.mass_displaced5 - 0.04153174
    if Q.ak02_1_z < 0.849996:
        z += 0.4687806 * Q.ak02_1_z - 0.3984616
    if Q.sj3_pair_mass_max >= 128.6605:
        z += -0.01743532 * Q.sj3_pair_mass_max + 2.243238
    if Q.n_dr_0p4_up < 4.0:
        z += -0.04294885 * Q.n_dr_0p4_up + 0.1717954
    if Q.lam1 < 0.01299032:
        z += 62.50572 * Q.lam1 - 0.8119694
    if Q.n_real_top50 >= 26.0:
        z += -0.01549151 * Q.n_real_top50 + 0.4027793
    if Q.M2 >= 0.07013948:
        z += 2.893719 * Q.M2 - 0.202964
    if Q.sv_1_sd0_sum < 51.56287:
        z += -0.003312663 * Q.sv_1_sd0_sum + 0.1708104
    if Q.sjf_3_1_max3d < 2.078395:
        z += -0.05609952 * Q.sjf_3_1_max3d - 0.2259485
    if 2.078395 <= Q.sjf_3_1_max3d < 2.924357:
        z += 0.06349196 * Q.sjf_3_1_max3d - 0.4745069
    if 2.924357 <= Q.sjf_3_1_max3d < 3.710567:
        z += 0.6861983 * Q.sjf_3_1_max3d - 2.295523
    if 3.710567 <= Q.sjf_3_1_max3d < 71.08287:
        z += -0.003720559 * Q.sjf_3_1_max3d + 0.264468
    if Q.sjf_2_2_max3d < 4.516968:
        z += 0.2164266 * Q.sjf_2_2_max3d - 0.5802039
    if 4.516968 <= Q.sjf_2_2_max3d < 55.92931:
        z += -0.007729431 * Q.sjf_2_2_max3d + 0.4323017
    if Q.sjf_2_1_z_d3 < 0.0828821:
        z += 3.30654 * Q.sjf_2_1_z_d3 - 0.274053
    if Q.sjq_2_1_nch < 16.0:
        z += -0.02619965 * Q.sjq_2_1_nch + 0.4191944
    if Q.sj2_mass2 < 6.465318:
        z += -0.06833263 * Q.sj2_mass2 + 0.4417922
    if Q.n_sdz_above_5 < 3.0:
        z += -0.07348049 * Q.n_sdz_above_5 + 0.2204415
    if Q.sip_3d_2 < 226.3008:
        z += -0.000211496 * Q.sip_3d_2 + 0.04786172
    if Q.sv_2_n >= 1.0:
        z += 0.1032676 * Q.sv_2_n - 0.1032676
    if Q.z_charged_had < 0.3603262:
        z += -1.687708 * Q.z_charged_had + 0.6081254
    if Q.n_for_90pct < 29.0:
        z += 0.03050563 * Q.n_for_90pct - 0.8846634
    if Q.tau5 < 0.03663959:
        z += -23.10435 * Q.tau5 + 0.8465339
    if Q.sdb_2_z < 0.5726046:
        z += -0.2075067 * Q.sdb_2_z + 0.1188193
    if Q.sdb_jp_top3 < 813.043:
        z += 0.0001669683 * Q.sdb_jp_top3 - 0.1357524
    if Q.n_pairs_kt_above_1 < 101.0:
        z += 0.0003951193 * Q.n_pairs_kt_above_1 - 0.03990705
    if Q.psi_0p1 < 0.624781:
        z += -0.2931677 * Q.psi_0p1 + 0.1831656
    if Q.mres_sd_mass_b0z02 >= 17.38085:
        z += -2.849456e-07 * Q.mres_sd_mass_b0z02 + 4.952597e-06
    if Q.N2 >= 0.4137858:
        z += -3.655413 * Q.N2 + 1.512558
    if Q.ak02_dr23 >= 0.3194837:
        z += -0.3123693 * Q.ak02_dr23 + 0.09979689
    if Q.sjq_2_2_nch < 2.0:
        z += 0.05830071 * Q.sjq_2_2_nch - 0.1166014
    z += -0.002173257 * Q.sjf_4_n2disp
    if Q.sjf_4_1_z_d3 >= 0.181327:
        z += 1.0789 * Q.sjf_4_1_z_d3 - 0.1956337
    if Q.n_dr_0p1_0p2 < 17.0:
        z += -0.004186085 * Q.n_dr_0p1_0p2 + 0.07116345
    if Q.ak02_3_z >= 0.07650476:
        z += -0.2994837 * Q.ak02_3_z + 0.02291193
    if Q.mass_charged >= 113.5019:
        z += 0.002411699 * Q.mass_charged - 0.2737325
    if Q.sjf_2_1_max3d < 3.32421:
        z += 0.08839934 * Q.sjf_2_1_max3d - 0.1230258
    if 3.32421 <= Q.sjf_2_1_max3d < 55.67805:
        z += -0.00326303 * Q.sjf_2_1_max3d + 0.1816792
    if Q.mass_neutral >= 59.66563:
        z += 0.001666808 * Q.mass_neutral - 0.09945115
    if Q.lepsj_3_dr >= 0.09790963:
        z += 0.4248221 * Q.lepsj_3_dr - 0.04159418
    if Q.sdb_4_z < 0.01435877:
        z += 1.214474 * Q.sdb_4_z - 0.01743835
    if Q.jd_n_d3_pt1 < 4.0:
        z += -0.05098441 * Q.jd_n_d3_pt1 + 0.2039376
    if Q.mass_top40 < 126.8853:
        z += 0.004051596 * Q.mass_top40 - 0.5140878
    if Q.mass_top30 < 146.3096:
        z += -0.002720222 * Q.mass_top30 + 0.3979946
    if Q.sip_3d_3 < 4.606241:
        z += 0.130242 * Q.sip_3d_3 - 0.5999261
    if Q.e3_b2 < 0.0002536827:
        z += -43.79547 * Q.e3_b2 - 0.2873772
    if 0.0002536827 <= Q.e3_b2 < 0.0007909605:
        z += 555.5551 * Q.e3_b2 - 0.4394221
    if Q.e3_b05 >= 0.008870191:
        z += -22.6743 * Q.e3_b05 + 0.2011254
    if Q.max_abs_dz < 17.21875:
        z += 0.006809515 * Q.max_abs_dz - 0.1172513
    if Q.nca_sj4_pair_mass_2nd >= 72.72359:
        z += 0.0005902491 * Q.nca_sj4_pair_mass_2nd - 0.04292503
    if Q.sdb_0_z < 0.05934469:
        z += 2.840143 * Q.sdb_0_z - 0.1685474
    if Q.lnptrel_2 >= -2.51737:
        z += 0.1174961 * Q.lnptrel_2 + 0.2957812
    if Q.sjf_3_3_max3d < 1.348343:
        z += -0.08382754 * Q.sjf_3_3_max3d + 0.1130282
    if Q.sjq_3_2_nch < 3.0:
        z += -0.06132758 * Q.sjq_3_2_nch + 0.1839827
    if Q.sj3_pairmin_over_m < 0.2072106:
        z += 1.296128 * Q.sj3_pairmin_over_m - 0.2685715
    if Q.pz_lnd0 < 0.1032185:
        z += -0.2403521 * Q.pz_lnd0 + 0.02480877
    if Q.sjf_3_1_n_d3 < 1.0:
        z += -0.2651481 * Q.sjf_3_1_n_d3 + 0.2651481
    if Q.lep_z < 0.221436 and Q.N2 < 0.3829918:
        z += 20.13293 * (0.221436 - Q.lep_z) * (0.3829918 - Q.N2)
    if Q.lep_z < 0.221436 and Q.sdb_2_z > 0.2509165:
        z += -6.373375 * (0.221436 - Q.lep_z) * (Q.sdb_2_z - 0.2509165)
    if Q.lepsj_3_n_d3 < 3.0 and Q.lep_dr < 0.08178299:
        z += -1.541879 * (3.0 - Q.lepsj_3_n_d3) * (0.08178299 - Q.lep_dr)
    if Q.lep_z < 0.221436 and Q.z_displaced5 < 0.08090366:
        z += -46.71712 * (0.221436 - Q.lep_z) * (0.08090366 - Q.z_displaced5)
    if Q.lep_z < 0.221436 and Q.lne_4 > 3.751708:
        z += 0.771714 * (0.221436 - Q.lep_z) * (Q.lne_4 - 3.751708)
    if Q.mass_displaced3 < 9.090532 and Q.sjq_2_2_nch > 4.0:
        z += -0.003340611 * (9.090532 - Q.mass_displaced3) * (Q.sjq_2_2_nch - 4.0)
    if Q.z_displaced5 < 0.08090366 and Q.z_neutral_had < 0.1912751:
        z += -0.3656962 * (0.08090366 - Q.z_displaced5) * (0.1912751 - Q.z_neutral_had)
    if Q.lepsj_3_n_d3 < 3.0 and Q.lepsj_3_dr > 0.0:
        z += -0.4005585 * (3.0 - Q.lepsj_3_n_d3) * (Q.lepsj_3_dr - 0.0)
    if Q.sv_1_sd0_sum < 51.56287 and Q.sjf_3_3_z_d3 > 0.006421567:
        z += -0.02108411 * (51.56287 - Q.sv_1_sd0_sum) * (Q.sjf_3_3_z_d3 - 0.006421567)
    if Q.sv_1_sd0_sum < 51.56287 and Q.dc_1_n_lep > 0.0:
        z += -0.001904307 * (51.56287 - Q.sv_1_sd0_sum) * (Q.dc_1_n_lep - 0.0)
    if Q.sjq_2_1_nch < 16.0 and Q.sum_e < 1651.766:
        z += -2.418886e-05 * (16.0 - Q.sjq_2_1_nch) * (1651.766 - Q.sum_e)
    if Q.n_real_top50 > 26.0 and Q.n_lund_kt_above_5 < 3.0:
        z += -0.006802664 * (Q.n_real_top50 - 26.0) * (3.0 - Q.n_lund_kt_above_5)
    if Q.lep_z < 0.221436 and Q.jet_charge_k05 > -0.2841306:
        z += -0.5808645 * (0.221436 - Q.lep_z) * (Q.jet_charge_k05 - -0.2841306)
    if Q.sdb_2_z < 0.5726046 and Q.eta_28 < 0.3249512:
        z += -0.07566473 * (0.5726046 - Q.sdb_2_z) * (0.3249512 - Q.eta_28)
    if Q.n_real_top50 > 26.0 and Q.d0err_0 > 0.0:
        z += 0.1209705 * (Q.n_real_top50 - 26.0) * (Q.d0err_0 - 0.0)
    return z


def neuron_19(Q):
    z = -1.407819e-06
    return z


def neuron_20(Q):
    z = 8.253685e-07
    return z


def neuron_21(Q):
    z = -0.001681615
    return z


def neuron_22(Q):
    z = -1.044848e-06
    return z


def neuron_23(Q):
    z = -6.978821e-07
    return z


def neuron_24(Q):
    z = 0.3424412
    if Q.lep_z < 0.004135872:
        z += 48.0385 * Q.lep_z - 1.4909
    if 0.004135872 <= Q.lep_z < 0.3396572:
        z += 3.851375 * Q.lep_z - 1.308147
    if Q.mass_top30 < 74.51927:
        z += 0.009731139 * Q.mass_top30 - 0.4590043
    if 74.51927 <= Q.mass_top30 < 134.2224:
        z += -0.004457944 * Q.mass_top30 + 0.5983559
    if Q.mass < 90.08945:
        z += 0.01407174 * Q.mass - 2.059709
    if 90.08945 <= Q.mass < 110.2019:
        z += 0.0006745365 * Q.mass - 0.8527627
    if 110.2019 <= Q.mass < 117.4867:
        z += 0.0211885 * Q.mass - 3.113441
    if 117.4867 <= Q.mass < 164.4374:
        z += 0.01329209 * Q.mass - 2.185718
    z += 0.1568665 * Q.pair_max_lnkt
    if Q.z_displaced3 < 0.1061578:
        z += 1.961173 * Q.z_displaced3 - 0.2081938
    if Q.e3_b2 < 7.452974e-05:
        z += 3285.702 * Q.e3_b2 - 0.6560506
    if 7.452974e-05 <= Q.e3_b2 < 0.0004126585:
        z += 1216.01 * Q.e3_b2 - 0.501797
    if Q.lep_ptrel < 12.15228:
        z += 0.00620301 * Q.lep_ptrel - 0.1584622
    if 12.15228 <= Q.lep_ptrel < 43.20788:
        z += 0.002675249 * Q.lep_ptrel - 0.1155919
    if Q.mres_sd_mass_b2z01 < 55.13212:
        z += -0.006104808 * Q.mres_sd_mass_b2z01 + 0.3376323
    if 55.13212 <= Q.mres_sd_mass_b2z01 < 81.19466:
        z += 0.02975404 * Q.mres_sd_mass_b2z01 - 1.639342
    if 81.19466 <= Q.mres_sd_mass_b2z01 < 118.2746:
        z += -0.02094195 * Q.mres_sd_mass_b2z01 + 2.476902
    if Q.n_s3d_above_10 >= 1.0:
        z += -0.02118423 * Q.n_s3d_above_10 + 0.02118423
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.01686564 * Q.lepsj_3_maxsd0 + 0.04070125
    if Q.lepsj_2_dr < 0.103005:
        z += 0.9749352 * Q.lepsj_2_dr - 0.1004232
    if Q.mass_displaced3 < 18.80005:
        z += -0.0193536 * Q.mass_displaced3 + 0.3638487
    if Q.n_s3d_above_3 < 1.0:
        z += -0.1339041 * Q.n_s3d_above_3 + 0.2176357
    if 1.0 <= Q.n_s3d_above_3 < 3.0:
        z += -0.0418658 * Q.n_s3d_above_3 + 0.1255974
    if Q.n_pairs_kt_above_1 < 366.0:
        z += -0.0002583317 * Q.n_pairs_kt_above_1 + 0.09454942
    if Q.N2 < 0.2423129:
        z += 4.306864 * Q.N2 - 1.043609
    if Q.lep_iso < 0.4381892:
        z += -2.467805 * Q.lep_iso + 1.397385
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += -0.3420472 * Q.lep_iso + 0.4659005
    if Q.jd_3d_4 < 2.953464:
        z += -0.0309363 * Q.jd_3d_4 + 0.09136925
    if Q.C2_b2 < 0.01057938:
        z += 28.98427 * Q.C2_b2 - 0.3066357
    if Q.sjq_3_2_k1 < -0.6014774:
        z += -1.750305 * Q.sjq_3_2_k1 - 1.052769
    if Q.sjq_2_2_k1 >= 0.277232:
        z += 0.5405362 * Q.sjq_2_2_k1 - 0.1498539
    if Q.jd_sum_abs_sd0_top5 < 84.11398:
        z += -0.007051792 * Q.jd_sum_abs_sd0_top5 + 0.5931543
    if Q.z_neutral_had < 0.212136:
        z += -1.333505 * Q.z_neutral_had + 0.2828844
    if Q.sip_3d_1 < 7.028704:
        z += -0.01913979 * Q.sip_3d_1 + 0.1345279
    if Q.lam1 < 0.006142967:
        z += 219.7767 * Q.lam1 - 1.350081
    if Q.z_dr_0p2_0p4 < 0.1468781:
        z += 3.260635 * Q.z_dr_0p2_0p4 - 0.478916
    if Q.n_sd0_above_10 >= 6.0:
        z += 0.02068393 * Q.n_sd0_above_10 - 0.1241036
    if Q.sj3_pair_mass_max < 83.63398:
        z += 0.005660931 * Q.sj3_pair_mass_max - 0.03819014
    if 83.63398 <= Q.sj3_pair_mass_max < 128.6605:
        z += -0.009666651 * Q.sj3_pair_mass_max + 1.243717
    z += 0.3645016 * Q.sjq_2_prod_k1
    if Q.tau21 >= 0.135772:
        z += 0.7564085 * Q.tau21 - 0.1026991
    if Q.tau32 < 0.456416:
        z += 3.853774 * Q.tau32 - 1.758924
    if Q.n_lund < 5.0:
        z += -0.3273979 * Q.n_lund + 1.636989
    if Q.dc_2_jp < 6.161303:
        z += -0.02056764 * Q.dc_2_jp + 0.1267234
    if Q.mres_pruned_mass < 60.18017:
        z += -0.005439917 * Q.mres_pruned_mass + 0.00630741
    if 60.18017 <= Q.mres_pruned_mass < 86.14266:
        z += 0.03027988 * Q.mres_pruned_mass - 2.143316
    if 86.14266 <= Q.mres_pruned_mass < 119.8459:
        z += -0.020435 * Q.mres_pruned_mass + 2.225399
    if 119.8459 <= Q.mres_pruned_mass < 171.3819:
        z += 0.004339729 * Q.mres_pruned_mass - 0.7437511
    if 0.2293319 <= Q.dc_split2_dr < 0.3591078:
        z += 2.679166 * Q.dc_split2_dr - 0.6144183
    if Q.dc_split2_dr >= 0.3591078:
        z += -3.719851 * Q.dc_split2_dr + 1.683518
    if Q.n_dr_0p4_up < 4.0:
        z += -0.06925435 * Q.n_dr_0p4_up + 0.2770174
    if Q.mass_top15 >= 70.93762:
        z += -0.004075482 * Q.mass_top15 + 0.289105
    if Q.max_abs_d0 < 10.52344:
        z += -0.02880317 * Q.max_abs_d0 + 0.3031084
    if Q.z_charged_had >= 0.5398733:
        z += -0.6460341 * Q.z_charged_had + 0.3487765
    if Q.n_dr_0p2_0p4 < 17.0:
        z += -0.04541992 * Q.n_dr_0p2_0p4 + 0.7721386
    if Q.dr_8 >= 0.3729651:
        z += -0.4086639 * Q.dr_8 + 0.1524174
    if Q.lep_z < 0.3396572 and Q.mass < 127.2317:
        z += 0.05066657 * (0.3396572 - Q.lep_z) * (127.2317 - Q.mass)
    if Q.lep_z < 0.3396572 and Q.tau32 < 0.456416:
        z += 19.02262 * (0.3396572 - Q.lep_z) * (0.456416 - Q.tau32)
    if Q.lep_ptrel < 43.20788 and Q.sdb_2_z > 0.2968888:
        z += 0.02549495 * (43.20788 - Q.lep_ptrel) * (Q.sdb_2_z - 0.2968888)
    if Q.e3_b2 < 0.0004126585 and Q.lne_8 > 2.11682:
        z += 255.702 * (0.0004126585 - Q.e3_b2) * (Q.lne_8 - 2.11682)
    if Q.lep_z < 0.3396572 and Q.sjq_2_sumabs_k1 > 0.5557674:
        z += 0.2186559 * (0.3396572 - Q.lep_z) * (Q.sjq_2_sumabs_k1 - 0.5557674)
    if Q.z_displaced3 < 0.1061578 and Q.pz_lnd2 < 0.1166862:
        z += -15.29147 * (0.1061578 - Q.z_displaced3) * (0.1166862 - Q.pz_lnd2)
    if Q.z_displaced3 < 0.1061578 and Q.D2_b2 < 15.17086:
        z += 0.2184585 * (0.1061578 - Q.z_displaced3) * (15.17086 - Q.D2_b2)
    if Q.lepsj_3_maxsd0 < 2.413264 and Q.ak02_3_n_lep > 0.0:
        z += 0.09896614 * (2.413264 - Q.lepsj_3_maxsd0) * (Q.ak02_3_n_lep - 0.0)
    if Q.mass < 164.4374 and Q.sv_1_n < 3.0:
        z += -0.001301506 * (164.4374 - Q.mass) * (3.0 - Q.sv_1_n)
    if Q.e3_b2 < 0.0004126585 and Q.sdb_0_n < 5.0:
        z += -131.9387 * (0.0004126585 - Q.e3_b2) * (5.0 - Q.sdb_0_n)
    if Q.lep_iso < 0.4381892 and Q.mres_sd_prong_mass2 < 9.442313:
        z += 0.001408699 * (0.4381892 - Q.lep_iso) * (9.442313 - Q.mres_sd_prong_mass2)
    if Q.mass < 164.4374 and Q.M3_b2 < 0.0309457:
        z += -0.06327271 * (164.4374 - Q.mass) * (0.0309457 - Q.M3_b2)
    if Q.lepsj_2_dr < 0.103005 and Q.jet_e > 926.0771:
        z += 0.001742261 * (0.103005 - Q.lepsj_2_dr) * (Q.jet_e - 926.0771)
    if Q.mass < 164.4374 and Q.n_dr_0p2_0p4 < 24.0:
        z += 4.48499e-05 * (164.4374 - Q.mass) * (24.0 - Q.n_dr_0p2_0p4)
    if Q.jd_sum_abs_sd0_top5 < 84.11398 and Q.jd_3d_5 < 6.771002:
        z += -0.001611732 * (84.11398 - Q.jd_sum_abs_sd0_top5) * (6.771002 - Q.jd_3d_5)
    if Q.lepsj_3_maxsd0 < 2.413264 and Q.dc_3_n_lep > 0.0:
        z += -0.005989675 * (2.413264 - Q.lepsj_3_maxsd0) * (Q.dc_3_n_lep - 0.0)
    if Q.mass_displaced3 < 18.80005 and Q.jd_3d_5 < 4.004982:
        z += 0.01073701 * (18.80005 - Q.mass_displaced3) * (4.004982 - Q.jd_3d_5)
    if Q.lep_ptrel < 43.20788 and Q.ak02_3_n_disp3 > 1.0:
        z += -0.002161785 * (43.20788 - Q.lep_ptrel) * (Q.ak02_3_n_disp3 - 1.0)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lep_iso < 0.4381892:
        z += -0.09302075 * (83.63398 - Q.sj3_pair_mass_max) * (0.4381892 - Q.lep_iso)
    if Q.lep_z < 0.3396572 and Q.tau21 > 0.1757731:
        z += -3.021835 * (0.3396572 - Q.lep_z) * (Q.tau21 - 0.1757731)
    if Q.lep_z < 0.3396572 and Q.n_neutral_had < 7.0:
        z += -0.1724345 * (0.3396572 - Q.lep_z) * (7.0 - Q.n_neutral_had)
    if Q.mass_displaced3 < 18.80005 and Q.z_photon < 0.3090294:
        z += 0.03132277 * (18.80005 - Q.mass_displaced3) * (0.3090294 - Q.z_photon)
    if Q.z_displaced3 < 0.1061578 and Q.mass_2photon > 3.013:
        z += 0.04108194 * (0.1061578 - Q.z_displaced3) * (Q.mass_2photon - 3.013)
    if Q.tau32 < 0.456416 and Q.dr_10 > 0.2577145:
        z += -0.5632834 * (0.456416 - Q.tau32) * (Q.dr_10 - 0.2577145)
    if Q.e3_b2 < 0.0004126585 and Q.kt2_min12_n_disp3 > 1.0:
        z += 273.1497 * (0.0004126585 - Q.e3_b2) * (Q.kt2_min12_n_disp3 - 1.0)
    if Q.lep_z < 0.004135872 and Q.sdb_4_z < 0.01435877:
        z += 3086.381 * (0.004135872 - Q.lep_z) * (0.01435877 - Q.sdb_4_z)
    if Q.lep_z < 0.3396572 and Q.e4 > 4.06096e-07:
        z += 26510.53 * (0.3396572 - Q.lep_z) * (Q.e4 - 4.06096e-07)
    if Q.lep_z < 0.004135872 and Q.n_pairs_kt_above_10 > 4.0:
        z += -1.802905 * (0.004135872 - Q.lep_z) * (Q.n_pairs_kt_above_10 - 4.0)
    if Q.sjq_3_2_k1 < -0.6014774 and Q.lep_iso < 0.4381892:
        z += -2.632301 * (-0.6014774 - Q.sjq_3_2_k1) * (0.4381892 - Q.lep_iso)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lep_iso < 14.38858:
        z += 0.001593491 * (83.63398 - Q.sj3_pair_mass_max) * (14.38858 - Q.lep_iso)
    if Q.lepsj_2_dr < 0.103005 and Q.lep_iso < 14.38858:
        z += -0.8014792 * (0.103005 - Q.lepsj_2_dr) * (14.38858 - Q.lep_iso)
    if Q.sjq_2_2_k1 > 0.277232 and Q.lep_iso < 2.907433:
        z += -0.04976882 * (Q.sjq_2_2_k1 - 0.277232) * (2.907433 - Q.lep_iso)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lepsj_2_dr < 0.1822601:
        z += 0.1020475 * (83.63398 - Q.sj3_pair_mass_max) * (0.1822601 - Q.lepsj_2_dr)
    if Q.lep_z < 0.004135872 and Q.lep_iso < 2.907433:
        z += -98.11187 * (0.004135872 - Q.lep_z) * (2.907433 - Q.lep_iso)
    if Q.lep_z < 0.3396572 and Q.lepsj_3_maxsd0 < 0.8036986:
        z += 2.14092 * (0.3396572 - Q.lep_z) * (0.8036986 - Q.lepsj_3_maxsd0)
    if Q.mass_top15 > 70.93762 and Q.dc_1_n_lep > 0.0:
        z += -0.0008639633 * (Q.mass_top15 - 70.93762) * (Q.dc_1_n_lep - 0.0)
    if Q.lep_z < 0.3396572 and Q.sj3_dr12 < 0.3683248:
        z += -1.582497 * (0.3396572 - Q.lep_z) * (0.3683248 - Q.sj3_dr12)
    if Q.mass_displaced3 < 18.80005 and Q.sip_3d_2 < 2325.187:
        z += -1.078954e-05 * (18.80005 - Q.mass_displaced3) * (2325.187 - Q.sip_3d_2)
    if Q.max_abs_d0 < 10.52344 and Q.jd_3d_5 < 4.004982:
        z += -0.004622269 * (10.52344 - Q.max_abs_d0) * (4.004982 - Q.jd_3d_5)
    if Q.mass_top15 > 70.93762 and Q.sv_2_z > 0.00726873:
        z += -0.2039626 * (Q.mass_top15 - 70.93762) * (Q.sv_2_z - 0.00726873)
    if Q.lep_iso < 0.4381892 and Q.sdb_4_n > 0.0:
        z += 0.2205713 * (0.4381892 - Q.lep_iso) * (Q.sdb_4_n - 0.0)
    if Q.lep_iso < 0.4381892 and Q.lepsj_2_dr < 0.1822601:
        z += 8.831116 * (0.4381892 - Q.lep_iso) * (0.1822601 - Q.lepsj_2_dr)
    if Q.mass_top15 > 70.93762 and Q.sv_2_dr < 0.1572038:
        z += -0.005146929 * (Q.mass_top15 - 70.93762) * (0.1572038 - Q.sv_2_dr)
    if Q.mass_top15 > 70.93762 and Q.sjf_3_2_mass_d3 > 0.0:
        z += -1.866304e-05 * (Q.mass_top15 - 70.93762) * (Q.sjf_3_2_mass_d3 - 0.0)
    return z


def neuron_25(Q):
    z = 0.2328893
    if Q.lep_z < 0.221436:
        z += -0.5405937 * Q.lep_z + 0.1909615
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -0.6027223 * Q.lep_z + 0.204719
    if Q.M2_b2 < 0.0417856:
        z += 2.544872 * Q.M2_b2 - 0.106339
    if 0.302405 <= Q.N2_b05 < 0.4265629:
        z += 0.3883073 * Q.N2_b05 - 0.1174261
    if Q.N2_b05 >= 0.4265629:
        z += -1.893012 * Q.N2_b05 + 0.8557001
    if Q.mass_top40 < 55.75601:
        z += -0.003780101 * Q.mass_top40 + 0.2107634
    if Q.mass_top40 >= 173.1022:
        z += -0.001857493 * Q.mass_top40 + 0.321536
    if Q.n_s3d_above_3 < 2.0:
        z += -0.08657584 * Q.n_s3d_above_3 + 0.1731517
    if Q.mass_neutral < 76.15079:
        z += -0.0002607909 * Q.mass_neutral + 0.01985943
    if Q.sv_1_sd0_sum < 35.45098:
        z += -0.003305029 * Q.sv_1_sd0_sum + 0.1171665
    if Q.sum_z_dr2_top10 < 0.0255865:
        z += -0.3074483 * Q.sum_z_dr2_top10 + 0.007866528
    if Q.n_neutral < 17.0:
        z += 0.001946063 * Q.n_neutral - 0.03502913
    if 17.0 <= Q.n_neutral < 18.0:
        z += 0.002969982 * Q.n_neutral - 0.05243575
    if Q.n_neutral >= 18.0:
        z += 0.001023919 * Q.n_neutral - 0.01740663
    if Q.nca_sj4_pair_mass_2nd < 69.01131:
        z += 0.001369476 * Q.nca_sj4_pair_mass_2nd - 0.09450931
    if Q.sj3_pair_mass_max < 76.91486:
        z += -0.002060517 * Q.sj3_pair_mass_max + 0.1584843
    if Q.sj3_pair_mass_max >= 114.7848:
        z += -0.001549549 * Q.sj3_pair_mass_max + 0.1778647
    if Q.dc_1_n_disp3 < 3.0:
        z += 0.04681139 * Q.dc_1_n_disp3 - 0.1404342
    if Q.pz_lnd0 < 0.06871203:
        z += -2.383637 * Q.pz_lnd0 + 0.1637845
    if Q.z_neutral_had >= 0.02425618:
        z += -0.9267235 * Q.z_neutral_had + 0.02247877
    if Q.tau32 < 0.5068038:
        z += 0.4218878 * Q.tau32 - 0.2138143
    if Q.sum_z_dr2 < 0.04303099:
        z += -0.7456096 * Q.sum_z_dr2 + 0.03208432
    if Q.mass < 79.47361:
        z += 0.009902859 * Q.mass - 0.922825
    if 79.47361 <= Q.mass < 90.08945:
        z += -0.009211853 * Q.mass + 0.5962902
    if 90.08945 <= Q.mass < 95.14961:
        z += 0.004455528 * Q.mass - 0.6349968
    if 95.14961 <= Q.mass < 110.2019:
        z += 0.009909992 * Q.mass - 1.153987
    if 110.2019 <= Q.mass < 131.3917:
        z += 0.002920599 * Q.mass - 0.3837425
    if Q.lep_ptrel < 27.3236:
        z += -0.002092309 * Q.lep_ptrel + 0.05716941
    if Q.sj2_mass1 < 21.21062:
        z += 0.001439718 * Q.sj2_mass1 - 0.0305373
    if Q.sj2_mass1 >= 77.42768:
        z += 0.008427124 * Q.sj2_mass1 - 0.6524927
    if Q.mres_sd_mass_b0z02 < 47.97531:
        z += 0.002616498 * Q.mres_sd_mass_b0z02 + 0.01789449
    if 47.97531 <= Q.mres_sd_mass_b0z02 < 73.90849:
        z += -0.005985324 * Q.mres_sd_mass_b0z02 + 0.4305696
    if 73.90849 <= Q.mres_sd_mass_b0z02 < 86.60355:
        z += 0.0009292312 * Q.mres_sd_mass_b0z02 - 0.08047472
    if Q.sum_z_dr2_top3 < 0.00878048:
        z += 2.859714 * Q.sum_z_dr2_top3 - 0.02510966
    if Q.lund_max_lnkt < 4.482525:
        z += -0.003497759 * Q.lund_max_lnkt + 0.01567879
    if Q.n_sdz_above_5 < 2.0:
        z += 0.05093191 * Q.n_sdz_above_5 - 0.1018638
    if Q.z_displaced3 < 0.03624058:
        z += -5.074915 * Q.z_displaced3 + 0.1839179
    if Q.mass_charged < 58.87188:
        z += 0.0008600154 * Q.mass_charged - 0.05063073
    if Q.sjf_2_2_max3d < 2.655772:
        z += -0.05555809 * Q.sjf_2_2_max3d + 0.1475496
    if Q.sdb_0_z < 0.170055:
        z += -0.1793874 * Q.sdb_0_z + 0.03050572
    if Q.nca_sj4_pair2nd_over_mass < 0.6275286:
        z += 0.09961022 * Q.nca_sj4_pair2nd_over_mass - 0.06250827
    if Q.mres_sd_rg_b0z005 >= 0.6103335:
        z += -0.6610286 * Q.mres_sd_rg_b0z005 + 0.4034479
    if Q.z_charged_had < 0.265564:
        z += -0.1019872 * Q.z_charged_had + 0.06682996
    if 0.265564 <= Q.z_charged_had < 0.6552778:
        z += -0.4031993 * Q.z_charged_had + 0.146821
    if Q.z_charged_had >= 0.6552778:
        z += -0.301212 * Q.z_charged_had + 0.07999108
    if Q.pz_lnd2 < 0.2100115:
        z += -0.1730421 * Q.pz_lnd2 + 0.03634082
    if Q.sum_z_dr2_top20 < 0.05414784:
        z += -1.816711 * Q.sum_z_dr2_top20 + 0.09837099
    if Q.e4_b05 < 9.76395e-05:
        z += -344.0505 * Q.e4_b05 + 0.03359292
    if Q.N2_b2 < 0.1104943:
        z += -0.760269 * Q.N2_b2 + 0.08400542
    if Q.jd_3d_5 < 2.552762:
        z += -0.03342104 * Q.jd_3d_5 + 0.08531596
    if Q.dc_1_sd0_1 < 16.47324:
        z += -0.0001498575 * Q.dc_1_sd0_1 + 0.002468639
    if Q.n_pairs_kt_above_10 < 23.0:
        z += 0.006660193 * Q.n_pairs_kt_above_10 - 0.1531844
    if Q.sj3_pairmax_over_m < 0.8501496:
        z += 1.025403 * Q.sj3_pairmax_over_m - 0.8717459
    if Q.sj3_mass3 < 11.07803:
        z += 0.002393185 * Q.sj3_mass3 - 0.02651179
    if Q.n_pairs_kt_above_3 < 99.0:
        z += -0.003781606 * Q.n_pairs_kt_above_3 + 0.374379
    if Q.pair_max_lnm2 < 7.075642:
        z += -0.05983955 * Q.pair_max_lnm2 + 0.4234032
    if Q.dr12 < 0.05058744:
        z += 0.7169708 * Q.dr12 - 0.03626971
    if 55.13212 <= Q.mres_sd_mass_b2z01 < 76.1529:
        z += -0.003099074 * Q.mres_sd_mass_b2z01 + 0.1708585
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 91.15481:
        z += 0.00572978 * Q.mres_sd_mass_b2z01 - 0.5014843
    if 91.15481 <= Q.mres_sd_mass_b2z01 < 102.264:
        z += 0.00276984 * Q.mres_sd_mass_b2z01 - 0.2316716
    if 102.264 <= Q.mres_sd_mass_b2z01 < 107.2089:
        z += -0.003464281 * Q.mres_sd_mass_b2z01 + 0.4058545
    if 107.2089 <= Q.mres_sd_mass_b2z01 < 130.0968:
        z += -0.0009482095 * Q.mres_sd_mass_b2z01 + 0.1361092
    if Q.mres_sd_mass_b2z01 >= 130.0968:
        z += 0.0005159749 * Q.mres_sd_mass_b2z01 - 0.0543766
    z += -0.41794 * Q.z_photon
    if Q.mres_sd_prong_mass2 < 5.543454:
        z += -0.02170261 * Q.mres_sd_prong_mass2 + 0.1203074
    if Q.sj2_mass2 < 1.851735:
        z += -0.04467561 * Q.sj2_mass2 + 0.08272741
    if Q.jd_3d_4 < 6.43167:
        z += 0.009610304 * Q.jd_3d_4 - 0.06181031
    if Q.n_sd0_above_2 < 9.0:
        z += 0.00317606 * Q.n_sd0_above_2 - 0.02858454
    if Q.lnpt_30 >= 1.274457:
        z += -0.02223468 * Q.lnpt_30 + 0.02833714
    if Q.n_sd0_above_3 < 2.0:
        z += 0.212411 * Q.n_sd0_above_3 - 0.424822
    if Q.sjq_3_2_k1 < -0.6014774:
        z += -0.03114802 * Q.sjq_3_2_k1 - 0.01873483
    if Q.M2_b2 < 0.0417856 and Q.sjf_4_1_n_d3 < 5.0:
        z += -1.195332 * (0.0417856 - Q.M2_b2) * (5.0 - Q.sjf_4_1_n_d3)
    if Q.lep_z < 0.3396572 and Q.sj3_pair_mass_max > 73.24742:
        z += -0.0003536918 * (0.3396572 - Q.lep_z) * (Q.sj3_pair_mass_max - 73.24742)
    if Q.lep_z < 0.3396572 and Q.ak02_3_z < 0.1014193:
        z += -2.499329 * (0.3396572 - Q.lep_z) * (0.1014193 - Q.ak02_3_z)
    if Q.lep_z < 0.3396572 and Q.pz_lnd2 < 0.2241366:
        z += -0.1945 * (0.3396572 - Q.lep_z) * (0.2241366 - Q.pz_lnd2)
    if Q.mass_neutral < 76.15079 and Q.ak02_2_n_lep < 1.0:
        z += 0.002568531 * (76.15079 - Q.mass_neutral) * (1.0 - Q.ak02_2_n_lep)
    if Q.n_s3d_above_3 < 2.0 and Q.lepsj_2_dr < 0.006470637:
        z += 18.96414 * (2.0 - Q.n_s3d_above_3) * (0.006470637 - Q.lepsj_2_dr)
    if Q.mass_neutral < 76.15079 and Q.lepsj_2_n_d3 < 2.0:
        z += -3.133076e-05 * (76.15079 - Q.mass_neutral) * (2.0 - Q.lepsj_2_n_d3)
    if Q.M2_b2 < 0.0417856 and Q.mass_displaced3 > 6.341631:
        z += 0.06499352 * (0.0417856 - Q.M2_b2) * (Q.mass_displaced3 - 6.341631)
    if Q.lep_z < 0.3396572 and Q.sv_n > 0.0:
        z += -0.003072187 * (0.3396572 - Q.lep_z) * (Q.sv_n - 0.0)
    if Q.lep_z < 0.3396572 and Q.pz_lnkt4 > 0.0:
        z += 4.954243 * (0.3396572 - Q.lep_z) * (Q.pz_lnkt4 - 0.0)
    if Q.mass < 131.3917 and Q.sjq_2_prod_k1 < -0.02569022:
        z += -0.008544289 * (131.3917 - Q.mass) * (-0.02569022 - Q.sjq_2_prod_k1)
    if Q.N2_b05 > 0.302405 and Q.lne_1 > 4.222261:
        z += -0.4813767 * (Q.N2_b05 - 0.302405) * (Q.lne_1 - 4.222261)
    if Q.sj3_pair_mass_max > 114.7848 and Q.lepsj_3_n_d3 > 3.0:
        z += 0.001052555 * (Q.sj3_pair_mass_max - 114.7848) * (Q.lepsj_3_n_d3 - 3.0)
    if Q.lep_z < 0.3396572 and Q.sj4_dr_min > 0.1169491:
        z += 1.026316 * (0.3396572 - Q.lep_z) * (Q.sj4_dr_min - 0.1169491)
    if Q.mass_top40 > 173.1022 and Q.C2_b2 < 0.09209404:
        z += 0.0345299 * (Q.mass_top40 - 173.1022) * (0.09209404 - Q.C2_b2)
    if Q.dc_1_n_disp3 < 3.0 and Q.sdb_2_n > 14.0:
        z += 0.002463597 * (3.0 - Q.dc_1_n_disp3) * (Q.sdb_2_n - 14.0)
    if Q.mass_neutral < 76.15079 and Q.dc_1_sd0_1 < 126.0617:
        z += 4.920629e-07 * (76.15079 - Q.mass_neutral) * (126.0617 - Q.dc_1_sd0_1)
    if Q.dc_1_n_disp3 < 3.0 and Q.dr_52 < 0.4046458:
        z += 0.01957417 * (3.0 - Q.dc_1_n_disp3) * (0.4046458 - Q.dr_52)
    if Q.M2_b2 < 0.0417856 and Q.dc_2_n_disp3 < 2.0:
        z += -0.679201 * (0.0417856 - Q.M2_b2) * (2.0 - Q.dc_2_n_disp3)
    if Q.z_charged_had < 0.6552778 and Q.sj2_mass2 > 6.465318:
        z += 0.008145723 * (0.6552778 - Q.z_charged_had) * (Q.sj2_mass2 - 6.465318)
    if Q.sj2_mass1 > 77.42768 and Q.sv_1_dr < 0.3689526:
        z += -0.0188724 * (Q.sj2_mass1 - 77.42768) * (0.3689526 - Q.sv_1_dr)
    if Q.n_sdz_above_5 < 2.0 and Q.ak02_2_sd0_3 > 0.0:
        z += 0.0364798 * (2.0 - Q.n_sdz_above_5) * (Q.ak02_2_sd0_3 - 0.0)
    if Q.M2_b2 < 0.0417856 and Q.mass_2photon > 9.82024:
        z += 0.128476 * (0.0417856 - Q.M2_b2) * (Q.mass_2photon - 9.82024)
    if Q.mres_sd_rg_b0z005 > 0.6103335 and Q.dzerr_8 > 0.0:
        z += 3.515383 * (Q.mres_sd_rg_b0z005 - 0.6103335) * (Q.dzerr_8 - 0.0)
    if Q.n_pairs_kt_above_3 < 99.0 and Q.sj3_dr13 > 0.3807968:
        z += -0.00113222 * (99.0 - Q.n_pairs_kt_above_3) * (Q.sj3_dr13 - 0.3807968)
    if Q.sj3_pairmax_over_m < 0.8501496 and Q.kt2_2_z_disp3 > 0.01515512:
        z += -5.408732 * (0.8501496 - Q.sj3_pairmax_over_m) * (Q.kt2_2_z_disp3 - 0.01515512)
    if Q.pair_max_lnm2 < 7.075642 and Q.dc_3_charge > 0.1143891:
        z += -0.09145002 * (7.075642 - Q.pair_max_lnm2) * (Q.dc_3_charge - 0.1143891)
    if Q.sj3_pairmax_over_m < 0.8501496 and Q.dc_4_mass_disp3 > 0.0:
        z += -0.02804029 * (0.8501496 - Q.sj3_pairmax_over_m) * (Q.dc_4_mass_disp3 - 0.0)
    if Q.n_sdz_above_5 < 2.0 and Q.iselectron_10 > 0.0:
        z += 0.05496821 * (2.0 - Q.n_sdz_above_5) * (Q.iselectron_10 - 0.0)
    if Q.pz_lnd2 < 0.2100115 and Q.eta_29 > 0.06262207:
        z += 0.2295156 * (0.2100115 - Q.pz_lnd2) * (Q.eta_29 - 0.06262207)
    return z


def neuron_26(Q):
    z = -3.23126e-06
    return z


def neuron_27(Q):
    z = -6.308499e-06
    return z


def neuron_28(Q):
    z = 9.61721e-06
    return z


def neuron_29(Q):
    z = 9.183866e-06
    return z


def neuron_30(Q):
    z = -8.603051e-06
    return z


def neuron_31(Q):
    z = 3.509522e-06
    return z


def neuron_32(Q):
    z = 1.441348e-05
    return z


def neuron_33(Q):
    z = 2.567326e-06
    return z


def neuron_34(Q):
    z = 1.388487e-05
    return z


def neuron_35(Q):
    z = -6.743296e-06
    return z


def neuron_36(Q):
    z = -4.831947e-06
    return z


def neuron_37(Q):
    z = -4.357436e-06
    return z


def neuron_38(Q):
    z = -9.411179e-06
    return z


def neuron_39(Q):
    z = 8.274279e-06
    return z


def neuron_40(Q):
    z = 7.775539e-07
    return z


def neuron_41(Q):
    z = 3.814745e-08
    return z


def neuron_42(Q):
    z = 8.779049e-06
    return z


def neuron_43(Q):
    z = -9.407709e-06
    return z


def neuron_44(Q):
    z = -3.324348e-06
    return z


def neuron_45(Q):
    z = 6.849337e-06
    return z


def neuron_46(Q):
    z = 1.368751e-07
    return z


def neuron_47(Q):
    z = 2.492076e-07
    return z


def neuron_48(Q):
    z = -8.791779e-06
    return z


def neuron_49(Q):
    z = -2.243815e-06
    return z


def neuron_50(Q):
    z = 5.175119e-06
    return z


def neuron_51(Q):
    z = -2.012201e-06
    return z


def neuron_52(Q):
    z = -0.9017627
    if Q.lep_z < 0.221436:
        z += 0.8541373 * Q.lep_z - 0.138257
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -0.4303779 * Q.lep_z + 0.146181
    if Q.z_displaced3 < 0.1342762:
        z += -5.981861 * Q.z_displaced3 + 0.8032216
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.1317973 * Q.lepsj_3_maxsd0 + 0.3180617
    if Q.mass_top40 < 70.88236:
        z += -0.003115491 * Q.mass_top40 + 0.2208334
    if Q.mass_top50 < 79.27954:
        z += -0.0298124 * Q.mass_top50 + 2.576043
    if 79.27954 <= Q.mass_top50 < 115.614:
        z += -0.005849266 * Q.mass_top50 + 0.676257
    if Q.mres_sd_mass_b0z005 < 81.62059:
        z += -0.007441387 * Q.mres_sd_mass_b0z005 + 1.681532
    if 81.62059 <= Q.mres_sd_mass_b0z005 < 131.0776:
        z += -0.003260788 * Q.mres_sd_mass_b0z005 + 1.340309
    if 131.0776 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += -0.02029823 * Q.mres_sd_mass_b0z005 + 3.573535
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += 0.004180599 * Q.mres_sd_mass_b0z005 - 0.3412229
    if Q.mass >= 182.8592:
        z += -0.01385931 * Q.mass + 2.534304
    if Q.lne_16 >= 0.4468807:
        z += -0.1450277 * Q.lne_16 + 0.06481007
    if Q.lep_ptrel < 12.15228:
        z += 0.0394996 * Q.lep_ptrel - 0.5089409
    if 12.15228 <= Q.lep_ptrel < 18.7678:
        z += 0.001906928 * Q.lep_ptrel - 0.05210414
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += 0.01429582 * Q.lep_ptrel - 0.2846165
    if Q.lep_ptrel >= 27.3236:
        z += 0.0123889 * Q.lep_ptrel - 0.2325123
    if Q.lepsj_2_dr < 0.103005:
        z += 4.499006 * Q.lepsj_2_dr - 0.46342
    if Q.z_neutral_had < 0.3788785:
        z += 0.2894128 * Q.z_neutral_had - 0.1096523
    if Q.sv_1_sd0_sum < 125.2765:
        z += 0.001874869 * Q.sv_1_sd0_sum - 0.2348771
    if Q.sjq_2_2_k1 < -0.6337755:
        z += -1.817869 * Q.sjq_2_2_k1 - 1.152121
    if Q.sjq_2_2_k1 >= 0.6477929:
        z += 1.470273 * Q.sjq_2_2_k1 - 0.9524321
    if Q.pz_lnd0 < 0.1122946:
        z += -0.4403901 * Q.pz_lnd0 + 0.04945341
    if Q.n_s3d_above_3 < 3.0:
        z += -0.09387428 * Q.n_s3d_above_3 + 0.2816228
    if Q.lep_dr < 0.3014662:
        z += -2.507758 * Q.lep_dr + 0.7560042
    if Q.lep_dr >= 0.4191372:
        z += -0.8640171 * Q.lep_dr + 0.3621417
    if Q.D2_b05 < 1.45443:
        z += 0.6496252 * Q.D2_b05 - 0.9448345
    if Q.n_lund < 7.0:
        z += -0.1324395 * Q.n_lund + 0.9270763
    if Q.lep_iso < 1.362094:
        z += -0.04213795 * Q.lep_iso + 0.05739587
    if Q.n_lepton < 1.0:
        z += 0.5921237 * Q.n_lepton - 0.5921237
    if Q.sj2_mass2 < 21.7295:
        z += 0.01257484 * Q.sj2_mass2 - 0.2732449
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.003117607 * Q.n_pairs_kt_above_1 + 0.2494086
    if Q.dc_1_n_lep < 1.0:
        z += -0.2922761 * Q.dc_1_n_lep + 0.2922761
    if Q.jd_sum_abs_sd0_top5 < 84.11398:
        z += 0.003572921 * Q.jd_sum_abs_sd0_top5 - 0.3005326
    if Q.sj3_pair_mass_min >= 80.02563:
        z += 0.02381081 * Q.sj3_pair_mass_min - 1.905475
    if Q.lund3_lndelta >= -2.817283:
        z += -0.05159704 * Q.lund3_lndelta - 0.1453635
    if Q.jet_charge_k03 >= -0.2409073:
        z += -0.08679879 * Q.jet_charge_k03 - 0.02091046
    if Q.z_charged_had < 0.1891627:
        z += 2.264385 * Q.z_charged_had - 0.4283372
    if Q.ak02_2_z < 0.2837384:
        z += -2.554322 * Q.ak02_2_z + 0.7247594
    if Q.sjf_2_2_max3d < 3.029824:
        z += -0.05928928 * Q.sjf_2_2_max3d + 0.1796361
    if Q.n_dr_0p2_0p4 < 21.0:
        z += 0.01912607 * Q.n_dr_0p2_0p4 - 0.4016474
    if Q.sj2_dr < 0.3740528:
        z += -0.6866233 * Q.sj2_dr + 0.2568334
    if Q.e3_b2 < 7.104488e-06:
        z += -21227.63 * Q.e3_b2 + 0.5152813
    if 7.104488e-06 <= Q.e3_b2 < 0.0004126585:
        z += -898.696 * Q.e3_b2 + 0.3708546
    if Q.D3_b2 < 0.001184159:
        z += 36.07462 * Q.D3_b2 - 0.04271809
    if Q.sjf_2_1_z_d3 < 0.03986247:
        z += 0.7308766 * Q.sjf_2_1_z_d3 - 0.02913455
    if Q.sjq_3_3_k1 >= 0.6535817:
        z += 0.2660399 * Q.sjq_3_3_k1 - 0.1738788
    if Q.tau32_b2 < 0.3377343:
        z += -0.02913458 * Q.tau32_b2 + 0.009839747
    if Q.mass_charged < 65.0404:
        z += 0.003261279 * Q.mass_charged - 0.2121149
    if Q.mass_2charged < 6.767937:
        z += 0.01092353 * Q.mass_2charged - 0.07392974
    if Q.mass_2charged >= 14.28253:
        z += -0.007520577 * Q.mass_2charged + 0.1074129
    if Q.N3_b05 >= 0.7591346:
        z += -0.07586795 * Q.N3_b05 + 0.05759399
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.005839702 * Q.n_pairs_kt_above_3 + 0.1635117
    if Q.mass_top15 >= 133.1648:
        z += 0.0009751058 * Q.mass_top15 - 0.1298498
    if Q.sdb_2_n >= 9.0:
        z += -0.04503712 * Q.sdb_2_n + 0.4053341
    if Q.sj2_mass1 >= 91.2852:
        z += 0.002591764 * Q.sj2_mass1 - 0.2365897
    if Q.sj3_mass1 >= 36.36236:
        z += -0.01384196 * Q.sj3_mass1 + 0.5033265
    if Q.D2 < 1.421532:
        z += 0.07522704 * Q.D2 - 0.1069377
    if Q.tau21_b2 < 0.3565533:
        z += -2.851608 * Q.tau21_b2 + 1.01675
    if Q.ak02_1_z >= 0.6342743:
        z += -1.276732 * Q.ak02_1_z + 0.8097985
    if Q.tau4 < 0.01143413:
        z += -69.09957 * Q.tau4 + 0.7900935
    if Q.n_dr_0p4_up < 10.0:
        z += 0.02498032 * Q.n_dr_0p4_up - 0.2498032
    if Q.ak02_3_z >= 0.1014193:
        z += -1.29639 * Q.ak02_3_z + 0.131479
    if Q.lund_max_lndelta >= -0.615738:
        z += 0.7701138 * Q.lund_max_lndelta + 0.4741883
    if Q.N2_b2 >= 0.08291719:
        z += 1.727152 * Q.N2_b2 - 0.1432106
    if Q.z_displaced3 < 0.1342762 and Q.N2 > 0.1824346:
        z += 8.111054 * (0.1342762 - Q.z_displaced3) * (Q.N2 - 0.1824346)
    if Q.lep_z < 0.3396572 and Q.n_charged_pt_above_1 > 11.0:
        z += 0.1218997 * (0.3396572 - Q.lep_z) * (Q.n_charged_pt_above_1 - 11.0)
    if Q.lepsj_3_maxsd0 < 2.413264 and Q.lep_dr < 0.3014662:
        z += 0.04330778 * (2.413264 - Q.lepsj_3_maxsd0) * (0.3014662 - Q.lep_dr)
    if Q.lep_z < 0.3396572 and Q.sj3_pair_mass_min > 48.40547:
        z += -0.01579322 * (0.3396572 - Q.lep_z) * (Q.sj3_pair_mass_min - 48.40547)
    if Q.mres_sd_mass_b0z005 < 159.9242 and Q.tau21_b2 < 0.3565533:
        z += -0.03224804 * (159.9242 - Q.mres_sd_mass_b0z005) * (0.3565533 - Q.tau21_b2)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.tau32 < 0.7544983:
        z += -0.01032856 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.7544983 - Q.tau32)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.sjf_2_2_z_d3 < 0.03385157:
        z += 0.03859485 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.03385157 - Q.sjf_2_2_z_d3)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.pz_lnd1 > 0.05758556:
        z += 0.003111716 * (Q.mres_sd_mass_b0z005 - 81.62059) * (Q.pz_lnd1 - 0.05758556)
    if Q.mres_sd_mass_b0z005 < 159.9242 and Q.mass_displaced3 < 1.251841:
        z += -8.641154e-05 * (159.9242 - Q.mres_sd_mass_b0z005) * (1.251841 - Q.mass_displaced3)
    if Q.mres_sd_mass_b0z005 < 159.9242 and Q.mass_displaced3 > 3.208089:
        z += -0.0002744828 * (159.9242 - Q.mres_sd_mass_b0z005) * (Q.mass_displaced3 - 3.208089)
    if Q.z_neutral_had < 0.3788785 and Q.ecf_g41 > 6.216954e-05:
        z += 1405.591 * (0.3788785 - Q.z_neutral_had) * (Q.ecf_g41 - 6.216954e-05)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.ak02_3_z < 0.07650476:
        z += 0.1258392 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.07650476 - Q.ak02_3_z)
    if Q.lepsj_2_dr < 0.103005 and Q.C3_b05 < 0.2250047:
        z += 6.734058 * (0.103005 - Q.lepsj_2_dr) * (0.2250047 - Q.C3_b05)
    if Q.mres_sd_mass_b0z005 < 131.0776 and Q.sip_3d_3 < 23.30856:
        z += -4.554017e-05 * (131.0776 - Q.mres_sd_mass_b0z005) * (23.30856 - Q.sip_3d_3)
    if Q.n_s3d_above_3 < 3.0 and Q.dzerr_0 > 0.02830505:
        z += -0.1067499 * (3.0 - Q.n_s3d_above_3) * (Q.dzerr_0 - 0.02830505)
    if Q.lep_dr < 0.3014662 and Q.sdb_2_z > 0.1530389:
        z += 0.8001209 * (0.3014662 - Q.lep_dr) * (Q.sdb_2_z - 0.1530389)
    if Q.z_displaced3 < 0.1342762 and Q.sip_3d_3 < 577.991:
        z += -0.007023917 * (0.1342762 - Q.z_displaced3) * (577.991 - Q.sip_3d_3)
    if Q.lund3_lndelta > -2.817283 and Q.sv_1_dr < 0.3689526:
        z += 0.2210095 * (Q.lund3_lndelta - -2.817283) * (0.3689526 - Q.sv_1_dr)
    if Q.sj2_mass2 < 21.7295 and Q.nca_sj4_pairmax_over_mass < 0.7817308:
        z += 0.05081765 * (21.7295 - Q.sj2_mass2) * (0.7817308 - Q.nca_sj4_pairmax_over_mass)
    if Q.z_neutral_had < 0.3788785 and Q.pz_lnkt1 > 0.03633353:
        z += -7.874087 * (0.3788785 - Q.z_neutral_had) * (Q.pz_lnkt1 - 0.03633353)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_2_prod_k05 > -0.5057096:
        z += -0.009256539 * (21.0 - Q.n_dr_0p2_0p4) * (Q.sjq_2_prod_k05 - -0.5057096)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_3_sumabs_k1 > 0.3376898:
        z += 0.01342769 * (21.0 - Q.n_dr_0p2_0p4) * (Q.sjq_3_sumabs_k1 - 0.3376898)
    if Q.z_displaced3 < 0.1342762 and Q.n_lund_kt_above_1 > 3.0:
        z += 0.08131465 * (0.1342762 - Q.z_displaced3) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.lepsj_2_dr < 0.103005 and Q.z_photon > 0.3318968:
        z += 3.075701 * (0.103005 - Q.lepsj_2_dr) * (Q.z_photon - 0.3318968)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_3_prod_k1 < 0.1145956:
        z += 0.02651823 * (21.0 - Q.n_dr_0p2_0p4) * (0.1145956 - Q.sjq_3_prod_k1)
    if Q.mass > 182.8592 and Q.sum_e < 1135.303:
        z += -1.578779e-06 * (Q.mass - 182.8592) * (1135.303 - Q.sum_e)
    if Q.z_displaced3 < 0.1342762 and Q.jd_3d_6 < 5.367501:
        z += 1.132059 * (0.1342762 - Q.z_displaced3) * (5.367501 - Q.jd_3d_6)
    if Q.n_lund < 7.0 and Q.jd_3d_6 < 2.591685:
        z += -0.0223959 * (7.0 - Q.n_lund) * (2.591685 - Q.jd_3d_6)
    if Q.D3_b2 < 0.001184159 and Q.jd_3d_6 < 5.367501:
        z += -36.2875 * (0.001184159 - Q.D3_b2) * (5.367501 - Q.jd_3d_6)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.jd_3d_6 < 3.151933:
        z += 0.001017162 * (Q.mres_sd_mass_b0z005 - 81.62059) * (3.151933 - Q.jd_3d_6)
    if Q.mass_charged < 65.0404 and Q.isphoton_19 < 1.0:
        z += 0.001369551 * (65.0404 - Q.mass_charged) * (1.0 - Q.isphoton_19)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.jet_e > 926.0771:
        z += 2.00118e-06 * (21.0 - Q.n_dr_0p2_0p4) * (Q.jet_e - 926.0771)
    if Q.D2 < 1.421532 and Q.ak02_2_sd0_1 > 18.03212:
        z += -1.682039e-05 * (1.421532 - Q.D2) * (Q.ak02_2_sd0_1 - 18.03212)
    if Q.sdb_2_n > 9.0 and Q.D2_b2 < 2.01745:
        z += 0.03580905 * (Q.sdb_2_n - 9.0) * (2.01745 - Q.D2_b2)
    if Q.z_displaced3 < 0.1342762 and Q.jd_3d_6 < 30.57088:
        z += -0.178261 * (0.1342762 - Q.z_displaced3) * (30.57088 - Q.jd_3d_6)
    if Q.mres_sd_mass_b0z005 < 131.0776 and Q.D2_b2 < 1.204663:
        z += -0.005619396 * (131.0776 - Q.mres_sd_mass_b0z005) * (1.204663 - Q.D2_b2)
    if Q.lep_ptrel < 12.15228 and Q.D2_b2 < 5.460234:
        z += 0.0004377723 * (12.15228 - Q.lep_ptrel) * (5.460234 - Q.D2_b2)
    return z


def neuron_53(Q):
    z = -1.889363e-06
    return z


def neuron_54(Q):
    z = 1.190659e-05
    return z


def neuron_55(Q):
    z = -2.06694e-05
    return z


def neuron_56(Q):
    z = -7.072322e-07
    return z


def neuron_57(Q):
    z = -1.959645e-06
    return z


def neuron_58(Q):
    z = -3.61715e-06
    return z


def neuron_59(Q):
    z = 6.029333e-06
    return z


def neuron_60(Q):
    z = -4.666764e-06
    return z


def neuron_61(Q):
    z = 5.840563e-06
    return z


def neuron_62(Q):
    z = 9.422978e-07
    return z


def neuron_63(Q):
    z = -8.308511e-06
    return z


def neuron_64(Q):
    z = 1.258627e-06
    return z


def neuron_65(Q):
    z = 5.791052e-06
    return z


def neuron_66(Q):
    z = 2.709777e-06
    return z


def neuron_67(Q):
    z = -5.219509e-06
    return z


def neuron_68(Q):
    z = 4.694148
    if Q.tau32 < 0.7981752:
        z += -0.6202956 * Q.tau32 + 0.4951045
    if 6.983043 <= Q.lep_ptrel < 18.7678:
        z += -0.01736527 * Q.lep_ptrel + 0.1212624
    if 18.7678 <= Q.lep_ptrel < 43.20788:
        z += 0.01925944 * Q.lep_ptrel - 0.5661027
    if Q.lep_ptrel >= 43.20788:
        z += 8.631311e-05 * Q.lep_ptrel + 0.2623274
    if Q.pair_mean_lnm2 < 3.38061:
        z += 0.03681793 * Q.pair_mean_lnm2 - 0.1244671
    if Q.n_s3d_above_3 >= 5.0:
        z += -0.01689681 * Q.n_s3d_above_3 + 0.08448403
    if Q.pz_lnd2 < 0.1339824:
        z += 1.983372 * Q.pz_lnd2 - 0.2657369
    if Q.e3_b2 < 0.0004126585:
        z += 1254.336 * Q.e3_b2 - 0.5176125
    if Q.sj3_pair_mass_max < 56.73313:
        z += 0.02242794 * Q.sj3_pair_mass_max - 1.272407
    if 83.63398 <= Q.sj3_pair_mass_max < 128.6605:
        z += -0.009284915 * Q.sj3_pair_mass_max + 0.7765344
    if Q.sj3_pair_mass_max >= 128.6605:
        z += 0.01166683 * Q.sj3_pair_mass_max - 1.919128
    if Q.mass < 90.08945:
        z += 0.00264373 * Q.mass - 1.424681
    if 90.08945 <= Q.mass < 110.2019:
        z += -0.0052882 * Q.mass - 0.7100977
    if 110.2019 <= Q.mass < 117.4867:
        z += 0.02328099 * Q.mass - 3.858476
    if 117.4867 <= Q.mass < 164.4374:
        z += 0.02392443 * Q.mass - 3.934072
    if Q.sj4_pair_mass_max >= 69.83554:
        z += -0.002377128 * Q.sj4_pair_mass_max + 0.166008
    if Q.z_charged_had >= 0.265564:
        z += 0.6040737 * Q.z_charged_had - 0.1604202
    if Q.z_displaced3 < 0.1061578:
        z += 0.5022371 * Q.z_displaced3 - 0.05331639
    if Q.z_displaced3 >= 0.1681173:
        z += 0.09627517 * Q.z_displaced3 - 0.01618552
    if Q.M2_b2 < 0.05966366:
        z += 6.40952 * Q.M2_b2 - 0.3824154
    if Q.mres_sd_mass_b2z01 < 55.13212:
        z += -0.005000745 * Q.mres_sd_mass_b2z01 - 0.9862391
    if 55.13212 <= Q.mres_sd_mass_b2z01 < 76.1529:
        z += 0.04395608 * Q.mres_sd_mass_b2z01 - 3.685332
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 118.2746:
        z += -0.01925655 * Q.mres_sd_mass_b2z01 + 1.128493
    if 118.2746 <= Q.mres_sd_mass_b2z01 < 130.0968:
        z += 0.01953062 * Q.mres_sd_mass_b2z01 - 3.459046
    if 130.0968 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.03266941 * Q.mres_sd_mass_b2z01 - 5.168361
    z += -0.06116175 * Q.n_particles
    if Q.dc_split2_dr >= 0.3591078:
        z += -2.384324 * Q.dc_split2_dr + 0.8562293
    if Q.lep_iso < 0.4381892:
        z += 1.238536 * Q.lep_iso - 0.5427129
    if Q.lep_dr < 0.04607888:
        z += -7.128165 * Q.lep_dr + 0.3284579
    if Q.lep_dr >= 0.114659:
        z += 0.1020358 * Q.lep_dr - 0.01169932
    if Q.sj3_pair_mass_min < 80.02563:
        z += 0.008457892 * Q.sj3_pair_mass_min - 0.6768481
    if Q.sdb_2_n < 7.0:
        z += -0.03881876 * Q.sdb_2_n + 0.2717313
    if Q.sdb_2_n >= 12.0:
        z += -0.02869951 * Q.sdb_2_n + 0.3443941
    if Q.M3 < 0.03259227:
        z += -13.46749 * Q.M3 + 0.438936
    if Q.mass_top20 >= 130.7093:
        z += -0.01186421 * Q.mass_top20 + 1.550762
    if Q.jet_abs_eta >= 0.5325716:
        z += 0.3494466 * Q.jet_abs_eta - 0.1861054
    z += 0.03495234 * Q.n_photon
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.002814372 * Q.n_pairs_kt_above_1 + 0.2251498
    if Q.sjq_2_2_nch < 3.0:
        z += 0.1296818 * Q.sjq_2_2_nch - 0.3890455
    if Q.sj2_dr < 0.2403736:
        z += 0.7448348 * Q.sj2_dr - 0.1790386
    if Q.tau5 < 0.05613495:
        z += -16.09579 * Q.tau5 + 0.9035364
    if Q.ktd_ln_d34 >= -9.359695:
        z += 0.1546685 * Q.ktd_ln_d34 + 1.44765
    if Q.D2_b2 < 1.774856:
        z += 0.1573507 * Q.D2_b2 - 0.279275
    if Q.nca_sj4_pair_mass_2nd >= 72.72359:
        z += -0.01718524 * Q.nca_sj4_pair_mass_2nd + 1.249772
    if Q.ak02_dr12 >= 0.3882673:
        z += -0.34096 * Q.ak02_dr12 + 0.1323836
    if Q.max_abs_dz < 0.4436035:
        z += 0.06641699 * Q.max_abs_dz - 0.02946281
    if Q.lepsj_2_maxsd0 < 1163.885:
        z += 9.425726e-05 * Q.lepsj_2_maxsd0 - 0.1097046
    if Q.sjf_3_1_n_d3 >= 6.0:
        z += 0.05800946 * Q.sjf_3_1_n_d3 - 0.3480568
    if 47.97531 <= Q.mres_sd_mass_b0z02 < 86.60355:
        z += 0.004029123 * Q.mres_sd_mass_b0z02 - 0.1932984
    if 86.60355 <= Q.mres_sd_mass_b0z02 < 134.2023:
        z += -0.006328819 * Q.mres_sd_mass_b0z02 + 0.7037361
    if Q.mres_sd_mass_b0z02 >= 134.2023:
        z += 0.002933367 * Q.mres_sd_mass_b0z02 - 0.5392707
    if 86.14266 <= Q.mres_pruned_mass < 124.3145:
        z += -0.009157327 * Q.mres_pruned_mass + 0.7888365
    if 124.3145 <= Q.mres_pruned_mass < 171.3819:
        z += 0.005996197 * Q.mres_pruned_mass - 1.094967
    if Q.mres_pruned_mass >= 171.3819:
        z += 0.001359204 * Q.mres_pruned_mass - 0.30027
    if Q.pz_lnd3 >= 0.01024929:
        z += 0.3855671 * Q.pz_lnd3 - 0.00395179
    if Q.n_dr_0p2_0p4 >= 7.0:
        z += -0.003559842 * Q.n_dr_0p2_0p4 + 0.02491889
    if Q.mres_sd_rg_b2z01 >= 0.457968:
        z += -1.284303 * Q.mres_sd_rg_b2z01 + 0.5881695
    if Q.n_sdz_above_5 < 3.0:
        z += -0.05945904 * Q.n_sdz_above_5 + 0.1783771
    if Q.sum_z_dr2_top2 < 0.005938474:
        z += -24.30055 * Q.sum_z_dr2_top2 + 0.1443082
    if Q.mass_top10 < 103.4976:
        z += 0.001116202 * Q.mass_top10 - 0.1155243
    if Q.sdb_2_z < 0.1530389:
        z += 0.9795502 * Q.sdb_2_z - 0.1499093
    if Q.jd_3d_5 < 4.004982:
        z += 0.08673749 * Q.jd_3d_5 - 0.3473821
    if Q.tau32 < 0.7981752 and Q.sj3_dr_min < 0.3190414:
        z += 1.875083 * (0.7981752 - Q.tau32) * (0.3190414 - Q.sj3_dr_min)
    if Q.pair_mean_lnm2 < 3.38061 and Q.z_top30_slots > 0.8316085:
        z += 0.2100434 * (3.38061 - Q.pair_mean_lnm2) * (Q.z_top30_slots - 0.8316085)
    if Q.e3_b2 < 0.0004126585 and Q.lnptrel_25 > -6.691703:
        z += 314.2224 * (0.0004126585 - Q.e3_b2) * (Q.lnptrel_25 - -6.691703)
    if Q.e3_b2 < 0.0004126585 and Q.ak02_dr23 > 0.3194837:
        z += 1408.244 * (0.0004126585 - Q.e3_b2) * (Q.ak02_dr23 - 0.3194837)
    if Q.mass < 117.4867 and Q.ak02_2_n_lep < 1.0:
        z += -0.000124937 * (117.4867 - Q.mass) * (1.0 - Q.ak02_2_n_lep)
    if Q.z_charged_had > 0.265564 and Q.z_photon > 0.09084052:
        z += 0.6161464 * (Q.z_charged_had - 0.265564) * (Q.z_photon - 0.09084052)
    if Q.sj4_pair_mass_max > 69.83554 and Q.nca_sj4_pair2nd_over_mass > 0.4488456:
        z += 0.02938354 * (Q.sj4_pair_mass_max - 69.83554) * (Q.nca_sj4_pair2nd_over_mass - 0.4488456)
    if Q.mass < 117.4867 and Q.n_lund_kt_above_1 > 3.0:
        z += -0.0001942064 * (117.4867 - Q.mass) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.n_s3d_above_3 > 5.0 and Q.mean_eta > -0.02869681:
        z += 0.08610445 * (Q.n_s3d_above_3 - 5.0) * (Q.mean_eta - -0.02869681)
    if Q.sj3_pair_mass_max > 128.6605 and Q.jet_e > 551.9263:
        z += 4.511829e-06 * (Q.sj3_pair_mass_max - 128.6605) * (Q.jet_e - 551.9263)
    if Q.mass < 117.4867 and Q.D2_b2 < 1.380731:
        z += 0.008058498 * (117.4867 - Q.mass) * (1.380731 - Q.D2_b2)
    if Q.e3_b2 < 0.0004126585 and Q.sj3_mass1 > 17.31003:
        z += 20.05597 * (0.0004126585 - Q.e3_b2) * (Q.sj3_mass1 - 17.31003)
    if Q.sj4_pair_mass_max > 69.83554 and Q.dc_3_z < 0.2037349:
        z += -0.03145654 * (Q.sj4_pair_mass_max - 69.83554) * (0.2037349 - Q.dc_3_z)
    if Q.n_s3d_above_3 > 5.0 and Q.sip_3d_2 < 226.3008:
        z += -0.0001639865 * (Q.n_s3d_above_3 - 5.0) * (226.3008 - Q.sip_3d_2)
    if Q.lep_ptrel > 6.983043 and Q.C2_b2 < 0.1682645:
        z += 0.03414988 * (Q.lep_ptrel - 6.983043) * (0.1682645 - Q.C2_b2)
    if Q.sj3_pair_mass_min < 80.02563 and Q.mass_displaced5 > 0.0:
        z += 0.0001351307 * (80.02563 - Q.sj3_pair_mass_min) * (Q.mass_displaced5 - 0.0)
    if Q.sj3_pair_mass_max > 128.6605 and Q.e4 > 7.184834e-06:
        z += 208.7861 * (Q.sj3_pair_mass_max - 128.6605) * (Q.e4 - 7.184834e-06)
    if Q.z_displaced3 < 0.1061578 and Q.ak02_3_n_lep > 0.0:
        z += 0.6252539 * (0.1061578 - Q.z_displaced3) * (Q.ak02_3_n_lep - 0.0)
    if Q.pz_lnd2 < 0.1339824 and Q.sj3_dr13 > 0.4691911:
        z += -5.979494 * (0.1339824 - Q.pz_lnd2) * (Q.sj3_dr13 - 0.4691911)
    if Q.mass < 164.4374 and Q.sjf_4_3_z_d3 > 0.0:
        z += -0.01829581 * (164.4374 - Q.mass) * (Q.sjf_4_3_z_d3 - 0.0)
    if Q.D2_b2 < 1.774856 and Q.sjf_4_4_z_d3 > 0.0:
        z += -0.9998201 * (1.774856 - Q.D2_b2) * (Q.sjf_4_4_z_d3 - 0.0)
    if Q.tau32 < 0.7981752 and Q.sjq_3_sumabs_k1 > 0.4372817:
        z += 0.5834617 * (0.7981752 - Q.tau32) * (Q.sjq_3_sumabs_k1 - 0.4372817)
    if Q.z_displaced3 < 0.1061578 and Q.lepsj_2_maxsd0 < 0.7828545:
        z += -5.08956 * (0.1061578 - Q.z_displaced3) * (0.7828545 - Q.lepsj_2_maxsd0)
    if Q.tau5 < 0.05613495 and Q.dc_ntag > 0.0:
        z += -0.06779569 * (0.05613495 - Q.tau5) * (Q.dc_ntag - 0.0)
    if Q.z_charged_had > 0.265564 and Q.pz_lnkt3 > 0.0:
        z += 3.330634 * (Q.z_charged_had - 0.265564) * (Q.pz_lnkt3 - 0.0)
    if Q.lep_dr < 0.04607888 and Q.lepsj_2_maxsd0 < 0.7828545:
        z += 11.60522 * (0.04607888 - Q.lep_dr) * (0.7828545 - Q.lepsj_2_maxsd0)
    if Q.sj4_pair_mass_max > 69.83554 and Q.sv_1_dr < 0.3689526:
        z += -0.00555177 * (Q.sj4_pair_mass_max - 69.83554) * (0.3689526 - Q.sv_1_dr)
    if Q.mass < 117.4867 and Q.C2_b2 > 0.01891146:
        z += -0.04777913 * (117.4867 - Q.mass) * (Q.C2_b2 - 0.01891146)
    if Q.z_displaced3 < 0.1061578 and Q.dr_29 < 0.02258554:
        z += 52.23215 * (0.1061578 - Q.z_displaced3) * (0.02258554 - Q.dr_29)
    if Q.e3_b2 < 0.0004126585 and Q.n_lund_kt_above_5 > 2.0:
        z += 229.344 * (0.0004126585 - Q.e3_b2) * (Q.n_lund_kt_above_5 - 2.0)
    if Q.sjf_3_1_n_d3 > 6.0 and Q.dr_10 < 0.3989771:
        z += -0.1022438 * (Q.sjf_3_1_n_d3 - 6.0) * (0.3989771 - Q.dr_10)
    if Q.z_displaced3 < 0.1061578 and Q.n_lund > 10.0:
        z += -0.5278397 * (0.1061578 - Q.z_displaced3) * (Q.n_lund - 10.0)
    if Q.mres_sd_mass_b0z02 > 47.97531 and Q.ismuon_4 > 0.0:
        z += 0.0005991903 * (Q.mres_sd_mass_b0z02 - 47.97531) * (Q.ismuon_4 - 0.0)
    if Q.M3 < 0.03259227 and Q.jd_3d_5 < 4.004982:
        z += -2.093463 * (0.03259227 - Q.M3) * (4.004982 - Q.jd_3d_5)
    if Q.ak02_dr12 > 0.3882673 and Q.dr_77 > 0.0:
        z += 1.146461 * (Q.ak02_dr12 - 0.3882673) * (Q.dr_77 - 0.0)
    if Q.z_charged_had > 0.265564 and Q.isnhad_29 > 0.0:
        z += -0.03301251 * (Q.z_charged_had - 0.265564) * (Q.isnhad_29 - 0.0)
    if Q.sj3_pair_mass_max > 128.6605 and Q.sjq_3_sumabs_k03 > 0.8323072:
        z += -0.002029924 * (Q.sj3_pair_mass_max - 128.6605) * (Q.sjq_3_sumabs_k03 - 0.8323072)
    return z


def neuron_69(Q):
    z = -6.926143e-06
    return z


def neuron_70(Q):
    z = 0.05005557
    if Q.lep_z < 0.221436:
        z += 0.4829356 * Q.lep_z - 0.09072462
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -0.1371557 * Q.lep_z + 0.04658592
    if Q.n_s3d_above_3 < 1.0:
        z += -0.04548711 * Q.n_s3d_above_3 + 0.331069
    if 1.0 <= Q.n_s3d_above_3 < 4.0:
        z += -0.06148761 * Q.n_s3d_above_3 + 0.3470695
    if 4.0 <= Q.n_s3d_above_3 < 10.0:
        z += -0.04085392 * Q.n_s3d_above_3 + 0.2645347
    if Q.n_s3d_above_3 >= 10.0:
        z += -0.0160005 * Q.n_s3d_above_3 + 0.0160005
    if Q.lepsj_3_n_d3 < 2.0:
        z += -0.03749511 * Q.lepsj_3_n_d3 + 0.07499022
    if Q.n_pairs_kt_above_1 < 58.0:
        z += -0.003493045 * Q.n_pairs_kt_above_1 + 0.2025966
    if Q.sip_3d_2 < 447.0873:
        z += 0.0001153908 * Q.sip_3d_2 - 0.05158976
    if 48.40547 <= Q.sj3_pair_mass_min < 80.02563:
        z += -0.008637961 * Q.sj3_pair_mass_min + 0.4181246
    if Q.sj3_pair_mass_min >= 80.02563:
        z += -0.001373523 * Q.sj3_pair_mass_min - 0.1632166
    if 69.14897 <= Q.mres_sd_mass_b0z005 < 86.65765:
        z += -0.002849857 * Q.mres_sd_mass_b0z005 + 0.1970646
    if 86.65765 <= Q.mres_sd_mass_b0z005 < 125.8718:
        z += 0.0002041785 * Q.mres_sd_mass_b0z005 - 0.06759089
    if 125.8718 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 8.902433e-05 * Q.mres_sd_mass_b0z005 - 0.05309623
    if 159.9242 <= Q.mres_sd_mass_b0z005 < 178.8957:
        z += -0.008547116 * Q.mres_sd_mass_b0z005 + 1.328032
    if Q.mres_sd_mass_b0z005 >= 178.8957:
        z += -0.009227171 * Q.mres_sd_mass_b0z005 + 1.449691
    if Q.N2 >= 0.2875658:
        z += 1.278462 * Q.N2 - 0.3676418
    if Q.mass < 79.47361:
        z += 0.003415876 * Q.mass - 0.271472
    if 120.653 <= Q.mass < 149.0507:
        z += -0.009886788 * Q.mass + 1.192871
    if Q.mass >= 149.0507:
        z += 0.0125051 * Q.mass - 2.144654
    if Q.lep_iso < 0.4381892:
        z += -1.003904 * Q.lep_iso + 0.1904947
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += 0.2699466 * Q.lep_iso - 0.3676927
    if Q.z_displaced5 < 0.08090366:
        z += 1.524067 * Q.z_displaced5 - 0.4269473
    if 0.08090366 <= Q.z_displaced5 < 0.2801368:
        z += 0.6257787 * Q.z_displaced5 - 0.3542725
    if Q.z_displaced5 >= 0.2801368:
        z += -0.8982882 * Q.z_displaced5 + 0.07267481
    if Q.lep_ptrel >= 27.3236:
        z += -0.008462423 * Q.lep_ptrel + 0.2312239
    if Q.lepsj_3_maxsd0 < 1.529925:
        z += -0.06261092 * Q.lepsj_3_maxsd0 + 0.09579001
    if Q.n_lepton < 1.0:
        z += 0.3321457 * Q.n_lepton - 0.3321457
    if Q.n_photon < 16.0:
        z += -0.05233426 * Q.n_photon + 0.8373482
    if Q.n_charged_had >= 21.0:
        z += -0.009448263 * Q.n_charged_had + 0.1984135
    if Q.lepsj_2_dr < 0.06573337:
        z += 1.402807 * Q.lepsj_2_dr - 0.09221121
    if Q.lep_dr >= 0.08178299:
        z += -0.2244161 * Q.lep_dr + 0.01835342
    z += -0.02955764 * Q.pz_lnkt1
    if Q.nca_sj4_pair_mass_2nd >= 83.41384:
        z += -0.002947891 * Q.nca_sj4_pair_mass_2nd + 0.2458949
    if Q.tau43 < 0.7947468:
        z += 0.5303866 * Q.tau43 - 0.421523
    if Q.lund1_lndelta >= -0.3624079:
        z += 0.02429897 * Q.lund1_lndelta + 0.008806138
    if Q.pz_lnd0 < 0.1122946:
        z += -3.959259 * Q.pz_lnd0 + 0.4446033
    if Q.kt2_dr12 < 0.2738646:
        z += -0.03174007 * Q.kt2_dr12 + 0.008692483
    if Q.kt2_dr12 >= 0.6465173:
        z += 1.079837 * Q.kt2_dr12 - 0.6981331
    if Q.sjq_3_2_k1 < -0.6014774:
        z += -1.6366 * Q.sjq_3_2_k1 - 0.984378
    if Q.tau3 < 0.01470468:
        z += -8.520743 * Q.tau3 + 0.1252948
    if Q.e3_b2 < 4.696862e-06:
        z += 38716.98 * Q.e3_b2 - 0.1818483
    if Q.lam2 < 0.02148541:
        z += -11.83375 * Q.lam2 + 0.2542531
    if Q.sj2_mass2 < 1.851735:
        z += -0.1646899 * Q.sj2_mass2 + 0.3049622
    if Q.sjf_4_n2disp >= 1.0:
        z += 0.1564829 * Q.sjf_4_n2disp - 0.1564829
    if Q.ak02_1_n_disp3 >= 3.0:
        z += 0.02448959 * Q.ak02_1_n_disp3 - 0.07346876
    if Q.ak02_dr23 < 0.2385164:
        z += -0.3043773 * Q.ak02_dr23 + 0.07259899
    if Q.sjq_3_3_k1 >= 0.6535817:
        z += 1.080688 * Q.sjq_3_3_k1 - 0.706318
    if Q.sj3_dr12 < 0.3078242:
        z += -0.01006793 * Q.sj3_dr12 + 0.003099152
    if Q.mres_sd_zg_b0z005 < 0.1828918:
        z += -0.9948704 * Q.mres_sd_zg_b0z005 + 0.1819536
    if Q.ecf_g42 < 1.874473e-05:
        z += -14275.94 * Q.ecf_g42 + 0.2675988
    if Q.mass_2charged >= 24.4079:
        z += 0.002215758 * Q.mass_2charged - 0.05408199
    if Q.sjq_3_2_nch >= 5.0:
        z += -0.001021578 * Q.sjq_3_2_nch + 0.005107889
    if Q.mass_charged < 22.56857:
        z += -0.05836486 * Q.mass_charged + 1.317211
    if Q.mass_neutral < 49.25971:
        z += 0.007532812 * Q.mass_neutral - 0.3710641
    if Q.sdb_2_z < 0.2509165:
        z += 0.6724493 * Q.sdb_2_z - 0.1687286
    if Q.sdb_2_z >= 0.3448514:
        z += 0.6173229 * Q.sdb_2_z - 0.2128847
    if Q.sv_2_n >= 1.0:
        z += -0.1525959 * Q.sv_2_n + 0.1525959
    if Q.sj2_zsoft < 0.08996752:
        z += 4.67996 * Q.sj2_zsoft - 0.4210444
    if Q.mass_top20 < 114.4658:
        z += 0.003146752 * Q.mass_top20 - 0.3601955
    if Q.mass_displaced5 >= 32.61679:
        z += -0.001172208 * Q.mass_displaced5 + 0.03823364
    if Q.sv_2_sd0_sum < 4.183454:
        z += 0.007926999 * Q.sv_2_sd0_sum - 0.03316224
    if Q.mass_2photon >= 22.18431:
        z += 0.001136302 * Q.mass_2photon - 0.02520808
    if Q.lep_z < 0.3396572 and Q.mass_top30 < 162.7874:
        z += 0.007437144 * (0.3396572 - Q.lep_z) * (162.7874 - Q.mass_top30)
    if Q.lepsj_3_n_d3 < 2.0 and Q.lep_dr < 0.114659:
        z += 0.5033461 * (2.0 - Q.lepsj_3_n_d3) * (0.114659 - Q.lep_dr)
    if Q.lep_z < 0.3396572 and Q.ak02_dr23 < 0.7941848:
        z += -0.9242209 * (0.3396572 - Q.lep_z) * (0.7941848 - Q.ak02_dr23)
    if Q.z_displaced5 > 0.08090366 and Q.sjf_2_2_z_d3 < 0.00953824:
        z += -44.59929 * (Q.z_displaced5 - 0.08090366) * (0.00953824 - Q.sjf_2_2_z_d3)
    if Q.sj3_pair_mass_min > 48.40547 and Q.n_lund_kt_above_5 < 4.0:
        z += 0.005593251 * (Q.sj3_pair_mass_min - 48.40547) * (4.0 - Q.n_lund_kt_above_5)
    if Q.n_s3d_above_3 < 10.0 and Q.lepsj_3_maxsd0 < 586.9572:
        z += -3.777772e-05 * (10.0 - Q.n_s3d_above_3) * (586.9572 - Q.lepsj_3_maxsd0)
    if Q.lepsj_3_n_d3 < 2.0 and Q.lepsj_3_dr > 0.0:
        z += 0.2053646 * (2.0 - Q.lepsj_3_n_d3) * (Q.lepsj_3_dr - 0.0)
    if Q.mres_sd_mass_b0z005 > 86.65765 and Q.ak02_3_z < 0.1807551:
        z += -0.0222225 * (Q.mres_sd_mass_b0z005 - 86.65765) * (0.1807551 - Q.ak02_3_z)
    if Q.lep_ptrel > 27.3236 and Q.ecf_g42 < 8.17233e-05:
        z += 62.03566 * (Q.lep_ptrel - 27.3236) * (8.17233e-05 - Q.ecf_g42)
    if Q.n_s3d_above_3 < 10.0 and Q.M2_b05 > 0.0815014:
        z += 0.2236594 * (10.0 - Q.n_s3d_above_3) * (Q.M2_b05 - 0.0815014)
    if Q.z_displaced5 > 0.08090366 and Q.dc_1_n_lep < 1.0:
        z += 0.7206889 * (Q.z_displaced5 - 0.08090366) * (1.0 - Q.dc_1_n_lep)
    if Q.n_s3d_above_3 < 4.0 and Q.z_photon > 0.05998812:
        z += 0.2143595 * (4.0 - Q.n_s3d_above_3) * (Q.z_photon - 0.05998812)
    if Q.sip_3d_2 < 447.0873 and Q.jd_3d_4 < 6.43167:
        z += -3.074146e-05 * (447.0873 - Q.sip_3d_2) * (6.43167 - Q.jd_3d_4)
    if Q.n_photon < 16.0 and Q.ak02_2_z < 0.4091922:
        z += -0.006823354 * (16.0 - Q.n_photon) * (0.4091922 - Q.ak02_2_z)
    if Q.mres_sd_mass_b0z005 > 86.65765 and Q.D2_b2 < 4.279435:
        z += 0.001187247 * (Q.mres_sd_mass_b0z005 - 86.65765) * (4.279435 - Q.D2_b2)
    if Q.N2 > 0.2875658 and Q.jd_3d_4 < 172.888:
        z += -0.006606939 * (Q.N2 - 0.2875658) * (172.888 - Q.jd_3d_4)
    if Q.mass < 79.47361 and Q.C2_b2 > 0.02858957:
        z += -4.55975e-05 * (79.47361 - Q.mass) * (Q.C2_b2 - 0.02858957)
    if Q.lep_dr > 0.08178299 and Q.ak02_2_z > 0.131395:
        z += 0.9181819 * (Q.lep_dr - 0.08178299) * (Q.ak02_2_z - 0.131395)
    if Q.n_photon < 16.0 and Q.jd_3d_5 < 4.004982:
        z += -0.02165541 * (16.0 - Q.n_photon) * (4.004982 - Q.jd_3d_5)
    if Q.n_photon < 16.0 and Q.lnerel_3 < -2.411946:
        z += 0.0001927575 * (16.0 - Q.n_photon) * (-2.411946 - Q.lnerel_3)
    if Q.n_s3d_above_3 < 10.0 and Q.sjf_3_3_n_d3 > 2.0:
        z += -0.01743393 * (10.0 - Q.n_s3d_above_3) * (Q.sjf_3_3_n_d3 - 2.0)
    if Q.sj3_pair_mass_min > 48.40547 and Q.lne_10 < 2.755351:
        z += -0.01310144 * (Q.sj3_pair_mass_min - 48.40547) * (2.755351 - Q.lne_10)
    if Q.lepsj_3_maxsd0 < 1.529925 and Q.mass_top3 > 16.326:
        z += -0.001022748 * (1.529925 - Q.lepsj_3_maxsd0) * (Q.mass_top3 - 16.326)
    if Q.lep_iso < 0.4381892 and Q.sjq_3_2_k1 < -0.6014774:
        z += -4.145933 * (0.4381892 - Q.lep_iso) * (-0.6014774 - Q.sjq_3_2_k1)
    if Q.N2 > 0.2875658 and Q.sjq_3_2_k1 > 0.6148962:
        z += 21.00222 * (Q.N2 - 0.2875658) * (Q.sjq_3_2_k1 - 0.6148962)
    if Q.lep_z < 0.221436 and Q.lepsj_2_n_d3 > 0.0:
        z += 0.04806206 * (0.221436 - Q.lep_z) * (Q.lepsj_2_n_d3 - 0.0)
    if Q.pz_lnd0 < 0.1122946 and Q.jd_3d_6 > 0.9115584:
        z += 0.000266049 * (0.1122946 - Q.pz_lnd0) * (Q.jd_3d_6 - 0.9115584)
    if Q.mres_sd_mass_b0z005 > 178.8957 and Q.sj3_mass2 > 12.74295:
        z += -0.000351958 * (Q.mres_sd_mass_b0z005 - 178.8957) * (Q.sj3_mass2 - 12.74295)
    if Q.pz_lnd0 < 0.1122946 and Q.sj3_mass2 > 1.824785:
        z += 0.04339959 * (0.1122946 - Q.pz_lnd0) * (Q.sj3_mass2 - 1.824785)
    if Q.z_displaced5 < 0.2801368 and Q.n_lund_kt_above_1 > 2.0:
        z += 0.1028737 * (0.2801368 - Q.z_displaced5) * (Q.n_lund_kt_above_1 - 2.0)
    if Q.lep_iso < 1.362094 and Q.sjq_3_3_k1 > 0.2939497:
        z += -0.4496689 * (1.362094 - Q.lep_iso) * (Q.sjq_3_3_k1 - 0.2939497)
    if Q.pz_lnd0 < 0.1122946 and Q.z_neutral_had > 0.08823774:
        z += -11.82807 * (0.1122946 - Q.pz_lnd0) * (Q.z_neutral_had - 0.08823774)
    if Q.z_displaced5 < 0.2801368 and Q.C3_b2 < 0.006832265:
        z += 4.356994 * (0.2801368 - Q.z_displaced5) * (0.006832265 - Q.C3_b2)
    if Q.lep_dr > 0.08178299 and Q.sjf_4_3_z_d3 > 0.0:
        z += -4.382856 * (Q.lep_dr - 0.08178299) * (Q.sjf_4_3_z_d3 - 0.0)
    if Q.n_s3d_above_3 > 1.0 and Q.tau54 < 0.8410552:
        z += 0.08588193 * (Q.n_s3d_above_3 - 1.0) * (0.8410552 - Q.tau54)
    if Q.lep_dr > 0.08178299 and Q.ismuon_14 > 0.0:
        z += 0.02260746 * (Q.lep_dr - 0.08178299) * (Q.ismuon_14 - 0.0)
    if Q.sjq_3_2_nch > 5.0 and Q.dzerr_9 < 0.1016846:
        z += 0.01390689 * (Q.sjq_3_2_nch - 5.0) * (0.1016846 - Q.dzerr_9)
    if Q.z_displaced5 > 0.08090366 and Q.iselectron_29 > 0.0:
        z += -0.4624423 * (Q.z_displaced5 - 0.08090366) * (Q.iselectron_29 - 0.0)
    if Q.lam2 < 0.02148541 and Q.dc_mass_disp_2nd > 0.0:
        z += 0.4708074 * (0.02148541 - Q.lam2) * (Q.dc_mass_disp_2nd - 0.0)
    return z


def neuron_71(Q):
    z = 3.029709e-06
    return z


def neuron_72(Q):
    z = -1.164122e-06
    return z


def neuron_73(Q):
    z = -1.69964e-05
    return z


def neuron_74(Q):
    z = -4.493399e-06
    return z


def neuron_75(Q):
    z = 1.97384e-06
    return z


def neuron_76(Q):
    z = -5.542788e-06
    return z


def neuron_77(Q):
    z = -3.087488e-06
    return z


def neuron_78(Q):
    z = 0.2540695
    if Q.lep_z < 0.3396572:
        z += 3.517297 * Q.lep_z - 1.194675
    if Q.n_s3d_above_3 < 4.0:
        z += -0.03076894 * Q.n_s3d_above_3 + 0.2769205
    if 4.0 <= Q.n_s3d_above_3 < 6.0:
        z += -0.05237525 * Q.n_s3d_above_3 + 0.3633457
    if 6.0 <= Q.n_s3d_above_3 < 9.0:
        z += -0.01714436 * Q.n_s3d_above_3 + 0.1519604
    if Q.n_s3d_above_3 >= 9.0:
        z += 0.01362458 * Q.n_s3d_above_3 - 0.1249601
    if Q.n_pairs_kt_above_1 < 195.0:
        z += -0.003004127 * Q.n_pairs_kt_above_1 + 0.5858048
    if Q.M3_b2 < 0.0142807:
        z += 19.74643 * Q.M3_b2 - 0.2819928
    if 9.380468 <= Q.mass_displaced5 < 21.12768:
        z += 0.009997396 * Q.mass_displaced5 - 0.09378026
    if Q.mass_displaced5 >= 21.12768:
        z += -0.005784638 * Q.mass_displaced5 + 0.2396575
    if Q.mass_top40 < 70.88236:
        z += -0.009693127 * Q.mass_top40 + 0.6870717
    if Q.mass_top40 >= 173.1022:
        z += -0.0002981879 * Q.mass_top40 + 0.05161698
    if Q.lep_ptrel < 18.7678:
        z += 0.05592889 * Q.lep_ptrel - 0.7656083
    if 18.7678 <= Q.lep_ptrel < 43.20788:
        z += -0.01162247 * Q.lep_ptrel + 0.5021821
    if Q.mass < 95.14961:
        z += 0.009336179 * Q.mass - 0.8883338
    if 114.0172 <= Q.mass < 164.4374:
        z += 0.00650288 * Q.mass - 0.7414405
    if Q.mass >= 164.4374:
        z += 0.006916809 * Q.mass - 0.8095058
    if Q.lepsj_3_n_d3 < 2.0:
        z += -0.03971775 * Q.lepsj_3_n_d3 + 0.0794355
    if Q.n_neutral < 22.0:
        z += -0.01190782 * Q.n_neutral + 0.2619721
    if Q.pair_max_lnm2 >= 7.347625:
        z += -0.3308671 * Q.pair_max_lnm2 + 2.431087
    if Q.lund1_lndelta < -0.5140858:
        z += 0.0001024954 * Q.lund1_lndelta + 5.26914e-05
    if Q.sdb_2_n < 4.0:
        z += -0.03177078 * Q.sdb_2_n + 0.2859371
    if 4.0 <= Q.sdb_2_n < 9.0:
        z += -0.01711355 * Q.sdb_2_n + 0.2273081
    if Q.sdb_2_n >= 9.0:
        z += 0.01465723 * Q.sdb_2_n - 0.05862893
    if Q.z_top15_slots >= 0.8008865:
        z += -1.984411 * Q.z_top15_slots + 1.589288
    if Q.n_sd0_above_5 >= 4.0:
        z += -0.01573835 * Q.n_sd0_above_5 + 0.06295338
    if Q.n_muon >= 1.0:
        z += -0.06101706 * Q.n_muon + 0.06101706
    if Q.pz_lnkt1 < 0.1020077:
        z += -0.2233613 * Q.pz_lnkt1 + 0.02278456
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.006725941 * Q.mres_sd_mass_b0z005 - 0.8212374
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.0129657 * Q.mres_sd_mass_b0z005 + 2.327934
    if Q.mass_top20 >= 130.7093:
        z += 0.01014437 * Q.mass_top20 - 1.325963
    if Q.lund3_lndelta >= -2.817283:
        z += 0.03754997 * Q.lund3_lndelta + 0.1057889
    if Q.sjq_3_3_k1 >= 0.6535817:
        z += 0.04051688 * Q.sjq_3_3_k1 - 0.02648109
    if Q.n_charged_had >= 22.0:
        z += 0.02310308 * Q.n_charged_had - 0.5082677
    if Q.sj4_pair_mass_max < 72.862:
        z += -0.001424937 * Q.sj4_pair_mass_max + 0.02542566
    if 72.862 <= Q.sj4_pair_mass_max < 115.5142:
        z += 0.001838079 * Q.sj4_pair_mass_max - 0.2123242
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.09506071 * Q.lepsj_3_maxsd0 + 0.1259276
    if 2.413264 <= Q.lepsj_3_maxsd0 < 586.9572:
        z += 0.0001770252 * Q.lepsj_3_maxsd0 - 0.1039062
    if Q.n_lepton < 1.0:
        z += 0.6755952 * Q.n_lepton - 0.6286151
    if 1.0 <= Q.n_lepton < 2.0:
        z += -0.04698011 * Q.n_lepton + 0.09396021
    if Q.sjf_4_1_z_d3 < 0.181327:
        z += -0.02416644 * Q.sjf_4_1_z_d3 + 0.004382027
    if Q.z_displaced5 < 0.2801368:
        z += -0.1634659 * Q.z_displaced5 + 0.04579282
    if Q.mass_displaced3 >= 39.09615:
        z += 0.01257829 * Q.mass_displaced3 - 0.4917629
    if Q.n_pairs_kt_above_3 < 99.0:
        z += -0.002878438 * Q.n_pairs_kt_above_3 + 0.2849653
    if Q.mass_2charged < 1.839882:
        z += -0.1257947 * Q.mass_2charged + 0.1197481
    if 1.839882 <= Q.mass_2charged < 10.83159:
        z += 0.01242248 * Q.mass_2charged - 0.1345552
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += 0.1241042 * Q.sjf_2_1_n_d3 - 0.6205209
    if Q.n_sdz_above_2 < 10.0:
        z += -0.00963235 * Q.n_sdz_above_2 + 0.0963235
    if Q.max_abs_d0 < 10.52344:
        z += -0.002433855 * Q.max_abs_d0 + 0.02561252
    if Q.lep_iso < 1.362094:
        z += 0.01089198 * Q.lep_iso - 0.0148359
    if Q.lepsj_2_dr < 0.103005:
        z += 5.175905 * Q.lepsj_2_dr - 0.533144
    if Q.sdb_0_z < 0.0772633:
        z += -1.055882 * Q.sdb_0_z + 0.0815809
    if Q.mass_top15 >= 83.68425:
        z += 0.005680171 * Q.mass_top15 - 0.4753408
    if Q.e4 < 2.097213e-06:
        z += -11592.34 * Q.e4 + 0.0243116
    if Q.sdb_2_z < 0.2040425:
        z += 1.725991 * Q.sdb_2_z - 0.3521756
    if Q.sdb_2_z >= 0.2740506:
        z += -0.2427584 * Q.sdb_2_z + 0.06652809
    if Q.dc_2_jp < 10.98011:
        z += 0.005836065 * Q.dc_2_jp - 0.06408062
    if Q.mass_top10 >= 103.4976:
        z += -0.006673865 * Q.mass_top10 + 0.6907293
    if Q.n_pairs_kt_above_10 < 7.0:
        z += -0.01398657 * Q.n_pairs_kt_above_10 + 0.09790598
    if Q.N2_b05 < 0.5083429:
        z += -3.436703 * Q.N2_b05 + 1.747024
    if Q.pair_mean_lndelta < -2.049856:
        z += 0.008373043 * Q.pair_mean_lndelta + 0.01716353
    if Q.ktd_ln_d34 < -8.400697:
        z += 0.0783582 * Q.ktd_ln_d34 + 0.6582635
    if Q.lepsj_2_maxsd0 < 2.220372:
        z += -0.071096 * Q.lepsj_2_maxsd0 + 0.1578596
    if Q.kt2_1_z < 0.8153708:
        z += 0.268636 * Q.kt2_1_z - 0.2190379
    if Q.sj3_pair_mass_min < 37.19471:
        z += 0.002585236 * Q.sj3_pair_mass_min - 0.0961571
    if Q.z_dr_0_0p05 < 0.007652966:
        z += -5.718852 * Q.z_dr_0_0p05 + 0.04376618
    if Q.tdz_1 < -0.1170754:
        z += -0.01845502 * Q.tdz_1 - 0.002160629
    if Q.mass_neutral < 46.19491:
        z += -0.002173458 * Q.mass_neutral + 0.1004027
    if Q.lep_z < 0.3396572 and Q.pz_lnd2 < 0.1956014:
        z += 1.010086 * (0.3396572 - Q.lep_z) * (0.1956014 - Q.pz_lnd2)
    if Q.lep_z < 0.3396572 and Q.mass_top20 > 86.78877:
        z += 0.01019754 * (0.3396572 - Q.lep_z) * (Q.mass_top20 - 86.78877)
    if Q.mass_displaced5 > 9.380468 and Q.mass_charged < 113.5019:
        z += -0.0001397491 * (Q.mass_displaced5 - 9.380468) * (113.5019 - Q.mass_charged)
    if Q.lep_ptrel < 18.7678 and Q.ktd_ln_d34 < -7.886954:
        z += 0.005578127 * (18.7678 - Q.lep_ptrel) * (-7.886954 - Q.ktd_ln_d34)
    if Q.n_s3d_above_3 > 4.0 and Q.sip_3d_2 < 226.3008:
        z += 0.0004355111 * (Q.n_s3d_above_3 - 4.0) * (226.3008 - Q.sip_3d_2)
    if Q.lep_ptrel < 18.7678 and Q.lepsj_2_maxsd0 < 131.3722:
        z += -3.037244e-05 * (18.7678 - Q.lep_ptrel) * (131.3722 - Q.lepsj_2_maxsd0)
    if Q.n_s3d_above_3 > 4.0 and Q.z_top50_slots > 0.9572293:
        z += 0.5582771 * (Q.n_s3d_above_3 - 4.0) * (Q.z_top50_slots - 0.9572293)
    if Q.lep_ptrel < 18.7678 and Q.nca_sj4_pairmax_over_mass > 0.6162845:
        z += 0.01179667 * (18.7678 - Q.lep_ptrel) * (Q.nca_sj4_pairmax_over_mass - 0.6162845)
    if Q.lep_ptrel < 18.7678 and Q.sj4_dr_min < 0.1631992:
        z += 0.06161661 * (18.7678 - Q.lep_ptrel) * (0.1631992 - Q.sj4_dr_min)
    if Q.lepsj_3_n_d3 < 2.0 and Q.jet_e > 835.8719:
        z += -0.0001170048 * (2.0 - Q.lepsj_3_n_d3) * (Q.jet_e - 835.8719)
    if Q.lep_ptrel < 18.7678 and Q.ak02_2_z > 0.08487383:
        z += -0.01524796 * (18.7678 - Q.lep_ptrel) * (Q.ak02_2_z - 0.08487383)
    if Q.lep_z < 0.3396572 and Q.ecf_g42 < 8.17233e-05:
        z += 10709.38 * (0.3396572 - Q.lep_z) * (8.17233e-05 - Q.ecf_g42)
    if Q.n_s3d_above_3 > 4.0 and Q.tau32 < 0.863865:
        z += -0.07417547 * (Q.n_s3d_above_3 - 4.0) * (0.863865 - Q.tau32)
    if Q.lep_ptrel < 18.7678 and Q.sum_pt_top30 < 749.0582:
        z += 4.343532e-05 * (18.7678 - Q.lep_ptrel) * (749.0582 - Q.sum_pt_top30)
    if Q.mass_displaced5 > 9.380468 and Q.jd_3d_5 < 6.771002:
        z += 0.0003700745 * (Q.mass_displaced5 - 9.380468) * (6.771002 - Q.jd_3d_5)
    if Q.lepsj_3_n_d3 < 2.0 and Q.ak02_3_n_lep > 0.0:
        z += 0.0428992 * (2.0 - Q.lepsj_3_n_d3) * (Q.ak02_3_n_lep - 0.0)
    if Q.lepsj_3_n_d3 < 2.0 and Q.dc_2_n_lep < 1.0:
        z += 0.01510616 * (2.0 - Q.lepsj_3_n_d3) * (1.0 - Q.dc_2_n_lep)
    if Q.z_top15_slots > 0.8008865 and Q.z_displaced5 > 0.2184442:
        z += -4.336836 * (Q.z_top15_slots - 0.8008865) * (Q.z_displaced5 - 0.2184442)
    if Q.lep_z < 0.3396572 and Q.ak02_n > 1.0:
        z += 0.04459031 * (0.3396572 - Q.lep_z) * (Q.ak02_n - 1.0)
    if Q.mass_displaced5 > 9.380468 and Q.sjq_2_prod_k03 < 0.00489684:
        z += 0.00149206 * (Q.mass_displaced5 - 9.380468) * (0.00489684 - Q.sjq_2_prod_k03)
    if Q.lep_ptrel < 18.7678 and Q.mass_2photon < 22.18431:
        z += -0.0005503088 * (18.7678 - Q.lep_ptrel) * (22.18431 - Q.mass_2photon)
    if Q.lep_ptrel < 18.7678 and Q.sjf_4_2_z_d3 < 0.1570831:
        z += 0.05657345 * (18.7678 - Q.lep_ptrel) * (0.1570831 - Q.sjf_4_2_z_d3)
    if Q.n_s3d_above_3 > 4.0 and Q.sjf_4_3_n_d3 < 3.0:
        z += -7.069087e-06 * (Q.n_s3d_above_3 - 4.0) * (3.0 - Q.sjf_4_3_n_d3)
    if Q.lund3_lndelta > -2.817283 and Q.jd_3d_4 < 172.888:
        z += 0.0001491639 * (Q.lund3_lndelta - -2.817283) * (172.888 - Q.jd_3d_4)
    if Q.lepsj_3_n_d3 < 2.0 and Q.lepsj_3_dr < 0.09790963:
        z += 0.3602473 * (2.0 - Q.lepsj_3_n_d3) * (0.09790963 - Q.lepsj_3_dr)
    if Q.lep_ptrel < 18.7678 and Q.ak02_3_z_disp3 > 0.01033072:
        z += -0.002228617 * (18.7678 - Q.lep_ptrel) * (Q.ak02_3_z_disp3 - 0.01033072)
    if Q.mass_displaced3 > 39.09615 and Q.sjq_2_prod_k05 < 0.004703917:
        z += -0.03234884 * (Q.mass_displaced3 - 39.09615) * (0.004703917 - Q.sjq_2_prod_k05)
    if Q.sdb_2_n < 9.0 and Q.sjq_2_prod_k03 > -0.9998116:
        z += -0.04245751 * (9.0 - Q.sdb_2_n) * (Q.sjq_2_prod_k03 - -0.9998116)
    if Q.lep_z < 0.3396572 and Q.lepsj_2_dr < 0.103005:
        z += 27.43965 * (0.3396572 - Q.lep_z) * (0.103005 - Q.lepsj_2_dr)
    if Q.lepsj_2_dr < 0.103005 and Q.jet_charge_k03 < 0.1164034:
        z += 0.7336052 * (0.103005 - Q.lepsj_2_dr) * (0.1164034 - Q.jet_charge_k03)
    if Q.N2_b05 < 0.5083429 and Q.kt2_min12_sd0_1 > 66.72222:
        z += 0.0003532806 * (0.5083429 - Q.N2_b05) * (Q.kt2_min12_sd0_1 - 66.72222)
    if Q.mass_displaced5 > 21.12768 and Q.charge_52 > 0.0:
        z += 0.001180929 * (Q.mass_displaced5 - 21.12768) * (Q.charge_52 - 0.0)
    if Q.sj4_pair_mass_max < 115.5142 and Q.dc_3_n_lep > 0.0:
        z += 0.000792539 * (115.5142 - Q.sj4_pair_mass_max) * (Q.dc_3_n_lep - 0.0)
    if Q.N2_b05 < 0.5083429 and Q.eta_28 < 0.0:
        z += 0.1825685 * (0.5083429 - Q.N2_b05) * (0.0 - Q.eta_28)
    if Q.sj4_pair_mass_max < 115.5142 and Q.n_neutral_had < 9.0:
        z += -0.0004379102 * (115.5142 - Q.sj4_pair_mass_max) * (9.0 - Q.n_neutral_had)
    return z


def neuron_79(Q):
    z = -3.602388e-05
    return z


def neuron_80(Q):
    z = 9.707706e-06
    return z


def neuron_81(Q):
    z = 0.06995548
    if Q.z_displaced3 < 0.03624058:
        z += 3.394601 * Q.z_displaced3 - 0.2178124
    if 0.03624058 <= Q.z_displaced3 < 0.06416437:
        z += 4.535075 * Q.z_displaced3 - 0.2591439
    if Q.z_displaced3 >= 0.06416437:
        z += 1.140474 * Q.z_displaced3 - 0.04133144
    if Q.pz_lnd2 < 0.1339824:
        z += 0.5716713 * Q.pz_lnd2 - 0.07659387
    if Q.n_pairs_kt_above_3 < 28.0:
        z += 0.002502131 * Q.n_pairs_kt_above_3 - 0.07005966
    if Q.mass_top40 >= 115.7429:
        z += 0.001961422 * Q.mass_top40 - 0.2270207
    z += -0.05148886 * Q.dc_ntag
    if 129.5874 <= Q.mass_top50 < 161.1264:
        z += -0.002650139 * Q.mass_top50 + 0.3434247
    if Q.mass_top50 >= 161.1264:
        z += -0.01267969 * Q.mass_top50 + 1.95945
    if Q.ecf_g31 < 0.01162881:
        z += -37.54153 * Q.ecf_g31 + 0.4365633
    if Q.mres_sd_mass_b2z01 < 76.1529:
        z += -0.00257593 * Q.mres_sd_mass_b2z01 - 0.07474526
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 91.15481:
        z += 0.01043257 * Q.mres_sd_mass_b2z01 - 1.06538
    if 91.15481 <= Q.mres_sd_mass_b2z01 < 96.62031:
        z += -0.02658129 * Q.mres_sd_mass_b2z01 + 2.308611
    if 96.62031 <= Q.mres_sd_mass_b2z01 < 115.0388:
        z += -0.01853354 * Q.mres_sd_mass_b2z01 + 1.531035
    if 115.0388 <= Q.mres_sd_mass_b2z01 < 125.2732:
        z += -0.0001451429 * Q.mres_sd_mass_b2z01 - 0.5843452
    if 125.2732 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.01829795 * Q.mres_sd_mass_b2z01 - 2.89477
    if Q.tau1 < 0.06074238:
        z += -7.149179 * Q.tau1 + 0.4342581
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += -0.04876933 * Q.sjf_2_1_n_d3 + 0.2438466
    if 2.0 <= Q.n_s3d_above_3 < 6.0:
        z += 0.006887517 * Q.n_s3d_above_3 - 0.01377503
    if 6.0 <= Q.n_s3d_above_3 < 10.0:
        z += 0.1193264 * Q.n_s3d_above_3 - 0.6884086
    if Q.n_s3d_above_3 >= 10.0:
        z += 0.1482245 * Q.n_s3d_above_3 - 0.977389
    if Q.sip_3d_2 < 226.3008:
        z += -0.0009607739 * Q.sip_3d_2 + 0.2174239
    if Q.sum_zz_dr2 < 0.06117886:
        z += 6.29388 * Q.sum_zz_dr2 - 0.3850524
    if Q.sdb_4_z < 0.04510459:
        z += -1.114038 * Q.sdb_4_z + 0.05024822
    if Q.jd_3d_5 < 4.004982:
        z += -0.1120964 * Q.jd_3d_5 + 0.4489439
    if Q.max_dr < 0.6403502:
        z += -0.02357712 * Q.max_dr + 0.01509761
    if Q.lepsj_3_n_d3 < 2.0:
        z += 0.01549149 * Q.lepsj_3_n_d3 - 0.03098299
    if Q.n_pairs_kt_above_1 < 58.0:
        z += -0.01034899 * Q.n_pairs_kt_above_1 + 0.6002417
    if Q.max_abs_d0 < 10.52344:
        z += -0.02177832 * Q.max_abs_d0 + 0.2291828
    if Q.sv_n >= 1.0:
        z += -0.08946681 * Q.sv_n + 0.08946681
    if Q.lepsj_3_dr < 0.005262883:
        z += 42.1307 * Q.lepsj_3_dr - 0.2217289
    if Q.sdb_2_z < 0.2040425:
        z += -0.1150537 * Q.sdb_2_z + 0.02347584
    z += -0.02409307 * Q.lepsj_2_n_d3
    if Q.sj4_pair_mass_max < 59.5457:
        z += 0.003721869 * Q.sj4_pair_mass_max - 0.2216213
    if Q.tdz_1 < -0.1170754:
        z += -0.1895333 * Q.tdz_1 - 0.02218969
    if Q.sjf_2_2_n_d3 >= 5.0:
        z += -0.02836121 * Q.sjf_2_2_n_d3 + 0.1418061
    if Q.n_s3d_above_10 >= 3.0:
        z += 0.04249055 * Q.n_s3d_above_10 - 0.1274717
    if Q.sdb_0_n < 3.0:
        z += -0.03542357 * Q.sdb_0_n + 0.04599017
    if 3.0 <= Q.sdb_0_n < 5.0:
        z += 0.03014027 * Q.sdb_0_n - 0.1507014
    if Q.mass_displaced3 < 1.777286:
        z += 0.0232276 * Q.mass_displaced3 - 0.04128208
    if Q.sum_z_dr2_top2 < 0.003472016:
        z += -45.99527 * Q.sum_z_dr2_top2 + 0.1596963
    if Q.sjf_4_n2disp >= 1.0:
        z += -0.0995474 * Q.sjf_4_n2disp + 0.0995474
    if Q.n_dr_0p4_up < 15.0:
        z += 0.01858979 * Q.n_dr_0p4_up - 0.2788469
    if Q.tau2 < 0.1076451:
        z += -0.4292558 * Q.tau2 + 0.04620728
    if Q.sjf_2_2_max3d < 2.044945:
        z += 0.2856095 * Q.sjf_2_2_max3d - 0.5840559
    if Q.sjf_2_1_mass_d3 < 12.49716:
        z += -0.002226137 * Q.sjf_2_1_mass_d3 + 0.02782039
    if Q.pair_mean_lndelta >= -1.352792:
        z += 0.6354237 * Q.pair_mean_lndelta + 0.8595958
    if Q.sjf_2_1_maxsd0 < 2.055951:
        z += -0.01755865 * Q.sjf_2_1_maxsd0 + 0.03609972
    if Q.sj2_mass2 < 6.465318:
        z += -0.005271457 * Q.sj2_mass2 + 0.03408164
    if Q.min_pair_mass < 1.514826:
        z += 0.03382356 * Q.min_pair_mass - 0.05123681
    if Q.pz_lnd0 < 0.1325326:
        z += -0.04469423 * Q.pz_lnd0 + 0.005923444
    if Q.lep_iso < 14.38858:
        z += 0.007088373 * Q.lep_iso - 0.1019916
    if Q.sj3_pairmin_over_m >= 0.2477126:
        z += 0.4035627 * Q.sj3_pairmin_over_m - 0.09996757
    if Q.z_displaced3 > 0.03624058 and Q.n_s3d_above_3 < 7.0:
        z += -0.3687303 * (Q.z_displaced3 - 0.03624058) * (7.0 - Q.n_s3d_above_3)
    if Q.z_displaced3 > 0.03624058 and Q.z_charged_had > 0.265564:
        z += -4.981574 * (Q.z_displaced3 - 0.03624058) * (Q.z_charged_had - 0.265564)
    if Q.z_displaced3 > 0.03624058 and Q.jd_sum_abs_sd0_top3 < 1262.672:
        z += 0.0002358766 * (Q.z_displaced3 - 0.03624058) * (1262.672 - Q.jd_sum_abs_sd0_top3)
    if Q.ecf_g31 < 0.01162881 and Q.n_s3d_above_3 > 4.0:
        z += 4.142461 * (0.01162881 - Q.ecf_g31) * (Q.n_s3d_above_3 - 4.0)
    if Q.ecf_g31 < 0.01162881 and Q.ak02_2_n_lep < 1.0:
        z += 23.86165 * (0.01162881 - Q.ecf_g31) * (1.0 - Q.ak02_2_n_lep)
    if Q.z_displaced3 > 0.03624058 and Q.sip_3d_3 < 16.23838:
        z += 0.06262444 * (Q.z_displaced3 - 0.03624058) * (16.23838 - Q.sip_3d_3)
    if Q.ecf_g31 < 0.01162881 and Q.sip_3d_3 < 577.991:
        z += 0.0005260011 * (0.01162881 - Q.ecf_g31) * (577.991 - Q.sip_3d_3)
    if Q.n_s3d_above_3 > 2.0 and Q.dc_n > 2.0:
        z += -0.004661736 * (Q.n_s3d_above_3 - 2.0) * (Q.dc_n - 2.0)
    if Q.pz_lnd2 < 0.1339824 and Q.sjq_2_2_nch < 15.0:
        z += -0.03132835 * (0.1339824 - Q.pz_lnd2) * (15.0 - Q.sjq_2_2_nch)
    if Q.sip_3d_2 < 226.3008 and Q.jd_3d_5 < 6.771002:
        z += -0.0003169807 * (226.3008 - Q.sip_3d_2) * (6.771002 - Q.jd_3d_5)
    if Q.n_s3d_above_3 > 2.0 and Q.n_lund_kt_above_5 < 4.0:
        z += 0.001291551 * (Q.n_s3d_above_3 - 2.0) * (4.0 - Q.n_lund_kt_above_5)
    if Q.sdb_4_z < 0.04510459 and Q.lep_z < 0.004135872:
        z += 1148.229 * (0.04510459 - Q.sdb_4_z) * (0.004135872 - Q.lep_z)
    if Q.sip_3d_2 < 226.3008 and Q.lepsj_2_n_d3 > 2.0:
        z += -0.0002674435 * (226.3008 - Q.sip_3d_2) * (Q.lepsj_2_n_d3 - 2.0)
    if Q.ecf_g31 < 0.01162881 and Q.sjq_2_prod_k05 > -0.5057096:
        z += -18.60193 * (0.01162881 - Q.ecf_g31) * (Q.sjq_2_prod_k05 - -0.5057096)
    if Q.max_abs_d0 < 10.52344 and Q.ak02_min12_n_disp3 < 1.0:
        z += 0.01109146 * (10.52344 - Q.max_abs_d0) * (1.0 - Q.ak02_min12_n_disp3)
    if Q.n_s3d_above_3 > 2.0 and Q.jet_e < 981.6443:
        z += 3.860355e-05 * (Q.n_s3d_above_3 - 2.0) * (981.6443 - Q.jet_e)
    if Q.z_displaced3 > 0.03624058 and Q.n_lund_kt_above_1 > 2.0:
        z += 0.1270685 * (Q.z_displaced3 - 0.03624058) * (Q.n_lund_kt_above_1 - 2.0)
    if Q.max_abs_d0 < 10.52344 and Q.jd_3d_4 < 6.43167:
        z += -0.004472851 * (10.52344 - Q.max_abs_d0) * (6.43167 - Q.jd_3d_4)
    if Q.mres_sd_mass_b2z01 < 158.2019 and Q.mass_2charged > 24.4079:
        z += 7.678824e-05 * (158.2019 - Q.mres_sd_mass_b2z01) * (Q.mass_2charged - 24.4079)
    if Q.mres_sd_mass_b2z01 < 125.2732 and Q.dr_3 > 0.1090736:
        z += 0.002327015 * (125.2732 - Q.mres_sd_mass_b2z01) * (Q.dr_3 - 0.1090736)
    if Q.lepsj_3_n_d3 < 2.0 and Q.ak02_min12_n_disp3 > 1.0:
        z += 0.09131389 * (2.0 - Q.lepsj_3_n_d3) * (Q.ak02_min12_n_disp3 - 1.0)
    if Q.lepsj_3_dr < 0.005262883 and Q.jd_3d_4 < 2.953464:
        z += 10.74889 * (0.005262883 - Q.lepsj_3_dr) * (2.953464 - Q.jd_3d_4)
    if Q.sum_zz_dr2 < 0.06117886 and Q.jd_3d_6 > 16.11752:
        z += -0.002376396 * (0.06117886 - Q.sum_zz_dr2) * (Q.jd_3d_6 - 16.11752)
    if Q.sv_n > 1.0 and Q.tau54 < 0.8953628:
        z += -0.02843111 * (Q.sv_n - 1.0) * (0.8953628 - Q.tau54)
    if Q.sdb_2_z < 0.2040425 and Q.tau43 < 0.6363796:
        z += -2.88578 * (0.2040425 - Q.sdb_2_z) * (0.6363796 - Q.tau43)
    if Q.z_displaced3 > 0.03624058 and Q.ismuon_3 > 0.0:
        z += 0.4893094 * (Q.z_displaced3 - 0.03624058) * (Q.ismuon_3 - 0.0)
    if Q.ecf_g31 < 0.01162881 and Q.sjq_2_sumabs_k1 > 0.1588551:
        z += 29.81575 * (0.01162881 - Q.ecf_g31) * (Q.sjq_2_sumabs_k1 - 0.1588551)
    if Q.n_s3d_above_3 > 2.0 and Q.sj3_pairmax_over_m < 0.8501496:
        z += -0.0238063 * (Q.n_s3d_above_3 - 2.0) * (0.8501496 - Q.sj3_pairmax_over_m)
    if Q.n_s3d_above_3 > 2.0 and Q.z_photon < 0.4982257:
        z += -0.007458881 * (Q.n_s3d_above_3 - 2.0) * (0.4982257 - Q.z_photon)
    if Q.ecf_g31 < 0.01162881 and Q.sjf_4_3_z_d3 > 0.005113028:
        z += 178.9268 * (0.01162881 - Q.ecf_g31) * (Q.sjf_4_3_z_d3 - 0.005113028)
    if Q.pz_lnd2 < 0.1339824 and Q.sjf_4_4_z_d3 > 0.003370318:
        z += 6.202703 * (0.1339824 - Q.pz_lnd2) * (Q.sjf_4_4_z_d3 - 0.003370318)
    if Q.sdb_4_z < 0.04510459 and Q.jet_abs_eta > 0.1941339:
        z += -2.055189 * (0.04510459 - Q.sdb_4_z) * (Q.jet_abs_eta - 0.1941339)
    if Q.z_displaced3 < 0.06416437 and Q.eccentricity > 0.7717404:
        z += -7.763976 * (0.06416437 - Q.z_displaced3) * (Q.eccentricity - 0.7717404)
    if Q.mass_top40 > 115.7429 and Q.tau43 > 0.6363796:
        z += 0.0008140396 * (Q.mass_top40 - 115.7429) * (Q.tau43 - 0.6363796)
    if Q.sjf_2_1_n_d3 > 5.0 and Q.kt2_2_charge > 0.06887309:
        z += 0.03045137 * (Q.sjf_2_1_n_d3 - 5.0) * (Q.kt2_2_charge - 0.06887309)
    if Q.n_dr_0p4_up < 15.0 and Q.nca_sj4_pair2nd_over_mass > 0.4488456:
        z += -0.01832947 * (15.0 - Q.n_dr_0p4_up) * (Q.nca_sj4_pair2nd_over_mass - 0.4488456)
    if Q.mass_displaced3 < 1.777286 and Q.kt2_1_sd0_3 > 1.376447:
        z += 0.001120517 * (1.777286 - Q.mass_displaced3) * (Q.kt2_1_sd0_3 - 1.376447)
    if Q.z_displaced3 > 0.03624058 and Q.nca_kt_above_2 > 4.0:
        z += -0.03985373 * (Q.z_displaced3 - 0.03624058) * (Q.nca_kt_above_2 - 4.0)
    if Q.pz_lnd2 < 0.1339824 and Q.iselectron_3 > 0.0:
        z += 1.7696 * (0.1339824 - Q.pz_lnd2) * (Q.iselectron_3 - 0.0)
    if Q.tdz_1 < -0.1170754 and Q.kt2_1_sd0_3 > 13.98058:
        z += 0.0002233976 * (-0.1170754 - Q.tdz_1) * (Q.kt2_1_sd0_3 - 13.98058)
    if Q.n_dr_0p4_up < 15.0 and Q.n_muon > 1.0:
        z += 0.009339429 * (15.0 - Q.n_dr_0p4_up) * (Q.n_muon - 1.0)
    if Q.z_displaced3 < 0.06416437 and Q.dc_2_n_lep < 1.0:
        z += 2.444336 * (0.06416437 - Q.z_displaced3) * (1.0 - Q.dc_2_n_lep)
    if Q.n_s3d_above_3 > 6.0 and Q.eta_18 > 0.3122559:
        z += -0.2250343 * (Q.n_s3d_above_3 - 6.0) * (Q.eta_18 - 0.3122559)
    if Q.mass_top40 > 115.7429 and Q.isnhad_34 > 0.0:
        z += -1.659693e-05 * (Q.mass_top40 - 115.7429) * (Q.isnhad_34 - 0.0)
    if Q.sj4_pair_mass_max < 59.5457 and Q.ak02_2_sd0_3 > -1.729543:
        z += -4.62987e-05 * (59.5457 - Q.sj4_pair_mass_max) * (Q.ak02_2_sd0_3 - -1.729543)
    if Q.sjf_2_1_n_d3 > 5.0 and Q.ak02_2_sd0_3 < 2.627723:
        z += -9.161983e-06 * (Q.sjf_2_1_n_d3 - 5.0) * (2.627723 - Q.ak02_2_sd0_3)
    if Q.z_displaced3 > 0.03624058 and Q.isphoton_32 > 0.0:
        z += 0.1576212 * (Q.z_displaced3 - 0.03624058) * (Q.isphoton_32 - 0.0)
    if Q.mass_top40 > 115.7429 and Q.tdz_13 < 0.0:
        z += 0.002544275 * (Q.mass_top40 - 115.7429) * (0.0 - Q.tdz_13)
    return z


def neuron_82(Q):
    z = -2.942073e-06
    return z


def neuron_83(Q):
    z = 1.322678
    if 78.33213 <= Q.mass_top40 < 155.8928:
        z += 0.003371534 * Q.mass_top40 - 0.2640994
    if Q.mass_top40 >= 155.8928:
        z += 0.02878717 * Q.mass_top40 - 4.226214
    if Q.n_s3d_above_3 < 3.0:
        z += -0.05215581 * Q.n_s3d_above_3 + 0.1564674
    if Q.n_s3d_above_3 >= 6.0:
        z += 0.02715561 * Q.n_s3d_above_3 - 0.1629337
    z += 0.2384762 * Q.pz_lnd2
    if Q.z_displaced3 < 0.06416437:
        z += -5.436815 * Q.z_displaced3 + 0.4216455
    if 0.06416437 <= Q.z_displaced3 < 0.1681173:
        z += -0.700276 * Q.z_displaced3 + 0.1177285
    if Q.mass_top30 < 162.7874:
        z += 0.0001224053 * Q.mass_top30 - 0.01992604
    if Q.n_s3d_above_10 < 6.0:
        z += 0.06690206 * Q.n_s3d_above_10 - 0.4014124
    if Q.mres_sd_mass_b2z01 < 81.19466:
        z += 0.01217824 * Q.mres_sd_mass_b2z01 - 1.695899
    if 81.19466 <= Q.mres_sd_mass_b2z01 < 130.0968:
        z += -0.004707282 * Q.mres_sd_mass_b2z01 - 0.3248845
    if 130.0968 <= Q.mres_sd_mass_b2z01 < 139.2565:
        z += 0.02277654 * Q.mres_sd_mass_b2z01 - 3.900444
    if 139.2565 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.0105983 * Q.mres_sd_mass_b2z01 - 2.204545
    if Q.mres_sd_mass_b2z01 >= 158.2019:
        z += 0.007375725 * Q.mres_sd_mass_b2z01 - 1.694727
    if Q.N2 < 0.258375:
        z += 1.49823 * Q.N2 - 0.387105
    if Q.M3_b2 < 0.02160244:
        z += 23.30977 * Q.M3_b2 - 0.5035479
    if Q.sdb_2_z < 0.1530389:
        z += -1.52841 * Q.sdb_2_z - 0.1097653
    if 0.1530389 <= Q.sdb_2_z < 0.5109872:
        z += 0.9601148 * Q.sdb_2_z - 0.4906064
    if Q.mass_top15 >= 77.22442:
        z += -0.00749181 * Q.mass_top15 + 0.5785507
    if Q.pair_max_lnm2 >= 7.524613:
        z += 0.2829571 * Q.pair_max_lnm2 - 2.129143
    if Q.sj4_pair_mass_max < 75.78208:
        z += -0.004349663 * Q.sj4_pair_mass_max + 0.8525592
    if 75.78208 <= Q.sj4_pair_mass_max < 115.5142:
        z += -0.01316146 * Q.sj4_pair_mass_max + 1.520335
    if Q.mass_displaced3 < 13.03663:
        z += 0.0004525836 * Q.mass_displaced3 - 0.005900166
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += 0.1355382 * Q.sjf_2_1_n_d3 - 0.6776909
    if Q.mass_displaced5 < 1.262841:
        z += 0.08708765 * Q.mass_displaced5 - 0.1099778
    if Q.max_abs_d0 < 10.52344:
        z += -0.03596643 * Q.max_abs_d0 + 0.3784904
    if Q.jd_sum_abs_sd0_top5 < 185.1889:
        z += 0.000685283 * Q.jd_sum_abs_sd0_top5 - 0.1269068
    if Q.sip_3d_1 < 7.028704:
        z += -0.08614054 * Q.sip_3d_1 + 0.6054564
    if Q.tau1 < 0.06074238:
        z += 10.42981 * Q.tau1 - 0.6335315
    if Q.sjf_4_n2disp >= 2.0:
        z += -0.1224912 * Q.sjf_4_n2disp + 0.2449823
    if 79.47361 <= Q.mass < 95.14961:
        z += -0.02547609 * Q.mass + 2.024677
    if 95.14961 <= Q.mass < 117.4867:
        z += -0.03270649 * Q.mass + 2.712646
    if Q.mass >= 117.4867:
        z += -0.01475324 * Q.mass + 0.6033794
    if Q.n_sd0_above_3 < 9.0:
        z += 0.008440096 * Q.n_sd0_above_3 - 0.07596086
    if Q.tdz_1 < -0.1170754:
        z += -0.1344032 * Q.tdz_1 - 0.01573531
    if Q.n_charged_pt_above_10 < 5.0:
        z += 0.02401963 * Q.n_charged_pt_above_10 - 0.1200981
    if Q.sdb_2_n < 10.0:
        z += -0.01252638 * Q.sdb_2_n + 0.1252638
    if Q.dr_max_012 < 0.02210827:
        z += 10.59114 * Q.dr_max_012 - 0.2341517
    if Q.e4 < 3.53193e-06:
        z += 77431.45 * Q.e4 - 0.2734824
    if Q.n_pt_above_5 < 24.0:
        z += 0.02348981 * Q.n_pt_above_5 - 0.5637555
    if Q.sv_2_z < 0.03767806:
        z += 3.553643 * Q.sv_2_z - 0.1338944
    if Q.tau5 >= 0.01467699:
        z += 12.518 * Q.tau5 - 0.1837265
    if Q.sum_z_dr2_top50 < 0.0925671:
        z += -11.41972 * Q.sum_z_dr2_top50 + 1.05709
    if Q.D3_b05 < 1.139376:
        z += 0.4024377 * Q.D3_b05 - 0.4585279
    if Q.mres_pruned_mass < 86.14266:
        z += 0.0006719475 * Q.mres_pruned_mass - 0.05788335
    if Q.lepsj_3_n_d3 < 2.0:
        z += -0.005519809 * Q.lepsj_3_n_d3 + 0.01103962
    if Q.lep_iso < 6.185635:
        z += 0.01249376 * Q.lep_iso - 0.07728186
    if Q.lepsj_2_dr < 0.06573337:
        z += -2.24333 * Q.lepsj_2_dr + 0.1474617
    if Q.lepsj_3_maxsd0 < 46.74905:
        z += -0.0008090221 * Q.lepsj_3_maxsd0 + 0.03782102
    if Q.mres_sd_prong_mass1 < 27.26979:
        z += -0.003596909 * Q.mres_sd_prong_mass1 + 0.09808695
    if Q.dc_2_jp < 4.498447:
        z += 0.006353429 * Q.dc_2_jp - 0.02858056
    if Q.n_dr_0p4_up < 4.0:
        z += -0.02664592 * Q.n_dr_0p4_up + 0.1065837
    if Q.mass_top10 >= 103.4976:
        z += -0.006523208 * Q.mass_top10 + 0.6751367
    if Q.n_s3d_above_10 < 6.0 and Q.ecf_g42 < 5.969698e-05:
        z += 127.5405 * (6.0 - Q.n_s3d_above_10) * (5.969698e-05 - Q.ecf_g42)
    if Q.n_s3d_above_3 > 6.0 and Q.jd_sum_abs_sd0_top5 < 1292.625:
        z += 7.138862e-05 * (Q.n_s3d_above_3 - 6.0) * (1292.625 - Q.jd_sum_abs_sd0_top5)
    if Q.mass_top40 > 78.33213 and Q.lep_z < 0.1275041:
        z += -0.05872378 * (Q.mass_top40 - 78.33213) * (0.1275041 - Q.lep_z)
    if Q.mass_top30 < 162.7874 and Q.mass_displaced5 > 9.380468:
        z += -0.0002291295 * (162.7874 - Q.mass_top30) * (Q.mass_displaced5 - 9.380468)
    if Q.n_s3d_above_10 < 6.0 and Q.sdb_2_z > 0.2277628:
        z += -0.1509389 * (6.0 - Q.n_s3d_above_10) * (Q.sdb_2_z - 0.2277628)
    if Q.n_s3d_above_3 > 6.0 and Q.sj3_pairmin_over_m < 0.4866692:
        z += 0.3330097 * (Q.n_s3d_above_3 - 6.0) * (0.4866692 - Q.sj3_pairmin_over_m)
    if Q.n_s3d_above_3 > 6.0 and Q.jd_3d_4 < 14.85973:
        z += -0.008206671 * (Q.n_s3d_above_3 - 6.0) * (14.85973 - Q.jd_3d_4)
    if Q.mres_sd_mass_b2z01 < 139.2565 and Q.dc_2_n_lep < 1.0:
        z += 0.00187879 * (139.2565 - Q.mres_sd_mass_b2z01) * (1.0 - Q.dc_2_n_lep)
    if Q.sdb_2_z < 0.1530389 and Q.jd_3d_5 < 4.004982:
        z += -1.783695 * (0.1530389 - Q.sdb_2_z) * (4.004982 - Q.jd_3d_5)
    if Q.n_s3d_above_10 < 6.0 and Q.lne_0 > 4.380463:
        z += 0.02705098 * (6.0 - Q.n_s3d_above_10) * (Q.lne_0 - 4.380463)
    if Q.max_abs_d0 < 10.52344 and Q.jd_3d_5 < 6.771002:
        z += -0.01067345 * (10.52344 - Q.max_abs_d0) * (6.771002 - Q.jd_3d_5)
    if Q.mres_sd_mass_b2z01 < 139.2565 and Q.lund1_lnz > -2.478371:
        z += 0.002918637 * (139.2565 - Q.mres_sd_mass_b2z01) * (Q.lund1_lnz - -2.478371)
    if Q.sj4_pair_mass_max < 115.5142 and Q.ak02_3_n_lep > 0.0:
        z += -0.001027521 * (115.5142 - Q.sj4_pair_mass_max) * (Q.ak02_3_n_lep - 0.0)
    if Q.mass_displaced3 < 13.03663 and Q.jd_3d_5 < 6.771002:
        z += 0.005317228 * (13.03663 - Q.mass_displaced3) * (6.771002 - Q.jd_3d_5)
    if Q.sj4_pair_mass_max < 115.5142 and Q.z_neutral_had < 0.3093201:
        z += -0.006733387 * (115.5142 - Q.sj4_pair_mass_max) * (0.3093201 - Q.z_neutral_had)
    if Q.mres_sd_mass_b2z01 > 130.0968 and Q.jd_3d_5 < 10.7744:
        z += 0.0002719312 * (Q.mres_sd_mass_b2z01 - 130.0968) * (10.7744 - Q.jd_3d_5)
    if Q.mres_sd_mass_b2z01 > 81.19466 and Q.jd_3d_5 < 16.9414:
        z += 4.824184e-05 * (Q.mres_sd_mass_b2z01 - 81.19466) * (16.9414 - Q.jd_3d_5)
    if Q.M3_b2 < 0.02160244 and Q.sjq_3_3_nch < 6.0:
        z += 1.180907 * (0.02160244 - Q.M3_b2) * (6.0 - Q.sjq_3_3_nch)
    if Q.n_s3d_above_3 > 6.0 and Q.jd_3d_4 < 172.888:
        z += 0.0008450091 * (Q.n_s3d_above_3 - 6.0) * (172.888 - Q.jd_3d_4)
    if Q.jd_sum_abs_sd0_top5 < 185.1889 and Q.sjf_4_4_z_d3 < 0.003370318:
        z += -0.1684359 * (185.1889 - Q.jd_sum_abs_sd0_top5) * (0.003370318 - Q.sjf_4_4_z_d3)
    if Q.n_s3d_above_3 > 6.0 and Q.lne_3 < 4.821289:
        z += -0.03408919 * (Q.n_s3d_above_3 - 6.0) * (4.821289 - Q.lne_3)
    if Q.z_displaced3 < 0.1681173 and Q.lepsj_3_mass < 12.91148:
        z += 0.0392055 * (0.1681173 - Q.z_displaced3) * (12.91148 - Q.lepsj_3_mass)
    if Q.n_pt_above_5 < 24.0 and Q.lepsj_2_dr < 0.103005:
        z += -0.04180097 * (24.0 - Q.n_pt_above_5) * (0.103005 - Q.lepsj_2_dr)
    if Q.sj4_pair_mass_max < 75.78208 and Q.sjf_4_3_z_d3 < 0.07303924:
        z += -0.1395671 * (75.78208 - Q.sj4_pair_mass_max) * (0.07303924 - Q.sjf_4_3_z_d3)
    if Q.sjf_2_1_n_d3 > 5.0 and Q.dc_tag_2nd < 0.1300849:
        z += -0.2364621 * (Q.sjf_2_1_n_d3 - 5.0) * (0.1300849 - Q.dc_tag_2nd)
    if Q.sv_2_z < 0.03767806 and Q.max_dr > 0.8033751:
        z += 7.007133 * (0.03767806 - Q.sv_2_z) * (Q.max_dr - 0.8033751)
    if Q.sj4_pair_mass_max < 115.5142 and Q.n_dr_0p2_0p4 > 9.0:
        z += -0.0001670918 * (115.5142 - Q.sj4_pair_mass_max) * (Q.n_dr_0p2_0p4 - 9.0)
    if Q.mass_top30 < 162.7874 and Q.dc_tag_2nd > 1.067968:
        z += 4.904376e-05 * (162.7874 - Q.mass_top30) * (Q.dc_tag_2nd - 1.067968)
    if Q.sj4_pair_mass_max < 115.5142 and Q.tdz_1 > 0.05116378:
        z += 0.006681179 * (115.5142 - Q.sj4_pair_mass_max) * (Q.tdz_1 - 0.05116378)
    if Q.mass > 79.47361 and Q.dr_15 < 0.2535652:
        z += 0.005080576 * (Q.mass - 79.47361) * (0.2535652 - Q.dr_15)
    return z


def neuron_84(Q):
    z = -7.427511e-05
    return z


def neuron_85(Q):
    z = -8.18552e-07
    return z


def neuron_86(Q):
    z = -1.709903e-05
    return z


def neuron_87(Q):
    z = 1.619163e-06
    return z


def neuron_88(Q):
    z = 2.996992e-07
    return z


def neuron_89(Q):
    z = 1.735376e-06
    return z


def neuron_90(Q):
    z = -0.1086133
    if Q.lep_z < 0.004135872:
        z += 8.154526 * Q.lep_z - 0.4691049
    if 0.004135872 <= Q.lep_z < 0.1275041:
        z += 4.320174 * Q.lep_z - 0.4532465
    if 0.1275041 <= Q.lep_z < 0.3396572:
        z += -0.4600136 * Q.lep_z + 0.156247
    if Q.z_displaced3 < 0.3261071:
        z += -0.1077049 * Q.z_displaced3 + 0.03512332
    z += -1.066796 * Q.pz_lnd2
    if Q.sj4_pair_mass_max < 128.0079:
        z += -0.002254752 * Q.sj4_pair_mass_max + 0.2886259
    if Q.tau1 < 0.06074238:
        z += 9.520698 * Q.tau1 - 0.5783098
    if Q.N2_b05 < 0.3667049:
        z += -5.111372 * Q.N2_b05 + 1.874366
    if Q.lep_ptrel < 18.7678:
        z += -0.01420179 * Q.lep_ptrel + 0.3069959
    if 18.7678 <= Q.lep_ptrel < 43.20788:
        z += -0.001655459 * Q.lep_ptrel + 0.07152887
    if Q.lepsj_3_n_d3 < 2.0:
        z += -0.04745936 * Q.lepsj_3_n_d3 + 0.09491872
    if Q.lepsj_2_dr < 0.006470637:
        z += -6.499033 * Q.lepsj_2_dr - 0.1319247
    if 0.006470637 <= Q.lepsj_2_dr < 0.06573337:
        z += 2.935699 * Q.lepsj_2_dr - 0.1929734
    if Q.pz_lnd0 < 0.1122946:
        z += -0.289782 * Q.pz_lnd0 + 0.03254094
    if Q.n_dr_0p4_up < 4.0:
        z += 0.01241451 * Q.n_dr_0p4_up - 0.04965805
    if Q.n_charged_had >= 21.0:
        z += -0.006966469 * Q.n_charged_had + 0.1462959
    if Q.mass_top40 < 115.7429:
        z += -0.0005244291 * Q.mass_top40 + 0.06069895
    if 164.4374 <= Q.mass < 182.8592:
        z += -0.0003386752 * Q.mass + 0.05569088
    if Q.mass >= 182.8592:
        z += -0.00197935 * Q.mass + 0.3557034
    if 76.40585 <= Q.mres_sd_mass_b0z005 < 91.82593:
        z += 0.005057333 * Q.mres_sd_mass_b0z005 - 0.3864098
    if 91.82593 <= Q.mres_sd_mass_b0z005 < 97.29337:
        z += 0.007538354 * Q.mres_sd_mass_b0z005 - 0.6142319
    if 97.29337 <= Q.mres_sd_mass_b0z005 < 122.1:
        z += 0.004202344 * Q.mres_sd_mass_b0z005 - 0.2896603
    if 122.1 <= Q.mres_sd_mass_b0z005 < 131.0776:
        z += -0.007887986 * Q.mres_sd_mass_b0z005 + 1.186569
    if 131.0776 <= Q.mres_sd_mass_b0z005 < 141.5589:
        z += -0.009883173 * Q.mres_sd_mass_b0z005 + 1.448093
    if Q.mres_sd_mass_b0z005 >= 141.5589:
        z += 0.004836425 * Q.mres_sd_mass_b0z005 - 0.6355966
    if Q.lepsj_3_maxsd0 < 1.529925:
        z += -0.09926196 * Q.lepsj_3_maxsd0 + 0.07211005
    if 1.529925 <= Q.lepsj_3_maxsd0 < 11.53372:
        z += 0.007972307 * Q.lepsj_3_maxsd0 - 0.09195033
    z += -0.1321052 * Q.z_displaced5
    if Q.lep_iso < 1.362094:
        z += -0.2767038 * Q.lep_iso + 0.3768967
    if Q.sdb_2_n < 10.0:
        z += 0.009535129 * Q.sdb_2_n - 0.09535129
    if Q.sip_3d_2 < 447.0873:
        z += 6.080105e-07 * Q.sip_3d_2 - 0.0002718338
    if Q.n_s3d_above_3 < 7.0:
        z += -0.03673337 * Q.n_s3d_above_3 + 0.2571336
    z += -0.06405288 * Q.dc_1_n_lep
    if Q.sjf_3_n2disp >= 1.0:
        z += -0.02849668 * Q.sjf_3_n2disp + 0.02849668
    if Q.M2 < 0.0740332:
        z += 9.718879 * Q.M2 - 0.7195197
    if Q.n_pairs_kt_above_1 < 58.0:
        z += -0.002424617 * Q.n_pairs_kt_above_1 + 0.2703918
    if 58.0 <= Q.n_pairs_kt_above_1 < 366.0:
        z += -0.0004213117 * Q.n_pairs_kt_above_1 + 0.1542001
    if Q.e2_b2 < 0.07664127:
        z += -4.629761 * Q.e2_b2 + 0.3548308
    z += 0.03096891 * Q.lepsj_2_n_d3
    if Q.max_abs_d0 < 3.113281:
        z += 0.04147314 * Q.max_abs_d0 - 0.1291176
    if Q.sjq_3_3_k1 < -0.6550393:
        z += -1.170511 * Q.sjq_3_3_k1 - 0.7667307
    if Q.sjq_3_3_k1 >= 0.6535817:
        z += 0.8879922 * Q.sjq_3_3_k1 - 0.5803754
    if Q.psi_0p3 >= 0.9925964:
        z += 10.47891 * Q.psi_0p3 - 10.40133
    if Q.mass_2charged >= 20.76537:
        z += -0.007991565 * Q.mass_2charged + 0.1659478
    if Q.n_dr_0p1_0p2 < 4.0:
        z += 0.01470511 * Q.n_dr_0p1_0p2 - 0.05882046
    if Q.sum_z_dr2_top2 < 0.001440167:
        z += 42.62938 * Q.sum_z_dr2_top2 - 0.06139341
    if Q.n_pt_above_1 < 23.0:
        z += -0.01622398 * Q.n_pt_above_1 + 0.3731516
    if Q.z_charged_had < 0.3603262:
        z += 0.8978208 * Q.z_charged_had - 0.3235084
    if Q.z_charged < 0.3265243:
        z += -1.015981 * Q.z_charged + 0.3317425
    if Q.sjf_4_4_maxsd0 < 0.6359875:
        z += -0.00562002 * Q.sjf_4_4_maxsd0 + 0.003574262
    if Q.tau21 < 0.2077175:
        z += 0.403245 * Q.tau21 - 0.08376106
    if Q.dc_split2_mass >= 32.73553:
        z += -0.002306451 * Q.dc_split2_mass + 0.07550289
    if Q.sj3_pairmin_over_m >= 0.2477126:
        z += -1.434746 * Q.sj3_pairmin_over_m + 0.3554048
    if Q.sum_pt_top10 >= 383.2656:
        z += 1.714107e-05 * Q.sum_pt_top10 - 0.006569583
    if Q.e3 < 0.0003489585:
        z += 657.2769 * Q.e3 - 0.2293624
    if Q.n_pairs_kt_above_3 >= 47.0:
        z += -0.0001541294 * Q.n_pairs_kt_above_3 + 0.007244081
    if Q.lep_ptrel < 18.7678 and Q.nca_sj4_pairmax_over_mass < 0.890007:
        z += -0.02085374 * (18.7678 - Q.lep_ptrel) * (0.890007 - Q.nca_sj4_pairmax_over_mass)
    if Q.lep_ptrel < 18.7678 and Q.lepsj_3_maxsd0 < 11.53372:
        z += -0.0007344803 * (18.7678 - Q.lep_ptrel) * (11.53372 - Q.lepsj_3_maxsd0)
    if Q.z_displaced3 < 0.3261071 and Q.M3_b2 < 0.0309457:
        z += 2.206108 * (0.3261071 - Q.z_displaced3) * (0.0309457 - Q.M3_b2)
    if Q.lep_iso < 1.362094 and Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.08091312 * (1.362094 - Q.lep_iso) * (2.413264 - Q.lepsj_3_maxsd0)
    if Q.z_displaced3 < 0.3261071 and Q.lep_dr < 0.04607888:
        z += -9.306473 * (0.3261071 - Q.z_displaced3) * (0.04607888 - Q.lep_dr)
    if Q.lep_z < 0.3396572 and Q.lepsj_2_dr < 0.06573337:
        z += 7.053065 * (0.3396572 - Q.lep_z) * (0.06573337 - Q.lepsj_2_dr)
    if Q.lep_ptrel < 18.7678 and Q.mres_sd_rg_b1z01 > 0.2472393:
        z += -0.007580779 * (18.7678 - Q.lep_ptrel) * (Q.mres_sd_rg_b1z01 - 0.2472393)
    if Q.sj4_pair_mass_max < 128.0079 and Q.sum_pt_top20 > 407.2186:
        z += 3.909965e-06 * (128.0079 - Q.sj4_pair_mass_max) * (Q.sum_pt_top20 - 407.2186)
    if Q.mres_sd_mass_b0z005 > 97.29337 and Q.tau3 < 0.07659457:
        z += -0.009964738 * (Q.mres_sd_mass_b0z005 - 97.29337) * (0.07659457 - Q.tau3)
    if Q.mres_sd_mass_b0z005 > 97.29337 and Q.sip_3d_3 < 3.157646:
        z += -0.0007110113 * (Q.mres_sd_mass_b0z005 - 97.29337) * (3.157646 - Q.sip_3d_3)
    if Q.lep_ptrel < 18.7678 and Q.sip_3d_3 < 3.157646:
        z += 0.01985214 * (18.7678 - Q.lep_ptrel) * (3.157646 - Q.sip_3d_3)
    if Q.lepsj_2_dr < 0.06573337 and Q.sdb_5_z > 0.0:
        z += 4.385482 * (0.06573337 - Q.lepsj_2_dr) * (Q.sdb_5_z - 0.0)
    if Q.pz_lnd0 < 0.1122946 and Q.z_neutral_had < 0.3788785:
        z += 19.98623 * (0.1122946 - Q.pz_lnd0) * (0.3788785 - Q.z_neutral_had)
    if Q.N2_b05 < 0.3667049 and Q.ak02_2_n_lep < 1.0:
        z += -1.47857 * (0.3667049 - Q.N2_b05) * (1.0 - Q.ak02_2_n_lep)
    if Q.lepsj_2_dr < 0.06573337 and Q.mass_displaced3 > 9.090532:
        z += -0.02419444 * (0.06573337 - Q.lepsj_2_dr) * (Q.mass_displaced3 - 9.090532)
    if Q.lep_z < 0.3396572 and Q.sj3_pairmin_over_m > 0.2278414:
        z += 4.351507 * (0.3396572 - Q.lep_z) * (Q.sj3_pairmin_over_m - 0.2278414)
    if Q.lep_z < 0.004135872 and Q.lep_iso < 0.4381892:
        z += -177.2621 * (0.004135872 - Q.lep_z) * (0.4381892 - Q.lep_iso)
    if Q.lepsj_3_n_d3 < 2.0 and Q.lep_dr < 0.04607888:
        z += 6.381027 * (2.0 - Q.lepsj_3_n_d3) * (0.04607888 - Q.lep_dr)
    if Q.e2_b2 < 0.07664127 and Q.sd_nremoved < 4.0:
        z += 0.1189632 * (0.07664127 - Q.e2_b2) * (4.0 - Q.sd_nremoved)
    if Q.lep_ptrel < 43.20788 and Q.sip_3d_3 < 3.157646:
        z += -0.006883256 * (43.20788 - Q.lep_ptrel) * (3.157646 - Q.sip_3d_3)
    if Q.pz_lnd0 < 0.1122946 and Q.jd_3d_4 < 172.888:
        z += -0.02204726 * (0.1122946 - Q.pz_lnd0) * (172.888 - Q.jd_3d_4)
    if Q.lepsj_2_dr < 0.06573337 and Q.sjq_3_3_k1 > 0.4037488:
        z += -3.933356 * (0.06573337 - Q.lepsj_2_dr) * (Q.sjq_3_3_k1 - 0.4037488)
    if Q.sj4_pair_mass_max < 128.0079 and Q.lne_5 > 3.452465:
        z += -0.0004551776 * (128.0079 - Q.sj4_pair_mass_max) * (Q.lne_5 - 3.452465)
    if Q.N2_b05 < 0.3667049 and Q.sjq_2_prod_k03 > 0.1042503:
        z += -0.903148 * (0.3667049 - Q.N2_b05) * (Q.sjq_2_prod_k03 - 0.1042503)
    if Q.lep_z < 0.3396572 and Q.jet_charge > 0.09659934:
        z += -2.365177 * (0.3396572 - Q.lep_z) * (Q.jet_charge - 0.09659934)
    if Q.sdb_2_n < 10.0 and Q.z_charged > 0.5767344:
        z += 0.05570051 * (10.0 - Q.sdb_2_n) * (Q.z_charged - 0.5767344)
    if Q.sdb_2_n < 10.0 and Q.e4_b2 > 7.0969e-08:
        z += -12.93761 * (10.0 - Q.sdb_2_n) * (Q.e4_b2 - 7.0969e-08)
    if Q.M2 < 0.0740332 and Q.mres_sd_prong_mass2 > 13.86227:
        z += -0.1147949 * (0.0740332 - Q.M2) * (Q.mres_sd_prong_mass2 - 13.86227)
    if Q.mass_top40 < 115.7429 and Q.phi_1 < -0.009407043:
        z += 0.008851627 * (115.7429 - Q.mass_top40) * (-0.009407043 - Q.phi_1)
    if Q.lep_iso < 1.362094 and Q.ecf_g43 < 2.885768e-05:
        z += -9302.013 * (1.362094 - Q.lep_iso) * (2.885768e-05 - Q.ecf_g43)
    if Q.mass_2charged > 20.76537 and Q.sv_2_dr > 0.08408739:
        z += 0.01285399 * (Q.mass_2charged - 20.76537) * (Q.sv_2_dr - 0.08408739)
    if Q.sjq_3_3_k1 < -0.6550393 and Q.dc_2_mass_disp3 < 0.1136997:
        z += -0.7596862 * (-0.6550393 - Q.sjq_3_3_k1) * (0.1136997 - Q.dc_2_mass_disp3)
    if Q.sjf_3_n2disp > 1.0 and Q.nca_sj4_pair2nd_over_mass > 0.3369906:
        z += 0.2949015 * (Q.sjf_3_n2disp - 1.0) * (Q.nca_sj4_pair2nd_over_mass - 0.3369906)
    if Q.sdb_2_n < 10.0 and Q.dc_3_charge > 0.0:
        z += 0.03626588 * (10.0 - Q.sdb_2_n) * (Q.dc_3_charge - 0.0)
    if Q.n_charged_had > 21.0 and Q.dc_2_mass_disp3 > 0.1136997:
        z += 2.410928e-05 * (Q.n_charged_had - 21.0) * (Q.dc_2_mass_disp3 - 0.1136997)
    if Q.lepsj_2_dr < 0.06573337 and Q.phi_3 < -0.03515625:
        z += -0.1019887 * (0.06573337 - Q.lepsj_2_dr) * (-0.03515625 - Q.phi_3)
    if Q.z_charged_had < 0.3603262 and Q.td0_25 < 0.0:
        z += 3.50919 * (0.3603262 - Q.z_charged_had) * (0.0 - Q.td0_25)
    if Q.e2_b2 < 0.07664127 and Q.sjq_3_sumabs_k1 < 0.9926336:
        z += -2.584389 * (0.07664127 - Q.e2_b2) * (0.9926336 - Q.sjq_3_sumabs_k1)
    if Q.e2_b2 < 0.07664127 and Q.kt2_min12_mass_disp3 < 0.2012988:
        z += -4.314234 * (0.07664127 - Q.e2_b2) * (0.2012988 - Q.kt2_min12_mass_disp3)
    if Q.sjf_4_4_maxsd0 < 0.6359875 and Q.eta_60 < 0.0:
        z += 0.4751986 * (0.6359875 - Q.sjf_4_4_maxsd0) * (0.0 - Q.eta_60)
    if Q.mres_sd_mass_b0z005 > 76.40585 and Q.tdz_13 < -0.1329965:
        z += 0.001726955 * (Q.mres_sd_mass_b0z005 - 76.40585) * (-0.1329965 - Q.tdz_13)
    if Q.n_pt_above_1 < 23.0 and Q.dc_2_z < 0.2748617:
        z += -0.1407541 * (23.0 - Q.n_pt_above_1) * (0.2748617 - Q.dc_2_z)
    if Q.lepsj_3_maxsd0 < 11.53372 and Q.dc_2_z < 0.3539053:
        z += 0.0246376 * (11.53372 - Q.lepsj_3_maxsd0) * (0.3539053 - Q.dc_2_z)
    return z


def neuron_91(Q):
    z = 3.992637e-06
    return z


def neuron_92(Q):
    z = -6.724735e-06
    return z


def neuron_93(Q):
    z = 8.464388e-07
    return z


def neuron_94(Q):
    z = -2.322172e-06
    return z


def neuron_95(Q):
    z = 4.369518e-07
    return z


def neuron_96(Q):
    z = 4.278042e-07
    return z


def neuron_97(Q):
    z = -1.911401
    if Q.mass_displaced3 < 18.80005:
        z += -0.0426463 * Q.mass_displaced3 + 1.180694
    if 18.80005 <= Q.mass_displaced3 < 39.09615:
        z += -0.01867066 * Q.mass_displaced3 + 0.7299508
    if Q.mass < 95.14961:
        z += 0.012381 * Q.mass - 3.19174
    if 95.14961 <= Q.mass < 100.4835:
        z += 0.03496147 * Q.mass - 5.340264
    if 100.4835 <= Q.mass < 149.0507:
        z += 0.04405824 * Q.mass - 6.254338
    if 149.0507 <= Q.mass < 164.4374:
        z += 0.005632589 * Q.mass - 0.526969
    if 164.4374 <= Q.mass < 182.8592:
        z += -0.02167214 * Q.mass + 3.962951
    if Q.e3_b2 < 1.365419e-05:
        z += 1936.822 * Q.e3_b2 - 0.197101
    if 1.365419e-05 <= Q.e3_b2 < 9.621843e-05:
        z += 3624.655 * Q.e3_b2 - 0.220147
    if 9.621843e-05 <= Q.e3_b2 < 0.0002536827:
        z += 1156.081 * Q.e3_b2 + 0.01737532
    if 0.0002536827 <= Q.e3_b2 < 0.0007909605:
        z += -578.198 * Q.e3_b2 + 0.4573318
    if Q.pair_mean_lnm2 >= 2.09826:
        z += -0.0185568 * Q.pair_mean_lnm2 + 0.038937
    if Q.tau32 < 0.6513932:
        z += -0.3572732 * Q.tau32 + 0.2327253
    if Q.sj3_pair_mass_max < 63.7508:
        z += -0.03602108 * Q.sj3_pair_mass_max + 3.731724
    if 63.7508 <= Q.sj3_pair_mass_max < 114.7848:
        z += -0.02812537 * Q.sj3_pair_mass_max + 3.228365
    if Q.mass_top50 < 71.80762:
        z += 0.02529331 * Q.mass_top50 - 2.172896
    if 71.80762 <= Q.mass_top50 < 122.0585:
        z += 0.007097253 * Q.mass_top50 - 0.8662804
    if Q.sip_3d_2 < 447.0873:
        z += 0.0007511197 * Q.sip_3d_2 - 0.3358161
    if Q.sj3_pair_mass_min >= 68.11898:
        z += 8.208266e-05 * Q.sj3_pair_mass_min - 0.005591387
    if Q.jd_3d_4 < 172.888:
        z += 0.001321683 * Q.jd_3d_4 - 0.2285031
    if Q.tau21 < 0.135772:
        z += -0.1753295 * Q.tau21 + 0.02380483
    if Q.n_pairs_kt_above_3 < 111.0:
        z += -0.003117824 * Q.n_pairs_kt_above_3 + 0.3460784
    if Q.ak02_min12_jp < 7.746064:
        z += -0.01055255 * Q.ak02_min12_jp + 0.08174075
    z += 2.651572 * Q.sj3_pairmax_over_m
    if Q.mass_top40 < 126.8853:
        z += -0.0173048 * Q.mass_top40 + 2.195724
    if Q.N2_b05 >= 0.4353632:
        z += -1.778289 * Q.N2_b05 + 0.7742018
    if Q.max_abs_d0 < 5.8125:
        z += 0.09957968 * Q.max_abs_d0 - 0.5788069
    if Q.lep_ptrel >= 3.53503:
        z += -0.008736707 * Q.lep_ptrel + 0.03088452
    if Q.sdb_2_z < 0.2740506:
        z += 0.7756418 * Q.sdb_2_z - 0.2125651
    if Q.pair_max_lnm2 < 7.075642:
        z += -0.04859099 * Q.pair_max_lnm2 + 0.3438124
    if Q.n_sd0_above_3 < 6.0:
        z += 0.1346267 * Q.n_sd0_above_3 - 0.8077599
    if Q.sjf_3_1_n_d5 < 5.0:
        z += -0.01874519 * Q.sjf_3_1_n_d5 + 0.09372595
    if Q.sj2_mass2 < 1.851735:
        z += -0.1207459 * Q.sj2_mass2 + 0.2235895
    if Q.mres_pruned_mass < 70.04065:
        z += 0.002458354 * Q.mres_pruned_mass - 1.100274
    if 70.04065 <= Q.mres_pruned_mass < 112.1947:
        z += -0.00816913 * Q.mres_pruned_mass - 0.3559183
    if 112.1947 <= Q.mres_pruned_mass < 124.3145:
        z += -0.02743892 * Q.mres_pruned_mass + 1.80605
    if 124.3145 <= Q.mres_pruned_mass < 171.3819:
        z += 0.01121805 * Q.mres_pruned_mass - 2.999574
    if Q.mres_pruned_mass >= 171.3819:
        z += -0.01062748 * Q.mres_pruned_mass + 0.7443559
    if Q.sd_mass >= 119.4443:
        z += 0.01381701 * Q.sd_mass - 1.650363
    if 53.57509 <= Q.mres_sd_mass_b0z005 < 76.40585:
        z += 0.02009379 * Q.mres_sd_mass_b0z005 - 1.076526
    if 76.40585 <= Q.mres_sd_mass_b0z005 < 91.82593:
        z += -0.06428837 * Q.mres_sd_mass_b0z005 + 5.370765
    if 91.82593 <= Q.mres_sd_mass_b0z005 < 97.29337:
        z += 0.03023615 * Q.mres_sd_mass_b0z005 - 3.309038
    if 97.29337 <= Q.mres_sd_mass_b0z005 < 122.1:
        z += 9.335205e-05 * Q.mres_sd_mass_b0z005 - 0.3763431
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.02288145 * Q.mres_sd_mass_b0z005 - 3.15877
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.006000375 * Q.mres_sd_mass_b0z005 + 1.460134
    if Q.n_pairs_kt_above_1 < 325.0:
        z += -0.001415682 * Q.n_pairs_kt_above_1 + 0.4600968
    if Q.lund3_lndelta >= -1.780944:
        z += 0.09102678 * Q.lund3_lndelta + 0.1621136
    if Q.n_s3d_above_3 < 9.0:
        z += -0.08306389 * Q.n_s3d_above_3 + 0.747575
    if Q.mass_charged >= 62.00562:
        z += 0.001529953 * Q.mass_charged - 0.0948657
    if Q.N2 >= 0.2052214:
        z += 0.2181331 * Q.N2 - 0.04476558
    if Q.z_displaced5 >= 0.1339658:
        z += -0.3551507 * Q.z_displaced5 + 0.04757803
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += -0.05190619 * Q.sjf_2_1_n_d3 + 0.2595309
    if Q.mres_sd_prong_mass1 >= 69.04524:
        z += -0.006229798 * Q.mres_sd_prong_mass1 + 0.4301379
    if Q.lund_max_lnkt < 3.734077:
        z += 0.1095831 * Q.lund_max_lnkt - 0.4091916
    if Q.lep_z < 0.221436:
        z += 0.9357109 * Q.lep_z - 0.2072001
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += 0.005757296 * Q.lepsj_3_maxsd0 - 0.01389388
    if Q.lepsj_2_dr < 0.103005:
        z += 0.2842343 * Q.lepsj_2_dr - 0.02927755
    if Q.n_photon < 8.0:
        z += -0.03257328 * Q.n_photon + 0.2605862
    if Q.lnerel_42 >= -6.337363:
        z += -0.09160776 * Q.lnerel_42 - 0.5805517
    if Q.nca_sj4_pair_mass_2nd < 65.88119:
        z += 0.0003794899 * Q.nca_sj4_pair_mass_2nd - 0.02500124
    if 72.72359 <= Q.nca_sj4_pair_mass_2nd < 83.41384:
        z += -0.02576474 * Q.nca_sj4_pair_mass_2nd + 1.873704
    if Q.nca_sj4_pair_mass_2nd >= 83.41384:
        z += 0.004457522 * Q.nca_sj4_pair_mass_2nd - 0.6472507
    if Q.pz_lnd3 < 0.2113485:
        z += 0.5185692 * Q.pz_lnd3 - 0.1095988
    if Q.dc_split2_dr >= 0.3591078:
        z += 0.5859115 * Q.dc_split2_dr - 0.2104054
    if Q.e3_b2 < 0.0002536827 and Q.sdb_2_z < 0.3709098:
        z += 688.8541 * (0.0002536827 - Q.e3_b2) * (0.3709098 - Q.sdb_2_z)
    if Q.mass_displaced3 < 39.09615 and Q.sum_pt_top10 > 324.6547:
        z += -2.091831e-06 * (39.09615 - Q.mass_displaced3) * (Q.sum_pt_top10 - 324.6547)
    if Q.mass < 100.4835 and Q.lep_ptrel < 1.477152:
        z += 0.006033571 * (100.4835 - Q.mass) * (1.477152 - Q.lep_ptrel)
    if Q.mass_displaced3 < 39.09615 and Q.lep_ptrel < 18.7678:
        z += -0.0004964293 * (39.09615 - Q.mass_displaced3) * (18.7678 - Q.lep_ptrel)
    if Q.mass < 182.8592 and Q.dc_ntag > 0.0:
        z += -0.0009664856 * (182.8592 - Q.mass) * (Q.dc_ntag - 0.0)
    if Q.mass < 182.8592 and Q.n_s3d_above_3 > 1.0:
        z += -0.0005845606 * (182.8592 - Q.mass) * (Q.n_s3d_above_3 - 1.0)
    if Q.e3_b2 < 0.0002536827 and Q.jet_charge_k03 > -0.05990128:
        z += 1311.105 * (0.0002536827 - Q.e3_b2) * (Q.jet_charge_k03 - -0.05990128)
    if Q.mass_top50 < 122.0585 and Q.ak02_2_n_lep < 1.0:
        z += 0.008719278 * (122.0585 - Q.mass_top50) * (1.0 - Q.ak02_2_n_lep)
    if Q.mass_displaced3 < 39.09615 and Q.pz_lnd3 < 0.18807:
        z += 0.08197302 * (39.09615 - Q.mass_displaced3) * (0.18807 - Q.pz_lnd3)
    if Q.mass_top50 < 122.0585 and Q.sum_e > 926.2598:
        z += 7.975567e-06 * (122.0585 - Q.mass_top50) * (Q.sum_e - 926.2598)
    if Q.sip_3d_2 < 447.0873 and Q.dc_1_n_lep < 1.0:
        z += 0.0003387324 * (447.0873 - Q.sip_3d_2) * (1.0 - Q.dc_1_n_lep)
    if Q.mass < 182.8592 and Q.ak02_3_n_lep > 0.0:
        z += 0.0006012018 * (182.8592 - Q.mass) * (Q.ak02_3_n_lep - 0.0)
    if Q.sj3_pair_mass_max < 114.7848 and Q.n_photon > 12.0:
        z += -0.0002904397 * (114.7848 - Q.sj3_pair_mass_max) * (Q.n_photon - 12.0)
    if Q.mass_top50 < 122.0585 and Q.sjq_2_prod_k05 < 0.125305:
        z += -0.01396301 * (122.0585 - Q.mass_top50) * (0.125305 - Q.sjq_2_prod_k05)
    if Q.max_abs_d0 < 5.8125 and Q.jd_3d_5 < 4.004982:
        z += 0.04528026 * (5.8125 - Q.max_abs_d0) * (4.004982 - Q.jd_3d_5)
    if Q.tau32 < 0.6513932 and Q.jd_3d_6 < 3.151933:
        z += -0.2632069 * (0.6513932 - Q.tau32) * (3.151933 - Q.jd_3d_6)
    if Q.mass < 164.4374 and Q.jd_3d_6 > 16.11752:
        z += 7.980356e-06 * (164.4374 - Q.mass) * (Q.jd_3d_6 - 16.11752)
    if Q.pair_mean_lnm2 > 2.09826 and Q.dc_3_z < 0.171574:
        z += -0.07945111 * (Q.pair_mean_lnm2 - 2.09826) * (0.171574 - Q.dc_3_z)
    if Q.sdb_2_z < 0.2740506 and Q.pz_lnkt1 < 0.1348361:
        z += 12.85218 * (0.2740506 - Q.sdb_2_z) * (0.1348361 - Q.pz_lnkt1)
    if Q.mass_top40 < 126.8853 and Q.n_muon > 1.0:
        z += -0.009985418 * (126.8853 - Q.mass_top40) * (Q.n_muon - 1.0)
    if Q.mass_displaced3 < 39.09615 and Q.D2 > 3.038341:
        z += 0.0009205172 * (39.09615 - Q.mass_displaced3) * (Q.D2 - 3.038341)
    if Q.mres_pruned_mass > 70.04065 and Q.lnerel_4 > -3.643775:
        z += -0.001782508 * (Q.mres_pruned_mass - 70.04065) * (Q.lnerel_4 - -3.643775)
    if Q.sj2_mass2 < 1.851735 and Q.mass_2photon > 1.504865:
        z += 0.01069197 * (1.851735 - Q.sj2_mass2) * (Q.mass_2photon - 1.504865)
    if Q.lund3_lndelta > -1.780944 and Q.z_photon > 0.2047275:
        z += 0.5824811 * (Q.lund3_lndelta - -1.780944) * (Q.z_photon - 0.2047275)
    if Q.sj3_pair_mass_min > 68.11898 and Q.sv_2_dr < 0.08408739:
        z += -0.04801475 * (Q.sj3_pair_mass_min - 68.11898) * (0.08408739 - Q.sv_2_dr)
    if Q.e3_b2 < 0.0007909605 and Q.sjq_3_sumabs_k1 > 0.4372817:
        z += 112.7767 * (0.0007909605 - Q.e3_b2) * (Q.sjq_3_sumabs_k1 - 0.4372817)
    if Q.sj3_pair_mass_min > 68.11898 and Q.e4_b2 > 7.0969e-08:
        z += 62.50032 * (Q.sj3_pair_mass_min - 68.11898) * (Q.e4_b2 - 7.0969e-08)
    if Q.mres_pruned_mass < 124.3145 and Q.ak02_2_charge < 0.09642216:
        z += -0.00106939 * (124.3145 - Q.mres_pruned_mass) * (0.09642216 - Q.ak02_2_charge)
    if Q.mass < 100.4835 and Q.lep_iso < 6.185635:
        z += -0.0008597799 * (100.4835 - Q.mass) * (6.185635 - Q.lep_iso)
    if Q.lep_z < 0.221436 and Q.lepsj_2_n_d3 < 4.0:
        z += 0.08217055 * (0.221436 - Q.lep_z) * (4.0 - Q.lepsj_2_n_d3)
    if Q.sjf_3_1_n_d5 < 5.0 and Q.dzerr_37 > 0.06842041:
        z += 0.0210572 * (5.0 - Q.sjf_3_1_n_d5) * (Q.dzerr_37 - 0.06842041)
    if Q.lep_z < 0.221436 and Q.kt2_min12_mass_disp3 > 0.0:
        z += -0.01869388 * (0.221436 - Q.lep_z) * (Q.kt2_min12_mass_disp3 - 0.0)
    if Q.nca_sj4_pair_mass_2nd < 65.88119 and Q.nca_kt_above_10 > 1.0:
        z += -0.002600764 * (65.88119 - Q.nca_sj4_pair_mass_2nd) * (Q.nca_kt_above_10 - 1.0)
    if Q.e3_b2 < 0.0007909605 and Q.dr_28 > 0.3368505:
        z += 45.28196 * (0.0007909605 - Q.e3_b2) * (Q.dr_28 - 0.3368505)
    if Q.lep_ptrel > 3.53503 and Q.mass_2photon > 0.4121793:
        z += 3.981572e-05 * (Q.lep_ptrel - 3.53503) * (Q.mass_2photon - 0.4121793)
    if Q.ak02_min12_jp < 7.746064 and Q.mass_2photon > 0.2753928:
        z += 0.0006303957 * (7.746064 - Q.ak02_min12_jp) * (Q.mass_2photon - 0.2753928)
    return z


def neuron_98(Q):
    z = 4.990171e-07
    return z


def neuron_99(Q):
    z = -0.001934241
    return z


def neuron_100(Q):
    z = -6.264077e-06
    return z


def neuron_101(Q):
    z = 8.878262e-06
    return z


def neuron_102(Q):
    z = 5.490515e-06
    return z


def neuron_103(Q):
    z = -3.414379e-06
    return z


def neuron_104(Q):
    z = -1.464256
    if Q.mass < 117.4867:
        z += -0.02667685 * Q.mass + 3.307561
    if 117.4867 <= Q.mass < 149.0507:
        z += -0.005493183 * Q.mass + 0.8187626
    if Q.pair_mean_lndelta < -1.352792:
        z += -0.4902653 * Q.pair_mean_lndelta - 0.6632267
    if Q.sjq_2_1_nch < 18.0:
        z += -0.02182755 * Q.sjq_2_1_nch + 0.3928959
    if Q.lep_ptrel >= 6.983043:
        z += 0.0119492 * Q.lep_ptrel - 0.08344176
    if 3.208089 <= Q.mass_displaced3 < 6.341631:
        z += 0.05430427 * Q.mass_displaced3 - 0.1742129
    if Q.mass_displaced3 >= 6.341631:
        z += 0.01326444 * Q.mass_displaced3 + 0.08604656
    if Q.n_pt_above_1 < 58.0:
        z += -0.0227243 * Q.n_pt_above_1 + 1.318009
    if Q.sj4_pair_mass_max >= 128.0079:
        z += 0.007537529 * Q.sj4_pair_mass_max - 0.9648629
    if Q.tau32_b2 < 0.4680886:
        z += 0.6329615 * Q.tau32_b2 - 0.296282
    if Q.D2 < 1.226724:
        z += -0.06857728 * Q.D2 + 0.08412541
    if Q.D2 >= 3.597891:
        z += 0.1141837 * Q.D2 - 0.4108207
    if Q.n_pairs_kt_above_3 < 34.0:
        z += 0.01333704 * Q.n_pairs_kt_above_3 - 0.4534592
    if Q.N2 < 0.312717:
        z += -0.6065542 * Q.N2 + 0.1896798
    if Q.nca_sj4_pair_mass_2nd >= 77.3635:
        z += 0.006385053 * Q.nca_sj4_pair_mass_2nd - 0.49397
    if Q.pz_lnd0 < 0.2827395:
        z += -1.853359 * Q.pz_lnd0 + 0.5240178
    if Q.n_s3d_above_10 < 6.0:
        z += -0.05395669 * Q.n_s3d_above_10 + 0.3237402
    if Q.sjf_2_1_max3d < 3.32421:
        z += 0.08965483 * Q.sjf_2_1_max3d - 0.2980315
    if Q.sjf_2_1_z_d3 >= 0.1149585:
        z += -0.1361711 * Q.sjf_2_1_z_d3 + 0.01565403
    if Q.sjf_3_2_max3d < 5.245219:
        z += 0.04545977 * Q.sjf_3_2_max3d - 0.2384465
    if Q.sjq_2_sumabs_k03 >= 0.6667228:
        z += -0.08203193 * Q.sjq_2_sumabs_k03 + 0.05469256
    if Q.sjf_3_1_z_d3 < 0.02312549:
        z += -0.2285452 * Q.sjf_3_1_z_d3 + 0.00528522
    if Q.z_neutral_had < 0.1010123:
        z += 0.09276432 * Q.z_neutral_had - 0.009370335
    if Q.tau3 < 0.05398263:
        z += 8.648906 * Q.tau3 - 0.4668906
    if Q.n_pairs_kt_above_1 < 146.0:
        z += 0.001302988 * Q.n_pairs_kt_above_1 - 0.1902363
    if Q.n_lund_kt_above_5 >= 1.0:
        z += -0.01256092 * Q.n_lund_kt_above_5 + 0.01256092
    if Q.lund_max_lndelta >= -0.529318:
        z += 0.5261104 * Q.lund_max_lndelta + 0.2784797
    if Q.sip_3d_2 < 226.3008:
        z += 0.0002439645 * Q.sip_3d_2 - 0.05520936
    if Q.sjf_3_2_maxsd0 < 61.73838:
        z += -0.001854428 * Q.sjf_3_2_maxsd0 + 0.1144894
    if Q.jd_3d_5 < 1.789401:
        z += -0.06288438 * Q.jd_3d_5 + 0.1125254
    if Q.sdb_2_z < 0.3988697:
        z += -0.1216298 * Q.sdb_2_z + 0.04851443
    if Q.ak02_min12_jp < 14.23948:
        z += 0.01215871 * Q.ak02_min12_jp - 0.1731337
    if Q.tau21_b2 >= 0.3898586:
        z += 0.8539143 * Q.tau21_b2 - 0.3329059
    if Q.sjf_4_4_maxsd0 < 0.8080863:
        z += -0.05584894 * Q.sjf_4_4_maxsd0 + 0.04513077
    if Q.sjq_3_2_nch < 1.0:
        z += -0.2338719 * Q.sjq_3_2_nch + 0.2338719
    if Q.pair_max_lnm2 >= 5.908788:
        z += 0.09788785 * Q.pair_max_lnm2 - 0.5783985
    if Q.z_charged_had < 0.4501484:
        z += 0.6545494 * Q.z_charged_had - 0.2946443
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 86.09683:
        z += 0.01921992 * Q.mres_sd_mass_b2z01 - 1.463653
    if Q.mres_sd_mass_b2z01 >= 86.09683:
        z += 0.001499971 * Q.mres_sd_mass_b2z01 + 0.06197861
    if 91.82593 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += -0.004932371 * Q.mres_sd_mass_b0z005 + 0.4529195
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.0008261367 * Q.mres_sd_mass_b0z005 - 0.2037668
    if Q.sj2_mass1 >= 53.51926:
        z += -0.00198104 * Q.sj2_mass1 + 0.1060238
    if Q.mass_top5 >= 17.63354:
        z += -0.00259094 * Q.mass_top5 + 0.04568745
    if Q.tau5 < 0.0230226:
        z += -33.50153 * Q.tau5 + 0.7712924
    if Q.mass_top50 < 135.5368:
        z += -0.002502135 * Q.mass_top50 + 0.3391313
    if Q.sjf_3_1_n_d3 < 4.0:
        z += 0.02195712 * Q.sjf_3_1_n_d3 - 0.08782846
    if Q.D3 >= 1.263537:
        z += -0.03280271 * Q.D3 + 0.04144743
    if Q.sjf_2_2_max3d < 4.516968:
        z += -0.004815272 * Q.sjf_2_2_max3d + 0.1104296
    if 4.516968 <= Q.sjf_2_2_max3d < 16.96512:
        z += -0.007123884 * Q.sjf_2_2_max3d + 0.1208575
    if Q.sjq_2_2_nch < 2.0:
        z += -0.2107552 * Q.sjq_2_2_nch + 0.6036406
    if 2.0 <= Q.sjq_2_2_nch < 15.0:
        z += -0.01401002 * Q.sjq_2_2_nch + 0.2101503
    if Q.mass_charged < 99.20396:
        z += -0.003188871 * Q.mass_charged + 0.3163486
    if Q.lepsj_2_maxsd0 < 1.546763:
        z += -0.05659009 * Q.lepsj_2_maxsd0 + 0.08753146
    if Q.sv_1_dr < 0.1040701:
        z += 0.60777 * Q.sv_1_dr - 0.06325067
    if Q.n_sdz_above_5 < 3.0:
        z += -0.007870104 * Q.n_sdz_above_5 + 0.02361031
    if Q.pz_lnd2 < 0.01637527:
        z += -7.400147 * Q.pz_lnd2 + 0.1211794
    if Q.pz_lnd2 >= 0.1811493:
        z += 0.7597597 * Q.pz_lnd2 - 0.1376299
    if Q.dc_tag_2nd < 1.914834:
        z += 0.08513284 * Q.dc_tag_2nd - 0.1630152
    if Q.mass_top30 >= 126.5007:
        z += 0.003992935 * Q.mass_top30 - 0.5051091
    if Q.z_top3_slots >= 0.5395924:
        z += 1.94705 * Q.z_top3_slots - 1.050613
    if Q.ecf_g41 < 6.216954e-05:
        z += 3680.444 * Q.ecf_g41 - 0.2288115
    if Q.sdb_2_n < 8.0:
        z += -0.01552237 * Q.sdb_2_n + 0.1241789
    if Q.C2 < 0.1798521:
        z += -1.118436 * Q.C2 + 0.201153
    if Q.C2_b05 < 0.2234678:
        z += -1.666654 * Q.C2_b05 + 0.3724436
    if Q.n_pt_above_1 < 58.0 and Q.sjf_2_1_n_d5 > 1.0:
        z += -0.001530863 * (58.0 - Q.n_pt_above_1) * (Q.sjf_2_1_n_d5 - 1.0)
    if Q.mass_displaced3 > 3.208089 and Q.ak02_3_z < 0.1014193:
        z += 0.06394455 * (Q.mass_displaced3 - 3.208089) * (0.1014193 - Q.ak02_3_z)
    if Q.n_pt_above_1 < 58.0 and Q.z_neutral > 0.1384639:
        z += -0.01137605 * (58.0 - Q.n_pt_above_1) * (Q.z_neutral - 0.1384639)
    if Q.mass < 149.0507 and Q.sum_e < 1401.904:
        z += -7.735322e-06 * (149.0507 - Q.mass) * (1401.904 - Q.sum_e)
    if Q.sjq_2_1_nch < 18.0 and Q.sjf_2_2_n_d5 > 1.0:
        z += -0.001445659 * (18.0 - Q.sjq_2_1_nch) * (Q.sjf_2_2_n_d5 - 1.0)
    if Q.n_pt_above_1 < 58.0 and Q.pair_max_lnkt > 1.555555:
        z += -0.003151364 * (58.0 - Q.n_pt_above_1) * (Q.pair_max_lnkt - 1.555555)
    if Q.lep_ptrel > 6.983043 and Q.ak02_dr23 < 0.2385164:
        z += -0.08035247 * (Q.lep_ptrel - 6.983043) * (0.2385164 - Q.ak02_dr23)
    if Q.n_pt_above_1 < 58.0 and Q.ak02_dr13 < 0.4490565:
        z += 0.02029677 * (58.0 - Q.n_pt_above_1) * (0.4490565 - Q.ak02_dr13)
    if Q.mass < 149.0507 and Q.tau21_b2 < 0.3265797:
        z += -0.003819062 * (149.0507 - Q.mass) * (0.3265797 - Q.tau21_b2)
    if Q.pz_lnd0 < 0.2827395 and Q.n_pairs_kt_above_10 > 1.0:
        z += -0.03643864 * (0.2827395 - Q.pz_lnd0) * (Q.n_pairs_kt_above_10 - 1.0)
    if Q.mass < 149.0507 and Q.lnerel_65 > -18.42068:
        z += -0.0003997556 * (149.0507 - Q.mass) * (Q.lnerel_65 - -18.42068)
    if Q.sjf_2_1_z_d3 > 0.1149585 and Q.sjq_2_sumabs_k1 < 0.9542229:
        z += -0.1779926 * (Q.sjf_2_1_z_d3 - 0.1149585) * (0.9542229 - Q.sjq_2_sumabs_k1)
    if Q.lep_ptrel > 6.983043 and Q.mass_2charged < 28.89659:
        z += 0.0004518176 * (Q.lep_ptrel - 6.983043) * (28.89659 - Q.mass_2charged)
    if Q.sjf_2_1_z_d3 > 0.1149585 and Q.sjq_2_prod_k1 < 0.09802954:
        z += -1.216439 * (Q.sjf_2_1_z_d3 - 0.1149585) * (0.09802954 - Q.sjq_2_prod_k1)
    if Q.sjf_3_2_max3d < 5.245219 and Q.mres_sd_zg_b1z01 < 0.325219:
        z += -0.01365685 * (5.245219 - Q.sjf_3_2_max3d) * (0.325219 - Q.mres_sd_zg_b1z01)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.iselectron_1 > 0.0:
        z += -0.005654559 * (34.0 - Q.n_pairs_kt_above_3) * (Q.iselectron_1 - 0.0)
    if Q.sjf_3_2_max3d < 5.245219 and Q.jet_charge > 0.09659934:
        z += 0.06966573 * (5.245219 - Q.sjf_3_2_max3d) * (Q.jet_charge - 0.09659934)
    if Q.sjq_2_1_nch < 18.0 and Q.dr02 > 0.04115773:
        z += -0.01872322 * (18.0 - Q.sjq_2_1_nch) * (Q.dr02 - 0.04115773)
    if Q.sjf_2_1_z_d3 > 0.1149585 and Q.ischhad_0 > 0.0:
        z += 0.4340281 * (Q.sjf_2_1_z_d3 - 0.1149585) * (Q.ischhad_0 - 0.0)
    if Q.tau21_b2 > 0.3898586 and Q.dr_50 < 0.09854228:
        z += -0.1168815 * (Q.tau21_b2 - 0.3898586) * (0.09854228 - Q.dr_50)
    if Q.N2 < 0.312717 and Q.e3_b2 < 4.662928e-05:
        z += -36711.51 * (0.312717 - Q.N2) * (4.662928e-05 - Q.e3_b2)
    if Q.sjq_2_1_nch < 18.0 and Q.dr01 < 0.3258728:
        z += 0.01817329 * (18.0 - Q.sjq_2_1_nch) * (0.3258728 - Q.dr01)
    if Q.sjf_3_1_z_d3 < 0.02312549 and Q.dr_19 < 0.08897484:
        z += 23.01978 * (0.02312549 - Q.sjf_3_1_z_d3) * (0.08897484 - Q.dr_19)
    if Q.mass_top5 > 17.63354 and Q.lnpt_78 > -18.42068:
        z += -0.0003815974 * (Q.mass_top5 - 17.63354) * (Q.lnpt_78 - -18.42068)
    if Q.sjf_2_2_max3d < 4.516968 and Q.e3_b2 < 0.0007909605:
        z += -86.06573 * (4.516968 - Q.sjf_2_2_max3d) * (0.0007909605 - Q.e3_b2)
    if Q.mres_sd_mass_b0z005 > 91.82593 and Q.sv_1_n < 3.0:
        z += 0.0007959729 * (Q.mres_sd_mass_b0z005 - 91.82593) * (3.0 - Q.sv_1_n)
    if Q.sdb_2_z < 0.3988697 and Q.e3 > 0.001381331:
        z += -14.89805 * (0.3988697 - Q.sdb_2_z) * (Q.e3 - 0.001381331)
    if Q.sj4_pair_mass_max > 128.0079 and Q.sv_2_sd0_sum > 1490.247:
        z += -2.002298e-06 * (Q.sj4_pair_mass_max - 128.0079) * (Q.sv_2_sd0_sum - 1490.247)
    if Q.mass_top5 > 17.63354 and Q.eta_1 > -0.0289917:
        z += 0.00124357 * (Q.mass_top5 - 17.63354) * (Q.eta_1 - -0.0289917)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.kt2_2_n_lep < 1.0:
        z += 0.003635581 * (34.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.kt2_2_n_lep)
    if Q.n_pt_above_1 < 58.0 and Q.lepsj_2_n_d3 < 1.0:
        z += 0.004777845 * (58.0 - Q.n_pt_above_1) * (1.0 - Q.lepsj_2_n_d3)
    if Q.lepsj_2_maxsd0 < 1.546763 and Q.lepsj_2_dr < 0.0008210142:
        z += -50.95709 * (1.546763 - Q.lepsj_2_maxsd0) * (0.0008210142 - Q.lepsj_2_dr)
    if Q.mass_charged < 99.20396 and Q.ak02_2_n_lep < 1.0:
        z += -0.002403371 * (99.20396 - Q.mass_charged) * (1.0 - Q.ak02_2_n_lep)
    if Q.lepsj_2_maxsd0 < 1.546763 and Q.eta_4 > 0.104187:
        z += 0.03565809 * (1.546763 - Q.lepsj_2_maxsd0) * (Q.eta_4 - 0.104187)
    if Q.sjq_2_1_nch < 18.0 and Q.dzerr_60 > 0.0:
        z += 0.03882919 * (18.0 - Q.sjq_2_1_nch) * (Q.dzerr_60 - 0.0)
    if Q.ak02_min12_jp < 14.23948 and Q.pz_lnkt1 > 0.06778673:
        z += 0.08128466 * (14.23948 - Q.ak02_min12_jp) * (Q.pz_lnkt1 - 0.06778673)
    return z


def neuron_105(Q):
    z = 2.732383e-05
    return z


def neuron_106(Q):
    z = 2.160935e-05
    return z


def neuron_107(Q):
    z = 1.15856e-05
    return z


def neuron_108(Q):
    z = -4.470702e-06
    return z


def neuron_109(Q):
    z = -5.200158e-07
    return z


def neuron_110(Q):
    z = 3.579609e-06
    return z


def neuron_111(Q):
    z = 3.669617e-06
    return z


def neuron_112(Q):
    z = 5.237823e-06
    return z


def neuron_113(Q):
    z = -2.665928e-05
    return z


def neuron_114(Q):
    z = 4.677418e-06
    return z


def neuron_115(Q):
    z = 1.452494
    if Q.pz_lnd0 < 0.06871203:
        z += -1.466563 * Q.pz_lnd0 - 0.003623972
    if 0.06871203 <= Q.pz_lnd0 < 0.1713451:
        z += 1.017162 * Q.pz_lnd0 - 0.1742857
    if Q.N2_b2 < 0.2243273:
        z += -2.718871 * Q.N2_b2 + 0.6099171
    if Q.mass_displaced3 < 39.09615:
        z += 0.02870415 * Q.mass_displaced3 - 1.122222
    if Q.tau2 < 0.09733903:
        z += -8.051571 * Q.tau2 + 0.7837321
    if Q.tau43 < 0.8939856:
        z += 1.418232 * Q.tau43 - 1.267879
    if Q.z_neutral_had < 0.3788785:
        z += 2.935941 * Q.z_neutral_had
    if Q.z_neutral_had >= 0.3788785:
        z += -0.03567004 * Q.z_neutral_had + 1.12588
    if Q.pair_mean_lnkt < 0.9367772:
        z += -0.8186479 * Q.pair_mean_lnkt + 0.7668907
    if -9.531553 <= Q.ktd_ln_d34 < -7.075141:
        z += -0.1056726 * Q.ktd_ln_d34 - 1.007224
    if Q.ktd_ln_d34 >= -7.075141:
        z += -0.009159483 * Q.ktd_ln_d34 - 0.3243802
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.01541684 * Q.n_pairs_kt_above_1 - 1.119759
    if 80.0 <= Q.n_pairs_kt_above_1 < 411.0:
        z += -0.0003431677 * Q.n_pairs_kt_above_1 + 0.1410419
    if Q.lepsj_3_n_d3 < 3.0:
        z += 0.05769576 * Q.lepsj_3_n_d3 - 0.1730873
    if Q.e3_b2 < 0.0004126585:
        z += 748.6688 * Q.e3_b2 - 0.3775536
    if 0.0004126585 <= Q.e3_b2 < 0.0007909605:
        z += 181.3605 * Q.e3_b2 - 0.143449
    if Q.n_s3d_above_3 >= 4.0:
        z += 0.09193777 * Q.n_s3d_above_3 - 0.3677511
    if Q.mass_top50 < 94.51361:
        z += 0.007718564 * Q.mass_top50 - 1.288043
    if 94.51361 <= Q.mass_top50 < 118.8408:
        z += 0.009123262 * Q.mass_top50 - 1.420806
    if 118.8408 <= Q.mass_top50 < 161.1264:
        z += 0.007959918 * Q.mass_top50 - 1.282553
    if Q.sj4_pair_mass_max < 75.78208:
        z += -0.009776616 * Q.sj4_pair_mass_max + 0.6316673
    if 75.78208 <= Q.sj4_pair_mass_max < 101.849:
        z += 0.004190176 * Q.sj4_pair_mass_max - 0.4267652
    if Q.sip_3d_2 < 447.0873:
        z += -0.0003480033 * Q.sip_3d_2 + 0.1555879
    if Q.lam1 >= 0.04304553:
        z += -0.5572004 * Q.lam1 + 0.02398498
    if Q.mass_neutral < 40.24169:
        z += 0.001981764 * Q.mass_neutral - 0.07974954
    if Q.M2 < 0.120439:
        z += -9.843364 * Q.M2 + 1.185525
    if Q.lep_z < 0.5187302:
        z += -1.420272 * Q.lep_z + 0.736738
    if Q.LHA < 0.5626523:
        z += 3.359777 * Q.LHA - 1.890387
    if Q.e3 < 0.00228569:
        z += -220.1938 * Q.e3 + 0.5032948
    if Q.sj3_pair_mass_min >= 59.49644:
        z += -0.001052354 * Q.sj3_pair_mass_min + 0.06261134
    if Q.n_sd0_above_3 >= 2.0:
        z += -0.07721989 * Q.n_sd0_above_3 + 0.1544398
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.02377404 * Q.lepsj_3_maxsd0 + 0.05737305
    if Q.sj4_pair_mass_min < 25.64898:
        z += -0.006612878 * Q.sj4_pair_mass_min + 0.1696136
    if Q.pt_entropy >= 3.22434:
        z += 0.627687 * Q.pt_entropy - 2.023877
    if Q.jet_abs_eta >= 1.465946:
        z += 0.4036967 * Q.jet_abs_eta - 0.5917976
    if Q.max_abs_dz < 6.402344:
        z += -0.01910819 * Q.max_abs_dz + 0.1223372
    if Q.mass_2charged < 1.398635:
        z += 0.1955805 * Q.mass_2charged - 0.1434314
    if 1.398635 <= Q.mass_2charged < 6.767937:
        z += -0.05228339 * Q.mass_2charged + 0.2032396
    if 6.767937 <= Q.mass_2charged < 20.76537:
        z += 0.01075991 * Q.mass_2charged - 0.2234334
    if Q.sum_pt_top10 < 428.9828:
        z += -0.002996226 * Q.sum_pt_top10 + 1.28533
    if Q.tau4 >= 0.04996:
        z += -10.07365 * Q.tau4 + 0.5032797
    if Q.psi_0p3 >= 0.9756505:
        z += 8.464824 * Q.psi_0p3 - 8.258709
    if Q.sj3_pair_mass_max < 114.7848:
        z += 0.01432947 * Q.sj3_pair_mass_max - 1.644805
    if Q.lep_iso < 0.4381892:
        z += 0.3829215 * Q.lep_iso - 0.1677921
    if Q.n_lepton < 1.0:
        z += -0.4899351 * Q.n_lepton + 0.4899351
    if Q.lep_ptrel < 18.7678:
        z += 0.03442947 * Q.lep_ptrel - 0.6461654
    if Q.mres_sd_prong_mass2 < 7.065114:
        z += 0.03038272 * Q.mres_sd_prong_mass2 - 0.01310934
    if 7.065114 <= Q.mres_sd_prong_mass2 < 21.04127:
        z += -0.01442085 * Q.mres_sd_prong_mass2 + 0.303433
    if Q.mass < 131.3917:
        z += -0.02318758 * Q.mass + 3.046656
    z += -1.297612 * Q.sj3_pairmax_over_m
    if Q.mass_top20 < 93.50967:
        z += -0.00559295 * Q.mass_top20 + 0.5229949
    if Q.psi_0p1 < 0.006220408:
        z += 11.62687 * Q.psi_0p1 - 0.07232387
    if Q.mass_displaced5 >= 21.12768:
        z += -0.002754177 * Q.mass_displaced5 + 0.05818938
    if Q.jd_3d_4 < 172.888:
        z += 0.0005391872 * Q.jd_3d_4 - 0.093219
    if Q.max_abs_d0 < 5.8125:
        z += 0.01504441 * Q.max_abs_d0 - 0.08744562
    if Q.sjf_3_3_maxsd0 < 0.3479096:
        z += -1.14796 * Q.sjf_3_3_maxsd0 + 0.3993861
    if Q.pair_max_lnkt >= 1.555555:
        z += -0.2138308 * Q.pair_max_lnkt + 0.3326255
    if Q.sum_z_dr2_top3 < 0.02745856:
        z += 6.61097 * Q.sum_z_dr2_top3 - 0.1815277
    if Q.sj3_mass3 < 5.460258:
        z += 0.04282489 * Q.sj3_mass3 - 0.233835
    if Q.dc_1_n_lep < 1.0:
        z += -0.07445876 * Q.dc_1_n_lep + 0.07445876
    if Q.lund_max_lnkt >= 3.388322:
        z += 0.11326 * Q.lund_max_lnkt - 0.3837613
    if Q.dc_1_z < 0.932165:
        z += 0.506543 * Q.dc_1_z - 0.4721817
    if Q.mres_sd_mass_b1z01 >= 88.79082:
        z += -0.001225856 * Q.mres_sd_mass_b1z01 + 0.1088448
    if Q.n_neutral_had >= 5.0:
        z += -0.006837596 * Q.n_neutral_had + 0.03418798
    if Q.mass_top15 < 133.1648:
        z += 0.0001615246 * Q.mass_top15 - 0.02150939
    if Q.M3_b2 < 0.01013989:
        z += -15.26651 * Q.M3_b2 + 0.1548007
    if Q.mass_displaced3 < 39.09615 and Q.lep_ptrel < 18.7678:
        z += 0.0009990931 * (39.09615 - Q.mass_displaced3) * (18.7678 - Q.lep_ptrel)
    if Q.mass_displaced3 < 39.09615 and Q.lam1_plus_lam2 > 0.04303099:
        z += -0.04805176 * (39.09615 - Q.mass_displaced3) * (Q.lam1_plus_lam2 - 0.04303099)
    if Q.pair_mean_lnkt < 0.9367772 and Q.sum_pt_top10 < 428.9828:
        z += -0.002005394 * (0.9367772 - Q.pair_mean_lnkt) * (428.9828 - Q.sum_pt_top10)
    if Q.lep_z < 0.5187302 and Q.tau54 < 0.8786609:
        z += -2.647988 * (0.5187302 - Q.lep_z) * (0.8786609 - Q.tau54)
    if Q.lep_z < 0.5187302 and Q.sj4_dr_min < 0.1845735:
        z += 2.129842 * (0.5187302 - Q.lep_z) * (0.1845735 - Q.sj4_dr_min)
    if Q.sip_3d_2 < 447.0873 and Q.jd_3d_4 < 9.982976:
        z += -5.793158e-05 * (447.0873 - Q.sip_3d_2) * (9.982976 - Q.jd_3d_4)
    if Q.lep_z < 0.5187302 and Q.lne_1 > 4.635336:
        z += 0.2433493 * (0.5187302 - Q.lep_z) * (Q.lne_1 - 4.635336)
    if Q.tau2 < 0.09733903 and Q.kt2_1_n_disp3 > 0.0:
        z += -1.428037 * (0.09733903 - Q.tau2) * (Q.kt2_1_n_disp3 - 0.0)
    if Q.lep_z < 0.5187302 and Q.ak02_n > 2.0:
        z += -0.1477022 * (0.5187302 - Q.lep_z) * (Q.ak02_n - 2.0)
    if Q.pz_lnd0 < 0.1713451 and Q.z_displaced5 > 0.1046203:
        z += 6.538107 * (0.1713451 - Q.pz_lnd0) * (Q.z_displaced5 - 0.1046203)
    if Q.lep_z < 0.5187302 and Q.dc_n > 1.0:
        z += -0.07911336 * (0.5187302 - Q.lep_z) * (Q.dc_n - 1.0)
    if Q.sj3_pair_mass_min > 59.49644 and Q.n_lund_kt_above_5 < 4.0:
        z += 0.01009763 * (Q.sj3_pair_mass_min - 59.49644) * (4.0 - Q.n_lund_kt_above_5)
    if Q.sj4_pair_mass_max < 101.849 and Q.jd_3d_6 < 9.356714:
        z += -0.001122528 * (101.849 - Q.sj4_pair_mass_max) * (9.356714 - Q.jd_3d_6)
    if Q.mass_top50 < 118.8408 and Q.jd_3d_6 < 9.356714:
        z += 0.00134606 * (118.8408 - Q.mass_top50) * (9.356714 - Q.jd_3d_6)
    if Q.pz_lnd0 < 0.1713451 and Q.jd_3d_5 > 1.108533:
        z += -0.0009650328 * (0.1713451 - Q.pz_lnd0) * (Q.jd_3d_5 - 1.108533)
    if Q.mass_top50 < 118.8408 and Q.lepsj_2_dr < 0.103005:
        z += 0.05103489 * (118.8408 - Q.mass_top50) * (0.103005 - Q.lepsj_2_dr)
    if Q.lep_z < 0.5187302 and Q.lep_iso < 14.38858:
        z += -0.01361361 * (0.5187302 - Q.lep_z) * (14.38858 - Q.lep_iso)
    if Q.pz_lnd0 < 0.1713451 and Q.kt2_2_n_lep < 1.0:
        z += -0.01437535 * (0.1713451 - Q.pz_lnd0) * (1.0 - Q.kt2_2_n_lep)
    if Q.n_s3d_above_3 > 4.0 and Q.pz_lnkt1 < 0.1525194:
        z += -0.1106373 * (Q.n_s3d_above_3 - 4.0) * (0.1525194 - Q.pz_lnkt1)
    if Q.lep_z < 0.5187302 and Q.sdb_2_z < 0.2509165:
        z += 0.9141135 * (0.5187302 - Q.lep_z) * (0.2509165 - Q.sdb_2_z)
    if Q.mass_top50 < 118.8408 and Q.n_dr_0_0p05 > 2.0:
        z += 3.455399e-05 * (118.8408 - Q.mass_top50) * (Q.n_dr_0_0p05 - 2.0)
    if Q.mass_displaced3 < 39.09615 and Q.dc_tag_2nd > 0.0:
        z += 1.231106e-05 * (39.09615 - Q.mass_displaced3) * (Q.dc_tag_2nd - 0.0)
    if Q.lep_z < 0.5187302 and Q.sdb_3_n > 2.0:
        z += 0.03034 * (0.5187302 - Q.lep_z) * (Q.sdb_3_n - 2.0)
    if Q.max_abs_dz < 6.402344 and Q.lepsj_3_dr < 0.01012269:
        z += -1.729345 * (6.402344 - Q.max_abs_dz) * (0.01012269 - Q.lepsj_3_dr)
    if Q.mass_2charged < 1.398635 and Q.pz_lnd1 > 0.0906501:
        z += 1.226532 * (1.398635 - Q.mass_2charged) * (Q.pz_lnd1 - 0.0906501)
    if Q.sj3_pair_mass_max < 114.7848 and Q.sjf_3_3_n_d3 > 3.0:
        z += 0.002139202 * (114.7848 - Q.sj3_pair_mass_max) * (Q.sjf_3_3_n_d3 - 3.0)
    if Q.n_sd0_above_3 > 2.0 and Q.tau43_b2 < 0.9433644:
        z += 0.03690839 * (Q.n_sd0_above_3 - 2.0) * (0.9433644 - Q.tau43_b2)
    if Q.N2_b2 < 0.2243273 and Q.d0err_52 > 0.0:
        z += -10.02907 * (0.2243273 - Q.N2_b2) * (Q.d0err_52 - 0.0)
    if Q.ktd_ln_d34 > -9.531553 and Q.pz_lnkt3 < 0.102661:
        z += 0.8395 * (Q.ktd_ln_d34 - -9.531553) * (0.102661 - Q.pz_lnkt3)
    if Q.e3_b2 < 0.0007909605 and Q.isphoton_53 < 1.0:
        z += -67.1958 * (0.0007909605 - Q.e3_b2) * (1.0 - Q.isphoton_53)
    if Q.tau43 < 0.8939856 and Q.dc_mass_disp_2nd > 0.0:
        z += -0.08516217 * (0.8939856 - Q.tau43) * (Q.dc_mass_disp_2nd - 0.0)
    if Q.e3_b2 < 0.0004126585 and Q.sdb_5_z > 0.04013001:
        z += 2920.693 * (0.0004126585 - Q.e3_b2) * (Q.sdb_5_z - 0.04013001)
    if Q.lepsj_3_maxsd0 < 2.413264 and Q.sv_1_sd0_sum < 509.6197:
        z += -0.0001495878 * (2.413264 - Q.lepsj_3_maxsd0) * (509.6197 - Q.sv_1_sd0_sum)
    if Q.ktd_ln_d34 > -9.531553 and Q.sjq_2_sumabs_k03 > 1.261566:
        z += 0.06479187 * (Q.ktd_ln_d34 - -9.531553) * (Q.sjq_2_sumabs_k03 - 1.261566)
    return z


def neuron_116(Q):
    z = -3.399319e-07
    return z


def neuron_117(Q):
    z = 2.862136e-06
    return z


def neuron_118(Q):
    z = -8.233224e-06
    return z


def neuron_119(Q):
    z = -6.336861e-07
    return z


def neuron_120(Q):
    z = 1.389019
    if Q.lep_ptrel < 3.53503:
        z += 0.1628006 * Q.lep_ptrel - 1.037026
    if 3.53503 <= Q.lep_ptrel < 6.983043:
        z += 0.05355775 * Q.lep_ptrel - 0.6508487
    if 6.983043 <= Q.lep_ptrel < 12.15228:
        z += 0.05286073 * Q.lep_ptrel - 0.6459815
    if 12.15228 <= Q.lep_ptrel < 27.3236:
        z += -0.000697012 * Q.lep_ptrel + 0.004867265
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.01085337 * Q.lep_ptrel - 0.3107308
    if Q.lep_ptrel >= 43.20788:
        z += 0.001310714 * Q.lep_ptrel + 0.1015871
    if Q.n_pairs_kt_above_3 < 62.0:
        z += 0.008937187 * Q.n_pairs_kt_above_3 - 0.5541056
    if Q.lep_z < 0.004135872:
        z += 120.3802 * Q.lep_z - 0.7315241
    if 0.004135872 <= Q.lep_z < 0.5187302:
        z += 0.4540413 * Q.lep_z - 0.2355249
    if Q.tau32 < 0.6800935:
        z += -0.7220845 * Q.tau32 + 0.491085
    if Q.n_sd0_above_5 < 3.0:
        z += 0.00525985 * Q.n_sd0_above_5 - 0.01577955
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.007862264 * Q.n_pairs_kt_above_1 - 0.6289811
    if Q.pair_mean_lnm2 >= 4.787589:
        z += -0.3436566 * Q.pair_mean_lnm2 + 1.645287
    if Q.mass_top10 >= 103.4976:
        z += -0.009372106 * Q.mass_top10 + 0.9699909
    if Q.pz_lnkt0 < 0.1589878:
        z += -4.511063 * Q.pz_lnkt0 + 0.7172037
    if Q.n_pairs_kt_above_10 < 11.0:
        z += 0.02541451 * Q.n_pairs_kt_above_10 - 0.2795596
    if Q.lep_iso < 0.4381892:
        z += 0.7346552 * Q.lep_iso - 0.1773199
    if 0.4381892 <= Q.lep_iso < 2.907433:
        z += -0.05855967 * Q.lep_iso + 0.1702583
    if Q.lepsj_2_dr < 0.0008210142:
        z += -353.5942 * Q.lepsj_2_dr - 0.02249815
    if 0.0008210142 <= Q.lepsj_2_dr < 0.06573337:
        z += 4.818867 * Q.lepsj_2_dr - 0.3167604
    if Q.lep_dr < 0.08178299:
        z += -4.259113 * Q.lep_dr + 0.2139331
    if 0.08178299 <= Q.lep_dr < 0.3014662:
        z += 0.6117441 * Q.lep_dr - 0.1844202
    if Q.sjq_3_3_k1 >= 0.4037488:
        z += 0.4616323 * Q.sjq_3_3_k1 - 0.1863835
    if Q.mass_charged < 68.20711:
        z += 0.01855148 * Q.mass_charged - 1.265343
    if Q.lam1 >= 0.04304553:
        z += -5.086169 * Q.lam1 + 0.2189368
    if Q.M2 < 0.05805236:
        z += -16.76507 * Q.M2 + 0.9732519
    if Q.mass_neutral < 63.87145:
        z += -0.003761738 * Q.mass_neutral + 0.2402677
    if Q.mass < 71.96396:
        z += 0.01647275 * Q.mass - 2.912446
    if 71.96396 <= Q.mass < 90.08945:
        z += 0.04674288 * Q.mass - 5.090805
    if 90.08945 <= Q.mass < 105.7234:
        z += 0.04354352 * Q.mass - 4.802575
    if 105.7234 <= Q.mass < 114.0172:
        z += 0.0239946 * Q.mass - 2.735798
    if Q.lund3_lndelta >= -2.036501:
        z += -0.1098854 * Q.lund3_lndelta - 0.2237818
    if Q.pz_lnd2 < 0.03277088:
        z += 5.550283 * Q.pz_lnd2 - 0.1818877
    if Q.e3 >= 0.00228569:
        z += 39.73351 * Q.e3 - 0.09081848
    if Q.dr_min_012 < 0.01394245:
        z += 3.848662 * Q.dr_min_012 - 0.05365978
    if Q.sjf_2_2_max3d < 3.029824:
        z += 0.04553355 * Q.sjf_2_2_max3d - 0.1379586
    if Q.jd_sum_abs_sd0_top5 < 124.9603:
        z += 0.0001292257 * Q.jd_sum_abs_sd0_top5 - 0.01614808
    if Q.n_s3d_above_10 < 3.0:
        z += -0.02739097 * Q.n_s3d_above_10 + 0.0821729
    if Q.sj3_dr13 >= 0.7462286:
        z += -0.3729901 * Q.sj3_dr13 + 0.2783359
    if Q.sj2_mass1 < 91.2852:
        z += -0.000695013 * Q.sj2_mass1 + 0.0634444
    if Q.kt2_min12_jp < 4.912483:
        z += 0.01836734 * Q.kt2_min12_jp - 0.09022926
    if Q.ak02_1_n_lep < 1.0:
        z += 0.1274664 * Q.ak02_1_n_lep - 0.1274664
    if Q.ak02_1_n_lep >= 1.0:
        z += -0.1883091 * Q.ak02_1_n_lep + 0.1883091
    if Q.max_dr < 0.2977501:
        z += -0.3855853 * Q.max_dr + 0.1148081
    if Q.pair_mean_lndelta >= -1.790445:
        z += -0.3398064 * Q.pair_mean_lndelta - 0.6084046
    if Q.pz_lnd0 < 0.05066241:
        z += -2.897576 * Q.pz_lnd0 + 0.1467982
    if Q.ak02_1_z < 0.7954618:
        z += -0.002979477 * Q.ak02_1_z + 0.00237006
    if Q.N2_b05 < 0.3667049:
        z += 4.417161 * Q.N2_b05 - 1.619795
    if Q.mass_top20 >= 114.4658:
        z += -0.007909825 * Q.mass_top20 + 0.9054045
    if Q.mass_top50 < 135.5368:
        z += -0.004935037 * Q.mass_top50 + 0.8820143
    if 135.5368 <= Q.mass_top50 < 161.1264:
        z += -0.00786453 * Q.mass_top50 + 1.279068
    if 161.1264 <= Q.mass_top50 < 178.725:
        z += -0.003611394 * Q.mass_top50 + 0.5937759
    if Q.mass_top50 >= 178.725:
        z += 0.001323643 * Q.mass_top50 - 0.2882384
    z += 0.02079301 * Q.sjf_3_n2disp
    if Q.mres_sd_prong_mass2 < 4.6892:
        z += 0.03259255 * Q.mres_sd_prong_mass2 - 0.152833
    if Q.sd_mass < 78.4753:
        z += -0.002610737 * Q.sd_mass - 0.1307971
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += -0.01236476 * Q.sd_mass + 0.634653
    if 88.81751 <= Q.sd_mass < 111.701:
        z += 0.02025719 * Q.sd_mass - 2.262748
    if Q.lep_ptrel < 3.53503 and Q.n_s3d_above_3 < 10.0:
        z += -0.04386944 * (3.53503 - Q.lep_ptrel) * (10.0 - Q.n_s3d_above_3)
    if Q.lep_z < 0.5187302 and Q.mass_neutral < 69.1565:
        z += -0.007352616 * (0.5187302 - Q.lep_z) * (69.1565 - Q.mass_neutral)
    if Q.lep_z < 0.5187302 and Q.mass_displaced3 > 0.0:
        z += 0.02281875 * (0.5187302 - Q.lep_z) * (Q.mass_displaced3 - 0.0)
    if Q.lep_ptrel < 3.53503 and Q.jd_3d_4 < 172.888:
        z += 0.0008132312 * (3.53503 - Q.lep_ptrel) * (172.888 - Q.jd_3d_4)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.sjf_2_1_z_d3 < 0.2710171:
        z += 0.006061228 * (62.0 - Q.n_pairs_kt_above_3) * (0.2710171 - Q.sjf_2_1_z_d3)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.lepsj_2_dr < 0.1822601:
        z += 0.01421455 * (62.0 - Q.n_pairs_kt_above_3) * (0.1822601 - Q.lepsj_2_dr)
    if Q.n_sd0_above_5 < 3.0 and Q.ak02_3_n_lep > 0.0:
        z += 0.05510064 * (3.0 - Q.n_sd0_above_5) * (Q.ak02_3_n_lep - 0.0)
    if Q.lep_ptrel < 3.53503 and Q.n_photon > 5.0:
        z += 0.006202294 * (3.53503 - Q.lep_ptrel) * (Q.n_photon - 5.0)
    if Q.lep_ptrel < 12.15228 and Q.sdb_2_n < 18.0:
        z += 0.002531786 * (12.15228 - Q.lep_ptrel) * (18.0 - Q.sdb_2_n)
    if Q.lep_z < 0.5187302 and Q.planar_flow < 0.7095008:
        z += 0.3890258 * (0.5187302 - Q.lep_z) * (0.7095008 - Q.planar_flow)
    if Q.lep_ptrel < 12.15228 and Q.jd_3d_6 < 3.151933:
        z += -0.0240799 * (12.15228 - Q.lep_ptrel) * (3.151933 - Q.jd_3d_6)
    if Q.lep_iso < 2.907433 and Q.jd_3d_6 < 5.367501:
        z += 0.03420012 * (2.907433 - Q.lep_iso) * (5.367501 - Q.jd_3d_6)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.dc_2_n_lep < 1.0:
        z += 0.001992998 * (62.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.dc_2_n_lep)
    if Q.pz_lnkt0 < 0.1589878 and Q.sj3_dr13 < 0.7462286:
        z += -1.661807 * (0.1589878 - Q.pz_lnkt0) * (0.7462286 - Q.sj3_dr13)
    if Q.lep_ptrel < 3.53503 and Q.e4_b05 > 4.363663e-05:
        z += 471.8028 * (3.53503 - Q.lep_ptrel) * (Q.e4_b05 - 4.363663e-05)
    if Q.mass_top10 > 103.4976 and Q.sum_pt_top40 > 502.3395:
        z += 2.447898e-05 * (Q.mass_top10 - 103.4976) * (Q.sum_pt_top40 - 502.3395)
    if Q.lep_ptrel < 3.53503 and Q.max_abs_d0 < 5.8125:
        z += 0.01307435 * (3.53503 - Q.lep_ptrel) * (5.8125 - Q.max_abs_d0)
    if Q.lepsj_2_dr < 0.06573337 and Q.sjq_3_3_k1 > 0.4037488:
        z += -4.576238 * (0.06573337 - Q.lepsj_2_dr) * (Q.sjq_3_3_k1 - 0.4037488)
    if Q.tau32 < 0.6800935 and Q.sjq_3_3_k1 < -0.4042365:
        z += 0.2192971 * (0.6800935 - Q.tau32) * (-0.4042365 - Q.sjq_3_3_k1)
    if Q.n_pairs_kt_above_10 < 11.0 and Q.jd_3d_6 < 3.151933:
        z += 0.009813236 * (11.0 - Q.n_pairs_kt_above_10) * (3.151933 - Q.jd_3d_6)
    if Q.lep_ptrel < 3.53503 and Q.sv_2_z > 0.0:
        z += -1.006485 * (3.53503 - Q.lep_ptrel) * (Q.sv_2_z - 0.0)
    if Q.lep_ptrel < 12.15228 and Q.n_muon > 0.0:
        z += 0.01931658 * (12.15228 - Q.lep_ptrel) * (Q.n_muon - 0.0)
    if Q.lep_iso < 0.4381892 and Q.dc_1_n_lep < 1.0:
        z += 0.8933378 * (0.4381892 - Q.lep_iso) * (1.0 - Q.dc_1_n_lep)
    if Q.mass_top10 > 103.4976 and Q.ak02_dr23 < 0.3728877:
        z += -0.00359593 * (Q.mass_top10 - 103.4976) * (0.3728877 - Q.ak02_dr23)
    if Q.mass_charged < 68.20711 and Q.jd_3d_6 < 5.367501:
        z += 0.003931446 * (68.20711 - Q.mass_charged) * (5.367501 - Q.jd_3d_6)
    if Q.lepsj_2_dr < 0.06573337 and Q.jd_3d_6 < 5.367501:
        z += -1.457737 * (0.06573337 - Q.lepsj_2_dr) * (5.367501 - Q.jd_3d_6)
    if Q.mass_neutral < 63.87145 and Q.jd_3d_5 < 4.004982:
        z += 0.0004247084 * (63.87145 - Q.mass_neutral) * (4.004982 - Q.jd_3d_5)
    if Q.M2 < 0.05805236 and Q.jd_3d_5 < 6.771002:
        z += -4.023013 * (0.05805236 - Q.M2) * (6.771002 - Q.jd_3d_5)
    if Q.mass < 114.0172 and Q.jd_3d_6 < 5.367501:
        z += 0.002777437 * (114.0172 - Q.mass) * (5.367501 - Q.jd_3d_6)
    if Q.pz_lnkt0 < 0.1589878 and Q.sjq_2_prod_k05 > -0.3383985:
        z += -2.759887 * (0.1589878 - Q.pz_lnkt0) * (Q.sjq_2_prod_k05 - -0.3383985)
    if Q.lep_ptrel < 12.15228 and Q.pz_lnd2 > 0.008992646:
        z += -0.01775883 * (12.15228 - Q.lep_ptrel) * (Q.pz_lnd2 - 0.008992646)
    if Q.mass < 114.0172 and Q.z_neutral_had < 0.1271955:
        z += 0.07816876 * (114.0172 - Q.mass) * (0.1271955 - Q.z_neutral_had)
    if Q.dr_min_012 < 0.01394245 and Q.iselectron_1 > 0.0:
        z += -11.23299 * (0.01394245 - Q.dr_min_012) * (Q.iselectron_1 - 0.0)
    if Q.tau32 < 0.6800935 and Q.sj2_mass2 < 3.928781:
        z += 0.01097552 * (0.6800935 - Q.tau32) * (3.928781 - Q.sj2_mass2)
    if Q.lep_z < 0.5187302 and Q.jd_3d_6 > 30.57088:
        z += -0.0003048209 * (0.5187302 - Q.lep_z) * (Q.jd_3d_6 - 30.57088)
    if Q.n_s3d_above_10 < 3.0 and Q.sdb_4_z > 0.02669833:
        z += -0.4579315 * (3.0 - Q.n_s3d_above_10) * (Q.sdb_4_z - 0.02669833)
    if Q.lep_z < 0.5187302 and Q.sjq_3_3_k1 < -0.2912597:
        z += 0.5625201 * (0.5187302 - Q.lep_z) * (-0.2912597 - Q.sjq_3_3_k1)
    if Q.lam1 > 0.04304553 and Q.dc_3_n_lep > 0.0:
        z += -5.36586 * (Q.lam1 - 0.04304553) * (Q.dc_3_n_lep - 0.0)
    if Q.lam1 > 0.04304553 and Q.lund3_lnkt > 0.3409807:
        z += 0.7359447 * (Q.lam1 - 0.04304553) * (Q.lund3_lnkt - 0.3409807)
    if Q.lep_ptrel < 12.15228 and Q.sdb_3_n < 5.0:
        z += 0.00211425 * (12.15228 - Q.lep_ptrel) * (5.0 - Q.sdb_3_n)
    if Q.sjq_3_3_k1 > 0.4037488 and Q.iselectron_12 > 0.0:
        z += 0.2127448 * (Q.sjq_3_3_k1 - 0.4037488) * (Q.iselectron_12 - 0.0)
    if Q.pz_lnd0 < 0.05066241 and Q.dr_19 < 0.0538419:
        z += 204.7144 * (0.05066241 - Q.pz_lnd0) * (0.0538419 - Q.dr_19)
    if Q.tau32 < 0.6800935 and Q.sjf_4_2_mass_d3 > 3.450645:
        z += -0.003002021 * (0.6800935 - Q.tau32) * (Q.sjf_4_2_mass_d3 - 3.450645)
    if Q.pz_lnd0 < 0.05066241 and Q.lund2_lnkt > 4.507197:
        z += 2.435757 * (0.05066241 - Q.pz_lnd0) * (Q.lund2_lnkt - 4.507197)
    return z


def neuron_121(Q):
    z = -5.730522e-06
    return z


def neuron_122(Q):
    z = -2.3959e-06
    return z


def neuron_123(Q):
    z = 0.03816738
    if Q.lep_ptrel < 6.983043:
        z += 0.00544503 * Q.lep_ptrel - 0.03802288
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += -0.03136232 * Q.lep_ptrel + 0.8569317
    if Q.lep_ptrel >= 43.20788:
        z += 0.01871168 * Q.lep_ptrel - 1.30666
    if Q.M2_b2 < 0.02926638:
        z += -15.673 * Q.M2_b2 + 0.4586919
    if Q.z_displaced5 < 0.3803178:
        z += 0.9208534 * Q.z_displaced5 - 0.3502169
    if Q.tau32 < 0.6800935:
        z += 0.2466656 * Q.tau32 - 0.1677557
    if Q.n_pairs_kt_above_1 < 58.0:
        z += 0.00206542 * Q.n_pairs_kt_above_1 - 0.1197943
    if Q.lep_z < 0.06457187:
        z += 4.729478 * Q.lep_z - 0.3053912
    if 0.1275041 <= Q.lep_z < 0.221436:
        z += 1.197954 * Q.lep_z - 0.152744
    if Q.lep_z >= 0.221436:
        z += 1.502676 * Q.lep_z - 0.2202205
    if Q.mass_top5 >= 68.52153:
        z += 0.004222974 * Q.mass_top5 - 0.2893646
    if Q.dc_1_n_disp3 < 3.0:
        z += 0.0681015 * Q.dc_1_n_disp3 - 0.2043045
    if 130.0968 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.008392303 * Q.mres_sd_mass_b2z01 - 1.091812
    if Q.mres_sd_mass_b2z01 >= 158.2019:
        z += -0.0007559638 * Q.mres_sd_mass_b2z01 + 0.3554607
    if Q.pz_lnd2 < 0.05462126:
        z += -0.05940505 * Q.pz_lnd2 + 0.003244778
    z += 0.008884771 * Q.n_s3d_above_3
    if 20.80297 <= Q.mres_sd_mass_b1z01 < 73.23918:
        z += 0.002097042 * Q.mres_sd_mass_b1z01 - 0.0436247
    if 73.23918 <= Q.mres_sd_mass_b1z01 < 88.79082:
        z += -0.001871509 * Q.mres_sd_mass_b1z01 + 0.2470287
    if 88.79082 <= Q.mres_sd_mass_b1z01 < 105.3331:
        z += -0.005892421 * Q.mres_sd_mass_b1z01 + 0.6040487
    if Q.mres_sd_mass_b1z01 >= 105.3331:
        z += 0.0009433455 * Q.mres_sd_mass_b1z01 - 0.1159836
    if Q.sjq_3_2_nch < 3.0:
        z += 0.04876871 * Q.sjq_3_2_nch - 0.1463061
    if Q.sip_3d_3 < 4.606241:
        z += -0.064716 * Q.sip_3d_3 + 0.2980975
    if 0.2130551 <= Q.sjq_2_sumabs_k1 < 0.4723988:
        z += 0.1176884 * Q.sjq_2_sumabs_k1 - 0.02507411
    if Q.sjq_2_sumabs_k1 >= 0.4723988:
        z += 0.5715917 * Q.sjq_2_sumabs_k1 - 0.2394975
    if Q.sip_3d_2 < 226.3008:
        z += 0.0001757416 * Q.sip_3d_2 - 0.03977046
    if Q.pair_mean_lndelta >= -1.523797:
        z += -0.1722084 * Q.pair_mean_lndelta - 0.2624105
    if Q.mass_displaced3 < 13.03663:
        z += -0.008305602 * Q.mass_displaced3 + 0.1082771
    if Q.sjf_2_1_n_d3 >= 7.0:
        z += 0.07523096 * Q.sjf_2_1_n_d3 - 0.5266167
    if Q.lep_iso < 0.4381892:
        z += -0.1968027 * Q.lep_iso - 0.04827564
    if 0.4381892 <= Q.lep_iso < 2.907433:
        z += 0.05447516 * Q.lep_iso - 0.1583829
    if Q.sjf_2_2_max3d < 2.224391:
        z += -0.01956655 * Q.sjf_2_2_max3d + 0.04352365
    if Q.sj2_zsoft < 0.08996752:
        z += 3.117988 * Q.sj2_zsoft - 0.2805177
    if Q.n_dr_0p1_0p2 < 5.0:
        z += -0.01463166 * Q.n_dr_0p1_0p2 + 0.07315828
    if Q.pz_lnkt0 < 0.07823361:
        z += 2.154175 * Q.pz_lnkt0 - 0.1685289
    if Q.n_lund < 5.0:
        z += -0.03258168 * Q.n_lund + 0.1629084
    if Q.mres_sd_prong_mass2 < 5.543454:
        z += 0.01107428 * Q.mres_sd_prong_mass2 - 0.06138977
    if Q.mass_neutral < 14.92637:
        z += 0.01607786 * Q.mass_neutral - 0.2399841
    if Q.M3_b05 >= 0.08895667:
        z += -1.751306 * Q.M3_b05 + 0.1557904
    if Q.N2_b05 < 0.302405:
        z += 2.475957 * Q.N2_b05 - 0.7487416
    if Q.n_pairs_kt_above_3 < 41.0:
        z += -0.01564632 * Q.n_pairs_kt_above_3 + 0.6414993
    if Q.jd_3d_4 < 3.969624:
        z += 0.1047892 * Q.jd_3d_4 - 0.4096396
    if 3.969624 <= Q.jd_3d_4 < 172.888:
        z += -3.749919e-05 * Q.jd_3d_4 + 0.00648316
    z += 0.01371712 * Q.n_s3d_above_10
    if Q.z_charged_had < 0.3603262:
        z += -1.239751 * Q.z_charged_had + 0.4467149
    if Q.z_charged < 0.3265243:
        z += 2.462285 * Q.z_charged - 0.8039958
    if Q.tau21 < 0.135772:
        z += -3.924178 * Q.tau21 + 0.5327934
    if Q.sjq_2_1_nch < 5.0:
        z += -0.1199386 * Q.sjq_2_1_nch + 0.5996932
    if Q.pair_mean_lnz < -2.572052:
        z += -0.5082111 * Q.pair_mean_lnz - 1.307145
    if Q.z_displaced5 < 0.3803178 and Q.lep_z < 0.06457187:
        z += 16.36289 * (0.3803178 - Q.z_displaced5) * (0.06457187 - Q.lep_z)
    if Q.M2_b2 < 0.02926638 and Q.kt2_2_n_lep < 1.0:
        z += -12.67233 * (0.02926638 - Q.M2_b2) * (1.0 - Q.kt2_2_n_lep)
    if Q.M2_b2 < 0.02926638 and Q.dr_1 > 0.1975394:
        z += 1.705778 * (0.02926638 - Q.M2_b2) * (Q.dr_1 - 0.1975394)
    if Q.z_displaced5 < 0.3803178 and Q.M2_b05 > 0.1152127:
        z += 9.325117 * (0.3803178 - Q.z_displaced5) * (Q.M2_b05 - 0.1152127)
    if Q.sjq_3_2_nch < 3.0 and Q.sip_3d_3 < 4.606241:
        z += -0.04039454 * (3.0 - Q.sjq_3_2_nch) * (4.606241 - Q.sip_3d_3)
    if Q.lep_z < 0.06457187 and Q.sip_3d_3 < 4.606241:
        z += -0.09025729 * (0.06457187 - Q.lep_z) * (4.606241 - Q.sip_3d_3)
    if Q.sjq_2_sumabs_k1 > 0.4723988 and Q.kt2_2_n_lep < 1.0:
        z += -0.5807914 * (Q.sjq_2_sumabs_k1 - 0.4723988) * (1.0 - Q.kt2_2_n_lep)
    if Q.lep_ptrel > 27.3236 and Q.ecf_g41 > 6.216954e-05:
        z += -23.14155 * (Q.lep_ptrel - 27.3236) * (Q.ecf_g41 - 6.216954e-05)
    if Q.sjq_2_sumabs_k1 > 0.4723988 and Q.lepsj_2_dr < 0.0008210142:
        z += -309.6988 * (Q.sjq_2_sumabs_k1 - 0.4723988) * (0.0008210142 - Q.lepsj_2_dr)
    if Q.sip_3d_2 < 226.3008 and Q.sjq_2_prod_k1 < 0.09802954:
        z += 0.0006099798 * (226.3008 - Q.sip_3d_2) * (0.09802954 - Q.sjq_2_prod_k1)
    if Q.mres_sd_mass_b1z01 > 73.23918 and Q.z_charged_had < 0.3603262:
        z += 0.00729878 * (Q.mres_sd_mass_b1z01 - 73.23918) * (0.3603262 - Q.z_charged_had)
    if Q.sip_3d_2 < 226.3008 and Q.ak02_1_n_lep > 0.0:
        z += -9.220103e-05 * (226.3008 - Q.sip_3d_2) * (Q.ak02_1_n_lep - 0.0)
    if Q.lep_iso < 2.907433 and Q.ak02_2_n_lep < 1.0:
        z += -0.005786478 * (2.907433 - Q.lep_iso) * (1.0 - Q.ak02_2_n_lep)
    if Q.mres_sd_mass_b1z01 > 73.23918 and Q.jd_3d_4 < 3.969624:
        z += 0.0002509198 * (Q.mres_sd_mass_b1z01 - 73.23918) * (3.969624 - Q.jd_3d_4)
    if Q.lep_z > 0.1275041 and Q.lund2_lndelta > -1.052328:
        z += 0.2291697 * (Q.lep_z - 0.1275041) * (Q.lund2_lndelta - -1.052328)
    if Q.mres_sd_mass_b1z01 > 73.23918 and Q.lund2_lndelta < -0.3493007:
        z += -0.0002938786 * (Q.mres_sd_mass_b1z01 - 73.23918) * (-0.3493007 - Q.lund2_lndelta)
    if Q.sip_3d_2 < 226.3008 and Q.jd_3d_4 < 9.982976:
        z += 4.896352e-05 * (226.3008 - Q.sip_3d_2) * (9.982976 - Q.jd_3d_4)
    if Q.mass_top5 > 68.52153 and Q.n_lund_kt_above_5 > 2.0:
        z += -0.003821432 * (Q.mass_top5 - 68.52153) * (Q.n_lund_kt_above_5 - 2.0)
    if Q.sip_3d_2 < 226.3008 and Q.lepsj_2_n_d3 > 2.0:
        z += 0.0003324903 * (226.3008 - Q.sip_3d_2) * (Q.lepsj_2_n_d3 - 2.0)
    if Q.M2_b2 < 0.02926638 and Q.dc_ntag > 0.0:
        z += -6.705234 * (0.02926638 - Q.M2_b2) * (Q.dc_ntag - 0.0)
    if Q.dc_1_n_disp3 < 3.0 and Q.lepsj_2_dr < 0.0146198:
        z += 3.974235 * (3.0 - Q.dc_1_n_disp3) * (0.0146198 - Q.lepsj_2_dr)
    if Q.sjq_2_sumabs_k1 > 0.4723988 and Q.lund3_lnz > -4.227393:
        z += -0.05448627 * (Q.sjq_2_sumabs_k1 - 0.4723988) * (Q.lund3_lnz - -4.227393)
    if Q.mass_displaced3 < 13.03663 and Q.iselectron_1 > 0.0:
        z += 0.004281532 * (13.03663 - Q.mass_displaced3) * (Q.iselectron_1 - 0.0)
    if Q.sip_3d_3 < 4.606241 and Q.sjq_2_prod_k05 < 0.186324:
        z += -0.1001518 * (4.606241 - Q.sip_3d_3) * (0.186324 - Q.sjq_2_prod_k05)
    if Q.mres_sd_mass_b2z01 > 130.0968 and Q.lepsj_2_maxsd0 < 28.85058:
        z += -6.676518e-05 * (Q.mres_sd_mass_b2z01 - 130.0968) * (28.85058 - Q.lepsj_2_maxsd0)
    if Q.dc_1_n_disp3 < 3.0 and Q.lnptrel_40 < -5.255594:
        z += 0.0005742742 * (3.0 - Q.dc_1_n_disp3) * (-5.255594 - Q.lnptrel_40)
    if Q.mres_sd_mass_b2z01 > 158.2019 and Q.jd_3d_4 < 2.953464:
        z += -0.005193854 * (Q.mres_sd_mass_b2z01 - 158.2019) * (2.953464 - Q.jd_3d_4)
    if Q.mres_sd_mass_b1z01 > 73.23918 and Q.sj2_mass1 < 67.99453:
        z += 6.495622e-05 * (Q.mres_sd_mass_b1z01 - 73.23918) * (67.99453 - Q.sj2_mass1)
    if Q.lep_ptrel > 43.20788 and Q.phi_4 < -0.2275391:
        z += -0.0456866 * (Q.lep_ptrel - 43.20788) * (-0.2275391 - Q.phi_4)
    if Q.mres_sd_mass_b2z01 > 130.0968 and Q.dc_split1_z < 0.2632495:
        z += -0.0001521305 * (Q.mres_sd_mass_b2z01 - 130.0968) * (0.2632495 - Q.dc_split1_z)
    if Q.lep_z > 0.1275041 and Q.pz_lnkt2 > 0.07998846:
        z += 1.020168 * (Q.lep_z - 0.1275041) * (Q.pz_lnkt2 - 0.07998846)
    if Q.sjq_2_sumabs_k1 > 0.4723988 and Q.sjq_2_sumabs_k03 < 1.013911:
        z += -0.8989054 * (Q.sjq_2_sumabs_k1 - 0.4723988) * (1.013911 - Q.sjq_2_sumabs_k03)
    if Q.lep_z > 0.221436 and Q.eta_11 < -0.1245117:
        z += 1.06496 * (Q.lep_z - 0.221436) * (-0.1245117 - Q.eta_11)
    if Q.n_pairs_kt_above_1 < 58.0 and Q.lepsj_3_maxsd0 < 11.53372:
        z += 0.0005698351 * (58.0 - Q.n_pairs_kt_above_1) * (11.53372 - Q.lepsj_3_maxsd0)
    if Q.tau32 < 0.6800935 and Q.lepsj_3_maxsd0 < 586.9572:
        z += -0.0003038413 * (0.6800935 - Q.tau32) * (586.9572 - Q.lepsj_3_maxsd0)
    if Q.M2_b2 < 0.02926638 and Q.lepsj_2_maxsd0 < 1.546763:
        z += 0.5610802 * (0.02926638 - Q.M2_b2) * (1.546763 - Q.lepsj_2_maxsd0)
    if Q.lep_z > 0.1275041 and Q.sj3_dr13 < 0.676889:
        z += -1.33946 * (Q.lep_z - 0.1275041) * (0.676889 - Q.sj3_dr13)
    if Q.sjq_3_2_nch < 3.0 and Q.ak02_2_z < 0.03240401:
        z += 1.935959 * (3.0 - Q.sjq_3_2_nch) * (0.03240401 - Q.ak02_2_z)
    if Q.N2_b05 < 0.302405 and Q.sj2_zsoft < 0.1645421:
        z += 67.85689 * (0.302405 - Q.N2_b05) * (0.1645421 - Q.sj2_zsoft)
    if Q.mres_sd_prong_mass2 < 5.543454 and Q.z_dr_0p05_0p1 < 0.4355562:
        z += -0.03336668 * (5.543454 - Q.mres_sd_prong_mass2) * (0.4355562 - Q.z_dr_0p05_0p1)
    if Q.mres_sd_mass_b1z01 > 73.23918 and Q.lnpt_1 > 4.37968:
        z += 0.0003228268 * (Q.mres_sd_mass_b1z01 - 73.23918) * (Q.lnpt_1 - 4.37968)
    if Q.mass_neutral < 14.92637 and Q.mres_sd_zg_b2z01 < 0.04397771:
        z += 1.25227 * (14.92637 - Q.mass_neutral) * (0.04397771 - Q.mres_sd_zg_b2z01)
    if Q.M2_b2 < 0.02926638 and Q.dc_pair_mass_min > 105.4151:
        z += -0.1457066 * (0.02926638 - Q.M2_b2) * (Q.dc_pair_mass_min - 105.4151)
    if Q.lep_ptrel < 6.983043 and Q.jd_3d_4 < 3.969624:
        z += 0.004552522 * (6.983043 - Q.lep_ptrel) * (3.969624 - Q.jd_3d_4)
    if Q.sj2_zsoft < 0.08996752 and Q.pz_lnkt3 > 0.1312056:
        z += -142.299 * (0.08996752 - Q.sj2_zsoft) * (Q.pz_lnkt3 - 0.1312056)
    if Q.lep_ptrel > 27.3236 and Q.jd_3d_4 < 21.98452:
        z += -7.467923e-06 * (Q.lep_ptrel - 27.3236) * (21.98452 - Q.jd_3d_4)
    if Q.n_pairs_kt_above_3 < 41.0 and Q.sdb_0_z < 0.1294021:
        z += -0.08232135 * (41.0 - Q.n_pairs_kt_above_3) * (0.1294021 - Q.sdb_0_z)
    if Q.sjq_3_2_nch < 3.0 and Q.sjf_3_3_mass_d3 < 0.1527273:
        z += 1.03178 * (3.0 - Q.sjq_3_2_nch) * (0.1527273 - Q.sjf_3_3_mass_d3)
    if Q.lep_ptrel > 43.20788 and Q.eta_29 > 0.0:
        z += -0.04055986 * (Q.lep_ptrel - 43.20788) * (Q.eta_29 - 0.0)
    if Q.tau32 < 0.6800935 and Q.eta_29 > 0.2073975:
        z += 1.914117 * (0.6800935 - Q.tau32) * (Q.eta_29 - 0.2073975)
    if Q.lep_z > 0.221436 and Q.lepsj_2_dr < 0.04178742:
        z += -91.89491 * (Q.lep_z - 0.221436) * (0.04178742 - Q.lepsj_2_dr)
    if Q.lep_ptrel > 27.3236 and Q.min_pair_mass < 10.59762:
        z += 0.000675556 * (Q.lep_ptrel - 27.3236) * (10.59762 - Q.min_pair_mass)
    return z


def neuron_124(Q):
    z = -0.2895829
    if Q.lep_z < 0.221436:
        z += -3.346794 * Q.lep_z + 0.7851649
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -0.3727271 * Q.lep_z + 0.1265995
    if Q.n_pairs_kt_above_1 < 58.0:
        z += 0.001804712 * Q.n_pairs_kt_above_1 - 0.7979417
    if 58.0 <= Q.n_pairs_kt_above_1 < 366.0:
        z += 0.002250871 * Q.n_pairs_kt_above_1 - 0.8238189
    if Q.n_s3d_above_3 < 1.0:
        z += 0.3486039 * Q.n_s3d_above_3 - 0.3486039
    if Q.n_s3d_above_3 >= 3.0:
        z += 0.02706655 * Q.n_s3d_above_3 - 0.08119965
    if Q.mass < 95.14961:
        z += 3.833254e-05 * Q.mass - 0.02791276
    if 95.14961 <= Q.mass < 100.4835:
        z += 0.002720213 * Q.mass - 0.2830927
    if 100.4835 <= Q.mass < 117.4867:
        z += 0.0005737876 * Q.mass - 0.0674124
    if Q.lep_ptrel < 43.20788:
        z += -0.005863894 * Q.lep_ptrel + 0.2533665
    if Q.n_dr_0p4_up < 15.0:
        z += 0.007505631 * Q.n_dr_0p4_up - 0.1125845
    if Q.lep_iso < 0.4381892:
        z += 0.9761427 * Q.lep_iso - 0.62045
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += 0.2085872 * Q.lep_iso - 0.2841155
    if Q.n_lepton < 1.0:
        z += -0.7198942 * Q.n_lepton + 0.7198942
    if Q.mass_top40 >= 70.88236:
        z += -0.005080342 * Q.mass_top40 + 0.3601067
    if 76.40585 <= Q.mres_sd_mass_b0z005 < 86.65765:
        z += -0.00940408 * Q.mres_sd_mass_b0z005 + 0.7185268
    if 86.65765 <= Q.mres_sd_mass_b0z005 < 91.82593:
        z += -0.002016512 * Q.mres_sd_mass_b0z005 + 0.07833747
    if 91.82593 <= Q.mres_sd_mass_b0z005 < 111.9362:
        z += 0.005350639 * Q.mres_sd_mass_b0z005 - 0.5981581
    if 111.9362 <= Q.mres_sd_mass_b0z005 < 125.8718:
        z += 0.001583267 * Q.mres_sd_mass_b0z005 - 0.1764529
    if 125.8718 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += -0.02108494 * Q.mres_sd_mass_b0z005 + 2.676834
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.008464079 * Q.mres_sd_mass_b0z005 + 0.658453
    if Q.lepsj_3_maxsd0 < 0.8036986:
        z += 0.4087675 * Q.lepsj_3_maxsd0 - 0.3383795
    if 0.8036986 <= Q.lepsj_3_maxsd0 < 2.413264:
        z += 0.006121936 * Q.lepsj_3_maxsd0 - 0.01477385
    if Q.z_displaced3 < 0.04884386:
        z += 8.944065 * Q.z_displaced3 - 0.4368627
    if Q.sjq_2_prod_k1 < 0.03330871:
        z += 2.522058 * Q.sjq_2_prod_k1 - 0.2078262
    if 0.03330871 <= Q.sjq_2_prod_k1 < 0.09802954:
        z += 1.913135 * Q.sjq_2_prod_k1 - 0.1875438
    if 0.1103483 <= Q.sjq_2_sumabs_k1 < 0.2769039:
        z += -0.1193362 * Q.sjq_2_sumabs_k1 + 0.01316854
    if 0.2769039 <= Q.sjq_2_sumabs_k1 < 0.6652978:
        z += -0.8172656 * Q.sjq_2_sumabs_k1 + 0.2064279
    if Q.sjq_2_sumabs_k1 >= 0.6652978:
        z += -0.8447591 * Q.sjq_2_sumabs_k1 + 0.2247193
    if Q.z_photon >= 0.1149688:
        z += -1.486176 * Q.z_photon + 0.1708638
    if Q.lepsj_2_dr < 0.04178742:
        z += -5.199819 * Q.lepsj_2_dr + 0.217287
    if Q.sjq_2_prod_k05 >= 0.004703917:
        z += 0.5709093 * Q.sjq_2_prod_k05 - 0.00268551
    if Q.mass_displaced5 < 1.777787:
        z += 0.02663443 * Q.mass_displaced5 - 0.04735036
    if Q.lne_17 >= -0.1601665:
        z += -0.009990321 * Q.lne_17 - 0.001600115
    if Q.sjf_2_1_n_d3 >= 7.0:
        z += 0.2524482 * Q.sjf_2_1_n_d3 - 1.767137
    if Q.sjf_2_2_max3d < 1.864422:
        z += 0.4752232 * Q.sjf_2_2_max3d - 0.8860167
    if Q.tau2 < 0.1076451:
        z += -2.088579 * Q.tau2 + 0.2248253
    if Q.sj4_pair_mass_max < 66.70506:
        z += 0.01059077 * Q.sj4_pair_mass_max - 0.7064577
    if Q.sjf_4_1_n_d3 >= 2.0:
        z += 0.0005363902 * Q.sjf_4_1_n_d3 - 0.00107278
    if Q.pz_lnd0 < 0.06871203:
        z += 3.498069 * Q.pz_lnd0 - 0.2403594
    if Q.kt2_2_n_lep < 1.0:
        z += -0.126389 * Q.kt2_2_n_lep + 0.126389
    if Q.lep_dr >= 0.04607888:
        z += 1.02483 * Q.lep_dr - 0.04722304
    if Q.n_particles >= 67.0:
        z += 0.001332326 * Q.n_particles - 0.08926582
    z += 0.03997933 * Q.lepsj_2_n_d3
    if Q.sv_1_sd0_sum < 35.45098:
        z += -0.003894143 * Q.sv_1_sd0_sum + 0.1380512
    if Q.max_abs_d0 < 10.52344:
        z += 0.01198815 * Q.max_abs_d0 - 0.1261565
    if Q.z_displaced5 >= 0.04748739:
        z += 1.226748 * Q.z_displaced5 - 0.05825504
    if 69.04524 <= Q.mres_sd_prong_mass1 < 84.19324:
        z += -0.02020166 * Q.mres_sd_prong_mass1 + 1.394829
    if Q.mres_sd_prong_mass1 >= 84.19324:
        z += 0.01080243 * Q.mres_sd_prong_mass1 - 1.215506
    if Q.n_photon >= 5.0:
        z += 0.006184239 * Q.n_photon - 0.03092119
    if Q.sj3_pair_mass_max < 73.24742:
        z += 0.006895357 * Q.sj3_pair_mass_max - 0.5050671
    if Q.tau21_b2 >= 0.3898586:
        z += -0.1261564 * Q.tau21_b2 + 0.04918316
    if Q.jd_sum_abs_sd0_top5 < 55.83114:
        z += -0.01860697 * Q.jd_sum_abs_sd0_top5 + 1.038849
    if Q.jd_sum_abs_sd0_top3 < 48.72265:
        z += 0.01612209 * Q.jd_sum_abs_sd0_top3 - 0.7855111
    if Q.n_s3d_above_10 >= 2.0:
        z += -0.06353138 * Q.n_s3d_above_10 + 0.1270628
    if Q.sjq_3_prod_k1 < -0.06334099:
        z += 0.4085999 * Q.sjq_3_prod_k1 + 0.02588112
    if Q.dc_3_charge >= 0.6283033:
        z += -0.4445851 * Q.dc_3_charge + 0.2793343
    if Q.sdb_2_z < 0.1530389:
        z += 3.751049 * Q.sdb_2_z - 0.5740564
    if Q.lepsj_3_dr < 0.04982189:
        z += -15.38055 * Q.lepsj_3_dr + 0.7662883
    if Q.tau32 < 0.5498426:
        z += -0.6061553 * Q.tau32 + 0.3332901
    if Q.psi_0p1 < 0.006220408:
        z += 27.72976 * Q.psi_0p1 - 0.1724904
    if Q.lund_max_lndelta >= -0.529318:
        z += 0.4820238 * Q.lund_max_lndelta + 0.2551439
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 91.15481:
        z += -0.01861632 * Q.mres_sd_mass_b2z01 + 1.417687
    if 91.15481 <= Q.mres_sd_mass_b2z01 < 107.2089:
        z += 0.02037998 * Q.mres_sd_mass_b2z01 - 2.137014
    if Q.mres_sd_mass_b2z01 >= 107.2089:
        z += 0.003428947 * Q.mres_sd_mass_b2z01 - 0.3197113
    if Q.lep_ptrel < 43.20788 and Q.sj3_pair_mass_max < 142.9952:
        z += 6.193371e-05 * (43.20788 - Q.lep_ptrel) * (142.9952 - Q.sj3_pair_mass_max)
    if Q.n_s3d_above_3 > 3.0 and Q.M2_b2 < 0.06762785:
        z += -0.8185084 * (Q.n_s3d_above_3 - 3.0) * (0.06762785 - Q.M2_b2)
    if Q.n_s3d_above_3 > 3.0 and Q.sip_3d_1 < 578.2954:
        z += -1.488568e-05 * (Q.n_s3d_above_3 - 3.0) * (578.2954 - Q.sip_3d_1)
    if Q.mass_top40 > 70.88236 and Q.sv_n < 3.0:
        z += 1.156839e-05 * (Q.mass_top40 - 70.88236) * (3.0 - Q.sv_n)
    if Q.n_dr_0p4_up < 15.0 and Q.pz_lnd2 < 0.05462126:
        z += -0.07270854 * (15.0 - Q.n_dr_0p4_up) * (0.05462126 - Q.pz_lnd2)
    if Q.n_dr_0p4_up < 15.0 and Q.dc_ntag > 0.0:
        z += 0.004931525 * (15.0 - Q.n_dr_0p4_up) * (Q.dc_ntag - 0.0)
    if Q.lep_z < 0.3396572 and Q.z_displaced5 > 0.1046203:
        z += -6.049973 * (0.3396572 - Q.lep_z) * (Q.z_displaced5 - 0.1046203)
    if Q.mass_top40 > 70.88236 and Q.ak02_3_z < 0.07650476:
        z += -0.011832 * (Q.mass_top40 - 70.88236) * (0.07650476 - Q.ak02_3_z)
    if Q.mass_top40 > 70.88236 and Q.sdb_4_z > 0.0:
        z += -0.0008581905 * (Q.mass_top40 - 70.88236) * (Q.sdb_4_z - 0.0)
    if Q.n_pairs_kt_above_1 < 366.0 and Q.C2_b2 < 0.221369:
        z += 0.0009325185 * (366.0 - Q.n_pairs_kt_above_1) * (0.221369 - Q.C2_b2)
    if Q.tau2 < 0.1076451 and Q.jet_charge_k03 > -0.05990128:
        z += 2.50679 * (0.1076451 - Q.tau2) * (Q.jet_charge_k03 - -0.05990128)
    if Q.lep_z < 0.221436 and Q.dc_4_z > 0.0:
        z += 5.01671 * (0.221436 - Q.lep_z) * (Q.dc_4_z - 0.0)
    if Q.mass_top40 > 70.88236 and Q.jd_3d_5 < 4.004982:
        z += 7.885926e-05 * (Q.mass_top40 - 70.88236) * (4.004982 - Q.jd_3d_5)
    if Q.z_photon > 0.1149688 and Q.jd_3d_5 < 6.771002:
        z += 0.1205765 * (Q.z_photon - 0.1149688) * (6.771002 - Q.jd_3d_5)
    if Q.sjq_2_sumabs_k1 > 0.2769039 and Q.jd_3d_4 < 2.953464:
        z += 0.6017449 * (Q.sjq_2_sumabs_k1 - 0.2769039) * (2.953464 - Q.jd_3d_4)
    if Q.n_lepton < 1.0 and Q.jd_3d_4 < 2.953464:
        z += -0.3044702 * (1.0 - Q.n_lepton) * (2.953464 - Q.jd_3d_4)
    if Q.n_s3d_above_3 > 3.0 and Q.jd_3d_4 < 172.888:
        z += -0.0003333121 * (Q.n_s3d_above_3 - 3.0) * (172.888 - Q.jd_3d_4)
    if Q.mass_top40 > 70.88236 and Q.lund1_lndelta < -0.4311181:
        z += 0.0136718 * (Q.mass_top40 - 70.88236) * (-0.4311181 - Q.lund1_lndelta)
    if Q.max_abs_d0 < 10.52344 and Q.ak02_min12_n_disp3 < 1.0:
        z += 0.03328617 * (10.52344 - Q.max_abs_d0) * (1.0 - Q.ak02_min12_n_disp3)
    if Q.tau2 < 0.1076451 and Q.sjf_4_3_z_d3 > 0.009093044:
        z += 21.53133 * (0.1076451 - Q.tau2) * (Q.sjf_4_3_z_d3 - 0.009093044)
    if Q.kt2_2_n_lep < 1.0 and Q.tdz_0 < -0.04205891:
        z += 0.04672054 * (1.0 - Q.kt2_2_n_lep) * (-0.04205891 - Q.tdz_0)
    if Q.sv_1_sd0_sum < 35.45098 and Q.nca_sj4_pair2nd_over_mass < 0.556239:
        z += -0.006120067 * (35.45098 - Q.sv_1_sd0_sum) * (0.556239 - Q.nca_sj4_pair2nd_over_mass)
    if Q.kt2_2_n_lep < 1.0 and Q.sdb_2_z > 0.3208052:
        z += -0.8231202 * (1.0 - Q.kt2_2_n_lep) * (Q.sdb_2_z - 0.3208052)
    if Q.sdb_2_z < 0.1530389 and Q.dr_30 < 0.04891969:
        z += 32.3933 * (0.1530389 - Q.sdb_2_z) * (0.04891969 - Q.dr_30)
    if Q.sjq_2_sumabs_k1 > 0.2769039 and Q.lepsj_3_maxsd0 < 0.8036986:
        z += 0.4498972 * (Q.sjq_2_sumabs_k1 - 0.2769039) * (0.8036986 - Q.lepsj_3_maxsd0)
    if Q.sjq_2_prod_k1 < 0.09802954 and Q.lepsj_3_maxsd0 < 2.413264:
        z += 0.4151482 * (0.09802954 - Q.sjq_2_prod_k1) * (2.413264 - Q.lepsj_3_maxsd0)
    if Q.lep_z < 0.221436 and Q.tau43 < 0.8096444:
        z += -1.525241 * (0.221436 - Q.lep_z) * (0.8096444 - Q.tau43)
    if Q.n_s3d_above_3 > 3.0 and Q.ak02_2_sd0_3 < 2.627723:
        z += 1.203596e-06 * (Q.n_s3d_above_3 - 3.0) * (2.627723 - Q.ak02_2_sd0_3)
    if Q.tau32 < 0.5498426 and Q.isnhad_38 > 0.0:
        z += 0.4182495 * (0.5498426 - Q.tau32) * (Q.isnhad_38 - 0.0)
    if Q.tau21_b2 > 0.3898586 and Q.isnhad_8 < 1.0:
        z += -0.09801768 * (Q.tau21_b2 - 0.3898586) * (1.0 - Q.isnhad_8)
    if Q.lepsj_3_dr < 0.04982189 and Q.lepsj_3_maxsd0 < 2.413264:
        z += -3.334685 * (0.04982189 - Q.lepsj_3_dr) * (2.413264 - Q.lepsj_3_maxsd0)
    return z


def neuron_125(Q):
    z = -1.190763e-06
    return z


def neuron_126(Q):
    z = 7.63263e-06
    return z


def neuron_127(Q):
    z = -1.966689e-07
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
