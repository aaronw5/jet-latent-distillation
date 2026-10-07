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
    z = 0.005213402
    return z


def neuron_1(Q):
    z = -5.325262e-06
    return z


def neuron_2(Q):
    z = 5.447298e-06
    return z


def neuron_3(Q):
    z = 8.399834e-06
    return z


def neuron_4(Q):
    z = 2.792135e-06
    return z


def neuron_5(Q):
    z = -2.156917e-06
    return z


def neuron_6(Q):
    z = 1.770105e-05
    return z


def neuron_7(Q):
    z = 6.430819e-06
    return z


def neuron_8(Q):
    z = 2.926468e-06
    return z


def neuron_9(Q):
    z = -2.443659e-06
    return z


def neuron_10(Q):
    z = -1.570104e-05
    return z


def neuron_11(Q):
    z = -1.887712e-05
    return z


def neuron_12(Q):
    z = 5.099297e-06
    return z


def neuron_13(Q):
    z = 4.422908e-06
    return z


def neuron_14(Q):
    z = 2.845303e-06
    return z


def neuron_15(Q):
    z = -6.459521e-06
    return z


def neuron_16(Q):
    z = 1.803364
    z += 0.09406803 * Q.lep_ptrel
    z += -3.808192 * Q.z_charged_had
    z += 0.006423014 * Q.n_photon
    z += -0.1931565 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 2.512026 * (0.01578816 * 0.5 * ((0.08321691 - Q.pz_lnd2) / 0.01578816) * (1 + math.erf(((0.08321691 - Q.pz_lnd2) / 0.01578816) / math.sqrt(2))))
    z += -0.0001677709 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2))))
    z += -0.01513532 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2))))
    z += 0.005246999 * (3.20498 * 0.5 * ((Q.mass_neutral - 34.02385) / 3.20498) * (1 + math.erf(((Q.mass_neutral - 34.02385) / 3.20498) / math.sqrt(2))))
    z += -4.698186 * (0.08093679 * 0.5 * ((0.2801368 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.2801368 - Q.z_displaced5) / 0.08093679) / math.sqrt(2))))
    z += 25.67834 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2))))
    z += -0.02170765 * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.014163 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) / math.sqrt(2))))
    z += 0.0001475463 * (3.499256 * 0.5 * ((Q.mres_sd_mass_b2z01 - 121.6067) / 3.499256) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 121.6067) / 3.499256) / math.sqrt(2))))
    z += 0.007806699 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.01463339 * (3.043206 * 0.5 * ((55.84091 - Q.mass_charged) / 3.043206) * (1 + math.erf(((55.84091 - Q.mass_charged) / 3.043206) / math.sqrt(2))))
    z += 0.02585256 * (0.08093679 * 0.5 * ((0.2801368 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.2801368 - Q.z_displaced5) / 0.08093679) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2))))
    z += -0.01707225 * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((18.0 - Q.n_photon) / 1.0) * (1 + math.erf(((18.0 - Q.n_photon) / 1.0) / math.sqrt(2))))
    z += 0.069346 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2))))
    z += -0.01220988 * (1.0 * 0.5 * ((9.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((9.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2))))
    z += -0.04668187 * (4.214028 * 0.5 * ((9.982976 - Q.jd_3d_4) / 4.214028) * (1 + math.erf(((9.982976 - Q.jd_3d_4) / 4.214028) / math.sqrt(2))))
    z += 0.0004366246 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (4.06561 * 0.5 * ((Q.mass_2charged - 24.4079) / 4.06561) * (1 + math.erf(((Q.mass_2charged - 24.4079) / 4.06561) / math.sqrt(2))))
    z += 0.05649575 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sv_1_n - 1.0) / 1.0) * (1 + math.erf(((Q.sv_1_n - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.001654856 * (5.55459 * 0.5 * ((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2))))
    z += -0.07618619 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2))))
    z += 0.07819103 * (12.22004 * 0.5 * ((27.3236 - Q.lep_ptrel) / 12.22004) * (1 + math.erf(((27.3236 - Q.lep_ptrel) / 12.22004) / math.sqrt(2))))
    z += -0.01866366 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) / math.sqrt(2))))
    z += 6.380907 * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2)))) * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2))))
    z += 40.52089 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.5 * 0.5 * ((Q.n_lund - 7.0) / 1.5) * (1 + math.erf(((Q.n_lund - 7.0) / 1.5) / math.sqrt(2))))
    z += 0.6815875 * (0.02907957 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) / math.sqrt(2))))
    z += -0.1644803 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2))))
    z += 0.007322024 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += 0.916963 * (1.0 * 0.5 * ((0.0 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.0 - Q.sv_2_n) / 1.0) / math.sqrt(2))))
    z += -0.1590775 * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.9640846 * (0.002042627 * 0.5 * ((0.01892655 - Q.tau5) / 0.002042627) * (1 + math.erf(((0.01892655 - Q.tau5) / 0.002042627) / math.sqrt(2))))
    z += 0.00541244 * (5.119423 * 0.5 * ((Q.mass_top5 - 62.84493) / 5.119423) * (1 + math.erf(((Q.mass_top5 - 62.84493) / 5.119423) / math.sqrt(2))))
    z += -0.004331956 * (3.20498 * 0.5 * ((Q.mass_neutral - 34.02385) / 3.20498) * (1 + math.erf(((Q.mass_neutral - 34.02385) / 3.20498) / math.sqrt(2)))) * (0.203947 * 0.5 * ((2.345351 - Q.jd_3d_6) / 0.203947) * (1 + math.erf(((2.345351 - Q.jd_3d_6) / 0.203947) / math.sqrt(2))))
    z += -0.06065405 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2))))
    z += 0.6687069 * (0.004067431 * 0.5 * ((Q.sdb_5_z - 0.01083984) / 0.004067431) * (1 + math.erf(((Q.sdb_5_z - 0.01083984) / 0.004067431) / math.sqrt(2))))
    z += 2.22104 * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2))))
    z += -26.18566 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 2.0) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 2.0) / 1.0) / math.sqrt(2))))
    z += -0.06520257 * (0.02907957 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2))))
    z += -0.2316497 * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) * (1.0 * 0.5 * ((5.0 - Q.sjf_4_1_n_d3) / 1.0) * (1 + math.erf(((5.0 - Q.sjf_4_1_n_d3) / 1.0) / math.sqrt(2))))
    z += 0.0008659499 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (0.2811803 * 0.5 * ((2.552762 - Q.jd_3d_5) / 0.2811803) * (1 + math.erf(((2.552762 - Q.jd_3d_5) / 0.2811803) / math.sqrt(2))))
    z += -0.0005940126 * (14.45336 * 0.5 * ((Q.jd_3d_6 - 30.57088) / 14.45336) * (1 + math.erf(((Q.jd_3d_6 - 30.57088) / 14.45336) / math.sqrt(2))))
    z += 23056.31 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.001717815 * 0.5 * ((Q.M3 - 0.02540381) / 0.001717815) * (1 + math.erf(((Q.M3 - 0.02540381) / 0.001717815) / math.sqrt(2))))
    z += 0.04346246 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.05798278 * 0.5 * ((3.751708 - Q.lne_4) / 0.05798278) * (1 + math.erf(((3.751708 - Q.lne_4) / 0.05798278) / math.sqrt(2))))
    z += 0.05722528 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 1.0) / 1.0) * (1 + math.erf(((Q.n_muon - 1.0) / 1.0) / math.sqrt(2))))
    z += 5.176939 * (0.008519048 * 0.5 * ((0.01556887 - Q.z_dr_0p05_0p1) / 0.008519048) * (1 + math.erf(((0.01556887 - Q.z_dr_0p05_0p1) / 0.008519048) / math.sqrt(2))))
    z += 0.06928959 * (5.55459 * 0.5 * ((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) * (0.03294892 * 0.5 * ((0.08598434 - Q.kt2_2_z_disp3) / 0.03294892) * (1 + math.erf(((0.08598434 - Q.kt2_2_z_disp3) / 0.03294892) / math.sqrt(2))))
    z += 0.4345269 * (0.08093679 * 0.5 * ((0.2801368 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.2801368 - Q.z_displaced5) / 0.08093679) / math.sqrt(2)))) * (0.1240004 * 0.5 * ((7.075642 - Q.pair_max_lnm2) / 0.1240004) * (1 + math.erf(((7.075642 - Q.pair_max_lnm2) / 0.1240004) / math.sqrt(2))))
    z += -1470.221 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.04590778 * 0.5 * ((0.2054406 - Q.dr_max_012) / 0.04590778) * (1 + math.erf(((0.2054406 - Q.dr_max_012) / 0.04590778) / math.sqrt(2))))
    z += 0.04951872 * (1.278204 * 0.5 * ((3.101398 - Q.sjf_3_2_max3d) / 1.278204) * (1 + math.erf(((3.101398 - Q.sjf_3_2_max3d) / 1.278204) / math.sqrt(2))))
    z += -0.1454715 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.151853 * 0.5 * ((0.1373477 - Q.z_electron) / 0.151853) * (1 + math.erf(((0.1373477 - Q.z_electron) / 0.151853) / math.sqrt(2))))
    z += 0.8849326 * (0.06782196 * 0.5 * ((0.03898651 - Q.z_muon) / 0.06782196) * (1 + math.erf(((0.03898651 - Q.z_muon) / 0.06782196) / math.sqrt(2))))
    z += 0.01127062 * (3.485807 * 0.5 * ((Q.mass_top40 - 119.1279) / 3.485807) * (1 + math.erf(((Q.mass_top40 - 119.1279) / 3.485807) / math.sqrt(2))))
    z += 0.02095547 * (6.5 * 0.5 * ((41.0 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((41.0 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2))))
    z += 3.500465 * (0.01685699 * 0.5 * ((0.0815014 - Q.M2_b05) / 0.01685699) * (1 + math.erf(((0.0815014 - Q.M2_b05) / 0.01685699) / math.sqrt(2))))
    z += 0.1776329 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.002556514 * 0.5 * ((Q.sjf_4_3_z_d3 - 0.001692006) / 0.002556514) * (1 + math.erf(((Q.sjf_4_3_z_d3 - 0.001692006) / 0.002556514) / math.sqrt(2))))
    z += 1.505489 * (0.03043217 * 0.5 * ((0.1114751 - Q.sv_1_z) / 0.03043217) * (1 + math.erf(((0.1114751 - Q.sv_1_z) / 0.03043217) / math.sqrt(2))))
    z += 0.001231842 * (248.3135 * 0.5 * ((175.9957 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((175.9957 - Q.sip_3d_3) / 248.3135) / math.sqrt(2))))
    z += 0.1633172 * (0.3133958 * 0.5 * ((0.8906353 - Q.sjf_2_2_maxsd0) / 0.3133958) * (1 + math.erf(((0.8906353 - Q.sjf_2_2_maxsd0) / 0.3133958) / math.sqrt(2))))
    z += -0.01278082 * (1.118837 * 0.5 * ((6.465318 - Q.sj2_mass2) / 1.118837) * (1 + math.erf(((6.465318 - Q.sj2_mass2) / 1.118837) / math.sqrt(2))))
    z += -3.9714 * (0.07843207 * 0.5 * ((Q.lep_z - 0.1275041) / 0.07843207) * (1 + math.erf(((Q.lep_z - 0.1275041) / 0.07843207) / math.sqrt(2))))
    z += -3.296206 * (0.05422308 * 0.5 * ((Q.z_neutral - 0.1384639) / 0.05422308) * (1 + math.erf(((Q.z_neutral - 0.1384639) / 0.05422308) / math.sqrt(2))))
    z += 0.5977774 * (0.02550179 * 0.5 * ((0.1792389 - Q.sdb_2_z) / 0.02550179) * (1 + math.erf(((0.1792389 - Q.sdb_2_z) / 0.02550179) / math.sqrt(2))))
    z += 0.04483779 * (0.3128853 * 0.5 * ((1.113424 - Q.sjf_2_1_maxsd0) / 0.3128853) * (1 + math.erf(((1.113424 - Q.sjf_2_1_maxsd0) / 0.3128853) / math.sqrt(2))))
    z += -0.1792696 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += -3.780366 * (0.03445022 * 0.5 * ((0.04982189 - Q.lepsj_3_dr) / 0.03445022) * (1 + math.erf(((0.04982189 - Q.lepsj_3_dr) / 0.03445022) / math.sqrt(2))))
    z += -0.01926956 * (1.0 * 0.5 * ((3.0 - Q.dc_1_n_disp3) / 1.0) * (1 + math.erf(((3.0 - Q.dc_1_n_disp3) / 1.0) / math.sqrt(2))))
    z += -0.00493905 * (16.26099 * 0.5 * ((Q.mass_top50 - 161.1264) / 16.26099) * (1 + math.erf(((Q.mass_top50 - 161.1264) / 16.26099) / math.sqrt(2))))
    z += -0.112773 * (0.02907957 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.2130551) / 0.02907957) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.0) / 1.0) / math.sqrt(2))))
    z += -0.1386184 * (0.06782196 * 0.5 * ((0.03898651 - Q.z_muon) / 0.06782196) * (1 + math.erf(((0.03898651 - Q.z_muon) / 0.06782196) / math.sqrt(2)))) * (0.7263644 * 0.5 * ((Q.mres_sd_prong_mass2 - 1.685874e-06) / 0.7263644) * (1 + math.erf(((Q.mres_sd_prong_mass2 - 1.685874e-06) / 0.7263644) / math.sqrt(2))))
    z += 0.01601344 * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) * (5.497214 * 0.5 * ((7.621251 - Q.sjf_4_1_max3d) / 5.497214) * (1 + math.erf(((7.621251 - Q.sjf_4_1_max3d) / 5.497214) / math.sqrt(2))))
    z += -0.0006992919 * (5.877113 * 0.5 * ((40.03188 - Q.mres_sd_prong_mass1) / 5.877113) * (1 + math.erf(((40.03188 - Q.mres_sd_prong_mass1) / 5.877113) / math.sqrt(2))))
    z += -0.0001735761 * (5.55459 * 0.5 * ((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) * (11.51919 * 0.5 * ((15.37467 - Q.dc_pair_mass_min) / 11.51919) * (1 + math.erf(((15.37467 - Q.dc_pair_mass_min) / 11.51919) / math.sqrt(2))))
    z += -0.5136256 * (0.1750793 * 0.5 * ((0.4436035 - Q.max_abs_dz) / 0.1750793) * (1 + math.erf(((0.4436035 - Q.max_abs_dz) / 0.1750793) / math.sqrt(2))))
    z += -0.4400193 * (1.0 * 0.5 * ((0.0 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.0 - Q.sv_2_n) / 1.0) / math.sqrt(2)))) * (0.0225807 * 0.5 * ((0.8090312 - Q.tau43_b2) / 0.0225807) * (1 + math.erf(((0.8090312 - Q.tau43_b2) / 0.0225807) / math.sqrt(2))))
    z += 0.00229028 * (164.9991 * 0.5 * ((226.3008 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((226.3008 - Q.sip_3d_2) / 164.9991) / math.sqrt(2))))
    z += 0.004862883 * (0.03445022 * 0.5 * ((0.04982189 - Q.lepsj_3_dr) / 0.03445022) * (1 + math.erf(((0.04982189 - Q.lepsj_3_dr) / 0.03445022) / math.sqrt(2)))) * (467.791 * 0.5 * ((593.8527 - Q.dc_1_sd0_1) / 467.791) * (1 + math.erf(((593.8527 - Q.dc_1_sd0_1) / 467.791) / math.sqrt(2))))
    z += -0.2392381 * (0.560057 * 0.5 * ((1.777787 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.777787 - Q.mass_displaced5) / 0.560057) / math.sqrt(2))))
    z += 0.04213063 * (1.0 * 0.5 * ((4.0 - Q.n_sd0_above_5) / 1.0) * (1 + math.erf(((4.0 - Q.n_sd0_above_5) / 1.0) / math.sqrt(2))))
    z += -0.06198912 * (1.5 * 0.5 * ((Q.lepsj_2_n_d3 - 2.0) / 1.5) * (1 + math.erf(((Q.lepsj_2_n_d3 - 2.0) / 1.5) / math.sqrt(2))))
    z += 0.02315014 * (0.07843207 * 0.5 * ((Q.lep_z - 0.1275041) / 0.07843207) * (1 + math.erf(((Q.lep_z - 0.1275041) / 0.07843207) / math.sqrt(2)))) * (3.620548 * 0.5 * ((4.394157 - Q.sv_1_sd0_sum) / 3.620548) * (1 + math.erf(((4.394157 - Q.sv_1_sd0_sum) / 3.620548) / math.sqrt(2))))
    z += 0.04316767 * (1.0 * 0.5 * ((Q.sdb_0_n - 3.0) / 1.0) * (1 + math.erf(((Q.sdb_0_n - 3.0) / 1.0) / math.sqrt(2))))
    z += -0.493548 * (1.0 * 0.5 * ((0.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.05273902 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2))))
    z += -0.01342735 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.2798636 * 0.5 * ((1.720031 - Q.sjq_2_sumabs_k03) / 0.2798636) * (1 + math.erf(((1.720031 - Q.sjq_2_sumabs_k03) / 0.2798636) / math.sqrt(2))))
    z += -1.171033 * (1.0 * 0.5 * ((0.0 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.0 - Q.sv_2_n) / 1.0) / math.sqrt(2)))) * (0.03988647 * 0.5 * ((Q.d0err_49 - 0.0) / 0.03988647) * (1 + math.erf(((Q.d0err_49 - 0.0) / 0.03988647) / math.sqrt(2))))
    z += -0.0002079348 * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) * (81.78161 * 0.5 * ((Q.sv_1_sd0_sum - 125.2765) / 81.78161) * (1 + math.erf(((Q.sv_1_sd0_sum - 125.2765) / 81.78161) / math.sqrt(2))))
    z += -0.01172818 * (146.0039 * 0.5 * ((262.8757 - Q.jd_sum_abs_sd0_top3) / 146.0039) * (1 + math.erf(((262.8757 - Q.jd_sum_abs_sd0_top3) / 146.0039) / math.sqrt(2))))
    z += 0.00880946 * (148.6964 * 0.5 * ((288.7714 - Q.jd_sum_abs_sd0_top5) / 148.6964) * (1 + math.erf(((288.7714 - Q.jd_sum_abs_sd0_top5) / 148.6964) / math.sqrt(2))))
    z += -0.009546098 * (17.91682 * 0.5 * ((33.44111 - Q.jd_3d_4) / 17.91682) * (1 + math.erf(((33.44111 - Q.jd_3d_4) / 17.91682) / math.sqrt(2))))
    z += -41.2227 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dc_mass_disp_2nd - 0.0) / 1e-06) * (1 + math.erf(((Q.dc_mass_disp_2nd - 0.0) / 1e-06) / math.sqrt(2))))
    z += -0.005306442 * (10.55249 * 0.5 * ((Q.mass_2charged - 45.73288) / 10.55249) * (1 + math.erf(((Q.mass_2charged - 45.73288) / 10.55249) / math.sqrt(2))))
    z += -1.287897 * (1.0 * 0.5 * ((0.0 - Q.n_sd0_above_5) / 1.0) * (1 + math.erf(((0.0 - Q.n_sd0_above_5) / 1.0) / math.sqrt(2))))
    z += 0.002325201 * (112.8732 * 0.5 * ((206.0654 - Q.sip_3d_1) / 112.8732) * (1 + math.erf(((206.0654 - Q.sip_3d_1) / 112.8732) / math.sqrt(2))))
    z += -0.007290772 * (1.0 * 0.5 * ((3.0 - Q.dc_1_n_disp3) / 1.0) * (1 + math.erf(((3.0 - Q.dc_1_n_disp3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_pt_above_50 - 1.0) / 1.0) * (1 + math.erf(((Q.n_pt_above_50 - 1.0) / 1.0) / math.sqrt(2))))
    z += -0.01426284 * (1.0 * 0.5 * ((Q.sdb_2_n - 5.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 5.0) / 1.0) / math.sqrt(2))))
    z += -0.009335515 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2))))
    z += 0.01057681 * (1.5 * 0.5 * ((Q.n_dr_0p2_0p4 - 9.0) / 1.5) * (1 + math.erf(((Q.n_dr_0p2_0p4 - 9.0) / 1.5) / math.sqrt(2))))
    return z


def neuron_17(Q):
    z = -1.035346e-05
    return z


def neuron_18(Q):
    z = 0.4032546
    z += 0.2684243 * Q.n_lepton
    z += 0.02582727 * Q.sjf_4_n2disp
    z += -3.307585 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2))))
    z += 20.34063 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.01348041 * 0.5 * ((0.3829918 - Q.N2) / 0.01348041) * (1 + math.erf(((0.3829918 - Q.N2) / 0.01348041) / math.sqrt(2))))
    z += -0.01102669 * (4.146938 * 0.5 * ((Q.mass - 110.2019) / 4.146938) * (1 + math.erf(((Q.mass - 110.2019) / 4.146938) / math.sqrt(2))))
    z += -0.01184404 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) / math.sqrt(2))))
    z += -0.01060892 * (5.97473 * 0.5 * ((114.4658 - Q.mass_top20) / 5.97473) * (1 + math.erf(((114.4658 - Q.mass_top20) / 5.97473) / math.sqrt(2))))
    z += -6.799208 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.02343699 * 0.5 * ((Q.sdb_2_z - 0.2277628) / 0.02343699) * (1 + math.erf(((Q.sdb_2_z - 0.2277628) / 0.02343699) / math.sqrt(2))))
    z += 0.1985785 * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2))))
    z += -12.90311 * (0.01141967 * 0.5 * ((0.03624058 - Q.z_displaced3) / 0.01141967) * (1 + math.erf(((0.03624058 - Q.z_displaced3) / 0.01141967) / math.sqrt(2))))
    z += 0.03334101 * (1.5 * 0.5 * ((6.0 - Q.n_s3d_above_10) / 1.5) * (1 + math.erf(((6.0 - Q.n_s3d_above_10) / 1.5) / math.sqrt(2))))
    z += -134.5332 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2))))
    z += 0.4312531 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2))))
    z += -0.006587205 * (4.85476 * 0.5 * ((13.03663 - Q.mass_displaced3) / 4.85476) * (1 + math.erf(((13.03663 - Q.mass_displaced3) / 4.85476) / math.sqrt(2))))
    z += -0.09835181 * (0.3893833 * 0.5 * ((Q.lund3_lndelta - -2.817283) / 0.3893833) * (1 + math.erf(((Q.lund3_lndelta - -2.817283) / 0.3893833) / math.sqrt(2))))
    z += -0.0008516091 * (540.2081 * 0.5 * ((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2))))
    z += 0.003124902 * (5.55459 * 0.5 * ((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) / math.sqrt(2))))
    z += -27.01944 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.02653106 * 0.5 * ((0.1046203 - Q.z_displaced5) / 0.02653106) * (1 + math.erf(((0.1046203 - Q.z_displaced5) / 0.02653106) / math.sqrt(2))))
    z += -0.3917734 * (0.005299632 * 0.5 * ((Q.z_displaced5 - 0.006242101) / 0.005299632) * (1 + math.erf(((Q.z_displaced5 - 0.006242101) / 0.005299632) / math.sqrt(2))))
    z += -0.1355869 * (2.113111 * 0.5 * ((3.373037 - Q.sip_3d_1) / 2.113111) * (1 + math.erf(((3.373037 - Q.sip_3d_1) / 2.113111) / math.sqrt(2))))
    z += 0.06562457 * (0.05815218 * 0.5 * ((Q.lne_3 - 3.907638) / 0.05815218) * (1 + math.erf(((Q.lne_3 - 3.907638) / 0.05815218) / math.sqrt(2))))
    z += 0.02432523 * (16.90428 * 0.5 * ((Q.mass - 164.4374) / 16.90428) * (1 + math.erf(((Q.mass - 164.4374) / 16.90428) / math.sqrt(2))))
    z += -0.01615439 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) / math.sqrt(2))))
    z += -0.003401754 * (4.883139 * 0.5 * ((Q.mass_top50 - 89.68964) / 4.883139) * (1 + math.erf(((Q.mass_top50 - 89.68964) / 4.883139) / math.sqrt(2))))
    z += -0.01592558 * (11.17405 * 0.5 * ((Q.sj3_pair_mass_max - 128.6605) / 11.17405) * (1 + math.erf(((Q.sj3_pair_mass_max - 128.6605) / 11.17405) / math.sqrt(2))))
    z += -4.257448 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.03286901 * 0.5 * ((0.4839362 - Q.ak02_dr12) / 0.03286901) * (1 + math.erf(((0.4839362 - Q.ak02_dr12) / 0.03286901) / math.sqrt(2))))
    z += -1.527105 * (0.03429006 * 0.5 * ((0.08178299 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.08178299 - Q.lep_dr) / 0.03429006) / math.sqrt(2))))
    z += 5.912461e-05 * (4.883139 * 0.5 * ((Q.mass_top50 - 89.68964) / 4.883139) * (1 + math.erf(((Q.mass_top50 - 89.68964) / 4.883139) / math.sqrt(2)))) * (0.1268041 * 0.5 * ((0.9100211 - Q.lnpt_32) / 0.1268041) * (1 + math.erf(((0.9100211 - Q.lnpt_32) / 0.1268041) / math.sqrt(2))))
    z += 5.590982 * (0.01670814 * 0.5 * ((0.06245248 - Q.z_displaced5) / 0.01670814) * (1 + math.erf(((0.06245248 - Q.z_displaced5) / 0.01670814) / math.sqrt(2))))
    z += 0.2503419 * (1.5 * 0.5 * ((6.0 - Q.n_s3d_above_10) / 1.5) * (1 + math.erf(((6.0 - Q.n_s3d_above_10) / 1.5) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.2370407 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.2370407 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2))))
    z += 0.05764515 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2))))
    z += -0.01023355 * (0.1505243 * 0.5 * ((Q.lund_max_lnkt - 3.388322) / 0.1505243) * (1 + math.erf(((Q.lund_max_lnkt - 3.388322) / 0.1505243) / math.sqrt(2))))
    z += 0.02754007 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) / math.sqrt(2))))
    z += 0.004261142 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2))))
    z += 0.2497099 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2))))
    z += -0.1419732 * (1.0 * 0.5 * ((3.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((3.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += -0.03681989 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) * (0.003401978 * 0.5 * ((Q.sjf_3_3_z_d3 - 0.0) / 0.003401978) * (1 + math.erf(((Q.sjf_3_3_z_d3 - 0.0) / 0.003401978) / math.sqrt(2))))
    z += 0.06641858 * (0.1783594 * 0.5 * ((Q.ktd_ln_d34 - -9.180886) / 0.1783594) * (1 + math.erf(((Q.ktd_ln_d34 - -9.180886) / 0.1783594) / math.sqrt(2))))
    z += -0.001340989 * (4.85476 * 0.5 * ((13.03663 - Q.mass_displaced3) / 4.85476) * (1 + math.erf(((13.03663 - Q.mass_displaced3) / 4.85476) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 12.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 12.0) / 1.0) / math.sqrt(2))))
    z += 0.003338416 * (5.307922 * 0.5 * ((85.02716 - Q.mass) / 5.307922) * (1 + math.erf(((85.02716 - Q.mass) / 5.307922) / math.sqrt(2))))
    z += -0.07114381 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 158.2019) / 19.14168) / math.sqrt(2)))) * (0.04048364 * 0.5 * ((0.2759933 - Q.ak02_dr23) / 0.04048364) * (1 + math.erf(((0.2759933 - Q.ak02_dr23) / 0.04048364) / math.sqrt(2))))
    z += 0.006421532 * (5.55459 * 0.5 * ((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) / math.sqrt(2)))) * (0.01806359 * 0.5 * ((Q.sjf_2_2_z_d3 - 0.04865675) / 0.01806359) * (1 + math.erf(((Q.sjf_2_2_z_d3 - 0.04865675) / 0.01806359) / math.sqrt(2))))
    z += 0.00123284 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) / math.sqrt(2)))) * (0.09796308 * 0.5 * ((-0.258759 - Q.jet_charge) / 0.09796308) * (1 + math.erf(((-0.258759 - Q.jet_charge) / 0.09796308) / math.sqrt(2))))
    z += 0.02529137 * (0.05815218 * 0.5 * ((Q.lne_3 - 3.907638) / 0.05815218) * (1 + math.erf(((Q.lne_3 - 3.907638) / 0.05815218) / math.sqrt(2)))) * (0.2385164 * 0.5 * ((Q.ak02_dr23 - 0.0) / 0.2385164) * (1 + math.erf(((Q.ak02_dr23 - 0.0) / 0.2385164) / math.sqrt(2))))
    z += 0.06331729 * (1.0 * 0.5 * ((2.0 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((2.0 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2))))
    z += 0.05629788 * (1.0 * 0.5 * ((9.0 - Q.jd_n_d3_pt1) / 1.0) * (1 + math.erf(((9.0 - Q.jd_n_d3_pt1) / 1.0) / math.sqrt(2))))
    z += 0.01427896 * (4.85476 * 0.5 * ((13.03663 - Q.mass_displaced3) / 4.85476) * (1 + math.erf(((13.03663 - Q.mass_displaced3) / 4.85476) / math.sqrt(2)))) * (0.06729888 * 0.5 * ((Q.jet_abs_eta - 0.3952159) / 0.06729888) * (1 + math.erf(((Q.jet_abs_eta - 0.3952159) / 0.06729888) / math.sqrt(2))))
    z += 0.009870436 * (14.89455 * 0.5 * ((145.6038 - Q.mass_top20) / 14.89455) * (1 + math.erf(((145.6038 - Q.mass_top20) / 14.89455) / math.sqrt(2))))
    z += -0.02312704 * (2.5 * 0.5 * ((Q.n_particles - 21.0) / 2.5) * (1 + math.erf(((Q.n_particles - 21.0) / 2.5) / math.sqrt(2))))
    z += -0.001938132 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_1_n_lep - 0.0) / 1.0) * (1 + math.erf(((Q.dc_1_n_lep - 0.0) / 1.0) / math.sqrt(2))))
    z += 4.156046 * (0.008548367 * 0.5 * ((Q.N2_b05 - 0.4353632) / 0.008548367) * (1 + math.erf(((Q.N2_b05 - 0.4353632) / 0.008548367) / math.sqrt(2))))
    z += -4.117435 * (0.01539698 * 0.5 * ((Q.N2 - 0.3970676) / 0.01539698) * (1 + math.erf(((Q.N2 - 0.3970676) / 0.01539698) / math.sqrt(2))))
    z += -0.5189887 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.06557918 * 0.5 * ((Q.jet_charge_k05 - -0.215816) / 0.06557918) * (1 + math.erf(((Q.jet_charge_k05 - -0.215816) / 0.06557918) / math.sqrt(2))))
    z += -0.0001929686 * (188.4037 * 0.5 * ((828.2826 - Q.sdb_jp_all) / 188.4037) * (1 + math.erf(((828.2826 - Q.sdb_jp_all) / 188.4037) / math.sqrt(2))))
    z += 0.001155259 * (50.53748 * 0.5 * ((124.9603 - Q.jd_sum_abs_sd0_top5) / 50.53748) * (1 + math.erf(((124.9603 - Q.jd_sum_abs_sd0_top5) / 50.53748) / math.sqrt(2))))
    z += 0.01200705 * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2))))
    z += 0.008127666 * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2)))) * (0.02455054 * 0.5 * ((0.1435369 - Q.pz_lnd1) / 0.02455054) * (1 + math.erf(((0.1435369 - Q.pz_lnd1) / 0.02455054) / math.sqrt(2))))
    z += 1.324395 * (0.03770036 * 0.5 * ((0.3603262 - Q.z_charged_had) / 0.03770036) * (1 + math.erf(((0.3603262 - Q.z_charged_had) / 0.03770036) / math.sqrt(2))))
    z += -0.2436427 * (2.463307 * 0.5 * ((3.710567 - Q.sjf_3_1_max3d) / 2.463307) * (1 + math.erf(((3.710567 - Q.sjf_3_1_max3d) / 2.463307) / math.sqrt(2))))
    z += -0.1203561 * (3.069852 * 0.5 * ((4.516968 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.516968 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2))))
    z += 0.9474452 * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.04808774 * 0.5 * ((0.09790963 - Q.lepsj_3_dr) / 0.04808774) * (1 + math.erf(((0.09790963 - Q.lepsj_3_dr) / 0.04808774) / math.sqrt(2))))
    z += 0.2622598 * (0.1763298 * 0.5 * ((2.078395 - Q.sjf_3_1_max3d) / 0.1763298) * (1 + math.erf(((2.078395 - Q.sjf_3_1_max3d) / 0.1763298) / math.sqrt(2))))
    z += 0.5845139 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2)))) * (117.3924 * 0.5 * ((132.2798 - Q.sjf_3_1_max3d) / 117.3924) * (1 + math.erf(((132.2798 - Q.sjf_3_1_max3d) / 117.3924) / math.sqrt(2))))
    z += -0.01214428 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (1.043533 * 0.5 * ((1.296719 - Q.ak02_3_mass) / 1.043533) * (1 + math.erf(((1.296719 - Q.ak02_3_mass) / 1.043533) / math.sqrt(2))))
    z += 38.66216 * (0.005299632 * 0.5 * ((Q.z_displaced5 - 0.006242101) / 0.005299632) * (1 + math.erf(((Q.z_displaced5 - 0.006242101) / 0.005299632) / math.sqrt(2)))) * (0.003791252 * 0.5 * ((0.01937651 - Q.dr12) / 0.003791252) * (1 + math.erf(((0.01937651 - Q.dr12) / 0.003791252) / math.sqrt(2))))
    z += 0.08498979 * (1.268269 * 0.5 * ((5.311506 - Q.sj2_mass2) / 1.268269) * (1 + math.erf(((5.311506 - Q.sj2_mass2) / 1.268269) / math.sqrt(2))))
    z += -0.0004862338 * (22.5 * 0.5 * ((123.0 - Q.n_pairs_kt_above_1) / 22.5) * (1 + math.erf(((123.0 - Q.n_pairs_kt_above_1) / 22.5) / math.sqrt(2))))
    z += 16.67574 * (0.005001016 * 0.5 * ((0.05045808 - Q.tau5) / 0.005001016) * (1 + math.erf(((0.05045808 - Q.tau5) / 0.005001016) / math.sqrt(2))))
    z += -32.3138 * (0.001754707 * 0.5 * ((0.009042742 - Q.sum_z_dr2_top2) / 0.001754707) * (1 + math.erf(((0.009042742 - Q.sum_z_dr2_top2) / 0.001754707) / math.sqrt(2))))
    z += 0.04436034 * (1.0 * 0.5 * ((Q.sv_2_n - 1.0) / 1.0) * (1 + math.erf(((Q.sv_2_n - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.07234675 * (1.0 * 0.5 * ((2.0 - Q.sjq_3_3_nch) / 1.0) * (1 + math.erf(((2.0 - Q.sjq_3_3_nch) / 1.0) / math.sqrt(2))))
    z += -0.03464308 * (4.481852 * 0.5 * ((6.220692 - Q.sjf_2_1_max3d) / 4.481852) * (1 + math.erf(((6.220692 - Q.sjf_2_1_max3d) / 4.481852) / math.sqrt(2))))
    z += 0.2409053 * (0.07970627 * 0.5 * ((0.5726046 - Q.sdb_2_z) / 0.07970627) * (1 + math.erf(((0.5726046 - Q.sdb_2_z) / 0.07970627) / math.sqrt(2))))
    z += 0.02150707 * (0.6933428 * 0.5 * ((2.382955 - Q.mass_displaced5) / 0.6933428) * (1 + math.erf(((2.382955 - Q.mass_displaced5) / 0.6933428) / math.sqrt(2))))
    z += 647.4588 * (3.838712e-05 * 0.5 * ((0.000125186 - Q.e3_b2) / 3.838712e-05) * (1 + math.erf(((0.000125186 - Q.e3_b2) / 3.838712e-05) / math.sqrt(2))))
    z += 2.240717e-05 * (4.176093 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 72.72359) / 4.176093) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 72.72359) / 4.176093) / math.sqrt(2))))
    z += -0.0566147 * (0.07970627 * 0.5 * ((0.5726046 - Q.sdb_2_z) / 0.07970627) * (1 + math.erf(((0.5726046 - Q.sdb_2_z) / 0.07970627) / math.sqrt(2)))) * (0.1182861 * 0.5 * ((0.3249512 - Q.eta_28) / 0.1182861) * (1 + math.erf(((0.3249512 - Q.eta_28) / 0.1182861) / math.sqrt(2))))
    z += -0.001309232 * (18.98174 * 0.5 * ((14.38858 - Q.lep_iso) / 18.98174) * (1 + math.erf(((14.38858 - Q.lep_iso) / 18.98174) / math.sqrt(2))))
    z += -3.476655 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2))))
    z += -0.3390543 * (1.0 * 0.5 * ((0.0 - Q.sdb_4_n) / 1.0) * (1 + math.erf(((0.0 - Q.sdb_4_n) / 1.0) / math.sqrt(2))))
    z += 0.00169094 * (25.87632 * 0.5 * ((29.2558 - Q.mres_sd_mass_b2z01) / 25.87632) * (1 + math.erf(((29.2558 - Q.mres_sd_mass_b2z01) / 25.87632) / math.sqrt(2))))
    z += -1.428204 * (0.02426666 * 0.5 * ((0.04397771 - Q.mres_sd_zg_b2z01) / 0.02426666) * (1 + math.erf(((0.04397771 - Q.mres_sd_zg_b2z01) / 0.02426666) / math.sqrt(2))))
    z += -0.002497659 * (16.90428 * 0.5 * ((Q.mass - 164.4374) / 16.90428) * (1 + math.erf(((Q.mass - 164.4374) / 16.90428) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dr_94 - 0.0) / 1e-06) * (1 + math.erf(((Q.dr_94 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 5.568233 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2)))) * (0.02420044 * 0.5 * ((0.04159546 - Q.dzerr_32) / 0.02420044) * (1 + math.erf(((0.04159546 - Q.dzerr_32) / 0.02420044) / math.sqrt(2))))
    z += 0.009891348 * (3.069852 * 0.5 * ((4.516968 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.516968 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) * (0.2888875 * 0.5 * ((Q.dc_3_charge - -0.6306676) / 0.2888875) * (1 + math.erf(((Q.dc_3_charge - -0.6306676) / 0.2888875) / math.sqrt(2))))
    z += -57.4184 * (0.005001016 * 0.5 * ((0.05045808 - Q.tau5) / 0.005001016) * (1 + math.erf(((0.05045808 - Q.tau5) / 0.005001016) / math.sqrt(2)))) * (0.01243073 * 0.5 * ((Q.ak02_3_z - 0.01956504) / 0.01243073) * (1 + math.erf(((Q.ak02_3_z - 0.01956504) / 0.01243073) / math.sqrt(2))))
    z += -0.08075444 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (0.4913027 * 0.5 * ((-5.150973 - Q.ktd_ln_d23) / 0.4913027) * (1 + math.erf(((-5.150973 - Q.ktd_ln_d23) / 0.4913027) / math.sqrt(2))))
    z += 0.0003067458 * (188.4037 * 0.5 * ((828.2826 - Q.sdb_jp_all) / 188.4037) * (1 + math.erf(((828.2826 - Q.sdb_jp_all) / 188.4037) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.ischhad_0) / 1.0) * (1 + math.erf(((0.0 - Q.ischhad_0) / 1.0) / math.sqrt(2))))
    z += 0.01213824 * (0.03770036 * 0.5 * ((0.3603262 - Q.z_charged_had) / 0.03770036) * (1 + math.erf(((0.3603262 - Q.z_charged_had) / 0.03770036) / math.sqrt(2)))) * (1.721646 * 0.5 * ((Q.mass_2photon - 5.763861) / 1.721646) * (1 + math.erf(((Q.mass_2photon - 5.763861) / 1.721646) / math.sqrt(2))))
    z += -0.7737151 * (0.117671 * 0.5 * ((Q.lep_dr - 0.4191372) / 0.117671) * (1 + math.erf(((Q.lep_dr - 0.4191372) / 0.117671) / math.sqrt(2))))
    z += 0.006264606 * (11.17405 * 0.5 * ((Q.sj3_pair_mass_max - 128.6605) / 11.17405) * (1 + math.erf(((Q.sj3_pair_mass_max - 128.6605) / 11.17405) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.tdz_72 - 0.0) / 1e-06) * (1 + math.erf(((Q.tdz_72 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.01248822 * (0.03429006 * 0.5 * ((0.08178299 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.08178299 - Q.lep_dr) / 0.03429006) / math.sqrt(2)))) * (0.1131331 * 0.5 * ((Q.sjf_4_1_mass_d3 - 0.0) / 0.1131331) * (1 + math.erf(((Q.sjf_4_1_mass_d3 - 0.0) / 0.1131331) / math.sqrt(2))))
    z += 0.45423 * (0.05815218 * 0.5 * ((Q.lne_3 - 3.907638) / 0.05815218) * (1 + math.erf(((Q.lne_3 - 3.907638) / 0.05815218) / math.sqrt(2)))) * (0.02199402 * 0.5 * ((Q.eta_9 - 0.07116699) / 0.02199402) * (1 + math.erf(((Q.eta_9 - 0.07116699) / 0.02199402) / math.sqrt(2))))
    z += -0.0200643 * (2.0 * 0.5 * ((27.0 - Q.n_for_90pct) / 2.0) * (1 + math.erf(((27.0 - Q.n_for_90pct) / 2.0) / math.sqrt(2))))
    z += -0.0002203673 * (50.53748 * 0.5 * ((124.9603 - Q.jd_sum_abs_sd0_top5) / 50.53748) * (1 + math.erf(((124.9603 - Q.jd_sum_abs_sd0_top5) / 50.53748) / math.sqrt(2)))) * (0.1431331 * 0.5 * ((Q.kt2_1_sd0_3 - 1.376447) / 0.1431331) * (1 + math.erf(((Q.kt2_1_sd0_3 - 1.376447) / 0.1431331) / math.sqrt(2))))
    z += -0.04959069 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += -740.579 * (0.001001007 * 0.5 * ((0.0006725139 - Q.sum_z_dr2_top3) / 0.001001007) * (1 + math.erf(((0.0006725139 - Q.sum_z_dr2_top3) / 0.001001007) / math.sqrt(2))))
    z += 0.0662777 * (0.07645116 * 0.5 * ((Q.lund2_lndelta - -1.124734) / 0.07645116) * (1 + math.erf(((Q.lund2_lndelta - -1.124734) / 0.07645116) / math.sqrt(2))))
    return z


def neuron_19(Q):
    z = -1.407828e-06
    return z


def neuron_20(Q):
    z = 8.253669e-07
    return z


def neuron_21(Q):
    z = -0.001683665
    return z


def neuron_22(Q):
    z = -1.044852e-06
    return z


def neuron_23(Q):
    z = -6.978848e-07
    return z


def neuron_24(Q):
    z = 0.6614335
    z += 0.2329454 * Q.pair_max_lnkt
    z += -2.938836 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2))))
    z += 0.01500506 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (5.110145 * 0.5 * ((131.3917 - Q.mass) / 5.110145) * (1 + math.erf(((131.3917 - Q.mass) / 5.110145) / math.sqrt(2))))
    z += -0.008635147 * (6.170706 * 0.5 * ((74.51927 - Q.mass_top30) / 6.170706) * (1 + math.erf(((74.51927 - Q.mass_top30) / 6.170706) / math.sqrt(2))))
    z += 0.01335694 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2))))
    z += 0.54359 * (0.02561531 * 0.5 * ((0.1061578 - Q.z_displaced3) / 0.02561531) * (1 + math.erf(((0.1061578 - Q.z_displaced3) / 0.02561531) / math.sqrt(2))))
    z += -905.3846 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2))))
    z += -0.007461182 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2))))
    z += -0.003877543 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2))))
    z += -0.07216365 * (4.971962 * 0.5 * ((81.19466 - Q.mres_sd_mass_b2z01) / 4.971962) * (1 + math.erf(((81.19466 - Q.mres_sd_mass_b2z01) / 4.971962) / math.sqrt(2))))
    z += 0.0215113 * (3.283929 * 0.5 * ((118.2746 - Q.mres_sd_mass_b2z01) / 3.283929) * (1 + math.erf(((118.2746 - Q.mres_sd_mass_b2z01) / 3.283929) / math.sqrt(2))))
    z += 0.174357 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2))))
    z += -0.02764843 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 1.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 1.0) / 1.0) / math.sqrt(2))))
    z += -6.646577 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2))))
    z += 5.074636 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.05582565 * 0.5 * ((0.456416 - Q.tau32) / 0.05582565) * (1 + math.erf(((0.456416 - Q.tau32) / 0.05582565) / math.sqrt(2))))
    z += 0.01451373 * (6.875137 * 0.5 * ((18.80005 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.80005 - Q.mass_displaced3) / 6.875137) / math.sqrt(2))))
    z += 0.6748785 * (2.113111 * 0.5 * ((3.373037 - Q.sip_3d_1) / 2.113111) * (1 + math.erf(((3.373037 - Q.sip_3d_1) / 2.113111) / math.sqrt(2))))
    z += 1.708699 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += 4.726912 * (0.001965812 * 0.5 * ((Q.M3 - 0.03259227) / 0.001965812) * (1 + math.erf(((Q.M3 - 0.03259227) / 0.001965812) / math.sqrt(2))))
    z += -0.01474726 * (18.42178 * 0.5 * ((182.8592 - Q.mass) / 18.42178) * (1 + math.erf(((182.8592 - Q.mass) / 18.42178) / math.sqrt(2))))
    z += 0.05914178 * (0.01435892 * 0.5 * ((Q.pz_lnd3 - 0.1187422) / 0.01435892) * (1 + math.erf(((Q.pz_lnd3 - 0.1187422) / 0.01435892) / math.sqrt(2))))
    z += 2.491033 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2))))
    z += -0.05663449 * (6.875137 * 0.5 * ((18.80005 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.80005 - Q.mass_displaced3) / 6.875137) / math.sqrt(2)))) * (0.09779513 * 0.5 * ((0.6703997 - Q.sdb_2_z) / 0.09779513) * (1 + math.erf(((0.6703997 - Q.sdb_2_z) / 0.09779513) / math.sqrt(2))))
    z += 0.07378573 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) * (0.01776485 * 0.5 * ((0.2714017 - Q.N2_b2) / 0.01776485) * (1 + math.erf(((0.2714017 - Q.N2_b2) / 0.01776485) / math.sqrt(2))))
    z += 0.005761052 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.08912976 * 0.5 * ((Q.lne_0 - 4.817758) / 0.08912976) * (1 + math.erf(((Q.lne_0 - 4.817758) / 0.08912976) / math.sqrt(2))))
    z += -119150.4 * (0.02561531 * 0.5 * ((0.1061578 - Q.z_displaced3) / 0.02561531) * (1 + math.erf(((0.1061578 - Q.z_displaced3) / 0.02561531) / math.sqrt(2)))) * (2.8693e-06 * 0.5 * ((2.139972e-05 - Q.ecf_g42) / 2.8693e-06) * (1 + math.erf(((2.139972e-05 - Q.ecf_g42) / 2.8693e-06) / math.sqrt(2))))
    z += -0.4297068 * (0.4432951 * 0.5 * ((2.802481 - Q.sip_3d_1) / 0.4432951) * (1 + math.erf(((2.802481 - Q.sip_3d_1) / 0.4432951) / math.sqrt(2))))
    z += -0.00237589 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.sv_1_n) / 1.0) * (1 + math.erf(((3.0 - Q.sv_1_n) / 1.0) / math.sqrt(2))))
    z += 0.004884035 * (4.971962 * 0.5 * ((81.19466 - Q.mres_sd_mass_b2z01) / 4.971962) * (1 + math.erf(((81.19466 - Q.mres_sd_mass_b2z01) / 4.971962) / math.sqrt(2)))) * (0.02709999 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.185133) / 0.02709999) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.185133) / 0.02709999) / math.sqrt(2))))
    z += 0.04035893 * (10.51039 * 0.5 * ((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) * (1 + math.erf(((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) / math.sqrt(2))))
    z += 0.002808074 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) * (1.15674 * 0.5 * ((Q.lne_17 - -0.1601665) / 1.15674) * (1 + math.erf(((Q.lne_17 - -0.1601665) / 1.15674) / math.sqrt(2))))
    z += 0.1363385 * (0.002798005 * 0.5 * ((-0.003234757 - Q.sjq_2_prod_k1) / 0.002798005) * (1 + math.erf(((-0.003234757 - Q.sjq_2_prod_k1) / 0.002798005) / math.sqrt(2))))
    z += -12.77074 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (0.02224606 * 0.5 * ((0.2644738 - Q.sj2_dr) / 0.02224606) * (1 + math.erf(((0.2644738 - Q.sj2_dr) / 0.02224606) / math.sqrt(2))))
    z += 23.11695 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) * (0.01972221 * 0.5 * ((0.2848657 - Q.sj2_dr) / 0.01972221) * (1 + math.erf(((0.2848657 - Q.sj2_dr) / 0.01972221) / math.sqrt(2))))
    z += 2.687479 * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2))))
    z += 0.5068515 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.1852805 * 0.5 * ((Q.sjq_2_2_k1 - 0.3796859) / 0.1852805) * (1 + math.erf(((Q.sjq_2_2_k1 - 0.3796859) / 0.1852805) / math.sqrt(2))))
    z += -0.01941567 * (5.270719 * 0.5 * ((86.14266 - Q.mres_pruned_mass) / 5.270719) * (1 + math.erf(((86.14266 - Q.mres_pruned_mass) / 5.270719) / math.sqrt(2))))
    z += 0.07539248 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dc_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.dc_3_n_lep - 0.0) / 1e-06) / math.sqrt(2))))
    z += 3.721837e-05 * (3.283929 * 0.5 * ((118.2746 - Q.mres_sd_mass_b2z01) / 3.283929) * (1 + math.erf(((118.2746 - Q.mres_sd_mass_b2z01) / 3.283929) / math.sqrt(2)))) * (9.32425 * 0.5 * ((Q.lnpt_43 - -0.08134564) / 9.32425) * (1 + math.erf(((Q.lnpt_43 - -0.08134564) / 9.32425) / math.sqrt(2))))
    z += -0.09148651 * (1.0 * 0.5 * ((9.0 - Q.n_sd0_above_3) / 1.0) * (1 + math.erf(((9.0 - Q.n_sd0_above_3) / 1.0) / math.sqrt(2))))
    z += 0.08879016 * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += -8.675677e-05 * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (395.2298 * 0.5 * ((509.6197 - Q.sv_1_sd0_sum) / 395.2298) * (1 + math.erf(((509.6197 - Q.sv_1_sd0_sum) / 395.2298) / math.sqrt(2))))
    z += -0.4940983 * (1.0 * 0.5 * ((9.0 - Q.n_sd0_above_3) / 1.0) * (1 + math.erf(((9.0 - Q.n_sd0_above_3) / 1.0) / math.sqrt(2)))) * (0.003395811 * 0.5 * ((Q.sv_2_z - 0.00726873) / 0.003395811) * (1 + math.erf(((Q.sv_2_z - 0.00726873) / 0.003395811) / math.sqrt(2))))
    z += 0.01590951 * (4.167088 * 0.5 * ((119.8459 - Q.mres_pruned_mass) / 4.167088) * (1 + math.erf(((119.8459 - Q.mres_pruned_mass) / 4.167088) / math.sqrt(2))))
    z += 433.4986 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.05415789 * 0.5 * ((-2.05292 - Q.lnerel_1) / 0.05415789) * (1 + math.erf(((-2.05292 - Q.lnerel_1) / 0.05415789) / math.sqrt(2))))
    z += -0.4153858 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.01780783 * (3.24083 * 0.5 * ((6.161303 - Q.dc_2_jp) / 3.24083) * (1 + math.erf(((6.161303 - Q.dc_2_jp) / 3.24083) / math.sqrt(2))))
    z += 0.02494634 * (1.0 * 0.5 * ((Q.n_sd0_above_10 - 4.0) / 1.0) * (1 + math.erf(((Q.n_sd0_above_10 - 4.0) / 1.0) / math.sqrt(2))))
    z += 73.26302 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (1.364915e-06 * 0.5 * ((Q.ecf_g43 - 6.840827e-06) / 1.364915e-06) * (1 + math.erf(((Q.ecf_g43 - 6.840827e-06) / 1.364915e-06) / math.sqrt(2))))
    z += -4.544719 * (0.07842787 * 0.5 * ((Q.dc_split2_dr - 0.3591078) / 0.07842787) * (1 + math.erf(((Q.dc_split2_dr - 0.3591078) / 0.07842787) / math.sqrt(2))))
    z += -16.44763 * (0.01304025 * 0.5 * ((0.01245833 - Q.lep_z) / 0.01304025) * (1 + math.erf(((0.01245833 - Q.lep_z) / 0.01304025) / math.sqrt(2))))
    z += 3.350682 * (0.06488792 * 0.5 * ((Q.dc_split2_dr - 0.2806799) / 0.06488792) * (1 + math.erf(((Q.dc_split2_dr - 0.2806799) / 0.06488792) / math.sqrt(2))))
    z += 0.004930576 * (8.537109 * 0.5 * ((17.60938 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((17.60938 - Q.max_abs_d0) / 8.537109) / math.sqrt(2))))
    z += -0.01347583 * (0.7154015 * 0.5 * ((Q.mass_displaced3 - 2.409613) / 0.7154015) * (1 + math.erf(((Q.mass_displaced3 - 2.409613) / 0.7154015) / math.sqrt(2))))
    z += -0.05051524 * (15.148 * 0.5 * ((Q.mres_sd_prong_mass1 - 84.19324) / 15.148) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 84.19324) / 15.148) / math.sqrt(2))))
    z += 0.03961324 * (14.176 * 0.5 * ((Q.mres_sd_prong_mass1 - 69.04524) / 14.176) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 69.04524) / 14.176) / math.sqrt(2))))
    z += 1.370956 * (0.06488792 * 0.5 * ((Q.dc_split2_dr - 0.2806799) / 0.06488792) * (1 + math.erf(((Q.dc_split2_dr - 0.2806799) / 0.06488792) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dr_86 - 0.0) / 1e-06) * (1 + math.erf(((Q.dr_86 - 0.0) / 1e-06) / math.sqrt(2))))
    z += -82.40536 * (0.002611265 * 0.5 * ((0.01015545 - Q.M2_b2) / 0.002611265) * (1 + math.erf(((0.01015545 - Q.M2_b2) / 0.002611265) / math.sqrt(2))))
    z += 0.0002772368 * (43.0 * 0.5 * ((366.0 - Q.n_pairs_kt_above_1) / 43.0) * (1 + math.erf(((366.0 - Q.n_pairs_kt_above_1) / 43.0) / math.sqrt(2))))
    z += 0.5009388 * (0.1026619 * 0.5 * ((0.9189173 - Q.N3_b05) / 0.1026619) * (1 + math.erf(((0.9189173 - Q.N3_b05) / 0.1026619) / math.sqrt(2))))
    z += 0.1024023 * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.0) / 1.0) / math.sqrt(2))))
    z += -1.016068 * (0.01129976 * 0.5 * ((Q.N2 - 0.3357559) / 0.01129976) * (1 + math.erf(((Q.N2 - 0.3357559) / 0.01129976) / math.sqrt(2))))
    z += -0.03259115 * (1.0 * 0.5 * ((9.0 - Q.n_sd0_above_3) / 1.0) * (1 + math.erf(((9.0 - Q.n_sd0_above_3) / 1.0) / math.sqrt(2)))) * (0.05422308 * 0.5 * ((Q.z_neutral - 0.1384639) / 0.05422308) * (1 + math.erf(((Q.z_neutral - 0.1384639) / 0.05422308) / math.sqrt(2))))
    z += 0.07135506 * (0.02561531 * 0.5 * ((0.1061578 - Q.z_displaced3) / 0.02561531) * (1 + math.erf(((0.1061578 - Q.z_displaced3) / 0.02561531) / math.sqrt(2)))) * (0.9519596 * 0.5 * ((Q.mass_2photon - 3.013) / 0.9519596) * (1 + math.erf(((Q.mass_2photon - 3.013) / 0.9519596) / math.sqrt(2))))
    z += 91.51039 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_neutral_had - 1.0) / 1.0) * (1 + math.erf(((Q.n_neutral_had - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.002039349 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 1.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 1.0) / 1.0) / math.sqrt(2)))) * (0.3893833 * 0.5 * ((Q.lund3_lndelta - -2.817283) / 0.3893833) * (1 + math.erf(((Q.lund3_lndelta - -2.817283) / 0.3893833) / math.sqrt(2))))
    z += -0.01377162 * (2.624262 * 0.5 * ((4.636903 - Q.sip_3d_2) / 2.624262) * (1 + math.erf(((4.636903 - Q.sip_3d_2) / 2.624262) / math.sqrt(2))))
    z += 0.07690625 * (0.560057 * 0.5 * ((1.777787 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.777787 - Q.mass_displaced5) / 0.560057) / math.sqrt(2))))
    z += 0.0004043095 * (12.49365 * 0.5 * ((128.0079 - Q.sj4_pair_mass_max) / 12.49365) * (1 + math.erf(((128.0079 - Q.sj4_pair_mass_max) / 12.49365) / math.sqrt(2))))
    z += -1.681801e-05 * (8.537109 * 0.5 * ((17.60938 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((17.60938 - Q.max_abs_d0) / 8.537109) / math.sqrt(2)))) * (10.60212 * 0.5 * ((Q.sjf_3_2_mass_d3 - 14.85065) / 10.60212) * (1 + math.erf(((Q.sjf_3_2_mass_d3 - 14.85065) / 10.60212) / math.sqrt(2))))
    z += -0.02714707 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.1300849 * 0.5 * ((Q.dc_tag_2nd - 0.0) / 0.1300849) * (1 + math.erf(((Q.dc_tag_2nd - 0.0) / 0.1300849) / math.sqrt(2))))
    z += 0.00847355 * (0.1340495 * 0.5 * ((Q.dc_tag_2nd - 0.1300849) / 0.1340495) * (1 + math.erf(((Q.dc_tag_2nd - 0.1300849) / 0.1340495) / math.sqrt(2))))
    z += -6.700822 * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2)))) * (0.02222194 * 0.5 * ((Q.sj3_z2 - 0.1801324) / 0.02222194) * (1 + math.erf(((Q.sj3_z2 - 0.1801324) / 0.02222194) / math.sqrt(2))))
    z += -0.1618638 * (15.148 * 0.5 * ((Q.mres_sd_prong_mass1 - 84.19324) / 15.148) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 84.19324) / 15.148) / math.sqrt(2)))) * (0.0947145 * 0.5 * ((0.0947145 - Q.dc_4_z) / 0.0947145) * (1 + math.erf(((0.0947145 - Q.dc_4_z) / 0.0947145) / math.sqrt(2))))
    z += -7.973574 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.0947145 * 0.5 * ((Q.dc_4_z - 0.0) / 0.0947145) * (1 + math.erf(((Q.dc_4_z - 0.0) / 0.0947145) / math.sqrt(2))))
    z += -0.3661594 * (0.071994 * 0.5 * ((0.4545826 - Q.N3_b2) / 0.071994) * (1 + math.erf(((0.4545826 - Q.N3_b2) / 0.071994) / math.sqrt(2))))
    z += -0.05939315 * (0.7154015 * 0.5 * ((Q.mass_displaced3 - 2.409613) / 0.7154015) * (1 + math.erf(((Q.mass_displaced3 - 2.409613) / 0.7154015) / math.sqrt(2)))) * (0.0361397 * 0.5 * ((Q.z_neutral_had - 0.2675458) / 0.0361397) * (1 + math.erf(((Q.z_neutral_had - 0.2675458) / 0.0361397) / math.sqrt(2))))
    z += 1.839097 * (0.2398529 * 0.5 * ((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) * (1 + math.erf(((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) / math.sqrt(2))))
    z += 0.1540922 * (1.5 * 0.5 * ((7.0 - Q.n_lund) / 1.5) * (1 + math.erf(((7.0 - Q.n_lund) / 1.5) / math.sqrt(2))))
    z += 1.348175 * (0.2508028 * 0.5 * ((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2))))
    z += -3.869058 * (0.003868055 * 0.5 * ((Q.psi_0p3 - 0.9925964) / 0.003868055) * (1 + math.erf(((Q.psi_0p3 - 0.9925964) / 0.003868055) / math.sqrt(2))))
    z += -0.5149963 * (0.01417841 * 0.5 * ((0.1049877 - Q.C2_b2) / 0.01417841) * (1 + math.erf(((0.1049877 - Q.C2_b2) / 0.01417841) / math.sqrt(2))))
    z += -0.004938484 * (6.785484 * 0.5 * ((Q.mass_top5 - 68.52153) / 6.785484) * (1 + math.erf(((Q.mass_top5 - 68.52153) / 6.785484) / math.sqrt(2))))
    z += -0.01153679 * (6.785484 * 0.5 * ((Q.mass_top5 - 68.52153) / 6.785484) * (1 + math.erf(((Q.mass_top5 - 68.52153) / 6.785484) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.0005422804 * (3.143401 * 0.5 * ((Q.mass_top15 - 74.10377) / 3.143401) * (1 + math.erf(((Q.mass_top15 - 74.10377) / 3.143401) / math.sqrt(2))))
    z += 0.006892806 * (3.143401 * 0.5 * ((Q.mass_top15 - 74.10377) / 3.143401) * (1 + math.erf(((Q.mass_top15 - 74.10377) / 3.143401) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2))))
    z += -1.760507 * (0.01063134 * 0.5 * ((0.4075226 - Q.N2_b05) / 0.01063134) * (1 + math.erf(((0.4075226 - Q.N2_b05) / 0.01063134) / math.sqrt(2))))
    z += -0.09302846 * (1.0 * 0.5 * ((0.0 - Q.sdb_4_n) / 1.0) * (1 + math.erf(((0.0 - Q.sdb_4_n) / 1.0) / math.sqrt(2))))
    z += -0.1865514 * (0.01282622 * 0.5 * ((0.1010123 - Q.z_neutral_had) / 0.01282622) * (1 + math.erf(((0.1010123 - Q.z_neutral_had) / 0.01282622) / math.sqrt(2))))
    z += -0.007117864 * (3.02693 * 0.5 * ((78.76797 - Q.sj4_pair_mass_max) / 3.02693) * (1 + math.erf(((78.76797 - Q.sj4_pair_mass_max) / 3.02693) / math.sqrt(2))))
    z += -0.2925391 * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.0) / 1.0) / math.sqrt(2)))) * (0.00991452 * 0.5 * ((-0.01644749 - Q.tdz_0) / 0.00991452) * (1 + math.erf(((-0.01644749 - Q.tdz_0) / 0.00991452) / math.sqrt(2))))
    z += -0.005804759 * (0.7154015 * 0.5 * ((Q.mass_displaced3 - 2.409613) / 0.7154015) * (1 + math.erf(((Q.mass_displaced3 - 2.409613) / 0.7154015) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dzerr_75 - 0.0) / 1e-06) * (1 + math.erf(((Q.dzerr_75 - 0.0) / 1e-06) / math.sqrt(2))))
    z += -0.0368299 * (0.01168879 * 0.5 * ((Q.tdz_2 - -0.02221619) / 0.01168879) * (1 + math.erf(((Q.tdz_2 - -0.02221619) / 0.01168879) / math.sqrt(2))))
    z += 1.858534 * (0.0947145 * 0.5 * ((Q.dc_4_z - 0.0) / 0.0947145) * (1 + math.erf(((Q.dc_4_z - 0.0) / 0.0947145) / math.sqrt(2))))
    z += 0.04711242 * (2.0 * 0.5 * ((17.0 - Q.n_dr_0p2_0p4) / 2.0) * (1 + math.erf(((17.0 - Q.n_dr_0p2_0p4) / 2.0) / math.sqrt(2))))
    z += -3.505226 * (0.0257051 * 0.5 * ((0.1468781 - Q.z_dr_0p2_0p4) / 0.0257051) * (1 + math.erf(((0.1468781 - Q.z_dr_0p2_0p4) / 0.0257051) / math.sqrt(2))))
    z += -2.292818 * (0.01435892 * 0.5 * ((Q.pz_lnd3 - 0.1187422) / 0.01435892) * (1 + math.erf(((Q.pz_lnd3 - 0.1187422) / 0.01435892) / math.sqrt(2)))) * (0.04200199 * 0.5 * ((Q.sj3_dr23 - 0.1712041) / 0.04200199) * (1 + math.erf(((Q.sj3_dr23 - 0.1712041) / 0.04200199) / math.sqrt(2))))
    z += 1.086535 * (0.2508028 * 0.5 * ((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2)))) * (0.04756694 * 0.5 * ((-0.04769897 - Q.eta_37) / 0.04756694) * (1 + math.erf(((-0.04769897 - Q.eta_37) / 0.04756694) / math.sqrt(2))))
    z += 0.003664168 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) * (0.02849867 * 0.5 * ((Q.z_dr_0p1_0p2 - 0.07090906) / 0.02849867) * (1 + math.erf(((Q.z_dr_0p1_0p2 - 0.07090906) / 0.02849867) / math.sqrt(2))))
    z += 802.9544 * (0.004691441 * 0.5 * ((0.0 - Q.z_muon) / 0.004691441) * (1 + math.erf(((0.0 - Q.z_muon) / 0.004691441) / math.sqrt(2))))
    return z


def neuron_25(Q):
    z = 0.09125754
    z += 0.004904827 * Q.n_dr_0p2_0p4
    z += 0.04544564 * Q.pair_max_lnkt
    z += -0.6588331 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2))))
    z += -1.366456 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2))))
    z += 0.09244935 * (0.04122765 * 0.5 * ((Q.N2_b05 - 0.302405) / 0.04122765) * (1 + math.erf(((Q.N2_b05 - 0.302405) / 0.04122765) / math.sqrt(2))))
    z += 0.01669367 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2))))
    z += -1.696963 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) * (1.0 * 0.5 * ((5.0 - Q.sjf_4_1_n_d3) / 1.0) * (1 + math.erf(((5.0 - Q.sjf_4_1_n_d3) / 1.0) / math.sqrt(2))))
    z += 0.01635245 * (2.113111 * 0.5 * ((3.373037 - Q.sip_3d_1) / 2.113111) * (1 + math.erf(((3.373037 - Q.sip_3d_1) / 2.113111) / math.sqrt(2))))
    z += 0.007870682 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (3.244489 * 0.5 * ((Q.sj4_pair_mass_max - 66.70506) / 3.244489) * (1 + math.erf(((Q.sj4_pair_mass_max - 66.70506) / 3.244489) / math.sqrt(2))))
    z += 0.003311147 * (3.003641 * 0.5 * ((40.24169 - Q.mass_neutral) / 3.003641) * (1 + math.erf(((40.24169 - Q.mass_neutral) / 3.003641) / math.sqrt(2))))
    z += -2.102023 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.03966789 * 0.5 * ((0.1333792 - Q.ak02_3_z) / 0.03966789) * (1 + math.erf(((0.1333792 - Q.ak02_3_z) / 0.03966789) / math.sqrt(2))))
    z += 2.520551 * (0.002590499 * 0.5 * ((0.0255865 - Q.sum_z_dr2_top10) / 0.002590499) * (1 + math.erf(((0.0255865 - Q.sum_z_dr2_top10) / 0.002590499) / math.sqrt(2))))
    z += 0.0005967423 * (2.0 * 0.5 * ((26.0 - Q.n_neutral) / 2.0) * (1 + math.erf(((26.0 - Q.n_neutral) / 2.0) / math.sqrt(2))))
    z += 0.01158169 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2))))
    z += 3.698483 * (0.002590499 * 0.5 * ((0.0255865 - Q.sum_z_dr2_top10) / 0.002590499) * (1 + math.erf(((0.0255865 - Q.sum_z_dr2_top10) / 0.002590499) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.3400995 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += 0.1532808 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) * (3.347501 * 0.5 * ((Q.mass_displaced3 - 9.090532) / 3.347501) * (1 + math.erf(((Q.mass_displaced3 - 9.090532) / 3.347501) / math.sqrt(2))))
    z += 0.0006672446 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2))))
    z += -771.7598 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (2.54381e-06 * 0.5 * ((3.53193e-06 - Q.e4) / 2.54381e-06) * (1 + math.erf(((3.53193e-06 - Q.e4) / 2.54381e-06) / math.sqrt(2))))
    z += 0.01332067 * (1.0 * 0.5 * ((Q.dc_1_n_disp3 - 1.0) / 1.0) * (1 + math.erf(((Q.dc_1_n_disp3 - 1.0) / 1.0) / math.sqrt(2))))
    z += -0.07013395 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.sdb_4_n) / 1.0) * (1 + math.erf(((1.0 - Q.sdb_4_n) / 1.0) / math.sqrt(2))))
    z += 0.0009892762 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2))))
    z += 4.41284e-08 * (3.421198 * 0.5 * ((69.01131 - Q.nca_sj4_pair_mass_2nd) / 3.421198) * (1 + math.erf(((69.01131 - Q.nca_sj4_pair_mass_2nd) / 3.421198) / math.sqrt(2))))
    z += -0.07944673 * (0.02184861 * 0.5 * ((0.6020351 - Q.nca_sj4_pair2nd_over_mass) / 0.02184861) * (1 + math.erf(((0.6020351 - Q.nca_sj4_pair2nd_over_mass) / 0.02184861) / math.sqrt(2))))
    z += 0.01124529 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2))))
    z += -0.008076334 * (3.556438 * 0.5 * ((76.91486 - Q.sj3_pair_mass_max) / 3.556438) * (1 + math.erf(((76.91486 - Q.sj3_pair_mass_max) / 3.556438) / math.sqrt(2))))
    z += -0.001574783 * (4.146938 * 0.5 * ((110.2019 - Q.mass) / 4.146938) * (1 + math.erf(((110.2019 - Q.mass) / 4.146938) / math.sqrt(2))))
    z += -0.02570458 * (6.531597 * 0.5 * ((79.47361 - Q.mass) / 6.531597) * (1 + math.erf(((79.47361 - Q.mass) / 6.531597) / math.sqrt(2))))
    z += -0.0007842782 * (3.799482 * 0.5 * ((127.2317 - Q.mass) / 3.799482) * (1 + math.erf(((127.2317 - Q.mass) / 3.799482) / math.sqrt(2))))
    z += -0.005270425 * (3.799482 * 0.5 * ((127.2317 - Q.mass) / 3.799482) * (1 + math.erf(((127.2317 - Q.mass) / 3.799482) / math.sqrt(2)))) * (0.005274752 * 0.5 * ((-0.01099959 - Q.sjq_2_prod_k1) / 0.005274752) * (1 + math.erf(((-0.01099959 - Q.sjq_2_prod_k1) / 0.005274752) / math.sqrt(2))))
    z += 14.04247 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) * (0.06215671 * 0.5 * ((0.3112717 - Q.sj4_dr_min) / 0.06215671) * (1 + math.erf(((0.3112717 - Q.sj4_dr_min) / 0.06215671) / math.sqrt(2))))
    z += 0.7152751 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.2765194 * 0.5 * ((2.605573 - Q.jd_3d_4) / 0.2765194) * (1 + math.erf(((2.605573 - Q.jd_3d_4) / 0.2765194) / math.sqrt(2))))
    z += -0.00027928 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2)))) * (0.1767711 * 0.5 * ((Q.lne_1 - 3.745821) / 0.1767711) * (1 + math.erf(((Q.lne_1 - 3.745821) / 0.1767711) / math.sqrt(2))))
    z += -1.376841 * (0.02184861 * 0.5 * ((0.6020351 - Q.nca_sj4_pair2nd_over_mass) / 0.02184861) * (1 + math.erf(((0.6020351 - Q.nca_sj4_pair2nd_over_mass) / 0.02184861) / math.sqrt(2)))) * (0.01280829 * 0.5 * ((Q.z_neutral_had - 0.05062943) / 0.01280829) * (1 + math.erf(((Q.z_neutral_had - 0.05062943) / 0.01280829) / math.sqrt(2))))
    z += 2.393756 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.01374893 * 0.5 * ((0.2241366 - Q.pz_lnd2) / 0.01374893) * (1 + math.erf(((0.2241366 - Q.pz_lnd2) / 0.01374893) / math.sqrt(2))))
    z += 0.05241824 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2))))
    z += 0.002023759 * (3.799482 * 0.5 * ((127.2317 - Q.mass) / 3.799482) * (1 + math.erf(((127.2317 - Q.mass) / 3.799482) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.03724001 * (3.799482 * 0.5 * ((127.2317 - Q.mass) / 3.799482) * (1 + math.erf(((127.2317 - Q.mass) / 3.799482) / math.sqrt(2)))) * (0.01108125 * 0.5 * ((0.1325326 - Q.pz_lnd0) / 0.01108125) * (1 + math.erf(((0.1325326 - Q.pz_lnd0) / 0.01108125) / math.sqrt(2))))
    z += -0.101814 * (0.04018594 * 0.5 * ((0.5498426 - Q.tau32) / 0.04018594) * (1 + math.erf(((0.5498426 - Q.tau32) / 0.04018594) / math.sqrt(2))))
    z += -0.3016067 * (0.04018594 * 0.5 * ((0.5498426 - Q.tau32) / 0.04018594) * (1 + math.erf(((0.5498426 - Q.tau32) / 0.04018594) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_pairs_kt_above_30 - 1.0) / 1.0) * (1 + math.erf(((Q.n_pairs_kt_above_30 - 1.0) / 1.0) / math.sqrt(2))))
    z += -6.840874e-05 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2)))) * (3.006676 * 0.5 * ((6.43167 - Q.jd_3d_4) / 3.006676) * (1 + math.erf(((6.43167 - Q.jd_3d_4) / 3.006676) / math.sqrt(2))))
    z += -0.001250628 * (3.003641 * 0.5 * ((40.24169 - Q.mass_neutral) / 3.003641) * (1 + math.erf(((40.24169 - Q.mass_neutral) / 3.003641) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 0.0) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 0.0) / 1.0) / math.sqrt(2))))
    z += -38.36121 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) * (0.01117424 * 0.5 * ((0.03578969 - Q.pz_lnkt4) / 0.01117424) * (1 + math.erf(((0.03578969 - Q.pz_lnkt4) / 0.01117424) / math.sqrt(2))))
    z += -0.04249693 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (0.6891756 * 0.5 * ((6.112692 - Q.sj3_mass3) / 0.6891756) * (1 + math.erf(((6.112692 - Q.sj3_mass3) / 0.6891756) / math.sqrt(2))))
    z += -0.01288715 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sdb_2_n - 3.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 3.0) / 1.0) / math.sqrt(2))))
    z += 1.638245e-06 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2)))) * (702.9612 * 0.5 * ((379.5631 - Q.sv_2_sd0_sum) / 702.9612) * (1 + math.erf(((379.5631 - Q.sv_2_sd0_sum) / 702.9612) / math.sqrt(2))))
    z += -0.134559 * (0.04018594 * 0.5 * ((0.5498426 - Q.tau32) / 0.04018594) * (1 + math.erf(((0.5498426 - Q.tau32) / 0.04018594) / math.sqrt(2)))) * (1.0 * 0.5 * ((5.0 - Q.n_pairs_kt_above_10) / 1.0) * (1 + math.erf(((5.0 - Q.n_pairs_kt_above_10) / 1.0) / math.sqrt(2))))
    z += -162.9634 * (0.04018594 * 0.5 * ((0.5498426 - Q.tau32) / 0.04018594) * (1 + math.erf(((0.5498426 - Q.tau32) / 0.04018594) / math.sqrt(2)))) * (0.0005676896 * 0.5 * ((Q.e3 - 0.00228569) / 0.0005676896) * (1 + math.erf(((Q.e3 - 0.00228569) / 0.0005676896) / math.sqrt(2))))
    z += 0.0007832278 * (13.55261 * 0.5 * ((35.45098 - Q.sv_1_sd0_sum) / 13.55261) * (1 + math.erf(((35.45098 - Q.sv_1_sd0_sum) / 13.55261) / math.sqrt(2))))
    z += 0.873053 * (0.01141967 * 0.5 * ((0.03624058 - Q.z_displaced3) / 0.01141967) * (1 + math.erf(((0.03624058 - Q.z_displaced3) / 0.01141967) / math.sqrt(2))))
    z += 0.05925085 * (0.07010728 * 0.5 * ((4.046329 - Q.lund_max_lnkt) / 0.07010728) * (1 + math.erf(((4.046329 - Q.lund_max_lnkt) / 0.07010728) / math.sqrt(2))))
    z += -1.970952 * (0.0125626 * 0.5 * ((Q.C2_b05 - 0.3223395) / 0.0125626) * (1 + math.erf(((Q.C2_b05 - 0.3223395) / 0.0125626) / math.sqrt(2))))
    z += -1.185246 * (0.05999923 * 0.5 * ((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) * (1 + math.erf(((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) / math.sqrt(2))))
    z += -1.218326 * (0.003546451 * 0.5 * ((0.0417856 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0417856 - Q.M2_b2) / 0.003546451) / math.sqrt(2)))) * (1.0 * 0.5 * ((2.0 - Q.dc_2_n_disp3) / 1.0) * (1 + math.erf(((2.0 - Q.dc_2_n_disp3) / 1.0) / math.sqrt(2))))
    z += -0.004430227 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2))))
    z += 0.01680273 * (3.5 * 0.5 * ((Q.n_dr_0p4_up - 15.0) / 3.5) * (1 + math.erf(((Q.n_dr_0p4_up - 15.0) / 3.5) / math.sqrt(2))))
    z += -0.0007024491 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.02288284 * 0.5 * ((0.212136 - Q.z_neutral_had) / 0.02288284) * (1 + math.erf(((0.212136 - Q.z_neutral_had) / 0.02288284) / math.sqrt(2))))
    z += 0.3416029 * (0.01476889 * 0.5 * ((0.1811493 - Q.pz_lnd2) / 0.01476889) * (1 + math.erf(((0.1811493 - Q.pz_lnd2) / 0.01476889) / math.sqrt(2))))
    z += -0.001751286 * (2.295472 * 0.5 * ((21.21062 - Q.sj2_mass1) / 2.295472) * (1 + math.erf(((21.21062 - Q.sj2_mass1) / 2.295472) / math.sqrt(2))))
    z += -4.435275 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.1380651 * (0.04122765 * 0.5 * ((Q.N2_b05 - 0.302405) / 0.04122765) * (1 + math.erf(((Q.N2_b05 - 0.302405) / 0.04122765) / math.sqrt(2)))) * (0.009933948 * 0.5 * ((Q.kt2_2_z_disp3 - 0.02323978) / 0.009933948) * (1 + math.erf(((Q.kt2_2_z_disp3 - 0.02323978) / 0.009933948) / math.sqrt(2))))
    z += -0.00104914 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.01535144 * 0.5 * ((0.1124664 - Q.sj4_zsoft) / 0.01535144) * (1 + math.erf(((0.1124664 - Q.sj4_zsoft) / 0.01535144) / math.sqrt(2))))
    z += -0.008044077 * (1.0 * 0.5 * ((10.0 - Q.n_photon) / 1.0) * (1 + math.erf(((10.0 - Q.n_photon) / 1.0) / math.sqrt(2))))
    z += -0.0947165 * (0.1359916 * 0.5 * ((Q.pair_max_lnm2 - 7.203613) / 0.1359916) * (1 + math.erf(((Q.pair_max_lnm2 - 7.203613) / 0.1359916) / math.sqrt(2))))
    z += -0.002017094 * (3.421198 * 0.5 * ((69.01131 - Q.nca_sj4_pair_mass_2nd) / 3.421198) * (1 + math.erf(((69.01131 - Q.nca_sj4_pair_mass_2nd) / 3.421198) / math.sqrt(2)))) * (0.1712927 * 0.5 * ((Q.dc_3_charge - 0.1143891) / 0.1712927) * (1 + math.erf(((Q.dc_3_charge - 0.1143891) / 0.1712927) / math.sqrt(2))))
    z += -0.01232353 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.523415 * 0.5 * ((2.133094 - Q.sjf_4_4_mass_d3) / 1.523415) * (1 + math.erf(((2.133094 - Q.sjf_4_4_mass_d3) / 1.523415) / math.sqrt(2))))
    z += 0.05007472 * (6.104222 * 0.5 * ((0.0 - Q.ak02_1_mass) / 6.104222) * (1 + math.erf(((0.0 - Q.ak02_1_mass) / 6.104222) / math.sqrt(2))))
    z += 0.7969673 * (0.007664379 * 0.5 * ((0.05414784 - Q.sum_z_dr2_top20) / 0.007664379) * (1 + math.erf(((0.05414784 - Q.sum_z_dr2_top20) / 0.007664379) / math.sqrt(2))))
    z += -52279.44 * (1.079379e-06 * 0.5 * ((2.097213e-06 - Q.e4) / 1.079379e-06) * (1 + math.erf(((2.097213e-06 - Q.e4) / 1.079379e-06) / math.sqrt(2))))
    z += 0.004727364 * (5.197 * 0.5 * ((95.14961 - Q.mass) / 5.197) * (1 + math.erf(((95.14961 - Q.mass) / 5.197) / math.sqrt(2))))
    z += -0.6770548 * (0.02314387 * 0.5 * ((0.2509165 - Q.sdb_2_z) / 0.02314387) * (1 + math.erf(((0.2509165 - Q.sdb_2_z) / 0.02314387) / math.sqrt(2))))
    z += -0.194887 * (0.09718233 * 0.5 * ((Q.z_displaced3 - 0.4232894) / 0.09718233) * (1 + math.erf(((Q.z_displaced3 - 0.4232894) / 0.09718233) / math.sqrt(2))))
    z += -0.4832636 * (0.09626377 * 0.5 * ((Q.sjf_2_1_z_d3 - 0.3672808) / 0.09626377) * (1 + math.erf(((Q.sjf_2_1_z_d3 - 0.3672808) / 0.09626377) / math.sqrt(2))))
    z += -0.08609085 * (0.01476889 * 0.5 * ((0.1811493 - Q.pz_lnd2) / 0.01476889) * (1 + math.erf(((0.1811493 - Q.pz_lnd2) / 0.01476889) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_4_3_n_d5 - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_4_3_n_d5 - 1.0) / 1.0) / math.sqrt(2))))
    z += -0.7580717 * (0.03445022 * 0.5 * ((0.04982189 - Q.lepsj_3_dr) / 0.03445022) * (1 + math.erf(((0.04982189 - Q.lepsj_3_dr) / 0.03445022) / math.sqrt(2))))
    z += 40.6788 * (0.05999923 * 0.5 * ((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) * (1 + math.erf(((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) / math.sqrt(2)))) * (0.01992156 * 0.5 * ((0.04178742 - Q.lepsj_2_dr) / 0.01992156) * (1 + math.erf(((0.04178742 - Q.lepsj_2_dr) / 0.01992156) / math.sqrt(2))))
    z += 0.0005405395 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.0795894 * (1.0 * 0.5 * ((1.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((1.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2))))
    z += 0.06500965 * (0.0125626 * 0.5 * ((Q.C2_b05 - 0.3223395) / 0.0125626) * (1 + math.erf(((Q.C2_b05 - 0.3223395) / 0.0125626) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dc_mass_disp_2nd - 0.0) / 1e-06) * (1 + math.erf(((Q.dc_mass_disp_2nd - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.003391164 * (12.96659 * 0.5 * ((63.78353 - Q.mres_sd_mass_b0z02) / 12.96659) * (1 + math.erf(((63.78353 - Q.mres_sd_mass_b0z02) / 12.96659) / math.sqrt(2))))
    z += -0.01632417 * (7.657485 * 0.5 * ((93.75785 - Q.mres_sd_mass_b0z02) / 7.657485) * (1 + math.erf(((93.75785 - Q.mres_sd_mass_b0z02) / 7.657485) / math.sqrt(2))))
    z += -0.004221328 * (16.47315 * 0.5 * ((47.97531 - Q.mres_sd_mass_b0z02) / 16.47315) * (1 + math.erf(((47.97531 - Q.mres_sd_mass_b0z02) / 16.47315) / math.sqrt(2))))
    z += 0.00594051 * (7.801503 * 0.5 * ((101.9185 - Q.mres_sd_mass_b0z02) / 7.801503) * (1 + math.erf(((101.9185 - Q.mres_sd_mass_b0z02) / 7.801503) / math.sqrt(2))))
    z += 0.01049724 * (6.347532 * 0.5 * ((80.36442 - Q.mres_sd_mass_b0z02) / 6.347532) * (1 + math.erf(((80.36442 - Q.mres_sd_mass_b0z02) / 6.347532) / math.sqrt(2))))
    z += 0.0001896128 * (6.955739 * 0.5 * ((125.6411 - Q.mres_sd_mass_b0z02) / 6.955739) * (1 + math.erf(((125.6411 - Q.mres_sd_mass_b0z02) / 6.955739) / math.sqrt(2))))
    z += 0.0008885157 * (29.74772 * 0.5 * ((163.95 - Q.mres_sd_mass_b0z02) / 29.74772) * (1 + math.erf(((163.95 - Q.mres_sd_mass_b0z02) / 29.74772) / math.sqrt(2))))
    z += 0.0002768396 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2)))) * (0.02812676 * 0.5 * ((Q.sj3_dr13 - 0.3524911) / 0.02812676) * (1 + math.erf(((Q.sj3_dr13 - 0.3524911) / 0.02812676) / math.sqrt(2))))
    z += -0.6563926 * (0.001664982 * 0.5 * ((0.007018285 - Q.sum_z_dr2_top3) / 0.001664982) * (1 + math.erf(((0.007018285 - Q.sum_z_dr2_top3) / 0.001664982) / math.sqrt(2))))
    z += -0.0001572247 * (9.707626 * 0.5 * ((16.47324 - Q.dc_1_sd0_1) / 9.707626) * (1 + math.erf(((16.47324 - Q.dc_1_sd0_1) / 9.707626) / math.sqrt(2))))
    z += 0.02562617 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.ischhad_10) / 1.0) * (1 + math.erf(((0.0 - Q.ischhad_10) / 1.0) / math.sqrt(2))))
    z += -3.58229e-06 * (3.003641 * 0.5 * ((40.24169 - Q.mass_neutral) / 3.003641) * (1 + math.erf(((40.24169 - Q.mass_neutral) / 3.003641) / math.sqrt(2)))) * (467.791 * 0.5 * ((593.8527 - Q.dc_1_sd0_1) / 467.791) * (1 + math.erf(((593.8527 - Q.dc_1_sd0_1) / 467.791) / math.sqrt(2))))
    z += 0.0006425938 * (3.521114 * 0.5 * ((Q.mass_top20 - 93.50967) / 3.521114) * (1 + math.erf(((Q.mass_top20 - 93.50967) / 3.521114) / math.sqrt(2))))
    z += -0.07092338 * (1.0 * 0.5 * ((2.0 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((2.0 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2))))
    z += -0.09410597 * (0.3121171 * 0.5 * ((2.17433 - Q.sip_3d_1) / 0.3121171) * (1 + math.erf(((2.17433 - Q.sip_3d_1) / 0.3121171) / math.sqrt(2))))
    z += 5.777016e-05 * (4.196564 * 0.5 * ((Q.sj4_pair_mass_max - 92.5542) / 4.196564) * (1 + math.erf(((Q.sj4_pair_mass_max - 92.5542) / 4.196564) / math.sqrt(2))))
    z += -0.0001100527 * (1.0 * 0.5 * ((Q.n_dr_0p4_up - 2.0) / 1.0) * (1 + math.erf(((Q.n_dr_0p4_up - 2.0) / 1.0) / math.sqrt(2))))
    z += -0.004911646 * (1.0 * 0.5 * ((Q.n_dr_0p4_up - 2.0) / 1.0) * (1 + math.erf(((Q.n_dr_0p4_up - 2.0) / 1.0) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.charge_76 - 0.0) / 1e-06) * (1 + math.erf(((Q.charge_76 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.005018624 * (6.93786 * 0.5 * ((120.6471 - Q.sj3_pair_mass_max) / 6.93786) * (1 + math.erf(((120.6471 - Q.sj3_pair_mass_max) / 6.93786) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.isphoton_29 - 1.0) / 1.0) * (1 + math.erf(((Q.isphoton_29 - 1.0) / 1.0) / math.sqrt(2))))
    z += -0.00546905 * (9.707626 * 0.5 * ((16.47324 - Q.dc_1_sd0_1) / 9.707626) * (1 + math.erf(((16.47324 - Q.dc_1_sd0_1) / 9.707626) / math.sqrt(2)))) * (0.03500366 * 0.5 * ((Q.dzerr_22 - 0.0) / 0.03500366) * (1 + math.erf(((Q.dzerr_22 - 0.0) / 0.03500366) / math.sqrt(2))))
    z += -0.4198866 * (0.04122765 * 0.5 * ((Q.N2_b05 - 0.302405) / 0.04122765) * (1 + math.erf(((Q.N2_b05 - 0.302405) / 0.04122765) / math.sqrt(2)))) * (1.0 * 0.5 * ((2.0 - Q.n_electron) / 1.0) * (1 + math.erf(((2.0 - Q.n_electron) / 1.0) / math.sqrt(2))))
    return z


def neuron_26(Q):
    z = -3.231261e-06
    return z


def neuron_27(Q):
    z = -6.308491e-06
    return z


def neuron_28(Q):
    z = 9.617203e-06
    return z


def neuron_29(Q):
    z = 9.183854e-06
    return z


def neuron_30(Q):
    z = -8.603046e-06
    return z


def neuron_31(Q):
    z = 3.509519e-06
    return z


def neuron_32(Q):
    z = 1.441347e-05
    return z


def neuron_33(Q):
    z = 2.567321e-06
    return z


def neuron_34(Q):
    z = 1.388485e-05
    return z


def neuron_35(Q):
    z = -6.743291e-06
    return z


def neuron_36(Q):
    z = -4.831943e-06
    return z


def neuron_37(Q):
    z = -4.357431e-06
    return z


def neuron_38(Q):
    z = -9.411165e-06
    return z


def neuron_39(Q):
    z = 8.27427e-06
    return z


def neuron_40(Q):
    z = 7.775515e-07
    return z


def neuron_41(Q):
    z = 3.814669e-08
    return z


def neuron_42(Q):
    z = 8.779036e-06
    return z


def neuron_43(Q):
    z = -9.407703e-06
    return z


def neuron_44(Q):
    z = -3.324341e-06
    return z


def neuron_45(Q):
    z = 6.849329e-06
    return z


def neuron_46(Q):
    z = 1.368757e-07
    return z


def neuron_47(Q):
    z = 2.492068e-07
    return z


def neuron_48(Q):
    z = -8.791772e-06
    return z


def neuron_49(Q):
    z = -2.243813e-06
    return z


def neuron_50(Q):
    z = 5.175112e-06
    return z


def neuron_51(Q):
    z = -2.012201e-06
    return z


def neuron_52(Q):
    z = -1.39592
    z += 482.6759 * Q.ecf_g41
    z += -1.63706 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2))))
    z += 6.65337 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2))))
    z += 6.253905 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (0.02727943 * 0.5 * ((Q.N2 - 0.1824346) / 0.02727943) * (1 + math.erf(((Q.N2 - 0.1824346) / 0.02727943) / math.sqrt(2))))
    z += 0.2093306 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2))))
    z += -0.02324223 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2))))
    z += 0.04789165 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 11.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 11.0) / 1.0) / math.sqrt(2))))
    z += -0.002054769 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (4.565952 * 0.5 * ((Q.sj3_pair_mass_min - 48.40547) / 4.565952) * (1 + math.erf(((Q.sj3_pair_mass_min - 48.40547) / 4.565952) / math.sqrt(2))))
    z += -3.452701 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) * (0.0330605 * 0.5 * ((0.114659 - Q.lep_dr) / 0.0330605) * (1 + math.erf(((0.114659 - Q.lep_dr) / 0.0330605) / math.sqrt(2))))
    z += 0.2945703 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2))))
    z += 0.05846247 * (11.49171 * 0.5 * ((71.96396 - Q.mass) / 11.49171) * (1 + math.erf(((71.96396 - Q.mass) / 11.49171) / math.sqrt(2))))
    z += 0.005234826 * (3.304036 * 0.5 * ((115.614 - Q.mass_top50) / 3.304036) * (1 + math.erf(((115.614 - Q.mass_top50) / 3.304036) / math.sqrt(2))))
    z += 0.03530322 * (18.66842 * 0.5 * ((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2))))
    z += -0.02804634 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) / math.sqrt(2))))
    z += 0.01729339 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2))))
    z += -0.1077684 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2))))
    z += -0.008154785 * (14.42334 * 0.5 * ((141.5589 - Q.mres_sd_mass_b0z005) / 14.42334) * (1 + math.erf(((141.5589 - Q.mres_sd_mass_b0z005) / 14.42334) / math.sqrt(2))))
    z += -0.0289864 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2))))
    z += 0.01907582 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2))))
    z += -0.01179388 * (18.66842 * 0.5 * ((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) * (0.04741882 * 0.5 * ((0.4690304 - Q.tau21_b2) / 0.04741882) * (1 + math.erf(((0.4690304 - Q.tau21_b2) / 0.04741882) / math.sqrt(2))))
    z += -0.02016684 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.03693697 * 0.5 * ((0.5838518 - Q.tau32_b2) / 0.03693697) * (1 + math.erf(((0.5838518 - Q.tau32_b2) / 0.03693697) / math.sqrt(2))))
    z += 0.1168103 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) * (0.05566634 * 0.5 * ((0.3093201 - Q.z_neutral_had) / 0.05566634) * (1 + math.erf(((0.3093201 - Q.z_neutral_had) / 0.05566634) / math.sqrt(2))))
    z += 0.1247314 * (3.069852 * 0.5 * ((4.516968 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.516968 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2))))
    z += 6.679153e-06 * (18.66842 * 0.5 * ((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) * (36.85684 * 0.5 * ((77.03428 - Q.sv_1_sd0_sum) / 36.85684) * (1 + math.erf(((77.03428 - Q.sv_1_sd0_sum) / 36.85684) / math.sqrt(2))))
    z += -0.0007731166 * (18.66842 * 0.5 * ((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.3636901 * (0.005494283 * 0.5 * ((Q.pz_lnd1 - 0.05758556) / 0.005494283) * (1 + math.erf(((Q.pz_lnd1 - 0.05758556) / 0.005494283) / math.sqrt(2))))
    z += 0.01655 * (11.90664 * 0.5 * ((Q.sj3_pair_mass_min - 80.02563) / 11.90664) * (1 + math.erf(((Q.sj3_pair_mass_min - 80.02563) / 11.90664) / math.sqrt(2))))
    z += -0.3920919 * (0.09740396 * 0.5 * ((1.325034 - Q.D2) / 0.09740396) * (1 + math.erf(((1.325034 - Q.D2) / 0.09740396) / math.sqrt(2))))
    z += 0.008613616 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += -0.01144652 * (4.488781 * 0.5 * ((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) * (1 + math.erf(((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) / math.sqrt(2))))
    z += 0.2602632 * (1.0 * 0.5 * ((1.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.07764775 * (1.0 * 0.5 * ((1.0 - Q.ak02_1_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_1_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.06956806 * (4.481852 * 0.5 * ((6.220692 - Q.sjf_2_1_max3d) / 4.481852) * (1 + math.erf(((6.220692 - Q.sjf_2_1_max3d) / 4.481852) / math.sqrt(2))))
    z += -0.007641501 * (17.1726 * 0.5 * ((34.81061 - Q.sjf_2_1_max3d) / 17.1726) * (1 + math.erf(((34.81061 - Q.sjf_2_1_max3d) / 17.1726) / math.sqrt(2))))
    z += -0.008877812 * (2.497115 * 0.5 * ((Q.mass_displaced5 - 6.387683) / 2.497115) * (1 + math.erf(((Q.mass_displaced5 - 6.387683) / 2.497115) / math.sqrt(2))))
    z += -0.0009569527 * (3.100743 * 0.5 * ((65.0404 - Q.mass_charged) / 3.100743) * (1 + math.erf(((65.0404 - Q.mass_charged) / 3.100743) / math.sqrt(2))))
    z += -0.2236236 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) * (0.02830505 * 0.5 * ((Q.dzerr_0 - 0.0) / 0.02830505) * (1 + math.erf(((Q.dzerr_0 - 0.0) / 0.02830505) / math.sqrt(2))))
    z += -0.01949779 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) * (0.3035439 * 0.5 * ((Q.sjq_2_prod_k03 - -0.9998116) / 0.3035439) * (1 + math.erf(((Q.sjq_2_prod_k03 - -0.9998116) / 0.3035439) / math.sqrt(2))))
    z += 0.06460797 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.03966789 * 0.5 * ((0.1333792 - Q.ak02_3_z) / 0.03966789) * (1 + math.erf(((0.1333792 - Q.ak02_3_z) / 0.03966789) / math.sqrt(2))))
    z += -0.014238 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (4.0 * 0.5 * ((24.0 - Q.n_dr_0p2_0p4) / 4.0) * (1 + math.erf(((24.0 - Q.n_dr_0p2_0p4) / 4.0) / math.sqrt(2))))
    z += 0.001689414 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.9414577 * 0.5 * ((3.157646 - Q.sip_3d_3) / 0.9414577) * (1 + math.erf(((3.157646 - Q.sip_3d_3) / 0.9414577) / math.sqrt(2))))
    z += 0.1375497 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2)))) * (0.09869392 * 0.5 * ((Q.sjq_3_sumabs_k1 - 0.5890081) / 0.09869392) * (1 + math.erf(((Q.sjq_3_sumabs_k1 - 0.5890081) / 0.09869392) / math.sqrt(2))))
    z += -4.992518 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2))))
    z += 1.577293 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.0792551 * 0.5 * ((Q.lepsj_2_dr - 0.1822601) / 0.0792551) * (1 + math.erf(((Q.lepsj_2_dr - 0.1822601) / 0.0792551) / math.sqrt(2))))
    z += -0.04922923 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) / math.sqrt(2))))
    z += 0.2020862 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2)))) * (0.05296894 * 0.5 * ((0.1145956 - Q.sjq_3_prod_k1) / 0.05296894) * (1 + math.erf(((0.1145956 - Q.sjq_3_prod_k1) / 0.05296894) / math.sqrt(2))))
    z += 0.004646061 * (4.488781 * 0.5 * ((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) * (1 + math.erf(((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) / math.sqrt(2)))) * (0.1152886 * 0.5 * ((-0.3970734 - Q.dc_2_charge) / 0.1152886) * (1 + math.erf(((-0.3970734 - Q.dc_2_charge) / 0.1152886) / math.sqrt(2))))
    z += -0.02818726 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (110.8422 * 0.5 * ((114.1102 - Q.sjf_2_2_max3d) / 110.8422) * (1 + math.erf(((114.1102 - Q.sjf_2_2_max3d) / 110.8422) / math.sqrt(2))))
    z += 3.6241 * (0.007529243 * 0.5 * ((0.01782783 - Q.z_displaced3) / 0.007529243) * (1 + math.erf(((0.01782783 - Q.z_displaced3) / 0.007529243) / math.sqrt(2))))
    z += -0.2846808 * (0.05582565 * 0.5 * ((0.456416 - Q.tau32) / 0.05582565) * (1 + math.erf(((0.456416 - Q.tau32) / 0.05582565) / math.sqrt(2))))
    z += 0.5123706 * (0.01743646 * 0.5 * ((0.3394107 - Q.sj2_dr) / 0.01743646) * (1 + math.erf(((0.3394107 - Q.sj2_dr) / 0.01743646) / math.sqrt(2))))
    z += 0.001091164 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2))))
    z += 1.191234 * (0.0326228 * 0.5 * ((Q.kt2_dr12 - 0.4918357) / 0.0326228) * (1 + math.erf(((Q.kt2_dr12 - 0.4918357) / 0.0326228) / math.sqrt(2))))
    z += -0.002690326 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2))))
    z += 0.006785166 * (22.16789 * 0.5 * ((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) * (1 + math.erf(((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) / math.sqrt(2))))
    z += -0.003349802 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (401.9953 * 0.5 * ((577.991 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((577.991 - Q.sip_3d_3) / 401.9953) / math.sqrt(2))))
    z += -0.000802384 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.06842582 * 0.5 * ((Q.lne_1 - 4.635336) / 0.06842582) * (1 + math.erf(((Q.lne_1 - 4.635336) / 0.06842582) / math.sqrt(2))))
    z += -0.05218156 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.01952245 * 0.5 * ((0.1701336 - Q.dr_21) / 0.01952245) * (1 + math.erf(((0.1701336 - Q.dr_21) / 0.01952245) / math.sqrt(2))))
    z += 380.7979 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2))))
    z += 0.07499882 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2))))
    z += 4407.802 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.01565591 * 0.5 * ((0.1681753 - Q.pz_lnkt1) / 0.01565591) * (1 + math.erf(((0.1681753 - Q.pz_lnkt1) / 0.01565591) / math.sqrt(2))))
    z += -0.06505142 * (1.5 * 0.5 * ((6.0 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((6.0 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2))))
    z += -0.0007086874 * (3.304036 * 0.5 * ((115.614 - Q.mass_top50) / 3.304036) * (1 + math.erf(((115.614 - Q.mass_top50) / 3.304036) / math.sqrt(2)))) * (1.0 * 0.5 * ((6.0 - Q.sdb_1_n) / 1.0) * (1 + math.erf(((6.0 - Q.sdb_1_n) / 1.0) / math.sqrt(2))))
    z += 0.1655141 * (0.09740396 * 0.5 * ((1.325034 - Q.D2) / 0.09740396) * (1 + math.erf(((1.325034 - Q.D2) / 0.09740396) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjq_3_2_nch - 4.0) / 1.0) * (1 + math.erf(((Q.sjq_3_2_nch - 4.0) / 1.0) / math.sqrt(2))))
    z += 0.02010438 * (1.0 * 0.5 * ((2.0 - Q.sum_charge) / 1.0) * (1 + math.erf(((2.0 - Q.sum_charge) / 1.0) / math.sqrt(2))))
    z += -0.07445225 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2)))) * (0.02417016 * 0.5 * ((Q.z_photon - 0.3318968) / 0.02417016) * (1 + math.erf(((Q.z_photon - 0.3318968) / 0.02417016) / math.sqrt(2))))
    z += -1.101185 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_3_3_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_3_3_n_d3 - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.005247873 * (13.85751 * 0.5 * ((Q.sj2_mass1 - 91.2852) / 13.85751) * (1 + math.erf(((Q.sj2_mass1 - 91.2852) / 13.85751) / math.sqrt(2))))
    z += 1.527154 * (0.2610212 * 0.5 * ((-0.6337755 - Q.sjq_2_2_k1) / 0.2610212) * (1 + math.erf(((-0.6337755 - Q.sjq_2_2_k1) / 0.2610212) / math.sqrt(2))))
    z += 2.093445 * (0.2681069 * 0.5 * ((Q.sjq_2_2_k1 - 0.6477929) / 0.2681069) * (1 + math.erf(((Q.sjq_2_2_k1 - 0.6477929) / 0.2681069) / math.sqrt(2))))
    z += 0.07271696 * (1.0 * 0.5 * ((8.0 - Q.n_lund) / 1.0) * (1 + math.erf(((8.0 - Q.n_lund) / 1.0) / math.sqrt(2))))
    z += -1.252001 * (0.0408915 * 0.5 * ((Q.lep_dr - 0.04607888) / 0.0408915) * (1 + math.erf(((Q.lep_dr - 0.04607888) / 0.0408915) / math.sqrt(2))))
    z += 0.09724856 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2)))) * (0.02550179 * 0.5 * ((Q.sdb_2_z - 0.1792389) / 0.02550179) * (1 + math.erf(((Q.sdb_2_z - 0.1792389) / 0.02550179) / math.sqrt(2))))
    z += -0.0126456 * (7.767937 * 0.5 * ((Q.sj3_mass1 - 44.1303) / 7.767937) * (1 + math.erf(((Q.sj3_mass1 - 44.1303) / 7.767937) / math.sqrt(2))))
    z += 0.02363198 * (0.7068263 * 0.5 * ((Q.lne_16 - 0.4468807) / 0.7068263) * (1 + math.erf(((Q.lne_16 - 0.4468807) / 0.7068263) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 0.0) / 1.0) * (1 + math.erf(((Q.n_muon - 0.0) / 1.0) / math.sqrt(2))))
    z += -7.059275 * (0.0408915 * 0.5 * ((Q.lep_dr - 0.04607888) / 0.0408915) * (1 + math.erf(((Q.lep_dr - 0.04607888) / 0.0408915) / math.sqrt(2)))) * (0.03296266 * 0.5 * ((0.04394648 - Q.sv_1_dr) / 0.03296266) * (1 + math.erf(((0.04394648 - Q.sv_1_dr) / 0.03296266) / math.sqrt(2))))
    z += 0.002271866 * (3.100743 * 0.5 * ((65.0404 - Q.mass_charged) / 3.100743) * (1 + math.erf(((65.0404 - Q.mass_charged) / 3.100743) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.isphoton_19) / 1.0) * (1 + math.erf(((1.0 - Q.isphoton_19) / 1.0) / math.sqrt(2))))
    z += 0.3561045 * (0.1159667 * 0.5 * ((0.4849193 - Q.sv_1_dr) / 0.1159667) * (1 + math.erf(((0.4849193 - Q.sv_1_dr) / 0.1159667) / math.sqrt(2))))
    z += 0.0002335808 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.036273 * 0.5 * ((Q.td0_13 - -0.06278656) / 0.036273) * (1 + math.erf(((Q.td0_13 - -0.06278656) / 0.036273) / math.sqrt(2))))
    z += 0.07291785 * (0.7202602 * 0.5 * ((1.462588 - Q.D3_b2) / 0.7202602) * (1 + math.erf(((1.462588 - Q.D3_b2) / 0.7202602) / math.sqrt(2))))
    z += -6690.938 * (0.03097976 * 0.5 * ((0.1342762 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.1342762 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (0.0001714083 * 0.5 * ((0.0002104854 - Q.C3_b2) / 0.0001714083) * (1 + math.erf(((0.0002104854 - Q.C3_b2) / 0.0001714083) / math.sqrt(2))))
    z += 0.03868231 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) / math.sqrt(2)))) * (0.03559215 * 0.5 * ((0.6203012 - Q.tau32_b2) / 0.03559215) * (1 + math.erf(((0.6203012 - Q.tau32_b2) / 0.03559215) / math.sqrt(2))))
    z += 0.0102573 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2)))) * (1.751638 * 0.5 * ((5.460234 - Q.D2_b2) / 1.751638) * (1 + math.erf(((5.460234 - Q.D2_b2) / 1.751638) / math.sqrt(2))))
    z += 2.359123 * (0.0496275 * 0.5 * ((0.1915984 - Q.kt2_dr12) / 0.0496275) * (1 + math.erf(((0.1915984 - Q.kt2_dr12) / 0.0496275) / math.sqrt(2))))
    z += -0.02024238 * (24.25363 * 0.5 * ((53.57509 - Q.mres_sd_mass_b0z005) / 24.25363) * (1 + math.erf(((53.57509 - Q.mres_sd_mass_b0z005) / 24.25363) / math.sqrt(2))))
    z += -0.002571128 * (8.307776 * 0.5 * ((Q.mass_top50 - 135.5368) / 8.307776) * (1 + math.erf(((Q.mass_top50 - 135.5368) / 8.307776) / math.sqrt(2))))
    z += -0.000679315 * (5.345127 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) / math.sqrt(2))))
    z += 3811.778 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.0003494239 * 0.5 * ((Q.C3_b2 - 0.0007327206) / 0.0003494239) * (1 + math.erf(((Q.C3_b2 - 0.0007327206) / 0.0003494239) / math.sqrt(2))))
    z += -0.02984792 * (1.0 * 0.5 * ((2.0 - Q.sum_charge) / 1.0) * (1 + math.erf(((2.0 - Q.sum_charge) / 1.0) / math.sqrt(2)))) * (0.01620483 * 0.5 * ((Q.d0err_34 - 0.02580261) / 0.01620483) * (1 + math.erf(((Q.d0err_34 - 0.02580261) / 0.01620483) / math.sqrt(2))))
    z += -0.1198393 * (0.08168809 * 0.5 * ((Q.jet_abs_eta - 0.8317778) / 0.08168809) * (1 + math.erf(((Q.jet_abs_eta - 0.8317778) / 0.08168809) / math.sqrt(2))))
    z += 0.09982856 * (1.5 * 0.5 * ((7.0 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.0 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2))))
    z += -0.18264 * (1.5 * 0.5 * ((6.0 - Q.n_sd0_above_3) / 1.5) * (1 + math.erf(((6.0 - Q.n_sd0_above_3) / 1.5) / math.sqrt(2))))
    z += 1.455543 * (0.08149619 * 0.5 * ((0.2477181 - Q.pt1_over_pt0) / 0.08149619) * (1 + math.erf(((0.2477181 - Q.pt1_over_pt0) / 0.08149619) / math.sqrt(2))))
    z += -0.1493551 * (0.02937692 * 0.5 * ((Q.sj3_pairmin_over_m - 0.4866692) / 0.02937692) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.4866692) / 0.02937692) / math.sqrt(2))))
    z += 0.00122063 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) * (1.375431 * 0.5 * ((Q.mass_2photon - 4.18513) / 1.375431) * (1 + math.erf(((Q.mass_2photon - 4.18513) / 1.375431) / math.sqrt(2))))
    z += -0.007249789 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2))))
    z += -0.6410883 * (1.0 * 0.5 * ((0.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((0.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2))))
    z += 0.0087518 * (2.5 * 0.5 * ((Q.n_particles - 26.0) / 2.5) * (1 + math.erf(((Q.n_particles - 26.0) / 2.5) / math.sqrt(2))))
    z += -0.7543991 * (0.01266368 * 0.5 * ((0.05462126 - Q.pz_lnd2) / 0.01266368) * (1 + math.erf(((0.05462126 - Q.pz_lnd2) / 0.01266368) / math.sqrt(2))))
    z += -0.2726798 * (0.08734879 * 0.5 * ((1.608106 - Q.jd_3d_6) / 0.08734879) * (1 + math.erf(((1.608106 - Q.jd_3d_6) / 0.08734879) / math.sqrt(2))))
    return z


def neuron_53(Q):
    z = -1.889358e-06
    return z


def neuron_54(Q):
    z = 1.190658e-05
    return z


def neuron_55(Q):
    z = -2.066937e-05
    return z


def neuron_56(Q):
    z = -7.072324e-07
    return z


def neuron_57(Q):
    z = -1.959646e-06
    return z


def neuron_58(Q):
    z = -3.617147e-06
    return z


def neuron_59(Q):
    z = 6.029329e-06
    return z


def neuron_60(Q):
    z = -4.666754e-06
    return z


def neuron_61(Q):
    z = 5.840554e-06
    return z


def neuron_62(Q):
    z = 9.422957e-07
    return z


def neuron_63(Q):
    z = -8.308504e-06
    return z


def neuron_64(Q):
    z = 1.258623e-06
    return z


def neuron_65(Q):
    z = 5.791053e-06
    return z


def neuron_66(Q):
    z = 2.709773e-06
    return z


def neuron_67(Q):
    z = -5.219507e-06
    return z


def neuron_68(Q):
    z = 1.704409
    z += -2.561278 * Q.C2
    z += 1.079681 * (0.0216761 * 0.5 * ((0.820003 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.820003 - Q.tau32) / 0.0216761) / math.sqrt(2))))
    z += -0.03014221 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.983043) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.983043) / 4.308625) / math.sqrt(2))))
    z += -0.1980101 * (0.2410611 * 0.5 * ((3.38061 - Q.pair_mean_lnm2) / 0.2410611) * (1 + math.erf(((3.38061 - Q.pair_mean_lnm2) / 0.2410611) / math.sqrt(2))))
    z += -555.0087 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2))))
    z += 54.40865 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2))))
    z += -2.964992 * (0.0138316 * 0.5 * ((0.2513409 - Q.pz_lnd2) / 0.0138316) * (1 + math.erf(((0.2513409 - Q.pz_lnd2) / 0.0138316) / math.sqrt(2))))
    z += -4600.26 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.03798057 * 0.5 * ((Q.z_top30_slots - 0.8316085) / 0.03798057) * (1 + math.erf(((Q.z_top30_slots - 0.8316085) / 0.03798057) / math.sqrt(2))))
    z += -0.01251934 * (10.8986 * 0.5 * ((67.20576 - Q.mass_top30) / 10.8986) * (1 + math.erf(((67.20576 - Q.mass_top30) / 10.8986) / math.sqrt(2))))
    z += 0.01214094 * (0.0216761 * 0.5 * ((0.820003 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.820003 - Q.tau32) / 0.0216761) / math.sqrt(2)))) * (11.90664 * 0.5 * ((80.02563 - Q.sj3_pair_mass_min) / 11.90664) * (1 + math.erf(((80.02563 - Q.sj3_pair_mass_min) / 11.90664) / math.sqrt(2))))
    z += 0.0113046 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2))))
    z += -0.009821332 * (16.26099 * 0.5 * ((161.1264 - Q.mass_top50) / 16.26099) * (1 + math.erf(((161.1264 - Q.mass_top50) / 16.26099) / math.sqrt(2))))
    z += 0.01066301 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2))))
    z += 0.0268268 * (22.10989 * 0.5 * ((Q.mres_pruned_mass - 171.3819) / 22.10989) * (1 + math.erf(((Q.mres_pruned_mass - 171.3819) / 22.10989) / math.sqrt(2))))
    z += 0.5046195 * (0.0218375 * 0.5 * ((0.07650476 - Q.ak02_3_z) / 0.0218375) * (1 + math.erf(((0.07650476 - Q.ak02_3_z) / 0.0218375) / math.sqrt(2))))
    z += 0.427135 * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.265564) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.265564) / 0.0649922) / math.sqrt(2))))
    z += -0.5287939 * (0.03746729 * 0.5 * ((Q.z_displaced3 - 0.1681173) / 0.03746729) * (1 + math.erf(((Q.z_displaced3 - 0.1681173) / 0.03746729) / math.sqrt(2))))
    z += -0.02293812 * (0.2410611 * 0.5 * ((3.38061 - Q.pair_mean_lnm2) / 0.2410611) * (1 + math.erf(((3.38061 - Q.pair_mean_lnm2) / 0.2410611) / math.sqrt(2)))) * (6.12936 * 0.5 * ((Q.lnptrel_31 - -6.668647) / 6.12936) * (1 + math.erf(((Q.lnptrel_31 - -6.668647) / 6.12936) / math.sqrt(2))))
    z += -0.06310116 * (6.036668 * 0.5 * ((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) * (1 + math.erf(((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) / math.sqrt(2))))
    z += 0.02041213 * (3.499256 * 0.5 * ((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) * (1 + math.erf(((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) / math.sqrt(2))))
    z += -0.01508586 * (19.14168 * 0.5 * ((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2))))
    z += 0.9639564 * (0.01332924 * 0.5 * ((0.9062492 - Q.tau43) / 0.01332924) * (1 + math.erf(((0.9062492 - Q.tau43) / 0.01332924) / math.sqrt(2))))
    z += -0.0001078664 * (3.499256 * 0.5 * ((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) * (1 + math.erf(((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.01170161 * (1.0 * 0.5 * ((Q.sdb_2_n - 11.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 11.0) / 1.0) / math.sqrt(2))))
    z += -0.7109491 * (0.02593915 * 0.5 * ((Q.mres_sd_rg_b1z01 - 0.4673199) / 0.02593915) * (1 + math.erf(((Q.mres_sd_rg_b1z01 - 0.4673199) / 0.02593915) / math.sqrt(2))))
    z += 248.6099 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 2.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 2.0) / 1.0) / math.sqrt(2))))
    z += 0.6096456 * (0.2410611 * 0.5 * ((3.38061 - Q.pair_mean_lnm2) / 0.2410611) * (1 + math.erf(((3.38061 - Q.pair_mean_lnm2) / 0.2410611) / math.sqrt(2)))) * (0.06955833 * 0.5 * ((0.3788785 - Q.z_neutral_had) / 0.06955833) * (1 + math.erf(((0.3788785 - Q.z_neutral_had) / 0.06955833) / math.sqrt(2))))
    z += 0.03471928 * (0.0138316 * 0.5 * ((0.2513409 - Q.pz_lnd2) / 0.0138316) * (1 + math.erf(((0.2513409 - Q.pz_lnd2) / 0.0138316) / math.sqrt(2)))) * (11.72898 * 0.5 * ((Q.lnptrel_25 - -18.42068) / 11.72898) * (1 + math.erf(((Q.lnptrel_25 - -18.42068) / 11.72898) / math.sqrt(2))))
    z += 0.0008530656 * (29.74772 * 0.5 * ((Q.mres_sd_mass_b0z02 - 163.95) / 29.74772) * (1 + math.erf(((Q.mres_sd_mass_b0z02 - 163.95) / 29.74772) / math.sqrt(2))))
    z += -0.1164632 * (1.0 * 0.5 * ((3.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((3.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2))))
    z += 0.03581281 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.983043) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.983043) / 4.308625) / math.sqrt(2)))) * (0.0531045 * 0.5 * ((0.221369 - Q.C2_b2) / 0.0531045) * (1 + math.erf(((0.221369 - Q.C2_b2) / 0.0531045) / math.sqrt(2))))
    z += 3.003844 * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.265564) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.265564) / 0.0649922) / math.sqrt(2)))) * (0.01039253 * 0.5 * ((0.9216755 - Q.tau54) / 0.01039253) * (1 + math.erf(((0.9216755 - Q.tau54) / 0.01039253) / math.sqrt(2))))
    z += 1.089562 * (0.007115881 * 0.5 * ((Q.sjf_4_1_z_d3 - 0.00625791) / 0.007115881) * (1 + math.erf(((Q.sjf_4_1_z_d3 - 0.00625791) / 0.007115881) / math.sqrt(2))))
    z += 0.04414074 * (19.93276 * 0.5 * ((55.13212 - Q.mres_sd_mass_b2z01) / 19.93276) * (1 + math.erf(((55.13212 - Q.mres_sd_mass_b2z01) / 19.93276) / math.sqrt(2))))
    z += 0.006075717 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += 0.01865383 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2))))
    z += -0.2110305 * (0.1659628 * 0.5 * ((1.465946 - Q.jet_abs_eta) / 0.1659628) * (1 + math.erf(((1.465946 - Q.jet_abs_eta) / 0.1659628) / math.sqrt(2))))
    z += -0.01489971 * (11.90664 * 0.5 * ((80.02563 - Q.sj3_pair_mass_min) / 11.90664) * (1 + math.erf(((80.02563 - Q.sj3_pair_mass_min) / 11.90664) / math.sqrt(2))))
    z += 0.004128983 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) * (0.3667324 * 0.5 * ((2.613544 - Q.D2_b2) / 0.3667324) * (1 + math.erf(((2.613544 - Q.D2_b2) / 0.3667324) / math.sqrt(2))))
    z += -0.2185393 * (0.298047 * 0.5 * ((2.286291 - Q.D2_b2) / 0.298047) * (1 + math.erf(((2.286291 - Q.D2_b2) / 0.298047) / math.sqrt(2))))
    z += -1.496087 * (0.02630693 * 0.5 * ((0.2376554 - Q.pz_lnd3) / 0.02630693) * (1 + math.erf(((0.2376554 - Q.pz_lnd3) / 0.02630693) / math.sqrt(2))))
    z += -0.02091883 * (1.0 * 0.5 * ((Q.sdb_2_n - 11.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 11.0) / 1.0) / math.sqrt(2)))) * (1.739103 * 0.5 * ((3.969624 - Q.jd_3d_4) / 1.739103) * (1 + math.erf(((3.969624 - Q.jd_3d_4) / 1.739103) / math.sqrt(2))))
    z += 0.02872216 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2))))
    z += -0.005548406 * (10.67504 * 0.5 * ((Q.mass_top10 - 103.4976) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 103.4976) / 10.67504) / math.sqrt(2))))
    z += -2.194385 * (0.07842787 * 0.5 * ((Q.dc_split2_dr - 0.3591078) / 0.07842787) * (1 + math.erf(((Q.dc_split2_dr - 0.3591078) / 0.07842787) / math.sqrt(2))))
    z += 0.03878566 * (7.585662 * 0.5 * ((Q.lep_ptrel - 18.7678) / 7.585662) * (1 + math.erf(((Q.lep_ptrel - 18.7678) / 7.585662) / math.sqrt(2))))
    z += -0.01262847 * (5.345127 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) / math.sqrt(2))))
    z += 0.01849118 * (2.0 * 0.5 * ((11.0 - Q.n_sd0_above_2) / 2.0) * (1 + math.erf(((11.0 - Q.n_sd0_above_2) / 2.0) / math.sqrt(2))))
    z += -0.07958535 * (2.0 * 0.5 * ((11.0 - Q.n_sd0_above_2) / 2.0) * (1 + math.erf(((11.0 - Q.n_sd0_above_2) / 2.0) / math.sqrt(2)))) * (0.117671 * 0.5 * ((0.4191372 - Q.lep_dr) / 0.117671) * (1 + math.erf(((0.4191372 - Q.lep_dr) / 0.117671) / math.sqrt(2))))
    z += 0.008164494 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.0001925121 * (0.007115881 * 0.5 * ((Q.sjf_4_1_z_d3 - 0.00625791) / 0.007115881) * (1 + math.erf(((Q.sjf_4_1_z_d3 - 0.00625791) / 0.007115881) / math.sqrt(2)))) * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2))))
    z += -4.155503 * (0.02099671 * 0.5 * ((0.08304558 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.08304558 - Q.z_displaced3) / 0.02099671) / math.sqrt(2))))
    z += 0.06763548 * (12.47875 * 0.5 * ((Q.mres_pruned_mass - 131.2974) / 12.47875) * (1 + math.erf(((Q.mres_pruned_mass - 131.2974) / 12.47875) / math.sqrt(2))))
    z += -0.02185378 * (5.270719 * 0.5 * ((Q.mres_pruned_mass - 86.14266) / 5.270719) * (1 + math.erf(((Q.mres_pruned_mass - 86.14266) / 5.270719) / math.sqrt(2))))
    z += -0.07845269 * (20.04228 * 0.5 * ((Q.mres_pruned_mass - 149.272) / 20.04228) * (1 + math.erf(((Q.mres_pruned_mass - 149.272) / 20.04228) / math.sqrt(2))))
    z += -0.02774349 * (5.270719 * 0.5 * ((Q.mres_pruned_mass - 86.14266) / 5.270719) * (1 + math.erf(((Q.mres_pruned_mass - 86.14266) / 5.270719) / math.sqrt(2)))) * (0.02798345 * 0.5 * ((0.171574 - Q.dc_3_z) / 0.02798345) * (1 + math.erf(((0.171574 - Q.dc_3_z) / 0.02798345) / math.sqrt(2))))
    z += 0.497324 * (0.05861841 * 0.5 * ((Q.ak02_dr23 - 0.3728877) / 0.05861841) * (1 + math.erf(((Q.ak02_dr23 - 0.3728877) / 0.05861841) / math.sqrt(2))))
    z += 0.0103999 * (18.57486 * 0.5 * ((Q.mres_pruned_mass - 41.77934) / 18.57486) * (1 + math.erf(((Q.mres_pruned_mass - 41.77934) / 18.57486) / math.sqrt(2))))
    z += -4470.397 * (3.274853e-06 * 0.5 * ((1.008706e-05 - Q.e3_b2) / 3.274853e-06) * (1 + math.erf(((1.008706e-05 - Q.e3_b2) / 3.274853e-06) / math.sqrt(2))))
    z += 5.195613 * (0.00254714 * 0.5 * ((0.0391767 - Q.M3) / 0.00254714) * (1 + math.erf(((0.0391767 - Q.M3) / 0.00254714) / math.sqrt(2))))
    z += -0.005449304 * (1.5 * 0.5 * ((Q.n_dr_0p2_0p4 - 7.0) / 1.5) * (1 + math.erf(((Q.n_dr_0p2_0p4 - 7.0) / 1.5) / math.sqrt(2))))
    z += 0.01470139 * (0.03746729 * 0.5 * ((Q.z_displaced3 - 0.1681173) / 0.03746729) * (1 + math.erf(((Q.z_displaced3 - 0.1681173) / 0.03746729) / math.sqrt(2)))) * (6.000774 * 0.5 * ((14.85973 - Q.jd_3d_4) / 6.000774) * (1 + math.erf(((14.85973 - Q.jd_3d_4) / 6.000774) / math.sqrt(2))))
    z += 0.02048391 * (4.590131 * 0.5 * ((107.2089 - Q.mres_sd_mass_b2z01) / 4.590131) * (1 + math.erf(((107.2089 - Q.mres_sd_mass_b2z01) / 4.590131) / math.sqrt(2))))
    z += 0.01235374 * (1.369872 * 0.5 * ((Q.sj3_mass1 - 17.31003) / 1.369872) * (1 + math.erf(((Q.sj3_mass1 - 17.31003) / 1.369872) / math.sqrt(2))))
    z += -0.494193 * (0.08347678 * 0.5 * ((Q.tau21_b2 - 0.5868564) / 0.08347678) * (1 + math.erf(((Q.tau21_b2 - 0.5868564) / 0.08347678) / math.sqrt(2))))
    z += -1.481697 * (0.02719315 * 0.5 * ((0.1530389 - Q.sdb_2_z) / 0.02719315) * (1 + math.erf(((0.1530389 - Q.sdb_2_z) / 0.02719315) / math.sqrt(2))))
    z += -0.01587632 * (1.0 * 0.5 * ((Q.n_lund_kt_above_1 - 4.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_1 - 4.0) / 1.0) / math.sqrt(2))))
    z += 0.2457757 * (0.1750793 * 0.5 * ((0.4436035 - Q.max_abs_dz) / 0.1750793) * (1 + math.erf(((0.4436035 - Q.max_abs_dz) / 0.1750793) / math.sqrt(2))))
    z += -0.05329619 * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 20.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 20.0) / 1.0) / math.sqrt(2))))
    z += 1.071188 * (0.0216761 * 0.5 * ((0.820003 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.820003 - Q.tau32) / 0.0216761) / math.sqrt(2)))) * (0.01950041 * 0.5 * ((Q.sv_1_dr - 0.1227033) / 0.01950041) * (1 + math.erf(((Q.sv_1_dr - 0.1227033) / 0.01950041) / math.sqrt(2))))
    z += -0.02414244 * (1.0 * 0.5 * ((Q.n_neutral_had - 6.0) / 1.0) * (1 + math.erf(((Q.n_neutral_had - 6.0) / 1.0) / math.sqrt(2))))
    z += 0.003699434 * (0.0138316 * 0.5 * ((0.2513409 - Q.pz_lnd2) / 0.0138316) * (1 + math.erf(((0.2513409 - Q.pz_lnd2) / 0.0138316) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2))))
    z += -17.96769 * (0.00254714 * 0.5 * ((0.0391767 - Q.M3) / 0.00254714) * (1 + math.erf(((0.0391767 - Q.M3) / 0.00254714) / math.sqrt(2)))) * (0.06042313 * 0.5 * ((Q.sjq_3_sumabs_k1 - 0.4372817) / 0.06042313) * (1 + math.erf(((Q.sjq_3_sumabs_k1 - 0.4372817) / 0.06042313) / math.sqrt(2))))
    z += 44.20557 * (12.47875 * 0.5 * ((Q.mres_pruned_mass - 131.2974) / 12.47875) * (1 + math.erf(((Q.mres_pruned_mass - 131.2974) / 12.47875) / math.sqrt(2)))) * (0.002064355 * 0.5 * ((0.0 - Q.lepsj_3_dr) / 0.002064355) * (1 + math.erf(((0.0 - Q.lepsj_3_dr) / 0.002064355) / math.sqrt(2))))
    z += -0.001301073 * (11.90664 * 0.5 * ((80.02563 - Q.sj3_pair_mass_min) / 11.90664) * (1 + math.erf(((80.02563 - Q.sj3_pair_mass_min) / 11.90664) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -42.63734 * (0.02099671 * 0.5 * ((0.08304558 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.08304558 - Q.z_displaced3) / 0.02099671) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.n_muon) / 1.0) * (1 + math.erf(((0.0 - Q.n_muon) / 1.0) / math.sqrt(2))))
    z += 0.004351888 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.n_muon) / 1.0) * (1 + math.erf(((1.0 - Q.n_muon) / 1.0) / math.sqrt(2))))
    z += -0.0846085 * (1.0 * 0.5 * ((Q.sdb_2_n - 11.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 11.0) / 1.0) / math.sqrt(2)))) * (0.01645773 * 0.5 * ((0.5838314 - Q.nca_sj4_pair2nd_over_mass) / 0.01645773) * (1 + math.erf(((0.5838314 - Q.nca_sj4_pair2nd_over_mass) / 0.01645773) / math.sqrt(2))))
    z += 0.0006942211 * (10.93201 * 0.5 * ((Q.mass_top10 - 27.42853) / 10.93201) * (1 + math.erf(((Q.mass_top10 - 27.42853) / 10.93201) / math.sqrt(2))))
    z += 0.1087902 * (0.1783594 * 0.5 * ((Q.ktd_ln_d34 - -9.180886) / 0.1783594) * (1 + math.erf(((Q.ktd_ln_d34 - -9.180886) / 0.1783594) / math.sqrt(2))))
    z += 0.01091451 * (11.17405 * 0.5 * ((Q.sj3_pair_mass_max - 128.6605) / 11.17405) * (1 + math.erf(((Q.sj3_pair_mass_max - 128.6605) / 11.17405) / math.sqrt(2))))
    z += -0.01296721 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2)))) * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2))))
    z += -0.008649636 * (2.973269 * 0.5 * ((Q.sj4_pair_mass_max - 72.862) / 2.973269) * (1 + math.erf(((Q.sj4_pair_mass_max - 72.862) / 2.973269) / math.sqrt(2))))
    z += 0.0005355648 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.ak02_1_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_1_n_lep) / 1.0) / math.sqrt(2))))
    z += -1.527574 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += 5.558263 * (0.03429006 * 0.5 * ((0.08178299 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.08178299 - Q.lep_dr) / 0.03429006) / math.sqrt(2))))
    z += 9.10393e-05 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (0.1600921 * 0.5 * ((Q.mass_displaced5 - 6.5555e-08) / 0.1600921) * (1 + math.erf(((Q.mass_displaced5 - 6.5555e-08) / 0.1600921) / math.sqrt(2))))
    z += 0.001404772 * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 20.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 20.0) / 1.0) / math.sqrt(2)))) * (1.5 * 0.5 * ((17.0 - Q.n_dr_0p1_0p2) / 1.5) * (1 + math.erf(((17.0 - Q.n_dr_0p1_0p2) / 1.5) / math.sqrt(2))))
    z += -0.5706879 * (0.179073 * 0.5 * ((Q.lep_z - 0.5187302) / 0.179073) * (1 + math.erf(((Q.lep_z - 0.5187302) / 0.179073) / math.sqrt(2))))
    z += -0.001832234 * (0.0218375 * 0.5 * ((0.07650476 - Q.ak02_3_z) / 0.0218375) * (1 + math.erf(((0.07650476 - Q.ak02_3_z) / 0.0218375) / math.sqrt(2)))) * (1032.513 * 0.5 * ((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) * (1 + math.erf(((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) / math.sqrt(2))))
    z += -0.6502618 * (0.298047 * 0.5 * ((2.286291 - Q.D2_b2) / 0.298047) * (1 + math.erf(((2.286291 - Q.D2_b2) / 0.298047) / math.sqrt(2)))) * (0.003370318 * 0.5 * ((Q.sjf_4_4_z_d3 - 0.0) / 0.003370318) * (1 + math.erf(((Q.sjf_4_4_z_d3 - 0.0) / 0.003370318) / math.sqrt(2))))
    z += -0.003592224 * (19.14168 * 0.5 * ((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2)))) * (0.002556514 * 0.5 * ((Q.sjf_4_3_z_d3 - 0.001692006) / 0.002556514) * (1 + math.erf(((Q.sjf_4_3_z_d3 - 0.001692006) / 0.002556514) / math.sqrt(2))))
    z += 3.258337 * (0.2297361 * 0.5 * ((0.0 - Q.ak02_dr13) / 0.2297361) * (1 + math.erf(((0.0 - Q.ak02_dr13) / 0.2297361) / math.sqrt(2))))
    z += 135.1309 * (0.001201988 * 0.5 * ((0.002792418 - Q.sum_z_dr2_top3) / 0.001201988) * (1 + math.erf(((0.002792418 - Q.sum_z_dr2_top3) / 0.001201988) / math.sqrt(2))))
    z += 3.27358 * (0.07842787 * 0.5 * ((Q.dc_split2_dr - 0.3591078) / 0.07842787) * (1 + math.erf(((Q.dc_split2_dr - 0.3591078) / 0.07842787) / math.sqrt(2)))) * (0.2807389 * 0.5 * ((Q.lund3_lnkt - 4.319803) / 0.2807389) * (1 + math.erf(((Q.lund3_lnkt - 4.319803) / 0.2807389) / math.sqrt(2))))
    z += 0.008479756 * (0.188475 * 0.5 * ((1.634536 - Q.dc_tag_max) / 0.188475) * (1 + math.erf(((1.634536 - Q.dc_tag_max) / 0.188475) / math.sqrt(2))))
    z += -2.252625 * (0.02434011 * 0.5 * ((Q.sj3_pairmin_over_m - 0.4610803) / 0.02434011) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.4610803) / 0.02434011) / math.sqrt(2))))
    z += 0.01115199 * (1.504711 * 0.5 * ((Q.sj3_mass2 - 15.2797) / 1.504711) * (1 + math.erf(((Q.sj3_mass2 - 15.2797) / 1.504711) / math.sqrt(2))))
    z += 516.9619 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2))))
    return z


def neuron_69(Q):
    z = -6.92614e-06
    return z


def neuron_70(Q):
    z = 0.01392552
    z += 0.1104868 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2))))
    z += 0.00377114 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 0.01111772 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (16.47782 * 0.5 * ((162.7874 - Q.mass_top30) / 16.47782) * (1 + math.erf(((162.7874 - Q.mass_top30) / 16.47782) / math.sqrt(2))))
    z += 0.1171579 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2))))
    z += -0.01738393 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2))))
    z += -0.000272542 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2))))
    z += 0.8559868 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.0330605 * 0.5 * ((0.114659 - Q.lep_dr) / 0.0330605) * (1 + math.erf(((0.114659 - Q.lep_dr) / 0.0330605) / math.sqrt(2))))
    z += 3.436973 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.03538975 * 0.5 * ((0.5871757 - Q.tau32) / 0.03538975) * (1 + math.erf(((0.5871757 - Q.tau32) / 0.03538975) / math.sqrt(2))))
    z += 0.09108467 * (0.8047829 * 0.5 * ((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) * (1 + math.erf(((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) / math.sqrt(2))))
    z += 95.72433 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2))))
    z += -1022.586 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.02942179 * 0.5 * ((Q.tau21 - 0.2377332) / 0.02942179) * (1 + math.erf(((Q.tau21 - 0.2377332) / 0.02942179) / math.sqrt(2))))
    z += -0.2268347 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2))))
    z += -0.0173484 * (19.33798 * 0.5 * ((Q.mres_sd_mass_b2z01 - 177.5398) / 19.33798) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 177.5398) / 19.33798) / math.sqrt(2))))
    z += 0.009029839 * (3.764443 * 0.5 * ((Q.mass_top50 - 125.4927) / 3.764443) * (1 + math.erf(((Q.mass_top50 - 125.4927) / 3.764443) / math.sqrt(2))))
    z += 3.790139 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (11.4891 * 0.5 * ((32.61679 - Q.mass_displaced5) / 11.4891) * (1 + math.erf(((32.61679 - Q.mass_displaced5) / 11.4891) / math.sqrt(2))))
    z += 0.007166946 * (11.49171 * 0.5 * ((71.96396 - Q.mass) / 11.49171) * (1 + math.erf(((71.96396 - Q.mass) / 11.49171) / math.sqrt(2))))
    z += 0.003303542 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2))))
    z += -0.004793615 * (16.26099 * 0.5 * ((Q.mass_top50 - 161.1264) / 16.26099) * (1 + math.erf(((Q.mass_top50 - 161.1264) / 16.26099) / math.sqrt(2))))
    z += 0.03991501 * (1.0 * 0.5 * ((18.0 - Q.n_neutral) / 1.0) * (1 + math.erf(((18.0 - Q.n_neutral) / 1.0) / math.sqrt(2))))
    z += -0.00999123 * (3.563419 * 0.5 * ((Q.mres_sd_mass_b0z005 - 122.1) / 3.563419) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 122.1) / 3.563419) / math.sqrt(2))))
    z += -0.007445341 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2))))
    z += 1.234553 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2))))
    z += 0.006213316 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.06755743 * 0.5 * ((0.5017201 - Q.ak02_dr23) / 0.06755743) * (1 + math.erf(((0.5017201 - Q.ak02_dr23) / 0.06755743) / math.sqrt(2))))
    z += 0.01400305 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) / math.sqrt(2))))
    z += 8.607935e-05 * (11.41538 * 0.5 * ((Q.mres_sd_mass_b0z005 - 69.14897) / 11.41538) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 69.14897) / 11.41538) / math.sqrt(2))))
    z += 5.992587 * (0.009408518 * 0.5 * ((0.1122946 - Q.pz_lnd0) / 0.009408518) * (1 + math.erf(((0.1122946 - Q.pz_lnd0) / 0.009408518) / math.sqrt(2))))
    z += 0.63259 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2))))
    z += -0.2553654 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (22.16789 * 0.5 * ((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) * (1 + math.erf(((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) / math.sqrt(2))))
    z += 0.06924768 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (22.16789 * 0.5 * ((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) * (1 + math.erf(((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) / math.sqrt(2))))
    z += -0.01469695 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (0.04737588 * 0.5 * ((0.1807551 - Q.ak02_3_z) / 0.04737588) * (1 + math.erf(((0.1807551 - Q.ak02_3_z) / 0.04737588) / math.sqrt(2))))
    z += -3.814182e-05 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (540.2081 * 0.5 * ((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2))))
    z += -0.03164056 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2))))
    z += 0.000522567 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.62059) / 5.1259) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2))))
    z += -0.04028905 * (2.0 * 0.5 * ((8.0 - Q.n_charged_pt_above_1) / 2.0) * (1 + math.erf(((8.0 - Q.n_charged_pt_above_1) / 2.0) / math.sqrt(2))))
    z += -0.004335489 * (0.009408518 * 0.5 * ((0.1122946 - Q.pz_lnd0) / 0.009408518) * (1 + math.erf(((0.1122946 - Q.pz_lnd0) / 0.009408518) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2))))
    z += -20.78733 * (0.009408518 * 0.5 * ((0.1122946 - Q.pz_lnd0) / 0.009408518) * (1 + math.erf(((0.1122946 - Q.pz_lnd0) / 0.009408518) / math.sqrt(2)))) * (0.01282622 * 0.5 * ((Q.z_neutral_had - 0.1010123) / 0.01282622) * (1 + math.erf(((Q.z_neutral_had - 0.1010123) / 0.01282622) / math.sqrt(2))))
    z += -5.584496 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.01417841 * 0.5 * ((0.1049877 - Q.C2_b2) / 0.01417841) * (1 + math.erf(((0.1049877 - Q.C2_b2) / 0.01417841) / math.sqrt(2))))
    z += -1.810729 * (0.0134569 * 0.5 * ((Q.C2 - 0.05691773) / 0.0134569) * (1 + math.erf(((Q.C2 - 0.05691773) / 0.0134569) / math.sqrt(2))))
    z += -0.01136784 * (3.974361 * 0.5 * ((Q.sj3_pair_mass_min - 44.1643) / 3.974361) * (1 + math.erf(((Q.sj3_pair_mass_min - 44.1643) / 3.974361) / math.sqrt(2))))
    z += -0.5413045 * (0.01743442 * 0.5 * ((0.3281884 - Q.kt2_dr12) / 0.01743442) * (1 + math.erf(((0.3281884 - Q.kt2_dr12) / 0.01743442) / math.sqrt(2))))
    z += 0.3980268 * (0.117671 * 0.5 * ((Q.lep_dr - 0.4191372) / 0.117671) * (1 + math.erf(((Q.lep_dr - 0.4191372) / 0.117671) / math.sqrt(2))))
    z += -3.521025 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2))))
    z += -0.01771565 * (1.5 * 0.5 * ((Q.n_charged_pt_above_1 - 22.0) / 1.5) * (1 + math.erf(((Q.n_charged_pt_above_1 - 22.0) / 1.5) / math.sqrt(2))))
    z += 0.114087 * (1.0 * 0.5 * ((2.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((2.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += -0.01371854 * (1.0 * 0.5 * ((18.0 - Q.n_neutral) / 1.0) * (1 + math.erf(((18.0 - Q.n_neutral) / 1.0) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += 0.0002324314 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.387908 * 0.5 * ((3.151933 - Q.jd_3d_6) / 1.387908) * (1 + math.erf(((3.151933 - Q.jd_3d_6) / 1.387908) / math.sqrt(2))))
    z += 0.01440448 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.751638 * 0.5 * ((Q.D2_b2 - 5.460234) / 1.751638) * (1 + math.erf(((Q.D2_b2 - 5.460234) / 1.751638) / math.sqrt(2))))
    z += 10.35833 * (0.0134569 * 0.5 * ((Q.C2 - 0.05691773) / 0.0134569) * (1 + math.erf(((Q.C2 - 0.05691773) / 0.0134569) / math.sqrt(2)))) * (0.005748204 * 0.5 * ((0.0548897 - Q.pz_lnkt3) / 0.005748204) * (1 + math.erf(((0.0548897 - Q.pz_lnkt3) / 0.005748204) / math.sqrt(2))))
    z += -0.4312917 * (1.0 * 0.5 * ((0.0 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.0 - Q.sv_2_n) / 1.0) / math.sqrt(2))))
    z += 1.498102 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) / math.sqrt(2))))
    z += 11.38756 * (0.01743442 * 0.5 * ((0.3281884 - Q.kt2_dr12) / 0.01743442) * (1 + math.erf(((0.3281884 - Q.kt2_dr12) / 0.01743442) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.0001035622 * (11.90664 * 0.5 * ((Q.sj3_pair_mass_min - 80.02563) / 11.90664) * (1 + math.erf(((Q.sj3_pair_mass_min - 80.02563) / 11.90664) / math.sqrt(2))))
    z += -0.00465869 * (3.329534 * 0.5 * ((Q.mass_2charged - 14.28253) / 3.329534) * (1 + math.erf(((Q.mass_2charged - 14.28253) / 3.329534) / math.sqrt(2))))
    z += -18.18863 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) * (0.01372201 * 0.5 * ((Q.pz_lnkt1 - 0.03633353) / 0.01372201) * (1 + math.erf(((Q.pz_lnkt1 - 0.03633353) / 0.01372201) / math.sqrt(2))))
    z += -0.7712111 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) * (0.1097229 * 0.5 * ((-2.222088 - Q.lnptrel_3) / 0.1097229) * (1 + math.erf(((-2.222088 - Q.lnptrel_3) / 0.1097229) / math.sqrt(2))))
    z += 0.178945 * (2.077045 * 0.5 * ((1.851735 - Q.sj2_mass2) / 2.077045) * (1 + math.erf(((1.851735 - Q.sj2_mass2) / 2.077045) / math.sqrt(2))))
    z += 0.004190483 * (5.345127 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 77.3635) / 5.345127) / math.sqrt(2))))
    z += 0.1647308 * (1.0 * 0.5 * ((Q.sjf_4_n2disp - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_4_n2disp - 1.0) / 1.0) / math.sqrt(2))))
    z += -0.2204491 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 1.0) / 1.0) / math.sqrt(2))))
    z += 1.616445 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) * (0.02203204 * 0.5 * ((Q.dr_3 - 0.1990664) / 0.02203204) * (1 + math.erf(((Q.dr_3 - 0.1990664) / 0.02203204) / math.sqrt(2))))
    z += 0.009765716 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.8957) / 18.97147) / math.sqrt(2))))
    z += -0.002217502 * (14.05251 * 0.5 * ((Q.mres_sd_mass_b2z01 - 139.2565) / 14.05251) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 139.2565) / 14.05251) / math.sqrt(2))))
    z += 5.983668 * (0.0134569 * 0.5 * ((Q.C2 - 0.05691773) / 0.0134569) * (1 + math.erf(((Q.C2 - 0.05691773) / 0.0134569) / math.sqrt(2)))) * (0.0329306 * 0.5 * ((0.4091922 - Q.ak02_2_z) / 0.0329306) * (1 + math.erf(((0.4091922 - Q.ak02_2_z) / 0.0329306) / math.sqrt(2))))
    z += -0.05311304 * (0.02505229 * 0.5 * ((Q.sdb_2_z - 0.3448514) / 0.02505229) * (1 + math.erf(((Q.sdb_2_z - 0.3448514) / 0.02505229) / math.sqrt(2))))
    z += 0.7016512 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += 0.001294445 * (3.710857 * 0.5 * ((Q.mass_top30 - 91.4679) / 3.710857) * (1 + math.erf(((Q.mass_top30 - 91.4679) / 3.710857) / math.sqrt(2))))
    z += -0.2113594 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) * (0.05310694 * 0.5 * ((0.4358177 - Q.pt1_over_pt0) / 0.05310694) * (1 + math.erf(((0.4358177 - Q.pt1_over_pt0) / 0.05310694) / math.sqrt(2))))
    z += 45.39193 * (0.003553237 * 0.5 * ((0.01143413 - Q.tau4) / 0.003553237) * (1 + math.erf(((0.01143413 - Q.tau4) / 0.003553237) / math.sqrt(2))))
    z += -0.02357071 * (3.710857 * 0.5 * ((Q.mass_top30 - 91.4679) / 3.710857) * (1 + math.erf(((Q.mass_top30 - 91.4679) / 3.710857) / math.sqrt(2)))) * (0.007817624 * 0.5 * ((Q.sjf_4_3_z_d3 - 0.01507439) / 0.007817624) * (1 + math.erf(((Q.sjf_4_3_z_d3 - 0.01507439) / 0.007817624) / math.sqrt(2))))
    z += -21207.05 * (2.210526e-06 * 0.5 * ((4.696862e-06 - Q.e3_b2) / 2.210526e-06) * (1 + math.erf(((4.696862e-06 - Q.e3_b2) / 2.210526e-06) / math.sqrt(2))))
    z += 0.04016802 * (8.963785 * 0.5 * ((22.56857 - Q.mass_charged) / 8.963785) * (1 + math.erf(((22.56857 - Q.mass_charged) / 8.963785) / math.sqrt(2))))
    z += -1.23471 * (0.02343699 * 0.5 * ((0.2277628 - Q.sdb_2_z) / 0.02343699) * (1 + math.erf(((0.2277628 - Q.sdb_2_z) / 0.02343699) / math.sqrt(2))))
    z += 0.2569468 * (0.0795439 * 0.5 * ((Q.ak02_dr23 - 0.5718354) / 0.0795439) * (1 + math.erf(((Q.ak02_dr23 - 0.5718354) / 0.0795439) / math.sqrt(2))))
    z += 0.004735879 * (1.0 * 0.5 * ((Q.sjq_3_2_nch - 5.0) / 1.0) * (1 + math.erf(((Q.sjq_3_2_nch - 5.0) / 1.0) / math.sqrt(2))))
    z += -0.001495437 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2))))
    z += -0.3043497 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.04373169 * 0.5 * ((Q.eta_14 - 0.1429443) / 0.04373169) * (1 + math.erf(((Q.eta_14 - 0.1429443) / 0.04373169) / math.sqrt(2))))
    z += -25.24171 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2))))
    z += -0.003742828 * (0.560057 * 0.5 * ((Q.mass_displaced5 - 1.777787) / 0.560057) * (1 + math.erf(((Q.mass_displaced5 - 1.777787) / 0.560057) / math.sqrt(2))))
    z += 0.0007990236 * (0.560057 * 0.5 * ((Q.mass_displaced5 - 1.777787) / 0.560057) * (1 + math.erf(((Q.mass_displaced5 - 1.777787) / 0.560057) / math.sqrt(2)))) * (5.37501 * 0.5 * ((9.356714 - Q.jd_3d_6) / 5.37501) * (1 + math.erf(((9.356714 - Q.jd_3d_6) / 5.37501) / math.sqrt(2))))
    z += 0.09075799 * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 1.0) / 1.0) / math.sqrt(2))))
    z += -0.000413553 * (5.143623 * 0.5 * ((Q.mres_sd_mass_b1z01 - 88.79082) / 5.143623) * (1 + math.erf(((Q.mres_sd_mass_b1z01 - 88.79082) / 5.143623) / math.sqrt(2))))
    z += -0.01320292 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2))))
    z += -0.001115416 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (1.387908 * 0.5 * ((3.151933 - Q.jd_3d_6) / 1.387908) * (1 + math.erf(((3.151933 - Q.jd_3d_6) / 1.387908) / math.sqrt(2))))
    z += 0.03297888 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2)))) * (0.04060641 * 0.5 * ((0.1322296 - Q.dr_39) / 0.04060641) * (1 + math.erf(((0.1322296 - Q.dr_39) / 0.04060641) / math.sqrt(2))))
    z += 4.632918e-05 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2)))) * (0.2399741 * 0.5 * ((-4.934509 - Q.lund3_lnz) / 0.2399741) * (1 + math.erf(((-4.934509 - Q.lund3_lnz) / 0.2399741) / math.sqrt(2))))
    z += -0.5656628 * (0.179073 * 0.5 * ((Q.lep_z - 0.5187302) / 0.179073) * (1 + math.erf(((Q.lep_z - 0.5187302) / 0.179073) / math.sqrt(2))))
    z += 0.02660007 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.01565591 * 0.5 * ((0.1681753 - Q.pz_lnkt1) / 0.01565591) * (1 + math.erf(((0.1681753 - Q.pz_lnkt1) / 0.01565591) / math.sqrt(2))))
    z += 2.918607 * (0.179073 * 0.5 * ((Q.lep_z - 0.5187302) / 0.179073) * (1 + math.erf(((Q.lep_z - 0.5187302) / 0.179073) / math.sqrt(2)))) * (0.7363527 * 0.5 * ((0.7363527 - Q.dc_4_jp) / 0.7363527) * (1 + math.erf(((0.7363527 - Q.dc_4_jp) / 0.7363527) / math.sqrt(2))))
    z += 0.02489502 * (0.5664529 * 0.5 * ((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) * (1 + math.erf(((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) / math.sqrt(2))))
    z += 2.892457 * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2))))
    z += -0.9015101 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2))))
    z += 1.303416 * (0.2398529 * 0.5 * ((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) * (1 + math.erf(((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) / math.sqrt(2))))
    z += -0.743859 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) / math.sqrt(2))))
    z += 0.3691721 * (0.01961104 * 0.5 * ((0.3273298 - Q.sj3_dr12) / 0.01961104) * (1 + math.erf(((0.3273298 - Q.sj3_dr12) / 0.01961104) / math.sqrt(2))))
    z += 1.099969 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2))))
    z += -0.1447751 * (1.0 * 0.5 * ((Q.n_electron - 1.0) / 1.0) * (1 + math.erf(((Q.n_electron - 1.0) / 1.0) / math.sqrt(2))))
    z += 2.420688 * (0.004103638 * 0.5 * ((Q.M2 - 0.06228948) / 0.004103638) * (1 + math.erf(((Q.M2 - 0.06228948) / 0.004103638) / math.sqrt(2))))
    z += 0.9280766 * (0.2508028 * 0.5 * ((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2))))
    z += -1.22362 * (0.04964267 * 0.5 * ((Q.sj4_dr_min - 0.249115) / 0.04964267) * (1 + math.erf(((Q.sj4_dr_min - 0.249115) / 0.04964267) / math.sqrt(2))))
    z += -5494.25 * (2.44598e-06 * 0.5 * ((1.874473e-05 - Q.ecf_g42) / 2.44598e-06) * (1 + math.erf(((1.874473e-05 - Q.ecf_g42) / 2.44598e-06) / math.sqrt(2))))
    return z


def neuron_71(Q):
    z = 3.029707e-06
    return z


def neuron_72(Q):
    z = -1.164118e-06
    return z


def neuron_73(Q):
    z = -1.699639e-05
    return z


def neuron_74(Q):
    z = -4.493394e-06
    return z


def neuron_75(Q):
    z = 1.973839e-06
    return z


def neuron_76(Q):
    z = -5.542785e-06
    return z


def neuron_77(Q):
    z = -3.087487e-06
    return z


def neuron_78(Q):
    z = -0.08632727
    z += -1.074317 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2))))
    z += -0.03833139 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2))))
    z += 0.6394216 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.01443111 * 0.5 * ((0.1956014 - Q.pz_lnd2) / 0.01443111) * (1 + math.erf(((0.1956014 - Q.pz_lnd2) / 0.01443111) / math.sqrt(2))))
    z += 0.00915835 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (3.260893 * 0.5 * ((Q.mass_top20 - 86.78877) / 3.260893) * (1 + math.erf(((Q.mass_top20 - 86.78877) / 3.260893) / math.sqrt(2))))
    z += 0.002114115 * (27.0 * 0.5 * ((195.0 - Q.n_pairs_kt_above_1) / 27.0) * (1 + math.erf(((195.0 - Q.n_pairs_kt_above_1) / 27.0) / math.sqrt(2))))
    z += -4.588196 * (0.001268491 * 0.5 * ((0.0142807 - Q.M3_b2) / 0.001268491) * (1 + math.erf(((0.0142807 - Q.M3_b2) / 0.001268491) / math.sqrt(2))))
    z += 0.006297867 * (3.793098 * 0.5 * ((Q.mass_displaced5 - 9.380468) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 9.380468) / 3.793098) / math.sqrt(2))))
    z += 0.01298828 * (17.59858 * 0.5 * ((Q.mass_top50 - 178.725) / 17.59858) * (1 + math.erf(((Q.mass_top50 - 178.725) / 17.59858) / math.sqrt(2))))
    z += -0.0002280919 * (3.793098 * 0.5 * ((Q.mass_displaced5 - 9.380468) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 9.380468) / 3.793098) / math.sqrt(2)))) * (14.29796 * 0.5 * ((113.5019 - Q.mass_charged) / 14.29796) * (1 + math.erf(((113.5019 - Q.mass_charged) / 14.29796) / math.sqrt(2))))
    z += 0.004771224 * (3.02693 * 0.5 * ((Q.sj4_pair_mass_max - 78.76797) / 3.02693) * (1 + math.erf(((Q.sj4_pair_mass_max - 78.76797) / 3.02693) / math.sqrt(2))))
    z += -0.07131069 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2))))
    z += 0.006782373 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.3051516 * 0.5 * ((-7.886954 - Q.ktd_ln_d34) / 0.3051516) * (1 + math.erf(((-7.886954 - Q.ktd_ln_d34) / 0.3051516) / math.sqrt(2))))
    z += 0.03620294 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2))))
    z += -0.06115105 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (0.02818185 * 0.5 * ((Q.sj3_pairmin_over_m - 0.1383534) / 0.02818185) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.1383534) / 0.02818185) / math.sqrt(2))))
    z += 0.0001043035 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (405.0217 * 0.5 * ((803.1736 - Q.jd_sum_abs_sd0_top5) / 405.0217) * (1 + math.erf(((803.1736 - Q.jd_sum_abs_sd0_top5) / 405.0217) / math.sqrt(2))))
    z += -0.005725868 * (11.28806 * 0.5 * ((70.88236 - Q.mass_top40) / 11.28806) * (1 + math.erf(((70.88236 - Q.mass_top40) / 11.28806) / math.sqrt(2))))
    z += 0.007381145 * (1.5 * 0.5 * ((22.0 - Q.n_neutral) / 1.5) * (1 + math.erf(((22.0 - Q.n_neutral) / 1.5) / math.sqrt(2))))
    z += -0.3066562 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2))))
    z += -0.2461202 * (0.0033606 * 0.5 * ((Q.M2_b05 - 0.1409711) / 0.0033606) * (1 + math.erf(((Q.M2_b05 - 0.1409711) / 0.0033606) / math.sqrt(2))))
    z += 1.486693 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) * (0.2429919 * 0.5 * ((6.304449 - Q.lne_0) / 0.2429919) * (1 + math.erf(((6.304449 - Q.lne_0) / 0.2429919) / math.sqrt(2))))
    z += -0.02750477 * (1.0 * 0.5 * ((10.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((10.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2))))
    z += 0.001275031 * (3.793098 * 0.5 * ((Q.mass_displaced5 - 9.380468) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 9.380468) / 3.793098) / math.sqrt(2)))) * (6.000774 * 0.5 * ((14.85973 - Q.jd_3d_4) / 6.000774) * (1 + math.erf(((14.85973 - Q.jd_3d_4) / 6.000774) / math.sqrt(2))))
    z += -0.0001766284 * (540.2081 * 0.5 * ((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2))))
    z += -0.1113415 * (0.04148383 * 0.5 * ((-0.4705779 - Q.lund1_lndelta) / 0.04148383) * (1 + math.erf(((-0.4705779 - Q.lund1_lndelta) / 0.04148383) / math.sqrt(2))))
    z += 0.0009978276 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (1.0 * 0.5 * ((4.0 - Q.ak02_n) / 1.0) * (1 + math.erf(((4.0 - Q.ak02_n) / 1.0) / math.sqrt(2))))
    z += -6.562011e-05 * (3.858926 * 0.5 * ((Q.mres_sd_mass_b0z005 - 111.9362) / 3.858926) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 111.9362) / 3.858926) / math.sqrt(2))))
    z += -0.01856347 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) / math.sqrt(2))))
    z += 0.005544708 * (10.67504 * 0.5 * ((Q.mass_top10 - 103.4976) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 103.4976) / 10.67504) / math.sqrt(2))))
    z += -0.01998266 * (1.0 * 0.5 * ((Q.n_sd0_above_5 - 5.0) / 1.0) * (1 + math.erf(((Q.n_sd0_above_5 - 5.0) / 1.0) / math.sqrt(2))))
    z += 2.242915 * (0.009441413 * 0.5 * ((0.4854269 - Q.N2_b05) / 0.009441413) * (1 + math.erf(((0.4854269 - Q.N2_b05) / 0.009441413) / math.sqrt(2))))
    z += 0.01180323 * (4.488781 * 0.5 * ((Q.mres_sd_mass_b0z005 - 125.8718) / 4.488781) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 125.8718) / 4.488781) / math.sqrt(2))))
    z += 0.03308736 * (1.5 * 0.5 * ((Q.n_charged_had - 23.0) / 1.5) * (1 + math.erf(((Q.n_charged_had - 23.0) / 1.5) / math.sqrt(2))))
    z += 0.1402743 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.04808774 * 0.5 * ((0.09790963 - Q.lepsj_3_dr) / 0.04808774) * (1 + math.erf(((0.09790963 - Q.lepsj_3_dr) / 0.04808774) / math.sqrt(2))))
    z += -0.6031383 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2))))
    z += 2.608102 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2))))
    z += -0.005839752 * (10.09897 * 0.5 * ((Q.sj4_pair_mass_max - 115.5142) / 10.09897) * (1 + math.erf(((Q.sj4_pair_mass_max - 115.5142) / 10.09897) / math.sqrt(2))))
    z += 0.002946399 * (11.5 * 0.5 * ((99.0 - Q.n_pairs_kt_above_3) / 11.5) * (1 + math.erf(((99.0 - Q.n_pairs_kt_above_3) / 11.5) / math.sqrt(2))))
    z += -0.003761246 * (3.357558 * 0.5 * ((Q.max_pair_mass - 31.44492) / 3.357558) * (1 + math.erf(((Q.max_pair_mass - 31.44492) / 3.357558) / math.sqrt(2))))
    z += -0.1073622 * (0.2260337 * 0.5 * ((-8.400697 - Q.ktd_ln_d34) / 0.2260337) * (1 + math.erf(((-8.400697 - Q.ktd_ln_d34) / 0.2260337) / math.sqrt(2))))
    z += 66795.02 * (2.54381e-06 * 0.5 * ((3.53193e-06 - Q.e4) / 2.54381e-06) * (1 + math.erf(((3.53193e-06 - Q.e4) / 2.54381e-06) / math.sqrt(2))))
    z += -0.005698686 * (16.90428 * 0.5 * ((Q.mass - 164.4374) / 16.90428) * (1 + math.erf(((Q.mass - 164.4374) / 16.90428) / math.sqrt(2))))
    z += 0.2290503 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) / math.sqrt(2))))
    z += 19.11551 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) * (0.001358032 * 0.5 * ((0.03179932 - Q.dzerr_0) / 0.001358032) * (1 + math.erf(((0.03179932 - Q.dzerr_0) / 0.001358032) / math.sqrt(2))))
    z += 0.09535559 * (0.005026783 * 0.5 * ((0.09704114 - Q.pz_lnkt1) / 0.005026783) * (1 + math.erf(((0.09704114 - Q.pz_lnkt1) / 0.005026783) / math.sqrt(2))))
    z += 0.08377441 * (0.03825143 * 0.5 * ((0.06059953 - Q.sjq_2_prod_k03) / 0.03825143) * (1 + math.erf(((0.06059953 - Q.sjq_2_prod_k03) / 0.03825143) / math.sqrt(2))))
    z += 1.996871e-05 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2))))
    z += 0.005576469 * (2.0 * 0.5 * ((10.0 - Q.n_sdz_above_2) / 2.0) * (1 + math.erf(((10.0 - Q.n_sdz_above_2) / 2.0) / math.sqrt(2))))
    z += -0.01131747 * (2.0 * 0.5 * ((10.0 - Q.n_sdz_above_2) / 2.0) * (1 + math.erf(((10.0 - Q.n_sdz_above_2) / 2.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.02953399 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2))))
    z += 0.001514613 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_electron - 1.0) / 1.0) * (1 + math.erf(((Q.n_electron - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.007835716 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2))))
    z += -0.001327362 * (3.373048 * 0.5 * ((Q.mass_top15 - 83.68425) / 3.373048) * (1 + math.erf(((Q.mass_top15 - 83.68425) / 3.373048) / math.sqrt(2))))
    z += -0.01653939 * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2))))
    z += 1.025126 * (0.04874922 * 0.5 * ((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) * (1 + math.erf(((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) / math.sqrt(2))))
    z += -2.002671e-06 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (98.63368 * 0.5 * ((Q.jet_e - 1134.686) / 98.63368) * (1 + math.erf(((Q.jet_e - 1134.686) / 98.63368) / math.sqrt(2))))
    z += 0.05952755 * (2.0 * 0.5 * ((10.0 - Q.n_sdz_above_2) / 2.0) * (1 + math.erf(((10.0 - Q.n_sdz_above_2) / 2.0) / math.sqrt(2)))) * (0.06215671 * 0.5 * ((0.3112717 - Q.sj4_dr_min) / 0.06215671) * (1 + math.erf(((0.3112717 - Q.sj4_dr_min) / 0.06215671) / math.sqrt(2))))
    z += 4.497916e-05 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (16.61023 * 0.5 * ((607.8256 - Q.sum_pt_top30) / 16.61023) * (1 + math.erf(((607.8256 - Q.sum_pt_top30) / 16.61023) / math.sqrt(2))))
    z += -0.09880331 * (0.06660296 * 0.5 * ((-2.115543 - Q.pair_mean_lndelta) / 0.06660296) * (1 + math.erf(((-2.115543 - Q.pair_mean_lndelta) / 0.06660296) / math.sqrt(2))))
    z += 0.07473116 * (0.3893833 * 0.5 * ((Q.lund3_lndelta - -2.817283) / 0.3893833) * (1 + math.erf(((Q.lund3_lndelta - -2.817283) / 0.3893833) / math.sqrt(2))))
    z += 0.08628151 * (2.0 * 0.5 * ((Q.sjf_2_1_n_d3 - 7.0) / 2.0) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 7.0) / 2.0) / math.sqrt(2))))
    z += 0.03986159 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (0.05461387 * 0.5 * ((0.2184442 - Q.z_displaced5) / 0.05461387) * (1 + math.erf(((0.2184442 - Q.z_displaced5) / 0.05461387) / math.sqrt(2))))
    z += 0.008685812 * (12.30925 * 0.5 * ((Q.mass_displaced3 - 39.09615) / 12.30925) * (1 + math.erf(((Q.mass_displaced3 - 39.09615) / 12.30925) / math.sqrt(2))))
    z += -0.01647587 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 0.03211702 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.05549181 * 0.5 * ((0.1570831 - Q.sjf_4_2_z_d3) / 0.05549181) * (1 + math.erf(((0.1570831 - Q.sjf_4_2_z_d3) / 0.05549181) / math.sqrt(2))))
    z += -0.004145301 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_4_3_n_d3 - 0.0) / 1.0) * (1 + math.erf(((Q.sjf_4_3_n_d3 - 0.0) / 1.0) / math.sqrt(2))))
    z += 0.007722734 * (12.30925 * 0.5 * ((Q.mass_displaced3 - 39.09615) / 12.30925) * (1 + math.erf(((Q.mass_displaced3 - 39.09615) / 12.30925) / math.sqrt(2)))) * (0.006172711 * 0.5 * ((Q.lepsj_3_dr - 0.01012269) / 0.006172711) * (1 + math.erf(((Q.lepsj_3_dr - 0.01012269) / 0.006172711) / math.sqrt(2))))
    z += 0.003515731 * (18.98174 * 0.5 * ((14.38858 - Q.lep_iso) / 18.98174) * (1 + math.erf(((14.38858 - Q.lep_iso) / 18.98174) / math.sqrt(2))))
    z += -0.007595869 * (5.197 * 0.5 * ((95.14961 - Q.mass) / 5.197) * (1 + math.erf(((95.14961 - Q.mass) / 5.197) / math.sqrt(2))))
    z += 0.08317363 * (1.0 * 0.5 * ((6.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((6.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 0.05204923 * (0.560057 * 0.5 * ((1.777787 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.777787 - Q.mass_displaced5) / 0.560057) / math.sqrt(2))))
    z += 0.0003226118 * (164.9991 * 0.5 * ((226.3008 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((226.3008 - Q.sip_3d_2) / 164.9991) / math.sqrt(2))))
    z += -0.2163122 * (0.06585074 * 0.5 * ((-0.1170754 - Q.tdz_1) / 0.06585074) * (1 + math.erf(((-0.1170754 - Q.tdz_1) / 0.06585074) / math.sqrt(2))))
    return z


def neuron_79(Q):
    z = -3.602387e-05
    return z


def neuron_80(Q):
    z = 9.707701e-06
    return z


def neuron_81(Q):
    z = 0.5203255
    z += -0.08516282 * Q.sjq_2_prod_k05
    z += -0.08176374 * Q.sv_n
    z += 3.055491 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2))))
    z += -0.08341973 * (0.01683601 * 0.5 * ((0.1339824 - Q.pz_lnd2) / 0.01683601) * (1 + math.erf(((0.1339824 - Q.pz_lnd2) / 0.01683601) / math.sqrt(2))))
    z += -0.1346059 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (1.5 * 0.5 * ((7.0 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.0 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2))))
    z += -0.007053514 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2))))
    z += 0.005800622 * (3.360928 * 0.5 * ((Q.mass_top40 - 115.7429) / 3.360928) * (1 + math.erf(((Q.mass_top40 - 115.7429) / 3.360928) / math.sqrt(2))))
    z += 0.036089 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2))))
    z += 4.213109e-07 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (275.2718 * 0.5 * ((578.2954 - Q.sip_3d_1) / 275.2718) * (1 + math.erf(((578.2954 - Q.sip_3d_1) / 275.2718) / math.sqrt(2))))
    z += -0.02947322 * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2))))
    z += -5.884685 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.265564) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.265564) / 0.0649922) / math.sqrt(2))))
    z += -0.005671569 * (17.59858 * 0.5 * ((Q.mass_top50 - 178.725) / 17.59858) * (1 + math.erf(((Q.mass_top50 - 178.725) / 17.59858) / math.sqrt(2))))
    z += 1.117097 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (0.006796583 * 0.5 * ((0.05966366 - Q.M2_b2) / 0.006796583) * (1 + math.erf(((0.05966366 - Q.M2_b2) / 0.006796583) / math.sqrt(2))))
    z += 0.03065538 * (3.499256 * 0.5 * ((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) * (1 + math.erf(((121.6067 - Q.mres_sd_mass_b2z01) / 3.499256) / math.sqrt(2))))
    z += 0.004186831 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += 0.01625036 * (5.55459 * 0.5 * ((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2))))
    z += -0.01508409 * (19.14168 * 0.5 * ((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2))))
    z += 0.000471641 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2))))
    z += 3.658361 * (0.04445521 * 0.5 * ((0.2686235 - Q.LHA) / 0.04445521) * (1 + math.erf(((0.2686235 - Q.LHA) / 0.04445521) / math.sqrt(2))))
    z += -0.0184541 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2))))
    z += -0.3933927 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2))))
    z += -0.4106912 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.2719952 * (0.01706594 * 0.5 * ((0.7339085 - Q.max_dr) / 0.01706594) * (1 + math.erf(((0.7339085 - Q.max_dr) / 0.01706594) / math.sqrt(2))))
    z += -0.4685474 * (1.0 * 0.5 * ((0.0 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.0 - Q.sv_2_n) / 1.0) / math.sqrt(2))))
    z += -6.243398e-06 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (8.422628 * 0.5 * ((23.30856 - Q.sip_3d_3) / 8.422628) * (1 + math.erf(((23.30856 - Q.sip_3d_3) / 8.422628) / math.sqrt(2))))
    z += -0.000343613 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.ak02_min12_n_disp3) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_min12_n_disp3) / 1.0) / math.sqrt(2))))
    z += -0.9412572 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2))))
    z += -0.1411797 * (0.2381965 * 0.5 * ((Q.lne_5 - 2.737811) / 0.2381965) * (1 + math.erf(((Q.lne_5 - 2.737811) / 0.2381965) / math.sqrt(2))))
    z += 0.0001433427 * (19.14168 * 0.5 * ((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2)))) * (5.386244 * 0.5 * ((Q.mass_2charged - 28.89659) / 5.386244) * (1 + math.erf(((Q.mass_2charged - 28.89659) / 5.386244) / math.sqrt(2))))
    z += 0.05651173 * (1.0 * 0.5 * ((Q.sjf_4_1_n_d3 - 3.0) / 1.0) * (1 + math.erf(((Q.sjf_4_1_n_d3 - 3.0) / 1.0) / math.sqrt(2))))
    z += 0.08192026 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2))))
    z += -0.003206593 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (6.084951 * 0.5 * ((16.23838 - Q.sip_3d_3) / 6.084951) * (1 + math.erf(((16.23838 - Q.sip_3d_3) / 6.084951) / math.sqrt(2))))
    z += 0.1368784 * (1.0 * 0.5 * ((6.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((6.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 8.210237e-06 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (401.9953 * 0.5 * ((577.991 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((577.991 - Q.sip_3d_3) / 401.9953) / math.sqrt(2))))
    z += -0.007768441 * (16.26099 * 0.5 * ((Q.mass_top50 - 161.1264) / 16.26099) * (1 + math.erf(((Q.mass_top50 - 161.1264) / 16.26099) / math.sqrt(2))))
    z += -0.009481247 * (10.51039 * 0.5 * ((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) * (1 + math.erf(((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) / math.sqrt(2))))
    z += -0.04152096 * (5.26174 * 0.5 * ((91.15481 - Q.mres_sd_mass_b2z01) / 5.26174) * (1 + math.erf(((91.15481 - Q.mres_sd_mass_b2z01) / 5.26174) / math.sqrt(2))))
    z += 0.7975627 * (0.02426199 * 0.5 * ((0.2040425 - Q.sdb_2_z) / 0.02426199) * (1 + math.erf(((0.2040425 - Q.sdb_2_z) / 0.02426199) / math.sqrt(2))))
    z += -0.9144809 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (0.02572129 * 0.5 * ((0.9433644 - Q.tau43_b2) / 0.02572129) * (1 + math.erf(((0.9433644 - Q.tau43_b2) / 0.02572129) / math.sqrt(2))))
    z += 0.007364273 * (19.14168 * 0.5 * ((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.2019 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2)))) * (0.0467133 * 0.5 * ((0.5068038 - Q.tau32) / 0.0467133) * (1 + math.erf(((0.5068038 - Q.tau32) / 0.0467133) / math.sqrt(2))))
    z += 1.797396 * (0.02426199 * 0.5 * ((0.2040425 - Q.sdb_2_z) / 0.02426199) * (1 + math.erf(((0.2040425 - Q.sdb_2_z) / 0.02426199) / math.sqrt(2)))) * (0.01654934 * 0.5 * ((Q.tau54 - 0.8015743) / 0.01654934) * (1 + math.erf(((Q.tau54 - 0.8015743) / 0.01654934) / math.sqrt(2))))
    z += 0.348461 * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2)))) * (0.05285263 * 0.5 * ((0.1270192 - Q.sdb_4_z) / 0.05285263) * (1 + math.erf(((0.1270192 - Q.sdb_4_z) / 0.05285263) / math.sqrt(2))))
    z += -0.244834 * (0.02426199 * 0.5 * ((0.2040425 - Q.sdb_2_z) / 0.02426199) * (1 + math.erf(((0.2040425 - Q.sdb_2_z) / 0.02426199) / math.sqrt(2)))) * (5.085197 * 0.5 * ((10.7744 - Q.jd_3d_5) / 5.085197) * (1 + math.erf(((10.7744 - Q.jd_3d_5) / 5.085197) / math.sqrt(2))))
    z += -0.00214146 * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 5.0) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 5.0) / 1.0) / math.sqrt(2))))
    z += 0.3371254 * (0.06585074 * 0.5 * ((-0.1170754 - Q.tdz_1) / 0.06585074) * (1 + math.erf(((-0.1170754 - Q.tdz_1) / 0.06585074) / math.sqrt(2))))
    z += 0.02143688 * (6.036668 * 0.5 * ((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) * (1 + math.erf(((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) / math.sqrt(2))))
    z += -0.2054692 * (0.5788858 * 0.5 * ((1.777286 - Q.mass_displaced3) / 0.5788858) * (1 + math.erf(((1.777286 - Q.mass_displaced3) / 0.5788858) / math.sqrt(2))))
    z += 0.06443197 * (1.0 * 0.5 * ((3.0 - Q.sdb_0_n) / 1.0) * (1 + math.erf(((3.0 - Q.sdb_0_n) / 1.0) / math.sqrt(2))))
    z += -0.06836937 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 0.0004729791 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (10.60708 * 0.5 * ((16.11752 - Q.jd_3d_6) / 10.60708) * (1 + math.erf(((16.11752 - Q.jd_3d_6) / 10.60708) / math.sqrt(2))))
    z += -0.0002059547 * (14.45336 * 0.5 * ((Q.jd_3d_6 - 30.57088) / 14.45336) * (1 + math.erf(((Q.jd_3d_6 - 30.57088) / 14.45336) / math.sqrt(2))))
    z += 0.1024562 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 3.0) / 1.0) / math.sqrt(2))))
    z += 0.04098999 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2))))
    z += 0.05118544 * (0.00837824 * 0.5 * ((0.2743483 - Q.C2_b05) / 0.00837824) * (1 + math.erf(((0.2743483 - Q.C2_b05) / 0.00837824) / math.sqrt(2))))
    z += 0.6434139 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) * (0.006013287 * 0.5 * ((0.06661369 - Q.pz_lnkt3) / 0.006013287) * (1 + math.erf(((0.06661369 - Q.pz_lnkt3) / 0.006013287) / math.sqrt(2))))
    z += -0.1688651 * (0.1799842 * 0.5 * ((2.044945 - Q.sjf_2_2_max3d) / 0.1799842) * (1 + math.erf(((2.044945 - Q.sjf_2_2_max3d) / 0.1799842) / math.sqrt(2))))
    z += 0.4958655 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ismuon_3 - 0.0) / 1e-06) * (1 + math.erf(((Q.ismuon_3 - 0.0) / 1e-06) / math.sqrt(2))))
    z += -0.0005545563 * (5.02206 * 0.5 * ((Q.mass_top50 - 129.5874) / 5.02206) * (1 + math.erf(((Q.mass_top50 - 129.5874) / 5.02206) / math.sqrt(2))))
    z += -0.01297186 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.001692006 * 0.5 * ((Q.sjf_4_3_z_d3 - 0.0) / 0.001692006) * (1 + math.erf(((Q.sjf_4_3_z_d3 - 0.0) / 0.001692006) / math.sqrt(2))))
    z += -124.7523 * (0.005213255 * 0.5 * ((0.0 - Q.z_displaced3) / 0.005213255) * (1 + math.erf(((0.0 - Q.z_displaced3) / 0.005213255) / math.sqrt(2))))
    z += -0.0007916088 * (76.66736 * 0.5 * ((164.4318 - Q.jd_sum_abs_sd0_top3) / 76.66736) * (1 + math.erf(((164.4318 - Q.jd_sum_abs_sd0_top3) / 76.66736) / math.sqrt(2))))
    z += -0.6884649 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (0.01789856 * 0.5 * ((Q.eta_8 - 0.04946899) / 0.01789856) * (1 + math.erf(((Q.eta_8 - 0.04946899) / 0.01789856) / math.sqrt(2))))
    z += 0.002229771 * (4.193136 * 0.5 * ((6.64573 - Q.sjf_2_1_mass_d3) / 4.193136) * (1 + math.erf(((6.64573 - Q.sjf_2_1_mass_d3) / 4.193136) / math.sqrt(2))))
    z += -0.00361694 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.isnhad_34 - 0.0) / 1e-06) * (1 + math.erf(((Q.isnhad_34 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.8776981 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.02367609 * 0.5 * ((0.7382159 - Q.tau43) / 0.02367609) * (1 + math.erf(((0.7382159 - Q.tau43) / 0.02367609) / math.sqrt(2))))
    z += -1.437606 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (0.0964495 * 0.5 * ((0.5557674 - Q.sjq_2_sumabs_k1) / 0.0964495) * (1 + math.erf(((0.5557674 - Q.sjq_2_sumabs_k1) / 0.0964495) / math.sqrt(2))))
    z += -0.1133244 * (1.0 * 0.5 * ((1.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((1.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2))))
    z += -0.01344186 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 3.0) / 1.0) / math.sqrt(2)))) * (2.0 * 0.5 * ((4.0 - Q.lepsj_2_n_d3) / 2.0) * (1 + math.erf(((4.0 - Q.lepsj_2_n_d3) / 2.0) / math.sqrt(2))))
    z += 0.6792876 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2))))
    z += -0.02594847 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_electron - 1.0) / 1.0) * (1 + math.erf(((Q.n_electron - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.001835415 * (6.0 * 0.5 * ((28.0 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.0 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) * (0.05748496 * 0.5 * ((0.2338497 - Q.lep_dr) / 0.05748496) * (1 + math.erf(((0.2338497 - Q.lep_dr) / 0.05748496) / math.sqrt(2))))
    z += -3.419866e-05 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_charged_pt_above_10 - 3.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_10 - 3.0) / 1.0) / math.sqrt(2))))
    z += 6.716493 * (0.009443246 * 0.5 * ((0.0176083 - Q.lepsj_3_dr) / 0.009443246) * (1 + math.erf(((0.0176083 - Q.lepsj_3_dr) / 0.009443246) / math.sqrt(2))))
    z += -0.004822346 * (29.76053 * 0.5 * ((44.14912 - Q.lep_iso) / 29.76053) * (1 + math.erf(((44.14912 - Q.lep_iso) / 29.76053) / math.sqrt(2))))
    z += 0.1480749 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.2370407 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.2370407 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2))))
    z += 0.002421398 * (11.52979 * 0.5 * ((Q.mass_charged - 99.20396) / 11.52979) * (1 + math.erf(((Q.mass_charged - 99.20396) / 11.52979) / math.sqrt(2))))
    z += 0.1925198 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.08090366) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.08090366) / 0.02108391) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.td0_50 - 0.0) / 1e-06) * (1 + math.erf(((Q.td0_50 - 0.0) / 1e-06) / math.sqrt(2))))
    z += -2.207594 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.2297361 * 0.5 * ((Q.ak02_dr13 - 0.0) / 0.2297361) * (1 + math.erf(((Q.ak02_dr13 - 0.0) / 0.2297361) / math.sqrt(2))))
    z += -0.036425 * (5.55459 * 0.5 * ((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.62031 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) * (0.01467775 * 0.5 * ((0.2243273 - Q.N2_b2) / 0.01467775) * (1 + math.erf(((0.2243273 - Q.N2_b2) / 0.01467775) / math.sqrt(2))))
    z += 1.316897 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.03624058) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.03624058) / 0.01141967) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dr_68 - 0.0) / 1e-06) * (1 + math.erf(((Q.dr_68 - 0.0) / 1e-06) / math.sqrt(2))))
    z += -0.0002328412 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2)))) * (17.93691 * 0.5 * ((28.56971 - Q.dc_1_sd0_1) / 17.93691) * (1 + math.erf(((28.56971 - Q.dc_1_sd0_1) / 17.93691) / math.sqrt(2))))
    z += -0.06266686 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) * (0.01480848 * 0.5 * ((Q.nca_sj4_pair2nd_over_mass - 0.4488456) / 0.01480848) * (1 + math.erf(((Q.nca_sj4_pair2nd_over_mass - 0.4488456) / 0.01480848) / math.sqrt(2))))
    z += -0.0625078 * (0.2210883 * 0.5 * ((1.514826 - Q.min_pair_mass) / 0.2210883) * (1 + math.erf(((1.514826 - Q.min_pair_mass) / 0.2210883) / math.sqrt(2))))
    z += 0.0006171324 * (0.2210883 * 0.5 * ((1.514826 - Q.min_pair_mass) / 0.2210883) * (1 + math.erf(((1.514826 - Q.min_pair_mass) / 0.2210883) / math.sqrt(2)))) * (9.707626 * 0.5 * ((16.47324 - Q.dc_1_sd0_1) / 9.707626) * (1 + math.erf(((16.47324 - Q.dc_1_sd0_1) / 9.707626) / math.sqrt(2))))
    z += 0.0002467969 * (0.06585074 * 0.5 * ((-0.1170754 - Q.tdz_1) / 0.06585074) * (1 + math.erf(((-0.1170754 - Q.tdz_1) / 0.06585074) / math.sqrt(2)))) * (5.901273 * 0.5 * ((Q.kt2_1_sd0_3 - 4.457634) / 5.901273) * (1 + math.erf(((Q.kt2_1_sd0_3 - 4.457634) / 5.901273) / math.sqrt(2))))
    z += 0.0005320364 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.nca_kt_above_2 - 4.0) / 1.0) * (1 + math.erf(((Q.nca_kt_above_2 - 4.0) / 1.0) / math.sqrt(2))))
    z += -0.02467596 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2))))
    z += 1.789079 * (0.00773283 * 0.5 * ((0.07991731 - Q.tau2) / 0.00773283) * (1 + math.erf(((0.07991731 - Q.tau2) / 0.00773283) / math.sqrt(2))))
    z += -0.006871738 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.0) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.isnhad_8 - 1.0) / 1.0) * (1 + math.erf(((Q.isnhad_8 - 1.0) / 1.0) / math.sqrt(2))))
    z += 5.933638 * (0.01154615 * 0.5 * ((0.01545532 - Q.psi_0p1) / 0.01154615) * (1 + math.erf(((0.01545532 - Q.psi_0p1) / 0.01154615) / math.sqrt(2))))
    z += -0.002970973 * (3.579676 * 0.5 * ((63.34656 - Q.sj4_pair_mass_max) / 3.579676) * (1 + math.erf(((63.34656 - Q.sj4_pair_mass_max) / 3.579676) / math.sqrt(2))))
    z += 0.4311019 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2))))
    z += -0.00853133 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2))))
    z += -0.002758332 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.006595008 * (4.106456 * 0.5 * ((5.245219 - Q.sjf_3_2_max3d) / 4.106456) * (1 + math.erf(((5.245219 - Q.sjf_3_2_max3d) / 4.106456) / math.sqrt(2))))
    z += 0.002669359 * (11.4891 * 0.5 * ((32.61679 - Q.mass_displaced5) / 11.4891) * (1 + math.erf(((32.61679 - Q.mass_displaced5) / 11.4891) / math.sqrt(2))))
    z += 0.1366937 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 3.0) / 1.0) / math.sqrt(2)))) * (0.06646541 * 0.5 * ((0.3206143 - Q.pt1_over_pt0) / 0.06646541) * (1 + math.erf(((0.3206143 - Q.pt1_over_pt0) / 0.06646541) / math.sqrt(2))))
    z += -0.1126606 * (1.0 * 0.5 * ((Q.sjf_4_n2disp - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_4_n2disp - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.3244356 * (0.008567977 * 0.5 * ((0.4920044 - Q.pt_balance01) / 0.008567977) * (1 + math.erf(((0.4920044 - Q.pt_balance01) / 0.008567977) / math.sqrt(2))))
    return z


def neuron_82(Q):
    z = -2.942071e-06
    return z


def neuron_83(Q):
    z = -0.7497348
    z += 0.7491826 * Q.pz_lnd2
    z += 0.0008204536 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.33213) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.33213) / 6.345399) / math.sqrt(2))))
    z += 0.02077761 * (15.58693 * 0.5 * ((Q.mass_top40 - 155.8928) / 15.58693) * (1 + math.erf(((Q.mass_top40 - 155.8928) / 15.58693) / math.sqrt(2))))
    z += -0.02329017 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.0) / 1.0) / math.sqrt(2))))
    z += 5.247586 * (0.02099671 * 0.5 * ((0.08304558 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.08304558 - Q.z_displaced3) / 0.02099671) / math.sqrt(2))))
    z += 0.002004721 * (16.47782 * 0.5 * ((162.7874 - Q.mass_top30) / 16.47782) * (1 + math.erf(((162.7874 - Q.mass_top30) / 16.47782) / math.sqrt(2))))
    z += -0.07195098 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.33213) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.33213) / 6.345399) / math.sqrt(2)))) * (0.01771551 * 0.5 * ((0.1204509 - Q.C2_b2) / 0.01771551) * (1 + math.erf(((0.1204509 - Q.C2_b2) / 0.01771551) / math.sqrt(2))))
    z += -0.06980681 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.0) / 1.0) / math.sqrt(2)))) * (0.02367418 * 0.5 * ((Q.z_top40_slots - 0.9370976) / 0.02367418) * (1 + math.erf(((Q.z_top40_slots - 0.9370976) / 0.02367418) / math.sqrt(2))))
    z += -18.20959 * (0.001766249 * 0.5 * ((0.01724773 - Q.M3_b2) / 0.001766249) * (1 + math.erf(((0.01724773 - Q.M3_b2) / 0.001766249) / math.sqrt(2))))
    z += 0.0009214011 * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_10) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_10) / 1.0) / math.sqrt(2))))
    z += 0.001096788 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.0) / 1.0) / math.sqrt(2)))) * (248.3135 * 0.5 * ((175.9957 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((175.9957 - Q.sip_3d_3) / 248.3135) / math.sqrt(2))))
    z += -2.284463 * (0.01556476 * 0.5 * ((0.258375 - Q.N2) / 0.01556476) * (1 + math.erf(((0.258375 - Q.N2) / 0.01556476) / math.sqrt(2))))
    z += -0.03444899 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.33213) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.33213) / 6.345399) / math.sqrt(2)))) * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2))))
    z += 0.2591928 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.0) / 1.0) / math.sqrt(2)))) * (0.02937692 * 0.5 * ((0.4866692 - Q.sj3_pairmin_over_m) / 0.02937692) * (1 + math.erf(((0.4866692 - Q.sj3_pairmin_over_m) / 0.02937692) / math.sqrt(2))))
    z += -0.1863132 * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_10) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_10) / 1.0) / math.sqrt(2)))) * (0.02426199 * 0.5 * ((Q.sdb_2_z - 0.2040425) / 0.02426199) * (1 + math.erf(((Q.sdb_2_z - 0.2040425) / 0.02426199) / math.sqrt(2))))
    z += -0.044441 * (6.036668 * 0.5 * ((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) * (1 + math.erf(((76.1529 - Q.mres_sd_mass_b2z01) / 6.036668) / math.sqrt(2))))
    z += 0.004251431 * (4.245069 * 0.5 * ((125.2732 - Q.mres_sd_mass_b2z01) / 4.245069) * (1 + math.erf(((125.2732 - Q.mres_sd_mass_b2z01) / 4.245069) / math.sqrt(2))))
    z += 2.06516e-06 * (16.47782 * 0.5 * ((162.7874 - Q.mass_top30) / 16.47782) * (1 + math.erf(((162.7874 - Q.mass_top30) / 16.47782) / math.sqrt(2)))) * (3.793098 * 0.5 * ((Q.mass_displaced5 - 9.380468) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 9.380468) / 3.793098) / math.sqrt(2))))
    z += 0.1040287 * (1.0 * 0.5 * ((3.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((3.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += -0.02202857 * (8.494762 * 0.5 * ((20.75111 - Q.sip_3d_2) / 8.494762) * (1 + math.erf(((20.75111 - Q.sip_3d_2) / 8.494762) / math.sqrt(2))))
    z += 0.006130123 * (8.537109 * 0.5 * ((17.60938 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((17.60938 - Q.max_abs_d0) / 8.537109) / math.sqrt(2))))
    z += 0.03127807 * (4.245069 * 0.5 * ((125.2732 - Q.mres_sd_mass_b2z01) / 4.245069) * (1 + math.erf(((125.2732 - Q.mres_sd_mass_b2z01) / 4.245069) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.000482418 * (3.311555 * 0.5 * ((80.27571 - Q.mass_top20) / 3.311555) * (1 + math.erf(((80.27571 - Q.mass_top20) / 3.311555) / math.sqrt(2))))
    z += 0.01292353 * (10.09897 * 0.5 * ((115.5142 - Q.sj4_pair_mass_max) / 10.09897) * (1 + math.erf(((115.5142 - Q.sj4_pair_mass_max) / 10.09897) / math.sqrt(2))))
    z += -0.00262643 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2))))
    z += 0.1175042 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2))))
    z += 0.009836751 * (3.153038 * 0.5 * ((Q.mass - 120.653) / 3.153038) * (1 + math.erf(((Q.mass - 120.653) / 3.153038) / math.sqrt(2))))
    z += -0.01008442 * (5.197 * 0.5 * ((Q.mass - 95.14961) / 5.197) * (1 + math.erf(((Q.mass - 95.14961) / 5.197) / math.sqrt(2))))
    z += 0.2992969 * (15.58693 * 0.5 * ((Q.mass_top40 - 155.8928) / 15.58693) * (1 + math.erf(((Q.mass_top40 - 155.8928) / 15.58693) / math.sqrt(2)))) * (0.01417841 * 0.5 * ((0.1049877 - Q.C2_b2) / 0.01417841) * (1 + math.erf(((0.1049877 - Q.C2_b2) / 0.01417841) / math.sqrt(2))))
    z += -0.002335455 * (2.952984 * 0.5 * ((75.78208 - Q.sj4_pair_mass_max) / 2.952984) * (1 + math.erf(((75.78208 - Q.sj4_pair_mass_max) / 2.952984) / math.sqrt(2))))
    z += 0.4906158 * (0.2024667 * 0.5 * ((Q.pair_max_lnm2 - 7.524613) / 0.2024667) * (1 + math.erf(((Q.pair_max_lnm2 - 7.524613) / 0.2024667) / math.sqrt(2))))
    z += -0.03950879 * (0.2916549 * 0.5 * ((Q.lne_21 - 1.025706) / 0.2916549) * (1 + math.erf(((Q.lne_21 - 1.025706) / 0.2916549) / math.sqrt(2))))
    z += 0.02516661 * (1.0 * 0.5 * ((11.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((11.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2))))
    z += -0.001705428 * (10.09897 * 0.5 * ((115.5142 - Q.sj4_pair_mass_max) / 10.09897) * (1 + math.erf(((115.5142 - Q.sj4_pair_mass_max) / 10.09897) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.008623313 * (10.14805 * 0.5 * ((26.78691 - Q.mass_displaced3) / 10.14805) * (1 + math.erf(((26.78691 - Q.mass_displaced3) / 10.14805) / math.sqrt(2))))
    z += 0.00196475 * (0.05184471 * 0.5 * ((-0.5637295 - Q.lund1_lndelta) / 0.05184471) * (1 + math.erf(((-0.5637295 - Q.lund1_lndelta) / 0.05184471) / math.sqrt(2))))
    z += -0.02611765 * (1.0 * 0.5 * ((5.0 - Q.n_charged_pt_above_10) / 1.0) * (1 + math.erf(((5.0 - Q.n_charged_pt_above_10) / 1.0) / math.sqrt(2))))
    z += -0.02138481 * (0.560057 * 0.5 * ((1.777787 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.777787 - Q.mass_displaced5) / 0.560057) / math.sqrt(2))))
    z += -0.05720948 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2))))
    z += -0.01519699 * (3.373048 * 0.5 * ((Q.mass_top15 - 83.68425) / 3.373048) * (1 + math.erf(((Q.mass_top15 - 83.68425) / 3.373048) / math.sqrt(2))))
    z += 0.02363529 * (0.2916549 * 0.5 * ((Q.lne_21 - 1.025706) / 0.2916549) * (1 + math.erf(((Q.lne_21 - 1.025706) / 0.2916549) / math.sqrt(2)))) * (0.08786014 * 0.5 * ((-1.279477 - Q.lnerel_0) / 0.08786014) * (1 + math.erf(((-1.279477 - Q.lnerel_0) / 0.08786014) / math.sqrt(2))))
    z += -0.003312003 * (1.0 * 0.5 * ((11.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((11.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2)))) * (0.76083 * 0.5 * ((6.306889 - Q.mres_sd_prong_mass2) / 0.76083) * (1 + math.erf(((6.306889 - Q.mres_sd_prong_mass2) / 0.76083) / math.sqrt(2))))
    z += 0.01152799 * (10.14805 * 0.5 * ((26.78691 - Q.mass_displaced3) / 10.14805) * (1 + math.erf(((26.78691 - Q.mass_displaced3) / 10.14805) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.2370407 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.2370407 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2))))
    z += 1720.795 * (1.951776e-05 * 0.5 * ((Q.ecf_g41 - 8.500146e-05) / 1.951776e-05) * (1 + math.erf(((Q.ecf_g41 - 8.500146e-05) / 1.951776e-05) / math.sqrt(2))))
    z += 0.0006519192 * (81.90557 * 0.5 * ((185.1889 - Q.jd_sum_abs_sd0_top5) / 81.90557) * (1 + math.erf(((185.1889 - Q.jd_sum_abs_sd0_top5) / 81.90557) / math.sqrt(2))))
    z += 0.09228016 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2))))
    z += -0.1876258 * (1.0 * 0.5 * ((2.0 - Q.sjf_2_n2disp) / 1.0) * (1 + math.erf(((2.0 - Q.sjf_2_n2disp) / 1.0) / math.sqrt(2))))
    z += -0.07837638 * (1.0 * 0.5 * ((Q.sjf_4_n2disp - 2.0) / 1.0) * (1 + math.erf(((Q.sjf_4_n2disp - 2.0) / 1.0) / math.sqrt(2))))
    z += 0.08682921 * (1.0 * 0.5 * ((2.0 - Q.sdb_5_n) / 1.0) * (1 + math.erf(((2.0 - Q.sdb_5_n) / 1.0) / math.sqrt(2))))
    z += -0.1265635 * (81.90557 * 0.5 * ((185.1889 - Q.jd_sum_abs_sd0_top5) / 81.90557) * (1 + math.erf(((185.1889 - Q.jd_sum_abs_sd0_top5) / 81.90557) / math.sqrt(2)))) * (0.003268159 * 0.5 * ((0.003370318 - Q.sjf_4_4_z_d3) / 0.003268159) * (1 + math.erf(((0.003370318 - Q.sjf_4_4_z_d3) / 0.003268159) / math.sqrt(2))))
    z += -0.1424528 * (0.06585074 * 0.5 * ((-0.1170754 - Q.tdz_1) / 0.06585074) * (1 + math.erf(((-0.1170754 - Q.tdz_1) / 0.06585074) / math.sqrt(2))))
    z += -9.486529 * (0.006687614 * 0.5 * ((0.02210827 - Q.dr_max_012) / 0.006687614) * (1 + math.erf(((0.02210827 - Q.dr_max_012) / 0.006687614) / math.sqrt(2))))
    z += -0.008326317 * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 5.0) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 5.0) / 1.0) / math.sqrt(2))))
    z += 1.750899 * (0.02719315 * 0.5 * ((0.1530389 - Q.sdb_2_z) / 0.02719315) * (1 + math.erf(((0.1530389 - Q.sdb_2_z) / 0.02719315) / math.sqrt(2))))
    z += -0.5516222 * (0.09779513 * 0.5 * ((0.6703997 - Q.sdb_2_z) / 0.09779513) * (1 + math.erf(((0.6703997 - Q.sdb_2_z) / 0.09779513) / math.sqrt(2))))
    z += 1.011294 * (0.2916549 * 0.5 * ((Q.lne_21 - 1.025706) / 0.2916549) * (1 + math.erf(((Q.lne_21 - 1.025706) / 0.2916549) / math.sqrt(2)))) * (0.02507469 * 0.5 * ((0.1545914 - Q.sj4_zsoft) / 0.02507469) * (1 + math.erf(((0.1545914 - Q.sj4_zsoft) / 0.02507469) / math.sqrt(2))))
    z += 0.009642487 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.33213) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.33213) / 6.345399) / math.sqrt(2)))) * (0.01759885 * 0.5 * ((Q.sj3_z1 - 0.4972199) / 0.01759885) * (1 + math.erf(((Q.sj3_z1 - 0.4972199) / 0.01759885) / math.sqrt(2))))
    z += -0.2561999 * (0.09779513 * 0.5 * ((0.6703997 - Q.sdb_2_z) / 0.09779513) * (1 + math.erf(((0.6703997 - Q.sdb_2_z) / 0.09779513) / math.sqrt(2)))) * (0.1379966 * 0.5 * ((Q.ak02_dr23 - 0.2385164) / 0.1379966) * (1 + math.erf(((Q.ak02_dr23 - 0.2385164) / 0.1379966) / math.sqrt(2))))
    z += 0.2849866 * (0.04874922 * 0.5 * ((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) * (1 + math.erf(((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) / math.sqrt(2))))
    z += 0.004022462 * (3.24083 * 0.5 * ((6.161303 - Q.dc_2_jp) / 3.24083) * (1 + math.erf(((6.161303 - Q.dc_2_jp) / 3.24083) / math.sqrt(2))))
    z += 0.06365818 * (1.5 * 0.5 * ((10.0 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((10.0 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2))))
    z += 0.0001177535 * (76.60049 * 0.5 * ((Q.sum_e - 1049.334) / 76.60049) * (1 + math.erf(((Q.sum_e - 1049.334) / 76.60049) / math.sqrt(2))))
    z += 0.04634731 * (0.04874922 * 0.5 * ((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) * (1 + math.erf(((0.1276795 - Q.sjf_4_1_z_d3) / 0.04874922) / math.sqrt(2)))) * (6.566474 * 0.5 * ((12.91148 - Q.lepsj_3_mass) / 6.566474) * (1 + math.erf(((12.91148 - Q.lepsj_3_mass) / 6.566474) / math.sqrt(2))))
    z += -0.01695576 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.5 * 0.5 * ((2.0 - Q.lepsj_2_n_d3) / 1.5) * (1 + math.erf(((2.0 - Q.lepsj_2_n_d3) / 1.5) / math.sqrt(2))))
    z += 0.001183756 * (5.740575 * 0.5 * ((6.185635 - Q.lep_iso) / 5.740575) * (1 + math.erf(((6.185635 - Q.lep_iso) / 5.740575) / math.sqrt(2))))
    z += 2.010043 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2))))
    z += 6.371359e-05 * (16.47782 * 0.5 * ((162.7874 - Q.mass_top30) / 16.47782) * (1 + math.erf(((162.7874 - Q.mass_top30) / 16.47782) / math.sqrt(2)))) * (0.1300849 * 0.5 * ((Q.dc_tag_2nd - 0.0) / 0.1300849) * (1 + math.erf(((Q.dc_tag_2nd - 0.0) / 0.1300849) / math.sqrt(2))))
    z += 0.01870351 * (1.0 * 0.5 * ((Q.sdb_0_n - 3.0) / 1.0) * (1 + math.erf(((Q.sdb_0_n - 3.0) / 1.0) / math.sqrt(2))))
    z += -1.122985 * (0.02719315 * 0.5 * ((0.1530389 - Q.sdb_2_z) / 0.02719315) * (1 + math.erf(((0.1530389 - Q.sdb_2_z) / 0.02719315) / math.sqrt(2)))) * (0.01728058 * 0.5 * ((Q.eta_4 - -0.06317139) / 0.01728058) * (1 + math.erf(((Q.eta_4 - -0.06317139) / 0.01728058) / math.sqrt(2))))
    z += -5.454128 * (0.01159153 * 0.5 * ((0.04504618 - Q.z_dr_0p4_up) / 0.01159153) * (1 + math.erf(((0.04504618 - Q.z_dr_0p4_up) / 0.01159153) / math.sqrt(2))))
    z += 3.510538 * (0.007063147 * 0.5 * ((0.01810676 - Q.z_displaced5) / 0.007063147) * (1 + math.erf(((0.01810676 - Q.z_displaced5) / 0.007063147) / math.sqrt(2))))
    z += 8.76433e-05 * (1.5 * 0.5 * ((10.0 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((10.0 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dc_4_mass_disp3 - 0.0) / 1e-06) * (1 + math.erf(((Q.dc_4_mass_disp3 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.0313637 * (1.0 * 0.5 * ((Q.sdb_0_n - 3.0) / 1.0) * (1 + math.erf(((Q.sdb_0_n - 3.0) / 1.0) / math.sqrt(2)))) * (0.08551811 * 0.5 * ((Q.tdz_10 - 0.1420982) / 0.08551811) * (1 + math.erf(((Q.tdz_10 - 0.1420982) / 0.08551811) / math.sqrt(2))))
    z += -0.04453215 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (3.384707 * 0.5 * ((6.771002 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.771002 - Q.jd_3d_5) / 3.384707) / math.sqrt(2))))
    z += -0.04601618 * (2.0 * 0.5 * ((8.0 - Q.n_s3d_above_10) / 2.0) * (1 + math.erf(((8.0 - Q.n_s3d_above_10) / 2.0) / math.sqrt(2))))
    z += 0.002564104 * (10.14805 * 0.5 * ((26.78691 - Q.mass_displaced3) / 10.14805) * (1 + math.erf(((26.78691 - Q.mass_displaced3) / 10.14805) / math.sqrt(2)))) * (3.384707 * 0.5 * ((6.771002 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.771002 - Q.jd_3d_5) / 3.384707) / math.sqrt(2))))
    z += 0.08601028 * (3.384707 * 0.5 * ((6.771002 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.771002 - Q.jd_3d_5) / 3.384707) / math.sqrt(2))))
    z += 0.0003647432 * (3.153038 * 0.5 * ((Q.mass - 120.653) / 3.153038) * (1 + math.erf(((Q.mass - 120.653) / 3.153038) / math.sqrt(2)))) * (5.085197 * 0.5 * ((10.7744 - Q.jd_3d_5) / 5.085197) * (1 + math.erf(((10.7744 - Q.jd_3d_5) / 5.085197) / math.sqrt(2))))
    z += 0.101362 * (5.171003 * 0.5 * ((7.028704 - Q.sip_3d_1) / 5.171003) * (1 + math.erf(((7.028704 - Q.sip_3d_1) / 5.171003) / math.sqrt(2))))
    z += -0.000151099 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.33213) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.33213) / 6.345399) / math.sqrt(2)))) * (18.22661 * 0.5 * ((27.25083 - Q.jd_3d_5) / 18.22661) * (1 + math.erf(((27.25083 - Q.jd_3d_5) / 18.22661) / math.sqrt(2))))
    z += 0.006420459 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (5.085197 * 0.5 * ((10.7744 - Q.jd_3d_5) / 5.085197) * (1 + math.erf(((10.7744 - Q.jd_3d_5) / 5.085197) / math.sqrt(2))))
    z += -0.01720309 * (8.537109 * 0.5 * ((17.60938 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((17.60938 - Q.max_abs_d0) / 8.537109) / math.sqrt(2)))) * (0.1631384 * 0.5 * ((0.611688 - Q.D3) / 0.1631384) * (1 + math.erf(((0.611688 - Q.D3) / 0.1631384) / math.sqrt(2))))
    z += -5.71554e-05 * (1.0 * 0.5 * ((11.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((11.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2)))) * (26.14378 * 0.5 * ((Q.jd_3d_5 - 53.39461) / 26.14378) * (1 + math.erf(((Q.jd_3d_5 - 53.39461) / 26.14378) / math.sqrt(2))))
    z += -10.15349 * (0.003751777 * 0.5 * ((0.007652966 - Q.z_dr_0_0p05) / 0.003751777) * (1 + math.erf(((0.007652966 - Q.z_dr_0_0p05) / 0.003751777) / math.sqrt(2))))
    z += 0.02190661 * (0.2916549 * 0.5 * ((Q.lne_21 - 1.025706) / 0.2916549) * (1 + math.erf(((Q.lne_21 - 1.025706) / 0.2916549) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((3.0 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2))))
    z += -0.001010034 * (2.952984 * 0.5 * ((75.78208 - Q.sj4_pair_mass_max) / 2.952984) * (1 + math.erf(((75.78208 - Q.sj4_pair_mass_max) / 2.952984) / math.sqrt(2)))) * (22.16789 * 0.5 * ((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) * (1 + math.erf(((11.53372 - Q.lepsj_3_maxsd0) / 22.16789) / math.sqrt(2))))
    z += 0.001644886 * (3.384707 * 0.5 * ((6.771002 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.771002 - Q.jd_3d_5) / 3.384707) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.nca_kt_above_20 - 1.0) / 1.0) * (1 + math.erf(((Q.nca_kt_above_20 - 1.0) / 1.0) / math.sqrt(2))))
    z += 8.688231 * (0.003751777 * 0.5 * ((0.007652966 - Q.z_dr_0_0p05) / 0.003751777) * (1 + math.erf(((0.007652966 - Q.z_dr_0_0p05) / 0.003751777) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.iselectron_11 - 0.0) / 1e-06) * (1 + math.erf(((Q.iselectron_11 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.03223925 * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_10) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_10) / 1.0) / math.sqrt(2)))) * (0.06371583 * 0.5 * ((Q.sv_2_dr - 0.2172495) / 0.06371583) * (1 + math.erf(((Q.sv_2_dr - 0.2172495) / 0.06371583) / math.sqrt(2))))
    z += 0.04643507 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ismuon_3 - 0.0) / 1e-06) * (1 + math.erf(((Q.ismuon_3 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.04151495 * (6.230332 * 0.5 * ((13.50273 - Q.sip_3d_2) / 6.230332) * (1 + math.erf(((13.50273 - Q.sip_3d_2) / 6.230332) / math.sqrt(2))))
    z += 0.4489138 * (0.04905322 * 0.5 * ((0.3377343 - Q.tau32_b2) / 0.04905322) * (1 + math.erf(((0.3377343 - Q.tau32_b2) / 0.04905322) / math.sqrt(2))))
    z += -0.007761925 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2)))) * (2.0 * 0.5 * ((9.0 - Q.n_neutral_had) / 2.0) * (1 + math.erf(((9.0 - Q.n_neutral_had) / 2.0) / math.sqrt(2))))
    z += -12.09768 * (0.00905862 * 0.5 * ((0.06519357 - Q.tau5) / 0.00905862) * (1 + math.erf(((0.06519357 - Q.tau5) / 0.00905862) / math.sqrt(2))))
    z += 0.009656996 * (16.90428 * 0.5 * ((164.4374 - Q.mass) / 16.90428) * (1 + math.erf(((164.4374 - Q.mass) / 16.90428) / math.sqrt(2))))
    z += -3.9798 * (1.951776e-05 * 0.5 * ((Q.ecf_g41 - 8.500146e-05) / 1.951776e-05) * (1 + math.erf(((Q.ecf_g41 - 8.500146e-05) / 1.951776e-05) / math.sqrt(2)))) * (57.98295 * 0.5 * ((767.6359 - Q.sum_pt_top40) / 57.98295) * (1 + math.erf(((767.6359 - Q.sum_pt_top40) / 57.98295) / math.sqrt(2))))
    z += 0.03043255 * (1.5 * 0.5 * ((10.0 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((10.0 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2)))) * (0.3391285 * 0.5 * ((Q.max_dr - 0.9206713) / 0.3391285) * (1 + math.erf(((Q.max_dr - 0.9206713) / 0.3391285) / math.sqrt(2))))
    z += 0.03953616 * (10.51039 * 0.5 * ((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) * (1 + math.erf(((69.12133 - Q.mres_sd_mass_b2z01) / 10.51039) / math.sqrt(2))))
    z += -37.25845 * (0.001766249 * 0.5 * ((0.01724773 - Q.M3_b2) / 0.001766249) * (1 + math.erf(((0.01724773 - Q.M3_b2) / 0.001766249) / math.sqrt(2)))) * (0.004593578 * 0.5 * ((Q.sjf_4_2_z_d3 - 0.004340802) / 0.004593578) * (1 + math.erf(((Q.sjf_4_2_z_d3 - 0.004340802) / 0.004593578) / math.sqrt(2))))
    z += 2.02248 * (0.09626377 * 0.5 * ((Q.sjf_2_1_z_d3 - 0.3672808) / 0.09626377) * (1 + math.erf(((Q.sjf_2_1_z_d3 - 0.3672808) / 0.09626377) / math.sqrt(2))))
    return z


def neuron_84(Q):
    z = -7.427509e-05
    return z


def neuron_85(Q):
    z = -8.185526e-07
    return z


def neuron_86(Q):
    z = -1.709902e-05
    return z


def neuron_87(Q):
    z = 1.619161e-06
    return z


def neuron_88(Q):
    z = 2.996977e-07
    return z


def neuron_89(Q):
    z = 1.735373e-06
    return z


def neuron_90(Q):
    z = -0.01547664
    z += -0.8260072 * Q.pz_lnd2
    z += -0.06828237 * Q.tau21
    z += -0.1322248 * Q.dc_1_n_lep
    z += 1.216199 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2))))
    z += 0.7842358 * (0.08158173 * 0.5 * ((0.3261071 - Q.z_displaced3) / 0.08158173) * (1 + math.erf(((0.3261071 - Q.z_displaced3) / 0.08158173) / math.sqrt(2))))
    z += -0.0002007187 * (12.49365 * 0.5 * ((128.0079 - Q.sj4_pair_mass_max) / 12.49365) * (1 + math.erf(((128.0079 - Q.sj4_pair_mass_max) / 12.49365) / math.sqrt(2))))
    z += 1.943169 * (0.04445521 * 0.5 * ((0.2686235 - Q.LHA) / 0.04445521) * (1 + math.erf(((0.2686235 - Q.LHA) / 0.04445521) / math.sqrt(2))))
    z += 3.255699 * (0.01958217 * 0.5 * ((0.3667049 - Q.N2_b05) / 0.01958217) * (1 + math.erf(((0.3667049 - Q.N2_b05) / 0.01958217) / math.sqrt(2))))
    z += -0.03171734 * (1.0 * 0.5 * ((2.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((2.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2))))
    z += -1.837269 * (0.08158173 * 0.5 * ((0.3261071 - Q.z_displaced3) / 0.08158173) * (1 + math.erf(((0.3261071 - Q.z_displaced3) / 0.08158173) / math.sqrt(2)))) * (0.0408915 * 0.5 * ((0.04607888 - Q.lep_dr) / 0.0408915) * (1 + math.erf(((0.04607888 - Q.lep_dr) / 0.0408915) / math.sqrt(2))))
    z += 1.248037 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.01981371 * 0.5 * ((Q.sj3_pairmin_over_m - 0.2477126) / 0.01981371) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.2477126) / 0.01981371) / math.sqrt(2))))
    z += 0.08253533 * (0.8047829 * 0.5 * ((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) * (1 + math.erf(((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) / math.sqrt(2))))
    z += 2.298138 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2))))
    z += -15.08707 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (0.005517122 * 0.5 * ((0.9919867 - Q.z_top40_slots) / 0.005517122) * (1 + math.erf(((0.9919867 - Q.z_top40_slots) / 0.005517122) / math.sqrt(2))))
    z += 0.4103359 * (0.002177355 * 0.5 * ((0.01919357 - Q.M3_b2) / 0.002177355) * (1 + math.erf(((0.01919357 - Q.M3_b2) / 0.002177355) / math.sqrt(2))))
    z += 0.005217903 * (16.90428 * 0.5 * ((164.4374 - Q.mass) / 16.90428) * (1 + math.erf(((164.4374 - Q.mass) / 16.90428) / math.sqrt(2))))
    z += 0.003333729 * (3.485807 * 0.5 * ((119.1279 - Q.mass_top40) / 3.485807) * (1 + math.erf(((119.1279 - Q.mass_top40) / 3.485807) / math.sqrt(2))))
    z += -0.003429917 * (5.286881 * 0.5 * ((Q.mass - 100.4835) / 5.286881) * (1 + math.erf(((Q.mass - 100.4835) / 5.286881) / math.sqrt(2))))
    z += 0.0157889 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2))))
    z += 0.005083051 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2))))
    z += 3.021408e-07 * (12.49365 * 0.5 * ((128.0079 - Q.sj4_pair_mass_max) / 12.49365) * (1 + math.erf(((128.0079 - Q.sj4_pair_mass_max) / 12.49365) / math.sqrt(2)))) * (540.2081 * 0.5 * ((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2))))
    z += 0.0101754 * (0.8047829 * 0.5 * ((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) * (1 + math.erf(((1.529925 - Q.lepsj_3_maxsd0) / 0.8047829) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_ntag - 0.0) / 1.0) * (1 + math.erf(((Q.dc_ntag - 0.0) / 1.0) / math.sqrt(2))))
    z += 0.03267691 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2))))
    z += 0.1446108 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2))))
    z += 0.00366342 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2))))
    z += 0.008346924 * (1.0 * 0.5 * ((10.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((10.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2))))
    z += -38.4814 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2))))
    z += 0.06219566 * (0.1314845 * 0.5 * ((Q.pair_mean_lnm2 - 2.708356) / 0.1314845) * (1 + math.erf(((Q.pair_mean_lnm2 - 2.708356) / 0.1314845) / math.sqrt(2))))
    z += -0.0006292947 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2)))) * (1.739103 * 0.5 * ((3.969624 - Q.jd_3d_4) / 1.739103) * (1 + math.erf(((3.969624 - Q.jd_3d_4) / 1.739103) / math.sqrt(2))))
    z += -6.078241e-05 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2))))
    z += 0.009171124 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.05461387 * 0.5 * ((0.2184442 - Q.z_displaced5) / 0.05461387) * (1 + math.erf(((0.2184442 - Q.z_displaced5) / 0.05461387) / math.sqrt(2))))
    z += -0.005166893 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.05461387 * 0.5 * ((0.2184442 - Q.z_displaced5) / 0.05461387) * (1 + math.erf(((0.2184442 - Q.z_displaced5) / 0.05461387) / math.sqrt(2))))
    z += 0.001554072 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (3.793098 * 0.5 * ((Q.mass_displaced5 - 9.380468) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 9.380468) / 3.793098) / math.sqrt(2))))
    z += -0.006428742 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2)))) * (0.007119697 * 0.5 * ((0.177312 - Q.M2_b05) / 0.007119697) * (1 + math.erf(((0.177312 - Q.M2_b05) / 0.007119697) / math.sqrt(2))))
    z += -2.097495 * (0.04445521 * 0.5 * ((0.2686235 - Q.LHA) / 0.04445521) * (1 + math.erf(((0.2686235 - Q.LHA) / 0.04445521) / math.sqrt(2)))) * (0.5801757 * 0.5 * ((1.544662 - Q.N3_b2) / 0.5801757) * (1 + math.erf(((1.544662 - Q.N3_b2) / 0.5801757) / math.sqrt(2))))
    z += -5.643937 * (0.01958217 * 0.5 * ((0.3667049 - Q.N2_b05) / 0.01958217) * (1 + math.erf(((0.3667049 - Q.N2_b05) / 0.01958217) / math.sqrt(2)))) * (0.03181498 * 0.5 * ((Q.z_charged_had - 0.3945478) / 0.03181498) * (1 + math.erf(((Q.z_charged_had - 0.3945478) / 0.03181498) / math.sqrt(2))))
    z += 2.16975 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (0.001294243 * 0.5 * ((Q.sdb_5_z - 0.0) / 0.001294243) * (1 + math.erf(((Q.sdb_5_z - 0.0) / 0.001294243) / math.sqrt(2))))
    z += -0.00145727 * (11.68693 * 0.5 * ((141.9283 - Q.mass_top40) / 11.68693) * (1 + math.erf(((141.9283 - Q.mass_top40) / 11.68693) / math.sqrt(2))))
    z += 0.5051471 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) / math.sqrt(2))))
    z += -0.01381576 * (0.1228932 * 0.5 * ((3.521178 - Q.lund_max_lnkt) / 0.1228932) * (1 + math.erf(((3.521178 - Q.lund_max_lnkt) / 0.1228932) / math.sqrt(2))))
    z += -0.322214 * (0.1314845 * 0.5 * ((Q.pair_mean_lnm2 - 2.708356) / 0.1314845) * (1 + math.erf(((Q.pair_mean_lnm2 - 2.708356) / 0.1314845) / math.sqrt(2)))) * (0.0330605 * 0.5 * ((Q.lep_dr - 0.114659) / 0.0330605) * (1 + math.erf(((Q.lep_dr - 0.114659) / 0.0330605) / math.sqrt(2))))
    z += 0.008900292 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2)))) * (0.2508028 * 0.5 * ((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.6550393 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2))))
    z += -0.727212 * (0.0278003 * 0.5 * ((0.4239562 - Q.z_charged_had) / 0.0278003) * (1 + math.erf(((0.4239562 - Q.z_charged_had) / 0.0278003) / math.sqrt(2))))
    z += 0.03427039 * (1.5 * 0.5 * ((7.0 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.0 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2))))
    z += -0.1208691 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_3_n2disp - 0.0) / 1.0) * (1 + math.erf(((Q.sjf_3_n2disp - 0.0) / 1.0) / math.sqrt(2))))
    z += -4.435258 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.05691776 * 0.5 * ((Q.jet_charge - 0.2072767) / 0.05691776) * (1 + math.erf(((Q.jet_charge - 0.2072767) / 0.05691776) / math.sqrt(2))))
    z += -0.002999123 * (1.5 * 0.5 * ((7.0 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.0 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_1_n_lep - 0.0) / 1.0) * (1 + math.erf(((Q.dc_1_n_lep - 0.0) / 1.0) / math.sqrt(2))))
    z += 0.004078751 * (4.902193 * 0.5 * ((126.8853 - Q.mass_top40) / 4.902193) * (1 + math.erf(((126.8853 - Q.mass_top40) / 4.902193) / math.sqrt(2))))
    z += -1.883451 * (0.02862925 * 0.5 * ((0.2665611 - Q.tau21) / 0.02862925) * (1 + math.erf(((0.2665611 - Q.tau21) / 0.02862925) / math.sqrt(2))))
    z += 0.002480252 * (4.06561 * 0.5 * ((Q.mass_2charged - 24.4079) / 4.06561) * (1 + math.erf(((Q.mass_2charged - 24.4079) / 4.06561) / math.sqrt(2))))
    z += -6.602654 * (0.07843207 * 0.5 * ((0.1275041 - Q.lep_z) / 0.07843207) * (1 + math.erf(((0.1275041 - Q.lep_z) / 0.07843207) / math.sqrt(2))))
    z += -0.0005007694 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += -0.1140499 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (29.76053 * 0.5 * ((44.14912 - Q.lep_iso) / 29.76053) * (1 + math.erf(((44.14912 - Q.lep_iso) / 29.76053) / math.sqrt(2))))
    z += 6.221553 * (0.006899392 * 0.5 * ((0.006470637 - Q.lepsj_2_dr) / 0.006899392) * (1 + math.erf(((0.006470637 - Q.lepsj_2_dr) / 0.006899392) / math.sqrt(2))))
    z += 0.003739849 * (1.0 * 0.5 * ((Q.n_charged_had - 18.0) / 1.0) * (1 + math.erf(((Q.n_charged_had - 18.0) / 1.0) / math.sqrt(2))))
    z += -0.01280502 * (4.06561 * 0.5 * ((Q.mass_2charged - 24.4079) / 4.06561) * (1 + math.erf(((Q.mass_2charged - 24.4079) / 4.06561) / math.sqrt(2)))) * (0.1334042 * 0.5 * ((0.5033607 - Q.sv_2_dr) / 0.1334042) * (1 + math.erf(((0.5033607 - Q.sv_2_dr) / 0.1334042) / math.sqrt(2))))
    z += 6.22029 * (0.003868055 * 0.5 * ((Q.psi_0p3 - 0.9925964) / 0.003868055) * (1 + math.erf(((Q.psi_0p3 - 0.9925964) / 0.003868055) / math.sqrt(2))))
    z += -0.04779766 * (1.0 * 0.5 * ((10.0 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((10.0 - Q.sdb_2_n) / 1.0) / math.sqrt(2)))) * (0.7057249 * 0.5 * ((0.6893409 - Q.dc_3_jp) / 0.7057249) * (1 + math.erf(((0.6893409 - Q.dc_3_jp) / 0.7057249) / math.sqrt(2))))
    z += -32.39721 * (0.01203596 * 0.5 * ((0.0318986 - Q.e2) / 0.01203596) * (1 + math.erf(((0.0318986 - Q.e2) / 0.01203596) / math.sqrt(2))))
    z += -0.00615399 * (3.560123 * 0.5 * ((120.8334 - Q.mres_sd_mass_b1z01) / 3.560123) * (1 + math.erf(((120.8334 - Q.mres_sd_mass_b1z01) / 3.560123) / math.sqrt(2))))
    z += -6.119438e-06 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2)))) * (4.522214 * 0.5 * ((Q.dc_split2_kt - 13.50797) / 4.522214) * (1 + math.erf(((Q.dc_split2_kt - 13.50797) / 4.522214) / math.sqrt(2))))
    z += 0.01043096 * (12.49365 * 0.5 * ((128.0079 - Q.sj4_pair_mass_max) / 12.49365) * (1 + math.erf(((128.0079 - Q.sj4_pair_mass_max) / 12.49365) / math.sqrt(2)))) * (0.06955833 * 0.5 * ((0.3788785 - Q.z_neutral_had) / 0.06955833) * (1 + math.erf(((0.3788785 - Q.z_neutral_had) / 0.06955833) / math.sqrt(2))))
    z += -0.008756535 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2))))
    z += -0.004728819 * (0.08158173 * 0.5 * ((0.3261071 - Q.z_displaced3) / 0.08158173) * (1 + math.erf(((0.3261071 - Q.z_displaced3) / 0.08158173) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2))))
    z += 20.92767 * (0.001059952 * 0.5 * ((0.001673521 - Q.sum_z_dr2_top3) / 0.001059952) * (1 + math.erf(((0.001673521 - Q.sum_z_dr2_top3) / 0.001059952) / math.sqrt(2))))
    z += -0.0009256982 * (18.42178 * 0.5 * ((182.8592 - Q.mass) / 18.42178) * (1 + math.erf(((182.8592 - Q.mass) / 18.42178) / math.sqrt(2))))
    z += -0.007867079 * (5.539613 * 0.5 * ((94.12155 - Q.mres_sd_mass_b1z01) / 5.539613) * (1 + math.erf(((94.12155 - Q.mres_sd_mass_b1z01) / 5.539613) / math.sqrt(2))))
    z += 0.003822875 * (13.85645 * 0.5 * ((137.6422 - Q.mres_sd_mass_b1z01) / 13.85645) * (1 + math.erf(((137.6422 - Q.mres_sd_mass_b1z01) / 13.85645) / math.sqrt(2))))
    z += -0.008923533 * (3.244489 * 0.5 * ((66.70506 - Q.sj4_pair_mass_max) / 3.244489) * (1 + math.erf(((66.70506 - Q.sj4_pair_mass_max) / 3.244489) / math.sqrt(2))))
    z += 0.02554584 * (5.286881 * 0.5 * ((Q.mass - 100.4835) / 5.286881) * (1 + math.erf(((Q.mass - 100.4835) / 5.286881) / math.sqrt(2)))) * (0.006231177 * 0.5 * ((0.03947764 - Q.sj4_zsoft) / 0.006231177) * (1 + math.erf(((0.03947764 - Q.sj4_zsoft) / 0.006231177) / math.sqrt(2))))
    z += -0.003605088 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (1.0 * 0.5 * ((6.0 - Q.n_neutral_had) / 1.0) * (1 + math.erf(((6.0 - Q.n_neutral_had) / 1.0) / math.sqrt(2))))
    z += -8.568617e-05 * (43.0 * 0.5 * ((366.0 - Q.n_pairs_kt_above_1) / 43.0) * (1 + math.erf(((366.0 - Q.n_pairs_kt_above_1) / 43.0) / math.sqrt(2))))
    z += -0.4385448 * (0.02005207 * 0.5 * ((Q.sj3_pairmin_over_m - 0.2674688) / 0.02005207) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.2674688) / 0.02005207) / math.sqrt(2))))
    z += 0.002408911 * (3.560123 * 0.5 * ((120.8334 - Q.mres_sd_mass_b1z01) / 3.560123) * (1 + math.erf(((120.8334 - Q.mres_sd_mass_b1z01) / 3.560123) / math.sqrt(2)))) * (0.01029703 * 0.5 * ((-0.009407043 - Q.phi_1) / 0.01029703) * (1 + math.erf(((-0.009407043 - Q.phi_1) / 0.01029703) / math.sqrt(2))))
    z += 0.01736406 * (5.297564 * 0.5 * ((78.89137 - Q.mres_sd_mass_b1z01) / 5.297564) * (1 + math.erf(((78.89137 - Q.mres_sd_mass_b1z01) / 5.297564) / math.sqrt(2))))
    z += 0.7545488 * (0.06646541 * 0.5 * ((0.3206143 - Q.pt1_over_pt0) / 0.06646541) * (1 + math.erf(((0.3206143 - Q.pt1_over_pt0) / 0.06646541) / math.sqrt(2))))
    z += -7.005345 * (0.01268163 * 0.5 * ((0.02077199 - Q.mres_sd_rg_b0z02) / 0.01268163) * (1 + math.erf(((0.02077199 - Q.mres_sd_rg_b0z02) / 0.01268163) / math.sqrt(2))))
    z += 0.0001340981 * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.01429749 * 0.5 * ((-0.03515625 - Q.phi_3) / 0.01429749) * (1 + math.erf(((-0.03515625 - Q.phi_3) / 0.01429749) / math.sqrt(2))))
    z += -0.002827253 * (2.340241 * 0.5 * ((Q.mass_displaced3 - 6.341631) / 2.340241) * (1 + math.erf(((Q.mass_displaced3 - 6.341631) / 2.340241) / math.sqrt(2))))
    z += -0.0005767921 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (0.2012988 * 0.5 * ((0.2012988 - Q.kt2_min12_mass_disp3) / 0.2012988) * (1 + math.erf(((0.2012988 - Q.kt2_min12_mass_disp3) / 0.2012988) / math.sqrt(2))))
    z += -0.2980728 * (0.0331444 * 0.5 * ((0.1339658 - Q.z_displaced5) / 0.0331444) * (1 + math.erf(((0.1339658 - Q.z_displaced5) / 0.0331444) / math.sqrt(2))))
    z += 0.0006040621 * (5.286881 * 0.5 * ((Q.mass - 100.4835) / 5.286881) * (1 + math.erf(((Q.mass - 100.4835) / 5.286881) / math.sqrt(2)))) * (0.07571368 * 0.5 * ((-0.1329965 - Q.tdz_13) / 0.07571368) * (1 + math.erf(((-0.1329965 - Q.tdz_13) / 0.07571368) / math.sqrt(2))))
    z += -1.360028 * (0.01958217 * 0.5 * ((0.3667049 - Q.N2_b05) / 0.01958217) * (1 + math.erf(((0.3667049 - Q.N2_b05) / 0.01958217) / math.sqrt(2)))) * (0.03825143 * 0.5 * ((Q.sjq_2_prod_k03 - 0.06059953) / 0.03825143) * (1 + math.erf(((Q.sjq_2_prod_k03 - 0.06059953) / 0.03825143) / math.sqrt(2))))
    z += 0.004649063 * (1.5 * 0.5 * ((Q.n_for_90pct - 11.0) / 1.5) * (1 + math.erf(((Q.n_for_90pct - 11.0) / 1.5) / math.sqrt(2))))
    z += 0.002765387 * (11.28806 * 0.5 * ((Q.mass_top40 - 70.88236) / 11.28806) * (1 + math.erf(((Q.mass_top40 - 70.88236) / 11.28806) / math.sqrt(2)))) * (0.03537257 * 0.5 * ((0.9798274 - Q.ak02_1_z) / 0.03537257) * (1 + math.erf(((0.9798274 - Q.ak02_1_z) / 0.03537257) / math.sqrt(2))))
    z += -0.9564163 * (0.001626539 * 0.5 * ((1.0 - Q.z_top20_slots) / 0.001626539) * (1 + math.erf(((1.0 - Q.z_top20_slots) / 0.001626539) / math.sqrt(2))))
    z += -0.689063 * (0.01869595 * 0.5 * ((Q.LHA - 0.5052443) / 0.01869595) * (1 + math.erf(((Q.LHA - 0.5052443) / 0.01869595) / math.sqrt(2))))
    z += 2.616286e-05 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_3_2_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_3_2_n_d3 - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.8720153 * (0.03675411 * 0.5 * ((Q.z_neutral - 0.570175) / 0.03675411) * (1 + math.erf(((Q.z_neutral - 0.570175) / 0.03675411) / math.sqrt(2))))
    z += 0.002340271 * (4.902193 * 0.5 * ((126.8853 - Q.mass_top40) / 4.902193) * (1 + math.erf(((126.8853 - Q.mass_top40) / 4.902193) / math.sqrt(2)))) * (0.02635594 * 0.5 * ((Q.eccentricity - 0.7995541) / 0.02635594) * (1 + math.erf(((Q.eccentricity - 0.7995541) / 0.02635594) / math.sqrt(2))))
    z += -0.004046998 * (6.549481 * 0.5 * ((129.1614 - Q.mres_sd_mass_b1z01) / 6.549481) * (1 + math.erf(((129.1614 - Q.mres_sd_mass_b1z01) / 6.549481) / math.sqrt(2))))
    z += -0.0431342 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.6535817) / 0.2498329) / math.sqrt(2)))) * (0.1175438 * 0.5 * ((Q.dc_3_charge - 0.0) / 0.1175438) * (1 + math.erf(((Q.dc_3_charge - 0.0) / 0.1175438) / math.sqrt(2))))
    z += 0.1506481 * (0.06042313 * 0.5 * ((Q.sjq_3_sumabs_k1 - 0.4372817) / 0.06042313) * (1 + math.erf(((Q.sjq_3_sumabs_k1 - 0.4372817) / 0.06042313) / math.sqrt(2))))
    z += 5.77889 * (0.002177355 * 0.5 * ((0.01919357 - Q.M3_b2) / 0.002177355) * (1 + math.erf(((0.01919357 - Q.M3_b2) / 0.002177355) / math.sqrt(2)))) * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2))))
    z += -0.0003188505 * (5.297564 * 0.5 * ((78.89137 - Q.mres_sd_mass_b1z01) / 5.297564) * (1 + math.erf(((78.89137 - Q.mres_sd_mass_b1z01) / 5.297564) / math.sqrt(2)))) * (0.7811923 * 0.5 * ((6.006437 - Q.sj3_mass2) / 0.7811923) * (1 + math.erf(((6.006437 - Q.sj3_mass2) / 0.7811923) / math.sqrt(2))))
    z += -0.7613842 * (0.06215671 * 0.5 * ((Q.sj4_dr_min - 0.3112717) / 0.06215671) * (1 + math.erf(((Q.sj4_dr_min - 0.3112717) / 0.06215671) / math.sqrt(2))))
    z += -0.001199842 * (22.11238 * 0.5 * ((34.60954 - Q.sjf_2_1_mass_d3) / 22.11238) * (1 + math.erf(((34.60954 - Q.sjf_2_1_mass_d3) / 22.11238) / math.sqrt(2))))
    z += -0.2177889 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2))))
    z += -335.1067 * (7.099104e-05 * 0.5 * ((0.0003489585 - Q.e3) / 7.099104e-05) * (1 + math.erf(((0.0003489585 - Q.e3) / 7.099104e-05) / math.sqrt(2))))
    z += -3.061711 * (0.01294944 * 0.5 * ((0.08643515 - Q.tau3) / 0.01294944) * (1 + math.erf(((0.08643515 - Q.tau3) / 0.01294944) / math.sqrt(2))))
    return z


def neuron_91(Q):
    z = 3.992636e-06
    return z


def neuron_92(Q):
    z = -6.72473e-06
    return z


def neuron_93(Q):
    z = 8.464408e-07
    return z


def neuron_94(Q):
    z = -2.322168e-06
    return z


def neuron_95(Q):
    z = 4.369498e-07
    return z


def neuron_96(Q):
    z = 4.278005e-07
    return z


def neuron_97(Q):
    z = -1.464491
    z += 0.01766467 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2))))
    z += 0.02586458 * (5.286881 * 0.5 * ((100.4835 - Q.mass) / 5.286881) * (1 + math.erf(((100.4835 - Q.mass) / 5.286881) / math.sqrt(2))))
    z += -284.927 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2))))
    z += 0.04461319 * (0.1256484 * 0.5 * ((Q.pair_mean_lnm2 - 2.09826) / 0.1256484) * (1 + math.erf(((Q.pair_mean_lnm2 - 2.09826) / 0.1256484) / math.sqrt(2))))
    z += 315.7214 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.02505229 * 0.5 * ((0.3448514 - Q.sdb_2_z) / 0.02505229) * (1 + math.erf(((0.3448514 - Q.sdb_2_z) / 0.02505229) / math.sqrt(2))))
    z += 0.4300103 * (0.0297357 * 0.5 * ((0.6513932 - Q.tau32) / 0.0297357) * (1 + math.erf(((0.6513932 - Q.tau32) / 0.0297357) / math.sqrt(2))))
    z += 1.035619e-05 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (23.09414 * 0.5 * ((Q.sum_pt_top5 - 219.4219) / 23.09414) * (1 + math.erf(((Q.sum_pt_top5 - 219.4219) / 23.09414) / math.sqrt(2))))
    z += 0.01118397 * (3.289335 * 0.5 * ((Q.mass - 123.7928) / 3.289335) * (1 + math.erf(((Q.mass - 123.7928) / 3.289335) / math.sqrt(2))))
    z += -0.02980639 * (16.90428 * 0.5 * ((Q.mass - 164.4374) / 16.90428) * (1 + math.erf(((Q.mass - 164.4374) / 16.90428) / math.sqrt(2))))
    z += -0.001411468 * (6.469856 * 0.5 * ((Q.mass_top50 - 79.27954) / 6.469856) * (1 + math.erf(((Q.mass_top50 - 79.27954) / 6.469856) / math.sqrt(2))))
    z += 0.00317128 * (5.286881 * 0.5 * ((100.4835 - Q.mass) / 5.286881) * (1 + math.erf(((100.4835 - Q.mass) / 5.286881) / math.sqrt(2)))) * (1.518442 * 0.5 * ((1.477152 - Q.lep_ptrel) / 1.518442) * (1 + math.erf(((1.477152 - Q.lep_ptrel) / 1.518442) / math.sqrt(2))))
    z += -0.0003588004 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2))))
    z += -2920.994 * (3.838712e-05 * 0.5 * ((0.000125186 - Q.e3_b2) / 3.838712e-05) * (1 + math.erf(((0.000125186 - Q.e3_b2) / 3.838712e-05) / math.sqrt(2))))
    z += -726.9158 * (3.838712e-05 * 0.5 * ((0.000125186 - Q.e3_b2) / 3.838712e-05) * (1 + math.erf(((0.000125186 - Q.e3_b2) / 3.838712e-05) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_ntag - 0.0) / 1.0) * (1 + math.erf(((Q.dc_ntag - 0.0) / 1.0) / math.sqrt(2))))
    z += 848.1415 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.08904938 * 0.5 * ((Q.jet_charge_k03 - -0.05990128) / 0.08904938) * (1 + math.erf(((Q.jet_charge_k03 - -0.05990128) / 0.08904938) / math.sqrt(2))))
    z += -0.001022999 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2))))
    z += 86.97855 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 0.01019031 * (6.93786 * 0.5 * ((Q.sj3_pair_mass_max - 120.6471) / 6.93786) * (1 + math.erf(((Q.sj3_pair_mass_max - 120.6471) / 6.93786) / math.sqrt(2))))
    z += -4.279844 * (0.0002686389 * 0.5 * ((0.0004126585 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.0004126585 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2))))
    z += 0.08716337 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) / math.sqrt(2))))
    z += -0.01244726 * (10.51039 * 0.5 * ((Q.mres_sd_mass_b2z01 - 69.12133) / 10.51039) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 69.12133) / 10.51039) / math.sqrt(2))))
    z += -0.08825794 * (14.05251 * 0.5 * ((Q.mres_sd_mass_b2z01 - 139.2565) / 14.05251) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 139.2565) / 14.05251) / math.sqrt(2))))
    z += -0.003006388 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.983043) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.983043) / 4.308625) / math.sqrt(2))))
    z += 0.01162086 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (0.03286901 * 0.5 * ((0.4839362 - Q.ak02_dr12) / 0.03286901) * (1 + math.erf(((0.4839362 - Q.ak02_dr12) / 0.03286901) / math.sqrt(2))))
    z += 0.08247592 * (1.0 * 0.5 * ((1.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.001636794 * (35.5 * 0.5 * ((288.0 - Q.n_pairs_kt_above_1) / 35.5) * (1 + math.erf(((288.0 - Q.n_pairs_kt_above_1) / 35.5) / math.sqrt(2))))
    z += 0.008659911 * (4.883139 * 0.5 * ((Q.mass_top50 - 89.68964) / 4.883139) * (1 + math.erf(((Q.mass_top50 - 89.68964) / 4.883139) / math.sqrt(2))))
    z += 0.1191361 * (0.1277784 * 0.5 * ((Q.lund3_lndelta - -1.902701) / 0.1277784) * (1 + math.erf(((Q.lund3_lndelta - -1.902701) / 0.1277784) / math.sqrt(2))))
    z += 0.02239776 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) / math.sqrt(2))))
    z += 7.275457 * (0.01301382 * 0.5 * ((0.1193588 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.1193588 - Q.tau2) / 0.01301382) / math.sqrt(2))))
    z += 0.06901567 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) / math.sqrt(2))))
    z += -0.006386172 * (2.328125 * 0.5 * ((6.402344 - Q.max_abs_dz) / 2.328125) * (1 + math.erf(((6.402344 - Q.max_abs_dz) / 2.328125) / math.sqrt(2))))
    z += 0.3094667 * (0.04000118 * 0.5 * ((0.135772 - Q.tau21) / 0.04000118) * (1 + math.erf(((0.135772 - Q.tau21) / 0.04000118) / math.sqrt(2))))
    z += -9.464399 * (0.01301382 * 0.5 * ((0.1193588 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.1193588 - Q.tau2) / 0.01301382) / math.sqrt(2)))) * (0.05096652 * 0.5 * ((0.125305 - Q.sjq_2_prod_k05) / 0.05096652) * (1 + math.erf(((0.125305 - Q.sjq_2_prod_k05) / 0.05096652) / math.sqrt(2))))
    z += -5.838802 * (0.01301382 * 0.5 * ((0.1193588 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.1193588 - Q.tau2) / 0.01301382) / math.sqrt(2)))) * (0.08093679 * 0.5 * ((0.2801368 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.2801368 - Q.z_displaced5) / 0.08093679) / math.sqrt(2))))
    z += 1.069941 * (0.02188437 * 0.5 * ((Q.z_top30_slots - 0.8942764) / 0.02188437) * (1 + math.erf(((Q.z_top30_slots - 0.8942764) / 0.02188437) / math.sqrt(2))))
    z += 0.006628252 * (4.879578 * 0.5 * ((121.0623 - Q.mass_top30) / 4.879578) * (1 + math.erf(((121.0623 - Q.mass_top30) / 4.879578) / math.sqrt(2))))
    z += 0.3291234 * (0.02188437 * 0.5 * ((Q.z_top30_slots - 0.8942764) / 0.02188437) * (1 + math.erf(((Q.z_top30_slots - 0.8942764) / 0.02188437) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2))))
    z += -0.01368907 * (4.879578 * 0.5 * ((121.0623 - Q.mass_top30) / 4.879578) * (1 + math.erf(((121.0623 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 1.0) / 1.0) * (1 + math.erf(((Q.n_muon - 1.0) / 1.0) / math.sqrt(2))))
    z += 5.935827e-06 * (4.879578 * 0.5 * ((121.0623 - Q.mass_top30) / 4.879578) * (1 + math.erf(((121.0623 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (52.10352 * 0.5 * ((Q.sum_e - 926.2598) / 52.10352) * (1 + math.erf(((Q.sum_e - 926.2598) / 52.10352) / math.sqrt(2))))
    z += -0.003421049 * (4.879578 * 0.5 * ((121.0623 - Q.mass_top30) / 4.879578) * (1 + math.erf(((121.0623 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_min12_n_disp3 - 0.0) / 1.0) * (1 + math.erf(((Q.ak02_min12_n_disp3 - 0.0) / 1.0) / math.sqrt(2))))
    z += 0.0003540898 * (289.0822 * 0.5 * ((447.0873 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((447.0873 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.929175 * 0.5 * ((4.004982 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((4.004982 - Q.jd_3d_5) / 1.929175) / math.sqrt(2))))
    z += -0.001120221 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += -0.4594012 * (0.02298615 * 0.5 * ((0.2740506 - Q.sdb_2_z) / 0.02298615) * (1 + math.erf(((0.2740506 - Q.sdb_2_z) / 0.02298615) / math.sqrt(2))))
    z += 0.01863708 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.2370407 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.2370407 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2))))
    z += 0.0492622 * (1.0 * 0.5 * ((1.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2)))) * (0.1648594 * 0.5 * ((Q.D2 - 0.7107791) / 0.1648594) * (1 + math.erf(((Q.D2 - 0.7107791) / 0.1648594) / math.sqrt(2))))
    z += 0.01183303 * (3.289335 * 0.5 * ((Q.mass - 123.7928) / 3.289335) * (1 + math.erf(((Q.mass - 123.7928) / 3.289335) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.08878111 * (0.1064494 * 0.5 * ((3.634108 - Q.lund_max_lnkt) / 0.1064494) * (1 + math.erf(((3.634108 - Q.lund_max_lnkt) / 0.1064494) / math.sqrt(2))))
    z += -0.1770437 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2))))
    z += 0.1292425 * (1.0 * 0.5 * ((Q.n_sd0_above_3 - 2.0) / 1.0) * (1 + math.erf(((Q.n_sd0_above_3 - 2.0) / 1.0) / math.sqrt(2))))
    z += 0.02146549 * (6.875137 * 0.5 * ((18.80005 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.80005 - Q.mass_displaced3) / 6.875137) / math.sqrt(2))))
    z += 0.004046937 * (19.15449 * 0.5 * ((Q.mres_sd_mass_b0z02 - 134.2023) / 19.15449) * (1 + math.erf(((Q.mres_sd_mass_b0z02 - 134.2023) / 19.15449) / math.sqrt(2))))
    z += -0.00978966 * (20.07181 * 0.5 * ((Q.mres_sd_mass_b1z01 - 176.9461) / 20.07181) * (1 + math.erf(((Q.mres_sd_mass_b1z01 - 176.9461) / 20.07181) / math.sqrt(2))))
    z += -0.01479188 * (5.5191 * 0.5 * ((Q.mres_pruned_mass - 76.10016) / 5.5191) * (1 + math.erf(((Q.mres_pruned_mass - 76.10016) / 5.5191) / math.sqrt(2))))
    z += 0.0272404 * (5.725739 * 0.5 * ((Q.mres_pruned_mass - 124.3145) / 5.725739) * (1 + math.erf(((Q.mres_pruned_mass - 124.3145) / 5.725739) / math.sqrt(2))))
    z += 0.4231749 * (0.02188437 * 0.5 * ((Q.z_top30_slots - 0.8942764) / 0.02188437) * (1 + math.erf(((Q.z_top30_slots - 0.8942764) / 0.02188437) / math.sqrt(2)))) * (0.0219027 * 0.5 * ((0.147768 - Q.dc_3_z) / 0.0219027) * (1 + math.erf(((0.147768 - Q.dc_3_z) / 0.0219027) / math.sqrt(2))))
    z += -0.002765709 * (22.10989 * 0.5 * ((171.3819 - Q.mres_pruned_mass) / 22.10989) * (1 + math.erf(((171.3819 - Q.mres_pruned_mass) / 22.10989) / math.sqrt(2))))
    z += -0.9833364 * (0.0331444 * 0.5 * ((Q.z_displaced5 - 0.1339658) / 0.0331444) * (1 + math.erf(((Q.z_displaced5 - 0.1339658) / 0.0331444) / math.sqrt(2))))
    z += 0.005357973 * (5.119423 * 0.5 * ((Q.mass_top5 - 62.84493) / 5.119423) * (1 + math.erf(((Q.mass_top5 - 62.84493) / 5.119423) / math.sqrt(2))))
    z += -0.00980785 * (5.294312 * 0.5 * ((Q.mres_sd_mass_b2z01 - 102.264) / 5.294312) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 102.264) / 5.294312) / math.sqrt(2))))
    z += -0.07027514 * (6.036668 * 0.5 * ((Q.mres_sd_mass_b2z01 - 76.1529) / 6.036668) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 76.1529) / 6.036668) / math.sqrt(2))))
    z += 0.04895386 * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2))))
    z += -0.04495245 * (3.289335 * 0.5 * ((Q.mass - 123.7928) / 3.289335) * (1 + math.erf(((Q.mass - 123.7928) / 3.289335) / math.sqrt(2)))) * (0.01084662 * 0.5 * ((Q.pz_lnkt4 - 0.02508543) / 0.01084662) * (1 + math.erf(((Q.pz_lnkt4 - 0.02508543) / 0.01084662) / math.sqrt(2))))
    z += -0.02393491 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) / math.sqrt(2)))) * (0.2610212 * 0.5 * ((-0.6337755 - Q.sjq_2_2_k1) / 0.2610212) * (1 + math.erf(((-0.6337755 - Q.sjq_2_2_k1) / 0.2610212) / math.sqrt(2))))
    z += -1.699138 * (0.01301382 * 0.5 * ((0.1193588 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.1193588 - Q.tau2) / 0.01301382) / math.sqrt(2)))) * (0.02453987 * 0.5 * ((-0.0190396 - Q.sjq_2_2_k1) / 0.02453987) * (1 + math.erf(((-0.0190396 - Q.sjq_2_2_k1) / 0.02453987) / math.sqrt(2))))
    z += 0.177843 * (2.077045 * 0.5 * ((1.851735 - Q.sj2_mass2) / 2.077045) * (1 + math.erf(((1.851735 - Q.sj2_mass2) / 2.077045) / math.sqrt(2))))
    z += -0.0004139867 * (3.082358 * 0.5 * ((Q.mass_charged - 58.87188) / 3.082358) * (1 + math.erf(((Q.mass_charged - 58.87188) / 3.082358) / math.sqrt(2))))
    z += 0.009599901 * (2.077045 * 0.5 * ((1.851735 - Q.sj2_mass2) / 2.077045) * (1 + math.erf(((1.851735 - Q.sj2_mass2) / 2.077045) / math.sqrt(2)))) * (0.3881729 * 0.5 * ((Q.mass_2photon - 1.818724) / 0.3881729) * (1 + math.erf(((Q.mass_2photon - 1.818724) / 0.3881729) / math.sqrt(2))))
    z += -0.01478997 * (2.991814 * 0.5 * ((65.88119 - Q.nca_sj4_pair_mass_2nd) / 2.991814) * (1 + math.erf(((65.88119 - Q.nca_sj4_pair_mass_2nd) / 2.991814) / math.sqrt(2))))
    z += -0.007291343 * (3.93783 * 0.5 * ((73.24742 - Q.sj3_pair_mass_max) / 3.93783) * (1 + math.erf(((73.24742 - Q.sj3_pair_mass_max) / 3.93783) / math.sqrt(2))))
    z += 1.694196 * (0.003267481 * 0.5 * ((0.01181938 - Q.z_dr_0p4_up) / 0.003267481) * (1 + math.erf(((0.01181938 - Q.z_dr_0p4_up) / 0.003267481) / math.sqrt(2))))
    z += 4.262558e-06 * (22.10989 * 0.5 * ((171.3819 - Q.mres_pruned_mass) / 22.10989) * (1 + math.erf(((171.3819 - Q.mres_pruned_mass) / 22.10989) / math.sqrt(2)))) * (14.45336 * 0.5 * ((Q.jd_3d_6 - 30.57088) / 14.45336) * (1 + math.erf(((Q.jd_3d_6 - 30.57088) / 14.45336) / math.sqrt(2))))
    z += 0.9125177 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (0.004196167 * 0.5 * ((Q.d0err_10 - 0.02780151) / 0.004196167) * (1 + math.erf(((Q.d0err_10 - 0.02780151) / 0.004196167) / math.sqrt(2))))
    z += 0.01147608 * (10.53967 * 0.5 * ((93.95351 - Q.nca_sj4_pair_mass_2nd) / 10.53967) * (1 + math.erf(((93.95351 - Q.nca_sj4_pair_mass_2nd) / 10.53967) / math.sqrt(2))))
    z += -0.002642598 * (175.2971 * 0.5 * ((84.32402 - Q.sv_2_sd0_sum) / 175.2971) * (1 + math.erf(((84.32402 - Q.sv_2_sd0_sum) / 175.2971) / math.sqrt(2))))
    z += 0.02826021 * (1.0 * 0.5 * ((Q.lepsj_3_n_d3 - 3.0) / 1.0) * (1 + math.erf(((Q.lepsj_3_n_d3 - 3.0) / 1.0) / math.sqrt(2))))
    z += -2.157683 * (0.06488792 * 0.5 * ((Q.dc_split2_dr - 0.2806799) / 0.06488792) * (1 + math.erf(((Q.dc_split2_dr - 0.2806799) / 0.06488792) / math.sqrt(2))))
    z += -0.05078213 * (0.1256484 * 0.5 * ((Q.pair_mean_lnm2 - 2.09826) / 0.1256484) * (1 + math.erf(((Q.pair_mean_lnm2 - 2.09826) / 0.1256484) / math.sqrt(2)))) * (0.1068554 * 0.5 * ((0.4825848 - Q.dr_28) / 0.1068554) * (1 + math.erf(((0.4825848 - Q.dr_28) / 0.1068554) / math.sqrt(2))))
    z += -0.03638036 * (1.0 * 0.5 * ((6.0 - Q.n_neutral_had) / 1.0) * (1 + math.erf(((6.0 - Q.n_neutral_had) / 1.0) / math.sqrt(2))))
    z += -0.005184056 * (14.29796 * 0.5 * ((Q.mass_charged - 113.5019) / 14.29796) * (1 + math.erf(((Q.mass_charged - 113.5019) / 14.29796) / math.sqrt(2))))
    z += 12446.0 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0968) / 6.991655) / math.sqrt(2)))) * (5.3089e-08 * 0.5 * ((7.0969e-08 - Q.e4_b2) / 5.3089e-08) * (1 + math.erf(((7.0969e-08 - Q.e4_b2) / 5.3089e-08) / math.sqrt(2))))
    z += 79.75525 * (5.294312 * 0.5 * ((Q.mres_sd_mass_b2z01 - 102.264) / 5.294312) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 102.264) / 5.294312) / math.sqrt(2)))) * (3.2258e-08 * 0.5 * ((Q.e4_b2 - 1.788e-08) / 3.2258e-08) * (1 + math.erf(((Q.e4_b2 - 1.788e-08) / 3.2258e-08) / math.sqrt(2))))
    z += 5.206591e-05 * (22.10989 * 0.5 * ((171.3819 - Q.mres_pruned_mass) / 22.10989) * (1 + math.erf(((171.3819 - Q.mres_pruned_mass) / 22.10989) / math.sqrt(2)))) * (0.1367866 * 0.5 * ((Q.mass_2photon - 0.2753928) / 0.1367866) * (1 + math.erf(((Q.mass_2photon - 0.2753928) / 0.1367866) / math.sqrt(2))))
    z += -0.0003414924 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (0.0136261 * 0.5 * ((Q.eta_4 - -0.03231812) / 0.0136261) * (1 + math.erf(((Q.eta_4 - -0.03231812) / 0.0136261) / math.sqrt(2))))
    z += 0.03083859 * (4.883139 * 0.5 * ((Q.mass_top50 - 89.68964) / 4.883139) * (1 + math.erf(((Q.mass_top50 - 89.68964) / 4.883139) / math.sqrt(2)))) * (0.01033072 * 0.5 * ((0.01033072 - Q.ak02_3_z_disp3) / 0.01033072) * (1 + math.erf(((0.01033072 - Q.ak02_3_z_disp3) / 0.01033072) / math.sqrt(2))))
    z += -0.04620013 * (9.332349 * 0.5 * ((Q.lnpt_31 - -0.1515499) / 9.332349) * (1 + math.erf(((Q.lnpt_31 - -0.1515499) / 9.332349) / math.sqrt(2))))
    return z


def neuron_98(Q):
    z = 4.990162e-07
    return z


def neuron_99(Q):
    z = -0.00193424
    return z


def neuron_100(Q):
    z = -6.264072e-06
    return z


def neuron_101(Q):
    z = 8.878259e-06
    return z


def neuron_102(Q):
    z = 5.490507e-06
    return z


def neuron_103(Q):
    z = -3.414372e-06
    return z


def neuron_104(Q):
    z = -0.8698674
    z += -0.01042198 * Q.n_charged_pt_above_10
    z += 0.0001021956 * (13.49273 * 0.5 * ((149.0507 - Q.mass) / 13.49273) * (1 + math.erf(((149.0507 - Q.mass) / 13.49273) / math.sqrt(2))))
    z += 0.5408077 * (0.171005 * 0.5 * ((-1.352792 - Q.pair_mean_lndelta) / 0.171005) * (1 + math.erf(((-1.352792 - Q.pair_mean_lndelta) / 0.171005) / math.sqrt(2))))
    z += 0.01991151 * (1.5 * 0.5 * ((18.0 - Q.sjq_2_1_nch) / 1.5) * (1 + math.erf(((18.0 - Q.sjq_2_1_nch) / 1.5) / math.sqrt(2))))
    z += 0.01815355 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.983043) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.983043) / 4.308625) / math.sqrt(2))))
    z += 0.09523086 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 3.208089) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 3.208089) / 1.000219) / math.sqrt(2))))
    z += 0.001448063 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2))))
    z += -0.001098889 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_1_n_d5 - 1.0) / 1.0) * (1 + math.erf(((Q.sjf_2_1_n_d5 - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.003830171 * (12.49365 * 0.5 * ((Q.sj4_pair_mass_max - 128.0079) / 12.49365) * (1 + math.erf(((Q.sj4_pair_mass_max - 128.0079) / 12.49365) / math.sqrt(2))))
    z += 0.0346018 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 3.208089) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 3.208089) / 1.000219) / math.sqrt(2)))) * (0.02843722 * 0.5 * ((0.1014193 - Q.ak02_3_z) / 0.02843722) * (1 + math.erf(((0.1014193 - Q.ak02_3_z) / 0.02843722) / math.sqrt(2))))
    z += -0.9481061 * (0.03993816 * 0.5 * ((0.4680886 - Q.tau32_b2) / 0.03993816) * (1 + math.erf(((0.4680886 - Q.tau32_b2) / 0.03993816) / math.sqrt(2))))
    z += 0.001145615 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (0.05422306 * 0.5 * ((0.8615361 - Q.z_charged) / 0.05422306) * (1 + math.erf(((0.8615361 - Q.z_charged) / 0.05422306) / math.sqrt(2))))
    z += -7.831979e-06 * (13.49273 * 0.5 * ((149.0507 - Q.mass) / 13.49273) * (1 + math.erf(((149.0507 - Q.mass) / 13.49273) / math.sqrt(2)))) * (202.2502 * 0.5 * ((1401.904 - Q.sum_e) / 202.2502) * (1 + math.erf(((1401.904 - Q.sum_e) / 202.2502) / math.sqrt(2))))
    z += -0.1812646 * (0.1006646 * 0.5 * ((1.226724 - Q.D2) / 0.1006646) * (1 + math.erf(((1.226724 - Q.D2) / 0.1006646) / math.sqrt(2))))
    z += 0.004345504 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (0.194914 * 0.5 * ((3.660196 - Q.pair_max_lnkt) / 0.194914) * (1 + math.erf(((3.660196 - Q.pair_max_lnkt) / 0.194914) / math.sqrt(2))))
    z += -0.001195527 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 2.0) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 2.0) / 1.0) / math.sqrt(2))))
    z += -0.2020083 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.983043) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.983043) / 4.308625) / math.sqrt(2)))) * (0.02843722 * 0.5 * ((0.1014193 - Q.ak02_3_z) / 0.02843722) * (1 + math.erf(((0.1014193 - Q.ak02_3_z) / 0.02843722) / math.sqrt(2))))
    z += 0.09167329 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (0.04737588 * 0.5 * ((0.1807551 - Q.ak02_3_z) / 0.04737588) * (1 + math.erf(((0.1807551 - Q.ak02_3_z) / 0.04737588) / math.sqrt(2))))
    z += -3.755558 * (0.009790654 * 0.5 * ((Q.C2_b05 - 0.2234678) / 0.009790654) * (1 + math.erf(((Q.C2_b05 - 0.2234678) / 0.009790654) / math.sqrt(2))))
    z += 0.006076793 * (8.295007 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 83.41384) / 8.295007) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 83.41384) / 8.295007) / math.sqrt(2))))
    z += 0.01501947 * (0.03993816 * 0.5 * ((0.4680886 - Q.tau32_b2) / 0.03993816) * (1 + math.erf(((0.4680886 - Q.tau32_b2) / 0.03993816) / math.sqrt(2)))) * (5.0 * 0.5 * ((17.0 - Q.n_pairs_kt_above_10) / 5.0) * (1 + math.erf(((17.0 - Q.n_pairs_kt_above_10) / 5.0) / math.sqrt(2))))
    z += 1.90311 * (0.01129976 * 0.5 * ((Q.N2 - 0.3357559) / 0.01129976) * (1 + math.erf(((Q.N2 - 0.3357559) / 0.01129976) / math.sqrt(2))))
    z += -0.01323094 * (6.5 * 0.5 * ((41.0 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((41.0 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2))))
    z += 0.04206623 * (6.5 * 0.5 * ((41.0 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((41.0 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2)))) * (0.04359399 * 0.5 * ((0.10729 - Q.sdb_5_z) / 0.04359399) * (1 + math.erf(((0.10729 - Q.sdb_5_z) / 0.04359399) / math.sqrt(2))))
    z += -5.473886 * (0.01171084 * 0.5 * ((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2))))
    z += 0.1151812 * (0.009790654 * 0.5 * ((Q.C2_b05 - 0.2234678) / 0.009790654) * (1 + math.erf(((Q.C2_b05 - 0.2234678) / 0.009790654) / math.sqrt(2)))) * (1.5 * 0.5 * ((6.0 - Q.n_s3d_above_10) / 1.5) * (1 + math.erf(((6.0 - Q.n_s3d_above_10) / 1.5) / math.sqrt(2))))
    z += -0.02026169 * (19.88146 * 0.5 * ((10.98011 - Q.dc_2_jp) / 19.88146) * (1 + math.erf(((10.98011 - Q.dc_2_jp) / 19.88146) / math.sqrt(2))))
    z += 0.458407 * (0.08021924 * 0.5 * ((Q.lund_max_lndelta - -0.615738) / 0.08021924) * (1 + math.erf(((Q.lund_max_lndelta - -0.615738) / 0.08021924) / math.sqrt(2))))
    z += -0.07636955 * (1.566771 * 0.5 * ((Q.mass_displaced3 - 4.41005) / 1.566771) * (1 + math.erf(((Q.mass_displaced3 - 4.41005) / 1.566771) / math.sqrt(2))))
    z += 0.02252103 * (1.5 * 0.5 * ((6.0 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.0 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2))))
    z += 0.4341232 * (0.01268564 * 0.5 * ((0.03385157 - Q.sjf_2_2_z_d3) / 0.01268564) * (1 + math.erf(((0.03385157 - Q.sjf_2_2_z_d3) / 0.01268564) / math.sqrt(2))))
    z += 0.0003959949 * (5.5 * 0.5 * ((58.0 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.0 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (3.876948 * 0.5 * ((6.767937 - Q.mass_2charged) / 3.876948) * (1 + math.erf(((6.767937 - Q.mass_2charged) / 3.876948) / math.sqrt(2))))
    z += -0.004338091 * (13.49273 * 0.5 * ((149.0507 - Q.mass) / 13.49273) * (1 + math.erf(((149.0507 - Q.mass) / 13.49273) / math.sqrt(2)))) * (0.07882828 * 0.5 * ((Q.sjq_2_sumabs_k03 - 0.7416858) / 0.07882828) * (1 + math.erf(((Q.sjq_2_sumabs_k03 - 0.7416858) / 0.07882828) / math.sqrt(2))))
    z += 0.1081719 * (0.01171084 * 0.5 * ((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) * (0.1472193 * 0.5 * ((Q.D2_b2 - 0.2571861) / 0.1472193) * (1 + math.erf(((Q.D2_b2 - 0.2571861) / 0.1472193) / math.sqrt(2))))
    z += 0.00545386 * (6.5 * 0.5 * ((41.0 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((41.0 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -2.145194 * (0.002703061 * 0.5 * ((0.03663959 - Q.tau5) / 0.002703061) * (1 + math.erf(((0.03663959 - Q.tau5) / 0.002703061) / math.sqrt(2))))
    z += -0.001249737 * (13.49273 * 0.5 * ((149.0507 - Q.mass) / 13.49273) * (1 + math.erf(((149.0507 - Q.mass) / 13.49273) / math.sqrt(2)))) * (18.34092 * 0.5 * ((Q.lnpt_65 - -0.07975792) / 18.34092) * (1 + math.erf(((Q.lnpt_65 - -0.07975792) / 18.34092) / math.sqrt(2))))
    z += -0.1021335 * (0.1006646 * 0.5 * ((1.226724 - Q.D2) / 0.1006646) * (1 + math.erf(((1.226724 - Q.D2) / 0.1006646) / math.sqrt(2)))) * (0.1809026 * 0.5 * ((Q.lund1_lndelta - -1.227355) / 0.1809026) * (1 + math.erf(((Q.lund1_lndelta - -1.227355) / 0.1809026) / math.sqrt(2))))
    z += 2.517715 * (0.01268564 * 0.5 * ((0.03385157 - Q.sjf_2_2_z_d3) / 0.01268564) * (1 + math.erf(((0.03385157 - Q.sjf_2_2_z_d3) / 0.01268564) / math.sqrt(2)))) * (0.09214564 * 0.5 * ((Q.dc_1_charge - 0.4233202) / 0.09214564) * (1 + math.erf(((Q.dc_1_charge - 0.4233202) / 0.09214564) / math.sqrt(2))))
    z += -2.925002 * (0.01003572 * 0.5 * ((Q.pz_lnd0 - 0.05066241) / 0.01003572) * (1 + math.erf(((Q.pz_lnd0 - 0.05066241) / 0.01003572) / math.sqrt(2))))
    z += -1.151594 * (0.01003572 * 0.5 * ((Q.pz_lnd0 - 0.05066241) / 0.01003572) * (1 + math.erf(((Q.pz_lnd0 - 0.05066241) / 0.01003572) / math.sqrt(2)))) * (0.00461869 * 0.5 * ((0.008363276 - Q.sjq_2_prod_k1) / 0.00461869) * (1 + math.erf(((0.008363276 - Q.sjq_2_prod_k1) / 0.00461869) / math.sqrt(2))))
    z += 0.5044795 * (1.5 * 0.5 * ((6.0 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.0 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2)))) * (0.005637773 * 0.5 * ((Q.mean_eta2 - 0.03501387) / 0.005637773) * (1 + math.erf(((Q.mean_eta2 - 0.03501387) / 0.005637773) / math.sqrt(2))))
    z += -0.2306232 * (1.0 * 0.5 * ((0.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.0 - Q.n_lepton) / 1.0) / math.sqrt(2))))
    z += 3.130167 * (0.00390151 * 0.5 * ((0.0740332 - Q.M2) / 0.00390151) * (1 + math.erf(((0.0740332 - Q.M2) / 0.00390151) / math.sqrt(2))))
    z += -10.21427 * (0.00390151 * 0.5 * ((0.0740332 - Q.M2) / 0.00390151) * (1 + math.erf(((0.0740332 - Q.M2) / 0.00390151) / math.sqrt(2)))) * (0.04921085 * 0.5 * ((Q.N3_b05 - 0.7045747) / 0.04921085) * (1 + math.erf(((Q.N3_b05 - 0.7045747) / 0.04921085) / math.sqrt(2))))
    z += 0.132689 * (0.01171084 * 0.5 * ((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) * (95.61515 * 0.5 * ((45.92423 - Q.dc_2_jp) / 95.61515) * (1 + math.erf(((45.92423 - Q.dc_2_jp) / 95.61515) / math.sqrt(2))))
    z += 0.1060388 * (1.5 * 0.5 * ((18.0 - Q.sjq_2_1_nch) / 1.5) * (1 + math.erf(((18.0 - Q.sjq_2_1_nch) / 1.5) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.d0err_60 - 0.0) / 1e-06) * (1 + math.erf(((Q.d0err_60 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.02882116 * (0.03993816 * 0.5 * ((0.4680886 - Q.tau32_b2) / 0.03993816) * (1 + math.erf(((0.4680886 - Q.tau32_b2) / 0.03993816) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2))))
    z += -0.8436403 * (0.01282622 * 0.5 * ((0.1010123 - Q.z_neutral_had) / 0.01282622) * (1 + math.erf(((0.1010123 - Q.z_neutral_had) / 0.01282622) / math.sqrt(2))))
    z += -0.0008368171 * (24.5 * 0.5 * ((169.0 - Q.n_pairs_kt_above_1) / 24.5) * (1 + math.erf(((169.0 - Q.n_pairs_kt_above_1) / 24.5) / math.sqrt(2))))
    z += 40.63061 * (0.01171084 * 0.5 * ((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) * (0.01528826 * 0.5 * ((0.106665 - Q.dr_18) / 0.01528826) * (1 + math.erf(((0.106665 - Q.dr_18) / 0.01528826) / math.sqrt(2))))
    z += 0.03008163 * (1.729885 * 0.5 * ((3.928781 - Q.sj2_mass2) / 1.729885) * (1 + math.erf(((3.928781 - Q.sj2_mass2) / 1.729885) / math.sqrt(2))))
    z += 0.02211879 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2))))
    z += -0.015118 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) / math.sqrt(2))))
    z += 0.008389196 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) / math.sqrt(2))))
    z += 0.01395995 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) / math.sqrt(2))))
    z += 0.001978281 * (6.58003 * 0.5 * ((Q.mass_top30 - 126.5007) / 6.58003) * (1 + math.erf(((Q.mass_top30 - 126.5007) / 6.58003) / math.sqrt(2))))
    z += -0.0003529438 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) / math.sqrt(2)))) * (1.842068 * 0.5 * ((Q.lnerel_78 - -18.42068) / 1.842068) * (1 + math.erf(((Q.lnerel_78 - -18.42068) / 1.842068) / math.sqrt(2))))
    z += -240.9054 * (8.295007 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 83.41384) / 8.295007) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 83.41384) / 8.295007) / math.sqrt(2)))) * (1.512315e-07 * 0.5 * ((Q.e4 - 5.29882e-07) / 1.512315e-07) * (1 + math.erf(((Q.e4 - 5.29882e-07) / 1.512315e-07) / math.sqrt(2))))
    z += 9.025843 * (0.009790654 * 0.5 * ((Q.C2_b05 - 0.2234678) / 0.009790654) * (1 + math.erf(((Q.C2_b05 - 0.2234678) / 0.009790654) / math.sqrt(2)))) * (0.006538313 * 0.5 * ((Q.pz_lnkt1 - 0.07470686) / 0.006538313) * (1 + math.erf(((Q.pz_lnkt1 - 0.07470686) / 0.006538313) / math.sqrt(2))))
    z += 0.0147433 * (1.5 * 0.5 * ((6.0 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.0 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2)))) * (0.02700916 * 0.5 * ((0.3709098 - Q.sdb_2_z) / 0.02700916) * (1 + math.erf(((0.3709098 - Q.sdb_2_z) / 0.02700916) / math.sqrt(2))))
    z += 3.867783 * (0.00390151 * 0.5 * ((0.0740332 - Q.M2) / 0.00390151) * (1 + math.erf(((0.0740332 - Q.M2) / 0.00390151) / math.sqrt(2)))) * (0.04063514 * 0.5 * ((Q.dr_21 - 0.0) / 0.04063514) * (1 + math.erf(((Q.dr_21 - 0.0) / 0.04063514) / math.sqrt(2))))
    z += 0.00349388 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 3.208089) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 3.208089) / 1.000219) / math.sqrt(2)))) * (0.04684085 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.3582549) / 0.04684085) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.3582549) / 0.04684085) / math.sqrt(2))))
    z += -0.09902072 * (3.069852 * 0.5 * ((4.516968 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.516968 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2))))
    z += 0.006654516 * (19.4821 * 0.5 * ((30.48974 - Q.sjf_2_2_max3d) / 19.4821) * (1 + math.erf(((30.48974 - Q.sjf_2_2_max3d) / 19.4821) / math.sqrt(2))))
    z += 0.007549068 * (6.58003 * 0.5 * ((Q.mass_top30 - 126.5007) / 6.58003) * (1 + math.erf(((Q.mass_top30 - 126.5007) / 6.58003) / math.sqrt(2)))) * (0.9957982 * 0.5 * ((3.041925 - Q.sip_3d_2) / 0.9957982) * (1 + math.erf(((3.041925 - Q.sip_3d_2) / 0.9957982) / math.sqrt(2))))
    z += -0.0002683589 * (3.069852 * 0.5 * ((4.516968 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.516968 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) * (164.9991 * 0.5 * ((226.3008 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((226.3008 - Q.sip_3d_2) / 164.9991) / math.sqrt(2))))
    z += 0.2320373 * (0.1799842 * 0.5 * ((2.044945 - Q.sjf_2_2_max3d) / 0.1799842) * (1 + math.erf(((2.044945 - Q.sjf_2_2_max3d) / 0.1799842) / math.sqrt(2))))
    z += -0.1049074 * (1.696027 * 0.5 * ((3.32421 - Q.sjf_2_1_max3d) / 1.696027) * (1 + math.erf(((3.32421 - Q.sjf_2_1_max3d) / 1.696027) / math.sqrt(2))))
    z += 0.01647713 * (0.430485 * 0.5 * ((2.508382 - Q.sjf_4_4_max3d) / 0.430485) * (1 + math.erf(((2.508382 - Q.sjf_4_4_max3d) / 0.430485) / math.sqrt(2))))
    z += 0.003949124 * (4.590131 * 0.5 * ((107.2089 - Q.mres_sd_mass_b2z01) / 4.590131) * (1 + math.erf(((107.2089 - Q.mres_sd_mass_b2z01) / 4.590131) / math.sqrt(2))))
    z += 0.01895293 * (1.0 * 0.5 * ((2.0 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((2.0 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2))))
    z += -0.153919 * (1.5 * 0.5 * ((6.0 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.0 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.240977 * (0.01926185 * 0.5 * ((Q.sd_rg - 0.3591864) / 0.01926185) * (1 + math.erf(((Q.sd_rg - 0.3591864) / 0.01926185) / math.sqrt(2))))
    z += -9.270417 * (0.01171084 * 0.5 * ((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.02623363 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) * (0.04330444 * 0.5 * ((-0.1473389 - Q.eta_1) / 0.04330444) * (1 + math.erf(((-0.1473389 - Q.eta_1) / 0.04330444) / math.sqrt(2))))
    z += 47.13166 * (0.002703061 * 0.5 * ((0.03663959 - Q.tau5) / 0.002703061) * (1 + math.erf(((0.03663959 - Q.tau5) / 0.002703061) / math.sqrt(2)))) * (0.001358032 * 0.5 * ((0.03421021 - Q.dzerr_1) / 0.001358032) * (1 + math.erf(((0.03421021 - Q.dzerr_1) / 0.001358032) / math.sqrt(2))))
    z += 0.004968719 * (5.726106 * 0.5 * ((4.183454 - Q.sv_2_sd0_sum) / 5.726106) * (1 + math.erf(((4.183454 - Q.sv_2_sd0_sum) / 5.726106) / math.sqrt(2))))
    z += -1.190413e-05 * (1.5 * 0.5 * ((18.0 - Q.sjq_2_1_nch) / 1.5) * (1 + math.erf(((18.0 - Q.sjq_2_1_nch) / 1.5) / math.sqrt(2)))) * (777.2244 * 0.5 * ((1031.057 - Q.sv_1_sd0_sum) / 777.2244) * (1 + math.erf(((1031.057 - Q.sv_1_sd0_sum) / 777.2244) / math.sqrt(2))))
    z += -0.04307061 * (0.2334666 * 0.5 * ((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.6014774 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2))))
    z += 52.86549 * (0.002048025 * 0.5 * ((0.02098847 - Q.tau5) / 0.002048025) * (1 + math.erf(((0.02098847 - Q.tau5) / 0.002048025) / math.sqrt(2))))
    z += 0.003064251 * (6.58003 * 0.5 * ((Q.mass_top30 - 126.5007) / 6.58003) * (1 + math.erf(((Q.mass_top30 - 126.5007) / 6.58003) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.iselectron_27 - 0.0) / 1e-06) * (1 + math.erf(((Q.iselectron_27 - 0.0) / 1e-06) / math.sqrt(2))))
    z += -1.273433e-06 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) / math.sqrt(2)))) * (316.716 * 0.5 * ((Q.kt2_min12_jp - 271.4622) / 316.716) * (1 + math.erf(((Q.kt2_min12_jp - 271.4622) / 316.716) / math.sqrt(2))))
    z += -0.496796 * (0.05354637 * 0.5 * ((0.7758695 - Q.z_charged_had) / 0.05354637) * (1 + math.erf(((0.7758695 - Q.z_charged_had) / 0.05354637) / math.sqrt(2))))
    z += 0.006250506 * (1.0 * 0.5 * ((Q.sdb_2_n - 9.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 9.0) / 1.0) / math.sqrt(2))))
    z += -0.1010875 * (1.5 * 0.5 * ((Q.sjf_3_1_n_d3 - 4.0) / 1.5) * (1 + math.erf(((Q.sjf_3_1_n_d3 - 4.0) / 1.5) / math.sqrt(2))))
    z += -0.0594338 * (1.0 * 0.5 * ((Q.sdb_2_n - 9.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 9.0) / 1.0) / math.sqrt(2)))) * (0.01417841 * 0.5 * ((0.1049877 - Q.C2_b2) / 0.01417841) * (1 + math.erf(((0.1049877 - Q.C2_b2) / 0.01417841) / math.sqrt(2))))
    z += 0.03685791 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 3.208089) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 3.208089) / 1.000219) / math.sqrt(2)))) * (0.2385164 * 0.5 * ((0.0 - Q.ak02_dr23) / 0.2385164) * (1 + math.erf(((0.0 - Q.ak02_dr23) / 0.2385164) / math.sqrt(2))))
    z += -0.00491997 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.13212) / 19.93276) / math.sqrt(2)))) * (0.05887118 * 0.5 * ((0.3853337 - Q.ak02_dr13) / 0.05887118) * (1 + math.erf(((0.3853337 - Q.ak02_dr13) / 0.05887118) / math.sqrt(2))))
    z += 0.01952191 * (19.88146 * 0.5 * ((10.98011 - Q.dc_2_jp) / 19.88146) * (1 + math.erf(((10.98011 - Q.dc_2_jp) / 19.88146) / math.sqrt(2)))) * (0.02390684 * 0.5 * ((Q.C2_b2 - 0.1404188) / 0.02390684) * (1 + math.erf(((Q.C2_b2 - 0.1404188) / 0.02390684) / math.sqrt(2))))
    z += 0.006790668 * (0.02801514 * 0.5 * ((Q.eta_9 - 0.09516602) / 0.02801514) * (1 + math.erf(((Q.eta_9 - 0.09516602) / 0.02801514) / math.sqrt(2))))
    z += -0.00886032 * (1.0 * 0.5 * ((Q.sdb_2_n - 9.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 9.0) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((3.0 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2))))
    z += 0.5923001 * (0.07989133 * 0.5 * ((Q.N3_b05 - 0.828622) / 0.07989133) * (1 + math.erf(((Q.N3_b05 - 0.828622) / 0.07989133) / math.sqrt(2))))
    z += -0.0148747 * (1.5 * 0.5 * ((6.0 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.0 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.isphoton_40 - 0.0) / 1.0) * (1 + math.erf(((Q.isphoton_40 - 0.0) / 1.0) / math.sqrt(2))))
    z += -0.3531086 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += 0.1267771 * (1.0 * 0.5 * ((1.0 - Q.lepsj_2_n_d3) / 1.0) * (1 + math.erf(((1.0 - Q.lepsj_2_n_d3) / 1.0) / math.sqrt(2))))
    z += 0.00339504 * (11.52979 * 0.5 * ((99.20396 - Q.mass_charged) / 11.52979) * (1 + math.erf(((99.20396 - Q.mass_charged) / 11.52979) / math.sqrt(2))))
    z += 0.001925341 * (1.696027 * 0.5 * ((3.32421 - Q.sjf_2_1_max3d) / 1.696027) * (1 + math.erf(((3.32421 - Q.sjf_2_1_max3d) / 1.696027) / math.sqrt(2)))) * (4.85271 * 0.5 * ((16.06148 - Q.mass_2photon) / 4.85271) * (1 + math.erf(((16.06148 - Q.mass_2photon) / 4.85271) / math.sqrt(2))))
    z += -8.328829e-05 * (567.5171 * 0.5 * ((131.3722 - Q.lepsj_2_maxsd0) / 567.5171) * (1 + math.erf(((131.3722 - Q.lepsj_2_maxsd0) / 567.5171) / math.sqrt(2))))
    z += 0.2368511 * (0.06999318 * 0.5 * ((Q.lund2_lndelta - -1.052328) / 0.06999318) * (1 + math.erf(((Q.lund2_lndelta - -1.052328) / 0.06999318) / math.sqrt(2))))
    return z


def neuron_105(Q):
    z = 2.732381e-05
    return z


def neuron_106(Q):
    z = 2.160933e-05
    return z


def neuron_107(Q):
    z = 1.15856e-05
    return z


def neuron_108(Q):
    z = -4.470698e-06
    return z


def neuron_109(Q):
    z = -5.200149e-07
    return z


def neuron_110(Q):
    z = 3.5796e-06
    return z


def neuron_111(Q):
    z = 3.669612e-06
    return z


def neuron_112(Q):
    z = 5.23782e-06
    return z


def neuron_113(Q):
    z = -2.665927e-05
    return z


def neuron_114(Q):
    z = 4.677414e-06
    return z


def neuron_115(Q):
    z = 0.4244723
    z += 0.002602067 * Q.mass_neutral
    z += -0.5141513 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2))))
    z += -0.7188367 * (0.01467775 * 0.5 * ((0.2243273 - Q.N2_b2) / 0.01467775) * (1 + math.erf(((0.2243273 - Q.N2_b2) / 0.01467775) / math.sqrt(2))))
    z += -0.02445129 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2))))
    z += 0.0004862524 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (7.585662 * 0.5 * ((18.7678 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((18.7678 - Q.lep_ptrel) / 7.585662) / math.sqrt(2))))
    z += 2.006657 * (0.009766867 * 0.5 * ((0.09733903 - Q.tau2) / 0.009766867) * (1 + math.erf(((0.09733903 - Q.tau2) / 0.009766867) / math.sqrt(2))))
    z += -1.083132 * (0.01206829 * 0.5 * ((0.8939856 - Q.tau43) / 0.01206829) * (1 + math.erf(((0.8939856 - Q.tau43) / 0.01206829) / math.sqrt(2))))
    z += -268.4847 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2))))
    z += -2668.995 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (0.06955833 * 0.5 * ((0.3788785 - Q.z_neutral_had) / 0.06955833) * (1 + math.erf(((0.3788785 - Q.z_neutral_had) / 0.06955833) / math.sqrt(2))))
    z += 754.3698 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (0.1133023 * 0.5 * ((1.030977 - Q.pair_mean_lnkt) / 0.1133023) * (1 + math.erf(((1.030977 - Q.pair_mean_lnkt) / 0.1133023) / math.sqrt(2))))
    z += -0.123483 * (0.1723618 * 0.5 * ((Q.ktd_ln_d34 - -9.531553) / 0.1723618) * (1 + math.erf(((Q.ktd_ln_d34 - -9.531553) / 0.1723618) / math.sqrt(2))))
    z += 0.001798104 * (14.63984 * 0.5 * ((414.6641 - Q.sum_pt_top10) / 14.63984) * (1 + math.erf(((414.6641 - Q.sum_pt_top10) / 14.63984) / math.sqrt(2))))
    z += -0.04366416 * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2))))
    z += -0.01341222 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += 0.04152467 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2))))
    z += -0.007867636 * (3.360928 * 0.5 * ((115.7429 - Q.mass_top40) / 3.360928) * (1 + math.erf(((115.7429 - Q.mass_top40) / 3.360928) / math.sqrt(2))))
    z += -0.01059053 * (6.832602 * 0.5 * ((107.8099 - Q.sj4_pair_mass_max) / 6.832602) * (1 + math.erf(((107.8099 - Q.sj4_pair_mass_max) / 6.832602) / math.sqrt(2))))
    z += 5.727824e-05 * (459.2021 * 0.5 * ((804.4651 - Q.sip_3d_2) / 459.2021) * (1 + math.erf(((804.4651 - Q.sip_3d_2) / 459.2021) / math.sqrt(2))))
    z += 11.31577 * (0.008001329 * 0.5 * ((0.120439 - Q.M2) / 0.008001329) * (1 + math.erf(((0.120439 - Q.M2) / 0.008001329) / math.sqrt(2))))
    z += 1.73275 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2))))
    z += -1.821183 * (0.009691066 * 0.5 * ((0.1117947 - Q.dr_0) / 0.009691066) * (1 + math.erf(((0.1117947 - Q.dr_0) / 0.009691066) / math.sqrt(2))))
    z += 1.476889 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.02439358 * 0.5 * ((0.1845735 - Q.sj4_dr_min) / 0.02439358) * (1 + math.erf(((0.1845735 - Q.sj4_dr_min) / 0.02439358) / math.sqrt(2))))
    z += -2.171318 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.008350939 * 0.5 * ((0.8870464 - Q.tau54) / 0.008350939) * (1 + math.erf(((0.8870464 - Q.tau54) / 0.008350939) / math.sqrt(2))))
    z += -44.24342 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (0.002122726 * 0.5 * ((Q.sjf_2_2_z_d3 - 0.0) / 0.002122726) * (1 + math.erf(((Q.sjf_2_2_z_d3 - 0.0) / 0.002122726) / math.sqrt(2))))
    z += 0.002658486 * (3.20498 * 0.5 * ((34.02385 - Q.mass_neutral) / 3.20498) * (1 + math.erf(((34.02385 - Q.mass_neutral) / 3.20498) / math.sqrt(2))))
    z += 0.007122886 * (3.764443 * 0.5 * ((125.4927 - Q.mass_top50) / 3.764443) * (1 + math.erf(((125.4927 - Q.mass_top50) / 3.764443) / math.sqrt(2))))
    z += -0.3366343 * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2))))
    z += 10.01584 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.0331444 * 0.5 * ((Q.z_displaced5 - 0.1339658) / 0.0331444) * (1 + math.erf(((Q.z_displaced5 - 0.1339658) / 0.0331444) / math.sqrt(2))))
    z += -0.0008795091 * (6.832602 * 0.5 * ((107.8099 - Q.sj4_pair_mass_max) / 6.832602) * (1 + math.erf(((107.8099 - Q.sj4_pair_mass_max) / 6.832602) / math.sqrt(2)))) * (5.37501 * 0.5 * ((9.356714 - Q.jd_3d_6) / 5.37501) * (1 + math.erf(((9.356714 - Q.jd_3d_6) / 5.37501) / math.sqrt(2))))
    z += -0.006634477 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.9748591 * 0.5 * ((Q.sj4_pair_mass_min - 8.513203) / 0.9748591) * (1 + math.erf(((Q.sj4_pair_mass_min - 8.513203) / 0.9748591) / math.sqrt(2))))
    z += 0.2916228 * (0.1312538 * 0.5 * ((Q.jet_abs_eta - 1.323111) / 0.1312538) * (1 + math.erf(((Q.jet_abs_eta - 1.323111) / 0.1312538) / math.sqrt(2))))
    z += 0.0006910033 * (3.360928 * 0.5 * ((115.7429 - Q.mass_top40) / 3.360928) * (1 + math.erf(((115.7429 - Q.mass_top40) / 3.360928) / math.sqrt(2)))) * (10.60708 * 0.5 * ((16.11752 - Q.jd_3d_6) / 10.60708) * (1 + math.erf(((16.11752 - Q.jd_3d_6) / 10.60708) / math.sqrt(2))))
    z += 0.0007669092 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (248.3135 * 0.5 * ((175.9957 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((175.9957 - Q.sip_3d_3) / 248.3135) / math.sqrt(2))))
    z += 0.2335982 * (0.01467775 * 0.5 * ((0.2243273 - Q.N2_b2) / 0.01467775) * (1 + math.erf(((0.2243273 - Q.N2_b2) / 0.01467775) / math.sqrt(2)))) * (5.37501 * 0.5 * ((9.356714 - Q.jd_3d_6) / 5.37501) * (1 + math.erf(((9.356714 - Q.jd_3d_6) / 5.37501) / math.sqrt(2))))
    z += -0.1166798 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_n - 1.0) / 1.0) * (1 + math.erf(((Q.ak02_n - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.0002786854 * (4.883139 * 0.5 * ((89.68964 - Q.mass_top50) / 4.883139) * (1 + math.erf(((89.68964 - Q.mass_top50) / 4.883139) / math.sqrt(2))))
    z += 0.06238142 * (3.876948 * 0.5 * ((6.767937 - Q.mass_2charged) / 3.876948) * (1 + math.erf(((6.767937 - Q.mass_2charged) / 3.876948) / math.sqrt(2))))
    z += -0.2833579 * (0.8395288 * 0.5 * ((1.839882 - Q.mass_2charged) / 0.8395288) * (1 + math.erf(((1.839882 - Q.mass_2charged) / 0.8395288) / math.sqrt(2))))
    z += -0.193115 * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) * (0.03352269 * 0.5 * ((0.6856395 - Q.z_charged_had) / 0.03352269) * (1 + math.erf(((0.6856395 - Q.z_charged_had) / 0.03352269) / math.sqrt(2))))
    z += -0.0092915 * (4.06561 * 0.5 * ((24.4079 - Q.mass_2charged) / 4.06561) * (1 + math.erf(((24.4079 - Q.mass_2charged) / 4.06561) / math.sqrt(2))))
    z += 0.4124673 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.1093626 * 0.5 * ((0.3699565 - Q.sv_2_dr) / 0.1093626) * (1 + math.erf(((0.3699565 - Q.sv_2_dr) / 0.1093626) / math.sqrt(2))))
    z += -14.52503 * (0.006239604 * 0.5 * ((0.02148541 - Q.lam2) / 0.006239604) * (1 + math.erf(((0.02148541 - Q.lam2) / 0.006239604) / math.sqrt(2))))
    z += -0.01414744 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2))))
    z += 80.02731 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2))))
    z += -0.0001283408 * (459.2021 * 0.5 * ((804.4651 - Q.sip_3d_2) / 459.2021) * (1 + math.erf(((804.4651 - Q.sip_3d_2) / 459.2021) / math.sqrt(2)))) * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2))))
    z += 0.007941485 * (3.360928 * 0.5 * ((115.7429 - Q.mass_top40) / 3.360928) * (1 + math.erf(((115.7429 - Q.mass_top40) / 3.360928) / math.sqrt(2)))) * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += -0.006394281 * (3.764443 * 0.5 * ((125.4927 - Q.mass_top50) / 3.764443) * (1 + math.erf(((125.4927 - Q.mass_top50) / 3.764443) / math.sqrt(2)))) * (0.01637522 * 0.5 * ((0.1243111 - Q.pz_lnd1) / 0.01637522) * (1 + math.erf(((0.1243111 - Q.pz_lnd1) / 0.01637522) / math.sqrt(2))))
    z += 333.375 * (0.0003783019 * 0.5 * ((0.0007909605 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.0007909605 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 1.0) / 1.0) * (1 + math.erf(((Q.n_muon - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.04054579 * (4.06561 * 0.5 * ((24.4079 - Q.mass_2charged) / 4.06561) * (1 + math.erf(((24.4079 - Q.mass_2charged) / 4.06561) / math.sqrt(2)))) * (0.01273628 * 0.5 * ((Q.sdb_5_z - 0.04013001) / 0.01273628) * (1 + math.erf(((Q.sdb_5_z - 0.04013001) / 0.01273628) / math.sqrt(2))))
    z += -1.099346 * (0.009766867 * 0.5 * ((0.09733903 - Q.tau2) / 0.009766867) * (1 + math.erf(((0.09733903 - Q.tau2) / 0.009766867) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_1_n_disp3 - 1.0) / 1.0) * (1 + math.erf(((Q.ak02_1_n_disp3 - 1.0) / 1.0) / math.sqrt(2))))
    z += -0.001367438 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.2411197 * 0.5 * ((Q.jd_3d_6 - 0.9115584) / 0.2411197) * (1 + math.erf(((Q.jd_3d_6 - 0.9115584) / 0.2411197) / math.sqrt(2))))
    z += 2.321322e-05 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (0.1300849 * 0.5 * ((Q.dc_tag_2nd - 0.0) / 0.1300849) * (1 + math.erf(((Q.dc_tag_2nd - 0.0) / 0.1300849) / math.sqrt(2))))
    z += 3.79317 * (0.0164642 * 0.5 * ((0.6162845 - Q.nca_sj4_pairmax_over_mass) / 0.0164642) * (1 + math.erf(((0.6162845 - Q.nca_sj4_pairmax_over_mass) / 0.0164642) / math.sqrt(2))))
    z += 0.2604127 * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) * (1.0 * 0.5 * ((4.0 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((4.0 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2))))
    z += -0.1846313 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.0001773114 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (12.86093 * 0.5 * ((Q.dc_split2_mass - 17.56529) / 12.86093) * (1 + math.erf(((Q.dc_split2_mass - 17.56529) / 12.86093) / math.sqrt(2))))
    z += 0.8734938 * (0.1723618 * 0.5 * ((Q.ktd_ln_d34 - -9.531553) / 0.1723618) * (1 + math.erf(((Q.ktd_ln_d34 - -9.531553) / 0.1723618) / math.sqrt(2)))) * (0.01029976 * 0.5 * ((0.102661 - Q.pz_lnkt3) / 0.01029976) * (1 + math.erf(((0.102661 - Q.pz_lnkt3) / 0.01029976) / math.sqrt(2))))
    z += -0.003232613 * (6.622935 * 0.5 * ((80.33914 - Q.dc_split1_kt) / 6.622935) * (1 + math.erf(((80.33914 - Q.dc_split1_kt) / 6.622935) / math.sqrt(2))))
    z += 0.001354883 * (6.832602 * 0.5 * ((107.8099 - Q.sj4_pair_mass_max) / 6.832602) * (1 + math.erf(((107.8099 - Q.sj4_pair_mass_max) / 6.832602) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_3_3_n_d3 - 3.0) / 1.0) * (1 + math.erf(((Q.sjf_3_3_n_d3 - 3.0) / 1.0) / math.sqrt(2))))
    z += 0.0387769 * (12.30925 * 0.5 * ((39.09615 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.09615 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2))))
    z += 0.01430437 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2))))
    z += -0.005138066 * (5.468702 * 0.5 * ((Q.sd_mass - 88.81751) / 5.468702) * (1 + math.erf(((Q.sd_mass - 88.81751) / 5.468702) / math.sqrt(2))))
    z += 0.006642329 * (3.260893 * 0.5 * ((86.78877 - Q.mass_top20) / 3.260893) * (1 + math.erf(((86.78877 - Q.mass_top20) / 3.260893) / math.sqrt(2))))
    z += 0.9607614 * (0.02680828 * 0.5 * ((Q.dc_1_z - 0.6122903) / 0.02680828) * (1 + math.erf(((Q.dc_1_z - 0.6122903) / 0.02680828) / math.sqrt(2))))
    z += -0.988553 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += -0.0007930472 * (13.56554 * 0.5 * ((133.1648 - Q.mass_top15) / 13.56554) * (1 + math.erf(((133.1648 - Q.mass_top15) / 13.56554) / math.sqrt(2))))
    z += 5.725185 * (0.008548367 * 0.5 * ((Q.N2_b05 - 0.4353632) / 0.008548367) * (1 + math.erf(((Q.N2_b05 - 0.4353632) / 0.008548367) / math.sqrt(2))))
    z += -0.009553789 * (5.413174 * 0.5 * ((114.7848 - Q.sj3_pair_mass_max) / 5.413174) * (1 + math.erf(((114.7848 - Q.sj3_pair_mass_max) / 5.413174) / math.sqrt(2))))
    z += 0.05759354 * (0.4768076 * 0.5 * ((Q.ktd_ln_d34 - -7.075141) / 0.4768076) * (1 + math.erf(((Q.ktd_ln_d34 - -7.075141) / 0.4768076) / math.sqrt(2))))
    z += -10.1915 * (0.003950899 * 0.5 * ((Q.tau4 - 0.04996) / 0.003950899) * (1 + math.erf(((Q.tau4 - 0.04996) / 0.003950899) / math.sqrt(2))))
    z += -0.9478902 * (0.05566634 * 0.5 * ((0.3093201 - Q.z_neutral_had) / 0.05566634) * (1 + math.erf(((0.3093201 - Q.z_neutral_had) / 0.05566634) / math.sqrt(2))))
    z += 1.062549 * (0.01559439 * 0.5 * ((0.1713451 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.1713451 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -1.979592 * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.isphoton_53) / 1.0) * (1 + math.erf(((0.0 - Q.isphoton_53) / 1.0) / math.sqrt(2))))
    z += -5.117058 * (0.01467775 * 0.5 * ((0.2243273 - Q.N2_b2) / 0.01467775) * (1 + math.erf(((0.2243273 - Q.N2_b2) / 0.01467775) / math.sqrt(2)))) * (0.03500366 * 0.5 * ((Q.d0err_52 - 0.03500366) / 0.03500366) * (1 + math.erf(((Q.d0err_52 - 0.03500366) / 0.03500366) / math.sqrt(2))))
    z += 0.04491087 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (0.02572129 * 0.5 * ((0.9433644 - Q.tau43_b2) / 0.02572129) * (1 + math.erf(((0.9433644 - Q.tau43_b2) / 0.02572129) / math.sqrt(2))))
    z += 0.002376986 * (18.98174 * 0.5 * ((14.38858 - Q.lep_iso) / 18.98174) * (1 + math.erf(((14.38858 - Q.lep_iso) / 18.98174) / math.sqrt(2))))
    z += 1.816222 * (0.8395288 * 0.5 * ((1.839882 - Q.mass_2charged) / 0.8395288) * (1 + math.erf(((1.839882 - Q.mass_2charged) / 0.8395288) / math.sqrt(2)))) * (0.04514027 * 0.5 * ((0.1015913 - Q.sjf_4_2_z_d3) / 0.04514027) * (1 + math.erf(((0.1015913 - Q.sjf_4_2_z_d3) / 0.04514027) / math.sqrt(2))))
    z += -0.002248619 * (9.321454 * 0.5 * ((Q.mass_displaced5 - 21.12768) / 9.321454) * (1 + math.erf(((Q.mass_displaced5 - 21.12768) / 9.321454) / math.sqrt(2))))
    z += 0.002701476 * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2))))
    z += -0.003587343 * (36.85684 * 0.5 * ((77.03428 - Q.sv_1_sd0_sum) / 36.85684) * (1 + math.erf(((77.03428 - Q.sv_1_sd0_sum) / 36.85684) / math.sqrt(2))))
    z += 2209.531 * (2.202631e-05 * 0.5 * ((8.17233e-05 - Q.ecf_g42) / 2.202631e-05) * (1 + math.erf(((8.17233e-05 - Q.ecf_g42) / 2.202631e-05) / math.sqrt(2))))
    z += 2.761768 * (0.008554338 * 0.5 * ((0.07722421 - Q.pz_lnd0) / 0.008554338) * (1 + math.erf(((0.07722421 - Q.pz_lnd0) / 0.008554338) / math.sqrt(2))))
    z += 0.7353103 * (0.0321369 * 0.5 * ((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.853812 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) * (0.1568179 * 0.5 * ((Q.sjq_2_sumabs_k03 - 1.261566) / 0.1568179) * (1 + math.erf(((Q.sjq_2_sumabs_k03 - 1.261566) / 0.1568179) / math.sqrt(2))))
    z += -0.02344532 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.0) / 1.0) / math.sqrt(2)))) * (0.07359024 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.4723988) / 0.07359024) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.4723988) / 0.07359024) / math.sqrt(2))))
    z += 0.2171282 * (0.09724171 * 0.5 * ((3.234227 - Q.pair_max_lnkt) / 0.09724171) * (1 + math.erf(((3.234227 - Q.pair_max_lnkt) / 0.09724171) / math.sqrt(2))))
    z += 1.208248 * (0.05566634 * 0.5 * ((0.3093201 - Q.z_neutral_had) / 0.05566634) * (1 + math.erf(((0.3093201 - Q.z_neutral_had) / 0.05566634) / math.sqrt(2)))) * (0.06273278 * 0.5 * ((Q.dr12 - 0.2116473) / 0.06273278) * (1 + math.erf(((Q.dr12 - 0.2116473) / 0.06273278) / math.sqrt(2))))
    z += -7.505841 * (0.01854575 * 0.5 * ((Q.N2 - 0.2250586) / 0.01854575) * (1 + math.erf(((Q.N2 - 0.2250586) / 0.01854575) / math.sqrt(2))))
    z += 1.139975 * (0.09740396 * 0.5 * ((1.325034 - Q.D2) / 0.09740396) * (1 + math.erf(((1.325034 - Q.D2) / 0.09740396) / math.sqrt(2))))
    z += 0.00725232 * (2.35787 * 0.5 * ((Q.sj3_pair_mass_min - 26.64603) / 2.35787) * (1 + math.erf(((Q.sj3_pair_mass_min - 26.64603) / 2.35787) / math.sqrt(2))))
    z += 2.619992 * (0.04122765 * 0.5 * ((Q.N2_b05 - 0.302405) / 0.04122765) * (1 + math.erf(((Q.N2_b05 - 0.302405) / 0.04122765) / math.sqrt(2))))
    z += -0.0003357649 * (48.0 * 0.5 * ((411.0 - Q.n_pairs_kt_above_1) / 48.0) * (1 + math.erf(((411.0 - Q.n_pairs_kt_above_1) / 48.0) / math.sqrt(2))))
    z += 0.0009027237 * (2.952984 * 0.5 * ((75.78208 - Q.sj4_pair_mass_max) / 2.952984) * (1 + math.erf(((75.78208 - Q.sj4_pair_mass_max) / 2.952984) / math.sqrt(2))))
    z += -0.06655711 * (0.01206829 * 0.5 * ((0.8939856 - Q.tau43) / 0.01206829) * (1 + math.erf(((0.8939856 - Q.tau43) / 0.01206829) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dc_mass_disp_2nd - 0.0) / 1e-06) * (1 + math.erf(((Q.dc_mass_disp_2nd - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.01853995 * (1.721646 * 0.5 * ((5.763861 - Q.mass_2photon) / 1.721646) * (1 + math.erf(((5.763861 - Q.mass_2photon) / 1.721646) / math.sqrt(2))))
    z += -0.02299751 * (1.729885 * 0.5 * ((3.928781 - Q.sj2_mass2) / 1.729885) * (1 + math.erf(((3.928781 - Q.sj2_mass2) / 1.729885) / math.sqrt(2))))
    z += 0.5027834 * (1.0 * 0.5 * ((1.0 - Q.sjq_3_3_nch) / 1.0) * (1 + math.erf(((1.0 - Q.sjq_3_3_nch) / 1.0) / math.sqrt(2))))
    z += 12.92901 * (0.03292552 * 0.5 * ((0.057042 - Q.sj2_zsoft) / 0.03292552) * (1 + math.erf(((0.057042 - Q.sj2_zsoft) / 0.03292552) / math.sqrt(2))))
    z += -0.03747442 * (0.6262913 * 0.5 * ((5.460258 - Q.sj3_mass3) / 0.6262913) * (1 + math.erf(((5.460258 - Q.sj3_mass3) / 0.6262913) / math.sqrt(2))))
    z += 0.01569949 * (5.110145 * 0.5 * ((131.3917 - Q.mass) / 5.110145) * (1 + math.erf(((131.3917 - Q.mass) / 5.110145) / math.sqrt(2))))
    return z


def neuron_116(Q):
    z = -3.399297e-07
    return z


def neuron_117(Q):
    z = 2.862133e-06
    return z


def neuron_118(Q):
    z = -8.233216e-06
    return z


def neuron_119(Q):
    z = -6.336853e-07
    return z


def neuron_120(Q):
    z = 0.5242535
    z += -0.004375675 * Q.lund3_lnkt
    z += 19.12721 * Q.e3_b05
    z += -0.1514206 * (2.752946 * 0.5 * ((3.53503 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.53503 - Q.lep_ptrel) / 2.752946) / math.sqrt(2))))
    z += -0.006812905 * (8.0 * 0.5 * ((62.0 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.0 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2))))
    z += -0.0343473 * (2.752946 * 0.5 * ((3.53503 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.53503 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += -0.007764927 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2))))
    z += -0.528416 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2))))
    z += 0.6762132 * (0.02757255 * 0.5 * ((0.6800935 - Q.tau32) / 0.02757255) * (1 + math.erf(((0.6800935 - Q.tau32) / 0.02757255) / math.sqrt(2))))
    z += -0.0198456 * (1.0 * 0.5 * ((3.0 - Q.n_sd0_above_5) / 1.0) * (1 + math.erf(((3.0 - Q.n_sd0_above_5) / 1.0) / math.sqrt(2))))
    z += -0.005923831 * (21.5 * 0.5 * ((80.0 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.0 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += -0.01656839 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (9.37985 * 0.5 * ((76.15079 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((76.15079 - Q.mass_neutral) / 9.37985) / math.sqrt(2))))
    z += 0.02782988 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (4.4703e-08 * 0.5 * ((Q.mass_displaced3 - 0.0) / 4.4703e-08) * (1 + math.erf(((Q.mass_displaced3 - 0.0) / 4.4703e-08) / math.sqrt(2))))
    z += -0.0446681 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2))))
    z += 0.001481414 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2))))
    z += -0.004346708 * (10.67504 * 0.5 * ((Q.mass_top10 - 103.4976) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 103.4976) / 10.67504) / math.sqrt(2))))
    z += 0.000319266 * (2.752946 * 0.5 * ((3.53503 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.53503 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (401.9953 * 0.5 * ((577.991 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((577.991 - Q.sip_3d_3) / 401.9953) / math.sqrt(2))))
    z += -0.001035633 * (8.0 * 0.5 * ((62.0 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.0 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) * (0.08055962 * 0.5 * ((0.2710171 - Q.sjf_2_1_z_d3) / 0.08055962) * (1 + math.erf(((0.2710171 - Q.sjf_2_1_z_d3) / 0.08055962) / math.sqrt(2))))
    z += -0.2482837 * (0.7185011 * 0.5 * ((Q.pair_mean_lnm2 - 4.787589) / 0.7185011) * (1 + math.erf(((Q.pair_mean_lnm2 - 4.787589) / 0.7185011) / math.sqrt(2))))
    z += 2.162898 * (0.008162106 * 0.5 * ((0.1589878 - Q.pz_lnkt0) / 0.008162106) * (1 + math.erf(((0.1589878 - Q.pz_lnkt0) / 0.008162106) / math.sqrt(2))))
    z += 0.05822356 * (1.0 * 0.5 * ((3.0 - Q.n_sd0_above_5) / 1.0) * (1 + math.erf(((3.0 - Q.n_sd0_above_5) / 1.0) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.01047829 * (8.0 * 0.5 * ((62.0 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.0 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) * (0.0792551 * 0.5 * ((0.1822601 - Q.lepsj_2_dr) / 0.0792551) * (1 + math.erf(((0.1822601 - Q.lepsj_2_dr) / 0.0792551) / math.sqrt(2))))
    z += -0.007368078 * (1.5 * 0.5 * ((11.0 - Q.n_pairs_kt_above_10) / 1.5) * (1 + math.erf(((11.0 - Q.n_pairs_kt_above_10) / 1.5) / math.sqrt(2))))
    z += 0.006150065 * (2.752946 * 0.5 * ((3.53503 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.53503 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (2.0 * 0.5 * ((Q.n_photon - 5.0) / 2.0) * (1 + math.erf(((Q.n_photon - 5.0) / 2.0) / math.sqrt(2))))
    z += 0.002523168 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (1.5 * 0.5 * ((16.0 - Q.sdb_2_n) / 1.5) * (1 + math.erf(((16.0 - Q.sdb_2_n) / 1.5) / math.sqrt(2))))
    z += 0.1201624 * (2.41177 * 0.5 * ((2.907433 - Q.lep_iso) / 2.41177) * (1 + math.erf(((2.907433 - Q.lep_iso) / 2.41177) / math.sqrt(2))))
    z += -0.1155598 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += -0.007693043 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (1.387908 * 0.5 * ((3.151933 - Q.jd_3d_6) / 1.387908) * (1 + math.erf(((3.151933 - Q.jd_3d_6) / 1.387908) / math.sqrt(2))))
    z += 0.03551013 * (2.41177 * 0.5 * ((2.907433 - Q.lep_iso) / 2.41177) * (1 + math.erf(((2.907433 - Q.lep_iso) / 2.41177) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += -5.71296 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2))))
    z += -0.08321059 * (0.3071165 * 0.5 * ((Q.lund3_lndelta - -2.4279) / 0.3071165) * (1 + math.erf(((Q.lund3_lndelta - -2.4279) / 0.3071165) / math.sqrt(2))))
    z += 0.02441924 * (8.0 * 0.5 * ((62.0 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.0 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 5.071749 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.01055312 * (3.277554 * 0.5 * ((68.20711 - Q.mass_charged) / 3.277554) * (1 + math.erf(((68.20711 - Q.mass_charged) / 3.277554) / math.sqrt(2))))
    z += 0.003595922 * (3.277554 * 0.5 * ((68.20711 - Q.mass_charged) / 3.277554) * (1 + math.erf(((68.20711 - Q.mass_charged) / 3.277554) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += -1.047052 * (0.009387389 * 0.5 * ((0.03277088 - Q.pz_lnd2) / 0.009387389) * (1 + math.erf(((0.03277088 - Q.pz_lnd2) / 0.009387389) / math.sqrt(2))))
    z += -1.514297 * (1.0 * 0.5 * ((0.0 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2))))
    z += 1.19784 * (1.0 * 0.5 * ((0.0 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2)))) * (0.0812128 * 0.5 * ((Q.sjq_2_prod_k05 - -0.2405169) / 0.0812128) * (1 + math.erf(((Q.sjq_2_prod_k05 - -0.2405169) / 0.0812128) / math.sqrt(2))))
    z += -0.01818441 * (10.67504 * 0.5 * ((Q.mass_top10 - 103.4976) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 103.4976) / 10.67504) / math.sqrt(2)))) * (0.04703017 * 0.5 * ((0.5081296 - Q.planar_flow) / 0.04703017) * (1 + math.erf(((0.5081296 - Q.planar_flow) / 0.04703017) / math.sqrt(2))))
    z += 0.5433267 * (0.008162106 * 0.5 * ((0.1589878 - Q.pz_lnkt0) / 0.008162106) * (1 + math.erf(((0.1589878 - Q.pz_lnkt0) / 0.008162106) / math.sqrt(2)))) * (0.04133456 * 0.5 * ((Q.sj3_dr13 - 0.1541) / 0.04133456) * (1 + math.erf(((Q.sj3_dr13 - 0.1541) / 0.04133456) / math.sqrt(2))))
    z += 0.5128508 * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) / math.sqrt(2))))
    z += -0.0009534307 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.007382626 * 0.5 * ((Q.pz_lnd2 - 0.008992646) / 0.007382626) * (1 + math.erf(((Q.pz_lnd2 - 0.008992646) / 0.007382626) / math.sqrt(2))))
    z += -6.353035 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) / math.sqrt(2))))
    z += 0.01084119 * (9.37985 * 0.5 * ((76.15079 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((76.15079 - Q.mass_neutral) / 9.37985) / math.sqrt(2))))
    z += -0.02304058 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2))))
    z += 0.02982756 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.n_electron) / 1.0) * (1 + math.erf(((1.0 - Q.n_electron) / 1.0) / math.sqrt(2))))
    z += 0.0004056015 * (5.416144 * 0.5 * ((Q.sj4_pair_mass_max - 101.849) / 5.416144) * (1 + math.erf(((Q.sj4_pair_mass_max - 101.849) / 5.416144) / math.sqrt(2))))
    z += 0.673035 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.09432096 * 0.5 * ((-0.2912597 - Q.sjq_3_3_k1) / 0.09432096) * (1 + math.erf(((-0.2912597 - Q.sjq_3_3_k1) / 0.09432096) / math.sqrt(2))))
    z += -1.749922 * (0.03214999 * 0.5 * ((0.3436326 - Q.N2_b05) / 0.03214999) * (1 + math.erf(((0.3436326 - Q.N2_b05) / 0.03214999) / math.sqrt(2))))
    z += -0.01467624 * (3.529599 * 0.5 * ((109.3251 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.3251 - Q.mass_top30) / 3.529599) / math.sqrt(2))))
    z += 0.002460207 * (3.529599 * 0.5 * ((109.3251 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.3251 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += -1.89753 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.367501 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.367501 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += -0.004244334 * (3.277554 * 0.5 * ((68.20711 - Q.mass_charged) / 3.277554) * (1 + math.erf(((68.20711 - Q.mass_charged) / 3.277554) / math.sqrt(2)))) * (0.04703017 * 0.5 * ((0.5081296 - Q.planar_flow) / 0.04703017) * (1 + math.erf(((0.5081296 - Q.planar_flow) / 0.04703017) / math.sqrt(2))))
    z += -0.1671252 * (3.529599 * 0.5 * ((109.3251 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.3251 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (0.005973664 * 0.5 * ((Q.sv_2_z - 0.01148432) / 0.005973664) * (1 + math.erf(((Q.sv_2_z - 0.01148432) / 0.005973664) / math.sqrt(2))))
    z += 122.7178 * (0.003235318 * 0.5 * ((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) * (1 + math.erf(((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) / math.sqrt(2))))
    z += 0.002090411 * (1.5 * 0.5 * ((7.0 - Q.n_dr_0p1_0p2) / 1.5) * (1 + math.erf(((7.0 - Q.n_dr_0p1_0p2) / 1.5) / math.sqrt(2))))
    z += 2.983591 * (0.03060878 * 0.5 * ((0.06573337 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.06573337 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (0.2385164 * 0.5 * ((Q.ak02_dr23 - 0.0) / 0.2385164) * (1 + math.erf(((Q.ak02_dr23 - 0.0) / 0.2385164) / math.sqrt(2))))
    z += -0.001671342 * (13.56554 * 0.5 * ((Q.mass_top15 - 133.1648) / 13.56554) * (1 + math.erf(((Q.mass_top15 - 133.1648) / 13.56554) / math.sqrt(2))))
    z += 0.05743276 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.kt2_2_n_disp3 - 0.0) / 1.0) * (1 + math.erf(((Q.kt2_2_n_disp3 - 0.0) / 1.0) / math.sqrt(2))))
    z += -7.767129 * (0.00189604 * 0.5 * ((0.01394245 - Q.dr_min_012) / 0.00189604) * (1 + math.erf(((0.01394245 - Q.dr_min_012) / 0.00189604) / math.sqrt(2))))
    z += 0.07177711 * (3.529599 * 0.5 * ((109.3251 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.3251 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (0.0139093 * 0.5 * ((0.1271955 - Q.z_neutral_had) / 0.0139093) * (1 + math.erf(((0.1271955 - Q.z_neutral_had) / 0.0139093) / math.sqrt(2))))
    z += -0.01998456 * (4.146938 * 0.5 * ((110.2019 - Q.mass) / 4.146938) * (1 + math.erf(((110.2019 - Q.mass) / 4.146938) / math.sqrt(2))))
    z += -0.009062522 * (9.37985 * 0.5 * ((76.15079 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((76.15079 - Q.mass_neutral) / 9.37985) / math.sqrt(2)))) * (0.03369438 * 0.5 * ((Q.sj3_dr_max - 0.5700825) / 0.03369438) * (1 + math.erf(((Q.sj3_dr_max - 0.5700825) / 0.03369438) / math.sqrt(2))))
    z += 0.002774234 * (5.061224 * 0.5 * ((90.08945 - Q.mass) / 5.061224) * (1 + math.erf(((90.08945 - Q.mass) / 5.061224) / math.sqrt(2))))
    z += -0.005399401 * (5.110145 * 0.5 * ((131.3917 - Q.mass) / 5.110145) * (1 + math.erf(((131.3917 - Q.mass) / 5.110145) / math.sqrt(2))))
    z += 1.541007e-05 * (13.56554 * 0.5 * ((Q.mass_top15 - 133.1648) / 13.56554) * (1 + math.erf(((Q.mass_top15 - 133.1648) / 13.56554) / math.sqrt(2)))) * (10.58345 * 0.5 * ((Q.sum_pt_top40 - 569.6282) / 10.58345) * (1 + math.erf(((Q.sum_pt_top40 - 569.6282) / 10.58345) / math.sqrt(2))))
    z += -7264.412 * (2.752946 * 0.5 * ((3.53503 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.53503 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (3.652904e-06 * 0.5 * ((7.184834e-06 - Q.e4) / 3.652904e-06) * (1 + math.erf(((7.184834e-06 - Q.e4) / 3.652904e-06) / math.sqrt(2))))
    z += 126.1814 * (0.0005676896 * 0.5 * ((0.00228569 - Q.e3) / 0.0005676896) * (1 + math.erf(((0.00228569 - Q.e3) / 0.0005676896) / math.sqrt(2))))
    z += 0.001915809 * (5.110145 * 0.5 * ((131.3917 - Q.mass) / 5.110145) * (1 + math.erf(((131.3917 - Q.mass) / 5.110145) / math.sqrt(2)))) * (1.929175 * 0.5 * ((4.004982 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((4.004982 - Q.jd_3d_5) / 1.929175) / math.sqrt(2))))
    z += -0.01377072 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (1.929175 * 0.5 * ((4.004982 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((4.004982 - Q.jd_3d_5) / 1.929175) / math.sqrt(2))))
    z += -0.07644173 * (0.2156905 * 0.5 * ((2.41868 - Q.sjf_2_2_max3d) / 0.2156905) * (1 + math.erf(((2.41868 - Q.sjf_2_2_max3d) / 0.2156905) / math.sqrt(2))))
    z += -0.0004330605 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (14.45336 * 0.5 * ((Q.jd_3d_6 - 30.57088) / 14.45336) * (1 + math.erf(((Q.jd_3d_6 - 30.57088) / 14.45336) / math.sqrt(2))))
    z += 0.0006931307 * (13.85751 * 0.5 * ((91.2852 - Q.sj2_mass1) / 13.85751) * (1 + math.erf(((91.2852 - Q.sj2_mass1) / 13.85751) / math.sqrt(2))))
    z += -4.465944 * (0.00189604 * 0.5 * ((0.01394245 - Q.dr_min_012) / 0.00189604) * (1 + math.erf(((0.01394245 - Q.dr_min_012) / 0.00189604) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.iselectron_1 - 0.0) / 1e-06) * (1 + math.erf(((Q.iselectron_1 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 1.297298 * (0.01151941 * 0.5 * ((Q.N2 - 0.3245983) / 0.01151941) * (1 + math.erf(((Q.N2 - 0.3245983) / 0.01151941) / math.sqrt(2))))
    z += -5.203809e-05 * (2.41177 * 0.5 * ((2.907433 - Q.lep_iso) / 2.41177) * (1 + math.erf(((2.907433 - Q.lep_iso) / 2.41177) / math.sqrt(2)))) * (1032.513 * 0.5 * ((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) * (1 + math.erf(((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) / math.sqrt(2))))
    z += 4.654787 * (0.03429006 * 0.5 * ((0.08178299 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.08178299 - Q.lep_dr) / 0.03429006) / math.sqrt(2))))
    z += -54.44116 * (0.006229165 * 0.5 * ((0.004135872 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.004135872 - Q.lep_z) / 0.006229165) / math.sqrt(2))))
    z += -0.1560686 * (0.2186175 * 0.5 * ((-3.643775 - Q.lnerel_4) / 0.2186175) * (1 + math.erf(((-3.643775 - Q.lnerel_4) / 0.2186175) / math.sqrt(2))))
    z += 1.964693 * (0.01538599 * 0.5 * ((0.07823361 - Q.pz_lnkt0) / 0.01538599) * (1 + math.erf(((0.07823361 - Q.pz_lnkt0) / 0.01538599) / math.sqrt(2))))
    z += -0.002357471 * (20.30421 * 0.5 * ((Q.dc_split1_mass - 154.6884) / 20.30421) * (1 + math.erf(((Q.dc_split1_mass - 154.6884) / 20.30421) / math.sqrt(2))))
    z += 0.05072792 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.15481) / 5.26174) / math.sqrt(2))))
    z += -0.004865255 * (10.51039 * 0.5 * ((Q.mres_sd_mass_b2z01 - 69.12133) / 10.51039) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 69.12133) / 10.51039) / math.sqrt(2))))
    z += -0.04638982 * (5.55459 * 0.5 * ((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 96.62031) / 5.55459) / math.sqrt(2))))
    z += 15.45181 * (0.006899392 * 0.5 * ((0.006470637 - Q.lepsj_2_dr) / 0.006899392) * (1 + math.erf(((0.006470637 - Q.lepsj_2_dr) / 0.006899392) / math.sqrt(2))))
    z += -0.5900184 * (1.0 * 0.5 * ((0.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.03086866 * (0.02153986 * 0.5 * ((0.4646004 - Q.z_neutral) / 0.02153986) * (1 + math.erf(((0.4646004 - Q.z_neutral) / 0.02153986) / math.sqrt(2))))
    z += 3.456632 * (1.0 * 0.5 * ((0.0 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2)))) * (0.01040872 * 0.5 * ((Q.sdb_4_z - 0.01435877) / 0.01040872) * (1 + math.erf(((Q.sdb_4_z - 0.01435877) / 0.01040872) / math.sqrt(2))))
    z += -0.01020981 * (1.0 * 0.5 * ((0.0 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.isphoton_42) / 1.0) * (1 + math.erf(((1.0 - Q.isphoton_42) / 1.0) / math.sqrt(2))))
    z += -10.81571 * (0.01538599 * 0.5 * ((0.07823361 - Q.pz_lnkt0) / 0.01538599) * (1 + math.erf(((0.07823361 - Q.pz_lnkt0) / 0.01538599) / math.sqrt(2)))) * (0.1714986 * 0.5 * ((Q.lund2_lnz - -0.9953121) / 0.1714986) * (1 + math.erf(((Q.lund2_lnz - -0.9953121) / 0.1714986) / math.sqrt(2))))
    z += -0.2087606 * (3.705078 * 0.5 * ((5.8125 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8125 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ismuon_0 - 1.0) / 1.0) * (1 + math.erf(((Q.ismuon_0 - 1.0) / 1.0) / math.sqrt(2))))
    z += -9.946755e-05 * (164.9991 * 0.5 * ((226.3008 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((226.3008 - Q.sip_3d_2) / 164.9991) / math.sqrt(2))))
    z += 0.003831314 * (11.4891 * 0.5 * ((32.61679 - Q.mass_displaced5) / 11.4891) * (1 + math.erf(((32.61679 - Q.mass_displaced5) / 11.4891) / math.sqrt(2))))
    z += 0.001369741 * (36.4359 * 0.5 * ((28.96878 - Q.sv_2_sd0_sum) / 36.4359) * (1 + math.erf(((28.96878 - Q.sv_2_sd0_sum) / 36.4359) / math.sqrt(2))))
    z += 0.03133843 * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.4037488) / 0.179816) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.iselectron_12 - 0.0) / 1e-06) * (1 + math.erf(((Q.iselectron_12 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.00655425 * (5.892378 * 0.5 * ((12.15228 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.15228 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.09724841 * 0.5 * ((0.4305249 - Q.sjq_3_1_k05) / 0.09724841) * (1 + math.erf(((0.4305249 - Q.sjq_3_1_k05) / 0.09724841) / math.sqrt(2))))
    z += -0.6039386 * (0.08036477 * 0.5 * ((Q.pair_mean_lndelta - -1.714682) / 0.08036477) * (1 + math.erf(((Q.pair_mean_lndelta - -1.714682) / 0.08036477) / math.sqrt(2))))
    z += 0.9355415 * (0.009024806 * 0.5 * ((0.05997821 - Q.pz_lnd0) / 0.009024806) * (1 + math.erf(((0.05997821 - Q.pz_lnd0) / 0.009024806) / math.sqrt(2))))
    z += 1.879386 * (0.01519451 * 0.5 * ((Q.psi_0p2 - 0.9892758) / 0.01519451) * (1 + math.erf(((Q.psi_0p2 - 0.9892758) / 0.01519451) / math.sqrt(2))))
    z += -0.2296896 * (0.02132677 * 0.5 * ((Q.dc_split1_dr - 0.2818852) / 0.02132677) * (1 + math.erf(((Q.dc_split1_dr - 0.2818852) / 0.02132677) / math.sqrt(2))))
    z += -0.0066717 * (0.9486798 * 0.5 * ((4.6892 - Q.mres_sd_prong_mass2) / 0.9486798) * (1 + math.erf(((4.6892 - Q.mres_sd_prong_mass2) / 0.9486798) / math.sqrt(2))))
    return z


def neuron_121(Q):
    z = -5.730511e-06
    return z


def neuron_122(Q):
    z = -2.395899e-06
    return z


def neuron_123(Q):
    z = -0.2217608
    z += 0.6981297 * Q.sjf_2_1_z_d3
    z += -0.0266199 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2))))
    z += -2.029584 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2))))
    z += -0.6537192 * (0.100181 * 0.5 * ((0.3803178 - Q.z_displaced5) / 0.100181) * (1 + math.erf(((0.3803178 - Q.z_displaced5) / 0.100181) / math.sqrt(2))))
    z += 21.95881 * (0.100181 * 0.5 * ((0.3803178 - Q.z_displaced5) / 0.100181) * (1 + math.erf(((0.3803178 - Q.z_displaced5) / 0.100181) / math.sqrt(2)))) * (0.04864387 * 0.5 * ((0.06457187 - Q.lep_z) / 0.04864387) * (1 + math.erf(((0.06457187 - Q.lep_z) / 0.04864387) / math.sqrt(2))))
    z += 0.02064469 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2))))
    z += -0.140681 * (0.02757255 * 0.5 * ((0.6800935 - Q.tau32) / 0.02757255) * (1 + math.erf(((0.6800935 - Q.tau32) / 0.02757255) / math.sqrt(2))))
    z += -11.44808 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.01704903 * (23.0 * 0.5 * ((58.0 - Q.n_pairs_kt_above_1) / 23.0) * (1 + math.erf(((58.0 - Q.n_pairs_kt_above_1) / 23.0) / math.sqrt(2))))
    z += 26.13256 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2)))) * (0.02245414 * 0.5 * ((Q.dr_1 - 0.1975394) / 0.02245414) * (1 + math.erf(((Q.dr_1 - 0.1975394) / 0.02245414) / math.sqrt(2))))
    z += -5.999201 * (0.04864387 * 0.5 * ((0.06457187 - Q.lep_z) / 0.04864387) * (1 + math.erf(((0.06457187 - Q.lep_z) / 0.04864387) / math.sqrt(2))))
    z += -0.1096441 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2))))
    z += 6.557445 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.sv_1_n) / 1.0) * (1 + math.erf(((3.0 - Q.sv_1_n) / 1.0) / math.sqrt(2))))
    z += 1.160256 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.117671 * 0.5 * ((0.4191372 - Q.lep_dr) / 0.117671) * (1 + math.erf(((0.4191372 - Q.lep_dr) / 0.117671) / math.sqrt(2))))
    z += 5.565402 * (0.01092519 * 0.5 * ((0.04286245 - Q.pz_lnd2) / 0.01092519) * (1 + math.erf(((0.04286245 - Q.pz_lnd2) / 0.01092519) / math.sqrt(2))))
    z += -0.04473476 * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.04544863 * (0.5664529 * 0.5 * ((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) * (1 + math.erf(((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) / math.sqrt(2))))
    z += 2.588608 * (0.0106503 * 0.5 * ((0.09036178 - Q.pz_lnkt0) / 0.0106503) * (1 + math.erf(((0.09036178 - Q.pz_lnkt0) / 0.0106503) / math.sqrt(2))))
    z += -0.005706174 * (23.0 * 0.5 * ((58.0 - Q.n_pairs_kt_above_1) / 23.0) * (1 + math.erf(((58.0 - Q.n_pairs_kt_above_1) / 23.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.0 - Q.sjf_3_1_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.sjf_3_1_n_d3) / 1.0) / math.sqrt(2))))
    z += -0.1721431 * (0.5664529 * 0.5 * ((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) * (1 + math.erf(((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) / math.sqrt(2)))) * (1.0 * 0.5 * ((2.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((2.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 0.006403196 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2))))
    z += -0.01597785 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((5.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((5.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 7.402666 * (0.04864387 * 0.5 * ((0.06457187 - Q.lep_z) / 0.04864387) * (1 + math.erf(((0.06457187 - Q.lep_z) / 0.04864387) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_min12_n_disp3 - 1.0) / 1.0) * (1 + math.erf(((Q.ak02_min12_n_disp3 - 1.0) / 1.0) / math.sqrt(2))))
    z += 1.818778 * (0.003235318 * 0.5 * ((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) * (1 + math.erf(((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) / math.sqrt(2))))
    z += 4.740442 * (0.005331567 * 0.5 * ((Q.M2_b05 - 0.1209237) / 0.005331567) * (1 + math.erf(((Q.M2_b05 - 0.1209237) / 0.005331567) / math.sqrt(2))))
    z += -49.25701 * (0.0106503 * 0.5 * ((0.09036178 - Q.pz_lnkt0) / 0.0106503) * (1 + math.erf(((0.09036178 - Q.pz_lnkt0) / 0.0106503) / math.sqrt(2)))) * (0.05391589 * 0.5 * ((0.2792084 - Q.dr_2) / 0.05391589) * (1 + math.erf(((0.2792084 - Q.dr_2) / 0.05391589) / math.sqrt(2))))
    z += -1.532805 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (0.01968664 * 0.5 * ((0.06216672 - Q.pz_lnkt4) / 0.01968664) * (1 + math.erf(((0.06216672 - Q.pz_lnkt4) / 0.01968664) / math.sqrt(2))))
    z += -0.009490322 * (15.47377 * 0.5 * ((56.49019 - Q.mass) / 15.47377) * (1 + math.erf(((56.49019 - Q.mass) / 15.47377) / math.sqrt(2))))
    z += -0.0198954 * (1.400504 * 0.5 * ((14.42172 - Q.sj2_mass2) / 1.400504) * (1 + math.erf(((14.42172 - Q.sj2_mass2) / 1.400504) / math.sqrt(2))))
    z += 0.3395439 * (0.0964495 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) / math.sqrt(2))))
    z += 1.564178 * (0.02363876 * 0.5 * ((0.4744137 - Q.z_charged_had) / 0.02363876) * (1 + math.erf(((0.4744137 - Q.z_charged_had) / 0.02363876) / math.sqrt(2))))
    z += -0.7887649 * (0.02363876 * 0.5 * ((0.4744137 - Q.z_charged_had) / 0.02363876) * (1 + math.erf(((0.4744137 - Q.z_charged_had) / 0.02363876) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.dc_ntag) / 1.0) * (1 + math.erf(((1.0 - Q.dc_ntag) / 1.0) / math.sqrt(2))))
    z += 7.646791e-05 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (257.2011 * 0.5 * ((482.5817 - Q.jd_sum_abs_sd0_top5) / 257.2011) * (1 + math.erf(((482.5817 - Q.jd_sum_abs_sd0_top5) / 257.2011) / math.sqrt(2))))
    z += 3.251892 * (0.01140748 * 0.5 * ((0.0573796 - Q.pz_lnd3) / 0.01140748) * (1 + math.erf(((0.0573796 - Q.pz_lnd3) / 0.01140748) / math.sqrt(2))))
    z += -23.71308 * (0.01140748 * 0.5 * ((0.0573796 - Q.pz_lnd3) / 0.01140748) * (1 + math.erf(((0.0573796 - Q.pz_lnd3) / 0.01140748) / math.sqrt(2)))) * (0.05999923 * 0.5 * ((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) * (1 + math.erf(((0.1940813 - Q.sjf_3_1_z_d3) / 0.05999923) / math.sqrt(2))))
    z += -1.855309 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.04048364 * 0.5 * ((Q.ak02_dr23 - 0.2759933) / 0.04048364) * (1 + math.erf(((Q.ak02_dr23 - 0.2759933) / 0.04048364) / math.sqrt(2))))
    z += 0.3903474 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sdb_0_n - 2.0) / 1.0) * (1 + math.erf(((Q.sdb_0_n - 2.0) / 1.0) / math.sqrt(2))))
    z += -26.06621 * (0.002789868 * 0.5 * ((0.02926638 - Q.M2_b2) / 0.002789868) * (1 + math.erf(((0.02926638 - Q.M2_b2) / 0.002789868) / math.sqrt(2)))) * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2))))
    z += -0.02457868 * (0.100181 * 0.5 * ((0.3803178 - Q.z_displaced5) / 0.100181) * (1 + math.erf(((0.3803178 - Q.z_displaced5) / 0.100181) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.kt2_1_n_lep - 0.0) / 1.0) * (1 + math.erf(((Q.kt2_1_n_lep - 0.0) / 1.0) / math.sqrt(2))))
    z += 8.564953 * (0.003235318 * 0.5 * ((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) * (1 + math.erf(((0.0008210142 - Q.lepsj_2_dr) / 0.003235318) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2))))
    z += -0.04343015 * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2))))
    z += -0.0005180048 * (6.785484 * 0.5 * ((Q.mass_top5 - 68.52153) / 6.785484) * (1 + math.erf(((Q.mass_top5 - 68.52153) / 6.785484) / math.sqrt(2))))
    z += -0.0004021979 * (5.02206 * 0.5 * ((Q.mass_top50 - 129.5874) / 5.02206) * (1 + math.erf(((Q.mass_top50 - 129.5874) / 5.02206) / math.sqrt(2))))
    z += 0.00287871 * (16.90428 * 0.5 * ((Q.mass - 164.4374) / 16.90428) * (1 + math.erf(((Q.mass - 164.4374) / 16.90428) / math.sqrt(2))))
    z += 0.1861927 * (2.0 * 0.5 * ((3.0 - Q.sjq_2_1_nch) / 2.0) * (1 + math.erf(((3.0 - Q.sjq_2_1_nch) / 2.0) / math.sqrt(2))))
    z += -1.185148 * (0.0964495 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) / math.sqrt(2)))) * (0.7733816 * 0.5 * ((0.7828545 - Q.lepsj_2_maxsd0) / 0.7733816) * (1 + math.erf(((0.7828545 - Q.lepsj_2_maxsd0) / 0.7733816) / math.sqrt(2))))
    z += 0.7949101 * (0.04048364 * 0.5 * ((Q.ak02_dr23 - 0.2759933) / 0.04048364) * (1 + math.erf(((Q.ak02_dr23 - 0.2759933) / 0.04048364) / math.sqrt(2))))
    z += 1.443146 * (0.04864387 * 0.5 * ((0.06457187 - Q.lep_z) / 0.04864387) * (1 + math.erf(((0.06457187 - Q.lep_z) / 0.04864387) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2))))
    z += 0.001402575 * (0.02363876 * 0.5 * ((0.4744137 - Q.z_charged_had) / 0.02363876) * (1 + math.erf(((0.4744137 - Q.z_charged_had) / 0.02363876) / math.sqrt(2)))) * (164.9991 * 0.5 * ((226.3008 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((226.3008 - Q.sip_3d_2) / 164.9991) / math.sqrt(2))))
    z += -0.0008696046 * (6.235814 * 0.5 * ((Q.mres_sd_mass_b0z005 - 76.40585) / 6.235814) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 76.40585) / 6.235814) / math.sqrt(2))))
    z += 0.001669305 * (7.843552 * 0.5 * ((Q.mres_sd_mass_b0z005 - 131.0776) / 7.843552) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 131.0776) / 7.843552) / math.sqrt(2))))
    z += -0.03316416 * (1.278204 * 0.5 * ((3.101398 - Q.sjf_3_2_max3d) / 1.278204) * (1 + math.erf(((3.101398 - Q.sjf_3_2_max3d) / 1.278204) / math.sqrt(2))))
    z += -0.02443467 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (0.2342104 * 0.5 * ((Q.lund3_lnz - -4.699653) / 0.2342104) * (1 + math.erf(((Q.lund3_lnz - -4.699653) / 0.2342104) / math.sqrt(2))))
    z += 0.7317879 * (0.02363876 * 0.5 * ((0.4744137 - Q.z_charged_had) / 0.02363876) * (1 + math.erf(((0.4744137 - Q.z_charged_had) / 0.02363876) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.iselectron_1 - 0.0) / 1e-06) * (1 + math.erf(((Q.iselectron_1 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.6029426 * (0.01140748 * 0.5 * ((0.0573796 - Q.pz_lnd3) / 0.01140748) * (1 + math.erf(((0.0573796 - Q.pz_lnd3) / 0.01140748) / math.sqrt(2)))) * (3.006676 * 0.5 * ((6.43167 - Q.jd_3d_4) / 3.006676) * (1 + math.erf(((6.43167 - Q.jd_3d_4) / 3.006676) / math.sqrt(2))))
    z += -5.390028 * (0.005331567 * 0.5 * ((Q.M2_b05 - 0.1209237) / 0.005331567) * (1 + math.erf(((Q.M2_b05 - 0.1209237) / 0.005331567) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.0) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.308801 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += 0.01113875 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2))))
    z += -0.004426477 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.606241 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.606241 - Q.sip_3d_3) / 2.093785) / math.sqrt(2))))
    z += 0.003526663 * (22.5 * 0.5 * ((123.0 - Q.n_pairs_kt_above_1) / 22.5) * (1 + math.erf(((123.0 - Q.n_pairs_kt_above_1) / 22.5) / math.sqrt(2))))
    z += -8.044138e-05 * (1032.513 * 0.5 * ((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) * (1 + math.erf(((1163.885 - Q.lepsj_2_maxsd0) / 1032.513) / math.sqrt(2))))
    z += -0.03336271 * (1.0 * 0.5 * ((4.0 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((4.0 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2))))
    z += 0.0006437333 * (115.0698 * 0.5 * ((172.888 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.888 - Q.jd_3d_4) / 115.0698) / math.sqrt(2))))
    z += 1.438514 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (0.03294892 * 0.5 * ((0.08598434 - Q.kt2_2_z_disp3) / 0.03294892) * (1 + math.erf(((0.08598434 - Q.kt2_2_z_disp3) / 0.03294892) / math.sqrt(2))))
    z += -0.1570336 * (0.1721579 * 0.5 * ((2.133633 - Q.pt_entropy) / 0.1721579) * (1 + math.erf(((2.133633 - Q.pt_entropy) / 0.1721579) / math.sqrt(2))))
    z += 231.9861 * (0.02363876 * 0.5 * ((0.4744137 - Q.z_charged_had) / 0.02363876) * (1 + math.erf(((0.4744137 - Q.z_charged_had) / 0.02363876) / math.sqrt(2)))) * (0.000830722 * 0.5 * ((0.00678572 - Q.M3_b2) / 0.000830722) * (1 + math.erf(((0.00678572 - Q.M3_b2) / 0.000830722) / math.sqrt(2))))
    z += 0.04535613 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (0.03240401 * 0.5 * ((Q.ak02_2_z - 0.0) / 0.03240401) * (1 + math.erf(((Q.ak02_2_z - 0.0) / 0.03240401) / math.sqrt(2))))
    z += 0.1915704 * (1.0 * 0.5 * ((4.0 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((4.0 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2)))) * (0.03240401 * 0.5 * ((Q.ak02_2_z - 0.0) / 0.03240401) * (1 + math.erf(((Q.ak02_2_z - 0.0) / 0.03240401) / math.sqrt(2))))
    z += 0.005327971 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.0) / 1.0) / math.sqrt(2))))
    z += 0.02853113 * (2.0 * 0.5 * ((Q.sjf_2_1_n_d3 - 7.0) / 2.0) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 7.0) / 2.0) / math.sqrt(2))))
    z += -0.05406182 * (0.1664357 * 0.5 * ((-3.425158 - Q.lnerel_4) / 0.1664357) * (1 + math.erf(((-3.425158 - Q.lnerel_4) / 0.1664357) / math.sqrt(2))))
    z += -0.00582414 * (5.10267 * 0.5 * ((Q.mres_sd_mass_b0z005 - 86.65765) / 5.10267) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 86.65765) / 5.10267) / math.sqrt(2))))
    z += 0.001943838 * (24.25363 * 0.5 * ((Q.mres_sd_mass_b0z005 - 53.57509) / 24.25363) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 53.57509) / 24.25363) / math.sqrt(2))))
    z += 0.002449036 * (7.17138 * 0.5 * ((Q.mass_top10 - 95.24893) / 7.17138) * (1 + math.erf(((Q.mass_top10 - 95.24893) / 7.17138) / math.sqrt(2))))
    z += 0.3797841 * (0.03953735 * 0.5 * ((Q.mres_sd_rg_b1z01 - 0.5299323) / 0.03953735) * (1 + math.erf(((Q.mres_sd_rg_b1z01 - 0.5299323) / 0.03953735) / math.sqrt(2))))
    z += 0.009075644 * (4.539253 * 0.5 * ((Q.mres_sd_mass_b0z005 - 107.7493) / 4.539253) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 107.7493) / 4.539253) / math.sqrt(2))))
    z += 0.01192756 * (6.875137 * 0.5 * ((18.80005 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.80005 - Q.mass_displaced3) / 6.875137) / math.sqrt(2))))
    z += -0.5753293 * (0.0964495 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) / math.sqrt(2)))) * (0.1568179 * 0.5 * ((1.261566 - Q.sjq_2_sumabs_k03) / 0.1568179) * (1 + math.erf(((1.261566 - Q.sjq_2_sumabs_k03) / 0.1568179) / math.sqrt(2))))
    z += -0.07076026 * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) * (0.2881506 * 0.5 * ((Q.lund1_lnkt - 3.599502) / 0.2881506) * (1 + math.erf(((Q.lund1_lnkt - 3.599502) / 0.2881506) / math.sqrt(2))))
    z += -0.009320946 * (0.5664529 * 0.5 * ((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) * (1 + math.erf(((Q.pair_mean_lnm2 - 4.069088) / 0.5664529) / math.sqrt(2)))) * (4.214136 * 0.5 * ((10.59762 - Q.min_pair_mass) / 4.214136) * (1 + math.erf(((10.59762 - Q.min_pair_mass) / 4.214136) / math.sqrt(2))))
    z += 1.123044e-05 * (24.25363 * 0.5 * ((Q.mres_sd_mass_b0z005 - 53.57509) / 24.25363) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 53.57509) / 24.25363) / math.sqrt(2)))) * (13.98191 * 0.5 * ((12.49716 - Q.sjf_2_1_mass_d3) / 13.98191) * (1 + math.erf(((12.49716 - Q.sjf_2_1_mass_d3) / 13.98191) / math.sqrt(2))))
    z += -0.008950138 * (7.17138 * 0.5 * ((Q.mass_top10 - 95.24893) / 7.17138) * (1 + math.erf(((Q.mass_top10 - 95.24893) / 7.17138) / math.sqrt(2)))) * (9.373106 * 0.5 * ((Q.lnpt_40 - 0.0508341) / 9.373106) * (1 + math.erf(((Q.lnpt_40 - 0.0508341) / 9.373106) / math.sqrt(2))))
    z += -0.5819511 * (0.02314658 * 0.5 * ((Q.z_neutral - 0.4865925) / 0.02314658) * (1 + math.erf(((Q.z_neutral - 0.4865925) / 0.02314658) / math.sqrt(2))))
    z += -0.2220038 * (0.1667799 * 0.5 * ((-0.3680108 - Q.sjq_3_2_k1) / 0.1667799) * (1 + math.erf(((-0.3680108 - Q.sjq_3_2_k1) / 0.1667799) / math.sqrt(2))))
    z += -0.273645 * (0.2398529 * 0.5 * ((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) * (1 + math.erf(((Q.sjq_3_2_k1 - 0.6148962) / 0.2398529) / math.sqrt(2))))
    z += 0.01339741 * (0.100181 * 0.5 * ((0.3803178 - Q.z_displaced5) / 0.100181) * (1 + math.erf(((0.3803178 - Q.z_displaced5) / 0.100181) / math.sqrt(2)))) * (18.73129 * 0.5 * ((-18.42068 - Q.lne_40) / 18.73129) * (1 + math.erf(((-18.42068 - Q.lne_40) / 18.73129) / math.sqrt(2))))
    z += 0.05565603 * (2.0 * 0.5 * ((5.0 - Q.n_lund) / 2.0) * (1 + math.erf(((5.0 - Q.n_lund) / 2.0) / math.sqrt(2))))
    z += -0.02012721 * (4.934653 * 0.5 * ((14.92637 - Q.mass_neutral) / 4.934653) * (1 + math.erf(((14.92637 - Q.mass_neutral) / 4.934653) / math.sqrt(2))))
    z += 67.92088 * (0.1721579 * 0.5 * ((2.133633 - Q.pt_entropy) / 0.1721579) * (1 + math.erf(((2.133633 - Q.pt_entropy) / 0.1721579) / math.sqrt(2)))) * (0.002556514 * 0.5 * ((0.001692006 - Q.sjf_4_3_z_d3) / 0.002556514) * (1 + math.erf(((0.001692006 - Q.sjf_4_3_z_d3) / 0.002556514) / math.sqrt(2))))
    z += 0.08558274 * (0.0964495 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.0) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.0) / 1.0) / math.sqrt(2))))
    z += -3.541706 * (0.04864387 * 0.5 * ((0.06457187 - Q.lep_z) / 0.04864387) * (1 + math.erf(((0.06457187 - Q.lep_z) / 0.04864387) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.0 - Q.lepsj_2_n_d3) / 1.0) * (1 + math.erf(((1.0 - Q.lepsj_2_n_d3) / 1.0) / math.sqrt(2))))
    z += 0.1644453 * (1.0 * 0.5 * ((4.0 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((4.0 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2)))) * (0.01992156 * 0.5 * ((Q.lepsj_2_dr - 0.04178742) / 0.01992156) * (1 + math.erf(((Q.lepsj_2_dr - 0.04178742) / 0.01992156) / math.sqrt(2))))
    z += 0.003619247 * (0.179073 * 0.5 * ((0.5187302 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.5187302 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (2.0 * 0.5 * ((4.0 - Q.lepsj_2_n_d3) / 2.0) * (1 + math.erf(((4.0 - Q.lepsj_2_n_d3) / 2.0) / math.sqrt(2))))
    z += -0.003074329 * (12.22004 * 0.5 * ((Q.lep_ptrel - 27.3236) / 12.22004) * (1 + math.erf(((Q.lep_ptrel - 27.3236) / 12.22004) / math.sqrt(2)))) * (0.06161476 * 0.5 * ((0.676889 - Q.sj3_dr13) / 0.06161476) * (1 + math.erf(((0.676889 - Q.sj3_dr13) / 0.06161476) / math.sqrt(2))))
    z += -0.04991509 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.20788) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.20788) / 15.88428) / math.sqrt(2)))) * (0.05374146 * 0.5 * ((-0.1588135 - Q.phi_4) / 0.05374146) * (1 + math.erf(((-0.1588135 - Q.phi_4) / 0.05374146) / math.sqrt(2))))
    z += 0.172943 * (0.03953735 * 0.5 * ((Q.mres_sd_rg_b1z01 - 0.5299323) / 0.03953735) * (1 + math.erf(((Q.mres_sd_rg_b1z01 - 0.5299323) / 0.03953735) / math.sqrt(2)))) * (0.07882828 * 0.5 * ((0.7416858 - Q.sjq_2_sumabs_k03) / 0.07882828) * (1 + math.erf(((0.7416858 - Q.sjq_2_sumabs_k03) / 0.07882828) / math.sqrt(2))))
    z += 1.424681 * (0.0964495 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.5557674) / 0.0964495) / math.sqrt(2)))) * (0.0131537 * 0.5 * ((0.1142534 - Q.dr_11) / 0.0131537) * (1 + math.erf(((0.1142534 - Q.dr_11) / 0.0131537) / math.sqrt(2))))
    z += 1.727382 * (0.02757255 * 0.5 * ((0.6800935 - Q.tau32) / 0.02757255) * (1 + math.erf(((0.6800935 - Q.tau32) / 0.02757255) / math.sqrt(2)))) * (0.1102744 * 0.5 * ((Q.dr_11 - 0.5195105) / 0.1102744) * (1 + math.erf(((Q.dr_11 - 0.5195105) / 0.1102744) / math.sqrt(2))))
    z += 0.01057825 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 1.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 1.0) / 1.0) / math.sqrt(2))))
    z += -0.009979744 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.9242) / 18.66842) / math.sqrt(2))))
    return z


def neuron_124(Q):
    z = -0.5158927
    z += 0.8302411 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2))))
    z += -0.001347997 * (43.0 * 0.5 * ((366.0 - Q.n_pairs_kt_above_1) / 43.0) * (1 + math.erf(((366.0 - Q.n_pairs_kt_above_1) / 43.0) / math.sqrt(2))))
    z += -0.07577223 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2))))
    z += -0.005125233 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2))))
    z += 0.00182009 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2))))
    z += -2.151035e-05 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (14.3347 * 0.5 * ((142.9952 - Q.sj3_pair_mass_max) / 14.3347) * (1 + math.erf(((142.9952 - Q.sj3_pair_mass_max) / 14.3347) / math.sqrt(2))))
    z += -0.2300698 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (0.007964189 * 0.5 * ((0.06762785 - Q.M2_b2) / 0.007964189) * (1 + math.erf(((0.06762785 - Q.M2_b2) / 0.007964189) / math.sqrt(2))))
    z += 0.004169663 * (5.197 * 0.5 * ((95.14961 - Q.mass) / 5.197) * (1 + math.erf(((95.14961 - Q.mass) / 5.197) / math.sqrt(2))))
    z += 0.004366538 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2))))
    z += 0.005173462 * (9.773951 * 0.5 * ((56.73313 - Q.sj3_pair_mass_max) / 9.773951) * (1 + math.erf(((56.73313 - Q.sj3_pair_mass_max) / 9.773951) / math.sqrt(2))))
    z += -0.6699026 * (0.6671377 * 0.5 * ((0.4381892 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.4381892 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += -0.006945771 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2))))
    z += 4.431675 * (0.01992156 * 0.5 * ((0.04178742 - Q.lepsj_2_dr) / 0.01992156) * (1 + math.erf(((0.04178742 - Q.lepsj_2_dr) / 0.01992156) / math.sqrt(2))))
    z += 0.1304837 * (0.1605002 * 0.5 * ((Q.pair_max_lnm2 - 7.347625) / 0.1605002) * (1 + math.erf(((Q.pair_max_lnm2 - 7.347625) / 0.1605002) / math.sqrt(2))))
    z += -0.1957402 * (0.9305981 * 0.5 * ((3.029824 - Q.sjf_2_2_max3d) / 0.9305981) * (1 + math.erf(((3.029824 - Q.sjf_2_2_max3d) / 0.9305981) / math.sqrt(2))))
    z += -0.01370759 * (2.5 * 0.5 * ((31.0 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((31.0 - Q.n_for_90pct) / 2.5) / math.sqrt(2))))
    z += 0.006002228 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_ntag - 0.0) / 1.0) * (1 + math.erf(((Q.dc_ntag - 0.0) / 1.0) / math.sqrt(2))))
    z += -5.592237 * (0.01710086 * 0.5 * ((0.06416437 - Q.z_displaced3) / 0.01710086) * (1 + math.erf(((0.06416437 - Q.z_displaced3) / 0.01710086) / math.sqrt(2))))
    z += -0.004737858 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.003882417 * 0.5 * ((-0.006552442 - Q.sjq_2_prod_k1) / 0.003882417) * (1 + math.erf(((-0.006552442 - Q.sjq_2_prod_k1) / 0.003882417) / math.sqrt(2))))
    z += -0.2319391 * (0.03580654 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.2769039) / 0.03580654) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.2769039) / 0.03580654) / math.sqrt(2))))
    z += 0.01853921 * (2.5 * 0.5 * ((31.0 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((31.0 - Q.n_for_90pct) / 2.5) / math.sqrt(2)))) * (0.1673111 * 0.5 * ((Q.sjq_2_prod_k05 - -0.5057096) / 0.1673111) * (1 + math.erf(((Q.sjq_2_prod_k05 - -0.5057096) / 0.1673111) / math.sqrt(2))))
    z += -6.057859 * (0.1486471 * 0.5 * ((0.3396572 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.3396572 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.02653106 * 0.5 * ((Q.z_displaced5 - 0.1046203) / 0.02653106) * (1 + math.erf(((Q.z_displaced5 - 0.1046203) / 0.02653106) / math.sqrt(2))))
    z += 0.007206046 * (18.42178 * 0.5 * ((182.8592 - Q.mass) / 18.42178) * (1 + math.erf(((182.8592 - Q.mass) / 18.42178) / math.sqrt(2))))
    z += 0.0149145 * (2.5 * 0.5 * ((31.0 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((31.0 - Q.n_for_90pct) / 2.5) / math.sqrt(2)))) * (0.03966789 * 0.5 * ((0.1333792 - Q.ak02_3_z) / 0.03966789) * (1 + math.erf(((0.1333792 - Q.ak02_3_z) / 0.03966789) / math.sqrt(2))))
    z += -0.4806679 * (0.02246636 * 0.5 * ((Q.z_photon - 0.1149688) / 0.02246636) * (1 + math.erf(((Q.z_photon - 0.1149688) / 0.02246636) / math.sqrt(2))))
    z += 3.571028 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2))))
    z += -0.00097381 * (2.5 * 0.5 * ((31.0 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((31.0 - Q.n_for_90pct) / 2.5) / math.sqrt(2)))) * (0.1509271 * 0.5 * ((Q.lund1_lndelta - -1.046452) / 0.1509271) * (1 + math.erf(((Q.lund1_lndelta - -1.046452) / 0.1509271) / math.sqrt(2))))
    z += 0.007832783 * (5.001896 * 0.5 * ((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.413264 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2))))
    z += 0.6196245 * (1.0 * 0.5 * ((1.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((1.0 - Q.n_lepton) / 1.0) / math.sqrt(2))))
    z += -0.8983939 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.01119633 * 0.5 * ((0.8709334 - Q.tau43) / 0.01119633) * (1 + math.erf(((0.8709334 - Q.tau43) / 0.01119633) / math.sqrt(2))))
    z += 0.1234843 * (2.0 * 0.5 * ((Q.sjf_2_1_n_d3 - 7.0) / 2.0) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 7.0) / 2.0) / math.sqrt(2))))
    z += -23.59126 * (1.0 * 0.5 * ((1.0 - Q.n_lepton) / 1.0) * (1 + math.erf(((1.0 - Q.n_lepton) / 1.0) / math.sqrt(2)))) * (0.01040872 * 0.5 * ((0.01435877 - Q.sdb_4_z) / 0.01040872) * (1 + math.erf(((0.01435877 - Q.sdb_4_z) / 0.01040872) / math.sqrt(2))))
    z += 0.0004804029 * (3.317894 * 0.5 * ((117.4867 - Q.mass) / 3.317894) * (1 + math.erf(((117.4867 - Q.mass) / 3.317894) / math.sqrt(2)))) * (6.000774 * 0.5 * ((14.85973 - Q.jd_3d_4) / 6.000774) * (1 + math.erf(((14.85973 - Q.jd_3d_4) / 6.000774) / math.sqrt(2))))
    z += 0.00486978 * (15.88428 * 0.5 * ((43.20788 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.20788 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.05556614 * 0.5 * ((Q.jet_charge_k05 - 0.01860317) / 0.05556614) * (1 + math.erf(((Q.jet_charge_k05 - 0.01860317) / 0.05556614) / math.sqrt(2))))
    z += -6.401871e-05 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (401.9953 * 0.5 * ((577.991 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((577.991 - Q.sip_3d_3) / 401.9953) / math.sqrt(2))))
    z += -0.5412654 * (0.0439522 * 0.5 * ((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.09802954 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2))))
    z += 0.0213944 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (0.02145059 * 0.5 * ((0.7981752 - Q.tau32) / 0.02145059) * (1 + math.erf(((0.7981752 - Q.tau32) / 0.02145059) / math.sqrt(2))))
    z += -0.09895512 * (1.696027 * 0.5 * ((3.32421 - Q.sjf_2_1_max3d) / 1.696027) * (1 + math.erf(((3.32421 - Q.sjf_2_1_max3d) / 1.696027) / math.sqrt(2))))
    z += 0.0002410443 * (20.79165 * 0.5 * ((51.56287 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((51.56287 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2))))
    z += -0.1404964 * (0.02271025 * 0.5 * ((0.1465679 - Q.tau21_b2) / 0.02271025) * (1 + math.erf(((0.1465679 - Q.tau21_b2) / 0.02271025) / math.sqrt(2))))
    z += -12.26842 * (0.1060766 * 0.5 * ((0.221436 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221436 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.0947145 * 0.5 * ((0.0947145 - Q.dc_4_z) / 0.0947145) * (1 + math.erf(((0.0947145 - Q.dc_4_z) / 0.0947145) / math.sqrt(2))))
    z += 0.0724107 * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 1.0) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 1.0) / 1.0) / math.sqrt(2))))
    z += -0.001982414 * (5.197 * 0.5 * ((95.14961 - Q.mass) / 5.197) * (1 + math.erf(((95.14961 - Q.mass) / 5.197) / math.sqrt(2)))) * (0.5801757 * 0.5 * ((1.544662 - Q.N3_b2) / 0.5801757) * (1 + math.erf(((1.544662 - Q.N3_b2) / 0.5801757) / math.sqrt(2))))
    z += 5.211122e-07 * (18.42178 * 0.5 * ((182.8592 - Q.mass) / 18.42178) * (1 + math.erf(((182.8592 - Q.mass) / 18.42178) / math.sqrt(2)))) * (1110.683 * 0.5 * ((1490.247 - Q.sv_2_sd0_sum) / 1110.683) * (1 + math.erf(((1490.247 - Q.sv_2_sd0_sum) / 1110.683) / math.sqrt(2))))
    z += -0.01264306 * (3.244489 * 0.5 * ((66.70506 - Q.sj4_pair_mass_max) / 3.244489) * (1 + math.erf(((66.70506 - Q.sj4_pair_mass_max) / 3.244489) / math.sqrt(2))))
    z += -0.1865263 * (0.1223989 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.6652978) / 0.1223989) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.6652978) / 0.1223989) / math.sqrt(2))))
    z += -0.4403412 * (0.02505229 * 0.5 * ((Q.sdb_2_z - 0.3448514) / 0.02505229) * (1 + math.erf(((Q.sdb_2_z - 0.3448514) / 0.02505229) / math.sqrt(2))))
    z += 1.603807 * (0.009806371 * 0.5 * ((Q.sjq_2_prod_k1 - 0.02149519) / 0.009806371) * (1 + math.erf(((Q.sjq_2_prod_k1 - 0.02149519) / 0.009806371) / math.sqrt(2))))
    z += 0.01496781 * (1.0 * 0.5 * ((Q.sjf_4_1_n_d3 - 2.0) / 1.0) * (1 + math.erf(((Q.sjf_4_1_n_d3 - 2.0) / 1.0) / math.sqrt(2))))
    z += 0.9614724 * (0.01806359 * 0.5 * ((0.04865675 - Q.sjf_2_2_z_d3) / 0.01806359) * (1 + math.erf(((0.04865675 - Q.sjf_2_2_z_d3) / 0.01806359) / math.sqrt(2))))
    z += -0.4922728 * (1.234622 * 0.5 * ((1.362094 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.362094 - Q.lep_iso) / 1.234622) / math.sqrt(2))))
    z += -0.002825944 * (11.49171 * 0.5 * ((71.96396 - Q.mass) / 11.49171) * (1 + math.erf(((71.96396 - Q.mass) / 11.49171) / math.sqrt(2))))
    z += -2.923323 * (0.02550179 * 0.5 * ((0.1792389 - Q.sdb_2_z) / 0.02550179) * (1 + math.erf(((0.1792389 - Q.sdb_2_z) / 0.02550179) / math.sqrt(2))))
    z += -0.0006623594 * (137.5 * 0.5 * ((702.0 - Q.n_pairs_kt_above_1) / 137.5) * (1 + math.erf(((702.0 - Q.n_pairs_kt_above_1) / 137.5) / math.sqrt(2))))
    z += -0.03527974 * (1.0 * 0.5 * ((6.0 - Q.n_dr_0p1_0p2) / 1.0) * (1 + math.erf(((6.0 - Q.n_dr_0p1_0p2) / 1.0) / math.sqrt(2))))
    z += -0.5375839 * (0.04339825 * 0.5 * ((0.5454864 - Q.sj2_dr) / 0.04339825) * (1 + math.erf(((0.5454864 - Q.sj2_dr) / 0.04339825) / math.sqrt(2))))
    z += -0.04380911 * (0.04339825 * 0.5 * ((0.5454864 - Q.sj2_dr) / 0.04339825) * (1 + math.erf(((0.5454864 - Q.sj2_dr) / 0.04339825) / math.sqrt(2)))) * (0.0820479 * 0.5 * ((4.855929 - Q.lne_1) / 0.0820479) * (1 + math.erf(((4.855929 - Q.lne_1) / 0.0820479) / math.sqrt(2))))
    z += 0.1138483 * (1.0 * 0.5 * ((0.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.007587729 * (5.286881 * 0.5 * ((100.4835 - Q.mass) / 5.286881) * (1 + math.erf(((100.4835 - Q.mass) / 5.286881) / math.sqrt(2))))
    z += -0.02301337 * (4.488781 * 0.5 * ((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) * (1 + math.erf(((125.8718 - Q.mres_sd_mass_b0z005) / 4.488781) / math.sqrt(2))))
    z += 0.05214899 * (5.31786 * 0.5 * ((91.82593 - Q.mres_sd_mass_b0z005) / 5.31786) * (1 + math.erf(((91.82593 - Q.mres_sd_mass_b0z005) / 5.31786) / math.sqrt(2))))
    z += 0.007544399 * (14.42334 * 0.5 * ((141.5589 - Q.mres_sd_mass_b0z005) / 14.42334) * (1 + math.erf(((141.5589 - Q.mres_sd_mass_b0z005) / 14.42334) / math.sqrt(2))))
    z += -0.02958742 * (6.235814 * 0.5 * ((76.40585 - Q.mres_sd_mass_b0z005) / 6.235814) * (1 + math.erf(((76.40585 - Q.mres_sd_mass_b0z005) / 6.235814) / math.sqrt(2))))
    z += -0.01742523 * (5.227956 * 0.5 * ((102.8576 - Q.mres_sd_mass_b0z005) / 5.227956) * (1 + math.erf(((102.8576 - Q.mres_sd_mass_b0z005) / 5.227956) / math.sqrt(2))))
    z += -1.542915 * (0.07655927 * 0.5 * ((Q.sj3_dr12 - 0.5966255) / 0.07655927) * (1 + math.erf(((Q.sj3_dr12 - 0.5966255) / 0.07655927) / math.sqrt(2))))
    z += -0.01846816 * (0.5788858 * 0.5 * ((1.777286 - Q.mass_displaced3) / 0.5788858) * (1 + math.erf(((1.777286 - Q.mass_displaced3) / 0.5788858) / math.sqrt(2))))
    z += 0.009996858 * (5.898438 * 0.5 * ((10.52344 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.52344 - Q.max_abs_d0) / 5.898438) / math.sqrt(2))))
    z += -0.04465485 * (1.5 * 0.5 * ((7.0 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.0 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2))))
    z += 0.00135556 * (34.56457 * 0.5 * ((84.11398 - Q.jd_sum_abs_sd0_top5) / 34.56457) * (1 + math.erf(((84.11398 - Q.jd_sum_abs_sd0_top5) / 34.56457) / math.sqrt(2))))
    z += 0.0007328284 * (12.30925 * 0.5 * ((Q.mass_displaced3 - 39.09615) / 12.30925) * (1 + math.erf(((Q.mass_displaced3 - 39.09615) / 12.30925) / math.sqrt(2))))
    z += 2.158106 * (0.01806359 * 0.5 * ((0.04865675 - Q.sjf_2_2_z_d3) / 0.01806359) * (1 + math.erf(((0.04865675 - Q.sjf_2_2_z_d3) / 0.01806359) / math.sqrt(2)))) * (0.007274976 * 0.5 * ((0.0 - Q.tdz_0) / 0.007274976) * (1 + math.erf(((0.0 - Q.tdz_0) / 0.007274976) / math.sqrt(2))))
    z += -0.08452936 * (0.02425343 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.1343792) / 0.02425343) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.1343792) / 0.02425343) / math.sqrt(2))))
    z += -27.19802 * (0.007727658 * 0.5 * ((0.006220408 - Q.psi_0p1) / 0.007727658) * (1 + math.erf(((0.006220408 - Q.psi_0p1) / 0.007727658) / math.sqrt(2))))
    z += -0.00437638 * (0.04808774 * 0.5 * ((Q.lepsj_3_dr - 0.09790963) / 0.04808774) * (1 + math.erf(((Q.lepsj_3_dr - 0.09790963) / 0.04808774) / math.sqrt(2))))
    z += -0.4985038 * (0.2857178 * 0.5 * ((Q.dc_3_charge - 0.6283033) / 0.2857178) * (1 + math.erf(((Q.dc_3_charge - 0.6283033) / 0.2857178) / math.sqrt(2))))
    z += -2.796039 * (0.02246636 * 0.5 * ((Q.z_photon - 0.1149688) / 0.02246636) * (1 + math.erf(((Q.z_photon - 0.1149688) / 0.02246636) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.dzerr_76 - 0.0) / 1e-06) * (1 + math.erf(((Q.dzerr_76 - 0.0) / 1e-06) / math.sqrt(2))))
    z += 0.001832486 * (0.02365497 * 0.5 * ((Q.psi_0p3 - 0.8413991) / 0.02365497) * (1 + math.erf(((Q.psi_0p3 - 0.8413991) / 0.02365497) / math.sqrt(2))))
    z += -0.08257402 * (1.459864 * 0.5 * ((2.219501 - Q.mres_sd_prong_mass2) / 1.459864) * (1 + math.erf(((2.219501 - Q.mres_sd_prong_mass2) / 1.459864) / math.sqrt(2))))
    z += -0.8569692 * (0.008959514 * 0.5 * ((0.2915083 - Q.C2_b05) / 0.008959514) * (1 + math.erf(((0.2915083 - Q.C2_b05) / 0.008959514) / math.sqrt(2))))
    z += -0.003900364 * (13.98191 * 0.5 * ((12.49716 - Q.sjf_2_1_mass_d3) / 13.98191) * (1 + math.erf(((12.49716 - Q.sjf_2_1_mass_d3) / 13.98191) / math.sqrt(2))))
    z += 3.16848 * (0.01355407 * 0.5 * ((0.03067242 - Q.kt2_1_z_disp3) / 0.01355407) * (1 + math.erf(((0.03067242 - Q.kt2_1_z_disp3) / 0.01355407) / math.sqrt(2))))
    z += 1.658217 * (0.01670814 * 0.5 * ((Q.z_displaced5 - 0.06245248) / 0.01670814) * (1 + math.erf(((Q.z_displaced5 - 0.06245248) / 0.01670814) / math.sqrt(2))))
    z += -12.71663 * (0.08408739 * 0.5 * ((0.0 - Q.sv_2_dr) / 0.08408739) * (1 + math.erf(((0.0 - Q.sv_2_dr) / 0.08408739) / math.sqrt(2))))
    z += -0.2050209 * (0.05547861 * 0.5 * ((Q.tdz_2 - -0.05727976) / 0.05547861) * (1 + math.erf(((Q.tdz_2 - -0.05727976) / 0.05547861) / math.sqrt(2))))
    z += 0.3740734 * (0.01804395 * 0.5 * ((0.1244388 - Q.mres_sd_zg_b1z01) / 0.01804395) * (1 + math.erf(((0.1244388 - Q.mres_sd_zg_b1z01) / 0.01804395) / math.sqrt(2))))
    z += 0.01476734 * (1.0 * 0.5 * ((9.0 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((9.0 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2))))
    z += -0.008581103 * (3.93783 * 0.5 * ((73.24742 - Q.sj3_pair_mass_max) / 3.93783) * (1 + math.erf(((73.24742 - Q.sj3_pair_mass_max) / 3.93783) / math.sqrt(2))))
    z += 0.00770205 * (18.66842 * 0.5 * ((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((159.9242 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2))))
    z += 16.27306 * (0.000494246 * 0.5 * ((Q.ecf_g31 - 0.006877516) / 0.000494246) * (1 + math.erf(((Q.ecf_g31 - 0.006877516) / 0.000494246) / math.sqrt(2))))
    z += -1.81818e-06 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.0) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.0) / 1.0) / math.sqrt(2)))) * (1.26387 * 0.5 * ((2.627723 - Q.ak02_2_sd0_3) / 1.26387) * (1 + math.erf(((2.627723 - Q.ak02_2_sd0_3) / 1.26387) / math.sqrt(2))))
    z += 0.09899092 * (0.03580654 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.2769039) / 0.03580654) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.2769039) / 0.03580654) / math.sqrt(2)))) * (0.7649624 * 0.5 * ((0.8036986 - Q.lepsj_3_maxsd0) / 0.7649624) * (1 + math.erf(((0.8036986 - Q.lepsj_3_maxsd0) / 0.7649624) / math.sqrt(2))))
    z += 0.0005980069 * (0.04808774 * 0.5 * ((Q.lepsj_3_dr - 0.09790963) / 0.04808774) * (1 + math.erf(((Q.lepsj_3_dr - 0.09790963) / 0.04808774) / math.sqrt(2)))) * (540.2081 * 0.5 * ((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.9572 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2))))
    z += 2.825578 * (0.05826336 * 0.5 * ((0.103005 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103005 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2))))
    z += 1.711156 * (0.0947145 * 0.5 * ((0.0947145 - Q.dc_4_z) / 0.0947145) * (1 + math.erf(((0.0947145 - Q.dc_4_z) / 0.0947145) / math.sqrt(2))))
    z += -0.7721061 * (0.7649624 * 0.5 * ((0.8036986 - Q.lepsj_3_maxsd0) / 0.7649624) * (1 + math.erf(((0.8036986 - Q.lepsj_3_maxsd0) / 0.7649624) / math.sqrt(2))))
    z += 18.82748 * (0.01610679 * 0.5 * ((0.02900919 - Q.lepsj_3_dr) / 0.01610679) * (1 + math.erf(((0.02900919 - Q.lepsj_3_dr) / 0.01610679) / math.sqrt(2))))
    z += 868.0759 * (0.0001198329 * 0.5 * ((0.0002536827 - Q.e3_b2) / 0.0001198329) * (1 + math.erf(((0.0002536827 - Q.e3_b2) / 0.0001198329) / math.sqrt(2))))
    return z


def neuron_125(Q):
    z = -1.19076e-06
    return z


def neuron_126(Q):
    z = 7.632629e-06
    return z


def neuron_127(Q):
    z = -1.9667e-07
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
