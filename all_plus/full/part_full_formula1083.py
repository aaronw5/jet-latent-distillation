"""Particle Transformer (ParT) jet tagger, JetClass, input features 'full': tuned on the network's predictions (from 100 if-statements per neuron, pruned), as if-statements.

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

Whole test file (2,000,000 jets): accuracy 75.26% (the network: 86.03%); same class as the network for 80.09% of jets.

Quantities:
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b05                 e3/e2² with β = 0.5
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3_b05                 e4·e2/e3² with β = 0.5
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b05                 e3/e2³ with β = 0.5
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3_b05                 e4·e2³/e3³ with β = 0.5
  Q.D3_b2                  e4·e2³/e3³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M2_b2                  ₁e₃/e2 with β = 2
  Q.M3                     generalized ECF ratio M3
  Q.M3_b2                  ₁e₄/₁e₃ with β = 2
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N2_b05                 ₂e₃/(e2)² with β = 0.5
  Q.N2_b2                  ₂e₃/(e2)² with β = 2
  Q.N3_b05                 ₂e₄/(₁e₃)² with β = 0.5
  Q.ak02_1_n_lep           hardest anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_1_z               pT share of the hardest anti-kT 0.2 subjet (0 if none)
  Q.ak02_2_charge          2nd anti-kT 0.2 subjet: Σ q √pT / √(Σ pT) of its particles
  Q.ak02_2_n_lep           2nd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_2_z               pT share of the 2nd anti-kT 0.2 subjet (0 if none)
  Q.ak02_3_n_disp3         3rd anti-kT 0.2 subjet: number of its tracks with d0/σ > 3
  Q.ak02_3_n_lep           3rd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_3_z               pT share of the 3rd anti-kT 0.2 subjet (0 if none)
  Q.ak02_dr12              distance between the pT-weighted centres of the hardest and 2nd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr13              distance between the pT-weighted centres of the hardest and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr23              distance between the pT-weighted centres of the 2nd and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_min12_jp          the smaller of the two hardest anti-kT 0.2 subjets: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.ak02_min12_n_disp3     the smaller of the two hardest anti-kT 0.2 subjets: number of its tracks with d0/σ > 3
  Q.ak02_n                 number of anti-kT R = 0.2 subjets with pT > 10 GeV
  Q.d0err_52               σ(d0), clipped to [0, 1] of particle 52 (ParT input; empty slot: 0)
  Q.dc_1_n_lep             hardest prong: number of its electrons and muons
  Q.dc_1_z                 pT share of the hardest prong (0 if none)
  Q.dc_1_z_disp3           hardest prong: pT share (of the jet) of its tracks with d0/σ > 3
  Q.dc_2_jp                2nd prong: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.dc_2_n_lep             2nd prong: number of its electrons and muons
  Q.dc_3_charge            3rd prong: Σ q √pT / √(Σ pT) of its particles
  Q.dc_3_n_lep             3rd prong: number of its electrons and muons
  Q.dc_3_z                 pT share of the 3rd prong (0 if none)
  Q.dc_4_z                 pT share of the 4th prong (0 if none)
  Q.dc_n                   number of prongs: reverse the C/A tree; ΔR ≤ 0.1 is a prong, a branch with < 10 % of the jet pT is dropped, else both branches are declustered
  Q.dc_ntag                number of the 4 hardest prongs whose 2nd largest d0/σ is above 3
  Q.dc_pair_mass_min       smallest mass (E) of two of the 4 hardest prongs [GeV] (0 if fewer than 2)
  Q.dc_split2_dr           ΔR of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_tag_2nd             second largest 2nd-largest d0/σ among the 4 hardest prongs (double tag)
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr02                   ΔR between particles 0 and 2
  Q.dr_15                  ΔR from the jet axis of particle 15 (ParT input; empty slot: 0)
  Q.dr_19                  ΔR from the jet axis of particle 19 (ParT input; empty slot: 0)
  Q.dr_29                  ΔR from the jet axis of particle 29 (ParT input; empty slot: 0)
  Q.dr_30                  ΔR from the jet axis of particle 30 (ParT input; empty slot: 0)
  Q.dr_8                   ΔR from the jet axis of particle 8 (ParT input; empty slot: 0)
  Q.dr_max_012             largest distance among the 3 hardest
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.e3                     energy correlation e3 (β=1, 32 hardest)
  Q.e3_b05                 energy correlation e3 with β = 0.5
  Q.e3_b2                  energy correlation e3 with β = 2
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 16 hardest)
  Q.e4_b05                 energy correlation e4 with β = 0.5
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.ecf_g31                generalized energy correlation ₁e₃ (β=1; products of the smallest angles)
  Q.ecf_g41                generalized energy correlation ₁e₄ (β=1; products of the smallest angles)
  Q.ecf_g42                generalized energy correlation ₂e₄ (β=1; products of the smallest angles)
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.ischhad_0              1 if a charged hadron of particle 0 (ParT input; empty slot: 0)
  Q.iselectron_1           1 if an electron of particle 1 (ParT input; empty slot: 0)
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
  Q.kt2_1_z                pT share of the hardest kT subjet (0 if none)
  Q.kt2_2_n_lep            2nd kT subjet: number of its electrons and muons
  Q.kt2_min12_jp           the smaller of the two hardest kT subjets: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.kt2_min12_n_disp3      the smaller of the two hardest kT subjets: number of its tracks with d0/σ > 3
  Q.ktd_ln_d34             ln of the exclusive-kT merging scale from 4 to 3 subjets, min(pT²)ΔR², over (Σ pT)² (0 if fewer particles)
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
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
  Q.lne_3                  ln E [GeV] of particle 3 (ParT input; empty slot: ln 1e-8)
  Q.lne_4                  ln E [GeV] of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lne_8                  ln E [GeV] of particle 8 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_4               ln(E / E of the jet) of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_42              ln(E / E of the jet) of particle 42 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_65              ln(E / E of the jet) of particle 65 (ParT input; empty slot: ln 1e-8)
  Q.lnpt_78                ln pT [GeV] of particle 78 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_2              ln(pT / pT of the jet) of particle 2 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_25             ln(pT / pT of the jet) of particle 25 (ParT input; empty slot: ln 1e-8)
  Q.lund1_lndelta          ln Δ of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund1_lnz              ln z of the 1. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lndelta          ln Δ of the 3. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lnkt             ln kT of the 3. primary C/A splitting (ln 1e-8 if none)
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
  Q.mres_pruned_mass       pruned mass: C/A tree, a merge with z < 0.1 and ΔR > m/pT of the jet keeps only its harder branch [GeV]
  Q.mres_sd_mass_b0z005    soft-drop mass, C/A, β = 0, z_cut = 0.05 [GeV]
  Q.mres_sd_mass_b0z02     soft-drop mass, C/A, β = 0, z_cut = 0.2 [GeV]
  Q.mres_sd_mass_b1z01     soft-drop mass, C/A, β = 1, z_cut = 0.1 [GeV]
  Q.mres_sd_mass_b2z01     soft-drop mass, C/A, β = 2, z_cut = 0.1 [GeV]
  Q.mres_sd_prong_mass1    mass of the harder branch at the soft-drop splitting (β = 0, z_cut = 0.1) [GeV]
  Q.mres_sd_prong_mass2    mass of the softer branch at the soft-drop splitting (β = 0, z_cut = 0.1) [GeV]
  Q.mres_sd_rg_b2z01       soft-drop R_g, β = 2, z_cut = 0.1
  Q.n_charged_had          number of charged hadrons
  Q.n_charged_pt_above_1   number of charged particles with pT > 1 GeV
  Q.n_charged_pt_above_10  number of charged particles with pT > 10 GeV
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
  Q.n_sd0_above_3          number of charged particles with d0 significance > 3
  Q.n_sd0_above_5          number of charged particles with d0 significance > 5
  Q.n_sdz_above_2          number of charged particles with dz significance > 2
  Q.n_sdz_above_5          number of charged particles with dz significance > 5
  Q.nca_kt_above_10        number of C/A subjets when every branching with kT = min(pT)·ΔR > 10 GeV is split
  Q.nca_sj4_pair2nd_over_mass second largest mass of two of the 4 subjets over the jet mass
  Q.nca_sj4_pair_mass_2nd  second largest mass of two of the 4 subjets [GeV]
  Q.nca_sj4_pairmax_over_mass largest mass of two of the 4 subjets over the jet mass
  Q.pair_max_lnkt          largest ln kT among all pairs
  Q.pair_max_lnm2          largest ln m² among all pairs
  Q.pair_mean_lndelta      zᵢzⱼ-weighted mean of ln ΔRᵢⱼ over all pairs
  Q.pair_mean_lnkt         zᵢzⱼ-weighted mean of ln kT = ln(min(pTᵢ, pTⱼ)·ΔRᵢⱼ) over all pairs
  Q.pair_mean_lnm2         zᵢzⱼ-weighted mean of ln mᵢⱼ² (massless) over all pairs
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
  Q.pz_lnkt3               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 2 < ln kT ≤ 3
  Q.sd_mass                soft-drop groomed mass, C/A on the 128 hardest, β=0, z_cut=0.1 [GeV]
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
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmax_over_m     largest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj4_dr_min             smallest distance among the 4 subjet axes
  Q.sj4_pair_mass_max      largest mass of two of the 4 subjets [GeV]
  Q.sj4_pair_mass_min      smallest mass of two of the 4 subjets [GeV]
  Q.sjf_2_1_max3d          subjet 1 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_1_maxsd0         subjet 1 of 2 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_2_1_n_d3           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_2_1_n_d5           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_2_1_z_d3           subjet 1 of 2 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_2_2_max3d          subjet 2 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_2_maxsd0         subjet 2 of 2 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_2_2_n_d5           subjet 2 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_2_2_z_d3           subjet 2 of 2 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_3_1_max3d          subjet 1 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_1_n_d3           subjet 1 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_3_1_n_d5           subjet 1 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_3_1_z_d3           subjet 1 of 3 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_3_2_max3d          subjet 2 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_2_maxsd0         subjet 2 of 3 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_3_3_max3d          subjet 3 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_3_maxsd0         subjet 3 of 3 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_3_3_n_d3           subjet 3 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_3_3_z_d3           subjet 3 of 3 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_1_max3d          subjet 1 of 4 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_4_1_n_d3           subjet 1 of 4 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_4_1_z_d3           subjet 1 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_2_z_d3           subjet 2 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
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
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sv_1_dr                hardest displaced-track cluster: ΔR of its centre from the jet axis (0 if none)
  Q.sv_1_n                 hardest displaced-track cluster: number of tracks (0 if none)
  Q.sv_1_sd0_sum           hardest displaced-track cluster: Σ d0/σ of its tracks (0 if none)
  Q.sv_2_dr                2nd displaced-track cluster: ΔR of its centre from the jet axis (0 if none)
  Q.sv_2_n                 2nd displaced-track cluster: number of tracks (0 if none)
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
  Q.tdz_1                  tanh(dz [mm]) of particle 1 (ParT input; empty slot: 0)
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
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b05=ecfb('e3', 0.5) / max(ecfb('e2', 0.5) ** 3, 1e-30),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3_b05=ecfb('e4', 0.5) * ecfb('e2', 0.5) ** 3 / max(ecfb('e3', 0.5) ** 3, 1e-30),
        D3_b2=ecfb('e4', 2) * ecfb('e2', 2) ** 3 / max(ecfb('e3', 2) ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M2_b2=ecfb('g31', 2) / max(ecfb('e2', 2), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        M3_b2=ecfb('g41', 2) / max(ecfb('g31', 2), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N2_b05=ecfb('g32', 0.5) / max(ecfb('e2', 0.5) ** 2, 1e-30),
        N2_b2=ecfb('g32', 2) / max(ecfb('e2', 2) ** 2, 1e-30),
        N3_b05=ecfb('g42', 0.5) / max(ecfb('g31', 0.5) ** 2, 1e-30),
        ak02_1_n_lep=pq('ak02_1_n_lep'),
        ak02_1_z=pq('ak02_1_z'),
        ak02_2_charge=pq('ak02_2_charge'),
        ak02_2_n_lep=pq('ak02_2_n_lep'),
        ak02_2_z=pq('ak02_2_z'),
        ak02_3_n_disp3=pq('ak02_3_n_disp3'),
        ak02_3_n_lep=pq('ak02_3_n_lep'),
        ak02_3_z=pq('ak02_3_z'),
        ak02_dr12=pq('ak02_dr12'),
        ak02_dr13=pq('ak02_dr13'),
        ak02_dr23=pq('ak02_dr23'),
        ak02_min12_jp=pq('ak02_min12_jp'),
        ak02_min12_n_disp3=pq('ak02_min12_n_disp3'),
        ak02_n=pq('ak02_n'),
        d0err_52=pfeat(52, 'd0err'),
        dc_1_n_lep=pq('dc_1_n_lep'),
        dc_1_z=pq('dc_1_z'),
        dc_1_z_disp3=pq('dc_1_z_disp3'),
        dc_2_jp=pq('dc_2_jp'),
        dc_2_n_lep=pq('dc_2_n_lep'),
        dc_3_charge=pq('dc_3_charge'),
        dc_3_n_lep=pq('dc_3_n_lep'),
        dc_3_z=pq('dc_3_z'),
        dc_4_z=pq('dc_4_z'),
        dc_n=pq('dc_n'),
        dc_ntag=pq('dc_ntag'),
        dc_pair_mass_min=pq('dc_pair_mass_min'),
        dc_split2_dr=pq('dc_split2_dr'),
        dc_tag_2nd=pq('dc_tag_2nd'),
        dr01=math.sqrt(dist2(0, 1)),
        dr02=math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        dr_15=pfeat(15, 'dr'),
        dr_19=pfeat(19, 'dr'),
        dr_29=pfeat(29, 'dr'),
        dr_30=pfeat(30, 'dr'),
        dr_8=pfeat(8, 'dr'),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        e3=ecf('e3'),
        e3_b05=ecfb('e3', 0.5),
        e3_b2=ecfb('e3', 2),
        e4=ecf('e4'),
        e4_b05=ecfb('e4', 0.5),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        ecf_g31=ecfb('g31', 1),
        ecf_g41=ecfb('g41', 1),
        ecf_g42=ecfb('g42', 1),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        ischhad_0=pfeat(0, 'ischhad'),
        iselectron_1=pfeat(1, 'iselectron'),
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
        kt2_1_z=pq('kt2_1_z'),
        kt2_2_n_lep=pq('kt2_2_n_lep'),
        kt2_min12_jp=pq('kt2_min12_jp'),
        kt2_min12_n_disp3=pq('kt2_min12_n_disp3'),
        ktd_ln_d34=pq('ktd_ln_d34'),
        lam1=lam1,
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
        lne_3=pfeat(3, 'lne'),
        lne_4=pfeat(4, 'lne'),
        lne_8=pfeat(8, 'lne'),
        lnerel_4=pfeat(4, 'lnerel'),
        lnerel_42=pfeat(42, 'lnerel'),
        lnerel_65=pfeat(65, 'lnerel'),
        lnpt_78=pfeat(78, 'lnpt'),
        lnptrel_2=pfeat(2, 'lnptrel'),
        lnptrel_25=pfeat(25, 'lnptrel'),
        lund1_lndelta=lund(1, 'lndelta'),
        lund1_lnz=lund(1, 'lnz'),
        lund3_lndelta=lund(3, 'lndelta'),
        lund3_lnkt=lund(3, 'lnkt'),
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
        mres_pruned_mass=pq('mres_pruned_mass'),
        mres_sd_mass_b0z005=pq('mres_sd_mass_b0z005'),
        mres_sd_mass_b0z02=pq('mres_sd_mass_b0z02'),
        mres_sd_mass_b1z01=pq('mres_sd_mass_b1z01'),
        mres_sd_mass_b2z01=pq('mres_sd_mass_b2z01'),
        mres_sd_prong_mass1=pq('mres_sd_prong_mass1'),
        mres_sd_prong_mass2=pq('mres_sd_prong_mass2'),
        mres_sd_rg_b2z01=pq('mres_sd_rg_b2z01'),
        n_charged_had=sum(1 for i in real if ptype[i] == 1),
        n_charged_pt_above_1=sum(1 for i in real if charge[i] != 0 and pt[i] > 1),
        n_charged_pt_above_10=sum(1 for i in real if charge[i] != 0 and pt[i] > 10),
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
        n_sd0_above_3=nsig('d0', 3),
        n_sd0_above_5=nsig('d0', 5),
        n_sdz_above_2=nsig('dz', 2),
        n_sdz_above_5=nsig('dz', 5),
        nca_kt_above_10=pq('nca_kt_above_10'),
        nca_sj4_pair2nd_over_mass=pq('nca_sj4_pair2nd_over_mass'),
        nca_sj4_pair_mass_2nd=pq('nca_sj4_pair_mass_2nd'),
        nca_sj4_pairmax_over_mass=pq('nca_sj4_pairmax_over_mass'),
        pair_max_lnkt=pairmax('lnkt'),
        pair_max_lnm2=pairmax('lnm2'),
        pair_mean_lndelta=pairsum('lndelta'),
        pair_mean_lnkt=pairsum('lnkt'),
        pair_mean_lnm2=pairsum('lnm2'),
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
        pz_lnkt3=pq('pz_lnkt3'),
        sd_mass=softdrop("mass"),
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
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr_min=min(subjets(3)["dr"]),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass3=subjets(3)["mass"][2],
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmax_over_m=max(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj4_dr_min=min(subjets(4)["dr"]),
        sj4_pair_mass_max=max(subjets(4)["mpair"]),
        sj4_pair_mass_min=min(subjets(4)["mpair"]),
        sjf_2_1_max3d=pq('sjf_2_1_max3d'),
        sjf_2_1_maxsd0=pq('sjf_2_1_maxsd0'),
        sjf_2_1_n_d3=pq('sjf_2_1_n_d3'),
        sjf_2_1_n_d5=pq('sjf_2_1_n_d5'),
        sjf_2_1_z_d3=pq('sjf_2_1_z_d3'),
        sjf_2_2_max3d=pq('sjf_2_2_max3d'),
        sjf_2_2_maxsd0=pq('sjf_2_2_maxsd0'),
        sjf_2_2_n_d5=pq('sjf_2_2_n_d5'),
        sjf_2_2_z_d3=pq('sjf_2_2_z_d3'),
        sjf_3_1_max3d=pq('sjf_3_1_max3d'),
        sjf_3_1_n_d3=pq('sjf_3_1_n_d3'),
        sjf_3_1_n_d5=pq('sjf_3_1_n_d5'),
        sjf_3_1_z_d3=pq('sjf_3_1_z_d3'),
        sjf_3_2_max3d=pq('sjf_3_2_max3d'),
        sjf_3_2_maxsd0=pq('sjf_3_2_maxsd0'),
        sjf_3_3_max3d=pq('sjf_3_3_max3d'),
        sjf_3_3_maxsd0=pq('sjf_3_3_maxsd0'),
        sjf_3_3_n_d3=pq('sjf_3_3_n_d3'),
        sjf_3_3_z_d3=pq('sjf_3_3_z_d3'),
        sjf_4_1_max3d=pq('sjf_4_1_max3d'),
        sjf_4_1_n_d3=pq('sjf_4_1_n_d3'),
        sjf_4_1_z_d3=pq('sjf_4_1_z_d3'),
        sjf_4_2_z_d3=pq('sjf_4_2_z_d3'),
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
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sv_1_dr=pq('sv_1_dr'),
        sv_1_n=pq('sv_1_n'),
        sv_1_sd0_sum=pq('sv_1_sd0_sum'),
        sv_2_dr=pq('sv_2_dr'),
        sv_2_n=pq('sv_2_n'),
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
        tdz_1=pfeat(1, 'tdz'),
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
    z = 0.005187402
    return z


def neuron_1(Q):
    z = -5.446781e-06
    return z


def neuron_2(Q):
    z = 4.874348e-06
    return z


def neuron_3(Q):
    z = 8.840496e-06
    return z


def neuron_4(Q):
    z = 1.996732e-06
    return z


def neuron_5(Q):
    z = -1.930161e-06
    return z


def neuron_6(Q):
    z = 1.685008e-05
    return z


def neuron_7(Q):
    z = 6.81118e-06
    return z


def neuron_8(Q):
    z = 2.929219e-06
    return z


def neuron_9(Q):
    z = -2.514038e-06
    return z


def neuron_10(Q):
    z = -1.605471e-05
    return z


def neuron_11(Q):
    z = -1.837035e-05
    return z


def neuron_12(Q):
    z = 5.449053e-06
    return z


def neuron_13(Q):
    z = 4.547841e-06
    return z


def neuron_14(Q):
    z = 2.656179e-06
    return z


def neuron_15(Q):
    z = -7.408109e-06
    return z


def neuron_16(Q):
    z = -0.3696972
    if Q.lep_ptrel < 27.3236:
        z += 0.03053298 * Q.lep_ptrel + 1.961167
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.1023085 * Q.lep_ptrel
    if Q.lep_ptrel >= 43.20788:
        z += 0.002334371 * Q.lep_ptrel + 4.319672
    if Q.n_s3d_above_3 < 10.0:
        z += 0.2129686 * Q.n_s3d_above_3 - 2.129686
    if Q.pz_lnd2 < 0.08321691:
        z += -3.372466 * Q.pz_lnd2 + 0.2806462
    if Q.sip_3d_2 < 226.3008:
        z += -0.002106152 * Q.sip_3d_2 + 0.508992
    if 226.3008 <= Q.sip_3d_2 < 447.0873:
        z += -0.0001466045 * Q.sip_3d_2 + 0.06554499
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.005434111 * Q.n_pairs_kt_above_3 + 0.4658415
    if 28.0 <= Q.n_pairs_kt_above_3 < 34.0:
        z += -0.05228106 * Q.n_pairs_kt_above_3 + 1.777556
    if Q.mass_neutral >= 34.02385:
        z += 0.004971162 * Q.mass_neutral - 0.1691381
    if Q.z_displaced5 < 0.2801368:
        z += 3.941668 * Q.z_displaced5 - 1.104206
    if Q.e3_b2 < 0.0004126585:
        z += -180.6961 * Q.e3_b2 + 0.265615
    if 0.0004126585 <= Q.e3_b2 < 0.0007909605:
        z += -505.0179 * Q.e3_b2 + 0.3994492
    if Q.pz_lnd0 < 0.156893:
        z += 1.059416 * Q.pz_lnd0 - 0.166215
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.004492069 * Q.mres_sd_mass_b0z005 - 0.5484816
    if 159.9242 <= Q.mres_sd_mass_b0z005 < 178.8957:
        z += -0.03079577 * Q.mres_sd_mass_b0z005 + 5.094899
    if Q.mres_sd_mass_b0z005 >= 178.8957:
        z += -0.01379245 * Q.mres_sd_mass_b0z005 + 2.053079
    z += -4.032047 * Q.z_charged_had
    if Q.mres_sd_mass_b2z01 < 91.15481:
        z += -0.004861841 * Q.mres_sd_mass_b2z01 + 0.4431802
    if 20.76537 <= Q.mass_2charged < 45.73288:
        z += 0.006179129 * Q.mass_2charged - 0.1283119
    if Q.mass_2charged >= 45.73288:
        z += -0.006811732 * Q.mass_2charged + 0.4657975
    if Q.max_abs_d0 < 5.8125:
        z += -0.04079924 * Q.max_abs_d0 + 0.2371456
    if Q.sdb_2_n < 16.0:
        z += -0.02315893 * Q.sdb_2_n + 0.3705428
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.01106909 * Q.n_pairs_kt_above_1 + 0.885527
    if Q.z_dr_0p4_up < 0.01553665:
        z += 4.951855 * Q.z_dr_0p4_up - 0.07693523
    if Q.mass_top5 >= 62.84493:
        z += 0.003274912 * Q.mass_top5 - 0.2058116
    if Q.mass_charged < 62.00562:
        z += -0.01607043 * Q.mass_charged + 0.9964571
    if Q.sjq_2_sumabs_k1 >= 0.2432922:
        z += 0.9118496 * Q.sjq_2_sumabs_k1 - 0.2218459
    if Q.ak02_min12_n_disp3 >= 1.0:
        z += 0.09279014 * Q.ak02_min12_n_disp3 - 0.09279014
    if Q.sjq_2_prod_k1 < 0.09802954:
        z += -2.080868 * Q.sjq_2_prod_k1 + 0.2039866
    if Q.mres_sd_prong_mass1 < 46.61784:
        z += -0.003236186 * Q.mres_sd_prong_mass1 + 0.150864
    if Q.dr_min_012 < 0.01841101:
        z += 9.535737 * Q.dr_min_012 - 0.1755625
    if Q.jd_3d_6 >= 30.57088:
        z += -0.0008290402 * Q.jd_3d_6 + 0.02534449
    if Q.mass_top40 >= 112.406:
        z += 0.01635017 * Q.mass_top40 - 1.837858
    if Q.sjf_3_2_max3d < 3.101398:
        z += -0.04857201 * Q.sjf_3_2_max3d + 0.1506411
    if Q.n_photon < 18.0:
        z += 0.02775258 * Q.n_photon - 0.4995464
    if Q.n_dr_0p1_0p2 < 6.0:
        z += -0.04021418 * Q.n_dr_0p1_0p2 + 0.2412851
    z += 0.04882054 * Q.n_sdz_above_2
    if Q.sip_3d_3 < 175.9957:
        z += -0.00177982 * Q.sip_3d_3 + 0.3132406
    if Q.jd_3d_4 < 14.85973:
        z += 0.02999397 * Q.jd_3d_4 - 0.4457021
    if Q.z_dr_0p05_0p1 < 0.01556887:
        z += -10.80554 * Q.z_dr_0p05_0p1 + 0.16823
    if Q.M3 >= 0.02716047:
        z += 7.947107 * Q.M3 - 0.2158472
    if Q.lne_4 < 3.577424:
        z += -0.09007274 * Q.lne_4 + 0.3222284
    if Q.sj2_mass2 < 7.54918:
        z += 0.03541807 * Q.sj2_mass2 - 0.2673774
    if Q.sjq_2_2_nch < 3.0:
        z += -0.08527683 * Q.sjq_2_2_nch + 0.2558305
    if Q.D3_b05 < 0.8860453:
        z += 0.04306287 * Q.D3_b05 - 0.03815565
    if Q.z_electron < 0.1373477:
        z += 1.78437 * Q.z_electron - 0.245079
    if Q.max_abs_dz < 0.4436035:
        z += 0.5196241 * Q.max_abs_dz - 0.01860968
    if 0.4436035 <= Q.max_abs_dz < 2.957031:
        z += -0.08430614 * Q.max_abs_dz + 0.2492959
    if Q.sdb_0_n >= 3.0:
        z += 0.03605798 * Q.sdb_0_n - 0.1081739
    if Q.sjf_4_1_n_d3 >= 2.0:
        z += 0.03056462 * Q.sjf_4_1_n_d3 - 0.06112925
    if Q.mass_displaced3 < 1.777286:
        z += 0.2488586 * Q.mass_displaced3 - 0.4422928
    if Q.mres_sd_rg_b2z01 >= 0.457968:
        z += 0.7942908 * Q.mres_sd_rg_b2z01 - 0.3637598
    if Q.tdz_1 < -0.1170754:
        z += -0.2979172 * Q.tdz_1 - 0.03487877
    if Q.sdb_2_z < 0.1530389:
        z += -0.393734 * Q.sdb_2_z + 0.06025663
    if Q.D2 < 1.00791:
        z += 0.1027817 * Q.D2 - 0.1035947
    if Q.jd_sum_abs_sd0_top5 < 9.408315:
        z += 0.1273796 * Q.jd_sum_abs_sd0_top5 - 1.198427
    if Q.mres_sd_mass_b0z02 >= 93.75785:
        z += 0.001242312 * Q.mres_sd_mass_b0z02 - 0.1164765
    if Q.ak02_dr23 >= 0.3194837:
        z += 0.1872559 * Q.ak02_dr23 - 0.05982521
    if Q.sjf_2_2_maxsd0 < 1.141642:
        z += -0.1293533 * Q.sjf_2_2_maxsd0 + 0.1476751
    if Q.sjf_2_1_maxsd0 < 1.113424:
        z += -0.2041775 * Q.sjf_2_1_maxsd0 + 0.227336
    if Q.ecf_g41 < 0.0002703514:
        z += 406.6612 * Q.ecf_g41 - 0.1099414
    if Q.n_pairs_kt_above_10 < 7.0:
        z += -0.03512504 * Q.n_pairs_kt_above_10 + 0.2458753
    if Q.N2_b05 < 0.302405:
        z += 1.354863 * Q.N2_b05 - 0.4097172
    if Q.N2 < 0.1506626:
        z += 2.020272 * Q.N2 - 0.3043794
    if Q.sjf_4_n2disp < 2.0:
        z += -0.03854214 * Q.sjf_4_n2disp + 0.07708428
    if Q.z_muon < 0.03898651:
        z += 4.55992 * Q.z_muon - 0.1777754
    if Q.lep_z < 0.221436:
        z += -2.74865 * Q.lep_z + 1.425808
    if 0.221436 <= Q.lep_z < 0.5187302:
        z += -4.821064 * Q.lep_z + 1.884715
    if Q.lep_z >= 0.5187302:
        z += -2.072414 * Q.lep_z + 0.458907
    if Q.z_charged < 0.8615361:
        z += 3.553372 * Q.z_charged - 3.061359
    if Q.n_pairs_kt_above_3 < 28.0 and Q.kt2_2_n_lep < 1.0:
        z += -0.003315325 * (28.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.kt2_2_n_lep)
    if Q.z_displaced5 < 0.2801368 and Q.jd_3d_4 < 172.888:
        z += 0.02074894 * (0.2801368 - Q.z_displaced5) * (172.888 - Q.jd_3d_4)
    if Q.n_s3d_above_3 < 10.0 and Q.n_neutral > 10.0:
        z += 0.001267649 * (10.0 - Q.n_s3d_above_3) * (Q.n_neutral - 10.0)
    if Q.n_s3d_above_3 < 10.0 and Q.sv_1_n > 1.0:
        z += 0.05789472 * (10.0 - Q.n_s3d_above_3) * (Q.sv_1_n - 1.0)
    if Q.e3_b2 < 0.0004126585 and Q.n_lund > 5.0:
        z += 23.98837 * (0.0004126585 - Q.e3_b2) * (Q.n_lund - 5.0)
    if Q.n_s3d_above_3 < 10.0 and Q.sdb_4_z < 0.1270192:
        z += 0.1797919 * (10.0 - Q.n_s3d_above_3) * (0.1270192 - Q.sdb_4_z)
    if Q.n_pairs_kt_above_1 < 80.0 and Q.tau32 < 0.6513932:
        z += -0.01415154 * (80.0 - Q.n_pairs_kt_above_1) * (0.6513932 - Q.tau32)
    if Q.mass_neutral > 34.02385 and Q.jd_3d_5 < 2.912651:
        z += -0.005709175 * (Q.mass_neutral - 34.02385) * (2.912651 - Q.jd_3d_5)
    if Q.sjq_2_prod_k1 < 0.09802954 and Q.ak02_2_n_lep < 1.0:
        z += -0.6848952 * (0.09802954 - Q.sjq_2_prod_k1) * (1.0 - Q.ak02_2_n_lep)
    if Q.sdb_2_n < 16.0 and Q.tau43 < 0.8479222:
        z += -0.04071808 * (16.0 - Q.sdb_2_n) * (0.8479222 - Q.tau43)
    if Q.sjq_2_sumabs_k1 > 0.2432922 and Q.n_lepton < 2.0:
        z += -0.1917227 * (Q.sjq_2_sumabs_k1 - 0.2432922) * (2.0 - Q.n_lepton)
    if Q.sdb_2_n < 16.0 and Q.dc_pair_mass_min < 105.4151:
        z += -6.556751e-05 * (16.0 - Q.sdb_2_n) * (105.4151 - Q.dc_pair_mass_min)
    if Q.sip_3d_2 < 447.0873 and Q.lepsj_2_n_d3 > 2.0:
        z += -0.0002499755 * (447.0873 - Q.sip_3d_2) * (Q.lepsj_2_n_d3 - 2.0)
    if Q.sip_3d_3 < 175.9957 and Q.mres_sd_prong_mass2 > 1.685874e-06:
        z += -3.531584e-05 * (175.9957 - Q.sip_3d_3) * (Q.mres_sd_prong_mass2 - 1.685874e-06)
    if Q.sjq_2_prod_k1 < 0.09802954 and Q.dc_1_z_disp3 < 0.200234:
        z += -2.412184 * (0.09802954 - Q.sjq_2_prod_k1) * (0.200234 - Q.dc_1_z_disp3)
    if Q.z_displaced5 < 0.2801368 and Q.n_lund_kt_above_5 > 1.0:
        z += -0.3230492 * (0.2801368 - Q.z_displaced5) * (Q.n_lund_kt_above_5 - 1.0)
    if Q.n_s3d_above_3 < 10.0 and Q.sjf_4_3_z_d3 > 0.07303924:
        z += 0.214498 * (10.0 - Q.n_s3d_above_3) * (Q.sjf_4_3_z_d3 - 0.07303924)
    if Q.e3_b2 < 0.0004126585 and Q.mass_2photon > 22.18431:
        z += 38.93025 * (0.0004126585 - Q.e3_b2) * (Q.mass_2photon - 22.18431)
    if Q.sdb_2_n < 16.0 and Q.sjq_3_sumabs_k03 > 0.5318777:
        z += 0.007240666 * (16.0 - Q.sdb_2_n) * (Q.sjq_3_sumabs_k03 - 0.5318777)
    if Q.dr_min_012 < 0.01841101 and Q.sjf_4_1_max3d < 3.652671:
        z += 3.326822 * (0.01841101 - Q.dr_min_012) * (3.652671 - Q.sjf_4_1_max3d)
    if Q.sjf_2_2_maxsd0 < 1.141642 and Q.iselectron_1 > 0.0:
        z += -0.2509584 * (1.141642 - Q.sjf_2_2_maxsd0) * (Q.iselectron_1 - 0.0)
    if Q.M3 > 0.02716047 and Q.jd_mass_d3_over_z < 486.2148:
        z += 0.002631475 * (Q.M3 - 0.02716047) * (486.2148 - Q.jd_mass_d3_over_z)
    return z


def neuron_17(Q):
    z = -1.078667e-05
    return z


def neuron_18(Q):
    z = 2.065392
    if Q.lep_z < 0.004135872:
        z += 137.357 * Q.lep_z - 1.579206
    if 0.004135872 <= Q.lep_z < 0.221436:
        z += 4.653082 * Q.lep_z - 1.03036
    if 100.4835 <= Q.mass < 110.2019:
        z += -0.01258089 * Q.mass + 1.264172
    if 110.2019 <= Q.mass < 164.4374:
        z += -0.02078123 * Q.mass + 2.167865
    if 164.4374 <= Q.mass < 182.8592:
        z += 0.01814914 * Q.mass - 4.233746
    if Q.mass >= 182.8592:
        z += -0.0097494 * Q.mass + 0.867759
    if Q.mass_top20 < 114.4658:
        z += 0.003132047 * Q.mass_top20 - 0.3585123
    if Q.lepsj_3_n_d3 < 2.0:
        z += -0.4474297 * Q.lepsj_3_n_d3 + 1.205112
    if 2.0 <= Q.lepsj_3_n_d3 < 3.0:
        z += -0.3102525 * Q.lepsj_3_n_d3 + 0.9307576
    if Q.z_displaced3 < 0.03624058:
        z += 13.50319 * Q.z_displaced3 - 0.4893635
    if Q.z_displaced3 >= 0.2092108:
        z += -1.841352 * Q.z_displaced3 + 0.3852307
    if Q.n_s3d_above_10 < 6.0:
        z += -0.03650812 * Q.n_s3d_above_10 + 0.2190487
    if Q.mass_displaced3 < 2.409613:
        z += -0.1316457 * Q.mass_displaced3 + 0.4158991
    if 2.409613 <= Q.mass_displaced3 < 9.090532:
        z += -0.01477103 * Q.mass_displaced3 + 0.1342765
    if Q.lep_iso < 1.362094:
        z += -0.1608644 * Q.lep_iso - 0.0833993
    if 1.362094 <= Q.lep_iso < 6.185635:
        z += 0.06271572 * Q.lep_iso - 0.3879366
    if Q.z_displaced5 < 0.08090366:
        z += -8.459662 * Q.z_displaced5 + 0.6844177
    if Q.lund3_lndelta >= -2.4279:
        z += -0.07239264 * Q.lund3_lndelta - 0.1757621
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.179528 * Q.lepsj_3_maxsd0 - 0.3470567
    if 2.413264 <= Q.lepsj_3_maxsd0 < 46.74905:
        z += 0.0100135 * Q.lepsj_3_maxsd0 - 0.8044704
    if 46.74905 <= Q.lepsj_3_maxsd0 < 586.9572:
        z += 0.0006226277 * Q.lepsj_3_maxsd0 - 0.3654558
    if 111.4443 <= Q.mres_sd_mass_b2z01 < 121.6067:
        z += 0.01233559 * Q.mres_sd_mass_b2z01 - 1.374731
    if 121.6067 <= Q.mres_sd_mass_b2z01 < 177.5398:
        z += -0.005443642 * Q.mres_sd_mass_b2z01 + 0.7873432
    if Q.mres_sd_mass_b2z01 >= 177.5398:
        z += 0.01471937 * Q.mres_sd_mass_b2z01 - 2.792394
    if Q.n_s3d_above_3 < 3.0:
        z += 0.06460419 * Q.n_s3d_above_3 - 0.4482218
    if 3.0 <= Q.n_s3d_above_3 < 10.0:
        z += 0.03634417 * Q.n_s3d_above_3 - 0.3634417
    if Q.mass_displaced5 < 1.262841:
        z += 0.02513734 * Q.mass_displaced5 - 0.03174445
    if Q.ak02_1_z < 0.849996:
        z += 0.4802513 * Q.ak02_1_z - 0.4082117
    if Q.sj3_pair_mass_max >= 128.6605:
        z += -0.01684031 * Q.sj3_pair_mass_max + 2.166684
    if Q.n_dr_0p4_up < 4.0:
        z += -0.03842198 * Q.n_dr_0p4_up + 0.1536879
    if Q.lam1 < 0.01299032:
        z += 62.33569 * Q.lam1 - 0.8097607
    if Q.n_real_top50 >= 26.0:
        z += -0.01832421 * Q.n_real_top50 + 0.4764294
    if Q.M2 >= 0.07013948:
        z += 2.69369 * Q.M2 - 0.188934
    if Q.sv_1_sd0_sum < 51.56287:
        z += -0.003430254 * Q.sv_1_sd0_sum + 0.1768737
    if Q.sjf_3_1_max3d < 2.078395:
        z += -0.06496604 * Q.sjf_3_1_max3d - 0.2266515
    if 2.078395 <= Q.sjf_3_1_max3d < 2.924357:
        z += 0.05662275 * Q.sjf_3_1_max3d - 0.479361
    if 2.924357 <= Q.sjf_3_1_max3d < 3.710567:
        z += 0.7146657 * Q.sjf_3_1_max3d - 2.403714
    if 3.710567 <= Q.sjf_3_1_max3d < 71.08287:
        z += -0.003682549 * Q.sjf_3_1_max3d + 0.2617662
    if Q.sjf_2_2_max3d < 4.516968:
        z += 0.2099988 * Q.sjf_2_2_max3d - 0.5704971
    if 4.516968 <= Q.sjf_2_2_max3d < 55.92931:
        z += -0.007353503 * Q.sjf_2_2_max3d + 0.4112763
    if Q.sjf_2_1_z_d3 < 0.0828821:
        z += 3.395895 * Q.sjf_2_1_z_d3 - 0.2814589
    if Q.sjq_2_1_nch < 16.0:
        z += -0.02747231 * Q.sjq_2_1_nch + 0.439557
    if Q.sj2_mass2 < 6.465318:
        z += -0.06215991 * Q.sj2_mass2 + 0.4018836
    if Q.n_sdz_above_5 < 3.0:
        z += -0.07997458 * Q.n_sdz_above_5 + 0.2399237
    if Q.sv_2_n >= 1.0:
        z += 0.05973116 * Q.sv_2_n - 0.05973116
    if Q.z_charged_had < 0.3603262:
        z += -1.874397 * Q.z_charged_had + 0.6753944
    if Q.n_for_90pct < 29.0:
        z += 0.03059807 * Q.n_for_90pct - 0.8873441
    if Q.tau5 < 0.03663959:
        z += -21.80014 * Q.tau5 + 0.7987484
    if Q.sdb_jp_top3 < 813.043:
        z += 0.0001581983 * Q.sdb_jp_top3 - 0.128622
    if Q.psi_0p1 < 0.624781:
        z += -0.2551937 * Q.psi_0p1 + 0.1594401
    if Q.N2 >= 0.4137858:
        z += -4.275965 * Q.N2 + 1.769333
    if Q.ak02_dr23 >= 0.3194837:
        z += -0.3061654 * Q.ak02_dr23 + 0.09781485
    if Q.sjf_4_1_z_d3 >= 0.181327:
        z += 0.9372559 * Q.sjf_4_1_z_d3 - 0.1699498
    if Q.n_dr_0p1_0p2 < 17.0:
        z += -0.004782866 * Q.n_dr_0p1_0p2 + 0.08130873
    if Q.mass_charged >= 113.5019:
        z += 0.002377679 * Q.mass_charged - 0.2698711
    if Q.sjf_2_1_max3d < 3.32421:
        z += 0.08086818 * Q.sjf_2_1_max3d - 0.1027884
    if 3.32421 <= Q.sjf_2_1_max3d < 55.67805:
        z += -0.003171389 * Q.sjf_2_1_max3d + 0.1765767
    if Q.mass_neutral >= 59.66563:
        z += 0.001408484 * Q.mass_neutral - 0.08403808
    if Q.lepsj_3_dr >= 0.09790963:
        z += 0.4231986 * Q.lepsj_3_dr - 0.04143522
    if Q.jd_n_d3_pt1 < 4.0:
        z += -0.04796721 * Q.jd_n_d3_pt1 + 0.1918688
    if Q.mass_top40 < 126.8853:
        z += 0.004345168 * Q.mass_top40 - 0.5513378
    if Q.mass_top30 < 146.3096:
        z += -0.002547743 * Q.mass_top30 + 0.3727592
    if Q.sip_3d_3 < 4.606241:
        z += 0.1402231 * Q.sip_3d_3 - 0.6459013
    if Q.e3_b2 < 0.0002536827:
        z += -241.4164 * Q.e3_b2 - 0.1862983
    if 0.0002536827 <= Q.e3_b2 < 0.0007909605:
        z += 460.7328 * Q.e3_b2 - 0.3644214
    if Q.e3_b05 >= 0.008870191:
        z += -16.19638 * Q.e3_b05 + 0.143665
    if Q.max_abs_dz < 17.21875:
        z += 0.007286085 * Q.max_abs_dz - 0.1254573
    if Q.sdb_0_z < 0.05934469:
        z += 2.945239 * Q.sdb_0_z - 0.1747843
    if Q.lnptrel_2 >= -2.51737:
        z += 0.1273716 * Q.lnptrel_2 + 0.3206415
    if Q.sjf_3_3_max3d < 1.348343:
        z += -0.08314563 * Q.sjf_3_3_max3d + 0.1121088
    if Q.sjq_3_2_nch < 3.0:
        z += -0.06546487 * Q.sjq_3_2_nch + 0.1963946
    if Q.sj3_pairmin_over_m < 0.2072106:
        z += 1.326927 * Q.sj3_pairmin_over_m - 0.2749533
    if Q.pz_lnd0 < 0.1032185:
        z += -1.125992 * Q.pz_lnd0 + 0.1162231
    if Q.sjf_3_1_n_d3 < 1.0:
        z += -0.2746715 * Q.sjf_3_1_n_d3 + 0.2746715
    if Q.lep_z < 0.221436 and Q.N2 < 0.3829918:
        z += 20.31276 * (0.221436 - Q.lep_z) * (0.3829918 - Q.N2)
    if Q.lep_z < 0.221436 and Q.sdb_2_z > 0.2509165:
        z += -6.60113 * (0.221436 - Q.lep_z) * (Q.sdb_2_z - 0.2509165)
    if Q.lepsj_3_n_d3 < 3.0 and Q.lep_dr < 0.08178299:
        z += -0.939992 * (3.0 - Q.lepsj_3_n_d3) * (0.08178299 - Q.lep_dr)
    if Q.lep_z < 0.221436 and Q.z_displaced5 < 0.08090366:
        z += -47.75638 * (0.221436 - Q.lep_z) * (0.08090366 - Q.z_displaced5)
    if Q.lep_z < 0.221436 and Q.lne_4 > 3.751708:
        z += 0.8114126 * (0.221436 - Q.lep_z) * (Q.lne_4 - 3.751708)
    if Q.mass_displaced3 < 9.090532 and Q.sjq_2_2_nch > 4.0:
        z += -0.003409466 * (9.090532 - Q.mass_displaced3) * (Q.sjq_2_2_nch - 4.0)
    if Q.z_displaced5 < 0.08090366 and Q.z_neutral_had < 0.1912751:
        z += 4.801929 * (0.08090366 - Q.z_displaced5) * (0.1912751 - Q.z_neutral_had)
    if Q.lepsj_3_n_d3 < 3.0 and Q.lepsj_3_dr > 0.0:
        z += -0.3656453 * (3.0 - Q.lepsj_3_n_d3) * (Q.lepsj_3_dr - 0.0)
    if Q.sv_1_sd0_sum < 51.56287 and Q.sjf_3_3_z_d3 > 0.006421567:
        z += -0.0272159 * (51.56287 - Q.sv_1_sd0_sum) * (Q.sjf_3_3_z_d3 - 0.006421567)
    if Q.sv_1_sd0_sum < 51.56287 and Q.dc_1_n_lep > 0.0:
        z += -0.0021161 * (51.56287 - Q.sv_1_sd0_sum) * (Q.dc_1_n_lep - 0.0)
    if Q.sjq_2_1_nch < 16.0 and Q.sum_e < 1651.766:
        z += -2.567912e-05 * (16.0 - Q.sjq_2_1_nch) * (1651.766 - Q.sum_e)
    if Q.n_real_top50 > 26.0 and Q.n_lund_kt_above_5 < 3.0:
        z += -0.006146039 * (Q.n_real_top50 - 26.0) * (3.0 - Q.n_lund_kt_above_5)
    if Q.lep_z < 0.221436 and Q.jet_charge_k05 > -0.2841306:
        z += -0.6911765 * (0.221436 - Q.lep_z) * (Q.jet_charge_k05 - -0.2841306)
    return z


def neuron_19(Q):
    z = -6.257597e-07
    return z


def neuron_20(Q):
    z = -8.103507e-08
    return z


def neuron_21(Q):
    z = -0.001683754
    return z


def neuron_22(Q):
    z = -9.132899e-08
    return z


def neuron_23(Q):
    z = -4.617664e-08
    return z


def neuron_24(Q):
    z = 0.2886766
    if Q.lep_z < 0.004135872:
        z += 63.90847 * Q.lep_z - 1.603596
    if 0.004135872 <= Q.lep_z < 0.3396572:
        z += 3.991635 * Q.lep_z - 1.355788
    if Q.mass_top30 < 74.51927:
        z += 0.01023602 * Q.mass_top30 - 0.5128616
    if 74.51927 <= Q.mass_top30 < 134.2224:
        z += -0.00418603 * Q.mass_top30 + 0.561859
    if Q.mass < 90.08945:
        z += 0.01108636 * Q.mass - 1.64052
    if 90.08945 <= Q.mass < 110.2019:
        z += -0.003169766 * Q.mass - 0.3561937
    if 110.2019 <= Q.mass < 117.4867:
        z += 0.01776814 * Q.mass - 2.663591
    if 117.4867 <= Q.mass < 164.4374:
        z += 0.01226968 * Q.mass - 2.017594
    z += 0.1766693 * Q.pair_max_lnkt
    if Q.z_displaced3 < 0.1061578:
        z += 3.018184 * Q.z_displaced3 - 0.3204037
    if Q.e3_b2 < 7.452974e-05:
        z += 3059.691 * Q.e3_b2 - 0.6548483
    if 7.452974e-05 <= Q.e3_b2 < 0.0004126585:
        z += 1262.271 * Q.e3_b2 - 0.520887
    if Q.mres_sd_mass_b2z01 < 55.13212:
        z += -0.005275074 * Q.mres_sd_mass_b2z01 + 0.3434548
    if 55.13212 <= Q.mres_sd_mass_b2z01 < 81.19466:
        z += 0.02744229 * Q.mres_sd_mass_b2z01 - 1.460323
    if 81.19466 <= Q.mres_sd_mass_b2z01 < 118.2746:
        z += -0.02070779 * Q.mres_sd_mass_b2z01 + 2.449207
    if Q.n_s3d_above_10 >= 1.0:
        z += -0.02627618 * Q.n_s3d_above_10 + 0.02627618
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.04971822 * Q.lepsj_3_maxsd0 + 0.1199832
    if Q.lepsj_2_dr < 0.103005:
        z += 1.353416 * Q.lepsj_2_dr - 0.1394086
    if Q.mass_displaced3 < 18.80005:
        z += -0.01977458 * Q.mass_displaced3 + 0.3717632
    if Q.n_s3d_above_3 < 1.0:
        z += -0.1084535 * Q.n_s3d_above_3 + 0.1382719
    if 1.0 <= Q.n_s3d_above_3 < 3.0:
        z += -0.01490919 * Q.n_s3d_above_3 + 0.04472758
    if Q.n_pairs_kt_above_1 < 366.0:
        z += -0.0001053624 * Q.n_pairs_kt_above_1 + 0.03856264
    if Q.N2 < 0.2423129:
        z += 3.535371 * Q.N2 - 0.8566661
    if Q.lep_iso < 0.4381892:
        z += -2.789265 * Q.lep_iso + 1.460474
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += -0.2578711 * Q.lep_iso + 0.3512447
    if Q.C2_b2 < 0.01057938:
        z += 29.39962 * Q.C2_b2 - 0.3110298
    if Q.sjq_3_2_k1 < -0.6014774:
        z += -2.548432 * Q.sjq_3_2_k1 - 1.532824
    if Q.sjq_2_2_k1 >= 0.277232:
        z += 0.6983254 * Q.sjq_2_2_k1 - 0.1935981
    if Q.jd_sum_abs_sd0_top5 < 84.11398:
        z += -0.008765041 * Q.jd_sum_abs_sd0_top5 + 0.7372625
    if Q.z_neutral_had < 0.212136:
        z += -1.698141 * Q.z_neutral_had + 0.3602368
    if Q.sip_3d_1 < 7.028704:
        z += -0.01658006 * Q.sip_3d_1 + 0.1165363
    if Q.lam1 < 0.006142967:
        z += 218.9622 * Q.lam1 - 1.345077
    if Q.z_dr_0p2_0p4 < 0.1468781:
        z += 3.161156 * Q.z_dr_0p2_0p4 - 0.4643048
    if Q.n_sd0_above_10 >= 6.0:
        z += 0.02953848 * Q.n_sd0_above_10 - 0.1772309
    if Q.sj3_pair_mass_max < 83.63398:
        z += 0.005354255 * Q.sj3_pair_mass_max - 0.09309888
    if 83.63398 <= Q.sj3_pair_mass_max < 128.6605:
        z += -0.007877544 * Q.sj3_pair_mass_max + 1.013529
    z += 0.4235194 * Q.sjq_2_prod_k1
    if Q.tau21 >= 0.135772:
        z += 0.7071599 * Q.tau21 - 0.09601248
    if Q.tau32 < 0.456416:
        z += 4.21226 * Q.tau32 - 1.922543
    if Q.n_lund < 5.0:
        z += -0.3991144 * Q.n_lund + 1.995572
    if Q.dc_2_jp < 6.161303:
        z += -0.02579136 * Q.dc_2_jp + 0.1589084
    if Q.mres_pruned_mass < 60.18017:
        z += -0.006281801 * Q.mres_pruned_mass + 0.06706534
    if 60.18017 <= Q.mres_pruned_mass < 86.14266:
        z += 0.03337818 * Q.mres_pruned_mass - 2.319679
    if 86.14266 <= Q.mres_pruned_mass < 119.8459:
        z += -0.02242023 * Q.mres_pruned_mass + 2.486945
    if 119.8459 <= Q.mres_pruned_mass < 171.3819:
        z += 0.003881324 * Q.mres_pruned_mass - 0.6651888
    if 0.2293319 <= Q.dc_split2_dr < 0.3591078:
        z += 2.769678 * Q.dc_split2_dr - 0.6351757
    if Q.dc_split2_dr >= 0.3591078:
        z += -3.706675 * Q.dc_split2_dr + 1.690533
    if Q.n_dr_0p4_up < 4.0:
        z += -0.06584966 * Q.n_dr_0p4_up + 0.2633986
    if Q.mass_top15 >= 70.93762:
        z += -0.00453927 * Q.mass_top15 + 0.322005
    if Q.max_abs_d0 < 10.52344:
        z += -0.03636611 * Q.max_abs_d0 + 0.3826965
    if Q.z_charged_had >= 0.5398733:
        z += -0.5889476 * Q.z_charged_had + 0.3179571
    if Q.n_dr_0p2_0p4 < 17.0:
        z += -0.04610763 * Q.n_dr_0p2_0p4 + 0.7838297
    if Q.dr_8 >= 0.3729651:
        z += -0.4517878 * Q.dr_8 + 0.1685011
    if Q.lep_ptrel < 12.15228:
        z += 0.006215281 * Q.lep_ptrel - 0.07552984
    if Q.lep_z < 0.3396572 and Q.mass < 127.2317:
        z += 0.04877386 * (0.3396572 - Q.lep_z) * (127.2317 - Q.mass)
    if Q.lep_z < 0.3396572 and Q.tau32 < 0.456416:
        z += 20.012 * (0.3396572 - Q.lep_z) * (0.456416 - Q.tau32)
    if Q.lep_ptrel < 43.20788 and Q.sdb_2_z > 0.2968888:
        z += 0.02156487 * (43.20788 - Q.lep_ptrel) * (Q.sdb_2_z - 0.2968888)
    if Q.e3_b2 < 0.0004126585 and Q.lne_8 > 2.11682:
        z += 272.5182 * (0.0004126585 - Q.e3_b2) * (Q.lne_8 - 2.11682)
    if Q.z_displaced3 < 0.1061578 and Q.pz_lnd2 < 0.1166862:
        z += -16.48103 * (0.1061578 - Q.z_displaced3) * (0.1166862 - Q.pz_lnd2)
    if Q.z_displaced3 < 0.1061578 and Q.D2_b2 < 15.17086:
        z += 0.2248672 * (0.1061578 - Q.z_displaced3) * (15.17086 - Q.D2_b2)
    if Q.lepsj_3_maxsd0 < 2.413264 and Q.ak02_3_n_lep > 0.0:
        z += 0.07828736 * (2.413264 - Q.lepsj_3_maxsd0) * (Q.ak02_3_n_lep - 0.0)
    if Q.mass < 164.4374 and Q.sv_1_n < 3.0:
        z += -0.001498427 * (164.4374 - Q.mass) * (3.0 - Q.sv_1_n)
    if Q.e3_b2 < 0.0004126585 and Q.sdb_0_n < 5.0:
        z += -144.7918 * (0.0004126585 - Q.e3_b2) * (5.0 - Q.sdb_0_n)
    if Q.mass < 164.4374 and Q.M3_b2 < 0.0309457:
        z += -0.05876463 * (164.4374 - Q.mass) * (0.0309457 - Q.M3_b2)
    if Q.lepsj_2_dr < 0.103005 and Q.jet_e > 926.0771:
        z += 0.001518521 * (0.103005 - Q.lepsj_2_dr) * (Q.jet_e - 926.0771)
    if Q.mass < 164.4374 and Q.n_dr_0p2_0p4 < 24.0:
        z += 4.090031e-05 * (164.4374 - Q.mass) * (24.0 - Q.n_dr_0p2_0p4)
    if Q.jd_sum_abs_sd0_top5 < 84.11398 and Q.jd_3d_5 < 6.771002:
        z += -0.001954032 * (84.11398 - Q.jd_sum_abs_sd0_top5) * (6.771002 - Q.jd_3d_5)
    if Q.mass_displaced3 < 18.80005 and Q.jd_3d_5 < 4.004982:
        z += 0.01060073 * (18.80005 - Q.mass_displaced3) * (4.004982 - Q.jd_3d_5)
    if Q.lep_ptrel < 43.20788 and Q.ak02_3_n_disp3 > 1.0:
        z += -0.002235418 * (43.20788 - Q.lep_ptrel) * (Q.ak02_3_n_disp3 - 1.0)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lep_iso < 0.4381892:
        z += -0.08930747 * (83.63398 - Q.sj3_pair_mass_max) * (0.4381892 - Q.lep_iso)
    if Q.lep_z < 0.3396572 and Q.tau21 > 0.1757731:
        z += -2.970069 * (0.3396572 - Q.lep_z) * (Q.tau21 - 0.1757731)
    if Q.lep_z < 0.3396572 and Q.n_neutral_had < 7.0:
        z += -0.1590462 * (0.3396572 - Q.lep_z) * (7.0 - Q.n_neutral_had)
    if Q.mass_displaced3 < 18.80005 and Q.z_photon < 0.3090294:
        z += 0.03571356 * (18.80005 - Q.mass_displaced3) * (0.3090294 - Q.z_photon)
    if Q.z_displaced3 < 0.1061578 and Q.mass_2photon > 3.013:
        z += 0.05909478 * (0.1061578 - Q.z_displaced3) * (Q.mass_2photon - 3.013)
    if Q.e3_b2 < 0.0004126585 and Q.kt2_min12_n_disp3 > 1.0:
        z += 306.4243 * (0.0004126585 - Q.e3_b2) * (Q.kt2_min12_n_disp3 - 1.0)
    if Q.lep_z < 0.004135872 and Q.sdb_4_z < 0.01435877:
        z += 3311.034 * (0.004135872 - Q.lep_z) * (0.01435877 - Q.sdb_4_z)
    if Q.lep_z < 0.3396572 and Q.e4 > 4.06096e-07:
        z += 26699.28 * (0.3396572 - Q.lep_z) * (Q.e4 - 4.06096e-07)
    if Q.lep_z < 0.004135872 and Q.n_pairs_kt_above_10 > 4.0:
        z += -1.977076 * (0.004135872 - Q.lep_z) * (Q.n_pairs_kt_above_10 - 4.0)
    if Q.sjq_3_2_k1 < -0.6014774 and Q.lep_iso < 0.4381892:
        z += -4.360889 * (-0.6014774 - Q.sjq_3_2_k1) * (0.4381892 - Q.lep_iso)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lep_iso < 14.38858:
        z += 0.00150718 * (83.63398 - Q.sj3_pair_mass_max) * (14.38858 - Q.lep_iso)
    if Q.lepsj_2_dr < 0.103005 and Q.lep_iso < 14.38858:
        z += -0.8338797 * (0.103005 - Q.lepsj_2_dr) * (14.38858 - Q.lep_iso)
    if Q.sjq_2_2_k1 > 0.277232 and Q.lep_iso < 2.907433:
        z += -0.09448999 * (Q.sjq_2_2_k1 - 0.277232) * (2.907433 - Q.lep_iso)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lepsj_2_dr < 0.1822601:
        z += 0.09688585 * (83.63398 - Q.sj3_pair_mass_max) * (0.1822601 - Q.lepsj_2_dr)
    if Q.lep_z < 0.004135872 and Q.lep_iso < 2.907433:
        z += -108.8276 * (0.004135872 - Q.lep_z) * (2.907433 - Q.lep_iso)
    if Q.lep_z < 0.3396572 and Q.lepsj_3_maxsd0 < 0.8036986:
        z += 2.245642 * (0.3396572 - Q.lep_z) * (0.8036986 - Q.lepsj_3_maxsd0)
    if Q.mass_top15 > 70.93762 and Q.dc_1_n_lep > 0.0:
        z += -0.001354684 * (Q.mass_top15 - 70.93762) * (Q.dc_1_n_lep - 0.0)
    if Q.lep_z < 0.3396572 and Q.sj3_dr12 < 0.3683248:
        z += -1.700914 * (0.3396572 - Q.lep_z) * (0.3683248 - Q.sj3_dr12)
    if Q.mass_displaced3 < 18.80005 and Q.sip_3d_2 < 2325.187:
        z += -1.043987e-05 * (18.80005 - Q.mass_displaced3) * (2325.187 - Q.sip_3d_2)
    if Q.max_abs_d0 < 10.52344 and Q.jd_3d_5 < 4.004982:
        z += -0.005651366 * (10.52344 - Q.max_abs_d0) * (4.004982 - Q.jd_3d_5)
    if Q.mass_top15 > 70.93762 and Q.sv_2_z > 0.00726873:
        z += -0.1993446 * (Q.mass_top15 - 70.93762) * (Q.sv_2_z - 0.00726873)
    if Q.lep_iso < 0.4381892 and Q.sdb_4_n > 0.0:
        z += 0.2407793 * (0.4381892 - Q.lep_iso) * (Q.sdb_4_n - 0.0)
    if Q.lep_iso < 0.4381892 and Q.lepsj_2_dr < 0.1822601:
        z += 9.781433 * (0.4381892 - Q.lep_iso) * (0.1822601 - Q.lepsj_2_dr)
    return z


def neuron_25(Q):
    z = 0.02938367
    return z


def neuron_26(Q):
    z = -3.074858e-06
    return z


def neuron_27(Q):
    z = -7.258977e-06
    return z


def neuron_28(Q):
    z = 1.053252e-05
    return z


def neuron_29(Q):
    z = 8.611535e-06
    return z


def neuron_30(Q):
    z = -9.494941e-06
    return z


def neuron_31(Q):
    z = 3.098268e-06
    return z


def neuron_32(Q):
    z = 1.534171e-05
    return z


def neuron_33(Q):
    z = 3.042679e-06
    return z


def neuron_34(Q):
    z = 1.49241e-05
    return z


def neuron_35(Q):
    z = -7.29049e-06
    return z


def neuron_36(Q):
    z = -4.849001e-06
    return z


def neuron_37(Q):
    z = -4.057251e-06
    return z


def neuron_38(Q):
    z = -1.063661e-05
    return z


def neuron_39(Q):
    z = 8.891538e-06
    return z


def neuron_40(Q):
    z = 6.493206e-07
    return z


def neuron_41(Q):
    z = -3.28459e-07
    return z


def neuron_42(Q):
    z = 7.958611e-06
    return z


def neuron_43(Q):
    z = -1.000759e-05
    return z


def neuron_44(Q):
    z = -3.355799e-06
    return z


def neuron_45(Q):
    z = 7.000797e-06
    return z


def neuron_46(Q):
    z = 4.371953e-07
    return z


def neuron_47(Q):
    z = -7.391782e-08
    return z


def neuron_48(Q):
    z = -8.966534e-06
    return z


def neuron_49(Q):
    z = -1.675087e-06
    return z


def neuron_50(Q):
    z = 4.873647e-06
    return z


def neuron_51(Q):
    z = -2.006738e-06
    return z


def neuron_52(Q):
    z = -0.9750853
    if Q.lep_z < 0.221436:
        z += 1.266432 * Q.lep_z - 0.2142097
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -0.5601692 * Q.lep_z + 0.1902655
    if Q.z_displaced3 < 0.1342762:
        z += -5.946672 * Q.z_displaced3 + 0.7984965
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.155154 * Q.lepsj_3_maxsd0 + 0.3744275
    if Q.mass_top40 < 70.88236:
        z += -0.002940563 * Q.mass_top40 + 0.2084341
    if Q.mass_top50 < 79.27954:
        z += -0.02993392 * Q.mass_top50 + 2.611452
    if 79.27954 <= Q.mass_top50 < 115.614:
        z += -0.006558627 * Q.mass_top50 + 0.7582691
    if Q.mres_sd_mass_b0z005 < 81.62059:
        z += -0.006642582 * Q.mres_sd_mass_b0z005 + 1.533993
    if 81.62059 <= Q.mres_sd_mass_b0z005 < 131.0776:
        z += -0.004210144 * Q.mres_sd_mass_b0z005 + 1.335456
    if 131.0776 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += -0.02056151 * Q.mres_sd_mass_b0z005 + 3.478753
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += 0.002432438 * Q.mres_sd_mass_b0z005 - 0.1985371
    if Q.mass >= 182.8592:
        z += -0.01534405 * Q.mass + 2.805801
    if Q.lne_16 >= 0.4468807:
        z += -0.1410416 * Q.lne_16 + 0.06302878
    if Q.lepsj_2_dr < 0.103005:
        z += 5.758491 * Q.lepsj_2_dr - 0.5931532
    if Q.sv_1_sd0_sum < 125.2765:
        z += 0.002006927 * Q.sv_1_sd0_sum - 0.2514209
    if Q.sjq_2_2_k1 < -0.6337755:
        z += -2.060922 * Q.sjq_2_2_k1 - 1.306162
    if Q.sjq_2_2_k1 >= 0.6477929:
        z += 1.602163 * Q.sjq_2_2_k1 - 1.03787
    if Q.pz_lnd0 < 0.1122946:
        z += -2.088552 * Q.pz_lnd0 + 0.234533
    if Q.n_s3d_above_3 < 3.0:
        z += -0.1509801 * Q.n_s3d_above_3 + 0.4529403
    if Q.lep_dr < 0.3014662:
        z += -2.858658 * Q.lep_dr + 0.8617888
    if Q.lep_dr >= 0.4191372:
        z += -0.9749092 * Q.lep_dr + 0.4086208
    if Q.D2_b05 < 1.45443:
        z += 0.3976063 * Q.D2_b05 - 0.5782907
    if Q.n_lund < 7.0:
        z += -0.1189322 * Q.n_lund + 0.8325252
    if Q.n_lepton < 1.0:
        z += 0.6372108 * Q.n_lepton - 0.6372108
    if Q.lep_ptrel < 12.15228:
        z += 0.0389837 * Q.lep_ptrel - 0.4737408
    if Q.lep_ptrel >= 18.7678:
        z += 0.01097668 * Q.lep_ptrel - 0.2060082
    if Q.sj2_mass2 < 21.7295:
        z += 0.01312692 * Q.sj2_mass2 - 0.2852415
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.003076395 * Q.n_pairs_kt_above_1 + 0.2461116
    if Q.dc_1_n_lep < 1.0:
        z += -0.3287535 * Q.dc_1_n_lep + 0.3287535
    if Q.jd_sum_abs_sd0_top5 < 84.11398:
        z += 0.004104844 * Q.jd_sum_abs_sd0_top5 - 0.3452747
    if Q.sj3_pair_mass_min >= 80.02563:
        z += 0.02373453 * Q.sj3_pair_mass_min - 1.899371
    if Q.lund3_lndelta >= -2.817283:
        z += -0.06034734 * Q.lund3_lndelta - 0.1700155
    if Q.jet_charge_k03 >= -0.2409073:
        z += -0.1076317 * Q.jet_charge_k03 - 0.02592926
    if Q.z_charged_had < 0.1891627:
        z += 1.572875 * Q.z_charged_had - 0.2975292
    if Q.ak02_2_z < 0.2837384:
        z += -2.577798 * Q.ak02_2_z + 0.7314203
    if Q.sjf_2_2_max3d < 3.029824:
        z += -0.08434448 * Q.sjf_2_2_max3d + 0.2555489
    if Q.n_dr_0p2_0p4 < 21.0:
        z += 0.02034225 * Q.n_dr_0p2_0p4 - 0.4271873
    if Q.sj2_dr < 0.3740528:
        z += -0.8498093 * Q.sj2_dr + 0.3178735
    if Q.e3_b2 < 7.104488e-06:
        z += -13755.45 * Q.e3_b2 + 0.5898554
    if 7.104488e-06 <= Q.e3_b2 < 0.0004126585:
        z += -1213.476 * Q.e3_b2 + 0.5007511
    if Q.sjq_3_3_k1 >= 0.6535817:
        z += 0.4904892 * Q.sjq_3_3_k1 - 0.3205747
    if Q.mass_charged < 65.0404:
        z += 0.00106681 * Q.mass_charged - 0.06938577
    if Q.mass_2charged < 6.767937:
        z += 0.01088069 * Q.mass_2charged - 0.07363979
    if Q.mass_2charged >= 14.28253:
        z += -0.008806191 * Q.mass_2charged + 0.1257747
    if Q.N3_b05 >= 0.7591346:
        z += -0.09999606 * Q.N3_b05 + 0.07591047
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.006698198 * Q.n_pairs_kt_above_3 + 0.1875495
    if Q.mass_top15 >= 133.1648:
        z += 0.002831598 * Q.mass_top15 - 0.3770692
    if Q.sdb_2_n >= 9.0:
        z += -0.04600928 * Q.sdb_2_n + 0.4140835
    if Q.sj3_mass1 >= 36.36236:
        z += -0.01178625 * Q.sj3_mass1 + 0.4285758
    if Q.D2 < 1.421532:
        z += 0.1569278 * Q.D2 - 0.2230779
    if Q.tau21_b2 < 0.3565533:
        z += -3.086115 * Q.tau21_b2 + 1.100365
    if Q.ak02_1_z >= 0.6342743:
        z += -1.190957 * Q.ak02_1_z + 0.7553937
    if Q.tau4 < 0.01143413:
        z += -74.98849 * Q.tau4 + 0.8574282
    if Q.n_dr_0p4_up < 10.0:
        z += 0.02519713 * Q.n_dr_0p4_up - 0.2519713
    if Q.ak02_3_z >= 0.1014193:
        z += -0.8441795 * Q.ak02_3_z + 0.08561608
    if Q.lund_max_lndelta >= -0.615738:
        z += 0.8669336 * Q.lund_max_lndelta + 0.5338039
    if Q.N2_b2 >= 0.08291719:
        z += 1.285749 * Q.N2_b2 - 0.1066107
    if Q.z_displaced3 < 0.1342762 and Q.N2 > 0.1824346:
        z += 8.9156 * (0.1342762 - Q.z_displaced3) * (Q.N2 - 0.1824346)
    if Q.lep_z < 0.3396572 and Q.n_charged_pt_above_1 > 11.0:
        z += 0.1190986 * (0.3396572 - Q.lep_z) * (Q.n_charged_pt_above_1 - 11.0)
    if Q.lep_z < 0.3396572 and Q.sj3_pair_mass_min > 48.40547:
        z += -0.01968867 * (0.3396572 - Q.lep_z) * (Q.sj3_pair_mass_min - 48.40547)
    if Q.mres_sd_mass_b0z005 < 159.9242 and Q.tau21_b2 < 0.3565533:
        z += -0.03643271 * (159.9242 - Q.mres_sd_mass_b0z005) * (0.3565533 - Q.tau21_b2)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.tau32 < 0.7544983:
        z += -0.005853311 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.7544983 - Q.tau32)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.sjf_2_2_z_d3 < 0.03385157:
        z += 0.04810924 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.03385157 - Q.sjf_2_2_z_d3)
    if Q.mres_sd_mass_b0z005 < 159.9242 and Q.mass_displaced3 < 1.251841:
        z += -0.0003014821 * (159.9242 - Q.mres_sd_mass_b0z005) * (1.251841 - Q.mass_displaced3)
    if Q.mres_sd_mass_b0z005 < 159.9242 and Q.mass_displaced3 > 3.208089:
        z += -0.0002851612 * (159.9242 - Q.mres_sd_mass_b0z005) * (Q.mass_displaced3 - 3.208089)
    if Q.z_neutral_had < 0.3788785 and Q.ecf_g41 > 6.216954e-05:
        z += 2826.44 * (0.3788785 - Q.z_neutral_had) * (Q.ecf_g41 - 6.216954e-05)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.ak02_3_z < 0.07650476:
        z += 0.1124453 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.07650476 - Q.ak02_3_z)
    if Q.lepsj_2_dr < 0.103005 and Q.C3_b05 < 0.2250047:
        z += 9.480889 * (0.103005 - Q.lepsj_2_dr) * (0.2250047 - Q.C3_b05)
    if Q.mres_sd_mass_b0z005 < 131.0776 and Q.sip_3d_3 < 23.30856:
        z += -4.521734e-05 * (131.0776 - Q.mres_sd_mass_b0z005) * (23.30856 - Q.sip_3d_3)
    if Q.lep_dr < 0.3014662 and Q.sdb_2_z > 0.1530389:
        z += 1.542788 * (0.3014662 - Q.lep_dr) * (Q.sdb_2_z - 0.1530389)
    if Q.z_displaced3 < 0.1342762 and Q.sip_3d_3 < 577.991:
        z += -0.00790691 * (0.1342762 - Q.z_displaced3) * (577.991 - Q.sip_3d_3)
    if Q.lund3_lndelta > -2.817283 and Q.sv_1_dr < 0.3689526:
        z += 0.2347025 * (Q.lund3_lndelta - -2.817283) * (0.3689526 - Q.sv_1_dr)
    if Q.sj2_mass2 < 21.7295 and Q.nca_sj4_pairmax_over_mass < 0.7817308:
        z += 0.04457563 * (21.7295 - Q.sj2_mass2) * (0.7817308 - Q.nca_sj4_pairmax_over_mass)
    if Q.z_neutral_had < 0.3788785 and Q.pz_lnkt1 > 0.03633353:
        z += -8.661086 * (0.3788785 - Q.z_neutral_had) * (Q.pz_lnkt1 - 0.03633353)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_2_prod_k05 > -0.5057096:
        z += -0.009465822 * (21.0 - Q.n_dr_0p2_0p4) * (Q.sjq_2_prod_k05 - -0.5057096)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_3_sumabs_k1 > 0.3376898:
        z += 0.0157441 * (21.0 - Q.n_dr_0p2_0p4) * (Q.sjq_3_sumabs_k1 - 0.3376898)
    if Q.z_displaced3 < 0.1342762 and Q.n_lund_kt_above_1 > 3.0:
        z += 0.2425992 * (0.1342762 - Q.z_displaced3) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.lepsj_2_dr < 0.103005 and Q.z_photon > 0.3318968:
        z += 7.345509 * (0.103005 - Q.lepsj_2_dr) * (Q.z_photon - 0.3318968)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_3_prod_k1 < 0.1145956:
        z += 0.02992845 * (21.0 - Q.n_dr_0p2_0p4) * (0.1145956 - Q.sjq_3_prod_k1)
    if Q.z_displaced3 < 0.1342762 and Q.jd_3d_6 < 5.367501:
        z += 1.248551 * (0.1342762 - Q.z_displaced3) * (5.367501 - Q.jd_3d_6)
    if Q.n_lund < 7.0 and Q.jd_3d_6 < 2.591685:
        z += -0.02554575 * (7.0 - Q.n_lund) * (2.591685 - Q.jd_3d_6)
    if Q.D3_b2 < 0.001184159 and Q.jd_3d_6 < 5.367501:
        z += -48.78408 * (0.001184159 - Q.D3_b2) * (5.367501 - Q.jd_3d_6)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.jd_3d_6 < 3.151933:
        z += 0.000895473 * (Q.mres_sd_mass_b0z005 - 81.62059) * (3.151933 - Q.jd_3d_6)
    if Q.mass_charged < 65.0404 and Q.isphoton_19 < 1.0:
        z += 0.001394635 * (65.0404 - Q.mass_charged) * (1.0 - Q.isphoton_19)
    if Q.sdb_2_n > 9.0 and Q.D2_b2 < 2.01745:
        z += 0.03808764 * (Q.sdb_2_n - 9.0) * (2.01745 - Q.D2_b2)
    if Q.z_displaced3 < 0.1342762 and Q.jd_3d_6 < 30.57088:
        z += -0.2134294 * (0.1342762 - Q.z_displaced3) * (30.57088 - Q.jd_3d_6)
    if Q.mres_sd_mass_b0z005 < 131.0776 and Q.D2_b2 < 1.204663:
        z += -0.00650845 * (131.0776 - Q.mres_sd_mass_b0z005) * (1.204663 - Q.D2_b2)
    return z


def neuron_53(Q):
    z = -2.119185e-06
    return z


def neuron_54(Q):
    z = 1.153068e-05
    return z


def neuron_55(Q):
    z = -2.157538e-05
    return z


def neuron_56(Q):
    z = -3.752157e-07
    return z


def neuron_57(Q):
    z = -1.293959e-06
    return z


def neuron_58(Q):
    z = -3.97713e-06
    return z


def neuron_59(Q):
    z = 6.502637e-06
    return z


def neuron_60(Q):
    z = -3.67915e-06
    return z


def neuron_61(Q):
    z = 5.435962e-06
    return z


def neuron_62(Q):
    z = 1.597762e-06
    return z


def neuron_63(Q):
    z = -7.32332e-06
    return z


def neuron_64(Q):
    z = 6.667274e-07
    return z


def neuron_65(Q):
    z = 5.969107e-06
    return z


def neuron_66(Q):
    z = 2.869698e-06
    return z


def neuron_67(Q):
    z = -4.654168e-06
    return z


def neuron_68(Q):
    z = 4.651738
    if Q.tau32 < 0.7981752:
        z += -0.6288472 * Q.tau32 + 0.5019303
    if 6.983043 <= Q.lep_ptrel < 18.7678:
        z += -0.02092482 * Q.lep_ptrel + 0.1461189
    if 18.7678 <= Q.lep_ptrel < 43.20788:
        z += 0.03004373 * Q.lep_ptrel - 0.8104487
    if Q.lep_ptrel >= 43.20788:
        z += 0.000299314 * Q.lep_ptrel + 0.4747446
    if Q.pair_mean_lnm2 < 3.38061:
        z += 0.02885297 * Q.pair_mean_lnm2 - 0.09754067
    if Q.n_s3d_above_3 >= 5.0:
        z += -0.01694694 * Q.n_s3d_above_3 + 0.08473469
    if Q.pz_lnd2 < 0.1339824:
        z += 2.228973 * Q.pz_lnd2 - 0.2986431
    if Q.e3_b2 < 0.0004126585:
        z += 1176.579 * Q.e3_b2 - 0.4855255
    if Q.sj3_pair_mass_max < 56.73313:
        z += 0.02503063 * Q.sj3_pair_mass_max - 1.420066
    if 83.63398 <= Q.sj3_pair_mass_max < 128.6605:
        z += -0.008546907 * Q.sj3_pair_mass_max + 0.7148119
    if Q.sj3_pair_mass_max >= 128.6605:
        z += 0.01122135 * Q.sj3_pair_mass_max - 1.828583
    if Q.mass < 90.08945:
        z += 0.001093481 * Q.mass - 1.308166
    if 90.08945 <= Q.mass < 110.2019:
        z += -0.005673053 * Q.mass - 0.6985727
    if 110.2019 <= Q.mass < 117.4867:
        z += 0.02157196 * Q.mass - 3.701025
    if 117.4867 <= Q.mass < 164.4374:
        z += 0.02484745 * Q.mass - 4.085851
    if Q.sj4_pair_mass_max >= 69.83554:
        z += -0.002580015 * Q.sj4_pair_mass_max + 0.1801768
    if Q.z_charged_had >= 0.265564:
        z += 0.6663395 * Q.z_charged_had - 0.1769558
    if Q.M2_b2 < 0.05966366:
        z += 7.121939 * Q.M2_b2 - 0.424921
    if Q.mres_sd_mass_b2z01 < 55.13212:
        z += -0.005482516 * Q.mres_sd_mass_b2z01 - 0.9130889
    if 55.13212 <= Q.mres_sd_mass_b2z01 < 76.1529:
        z += 0.04443379 * Q.mres_sd_mass_b2z01 - 3.665081
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 118.2746:
        z += -0.01969371 * Q.mres_sd_mass_b2z01 + 1.218414
    if 118.2746 <= Q.mres_sd_mass_b2z01 < 130.0968:
        z += 0.02184733 * Q.mres_sd_mass_b2z01 - 3.694838
    if 130.0968 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.0303351 * Q.mres_sd_mass_b2z01 - 4.799069
    z += -0.06229552 * Q.n_particles
    if Q.dc_split2_dr >= 0.3591078:
        z += -2.291952 * Q.dc_split2_dr + 0.8230577
    if Q.lep_iso < 0.4381892:
        z += 1.380097 * Q.lep_iso - 0.6047435
    if Q.lep_dr < 0.04607888:
        z += -7.018616 * Q.lep_dr + 0.32341
    if Q.sj3_pair_mass_min < 80.02563:
        z += 0.008994703 * Q.sj3_pair_mass_min - 0.7198068
    if Q.sdb_2_n < 7.0:
        z += -0.03235598 * Q.sdb_2_n + 0.2264919
    if Q.sdb_2_n >= 12.0:
        z += -0.02600915 * Q.sdb_2_n + 0.3121098
    if Q.M3 < 0.03259227:
        z += -13.44842 * Q.M3 + 0.4383145
    if Q.mass_top20 >= 130.7093:
        z += -0.01338726 * Q.mass_top20 + 1.749839
    if Q.jet_abs_eta >= 0.5325716:
        z += 0.357735 * Q.jet_abs_eta - 0.1905195
    z += 0.03759104 * Q.n_photon
    if Q.sjq_2_2_nch < 3.0:
        z += 0.1450129 * Q.sjq_2_2_nch - 0.4350386
    if Q.sj2_dr < 0.2403736:
        z += 0.8441331 * Q.sj2_dr - 0.2029073
    if Q.tau5 < 0.05613495:
        z += -16.17068 * Q.tau5 + 0.9077406
    if Q.ktd_ln_d34 >= -9.359695:
        z += 0.1630047 * Q.ktd_ln_d34 + 1.525674
    if Q.D2_b2 < 1.774856:
        z += 0.1621149 * Q.D2_b2 - 0.2877306
    if Q.nca_sj4_pair_mass_2nd >= 72.72359:
        z += -0.01715534 * Q.nca_sj4_pair_mass_2nd + 1.247598
    if Q.ak02_dr12 >= 0.3882673:
        z += -0.3366639 * Q.ak02_dr12 + 0.1307156
    if Q.lepsj_2_maxsd0 < 1163.885:
        z += 5.453743e-05 * Q.lepsj_2_maxsd0 - 0.06347529
    if Q.sjf_3_1_n_d3 >= 6.0:
        z += 0.03293068 * Q.sjf_3_1_n_d3 - 0.1975841
    if 47.97531 <= Q.mres_sd_mass_b0z02 < 86.60355:
        z += 0.003902612 * Q.mres_sd_mass_b0z02 - 0.187229
    if 86.60355 <= Q.mres_sd_mass_b0z02 < 134.2023:
        z += -0.006334098 * Q.mres_sd_mass_b0z02 + 0.6993064
    if Q.mres_sd_mass_b0z02 >= 134.2023:
        z += 0.00294294 * Q.mres_sd_mass_b0z02 - 0.5456936
    if 86.14266 <= Q.mres_pruned_mass < 124.3145:
        z += -0.009235669 * Q.mres_pruned_mass + 0.7955851
    if 124.3145 <= Q.mres_pruned_mass < 171.3819:
        z += 0.005339503 * Q.mres_pruned_mass - 1.016321
    if Q.mres_pruned_mass >= 171.3819:
        z += 0.001469493 * Q.mres_pruned_mass - 0.3530709
    if Q.pz_lnd3 >= 0.01024929:
        z += 0.4608479 * Q.pz_lnd3 - 0.004723366
    if Q.n_dr_0p2_0p4 >= 7.0:
        z += -0.003474481 * Q.n_dr_0p2_0p4 + 0.02432137
    if Q.mres_sd_rg_b2z01 >= 0.457968:
        z += -1.281126 * Q.mres_sd_rg_b2z01 + 0.5867148
    if Q.n_sdz_above_5 < 3.0:
        z += -0.05983895 * Q.n_sdz_above_5 + 0.1795169
    if Q.sum_z_dr2_top2 < 0.005938474:
        z += -23.25847 * Q.sum_z_dr2_top2 + 0.1381198
    if Q.mass_top10 < 103.4976:
        z += 0.001218547 * Q.mass_top10 - 0.1261168
    if Q.sdb_2_z < 0.1530389:
        z += 0.4141321 * Q.sdb_2_z - 0.06337834
    if Q.jd_3d_5 < 4.004982:
        z += 0.07034004 * Q.jd_3d_5 - 0.2817106
    if Q.tau32 < 0.7981752 and Q.sj3_dr_min < 0.3190414:
        z += 2.227521 * (0.7981752 - Q.tau32) * (0.3190414 - Q.sj3_dr_min)
    if Q.pair_mean_lnm2 < 3.38061 and Q.z_top30_slots > 0.8316085:
        z += 0.1482666 * (3.38061 - Q.pair_mean_lnm2) * (Q.z_top30_slots - 0.8316085)
    if Q.e3_b2 < 0.0004126585 and Q.lnptrel_25 > -6.691703:
        z += 313.4556 * (0.0004126585 - Q.e3_b2) * (Q.lnptrel_25 - -6.691703)
    if Q.e3_b2 < 0.0004126585 and Q.ak02_dr23 > 0.3194837:
        z += 1254.896 * (0.0004126585 - Q.e3_b2) * (Q.ak02_dr23 - 0.3194837)
    if Q.sj4_pair_mass_max > 69.83554 and Q.nca_sj4_pair2nd_over_mass > 0.4488456:
        z += 0.03139319 * (Q.sj4_pair_mass_max - 69.83554) * (Q.nca_sj4_pair2nd_over_mass - 0.4488456)
    if Q.sj3_pair_mass_max > 128.6605 and Q.jet_e > 551.9263:
        z += 5.493115e-06 * (Q.sj3_pair_mass_max - 128.6605) * (Q.jet_e - 551.9263)
    if Q.mass < 117.4867 and Q.D2_b2 < 1.380731:
        z += 0.007537176 * (117.4867 - Q.mass) * (1.380731 - Q.D2_b2)
    if Q.e3_b2 < 0.0004126585 and Q.sj3_mass1 > 17.31003:
        z += 21.04564 * (0.0004126585 - Q.e3_b2) * (Q.sj3_mass1 - 17.31003)
    if Q.sj4_pair_mass_max > 69.83554 and Q.dc_3_z < 0.2037349:
        z += -0.02883336 * (Q.sj4_pair_mass_max - 69.83554) * (0.2037349 - Q.dc_3_z)
    if Q.n_s3d_above_3 > 5.0 and Q.sip_3d_2 < 226.3008:
        z += -0.0001187008 * (Q.n_s3d_above_3 - 5.0) * (226.3008 - Q.sip_3d_2)
    if Q.lep_ptrel > 6.983043 and Q.C2_b2 < 0.1682645:
        z += 0.04013149 * (Q.lep_ptrel - 6.983043) * (0.1682645 - Q.C2_b2)
    if Q.sj3_pair_mass_min < 80.02563 and Q.mass_displaced5 > 0.0:
        z += 0.000128717 * (80.02563 - Q.sj3_pair_mass_min) * (Q.mass_displaced5 - 0.0)
    if Q.sj3_pair_mass_max > 128.6605 and Q.e4 > 7.184834e-06:
        z += 228.7533 * (Q.sj3_pair_mass_max - 128.6605) * (Q.e4 - 7.184834e-06)
    if Q.pz_lnd2 < 0.1339824 and Q.sj3_dr13 > 0.4691911:
        z += -5.956275 * (0.1339824 - Q.pz_lnd2) * (Q.sj3_dr13 - 0.4691911)
    if Q.mass < 164.4374 and Q.sjf_4_3_z_d3 > 0.0:
        z += -0.01630299 * (164.4374 - Q.mass) * (Q.sjf_4_3_z_d3 - 0.0)
    if Q.tau32 < 0.7981752 and Q.sjq_3_sumabs_k1 > 0.4372817:
        z += 0.3318 * (0.7981752 - Q.tau32) * (Q.sjq_3_sumabs_k1 - 0.4372817)
    if Q.z_displaced3 < 0.1061578 and Q.lepsj_2_maxsd0 < 0.7828545:
        z += -5.175826 * (0.1061578 - Q.z_displaced3) * (0.7828545 - Q.lepsj_2_maxsd0)
    if Q.z_charged_had > 0.265564 and Q.pz_lnkt3 > 0.0:
        z += 3.255288 * (Q.z_charged_had - 0.265564) * (Q.pz_lnkt3 - 0.0)
    if Q.lep_dr < 0.04607888 and Q.lepsj_2_maxsd0 < 0.7828545:
        z += 12.14096 * (0.04607888 - Q.lep_dr) * (0.7828545 - Q.lepsj_2_maxsd0)
    if Q.sj4_pair_mass_max > 69.83554 and Q.sv_1_dr < 0.3689526:
        z += -0.005845376 * (Q.sj4_pair_mass_max - 69.83554) * (0.3689526 - Q.sv_1_dr)
    if Q.mass < 117.4867 and Q.C2_b2 > 0.01891146:
        z += -0.0479084 * (117.4867 - Q.mass) * (Q.C2_b2 - 0.01891146)
    if Q.z_displaced3 < 0.1061578 and Q.dr_29 < 0.02258554:
        z += 66.74136 * (0.1061578 - Q.z_displaced3) * (0.02258554 - Q.dr_29)
    if Q.e3_b2 < 0.0004126585 and Q.n_lund_kt_above_5 > 2.0:
        z += 227.4772 * (0.0004126585 - Q.e3_b2) * (Q.n_lund_kt_above_5 - 2.0)
    if Q.z_displaced3 < 0.1061578 and Q.n_lund > 10.0:
        z += -0.537455 * (0.1061578 - Q.z_displaced3) * (Q.n_lund - 10.0)
    if Q.M3 < 0.03259227 and Q.jd_3d_5 < 4.004982:
        z += -3.080916 * (0.03259227 - Q.M3) * (4.004982 - Q.jd_3d_5)
    return z


def neuron_69(Q):
    z = -7.503376e-06
    return z


def neuron_70(Q):
    z = 0.01755179
    return z


def neuron_71(Q):
    z = 2.514683e-06
    return z


def neuron_72(Q):
    z = -1.100034e-06
    return z


def neuron_73(Q):
    z = -1.803503e-05
    return z


def neuron_74(Q):
    z = -4.800163e-06
    return z


def neuron_75(Q):
    z = 1.385876e-06
    return z


def neuron_76(Q):
    z = -4.474226e-06
    return z


def neuron_77(Q):
    z = -2.003327e-06
    return z


def neuron_78(Q):
    z = 0.2458175
    if Q.lep_z < 0.3396572:
        z += 3.433353 * Q.lep_z - 1.166163
    if Q.n_s3d_above_3 < 4.0:
        z += -0.03887039 * Q.n_s3d_above_3 + 0.3498335
    if 4.0 <= Q.n_s3d_above_3 < 6.0:
        z += -0.05810801 * Q.n_s3d_above_3 + 0.426784
    if 6.0 <= Q.n_s3d_above_3 < 9.0:
        z += -0.0255446 * Q.n_s3d_above_3 + 0.2314035
    if Q.n_s3d_above_3 >= 9.0:
        z += 0.01332579 * Q.n_s3d_above_3 - 0.11843
    if Q.n_pairs_kt_above_1 < 195.0:
        z += -0.002864256 * Q.n_pairs_kt_above_1 + 0.5585299
    if Q.M3_b2 < 0.0142807:
        z += 17.84902 * Q.M3_b2 - 0.2548965
    if 9.380468 <= Q.mass_displaced5 < 21.12768:
        z += 0.01214146 * Q.mass_displaced5 - 0.1138926
    if Q.mass_displaced5 >= 21.12768:
        z += -0.004492995 * Q.mass_displaced5 + 0.2375549
    if Q.lep_ptrel < 18.7678:
        z += 0.05861844 * Q.lep_ptrel - 0.8351639
    if 18.7678 <= Q.lep_ptrel < 43.20788:
        z += -0.01084183 * Q.lep_ptrel + 0.4684524
    if Q.mass < 95.14961:
        z += 0.008452771 * Q.mass - 0.8042779
    if Q.mass >= 114.0172:
        z += 0.005173358 * Q.mass - 0.589852
    if Q.lepsj_3_n_d3 < 2.0:
        z += -0.03678843 * Q.lepsj_3_n_d3 + 0.07357685
    if Q.mass_top40 < 70.88236:
        z += -0.008048531 * Q.mass_top40 + 0.5704989
    if Q.n_neutral < 22.0:
        z += -0.01405236 * Q.n_neutral + 0.3091518
    if Q.pair_max_lnm2 >= 7.347625:
        z += -0.3622111 * Q.pair_max_lnm2 + 2.661391
    if Q.sdb_2_n < 4.0:
        z += -0.03032927 * Q.sdb_2_n + 0.2729635
    if 4.0 <= Q.sdb_2_n < 9.0:
        z += -0.01569953 * Q.sdb_2_n + 0.2144445
    if Q.sdb_2_n >= 9.0:
        z += 0.01462975 * Q.sdb_2_n - 0.05851899
    if Q.z_top15_slots >= 0.8008865:
        z += -2.076848 * Q.z_top15_slots + 1.66332
    if Q.n_sd0_above_5 >= 4.0:
        z += -0.02051266 * Q.n_sd0_above_5 + 0.08205063
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.006132643 * Q.mres_sd_mass_b0z005 - 0.7487957
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.0114392 * Q.mres_sd_mass_b0z005 + 2.061368
    if Q.mass_top20 >= 130.7093:
        z += 0.01136216 * Q.mass_top20 - 1.48514
    if Q.lund3_lndelta >= -2.817283:
        z += 0.0440309 * Q.lund3_lndelta + 0.1240475
    if Q.n_charged_had >= 22.0:
        z += 0.02388223 * Q.n_charged_had - 0.5254091
    if Q.sj4_pair_mass_max < 72.862:
        z += -0.0007624761 * Q.sj4_pair_mass_max + 0.02256277
    if 72.862 <= Q.sj4_pair_mass_max < 115.5142:
        z += 0.0007735302 * Q.sj4_pair_mass_max - 0.08935372
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.1003332 * Q.lepsj_3_maxsd0 + 0.0897334
    if 2.413264 <= Q.lepsj_3_maxsd0 < 586.9572:
        z += 0.000260711 * Q.lepsj_3_maxsd0 - 0.1530262
    if Q.n_lepton < 1.0:
        z += 0.6246441 * Q.n_lepton - 0.5743919
    if 1.0 <= Q.n_lepton < 2.0:
        z += -0.05025216 * Q.n_lepton + 0.1005043
    if Q.mass_displaced3 >= 39.09615:
        z += 0.01345218 * Q.mass_displaced3 - 0.5259286
    if Q.n_pairs_kt_above_3 < 99.0:
        z += -0.004022877 * Q.n_pairs_kt_above_3 + 0.3982648
    if Q.mass_2charged < 1.839882:
        z += -0.1205715 * Q.mass_2charged + 0.1126726
    if 1.839882 <= Q.mass_2charged < 10.83159:
        z += 0.0121406 * Q.mass_2charged - 0.131502
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += 0.1286278 * Q.sjf_2_1_n_d3 - 0.643139
    if Q.n_sdz_above_2 < 10.0:
        z += -0.00859939 * Q.n_sdz_above_2 + 0.0859939
    if Q.max_abs_d0 < 10.52344:
        z += 0.00326004 * Q.max_abs_d0 - 0.03430683
    if Q.lep_iso < 1.362094:
        z += 0.02712958 * Q.lep_iso - 0.03695305
    if Q.lepsj_2_dr < 0.103005:
        z += 5.133832 * Q.lepsj_2_dr - 0.5288103
    if Q.sdb_0_z < 0.0772633:
        z += -1.31103 * Q.sdb_0_z + 0.1012945
    if Q.mass_top15 >= 83.68425:
        z += 0.006753179 * Q.mass_top15 - 0.5651347
    if Q.e4 < 2.097213e-06:
        z += -24554.84 * Q.e4 + 0.05149672
    if Q.sdb_2_z < 0.2040425:
        z += 2.11821 * Q.sdb_2_z - 0.4322048
    if Q.sdb_2_z >= 0.2740506:
        z += -0.1375603 * Q.sdb_2_z + 0.03769849
    if Q.dc_2_jp < 10.98011:
        z += 0.006423697 * Q.dc_2_jp - 0.07053288
    if Q.mass_top10 >= 103.4976:
        z += -0.007123201 * Q.mass_top10 + 0.7372345
    if Q.n_pairs_kt_above_10 < 7.0:
        z += -0.01302632 * Q.n_pairs_kt_above_10 + 0.09118422
    if Q.N2_b05 < 0.5083429:
        z += -3.306854 * Q.N2_b05 + 1.681016
    if Q.ktd_ln_d34 < -8.400697:
        z += 0.09784251 * Q.ktd_ln_d34 + 0.8219452
    if Q.lepsj_2_maxsd0 < 2.220372:
        z += -0.04963728 * Q.lepsj_2_maxsd0 + 0.1102132
    if Q.kt2_1_z < 0.8153708:
        z += 0.2710522 * Q.kt2_1_z - 0.2210081
    if Q.sj3_pair_mass_min < 37.19471:
        z += 0.003123076 * Q.sj3_pair_mass_min - 0.1161619
    if Q.z_dr_0_0p05 < 0.007652966:
        z += -5.436442 * Q.z_dr_0_0p05 + 0.04160491
    if Q.mass_neutral < 46.19491:
        z += -0.001586713 * Q.mass_neutral + 0.07329807
    if Q.lep_z < 0.3396572 and Q.pz_lnd2 < 0.1956014:
        z += 1.68641 * (0.3396572 - Q.lep_z) * (0.1956014 - Q.pz_lnd2)
    if Q.lep_z < 0.3396572 and Q.mass_top20 > 86.78877:
        z += 0.01110462 * (0.3396572 - Q.lep_z) * (Q.mass_top20 - 86.78877)
    if Q.mass_displaced5 > 9.380468 and Q.mass_charged < 113.5019:
        z += -0.0001386115 * (Q.mass_displaced5 - 9.380468) * (113.5019 - Q.mass_charged)
    if Q.lep_ptrel < 18.7678 and Q.ktd_ln_d34 < -7.886954:
        z += 0.0067974 * (18.7678 - Q.lep_ptrel) * (-7.886954 - Q.ktd_ln_d34)
    if Q.n_s3d_above_3 > 4.0 and Q.sip_3d_2 < 226.3008:
        z += 0.0004316492 * (Q.n_s3d_above_3 - 4.0) * (226.3008 - Q.sip_3d_2)
    if Q.lep_ptrel < 18.7678 and Q.lepsj_2_maxsd0 < 131.3722:
        z += -2.813093e-05 * (18.7678 - Q.lep_ptrel) * (131.3722 - Q.lepsj_2_maxsd0)
    if Q.n_s3d_above_3 > 4.0 and Q.z_top50_slots > 0.9572293:
        z += 0.6532394 * (Q.n_s3d_above_3 - 4.0) * (Q.z_top50_slots - 0.9572293)
    if Q.lep_ptrel < 18.7678 and Q.nca_sj4_pairmax_over_mass > 0.6162845:
        z += 0.01519294 * (18.7678 - Q.lep_ptrel) * (Q.nca_sj4_pairmax_over_mass - 0.6162845)
    if Q.lep_ptrel < 18.7678 and Q.sj4_dr_min < 0.1631992:
        z += 0.0556846 * (18.7678 - Q.lep_ptrel) * (0.1631992 - Q.sj4_dr_min)
    if Q.lepsj_3_n_d3 < 2.0 and Q.jet_e > 835.8719:
        z += -0.0001254218 * (2.0 - Q.lepsj_3_n_d3) * (Q.jet_e - 835.8719)
    if Q.lep_ptrel < 18.7678 and Q.ak02_2_z > 0.08487383:
        z += -0.01917171 * (18.7678 - Q.lep_ptrel) * (Q.ak02_2_z - 0.08487383)
    if Q.lep_z < 0.3396572 and Q.ecf_g42 < 8.17233e-05:
        z += 9325.521 * (0.3396572 - Q.lep_z) * (8.17233e-05 - Q.ecf_g42)
    if Q.n_s3d_above_3 > 4.0 and Q.tau32 < 0.863865:
        z += -0.07782804 * (Q.n_s3d_above_3 - 4.0) * (0.863865 - Q.tau32)
    if Q.lep_ptrel < 18.7678 and Q.sum_pt_top30 < 749.0582:
        z += 4.366737e-05 * (18.7678 - Q.lep_ptrel) * (749.0582 - Q.sum_pt_top30)
    if Q.lepsj_3_n_d3 < 2.0 and Q.ak02_3_n_lep > 0.0:
        z += 0.05543564 * (2.0 - Q.lepsj_3_n_d3) * (Q.ak02_3_n_lep - 0.0)
    if Q.lepsj_3_n_d3 < 2.0 and Q.dc_2_n_lep < 1.0:
        z += 0.02449914 * (2.0 - Q.lepsj_3_n_d3) * (1.0 - Q.dc_2_n_lep)
    if Q.z_top15_slots > 0.8008865 and Q.z_displaced5 > 0.2184442:
        z += -4.997081 * (Q.z_top15_slots - 0.8008865) * (Q.z_displaced5 - 0.2184442)
    if Q.lep_z < 0.3396572 and Q.ak02_n > 1.0:
        z += 0.06923107 * (0.3396572 - Q.lep_z) * (Q.ak02_n - 1.0)
    if Q.lep_ptrel < 18.7678 and Q.mass_2photon < 22.18431:
        z += -0.0006135291 * (18.7678 - Q.lep_ptrel) * (22.18431 - Q.mass_2photon)
    if Q.lep_ptrel < 18.7678 and Q.sjf_4_2_z_d3 < 0.1570831:
        z += 0.06140242 * (18.7678 - Q.lep_ptrel) * (0.1570831 - Q.sjf_4_2_z_d3)
    if Q.lund3_lndelta > -2.817283 and Q.jd_3d_4 < 172.888:
        z += 0.000140411 * (Q.lund3_lndelta - -2.817283) * (172.888 - Q.jd_3d_4)
    if Q.lepsj_3_n_d3 < 2.0 and Q.lepsj_3_dr < 0.09790963:
        z += 0.5227246 * (2.0 - Q.lepsj_3_n_d3) * (0.09790963 - Q.lepsj_3_dr)
    if Q.mass_displaced3 > 39.09615 and Q.sjq_2_prod_k05 < 0.004703917:
        z += -0.02822402 * (Q.mass_displaced3 - 39.09615) * (0.004703917 - Q.sjq_2_prod_k05)
    if Q.sdb_2_n < 9.0 and Q.sjq_2_prod_k03 > -0.9998116:
        z += -0.04206399 * (9.0 - Q.sdb_2_n) * (Q.sjq_2_prod_k03 - -0.9998116)
    if Q.lep_z < 0.3396572 and Q.lepsj_2_dr < 0.103005:
        z += 28.33952 * (0.3396572 - Q.lep_z) * (0.103005 - Q.lepsj_2_dr)
    if Q.lepsj_2_dr < 0.103005 and Q.jet_charge_k03 < 0.1164034:
        z += 0.7166957 * (0.103005 - Q.lepsj_2_dr) * (0.1164034 - Q.jet_charge_k03)
    if Q.sj4_pair_mass_max < 115.5142 and Q.n_neutral_had < 9.0:
        z += -0.0003915269 * (115.5142 - Q.sj4_pair_mass_max) * (9.0 - Q.n_neutral_had)
    return z


def neuron_79(Q):
    z = -3.648196e-05
    return z


def neuron_80(Q):
    z = 1.040857e-05
    return z


def neuron_81(Q):
    z = 0.1113035
    if Q.z_displaced3 < 0.03624058:
        z += 5.392102 * Q.z_displaced3 - 0.3459808
    if 0.03624058 <= Q.z_displaced3 < 0.06416437:
        z += 6.408797 * Q.z_displaced3 - 0.3828264
    if Q.z_displaced3 >= 0.06416437:
        z += 1.016694 * Q.z_displaced3 - 0.03684559
    if Q.pz_lnd2 < 0.1339824:
        z += 0.6694812 * Q.pz_lnd2 - 0.08969866
    if Q.mass_top40 >= 115.7429:
        z += 0.002791734 * Q.mass_top40 - 0.3231234
    z += -0.05579631 * Q.dc_ntag
    if 129.5874 <= Q.mass_top50 < 161.1264:
        z += -0.002450416 * Q.mass_top50 + 0.3175431
    if Q.mass_top50 >= 161.1264:
        z += -0.01679942 * Q.mass_top50 + 2.629547
    if Q.ecf_g31 < 0.01162881:
        z += -45.50261 * Q.ecf_g31 + 0.5291413
    if Q.mres_sd_mass_b2z01 < 76.1529:
        z += -0.002230303 * Q.mres_sd_mass_b2z01 - 0.1558184
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 91.15481:
        z += 0.006364241 * Q.mres_sd_mass_b2z01 - 0.8103179
    if 91.15481 <= Q.mres_sd_mass_b2z01 < 115.0388:
        z += -0.02153913 * Q.mres_sd_mass_b2z01 + 1.733209
    if 115.0388 <= Q.mres_sd_mass_b2z01 < 125.2732:
        z += 0.004503548 * Q.mres_sd_mass_b2z01 - 1.262711
    if 125.2732 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.02121364 * Q.mres_sd_mass_b2z01 - 3.356036
    if Q.tau1 < 0.06074238:
        z += -8.125196 * Q.tau1 + 0.4935437
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += -0.05208524 * Q.sjf_2_1_n_d3 + 0.2604262
    if 2.0 <= Q.n_s3d_above_3 < 6.0:
        z += 0.03968464 * Q.n_s3d_above_3 - 0.07936928
    if Q.n_s3d_above_3 >= 6.0:
        z += 0.1419044 * Q.n_s3d_above_3 - 0.6926877
    if Q.sip_3d_2 < 226.3008:
        z += -0.001012026 * Q.sip_3d_2 + 0.2290222
    if Q.sum_zz_dr2 < 0.06117886:
        z += 6.201415 * Q.sum_zz_dr2 - 0.3793955
    if Q.jd_3d_5 < 4.004982:
        z += -0.1368296 * Q.jd_3d_5 + 0.5480002
    if Q.n_pairs_kt_above_1 < 58.0:
        z += -0.008339492 * Q.n_pairs_kt_above_1 + 0.4836905
    if Q.max_abs_d0 < 10.52344:
        z += -0.01799783 * Q.max_abs_d0 + 0.1893991
    if Q.sv_n >= 1.0:
        z += -0.1071052 * Q.sv_n + 0.1071052
    if Q.lepsj_3_dr < 0.005262883:
        z += 30.0022 * Q.lepsj_3_dr - 0.1578981
    z += -0.04170843 * Q.lepsj_2_n_d3
    if Q.sj4_pair_mass_max < 59.5457:
        z += 0.004457208 * Q.sj4_pair_mass_max - 0.2654076
    if Q.n_s3d_above_10 >= 3.0:
        z += 0.03485776 * Q.n_s3d_above_10 - 0.1045733
    if Q.sdb_0_n < 3.0:
        z += -0.02711745 * Q.sdb_0_n + 0.04133887
    if 3.0 <= Q.sdb_0_n < 5.0:
        z += 0.02000674 * Q.sdb_0_n - 0.1000337
    if Q.mass_displaced3 < 1.777286:
        z += 0.03177373 * Q.mass_displaced3 - 0.056471
    if Q.sum_z_dr2_top2 < 0.003472016:
        z += -39.29478 * Q.sum_z_dr2_top2 + 0.1364321
    if Q.sjf_4_n2disp >= 1.0:
        z += -0.158836 * Q.sjf_4_n2disp + 0.158836
    if Q.n_dr_0p4_up < 15.0:
        z += 0.02142989 * Q.n_dr_0p4_up - 0.3214484
    if Q.tau2 < 0.1076451:
        z += -0.9693292 * Q.tau2 + 0.1043435
    if Q.sjf_2_2_max3d < 2.044945:
        z += 0.2742109 * Q.sjf_2_2_max3d - 0.5607463
    if Q.pair_mean_lndelta >= -1.352792:
        z += 0.708984 * Q.pair_mean_lndelta + 0.9591076
    if Q.lep_iso < 14.38858:
        z += 0.008887704 * Q.lep_iso - 0.1278815
    if Q.sj3_pairmin_over_m >= 0.2477126:
        z += 0.4546885 * Q.sj3_pairmin_over_m - 0.1126321
    if Q.z_displaced3 > 0.03624058 and Q.n_s3d_above_3 < 7.0:
        z += -0.3676903 * (Q.z_displaced3 - 0.03624058) * (7.0 - Q.n_s3d_above_3)
    if Q.z_displaced3 > 0.03624058 and Q.z_charged_had > 0.265564:
        z += -4.927558 * (Q.z_displaced3 - 0.03624058) * (Q.z_charged_had - 0.265564)
    if Q.z_displaced3 > 0.03624058 and Q.jd_sum_abs_sd0_top3 < 1262.672:
        z += 0.0002774013 * (Q.z_displaced3 - 0.03624058) * (1262.672 - Q.jd_sum_abs_sd0_top3)
    if Q.ecf_g31 < 0.01162881 and Q.n_s3d_above_3 > 4.0:
        z += 4.459356 * (0.01162881 - Q.ecf_g31) * (Q.n_s3d_above_3 - 4.0)
    if Q.ecf_g31 < 0.01162881 and Q.ak02_2_n_lep < 1.0:
        z += 19.40284 * (0.01162881 - Q.ecf_g31) * (1.0 - Q.ak02_2_n_lep)
    if Q.z_displaced3 > 0.03624058 and Q.sip_3d_3 < 16.23838:
        z += 0.06281216 * (Q.z_displaced3 - 0.03624058) * (16.23838 - Q.sip_3d_3)
    if Q.pz_lnd2 < 0.1339824 and Q.sjq_2_2_nch < 15.0:
        z += -0.05109733 * (0.1339824 - Q.pz_lnd2) * (15.0 - Q.sjq_2_2_nch)
    if Q.sip_3d_2 < 226.3008 and Q.jd_3d_5 < 6.771002:
        z += -0.0003479328 * (226.3008 - Q.sip_3d_2) * (6.771002 - Q.jd_3d_5)
    if Q.sdb_4_z < 0.04510459 and Q.lep_z < 0.004135872:
        z += 1259.131 * (0.04510459 - Q.sdb_4_z) * (0.004135872 - Q.lep_z)
    if Q.sip_3d_2 < 226.3008 and Q.lepsj_2_n_d3 > 2.0:
        z += -0.0003528944 * (226.3008 - Q.sip_3d_2) * (Q.lepsj_2_n_d3 - 2.0)
    if Q.ecf_g31 < 0.01162881 and Q.sjq_2_prod_k05 > -0.5057096:
        z += -20.78933 * (0.01162881 - Q.ecf_g31) * (Q.sjq_2_prod_k05 - -0.5057096)
    if Q.max_abs_d0 < 10.52344 and Q.ak02_min12_n_disp3 < 1.0:
        z += 0.01843744 * (10.52344 - Q.max_abs_d0) * (1.0 - Q.ak02_min12_n_disp3)
    if Q.n_s3d_above_3 > 2.0 and Q.jet_e < 981.6443:
        z += 3.874467e-05 * (Q.n_s3d_above_3 - 2.0) * (981.6443 - Q.jet_e)
    if Q.z_displaced3 > 0.03624058 and Q.n_lund_kt_above_1 > 2.0:
        z += 0.126208 * (Q.z_displaced3 - 0.03624058) * (Q.n_lund_kt_above_1 - 2.0)
    if Q.max_abs_d0 < 10.52344 and Q.jd_3d_4 < 6.43167:
        z += -0.003919635 * (10.52344 - Q.max_abs_d0) * (6.43167 - Q.jd_3d_4)
    if Q.mres_sd_mass_b2z01 < 158.2019 and Q.mass_2charged > 24.4079:
        z += 0.0001286405 * (158.2019 - Q.mres_sd_mass_b2z01) * (Q.mass_2charged - 24.4079)
    if Q.lepsj_3_n_d3 < 2.0 and Q.ak02_min12_n_disp3 > 1.0:
        z += 0.09940477 * (2.0 - Q.lepsj_3_n_d3) * (Q.ak02_min12_n_disp3 - 1.0)
    if Q.ecf_g31 < 0.01162881 and Q.sjq_2_sumabs_k1 > 0.1588551:
        z += 31.14803 * (0.01162881 - Q.ecf_g31) * (Q.sjq_2_sumabs_k1 - 0.1588551)
    if Q.ecf_g31 < 0.01162881 and Q.sjf_4_3_z_d3 > 0.005113028:
        z += 184.3725 * (0.01162881 - Q.ecf_g31) * (Q.sjf_4_3_z_d3 - 0.005113028)
    if Q.sdb_4_z < 0.04510459 and Q.jet_abs_eta > 0.1941339:
        z += -1.804261 * (0.04510459 - Q.sdb_4_z) * (Q.jet_abs_eta - 0.1941339)
    if Q.z_displaced3 < 0.06416437 and Q.eccentricity > 0.7717404:
        z += -8.664352 * (0.06416437 - Q.z_displaced3) * (Q.eccentricity - 0.7717404)
    if Q.z_displaced3 < 0.06416437 and Q.dc_2_n_lep < 1.0:
        z += 3.438552 * (0.06416437 - Q.z_displaced3) * (1.0 - Q.dc_2_n_lep)
    if Q.z_displaced3 > 0.03624058 and Q.isphoton_32 > 0.0:
        z += 0.2286088 * (Q.z_displaced3 - 0.03624058) * (Q.isphoton_32 - 0.0)
    return z


def neuron_82(Q):
    z = -3.398021e-06
    return z


def neuron_83(Q):
    z = 1.234208
    if 78.33213 <= Q.mass_top40 < 155.8928:
        z += 0.004678489 * Q.mass_top40 - 0.366476
    if Q.mass_top40 >= 155.8928:
        z += 0.03062748 * Q.mass_top40 - 4.411736
    if Q.n_s3d_above_3 < 3.0:
        z += -0.0706742 * Q.n_s3d_above_3 + 0.2120226
    if Q.n_s3d_above_3 >= 6.0:
        z += 0.03139741 * Q.n_s3d_above_3 - 0.1883845
    if Q.z_displaced3 < 0.06416437:
        z += -5.431294 * Q.z_displaced3 + 0.4565054
    if 0.06416437 <= Q.z_displaced3 < 0.1681173:
        z += -1.039026 * Q.z_displaced3 + 0.1746783
    if Q.mass_top30 < 162.7874:
        z += -0.001366569 * Q.mass_top30 + 0.2224602
    if Q.n_s3d_above_10 < 6.0:
        z += 0.07013637 * Q.n_s3d_above_10 - 0.4208182
    if Q.mres_sd_mass_b2z01 < 81.19466:
        z += 0.01301241 * Q.mres_sd_mass_b2z01 - 1.812062
    if 81.19466 <= Q.mres_sd_mass_b2z01 < 130.0968:
        z += -0.003771861 * Q.mres_sd_mass_b2z01 - 0.4492691
    if 130.0968 <= Q.mres_sd_mass_b2z01 < 139.2565:
        z += 0.02468542 * Q.mres_sd_mass_b2z01 - 4.151471
    if 139.2565 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.01167301 * Q.mres_sd_mass_b2z01 - 2.339409
    if Q.mres_sd_mass_b2z01 >= 158.2019:
        z += 0.007679575 * Q.mres_sd_mass_b2z01 - 1.70764
    if Q.N2 < 0.258375:
        z += 2.246896 * Q.N2 - 0.5805417
    if Q.M3_b2 < 0.02160244:
        z += 23.58629 * Q.M3_b2 - 0.5095214
    if Q.sdb_2_z < 0.1530389:
        z += -1.240479 * Q.sdb_2_z - 0.1658407
    if 0.1530389 <= Q.sdb_2_z < 0.5109872:
        z += 0.9936696 * Q.sdb_2_z - 0.5077524
    if Q.mass_top15 >= 77.22442:
        z += -0.006376761 * Q.mass_top15 + 0.4924417
    if Q.pair_max_lnm2 >= 7.524613:
        z += 0.2534408 * Q.pair_max_lnm2 - 1.907044
    if Q.sj4_pair_mass_max < 75.78208:
        z += -0.003955719 * Q.sj4_pair_mass_max + 0.8608888
    if 75.78208 <= Q.sj4_pair_mass_max < 115.5142:
        z += -0.01412248 * Q.sj4_pair_mass_max + 1.631347
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += 0.1382222 * Q.sjf_2_1_n_d3 - 0.6911112
    if Q.mass_displaced5 < 1.262841:
        z += 0.08892298 * Q.mass_displaced5 - 0.1122956
    if Q.max_abs_d0 < 10.52344:
        z += -0.02604381 * Q.max_abs_d0 + 0.2740704
    if Q.jd_sum_abs_sd0_top5 < 185.1889:
        z += 0.0007261743 * Q.jd_sum_abs_sd0_top5 - 0.1344795
    if Q.sip_3d_1 < 7.028704:
        z += -0.08583194 * Q.sip_3d_1 + 0.6032873
    if Q.tau1 < 0.06074238:
        z += 12.10919 * Q.tau1 - 0.7355412
    if Q.sjf_4_n2disp >= 2.0:
        z += -0.1066637 * Q.sjf_4_n2disp + 0.2133274
    if 79.47361 <= Q.mass < 95.14961:
        z += -0.02717966 * Q.mass + 2.160066
    if 95.14961 <= Q.mass < 117.4867:
        z += -0.03477372 * Q.mass + 2.882637
    if Q.mass >= 117.4867:
        z += -0.01608111 * Q.mass + 0.686505
    if Q.n_charged_pt_above_10 < 5.0:
        z += 0.03803735 * Q.n_charged_pt_above_10 - 0.1901868
    if Q.sdb_2_n < 10.0:
        z += -0.01056435 * Q.sdb_2_n + 0.1056435
    if Q.dr_max_012 < 0.02210827:
        z += 11.01351 * Q.dr_max_012 - 0.2434895
    if Q.e4 < 3.53193e-06:
        z += 73108.38 * Q.e4 - 0.2582137
    if Q.n_pt_above_5 < 24.0:
        z += 0.02401032 * Q.n_pt_above_5 - 0.5762477
    if Q.sv_2_z < 0.03767806:
        z += 4.088055 * Q.sv_2_z - 0.15403
    if Q.tau5 >= 0.01467699:
        z += 12.98371 * Q.tau5 - 0.1905618
    if Q.sum_z_dr2_top50 < 0.0925671:
        z += -11.83423 * Q.sum_z_dr2_top50 + 1.09546
    if Q.D3_b05 < 1.139376:
        z += 0.3936735 * Q.D3_b05 - 0.4485421
    if Q.mres_pruned_mass < 86.14266:
        z += 0.0007318674 * Q.mres_pruned_mass - 0.06304501
    if Q.lep_iso < 6.185635:
        z += 0.01361827 * Q.lep_iso - 0.08423762
    if Q.lepsj_2_dr < 0.06573337:
        z += -3.799293 * Q.lepsj_2_dr + 0.2497403
    if Q.mres_sd_prong_mass1 < 27.26979:
        z += -0.00389803 * Q.mres_sd_prong_mass1 + 0.1062985
    if Q.dc_2_jp < 4.498447:
        z += 0.01117183 * Q.dc_2_jp - 0.0502559
    if Q.n_dr_0p4_up < 4.0:
        z += -0.02440565 * Q.n_dr_0p4_up + 0.09762262
    if Q.mass_top10 >= 103.4976:
        z += -0.006912847 * Q.mass_top10 + 0.7154634
    if Q.n_s3d_above_3 > 6.0 and Q.jd_sum_abs_sd0_top5 < 1292.625:
        z += 8.178865e-05 * (Q.n_s3d_above_3 - 6.0) * (1292.625 - Q.jd_sum_abs_sd0_top5)
    if Q.mass_top40 > 78.33213 and Q.lep_z < 0.1275041:
        z += -0.0621273 * (Q.mass_top40 - 78.33213) * (0.1275041 - Q.lep_z)
    if Q.mass_top30 < 162.7874 and Q.mass_displaced5 > 9.380468:
        z += -0.0002131063 * (162.7874 - Q.mass_top30) * (Q.mass_displaced5 - 9.380468)
    if Q.n_s3d_above_10 < 6.0 and Q.sdb_2_z > 0.2277628:
        z += -0.1254464 * (6.0 - Q.n_s3d_above_10) * (Q.sdb_2_z - 0.2277628)
    if Q.n_s3d_above_3 > 6.0 and Q.sj3_pairmin_over_m < 0.4866692:
        z += 0.3287843 * (Q.n_s3d_above_3 - 6.0) * (0.4866692 - Q.sj3_pairmin_over_m)
    if Q.n_s3d_above_3 > 6.0 and Q.jd_3d_4 < 14.85973:
        z += -0.008351668 * (Q.n_s3d_above_3 - 6.0) * (14.85973 - Q.jd_3d_4)
    if Q.mres_sd_mass_b2z01 < 139.2565 and Q.dc_2_n_lep < 1.0:
        z += 0.002203566 * (139.2565 - Q.mres_sd_mass_b2z01) * (1.0 - Q.dc_2_n_lep)
    if Q.sdb_2_z < 0.1530389 and Q.jd_3d_5 < 4.004982:
        z += -1.797876 * (0.1530389 - Q.sdb_2_z) * (4.004982 - Q.jd_3d_5)
    if Q.n_s3d_above_10 < 6.0 and Q.lne_0 > 4.380463:
        z += 0.02816666 * (6.0 - Q.n_s3d_above_10) * (Q.lne_0 - 4.380463)
    if Q.max_abs_d0 < 10.52344 and Q.jd_3d_5 < 6.771002:
        z += -0.009603727 * (10.52344 - Q.max_abs_d0) * (6.771002 - Q.jd_3d_5)
    if Q.mres_sd_mass_b2z01 < 139.2565 and Q.lund1_lnz > -2.478371:
        z += 0.003245723 * (139.2565 - Q.mres_sd_mass_b2z01) * (Q.lund1_lnz - -2.478371)
    if Q.mass_displaced3 < 13.03663 and Q.jd_3d_5 < 6.771002:
        z += 0.005558424 * (13.03663 - Q.mass_displaced3) * (6.771002 - Q.jd_3d_5)
    if Q.sj4_pair_mass_max < 115.5142 and Q.z_neutral_had < 0.3093201:
        z += -0.007811807 * (115.5142 - Q.sj4_pair_mass_max) * (0.3093201 - Q.z_neutral_had)
    if Q.mres_sd_mass_b2z01 > 130.0968 and Q.jd_3d_5 < 10.7744:
        z += 0.0003345061 * (Q.mres_sd_mass_b2z01 - 130.0968) * (10.7744 - Q.jd_3d_5)
    if Q.M3_b2 < 0.02160244 and Q.sjq_3_3_nch < 6.0:
        z += 1.0599 * (0.02160244 - Q.M3_b2) * (6.0 - Q.sjq_3_3_nch)
    if Q.n_s3d_above_3 > 6.0 and Q.jd_3d_4 < 172.888:
        z += 0.0007980616 * (Q.n_s3d_above_3 - 6.0) * (172.888 - Q.jd_3d_4)
    if Q.jd_sum_abs_sd0_top5 < 185.1889 and Q.sjf_4_4_z_d3 < 0.003370318:
        z += -0.1776125 * (185.1889 - Q.jd_sum_abs_sd0_top5) * (0.003370318 - Q.sjf_4_4_z_d3)
    if Q.n_s3d_above_3 > 6.0 and Q.lne_3 < 4.821289:
        z += -0.03744612 * (Q.n_s3d_above_3 - 6.0) * (4.821289 - Q.lne_3)
    if Q.z_displaced3 < 0.1681173 and Q.lepsj_3_mass < 12.91148:
        z += 0.03983401 * (0.1681173 - Q.z_displaced3) * (12.91148 - Q.lepsj_3_mass)
    if Q.sj4_pair_mass_max < 75.78208 and Q.sjf_4_3_z_d3 < 0.07303924:
        z += -0.1438711 * (75.78208 - Q.sj4_pair_mass_max) * (0.07303924 - Q.sjf_4_3_z_d3)
    if Q.sjf_2_1_n_d3 > 5.0 and Q.dc_tag_2nd < 0.1300849:
        z += -0.2439581 * (Q.sjf_2_1_n_d3 - 5.0) * (0.1300849 - Q.dc_tag_2nd)
    if Q.sv_2_z < 0.03767806 and Q.max_dr > 0.8033751:
        z += 7.062341 * (0.03767806 - Q.sv_2_z) * (Q.max_dr - 0.8033751)
    if Q.sj4_pair_mass_max < 115.5142 and Q.n_dr_0p2_0p4 > 9.0:
        z += -0.00016412 * (115.5142 - Q.sj4_pair_mass_max) * (Q.n_dr_0p2_0p4 - 9.0)
    if Q.sj4_pair_mass_max < 115.5142 and Q.tdz_1 > 0.05116378:
        z += 0.006047348 * (115.5142 - Q.sj4_pair_mass_max) * (Q.tdz_1 - 0.05116378)
    if Q.mass > 79.47361 and Q.dr_15 < 0.2535652:
        z += 0.005110961 * (Q.mass - 79.47361) * (0.2535652 - Q.dr_15)
    return z


def neuron_84(Q):
    z = -7.395848e-05
    return z


def neuron_85(Q):
    z = -6.321626e-07
    return z


def neuron_86(Q):
    z = -1.738634e-05
    return z


def neuron_87(Q):
    z = 1.42902e-06
    return z


def neuron_88(Q):
    z = 2.865482e-07
    return z


def neuron_89(Q):
    z = 1.553079e-06
    return z


def neuron_90(Q):
    z = 0.0009332118
    return z


def neuron_91(Q):
    z = 4.769881e-06
    return z


def neuron_92(Q):
    z = -6.790605e-06
    return z


def neuron_93(Q):
    z = 1.013547e-06
    return z


def neuron_94(Q):
    z = -1.278939e-06
    return z


def neuron_95(Q):
    z = 6.499782e-07
    return z


def neuron_96(Q):
    z = 5.085312e-07
    return z


def neuron_97(Q):
    z = -1.883063
    if Q.mass_displaced3 < 18.80005:
        z += -0.04233999 * Q.mass_displaced3 + 1.160906
    if 18.80005 <= Q.mass_displaced3 < 39.09615:
        z += -0.01797943 * Q.mass_displaced3 + 0.7029264
    if Q.mass < 95.14961:
        z += 0.01031939 * Q.mass - 2.88636
    if 95.14961 <= Q.mass < 100.4835:
        z += 0.02611711 * Q.mass - 4.389508
    if 100.4835 <= Q.mass < 149.0507:
        z += 0.04324723 * Q.mass - 6.110801
    if 149.0507 <= Q.mass < 164.4374:
        z += 0.004435614 * Q.mass - 0.3259033
    if 164.4374 <= Q.mass < 182.8592:
        z += -0.02190221 * Q.mass + 4.005021
    if Q.e3_b2 < 9.621843e-05:
        z += 3661.767 * Q.e3_b2 - 0.2203666
    if 9.621843e-05 <= Q.e3_b2 < 0.0002536827:
        z += 1137.649 * Q.e3_b2 + 0.02250003
    if 0.0002536827 <= Q.e3_b2 < 0.0007909605:
        z += -579.0338 * Q.e3_b2 + 0.4579928
    if Q.pair_mean_lnm2 >= 2.09826:
        z += -0.0238964 * Q.pair_mean_lnm2 + 0.05014088
    if Q.tau32 < 0.6513932:
        z += -0.402457 * Q.tau32 + 0.2621577
    if Q.sj3_pair_mass_max < 63.7508:
        z += -0.03783076 * Q.sj3_pair_mass_max + 3.88719
    if 63.7508 <= Q.sj3_pair_mass_max < 114.7848:
        z += -0.02891109 * Q.sj3_pair_mass_max + 3.318554
    if Q.mass_top50 < 71.80762:
        z += 0.02505661 * Q.mass_top50 - 2.215503
    if 71.80762 <= Q.mass_top50 < 122.0585:
        z += 0.008283374 * Q.mass_top50 - 1.011057
    if Q.sip_3d_2 < 447.0873:
        z += 0.0008012569 * Q.sip_3d_2 - 0.3582318
    if Q.jd_3d_4 < 172.888:
        z += 0.001301888 * Q.jd_3d_4 - 0.2250809
    if Q.n_pairs_kt_above_3 < 111.0:
        z += -0.002442735 * Q.n_pairs_kt_above_3 + 0.2711436
    if Q.ak02_min12_jp < 7.746064:
        z += -0.01119843 * Q.ak02_min12_jp + 0.08674376
    z += 2.701382 * Q.sj3_pairmax_over_m
    if Q.mass_top40 < 126.8853:
        z += -0.01469645 * Q.mass_top40 + 1.864763
    if Q.N2_b05 >= 0.4353632:
        z += -1.454592 * Q.N2_b05 + 0.6332756
    if Q.max_abs_d0 < 5.8125:
        z += 0.1064209 * Q.max_abs_d0 - 0.6185713
    if Q.lep_ptrel >= 3.53503:
        z += -0.006042353 * Q.lep_ptrel + 0.0213599
    if Q.sdb_2_z < 0.2740506:
        z += 0.7585734 * Q.sdb_2_z - 0.2078875
    if Q.pair_max_lnm2 < 7.075642:
        z += -0.04838996 * Q.pair_max_lnm2 + 0.34239
    if Q.n_sd0_above_3 < 6.0:
        z += 0.1325379 * Q.n_sd0_above_3 - 0.7952277
    if Q.sjf_3_1_n_d5 < 5.0:
        z += -0.02182571 * Q.sjf_3_1_n_d5 + 0.1091286
    if Q.sj2_mass2 < 1.851735:
        z += -0.1351251 * Q.sj2_mass2 + 0.2502158
    if Q.mres_pruned_mass < 70.04065:
        z += 0.002640221 * Q.mres_pruned_mass - 1.166775
    if 70.04065 <= Q.mres_pruned_mass < 112.1947:
        z += -0.008571259 * Q.mres_pruned_mass - 0.381516
    if 112.1947 <= Q.mres_pruned_mass < 124.3145:
        z += -0.02662666 * Q.mres_pruned_mass + 1.644204
    if 124.3145 <= Q.mres_pruned_mass < 171.3819:
        z += 0.01125389 * Q.mres_pruned_mass - 3.0649
    if Q.mres_pruned_mass >= 171.3819:
        z += -0.01121148 * Q.mres_pruned_mass + 0.7852593
    if Q.sd_mass >= 119.4443:
        z += 0.01409993 * Q.sd_mass - 1.684157
    if 53.57509 <= Q.mres_sd_mass_b0z005 < 76.40585:
        z += 0.02031204 * Q.mres_sd_mass_b0z005 - 1.088219
    if 76.40585 <= Q.mres_sd_mass_b0z005 < 91.82593:
        z += -0.06471592 * Q.mres_sd_mass_b0z005 + 5.408415
    if 91.82593 <= Q.mres_sd_mass_b0z005 < 97.29337:
        z += 0.02762144 * Q.mres_sd_mass_b0z005 - 3.07055
    if 97.29337 <= Q.mres_sd_mass_b0z005 < 122.1:
        z += 0.0008693058 * Q.mres_sd_mass_b0z005 - 0.4677441
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.0227928 * Q.mres_sd_mass_b0z005 - 3.144602
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.006405571 * Q.mres_sd_mass_b0z005 + 1.524925
    if Q.n_pairs_kt_above_1 < 325.0:
        z += -0.001515882 * Q.n_pairs_kt_above_1 + 0.4926617
    if Q.lund3_lndelta >= -1.780944:
        z += 0.09206456 * Q.lund3_lndelta + 0.1639619
    if Q.n_s3d_above_3 < 9.0:
        z += -0.07855238 * Q.n_s3d_above_3 + 0.7069714
    if Q.mass_charged >= 62.00562:
        z += 0.001333159 * Q.mass_charged - 0.08266333
    if Q.z_displaced5 >= 0.1339658:
        z += -0.4750423 * Q.z_displaced5 + 0.06363942
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += -0.05706789 * Q.sjf_2_1_n_d3 + 0.2853395
    if Q.mres_sd_prong_mass1 >= 69.04524:
        z += -0.006094058 * Q.mres_sd_prong_mass1 + 0.4207657
    if Q.lund_max_lnkt < 3.734077:
        z += 0.08287458 * Q.lund_max_lnkt - 0.3094601
    if Q.lep_z < 0.221436:
        z += 1.076614 * Q.lep_z - 0.2384012
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += 0.01310245 * Q.lepsj_3_maxsd0 - 0.03161969
    if Q.lepsj_2_dr < 0.103005:
        z += 0.4506392 * Q.lepsj_2_dr - 0.04641808
    if Q.n_photon < 8.0:
        z += -0.03210417 * Q.n_photon + 0.2568334
    if Q.lnerel_42 >= -6.337363:
        z += -0.06793914 * Q.lnerel_42 - 0.430555
    if 72.72359 <= Q.nca_sj4_pair_mass_2nd < 83.41384:
        z += -0.02581807 * Q.nca_sj4_pair_mass_2nd + 1.877583
    if Q.nca_sj4_pair_mass_2nd >= 83.41384:
        z += 0.004124295 * Q.nca_sj4_pair_mass_2nd - 0.620025
    if Q.pz_lnd3 < 0.2113485:
        z += 0.3680644 * Q.pz_lnd3 - 0.07778984
    if Q.dc_split2_dr >= 0.3591078:
        z += 0.6166699 * Q.dc_split2_dr - 0.221451
    if Q.e3_b2 < 0.0002536827 and Q.sdb_2_z < 0.3709098:
        z += 789.1324 * (0.0002536827 - Q.e3_b2) * (0.3709098 - Q.sdb_2_z)
    if Q.mass < 100.4835 and Q.lep_ptrel < 1.477152:
        z += 0.00538635 * (100.4835 - Q.mass) * (1.477152 - Q.lep_ptrel)
    if Q.mass_displaced3 < 39.09615 and Q.lep_ptrel < 18.7678:
        z += -0.0004253443 * (39.09615 - Q.mass_displaced3) * (18.7678 - Q.lep_ptrel)
    if Q.mass < 182.8592 and Q.dc_ntag > 0.0:
        z += -0.001080806 * (182.8592 - Q.mass) * (Q.dc_ntag - 0.0)
    if Q.mass < 182.8592 and Q.n_s3d_above_3 > 1.0:
        z += -0.0006111838 * (182.8592 - Q.mass) * (Q.n_s3d_above_3 - 1.0)
    if Q.e3_b2 < 0.0002536827 and Q.jet_charge_k03 > -0.05990128:
        z += 1302.575 * (0.0002536827 - Q.e3_b2) * (Q.jet_charge_k03 - -0.05990128)
    if Q.mass_top50 < 122.0585 and Q.ak02_2_n_lep < 1.0:
        z += 0.008779585 * (122.0585 - Q.mass_top50) * (1.0 - Q.ak02_2_n_lep)
    if Q.mass_displaced3 < 39.09615 and Q.pz_lnd3 < 0.18807:
        z += 0.07661735 * (39.09615 - Q.mass_displaced3) * (0.18807 - Q.pz_lnd3)
    if Q.mass_top50 < 122.0585 and Q.sum_e > 926.2598:
        z += 8.426549e-06 * (122.0585 - Q.mass_top50) * (Q.sum_e - 926.2598)
    if Q.sip_3d_2 < 447.0873 and Q.dc_1_n_lep < 1.0:
        z += 0.000366065 * (447.0873 - Q.sip_3d_2) * (1.0 - Q.dc_1_n_lep)
    if Q.sj3_pair_mass_max < 114.7848 and Q.n_photon > 12.0:
        z += -0.0002778433 * (114.7848 - Q.sj3_pair_mass_max) * (Q.n_photon - 12.0)
    if Q.mass_top50 < 122.0585 and Q.sjq_2_prod_k05 < 0.125305:
        z += -0.01336394 * (122.0585 - Q.mass_top50) * (0.125305 - Q.sjq_2_prod_k05)
    if Q.max_abs_d0 < 5.8125 and Q.jd_3d_5 < 4.004982:
        z += 0.04810275 * (5.8125 - Q.max_abs_d0) * (4.004982 - Q.jd_3d_5)
    if Q.tau32 < 0.6513932 and Q.jd_3d_6 < 3.151933:
        z += -0.2822906 * (0.6513932 - Q.tau32) * (3.151933 - Q.jd_3d_6)
    if Q.mass < 164.4374 and Q.jd_3d_6 > 16.11752:
        z += 8.643025e-06 * (164.4374 - Q.mass) * (Q.jd_3d_6 - 16.11752)
    if Q.sdb_2_z < 0.2740506 and Q.pz_lnkt1 < 0.1348361:
        z += 13.2187 * (0.2740506 - Q.sdb_2_z) * (0.1348361 - Q.pz_lnkt1)
    if Q.mass_top40 < 126.8853 and Q.n_muon > 1.0:
        z += -0.01149548 * (126.8853 - Q.mass_top40) * (Q.n_muon - 1.0)
    if Q.mass_displaced3 < 39.09615 and Q.D2 > 3.038341:
        z += 0.0008786343 * (39.09615 - Q.mass_displaced3) * (Q.D2 - 3.038341)
    if Q.mres_pruned_mass > 70.04065 and Q.lnerel_4 > -3.643775:
        z += -0.001592947 * (Q.mres_pruned_mass - 70.04065) * (Q.lnerel_4 - -3.643775)
    if Q.sj2_mass2 < 1.851735 and Q.mass_2photon > 1.504865:
        z += 0.01103406 * (1.851735 - Q.sj2_mass2) * (Q.mass_2photon - 1.504865)
    if Q.lund3_lndelta > -1.780944 and Q.z_photon > 0.2047275:
        z += 0.6104251 * (Q.lund3_lndelta - -1.780944) * (Q.z_photon - 0.2047275)
    if Q.sj3_pair_mass_min > 68.11898 and Q.sv_2_dr < 0.08408739:
        z += -0.04201277 * (Q.sj3_pair_mass_min - 68.11898) * (0.08408739 - Q.sv_2_dr)
    if Q.e3_b2 < 0.0007909605 and Q.sjq_3_sumabs_k1 > 0.4372817:
        z += 110.2187 * (0.0007909605 - Q.e3_b2) * (Q.sjq_3_sumabs_k1 - 0.4372817)
    if Q.mres_pruned_mass < 124.3145 and Q.ak02_2_charge < 0.09642216:
        z += -0.001063594 * (124.3145 - Q.mres_pruned_mass) * (0.09642216 - Q.ak02_2_charge)
    if Q.mass < 100.4835 and Q.lep_iso < 6.185635:
        z += -0.0008643802 * (100.4835 - Q.mass) * (6.185635 - Q.lep_iso)
    if Q.lep_z < 0.221436 and Q.lepsj_2_n_d3 < 4.0:
        z += 0.1114191 * (0.221436 - Q.lep_z) * (4.0 - Q.lepsj_2_n_d3)
    if Q.nca_sj4_pair_mass_2nd < 65.88119 and Q.nca_kt_above_10 > 1.0:
        z += -0.002828665 * (65.88119 - Q.nca_sj4_pair_mass_2nd) * (Q.nca_kt_above_10 - 1.0)
    if Q.ak02_min12_jp < 7.746064 and Q.mass_2photon > 0.2753928:
        z += 0.0005921089 * (7.746064 - Q.ak02_min12_jp) * (Q.mass_2photon - 0.2753928)
    return z


def neuron_98(Q):
    z = 1.024956e-06
    return z


def neuron_99(Q):
    z = -0.001933738
    return z


def neuron_100(Q):
    z = -7.11702e-06
    return z


def neuron_101(Q):
    z = 8.393245e-06
    return z


def neuron_102(Q):
    z = 5.119171e-06
    return z


def neuron_103(Q):
    z = -2.320888e-06
    return z


def neuron_104(Q):
    z = -1.498492
    if Q.mass < 117.4867:
        z += -0.02554279 * Q.mass + 3.16174
    if 117.4867 <= Q.mass < 149.0507:
        z += -0.005094483 * Q.mass + 0.7593362
    if Q.pair_mean_lndelta < -1.352792:
        z += -0.4833944 * Q.pair_mean_lndelta - 0.6539319
    if Q.sjq_2_1_nch < 18.0:
        z += -0.02222182 * Q.sjq_2_1_nch + 0.3999927
    if Q.lep_ptrel >= 6.983043:
        z += 0.01493063 * Q.lep_ptrel - 0.1042612
    if 3.208089 <= Q.mass_displaced3 < 6.341631:
        z += 0.05638087 * Q.mass_displaced3 - 0.1808748
    if Q.mass_displaced3 >= 6.341631:
        z += 0.01413487 * Q.mass_displaced3 + 0.08703376
    if Q.n_pt_above_1 < 58.0:
        z += -0.02421232 * Q.n_pt_above_1 + 1.404315
    if Q.sj4_pair_mass_max >= 128.0079:
        z += 0.008100469 * Q.sj4_pair_mass_max - 1.036924
    if Q.tau32_b2 < 0.4680886:
        z += 0.5496393 * Q.tau32_b2 - 0.2572799
    if Q.D2 < 1.226724:
        z += -0.09351266 * Q.D2 + 0.1147142
    if Q.D2 >= 3.597891:
        z += 0.1216282 * Q.D2 - 0.4376051
    if Q.n_pairs_kt_above_3 < 34.0:
        z += 0.01689615 * Q.n_pairs_kt_above_3 - 0.5744692
    if Q.N2 < 0.312717:
        z += -0.4061821 * Q.N2 + 0.1270201
    if Q.nca_sj4_pair_mass_2nd >= 77.3635:
        z += 0.007047674 * Q.nca_sj4_pair_mass_2nd - 0.5452327
    if Q.pz_lnd0 < 0.2827395:
        z += -1.941607 * Q.pz_lnd0 + 0.5489689
    if Q.n_s3d_above_10 < 6.0:
        z += -0.0661267 * Q.n_s3d_above_10 + 0.3967602
    if Q.sjf_2_1_max3d < 3.32421:
        z += 0.0915142 * Q.sjf_2_1_max3d - 0.3042124
    if Q.sjf_2_1_z_d3 >= 0.1149585:
        z += -0.2378896 * Q.sjf_2_1_z_d3 + 0.02734743
    if Q.sjf_3_2_max3d < 5.245219:
        z += 0.04796664 * Q.sjf_3_2_max3d - 0.2515955
    if Q.sjq_2_sumabs_k03 >= 0.6667228:
        z += -0.08868524 * Q.sjq_2_sumabs_k03 + 0.05912848
    if Q.tau3 < 0.05398263:
        z += 9.148561 * Q.tau3 - 0.4938634
    if Q.n_pairs_kt_above_1 < 146.0:
        z += 0.001045333 * Q.n_pairs_kt_above_1 - 0.1526186
    if Q.n_lund_kt_above_5 >= 1.0:
        z += -0.01771849 * Q.n_lund_kt_above_5 + 0.01771849
    if Q.lund_max_lndelta >= -0.529318:
        z += 0.6489441 * Q.lund_max_lndelta + 0.3434978
    if Q.sip_3d_2 < 226.3008:
        z += 0.0001256448 * Q.sip_3d_2 - 0.02843351
    if Q.sjf_3_2_maxsd0 < 61.73838:
        z += -0.001962305 * Q.sjf_3_2_maxsd0 + 0.1211495
    if Q.sdb_2_z < 0.3988697:
        z += -0.1507154 * Q.sdb_2_z + 0.06011579
    if Q.ak02_min12_jp < 14.23948:
        z += 0.01254669 * Q.ak02_min12_jp - 0.1786584
    if Q.tau21_b2 >= 0.3898586:
        z += 0.8289054 * Q.tau21_b2 - 0.3231559
    if Q.sjf_4_4_maxsd0 < 0.8080863:
        z += -0.04987688 * Q.sjf_4_4_maxsd0 + 0.04030482
    if Q.sjq_3_2_nch < 1.0:
        z += -0.2102448 * Q.sjq_3_2_nch + 0.2102448
    if Q.pair_max_lnm2 >= 5.908788:
        z += 0.0939526 * Q.pair_max_lnm2 - 0.5551459
    if Q.z_charged_had < 0.4501484:
        z += 0.852533 * Q.z_charged_had - 0.3837664
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 86.09683:
        z += 0.01967035 * Q.mres_sd_mass_b2z01 - 1.497954
    if Q.mres_sd_mass_b2z01 >= 86.09683:
        z += 0.001173558 * Q.mres_sd_mass_b2z01 + 0.09456088
    if 91.82593 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += -0.005305424 * Q.mres_sd_mass_b0z005 + 0.4871755
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.0009218659 * Q.mres_sd_mass_b0z005 - 0.2138617
    if Q.sj2_mass1 >= 53.51926:
        z += -0.002302815 * Q.sj2_mass1 + 0.1232449
    if Q.mass_top5 >= 17.63354:
        z += -0.002470572 * Q.mass_top5 + 0.04356495
    if Q.tau5 < 0.0230226:
        z += -31.81639 * Q.tau5 + 0.7324959
    if Q.mass_top50 < 135.5368:
        z += -0.003590908 * Q.mass_top50 + 0.4867001
    if Q.sjf_3_1_n_d3 < 4.0:
        z += 0.01761152 * Q.sjf_3_1_n_d3 - 0.07044608
    if Q.sjf_2_2_max3d < 16.96512:
        z += -0.00762484 * Q.sjf_2_2_max3d + 0.1293563
    if Q.sjq_2_2_nch < 2.0:
        z += -0.1856191 * Q.sjq_2_2_nch + 0.5526966
    if 2.0 <= Q.sjq_2_2_nch < 15.0:
        z += -0.01395833 * Q.sjq_2_2_nch + 0.209375
    if Q.mass_charged < 99.20396:
        z += -0.003509294 * Q.mass_charged + 0.3481359
    if Q.lepsj_2_maxsd0 < 1.546763:
        z += -0.09191086 * Q.lepsj_2_maxsd0 + 0.1421643
    if Q.sv_1_dr < 0.1040701:
        z += 0.6887389 * Q.sv_1_dr - 0.0716771
    if Q.pz_lnd2 < 0.01637527:
        z += -5.114402 * Q.pz_lnd2 + 0.08374973
    if Q.pz_lnd2 >= 0.1811493:
        z += 0.9860856 * Q.pz_lnd2 - 0.1786287
    if Q.dc_tag_2nd < 1.914834:
        z += 0.08751472 * Q.dc_tag_2nd - 0.1675761
    if Q.mass_top30 >= 126.5007:
        z += 0.004339342 * Q.mass_top30 - 0.5489298
    if Q.z_top3_slots >= 0.5395924:
        z += 1.885809 * Q.z_top3_slots - 1.017568
    if Q.sdb_2_n < 8.0:
        z += -0.01662919 * Q.sdb_2_n + 0.1330335
    if Q.C2 < 0.1798521:
        z += -1.522461 * Q.C2 + 0.2738177
    if Q.C2_b05 < 0.2234678:
        z += -2.039565 * Q.C2_b05 + 0.4557772
    if Q.n_pt_above_1 < 58.0 and Q.sjf_2_1_n_d5 > 1.0:
        z += -0.001704331 * (58.0 - Q.n_pt_above_1) * (Q.sjf_2_1_n_d5 - 1.0)
    if Q.mass_displaced3 > 3.208089 and Q.ak02_3_z < 0.1014193:
        z += 0.05737884 * (Q.mass_displaced3 - 3.208089) * (0.1014193 - Q.ak02_3_z)
    if Q.n_pt_above_1 < 58.0 and Q.z_neutral > 0.1384639:
        z += -0.01513914 * (58.0 - Q.n_pt_above_1) * (Q.z_neutral - 0.1384639)
    if Q.mass < 149.0507 and Q.sum_e < 1401.904:
        z += -7.528992e-06 * (149.0507 - Q.mass) * (1401.904 - Q.sum_e)
    if Q.sjq_2_1_nch < 18.0 and Q.sjf_2_2_n_d5 > 1.0:
        z += -0.001631926 * (18.0 - Q.sjq_2_1_nch) * (Q.sjf_2_2_n_d5 - 1.0)
    if Q.n_pt_above_1 < 58.0 and Q.pair_max_lnkt > 1.555555:
        z += -0.003765115 * (58.0 - Q.n_pt_above_1) * (Q.pair_max_lnkt - 1.555555)
    if Q.lep_ptrel > 6.983043 and Q.ak02_dr23 < 0.2385164:
        z += -0.07372688 * (Q.lep_ptrel - 6.983043) * (0.2385164 - Q.ak02_dr23)
    if Q.n_pt_above_1 < 58.0 and Q.ak02_dr13 < 0.4490565:
        z += 0.01958915 * (58.0 - Q.n_pt_above_1) * (0.4490565 - Q.ak02_dr13)
    if Q.pz_lnd0 < 0.2827395 and Q.n_pairs_kt_above_10 > 1.0:
        z += -0.03352565 * (0.2827395 - Q.pz_lnd0) * (Q.n_pairs_kt_above_10 - 1.0)
    if Q.mass < 149.0507 and Q.lnerel_65 > -18.42068:
        z += -0.0004162861 * (149.0507 - Q.mass) * (Q.lnerel_65 - -18.42068)
    if Q.lep_ptrel > 6.983043 and Q.mass_2charged < 28.89659:
        z += 0.0004138152 * (Q.lep_ptrel - 6.983043) * (28.89659 - Q.mass_2charged)
    if Q.sjf_2_1_z_d3 > 0.1149585 and Q.sjq_2_prod_k1 < 0.09802954:
        z += -1.707554 * (Q.sjf_2_1_z_d3 - 0.1149585) * (0.09802954 - Q.sjq_2_prod_k1)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.iselectron_1 > 0.0:
        z += -0.01242169 * (34.0 - Q.n_pairs_kt_above_3) * (Q.iselectron_1 - 0.0)
    if Q.sjf_3_2_max3d < 5.245219 and Q.jet_charge > 0.09659934:
        z += 0.0674535 * (5.245219 - Q.sjf_3_2_max3d) * (Q.jet_charge - 0.09659934)
    if Q.sjq_2_1_nch < 18.0 and Q.dr02 > 0.04115773:
        z += -0.02095557 * (18.0 - Q.sjq_2_1_nch) * (Q.dr02 - 0.04115773)
    if Q.sjf_2_1_z_d3 > 0.1149585 and Q.ischhad_0 > 0.0:
        z += 0.4617445 * (Q.sjf_2_1_z_d3 - 0.1149585) * (Q.ischhad_0 - 0.0)
    if Q.N2 < 0.312717 and Q.e3_b2 < 4.662928e-05:
        z += -40074.99 * (0.312717 - Q.N2) * (4.662928e-05 - Q.e3_b2)
    if Q.sjq_2_1_nch < 18.0 and Q.dr01 < 0.3258728:
        z += 0.01793439 * (18.0 - Q.sjq_2_1_nch) * (0.3258728 - Q.dr01)
    if Q.sjf_3_1_z_d3 < 0.02312549 and Q.dr_19 < 0.08897484:
        z += 35.45861 * (0.02312549 - Q.sjf_3_1_z_d3) * (0.08897484 - Q.dr_19)
    if Q.mass_top5 > 17.63354 and Q.lnpt_78 > -18.42068:
        z += -0.0004398166 * (Q.mass_top5 - 17.63354) * (Q.lnpt_78 - -18.42068)
    if Q.sjf_2_2_max3d < 4.516968 and Q.e3_b2 < 0.0007909605:
        z += -91.79868 * (4.516968 - Q.sjf_2_2_max3d) * (0.0007909605 - Q.e3_b2)
    if Q.mres_sd_mass_b0z005 > 91.82593 and Q.sv_1_n < 3.0:
        z += 0.0008183558 * (Q.mres_sd_mass_b0z005 - 91.82593) * (3.0 - Q.sv_1_n)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.kt2_2_n_lep < 1.0:
        z += 0.006493213 * (34.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.kt2_2_n_lep)
    if Q.n_pt_above_1 < 58.0 and Q.lepsj_2_n_d3 < 1.0:
        z += 0.004017771 * (58.0 - Q.n_pt_above_1) * (1.0 - Q.lepsj_2_n_d3)
    if Q.lepsj_2_maxsd0 < 1.546763 and Q.lepsj_2_dr < 0.0008210142:
        z += -104.1674 * (1.546763 - Q.lepsj_2_maxsd0) * (0.0008210142 - Q.lepsj_2_dr)
    if Q.mass_charged < 99.20396 and Q.ak02_2_n_lep < 1.0:
        z += -0.002087039 * (99.20396 - Q.mass_charged) * (1.0 - Q.ak02_2_n_lep)
    if Q.ak02_min12_jp < 14.23948 and Q.pz_lnkt1 > 0.06778673:
        z += 0.0810531 * (14.23948 - Q.ak02_min12_jp) * (Q.pz_lnkt1 - 0.06778673)
    return z


def neuron_105(Q):
    z = 2.674846e-05
    return z


def neuron_106(Q):
    z = 2.133717e-05
    return z


def neuron_107(Q):
    z = 1.20886e-05
    return z


def neuron_108(Q):
    z = -4.903451e-06
    return z


def neuron_109(Q):
    z = -9.579426e-07
    return z


def neuron_110(Q):
    z = 3.692848e-06
    return z


def neuron_111(Q):
    z = 3.964047e-06
    return z


def neuron_112(Q):
    z = 4.617259e-06
    return z


def neuron_113(Q):
    z = -2.6423e-05
    return z


def neuron_114(Q):
    z = 3.802002e-06
    return z


def neuron_115(Q):
    z = 1.381264
    if Q.pz_lnd0 < 0.06871203:
        z += -1.021206 * Q.pz_lnd0 - 0.0245137
    if 0.06871203 <= Q.pz_lnd0 < 0.1713451:
        z += 0.9225367 * Q.pz_lnd0 - 0.1580722
    if Q.N2_b2 < 0.2243273:
        z += -2.625397 * Q.N2_b2 + 0.5889483
    if Q.mass_displaced3 < 39.09615:
        z += 0.02608669 * Q.mass_displaced3 - 1.019889
    if Q.tau2 < 0.09733903:
        z += -7.999157 * Q.tau2 + 0.7786302
    if Q.tau43 < 0.8939856:
        z += 1.347427 * Q.tau43 - 1.20458
    if Q.z_neutral_had < 0.3788785:
        z += 2.896847 * Q.z_neutral_had
    if Q.z_neutral_had >= 0.3788785:
        z += 0.4074128 * Q.z_neutral_had + 0.9431929
    if Q.pair_mean_lnkt < 0.9367772:
        z += -0.8426619 * Q.pair_mean_lnkt + 0.7893865
    if -9.531553 <= Q.ktd_ln_d34 < -7.075141:
        z += -0.1042339 * Q.ktd_ln_d34 - 0.9935112
    if Q.ktd_ln_d34 >= -7.075141:
        z += -0.01230183 * Q.ktd_ln_d34 - 0.3430787
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.01755852 * Q.n_pairs_kt_above_1 - 1.313086
    if 80.0 <= Q.n_pairs_kt_above_1 < 411.0:
        z += -0.0002767243 * Q.n_pairs_kt_above_1 + 0.1137337
    if Q.lepsj_3_n_d3 < 3.0:
        z += 0.0646852 * Q.lepsj_3_n_d3 - 0.1940556
    if Q.e3_b2 < 0.0004126585:
        z += 537.1423 * Q.e3_b2 - 0.3102031
    if 0.0004126585 <= Q.e3_b2 < 0.0007909605:
        z += 234.0638 * Q.e3_b2 - 0.1851352
    if Q.n_s3d_above_3 >= 4.0:
        z += 0.09273591 * Q.n_s3d_above_3 - 0.3709436
    if Q.mass_top50 < 94.51361:
        z += 0.005875375 * Q.mass_top50 - 1.097746
    if 94.51361 <= Q.mass_top50 < 118.8408:
        z += 0.007066589 * Q.mass_top50 - 1.210332
    if 118.8408 <= Q.mass_top50 < 161.1264:
        z += 0.008762613 * Q.mass_top50 - 1.411888
    if Q.sj4_pair_mass_max < 75.78208:
        z += -0.008852856 * Q.sj4_pair_mass_max + 0.5054415
    if 75.78208 <= Q.sj4_pair_mass_max < 101.849:
        z += 0.006346983 * Q.sj4_pair_mass_max - 0.6464339
    if Q.sip_3d_2 < 447.0873:
        z += -0.000370652 * Q.sip_3d_2 + 0.1657138
    if Q.lam1 >= 0.04304553:
        z += -1.631979 * Q.lam1 + 0.07024939
    if Q.M2 < 0.120439:
        z += -9.806691 * Q.M2 + 1.181108
    if Q.lep_z < 0.5187302:
        z += -1.222123 * Q.lep_z + 0.6339519
    if Q.LHA < 0.5626523:
        z += 3.446412 * Q.LHA - 1.939132
    if Q.e3 < 0.00228569:
        z += -201.2047 * Q.e3 + 0.4598916
    if Q.n_sd0_above_3 >= 2.0:
        z += -0.07521743 * Q.n_sd0_above_3 + 0.1504349
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.04602683 * Q.lepsj_3_maxsd0 + 0.1110749
    if Q.sj4_pair_mass_min < 25.64898:
        z += -0.006818225 * Q.sj4_pair_mass_min + 0.1748805
    if Q.pt_entropy >= 3.22434:
        z += 0.6293398 * Q.pt_entropy - 2.029206
    if Q.jet_abs_eta >= 1.465946:
        z += 0.4015214 * Q.jet_abs_eta - 0.5886086
    if Q.max_abs_dz < 6.402344:
        z += -0.02134073 * Q.max_abs_dz + 0.1366307
    if Q.mass_2charged < 1.398635:
        z += 0.2008009 * Q.mass_2charged - 0.1460611
    if 1.398635 <= Q.mass_2charged < 6.767937:
        z += -0.05261736 * Q.mass_2charged + 0.2083785
    if 6.767937 <= Q.mass_2charged < 20.76537:
        z += 0.01055425 * Q.mass_2charged - 0.219163
    if Q.sum_pt_top10 < 428.9828:
        z += -0.00296221 * Q.sum_pt_top10 + 1.270737
    if Q.tau4 >= 0.04996:
        z += -9.836274 * Q.tau4 + 0.4914203
    if Q.psi_0p3 >= 0.9756505:
        z += 9.009368 * Q.psi_0p3 - 8.789994
    if Q.sj3_pair_mass_max < 114.7848:
        z += 0.01390405 * Q.sj3_pair_mass_max - 1.595974
    if Q.lep_iso < 0.4381892:
        z += 0.3812463 * Q.lep_iso - 0.167058
    if Q.n_lepton < 1.0:
        z += -0.4529256 * Q.n_lepton + 0.4529256
    if Q.lep_ptrel < 18.7678:
        z += 0.03332102 * Q.lep_ptrel - 0.6253621
    if Q.mres_sd_prong_mass2 < 7.065114:
        z += 0.03072901 * Q.mres_sd_prong_mass2 - 0.008948725
    if 7.065114 <= Q.mres_sd_prong_mass2 < 21.04127:
        z += -0.0148936 * Q.mres_sd_prong_mass2 + 0.3133802
    if Q.mass < 131.3917:
        z += -0.0223164 * Q.mass + 2.932191
    z += -1.253096 * Q.sj3_pairmax_over_m
    if Q.mass_top20 < 93.50967:
        z += -0.00523475 * Q.mass_top20 + 0.4894997
    if Q.psi_0p1 < 0.006220408:
        z += 13.42335 * Q.psi_0p1 - 0.08349874
    if Q.jd_3d_4 < 172.888:
        z += 0.0003773909 * Q.jd_3d_4 - 0.06524635
    if Q.max_abs_d0 < 5.8125:
        z += 0.01205101 * Q.max_abs_d0 - 0.0700465
    if Q.sjf_3_3_maxsd0 < 0.3479096:
        z += -1.117887 * Q.sjf_3_3_maxsd0 + 0.3889234
    if Q.pair_max_lnkt >= 1.555555:
        z += -0.2037217 * Q.pair_max_lnkt + 0.3169003
    if Q.sum_z_dr2_top3 < 0.02745856:
        z += 6.211377 * Q.sum_z_dr2_top3 - 0.1705555
    if Q.sj3_mass3 < 5.460258:
        z += 0.04384931 * Q.sj3_mass3 - 0.2394286
    if Q.dc_1_n_lep < 1.0:
        z += -0.08729424 * Q.dc_1_n_lep + 0.08729424
    if Q.lund_max_lnkt >= 3.388322:
        z += 0.1240695 * Q.lund_max_lnkt - 0.4203873
    if Q.dc_1_z < 0.932165:
        z += 0.4507625 * Q.dc_1_z - 0.420185
    if Q.mres_sd_mass_b1z01 >= 88.79082:
        z += -0.001582651 * Q.mres_sd_mass_b1z01 + 0.1405248
    if Q.M3_b2 < 0.01013989:
        z += -13.58255 * Q.M3_b2 + 0.1377255
    if Q.mass_displaced3 < 39.09615 and Q.lep_ptrel < 18.7678:
        z += 0.0009635926 * (39.09615 - Q.mass_displaced3) * (18.7678 - Q.lep_ptrel)
    if Q.mass_displaced3 < 39.09615 and Q.lam1_plus_lam2 > 0.04303099:
        z += -0.03666977 * (39.09615 - Q.mass_displaced3) * (Q.lam1_plus_lam2 - 0.04303099)
    if Q.pair_mean_lnkt < 0.9367772 and Q.sum_pt_top10 < 428.9828:
        z += -0.002092753 * (0.9367772 - Q.pair_mean_lnkt) * (428.9828 - Q.sum_pt_top10)
    if Q.lep_z < 0.5187302 and Q.tau54 < 0.8786609:
        z += -2.700989 * (0.5187302 - Q.lep_z) * (0.8786609 - Q.tau54)
    if Q.lep_z < 0.5187302 and Q.sj4_dr_min < 0.1845735:
        z += 2.266762 * (0.5187302 - Q.lep_z) * (0.1845735 - Q.sj4_dr_min)
    if Q.sip_3d_2 < 447.0873 and Q.jd_3d_4 < 9.982976:
        z += -5.298449e-05 * (447.0873 - Q.sip_3d_2) * (9.982976 - Q.jd_3d_4)
    if Q.lep_z < 0.5187302 and Q.lne_1 > 4.635336:
        z += 0.2479345 * (0.5187302 - Q.lep_z) * (Q.lne_1 - 4.635336)
    if Q.tau2 < 0.09733903 and Q.kt2_1_n_disp3 > 0.0:
        z += -1.636587 * (0.09733903 - Q.tau2) * (Q.kt2_1_n_disp3 - 0.0)
    if Q.lep_z < 0.5187302 and Q.ak02_n > 2.0:
        z += -0.159663 * (0.5187302 - Q.lep_z) * (Q.ak02_n - 2.0)
    if Q.pz_lnd0 < 0.1713451 and Q.z_displaced5 > 0.1046203:
        z += 5.306849 * (0.1713451 - Q.pz_lnd0) * (Q.z_displaced5 - 0.1046203)
    if Q.lep_z < 0.5187302 and Q.dc_n > 1.0:
        z += -0.07678311 * (0.5187302 - Q.lep_z) * (Q.dc_n - 1.0)
    if Q.sj3_pair_mass_min > 59.49644 and Q.n_lund_kt_above_5 < 4.0:
        z += 0.009339241 * (Q.sj3_pair_mass_min - 59.49644) * (4.0 - Q.n_lund_kt_above_5)
    if Q.sj4_pair_mass_max < 101.849 and Q.jd_3d_6 < 9.356714:
        z += -0.0009622605 * (101.849 - Q.sj4_pair_mass_max) * (9.356714 - Q.jd_3d_6)
    if Q.mass_top50 < 118.8408 and Q.jd_3d_6 < 9.356714:
        z += 0.001208491 * (118.8408 - Q.mass_top50) * (9.356714 - Q.jd_3d_6)
    if Q.mass_top50 < 118.8408 and Q.lepsj_2_dr < 0.103005:
        z += 0.05433538 * (118.8408 - Q.mass_top50) * (0.103005 - Q.lepsj_2_dr)
    if Q.lep_z < 0.5187302 and Q.lep_iso < 14.38858:
        z += -0.01725212 * (0.5187302 - Q.lep_z) * (14.38858 - Q.lep_iso)
    if Q.n_s3d_above_3 > 4.0 and Q.pz_lnkt1 < 0.1525194:
        z += -0.1237714 * (Q.n_s3d_above_3 - 4.0) * (0.1525194 - Q.pz_lnkt1)
    if Q.lep_z < 0.5187302 and Q.sdb_2_z < 0.2509165:
        z += 1.170273 * (0.5187302 - Q.lep_z) * (0.2509165 - Q.sdb_2_z)
    if Q.lep_z < 0.5187302 and Q.sdb_3_n > 2.0:
        z += 0.02981794 * (0.5187302 - Q.lep_z) * (Q.sdb_3_n - 2.0)
    if Q.max_abs_dz < 6.402344 and Q.lepsj_3_dr < 0.01012269:
        z += -1.884431 * (6.402344 - Q.max_abs_dz) * (0.01012269 - Q.lepsj_3_dr)
    if Q.mass_2charged < 1.398635 and Q.pz_lnd1 > 0.0906501:
        z += 1.195912 * (1.398635 - Q.mass_2charged) * (Q.pz_lnd1 - 0.0906501)
    if Q.sj3_pair_mass_max < 114.7848 and Q.sjf_3_3_n_d3 > 3.0:
        z += 0.00252434 * (114.7848 - Q.sj3_pair_mass_max) * (Q.sjf_3_3_n_d3 - 3.0)
    if Q.n_sd0_above_3 > 2.0 and Q.tau43_b2 < 0.9433644:
        z += 0.03475334 * (Q.n_sd0_above_3 - 2.0) * (0.9433644 - Q.tau43_b2)
    if Q.N2_b2 < 0.2243273 and Q.d0err_52 > 0.0:
        z += -10.34915 * (0.2243273 - Q.N2_b2) * (Q.d0err_52 - 0.0)
    if Q.ktd_ln_d34 > -9.531553 and Q.pz_lnkt3 < 0.102661:
        z += 0.8769117 * (Q.ktd_ln_d34 - -9.531553) * (0.102661 - Q.pz_lnkt3)
    if Q.e3_b2 < 0.0007909605 and Q.isphoton_53 < 1.0:
        z += -70.99541 * (0.0007909605 - Q.e3_b2) * (1.0 - Q.isphoton_53)
    if Q.e3_b2 < 0.0004126585 and Q.sdb_5_z > 0.04013001:
        z += 3084.277 * (0.0004126585 - Q.e3_b2) * (Q.sdb_5_z - 0.04013001)
    if Q.lepsj_3_maxsd0 < 2.413264 and Q.sv_1_sd0_sum < 509.6197:
        z += -0.0001624125 * (2.413264 - Q.lepsj_3_maxsd0) * (509.6197 - Q.sv_1_sd0_sum)
    if Q.ktd_ln_d34 > -9.531553 and Q.sjq_2_sumabs_k03 > 1.261566:
        z += 0.0656778 * (Q.ktd_ln_d34 - -9.531553) * (Q.sjq_2_sumabs_k03 - 1.261566)
    return z


def neuron_116(Q):
    z = -4.5462e-08
    return z


def neuron_117(Q):
    z = 2.534014e-06
    return z


def neuron_118(Q):
    z = -8.126637e-06
    return z


def neuron_119(Q):
    z = -1.123246e-06
    return z


def neuron_120(Q):
    z = 1.423138
    if Q.lep_ptrel < 3.53503:
        z += 0.1777336 * Q.lep_ptrel - 1.053954
    if 3.53503 <= Q.lep_ptrel < 6.983043:
        z += 0.04939628 * Q.lep_ptrel - 0.6002774
    if 6.983043 <= Q.lep_ptrel < 12.15228:
        z += 0.04712137 * Q.lep_ptrel - 0.5843916
    if 12.15228 <= Q.lep_ptrel < 27.3236:
        z += -0.002274911 * Q.lep_ptrel + 0.0158858
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.03163114 * Q.lep_ptrel - 0.9105498
    if Q.lep_ptrel >= 43.20788:
        z += -0.006042751 * Q.lep_ptrel + 0.7172594
    if Q.n_pairs_kt_above_3 < 62.0:
        z += 0.01312293 * Q.n_pairs_kt_above_3 - 0.8136218
    if Q.lep_z < 0.004135872:
        z += 147.6739 * Q.lep_z - 0.7015502
    if 0.004135872 <= Q.lep_z < 0.5187302:
        z += 0.1764297 * Q.lep_z - 0.09151941
    if Q.tau32 < 0.6800935:
        z += -0.7852207 * Q.tau32 + 0.5340235
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.0091155 * Q.n_pairs_kt_above_1 - 0.72924
    if Q.pair_mean_lnm2 >= 4.787589:
        z += -0.3413979 * Q.pair_mean_lnm2 + 1.634473
    if Q.mass_top10 >= 103.4976:
        z += -0.008944597 * Q.mass_top10 + 0.9257448
    if Q.pz_lnkt0 < 0.1589878:
        z += -4.362693 * Q.pz_lnkt0 + 0.6936148
    if Q.n_pairs_kt_above_10 < 11.0:
        z += 0.02696633 * Q.n_pairs_kt_above_10 - 0.2966296
    if Q.lep_iso < 0.4381892:
        z += 0.7696514 * Q.lep_iso - 0.1976192
    if 0.4381892 <= Q.lep_iso < 2.907433:
        z += -0.05654918 * Q.lep_iso + 0.1644129
    if Q.lepsj_2_dr < 0.0008210142:
        z += -484.093 * Q.lepsj_2_dr + 0.08083147
    if 0.0008210142 <= Q.lepsj_2_dr < 0.06573337:
        z += 4.877588 * Q.lepsj_2_dr - 0.3206203
    if Q.lep_dr < 0.08178299:
        z += -3.791596 * Q.lep_dr + 0.1543275
    if 0.08178299 <= Q.lep_dr < 0.3014662:
        z += 0.7090235 * Q.lep_dr - 0.2137466
    if Q.sjq_3_3_k1 >= 0.4037488:
        z += 0.653075 * Q.sjq_3_3_k1 - 0.2636783
    if Q.mass_charged < 68.20711:
        z += 0.01952197 * Q.mass_charged - 1.331537
    if Q.lam1 >= 0.04304553:
        z += -5.042574 * Q.lam1 + 0.2170603
    if Q.M2 < 0.05805236:
        z += -14.43076 * Q.M2 + 0.83774
    if Q.mass_neutral < 63.87145:
        z += -0.007253356 * Q.mass_neutral + 0.4632824
    if Q.mass < 71.96396:
        z += 0.01658354 * Q.mass - 2.899511
    if 71.96396 <= Q.mass < 90.08945:
        z += 0.04663502 * Q.mass - 5.062135
    if 90.08945 <= Q.mass < 105.7234:
        z += 0.04165556 * Q.mass - 4.613538
    if 105.7234 <= Q.mass < 114.0172:
        z += 0.02526821 * Q.mass - 2.881012
    if Q.lund3_lndelta >= -2.036501:
        z += -0.1189519 * Q.lund3_lndelta - 0.2422457
    if Q.pz_lnd2 < 0.03277088:
        z += 7.236154 * Q.pz_lnd2 - 0.2371351
    if Q.e3 >= 0.00228569:
        z += 54.09656 * Q.e3 - 0.123648
    if Q.dr_min_012 < 0.01394245:
        z += 3.470457 * Q.dr_min_012 - 0.04838667
    if Q.sjf_2_2_max3d < 3.029824:
        z += 0.04537917 * Q.sjf_2_2_max3d - 0.1374909
    if Q.jd_sum_abs_sd0_top5 < 124.9603:
        z += 0.000248911 * Q.jd_sum_abs_sd0_top5 - 0.03110399
    if Q.n_s3d_above_10 < 3.0:
        z += -0.03336141 * Q.n_s3d_above_10 + 0.1000842
    if Q.sj3_dr13 >= 0.7462286:
        z += -0.3936524 * Q.sj3_dr13 + 0.2937547
    if Q.kt2_min12_jp < 4.912483:
        z += 0.02261566 * Q.kt2_min12_jp - 0.1110991
    if Q.ak02_1_n_lep < 1.0:
        z += 0.1348 * Q.ak02_1_n_lep - 0.1348
    if Q.ak02_1_n_lep >= 1.0:
        z += -0.1889212 * Q.ak02_1_n_lep + 0.1889212
    if Q.pair_mean_lndelta >= -1.790445:
        z += -0.4841062 * Q.pair_mean_lndelta - 0.8667655
    if Q.pz_lnd0 < 0.05066241:
        z += -2.678753 * Q.pz_lnd0 + 0.1357121
    if Q.N2_b05 < 0.3667049:
        z += 3.401716 * Q.N2_b05 - 1.247426
    if Q.mass_top20 >= 114.4658:
        z += -0.00890706 * Q.mass_top20 + 1.019554
    if Q.mass_top50 < 135.5368:
        z += -0.005004371 * Q.mass_top50 + 0.8944061
    if 135.5368 <= Q.mass_top50 < 161.1264:
        z += -0.01122949 * Q.mass_top50 + 1.738138
    if 161.1264 <= Q.mass_top50 < 178.725:
        z += -0.00204599 * Q.mass_top50 + 0.2584344
    if Q.mass_top50 >= 178.725:
        z += 0.002958382 * Q.mass_top50 - 0.6359717
    if Q.mres_sd_prong_mass2 < 4.6892:
        z += 0.02422209 * Q.mres_sd_prong_mass2 - 0.1135822
    if Q.sd_mass < 78.4753:
        z += -0.002506578 * Q.sd_mass - 0.1576013
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += -0.01176948 * Q.sd_mass + 0.5693074
    if 88.81751 <= Q.sd_mass < 111.701:
        z += 0.02080229 * Q.sd_mass - 2.323636
    if Q.lep_ptrel < 3.53503 and Q.n_s3d_above_3 < 10.0:
        z += -0.04271611 * (3.53503 - Q.lep_ptrel) * (10.0 - Q.n_s3d_above_3)
    if Q.lep_z < 0.5187302 and Q.mass_neutral < 69.1565:
        z += -0.01525566 * (0.5187302 - Q.lep_z) * (69.1565 - Q.mass_neutral)
    if Q.lep_z < 0.5187302 and Q.mass_displaced3 > 0.0:
        z += 0.02269489 * (0.5187302 - Q.lep_z) * (Q.mass_displaced3 - 0.0)
    if Q.lep_ptrel < 3.53503 and Q.jd_3d_4 < 172.888:
        z += 0.0008852673 * (3.53503 - Q.lep_ptrel) * (172.888 - Q.jd_3d_4)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.sjf_2_1_z_d3 < 0.2710171:
        z += 0.01605531 * (62.0 - Q.n_pairs_kt_above_3) * (0.2710171 - Q.sjf_2_1_z_d3)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.lepsj_2_dr < 0.1822601:
        z += 0.01244685 * (62.0 - Q.n_pairs_kt_above_3) * (0.1822601 - Q.lepsj_2_dr)
    if Q.n_sd0_above_5 < 3.0 and Q.ak02_3_n_lep > 0.0:
        z += 0.07614529 * (3.0 - Q.n_sd0_above_5) * (Q.ak02_3_n_lep - 0.0)
    if Q.lep_ptrel < 3.53503 and Q.n_photon > 5.0:
        z += 0.006509946 * (3.53503 - Q.lep_ptrel) * (Q.n_photon - 5.0)
    if Q.lep_ptrel < 12.15228 and Q.sdb_2_n < 18.0:
        z += 0.00244015 * (12.15228 - Q.lep_ptrel) * (18.0 - Q.sdb_2_n)
    if Q.lep_z < 0.5187302 and Q.planar_flow < 0.7095008:
        z += 0.3181737 * (0.5187302 - Q.lep_z) * (0.7095008 - Q.planar_flow)
    if Q.lep_ptrel < 12.15228 and Q.jd_3d_6 < 3.151933:
        z += -0.02547817 * (12.15228 - Q.lep_ptrel) * (3.151933 - Q.jd_3d_6)
    if Q.lep_iso < 2.907433 and Q.jd_3d_6 < 5.367501:
        z += 0.03460871 * (2.907433 - Q.lep_iso) * (5.367501 - Q.jd_3d_6)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.dc_2_n_lep < 1.0:
        z += 0.003205749 * (62.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.dc_2_n_lep)
    if Q.lep_ptrel < 3.53503 and Q.e4_b05 > 4.363663e-05:
        z += 438.7903 * (3.53503 - Q.lep_ptrel) * (Q.e4_b05 - 4.363663e-05)
    if Q.mass_top10 > 103.4976 and Q.sum_pt_top40 > 502.3395:
        z += 2.377958e-05 * (Q.mass_top10 - 103.4976) * (Q.sum_pt_top40 - 502.3395)
    if Q.lep_ptrel < 3.53503 and Q.max_abs_d0 < 5.8125:
        z += 0.01303969 * (3.53503 - Q.lep_ptrel) * (5.8125 - Q.max_abs_d0)
    if Q.lepsj_2_dr < 0.06573337 and Q.sjq_3_3_k1 > 0.4037488:
        z += -7.463951 * (0.06573337 - Q.lepsj_2_dr) * (Q.sjq_3_3_k1 - 0.4037488)
    if Q.n_pairs_kt_above_10 < 11.0 and Q.jd_3d_6 < 3.151933:
        z += 0.01047878 * (11.0 - Q.n_pairs_kt_above_10) * (3.151933 - Q.jd_3d_6)
    if Q.lep_ptrel < 3.53503 and Q.sv_2_z > 0.0:
        z += -0.9703662 * (3.53503 - Q.lep_ptrel) * (Q.sv_2_z - 0.0)
    if Q.lep_ptrel < 12.15228 and Q.n_muon > 0.0:
        z += 0.02110722 * (12.15228 - Q.lep_ptrel) * (Q.n_muon - 0.0)
    if Q.lep_iso < 0.4381892 and Q.dc_1_n_lep < 1.0:
        z += 0.9008307 * (0.4381892 - Q.lep_iso) * (1.0 - Q.dc_1_n_lep)
    if Q.mass_top10 > 103.4976 and Q.ak02_dr23 < 0.3728877:
        z += -0.008316799 * (Q.mass_top10 - 103.4976) * (0.3728877 - Q.ak02_dr23)
    if Q.mass_charged < 68.20711 and Q.jd_3d_6 < 5.367501:
        z += 0.004328873 * (68.20711 - Q.mass_charged) * (5.367501 - Q.jd_3d_6)
    if Q.lepsj_2_dr < 0.06573337 and Q.jd_3d_6 < 5.367501:
        z += -1.33751 * (0.06573337 - Q.lepsj_2_dr) * (5.367501 - Q.jd_3d_6)
    if Q.mass_neutral < 63.87145 and Q.jd_3d_5 < 4.004982:
        z += 0.0005722868 * (63.87145 - Q.mass_neutral) * (4.004982 - Q.jd_3d_5)
    if Q.M2 < 0.05805236 and Q.jd_3d_5 < 6.771002:
        z += -4.460404 * (0.05805236 - Q.M2) * (6.771002 - Q.jd_3d_5)
    if Q.mass < 114.0172 and Q.jd_3d_6 < 5.367501:
        z += 0.002553246 * (114.0172 - Q.mass) * (5.367501 - Q.jd_3d_6)
    if Q.pz_lnkt0 < 0.1589878 and Q.sjq_2_prod_k05 > -0.3383985:
        z += -4.186494 * (0.1589878 - Q.pz_lnkt0) * (Q.sjq_2_prod_k05 - -0.3383985)
    if Q.lep_ptrel < 12.15228 and Q.pz_lnd2 > 0.008992646:
        z += -0.02938191 * (12.15228 - Q.lep_ptrel) * (Q.pz_lnd2 - 0.008992646)
    if Q.mass < 114.0172 and Q.z_neutral_had < 0.1271955:
        z += 0.08602759 * (114.0172 - Q.mass) * (0.1271955 - Q.z_neutral_had)
    if Q.n_s3d_above_10 < 3.0 and Q.sdb_4_z > 0.02669833:
        z += -0.4204153 * (3.0 - Q.n_s3d_above_10) * (Q.sdb_4_z - 0.02669833)
    if Q.lep_z < 0.5187302 and Q.sjq_3_3_k1 < -0.2912597:
        z += 0.7950054 * (0.5187302 - Q.lep_z) * (-0.2912597 - Q.sjq_3_3_k1)
    if Q.lam1 > 0.04304553 and Q.dc_3_n_lep > 0.0:
        z += -5.345132 * (Q.lam1 - 0.04304553) * (Q.dc_3_n_lep - 0.0)
    if Q.lam1 > 0.04304553 and Q.lund3_lnkt > 0.3409807:
        z += 0.8741571 * (Q.lam1 - 0.04304553) * (Q.lund3_lnkt - 0.3409807)
    if Q.lep_ptrel < 12.15228 and Q.sdb_3_n < 5.0:
        z += 0.002118569 * (12.15228 - Q.lep_ptrel) * (5.0 - Q.sdb_3_n)
    if Q.pz_lnd0 < 0.05066241 and Q.dr_19 < 0.0538419:
        z += 264.7917 * (0.05066241 - Q.pz_lnd0) * (0.0538419 - Q.dr_19)
    return z


def neuron_121(Q):
    z = -4.817918e-06
    return z


def neuron_122(Q):
    z = -1.850685e-06
    return z


def neuron_123(Q):
    z = 0.02041437
    return z


def neuron_124(Q):
    z = -0.2257918
    if Q.lep_z < 0.221436:
        z += -3.632413 * Q.lep_z + 0.8522745
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -0.4054047 * Q.lep_z + 0.1376986
    if Q.n_pairs_kt_above_1 < 58.0:
        z += 0.005362115 * Q.n_pairs_kt_above_1 - 1.146751
    if 58.0 <= Q.n_pairs_kt_above_1 < 366.0:
        z += 0.002713467 * Q.n_pairs_kt_above_1 - 0.9931289
    if Q.n_s3d_above_3 < 1.0:
        z += 0.404703 * Q.n_s3d_above_3 - 0.404703
    if Q.n_s3d_above_3 >= 3.0:
        z += 0.05572831 * Q.n_s3d_above_3 - 0.1671849
    if Q.mass < 95.14961:
        z += 0.003805213 * Q.mass - 0.2334734
    if 95.14961 <= Q.mass < 100.4835:
        z += -0.007746187 * Q.mass + 0.8656379
    if 100.4835 <= Q.mass < 117.4867:
        z += -0.005132805 * Q.mass + 0.6030362
    if Q.lep_ptrel < 43.20788:
        z += -0.005556046 * Q.lep_ptrel + 0.240065
    if Q.n_dr_0p4_up < 15.0:
        z += 0.01209119 * Q.n_dr_0p4_up - 0.1813678
    if Q.lep_iso < 0.4381892:
        z += 1.171495 * Q.lep_iso - 0.7395692
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += 0.2448659 * Q.lep_iso - 0.3335304
    if Q.n_lepton < 1.0:
        z += -0.9755 * Q.n_lepton + 0.9755
    if Q.mass_top40 >= 70.88236:
        z += -0.006533632 * Q.mass_top40 + 0.4631193
    if 76.40585 <= Q.mres_sd_mass_b0z005 < 86.65765:
        z += -0.009350866 * Q.mres_sd_mass_b0z005 + 0.7144609
    if 86.65765 <= Q.mres_sd_mass_b0z005 < 91.82593:
        z += -0.002537917 * Q.mres_sd_mass_b0z005 + 0.1240667
    if 91.82593 <= Q.mres_sd_mass_b0z005 < 111.9362:
        z += 0.007465023 * Q.mres_sd_mass_b0z005 - 0.7944626
    if 111.9362 <= Q.mres_sd_mass_b0z005 < 125.8718:
        z += 0.005324394 * Q.mres_sd_mass_b0z005 - 0.5548487
    if 125.8718 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += -0.02085224 * Q.mres_sd_mass_b0z005 + 2.740051
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.01016903 * Q.mres_sd_mass_b0z005 + 1.031546
    if Q.lepsj_3_maxsd0 < 0.8036986:
        z += 0.4275213 * Q.lepsj_3_maxsd0 - 0.2118686
    if 0.8036986 <= Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.08184178 * Q.lepsj_3_maxsd0 + 0.1975059
    if Q.z_displaced3 < 0.04884386:
        z += 12.87205 * Q.z_displaced3 - 0.6287207
    if Q.sjq_2_prod_k1 < 0.03330871:
        z += 2.61067 * Q.sjq_2_prod_k1 - 0.2036961
    if 0.03330871 <= Q.sjq_2_prod_k1 < 0.09802954:
        z += 1.803717 * Q.sjq_2_prod_k1 - 0.1768176
    if 0.1103483 <= Q.sjq_2_sumabs_k1 < 0.2769039:
        z += -0.1618266 * Q.sjq_2_sumabs_k1 + 0.01785729
    if Q.sjq_2_sumabs_k1 >= 0.2769039:
        z += -0.9424328 * Q.sjq_2_sumabs_k1 + 0.2340102
    if Q.z_photon >= 0.1149688:
        z += -1.682653 * Q.z_photon + 0.1934525
    if Q.lepsj_2_dr < 0.04178742:
        z += -5.744777 * Q.lepsj_2_dr + 0.2400594
    if Q.sjq_2_prod_k05 >= 0.004703917:
        z += 0.5978847 * Q.sjq_2_prod_k05 - 0.0028124
    if Q.mass_displaced5 < 1.777787:
        z += 0.02566498 * Q.mass_displaced5 - 0.04562688
    if Q.sjf_2_1_n_d3 >= 7.0:
        z += 0.2326251 * Q.sjf_2_1_n_d3 - 1.628376
    if Q.sjf_2_2_max3d < 1.864422:
        z += 0.5239474 * Q.sjf_2_2_max3d - 0.9768591
    if Q.tau2 < 0.1076451:
        z += -3.37243 * Q.tau2 + 0.3630255
    if Q.sj4_pair_mass_max < 66.70506:
        z += 0.01170432 * Q.sj4_pair_mass_max - 0.7807375
    if Q.sjf_4_1_n_d3 >= 2.0:
        z += -0.01461154 * Q.sjf_4_1_n_d3 + 0.02922308
    if Q.pz_lnd0 < 0.06871203:
        z += 4.133101 * Q.pz_lnd0 - 0.2839937
    if Q.kt2_2_n_lep < 1.0:
        z += -0.126365 * Q.kt2_2_n_lep + 0.126365
    if Q.lep_dr >= 0.04607888:
        z += 1.045974 * Q.lep_dr - 0.04819734
    if Q.sv_1_sd0_sum < 35.45098:
        z += -0.002023455 * Q.sv_1_sd0_sum + 0.07173347
    if Q.max_abs_d0 < 10.52344:
        z += 0.01616949 * Q.max_abs_d0 - 0.1701586
    if Q.z_displaced5 >= 0.04748739:
        z += 1.249812 * Q.z_displaced5 - 0.05935029
    if 69.04524 <= Q.mres_sd_prong_mass1 < 84.19324:
        z += -0.01715462 * Q.mres_sd_prong_mass1 + 1.184445
    if Q.mres_sd_prong_mass1 >= 84.19324:
        z += 0.01036539 * Q.mres_sd_prong_mass1 - 1.132554
    if Q.n_photon >= 5.0:
        z += 0.007104856 * Q.n_photon - 0.03552428
    if Q.sj3_pair_mass_max < 73.24742:
        z += 0.001266424 * Q.sj3_pair_mass_max - 0.09276229
    if Q.tau21_b2 >= 0.3898586:
        z += -0.2889846 * Q.tau21_b2 + 0.1126631
    if Q.jd_sum_abs_sd0_top5 < 55.83114:
        z += -0.02001393 * Q.jd_sum_abs_sd0_top5 + 1.117401
    if Q.jd_sum_abs_sd0_top3 < 48.72265:
        z += 0.015872 * Q.jd_sum_abs_sd0_top3 - 0.7733261
    if Q.n_s3d_above_10 >= 2.0:
        z += -0.07716309 * Q.n_s3d_above_10 + 0.1543262
    if Q.sjq_3_prod_k1 < -0.06334099:
        z += 0.3661602 * Q.sjq_3_prod_k1 + 0.02319295
    if Q.dc_3_charge >= 0.6283033:
        z += -0.4216487 * Q.dc_3_charge + 0.2649232
    if Q.sdb_2_z < 0.1530389:
        z += 4.21065 * Q.sdb_2_z - 0.6443933
    if Q.lepsj_3_dr < 0.04982189:
        z += -16.54941 * Q.lepsj_3_dr + 0.8245228
    if Q.tau32 < 0.5498426:
        z += -0.8450987 * Q.tau32 + 0.4646713
    if Q.psi_0p1 < 0.006220408:
        z += 31.71996 * Q.psi_0p1 - 0.1973111
    if Q.lund_max_lndelta >= -0.529318:
        z += 0.7215744 * Q.lund_max_lndelta + 0.3819424
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 91.15481:
        z += -0.02170968 * Q.mres_sd_mass_b2z01 + 1.653255
    if 91.15481 <= Q.mres_sd_mass_b2z01 < 107.2089:
        z += 0.01620757 * Q.mres_sd_mass_b2z01 - 1.803085
    if Q.mres_sd_mass_b2z01 >= 107.2089:
        z += 0.003631675 * Q.mres_sd_mass_b2z01 - 0.4548363
    if Q.lep_ptrel < 43.20788 and Q.sj3_pair_mass_max < 142.9952:
        z += 8.380188e-05 * (43.20788 - Q.lep_ptrel) * (142.9952 - Q.sj3_pair_mass_max)
    if Q.n_s3d_above_3 > 3.0 and Q.M2_b2 < 0.06762785:
        z += -1.062971 * (Q.n_s3d_above_3 - 3.0) * (0.06762785 - Q.M2_b2)
    if Q.mass_top40 > 70.88236 and Q.sv_n < 3.0:
        z += 0.0004337732 * (Q.mass_top40 - 70.88236) * (3.0 - Q.sv_n)
    if Q.n_dr_0p4_up < 15.0 and Q.dc_ntag > 0.0:
        z += 0.00448724 * (15.0 - Q.n_dr_0p4_up) * (Q.dc_ntag - 0.0)
    if Q.lep_z < 0.3396572 and Q.z_displaced5 > 0.1046203:
        z += -6.186712 * (0.3396572 - Q.lep_z) * (Q.z_displaced5 - 0.1046203)
    if Q.mass_top40 > 70.88236 and Q.ak02_3_z < 0.07650476:
        z += -0.01139116 * (Q.mass_top40 - 70.88236) * (0.07650476 - Q.ak02_3_z)
    if Q.n_pairs_kt_above_1 < 366.0 and Q.C2_b2 < 0.221369:
        z += 0.001401453 * (366.0 - Q.n_pairs_kt_above_1) * (0.221369 - Q.C2_b2)
    if Q.tau2 < 0.1076451 and Q.jet_charge_k03 > -0.05990128:
        z += 2.411338 * (0.1076451 - Q.tau2) * (Q.jet_charge_k03 - -0.05990128)
    if Q.lep_z < 0.221436 and Q.dc_4_z > 0.0:
        z += 7.566089 * (0.221436 - Q.lep_z) * (Q.dc_4_z - 0.0)
    if Q.z_photon > 0.1149688 and Q.jd_3d_5 < 6.771002:
        z += 0.1467544 * (Q.z_photon - 0.1149688) * (6.771002 - Q.jd_3d_5)
    if Q.sjq_2_sumabs_k1 > 0.2769039 and Q.jd_3d_4 < 2.953464:
        z += 0.6930624 * (Q.sjq_2_sumabs_k1 - 0.2769039) * (2.953464 - Q.jd_3d_4)
    if Q.n_lepton < 1.0 and Q.jd_3d_4 < 2.953464:
        z += -0.4243986 * (1.0 - Q.n_lepton) * (2.953464 - Q.jd_3d_4)
    if Q.n_s3d_above_3 > 3.0 and Q.jd_3d_4 < 172.888:
        z += -0.0003645154 * (Q.n_s3d_above_3 - 3.0) * (172.888 - Q.jd_3d_4)
    if Q.mass_top40 > 70.88236 and Q.lund1_lndelta < -0.4311181:
        z += 0.01328928 * (Q.mass_top40 - 70.88236) * (-0.4311181 - Q.lund1_lndelta)
    if Q.max_abs_d0 < 10.52344 and Q.ak02_min12_n_disp3 < 1.0:
        z += 0.0422136 * (10.52344 - Q.max_abs_d0) * (1.0 - Q.ak02_min12_n_disp3)
    if Q.tau2 < 0.1076451 and Q.sjf_4_3_z_d3 > 0.009093044:
        z += 24.63664 * (0.1076451 - Q.tau2) * (Q.sjf_4_3_z_d3 - 0.009093044)
    if Q.kt2_2_n_lep < 1.0 and Q.sdb_2_z > 0.3208052:
        z += -0.8923092 * (1.0 - Q.kt2_2_n_lep) * (Q.sdb_2_z - 0.3208052)
    if Q.sdb_2_z < 0.1530389 and Q.dr_30 < 0.04891969:
        z += 34.76419 * (0.1530389 - Q.sdb_2_z) * (0.04891969 - Q.dr_30)
    if Q.sjq_2_sumabs_k1 > 0.2769039 and Q.lepsj_3_maxsd0 < 0.8036986:
        z += 0.4390182 * (Q.sjq_2_sumabs_k1 - 0.2769039) * (0.8036986 - Q.lepsj_3_maxsd0)
    if Q.sjq_2_prod_k1 < 0.09802954 and Q.lepsj_3_maxsd0 < 2.413264:
        z += 0.4236838 * (0.09802954 - Q.sjq_2_prod_k1) * (2.413264 - Q.lepsj_3_maxsd0)
    if Q.lep_z < 0.221436 and Q.tau43 < 0.8096444:
        z += -1.744995 * (0.221436 - Q.lep_z) * (0.8096444 - Q.tau43)
    if Q.lepsj_3_dr < 0.04982189 and Q.lepsj_3_maxsd0 < 2.413264:
        z += -5.439985 * (0.04982189 - Q.lepsj_3_dr) * (2.413264 - Q.lepsj_3_maxsd0)
    return z


def neuron_125(Q):
    z = -5.899756e-07
    return z


def neuron_126(Q):
    z = 8.16904e-06
    return z


def neuron_127(Q):
    z = -6.845022e-08
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
