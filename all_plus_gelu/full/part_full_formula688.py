"""Particle Transformer (ParT) jet tagger, JetClass, input features 'full': smaller version of the 869 (tuned on the network's predictions; step 4), as if-statements.

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

Whole test file (2,000,000 jets): accuracy 75.25% (the network: 86.03%); same class as the network for 80.09% of jets.

Quantities:
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b05                 e3/e2² with β = 0.5
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3_b2                  e4·e2/e3² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M2_b2                  ₁e₃/e2 with β = 2
  Q.M3                     generalized ECF ratio M3
  Q.M3_b2                  ₁e₄/₁e₃ with β = 2
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N2_b05                 ₂e₃/(e2)² with β = 0.5
  Q.N2_b2                  ₂e₃/(e2)² with β = 2
  Q.N3_b05                 ₂e₄/(₁e₃)² with β = 0.5
  Q.N3_b2                  ₂e₄/(₁e₃)² with β = 2
  Q.ak02_1_n_disp3         hardest anti-kT 0.2 subjet: number of its tracks with d0/σ > 3
  Q.ak02_2_n_lep           2nd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_2_z               pT share of the 2nd anti-kT 0.2 subjet (0 if none)
  Q.ak02_3_n_lep           3rd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_3_z               pT share of the 3rd anti-kT 0.2 subjet (0 if none)
  Q.ak02_dr12              distance between the pT-weighted centres of the hardest and 2nd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr13              distance between the pT-weighted centres of the hardest and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr23              distance between the pT-weighted centres of the 2nd and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_min12_n_disp3     the smaller of the two hardest anti-kT 0.2 subjets: number of its tracks with d0/σ > 3
  Q.ak02_n                 number of anti-kT R = 0.2 subjets with pT > 10 GeV
  Q.dc_1_n_disp3           hardest prong: number of its tracks with d0/σ > 3
  Q.dc_1_n_lep             hardest prong: number of its electrons and muons
  Q.dc_1_z                 pT share of the hardest prong (0 if none)
  Q.dc_2_jp                2nd prong: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.dc_2_n_lep             2nd prong: number of its electrons and muons
  Q.dc_3_charge            3rd prong: Σ q √pT / √(Σ pT) of its particles
  Q.dc_3_z                 pT share of the 3rd prong (0 if none)
  Q.dc_4_z                 pT share of the 4th prong (0 if none)
  Q.dc_n                   number of prongs: reverse the C/A tree; ΔR ≤ 0.1 is a prong, a branch with < 10 % of the jet pT is dropped, else both branches are declustered
  Q.dc_split1_kt           kT = min(pT)·ΔR [GeV] of the hardest hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_split2_dr           ΔR of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_split2_mass         mass of the splitting node [GeV] of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dr12                   ΔR between particles 1 and 2
  Q.dr_0                   ΔR from the jet axis of particle 0 (ParT input; empty slot: 0)
  Q.dr_max_012             largest distance among the 3 hardest
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.e3                     energy correlation e3 (β=1, 32 hardest)
  Q.e3_b05                 energy correlation e3 with β = 0.5
  Q.e3_b2                  energy correlation e3 with β = 2
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 16 hardest)
  Q.e4_b2                  energy correlation e4 with β = 2
  Q.ecf_g41                generalized energy correlation ₁e₄ (β=1; products of the smallest angles)
  Q.ecf_g42                generalized energy correlation ₂e₄ (β=1; products of the smallest angles)
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.ismuon_0               1 if a muon of particle 0 (ParT input; empty slot: 0)
  Q.jd_3d_4                the 4th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_5                the 5th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_6                the 6th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_n_d3_pt1            number of tracks with |d0/σ| > 3 and pT > 1 GeV
  Q.jd_sum_abs_sd0_top3    sum of the 3 largest |d0/σ|
  Q.jd_sum_abs_sd0_top5    sum of the 5 largest |d0/σ|
  Q.jet_abs_eta            absolute pseudorapidity of the jet axis
  Q.jet_charge_k03         jet charge with κ = 0.3
  Q.jet_charge_k05         jet charge with κ = 0.5
  Q.kt2_1_n_lep            hardest kT subjet: number of its electrons and muons
  Q.kt2_2_n_lep            2nd kT subjet: number of its electrons and muons
  Q.kt2_2_z_disp3          2nd kT subjet: pT share (of the jet) of its tracks with d0/σ > 3
  Q.kt2_dr12               distance between the pT-weighted centres of the hardest and 2nd kT subjet (0 if missing)
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
  Q.lne_17                 ln E [GeV] of particle 17 (ParT input; empty slot: ln 1e-8)
  Q.lne_21                 ln E [GeV] of particle 21 (ParT input; empty slot: ln 1e-8)
  Q.lne_5                  ln E [GeV] of particle 5 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_1               ln(E / E of the jet) of particle 1 (ParT input; empty slot: ln 1e-8)
  Q.lnerel_78              ln(E / E of the jet) of particle 78 (ParT input; empty slot: ln 1e-8)
  Q.lund2_lndelta          ln Δ of the 2. primary C/A splitting (ln 1e-8 if none)
  Q.lund3_lndelta          ln Δ of the 3. primary C/A splitting (ln 1e-8 if none)
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
  Q.mres_sd_rg_b1z01       soft-drop R_g, β = 1, z_cut = 0.1
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
  Q.n_pairs_kt_above_3     number of pairs with kT = min(pTᵢ, pTⱼ)·ΔRᵢⱼ > 3 GeV
  Q.n_particles            number of real particles (pT > 0)
  Q.n_photon               number of photons
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.n_s3d_above_10         number of charged particles with 3d significance > 10
  Q.n_s3d_above_3          number of charged particles with 3d significance > 3
  Q.n_sd0_above_2          number of charged particles with d0 significance > 2
  Q.n_sd0_above_3          number of charged particles with d0 significance > 3
  Q.n_sd0_above_5          number of charged particles with d0 significance > 5
  Q.n_sdz_above_2          number of charged particles with dz significance > 2
  Q.n_sdz_above_5          number of charged particles with dz significance > 5
  Q.nca_sj4_pair_mass_2nd  second largest mass of two of the 4 subjets [GeV]
  Q.nca_sj4_pairmax_over_mass largest mass of two of the 4 subjets over the jet mass
  Q.pair_max_lnkt          largest ln kT among all pairs
  Q.pair_max_lnm2          largest ln m² among all pairs
  Q.pair_mean_lndelta      zᵢzⱼ-weighted mean of ln ΔRᵢⱼ over all pairs
  Q.pair_mean_lnkt         zᵢzⱼ-weighted mean of ln kT = ln(min(pTᵢ, pTⱼ)·ΔRᵢⱼ) over all pairs
  Q.pair_mean_lnm2         zᵢzⱼ-weighted mean of ln mᵢⱼ² (massless) over all pairs
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.pt1_over_pt0           pT1 / pT0
  Q.pz_lnd0                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -9 < ln ΔRᵢⱼ ≤ -3
  Q.pz_lnd2                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -2 < ln ΔRᵢⱼ ≤ -1
  Q.pz_lnd3                Σ zᵢzⱼ over the pairs of the 40 hardest particles with -1 < ln ΔRᵢⱼ ≤ 9
  Q.pz_lnkt0               Σ zᵢzⱼ over the pairs of the 40 hardest particles with -9 < ln kT ≤ 0
  Q.pz_lnkt1               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 0 < ln kT ≤ 1
  Q.pz_lnkt3               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 2 < ln kT ≤ 3
  Q.pz_lnkt4               Σ zᵢzⱼ over the pairs of the 40 hardest particles with 3 < ln kT ≤ 9
  Q.sd_mass                soft-drop groomed mass, C/A on the 128 hardest, β=0, z_cut=0.1 [GeV]
  Q.sdb_0_n                number of tracks with d0/σ ≤ -3
  Q.sdb_2_n                number of tracks with -1 < d0/σ ≤ 1
  Q.sdb_2_z                pT share of the tracks with -1 < d0/σ ≤ 1
  Q.sdb_4_z                pT share of the tracks with 3 < d0/σ ≤ 10
  Q.sdb_5_n                number of tracks with d0/σ > 10
  Q.sdb_5_z                pT share of the tracks with d0/σ > 10
  Q.sip_3d_1               the 1. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sip_3d_2               the 2. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sip_3d_3               the 3. largest 3D significance among the charged particles (√((d0/σ)² + (dz/σ)²)) (0 if fewer)
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_z1                 pT share of subjet 1 of 3 (by pT)
  Q.sj4_dr_min             smallest distance among the 4 subjet axes
  Q.sj4_pair_mass_max      largest mass of two of the 4 subjets [GeV]
  Q.sj4_zsoft              pT share of the softest of 4 subjets
  Q.sjf_2_1_max3d          subjet 1 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_1_n_d3           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_2_1_n_d5           subjet 1 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_2_1_z_d3           subjet 1 of 2 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_2_2_max3d          subjet 2 of 2 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_2_2_n_d3           subjet 2 of 2 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_2_n2disp           number of the 2 subjets with at least 2 tracks with |d0/σ| > 3
  Q.sjf_3_1_max3d          subjet 1 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_1_n_d3           subjet 1 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_3_3_n_d3           subjet 3 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_4_1_max3d          subjet 1 of 4 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_4_1_n_d3           subjet 1 of 4 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_4_1_z_d3           subjet 1 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_2_z_d3           subjet 2 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
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
  Q.sjq_3_sumabs_k1        |sum| of the charges (κ = 1) of the two hardest of 3 subjets
  Q.sum_e                  total energy of the particles [GeV]
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sv_1_dr                hardest displaced-track cluster: ΔR of its centre from the jet axis (0 if none)
  Q.sv_1_n                 hardest displaced-track cluster: number of tracks (0 if none)
  Q.sv_1_sd0_sum           hardest displaced-track cluster: Σ d0/σ of its tracks (0 if none)
  Q.sv_1_z                 hardest displaced-track cluster: pT share of the jet (0 if none)
  Q.sv_2_n                 2nd displaced-track cluster: number of tracks (0 if none)
  Q.sv_2_sd0_sum           2nd displaced-track cluster: Σ d0/σ of its tracks (0 if none)
  Q.sv_2_z                 2nd displaced-track cluster: pT share of the jet (0 if none)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau32_b2               N-subjettiness τ3/τ2 with β = 2
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.tau43                  N-subjettiness τ4/τ3
  Q.tau5                   N-subjettiness τ5 (β=1)
  Q.tau54                  N-subjettiness τ5/τ4
  Q.z_charged_had          pT share of charged hadrons
  Q.z_displaced3           pT share of the charged particles with |d0|/σ > 3
  Q.z_displaced5           pT share of the charged particles with |d0|/σ > 5
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.z_dr_0p4_up            pT share of the particles with ΔR ≥ 0.4
  Q.z_electron             pT share of electrons
  Q.z_muon                 pT share of muons
  Q.z_neutral              pT share of neutral particles
  Q.z_neutral_had          pT share of neutral hadrons
  Q.z_photon               pT share of photons
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
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M2_b2=ecfb('g31', 2) / max(ecfb('e2', 2), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        M3_b2=ecfb('g41', 2) / max(ecfb('g31', 2), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N2_b05=ecfb('g32', 0.5) / max(ecfb('e2', 0.5) ** 2, 1e-30),
        N2_b2=ecfb('g32', 2) / max(ecfb('e2', 2) ** 2, 1e-30),
        N3_b05=ecfb('g42', 0.5) / max(ecfb('g31', 0.5) ** 2, 1e-30),
        N3_b2=ecfb('g42', 2) / max(ecfb('g31', 2) ** 2, 1e-30),
        ak02_1_n_disp3=pq('ak02_1_n_disp3'),
        ak02_2_n_lep=pq('ak02_2_n_lep'),
        ak02_2_z=pq('ak02_2_z'),
        ak02_3_n_lep=pq('ak02_3_n_lep'),
        ak02_3_z=pq('ak02_3_z'),
        ak02_dr12=pq('ak02_dr12'),
        ak02_dr13=pq('ak02_dr13'),
        ak02_dr23=pq('ak02_dr23'),
        ak02_min12_n_disp3=pq('ak02_min12_n_disp3'),
        ak02_n=pq('ak02_n'),
        dc_1_n_disp3=pq('dc_1_n_disp3'),
        dc_1_n_lep=pq('dc_1_n_lep'),
        dc_1_z=pq('dc_1_z'),
        dc_2_jp=pq('dc_2_jp'),
        dc_2_n_lep=pq('dc_2_n_lep'),
        dc_3_charge=pq('dc_3_charge'),
        dc_3_z=pq('dc_3_z'),
        dc_4_z=pq('dc_4_z'),
        dc_n=pq('dc_n'),
        dc_split1_kt=pq('dc_split1_kt'),
        dc_split2_dr=pq('dc_split2_dr'),
        dc_split2_mass=pq('dc_split2_mass'),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        dr_0=pfeat(0, 'dr'),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        e3=ecf('e3'),
        e3_b05=ecfb('e3', 0.5),
        e3_b2=ecfb('e3', 2),
        e4=ecf('e4'),
        e4_b2=ecfb('e4', 2),
        ecf_g41=ecfb('g41', 1),
        ecf_g42=ecfb('g42', 1),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        ismuon_0=pfeat(0, 'ismuon'),
        jd_3d_4=pq('jd_3d_4'),
        jd_3d_5=pq('jd_3d_5'),
        jd_3d_6=pq('jd_3d_6'),
        jd_n_d3_pt1=pq('jd_n_d3_pt1'),
        jd_sum_abs_sd0_top3=pq('jd_sum_abs_sd0_top3'),
        jd_sum_abs_sd0_top5=pq('jd_sum_abs_sd0_top5'),
        jet_abs_eta=abs(jet_eta),
        jet_charge_k03=sum(charge[i] * z[i] ** 0.3 for i in real),
        jet_charge_k05=sum(charge[i] * z[i] ** 0.5 for i in real),
        kt2_1_n_lep=pq('kt2_1_n_lep'),
        kt2_2_n_lep=pq('kt2_2_n_lep'),
        kt2_2_z_disp3=pq('kt2_2_z_disp3'),
        kt2_dr12=pq('kt2_dr12'),
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
        lne_17=pfeat(17, 'lne'),
        lne_21=pfeat(21, 'lne'),
        lne_5=pfeat(5, 'lne'),
        lnerel_1=pfeat(1, 'lnerel'),
        lnerel_78=pfeat(78, 'lnerel'),
        lund2_lndelta=lund(2, 'lndelta'),
        lund3_lndelta=lund(3, 'lndelta'),
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
        mres_sd_rg_b1z01=pq('mres_sd_rg_b1z01'),
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
        n_pairs_kt_above_3=paircount(3),
        n_particles=len(real),
        n_photon=sum(1 for i in real if ptype[i] == 3),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        n_s3d_above_10=nsig('3d', 10),
        n_s3d_above_3=nsig('3d', 3),
        n_sd0_above_2=nsig('d0', 2),
        n_sd0_above_3=nsig('d0', 3),
        n_sd0_above_5=nsig('d0', 5),
        n_sdz_above_2=nsig('dz', 2),
        n_sdz_above_5=nsig('dz', 5),
        nca_sj4_pair_mass_2nd=pq('nca_sj4_pair_mass_2nd'),
        nca_sj4_pairmax_over_mass=pq('nca_sj4_pairmax_over_mass'),
        pair_max_lnkt=pairmax('lnkt'),
        pair_max_lnm2=pairmax('lnm2'),
        pair_mean_lndelta=pairsum('lndelta'),
        pair_mean_lnkt=pairsum('lnkt'),
        pair_mean_lnm2=pairsum('lnm2'),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        pz_lnd0=pq('pz_lnd0'),
        pz_lnd2=pq('pz_lnd2'),
        pz_lnd3=pq('pz_lnd3'),
        pz_lnkt0=pq('pz_lnkt0'),
        pz_lnkt1=pq('pz_lnkt1'),
        pz_lnkt3=pq('pz_lnkt3'),
        pz_lnkt4=pq('pz_lnkt4'),
        sd_mass=softdrop("mass"),
        sdb_0_n=pq('sdb_0_n'),
        sdb_2_n=pq('sdb_2_n'),
        sdb_2_z=pq('sdb_2_z'),
        sdb_4_z=pq('sdb_4_z'),
        sdb_5_n=pq('sdb_5_n'),
        sdb_5_z=pq('sdb_5_z'),
        sip_3d_1=sip('3d', 1),
        sip_3d_2=sip('3d', 2),
        sip_3d_3=sip('3d', 3),
        sj2_dr=subjets(2)["dr"][0],
        sj2_mass2=subjets(2)["mass"][1],
        sj2_zsoft=subjets(2)["z"][1],
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr_max=max(subjets(3)["dr"]),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_z1=subjets(3)["z"][0],
        sj4_dr_min=min(subjets(4)["dr"]),
        sj4_pair_mass_max=max(subjets(4)["mpair"]),
        sj4_zsoft=subjets(4)["z"][3],
        sjf_2_1_max3d=pq('sjf_2_1_max3d'),
        sjf_2_1_n_d3=pq('sjf_2_1_n_d3'),
        sjf_2_1_n_d5=pq('sjf_2_1_n_d5'),
        sjf_2_1_z_d3=pq('sjf_2_1_z_d3'),
        sjf_2_2_max3d=pq('sjf_2_2_max3d'),
        sjf_2_2_n_d3=pq('sjf_2_2_n_d3'),
        sjf_2_n2disp=pq('sjf_2_n2disp'),
        sjf_3_1_max3d=pq('sjf_3_1_max3d'),
        sjf_3_1_n_d3=pq('sjf_3_1_n_d3'),
        sjf_3_3_n_d3=pq('sjf_3_3_n_d3'),
        sjf_4_1_max3d=pq('sjf_4_1_max3d'),
        sjf_4_1_n_d3=pq('sjf_4_1_n_d3'),
        sjf_4_1_z_d3=pq('sjf_4_1_z_d3'),
        sjf_4_2_z_d3=pq('sjf_4_2_z_d3'),
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
        sjq_3_sumabs_k1=pq('sjq_3_sumabs_k1'),
        sum_e=sum(energy[i] for i in real),
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sv_1_dr=pq('sv_1_dr'),
        sv_1_n=pq('sv_1_n'),
        sv_1_sd0_sum=pq('sv_1_sd0_sum'),
        sv_1_z=pq('sv_1_z'),
        sv_2_n=pq('sv_2_n'),
        sv_2_sd0_sum=pq('sv_2_sd0_sum'),
        sv_2_z=pq('sv_2_z'),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau32_b2=tau_n(3, 2) / max(tau_n(2, 2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        tau5=tau_n(5),
        tau54=tau_n(5) / max(tau_n(4), 1e-12),
        z_charged_had=sum(z[i] for i in real if ptype[i] == 1),
        z_displaced3=displaced(3, 'z'),
        z_displaced5=displaced(5, 'z'),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        z_dr_0p4_up=sum(z[i] for i in real if 0.4 <= dr[i] < 10),
        z_electron=sum(z[i] for i in real if ptype[i] == 4),
        z_muon=sum(z[i] for i in real if ptype[i] == 5),
        z_neutral=sum(z[i] for i in real if charge[i] == 0),
        z_neutral_had=sum(z[i] for i in real if ptype[i] == 2),
        z_photon=sum(z[i] for i in real if ptype[i] == 3),
        z_top30_slots=sum(pt[:30]) / tot,
    )


def neuron_0(Q):
    z = 0.00521
    return z


def neuron_1(Q):
    z = -5.32e-06
    return z


def neuron_2(Q):
    z = 5.45e-06
    return z


def neuron_3(Q):
    z = 8.39e-06
    return z


def neuron_4(Q):
    z = 2.79e-06
    return z


def neuron_5(Q):
    z = -2.16e-06
    return z


def neuron_6(Q):
    z = 1.77e-05
    return z


def neuron_7(Q):
    z = 6.43e-06
    return z


def neuron_8(Q):
    z = 2.92e-06
    return z


def neuron_9(Q):
    z = -2.44e-06
    return z


def neuron_10(Q):
    z = -1.57e-05
    return z


def neuron_11(Q):
    z = -1.89e-05
    return z


def neuron_12(Q):
    z = 5.1e-06
    return z


def neuron_13(Q):
    z = 4.42e-06
    return z


def neuron_14(Q):
    z = 2.84e-06
    return z


def neuron_15(Q):
    z = -6.45e-06
    return z


def neuron_16(Q):
    z = -0.987
    z += 0.0901 * Q.lep_ptrel
    z += 0.00724 * Q.n_photon
    z += -0.672 * (1.0 * 0.5 * ((0.00438 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.00438 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.0375 * (4.214028 * 0.5 * ((10.1 - Q.jd_3d_4) / 4.214028) * (1 + math.erf(((10.1 - Q.jd_3d_4) / 4.214028) / math.sqrt(2))))
    z += -0.0108 * (17.91682 * 0.5 * ((33.6 - Q.jd_3d_4) / 17.91682) * (1 + math.erf(((33.6 - Q.jd_3d_4) / 17.91682) / math.sqrt(2))))
    z += -0.000844 * (14.45336 * 0.5 * ((Q.jd_3d_6 - 30.9) / 14.45336) * (1 + math.erf(((Q.jd_3d_6 - 30.9) / 14.45336) / math.sqrt(2))))
    z += -0.0135 * (146.0039 * 0.5 * ((263.0 - Q.jd_sum_abs_sd0_top3) / 146.0039) * (1 + math.erf(((263.0 - Q.jd_sum_abs_sd0_top3) / 146.0039) / math.sqrt(2))))
    z += 0.0104 * (148.6964 * 0.5 * ((289.0 - Q.jd_sum_abs_sd0_top5) / 148.6964) * (1 + math.erf(((289.0 - Q.jd_sum_abs_sd0_top5) / 148.6964) / math.sqrt(2))))
    z += -0.221 * (1.0 * 0.5 * ((1.0 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.0798 * (15.88428 * 0.5 * ((Q.lep_ptrel - 43.5) / 15.88428) * (1 + math.erf(((Q.lep_ptrel - 43.5) / 15.88428) / math.sqrt(2))))
    z += 0.0741 * (12.22004 * 0.5 * ((27.4 - Q.lep_ptrel) / 12.22004) * (1 + math.erf(((27.4 - Q.lep_ptrel) / 12.22004) / math.sqrt(2))))
    z += -4.53 * (0.07843207 * 0.5 * ((Q.lep_z - 0.13) / 0.07843207) * (1 + math.erf(((Q.lep_z - 0.13) / 0.07843207) / math.sqrt(2))))
    z += -0.0578 * (1.5 * 0.5 * ((Q.lepsj_2_n_d3 - 2.18) / 1.5) * (1 + math.erf(((Q.lepsj_2_n_d3 - 2.18) / 1.5) / math.sqrt(2))))
    z += -2.64 * (0.03445022 * 0.5 * ((0.0493 - Q.lepsj_3_dr) / 0.03445022) * (1 + math.erf(((0.0493 - Q.lepsj_3_dr) / 0.03445022) / math.sqrt(2))))
    z += 0.0161 * (3.043206 * 0.5 * ((56.2 - Q.mass_charged) / 3.043206) * (1 + math.erf(((56.2 - Q.mass_charged) / 3.043206) / math.sqrt(2))))
    z += -0.22 * (0.560057 * 0.5 * ((1.78 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.78 - Q.mass_displaced5) / 0.560057) / math.sqrt(2))))
    z += 0.00478 * (3.20498 * 0.5 * ((Q.mass_neutral - 34.8) / 3.20498) * (1 + math.erf(((Q.mass_neutral - 34.8) / 3.20498) / math.sqrt(2))))
    z += 0.01 * (3.485807 * 0.5 * ((Q.mass_top40 - 118.0) / 3.485807) * (1 + math.erf(((Q.mass_top40 - 118.0) / 3.485807) / math.sqrt(2))))
    z += 0.00649 * (5.119423 * 0.5 * ((Q.mass_top5 - 63.7) / 5.119423) * (1 + math.erf(((Q.mass_top5 - 63.7) / 5.119423) / math.sqrt(2))))
    z += 0.0725 * (3.705078 * 0.5 * ((5.79 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.79 - Q.max_abs_d0) / 3.705078) / math.sqrt(2))))
    z += -0.588 * (0.1750793 * 0.5 * ((0.452 - Q.max_abs_dz) / 0.1750793) * (1 + math.erf(((0.452 - Q.max_abs_dz) / 0.1750793) / math.sqrt(2))))
    z += 0.0125 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 180.0) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 180.0) / 18.97147) / math.sqrt(2))))
    z += -0.0207 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 156.0) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 156.0) / 19.14168) / math.sqrt(2))))
    z += 0.00971 * (1.5 * 0.5 * ((Q.n_dr_0p2_0p4 - 9.34) / 1.5) * (1 + math.erf(((Q.n_dr_0p2_0p4 - 9.34) / 1.5) / math.sqrt(2))))
    z += -0.0459 * (1.0 * 0.5 * ((4.0 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.0 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2))))
    z += -0.0101 * (3.5 * 0.5 * ((15.0 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((15.0 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2))))
    z += 0.0118 * (21.5 * 0.5 * ((80.8 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((80.8 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += -0.016 * (6.0 * 0.5 * ((28.1 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.1 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2))))
    z += 0.0228 * (6.5 * 0.5 * ((41.2 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((41.2 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2))))
    z += -0.196 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += -1.63 * (1.0 * 0.5 * ((-0.0551 - Q.n_sd0_above_5) / 1.0) * (1 + math.erf(((-0.0551 - Q.n_sd0_above_5) / 1.0) / math.sqrt(2))))
    z += 3.08 * (0.01578816 * 0.5 * ((0.0834 - Q.pz_lnd2) / 0.01578816) * (1 + math.erf(((0.0834 - Q.pz_lnd2) / 0.01578816) / math.sqrt(2))))
    z += 0.0381 * (1.0 * 0.5 * ((Q.sdb_0_n - 3.09) / 1.0) * (1 + math.erf(((Q.sdb_0_n - 3.09) / 1.0) / math.sqrt(2))))
    z += -0.0131 * (1.0 * 0.5 * ((Q.sdb_2_n - 5.75) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 5.75) / 1.0) / math.sqrt(2))))
    z += 0.848 * (0.004067431 * 0.5 * ((Q.sdb_5_z - 0.0135) / 0.004067431) * (1 + math.erf(((Q.sdb_5_z - 0.0135) / 0.004067431) / math.sqrt(2))))
    z += 0.00234 * (112.8732 * 0.5 * ((207.0 - Q.sip_3d_1) / 112.8732) * (1 + math.erf(((207.0 - Q.sip_3d_1) / 112.8732) / math.sqrt(2))))
    z += 0.00198 * (164.9991 * 0.5 * ((228.0 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((228.0 - Q.sip_3d_2) / 164.9991) / math.sqrt(2))))
    z += 0.00124 * (248.3135 * 0.5 * ((178.0 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((178.0 - Q.sip_3d_3) / 248.3135) / math.sqrt(2))))
    z += -0.0316 * (1.118837 * 0.5 * ((6.58 - Q.sj2_mass2) / 1.118837) * (1 + math.erf(((6.58 - Q.sj2_mass2) / 1.118837) / math.sqrt(2))))
    z += 0.0792 * (1.0 * 0.5 * ((3.65 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((3.65 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2))))
    z += 2.11 * (0.0439522 * 0.5 * ((0.0968 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.0968 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2))))
    z += 0.524 * (0.02907957 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.212) / 0.02907957) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.212) / 0.02907957) / math.sqrt(2))))
    z += 1.59 * (0.03043217 * 0.5 * ((0.111 - Q.sv_1_z) / 0.03043217) * (1 + math.erf(((0.111 - Q.sv_1_z) / 0.03043217) / math.sqrt(2))))
    z += 0.828 * (1.0 * 0.5 * ((0.00403 - Q.sv_2_n) / 1.0) * (1 + math.erf(((0.00403 - Q.sv_2_n) / 1.0) / math.sqrt(2))))
    z += -4.4 * (0.09329774 * 0.5 * ((Q.z_charged_had - 0.72) / 0.09329774) * (1 + math.erf(((Q.z_charged_had - 0.72) / 0.09329774) / math.sqrt(2))))
    z += 4.47 * (0.09329774 * 0.5 * ((0.723 - Q.z_charged_had) / 0.09329774) * (1 + math.erf(((0.723 - Q.z_charged_had) / 0.09329774) / math.sqrt(2))))
    z += -5.04 * (0.08093679 * 0.5 * ((0.281 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.281 - Q.z_displaced5) / 0.08093679) / math.sqrt(2))))
    z += 7.14 * (0.008519048 * 0.5 * ((0.0152 - Q.z_dr_0p05_0p1) / 0.008519048) * (1 + math.erf(((0.0152 - Q.z_dr_0p05_0p1) / 0.008519048) / math.sqrt(2))))
    z += -3.78 * (0.05422308 * 0.5 * ((Q.z_neutral - 0.139) / 0.05422308) * (1 + math.erf(((Q.z_neutral - 0.139) / 0.05422308) / math.sqrt(2))))
    z += -0.00907 * (1.0 * 0.5 * ((3.03 - Q.dc_1_n_disp3) / 1.0) * (1 + math.erf(((3.03 - Q.dc_1_n_disp3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_pt_above_50 - 1.07) / 1.0) * (1 + math.erf(((Q.n_pt_above_50 - 1.07) / 1.0) / math.sqrt(2))))
    z += 6.11 * (1.0 * 0.5 * ((Q.dc_n - 0.986) / 1.0) * (1 + math.erf(((Q.dc_n - 0.986) / 1.0) / math.sqrt(2)))) * (0.003546451 * 0.5 * ((0.0419 - Q.M2_b2) / 0.003546451) * (1 + math.erf(((0.0419 - Q.M2_b2) / 0.003546451) / math.sqrt(2))))
    z += -0.0182 * (1.0 * 0.5 * ((Q.dc_n - 1.03) / 1.0) * (1 + math.erf(((Q.dc_n - 1.03) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((18.0 - Q.n_photon) / 1.0) * (1 + math.erf(((18.0 - Q.n_photon) / 1.0) / math.sqrt(2))))
    z += 27200.0 * (0.0002686389 * 0.5 * ((0.00041 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.00041 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.001717815 * 0.5 * ((Q.M3 - 0.0252) / 0.001717815) * (1 + math.erf(((Q.M3 - 0.0252) / 0.001717815) / math.sqrt(2))))
    z += -1770.0 * (0.0002686389 * 0.5 * ((0.000411 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000411 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.04590778 * 0.5 * ((0.19 - Q.dr_max_012) / 0.04590778) * (1 + math.erf(((0.19 - Q.dr_max_012) / 0.04590778) / math.sqrt(2))))
    z += 56.7 * (0.0002686389 * 0.5 * ((0.000414 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000414 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.5 * 0.5 * ((Q.n_lund - 7.08) / 1.5) * (1 + math.erf(((Q.n_lund - 7.08) / 1.5) / math.sqrt(2))))
    z += 0.0191 * (1.0 * 0.5 * ((0.991 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.991 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2)))) * (5.497214 * 0.5 * ((8.21 - Q.sjf_4_1_max3d) / 5.497214) * (1 + math.erf(((8.21 - Q.sjf_4_1_max3d) / 5.497214) / math.sqrt(2))))
    z += -0.00382 * (3.20498 * 0.5 * ((Q.mass_neutral - 34.5) / 3.20498) * (1 + math.erf(((Q.mass_neutral - 34.5) / 3.20498) / math.sqrt(2)))) * (0.203947 * 0.5 * ((2.37 - Q.jd_3d_6) / 0.203947) * (1 + math.erf(((2.37 - Q.jd_3d_6) / 0.203947) / math.sqrt(2))))
    z += 0.0794 * (5.55459 * 0.5 * ((98.2 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((98.2 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) * (0.03294892 * 0.5 * ((0.0872 - Q.kt2_2_z_disp3) / 0.03294892) * (1 + math.erf(((0.0872 - Q.kt2_2_z_disp3) / 0.03294892) / math.sqrt(2))))
    z += 0.053 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 0.989) / 1.0) * (1 + math.erf(((Q.n_muon - 0.989) / 1.0) / math.sqrt(2))))
    z += -0.0149 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.2798636 * 0.5 * ((1.71 - Q.sjq_2_sumabs_k03) / 0.2798636) * (1 + math.erf(((1.71 - Q.sjq_2_sumabs_k03) / 0.2798636) / math.sqrt(2))))
    z += 0.0583 * (1.0 * 0.5 * ((9.98 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((9.98 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sv_1_n - 1.01) / 1.0) * (1 + math.erf(((Q.sv_1_n - 1.01) / 1.0) / math.sqrt(2))))
    z += -0.179 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.151853 * 0.5 * ((0.139 - Q.z_electron) / 0.151853) * (1 + math.erf(((0.139 - Q.z_electron) / 0.151853) / math.sqrt(2))))
    z += -0.254 * (0.0439522 * 0.5 * ((0.0995 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.0995 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2)))) * (1.0 * 0.5 * ((4.99 - Q.sjf_4_1_n_d3) / 1.0) * (1 + math.erf(((4.99 - Q.sjf_4_1_n_d3) / 1.0) / math.sqrt(2))))
    z += -0.0977 * (0.02907957 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.22) / 0.02907957) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.22) / 0.02907957) / math.sqrt(2)))) * (2.093785 * 0.5 * ((4.46 - Q.sip_3d_3) / 2.093785) * (1 + math.erf(((4.46 - Q.sip_3d_3) / 2.093785) / math.sqrt(2))))
    z += 0.0276 * (0.08093679 * 0.5 * ((0.28 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.28 - Q.z_displaced5) / 0.08093679) / math.sqrt(2)))) * (115.0698 * 0.5 * ((172.0 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((172.0 - Q.jd_3d_4) / 115.0698) / math.sqrt(2))))
    z += 0.364 * (0.08093679 * 0.5 * ((0.282 - Q.z_displaced5) / 0.08093679) * (1 + math.erf(((0.282 - Q.z_displaced5) / 0.08093679) / math.sqrt(2)))) * (0.1240004 * 0.5 * ((7.09 - Q.pair_max_lnm2) / 0.1240004) * (1 + math.erf(((7.09 - Q.pair_max_lnm2) / 0.1240004) / math.sqrt(2))))
    z += -0.123 * (0.06782196 * 0.5 * ((0.0379 - Q.z_muon) / 0.06782196) * (1 + math.erf(((0.0379 - Q.z_muon) / 0.06782196) / math.sqrt(2)))) * (0.7263644 * 0.5 * ((Q.mres_sd_prong_mass2 - -0.105) / 0.7263644) * (1 + math.erf(((Q.mres_sd_prong_mass2 - -0.105) / 0.7263644) / math.sqrt(2))))
    return z


def neuron_17(Q):
    z = -1.04e-05
    return z


def neuron_18(Q):
    z = 0.576
    z += -3.5 * (0.01539698 * 0.5 * ((Q.N2 - 0.396) / 0.01539698) * (1 + math.erf(((Q.N2 - 0.396) / 0.01539698) / math.sqrt(2))))
    z += 3.74 * (0.008548367 * 0.5 * ((Q.N2_b05 - 0.435) / 0.008548367) * (1 + math.erf(((Q.N2_b05 - 0.435) / 0.008548367) / math.sqrt(2))))
    z += -35.8 * (0.001754707 * 0.5 * ((0.00932 - Q.sum_z_dr2_top2) / 0.001754707) * (1 + math.erf(((0.00932 - Q.sum_z_dr2_top2) / 0.001754707) / math.sqrt(2))))
    z += -735.0 * (0.001001007 * 0.5 * ((0.000643 - Q.sum_z_dr2_top3) / 0.001001007) * (1 + math.erf(((0.000643 - Q.sum_z_dr2_top3) / 0.001001007) / math.sqrt(2))))
    z += 0.0613 * (1.0 * 0.5 * ((8.97 - Q.jd_n_d3_pt1) / 1.0) * (1 + math.erf(((8.97 - Q.jd_n_d3_pt1) / 1.0) / math.sqrt(2))))
    z += 0.0771 * (0.1783594 * 0.5 * ((Q.ktd_ln_d34 - -9.17) / 0.1783594) * (1 + math.erf(((Q.ktd_ln_d34 - -9.17) / 0.1783594) / math.sqrt(2))))
    z += -0.724 * (0.117671 * 0.5 * ((Q.lep_dr - 0.428) / 0.117671) * (1 + math.erf(((Q.lep_dr - 0.428) / 0.117671) / math.sqrt(2))))
    z += 0.298 * (1.234622 * 0.5 * ((1.38 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.38 - Q.lep_iso) / 1.234622) / math.sqrt(2))))
    z += -161.0 * (0.006229165 * 0.5 * ((0.0041 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.0041 - Q.lep_z) / 0.006229165) / math.sqrt(2))))
    z += -2.8 * (0.1060766 * 0.5 * ((0.221 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221 - Q.lep_z) / 0.1060766) / math.sqrt(2))))
    z += 0.237 * (5.001896 * 0.5 * ((2.45 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.45 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2))))
    z += -0.000828 * (540.2081 * 0.5 * ((586.0 - Q.lepsj_3_maxsd0) / 540.2081) * (1 + math.erf(((586.0 - Q.lepsj_3_maxsd0) / 540.2081) / math.sqrt(2))))
    z += 0.213 * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2))))
    z += -0.0888 * (0.3893833 * 0.5 * ((Q.lund3_lndelta - -2.81) / 0.3893833) * (1 + math.erf(((Q.lund3_lndelta - -2.81) / 0.3893833) / math.sqrt(2))))
    z += -0.018 * (4.146938 * 0.5 * ((Q.mass - 110.0) / 4.146938) * (1 + math.erf(((Q.mass - 110.0) / 4.146938) / math.sqrt(2))))
    z += 0.0285 * (16.90428 * 0.5 * ((Q.mass - 165.0) / 16.90428) * (1 + math.erf(((Q.mass - 165.0) / 16.90428) / math.sqrt(2))))
    z += -0.00754 * (5.97473 * 0.5 * ((115.0 - Q.mass_top20) / 5.97473) * (1 + math.erf(((115.0 - Q.mass_top20) / 5.97473) / math.sqrt(2))))
    z += 0.0105 * (14.89455 * 0.5 * ((146.0 - Q.mass_top20) / 14.89455) * (1 + math.erf(((146.0 - Q.mass_top20) / 14.89455) / math.sqrt(2))))
    z += 0.0168 * (5.898438 * 0.5 * ((9.51 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((9.51 - Q.max_abs_d0) / 5.898438) / math.sqrt(2))))
    z += -0.0106 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 179.0) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 179.0) / 18.97147) / math.sqrt(2))))
    z += 0.00666 * (5.55459 * 0.5 * ((Q.mres_sd_mass_b2z01 - 96.9) / 5.55459) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 96.9) / 5.55459) / math.sqrt(2))))
    z += -0.0136 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0) / 6.991655) / math.sqrt(2))))
    z += 0.0207 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 158.0) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 158.0) / 19.14168) / math.sqrt(2))))
    z += 0.045 * (1.0 * 0.5 * ((4.08 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.08 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2))))
    z += -0.0144 * (2.0 * 0.5 * ((27.1 - Q.n_for_90pct) / 2.0) * (1 + math.erf(((27.1 - Q.n_for_90pct) / 2.0) / math.sqrt(2))))
    z += 0.326 * (1.0 * 0.5 * ((Q.n_lepton - 0.94) / 1.0) * (1 + math.erf(((Q.n_lepton - 0.94) / 1.0) / math.sqrt(2))))
    z += -4.35 * (1.0 * 0.5 * ((0.00941 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.00941 - Q.n_lepton) / 1.0) / math.sqrt(2))))
    z += -0.0221 * (2.5 * 0.5 * ((Q.n_particles - 20.9) / 2.5) * (1 + math.erf(((Q.n_particles - 20.9) / 2.5) / math.sqrt(2))))
    z += -0.183 * (1.0 * 0.5 * ((3.03 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((3.03 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += -0.0627 * (1.0 * 0.5 * ((10.0 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.0 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 0.0986 * (1.0 * 0.5 * ((2.01 - Q.n_sdz_above_5) / 1.0) * (1 + math.erf(((2.01 - Q.n_sdz_above_5) / 1.0) / math.sqrt(2))))
    z += 0.11 * (1.268269 * 0.5 * ((5.13 - Q.sj2_mass2) / 1.268269) * (1 + math.erf(((5.13 - Q.sj2_mass2) / 1.268269) / math.sqrt(2))))
    z += -0.0165 * (11.17405 * 0.5 * ((Q.sj3_pair_mass_max - 129.0) / 11.17405) * (1 + math.erf(((Q.sj3_pair_mass_max - 129.0) / 11.17405) / math.sqrt(2))))
    z += 340.0 * (3.069852 * 0.5 * ((-11.1 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((-11.1 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2))))
    z += 0.318 * (0.1763298 * 0.5 * ((2.06 - Q.sjf_3_1_max3d) / 0.1763298) * (1 + math.erf(((2.06 - Q.sjf_3_1_max3d) / 0.1763298) / math.sqrt(2))))
    z += -0.298 * (2.463307 * 0.5 * ((3.59 - Q.sjf_3_1_max3d) / 2.463307) * (1 + math.erf(((3.59 - Q.sjf_3_1_max3d) / 2.463307) / math.sqrt(2))))
    z += 0.0859 * (1.0 * 0.5 * ((1.96 - Q.sjq_3_3_nch) / 1.0) * (1 + math.erf(((1.96 - Q.sjq_3_3_nch) / 1.0) / math.sqrt(2))))
    z += 0.00301 * (20.79165 * 0.5 * ((57.3 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((57.3 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2))))
    z += 14.3 * (0.005001016 * 0.5 * ((0.0503 - Q.tau5) / 0.005001016) * (1 + math.erf(((0.0503 - Q.tau5) / 0.005001016) / math.sqrt(2))))
    z += 1.56 * (0.03770036 * 0.5 * ((0.359 - Q.z_charged_had) / 0.03770036) * (1 + math.erf(((0.359 - Q.z_charged_had) / 0.03770036) / math.sqrt(2))))
    z += -15.9 * (0.01141967 * 0.5 * ((0.0353 - Q.z_displaced3) / 0.01141967) * (1 + math.erf(((0.0353 - Q.z_displaced3) / 0.01141967) / math.sqrt(2))))
    z += -0.499 * (0.005299632 * 0.5 * ((Q.z_displaced5 - 0.0108) / 0.005299632) * (1 + math.erf(((Q.z_displaced5 - 0.0108) / 0.005299632) / math.sqrt(2))))
    z += 6.06 * (0.01670814 * 0.5 * ((0.0629 - Q.z_displaced5) / 0.01670814) * (1 + math.erf(((0.0629 - Q.z_displaced5) / 0.01670814) / math.sqrt(2))))
    z += -0.0867 * (1.234622 * 0.5 * ((1.35 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.35 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (0.4913027 * 0.5 * ((-5.15 - Q.ktd_ln_d23) / 0.4913027) * (1 + math.erf(((-5.15 - Q.ktd_ln_d23) / 0.4913027) / math.sqrt(2))))
    z += 17.8 * (0.1060766 * 0.5 * ((0.222 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.222 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.01348041 * 0.5 * ((0.383 - Q.N2) / 0.01348041) * (1 + math.erf(((0.383 - Q.N2) / 0.01348041) / math.sqrt(2))))
    z += -4.01 * (0.1060766 * 0.5 * ((0.22 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.22 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.03286901 * 0.5 * ((0.484 - Q.ak02_dr12) / 0.03286901) * (1 + math.erf(((0.484 - Q.ak02_dr12) / 0.03286901) / math.sqrt(2))))
    z += -8.44 * (0.1060766 * 0.5 * ((0.222 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.222 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.02343699 * 0.5 * ((Q.sdb_2_z - 0.225) / 0.02343699) * (1 + math.erf(((Q.sdb_2_z - 0.225) / 0.02343699) / math.sqrt(2))))
    z += 0.537 * (0.006229165 * 0.5 * ((0.00471 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.00471 - Q.lep_z) / 0.006229165) / math.sqrt(2)))) * (117.3924 * 0.5 * ((127.0 - Q.sjf_3_1_max3d) / 117.3924) * (1 + math.erf(((127.0 - Q.sjf_3_1_max3d) / 117.3924) / math.sqrt(2))))
    z += -27.0 * (0.1060766 * 0.5 * ((0.221 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.221 - Q.lep_z) / 0.1060766) / math.sqrt(2)))) * (0.02653106 * 0.5 * ((0.104 - Q.z_displaced5) / 0.02653106) * (1 + math.erf(((0.104 - Q.z_displaced5) / 0.02653106) / math.sqrt(2))))
    z += 0.95 * (1.0 * 0.5 * ((3.0 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.0 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.04808774 * 0.5 * ((0.0987 - Q.lepsj_3_dr) / 0.04808774) * (1 + math.erf(((0.0987 - Q.lepsj_3_dr) / 0.04808774) / math.sqrt(2))))
    z += 0.0146 * (4.85476 * 0.5 * ((13.0 - Q.mass_displaced3) / 4.85476) * (1 + math.erf(((13.0 - Q.mass_displaced3) / 4.85476) / math.sqrt(2)))) * (0.06729888 * 0.5 * ((Q.jet_abs_eta - 0.413) / 0.06729888) * (1 + math.erf(((Q.jet_abs_eta - 0.413) / 0.06729888) / math.sqrt(2))))
    z += -0.00151 * (4.85476 * 0.5 * ((13.3 - Q.mass_displaced3) / 4.85476) * (1 + math.erf(((13.3 - Q.mass_displaced3) / 4.85476) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 11.6) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 11.6) / 1.0) / math.sqrt(2))))
    z += -0.0698 * (19.14168 * 0.5 * ((Q.mres_sd_mass_b2z01 - 157.0) / 19.14168) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 157.0) / 19.14168) / math.sqrt(2)))) * (0.04048364 * 0.5 * ((0.274 - Q.ak02_dr23) / 0.04048364) * (1 + math.erf(((0.274 - Q.ak02_dr23) / 0.04048364) / math.sqrt(2))))
    z += 0.228 * (1.5 * 0.5 * ((5.97 - Q.n_s3d_above_10) / 1.5) * (1 + math.erf(((5.97 - Q.n_s3d_above_10) / 1.5) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.238 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.238 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2))))
    z += -0.00254 * (20.79165 * 0.5 * ((52.8 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((52.8 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_1_n_lep - 0.000424) / 1.0) * (1 + math.erf(((Q.dc_1_n_lep - 0.000424) / 1.0) / math.sqrt(2))))
    z += -58.1 * (0.005001016 * 0.5 * ((0.0503 - Q.tau5) / 0.005001016) * (1 + math.erf(((0.0503 - Q.tau5) / 0.005001016) / math.sqrt(2)))) * (0.01243073 * 0.5 * ((Q.ak02_3_z - 0.017) / 0.01243073) * (1 + math.erf(((Q.ak02_3_z - 0.017) / 0.01243073) / math.sqrt(2))))
    z += 47.8 * (0.005299632 * 0.5 * ((Q.z_displaced5 - -0.000394) / 0.005299632) * (1 + math.erf(((Q.z_displaced5 - -0.000394) / 0.005299632) / math.sqrt(2)))) * (0.003791252 * 0.5 * ((0.0194 - Q.dr12) / 0.003791252) * (1 + math.erf(((0.0194 - Q.dr12) / 0.003791252) / math.sqrt(2))))
    return z


def neuron_19(Q):
    z = -1.41e-06
    return z


def neuron_20(Q):
    z = 8.21e-07
    return z


def neuron_21(Q):
    z = -0.00168
    return z


def neuron_22(Q):
    z = -1.05e-06
    return z


def neuron_23(Q):
    z = -6.98e-07
    return z


def neuron_24(Q):
    z = 0.885
    z += -84.2 * (0.002611265 * 0.5 * ((0.0105 - Q.M2_b2) / 0.002611265) * (1 + math.erf(((0.0105 - Q.M2_b2) / 0.002611265) / math.sqrt(2))))
    z += 5.44 * (0.001965812 * 0.5 * ((Q.M3 - 0.0324) / 0.001965812) * (1 + math.erf(((Q.M3 - 0.0324) / 0.001965812) / math.sqrt(2))))
    z += 0.563 * (0.1026619 * 0.5 * ((0.919 - Q.N3_b05) / 0.1026619) * (1 + math.erf(((0.919 - Q.N3_b05) / 0.1026619) / math.sqrt(2))))
    z += -0.461 * (0.071994 * 0.5 * ((0.458 - Q.N3_b2) / 0.071994) * (1 + math.erf(((0.458 - Q.N3_b2) / 0.071994) / math.sqrt(2))))
    z += 0.0241 * (3.24083 * 0.5 * ((6.92 - Q.dc_2_jp) / 3.24083) * (1 + math.erf(((6.92 - Q.dc_2_jp) / 3.24083) / math.sqrt(2))))
    z += 3.92 * (0.06488792 * 0.5 * ((Q.dc_split2_dr - 0.279) / 0.06488792) * (1 + math.erf(((Q.dc_split2_dr - 0.279) / 0.06488792) / math.sqrt(2))))
    z += -5.47 * (0.07842787 * 0.5 * ((Q.dc_split2_dr - 0.354) / 0.07842787) * (1 + math.erf(((Q.dc_split2_dr - 0.354) / 0.07842787) / math.sqrt(2))))
    z += -738.0 * (0.0002686389 * 0.5 * ((0.000412 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000412 - Q.e3_b2) / 0.0002686389) / math.sqrt(2))))
    z += 1.56 * (0.6671377 * 0.5 * ((0.448 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.448 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += -20.6 * (0.01304025 * 0.5 * ((0.0126 - Q.lep_z) / 0.01304025) * (1 + math.erf(((0.0126 - Q.lep_z) / 0.01304025) / math.sqrt(2))))
    z += -2.45 * (0.1486471 * 0.5 * ((0.34 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.34 - Q.lep_z) / 0.1486471) / math.sqrt(2))))
    z += -7.43 * (0.05826336 * 0.5 * ((0.103 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.103 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2))))
    z += 0.266 * (5.001896 * 0.5 * ((2.42 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.42 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2))))
    z += 0.016 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2))))
    z += -0.0126 * (18.42178 * 0.5 * ((183.0 - Q.mass) / 18.42178) * (1 + math.erf(((183.0 - Q.mass) / 18.42178) / math.sqrt(2))))
    z += -0.00971 * (0.7154015 * 0.5 * ((Q.mass_displaced3 - 1.76) / 0.7154015) * (1 + math.erf(((Q.mass_displaced3 - 1.76) / 0.7154015) / math.sqrt(2))))
    z += 0.0156 * (6.875137 * 0.5 * ((18.6 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.6 - Q.mass_displaced3) / 6.875137) / math.sqrt(2))))
    z += 0.0692 * (0.560057 * 0.5 * ((1.77 - Q.mass_displaced5) / 0.560057) * (1 + math.erf(((1.77 - Q.mass_displaced5) / 0.560057) / math.sqrt(2))))
    z += -0.0144 * (6.170706 * 0.5 * ((76.1 - Q.mass_top30) / 6.170706) * (1 + math.erf(((76.1 - Q.mass_top30) / 6.170706) / math.sqrt(2))))
    z += -0.0065 * (6.785484 * 0.5 * ((Q.mass_top5 - 67.7) / 6.785484) * (1 + math.erf(((Q.mass_top5 - 67.7) / 6.785484) / math.sqrt(2))))
    z += 0.00845 * (8.537109 * 0.5 * ((18.2 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((18.2 - Q.max_abs_d0) / 8.537109) / math.sqrt(2))))
    z += -0.0226 * (5.270719 * 0.5 * ((86.1 - Q.mres_pruned_mass) / 5.270719) * (1 + math.erf(((86.1 - Q.mres_pruned_mass) / 5.270719) / math.sqrt(2))))
    z += 0.0183 * (4.167088 * 0.5 * ((119.0 - Q.mres_pruned_mass) / 4.167088) * (1 + math.erf(((119.0 - Q.mres_pruned_mass) / 4.167088) / math.sqrt(2))))
    z += 0.0506 * (10.51039 * 0.5 * ((69.2 - Q.mres_sd_mass_b2z01) / 10.51039) * (1 + math.erf(((69.2 - Q.mres_sd_mass_b2z01) / 10.51039) / math.sqrt(2))))
    z += -0.0852 * (4.971962 * 0.5 * ((81.3 - Q.mres_sd_mass_b2z01) / 4.971962) * (1 + math.erf(((81.3 - Q.mres_sd_mass_b2z01) / 4.971962) / math.sqrt(2))))
    z += 0.025 * (3.283929 * 0.5 * ((118.0 - Q.mres_sd_mass_b2z01) / 3.283929) * (1 + math.erf(((118.0 - Q.mres_sd_mass_b2z01) / 3.283929) / math.sqrt(2))))
    z += 0.0384 * (14.176 * 0.5 * ((Q.mres_sd_prong_mass1 - 69.0) / 14.176) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 69.0) / 14.176) / math.sqrt(2))))
    z += -0.0475 * (15.148 * 0.5 * ((Q.mres_sd_prong_mass1 - 83.9) / 15.148) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 83.9) / 15.148) / math.sqrt(2))))
    z += 0.0467 * (2.0 * 0.5 * ((17.0 - Q.n_dr_0p2_0p4) / 2.0) * (1 + math.erf(((17.0 - Q.n_dr_0p2_0p4) / 2.0) / math.sqrt(2))))
    z += 0.148 * (1.5 * 0.5 * ((7.06 - Q.n_lund) / 1.5) * (1 + math.erf(((7.06 - Q.n_lund) / 1.5) / math.sqrt(2))))
    z += 0.179 * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 1.99) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 1.99) / 1.0) / math.sqrt(2))))
    z += -0.154 * (1.0 * 0.5 * ((2.03 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((2.03 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2))))
    z += -0.0546 * (1.0 * 0.5 * ((9.06 - Q.n_sd0_above_3) / 1.0) * (1 + math.erf(((9.06 - Q.n_sd0_above_3) / 1.0) / math.sqrt(2))))
    z += 0.266 * (0.4384575 * 0.5 * ((Q.pair_max_lnkt - 2.33) / 0.4384575) * (1 + math.erf(((Q.pair_max_lnkt - 2.33) / 0.4384575) / math.sqrt(2))))
    z += -0.286 * (0.4384575 * 0.5 * ((2.35 - Q.pair_max_lnkt) / 0.4384575) * (1 + math.erf(((2.35 - Q.pair_max_lnkt) / 0.4384575) / math.sqrt(2))))
    z += -0.00497 * (12.49365 * 0.5 * ((128.0 - Q.sj4_pair_mass_max) / 12.49365) * (1 + math.erf(((128.0 - Q.sj4_pair_mass_max) / 12.49365) / math.sqrt(2))))
    z += 2.04 * (0.2398529 * 0.5 * ((Q.sjq_3_2_k1 - 0.614) / 0.2398529) * (1 + math.erf(((Q.sjq_3_2_k1 - 0.614) / 0.2398529) / math.sqrt(2))))
    z += 2.38 * (0.2334666 * 0.5 * ((-0.605 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.605 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2))))
    z += 1.4 * (0.2508028 * 0.5 * ((-0.661 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.661 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2))))
    z += -2.95 * (0.0257051 * 0.5 * ((0.148 - Q.z_dr_0p2_0p4) / 0.0257051) * (1 + math.erf(((0.148 - Q.z_dr_0p2_0p4) / 0.0257051) / math.sqrt(2))))
    z += 918.0 * (0.004691441 * 0.5 * ((2.34e-05 - Q.z_muon) / 0.004691441) * (1 + math.erf(((2.34e-05 - Q.z_muon) / 0.004691441) / math.sqrt(2))))
    z += 580.0 * (0.0002686389 * 0.5 * ((0.00041 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.00041 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.05415789 * 0.5 * ((-2.05 - Q.lnerel_1) / 0.05415789) * (1 + math.erf(((-2.05 - Q.lnerel_1) / 0.05415789) / math.sqrt(2))))
    z += 77.2 * (0.0002686389 * 0.5 * ((0.000412 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000412 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_neutral_had - 0.998) / 1.0) * (1 + math.erf(((Q.n_neutral_had - 0.998) / 1.0) / math.sqrt(2))))
    z += 2.43 * (0.6671377 * 0.5 * ((0.444 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.444 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.00632) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.00632) / 1e-06) / math.sqrt(2))))
    z += -8.0 * (0.6671377 * 0.5 * ((0.464 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.464 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (0.02224606 * 0.5 * ((0.255 - Q.sj2_dr) / 0.02224606) * (1 + math.erf(((0.255 - Q.sj2_dr) / 0.02224606) / math.sqrt(2))))
    z += 0.00694 * (15.88428 * 0.5 * ((43.0 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((43.0 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.08912976 * 0.5 * ((Q.lne_0 - 4.82) / 0.08912976) * (1 + math.erf(((Q.lne_0 - 4.82) / 0.08912976) / math.sqrt(2))))
    z += -0.728 * (0.1486471 * 0.5 * ((0.339 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.339 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 0.994) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 0.994) / 1.0) / math.sqrt(2))))
    z += 5.24 * (0.1486471 * 0.5 * ((0.34 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.34 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.05582565 * 0.5 * ((0.458 - Q.tau32) / 0.05582565) * (1 + math.erf(((0.458 - Q.tau32) / 0.05582565) / math.sqrt(2))))
    z += 0.0619 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2)))) * (0.01776485 * 0.5 * ((0.272 - Q.N2_b2) / 0.01776485) * (1 + math.erf(((0.272 - Q.N2_b2) / 0.01776485) / math.sqrt(2))))
    z += 0.0039 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2)))) * (1.15674 * 0.5 * ((Q.lne_17 - -0.102) / 1.15674) * (1 + math.erf(((Q.lne_17 - -0.102) / 1.15674) / math.sqrt(2))))
    z += -0.00393 * (5.061224 * 0.5 * ((88.9 - Q.mass) / 5.061224) * (1 + math.erf(((88.9 - Q.mass) / 5.061224) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.04 - Q.sv_1_n) / 1.0) * (1 + math.erf(((3.04 - Q.sv_1_n) / 1.0) / math.sqrt(2))))
    z += -0.0568 * (6.875137 * 0.5 * ((19.2 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((19.2 - Q.mass_displaced3) / 6.875137) / math.sqrt(2)))) * (0.09779513 * 0.5 * ((0.671 - Q.sdb_2_z) / 0.09779513) * (1 + math.erf(((0.671 - Q.sdb_2_z) / 0.09779513) / math.sqrt(2))))
    z += 0.00336 * (3.143401 * 0.5 * ((Q.mass_top15 - 65.1) / 3.143401) * (1 + math.erf(((Q.mass_top15 - 65.1) / 3.143401) / math.sqrt(2)))) * (1e-06 * 0.5 * ((Q.ak02_3_n_lep - 0.00436) / 1e-06) * (1 + math.erf(((Q.ak02_3_n_lep - 0.00436) / 1e-06) / math.sqrt(2))))
    z += -0.183 * (15.148 * 0.5 * ((Q.mres_sd_prong_mass1 - 83.7) / 15.148) * (1 + math.erf(((Q.mres_sd_prong_mass1 - 83.7) / 15.148) / math.sqrt(2)))) * (0.0947145 * 0.5 * ((0.0946 - Q.dc_4_z) / 0.0947145) * (1 + math.erf(((0.0946 - Q.dc_4_z) / 0.0947145) / math.sqrt(2))))
    z += -0.0311 * (1.0 * 0.5 * ((9.03 - Q.n_sd0_above_3) / 1.0) * (1 + math.erf(((9.03 - Q.n_sd0_above_3) / 1.0) / math.sqrt(2)))) * (0.05422308 * 0.5 * ((Q.z_neutral - 0.141) / 0.05422308) * (1 + math.erf(((Q.z_neutral - 0.141) / 0.05422308) / math.sqrt(2))))
    z += -88900.0 * (0.02561531 * 0.5 * ((0.108 - Q.z_displaced3) / 0.02561531) * (1 + math.erf(((0.108 - Q.z_displaced3) / 0.02561531) / math.sqrt(2)))) * (2.8693e-06 * 0.5 * ((2.04e-05 - Q.ecf_g42) / 2.8693e-06) * (1 + math.erf(((2.04e-05 - Q.ecf_g42) / 2.8693e-06) / math.sqrt(2))))
    z += 0.114 * (0.02561531 * 0.5 * ((0.1 - Q.z_displaced3) / 0.02561531) * (1 + math.erf(((0.1 - Q.z_displaced3) / 0.02561531) / math.sqrt(2)))) * (0.9519596 * 0.5 * ((Q.mass_2photon - 2.47) / 0.9519596) * (1 + math.erf(((Q.mass_2photon - 2.47) / 0.9519596) / math.sqrt(2))))
    return z


def neuron_25(Q):
    z = 0.0305
    return z


def neuron_26(Q):
    z = -3.23e-06
    return z


def neuron_27(Q):
    z = -6.3e-06
    return z


def neuron_28(Q):
    z = 9.61e-06
    return z


def neuron_29(Q):
    z = 9.18e-06
    return z


def neuron_30(Q):
    z = -8.6e-06
    return z


def neuron_31(Q):
    z = 3.51e-06
    return z


def neuron_32(Q):
    z = 1.44e-05
    return z


def neuron_33(Q):
    z = 2.56e-06
    return z


def neuron_34(Q):
    z = 1.39e-05
    return z


def neuron_35(Q):
    z = -6.74e-06
    return z


def neuron_36(Q):
    z = -4.84e-06
    return z


def neuron_37(Q):
    z = -4.35e-06
    return z


def neuron_38(Q):
    z = -9.41e-06
    return z


def neuron_39(Q):
    z = 8.27e-06
    return z


def neuron_40(Q):
    z = 7.83e-07
    return z


def neuron_41(Q):
    z = 4.12e-08
    return z


def neuron_42(Q):
    z = 8.78e-06
    return z


def neuron_43(Q):
    z = -9.4e-06
    return z


def neuron_44(Q):
    z = -3.33e-06
    return z


def neuron_45(Q):
    z = 6.85e-06
    return z


def neuron_46(Q):
    z = 1.31e-07
    return z


def neuron_47(Q):
    z = 2.46e-07
    return z


def neuron_48(Q):
    z = -8.8e-06
    return z


def neuron_49(Q):
    z = -2.24e-06
    return z


def neuron_50(Q):
    z = 5.17e-06
    return z


def neuron_51(Q):
    z = -2.02e-06
    return z


def neuron_52(Q):
    z = -1.34
    z += -0.516 * (0.09740396 * 0.5 * ((1.31 - Q.D2) / 0.09740396) * (1 + math.erf(((1.31 - Q.D2) / 0.09740396) / math.sqrt(2))))
    z += 0.263 * (1.0 * 0.5 * ((1.0 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((1.0 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2))))
    z += 492.0 * (0.0002686389 * 0.5 * ((0.000414 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000414 - Q.e3_b2) / 0.0002686389) / math.sqrt(2))))
    z += 808.0 * (8.859433e-05 * 0.5 * ((Q.ecf_g41 - 0.000269) / 8.859433e-05) * (1 + math.erf(((Q.ecf_g41 - 0.000269) / 8.859433e-05) / math.sqrt(2))))
    z += -704.0 * (8.859433e-05 * 0.5 * ((0.000272 - Q.ecf_g41) / 8.859433e-05) * (1 + math.erf(((0.000272 - Q.ecf_g41) / 8.859433e-05) / math.sqrt(2))))
    z += -0.242 * (0.08734879 * 0.5 * ((1.6 - Q.jd_3d_6) / 0.08734879) * (1 + math.erf(((1.6 - Q.jd_3d_6) / 0.08734879) / math.sqrt(2))))
    z += -0.321 * (0.08168809 * 0.5 * ((Q.jet_abs_eta - 0.83) / 0.08168809) * (1 + math.erf(((Q.jet_abs_eta - 0.83) / 0.08168809) / math.sqrt(2))))
    z += 1.19 * (0.0326228 * 0.5 * ((Q.kt2_dr12 - 0.494) / 0.0326228) * (1 + math.erf(((Q.kt2_dr12 - 0.494) / 0.0326228) / math.sqrt(2))))
    z += 2.71 * (0.0496275 * 0.5 * ((0.198 - Q.kt2_dr12) / 0.0496275) * (1 + math.erf(((0.198 - Q.kt2_dr12) / 0.0496275) / math.sqrt(2))))
    z += -1.04 * (0.0408915 * 0.5 * ((Q.lep_dr - 0.0653) / 0.0408915) * (1 + math.erf(((Q.lep_dr - 0.0653) / 0.0408915) / math.sqrt(2))))
    z += 0.266 * (1.234622 * 0.5 * ((1.5 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.5 - Q.lep_iso) / 1.234622) / math.sqrt(2))))
    z += -2.4 * (0.1486471 * 0.5 * ((0.339 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.339 - Q.lep_z) / 0.1486471) / math.sqrt(2))))
    z += -6.18 * (0.03060878 * 0.5 * ((0.0667 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0667 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2))))
    z += -26.3 * (5.001896 * 0.5 * ((-15.4 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((-15.4 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2))))
    z += 0.0486 * (11.49171 * 0.5 * ((72.9 - Q.mass) / 11.49171) * (1 + math.erf(((72.9 - Q.mass) / 11.49171) / math.sqrt(2))))
    z += -0.0143 * (2.497115 * 0.5 * ((Q.mass_displaced5 - 6.01) / 2.497115) * (1 + math.erf(((Q.mass_displaced5 - 6.01) / 2.497115) / math.sqrt(2))))
    z += -0.00795 * (8.307776 * 0.5 * ((Q.mass_top50 - 137.0) / 8.307776) * (1 + math.erf(((Q.mass_top50 - 137.0) / 8.307776) / math.sqrt(2))))
    z += 0.00573 * (3.304036 * 0.5 * ((116.0 - Q.mass_top50) / 3.304036) * (1 + math.erf(((116.0 - Q.mass_top50) / 3.304036) / math.sqrt(2))))
    z += 0.0222 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.3) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.3) / 5.1259) / math.sqrt(2))))
    z += -0.0245 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 177.0) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 177.0) / 18.97147) / math.sqrt(2))))
    z += -0.0231 * (24.25363 * 0.5 * ((53.0 - Q.mres_sd_mass_b0z005) / 24.25363) * (1 + math.erf(((53.0 - Q.mres_sd_mass_b0z005) / 24.25363) / math.sqrt(2))))
    z += -0.0106 * (4.488781 * 0.5 * ((126.0 - Q.mres_sd_mass_b0z005) / 4.488781) * (1 + math.erf(((126.0 - Q.mres_sd_mass_b0z005) / 4.488781) / math.sqrt(2))))
    z += 0.0296 * (18.66842 * 0.5 * ((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2))))
    z += -0.0696 * (1.5 * 0.5 * ((5.91 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((5.91 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2))))
    z += 0.0757 * (1.0 * 0.5 * ((7.78 - Q.n_lund) / 1.0) * (1 + math.erf(((7.78 - Q.n_lund) / 1.0) / math.sqrt(2))))
    z += 0.0118 * (21.5 * 0.5 * ((79.8 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((79.8 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += 0.0113 * (2.5 * 0.5 * ((Q.n_particles - 25.9) / 2.5) * (1 + math.erf(((Q.n_particles - 25.9) / 2.5) / math.sqrt(2))))
    z += 0.117 * (1.5 * 0.5 * ((7.01 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((7.01 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2))))
    z += -0.197 * (1.5 * 0.5 * ((5.99 - Q.n_sd0_above_3) / 1.5) * (1 + math.erf(((5.99 - Q.n_sd0_above_3) / 1.5) / math.sqrt(2))))
    z += 1.83 * (0.08149619 * 0.5 * ((0.242 - Q.pt1_over_pt0) / 0.08149619) * (1 + math.erf(((0.242 - Q.pt1_over_pt0) / 0.08149619) / math.sqrt(2))))
    z += -0.0112 * (7.767937 * 0.5 * ((Q.sj3_mass1 - 43.5) / 7.767937) * (1 + math.erf(((Q.sj3_mass1 - 43.5) / 7.767937) / math.sqrt(2))))
    z += 0.0196 * (11.90664 * 0.5 * ((Q.sj3_pair_mass_min - 80.7) / 11.90664) * (1 + math.erf(((Q.sj3_pair_mass_min - 80.7) / 11.90664) / math.sqrt(2))))
    z += 0.0804 * (4.481852 * 0.5 * ((6.48 - Q.sjf_2_1_max3d) / 4.481852) * (1 + math.erf(((6.48 - Q.sjf_2_1_max3d) / 4.481852) / math.sqrt(2))))
    z += -0.00911 * (17.1726 * 0.5 * ((36.4 - Q.sjf_2_1_max3d) / 17.1726) * (1 + math.erf(((36.4 - Q.sjf_2_1_max3d) / 17.1726) / math.sqrt(2))))
    z += 0.128 * (3.069852 * 0.5 * ((4.65 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.65 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2))))
    z += 2.12 * (0.2681069 * 0.5 * ((Q.sjq_2_2_k1 - 0.646) / 0.2681069) * (1 + math.erf(((Q.sjq_2_2_k1 - 0.646) / 0.2681069) / math.sqrt(2))))
    z += 1.15 * (0.2610212 * 0.5 * ((-0.606 - Q.sjq_2_2_k1) / 0.2610212) * (1 + math.erf(((-0.606 - Q.sjq_2_2_k1) / 0.2610212) / math.sqrt(2))))
    z += 0.391 * (0.1159667 * 0.5 * ((0.485 - Q.sv_1_dr) / 0.1159667) * (1 + math.erf(((0.485 - Q.sv_1_dr) / 0.1159667) / math.sqrt(2))))
    z += -0.00678 * (20.79165 * 0.5 * ((52.1 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((52.1 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2))))
    z += 7.44 * (0.03097976 * 0.5 * ((0.134 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.134 - Q.z_displaced3) / 0.03097976) / math.sqrt(2))))
    z += 0.177 * (0.09740396 * 0.5 * ((1.32 - Q.D2) / 0.09740396) * (1 + math.erf(((1.32 - Q.D2) / 0.09740396) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjq_3_2_nch - 4.05) / 1.0) * (1 + math.erf(((Q.sjq_3_2_nch - 4.05) / 1.0) / math.sqrt(2))))
    z += 5020.0 * (0.0002686389 * 0.5 * ((0.000416 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000416 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.01565591 * 0.5 * ((0.168 - Q.pz_lnkt1) / 0.01565591) * (1 + math.erf(((0.168 - Q.pz_lnkt1) / 0.01565591) / math.sqrt(2))))
    z += -0.0148 * (1.234622 * 0.5 * ((1.42 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.42 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (4.0 * 0.5 * ((23.9 - Q.n_dr_0p2_0p4) / 4.0) * (1 + math.erf(((23.9 - Q.n_dr_0p2_0p4) / 4.0) / math.sqrt(2))))
    z += 0.0463 * (0.1486471 * 0.5 * ((0.339 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.339 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 11.0) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 11.0) / 1.0) / math.sqrt(2))))
    z += -6.54 * (5.001896 * 0.5 * ((1.59 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((1.59 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2)))) * (0.0330605 * 0.5 * ((0.103 - Q.lep_dr) / 0.0330605) * (1 + math.erf(((0.103 - Q.lep_dr) / 0.0330605) / math.sqrt(2))))
    z += 0.0679 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 81.5) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 81.5) / 5.1259) / math.sqrt(2)))) * (0.03966789 * 0.5 * ((0.134 - Q.ak02_3_z) / 0.03966789) * (1 + math.erf(((0.134 - Q.ak02_3_z) / 0.03966789) / math.sqrt(2))))
    z += -0.0136 * (18.66842 * 0.5 * ((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2)))) * (0.04741882 * 0.5 * ((0.466 - Q.tau21_b2) / 0.04741882) * (1 + math.erf(((0.466 - Q.tau21_b2) / 0.04741882) / math.sqrt(2))))
    z += -0.0207 * (5.1259 * 0.5 * ((Q.mres_sd_mass_b0z005 - 82.1) / 5.1259) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 82.1) / 5.1259) / math.sqrt(2)))) * (0.03693697 * 0.5 * ((0.584 - Q.tau32_b2) / 0.03693697) * (1 + math.erf(((0.584 - Q.tau32_b2) / 0.03693697) / math.sqrt(2))))
    z += 0.038 * (18.97147 * 0.5 * ((Q.mres_sd_mass_b0z005 - 178.0) / 18.97147) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 178.0) / 18.97147) / math.sqrt(2)))) * (0.03559215 * 0.5 * ((0.624 - Q.tau32_b2) / 0.03559215) * (1 + math.erf(((0.624 - Q.tau32_b2) / 0.03559215) / math.sqrt(2))))
    z += -0.0328 * (1.0 * 0.5 * ((8.95 - Q.n_lund_kt_above_1) / 1.0) * (1 + math.erf(((8.95 - Q.n_lund_kt_above_1) / 1.0) / math.sqrt(2)))) * (0.3035439 * 0.5 * ((Q.sjq_2_prod_k03 - -1.02) / 0.3035439) * (1 + math.erf(((Q.sjq_2_prod_k03 - -1.02) / 0.3035439) / math.sqrt(2))))
    z += 0.00142 * (6.0 * 0.5 * ((28.8 - Q.n_pairs_kt_above_3) / 6.0) * (1 + math.erf(((28.8 - Q.n_pairs_kt_above_3) / 6.0) / math.sqrt(2)))) * (1.375431 * 0.5 * ((Q.mass_2photon - 3.9) / 1.375431) * (1 + math.erf(((Q.mass_2photon - 3.9) / 1.375431) / math.sqrt(2))))
    z += -6720.0 * (0.03097976 * 0.5 * ((0.134 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.134 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (0.0001714083 * 0.5 * ((0.00019 - Q.C3_b2) / 0.0001714083) * (1 + math.erf(((0.00019 - Q.C3_b2) / 0.0001714083) / math.sqrt(2))))
    z += 7.01 * (0.03097976 * 0.5 * ((0.134 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.134 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (0.02727943 * 0.5 * ((Q.N2 - 0.183) / 0.02727943) * (1 + math.erf(((Q.N2 - 0.183) / 0.02727943) / math.sqrt(2))))
    z += -0.00403 * (0.03097976 * 0.5 * ((0.135 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.135 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (401.9953 * 0.5 * ((578.0 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((578.0 - Q.sip_3d_3) / 401.9953) / math.sqrt(2))))
    z += -0.0303 * (0.03097976 * 0.5 * ((0.134 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.134 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (110.8422 * 0.5 * ((116.0 - Q.sjf_2_2_max3d) / 110.8422) * (1 + math.erf(((116.0 - Q.sjf_2_2_max3d) / 110.8422) / math.sqrt(2))))
    z += -1.2 * (0.03097976 * 0.5 * ((0.136 - Q.z_displaced3) / 0.03097976) * (1 + math.erf(((0.136 - Q.z_displaced3) / 0.03097976) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_3_3_n_d3 - 0.971) / 1.0) * (1 + math.erf(((Q.sjf_3_3_n_d3 - 0.971) / 1.0) / math.sqrt(2))))
    return z


def neuron_53(Q):
    z = -1.9e-06
    return z


def neuron_54(Q):
    z = 1.19e-05
    return z


def neuron_55(Q):
    z = -2.07e-05
    return z


def neuron_56(Q):
    z = -7.13e-07
    return z


def neuron_57(Q):
    z = -1.95e-06
    return z


def neuron_58(Q):
    z = -3.61e-06
    return z


def neuron_59(Q):
    z = 6.03e-06
    return z


def neuron_60(Q):
    z = -4.67e-06
    return z


def neuron_61(Q):
    z = 5.84e-06
    return z


def neuron_62(Q):
    z = 9.41e-07
    return z


def neuron_63(Q):
    z = -8.31e-06
    return z


def neuron_64(Q):
    z = 1.25e-06
    return z


def neuron_65(Q):
    z = 5.79e-06
    return z


def neuron_66(Q):
    z = 2.71e-06
    return z


def neuron_67(Q):
    z = -5.23e-06
    return z


def neuron_68(Q):
    z = 1.14
    z += -2.5 * (0.04853964 * 0.5 * ((Q.C2 - 0.193) / 0.04853964) * (1 + math.erf(((Q.C2 - 0.193) / 0.04853964) / math.sqrt(2))))
    z += 2.59 * (0.04853964 * 0.5 * ((0.19 - Q.C2) / 0.04853964) * (1 + math.erf(((0.19 - Q.C2) / 0.04853964) / math.sqrt(2))))
    z += -0.234 * (0.298047 * 0.5 * ((2.29 - Q.D2_b2) / 0.298047) * (1 + math.erf(((2.29 - Q.D2_b2) / 0.298047) / math.sqrt(2))))
    z += 3.71 * (0.2297361 * 0.5 * ((-5.68e-05 - Q.ak02_dr13) / 0.2297361) * (1 + math.erf(((-5.68e-05 - Q.ak02_dr13) / 0.2297361) / math.sqrt(2))))
    z += 0.491 * (0.05861841 * 0.5 * ((Q.ak02_dr23 - 0.369) / 0.05861841) * (1 + math.erf(((Q.ak02_dr23 - 0.369) / 0.05861841) / math.sqrt(2))))
    z += -2.1 * (0.07842787 * 0.5 * ((Q.dc_split2_dr - 0.362) / 0.07842787) * (1 + math.erf(((Q.dc_split2_dr - 0.362) / 0.07842787) / math.sqrt(2))))
    z += 497.0 * (0.0003783019 * 0.5 * ((0.000807 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.000807 - Q.e3_b2) / 0.0003783019) / math.sqrt(2))))
    z += 130.0 * (0.001201988 * 0.5 * ((0.00282 - Q.sum_z_dr2_top3) / 0.001201988) * (1 + math.erf(((0.00282 - Q.sum_z_dr2_top3) / 0.001201988) / math.sqrt(2))))
    z += -0.148 * (0.1659628 * 0.5 * ((1.46 - Q.jet_abs_eta) / 0.1659628) * (1 + math.erf(((1.46 - Q.jet_abs_eta) / 0.1659628) / math.sqrt(2))))
    z += 0.103 * (0.1783594 * 0.5 * ((Q.ktd_ln_d34 - -9.17) / 0.1783594) * (1 + math.erf(((Q.ktd_ln_d34 - -9.17) / 0.1783594) / math.sqrt(2))))
    z += 4.93 * (0.03429006 * 0.5 * ((0.0802 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.0802 - Q.lep_dr) / 0.03429006) / math.sqrt(2))))
    z += -1.63 * (0.6671377 * 0.5 * ((0.438 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.438 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += -0.0334 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.9) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.9) / 4.308625) / math.sqrt(2))))
    z += 0.06 * (7.585662 * 0.5 * ((Q.lep_ptrel - 18.7) / 7.585662) * (1 + math.erf(((Q.lep_ptrel - 18.7) / 7.585662) / math.sqrt(2))))
    z += -2.28 * (0.179073 * 0.5 * ((Q.lep_z - 0.517) / 0.179073) * (1 + math.erf(((Q.lep_z - 0.517) / 0.179073) / math.sqrt(2))))
    z += 0.0112 * (5.061224 * 0.5 * ((90.3 - Q.mass) / 5.061224) * (1 + math.erf(((90.3 - Q.mass) / 5.061224) / math.sqrt(2))))
    z += 0.0121 * (3.317894 * 0.5 * ((117.0 - Q.mass) / 3.317894) * (1 + math.erf(((117.0 - Q.mass) / 3.317894) / math.sqrt(2))))
    z += -0.00725 * (10.67504 * 0.5 * ((Q.mass_top10 - 103.0) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 103.0) / 10.67504) / math.sqrt(2))))
    z += -0.0125 * (10.8986 * 0.5 * ((67.3 - Q.mass_top30) / 10.8986) * (1 + math.erf(((67.3 - Q.mass_top30) / 10.8986) / math.sqrt(2))))
    z += -0.011 * (16.26099 * 0.5 * ((161.0 - Q.mass_top50) / 16.26099) * (1 + math.erf(((161.0 - Q.mass_top50) / 16.26099) / math.sqrt(2))))
    z += 0.00991 * (18.57486 * 0.5 * ((Q.mres_pruned_mass - 41.6) / 18.57486) * (1 + math.erf(((Q.mres_pruned_mass - 41.6) / 18.57486) / math.sqrt(2))))
    z += -0.0202 * (5.270719 * 0.5 * ((Q.mres_pruned_mass - 86.1) / 5.270719) * (1 + math.erf(((Q.mres_pruned_mass - 86.1) / 5.270719) / math.sqrt(2))))
    z += 0.0592 * (12.47875 * 0.5 * ((Q.mres_pruned_mass - 131.0) / 12.47875) * (1 + math.erf(((Q.mres_pruned_mass - 131.0) / 12.47875) / math.sqrt(2))))
    z += -0.0656 * (20.04228 * 0.5 * ((Q.mres_pruned_mass - 149.0) / 20.04228) * (1 + math.erf(((Q.mres_pruned_mass - 149.0) / 20.04228) / math.sqrt(2))))
    z += 0.021 * (22.10989 * 0.5 * ((Q.mres_pruned_mass - 171.0) / 22.10989) * (1 + math.erf(((Q.mres_pruned_mass - 171.0) / 22.10989) / math.sqrt(2))))
    z += 0.0413 * (19.93276 * 0.5 * ((55.3 - Q.mres_sd_mass_b2z01) / 19.93276) * (1 + math.erf(((55.3 - Q.mres_sd_mass_b2z01) / 19.93276) / math.sqrt(2))))
    z += -0.0625 * (6.036668 * 0.5 * ((76.2 - Q.mres_sd_mass_b2z01) / 6.036668) * (1 + math.erf(((76.2 - Q.mres_sd_mass_b2z01) / 6.036668) / math.sqrt(2))))
    z += 0.0213 * (4.590131 * 0.5 * ((107.0 - Q.mres_sd_mass_b2z01) / 4.590131) * (1 + math.erf(((107.0 - Q.mres_sd_mass_b2z01) / 4.590131) / math.sqrt(2))))
    z += 0.0174 * (3.499256 * 0.5 * ((122.0 - Q.mres_sd_mass_b2z01) / 3.499256) * (1 + math.erf(((122.0 - Q.mres_sd_mass_b2z01) / 3.499256) / math.sqrt(2))))
    z += -0.0127 * (19.14168 * 0.5 * ((158.0 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((158.0 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2))))
    z += -0.613 * (0.02593915 * 0.5 * ((Q.mres_sd_rg_b1z01 - 0.471) / 0.02593915) * (1 + math.erf(((Q.mres_sd_rg_b1z01 - 0.471) / 0.02593915) / math.sqrt(2))))
    z += -0.0508 * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 19.9) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 19.9) / 1.0) / math.sqrt(2))))
    z += 0.00996 * (21.5 * 0.5 * ((79.2 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((79.2 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += 0.022 * (7.0 * 0.5 * ((64.9 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((64.9 - Q.n_pt_above_1) / 7.0) / math.sqrt(2))))
    z += -0.011 * (5.345127 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 77.4) / 5.345127) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 77.4) / 5.345127) / math.sqrt(2))))
    z += -0.211 * (0.2410611 * 0.5 * ((3.38 - Q.pair_mean_lnm2) / 0.2410611) * (1 + math.erf(((3.38 - Q.pair_mean_lnm2) / 0.2410611) / math.sqrt(2))))
    z += -3.13 * (0.0138316 * 0.5 * ((0.251 - Q.pz_lnd2) / 0.0138316) * (1 + math.erf(((0.251 - Q.pz_lnd2) / 0.0138316) / math.sqrt(2))))
    z += -1.94 * (0.02630693 * 0.5 * ((0.237 - Q.pz_lnd3) / 0.02630693) * (1 + math.erf(((0.237 - Q.pz_lnd3) / 0.02630693) / math.sqrt(2))))
    z += -2.16 * (0.02719315 * 0.5 * ((0.154 - Q.sdb_2_z) / 0.02719315) * (1 + math.erf(((0.154 - Q.sdb_2_z) / 0.02719315) / math.sqrt(2))))
    z += 0.0114 * (1.369872 * 0.5 * ((Q.sj3_mass1 - 17.3) / 1.369872) * (1 + math.erf(((Q.sj3_mass1 - 17.3) / 1.369872) / math.sqrt(2))))
    z += 0.00977 * (1.504711 * 0.5 * ((Q.sj3_mass2 - 15.3) / 1.504711) * (1 + math.erf(((Q.sj3_mass2 - 15.3) / 1.504711) / math.sqrt(2))))
    z += 0.012 * (11.17405 * 0.5 * ((Q.sj3_pair_mass_max - 129.0) / 11.17405) * (1 + math.erf(((Q.sj3_pair_mass_max - 129.0) / 11.17405) / math.sqrt(2))))
    z += -0.0159 * (11.90664 * 0.5 * ((79.9 - Q.sj3_pair_mass_min) / 11.90664) * (1 + math.erf(((79.9 - Q.sj3_pair_mass_min) / 11.90664) / math.sqrt(2))))
    z += -2.16 * (0.02434011 * 0.5 * ((Q.sj3_pairmin_over_m - 0.461) / 0.02434011) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.461) / 0.02434011) / math.sqrt(2))))
    z += -0.00774 * (2.973269 * 0.5 * ((Q.sj4_pair_mass_max - 72.8) / 2.973269) * (1 + math.erf(((Q.sj4_pair_mass_max - 72.8) / 2.973269) / math.sqrt(2))))
    z += 1.05 * (0.007115881 * 0.5 * ((Q.sjf_4_1_z_d3 - 0.00808) / 0.007115881) * (1 + math.erf(((Q.sjf_4_1_z_d3 - 0.00808) / 0.007115881) / math.sqrt(2))))
    z += -0.111 * (1.0 * 0.5 * ((2.98 - Q.sjq_2_2_nch) / 1.0) * (1 + math.erf(((2.98 - Q.sjq_2_2_nch) / 1.0) / math.sqrt(2))))
    z += 0.983 * (0.0216761 * 0.5 * ((0.819 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.819 - Q.tau32) / 0.0216761) / math.sqrt(2))))
    z += 1.01 * (0.01332924 * 0.5 * ((0.907 - Q.tau43) / 0.01332924) * (1 + math.erf(((0.907 - Q.tau43) / 0.01332924) / math.sqrt(2))))
    z += 0.364 * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.267) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.267) / 0.0649922) / math.sqrt(2))))
    z += -0.763 * (0.03746729 * 0.5 * ((Q.z_displaced3 - 0.163) / 0.03746729) * (1 + math.erf(((Q.z_displaced3 - 0.163) / 0.03746729) / math.sqrt(2))))
    z += -3.44 * (0.02099671 * 0.5 * ((0.0829 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.0829 - Q.z_displaced3) / 0.02099671) / math.sqrt(2))))
    z += -21.8 * (0.00254714 * 0.5 * ((0.0388 - Q.M3) / 0.00254714) * (1 + math.erf(((0.0388 - Q.M3) / 0.00254714) / math.sqrt(2)))) * (0.06042313 * 0.5 * ((Q.sjq_3_sumabs_k1 - 0.439) / 0.06042313) * (1 + math.erf(((Q.sjq_3_sumabs_k1 - 0.439) / 0.06042313) / math.sqrt(2))))
    z += -0.00186 * (0.0218375 * 0.5 * ((0.0762 - Q.ak02_3_z) / 0.0218375) * (1 + math.erf(((0.0762 - Q.ak02_3_z) / 0.0218375) / math.sqrt(2)))) * (1032.513 * 0.5 * ((1160.0 - Q.lepsj_2_maxsd0) / 1032.513) * (1 + math.erf(((1160.0 - Q.lepsj_2_maxsd0) / 1032.513) / math.sqrt(2))))
    z += 244.0 * (0.0002686389 * 0.5 * ((0.000437 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000437 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_lund_kt_above_5 - 2.04) / 1.0) * (1 + math.erf(((Q.n_lund_kt_above_5 - 2.04) / 1.0) / math.sqrt(2))))
    z += -5380.0 * (0.0002686389 * 0.5 * ((0.000403 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000403 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.03798057 * 0.5 * ((Q.z_top30_slots - 0.83) / 0.03798057) * (1 + math.erf(((Q.z_top30_slots - 0.83) / 0.03798057) / math.sqrt(2))))
    z += 0.00495 * (3.317894 * 0.5 * ((117.0 - Q.mass) / 3.317894) * (1 + math.erf(((117.0 - Q.mass) / 3.317894) / math.sqrt(2)))) * (0.3667324 * 0.5 * ((2.59 - Q.D2_b2) / 0.3667324) * (1 + math.erf(((2.59 - Q.D2_b2) / 0.3667324) / math.sqrt(2))))
    z += -0.0118 * (5.061224 * 0.5 * ((90.4 - Q.mass) / 5.061224) * (1 + math.erf(((90.4 - Q.mass) / 5.061224) / math.sqrt(2)))) * (5.001896 * 0.5 * ((2.43 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((2.43 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2))))
    z += -0.029 * (5.270719 * 0.5 * ((Q.mres_pruned_mass - 86.4) / 5.270719) * (1 + math.erf(((Q.mres_pruned_mass - 86.4) / 5.270719) / math.sqrt(2)))) * (0.02798345 * 0.5 * ((0.172 - Q.dc_3_z) / 0.02798345) * (1 + math.erf(((0.172 - Q.dc_3_z) / 0.02798345) / math.sqrt(2))))
    z += 44.6 * (12.47875 * 0.5 * ((Q.mres_pruned_mass - 131.0) / 12.47875) * (1 + math.erf(((Q.mres_pruned_mass - 131.0) / 12.47875) / math.sqrt(2)))) * (0.002064355 * 0.5 * ((1.6e-05 - Q.lepsj_3_dr) / 0.002064355) * (1 + math.erf(((1.6e-05 - Q.lepsj_3_dr) / 0.002064355) / math.sqrt(2))))
    z += 0.00129 * (1.0 * 0.5 * ((Q.n_charged_pt_above_1 - 19.9) / 1.0) * (1 + math.erf(((Q.n_charged_pt_above_1 - 19.9) / 1.0) / math.sqrt(2)))) * (1.5 * 0.5 * ((17.1 - Q.n_dr_0p1_0p2) / 1.5) * (1 + math.erf(((17.1 - Q.n_dr_0p1_0p2) / 1.5) / math.sqrt(2))))
    z += 0.00768 * (7.0 * 0.5 * ((65.0 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.0 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.998 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((0.998 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.00429 * (7.0 * 0.5 * ((65.1 - Q.n_pt_above_1) / 7.0) * (1 + math.erf(((65.1 - Q.n_pt_above_1) / 7.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.997 - Q.n_muon) / 1.0) * (1 + math.erf(((0.997 - Q.n_muon) / 1.0) / math.sqrt(2))))
    z += -0.0685 * (2.0 * 0.5 * ((11.1 - Q.n_sd0_above_2) / 2.0) * (1 + math.erf(((11.1 - Q.n_sd0_above_2) / 2.0) / math.sqrt(2)))) * (0.117671 * 0.5 * ((0.414 - Q.lep_dr) / 0.117671) * (1 + math.erf(((0.414 - Q.lep_dr) / 0.117671) / math.sqrt(2))))
    z += 0.724 * (0.2410611 * 0.5 * ((3.38 - Q.pair_mean_lnm2) / 0.2410611) * (1 + math.erf(((3.38 - Q.pair_mean_lnm2) / 0.2410611) / math.sqrt(2)))) * (0.06955833 * 0.5 * ((0.378 - Q.z_neutral_had) / 0.06955833) * (1 + math.erf(((0.378 - Q.z_neutral_had) / 0.06955833) / math.sqrt(2))))
    z += 0.00473 * (0.0138316 * 0.5 * ((0.25 - Q.pz_lnd2) / 0.0138316) * (1 + math.erf(((0.25 - Q.pz_lnd2) / 0.0138316) / math.sqrt(2)))) * (115.0698 * 0.5 * ((170.0 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((170.0 - Q.jd_3d_4) / 115.0698) / math.sqrt(2))))
    z += -0.0182 * (1.0 * 0.5 * ((Q.sdb_2_n - 11.0) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 11.0) / 1.0) / math.sqrt(2)))) * (1.739103 * 0.5 * ((4.02 - Q.jd_3d_4) / 1.739103) * (1 + math.erf(((4.02 - Q.jd_3d_4) / 1.739103) / math.sqrt(2))))
    z += 0.0158 * (0.0216761 * 0.5 * ((0.819 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.819 - Q.tau32) / 0.0216761) / math.sqrt(2)))) * (11.90664 * 0.5 * ((80.2 - Q.sj3_pair_mass_min) / 11.90664) * (1 + math.erf(((80.2 - Q.sj3_pair_mass_min) / 11.90664) / math.sqrt(2))))
    z += 1.17 * (0.0216761 * 0.5 * ((0.817 - Q.tau32) / 0.0216761) * (1 + math.erf(((0.817 - Q.tau32) / 0.0216761) / math.sqrt(2)))) * (0.01950041 * 0.5 * ((Q.sv_1_dr - 0.132) / 0.01950041) * (1 + math.erf(((Q.sv_1_dr - 0.132) / 0.01950041) / math.sqrt(2))))
    z += 3.24 * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.266) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.266) / 0.0649922) / math.sqrt(2)))) * (0.01039253 * 0.5 * ((0.922 - Q.tau54) / 0.01039253) * (1 + math.erf(((0.922 - Q.tau54) / 0.01039253) / math.sqrt(2))))
    z += -44.9 * (0.02099671 * 0.5 * ((0.0826 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.0826 - Q.z_displaced3) / 0.02099671) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.00259 - Q.n_muon) / 1.0) * (1 + math.erf(((0.00259 - Q.n_muon) / 1.0) / math.sqrt(2))))
    return z


def neuron_69(Q):
    z = -6.92e-06
    return z


def neuron_70(Q):
    z = 0.0112
    z += -2.22 * (0.0134569 * 0.5 * ((Q.C2 - 0.066) / 0.0134569) * (1 + math.erf(((Q.C2 - 0.066) / 0.0134569) / math.sqrt(2))))
    z += -5.68 * (0.03060878 * 0.5 * ((0.0616 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0616 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2))))
    z += 0.0928 * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 1.71) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 1.71) / 1.0) / math.sqrt(2))))
    z += -0.00724 * (3.329534 * 0.5 * ((Q.mass_2charged - 14.7) / 3.329534) * (1 + math.erf(((Q.mass_2charged - 14.7) / 3.329534) / math.sqrt(2))))
    z += 0.0406 * (8.963785 * 0.5 * ((23.6 - Q.mass_charged) / 8.963785) * (1 + math.erf(((23.6 - Q.mass_charged) / 8.963785) / math.sqrt(2))))
    z += 0.0157 * (3.764443 * 0.5 * ((Q.mass_top50 - 127.0) / 3.764443) * (1 + math.erf(((Q.mass_top50 - 127.0) / 3.764443) / math.sqrt(2))))
    z += 0.0053 * (11.41538 * 0.5 * ((Q.mres_sd_mass_b0z005 - 79.6) / 11.41538) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 79.6) / 11.41538) / math.sqrt(2))))
    z += -0.0242 * (3.563419 * 0.5 * ((Q.mres_sd_mass_b0z005 - 122.0) / 3.563419) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 122.0) / 3.563419) / math.sqrt(2))))
    z += 0.028 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 162.0) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 162.0) / 18.66842) / math.sqrt(2))))
    z += -0.0206 * (19.33798 * 0.5 * ((Q.mres_sd_mass_b2z01 - 175.0) / 19.33798) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 175.0) / 19.33798) / math.sqrt(2))))
    z += 0.0521 * (1.0 * 0.5 * ((18.6 - Q.n_neutral) / 1.0) * (1 + math.erf(((18.6 - Q.n_neutral) / 1.0) / math.sqrt(2))))
    z += 6.99 * (0.009408518 * 0.5 * ((0.111 - Q.pz_lnd0) / 0.009408518) * (1 + math.erf(((0.111 - Q.pz_lnd0) / 0.009408518) / math.sqrt(2))))
    z += -1.81 * (0.02343699 * 0.5 * ((0.221 - Q.sdb_2_z) / 0.02343699) * (1 + math.erf(((0.221 - Q.sdb_2_z) / 0.02343699) / math.sqrt(2))))
    z += -0.0119 * (3.974361 * 0.5 * ((Q.sj3_pair_mass_min - 40.4) / 3.974361) * (1 + math.erf(((Q.sj3_pair_mass_min - 40.4) / 3.974361) / math.sqrt(2))))
    z += -1.64 * (0.04964267 * 0.5 * ((Q.sj4_dr_min - 0.261) / 0.04964267) * (1 + math.erf(((Q.sj4_dr_min - 0.261) / 0.04964267) / math.sqrt(2))))
    z += 0.19 * (1.0 * 0.5 * ((Q.sjf_4_n2disp - 1.11) / 1.0) * (1 + math.erf(((Q.sjf_4_n2disp - 1.11) / 1.0) / math.sqrt(2))))
    z += 1.64 * (0.2334666 * 0.5 * ((-0.59 - Q.sjq_3_2_k1) / 0.2334666) * (1 + math.erf(((-0.59 - Q.sjq_3_2_k1) / 0.2334666) / math.sqrt(2))))
    z += 2.1 * (0.2498329 * 0.5 * ((Q.sjq_3_3_k1 - 0.64) / 0.2498329) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.64) / 0.2498329) / math.sqrt(2))))
    z += 1.27 * (0.2508028 * 0.5 * ((-0.673 - Q.sjq_3_3_k1) / 0.2508028) * (1 + math.erf(((-0.673 - Q.sjq_3_3_k1) / 0.2508028) / math.sqrt(2))))
    z += 7.21 * (0.0134569 * 0.5 * ((Q.C2 - 0.0645) / 0.0134569) * (1 + math.erf(((Q.C2 - 0.0645) / 0.0134569) / math.sqrt(2)))) * (0.0329306 * 0.5 * ((0.434 - Q.ak02_2_z) / 0.0329306) * (1 + math.erf(((0.434 - Q.ak02_2_z) / 0.0329306) / math.sqrt(2))))
    z += -0.609 * (1.234622 * 0.5 * ((1.79 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.79 - Q.lep_iso) / 1.234622) / math.sqrt(2)))) * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.388) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.388) / 0.179816) / math.sqrt(2))))
    z += -0.355 * (0.1486471 * 0.5 * ((0.233 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.233 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_2_n_d3 - 0.189) / 1.0) * (1 + math.erf(((Q.sjf_2_2_n_d3 - 0.189) / 1.0) / math.sqrt(2))))
    z += 2.41 * (0.1486471 * 0.5 * ((0.452 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.452 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.03538975 * 0.5 * ((0.591 - Q.tau32) / 0.03538975) * (1 + math.erf(((0.591 - Q.tau32) / 0.03538975) / math.sqrt(2))))
    z += 1.26 * (1.0 * 0.5 * ((1.68 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((1.68 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2)))) * (0.0330605 * 0.5 * ((0.176 - Q.lep_dr) / 0.0330605) * (1 + math.erf(((0.176 - Q.lep_dr) / 0.0330605) / math.sqrt(2))))
    z += -0.00853 * (1.0 * 0.5 * ((18.9 - Q.n_neutral) / 1.0) * (1 + math.erf(((18.9 - Q.n_neutral) / 1.0) / math.sqrt(2)))) * (3.102391 * 0.5 * ((7.31 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((7.31 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += -21.4 * (0.009408518 * 0.5 * ((0.115 - Q.pz_lnd0) / 0.009408518) * (1 + math.erf(((0.115 - Q.pz_lnd0) / 0.009408518) / math.sqrt(2)))) * (0.01282622 * 0.5 * ((Q.z_neutral_had - 0.12) / 0.01282622) * (1 + math.erf(((Q.z_neutral_had - 0.12) / 0.01282622) / math.sqrt(2))))
    z += -16.8 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.0926) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.0926) / 0.02108391) / math.sqrt(2)))) * (0.01372201 * 0.5 * ((Q.pz_lnkt1 - 0.0651) / 0.01372201) * (1 + math.erf(((Q.pz_lnkt1 - 0.0651) / 0.01372201) / math.sqrt(2))))
    return z


def neuron_71(Q):
    z = 3.03e-06
    return z


def neuron_72(Q):
    z = -1.16e-06
    return z


def neuron_73(Q):
    z = -1.7e-05
    return z


def neuron_74(Q):
    z = -4.49e-06
    return z


def neuron_75(Q):
    z = 1.97e-06
    return z


def neuron_76(Q):
    z = -5.54e-06
    return z


def neuron_77(Q):
    z = -3.09e-06
    return z


def neuron_78(Q):
    z = -0.102
    z += 2.94 * (0.009441413 * 0.5 * ((0.488 - Q.N2_b05) / 0.009441413) * (1 + math.erf(((0.488 - Q.N2_b05) / 0.009441413) / math.sqrt(2))))
    z += 46500.0 * (2.54381e-06 * 0.5 * ((3.49e-06 - Q.e4) / 2.54381e-06) * (1 + math.erf(((3.49e-06 - Q.e4) / 2.54381e-06) / math.sqrt(2))))
    z += -0.145 * (0.2260337 * 0.5 * ((-8.43 - Q.ktd_ln_d34) / 0.2260337) * (1 + math.erf(((-8.43 - Q.ktd_ln_d34) / 0.2260337) / math.sqrt(2))))
    z += -0.0691 * (7.585662 * 0.5 * ((17.8 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((17.8 - Q.lep_ptrel) / 7.585662) / math.sqrt(2))))
    z += 0.0901 * (0.3893833 * 0.5 * ((Q.lund3_lndelta - -2.76) / 0.3893833) * (1 + math.erf(((Q.lund3_lndelta - -2.76) / 0.3893833) / math.sqrt(2))))
    z += -0.00836 * (5.197 * 0.5 * ((95.5 - Q.mass) / 5.197) * (1 + math.erf(((95.5 - Q.mass) / 5.197) / math.sqrt(2))))
    z += 0.011 * (12.30925 * 0.5 * ((Q.mass_displaced3 - 37.3) / 12.30925) * (1 + math.erf(((Q.mass_displaced3 - 37.3) / 12.30925) / math.sqrt(2))))
    z += 0.00663 * (3.793098 * 0.5 * ((Q.mass_displaced5 - 8.86) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 8.86) / 3.793098) / math.sqrt(2))))
    z += 0.0188 * (17.59858 * 0.5 * ((Q.mass_top50 - 183.0) / 17.59858) * (1 + math.erf(((Q.mass_top50 - 183.0) / 17.59858) / math.sqrt(2))))
    z += -0.02 * (5.898438 * 0.5 * ((10.2 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((10.2 - Q.max_abs_d0) / 5.898438) / math.sqrt(2))))
    z += 0.00815 * (4.488781 * 0.5 * ((Q.mres_sd_mass_b0z005 - 125.0) / 4.488781) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 125.0) / 4.488781) / math.sqrt(2))))
    z += -0.0222 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 163.0) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 163.0) / 18.66842) / math.sqrt(2))))
    z += 0.0272 * (1.5 * 0.5 * ((Q.n_charged_had - 22.7) / 1.5) * (1 + math.erf(((Q.n_charged_had - 22.7) / 1.5) / math.sqrt(2))))
    z += 0.00197 * (27.0 * 0.5 * ((201.0 - Q.n_pairs_kt_above_1) / 27.0) * (1 + math.erf(((201.0 - Q.n_pairs_kt_above_1) / 27.0) / math.sqrt(2))))
    z += 0.00327 * (11.5 * 0.5 * ((99.0 - Q.n_pairs_kt_above_3) / 11.5) * (1 + math.erf(((99.0 - Q.n_pairs_kt_above_3) / 11.5) / math.sqrt(2))))
    z += 0.0969 * (1.0 * 0.5 * ((5.73 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((5.73 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += -0.0371 * (1.0 * 0.5 * ((10.1 - Q.sdb_2_n) / 1.0) * (1 + math.erf(((10.1 - Q.sdb_2_n) / 1.0) / math.sqrt(2))))
    z += 0.0648 * (2.0 * 0.5 * ((Q.sjf_2_1_n_d3 - 6.96) / 2.0) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 6.96) / 2.0) / math.sqrt(2))))
    z += 0.869 * (0.04874922 * 0.5 * ((0.13 - Q.sjf_4_1_z_d3) / 0.04874922) * (1 + math.erf(((0.13 - Q.sjf_4_1_z_d3) / 0.04874922) / math.sqrt(2))))
    z += 0.00668 * (7.585662 * 0.5 * ((20.5 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((20.5 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (0.3051516 * 0.5 * ((-7.89 - Q.ktd_ln_d34) / 0.3051516) * (1 + math.erf(((-7.89 - Q.ktd_ln_d34) / 0.3051516) / math.sqrt(2))))
    z += 4.03e-05 * (7.585662 * 0.5 * ((20.3 - Q.lep_ptrel) / 7.585662) * (1 + math.erf(((20.3 - Q.lep_ptrel) / 7.585662) / math.sqrt(2)))) * (16.61023 * 0.5 * ((616.0 - Q.sum_pt_top30) / 16.61023) * (1 + math.erf(((616.0 - Q.sum_pt_top30) / 16.61023) / math.sqrt(2))))
    z += 2.37 * (0.1486471 * 0.5 * ((0.299 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.299 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.01443111 * 0.5 * ((0.194 - Q.pz_lnd2) / 0.01443111) * (1 + math.erf(((0.194 - Q.pz_lnd2) / 0.01443111) / math.sqrt(2))))
    z += 1.63 * (0.05826336 * 0.5 * ((0.102 - Q.lepsj_2_dr) / 0.05826336) * (1 + math.erf(((0.102 - Q.lepsj_2_dr) / 0.05826336) / math.sqrt(2)))) * (0.2429919 * 0.5 * ((6.3 - Q.lne_0) / 0.2429919) * (1 + math.erf(((6.3 - Q.lne_0) / 0.2429919) / math.sqrt(2))))
    z += -0.000216 * (3.793098 * 0.5 * ((Q.mass_displaced5 - 8.44) / 3.793098) * (1 + math.erf(((Q.mass_displaced5 - 8.44) / 3.793098) / math.sqrt(2)))) * (14.29796 * 0.5 * ((109.0 - Q.mass_charged) / 14.29796) * (1 + math.erf(((109.0 - Q.mass_charged) / 14.29796) / math.sqrt(2))))
    z += 0.000119 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.49) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.49) / 1.0) / math.sqrt(2)))) * (405.0217 * 0.5 * ((779.0 - Q.jd_sum_abs_sd0_top5) / 405.0217) * (1 + math.erf(((779.0 - Q.jd_sum_abs_sd0_top5) / 405.0217) / math.sqrt(2))))
    z += -0.0996 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.27) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.27) / 1.0) / math.sqrt(2)))) * (0.02818185 * 0.5 * ((Q.sj3_pairmin_over_m - 0.0864) / 0.02818185) * (1 + math.erf(((Q.sj3_pairmin_over_m - 0.0864) / 0.02818185) / math.sqrt(2))))
    z += 0.0503 * (2.0 * 0.5 * ((11.0 - Q.n_sdz_above_2) / 2.0) * (1 + math.erf(((11.0 - Q.n_sdz_above_2) / 2.0) / math.sqrt(2)))) * (0.06215671 * 0.5 * ((0.309 - Q.sj4_dr_min) / 0.06215671) * (1 + math.erf(((0.309 - Q.sj4_dr_min) / 0.06215671) / math.sqrt(2))))
    return z


def neuron_79(Q):
    z = -3.6e-05
    return z


def neuron_80(Q):
    z = 9.71e-06
    return z


def neuron_81(Q):
    z = 0.833
    z += 3.67 * (0.04445521 * 0.5 * ((0.262 - Q.LHA) / 0.04445521) * (1 + math.erf(((0.262 - Q.LHA) / 0.04445521) / math.sqrt(2))))
    z += -0.00588 * (29.76053 * 0.5 * ((44.4 - Q.lep_iso) / 29.76053) * (1 + math.erf(((44.4 - Q.lep_iso) / 29.76053) / math.sqrt(2))))
    z += 14.2 * (0.009443246 * 0.5 * ((0.0156 - Q.lepsj_3_dr) / 0.009443246) * (1 + math.erf(((0.0156 - Q.lepsj_3_dr) / 0.009443246) / math.sqrt(2))))
    z += -0.156 * (0.2381965 * 0.5 * ((Q.lne_5 - 2.73) / 0.2381965) * (1 + math.erf(((Q.lne_5 - 2.73) / 0.2381965) / math.sqrt(2))))
    z += -0.155 * (0.5788858 * 0.5 * ((1.9 - Q.mass_displaced3) / 0.5788858) * (1 + math.erf(((1.9 - Q.mass_displaced3) / 0.5788858) / math.sqrt(2))))
    z += -0.00696 * (17.59858 * 0.5 * ((Q.mass_top50 - 175.0) / 17.59858) * (1 + math.erf(((Q.mass_top50 - 175.0) / 17.59858) / math.sqrt(2))))
    z += 0.101 * (3.705078 * 0.5 * ((6.2 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((6.2 - Q.max_abs_d0) / 3.705078) / math.sqrt(2))))
    z += -0.0152 * (5.26174 * 0.5 * ((91.1 - Q.mres_sd_mass_b2z01) / 5.26174) * (1 + math.erf(((91.1 - Q.mres_sd_mass_b2z01) / 5.26174) / math.sqrt(2))))
    z += 0.0358 * (3.499256 * 0.5 * ((121.0 - Q.mres_sd_mass_b2z01) / 3.499256) * (1 + math.erf(((121.0 - Q.mres_sd_mass_b2z01) / 3.499256) / math.sqrt(2))))
    z += -0.0199 * (19.14168 * 0.5 * ((157.0 - Q.mres_sd_mass_b2z01) / 19.14168) * (1 + math.erf(((157.0 - Q.mres_sd_mass_b2z01) / 19.14168) / math.sqrt(2))))
    z += -0.0331 * (3.5 * 0.5 * ((14.5 - Q.n_dr_0p4_up) / 3.5) * (1 + math.erf(((14.5 - Q.n_dr_0p4_up) / 3.5) / math.sqrt(2))))
    z += 0.107 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 3.46) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 3.46) / 1.0) / math.sqrt(2))))
    z += 0.137 * (1.0 * 0.5 * ((6.11 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((6.11 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += -0.0894 * (1.0 * 0.5 * ((10.1 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((10.1 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 1.44 * (0.02426199 * 0.5 * ((0.212 - Q.sdb_2_z) / 0.02426199) * (1 + math.erf(((0.212 - Q.sdb_2_z) / 0.02426199) / math.sqrt(2))))
    z += -0.182 * (1.0 * 0.5 * ((Q.sjf_4_n2disp - 0.94) / 1.0) * (1 + math.erf(((Q.sjf_4_n2disp - 0.94) / 1.0) / math.sqrt(2))))
    z += 2.91 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.0314) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.0314) / 0.01141967) / math.sqrt(2))))
    z += -0.617 * (0.02108391 * 0.5 * ((Q.z_displaced5 - 0.0866) / 0.02108391) * (1 + math.erf(((Q.z_displaced5 - 0.0866) / 0.02108391) / math.sqrt(2))))
    z += -0.00438 * (3.705078 * 0.5 * ((6.05 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((6.05 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (6.084951 * 0.5 * ((18.7 - Q.sip_3d_3) / 6.084951) * (1 + math.erf(((18.7 - Q.sip_3d_3) / 6.084951) / math.sqrt(2))))
    z += -0.0392 * (5.55459 * 0.5 * ((96.2 - Q.mres_sd_mass_b2z01) / 5.55459) * (1 + math.erf(((96.2 - Q.mres_sd_mass_b2z01) / 5.55459) / math.sqrt(2)))) * (0.01467775 * 0.5 * ((0.223 - Q.N2_b2) / 0.01467775) * (1 + math.erf(((0.223 - Q.N2_b2) / 0.01467775) / math.sqrt(2))))
    z += 0.857 * (1.0 * 0.5 * ((4.38 - Q.n_dr_0p4_up) / 1.0) * (1 + math.erf(((4.38 - Q.n_dr_0p4_up) / 1.0) / math.sqrt(2)))) * (0.006013287 * 0.5 * ((0.0622 - Q.pz_lnkt3) / 0.006013287) * (1 + math.erf(((0.0622 - Q.pz_lnkt3) / 0.006013287) / math.sqrt(2))))
    z += -0.0121 * (1.0 * 0.5 * ((Q.n_s3d_above_10 - 3.13) / 1.0) * (1 + math.erf(((Q.n_s3d_above_10 - 3.13) / 1.0) / math.sqrt(2)))) * (2.0 * 0.5 * ((4.33 - Q.lepsj_2_n_d3) / 2.0) * (1 + math.erf(((4.33 - Q.lepsj_2_n_d3) / 2.0) / math.sqrt(2))))
    z += 1.6 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.21) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.21) / 1.0) / math.sqrt(2)))) * (0.006796583 * 0.5 * ((0.0634 - Q.M2_b2) / 0.006796583) * (1 + math.erf(((0.0634 - Q.M2_b2) / 0.006796583) / math.sqrt(2))))
    z += 0.138 * (1.0 * 0.5 * ((9.69 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((9.69 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.235 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.235 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2))))
    z += -2.86 * (0.01559439 * 0.5 * ((0.167 - Q.pz_lnd0) / 0.01559439) * (1 + math.erf(((0.167 - Q.pz_lnd0) / 0.01559439) / math.sqrt(2)))) * (0.2297361 * 0.5 * ((Q.ak02_dr13 - 0.0205) / 0.2297361) * (1 + math.erf(((Q.ak02_dr13 - 0.0205) / 0.2297361) / math.sqrt(2))))
    z += -0.284 * (0.02426199 * 0.5 * ((0.209 - Q.sdb_2_z) / 0.02426199) * (1 + math.erf(((0.209 - Q.sdb_2_z) / 0.02426199) / math.sqrt(2)))) * (5.085197 * 0.5 * ((10.5 - Q.jd_3d_5) / 5.085197) * (1 + math.erf(((10.5 - Q.jd_3d_5) / 5.085197) / math.sqrt(2))))
    z += -0.627 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.0312) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.0312) / 0.01141967) / math.sqrt(2)))) * (1.0 * 0.5 * ((1.01 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.01 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2))))
    z += -0.233 * (0.01141967 * 0.5 * ((Q.z_displaced3 - -0.00479) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - -0.00479) / 0.01141967) / math.sqrt(2)))) * (1.5 * 0.5 * ((6.84 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((6.84 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2))))
    z += -1.46 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.0323) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.0323) / 0.01141967) / math.sqrt(2)))) * (0.0964495 * 0.5 * ((0.572 - Q.sjq_2_sumabs_k1) / 0.0964495) * (1 + math.erf(((0.572 - Q.sjq_2_sumabs_k1) / 0.0964495) / math.sqrt(2))))
    z += -5.24 * (0.01141967 * 0.5 * ((Q.z_displaced3 - 0.0405) / 0.01141967) * (1 + math.erf(((Q.z_displaced3 - 0.0405) / 0.01141967) / math.sqrt(2)))) * (0.0649922 * 0.5 * ((Q.z_charged_had - 0.267) / 0.0649922) * (1 + math.erf(((Q.z_charged_had - 0.267) / 0.0649922) / math.sqrt(2))))
    return z


def neuron_82(Q):
    z = -2.95e-06
    return z


def neuron_83(Q):
    z = -1.16
    z += -16.2 * (0.001766249 * 0.5 * ((0.0172 - Q.M3_b2) / 0.001766249) * (1 + math.erf(((0.0172 - Q.M3_b2) / 0.001766249) / math.sqrt(2))))
    z += -2.0 * (0.01556476 * 0.5 * ((0.259 - Q.N2) / 0.01556476) * (1 + math.erf(((0.259 - Q.N2) / 0.01556476) / math.sqrt(2))))
    z += 1970.0 * (1.951776e-05 * 0.5 * ((Q.ecf_g41 - 8.42e-05) / 1.951776e-05) * (1 + math.erf(((Q.ecf_g41 - 8.42e-05) / 1.951776e-05) / math.sqrt(2))))
    z += 0.138 * (3.384707 * 0.5 * ((6.65 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.65 - Q.jd_3d_5) / 3.384707) / math.sqrt(2))))
    z += 1.62 * (0.03060878 * 0.5 * ((0.0658 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0658 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2))))
    z += -0.0141 * (5.197 * 0.5 * ((Q.mass - 94.6) / 5.197) * (1 + math.erf(((Q.mass - 94.6) / 5.197) / math.sqrt(2))))
    z += 0.0124 * (3.153038 * 0.5 * ((Q.mass - 121.0) / 3.153038) * (1 + math.erf(((Q.mass - 121.0) / 3.153038) / math.sqrt(2))))
    z += -0.0501 * (15.47377 * 0.5 * ((54.8 - Q.mass) / 15.47377) * (1 + math.erf(((54.8 - Q.mass) / 15.47377) / math.sqrt(2))))
    z += 0.0124 * (16.90428 * 0.5 * ((165.0 - Q.mass) / 16.90428) * (1 + math.erf(((165.0 - Q.mass) / 16.90428) / math.sqrt(2))))
    z += 0.0184 * (10.14805 * 0.5 * ((26.4 - Q.mass_displaced3) / 10.14805) * (1 + math.erf(((26.4 - Q.mass_displaced3) / 10.14805) / math.sqrt(2))))
    z += -0.0174 * (3.373048 * 0.5 * ((Q.mass_top15 - 82.9) / 3.373048) * (1 + math.erf(((Q.mass_top15 - 82.9) / 3.373048) / math.sqrt(2))))
    z += 0.024 * (15.58693 * 0.5 * ((Q.mass_top40 - 159.0) / 15.58693) * (1 + math.erf(((Q.mass_top40 - 159.0) / 15.58693) / math.sqrt(2))))
    z += 0.117 * (3.705078 * 0.5 * ((5.75 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.75 - Q.max_abs_d0) / 3.705078) / math.sqrt(2))))
    z += -0.00289 * (6.036668 * 0.5 * ((85.8 - Q.mres_sd_mass_b2z01) / 6.036668) * (1 + math.erf(((85.8 - Q.mres_sd_mass_b2z01) / 6.036668) / math.sqrt(2))))
    z += 0.062 * (1.5 * 0.5 * ((9.98 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((9.98 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2))))
    z += -0.066 * (2.0 * 0.5 * ((8.03 - Q.n_s3d_above_10) / 2.0) * (1 + math.erf(((8.03 - Q.n_s3d_above_10) / 2.0) / math.sqrt(2))))
    z += 0.132 * (1.0 * 0.5 * ((3.11 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((3.11 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 0.594 * (0.2024667 * 0.5 * ((Q.pair_max_lnm2 - 7.52) / 0.2024667) * (1 + math.erf(((Q.pair_max_lnm2 - 7.52) / 0.2024667) / math.sqrt(2))))
    z += 1.6 * (0.02719315 * 0.5 * ((0.156 - Q.sdb_2_z) / 0.02719315) * (1 + math.erf(((0.156 - Q.sdb_2_z) / 0.02719315) / math.sqrt(2))))
    z += -0.512 * (0.09779513 * 0.5 * ((0.67 - Q.sdb_2_z) / 0.09779513) * (1 + math.erf(((0.67 - Q.sdb_2_z) / 0.09779513) / math.sqrt(2))))
    z += 0.0889 * (1.0 * 0.5 * ((1.99 - Q.sdb_5_n) / 1.0) * (1 + math.erf(((1.99 - Q.sdb_5_n) / 1.0) / math.sqrt(2))))
    z += 0.104 * (5.171003 * 0.5 * ((7.28 - Q.sip_3d_1) / 5.171003) * (1 + math.erf(((7.28 - Q.sip_3d_1) / 5.171003) / math.sqrt(2))))
    z += 0.0455 * (6.230332 * 0.5 * ((13.7 - Q.sip_3d_2) / 6.230332) * (1 + math.erf(((13.7 - Q.sip_3d_2) / 6.230332) / math.sqrt(2))))
    z += -0.0251 * (8.494762 * 0.5 * ((20.9 - Q.sip_3d_2) / 8.494762) * (1 + math.erf(((20.9 - Q.sip_3d_2) / 8.494762) / math.sqrt(2))))
    z += -0.0163 * (2.952984 * 0.5 * ((75.6 - Q.sj4_pair_mass_max) / 2.952984) * (1 + math.erf(((75.6 - Q.sj4_pair_mass_max) / 2.952984) / math.sqrt(2))))
    z += 0.0186 * (10.09897 * 0.5 * ((116.0 - Q.sj4_pair_mass_max) / 10.09897) * (1 + math.erf(((116.0 - Q.sj4_pair_mass_max) / 10.09897) / math.sqrt(2))))
    z += 0.0985 * (1.5 * 0.5 * ((Q.sjf_2_1_n_d3 - 5.0) / 1.5) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 5.0) / 1.5) / math.sqrt(2))))
    z += 1.88 * (0.09626377 * 0.5 * ((Q.sjf_2_1_z_d3 - 0.37) / 0.09626377) * (1 + math.erf(((Q.sjf_2_1_z_d3 - 0.37) / 0.09626377) / math.sqrt(2))))
    z += 0.19 * (1.0 * 0.5 * ((Q.sjf_2_n2disp - -0.0463) / 1.0) * (1 + math.erf(((Q.sjf_2_n2disp - -0.0463) / 1.0) / math.sqrt(2))))
    z += -0.00317 * (20.79165 * 0.5 * ((48.7 - Q.sv_1_sd0_sum) / 20.79165) * (1 + math.erf(((48.7 - Q.sv_1_sd0_sum) / 20.79165) / math.sqrt(2))))
    z += -14.9 * (0.00905862 * 0.5 * ((0.0653 - Q.tau5) / 0.00905862) * (1 + math.erf(((0.0653 - Q.tau5) / 0.00905862) / math.sqrt(2))))
    z += 5.9 * (0.02099671 * 0.5 * ((0.083 - Q.z_displaced3) / 0.02099671) * (1 + math.erf(((0.083 - Q.z_displaced3) / 0.02099671) / math.sqrt(2))))
    z += -4.09 * (0.01159153 * 0.5 * ((0.0445 - Q.z_dr_0p4_up) / 0.01159153) * (1 + math.erf(((0.0445 - Q.z_dr_0p4_up) / 0.01159153) / math.sqrt(2))))
    z += -4.31 * (1.951776e-05 * 0.5 * ((Q.ecf_g41 - 8.18e-05) / 1.951776e-05) * (1 + math.erf(((Q.ecf_g41 - 8.18e-05) / 1.951776e-05) / math.sqrt(2)))) * (57.98295 * 0.5 * ((768.0 - Q.sum_pt_top40) / 57.98295) * (1 + math.erf(((768.0 - Q.sum_pt_top40) / 57.98295) / math.sqrt(2))))
    z += 1.18 * (0.2916549 * 0.5 * ((Q.lne_21 - 1.09) / 0.2916549) * (1 + math.erf(((Q.lne_21 - 1.09) / 0.2916549) / math.sqrt(2)))) * (0.02507469 * 0.5 * ((0.155 - Q.sj4_zsoft) / 0.02507469) * (1 + math.erf(((0.155 - Q.sj4_zsoft) / 0.02507469) / math.sqrt(2))))
    z += -0.0841 * (6.345399 * 0.5 * ((Q.mass_top40 - 80.0) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 80.0) / 6.345399) / math.sqrt(2)))) * (0.01771551 * 0.5 * ((0.119 - Q.C2_b2) / 0.01771551) * (1 + math.erf(((0.119 - Q.C2_b2) / 0.01771551) / math.sqrt(2))))
    z += 0.305 * (15.58693 * 0.5 * ((Q.mass_top40 - 154.0) / 15.58693) * (1 + math.erf(((Q.mass_top40 - 154.0) / 15.58693) / math.sqrt(2)))) * (0.01417841 * 0.5 * ((0.106 - Q.C2_b2) / 0.01417841) * (1 + math.erf(((0.106 - Q.C2_b2) / 0.01417841) / math.sqrt(2))))
    z += -0.0333 * (6.345399 * 0.5 * ((Q.mass_top40 - 78.5) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 78.5) / 6.345399) / math.sqrt(2)))) * (0.1060766 * 0.5 * ((0.224 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.224 - Q.lep_z) / 0.1060766) / math.sqrt(2))))
    z += 0.00802 * (6.345399 * 0.5 * ((Q.mass_top40 - 81.4) / 6.345399) * (1 + math.erf(((Q.mass_top40 - 81.4) / 6.345399) / math.sqrt(2)))) * (0.01759885 * 0.5 * ((Q.sj3_z1 - 0.492) / 0.01759885) * (1 + math.erf(((Q.sj3_z1 - 0.492) / 0.01759885) / math.sqrt(2))))
    z += -0.0161 * (8.537109 * 0.5 * ((17.0 - Q.max_abs_d0) / 8.537109) * (1 + math.erf(((17.0 - Q.max_abs_d0) / 8.537109) / math.sqrt(2)))) * (0.1631384 * 0.5 * ((0.618 - Q.D3) / 0.1631384) * (1 + math.erf(((0.618 - Q.D3) / 0.1631384) / math.sqrt(2))))
    z += -0.0334 * (3.705078 * 0.5 * ((5.98 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.98 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (3.384707 * 0.5 * ((6.53 - Q.jd_3d_5) / 3.384707) * (1 + math.erf(((6.53 - Q.jd_3d_5) / 3.384707) / math.sqrt(2))))
    z += -0.0246 * (3.705078 * 0.5 * ((5.82 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.82 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.5 * 0.5 * ((2.0 - Q.lepsj_2_n_d3) / 1.5) * (1 + math.erf(((2.0 - Q.lepsj_2_n_d3) / 1.5) / math.sqrt(2))))
    z += 0.0273 * (4.245069 * 0.5 * ((125.0 - Q.mres_sd_mass_b2z01) / 4.245069) * (1 + math.erf(((125.0 - Q.mres_sd_mass_b2z01) / 4.245069) / math.sqrt(2)))) * (1.0 * 0.5 * ((-0.000129 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((-0.000129 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.0304 * (1.5 * 0.5 * ((10.5 - Q.n_dr_0p4_up) / 1.5) * (1 + math.erf(((10.5 - Q.n_dr_0p4_up) / 1.5) / math.sqrt(2)))) * (0.3391285 * 0.5 * ((Q.max_dr - 0.922) / 0.3391285) * (1 + math.erf(((Q.max_dr - 0.922) / 0.3391285) / math.sqrt(2))))
    z += -0.155 * (1.0 * 0.5 * ((5.01 - Q.n_s3d_above_10) / 1.0) * (1 + math.erf(((5.01 - Q.n_s3d_above_10) / 1.0) / math.sqrt(2)))) * (0.02426199 * 0.5 * ((Q.sdb_2_z - 0.208) / 0.02426199) * (1 + math.erf(((Q.sdb_2_z - 0.208) / 0.02426199) / math.sqrt(2))))
    z += 0.00104 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 5.07) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 5.07) / 1.0) / math.sqrt(2)))) * (248.3135 * 0.5 * ((182.0 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((182.0 - Q.sip_3d_3) / 248.3135) / math.sqrt(2))))
    z += 0.318 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.99) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.99) / 1.0) / math.sqrt(2)))) * (0.02937692 * 0.5 * ((0.496 - Q.sj3_pairmin_over_m) / 0.02937692) * (1 + math.erf(((0.496 - Q.sj3_pairmin_over_m) / 0.02937692) / math.sqrt(2))))
    z += 0.0578 * (0.04874922 * 0.5 * ((0.126 - Q.sjf_4_1_z_d3) / 0.04874922) * (1 + math.erf(((0.126 - Q.sjf_4_1_z_d3) / 0.04874922) / math.sqrt(2)))) * (6.566474 * 0.5 * ((13.2 - Q.lepsj_3_mass) / 6.566474) * (1 + math.erf(((13.2 - Q.lepsj_3_mass) / 6.566474) / math.sqrt(2))))
    return z


def neuron_84(Q):
    z = -7.43e-05
    return z


def neuron_85(Q):
    z = -8.15e-07
    return z


def neuron_86(Q):
    z = -1.71e-05
    return z


def neuron_87(Q):
    z = 1.61e-06
    return z


def neuron_88(Q):
    z = 3.03e-07
    return z


def neuron_89(Q):
    z = 1.74e-06
    return z


def neuron_90(Q):
    z = -0.00023
    return z


def neuron_91(Q):
    z = 3.99e-06
    return z


def neuron_92(Q):
    z = -6.72e-06
    return z


def neuron_93(Q):
    z = 8.47e-07
    return z


def neuron_94(Q):
    z = -2.32e-06
    return z


def neuron_95(Q):
    z = 4.35e-07
    return z


def neuron_96(Q):
    z = 4.23e-07
    return z


def neuron_97(Q):
    z = -1.5
    z += -2.12 * (0.06488792 * 0.5 * ((Q.dc_split2_dr - 0.281) / 0.06488792) * (1 + math.erf(((Q.dc_split2_dr - 0.281) / 0.06488792) / math.sqrt(2))))
    z += -3380.0 * (3.838712e-05 * 0.5 * ((0.00013 - Q.e3_b2) / 3.838712e-05) * (1 + math.erf(((0.00013 - Q.e3_b2) / 3.838712e-05) / math.sqrt(2))))
    z += 0.121 * (0.1277784 * 0.5 * ((Q.lund3_lndelta - -1.89) / 0.1277784) * (1 + math.erf(((Q.lund3_lndelta - -1.89) / 0.1277784) / math.sqrt(2))))
    z += -0.153 * (0.1064494 * 0.5 * ((3.63 - Q.lund_max_lnkt) / 0.1064494) * (1 + math.erf(((3.63 - Q.lund_max_lnkt) / 0.1064494) / math.sqrt(2))))
    z += 0.00799 * (3.289335 * 0.5 * ((Q.mass - 125.0) / 3.289335) * (1 + math.erf(((Q.mass - 125.0) / 3.289335) / math.sqrt(2))))
    z += -0.0303 * (16.90428 * 0.5 * ((Q.mass - 164.0) / 16.90428) * (1 + math.erf(((Q.mass - 164.0) / 16.90428) / math.sqrt(2))))
    z += 0.0299 * (5.286881 * 0.5 * ((101.0 - Q.mass) / 5.286881) * (1 + math.erf(((101.0 - Q.mass) / 5.286881) / math.sqrt(2))))
    z += 0.0221 * (6.875137 * 0.5 * ((18.9 - Q.mass_displaced3) / 6.875137) * (1 + math.erf(((18.9 - Q.mass_displaced3) / 6.875137) / math.sqrt(2))))
    z += 0.0124 * (12.30925 * 0.5 * ((39.1 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.1 - Q.mass_displaced3) / 12.30925) / math.sqrt(2))))
    z += 0.00575 * (4.879578 * 0.5 * ((121.0 - Q.mass_top30) / 4.879578) * (1 + math.erf(((121.0 - Q.mass_top30) / 4.879578) / math.sqrt(2))))
    z += 0.00636 * (5.119423 * 0.5 * ((Q.mass_top5 - 63.5) / 5.119423) * (1 + math.erf(((Q.mass_top5 - 63.5) / 5.119423) / math.sqrt(2))))
    z += 0.00969 * (4.883139 * 0.5 * ((Q.mass_top50 - 89.5) / 4.883139) * (1 + math.erf(((Q.mass_top50 - 89.5) / 4.883139) / math.sqrt(2))))
    z += -0.0141 * (5.5191 * 0.5 * ((Q.mres_pruned_mass - 76.5) / 5.5191) * (1 + math.erf(((Q.mres_pruned_mass - 76.5) / 5.5191) / math.sqrt(2))))
    z += 0.027 * (5.725739 * 0.5 * ((Q.mres_pruned_mass - 125.0) / 5.725739) * (1 + math.erf(((Q.mres_pruned_mass - 125.0) / 5.725739) / math.sqrt(2))))
    z += -0.00221 * (22.10989 * 0.5 * ((172.0 - Q.mres_pruned_mass) / 22.10989) * (1 + math.erf(((172.0 - Q.mres_pruned_mass) / 22.10989) / math.sqrt(2))))
    z += 0.00388 * (19.15449 * 0.5 * ((Q.mres_sd_mass_b0z02 - 134.0) / 19.15449) * (1 + math.erf(((Q.mres_sd_mass_b0z02 - 134.0) / 19.15449) / math.sqrt(2))))
    z += -0.0138 * (20.07181 * 0.5 * ((Q.mres_sd_mass_b1z01 - 180.0) / 20.07181) * (1 + math.erf(((Q.mres_sd_mass_b1z01 - 180.0) / 20.07181) / math.sqrt(2))))
    z += 0.0189 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 53.2) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 53.2) / 19.93276) / math.sqrt(2))))
    z += -0.0787 * (6.036668 * 0.5 * ((Q.mres_sd_mass_b2z01 - 75.7) / 6.036668) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 75.7) / 6.036668) / math.sqrt(2))))
    z += 0.0624 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 90.9) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 90.9) / 5.26174) / math.sqrt(2))))
    z += 0.0786 * (6.991655 * 0.5 * ((Q.mres_sd_mass_b2z01 - 130.0) / 6.991655) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 130.0) / 6.991655) / math.sqrt(2))))
    z += -0.0804 * (14.05251 * 0.5 * ((Q.mres_sd_mass_b2z01 - 139.0) / 14.05251) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 139.0) / 14.05251) / math.sqrt(2))))
    z += -0.0344 * (1.0 * 0.5 * ((6.01 - Q.n_neutral_had) / 1.0) * (1 + math.erf(((6.01 - Q.n_neutral_had) / 1.0) / math.sqrt(2))))
    z += 0.00207 * (35.5 * 0.5 * ((284.0 - Q.n_pairs_kt_above_1) / 35.5) * (1 + math.erf(((284.0 - Q.n_pairs_kt_above_1) / 35.5) / math.sqrt(2))))
    z += -0.182 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 2.98) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 2.98) / 1.0) / math.sqrt(2))))
    z += 0.124 * (1.0 * 0.5 * ((Q.n_sd0_above_3 - 1.96) / 1.0) * (1 + math.erf(((Q.n_sd0_above_3 - 1.96) / 1.0) / math.sqrt(2))))
    z += -0.0156 * (2.991814 * 0.5 * ((65.8 - Q.nca_sj4_pair_mass_2nd) / 2.991814) * (1 + math.erf(((65.8 - Q.nca_sj4_pair_mass_2nd) / 2.991814) / math.sqrt(2))))
    z += 0.0119 * (10.53967 * 0.5 * ((93.8 - Q.nca_sj4_pair_mass_2nd) / 10.53967) * (1 + math.erf(((93.8 - Q.nca_sj4_pair_mass_2nd) / 10.53967) / math.sqrt(2))))
    z += -0.000966 * (289.0822 * 0.5 * ((446.0 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((446.0 - Q.sip_3d_2) / 289.0822) / math.sqrt(2))))
    z += 0.241 * (2.077045 * 0.5 * ((1.84 - Q.sj2_mass2) / 2.077045) * (1 + math.erf(((1.84 - Q.sj2_mass2) / 2.077045) / math.sqrt(2))))
    z += 0.00965 * (6.93786 * 0.5 * ((Q.sj3_pair_mass_max - 120.0) / 6.93786) * (1 + math.erf(((Q.sj3_pair_mass_max - 120.0) / 6.93786) / math.sqrt(2))))
    z += -0.00681 * (3.93783 * 0.5 * ((73.1 - Q.sj3_pair_mass_max) / 3.93783) * (1 + math.erf(((73.1 - Q.sj3_pair_mass_max) / 3.93783) / math.sqrt(2))))
    z += -0.00265 * (175.2971 * 0.5 * ((84.2 - Q.sv_2_sd0_sum) / 175.2971) * (1 + math.erf(((84.2 - Q.sv_2_sd0_sum) / 175.2971) / math.sqrt(2))))
    z += 5.2 * (0.01301382 * 0.5 * ((0.119 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.119 - Q.tau2) / 0.01301382) / math.sqrt(2))))
    z += 0.466 * (0.0297357 * 0.5 * ((0.651 - Q.tau32) / 0.0297357) * (1 + math.erf(((0.651 - Q.tau32) / 0.0297357) / math.sqrt(2))))
    z += -0.982 * (0.0331444 * 0.5 * ((Q.z_displaced5 - 0.171) / 0.0331444) * (1 + math.erf(((Q.z_displaced5 - 0.171) / 0.0331444) / math.sqrt(2))))
    z += 1.59 * (0.02188437 * 0.5 * ((Q.z_top30_slots - 0.896) / 0.02188437) * (1 + math.erf(((Q.z_top30_slots - 0.896) / 0.02188437) / math.sqrt(2))))
    z += 0.0592 * (1.0 * 0.5 * ((1.01 - Q.ak02_2_n_lep) / 1.0) * (1 + math.erf(((1.01 - Q.ak02_2_n_lep) / 1.0) / math.sqrt(2)))) * (0.1648594 * 0.5 * ((Q.D2 - 0.837) / 0.1648594) * (1 + math.erf(((Q.D2 - 0.837) / 0.1648594) / math.sqrt(2))))
    z += -5.14 * (0.0002686389 * 0.5 * ((0.000418 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000418 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (115.0698 * 0.5 * ((174.0 - Q.jd_3d_4) / 115.0698) * (1 + math.erf(((174.0 - Q.jd_3d_4) / 115.0698) / math.sqrt(2))))
    z += 917.0 * (0.0002686389 * 0.5 * ((0.00041 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.00041 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (0.08904938 * 0.5 * ((Q.jet_charge_k03 - -0.0536) / 0.08904938) * (1 + math.erf(((Q.jet_charge_k03 - -0.0536) / 0.08904938) / math.sqrt(2))))
    z += 89.5 * (0.0002686389 * 0.5 * ((0.000404 - Q.e3_b2) / 0.0002686389) * (1 + math.erf(((0.000404 - Q.e3_b2) / 0.0002686389) / math.sqrt(2)))) * (1.0 * 0.5 * ((9.85 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((9.85 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += -0.0532 * (3.289335 * 0.5 * ((Q.mass - 127.0) / 3.289335) * (1 + math.erf(((Q.mass - 127.0) / 3.289335) / math.sqrt(2)))) * (0.01084662 * 0.5 * ((Q.pz_lnkt4 - 0.0251) / 0.01084662) * (1 + math.erf(((Q.pz_lnkt4 - 0.0251) / 0.01084662) / math.sqrt(2))))
    z += 0.0126 * (12.30925 * 0.5 * ((38.6 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((38.6 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (0.03286901 * 0.5 * ((0.489 - Q.ak02_dr12) / 0.03286901) * (1 + math.erf(((0.489 - Q.ak02_dr12) / 0.03286901) / math.sqrt(2))))
    z += 8.56e-06 * (12.30925 * 0.5 * ((39.2 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.2 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (23.09414 * 0.5 * ((Q.sum_pt_top5 - 220.0) / 23.09414) * (1 + math.erf(((Q.sum_pt_top5 - 220.0) / 23.09414) / math.sqrt(2))))
    z += -0.00412 * (4.879578 * 0.5 * ((119.0 - Q.mass_top30) / 4.879578) * (1 + math.erf(((119.0 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_min12_n_disp3 - 0.00984) / 1.0) * (1 + math.erf(((Q.ak02_min12_n_disp3 - 0.00984) / 1.0) / math.sqrt(2))))
    z += -0.0151 * (4.879578 * 0.5 * ((116.0 - Q.mass_top30) / 4.879578) * (1 + math.erf(((116.0 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.n_muon - 0.925) / 1.0) * (1 + math.erf(((Q.n_muon - 0.925) / 1.0) / math.sqrt(2))))
    z += 5.82e-06 * (4.879578 * 0.5 * ((120.0 - Q.mass_top30) / 4.879578) * (1 + math.erf(((120.0 - Q.mass_top30) / 4.879578) / math.sqrt(2)))) * (52.10352 * 0.5 * ((Q.sum_e - 931.0) / 52.10352) * (1 + math.erf(((Q.sum_e - 931.0) / 52.10352) / math.sqrt(2))))
    z += 81.8 * (5.294312 * 0.5 * ((Q.mres_sd_mass_b2z01 - 104.0) / 5.294312) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 104.0) / 5.294312) / math.sqrt(2)))) * (3.2258e-08 * 0.5 * ((Q.e4_b2 - -1.71e-08) / 3.2258e-08) * (1 + math.erf(((Q.e4_b2 - -1.71e-08) / 3.2258e-08) / math.sqrt(2))))
    z += -0.000871 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 89.7) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 89.7) / 5.26174) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.5 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.5 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += 0.0172 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 51.7) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 51.7) / 19.93276) / math.sqrt(2)))) * (0.0277049 * 0.5 * ((0.237 - Q.z_neutral_had) / 0.0277049) * (1 + math.erf(((0.237 - Q.z_neutral_had) / 0.0277049) / math.sqrt(2))))
    z += 0.000307 * (289.0822 * 0.5 * ((449.0 - Q.sip_3d_2) / 289.0822) * (1 + math.erf(((449.0 - Q.sip_3d_2) / 289.0822) / math.sqrt(2)))) * (1.929175 * 0.5 * ((4.01 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((4.01 - Q.jd_3d_5) / 1.929175) / math.sqrt(2))))
    z += -9.34 * (0.01301382 * 0.5 * ((0.119 - Q.tau2) / 0.01301382) * (1 + math.erf(((0.119 - Q.tau2) / 0.01301382) / math.sqrt(2)))) * (0.05096652 * 0.5 * ((0.128 - Q.sjq_2_prod_k05) / 0.05096652) * (1 + math.erf(((0.128 - Q.sjq_2_prod_k05) / 0.05096652) / math.sqrt(2))))
    return z


def neuron_98(Q):
    z = 4.97e-07
    return z


def neuron_99(Q):
    z = -0.00193
    return z


def neuron_100(Q):
    z = -6.27e-06
    return z


def neuron_101(Q):
    z = 8.88e-06
    return z


def neuron_102(Q):
    z = 5.49e-06
    return z


def neuron_103(Q):
    z = -3.42e-06
    return z


def neuron_104(Q):
    z = -0.954
    z += -0.0122 * Q.n_charged_pt_above_10
    z += -2.8 * (0.009790654 * 0.5 * ((Q.C2_b05 - 0.224) / 0.009790654) * (1 + math.erf(((Q.C2_b05 - 0.224) / 0.009790654) / math.sqrt(2))))
    z += 1.83 * (0.01129976 * 0.5 * ((Q.N2 - 0.335) / 0.01129976) * (1 + math.erf(((Q.N2 - 0.335) / 0.01129976) / math.sqrt(2))))
    z += 0.646 * (0.07989133 * 0.5 * ((Q.N3_b05 - 0.843) / 0.07989133) * (1 + math.erf(((Q.N3_b05 - 0.843) / 0.07989133) / math.sqrt(2))))
    z += -0.022 * (19.88146 * 0.5 * ((10.5 - Q.dc_2_jp) / 19.88146) * (1 + math.erf(((10.5 - Q.dc_2_jp) / 19.88146) / math.sqrt(2))))
    z += 0.045 * (1.5 * 0.5 * ((6.05 - Q.jd_n_d3_pt1) / 1.5) * (1 + math.erf(((6.05 - Q.jd_n_d3_pt1) / 1.5) / math.sqrt(2))))
    z += -0.46 * (0.6671377 * 0.5 * ((0.458 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.458 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += 0.0208 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.84) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.84) / 4.308625) / math.sqrt(2))))
    z += 0.125 * (1.0 * 0.5 * ((1.01 - Q.lepsj_2_n_d3) / 1.0) * (1 + math.erf(((1.01 - Q.lepsj_2_n_d3) / 1.0) / math.sqrt(2))))
    z += 0.263 * (0.06999318 * 0.5 * ((Q.lund2_lndelta - -1.06) / 0.06999318) * (1 + math.erf(((Q.lund2_lndelta - -1.06) / 0.06999318) / math.sqrt(2))))
    z += 0.511 * (0.08021924 * 0.5 * ((Q.lund_max_lndelta - -0.623) / 0.08021924) * (1 + math.erf(((Q.lund_max_lndelta - -0.623) / 0.08021924) / math.sqrt(2))))
    z += 0.0262 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2))))
    z += 0.00413 * (11.52979 * 0.5 * ((99.3 - Q.mass_charged) / 11.52979) * (1 + math.erf(((99.3 - Q.mass_charged) / 11.52979) / math.sqrt(2))))
    z += 0.0983 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 3.23) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 3.23) / 1.000219) / math.sqrt(2))))
    z += -0.0801 * (1.566771 * 0.5 * ((Q.mass_displaced3 - 4.44) / 1.566771) * (1 + math.erf(((Q.mass_displaced3 - 4.44) / 1.566771) / math.sqrt(2))))
    z += 0.0108 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 159.0) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 159.0) / 18.66842) / math.sqrt(2))))
    z += 0.0165 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 55.3) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 55.3) / 19.93276) / math.sqrt(2))))
    z += -0.0194 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.0) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.0) / 5.26174) / math.sqrt(2))))
    z += 0.00533 * (4.590131 * 0.5 * ((107.0 - Q.mres_sd_mass_b2z01) / 4.590131) * (1 + math.erf(((107.0 - Q.mres_sd_mass_b2z01) / 4.590131) / math.sqrt(2))))
    z += -0.0132 * (6.5 * 0.5 * ((40.7 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((40.7 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2))))
    z += 0.00706 * (8.295007 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 83.4) / 8.295007) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 83.4) / 8.295007) / math.sqrt(2))))
    z += 0.604 * (0.171005 * 0.5 * ((-1.36 - Q.pair_mean_lndelta) / 0.171005) * (1 + math.erf(((-1.36 - Q.pair_mean_lndelta) / 0.171005) / math.sqrt(2))))
    z += -2.93 * (0.01003572 * 0.5 * ((Q.pz_lnd0 - 0.0503) / 0.01003572) * (1 + math.erf(((Q.pz_lnd0 - 0.0503) / 0.01003572) / math.sqrt(2))))
    z += 0.00544 * (12.49365 * 0.5 * ((Q.sj4_pair_mass_max - 127.0) / 12.49365) * (1 + math.erf(((Q.sj4_pair_mass_max - 127.0) / 12.49365) / math.sqrt(2))))
    z += -7.63 * (0.01171084 * 0.5 * ((0.0258 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.0258 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2))))
    z += 0.247 * (0.1799842 * 0.5 * ((2.1 - Q.sjf_2_2_max3d) / 0.1799842) * (1 + math.erf(((2.1 - Q.sjf_2_2_max3d) / 0.1799842) / math.sqrt(2))))
    z += -0.109 * (3.069852 * 0.5 * ((4.66 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.66 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2))))
    z += 0.00798 * (19.4821 * 0.5 * ((31.1 - Q.sjf_2_2_max3d) / 19.4821) * (1 + math.erf(((31.1 - Q.sjf_2_2_max3d) / 19.4821) / math.sqrt(2))))
    z += -0.103 * (1.5 * 0.5 * ((Q.sjf_3_1_n_d3 - 3.94) / 1.5) * (1 + math.erf(((Q.sjf_3_1_n_d3 - 3.94) / 1.5) / math.sqrt(2))))
    z += 0.0182 * (1.5 * 0.5 * ((18.1 - Q.sjq_2_1_nch) / 1.5) * (1 + math.erf(((18.1 - Q.sjq_2_1_nch) / 1.5) / math.sqrt(2))))
    z += -0.708 * (0.03993816 * 0.5 * ((0.469 - Q.tau32_b2) / 0.03993816) * (1 + math.erf(((0.469 - Q.tau32_b2) / 0.03993816) / math.sqrt(2))))
    z += 48.1 * (0.002048025 * 0.5 * ((0.0207 - Q.tau5) / 0.002048025) * (1 + math.erf(((0.0207 - Q.tau5) / 0.002048025) / math.sqrt(2))))
    z += -0.502 * (0.05354637 * 0.5 * ((0.775 - Q.z_charged_had) / 0.05354637) * (1 + math.erf(((0.775 - Q.z_charged_had) / 0.05354637) / math.sqrt(2))))
    z += -11.3 * (0.00390151 * 0.5 * ((0.0733 - Q.M2) / 0.00390151) * (1 + math.erf(((0.0733 - Q.M2) / 0.00390151) / math.sqrt(2)))) * (0.04921085 * 0.5 * ((Q.N3_b05 - 0.756) / 0.04921085) * (1 + math.erf(((Q.N3_b05 - 0.756) / 0.04921085) / math.sqrt(2))))
    z += -0.231 * (4.308625 * 0.5 * ((Q.lep_ptrel - 6.97) / 4.308625) * (1 + math.erf(((Q.lep_ptrel - 6.97) / 4.308625) / math.sqrt(2)))) * (0.02843722 * 0.5 * ((0.102 - Q.ak02_3_z) / 0.02843722) * (1 + math.erf(((0.102 - Q.ak02_3_z) / 0.02843722) / math.sqrt(2))))
    z += -0.00496 * (13.49273 * 0.5 * ((149.0 - Q.mass) / 13.49273) * (1 + math.erf(((149.0 - Q.mass) / 13.49273) / math.sqrt(2)))) * (0.07882828 * 0.5 * ((Q.sjq_2_sumabs_k03 - 0.743) / 0.07882828) * (1 + math.erf(((Q.sjq_2_sumabs_k03 - 0.743) / 0.07882828) / math.sqrt(2))))
    z += -8.3e-06 * (13.49273 * 0.5 * ((149.0 - Q.mass) / 13.49273) * (1 + math.erf(((149.0 - Q.mass) / 13.49273) / math.sqrt(2)))) * (202.2502 * 0.5 * ((1400.0 - Q.sum_e) / 202.2502) * (1 + math.erf(((1400.0 - Q.sum_e) / 202.2502) / math.sqrt(2))))
    z += 0.058 * (1.000219 * 0.5 * ((Q.mass_displaced3 - 1.6) / 1.000219) * (1 + math.erf(((Q.mass_displaced3 - 1.6) / 1.000219) / math.sqrt(2)))) * (0.02843722 * 0.5 * ((0.101 - Q.ak02_3_z) / 0.02843722) * (1 + math.erf(((0.101 - Q.ak02_3_z) / 0.02843722) / math.sqrt(2))))
    z += -6.88e+07 * (6.58003 * 0.5 * ((Q.mass_top30 - 127.0) / 6.58003) * (1 + math.erf(((Q.mass_top30 - 127.0) / 6.58003) / math.sqrt(2)))) * (0.9957982 * 0.5 * ((-5.57 - Q.sip_3d_2) / 0.9957982) * (1 + math.erf(((-5.57 - Q.sip_3d_2) / 0.9957982) / math.sqrt(2))))
    z += -0.000381 * (18.66842 * 0.5 * ((Q.mres_sd_mass_b0z005 - 160.0) / 18.66842) * (1 + math.erf(((Q.mres_sd_mass_b0z005 - 160.0) / 18.66842) / math.sqrt(2)))) * (1.842068 * 0.5 * ((Q.lnerel_78 - -18.4) / 1.842068) * (1 + math.erf(((Q.lnerel_78 - -18.4) / 1.842068) / math.sqrt(2))))
    z += -0.00384 * (19.93276 * 0.5 * ((Q.mres_sd_mass_b2z01 - 54.7) / 19.93276) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 54.7) / 19.93276) / math.sqrt(2)))) * (0.05887118 * 0.5 * ((0.388 - Q.ak02_dr13) / 0.05887118) * (1 + math.erf(((0.388 - Q.ak02_dr13) / 0.05887118) / math.sqrt(2))))
    z += 0.00675 * (6.5 * 0.5 * ((40.7 - Q.n_pairs_kt_above_3) / 6.5) * (1 + math.erf(((40.7 - Q.n_pairs_kt_above_3) / 6.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.994 - Q.kt2_2_n_lep) / 1.0) * (1 + math.erf(((0.994 - Q.kt2_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.105 * (5.5 * 0.5 * ((58.2 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.2 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (0.04737588 * 0.5 * ((0.18 - Q.ak02_3_z) / 0.04737588) * (1 + math.erf(((0.18 - Q.ak02_3_z) / 0.04737588) / math.sqrt(2))))
    z += 0.00515 * (5.5 * 0.5 * ((58.4 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((58.4 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (0.194914 * 0.5 * ((3.65 - Q.pair_max_lnkt) / 0.194914) * (1 + math.erf(((3.65 - Q.pair_max_lnkt) / 0.194914) / math.sqrt(2))))
    z += -0.00136 * (5.5 * 0.5 * ((57.3 - Q.n_pt_above_1) / 5.5) * (1 + math.erf(((57.3 - Q.n_pt_above_1) / 5.5) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.sjf_2_1_n_d5 - 1.12) / 1.0) * (1 + math.erf(((Q.sjf_2_1_n_d5 - 1.12) / 1.0) / math.sqrt(2))))
    z += -237.0 * (8.295007 * 0.5 * ((Q.nca_sj4_pair_mass_2nd - 84.4) / 8.295007) * (1 + math.erf(((Q.nca_sj4_pair_mass_2nd - 84.4) / 8.295007) / math.sqrt(2)))) * (1.512315e-07 * 0.5 * ((Q.e4 - 5.35e-07) / 1.512315e-07) * (1 + math.erf(((Q.e4 - 5.35e-07) / 1.512315e-07) / math.sqrt(2))))
    z += -0.0088 * (1.0 * 0.5 * ((Q.sdb_2_n - 9.13) / 1.0) * (1 + math.erf(((Q.sdb_2_n - 9.13) / 1.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((3.02 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((3.02 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2))))
    z += 0.185 * (0.01171084 * 0.5 * ((0.0277 - Q.sjf_2_1_z_d3) / 0.01171084) * (1 + math.erf(((0.0277 - Q.sjf_2_1_z_d3) / 0.01171084) / math.sqrt(2)))) * (95.61515 * 0.5 * ((41.3 - Q.dc_2_jp) / 95.61515) * (1 + math.erf(((41.3 - Q.dc_2_jp) / 95.61515) / math.sqrt(2))))
    z += -0.000256 * (3.069852 * 0.5 * ((4.46 - Q.sjf_2_2_max3d) / 3.069852) * (1 + math.erf(((4.46 - Q.sjf_2_2_max3d) / 3.069852) / math.sqrt(2)))) * (164.9991 * 0.5 * ((216.0 - Q.sip_3d_2) / 164.9991) * (1 + math.erf(((216.0 - Q.sip_3d_2) / 164.9991) / math.sqrt(2))))
    z += -1.33e-05 * (1.5 * 0.5 * ((18.0 - Q.sjq_2_1_nch) / 1.5) * (1 + math.erf(((18.0 - Q.sjq_2_1_nch) / 1.5) / math.sqrt(2)))) * (777.2244 * 0.5 * ((1040.0 - Q.sv_1_sd0_sum) / 777.2244) * (1 + math.erf(((1040.0 - Q.sv_1_sd0_sum) / 777.2244) / math.sqrt(2))))
    return z


def neuron_105(Q):
    z = 2.73e-05
    return z


def neuron_106(Q):
    z = 2.16e-05
    return z


def neuron_107(Q):
    z = 1.16e-05
    return z


def neuron_108(Q):
    z = -4.46e-06
    return z


def neuron_109(Q):
    z = -5.17e-07
    return z


def neuron_110(Q):
    z = 3.58e-06
    return z


def neuron_111(Q):
    z = 3.67e-06
    return z


def neuron_112(Q):
    z = 5.24e-06
    return z


def neuron_113(Q):
    z = -2.67e-05
    return z


def neuron_114(Q):
    z = 4.68e-06
    return z


def neuron_115(Q):
    z = 0.179
    z += 1.2 * (0.09740396 * 0.5 * ((1.33 - Q.D2) / 0.09740396) * (1 + math.erf(((1.33 - Q.D2) / 0.09740396) / math.sqrt(2))))
    z += 10.9 * (0.008001329 * 0.5 * ((0.12 - Q.M2) / 0.008001329) * (1 + math.erf(((0.12 - Q.M2) / 0.008001329) / math.sqrt(2))))
    z += -10.2 * (0.01854575 * 0.5 * ((Q.N2 - 0.225) / 0.01854575) * (1 + math.erf(((Q.N2 - 0.225) / 0.01854575) / math.sqrt(2))))
    z += 3.9 * (0.04122765 * 0.5 * ((Q.N2_b05 - 0.303) / 0.04122765) * (1 + math.erf(((Q.N2_b05 - 0.303) / 0.04122765) / math.sqrt(2))))
    z += 7.65 * (0.008548367 * 0.5 * ((Q.N2_b05 - 0.435) / 0.008548367) * (1 + math.erf(((Q.N2_b05 - 0.435) / 0.008548367) / math.sqrt(2))))
    z += 0.768 * (0.02680828 * 0.5 * ((Q.dc_1_z - 0.616) / 0.02680828) * (1 + math.erf(((Q.dc_1_z - 0.616) / 0.02680828) / math.sqrt(2))))
    z += -0.00226 * (6.622935 * 0.5 * ((80.4 - Q.dc_split1_kt) / 6.622935) * (1 + math.erf(((80.4 - Q.dc_split1_kt) / 6.622935) / math.sqrt(2))))
    z += -1.48 * (0.009691066 * 0.5 * ((0.111 - Q.dr_0) / 0.009691066) * (1 + math.erf(((0.111 - Q.dr_0) / 0.009691066) / math.sqrt(2))))
    z += -0.132 * (0.1723618 * 0.5 * ((Q.ktd_ln_d34 - -9.52) / 0.1723618) * (1 + math.erf(((Q.ktd_ln_d34 - -9.52) / 0.1723618) / math.sqrt(2))))
    z += -16.3 * (0.006239604 * 0.5 * ((0.0214 - Q.lam2) / 0.006239604) * (1 + math.erf(((0.0214 - Q.lam2) / 0.006239604) / math.sqrt(2))))
    z += -0.898 * (0.6671377 * 0.5 * ((0.468 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.468 - Q.lep_iso) / 0.6671377) / math.sqrt(2))))
    z += 78.7 * (0.006229165 * 0.5 * ((0.00454 - Q.lep_z) / 0.006229165) * (1 + math.erf(((0.00454 - Q.lep_z) / 0.006229165) / math.sqrt(2))))
    z += 1.73 * (0.179073 * 0.5 * ((0.517 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.517 - Q.lep_z) / 0.179073) / math.sqrt(2))))
    z += -0.0488 * (1.0 * 0.5 * ((3.01 - Q.lepsj_3_n_d3) / 1.0) * (1 + math.erf(((3.01 - Q.lepsj_3_n_d3) / 1.0) / math.sqrt(2))))
    z += 0.0182 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2))))
    z += 0.0223 * (5.110145 * 0.5 * ((131.0 - Q.mass) / 5.110145) * (1 + math.erf(((131.0 - Q.mass) / 5.110145) / math.sqrt(2))))
    z += -0.31 * (0.8395288 * 0.5 * ((1.84 - Q.mass_2charged) / 0.8395288) * (1 + math.erf(((1.84 - Q.mass_2charged) / 0.8395288) / math.sqrt(2))))
    z += 0.0652 * (3.876948 * 0.5 * ((6.77 - Q.mass_2charged) / 3.876948) * (1 + math.erf(((6.77 - Q.mass_2charged) / 3.876948) / math.sqrt(2))))
    z += -0.00941 * (4.06561 * 0.5 * ((24.4 - Q.mass_2charged) / 4.06561) * (1 + math.erf(((24.4 - Q.mass_2charged) / 4.06561) / math.sqrt(2))))
    z += 0.0196 * (1.721646 * 0.5 * ((5.8 - Q.mass_2photon) / 1.721646) * (1 + math.erf(((5.8 - Q.mass_2photon) / 1.721646) / math.sqrt(2))))
    z += -0.0102 * (12.30925 * 0.5 * ((39.0 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.0 - Q.mass_displaced3) / 12.30925) / math.sqrt(2))))
    z += 0.00284 * (14.06404 * 0.5 * ((Q.mass_neutral - 26.3) / 14.06404) * (1 + math.erf(((Q.mass_neutral - 26.3) / 14.06404) / math.sqrt(2))))
    z += 0.0082 * (3.260893 * 0.5 * ((87.5 - Q.mass_top20) / 3.260893) * (1 + math.erf(((87.5 - Q.mass_top20) / 3.260893) / math.sqrt(2))))
    z += -0.0105 * (21.5 * 0.5 * ((77.8 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((77.8 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += 0.0524 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.9) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.9) / 1.0) / math.sqrt(2))))
    z += 4.1 * (0.0164642 * 0.5 * ((0.616 - Q.nca_sj4_pairmax_over_mass) / 0.0164642) * (1 + math.erf(((0.616 - Q.nca_sj4_pairmax_over_mass) / 0.0164642) / math.sqrt(2))))
    z += 0.217 * (0.09724171 * 0.5 * ((3.23 - Q.pair_max_lnkt) / 0.09724171) * (1 + math.erf(((3.23 - Q.pair_max_lnkt) / 0.09724171) / math.sqrt(2))))
    z += 2.1 * (0.008554338 * 0.5 * ((0.0788 - Q.pz_lnd0) / 0.008554338) * (1 + math.erf(((0.0788 - Q.pz_lnd0) / 0.008554338) / math.sqrt(2))))
    z += -0.00518 * (5.468702 * 0.5 * ((Q.sd_mass - 89.7) / 5.468702) * (1 + math.erf(((Q.sd_mass - 89.7) / 5.468702) / math.sqrt(2))))
    z += 14.7 * (0.03292552 * 0.5 * ((0.0563 - Q.sj2_zsoft) / 0.03292552) * (1 + math.erf(((0.0563 - Q.sj2_zsoft) / 0.03292552) / math.sqrt(2))))
    z += -0.0381 * (0.6262913 * 0.5 * ((5.49 - Q.sj3_mass3) / 0.6262913) * (1 + math.erf(((5.49 - Q.sj3_mass3) / 0.6262913) / math.sqrt(2))))
    z += -0.00944 * (5.413174 * 0.5 * ((115.0 - Q.sj3_pair_mass_max) / 5.413174) * (1 + math.erf(((115.0 - Q.sj3_pair_mass_max) / 5.413174) / math.sqrt(2))))
    z += 0.00757 * (2.35787 * 0.5 * ((Q.sj3_pair_mass_min - 26.6) / 2.35787) * (1 + math.erf(((Q.sj3_pair_mass_min - 26.6) / 2.35787) / math.sqrt(2))))
    z += -0.02 * (6.832602 * 0.5 * ((108.0 - Q.sj4_pair_mass_max) / 6.832602) * (1 + math.erf(((108.0 - Q.sj4_pair_mass_max) / 6.832602) / math.sqrt(2))))
    z += 0.524 * (1.0 * 0.5 * ((0.983 - Q.sjq_3_3_nch) / 1.0) * (1 + math.erf(((0.983 - Q.sjq_3_3_nch) / 1.0) / math.sqrt(2))))
    z += 0.00162 * (14.63984 * 0.5 * ((414.0 - Q.sum_pt_top10) / 14.63984) * (1 + math.erf(((414.0 - Q.sum_pt_top10) / 14.63984) / math.sqrt(2))))
    z += -0.00367 * (36.85684 * 0.5 * ((75.9 - Q.sv_1_sd0_sum) / 36.85684) * (1 + math.erf(((75.9 - Q.sv_1_sd0_sum) / 36.85684) / math.sqrt(2))))
    z += 2.63 * (0.009766867 * 0.5 * ((0.097 - Q.tau2) / 0.009766867) * (1 + math.erf(((0.097 - Q.tau2) / 0.009766867) / math.sqrt(2))))
    z += -9.88 * (0.003950899 * 0.5 * ((Q.tau4 - 0.0507) / 0.003950899) * (1 + math.erf(((Q.tau4 - 0.0507) / 0.003950899) / math.sqrt(2))))
    z += -1.09 * (0.01206829 * 0.5 * ((0.895 - Q.tau43) / 0.01206829) * (1 + math.erf(((0.895 - Q.tau43) / 0.01206829) / math.sqrt(2))))
    z += 912.0 * (0.0003783019 * 0.5 * ((0.000779 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.000779 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (0.1133023 * 0.5 * ((1.04 - Q.pair_mean_lnkt) / 0.1133023) * (1 + math.erf(((1.04 - Q.pair_mean_lnkt) / 0.1133023) / math.sqrt(2))))
    z += -3790.0 * (0.0003783019 * 0.5 * ((0.000805 - Q.e3_b2) / 0.0003783019) * (1 + math.erf(((0.000805 - Q.e3_b2) / 0.0003783019) / math.sqrt(2)))) * (0.06955833 * 0.5 * ((0.374 - Q.z_neutral_had) / 0.06955833) * (1 + math.erf(((0.374 - Q.z_neutral_had) / 0.06955833) / math.sqrt(2))))
    z += 0.85 * (0.1723618 * 0.5 * ((Q.ktd_ln_d34 - -9.54) / 0.1723618) * (1 + math.erf(((Q.ktd_ln_d34 - -9.54) / 0.1723618) / math.sqrt(2)))) * (0.01029976 * 0.5 * ((0.103 - Q.pz_lnkt3) / 0.01029976) * (1 + math.erf(((0.103 - Q.pz_lnkt3) / 0.01029976) / math.sqrt(2))))
    z += -0.117 * (0.179073 * 0.5 * ((0.519 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.519 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_n - 0.999) / 1.0) * (1 + math.erf(((Q.ak02_n - 0.999) / 1.0) / math.sqrt(2))))
    z += -0.159 * (0.179073 * 0.5 * ((0.521 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.521 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.dc_n - 1.0) / 1.0) * (1 + math.erf(((Q.dc_n - 1.0) / 1.0) / math.sqrt(2))))
    z += 1.4 * (0.179073 * 0.5 * ((0.517 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.517 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.02439358 * 0.5 * ((0.185 - Q.sj4_dr_min) / 0.02439358) * (1 + math.erf(((0.185 - Q.sj4_dr_min) / 0.02439358) / math.sqrt(2))))
    z += -2.24 * (0.179073 * 0.5 * ((0.519 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.519 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.008350939 * 0.5 * ((0.888 - Q.tau54) / 0.008350939) * (1 + math.erf(((0.888 - Q.tau54) / 0.008350939) / math.sqrt(2))))
    z += 0.0486 * (4.06561 * 0.5 * ((24.9 - Q.mass_2charged) / 4.06561) * (1 + math.erf(((24.9 - Q.mass_2charged) / 4.06561) / math.sqrt(2)))) * (0.01273628 * 0.5 * ((Q.sdb_5_z - 0.0394) / 0.01273628) * (1 + math.erf(((Q.sdb_5_z - 0.0394) / 0.01273628) / math.sqrt(2))))
    z += 1.96 * (0.8395288 * 0.5 * ((1.85 - Q.mass_2charged) / 0.8395288) * (1 + math.erf(((1.85 - Q.mass_2charged) / 0.8395288) / math.sqrt(2)))) * (0.04514027 * 0.5 * ((0.102 - Q.sjf_4_2_z_d3) / 0.04514027) * (1 + math.erf(((0.102 - Q.sjf_4_2_z_d3) / 0.04514027) / math.sqrt(2))))
    z += 0.000163 * (12.30925 * 0.5 * ((39.4 - Q.mass_displaced3) / 12.30925) * (1 + math.erf(((39.4 - Q.mass_displaced3) / 12.30925) / math.sqrt(2)))) * (12.86093 * 0.5 * ((Q.dc_split2_mass - 17.9) / 12.86093) * (1 + math.erf(((Q.dc_split2_mass - 17.9) / 12.86093) / math.sqrt(2))))
    z += 0.000827 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 4.01) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 4.01) / 1.0) / math.sqrt(2)))) * (248.3135 * 0.5 * ((178.0 - Q.sip_3d_3) / 248.3135) * (1 + math.erf(((178.0 - Q.sip_3d_3) / 248.3135) / math.sqrt(2))))
    z += 0.234 * (0.0321369 * 0.5 * ((0.854 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) * (1 + math.erf(((0.854 - Q.nca_sj4_pairmax_over_mass) / 0.0321369) / math.sqrt(2)))) * (1.0 * 0.5 * ((4.01 - Q.n_lund_kt_above_5) / 1.0) * (1 + math.erf(((4.01 - Q.n_lund_kt_above_5) / 1.0) / math.sqrt(2))))
    z += -1.96 * (0.009766867 * 0.5 * ((0.0955 - Q.tau2) / 0.009766867) * (1 + math.erf(((0.0955 - Q.tau2) / 0.009766867) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ak02_1_n_disp3 - 0.903) / 1.0) * (1 + math.erf(((Q.ak02_1_n_disp3 - 0.903) / 1.0) / math.sqrt(2))))
    return z


def neuron_116(Q):
    z = -3.4e-07
    return z


def neuron_117(Q):
    z = 2.86e-06
    return z


def neuron_118(Q):
    z = -8.23e-06
    return z


def neuron_119(Q):
    z = -6.35e-07
    return z


def neuron_120(Q):
    z = 0.367
    z += 1.14 * (0.01151941 * 0.5 * ((Q.N2 - 0.327) / 0.01151941) * (1 + math.erf(((Q.N2 - 0.327) / 0.01151941) / math.sqrt(2))))
    z += -2.68 * (0.03214999 * 0.5 * ((0.339 - Q.N2_b05) / 0.03214999) * (1 + math.erf(((0.339 - Q.N2_b05) / 0.03214999) / math.sqrt(2))))
    z += -8.37 * (0.00189604 * 0.5 * ((0.0142 - Q.dr_min_012) / 0.00189604) * (1 + math.erf(((0.0142 - Q.dr_min_012) / 0.00189604) / math.sqrt(2))))
    z += 131.0 * (0.0005676896 * 0.5 * ((0.00228 - Q.e3) / 0.0005676896) * (1 + math.erf(((0.00228 - Q.e3) / 0.0005676896) / math.sqrt(2))))
    z += 17.9 * (0.002623498 * 0.5 * ((Q.e3_b05 - 0.00575) / 0.002623498) * (1 + math.erf(((Q.e3_b05 - 0.00575) / 0.002623498) / math.sqrt(2))))
    z += -1.39 * (1.0 * 0.5 * ((-0.00345 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((-0.00345 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2))))
    z += 3.95 * (0.03429006 * 0.5 * ((0.0797 - Q.lep_dr) / 0.03429006) * (1 + math.erf(((0.0797 - Q.lep_dr) / 0.03429006) / math.sqrt(2))))
    z += 0.0912 * (2.41177 * 0.5 * ((2.81 - Q.lep_iso) / 2.41177) * (1 + math.erf(((2.81 - Q.lep_iso) / 2.41177) / math.sqrt(2))))
    z += -0.232 * (2.752946 * 0.5 * ((3.47 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.47 - Q.lep_ptrel) / 2.752946) / math.sqrt(2))))
    z += -0.0465 * (5.892378 * 0.5 * ((12.4 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.4 - Q.lep_ptrel) / 5.892378) / math.sqrt(2))))
    z += -6.49 * (0.03060878 * 0.5 * ((0.0646 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0646 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2))))
    z += -0.0892 * (0.3071165 * 0.5 * ((Q.lund3_lndelta - -2.49) / 0.3071165) * (1 + math.erf(((Q.lund3_lndelta - -2.49) / 0.3071165) / math.sqrt(2))))
    z += -0.0125 * (4.146938 * 0.5 * ((110.0 - Q.mass) / 4.146938) * (1 + math.erf(((110.0 - Q.mass) / 4.146938) / math.sqrt(2))))
    z += -0.00715 * (5.110145 * 0.5 * ((131.0 - Q.mass) / 5.110145) * (1 + math.erf(((131.0 - Q.mass) / 5.110145) / math.sqrt(2))))
    z += -0.0123 * (3.277554 * 0.5 * ((68.5 - Q.mass_charged) / 3.277554) * (1 + math.erf(((68.5 - Q.mass_charged) / 3.277554) / math.sqrt(2))))
    z += 0.00642 * (11.4891 * 0.5 * ((32.8 - Q.mass_displaced5) / 11.4891) * (1 + math.erf(((32.8 - Q.mass_displaced5) / 11.4891) / math.sqrt(2))))
    z += 0.0157 * (9.37985 * 0.5 * ((75.7 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((75.7 - Q.mass_neutral) / 9.37985) / math.sqrt(2))))
    z += -0.018 * (3.529599 * 0.5 * ((110.0 - Q.mass_top30) / 3.529599) * (1 + math.erf(((110.0 - Q.mass_top30) / 3.529599) / math.sqrt(2))))
    z += -0.0349 * (3.705078 * 0.5 * ((5.86 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.86 - Q.max_abs_d0) / 3.705078) / math.sqrt(2))))
    z += 0.0562 * (5.26174 * 0.5 * ((Q.mres_sd_mass_b2z01 - 91.4) / 5.26174) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 91.4) / 5.26174) / math.sqrt(2))))
    z += -0.0579 * (5.55459 * 0.5 * ((Q.mres_sd_mass_b2z01 - 96.7) / 5.55459) * (1 + math.erf(((Q.mres_sd_mass_b2z01 - 96.7) / 5.55459) / math.sqrt(2))))
    z += -0.00762 * (21.5 * 0.5 * ((81.9 - Q.n_pairs_kt_above_1) / 21.5) * (1 + math.erf(((81.9 - Q.n_pairs_kt_above_1) / 21.5) / math.sqrt(2))))
    z += -0.00849 * (8.0 * 0.5 * ((62.4 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.4 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2))))
    z += -0.779 * (0.08036477 * 0.5 * ((Q.pair_mean_lndelta - -1.72) / 0.08036477) * (1 + math.erf(((Q.pair_mean_lndelta - -1.72) / 0.08036477) / math.sqrt(2))))
    z += -4.09 * (0.009387389 * 0.5 * ((0.0394 - Q.pz_lnd2) / 0.009387389) * (1 + math.erf(((0.0394 - Q.pz_lnd2) / 0.009387389) / math.sqrt(2))))
    z += 2.88 * (0.008162106 * 0.5 * ((0.158 - Q.pz_lnkt0) / 0.008162106) * (1 + math.erf(((0.158 - Q.pz_lnkt0) / 0.008162106) / math.sqrt(2))))
    z += 0.331 * (0.179816 * 0.5 * ((Q.sjq_3_3_k1 - 0.402) / 0.179816) * (1 + math.erf(((Q.sjq_3_3_k1 - 0.402) / 0.179816) / math.sqrt(2))))
    z += 0.649 * (0.02757255 * 0.5 * ((0.69 - Q.tau32) / 0.02757255) * (1 + math.erf(((0.69 - Q.tau32) / 0.02757255) / math.sqrt(2))))
    z += 1.8 * (1.0 * 0.5 * ((-0.00233 - Q.kt2_1_n_lep) / 1.0) * (1 + math.erf(((-0.00233 - Q.kt2_1_n_lep) / 1.0) / math.sqrt(2)))) * (0.0812128 * 0.5 * ((Q.sjq_2_prod_k05 - -0.239) / 0.0812128) * (1 + math.erf(((Q.sjq_2_prod_k05 - -0.239) / 0.0812128) / math.sqrt(2))))
    z += 6.65 * (0.6671377 * 0.5 * ((0.442 - Q.lep_iso) / 0.6671377) * (1 + math.erf(((0.442 - Q.lep_iso) / 0.6671377) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.000627 - Q.dc_1_n_lep) / 1.0) * (1 + math.erf(((0.000627 - Q.dc_1_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.0277 * (2.41177 * 0.5 * ((2.83 - Q.lep_iso) / 2.41177) * (1 + math.erf(((2.83 - Q.lep_iso) / 2.41177) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.58 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.58 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += -0.0192 * (5.892378 * 0.5 * ((12.5 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.5 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (1.929175 * 0.5 * ((3.85 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((3.85 - Q.jd_3d_5) / 1.929175) / math.sqrt(2))))
    z += 0.00647 * (2.752946 * 0.5 * ((3.68 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.68 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (2.0 * 0.5 * ((Q.n_photon - 5.04) / 2.0) * (1 + math.erf(((Q.n_photon - 5.04) / 2.0) / math.sqrt(2))))
    z += -0.0328 * (2.752946 * 0.5 * ((3.66 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.66 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (1.0 * 0.5 * ((9.97 - Q.n_s3d_above_3) / 1.0) * (1 + math.erf(((9.97 - Q.n_s3d_above_3) / 1.0) / math.sqrt(2))))
    z += 0.00253 * (5.892378 * 0.5 * ((11.9 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((11.9 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (1.5 * 0.5 * ((15.9 - Q.sdb_2_n) / 1.5) * (1 + math.erf(((15.9 - Q.sdb_2_n) / 1.5) / math.sqrt(2))))
    z += 0.000324 * (2.752946 * 0.5 * ((3.55 - Q.lep_ptrel) / 2.752946) * (1 + math.erf(((3.55 - Q.lep_ptrel) / 2.752946) / math.sqrt(2)))) * (401.9953 * 0.5 * ((574.0 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((574.0 - Q.sip_3d_3) / 401.9953) / math.sqrt(2))))
    z += 0.00804 * (5.892378 * 0.5 * ((12.2 - Q.lep_ptrel) / 5.892378) * (1 + math.erf(((12.2 - Q.lep_ptrel) / 5.892378) / math.sqrt(2)))) * (0.09724841 * 0.5 * ((0.432 - Q.sjq_3_1_k05) / 0.09724841) * (1 + math.erf(((0.432 - Q.sjq_3_1_k05) / 0.09724841) / math.sqrt(2))))
    z += 0.0325 * (0.179073 * 0.5 * ((0.513 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.513 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (4.4703e-08 * 0.5 * ((Q.mass_displaced3 - -0.485) / 4.4703e-08) * (1 + math.erf(((Q.mass_displaced3 - -0.485) / 4.4703e-08) / math.sqrt(2))))
    z += -0.0249 * (0.179073 * 0.5 * ((0.518 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.518 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (9.37985 * 0.5 * ((76.1 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((76.1 - Q.mass_neutral) / 9.37985) / math.sqrt(2))))
    z += 0.752 * (0.179073 * 0.5 * ((0.516 - Q.lep_z) / 0.179073) * (1 + math.erf(((0.516 - Q.lep_z) / 0.179073) / math.sqrt(2)))) * (0.09432096 * 0.5 * ((-0.289 - Q.sjq_3_3_k1) / 0.09432096) * (1 + math.erf(((-0.289 - Q.sjq_3_3_k1) / 0.09432096) / math.sqrt(2))))
    z += 3.74 * (0.03060878 * 0.5 * ((0.0623 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0623 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (0.2385164 * 0.5 * ((Q.ak02_dr23 - 0.0104) / 0.2385164) * (1 + math.erf(((Q.ak02_dr23 - 0.0104) / 0.2385164) / math.sqrt(2))))
    z += -1.77 * (0.03060878 * 0.5 * ((0.0687 - Q.lepsj_2_dr) / 0.03060878) * (1 + math.erf(((0.0687 - Q.lepsj_2_dr) / 0.03060878) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.25 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.25 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += 0.00182 * (5.110145 * 0.5 * ((132.0 - Q.mass) / 5.110145) * (1 + math.erf(((132.0 - Q.mass) / 5.110145) / math.sqrt(2)))) * (1.929175 * 0.5 * ((4.2 - Q.jd_3d_5) / 1.929175) * (1 + math.erf(((4.2 - Q.jd_3d_5) / 1.929175) / math.sqrt(2))))
    z += 0.00309 * (3.277554 * 0.5 * ((67.9 - Q.mass_charged) / 3.277554) * (1 + math.erf(((67.9 - Q.mass_charged) / 3.277554) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.6 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.6 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += -0.00889 * (9.37985 * 0.5 * ((77.9 - Q.mass_neutral) / 9.37985) * (1 + math.erf(((77.9 - Q.mass_neutral) / 9.37985) / math.sqrt(2)))) * (0.03369438 * 0.5 * ((Q.sj3_dr_max - 0.542) / 0.03369438) * (1 + math.erf(((Q.sj3_dr_max - 0.542) / 0.03369438) / math.sqrt(2))))
    z += -0.0221 * (10.67504 * 0.5 * ((Q.mass_top10 - 106.0) / 10.67504) * (1 + math.erf(((Q.mass_top10 - 106.0) / 10.67504) / math.sqrt(2)))) * (0.04703017 * 0.5 * ((0.672 - Q.planar_flow) / 0.04703017) * (1 + math.erf(((0.672 - Q.planar_flow) / 0.04703017) / math.sqrt(2))))
    z += 0.0024 * (3.529599 * 0.5 * ((109.0 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.0 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (3.102391 * 0.5 * ((5.33 - Q.jd_3d_6) / 3.102391) * (1 + math.erf(((5.33 - Q.jd_3d_6) / 3.102391) / math.sqrt(2))))
    z += -0.147 * (3.529599 * 0.5 * ((109.0 - Q.mass_top30) / 3.529599) * (1 + math.erf(((109.0 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (0.005973664 * 0.5 * ((Q.sv_2_z - 0.0116) / 0.005973664) * (1 + math.erf(((Q.sv_2_z - 0.0116) / 0.005973664) / math.sqrt(2))))
    z += 0.0883 * (3.529599 * 0.5 * ((108.0 - Q.mass_top30) / 3.529599) * (1 + math.erf(((108.0 - Q.mass_top30) / 3.529599) / math.sqrt(2)))) * (0.0139093 * 0.5 * ((0.128 - Q.z_neutral_had) / 0.0139093) * (1 + math.erf(((0.128 - Q.z_neutral_had) / 0.0139093) / math.sqrt(2))))
    z += -0.242 * (3.705078 * 0.5 * ((5.8 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.8 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.0 * 0.5 * ((Q.ismuon_0 - 1.0) / 1.0) * (1 + math.erf(((Q.ismuon_0 - 1.0) / 1.0) / math.sqrt(2))))
    z += 0.0321 * (3.705078 * 0.5 * ((5.84 - Q.max_abs_d0) / 3.705078) * (1 + math.erf(((5.84 - Q.max_abs_d0) / 3.705078) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.989 - Q.n_electron) / 1.0) * (1 + math.erf(((0.989 - Q.n_electron) / 1.0) / math.sqrt(2))))
    z += 0.0232 * (8.0 * 0.5 * ((63.5 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((63.5 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) * (1.0 * 0.5 * ((0.00191 - Q.dc_2_n_lep) / 1.0) * (1 + math.erf(((0.00191 - Q.dc_2_n_lep) / 1.0) / math.sqrt(2))))
    z += 0.0153 * (8.0 * 0.5 * ((62.9 - Q.n_pairs_kt_above_3) / 8.0) * (1 + math.erf(((62.9 - Q.n_pairs_kt_above_3) / 8.0) / math.sqrt(2)))) * (0.08055962 * 0.5 * ((0.272 - Q.sjf_2_1_z_d3) / 0.08055962) * (1 + math.erf(((0.272 - Q.sjf_2_1_z_d3) / 0.08055962) / math.sqrt(2))))
    return z


def neuron_121(Q):
    z = -5.73e-06
    return z


def neuron_122(Q):
    z = -2.39e-06
    return z


def neuron_123(Q):
    z = 0.0224
    return z


def neuron_124(Q):
    z = -0.0411
    z += -0.556 * (0.2857178 * 0.5 * ((Q.dc_3_charge - 0.642) / 0.2857178) * (1 + math.erf(((Q.dc_3_charge - 0.642) / 0.2857178) / math.sqrt(2))))
    z += 721.0 * (0.0001198329 * 0.5 * ((0.000264 - Q.e3_b2) / 0.0001198329) * (1 + math.erf(((0.000264 - Q.e3_b2) / 0.0001198329) / math.sqrt(2))))
    z += -0.779 * (1.234622 * 0.5 * ((1.52 - Q.lep_iso) / 1.234622) * (1 + math.erf(((1.52 - Q.lep_iso) / 1.234622) / math.sqrt(2))))
    z += 3.81 * (0.1060766 * 0.5 * ((0.219 - Q.lep_z) / 0.1060766) * (1 + math.erf(((0.219 - Q.lep_z) / 0.1060766) / math.sqrt(2))))
    z += 0.0422 * (1.0 * 0.5 * ((Q.lepsj_2_n_d3 - 0.928) / 1.0) * (1 + math.erf(((Q.lepsj_2_n_d3 - 0.928) / 1.0) / math.sqrt(2))))
    z += 22.7 * (0.01610679 * 0.5 * ((0.0268 - Q.lepsj_3_dr) / 0.01610679) * (1 + math.erf(((0.0268 - Q.lepsj_3_dr) / 0.01610679) / math.sqrt(2))))
    z += 0.113 * (5.001896 * 0.5 * ((-8.49 - Q.lepsj_3_maxsd0) / 5.001896) * (1 + math.erf(((-8.49 - Q.lepsj_3_maxsd0) / 5.001896) / math.sqrt(2))))
    z += -0.0159 * (5.197 * 0.5 * ((97.7 - Q.mass) / 5.197) * (1 + math.erf(((97.7 - Q.mass) / 5.197) / math.sqrt(2))))
    z += 0.0144 * (18.42178 * 0.5 * ((183.0 - Q.mass) / 18.42178) * (1 + math.erf(((183.0 - Q.mass) / 18.42178) / math.sqrt(2))))
    z += 0.0223 * (5.898438 * 0.5 * ((8.78 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((8.78 - Q.max_abs_d0) / 5.898438) / math.sqrt(2))))
    z += -0.0323 * (6.235814 * 0.5 * ((75.3 - Q.mres_sd_mass_b0z005) / 6.235814) * (1 + math.erf(((75.3 - Q.mres_sd_mass_b0z005) / 6.235814) / math.sqrt(2))))
    z += 0.046 * (5.31786 * 0.5 * ((91.5 - Q.mres_sd_mass_b0z005) / 5.31786) * (1 + math.erf(((91.5 - Q.mres_sd_mass_b0z005) / 5.31786) / math.sqrt(2))))
    z += -0.0345 * (4.488781 * 0.5 * ((125.0 - Q.mres_sd_mass_b0z005) / 4.488781) * (1 + math.erf(((125.0 - Q.mres_sd_mass_b0z005) / 4.488781) / math.sqrt(2))))
    z += 0.0163 * (18.66842 * 0.5 * ((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) * (1 + math.erf(((160.0 - Q.mres_sd_mass_b0z005) / 18.66842) / math.sqrt(2))))
    z += -0.045 * (1.0 * 0.5 * ((5.99 - Q.n_dr_0p1_0p2) / 1.0) * (1 + math.erf(((5.99 - Q.n_dr_0p1_0p2) / 1.0) / math.sqrt(2))))
    z += -0.0192 * (2.5 * 0.5 * ((30.6 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((30.6 - Q.n_for_90pct) / 2.5) / math.sqrt(2))))
    z += 0.751 * (1.0 * 0.5 * ((0.933 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.933 - Q.n_lepton) / 1.0) / math.sqrt(2))))
    z += -0.00182 * (43.0 * 0.5 * ((362.0 - Q.n_pairs_kt_above_1) / 43.0) * (1 + math.erf(((362.0 - Q.n_pairs_kt_above_1) / 43.0) / math.sqrt(2))))
    z += -0.00119 * (137.5 * 0.5 * ((700.0 - Q.n_pairs_kt_above_1) / 137.5) * (1 + math.erf(((700.0 - Q.n_pairs_kt_above_1) / 137.5) / math.sqrt(2))))
    z += -0.0598 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.33) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.33) / 1.0) / math.sqrt(2))))
    z += -0.1 * (1.5 * 0.5 * ((6.84 - Q.n_s3d_above_3) / 1.5) * (1 + math.erf(((6.84 - Q.n_s3d_above_3) / 1.5) / math.sqrt(2))))
    z += 0.194 * (0.1605002 * 0.5 * ((Q.pair_max_lnm2 - 7.34) / 0.1605002) * (1 + math.erf(((Q.pair_max_lnm2 - 7.34) / 0.1605002) / math.sqrt(2))))
    z += -0.621 * (0.02505229 * 0.5 * ((Q.sdb_2_z - 0.347) / 0.02505229) * (1 + math.erf(((Q.sdb_2_z - 0.347) / 0.02505229) / math.sqrt(2))))
    z += -2.52 * (0.02550179 * 0.5 * ((0.177 - Q.sdb_2_z) / 0.02550179) * (1 + math.erf(((0.177 - Q.sdb_2_z) / 0.02550179) / math.sqrt(2))))
    z += -0.67 * (0.04339825 * 0.5 * ((0.536 - Q.sj2_dr) / 0.04339825) * (1 + math.erf(((0.536 - Q.sj2_dr) / 0.04339825) / math.sqrt(2))))
    z += -1.78 * (0.07655927 * 0.5 * ((Q.sj3_dr12 - 0.615) / 0.07655927) * (1 + math.erf(((Q.sj3_dr12 - 0.615) / 0.07655927) / math.sqrt(2))))
    z += -0.0136 * (3.244489 * 0.5 * ((69.0 - Q.sj4_pair_mass_max) / 3.244489) * (1 + math.erf(((69.0 - Q.sj4_pair_mass_max) / 3.244489) / math.sqrt(2))))
    z += 0.17 * (2.0 * 0.5 * ((Q.sjf_2_1_n_d3 - 7.22) / 2.0) * (1 + math.erf(((Q.sjf_2_1_n_d3 - 7.22) / 2.0) / math.sqrt(2))))
    z += -1.07 * (0.0439522 * 0.5 * ((0.0969 - Q.sjq_2_prod_k1) / 0.0439522) * (1 + math.erf(((0.0969 - Q.sjq_2_prod_k1) / 0.0439522) / math.sqrt(2))))
    z += -0.379 * (0.03580654 * 0.5 * ((Q.sjq_2_sumabs_k1 - 0.281) / 0.03580654) * (1 + math.erf(((Q.sjq_2_sumabs_k1 - 0.281) / 0.03580654) / math.sqrt(2))))
    z += -5.75 * (0.01710086 * 0.5 * ((0.0594 - Q.z_displaced3) / 0.01710086) * (1 + math.erf(((0.0594 - Q.z_displaced3) / 0.01710086) / math.sqrt(2))))
    z += 1.24 * (0.01670814 * 0.5 * ((Q.z_displaced5 - 0.0565) / 0.01670814) * (1 + math.erf(((Q.z_displaced5 - 0.0565) / 0.01670814) / math.sqrt(2))))
    z += -0.765 * (0.02246636 * 0.5 * ((Q.z_photon - 0.12) / 0.02246636) * (1 + math.erf(((Q.z_photon - 0.12) / 0.02246636) / math.sqrt(2))))
    z += 0.00428 * (15.88428 * 0.5 * ((44.7 - Q.lep_ptrel) / 15.88428) * (1 + math.erf(((44.7 - Q.lep_ptrel) / 15.88428) / math.sqrt(2)))) * (0.05556614 * 0.5 * ((Q.jet_charge_k05 - 0.0289) / 0.05556614) * (1 + math.erf(((Q.jet_charge_k05 - 0.0289) / 0.05556614) / math.sqrt(2))))
    z += -5.56 * (0.1486471 * 0.5 * ((0.338 - Q.lep_z) / 0.1486471) * (1 + math.erf(((0.338 - Q.lep_z) / 0.1486471) / math.sqrt(2)))) * (0.02653106 * 0.5 * ((Q.z_displaced5 - 0.093) / 0.02653106) * (1 + math.erf(((Q.z_displaced5 - 0.093) / 0.02653106) / math.sqrt(2))))
    z += 0.000355 * (3.317894 * 0.5 * ((118.0 - Q.mass) / 3.317894) * (1 + math.erf(((118.0 - Q.mass) / 3.317894) / math.sqrt(2)))) * (6.000774 * 0.5 * ((16.5 - Q.jd_3d_4) / 6.000774) * (1 + math.erf(((16.5 - Q.jd_3d_4) / 6.000774) / math.sqrt(2))))
    z += 0.0259 * (2.5 * 0.5 * ((31.3 - Q.n_for_90pct) / 2.5) * (1 + math.erf(((31.3 - Q.n_for_90pct) / 2.5) / math.sqrt(2)))) * (0.1673111 * 0.5 * ((Q.sjq_2_prod_k05 - -0.501) / 0.1673111) * (1 + math.erf(((Q.sjq_2_prod_k05 - -0.501) / 0.1673111) / math.sqrt(2))))
    z += -24.4 * (1.0 * 0.5 * ((0.969 - Q.n_lepton) / 1.0) * (1 + math.erf(((0.969 - Q.n_lepton) / 1.0) / math.sqrt(2)))) * (0.01040872 * 0.5 * ((0.0177 - Q.sdb_4_z) / 0.01040872) * (1 + math.erf(((0.0177 - Q.sdb_4_z) / 0.01040872) / math.sqrt(2))))
    z += -0.00649 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.09) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.09) / 1.0) / math.sqrt(2)))) * (5.898438 * 0.5 * ((9.91 - Q.max_abs_d0) / 5.898438) * (1 + math.erf(((9.91 - Q.max_abs_d0) / 5.898438) / math.sqrt(2))))
    z += -9.17e-05 * (1.0 * 0.5 * ((Q.n_s3d_above_3 - 3.21) / 1.0) * (1 + math.erf(((Q.n_s3d_above_3 - 3.21) / 1.0) / math.sqrt(2)))) * (401.9953 * 0.5 * ((580.0 - Q.sip_3d_3) / 401.9953) * (1 + math.erf(((580.0 - Q.sip_3d_3) / 401.9953) / math.sqrt(2))))
    return z


def neuron_125(Q):
    z = -1.19e-06
    return z


def neuron_126(Q):
    z = 7.63e-06
    return z


def neuron_127(Q):
    z = -2.01e-07
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
