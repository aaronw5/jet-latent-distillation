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

Whole test file (2,000,000 jets): accuracy 75.16% (the network: 86.03%); same class as the network for 79.96% of jets.

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
  Q.ak02_2_n_lep           2nd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_2_z               pT share of the 2nd anti-kT 0.2 subjet (0 if none)
  Q.ak02_3_n_lep           3rd anti-kT 0.2 subjet: number of its electrons and muons
  Q.ak02_3_z               pT share of the 3rd anti-kT 0.2 subjet (0 if none)
  Q.ak02_dr12              distance between the pT-weighted centres of the hardest and 2nd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr13              distance between the pT-weighted centres of the hardest and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr23              distance between the pT-weighted centres of the 2nd and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_min12_jp          the smaller of the two hardest anti-kT 0.2 subjets: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.ak02_min12_n_disp3     the smaller of the two hardest anti-kT 0.2 subjets: number of its tracks with d0/σ > 3
  Q.ak02_n                 number of anti-kT R = 0.2 subjets with pT > 10 GeV
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
  Q.dr_19                  ΔR from the jet axis of particle 19 (ParT input; empty slot: 0)
  Q.dr_29                  ΔR from the jet axis of particle 29 (ParT input; empty slot: 0)
  Q.dr_30                  ΔR from the jet axis of particle 30 (ParT input; empty slot: 0)
  Q.dr_8                   ΔR from the jet axis of particle 8 (ParT input; empty slot: 0)
  Q.dr_max_012             largest distance among the 3 hardest
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.e3                     energy correlation e3 (β=1, 32 hardest)
  Q.e3_b2                  energy correlation e3 with β = 2
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 16 hardest)
  Q.e4_b05                 energy correlation e4 with β = 0.5
  Q.ecf_g41                generalized energy correlation ₁e₄ (β=1; products of the smallest angles)
  Q.ecf_g42                generalized energy correlation ₂e₄ (β=1; products of the smallest angles)
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.ischhad_0              1 if a charged hadron of particle 0 (ParT input; empty slot: 0)
  Q.iselectron_1           1 if an electron of particle 1 (ParT input; empty slot: 0)
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
  Q.sjf_2_2_z_d3           subjet 2 of 2 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_3_1_max3d          subjet 1 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_1_n_d3           subjet 1 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_3_1_n_d5           subjet 1 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 5
  Q.sjf_3_2_max3d          subjet 2 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_2_maxsd0         subjet 2 of 3 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_3_3_max3d          subjet 3 of 3 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_3_3_maxsd0         subjet 3 of 3 (k-means axes, by pT): largest |d0/σ| (0 if no track)
  Q.sjf_3_3_n_d3           subjet 3 of 3 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_4_1_max3d          subjet 1 of 4 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_4_1_n_d3           subjet 1 of 4 (k-means axes, by pT): number of tracks with |d0/σ| > 3
  Q.sjf_4_1_z_d3           subjet 1 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_2_z_d3           subjet 2 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_3_z_d3           subjet 3 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
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
  Q.sv_2_z                 2nd displaced-track cluster: pT share of the jet (0 if none)
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau32_b2               N-subjettiness τ3/τ2 with β = 2
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.tau43                  N-subjettiness τ4/τ3
  Q.tau5                   N-subjettiness τ5 (β=1)
  Q.tau54                  N-subjettiness τ5/τ4
  Q.tdz_1                  tanh(dz [mm]) of particle 1 (ParT input; empty slot: 0)
  Q.z_charged              pT share of charged particles
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
        ak02_2_n_lep=pq('ak02_2_n_lep'),
        ak02_2_z=pq('ak02_2_z'),
        ak02_3_n_lep=pq('ak02_3_n_lep'),
        ak02_3_z=pq('ak02_3_z'),
        ak02_dr12=pq('ak02_dr12'),
        ak02_dr13=pq('ak02_dr13'),
        ak02_dr23=pq('ak02_dr23'),
        ak02_min12_jp=pq('ak02_min12_jp'),
        ak02_min12_n_disp3=pq('ak02_min12_n_disp3'),
        ak02_n=pq('ak02_n'),
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
        dr_19=pfeat(19, 'dr'),
        dr_29=pfeat(29, 'dr'),
        dr_30=pfeat(30, 'dr'),
        dr_8=pfeat(8, 'dr'),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        e3=ecf('e3'),
        e3_b2=ecfb('e3', 2),
        e4=ecf('e4'),
        e4_b05=ecfb('e4', 0.5),
        ecf_g41=ecfb('g41', 1),
        ecf_g42=ecfb('g42', 1),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        ischhad_0=pfeat(0, 'ischhad'),
        iselectron_1=pfeat(1, 'iselectron'),
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
        sjf_2_2_z_d3=pq('sjf_2_2_z_d3'),
        sjf_3_1_max3d=pq('sjf_3_1_max3d'),
        sjf_3_1_n_d3=pq('sjf_3_1_n_d3'),
        sjf_3_1_n_d5=pq('sjf_3_1_n_d5'),
        sjf_3_2_max3d=pq('sjf_3_2_max3d'),
        sjf_3_2_maxsd0=pq('sjf_3_2_maxsd0'),
        sjf_3_3_max3d=pq('sjf_3_3_max3d'),
        sjf_3_3_maxsd0=pq('sjf_3_3_maxsd0'),
        sjf_3_3_n_d3=pq('sjf_3_3_n_d3'),
        sjf_4_1_max3d=pq('sjf_4_1_max3d'),
        sjf_4_1_n_d3=pq('sjf_4_1_n_d3'),
        sjf_4_1_z_d3=pq('sjf_4_1_z_d3'),
        sjf_4_2_z_d3=pq('sjf_4_2_z_d3'),
        sjf_4_3_z_d3=pq('sjf_4_3_z_d3'),
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
        sv_2_z=pq('sv_2_z'),
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau32_b2=tau_n(3, 2) / max(tau_n(2, 2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
        tau5=tau_n(5),
        tau54=tau_n(5) / max(tau_n(4), 1e-12),
        tdz_1=pfeat(1, 'tdz'),
        z_charged=sum(z[i] for i in real if charge[i] != 0),
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
        z_top15_slots=sum(pt[:15]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top3_slots=sum(pt[:3]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
    )


def neuron_0(Q):
    z = 0.005191629
    return z


def neuron_1(Q):
    z = -5.325267e-06
    return z


def neuron_2(Q):
    z = 5.4473e-06
    return z


def neuron_3(Q):
    z = 8.39984e-06
    return z


def neuron_4(Q):
    z = 2.79214e-06
    return z


def neuron_5(Q):
    z = -2.156947e-06
    return z


def neuron_6(Q):
    z = 1.770106e-05
    return z


def neuron_7(Q):
    z = 6.430826e-06
    return z


def neuron_8(Q):
    z = 2.926472e-06
    return z


def neuron_9(Q):
    z = -2.443663e-06
    return z


def neuron_10(Q):
    z = -1.570106e-05
    return z


def neuron_11(Q):
    z = -1.887713e-05
    return z


def neuron_12(Q):
    z = 5.099299e-06
    return z


def neuron_13(Q):
    z = 4.422914e-06
    return z


def neuron_14(Q):
    z = 2.845305e-06
    return z


def neuron_15(Q):
    z = -6.459527e-06
    return z


def neuron_16(Q):
    z = -0.2287291
    if Q.lep_ptrel < 27.3236:
        z += 0.02666054 * Q.lep_ptrel + 1.574576
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.08428749 * Q.lep_ptrel
    if Q.lep_ptrel >= 43.20788:
        z += -0.01772597 * Q.lep_ptrel + 4.407786
    if Q.n_s3d_above_3 < 10.0:
        z += 0.245625 * Q.n_s3d_above_3 - 2.45625
    if Q.pz_lnd2 < 0.08321691:
        z += -3.012479 * Q.pz_lnd2 + 0.2506892
    if Q.sip_3d_2 < 226.3008:
        z += -0.002039652 * Q.sip_3d_2 + 0.5139325
    if 226.3008 <= Q.sip_3d_2 < 447.0873:
        z += -0.0002371416 * Q.sip_3d_2 + 0.106023
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.003606103 * Q.n_pairs_kt_above_3 + 0.4230039
    if 28.0 <= Q.n_pairs_kt_above_3 < 34.0:
        z += -0.05367216 * Q.n_pairs_kt_above_3 + 1.824854
    if Q.mass_neutral >= 34.02385:
        z += 0.005089335 * Q.mass_neutral - 0.1731588
    if Q.z_displaced5 < 0.2801368:
        z += 4.355842 * Q.z_displaced5 - 1.220232
    if Q.e3_b2 < 0.0004126585:
        z += 191.0231 * Q.e3_b2 + 0.1376811
    if 0.0004126585 <= Q.e3_b2 < 0.0007909605:
        z += -572.3163 * Q.e3_b2 + 0.4526796
    if Q.pz_lnd0 < 0.156893:
        z += 1.487204 * Q.pz_lnd0 - 0.2333319
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.01495361 * Q.mres_sd_mass_b0z005 - 1.825835
    if 159.9242 <= Q.mres_sd_mass_b0z005 < 178.8957:
        z += -0.03536111 * Q.mres_sd_mass_b0z005 + 6.220708
    if Q.mres_sd_mass_b0z005 >= 178.8957:
        z += -0.01668251 * Q.mres_sd_mass_b0z005 + 2.879186
    z += -4.303124 * Q.z_charged_had
    if Q.mres_sd_mass_b2z01 < 91.15481:
        z += -0.006429994 * Q.mres_sd_mass_b2z01 + 0.5861248
    if 20.76537 <= Q.mass_2charged < 45.73288:
        z += 0.007274136 * Q.mass_2charged - 0.1510501
    if Q.mass_2charged >= 45.73288:
        z += -0.006545953 * Q.mass_2charged + 0.4809823
    if Q.max_abs_d0 < 5.8125:
        z += -0.05075599 * Q.max_abs_d0 + 0.2950192
    if Q.sdb_2_n < 16.0:
        z += -0.02336806 * Q.sdb_2_n + 0.3738889
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.008189518 * Q.n_pairs_kt_above_1 + 0.6551614
    if Q.z_dr_0p4_up < 0.01553665:
        z += 9.115496 * Q.z_dr_0p4_up - 0.1416243
    if Q.mass_top5 >= 62.84493:
        z += 0.003080412 * Q.mass_top5 - 0.1935883
    if Q.mass_charged < 62.00562:
        z += -0.01485765 * Q.mass_charged + 0.9212581
    if Q.sjq_2_sumabs_k1 >= 0.2432922:
        z += 0.9178805 * Q.sjq_2_sumabs_k1 - 0.2233131
    if Q.ak02_min12_n_disp3 >= 1.0:
        z += 0.1534852 * Q.ak02_min12_n_disp3 - 0.1534852
    if Q.sjq_2_prod_k1 < 0.09802954:
        z += -2.103827 * Q.sjq_2_prod_k1 + 0.2062372
    if Q.mres_sd_prong_mass1 < 46.61784:
        z += -0.003616882 * Q.mres_sd_prong_mass1 + 0.1686112
    if Q.dr_min_012 < 0.01841101:
        z += 11.12838 * Q.dr_min_012 - 0.2048846
    if Q.jd_3d_6 >= 30.57088:
        z += -0.0008190181 * Q.jd_3d_6 + 0.02503811
    if Q.mass_top40 >= 112.406:
        z += 0.01616797 * Q.mass_top40 - 1.817377
    if Q.sjf_3_2_max3d < 3.101398:
        z += -0.05297454 * Q.sjf_3_2_max3d + 0.1642951
    if Q.n_photon < 18.0:
        z += 0.02791343 * Q.n_photon - 0.5024418
    if Q.n_dr_0p1_0p2 < 6.0:
        z += -0.04280733 * Q.n_dr_0p1_0p2 + 0.256844
    z += 0.05939543 * Q.n_sdz_above_2
    if Q.sip_3d_3 < 175.9957:
        z += -0.002042177 * Q.sip_3d_3 + 0.3594144
    if Q.jd_3d_4 < 14.85973:
        z += 0.02786235 * Q.jd_3d_4 - 0.4140269
    if Q.z_dr_0p05_0p1 < 0.01556887:
        z += -10.68403 * Q.z_dr_0p05_0p1 + 0.1663383
    if Q.M3 >= 0.02716047:
        z += 10.50276 * Q.M3 - 0.2852598
    if Q.lne_4 < 3.577424:
        z += -0.1105691 * Q.lne_4 + 0.3955526
    if Q.sj2_mass2 < 7.54918:
        z += 0.0354531 * Q.sj2_mass2 - 0.2676418
    if Q.sjq_2_2_nch < 3.0:
        z += -0.07877822 * Q.sjq_2_2_nch + 0.2363347
    if Q.z_electron < 0.1373477:
        z += 1.89364 * Q.z_electron - 0.2600871
    if Q.max_abs_dz < 0.4436035:
        z += 0.4881927 * Q.max_abs_dz + 0.009351903
    if 0.4436035 <= Q.max_abs_dz < 2.957031:
        z += -0.08988358 * Q.max_abs_dz + 0.2657886
    if Q.sdb_0_n >= 3.0:
        z += 0.07043633 * Q.sdb_0_n - 0.211309
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += -0.02664284 * Q.sjf_2_1_n_d3 + 0.1332142
    if Q.sjf_4_1_n_d3 >= 2.0:
        z += 0.02994228 * Q.sjf_4_1_n_d3 - 0.05988455
    if Q.mass_displaced3 < 1.777286:
        z += 0.2695281 * Q.mass_displaced3 - 0.4790284
    if Q.mres_sd_rg_b2z01 >= 0.457968:
        z += 0.9872803 * Q.mres_sd_rg_b2z01 - 0.4521428
    if Q.tdz_1 < -0.1170754:
        z += -0.2853842 * Q.tdz_1 - 0.03341147
    if Q.sdb_2_z < 0.1530389:
        z += -0.8051314 * Q.sdb_2_z + 0.1232164
    if Q.jd_sum_abs_sd0_top5 < 9.408315:
        z += 0.1328621 * Q.jd_sum_abs_sd0_top5 - 1.250008
    if Q.mres_sd_mass_b0z02 >= 93.75785:
        z += 0.001341937 * Q.mres_sd_mass_b0z02 - 0.1258172
    if Q.ak02_dr23 >= 0.3194837:
        z += 0.1945101 * Q.ak02_dr23 - 0.0621428
    if Q.sjf_2_1_maxsd0 < 1.113424:
        z += -0.2132594 * Q.sjf_2_1_maxsd0 + 0.2374481
    if Q.n_pairs_kt_above_10 < 7.0:
        z += -0.03751326 * Q.n_pairs_kt_above_10 + 0.2625928
    if Q.N2_b05 < 0.302405:
        z += 1.448434 * Q.N2_b05 - 0.4380137
    if Q.N2 < 0.1506626:
        z += 2.081478 * Q.N2 - 0.3136008
    if Q.sjf_4_n2disp < 2.0:
        z += -0.09512839 * Q.sjf_4_n2disp + 0.1902568
    if Q.z_muon < 0.03898651:
        z += 4.990124 * Q.z_muon - 0.1945475
    if Q.lep_z < 0.221436:
        z += -2.763402 * Q.lep_z + 1.43346
    if 0.221436 <= Q.lep_z < 0.5187302:
        z += -4.913926 * Q.lep_z + 1.909664
    if Q.lep_z >= 0.5187302:
        z += -2.150524 * Q.lep_z + 0.4762034
    if Q.z_charged < 0.8615361:
        z += 3.55077 * Q.z_charged - 3.059117
    if Q.z_displaced5 < 0.2801368 and Q.jd_3d_4 < 172.888:
        z += 0.02246691 * (0.2801368 - Q.z_displaced5) * (172.888 - Q.jd_3d_4)
    if Q.n_s3d_above_3 < 10.0 and Q.n_neutral > 10.0:
        z += 0.001123086 * (10.0 - Q.n_s3d_above_3) * (Q.n_neutral - 10.0)
    if Q.n_s3d_above_3 < 10.0 and Q.sv_1_n > 1.0:
        z += 0.05841611 * (10.0 - Q.n_s3d_above_3) * (Q.sv_1_n - 1.0)
    if Q.e3_b2 < 0.0004126585 and Q.n_lund > 5.0:
        z += 31.38159 * (0.0004126585 - Q.e3_b2) * (Q.n_lund - 5.0)
    if Q.n_s3d_above_3 < 10.0 and Q.sdb_4_z < 0.1270192:
        z += 0.245572 * (10.0 - Q.n_s3d_above_3) * (0.1270192 - Q.sdb_4_z)
    if Q.n_pairs_kt_above_1 < 80.0 and Q.tau32 < 0.6513932:
        z += -0.01475825 * (80.0 - Q.n_pairs_kt_above_1) * (0.6513932 - Q.tau32)
    if Q.mass_neutral > 34.02385 and Q.jd_3d_5 < 2.912651:
        z += -0.005214938 * (Q.mass_neutral - 34.02385) * (2.912651 - Q.jd_3d_5)
    if Q.sjq_2_prod_k1 < 0.09802954 and Q.ak02_2_n_lep < 1.0:
        z += -0.5421745 * (0.09802954 - Q.sjq_2_prod_k1) * (1.0 - Q.ak02_2_n_lep)
    if Q.sdb_2_n < 16.0 and Q.tau43 < 0.8479222:
        z += -0.04025153 * (16.0 - Q.sdb_2_n) * (0.8479222 - Q.tau43)
    if Q.sjq_2_sumabs_k1 > 0.2432922 and Q.n_lepton < 2.0:
        z += -0.1690966 * (Q.sjq_2_sumabs_k1 - 0.2432922) * (2.0 - Q.n_lepton)
    if Q.sdb_2_n < 16.0 and Q.dc_pair_mass_min < 105.4151:
        z += -8.083399e-05 * (16.0 - Q.sdb_2_n) * (105.4151 - Q.dc_pair_mass_min)
    if Q.sip_3d_2 < 447.0873 and Q.lepsj_2_n_d3 > 2.0:
        z += -0.0002790866 * (447.0873 - Q.sip_3d_2) * (Q.lepsj_2_n_d3 - 2.0)
    if Q.sip_3d_3 < 175.9957 and Q.mres_sd_prong_mass2 > 1.685874e-06:
        z += -4.116638e-05 * (175.9957 - Q.sip_3d_3) * (Q.mres_sd_prong_mass2 - 1.685874e-06)
    if Q.sjq_2_prod_k1 < 0.09802954 and Q.dc_1_z_disp3 < 0.200234:
        z += -2.417205 * (0.09802954 - Q.sjq_2_prod_k1) * (0.200234 - Q.dc_1_z_disp3)
    if Q.z_displaced5 < 0.2801368 and Q.n_lund_kt_above_5 > 1.0:
        z += -0.309916 * (0.2801368 - Q.z_displaced5) * (Q.n_lund_kt_above_5 - 1.0)
    if Q.e3_b2 < 0.0004126585 and Q.mass_2photon > 22.18431:
        z += 41.98616 * (0.0004126585 - Q.e3_b2) * (Q.mass_2photon - 22.18431)
    if Q.sdb_2_n < 16.0 and Q.sjq_3_sumabs_k03 > 0.5318777:
        z += 0.006933694 * (16.0 - Q.sdb_2_n) * (Q.sjq_3_sumabs_k03 - 0.5318777)
    if Q.dr_min_012 < 0.01841101 and Q.sjf_4_1_max3d < 3.652671:
        z += 4.961947 * (0.01841101 - Q.dr_min_012) * (3.652671 - Q.sjf_4_1_max3d)
    return z


def neuron_17(Q):
    z = -1.035347e-05
    return z


def neuron_18(Q):
    z = 2.319113
    if Q.lep_z < 0.004135872:
        z += 160.7518 * Q.lep_z - 1.579747
    if 0.004135872 <= Q.lep_z < 0.221436:
        z += 4.210296 * Q.lep_z - 0.9323112
    if 100.4835 <= Q.mass < 110.2019:
        z += -0.009246718 * Q.mass + 0.9291422
    if 110.2019 <= Q.mass < 164.4374:
        z += -0.02580126 * Q.mass + 2.753484
    if 164.4374 <= Q.mass < 182.8592:
        z += 0.03077813 * Q.mass - 6.550286
    if Q.mass >= 182.8592:
        z += -0.002859797 * Q.mass - 0.3992812
    if Q.mass_top20 < 114.4658:
        z += 0.005422106 * Q.mass_top20 - 0.6206458
    if Q.lepsj_3_n_d3 < 2.0:
        z += -0.436255 * Q.lepsj_3_n_d3 + 1.179056
    if 2.0 <= Q.lepsj_3_n_d3 < 3.0:
        z += -0.3065457 * Q.lepsj_3_n_d3 + 0.919637
    if Q.z_displaced3 < 0.03624058:
        z += 12.10414 * Q.z_displaced3 - 0.438661
    if Q.z_displaced3 >= 0.2092108:
        z += -1.882253 * Q.z_displaced3 + 0.3937877
    if Q.n_s3d_above_10 < 6.0:
        z += -0.04797893 * Q.n_s3d_above_10 + 0.2878736
    if Q.lep_iso < 1.362094:
        z += -0.1406913 * Q.lep_iso - 0.01664769
    if 1.362094 <= Q.lep_iso < 6.185635:
        z += 0.04318042 * Q.lep_iso - 0.2670983
    if Q.z_displaced5 < 0.08090366:
        z += -10.47154 * Q.z_displaced5 + 0.8471858
    if Q.lund3_lndelta >= -2.4279:
        z += -0.09448562 * Q.lund3_lndelta - 0.2294016
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.1809283 * Q.lepsj_3_maxsd0 - 0.3577354
    if 2.413264 <= Q.lepsj_3_maxsd0 < 46.74905:
        z += 0.009923575 * Q.lepsj_3_maxsd0 - 0.8183114
    if 46.74905 <= Q.lepsj_3_maxsd0 < 586.9572:
        z += 0.0006560318 * Q.lepsj_3_maxsd0 - 0.3850626
    if 111.4443 <= Q.mres_sd_mass_b2z01 < 121.6067:
        z += 0.008873686 * Q.mres_sd_mass_b2z01 - 0.9889213
    if 121.6067 <= Q.mres_sd_mass_b2z01 < 177.5398:
        z += -0.01364716 * Q.mres_sd_mass_b2z01 + 1.749764
    if Q.mres_sd_mass_b2z01 >= 177.5398:
        z += 0.0173065 * Q.mres_sd_mass_b2z01 - 3.745742
    if Q.C2_b05 >= 0.3223395:
        z += -1.338127 * Q.C2_b05 + 0.4313313
    if Q.n_s3d_above_3 < 3.0:
        z += 0.07266568 * Q.n_s3d_above_3 - 0.2179971
    if Q.ak02_1_z < 0.849996:
        z += 0.3901002 * Q.ak02_1_z - 0.3315836
    if Q.sj3_pair_mass_max >= 128.6605:
        z += -0.01603337 * Q.sj3_pair_mass_max + 2.062862
    if Q.n_dr_0p4_up < 4.0:
        z += -0.0485347 * Q.n_dr_0p4_up + 0.1941388
    if Q.lam1 < 0.01299032:
        z += 78.48108 * Q.lam1 - 1.019494
    if Q.n_real_top50 >= 26.0:
        z += -0.02013785 * Q.n_real_top50 + 0.5235841
    if Q.M2 >= 0.07013948:
        z += 5.011652 * Q.M2 - 0.3515147
    if Q.sv_1_sd0_sum < 51.56287:
        z += -0.005137784 * Q.sv_1_sd0_sum + 0.2649189
    if Q.sjf_3_1_max3d < 2.078395:
        z += -0.06739642 * Q.sjf_3_1_max3d - 0.2436184
    if 2.078395 <= Q.sjf_3_1_max3d < 2.924357:
        z += 0.0767418 * Q.sjf_3_1_max3d - 0.5431946
    if 2.924357 <= Q.sjf_3_1_max3d < 3.710567:
        z += 0.7311534 * Q.sjf_3_1_max3d - 2.456928
    if 3.710567 <= Q.sjf_3_1_max3d < 71.08287:
        z += -0.003800765 * Q.sjf_3_1_max3d + 0.2701693
    if Q.sjf_2_2_max3d < 4.516968:
        z += 0.1563694 * Q.sjf_2_2_max3d - 0.3925672
    if 4.516968 <= Q.sjf_2_2_max3d < 55.92931:
        z += -0.006102587 * Q.sjf_2_2_max3d + 0.3413134
    if Q.sjf_2_1_z_d3 < 0.0828821:
        z += 3.163692 * Q.sjf_2_1_z_d3 - 0.2622134
    if Q.sjq_2_1_nch < 16.0:
        z += -0.0286676 * Q.sjq_2_1_nch + 0.4586816
    if Q.sj2_mass2 < 6.465318:
        z += -0.05664145 * Q.sj2_mass2 + 0.366205
    if Q.n_sdz_above_5 < 3.0:
        z += -0.09050633 * Q.n_sdz_above_5 + 0.271519
    if Q.z_charged_had < 0.3603262:
        z += -1.718468 * Q.z_charged_had + 0.619209
    if Q.n_for_90pct < 29.0:
        z += 0.03669081 * Q.n_for_90pct - 1.064033
    if Q.tau5 < 0.03663959:
        z += -20.04651 * Q.tau5 + 0.734496
    if Q.sdb_jp_top3 < 813.043:
        z += 0.0002592786 * Q.sdb_jp_top3 - 0.2108047
    if Q.n_pairs_kt_above_1 < 101.0:
        z += 0.001333115 * Q.n_pairs_kt_above_1 - 0.1346446
    if Q.psi_0p1 < 0.624781:
        z += -0.325357 * Q.psi_0p1 + 0.2032769
    if Q.N2 >= 0.4137858:
        z += -4.174646 * Q.N2 + 1.727409
    if Q.ak02_dr23 >= 0.3194837:
        z += -0.3160128 * Q.ak02_dr23 + 0.1009609
    if Q.sjq_2_2_nch < 2.0:
        z += -0.1108585 * Q.sjq_2_2_nch + 0.2217171
    z += 0.04657787 * Q.sjf_4_n2disp
    if Q.sjf_4_1_z_d3 >= 0.181327:
        z += 1.509421 * Q.sjf_4_1_z_d3 - 0.2736989
    if Q.n_dr_0p1_0p2 < 17.0:
        z += -0.004970596 * Q.n_dr_0p1_0p2 + 0.08450013
    if Q.mass_charged >= 113.5019:
        z += 0.004002567 * Q.mass_charged - 0.4542991
    if Q.sjf_2_1_max3d < 3.32421:
        z += 0.06248488 * Q.sjf_2_1_max3d - 0.07500077
    if 3.32421 <= Q.sjf_2_1_max3d < 55.67805:
        z += -0.002534906 * Q.sjf_2_1_max3d + 0.1411386
    if Q.mass_neutral >= 59.66563:
        z += 0.001995829 * Q.mass_neutral - 0.1190824
    if Q.lepsj_3_dr >= 0.09790963:
        z += 0.4499401 * Q.lepsj_3_dr - 0.04405347
    if Q.jd_n_d3_pt1 < 4.0:
        z += -0.01721607 * Q.jd_n_d3_pt1 + 0.06886427
    if Q.mass_top40 < 126.8853:
        z += 0.009986507 * Q.mass_top40 - 1.26714
    if Q.mass_top30 < 146.3096:
        z += -0.002119747 * Q.mass_top30 + 0.3101393
    if Q.sip_3d_3 < 4.606241:
        z += 0.134839 * Q.sip_3d_3 - 0.6211009
    if Q.e3_b2 < 0.0002536827:
        z += -492.8892 * Q.e3_b2 - 0.1652698
    if 0.0002536827 <= Q.e3_b2 < 0.0007909605:
        z += 540.3298 * Q.e3_b2 - 0.4273795
    if Q.max_abs_dz < 17.21875:
        z += 0.01059728 * Q.max_abs_dz - 0.1824718
    if Q.sdb_0_z < 0.05934469:
        z += 2.076773 * Q.sdb_0_z - 0.1232455
    if Q.lnptrel_2 >= -2.51737:
        z += 0.1236019 * Q.lnptrel_2 + 0.3111516
    if Q.sjf_3_3_max3d < 1.348343:
        z += -0.1083858 * Q.sjf_3_3_max3d + 0.1461411
    if Q.sjq_3_2_nch < 3.0:
        z += -0.06425688 * Q.sjq_3_2_nch + 0.1927707
    if Q.sj3_pairmin_over_m < 0.2072106:
        z += 1.496272 * Q.sj3_pairmin_over_m - 0.3100435
    if Q.pz_lnd0 < 0.1032185:
        z += -1.470524 * Q.pz_lnd0 + 0.1517852
    if Q.mass_displaced3 < 2.409613:
        z += -0.1529669 * Q.mass_displaced3 + 0.368591
    if Q.sjf_3_1_n_d3 < 1.0:
        z += -0.2856171 * Q.sjf_3_1_n_d3 + 0.2856171
    if Q.lep_z < 0.221436 and Q.N2 < 0.3829918:
        z += 22.43078 * (0.221436 - Q.lep_z) * (0.3829918 - Q.N2)
    if Q.lep_z < 0.221436 and Q.sdb_2_z > 0.2509165:
        z += -6.469633 * (0.221436 - Q.lep_z) * (Q.sdb_2_z - 0.2509165)
    if Q.lepsj_3_n_d3 < 3.0 and Q.lep_dr < 0.08178299:
        z += -0.8210179 * (3.0 - Q.lepsj_3_n_d3) * (0.08178299 - Q.lep_dr)
    if Q.lep_z < 0.221436 and Q.z_displaced5 < 0.08090366:
        z += -55.93586 * (0.221436 - Q.lep_z) * (0.08090366 - Q.z_displaced5)
    if Q.lep_z < 0.221436 and Q.lne_4 > 3.751708:
        z += 0.9731357 * (0.221436 - Q.lep_z) * (Q.lne_4 - 3.751708)
    if Q.mass_displaced3 < 9.090532 and Q.sjq_2_2_nch > 4.0:
        z += -0.003217239 * (9.090532 - Q.mass_displaced3) * (Q.sjq_2_2_nch - 4.0)
    if Q.lepsj_3_n_d3 < 3.0 and Q.lepsj_3_dr > 0.0:
        z += -0.4087872 * (3.0 - Q.lepsj_3_n_d3) * (Q.lepsj_3_dr - 0.0)
    if Q.sv_1_sd0_sum < 51.56287 and Q.dc_1_n_lep > 0.0:
        z += -0.002844845 * (51.56287 - Q.sv_1_sd0_sum) * (Q.dc_1_n_lep - 0.0)
    if Q.sjq_2_1_nch < 16.0 and Q.sum_e < 1651.766:
        z += -3.2231e-05 * (16.0 - Q.sjq_2_1_nch) * (1651.766 - Q.sum_e)
    if Q.n_real_top50 > 26.0 and Q.n_lund_kt_above_5 < 3.0:
        z += -0.005825325 * (Q.n_real_top50 - 26.0) * (3.0 - Q.n_lund_kt_above_5)
    if Q.lep_z < 0.221436 and Q.jet_charge_k05 > -0.2841306:
        z += -0.7006763 * (0.221436 - Q.lep_z) * (Q.jet_charge_k05 - -0.2841306)
    return z


def neuron_19(Q):
    z = -1.407816e-06
    return z


def neuron_20(Q):
    z = 8.253695e-07
    return z


def neuron_21(Q):
    z = -0.001682362
    return z


def neuron_22(Q):
    z = -1.044845e-06
    return z


def neuron_23(Q):
    z = -6.978682e-07
    return z


def neuron_24(Q):
    z = 0.09413664
    if Q.lep_z < 0.004135872:
        z += 60.7472 * Q.lep_z - 1.611957
    if 0.004135872 <= Q.lep_z < 0.3396572:
        z += 4.055522 * Q.lep_z - 1.377487
    if Q.mass_top30 < 74.51927:
        z += 0.0106023 * Q.mass_top30 - 0.4853889
    if 74.51927 <= Q.mass_top30 < 134.2224:
        z += -0.005103361 * Q.mass_top30 + 0.6849853
    if Q.mass < 90.08945:
        z += -0.001698509 * Q.mass - 0.3069249
    if 90.08945 <= Q.mass < 110.2019:
        z += -0.004950979 * Q.mass - 0.01391158
    if 110.2019 <= Q.mass < 117.4867:
        z += 0.01417157 * Q.mass - 2.121253
    if 117.4867 <= Q.mass < 164.4374:
        z += 0.009718307 * Q.mass - 1.598054
    z += 0.1827816 * Q.pair_max_lnkt
    if Q.z_displaced3 < 0.1061578:
        z += 2.916212 * Q.z_displaced3 - 0.3095786
    if Q.e3_b2 < 7.452974e-05:
        z += 2792.721 * Q.e3_b2 - 0.5897637
    if 7.452974e-05 <= Q.e3_b2 < 0.0004126585:
        z += 1128.632 * Q.e3_b2 - 0.4657397
    if Q.mres_sd_mass_b2z01 < 55.13212:
        z += -0.004021855 * Q.mres_sd_mass_b2z01 + 0.08647528
    if 55.13212 <= Q.mres_sd_mass_b2z01 < 81.19466:
        z += 0.02753505 * Q.mres_sd_mass_b2z01 - 1.653324
    if 81.19466 <= Q.mres_sd_mass_b2z01 < 118.2746:
        z += -0.01570592 * Q.mres_sd_mass_b2z01 + 1.857612
    if Q.n_s3d_above_10 >= 1.0:
        z += -0.04654606 * Q.n_s3d_above_10 + 0.04654606
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.04636267 * Q.lepsj_3_maxsd0 + 0.1118854
    if Q.lepsj_2_dr < 0.103005:
        z += 1.560294 * Q.lepsj_2_dr - 0.1607181
    if Q.mass_displaced3 < 18.80005:
        z += -0.02093958 * Q.mass_displaced3 + 0.3936652
    if Q.n_s3d_above_3 < 1.0:
        z += -0.1319875 * Q.n_s3d_above_3 + 0.1319875
    if Q.n_pairs_kt_above_1 < 366.0:
        z += -0.0003427703 * Q.n_pairs_kt_above_1 + 0.1254539
    if Q.N2 < 0.2423129:
        z += 3.894977 * Q.N2 - 0.9438031
    if Q.lep_iso < 0.4381892:
        z += -2.90965 * Q.lep_iso + 1.397466
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += -0.1325773 * Q.lep_iso + 0.1805828
    if Q.C2_b2 < 0.01057938:
        z += 24.73909 * Q.C2_b2 - 0.2617243
    if Q.sjq_3_2_k1 < -0.6014774:
        z += -2.409318 * Q.sjq_3_2_k1 - 1.44915
    if Q.sjq_2_2_k1 >= 0.277232:
        z += 0.6211605 * Q.sjq_2_2_k1 - 0.1722056
    if Q.jd_sum_abs_sd0_top5 < 84.11398:
        z += -0.01063192 * Q.jd_sum_abs_sd0_top5 + 0.8942928
    if Q.z_neutral_had < 0.212136:
        z += -1.768907 * Q.z_neutral_had + 0.3752488
    if Q.sip_3d_1 < 7.028704:
        z += -0.02154839 * Q.sip_3d_1 + 0.1514573
    if Q.lam1 < 0.006142967:
        z += 215.1838 * Q.lam1 - 1.321867
    if Q.z_dr_0p2_0p4 < 0.1468781:
        z += 3.125764 * Q.z_dr_0p2_0p4 - 0.4591064
    if Q.n_sd0_above_10 >= 6.0:
        z += 0.04165483 * Q.n_sd0_above_10 - 0.249929
    if Q.sj3_pair_mass_max < 83.63398:
        z += 0.003428948 * Q.sj3_pair_mass_max + 0.03851658
    if 83.63398 <= Q.sj3_pair_mass_max < 128.6605:
        z += -0.007224472 * Q.sj3_pair_mass_max + 0.9295045
    if Q.tau21 >= 0.135772:
        z += 0.7569013 * Q.tau21 - 0.102766
    if Q.tau32 < 0.456416:
        z += 4.646615 * Q.tau32 - 2.120789
    if Q.n_lund < 5.0:
        z += -0.3665165 * Q.n_lund + 1.832583
    if Q.dc_2_jp < 6.161303:
        z += -0.01695636 * Q.dc_2_jp + 0.1044733
    if Q.mres_pruned_mass < 60.18017:
        z += -0.007043571 * Q.mres_pruned_mass + 0.1391269
    if 60.18017 <= Q.mres_pruned_mass < 86.14266:
        z += 0.03452189 * Q.mres_pruned_mass - 2.362289
    if 86.14266 <= Q.mres_pruned_mass < 119.8459:
        z += -0.02379337 * Q.mres_pruned_mass + 2.661142
    if 119.8459 <= Q.mres_pruned_mass < 171.3819:
        z += 0.00369442 * Q.mres_pruned_mass - 0.6331569
    if 0.2293319 <= Q.dc_split2_dr < 0.3591078:
        z += 2.545459 * Q.dc_split2_dr - 0.5837551
    if Q.dc_split2_dr >= 0.3591078:
        z += -3.593459 * Q.dc_split2_dr + 1.620778
    if Q.n_dr_0p4_up < 4.0:
        z += -0.08994788 * Q.n_dr_0p4_up + 0.3597915
    if Q.mass_top15 >= 70.93762:
        z += -0.002243975 * Q.mass_top15 + 0.1591823
    if Q.max_abs_d0 < 10.52344:
        z += -0.01787142 * Q.max_abs_d0 + 0.1880688
    if Q.z_charged_had >= 0.5398733:
        z += -0.9612244 * Q.z_charged_had + 0.5189393
    if Q.n_dr_0p2_0p4 < 17.0:
        z += -0.04503234 * Q.n_dr_0p2_0p4 + 0.7655499
    if Q.dr_8 >= 0.3729651:
        z += -0.5387892 * Q.dr_8 + 0.2009496
    if Q.lep_ptrel < 12.15228:
        z += 0.008334485 * Q.lep_ptrel - 0.101283
    if Q.lep_z < 0.3396572 and Q.mass < 127.2317:
        z += 0.04492088 * (0.3396572 - Q.lep_z) * (127.2317 - Q.mass)
    if Q.lep_z < 0.3396572 and Q.tau32 < 0.456416:
        z += 21.82058 * (0.3396572 - Q.lep_z) * (0.456416 - Q.tau32)
    if Q.lep_ptrel < 43.20788 and Q.sdb_2_z > 0.2968888:
        z += 0.02058916 * (43.20788 - Q.lep_ptrel) * (Q.sdb_2_z - 0.2968888)
    if Q.e3_b2 < 0.0004126585 and Q.lne_8 > 2.11682:
        z += 248.8049 * (0.0004126585 - Q.e3_b2) * (Q.lne_8 - 2.11682)
    if Q.z_displaced3 < 0.1061578 and Q.pz_lnd2 < 0.1166862:
        z += -16.81014 * (0.1061578 - Q.z_displaced3) * (0.1166862 - Q.pz_lnd2)
    if Q.z_displaced3 < 0.1061578 and Q.D2_b2 < 15.17086:
        z += 0.2410171 * (0.1061578 - Q.z_displaced3) * (15.17086 - Q.D2_b2)
    if Q.mass < 164.4374 and Q.sv_1_n < 3.0:
        z += -0.00128397 * (164.4374 - Q.mass) * (3.0 - Q.sv_1_n)
    if Q.e3_b2 < 0.0004126585 and Q.sdb_0_n < 5.0:
        z += -139.7293 * (0.0004126585 - Q.e3_b2) * (5.0 - Q.sdb_0_n)
    if Q.mass < 164.4374 and Q.M3_b2 < 0.0309457:
        z += -0.07859527 * (164.4374 - Q.mass) * (0.0309457 - Q.M3_b2)
    if Q.lepsj_2_dr < 0.103005 and Q.jet_e > 926.0771:
        z += 0.001380983 * (0.103005 - Q.lepsj_2_dr) * (Q.jet_e - 926.0771)
    if Q.mass < 164.4374 and Q.n_dr_0p2_0p4 < 24.0:
        z += 6.472936e-05 * (164.4374 - Q.mass) * (24.0 - Q.n_dr_0p2_0p4)
    if Q.jd_sum_abs_sd0_top5 < 84.11398 and Q.jd_3d_5 < 6.771002:
        z += -0.002271191 * (84.11398 - Q.jd_sum_abs_sd0_top5) * (6.771002 - Q.jd_3d_5)
    if Q.mass_displaced3 < 18.80005 and Q.jd_3d_5 < 4.004982:
        z += 0.01262823 * (18.80005 - Q.mass_displaced3) * (4.004982 - Q.jd_3d_5)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lep_iso < 0.4381892:
        z += -0.08588921 * (83.63398 - Q.sj3_pair_mass_max) * (0.4381892 - Q.lep_iso)
    if Q.lep_z < 0.3396572 and Q.tau21 > 0.1757731:
        z += -2.688603 * (0.3396572 - Q.lep_z) * (Q.tau21 - 0.1757731)
    if Q.lep_z < 0.3396572 and Q.n_neutral_had < 7.0:
        z += -0.1540879 * (0.3396572 - Q.lep_z) * (7.0 - Q.n_neutral_had)
    if Q.mass_displaced3 < 18.80005 and Q.z_photon < 0.3090294:
        z += 0.05213864 * (18.80005 - Q.mass_displaced3) * (0.3090294 - Q.z_photon)
    if Q.z_displaced3 < 0.1061578 and Q.mass_2photon > 3.013:
        z += 0.06655432 * (0.1061578 - Q.z_displaced3) * (Q.mass_2photon - 3.013)
    if Q.e3_b2 < 0.0004126585 and Q.kt2_min12_n_disp3 > 1.0:
        z += 443.9901 * (0.0004126585 - Q.e3_b2) * (Q.kt2_min12_n_disp3 - 1.0)
    if Q.lep_z < 0.004135872 and Q.sdb_4_z < 0.01435877:
        z += 4317.387 * (0.004135872 - Q.lep_z) * (0.01435877 - Q.sdb_4_z)
    if Q.lep_z < 0.3396572 and Q.e4 > 4.06096e-07:
        z += 26028.64 * (0.3396572 - Q.lep_z) * (Q.e4 - 4.06096e-07)
    if Q.lep_z < 0.004135872 and Q.n_pairs_kt_above_10 > 4.0:
        z += -1.88512 * (0.004135872 - Q.lep_z) * (Q.n_pairs_kt_above_10 - 4.0)
    if Q.sjq_3_2_k1 < -0.6014774 and Q.lep_iso < 0.4381892:
        z += -4.503736 * (-0.6014774 - Q.sjq_3_2_k1) * (0.4381892 - Q.lep_iso)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lep_iso < 14.38858:
        z += 0.001431897 * (83.63398 - Q.sj3_pair_mass_max) * (14.38858 - Q.lep_iso)
    if Q.lepsj_2_dr < 0.103005 and Q.lep_iso < 14.38858:
        z += -0.8775225 * (0.103005 - Q.lepsj_2_dr) * (14.38858 - Q.lep_iso)
    if Q.sjq_2_2_k1 > 0.277232 and Q.lep_iso < 2.907433:
        z += -0.09614089 * (Q.sjq_2_2_k1 - 0.277232) * (2.907433 - Q.lep_iso)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lepsj_2_dr < 0.1822601:
        z += 0.1009342 * (83.63398 - Q.sj3_pair_mass_max) * (0.1822601 - Q.lepsj_2_dr)
    if Q.lep_z < 0.004135872 and Q.lep_iso < 2.907433:
        z += -107.2658 * (0.004135872 - Q.lep_z) * (2.907433 - Q.lep_iso)
    if Q.lep_z < 0.3396572 and Q.lepsj_3_maxsd0 < 0.8036986:
        z += 2.636026 * (0.3396572 - Q.lep_z) * (0.8036986 - Q.lepsj_3_maxsd0)
    if Q.lep_z < 0.3396572 and Q.sj3_dr12 < 0.3683248:
        z += -1.616421 * (0.3396572 - Q.lep_z) * (0.3683248 - Q.sj3_dr12)
    if Q.mass_displaced3 < 18.80005 and Q.sip_3d_2 < 2325.187:
        z += -1.088262e-05 * (18.80005 - Q.mass_displaced3) * (2325.187 - Q.sip_3d_2)
    if Q.max_abs_d0 < 10.52344 and Q.jd_3d_5 < 4.004982:
        z += -0.007639198 * (10.52344 - Q.max_abs_d0) * (4.004982 - Q.jd_3d_5)
    if Q.mass_top15 > 70.93762 and Q.sv_2_z > 0.00726873:
        z += -0.1963286 * (Q.mass_top15 - 70.93762) * (Q.sv_2_z - 0.00726873)
    if Q.lep_iso < 0.4381892 and Q.sdb_4_n > 0.0:
        z += 0.2015499 * (0.4381892 - Q.lep_iso) * (Q.sdb_4_n - 0.0)
    if Q.lep_iso < 0.4381892 and Q.lepsj_2_dr < 0.1822601:
        z += 8.473043 * (0.4381892 - Q.lep_iso) * (0.1822601 - Q.lepsj_2_dr)
    if Q.mass_top15 > 70.93762 and Q.sv_2_dr < 0.1572038:
        z += -0.00955666 * (Q.mass_top15 - 70.93762) * (0.1572038 - Q.sv_2_dr)
    return z


def neuron_25(Q):
    z = 0.03116744
    return z


def neuron_26(Q):
    z = -3.231261e-06
    return z


def neuron_27(Q):
    z = -6.308503e-06
    return z


def neuron_28(Q):
    z = 9.617207e-06
    return z


def neuron_29(Q):
    z = 9.183867e-06
    return z


def neuron_30(Q):
    z = -8.603049e-06
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
    z = -6.743299e-06
    return z


def neuron_36(Q):
    z = -4.831948e-06
    return z


def neuron_37(Q):
    z = -4.357438e-06
    return z


def neuron_38(Q):
    z = -9.411175e-06
    return z


def neuron_39(Q):
    z = 8.27428e-06
    return z


def neuron_40(Q):
    z = 7.775539e-07
    return z


def neuron_41(Q):
    z = 3.814734e-08
    return z


def neuron_42(Q):
    z = 8.779051e-06
    return z


def neuron_43(Q):
    z = -9.407709e-06
    return z


def neuron_44(Q):
    z = -3.324349e-06
    return z


def neuron_45(Q):
    z = 6.849351e-06
    return z


def neuron_46(Q):
    z = 1.368759e-07
    return z


def neuron_47(Q):
    z = 2.492075e-07
    return z


def neuron_48(Q):
    z = -8.791778e-06
    return z


def neuron_49(Q):
    z = -2.243815e-06
    return z


def neuron_50(Q):
    z = 5.175118e-06
    return z


def neuron_51(Q):
    z = -2.012202e-06
    return z


def neuron_52(Q):
    z = -1.062184
    if Q.lep_z < 0.221436:
        z += 1.274566 * Q.lep_z - 0.2058531
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -0.6460913 * Q.lep_z + 0.2194496
    if Q.z_displaced3 < 0.1342762:
        z += -6.173054 * Q.z_displaced3 + 0.8288943
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.1722473 * Q.lepsj_3_maxsd0 + 0.4156782
    if Q.mass_top40 < 70.88236:
        z += 0.005059991 * Q.mass_top40 - 0.3586641
    if Q.mres_sd_mass_b0z005 < 81.62059:
        z += -0.00652509 * Q.mres_sd_mass_b0z005 + 1.490936
    if 81.62059 <= Q.mres_sd_mass_b0z005 < 131.0776:
        z += -0.002644995 * Q.mres_sd_mass_b0z005 + 1.17424
    if 131.0776 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += -0.01815511 * Q.mres_sd_mass_b0z005 + 3.207268
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += 0.003880095 * Q.mres_sd_mass_b0z005 - 0.3166956
    if Q.mass >= 182.8592:
        z += -0.01032129 * Q.mass + 1.887343
    if Q.lne_16 >= 0.4468807:
        z += -0.1254205 * Q.lne_16 + 0.05604801
    if Q.lep_ptrel < 12.15228:
        z += 0.04274808 * Q.lep_ptrel - 0.5944036
    if 12.15228 <= Q.lep_ptrel < 18.7678:
        z += 0.004938057 * Q.lep_ptrel - 0.1349255
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += 0.01979724 * Q.lep_ptrel - 0.4137997
    if Q.lep_ptrel >= 27.3236:
        z += 0.01485918 * Q.lep_ptrel - 0.2788742
    if Q.lepsj_2_dr < 0.103005:
        z += 5.796117 * Q.lepsj_2_dr - 0.5970289
    if Q.sv_1_sd0_sum < 125.2765:
        z += 0.001806679 * Q.sv_1_sd0_sum - 0.2263345
    if Q.sjq_2_2_k1 < -0.6337755:
        z += -2.055657 * Q.sjq_2_2_k1 - 1.302825
    if Q.sjq_2_2_k1 >= 0.6477929:
        z += 1.68999 * Q.sjq_2_2_k1 - 1.094763
    if Q.pz_lnd0 < 0.1122946:
        z += -2.17811 * Q.pz_lnd0 + 0.2445899
    if Q.n_s3d_above_3 < 3.0:
        z += -0.142516 * Q.n_s3d_above_3 + 0.4275481
    if Q.lep_dr < 0.3014662:
        z += -3.04431 * Q.lep_dr + 0.9177564
    if Q.lep_dr >= 0.4191372:
        z += -1.121735 * Q.lep_dr + 0.4701608
    if Q.D2_b05 < 1.45443:
        z += 0.4460647 * Q.D2_b05 - 0.6487699
    if Q.n_lund < 7.0:
        z += -0.1395013 * Q.n_lund + 0.9765093
    if Q.lep_iso < 1.362094:
        z += -0.09253263 * Q.lep_iso + 0.1260382
    if Q.n_lepton < 1.0:
        z += 0.8384337 * Q.n_lepton - 0.8384337
    if Q.sj2_mass2 < 21.7295:
        z += 0.01205942 * Q.sj2_mass2 - 0.2620452
    if Q.dc_1_n_lep < 1.0:
        z += -0.3334205 * Q.dc_1_n_lep + 0.3334205
    if Q.jd_sum_abs_sd0_top5 < 84.11398:
        z += 0.00440592 * Q.jd_sum_abs_sd0_top5 - 0.3705994
    if Q.sj3_pair_mass_min >= 80.02563:
        z += 0.02197828 * Q.sj3_pair_mass_min - 1.758826
    if Q.lund3_lndelta >= -2.817283:
        z += -0.07449829 * Q.lund3_lndelta - 0.2098827
    if Q.jet_charge_k03 >= -0.2409073:
        z += -0.1096894 * Q.jet_charge_k03 - 0.02642498
    if Q.ak02_2_z < 0.2837384:
        z += -2.602231 * Q.ak02_2_z + 0.7383529
    if Q.sjf_2_2_max3d < 3.029824:
        z += -0.1037733 * Q.sjf_2_2_max3d + 0.3144146
    if Q.n_dr_0p2_0p4 < 21.0:
        z += 0.02059131 * Q.n_dr_0p2_0p4 - 0.4324175
    if Q.sj2_dr < 0.3740528:
        z += -0.7591812 * Q.sj2_dr + 0.2839738
    if Q.sjq_3_3_k1 >= 0.6535817:
        z += 0.4981388 * Q.sjq_3_3_k1 - 0.3255744
    if Q.mass_2charged < 6.767937:
        z += 0.01075594 * Q.mass_2charged - 0.0727955
    if Q.mass_2charged >= 14.28253:
        z += -0.008392892 * Q.mass_2charged + 0.1198717
    if Q.N3_b05 >= 0.7591346:
        z += -0.09051268 * Q.N3_b05 + 0.06871131
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.007890109 * Q.n_pairs_kt_above_3 + 0.2209231
    if Q.sdb_2_n >= 9.0:
        z += -0.04176984 * Q.sdb_2_n + 0.3759285
    if Q.sj3_mass1 >= 36.36236:
        z += -0.01204863 * Q.sj3_mass1 + 0.4381166
    if Q.D2 < 1.421532:
        z += 0.09437197 * Q.D2 - 0.1341528
    if Q.tau21_b2 < 0.3565533:
        z += -3.081039 * Q.tau21_b2 + 1.098555
    if Q.ak02_1_z >= 0.6342743:
        z += -1.224412 * Q.ak02_1_z + 0.7766133
    if Q.mass_top50 < 79.27954:
        z += -0.02023601 * Q.mass_top50 + 1.604302
    if Q.tau4 < 0.01143413:
        z += -81.21446 * Q.tau4 + 0.9286168
    if Q.e3_b2 < 0.0004126585:
        z += -887.1065 * Q.e3_b2 + 0.3660721
    if Q.n_dr_0p4_up < 10.0:
        z += 0.02065361 * Q.n_dr_0p4_up - 0.2065361
    if Q.lund_max_lndelta >= -0.615738:
        z += 0.907127 * Q.lund_max_lndelta + 0.5585526
    if Q.N2_b2 >= 0.08291719:
        z += 1.377924 * Q.N2_b2 - 0.1142536
    if Q.z_displaced3 < 0.1342762 and Q.N2 > 0.1824346:
        z += 8.153059 * (0.1342762 - Q.z_displaced3) * (Q.N2 - 0.1824346)
    if Q.lep_z < 0.3396572 and Q.n_charged_pt_above_1 > 11.0:
        z += 0.1045769 * (0.3396572 - Q.lep_z) * (Q.n_charged_pt_above_1 - 11.0)
    if Q.lep_z < 0.3396572 and Q.sj3_pair_mass_min > 48.40547:
        z += -0.02276082 * (0.3396572 - Q.lep_z) * (Q.sj3_pair_mass_min - 48.40547)
    if Q.mres_sd_mass_b0z005 < 159.9242 and Q.tau21_b2 < 0.3565533:
        z += -0.03763692 * (159.9242 - Q.mres_sd_mass_b0z005) * (0.3565533 - Q.tau21_b2)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.tau32 < 0.7544983:
        z += -0.004668527 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.7544983 - Q.tau32)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.sjf_2_2_z_d3 < 0.03385157:
        z += 0.0600669 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.03385157 - Q.sjf_2_2_z_d3)
    if Q.mres_sd_mass_b0z005 < 159.9242 and Q.mass_displaced3 > 3.208089:
        z += -0.0002471949 * (159.9242 - Q.mres_sd_mass_b0z005) * (Q.mass_displaced3 - 3.208089)
    if Q.z_neutral_had < 0.3788785 and Q.ecf_g41 > 6.216954e-05:
        z += 2844.682 * (0.3788785 - Q.z_neutral_had) * (Q.ecf_g41 - 6.216954e-05)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.ak02_3_z < 0.07650476:
        z += 0.1147755 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.07650476 - Q.ak02_3_z)
    if Q.lepsj_2_dr < 0.103005 and Q.C3_b05 < 0.2250047:
        z += 9.536534 * (0.103005 - Q.lepsj_2_dr) * (0.2250047 - Q.C3_b05)
    if Q.mres_sd_mass_b0z005 < 131.0776 and Q.sip_3d_3 < 23.30856:
        z += -4.392182e-05 * (131.0776 - Q.mres_sd_mass_b0z005) * (23.30856 - Q.sip_3d_3)
    if Q.lep_dr < 0.3014662 and Q.sdb_2_z > 0.1530389:
        z += 1.529963 * (0.3014662 - Q.lep_dr) * (Q.sdb_2_z - 0.1530389)
    if Q.z_displaced3 < 0.1342762 and Q.sip_3d_3 < 577.991:
        z += -0.007498152 * (0.1342762 - Q.z_displaced3) * (577.991 - Q.sip_3d_3)
    if Q.lund3_lndelta > -2.817283 and Q.sv_1_dr < 0.3689526:
        z += 0.2238107 * (Q.lund3_lndelta - -2.817283) * (0.3689526 - Q.sv_1_dr)
    if Q.sj2_mass2 < 21.7295 and Q.nca_sj4_pairmax_over_mass < 0.7817308:
        z += 0.0413951 * (21.7295 - Q.sj2_mass2) * (0.7817308 - Q.nca_sj4_pairmax_over_mass)
    if Q.z_neutral_had < 0.3788785 and Q.pz_lnkt1 > 0.03633353:
        z += -8.742846 * (0.3788785 - Q.z_neutral_had) * (Q.pz_lnkt1 - 0.03633353)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_2_prod_k05 > -0.5057096:
        z += -0.01285099 * (21.0 - Q.n_dr_0p2_0p4) * (Q.sjq_2_prod_k05 - -0.5057096)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_3_sumabs_k1 > 0.3376898:
        z += 0.01534015 * (21.0 - Q.n_dr_0p2_0p4) * (Q.sjq_3_sumabs_k1 - 0.3376898)
    if Q.z_displaced3 < 0.1342762 and Q.n_lund_kt_above_1 > 3.0:
        z += 0.2648037 * (0.1342762 - Q.z_displaced3) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.lepsj_2_dr < 0.103005 and Q.z_photon > 0.3318968:
        z += 7.325321 * (0.103005 - Q.lepsj_2_dr) * (Q.z_photon - 0.3318968)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_3_prod_k1 < 0.1145956:
        z += 0.03309149 * (21.0 - Q.n_dr_0p2_0p4) * (0.1145956 - Q.sjq_3_prod_k1)
    if Q.z_displaced3 < 0.1342762 and Q.jd_3d_6 < 5.367501:
        z += 1.321681 * (0.1342762 - Q.z_displaced3) * (5.367501 - Q.jd_3d_6)
    if Q.n_lund < 7.0 and Q.jd_3d_6 < 2.591685:
        z += -0.03295081 * (7.0 - Q.n_lund) * (2.591685 - Q.jd_3d_6)
    if Q.D3_b2 < 0.001184159 and Q.jd_3d_6 < 5.367501:
        z += -52.29578 * (0.001184159 - Q.D3_b2) * (5.367501 - Q.jd_3d_6)
    if Q.sdb_2_n > 9.0 and Q.D2_b2 < 2.01745:
        z += 0.03980899 * (Q.sdb_2_n - 9.0) * (2.01745 - Q.D2_b2)
    if Q.z_displaced3 < 0.1342762 and Q.jd_3d_6 < 30.57088:
        z += -0.2182351 * (0.1342762 - Q.z_displaced3) * (30.57088 - Q.jd_3d_6)
    if Q.mres_sd_mass_b0z005 < 131.0776 and Q.D2_b2 < 1.204663:
        z += -0.006806259 * (131.0776 - Q.mres_sd_mass_b0z005) * (1.204663 - Q.D2_b2)
    return z


def neuron_53(Q):
    z = -1.889364e-06
    return z


def neuron_54(Q):
    z = 1.19066e-05
    return z


def neuron_55(Q):
    z = -2.06694e-05
    return z


def neuron_56(Q):
    z = -7.072325e-07
    return z


def neuron_57(Q):
    z = -1.959647e-06
    return z


def neuron_58(Q):
    z = -3.617151e-06
    return z


def neuron_59(Q):
    z = 6.029336e-06
    return z


def neuron_60(Q):
    z = -4.666765e-06
    return z


def neuron_61(Q):
    z = 5.840561e-06
    return z


def neuron_62(Q):
    z = 9.422977e-07
    return z


def neuron_63(Q):
    z = -8.308513e-06
    return z


def neuron_64(Q):
    z = 1.25863e-06
    return z


def neuron_65(Q):
    z = 5.791055e-06
    return z


def neuron_66(Q):
    z = 2.709779e-06
    return z


def neuron_67(Q):
    z = -5.219512e-06
    return z


def neuron_68(Q):
    z = 4.481455
    if Q.tau32 < 0.7981752:
        z += -0.6720207 * Q.tau32 + 0.5363902
    if 6.983043 <= Q.lep_ptrel < 18.7678:
        z += -0.02404253 * Q.lep_ptrel + 0.16789
    if 18.7678 <= Q.lep_ptrel < 43.20788:
        z += 0.01897851 * Q.lep_ptrel - 0.6395203
    if Q.lep_ptrel >= 43.20788:
        z += -0.02754648 * Q.lep_ptrel + 1.370726
    if Q.pair_mean_lnm2 < 3.38061:
        z += 0.04416516 * Q.pair_mean_lnm2 - 0.1493052
    if Q.pz_lnd2 < 0.1339824:
        z += 2.293248 * Q.pz_lnd2 - 0.3072548
    if Q.e3_b2 < 0.0004126585:
        z += 1395.003 * Q.e3_b2 - 0.57566
    if Q.sj3_pair_mass_max < 56.73313:
        z += 0.03000136 * Q.sj3_pair_mass_max - 1.702071
    if 83.63398 <= Q.sj3_pair_mass_max < 128.6605:
        z += -0.008224964 * Q.sj3_pair_mass_max + 0.6878865
    if Q.sj3_pair_mass_max >= 128.6605:
        z += 0.01155023 * Q.sj3_pair_mass_max - 1.856401
    if Q.mass < 90.08945:
        z += 0.005788903 * Q.mass - 1.927001
    if 90.08945 <= Q.mass < 110.2019:
        z += -0.002364683 * Q.mass - 1.192449
    if 110.2019 <= Q.mass < 117.4867:
        z += 0.02515277 * Q.mass - 4.224924
    if 117.4867 <= Q.mass < 164.4374:
        z += 0.02704554 * Q.mass - 4.447299
    if Q.sj4_pair_mass_max >= 69.83554:
        z += -0.001922032 * Q.sj4_pair_mass_max + 0.1342261
    if Q.z_charged_had >= 0.265564:
        z += 0.6679449 * Q.z_charged_had - 0.1773821
    if Q.M2_b2 < 0.05966366:
        z += 6.139922 * Q.M2_b2 - 0.3663302
    if Q.mres_sd_mass_b2z01 < 55.13212:
        z += -0.004823666 * Q.mres_sd_mass_b2z01 - 0.8606962
    if 55.13212 <= Q.mres_sd_mass_b2z01 < 76.1529:
        z += 0.04484187 * Q.mres_sd_mass_b2z01 - 3.598863
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 118.2746:
        z += -0.02179578 * Q.mres_sd_mass_b2z01 + 1.475788
    if 118.2746 <= Q.mres_sd_mass_b2z01 < 130.0968:
        z += 0.02471631 * Q.mres_sd_mass_b2z01 - 4.025413
    if 130.0968 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.02881688 * Q.mres_sd_mass_b2z01 - 4.558884
    z += -0.06457409 * Q.n_particles
    if Q.dc_split2_dr >= 0.3591078:
        z += -2.197792 * Q.dc_split2_dr + 0.7892441
    if Q.lep_iso < 0.4381892:
        z += 1.37893 * Q.lep_iso - 0.6042321
    if Q.lep_dr < 0.04607888:
        z += -7.008069 * Q.lep_dr + 0.322924
    if Q.sj3_pair_mass_min < 80.02563:
        z += 0.009120784 * Q.sj3_pair_mass_min - 0.7298964
    if Q.sdb_2_n < 7.0:
        z += -0.02383854 * Q.sdb_2_n + 0.1668698
    if Q.sdb_2_n >= 12.0:
        z += -0.02351745 * Q.sdb_2_n + 0.2822094
    if Q.M3 < 0.03259227:
        z += -14.08141 * Q.M3 + 0.4589451
    if Q.mass_top20 >= 130.7093:
        z += -0.01550613 * Q.mass_top20 + 2.026795
    if Q.jet_abs_eta >= 0.5325716:
        z += 0.3686769 * Q.jet_abs_eta - 0.1963468
    z += 0.03947924 * Q.n_photon
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.007910009 * Q.n_pairs_kt_above_1 - 0.6328008
    if Q.sjq_2_2_nch < 3.0:
        z += 0.1395783 * Q.sjq_2_2_nch - 0.4187349
    if Q.sj2_dr < 0.2403736:
        z += 1.067075 * Q.sj2_dr - 0.2564966
    if Q.tau5 < 0.05613495:
        z += -16.11404 * Q.tau5 + 0.904561
    if Q.ktd_ln_d34 >= -9.359695:
        z += 0.1692034 * Q.ktd_ln_d34 + 1.583692
    if Q.D2_b2 < 1.774856:
        z += 0.1753276 * Q.D2_b2 - 0.3111813
    if Q.nca_sj4_pair_mass_2nd >= 72.72359:
        z += -0.01561628 * Q.nca_sj4_pair_mass_2nd + 1.135672
    if Q.ak02_dr12 >= 0.3882673:
        z += -0.3166561 * Q.ak02_dr12 + 0.1229472
    if Q.sjf_3_1_n_d3 >= 6.0:
        z += 0.04920374 * Q.sjf_3_1_n_d3 - 0.2952224
    if 47.97531 <= Q.mres_sd_mass_b0z02 < 86.60355:
        z += 0.00398337 * Q.mres_sd_mass_b0z02 - 0.1911034
    if 86.60355 <= Q.mres_sd_mass_b0z02 < 134.2023:
        z += -0.006865197 * Q.mres_sd_mass_b0z02 + 0.748421
    if Q.mres_sd_mass_b0z02 >= 134.2023:
        z += 0.003523157 * Q.mres_sd_mass_b0z02 - 0.6457201
    if 86.14266 <= Q.mres_pruned_mass < 124.3145:
        z += -0.009463071 * Q.mres_pruned_mass + 0.8151741
    if 124.3145 <= Q.mres_pruned_mass < 171.3819:
        z += 0.006479516 * Q.mres_pruned_mass - 1.166721
    if Q.mres_pruned_mass >= 171.3819:
        z += 0.001777862 * Q.mres_pruned_mass - 0.3609427
    if Q.pz_lnd3 >= 0.01024929:
        z += 0.4896849 * Q.pz_lnd3 - 0.005018924
    if Q.mres_sd_rg_b2z01 >= 0.457968:
        z += -1.082904 * Q.mres_sd_rg_b2z01 + 0.4959352
    if Q.n_sdz_above_5 < 3.0:
        z += -0.07335697 * Q.n_sdz_above_5 + 0.2200709
    if Q.sum_z_dr2_top2 < 0.005938474:
        z += -19.69227 * Q.sum_z_dr2_top2 + 0.116942
    if Q.mass_top10 < 103.4976:
        z += 0.001024113 * Q.mass_top10 - 0.1059933
    if Q.jd_3d_5 < 4.004982:
        z += 0.07023316 * Q.jd_3d_5 - 0.2812825
    if Q.tau32 < 0.7981752 and Q.sj3_dr_min < 0.3190414:
        z += 2.084139 * (0.7981752 - Q.tau32) * (0.3190414 - Q.sj3_dr_min)
    if Q.pair_mean_lnm2 < 3.38061 and Q.z_top30_slots > 0.8316085:
        z += 0.293219 * (3.38061 - Q.pair_mean_lnm2) * (Q.z_top30_slots - 0.8316085)
    if Q.e3_b2 < 0.0004126585 and Q.lnptrel_25 > -6.691703:
        z += 330.5634 * (0.0004126585 - Q.e3_b2) * (Q.lnptrel_25 - -6.691703)
    if Q.e3_b2 < 0.0004126585 and Q.ak02_dr23 > 0.3194837:
        z += 1339.913 * (0.0004126585 - Q.e3_b2) * (Q.ak02_dr23 - 0.3194837)
    if Q.sj4_pair_mass_max > 69.83554 and Q.nca_sj4_pair2nd_over_mass > 0.4488456:
        z += 0.02718156 * (Q.sj4_pair_mass_max - 69.83554) * (Q.nca_sj4_pair2nd_over_mass - 0.4488456)
    if Q.sj3_pair_mass_max > 128.6605 and Q.jet_e > 551.9263:
        z += 5.487977e-06 * (Q.sj3_pair_mass_max - 128.6605) * (Q.jet_e - 551.9263)
    if Q.mass < 117.4867 and Q.D2_b2 < 1.380731:
        z += 0.007766805 * (117.4867 - Q.mass) * (1.380731 - Q.D2_b2)
    if Q.e3_b2 < 0.0004126585 and Q.sj3_mass1 > 17.31003:
        z += 22.12907 * (0.0004126585 - Q.e3_b2) * (Q.sj3_mass1 - 17.31003)
    if Q.sj4_pair_mass_max > 69.83554 and Q.dc_3_z < 0.2037349:
        z += -0.02873186 * (Q.sj4_pair_mass_max - 69.83554) * (0.2037349 - Q.dc_3_z)
    if Q.lep_ptrel > 6.983043 and Q.C2_b2 < 0.1682645:
        z += 0.04672761 * (Q.lep_ptrel - 6.983043) * (0.1682645 - Q.C2_b2)
    if Q.sj3_pair_mass_min < 80.02563 and Q.mass_displaced5 > 0.0:
        z += 0.0001476139 * (80.02563 - Q.sj3_pair_mass_min) * (Q.mass_displaced5 - 0.0)
    if Q.sj3_pair_mass_max > 128.6605 and Q.e4 > 7.184834e-06:
        z += 245.6859 * (Q.sj3_pair_mass_max - 128.6605) * (Q.e4 - 7.184834e-06)
    if Q.pz_lnd2 < 0.1339824 and Q.sj3_dr13 > 0.4691911:
        z += -6.310158 * (0.1339824 - Q.pz_lnd2) * (Q.sj3_dr13 - 0.4691911)
    if Q.mass < 164.4374 and Q.sjf_4_3_z_d3 > 0.0:
        z += -0.02562284 * (164.4374 - Q.mass) * (Q.sjf_4_3_z_d3 - 0.0)
    if Q.z_displaced3 < 0.1061578 and Q.lepsj_2_maxsd0 < 0.7828545:
        z += -5.278854 * (0.1061578 - Q.z_displaced3) * (0.7828545 - Q.lepsj_2_maxsd0)
    if Q.z_charged_had > 0.265564 and Q.pz_lnkt3 > 0.0:
        z += 3.116497 * (Q.z_charged_had - 0.265564) * (Q.pz_lnkt3 - 0.0)
    if Q.lep_dr < 0.04607888 and Q.lepsj_2_maxsd0 < 0.7828545:
        z += 12.00342 * (0.04607888 - Q.lep_dr) * (0.7828545 - Q.lepsj_2_maxsd0)
    if Q.sj4_pair_mass_max > 69.83554 and Q.sv_1_dr < 0.3689526:
        z += -0.006153722 * (Q.sj4_pair_mass_max - 69.83554) * (0.3689526 - Q.sv_1_dr)
    if Q.mass < 117.4867 and Q.C2_b2 > 0.01891146:
        z += -0.04496719 * (117.4867 - Q.mass) * (Q.C2_b2 - 0.01891146)
    if Q.z_displaced3 < 0.1061578 and Q.dr_29 < 0.02258554:
        z += 73.9789 * (0.1061578 - Q.z_displaced3) * (0.02258554 - Q.dr_29)
    if Q.e3_b2 < 0.0004126585 and Q.n_lund_kt_above_5 > 2.0:
        z += 219.1762 * (0.0004126585 - Q.e3_b2) * (Q.n_lund_kt_above_5 - 2.0)
    if Q.z_displaced3 < 0.1061578 and Q.n_lund > 10.0:
        z += -0.5392346 * (0.1061578 - Q.z_displaced3) * (Q.n_lund - 10.0)
    if Q.M3 < 0.03259227 and Q.jd_3d_5 < 4.004982:
        z += -2.998122 * (0.03259227 - Q.M3) * (4.004982 - Q.jd_3d_5)
    return z


def neuron_69(Q):
    z = -6.926144e-06
    return z


def neuron_70(Q):
    z = 0.01609948
    return z


def neuron_71(Q):
    z = 3.029709e-06
    return z


def neuron_72(Q):
    z = -1.164121e-06
    return z


def neuron_73(Q):
    z = -1.699641e-05
    return z


def neuron_74(Q):
    z = -4.493398e-06
    return z


def neuron_75(Q):
    z = 1.973841e-06
    return z


def neuron_76(Q):
    z = -5.542788e-06
    return z


def neuron_77(Q):
    z = -3.087491e-06
    return z


def neuron_78(Q):
    z = 0.2329084
    if Q.lep_z < 0.3396572:
        z += 3.228388 * Q.lep_z - 1.096545
    if Q.n_s3d_above_3 < 4.0:
        z += -0.05943179 * Q.n_s3d_above_3 + 0.5348861
    if 4.0 <= Q.n_s3d_above_3 < 6.0:
        z += -0.09162893 * Q.n_s3d_above_3 + 0.6636747
    if 6.0 <= Q.n_s3d_above_3 < 9.0:
        z += -0.05389724 * Q.n_s3d_above_3 + 0.4372846
    if Q.n_s3d_above_3 >= 9.0:
        z += 0.005534548 * Q.n_s3d_above_3 - 0.09760157
    if Q.n_pairs_kt_above_1 < 195.0:
        z += -0.002994142 * Q.n_pairs_kt_above_1 + 0.5838576
    if Q.M3_b2 < 0.0142807:
        z += 13.77212 * Q.M3_b2 - 0.1966756
    if 9.380468 <= Q.mass_displaced5 < 21.12768:
        z += 0.01215313 * Q.mass_displaced5 - 0.114002
    if Q.mass_displaced5 >= 21.12768:
        z += -0.004243688 * Q.mass_displaced5 + 0.2324246
    if Q.lep_ptrel < 18.7678:
        z += 0.05819431 * Q.lep_ptrel - 0.8011871
    if 18.7678 <= Q.lep_ptrel < 43.20788:
        z += -0.01190635 * Q.lep_ptrel + 0.514448
    if Q.mass < 95.14961:
        z += 0.002358645 * Q.mass - 0.2244242
    if Q.mass >= 114.0172:
        z += 0.006276541 * Q.mass - 0.715634
    if Q.mass_top40 < 70.88236:
        z += -0.02398229 * Q.mass_top40 + 1.699921
    if Q.n_neutral < 22.0:
        z += -0.01666929 * Q.n_neutral + 0.3667243
    if Q.pair_max_lnm2 >= 7.347625:
        z += -0.354468 * Q.pair_max_lnm2 + 2.604498
    if Q.sdb_2_n < 4.0:
        z += -0.02041298 * Q.sdb_2_n + 0.1837169
    if 4.0 <= Q.sdb_2_n < 9.0:
        z += -0.01097155 * Q.sdb_2_n + 0.1459511
    if Q.sdb_2_n >= 9.0:
        z += 0.009441436 * Q.sdb_2_n - 0.03776575
    if Q.z_top15_slots >= 0.8008865:
        z += -2.009609 * Q.z_top15_slots + 1.609469
    if Q.n_sd0_above_5 >= 4.0:
        z += -0.02532572 * Q.n_sd0_above_5 + 0.1013029
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.009253883 * Q.mres_sd_mass_b0z005 - 1.129899
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.0141121 * Q.mres_sd_mass_b0z005 + 2.606888
    if Q.mass_top20 >= 130.7093:
        z += 0.009414964 * Q.mass_top20 - 1.230623
    if Q.lund3_lndelta >= -2.817283:
        z += 0.08012384 * Q.lund3_lndelta + 0.2257315
    if Q.n_charged_had >= 22.0:
        z += 0.02650928 * Q.n_charged_had - 0.5832042
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.1326921 * Q.lepsj_3_maxsd0 + 0.1641061
    if 2.413264 <= Q.lepsj_3_maxsd0 < 586.9572:
        z += 0.0002670715 * Q.lepsj_3_maxsd0 - 0.1567595
    if Q.n_lepton < 1.0:
        z += 0.603238 * Q.n_lepton - 0.5491526
    if 1.0 <= Q.n_lepton < 2.0:
        z += -0.05408543 * Q.n_lepton + 0.1081709
    if Q.mass_displaced3 >= 39.09615:
        z += 0.00961632 * Q.mass_displaced3 - 0.3759611
    if Q.n_pairs_kt_above_3 < 99.0:
        z += -0.003677734 * Q.n_pairs_kt_above_3 + 0.3640957
    if Q.mass_2charged < 1.839882:
        z += -0.1214438 * Q.mass_2charged + 0.1191215
    if 1.839882 <= Q.mass_2charged < 10.83159:
        z += 0.01160191 * Q.mass_2charged - 0.1256671
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += 0.08034984 * Q.sjf_2_1_n_d3 - 0.4017492
    if Q.max_abs_d0 < 10.52344:
        z += 0.009041567 * Q.max_abs_d0 - 0.09514837
    if Q.sj4_pair_mass_max < 115.5142:
        z += 0.001907885 * Q.sj4_pair_mass_max - 0.2203878
    if Q.lepsj_2_dr < 0.103005:
        z += 5.907772 * Q.lepsj_2_dr - 0.6085299
    if Q.sdb_0_z < 0.0772633:
        z += -1.017037 * Q.sdb_0_z + 0.07857966
    if Q.mass_top15 >= 83.68425:
        z += 0.006189616 * Q.mass_top15 - 0.5179734
    if Q.sdb_2_z < 0.2040425:
        z += 1.72622 * Q.sdb_2_z - 0.3522223
    if Q.mass_top10 >= 103.4976:
        z += -0.00572762 * Q.mass_top10 + 0.5927952
    if Q.n_pairs_kt_above_10 < 7.0:
        z += -0.01429139 * Q.n_pairs_kt_above_10 + 0.1000397
    if Q.N2_b05 < 0.5083429:
        z += -3.53221 * Q.N2_b05 + 1.795574
    if Q.ktd_ln_d34 < -8.400697:
        z += 0.1111364 * Q.ktd_ln_d34 + 0.9336228
    if Q.lepsj_2_maxsd0 < 2.220372:
        z += -0.0290978 * Q.lepsj_2_maxsd0 + 0.06460796
    if Q.kt2_1_z < 0.8153708:
        z += 0.272042 * Q.kt2_1_z - 0.2218151
    if Q.sj3_pair_mass_min < 37.19471:
        z += 0.003943467 * Q.sj3_pair_mass_min - 0.1466761
    if Q.mass_neutral < 46.19491:
        z += -0.002505849 * Q.mass_neutral + 0.1157575
    if Q.lep_z < 0.3396572 and Q.pz_lnd2 < 0.1956014:
        z += 2.296817 * (0.3396572 - Q.lep_z) * (0.1956014 - Q.pz_lnd2)
    if Q.lep_z < 0.3396572 and Q.mass_top20 > 86.78877:
        z += 0.0155884 * (0.3396572 - Q.lep_z) * (Q.mass_top20 - 86.78877)
    if Q.mass_displaced5 > 9.380468 and Q.mass_charged < 113.5019:
        z += -0.0001651966 * (Q.mass_displaced5 - 9.380468) * (113.5019 - Q.mass_charged)
    if Q.lep_ptrel < 18.7678 and Q.ktd_ln_d34 < -7.886954:
        z += 0.008024788 * (18.7678 - Q.lep_ptrel) * (-7.886954 - Q.ktd_ln_d34)
    if Q.n_s3d_above_3 > 4.0 and Q.sip_3d_2 < 226.3008:
        z += 0.0003149983 * (Q.n_s3d_above_3 - 4.0) * (226.3008 - Q.sip_3d_2)
    if Q.lep_ptrel < 18.7678 and Q.lepsj_2_maxsd0 < 131.3722:
        z += -2.462651e-05 * (18.7678 - Q.lep_ptrel) * (131.3722 - Q.lepsj_2_maxsd0)
    if Q.n_s3d_above_3 > 4.0 and Q.z_top50_slots > 0.9572293:
        z += 0.8324227 * (Q.n_s3d_above_3 - 4.0) * (Q.z_top50_slots - 0.9572293)
    if Q.lep_ptrel < 18.7678 and Q.sj4_dr_min < 0.1631992:
        z += 0.05439221 * (18.7678 - Q.lep_ptrel) * (0.1631992 - Q.sj4_dr_min)
    if Q.lepsj_3_n_d3 < 2.0 and Q.jet_e > 835.8719:
        z += -0.0001269044 * (2.0 - Q.lepsj_3_n_d3) * (Q.jet_e - 835.8719)
    if Q.lep_ptrel < 18.7678 and Q.ak02_2_z > 0.08487383:
        z += -0.01880944 * (18.7678 - Q.lep_ptrel) * (Q.ak02_2_z - 0.08487383)
    if Q.lep_z < 0.3396572 and Q.ecf_g42 < 8.17233e-05:
        z += 9183.45 * (0.3396572 - Q.lep_z) * (8.17233e-05 - Q.ecf_g42)
    if Q.n_s3d_above_3 > 4.0 and Q.tau32 < 0.863865:
        z += -0.06884053 * (Q.n_s3d_above_3 - 4.0) * (0.863865 - Q.tau32)
    if Q.lep_ptrel < 18.7678 and Q.sum_pt_top30 < 749.0582:
        z += 4.230007e-05 * (18.7678 - Q.lep_ptrel) * (749.0582 - Q.sum_pt_top30)
    if Q.z_top15_slots > 0.8008865 and Q.z_displaced5 > 0.2184442:
        z += -3.440989 * (Q.z_top15_slots - 0.8008865) * (Q.z_displaced5 - 0.2184442)
    if Q.lep_z < 0.3396572 and Q.ak02_n > 1.0:
        z += 0.08614386 * (0.3396572 - Q.lep_z) * (Q.ak02_n - 1.0)
    if Q.lep_ptrel < 18.7678 and Q.mass_2photon < 22.18431:
        z += -0.0006085781 * (18.7678 - Q.lep_ptrel) * (22.18431 - Q.mass_2photon)
    if Q.lep_ptrel < 18.7678 and Q.sjf_4_2_z_d3 < 0.1570831:
        z += 0.05019734 * (18.7678 - Q.lep_ptrel) * (0.1570831 - Q.sjf_4_2_z_d3)
    if Q.lepsj_3_n_d3 < 2.0 and Q.lepsj_3_dr < 0.09790963:
        z += 0.8193669 * (2.0 - Q.lepsj_3_n_d3) * (0.09790963 - Q.lepsj_3_dr)
    if Q.sdb_2_n < 9.0 and Q.sjq_2_prod_k03 > -0.9998116:
        z += -0.03330797 * (9.0 - Q.sdb_2_n) * (Q.sjq_2_prod_k03 - -0.9998116)
    if Q.lep_z < 0.3396572 and Q.lepsj_2_dr < 0.103005:
        z += 27.38653 * (0.3396572 - Q.lep_z) * (0.103005 - Q.lepsj_2_dr)
    if Q.lepsj_2_dr < 0.103005 and Q.jet_charge_k03 < 0.1164034:
        z += 0.8271982 * (0.103005 - Q.lepsj_2_dr) * (0.1164034 - Q.jet_charge_k03)
    if Q.sj4_pair_mass_max < 115.5142 and Q.n_neutral_had < 9.0:
        z += -0.0003758619 * (115.5142 - Q.sj4_pair_mass_max) * (9.0 - Q.n_neutral_had)
    return z


def neuron_79(Q):
    z = -3.602388e-05
    return z


def neuron_80(Q):
    z = 9.707706e-06
    return z


def neuron_81(Q):
    z = 0.1455253
    return z


def neuron_82(Q):
    z = -2.942075e-06
    return z


def neuron_83(Q):
    z = 1.187956
    if 78.33213 <= Q.mass_top40 < 155.8928:
        z += 0.007457161 * Q.mass_top40 - 0.5841353
    if Q.mass_top40 >= 155.8928:
        z += 0.03083473 * Q.mass_top40 - 4.228529
    if Q.n_s3d_above_3 < 3.0:
        z += -0.06445462 * Q.n_s3d_above_3 + 0.1933639
    if Q.n_s3d_above_3 >= 6.0:
        z += 0.04869723 * Q.n_s3d_above_3 - 0.2921834
    if Q.z_displaced3 < 0.06416437:
        z += -5.164023 * Q.z_displaced3 + 0.4051886
    if 0.06416437 <= Q.z_displaced3 < 0.1681173:
        z += -0.7103438 * Q.z_displaced3 + 0.1194211
    if Q.mass_top30 < 162.7874:
        z += 0.0008974005 * Q.mass_top30 - 0.1460855
    if Q.n_s3d_above_10 < 6.0:
        z += 0.07810839 * Q.n_s3d_above_10 - 0.4686503
    if Q.mres_sd_mass_b2z01 < 81.19466:
        z += 0.01194326 * Q.mres_sd_mass_b2z01 - 1.663176
    if 81.19466 <= Q.mres_sd_mass_b2z01 < 130.0968:
        z += -0.005026376 * Q.mres_sd_mass_b2z01 - 0.2853324
    if 130.0968 <= Q.mres_sd_mass_b2z01 < 139.2565:
        z += 0.02339973 * Q.mres_sd_mass_b2z01 - 3.983479
    if 139.2565 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.01145647 * Q.mres_sd_mass_b2z01 - 2.320303
    if Q.mres_sd_mass_b2z01 >= 158.2019:
        z += 0.006276953 * Q.mres_sd_mass_b2z01 - 1.500894
    if Q.N2 < 0.258375:
        z += 1.588317 * Q.N2 - 0.4103813
    if Q.M3_b2 < 0.02160244:
        z += 20.66751 * Q.M3_b2 - 0.4464686
    if Q.sdb_2_z < 0.1530389:
        z += -2.062194 * Q.sdb_2_z - 0.0217741
    if 0.1530389 <= Q.sdb_2_z < 0.5109872:
        z += 0.9425105 * Q.sdb_2_z - 0.4816108
    if Q.mass_top15 >= 77.22442:
        z += -0.006674744 * Q.mass_top15 + 0.5154533
    if Q.pair_max_lnm2 >= 7.524613:
        z += 0.2734338 * Q.pair_max_lnm2 - 2.057484
    if Q.sj4_pair_mass_max < 75.78208:
        z += -0.003949583 * Q.sj4_pair_mass_max + 0.8726351
    if 75.78208 <= Q.sj4_pair_mass_max < 115.5142:
        z += -0.01442982 * Q.sj4_pair_mass_max + 1.666849
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += 0.07503691 * Q.sjf_2_1_n_d3 - 0.3751846
    if Q.mass_displaced5 < 1.262841:
        z += 0.07205032 * Q.mass_displaced5 - 0.09098808
    if Q.max_abs_d0 < 10.52344:
        z += -0.02578869 * Q.max_abs_d0 + 0.2713857
    if Q.jd_sum_abs_sd0_top5 < 185.1889:
        z += 0.0008130237 * Q.jd_sum_abs_sd0_top5 - 0.150563
    if Q.sip_3d_1 < 7.028704:
        z += -0.08564878 * Q.sip_3d_1 + 0.6019999
    if Q.tau1 < 0.06074238:
        z += 15.10393 * Q.tau1 - 0.9174486
    if Q.sjf_4_n2disp >= 2.0:
        z += -0.150263 * Q.sjf_4_n2disp + 0.3005259
    if 79.47361 <= Q.mass < 95.14961:
        z += -0.02681892 * Q.mass + 2.131396
    if 95.14961 <= Q.mass < 117.4867:
        z += -0.03381754 * Q.mass + 2.797312
    if Q.mass >= 117.4867:
        z += -0.01482745 * Q.mass + 0.5662297
    if Q.sdb_2_n < 10.0:
        z += -0.01288611 * Q.sdb_2_n + 0.1288611
    if Q.dr_max_012 < 0.02210827:
        z += 11.60898 * Q.dr_max_012 - 0.2566545
    if Q.e4 < 3.53193e-06:
        z += 83517.16 * Q.e4 - 0.2949767
    if Q.n_pt_above_5 < 24.0:
        z += 0.02718687 * Q.n_pt_above_5 - 0.6524848
    if Q.tau5 >= 0.01467699:
        z += 12.52906 * Q.tau5 - 0.1838888
    if Q.sum_z_dr2_top50 < 0.0925671:
        z += -10.73804 * Q.sum_z_dr2_top50 + 0.9939888
    if Q.D3_b05 < 1.139376:
        z += 0.4346104 * Q.D3_b05 - 0.4951847
    if Q.lepsj_2_dr < 0.06573337:
        z += -2.789168 * Q.lepsj_2_dr + 0.1833414
    if Q.mass_top10 >= 103.4976:
        z += -0.006112264 * Q.mass_top10 + 0.6326049
    if Q.n_s3d_above_3 > 6.0 and Q.jd_sum_abs_sd0_top5 < 1292.625:
        z += 6.431134e-05 * (Q.n_s3d_above_3 - 6.0) * (1292.625 - Q.jd_sum_abs_sd0_top5)
    if Q.mass_top40 > 78.33213 and Q.lep_z < 0.1275041:
        z += -0.05750573 * (Q.mass_top40 - 78.33213) * (0.1275041 - Q.lep_z)
    if Q.mass_top30 < 162.7874 and Q.mass_displaced5 > 9.380468:
        z += -0.0001471728 * (162.7874 - Q.mass_top30) * (Q.mass_displaced5 - 9.380468)
    if Q.n_s3d_above_10 < 6.0 and Q.sdb_2_z > 0.2277628:
        z += -0.1032428 * (6.0 - Q.n_s3d_above_10) * (Q.sdb_2_z - 0.2277628)
    if Q.n_s3d_above_3 > 6.0 and Q.sj3_pairmin_over_m < 0.4866692:
        z += 0.3615918 * (Q.n_s3d_above_3 - 6.0) * (0.4866692 - Q.sj3_pairmin_over_m)
    if Q.mres_sd_mass_b2z01 < 139.2565 and Q.dc_2_n_lep < 1.0:
        z += 0.002506905 * (139.2565 - Q.mres_sd_mass_b2z01) * (1.0 - Q.dc_2_n_lep)
    if Q.sdb_2_z < 0.1530389 and Q.jd_3d_5 < 4.004982:
        z += -1.859987 * (0.1530389 - Q.sdb_2_z) * (4.004982 - Q.jd_3d_5)
    if Q.n_s3d_above_10 < 6.0 and Q.lne_0 > 4.380463:
        z += 0.02908164 * (6.0 - Q.n_s3d_above_10) * (Q.lne_0 - 4.380463)
    if Q.max_abs_d0 < 10.52344 and Q.jd_3d_5 < 6.771002:
        z += -0.008547793 * (10.52344 - Q.max_abs_d0) * (6.771002 - Q.jd_3d_5)
    if Q.mres_sd_mass_b2z01 < 139.2565 and Q.lund1_lnz > -2.478371:
        z += 0.002960857 * (139.2565 - Q.mres_sd_mass_b2z01) * (Q.lund1_lnz - -2.478371)
    if Q.mass_displaced3 < 13.03663 and Q.jd_3d_5 < 6.771002:
        z += 0.005134675 * (13.03663 - Q.mass_displaced3) * (6.771002 - Q.jd_3d_5)
    if Q.sj4_pair_mass_max < 115.5142 and Q.z_neutral_had < 0.3093201:
        z += -0.007223813 * (115.5142 - Q.sj4_pair_mass_max) * (0.3093201 - Q.z_neutral_had)
    if Q.mres_sd_mass_b2z01 > 130.0968 and Q.jd_3d_5 < 10.7744:
        z += 0.0003423932 * (Q.mres_sd_mass_b2z01 - 130.0968) * (10.7744 - Q.jd_3d_5)
    if Q.M3_b2 < 0.02160244 and Q.sjq_3_3_nch < 6.0:
        z += 1.164105 * (0.02160244 - Q.M3_b2) * (6.0 - Q.sjq_3_3_nch)
    if Q.n_s3d_above_3 > 6.0 and Q.jd_3d_4 < 172.888:
        z += 0.000742925 * (Q.n_s3d_above_3 - 6.0) * (172.888 - Q.jd_3d_4)
    if Q.jd_sum_abs_sd0_top5 < 185.1889 and Q.sjf_4_4_z_d3 < 0.003370318:
        z += -0.1822818 * (185.1889 - Q.jd_sum_abs_sd0_top5) * (0.003370318 - Q.sjf_4_4_z_d3)
    if Q.n_s3d_above_3 > 6.0 and Q.lne_3 < 4.821289:
        z += -0.0333854 * (Q.n_s3d_above_3 - 6.0) * (4.821289 - Q.lne_3)
    if Q.z_displaced3 < 0.1681173 and Q.lepsj_3_mass < 12.91148:
        z += 0.04774948 * (0.1681173 - Q.z_displaced3) * (12.91148 - Q.lepsj_3_mass)
    if Q.sj4_pair_mass_max < 75.78208 and Q.sjf_4_3_z_d3 < 0.07303924:
        z += -0.1533058 * (75.78208 - Q.sj4_pair_mass_max) * (0.07303924 - Q.sjf_4_3_z_d3)
    if Q.sv_2_z < 0.03767806 and Q.max_dr > 0.8033751:
        z += 7.136178 * (0.03767806 - Q.sv_2_z) * (Q.max_dr - 0.8033751)
    if Q.sj4_pair_mass_max < 115.5142 and Q.n_dr_0p2_0p4 > 9.0:
        z += -0.000170207 * (115.5142 - Q.sj4_pair_mass_max) * (Q.n_dr_0p2_0p4 - 9.0)
    return z


def neuron_84(Q):
    z = -7.427513e-05
    return z


def neuron_85(Q):
    z = -8.185523e-07
    return z


def neuron_86(Q):
    z = -1.709903e-05
    return z


def neuron_87(Q):
    z = 1.619163e-06
    return z


def neuron_88(Q):
    z = 2.996983e-07
    return z


def neuron_89(Q):
    z = 1.735375e-06
    return z


def neuron_90(Q):
    z = 0.001285483
    return z


def neuron_91(Q):
    z = 3.992639e-06
    return z


def neuron_92(Q):
    z = -6.724739e-06
    return z


def neuron_93(Q):
    z = 8.464389e-07
    return z


def neuron_94(Q):
    z = -2.322172e-06
    return z


def neuron_95(Q):
    z = 4.369519e-07
    return z


def neuron_96(Q):
    z = 4.278036e-07
    return z


def neuron_97(Q):
    z = -1.844001
    if Q.mass_displaced3 < 18.80005:
        z += -0.04409485 * Q.mass_displaced3 + 1.244435
    if 18.80005 <= Q.mass_displaced3 < 39.09615:
        z += -0.02046943 * Q.mass_displaced3 + 0.800276
    if Q.mass < 95.14961:
        z += 0.00961646 * Q.mass - 2.675335
    if 95.14961 <= Q.mass < 100.4835:
        z += 0.02413968 * Q.mass - 4.057215
    if 100.4835 <= Q.mass < 149.0507:
        z += 0.04135702 * Q.mass - 5.787272
    if 149.0507 <= Q.mass < 164.4374:
        z += 0.002626488 * Q.mass - 0.0144602
    if 164.4374 <= Q.mass < 182.8592:
        z += -0.02265974 * Q.mass + 4.143543
    if Q.e3_b2 < 9.621843e-05:
        z += 3431.762 * Q.e3_b2 - 0.1299299
    if 9.621843e-05 <= Q.e3_b2 < 0.0002536827:
        z += 980.9491 * Q.e3_b2 + 0.1058834
    if 0.0002536827 <= Q.e3_b2 < 0.0007909605:
        z += -660.2417 * Q.e3_b2 + 0.5222251
    if Q.tau32 < 0.6513932:
        z += -0.3511602 * Q.tau32 + 0.2287433
    if Q.sj3_pair_mass_max < 63.7508:
        z += -0.03901508 * Q.sj3_pair_mass_max + 4.065861
    if 63.7508 <= Q.sj3_pair_mass_max < 114.7848:
        z += -0.03093266 * Q.sj3_pair_mass_max + 3.5506
    if Q.mass_top50 < 71.80762:
        z += 0.02610941 * Q.mass_top50 - 2.318093
    if 71.80762 <= Q.mass_top50 < 122.0585:
        z += 0.0088205 * Q.mass_top50 - 1.076617
    if Q.sip_3d_2 < 447.0873:
        z += 0.0007692368 * Q.sip_3d_2 - 0.343916
    if Q.jd_3d_4 < 172.888:
        z += 0.001461674 * Q.jd_3d_4 - 0.2527059
    if Q.n_pairs_kt_above_3 < 111.0:
        z += -0.00274173 * Q.n_pairs_kt_above_3 + 0.304332
    if Q.ak02_min12_jp < 7.746064:
        z += -0.01032919 * Q.ak02_min12_jp + 0.08001056
    z += 2.806752 * Q.sj3_pairmax_over_m
    if Q.mass_top40 < 126.8853:
        z += -0.01250785 * Q.mass_top40 + 1.587062
    if Q.N2_b05 >= 0.4353632:
        z += -1.297631 * Q.N2_b05 + 0.5649408
    if Q.max_abs_d0 < 5.8125:
        z += 0.107621 * Q.max_abs_d0 - 0.6255468
    if Q.lep_ptrel >= 3.53503:
        z += -0.003163823 * Q.lep_ptrel + 0.01118421
    if Q.sdb_2_z < 0.2740506:
        z += 1.027312 * Q.sdb_2_z - 0.2815354
    if Q.pair_max_lnm2 < 7.075642:
        z += -0.0466537 * Q.pair_max_lnm2 + 0.3301048
    if Q.n_sd0_above_3 < 6.0:
        z += 0.145384 * Q.n_sd0_above_3 - 0.8723041
    if Q.sjf_3_1_n_d5 < 5.0:
        z += -0.02065456 * Q.sjf_3_1_n_d5 + 0.1032728
    if Q.sj2_mass2 < 1.851735:
        z += -0.1746233 * Q.sj2_mass2 + 0.3233562
    if Q.mres_pruned_mass < 70.04065:
        z += 0.003263853 * Q.mres_pruned_mass - 1.261318
    if 70.04065 <= Q.mres_pruned_mass < 112.1947:
        z += -0.008685623 * Q.mres_pruned_mass - 0.4243686
    if 112.1947 <= Q.mres_pruned_mass < 124.3145:
        z += -0.0266478 * Q.mres_pruned_mass + 1.590892
    if 124.3145 <= Q.mres_pruned_mass < 171.3819:
        z += 0.01085341 * Q.mres_pruned_mass - 3.071054
    if Q.mres_pruned_mass >= 171.3819:
        z += -0.01194948 * Q.mres_pruned_mass + 0.8369491
    if Q.sd_mass >= 119.4443:
        z += 0.01481675 * Q.sd_mass - 1.769777
    if 53.57509 <= Q.mres_sd_mass_b0z005 < 76.40585:
        z += 0.02113864 * Q.mres_sd_mass_b0z005 - 1.132504
    if 76.40585 <= Q.mres_sd_mass_b0z005 < 91.82593:
        z += -0.06487595 * Q.mres_sd_mass_b0z005 + 5.439513
    if 91.82593 <= Q.mres_sd_mass_b0z005 < 97.29337:
        z += 0.01950975 * Q.mres_sd_mass_b0z005 - 2.309281
    if 97.29337 <= Q.mres_sd_mass_b0z005 < 122.1:
        z += 4.311837e-05 * Q.mres_sd_mass_b0z005 - 0.4153074
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.02096958 * Q.mres_sd_mass_b0z005 - 2.970428
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.003820905 * Q.mres_sd_mass_b0z005 + 0.9941712
    if Q.n_pairs_kt_above_1 < 325.0:
        z += -0.00138952 * Q.n_pairs_kt_above_1 + 0.451594
    if Q.lund3_lndelta >= -1.780944:
        z += 0.08701389 * Q.lund3_lndelta + 0.1549669
    if Q.n_s3d_above_3 < 9.0:
        z += -0.07499652 * Q.n_s3d_above_3 + 0.6749687
    if Q.mass_charged >= 62.00562:
        z += 0.001289895 * Q.mass_charged - 0.07998073
    if Q.z_displaced5 >= 0.1339658:
        z += -0.3016943 * Q.z_displaced5 + 0.04041671
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += -0.03897141 * Q.sjf_2_1_n_d3 + 0.194857
    if Q.mres_sd_prong_mass1 >= 69.04524:
        z += -0.006089324 * Q.mres_sd_prong_mass1 + 0.4204388
    if Q.lund_max_lnkt < 3.734077:
        z += 0.1332859 * Q.lund_max_lnkt - 0.4976999
    if Q.lep_z < 0.221436:
        z += 1.014191 * Q.lep_z - 0.2245783
    if Q.n_photon < 8.0:
        z += -0.02906345 * Q.n_photon + 0.2325076
    if Q.lnerel_42 >= -6.337363:
        z += -0.09056531 * Q.lnerel_42 - 0.5739453
    if 72.72359 <= Q.nca_sj4_pair_mass_2nd < 83.41384:
        z += -0.02671077 * Q.nca_sj4_pair_mass_2nd + 1.942503
    if Q.nca_sj4_pair_mass_2nd >= 83.41384:
        z += 0.004097182 * Q.nca_sj4_pair_mass_2nd - 0.6273066
    if Q.e3_b2 < 0.0002536827 and Q.sdb_2_z < 0.3709098:
        z += 991.0005 * (0.0002536827 - Q.e3_b2) * (0.3709098 - Q.sdb_2_z)
    if Q.mass_displaced3 < 39.09615 and Q.sum_pt_top10 > 324.6547:
        z += -5.90446e-06 * (39.09615 - Q.mass_displaced3) * (Q.sum_pt_top10 - 324.6547)
    if Q.mass < 100.4835 and Q.lep_ptrel < 1.477152:
        z += 0.005693539 * (100.4835 - Q.mass) * (1.477152 - Q.lep_ptrel)
    if Q.mass_displaced3 < 39.09615 and Q.lep_ptrel < 18.7678:
        z += -0.000466582 * (39.09615 - Q.mass_displaced3) * (18.7678 - Q.lep_ptrel)
    if Q.mass < 182.8592 and Q.dc_ntag > 0.0:
        z += -0.0009814309 * (182.8592 - Q.mass) * (Q.dc_ntag - 0.0)
    if Q.mass < 182.8592 and Q.n_s3d_above_3 > 1.0:
        z += -0.0007923908 * (182.8592 - Q.mass) * (Q.n_s3d_above_3 - 1.0)
    if Q.e3_b2 < 0.0002536827 and Q.jet_charge_k03 > -0.05990128:
        z += 1375.942 * (0.0002536827 - Q.e3_b2) * (Q.jet_charge_k03 - -0.05990128)
    if Q.mass_top50 < 122.0585 and Q.ak02_2_n_lep < 1.0:
        z += 0.007790683 * (122.0585 - Q.mass_top50) * (1.0 - Q.ak02_2_n_lep)
    if Q.mass_displaced3 < 39.09615 and Q.pz_lnd3 < 0.18807:
        z += 0.06939019 * (39.09615 - Q.mass_displaced3) * (0.18807 - Q.pz_lnd3)
    if Q.mass_top50 < 122.0585 and Q.sum_e > 926.2598:
        z += 9.80614e-06 * (122.0585 - Q.mass_top50) * (Q.sum_e - 926.2598)
    if Q.sip_3d_2 < 447.0873 and Q.dc_1_n_lep < 1.0:
        z += 0.0004034015 * (447.0873 - Q.sip_3d_2) * (1.0 - Q.dc_1_n_lep)
    if Q.sj3_pair_mass_max < 114.7848 and Q.n_photon > 12.0:
        z += -0.0002733941 * (114.7848 - Q.sj3_pair_mass_max) * (Q.n_photon - 12.0)
    if Q.mass_top50 < 122.0585 and Q.sjq_2_prod_k05 < 0.125305:
        z += -0.0151189 * (122.0585 - Q.mass_top50) * (0.125305 - Q.sjq_2_prod_k05)
    if Q.max_abs_d0 < 5.8125 and Q.jd_3d_5 < 4.004982:
        z += 0.05142425 * (5.8125 - Q.max_abs_d0) * (4.004982 - Q.jd_3d_5)
    if Q.tau32 < 0.6513932 and Q.jd_3d_6 < 3.151933:
        z += -0.264106 * (0.6513932 - Q.tau32) * (3.151933 - Q.jd_3d_6)
    if Q.mass < 164.4374 and Q.jd_3d_6 > 16.11752:
        z += 8.631555e-06 * (164.4374 - Q.mass) * (Q.jd_3d_6 - 16.11752)
    if Q.pair_mean_lnm2 > 2.09826 and Q.dc_3_z < 0.171574:
        z += -0.2067068 * (Q.pair_mean_lnm2 - 2.09826) * (0.171574 - Q.dc_3_z)
    if Q.sdb_2_z < 0.2740506 and Q.pz_lnkt1 < 0.1348361:
        z += 13.79868 * (0.2740506 - Q.sdb_2_z) * (0.1348361 - Q.pz_lnkt1)
    if Q.mass_top40 < 126.8853 and Q.n_muon > 1.0:
        z += -0.01192681 * (126.8853 - Q.mass_top40) * (Q.n_muon - 1.0)
    if Q.mass_displaced3 < 39.09615 and Q.D2 > 3.038341:
        z += 0.0009585745 * (39.09615 - Q.mass_displaced3) * (Q.D2 - 3.038341)
    if Q.mres_pruned_mass > 70.04065 and Q.lnerel_4 > -3.643775:
        z += -0.001541146 * (Q.mres_pruned_mass - 70.04065) * (Q.lnerel_4 - -3.643775)
    if Q.sj2_mass2 < 1.851735 and Q.mass_2photon > 1.504865:
        z += 0.01283004 * (1.851735 - Q.sj2_mass2) * (Q.mass_2photon - 1.504865)
    if Q.lund3_lndelta > -1.780944 and Q.z_photon > 0.2047275:
        z += 0.5657514 * (Q.lund3_lndelta - -1.780944) * (Q.z_photon - 0.2047275)
    if Q.mass < 100.4835 and Q.lep_iso < 6.185635:
        z += -0.00110705 * (100.4835 - Q.mass) * (6.185635 - Q.lep_iso)
    if Q.lep_z < 0.221436 and Q.lepsj_2_n_d3 < 4.0:
        z += 0.06368379 * (0.221436 - Q.lep_z) * (4.0 - Q.lepsj_2_n_d3)
    if Q.nca_sj4_pair_mass_2nd < 65.88119 and Q.nca_kt_above_10 > 1.0:
        z += -0.002754879 * (65.88119 - Q.nca_sj4_pair_mass_2nd) * (Q.nca_kt_above_10 - 1.0)
    if Q.ak02_min12_jp < 7.746064 and Q.mass_2photon > 0.2753928:
        z += 0.0006206068 * (7.746064 - Q.ak02_min12_jp) * (Q.mass_2photon - 0.2753928)
    return z


def neuron_98(Q):
    z = 4.990173e-07
    return z


def neuron_99(Q):
    z = -0.001934242
    return z


def neuron_100(Q):
    z = -6.264084e-06
    return z


def neuron_101(Q):
    z = 8.878268e-06
    return z


def neuron_102(Q):
    z = 5.490517e-06
    return z


def neuron_103(Q):
    z = -3.414383e-06
    return z


def neuron_104(Q):
    z = -1.637735
    if Q.mass < 117.4867:
        z += -0.0263755 * Q.mass + 3.173634
    if 117.4867 <= Q.mass < 149.0507:
        z += -0.002371796 * Q.mass + 0.3535178
    if Q.pair_mean_lndelta < -1.352792:
        z += -0.4693735 * Q.pair_mean_lndelta - 0.6349645
    if Q.sjq_2_1_nch < 18.0:
        z += -0.02158678 * Q.sjq_2_1_nch + 0.3885621
    if Q.lep_ptrel >= 6.983043:
        z += 0.005240457 * Q.lep_ptrel - 0.03659434
    if 3.208089 <= Q.mass_displaced3 < 6.341631:
        z += 0.05342958 * Q.mass_displaced3 - 0.1714068
    if Q.mass_displaced3 >= 6.341631:
        z += 0.01494392 * Q.mass_displaced3 + 0.07265505
    if Q.n_pt_above_1 < 58.0:
        z += -0.02150941 * Q.n_pt_above_1 + 1.247546
    if Q.sj4_pair_mass_max >= 128.0079:
        z += 0.007406392 * Q.sj4_pair_mass_max - 0.9480763
    if Q.tau32_b2 < 0.4680886:
        z += 0.54054 * Q.tau32_b2 - 0.2530206
    if Q.D2 < 1.226724:
        z += -0.1495775 * Q.D2 + 0.1834903
    if Q.D2 >= 3.597891:
        z += 0.1280967 * Q.D2 - 0.4608781
    if Q.n_pairs_kt_above_3 < 34.0:
        z += 0.01886378 * Q.n_pairs_kt_above_3 - 0.6413686
    if Q.N2 < 0.312717:
        z += -0.3440786 * Q.N2 + 0.1075992
    if Q.nca_sj4_pair_mass_2nd >= 77.3635:
        z += 0.007131771 * Q.nca_sj4_pair_mass_2nd - 0.5517388
    if Q.pz_lnd0 < 0.2827395:
        z += -1.357891 * Q.pz_lnd0 + 0.3839294
    if Q.n_s3d_above_10 < 6.0:
        z += -0.07109578 * Q.n_s3d_above_10 + 0.4265747
    if Q.sjf_2_1_max3d < 3.32421:
        z += 0.09365521 * Q.sjf_2_1_max3d - 0.3113295
    if Q.sjf_3_2_max3d < 5.245219:
        z += 0.0552473 * Q.sjf_3_2_max3d - 0.2897842
    if Q.sjq_2_sumabs_k03 >= 0.6667228:
        z += -0.09757262 * Q.sjq_2_sumabs_k03 + 0.06505389
    if Q.tau3 < 0.05398263:
        z += 7.791493 * Q.tau3 - 0.4206053
    if Q.n_pairs_kt_above_1 < 146.0:
        z += 0.001557748 * Q.n_pairs_kt_above_1 - 0.2274312
    if Q.lund_max_lndelta >= -0.529318:
        z += 0.6498584 * Q.lund_max_lndelta + 0.3439817
    if Q.sip_3d_2 < 226.3008:
        z += 0.0003448481 * Q.sip_3d_2 - 0.07803939
    if Q.sjf_3_2_maxsd0 < 61.73838:
        z += -0.002207696 * Q.sjf_3_2_maxsd0 + 0.1362996
    if Q.sdb_2_z < 0.3988697:
        z += -0.1773416 * Q.sdb_2_z + 0.07073621
    if Q.ak02_min12_jp < 14.23948:
        z += 0.01277022 * Q.ak02_min12_jp - 0.1818412
    if Q.tau21_b2 >= 0.3898586:
        z += 0.8137724 * Q.tau21_b2 - 0.3172562
    if Q.sjq_3_2_nch < 1.0:
        z += -0.1957639 * Q.sjq_3_2_nch + 0.1957639
    if Q.pair_max_lnm2 >= 5.908788:
        z += 0.09854948 * Q.pair_max_lnm2 - 0.5823079
    if Q.z_charged_had < 0.4501484:
        z += 0.7775224 * Q.z_charged_had - 0.3500005
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 86.09683:
        z += 0.02040582 * Q.mres_sd_mass_b2z01 - 1.553962
    if Q.mres_sd_mass_b2z01 >= 86.09683:
        z += 0.001591206 * Q.mres_sd_mass_b2z01 + 0.06591613
    if Q.sj2_mass1 >= 53.51926:
        z += -0.002373052 * Q.sj2_mass1 + 0.127004
    if Q.mass_top5 >= 17.63354:
        z += -0.002682308 * Q.mass_top5 + 0.04729858
    if Q.tau5 < 0.0230226:
        z += -32.51358 * Q.tau5 + 0.7485472
    if Q.mass_top50 < 135.5368:
        z += -0.00462696 * Q.mass_top50 + 0.6271232
    if Q.sjf_3_1_n_d3 < 4.0:
        z += 0.01647107 * Q.sjf_3_1_n_d3 - 0.06588428
    if Q.sjf_2_2_max3d < 16.96512:
        z += -0.008951423 * Q.sjf_2_2_max3d + 0.1518619
    if Q.sjq_2_2_nch < 2.0:
        z += -0.1424447 * Q.sjq_2_2_nch + 0.4826655
    if 2.0 <= Q.sjq_2_2_nch < 15.0:
        z += -0.01521354 * Q.sjq_2_2_nch + 0.2282031
    if Q.mres_sd_mass_b0z005 >= 91.82593:
        z += -0.002785217 * Q.mres_sd_mass_b0z005 + 0.2557552
    if Q.mass_charged < 99.20396:
        z += -0.002710514 * Q.mass_charged + 0.2688937
    if Q.lepsj_2_maxsd0 < 1.546763:
        z += -0.08828513 * Q.lepsj_2_maxsd0 + 0.1365562
    if Q.sv_1_dr < 0.1040701:
        z += 0.5516726 * Q.sv_1_dr - 0.05741261
    if Q.dc_tag_2nd < 1.914834:
        z += 0.09920179 * Q.dc_tag_2nd - 0.1899549
    if Q.mass_top30 >= 126.5007:
        z += 0.003449601 * Q.mass_top30 - 0.436377
    if Q.z_top3_slots >= 0.5395924:
        z += 1.986107 * Q.z_top3_slots - 1.071688
    if Q.sdb_2_n < 8.0:
        z += -0.01592916 * Q.sdb_2_n + 0.1274333
    if Q.C2 < 0.1798521:
        z += -0.8951628 * Q.C2 + 0.1609969
    if Q.C2_b05 < 0.2234678:
        z += -2.794782 * Q.C2_b05 + 0.6245439
    if Q.n_pt_above_1 < 58.0 and Q.sjf_2_1_n_d5 > 1.0:
        z += -0.001752445 * (58.0 - Q.n_pt_above_1) * (Q.sjf_2_1_n_d5 - 1.0)
    if Q.mass_displaced3 > 3.208089 and Q.ak02_3_z < 0.1014193:
        z += 0.04943253 * (Q.mass_displaced3 - 3.208089) * (0.1014193 - Q.ak02_3_z)
    if Q.n_pt_above_1 < 58.0 and Q.z_neutral > 0.1384639:
        z += -0.01004711 * (58.0 - Q.n_pt_above_1) * (Q.z_neutral - 0.1384639)
    if Q.mass < 149.0507 and Q.sum_e < 1401.904:
        z += -6.276704e-06 * (149.0507 - Q.mass) * (1401.904 - Q.sum_e)
    if Q.n_pt_above_1 < 58.0 and Q.pair_max_lnkt > 1.555555:
        z += -0.00330988 * (58.0 - Q.n_pt_above_1) * (Q.pair_max_lnkt - 1.555555)
    if Q.lep_ptrel > 6.983043 and Q.ak02_dr23 < 0.2385164:
        z += -0.06643865 * (Q.lep_ptrel - 6.983043) * (0.2385164 - Q.ak02_dr23)
    if Q.n_pt_above_1 < 58.0 and Q.ak02_dr13 < 0.4490565:
        z += 0.02068115 * (58.0 - Q.n_pt_above_1) * (0.4490565 - Q.ak02_dr13)
    if Q.mass < 149.0507 and Q.tau21_b2 < 0.3265797:
        z += -0.005083815 * (149.0507 - Q.mass) * (0.3265797 - Q.tau21_b2)
    if Q.pz_lnd0 < 0.2827395 and Q.n_pairs_kt_above_10 > 1.0:
        z += -0.03698367 * (0.2827395 - Q.pz_lnd0) * (Q.n_pairs_kt_above_10 - 1.0)
    if Q.mass < 149.0507 and Q.lnerel_65 > -18.42068:
        z += -0.0004644384 * (149.0507 - Q.mass) * (Q.lnerel_65 - -18.42068)
    if Q.lep_ptrel > 6.983043 and Q.mass_2charged < 28.89659:
        z += 0.0003976931 * (Q.lep_ptrel - 6.983043) * (28.89659 - Q.mass_2charged)
    if Q.sjf_2_1_z_d3 > 0.1149585 and Q.sjq_2_prod_k1 < 0.09802954:
        z += -1.984972 * (Q.sjf_2_1_z_d3 - 0.1149585) * (0.09802954 - Q.sjq_2_prod_k1)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.iselectron_1 > 0.0:
        z += -0.009011353 * (34.0 - Q.n_pairs_kt_above_3) * (Q.iselectron_1 - 0.0)
    if Q.sjf_3_2_max3d < 5.245219 and Q.jet_charge > 0.09659934:
        z += 0.07383421 * (5.245219 - Q.sjf_3_2_max3d) * (Q.jet_charge - 0.09659934)
    if Q.sjq_2_1_nch < 18.0 and Q.dr02 > 0.04115773:
        z += -0.02260476 * (18.0 - Q.sjq_2_1_nch) * (Q.dr02 - 0.04115773)
    if Q.sjf_2_1_z_d3 > 0.1149585 and Q.ischhad_0 > 0.0:
        z += 0.3029359 * (Q.sjf_2_1_z_d3 - 0.1149585) * (Q.ischhad_0 - 0.0)
    if Q.N2 < 0.312717 and Q.e3_b2 < 4.662928e-05:
        z += -34169.09 * (0.312717 - Q.N2) * (4.662928e-05 - Q.e3_b2)
    if Q.sjq_2_1_nch < 18.0 and Q.dr01 < 0.3258728:
        z += 0.02025496 * (18.0 - Q.sjq_2_1_nch) * (0.3258728 - Q.dr01)
    if Q.mass_top5 > 17.63354 and Q.lnpt_78 > -18.42068:
        z += -0.0004417496 * (Q.mass_top5 - 17.63354) * (Q.lnpt_78 - -18.42068)
    if Q.sjf_2_2_max3d < 4.516968 and Q.e3_b2 < 0.0007909605:
        z += -118.9923 * (4.516968 - Q.sjf_2_2_max3d) * (0.0007909605 - Q.e3_b2)
    if Q.mres_sd_mass_b0z005 > 91.82593 and Q.sv_1_n < 3.0:
        z += 0.0008614113 * (Q.mres_sd_mass_b0z005 - 91.82593) * (3.0 - Q.sv_1_n)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.kt2_2_n_lep < 1.0:
        z += 0.009903869 * (34.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.kt2_2_n_lep)
    if Q.n_pt_above_1 < 58.0 and Q.lepsj_2_n_d3 < 1.0:
        z += 0.004005442 * (58.0 - Q.n_pt_above_1) * (1.0 - Q.lepsj_2_n_d3)
    if Q.lepsj_2_maxsd0 < 1.546763 and Q.lepsj_2_dr < 0.0008210142:
        z += -127.2339 * (1.546763 - Q.lepsj_2_maxsd0) * (0.0008210142 - Q.lepsj_2_dr)
    if Q.mass_charged < 99.20396 and Q.ak02_2_n_lep < 1.0:
        z += -0.001945868 * (99.20396 - Q.mass_charged) * (1.0 - Q.ak02_2_n_lep)
    if Q.ak02_min12_jp < 14.23948 and Q.pz_lnkt1 > 0.06778673:
        z += 0.07418912 * (14.23948 - Q.ak02_min12_jp) * (Q.pz_lnkt1 - 0.06778673)
    return z


def neuron_105(Q):
    z = 2.732383e-05
    return z


def neuron_106(Q):
    z = 2.160936e-05
    return z


def neuron_107(Q):
    z = 1.15856e-05
    return z


def neuron_108(Q):
    z = -4.470702e-06
    return z


def neuron_109(Q):
    z = -5.200161e-07
    return z


def neuron_110(Q):
    z = 3.579609e-06
    return z


def neuron_111(Q):
    z = 3.669616e-06
    return z


def neuron_112(Q):
    z = 5.237824e-06
    return z


def neuron_113(Q):
    z = -2.665928e-05
    return z


def neuron_114(Q):
    z = 4.677418e-06
    return z


def neuron_115(Q):
    z = 1.385904
    if Q.pz_lnd0 < 0.06871203:
        z += -1.507922 * Q.pz_lnd0 + 0.02868039
    if 0.06871203 <= Q.pz_lnd0 < 0.1713451:
        z += 0.7300953 * Q.pz_lnd0 - 0.1250983
    if Q.N2_b2 < 0.2243273:
        z += -2.26289 * Q.N2_b2 + 0.507628
    if Q.mass_displaced3 < 39.09615:
        z += 0.02684339 * Q.mass_displaced3 - 1.049473
    if Q.tau2 < 0.09733903:
        z += -8.395576 * Q.tau2 + 0.8172173
    if Q.tau43 < 0.8939856:
        z += 1.250255 * Q.tau43 - 1.11771
    if Q.z_neutral_had < 0.3788785:
        z += 2.906121 * Q.z_neutral_had
    if Q.z_neutral_had >= 0.3788785:
        z += 0.3414128 * Q.z_neutral_had + 0.9717128
    if Q.pair_mean_lnkt < 0.9367772:
        z += -0.8414974 * Q.pair_mean_lnkt + 0.7882956
    if Q.ktd_ln_d34 >= -9.531553:
        z += -0.08784699 * Q.ktd_ln_d34 - 0.8373183
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.01373687 * Q.n_pairs_kt_above_1 - 1.043395
    if 80.0 <= Q.n_pairs_kt_above_1 < 411.0:
        z += -0.0001678393 * Q.n_pairs_kt_above_1 + 0.06898195
    if Q.lepsj_3_n_d3 < 3.0:
        z += 0.0468771 * Q.lepsj_3_n_d3 - 0.1406313
    if Q.e3_b2 < 0.0004126585:
        z += 679.3729 * Q.e3_b2 - 0.3461448
    if 0.0004126585 <= Q.e3_b2 < 0.0007909605:
        z += 173.924 * Q.e3_b2 - 0.137567
    if Q.n_s3d_above_3 >= 4.0:
        z += 0.1312121 * Q.n_s3d_above_3 - 0.5248485
    if Q.sj4_pair_mass_max < 75.78208:
        z += -0.007721815 * Q.sj4_pair_mass_max + 0.4013285
    if 75.78208 <= Q.sj4_pair_mass_max < 101.849:
        z += 0.007052872 * Q.sj4_pair_mass_max - 0.718328
    if Q.sip_3d_2 < 447.0873:
        z += -0.0004945925 * Q.sip_3d_2 + 0.221126
    if Q.M2 < 0.120439:
        z += -11.60432 * Q.M2 + 1.397612
    if Q.lep_z < 0.5187302:
        z += -0.9124666 * Q.lep_z + 0.473324
    if Q.LHA < 0.5626523:
        z += 3.321041 * Q.LHA - 1.868592
    if Q.e3 < 0.00228569:
        z += -210.1242 * Q.e3 + 0.4802787
    if Q.n_sd0_above_3 >= 2.0:
        z += -0.08117867 * Q.n_sd0_above_3 + 0.1623573
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.06256478 * Q.lepsj_3_maxsd0 + 0.1509853
    if Q.sj4_pair_mass_min < 25.64898:
        z += -0.007054733 * Q.sj4_pair_mass_min + 0.1809467
    if Q.pt_entropy >= 3.22434:
        z += 0.7497766 * Q.pt_entropy - 2.417535
    if Q.jet_abs_eta >= 1.465946:
        z += 0.4051064 * Q.jet_abs_eta - 0.5938641
    if Q.max_abs_dz < 6.402344:
        z += -0.02069712 * Q.max_abs_dz + 0.1325101
    if Q.mass_2charged < 1.398635:
        z += 0.1799391 * Q.mass_2charged - 0.1385734
    if 1.398635 <= Q.mass_2charged < 6.767937:
        z += -0.05306417 * Q.mass_2charged + 0.1873131
    if 6.767937 <= Q.mass_2charged < 20.76537:
        z += 0.01227524 * Q.mass_2charged - 0.2548999
    if Q.sum_pt_top10 < 428.9828:
        z += -0.002920884 * Q.sum_pt_top10 + 1.253009
    if Q.tau4 >= 0.04996:
        z += -9.315407 * Q.tau4 + 0.4653978
    if Q.psi_0p3 >= 0.9756505:
        z += 8.738616 * Q.psi_0p3 - 8.525835
    if Q.sj3_pair_mass_max < 114.7848:
        z += 0.01466514 * Q.sj3_pair_mass_max - 1.683336
    if Q.lep_iso < 0.4381892:
        z += 0.2562965 * Q.lep_iso - 0.1123063
    if Q.n_lepton < 1.0:
        z += -0.4297433 * Q.n_lepton + 0.4297433
    if Q.lep_ptrel < 18.7678:
        z += 0.03956993 * Q.lep_ptrel - 0.7426406
    if Q.mres_sd_prong_mass2 < 7.065114:
        z += 0.03184806 * Q.mres_sd_prong_mass2 - 0.01289151
    if 7.065114 <= Q.mres_sd_prong_mass2 < 21.04127:
        z += -0.01517718 * Q.mres_sd_prong_mass2 + 0.3193471
    if Q.mass_top50 < 161.1264:
        z += 0.009728407 * Q.mass_top50 - 1.567503
    if Q.mass < 131.3917:
        z += -0.02353756 * Q.mass + 3.09264
    z += -1.336562 * Q.sj3_pairmax_over_m
    if Q.mass_top20 < 93.50967:
        z += -0.00605986 * Q.mass_top20 + 0.5666555
    if Q.sjf_3_3_maxsd0 < 0.3479096:
        z += -1.118275 * Q.sjf_3_3_maxsd0 + 0.3890585
    if Q.pair_max_lnkt >= 1.555555:
        z += -0.2146837 * Q.pair_max_lnkt + 0.3339522
    if Q.sum_z_dr2_top3 < 0.02745856:
        z += 6.334687 * Q.sum_z_dr2_top3 - 0.1739414
    if Q.sj3_mass3 < 5.460258:
        z += 0.04529282 * Q.sj3_mass3 - 0.2473105
    if Q.dc_1_n_lep < 1.0:
        z += -0.07397998 * Q.dc_1_n_lep + 0.07397998
    if Q.lund_max_lnkt >= 3.388322:
        z += 0.1055569 * Q.lund_max_lnkt - 0.3576607
    if Q.dc_1_z < 0.932165:
        z += 0.4110042 * Q.dc_1_z - 0.3831238
    if Q.mres_sd_mass_b1z01 >= 88.79082:
        z += -0.001817775 * Q.mres_sd_mass_b1z01 + 0.1614018
    if Q.M3_b2 < 0.01013989:
        z += -11.25035 * Q.M3_b2 + 0.1140773
    if Q.mass_displaced3 < 39.09615 and Q.lep_ptrel < 18.7678:
        z += 0.001024523 * (39.09615 - Q.mass_displaced3) * (18.7678 - Q.lep_ptrel)
    if Q.pair_mean_lnkt < 0.9367772 and Q.sum_pt_top10 < 428.9828:
        z += -0.002259014 * (0.9367772 - Q.pair_mean_lnkt) * (428.9828 - Q.sum_pt_top10)
    if Q.lep_z < 0.5187302 and Q.tau54 < 0.8786609:
        z += -2.769087 * (0.5187302 - Q.lep_z) * (0.8786609 - Q.tau54)
    if Q.lep_z < 0.5187302 and Q.sj4_dr_min < 0.1845735:
        z += 2.157859 * (0.5187302 - Q.lep_z) * (0.1845735 - Q.sj4_dr_min)
    if Q.sip_3d_2 < 447.0873 and Q.jd_3d_4 < 9.982976:
        z += -7.258506e-05 * (447.0873 - Q.sip_3d_2) * (9.982976 - Q.jd_3d_4)
    if Q.lep_z < 0.5187302 and Q.lne_1 > 4.635336:
        z += 0.2225582 * (0.5187302 - Q.lep_z) * (Q.lne_1 - 4.635336)
    if Q.tau2 < 0.09733903 and Q.kt2_1_n_disp3 > 0.0:
        z += -1.806758 * (0.09733903 - Q.tau2) * (Q.kt2_1_n_disp3 - 0.0)
    if Q.lep_z < 0.5187302 and Q.ak02_n > 2.0:
        z += -0.16018 * (0.5187302 - Q.lep_z) * (Q.ak02_n - 2.0)
    if Q.pz_lnd0 < 0.1713451 and Q.z_displaced5 > 0.1046203:
        z += 6.321993 * (0.1713451 - Q.pz_lnd0) * (Q.z_displaced5 - 0.1046203)
    if Q.lep_z < 0.5187302 and Q.dc_n > 1.0:
        z += -0.07850774 * (0.5187302 - Q.lep_z) * (Q.dc_n - 1.0)
    if Q.sj3_pair_mass_min > 59.49644 and Q.n_lund_kt_above_5 < 4.0:
        z += 0.009939437 * (Q.sj3_pair_mass_min - 59.49644) * (4.0 - Q.n_lund_kt_above_5)
    if Q.sj4_pair_mass_max < 101.849 and Q.jd_3d_6 < 9.356714:
        z += -0.0009047321 * (101.849 - Q.sj4_pair_mass_max) * (9.356714 - Q.jd_3d_6)
    if Q.mass_top50 < 118.8408 and Q.jd_3d_6 < 9.356714:
        z += 0.001142872 * (118.8408 - Q.mass_top50) * (9.356714 - Q.jd_3d_6)
    if Q.mass_top50 < 118.8408 and Q.lepsj_2_dr < 0.103005:
        z += 0.05881395 * (118.8408 - Q.mass_top50) * (0.103005 - Q.lepsj_2_dr)
    if Q.lep_z < 0.5187302 and Q.lep_iso < 14.38858:
        z += -0.02542391 * (0.5187302 - Q.lep_z) * (14.38858 - Q.lep_iso)
    if Q.lep_z < 0.5187302 and Q.sdb_2_z < 0.2509165:
        z += 1.352006 * (0.5187302 - Q.lep_z) * (0.2509165 - Q.sdb_2_z)
    if Q.max_abs_dz < 6.402344 and Q.lepsj_3_dr < 0.01012269:
        z += -2.096206 * (6.402344 - Q.max_abs_dz) * (0.01012269 - Q.lepsj_3_dr)
    if Q.sj3_pair_mass_max < 114.7848 and Q.sjf_3_3_n_d3 > 3.0:
        z += 0.00274188 * (114.7848 - Q.sj3_pair_mass_max) * (Q.sjf_3_3_n_d3 - 3.0)
    if Q.ktd_ln_d34 > -9.531553 and Q.pz_lnkt3 < 0.102661:
        z += 0.9149784 * (Q.ktd_ln_d34 - -9.531553) * (0.102661 - Q.pz_lnkt3)
    if Q.e3_b2 < 0.0004126585 and Q.sdb_5_z > 0.04013001:
        z += 3085.764 * (0.0004126585 - Q.e3_b2) * (Q.sdb_5_z - 0.04013001)
    if Q.lepsj_3_maxsd0 < 2.413264 and Q.sv_1_sd0_sum < 509.6197:
        z += -0.0001773361 * (2.413264 - Q.lepsj_3_maxsd0) * (509.6197 - Q.sv_1_sd0_sum)
    return z


def neuron_116(Q):
    z = -3.399341e-07
    return z


def neuron_117(Q):
    z = 2.862137e-06
    return z


def neuron_118(Q):
    z = -8.233226e-06
    return z


def neuron_119(Q):
    z = -6.33687e-07
    return z


def neuron_120(Q):
    z = 1.378949
    if Q.lep_ptrel < 3.53503:
        z += 0.1546489 * Q.lep_ptrel - 0.938807
    if 3.53503 <= Q.lep_ptrel < 12.15228:
        z += 0.04550391 * Q.lep_ptrel - 0.5529763
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.01813151 * Q.lep_ptrel - 0.4954183
    if Q.lep_ptrel >= 43.20788:
        z += -0.01096814 * Q.lep_ptrel + 0.761916
    if Q.n_pairs_kt_above_3 < 62.0:
        z += 0.0145336 * Q.n_pairs_kt_above_3 - 0.9010831
    if Q.lep_z < 0.004135872:
        z += 150.5242 * Q.lep_z - 0.6747195
    if 0.004135872 <= Q.lep_z < 0.5187302:
        z += 0.101382 * Q.lep_z - 0.05258989
    if Q.tau32 < 0.6800935:
        z += -0.786154 * Q.tau32 + 0.5346582
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.008998152 * Q.n_pairs_kt_above_1 - 0.7198521
    if Q.pair_mean_lnm2 >= 4.787589:
        z += -0.3540441 * Q.pair_mean_lnm2 + 1.695018
    if Q.mass_top10 >= 103.4976:
        z += -0.008510876 * Q.mass_top10 + 0.8808557
    if Q.pz_lnkt0 < 0.1589878:
        z += -4.65859 * Q.pz_lnkt0 + 0.7406588
    if Q.n_pairs_kt_above_10 < 11.0:
        z += 0.02672583 * Q.n_pairs_kt_above_10 - 0.2939842
    if Q.lep_iso < 0.4381892:
        z += 0.9189832 * Q.lep_iso - 0.2746878
    if 0.4381892 <= Q.lep_iso < 2.907433:
        z += -0.05183803 * Q.lep_iso + 0.1507156
    if Q.lepsj_2_dr < 0.0008210142:
        z += -554.8759 * Q.lepsj_2_dr + 0.1328081
    if 0.0008210142 <= Q.lepsj_2_dr < 0.06573337:
        z += 4.972133 * Q.lepsj_2_dr - 0.3268351
    if Q.lep_dr < 0.08178299:
        z += -3.380594 * Q.lep_dr + 0.1118609
    if 0.08178299 <= Q.lep_dr < 0.3014662:
        z += 0.7493254 * Q.lep_dr - 0.2258963
    if Q.sjq_3_3_k1 >= 0.4037488:
        z += 0.6160222 * Q.sjq_3_3_k1 - 0.2487182
    if Q.mass_charged < 68.20711:
        z += 0.02045832 * Q.mass_charged - 1.395403
    if Q.lam1 >= 0.04304553:
        z += -5.583053 * Q.lam1 + 0.2403255
    if Q.M2 < 0.05805236:
        z += -14.53741 * Q.M2 + 0.8439308
    if Q.mass_neutral < 63.87145:
        z += -0.008936661 * Q.mass_neutral + 0.5707975
    if Q.mass < 71.96396:
        z += 0.01299327 * Q.mass - 2.499613
    if 71.96396 <= Q.mass < 90.08945:
        z += 0.04253461 * Q.mass - 4.625525
    if 90.08945 <= Q.mass < 105.7234:
        z += 0.03770301 * Q.mass - 4.190248
    if 105.7234 <= Q.mass < 114.0172:
        z += 0.02461563 * Q.mass - 2.806607
    if Q.lund3_lndelta >= -2.036501:
        z += -0.1226097 * Q.lund3_lndelta - 0.2496948
    if Q.pz_lnd2 < 0.03277088:
        z += 7.940003 * Q.pz_lnd2 - 0.2602009
    if Q.e3 >= 0.00228569:
        z += 58.4016 * Q.e3 - 0.133488
    if Q.sjf_2_2_max3d < 3.029824:
        z += 0.05320586 * Q.sjf_2_2_max3d - 0.1612044
    if Q.n_s3d_above_10 < 3.0:
        z += -0.02541496 * Q.n_s3d_above_10 + 0.07624488
    if Q.sj3_dr13 >= 0.7462286:
        z += -0.3771521 * Q.sj3_dr13 + 0.2814417
    if Q.kt2_min12_jp < 4.912483:
        z += 0.02133329 * Q.kt2_min12_jp - 0.1047995
    if Q.ak02_1_n_lep < 1.0:
        z += 0.1442574 * Q.ak02_1_n_lep - 0.1442574
    if Q.ak02_1_n_lep >= 1.0:
        z += -0.1873555 * Q.ak02_1_n_lep + 0.1873555
    if Q.pair_mean_lndelta >= -1.790445:
        z += -0.5070404 * Q.pair_mean_lndelta - 0.9078279
    if Q.pz_lnd0 < 0.05066241:
        z += -2.517115 * Q.pz_lnd0 + 0.1275231
    if Q.N2_b05 < 0.3667049:
        z += 3.716455 * Q.N2_b05 - 1.362843
    if Q.mass_top20 >= 114.4658:
        z += -0.008885307 * Q.mass_top20 + 1.017064
    if Q.mass_top50 < 135.5368:
        z += -0.004832483 * Q.mass_top50 + 0.8636855
    if 135.5368 <= Q.mass_top50 < 161.1264:
        z += -0.01360422 * Q.mass_top50 + 2.052578
    if 161.1264 <= Q.mass_top50 < 178.725:
        z += -0.002981077 * Q.mass_top50 + 0.3409097
    if Q.mass_top50 >= 178.725:
        z += 0.001851406 * Q.mass_top50 - 0.5227757
    if Q.mres_sd_prong_mass2 < 4.6892:
        z += 0.02344802 * Q.mres_sd_prong_mass2 - 0.1099525
    if Q.sd_mass < 78.4753:
        z += -0.002397963 * Q.sd_mass - 0.1589319
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += -0.01240703 * Q.sd_mass + 0.6265324
    if 88.81751 <= Q.sd_mass < 111.701:
        z += 0.0207761 * Q.sd_mass - 2.32071
    if Q.lep_ptrel < 3.53503 and Q.n_s3d_above_3 < 10.0:
        z += -0.04358387 * (3.53503 - Q.lep_ptrel) * (10.0 - Q.n_s3d_above_3)
    if Q.lep_z < 0.5187302 and Q.mass_neutral < 69.1565:
        z += -0.0193434 * (0.5187302 - Q.lep_z) * (69.1565 - Q.mass_neutral)
    if Q.lep_z < 0.5187302 and Q.mass_displaced3 > 0.0:
        z += 0.02036718 * (0.5187302 - Q.lep_z) * (Q.mass_displaced3 - 0.0)
    if Q.lep_ptrel < 3.53503 and Q.jd_3d_4 < 172.888:
        z += 0.0007592704 * (3.53503 - Q.lep_ptrel) * (172.888 - Q.jd_3d_4)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.sjf_2_1_z_d3 < 0.2710171:
        z += 0.01915683 * (62.0 - Q.n_pairs_kt_above_3) * (0.2710171 - Q.sjf_2_1_z_d3)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.lepsj_2_dr < 0.1822601:
        z += 0.01585061 * (62.0 - Q.n_pairs_kt_above_3) * (0.1822601 - Q.lepsj_2_dr)
    if Q.n_sd0_above_5 < 3.0 and Q.ak02_3_n_lep > 0.0:
        z += 0.09476806 * (3.0 - Q.n_sd0_above_5) * (Q.ak02_3_n_lep - 0.0)
    if Q.lep_ptrel < 3.53503 and Q.n_photon > 5.0:
        z += 0.006925147 * (3.53503 - Q.lep_ptrel) * (Q.n_photon - 5.0)
    if Q.lep_ptrel < 12.15228 and Q.sdb_2_n < 18.0:
        z += 0.002398713 * (12.15228 - Q.lep_ptrel) * (18.0 - Q.sdb_2_n)
    if Q.lep_z < 0.5187302 and Q.planar_flow < 0.7095008:
        z += 0.3641877 * (0.5187302 - Q.lep_z) * (0.7095008 - Q.planar_flow)
    if Q.lep_ptrel < 12.15228 and Q.jd_3d_6 < 3.151933:
        z += -0.02606454 * (12.15228 - Q.lep_ptrel) * (3.151933 - Q.jd_3d_6)
    if Q.lep_iso < 2.907433 and Q.jd_3d_6 < 5.367501:
        z += 0.03417461 * (2.907433 - Q.lep_iso) * (5.367501 - Q.jd_3d_6)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.dc_2_n_lep < 1.0:
        z += 0.003343179 * (62.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.dc_2_n_lep)
    if Q.lep_ptrel < 3.53503 and Q.e4_b05 > 4.363663e-05:
        z += 457.648 * (3.53503 - Q.lep_ptrel) * (Q.e4_b05 - 4.363663e-05)
    if Q.mass_top10 > 103.4976 and Q.sum_pt_top40 > 502.3395:
        z += 2.357606e-05 * (Q.mass_top10 - 103.4976) * (Q.sum_pt_top40 - 502.3395)
    if Q.lep_ptrel < 3.53503 and Q.max_abs_d0 < 5.8125:
        z += 0.01300314 * (3.53503 - Q.lep_ptrel) * (5.8125 - Q.max_abs_d0)
    if Q.lepsj_2_dr < 0.06573337 and Q.sjq_3_3_k1 > 0.4037488:
        z += -6.967464 * (0.06573337 - Q.lepsj_2_dr) * (Q.sjq_3_3_k1 - 0.4037488)
    if Q.n_pairs_kt_above_10 < 11.0 and Q.jd_3d_6 < 3.151933:
        z += 0.009581959 * (11.0 - Q.n_pairs_kt_above_10) * (3.151933 - Q.jd_3d_6)
    if Q.lep_ptrel < 3.53503 and Q.sv_2_z > 0.0:
        z += -1.153988 * (3.53503 - Q.lep_ptrel) * (Q.sv_2_z - 0.0)
    if Q.lep_ptrel < 12.15228 and Q.n_muon > 0.0:
        z += 0.02092896 * (12.15228 - Q.lep_ptrel) * (Q.n_muon - 0.0)
    if Q.lep_iso < 0.4381892 and Q.dc_1_n_lep < 1.0:
        z += 0.9652748 * (0.4381892 - Q.lep_iso) * (1.0 - Q.dc_1_n_lep)
    if Q.mass_charged < 68.20711 and Q.jd_3d_6 < 5.367501:
        z += 0.004419242 * (68.20711 - Q.mass_charged) * (5.367501 - Q.jd_3d_6)
    if Q.lepsj_2_dr < 0.06573337 and Q.jd_3d_6 < 5.367501:
        z += -1.263127 * (0.06573337 - Q.lepsj_2_dr) * (5.367501 - Q.jd_3d_6)
    if Q.M2 < 0.05805236 and Q.jd_3d_5 < 6.771002:
        z += -4.606488 * (0.05805236 - Q.M2) * (6.771002 - Q.jd_3d_5)
    if Q.mass < 114.0172 and Q.jd_3d_6 < 5.367501:
        z += 0.003041012 * (114.0172 - Q.mass) * (5.367501 - Q.jd_3d_6)
    if Q.pz_lnkt0 < 0.1589878 and Q.sjq_2_prod_k05 > -0.3383985:
        z += -4.344312 * (0.1589878 - Q.pz_lnkt0) * (Q.sjq_2_prod_k05 - -0.3383985)
    if Q.lep_ptrel < 12.15228 and Q.pz_lnd2 > 0.008992646:
        z += -0.04078254 * (12.15228 - Q.lep_ptrel) * (Q.pz_lnd2 - 0.008992646)
    if Q.mass < 114.0172 and Q.z_neutral_had < 0.1271955:
        z += 0.08702699 * (114.0172 - Q.mass) * (0.1271955 - Q.z_neutral_had)
    if Q.n_s3d_above_10 < 3.0 and Q.sdb_4_z > 0.02669833:
        z += -0.3466419 * (3.0 - Q.n_s3d_above_10) * (Q.sdb_4_z - 0.02669833)
    if Q.lep_z < 0.5187302 and Q.sjq_3_3_k1 < -0.2912597:
        z += 0.7590602 * (0.5187302 - Q.lep_z) * (-0.2912597 - Q.sjq_3_3_k1)
    if Q.lam1 > 0.04304553 and Q.dc_3_n_lep > 0.0:
        z += -5.208384 * (Q.lam1 - 0.04304553) * (Q.dc_3_n_lep - 0.0)
    if Q.lam1 > 0.04304553 and Q.lund3_lnkt > 0.3409807:
        z += 0.8704833 * (Q.lam1 - 0.04304553) * (Q.lund3_lnkt - 0.3409807)
    if Q.lep_ptrel < 12.15228 and Q.sdb_3_n < 5.0:
        z += 0.002083805 * (12.15228 - Q.lep_ptrel) * (5.0 - Q.sdb_3_n)
    if Q.pz_lnd0 < 0.05066241 and Q.dr_19 < 0.0538419:
        z += 275.4576 * (0.05066241 - Q.pz_lnd0) * (0.0538419 - Q.dr_19)
    return z


def neuron_121(Q):
    z = -5.730521e-06
    return z


def neuron_122(Q):
    z = -2.395902e-06
    return z


def neuron_123(Q):
    z = 0.01961422
    return z


def neuron_124(Q):
    z = -0.1965684
    if Q.lep_z < 0.221436:
        z += -3.444204 * Q.lep_z + 0.7844587
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -0.1842983 * Q.lep_z + 0.06259824
    if Q.n_pairs_kt_above_1 < 58.0:
        z += 0.01448955 * Q.n_pairs_kt_above_1 - 1.651347
    if 58.0 <= Q.n_pairs_kt_above_1 < 366.0:
        z += 0.002632965 * Q.n_pairs_kt_above_1 - 0.9636651
    if Q.n_s3d_above_3 < 1.0:
        z += 0.3633161 * Q.n_s3d_above_3 - 0.3633161
    if Q.n_s3d_above_3 >= 3.0:
        z += 0.02004976 * Q.n_s3d_above_3 - 0.06014929
    if Q.mass < 117.4867:
        z += -0.006127872 * Q.mass + 0.7199433
    if Q.lep_ptrel < 43.20788:
        z += 6.565232e-05 * Q.lep_ptrel - 0.002836698
    if Q.lep_iso < 0.4381892:
        z += 1.015476 * Q.lep_iso - 0.6854169
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += 0.2602499 * Q.lep_iso - 0.354485
    if Q.n_lepton < 1.0:
        z += -0.9622969 * Q.n_lepton + 0.9622969
    if Q.mass_top40 >= 70.88236:
        z += -0.001846506 * Q.mass_top40 + 0.1308847
    if 76.40585 <= Q.mres_sd_mass_b0z005 < 86.65765:
        z += -0.01010043 * Q.mres_sd_mass_b0z005 + 0.7717321
    if 86.65765 <= Q.mres_sd_mass_b0z005 < 91.82593:
        z += -0.001537839 * Q.mres_sd_mass_b0z005 + 0.02971791
    if 91.82593 <= Q.mres_sd_mass_b0z005 < 111.9362:
        z += 0.02116813 * Q.mres_sd_mass_b0z005 - 2.055279
    if 111.9362 <= Q.mres_sd_mass_b0z005 < 125.8718:
        z += 0.01264878 * Q.mres_sd_mass_b0z005 - 1.101656
    if 125.8718 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += -0.02996682 * Q.mres_sd_mass_b0z005 + 4.262445
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += 0.001579821 * Q.mres_sd_mass_b0z005 - 0.7826271
    if Q.lepsj_3_maxsd0 < 0.8036986:
        z += 0.3101886 * Q.lepsj_3_maxsd0 - 0.09369789
    if 0.8036986 <= Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.09667219 * Q.lepsj_3_maxsd0 + 0.2332956
    if Q.z_displaced3 < 0.04884386:
        z += 10.55607 * Q.z_displaced3 - 0.515599
    if Q.sjq_2_prod_k1 < 0.03330871:
        z += 2.308118 * Q.sjq_2_prod_k1 - 0.1715979
    if 0.03330871 <= Q.sjq_2_prod_k1 < 0.09802954:
        z += 1.463477 * Q.sjq_2_prod_k1 - 0.143464
    if 0.1103483 <= Q.sjq_2_sumabs_k1 < 0.2769039:
        z += -0.2836637 * Q.sjq_2_sumabs_k1 + 0.0313018
    if Q.sjq_2_sumabs_k1 >= 0.2769039:
        z += -1.020977 * Q.sjq_2_sumabs_k1 + 0.2354667
    if Q.z_photon >= 0.1149688:
        z += -1.558434 * Q.z_photon + 0.1791712
    if Q.lepsj_2_dr < 0.04178742:
        z += -4.29877 * Q.lepsj_2_dr + 0.1796345
    if Q.sjq_2_prod_k05 >= 0.004703917:
        z += 0.5895737 * Q.sjq_2_prod_k05 - 0.002773306
    if Q.sjf_2_1_n_d3 >= 7.0:
        z += 0.1796797 * Q.sjf_2_1_n_d3 - 1.257758
    if Q.sjf_2_2_max3d < 1.864422:
        z += 0.2299805 * Q.sjf_2_2_max3d - 0.4287807
    if Q.tau2 < 0.1076451:
        z += -0.9305078 * Q.tau2 + 0.1001646
    if Q.sj4_pair_mass_max < 66.70506:
        z += 0.009071784 * Q.sj4_pair_mass_max - 0.6051339
    if Q.sjf_4_1_n_d3 >= 2.0:
        z += -0.03924684 * Q.sjf_4_1_n_d3 + 0.07849368
    if Q.pz_lnd0 < 0.06871203:
        z += 3.081162 * Q.pz_lnd0 - 0.2117129
    if Q.kt2_2_n_lep < 1.0:
        z += -0.06255722 * Q.kt2_2_n_lep + 0.06255722
    if Q.lep_dr >= 0.04607888:
        z += 1.06482 * Q.lep_dr - 0.04906572
    if Q.n_particles >= 67.0:
        z += -0.008286923 * Q.n_particles + 0.5552239
    z += 0.04223463 * Q.lepsj_2_n_d3
    if Q.sv_1_sd0_sum < 35.45098:
        z += -0.004719786 * Q.sv_1_sd0_sum + 0.167321
    if Q.max_abs_d0 < 10.52344:
        z += 0.03542686 * Q.max_abs_d0 - 0.3728123
    if Q.z_displaced5 >= 0.04748739:
        z += 1.233919 * Q.z_displaced5 - 0.05859557
    if 69.04524 <= Q.mres_sd_prong_mass1 < 84.19324:
        z += -0.01685226 * Q.mres_sd_prong_mass1 + 1.163568
    if Q.mres_sd_prong_mass1 >= 84.19324:
        z += 0.008440144 * Q.mres_sd_prong_mass1 - 0.9658811
    if Q.tau21_b2 >= 0.3898586:
        z += -0.4896891 * Q.tau21_b2 + 0.1909095
    if Q.jd_sum_abs_sd0_top5 < 55.83114:
        z += -0.02111543 * Q.jd_sum_abs_sd0_top5 + 1.178898
    if Q.jd_sum_abs_sd0_top3 < 48.72265:
        z += 0.01590828 * Q.jd_sum_abs_sd0_top3 - 0.7750935
    if Q.n_s3d_above_10 >= 2.0:
        z += -0.1082467 * Q.n_s3d_above_10 + 0.2164934
    if Q.dc_3_charge >= 0.6283033:
        z += -0.4060263 * Q.dc_3_charge + 0.2551076
    if Q.sdb_2_z < 0.1530389:
        z += 3.813523 * Q.sdb_2_z - 0.5836174
    if Q.lepsj_3_dr < 0.04982189:
        z += -15.99096 * Q.lepsj_3_dr + 0.7967
    if Q.tau32 < 0.5498426:
        z += -0.9678946 * Q.tau32 + 0.5321897
    if Q.psi_0p1 < 0.006220408:
        z += 22.83506 * Q.psi_0p1 - 0.1420434
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 91.15481:
        z += -0.01987078 * Q.mres_sd_mass_b2z01 + 1.513217
    if 91.15481 <= Q.mres_sd_mass_b2z01 < 107.2089:
        z += 0.02756273 * Q.mres_sd_mass_b2z01 - 2.810575
    if Q.mres_sd_mass_b2z01 >= 107.2089:
        z += 0.001284519 * Q.mres_sd_mass_b2z01 + 0.006683962
    if Q.lep_ptrel < 43.20788 and Q.sj3_pair_mass_max < 142.9952:
        z += 8.072417e-05 * (43.20788 - Q.lep_ptrel) * (142.9952 - Q.sj3_pair_mass_max)
    if Q.n_s3d_above_3 > 3.0 and Q.M2_b2 < 0.06762785:
        z += -1.484814 * (Q.n_s3d_above_3 - 3.0) * (0.06762785 - Q.M2_b2)
    if Q.n_s3d_above_3 > 3.0 and Q.sip_3d_1 < 578.2954:
        z += -7.469397e-05 * (Q.n_s3d_above_3 - 3.0) * (578.2954 - Q.sip_3d_1)
    if Q.n_dr_0p4_up < 15.0 and Q.dc_ntag > 0.0:
        z += 0.006291314 * (15.0 - Q.n_dr_0p4_up) * (Q.dc_ntag - 0.0)
    if Q.lep_z < 0.3396572 and Q.z_displaced5 > 0.1046203:
        z += -2.8437 * (0.3396572 - Q.lep_z) * (Q.z_displaced5 - 0.1046203)
    if Q.mass_top40 > 70.88236 and Q.ak02_3_z < 0.07650476:
        z += -0.01689323 * (Q.mass_top40 - 70.88236) * (0.07650476 - Q.ak02_3_z)
    if Q.n_pairs_kt_above_1 < 366.0 and Q.C2_b2 < 0.221369:
        z += 0.002481339 * (366.0 - Q.n_pairs_kt_above_1) * (0.221369 - Q.C2_b2)
    if Q.tau2 < 0.1076451 and Q.jet_charge_k03 > -0.05990128:
        z += 2.452109 * (0.1076451 - Q.tau2) * (Q.jet_charge_k03 - -0.05990128)
    if Q.lep_z < 0.221436 and Q.dc_4_z > 0.0:
        z += 7.449652 * (0.221436 - Q.lep_z) * (Q.dc_4_z - 0.0)
    if Q.z_photon > 0.1149688 and Q.jd_3d_5 < 6.771002:
        z += 0.112877 * (Q.z_photon - 0.1149688) * (6.771002 - Q.jd_3d_5)
    if Q.sjq_2_sumabs_k1 > 0.2769039 and Q.jd_3d_4 < 2.953464:
        z += 0.5410535 * (Q.sjq_2_sumabs_k1 - 0.2769039) * (2.953464 - Q.jd_3d_4)
    if Q.n_lepton < 1.0 and Q.jd_3d_4 < 2.953464:
        z += -0.4626323 * (1.0 - Q.n_lepton) * (2.953464 - Q.jd_3d_4)
    if Q.n_s3d_above_3 > 3.0 and Q.jd_3d_4 < 172.888:
        z += -0.0003134113 * (Q.n_s3d_above_3 - 3.0) * (172.888 - Q.jd_3d_4)
    if Q.mass_top40 > 70.88236 and Q.lund1_lndelta < -0.4311181:
        z += 0.01373567 * (Q.mass_top40 - 70.88236) * (-0.4311181 - Q.lund1_lndelta)
    if Q.max_abs_d0 < 10.52344 and Q.ak02_min12_n_disp3 < 1.0:
        z += 0.02443036 * (10.52344 - Q.max_abs_d0) * (1.0 - Q.ak02_min12_n_disp3)
    if Q.kt2_2_n_lep < 1.0 and Q.sdb_2_z > 0.3208052:
        z += -1.0364 * (1.0 - Q.kt2_2_n_lep) * (Q.sdb_2_z - 0.3208052)
    if Q.sdb_2_z < 0.1530389 and Q.dr_30 < 0.04891969:
        z += 39.64732 * (0.1530389 - Q.sdb_2_z) * (0.04891969 - Q.dr_30)
    if Q.sjq_2_sumabs_k1 > 0.2769039 and Q.lepsj_3_maxsd0 < 0.8036986:
        z += 0.4031681 * (Q.sjq_2_sumabs_k1 - 0.2769039) * (0.8036986 - Q.lepsj_3_maxsd0)
    if Q.sjq_2_prod_k1 < 0.09802954 and Q.lepsj_3_maxsd0 < 2.413264:
        z += 0.27797 * (0.09802954 - Q.sjq_2_prod_k1) * (2.413264 - Q.lepsj_3_maxsd0)
    if Q.lepsj_3_dr < 0.04982189 and Q.lepsj_3_maxsd0 < 2.413264:
        z += -5.25704 * (0.04982189 - Q.lepsj_3_dr) * (2.413264 - Q.lepsj_3_maxsd0)
    return z


def neuron_125(Q):
    z = -1.190763e-06
    return z


def neuron_126(Q):
    z = 7.632632e-06
    return z


def neuron_127(Q):
    z = -1.966694e-07
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
