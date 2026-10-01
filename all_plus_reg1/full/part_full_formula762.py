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

Whole test file (2,000,000 jets): accuracy 74.99% (the network: 86.03%); same class as the network for 79.75% of jets.

Quantities:
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3_b05                 e4·e2/e3² with β = 0.5
  Q.D2                     energy correlation ratio e3/e2³
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
  Q.ak02_3_z               pT share of the 3rd anti-kT 0.2 subjet (0 if none)
  Q.ak02_dr13              distance between the pT-weighted centres of the hardest and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_dr23              distance between the pT-weighted centres of the 2nd and 3rd anti-kT 0.2 subjet (0 if missing)
  Q.ak02_min12_jp          the smaller of the two hardest anti-kT 0.2 subjets: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.ak02_n                 number of anti-kT R = 0.2 subjets with pT > 10 GeV
  Q.dc_1_n_lep             hardest prong: number of its electrons and muons
  Q.dc_1_z                 pT share of the hardest prong (0 if none)
  Q.dc_1_z_disp3           hardest prong: pT share (of the jet) of its tracks with d0/σ > 3
  Q.dc_2_jp                2nd prong: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
  Q.dc_2_n_lep             2nd prong: number of its electrons and muons
  Q.dc_3_z                 pT share of the 3rd prong (0 if none)
  Q.dc_ntag                number of the 4 hardest prongs whose 2nd largest d0/σ is above 3
  Q.dc_pair_mass_min       smallest mass (E) of two of the 4 hardest prongs [GeV] (0 if fewer than 2)
  Q.dc_split2_dr           ΔR of the 2nd hard splitting of the prong finding (ranked by pT; 0 if none)
  Q.dc_tag_2nd             second largest 2nd-largest d0/σ among the 4 hardest prongs (double tag)
  Q.dr02                   ΔR between particles 0 and 2
  Q.dr_19                  ΔR from the jet axis of particle 19 (ParT input; empty slot: 0)
  Q.dr_29                  ΔR from the jet axis of particle 29 (ParT input; empty slot: 0)
  Q.dr_min_012             smallest distance among the 3 hardest
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.e3                     energy correlation e3 (β=1, 32 hardest)
  Q.e3_b05                 energy correlation e3 with β = 0.5
  Q.e3_b2                  energy correlation e3 with β = 2
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 16 hardest)
  Q.e4_b05                 energy correlation e4 with β = 0.5
  Q.ecf_g31                generalized energy correlation ₁e₃ (β=1; products of the smallest angles)
  Q.ecf_g41                generalized energy correlation ₁e₄ (β=1; products of the smallest angles)
  Q.ecf_g42                generalized energy correlation ₂e₄ (β=1; products of the smallest angles)
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.ischhad_0              1 if a charged hadron of particle 0 (ParT input; empty slot: 0)
  Q.jd_3d_4                the 4th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_5                the 5th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_3d_6                the 6th largest 3D significance (σ clipped to 1) (0 if fewer)
  Q.jd_sum_abs_sd0_top3    sum of the 3 largest |d0/σ|
  Q.jd_sum_abs_sd0_top5    sum of the 5 largest |d0/σ|
  Q.jet_abs_eta            absolute pseudorapidity of the jet axis
  Q.jet_charge_k03         jet charge with κ = 0.3
  Q.jet_charge_k05         jet charge with κ = 0.5
  Q.jet_e                  jet energy [GeV]
  Q.kt2_1_n_disp3          hardest kT subjet: number of its tracks with d0/σ > 3
  Q.kt2_2_n_lep            2nd kT subjet: number of its electrons and muons
  Q.kt2_min12_jp           the smaller of the two hardest kT subjets: Σ −ln P(|d0/σ|) of its tracks with d0/σ > 0, P = erfc(|d0/σ|/√2)
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
  Q.lepsj_3_maxsd0         largest |d0/σ| in the lepton’s nearest subjet (without the lepton); 3 subjets (0 if no lepton)
  Q.lepsj_3_n_d3           number of tracks with |d0/σ| > 3 in the lepton’s nearest subjet (without the lepton); 3 subjets (0 if no lepton)
  Q.lne_0                  ln E [GeV] of particle 0 (ParT input; empty slot: ln 1e-8)
  Q.lne_1                  ln E [GeV] of particle 1 (ParT input; empty slot: ln 1e-8)
  Q.lne_16                 ln E [GeV] of particle 16 (ParT input; empty slot: ln 1e-8)
  Q.lne_4                  ln E [GeV] of particle 4 (ParT input; empty slot: ln 1e-8)
  Q.lne_8                  ln E [GeV] of particle 8 (ParT input; empty slot: ln 1e-8)
  Q.lnptrel_25             ln(pT / pT of the jet) of particle 25 (ParT input; empty slot: ln 1e-8)
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
  Q.n_sd0_above_3          number of charged particles with d0 significance > 3
  Q.n_sdz_above_2          number of charged particles with dz significance > 2
  Q.n_sdz_above_5          number of charged particles with dz significance > 5
  Q.nca_sj4_pair2nd_over_mass second largest mass of two of the 4 subjets over the jet mass
  Q.nca_sj4_pair_mass_2nd  second largest mass of two of the 4 subjets [GeV]
  Q.pair_max_lnkt          largest ln kT among all pairs
  Q.pair_max_lnm2          largest ln m² among all pairs
  Q.pair_mean_lndelta      zᵢzⱼ-weighted mean of ln ΔRᵢⱼ over all pairs
  Q.pair_mean_lnkt         zᵢzⱼ-weighted mean of ln kT = ln(min(pTᵢ, pTⱼ)·ΔRᵢⱼ) over all pairs
  Q.pair_mean_lnm2         zᵢzⱼ-weighted mean of ln mᵢⱼ² (massless) over all pairs
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
  Q.sjf_4_1_max3d          subjet 1 of 4 (k-means axes, by pT): largest 3D significance (0 if no track)
  Q.sjf_4_1_z_d3           subjet 1 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_2_z_d3           subjet 2 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
  Q.sjf_4_3_z_d3           subjet 3 of 4 (k-means axes, by pT): pT share of the tracks with |d0/σ| > 3
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
  Q.sjq_3_prod_k1          product of the charges (κ = 1) of the two hardest of 3 subjets
  Q.sjq_3_sumabs_k1        |sum| of the charges (κ = 1) of the two hardest of 3 subjets
  Q.sum_e                  total energy of the particles [GeV]
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sv_1_dr                hardest displaced-track cluster: ΔR of its centre from the jet axis (0 if none)
  Q.sv_1_n                 hardest displaced-track cluster: number of tracks (0 if none)
  Q.sv_1_sd0_sum           hardest displaced-track cluster: Σ d0/σ of its tracks (0 if none)
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
  Q.tau5                   N-subjettiness τ5 (β=1)
  Q.tau54                  N-subjettiness τ5/τ4
  Q.z_charged              pT share of charged particles
  Q.z_charged_had          pT share of charged hadrons
  Q.z_displaced3           pT share of the charged particles with |d0|/σ > 3
  Q.z_displaced5           pT share of the charged particles with |d0|/σ > 5
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.z_dr_0p4_up            pT share of the particles with ΔR ≥ 0.4
  Q.z_electron             pT share of electrons
  Q.z_muon                 pT share of muons
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
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3_b05=ecfb('e4', 0.5) * ecfb('e2', 0.5) / max(ecfb('e3', 0.5) ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
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
        ak02_3_z=pq('ak02_3_z'),
        ak02_dr13=pq('ak02_dr13'),
        ak02_dr23=pq('ak02_dr23'),
        ak02_min12_jp=pq('ak02_min12_jp'),
        ak02_n=pq('ak02_n'),
        dc_1_n_lep=pq('dc_1_n_lep'),
        dc_1_z=pq('dc_1_z'),
        dc_1_z_disp3=pq('dc_1_z_disp3'),
        dc_2_jp=pq('dc_2_jp'),
        dc_2_n_lep=pq('dc_2_n_lep'),
        dc_3_z=pq('dc_3_z'),
        dc_ntag=pq('dc_ntag'),
        dc_pair_mass_min=pq('dc_pair_mass_min'),
        dc_split2_dr=pq('dc_split2_dr'),
        dc_tag_2nd=pq('dc_tag_2nd'),
        dr02=math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        dr_19=pfeat(19, 'dr'),
        dr_29=pfeat(29, 'dr'),
        dr_min_012=min(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        e3=ecf('e3'),
        e3_b05=ecfb('e3', 0.5),
        e3_b2=ecfb('e3', 2),
        e4=ecf('e4'),
        e4_b05=ecfb('e4', 0.5),
        ecf_g31=ecfb('g31', 1),
        ecf_g41=ecfb('g41', 1),
        ecf_g42=ecfb('g42', 1),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        ischhad_0=pfeat(0, 'ischhad'),
        jd_3d_4=pq('jd_3d_4'),
        jd_3d_5=pq('jd_3d_5'),
        jd_3d_6=pq('jd_3d_6'),
        jd_sum_abs_sd0_top3=pq('jd_sum_abs_sd0_top3'),
        jd_sum_abs_sd0_top5=pq('jd_sum_abs_sd0_top5'),
        jet_abs_eta=abs(jet_eta),
        jet_charge_k03=sum(charge[i] * z[i] ** 0.3 for i in real),
        jet_charge_k05=sum(charge[i] * z[i] ** 0.5 for i in real),
        jet_e=jet_energy,
        kt2_1_n_disp3=pq('kt2_1_n_disp3'),
        kt2_2_n_lep=pq('kt2_2_n_lep'),
        kt2_min12_jp=pq('kt2_min12_jp'),
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
        lepsj_3_maxsd0=pq('lepsj_3_maxsd0'),
        lepsj_3_n_d3=pq('lepsj_3_n_d3'),
        lne_0=pfeat(0, 'lne'),
        lne_1=pfeat(1, 'lne'),
        lne_16=pfeat(16, 'lne'),
        lne_4=pfeat(4, 'lne'),
        lne_8=pfeat(8, 'lne'),
        lnptrel_25=pfeat(25, 'lnptrel'),
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
        n_sd0_above_3=nsig('d0', 3),
        n_sdz_above_2=nsig('dz', 2),
        n_sdz_above_5=nsig('dz', 5),
        nca_sj4_pair2nd_over_mass=pq('nca_sj4_pair2nd_over_mass'),
        nca_sj4_pair_mass_2nd=pq('nca_sj4_pair_mass_2nd'),
        pair_max_lnkt=pairmax('lnkt'),
        pair_max_lnm2=pairmax('lnm2'),
        pair_mean_lndelta=pairsum('lndelta'),
        pair_mean_lnkt=pairsum('lnkt'),
        pair_mean_lnm2=pairsum('lnm2'),
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
        sjf_4_1_max3d=pq('sjf_4_1_max3d'),
        sjf_4_1_z_d3=pq('sjf_4_1_z_d3'),
        sjf_4_2_z_d3=pq('sjf_4_2_z_d3'),
        sjf_4_3_z_d3=pq('sjf_4_3_z_d3'),
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
        sjq_3_prod_k1=pq('sjq_3_prod_k1'),
        sjq_3_sumabs_k1=pq('sjq_3_sumabs_k1'),
        sum_e=sum(energy[i] for i in real),
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sv_1_dr=pq('sv_1_dr'),
        sv_1_n=pq('sv_1_n'),
        sv_1_sd0_sum=pq('sv_1_sd0_sum'),
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
        tau5=tau_n(5),
        tau54=tau_n(5) / max(tau_n(4), 1e-12),
        z_charged=sum(z[i] for i in real if charge[i] != 0),
        z_charged_had=sum(z[i] for i in real if ptype[i] == 1),
        z_displaced3=displaced(3, 'z'),
        z_displaced5=displaced(5, 'z'),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        z_dr_0p4_up=sum(z[i] for i in real if 0.4 <= dr[i] < 10),
        z_electron=sum(z[i] for i in real if ptype[i] == 4),
        z_muon=sum(z[i] for i in real if ptype[i] == 5),
        z_neutral_had=sum(z[i] for i in real if ptype[i] == 2),
        z_photon=sum(z[i] for i in real if ptype[i] == 3),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top3_slots=sum(pt[:3]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
    )


def neuron_0(Q):
    z = 0.00518969
    return z


def neuron_1(Q):
    z = -5.325267e-06
    return z


def neuron_2(Q):
    z = 5.447303e-06
    return z


def neuron_3(Q):
    z = 8.399839e-06
    return z


def neuron_4(Q):
    z = 2.79214e-06
    return z


def neuron_5(Q):
    z = -2.156924e-06
    return z


def neuron_6(Q):
    z = 1.770106e-05
    return z


def neuron_7(Q):
    z = 6.430825e-06
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
    z = 5.099298e-06
    return z


def neuron_13(Q):
    z = 4.422911e-06
    return z


def neuron_14(Q):
    z = 2.845304e-06
    return z


def neuron_15(Q):
    z = -6.459527e-06
    return z


def neuron_16(Q):
    z = 0.216396
    if Q.lep_ptrel < 27.3236:
        z += 0.02122965 * Q.lep_ptrel + 0.9572709
    if 27.3236 <= Q.lep_ptrel < 43.20788:
        z += 0.05626423 * Q.lep_ptrel
    if Q.lep_ptrel >= 43.20788:
        z += -0.01479761 * Q.lep_ptrel + 3.070431
    if Q.n_s3d_above_3 < 10.0:
        z += 0.2021991 * Q.n_s3d_above_3 - 2.021991
    if Q.pz_lnd2 < 0.08321691:
        z += -3.126151 * Q.pz_lnd2 + 0.2601486
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.001285899 * Q.n_pairs_kt_above_3 + 0.3458269
    if 28.0 <= Q.n_pairs_kt_above_3 < 34.0:
        z += -0.05163695 * Q.n_pairs_kt_above_3 + 1.755656
    if Q.mass_neutral >= 34.02385:
        z += 0.006200693 * Q.mass_neutral - 0.2109714
    if Q.z_displaced5 < 0.2801368:
        z += 4.351751 * Q.z_displaced5 - 1.219086
    if Q.e3_b2 < 0.0004126585:
        z += 451.7969 * Q.e3_b2 + 0.03728703
    if 0.0004126585 <= Q.e3_b2 < 0.0007909605:
        z += -591.3923 * Q.e3_b2 + 0.467768
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.007948513 * Q.mres_sd_mass_b0z005 - 0.9705134
    if 159.9242 <= Q.mres_sd_mass_b0z005 < 178.8957:
        z += -0.033779 * Q.mres_sd_mass_b0z005 + 5.702728
    if Q.mres_sd_mass_b0z005 >= 178.8957:
        z += -0.014927 * Q.mres_sd_mass_b0z005 + 2.330185
    z += -4.012375 * Q.z_charged_had
    if Q.mres_sd_mass_b2z01 < 91.15481:
        z += -0.004094017 * Q.mres_sd_mass_b2z01 + 0.3731893
    if 20.76537 <= Q.mass_2charged < 45.73288:
        z += 0.007177973 * Q.mass_2charged - 0.1490533
    if Q.mass_2charged >= 45.73288:
        z += -0.005884511 * Q.mass_2charged + 0.4483317
    if Q.max_abs_d0 < 5.8125:
        z += -0.04490618 * Q.max_abs_d0 + 0.2610172
    if Q.sdb_2_n < 16.0:
        z += -0.03090066 * Q.sdb_2_n + 0.4944106
    if Q.n_pairs_kt_above_1 < 80.0:
        z += -0.005062358 * Q.n_pairs_kt_above_1 + 0.4049886
    if Q.z_dr_0p4_up < 0.01553665:
        z += 9.185964 * Q.z_dr_0p4_up - 0.1427191
    if Q.mass_top5 >= 62.84493:
        z += 0.003266874 * Q.mass_top5 - 0.2053064
    if Q.mass_charged < 62.00562:
        z += -0.01548428 * Q.mass_charged + 0.9601127
    if Q.sjq_2_sumabs_k1 >= 0.2432922:
        z += 0.9404963 * Q.sjq_2_sumabs_k1 - 0.2288154
    if Q.sjq_2_prod_k1 < 0.09802954:
        z += -2.238593 * Q.sjq_2_prod_k1 + 0.2194483
    if Q.mres_sd_prong_mass1 < 46.61784:
        z += -0.003775957 * Q.mres_sd_prong_mass1 + 0.176027
    if Q.dr_min_012 < 0.01841101:
        z += 10.74097 * Q.dr_min_012 - 0.1977521
    if Q.jd_3d_6 >= 30.57088:
        z += -0.0008264276 * Q.jd_3d_6 + 0.02526462
    if Q.mass_top40 >= 112.406:
        z += 0.0147748 * Q.mass_top40 - 1.660777
    if Q.sjf_3_2_max3d < 3.101398:
        z += -0.05187902 * Q.sjf_3_2_max3d + 0.1608975
    if Q.n_photon < 18.0:
        z += 0.03036423 * Q.n_photon - 0.5465561
    if Q.n_dr_0p1_0p2 < 6.0:
        z += -0.04539403 * Q.n_dr_0p1_0p2 + 0.2723642
    z += 0.06390271 * Q.n_sdz_above_2
    if Q.sip_3d_3 < 175.9957:
        z += -0.001879266 * Q.sip_3d_3 + 0.3307426
    if Q.jd_3d_4 < 14.85973:
        z += 0.02892145 * Q.jd_3d_4 - 0.4297648
    if Q.z_dr_0p05_0p1 < 0.01556887:
        z += -12.48228 * Q.z_dr_0p05_0p1 + 0.194335
    if Q.M3 >= 0.02716047:
        z += 10.47668 * Q.M3 - 0.2845516
    if Q.lne_4 < 3.577424:
        z += -0.1073207 * Q.lne_4 + 0.3839317
    if Q.sj2_mass2 < 7.54918:
        z += 0.03980434 * Q.sj2_mass2 - 0.3004901
    if Q.sjq_2_2_nch < 3.0:
        z += -0.1075549 * Q.sjq_2_2_nch + 0.3226648
    if Q.z_electron < 0.1373477:
        z += 2.429763 * Q.z_electron - 0.3337222
    if Q.max_abs_dz < 0.4436035:
        z += 0.5082054 * Q.max_abs_dz + 0.01527051
    if 0.4436035 <= Q.max_abs_dz < 2.957031:
        z += -0.09577049 * Q.max_abs_dz + 0.2831963
    if Q.mass_displaced3 < 1.777286:
        z += 0.2267082 * Q.mass_displaced3 - 0.4029252
    if Q.sip_3d_2 < 226.3008:
        z += -0.002332651 * Q.sip_3d_2 + 0.5278806
    if Q.mres_sd_rg_b2z01 >= 0.457968:
        z += 0.6646428 * Q.mres_sd_rg_b2z01 - 0.3043851
    if Q.jd_sum_abs_sd0_top5 < 9.408315:
        z += 0.1378138 * Q.jd_sum_abs_sd0_top5 - 1.296596
    if Q.mres_sd_mass_b0z02 >= 93.75785:
        z += 0.001313407 * Q.mres_sd_mass_b0z02 - 0.1231422
    if Q.ak02_dr23 >= 0.3194837:
        z += 0.2098633 * Q.ak02_dr23 - 0.0670479
    if Q.n_pairs_kt_above_10 < 7.0:
        z += -0.02937797 * Q.n_pairs_kt_above_10 + 0.2056458
    if Q.z_muon < 0.03898651:
        z += 6.538987 * Q.z_muon - 0.2549323
    if Q.lep_z < 0.221436:
        z += -2.888394 * Q.lep_z + 1.498297
    if 0.221436 <= Q.lep_z < 0.5187302:
        z += -4.870621 * Q.lep_z + 1.937234
    if Q.lep_z >= 0.5187302:
        z += -1.982227 * Q.lep_z + 0.4389364
    if Q.z_charged < 0.8615361:
        z += 3.363306 * Q.z_charged - 2.897609
    if Q.z_displaced5 < 0.2801368 and Q.jd_3d_4 < 172.888:
        z += 0.02313307 * (0.2801368 - Q.z_displaced5) * (172.888 - Q.jd_3d_4)
    if Q.n_s3d_above_3 < 10.0 and Q.n_neutral > 10.0:
        z += 0.0009718027 * (10.0 - Q.n_s3d_above_3) * (Q.n_neutral - 10.0)
    if Q.n_s3d_above_3 < 10.0 and Q.sv_1_n > 1.0:
        z += 0.05400528 * (10.0 - Q.n_s3d_above_3) * (Q.sv_1_n - 1.0)
    if Q.e3_b2 < 0.0004126585 and Q.n_lund > 5.0:
        z += 33.41874 * (0.0004126585 - Q.e3_b2) * (Q.n_lund - 5.0)
    if Q.n_s3d_above_3 < 10.0 and Q.sdb_4_z < 0.1270192:
        z += 0.2092106 * (10.0 - Q.n_s3d_above_3) * (0.1270192 - Q.sdb_4_z)
    if Q.mass_neutral > 34.02385 and Q.jd_3d_5 < 2.912651:
        z += -0.006205989 * (Q.mass_neutral - 34.02385) * (2.912651 - Q.jd_3d_5)
    if Q.sjq_2_prod_k1 < 0.09802954 and Q.ak02_2_n_lep < 1.0:
        z += -0.6316046 * (0.09802954 - Q.sjq_2_prod_k1) * (1.0 - Q.ak02_2_n_lep)
    if Q.sdb_2_n < 16.0 and Q.tau43 < 0.8479222:
        z += -0.04730549 * (16.0 - Q.sdb_2_n) * (0.8479222 - Q.tau43)
    if Q.sjq_2_sumabs_k1 > 0.2432922 and Q.n_lepton < 2.0:
        z += -0.1978928 * (Q.sjq_2_sumabs_k1 - 0.2432922) * (2.0 - Q.n_lepton)
    if Q.sdb_2_n < 16.0 and Q.dc_pair_mass_min < 105.4151:
        z += -8.389274e-05 * (16.0 - Q.sdb_2_n) * (105.4151 - Q.dc_pair_mass_min)
    if Q.sip_3d_2 < 447.0873 and Q.lepsj_2_n_d3 > 2.0:
        z += -0.0002941873 * (447.0873 - Q.sip_3d_2) * (Q.lepsj_2_n_d3 - 2.0)
    if Q.sip_3d_3 < 175.9957 and Q.mres_sd_prong_mass2 > 1.685874e-06:
        z += -4.559359e-05 * (175.9957 - Q.sip_3d_3) * (Q.mres_sd_prong_mass2 - 1.685874e-06)
    if Q.sjq_2_prod_k1 < 0.09802954 and Q.dc_1_z_disp3 < 0.200234:
        z += -2.863028 * (0.09802954 - Q.sjq_2_prod_k1) * (0.200234 - Q.dc_1_z_disp3)
    if Q.z_displaced5 < 0.2801368 and Q.n_lund_kt_above_5 > 1.0:
        z += -0.2913182 * (0.2801368 - Q.z_displaced5) * (Q.n_lund_kt_above_5 - 1.0)
    if Q.e3_b2 < 0.0004126585 and Q.mass_2photon > 22.18431:
        z += 55.85954 * (0.0004126585 - Q.e3_b2) * (Q.mass_2photon - 22.18431)
    if Q.dr_min_012 < 0.01841101 and Q.sjf_4_1_max3d < 3.652671:
        z += 4.212957 * (0.01841101 - Q.dr_min_012) * (3.652671 - Q.sjf_4_1_max3d)
    return z


def neuron_17(Q):
    z = -1.035347e-05
    return z


def neuron_18(Q):
    z = 2.511786
    if Q.lep_z < 0.004135872:
        z += 162.9915 * Q.lep_z - 1.758673
    if 0.004135872 <= Q.lep_z < 0.221436:
        z += 4.991075 * Q.lep_z - 1.105204
    if 100.4835 <= Q.mass < 110.2019:
        z += -0.007083786 * Q.mass + 0.7118033
    if 110.2019 <= Q.mass < 164.4374:
        z += -0.0214399 * Q.mass + 2.293874
    if 164.4374 <= Q.mass < 182.8592:
        z += 0.02562685 * Q.mass - 5.445662
    if Q.mass >= 182.8592:
        z += -0.00378558 * Q.mass - 0.06732773
    if Q.mass_top20 < 114.4658:
        z += 0.00248406 * Q.mass_top20 - 0.2843399
    if Q.lepsj_3_n_d3 < 2.0:
        z += -0.3993098 * Q.lepsj_3_n_d3 + 1.070945
    if 2.0 <= Q.lepsj_3_n_d3 < 3.0:
        z += -0.2723252 * Q.lepsj_3_n_d3 + 0.8169757
    if Q.z_displaced3 < 0.03624058:
        z += 6.943359 * Q.z_displaced3 - 0.2516314
    if Q.z_displaced3 >= 0.2092108:
        z += -1.213625 * Q.z_displaced3 + 0.2539034
    if Q.n_s3d_above_10 < 6.0:
        z += -0.03732689 * Q.n_s3d_above_10 + 0.2239613
    if Q.lep_iso < 1.362094:
        z += -0.1174533 * Q.lep_iso - 0.03172252
    if 1.362094 <= Q.lep_iso < 6.185635:
        z += 0.03974362 * Q.lep_iso - 0.2458395
    if Q.z_displaced5 < 0.08090366:
        z += -10.41139 * Q.z_displaced5 + 0.8423199
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.1742067 * Q.lepsj_3_maxsd0 - 0.3361137
    if 2.413264 <= Q.lepsj_3_maxsd0 < 46.74905:
        z += 0.009807631 * Q.lepsj_3_maxsd0 - 0.7801889
    if 46.74905 <= Q.lepsj_3_maxsd0 < 586.9572:
        z += 0.0005954955 * Q.lepsj_3_maxsd0 - 0.3495303
    if 111.4443 <= Q.mres_sd_mass_b2z01 < 121.6067:
        z += 0.001934724 * Q.mres_sd_mass_b2z01 - 0.2156139
    if 121.6067 <= Q.mres_sd_mass_b2z01 < 177.5398:
        z += -0.005258754 * Q.mres_sd_mass_b2z01 + 0.6591613
    if Q.mres_sd_mass_b2z01 >= 177.5398:
        z += 0.01671649 * Q.mres_sd_mass_b2z01 - 3.242319
    if Q.n_s3d_above_3 < 3.0:
        z += 0.1257069 * Q.n_s3d_above_3 - 0.7306407
    if 3.0 <= Q.n_s3d_above_3 < 10.0:
        z += 0.05050284 * Q.n_s3d_above_3 - 0.5050284
    if Q.ak02_1_z < 0.849996:
        z += 0.6356037 * Q.ak02_1_z - 0.5402606
    if Q.sj3_pair_mass_max >= 128.6605:
        z += -0.01534457 * Q.sj3_pair_mass_max + 1.974241
    if Q.n_dr_0p4_up < 4.0:
        z += -0.03604184 * Q.n_dr_0p4_up + 0.1441674
    if Q.lam1 < 0.01299032:
        z += 64.32186 * Q.lam1 - 0.8355617
    if Q.n_real_top50 >= 26.0:
        z += -0.014948 * Q.n_real_top50 + 0.3886479
    if Q.M2 >= 0.07013948:
        z += 3.38281 * Q.M2 - 0.2372685
    if Q.sv_1_sd0_sum < 51.56287:
        z += -0.003783468 * Q.sv_1_sd0_sum + 0.1950865
    if Q.sjf_3_1_max3d < 2.078395:
        z += -0.08600457 * Q.sjf_3_1_max3d - 0.2572415
    if 2.078395 <= Q.sjf_3_1_max3d < 2.924357:
        z += 0.1127697 * Q.sjf_3_1_max3d - 0.670373
    if 2.924357 <= Q.sjf_3_1_max3d < 3.710567:
        z += 0.7456673 * Q.sjf_3_1_max3d - 2.521191
    if 3.710567 <= Q.sjf_3_1_max3d < 71.08287:
        z += -0.003646266 * Q.sjf_3_1_max3d + 0.259187
    if Q.sjf_2_2_max3d < 4.516968:
        z += 0.1498748 * Q.sjf_2_2_max3d - 0.3746075
    if 4.516968 <= Q.sjf_2_2_max3d < 55.92931:
        z += -0.005881313 * Q.sjf_2_2_max3d + 0.3289378
    if Q.sjf_2_1_z_d3 < 0.0828821:
        z += 3.655257 * Q.sjf_2_1_z_d3 - 0.3029554
    if Q.sjq_2_1_nch < 16.0:
        z += -0.03015925 * Q.sjq_2_1_nch + 0.482548
    if Q.sj2_mass2 < 6.465318:
        z += -0.07726532 * Q.sj2_mass2 + 0.4995449
    if Q.n_sdz_above_5 < 3.0:
        z += -0.08382449 * Q.n_sdz_above_5 + 0.2514735
    if Q.z_charged_had < 0.3603262:
        z += -1.523456 * Q.z_charged_had + 0.5489411
    if Q.n_for_90pct < 29.0:
        z += 0.03199713 * Q.n_for_90pct - 0.9279168
    if Q.tau5 < 0.03663959:
        z += -19.22162 * Q.tau5 + 0.7042722
    if Q.sdb_jp_top3 < 813.043:
        z += 0.0001788215 * Q.sdb_jp_top3 - 0.1453896
    if Q.psi_0p1 < 0.624781:
        z += -0.3222961 * Q.psi_0p1 + 0.2013644
    if Q.N2 >= 0.4137858:
        z += -5.388153 * Q.N2 + 2.229541
    if Q.ak02_dr23 >= 0.3194837:
        z += -0.2955885 * Q.ak02_dr23 + 0.0944357
    if Q.sjf_4_1_z_d3 >= 0.181327:
        z += 1.166237 * Q.sjf_4_1_z_d3 - 0.2114702
    if Q.n_dr_0p1_0p2 < 17.0:
        z += -0.008474986 * Q.n_dr_0p1_0p2 + 0.1440748
    if Q.mass_top40 < 126.8853:
        z += 0.008499053 * Q.mass_top40 - 1.078405
    if Q.sip_3d_3 < 4.606241:
        z += 0.1033693 * Q.sip_3d_3 - 0.4761439
    if Q.e3_b2 < 0.0002536827:
        z += -126.6597 * Q.e3_b2 - 0.2777593
    if 0.0002536827 <= Q.e3_b2 < 0.0007909605:
        z += 576.7792 * Q.e3_b2 - 0.4562096
    if Q.e3_b05 >= 0.008870191:
        z += -17.07998 * Q.e3_b05 + 0.1515027
    if Q.max_abs_dz < 17.21875:
        z += 0.007956946 * Q.max_abs_dz - 0.1370087
    if Q.sjf_2_1_max3d < 55.67805:
        z += -0.00240005 * Q.sjf_2_1_max3d + 0.1336301
    if Q.sdb_0_z < 0.05934469:
        z += 2.417113 * Q.sdb_0_z - 0.1434428
    if Q.sjf_3_3_max3d < 1.348343:
        z += -0.113101 * Q.sjf_3_3_max3d + 0.1524989
    if Q.sjq_3_2_nch < 3.0:
        z += -0.08493742 * Q.sjq_3_2_nch + 0.2548123
    if Q.sj3_pairmin_over_m < 0.2072106:
        z += 1.314911 * Q.sj3_pairmin_over_m - 0.2724636
    if Q.pz_lnd0 < 0.1032185:
        z += -1.518973 * Q.pz_lnd0 + 0.1567861
    if Q.mass_displaced3 < 2.409613:
        z += -0.1376599 * Q.mass_displaced3 + 0.331707
    if Q.sjf_3_1_n_d3 < 1.0:
        z += -0.299681 * Q.sjf_3_1_n_d3 + 0.299681
    if Q.lep_z < 0.221436 and Q.N2 < 0.3829918:
        z += 23.41462 * (0.221436 - Q.lep_z) * (0.3829918 - Q.N2)
    if Q.lep_z < 0.221436 and Q.sdb_2_z > 0.2509165:
        z += -6.120071 * (0.221436 - Q.lep_z) * (Q.sdb_2_z - 0.2509165)
    if Q.lepsj_3_n_d3 < 3.0 and Q.lep_dr < 0.08178299:
        z += -0.7298912 * (3.0 - Q.lepsj_3_n_d3) * (0.08178299 - Q.lep_dr)
    if Q.lep_z < 0.221436 and Q.z_displaced5 < 0.08090366:
        z += -54.7277 * (0.221436 - Q.lep_z) * (0.08090366 - Q.z_displaced5)
    if Q.lep_z < 0.221436 and Q.lne_4 > 3.751708:
        z += 1.012319 * (0.221436 - Q.lep_z) * (Q.lne_4 - 3.751708)
    if Q.mass_displaced3 < 9.090532 and Q.sjq_2_2_nch > 4.0:
        z += -0.003485241 * (9.090532 - Q.mass_displaced3) * (Q.sjq_2_2_nch - 4.0)
    if Q.sv_1_sd0_sum < 51.56287 and Q.dc_1_n_lep > 0.0:
        z += -0.002718207 * (51.56287 - Q.sv_1_sd0_sum) * (Q.dc_1_n_lep - 0.0)
    if Q.sjq_2_1_nch < 16.0 and Q.sum_e < 1651.766:
        z += -2.795334e-05 * (16.0 - Q.sjq_2_1_nch) * (1651.766 - Q.sum_e)
    if Q.n_real_top50 > 26.0 and Q.n_lund_kt_above_5 < 3.0:
        z += -0.005628644 * (Q.n_real_top50 - 26.0) * (3.0 - Q.n_lund_kt_above_5)
    if Q.lep_z < 0.221436 and Q.jet_charge_k05 > -0.2841306:
        z += -0.6286789 * (0.221436 - Q.lep_z) * (Q.jet_charge_k05 - -0.2841306)
    return z


def neuron_19(Q):
    z = -1.40782e-06
    return z


def neuron_20(Q):
    z = 8.253686e-07
    return z


def neuron_21(Q):
    z = -0.001682464
    return z


def neuron_22(Q):
    z = -1.044846e-06
    return z


def neuron_23(Q):
    z = -6.978799e-07
    return z


def neuron_24(Q):
    z = -0.4868901
    if Q.lep_z < 0.004135872:
        z += 59.14481 * Q.lep_z - 1.899091
    if 0.004135872 <= Q.lep_z < 0.3396572:
        z += 4.93106 * Q.lep_z - 1.67487
    if Q.mass_top30 < 74.51927:
        z += 0.00515628 * Q.mass_top30 + 0.1859544
    if 74.51927 <= Q.mass_top30 < 134.2224:
        z += -0.009550534 * Q.mass_top30 + 1.281895
    z += 0.1977999 * Q.pair_max_lnkt
    if Q.e3_b2 < 7.452974e-05:
        z += 3183.433 * Q.e3_b2 - 0.5286124
    if 7.452974e-05 <= Q.e3_b2 < 0.0004126585:
        z += 861.6595 * Q.e3_b2 - 0.3555712
    if Q.mass < 90.08945:
        z += 0.004229707 * Q.mass - 0.8555424
    if 90.08945 <= Q.mass < 110.2019:
        z += -0.007267297 * Q.mass + 0.1802164
    if 110.2019 <= Q.mass < 164.4374:
        z += 0.01144366 * Q.mass - 1.881767
    if Q.lep_ptrel < 12.15228:
        z += 0.01161069 * Q.lep_ptrel + 0.08299211
    if 12.15228 <= Q.lep_ptrel < 43.20788:
        z += -0.007215718 * Q.lep_ptrel + 0.3117759
    if Q.mres_sd_mass_b2z01 < 55.13212:
        z += -0.001843685 * Q.mres_sd_mass_b2z01 + 0.1319318
    if 55.13212 <= Q.mres_sd_mass_b2z01 < 81.19466:
        z += 0.02794767 * Q.mres_sd_mass_b2z01 - 1.510529
    if 81.19466 <= Q.mres_sd_mass_b2z01 < 118.2746:
        z += -0.02046044 * Q.mres_sd_mass_b2z01 + 2.419951
    if Q.n_s3d_above_10 >= 1.0:
        z += 0.04757052 * Q.n_s3d_above_10 - 0.04757052
    if Q.lepsj_2_dr < 0.103005:
        z += 2.348934 * Q.lepsj_2_dr - 0.2419519
    if Q.mass_displaced3 < 18.80005:
        z += -0.02504117 * Q.mass_displaced3 + 0.4707753
    if Q.n_s3d_above_3 < 1.0:
        z += -0.3629069 * Q.n_s3d_above_3 + 0.4763524
    if 1.0 <= Q.n_s3d_above_3 < 3.0:
        z += -0.05672274 * Q.n_s3d_above_3 + 0.1701682
    if Q.n_pairs_kt_above_1 < 366.0:
        z += -0.001643814 * Q.n_pairs_kt_above_1 + 0.6016358
    if Q.N2 < 0.2423129:
        z += 3.167711 * Q.N2 - 0.7675774
    if Q.lep_iso < 0.4381892:
        z += -2.706417 * Q.lep_iso + 1.310561
    if 0.4381892 <= Q.lep_iso < 1.362094:
        z += -0.1349036 * Q.lep_iso + 0.1837514
    if Q.jd_3d_4 < 2.953464:
        z += -0.09279909 * Q.jd_3d_4 + 0.2740788
    if Q.sjq_3_2_k1 < -0.6014774:
        z += -2.624877 * Q.sjq_3_2_k1 - 1.578804
    if Q.sjq_2_2_k1 >= 0.277232:
        z += 1.034133 * Q.sjq_2_2_k1 - 0.2866948
    if Q.jd_sum_abs_sd0_top5 < 84.11398:
        z += -0.006542188 * Q.jd_sum_abs_sd0_top5 + 0.5502895
    if Q.z_neutral_had < 0.212136:
        z += -1.58631 * Q.z_neutral_had + 0.3365135
    if Q.sip_3d_1 < 7.028704:
        z += -0.04219696 * Q.sip_3d_1 + 0.2965899
    if Q.lam1 < 0.006142967:
        z += 188.2722 * Q.lam1 - 1.15655
    if Q.z_dr_0p2_0p4 < 0.1468781:
        z += 3.000778 * Q.z_dr_0p2_0p4 - 0.4407487
    if Q.sj3_pair_mass_max < 83.63398:
        z += 0.01016142 * Q.sj3_pair_mass_max - 0.7535952
    if 83.63398 <= Q.sj3_pair_mass_max < 128.6605:
        z += -0.002137507 * Q.sj3_pair_mass_max + 0.2750128
    z += -0.4783799 * Q.sjq_2_prod_k1
    if Q.tau21 >= 0.135772:
        z += 1.151224 * Q.tau21 - 0.1563039
    if Q.tau32 < 0.456416:
        z += 5.119262 * Q.tau32 - 2.336513
    if Q.n_lund < 5.0:
        z += -0.2692085 * Q.n_lund + 1.346042
    if Q.dc_2_jp < 6.161303:
        z += -0.01552742 * Q.dc_2_jp + 0.09566917
    if Q.mres_pruned_mass < 60.18017:
        z += -0.006671031 * Q.mres_pruned_mass - 0.02591466
    if 60.18017 <= Q.mres_pruned_mass < 86.14266:
        z += 0.03483314 * Q.mres_pruned_mass - 2.523642
    if 86.14266 <= Q.mres_pruned_mass < 119.8459:
        z += -0.0260734 * Q.mres_pruned_mass + 2.723009
    if 119.8459 <= Q.mres_pruned_mass < 171.3819:
        z += 0.007796127 * Q.mres_pruned_mass - 1.336115
    if 0.2293319 <= Q.dc_split2_dr < 0.3591078:
        z += 1.692356 * Q.dc_split2_dr - 0.3881113
    if Q.dc_split2_dr >= 0.3591078:
        z += -3.778501 * Q.dc_split2_dr + 1.576516
    if Q.n_dr_0p4_up < 4.0:
        z += -0.06983348 * Q.n_dr_0p4_up + 0.2793339
    if Q.max_abs_d0 < 10.52344:
        z += -0.0287057 * Q.max_abs_d0 + 0.3020827
    if Q.z_charged_had >= 0.5398733:
        z += -0.9436171 * Q.z_charged_had + 0.5094336
    if Q.n_dr_0p2_0p4 < 17.0:
        z += -0.03965385 * Q.n_dr_0p2_0p4 + 0.6741154
    if Q.lep_z < 0.3396572 and Q.mass < 127.2317:
        z += 0.05909448 * (0.3396572 - Q.lep_z) * (127.2317 - Q.mass)
    if Q.lep_z < 0.3396572 and Q.tau32 < 0.456416:
        z += 22.0471 * (0.3396572 - Q.lep_z) * (0.456416 - Q.tau32)
    if Q.lep_ptrel < 43.20788 and Q.sdb_2_z > 0.2968888:
        z += 0.02245034 * (43.20788 - Q.lep_ptrel) * (Q.sdb_2_z - 0.2968888)
    if Q.e3_b2 < 0.0004126585 and Q.lne_8 > 2.11682:
        z += 216.6214 * (0.0004126585 - Q.e3_b2) * (Q.lne_8 - 2.11682)
    if Q.z_displaced3 < 0.1061578 and Q.pz_lnd2 < 0.1166862:
        z += -15.81936 * (0.1061578 - Q.z_displaced3) * (0.1166862 - Q.pz_lnd2)
    if Q.z_displaced3 < 0.1061578 and Q.D2_b2 < 15.17086:
        z += 0.2081816 * (0.1061578 - Q.z_displaced3) * (15.17086 - Q.D2_b2)
    if Q.mass < 164.4374 and Q.sv_1_n < 3.0:
        z += -0.001747485 * (164.4374 - Q.mass) * (3.0 - Q.sv_1_n)
    if Q.e3_b2 < 0.0004126585 and Q.sdb_0_n < 5.0:
        z += -129.7146 * (0.0004126585 - Q.e3_b2) * (5.0 - Q.sdb_0_n)
    if Q.mass < 164.4374 and Q.M3_b2 < 0.0309457:
        z += -0.1292737 * (164.4374 - Q.mass) * (0.0309457 - Q.M3_b2)
    if Q.lepsj_2_dr < 0.103005 and Q.jet_e > 926.0771:
        z += 0.001744871 * (0.103005 - Q.lepsj_2_dr) * (Q.jet_e - 926.0771)
    if Q.mass < 164.4374 and Q.n_dr_0p2_0p4 < 24.0:
        z += 0.0001245524 * (164.4374 - Q.mass) * (24.0 - Q.n_dr_0p2_0p4)
    if Q.jd_sum_abs_sd0_top5 < 84.11398 and Q.jd_3d_5 < 6.771002:
        z += -0.002075046 * (84.11398 - Q.jd_sum_abs_sd0_top5) * (6.771002 - Q.jd_3d_5)
    if Q.mass_displaced3 < 18.80005 and Q.jd_3d_5 < 4.004982:
        z += 0.008775926 * (18.80005 - Q.mass_displaced3) * (4.004982 - Q.jd_3d_5)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lep_iso < 0.4381892:
        z += -0.08377076 * (83.63398 - Q.sj3_pair_mass_max) * (0.4381892 - Q.lep_iso)
    if Q.lep_z < 0.3396572 and Q.tau21 > 0.1757731:
        z += -3.429851 * (0.3396572 - Q.lep_z) * (Q.tau21 - 0.1757731)
    if Q.lep_z < 0.3396572 and Q.n_neutral_had < 7.0:
        z += -0.1434335 * (0.3396572 - Q.lep_z) * (7.0 - Q.n_neutral_had)
    if Q.z_displaced3 < 0.1061578 and Q.mass_2photon > 3.013:
        z += 0.09915987 * (0.1061578 - Q.z_displaced3) * (Q.mass_2photon - 3.013)
    if Q.lep_z < 0.004135872 and Q.sdb_4_z < 0.01435877:
        z += 4100.99 * (0.004135872 - Q.lep_z) * (0.01435877 - Q.sdb_4_z)
    if Q.lep_z < 0.3396572 and Q.e4 > 4.06096e-07:
        z += 25495.63 * (0.3396572 - Q.lep_z) * (Q.e4 - 4.06096e-07)
    if Q.lep_z < 0.004135872 and Q.n_pairs_kt_above_10 > 4.0:
        z += -1.679852 * (0.004135872 - Q.lep_z) * (Q.n_pairs_kt_above_10 - 4.0)
    if Q.sjq_3_2_k1 < -0.6014774 and Q.lep_iso < 0.4381892:
        z += -5.722001 * (-0.6014774 - Q.sjq_3_2_k1) * (0.4381892 - Q.lep_iso)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lep_iso < 14.38858:
        z += 0.001339018 * (83.63398 - Q.sj3_pair_mass_max) * (14.38858 - Q.lep_iso)
    if Q.lepsj_2_dr < 0.103005 and Q.lep_iso < 14.38858:
        z += -0.7681552 * (0.103005 - Q.lepsj_2_dr) * (14.38858 - Q.lep_iso)
    if Q.sjq_2_2_k1 > 0.277232 and Q.lep_iso < 2.907433:
        z += -0.2326815 * (Q.sjq_2_2_k1 - 0.277232) * (2.907433 - Q.lep_iso)
    if Q.sj3_pair_mass_max < 83.63398 and Q.lepsj_2_dr < 0.1822601:
        z += 0.1191918 * (83.63398 - Q.sj3_pair_mass_max) * (0.1822601 - Q.lepsj_2_dr)
    if Q.lep_z < 0.004135872 and Q.lep_iso < 2.907433:
        z += -86.98977 * (0.004135872 - Q.lep_z) * (2.907433 - Q.lep_iso)
    if Q.lep_z < 0.3396572 and Q.lepsj_3_maxsd0 < 0.8036986:
        z += 3.335006 * (0.3396572 - Q.lep_z) * (0.8036986 - Q.lepsj_3_maxsd0)
    if Q.lep_z < 0.3396572 and Q.sj3_dr12 < 0.3683248:
        z += -1.588182 * (0.3396572 - Q.lep_z) * (0.3683248 - Q.sj3_dr12)
    if Q.mass_displaced3 < 18.80005 and Q.sip_3d_2 < 2325.187:
        z += -1.021033e-05 * (18.80005 - Q.mass_displaced3) * (2325.187 - Q.sip_3d_2)
    if Q.max_abs_d0 < 10.52344 and Q.jd_3d_5 < 4.004982:
        z += -0.01202584 * (10.52344 - Q.max_abs_d0) * (4.004982 - Q.jd_3d_5)
    if Q.mass_top15 > 70.93762 and Q.sv_2_z > 0.00726873:
        z += -0.1379641 * (Q.mass_top15 - 70.93762) * (Q.sv_2_z - 0.00726873)
    if Q.lep_iso < 0.4381892 and Q.sdb_4_n > 0.0:
        z += 0.274617 * (0.4381892 - Q.lep_iso) * (Q.sdb_4_n - 0.0)
    if Q.lep_iso < 0.4381892 and Q.lepsj_2_dr < 0.1822601:
        z += 6.403238 * (0.4381892 - Q.lep_iso) * (0.1822601 - Q.lepsj_2_dr)
    return z


def neuron_25(Q):
    z = 0.03136871
    return z


def neuron_26(Q):
    z = -3.231261e-06
    return z


def neuron_27(Q):
    z = -6.308501e-06
    return z


def neuron_28(Q):
    z = 9.617208e-06
    return z


def neuron_29(Q):
    z = 9.183865e-06
    return z


def neuron_30(Q):
    z = -8.603055e-06
    return z


def neuron_31(Q):
    z = 3.509523e-06
    return z


def neuron_32(Q):
    z = 1.441348e-05
    return z


def neuron_33(Q):
    z = 2.567326e-06
    return z


def neuron_34(Q):
    z = 1.388486e-05
    return z


def neuron_35(Q):
    z = -6.743301e-06
    return z


def neuron_36(Q):
    z = -4.831948e-06
    return z


def neuron_37(Q):
    z = -4.357436e-06
    return z


def neuron_38(Q):
    z = -9.411176e-06
    return z


def neuron_39(Q):
    z = 8.274277e-06
    return z


def neuron_40(Q):
    z = 7.77554e-07
    return z


def neuron_41(Q):
    z = 3.814691e-08
    return z


def neuron_42(Q):
    z = 8.779052e-06
    return z


def neuron_43(Q):
    z = -9.407709e-06
    return z


def neuron_44(Q):
    z = -3.324349e-06
    return z


def neuron_45(Q):
    z = 6.849337e-06
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
    z = 5.175116e-06
    return z


def neuron_51(Q):
    z = -2.012202e-06
    return z


def neuron_52(Q):
    z = -1.069897
    if Q.lep_z < 0.221436:
        z += 1.575383 * Q.lep_z - 0.2271869
    if 0.221436 <= Q.lep_z < 0.3396572:
        z += -1.029085 * Q.lep_z + 0.3495362
    if Q.z_displaced3 < 0.1342762:
        z += -6.086347 * Q.z_displaced3 + 0.8172515
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.1708225 * Q.lepsj_3_maxsd0 + 0.41224
    if Q.mass_top50 < 79.27954:
        z += -0.01521152 * Q.mass_top50 + 1.071841
    if 79.27954 <= Q.mass_top50 < 115.614:
        z += 0.003691302 * Q.mass_top50 - 0.4267662
    if Q.mres_sd_mass_b0z005 < 81.62059:
        z += -0.004312642 * Q.mres_sd_mass_b0z005 + 1.119945
    if 81.62059 <= Q.mres_sd_mass_b0z005 < 131.0776:
        z += -0.0009853386 * Q.mres_sd_mass_b0z005 + 0.8483682
    if 131.0776 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += -0.01590035 * Q.mres_sd_mass_b0z005 + 2.803391
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += 0.003327303 * Q.mres_sd_mass_b0z005 - 0.2715765
    if Q.mass >= 182.8592:
        z += -0.005527351 * Q.mass + 1.010727
    if Q.lne_16 >= 0.4468807:
        z += -0.1370061 * Q.lne_16 + 0.06122539
    if Q.lep_ptrel < 12.15228:
        z += 0.0384937 * Q.lep_ptrel - 0.6185113
    if 12.15228 <= Q.lep_ptrel < 18.7678:
        z += 0.009934868 * Q.lep_ptrel - 0.2714564
    if 18.7678 <= Q.lep_ptrel < 27.3236:
        z += 0.01715977 * Q.lep_ptrel - 0.4070519
    if Q.lep_ptrel >= 27.3236:
        z += 0.007224899 * Q.lep_ptrel - 0.1355955
    if Q.lepsj_2_dr < 0.103005:
        z += 5.227883 * Q.lepsj_2_dr - 0.538498
    if Q.sv_1_sd0_sum < 125.2765:
        z += 0.001485874 * Q.sv_1_sd0_sum - 0.1861451
    if Q.sjq_2_2_k1 < -0.6337755:
        z += -2.220338 * Q.sjq_2_2_k1 - 1.407196
    if Q.sjq_2_2_k1 >= 0.6477929:
        z += 2.14395 * Q.sjq_2_2_k1 - 1.388835
    if Q.pz_lnd0 < 0.1122946:
        z += -2.859535 * Q.pz_lnd0 + 0.3211102
    if Q.n_s3d_above_3 < 3.0:
        z += -0.1450965 * Q.n_s3d_above_3 + 0.4352894
    if Q.lep_dr < 0.3014662:
        z += -2.660739 * Q.lep_dr + 0.8021229
    if Q.lep_dr >= 0.4191372:
        z += -1.105322 * Q.lep_dr + 0.4632815
    if Q.n_lund < 7.0:
        z += -0.166168 * Q.n_lund + 1.163176
    if Q.lep_iso < 1.362094:
        z += -0.3274609 * Q.lep_iso + 0.4460327
    if Q.n_lepton < 1.0:
        z += 1.143421 * Q.n_lepton - 1.143421
    if Q.sj2_mass2 < 21.7295:
        z += 0.006191572 * Q.sj2_mass2 - 0.1345397
    if Q.dc_1_n_lep < 1.0:
        z += -0.2813231 * Q.dc_1_n_lep + 0.2813231
    if Q.jd_sum_abs_sd0_top5 < 84.11398:
        z += 0.004153473 * Q.jd_sum_abs_sd0_top5 - 0.3493652
    if Q.sj3_pair_mass_min >= 80.02563:
        z += 0.01837988 * Q.sj3_pair_mass_min - 1.470861
    if Q.jet_charge_k03 >= -0.2409073:
        z += -0.1125293 * Q.jet_charge_k03 - 0.02710914
    if Q.ak02_2_z < 0.2837384:
        z += -2.163689 * Q.ak02_2_z + 0.6139217
    if Q.sjf_2_2_max3d < 3.029824:
        z += -0.1100433 * Q.sjf_2_2_max3d + 0.3334117
    if Q.n_dr_0p2_0p4 < 21.0:
        z += 0.01731146 * Q.n_dr_0p2_0p4 - 0.3635406
    if Q.sj2_dr < 0.3740528:
        z += -0.8184416 * Q.sj2_dr + 0.3061404
    if Q.mass_2charged >= 14.28253:
        z += -0.005333097 * Q.mass_2charged + 0.07617011
    if Q.N3_b05 >= 0.7591346:
        z += -0.08542708 * Q.N3_b05 + 0.06485066
    if Q.sdb_2_n >= 9.0:
        z += -0.03235198 * Q.sdb_2_n + 0.2911678
    if Q.sj3_mass1 >= 36.36236:
        z += -0.01093463 * Q.sj3_mass1 + 0.3976091
    if Q.D2 < 1.421532:
        z += 0.289228 * Q.D2 - 0.4111469
    if Q.tau21_b2 < 0.3565533:
        z += -2.533141 * Q.tau21_b2 + 0.9031998
    if Q.ak02_1_z >= 0.6342743:
        z += -0.9382293 * Q.ak02_1_z + 0.5950948
    if Q.e3_b2 < 0.0004126585:
        z += -459.954 * Q.e3_b2 + 0.1898039
    if Q.n_dr_0p4_up < 10.0:
        z += 0.01355054 * Q.n_dr_0p4_up - 0.1355054
    if Q.lund_max_lndelta >= -0.615738:
        z += 0.6746479 * Q.lund_max_lndelta + 0.4154064
    if Q.N2_b2 >= 0.08291719:
        z += 1.509725 * Q.N2_b2 - 0.1251822
    if Q.z_displaced3 < 0.1342762 and Q.N2 > 0.1824346:
        z += 8.803055 * (0.1342762 - Q.z_displaced3) * (Q.N2 - 0.1824346)
    if Q.lep_z < 0.3396572 and Q.n_charged_pt_above_1 > 11.0:
        z += 0.09774636 * (0.3396572 - Q.lep_z) * (Q.n_charged_pt_above_1 - 11.0)
    if Q.lep_z < 0.3396572 and Q.sj3_pair_mass_min > 48.40547:
        z += -0.02302565 * (0.3396572 - Q.lep_z) * (Q.sj3_pair_mass_min - 48.40547)
    if Q.mres_sd_mass_b0z005 < 159.9242 and Q.tau21_b2 < 0.3565533:
        z += -0.03266819 * (159.9242 - Q.mres_sd_mass_b0z005) * (0.3565533 - Q.tau21_b2)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.tau32 < 0.7544983:
        z += -0.006091063 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.7544983 - Q.tau32)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.sjf_2_2_z_d3 < 0.03385157:
        z += 0.06246657 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.03385157 - Q.sjf_2_2_z_d3)
    if Q.mres_sd_mass_b0z005 < 159.9242 and Q.mass_displaced3 > 3.208089:
        z += -9.214094e-05 * (159.9242 - Q.mres_sd_mass_b0z005) * (Q.mass_displaced3 - 3.208089)
    if Q.z_neutral_had < 0.3788785 and Q.ecf_g41 > 6.216954e-05:
        z += 4306.114 * (0.3788785 - Q.z_neutral_had) * (Q.ecf_g41 - 6.216954e-05)
    if Q.mres_sd_mass_b0z005 > 81.62059 and Q.ak02_3_z < 0.07650476:
        z += 0.1008603 * (Q.mres_sd_mass_b0z005 - 81.62059) * (0.07650476 - Q.ak02_3_z)
    if Q.lepsj_2_dr < 0.103005 and Q.C3_b05 < 0.2250047:
        z += 10.93207 * (0.103005 - Q.lepsj_2_dr) * (0.2250047 - Q.C3_b05)
    if Q.lep_dr < 0.3014662 and Q.sdb_2_z > 0.1530389:
        z += 2.084991 * (0.3014662 - Q.lep_dr) * (Q.sdb_2_z - 0.1530389)
    if Q.z_displaced3 < 0.1342762 and Q.sip_3d_3 < 577.991:
        z += -0.007385492 * (0.1342762 - Q.z_displaced3) * (577.991 - Q.sip_3d_3)
    if Q.lund3_lndelta > -2.817283 and Q.sv_1_dr < 0.3689526:
        z += 0.2310666 * (Q.lund3_lndelta - -2.817283) * (0.3689526 - Q.sv_1_dr)
    if Q.z_neutral_had < 0.3788785 and Q.pz_lnkt1 > 0.03633353:
        z += -7.384605 * (0.3788785 - Q.z_neutral_had) * (Q.pz_lnkt1 - 0.03633353)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_2_prod_k05 > -0.5057096:
        z += -0.01307186 * (21.0 - Q.n_dr_0p2_0p4) * (Q.sjq_2_prod_k05 - -0.5057096)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_3_sumabs_k1 > 0.3376898:
        z += 0.0169487 * (21.0 - Q.n_dr_0p2_0p4) * (Q.sjq_3_sumabs_k1 - 0.3376898)
    if Q.z_displaced3 < 0.1342762 and Q.n_lund_kt_above_1 > 3.0:
        z += 0.2130979 * (0.1342762 - Q.z_displaced3) * (Q.n_lund_kt_above_1 - 3.0)
    if Q.lepsj_2_dr < 0.103005 and Q.z_photon > 0.3318968:
        z += 8.312802 * (0.103005 - Q.lepsj_2_dr) * (Q.z_photon - 0.3318968)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.sjq_3_prod_k1 < 0.1145956:
        z += 0.035343 * (21.0 - Q.n_dr_0p2_0p4) * (0.1145956 - Q.sjq_3_prod_k1)
    if Q.z_displaced3 < 0.1342762 and Q.jd_3d_6 < 5.367501:
        z += 1.456462 * (0.1342762 - Q.z_displaced3) * (5.367501 - Q.jd_3d_6)
    if Q.n_lund < 7.0 and Q.jd_3d_6 < 2.591685:
        z += -0.03942204 * (7.0 - Q.n_lund) * (2.591685 - Q.jd_3d_6)
    if Q.D3_b2 < 0.001184159 and Q.jd_3d_6 < 5.367501:
        z += -47.5293 * (0.001184159 - Q.D3_b2) * (5.367501 - Q.jd_3d_6)
    if Q.sdb_2_n > 9.0 and Q.D2_b2 < 2.01745:
        z += 0.03138686 * (Q.sdb_2_n - 9.0) * (2.01745 - Q.D2_b2)
    if Q.z_displaced3 < 0.1342762 and Q.jd_3d_6 < 30.57088:
        z += -0.2090741 * (0.1342762 - Q.z_displaced3) * (30.57088 - Q.jd_3d_6)
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
    z = -7.072331e-07
    return z


def neuron_57(Q):
    z = -1.959648e-06
    return z


def neuron_58(Q):
    z = -3.617151e-06
    return z


def neuron_59(Q):
    z = 6.029332e-06
    return z


def neuron_60(Q):
    z = -4.666766e-06
    return z


def neuron_61(Q):
    z = 5.840559e-06
    return z


def neuron_62(Q):
    z = 9.422976e-07
    return z


def neuron_63(Q):
    z = -8.308512e-06
    return z


def neuron_64(Q):
    z = 1.258629e-06
    return z


def neuron_65(Q):
    z = 5.791054e-06
    return z


def neuron_66(Q):
    z = 2.709779e-06
    return z


def neuron_67(Q):
    z = -5.219511e-06
    return z


def neuron_68(Q):
    z = 4.483712
    if Q.tau32 < 0.7981752:
        z += -0.6540307 * Q.tau32 + 0.5220311
    if 6.983043 <= Q.lep_ptrel < 18.7678:
        z += -0.03500523 * Q.lep_ptrel + 0.244443
    if 18.7678 <= Q.lep_ptrel < 43.20788:
        z += -0.005515613 * Q.lep_ptrel - 0.3090122
    if Q.lep_ptrel >= 43.20788:
        z += -0.02037334 * Q.lep_ptrel + 0.3329589
    if Q.pair_mean_lnm2 < 3.38061:
        z += 0.1036863 * Q.pair_mean_lnm2 - 0.350523
    if Q.pz_lnd2 < 0.1339824:
        z += 2.882288 * Q.pz_lnd2 - 0.3861757
    if Q.e3_b2 < 0.0004126585:
        z += 1779.422 * Q.e3_b2 - 0.7342936
    if Q.sj3_pair_mass_max < 56.73313:
        z += 0.02498294 * Q.sj3_pair_mass_max - 1.417361
    if 83.63398 <= Q.sj3_pair_mass_max < 128.6605:
        z += -0.009374095 * Q.sj3_pair_mass_max + 0.7839929
    if Q.sj3_pair_mass_max >= 128.6605:
        z += 0.01359081 * Q.sj3_pair_mass_max - 2.170684
    if Q.mass < 90.08945:
        z += 0.008790338 * Q.mass - 2.215728
    if 90.08945 <= Q.mass < 110.2019:
        z += -0.002129436 * Q.mass - 1.231972
    if 110.2019 <= Q.mass < 164.4374:
        z += 0.02704203 * Q.mass - 4.446722
    if Q.z_charged_had >= 0.265564:
        z += 0.8651246 * Q.z_charged_had - 0.229746
    if Q.M2_b2 < 0.05966366:
        z += 9.167994 * Q.M2_b2 - 0.5469961
    if Q.mres_sd_mass_b2z01 < 55.13212:
        z += -0.007485028 * Q.mres_sd_mass_b2z01 - 0.8268243
    if 55.13212 <= Q.mres_sd_mass_b2z01 < 76.1529:
        z += 0.04996888 * Q.mres_sd_mass_b2z01 - 3.99438
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 118.2746:
        z += -0.0217554 * Q.mres_sd_mass_b2z01 + 1.467632
    if 118.2746 <= Q.mres_sd_mass_b2z01 < 130.0968:
        z += 0.02768265 * Q.mres_sd_mass_b2z01 - 4.379636
    if 130.0968 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.02768938 * Q.mres_sd_mass_b2z01 - 4.380511
    z += -0.06551369 * Q.n_particles
    if Q.dc_split2_dr >= 0.3591078:
        z += -2.527625 * Q.dc_split2_dr + 0.9076899
    if Q.lep_iso < 0.4381892:
        z += 1.411853 * Q.lep_iso - 0.6186588
    if Q.lep_dr < 0.04607888:
        z += -6.980276 * Q.lep_dr + 0.3216433
    if Q.sj3_pair_mass_min < 80.02563:
        z += 0.008248717 * Q.sj3_pair_mass_min - 0.6601087
    if Q.sdb_2_n >= 12.0:
        z += -0.01816349 * Q.sdb_2_n + 0.2179618
    if Q.M3 < 0.03259227:
        z += -8.171691 * Q.M3 + 0.266334
    if Q.mass_top20 >= 130.7093:
        z += -0.01776351 * Q.mass_top20 + 2.321855
    if Q.jet_abs_eta >= 0.5325716:
        z += 0.3639798 * Q.jet_abs_eta - 0.1938453
    z += 0.04031186 * Q.n_photon
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.008780359 * Q.n_pairs_kt_above_1 - 0.7024287
    if Q.sjq_2_2_nch < 3.0:
        z += 0.1158161 * Q.sjq_2_2_nch - 0.3474482
    if Q.sj2_dr < 0.2403736:
        z += 1.323881 * Q.sj2_dr - 0.3182261
    if Q.tau5 < 0.05613495:
        z += -16.47897 * Q.tau5 + 0.9250459
    if Q.ktd_ln_d34 >= -9.359695:
        z += 0.1838056 * Q.ktd_ln_d34 + 1.720364
    if Q.D2_b2 < 1.774856:
        z += 0.2524995 * Q.D2_b2 - 0.4481503
    if Q.nca_sj4_pair_mass_2nd >= 72.72359:
        z += -0.01585185 * Q.nca_sj4_pair_mass_2nd + 1.152804
    if 47.97531 <= Q.mres_sd_mass_b0z02 < 86.60355:
        z += 0.004749006 * Q.mres_sd_mass_b0z02 - 0.227835
    if 86.60355 <= Q.mres_sd_mass_b0z02 < 134.2023:
        z += -0.007506888 * Q.mres_sd_mass_b0z02 + 0.8335689
    if Q.mres_sd_mass_b0z02 >= 134.2023:
        z += 0.003181796 * Q.mres_sd_mass_b0z02 - 0.6008772
    if 86.14266 <= Q.mres_pruned_mass < 124.3145:
        z += -0.009355783 * Q.mres_pruned_mass + 0.805932
    if Q.mres_pruned_mass >= 124.3145:
        z += 0.005076483 * Q.mres_pruned_mass - 0.9882085
    if Q.pz_lnd3 >= 0.01024929:
        z += 0.7860092 * Q.pz_lnd3 - 0.008056039
    if Q.mres_sd_rg_b2z01 >= 0.457968:
        z += -1.50545 * Q.mres_sd_rg_b2z01 + 0.6894481
    if Q.n_sdz_above_5 < 3.0:
        z += -0.07397865 * Q.n_sdz_above_5 + 0.2219359
    if Q.sum_z_dr2_top2 < 0.005938474:
        z += -25.07804 * Q.sum_z_dr2_top2 + 0.1489253
    if Q.tau32 < 0.7981752 and Q.sj3_dr_min < 0.3190414:
        z += 2.347932 * (0.7981752 - Q.tau32) * (0.3190414 - Q.sj3_dr_min)
    if Q.pair_mean_lnm2 < 3.38061 and Q.z_top30_slots > 0.8316085:
        z += 0.5528035 * (3.38061 - Q.pair_mean_lnm2) * (Q.z_top30_slots - 0.8316085)
    if Q.e3_b2 < 0.0004126585 and Q.lnptrel_25 > -6.691703:
        z += 405.4624 * (0.0004126585 - Q.e3_b2) * (Q.lnptrel_25 - -6.691703)
    if Q.e3_b2 < 0.0004126585 and Q.ak02_dr23 > 0.3194837:
        z += 1464.161 * (0.0004126585 - Q.e3_b2) * (Q.ak02_dr23 - 0.3194837)
    if Q.sj4_pair_mass_max > 69.83554 and Q.nca_sj4_pair2nd_over_mass > 0.4488456:
        z += 0.02631643 * (Q.sj4_pair_mass_max - 69.83554) * (Q.nca_sj4_pair2nd_over_mass - 0.4488456)
    if Q.mass < 117.4867 and Q.D2_b2 < 1.380731:
        z += 0.01304915 * (117.4867 - Q.mass) * (1.380731 - Q.D2_b2)
    if Q.e3_b2 < 0.0004126585 and Q.sj3_mass1 > 17.31003:
        z += 24.35142 * (0.0004126585 - Q.e3_b2) * (Q.sj3_mass1 - 17.31003)
    if Q.sj4_pair_mass_max > 69.83554 and Q.dc_3_z < 0.2037349:
        z += -0.03519902 * (Q.sj4_pair_mass_max - 69.83554) * (0.2037349 - Q.dc_3_z)
    if Q.lep_ptrel > 6.983043 and Q.C2_b2 < 0.1682645:
        z += 0.05753703 * (Q.lep_ptrel - 6.983043) * (0.1682645 - Q.C2_b2)
    if Q.sj3_pair_mass_max > 128.6605 and Q.e4 > 7.184834e-06:
        z += 238.5692 * (Q.sj3_pair_mass_max - 128.6605) * (Q.e4 - 7.184834e-06)
    if Q.pz_lnd2 < 0.1339824 and Q.sj3_dr13 > 0.4691911:
        z += -5.995172 * (0.1339824 - Q.pz_lnd2) * (Q.sj3_dr13 - 0.4691911)
    if Q.z_displaced3 < 0.1061578 and Q.lepsj_2_maxsd0 < 0.7828545:
        z += -5.246693 * (0.1061578 - Q.z_displaced3) * (0.7828545 - Q.lepsj_2_maxsd0)
    if Q.tau5 < 0.05613495 and Q.dc_ntag > 0.0:
        z += -3.412682 * (0.05613495 - Q.tau5) * (Q.dc_ntag - 0.0)
    if Q.z_charged_had > 0.265564 and Q.pz_lnkt3 > 0.0:
        z += 4.23353 * (Q.z_charged_had - 0.265564) * (Q.pz_lnkt3 - 0.0)
    if Q.lep_dr < 0.04607888 and Q.lepsj_2_maxsd0 < 0.7828545:
        z += 11.10832 * (0.04607888 - Q.lep_dr) * (0.7828545 - Q.lepsj_2_maxsd0)
    if Q.mass < 117.4867 and Q.C2_b2 > 0.01891146:
        z += -0.03725553 * (117.4867 - Q.mass) * (Q.C2_b2 - 0.01891146)
    if Q.z_displaced3 < 0.1061578 and Q.dr_29 < 0.02258554:
        z += 102.7889 * (0.1061578 - Q.z_displaced3) * (0.02258554 - Q.dr_29)
    if Q.e3_b2 < 0.0004126585 and Q.n_lund_kt_above_5 > 2.0:
        z += 213.1262 * (0.0004126585 - Q.e3_b2) * (Q.n_lund_kt_above_5 - 2.0)
    if Q.z_displaced3 < 0.1061578 and Q.n_lund > 10.0:
        z += -0.5483097 * (0.1061578 - Q.z_displaced3) * (Q.n_lund - 10.0)
    return z


def neuron_69(Q):
    z = -6.926144e-06
    return z


def neuron_70(Q):
    z = 0.01601256
    return z


def neuron_71(Q):
    z = 3.029709e-06
    return z


def neuron_72(Q):
    z = -1.16412e-06
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
    z = -5.542789e-06
    return z


def neuron_77(Q):
    z = -3.087486e-06
    return z


def neuron_78(Q):
    z = 0.200681
    if Q.lep_z < 0.3396572:
        z += 3.961272 * Q.lep_z - 1.345475
    if Q.n_pairs_kt_above_1 < 195.0:
        z += -0.0031798 * Q.n_pairs_kt_above_1 + 0.620061
    if 9.380468 <= Q.mass_displaced5 < 21.12768:
        z += 0.01445477 * Q.mass_displaced5 - 0.1355925
    if Q.mass_displaced5 >= 21.12768:
        z += -0.001099025 * Q.mass_displaced5 + 0.1930231
    if Q.lep_ptrel < 18.7678:
        z += 0.05459585 * Q.lep_ptrel - 0.7077334
    if 18.7678 <= Q.lep_ptrel < 43.20788:
        z += -0.01296683 * Q.lep_ptrel + 0.5602695
    if Q.mass < 95.14961:
        z += 0.007819425 * Q.mass - 0.7440153
    if 114.0172 <= Q.mass < 164.4374:
        z += 0.008976511 * Q.mass - 1.023477
    if Q.mass >= 164.4374:
        z += 0.003401275 * Q.mass - 0.1066995
    if Q.lepsj_3_n_d3 < 2.0:
        z += -0.09895811 * Q.lepsj_3_n_d3 + 0.1979162
    if Q.mass_top40 < 70.88236:
        z += -0.0248104 * Q.mass_top40 + 1.758619
    if Q.n_neutral < 22.0:
        z += -0.0165848 * Q.n_neutral + 0.3648656
    if Q.pair_max_lnm2 >= 7.347625:
        z += -0.2460556 * Q.pair_max_lnm2 + 1.807924
    if Q.sdb_2_n < 4.0:
        z += -0.02961375 * Q.sdb_2_n + 0.2665238
    if 4.0 <= Q.sdb_2_n < 9.0:
        z += -0.0145832 * Q.sdb_2_n + 0.2064016
    if Q.sdb_2_n >= 9.0:
        z += 0.01503055 * Q.sdb_2_n - 0.0601222
    if Q.z_top15_slots >= 0.8008865:
        z += -1.165863 * Q.z_top15_slots + 0.9337238
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.008393501 * Q.mres_sd_mass_b0z005 - 1.024846
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.01375969 * Q.mres_sd_mass_b0z005 + 2.517986
    if Q.mass_top20 >= 130.7093:
        z += 0.008199648 * Q.mass_top20 - 1.07177
    if Q.lund3_lndelta >= -2.817283:
        z += 0.08642481 * Q.lund3_lndelta + 0.2434831
    if Q.n_charged_had >= 22.0:
        z += 0.02106266 * Q.n_charged_had - 0.4633784
    if Q.sj4_pair_mass_max < 72.862:
        z += -0.004298599 * Q.sj4_pair_mass_max + 0.3132045
    if Q.lepsj_3_maxsd0 < 2.413264:
        z += -0.1502495 * Q.lepsj_3_maxsd0 + 0.1344615
    if 2.413264 <= Q.lepsj_3_maxsd0 < 586.9572:
        z += 0.0003902705 * Q.lepsj_3_maxsd0 - 0.2290721
    if Q.n_lepton < 1.0:
        z += 0.6736446 * Q.n_lepton - 0.6736446
    if Q.n_pairs_kt_above_3 < 99.0:
        z += -0.003582742 * Q.n_pairs_kt_above_3 + 0.3546914
    if Q.mass_2charged < 1.839882:
        z += -0.129505 * Q.mass_2charged + 0.1104387
    if 1.839882 <= Q.mass_2charged < 10.83159:
        z += 0.01421703 * Q.mass_2charged - 0.153993
    if Q.lep_iso < 1.362094:
        z += -0.1008004 * Q.lep_iso + 0.1372996
    if Q.lepsj_2_dr < 0.103005:
        z += 7.34972 * Q.lepsj_2_dr - 0.7570578
    if Q.n_s3d_above_3 < 9.0:
        z += -0.06941678 * Q.n_s3d_above_3 + 0.6247511
    if Q.mass_top15 >= 83.68425:
        z += 0.003902172 * Q.mass_top15 - 0.3265503
    if Q.mass_top10 >= 103.4976:
        z += -0.005799451 * Q.mass_top10 + 0.6002296
    if Q.N2_b05 < 0.5083429:
        z += -3.250205 * Q.N2_b05 + 1.652219
    if Q.ktd_ln_d34 < -8.400697:
        z += 0.1083978 * Q.ktd_ln_d34 + 0.9106168
    if Q.lep_z < 0.3396572 and Q.pz_lnd2 < 0.1956014:
        z += 3.307549 * (0.3396572 - Q.lep_z) * (0.1956014 - Q.pz_lnd2)
    if Q.lep_z < 0.3396572 and Q.mass_top20 > 86.78877:
        z += 0.02165609 * (0.3396572 - Q.lep_z) * (Q.mass_top20 - 86.78877)
    if Q.mass_displaced5 > 9.380468 and Q.mass_charged < 113.5019:
        z += -0.0002301832 * (Q.mass_displaced5 - 9.380468) * (113.5019 - Q.mass_charged)
    if Q.lep_ptrel < 18.7678 and Q.ktd_ln_d34 < -7.886954:
        z += 0.008543691 * (18.7678 - Q.lep_ptrel) * (-7.886954 - Q.ktd_ln_d34)
    if Q.n_s3d_above_3 > 4.0 and Q.sip_3d_2 < 226.3008:
        z += 0.0002605735 * (Q.n_s3d_above_3 - 4.0) * (226.3008 - Q.sip_3d_2)
    if Q.n_s3d_above_3 > 4.0 and Q.z_top50_slots > 0.9572293:
        z += 0.2220186 * (Q.n_s3d_above_3 - 4.0) * (Q.z_top50_slots - 0.9572293)
    if Q.lep_ptrel < 18.7678 and Q.sj4_dr_min < 0.1631992:
        z += 0.03309953 * (18.7678 - Q.lep_ptrel) * (0.1631992 - Q.sj4_dr_min)
    if Q.lepsj_3_n_d3 < 2.0 and Q.jet_e > 835.8719:
        z += -0.0001136693 * (2.0 - Q.lepsj_3_n_d3) * (Q.jet_e - 835.8719)
    if Q.lep_ptrel < 18.7678 and Q.ak02_2_z > 0.08487383:
        z += -0.02717349 * (18.7678 - Q.lep_ptrel) * (Q.ak02_2_z - 0.08487383)
    if Q.lep_z < 0.3396572 and Q.ecf_g42 < 8.17233e-05:
        z += 11173.13 * (0.3396572 - Q.lep_z) * (8.17233e-05 - Q.ecf_g42)
    if Q.lep_ptrel < 18.7678 and Q.sum_pt_top30 < 749.0582:
        z += 4.195037e-05 * (18.7678 - Q.lep_ptrel) * (749.0582 - Q.sum_pt_top30)
    if Q.z_top15_slots > 0.8008865 and Q.z_displaced5 > 0.2184442:
        z += -5.596133 * (Q.z_top15_slots - 0.8008865) * (Q.z_displaced5 - 0.2184442)
    if Q.lep_ptrel < 18.7678 and Q.mass_2photon < 22.18431:
        z += -0.0006512748 * (18.7678 - Q.lep_ptrel) * (22.18431 - Q.mass_2photon)
    if Q.lep_ptrel < 18.7678 and Q.sjf_4_2_z_d3 < 0.1570831:
        z += 0.04582237 * (18.7678 - Q.lep_ptrel) * (0.1570831 - Q.sjf_4_2_z_d3)
    if Q.sdb_2_n < 9.0 and Q.sjq_2_prod_k03 > -0.9998116:
        z += -0.04124967 * (9.0 - Q.sdb_2_n) * (Q.sjq_2_prod_k03 - -0.9998116)
    if Q.lep_z < 0.3396572 and Q.lepsj_2_dr < 0.103005:
        z += 30.06639 * (0.3396572 - Q.lep_z) * (0.103005 - Q.lepsj_2_dr)
    if Q.sj4_pair_mass_max < 115.5142 and Q.n_neutral_had < 9.0:
        z += -0.0003997466 * (115.5142 - Q.sj4_pair_mass_max) * (9.0 - Q.n_neutral_had)
    return z


def neuron_79(Q):
    z = -3.602388e-05
    return z


def neuron_80(Q):
    z = 9.707706e-06
    return z


def neuron_81(Q):
    z = 0.01274317
    if Q.z_displaced3 >= 0.03624058:
        z += 3.064015 * Q.z_displaced3 - 0.1110417
    if Q.n_pairs_kt_above_3 < 28.0:
        z += -0.007745469 * Q.n_pairs_kt_above_3 + 0.2168731
    if Q.mass_top40 >= 115.7429:
        z += 0.006369834 * Q.mass_top40 - 0.7372632
    if Q.mass_top50 >= 161.1264:
        z += -0.01046151 * Q.mass_top50 + 1.685626
    if Q.ecf_g31 < 0.01162881:
        z += -69.21011 * Q.ecf_g31 + 0.8048314
    if Q.mres_sd_mass_b2z01 < 76.1529:
        z += -0.003165079 * Q.mres_sd_mass_b2z01 - 0.1064536
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 91.15481:
        z += 0.01552412 * Q.mres_sd_mass_b2z01 - 1.52969
    if 91.15481 <= Q.mres_sd_mass_b2z01 < 115.0388:
        z += -0.02737351 * Q.mres_sd_mass_b2z01 + 2.380635
    if 115.0388 <= Q.mres_sd_mass_b2z01 < 125.2732:
        z += 0.003863933 * Q.mres_sd_mass_b2z01 - 1.212884
    if 125.2732 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.0221338 * Q.mres_sd_mass_b2z01 - 3.501609
    if Q.tau1 < 0.06074238:
        z += -11.50323 * Q.tau1 + 0.6987337
    if Q.sjf_2_1_n_d3 >= 5.0:
        z += -0.0961644 * Q.sjf_2_1_n_d3 + 0.480822
    if 2.0 <= Q.n_s3d_above_3 < 6.0:
        z += 0.08705317 * Q.n_s3d_above_3 - 0.1741063
    if Q.n_s3d_above_3 >= 6.0:
        z += 0.1577148 * Q.n_s3d_above_3 - 0.5980759
    if Q.sip_3d_2 < 226.3008:
        z += -0.001136699 * Q.sip_3d_2 + 0.257236
    if Q.sum_zz_dr2 < 0.06117886:
        z += 5.254972 * Q.sum_zz_dr2 - 0.3214932
    if Q.jd_3d_5 < 4.004982:
        z += -0.1264279 * Q.jd_3d_5 + 0.5063413
    if Q.n_pairs_kt_above_1 < 58.0:
        z += -0.007608365 * Q.n_pairs_kt_above_1 + 0.4412852
    if Q.max_abs_d0 < 10.52344:
        z += -0.02713943 * Q.max_abs_d0 + 0.2856001
    if Q.sv_n >= 1.0:
        z += -0.1014532 * Q.sv_n + 0.1014532
    if Q.lepsj_3_dr < 0.005262883:
        z += 72.86868 * Q.lepsj_3_dr - 0.3834993
    z += -0.05356498 * Q.lepsj_2_n_d3
    if Q.sj4_pair_mass_max < 59.5457:
        z += -0.007502227 * Q.sj4_pair_mass_max + 0.4467254
    if Q.n_s3d_above_10 >= 3.0:
        z += 0.06901132 * Q.n_s3d_above_10 - 0.2070339
    if Q.sdb_0_n < 3.0:
        z += -0.03977636 * Q.sdb_0_n + 0.02066215
    if 3.0 <= Q.sdb_0_n < 5.0:
        z += 0.04933346 * Q.sdb_0_n - 0.2466673
    if Q.mass_displaced3 < 1.777286:
        z += 0.09275652 * Q.mass_displaced3 - 0.1648549
    if Q.sjf_4_n2disp >= 1.0:
        z += -0.1845095 * Q.sjf_4_n2disp + 0.1845095
    if Q.n_dr_0p4_up < 15.0:
        z += 0.01312843 * Q.n_dr_0p4_up - 0.1969265
    if Q.z_displaced3 > 0.03624058 and Q.n_s3d_above_3 < 7.0:
        z += -0.4705695 * (Q.z_displaced3 - 0.03624058) * (7.0 - Q.n_s3d_above_3)
    if Q.z_displaced3 > 0.03624058 and Q.z_charged_had > 0.265564:
        z += -6.858511 * (Q.z_displaced3 - 0.03624058) * (Q.z_charged_had - 0.265564)
    if Q.z_displaced3 > 0.03624058 and Q.jd_sum_abs_sd0_top3 < 1262.672:
        z += 0.0005350134 * (Q.z_displaced3 - 0.03624058) * (1262.672 - Q.jd_sum_abs_sd0_top3)
    if Q.ecf_g31 < 0.01162881 and Q.n_s3d_above_3 > 4.0:
        z += 6.358389 * (0.01162881 - Q.ecf_g31) * (Q.n_s3d_above_3 - 4.0)
    if Q.sip_3d_2 < 226.3008 and Q.jd_3d_5 < 6.771002:
        z += -0.0003833284 * (226.3008 - Q.sip_3d_2) * (6.771002 - Q.jd_3d_5)
    if Q.n_s3d_above_3 > 2.0 and Q.n_lund_kt_above_5 < 4.0:
        z += 0.01135027 * (Q.n_s3d_above_3 - 2.0) * (4.0 - Q.n_lund_kt_above_5)
    if Q.sdb_4_z < 0.04510459 and Q.lep_z < 0.004135872:
        z += 961.7342 * (0.04510459 - Q.sdb_4_z) * (0.004135872 - Q.lep_z)
    if Q.ecf_g31 < 0.01162881 and Q.sjq_2_prod_k05 > -0.5057096:
        z += -61.86287 * (0.01162881 - Q.ecf_g31) * (Q.sjq_2_prod_k05 - -0.5057096)
    if Q.z_displaced3 > 0.03624058 and Q.n_lund_kt_above_1 > 2.0:
        z += 0.1236444 * (Q.z_displaced3 - 0.03624058) * (Q.n_lund_kt_above_1 - 2.0)
    if Q.max_abs_d0 < 10.52344 and Q.jd_3d_4 < 6.43167:
        z += -0.006537757 * (10.52344 - Q.max_abs_d0) * (6.43167 - Q.jd_3d_4)
    if Q.mres_sd_mass_b2z01 < 158.2019 and Q.mass_2charged > 24.4079:
        z += 0.0002829437 * (158.2019 - Q.mres_sd_mass_b2z01) * (Q.mass_2charged - 24.4079)
    if Q.lepsj_3_dr < 0.005262883 and Q.jd_3d_4 < 2.953464:
        z += 48.37366 * (0.005262883 - Q.lepsj_3_dr) * (2.953464 - Q.jd_3d_4)
    if Q.ecf_g31 < 0.01162881 and Q.sjq_2_sumabs_k1 > 0.1588551:
        z += 46.28993 * (0.01162881 - Q.ecf_g31) * (Q.sjq_2_sumabs_k1 - 0.1588551)
    if Q.n_s3d_above_3 > 2.0 and Q.z_photon < 0.4982257:
        z += -0.07232887 * (Q.n_s3d_above_3 - 2.0) * (0.4982257 - Q.z_photon)
    if Q.z_displaced3 < 0.06416437 and Q.dc_2_n_lep < 1.0:
        z += 4.333104 * (0.06416437 - Q.z_displaced3) * (1.0 - Q.dc_2_n_lep)
    return z


def neuron_82(Q):
    z = -2.942075e-06
    return z


def neuron_83(Q):
    z = 1.206535
    if 78.33213 <= Q.mass_top40 < 155.8928:
        z += 0.01042447 * Q.mass_top40 - 0.8165707
    if Q.mass_top40 >= 155.8928:
        z += 0.03012342 * Q.mass_top40 - 3.887495
    if Q.z_displaced3 < 0.06416437:
        z += -5.676748 * Q.z_displaced3 + 0.364245
    if Q.mass_top30 < 162.7874:
        z += 0.002517485 * Q.mass_top30 - 0.4098149
    if Q.n_s3d_above_10 < 6.0:
        z += 0.07406672 * Q.n_s3d_above_10 - 0.4444003
    if Q.mres_sd_mass_b2z01 < 81.19466:
        z += 0.01416554 * Q.mres_sd_mass_b2z01 - 1.972644
    if 81.19466 <= Q.mres_sd_mass_b2z01 < 130.0968:
        z += -0.007155901 * Q.mres_sd_mass_b2z01 - 0.2414562
    if 130.0968 <= Q.mres_sd_mass_b2z01 < 139.2565:
        z += 0.02739475 * Q.mres_sd_mass_b2z01 - 4.736387
    if 139.2565 <= Q.mres_sd_mass_b2z01 < 158.2019:
        z += 0.01322921 * Q.mres_sd_mass_b2z01 - 2.763744
    if Q.mres_sd_mass_b2z01 >= 158.2019:
        z += 0.006086495 * Q.mres_sd_mass_b2z01 - 1.633753
    if Q.N2 < 0.258375:
        z += 2.090275 * Q.N2 - 0.5400746
    if Q.M3_b2 < 0.02160244:
        z += 14.52947 * Q.M3_b2 - 0.3138719
    if Q.sdb_2_z < 0.1530389:
        z += -4.070851 * Q.sdb_2_z + 0.4014704
    if 0.1530389 <= Q.sdb_2_z < 0.5109872:
        z += 0.6188837 * Q.sdb_2_z - 0.3162416
    if Q.n_s3d_above_3 < 3.0:
        z += -0.0500585 * Q.n_s3d_above_3 + 0.1501755
    if Q.mass_top15 >= 77.22442:
        z += -0.01022329 * Q.mass_top15 + 0.7894878
    if Q.pair_max_lnm2 >= 7.524613:
        z += 0.4150206 * Q.pair_max_lnm2 - 3.122869
    if Q.sj4_pair_mass_max < 75.78208:
        z += -0.003106337 * Q.sj4_pair_mass_max + 0.7867813
    if 75.78208 <= Q.sj4_pair_mass_max < 115.5142:
        z += -0.01387735 * Q.sj4_pair_mass_max + 1.603031
    if Q.max_abs_d0 < 10.52344:
        z += -0.02540739 * Q.max_abs_d0 + 0.267373
    if Q.jd_sum_abs_sd0_top5 < 185.1889:
        z += 0.001329836 * Q.jd_sum_abs_sd0_top5 - 0.2462708
    if Q.sip_3d_1 < 7.028704:
        z += -0.06731308 * Q.sip_3d_1 + 0.4731237
    if Q.tau1 < 0.06074238:
        z += 13.25452 * Q.tau1 - 0.805111
    if 79.47361 <= Q.mass < 95.14961:
        z += -0.02301835 * Q.mass + 1.829352
    if 95.14961 <= Q.mass < 117.4867:
        z += -0.02902508 * Q.mass + 2.400889
    if Q.mass >= 117.4867:
        z += -0.0153558 * Q.mass + 0.794931
    if Q.sdb_2_n < 10.0:
        z += -0.02252318 * Q.sdb_2_n + 0.2252318
    if Q.e4 < 3.53193e-06:
        z += 91865.45 * Q.e4 - 0.3244623
    if Q.n_pt_above_5 < 24.0:
        z += 0.02627983 * Q.n_pt_above_5 - 0.630716
    if Q.tau5 >= 0.01467699:
        z += 12.0469 * Q.tau5 - 0.1768122
    if Q.sum_z_dr2_top50 < 0.0925671:
        z += -10.213 * Q.sum_z_dr2_top50 + 0.9453876
    if Q.D3_b05 < 1.139376:
        z += 0.4406897 * Q.D3_b05 - 0.5021113
    if Q.mass_top10 >= 103.4976:
        z += -0.006064146 * Q.mass_top10 + 0.6276249
    if Q.n_s3d_above_3 > 6.0 and Q.jd_sum_abs_sd0_top5 < 1292.625:
        z += 7.700892e-05 * (Q.n_s3d_above_3 - 6.0) * (1292.625 - Q.jd_sum_abs_sd0_top5)
    if Q.mass_top40 > 78.33213 and Q.lep_z < 0.1275041:
        z += -0.04711751 * (Q.mass_top40 - 78.33213) * (0.1275041 - Q.lep_z)
    if Q.mass_top30 < 162.7874 and Q.mass_displaced5 > 9.380468:
        z += -0.0001497269 * (162.7874 - Q.mass_top30) * (Q.mass_displaced5 - 9.380468)
    if Q.n_s3d_above_3 > 6.0 and Q.sj3_pairmin_over_m < 0.4866692:
        z += 0.2629887 * (Q.n_s3d_above_3 - 6.0) * (0.4866692 - Q.sj3_pairmin_over_m)
    if Q.mres_sd_mass_b2z01 < 139.2565 and Q.dc_2_n_lep < 1.0:
        z += 0.002369081 * (139.2565 - Q.mres_sd_mass_b2z01) * (1.0 - Q.dc_2_n_lep)
    if Q.sdb_2_z < 0.1530389 and Q.jd_3d_5 < 4.004982:
        z += -1.857927 * (0.1530389 - Q.sdb_2_z) * (4.004982 - Q.jd_3d_5)
    if Q.n_s3d_above_10 < 6.0 and Q.lne_0 > 4.380463:
        z += 0.02859163 * (6.0 - Q.n_s3d_above_10) * (Q.lne_0 - 4.380463)
    if Q.max_abs_d0 < 10.52344 and Q.jd_3d_5 < 6.771002:
        z += -0.006830802 * (10.52344 - Q.max_abs_d0) * (6.771002 - Q.jd_3d_5)
    if Q.mass_displaced3 < 13.03663 and Q.jd_3d_5 < 6.771002:
        z += 0.006377215 * (13.03663 - Q.mass_displaced3) * (6.771002 - Q.jd_3d_5)
    if Q.n_s3d_above_3 > 6.0 and Q.jd_3d_4 < 172.888:
        z += 0.0008468669 * (Q.n_s3d_above_3 - 6.0) * (172.888 - Q.jd_3d_4)
    if Q.sj4_pair_mass_max < 75.78208 and Q.sjf_4_3_z_d3 < 0.07303924:
        z += -0.1134615 * (75.78208 - Q.sj4_pair_mass_max) * (0.07303924 - Q.sjf_4_3_z_d3)
    if Q.sv_2_z < 0.03767806 and Q.max_dr > 0.8033751:
        z += 6.919964 * (0.03767806 - Q.sv_2_z) * (Q.max_dr - 0.8033751)
    return z


def neuron_84(Q):
    z = -7.427514e-05
    return z


def neuron_85(Q):
    z = -8.185525e-07
    return z


def neuron_86(Q):
    z = -1.709903e-05
    return z


def neuron_87(Q):
    z = 1.619162e-06
    return z


def neuron_88(Q):
    z = 2.99698e-07
    return z


def neuron_89(Q):
    z = 1.735374e-06
    return z


def neuron_90(Q):
    z = 0.001229865
    return z


def neuron_91(Q):
    z = 3.992639e-06
    return z


def neuron_92(Q):
    z = -6.724736e-06
    return z


def neuron_93(Q):
    z = 8.46439e-07
    return z


def neuron_94(Q):
    z = -2.322172e-06
    return z


def neuron_95(Q):
    z = 4.369519e-07
    return z


def neuron_96(Q):
    z = 4.278031e-07
    return z


def neuron_97(Q):
    z = -1.836736
    if Q.mass_displaced3 < 18.80005:
        z += -0.0468278 * Q.mass_displaced3 + 1.457675
    if 18.80005 <= Q.mass_displaced3 < 39.09615:
        z += -0.02844436 * Q.mass_displaced3 + 1.112065
    if Q.mass < 95.14961:
        z += 0.004492254 * Q.mass - 2.100618
    if 95.14961 <= Q.mass < 100.4835:
        z += 0.01408652 * Q.mass - 3.013509
    if 100.4835 <= Q.mass < 149.0507:
        z += 0.04047148 * Q.mass - 5.664761
    if 149.0507 <= Q.mass < 164.4374:
        z += 0.002160428 * Q.mass + 0.04552748
    if 164.4374 <= Q.mass < 182.8592:
        z += -0.02175592 * Q.mass + 3.97827
    if Q.e3_b2 < 9.621843e-05:
        z += 3671.387 * Q.e3_b2 - 0.2909184
    if 9.621843e-05 <= Q.e3_b2 < 0.0002536827:
        z += 1332.387 * Q.e3_b2 - 0.06586351
    if 0.0002536827 <= Q.e3_b2 < 0.0007909605:
        z += -506.5166 * Q.e3_b2 + 0.4006346
    if Q.sj3_pair_mass_max < 63.7508:
        z += -0.03833658 * Q.sj3_pair_mass_max + 4.096229
    if 63.7508 <= Q.sj3_pair_mass_max < 114.7848:
        z += -0.03237529 * Q.sj3_pair_mass_max + 3.716192
    if Q.mass_top50 < 71.80762:
        z += 0.0232366 * Q.mass_top50 - 2.19328
    if 71.80762 <= Q.mass_top50 < 122.0585:
        z += 0.01044189 * Q.mass_top50 - 1.274522
    if Q.sip_3d_2 < 447.0873:
        z += 0.0008375384 * Q.sip_3d_2 - 0.3744528
    if Q.jd_3d_4 < 172.888:
        z += 0.001229958 * Q.jd_3d_4 - 0.212645
    if Q.n_pairs_kt_above_3 < 111.0:
        z += -0.003078462 * Q.n_pairs_kt_above_3 + 0.3417092
    if Q.ak02_min12_jp < 7.746064:
        z += -0.01583975 * Q.ak02_min12_jp + 0.1226957
    z += 2.851042 * Q.sj3_pairmax_over_m
    if Q.mass_top40 < 126.8853:
        z += -0.01218541 * Q.mass_top40 + 1.546149
    if Q.N2_b05 >= 0.4353632:
        z += -1.864396 * Q.N2_b05 + 0.8116893
    if Q.max_abs_d0 < 5.8125:
        z += 0.08973733 * Q.max_abs_d0 - 0.5215983
    if Q.lep_ptrel >= 3.53503:
        z += 0.006845198 * Q.lep_ptrel - 0.02419798
    if Q.sdb_2_z < 0.2740506:
        z += 1.558766 * Q.sdb_2_z - 0.4271807
    if Q.pair_max_lnm2 < 7.075642:
        z += -0.06843682 * Q.pair_max_lnm2 + 0.4842344
    if Q.n_sd0_above_3 < 6.0:
        z += 0.1399207 * Q.n_sd0_above_3 - 0.8395244
    if Q.sjf_3_1_n_d5 < 5.0:
        z += -0.02570452 * Q.sjf_3_1_n_d5 + 0.1285226
    if Q.sj2_mass2 < 1.851735:
        z += -0.2510223 * Q.sj2_mass2 + 0.4648269
    if Q.mres_pruned_mass < 70.04065:
        z += 0.003285808 * Q.mres_pruned_mass - 1.362385
    if 70.04065 <= Q.mres_pruned_mass < 112.1947:
        z += -0.009964224 * Q.mres_pruned_mass - 0.4343437
    if 112.1947 <= Q.mres_pruned_mass < 124.3145:
        z += -0.02688484 * Q.mres_pruned_mass + 1.464059
    if 124.3145 <= Q.mres_pruned_mass < 171.3819:
        z += 0.01137394 * Q.mres_pruned_mass - 3.292064
    if Q.mres_pruned_mass >= 171.3819:
        z += -0.01325003 * Q.mres_pruned_mass + 0.9280409
    if Q.sd_mass >= 119.4443:
        z += 0.01256525 * Q.sd_mass - 1.500848
    if 53.57509 <= Q.mres_sd_mass_b0z005 < 76.40585:
        z += 0.02155932 * Q.mres_sd_mass_b0z005 - 1.155043
    if 76.40585 <= Q.mres_sd_mass_b0z005 < 91.82593:
        z += -0.06289417 * Q.mres_sd_mass_b0z005 + 5.297699
    if 91.82593 <= Q.mres_sd_mass_b0z005 < 97.29337:
        z += 0.01863774 * Q.mres_sd_mass_b0z005 - 2.189045
    if 97.29337 <= Q.mres_sd_mass_b0z005 < 122.1:
        z += -0.003465289 * Q.mres_sd_mass_b0z005 - 0.03856659
    if 122.1 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += 0.02568101 * Q.mres_sd_mass_b0z005 - 3.597329
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.001299756 * Q.mres_sd_mass_b0z005 + 0.7175489
    if Q.n_pairs_kt_above_1 < 325.0:
        z += -0.0009989868 * Q.n_pairs_kt_above_1 + 0.3246707
    if Q.lund3_lndelta >= -1.780944:
        z += 0.1738331 * Q.lund3_lndelta + 0.3095871
    if Q.n_s3d_above_3 < 9.0:
        z += -0.03836546 * Q.n_s3d_above_3 + 0.3452891
    if Q.z_displaced5 >= 0.1339658:
        z += -0.547458 * Q.z_displaced5 + 0.07334063
    if Q.mres_sd_prong_mass1 >= 69.04524:
        z += -0.006374701 * Q.mres_sd_prong_mass1 + 0.4401428
    if Q.lund_max_lnkt < 3.734077:
        z += 0.1276991 * Q.lund_max_lnkt - 0.4768384
    if Q.lep_z < 0.221436:
        z += 0.9850135 * Q.lep_z - 0.2181175
    if 72.72359 <= Q.nca_sj4_pair_mass_2nd < 83.41384:
        z += -0.02592031 * Q.nca_sj4_pair_mass_2nd + 1.885018
    if Q.nca_sj4_pair_mass_2nd >= 83.41384:
        z += 0.004490975 * Q.nca_sj4_pair_mass_2nd - 0.6517042
    if Q.pz_lnd3 < 0.2113485:
        z += -1.146978 * Q.pz_lnd3 + 0.2424121
    if Q.e3_b2 < 0.0002536827 and Q.sdb_2_z < 0.3709098:
        z += 2195.149 * (0.0002536827 - Q.e3_b2) * (0.3709098 - Q.sdb_2_z)
    if Q.mass < 100.4835 and Q.lep_ptrel < 1.477152:
        z += 0.005932264 * (100.4835 - Q.mass) * (1.477152 - Q.lep_ptrel)
    if Q.mass_displaced3 < 39.09615 and Q.lep_ptrel < 18.7678:
        z += -0.0004157222 * (39.09615 - Q.mass_displaced3) * (18.7678 - Q.lep_ptrel)
    if Q.mass < 182.8592 and Q.n_s3d_above_3 > 1.0:
        z += -0.0009660866 * (182.8592 - Q.mass) * (Q.n_s3d_above_3 - 1.0)
    if Q.e3_b2 < 0.0002536827 and Q.jet_charge_k03 > -0.05990128:
        z += 1511.199 * (0.0002536827 - Q.e3_b2) * (Q.jet_charge_k03 - -0.05990128)
    if Q.mass_top50 < 122.0585 and Q.ak02_2_n_lep < 1.0:
        z += 0.007770317 * (122.0585 - Q.mass_top50) * (1.0 - Q.ak02_2_n_lep)
    if Q.mass_displaced3 < 39.09615 and Q.pz_lnd3 < 0.18807:
        z += 0.03423624 * (39.09615 - Q.mass_displaced3) * (0.18807 - Q.pz_lnd3)
    if Q.mass_top50 < 122.0585 and Q.sum_e > 926.2598:
        z += 9.622586e-06 * (122.0585 - Q.mass_top50) * (Q.sum_e - 926.2598)
    if Q.sip_3d_2 < 447.0873 and Q.dc_1_n_lep < 1.0:
        z += 0.0003481107 * (447.0873 - Q.sip_3d_2) * (1.0 - Q.dc_1_n_lep)
    if Q.sj3_pair_mass_max < 114.7848 and Q.n_photon > 12.0:
        z += -0.0003099577 * (114.7848 - Q.sj3_pair_mass_max) * (Q.n_photon - 12.0)
    if Q.mass_top50 < 122.0585 and Q.sjq_2_prod_k05 < 0.125305:
        z += -0.01509263 * (122.0585 - Q.mass_top50) * (0.125305 - Q.sjq_2_prod_k05)
    if Q.max_abs_d0 < 5.8125 and Q.jd_3d_5 < 4.004982:
        z += 0.04216214 * (5.8125 - Q.max_abs_d0) * (4.004982 - Q.jd_3d_5)
    if Q.sdb_2_z < 0.2740506 and Q.pz_lnkt1 < 0.1348361:
        z += 14.32548 * (0.2740506 - Q.sdb_2_z) * (0.1348361 - Q.pz_lnkt1)
    if Q.mass_top40 < 126.8853 and Q.n_muon > 1.0:
        z += -0.01268683 * (126.8853 - Q.mass_top40) * (Q.n_muon - 1.0)
    if Q.mass_displaced3 < 39.09615 and Q.D2 > 3.038341:
        z += 0.001323137 * (39.09615 - Q.mass_displaced3) * (Q.D2 - 3.038341)
    if Q.e3_b2 < 0.0007909605 and Q.sjq_3_sumabs_k1 > 0.4372817:
        z += 299.5746 * (0.0007909605 - Q.e3_b2) * (Q.sjq_3_sumabs_k1 - 0.4372817)
    if Q.mass < 100.4835 and Q.lep_iso < 6.185635:
        z += -0.001315948 * (100.4835 - Q.mass) * (6.185635 - Q.lep_iso)
    return z


def neuron_98(Q):
    z = 4.990169e-07
    return z


def neuron_99(Q):
    z = -0.001934242
    return z


def neuron_100(Q):
    z = -6.264077e-06
    return z


def neuron_101(Q):
    z = 8.878263e-06
    return z


def neuron_102(Q):
    z = 5.490515e-06
    return z


def neuron_103(Q):
    z = -3.41438e-06
    return z


def neuron_104(Q):
    z = -1.851988
    if Q.mass < 117.4867:
        z += -0.02296197 * Q.mass + 2.765484
    if 117.4867 <= Q.mass < 149.0507:
        z += -0.002146709 * Q.mass + 0.3199684
    if Q.pair_mean_lndelta < -1.352792:
        z += -0.5004033 * Q.pair_mean_lndelta - 0.6769414
    if Q.sjq_2_1_nch < 18.0:
        z += -0.03069993 * Q.sjq_2_1_nch + 0.5525988
    if 3.208089 <= Q.mass_displaced3 < 6.341631:
        z += 0.05585368 * Q.mass_displaced3 - 0.1791836
    if Q.mass_displaced3 >= 6.341631:
        z += 0.01279962 * Q.mass_displaced3 + 0.09384941
    if Q.n_pt_above_1 < 58.0:
        z += -0.02043607 * Q.n_pt_above_1 + 1.185292
    if Q.sj4_pair_mass_max >= 128.0079:
        z += 0.008140464 * Q.sj4_pair_mass_max - 1.042043
    if Q.tau32_b2 < 0.4680886:
        z += 0.4790914 * Q.tau32_b2 - 0.2242572
    if Q.D2 < 1.226724:
        z += -0.1071569 * Q.D2 + 0.1314519
    if Q.D2 >= 3.597891:
        z += 0.06326615 * Q.D2 - 0.2276247
    if Q.n_pairs_kt_above_3 < 34.0:
        z += 0.02281268 * Q.n_pairs_kt_above_3 - 0.775631
    if Q.nca_sj4_pair_mass_2nd >= 77.3635:
        z += 0.006792649 * Q.nca_sj4_pair_mass_2nd - 0.5255031
    if Q.pz_lnd0 < 0.2827395:
        z += -1.819209 * Q.pz_lnd0 + 0.5143624
    if Q.n_s3d_above_10 < 6.0:
        z += -0.06244056 * Q.n_s3d_above_10 + 0.3746434
    if Q.sjf_2_1_max3d < 3.32421:
        z += 0.09722641 * Q.sjf_2_1_max3d - 0.323201
    if Q.sjf_2_1_z_d3 >= 0.1149585:
        z += -0.6373032 * Q.sjf_2_1_z_d3 + 0.07326342
    if Q.sjf_3_2_max3d < 5.245219:
        z += 0.03847132 * Q.sjf_3_2_max3d - 0.2017905
    if Q.sjq_2_sumabs_k03 >= 0.6667228:
        z += -0.1187983 * Q.sjq_2_sumabs_k03 + 0.07920551
    if Q.tau3 < 0.05398263:
        z += 7.452743 * Q.tau3 - 0.4023186
    if Q.n_pairs_kt_above_1 < 146.0:
        z += 0.002084377 * Q.n_pairs_kt_above_1 - 0.304319
    if Q.sjf_3_2_maxsd0 < 61.73838:
        z += -0.001637397 * Q.sjf_3_2_maxsd0 + 0.1010902
    if Q.ak02_min12_jp < 14.23948:
        z += 0.01147893 * Q.ak02_min12_jp - 0.163454
    if Q.tau21_b2 >= 0.3898586:
        z += 0.8504038 * Q.tau21_b2 - 0.3315373
    if Q.pair_max_lnm2 >= 5.908788:
        z += 0.1018061 * Q.pair_max_lnm2 - 0.6015509
    if Q.z_charged_had < 0.4501484:
        z += 0.744006 * Q.z_charged_had - 0.3349131
    if 76.1529 <= Q.mres_sd_mass_b2z01 < 86.09683:
        z += 0.02333951 * Q.mres_sd_mass_b2z01 - 1.777371
    if Q.mres_sd_mass_b2z01 >= 86.09683:
        z += 0.0002099071 * Q.mres_sd_mass_b2z01 + 0.2140139
    if 91.82593 <= Q.mres_sd_mass_b0z005 < 159.9242:
        z += -0.004776219 * Q.mres_sd_mass_b0z005 + 0.4385808
    if Q.mres_sd_mass_b0z005 >= 159.9242:
        z += -0.000639969 * Q.mres_sd_mass_b0z005 - 0.2229059
    if Q.sj2_mass1 >= 53.51926:
        z += -0.002459522 * Q.sj2_mass1 + 0.1316318
    if Q.mass_top5 >= 17.63354:
        z += -0.002819598 * Q.mass_top5 + 0.04971951
    if Q.tau5 < 0.0230226:
        z += -28.99461 * Q.tau5 + 0.6675312
    if Q.mass_top50 < 135.5368:
        z += -0.004069038 * Q.mass_top50 + 0.5515043
    if Q.sjf_2_2_max3d < 16.96512:
        z += -0.007876142 * Q.sjf_2_2_max3d + 0.1336197
    if Q.sjq_2_2_nch < 2.0:
        z += -0.2100849 * Q.sjq_2_2_nch + 0.6880182
    if 2.0 <= Q.sjq_2_2_nch < 15.0:
        z += -0.02060372 * Q.sjq_2_2_nch + 0.3090558
    if Q.mass_charged < 99.20396:
        z += -0.003136049 * Q.mass_charged + 0.3111085
    if Q.dc_tag_2nd < 1.914834:
        z += 0.08894153 * Q.dc_tag_2nd - 0.1703082
    if Q.mass_top30 >= 126.5007:
        z += 0.00389996 * Q.mass_top30 - 0.4933477
    if Q.z_top3_slots >= 0.5395924:
        z += 1.540174 * Q.z_top3_slots - 0.8310663
    if Q.C2 < 0.1798521:
        z += -1.038779 * Q.C2 + 0.1868266
    if Q.n_pt_above_1 < 58.0 and Q.sjf_2_1_n_d5 > 1.0:
        z += -0.001596488 * (58.0 - Q.n_pt_above_1) * (Q.sjf_2_1_n_d5 - 1.0)
    if Q.mass_displaced3 > 3.208089 and Q.ak02_3_z < 0.1014193:
        z += 0.05854573 * (Q.mass_displaced3 - 3.208089) * (0.1014193 - Q.ak02_3_z)
    if Q.mass < 149.0507 and Q.sum_e < 1401.904:
        z += -6.084019e-06 * (149.0507 - Q.mass) * (1401.904 - Q.sum_e)
    if Q.n_pt_above_1 < 58.0 and Q.pair_max_lnkt > 1.555555:
        z += -0.003795898 * (58.0 - Q.n_pt_above_1) * (Q.pair_max_lnkt - 1.555555)
    if Q.lep_ptrel > 6.983043 and Q.ak02_dr23 < 0.2385164:
        z += -0.05446649 * (Q.lep_ptrel - 6.983043) * (0.2385164 - Q.ak02_dr23)
    if Q.n_pt_above_1 < 58.0 and Q.ak02_dr13 < 0.4490565:
        z += 0.02297834 * (58.0 - Q.n_pt_above_1) * (0.4490565 - Q.ak02_dr13)
    if Q.pz_lnd0 < 0.2827395 and Q.n_pairs_kt_above_10 > 1.0:
        z += -0.03492956 * (0.2827395 - Q.pz_lnd0) * (Q.n_pairs_kt_above_10 - 1.0)
    if Q.sjq_2_1_nch < 18.0 and Q.dr02 > 0.04115773:
        z += -0.023347 * (18.0 - Q.sjq_2_1_nch) * (Q.dr02 - 0.04115773)
    if Q.sjf_2_1_z_d3 > 0.1149585 and Q.ischhad_0 > 0.0:
        z += 0.5277122 * (Q.sjf_2_1_z_d3 - 0.1149585) * (Q.ischhad_0 - 0.0)
    if Q.sjf_2_2_max3d < 4.516968 and Q.e3_b2 < 0.0007909605:
        z += -101.0925 * (4.516968 - Q.sjf_2_2_max3d) * (0.0007909605 - Q.e3_b2)
    if Q.mres_sd_mass_b0z005 > 91.82593 and Q.sv_1_n < 3.0:
        z += 0.0009532369 * (Q.mres_sd_mass_b0z005 - 91.82593) * (3.0 - Q.sv_1_n)
    if Q.n_pairs_kt_above_3 < 34.0 and Q.kt2_2_n_lep < 1.0:
        z += 0.01058136 * (34.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.kt2_2_n_lep)
    if Q.n_pt_above_1 < 58.0 and Q.lepsj_2_n_d3 < 1.0:
        z += 0.005474625 * (58.0 - Q.n_pt_above_1) * (1.0 - Q.lepsj_2_n_d3)
    if Q.mass_charged < 99.20396 and Q.ak02_2_n_lep < 1.0:
        z += -0.00171806 * (99.20396 - Q.mass_charged) * (1.0 - Q.ak02_2_n_lep)
    return z


def neuron_105(Q):
    z = 2.732383e-05
    return z


def neuron_106(Q):
    z = 2.160934e-05
    return z


def neuron_107(Q):
    z = 1.15856e-05
    return z


def neuron_108(Q):
    z = -4.470701e-06
    return z


def neuron_109(Q):
    z = -5.200155e-07
    return z


def neuron_110(Q):
    z = 3.579609e-06
    return z


def neuron_111(Q):
    z = 3.669616e-06
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
    z = 1.335709
    if Q.N2_b2 < 0.2243273:
        z += -1.749764 * Q.N2_b2 + 0.39252
    if Q.mass_displaced3 < 39.09615:
        z += 0.03135528 * Q.mass_displaced3 - 1.225871
    if Q.tau2 < 0.09733903:
        z += -8.744916 * Q.tau2 + 0.8512217
    if Q.tau43 < 0.8939856:
        z += 1.298884 * Q.tau43 - 1.161184
    if Q.z_neutral_had < 0.3788785:
        z += 2.661584 * Q.z_neutral_had
    if Q.z_neutral_had >= 0.3788785:
        z += 0.4887621 * Q.z_neutral_had + 0.8232353
    if Q.pair_mean_lnkt < 0.9367772:
        z += -0.8137547 * Q.pair_mean_lnkt + 0.7623069
    if Q.ktd_ln_d34 >= -9.531553:
        z += -0.0888127 * Q.ktd_ln_d34 - 0.846523
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.01477561 * Q.n_pairs_kt_above_1 - 1.029436
    if 80.0 <= Q.n_pairs_kt_above_1 < 411.0:
        z += -0.0004610637 * Q.n_pairs_kt_above_1 + 0.1894972
    if Q.lepsj_3_n_d3 < 3.0:
        z += 0.05749263 * Q.lepsj_3_n_d3 - 0.1724779
    if Q.e3_b2 < 0.0004126585:
        z += 865.0492 * Q.e3_b2 - 0.4547006
    if 0.0004126585 <= Q.e3_b2 < 0.0007909605:
        z += 258.3405 * Q.e3_b2 - 0.2043371
    if Q.n_s3d_above_3 >= 4.0:
        z += 0.1204367 * Q.n_s3d_above_3 - 0.4817469
    if Q.mass_top50 < 94.51361:
        z += 0.01063137 * Q.mass_top50 - 1.506386
    if 94.51361 <= Q.mass_top50 < 118.8408:
        z += 0.005665428 * Q.mass_top50 - 1.037036
    if 118.8408 <= Q.mass_top50 < 161.1264:
        z += 0.008602267 * Q.mass_top50 - 1.386052
    if Q.sj4_pair_mass_max < 75.78208:
        z += -0.003933109 * Q.sj4_pair_mass_max + 0.04409095
    if 75.78208 <= Q.sj4_pair_mass_max < 101.849:
        z += 0.009742932 * Q.sj4_pair_mass_max - 0.9923079
    if Q.sip_3d_2 < 447.0873:
        z += -0.000389335 * Q.sip_3d_2 + 0.1740667
    if Q.M2 < 0.120439:
        z += -10.86372 * Q.M2 + 1.308415
    if Q.lep_z < 0.5187302:
        z += -1.234453 * Q.lep_z + 0.640348
    if Q.LHA < 0.5626523:
        z += 3.374607 * Q.LHA - 1.89873
    if Q.e3 < 0.00228569:
        z += -200.0811 * Q.e3 + 0.4573233
    if Q.n_sd0_above_3 >= 2.0:
        z += -0.08385018 * Q.n_sd0_above_3 + 0.1677004
    if Q.sj4_pair_mass_min < 25.64898:
        z += -0.007444179 * Q.sj4_pair_mass_min + 0.1909356
    if Q.pt_entropy >= 3.22434:
        z += 0.6873649 * Q.pt_entropy - 2.216299
    if Q.max_abs_dz < 6.402344:
        z += -0.01833701 * Q.max_abs_dz + 0.1173998
    if Q.mass_2charged < 1.398635:
        z += 0.187408 * Q.mass_2charged - 0.1183654
    if 1.398635 <= Q.mass_2charged < 6.767937:
        z += -0.0590807 * Q.mass_2charged + 0.2263823
    if 6.767937 <= Q.mass_2charged < 20.76537:
        z += 0.01239314 * Q.mass_2charged - 0.2573482
    if Q.sum_pt_top10 < 428.9828:
        z += -0.002934931 * Q.sum_pt_top10 + 1.259035
    if Q.tau4 >= 0.04996:
        z += -8.612072 * Q.tau4 + 0.4302591
    if Q.psi_0p3 >= 0.9756505:
        z += 8.89299 * Q.psi_0p3 - 8.67645
    if Q.sj3_pair_mass_max < 114.7848:
        z += 0.0108122 * Q.sj3_pair_mass_max - 1.241076
    if Q.n_lepton < 1.0:
        z += -0.4820306 * Q.n_lepton + 0.4820306
    if Q.lep_ptrel < 18.7678:
        z += 0.04901043 * Q.lep_ptrel - 0.919818
    if Q.mres_sd_prong_mass2 < 7.065114:
        z += 0.03015425 * Q.mres_sd_prong_mass2 + 0.02979805
    if 7.065114 <= Q.mres_sd_prong_mass2 < 21.04127:
        z += -0.01737539 * Q.mres_sd_prong_mass2 + 0.3656004
    if Q.mass < 131.3917:
        z += -0.02056806 * Q.mass + 2.702473
    z += -0.8444259 * Q.sj3_pairmax_over_m
    if Q.mass_top20 < 93.50967:
        z += -0.00580091 * Q.mass_top20 + 0.5424411
    if Q.sjf_3_3_maxsd0 < 0.3479096:
        z += -1.126605 * Q.sjf_3_3_maxsd0 + 0.3919568
    if Q.pair_max_lnkt >= 1.555555:
        z += -0.2229489 * Q.pair_max_lnkt + 0.3468092
    if Q.sum_z_dr2_top3 < 0.02745856:
        z += 6.730597 * Q.sum_z_dr2_top3 - 0.1848125
    if Q.sj3_mass3 < 5.460258:
        z += 0.04267107 * Q.sj3_mass3 - 0.232995
    if Q.lund_max_lnkt >= 3.388322:
        z += 0.1413376 * Q.lund_max_lnkt - 0.4788974
    if Q.dc_1_z < 0.932165:
        z += 0.4608213 * Q.dc_1_z - 0.4295615
    if Q.mres_sd_mass_b1z01 >= 88.79082:
        z += -0.001631567 * Q.mres_sd_mass_b1z01 + 0.1448682
    if Q.mass_displaced3 < 39.09615 and Q.lep_ptrel < 18.7678:
        z += 0.0009858569 * (39.09615 - Q.mass_displaced3) * (18.7678 - Q.lep_ptrel)
    if Q.pair_mean_lnkt < 0.9367772 and Q.sum_pt_top10 < 428.9828:
        z += -0.002638825 * (0.9367772 - Q.pair_mean_lnkt) * (428.9828 - Q.sum_pt_top10)
    if Q.lep_z < 0.5187302 and Q.tau54 < 0.8786609:
        z += -2.874781 * (0.5187302 - Q.lep_z) * (0.8786609 - Q.tau54)
    if Q.lep_z < 0.5187302 and Q.sj4_dr_min < 0.1845735:
        z += 2.085276 * (0.5187302 - Q.lep_z) * (0.1845735 - Q.sj4_dr_min)
    if Q.sip_3d_2 < 447.0873 and Q.jd_3d_4 < 9.982976:
        z += -3.883918e-05 * (447.0873 - Q.sip_3d_2) * (9.982976 - Q.jd_3d_4)
    if Q.lep_z < 0.5187302 and Q.lne_1 > 4.635336:
        z += 0.3699208 * (0.5187302 - Q.lep_z) * (Q.lne_1 - 4.635336)
    if Q.tau2 < 0.09733903 and Q.kt2_1_n_disp3 > 0.0:
        z += -1.262951 * (0.09733903 - Q.tau2) * (Q.kt2_1_n_disp3 - 0.0)
    if Q.lep_z < 0.5187302 and Q.ak02_n > 2.0:
        z += -0.1320303 * (0.5187302 - Q.lep_z) * (Q.ak02_n - 2.0)
    if Q.sj3_pair_mass_min > 59.49644 and Q.n_lund_kt_above_5 < 4.0:
        z += 0.01075762 * (Q.sj3_pair_mass_min - 59.49644) * (4.0 - Q.n_lund_kt_above_5)
    if Q.sj4_pair_mass_max < 101.849 and Q.jd_3d_6 < 9.356714:
        z += -0.0007131147 * (101.849 - Q.sj4_pair_mass_max) * (9.356714 - Q.jd_3d_6)
    if Q.mass_top50 < 118.8408 and Q.jd_3d_6 < 9.356714:
        z += 0.001046344 * (118.8408 - Q.mass_top50) * (9.356714 - Q.jd_3d_6)
    if Q.mass_top50 < 118.8408 and Q.lepsj_2_dr < 0.103005:
        z += 0.05074921 * (118.8408 - Q.mass_top50) * (0.103005 - Q.lepsj_2_dr)
    if Q.lep_z < 0.5187302 and Q.lep_iso < 14.38858:
        z += -0.02580231 * (0.5187302 - Q.lep_z) * (14.38858 - Q.lep_iso)
    if Q.max_abs_dz < 6.402344 and Q.lepsj_3_dr < 0.01012269:
        z += -1.923353 * (6.402344 - Q.max_abs_dz) * (0.01012269 - Q.lepsj_3_dr)
    if Q.ktd_ln_d34 > -9.531553 and Q.pz_lnkt3 < 0.102661:
        z += 1.038904 * (Q.ktd_ln_d34 - -9.531553) * (0.102661 - Q.pz_lnkt3)
    if Q.e3_b2 < 0.0004126585 and Q.sdb_5_z > 0.04013001:
        z += 3743.951 * (0.0004126585 - Q.e3_b2) * (Q.sdb_5_z - 0.04013001)
    if Q.lepsj_3_maxsd0 < 2.413264 and Q.sv_1_sd0_sum < 509.6197:
        z += -0.0001616268 * (2.413264 - Q.lepsj_3_maxsd0) * (509.6197 - Q.sv_1_sd0_sum)
    return z


def neuron_116(Q):
    z = -3.399317e-07
    return z


def neuron_117(Q):
    z = 2.862138e-06
    return z


def neuron_118(Q):
    z = -8.233224e-06
    return z


def neuron_119(Q):
    z = -6.336862e-07
    return z


def neuron_120(Q):
    z = 1.231577
    if Q.lep_ptrel < 3.53503:
        z += 0.1413766 * Q.lep_ptrel - 0.9129378
    if 3.53503 <= Q.lep_ptrel < 12.15228:
        z += 0.04794654 * Q.lep_ptrel - 0.5826597
    if Q.lep_ptrel >= 27.3236:
        z += -0.01309662 * Q.lep_ptrel + 0.3578468
    if Q.n_pairs_kt_above_3 < 62.0:
        z += 0.01760255 * Q.n_pairs_kt_above_3 - 1.091358
    if Q.tau32 < 0.6800935:
        z += -0.8912581 * Q.tau32 + 0.6061389
    if Q.n_pairs_kt_above_1 < 80.0:
        z += 0.008073717 * Q.n_pairs_kt_above_1 - 0.6458974
    if Q.pair_mean_lnm2 >= 4.787589:
        z += -0.270736 * Q.pair_mean_lnm2 + 1.296173
    if Q.mass_top10 >= 103.4976:
        z += -0.009078803 * Q.mass_top10 + 0.9396348
    if Q.pz_lnkt0 < 0.1589878:
        z += -5.00246 * Q.pz_lnkt0 + 0.7953298
    if Q.n_pairs_kt_above_10 < 11.0:
        z += 0.02672116 * Q.n_pairs_kt_above_10 - 0.2939327
    if Q.lep_iso < 0.4381892:
        z += 1.233712 * Q.lep_iso - 0.3547033
    if 0.4381892 <= Q.lep_iso < 2.907433:
        z += -0.07528466 * Q.lep_iso + 0.2188851
    if Q.lepsj_2_dr < 0.0008210142:
        z += -432.7734 * Q.lepsj_2_dr + 0.05484431
    if 0.0008210142 <= Q.lepsj_2_dr < 0.06573337:
        z += 4.628838 * Q.lepsj_2_dr - 0.3042691
    if Q.lep_dr < 0.08178299:
        z += -3.709732 * Q.lep_dr + 0.1110993
    if 0.08178299 <= Q.lep_dr < 0.3014662:
        z += 0.8753229 * Q.lep_dr - 0.2638803
    if Q.sjq_3_3_k1 >= 0.4037488:
        z += 0.7361726 * Q.sjq_3_3_k1 - 0.2972288
    if Q.mass_charged < 68.20711:
        z += 0.02159639 * Q.mass_charged - 1.473028
    if Q.lam1 >= 0.04304553:
        z += -4.763646 * Q.lam1 + 0.2050537
    if Q.M2 < 0.05805236:
        z += -7.21197 * Q.M2 + 0.4186719
    if Q.mass_neutral < 63.87145:
        z += -0.0113494 * Q.mass_neutral + 0.7249025
    if Q.mass < 71.96396:
        z += 0.01408069 * Q.mass - 2.371199
    if 71.96396 <= Q.mass < 90.08945:
        z += 0.03889914 * Q.mass - 4.157233
    if 90.08945 <= Q.mass < 105.7234:
        z += 0.0337185 * Q.mass - 3.690512
    if 105.7234 <= Q.mass < 114.0172:
        z += 0.01515316 * Q.mass - 1.727721
    if Q.lund3_lndelta >= -2.036501:
        z += -0.1375665 * Q.lund3_lndelta - 0.2801544
    if Q.pz_lnd2 < 0.03277088:
        z += 9.121622 * Q.pz_lnd2 - 0.2989236
    if Q.e3 >= 0.00228569:
        z += 54.39497 * Q.e3 - 0.12433
    if Q.sjf_2_2_max3d < 3.029824:
        z += 0.05062492 * Q.sjf_2_2_max3d - 0.1533846
    if Q.sj3_dr13 >= 0.7462286:
        z += -0.405988 * Q.sj3_dr13 + 0.3029599
    if Q.kt2_min12_jp < 4.912483:
        z += 0.01935287 * Q.kt2_min12_jp - 0.09507065
    if Q.ak02_1_n_lep < 1.0:
        z += 0.1557722 * Q.ak02_1_n_lep - 0.1557722
    if Q.ak02_1_n_lep >= 1.0:
        z += -0.1769827 * Q.ak02_1_n_lep + 0.1769827
    if Q.pair_mean_lndelta >= -1.790445:
        z += -0.5984899 * Q.pair_mean_lndelta - 1.071563
    if Q.pz_lnd0 < 0.05066241:
        z += -4.019132 * Q.pz_lnd0 + 0.2036189
    if Q.N2_b05 < 0.3667049:
        z += 3.679037 * Q.N2_b05 - 1.349121
    if Q.mass_top20 >= 114.4658:
        z += -0.009877954 * Q.mass_top20 + 1.130688
    if Q.mass_top50 < 135.5368:
        z += -0.005795377 * Q.mass_top50 + 1.035779
    if 135.5368 <= Q.mass_top50 < 161.1264:
        z += -0.01047901 * Q.mass_top50 + 1.670583
    if 161.1264 <= Q.mass_top50 < 178.725:
        z += -0.001026706 * Q.mass_top50 + 0.1475675
    if Q.mass_top50 >= 178.725:
        z += 0.004768671 * Q.mass_top50 - 0.8882112
    if Q.lep_z < 0.004135872:
        z += 109.3701 * Q.lep_z - 0.4523408
    if Q.mres_sd_prong_mass2 < 4.6892:
        z += 0.02464979 * Q.mres_sd_prong_mass2 - 0.1155878
    if Q.sd_mass < 78.4753:
        z += -0.002266927 * Q.sd_mass - 0.138715
    if 78.4753 <= Q.sd_mass < 88.81751:
        z += -0.01153228 * Q.sd_mass + 0.5883862
    if 88.81751 <= Q.sd_mass < 111.701:
        z += 0.01904792 * Q.sd_mass - 2.12767
    if Q.lep_ptrel < 3.53503 and Q.n_s3d_above_3 < 10.0:
        z += -0.0442463 * (3.53503 - Q.lep_ptrel) * (10.0 - Q.n_s3d_above_3)
    if Q.lep_z < 0.5187302 and Q.mass_neutral < 69.1565:
        z += -0.02353427 * (0.5187302 - Q.lep_z) * (69.1565 - Q.mass_neutral)
    if Q.lep_z < 0.5187302 and Q.mass_displaced3 > 0.0:
        z += 0.01627646 * (0.5187302 - Q.lep_z) * (Q.mass_displaced3 - 0.0)
    if Q.lep_ptrel < 3.53503 and Q.jd_3d_4 < 172.888:
        z += 0.0007705454 * (3.53503 - Q.lep_ptrel) * (172.888 - Q.jd_3d_4)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.sjf_2_1_z_d3 < 0.2710171:
        z += 0.02588022 * (62.0 - Q.n_pairs_kt_above_3) * (0.2710171 - Q.sjf_2_1_z_d3)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.lepsj_2_dr < 0.1822601:
        z += 0.02024887 * (62.0 - Q.n_pairs_kt_above_3) * (0.1822601 - Q.lepsj_2_dr)
    if Q.lep_ptrel < 3.53503 and Q.n_photon > 5.0:
        z += 0.005753839 * (3.53503 - Q.lep_ptrel) * (Q.n_photon - 5.0)
    if Q.lep_ptrel < 12.15228 and Q.sdb_2_n < 18.0:
        z += 0.00269458 * (12.15228 - Q.lep_ptrel) * (18.0 - Q.sdb_2_n)
    if Q.lep_ptrel < 12.15228 and Q.jd_3d_6 < 3.151933:
        z += -0.02516169 * (12.15228 - Q.lep_ptrel) * (3.151933 - Q.jd_3d_6)
    if Q.lep_iso < 2.907433 and Q.jd_3d_6 < 5.367501:
        z += 0.02924375 * (2.907433 - Q.lep_iso) * (5.367501 - Q.jd_3d_6)
    if Q.n_pairs_kt_above_3 < 62.0 and Q.dc_2_n_lep < 1.0:
        z += 0.004780958 * (62.0 - Q.n_pairs_kt_above_3) * (1.0 - Q.dc_2_n_lep)
    if Q.lep_ptrel < 3.53503 and Q.e4_b05 > 4.363663e-05:
        z += 420.561 * (3.53503 - Q.lep_ptrel) * (Q.e4_b05 - 4.363663e-05)
    if Q.mass_top10 > 103.4976 and Q.sum_pt_top40 > 502.3395:
        z += 2.683089e-05 * (Q.mass_top10 - 103.4976) * (Q.sum_pt_top40 - 502.3395)
    if Q.lep_ptrel < 3.53503 and Q.max_abs_d0 < 5.8125:
        z += 0.01324201 * (3.53503 - Q.lep_ptrel) * (5.8125 - Q.max_abs_d0)
    if Q.lepsj_2_dr < 0.06573337 and Q.sjq_3_3_k1 > 0.4037488:
        z += -8.012261 * (0.06573337 - Q.lepsj_2_dr) * (Q.sjq_3_3_k1 - 0.4037488)
    if Q.n_pairs_kt_above_10 < 11.0 and Q.jd_3d_6 < 3.151933:
        z += 0.01026078 * (11.0 - Q.n_pairs_kt_above_10) * (3.151933 - Q.jd_3d_6)
    if Q.lep_ptrel < 3.53503 and Q.sv_2_z > 0.0:
        z += -1.142267 * (3.53503 - Q.lep_ptrel) * (Q.sv_2_z - 0.0)
    if Q.lep_ptrel < 12.15228 and Q.n_muon > 0.0:
        z += 0.01903004 * (12.15228 - Q.lep_ptrel) * (Q.n_muon - 0.0)
    if Q.lep_iso < 0.4381892 and Q.dc_1_n_lep < 1.0:
        z += 1.140974 * (0.4381892 - Q.lep_iso) * (1.0 - Q.dc_1_n_lep)
    if Q.mass_charged < 68.20711 and Q.jd_3d_6 < 5.367501:
        z += 0.005046259 * (68.20711 - Q.mass_charged) * (5.367501 - Q.jd_3d_6)
    if Q.lepsj_2_dr < 0.06573337 and Q.jd_3d_6 < 5.367501:
        z += -1.141991 * (0.06573337 - Q.lepsj_2_dr) * (5.367501 - Q.jd_3d_6)
    if Q.M2 < 0.05805236 and Q.jd_3d_5 < 6.771002:
        z += -2.619272 * (0.05805236 - Q.M2) * (6.771002 - Q.jd_3d_5)
    if Q.mass < 114.0172 and Q.jd_3d_6 < 5.367501:
        z += 0.002296004 * (114.0172 - Q.mass) * (5.367501 - Q.jd_3d_6)
    if Q.pz_lnkt0 < 0.1589878 and Q.sjq_2_prod_k05 > -0.3383985:
        z += -4.99578 * (0.1589878 - Q.pz_lnkt0) * (Q.sjq_2_prod_k05 - -0.3383985)
    if Q.lep_ptrel < 12.15228 and Q.pz_lnd2 > 0.008992646:
        z += -0.04124402 * (12.15228 - Q.lep_ptrel) * (Q.pz_lnd2 - 0.008992646)
    if Q.mass < 114.0172 and Q.z_neutral_had < 0.1271955:
        z += 0.07607861 * (114.0172 - Q.mass) * (0.1271955 - Q.z_neutral_had)
    if Q.lep_z < 0.5187302 and Q.sjq_3_3_k1 < -0.2912597:
        z += 0.8656015 * (0.5187302 - Q.lep_z) * (-0.2912597 - Q.sjq_3_3_k1)
    if Q.lep_ptrel < 12.15228 and Q.sdb_3_n < 5.0:
        z += 0.002387767 * (12.15228 - Q.lep_ptrel) * (5.0 - Q.sdb_3_n)
    if Q.pz_lnd0 < 0.05066241 and Q.dr_19 < 0.0538419:
        z += 264.9319 * (0.05066241 - Q.pz_lnd0) * (0.0538419 - Q.dr_19)
    return z


def neuron_121(Q):
    z = -5.73052e-06
    return z


def neuron_122(Q):
    z = -2.395902e-06
    return z


def neuron_123(Q):
    z = 0.0191935
    return z


def neuron_124(Q):
    z = -0.1131311
    return z


def neuron_125(Q):
    z = -1.190763e-06
    return z


def neuron_126(Q):
    z = 7.632632e-06
    return z


def neuron_127(Q):
    z = -1.966697e-07
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
