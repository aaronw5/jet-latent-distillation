# Slides

The talk as it is in the slides app (claude.ai artifact, private): `deck.json` (title, order, sections) and one
`slides/<id>.html` per slide, each a 1920×1080 `<section>`. Images are referenced as `/_blob/<id>` in the slides and stored
here as `images/<id>.png`.

- `python slides/build_html.py` writes `deck.html`, the whole deck as one page that opens in a browser.
- Figures: `python -m jetdistill.figures <name> [args]` writes `results/figures/<name>….png` (the table below gives the
  arguments). Numbers on the slides are on the whole test file (260,000 jets), from the results of the pipeline.
- Screenshots of the site pages: `slides/screenshots/shoot.sh site/all/n8 out/` (headless Chrome; the steps are in
  `screenshots/steps_*.json` and pick formula 399, the smaller version tuned on the network, 8 particles).
- Equation images: rendered with matplotlib mathtext (`figures equations` is the one on the jet-quantities slide; the
  others show the formulas written in their alt text).

Quantities are written as formulas (ΣzΔR², λ₁, τ₂₁(β=2), …; zᵢ = pTᵢ / ΣpT); the table of symbols is
`jetdistill/observables/symbols.py` and the site's `quantities.html`.

| slide | image | made by | content (alt text) |
|---|---|---|---|
| nest | `images/36a055f6fea3b4f6901a9833208524d9.png` | equation image | h = f_L( ... f_2( f_1( {(pT_i, eta_i, phi_i)} ) ) ... ), with f_k(u) = max(0, W_k u + b_k); s = K^T h + b; cla… |
| motivgrid | `images/b336a158e898777f6fd9d46f2fa1bd9f.png` | `figures linear 64` | Four network neurons (64 particles) against one observable each with a straight line fitted to the jets in the… |
| motiv | `images/d0f07eb21667e51088d7ad0f882ab142.png` | `figures hinge 8` | The network's neuron 3 against the pT-weighted mean Delta R for all test jets: exactly 0 below 0.067, rising a… |
| motiv | `images/18a397ef5518d07fd9d9a83559ac6d47.png` | screenshot, JEDI-linear explorer page | Network neuron 10 against m/sum pT: near zero below 0.024, rising above |
| motiv | `images/84e201d24ff047be9ef901fe80636938.png` | screenshot, JEDI-linear explorer page | Network neuron 2 against jet mass: decreasing below 67 GeV, flat above |
| select | `images/f7f4de08407963371d95f92271ea5fd6.png` | `figures mars_steps 8` (left panel) | First step of the search for network neuron 3 against ln(total pT): candidate hinges at many thresholds (dotte… |
| step2 | `images/f753d7f2b080bf20448904116d9decad.png` | equation image | h_hat_j = max(0, z_hat_j); p_c = softmax(b_c + sum_j K_jc h_hat_j); L_network = -(1/N) sum_n sum_c p_net_c(n) … |
| step3 | `images/ca35127da495abab30af702a1db5d87b.png` | equation image | I_m = /beta_m/ times the standard deviation of B_m over jets; keep the cut if A_val >= A_val_ref - 0.1%, else … |
| step4 | `images/2d05bd9abd5a10dbd4c23bb649833511.png` | equation image | minimize sum_n w_nj (z_hat_j(n) - z_big_j(n))^2 with w_nj = K_j^T (diag p(n) - p(n) p(n)^T) K_j |
| eval | `images/00bf865da631a0187e7d4a48db7533c3.png` | equation image | agreement = fraction of test jets where the formula's class equals the network's class; accuracy = fraction wh… |
| targets | `images/11835e5aecc4125b0f55842bfce5c5f6.png` | `figures targets` | Agreement with the network for four tuning targets. 8 particles, tuned and smaller: class probabilities 90.5 a… |
| control | `images/c00a663b3296a20cfffc5c6feca4a11f.png` | `figures control` | Left: median R² of step 1 per neuron, trained against untrained: 0.89 and 0.55 (8 particles), 0.92 and 0.16 (6… |
| controlin | `images/a183bad93222750e208649088afe3367.png` | `figures inputs` | Median R² per neuron: MARS on physics observables 0.89 (8 particles) and 0.92 (64), MARS on raw inputs 0.58 an… |
| tool | `images/1ca2f19541b4b3bbd65c4828f1b8aa22.png` | screenshot `setup_top` | Top of the 8-particle page: title, tabs, how to read the explanation tab, and the buttons for every formula |
| match | `images/82c969a7ec9d527d339abaed50b8e38e.png` | screenshot `n10_match` | 2D histogram of the formula's neuron 10 against the network's neuron 10; points cluster along the diagonal |
| match | `images/745147ccc9605a1090869c86c4520270.png` | screenshot `n5_match` | 2D histogram of the formula's neuron 5 against the network's neuron 5 |
| n10 | `images/aba341a20c6803608dad75ad8038002c.png` | screenshot `n10_terms8` | First eight if-statements of neuron 10, each with the true-class mix of the jets that pass it |
| n10term | `images/91fa6ad85fb7662ac9e64d182df5cc5b.png` | `figures term all 8` | Left: what the if-statement adds to neuron 10 as a function of ΣzΔR²: 0 below 0.0075, then rising linearly wit… |
| n10use | `images/5b8d343d6b8a88e295e660b617277238.png` | screenshot, neuron 10 section 2 (`n10_sec2`) | Average amount neuron 10 adds to each class score, overall and per true class: t positive, q negative, g W Z z… |
| n10groups | `images/769ce900cd4c1405c3eedd59d1973e63.png` | screenshot `n10_groups_top` | The 10 main groups of jets for neuron 10, each with a physics name, share of jets, mean neuron value, class mi… |
| n10groups | `images/d2e6b6d669ad3b8203bc8b8752c0d64c.png` | equation image | a_n = (beta_1 B_1(x_n), ..., beta_M B_M(x_n)); k-means with 10 clusters on the a_n |
| n5 | `images/cca097ac0bc30704a4c1f32f94098f70.png` | screenshot `n5_groups_top` | The 10 main groups of jets for neuron 5 with their if-statement chips and class mix |
| massnet | `images/3804793f576eb2989069fa17e3b38290.png` | `figures massnet 64` | Mean value of network neurons 0, 7 and 10 against jet mass for all 260,000 test jets. Neuron 0 has its maximum… |
| allift | `images/9b2431900c39a7d3b210dc94295df8c2.png` | screenshot `n13_dist` | Distribution of neuron 13 for each true class: top jets low, all others high |
| ex1 | `images/1f61d6a817fcf9fbed3f52736b3cebb9.png` | screenshot `tj_t_main` | A top jet in the detector plane with three prongs, the five class scores (t largest at 2.94) and the three neu… |
| ex1n10 | `images/e9ce24ab9352ce7327950dae9630875f.png` | screenshot `tj_t_n10` | Neuron 10 for the top jet: every if-statement with the jet's value, whether it passes and what it adds; the su… |
| ex23 | `images/a1e2d7841eb360da179228ec83842d11.png` | screenshot `tj_W_main` | A W jet: two close hard particles, W score 3.24 largest, Z second at 1.82 |
| ex23 | `images/135c5f8aee9368cd51ab616c5d967914.png` | screenshot `tj_q_main` | A quark jet: narrow, few hard particles, q score 2.01 largest, g second at 1.09 |
| fidelity | `images/9c33bd507392c7d0388eb0c652271dca.png` | `figures fidelity 8` | Left: mean value of neuron 6 against sum pT_i Delta R_i^2 / sum pT_i for the network, the step-1 formula, the … |
| fidelity2 | `images/f8125c49e2fb25d1ae3fa45efa578cc1.png` | `figures neuron_hist 8 0` | 2D histograms of formula neuron against network neuron for neurons 4, 6, 8, 13, 15 and 2. Lambda 0: R squared … |
| fidelity3 | `images/372a8c2457b92b5e8aebea7c6afbf881.png` | `figures neuron_hist 8 0.01 all` | 2D histograms of formula neuron against network neuron for all 16 neurons with lambda 0.01; all follow the dia… |
| converge | `images/fe3cc7d1fbea2f604eab2f6ca0876e3b.png` | `figures convergence` | Agreement with the network against Adam step, training and validation jets, 8 and 64 particles; validation agr… |
| allifw | `images/1f2f31b83e18490642f1d4e246a2f9cc.png` | screenshot, 64-particle page, formula 150, neuron 12 if-statements | All 6 if-statements of neuron 12 of the 150 formula, each with the class mix of the jets that pass it |
| quant | `images/d8967859f542d83ed79b363c1c53c34c.png` | `figures equations` | Definitions: z_i = pT_i / sum pT; m^2 = (sum p_i)^2; r_i = (Delta eta_i, Delta phi_i); sum z Delta R^2 = sum_i… |
| massno | `images/ea14c9da521d35da7bd457d55966083f.png` | `figures mass_nomass 64 nomass` | Mean value of network and formula neurons 0 and 7 against jet mass; both pairs peak at 79.5 GeV (neuron 0, nea… |
| mass | `images/09a20869a02daac3db9068d770c8d6e9.png` | `figures mass_nomass 64 all <tag> 7` | Mean value of network neuron 7 and of formula neuron 7 against jet mass; both peak at the Z mass (network 2.6,… |
| setups | `images/96c8df30ee662c6f83b31ac9818a3168.png` | `figures setups` | Agreement with the network for four sets of observables. 8 particles, tuned and smaller: 90.5 and 90.1 (all), … |
