"""A2 — the class-token fit's neurons (S15q) on 100,000 validation jets: each neuron's change of every class score per
standard deviation (last-layer weight × spread), R² against ParT's own neuron, and the neuron's largest inputs.
Saved to research/results/A2_neurons_100k.json."""
import json, numpy as np
from jetdistill.pipeline import jets
from jetdistill.part.network import ParTNetwork
from jetdistill.research.heads import extract, phi_pooled, rows_of_split, OUT
from jetdistill.research.one_term import extend
n = 100000; model = ParTNetwork('full').model; model.eval(); J = jets('full', 'dev'); rows = rows_of_split('dev', n)
_, A, M, L, _ = extract(model, 'dev', n, 'mps'); P = np.load(OUT / 'S15q_model.npz'); W = P['W'].reshape(16, -1, 128)
Pd, _ = phi_pooled(J, rows, A, P['kn']); E = extend(Pd, P['knv'], 38); V = np.einsum('nhk,hko->no', E, W); H = np.asarray(J['H'][rows], np.float32)
FW = model.fc[0].weight.detach().cpu().numpy(); FB = model.fc[0].bias.detach().cpu().numpy()
agree = float(((V @ FW.T + FB).argmax(1) == L.argmax(1)).mean()); sd = V.std(0)
act = [o for o in range(128) if np.any(W[:, :-2, o])]
out = dict(n_jets=n, agreement=agree, neurons=[dict(n=o, sd=float(sd[o]), effect=(FW[:, o] * sd[o]).tolist(), strength=float(np.abs(FW[:, o]).mean() * sd[o]),
                                                   r2=float(1 - ((V[:, o] - H[:, o]) ** 2).mean() / H[:, o].var())) for o in act])
(OUT / 'A2_neurons_100k.json').write_text(json.dumps(out)); print(f'agreement on {n} jets {100 * agree:.2f}%, {len(act)} active neurons')
