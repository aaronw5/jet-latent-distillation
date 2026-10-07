"""E2 / E3 — how much of ParT can a function of each particle's own inputs (+ jet quantities) [+ its neighbourhood]
reproduce, through ParT's own frozen class blocks? A per-particle MLP gives each particle's 128-dim embedding; trained
toward ParT's probabilities. The ceiling of the class-fit route for a given set of particle inputs.

  python -m jetdistill.research.ceiling_cls own|nbr [n_fit epochs]"""
import json, sys, time, pathlib
import numpy as np
from ..pipeline import jets
from ..part.clsfit import particle_features, cls_forward
from ..part.network import ParTNetwork
from .nbr import nbr_features

OUT = pathlib.Path(__file__).resolve().parents[2] / 'research' / 'results'


def feats(J, n, kind):
    F, ok = particle_features(J, np.arange(n))
    if kind == 'nbr': F = np.concatenate([F, nbr_features(J['x'][:n], J['ext'][:n], J['jet'][:n])], -1)
    return F, ok


def run(kind='own', n_fit=200000, epochs=8, width=256, bs=256, device='mps', log=print):
    import torch
    t0 = time.time(); Jf, Jd = jets('full', 'fit'), jets('full', 'dev')
    Ff, okf = feats(Jf, n_fit, kind); Fd, okd = feats(Jd, len(Jd['y']), kind)
    mu, sd = Ff[okf].mean(0), Ff[okf].std(0) + 1e-6; Ff = ((Ff - mu) / sd) * okf[..., None]; Fd = ((Fd - mu) / sd) * okd[..., None]
    log(f'{kind}: {Ff.shape[-1]} inputs per particle, {n_fit} fitting jets, features in {time.time() - t0:.0f} s')
    T = lambda a, dt=torch.float16: torch.from_numpy(a).to(device, dt)
    xf, mf, pf = T(Ff), torch.from_numpy(okf).to(device), T(np.asarray(Jf['P'][:n_fit], np.float32), torch.float32)
    xd, md = T(Fd), torch.from_numpy(okd).to(device); del Ff, Fd
    model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    net = torch.nn.Sequential(torch.nn.Linear(xf.shape[-1], width), torch.nn.GELU(), torch.nn.Linear(width, width), torch.nn.GELU(),
                              torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, 128)).to(device)
    steps = epochs * (n_fit // bs); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    def fwd(x, m):
        P = int(m.sum(1).max()); x, m = x[:, :P].float(), m[:, :P]; return cls_forward(model, net(x) * m[..., None], m)[1]
    def val():
        net.eval(); o = np.argsort(okd.sum(1)); pred = np.empty(len(o), int)
        with torch.no_grad():
            for a in range(0, len(o), 2000):
                i = torch.from_numpy(o[a:a + 2000]).to(device); pred[o[a:a + 2000]] = fwd(xd[i], md[i]).argmax(1).cpu().numpy()
        net.train(); return float((pred == Jd['net']).mean()), float((pred == Jd['y']).mean())
    best, hist = (0, 0), []
    for ep in range(epochs):
        perm = torch.randperm(n_fit, device=device)
        for a in range(0, n_fit - bs + 1, bs):
            i = perm[a:a + bs]; loss = -(pf[i] * torch.log_softmax(fwd(xf[i], mf[i]), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        s = val(); hist.append(s); best = max(best, s); torch.mps.empty_cache()
        log(f'  {kind} epoch {ep + 1}: validation same class as ParT {100 * s[0]:.2f}%, accuracy {100 * s[1]:.2f}%, {time.time() - t0:.0f} s')
    return dict(experiment='E2' if kind == 'own' else 'E3', kind=kind, n_inputs=int(xf.shape[-1]), n_fit=n_fit, epochs=epochs, width=width,
                agreement=best[0], accuracy=best[1], history=hist, seconds=time.time() - t0)


if __name__ == '__main__':
    kind = sys.argv[1]; a = [int(v) for v in sys.argv[2:]]
    r = run(kind, *a); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{r["experiment"]}_{kind}_{r["n_fit"]}.json').write_text(json.dumps(r, indent=1)); print('RESULT', json.dumps(r))
