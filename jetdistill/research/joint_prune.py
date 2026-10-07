"""Pruning the all-formula class attention (S10 model): which per-particle features does it need? Ablation per feature
(its terms zeroed in the score and value formulas; the agreement drop on the balanced dev sample), features whose drop
is below TOL removed together, the rest re-tuned (joint.run's loop, from the pruned coefficients); repeated until no
feature can go. Saves research/results/S10_pruned.npz (+ the kept features) and the path.

  python -m jetdistill.research.joint_prune [tol_pt epochs]"""
import json, sys, time, pathlib
import numpy as np
from ..pipeline import jets
from ..part.clsfit import PFEAT
from ..part.network import ParTNetwork
from .nbr import NBR, PK
from .heads import extract, downstream, rows_of_split
from .joint import Model, all_feats, OUT, NV

NAMES = PFEAT + NBR + PK                                                     # the 126 per-particle features


def term_index(nf, kn_rows):
    """for each feature f: the columns of the score basis (1, F, 5 hinges) and of the value basis (F, 5 hinges; first NV features only)"""
    sc = {f: [1 + f] + [1 + nf + k * nf + f for k in range(kn_rows)] for f in range(nf)}
    vl = {f: ([f] + [NV + k * NV + f for k in range(kn_rows)]) if f < NV else [] for f in range(nf)}
    return sc, vl


def run(tol=0.1, epochs=30, n_fit=100000, n_dev=20000, lr=3e-4, lam=0.01, chunk=2000, device='mps', log=print):
    import torch
    t0 = time.time(); Pz = dict(np.load(OUT / 'S10_model.npz')); model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    Of, _, Mf, Lf, _ = extract(model, 'fit', n_fit, device); _, _, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rows_f, rows_d = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    Ff, _ = all_feats(Jf, rows_f, model); Fd, _ = all_feats(Jd, rows_d, model); log(f'  features, {time.time() - t0:.0f} s')
    mdl = Model(dict(kn=Pz['kns'], mu=Pz['mus'], sd=Pz['sds']), Pz['knv'], Pz['muv'], Pz['sdv'], device, Pz.get('ms'), Pz.get('mv'))
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)
    Fft, Fdt, Mft, Mdt = torch.from_numpy(Ff).to(device), torch.from_numpy(Fd).to(device), torch.from_numpy(Mf).to(device), torch.from_numpy(Md).to(device); del Ff, Fd
    Ws, Wv, cs, b = (torch.nn.Parameter(T(Pz[k])) for k in ('Ws', 'Wv', 'cself', 'bias')); nf = len(NAMES); sc, vl = term_index(nf, len(Pz['kns']))
    def agree(ws, wv):
        with torch.no_grad():
            pred = np.concatenate([downstream(model, *mdl.heads(Fdt[a:a + chunk].float(), Mdt[a:a + chunk], ws, wv, cs, b).split(8, 1)).argmax(1).cpu().numpy() for a in range(0, n_dev, chunk)])
        return float((pred == ref).mean())
    keep = np.ones(nf, bool); pf = torch.softmax(T(Lf), 1); Oft = T(Of.transpose(1, 0, 2, 3).reshape(n_fit, 16, 16)); vo = Oft.var(0) + 1e-6
    a_now = agree(Ws, Wv); a_ref = a_now; path = [dict(features=int(keep.sum()), agreement=a_now)]; log(f'  start: {nf} features, {100 * a_now:.2f}%')
    for rnd in range(6):
        drops = {}
        for f in np.flatnonzero(keep):
            ws, wv = Ws.detach().clone(), Wv.detach().clone(); ws[sc[f]] = 0; wv[:, vl[f]] = 0
            drops[f] = a_now - agree(ws, wv)
        order = sorted(drops, key=drops.get); log('  least needed: ' + ', '.join(f'{NAMES[f]} ({100 * drops[f]:+.2f})' for f in order[:8]))
        # remove, from the least needed, as many as keep the summed drop under the tolerance (then re-tune)
        cum, rem = 0.0, []
        for f in order:
            if cum + max(drops[f], 0) > tol / 100: break
            cum += max(drops[f], 0); rem.append(f)
        if not rem: break
        with torch.no_grad():
            for f in rem: Ws[sc[f]] = 0; Wv[:, vl[f]] = 0; keep[f] = False
        msk_s = torch.ones_like(Ws); msk_v = torch.ones_like(Wv)
        for f in np.flatnonzero(~keep): msk_s[sc[f]] = 0; msk_v[:, vl[f]] = 0
        opt = torch.optim.Adam([Ws, Wv, cs, b], lr)
        for ep in range(epochs):
            for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, chunk)):
                o = mdl.heads(Fft[a:a + chunk].float(), Mft[a:a + chunk], Ws * msk_s, Wv * msk_v, cs, b); Lg = downstream(model, o[:, :8], o[:, 8:])
                loss = -(pf[a:a + chunk] * torch.log_softmax(Lg, 1)).sum(1).mean() + lam * ((o - Oft[a:a + chunk]) ** 2 / vo).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if device == 'mps': torch.mps.empty_cache()
        with torch.no_grad(): Ws.mul_(msk_s); Wv.mul_(msk_v)
        a_now = agree(Ws, Wv); path.append(dict(features=int(keep.sum()), removed=[NAMES[f] for f in rem], agreement=a_now))
        log(f'  round {rnd + 1}: removed {len(rem)} features → {int(keep.sum())} kept, re-tuned: {100 * a_now:.2f}% (start {100 * a_ref:.2f}%), {time.time() - t0:.0f} s')
        if a_now < a_ref - tol / 100: log('  below the tolerance: stop (this round kept)'); break
    np.savez(OUT / 'S10_pruned.npz', Ws=Ws.detach().cpu().numpy(), Wv=Wv.detach().cpu().numpy(), cself=cs.detach().cpu().numpy(), bias=b.detach().cpu().numpy(),
             kns=Pz['kns'], mus=Pz['mus'], sds=Pz['sds'], knv=Pz['knv'], muv=Pz['muv'], sdv=Pz['sdv'], keep=keep)
    r = dict(tol=tol, epochs=epochs, kept=[NAMES[f] for f in np.flatnonzero(keep)], removed=[NAMES[f] for f in np.flatnonzero(~keep)], path=path, final=a_now, seconds=time.time() - t0)
    (OUT / 'S10_pruned.json').write_text(json.dumps(r, indent=1)); log(f'pruned: {int(keep.sum())} of {nf} features kept, {100 * a_now:.2f}%'); return r


if __name__ == '__main__':
    a = sys.argv[1:]; run(float(a[0]) if a else 0.1, int(a[1]) if len(a) > 1 else 30)
