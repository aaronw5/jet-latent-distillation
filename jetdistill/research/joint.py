"""S10 — the all-formula class attention, tuned jointly. Per jet, from each particle's physics F (18 own inputs, 20
neighbourhood features, 88 ParT pair-kernel context features):
  block 1, head h:  score s_hi = φs(F_i)·ws_h  (φs: 1, F, hinges at 5 thresholds: 757 terms), self score 0,
                    α_h = softmax over [self, particles];  o_1h = Σ_i α_hi φv(F_i)·Wv_1h + α_h,self·c_1h + b_1h
                    (φv: own + neighbourhood features and their hinges: 228 terms)
  block 2, head h:  uniform α (1/(n+1)); o_2h as above
  then ParT's own exact downstream (heads.downstream) → class scores
Start: the least-squares score formulas (heads.formula_weights) and least-squares values; then Adam on all coefficients
(whitened terms) toward ParT's probabilities + λ·R on the head outputs. Saves the model (research/results/S10_model.npz).

  python -m jetdistill.research.joint [n_fit n_dev epochs]"""
import json, sys, time, pathlib
import numpy as np
from ..config import RESULTS
from ..pipeline import jets
from ..part.clsfit import particle_features
from ..part.network import ParTNetwork
from .nbr import nbr_features, pk_features
from .heads import extract, downstream, formula_weights, rows_of_split

OUT = pathlib.Path(__file__).resolve().parents[2] / 'research' / 'results'
import os
NV = int(os.environ.get('JOINT_NV', 38))                                    # features in the value basis: 38 (own + neighbourhood) or 126 (+ pair-kernel context)


def feats_rows(J, r, model):
    F, ok = particle_features(J, r, ctx=[])
    return np.concatenate([F, nbr_features(J['x'][r], J['ext'][r], J['jet'][r]), pk_features(J['x'][r], J['ext'][r], J['jet'][r], model)], -1), ok


def all_feats(J, rows, model, chunk=4000):
    F = None; n = len(rows)
    for a in range(0, n, chunk):
        r = rows[a:a + chunk]; f, ok = feats_rows(J, r, model)
        if F is None: F = np.zeros((n,) + f.shape[1:], np.float16); M = np.zeros((n, 128), bool)
        F[a:a + len(r)] = f; M[a:a + len(r)] = ok
    return F, M


class Model:
    """the formula class attention on the GPU; parameters in whitened coordinates. Hinge columns that are degenerate on the fitting
    particles (a repeated threshold, or one at the feature's clip value: the term is constant) are masked out (masks ms, mv)."""
    def __init__(self, prm, knv, muv, sdv, device, ms=None, mv=None):
        import torch
        self.t = torch; T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)
        self.kns, self.mus, self.sds = T(prm['kn']), T(prm['mu']), T(prm['sd']); self.knv, self.muv, self.sdv = T(knv), T(muv), T(sdv); self.device = device
        self.ms = None if ms is None else T(ms); self.mv = None if mv is None else T(mv)
    def phis(self, F):                                                       # (…, 757) score basis, as heads.formula_weights
        t = self.t; Fs = (F - self.mus) / self.sds
        out = t.cat([t.ones_like(F[..., :1]), Fs] + [t.clamp(F - k, min=0) / self.sds for k in self.kns], -1)
        return out if self.ms is None else out * self.ms
    def phiv(self, F):                                                       # (…, 228) value basis: own + nbr features and hinges
        t = self.t; Fv = F[..., :NV]; Fs = (Fv - self.muv) / self.sdv
        out = t.cat([Fs] + [t.clamp(Fv - k, min=0) / self.sdv for k in self.knv], -1)
        return out if self.mv is None else out * self.mv
    def heads(self, F, m, Ws, Wv, cself, bias):
        """F (n, P, 126), m (n, P) → head outputs (n, 16, 16)"""
        t = self.t; n = F.shape[0]; ps, pv = self.phis(F), self.phiv(F)
        s = (ps @ Ws).masked_fill(~m[..., None], -1e9)                        # (n, P, 8) block-1 scores
        a1 = t.softmax(t.cat([t.zeros(n, 1, 8, device=F.device), s], 1), 1)   # (n, 1+P, 8)
        cnt = m.sum(1, keepdim=True).float() + 1; a2 = (m.float() / cnt)[..., None].expand(-1, -1, 8); self2 = (1 / cnt).expand(-1, 8)
        pooled = t.cat([t.einsum('nph,npk->nhk', a1[:, 1:], pv), t.einsum('nph,npk->nhk', a2, pv)], 1)         # (n, 16, K)
        selfw = t.cat([a1[:, 0], self2], 1)                                   # (n, 16)
        return t.einsum('nhk,hko->nho', pooled, Wv) + selfw[..., None] * cself + bias


def run(n_fit=100000, n_dev=20000, epochs=100, lr=3e-4, lam=0.01, chunk=2000, device='mps', log=print, tag=os.environ.get('JOINT_TAG', ''),
        ridge=float(os.environ.get('JOINT_RIDGE', 1e-7)), l1=float(os.environ.get('JOINT_L1', 0)), ev_floor=float(os.environ.get('JOINT_EVFLOOR', 1e-6))):
    import torch
    t0 = time.time(); model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    Of, Af, Mf, Lf, _ = extract(model, 'fit', n_fit, device); Od, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rows_f, rows_d = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev); n_fit, n_dev = len(rows_f), len(rows_d)
    A1f, prm = formula_weights(model, Jf, rows_f, Af[0], Mf, log=log)        # the score formulas (least squares) and block-1 weights from them
    Ff, _ = all_feats(Jf, rows_f, model); Fd, _ = all_feats(Jd, rows_d, model); log(f'  features of {n_fit} + {n_dev} jets, {time.time() - t0:.0f} s')
    Fv = Ff[:, :, :NV][Mf]; knv = np.quantile(Fv[::7].astype(np.float32), np.linspace(.15, .85, 5), axis=0).astype(np.float32); muv, sdv = Fv.mean(0).astype(np.float32), Fv.std(0).astype(np.float32) + 1e-6; del Fv
    def live(Fp, kn, nf_):                                                 # columns of [.., features, hinges] that vary on the fitting particles (with 1 for the score basis)
        cols = [Fp.std(0) > 1e-6] + [((np.maximum(0, Fp - k)).std(0) > 1e-6) & (np.maximum(0, Fp - k).mean(0) > 1e-6) for k in kn]; return np.concatenate(cols).astype(np.float32)
    Fs_ = Ff[Mf][::9].astype(np.float32); ms = np.concatenate([[1.0], live(Fs_, prm['kn'], None)]).astype(np.float32); mv = live(Fs_[:, :NV], knv, None); del Fs_
    log(f'  score basis: {int(ms.sum())} of {len(ms)} columns live; value basis: {int(mv.sum())} of {len(mv)}')
    mdl = Model(prm, knv, muv, sdv, device, ms, mv); T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)
    Fft, Fdt = torch.from_numpy(Ff).to(device), torch.from_numpy(Fd).to(device); Mft, Mdt = torch.from_numpy(Mf).to(device), torch.from_numpy(Md).to(device); del Ff, Fd
    # least-squares start of the values: pooled bases with the start weights (block 1 formula, block 2 uniform)
    Ws0 = T(prm['W']); Of_ = Of.transpose(1, 0, 2, 3).reshape(n_fit, 16, 16)
    with torch.no_grad():
        Ps, Ys = [], []
        for a in range(0, n_fit, chunk):
            F, m = Fft[a:a + chunk].float(), Mft[a:a + chunk]; n = F.shape[0]; pv = mdl.phiv(F)
            s = (mdl.phis(F) @ Ws0).masked_fill(~m[..., None], -1e9); a1 = torch.softmax(torch.cat([torch.zeros(n, 1, 8, device=device), s], 1), 1)
            cnt = m.sum(1, keepdim=True).float() + 1; a2 = (m.float() / cnt)[..., None].expand(-1, -1, 8)
            pooled = torch.cat([torch.einsum('nph,npk->nhk', a1[:, 1:], pv), torch.einsum('nph,npk->nhk', a2, pv)], 1); selfw = torch.cat([a1[:, 0], (1 / cnt).expand(-1, 8)], 1)
            Ps.append(torch.cat([pooled, selfw[..., None], torch.ones(n, 16, 1, device=device)], -1).cpu().numpy())
        P = np.concatenate(Ps); del Ps
    K = P.shape[-1]; Wv0 = np.zeros((16, K, 16)); Ms = np.zeros((16, K - 2, K - 2), np.float32); Us = np.zeros((16, K - 2, 16), np.float32)
    for h in range(16):
        B = P[:, h].astype(np.float64); G = B.T @ B / len(B); d = np.sqrt(np.maximum(np.diag(G), 1e-12))
        Wv0[h] = np.linalg.solve(G / d[:, None] / d[None] + ridge * np.eye(K), (B.T @ Of_[:, h] / len(B)) / d[:, None]) / d[:, None]
        Gp = G[:K - 2, :K - 2] / d[:K - 2, None] / d[None, :K - 2]; ev, Q = np.linalg.eigh(Gp); ev = np.maximum(ev, ev_floor * ev.max())
        Ms[h] = Q / np.sqrt(ev)[None] / d[:K - 2, None]; Us[h] = np.sqrt(ev)[:, None] * (Q.T @ (d[:K - 2, None] * Wv0[h, :K - 2]))
    del P
    # whitening of the score basis too (on a particle sample)
    with torch.no_grad():
        ps = mdl.phis(Fft[:3000].float())[Mft[:3000]]; ps = ps[::5].cpu().double().numpy(); Gs = ps.T @ ps / len(ps); ds = np.sqrt(np.maximum(np.diag(Gs), 1e-12))
    ev, Q = np.linalg.eigh(Gs / ds[:, None] / ds[None]); ev = np.maximum(ev, ev_floor * ev.max()); Mss = T(Q / np.sqrt(ev)[None] / ds[:, None]); Us_s = T(np.sqrt(ev)[:, None] * (Q.T @ (ds[:, None] * prm['W'].astype(np.float64))))
    Mvt = T(Ms); Uv = torch.nn.Parameter(T(Us)); Us_p = torch.nn.Parameter(Us_s); cself = torch.nn.Parameter(T(Wv0[:, K - 2])); bias = torch.nn.Parameter(T(Wv0[:, K - 1]))
    params = [Uv, Us_p, cself, bias]; opt = torch.optim.Adam(params, lr); pf = torch.softmax(T(Lf), 1); Oft = T(Of_); vo = Oft.var(0) + 1e-6
    def fwd(F, m): return mdl.heads(F, m, Mss @ Us_p, torch.einsum('hkj,hjo->hko', Mvt, Uv), cself, bias)
    def agree():
        with torch.no_grad():
            pred = np.concatenate([downstream(model, *fwd(Fdt[a:a + chunk].float(), Mdt[a:a + chunk]).split(8, 1)).argmax(1).cpu().numpy() for a in range(0, n_dev, chunk)])
        return float((pred == ref).mean())
    a0 = agree(); best = (a0, 0); path = [(0, a0)]; log(f'  least squares (formula block-1 weights, uniform block 2, formula values): {100 * a0:.2f}%, {time.time() - t0:.0f} s')
    def save():
        np.savez(OUT / f'S10{tag}_model.npz', Ws=(Mss @ Us_p).detach().cpu().numpy(), Wv=torch.einsum('hkj,hjo->hko', Mvt, Uv).detach().cpu().numpy(), cself=cself.detach().cpu().numpy(), bias=bias.detach().cpu().numpy(),
                 kns=prm['kn'], mus=prm['mu'], sds=prm['sd'], knv=knv, muv=muv, sdv=sdv, ms=ms, mv=mv)
    OUT.mkdir(parents=True, exist_ok=True); save()
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, chunk)):
            o = fwd(Fft[a:a + chunk].float(), Mft[a:a + chunk]); Lg = downstream(model, o[:, :8], o[:, 8:])
            loss = -(pf[a:a + chunk] * torch.log_softmax(Lg, 1)).sum(1).mean() + lam * ((o - Oft[a:a + chunk]) ** 2 / vo).mean()
            if l1: loss = loss + l1 * ((Mss @ Us_p).abs().mean() + torch.einsum('hkj,hjo->hko', Mvt, Uv).abs().mean())     # sparse, non-cancelling raw coefficients
            opt.zero_grad(); loss.backward(); opt.step()
        if device == 'mps': torch.mps.empty_cache()
        if ep % 5 == 4 or ep == epochs - 1:
            with torch.no_grad(): ws_raw = (Mss @ Us_p).abs(); log(f'    raw score coefficients: mean |c| {ws_raw.mean():.3f}, max {ws_raw.max():.1f}, share < 0.01: {100 * (ws_raw < .01).float().mean():.0f}%')
            ag = agree(); path.append((ep + 1, ag))
            if ag > best[0]: best = (ag, ep + 1); save()
            log(f'  S10 epoch {ep + 1}: same class as ParT {100 * ag:.2f}%, {time.time() - t0:.0f} s')
    return dict(experiment='S10' + tag, nv=NV, ridge=ridge, l1=l1, ev_floor=ev_floor, n_fit=n_fit, n_dev=n_dev, epochs=epochs, lr=lr, lam=lam, least_squares=a0, best=best[0], best_epoch=best[1], path=path, score_r2=[float(v) for v in prm['r2']], seconds=time.time() - t0)


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; r = run(*a); (OUT / f'S10{os.environ.get("JOINT_TAG", "")}_joint.json').write_text(json.dumps(r, indent=1)); print('RESULT', json.dumps(r))
