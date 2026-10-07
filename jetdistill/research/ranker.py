"""A9 — the class-token selection as a formula RANKER. ParT's per-head ordering of the particles carries 90 of the
93 points of the formula-value model (the magnitudes < 3). Per head h: a per-particle score r_h(x_i) (hinge terms of
the own + neighbourhood + kernel-context + within-jet relative inputs) trained with a pairwise ranking loss against
ParT's order within each jet (RankNet: for pairs i, j of one jet, P(i above j) = σ(r_i − r_j), target = ParT's α_i > α_j,
pairs weighted by |α_i − α_j|). Weights from the rank: α_i = (1 − α_cls) · g_h(rank_i) / Σ_j g_h(rank_j) with g_h the
average ParT weight at each rank (measured; dev jets are never used for it), α_cls from the jet-level formula (S11's).
Measured: Kendall τ of the formula ranking vs ParT's per head; agreement with (a) ParT's values, (b) S5rpc1's formula
values; then the formula α is written like S11's for the values re-tune (S12).

  python -m jetdistill.research.ranker [n_fit n_dev epochs]"""
import json, os, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, downstream, phi_pooled, rows_of_split, OUT
from .heads_attn import feats
from .alpha_fit import rel_feats
from .alpha_e2e import zfeat
from .weights import internals, logits as part_logits
from .one_term import to_standard
from ..config import RESULTS


def run(n_fit=30000, n_dev=20000, epochs=20, n_jet=100000, lr=1e-3, bs=200, device='mps', log=print):
    import torch
    t0 = time.time(); tag = os.environ.get('OUT_TAG', 'A9'); model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    _, Afa, Mfa, Lfa, _ = extract(model, 'fit', n_jet, device); _, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Af, Mf = Afa[:, :n_fit], Mfa[:n_fit]; Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    Ff, okf = feats(Jf, rf, model); Fd, okd = feats(Jd, rd, model); Ff = np.concatenate([Ff, rel_feats(Ff, okf)], -1); Fd = np.concatenate([Fd, rel_feats(Fd, okd)], -1)
    Zf_all, Zd = zfeat(Jf, Jd, n_jet, rd, Mfa, Md)
    P11 = np.load(OUT / 'S11_alpha.npz'); CZ = P11['cz']                                                  # α_cls: the S11 jet-level formula
    kn = np.quantile(Ff[okf][::5], np.linspace(.15, .85, 5), axis=0).astype(np.float32); mu, sd = Ff[okf].mean(0), Ff[okf].std(0) + 1e-6
    T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt); knt, mut, sdt = T(kn), T(mu), T(sd)
    def phi(F): return torch.cat([(F - mut) / sdt] + [torch.clamp(F - knt[k], min=0) / sdt for k in range(len(kn))], -1)
    # the per-rank weight profile g_h (fit jets): average ParT particle weight at each rank, normalized within the jet to the particle share
    part = Af[:, :, :, 1:] * Mf[None, :, None, :]; rank = np.argsort(np.argsort(-part, -1), -1); g = np.zeros((16, 128))
    for h in range(16):
        b, hh = divmod(h, 8); rel = part[b, :, hh] / (1 - Af[b, :, hh, :1]).clip(1e-9)
        for r in range(128):
            sel = (rank[b, :, hh] == r) & Mf; g[h, r] = rel[sel].mean() if sel.any() else 0
    gt = T(g)
    Wr = torch.nn.Parameter(torch.zeros(phi(T(Ff[:1, :1])).shape[-1], 16, device=device))
    Fft, Fdt, okft, okdt = T(Ff, torch.float16), T(Fd, torch.float16), T(okf, torch.bool), T(okd, torch.bool)
    Aft = T(Af.transpose(1, 0, 2, 3).reshape(n_fit, 16, 129)); of = np.argsort(okf.sum(1)); od = np.argsort(okd.sum(1))
    def scores(F, ok): return (phi(F) @ Wr).masked_fill(~ok[..., None], -1e9)                         # (n, P, 16)
    opt = torch.optim.Adam([Wr], lr)
    def alpha_from_rank(F, ok, Z):                                                                      # (n, 16, 1+P): class-token share from S11's formula, particles by rank profile
        s = scores(F, ok); P = F.shape[1]; rk = torch.argsort(torch.argsort(-s, 1), 1)                 # (n, P, 16)
        w = torch.gather(gt.T[None].expand(F.shape[0], -1, -1), 1, rk) * ok[..., None]                   # (n, P, 16) profile at the formula rank
        w = w / w.sum(1, keepdim=True).clamp(min=1e-12); acls = torch.sigmoid(-(Z @ T(CZ.T)))
        return torch.cat([acls[:, :, None], (1 - acls)[:, :, None] * w.permute(0, 2, 1)], 2)
    def kendall():
        with torch.no_grad():
            taus = np.zeros(16); cnt = 0
            for a in range(0, min(n_dev, 4000), bs):
                i = torch.from_numpy(od[a:a + bs]).to(device); P = int(okdt[i].sum(1).max()); s = scores(Fdt[i, :P].float(), okdt[i, :P]).cpu().numpy(); ok_ = okd[od[a:a + bs], :P]
                ap = Ad[:, od[a:a + bs], :, 1:1 + P].transpose(1, 0, 2, 3).reshape(len(i), 16, P)
                for n_ in range(len(i)):
                    m = ok_[n_]
                    if m.sum() < 3: continue
                    from scipy.stats import kendalltau
                    for h in range(16): taus[h] += kendalltau(s[n_, m, h], ap[n_, h, m])[0]
                    cnt += 1
            return taus / max(cnt, 1)
    for ep in range(epochs):
        tot = 0.0
        for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, bs)):
            i = torch.from_numpy(of[a:a + bs]).to(device); P = int(okft[i].sum(1).max()); ok = okft[i, :P]; s = scores(Fft[i, :P].float(), ok)      # (n, P, 16)
            ap = Aft[i, :, 1:1 + P].permute(0, 2, 1)                                                    # ParT particle weights (n, P, 16)
            d = s[:, :, None] - s[:, None]; da = ap[:, :, None] - ap[:, None]                           # (n, P, P, 16) pair differences
            pair = (ok[:, :, None] & ok[:, None])[..., None]; wgt = da.abs() * pair                      # pairs weighted by how different ParT's weights are
            loss = (torch.nn.functional.softplus(-torch.sign(da) * d) * wgt).sum() / wgt.sum().clamp(min=1e-9)   # RankNet
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        if device == 'mps': torch.mps.empty_cache()
        if ep % 5 == 4 or ep == epochs - 1:
            tau = kendall(); log(f'  A9 epoch {ep + 1}: ranking loss {tot / (n_fit // bs):.4f}; Kendall τ vs ParT per head: block 1 {" ".join(f"{t:.2f}" for t in tau[:8])} | block 2 {" ".join(f"{t:.2f}" for t in tau[8:])}, {time.time() - t0:.0f} s')
    # the formula α (rank profile) on the dev jets → agreement with ParT's values and with S5rpc1's formula values
    Anew = np.zeros((2, n_dev, 8, 129), np.float32)
    with torch.no_grad():
        for a in range(0, n_dev, bs):
            i = od[a:a + bs]; it = torch.from_numpy(i).to(device); P = int(okd[i].sum(1).max()); al = alpha_from_rank(Fdt[it, :P].float(), okdt[it, :P], T(Zd[i])).cpu().numpy()   # (n, 16, 1+P)
            for h in range(16): Anew[h // 8, i, h % 8, :1 + P] = al[:, h]
    D = RESULTS / '_cls' / 'full'; S, V = internals(model, np.load(D / 'dev' / 'x_f16.npy', mmap_mode='r')[rd], np.load(D / 'dev' / 'mask.npy')[rd], device)
    toS = lambda A_: np.where(A_ > 0, np.log(np.maximum(A_, 1e-30)), -np.inf).astype(np.float32)
    Sn = S.copy(); Sn[0] = toS(Anew[0]); ag_part1 = float((part_logits(model, Sn, V, device) == ref).mean()); Sn[1] = toS(Anew[1]); ag_part = float((part_logits(model, Sn, V, device) == ref).mean())
    P5 = np.load(OUT / 'S5rpc1_model.npz'); W5 = T(to_standard(P5['W'], P5['knv'], int(P5['nv'])))
    def s5(Ax):
        Pp, _ = phi_pooled(Jd, rd, Ax, P5['kn'])
        with torch.no_grad(): return float((np.concatenate([downstream(model, *torch.einsum('nhk,hko->nho', T(Pp[a:a + 5000]), W5).split(8, 1)).argmax(1).cpu().numpy() for a in range(0, n_dev, 5000)]) == ref).mean())
    mix1 = np.stack([Anew[0], Ad[1]]); ag_s5_1, ag_s5 = s5(mix1), s5(Anew)
    log(f'  rank-formula α: with ParT values block 1 {100 * ag_part1:.2f}%, both {100 * ag_part:.2f}%; with S5rpc1 formula values block 1 {100 * ag_s5_1:.2f}%, both {100 * ag_s5:.2f}% (ParT α: 93.06 %), {time.time() - t0:.0f} s')
    # write the formula α of the fit and dev jets for the values re-tune
    np.savez(OUT / f'{tag}_alpha.npz', kn=kn, mu=mu, sd=sd, Wr=Wr.detach().cpu().numpy(), g=g, cz=CZ)
    def alpha_of(J, rows, Z, chunk=2000):
        A_ = np.zeros((2, len(rows), 8, 129), np.float16)
        with torch.no_grad():
            for a in range(0, len(rows), chunk):
                r = rows[a:a + chunk]; F, ok = feats(J, r, model); F = np.concatenate([F, rel_feats(F, ok)], -1)
                for c0 in range(0, len(r), bs):
                    sl = slice(c0, c0 + bs); okc = ok[sl]; P = int(okc.sum(1).max()); al = alpha_from_rank(T(F[sl, :P]), T(okc[:, :P], torch.bool), T(Z[a + c0:a + c0 + len(okc)])).cpu().numpy()
                    for h in range(16): A_[h // 8, a + c0:a + c0 + len(okc), h % 8, :1 + P] = al[:, h]
        return A_
    np.save(OUT / f'{tag}_alpha_fit.npy', alpha_of(Jf, rows_of_split('fit', n_jet), Zf_all)); np.save(OUT / f'{tag}_alpha_dev.npy', alpha_of(Jd, rd, Zd))
    r = dict(experiment=tag, n_fit=n_fit, epochs=epochs, kendall=kendall().tolist(), part_values_block1=ag_part1, part_values_both=ag_part, s5_values_block1=ag_s5_1, s5_values_both=ag_s5, seconds=time.time() - t0)
    (OUT / f'{tag}_ranker.json').write_text(json.dumps(r, indent=1)); log(f'{tag} done, {time.time() - t0:.0f} s'); return r


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; run(*a)
