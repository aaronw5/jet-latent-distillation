"""A10 — the all-formula per-head model tuned hard: coefficients AND thresholds learnable, long schedule.
  α (block 1): per-particle score formula s_i = Σ_f [a_f x_if + Σ_k b_fk max(0, x_if − θ_fk)] (A2h's start), α = softmax over
               [the class token (score from the jet-level formula u(jet)), the particles]
  α (block 2): the same form with its own coefficients (start: A2h's block-2 formulas)
  values:      per head 16 formulas Σ_f [c_f x_if + Σ_k d_fk max(0, x_if − θ'_fk)] on own + neighbourhood inputs (start: VALUES_TAG)
  o_h = Σ_i α_i v_i + α_cls e_h + b_h → ParT's fixed downstream.
Every threshold θ, θ' is a parameter (hinges give a gradient to their threshold); coefficients and thresholds tuned
together toward ParT's probabilities (+ λ·R on the head outputs), cosine schedule, best validation epoch kept.

  CTX_HOPS=1 ALPHA_TAG=A2h VALUES_TAG=S12A2h OUT_TAG=A10 python -m jetdistill.research.fulltune [n_fit n_dev epochs]"""
import json, os, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, downstream, rows_of_split, OUT
from .heads_attn import feats
from .alpha_fit import rel_feats
from .alpha_e2e import zfeat


def run(n_fit=100000, n_dev=20000, epochs=60, n_jet=100000, lr=5e-4, lam=0.01, bs=500, device='mps', log=print):
    import torch
    t0 = time.time(); at, vt, tag = os.environ.get('ALPHA_TAG', 'A2h'), os.environ.get('VALUES_TAG', 'S12A2h'), os.environ.get('OUT_TAG', 'A10')
    model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    Ofa, _, Mfa, Lfa, _ = extract(model, 'fit', n_jet, device); _, _, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    Ff, okf = feats(Jf, rf, model); Fd, okd = feats(Jd, rd, model); Ff = np.concatenate([Ff, rel_feats(Ff, okf)], -1); Fd = np.concatenate([Fd, rel_feats(Fd, okd)], -1)
    Zf_all, Zd = zfeat(Jf, Jd, n_jet, rd, Mfa, Md); Zf = Zf_all[:n_fit]
    PA = np.load(OUT / f'{at}_alpha.npz'); PV = np.load(OUT / f'{vt}_model.npz'); nv = 38
    T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
    mu, sd = PA['mu'], PA['sd']; mut, sdt = T(mu), T(sd); lo, hi = T(np.quantile(Ff[okf][::9], .01, axis=0)), T(np.quantile(Ff[okf][::9], .99, axis=0))
    # score formulas: coefficients of [x (standardized), hinges at θ] from A2h (its terms are (x − mu)/sd and max(0, x − kn)/sd)
    COEF = PA['coef'].T; nf = Ff.shape[-1]; K = PA['kn'].shape[0]
    Wa = torch.nn.Parameter(T(COEF[:nf])); Wb = torch.nn.Parameter(T(COEF[nf:].reshape(K, nf, 16))); Th = torch.nn.Parameter(T(PA['kn']))        # (nf,16), (K,nf,16), (K,nf)
    Wz = torch.nn.Parameter(T(PA['cz'].T))
    Wv = torch.nn.Parameter(T(PV['W'])); Thv = torch.nn.Parameter(T(PV['kn'][:, :nv]))                                                                 # (16, 230, 16), (5, 38)
    def heads(F, ok, Z):
        n, P = F.shape[0], F.shape[1]; Fs = (F - mut) / sdt
        s = Fs @ Wa + sum(torch.clamp(F - Th[k], min=0) / sdt @ Wb[k] for k in range(K))                                                           # (n, P, 16)
        rel = torch.softmax(s.masked_fill(~ok[..., None], -1e9), 1); acls = torch.sigmoid(-(Z @ Wz)); al = (1 - acls)[:, None] * rel
        X = F[..., :nv]; pv = torch.cat([X] + [torch.clamp(X - Thv[k], min=0) for k in range(5)], -1)                                             # (n, P, 228)
        pooled = torch.einsum('nph,npk->nhk', al, pv)                                                                                               # (n, 16, 228)
        return torch.einsum('nhk,hko->nho', pooled, Wv[:, :228]) + acls[..., None] * Wv[:, 228][None] + Wv[:, 229][None]
    Fft, Fdt, okft, okdt = T(Ff, torch.float16), T(Fd, torch.float16), T(okf, torch.bool), T(okd, torch.bool); Zft, Zdt = T(Zf), T(Zd); del Ff, Fd
    pf = torch.softmax(T(Lfa[:n_fit]), 1); Oft = T(Ofa[:, :n_fit].transpose(1, 0, 2, 3).reshape(n_fit, 16, 16)); vo = Oft.var(0) + 1e-6
    od, of = np.argsort(okd.sum(1)), np.argsort(okf.sum(1))
    def run_b(Fx, okx, Zx, i):
        P = int(okx[i].sum(1).max()); return heads(Fx[i, :P].float(), okx[i, :P], Zx[i])
    def agree():
        with torch.no_grad():
            pred = np.empty(n_dev, int)
            for a in range(0, n_dev, bs):
                i = torch.from_numpy(od[a:a + bs]).to(device); o = run_b(Fdt, okdt, Zdt, i); pred[od[a:a + bs]] = downstream(model, o[:, :8], o[:, 8:]).argmax(1).cpu().numpy()
        return float((pred == ref).mean())
    params = [Wa, Wb, Wz, Wv, Th, Thv]; a0 = agree(); best = (a0, [p.detach().clone() for p in params], 0); path = [(0, a0)]
    log(f'  start ({at} weights, {vt} values; {nf} inputs, thresholds learnable): {100 * a0:.2f}%, {time.time() - t0:.0f} s')
    steps = epochs * (n_fit // bs); opt = torch.optim.Adam([{'params': [Wa, Wb, Wz], 'lr': lr}, {'params': [Wv], 'lr': lr / 3}, {'params': [Th, Thv], 'lr': lr * 3}])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr / 3, lr * 3], total_steps=steps, pct_start=0.1)
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit - bs + 1, bs)):
            i = torch.from_numpy(of[a:a + bs]).to(device); o = run_b(Fft, okft, Zft, i); Lg = downstream(model, o[:, :8], o[:, 8:])
            loss = -(pf[i] * torch.log_softmax(Lg, 1)).sum(1).mean() + lam * ((o - Oft[i]) ** 2 / vo).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            with torch.no_grad(): Th.data = torch.maximum(torch.minimum(Th, hi), lo); Thv.data = torch.maximum(torch.minimum(Thv, hi[:nv]), lo[:nv])
        if device == 'mps': torch.mps.empty_cache()
        ag = agree(); path.append((ep + 1, ag))
        if ag > best[0]: best = (ag, [p.detach().clone() for p in params], ep + 1)
        if ep % 5 == 4 or ep == epochs - 1:
            with torch.no_grad(): mv = float((Th - T(PA['kn'])).abs().mean() / sdt.mean()), float((Thv - T(PV['kn'][:, :nv])).abs().mean())
            log(f'  A10 epoch {ep + 1}: {100 * ag:.2f}% (best {100 * best[0]:.2f}% at {best[2]}); thresholds moved: scores {mv[0]:.3f} sd, values {mv[1]:.4f}, {time.time() - t0:.0f} s')
    np.savez(OUT / f'{tag}_model.npz', **{k: v.cpu().numpy() for k, v in zip(('Wa', 'Wb', 'Wz', 'Wv', 'Th', 'Thv'), best[1])}, mu=mu, sd=sd, nv=nv)
    r = dict(experiment=tag, alpha=at, values=vt, n_fit=n_fit, epochs=epochs, lr=lr, lam=lam, start=a0, best=best[0], best_epoch=best[2], path=path, seconds=time.time() - t0)
    (OUT / f'{tag}_fulltune.json').write_text(json.dumps(r, indent=1)); log(f'{tag}: all formulas, coefficients and thresholds tuned: {100 * best[0]:.2f}% (start {100 * a0:.2f}%)'); return r


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; run(*a)
