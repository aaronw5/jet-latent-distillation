"""W1 — the full loop on the selection formulas (α as formulas, ParT's values kept): the A2h score formulas
(per head, hinge terms of 314 per-particle inputs; α = (1 − α_cls) softmax(s), α_cls from the jet-level formula)
→ at most one term per input per head → pruning by (head, input) with re-tuning → tuned toward ParT's probabilities
through ParT's values and downstream. Saves W1_alpha.npz (same format as A2h_alpha.npz: coef (16, terms), cz).

  CTX_HOPS=1 python -m jetdistill.research.wloop one_term|prune [args]"""
import json, os, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, after, rows_of_split, OUT
from .heads_attn import feats
from .alpha_fit import rel_feats
from .alpha_e2e import zfeat
from .weights import internals
from ..config import RESULTS


def setup(n_fit, n_dev, device, log, src='A2h'):
    import torch
    t0 = time.time(); model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    _, _, Mfa, Lfa, _ = extract(model, 'fit', 100000, device); _, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    Ff, okf = feats(Jf, rf, model); Fd, okd = feats(Jd, rd, model); Ff = np.concatenate([Ff, rel_feats(Ff, okf)], -1); Fd = np.concatenate([Fd, rel_feats(Fd, okd)], -1)
    Zf_all, Zd = zfeat(Jf, Jd, 100000, rd, Mfa, Md); Zf = Zf_all[:n_fit]
    PA = np.load(OUT / f'{src}_alpha.npz'); D = RESULTS / '_cls' / 'full'
    _, Vf = internals(model, np.load(D / 'fit' / 'x_f16.npy', mmap_mode='r')[:n_fit], np.load(D / 'fit' / 'mask.npy')[:n_fit], device)
    _, Vd = internals(model, np.load(D / 'dev' / 'x_f16.npy', mmap_mode='r')[rd], np.load(D / 'dev' / 'mask.npy')[rd], device)
    log(f'  inputs, ParT values, {src} formulas, {time.time() - t0:.0f} s')
    return dict(model=model, Ff=Ff, Fd=Fd, okf=okf, okd=okd, Zf=Zf, Zd=Zd, Vf=Vf.transpose(1, 0, 2, 3, 4), Vd=Vd.transpose(1, 0, 2, 3, 4), Lf=Lfa[:n_fit], ref=ref, PA=PA, n_fit=n_fit, n_dev=n_dev)


class Net:
    """α from the score formulas (extended basis: x, max(0, x − θ_k), max(0, θ_k − x)) and the jet-level class-token share; ParT's values and downstream"""
    def __init__(self, S, device):
        import torch
        self.t, self.S, self.device = torch, S, device; T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
        PA = S['PA']; self.kn, self.mu, self.sd = T(PA['kn']), T(PA['mu']), T(PA['sd']); self.K = PA['kn'].shape[0]; self.nf = PA['mu'].shape[0]
        self.Fft, self.Fdt = T(S['Ff'], torch.float16), T(S['Fd'], torch.float16); self.okft, self.okdt = T(S['okf'], torch.bool), T(S['okd'], torch.bool)
        self.Zft, self.Zdt = T(S['Zf']), T(S['Zd']); self.Vft, self.Vdt = T(S['Vf'], torch.float16), T(S['Vd'], torch.float16); self.pf = torch.softmax(T(S['Lf']), 1)
        self.of, self.od = np.argsort(S['okf'].sum(1)), np.argsort(S['okd'].sum(1)); self.model = S['model']
    def phi(self, F):                                                        # (…, 1 + nf + 2 K nf): 1, x, x > θ_k, θ_k > x  (standardized)
        t = self.t; Fs = (F - self.mu) / self.sd
        return t.cat([t.ones_like(F[..., :1]), Fs] + [t.clamp(F - self.kn[k], min=0) / self.sd for k in range(self.K)] + [t.clamp(self.kn[k] - F, min=0) / self.sd for k in range(self.K)], -1)
    def logits(self, F, ok, Z, V, Wc, Wz):
        t = self.t; n = F.shape[0]; P = int(ok.sum(1).max()); F, ok, V = F[:, :P].float(), ok[:, :P], V[:, :, :1 + P]
        s = (self.phi(F) @ Wc).masked_fill(~ok[..., None], -1e9); rel = t.softmax(s, 1); acls = t.sigmoid(-(Z @ Wz)); cls = self.model.cls_token[0, 0].expand(n, -1)
        for b, blk in enumerate(self.model.cls_blocks):
            a = t.cat([acls[:, 8 * b:8 * b + 8, None], (1 - acls[:, 8 * b:8 * b + 8, None]) * rel[:, :, 8 * b:8 * b + 8].permute(0, 2, 1)], 2)
            cls = after(blk, t.einsum('nhp,nphd->nhd', a, V[:, b].float()), cls)
        return self.model.fc(self.model.norm(cls))
    def agree(self, Wc, Wz, bs=500):
        t = self.t; n = self.S['n_dev']
        with t.no_grad():
            pred = np.empty(n, int)
            for a in range(0, n, bs):
                i = t.from_numpy(self.od[a:a + bs]).to(self.device); pred[self.od[a:a + bs]] = self.logits(self.Fdt[i], self.okdt[i], self.Zdt[i], self.Vdt[i], Wc, Wz).argmax(1).cpu().numpy()
        return float((pred == self.S['ref']).mean())
    def tune(self, Wc0, Wz0, mask, epochs, lr=3e-4, bs=500, log=print, label=''):
        t = self.t; Wc = t.nn.Parameter(Wc0.clone()); Wz = t.nn.Parameter(Wz0.clone()); opt = t.optim.Adam([Wc, Wz], lr); n = self.S['n_fit']
        a0 = self.agree(Wc0 * mask, Wz0); best = (a0, Wc0.clone(), Wz0.clone(), 0)
        for ep in range(epochs):
            for a in np.random.default_rng(ep).permutation(np.arange(0, n, bs)):
                i = t.from_numpy(self.of[a:a + bs]).to(self.device); L = self.logits(self.Fft[i], self.okft[i], self.Zft[i], self.Vft[i], Wc * mask, Wz)
                loss = -(self.pf[i] * t.log_softmax(L, 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if self.device == 'mps': t.mps.empty_cache()
            ag = self.agree(Wc.detach() * mask, Wz.detach())
            if ag > best[0]: best = (ag, (Wc.detach() * mask).clone(), Wz.detach().clone(), ep + 1)
            if ep % 5 == 4 or ep == epochs - 1: log(f'  {label} epoch {ep + 1}: {100 * ag:.2f}% (best {100 * best[0]:.2f}%)')
        return best


def to_extended(coef, nf, K):
    """A2h's (16, nf + K nf) coefficients of [x, x > θ_k] → the extended basis [1, x, x > θ_k, θ_k > x]: (1 + nf + 2 K nf, 16)"""
    W = np.zeros((1 + nf + 2 * K * nf, 16), np.float32); W[1:1 + nf + K * nf] = coef.T; return W


def one_term(n_fit=30000, n_dev=20000, epochs=20, device='mps', log=print):
    import torch
    t0 = time.time(); S = setup(n_fit, n_dev, device, log); net = Net(S, device); PA = S['PA']; nf, K = net.nf, net.K
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)
    W0 = T(to_extended(PA['coef'], nf, K)); Wz = T(PA['cz'].T); a0 = net.agree(W0, Wz); log(f'  start (A2h, ParT values): {100 * a0:.2f}%, {time.time() - t0:.0f} s')
    # per (head, input): the single best term for the input's current piece, least squares over a sample of real particles
    with torch.no_grad():
        Fs = S['Ff'][S['okf']][::max(1, S['okf'].sum() // 300000)]; B = net.phi(T(Fs)).cpu().numpy()
    W1 = np.zeros_like(W0.cpu().numpy()); W1[0] = W0[0].cpu().numpy(); W0n = W0.cpu().numpy()
    cols = lambda f: [1 + f] + [1 + nf + k * nf + f for k in range(K)] + [1 + nf + K * nf + k * nf + f for k in range(K)]
    for h in range(16):
        for f in range(nf):
            cf = cols(f); w = W0n[cf, h]
            if not np.any(w): continue
            y = B[:, cf] @ w; ym = y.mean(); best = (np.inf, None, 0.0, 0.0)
            for j in cf:
                x = B[:, j]; xc = x - x.mean(); den = xc @ xc
                if den < 1e-12: continue
                a = float(xc @ (y - ym) / den); r = (y - ym) - a * xc; err = float(r @ r)
                if err < best[0]: best = (err, j, a, float(ym - a * x.mean()))
            if best[1] is None: continue
            W1[best[1], h] = best[2]; W1[0, h] += best[3]
    mask = (W1 != 0).astype(np.float32); mask[0] = 1; W1t = T(W1); a1 = net.agree(W1t, Wz)
    nterm = int((W1[1:] != 0).sum()); log(f'  one term per input per head: {100 * a1:.2f}% ({nterm} terms; start {int((W0n[1:] != 0).sum())}), {time.time() - t0:.0f} s')
    best = net.tune(W1t, Wz, T(mask), epochs, log=log, label='W1 one-term')
    np.savez(OUT / 'W1o_alpha.npz', W=best[1].cpu().numpy(), cz=best[2].cpu().numpy().T, kn=PA['kn'], mu=PA['mu'], sd=PA['sd'], basis='extended')
    r = dict(experiment='W1o', start=a0, one_term_ls=a1, best=best[0], best_epoch=best[3], terms=nterm, seconds=time.time() - t0)
    (OUT / 'W1o_one_term.json').write_text(json.dumps(r, indent=1)); log(f'W1o: selection formulas with one term per input per head, ParT values: {100 * best[0]:.2f}% (start {100 * a0:.2f}%)'); return r


def prune(tol=0.1, rounds=6, epochs=8, n_fit=30000, n_dev=20000, device='mps', log=print):
    import torch
    t0 = time.time(); S = setup(n_fit, n_dev, device, log); net = Net(S, device); nf, K = net.nf, net.K; P1 = np.load(OUT / 'W1o_alpha.npz')
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); W = P1['W'].copy(); Wz = T(P1['cz'].T)
    cols = lambda f: [1 + f] + [1 + nf + k * nf + f for k in range(K)] + [1 + nf + K * nf + k * nf + f for k in range(K)]
    keep = np.array([[np.any(W[cols(f), h]) for f in range(nf)] for h in range(16)])
    a_ref = a_now = net.agree(T(W), Wz); path = [dict(pairs=int(keep.sum()), agreement=a_now)]; log(f'  start: {int(keep.sum())} (head, input) pairs, {100 * a_now:.2f}%')
    for rnd in range(rounds):
        drops = {}
        for h in range(16):
            for f in np.flatnonzero(keep[h]):
                Wt = W.copy(); Wt[cols(f), h] = 0; drops[(h, f)] = a_now - net.agree(T(Wt), Wz)
        order = sorted(drops, key=drops.get); cum, rem = 0.0, []
        for hf in order:
            if cum + max(drops[hf], 0) > tol / 100: break
            cum += max(drops[hf], 0); rem.append(hf)
        if not rem: break
        for h, f in rem: keep[h, f] = False; W[cols(f), h] = 0
        mask = (W != 0).astype(np.float32); mask[0] = 1; best = net.tune(T(W), Wz, T(mask), epochs, log=log, label=f'W1 prune round {rnd + 1}'); W = best[1].cpu().numpy(); Wz = best[2]; a_now = best[0]
        path.append(dict(round=rnd + 1, removed=len(rem), pairs=int(keep.sum()), agreement=a_now)); np.savez(OUT / 'W1p_alpha.npz', W=W, cz=Wz.cpu().numpy().T, kn=P1['kn'], mu=P1['mu'], sd=P1['sd'], basis='extended', keep=keep)
        log(f'  round {rnd + 1}: removed {len(rem)} (head, input) pairs → {int(keep.sum())} kept ({keep.sum(1).min()}–{keep.sum(1).max()} inputs per head), re-tuned {100 * a_now:.2f}% (start {100 * a_ref:.2f}%), {time.time() - t0:.0f} s')
        if a_now < a_ref - tol / 100: log('  below the tolerance: stop'); break
    r = dict(tol=tol, start=a_ref, final=a_now, inputs_per_head=keep.sum(1).tolist(), path=path, seconds=time.time() - t0)
    (OUT / 'W1p_pruned.json').write_text(json.dumps(r, indent=1)); log(f'pruned W1: {int(keep.sum())} (head, input) pairs, {100 * a_now:.2f}%'); return r


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'one_term': one_term(*(int(v) for v in a))
    else: prune(*(float(v) if i == 0 else int(v) for i, v in enumerate(a)))
