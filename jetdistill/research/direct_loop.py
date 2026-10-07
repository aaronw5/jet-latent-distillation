"""The full loop for the direct-128 model (S15): one term per input per (neuron, head) → prune by (neuron, input) →
re-tune toward ParT's probabilities. Model: neuron_n = Σ_h [Σ_i α_hi φ(x_i)·W_hn + α_h,cls c_hn] + b_n → ParT's last
layer. Basis per head: [x (38), max(0, x − θ_k) (5 × 38), max(0, θ_k − x) (5 × 38, from the pooled terms exactly),
α_cls, 1] = 420 columns; W (16, 420, 128).

  python -m jetdistill.research.direct_loop one_term|prune [args]"""
import json, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, phi_pooled, rows_of_split, OUT
from .one_term import extend, cols_of

NV = 38


def load(n_fit, n_dev, device, log):
    model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    _, Af, Mf, Lf, _ = extract(model, 'fit', n_fit, device); _, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    P15 = np.load(OUT / 'S15_post_model.npz'); kn = P15['kn']
    Pf, _ = phi_pooled(Jf, rf, Af, kn); Pd, _ = phi_pooled(Jd, rd, Ad, kn)
    Ef, Ed = extend(Pf, kn[:, :NV], NV), extend(Pd, kn[:, :NV], NV)                      # (n, 16, 420)
    Hf = np.asarray(Jf['H'][rf], np.float32); W0 = P15['W'].reshape(16, Pf.shape[2], 128)
    We = np.zeros((16, Ef.shape[2], 128), np.float32); We[:, :6 * NV] = W0[:, :6 * NV]; We[:, -2:] = W0[:, 6 * NV:]
    return model, Ef, Ed, Hf, Lf, Ld.argmax(1), kn, We


def tune(model, Ef, Ed, Hf, Lf, ref, W1, mask, epochs, lr=1e-4, lam=0.01, device='mps', log=print, label=''):
    import torch
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); n = len(Ef); K = Ef.shape[2]
    sdc = Ef[:20000].std(0); sdc = np.where(sdc < 1e-6, 1.0, sdc)
    V = torch.nn.Parameter(T(W1 * sdc[..., None])); Mk = T(mask); Sd = T(sdc)[..., None]; Eft, Edt, Hft = T(Ef), T(Ed), T(Hf); vh = Hft.var(0) + 1e-6; pf = torch.softmax(T(Lf), 1)
    Wof = lambda: V * Mk / Sd
    def agree(W):
        with torch.no_grad(): return float((torch.cat([model.fc(torch.einsum('nhk,hko->no', Edt[a:a + 5000], W)).argmax(1) for a in range(0, len(Ed), 5000)]).cpu().numpy() == ref).mean())
    a0 = agree(T(W1)); best = (a0, W1.copy(), 0); opt = torch.optim.Adam([V], lr)
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n, 5000)):
            y = torch.einsum('nhk,hko->no', Eft[a:a + 5000], Wof()); loss = -(pf[a:a + 5000] * torch.log_softmax(model.fc(y), 1)).sum(1).mean() + lam * ((y - Hft[a:a + 5000]) ** 2 / vh).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            Wn = Wof().detach(); ag = agree(Wn)
            if ag > best[0]: best = (ag, Wn.cpu().numpy(), ep + 1)
            if ep % 50 == 49: log(f'  {label} epoch {ep + 1}: {100 * ag:.2f}% (best {100 * best[0]:.2f}%)')
    return best


def one_term(n_fit=100000, n_dev=20000, epochs=150, device='mps', log=print):
    t0 = time.time(); model, Ef, Ed, Hf, Lf, ref, kn, We = load(n_fit, n_dev, device, log); K = Ef.shape[2]; S = Ef[:20000]
    import torch
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)
    with torch.no_grad(): a0 = float((torch.cat([model.fc(torch.einsum('nhk,hko->no', T(Ed[a:a + 5000]), T(We))).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    log(f'  S15: {100 * a0:.2f}% ({int((We[:, :6 * NV] != 0).sum())} nonzero term coefficients), {time.time() - t0:.0f} s')
    W1 = np.zeros_like(We); W1[:, -2:] = We[:, -2:]
    for h in range(16):
        for o in range(128):
            for f in range(NV):
                c6 = [f] + [NV + k * NV + f for k in range(5)]; w = We[h, c6, o]
                if not np.any(w): continue
                y = S[:, h, c6] @ w; ym = y.mean(); best = (np.inf, None, 0.0, 0.0)
                for j in cols_of(f, NV):
                    x = S[:, h, j]; xc = x - x.mean(); den = xc @ xc
                    if den < 1e-12: continue
                    a = float(xc @ (y - ym) / den); r = (y - ym) - a * xc; err = float(r @ r)
                    if err < best[0]: best = (err, j, a, float(ym - a * x.mean()))
                if best[1] is None: continue
                W1[h, best[1], o] = best[2]; W1[h, K - 1, o] += best[3]
    with torch.no_grad(): a1 = float((torch.cat([model.fc(torch.einsum('nhk,hko->no', T(Ed[a:a + 5000]), T(W1))).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    nterm = int((W1[:, :K - 2] != 0).sum()); log(f'  one term per input per (neuron, head): least squares {100 * a1:.2f}% ({nterm} terms), {time.time() - t0:.0f} s')
    mask = (W1 != 0).astype(np.float32); mask[:, -2:] = 1
    best = tune(model, Ef, Ed, Hf, Lf, ref, W1, mask, epochs, device=device, log=log, label='S15 one-term')
    np.savez(OUT / 'S15o_model.npz', W=best[1].reshape(16 * K, 128), kn=kn, knv=kn[:, :NV], nv=NV, basis='extended', direct=True)
    r = dict(experiment='S15o', start=a0, one_term_ls=a1, best=best[0], best_epoch=best[2], terms=nterm, terms_start=int((We[:, :6 * NV] != 0).sum()), seconds=time.time() - t0)
    (OUT / 'S15o_one_term.json').write_text(json.dumps(r, indent=1)); log(f'S15o: {100 * best[0]:.2f}% with one term per input per (neuron, head) ({nterm} terms; start {r["terms_start"]})'); return r


def prune(tol=0.1, rounds=8, epochs=40, n_fit=100000, n_dev=20000, device='mps', log=print):
    """remove (neuron, input) groups — the input's terms in all 16 heads of that neuron — least needed first, re-tune"""
    import torch
    t0 = time.time(); model, Ef, Ed, Hf, Lf, ref, kn, _ = load(n_fit, n_dev, device, log); K = Ef.shape[2]
    P = np.load(OUT / 'S15o_model.npz'); W = P['W'].reshape(16, K, 128).astype(np.float32); T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); Edt = T(Ed)
    def agree(Wx):
        with torch.no_grad(): return float((torch.cat([model.fc(torch.einsum('nhk,hko->no', Edt[a:a + 5000], T(Wx))).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    group = lambda f: [f] + [NV + k * NV + f for k in range(5)] + [6 * NV + k * NV + f for k in range(5)]
    keep = np.zeros((128, NV), bool)
    for o in range(128):
        for f in range(NV): keep[o, f] = np.any(W[:, group(f), o])
    a_ref = a_now = agree(W); path = [dict(pairs=int(keep.sum()), agreement=a_now)]; log(f'  start: {int(keep.sum())} (neuron, input) pairs, {100 * a_now:.2f}%')
    for rnd in range(rounds):
        drops = {}
        for o in range(128):
            for f in np.flatnonzero(keep[o]):
                Wt = W.copy(); Wt[:, group(f), o] = 0; drops[(o, f)] = a_now - agree(Wt)
        order = sorted(drops, key=drops.get); cum, rem = 0.0, []
        for of_ in order:
            if cum + max(drops[of_], 0) > tol / 100: break
            cum += max(drops[of_], 0); rem.append(of_)
        if not rem: break
        for o, f in rem: keep[o, f] = False; W[:, group(f), o] = 0
        mask = (W != 0).astype(np.float32); mask[:, -2:] = 1
        best = tune(model, Ef, Ed, Hf, Lf, ref, W, mask, epochs, device=device, log=log, label=f'prune round {rnd + 1}'); W = best[1] * mask; a_now = agree(W)
        path.append(dict(round=rnd + 1, removed=len(rem), pairs=int(keep.sum()), agreement=a_now)); np.savez(OUT / 'S15p_model.npz', W=W.reshape(16 * K, 128), kn=kn, knv=kn[:, :NV], nv=NV, basis='extended', direct=True, keep=keep)
        log(f'  round {rnd + 1}: removed {len(rem)} (neuron, input) pairs → {int(keep.sum())} kept ({keep.sum(1).min()}–{keep.sum(1).max()} inputs per neuron), re-tuned {100 * a_now:.2f}% (start {100 * a_ref:.2f}%), {time.time() - t0:.0f} s')
        if a_now < a_ref - tol / 100: log('  below the tolerance: stop'); break
    r = dict(tol=tol, start=a_ref, final=a_now, inputs_per_neuron=keep.sum(1).tolist(), path=path, seconds=time.time() - t0)
    (OUT / 'S15p_pruned.json').write_text(json.dumps(r, indent=1)); log(f'pruned S15: {int(keep.sum())} (neuron, input) pairs, {100 * a_now:.2f}%'); return r


def prune_terms(tol=0.1, rounds=6, epochs=30, n_fit=100000, n_dev=20000, src='S15p', out='S15q', device='mps', log=print):
    """statement-level pruning: single terms (neuron, input, head) removed smallest first — how many per round by bisection on the
    agreement drop (≤ tol points) — then re-tuned; stops when the re-tuned model falls below the start by more than tol"""
    import torch
    t0 = time.time(); model, Ef, Ed, Hf, Lf, ref, kn, _ = load(n_fit, n_dev, device, log); K = Ef.shape[2]
    P = np.load(OUT / f'{src}_model.npz'); W = P['W'].reshape(16, K, 128).astype(np.float32); T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); Edt = T(Ed)
    def agree(Wx):
        with torch.no_grad(): return float((torch.cat([model.fc(torch.einsum('nhk,hko->no', Edt[a:a + 5000], T(Wx))).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    sd = Ed.std(0)                                                                                   # (16, K): spread of every pooled term over jets
    a_ref = a_now = agree(W); n0 = int((W[:, :K - 2] != 0).sum()); path = [dict(terms=n0, agreement=a_now)]; log(f'  start: {n0} statements, {100 * a_now:.2f}%')
    for rnd in range(rounds):
        hs, ks, os_ = np.nonzero(W[:, :K - 2]); size = np.abs(W[hs, ks, os_]) * sd[hs, ks]; order = np.argsort(size)
        def after(m):
            Wt = W.copy(); Wt[hs[order[:m]], ks[order[:m]], os_[order[:m]]] = 0; return Wt
        lo, hi = 0, len(order)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if a_now - agree(after(mid)) <= tol / 100: lo = mid
            else: hi = mid
        if lo == 0: log('  nothing removable within the tolerance: stop'); break
        W = after(lo); mask = (W != 0).astype(np.float32); mask[:, -2:] = 1
        best = tune(model, Ef, Ed, Hf, Lf, ref, W, mask, epochs, device=device, log=log, label=f'term prune round {rnd + 1}'); W = best[1] * mask; a_now = agree(W); nt = int((W[:, :K - 2] != 0).sum())
        path.append(dict(round=rnd + 1, removed=int(lo), terms=nt, agreement=a_now)); np.savez(OUT / f'{out}_model.npz', W=W.reshape(16 * K, 128), kn=kn, knv=kn[:, :NV], nv=NV, basis='extended', direct=True)
        log(f'  round {rnd + 1}: removed {lo} statements → {nt} left, re-tuned {100 * a_now:.2f}% (start {100 * a_ref:.2f}%), {time.time() - t0:.0f} s')
        if a_now < a_ref - tol / 100: log('  below the tolerance: stop'); break
    r = dict(tol=tol, src=src, start=a_ref, final=a_now, terms_start=n0, terms=int((W[:, :K - 2] != 0).sum()), path=path, seconds=time.time() - t0)
    (OUT / f'{out}_pruned_terms.json').write_text(json.dumps(r, indent=1)); log(f'{out}: {r["terms"]} statements (from {n0}), {100 * a_now:.2f}%'); return r


def share(src='S15q', out='S17', epochs=60, n_fit=100000, n_dev=20000, device='mps', log=print):
    """one STATEMENT per (neuron, input): a single column (kind + threshold) shared by all 16 heads, chosen as the column that best
    reproduces the heads' current pieces jointly (least squares over a particle sample); per-head coefficients kept; re-tuned"""
    import torch
    t0 = time.time(); model, Ef, Ed, Hf, Lf, ref, kn, _ = load(n_fit, n_dev, device, log); K = Ef.shape[2]
    P = np.load(OUT / f'{src}_model.npz'); W = P['W'].reshape(16, K, 128).astype(np.float32); T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); Edt = T(Ed)
    def agree(Wx):
        with torch.no_grad(): return float((torch.cat([model.fc(torch.einsum('nhk,hko->no', Edt[a:a + 5000], T(Wx))).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    a_src = agree(W); S = Ef[:20000]; W1 = np.zeros_like(W); W1[:, -2:] = W[:, -2:]; nst = 0
    for o in range(128):
        for f in range(NV):
            cf = cols_of(f, NV); Wf = W[:, cf, o]                                                   # (16, 11) the heads' terms of this input
            if not np.any(Wf): continue
            Y = np.einsum('nhk,hk->nh', S[:, :, cf], Wf); best = (np.inf, None)
            for j in cf:                                                                            # one column for all heads: per-head least squares on the pooled term
                x = S[:, :, j]; den = (x * x).sum(0); c = np.where(den > 1e-12, (x * Y).sum(0) / np.maximum(den, 1e-12), 0.0); err = ((Y - x * c) ** 2).sum()
                if err < best[0]: best = (err, j, c)
            W1[:, best[1], o] = best[2]; nst += 1
    a1 = agree(W1); log(f'  {src}: {100 * a_src:.2f}%; one shared statement per (neuron, input), per-head coefficients: {100 * a1:.2f}% before tuning ({nst} statements), {time.time() - t0:.0f} s')
    mask = (W1 != 0).astype(np.float32); mask[:, -2:] = 1; best = tune(model, Ef, Ed, Hf, Lf, ref, W1, mask, epochs, device=device, log=log, label='S17')
    np.savez(OUT / f'{out}_model.npz', W=best[1].reshape(16 * K, 128), kn=kn, knv=kn[:, :NV], nv=NV, basis='extended', direct=True, shared=True)
    r = dict(experiment=out, src=src, start_src=a_src, start=a1, best=best[0], best_epoch=best[2], statements=nst, coefficients=int((best[1][:, :K - 2] != 0).sum()), seconds=time.time() - t0)
    (OUT / f'{out}_shared.json').write_text(json.dumps(r, indent=1)); log(f'{out}: {nst} statements (one per neuron and input, shared by the heads; {r["coefficients"]} per-head coefficients): {100 * best[0]:.2f}% (from {src} {100 * a_src:.2f}%)'); return r


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'one_term': one_term(*(int(v) for v in a))
    elif cmd == 'share': share(*a[:2], *(int(v) for v in a[2:]))
    elif cmd == 'prune_terms': prune_terms(*(float(v) if i == 0 else int(v) for i, v in enumerate(a)))
    else: prune(*(float(v) if i == 0 else int(v) for i, v in enumerate(a)))
