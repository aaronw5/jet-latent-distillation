"""The full loop for the direct-128 model (S15): one term per input per (neuron, head) → prune by (neuron, input) →
re-tune toward ParT's probabilities. Model: neuron_n = Σ_h [Σ_i α_hi φ(x_i)·W_hn + α_h,cls c_hn] + b_n → ParT's last
layer. Basis per head: [x (38), max(0, x − θ_k) (5 × 38), max(0, θ_k − x) (5 × 38, from the pooled terms exactly),
α_cls, 1] = 420 columns; W (16, 420, 128).

  python -m jetdistill.research.direct_loop one_term|prune [args]"""
import json, os, sys, time, types
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, phi_pooled, rows_of_split, OUT
from .one_term import extend, cols_of

NV = 38
PRE = os.environ.get('DPRE', 'S15'); KTERM = int(os.environ.get('KTERM', '1')); LOGITS = bool(os.environ.get('LOGITS')); AF = os.environ.get('ALPHA_FROM', ''); XA = dict(alpha_from=AF) if AF else {}


def load(n_fit, n_dev, device, log):
    model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    _, Af, Mf, Lf, _ = extract(model, 'fit', n_fit, device); _, Ad, Md, Ld, _ = extract(model, 'dev', n_dev, device)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    if AF:                                                                                   # the combined models: the weights from the selection formulas, not ParT
        from .direct_alpha import zfun, formula_alpha
        zf = zfun(model, device); Af = formula_alpha(AF, Jf, rf, model, zf(Jf, rf, Mf), device); Ad = formula_alpha(AF, Jd, rd, model, zf(Jd, rd, Md), device); log(f'  weights from the selection formulas {AF}')
    P15 = np.load(OUT / f'{PRE}_post_model.npz'); kn = P15['kn']
    Pf, _ = phi_pooled(Jf, rf, Af, kn); Pd, _ = phi_pooled(Jd, rd, Ad, kn)
    Ef, Ed = extend(Pf, kn[:, :NV], NV), extend(Pd, kn[:, :NV], NV)                      # (n, 16, 420)
    Hf = Lf.astype(np.float32) if LOGITS else np.asarray(Jf['H'][rf], np.float32); W0 = P15['W'].reshape(16, Pf.shape[2], -1)
    if LOGITS: model = types.SimpleNamespace(fc=lambda y: y)
    We = np.zeros((16, Ef.shape[2], W0.shape[2]), np.float32); We[:, :6 * NV] = W0[:, :6 * NV]; We[:, -2:] = W0[:, 6 * NV:]
    return model, Ef, Ed, Hf, Lf, Ld.argmax(1), kn, We


def tune(model, Ef, Ed, Hf, Lf, ref, W1, mask, epochs, lr=1e-4, lam=0.01, device='mps', log=print, label='', l1=0.0):
    import torch
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); n = len(Ef); K = Ef.shape[2]
    sdc = Ef[:20000].std(0); sdc = np.where(sdc < 1e-6, 1.0, sdc)
    V = torch.nn.Parameter(T(W1 * sdc[..., None])); Mk = T(mask); Sd = T(sdc)[..., None]; Eft, Edt, Hft = T(Ef), T(Ed), T(Hf); vh = Hft.var(0) + 1e-6; pf = torch.softmax(T(Lf), 1)
    Wof = lambda: V * Mk / Sd
    def agree(W):
        with torch.no_grad(): return float((torch.cat([model.fc(torch.einsum('nhk,hko->no', Edt[a:a + 5000], W)).argmax(1) for a in range(0, len(Ed), 5000)]).cpu().numpy() == ref).mean())
    a0 = agree(T(W1)); best = (-1.0 if l1 else a0, W1.copy(), 0); opt = torch.optim.Adam([V], lr)        # with an L1 penalty: keep the penalized weights (agreement is expected to fall)
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n, 5000)):
            y = torch.einsum('nhk,hko->no', Eft[a:a + 5000], Wof()); loss = -(pf[a:a + 5000] * torch.log_softmax(model.fc(y), 1)).sum(1).mean() + lam * ((y - Hft[a:a + 5000]) ** 2 / vh).mean() + (l1 * (V * Mk)[:, :K - 2].abs().sum() if l1 else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            Wn = Wof().detach(); ag = agree(Wn)
            if ag > best[0] or (l1 and ep >= epochs - 10): best = (ag, Wn.cpu().numpy(), ep + 1)
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
        for o in range(We.shape[2]):
            for f in range(NV):
                c6 = [f] + [NV + k * NV + f for k in range(5)]; w = We[h, c6, o]
                if not np.any(w): continue
                y = S[:, h, c6] @ w; ym = y.mean(); chosen = []                       # greedy: the best term, then (KTERM ≥ 2) the best next term given the chosen ones
                for _ in range(KTERM):
                    best = (np.inf, None, None)
                    for j in cols_of(f, NV):
                        if j in chosen: continue
                        X_ = S[:, h, chosen + [j]]; Xc = X_ - X_.mean(0); G_ = Xc.T @ Xc
                        if np.linalg.cond(G_) > 1e10: continue
                        a = np.linalg.solve(G_, Xc.T @ (y - ym)); r = (y - ym) - Xc @ a; err = float(r @ r)
                        if err < best[0]: best = (err, j, a)
                    if best[1] is None: break
                    chosen.append(best[1]); coefs = best[2]
                if not chosen: continue
                X_ = S[:, h, chosen]; W1[h, chosen, o] = coefs; W1[h, K - 1, o] += float(ym - X_.mean(0) @ coefs)
    with torch.no_grad(): a1 = float((torch.cat([model.fc(torch.einsum('nhk,hko->no', T(Ed[a:a + 5000]), T(W1))).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    nterm = int((W1[:, :K - 2] != 0).sum()); log(f'  at most {KTERM} term(s) per input per (neuron, head): least squares {100 * a1:.2f}% ({nterm} terms), {time.time() - t0:.0f} s')
    mask = (W1 != 0).astype(np.float32); mask[:, -2:] = 1
    best = tune(model, Ef, Ed, Hf, Lf, ref, W1, mask, epochs, device=device, log=log, label='S15 one-term')
    np.savez(OUT / f'{PRE}o_model.npz', W=best[1].reshape(16 * K, -1), kn=kn, knv=kn[:, :NV], nv=NV, basis='extended', direct=True, logits=LOGITS, **XA)
    r = dict(experiment=f'{PRE}o', start=a0, one_term_ls=a1, best=best[0], best_epoch=best[2], terms=nterm, terms_start=int((We[:, :6 * NV] != 0).sum()), seconds=time.time() - t0)
    (OUT / f'{PRE}o_one_term.json').write_text(json.dumps(r, indent=1)); log(f'{PRE}o: {100 * best[0]:.2f}% with one term per input per (neuron, head) ({nterm} terms; start {r["terms_start"]})'); return r


def prune(tol=0.1, rounds=8, epochs=40, n_fit=100000, n_dev=20000, device='mps', log=print):
    """remove (neuron, input) groups — the input's terms in all 16 heads of that neuron — least needed first, re-tune"""
    import torch
    t0 = time.time(); model, Ef, Ed, Hf, Lf, ref, kn, _ = load(n_fit, n_dev, device, log); K = Ef.shape[2]
    P = np.load(OUT / f'{PRE}o_model.npz'); W = P['W'].reshape(16, K, -1).astype(np.float32); NO = W.shape[2]; T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); Edt = T(Ed)
    def agree(Wx):
        with torch.no_grad(): return float((torch.cat([model.fc(torch.einsum('nhk,hko->no', Edt[a:a + 5000], T(Wx))).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    group = lambda f: [f] + [NV + k * NV + f for k in range(5)] + [6 * NV + k * NV + f for k in range(5)]
    keep = np.zeros((NO, NV), bool)
    for o in range(NO):
        for f in range(NV): keep[o, f] = np.any(W[:, group(f), o])
    a_ref = a_now = agree(W); path = [dict(pairs=int(keep.sum()), agreement=a_now)]; log(f'  start: {int(keep.sum())} (neuron, input) pairs, {100 * a_now:.2f}%')
    for rnd in range(rounds):
        drops = {}
        for o in range(NO):
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
        path.append(dict(round=rnd + 1, removed=len(rem), pairs=int(keep.sum()), agreement=a_now)); np.savez(OUT / f'{PRE}p_model.npz', W=W.reshape(16 * K, -1), kn=kn, knv=kn[:, :NV], nv=NV, basis='extended', direct=True, keep=keep, logits=LOGITS, **XA)
        log(f'  round {rnd + 1}: removed {len(rem)} (neuron, input) pairs → {int(keep.sum())} kept ({keep.sum(1).min()}–{keep.sum(1).max()} inputs per neuron), re-tuned {100 * a_now:.2f}% (start {100 * a_ref:.2f}%), {time.time() - t0:.0f} s')
        if a_now < a_ref - tol / 100: log('  below the tolerance: stop'); break
    r = dict(tol=tol, start=a_ref, final=a_now, inputs_per_neuron=keep.sum(1).tolist(), path=path, seconds=time.time() - t0)
    (OUT / f'{PRE}p_pruned.json').write_text(json.dumps(r, indent=1)); log(f'pruned {PRE}: {int(keep.sum())} (neuron, input) pairs, {100 * a_now:.2f}%'); return r


def prune_terms(tol=0.1, rounds=6, epochs=30, n_fit=100000, n_dev=20000, src='S15p', out='S15q', device='mps', log=print):
    """statement-level pruning: single terms (neuron, input, head) removed smallest first — how many per round by bisection on the
    agreement drop (≤ tol points) — then re-tuned; stops when the re-tuned model falls below the start by more than tol"""
    import torch
    t0 = time.time(); model, Ef, Ed, Hf, Lf, ref, kn, _ = load(n_fit, n_dev, device, log); K = Ef.shape[2]
    P = np.load(OUT / f'{src}_model.npz'); W = P['W'].reshape(16, K, -1).astype(np.float32); T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); Edt = T(Ed)
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
        path.append(dict(round=rnd + 1, removed=int(lo), terms=nt, agreement=a_now)); np.savez(OUT / f'{out}_model.npz', W=W.reshape(16 * K, -1), kn=kn, knv=kn[:, :NV], nv=NV, basis='extended', direct=True, logits=LOGITS, **XA)
        log(f'  round {rnd + 1}: removed {lo} statements → {nt} left, re-tuned {100 * a_now:.2f}% (start {100 * a_ref:.2f}%), {time.time() - t0:.0f} s')
        if a_now < a_ref - tol / 100: log('  below the tolerance: stop'); break
    r = dict(tol=tol, src=src, start=a_ref, final=a_now, terms_start=n0, terms=int((W[:, :K - 2] != 0).sum()), path=path, seconds=time.time() - t0)
    (OUT / f'{out}_pruned_terms.json').write_text(json.dumps(r, indent=1)); log(f'{out}: {r["terms"]} statements (from {n0}), {100 * a_now:.2f}%'); return r


def share(src='S15q', out='S17', epochs=60, n_fit=100000, n_dev=20000, device='mps', log=print):
    """one STATEMENT per (neuron, input): a single column (kind + threshold) shared by all 16 heads, chosen as the column that best
    reproduces the heads' current pieces jointly (least squares over a particle sample); per-head coefficients kept; re-tuned"""
    import torch
    t0 = time.time(); model, Ef, Ed, Hf, Lf, ref, kn, _ = load(n_fit, n_dev, device, log); K = Ef.shape[2]
    P = np.load(OUT / f'{src}_model.npz'); W = P['W'].reshape(16, K, -1).astype(np.float32); T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); Edt = T(Ed)
    def agree(Wx):
        with torch.no_grad(): return float((torch.cat([model.fc(torch.einsum('nhk,hko->no', Edt[a:a + 5000], T(Wx))).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    a_src = agree(W); S = Ef[:20000]; W1 = np.zeros_like(W); W1[:, -2:] = W[:, -2:]; nst = 0
    for o in range(W.shape[2]):
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
    np.savez(OUT / f'{out}_model.npz', W=best[1].reshape(16 * K, -1), kn=kn, knv=kn[:, :NV], nv=NV, basis='extended', direct=True, shared=True, logits=LOGITS, **XA)
    r = dict(experiment=out, src=src, start_src=a_src, start=a1, best=best[0], best_epoch=best[2], statements=nst, coefficients=int((best[1][:, :K - 2] != 0).sum()), seconds=time.time() - t0)
    (OUT / f'{out}_shared.json').write_text(json.dumps(r, indent=1)); log(f'{out}: {nst} statements (one per neuron and input, shared by the heads; {r["coefficients"]} per-head coefficients): {100 * best[0]:.2f}% (from {src} {100 * a_src:.2f}%)'); return r


def prune_pairs(tol=0.1, rounds=8, epochs=40, n_fit=100000, n_dev=20000, src=None, out=None, device='mps', log=print):
    """(neuron, input) pairs removed smallest first (size = spread of the pair's contribution over the jets); how many per round by
    bisection on the measured agreement (≤ tol points), then re-tuned — no overshoot from adding up single-pair costs"""
    import torch
    src, out = src or f'{PRE}o', out or f'{PRE}p'
    t0 = time.time(); model, Ef, Ed, Hf, Lf, ref, kn, _ = load(n_fit, n_dev, device, log); K = Ef.shape[2]
    P = np.load(OUT / f'{src}_model.npz'); W = P['W'].reshape(16, K, -1).astype(np.float32); NO = W.shape[2]; T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device); Edt = T(Ed)
    def agree(Wx):
        with torch.no_grad(): return float((torch.cat([model.fc(torch.einsum('nhk,hko->no', Edt[a:a + 5000], T(Wx))).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
    group = lambda f: [f] + [NV + k * NV + f for k in range(5)] + [6 * NV + k * NV + f for k in range(5)]
    keep = np.array([[np.any(W[:, group(f), o]) for f in range(NV)] for o in range(NO)])
    a_ref = a_now = agree(W); path = [dict(pairs=int(keep.sum()), agreement=a_now)]; log(f'  start: {int(keep.sum())} (neuron, input) pairs, {100 * a_now:.2f}%')
    Es = Ed[:5000]
    for rnd in range(rounds):
        prs = [(o, f) for o in range(NO) for f in np.flatnonzero(keep[o])]
        size = np.array([float(np.einsum('nhk,hk->n', Es[:, :, group(f)], W[:, group(f), o]).std()) for o, f in prs]); order = np.argsort(size)
        def after(m):
            Wt = W.copy()
            for j in order[:m]: o, f = prs[j]; Wt[:, group(f), o] = 0
            return Wt
        lo, hi = 0, len(order)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if a_now - agree(after(mid)) <= tol / 100: lo = mid
            else: hi = mid
        if lo == 0: log('  nothing removable within the tolerance: stop'); break
        for j in order[:lo]: o, f = prs[j]; keep[o, f] = False
        W = after(lo); mask = (W != 0).astype(np.float32); mask[:, -2:] = 1
        best = tune(model, Ef, Ed, Hf, Lf, ref, W, mask, epochs, device=device, log=log, label=f'prune round {rnd + 1}'); W = best[1] * mask; a_now = agree(W)
        path.append(dict(round=rnd + 1, removed=int(lo), pairs=int(keep.sum()), agreement=a_now)); np.savez(OUT / f'{out}_model.npz', W=W.reshape(16 * K, -1), kn=kn, knv=kn[:, :NV], nv=NV, basis='extended', direct=True, keep=keep, logits=LOGITS, **XA)
        log(f'  round {rnd + 1}: removed {lo} (neuron, input) pairs → {int(keep.sum())} kept ({keep.sum(1).min()}–{keep.sum(1).max()} inputs per neuron), re-tuned {100 * a_now:.2f}% (start {100 * a_ref:.2f}%), {time.time() - t0:.0f} s')
        if a_now < a_ref - tol / 100: log('  below the tolerance: stop'); break
    r = dict(tol=tol, start=a_ref, final=a_now, inputs_per_neuron=keep.sum(1).tolist(), path=path, seconds=time.time() - t0)
    (OUT / f'{out}_pruned.json').write_text(json.dumps(r, indent=1)); log(f'pruned {out}: {int(keep.sum())} (neuron, input) pairs, {100 * a_now:.2f}%'); return r


def sparsify(src=None, out=None, target=0.81, epochs=20, n_fit=100000, n_dev=20000, device='mps', log=print, lams=(3e-6, 1e-5, 3e-5, 1e-4, 3e-4)):
    """L1 path: re-tune with a penalty on the size of every statement (coefficient × spread of its pooled term) at increasing
    strengths; statements whose size falls below 1 % of the largest are dropped and the rest re-tuned without the penalty;
    the sparsest model with agreement ≥ target is kept"""
    src, out = src or f'{PRE}q', out or f'{PRE}s'
    if os.environ.get('L1S'): lams = tuple(float(v) for v in os.environ['L1S'].split(','))
    t0 = time.time(); model, Ef, Ed, Hf, Lf, ref, kn, _ = load(n_fit, n_dev, device, log); K = Ef.shape[2]
    W = np.load(OUT / f'{src}_model.npz')['W'].reshape(16, K, -1).astype(np.float32); sdc = Ef[:20000].std(0); sdc = np.where(sdc < 1e-6, 1.0, sdc)
    n0 = int((W[:, :K - 2] != 0).sum()); path = []; best = None; log(f'  start {src}: {n0} statements')
    for lam_ in lams:
        m0 = (W != 0).astype(np.float32); m0[:, -2:] = 1
        b1 = tune(model, Ef, Ed, Hf, Lf, ref, W, m0, epochs, device=device, log=log, label=f'L1 {lam_:g}', l1=lam_)
        Wl = b1[1]; size = np.abs(Wl * sdc[..., None]); size[:, K - 2:] = np.inf; keep = size > 0.01 * size[:, :K - 2].max()
        Wk = Wl * keep; mk = keep.astype(np.float32); mk[:, -2:] = 1
        b2 = tune(model, Ef, Ed, Hf, Lf, ref, Wk, mk, epochs, device=device, log=log, label=f'after L1 {lam_:g}')
        ns = int((b2[1][:, :K - 2] != 0).sum()); path.append(dict(lam=lam_, statements=ns, agreement=b2[0]))
        log(f'  L1 {lam_:g}: {ns} statements (from {n0}), {100 * b2[0]:.2f}%, {time.time() - t0:.0f} s')
        if b2[0] >= target and (best is None or ns < best[1]): best = (b2[1], ns, b2[0], lam_)
        if b2[0] < target - 0.01: break
    if best is None: log('  no setting reached the target'); best = (W, n0, None, None)
    np.savez(OUT / f'{out}_model.npz', W=best[0].reshape(16 * K, -1), kn=kn, knv=kn[:, :NV], nv=NV, basis='extended', direct=True, logits=LOGITS, **XA)
    r = dict(src=src, target=target, statements_start=n0, statements=best[1], agreement=best[2], lam=best[3], path=path, seconds=time.time() - t0)
    (OUT / f'{out}_sparsify.json').write_text(json.dumps(r, indent=1)); log(f'{out}: {best[1]} statements (from {n0}), {100 * (best[2] or 0):.2f}% (L1 {best[3]})'); return r


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'one_term': one_term(*(int(v) for v in a))
    elif cmd == 'prune_pairs': prune_pairs(*(float(v) if i_ == 0 else int(v) if v.lstrip('-').isdigit() else v for i_, v in enumerate(a)))
    elif cmd == 'sparsify': sparsify(*(a[:2] if a else []), *(float(v) for v in a[2:3]))
    elif cmd == 'share': share(*a[:2], *(int(v) for v in a[2:]))
    elif cmd == 'prune_terms': prune_terms(*(float(v) if i == 0 else int(v) if v.lstrip('-').isdigit() else v for i, v in enumerate(a)))
    else: prune(*(float(v) if i == 0 else int(v) for i, v in enumerate(a)))
