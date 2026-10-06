"""Step 1 on a GPU (torch, JETDISTILL_STEP1_DEVICE=mps|cuda): the forward selection of mars.fit_neuron for all neurons at
once (in lockstep), with the same candidates and choices:
  candidates  every threshold term of the jet-level observables; the N_TOP per-particle / pair quantities screened for
              the current residual (on the first N_SCREEN selection jets); products of a chosen threshold term with a
              threshold term of one of the 10 quantities most correlated with the residual
  gain        (Cpᵀ r)² / |Cp|², Cp = the candidate columns minus their projection on the chosen terms (float32, as mars);
              the best column of each quantity, then the best quantity, first in mars's order (jet-level, screened,
              products)
  then        least squares on all fitting jets and R² on the validation jets (float64, CPU), the same stopping rule
The scoring of the candidates (the bulk of the work) runs on the GPU, one batched call per neuron and step."""
import time
import numpy as np
from .mars import term_value, thresholds, MAX_TERMS, STALL, N_SCREEN, N_TOP
from .formula import act


def select(Z, Q, keys, knots, sel, screen, Qblock_sel, device='mps', log=print, chunk=8192, neurons=None):
    """{neuron: (terms, coef, path)} for the neurons that are not constant. Qblock_sel: (ids, (len(sel), ids) float32),
    the per-particle / pair quantities on the selection jets."""
    import torch
    dt = torch.float32; S = len(sel); t0 = time.time(); keys = list(keys); kidx = {k: i for i, k in enumerate(keys)}
    # ---- jet-level candidate columns on the selection jets (as mars.fit_neuron's cache)
    starts, descs, mats = [], [], []; c = 0
    for k in keys:
        v = np.asarray(Q['fit'][k], np.float64)[sel]; kn = knots[k]
        M = np.stack([v] + [np.maximum(0, v - t) for t in kn] + [np.maximum(0, t - v) for t in kn], 1).astype(np.float32)
        mats.append(M); starts.append(c); descs += [(k, 'lin', None)] + [(k, 'gt', t) for t in kn] + [(k, 'lt', t) for t in kn]; c += M.shape[1]
    ends = starts[1:] + [c]; CJ = torch.from_numpy(np.concatenate(mats, 1)).to(device); del mats
    Lraw = CJ[:, torch.tensor(starts, device=device)]; Lc = Lraw - Lraw.mean(0); Ln = Lc / torch.clamp(Lc.norm(dim=0), min=1e-30)
    lin_ok = (Lraw.max(0).values - Lraw.min(0).values) > 0
    # ---- per-particle / pair quantities: screening matrices, raw values on the selection jets
    bids, Bsel = Qblock_sel; bcol = {k: i for i, k in enumerate(bids)}
    BV = torch.from_numpy(np.ascontiguousarray(Bsel)).to(device)
    SV = torch.from_numpy(screen.V).to(device); SA = torch.from_numpy(screen.A).to(device); sids = screen.ids
    KN = {}
    def knots_of(k):
        if k not in KN: KN[k] = screen.knots.setdefault(k, thresholds(screen.Qfit[k], k in screen.mids, screen.offer))
        return KN[k]
    CC = {}
    def cand(k):                                   # (S, 1 + 2 nk) on the GPU and its descriptors (block quantities cached)
        if k in kidx: i = kidx[k]; return CJ[:, starts[i]:ends[i]], descs[starts[i]:ends[i]]
        if k not in CC:
            if len(CC) > 3000: CC.pop(next(iter(CC)))
            kn = knots_of(k); v = BV[:, bcol[k]]; T = torch.tensor(kn, dtype=dt, device=device)
            M = torch.cat([v[:, None], torch.clamp(v[:, None] - T[None], min=0), torch.clamp(T[None] - v[:, None], min=0)], 1)
            CC[k] = (M, [(k, 'lin', None)] + [(k, 'gt', t) for t in kn] + [(k, 'lt', t) for t in kn])
        return CC[k]
    def gains(C, Qb, r):                           # mars.fit_neuron's gain, column-chunked
        out = []
        for a in range(0, C.shape[1], chunk):
            Cc = C[:, a:a + chunk]; Cp = Cc - Qb @ (Qb.T @ Cc); nn = (Cp * Cp).sum(0)
            out.append(torch.where(nn > 1e-9, (Cp.T @ r) ** 2 / torch.clamp(nn, min=1e-12), torch.zeros_like(nn)))
        return torch.cat(out)
    def best_of(g, st_, en_):                      # per quantity the best column, then the best quantity (first in order)
        if not len(st_): return -np.inf, None
        mx = np.maximum.reduceat(g, st_); q = int(np.argmax(mx)); col = st_[q] + int(np.argmax(g[st_[q]:en_[q]]))
        return float(mx[q]), col
    QS = {}
    def qsel(q):                                   # a quantity on the selection jets (float64), cached
        if q not in QS: QS[q] = np.asarray(Q['fit'][q], np.float64)[sel]
        return QS[q]
    # ---- the neurons
    NF = len(Z['fit']); st = {}
    for j in (neurons if neurons is not None else range(Z['fit'].shape[1])):
        zf = Z['fit'][:, j]
        if np.ptp(zf) < 1e-9: continue
        Bf = np.empty((NF, MAX_TERMS + 1), np.float32); Bf[:, 0] = 1; Bd = np.empty((len(Z['dev']), MAX_TERMS + 1), np.float32); Bd[:, 0] = 1
        G = np.zeros((MAX_TERMS + 1, MAX_TERMS + 1)); G[0, 0] = NF; bz = np.zeros(MAX_TERMS + 1); bz[0] = zf.sum()
        st[j] = dict(chosen=[], basis=[np.ones(S) / np.sqrt(S)], pv=[], Bf=Bf, Bd=Bd, G=G, bz=bz, ys=zf[sel].astype(np.float64),
                     zf=zf, hd=act(Z['dev'][:, j]), path=[], best=(-np.inf, 0), stall=0, done=False)
    log(f'GPU step 1: {len(st)} neurons, {CJ.shape[1]} jet-level candidate columns, {len(sids)} screened quantities, {time.time() - t0:.0f} s')
    for step in range(MAX_TERMS):
        act_ = [j for j, s in st.items() if not s['done']]
        if not act_: break
        for j in act_:
            s = st[j]; Qb64 = np.stack(s['basis'], 1); r64 = s['ys'] - Qb64 @ (Qb64.T @ s['ys'])
            Qb = torch.from_numpy(Qb64.astype(np.float32)).to(device); r = torch.from_numpy(r64.astype(np.float32)).to(device)
            # 1. jet-level quantities
            g, col = best_of(gains(CJ, Qb, r).cpu().numpy(), np.array(starts), np.array(ends))
            best_g, best_t = g, dict(q=descs[col][0], kind=descs[col][1], t=descs[col][2])
            # 2. the screened per-particle / pair quantities
            rs = r[:N_SCREEN] - r[:N_SCREEN].mean(); sc = torch.maximum((rs @ SV).abs(), (rs @ SA).abs())
            top_ids = [sids[i] for i in torch.argsort(-sc)[:N_TOP].tolist()]
            Ms, Ds = zip(*[cand(k) for k in top_ids]) if top_ids else ((), ())
            if Ms:
                w = np.array([m.shape[1] for m in Ms]); st_ = np.concatenate([[0], np.cumsum(w)[:-1]]); en_ = st_ + w
                gb, colb = best_of(gains(torch.cat(Ms, 1), Qb, r).cpu().numpy(), st_, en_)
                if gb > best_g:
                    q = int(np.searchsorted(en_, colb, side='right')); d = Ds[q][colb - st_[q]]; best_g, best_t = gb, dict(q=d[0], kind=d[1], t=d[2])
            # 3. products: chosen threshold terms × threshold terms of the 10 quantities most correlated with the residual
            rc = r - r.mean(); rn = rc / torch.clamp(rc.norm(), min=1e-30)
            rel = torch.where(lin_ok, (Ln.T @ rn).abs(), torch.zeros_like(lin_ok, dtype=dt))
            if Ms:
                V0 = torch.stack([m[:, 0] for m in Ms], 1); Vc = V0 - V0.mean(0); okb = (V0.max(0).values - V0.min(0).values) > 0
                rel = torch.cat([rel, torch.where(okb, (Vc.T @ rn).abs() / torch.clamp(Vc.norm(dim=0), min=1e-30), torch.zeros_like(okb, dtype=dt))])
            names = keys + top_ids; relv = rel.cpu().numpy(); top = [names[i] for i in sorted(range(len(names)), key=lambda i: -relv[i])[:10]]
            parents = [(i, tm) for i, tm in enumerate(s['chosen']) if tm.get('q2') is None and tm['kind'] != 'lin']
            if parents:
                TM, TD = zip(*[cand(k) for k in top]); TT = torch.cat([m[:, 1:] for m in TM], 1)          # threshold columns only
                tw = np.array([m.shape[1] - 1 for m in TM]); tst = np.concatenate([[0], np.cumsum(tw)[:-1]])
                PV = torch.stack([s['pv'][i] for i, _ in parents], 1)                                         # (S, parents)
                X = (PV[:, :, None] * TT[:, None, :]).reshape(S, -1); gp = gains(X, Qb, r).cpu().numpy().reshape(len(parents), -1)
                for pi, (_, tm) in enumerate(parents):          # mars: per parent, per related quantity (not its own) the best column
                    for ki, k in enumerate(top):
                        if k == tm['q']: continue
                        seg = gp[pi, tst[ki]:tst[ki] + tw[ki]]; ii = int(np.argmax(seg)); gg = float(seg[ii])
                        if gg > best_g: d = TD[ki][ii + 1]; best_g, best_t = gg, dict(q=tm['q'], kind=tm['kind'], t=tm['t'], q2=k, kind2=d[1], t2=d[2])
            if best_g <= 0: s['done'] = True; continue
            # add the term: basis on the selection jets (Gram-Schmidt, float64), least squares on all fitting jets, R² on validation
            tm = best_t; s['chosen'].append(tm)
            u = term_value(tm, {q: qsel(q) for q in (tm['q'], tm.get('q2')) if q})
            s['pv'].append(torch.from_numpy(u.astype(np.float32)).to(device))
            for _ in range(2): u = u - Qb64 @ (Qb64.T @ u)
            nu = np.linalg.norm(u)
            if nu > 1e-12: s['basis'].append(u / nu)
            t_ = len(s['chosen']); b = term_value(tm, Q['fit']); s['Bf'][:, t_] = b; s['Bd'][:, t_] = term_value(tm, Q['dev'])
            g_ = s['Bf'][:, :t_ + 1].T.astype(np.float64) @ b; s['G'][t_, :t_ + 1] = g_; s['G'][:t_ + 1, t_] = g_; s['bz'][t_] = b @ s['zf']   # incremental normal equations
            Gt = s['G'][:t_ + 1, :t_ + 1]; d = np.sqrt(np.maximum(np.diag(Gt), 1e-300))          # columns scaled to unit norm, then least squares on the Gram matrix
            coef = np.linalg.lstsq(Gt / d[:, None] / d[None], s['bz'][:t_ + 1] / d, rcond=1e-13)[0] / d
            r2 = 1 - ((s['hd'] - act(s['Bd'][:, :t_ + 1].astype(np.float64) @ coef)) ** 2).mean() / max(s['hd'].var(), 1e-12); s['path'].append(round(float(r2), 4))
            if r2 > s['best'][0] + 1e-4: s['best'], s['stall'] = (r2, len(s['chosen'])), 0
            else: s['stall'] += 1
            if s['stall'] >= STALL or r2 > .9995: s['done'] = True
        if step % 10 == 9 or step == 0: log(f'  GPU step 1, term {step + 1}: {sum(1 for s in st.values() if not s["done"])} neurons still growing, {time.time() - t0:.0f} s')
    out = {}
    for j, s in st.items():
        terms = s['chosen'][:s['best'][1]]
        B = np.stack([np.ones(NF)] + [term_value(t, Q['fit']) for t in terms], 1)        # the final coefficients: least squares in float64, as mars
        out[j] = (terms, np.linalg.lstsq(B, s['zf'], rcond=None)[0], s['path'])
    return out
