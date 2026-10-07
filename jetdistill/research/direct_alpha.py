"""A11 — the direct-128 model with FORMULA attention: α from the A2h score formulas (per head a hinge formula of 314
per-particle inputs; α_cls from the jet-level formula), the 128 neurons = Σ_h [Σ_i α_hi φ(x_i)·W_h + α_h,cls c_h] + b
(S15p's sparse W as the start), ParT's last layer. Scores, α_cls formula and W tuned together toward ParT's
probabilities. Nothing of ParT but its last layer: the fully-formula counterpart of S15.

  CTX_HOPS=1 python -m jetdistill.research.direct_alpha [n_fit n_dev epochs]"""
import json, os, sys, time
import numpy as np
from ..pipeline import jets
from ..part.network import ParTNetwork
from .heads import extract, rows_of_split, OUT
from .heads_attn import feats
from .alpha_fit import rel_feats
from .alpha_e2e import zfeat

NV = 38


def run(n_fit=30000, n_dev=20000, epochs=20, n_jet=100000, lr=3e-4, bs=500, device='mps', log=print):
    import torch
    t0 = time.time(); at, wt, tag = os.environ.get('ALPHA_TAG', 'A2h'), os.environ.get('W_TAG', 'S15p'), os.environ.get('OUT_TAG', 'A11')
    model = ParTNetwork('full').model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    _, _, Mfa, Lfa, _ = extract(model, 'fit', n_jet, device); _, _, Md, Ld, _ = extract(model, 'dev', n_dev, device); ref = Ld.argmax(1)
    Jf, Jd = jets('full', 'fit'), jets('full', 'dev'); rf, rd = rows_of_split('fit', n_fit), rows_of_split('dev', n_dev)
    Ff, okf = feats(Jf, rf, model); Fd, okd = feats(Jd, rd, model); Ff = np.concatenate([Ff, rel_feats(Ff, okf)], -1); Fd = np.concatenate([Fd, rel_feats(Fd, okd)], -1)
    Zf_all, Zd = zfeat(Jf, Jd, n_jet, rd, Mfa, Md); Zf = Zf_all[:n_fit]
    PA = np.load(OUT / f'{at}_alpha.npz'); PW = np.load(OUT / f'{wt}_model.npz'); K = 11 * NV + 2; W0 = PW['W'].reshape(16, K, 128).astype(np.float32); knv = PW['knv']
    T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
    knt, mut, sdt, kvt = T(PA['kn']), T(PA['mu']), T(PA['sd']), T(knv); Kc = PA['kn'].shape[0]; EXT = 'W' in PA.files            # W-family α (extended basis with intercept) or A2h-family
    def phi(F):
        if EXT: return torch.cat([torch.ones_like(F[..., :1]), (F - mut) / sdt] + [torch.clamp(F - knt[k], min=0) / sdt for k in range(Kc)] + [torch.clamp(knt[k] - F, min=0) / sdt for k in range(Kc)], -1)
        return torch.cat([(F - mut) / sdt] + [torch.clamp(F - knt[k], min=0) / sdt for k in range(Kc)], -1)
    def tv(X): return torch.cat([X] + [torch.clamp(X - kvt[k], min=0) for k in range(5)] + [torch.clamp(kvt[k] - X, min=0) for k in range(5)], -1)      # (n, P, 418)
    Wc = torch.nn.Parameter(T(PA['W'] if EXT else PA['coef'].T)); Wz = torch.nn.Parameter(T(PA['cz'].T)); mask = T((W0 != 0).astype(np.float32)); mask[:, -2:] = 1
    sdc = np.ones((16, K), np.float32); Wv = torch.nn.Parameter(T(W0)); fc = model.fc
    def logits(F, ok, Z):
        n = F.shape[0]; P = int(ok.sum(1).max()); F, ok = F[:, :P].float(), ok[:, :P]
        s = (phi(F) @ Wc).masked_fill(~ok[..., None], -1e9); rel = torch.softmax(s, 1); acls = torch.sigmoid(-(Z @ Wz))        # (n, P, 16), (n, 16)
        al = (1 - acls)[:, None] * rel                                                                                            # (n, P, 16)
        pooled = torch.einsum('nph,npk->nhk', al, tv(F[..., :NV]))                                                                 # (n, 16, 418)
        E = torch.cat([pooled, acls[..., None], torch.ones_like(acls)[..., None]], -1)                                             # (n, 16, 420)
        return fc(torch.einsum('nhk,hko->no', E, Wv * mask))
    Fft, Fdt, okft, okdt = T(Ff, torch.float16), T(Fd, torch.float16), T(okf, torch.bool), T(okd, torch.bool); Zft, Zdt = T(Zf), T(Zd); pf = torch.softmax(T(Lfa[:n_fit]), 1); del Ff, Fd
    od, of = np.argsort(okd.sum(1)), np.argsort(okf.sum(1))
    def agree():
        with torch.no_grad():
            pred = np.empty(n_dev, int)
            for a in range(0, n_dev, bs):
                i = torch.from_numpy(od[a:a + bs]).to(device); pred[od[a:a + bs]] = logits(Fdt[i], okdt[i], Zdt[i]).argmax(1).cpu().numpy()
        return float((pred == ref).mean())
    a_s15 = agree(); log(f'  {wt} neuron formulas with the {at} formula α as they are: {100 * a_s15:.2f}%')
    if os.environ.get('JOINT_ONLY'):                                                 # C2: keep the given neuron statements (mask) and tune selection + neurons together
        a_ls = bw0 = a_s15; bw = (a_s15, Wv.detach().clone())
    else:
      log('  → the neuron formulas are refit (stage 1)')
      # stage 1: the pooled terms with the formula α (fixed), least squares of the 128 neurons on them (whitened ridge), then W tuned alone
      def pooled(Fx, okx, Zx, idx):
          out = np.zeros((len(idx), 16, K), np.float32)
          with torch.no_grad():
              for a in range(0, len(idx), bs):
                  i = torch.from_numpy(idx[a:a + bs]).to(device); F, ok, Z = Fx[i], okx[i], Zx[i]; P = int(ok.sum(1).max()); F, ok = F[:, :P].float(), ok[:, :P]
                  s = (phi(F) @ Wc).masked_fill(~ok[..., None], -1e9); rel = torch.softmax(s, 1); acls = torch.sigmoid(-(Z @ Wz)); al = (1 - acls)[:, None] * rel
                  out[idx[a:a + bs]] = torch.cat([torch.einsum('nph,npk->nhk', al, tv(F[..., :NV])), acls[..., None], torch.ones_like(acls)[..., None]], -1).cpu().numpy()
          return out
      Ef, Ed = pooled(Fft, okft, Zft, of), pooled(Fdt, okdt, Zdt, od); Hf = np.asarray(Jf['H'][rf], np.float32); Bf = Ef.reshape(n_fit, 16 * K)
      G = Bf.T.astype(np.float64) @ Bf / n_fit; d = np.sqrt(np.maximum(np.diag(G), 1e-12)); okc = d > 1e-9; Gs = G[np.ix_(okc, okc)] / d[okc][:, None] / d[okc][None]
      Wls = np.zeros((16 * K, 128)); Wls[okc] = np.linalg.solve(Gs + 1e-3 * np.eye(okc.sum()), (Bf[:, okc].T.astype(np.float64) @ Hf / n_fit) / d[okc][:, None]) / d[okc][:, None]
      with torch.no_grad(): Wv.data.copy_(T(Wls.reshape(16, K, 128).astype(np.float32))); mask.fill_(1)
      a_ls = agree(); log(f'  stage 1, least squares of the 128 neurons on the formula-α pooled terms ({n_fit} jets): {100 * a_ls:.2f}%, {time.time() - t0:.0f} s')
      Eft_, Edt_, Hft = T(Ef), T(Ed), T(Hf); vh = Hft.var(0) + 1e-6; sdc = T(np.where(Ef.std(0) < 1e-6, 1.0, Ef.std(0)).astype(np.float32)); U = torch.nn.Parameter((Wv.detach() * sdc[..., None]).clone()); optw = torch.optim.Adam([U], 3e-4)
      def agree_w(Wx):
          with torch.no_grad(): return float((torch.cat([fc(torch.einsum('nhk,hko->no', Edt_[a:a + 5000], Wx)).argmax(1) for a in range(0, n_dev, 5000)]).cpu().numpy() == ref).mean())
      bw = (a_ls, Wv.detach().clone())
      for ep in range(150):
          for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit, 5000)):
              y = torch.einsum('nhk,hko->no', Eft_[a:a + 5000], U / sdc[..., None]); loss = -(pf[a:a + 5000] * torch.log_softmax(fc(y), 1)).sum(1).mean() + 0.01 * ((y - Hft[a:a + 5000]) ** 2 / vh).mean()
              optw.zero_grad(); loss.backward(); optw.step()
          if ep % 10 == 9:
              ag = agree_w((U / sdc[..., None]).detach())
              if ag > bw[0]: bw = (ag, (U / sdc[..., None]).detach().clone())
      with torch.no_grad(): Wv.data.copy_(bw[1])
      log(f'  stage 1, neuron formulas tuned (α fixed): {100 * bw[0]:.2f}%, {time.time() - t0:.0f} s'); del Eft_, Edt_
    params = [Wc, Wz, Wv]; a0 = agree(); best = (a0, [p.detach().clone() for p in params], 0); path = [(0, a0)]
    log(f'  stage 2 start (everything tuned together): {100 * a0:.2f}%, {time.time() - t0:.0f} s')
    opt = torch.optim.Adam([{'params': [Wc, Wz], 'lr': lr}, {'params': [Wv], 'lr': lr / 3}])
    for ep in range(epochs):
        for a in np.random.default_rng(ep).permutation(np.arange(0, n_fit - bs + 1, bs)):
            i = torch.from_numpy(of[a:a + bs]).to(device); L = logits(Fft[i], okft[i], Zft[i]); loss = -(pf[i] * torch.log_softmax(L, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if device == 'mps': torch.mps.empty_cache()
        ag = agree(); path.append((ep + 1, ag))
        if ag > best[0]: best = (ag, [p.detach().clone() for p in params], ep + 1)
        log(f'  {tag} epoch {ep + 1}: {100 * ag:.2f}% (best {100 * best[0]:.2f}% at {best[2]}), {time.time() - t0:.0f} s')
    Wc2, Wz2, Wv2 = (p.cpu().numpy() for p in best[1])
    np.savez(OUT / f'{tag}_model.npz', W=(Wv2 * mask.cpu().numpy()).reshape(16 * K, 128), kn=PW['kn'], knv=knv, nv=NV, basis='extended', direct=True, coef=Wc2.T, cz=Wz2.T, akn=PA['kn'], amu=PA['mu'], asd=PA['sd'], alpha_from=tag)
    np.savez(OUT / f'{tag}_alpha.npz', **({'W': Wc2} if EXT else {'coef': Wc2.T}), cz=Wz2.T, kn=PA['kn'], mu=PA['mu'], sd=PA['sd'], basis='extended' if EXT else 'standard')       # the tuned selection formulas
    from .one_term import to_standard
    np.savez(OUT / f'{tag}_post_model.npz', W=to_standard((Wv2 * mask.cpu().numpy()).astype(np.float64), knv, NV).reshape(16 * (6 * NV + 2), 128), kn=PW['kn'])   # start for the neuron loop (direct_loop, DPRE=tag)
    r = dict(experiment=tag, alpha=at, w=wt, n_fit=n_fit, epochs=epochs, s15_as_is=a_s15, least_squares=a_ls, stage1=bw[0], start=a0, best=best[0], best_epoch=best[2], path=path, seconds=time.time() - t0)
    (OUT / f'{tag}_direct_alpha.json').write_text(json.dumps(r, indent=1)); log(f'{tag}: direct-128 neurons with formula attention, all tuned: {100 * best[0]:.2f}% (start {100 * a0:.2f}%)'); return r


def zfun(model, device='mps'):
    """the jet-level inputs of the class-token share formulas for any jets: z(J, rows, mask)"""
    from .ceiling_jet import matrix
    from sklearn.preprocessing import QuantileTransformer
    Jf = jets('full', 'fit'); _, _, Mf, _, _ = extract(model, 'fit', 100000, device); keys = [k for k in Jf['Q']]; Q100 = matrix(Jf, keys)[:100000]; okq = np.isfinite(Q100).all(0) & (Q100.std(0) > 0)
    qt = QuantileTransformer(n_quantiles=500, output_distribution='normal', subsample=50000, random_state=0).fit(Q100[:, okq])
    return lambda J, rows, M: np.concatenate([np.clip(qt.transform(matrix(J, keys)[rows][:, okq]), -5, 5), np.log(M.sum(1, keepdims=True)), np.ones((len(M), 1))], 1).astype(np.float32)


def formula_alpha(tag, J, rows, model, Z, device='mps', chunk=2000, bs=500):
    """the attention weights (2, n, 8, 129) of the jets `rows` from the selection formulas `tag`_alpha.npz (W- or A2h-family)"""
    import torch
    PA = np.load(OUT / f'{tag}_alpha.npz'); EXT = 'W' in PA.files; T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
    knt, mut, sdt, Kc = T(PA['kn']), T(PA['mu']), T(PA['sd']), PA['kn'].shape[0]; Wc = T(PA['W'] if EXT else PA['coef'].T); Wz = T(PA['cz'].T)
    def phi(F):
        if EXT: return torch.cat([torch.ones_like(F[..., :1]), (F - mut) / sdt] + [torch.clamp(F - knt[k], min=0) / sdt for k in range(Kc)] + [torch.clamp(knt[k] - F, min=0) / sdt for k in range(Kc)], -1)
        return torch.cat([(F - mut) / sdt] + [torch.clamp(F - knt[k], min=0) / sdt for k in range(Kc)], -1)
    A = np.zeros((2, len(rows), 8, 129), np.float32)
    with torch.no_grad():
        for a in range(0, len(rows), chunk):
            r = rows[a:a + chunk]; F, ok = feats(J, r, model); F = np.concatenate([F, rel_feats(F, ok)], -1)
            for c0 in range(0, len(r), bs):
                sl = slice(c0, c0 + bs); okc = ok[sl]; P = int(okc.sum(1).max()); Ft, okt = T(F[sl, :P]), T(okc[:, :P], torch.bool)
                s = (phi(Ft) @ Wc).masked_fill(~okt[..., None], -1e9); rel = torch.softmax(s, 1); acls = torch.sigmoid(-(T(Z[a + c0:a + c0 + len(okc)]) @ Wz))
                al = torch.cat([acls[:, :, None], (1 - acls)[:, :, None] * rel.permute(0, 2, 1)], 2).cpu().numpy()            # (b, 16, 1+P)
                for h in range(16): A[h // 8, a + c0:a + c0 + len(okc), h % 8, :1 + P] = al[:, h]
    return A


if __name__ == '__main__':
    run(*(int(v) for v in sys.argv[1:]))
