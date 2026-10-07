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
    knt, mut, sdt, kvt = T(PA['kn']), T(PA['mu']), T(PA['sd']), T(knv); Kc = PA['kn'].shape[0]
    def phi(F): return torch.cat([(F - mut) / sdt] + [torch.clamp(F - knt[k], min=0) / sdt for k in range(Kc)], -1)
    def tv(X): return torch.cat([X] + [torch.clamp(X - kvt[k], min=0) for k in range(5)] + [torch.clamp(kvt[k] - X, min=0) for k in range(5)], -1)      # (n, P, 418)
    Wc = torch.nn.Parameter(T(PA['coef'].T)); Wz = torch.nn.Parameter(T(PA['cz'].T)); mask = T((W0 != 0).astype(np.float32)); mask[:, -2:] = 1
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
    params = [Wc, Wz, Wv]; a0 = agree(); best = (a0, [p.detach().clone() for p in params], 0); path = [(0, a0)]
    log(f'  start ({at} score formulas, {wt} neuron formulas, last layer): {100 * a0:.2f}%, {time.time() - t0:.0f} s')
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
    np.savez(OUT / f'{tag}_model.npz', W=(Wv2 * mask.cpu().numpy()).reshape(16 * K, 128), kn=PW['kn'], knv=knv, nv=NV, basis='extended', direct=True, coef=Wc2.T, cz=Wz2.T, akn=PA['kn'], amu=PA['mu'], asd=PA['sd'])
    r = dict(experiment=tag, alpha=at, w=wt, n_fit=n_fit, epochs=epochs, start=a0, best=best[0], best_epoch=best[2], path=path, seconds=time.time() - t0)
    (OUT / f'{tag}_direct_alpha.json').write_text(json.dumps(r, indent=1)); log(f'{tag}: direct-128 neurons with formula attention, all tuned: {100 * best[0]:.2f}% (start {100 * a0:.2f}%)'); return r


if __name__ == '__main__':
    run(*(int(v) for v in sys.argv[1:]))
