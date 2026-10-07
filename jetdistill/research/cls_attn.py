"""Step 1 (JEDI spirit) — ParT's class attention read as physics. For both class blocks and each of the 8 heads:
the score of each particle s_hi = q_h·k_hi/4 (and of the class token itself), the weights α = softmax over
[class token, particles].
  - what each head selects: α-weighted means of particle properties vs the plain mean; the self-weight
  - how well a hinge formula of physics per particle reproduces the score (R², least squares on fitting jets,
    measured on other jets) with inputs: own (ParT's inputs + pT rank + 20 jet quantities), + neighbourhood (nbr),
    + ParT pair-kernel context (pk); and how close the pooled sums Σ α f come with the formula weights

  python -m jetdistill.research.cls_attn [n_fit n_check]"""
import json, sys, time, pathlib
import numpy as np
from ..config import RESULTS
from ..pipeline import jets
from ..part.clsfit import particle_features, PFEAT, CTX
from ..part.network import ParTNetwork
from .nbr import nbr_features, pk_features, NBR, PK

OUT = pathlib.Path(__file__).resolve().parents[2] / 'research' / 'results'


def scores(model, X, M, device='mps'):
    """(blocks 2, N, heads 8, 1 + P) attention scores of the class token on [itself, the particles]; −inf for empty slots"""
    import torch
    N = len(M); S = np.full((2, N, 8, 129), -np.inf, np.float32)
    with torch.no_grad():
        for a in range(0, N, 1000):
            x = torch.from_numpy(np.asarray(X[a:a + 1000], np.float32)).to(device).permute(1, 0, 2); m = torch.from_numpy(M[a:a + 1000]).to(device)
            n = x.shape[1]; cls = model.cls_token.expand(1, n, -1)
            for b, blk in enumerate(model.cls_blocks):
                u = blk.pre_attn_norm(torch.cat([cls, x], 0)); W, bi = blk.attn.in_proj_weight, blk.attn.in_proj_bias
                q = (u[:1] @ W[:128].T + bi[:128]).view(1, n, 8, 16); k = (u @ W[128:256].T + bi[128:256]).view(-1, n, 8, 16)
                s = torch.einsum('qnhd,pnhd->nhp', q, k) / 4.0
                s = s.masked_fill(~torch.cat([torch.ones_like(m[:, :1]), m], 1)[:, None], -float('inf'))
                S[b, a:a + n] = s.cpu().numpy(); cls = blk(x, x_cls=cls, padding_mask=~m)
    return S


def hinge(F, kn):
    return np.concatenate([np.ones((len(F), 1), np.float32), F] + [np.maximum(0, F - t) for t in kn] + [np.maximum(0, t - F) for t in kn], 1).astype(np.float32)


def run(n_fit=20000, n_chk=10000, device='mps', log=print):
    t0 = time.time(); D = RESULTS / '_cls' / 'full' / 'fit'; X = np.load(D / 'x_f16.npy', mmap_mode='r'); M = np.load(D / 'mask.npy')
    model = ParTNetwork('full').model; model.eval(); J = jets('full', 'fit'); n = n_fit + n_chk
    S = scores(model, X[:n], M[:n], device); A = np.exp(S - S.max(-1, keepdims=True)); A /= A.sum(-1, keepdims=True)   # weights
    F0, ok = particle_features(J, np.arange(n)); assert (ok == M[:n]).all()
    Fn = nbr_features(J['x'][:n], J['ext'][:n], J['jet'][:n]); Fp = pk_features(J['x'][:n], J['ext'][:n], J['jet'][:n], model, device)
    names = PFEAT + CTX + NBR + PK; Fall = np.concatenate([F0, Fn, Fp], -1)
    res = dict(n_fit=n_fit, n_check=n_chk, blocks=[])
    # what each head selects: α-weighted mean of a few properties vs the plain mean over particles
    props = dict(zip(['ln pT/pT_jet', 'charge≠0', 'photon', 'lepton', 'tanh d0 (abs)', 'ΔR', 'ln(1+rank)', 'prong pT share', 'ΔR nearest displaced'],
                     [F0[..., 2], (F0[..., 7] != 0).astype(np.float32), F0[..., 10], F0[..., 11] + F0[..., 12], np.abs(F0[..., 13]), F0[..., 4], F0[..., 17], Fn[..., 13], Fn[..., 9]]))
    plain = {k: float((v * ok).sum() / ok.sum()) for k, v in props.items()}
    sets = dict(own=slice(0, F0.shape[-1]), own_nbr=slice(0, F0.shape[-1] + Fn.shape[-1]), own_nbr_pk=slice(0, Fall.shape[-1]))
    okf, okc = ok[:n_fit], ok[n_fit:]
    for b in range(2):
        blk = dict(block=b + 1, heads=[])
        for h in range(8):
            a = A[b, :, h]; self_w = a[:, 0]; ap = a[:, 1:]
            sel = {k: float((ap * v).sum() / ap.sum()) for k, v in props.items()}
            s = S[b, :, h, 1:]; r = dict(head=h + 1, self_weight_mean=float(self_w.mean()), self_weight_corr_multiplicity=float(np.corrcoef(self_w, ok.sum(1))[0, 1]),
                                         selects=sel, plain=plain, top1_share=float(ap.max(1).mean()), entropy_eff_n=float(np.exp(-(ap * np.log(ap + 1e-12)).sum(1)).mean()), r2={})
            for sname, sl in sets.items():
                Ff = Fall[:n_fit][okf][:, sl]; kn = np.quantile(Ff[::5], np.linspace(.1, .9, 5), axis=0).astype(np.float32)
                Bf = hinge(Ff, kn); yf = s[:n_fit][okf]; G = Bf.T.astype(np.float64) @ Bf; dsc = np.sqrt(np.maximum(np.diag(G), 1e-12))
                w = np.linalg.solve(G / dsc[:, None] / dsc[None] + 1e-6 * np.eye(len(dsc)), (Bf.T.astype(np.float64) @ yf) / dsc) / dsc
                Bc = hinge(Fall[n_fit:][okc][:, sl], kn); yc = s[n_fit:][okc]; pc = Bc @ w
                # within-jet R² (the softmax ignores per-jet offsets): centre score and prediction per jet
                jid = np.repeat(np.arange(n_chk), okc.sum(1)); cnt = np.bincount(jid); mu_y = np.bincount(jid, yc) / cnt; mu_p = np.bincount(jid, pc) / cnt
                yc0, pc0 = yc - mu_y[jid], pc - mu_p[jid]; c = (yc0 @ pc0) / max(pc0 @ pc0, 1e-12)
                r2w = 1 - ((yc0 - c * pc0) ** 2).sum() / (yc0 ** 2).sum()
                r['r2'][sname] = float(r2w)
            blk['heads'].append(r)
            log(f"  class block {b + 1} head {h + 1}: self-weight {r['self_weight_mean']:.3f} (corr mult {r['self_weight_corr_multiplicity']:+.2f}), eff. particles {r['entropy_eff_n']:.1f}, "
                f"within-jet score R² own {r['r2']['own']:.3f} / +nbr {r['r2']['own_nbr']:.3f} / +pk {r['r2']['own_nbr_pk']:.3f} | "
                + ', '.join(f"{k} {sel[k]:.2f} ({plain[k]:.2f})" for k in ('ln pT/pT_jet', 'charge≠0', 'photon', 'tanh d0 (abs)', 'ΔR', 'prong pT share')))
        res['blocks'].append(blk)
    res['seconds'] = time.time() - t0
    return res


if __name__ == '__main__':
    a = [int(v) for v in sys.argv[1:]]; r = run(*a); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'S1_cls_attention.json').write_text(json.dumps(r, indent=1)); print('RESULT done', f"{r['seconds']:.0f} s")
