"""The layer after class attention, written with an attention of its own: ParT's 128 neurons as combinations of particle
terms pooled by 8 heads, each head a softmax weighting over the jet's particles (as ParT's class attention weights its
particles). Then ParT's own last layer.

  token i      the particle's 18 inputs (ParT's 17 and its pT rank) and 20 jet quantities, as hinge terms φᵢ (clsfit.basis)
  head h       score sₕᵢ = gₕ(particle i) + φᵢ·aₕ  (gₕ a fixed physics start: hardest, uniform, near the axis, charged,
               displaced, leptons, photons, most energetic; aₕ learned); as in ParT's class attention the head also attends
               to the class token itself, with a learned constant score bₕ: αₕ = softmax over [bₕ, sₕ₁, …, sₕₙ], so the
               weight α_cls,h = 1 − Σᵢ αₕᵢ is jet-dependent (how much the particles count at all)
  neurons      Σₕ [Σᵢ αₕᵢ φᵢ, α_cls,h]·Wₕ + c (α_cls,h·Wₕ,last: the class token's own value), then ParT's last layer
Fit: least squares of ParT's neurons on the pooled terms (the start weightings), then everything tuned toward ParT's
probabilities + λ·R on its neurons, full batch, terms whitened (Adam on correlated hinge terms is unstable).

  python -m jetdistill.part.attnfit tune [n_fit steps]"""
import json, sys, time
import numpy as np
from ..config import RESULTS
from .clsfit import particle_features, knots_of, basis, CTX, PFEAT, log

HEADS = ['hardest (ln pT)', 'uniform', 'near the axis (−ΔR)', 'charged', 'displaced (|tanh d0|)', 'leptons', 'photons', 'most energetic (ln E)']


def start_scores(F):
    """(…, 8) the fixed start of each head's score from the particle inputs (the PFEAT order of clsfit.particle_features)"""
    import torch
    lnpt, lne, dr, q = F[..., 0], F[..., 1], F[..., 4], F[..., 7]; lep = F[..., 11] + F[..., 12]; pho = F[..., 10]; td0 = F[..., 13]
    return torch.stack([lnpt, torch.zeros_like(lnpt), -5 * dr, 2 * (q != 0).float(), 3 * td0.abs(), 3 * lep, 2 * pho, lne], -1)


def tune(net='full', n_fit=100000, steps=800, lr=1e-3, lam=0.01, chunk=4000, n_ls=40000, device='mps', log=log, out=None):
    import torch
    from ..pipeline import jets
    from .network import ParTNetwork
    t0 = time.time(); H8 = len(HEADS); Jf, Jd = jets(net, 'fit'), jets(net, 'dev')
    Kw = torch.from_numpy(np.asarray(ParTNetwork(net).last[0], np.float32)).to(device); bw = torch.from_numpy(np.asarray(ParTNetwork(net).last[1], np.float32)).to(device)
    # thresholds and whitening of the per-particle terms (20k jets)
    F0, ok0 = particle_features(Jf, np.arange(20000)); knots = knots_of(F0, ok0)
    B0 = basis(torch.from_numpy(F0[ok0]).to(device), knots); K = B0.shape[1]; Np = B0.shape[0]
    G = (B0.T @ B0).cpu().numpy().astype(np.float64); del B0
    d = np.sqrt(np.maximum(np.diag(G), 1e-12)); ev, Qe = np.linalg.eigh(G / d[:, None] / d[None]); ev = np.maximum(ev, 1e-6 * ev.max())
    Mt = torch.from_numpy(((Qe / np.sqrt(ev)[None] / d[:, None]) * np.sqrt(Np)).astype(np.float32)).to(device)     # whitened terms: basis @ Mt
    def prep(J, n):
        r = np.arange(min(n, len(J['y']))); F, ok = particle_features(J, r); o = np.argsort(ok.sum(1), kind='stable')
        return dict(F=torch.from_numpy(F[o]).to(device, torch.float16), ok=torch.from_numpy(ok[o]).to(device), n=len(r), o=o,
                    P=torch.from_numpy(np.asarray(J['P'][r][o], np.float32)).to(device), H=torch.from_numpy(np.asarray(J['H'][r][o], np.float32)).to(device),
                    net=J['net'][r][o], y=J['y'][r][o])
    tr, va = prep(Jf, n_fit), prep(Jd, len(Jd['y'])); vn = tr['H'].var(0) + 1e-6
    A = torch.zeros((K, H8), device=device)                                  # learned part of the head scores (whitened terms)
    bc = torch.zeros(H8, device=device)                                      # each head's score for the class token itself
    def pooled(S, a, b_, A_, bc_):
        ok = S['ok'][a:b_]; P = int(ok.sum(1).max()); F = S['F'][a:b_, :P].float(); m = ok[:, :P]
        Phi = basis(F, knots) @ Mt                                           # (n, P, K) whitened terms
        sc = (start_scores(F) + Phi @ A_).masked_fill(~m[..., None], -1e9)  # (n, P, 8)
        sc = torch.cat([bc_.expand(len(m), 1, -1), sc], 1); al = torch.softmax(sc, 1)    # (n, 1 + P, 8): the class token itself, then the particles
        pp = torch.einsum('nph,npk->nhk', al[:, 1:], Phi)                    # (n, 8, K) the particles' pooled terms
        return torch.cat([pp, al[:, 0, :, None]], 2).reshape(len(m), -1)    # (n, 8 (K + 1)): + each head's self-weight
    # least-squares start of the neuron weights on the pooled terms (start weightings)
    with torch.no_grad():
        Xl = torch.cat([pooled(tr, a, min(a + chunk, n_ls), A, bc) for a in range(0, min(n_ls, tr['n']), chunk)])
        Xl = torch.cat([Xl, torch.ones_like(Xl[:, :1])], 1); Gx = (Xl.T @ Xl).cpu().numpy().astype(np.float64); bx = (Xl.T @ tr['H'][:len(Xl)]).cpu().numpy().astype(np.float64)
    dx = np.sqrt(np.maximum(np.diag(Gx), 1e-12)); W0 = np.linalg.lstsq(Gx / dx[:, None] / dx[None] + 1e-6 * np.eye(len(dx)), bx / dx[:, None], rcond=None)[0] / dx[:, None]
    W = torch.from_numpy(W0[:-1].astype(np.float32)).to(device); c = torch.from_numpy(W0[-1].astype(np.float32)).to(device); del Xl
    log(f'attention fit: {len(PFEAT)} particle inputs + {len(CTX)} jet quantities, {K} terms per particle, {H8} heads (+ class-token self-weights), {W.numel() + A.numel() + bc.numel()} coefficients; start {time.time() - t0:.0f} s')
    params = [A.requires_grad_(True), bc.requires_grad_(True), W.requires_grad_(True), c.requires_grad_(True)]
    def run(S, a, b_, ps):
        h = pooled(S, a, b_, ps[0], ps[1]) @ ps[2] + ps[3]; return h, h @ Kw + bw
    def score(ps):
        with torch.no_grad():
            pred = np.concatenate([run(va, a, a + chunk, ps)[1].argmax(1).cpu().numpy() for a in range(0, va['n'], chunk)])
        return float((pred == va['net']).mean()), float((pred == va['y']).mean())
    s0 = score(params); best = (s0[0], [p.detach().clone() for p in params], 0); path = [(0, *s0)]
    log(f'  least squares: validation same class as ParT {100 * s0[0]:.2f}%, accuracy {100 * s0[1]:.2f}%, {time.time() - t0:.0f} s')
    m_ = [torch.zeros_like(p) for p in params]; v_ = [torch.zeros_like(p) for p in params]
    for i in range(steps):
        gs = [torch.zeros_like(p) for p in params]
        for a in range(0, tr['n'], chunk):
            ps = [p.detach().requires_grad_(True) for p in params]; h, L = run(tr, a, a + chunk, ps)
            loss = -(tr['P'][a:a + chunk] * torch.log_softmax(L, 1)).sum() / tr['n'] + lam * ((h - tr['H'][a:a + chunk]) ** 2 / vn).sum() / (tr['n'] * 128)
            for g, gg in zip(gs, torch.autograd.grad(loss, ps)): g += gg
        with torch.no_grad():
            for p, g, m, v in zip(params, gs, m_, v_):
                m.mul_(.9).add_(.1 * g); v.mul_(.999).add_(.001 * g * g); p.sub_(lr * (m / (1 - .9 ** (i + 1))) / (torch.sqrt(v / (1 - .999 ** (i + 1))) + 1e-8))
        if i % 50 == 49 or i == steps - 1:
            s = score(params); path.append((i + 1, *s))
            if s[0] > best[0]: best = (s[0], [p.detach().clone() for p in params], i + 1)
            log(f'  step {i + 1}: validation same class as ParT {100 * s[0]:.2f}%, accuracy {100 * s[1]:.2f}%, {time.time() - t0:.0f} s')
    out = out or RESULTS / 'attn_fit' / str(net); out.mkdir(parents=True, exist_ok=True)
    Ab, bb, Wb, cb = best[1]
    np.savez(out / 'attn_fit.npz', A=(Mt @ Ab).cpu().numpy(), b_cls=bb.cpu().numpy(), W=Wb.cpu().numpy(), c=cb.cpu().numpy(), M=Mt.cpu().numpy())
    (out / 'attn_fit.json').write_text(json.dumps(dict(heads=HEADS, terms=K, n_fit=tr['n'], lam=lam, lr=lr, steps=steps, best_step=best[2], path=path, ctx=CTX, pfeat=PFEAT,
                                                       knots=[k.tolist() for k in knots])))
    log(f'attention fit: best validation same class as ParT {100 * best[0]:.2f}% (step {best[2]}), {time.time() - t0:.0f} s')
    return best


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'tune': tune('full', *(int(x) for x in a))
