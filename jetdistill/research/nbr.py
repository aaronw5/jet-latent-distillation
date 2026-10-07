"""Neighbourhood features of each particle (physics context that ParT's particle attention gives each embedding):
local density, nearest neighbours, distances to the hardest / displaced / lepton particles, and its prong (the particle
assigned to the nearest of the 3 hardest particles' directions, a cheap stand-in for exclusive subjets) with the
prong's pT share, mass, charge and displaced count.

nbr_features(x, ext, jet) -> (J, 128, len(NBR)) float32, 0 for empty slots"""
import numpy as np

NBR = ['ln(1+n within 0.1)', 'ln(1+n within 0.2)', 'pT share within 0.1', 'pT share within 0.2', 'pT share within 0.4',
       'ΔR nearest', 'ln pT nearest / pT', 'ΔR hardest', 'ΔR 2nd hardest', 'ΔR nearest displaced', 'ΔR nearest lepton',
       'ΔR nearest photon', 'prong index', 'prong pT share', 'prong ln mass', 'prong charge', 'prong n displaced',
       'pT share in prong', 'ΔR prong axis', 'ln kT with nearest']


def nbr_features(x, ext, jet, chunk=2000):
    out = np.zeros(x.shape[:2] + (len(NBR),), np.float32)
    for a in range(0, len(x), chunk):
        out[a:a + chunk] = _chunk(np.asarray(x[a:a + chunk], np.float32), np.asarray(ext[a:a + chunk], np.float32), np.asarray(jet[a:a + chunk], np.float32))
    return out


def _chunk(x, e, jet):
    pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]; E, q, typ, d0, d0e = e[..., 0], e[..., 1], e[..., 2], e[..., 3], e[..., 4]
    ok = pt > 0; n, P = pt.shape; big = 9.0
    dR = np.sqrt((eta[:, :, None] - eta[:, None]) ** 2 + (phi[:, :, None] - phi[:, None]) ** 2)
    pair = ok[:, :, None] & ok[:, None]; eye = np.eye(P, dtype=bool)[None]
    dRo = np.where(pair & ~eye, dR, big)                                     # to the other real particles
    ptj = np.maximum(pt.sum(1, keepdims=True), 1e-6)
    f = {}
    for r in (0.1, 0.2): f[f'n{r}'] = np.log1p((dRo < r).sum(2))
    for r in (0.1, 0.2, 0.4): f[f's{r}'] = ((dRo < r) * pt[:, None]).sum(2) / ptj
    j = dRo.argmin(2); dn = np.take_along_axis(dRo, j[..., None], 2)[..., 0]; ptn = np.take_along_axis(pt, j, 1)
    f['dn'] = np.minimum(dn, 1.5); f['ptn'] = np.log(np.maximum(ptn, 1e-6) / np.maximum(pt, 1e-6)).clip(-10, 10)
    order = np.argsort(-pt, 1); h1, h2, h3 = order[:, 0], order[:, 1], order[:, 2]
    tk = lambda idx: np.take_along_axis(dR, idx[:, None, None].repeat(P, 1), 2)[..., 0]
    f['dh1'] = np.minimum(tk(h1), 1.5); f['dh2'] = np.minimum(tk(h2), 1.5)
    disp = ok & (np.abs(d0) / np.maximum(d0e, 1e-6) > 3) & (q != 0); lep = ok & ((typ == 4) | (typ == 5)); pho = ok & (typ == 3)
    for k, m in (('dd', disp), ('dl', lep), ('dp', pho)): f[k] = np.minimum(np.where(m[:, None] & ~eye, dR, big).min(2), 1.5)
    seeds = np.stack([h1, h2, h3], 1)                                        # prongs: nearest of the 3 hardest directions
    ds = np.stack([tk(seeds[:, s]) for s in range(3)], -1); pr = ds.argmin(-1); f['pi'] = pr.astype(np.float32)
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    pp = {k: np.zeros((n, 3), np.float32) for k in ('pt', 'E', 'px', 'py', 'pz', 'q', 'nd')}
    for s in range(3):
        m = ok & (pr == s)
        for k, v in (('pt', pt), ('E', E), ('px', px), ('py', py), ('pz', pz), ('q', q), ('nd', disp.astype(np.float32))): pp[k][:, s] = (v * m).sum(1)
    m2 = np.maximum(pp['E'] ** 2 - pp['px'] ** 2 - pp['py'] ** 2 - pp['pz'] ** 2, 0)
    g = lambda arr: np.take_along_axis(arr, pr, 1)
    f['ppt'] = g(pp['pt']) / ptj; f['pm'] = np.log1p(np.sqrt(g(m2))); f['pq'] = g(pp['q']); f['pnd'] = g(pp['nd'])
    f['sin'] = pt / np.maximum(g(pp['pt']), 1e-6); f['dax'] = np.minimum(np.take_along_axis(ds, pr[..., None], 2)[..., 0], 1.5)
    f['kt'] = np.log(np.maximum(np.minimum(pt, ptn) * dn, 1e-6)).clip(-15, 10)
    keys = ['n0.1', 'n0.2', 's0.1', 's0.2', 's0.4', 'dn', 'ptn', 'dh1', 'dh2', 'dd', 'dl', 'dp', 'pi', 'ppt', 'pm', 'pq', 'pnd', 'sin', 'dax', 'kt']
    return (np.stack([f[k] for k in keys], -1) * ok[..., None]).astype(np.float32)


# ---------------------------------------------------------------- E5: context from ParT's own pair kernels
PK_PROPS = ['ln pT/pT_jet', 'charge', 'charged hadron', 'neutral hadron', 'photon', 'electron', 'muon', 'tanh d0', 'd0 significance>3', 'ΔR to axis', 'ln ΔR to i']
PK = [f'h{h + 1}: {p}' for h in range(8) for p in PK_PROPS]


def pk_features(x, ext, jet, model, device='mps', chunk=250):
    """(J, 128, 8·11): for each particle i and head h, the average of the other particles' properties weighted by
    softmax_j U_h(i, j), ParT's pair bias (its pair_embed on ln kT, ln z, ln ΔR, ln m²; massless 4-vectors in jet
    coordinates) — the block-1 attention as far as the physics kernel sets it"""
    import torch
    out = np.zeros(x.shape[:2] + (len(PK),), np.float32); emb = model.pair_embed.embed
    for a in range(0, len(x), chunk):
        xx = np.asarray(x[a:a + chunk], np.float32); ee = np.asarray(ext[a:a + chunk], np.float32); jj = np.asarray(jet[a:a + chunk], np.float32)
        ok = xx[..., 0] > 0; P = int(ok.sum(1).max()); n = len(xx)
        T = lambda q: torch.from_numpy(np.ascontiguousarray(q)).to(device)
        pt, eta, phi = T(xx[:, :P, 0]), T(xx[:, :P, 1]), T(xx[:, :P, 2]); m = T(ok[:, :P])
        with torch.no_grad():
            deta = eta[:, :, None] - eta[:, None]; dphi = phi[:, :, None] - phi[:, None]; dR = torch.sqrt(deta ** 2 + dphi ** 2).clamp(min=1e-8)
            ptmin = torch.minimum(pt[:, :, None], pt[:, None]).clamp(min=1e-8)
            lnkt = torch.log((ptmin * dR).clamp(min=1e-8)); lnz = torch.log((ptmin / (pt[:, :, None] + pt[:, None]).clamp(min=1e-8)).clamp(min=1e-8))
            m2 = 2 * pt[:, :, None] * pt[:, None] * (torch.cosh(deta) - torch.cos(dphi)); lnm2 = torch.log(m2.clamp(min=1e-8))
            F = torch.stack([lnkt, lnz, torch.log(dR), lnm2], 1).reshape(n, 4, P * P)
            U = emb(F).reshape(n, 8, P, P)                                    # (n, heads, i, j)
            pair = m[:, None, :, None] & m[:, None, None, :] & ~torch.eye(P, dtype=torch.bool, device=device)[None, None]
            w = torch.softmax(U.masked_fill(~pair, -1e9), -1) * m[:, None, None, :]     # rows of padded i: ignored later
            E_, q, typ, d0, d0e = (T(ee[:, :P, k]) for k in range(5))
            props = torch.stack([torch.log((pt / T(jj[:, :1])).clamp(min=1e-8)), q] + [(typ == t).float() for t in (1, 2, 3, 4, 5)]
                                + [torch.tanh(d0), ((d0.abs() / d0e.clamp(min=1e-6)) > 3).float() * (q != 0).float(), torch.sqrt(eta ** 2 + phi ** 2)], -1)   # (n, P, 10)
            ctx = torch.einsum('nhij,njp->nihp', w, props)                     # (n, i, 8, 10)
            ldr = torch.einsum('nhij,nij->nih', w, torch.log(dR))[..., None]  # how far the head looks
            o = torch.cat([ctx, ldr], -1).reshape(n, P, -1) * m[..., None]
        out[a:a + chunk, :P] = o.cpu().numpy()
    return out
