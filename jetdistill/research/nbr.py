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
