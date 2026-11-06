from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _sgn(v):
    if v > 0.0:
        return 1.0
    if v < 0.0:
        return -1.0
    return 0.0


@njit(cache=True)
def energy_grad(z, n, L, g):
    for i in range(3 * n):
        g[i] = 0.0
    c = np.empty(n)
    s = np.empty(n)
    w = np.empty(n)
    dw = np.empty(n)
    E = 0.0
    for i in range(n):
        a = z[2 * n + i]
        c[i] = np.cos(a)
        s[i] = np.sin(a)
        w[i] = 0.5 * (abs(c[i]) + abs(s[i]))
        dw[i] = 0.5 * (-_sgn(c[i]) * s[i] + _sgn(s[i]) * c[i])
    for i in range(n):
        for off in range(2):
            v = z[off * n + i]
            lo = w[i] - v
            if lo > 0.0:
                E += lo * lo
                g[off * n + i] -= 2.0 * lo
                g[2 * n + i] += 2.0 * lo * dw[i]
            hi = v + w[i] - L
            if hi > 0.0:
                E += hi * hi
                g[off * n + i] += 2.0 * hi
                g[2 * n + i] += 2.0 * hi * dw[i]
    for i in range(n):
        for j in range(i + 1, n):
            dx = z[j] - z[i]
            dy = z[n + j] - z[n + i]
            r = w[i] + w[j]
            if abs(dx) >= r or abs(dy) >= r:
                continue
            cf = c[i] * c[j] + s[i] * s[j]
            sf = s[j] * c[i] - c[j] * s[i]
            S = 0.5 + 0.5 * (abs(cf) + abs(sf))
            dH = 0.5 * (-_sgn(cf) * sf + _sgn(sf) * cf)
            p0 = c[i] * dx + s[i] * dy
            p1 = -s[i] * dx + c[i] * dy
            p2 = c[j] * dx + s[j] * dy
            p3 = -s[j] * dx + c[j] * dy
            om = S - abs(p0)
            k = 0
            o = S - abs(p1)
            if o < om:
                om = o
                k = 1
            o = S - abs(p2)
            if o < om:
                om = o
                k = 2
            o = S - abs(p3)
            if o < om:
                om = o
                k = 3
            if om <= 0.0:
                continue
            if k == 0:
                p, ux, uy, dpi, dpj = p0, c[i], s[i], p1, 0.0
            elif k == 1:
                p, ux, uy, dpi, dpj = p1, -s[i], c[i], -p0, 0.0
            elif k == 2:
                p, ux, uy, dpi, dpj = p2, c[j], s[j], 0.0, p3
            else:
                p, ux, uy, dpi, dpj = p3, -s[j], c[j], 0.0, -p2
            sg = _sgn(p)
            f = 2.0 * om
            E += om * om
            gx = -sg * ux * f
            gy = -sg * uy * f
            g[i] -= gx
            g[j] += gx
            g[n + i] -= gy
            g[n + j] += gy
            g[2 * n + i] += f * (-dH - sg * dpi)
            g[2 * n + j] += f * (dH - sg * dpj)
    return E
