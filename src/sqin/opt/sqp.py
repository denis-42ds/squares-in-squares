from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import minimize

from sqin.geometry import min_sat_gap, normalize, side_length
from sqin.opt.energy import _sgn

GAP_TARGET = 3e-12
PAIR_CUT = 0.2


def _pairs(z, n):
    x, y, a = z[:n], z[n : 2 * n], z[2 * n :]
    c, s = np.cos(a), np.sin(a)
    w = 0.5 * (np.abs(c) + np.abs(s))
    I, J = np.triu_indices(n, 1)
    dx, dy = x[J] - x[I], y[J] - y[I]
    r = w[I] + w[J] + PAIR_CUT
    m = (np.abs(dx) < r) & (np.abs(dy) < r)
    I, J, dx, dy = I[m], J[m], dx[m], dy[m]
    cf = c[I] * c[J] + s[I] * s[J]
    sf = s[J] * c[I] - c[J] * s[I]
    S = 0.5 + 0.5 * (np.abs(cf) + np.abs(sf))
    p = np.stack(
        [
            c[I] * dx + s[I] * dy,
            -s[I] * dx + c[I] * dy,
            c[J] * dx + s[J] * dy,
            -s[J] * dx + c[J] * dy,
        ],
        axis=1,
    )
    G = np.abs(p) - S[:, None]
    K = G.argmax(1)
    keep = G.max(1) < PAIR_CUT
    return I[keep].astype(np.int64), J[keep].astype(np.int64), K[keep].astype(np.int64)


@njit(cache=True)
def _cons(v, n, I, J, K, gap, c, Jm):
    L = v[3 * n]
    Jm[:, :] = 0.0
    for i in range(n):
        a = v[2 * n + i]
        cc = np.cos(a)
        ss = np.sin(a)
        w = 0.5 * (abs(cc) + abs(ss))
        dw = 0.5 * (-_sgn(cc) * ss + _sgn(ss) * cc)
        r = 4 * i
        c[r] = v[i] - w
        Jm[r, i] = 1.0
        Jm[r, 2 * n + i] = -dw
        c[r + 1] = L - v[i] - w
        Jm[r + 1, 3 * n] = 1.0
        Jm[r + 1, i] = -1.0
        Jm[r + 1, 2 * n + i] = -dw
        c[r + 2] = v[n + i] - w
        Jm[r + 2, n + i] = 1.0
        Jm[r + 2, 2 * n + i] = -dw
        c[r + 3] = L - v[n + i] - w
        Jm[r + 3, 3 * n] = 1.0
        Jm[r + 3, n + i] = -1.0
        Jm[r + 3, 2 * n + i] = -dw
    base = 4 * n
    for q in range(len(I)):
        i = I[q]
        j = J[q]
        k = K[q]
        ai = v[2 * n + i]
        aj = v[2 * n + j]
        ci = np.cos(ai)
        si = np.sin(ai)
        cj = np.cos(aj)
        sj = np.sin(aj)
        dx = v[j] - v[i]
        dy = v[n + j] - v[n + i]
        cf = ci * cj + si * sj
        sf = sj * ci - cj * si
        S = 0.5 + 0.5 * (abs(cf) + abs(sf))
        dH = 0.5 * (-_sgn(cf) * sf + _sgn(sf) * cf)
        p0 = ci * dx + si * dy
        p1 = -si * dx + ci * dy
        p2 = cj * dx + sj * dy
        p3 = -sj * dx + cj * dy
        if k == 0:
            p, ux, uy, dpi, dpj = p0, ci, si, p1, 0.0
        elif k == 1:
            p, ux, uy, dpi, dpj = p1, -si, ci, -p0, 0.0
        elif k == 2:
            p, ux, uy, dpi, dpj = p2, cj, sj, 0.0, p3
        else:
            p, ux, uy, dpi, dpj = p3, -sj, cj, 0.0, -p2
        sg = _sgn(p)
        if sg == 0.0:
            sg = 1.0
        r = base + q
        c[r] = abs(p) - S - gap
        Jm[r, i] = -sg * ux
        Jm[r, j] = sg * ux
        Jm[r, n + i] = -sg * uy
        Jm[r, n + j] = sg * uy
        Jm[r, 2 * n + i] = dH + sg * dpi
        Jm[r, 2 * n + j] = -dH + sg * dpj


def _xya(z, n):
    return np.stack([z[:n], z[n : 2 * n], z[2 * n :]], axis=1)


def _constraint(n, I, J, K):
    m = 4 * n + len(I)
    cbuf = np.empty(m)
    Jbuf = np.zeros((m, 3 * n + 1))

    def fun(x):
        _cons(x, n, I, J, K, GAP_TARGET, cbuf, Jbuf)
        return cbuf.copy()

    def jac(x):
        _cons(x, n, I, J, K, GAP_TARGET, cbuf, Jbuf)
        return Jbuf.copy()

    return fun, jac


def sqp_polish(z, n, L, rounds=3, maxiter=300):
    best = np.array(z, dtype=np.float64, copy=True)
    best_side = side_length(_xya(best, n))
    v = np.append(best, L)
    obj_jac = np.zeros(3 * n + 1)
    obj_jac[-1] = 1.0
    for _ in range(rounds):
        I, J, K = _pairs(v[: 3 * n], n)
        fun, jac = _constraint(n, I, J, K)

        try:
            res = minimize(
                lambda x: x[-1], v, jac=lambda x: obj_jac, method="SLSQP",
                constraints=[{"type": "ineq", "fun": fun, "jac": jac}],
                options={"maxiter": maxiter, "ftol": 1e-15},
            )
        except Exception:  # noqa: BLE001
            break
        zn = res.x[: 3 * n]
        if not np.all(np.isfinite(zn)):
            break
        xn = _xya(zn, n)
        side = side_length(xn)
        if min_sat_gap(normalize(xn)) >= -1e-9 and side < best_side - 1e-13:
            best, best_side, v = zn.copy(), side, res.x.copy()
        else:
            break
    return best
