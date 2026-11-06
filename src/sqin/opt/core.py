from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from sqin.geometry import HALF_PI, extents
from sqin.opt.energy import energy_grad

TOL_E = 1e-24
SEED_BASE = 20261106
MIN_STEP = 1e-10
MAX_RELAX_CALLS = 400


def rng_for(n, run):
    return np.random.default_rng([SEED_BASE, n, run])


def to_xya(z, n):
    return np.stack([z[:n], z[n : 2 * n], z[2 * n :]], axis=1)


def from_xya(xya):
    a = np.asarray(xya, dtype=np.float64)
    return np.concatenate([a[:, 0], a[:, 1], a[:, 2]])


def relax(z0, n, L, maxiter):
    g = np.empty(3 * n)
    e0 = energy_grad(z0, n, L, g)
    if e0 < TOL_E:
        return z0.copy(), e0
    best = [e0, z0.copy()]
    hist = []

    def fun(z):
        e = energy_grad(z, n, L, g)
        if e < best[0]:
            best[0] = e
            best[1] = z.copy()
        return e, g.copy()

    def cb(_xk):
        hist.append(best[0])
        if best[0] < TOL_E:
            raise StopIteration
        k = len(hist)
        if k >= 200 and k % 100 == 0 and best[0] > 0.7 * hist[k - 101]:
            raise StopIteration

    minimize(
        fun, z0, jac=True, method="L-BFGS-B", callback=cb,
        options={"maxiter": maxiter, "maxfun": 2 * maxiter, "maxcor": 20, "ftol": 1e-30, "gtol": 1e-14},
    )
    return best[1], best[0]


def feasible_start(z, n, L, tries=14):
    for _ in range(tries):
        z, e = relax(z, n, L, 2000)
        if e < TOL_E:
            return z, L
        z = z.copy()
        z[: 2 * n] *= 1.03
        L *= 1.03
    return None


def compress(z, n, L, step=0.02):
    calls = 0
    while step >= MIN_STEP and calls < MAX_RELAX_CALLS:
        r = 1.0 - step
        zt = z.copy()
        zt[: 2 * n] *= r
        zt, e = relax(zt, n, L * r, 300 if step > 1e-4 else 4000)
        calls += 1
        if e < TOL_E:
            z, L = zt, L * r
            step = min(step * 1.5, 0.02)
        else:
            step *= 0.5
    return z, L


def _ceil_sqrt(n):
    k = math.isqrt(n)
    return k if k * k >= n else k + 1


def _neighbor_start(n, rng):
    from sqin.store import has_best, load_best

    add = False
    if has_best(n + 1):
        a = load_best(n + 1)[0]
        a = np.delete(a, int(rng.integers(len(a))), axis=0)
    elif n > 2 and has_best(n - 1):
        a = load_best(n - 1)[0]
        add = True
    else:
        return None
    a = np.array(a, dtype=np.float64, copy=True)
    e = extents(a)
    a[:, 0] -= e[:, 0].min()
    a[:, 1] -= e[:, 2].min()
    side = float(max(e[:, 1].max() - e[:, 0].min(), e[:, 3].max() - e[:, 2].min()))
    if add:
        new = [rng.uniform(0.5, side - 0.5), rng.uniform(0.5, side - 0.5), rng.uniform(0, HALF_PI)]
        a = np.vstack([a, new])
    return from_xya(a), side * (1.1 if add else 1.02)


def init_config(n, mode, rng):
    k = _ceil_sqrt(n)
    if mode in (0, 1):
        pitch = 1.02
        cells = rng.choice(k * k, size=n, replace=False)
        x = (cells % k + 0.5) * pitch + rng.normal(0, 0.05, n)
        y = (cells // k + 0.5) * pitch + rng.normal(0, 0.05, n)
        a = rng.normal(0, 0.15, n) if mode == 0 else rng.uniform(0, HALF_PI, n)
        return np.concatenate([x, y, a]), k * pitch
    L = 1.3 * k
    x = rng.uniform(0.5, L - 0.5, n)
    y = rng.uniform(0.5, L - 0.5, n)
    a = rng.uniform(0, HALF_PI, n)
    return np.concatenate([x, y, a]), L


def solve_run(n, run, neighbors=False):
    rng = rng_for(n, run)
    mode = run % (4 if neighbors else 3)
    start = None
    if mode == 3:
        start = _neighbor_start(n, rng)
        mode = 0
    z0, L0 = start if start is not None else init_config(n, mode, rng)
    fs = feasible_start(z0, n, L0)
    if fs is None:
        return None
    z, _L = compress(fs[0], n, fs[1])
    return to_xya(z, n)
