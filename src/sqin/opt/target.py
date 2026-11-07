from __future__ import annotations

import sys
import time

import numpy as np

from sqin.geometry import HALF_PI
from sqin.opt.core import TOL_E, compress, relax, rng_for, to_xya
from sqin.opt.sqp import sqp_polish


def _rand_z(n, L, rng):
    x = rng.uniform(0.5, L - 0.5, n)
    y = rng.uniform(0.5, L - 0.5, n)
    a = rng.uniform(0.0, HALF_PI, n)
    return np.concatenate([x, y, a])


def search_at_L(n, L, rng, deadline, kicks=30):
    st = {"restarts": 0, "kicks": 0, "best_e": np.inf}
    while time.time() < deadline:
        st["restarts"] += 1
        z, e = relax(_rand_z(n, L, rng), n, L, 1500)
        st["best_e"] = min(st["best_e"], e)
        if e < TOL_E:
            return z, st
        for _ in range(kicks):
            if time.time() >= deadline:
                return None, st
            st["kicks"] += 1
            zt = z.copy()
            m = 1 + int(rng.integers(2))
            idx = rng.choice(n, m, replace=False)
            zt[idx] = rng.uniform(0.5, L - 0.5, m)
            zt[n + idx] = rng.uniform(0.5, L - 0.5, m)
            zt[2 * n + idx] = rng.uniform(0.0, HALF_PI, m)
            zt, et = relax(zt, n, L, 800)
            st["best_e"] = min(st["best_e"], et)
            if et < TOL_E:
                return zt, st
            if et < e:
                z, e = zt, et
    return None, st


def solve_target(n, run, L_target, run_secs):
    rng = rng_for(n, 1000 + run)
    z, st = search_at_L(n, L_target, rng, time.time() + run_secs)
    print(
        f"[target] n={n} run={run} found={z is not None} restarts={st['restarts']} "
        f"kicks={st['kicks']} best_E={st['best_e']:.3e}",
        file=sys.stderr, flush=True,
    )
    if z is None:
        return None
    z, L = compress(z, n, L_target)
    z = sqp_polish(z, n, L)
    return to_xya(z, n)
