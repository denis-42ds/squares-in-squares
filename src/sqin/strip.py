"""Goebel diagonal strip: L = k + sqrt(2)/2, n(k) = k(k-1) + 3 + floor(sqrt(2)(k-2)). Usage: python -m sqin.strip [--dry]"""
import argparse
import math

import numpy as np

from . import store
from .construct import _score

S2 = math.sqrt(2.0)


def strip_n(k):
    return k * (k - 1) + 3 + math.floor(S2 * (k - 2))


def strip(k):
    L = k + S2 / 2
    m = math.floor(S2 * (k - 2)) + 1
    a = np.array([(c + 0.5, r + 0.5) for r in range(k - 1) for c in range(k - 1 - r)])
    corners = np.array([[L - 0.5, 0.5], [0.5, L - 0.5]])
    t = np.arange(m) + 0.5 - m / 2
    h = S2 / 2
    s = np.column_stack([k / 2 + h * (0.5 - t), k / 2 + h * (0.5 + t), np.full(m, math.pi / 4)])
    ax = np.vstack([a, L - a, corners])
    return np.vstack([np.column_stack([ax, np.zeros(len(ax))]), s])


def run(kmin, kmax, dry, procs):
    cands, new = [], []
    for k in range(kmin, kmax + 1):
        n, L = strip_n(k), k + S2 / 2
        if n > store.MAX_N:
            break
        arr = strip(k)
        assert arr.shape == (n, 3), (k, arr.shape)
        built = float(store.side_length(store.normalize(arr)))
        best = float(store.best_L(n))
        print(f"k={k:2d} n={n:4d} L={L:.9f} built-L={built - L:+.1e} best={best:.9f} best-L={best - L:+.1e}")
        if best > L + 1e-9:
            cands.append((n, arr, f"goebel strip(k={k})"))
            new.append((n, L))
    Ls = {n: float(store.best_L(n)) for n in range(1, store.MAX_N + 1)}
    old = dict(Ls)
    for n, L in new:
        Ls[n] = min(Ls[n], L)
    for n in range(store.MAX_N - 1, 0, -1):
        Ls[n] = min(Ls[n], Ls[n + 1])
    so = sum(_score(old[n], n) for n in old)
    sn = sum(_score(Ls[n], n) for n in Ls)
    imp = [n for n in Ls if Ls[n] < old[n] - 1e-9]
    print(f"cands={len(cands)} dry total {so:.10f} -> {sn:.10f}; improved n: {len(imp)} {imp[:20]}")
    if dry or not cands:
        return
    res = store.try_update_many(cands, polish_if_needed=True, processes=procs)
    for x in res:
        print(" ", x)
    store.push_down(verbose=False)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=40)
    p.add_argument("--procs", type=int, default=6)
    a = p.parse_args(argv)
    run(a.kmin, a.kmax, a.dry, a.procs)


if __name__ == "__main__":
    main()
