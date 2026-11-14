"""Read-only survey of rotated squares in best[] by position band. Usage: python -m sqin.survey [--nmin 101 --nmax 324 --bands 2 3]"""
import argparse
import math

import numpy as np

from . import store


def band(n, bands=8):
    m = math.isqrt(n)
    return min(bands - 1, int(bands * (n - m * m) / (2 * m + 1)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nmin", type=int, default=101)
    p.add_argument("--nmax", type=int, default=324)
    p.add_argument("--bands", type=int, nargs="+", default=[2, 3])
    a = p.parse_args(argv)
    kinds = {}
    for n in range(a.nmin, a.nmax + 1):
        if band(n) not in a.bands:
            continue
        xya, L, src = store.load_best(n)
        L = float(L)
        k = math.isqrt(n)
        ang = np.asarray(xya)[:, 2] % (math.pi / 2)
        rot = (ang > 1e-9) & (ang < math.pi / 2 - 1e-9)
        if not rot.any():
            continue
        u, c = np.unique(np.round(np.degrees(ang[rot]), 3), return_counts=True)
        top = " ".join(f"{x:g}x{y}" for x, y in sorted(zip(u, c), key=lambda t: -t[1])[:3])
        near45 = np.abs(u - 45.0) < 1e-3
        kind = "45only" if near45.all() else ("45+" if near45.any() else "other")
        kinds[kind] = kinds.get(kind, 0) + 1
        print(f"n={n:3d} k={k:2d} j={n - k * k:3d} L={L:.6f} exc={L - math.floor(L):.6f} rot={int(rot.sum()):3d} ang[{len(u)}]={top} {kind} | {str(src)[:48]}")
    print(kinds)


if __name__ == "__main__":
    main()
