from __future__ import annotations

import argparse
import math
from collections import defaultdict

from sqin.grid import grid_side
from sqin.profile import FR_TOL, load_friedman
from sqin.store import root


def main(argv=None):
    p = argparse.ArgumentParser(prog="families")
    p.add_argument("--friedman", default=str(root() / "data" / "external" / "friedman_L_compact.csv"))
    p.add_argument("--min-size", type=int, default=2)
    p.add_argument("--k", type=int, default=0)
    a = p.parse_args(argv)
    fr = load_friedman(a.friedman)
    if a.k:
        k = a.k
        print(f"staircase for k={k}: n, j=n-k^2, excess=L-k (1.0 = grid side k+1)")
        for n in range(k * k + 1, (k + 1) ** 2 + 1):
            L = fr[n][0] if n in fr else float(grid_side(n))
            print(f"  n={n} j={n - k * k} excess={L - k:.8f}")
        return 0
    groups = defaultdict(list)
    for n, (L, _) in sorted(fr.items()):
        if L < float(grid_side(n)) - FR_TOL:
            groups[round(L - math.floor(L), 8)].append((n, math.floor(L)))
    print("excess = L - floor(L); members as n(k=floor L)")
    for ex, mem in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(mem) >= a.min_size:
            print(f"  excess={ex:.8f}  count={len(mem)}  " + " ".join(f"{n}({k})" for n, k in mem))
    single = sum(1 for m in groups.values() if len(m) < a.min_size)
    print(f"singletons (unique excess): {single}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
