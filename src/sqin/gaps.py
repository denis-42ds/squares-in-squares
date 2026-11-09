from __future__ import annotations

import argparse
import math

from sqin.grid import grid_score, grid_side
from sqin.profile import FR_TOL, load_friedman
from sqin.store import MAX_N, has_best, load_best, root, score_n


def main(argv=None):
    p = argparse.ArgumentParser(prog="gaps")
    p.add_argument("--friedman", default=str(root() / "data" / "external" / "friedman_L_compact.csv"))
    p.add_argument("--top", type=int, default=40)
    a = p.parse_args(argv)
    fr = load_friedman(a.friedman)
    rows = []
    for n, (L, _) in sorted(fr.items()):
        if L >= float(grid_side(n)) - FR_TOL:
            continue
        ours = float(load_best(n)[1])
        rows.append((n, ours, L, (ours - L) / math.sqrt(n), load_best(n)[2]))
    short = [r for r in rows if r[1] > r[2] + 1e-9]
    beyond = [r for r in rows if r[1] < r[2] - 1e-9]
    print(f"csv nontrivial n: {len(rows)}; reached: {len(rows) - len(short)}; short: {len(short)}; remaining score gap: {math.fsum(r[3] for r in short):.6f}")
    for n, o, L, d, src in sorted(short, key=lambda r: -r[3])[: a.top]:
        print(f"  n={n:4d} ours={o:.12f} csv={L:.12f} dscore={d:.5f}  {src[:60]}")
    print(f"n where ours < csv - 1e-9: {len(beyond)} {[r[0] for r in beyond][:40]}")
    big = [n for n in range(325, MAX_N + 1) if has_best(n)]
    gain = math.fsum(grid_score(n) - score_n(n) for n in big)
    print(f"n>324 with non-grid best: {len(big)}; gain vs grid: {gain:.6f}; max n = {max(big) if big else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
