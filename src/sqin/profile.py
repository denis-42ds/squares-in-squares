from __future__ import annotations

import argparse
import csv
import math

from sqin.grid import grid_score, grid_side
from sqin.store import root

MAX_N = 1000
TOP10_SCORE = 22.1694276989
FR_TOL = 1e-9


def group(n):
    return "<=100" if n <= 100 else "101-400" if n <= 400 else "401-1000"


def load_friedman(path):
    out = {}
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[int(r["n"])] = (float(r["L"]), r.get("kind", ""))
    except FileNotFoundError:
        pass
    return out


def build_table(fr):
    rows = []
    for n in range(1, MAX_N + 1):
        m = math.isqrt(n)
        gs, gside = grid_score(n), float(grid_side(n))
        fL, kind = fr.get(n, (None, ""))
        fs = None if fL is None else max(0.0, fL / math.sqrt(n) - 1.0)
        nontriv = fL is not None and fL < gside - FR_TOL
        rows.append(
                {
                    "n": n, "m": m, "holes": (m + 1) ** 2 - n if m * m != n else 0,
                    "pos": (n - m * m) / (2 * m + 1), "grid_side": gside, "grid_score": gs,
                    "fr_L": fL, "fr_score": fs, "kind": kind, "nontriv": nontriv,
                    "gain": (gs - fs) if nontriv else 0.0,
                    }
                )
    return rows


def _sum(rows, pred, key):
    return math.fsum(r[key] for r in rows if pred(r))


def main(argv=None):
    p = argparse.ArgumentParser(prog="profile")
    p.add_argument("--friedman", default=str(root() / "data" / "external" / "friedman_L_compact.csv"))
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default=str(root() / "runs" / "profile_grid.csv"))
    a = p.parse_args(argv)

    fr = load_friedman(a.friedman)
    rows = build_table(fr)
    print(f"friedman rows loaded: {len(fr)} (n max = {max(fr) if fr else None})")

    total_grid = math.fsum(r["grid_score"] for r in rows)
    gap10 = total_grid - TOP10_SCORE
    print(f"\ngrid total = {total_grid:.6f}; gap to top10 ({TOP10_SCORE}) = {gap10:.6f}")
    for g in ("<=100", "101-400", "401-1000"):
        sel = [r for r in rows if group(r["n"]) == g]
        print(f"  grid {g:9s}: {math.fsum(r['grid_score'] for r in sel):.6f}  (n={len(sel)})")

    print("\ngrid score by position in [m^2,(m+1)^2) (mean over n):")
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sel = [r["grid_score"] for r in rows if lo <= r["pos"] < hi and r["n"] > 4]
        print(f"  pos [{lo:.2f},{min(hi, 1):.2f}): mean={sum(sel) / len(sel):.5f} n={len(sel)}")

    free = [r for r in rows if r["holes"] == 0 or r["n"] in {(r["m"] + 1) ** 2 - 1, (r["m"] + 1) ** 2 - 2}]
    free = [r for r in rows if (r["m"] + 1) ** 2 - r["n"] in (0, 1, 2) or r["holes"] == 0]
    print(f"\nn = k^2, k^2-1, k^2-2: {len(free)} puzzles, grid score sum = {math.fsum(r['grid_score'] for r in free):.6f}")
    conf = [r["n"] for r in free if r["nontriv"]]
    print(f"  conflicts with friedman (nontrivial L < grid side): {conf}")
    worse = [r["n"] for r in rows if r["fr_L"] is not None and r["fr_L"] > r["grid_side"] + FR_TOL]
    below = [r["n"] for r in rows if r["fr_L"] is not None and r["fr_L"] < math.sqrt(r["n"]) - FR_TOL]
    print(f"  friedman L worse than grid: {worse}; below sqrt(n): {below}")

    nt = [r for r in rows if r["nontriv"]]
    s_le100 = _sum(nt, lambda r: r["n"] <= 100, "gain")
    s_mid = _sum(nt, lambda r: 100 < r["n"] <= 324, "gain")
    s_all = s_le100 + s_mid
    print(f"\nnontrivial friedman n: {len(nt)} (<=100: {sum(r['n'] <= 100 for r in nt)}, 101-324: {sum(100 < r['n'] <= 324 for r in nt)})")
    print(f"sum(grid_score - friedman_score): <=100: {s_le100:.6f}  101-324: {s_mid:.6f}  all: {s_all:.6f}")
    print(f"coverage of gap to top10: {s_all / gap10:.1%}   remaining to find: {gap10 - s_all:.6f}")

    g_gt324 = _sum(rows, lambda r: r["n"] > 324, "grid_score")
    g_mid = _sum(rows, lambda r: 100 < r["n"] <= 324, "grid_score")
    print(f"grid mass n>324 (no data): {g_gt324:.6f}; needed relative cut there: {(gap10 - s_all) / g_gt324:.1%}")
    rel_mid = s_mid / g_mid if g_mid else 0.0
    proj = rel_mid * g_gt324
    print(f"ASSUMPTION: same relative cut as on 101-324 ({rel_mid:.1%}) -> n>324 gives ~{proj:.4f}; total ~{s_all + proj:.4f} vs gap {gap10:.4f}")

    print(f"\ntop-{a.top} n by friedman gain:")
    for r in sorted(nt, key=lambda r: -r["gain"])[: a.top]:
        print(f"  n={r['n']:4d} gain={r['gain']:.5f} grid={r['grid_score']:.5f} fr={r['fr_score']:.5f} L={r['fr_L']:.12f} ({r['kind']})")

    print("\nchat 03 priority (n<=150, gain / n^1.5):")
    cand = [r for r in nt if r["n"] <= 150]
    for r in sorted(cand, key=lambda r: -r["gain"] / r["n"] ** 1.5)[:30]:
        print(f"  n={r['n']:4d} gain={r['gain']:.5f} prio={r['gain'] / r['n'] ** 1.5:.6f}")

    from pathlib import Path

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = ["n", "m", "holes", "pos", "grid_side", "grid_score", "fr_L", "fr_score", "kind", "nontriv", "gain"]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])
    print(f"\nwrote {a.out}")
    return 0
