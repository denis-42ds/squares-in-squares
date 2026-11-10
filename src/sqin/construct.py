"""Border growth: (n, L) -> (n + 2*floor(L) + 1, L + 1). Usage: python -m sqin.construct grow [--dry]"""
import argparse
import math
import time

import numpy as np

from . import store

GAP = 1e-10
MIN_FRAC = 0.05
TOL = 1e-9


def _total():
    t = store.total_score()
    return float(t[0] if isinstance(t, tuple) else t)


def grow(xya):
    a = np.asarray(xya, dtype=np.float64)
    ex = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    x0, y0 = (a[:, 0] - ex).min(), (a[:, 1] - ex).min()
    L = max((a[:, 0] + ex).max() - x0, (a[:, 1] + ex).max() - y0)
    k = math.floor(L)
    if L - k < MIN_FRAC:
        return None
    i = np.arange(k + 1) * (1.0 + GAP)
    j = np.arange(k) * (1.0 + GAP)
    row = np.column_stack([x0 + 0.5 + i, np.full(k + 1, y0 + L + GAP + 0.5), np.zeros(k + 1)])
    col = np.column_stack([np.full(k, x0 + L + GAP + 0.5), y0 + 0.5 + j, np.zeros(k)])
    return np.vstack([a, row, col])


def _score(L, n):
    return max(0.0, L / math.sqrt(n) - 1.0)


def dry(nmax):
    Ls = {n: float(store.best_L(n)) for n in range(1, nmax + 1)}
    old = dict(Ls)
    changed = True
    while changed:
        changed = False
        for n in range(nmax - 1, 0, -1):
            if Ls[n + 1] < Ls[n] - TOL:
                Ls[n] = Ls[n + 1]
                changed = True
        for n in range(1, nmax + 1):
            k = math.floor(Ls[n])
            m = n + 2 * k + 1
            if m <= nmax and Ls[n] - k >= MIN_FRAC and Ls[n] + 1 + 2 * GAP < Ls[m] - TOL:
                Ls[m] = Ls[n] + 1 + 2 * GAP
                changed = True
    g = lambda lo, hi, d: sum(_score(d[n], n) for n in range(lo, min(hi, nmax) + 1))
    print(f"total old={g(1, nmax, old):.10f} new={g(1, nmax, Ls):.10f}")
    for lo, hi in ((1, 100), (101, 400), (401, nmax)):
        print(f"  {lo}-{hi}: old={g(lo, hi, old):.6f} new={g(lo, hi, Ls):.6f}")
    rows = sorted(((_score(old[n], n) - _score(Ls[n], n), n) for n in old if Ls[n] < old[n] - TOL), reverse=True)
    print(f"improved n: {len(rows)}; n>400: {sum(1 for _, n in rows if n > 400)}")
    for gain, n in rows[:20]:
        print(f"  n={n:4d} L {old[n]:.9f} -> {Ls[n]:.9f} gain={gain:.6f}")


def run(nmax, rounds, procs):
    t0 = _total()
    for r in range(1, rounds + 1):
        cands = []
        for n in range(1, nmax + 1):
            L = float(store.best_L(n))
            k = math.floor(L)
            m = n + 2 * k + 1
            if m > nmax or L - k < MIN_FRAC or L + 1 >= float(store.best_L(m)) - TOL:
                continue
            xya, _, src = store.load_best(n)
            arr = grow(xya)
            if arr is not None:
                cands.append((m, arr, f"grow(n={n}) <- {src}"))
        t = time.time()
        res = store.try_update_many(cands, polish_if_needed=True, processes=procs)
        upd = [x for x in res if x[1] == "updated"]
        stat = {}
        for x in res:
            stat[x[1]] = stat.get(x[1], 0) + 1
        print(f"round {r}: cands={len(cands)} {stat} {time.time() - t:.1f}s")
        for x in res:
            if x[1] in ("rejected_fast", "rejected_gate"):
                print("  REJECT", x)
        store.push_down(verbose=False)
        print(f"  total={_total():.10f}")
        if not upd:
            break
    print(f"total {t0:.10f} -> {_total():.10f}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["grow"])
    p.add_argument("--dry", action="store_true")
    p.add_argument("--nmax", type=int, default=store.MAX_N)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--procs", type=int, default=6)
    a = p.parse_args(argv)
    dry(a.nmax) if a.dry else run(a.nmax, a.rounds, a.procs)


if __name__ == "__main__":
    main()
