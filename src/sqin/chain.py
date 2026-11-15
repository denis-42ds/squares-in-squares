"""Chain composition of equal-excess packings with TR/BL lattice fillers: n = K^2 + sum j_i, L = K + e.
Usage: python -m sqin.chain [--dry] [--cmax 8] [--rounds 5] [--procs 6]"""
import argparse
import math

import numpy as np

from . import store
from .construct import _score
from .seeds import _close

KMAX = 31
EXC_TOL = 1e-8


def classes():
    items = []
    for n in range(1, store.MAX_N + 1):
        L = float(store.best_L(n))
        k = math.floor(L + 1e-9)
        if L - k >= 1e-6 and n > k * k:
            items.append((L - k, k, n))
    items.sort()
    groups, cur = [], []
    for it in items:
        if cur and it[0] - cur[-1][0] > EXC_TOL:
            groups.append(cur)
            cur = []
        cur.append(it)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        seed = {}
        for _, k, n in g:
            if n > seed.get(k, 0):
                seed[k] = n
        out.append((max(x[0] for x in g), seed))
    return out


def best_chains(seed, cmax):
    jj = {k: n - k * k for k, n in seed.items() if k <= KMAX}
    f = {(1, k): (j, (k,)) for k, j in jj.items()}
    res = {}
    for c in range(2, cmax + 1):
        for (_, S), (v, ks) in [x for x in f.items() if x[0][0] == c - 1]:
            for k, j in jj.items():
                if S + k < KMAX and f.get((c, S + k), (-1,))[0] < v + j:
                    f[(c, S + k)] = (v + j, ks + (k,))
        for (_, S), (v, ks) in [x for x in f.items() if x[0][0] == c]:
            for s in range(-(-(c - 1) // 2), c):
                if S + s <= KMAX and res.get(S + s, (-1,))[0] < v:
                    res[S + s] = (v, ks, c - 1 - s, 2 * s - (c - 1))
    return res


def _norm(xya):
    a = np.array(xya, dtype=np.float64)
    ex = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    a[:, 0] -= (a[:, 0] - ex).min()
    a[:, 1] -= (a[:, 1] - ex).min()
    return a


def build(ks, steps, arrs, e):
    K = sum(ks) + sum(t != "D" for t in steps)
    L = K + e
    xs, ys = [0], [0]
    for k, t in zip(ks, steps):
        xs.append(xs[-1] + k + (t != "D"))
        ys.append(ys[-1] + k + (t != "R"))
    assert xs[-1] + ks[-1] == K and ys[-1] + ks[-1] == K, (ks, steps)
    parts = [a + np.array([x, y, 0.0]) for a, x, y in zip(arrs, xs, ys)]
    tr, bl = [], []
    for i in range(len(ks) - 1):
        for a in range(K - xs[i] - ks[i]):
            for b in range(ys[i + 1] - ys[i]):
                tr.append((L - a - 0.5, ys[i] + b + 0.5))
        for a in range(xs[i + 1] - xs[i]):
            for b in range(K - ys[i] - ks[i]):
                bl.append((xs[i] + a + 0.5, L - b - 0.5))
    tr, bl = np.array(tr).reshape(-1, 2), np.array(bl).reshape(-1, 2)
    keep = np.ones(len(bl), dtype=bool)
    for t, (x, y) in enumerate(bl):
        if len(tr) and ((np.abs(tr[:, 0] - x) < 1 - 1e-9) & (np.abs(tr[:, 1] - y) < 1 - 1e-9)).any():
            keep[t] = False
    cells = np.vstack([tr, bl[keep]])
    return np.vstack(parts + [np.column_stack([cells, np.zeros(len(cells))])])


def candidates(cmax):
    old = {n: float(store.best_L(n)) for n in range(1, store.MAX_N + 1)}
    cands = []
    for e, seed in classes():
        arrs = {}
        for K, (J, ks, r, z) in best_chains(seed, cmax).items():
            n, L = K * K + J, K + e
            if J < 1 or n > store.MAX_N or L >= old[n] - 1e-9:
                continue
            for k in ks:
                if k not in arrs:
                    arrs[k] = _norm(store.load_best(seed[k])[0])
            arr = build(ks, ["R", "D"] * r + ["Z"] * z, [arrs[k] for k in ks], e)
            if len(arr) != n:
                print(f"MISMATCH K={K} ks={ks} r={r} z={z}: built {len(arr)} expected {n}")
                continue
            gain = _score(old[n], n) - _score(L, n)
            cands.append((gain, n, L, arr, f"chain ks={list(ks)} r={r} z={z} e={e:.6f}"))
    return old, cands


def estimate(old, cands):
    Ls = dict(old)
    for _, n, L, _, _ in cands:
        Ls[n] = min(Ls[n], L)
    _close(Ls, store.MAX_N)
    so = sum(_score(old[n], n) for n in old)
    sn = sum(_score(Ls[n], n) for n in Ls)
    imp = [n for n in Ls if Ls[n] < old[n] - 1e-9]
    print(f"cands={len(cands)} dry total {so:.10f} -> {sn:.10f}; improved n: {len(imp)}, n>400: {sum(n > 400 for n in imp)}")
    for gain, n, L, _, src in sorted(cands, key=lambda c: -c[0])[:20]:
        print(f"  n={n:4d} L={L:.9f} gain={gain:.6f} {src}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--cmax", type=int, default=8)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--procs", type=int, default=6)
    a = p.parse_args(argv)
    for r in range(1, a.rounds + 1):
        old, cands = candidates(a.cmax)
        if a.dry:
            estimate(old, cands)
            return
        if not cands:
            break
        res = store.try_update_many([(n, arr, src) for _, n, _, arr, src in cands], polish_if_needed=True, processes=a.procs)
        stat = {}
        for x in res:
            stat[x[1]] = stat.get(x[1], 0) + 1
            if x[1].startswith("rejected"):
                print("  REJECT", x)
        store.push_down(verbose=False)
        print(f"round {r}: cands={len(cands)} {stat} total={store.total_score()[0]:.10f}")
        if not stat.get("updated"):
            break


if __name__ == "__main__":
    main()
