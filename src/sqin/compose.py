"""Dry run: chain composition of equal-excess packings. n = K^2 + sum j_i, L = K + e, K = sum k_i + s,
ceil((c-1)/2) <= s <= c-1; frame keeps j. Usage: python -m sqin.compose"""
import math

from . import store
from .construct import _score
from .seeds import _close

KMAX = 31
EXC_TOL = 1e-8


def groups():
    items = []
    for n in range(1, store.MAX_N + 1):
        L = float(store.best_L(n))
        k = math.floor(L + 1e-9)
        e = L - k
        if e >= 1e-6 and n > k * k:
            items.append((e, k, n - k * k))
    items.sort()
    out, cur = [], []
    for it in items:
        if cur and it[0] - cur[-1][0] > EXC_TOL:
            out.append(cur)
            cur = []
        cur.append(it)
    if cur:
        out.append(cur)
    return out


def compose(seed):
    J = [0] * (KMAX + 1)
    rec = [""] * (KMAX + 1)
    for k, j in seed.items():
        if k <= KMAX:
            J[k], rec[k] = j, "seed"
    changed = True
    while changed:
        changed = False
        for K in range(1, KMAX):
            if J[K + 1] < J[K]:
                J[K + 1], rec[K + 1], changed = J[K], f"frame({K})", True
        f = {(1, k): (J[k], (k,)) for k in range(1, KMAX + 1) if J[k] >= 1}
        for c in range(2, KMAX + 1):
            for (_, S), (v, ks) in [x for x in f.items() if x[0][0] == c - 1]:
                for k in range(1, KMAX + 1 - S):
                    if J[k] >= 1 and f.get((c, S + k), (-1,))[0] < v + J[k]:
                        f[(c, S + k)] = (v + J[k], ks + (k,))
            for (_, S), (v, ks) in [x for x in f.items() if x[0][0] == c]:
                for s in range(-(-(c - 1) // 2), c):
                    K = S + s
                    if K <= KMAX and J[K] < v:
                        J[K], rec[K], changed = v, f"chain{list(ks)} s={s}", True
    return J, rec


def main():
    old = {n: float(store.best_L(n)) for n in range(1, store.MAX_N + 1)}
    Ls = dict(old)
    gs = groups()
    cands = []
    for g in gs:
        e = min(x[0] for x in g)
        seed = {}
        for _, k, j in g:
            seed[k] = max(seed.get(k, 0), j)
        J, rec = compose(seed)
        for K in range(1, KMAX + 1):
            n = K * K + J[K]
            if J[K] >= 1 and n <= store.MAX_N and K + e < old[n] - 1e-9:
                cands.append((_score(old[n], n) - _score(K + e, n), n, K + e, J[K], rec[K], e, len(g)))
    for _, n, L, *_ in cands:
        Ls[n] = min(Ls[n], L)
    _close(Ls, store.MAX_N)
    tot = lambda d, lo, hi: sum(_score(d[n], n) for n in range(lo, hi + 1))
    print(f"classes: {len(gs)}; direct cands: {len(cands)}")
    print(f"total old={tot(old, 1, store.MAX_N):.10f} new={tot(Ls, 1, store.MAX_N):.10f}")
    for lo, hi in ((1, 100), (101, 400), (401, store.MAX_N)):
        print(f"  {lo}-{hi}: old={tot(old, lo, hi):.6f} new={tot(Ls, lo, hi):.6f}")
    imp = [n for n in Ls if Ls[n] < old[n] - 1e-9]
    print(f"improved n: {len(imp)}; n>400: {sum(1 for n in imp if n > 400)}")
    for gain, n, L, j, rc, e, ns in sorted(cands, reverse=True)[:30]:
        print(f"  n={n:4d} L={L:.9f} e={e:.6f} j={j:2d} gain={gain:.6f} seeds={ns} {rc}")


if __name__ == "__main__":
    main()
