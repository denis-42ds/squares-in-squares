"""Hypothesised family members n(k), L = k + exc; read-only dry run with frame+pushdown closure. Usage: python -m sqin.seeds"""
import math

from . import store
from .construct import GAP, MIN_FRAC, TOL, _score

FAMILIES = (
    ("0.8467", 267, lambda k: k * k + (2 * k + 1) // 3, lambda k: k % 3 == 1),
    ("0.8229", 299, lambda k: k * k + (3 * k) // 5, lambda k: True),
)
KCHECK = range(4, 18)
KNEW = range(18, 33)


def _close(Ls, nmax):
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


def main():
    nmax = store.MAX_N
    Ls = {n: float(store.best_L(n)) for n in range(1, nmax + 1)}
    old = dict(Ls)
    inject = []
    for name, ref, nf, ok in FAMILIES:
        Lr = float(store.best_L(ref))
        exc = Lr - math.floor(Lr)
        print(f"family {name}: ref n={ref} exc={exc:.9f}")
        for k in KCHECK:
            if ok(k) and nf(k) <= nmax:
                n = nf(k)
                print(f"  check k={k:2d} n={n:4d} best-L={float(store.best_L(n)) - (k + exc):+.2e}")
        for k in KNEW:
            if ok(k) and nf(k) <= nmax:
                n, L = nf(k), k + exc
                tag = "IMPROVES" if L < Ls[n] - TOL else "-"
                print(f"  new   k={k:2d} n={n:4d} L={L:.9f} best={Ls[n]:.9f} {tag}")
                inject.append((n, L))
    for n, L in inject:
        Ls[n] = min(Ls[n], L)
    _close(Ls, nmax)
    g = lambda lo, hi, d: sum(_score(d[n], n) for n in range(lo, hi + 1))
    print(f"total old={g(1, nmax, old):.10f} new={g(1, nmax, Ls):.10f}")
    for lo, hi in ((1, 100), (101, 400), (401, nmax)):
        print(f"  {lo}-{hi}: old={g(lo, hi, old):.6f} new={g(lo, hi, Ls):.6f}")
    imp = sorted(n for n in Ls if Ls[n] < old[n] - TOL)
    print(f"improved n: {len(imp)}; n>400: {sum(1 for n in imp if n > 400)}; first: {imp[:15]}")


if __name__ == "__main__":
    main()
