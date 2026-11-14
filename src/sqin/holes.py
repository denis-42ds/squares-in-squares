"""Read-only diagnostic: relative cut vs grid by position in [m^2,(m+1)^2) and n group. Usage: python -m sqin.holes"""
import math

from . import store

BANDS = 8
GROUPS = ((101, 324), (325, 600), (601, 1000))


def _grid_side(n):
    k = math.isqrt(n)
    return k if k * k == n else k + 1


def _sc(L, n):
    return max(0.0, L / math.sqrt(n) - 1.0)


def main():
    acc = {}
    for n in range(1, store.MAX_N + 1):
        m = math.isqrt(n)
        b = min(BANDS - 1, int(BANDS * (n - m * m) / (2 * m + 1)))
        g = _sc(_grid_side(n), n)
        o = _sc(float(store.best_L(n)), n)
        for gi, (lo, hi) in enumerate(GROUPS):
            if lo <= n <= hi:
                a = acc.setdefault((gi, b), [0.0, 0.0, 0, 0])
                a[0] += g
                a[1] += o
                a[2] += 1
                a[3] += o >= g - 1e-12
    def cut(gi, b):
        a = acc.get((gi, b))
        return 1.0 - a[1] / a[0] if a and a[0] > 0 else 0.0
    print("band | cut 101-324 | cut 325-600 (n, gridonly, hole) | cut 601-1000 (n, gridonly, hole)")
    tot = [0.0, 0.0]
    for b in range(BANDS):
        parts = [f"{b:4d} | {cut(0, b):10.3f}"]
        for gi in (1, 2):
            a = acc.get((gi, b), [0.0, 0.0, 0, 0])
            hole = a[0] * max(0.0, cut(0, b) - cut(gi, b))
            tot[gi - 1] += hole
            parts.append(f"{cut(gi, b):8.3f} ({a[2]:3d}, {a[3]:3d}, {hole:.4f})")
        print(" | ".join(parts))
    print(f"hole to table-level cut: 325-600 {tot[0]:.4f}, 601-1000 {tot[1]:.4f}, sum {sum(tot):.4f}")
    print(f"total={store.total_score()[0]:.10f}")


if __name__ == "__main__":
    main()
