"""Family exc=0.846667 (DeVincentis s(54) extended): k=7+3m, n=k^2+(2k+1)/3. Usage: python -m sqin.fam8467 [--dry]"""
import argparse
import math

import numpy as np

from . import store
from .construct import _score
from .seeds import _close
from .validate import polish

S1 = 7.84666719284348978294331459096
A = math.radians(55.2650997396489041743253949403)
Y2 = 6.67333359642174489147165729548
X3 = 0.776886987015018653672079476532
Y3 = 5.32179712645279131236772171871
X4 = 1.52688698701501865367207947653
Y4 = 5.99513072287453620383937901419
X5 = 1.55377397403003730734415895306
Y5 = 4.64359425290558262473544343742
R1 = 0.077284923780532858179616104875
CA, SA = math.cos(A), math.sin(A)
ANG = (-A) % (math.pi / 2)


def _rot(tx, ty, lx, ly):
    return tx + lx * CA + ly * SA, ty - lx * SA + ly * CA


def _pts(lst):
    return np.array(lst, dtype=float).reshape(-1, 2)


def _refl(p, tx, ty):
    return np.column_stack([tx + S1 - p[:, 0], ty + S1 - p[:, 1]])


def _addon():
    p = [_rot(0.0, 6.0, 0.5, 0.5), _rot(0.75, Y2, 0.5, 0.5), _rot(X3, Y3, 0.5, 0.5), _rot(X4, Y4, 0.5, 0.5)]
    for lx, ly in ((0.5, 0.5), (1.5, 0.5), (0.5 - R1, 1.5), (1.5 - R1, 1.5)):
        p.append(_rot(X5, Y5, lx, ly))
    return _pts(p)


def _stairs():
    return _pts([(x + 0.5, y + 0.5) for x, h in enumerate((5, 4, 3, 3, 2, 1)) for y in range(h)])


def member_n(k):
    return k * k + (2 * k + 1) // 3


def member(k):
    assert k >= 7 and (k - 7) % 3 == 0
    j = (k - 10) // 3
    d = k - 7
    s = S1 + d
    ad, st = _addon(), _stairs()
    rot = [ad + np.array([0.0, d])] + [_refl(ad, 3.0 * t, d - 3.0 * t) for t in range(j + 2)]
    top = _pts([(x + 0.5, r + 0.5) for r in range(d) for x in range(k - 1 - r)])
    bot = _pts([(s - c - 0.5, s - y - 0.5) for c in range(d) for y in range(k - 2 - c)])
    cor = _pts([(0.5, s - 0.5), (s - 0.5, 0.5)])
    flat = np.vstack([top, bot, cor, st + np.array([0.0, d]), _refl(st, 0.0, d)])
    r = np.vstack(rot)
    return np.vstack([np.column_stack([flat, np.zeros(len(flat))]), np.column_stack([r, np.full(len(r), ANG)])])


def run(kmax, dry, procs):
    cands, new = [], []
    for k in range(7, kmax + 1, 3):
        n = member_n(k)
        if n > store.MAX_N:
            break
        arr = member(k)
        assert arr.shape == (n, 3), (k, arr.shape)
        L = S1 + (k - 7)
        built = float(store.side_length(store.normalize(arr)))
        fixed = polish(store.normalize(arr))
        pl = "FAIL" if fixed is None else f"{float(store.side_length(fixed)) - L:+.1e}"
        best = float(store.best_L(n))
        print(f"k={k:2d} n={n:4d} L={L:.9f} built-L={built - L:+.1e} polish-L={pl} best={best:.9f} best-L={best - L:+.1e}")
        if fixed is not None and best > L + 1e-9:
            cands.append((n, arr, f"fam0.8467 DeVincentis/Ellsworth k={k}"))
            new.append((n, L))
    Ls = {n: float(store.best_L(n)) for n in range(1, store.MAX_N + 1)}
    old = dict(Ls)
    for n, L in new:
        Ls[n] = min(Ls[n], L)
    _close(Ls, store.MAX_N)
    so = sum(_score(old[n], n) for n in old)
    sn = sum(_score(Ls[n], n) for n in Ls)
    imp = sorted(n for n in Ls if Ls[n] < old[n] - 1e-9)
    print(f"cands={len(cands)} dry total {so:.10f} -> {sn:.10f}; improved n: {len(imp)} {imp[:20]}")
    if dry or not cands:
        return
    for x in store.try_update_many(cands, polish_if_needed=True, processes=procs):
        print(" ", x)
    store.push_down(verbose=False)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--kmax", type=int, default=40)
    p.add_argument("--procs", type=int, default=6)
    a = p.parse_args(argv)
    run(a.kmax, a.dry, a.procs)


if __name__ == "__main__":
    main()
