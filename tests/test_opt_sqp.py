import math

import numpy as np

from sqin.geometry import side_length
from sqin.opt.sqp import _cons, sqp_polish


def _n5_start(L0):
    c = L0 / 2
    x = [0.5, L0 - 0.5, 0.5, L0 - 0.5, c]
    y = [0.5, 0.5, L0 - 0.5, L0 - 0.5, c]
    a = [0.0, 0.0, 0.0, 0.0, math.pi / 4]
    return np.array(x + y + a)


def test_sqp_n5_exact():
    z = sqp_polish(_n5_start(2.72), 5, 2.72)
    xya = np.stack([z[:5], z[5:10], z[10:]], axis=1)
    assert abs(side_length(xya) - (2 + 1 / math.sqrt(2))) < 1e-9


def test_sqp_n5_perturbed():
    rng = np.random.default_rng(7)
    z = _n5_start(2.72)
    z[:10] += rng.normal(0, 1e-4, 10)
    z = sqp_polish(z, 5, 2.72)
    xya = np.stack([z[:5], z[5:10], z[10:]], axis=1)
    assert abs(side_length(xya) - (2 + 1 / math.sqrt(2))) < 1e-8


def test_cons_jacobian_fd():
    rng = np.random.default_rng(3)
    n = 4
    v = np.concatenate([rng.uniform(0.5, 2.5, 2 * n), rng.uniform(0.1, 1.4, n), [3.0]])
    I = np.array([0, 0, 1, 2], dtype=np.int64)
    J = np.array([1, 2, 3, 3], dtype=np.int64)
    K = np.array([0, 2, 1, 3], dtype=np.int64)
    m = 4 * n + len(I)
    c = np.empty(m)
    Jm = np.zeros((m, 3 * n + 1))
    _cons(v, n, I, J, K, 0.0, c, Jm)
    num = np.zeros_like(Jm)
    h = 1e-6
    for t in range(len(v)):
        vp, vm = v.copy(), v.copy()
        vp[t] += h
        vm[t] -= h
        cp, cm, tmp = np.empty(m), np.empty(m), np.zeros_like(Jm)
        _cons(vp, n, I, J, K, 0.0, cp, tmp)
        _cons(vm, n, I, J, K, 0.0, cm, tmp)
        num[:, t] = (cp - cm) / (2 * h)
    assert np.allclose(Jm, num, rtol=1e-4, atol=1e-6)
