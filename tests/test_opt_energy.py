import numpy as np

from sqin.geometry import min_sat_gap
from sqin.opt.energy import energy_grad


def _e(z, n, L):
    g = np.empty(3 * n)
    return energy_grad(z, n, L, g), g


def test_gradient_fd():
    rng = np.random.default_rng(1)
    n = 6
    z = np.concatenate([rng.uniform(0.3, 2.2, 2 * n), rng.uniform(0.1, 1.4, n)])
    L = 2.5
    e, g = _e(z, n, L)
    num = np.zeros_like(z)
    h = 1e-6
    for i in range(len(z)):
        zp, zm = z.copy(), z.copy()
        zp[i] += h
        zm[i] -= h
        num[i] = (_e(zp, n, L)[0] - _e(zm, n, L)[0]) / (2 * h)
    assert e > 0
    assert np.allclose(g, num, rtol=1e-4, atol=1e-6)


def test_depth_matches_geometry():
    rng = np.random.default_rng(2)
    for _ in range(50):
        xya = np.array([[50.0, 50.0, rng.uniform(0.1, 1.4)],
                        [50.0 + rng.uniform(-0.9, 0.9), 50.0 + rng.uniform(-0.9, 0.9), rng.uniform(0.1, 1.4)]])
        z = np.concatenate([xya[:, 0], xya[:, 1], xya[:, 2]])
        e, _ = _e(z, 2, 100.0)
        gap = min_sat_gap(xya)
        assert abs(e - max(0.0, -gap) ** 2) < 1e-12


def test_grid_zero_energy():
    k = 3
    xs = np.array([(i % k + 0.5) * 1.001 for i in range(9)])
    ys = np.array([(i // k + 0.5) * 1.001 for i in range(9)])
    z = np.concatenate([xs, ys, np.zeros(9)])
    assert _e(z, 9, 3.003)[0] == 0.0
