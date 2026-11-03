from __future__ import annotations

import math

import numpy as np


def grid_side(n):
    k = math.isqrt(n)
    return k if k * k >= n else k + 1


def grid(n):
    k = grid_side(n)
    idx = np.arange(n)
    out = np.zeros((n, 3))
    out[:, 0] = idx % k + 0.5
    out[:, 1] = idx // k + 0.5
    return out


def grid_score(n):
    return grid_side(n) / math.sqrt(n) - 1.0


def grid_total(max_n=1000):
    return math.fsum(grid_score(n) for n in range(1, max_n + 1))
