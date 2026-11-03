from __future__ import annotations

import math

import numpy as np
from numba import njit

HALF_PI = math.pi / 2.0
GRID_QUANT = 1024.0
ANGLE_SNAP = 1e-15


def normalize(xya):
    out = np.array(xya, dtype=np.float64, copy=True)
    a = np.mod(out[:, 2], HALF_PI)
    a[(a < ANGLE_SNAP) | (a > HALF_PI - ANGLE_SNAP)] = 0.0
    out[:, 2] = a
    out[out == 0.0] = 0.0
    return out


def corners(xya):
    x, y, a = xya[:, 0], xya[:, 1], xya[:, 2]
    c, s = np.cos(a), np.sin(a)
    dx1, dy1 = 0.5 * c - 0.5 * s, 0.5 * s + 0.5 * c
    dx2, dy2 = -0.5 * c - 0.5 * s, -0.5 * s + 0.5 * c
    out = np.empty((len(xya), 4, 2))
    out[:, 0, 0], out[:, 0, 1] = x + dx1, y + dy1
    out[:, 1, 0], out[:, 1, 1] = x + dx2, y + dy2
    out[:, 2, 0], out[:, 2, 1] = x - dx1, y - dy1
    out[:, 3, 0], out[:, 3, 1] = x - dx2, y - dy2
    return out


def extents(xya):
    c = corners(xya)
    return np.stack(
        [c[:, :, 0].min(1), c[:, :, 0].max(1), c[:, :, 1].min(1), c[:, :, 1].max(1)], axis=1
    )


def side_length(xya):
    e = extents(xya)
    return float(max(e[:, 1].max() - e[:, 0].min(), e[:, 3].max() - e[:, 2].min()))


def _excl_min(v):
    i = int(np.argmin(v))
    out = np.full(len(v), v[i])
    out[i] = np.min(np.delete(v, i))
    return out


def _excl_max(v):
    return -_excl_min(-v)


def side_without_each(xya):
    e = extents(np.asarray(xya, dtype=np.float64))
    w = _excl_max(e[:, 1]) - _excl_min(e[:, 0])
    h = _excl_max(e[:, 3]) - _excl_min(e[:, 2])
    return np.maximum(w, h)


def grid_flags(xya):
    gx = xya[:, 0] * GRID_QUANT
    gy = xya[:, 1] * GRID_QUANT
    return (xya[:, 2] == 0.0) & (gx == np.floor(gx)) & (gy == np.floor(gy))


@njit(cache=True)
def _pair_gap(xi, yi, ci, si, xj, yj, cj, sj):
    dx = xj - xi
    dy = yj - yi
    best = -1e300
    for k in range(4):
        if k == 0:
            ux, uy = ci, si
        elif k == 1:
            ux, uy = -si, ci
        elif k == 2:
            ux, uy = cj, sj
        else:
            ux, uy = -sj, cj
        proj = abs(ux * dx + uy * dy)
        h1 = 0.5 * (abs(ux * ci + uy * si) + abs(-ux * si + uy * ci))
        h2 = 0.5 * (abs(ux * cj + uy * sj) + abs(-ux * sj + uy * cj))
        g = proj - h1 - h2
        best = max(best, g)
    return best


@njit(cache=True)
def _min_gap_core(x, y, c, s, grid, pad):
    n = len(x)
    h = 0.5 * (np.abs(c) + np.abs(s))
    order = np.argsort(x - h)
    best = np.inf
    for a in range(n):
        i = order[a]
        xmax_i = x[i] + h[i] + pad
        for b in range(a + 1, n):
            j = order[b]
            if x[j] - h[j] > xmax_i:
                break
            if abs(y[i] - y[j]) > h[i] + h[j] + pad:
                continue
            if grid[i] and grid[j]:
                g = max(abs(x[i] - x[j]), abs(y[i] - y[j])) - 1.0
                if g >= 0.0:
                    continue
            else:
                g = _pair_gap(x[i], y[i], c[i], s[i], x[j], y[j], c[j], s[j])
            best = min(best, g)
    return best


def min_sat_gap(xya, pad=1e-6):
    a = np.ascontiguousarray(xya, dtype=np.float64)
    x = np.ascontiguousarray(a[:, 0])
    y = np.ascontiguousarray(a[:, 1])
    c = np.cos(a[:, 2])
    s = np.sin(a[:, 2])
    return float(_min_gap_core(x, y, c, s, grid_flags(a), pad))
