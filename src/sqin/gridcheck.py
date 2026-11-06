from __future__ import annotations

import numpy as np

from sqin.grid import grid


def rows_are_grid(n, rows):
    if len(rows) != n:
        return False
    try:
        rs = sorted(rows, key=lambda r: int(r["square_id"]))
        arr = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rs])
    except (KeyError, ValueError):
        return False
    ref = np.asarray(grid(n), dtype=np.float64)
    if arr.shape != ref.shape or np.any(arr[:, 2] != 0.0) or np.any(ref[:, 2] != 0.0):
        return False
    return bool(np.array_equal(arr[:, :2] - arr[0, :2], ref[:, :2] - ref[0, :2]))
