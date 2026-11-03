from __future__ import annotations

import csv
import io
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from sqin.geometry import normalize
from sqin.grid import grid

MAX_N = 1000
HEADER = "id,n,square_id,x,y,angle"


def fmt(v):
    return repr(float(v) + 0.0)


def fmt_angle(a):
    a = float(a)
    return "0" if a == 0.0 else repr(a)


def row_id(n, i):
    return n * (n - 1) // 2 + i


def _fmt_xya(xya):
    arr = normalize(xya)
    return [(fmt(r[0]), fmt(r[1]), fmt_angle(r[2])) for r in arr]


def format_rows(n, xya):
    if len(xya) != n:
        raise ValueError(f"expected {n} squares, got {len(xya)}")
    return [
        {"n": str(n), "square_id": str(i), "x": x, "y": y, "angle": a}
        for i, (x, y, a) in enumerate(_fmt_xya(xya))
    ]


def write_submission(path, sol, fill_grid=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        f.write(HEADER + "\n")
        for n in range(1, MAX_N + 1):
            xya = sol.get(n)
            if xya is None:
                if not fill_grid:
                    continue
                xya = grid(n)
            f.writelines(f"{row_id(n, i)},{n},{i},{x},{y},{a}\n" for i, (x, y, a) in enumerate(_fmt_xya(xya)))
    os.replace(tmp, path)


def read_rows(path, via_pandas=False):
    if via_pandas:
        import pandas as pd

        text = pd.read_csv(path).drop(columns=["id"], errors="ignore").to_csv(index=False)
    else:
        text = Path(path).read_text()
    buckets = defaultdict(list)
    for row in csv.DictReader(io.StringIO(text)):
        buckets[int(row["n"])].append(row)
    return dict(buckets)


def read_submission(path):
    out = {}
    for n, rows in sorted(read_rows(path).items()):
        if not 1 <= n <= MAX_N or len(rows) != n:
            warnings.warn(f"n={n}: skipped (bad n or row count)")
            continue
        try:
            arr = np.empty((n, 3))
            seen = set()
            for r in rows:
                i = int(r["square_id"])
                arr[i] = (float(r["x"]), float(r["y"]), float(r["angle"]))
                seen.add(i)
            if seen != set(range(n)):
                raise ValueError("square_id set")
        except (ValueError, TypeError, IndexError):
            warnings.warn(f"n={n}: skipped (unparseable)")
            continue
        out[n] = arr
    return out
