from __future__ import annotations

import math
import os
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

from sqin.geometry import normalize, side_length, side_without_each
from sqin.grid import grid, grid_score, grid_side
from sqin.validate import DEFAULT_PROCS, fast_check, gate_many, polish

MAX_N = 1000
MIN_GAIN = 1e-11
CHUNK = 32


def root():
    return Path(os.environ.get("SQIN_ROOT", "."))


def best_dir():
    d = root() / "data" / "best"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(n):
    return best_dir() / f"{n:04d}.npz"


def has_best(n):
    return _path(n).exists()


def load_best(n):
    p = _path(n)
    if not p.exists():
        return grid(n), Decimal(grid_side(n)), "grid"
    with np.load(p, allow_pickle=False) as z:
        return z["xya"].copy(), Decimal(str(z["L"])), str(z["source"])


def best_L(n):
    return load_best(n)[1]


def _save(n, xya, L_str, source):
    p = _path(n)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, xya=xya, L=np.array(L_str), source=np.array(source))
    os.replace(tmp, p)


def _log(n, old, new, source):
    d = root() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "updates.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{n}\t{old:.15f}\t{new:.15f}\t{source}\n")


def try_update_many(cands, polish_if_needed=False, processes=DEFAULT_PROCS):
    results = [None] * len(cands)
    pending = []
    for idx, (n, xya, source) in enumerate(cands):
        arr = normalize(xya)
        if arr.shape != (n, 3) or not np.all(np.isfinite(arr)):
            results[idx] = (n, "rejected_fast", "bad shape or non-finite")
            continue
        best_f = float(best_L(n))
        Lf = side_length(arr)
        if Lf >= best_f - MIN_GAIN:
            results[idx] = (n, "not_better", f"L={Lf:.12f}")
            continue
        if not fast_check(arr):
            fixed = polish(arr) if polish_if_needed else None
            if fixed is None:
                results[idx] = (n, "rejected_fast", "separation below SEP_EPS")
                continue
            arr = fixed
            Lf = side_length(arr)
            if Lf >= best_f - MIN_GAIN:
                results[idx] = (n, "not_better", f"after polish L={Lf:.12f}")
                continue
        pending.append((idx, n, arr))
    for s in range(0, len(pending), CHUNK):
        chunk = pending[s : s + CHUNK]
        gres = gate_many([(n, arr) for _, n, arr in chunk], processes=processes)
        for (idx, n, arr), (ok, L_str, reason) in zip(chunk, gres):
            if not ok:
                results[idx] = (n, "rejected_gate", reason)
                continue
            old, new = best_L(n), Decimal(L_str)
            if new < old:
                source = cands[idx][2]
                _save(n, arr, L_str, source)
                _log(n, old, new, source)
                results[idx] = (n, "updated", f"{old:.12f} -> {new:.12f}")
            else:
                results[idx] = (n, "not_better", f"gate L={new:.12f}")
    return results


def try_update(n, xya, source, processes=1):
    r = try_update_many([(n, xya, source)], processes=processes)[0]
    return r[1] == "updated", r


def push_down_candidate(xya):
    arr = np.asarray(xya, dtype=np.float64)
    if len(arr) < 2:
        return None
    ls = side_without_each(arr)
    i = int(np.argmin(ls))
    return np.delete(arr, i, axis=0), float(ls[i])


def push_down(start=MAX_N - 1, verbose=True):
    updated = []
    for n in range(start, 0, -1):
        up, _, src = load_best(n + 1)
        cand = push_down_candidate(up)
        if cand is None:
            continue
        arr, lf = cand
        if lf >= float(best_L(n)) - MIN_GAIN:
            continue
        status = try_update_many([(n, arr, f"pushdown(n={n + 1}) <- {src}")], processes=1)[0]
        if status[1] == "updated":
            updated.append(n)
        if verbose:
            print(f"n={n}: {status[1]} {status[2]}", flush=True)
    return updated


def score_n(n):
    if not has_best(n):
        return grid_score(n)
    return max(0.0, float(best_L(n)) / math.sqrt(n) - 1.0)


def total_score():
    groups = {"<=100": [], "101-400": [], "401-1000": []}
    for n in range(1, MAX_N + 1):
        key = "<=100" if n <= 100 else "101-400" if n <= 400 else "401-1000"
        groups[key].append(score_n(n))
    sums = {k: math.fsum(v) for k, v in groups.items()}
    return math.fsum(sums.values()), sums
