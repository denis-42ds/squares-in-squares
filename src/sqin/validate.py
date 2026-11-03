from __future__ import annotations

import multiprocessing
import sys

import numpy as np

from sqin.geometry import min_sat_gap, normalize
from sqin.io import format_rows
from sqin.vendor.metric import evaluate_single_puzzle

SEP_EPS = 1e-12
MAX_COORD = 1e4
MAX_SCALE = 1e-3
DEFAULT_PROCS = 8


def fast_check(xya, eps=SEP_EPS):
    a = np.asarray(xya, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) == 0 or not np.all(np.isfinite(a)):
        return False
    a = normalize(a)
    if np.abs(a[:, :2]).max() > MAX_COORD:
        return False
    return min_sat_gap(a) >= eps


def polish(xya, eps=SEP_EPS, max_iter=12):
    cur = normalize(xya)
    g = min_sat_gap(cur)
    if g >= eps:
        return cur
    cen = cur[:, :2].mean(axis=0)
    d = 2.0 * max(eps - g, eps)
    for _ in range(max_iter):
        if d > MAX_SCALE:
            break
        out = cur.copy()
        out[:, :2] = cen + (1.0 + d) * (cur[:, :2] - cen)
        out = normalize(out)
        if fast_check(out, eps):
            return out
        d *= 2.0
    return None


def gate_rows(n, rows):
    try:
        _, _, L, reason = evaluate_single_puzzle(n, rows)
    except Exception as e:  # noqa: BLE001
        return (False, None, f"error: {e}")
    return (L is not None, L, reason)


def gate(n, xya):
    return gate_rows(n, format_rows(n, xya))


def _worker(task):
    kind, n, payload = task
    rows = format_rows(n, payload) if kind == "xya" else payload
    return gate_rows(n, rows)


def _run(tasks, processes, progress):
    if not tasks:
        return []
    if processes <= 1 or len(tasks) == 1:
        out = []
        for k, t in enumerate(tasks):
            out.append(_worker(t))
            if progress:
                print(f"\rgate {k + 1}/{len(tasks)}", end="", file=sys.stderr, flush=True)
        if progress:
            print(file=sys.stderr)
        return out
    order = sorted(range(len(tasks)), key=lambda i: (-tasks[i][1], i))
    out = [None] * len(tasks)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=min(processes, len(tasks))) as pool:
        it = pool.imap(_worker, [tasks[i] for i in order], chunksize=1)
        for k, (i, r) in enumerate(zip(order, it)):
            out[i] = r
            if progress and (k + 1) % 50 == 0:
                print(f"\rgate {k + 1}/{len(tasks)}", end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    return out


def gate_many(items, processes=DEFAULT_PROCS, progress=False):
    return _run([("xya", n, xya) for n, xya in items], processes, progress)


def gate_rows_many(items, processes=DEFAULT_PROCS, progress=False):
    return _run([("rows", n, rows) for n, rows in items], processes, progress)
