from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import os
import time

import numpy as np

from sqin.geometry import normalize, side_length
from sqin.grid import grid_side
from sqin.opt.core import SEED_BASE, solve_run
from sqin.profile import load_friedman
from sqin.store import push_down, root, total_score, try_update_many
from sqin.validate import fast_check, gate, polish

MAX_POLISH_GROWTH = 1e-7
N5_LIMIT = 2.70710678119 + 1e-9


def finalize(xya):
    arr = normalize(xya)
    l0 = side_length(arr)
    if not fast_check(arr):
        arr = polish(arr)
        if arr is None:
            return None
    l1 = side_length(arr)
    if l1 > l0 + MAX_POLISH_GROWTH:
        return None
    return arr, l1


def _dir():
    d = root() / "runs" / "opt"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ckpt(n, run):
    return _dir() / f"n{n:04d}_r{run:03d}.npz"


def _save(p, xya, L, secs):
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, xya=xya, L=np.float64(L), secs=np.float64(secs))
    os.replace(tmp, p)


def _load(p):
    with np.load(p) as z:
        return z["xya"].copy(), float(z["L"]), float(z["secs"])


def work(task):
    n, runs, neighbors = task
    best_xya, best_L, secs, new = None, math.inf, 0.0, 0
    for run in range(runs):
        p = _ckpt(n, run)
        if p.exists():
            xya, L, t = _load(p)
        else:
            t0 = time.time()
            res = solve_run(n, run, neighbors)
            fin = finalize(res) if res is not None else None
            t = time.time() - t0
            xya, L = (fin[0], fin[1]) if fin is not None else (np.empty((0, 3)), math.inf)
            _save(p, xya, L, t)
            new += 1
        secs += t
        if L < best_L and len(xya) == n:
            best_xya, best_L = xya, L
    return n, best_xya, best_L, secs, new


def _time_gate(ns):
    for n in ns:
        t0 = time.time()
        res = solve_run(n, 0)
        fin = finalize(res) if res is not None else None
        ts = time.time() - t0
        if fin is None:
            print(f"n={n}: no candidate (solve {ts:.1f}s)")
            continue
        t0 = time.time()
        ok, _L, reason = gate(n, fin[0])
        print(f"n={n}: solve={ts:.1f}s gate={time.time() - t0:.1f}s ok={ok} L={fin[1]:.12f} {reason}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sqin-opt")
    p.add_argument("--nmin", type=int, default=2)
    p.add_argument("--nmax", type=int, default=30)
    p.add_argument("--runs", type=int, default=24)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--neighbors", action="store_true")
    p.add_argument("--friedman", default=str(root() / "data" / "external" / "friedman_L_compact.csv"))
    p.add_argument("--time-gate", type=int, nargs="*")
    a = p.parse_args(argv)

    if a.time_gate:
        _time_gate(a.time_gate)
        return 0

    fr = load_friedman(a.friedman)
    ns = sorted(range(a.nmin, a.nmax + 1), reverse=True)
    tasks = [(n, a.runs, a.neighbors) for n in ns]
    ctx = multiprocessing.get_context("fork")
    rows = []
    print(f"seed_base={SEED_BASE} runs={a.runs} procs={a.procs}")
    print(f"{'n':>4} {'L_ours':>16} {'L_friedman':>16} {'diff':>11} {'score':>10} {'secs':>8}  status")
    with ctx.Pool(processes=min(a.procs, len(tasks))) as pool:
        for n, xya, L, secs, new in pool.imap_unordered(work, tasks, chunksize=1):
            status = "no_candidate"
            if xya is not None:
                src = f"own opt n={n} runs={a.runs} seed={SEED_BASE}; own work; no external license"
                status = try_update_many([(n, xya, src)], polish_if_needed=True, processes=1)[0][1]
            fL, tag = (fr[n][0], "fr") if n in fr else (float(grid_side(n)), "grid")
            diff = L - fL if xya is not None else math.nan
            score = max(0.0, L / math.sqrt(n) - 1.0) if xya is not None else math.nan
            rows.append((n, L, fL, tag, diff, score, secs, status))
            print(f"{n:4d} {L:16.12f} {fL:16.12f} {diff:11.3e} {score:10.6f} {secs:8.1f}  {status} [{tag}]", flush=True)
            if n == 5 and xya is not None:
                print(f"     n=5 criterion L <= {N5_LIMIT:.12f}: {'OK' if L <= N5_LIMIT else 'FAIL'}")
    rows.sort()
    out = root() / "runs" / f"opt_summary_{a.nmin}_{a.nmax}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "L_ours", "L_ref", "ref", "diff", "score", "secs", "status"])
        w.writerows(rows)
    bad = [r[0] for r in rows if r[3] == "fr" and (math.isnan(r[4]) or r[4] > 1e-6)]
    print(f"\nn with diff > 1e-6 vs friedman: {bad}")
    print(f"pushdown updated: {push_down(start=a.nmax)}")
    t, g = total_score()
    print(f"total = {t:.10f}   [" + " / ".join(f"{k}: {v:.6f}" for k, v in g.items()) + "]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
