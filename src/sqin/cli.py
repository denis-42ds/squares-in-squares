from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

from sqin.grid import grid_score, grid_total
from sqin.io import MAX_N, read_rows, read_submission, write_submission
from sqin.store import (
    has_best,
    load_best,
    push_down,
    root,
    total_score,
    try_update_many,
)
from sqin.validate import DEFAULT_PROCS, gate_rows_many


def _print_total():
    total, groups = total_score()
    g = " / ".join(f"{k}: {v:.6f}" for k, v in groups.items())
    print(f"total = {total:.10f}   [{g}]")


def verify_file(path, procs=DEFAULT_PROCS, raw=False):
    t0 = time.time()
    rows = read_rows(path, via_pandas=not raw)
    ns = sorted(n for n in rows if 1 <= n <= MAX_N)
    res = gate_rows_many([(n, rows[n]) for n in ns], processes=procs, progress=True)
    got = dict(zip(ns, res))
    total, bad, lines = 0.0, [], []
    for n in range(1, MAX_N + 1):
        r = got.get(n)
        if r is not None and r[0]:
            total += max(0.0, float(Decimal(r[1])) / math.sqrt(n) - 1.0)
            lines.append((n, 1, r[1], r[2]))
        else:
            total += grid_score(n)
            reason = "missing" if r is None else r[2]
            bad.append((n, reason))
            lines.append((n, 0, "", reason))
    out = root() / "runs"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"verify_{Path(path).stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "ok", "L", "reason"])
        w.writerows(lines)
    print(f"ok={MAX_N - len(bad)} bad={len(bad)} total={total:.10f} time={time.time() - t0:.1f}s")
    for n, reason in bad[:20]:
        print(f"  n={n}: {reason}")
    return total, bad


def cmd_baseline(a):
    out = Path(a.out) if a.out else root() / "data" / "submissions" / "baseline.csv"
    write_submission(out, {})
    print(f"wrote {out}; grid total = {grid_total():.10f}")


def cmd_verify(a):
    verify_file(a.file, a.procs, a.raw)


def cmd_ingest(a):
    sol = read_submission(a.file)
    cands = [(n, sol[n], a.source) for n in sorted(sol)]
    res = try_update_many(cands, polish_if_needed=not a.no_polish, processes=a.procs)
    print(dict(Counter(s for _, s, _ in res)))
    for n, s, d in res:
        if s in ("updated", "rejected_fast", "rejected_gate"):
            print(f"  n={n}: {s} {d}")
    _print_total()


def cmd_build(a):
    out = Path(a.out) if a.out else root() / "data" / "submissions" / "submission.csv"
    sol = {n: load_best(n)[0] for n in range(1, MAX_N + 1) if has_best(n)}
    write_submission(out, sol)
    print(f"wrote {out} ({len(sol)} non-grid puzzles)")
    _print_total()
    if a.verify:
        verify_file(out, a.procs)


def cmd_status(a):
    _print_total()
    rows = [(n, *load_best(n)[1:]) for n in range(1, MAX_N + 1) if has_best(n)]
    print(f"non-grid puzzles: {len(rows)}")
    for n, L, src in rows[:30]:
        print(f"  n={n}: L={L:.12f} source={src}")


def cmd_pushdown(a):
    upd = push_down(start=a.start)
    print(f"updated: {upd}")
    _print_total()


def main(argv=None):
    p = argparse.ArgumentParser(prog="sqin")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("baseline")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_baseline)

    s = sub.add_parser("verify")
    s.add_argument("file")
    s.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    s.add_argument("--raw", action="store_true")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("ingest")
    s.add_argument("file")
    s.add_argument("--source", required=True)
    s.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    s.add_argument("--no-polish", action="store_true")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("build")
    s.add_argument("--out")
    s.add_argument("--verify", action="store_true")
    s.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("status")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("pushdown")
    s.add_argument("--start", type=int, default=MAX_N - 1)
    s.set_defaults(fn=cmd_pushdown)

    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
