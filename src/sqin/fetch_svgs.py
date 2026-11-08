from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

from sqin.grid import grid_side
from sqin.profile import FR_TOL, load_friedman
from sqin.store import root

BASE = "https://kingbird.myphotos.cc/packing/"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sqin-research/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def main(argv=None):
    p = argparse.ArgumentParser(prog="fetch_svgs")
    p.add_argument("--base", default=BASE)
    p.add_argument("--out", default=str(root() / "data" / "external" / "friedman_svg"))
    p.add_argument("--friedman", default=str(root() / "data" / "external" / "friedman_L_compact.csv"))
    p.add_argument("--extra", type=int, nargs="*", default=[])
    p.add_argument("--suffixes", nargs="*", default=["", "b", "c"])
    p.add_argument("--delay", type=float, default=1.0)
    a = p.parse_args(argv)
    fr = load_friedman(a.friedman)
    ns = {n for n, (L, _) in fr.items() if L < float(grid_side(n)) - FR_TOL} | set(a.extra)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    missing, new = [], 0
    for n in sorted(ns):
        have = False
        for suf in a.suffixes:
            name = f"square-{n}{suf}.svg"
            if (out / name).exists():
                have = True
                continue
            try:
                data = _get(a.base + name)
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    print(f"{name}: HTTP {e.code}", flush=True)
                time.sleep(a.delay)
                continue
            except Exception as e:  # noqa: BLE001
                print(f"{name}: {e}", flush=True)
                time.sleep(a.delay)
                continue
            if b"<svg" in data:
                (out / name).write_bytes(data)
                have = True
                new += 1
                print(f"saved {name} ({len(data)} bytes)", flush=True)
            time.sleep(a.delay)
        if not have:
            missing.append(n)
    print(f"new files: {new}; n without any file: {len(missing)} {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
