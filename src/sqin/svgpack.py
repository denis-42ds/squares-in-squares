from __future__ import annotations

import argparse
import itertools
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from sqin.geometry import min_sat_gap, normalize, side_length
from sqin.grid import grid_side
from sqin.profile import FR_TOL, load_friedman
from sqin.store import root, total_score, try_update_many

XLINK = "{http://www.w3.org/1999/xlink}href"
_SKIP = {"defs", "script", "title", "desc", "style", "metadata"}
_TOK = re.compile(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
TOL = 1e-9
BAD_GAP = -1e-9
_ENT = re.compile(r"<!ENTITY\s+([^\s\"'>]+)\s+(?:\"([^\"]*)\"|'([^']*)')\s*>")
_REF = re.compile(r"&([^\s;&<>\"']+);")
_STD = {"amp", "lt", "gt", "quot", "apos"}


def _expand(text, ents):
    for _ in range(10):
        new = _REF.sub(lambda m: ents[m.group(1)] if m.group(1) in ents else m.group(0), text)
        if new == text:
            break
        text = new
    left = sorted({m.group(1) for m in _REF.finditer(text) if m.group(1) not in _STD and not m.group(1).startswith("#")})
    if left:
        raise ValueError(f"undefined entities: {left}")
    return text


def _T(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _transform(t):
    M = np.eye(3)
    for name, args in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", t or ""):
        v = [float(x) for x in re.split(r"[,\s]+", args.strip()) if x]
        if name == "translate":
            T = _T(v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale":
            T = np.diag([v[0], v[1] if len(v) > 1 else v[0], 1.0])
        elif name == "rotate":
            a = math.radians(v[0])
            c, s = math.cos(a), math.sin(a)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            T = _T(v[1], v[2]) @ R @ _T(-v[1], -v[2]) if len(v) == 3 else R
        elif name == "matrix":
            a, b, c, d, e, f = v
            T = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
        else:
            raise ValueError(f"unsupported transform {name}")
        M = M @ T
    return M


def _parse_path(d):
    if _TOK.sub("", d).replace(",", "").strip():
        raise ValueError("unsupported characters in path")
    toks = _TOK.findall(d)
    need = {"M": 2, "L": 2, "H": 1, "V": 1}
    subs, cur = [], []
    pos = start = (0.0, 0.0)
    cmd = None
    i = 0

    def close():
        nonlocal cur
        if len(cur) >= 3:
            subs.append(cur)
        cur = []

    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            if t.upper() not in "MLHVZ":
                raise ValueError(f"unsupported path command {t}")
            cmd = t
            i += 1
            if t in "Zz":
                close()
                pos = start
                continue
        elif cmd is None or cmd in "Zz":
            raise ValueError("number without command")
        up = cmd.upper()
        k = need[up]
        v = [float(x) for x in toks[i : i + k]]
        if len(v) < k:
            raise ValueError("truncated path")
        i += k
        rel = cmd.islower()
        x, y = pos
        if up in "ML":
            nx, ny = (x + v[0], y + v[1]) if rel else (v[0], v[1])
        elif up == "H":
            nx, ny = (x + v[0] if rel else v[0]), y
        else:
            nx, ny = x, (y + v[0] if rel else v[0])
        if up == "M":
            close()
            start = (nx, ny)
            cur = [start]
            cmd = "l" if rel else "L"
        else:
            if not cur:
                cur = [pos]
            cur.append((nx, ny))
        pos = (nx, ny)
    close()
    return subs


def _slabs(subs):
    ys = sorted({p[1] for s in subs for p in s})
    cl = []
    for y in ys:
        if not cl or y - cl[-1] > TOL:
            cl.append(y)
    edges = []
    for s in subs:
        m = len(s)
        for k in range(m):
            (x1, y1), (x2, y2) = s[k], s[(k + 1) % m]
            if abs(x1 - x2) <= TOL:
                if abs(y1 - y2) > TOL:
                    edges.append((x1, min(y1, y2), max(y1, y2)))
            elif abs(y1 - y2) > TOL:
                raise ValueError("non-orthogonal path edge")
    frags = []
    for y0, y1 in itertools.pairwise(cl):
        ym = 0.5 * (y0 + y1)
        xs = sorted(x for x, lo, hi in edges if lo < ym < hi)
        if len(xs) % 2:
            raise ValueError("odd edge crossing")
        for a, b in zip(xs[0::2], xs[1::2]):
            if b - a > TOL:
                frags.append((a, b, y0, y1))
    return frags


def _cells(subs):
    frags = _slabs(subs)
    out = []
    for _ in range(100000):
        if not frags:
            return out
        ymin = min(f[2] for f in frags)
        f = min((g for g in frags if g[2] <= ymin + TOL), key=lambda g: g[0])
        sx, sy = f[0], ymin
        rest, removed = [], 0.0
        for a0, a1, b0, b1 in frags:
            i0, i1 = max(a0, sx), min(a1, sx + 1.0)
            j0, j1 = max(b0, sy), min(b1, sy + 1.0)
            if i1 - i0 <= TOL or j1 - j0 <= TOL:
                rest.append((a0, a1, b0, b1))
                continue
            removed += (i1 - i0) * (j1 - j0)
            for q in ((a0, a1, b0, j0), (a0, a1, j1, b1), (a0, i0, j0, j1), (i1, a1, j0, j1)):
                if q[1] - q[0] > TOL and q[3] - q[2] > TOL:
                    rest.append(q)
        if abs(removed - 1.0) > 1e-6:
            raise ValueError(f"region is not tiled by unit squares near ({sx:.6f},{sy:.6f})")
        out.append((sx + 0.5, sy + 0.5))
        frags = rest
    raise ValueError("tiling did not terminate") 


def _add(out, M, lx, ly):
    v = M @ np.array([lx, ly, 1.0])
    out.append((float(v[0]), float(v[1]), math.atan2(M[1, 0], M[0, 0])))


def _is_int(v):
    return abs(v - round(v)) < TOL and round(v) >= 1


def _walk(el, M, fill_none, ids, out, depth=0):
    if depth > 40:
        raise ValueError("recursion too deep")
    tag = el.tag.rsplit("}", 1)[-1]
    if tag in _SKIP:
        return
    fn = fill_none
    if el.get("fill") is not None:
        fn = el.get("fill").strip() == "none"
    m = re.search(r"(?:^|;)\s*fill\s*:\s*([^;]+)", el.get("style") or "")
    if m:
        fn = m.group(1).strip() == "none"
    M = M @ _transform(el.get("transform"))
    if tag == "use":
        href = el.get(XLINK) or el.get("href") or ""
        ref = ids.get(href.lstrip("#"))
        if ref is None:
            raise ValueError(f"unresolved use {href}")
        _walk(ref, M @ _T(float(el.get("x", 0)), float(el.get("y", 0))), fn, ids, out, depth + 1)
    elif tag in ("g", "svg"):
        for ch in el:
            _walk(ch, M, fn, ids, out, depth + 1)
    elif tag == "rect":
        if fn or el.get("id") == "outer":
            return
        w, h = float(el.get("width", 0)), float(el.get("height", 0))
        if not (_is_int(w) and _is_int(h)):
            return
        x0, y0 = float(el.get("x", 0)), float(el.get("y", 0))
        for i in range(round(w)):
            for j in range(round(h)):
                _add(out, M, x0 + i + 0.5, y0 + j + 0.5)
    elif tag == "path":
        if fn:
            return
        for lx, ly in _cells(_parse_path(el.get("d", ""))):
            _add(out, M, lx, ly)
    elif tag in ("polygon", "polyline"):
        raise ValueError(f"unsupported element {tag}")


def parse_svg_text(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<\?xml[^>]*\?>", "", text)
    ents = {m.group(1): (m.group(2) if m.group(2) is not None else m.group(3)) for m in _ENT.finditer(text)}
    text = re.sub(r"<!DOCTYPE[^\[>]*\[.*?\]\s*>", "", text, flags=re.DOTALL)
    text = _expand(text, ents)
    tree = ET.fromstring(text)
    ids = {}
    for e in tree.iter():
        i = e.get("id")
        if i and i not in ids:
            ids[i] = e
    out = []
    _walk(tree, np.eye(3), False, ids, out)
    if not out:
        raise ValueError("no squares found")
    return np.array(out, dtype=np.float64)


def parse_svg(path):
    return parse_svg_text(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="svgpack")
    p.add_argument("files", nargs="+")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--friedman", default=str(root() / "data" / "external" / "friedman_L_compact.csv"))
    a = p.parse_args(argv)
    fr = load_friedman(a.friedman)
    cands, failed = [], []
    for f in sorted(a.files):
        name = Path(f).name
        m = re.search(r"(\d+)", name)
        try:
            xya = parse_svg(f)
        except Exception as e:  # noqa: BLE001
            failed.append((name, f"parse error: {e}"))
            continue
        n = len(xya)
        if m and int(m.group(1)) != n:
            failed.append((name, f"count {n} != n in name {m.group(1)}"))
            continue
        arr = normalize(xya)
        gap, L = min_sat_gap(arr), side_length(arr)
        if gap < BAD_GAP:
            failed.append((name, f"overlap gap={gap:.3e}"))
            continue
        ref = f" diff_vs_csv={L - fr[n][0]:.3e}" if n in fr else ""
        print(f"{name}: n={n} L={L:.14f} min_gap={gap:.3e}{ref}", flush=True)
        src = f"friedman svg {name}; erich-friedman.github.io (authors in file header); license: not stated"
        cands.append((n, xya, src))
    have = {n for n, _, _ in cands}
    todo = sorted(n for n, (L, _) in fr.items() if L < float(grid_side(n)) - FR_TOL and n not in have)
    print(f"nontrivial csv n without candidate: {len(todo)} {todo[:80]}")
    for name, why in failed:
        print(f"FAILED {name}: {why}")
    if a.dry or not cands:
        return 0
    res = try_update_many(cands, polish_if_needed=True, processes=a.procs)
    print(dict(Counter(s for _, s, _ in res)))
    for n, s, d in res:
        if s in ("rejected_fast", "rejected_gate"):
            print(f"  n={n}: {s} {d}")
    t, g = total_score()
    print(f"total = {t:.10f}   [" + " / ".join(f"{k}: {v:.6f}" for k, v in g.items()) + "]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
