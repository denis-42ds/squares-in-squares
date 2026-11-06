import csv
import html
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/external/friedman_L.csv")
SOURCE = "kingbird.myphotos.cc/packing triangular table (D. Ellsworth, after E. Friedman), saved 2026-11-06"

text = SRC.read_text(encoding="utf-8")
chunks = text.split('<td class="box">')[1:]

tag_re = re.compile(r"<[^>]+>")
label_re = re.compile(r'<font size="\+3">(.*?)<br>', re.DOTALL)
ann_re = re.compile(r'<annotation encoding="application/x-tex">(.*?)</annotation>', re.DOTALL)
desc_re = re.compile(r'<div class="desc">(.*)</div>', re.DOTALL)
nn_re = re.compile(r"\\Nn\{([0-9.]+)\}")
num_re = re.compile(r"\d+\s*[\u2013-]\s*\d+|\d+")


def parse_ns(label):
    out = []
    for tok in num_re.findall(label):
        parts = re.split(r"\s*[\u2013-]\s*", tok)
        if len(parts) == 2:
            out.extend(range(int(parts[0]), int(parts[1]) + 1))
        else:
            out.append(int(parts[0]))
    return out


rows = {}
for chunk in chunks:
    chunk = chunk.split("</td>")[0]
    lm = label_re.search(chunk)
    am = ann_re.search(chunk)
    if not lm or not am:
        continue
    label = html.unescape(tag_re.sub(" ", lm.group(1)))
    ns = parse_ns(label)
    if not ns:
        continue
    tex = html.unescape(am.group(1)).strip()
    nn = nn_re.findall(tex)
    value, kind = "", "none"
    if nn:
        value, kind = nn[-1], "approx14"
    else:
        last = tex.split("=")[-1].strip().rstrip(".")
        try:
            value, kind = repr(float(last)), "exact"
        except ValueError:
            pass
    note = ""
    dm = desc_re.search(chunk)
    if dm and "<br>" in dm.group(1):
        note = html.unescape(tag_re.sub(" ", dm.group(1).split("<br>", 1)[1]))
        note = " ".join(note.split())
    for n in ns:
        rows[n] = (n, value, kind, SOURCE, note, " ".join(tex.split()))

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["n", "L", "kind", "source", "note", "tex"])
    for n in sorted(rows):
        w.writerow(rows[n])

missing = [n for n in sorted(rows) if not rows[n][1]]
print(f"rows: {len(rows)}, with value: {len(rows) - len(missing)}")
print(f"n range: {min(rows)}..{max(rows)}")
print("no value for n:", missing)
print("n=5:", rows.get(5))
