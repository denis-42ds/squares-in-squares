import math

import numpy as np

from sqin.svgpack import parse_svg_text

SVG = """<?xml version="1.0" encoding="UTF-8"?>
<!-- header [a] -->
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [
    <!ENTITY s  "3.5" > <!-- x [a] -->
]>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 &s; &s;" style="fill:#B2B2B2; stroke:black">
    <defs>
        <rect width="&s;" height="&s;" id="outer"/>
        <rect width="1" height="1" id="one"/>
    </defs>
    <use xlink:href="#outer" style="fill:white; stroke:none"/>
    <g transform="translate(0 &s;) scale(1 -1)">
        <path style="stroke:none" d="M0,2 H1 V1 H2 V0 H0"/>
        <path style="fill:none" d="M0,2 L1,2 L1,0"/>
        <use xlink:href="#one" transform="translate(2 1) rotate(45)"/>
    </g>
    <use xlink:href="#outer" style="fill:none"/>
</svg>"""


def _circ(a, b):
    d = abs(a - b) % (math.pi / 2)
    return min(d, math.pi / 2 - d)


def test_parse_small():
    a = parse_svg_text(SVG)
    assert a.shape == (4, 3)
    a = a[np.lexsort((a[:, 1], a[:, 0]))]
    exp = [(0.5, 2.0, 0.0), (0.5, 3.0, 0.0), (1.5, 3.0, 0.0), (2.0, 3.5 - (1 + math.sqrt(0.5)), math.pi / 4)]
    for r, e in zip(a, exp):
        assert abs(r[0] - e[0]) < 1e-12 and abs(r[1] - e[1]) < 1e-12 and _circ(r[2], e[2]) < 1e-12


def test_parse_strip():
    svg = SVG.replace('<use xlink:href="#one" transform="translate(2 1) rotate(45)"/>', '<rect width="2" height="1"/>')
    assert len(parse_svg_text(svg)) == 5
