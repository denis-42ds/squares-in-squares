import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sqin.vendor import metric as M

SAMPLE = Path("data/submissions/sample_submission.csv")
if not SAMPLE.exists():
    pytest.skip("local Kaggle sample file is not available", allow_module_level=True)


def check(n, pts):
    rows = [{"n": str(n), "square_id": str(i), "x": repr(x), "y": repr(y), "angle": repr(a)}
            for i, (x, y, a) in enumerate(pts)]
    return M.evaluate_single_puzzle(n, rows)

d = 1e-12
print("T1 dyadic touch   ", check(2, [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0)]))
print("T2 0.1 touch      ", check(2, [(0.1, 0.5, 0.0), (1.1, 0.5, 0.0)]))
print("T3 0.1 + gap 1e-12", check(2, [(0.1, 0.5, 0.0), (1.1 + d, 0.5, 0.0)]))

L = 2 + 1 / math.sqrt(2) + 3e-12
h = 0.5
pts5 = [(h, h, 0.0), (L - h, h, 0.0), (h, L - h, 0.0), (L - h, L - h, 0.0),
        (L / 2, L / 2, math.pi / 4)]
print("T4 n=5            ", check(5, pts5))

s = pd.read_csv("data/submissions/sample_submission.csv", dtype=str)
m = s["n"] == "5"
s.loc[m, ["x", "y", "angle"]] = np.array([[repr(x), repr(y), repr(a)] for x, y, a in pts5], dtype=object)
s.to_csv("data/submissions/sub_n5.csv", index=False)
