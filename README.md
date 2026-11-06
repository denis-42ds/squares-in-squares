# Squares in Squares

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Numba](https://img.shields.io/badge/Numba-00A3E0?logo=numba&logoColor=white)](https://numba.pydata.org/)
[![mpmath](https://img.shields.io/badge/mpmath-exact%20arithmetic-4B8BBE)](https://mpmath.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Squares%20in%20Squares-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/squares-in-squares)

Solver toolkit for the Kaggle competition
**[Squares in Squares](https://www.kaggle.com/competitions/squares-in-squares)**:
pack `n` unit squares (`n = 1..1000`) into the smallest possible axis-aligned square container.

> Work in progress. The goal is a competitive, fully reproducible solution.

## The problem

For every puzzle `n` in `1..1000` a submission provides the center `(x, y)` and rotation
angle (radians) of each of the `n` unit squares. A puzzle is valid if no two squares overlap.
The container side `L` is not submitted: it is computed as the side of the tightest
axis-aligned bounding square around all squares.

```
score(n) = L / sqrt(n) - 1          (lower is better)
total    = sum of score(n) over n = 1..1000
```

Partial submissions are allowed; a missing or invalid puzzle is charged the trivial grid
packing `ceil(sqrt(n)) / sqrt(n) - 1`. The trivial grid baseline totals about `28.9377`.

Submission format:

```
id,n,square_id,x,y,angle
0,1,0,0.5,0.5,0
1,2,0,0.5,0.5,0
2,2,1,1.5,0.5,0
```

## Approach

- **Strict validity first.** Every candidate passes a fast float/numba separating-axis (SAT)
  prefilter with a safety gap (`SEP_EPS = 1e-12`), then the official metric code
  (`evaluate_single_puzzle`) before it can enter the store.
- **Per-puzzle storage.** `best[n]` is updated only on strict improvement of `L`.
- **Monotone push-down.** A packing for `n + 1` with one square removed is a candidate for `n`.
- **Optimization layers**: overlap penalty with container shrinking (L-BFGS),
  simulated annealing / basin hopping over positions and angles, final polishing to exact
  feasibility, and structured constructions (grid plus optimized defect zone) for large `n`.
- **Reproducibility.** Deterministic seeds, documented origin of every starting solution.

## Repository layout

```
squares-in-squares/
├── pyproject.toml
├── STATE.md                 # project status, updated after every working session
├── docs/
│   └── metric_notes.md      # validator behavior, tolerances, submission format
├── src/sqin/
│   ├── vendor/metric.py     # official metric code, unmodified
│   ├── io.py                # read/write submissions
│   ├── geometry.py          # corners, side length, numba SAT prefilter
│   ├── validate.py          # fast check, polishing, gate via the official metric
│   ├── store.py             # best[n], strict-improvement updates, push-down
│   ├── grid.py              # trivial grid baseline
│   └── opt/                 # optimizers (penalty, annealing, polishing)
├── scripts/                 # thin launchers
├── tests/                   # metric assumption tests and unit tests
├── data/                    # best solutions, external data, submissions (git-ignored)
└── runs/                    # logs (git-ignored)
```

## Getting started

```bash
git clone https://github.com/denis-42ds/squares-in-squares.git
cd squares-in-squares
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Command-line interface: `baseline`, `verify FILE`, `ingest FILE`, `build`.

## Data and external sources

The competition provides no data. External solutions are used only if publicly available to
all participants; each is stored under `data/external/` with its source and license.
Large files (`data/`, `runs/`, `*.csv`) are not tracked by git.

## Competition timeline

| Date | Event |
|---|---|
| 2026-10-21 | Start |
| 2027-01-17 | 3-month milestone cut-off |
| 2027-04-18 | 6-month milestone cut-off |
| 2027-07-11 | Entry and team merger deadline |
| 2027-07-18 | Final submission deadline |

## License

Apache 2.0

## Acknowledgements

- [Kaggle](https://www.kaggle.com/) and Google for hosting and sponsoring the competition.
- Erich Friedman's [Packing Center](https://erich-friedman.github.io/packing/) for decades of
  documented best-known packings.
