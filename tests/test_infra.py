import math
from decimal import Decimal

import numpy as np
import pytest

from sqin import cli, store
from sqin.geometry import normalize, side_length, side_without_each
from sqin.grid import grid, grid_total
from sqin.io import read_submission, write_submission
from sqin.validate import fast_check, gate, polish


def n5(d):
    s2 = 1.0 / math.sqrt(2.0)
    lo = 0.5
    hi = 1.5 + s2 + d
    c = 0.5 * (lo + hi)
    return np.array(
        [[lo, lo, 0.0], [hi, lo, 0.0], [lo, hi, 0.0], [hi, hi, 0.0], [c, c, math.pi / 4]]
    )


def test_grid_total():
    assert grid_total() == pytest.approx(28.937669916550586, rel=1e-12)


def test_gate_grid():
    for n in (1, 2, 5, 10, 17):
        ok, L, reason = gate(n, grid(n))
        assert ok, reason
        assert Decimal(L) == math.isqrt(n - 1) + 1


def test_fast_check_grid_exempt():
    assert fast_check(grid(7))
    a = grid(7).copy()
    a[1, 0] -= 1e-9
    assert not fast_check(a)


def test_n5_gap_and_gate():
    a = n5(3e-12)
    assert fast_check(a)
    ok, L, reason = gate(5, a)
    assert ok, reason
    assert abs(float(Decimal(L)) - 2.7071067811895473) < 1e-12


def test_polish():
    a = n5(2e-13)
    assert not fast_check(a)
    b = polish(a)
    assert b is not None and fast_check(b)
    ok, _, reason = gate(5, b)
    assert ok, reason


def test_side_without_each():
    rng = np.random.default_rng(0)
    a = np.column_stack([rng.uniform(0, 5, 12), rng.uniform(0, 5, 12), rng.uniform(0, 1.5, 12)])
    brute = np.array([side_length(np.delete(a, i, axis=0)) for i in range(12)])
    assert np.allclose(side_without_each(a), brute, atol=1e-12)


def test_roundtrip(tmp_path):
    a = n5(3e-12)
    f = tmp_path / "s.csv"
    write_submission(f, {5: a})
    sol = read_submission(f)
    assert len(sol) == 1000
    assert np.array_equal(sol[5], normalize(a))
    assert np.array_equal(sol[7], grid(7))
    lines = f.read_text().splitlines()
    assert lines[0] == "id,n,square_id,x,y,angle"
    assert lines[1] == "0,1,0,0.5,0.5,0"
    assert lines[11].startswith("10,5,0,")


def test_ingest_build(tmp_path, monkeypatch):
    monkeypatch.setenv("SQIN_ROOT", str(tmp_path))
    a = n5(3e-12)
    f = tmp_path / "s.csv"
    write_submission(f, {5: a})
    assert cli.main(["ingest", str(f), "--source", "test", "--procs", "2"]) == 0
    L = store.best_L(5)
    assert abs(float(L) - 2.7071067811895473) < 1e-12
    total, _ = store.total_score()
    assert total == pytest.approx(grid_total() - (3.0 - float(L)) / math.sqrt(5.0), abs=1e-9)
    ok, r = store.try_update(5, a, "again")
    assert not ok and r[1] == "not_better"
    out = tmp_path / "b.csv"
    assert cli.main(["build", "--out", str(out)]) == 0
    sol = read_submission(out)
    assert len(sol) == 1000 and np.array_equal(sol[5], normalize(a))


def test_push_down(tmp_path, monkeypatch):
    monkeypatch.setenv("SQIN_ROOT", str(tmp_path))
    a = n5(3e-12)
    arr, lf = store.push_down_candidate(a)
    assert len(arr) == 4 and fast_check(arr) and lf == pytest.approx(side_length(arr))
    store.try_update(5, a, "t")
    assert store.push_down(start=5, verbose=False) == []
    assert store.best_L(4) == 2
