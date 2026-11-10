import math

from sqin import store
from sqin.strip import strip, strip_n


def test_strip_counts_and_side():
    for k in (2, 6, 7, 12):
        a = strip(k)
        assert a.shape == (strip_n(k), 3)
        assert abs(store.side_length(store.normalize(a)) - (k + math.sqrt(2) / 2)) < 1e-9


def test_strip_family_values():
    ks = (2, 3, 5, 6, 7, 12, 17, 19, 24, 26)
    assert [strip_n(k) for k in ks] == [5, 10, 27, 38, 52, 149, 296, 369, 586, 686]
