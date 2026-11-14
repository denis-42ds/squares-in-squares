from sqin import store
from sqin.fam8467 import S1, member


def test_member_counts_and_side():
    for k, n, r in ((7, 54, 16), (10, 107, 24), (13, 178, 32), (16, 267, 40)):
        a = member(k)
        assert a.shape == (n, 3)
        assert int((a[:, 2] > 1e-9).sum()) == r
        assert abs(float(store.side_length(store.normalize(a))) - (S1 + k - 7)) < 1e-9
