import numpy as np

from sqin.construct import grow
from sqin.validate import fast_check


def test_grow_single_tilted_square():
    arr = grow(np.array([[0.0, 0.0, 0.3]]))
    assert arr.shape == (4, 3)
    assert fast_check(arr)
