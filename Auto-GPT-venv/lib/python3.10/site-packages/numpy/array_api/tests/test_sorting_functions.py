import pytest

from numpy import array_api as xp


@pytest.mark.parametrize(
    "obj, axis, expected",
    [
        ([0, 0], -1, [0, 1]),
        ([0, 1, 0], -1, [1, 0, 2]),
        ([[0, 1], [1, 1]], 0, [[1, 0], [0, 1]]),
        ([[0, 1], [1, 1]], 1, [[1, 0], [0, 1]]),
    ],
)
def test_stable_desc_argsort(obj, axis, expected):
    """
    Indices respect relative order of a descending stable-sort

    See https://github.com/numpy/numpy/issues/20778
    """
    x = xp.asarray(obj)
    out = xp.argsort(x, axis=axis, stable=True, descending=True)
    assert xp.all(out == xp.asarray(expected))
