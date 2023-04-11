import pytest
from hypothesis import given
from hypothesis.extra.array_api import make_strategies_namespace

from numpy import array_api as xp

xps = make_strategies_namespace(xp)


@pytest.mark.parametrize("func", [xp.unique_all, xp.unique_inverse])
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=xps.array_shapes()))
def test_inverse_indices_shape(func, x):
    """
    Inverse indices share shape of input array

    See https://github.com/numpy/numpy/issues/20638
    """
    out = func(x)
    assert out.inverse_indices.shape == x.shape
