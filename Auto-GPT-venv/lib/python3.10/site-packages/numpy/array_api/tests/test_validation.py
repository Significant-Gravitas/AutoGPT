from typing import Callable

import pytest

from numpy import array_api as xp


def p(func: Callable, *args, **kwargs):
    f_sig = ", ".join(
        [str(a) for a in args] + [f"{k}={v}" for k, v in kwargs.items()]
    )
    id_ = f"{func.__name__}({f_sig})"
    return pytest.param(func, args, kwargs, id=id_)


@pytest.mark.parametrize(
    "func, args, kwargs",
    [
        p(xp.can_cast, 42, xp.int8),
        p(xp.can_cast, xp.int8, 42),
        p(xp.result_type, 42),
    ],
)
def test_raises_on_invalid_types(func, args, kwargs):
    """Function raises TypeError when passed invalidly-typed inputs"""
    with pytest.raises(TypeError):
        func(*args, **kwargs)
