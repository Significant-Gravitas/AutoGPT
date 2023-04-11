import sys
import pytest
import numpy as np


def test_financial_expired():
    match = 'NEP 32'
    with pytest.warns(DeprecationWarning, match=match):
        func = np.fv
    with pytest.raises(RuntimeError, match=match):
        func(1, 2, 3)
