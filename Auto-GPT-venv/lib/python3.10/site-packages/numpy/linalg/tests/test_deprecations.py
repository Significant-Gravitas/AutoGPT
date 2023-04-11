"""Test deprecation and future warnings.

"""
import numpy as np
from numpy.testing import assert_warns


def test_qr_mode_full_future_warning():
    """Check mode='full' FutureWarning.

    In numpy 1.8 the mode options 'full' and 'economic' in linalg.qr were
    deprecated. The release date will probably be sometime in the summer
    of 2013.

    """
    a = np.eye(2)
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='full')
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='f')
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='economic')
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='e')
