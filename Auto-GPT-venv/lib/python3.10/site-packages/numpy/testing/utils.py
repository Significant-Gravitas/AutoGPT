"""
Back compatibility utils module. It will import the appropriate
set of tools

"""
import warnings

# 2018-04-04, numpy 1.15.0 ImportWarning
# 2019-09-18, numpy 1.18.0 DeprecatonWarning (changed)
warnings.warn("Importing from numpy.testing.utils is deprecated "
              "since 1.15.0, import from numpy.testing instead.",
              DeprecationWarning, stacklevel=2)

from ._private.utils import *
from ._private.utils import _assert_valid_refcount, _gen_alignment_data

__all__ = [
        'assert_equal', 'assert_almost_equal', 'assert_approx_equal',
        'assert_array_equal', 'assert_array_less', 'assert_string_equal',
        'assert_array_almost_equal', 'assert_raises', 'build_err_msg',
        'decorate_methods', 'jiffies', 'memusage', 'print_assert_equal',
        'raises', 'rundocs', 'runstring', 'verbose', 'measure',
        'assert_', 'assert_array_almost_equal_nulp', 'assert_raises_regex',
        'assert_array_max_ulp', 'assert_warns', 'assert_no_warnings',
        'assert_allclose', 'IgnoreException', 'clear_and_catch_warnings',
        'SkipTest', 'KnownFailureException', 'temppath', 'tempdir', 'IS_PYPY',
        'HAS_REFCOUNT', 'suppress_warnings', 'assert_array_compare',
        'assert_no_gc_cycles'
        ]
