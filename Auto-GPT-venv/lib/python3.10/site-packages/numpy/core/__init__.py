"""
Contains the core of NumPy: ndarray, ufuncs, dtypes, etc.

Please note that this module is private.  All functions and objects
are available in the main ``numpy`` namespace - use that instead.

"""

from numpy.version import version as __version__

import os
import warnings

# disables OpenBLAS affinity setting of the main thread that limits
# python threads or processes to one core
env_added = []
for envkey in ['OPENBLAS_MAIN_FREE', 'GOTOBLAS_MAIN_FREE']:
    if envkey not in os.environ:
        os.environ[envkey] = '1'
        env_added.append(envkey)

try:
    from . import multiarray
except ImportError as exc:
    import sys
    msg = """

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python%d.%d from "%s"
  * The NumPy version is: "%s"

and make sure that they are the versions you expect.
Please carefully study the documentation linked above for further help.

Original error was: %s
""" % (sys.version_info[0], sys.version_info[1], sys.executable,
        __version__, exc)
    raise ImportError(msg)
finally:
    for envkey in env_added:
        del os.environ[envkey]
del envkey
del env_added
del os

from . import umath

# Check that multiarray,umath are pure python modules wrapping
# _multiarray_umath and not either of the old c-extension modules
if not (hasattr(multiarray, '_multiarray_umath') and
        hasattr(umath, '_multiarray_umath')):
    import sys
    path = sys.modules['numpy'].__path__
    msg = ("Something is wrong with the numpy installation. "
        "While importing we detected an older version of "
        "numpy in {}. One method of fixing this is to repeatedly uninstall "
        "numpy until none is found, then reinstall this version.")
    raise ImportError(msg.format(path))

from . import numerictypes as nt
multiarray.set_typeDict(nt.sctypeDict)
from . import numeric
from .numeric import *
from . import fromnumeric
from .fromnumeric import *
from . import defchararray as char
from . import records
from . import records as rec
from .records import record, recarray, format_parser
# Note: module name memmap is overwritten by a class with same name
from .memmap import *
from .defchararray import chararray
from . import function_base
from .function_base import *
from . import _machar
from ._machar import *
from . import getlimits
from .getlimits import *
from . import shape_base
from .shape_base import *
from . import einsumfunc
from .einsumfunc import *
del nt

from .fromnumeric import amax as max, amin as min, round_ as round
from .numeric import absolute as abs

# do this after everything else, to minimize the chance of this misleadingly
# appearing in an import-time traceback
from . import _add_newdocs
from . import _add_newdocs_scalars
# add these for module-freeze analysis (like PyInstaller)
from . import _dtype_ctypes
from . import _internal
from . import _dtype
from . import _methods

__all__ = ['char', 'rec', 'memmap']
__all__ += numeric.__all__
__all__ += ['record', 'recarray', 'format_parser']
__all__ += ['chararray']
__all__ += function_base.__all__
__all__ += getlimits.__all__
__all__ += shape_base.__all__
__all__ += einsumfunc.__all__

# We used to use `np.core._ufunc_reconstruct` to unpickle. This is unnecessary,
# but old pickles saved before 1.20 will be using it, and there is no reason
# to break loading them.
def _ufunc_reconstruct(module, name):
    # The `fromlist` kwarg is required to ensure that `mod` points to the
    # inner-most module rather than the parent package when module name is
    # nested. This makes it possible to pickle non-toplevel ufuncs such as
    # scipy.special.expit for instance.
    mod = __import__(module, fromlist=[name])
    return getattr(mod, name)


def _ufunc_reduce(func):
    # Report the `__name__`. pickle will try to find the module. Note that
    # pickle supports for this `__name__` to be a `__qualname__`. It may
    # make sense to add a `__qualname__` to ufuncs, to allow this more
    # explicitly (Numba has ufuncs as attributes).
    # See also: https://github.com/dask/distributed/issues/3450
    return func.__name__


def _DType_reconstruct(scalar_type):
    # This is a work-around to pickle type(np.dtype(np.float64)), etc.
    # and it should eventually be replaced with a better solution, e.g. when
    # DTypes become HeapTypes.
    return type(dtype(scalar_type))


def _DType_reduce(DType):
    # To pickle a DType without having to add top-level names, pickle the
    # scalar type for now (and assume that reconstruction will be possible).
    if DType is dtype:
        return "dtype"  # must pickle `np.dtype` as a singleton.
    scalar_type = DType.type  # pickle the scalar type for reconstruction
    return _DType_reconstruct, (scalar_type,)


def __getattr__(name):
    # Deprecated 2021-10-20, NumPy 1.22
    if name == "machar":
        warnings.warn(
            "The `np.core.machar` module is deprecated (NumPy 1.22)",
            DeprecationWarning, stacklevel=2,
        )
        return _machar
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}")


import copyreg

copyreg.pickle(ufunc, _ufunc_reduce)
copyreg.pickle(type(dtype), _DType_reduce, _DType_reconstruct)

# Unclutter namespace (must keep _*_reconstruct for unpickling)
del copyreg
del _ufunc_reduce
del _DType_reduce

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
