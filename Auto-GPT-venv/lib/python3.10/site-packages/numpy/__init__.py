"""
NumPy
=====

Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the NumPy homepage <https://numpy.org>`_.

We recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as ``np``::

  >>> import numpy as np

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)
  ... # doctest: +SKIP

For some objects, ``np.info(obj)`` may provide additional help.  This is
particularly true if you see the line "Help on ufunc object:" at the top
of the help() page.  Ufuncs are implemented in C, not Python, for speed.
The native Python help() does not know how to view their help, but our
np.info() function does.

To search for documents containing a keyword, do::

  >>> np.lookfor('keyword')
  ... # doctest: +SKIP

General-purpose documents like a glossary and help on the basic concepts
of numpy are available under the ``doc`` sub-module::

  >>> from numpy import doc
  >>> help(doc)
  ... # doctest: +SKIP

Available subpackages
---------------------
lib
    Basic functions used by several sub-packages.
random
    Core Random Tools
linalg
    Core Linear Algebra Tools
fft
    Core FFT routines
polynomial
    Polynomial tools
testing
    NumPy testing tools
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more.

Utilities
---------
test
    Run numpy unittests
show_config
    Show numpy build configuration
dual
    Overwrite certain functions with high-performance SciPy tools.
    Note: `numpy.dual` is deprecated.  Use the functions from NumPy or Scipy
    directly instead of importing them from `numpy.dual`.
matlib
    Make everything matrices.
__version__
    NumPy version string

Viewing documentation using IPython
-----------------------------------

Start IPython and import `numpy` usually under the alias ``np``: `import
numpy as np`.  Then, directly past or use the ``%cpaste`` magic to paste
examples into the shell.  To see which functions are available in `numpy`,
type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
down the list.  To view the docstring for a function, use
``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
the source code).

Copies vs. in-place operation
-----------------------------
Most of the functions in `numpy` return a copy of the array argument
(e.g., `np.sort`).  In-place versions of these functions are often
available as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.
Exceptions to this rule are documented.

"""
import sys
import warnings

from ._globals import (
    ModuleDeprecationWarning, VisibleDeprecationWarning,
    _NoValue, _CopyMode
)

# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
try:
    __NUMPY_SETUP__
except NameError:
    __NUMPY_SETUP__ = False

if __NUMPY_SETUP__:
    sys.stderr.write('Running from numpy source directory.\n')
else:
    try:
        from numpy.__config__ import show as show_config
    except ImportError as e:
        msg = """Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there."""
        raise ImportError(msg) from e

    __all__ = ['ModuleDeprecationWarning',
               'VisibleDeprecationWarning']

    # mapping of {name: (value, deprecation_msg)}
    __deprecated_attrs__ = {}

    # Allow distributors to run custom init code
    from . import _distributor_init

    from . import core
    from .core import *
    from . import compat
    from . import lib
    # NOTE: to be revisited following future namespace cleanup.
    # See gh-14454 and gh-15672 for discussion.
    from .lib import *

    from . import linalg
    from . import fft
    from . import polynomial
    from . import random
    from . import ctypeslib
    from . import ma
    from . import matrixlib as _mat
    from .matrixlib import *

    # Deprecations introduced in NumPy 1.20.0, 2020-06-06
    import builtins as _builtins

    _msg = (
        "module 'numpy' has no attribute '{n}'.\n"
        "`np.{n}` was a deprecated alias for the builtin `{n}`. "
        "To avoid this error in existing code, use `{n}` by itself. "
        "Doing this will not modify any behavior and is safe. {extended_msg}\n"
        "The aliases was originally deprecated in NumPy 1.20; for more "
        "details and guidance see the original release note at:\n"
        "    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations")

    _specific_msg = (
        "If you specifically wanted the numpy scalar type, use `np.{}` here.")

    _int_extended_msg = (
        "When replacing `np.{}`, you may wish to use e.g. `np.int64` "
        "or `np.int32` to specify the precision. If you wish to review "
        "your current use, check the release note link for "
        "additional information.")

    _type_info = [
        ("object", ""),  # The NumPy scalar only exists by name.
        ("bool", _specific_msg.format("bool_")),
        ("float", _specific_msg.format("float64")),
        ("complex", _specific_msg.format("complex128")),
        ("str", _specific_msg.format("str_")),
        ("int", _int_extended_msg.format("int"))]

    __former_attrs__ = {
         n: _msg.format(n=n, extended_msg=extended_msg)
         for n, extended_msg in _type_info
     }

    # Future warning introduced in NumPy 1.24.0, 2022-11-17
    _msg = (
        "`np.{n}` is a deprecated alias for `{an}`.  (Deprecated NumPy 1.24)")

    # Some of these are awkward (since `np.str` may be preferable in the long
    # term), but overall the names ending in 0 seem undesireable
    _type_info = [
        ("bool8", bool_, "np.bool_"),
        ("int0", intp, "np.intp"),
        ("uint0", uintp, "np.uintp"),
        ("str0", str_, "np.str_"),
        ("bytes0", bytes_, "np.bytes_"),
        ("void0", void, "np.void"),
        ("object0", object_,
            "`np.object0` is a deprecated alias for `np.object_`. "
            "`object` can be used instead.  (Deprecated NumPy 1.24)")]

    # Some of these could be defined right away, but most were aliases to
    # the Python objects and only removed in NumPy 1.24.  Defining them should
    # probably wait for NumPy 1.26 or 2.0.
    # When defined, these should possibly not be added to `__all__` to avoid
    # import with `from numpy import *`.
    __future_scalars__ = {"bool", "long", "ulong", "str", "bytes", "object"}

    __deprecated_attrs__.update({
        n: (alias, _msg.format(n=n, an=an)) for n, alias, an in _type_info})

    del _msg, _type_info

    from .core import round, abs, max, min
    # now that numpy modules are imported, can initialize limits
    core.getlimits._register_known_types()

    __all__.extend(['__version__', 'show_config'])
    __all__.extend(core.__all__)
    __all__.extend(_mat.__all__)
    __all__.extend(lib.__all__)
    __all__.extend(['linalg', 'fft', 'random', 'ctypeslib', 'ma'])

    # Remove one of the two occurrences of `issubdtype`, which is exposed as
    # both `numpy.core.issubdtype` and `numpy.lib.issubdtype`.
    __all__.remove('issubdtype')

    # These are exported by np.core, but are replaced by the builtins below
    # remove them to ensure that we don't end up with `np.long == np.int_`,
    # which would be a breaking change.
    del long, unicode
    __all__.remove('long')
    __all__.remove('unicode')

    # Remove things that are in the numpy.lib but not in the numpy namespace
    # Note that there is a test (numpy/tests/test_public_api.py:test_numpy_namespace)
    # that prevents adding more things to the main namespace by accident.
    # The list below will grow until the `from .lib import *` fixme above is
    # taken care of
    __all__.remove('Arrayterator')
    del Arrayterator

    # These names were removed in NumPy 1.20.  For at least one release,
    # attempts to access these names in the numpy namespace will trigger
    # a warning, and calling the function will raise an exception.
    _financial_names = ['fv', 'ipmt', 'irr', 'mirr', 'nper', 'npv', 'pmt',
                        'ppmt', 'pv', 'rate']
    __expired_functions__ = {
        name: (f'In accordance with NEP 32, the function {name} was removed '
               'from NumPy version 1.20.  A replacement for this function '
               'is available in the numpy_financial library: '
               'https://pypi.org/project/numpy-financial')
        for name in _financial_names}

    # Filter out Cython harmless warnings
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    # oldnumeric and numarray were removed in 1.9. In case some packages import
    # but do not use them, we define them here for backward compatibility.
    oldnumeric = 'removed'
    numarray = 'removed'

    def __getattr__(attr):
        # Warn for expired attributes, and return a dummy function
        # that always raises an exception.
        import warnings
        try:
            msg = __expired_functions__[attr]
        except KeyError:
            pass
        else:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

            def _expired(*args, **kwds):
                raise RuntimeError(msg)

            return _expired

        # Emit warnings for deprecated attributes
        try:
            val, msg = __deprecated_attrs__[attr]
        except KeyError:
            pass
        else:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return val

        if attr in __future_scalars__:
            # And future warnings for those that will change, but also give
            # the AttributeError
            warnings.warn(
                f"In the future `np.{attr}` will be defined as the "
                "corresponding NumPy scalar.", FutureWarning, stacklevel=2)

        if attr in __former_attrs__:
            raise AttributeError(__former_attrs__[attr])

        # Importing Tester requires importing all of UnitTest which is not a
        # cheap import Since it is mainly used in test suits, we lazy import it
        # here to save on the order of 10 ms of import time for most users
        #
        # The previous way Tester was imported also had a side effect of adding
        # the full `numpy.testing` namespace
        if attr == 'testing':
            import numpy.testing as testing
            return testing
        elif attr == 'Tester':
            from .testing import Tester
            return Tester

        raise AttributeError("module {!r} has no attribute "
                             "{!r}".format(__name__, attr))

    def __dir__():
        public_symbols = globals().keys() | {'Tester', 'testing'}
        public_symbols -= {
            "core", "matrixlib",
        }
        return list(public_symbols)

    # Pytest testing
    from numpy._pytesttester import PytestTester
    test = PytestTester(__name__)
    del PytestTester

    def _sanity_check():
        """
        Quick sanity checks for common bugs caused by environment.
        There are some cases e.g. with wrong BLAS ABI that cause wrong
        results under specific runtime conditions that are not necessarily
        achieved during test suite runs, and it is useful to catch those early.

        See https://github.com/numpy/numpy/issues/8577 and other
        similar bug reports.

        """
        try:
            x = ones(2, dtype=float32)
            if not abs(x.dot(x) - float32(2.0)) < 1e-5:
                raise AssertionError()
        except AssertionError:
            msg = ("The current Numpy installation ({!r}) fails to "
                   "pass simple sanity checks. This can be caused for example "
                   "by incorrect BLAS library being linked in, or by mixing "
                   "package managers (pip, conda, apt, ...). Search closed "
                   "numpy issues for similar problems.")
            raise RuntimeError(msg.format(__file__)) from None

    _sanity_check()
    del _sanity_check

    def _mac_os_check():
        """
        Quick Sanity check for Mac OS look for accelerate build bugs.
        Testing numpy polyfit calls init_dgelsd(LAPACK)
        """
        try:
            c = array([3., 2., 1.])
            x = linspace(0, 2, 5)
            y = polyval(c, x)
            _ = polyfit(x, y, 2, cov=True)
        except ValueError:
            pass

    if sys.platform == "darwin":
        with warnings.catch_warnings(record=True) as w:
            _mac_os_check()
            # Throw runtime error, if the test failed Check for warning and error_message
            error_message = ""
            if len(w) > 0:
                error_message = "{}: {}".format(w[-1].category.__name__, str(w[-1].message))
                msg = (
                    "Polyfit sanity test emitted a warning, most likely due "
                    "to using a buggy Accelerate backend."
                    "\nIf you compiled yourself, more information is available at:"
                    "\nhttps://numpy.org/doc/stable/user/building.html#accelerated-blas-lapack-libraries"
                    "\nOtherwise report this to the vendor "
                    "that provided NumPy.\n{}\n".format(error_message))
                raise RuntimeError(msg)
    del _mac_os_check

    # We usually use madvise hugepages support, but on some old kernels it
    # is slow and thus better avoided.
    # Specifically kernel version 4.6 had a bug fix which probably fixed this:
    # https://github.com/torvalds/linux/commit/7cf91a98e607c2f935dbcc177d70011e95b8faff
    import os
    use_hugepage = os.environ.get("NUMPY_MADVISE_HUGEPAGE", None)
    if sys.platform == "linux" and use_hugepage is None:
        # If there is an issue with parsing the kernel version,
        # set use_hugepages to 0. Usage of LooseVersion will handle
        # the kernel version parsing better, but avoided since it
        # will increase the import time. See: #16679 for related discussion.
        try:
            use_hugepage = 1
            kernel_version = os.uname().release.split(".")[:2]
            kernel_version = tuple(int(v) for v in kernel_version)
            if kernel_version < (4, 6):
                use_hugepage = 0
        except ValueError:
            use_hugepages = 0
    elif use_hugepage is None:
        # This is not Linux, so it should not matter, just enable anyway
        use_hugepage = 1
    else:
        use_hugepage = int(use_hugepage)

    # Note that this will currently only make a difference on Linux
    core.multiarray._set_madvise_hugepage(use_hugepage)

    # Give a warning if NumPy is reloaded or imported on a sub-interpreter
    # We do this from python, since the C-module may not be reloaded and
    # it is tidier organized.
    core.multiarray._multiarray_umath._reload_guard()

    core._set_promotion_state(os.environ.get("NPY_PROMOTION_STATE", "legacy"))

    # Tell PyInstaller where to find hook-numpy.py
    def _pyinstaller_hooks_dir():
        from pathlib import Path
        return [str(Path(__file__).with_name("_pyinstaller").resolve())]

    # Remove symbols imported for internal use
    del os


# get the version using versioneer
from .version import __version__, git_revision as __git_version__

# Remove symbols imported for internal use
del sys, warnings
