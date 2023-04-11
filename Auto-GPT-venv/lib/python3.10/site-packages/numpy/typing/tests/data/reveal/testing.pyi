from __future__ import annotations

import re
import sys
from collections.abc import Callable
from typing import Any, TypeVar
from pathlib import Path

import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]
AR_i8: npt.NDArray[np.int64]

bool_obj: bool
suppress_obj: np.testing.suppress_warnings
FT = TypeVar("FT", bound=Callable[..., Any])

def func() -> int: ...

def func2(
    x: npt.NDArray[np.number[Any]],
    y: npt.NDArray[np.number[Any]],
) -> npt.NDArray[np.bool_]: ...

reveal_type(np.testing.KnownFailureException())  # E: KnownFailureException
reveal_type(np.testing.IgnoreException())  # E: IgnoreException

reveal_type(np.testing.clear_and_catch_warnings(modules=[np.testing]))  # E: _clear_and_catch_warnings_without_records
reveal_type(np.testing.clear_and_catch_warnings(True))  # E: _clear_and_catch_warnings_with_records
reveal_type(np.testing.clear_and_catch_warnings(False))  # E: _clear_and_catch_warnings_without_records
reveal_type(np.testing.clear_and_catch_warnings(bool_obj))  # E: clear_and_catch_warnings
reveal_type(np.testing.clear_and_catch_warnings.class_modules)  # E: tuple[types.ModuleType, ...]
reveal_type(np.testing.clear_and_catch_warnings.modules)  # E: set[types.ModuleType]

with np.testing.clear_and_catch_warnings(True) as c1:
    reveal_type(c1)  # E: builtins.list[warnings.WarningMessage]
with np.testing.clear_and_catch_warnings() as c2:
    reveal_type(c2)  # E: None

reveal_type(np.testing.suppress_warnings("once"))  # E: suppress_warnings
reveal_type(np.testing.suppress_warnings()(func))  # E: def () -> builtins.int
reveal_type(suppress_obj.filter(RuntimeWarning))  # E: None
reveal_type(suppress_obj.record(RuntimeWarning))  # E: list[warnings.WarningMessage]
with suppress_obj as c3:
    reveal_type(c3)  # E: suppress_warnings

reveal_type(np.testing.verbose)  # E: int
reveal_type(np.testing.IS_PYPY)  # E: bool
reveal_type(np.testing.HAS_REFCOUNT)  # E: bool
reveal_type(np.testing.HAS_LAPACK64)  # E: bool

reveal_type(np.testing.assert_(1, msg="test"))  # E: None
reveal_type(np.testing.assert_(2, msg=lambda: "test"))  # E: None

if sys.platform == "win32" or sys.platform == "cygwin":
    reveal_type(np.testing.memusage())  # E: builtins.int
elif sys.platform == "linux":
    reveal_type(np.testing.memusage())  # E: Union[None, builtins.int]
else:
    reveal_type(np.testing.memusage())  # E: <nothing>

reveal_type(np.testing.jiffies())  # E: builtins.int

reveal_type(np.testing.build_err_msg([0, 1, 2], "test"))  # E: str
reveal_type(np.testing.build_err_msg(range(2), "test", header="header"))  # E: str
reveal_type(np.testing.build_err_msg(np.arange(9).reshape(3, 3), "test", verbose=False))  # E: str
reveal_type(np.testing.build_err_msg("abc", "test", names=["x", "y"]))  # E: str
reveal_type(np.testing.build_err_msg([1.0, 2.0], "test", precision=5))  # E: str

reveal_type(np.testing.assert_equal({1}, {1}))  # E: None
reveal_type(np.testing.assert_equal([1, 2, 3], [1, 2, 3], err_msg="fail"))  # E: None
reveal_type(np.testing.assert_equal(1, 1.0, verbose=True))  # E: None

reveal_type(np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 1]))  # E: None

reveal_type(np.testing.assert_almost_equal(1.0, 1.1))  # E: None
reveal_type(np.testing.assert_almost_equal([1, 2, 3], [1, 2, 3], err_msg="fail"))  # E: None
reveal_type(np.testing.assert_almost_equal(1, 1.0, verbose=True))  # E: None
reveal_type(np.testing.assert_almost_equal(1, 1.0001, decimal=2))  # E: None

reveal_type(np.testing.assert_approx_equal(1.0, 1.1))  # E: None
reveal_type(np.testing.assert_approx_equal("1", "2", err_msg="fail"))  # E: None
reveal_type(np.testing.assert_approx_equal(1, 1.0, verbose=True))  # E: None
reveal_type(np.testing.assert_approx_equal(1, 1.0001, significant=2))  # E: None

reveal_type(np.testing.assert_array_compare(func2, AR_i8, AR_f8, err_msg="test"))  # E: None
reveal_type(np.testing.assert_array_compare(func2, AR_i8, AR_f8, verbose=True))  # E: None
reveal_type(np.testing.assert_array_compare(func2, AR_i8, AR_f8, header="header"))  # E: None
reveal_type(np.testing.assert_array_compare(func2, AR_i8, AR_f8, precision=np.int64()))  # E: None
reveal_type(np.testing.assert_array_compare(func2, AR_i8, AR_f8, equal_nan=False))  # E: None
reveal_type(np.testing.assert_array_compare(func2, AR_i8, AR_f8, equal_inf=True))  # E: None

reveal_type(np.testing.assert_array_equal(AR_i8, AR_f8))  # E: None
reveal_type(np.testing.assert_array_equal(AR_i8, AR_f8, err_msg="test"))  # E: None
reveal_type(np.testing.assert_array_equal(AR_i8, AR_f8, verbose=True))  # E: None

reveal_type(np.testing.assert_array_almost_equal(AR_i8, AR_f8))  # E: None
reveal_type(np.testing.assert_array_almost_equal(AR_i8, AR_f8, err_msg="test"))  # E: None
reveal_type(np.testing.assert_array_almost_equal(AR_i8, AR_f8, verbose=True))  # E: None
reveal_type(np.testing.assert_array_almost_equal(AR_i8, AR_f8, decimal=1))  # E: None

reveal_type(np.testing.assert_array_less(AR_i8, AR_f8))  # E: None
reveal_type(np.testing.assert_array_less(AR_i8, AR_f8, err_msg="test"))  # E: None
reveal_type(np.testing.assert_array_less(AR_i8, AR_f8, verbose=True))  # E: None

reveal_type(np.testing.runstring("1 + 1", {}))  # E: Any
reveal_type(np.testing.runstring("int64() + 1", {"int64": np.int64}))  # E: Any

reveal_type(np.testing.assert_string_equal("1", "1"))  # E: None

reveal_type(np.testing.rundocs())  # E: None
reveal_type(np.testing.rundocs("test.py"))  # E: None
reveal_type(np.testing.rundocs(Path("test.py"), raise_on_error=True))  # E: None

@np.testing.raises(RuntimeError, RuntimeWarning)
def func3(a: int) -> bool: ...

reveal_type(func3)  # E: def (a: builtins.int) -> builtins.bool

reveal_type(np.testing.assert_raises(RuntimeWarning))  # E: _AssertRaisesContext[builtins.RuntimeWarning]
reveal_type(np.testing.assert_raises(RuntimeWarning, func3, 5))  # E: None

reveal_type(np.testing.assert_raises_regex(RuntimeWarning, r"test"))  # E: _AssertRaisesContext[builtins.RuntimeWarning]
reveal_type(np.testing.assert_raises_regex(RuntimeWarning, b"test", func3, 5))  # E: None
reveal_type(np.testing.assert_raises_regex(RuntimeWarning, re.compile(b"test"), func3, 5))  # E: None

class Test: ...

def decorate(a: FT) -> FT:
    return a

reveal_type(np.testing.decorate_methods(Test, decorate))  # E: None
reveal_type(np.testing.decorate_methods(Test, decorate, None))  # E: None
reveal_type(np.testing.decorate_methods(Test, decorate, "test"))  # E: None
reveal_type(np.testing.decorate_methods(Test, decorate, b"test"))  # E: None
reveal_type(np.testing.decorate_methods(Test, decorate, re.compile("test")))  # E: None

reveal_type(np.testing.measure("for i in range(1000): np.sqrt(i**2)"))  # E: float
reveal_type(np.testing.measure(b"for i in range(1000): np.sqrt(i**2)", times=5))  # E: float

reveal_type(np.testing.assert_allclose(AR_i8, AR_f8))  # E: None
reveal_type(np.testing.assert_allclose(AR_i8, AR_f8, rtol=0.005))  # E: None
reveal_type(np.testing.assert_allclose(AR_i8, AR_f8, atol=1))  # E: None
reveal_type(np.testing.assert_allclose(AR_i8, AR_f8, equal_nan=True))  # E: None
reveal_type(np.testing.assert_allclose(AR_i8, AR_f8, err_msg="err"))  # E: None
reveal_type(np.testing.assert_allclose(AR_i8, AR_f8, verbose=False))  # E: None

reveal_type(np.testing.assert_array_almost_equal_nulp(AR_i8, AR_f8, nulp=2))  # E: None

reveal_type(np.testing.assert_array_max_ulp(AR_i8, AR_f8, maxulp=2))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.testing.assert_array_max_ulp(AR_i8, AR_f8, dtype=np.float32))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.testing.assert_warns(RuntimeWarning))  # E: _GeneratorContextManager[None]
reveal_type(np.testing.assert_warns(RuntimeWarning, func3, 5))  # E: bool

def func4(a: int, b: str) -> bool: ...

reveal_type(np.testing.assert_no_warnings())  # E: _GeneratorContextManager[None]
reveal_type(np.testing.assert_no_warnings(func3, 5))  # E: bool
reveal_type(np.testing.assert_no_warnings(func4, a=1, b="test"))  # E: bool
reveal_type(np.testing.assert_no_warnings(func4, 1, "test"))  # E: bool

reveal_type(np.testing.tempdir("test_dir"))  # E: _GeneratorContextManager[builtins.str]
reveal_type(np.testing.tempdir(prefix=b"test"))  # E: _GeneratorContextManager[builtins.bytes]
reveal_type(np.testing.tempdir("test_dir", dir=Path("here")))  # E: _GeneratorContextManager[builtins.str]

reveal_type(np.testing.temppath("test_dir", text=True))  # E: _GeneratorContextManager[builtins.str]
reveal_type(np.testing.temppath(prefix=b"test"))  # E: _GeneratorContextManager[builtins.bytes]
reveal_type(np.testing.temppath("test_dir", dir=Path("here")))  # E: _GeneratorContextManager[builtins.str]

reveal_type(np.testing.assert_no_gc_cycles())  # E: _GeneratorContextManager[None]
reveal_type(np.testing.assert_no_gc_cycles(func3, 5))  # E: None

reveal_type(np.testing.break_cycles())  # E: None

reveal_type(np.testing.TestCase())  # E: unittest.case.TestCase
reveal_type(np.testing.run_module_suite(file_to_run="numpy/tests/test_matlib.py"))  # E: None
