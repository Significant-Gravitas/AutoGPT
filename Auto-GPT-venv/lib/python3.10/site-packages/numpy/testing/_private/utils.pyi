import os
import sys
import ast
import types
import warnings
import unittest
import contextlib
from re import Pattern
from collections.abc import Callable, Iterable, Sequence
from typing import (
    Literal as L,
    Any,
    AnyStr,
    ClassVar,
    NoReturn,
    overload,
    type_check_only,
    TypeVar,
    Union,
    Final,
    SupportsIndex,
)
from typing_extensions import ParamSpec

from numpy import generic, dtype, number, object_, bool_, _FloatValue
from numpy._typing import (
    NDArray,
    ArrayLike,
    DTypeLike,
    _ArrayLikeNumber_co,
    _ArrayLikeObject_co,
    _ArrayLikeTD64_co,
    _ArrayLikeDT64_co,
)

from unittest.case import (
    SkipTest as SkipTest,
)

_P = ParamSpec("_P")
_T = TypeVar("_T")
_ET = TypeVar("_ET", bound=BaseException)
_FT = TypeVar("_FT", bound=Callable[..., Any])

# Must return a bool or an ndarray/generic type
# that is supported by `np.logical_and.reduce`
_ComparisonFunc = Callable[
    [NDArray[Any], NDArray[Any]],
    Union[
        bool,
        bool_,
        number[Any],
        NDArray[Union[bool_, number[Any], object_]],
    ],
]

__all__: list[str]

class KnownFailureException(Exception): ...
class IgnoreException(Exception): ...

class clear_and_catch_warnings(warnings.catch_warnings):
    class_modules: ClassVar[tuple[types.ModuleType, ...]]
    modules: set[types.ModuleType]
    @overload
    def __new__(
        cls,
        record: L[False] = ...,
        modules: Iterable[types.ModuleType] = ...,
    ) -> _clear_and_catch_warnings_without_records: ...
    @overload
    def __new__(
        cls,
        record: L[True],
        modules: Iterable[types.ModuleType] = ...,
    ) -> _clear_and_catch_warnings_with_records: ...
    @overload
    def __new__(
        cls,
        record: bool,
        modules: Iterable[types.ModuleType] = ...,
    ) -> clear_and_catch_warnings: ...
    def __enter__(self) -> None | list[warnings.WarningMessage]: ...
    def __exit__(
        self,
        __exc_type: None | type[BaseException] = ...,
        __exc_val: None | BaseException = ...,
        __exc_tb: None | types.TracebackType = ...,
    ) -> None: ...

# Type-check only `clear_and_catch_warnings` subclasses for both values of the
# `record` parameter. Copied from the stdlib `warnings` stubs.

@type_check_only
class _clear_and_catch_warnings_with_records(clear_and_catch_warnings):
    def __enter__(self) -> list[warnings.WarningMessage]: ...

@type_check_only
class _clear_and_catch_warnings_without_records(clear_and_catch_warnings):
    def __enter__(self) -> None: ...

class suppress_warnings:
    log: list[warnings.WarningMessage]
    def __init__(
        self,
        forwarding_rule: L["always", "module", "once", "location"] = ...,
    ) -> None: ...
    def filter(
        self,
        category: type[Warning] = ...,
        message: str = ...,
        module: None | types.ModuleType = ...,
    ) -> None: ...
    def record(
        self,
        category: type[Warning] = ...,
        message: str = ...,
        module: None | types.ModuleType = ...,
    ) -> list[warnings.WarningMessage]: ...
    def __enter__(self: _T) -> _T: ...
    def __exit__(
        self,
        __exc_type: None | type[BaseException] = ...,
        __exc_val: None | BaseException = ...,
        __exc_tb: None | types.TracebackType = ...,
    ) -> None: ...
    def __call__(self, func: _FT) -> _FT: ...

verbose: int
IS_PYPY: Final[bool]
IS_PYSTON: Final[bool]
HAS_REFCOUNT: Final[bool]
HAS_LAPACK64: Final[bool]

def assert_(val: object, msg: str | Callable[[], str] = ...) -> None: ...

# Contrary to runtime we can't do `os.name` checks while type checking,
# only `sys.platform` checks
if sys.platform == "win32" or sys.platform == "cygwin":
    def memusage(processName: str = ..., instance: int = ...) -> int: ...
elif sys.platform == "linux":
    def memusage(_proc_pid_stat: str | bytes | os.PathLike[Any] = ...) -> None | int: ...
else:
    def memusage() -> NoReturn: ...

if sys.platform == "linux":
    def jiffies(
        _proc_pid_stat: str | bytes | os.PathLike[Any] = ...,
        _load_time: list[float] = ...,
    ) -> int: ...
else:
    def jiffies(_load_time: list[float] = ...) -> int: ...

def build_err_msg(
    arrays: Iterable[object],
    err_msg: str,
    header: str = ...,
    verbose: bool = ...,
    names: Sequence[str] = ...,
    precision: None | SupportsIndex = ...,
) -> str: ...

def assert_equal(
    actual: object,
    desired: object,
    err_msg: str = ...,
    verbose: bool = ...,
) -> None: ...

def print_assert_equal(
    test_string: str,
    actual: object,
    desired: object,
) -> None: ...

def assert_almost_equal(
    actual: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    desired: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    decimal: int = ...,
    err_msg: str = ...,
    verbose: bool = ...,
) -> None: ...

# Anything that can be coerced into `builtins.float`
def assert_approx_equal(
    actual: _FloatValue,
    desired: _FloatValue,
    significant: int = ...,
    err_msg: str = ...,
    verbose: bool = ...,
) -> None: ...

def assert_array_compare(
    comparison: _ComparisonFunc,
    x: ArrayLike,
    y: ArrayLike,
    err_msg: str = ...,
    verbose: bool = ...,
    header: str = ...,
    precision: SupportsIndex = ...,
    equal_nan: bool = ...,
    equal_inf: bool = ...,
    *,
    strict: bool = ...
) -> None: ...

def assert_array_equal(
    x: ArrayLike,
    y: ArrayLike,
    err_msg: str = ...,
    verbose: bool = ...,
    *,
    strict: bool = ...
) -> None: ...

def assert_array_almost_equal(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    decimal: float = ...,
    err_msg: str = ...,
    verbose: bool = ...,
) -> None: ...

@overload
def assert_array_less(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    err_msg: str = ...,
    verbose: bool = ...,
) -> None: ...
@overload
def assert_array_less(
    x: _ArrayLikeTD64_co,
    y: _ArrayLikeTD64_co,
    err_msg: str = ...,
    verbose: bool = ...,
) -> None: ...
@overload
def assert_array_less(
    x: _ArrayLikeDT64_co,
    y: _ArrayLikeDT64_co,
    err_msg: str = ...,
    verbose: bool = ...,
) -> None: ...

def runstring(
    astr: str | bytes | types.CodeType,
    dict: None | dict[str, Any],
) -> Any: ...

def assert_string_equal(actual: str, desired: str) -> None: ...

def rundocs(
    filename: None | str | os.PathLike[str] = ...,
    raise_on_error: bool = ...,
) -> None: ...

def raises(*args: type[BaseException]) -> Callable[[_FT], _FT]: ...

@overload
def assert_raises(  # type: ignore
    expected_exception: type[BaseException] | tuple[type[BaseException], ...],
    callable: Callable[_P, Any],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> None: ...
@overload
def assert_raises(
    expected_exception: type[_ET] | tuple[type[_ET], ...],
    *,
    msg: None | str = ...,
) -> unittest.case._AssertRaisesContext[_ET]: ...

@overload
def assert_raises_regex(
    expected_exception: type[BaseException] | tuple[type[BaseException], ...],
    expected_regex: str | bytes | Pattern[Any],
    callable: Callable[_P, Any],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> None: ...
@overload
def assert_raises_regex(
    expected_exception: type[_ET] | tuple[type[_ET], ...],
    expected_regex: str | bytes | Pattern[Any],
    *,
    msg: None | str = ...,
) -> unittest.case._AssertRaisesContext[_ET]: ...

def decorate_methods(
    cls: type[Any],
    decorator: Callable[[Callable[..., Any]], Any],
    testmatch: None | str | bytes | Pattern[Any] = ...,
) -> None: ...

def measure(
    code_str: str | bytes | ast.mod | ast.AST,
    times: int = ...,
    label: None | str = ...,
) -> float: ...

@overload
def assert_allclose(
    actual: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    desired: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
    err_msg: str = ...,
    verbose: bool = ...,
) -> None: ...
@overload
def assert_allclose(
    actual: _ArrayLikeTD64_co,
    desired: _ArrayLikeTD64_co,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
    err_msg: str = ...,
    verbose: bool = ...,
) -> None: ...

def assert_array_almost_equal_nulp(
    x: _ArrayLikeNumber_co,
    y: _ArrayLikeNumber_co,
    nulp: float = ...,
) -> None: ...

def assert_array_max_ulp(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co,
    maxulp: float = ...,
    dtype: DTypeLike = ...,
) -> NDArray[Any]: ...

@overload
def assert_warns(
    warning_class: type[Warning],
) -> contextlib._GeneratorContextManager[None]: ...
@overload
def assert_warns(
    warning_class: type[Warning],
    func: Callable[_P, _T],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T: ...

@overload
def assert_no_warnings() -> contextlib._GeneratorContextManager[None]: ...
@overload
def assert_no_warnings(
    func: Callable[_P, _T],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T: ...

@overload
def tempdir(
    suffix: None = ...,
    prefix: None = ...,
    dir: None = ...,
) -> contextlib._GeneratorContextManager[str]: ...
@overload
def tempdir(
    suffix: None | AnyStr = ...,
    prefix: None | AnyStr = ...,
    dir: None | AnyStr | os.PathLike[AnyStr] = ...,
) -> contextlib._GeneratorContextManager[AnyStr]: ...

@overload
def temppath(
    suffix: None = ...,
    prefix: None = ...,
    dir: None = ...,
    text: bool = ...,
) -> contextlib._GeneratorContextManager[str]: ...
@overload
def temppath(
    suffix: None | AnyStr = ...,
    prefix: None | AnyStr = ...,
    dir: None | AnyStr | os.PathLike[AnyStr] = ...,
    text: bool = ...,
) -> contextlib._GeneratorContextManager[AnyStr]: ...

@overload
def assert_no_gc_cycles() -> contextlib._GeneratorContextManager[None]: ...
@overload
def assert_no_gc_cycles(
    func: Callable[_P, Any],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> None: ...

def break_cycles() -> None: ...
