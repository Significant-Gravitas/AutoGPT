import datetime as dt
from typing import Any, TypeVar
from pathlib import Path

import numpy as np
import numpy.typing as npt

_SCT = TypeVar("_SCT", bound=np.generic, covariant=True)

class SubClass(np.ndarray[Any, np.dtype[_SCT]]): ...

subclass: SubClass[np.float64]

AR_f8: npt.NDArray[np.float64]
AR_i8: npt.NDArray[np.int64]
AR_u1: npt.NDArray[np.uint8]
AR_m: npt.NDArray[np.timedelta64]
AR_M: npt.NDArray[np.datetime64]

AR_LIKE_f: list[float]
AR_LIKE_i: list[int]

m: np.timedelta64
M: np.datetime64

b_f8 = np.broadcast(AR_f8)
b_i8_f8_f8 = np.broadcast(AR_i8, AR_f8, AR_f8)

nditer_obj: np.nditer

date_scalar: dt.date
date_seq: list[dt.date]
timedelta_seq: list[dt.timedelta]

def func(a: int) -> bool: ...

reveal_type(next(b_f8))  # E: tuple[Any, ...]
reveal_type(b_f8.reset())  # E: None
reveal_type(b_f8.index)  # E: int
reveal_type(b_f8.iters)  # E: tuple[flatiter[Any], ...]
reveal_type(b_f8.nd)  # E: int
reveal_type(b_f8.ndim)  # E: int
reveal_type(b_f8.numiter)  # E: int
reveal_type(b_f8.shape)  # E: tuple[builtins.int, ...]
reveal_type(b_f8.size)  # E: int

reveal_type(next(b_i8_f8_f8))  # E: tuple[Any, ...]
reveal_type(b_i8_f8_f8.reset())  # E: None
reveal_type(b_i8_f8_f8.index)  # E: int
reveal_type(b_i8_f8_f8.iters)  # E: tuple[flatiter[Any], ...]
reveal_type(b_i8_f8_f8.nd)  # E: int
reveal_type(b_i8_f8_f8.ndim)  # E: int
reveal_type(b_i8_f8_f8.numiter)  # E: int
reveal_type(b_i8_f8_f8.shape)  # E: tuple[builtins.int, ...]
reveal_type(b_i8_f8_f8.size)  # E: int

reveal_type(np.inner(AR_f8, AR_i8))  # E: Any

reveal_type(np.where([True, True, False]))  # E: tuple[ndarray[Any, dtype[{intp}]], ...]
reveal_type(np.where([True, True, False], 1, 0))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.lexsort([0, 1, 2]))  # E: Any

reveal_type(np.can_cast(np.dtype("i8"), int))  # E: bool
reveal_type(np.can_cast(AR_f8, "f8"))  # E: bool
reveal_type(np.can_cast(AR_f8, np.complex128, casting="unsafe"))  # E: bool

reveal_type(np.min_scalar_type([1]))  # E: dtype[Any]
reveal_type(np.min_scalar_type(AR_f8))  # E: dtype[Any]

reveal_type(np.result_type(int, [1]))  # E: dtype[Any]
reveal_type(np.result_type(AR_f8, AR_u1))  # E: dtype[Any]
reveal_type(np.result_type(AR_f8, np.complex128))  # E: dtype[Any]

reveal_type(np.dot(AR_LIKE_f, AR_i8))  # E: Any
reveal_type(np.dot(AR_u1, 1))  # E: Any
reveal_type(np.dot(1.5j, 1))  # E: Any
reveal_type(np.dot(AR_u1, 1, out=AR_f8))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(np.vdot(AR_LIKE_f, AR_i8))  # E: floating[Any]
reveal_type(np.vdot(AR_u1, 1))  # E: signedinteger[Any]
reveal_type(np.vdot(1.5j, 1))  # E: complexfloating[Any, Any]

reveal_type(np.bincount(AR_i8))  # E: ndarray[Any, dtype[{intp}]]

reveal_type(np.copyto(AR_f8, [1., 1.5, 1.6]))  # E: None

reveal_type(np.putmask(AR_f8, [True, True, False], 1.5))  # E: None

reveal_type(np.packbits(AR_i8))  # ndarray[Any, dtype[{uint8}]]
reveal_type(np.packbits(AR_u1))  # ndarray[Any, dtype[{uint8}]]

reveal_type(np.unpackbits(AR_u1))  # ndarray[Any, dtype[{uint8}]]

reveal_type(np.shares_memory(1, 2))  # E: bool
reveal_type(np.shares_memory(AR_f8, AR_f8, max_work=1))  # E: bool

reveal_type(np.may_share_memory(1, 2))  # E: bool
reveal_type(np.may_share_memory(AR_f8, AR_f8, max_work=1))  # E: bool

reveal_type(np.geterrobj())  # E: list[Any]

reveal_type(np.seterrobj([8192, 521, None]))  # E: None

reveal_type(np.promote_types(np.int32, np.int64))  # E: dtype[Any]
reveal_type(np.promote_types("f4", float))  # E: dtype[Any]

reveal_type(np.frompyfunc(func, 1, 1, identity=None))  # ufunc

reveal_type(np.datetime_data("m8[D]"))  # E: Tuple[builtins.str, builtins.int]
reveal_type(np.datetime_data(np.datetime64))  # E: Tuple[builtins.str, builtins.int]
reveal_type(np.datetime_data(np.dtype(np.timedelta64)))  # E: Tuple[builtins.str, builtins.int]

reveal_type(np.busday_count("2011-01", "2011-02"))  # E: {int_}
reveal_type(np.busday_count(["2011-01"], "2011-02"))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.busday_count(["2011-01"], date_scalar))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(np.busday_offset(M, m))  # E: datetime64
reveal_type(np.busday_offset(date_scalar, m))  # E: datetime64
reveal_type(np.busday_offset(M, 5))  # E: datetime64
reveal_type(np.busday_offset(AR_M, m))  # E: ndarray[Any, dtype[datetime64]]
reveal_type(np.busday_offset(M, timedelta_seq))  # E: ndarray[Any, dtype[datetime64]]
reveal_type(np.busday_offset("2011-01", "2011-02", roll="forward"))  # E: datetime64
reveal_type(np.busday_offset(["2011-01"], "2011-02", roll="forward"))  # E: ndarray[Any, dtype[datetime64]]

reveal_type(np.is_busday("2012"))  # E: bool_
reveal_type(np.is_busday(date_scalar))  # E: bool_
reveal_type(np.is_busday(["2012"]))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.datetime_as_string(M))  # E: str_
reveal_type(np.datetime_as_string(AR_M))  # E: ndarray[Any, dtype[str_]]

reveal_type(np.busdaycalendar(holidays=date_seq))  # E: busdaycalendar
reveal_type(np.busdaycalendar(holidays=[M]))  # E: busdaycalendar

reveal_type(np.compare_chararrays("a", "b", "!=", rstrip=False))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.compare_chararrays(b"a", b"a", "==", True))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.add_docstring(func, "test"))  # E: None

reveal_type(np.nested_iters([AR_i8, AR_i8], [[0], [1]], flags=["c_index"]))  # E: tuple[nditer, ...]
reveal_type(np.nested_iters([AR_i8, AR_i8], [[0], [1]], op_flags=[["readonly", "readonly"]]))  # E: tuple[nditer, ...]
reveal_type(np.nested_iters([AR_i8, AR_i8], [[0], [1]], op_dtypes=np.int_))  # E: tuple[nditer, ...]
reveal_type(np.nested_iters([AR_i8, AR_i8], [[0], [1]], order="C", casting="no"))  # E: tuple[nditer, ...]
