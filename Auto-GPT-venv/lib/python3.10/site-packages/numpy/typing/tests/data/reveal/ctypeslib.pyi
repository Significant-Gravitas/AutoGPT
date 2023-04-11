import ctypes
from typing import Any

import numpy as np
import numpy.typing as npt

AR_bool: npt.NDArray[np.bool_]
AR_ubyte: npt.NDArray[np.ubyte]
AR_ushort: npt.NDArray[np.ushort]
AR_uintc: npt.NDArray[np.uintc]
AR_uint: npt.NDArray[np.uint]
AR_ulonglong: npt.NDArray[np.ulonglong]
AR_byte: npt.NDArray[np.byte]
AR_short: npt.NDArray[np.short]
AR_intc: npt.NDArray[np.intc]
AR_int: npt.NDArray[np.int_]
AR_longlong: npt.NDArray[np.longlong]
AR_single: npt.NDArray[np.single]
AR_double: npt.NDArray[np.double]
AR_longdouble: npt.NDArray[np.longdouble]
AR_void: npt.NDArray[np.void]

pointer: ctypes._Pointer[Any]

reveal_type(np.ctypeslib.c_intp())  # E: {c_intp}

reveal_type(np.ctypeslib.ndpointer())  # E: Type[ctypeslib._ndptr[None]]
reveal_type(np.ctypeslib.ndpointer(dtype=np.float64))  # E: Type[ctypeslib._ndptr[dtype[{float64}]]]
reveal_type(np.ctypeslib.ndpointer(dtype=float))  # E: Type[ctypeslib._ndptr[dtype[Any]]]
reveal_type(np.ctypeslib.ndpointer(shape=(10, 3)))  # E: Type[ctypeslib._ndptr[None]]
reveal_type(np.ctypeslib.ndpointer(np.int64, shape=(10, 3)))  # E: Type[ctypeslib._concrete_ndptr[dtype[{int64}]]]
reveal_type(np.ctypeslib.ndpointer(int, shape=(1,)))  # E: Type[ctypeslib._concrete_ndptr[dtype[Any]]]

reveal_type(np.ctypeslib.as_ctypes_type(np.bool_))  # E: Type[ctypes.c_bool]
reveal_type(np.ctypeslib.as_ctypes_type(np.ubyte))  # E: Type[{c_ubyte}]
reveal_type(np.ctypeslib.as_ctypes_type(np.ushort))  # E: Type[{c_ushort}]
reveal_type(np.ctypeslib.as_ctypes_type(np.uintc))  # E: Type[{c_uint}]
reveal_type(np.ctypeslib.as_ctypes_type(np.uint))  # E: Type[{c_ulong}]
reveal_type(np.ctypeslib.as_ctypes_type(np.ulonglong))  # E: Type[{c_ulonglong}]
reveal_type(np.ctypeslib.as_ctypes_type(np.byte))  # E: Type[{c_byte}]
reveal_type(np.ctypeslib.as_ctypes_type(np.short))  # E: Type[{c_short}]
reveal_type(np.ctypeslib.as_ctypes_type(np.intc))  # E: Type[{c_int}]
reveal_type(np.ctypeslib.as_ctypes_type(np.int_))  # E: Type[{c_long}]
reveal_type(np.ctypeslib.as_ctypes_type(np.longlong))  # E: Type[{c_longlong}]
reveal_type(np.ctypeslib.as_ctypes_type(np.single))  # E: Type[{c_float}]
reveal_type(np.ctypeslib.as_ctypes_type(np.double))  # E: Type[{c_double}]
reveal_type(np.ctypeslib.as_ctypes_type(np.longdouble))  # E: Type[{c_longdouble}]
reveal_type(np.ctypeslib.as_ctypes_type(ctypes.c_double))  # E: Type[{c_double}]
reveal_type(np.ctypeslib.as_ctypes_type("q"))  # E: Type[ctypes.c_longlong]
reveal_type(np.ctypeslib.as_ctypes_type([("i8", np.int64), ("f8", np.float64)]))  # E: Type[Any]
reveal_type(np.ctypeslib.as_ctypes_type("i8"))  # E: Type[Any]
reveal_type(np.ctypeslib.as_ctypes_type("f8"))  # E: Type[Any]

reveal_type(np.ctypeslib.as_ctypes(AR_bool.take(0)))  # E: ctypes.c_bool
reveal_type(np.ctypeslib.as_ctypes(AR_ubyte.take(0)))  # E: {c_ubyte}
reveal_type(np.ctypeslib.as_ctypes(AR_ushort.take(0)))  # E: {c_ushort}
reveal_type(np.ctypeslib.as_ctypes(AR_uintc.take(0)))  # E: {c_uint}
reveal_type(np.ctypeslib.as_ctypes(AR_uint.take(0)))  # E: {c_ulong}
reveal_type(np.ctypeslib.as_ctypes(AR_ulonglong.take(0)))  # E: {c_ulonglong}
reveal_type(np.ctypeslib.as_ctypes(AR_byte.take(0)))  # E: {c_byte}
reveal_type(np.ctypeslib.as_ctypes(AR_short.take(0)))  # E: {c_short}
reveal_type(np.ctypeslib.as_ctypes(AR_intc.take(0)))  # E: {c_int}
reveal_type(np.ctypeslib.as_ctypes(AR_int.take(0)))  # E: {c_long}
reveal_type(np.ctypeslib.as_ctypes(AR_longlong.take(0)))  # E: {c_longlong}
reveal_type(np.ctypeslib.as_ctypes(AR_single.take(0)))  # E: {c_float}
reveal_type(np.ctypeslib.as_ctypes(AR_double.take(0)))  # E: {c_double}
reveal_type(np.ctypeslib.as_ctypes(AR_longdouble.take(0)))  # E: {c_longdouble}
reveal_type(np.ctypeslib.as_ctypes(AR_void.take(0)))  # E: Any
reveal_type(np.ctypeslib.as_ctypes(AR_bool))  # E: ctypes.Array[ctypes.c_bool]
reveal_type(np.ctypeslib.as_ctypes(AR_ubyte))  # E: ctypes.Array[{c_ubyte}]
reveal_type(np.ctypeslib.as_ctypes(AR_ushort))  # E: ctypes.Array[{c_ushort}]
reveal_type(np.ctypeslib.as_ctypes(AR_uintc))  # E: ctypes.Array[{c_uint}]
reveal_type(np.ctypeslib.as_ctypes(AR_uint))  # E: ctypes.Array[{c_ulong}]
reveal_type(np.ctypeslib.as_ctypes(AR_ulonglong))  # E: ctypes.Array[{c_ulonglong}]
reveal_type(np.ctypeslib.as_ctypes(AR_byte))  # E: ctypes.Array[{c_byte}]
reveal_type(np.ctypeslib.as_ctypes(AR_short))  # E: ctypes.Array[{c_short}]
reveal_type(np.ctypeslib.as_ctypes(AR_intc))  # E: ctypes.Array[{c_int}]
reveal_type(np.ctypeslib.as_ctypes(AR_int))  # E: ctypes.Array[{c_long}]
reveal_type(np.ctypeslib.as_ctypes(AR_longlong))  # E: ctypes.Array[{c_longlong}]
reveal_type(np.ctypeslib.as_ctypes(AR_single))  # E: ctypes.Array[{c_float}]
reveal_type(np.ctypeslib.as_ctypes(AR_double))  # E: ctypes.Array[{c_double}]
reveal_type(np.ctypeslib.as_ctypes(AR_longdouble))  # E: ctypes.Array[{c_longdouble}]
reveal_type(np.ctypeslib.as_ctypes(AR_void))  # E: ctypes.Array[Any]

reveal_type(np.ctypeslib.as_array(AR_ubyte))  # E: ndarray[Any, dtype[{ubyte}]]
reveal_type(np.ctypeslib.as_array(1))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.ctypeslib.as_array(pointer))  # E: ndarray[Any, dtype[Any]]
