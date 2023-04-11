import ctypes as ct
import numpy as np

dtype_U: np.dtype[np.str_]
dtype_V: np.dtype[np.void]
dtype_i8: np.dtype[np.int64]

reveal_type(np.dtype(np.float64))  # E: dtype[{float64}]
reveal_type(np.dtype(np.int64))  # E: dtype[{int64}]

# String aliases
reveal_type(np.dtype("float64"))  # E: dtype[{float64}]
reveal_type(np.dtype("float32"))  # E: dtype[{float32}]
reveal_type(np.dtype("int64"))  # E: dtype[{int64}]
reveal_type(np.dtype("int32"))  # E: dtype[{int32}]
reveal_type(np.dtype("bool"))  # E: dtype[bool_]
reveal_type(np.dtype("bytes"))  # E: dtype[bytes_]
reveal_type(np.dtype("str"))  # E: dtype[str_]

# Python types
reveal_type(np.dtype(complex))  # E: dtype[{cdouble}]
reveal_type(np.dtype(float))  # E: dtype[{double}]
reveal_type(np.dtype(int))  # E: dtype[{int_}]
reveal_type(np.dtype(bool))  # E: dtype[bool_]
reveal_type(np.dtype(str))  # E: dtype[str_]
reveal_type(np.dtype(bytes))  # E: dtype[bytes_]
reveal_type(np.dtype(object))  # E: dtype[object_]

# ctypes
reveal_type(np.dtype(ct.c_double))  # E: dtype[{double}]
reveal_type(np.dtype(ct.c_longlong))  # E: dtype[{longlong}]
reveal_type(np.dtype(ct.c_uint32))  # E: dtype[{uint32}]
reveal_type(np.dtype(ct.c_bool))  # E: dtype[bool_]
reveal_type(np.dtype(ct.c_char))  # E: dtype[bytes_]
reveal_type(np.dtype(ct.py_object))  # E: dtype[object_]

# Special case for None
reveal_type(np.dtype(None))  # E: dtype[{double}]

# Dtypes of dtypes
reveal_type(np.dtype(np.dtype(np.float64)))  # E: dtype[{float64}]

# Parameterized dtypes
reveal_type(np.dtype("S8"))  # E: dtype

# Void
reveal_type(np.dtype(("U", 10)))  # E: dtype[void]

# Methods and attributes
reveal_type(dtype_U.base)  # E: dtype[Any]
reveal_type(dtype_U.subdtype)  # E: Union[None, Tuple[dtype[Any], builtins.tuple[builtins.int, ...]]]
reveal_type(dtype_U.newbyteorder())  # E: dtype[str_]
reveal_type(dtype_U.type)  # E: Type[str_]
reveal_type(dtype_U.name)  # E: str
reveal_type(dtype_U.names)  # E: Union[None, builtins.tuple[builtins.str, ...]]

reveal_type(dtype_U * 0)  # E: dtype[str_]
reveal_type(dtype_U * 1)  # E: dtype[str_]
reveal_type(dtype_U * 2)  # E: dtype[str_]

reveal_type(dtype_i8 * 0)  # E: dtype[void]
reveal_type(dtype_i8 * 1)  # E: dtype[{int64}]
reveal_type(dtype_i8 * 2)  # E: dtype[void]

reveal_type(0 * dtype_U)  # E: dtype[str_]
reveal_type(1 * dtype_U)  # E: dtype[str_]
reveal_type(2 * dtype_U)  # E: dtype[str_]

reveal_type(0 * dtype_i8)  # E: dtype[Any]
reveal_type(1 * dtype_i8)  # E: dtype[Any]
reveal_type(2 * dtype_i8)  # E: dtype[Any]

reveal_type(dtype_V["f0"])  # E: dtype[Any]
reveal_type(dtype_V[0])  # E: dtype[Any]
reveal_type(dtype_V[["f0", "f1"]])  # E: dtype[void]
reveal_type(dtype_V[["f0"]])  # E: dtype[void]
