from typing import Any

import numpy as np
from numpy._typing import NDArray, _128Bit

# Can't directly import `np.float128` as it is not available on all platforms
f16: np.floating[_128Bit]

c16 = np.complex128()
f8 = np.float64()
i8 = np.int64()
u8 = np.uint64()

c8 = np.complex64()
f4 = np.float32()
i4 = np.int32()
u4 = np.uint32()

dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

b_ = np.bool_()

b = bool()
c = complex()
f = float()
i = int()

AR_b: np.ndarray[Any, np.dtype[np.bool_]]
AR_u: np.ndarray[Any, np.dtype[np.uint32]]
AR_i: np.ndarray[Any, np.dtype[np.int64]]
AR_f: np.ndarray[Any, np.dtype[np.float64]]
AR_c: np.ndarray[Any, np.dtype[np.complex128]]
AR_m: np.ndarray[Any, np.dtype[np.timedelta64]]
AR_M: np.ndarray[Any, np.dtype[np.datetime64]]
AR_O: np.ndarray[Any, np.dtype[np.object_]]
AR_number: NDArray[np.number[Any]]

AR_LIKE_b: list[bool]
AR_LIKE_u: list[np.uint32]
AR_LIKE_i: list[int]
AR_LIKE_f: list[float]
AR_LIKE_c: list[complex]
AR_LIKE_m: list[np.timedelta64]
AR_LIKE_M: list[np.datetime64]
AR_LIKE_O: list[np.object_]

# Array subtraction

reveal_type(AR_number - AR_number)  # E: ndarray[Any, dtype[number[Any]]]

reveal_type(AR_b - AR_LIKE_u)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_b - AR_LIKE_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_b - AR_LIKE_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_b - AR_LIKE_c)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_b - AR_LIKE_m)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_b - AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_u - AR_b)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_LIKE_i - AR_b)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_LIKE_f - AR_b)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_c - AR_b)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_LIKE_m - AR_b)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_LIKE_M - AR_b)  # E: ndarray[Any, dtype[datetime64]]
reveal_type(AR_LIKE_O - AR_b)  # E: Any

reveal_type(AR_u - AR_LIKE_b)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_u - AR_LIKE_u)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_u - AR_LIKE_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_u - AR_LIKE_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_u - AR_LIKE_c)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_u - AR_LIKE_m)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_u - AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_b - AR_u)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_LIKE_u - AR_u)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_LIKE_i - AR_u)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_LIKE_f - AR_u)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_c - AR_u)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_LIKE_m - AR_u)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_LIKE_M - AR_u)  # E: ndarray[Any, dtype[datetime64]]
reveal_type(AR_LIKE_O - AR_u)  # E: Any

reveal_type(AR_i - AR_LIKE_b)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_i - AR_LIKE_u)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_i - AR_LIKE_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_i - AR_LIKE_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_i - AR_LIKE_c)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_i - AR_LIKE_m)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_i - AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_b - AR_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_LIKE_u - AR_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_LIKE_i - AR_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_LIKE_f - AR_i)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_c - AR_i)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_LIKE_m - AR_i)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_LIKE_M - AR_i)  # E: ndarray[Any, dtype[datetime64]]
reveal_type(AR_LIKE_O - AR_i)  # E: Any

reveal_type(AR_f - AR_LIKE_b)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_f - AR_LIKE_u)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_f - AR_LIKE_i)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_f - AR_LIKE_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_f - AR_LIKE_c)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_f - AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_b - AR_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_u - AR_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_i - AR_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_f - AR_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_c - AR_f)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_LIKE_O - AR_f)  # E: Any

reveal_type(AR_c - AR_LIKE_b)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_c - AR_LIKE_u)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_c - AR_LIKE_i)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_c - AR_LIKE_f)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_c - AR_LIKE_c)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_c - AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_b - AR_c)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_LIKE_u - AR_c)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_LIKE_i - AR_c)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_LIKE_f - AR_c)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_LIKE_c - AR_c)  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(AR_LIKE_O - AR_c)  # E: Any

reveal_type(AR_m - AR_LIKE_b)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_m - AR_LIKE_u)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_m - AR_LIKE_i)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_m - AR_LIKE_m)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_m - AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_b - AR_m)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_LIKE_u - AR_m)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_LIKE_i - AR_m)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_LIKE_m - AR_m)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_LIKE_M - AR_m)  # E: ndarray[Any, dtype[datetime64]]
reveal_type(AR_LIKE_O - AR_m)  # E: Any

reveal_type(AR_M - AR_LIKE_b)  # E: ndarray[Any, dtype[datetime64]]
reveal_type(AR_M - AR_LIKE_u)  # E: ndarray[Any, dtype[datetime64]]
reveal_type(AR_M - AR_LIKE_i)  # E: ndarray[Any, dtype[datetime64]]
reveal_type(AR_M - AR_LIKE_m)  # E: ndarray[Any, dtype[datetime64]]
reveal_type(AR_M - AR_LIKE_M)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_M - AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_M - AR_M)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_LIKE_O - AR_M)  # E: Any

reveal_type(AR_O - AR_LIKE_b)  # E: Any
reveal_type(AR_O - AR_LIKE_u)  # E: Any
reveal_type(AR_O - AR_LIKE_i)  # E: Any
reveal_type(AR_O - AR_LIKE_f)  # E: Any
reveal_type(AR_O - AR_LIKE_c)  # E: Any
reveal_type(AR_O - AR_LIKE_m)  # E: Any
reveal_type(AR_O - AR_LIKE_M)  # E: Any
reveal_type(AR_O - AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_b - AR_O)  # E: Any
reveal_type(AR_LIKE_u - AR_O)  # E: Any
reveal_type(AR_LIKE_i - AR_O)  # E: Any
reveal_type(AR_LIKE_f - AR_O)  # E: Any
reveal_type(AR_LIKE_c - AR_O)  # E: Any
reveal_type(AR_LIKE_m - AR_O)  # E: Any
reveal_type(AR_LIKE_M - AR_O)  # E: Any
reveal_type(AR_LIKE_O - AR_O)  # E: Any

# Array floor division

reveal_type(AR_b // AR_LIKE_b)  # E: ndarray[Any, dtype[{int8}]]
reveal_type(AR_b // AR_LIKE_u)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_b // AR_LIKE_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_b // AR_LIKE_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_b // AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_b // AR_b)  # E: ndarray[Any, dtype[{int8}]]
reveal_type(AR_LIKE_u // AR_b)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_LIKE_i // AR_b)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_LIKE_f // AR_b)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_O // AR_b)  # E: Any

reveal_type(AR_u // AR_LIKE_b)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_u // AR_LIKE_u)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_u // AR_LIKE_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_u // AR_LIKE_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_u // AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_b // AR_u)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_LIKE_u // AR_u)  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(AR_LIKE_i // AR_u)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_LIKE_f // AR_u)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_m // AR_u)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_LIKE_O // AR_u)  # E: Any

reveal_type(AR_i // AR_LIKE_b)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_i // AR_LIKE_u)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_i // AR_LIKE_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_i // AR_LIKE_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_i // AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_b // AR_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_LIKE_u // AR_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_LIKE_i // AR_i)  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(AR_LIKE_f // AR_i)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_m // AR_i)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_LIKE_O // AR_i)  # E: Any

reveal_type(AR_f // AR_LIKE_b)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_f // AR_LIKE_u)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_f // AR_LIKE_i)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_f // AR_LIKE_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_f // AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_b // AR_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_u // AR_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_i // AR_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_f // AR_f)  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(AR_LIKE_m // AR_f)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_LIKE_O // AR_f)  # E: Any

reveal_type(AR_m // AR_LIKE_u)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_m // AR_LIKE_i)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_m // AR_LIKE_f)  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(AR_m // AR_LIKE_m)  # E: ndarray[Any, dtype[{int64}]]
reveal_type(AR_m // AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_m // AR_m)  # E: ndarray[Any, dtype[{int64}]]
reveal_type(AR_LIKE_O // AR_m)  # E: Any

reveal_type(AR_O // AR_LIKE_b)  # E: Any
reveal_type(AR_O // AR_LIKE_u)  # E: Any
reveal_type(AR_O // AR_LIKE_i)  # E: Any
reveal_type(AR_O // AR_LIKE_f)  # E: Any
reveal_type(AR_O // AR_LIKE_m)  # E: Any
reveal_type(AR_O // AR_LIKE_M)  # E: Any
reveal_type(AR_O // AR_LIKE_O)  # E: Any

reveal_type(AR_LIKE_b // AR_O)  # E: Any
reveal_type(AR_LIKE_u // AR_O)  # E: Any
reveal_type(AR_LIKE_i // AR_O)  # E: Any
reveal_type(AR_LIKE_f // AR_O)  # E: Any
reveal_type(AR_LIKE_m // AR_O)  # E: Any
reveal_type(AR_LIKE_M // AR_O)  # E: Any
reveal_type(AR_LIKE_O // AR_O)  # E: Any

# unary ops

reveal_type(-f16)  # E: {float128}
reveal_type(-c16)  # E: {complex128}
reveal_type(-c8)  # E: {complex64}
reveal_type(-f8)  # E: {float64}
reveal_type(-f4)  # E: {float32}
reveal_type(-i8)  # E: {int64}
reveal_type(-i4)  # E: {int32}
reveal_type(-u8)  # E: {uint64}
reveal_type(-u4)  # E: {uint32}
reveal_type(-td)  # E: timedelta64
reveal_type(-AR_f)  # E: Any

reveal_type(+f16)  # E: {float128}
reveal_type(+c16)  # E: {complex128}
reveal_type(+c8)  # E: {complex64}
reveal_type(+f8)  # E: {float64}
reveal_type(+f4)  # E: {float32}
reveal_type(+i8)  # E: {int64}
reveal_type(+i4)  # E: {int32}
reveal_type(+u8)  # E: {uint64}
reveal_type(+u4)  # E: {uint32}
reveal_type(+td)  # E: timedelta64
reveal_type(+AR_f)  # E: Any

reveal_type(abs(f16))  # E: {float128}
reveal_type(abs(c16))  # E: {float64}
reveal_type(abs(c8))  # E: {float32}
reveal_type(abs(f8))  # E: {float64}
reveal_type(abs(f4))  # E: {float32}
reveal_type(abs(i8))  # E: {int64}
reveal_type(abs(i4))  # E: {int32}
reveal_type(abs(u8))  # E: {uint64}
reveal_type(abs(u4))  # E: {uint32}
reveal_type(abs(td))  # E: timedelta64
reveal_type(abs(b_))  # E: bool_
reveal_type(abs(AR_f))  # E: Any

# Time structures

reveal_type(dt + td)  # E: datetime64
reveal_type(dt + i)  # E: datetime64
reveal_type(dt + i4)  # E: datetime64
reveal_type(dt + i8)  # E: datetime64
reveal_type(dt - dt)  # E: timedelta64
reveal_type(dt - i)  # E: datetime64
reveal_type(dt - i4)  # E: datetime64
reveal_type(dt - i8)  # E: datetime64

reveal_type(td + td)  # E: timedelta64
reveal_type(td + i)  # E: timedelta64
reveal_type(td + i4)  # E: timedelta64
reveal_type(td + i8)  # E: timedelta64
reveal_type(td - td)  # E: timedelta64
reveal_type(td - i)  # E: timedelta64
reveal_type(td - i4)  # E: timedelta64
reveal_type(td - i8)  # E: timedelta64
reveal_type(td / f)  # E: timedelta64
reveal_type(td / f4)  # E: timedelta64
reveal_type(td / f8)  # E: timedelta64
reveal_type(td / td)  # E: {float64}
reveal_type(td // td)  # E: {int64}

# boolean

reveal_type(b_ / b)  # E: {float64}
reveal_type(b_ / b_)  # E: {float64}
reveal_type(b_ / i)  # E: {float64}
reveal_type(b_ / i8)  # E: {float64}
reveal_type(b_ / i4)  # E: {float64}
reveal_type(b_ / u8)  # E: {float64}
reveal_type(b_ / u4)  # E: {float64}
reveal_type(b_ / f)  # E: {float64}
reveal_type(b_ / f16)  # E: {float128}
reveal_type(b_ / f8)  # E: {float64}
reveal_type(b_ / f4)  # E: {float32}
reveal_type(b_ / c)  # E: {complex128}
reveal_type(b_ / c16)  # E: {complex128}
reveal_type(b_ / c8)  # E: {complex64}

reveal_type(b / b_)  # E: {float64}
reveal_type(b_ / b_)  # E: {float64}
reveal_type(i / b_)  # E: {float64}
reveal_type(i8 / b_)  # E: {float64}
reveal_type(i4 / b_)  # E: {float64}
reveal_type(u8 / b_)  # E: {float64}
reveal_type(u4 / b_)  # E: {float64}
reveal_type(f / b_)  # E: {float64}
reveal_type(f16 / b_)  # E: {float128}
reveal_type(f8 / b_)  # E: {float64}
reveal_type(f4 / b_)  # E: {float32}
reveal_type(c / b_)  # E: {complex128}
reveal_type(c16 / b_)  # E: {complex128}
reveal_type(c8 / b_)  # E: {complex64}

# Complex

reveal_type(c16 + f16)  # E: {complex256}
reveal_type(c16 + c16)  # E: {complex128}
reveal_type(c16 + f8)  # E: {complex128}
reveal_type(c16 + i8)  # E: {complex128}
reveal_type(c16 + c8)  # E: {complex128}
reveal_type(c16 + f4)  # E: {complex128}
reveal_type(c16 + i4)  # E: {complex128}
reveal_type(c16 + b_)  # E: {complex128}
reveal_type(c16 + b)  # E: {complex128}
reveal_type(c16 + c)  # E: {complex128}
reveal_type(c16 + f)  # E: {complex128}
reveal_type(c16 + i)  # E: {complex128}
reveal_type(c16 + AR_f)  # E: Any

reveal_type(f16 + c16)  # E: {complex256}
reveal_type(c16 + c16)  # E: {complex128}
reveal_type(f8 + c16)  # E: {complex128}
reveal_type(i8 + c16)  # E: {complex128}
reveal_type(c8 + c16)  # E: {complex128}
reveal_type(f4 + c16)  # E: {complex128}
reveal_type(i4 + c16)  # E: {complex128}
reveal_type(b_ + c16)  # E: {complex128}
reveal_type(b + c16)  # E: {complex128}
reveal_type(c + c16)  # E: {complex128}
reveal_type(f + c16)  # E: {complex128}
reveal_type(i + c16)  # E: {complex128}
reveal_type(AR_f + c16)  # E: Any

reveal_type(c8 + f16)  # E: {complex256}
reveal_type(c8 + c16)  # E: {complex128}
reveal_type(c8 + f8)  # E: {complex128}
reveal_type(c8 + i8)  # E: {complex128}
reveal_type(c8 + c8)  # E: {complex64}
reveal_type(c8 + f4)  # E: {complex64}
reveal_type(c8 + i4)  # E: {complex64}
reveal_type(c8 + b_)  # E: {complex64}
reveal_type(c8 + b)  # E: {complex64}
reveal_type(c8 + c)  # E: {complex128}
reveal_type(c8 + f)  # E: {complex128}
reveal_type(c8 + i)  # E: complexfloating[{_NBitInt}, {_NBitInt}]
reveal_type(c8 + AR_f)  # E: Any

reveal_type(f16 + c8)  # E: {complex256}
reveal_type(c16 + c8)  # E: {complex128}
reveal_type(f8 + c8)  # E: {complex128}
reveal_type(i8 + c8)  # E: {complex128}
reveal_type(c8 + c8)  # E: {complex64}
reveal_type(f4 + c8)  # E: {complex64}
reveal_type(i4 + c8)  # E: {complex64}
reveal_type(b_ + c8)  # E: {complex64}
reveal_type(b + c8)  # E: {complex64}
reveal_type(c + c8)  # E: {complex128}
reveal_type(f + c8)  # E: {complex128}
reveal_type(i + c8)  # E: complexfloating[{_NBitInt}, {_NBitInt}]
reveal_type(AR_f + c8)  # E: Any

# Float

reveal_type(f8 + f16)  # E: {float128}
reveal_type(f8 + f8)  # E: {float64}
reveal_type(f8 + i8)  # E: {float64}
reveal_type(f8 + f4)  # E: {float64}
reveal_type(f8 + i4)  # E: {float64}
reveal_type(f8 + b_)  # E: {float64}
reveal_type(f8 + b)  # E: {float64}
reveal_type(f8 + c)  # E: {complex128}
reveal_type(f8 + f)  # E: {float64}
reveal_type(f8 + i)  # E: {float64}
reveal_type(f8 + AR_f)  # E: Any

reveal_type(f16 + f8)  # E: {float128}
reveal_type(f8 + f8)  # E: {float64}
reveal_type(i8 + f8)  # E: {float64}
reveal_type(f4 + f8)  # E: {float64}
reveal_type(i4 + f8)  # E: {float64}
reveal_type(b_ + f8)  # E: {float64}
reveal_type(b + f8)  # E: {float64}
reveal_type(c + f8)  # E: {complex128}
reveal_type(f + f8)  # E: {float64}
reveal_type(i + f8)  # E: {float64}
reveal_type(AR_f + f8)  # E: Any

reveal_type(f4 + f16)  # E: {float128}
reveal_type(f4 + f8)  # E: {float64}
reveal_type(f4 + i8)  # E: {float64}
reveal_type(f4 + f4)  # E: {float32}
reveal_type(f4 + i4)  # E: {float32}
reveal_type(f4 + b_)  # E: {float32}
reveal_type(f4 + b)  # E: {float32}
reveal_type(f4 + c)  # E: {complex128}
reveal_type(f4 + f)  # E: {float64}
reveal_type(f4 + i)  # E: floating[{_NBitInt}]
reveal_type(f4 + AR_f)  # E: Any

reveal_type(f16 + f4)  # E: {float128}
reveal_type(f8 + f4)  # E: {float64}
reveal_type(i8 + f4)  # E: {float64}
reveal_type(f4 + f4)  # E: {float32}
reveal_type(i4 + f4)  # E: {float32}
reveal_type(b_ + f4)  # E: {float32}
reveal_type(b + f4)  # E: {float32}
reveal_type(c + f4)  # E: {complex128}
reveal_type(f + f4)  # E: {float64}
reveal_type(i + f4)  # E: floating[{_NBitInt}]
reveal_type(AR_f + f4)  # E: Any

# Int

reveal_type(i8 + i8)  # E: {int64}
reveal_type(i8 + u8)  # E: Any
reveal_type(i8 + i4)  # E: {int64}
reveal_type(i8 + u4)  # E: Any
reveal_type(i8 + b_)  # E: {int64}
reveal_type(i8 + b)  # E: {int64}
reveal_type(i8 + c)  # E: {complex128}
reveal_type(i8 + f)  # E: {float64}
reveal_type(i8 + i)  # E: {int64}
reveal_type(i8 + AR_f)  # E: Any

reveal_type(u8 + u8)  # E: {uint64}
reveal_type(u8 + i4)  # E: Any
reveal_type(u8 + u4)  # E: {uint64}
reveal_type(u8 + b_)  # E: {uint64}
reveal_type(u8 + b)  # E: {uint64}
reveal_type(u8 + c)  # E: {complex128}
reveal_type(u8 + f)  # E: {float64}
reveal_type(u8 + i)  # E: Any
reveal_type(u8 + AR_f)  # E: Any

reveal_type(i8 + i8)  # E: {int64}
reveal_type(u8 + i8)  # E: Any
reveal_type(i4 + i8)  # E: {int64}
reveal_type(u4 + i8)  # E: Any
reveal_type(b_ + i8)  # E: {int64}
reveal_type(b + i8)  # E: {int64}
reveal_type(c + i8)  # E: {complex128}
reveal_type(f + i8)  # E: {float64}
reveal_type(i + i8)  # E: {int64}
reveal_type(AR_f + i8)  # E: Any

reveal_type(u8 + u8)  # E: {uint64}
reveal_type(i4 + u8)  # E: Any
reveal_type(u4 + u8)  # E: {uint64}
reveal_type(b_ + u8)  # E: {uint64}
reveal_type(b + u8)  # E: {uint64}
reveal_type(c + u8)  # E: {complex128}
reveal_type(f + u8)  # E: {float64}
reveal_type(i + u8)  # E: Any
reveal_type(AR_f + u8)  # E: Any

reveal_type(i4 + i8)  # E: {int64}
reveal_type(i4 + i4)  # E: {int32}
reveal_type(i4 + i)  # E: {int_}
reveal_type(i4 + b_)  # E: {int32}
reveal_type(i4 + b)  # E: {int32}
reveal_type(i4 + AR_f)  # E: Any

reveal_type(u4 + i8)  # E: Any
reveal_type(u4 + i4)  # E: Any
reveal_type(u4 + u8)  # E: {uint64}
reveal_type(u4 + u4)  # E: {uint32}
reveal_type(u4 + i)  # E: Any
reveal_type(u4 + b_)  # E: {uint32}
reveal_type(u4 + b)  # E: {uint32}
reveal_type(u4 + AR_f)  # E: Any

reveal_type(i8 + i4)  # E: {int64}
reveal_type(i4 + i4)  # E: {int32}
reveal_type(i + i4)  # E: {int_}
reveal_type(b_ + i4)  # E: {int32}
reveal_type(b + i4)  # E: {int32}
reveal_type(AR_f + i4)  # E: Any

reveal_type(i8 + u4)  # E: Any
reveal_type(i4 + u4)  # E: Any
reveal_type(u8 + u4)  # E: {uint64}
reveal_type(u4 + u4)  # E: {uint32}
reveal_type(b_ + u4)  # E: {uint32}
reveal_type(b + u4)  # E: {uint32}
reveal_type(i + u4)  # E: Any
reveal_type(AR_f + u4)  # E: Any
