from __future__ import annotations

from typing import Any
import numpy as np
import pytest

c16 = np.complex128(1)
f8 = np.float64(1)
i8 = np.int64(1)
u8 = np.uint64(1)

c8 = np.complex64(1)
f4 = np.float32(1)
i4 = np.int32(1)
u4 = np.uint32(1)

dt = np.datetime64(1, "D")
td = np.timedelta64(1, "D")

b_ = np.bool_(1)

b = bool(1)
c = complex(1)
f = float(1)
i = int(1)


class Object:
    def __array__(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        ret = np.empty((), dtype=object)
        ret[()] = self
        return ret

    def __sub__(self, value: Any) -> Object:
        return self

    def __rsub__(self, value: Any) -> Object:
        return self

    def __floordiv__(self, value: Any) -> Object:
        return self

    def __rfloordiv__(self, value: Any) -> Object:
        return self

    def __mul__(self, value: Any) -> Object:
        return self

    def __rmul__(self, value: Any) -> Object:
        return self

    def __pow__(self, value: Any) -> Object:
        return self

    def __rpow__(self, value: Any) -> Object:
        return self


AR_b: np.ndarray[Any, np.dtype[np.bool_]] = np.array([True])
AR_u: np.ndarray[Any, np.dtype[np.uint32]] = np.array([1], dtype=np.uint32)
AR_i: np.ndarray[Any, np.dtype[np.int64]] = np.array([1])
AR_f: np.ndarray[Any, np.dtype[np.float64]] = np.array([1.0])
AR_c: np.ndarray[Any, np.dtype[np.complex128]] = np.array([1j])
AR_m: np.ndarray[Any, np.dtype[np.timedelta64]] = np.array([np.timedelta64(1, "D")])
AR_M: np.ndarray[Any, np.dtype[np.datetime64]] = np.array([np.datetime64(1, "D")])
AR_O: np.ndarray[Any, np.dtype[np.object_]] = np.array([Object()])

AR_LIKE_b = [True]
AR_LIKE_u = [np.uint32(1)]
AR_LIKE_i = [1]
AR_LIKE_f = [1.0]
AR_LIKE_c = [1j]
AR_LIKE_m = [np.timedelta64(1, "D")]
AR_LIKE_M = [np.datetime64(1, "D")]
AR_LIKE_O = [Object()]

# Array subtractions

AR_b - AR_LIKE_u
AR_b - AR_LIKE_i
AR_b - AR_LIKE_f
AR_b - AR_LIKE_c
AR_b - AR_LIKE_m
AR_b - AR_LIKE_O

AR_LIKE_u - AR_b
AR_LIKE_i - AR_b
AR_LIKE_f - AR_b
AR_LIKE_c - AR_b
AR_LIKE_m - AR_b
AR_LIKE_M - AR_b
AR_LIKE_O - AR_b

AR_u - AR_LIKE_b
AR_u - AR_LIKE_u
AR_u - AR_LIKE_i
AR_u - AR_LIKE_f
AR_u - AR_LIKE_c
AR_u - AR_LIKE_m
AR_u - AR_LIKE_O

AR_LIKE_b - AR_u
AR_LIKE_u - AR_u
AR_LIKE_i - AR_u
AR_LIKE_f - AR_u
AR_LIKE_c - AR_u
AR_LIKE_m - AR_u
AR_LIKE_M - AR_u
AR_LIKE_O - AR_u

AR_i - AR_LIKE_b
AR_i - AR_LIKE_u
AR_i - AR_LIKE_i
AR_i - AR_LIKE_f
AR_i - AR_LIKE_c
AR_i - AR_LIKE_m
AR_i - AR_LIKE_O

AR_LIKE_b - AR_i
AR_LIKE_u - AR_i
AR_LIKE_i - AR_i
AR_LIKE_f - AR_i
AR_LIKE_c - AR_i
AR_LIKE_m - AR_i
AR_LIKE_M - AR_i
AR_LIKE_O - AR_i

AR_f - AR_LIKE_b
AR_f - AR_LIKE_u
AR_f - AR_LIKE_i
AR_f - AR_LIKE_f
AR_f - AR_LIKE_c
AR_f - AR_LIKE_O

AR_LIKE_b - AR_f
AR_LIKE_u - AR_f
AR_LIKE_i - AR_f
AR_LIKE_f - AR_f
AR_LIKE_c - AR_f
AR_LIKE_O - AR_f

AR_c - AR_LIKE_b
AR_c - AR_LIKE_u
AR_c - AR_LIKE_i
AR_c - AR_LIKE_f
AR_c - AR_LIKE_c
AR_c - AR_LIKE_O

AR_LIKE_b - AR_c
AR_LIKE_u - AR_c
AR_LIKE_i - AR_c
AR_LIKE_f - AR_c
AR_LIKE_c - AR_c
AR_LIKE_O - AR_c

AR_m - AR_LIKE_b
AR_m - AR_LIKE_u
AR_m - AR_LIKE_i
AR_m - AR_LIKE_m

AR_LIKE_b - AR_m
AR_LIKE_u - AR_m
AR_LIKE_i - AR_m
AR_LIKE_m - AR_m
AR_LIKE_M - AR_m

AR_M - AR_LIKE_b
AR_M - AR_LIKE_u
AR_M - AR_LIKE_i
AR_M - AR_LIKE_m
AR_M - AR_LIKE_M

AR_LIKE_M - AR_M

AR_O - AR_LIKE_b
AR_O - AR_LIKE_u
AR_O - AR_LIKE_i
AR_O - AR_LIKE_f
AR_O - AR_LIKE_c
AR_O - AR_LIKE_O

AR_LIKE_b - AR_O
AR_LIKE_u - AR_O
AR_LIKE_i - AR_O
AR_LIKE_f - AR_O
AR_LIKE_c - AR_O
AR_LIKE_O - AR_O

AR_u += AR_b
AR_u += AR_u
AR_u += 1  # Allowed during runtime as long as the object is 0D and >=0

# Array floor division

AR_b // AR_LIKE_b
AR_b // AR_LIKE_u
AR_b // AR_LIKE_i
AR_b // AR_LIKE_f
AR_b // AR_LIKE_O

AR_LIKE_b // AR_b
AR_LIKE_u // AR_b
AR_LIKE_i // AR_b
AR_LIKE_f // AR_b
AR_LIKE_O // AR_b

AR_u // AR_LIKE_b
AR_u // AR_LIKE_u
AR_u // AR_LIKE_i
AR_u // AR_LIKE_f
AR_u // AR_LIKE_O

AR_LIKE_b // AR_u
AR_LIKE_u // AR_u
AR_LIKE_i // AR_u
AR_LIKE_f // AR_u
AR_LIKE_m // AR_u
AR_LIKE_O // AR_u

AR_i // AR_LIKE_b
AR_i // AR_LIKE_u
AR_i // AR_LIKE_i
AR_i // AR_LIKE_f
AR_i // AR_LIKE_O

AR_LIKE_b // AR_i
AR_LIKE_u // AR_i
AR_LIKE_i // AR_i
AR_LIKE_f // AR_i
AR_LIKE_m // AR_i
AR_LIKE_O // AR_i

AR_f // AR_LIKE_b
AR_f // AR_LIKE_u
AR_f // AR_LIKE_i
AR_f // AR_LIKE_f
AR_f // AR_LIKE_O

AR_LIKE_b // AR_f
AR_LIKE_u // AR_f
AR_LIKE_i // AR_f
AR_LIKE_f // AR_f
AR_LIKE_m // AR_f
AR_LIKE_O // AR_f

AR_m // AR_LIKE_u
AR_m // AR_LIKE_i
AR_m // AR_LIKE_f
AR_m // AR_LIKE_m

AR_LIKE_m // AR_m

AR_O // AR_LIKE_b
AR_O // AR_LIKE_u
AR_O // AR_LIKE_i
AR_O // AR_LIKE_f
AR_O // AR_LIKE_O

AR_LIKE_b // AR_O
AR_LIKE_u // AR_O
AR_LIKE_i // AR_O
AR_LIKE_f // AR_O
AR_LIKE_O // AR_O

# Inplace multiplication

AR_b *= AR_LIKE_b

AR_u *= AR_LIKE_b
AR_u *= AR_LIKE_u

AR_i *= AR_LIKE_b
AR_i *= AR_LIKE_u
AR_i *= AR_LIKE_i

AR_f *= AR_LIKE_b
AR_f *= AR_LIKE_u
AR_f *= AR_LIKE_i
AR_f *= AR_LIKE_f

AR_c *= AR_LIKE_b
AR_c *= AR_LIKE_u
AR_c *= AR_LIKE_i
AR_c *= AR_LIKE_f
AR_c *= AR_LIKE_c

AR_m *= AR_LIKE_b
AR_m *= AR_LIKE_u
AR_m *= AR_LIKE_i
AR_m *= AR_LIKE_f

AR_O *= AR_LIKE_b
AR_O *= AR_LIKE_u
AR_O *= AR_LIKE_i
AR_O *= AR_LIKE_f
AR_O *= AR_LIKE_c
AR_O *= AR_LIKE_O

# Inplace power

AR_u **= AR_LIKE_b
AR_u **= AR_LIKE_u

AR_i **= AR_LIKE_b
AR_i **= AR_LIKE_u
AR_i **= AR_LIKE_i

AR_f **= AR_LIKE_b
AR_f **= AR_LIKE_u
AR_f **= AR_LIKE_i
AR_f **= AR_LIKE_f

AR_c **= AR_LIKE_b
AR_c **= AR_LIKE_u
AR_c **= AR_LIKE_i
AR_c **= AR_LIKE_f
AR_c **= AR_LIKE_c

AR_O **= AR_LIKE_b
AR_O **= AR_LIKE_u
AR_O **= AR_LIKE_i
AR_O **= AR_LIKE_f
AR_O **= AR_LIKE_c
AR_O **= AR_LIKE_O

# unary ops

-c16
-c8
-f8
-f4
-i8
-i4
with pytest.warns(RuntimeWarning):
    -u8
    -u4
-td
-AR_f

+c16
+c8
+f8
+f4
+i8
+i4
+u8
+u4
+td
+AR_f

abs(c16)
abs(c8)
abs(f8)
abs(f4)
abs(i8)
abs(i4)
abs(u8)
abs(u4)
abs(td)
abs(b_)
abs(AR_f)

# Time structures

dt + td
dt + i
dt + i4
dt + i8
dt - dt
dt - i
dt - i4
dt - i8

td + td
td + i
td + i4
td + i8
td - td
td - i
td - i4
td - i8
td / f
td / f4
td / f8
td / td
td // td
td % td


# boolean

b_ / b
b_ / b_
b_ / i
b_ / i8
b_ / i4
b_ / u8
b_ / u4
b_ / f
b_ / f8
b_ / f4
b_ / c
b_ / c16
b_ / c8

b / b_
b_ / b_
i / b_
i8 / b_
i4 / b_
u8 / b_
u4 / b_
f / b_
f8 / b_
f4 / b_
c / b_
c16 / b_
c8 / b_

# Complex

c16 + c16
c16 + f8
c16 + i8
c16 + c8
c16 + f4
c16 + i4
c16 + b_
c16 + b
c16 + c
c16 + f
c16 + i
c16 + AR_f

c16 + c16
f8 + c16
i8 + c16
c8 + c16
f4 + c16
i4 + c16
b_ + c16
b + c16
c + c16
f + c16
i + c16
AR_f + c16

c8 + c16
c8 + f8
c8 + i8
c8 + c8
c8 + f4
c8 + i4
c8 + b_
c8 + b
c8 + c
c8 + f
c8 + i
c8 + AR_f

c16 + c8
f8 + c8
i8 + c8
c8 + c8
f4 + c8
i4 + c8
b_ + c8
b + c8
c + c8
f + c8
i + c8
AR_f + c8

# Float

f8 + f8
f8 + i8
f8 + f4
f8 + i4
f8 + b_
f8 + b
f8 + c
f8 + f
f8 + i
f8 + AR_f

f8 + f8
i8 + f8
f4 + f8
i4 + f8
b_ + f8
b + f8
c + f8
f + f8
i + f8
AR_f + f8

f4 + f8
f4 + i8
f4 + f4
f4 + i4
f4 + b_
f4 + b
f4 + c
f4 + f
f4 + i
f4 + AR_f

f8 + f4
i8 + f4
f4 + f4
i4 + f4
b_ + f4
b + f4
c + f4
f + f4
i + f4
AR_f + f4

# Int

i8 + i8
i8 + u8
i8 + i4
i8 + u4
i8 + b_
i8 + b
i8 + c
i8 + f
i8 + i
i8 + AR_f

u8 + u8
u8 + i4
u8 + u4
u8 + b_
u8 + b
u8 + c
u8 + f
u8 + i
u8 + AR_f

i8 + i8
u8 + i8
i4 + i8
u4 + i8
b_ + i8
b + i8
c + i8
f + i8
i + i8
AR_f + i8

u8 + u8
i4 + u8
u4 + u8
b_ + u8
b + u8
c + u8
f + u8
i + u8
AR_f + u8

i4 + i8
i4 + i4
i4 + i
i4 + b_
i4 + b
i4 + AR_f

u4 + i8
u4 + i4
u4 + u8
u4 + u4
u4 + i
u4 + b_
u4 + b
u4 + AR_f

i8 + i4
i4 + i4
i + i4
b_ + i4
b + i4
AR_f + i4

i8 + u4
i4 + u4
u8 + u4
u4 + u4
b_ + u4
b + u4
i + u4
AR_f + u4
