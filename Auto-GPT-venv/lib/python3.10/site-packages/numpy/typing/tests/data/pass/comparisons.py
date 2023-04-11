from __future__ import annotations

from typing import Any
import numpy as np

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

SEQ = (0, 1, 2, 3, 4)

AR_b: np.ndarray[Any, np.dtype[np.bool_]] = np.array([True])
AR_u: np.ndarray[Any, np.dtype[np.uint32]] = np.array([1], dtype=np.uint32)
AR_i: np.ndarray[Any, np.dtype[np.int_]] = np.array([1])
AR_f: np.ndarray[Any, np.dtype[np.float_]] = np.array([1.0])
AR_c: np.ndarray[Any, np.dtype[np.complex_]] = np.array([1.0j])
AR_m: np.ndarray[Any, np.dtype[np.timedelta64]] = np.array([np.timedelta64("1")])
AR_M: np.ndarray[Any, np.dtype[np.datetime64]] = np.array([np.datetime64("1")])
AR_O: np.ndarray[Any, np.dtype[np.object_]] = np.array([1], dtype=object)

# Arrays

AR_b > AR_b
AR_b > AR_u
AR_b > AR_i
AR_b > AR_f
AR_b > AR_c

AR_u > AR_b
AR_u > AR_u
AR_u > AR_i
AR_u > AR_f
AR_u > AR_c

AR_i > AR_b
AR_i > AR_u
AR_i > AR_i
AR_i > AR_f
AR_i > AR_c

AR_f > AR_b
AR_f > AR_u
AR_f > AR_i
AR_f > AR_f
AR_f > AR_c

AR_c > AR_b
AR_c > AR_u
AR_c > AR_i
AR_c > AR_f
AR_c > AR_c

AR_m > AR_b
AR_m > AR_u
AR_m > AR_i
AR_b > AR_m
AR_u > AR_m
AR_i > AR_m

AR_M > AR_M

AR_O > AR_O
1 > AR_O
AR_O > 1

# Time structures

dt > dt

td > td
td > i
td > i4
td > i8
td > AR_i
td > SEQ

# boolean

b_ > b
b_ > b_
b_ > i
b_ > i8
b_ > i4
b_ > u8
b_ > u4
b_ > f
b_ > f8
b_ > f4
b_ > c
b_ > c16
b_ > c8
b_ > AR_i
b_ > SEQ

# Complex

c16 > c16
c16 > f8
c16 > i8
c16 > c8
c16 > f4
c16 > i4
c16 > b_
c16 > b
c16 > c
c16 > f
c16 > i
c16 > AR_i
c16 > SEQ

c16 > c16
f8 > c16
i8 > c16
c8 > c16
f4 > c16
i4 > c16
b_ > c16
b > c16
c > c16
f > c16
i > c16
AR_i > c16
SEQ > c16

c8 > c16
c8 > f8
c8 > i8
c8 > c8
c8 > f4
c8 > i4
c8 > b_
c8 > b
c8 > c
c8 > f
c8 > i
c8 > AR_i
c8 > SEQ

c16 > c8
f8 > c8
i8 > c8
c8 > c8
f4 > c8
i4 > c8
b_ > c8
b > c8
c > c8
f > c8
i > c8
AR_i > c8
SEQ > c8

# Float

f8 > f8
f8 > i8
f8 > f4
f8 > i4
f8 > b_
f8 > b
f8 > c
f8 > f
f8 > i
f8 > AR_i
f8 > SEQ

f8 > f8
i8 > f8
f4 > f8
i4 > f8
b_ > f8
b > f8
c > f8
f > f8
i > f8
AR_i > f8
SEQ > f8

f4 > f8
f4 > i8
f4 > f4
f4 > i4
f4 > b_
f4 > b
f4 > c
f4 > f
f4 > i
f4 > AR_i
f4 > SEQ

f8 > f4
i8 > f4
f4 > f4
i4 > f4
b_ > f4
b > f4
c > f4
f > f4
i > f4
AR_i > f4
SEQ > f4

# Int

i8 > i8
i8 > u8
i8 > i4
i8 > u4
i8 > b_
i8 > b
i8 > c
i8 > f
i8 > i
i8 > AR_i
i8 > SEQ

u8 > u8
u8 > i4
u8 > u4
u8 > b_
u8 > b
u8 > c
u8 > f
u8 > i
u8 > AR_i
u8 > SEQ

i8 > i8
u8 > i8
i4 > i8
u4 > i8
b_ > i8
b > i8
c > i8
f > i8
i > i8
AR_i > i8
SEQ > i8

u8 > u8
i4 > u8
u4 > u8
b_ > u8
b > u8
c > u8
f > u8
i > u8
AR_i > u8
SEQ > u8

i4 > i8
i4 > i4
i4 > i
i4 > b_
i4 > b
i4 > AR_i
i4 > SEQ

u4 > i8
u4 > i4
u4 > u8
u4 > u4
u4 > i
u4 > b_
u4 > b
u4 > AR_i
u4 > SEQ

i8 > i4
i4 > i4
i > i4
b_ > i4
b > i4
AR_i > i4
SEQ > i4

i8 > u4
i4 > u4
u8 > u4
u4 > u4
b_ > u4
b > u4
i > u4
AR_i > u4
SEQ > u4
