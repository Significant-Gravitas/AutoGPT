import numpy as np

f8 = np.float64(1)
i8 = np.int64(1)
u8 = np.uint64(1)

f4 = np.float32(1)
i4 = np.int32(1)
u4 = np.uint32(1)

td = np.timedelta64(1, "D")
b_ = np.bool_(1)

b = bool(1)
f = float(1)
i = int(1)

AR = np.array([1], dtype=np.bool_)
AR.setflags(write=False)

AR2 = np.array([1], dtype=np.timedelta64)
AR2.setflags(write=False)

# Time structures

td % td
td % AR2
AR2 % td

divmod(td, td)
divmod(td, AR2)
divmod(AR2, td)

# Bool

b_ % b
b_ % i
b_ % f
b_ % b_
b_ % i8
b_ % u8
b_ % f8
b_ % AR

divmod(b_, b)
divmod(b_, i)
divmod(b_, f)
divmod(b_, b_)
divmod(b_, i8)
divmod(b_, u8)
divmod(b_, f8)
divmod(b_, AR)

b % b_
i % b_
f % b_
b_ % b_
i8 % b_
u8 % b_
f8 % b_
AR % b_

divmod(b, b_)
divmod(i, b_)
divmod(f, b_)
divmod(b_, b_)
divmod(i8, b_)
divmod(u8, b_)
divmod(f8, b_)
divmod(AR, b_)

# int

i8 % b
i8 % i
i8 % f
i8 % i8
i8 % f8
i4 % i8
i4 % f8
i4 % i4
i4 % f4
i8 % AR

divmod(i8, b)
divmod(i8, i)
divmod(i8, f)
divmod(i8, i8)
divmod(i8, f8)
divmod(i8, i4)
divmod(i8, f4)
divmod(i4, i4)
divmod(i4, f4)
divmod(i8, AR)

b % i8
i % i8
f % i8
i8 % i8
f8 % i8
i8 % i4
f8 % i4
i4 % i4
f4 % i4
AR % i8

divmod(b, i8)
divmod(i, i8)
divmod(f, i8)
divmod(i8, i8)
divmod(f8, i8)
divmod(i4, i8)
divmod(f4, i8)
divmod(i4, i4)
divmod(f4, i4)
divmod(AR, i8)

# float

f8 % b
f8 % i
f8 % f
i8 % f4
f4 % f4
f8 % AR

divmod(f8, b)
divmod(f8, i)
divmod(f8, f)
divmod(f8, f8)
divmod(f8, f4)
divmod(f4, f4)
divmod(f8, AR)

b % f8
i % f8
f % f8
f8 % f8
f8 % f8
f4 % f4
AR % f8

divmod(b, f8)
divmod(i, f8)
divmod(f, f8)
divmod(f8, f8)
divmod(f4, f8)
divmod(f4, f4)
divmod(AR, f8)
