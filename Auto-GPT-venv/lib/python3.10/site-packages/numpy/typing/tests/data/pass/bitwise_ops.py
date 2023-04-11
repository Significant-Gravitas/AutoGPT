import numpy as np

i8 = np.int64(1)
u8 = np.uint64(1)

i4 = np.int32(1)
u4 = np.uint32(1)

b_ = np.bool_(1)

b = bool(1)
i = int(1)

AR = np.array([0, 1, 2], dtype=np.int32)
AR.setflags(write=False)


i8 << i8
i8 >> i8
i8 | i8
i8 ^ i8
i8 & i8

i8 << AR
i8 >> AR
i8 | AR
i8 ^ AR
i8 & AR

i4 << i4
i4 >> i4
i4 | i4
i4 ^ i4
i4 & i4

i8 << i4
i8 >> i4
i8 | i4
i8 ^ i4
i8 & i4

i8 << i
i8 >> i
i8 | i
i8 ^ i
i8 & i

i8 << b_
i8 >> b_
i8 | b_
i8 ^ b_
i8 & b_

i8 << b
i8 >> b
i8 | b
i8 ^ b
i8 & b

u8 << u8
u8 >> u8
u8 | u8
u8 ^ u8
u8 & u8

u8 << AR
u8 >> AR
u8 | AR
u8 ^ AR
u8 & AR

u4 << u4
u4 >> u4
u4 | u4
u4 ^ u4
u4 & u4

u4 << i4
u4 >> i4
u4 | i4
u4 ^ i4
u4 & i4

u4 << i
u4 >> i
u4 | i
u4 ^ i
u4 & i

u8 << b_
u8 >> b_
u8 | b_
u8 ^ b_
u8 & b_

u8 << b
u8 >> b
u8 | b
u8 ^ b
u8 & b

b_ << b_
b_ >> b_
b_ | b_
b_ ^ b_
b_ & b_

b_ << AR
b_ >> AR
b_ | AR
b_ ^ AR
b_ & AR

b_ << b
b_ >> b
b_ | b
b_ ^ b
b_ & b

b_ << i
b_ >> i
b_ | i
b_ ^ i
b_ & i

~i8
~i4
~u8
~u4
~b_
~AR
