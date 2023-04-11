import sys
import datetime as dt

import pytest
import numpy as np

b =  np.bool_()
u8 = np.uint64()
i8 = np.int64()
f8 = np.float64()
c16 = np.complex128()
U = np.str_()
S = np.bytes_()


# Construction
class D:
    def __index__(self) -> int:
        return 0


class C:
    def __complex__(self) -> complex:
        return 3j


class B:
    def __int__(self) -> int:
        return 4


class A:
    def __float__(self) -> float:
        return 4.0


np.complex64(3j)
np.complex64(A())
np.complex64(C())
np.complex128(3j)
np.complex128(C())
np.complex128(None)
np.complex64("1.2")
np.complex128(b"2j")

np.int8(4)
np.int16(3.4)
np.int32(4)
np.int64(-1)
np.uint8(B())
np.uint32()
np.int32("1")
np.int64(b"2")

np.float16(A())
np.float32(16)
np.float64(3.0)
np.float64(None)
np.float32("1")
np.float16(b"2.5")

np.uint64(D())
np.float32(D())
np.complex64(D())

np.bytes_(b"hello")
np.bytes_("hello", 'utf-8')
np.bytes_("hello", encoding='utf-8')
np.str_("hello")
np.str_(b"hello", 'utf-8')
np.str_(b"hello", encoding='utf-8')

# Array-ish semantics
np.int8().real
np.int16().imag
np.int32().data
np.int64().flags

np.uint8().itemsize * 2
np.uint16().ndim + 1
np.uint32().strides
np.uint64().shape

# Time structures
np.datetime64()
np.datetime64(0, "D")
np.datetime64(0, b"D")
np.datetime64(0, ('ms', 3))
np.datetime64("2019")
np.datetime64(b"2019")
np.datetime64("2019", "D")
np.datetime64(np.datetime64())
np.datetime64(dt.datetime(2000, 5, 3))
np.datetime64(dt.date(2000, 5, 3))
np.datetime64(None)
np.datetime64(None, "D")

np.timedelta64()
np.timedelta64(0)
np.timedelta64(0, "D")
np.timedelta64(0, ('ms', 3))
np.timedelta64(0, b"D")
np.timedelta64("3")
np.timedelta64(b"5")
np.timedelta64(np.timedelta64(2))
np.timedelta64(dt.timedelta(2))
np.timedelta64(None)
np.timedelta64(None, "D")

np.void(1)
np.void(np.int64(1))
np.void(True)
np.void(np.bool_(True))
np.void(b"test")
np.void(np.bytes_("test"))
np.void(object(), [("a", "O"), ("b", "O")])
np.void(object(), dtype=[("a", "O"), ("b", "O")])

# Protocols
i8 = np.int64()
u8 = np.uint64()
f8 = np.float64()
c16 = np.complex128()
b_ = np.bool_()
td = np.timedelta64()
U = np.str_("1")
S = np.bytes_("1")
AR = np.array(1, dtype=np.float64)

int(i8)
int(u8)
int(f8)
int(b_)
int(td)
int(U)
int(S)
int(AR)
with pytest.warns(np.ComplexWarning):
    int(c16)

float(i8)
float(u8)
float(f8)
float(b_)
float(td)
float(U)
float(S)
float(AR)
with pytest.warns(np.ComplexWarning):
    float(c16)

complex(i8)
complex(u8)
complex(f8)
complex(c16)
complex(b_)
complex(td)
complex(U)
complex(AR)


# Misc
c16.dtype
c16.real
c16.imag
c16.real.real
c16.real.imag
c16.ndim
c16.size
c16.itemsize
c16.shape
c16.strides
c16.squeeze()
c16.byteswap()
c16.transpose()

# Aliases
np.string_()

np.byte()
np.short()
np.intc()
np.intp()
np.int_()
np.longlong()

np.ubyte()
np.ushort()
np.uintc()
np.uintp()
np.uint()
np.ulonglong()

np.half()
np.single()
np.double()
np.float_()
np.longdouble()
np.longfloat()

np.csingle()
np.singlecomplex()
np.cdouble()
np.complex_()
np.cfloat()
np.clongdouble()
np.clongfloat()
np.longcomplex()

b.item()
i8.item()
u8.item()
f8.item()
c16.item()
U.item()
S.item()

b.tolist()
i8.tolist()
u8.tolist()
f8.tolist()
c16.tolist()
U.tolist()
S.tolist()

b.ravel()
i8.ravel()
u8.ravel()
f8.ravel()
c16.ravel()
U.ravel()
S.ravel()

b.flatten()
i8.flatten()
u8.flatten()
f8.flatten()
c16.flatten()
U.flatten()
S.flatten()

b.reshape(1)
i8.reshape(1)
u8.reshape(1)
f8.reshape(1)
c16.reshape(1)
U.reshape(1)
S.reshape(1)
