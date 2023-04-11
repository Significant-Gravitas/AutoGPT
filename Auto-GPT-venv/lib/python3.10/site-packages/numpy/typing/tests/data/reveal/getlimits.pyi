import numpy as np
f: float
f8: np.float64
c8: np.complex64

i: int
i8: np.int64
u4: np.uint32

finfo_f8: np.finfo[np.float64]
iinfo_i8: np.iinfo[np.int64]

reveal_type(np.finfo(f))  # E: finfo[{double}]
reveal_type(np.finfo(f8))  # E: finfo[{float64}]
reveal_type(np.finfo(c8))  # E: finfo[{float32}]
reveal_type(np.finfo('f2'))  # E: finfo[floating[Any]]

reveal_type(finfo_f8.dtype)  # E: dtype[{float64}]
reveal_type(finfo_f8.bits)  # E: int
reveal_type(finfo_f8.eps)  # E: {float64}
reveal_type(finfo_f8.epsneg)  # E: {float64}
reveal_type(finfo_f8.iexp)  # E: int
reveal_type(finfo_f8.machep)  # E: int
reveal_type(finfo_f8.max)  # E: {float64}
reveal_type(finfo_f8.maxexp)  # E: int
reveal_type(finfo_f8.min)  # E: {float64}
reveal_type(finfo_f8.minexp)  # E: int
reveal_type(finfo_f8.negep)  # E: int
reveal_type(finfo_f8.nexp)  # E: int
reveal_type(finfo_f8.nmant)  # E: int
reveal_type(finfo_f8.precision)  # E: int
reveal_type(finfo_f8.resolution)  # E: {float64}
reveal_type(finfo_f8.tiny)  # E: {float64}
reveal_type(finfo_f8.smallest_normal)  # E: {float64}
reveal_type(finfo_f8.smallest_subnormal)  # E: {float64}

reveal_type(np.iinfo(i))  # E: iinfo[{int_}]
reveal_type(np.iinfo(i8))  # E: iinfo[{int64}]
reveal_type(np.iinfo(u4))  # E: iinfo[{uint32}]
reveal_type(np.iinfo('i2'))  # E: iinfo[Any]

reveal_type(iinfo_i8.dtype)  # E: dtype[{int64}]
reveal_type(iinfo_i8.kind)  # E: str
reveal_type(iinfo_i8.bits)  # E: int
reveal_type(iinfo_i8.key)  # E: str
reveal_type(iinfo_i8.min)  # E: int
reveal_type(iinfo_i8.max)  # E: int
