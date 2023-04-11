import numpy as np
import numpy.typing as npt
from numpy._typing import _128Bit

f8: np.float64
f: float

# NOTE: Avoid importing the platform specific `np.float128` type
AR_i8: npt.NDArray[np.int64]
AR_i4: npt.NDArray[np.int32]
AR_f2: npt.NDArray[np.float16]
AR_f8: npt.NDArray[np.float64]
AR_f16: npt.NDArray[np.floating[_128Bit]]
AR_c8: npt.NDArray[np.complex64]
AR_c16: npt.NDArray[np.complex128]

AR_LIKE_f: list[float]

class RealObj:
    real: slice

class ImagObj:
    imag: slice

reveal_type(np.mintypecode(["f8"], typeset="qfQF"))

reveal_type(np.asfarray(AR_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.asfarray(AR_LIKE_f))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.asfarray(AR_f8, dtype="c16"))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.asfarray(AR_f8, dtype="i8"))  # E: ndarray[Any, dtype[floating[Any]]]

reveal_type(np.real(RealObj()))  # E: slice
reveal_type(np.real(AR_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.real(AR_c16))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.real(AR_LIKE_f))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.imag(ImagObj()))  # E: slice
reveal_type(np.imag(AR_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.imag(AR_c16))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.imag(AR_LIKE_f))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.iscomplex(f8))  # E: bool_
reveal_type(np.iscomplex(AR_f8))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.iscomplex(AR_LIKE_f))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.isreal(f8))  # E: bool_
reveal_type(np.isreal(AR_f8))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isreal(AR_LIKE_f))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.iscomplexobj(f8))  # E: bool
reveal_type(np.isrealobj(f8))  # E: bool

reveal_type(np.nan_to_num(f8))  # E: {float64}
reveal_type(np.nan_to_num(f, copy=True))  # E: Any
reveal_type(np.nan_to_num(AR_f8, nan=1.5))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.nan_to_num(AR_LIKE_f, posinf=9999))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.real_if_close(AR_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.real_if_close(AR_c16))  # E: Union[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{complex128}]]]
reveal_type(np.real_if_close(AR_c8))  # E: Union[ndarray[Any, dtype[{float32}]], ndarray[Any, dtype[{complex64}]]]
reveal_type(np.real_if_close(AR_LIKE_f))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.typename("h"))  # E: Literal['short']
reveal_type(np.typename("B"))  # E: Literal['unsigned char']
reveal_type(np.typename("V"))  # E: Literal['void']
reveal_type(np.typename("S1"))  # E: Literal['character']

reveal_type(np.common_type(AR_i4))  # E: Type[{float64}]
reveal_type(np.common_type(AR_f2))  # E: Type[{float16}]
reveal_type(np.common_type(AR_f2, AR_i4))  # E: Type[{float64}]
reveal_type(np.common_type(AR_f16, AR_i4))  # E: Type[{float128}]
reveal_type(np.common_type(AR_c8, AR_f2))  # E: Type[{complex64}]
reveal_type(np.common_type(AR_f2, AR_c8, AR_i4))  # E: Type[{complex128}]
