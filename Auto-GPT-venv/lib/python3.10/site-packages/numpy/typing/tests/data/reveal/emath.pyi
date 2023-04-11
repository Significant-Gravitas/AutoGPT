import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
f8: np.float64
c16: np.complex128

reveal_type(np.emath.sqrt(f8))  # E: Any
reveal_type(np.emath.sqrt(AR_f8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.emath.sqrt(c16))  # E: complexfloating[Any, Any]
reveal_type(np.emath.sqrt(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.emath.log(f8))  # E: Any
reveal_type(np.emath.log(AR_f8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.emath.log(c16))  # E: complexfloating[Any, Any]
reveal_type(np.emath.log(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.emath.log10(f8))  # E: Any
reveal_type(np.emath.log10(AR_f8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.emath.log10(c16))  # E: complexfloating[Any, Any]
reveal_type(np.emath.log10(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.emath.log2(f8))  # E: Any
reveal_type(np.emath.log2(AR_f8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.emath.log2(c16))  # E: complexfloating[Any, Any]
reveal_type(np.emath.log2(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.emath.logn(f8, 2))  # E: Any
reveal_type(np.emath.logn(AR_f8, 4))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.emath.logn(f8, 1j))  # E: complexfloating[Any, Any]
reveal_type(np.emath.logn(AR_c16, 1.5))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.emath.power(f8, 2))  # E: Any
reveal_type(np.emath.power(AR_f8, 4))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.emath.power(f8, 2j))  # E: complexfloating[Any, Any]
reveal_type(np.emath.power(AR_c16, 1.5))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.emath.arccos(f8))  # E: Any
reveal_type(np.emath.arccos(AR_f8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.emath.arccos(c16))  # E: complexfloating[Any, Any]
reveal_type(np.emath.arccos(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.emath.arcsin(f8))  # E: Any
reveal_type(np.emath.arcsin(AR_f8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.emath.arcsin(c16))  # E: complexfloating[Any, Any]
reveal_type(np.emath.arcsin(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.emath.arctanh(f8))  # E: Any
reveal_type(np.emath.arctanh(AR_f8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.emath.arctanh(c16))  # E: complexfloating[Any, Any]
reveal_type(np.emath.arctanh(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
