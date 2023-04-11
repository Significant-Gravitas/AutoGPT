from typing import Any

import numpy as np
import numpy.typing as npt

vectorized_func: np.vectorize

f8: np.float64
AR_LIKE_f8: list[float]

AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
AR_m: npt.NDArray[np.timedelta64]
AR_M: npt.NDArray[np.datetime64]
AR_O: npt.NDArray[np.object_]
AR_b: npt.NDArray[np.bool_]
AR_U: npt.NDArray[np.str_]
CHAR_AR_U: np.chararray[Any, np.dtype[np.str_]]

def func(*args: Any, **kwargs: Any) -> Any: ...

reveal_type(vectorized_func.pyfunc)  #  E: def (*Any, **Any) -> Any
reveal_type(vectorized_func.cache)  # E: bool
reveal_type(vectorized_func.signature)  # E: Union[None, builtins.str]
reveal_type(vectorized_func.otypes)  # E: Union[None, builtins.str]
reveal_type(vectorized_func.excluded)  # E: set[Union[builtins.int, builtins.str]]
reveal_type(vectorized_func.__doc__)  # E: Union[None, builtins.str]
reveal_type(vectorized_func([1]))  # E: Any
reveal_type(np.vectorize(int))  # E: vectorize
reveal_type(np.vectorize(  # E: vectorize
    int, otypes="i", doc="doc", excluded=(), cache=True, signature=None
))

reveal_type(np.add_newdoc("__main__", "blabla", doc="test doc"))  # E: None
reveal_type(np.add_newdoc("__main__", "blabla", doc=("meth", "test doc")))  # E: None
reveal_type(np.add_newdoc("__main__", "blabla", doc=[("meth", "test doc")]))  # E: None

reveal_type(np.rot90(AR_f8, k=2))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.rot90(AR_LIKE_f8, axes=(0, 1)))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.flip(f8))  # E: {float64}
reveal_type(np.flip(1.0))  # E: Any
reveal_type(np.flip(AR_f8, axis=(0, 1)))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.flip(AR_LIKE_f8, axis=0))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.iterable(1))  # E: bool
reveal_type(np.iterable([1]))  # E: bool

reveal_type(np.average(AR_f8))  # E: floating[Any]
reveal_type(np.average(AR_f8, weights=AR_c16))  # E: complexfloating[Any, Any]
reveal_type(np.average(AR_O))  # E: Any
reveal_type(np.average(AR_f8, returned=True))  # E: Tuple[floating[Any], floating[Any]]
reveal_type(np.average(AR_f8, weights=AR_c16, returned=True))  # E: Tuple[complexfloating[Any, Any], complexfloating[Any, Any]]
reveal_type(np.average(AR_O, returned=True))  # E: Tuple[Any, Any]
reveal_type(np.average(AR_f8, axis=0))  # E: Any
reveal_type(np.average(AR_f8, axis=0, returned=True))  # E: Tuple[Any, Any]

reveal_type(np.asarray_chkfinite(AR_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.asarray_chkfinite(AR_LIKE_f8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.asarray_chkfinite(AR_f8, dtype=np.float64))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.asarray_chkfinite(AR_f8, dtype=float))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.piecewise(AR_f8, AR_b, [func]))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.piecewise(AR_LIKE_f8, AR_b, [func]))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.select([AR_f8], [AR_f8]))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.copy(AR_LIKE_f8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.copy(AR_U))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.copy(CHAR_AR_U))  # E: ndarray[Any, Any]
reveal_type(np.copy(CHAR_AR_U, "K", subok=True))  # E: chararray[Any, dtype[str_]]
reveal_type(np.copy(CHAR_AR_U, subok=True))  # E: chararray[Any, dtype[str_]]

reveal_type(np.gradient(AR_f8, axis=None))  # E: Any
reveal_type(np.gradient(AR_LIKE_f8, edge_order=2))  # E: Any

reveal_type(np.diff("bob", n=0))  # E: str
reveal_type(np.diff(AR_f8, axis=0))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.diff(AR_LIKE_f8, prepend=1.5))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.angle(f8))  # E: floating[Any]
reveal_type(np.angle(AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.angle(AR_c16, deg=True))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.angle(AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.unwrap(AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.unwrap(AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.sort_complex(AR_f8))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.trim_zeros(AR_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.trim_zeros(AR_LIKE_f8))  # E: list[builtins.float]

reveal_type(np.extract(AR_i8, AR_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.extract(AR_i8, AR_LIKE_f8))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.place(AR_f8, mask=AR_i8, vals=5.0))  # E: None

reveal_type(np.disp(1, linefeed=True))  # E: None
with open("test", "w") as f:
    reveal_type(np.disp("message", device=f))  # E: None

reveal_type(np.cov(AR_f8, bias=True))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.cov(AR_f8, AR_c16, ddof=1))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.cov(AR_f8, aweights=AR_f8, dtype=np.float32))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.cov(AR_f8, fweights=AR_f8, dtype=float))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.corrcoef(AR_f8, rowvar=True))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.corrcoef(AR_f8, AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.corrcoef(AR_f8, dtype=np.float32))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.corrcoef(AR_f8, dtype=float))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.blackman(5))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.bartlett(6))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.hanning(4.5))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.hamming(0))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.i0(AR_i8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.kaiser(4, 5.9))  # E: ndarray[Any, dtype[floating[Any]]]

reveal_type(np.sinc(1.0))  # E: floating[Any]
reveal_type(np.sinc(1j))  # E: complexfloating[Any, Any]
reveal_type(np.sinc(AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.sinc(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.median(AR_f8, keepdims=False))  # E: floating[Any]
reveal_type(np.median(AR_c16, overwrite_input=True))  # E: complexfloating[Any, Any]
reveal_type(np.median(AR_m))  # E: timedelta64
reveal_type(np.median(AR_O))  # E: Any
reveal_type(np.median(AR_f8, keepdims=True))  # E: Any
reveal_type(np.median(AR_c16, axis=0))  # E: Any
reveal_type(np.median(AR_LIKE_f8, out=AR_c16))  # E: ndarray[Any, dtype[{complex128}]]

reveal_type(np.add_newdoc_ufunc(np.add, "docstring"))  # E: None

reveal_type(np.percentile(AR_f8, 50))  # E: floating[Any]
reveal_type(np.percentile(AR_c16, 50))  # E: complexfloating[Any, Any]
reveal_type(np.percentile(AR_m, 50))  # E: timedelta64
reveal_type(np.percentile(AR_M, 50, overwrite_input=True))  # E: datetime64
reveal_type(np.percentile(AR_O, 50))  # E: Any
reveal_type(np.percentile(AR_f8, [50]))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.percentile(AR_c16, [50]))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.percentile(AR_m, [50]))  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(np.percentile(AR_M, [50], method="nearest"))  # E: ndarray[Any, dtype[datetime64]]
reveal_type(np.percentile(AR_O, [50]))  # E: ndarray[Any, dtype[object_]]
reveal_type(np.percentile(AR_f8, [50], keepdims=True))  # E: Any
reveal_type(np.percentile(AR_f8, [50], axis=[1]))  # E: Any
reveal_type(np.percentile(AR_f8, [50], out=AR_c16))  # E: ndarray[Any, dtype[{complex128}]]

reveal_type(np.quantile(AR_f8, 0.5))  # E: floating[Any]
reveal_type(np.quantile(AR_c16, 0.5))  # E: complexfloating[Any, Any]
reveal_type(np.quantile(AR_m, 0.5))  # E: timedelta64
reveal_type(np.quantile(AR_M, 0.5, overwrite_input=True))  # E: datetime64
reveal_type(np.quantile(AR_O, 0.5))  # E: Any
reveal_type(np.quantile(AR_f8, [0.5]))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.quantile(AR_c16, [0.5]))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.quantile(AR_m, [0.5]))  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(np.quantile(AR_M, [0.5], method="nearest"))  # E: ndarray[Any, dtype[datetime64]]
reveal_type(np.quantile(AR_O, [0.5]))  # E: ndarray[Any, dtype[object_]]
reveal_type(np.quantile(AR_f8, [0.5], keepdims=True))  # E: Any
reveal_type(np.quantile(AR_f8, [0.5], axis=[1]))  # E: Any
reveal_type(np.quantile(AR_f8, [0.5], out=AR_c16))  # E: ndarray[Any, dtype[{complex128}]]

reveal_type(np.meshgrid(AR_f8, AR_i8, copy=False))  # E: list[ndarray[Any, dtype[Any]]]
reveal_type(np.meshgrid(AR_f8, AR_i8, AR_c16, indexing="ij"))  # E: list[ndarray[Any, dtype[Any]]]

reveal_type(np.delete(AR_f8, np.s_[:5]))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.delete(AR_LIKE_f8, [0, 4, 9], axis=0))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.insert(AR_f8, np.s_[:5], 5))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.insert(AR_LIKE_f8, [0, 4, 9], [0.5, 9.2, 7], axis=0))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.append(AR_f8, 5))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.append(AR_LIKE_f8, 1j, axis=0))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.digitize(4.5, [1]))  # E: {intp}
reveal_type(np.digitize(AR_f8, [1, 2, 3]))  # E: ndarray[Any, dtype[{intp}]]
