from typing import Any
import numpy as np
import numpy.typing as npt

mat: np.matrix[Any, np.dtype[np.int64]]
ar_f8: npt.NDArray[np.float64]

reveal_type(mat * 5)  # E: matrix[Any, Any]
reveal_type(5 * mat)  # E: matrix[Any, Any]
mat *= 5

reveal_type(mat**5)  # E: matrix[Any, Any]
mat **= 5

reveal_type(mat.sum())  # E: Any
reveal_type(mat.mean())  # E: Any
reveal_type(mat.std())  # E: Any
reveal_type(mat.var())  # E: Any
reveal_type(mat.prod())  # E: Any
reveal_type(mat.any())  # E: bool_
reveal_type(mat.all())  # E: bool_
reveal_type(mat.max())  # E: {int64}
reveal_type(mat.min())  # E: {int64}
reveal_type(mat.argmax())  # E: {intp}
reveal_type(mat.argmin())  # E: {intp}
reveal_type(mat.ptp())  # E: {int64}

reveal_type(mat.sum(axis=0))  # E: matrix[Any, Any]
reveal_type(mat.mean(axis=0))  # E: matrix[Any, Any]
reveal_type(mat.std(axis=0))  # E: matrix[Any, Any]
reveal_type(mat.var(axis=0))  # E: matrix[Any, Any]
reveal_type(mat.prod(axis=0))  # E: matrix[Any, Any]
reveal_type(mat.any(axis=0))  # E: matrix[Any, dtype[bool_]]
reveal_type(mat.all(axis=0))  # E: matrix[Any, dtype[bool_]]
reveal_type(mat.max(axis=0))  # E: matrix[Any, dtype[{int64}]]
reveal_type(mat.min(axis=0))  # E: matrix[Any, dtype[{int64}]]
reveal_type(mat.argmax(axis=0))  # E: matrix[Any, dtype[{intp}]]
reveal_type(mat.argmin(axis=0))  # E: matrix[Any, dtype[{intp}]]
reveal_type(mat.ptp(axis=0))  # E: matrix[Any, dtype[{int64}]]

reveal_type(mat.sum(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(mat.mean(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(mat.std(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(mat.var(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(mat.prod(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(mat.any(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(mat.all(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(mat.max(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(mat.min(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(mat.argmax(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(mat.argmin(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(mat.ptp(out=ar_f8))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(mat.T)  # E: matrix[Any, dtype[{int64}]]
reveal_type(mat.I)  # E: matrix[Any, Any]
reveal_type(mat.A)  # E: ndarray[Any, dtype[{int64}]]
reveal_type(mat.A1)  # E: ndarray[Any, dtype[{int64}]]
reveal_type(mat.H)  # E: matrix[Any, dtype[{int64}]]
reveal_type(mat.getT())  # E: matrix[Any, dtype[{int64}]]
reveal_type(mat.getI())  # E: matrix[Any, Any]
reveal_type(mat.getA())  # E: ndarray[Any, dtype[{int64}]]
reveal_type(mat.getA1())  # E: ndarray[Any, dtype[{int64}]]
reveal_type(mat.getH())  # E: matrix[Any, dtype[{int64}]]

reveal_type(np.bmat(ar_f8))  # E: matrix[Any, Any]
reveal_type(np.bmat([[0, 1, 2]]))  # E: matrix[Any, Any]
reveal_type(np.bmat("mat"))  # E: matrix[Any, Any]

reveal_type(np.asmatrix(ar_f8, dtype=np.int64))  # E: matrix[Any, Any]
