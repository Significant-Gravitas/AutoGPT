import numpy as np
import numpy.typing as npt

AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
AR_O: npt.NDArray[np.object_]
AR_m: npt.NDArray[np.timedelta64]
AR_S: npt.NDArray[np.str_]

reveal_type(np.linalg.tensorsolve(AR_i8, AR_i8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.linalg.tensorsolve(AR_i8, AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.linalg.tensorsolve(AR_c16, AR_f8))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.linalg.solve(AR_i8, AR_i8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.linalg.solve(AR_i8, AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.linalg.solve(AR_c16, AR_f8))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.linalg.tensorinv(AR_i8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.linalg.tensorinv(AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.linalg.tensorinv(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.linalg.inv(AR_i8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.linalg.inv(AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.linalg.inv(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.linalg.matrix_power(AR_i8, -1))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.linalg.matrix_power(AR_f8, 0))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.linalg.matrix_power(AR_c16, 1))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.linalg.matrix_power(AR_O, 2))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.linalg.cholesky(AR_i8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.linalg.cholesky(AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.linalg.cholesky(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.linalg.qr(AR_i8))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{float64}]]]
reveal_type(np.linalg.qr(AR_f8))  # E: Tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]
reveal_type(np.linalg.qr(AR_c16))  # E: Tuple[ndarray[Any, dtype[complexfloating[Any, Any]]], ndarray[Any, dtype[complexfloating[Any, Any]]]]

reveal_type(np.linalg.eigvals(AR_i8))  # E: Union[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{complex128}]]]
reveal_type(np.linalg.eigvals(AR_f8))  # E: Union[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[complexfloating[Any, Any]]]]
reveal_type(np.linalg.eigvals(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.linalg.eigvalsh(AR_i8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.linalg.eigvalsh(AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.linalg.eigvalsh(AR_c16))  # E: ndarray[Any, dtype[floating[Any]]]

reveal_type(np.linalg.eig(AR_i8))  # E: Union[Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{float64}]]], Tuple[ndarray[Any, dtype[{complex128}]], ndarray[Any, dtype[{complex128}]]]]
reveal_type(np.linalg.eig(AR_f8))  # E: Union[Tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]], Tuple[ndarray[Any, dtype[complexfloating[Any, Any]]], ndarray[Any, dtype[complexfloating[Any, Any]]]]]
reveal_type(np.linalg.eig(AR_c16))  # E: Tuple[ndarray[Any, dtype[complexfloating[Any, Any]]], ndarray[Any, dtype[complexfloating[Any, Any]]]]

reveal_type(np.linalg.eigh(AR_i8))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{float64}]]]
reveal_type(np.linalg.eigh(AR_f8))  # E: Tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]
reveal_type(np.linalg.eigh(AR_c16))  # E: Tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[complexfloating[Any, Any]]]]

reveal_type(np.linalg.svd(AR_i8))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{float64}]]]
reveal_type(np.linalg.svd(AR_f8))  # E: Tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]
reveal_type(np.linalg.svd(AR_c16))  # E: Tuple[ndarray[Any, dtype[complexfloating[Any, Any]]], ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[complexfloating[Any, Any]]]]
reveal_type(np.linalg.svd(AR_i8, compute_uv=False))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.linalg.svd(AR_f8, compute_uv=False))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.linalg.svd(AR_c16, compute_uv=False))  # E: ndarray[Any, dtype[floating[Any]]]

reveal_type(np.linalg.cond(AR_i8))  # E: Any
reveal_type(np.linalg.cond(AR_f8))  # E: Any
reveal_type(np.linalg.cond(AR_c16))  # E: Any

reveal_type(np.linalg.matrix_rank(AR_i8))  # E: Any
reveal_type(np.linalg.matrix_rank(AR_f8))  # E: Any
reveal_type(np.linalg.matrix_rank(AR_c16))  # E: Any

reveal_type(np.linalg.pinv(AR_i8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.linalg.pinv(AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.linalg.pinv(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]

reveal_type(np.linalg.slogdet(AR_i8))  # E: Tuple[Any, Any]
reveal_type(np.linalg.slogdet(AR_f8))  # E: Tuple[Any, Any]
reveal_type(np.linalg.slogdet(AR_c16))  # E: Tuple[Any, Any]

reveal_type(np.linalg.det(AR_i8))  # E: Any
reveal_type(np.linalg.det(AR_f8))  # E: Any
reveal_type(np.linalg.det(AR_c16))  # E: Any

reveal_type(np.linalg.lstsq(AR_i8, AR_i8))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{float64}]], {int32}, ndarray[Any, dtype[{float64}]]]
reveal_type(np.linalg.lstsq(AR_i8, AR_f8))  # E: Tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]], {int32}, ndarray[Any, dtype[floating[Any]]]]
reveal_type(np.linalg.lstsq(AR_f8, AR_c16))  # E: Tuple[ndarray[Any, dtype[complexfloating[Any, Any]]], ndarray[Any, dtype[floating[Any]]], {int32}, ndarray[Any, dtype[floating[Any]]]]

reveal_type(np.linalg.norm(AR_i8))  # E: floating[Any]
reveal_type(np.linalg.norm(AR_f8))  # E: floating[Any]
reveal_type(np.linalg.norm(AR_c16))  # E: floating[Any]
reveal_type(np.linalg.norm(AR_S))  # E: floating[Any]
reveal_type(np.linalg.norm(AR_f8, axis=0))  # E: Any

reveal_type(np.linalg.multi_dot([AR_i8, AR_i8]))  # E: Any
reveal_type(np.linalg.multi_dot([AR_i8, AR_f8]))  # E: Any
reveal_type(np.linalg.multi_dot([AR_f8, AR_c16]))  # E: Any
reveal_type(np.linalg.multi_dot([AR_O, AR_O]))  # E: Any
reveal_type(np.linalg.multi_dot([AR_m, AR_m]))  # E: Any
