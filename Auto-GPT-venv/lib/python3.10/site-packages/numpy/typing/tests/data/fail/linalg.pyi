import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]
AR_O: npt.NDArray[np.object_]
AR_M: npt.NDArray[np.datetime64]

np.linalg.tensorsolve(AR_O, AR_O)  # E: incompatible type

np.linalg.solve(AR_O, AR_O)  # E: incompatible type

np.linalg.tensorinv(AR_O)  # E: incompatible type

np.linalg.inv(AR_O)  # E: incompatible type

np.linalg.matrix_power(AR_M, 5)  # E: incompatible type

np.linalg.cholesky(AR_O)  # E: incompatible type

np.linalg.qr(AR_O)  # E: incompatible type
np.linalg.qr(AR_f8, mode="bob")  # E: No overload variant

np.linalg.eigvals(AR_O)  # E: incompatible type

np.linalg.eigvalsh(AR_O)  # E: incompatible type
np.linalg.eigvalsh(AR_O, UPLO="bob")  # E: No overload variant

np.linalg.eig(AR_O)  # E: incompatible type

np.linalg.eigh(AR_O)  # E: incompatible type
np.linalg.eigh(AR_O, UPLO="bob")  # E: No overload variant

np.linalg.svd(AR_O)  # E: incompatible type

np.linalg.cond(AR_O)  # E: incompatible type
np.linalg.cond(AR_f8, p="bob")  # E: incompatible type

np.linalg.matrix_rank(AR_O)  # E: incompatible type

np.linalg.pinv(AR_O)  # E: incompatible type

np.linalg.slogdet(AR_O)  # E: incompatible type

np.linalg.det(AR_O)  # E: incompatible type

np.linalg.norm(AR_f8, ord="bob")  # E: No overload variant

np.linalg.multi_dot([AR_M])  # E: incompatible type
