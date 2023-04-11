import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64] = np.array([1.0])
AR_i4 = np.array([1], dtype=np.int32)
AR_u1 = np.array([1], dtype=np.uint8)

AR_LIKE_f = [1.5]
AR_LIKE_i = [1]

b_f8 = np.broadcast(AR_f8)
b_i4_f8_f8 = np.broadcast(AR_i4, AR_f8, AR_f8)

next(b_f8)
b_f8.reset()
b_f8.index
b_f8.iters
b_f8.nd
b_f8.ndim
b_f8.numiter
b_f8.shape
b_f8.size

next(b_i4_f8_f8)
b_i4_f8_f8.reset()
b_i4_f8_f8.ndim
b_i4_f8_f8.index
b_i4_f8_f8.iters
b_i4_f8_f8.nd
b_i4_f8_f8.numiter
b_i4_f8_f8.shape
b_i4_f8_f8.size

np.inner(AR_f8, AR_i4)

np.where([True, True, False])
np.where([True, True, False], 1, 0)

np.lexsort([0, 1, 2])

np.can_cast(np.dtype("i8"), int)
np.can_cast(AR_f8, "f8")
np.can_cast(AR_f8, np.complex128, casting="unsafe")

np.min_scalar_type([1])
np.min_scalar_type(AR_f8)

np.result_type(int, AR_i4)
np.result_type(AR_f8, AR_u1)
np.result_type(AR_f8, np.complex128)

np.dot(AR_LIKE_f, AR_i4)
np.dot(AR_u1, 1)
np.dot(1.5j, 1)
np.dot(AR_u1, 1, out=AR_f8)

np.vdot(AR_LIKE_f, AR_i4)
np.vdot(AR_u1, 1)
np.vdot(1.5j, 1)

np.bincount(AR_i4)

np.copyto(AR_f8, [1.6])

np.putmask(AR_f8, [True], 1.5)

np.packbits(AR_i4)
np.packbits(AR_u1)

np.unpackbits(AR_u1)

np.shares_memory(1, 2)
np.shares_memory(AR_f8, AR_f8, max_work=1)

np.may_share_memory(1, 2)
np.may_share_memory(AR_f8, AR_f8, max_work=1)
