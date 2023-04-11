import numpy as np
import numpy.typing as npt

i8: np.int64

AR_b: npt.NDArray[np.bool_]
AR_u1: npt.NDArray[np.uint8]
AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_M: npt.NDArray[np.datetime64]

M: np.datetime64

AR_LIKE_f: list[float]

def func(a: int) -> None: ...

np.where(AR_b, 1)  # E: No overload variant

np.can_cast(AR_f8, 1)  # E: incompatible type

np.vdot(AR_M, AR_M)  # E: incompatible type

np.copyto(AR_LIKE_f, AR_f8)  # E: incompatible type

np.putmask(AR_LIKE_f, [True, True, False], 1.5)  # E: incompatible type

np.packbits(AR_f8)  # E: incompatible type
np.packbits(AR_u1, bitorder=">")  # E: incompatible type

np.unpackbits(AR_i8)  # E: incompatible type
np.unpackbits(AR_u1, bitorder=">")  # E: incompatible type

np.shares_memory(1, 1, max_work=i8)  # E: incompatible type
np.may_share_memory(1, 1, max_work=i8)  # E: incompatible type

np.arange(M)  # E: No overload variant
np.arange(stop=10)  # E: No overload variant

np.datetime_data(int)  # E: incompatible type

np.busday_offset("2012", 10)  # E: No overload variant

np.datetime_as_string("2012")  # E: No overload variant

np.compare_chararrays("a", b"a", "==", False)  # E: No overload variant

np.add_docstring(func, None)  # E: incompatible type

np.nested_iters([AR_i8, AR_i8])  # E: Missing positional argument
np.nested_iters([AR_i8, AR_i8], 0)  # E: incompatible type
np.nested_iters([AR_i8, AR_i8], [0])  # E: incompatible type
np.nested_iters([AR_i8, AR_i8], [[0], [1]], flags=["test"])  # E: incompatible type
np.nested_iters([AR_i8, AR_i8], [[0], [1]], op_flags=[["test"]])  # E: incompatible type
np.nested_iters([AR_i8, AR_i8], [[0], [1]], buffersize=1.0)  # E: incompatible type
