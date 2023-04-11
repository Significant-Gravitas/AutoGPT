import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
AR_O: npt.NDArray[np.object_]
AR_U: npt.NDArray[np.str_]

poly_obj: np.poly1d

np.polyint(AR_U)  # E: incompatible type
np.polyint(AR_f8, m=1j)  # E: No overload variant

np.polyder(AR_U)  # E: incompatible type
np.polyder(AR_f8, m=1j)  # E: No overload variant

np.polyfit(AR_O, AR_f8, 1)  # E: incompatible type
np.polyfit(AR_f8, AR_f8, 1, rcond=1j)  # E: No overload variant
np.polyfit(AR_f8, AR_f8, 1, w=AR_c16)  # E: incompatible type
np.polyfit(AR_f8, AR_f8, 1, cov="bob")  # E: No overload variant

np.polyval(AR_f8, AR_U)  # E: incompatible type
np.polyadd(AR_f8, AR_U)  # E: incompatible type
np.polysub(AR_f8, AR_U)  # E: incompatible type
np.polymul(AR_f8, AR_U)  # E: incompatible type
np.polydiv(AR_f8, AR_U)  # E: incompatible type

5**poly_obj  # E: No overload variant
hash(poly_obj)
