import numpy as np
import numpy.typing as npt

AR_U: npt.NDArray[np.str_]

def func() -> bool: ...

np.testing.assert_(True, msg=1)  # E: incompatible type
np.testing.build_err_msg(1, "test")  # E: incompatible type
np.testing.assert_almost_equal(AR_U, AR_U)  # E: incompatible type
np.testing.assert_approx_equal([1, 2, 3], [1, 2, 3])  # E: incompatible type
np.testing.assert_array_almost_equal(AR_U, AR_U)  # E: incompatible type
np.testing.assert_array_less(AR_U, AR_U)  # E: incompatible type
np.testing.assert_string_equal(b"a", b"a")  # E: incompatible type

np.testing.assert_raises(expected_exception=TypeError, callable=func)  # E: No overload variant
np.testing.assert_raises_regex(expected_exception=TypeError, expected_regex="T", callable=func)  # E: No overload variant

np.testing.assert_allclose(AR_U, AR_U)  # E: incompatible type
np.testing.assert_array_almost_equal_nulp(AR_U, AR_U)  # E: incompatible type
np.testing.assert_array_max_ulp(AR_U, AR_U)  # E: incompatible type

np.testing.assert_warns(warning_class=RuntimeWarning, func=func)  # E: No overload variant
np.testing.assert_no_warnings(func=func)  # E: No overload variant
np.testing.assert_no_warnings(func, None)  # E: Too many arguments
np.testing.assert_no_warnings(func, test=None)  # E: Unexpected keyword argument

np.testing.assert_no_gc_cycles(func=func)  # E: No overload variant
