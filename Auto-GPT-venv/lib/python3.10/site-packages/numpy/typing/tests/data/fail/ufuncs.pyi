import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]

np.sin.nin + "foo"  # E: Unsupported operand types
np.sin(1, foo="bar")  # E: No overload variant

np.abs(None)  # E: No overload variant

np.add(1, 1, 1)  # E: No overload variant
np.add(1, 1, axis=0)  # E: No overload variant

np.matmul(AR_f8, AR_f8, where=True)  # E: No overload variant

np.frexp(AR_f8, out=None)  # E: No overload variant
np.frexp(AR_f8, out=AR_f8)  # E: No overload variant

np.absolute.outer()  # E: "None" not callable
np.frexp.outer()  # E: "None" not callable
np.divmod.outer()  # E: "None" not callable
np.matmul.outer()  # E: "None" not callable

np.absolute.reduceat()  # E: "None" not callable
np.frexp.reduceat()  # E: "None" not callable
np.divmod.reduceat()  # E: "None" not callable
np.matmul.reduceat()  # E: "None" not callable

np.absolute.reduce()  # E: "None" not callable
np.frexp.reduce()  # E: "None" not callable
np.divmod.reduce()  # E: "None" not callable
np.matmul.reduce()  # E: "None" not callable

np.absolute.accumulate()  # E: "None" not callable
np.frexp.accumulate()  # E: "None" not callable
np.divmod.accumulate()  # E: "None" not callable
np.matmul.accumulate()  # E: "None" not callable

np.frexp.at()  # E: "None" not callable
np.divmod.at()  # E: "None" not callable
np.matmul.at()  # E: "None" not callable
