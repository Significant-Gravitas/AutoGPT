import numpy as np
import numpy.typing as npt

AR_U: npt.NDArray[np.str_]
AR_S: npt.NDArray[np.bytes_]

np.char.equal(AR_U, AR_S)  # E: incompatible type

np.char.not_equal(AR_U, AR_S)  # E: incompatible type

np.char.greater_equal(AR_U, AR_S)  # E: incompatible type

np.char.less_equal(AR_U, AR_S)  # E: incompatible type

np.char.greater(AR_U, AR_S)  # E: incompatible type

np.char.less(AR_U, AR_S)  # E: incompatible type

np.char.encode(AR_S)  # E: incompatible type
np.char.decode(AR_U)  # E: incompatible type

np.char.join(AR_U, b"_")  # E: incompatible type
np.char.join(AR_S, "_")  # E: incompatible type

np.char.ljust(AR_U, 5, fillchar=b"a")  # E: incompatible type
np.char.ljust(AR_S, 5, fillchar="a")  # E: incompatible type
np.char.rjust(AR_U, 5, fillchar=b"a")  # E: incompatible type
np.char.rjust(AR_S, 5, fillchar="a")  # E: incompatible type

np.char.lstrip(AR_U, chars=b"a")  # E: incompatible type
np.char.lstrip(AR_S, chars="a")  # E: incompatible type
np.char.strip(AR_U, chars=b"a")  # E: incompatible type
np.char.strip(AR_S, chars="a")  # E: incompatible type
np.char.rstrip(AR_U, chars=b"a")  # E: incompatible type
np.char.rstrip(AR_S, chars="a")  # E: incompatible type

np.char.partition(AR_U, b"a")  # E: incompatible type
np.char.partition(AR_S, "a")  # E: incompatible type
np.char.rpartition(AR_U, b"a")  # E: incompatible type
np.char.rpartition(AR_S, "a")  # E: incompatible type

np.char.replace(AR_U, b"_", b"-")  # E: incompatible type
np.char.replace(AR_S, "_", "-")  # E: incompatible type

np.char.split(AR_U, b"_")  # E: incompatible type
np.char.split(AR_S, "_")  # E: incompatible type
np.char.rsplit(AR_U, b"_")  # E: incompatible type
np.char.rsplit(AR_S, "_")  # E: incompatible type

np.char.count(AR_U, b"a", start=[1, 2, 3])  # E: incompatible type
np.char.count(AR_S, "a", end=9)  # E: incompatible type

np.char.endswith(AR_U, b"a", start=[1, 2, 3])  # E: incompatible type
np.char.endswith(AR_S, "a", end=9)  # E: incompatible type
np.char.startswith(AR_U, b"a", start=[1, 2, 3])  # E: incompatible type
np.char.startswith(AR_S, "a", end=9)  # E: incompatible type

np.char.find(AR_U, b"a", start=[1, 2, 3])  # E: incompatible type
np.char.find(AR_S, "a", end=9)  # E: incompatible type
np.char.rfind(AR_U, b"a", start=[1, 2, 3])  # E: incompatible type
np.char.rfind(AR_S, "a", end=9)  # E: incompatible type

np.char.index(AR_U, b"a", start=[1, 2, 3])  # E: incompatible type
np.char.index(AR_S, "a", end=9)  # E: incompatible type
np.char.rindex(AR_U, b"a", start=[1, 2, 3])  # E: incompatible type
np.char.rindex(AR_S, "a", end=9)  # E: incompatible type
