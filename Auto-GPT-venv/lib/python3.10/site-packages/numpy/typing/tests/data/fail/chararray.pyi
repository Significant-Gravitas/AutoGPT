import numpy as np
from typing import Any

AR_U: np.chararray[Any, np.dtype[np.str_]]
AR_S: np.chararray[Any, np.dtype[np.bytes_]]

AR_S.encode()  # E: Invalid self argument
AR_U.decode()  # E: Invalid self argument

AR_U.join(b"_")  # E: incompatible type
AR_S.join("_")  # E: incompatible type

AR_U.ljust(5, fillchar=b"a")  # E: incompatible type
AR_S.ljust(5, fillchar="a")  # E: incompatible type
AR_U.rjust(5, fillchar=b"a")  # E: incompatible type
AR_S.rjust(5, fillchar="a")  # E: incompatible type

AR_U.lstrip(chars=b"a")  # E: incompatible type
AR_S.lstrip(chars="a")  # E: incompatible type
AR_U.strip(chars=b"a")  # E: incompatible type
AR_S.strip(chars="a")  # E: incompatible type
AR_U.rstrip(chars=b"a")  # E: incompatible type
AR_S.rstrip(chars="a")  # E: incompatible type

AR_U.partition(b"a")  # E: incompatible type
AR_S.partition("a")  # E: incompatible type
AR_U.rpartition(b"a")  # E: incompatible type
AR_S.rpartition("a")  # E: incompatible type

AR_U.replace(b"_", b"-")  # E: incompatible type
AR_S.replace("_", "-")  # E: incompatible type

AR_U.split(b"_")  # E: incompatible type
AR_S.split("_")  # E: incompatible type
AR_S.split(1)  # E: incompatible type
AR_U.rsplit(b"_")  # E: incompatible type
AR_S.rsplit("_")  # E: incompatible type

AR_U.count(b"a", start=[1, 2, 3])  # E: incompatible type
AR_S.count("a", end=9)  # E: incompatible type

AR_U.endswith(b"a", start=[1, 2, 3])  # E: incompatible type
AR_S.endswith("a", end=9)  # E: incompatible type
AR_U.startswith(b"a", start=[1, 2, 3])  # E: incompatible type
AR_S.startswith("a", end=9)  # E: incompatible type

AR_U.find(b"a", start=[1, 2, 3])  # E: incompatible type
AR_S.find("a", end=9)  # E: incompatible type
AR_U.rfind(b"a", start=[1, 2, 3])  # E: incompatible type
AR_S.rfind("a", end=9)  # E: incompatible type

AR_U.index(b"a", start=[1, 2, 3])  # E: incompatible type
AR_S.index("a", end=9)  # E: incompatible type
AR_U.rindex(b"a", start=[1, 2, 3])  # E: incompatible type
AR_S.rindex("a", end=9)  # E: incompatible type

AR_U == AR_S  # E: Unsupported operand types
AR_U != AR_S  # E: Unsupported operand types
AR_U >= AR_S  # E: Unsupported operand types
AR_U <= AR_S  # E: Unsupported operand types
AR_U > AR_S  # E: Unsupported operand types
AR_U < AR_S  # E: Unsupported operand types
