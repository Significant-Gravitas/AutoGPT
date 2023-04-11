import numpy as np
from typing import Any

AR_U: np.chararray[Any, np.dtype[np.str_]]
AR_S: np.chararray[Any, np.dtype[np.bytes_]]

reveal_type(AR_U == AR_U)  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S == AR_S)  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U != AR_U)  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S != AR_S)  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U >= AR_U)  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S >= AR_S)  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U <= AR_U)  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S <= AR_S)  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U > AR_U)  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S > AR_S)  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U < AR_U)  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S < AR_S)  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U * 5)  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S * [5])  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U % "test")  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S % b"test")  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.capitalize())  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.capitalize())  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.center(5))  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.center([2, 3, 4], b"a"))  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.encode())  # E: chararray[Any, dtype[bytes_]]
reveal_type(AR_S.decode())  # E: chararray[Any, dtype[str_]]

reveal_type(AR_U.expandtabs())  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.expandtabs(tabsize=4))  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.join("_"))  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.join([b"_", b""]))  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.ljust(5))  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.ljust([4, 3, 1], fillchar=[b"a", b"b", b"c"]))  # E: chararray[Any, dtype[bytes_]]
reveal_type(AR_U.rjust(5))  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.rjust([4, 3, 1], fillchar=[b"a", b"b", b"c"]))  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.lstrip())  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.lstrip(chars=b"_"))  # E: chararray[Any, dtype[bytes_]]
reveal_type(AR_U.rstrip())  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.rstrip(chars=b"_"))  # E: chararray[Any, dtype[bytes_]]
reveal_type(AR_U.strip())  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.strip(chars=b"_"))  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.partition("\n"))  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.partition([b"a", b"b", b"c"]))  # E: chararray[Any, dtype[bytes_]]
reveal_type(AR_U.rpartition("\n"))  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.rpartition([b"a", b"b", b"c"]))  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.replace("_", "-"))  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.replace([b"_", b""], [b"a", b"b"]))  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.split("_"))  # E: ndarray[Any, dtype[object_]]
reveal_type(AR_S.split(maxsplit=[1, 2, 3]))  # E: ndarray[Any, dtype[object_]]
reveal_type(AR_U.rsplit("_"))  # E: ndarray[Any, dtype[object_]]
reveal_type(AR_S.rsplit(maxsplit=[1, 2, 3]))  # E: ndarray[Any, dtype[object_]]

reveal_type(AR_U.splitlines())  # E: ndarray[Any, dtype[object_]]
reveal_type(AR_S.splitlines(keepends=[True, True, False]))  # E: ndarray[Any, dtype[object_]]

reveal_type(AR_U.swapcase())  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.swapcase())  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.title())  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.title())  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.upper())  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.upper())  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.zfill(5))  # E: chararray[Any, dtype[str_]]
reveal_type(AR_S.zfill([2, 3, 4]))  # E: chararray[Any, dtype[bytes_]]

reveal_type(AR_U.count("a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(AR_S.count([b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(AR_U.endswith("a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S.endswith([b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_U.startswith("a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S.startswith([b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U.find("a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(AR_S.find([b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(AR_U.rfind("a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(AR_S.rfind([b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(AR_U.index("a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(AR_S.index([b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(AR_U.rindex("a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(AR_S.rindex([b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(AR_U.isalpha())  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S.isalpha())  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U.isalnum())  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S.isalnum())  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U.isdecimal())  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S.isdecimal())  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U.isdigit())  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S.isdigit())  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U.islower())  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S.islower())  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U.isnumeric())  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S.isnumeric())  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U.isspace())  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S.isspace())  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U.istitle())  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S.istitle())  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U.isupper())  # E: ndarray[Any, dtype[bool_]]
reveal_type(AR_S.isupper())  # E: ndarray[Any, dtype[bool_]]

reveal_type(AR_U.__array_finalize__(object()))  # E: None
reveal_type(AR_S.__array_finalize__(object()))  # E: None
