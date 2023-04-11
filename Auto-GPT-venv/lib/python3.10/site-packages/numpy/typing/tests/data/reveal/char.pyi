import numpy as np
import numpy.typing as npt
from collections.abc import Sequence

AR_U: npt.NDArray[np.str_]
AR_S: npt.NDArray[np.bytes_]

reveal_type(np.char.equal(AR_U, AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.equal(AR_S, AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.not_equal(AR_U, AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.not_equal(AR_S, AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.greater_equal(AR_U, AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.greater_equal(AR_S, AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.less_equal(AR_U, AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.less_equal(AR_S, AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.greater(AR_U, AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.greater(AR_S, AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.less(AR_U, AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.less(AR_S, AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.multiply(AR_U, 5))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.multiply(AR_S, [5, 4, 3]))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.mod(AR_U, "test"))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.mod(AR_S, "test"))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.capitalize(AR_U))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.capitalize(AR_S))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.center(AR_U, 5))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.center(AR_S, [2, 3, 4], b"a"))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.encode(AR_U))  # E: ndarray[Any, dtype[bytes_]]
reveal_type(np.char.decode(AR_S))  # E: ndarray[Any, dtype[str_]]

reveal_type(np.char.expandtabs(AR_U))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.expandtabs(AR_S, tabsize=4))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.join(AR_U, "_"))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.join(AR_S, [b"_", b""]))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.ljust(AR_U, 5))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.ljust(AR_S, [4, 3, 1], fillchar=[b"a", b"b", b"c"]))  # E: ndarray[Any, dtype[bytes_]]
reveal_type(np.char.rjust(AR_U, 5))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.rjust(AR_S, [4, 3, 1], fillchar=[b"a", b"b", b"c"]))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.lstrip(AR_U))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.lstrip(AR_S, chars=b"_"))  # E: ndarray[Any, dtype[bytes_]]
reveal_type(np.char.rstrip(AR_U))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.rstrip(AR_S, chars=b"_"))  # E: ndarray[Any, dtype[bytes_]]
reveal_type(np.char.strip(AR_U))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.strip(AR_S, chars=b"_"))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.partition(AR_U, "\n"))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.partition(AR_S, [b"a", b"b", b"c"]))  # E: ndarray[Any, dtype[bytes_]]
reveal_type(np.char.rpartition(AR_U, "\n"))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.rpartition(AR_S, [b"a", b"b", b"c"]))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.replace(AR_U, "_", "-"))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.replace(AR_S, [b"_", b""], [b"a", b"b"]))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.split(AR_U, "_"))  # E: ndarray[Any, dtype[object_]]
reveal_type(np.char.split(AR_S, maxsplit=[1, 2, 3]))  # E: ndarray[Any, dtype[object_]]
reveal_type(np.char.rsplit(AR_U, "_"))  # E: ndarray[Any, dtype[object_]]
reveal_type(np.char.rsplit(AR_S, maxsplit=[1, 2, 3]))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.char.splitlines(AR_U))  # E: ndarray[Any, dtype[object_]]
reveal_type(np.char.splitlines(AR_S, keepends=[True, True, False]))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.char.swapcase(AR_U))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.swapcase(AR_S))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.title(AR_U))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.title(AR_S))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.upper(AR_U))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.upper(AR_S))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.zfill(AR_U, 5))  # E: ndarray[Any, dtype[str_]]
reveal_type(np.char.zfill(AR_S, [2, 3, 4]))  # E: ndarray[Any, dtype[bytes_]]

reveal_type(np.char.count(AR_U, "a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.char.count(AR_S, [b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(np.char.endswith(AR_U, "a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.endswith(AR_S, [b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.startswith(AR_U, "a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.startswith(AR_S, [b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.find(AR_U, "a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.char.find(AR_S, [b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.char.rfind(AR_U, "a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.char.rfind(AR_S, [b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(np.char.index(AR_U, "a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.char.index(AR_S, [b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.char.rindex(AR_U, "a", start=[1, 2, 3]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.char.rindex(AR_S, [b"a", b"b", b"c"], end=9))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(np.char.isalpha(AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.isalpha(AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.isalnum(AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.isalnum(AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.isdecimal(AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.isdecimal(AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.isdigit(AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.isdigit(AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.islower(AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.islower(AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.isnumeric(AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.isnumeric(AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.isspace(AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.isspace(AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.istitle(AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.istitle(AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.isupper(AR_U))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.char.isupper(AR_S))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.char.str_len(AR_U))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.char.str_len(AR_S))  # E: ndarray[Any, dtype[{int_}]]

reveal_type(np.char.array(AR_U))  # E: chararray[Any, dtype[str_]]
reveal_type(np.char.array(AR_S, order="K"))  # E: chararray[Any, dtype[bytes_]]
reveal_type(np.char.array("bob", copy=True))  # E: chararray[Any, dtype[str_]]
reveal_type(np.char.array(b"bob", itemsize=5))  # E: chararray[Any, dtype[bytes_]]
reveal_type(np.char.array(1, unicode=False))  # E: chararray[Any, dtype[bytes_]]
reveal_type(np.char.array(1, unicode=True))  # E: chararray[Any, dtype[str_]]

reveal_type(np.char.asarray(AR_U))  # E: chararray[Any, dtype[str_]]
reveal_type(np.char.asarray(AR_S, order="K"))  # E: chararray[Any, dtype[bytes_]]
reveal_type(np.char.asarray("bob"))  # E: chararray[Any, dtype[str_]]
reveal_type(np.char.asarray(b"bob", itemsize=5))  # E: chararray[Any, dtype[bytes_]]
reveal_type(np.char.asarray(1, unicode=False))  # E: chararray[Any, dtype[bytes_]]
reveal_type(np.char.asarray(1, unicode=True))  # E: chararray[Any, dtype[str_]]
