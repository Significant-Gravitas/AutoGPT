import io
from typing import Any

import numpy as np
import numpy.typing as npt

AR_i8: npt.NDArray[np.int64]
REC_AR_V: np.recarray[Any, np.dtype[np.record]]
AR_LIST: list[npt.NDArray[np.int64]]

format_parser: np.format_parser
record: np.record
file_obj: io.BufferedIOBase

reveal_type(np.format_parser(  # E: format_parser
    formats=[np.float64, np.int64, np.bool_],
    names=["f8", "i8", "?"],
    titles=None,
    aligned=True,
))
reveal_type(format_parser.dtype)  # E: dtype[void]

reveal_type(record.field_a)  # E: Any
reveal_type(record.field_b)  # E: Any
reveal_type(record["field_a"])  # E: Any
reveal_type(record["field_b"])  # E: Any
reveal_type(record.pprint())  # E: str
record.field_c = 5

reveal_type(REC_AR_V.field(0))  # E: Any
reveal_type(REC_AR_V.field("field_a"))  # E: Any
reveal_type(REC_AR_V.field(0, AR_i8))  # E: None
reveal_type(REC_AR_V.field("field_a", AR_i8))  # E: None
reveal_type(REC_AR_V["field_a"])  # E: Any
reveal_type(REC_AR_V.field_a)  # E: Any
reveal_type(REC_AR_V.__array_finalize__(object()))  # E: None

reveal_type(np.recarray(  # recarray[Any, dtype[record]]
    shape=(10, 5),
    formats=[np.float64, np.int64, np.bool_],
    order="K",
    byteorder="|",
))
reveal_type(np.recarray(  # recarray[Any, dtype[Any]]
    shape=(10, 5),
    dtype=[("f8", np.float64), ("i8", np.int64)],
    strides=(5, 5),
))

reveal_type(np.rec.fromarrays(  # recarray[Any, dtype[record]]
    AR_LIST,
))
reveal_type(np.rec.fromarrays(  # recarray[Any, dtype[Any]]
    AR_LIST,
    dtype=np.int64,
))
reveal_type(np.rec.fromarrays(  # recarray[Any, dtype[Any]]
    AR_LIST,
    formats=[np.int64, np.float64],
    names=["i8", "f8"]
))

reveal_type(np.rec.fromrecords(  # recarray[Any, dtype[record]]
    (1, 1.5),
))
reveal_type(np.rec.fromrecords(  # recarray[Any, dtype[record]]
    [(1, 1.5)],
    dtype=[("i8", np.int64), ("f8", np.float64)],
))
reveal_type(np.rec.fromrecords(  # recarray[Any, dtype[record]]
    REC_AR_V,
    formats=[np.int64, np.float64],
    names=["i8", "f8"]
))

reveal_type(np.rec.fromstring(  # recarray[Any, dtype[record]]
    b"(1, 1.5)",
    dtype=[("i8", np.int64), ("f8", np.float64)],
))
reveal_type(np.rec.fromstring(  # recarray[Any, dtype[record]]
    REC_AR_V,
    formats=[np.int64, np.float64],
    names=["i8", "f8"]
))

reveal_type(np.rec.fromfile(  # recarray[Any, dtype[Any]]
    "test_file.txt",
    dtype=[("i8", np.int64), ("f8", np.float64)],
))
reveal_type(np.rec.fromfile(  # recarray[Any, dtype[record]]
    file_obj,
    formats=[np.int64, np.float64],
    names=["i8", "f8"]
))

reveal_type(np.rec.array(  # recarray[Any, dtype[{int64}]]
    AR_i8,
))
reveal_type(np.rec.array(  # recarray[Any, dtype[Any]]
    [(1, 1.5)],
    dtype=[("i8", np.int64), ("f8", np.float64)],
))
reveal_type(np.rec.array(  # recarray[Any, dtype[record]]
    [(1, 1.5)],
    formats=[np.int64, np.float64],
    names=["i8", "f8"]
))

reveal_type(np.rec.array(  # recarray[Any, dtype[Any]]
    None,
    dtype=np.float64,
    shape=(10, 3),
))
reveal_type(np.rec.array(  # recarray[Any, dtype[Any]]
    None,
    formats=[np.int64, np.float64],
    names=["i8", "f8"],
    shape=(10, 3),
))
reveal_type(np.rec.array(  # recarray[Any, dtype[Any]]
    file_obj,
    dtype=np.float64,
))
reveal_type(np.rec.array(  # recarray[Any, dtype[Any]]
    file_obj,
    formats=[np.int64, np.float64],
    names=["i8", "f8"],
))
