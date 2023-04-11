import pathlib
from typing import IO

import numpy.typing as npt
import numpy as np

str_path: str
bytes_path: bytes
pathlib_path: pathlib.Path
str_file: IO[str]
AR_i8: npt.NDArray[np.int64]

np.load(str_file)  # E: incompatible type

np.save(bytes_path, AR_i8)  # E: incompatible type
np.save(str_file, AR_i8)  # E: incompatible type

np.savez(bytes_path, AR_i8)  # E: incompatible type
np.savez(str_file, AR_i8)  # E: incompatible type

np.savez_compressed(bytes_path, AR_i8)  # E: incompatible type
np.savez_compressed(str_file, AR_i8)  # E: incompatible type

np.loadtxt(bytes_path)  # E: incompatible type

np.fromregex(bytes_path, ".", np.int64)  # E: No overload variant

np.recfromtxt(bytes_path)  # E: incompatible type

np.recfromcsv(bytes_path)  # E: incompatible type
