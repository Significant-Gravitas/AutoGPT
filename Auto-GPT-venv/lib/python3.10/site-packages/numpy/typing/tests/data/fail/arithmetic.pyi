from typing import Any
import numpy as np

b_ = np.bool_()
dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

AR_b: np.ndarray[Any, np.dtype[np.bool_]]
AR_u: np.ndarray[Any, np.dtype[np.uint32]]
AR_i: np.ndarray[Any, np.dtype[np.int64]]
AR_f: np.ndarray[Any, np.dtype[np.float64]]
AR_c: np.ndarray[Any, np.dtype[np.complex128]]
AR_m: np.ndarray[Any, np.dtype[np.timedelta64]]
AR_M: np.ndarray[Any, np.dtype[np.datetime64]]

ANY: Any

AR_LIKE_b: list[bool]
AR_LIKE_u: list[np.uint32]
AR_LIKE_i: list[int]
AR_LIKE_f: list[float]
AR_LIKE_c: list[complex]
AR_LIKE_m: list[np.timedelta64]
AR_LIKE_M: list[np.datetime64]

# Array subtraction

# NOTE: mypys `NoReturn` errors are, unfortunately, not that great
_1 = AR_b - AR_LIKE_b  # E: Need type annotation
_2 = AR_LIKE_b - AR_b  # E: Need type annotation
AR_i - bytes()  # E: No overload variant

AR_f - AR_LIKE_m  # E: Unsupported operand types
AR_f - AR_LIKE_M  # E: Unsupported operand types
AR_c - AR_LIKE_m  # E: Unsupported operand types
AR_c - AR_LIKE_M  # E: Unsupported operand types

AR_m - AR_LIKE_f  # E: Unsupported operand types
AR_M - AR_LIKE_f  # E: Unsupported operand types
AR_m - AR_LIKE_c  # E: Unsupported operand types
AR_M - AR_LIKE_c  # E: Unsupported operand types

AR_m - AR_LIKE_M  # E: Unsupported operand types
AR_LIKE_m - AR_M  # E: Unsupported operand types

# array floor division

AR_M // AR_LIKE_b  # E: Unsupported operand types
AR_M // AR_LIKE_u  # E: Unsupported operand types
AR_M // AR_LIKE_i  # E: Unsupported operand types
AR_M // AR_LIKE_f  # E: Unsupported operand types
AR_M // AR_LIKE_c  # E: Unsupported operand types
AR_M // AR_LIKE_m  # E: Unsupported operand types
AR_M // AR_LIKE_M  # E: Unsupported operand types

AR_b // AR_LIKE_M  # E: Unsupported operand types
AR_u // AR_LIKE_M  # E: Unsupported operand types
AR_i // AR_LIKE_M  # E: Unsupported operand types
AR_f // AR_LIKE_M  # E: Unsupported operand types
AR_c // AR_LIKE_M  # E: Unsupported operand types
AR_m // AR_LIKE_M  # E: Unsupported operand types
AR_M // AR_LIKE_M  # E: Unsupported operand types

_3 = AR_m // AR_LIKE_b  # E: Need type annotation
AR_m // AR_LIKE_c  # E: Unsupported operand types

AR_b // AR_LIKE_m  # E: Unsupported operand types
AR_u // AR_LIKE_m  # E: Unsupported operand types
AR_i // AR_LIKE_m  # E: Unsupported operand types
AR_f // AR_LIKE_m  # E: Unsupported operand types
AR_c // AR_LIKE_m  # E: Unsupported operand types

# Array multiplication

AR_b *= AR_LIKE_u  # E: incompatible type
AR_b *= AR_LIKE_i  # E: incompatible type
AR_b *= AR_LIKE_f  # E: incompatible type
AR_b *= AR_LIKE_c  # E: incompatible type
AR_b *= AR_LIKE_m  # E: incompatible type

AR_u *= AR_LIKE_i  # E: incompatible type
AR_u *= AR_LIKE_f  # E: incompatible type
AR_u *= AR_LIKE_c  # E: incompatible type
AR_u *= AR_LIKE_m  # E: incompatible type

AR_i *= AR_LIKE_f  # E: incompatible type
AR_i *= AR_LIKE_c  # E: incompatible type
AR_i *= AR_LIKE_m  # E: incompatible type

AR_f *= AR_LIKE_c  # E: incompatible type
AR_f *= AR_LIKE_m  # E: incompatible type

# Array power

AR_b **= AR_LIKE_b  # E: Invalid self argument
AR_b **= AR_LIKE_u  # E: Invalid self argument
AR_b **= AR_LIKE_i  # E: Invalid self argument
AR_b **= AR_LIKE_f  # E: Invalid self argument
AR_b **= AR_LIKE_c  # E: Invalid self argument

AR_u **= AR_LIKE_i  # E: incompatible type
AR_u **= AR_LIKE_f  # E: incompatible type
AR_u **= AR_LIKE_c  # E: incompatible type

AR_i **= AR_LIKE_f  # E: incompatible type
AR_i **= AR_LIKE_c  # E: incompatible type

AR_f **= AR_LIKE_c  # E: incompatible type

# Scalars

b_ - b_  # E: No overload variant

dt + dt  # E: Unsupported operand types
td - dt  # E: Unsupported operand types
td % 1  # E: Unsupported operand types
td / dt  # E: No overload
td % dt  # E: Unsupported operand types

-b_  # E: Unsupported operand type
+b_  # E: Unsupported operand type
