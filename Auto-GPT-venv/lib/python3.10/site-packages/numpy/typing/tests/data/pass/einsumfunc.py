from __future__ import annotations

from typing import Any

import numpy as np

AR_LIKE_b = [True, True, True]
AR_LIKE_u = [np.uint32(1), np.uint32(2), np.uint32(3)]
AR_LIKE_i = [1, 2, 3]
AR_LIKE_f = [1.0, 2.0, 3.0]
AR_LIKE_c = [1j, 2j, 3j]
AR_LIKE_U = ["1", "2", "3"]

OUT_f: np.ndarray[Any, np.dtype[np.float64]] = np.empty(3, dtype=np.float64)
OUT_c: np.ndarray[Any, np.dtype[np.complex128]] = np.empty(3, dtype=np.complex128)

np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_b)
np.einsum("i,i->i", AR_LIKE_u, AR_LIKE_u)
np.einsum("i,i->i", AR_LIKE_i, AR_LIKE_i)
np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f)
np.einsum("i,i->i", AR_LIKE_c, AR_LIKE_c)
np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_i)
np.einsum("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c)

np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f, dtype="c16")
np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=bool, casting="unsafe")
np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f, out=OUT_c)
np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=int, casting="unsafe", out=OUT_f)

np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_b)
np.einsum_path("i,i->i", AR_LIKE_u, AR_LIKE_u)
np.einsum_path("i,i->i", AR_LIKE_i, AR_LIKE_i)
np.einsum_path("i,i->i", AR_LIKE_f, AR_LIKE_f)
np.einsum_path("i,i->i", AR_LIKE_c, AR_LIKE_c)
np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_i)
np.einsum_path("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c)
