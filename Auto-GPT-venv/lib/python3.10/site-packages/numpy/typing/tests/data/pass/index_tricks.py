from __future__ import annotations
from typing import Any
import numpy as np

AR_LIKE_b = [[True, True], [True, True]]
AR_LIKE_i = [[1, 2], [3, 4]]
AR_LIKE_f = [[1.0, 2.0], [3.0, 4.0]]
AR_LIKE_U = [["1", "2"], ["3", "4"]]

AR_i8: np.ndarray[Any, np.dtype[np.int64]] = np.array(AR_LIKE_i, dtype=np.int64)

np.ndenumerate(AR_i8)
np.ndenumerate(AR_LIKE_f)
np.ndenumerate(AR_LIKE_U)

np.ndenumerate(AR_i8).iter
np.ndenumerate(AR_LIKE_f).iter
np.ndenumerate(AR_LIKE_U).iter

next(np.ndenumerate(AR_i8))
next(np.ndenumerate(AR_LIKE_f))
next(np.ndenumerate(AR_LIKE_U))

iter(np.ndenumerate(AR_i8))
iter(np.ndenumerate(AR_LIKE_f))
iter(np.ndenumerate(AR_LIKE_U))

iter(np.ndindex(1, 2, 3))
next(np.ndindex(1, 2, 3))

np.unravel_index([22, 41, 37], (7, 6))
np.unravel_index([31, 41, 13], (7, 6), order='F')
np.unravel_index(1621, (6, 7, 8, 9))

np.ravel_multi_index(AR_LIKE_i, (7, 6))
np.ravel_multi_index(AR_LIKE_i, (7, 6), order='F')
np.ravel_multi_index(AR_LIKE_i, (4, 6), mode='clip')
np.ravel_multi_index(AR_LIKE_i, (4, 4), mode=('clip', 'wrap'))
np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9))

np.mgrid[1:1:2]
np.mgrid[1:1:2, None:10]

np.ogrid[1:1:2]
np.ogrid[1:1:2, None:10]

np.index_exp[0:1]
np.index_exp[0:1, None:3]
np.index_exp[0, 0:1, ..., [0, 1, 3]]

np.s_[0:1]
np.s_[0:1, None:3]
np.s_[0, 0:1, ..., [0, 1, 3]]

np.ix_(AR_LIKE_b[0])
np.ix_(AR_LIKE_i[0], AR_LIKE_f[0])
np.ix_(AR_i8[0])

np.fill_diagonal(AR_i8, 5)

np.diag_indices(4)
np.diag_indices(2, 3)

np.diag_indices_from(AR_i8)
