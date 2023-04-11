import numpy as np

reveal_type(np.Inf)  # E: float
reveal_type(np.Infinity)  # E: float
reveal_type(np.NAN)  # E: float
reveal_type(np.NINF)  # E: float
reveal_type(np.NZERO)  # E: float
reveal_type(np.NaN)  # E: float
reveal_type(np.PINF)  # E: float
reveal_type(np.PZERO)  # E: float
reveal_type(np.e)  # E: float
reveal_type(np.euler_gamma)  # E: float
reveal_type(np.inf)  # E: float
reveal_type(np.infty)  # E: float
reveal_type(np.nan)  # E: float
reveal_type(np.pi)  # E: float

reveal_type(np.ALLOW_THREADS)  # E: int
reveal_type(np.BUFSIZE)  # E: Literal[8192]
reveal_type(np.CLIP)  # E: Literal[0]
reveal_type(np.ERR_CALL)  # E: Literal[3]
reveal_type(np.ERR_DEFAULT)  # E: Literal[521]
reveal_type(np.ERR_IGNORE)  # E: Literal[0]
reveal_type(np.ERR_LOG)  # E: Literal[5]
reveal_type(np.ERR_PRINT)  # E: Literal[4]
reveal_type(np.ERR_RAISE)  # E: Literal[2]
reveal_type(np.ERR_WARN)  # E: Literal[1]
reveal_type(np.FLOATING_POINT_SUPPORT)  # E: Literal[1]
reveal_type(np.FPE_DIVIDEBYZERO)  # E: Literal[1]
reveal_type(np.FPE_INVALID)  # E: Literal[8]
reveal_type(np.FPE_OVERFLOW)  # E: Literal[2]
reveal_type(np.FPE_UNDERFLOW)  # E: Literal[4]
reveal_type(np.MAXDIMS)  # E: Literal[32]
reveal_type(np.MAY_SHARE_BOUNDS)  # E: Literal[0]
reveal_type(np.MAY_SHARE_EXACT)  # E: Literal[-1]
reveal_type(np.RAISE)  # E: Literal[2]
reveal_type(np.SHIFT_DIVIDEBYZERO)  # E: Literal[0]
reveal_type(np.SHIFT_INVALID)  # E: Literal[9]
reveal_type(np.SHIFT_OVERFLOW)  # E: Literal[3]
reveal_type(np.SHIFT_UNDERFLOW)  # E: Literal[6]
reveal_type(np.UFUNC_BUFSIZE_DEFAULT)  # E: Literal[8192]
reveal_type(np.WRAP)  # E: Literal[1]
reveal_type(np.tracemalloc_domain)  # E: Literal[389047]

reveal_type(np.little_endian)  # E: bool
reveal_type(np.True_)  # E: bool_
reveal_type(np.False_)  # E: bool_

reveal_type(np.UFUNC_PYVALS_NAME)  # E: Literal['UFUNC_PYVALS']

reveal_type(np.sctypeDict)  # E: dict
reveal_type(np.sctypes)  # E: TypedDict
