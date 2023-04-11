import numpy as np
from numpy import f2py

reveal_type(np)  # E: ModuleType

reveal_type(np.char)  # E: ModuleType
reveal_type(np.ctypeslib)  # E: ModuleType
reveal_type(np.emath)  # E: ModuleType
reveal_type(np.fft)  # E: ModuleType
reveal_type(np.lib)  # E: ModuleType
reveal_type(np.linalg)  # E: ModuleType
reveal_type(np.ma)  # E: ModuleType
reveal_type(np.matrixlib)  # E: ModuleType
reveal_type(np.polynomial)  # E: ModuleType
reveal_type(np.random)  # E: ModuleType
reveal_type(np.rec)  # E: ModuleType
reveal_type(np.testing)  # E: ModuleType
reveal_type(np.version)  # E: ModuleType

reveal_type(np.lib.format)  # E: ModuleType
reveal_type(np.lib.mixins)  # E: ModuleType
reveal_type(np.lib.scimath)  # E: ModuleType
reveal_type(np.lib.stride_tricks)  # E: ModuleType
reveal_type(np.ma.extras)  # E: ModuleType
reveal_type(np.polynomial.chebyshev)  # E: ModuleType
reveal_type(np.polynomial.hermite)  # E: ModuleType
reveal_type(np.polynomial.hermite_e)  # E: ModuleType
reveal_type(np.polynomial.laguerre)  # E: ModuleType
reveal_type(np.polynomial.legendre)  # E: ModuleType
reveal_type(np.polynomial.polynomial)  # E: ModuleType

reveal_type(np.__path__)  # E: list[builtins.str]
reveal_type(np.__version__)  # E: str
reveal_type(np.__git_version__)  # E: str
reveal_type(np.test)  # E: _pytesttester.PytestTester
reveal_type(np.test.module_name)  # E: str

reveal_type(np.__all__)  # E: list[builtins.str]
reveal_type(np.char.__all__)  # E: list[builtins.str]
reveal_type(np.ctypeslib.__all__)  # E: list[builtins.str]
reveal_type(np.emath.__all__)  # E: list[builtins.str]
reveal_type(np.lib.__all__)  # E: list[builtins.str]
reveal_type(np.ma.__all__)  # E: list[builtins.str]
reveal_type(np.random.__all__)  # E: list[builtins.str]
reveal_type(np.rec.__all__)  # E: list[builtins.str]
reveal_type(np.testing.__all__)  # E: list[builtins.str]
reveal_type(f2py.__all__)  # E: list[builtins.str]
