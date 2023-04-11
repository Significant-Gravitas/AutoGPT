"""Typing tests for `core._ufunc_config`."""

import numpy as np

def func(a: str, b: int) -> None: ...

class Write:
    def write(self, value: str) -> None: ...

reveal_type(np.seterr(all=None))  # E: TypedDict('core._ufunc_config._ErrDict'
reveal_type(np.seterr(divide="ignore"))  # E: TypedDict('core._ufunc_config._ErrDict'
reveal_type(np.seterr(over="warn"))  # E: TypedDict('core._ufunc_config._ErrDict'
reveal_type(np.seterr(under="call"))  # E: TypedDict('core._ufunc_config._ErrDict'
reveal_type(np.seterr(invalid="raise"))  # E: TypedDict('core._ufunc_config._ErrDict'
reveal_type(np.geterr())  # E: TypedDict('core._ufunc_config._ErrDict'

reveal_type(np.setbufsize(4096))  # E: int
reveal_type(np.getbufsize())  # E: int

reveal_type(np.seterrcall(func))  # E: Union[None, def (builtins.str, builtins.int) -> Any, _SupportsWrite[builtins.str]]
reveal_type(np.seterrcall(Write()))  # E: Union[None, def (builtins.str, builtins.int) -> Any, _SupportsWrite[builtins.str]]
reveal_type(np.geterrcall())  # E: Union[None, def (builtins.str, builtins.int) -> Any, _SupportsWrite[builtins.str]]

reveal_type(np.errstate(call=func, all="call"))  # E: errstate[def (a: builtins.str, b: builtins.int)]
reveal_type(np.errstate(call=Write(), divide="log", over="log"))  # E: errstate[ufunc_config.Write]
