import os
import subprocess
from collections.abc import Iterable
from typing import Literal as L, Any, overload, TypedDict

from numpy._pytesttester import PytestTester

class _F2PyDictBase(TypedDict):
    csrc: list[str]
    h: list[str]

class _F2PyDict(_F2PyDictBase, total=False):
    fsrc: list[str]
    ltx: list[str]

__all__: list[str]
__path__: list[str]
test: PytestTester

def run_main(comline_list: Iterable[str]) -> dict[str, _F2PyDict]: ...

@overload
def compile(  # type: ignore[misc]
    source: str | bytes,
    modulename: str = ...,
    extra_args: str | list[str] = ...,
    verbose: bool = ...,
    source_fn: None | str | bytes | os.PathLike[Any] = ...,
    extension: L[".f", ".f90"] = ...,
    full_output: L[False] = ...,
) -> int: ...
@overload
def compile(
    source: str | bytes,
    modulename: str = ...,
    extra_args: str | list[str] = ...,
    verbose: bool = ...,
    source_fn: None | str | bytes | os.PathLike[Any] = ...,
    extension: L[".f", ".f90"] = ...,
    full_output: L[True] = ...,
) -> subprocess.CompletedProcess[bytes]: ...

def get_include() -> str: ...
