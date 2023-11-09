import os
import sys

__all__ = ("_Quoter", "_Unquoter")


NO_EXTENSIONS = bool(os.environ.get("YARL_NO_EXTENSIONS"))  # type: bool
if sys.implementation.name != "cpython":
    NO_EXTENSIONS = True


if not NO_EXTENSIONS:  # pragma: no branch
    try:
        from ._quoting_c import _Quoter, _Unquoter  # type: ignore[misc]
    except ImportError:  # pragma: no cover
        from ._quoting_py import _Quoter, _Unquoter  # type: ignore[misc]
else:
    from ._quoting_py import _Quoter, _Unquoter  # type: ignore[misc]
