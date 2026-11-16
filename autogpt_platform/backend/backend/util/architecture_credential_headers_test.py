import ast
import pathlib

from backend.util.architecture_test import BACKEND_ROOT, _iter_backend_py_files

# ---------------------------------------------------------------------------
# Rule: every caller of HostScopedCredentials.get_headers_dict is known
# ---------------------------------------------------------------------------
#
# Motivation: ``Requests`` drops a host-scoped credential's headers on a
# cross-origin redirect when the caller names them in ``sensitive_headers``,
# as ``SendAuthenticatedWebRequestBlock`` does. Listing the callers makes a
# new one make that choice on purpose.

_CREDENTIAL_HEADER_CALLERS = frozenset({"blocks/http.py"})


def test_host_scoped_credential_headers_have_only_known_callers():
    callers = {
        py.relative_to(BACKEND_ROOT).as_posix()
        for py in _iter_backend_py_files()
        if not _is_test_module(py) and _calls_get_headers_dict(py)
    }
    assert callers == _CREDENTIAL_HEADER_CALLERS, (
        "Callers of HostScopedCredentials.get_headers_dict() changed. A new "
        "caller must pass the header names to Requests(sensitive_headers=...) "
        "and be added to _CREDENTIAL_HEADER_CALLERS; a removed one leaves it.\n"
        f"  new: {sorted(callers - _CREDENTIAL_HEADER_CALLERS)}\n"
        f"  gone: {sorted(_CREDENTIAL_HEADER_CALLERS - callers)}"
    )


def _is_test_module(path: pathlib.Path) -> bool:
    return (
        path.name.endswith("_test.py")
        or path.name.startswith("test_")
        or path.name == "conftest.py"
        or bool({"test", "tests"} & set(path.relative_to(BACKEND_ROOT).parts))
    )


def _calls_get_headers_dict(path: pathlib.Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return False
    # The credential's own methods (``auth_header``) call it on ``self``.
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_headers_dict"
        and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "self")
        for node in ast.walk(tree)
    )
