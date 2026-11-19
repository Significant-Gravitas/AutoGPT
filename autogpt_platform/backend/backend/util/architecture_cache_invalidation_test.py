"""Shared-cache management can open a sync Redis connection or scan every shard."""

import ast

from backend.util.architecture_test import (
    BACKEND_ROOT,
    _decorator_name,
    _iter_backend_py_files,
)

_SYNC_CACHE_CALLS = {
    "cache_clear",
    "cache_delete",
    "cache_info",
    "clear_all_caches",
    "_clear_store_caches",
    "_clear_user_caches",
    "forget_store_listings",
    "forget_refusal",
    "forget_personal_org_owner",
    "forget_user_email_verification",
    "invalidate_subscription_caches",
    "invalidate_account_caches",
}


def _calls_in_body(node: ast.AST):
    """A nested function or lambda runs where it is called."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _calls_in_body(child)


def test_no_sync_cache_management_on_the_event_loop():
    offenders = []
    for path in _iter_backend_py_files():
        rel = path.relative_to(BACKEND_ROOT)
        if (
            path.name == "conftest.py"
            or path.name.startswith("test_")
            or path.name.endswith("_test.py")
            or "test" in rel.parts
            or "tests" in rel.parts
        ):
            continue
        source = path.read_text(encoding="utf-8")
        if not any(name in source for name in _SYNC_CACHE_CALLS):
            continue
        tree = ast.parse(source, filename=str(path))
        names = _SYNC_CACHE_CALLS | {
            alias.asname
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name in _SYNC_CACHE_CALLS and alias.asname
        }
        for func in ast.walk(tree):
            if isinstance(func, ast.AsyncFunctionDef):
                for call in _calls_in_body(func):
                    if _decorator_name(call.func) in names:
                        offenders.append(f"{rel.as_posix()}:{call.lineno} {func.name}")

    assert not offenders, "Offload synchronous cache management:\n" + "\n".join(
        offenders
    )
