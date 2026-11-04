"""Architecture rules for blocks that open E2B sandboxes."""

import ast

from backend.util.architecture_test import BACKEND_ROOT, _iter_backend_py_files

# ---------------------------------------------------------------------------
# Rule: a block opens a box only through a runner told whose E2B key it is on
# ---------------------------------------------------------------------------
#
# The runners pause a box they keep on the operator's key, so each call reads
# that off the credential through the one predicate that knows when the SDK
# falls back to it, and no block opens a box outside them.

_SANDBOX_RUNNERS = frozenset({"execute_code", "execute_claude_code"})
_PLATFORM_KEY_PREDICATE = "uses_platform_e2b_key"
_SANDBOX_OPENERS = frozenset({"create_sandbox", "connect_sandbox"})
# Opens boxes for AutoPilot only, which pauses them at turn end itself.
_SANDBOX_OPENER_MODULES = frozenset({"blocks/desktop/_api.py"})


def test_sandbox_runner_calls_state_pause_when_kept():
    calls = _sandbox_runner_calls()
    # The four block call sites today; a scan that sees none proves nothing.
    assert len(calls) >= 4, calls
    missing = [f"{rel}:{line} {name}()" for rel, line, name, ok in calls if not ok]
    assert not missing, (
        "A call to a block's sandbox runner must pass "
        f"pause_when_kept={_PLATFORM_KEY_PREDICATE}(<the E2B credential>). "
        "is_system_credential alone misses a credential whose empty key the "
        "SDK falls back from.\n\n"
        "Offenders:\n  " + "\n  ".join(missing)
    )


def test_blocks_open_sandboxes_only_inside_a_runner():
    opens = _block_sandbox_opens()
    # Both runners open boxes today; a scan that sees neither proves nothing.
    assert {where for *_, where in opens} >= _SANDBOX_RUNNERS, opens
    offenders = [
        f"{rel}:{line} {name}() in {where or 'module scope'}"
        for rel, line, name, where in opens
        if where not in _SANDBOX_RUNNERS and rel not in _SANDBOX_OPENER_MODULES
    ]
    assert not offenders, (
        "A block opens an E2B sandbox only inside execute_code or "
        "execute_claude_code, which pause a kept one on the platform's key.\n\n"
        "Offenders:\n  " + "\n  ".join(offenders)
    )


def _sandbox_runner_calls() -> list[tuple[str, int, str, bool]]:
    calls: list[tuple[str, int, str, bool]] = []
    for rel, tree in _non_test_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in _SANDBOX_RUNNERS:
                continue
            stated = any(
                k.arg == "pause_when_kept"
                and _called_name(k.value) == _PLATFORM_KEY_PREDICATE
                for k in node.keywords
            )
            calls.append((rel, node.lineno, name, stated))
    return calls


def _block_sandbox_opens() -> list[tuple[str, int, str, str]]:
    opens: list[tuple[str, int, str, str]] = []
    for rel, tree in _non_test_trees("blocks/"):
        for where, node in _calls_with_enclosing_function(tree, ""):
            name = _called_name(node)
            if name in _SANDBOX_OPENERS:
                opens.append((rel, node.lineno, name, where))
    return opens


def _non_test_trees(under: str = ""):
    for py in _iter_backend_py_files():
        rel = py.relative_to(BACKEND_ROOT).as_posix()
        if not rel.startswith(under):
            continue
        if py.name.endswith("_test.py") or py.name == "conftest.py":
            continue
        yield rel, ast.parse(py.read_text(encoding="utf-8"), filename=str(py))


def _calls_with_enclosing_function(node: ast.AST, where: str):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _calls_with_enclosing_function(child, child.name)
            continue
        if isinstance(child, ast.Call):
            yield where, child
        yield from _calls_with_enclosing_function(child, where)


def _called_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""
