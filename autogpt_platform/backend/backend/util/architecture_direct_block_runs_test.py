import ast

import pytest

from backend.util.architecture_test import BACKEND_ROOT, _iter_backend_py_files

# ---------------------------------------------------------------------------
# Rule: request-path code runs a block directly only where it meters that use
# ---------------------------------------------------------------------------
#
# Motivation: a direct ``Block.run_once`` call under ``api/`` or ``copilot/`` caps
# its own use per user; each file that makes one is listed here.

_DIRECT_BLOCK_RUN_SITES = frozenset(
    {
        # generate_agent_image: per-user daily cap (store/image_gen.py).
        "api/features/store/image_gen.py",
    }
)


def test_direct_block_runs_in_request_code_are_listed():
    found: set[str] = set()
    for py in _iter_backend_py_files():
        rel = py.relative_to(BACKEND_ROOT)
        if rel.parts[0] not in ("api", "copilot") or py.name.endswith("_test.py"):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_once"
            for node in ast.walk(tree)
        ):
            found.add(rel.as_posix())

    assert found == _DIRECT_BLOCK_RUN_SITES, (
        "Files under api/ or copilot/ that call a block's `run_once` directly "
        "changed. Each one must cap its own use per user, then be listed in "
        f"_DIRECT_BLOCK_RUN_SITES.\nFound: {sorted(found)}\n"
        f"Listed: {sorted(_DIRECT_BLOCK_RUN_SITES)}"
    )


_AGENT_IMAGE_PROVIDER_OWNERS = {
    "api/features/store/image_gen.py",
    "util/architecture_direct_block_runs_test.py",
}


def test_agent_image_providers_are_called_only_behind_the_cap():
    offenders = [
        rel
        for py in _iter_backend_py_files()
        if (rel := py.relative_to(BACKEND_ROOT).as_posix())
        not in _AGENT_IMAGE_PROVIDER_OWNERS
        and "_generate_agent_image_v" in py.read_text(encoding="utf-8")
    ]
    message = f"Call generate_agent_image(), which caps per user: {offenders}"
    assert not offenders, message


# ---------------------------------------------------------------------------
# Rule: a route runs a block for its caller only through `run_block_directly`
# ---------------------------------------------------------------------------
#
# Motivation: #17321. Both block-execute routes ran `block.execute()` on the
# request as sent and charged for it themselves, so no block that takes
# credentials could run (its `run` never got them), and each run was charged
# before it failed. `run_block_directly` loads the caller's own credentials for
# the run, checking them before it charges. A route that runs a block's
# `execute` (always given an `execution_context`) or calls the two things a
# direct run is made of skips that.

_DIRECT_RUN_MODULE = "api/features/blocks/direct_run.py"
_DIRECT_RUN_STEPS = frozenset(
    {"run_block_off_loop", "charge_for_direct_block_execution"}
)


def test_routes_run_a_block_for_the_caller_only_through_run_block_directly():
    offenders = [
        f"  backend/{rel}:{lineno} {what}"
        for py in _iter_backend_py_files()
        if not py.name.endswith("_test.py")
        and (rel := py.relative_to(BACKEND_ROOT).as_posix()) != _DIRECT_RUN_MODULE
        for lineno, what in _direct_runs(
            ast.parse(py.read_text(encoding="utf-8")), in_api=rel.startswith("api/")
        )
    ]

    assert not offenders, (
        "Run a block for an API caller with `run_block_directly` "
        f"(backend/{_DIRECT_RUN_MODULE}): it hands the block the caller's own "
        "credentials, and refuses ones it can't run on before it charges. "
        "These lines run a block, or charge for a direct run, themselves:\n"
        + "\n".join(sorted(offenders))
    )


def _direct_runs(tree: ast.Module, *, in_api: bool) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else None
        )
        if name in _DIRECT_RUN_STEPS:
            found.append((node.lineno, f"calls `{name}`"))
        elif (
            in_api
            and name == "execute"
            and any(keyword.arg == "execution_context" for keyword in node.keywords)
        ):
            found.append((node.lineno, "runs a block's `execute`"))
    return found


@pytest.mark.parametrize(
    "source, in_api, flagged",
    [
        ("obj.execute(data, execution_context=context)", True, True),
        ("run_block_off_loop(block.execute(data), slot)", False, True),
        ("execution_utils.charge_for_direct_block_execution(user_id=u)", False, True),
        ("charge_for_direct_block_execution(user_id=u)", True, True),
        # Outside api/, the executor and the copilot run blocks themselves.
        ("block.execute(data, execution_context=context)", False, False),
        ("await pipe.execute()", True, False),
        ("run_block_directly(obj, data, user_id=u)", True, False),
    ],
)
def test_direct_run_rule_flags_running_a_block_outside_run_block_directly(
    source: str, in_api: bool, flagged: bool
):
    assert bool(_direct_runs(ast.parse(source), in_api=in_api)) is flagged
