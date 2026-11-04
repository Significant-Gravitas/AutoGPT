import ast

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
