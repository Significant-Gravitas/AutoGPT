"""Every paywall, tier and limit reads the one entitlement check.

``backend.data.entitlement`` decides whether someone may use paid features and
at which tier: the user's own plan in a personal organization, the team
organization's plan in a team one. The per-user functions in
``backend.copilot.rate_limit`` apply only the user's own plan, so a direct
call from anywhere else would judge work in a team organization by the wrong
plan: paywalling a member the organization pays for, or using a personal
trial's limits for a team plan's work. This test refuses such calls outside
the entitlement modules, where they belong to the check itself.
"""

import ast
import pathlib

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
RATE_LIMIT_MODULE = "backend.copilot.rate_limit"

# Per-user checks, the helpers whose personal-tier logic the entitlement
# modules switch off for an organization's plan, and the resets, which must
# name the same counters as the checks: a member's own in a team organization.
PER_USER_FUNCTIONS = frozenset(
    {
        "check_rate_limit",
        "enforce_payment_paywall",
        "get_global_rate_limits",
        "get_remaining_usd_budget",
        "get_usage_status",
        "get_user_tier",
        "get_workspace_storage_limit_bytes",
        "is_user_paywalled",
        "rate_limits_for_tier",
        "reset_daily_usage",
        "reset_user_usage",
        "storage_limit_bytes_for_tier",
    }
)

# Where the check lives, and the module that defines the per-user functions.
_ENTITLEMENT_MODULES = frozenset(
    {
        "copilot/rate_limit.py",
        "data/entitlement.py",
        "data/entitlement_limits.py",
    }
)

# Call sites still to move onto the entitlement check, each with why it is
# waiting. Remove an entry once its file calls ``backend.data.entitlement``.
_PENDING: dict[str, str] = {}


def test_per_user_checks_are_called_only_by_the_entitlement_check():
    found = {site for site in _call_sites() if site not in _PENDING}
    assert not found, (
        "These call the per-user plan checks in backend.copilot.rate_limit "
        "directly. Call backend.data.entitlement (or entitlement_limits) "
        "with the organization the work runs in instead:\n" + "\n".join(sorted(found))
    )


def test_pending_call_sites_are_not_stale():
    stale = set(_PENDING) - set(_call_sites())
    assert not stale, f"Drop these from _PENDING, they're fixed: {sorted(stale)}"


def test_the_scan_reads_each_import_form():
    source = """
from backend.copilot.rate_limit import get_user_tier as tier_of
from backend.copilot import rate_limit
import backend.copilot.rate_limit as rl
from .rate_limit import is_user_paywalled

async def f(user_id):
    await tier_of(user_id)
    await rate_limit.get_global_rate_limits(user_id, 1, 1)
    await rl.check_rate_limit(user_id, 1, 1)
    await is_user_paywalled(user_id)
    tier_of.cache_delete(user_id)
"""
    calls = _module_calls(ast.parse(source), ("backend", "copilot", "probe"))
    assert calls == [
        "get_user_tier",
        "get_global_rate_limits",
        "check_rate_limit",
        "is_user_paywalled",
    ]


def _call_sites() -> list[str]:
    sites = []
    for path in BACKEND_ROOT.rglob("*.py"):
        rel = path.relative_to(BACKEND_ROOT)
        if _skipped(rel):
            continue
        module_parts = ("backend", *rel.with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        sites.extend(
            f"{rel.as_posix()} {name}" for name in _module_calls(tree, module_parts)
        )
    return sites


def _skipped(rel: pathlib.PurePath) -> bool:
    name = rel.name
    return (
        "__pycache__" in rel.parts
        or "test" in rel.parts
        or name.endswith("_test.py")
        or name.startswith("test_")
        or name == "conftest.py"
        or rel.as_posix() in _ENTITLEMENT_MODULES
    )


def _module_calls(tree: ast.Module, module_parts: tuple[str, ...]) -> list[str]:
    """The per-user functions this module calls, in source order."""
    functions, modules = _bindings(tree, module_parts)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in functions:
            calls.append((node.lineno, node.col_offset, functions[func.id]))
        elif (
            isinstance(func, ast.Attribute)
            and func.attr in PER_USER_FUNCTIONS
            and ast.unparse(func.value) in modules
        ):
            calls.append((node.lineno, node.col_offset, func.attr))
    return [name for _, _, name in sorted(calls)]


def _bindings(
    tree: ast.Module, module_parts: tuple[str, ...]
) -> tuple[dict[str, str], set[str]]:
    """Local names bound to per-user functions, and expressions naming the
    rate-limit module itself."""
    functions: dict[str, str] = {}
    modules = {RATE_LIMIT_MODULE}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == RATE_LIMIT_MODULE:
                    modules.add(alias.asname or alias.name)
        if not isinstance(node, ast.ImportFrom):
            continue
        source = _resolved(node, module_parts)
        for alias in node.names:
            local = alias.asname or alias.name
            if source == RATE_LIMIT_MODULE and alias.name in PER_USER_FUNCTIONS:
                functions[local] = alias.name
            elif f"{source}.{alias.name}" == RATE_LIMIT_MODULE:
                modules.add(local)
    return functions, modules


def _resolved(node: ast.ImportFrom, module_parts: tuple[str, ...]) -> str:
    if not node.level:
        return node.module or ""
    package = module_parts[: len(module_parts) - node.level]
    return ".".join([*package, *([node.module] if node.module else [])])
