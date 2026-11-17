"""Every platform-key model call and every direct turn publish is listed."""

import ast
import collections
import re

from backend.util.architecture_test import BACKEND_ROOT, _iter_backend_py_files

# ---------------------------------------------------------------------------
# Rule: platform-key model calls and raw turn publishes are listed
# ---------------------------------------------------------------------------
#
# Every call that pays a model provider on the platform's own key, and every
# turn published without ``schedule_turn``/``dispatch_turn``, is named below,
# so adding one is a reviewed change: a new model call checks the allowance
# first (``rate_limit.platform_spend_refusal``, or
# ``credit_billing.tool_spend_refusal`` in a tool) and a new publish takes a
# turn slot, or its entry says why it needn't.

_ALLOWANCE_CHECKS = frozenset({"platform_spend_refusal", "tool_spend_refusal"})

_CHECKS_ALLOWANCE = "checks the allowance before calling"
_DREAM_BUDGET = "a dream or skill-learning pass, gated by check_dream_budget"
_OUTSIDE_TURNS = "a platform feature's call outside any AutoPilot turn"

_PLATFORM_KEY_MODEL_CALLS = {
    "api/features/onboarding_dump/intro.py generate_intro": _OUTSIDE_TURNS,
    "api/features/onboarding_dump/quality.py _semantic_verdict": _OUTSIDE_TURNS,
    "api/features/onboarding_dump/recommend.py generate_recommendations": (
        _OUTSIDE_TURNS
    ),
    "api/features/onboarding_dump/recommend_experts.py "
    "generate_expert_recommendations": _OUTSIDE_TURNS,
    "api/features/store/category_classifier.py classify_category": _OUTSIDE_TURNS,
    "copilot/briefing/narrative.py _complete": _OUTSIDE_TURNS,
    "copilot/dream/orchestrator.py _run_phase": _DREAM_BUDGET,
    "copilot/eval/style/scorer.py judge_context": "an evaluation tool run by hand",
    "copilot/learning/reviewer.py _review_context": _DREAM_BUDGET,
    "copilot/tools/consult_teammate.py _complete": _CHECKS_ALLOWANCE,
    "copilot/tools/web_search.py WebSearchTool._execute": _CHECKS_ALLOWANCE,
    "data/tally.py extract_business_understanding": _OUTSIDE_TURNS,
    "executor/activity_status_generator.py "
    "generate_activity_status_for_execution": _OUTSIDE_TURNS,
    "executor/simulator.py _call_llm_for_simulation": _CHECKS_ALLOWANCE,
    "util/clients.py _get_local_openai_client": "the client factory itself",
}

_RAW_TURN_PUBLISHES = {
    "blocks/autopilot.py _enqueue_for_recovery": (
        "re-publishes a turn whose slot was taken when it was first dispatched"
    ),
}

_CHAT_CONFIG_KEY = re.compile(r"(?:_chat_config|chat_config|chat_cfg)\.api_key")


def test_platform_key_model_calls_are_listed():
    found = _call_sites(_is_platform_key_model_call)
    assert found == set(_PLATFORM_KEY_MODEL_CALLS), (
        "Model calls on the platform's own key changed.\n"
        f"New (check the allowance first, then list it here): "
        f"{sorted(found - set(_PLATFORM_KEY_MODEL_CALLS))}\n"
        f"Gone (remove from the list): "
        f"{sorted(set(_PLATFORM_KEY_MODEL_CALLS) - found)}"
    )


def test_sites_listed_as_checked_call_an_allowance_check():
    unchecked = [
        site
        for site, reason in _PLATFORM_KEY_MODEL_CALLS.items()
        if reason == _CHECKS_ALLOWANCE and not _calls_an_allowance_check(site)
    ]
    assert not unchecked, (
        f"Listed as checking the allowance, but neither the function nor a "
        f"caller in its file calls {sorted(_ALLOWANCE_CHECKS)}: {unchecked}"
    )


def test_turn_publishes_go_through_dispatch_turn():
    found = _call_sites(_is_raw_turn_publish) - {
        "copilot/executor/utils.py dispatch_turn"
    }
    assert found == set(_RAW_TURN_PUBLISHES), (
        "Direct enqueue_copilot_turn calls changed.\n"
        f"New (use schedule_turn, or acquire_turn_slot + dispatch_turn): "
        f"{sorted(found - set(_RAW_TURN_PUBLISHES))}\n"
        f"Gone (remove from the list): "
        f"{sorted(set(_RAW_TURN_PUBLISHES) - found)}"
    )


def _call_sites(matches) -> set[str]:
    """``"<path> <qualified function>"`` for every matching call outside tests."""
    sites: set[str] = set()
    for py in _iter_backend_py_files():
        if py.name.endswith("_test.py") or py.name == "conftest.py":
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        rel = py.relative_to(BACKEND_ROOT).as_posix()
        for scope, call in _scoped_calls(tree, ()):
            if matches(call):
                sites.add(f"{rel} {'.'.join(scope) or '<module>'}")
    return sites


def _scoped_calls(node: ast.AST, scope: tuple[str, ...]):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield from _scoped_calls(child, (*scope, child.name))
            continue
        if isinstance(child, ast.Call):
            yield scope, child
        yield from _scoped_calls(child, scope)


def _calls_an_allowance_check(site: str) -> bool:
    """The listed function, or one in its file that calls it, runs a check."""
    rel, qualname = site.split(" ", 1)
    calls = _calls_by_function(BACKEND_ROOT / rel)
    name = qualname.rsplit(".", 1)[-1]
    scopes = [calls[qualname], *(c for c in calls.values() if name in c)]
    return any(_ALLOWANCE_CHECKS & c for c in scopes)


def _calls_by_function(path) -> dict[str, set[str]]:
    calls: dict[str, set[str]] = collections.defaultdict(set)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for scope, call in _scoped_calls(tree, ()):
        if name := _call_name(call):
            calls[".".join(scope)].add(name)
    return calls


def _is_platform_key_model_call(call: ast.Call) -> bool:
    name = _call_name(call)
    if name == "resolve_route":
        return True
    keywords = {k.arg: k.value for k in call.keywords}
    if name == "get_openai_client":
        prefer = keywords.get("prefer_openrouter")
        return isinstance(prefer, ast.Constant) and prefer.value is True
    if name == "AsyncOpenAI" and "api_key" in keywords:
        return bool(_CHAT_CONFIG_KEY.fullmatch(ast.unparse(keywords["api_key"])))
    return False


def _is_raw_turn_publish(call: ast.Call) -> bool:
    return _call_name(call) == "enqueue_copilot_turn"


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None
