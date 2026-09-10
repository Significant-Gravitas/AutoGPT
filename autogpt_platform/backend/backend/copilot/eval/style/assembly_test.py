"""The check measures production: same rendering, same routing, and a
fingerprint whose components move when any of it does."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.briefing.narrative import _system_prompt
from backend.copilot.config import ChatConfig
from backend.copilot.expert_context import (
    build_expert_context,
    build_expert_identity_suffix,
)
from backend.copilot.prompting import get_sdk_supplement
from backend.copilot.service import CACHEABLE_SYSTEM_PROMPT

from .assembly import (
    HARNESS_MODULES,
    STYLE_DIR,
    attach_workflows,
    chat_system_prompt,
    fingerprint,
    fingerprint_parts,
    harness_fingerprint,
    lede_prompt,
    load_baseline,
    load_fixtures,
    load_rubric,
    resolve_chat_model,
    roster_experts,
    user_prefix,
)
from .models import PROMPT_KINDS, PROMPTS_PER_EXPERT, LedeFacts, LedeRun


def test_rubric_has_three_anchors_per_dimension():
    rubric = load_rubric()
    assert rubric.scale_min < rubric.scale_max
    assert len(rubric.dimensions) >= 3
    for dim in rubric.dimensions:
        assert set(dim.anchors) == {str(rubric.scale_min), "3", str(rubric.scale_max)}
        assert dim.question.endswith("?")


def test_the_baseline_covers_the_whole_roster_and_says_what_produced_it():
    baseline = load_baseline()
    assert {b.expert for b in baseline.experts} == {e.name for e in roster_experts()}
    for stored in baseline.experts:
        assert stored.scores.n == len(stored.by_prompt) > 0
        assert 0 <= stored.scores.mean <= 100
    assert baseline.judge_model and baseline.chat_model
    assert baseline.cost_usd > 0
    assert baseline.note
    assert baseline.fingerprint == fingerprint(baseline.parts)


@pytest.mark.parametrize("expert", [e.name for e in roster_experts()])
def test_every_expert_has_thirty_prompts_across_every_kind(expert: str):
    (fixture,) = load_fixtures([expert])
    assert len(fixture.prompts) == PROMPTS_PER_EXPERT
    kinds = {p.kind for p in fixture.prompts}
    assert kinds == set(PROMPT_KINDS)
    ids = [p.id for p in fixture.prompts]
    assert len(ids) == len(set(ids))


def test_fixture_set_covers_the_whole_roster():
    assert {f.expert for f in load_fixtures()} == {e.name for e in roster_experts()}


@pytest.mark.asyncio
async def test_expert_suffix_is_the_production_rendering():
    """The pure renderer and the DB-backed builder must emit the same block."""
    expert = roster_experts(["Maria"])[0]
    db = MagicMock()
    db.get_expert = AsyncMock(return_value=expert)
    db.resolve_private_expert_tenancy = AsyncMock(return_value=(None, None))
    with patch("backend.copilot.expert_context.experts_db", MagicMock(return_value=db)):
        production = await build_expert_identity_suffix(
            "user", expert.id, organization_id=None, team_id=None
        )
    assert chat_system_prompt(expert).endswith(production)


@pytest.mark.asyncio
async def test_user_prefix_is_the_production_first_turn_context():
    """``build_expert_context`` (DB-backed) and the pure renderers must emit
    the same workflow and teammate blocks, for an expert and for AutoPilot."""
    roster = _installed(roster_experts(), load_fixtures())
    max_ = next(e for e in roster if e.name == "Max")
    db = MagicMock()
    db.get_expert = AsyncMock(return_value=max_)
    db.list_experts = AsyncMock(return_value=roster)
    with (
        patch("backend.copilot.expert_context.experts_db", MagicMock(return_value=db)),
        patch(
            "backend.copilot.expert_context.is_feature_enabled",
            AsyncMock(return_value=True),
        ),
    ):
        assert await build_expert_context("user", max_.id) == user_prefix(max_, roster)
        assert await build_expert_context("user", None) == user_prefix(None, roster)
    assert "Lead Finder (Local Businesses)" in user_prefix(max_, roster)
    assert (
        "Maria" in user_prefix(max_, roster)
        and "Max" not in user_prefix(max_, roster).split("<team_context>")[1]
    )


def test_chat_prompt_is_base_plus_sdk_supplements_plus_suffix():
    expert = roster_experts(["Frankie"])[0]
    prompt = chat_system_prompt(expert)
    assert prompt.startswith(CACHEABLE_SYSTEM_PROMPT)
    assert get_sdk_supplement(use_e2b=True) in prompt
    assert prompt.endswith("</expert_identity>")
    assert chat_system_prompt(None).endswith(
        get_sdk_supplement(use_e2b=True)[-40:]
    ) or ("<expert_identity>" not in chat_system_prompt(None))


def test_lede_prompt_is_the_narrative_modules_own():
    expert = roster_experts(["Max"])[0]
    facts = LedeFacts(
        completed_total=1,
        runs=[LedeRun(agent_name="Lead Finder", status="COMPLETED", title="Found 12")],
    )
    system, user = lede_prompt(expert, facts)
    assert system == _system_prompt(expert)
    assert user.startswith("<briefing_facts>")
    assert "Max / Lead Finder: Found 12" in user


@pytest.mark.asyncio
async def test_chat_model_comes_from_the_router_without_launchdarkly():
    config = ChatConfig(
        use_claude_agent_sdk=True,
        thinking_standard_model="anthropic/claude-style-test",
        use_claude_code_subscription=False,
    )
    with patch(
        "backend.copilot.engine.is_feature_enabled", AsyncMock(return_value=True)
    ):
        routed = await resolve_chat_model(config)
    assert routed.mode == "thinking"
    assert routed.slug == "anthropic/claude-style-test"
    assert routed.source == "env"


def test_each_expert_is_installed_with_its_own_fixtures_workflows():
    """ROSTER order and the fixture files' alphabetical order differ, so
    zipping the two gives Max another expert's workflows and every assertion
    downstream compares one invalid roster against another."""
    by_name = {fixture.expert: fixture for fixture in load_fixtures()}
    for expert in _installed(roster_experts(), load_fixtures()):
        assert [w.name for w in expert.workflows] == [
            w.name for w in by_name[expert.name].workflows
        ]


def test_the_fingerprint_moves_when_the_harness_does(tmp_path):
    """A tool stub or a judge prompt decides a score as much as the prompts do,
    so an edit to one must not read as an unchanged baseline."""
    first, second = tmp_path / "a.py", tmp_path / "b.py"
    first.write_text("stub = 'queued'\n")
    second.write_text("cap = 10\n")
    before = harness_fingerprint([first, second])
    assert before == harness_fingerprint([second, first]), "order is not content"
    second.write_text("cap = 12\n")
    assert harness_fingerprint([first, second]) != before

    experts, fixtures, rubric = roster_experts(), load_fixtures(), load_rubric()
    parts = fingerprint_parts(
        experts, fixtures, rubric, chat_model="m", lede_model="l", judge_model="j"
    )
    assert parts["harness"] == harness_fingerprint(
        STYLE_DIR / name for name in HARNESS_MODULES
    )
    for name in HARNESS_MODULES:
        assert (STYLE_DIR / name).exists(), name


def test_the_fingerprint_names_the_component_that_moved():
    experts = roster_experts()
    fixtures, rubric = load_fixtures(), load_rubric()
    models = {"chat_model": "m", "lede_model": "l", "judge_model": "j"}
    base = fingerprint_parts(experts, fixtures, rubric, **models)
    assert base == fingerprint_parts(experts, fixtures, rubric, **models)
    assert _moved(
        base,
        fingerprint_parts(experts, fixtures, rubric, **{**models, "chat_model": "x"}),
    ) == {"models"}
    terse = [
        e.model_copy(update={"voice_preferences": "Terse."}) if e.name == "Max" else e
        for e in experts
    ]
    assert _moved(base, fingerprint_parts(terse, fixtures, rubric, **models)) == {
        "prompt:Max",
        "lede:Max",
    }, "the voice spec reaches the chat prompt and the briefing lede"
    installed = _installed(experts, fixtures)
    assert _moved(base, fingerprint_parts(installed, fixtures, rubric, **models)) == {
        "autopilot",
        *(f"context:{e.name}" for e in experts),
    }, "installed workflows show in every first-turn block, AutoPilot's included"


def _installed(experts, fixtures):
    """The roster with its preloads attached, paired the way the runner pairs
    them: ROSTER order and the fixture files' alphabetical order differ, so
    zipping them gives Max another expert's workflows."""
    by_name = {fixture.expert: fixture for fixture in fixtures}
    return [attach_workflows(expert, by_name[expert.name]) for expert in experts]


def _moved(before: dict[str, str], after: dict[str, str]) -> set[str]:
    return {k for k, v in after.items() if before.get(k) != v}
