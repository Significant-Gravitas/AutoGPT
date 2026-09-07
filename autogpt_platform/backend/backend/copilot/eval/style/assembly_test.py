"""The gate measures production: same rendering, same routing, same trigger
list as the workflow, and a fingerprint that moves when any of it does."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from backend.copilot.briefing.narrative import _system_prompt
from backend.copilot.config import ChatConfig
from backend.copilot.expert_context import (
    build_expert_context,
    build_expert_identity_suffix,
)
from backend.copilot.prompting import get_sdk_supplement
from backend.copilot.service import CACHEABLE_SYSTEM_PROMPT

from .assembly import (
    TRIGGER_PATHS,
    attach_workflows,
    chat_system_prompt,
    fingerprint,
    lede_prompt,
    load_fixtures,
    load_gate,
    load_rubric,
    resolve_chat_model,
    roster_experts,
    user_prefix,
)
from .models import PROMPT_KINDS, PROMPTS_PER_EXPERT, LedeFacts, LedeRun

REPO_ROOT = Path(__file__).resolve().parents[6]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "platform-expert-style-gate.yml"


def test_rubric_has_three_anchors_per_dimension():
    rubric = load_rubric()
    assert rubric.scale_min < rubric.scale_max
    assert len(rubric.dimensions) >= 3
    for dim in rubric.dimensions:
        assert set(dim.anchors) == {str(rubric.scale_min), "3", str(rubric.scale_max)}
        assert dim.question.endswith("?")


def test_gate_config_loads():
    gate = load_gate()
    assert 0 < gate.pass_threshold < 100
    assert gate.judge_model


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
    roster = [attach_workflows(e, f) for e, f in zip(roster_experts(), load_fixtures())]
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


def test_fingerprint_moves_with_the_voice_spec_and_models():
    experts = roster_experts()
    fixtures, rubric = load_fixtures(), load_rubric()
    models = {"chat_model": "m", "lede_model": "l", "judge_model": "j"}
    base = fingerprint(experts, fixtures, rubric, **models)
    assert base == fingerprint(experts, fixtures, rubric, **models)
    assert base != fingerprint(
        experts, fixtures, rubric, **{**models, "chat_model": "m2"}
    )
    changed = [e.model_copy(update={"voice_preferences": "Terse."}) for e in experts]
    assert base != fingerprint(changed, fixtures, rubric, **models)
    with_workflows = [attach_workflows(e, f) for e, f in zip(experts, fixtures)]
    assert base != fingerprint(with_workflows, fixtures, rubric, **models)


def test_workflow_triggers_on_exactly_the_declared_paths():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert set(workflow[True]["pull_request"]["paths"]) == set(TRIGGER_PATHS)
    for path in TRIGGER_PATHS:
        target = REPO_ROOT / path.removesuffix("/**")
        assert target.exists(), path
