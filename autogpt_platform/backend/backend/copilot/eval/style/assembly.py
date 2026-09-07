"""What production would send: the assembled system prompt and the routed model.

Every piece is imported from the engine modules rather than retyped, so a
change to any of them changes what the gate measures.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from backend.api.features.experts.models import (
    PROTECTED_SOUL_RULES,
    Expert,
    ExpertWorkflowRef,
)
from backend.api.features.experts.seed import ROSTER, RosterEntry
from backend.copilot.briefing.models import BriefingContent, BriefingRunItem
from backend.copilot.briefing.narrative import _facts_block, _system_prompt
from backend.copilot.config import ChatConfig
from backend.copilot.engine import resolve_use_sdk
from backend.copilot.expert_context import (
    render_expert_identity_suffix,
    render_expert_workflows_block,
    render_team_context,
)
from backend.copilot.model_normalize import normalize_model_for_transport
from backend.copilot.model_router import ModelMode, resolve_model_route
from backend.copilot.prompting import (
    get_delegation_supplement,
    get_graphiti_supplement,
    get_sdk_supplement,
)
from backend.copilot.service import CACHEABLE_SYSTEM_PROMPT

from .models import ExpertFixture, GateConfig, LedeFacts, Rubric

STYLE_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = STYLE_DIR / "fixtures"
RUBRIC_PATH = STYLE_DIR / "rubric.json"
GATE_PATH = STYLE_DIR / "gate.json"

# Repo-relative paths whose change re-runs the gate; the workflow's ``paths:``
# filter must equal this list (pinned by assembly_test). Prompt-bearing and
# model-routing files, plus the gate's own inputs.
TRIGGER_PATHS: tuple[str, ...] = (
    ".github/workflows/platform-expert-style-gate.yml",
    "autogpt_platform/backend/backend/copilot/eval/style/**",
    "autogpt_platform/backend/backend/copilot/service.py",
    "autogpt_platform/backend/backend/copilot/prompting.py",
    "autogpt_platform/backend/backend/copilot/expert_context.py",
    "autogpt_platform/backend/backend/api/features/experts/seed.py",
    "autogpt_platform/backend/backend/api/features/experts/models.py",
    "autogpt_platform/backend/backend/copilot/briefing/narrative.py",
    "autogpt_platform/backend/backend/copilot/model_router.py",
    "autogpt_platform/backend/backend/copilot/model_normalize.py",
    "autogpt_platform/backend/backend/copilot/config.py",
    "autogpt_platform/backend/backend/copilot/engine.py",
    "autogpt_platform/backend/backend/data/llm_registry/catalog.py",
    "autogpt_platform/backend/backend/copilot/sdk/service.py",
    "autogpt_platform/backend/backend/copilot/baseline/service.py",
)


class RoutedModel(BaseModel):
    mode: ModelMode
    slug: str
    transport_slug: str
    source: str


def roster_experts(names: list[str] | None = None) -> list[Expert]:
    wanted = {n.lower() for n in names} if names else None
    return [
        roster_expert(entry)
        for entry in ROSTER
        if wanted is None or entry["name"].lower() in wanted
    ]


def roster_expert(entry: RosterEntry) -> Expert:
    """A hired copy of the template, as ``experts_db.hire_expert`` writes it:
    the plain voice description, never the sample envelope."""
    return Expert(
        id=f"style-eval-{entry['name'].lower()}",
        name=entry["name"],
        avatar_url=entry["avatar_url"],
        role=entry["role"],
        tagline=entry["tagline"],
        bio=entry["bio"],
        skills=list(entry["skills"]),
        identity=entry["identity"],
        voice_preferences=entry["voice_preferences"],
        voice_samples=list(entry["voice_samples"]),
        boundaries=entry["boundaries"],
        protected_soul_rules=list(PROTECTED_SOUL_RULES),
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
    )


def attach_workflows(expert: Expert, fixture: ExpertFixture) -> Expert:
    """The roster preloads as a hire installs them, with synthetic ids."""
    slug = expert.name.lower()
    return expert.model_copy(
        update={
            "workflows": [
                ExpertWorkflowRef(
                    id=f"wf-{slug}-{i}",
                    store_listing_version_id=None,
                    library_agent_id=f"la-{slug}-{i}",
                    graph_id=f"graph-{slug}-{i}",
                    name=w.name,
                    description=w.description,
                )
                for i, w in enumerate(fixture.workflows, start=1)
            ]
        }
    )


def user_prefix(expert: Expert | None, roster: list[Expert]) -> str:
    """The first-turn context production prepends to the user's message
    (``build_expert_context``): the expert's installed workflows and the rest
    of the roster as teammates; plain AutoPilot gets the whole roster."""
    if expert is None:
        return render_team_context(roster, delegation_enabled=True)
    return render_expert_workflows_block(expert) + render_team_context(
        roster, delegation_enabled=True, exclude_expert_id=expert.id
    )


def chat_system_prompt(expert: Expert | None) -> str:
    """The SDK engine's system prompt for an expert session (the assembly in
    ``sdk/service.py``) with the flags such a session has on in production:
    cloud sandbox, hire-experts, memory. ``None`` is plain AutoPilot."""
    suffix = render_expert_identity_suffix(expert) if expert else ""
    return (
        CACHEABLE_SYSTEM_PROMPT
        + get_sdk_supplement(use_e2b=True)
        + get_delegation_supplement()
        + get_graphiti_supplement()
        + suffix
    )


def lede_prompt(expert: Expert, facts: LedeFacts) -> tuple[str, str]:
    """``narrative.py``'s own (system, user) pair for a briefing in this voice."""
    content = BriefingContent(
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        timezone="UTC",
        zero_expert_fallback=False,
        run_items=[
            BriefingRunItem(
                expert_id=expert.id,
                expert_name=expert.name,
                expert_avatar_url=None,
                agent_name=run.agent_name,
                graph_id="",
                execution_id="",
                library_agent_id=None,
                status=run.status,
                summary=None,
                link=None,
                title=run.title,
            )
            for run in facts.runs
        ],
        decision_items=[],
        decision_total=facts.decision_total,
        completed_total=facts.completed_total,
        failed_total=facts.failed_total,
    )
    return _system_prompt(expert), _facts_block(content)


async def resolve_chat_model(config: ChatConfig) -> RoutedModel:
    """The model an expert turn runs on: the engine decision, then the
    router's ``(mode, standard)`` cell. No user id, so LaunchDarkly is
    skipped and this is the catalog/env layer; a proposed LD value is
    checked by hand with ``--model``."""
    use_sdk = await resolve_use_sdk(
        None,
        use_claude_code_subscription=config.use_claude_code_subscription,
        config_default=config.use_claude_agent_sdk,
        thinking_available=config.thinking_available,
    )
    mode: ModelMode = "thinking" if use_sdk else "fast"
    route = await resolve_model_route(mode, "standard", None, config=config)
    return RoutedModel(
        mode=mode,
        slug=route.model,
        transport_slug=normalize_model_for_transport(route.model, config),
        source=route.source,
    )


def fingerprint(
    experts: list[Expert],
    fixtures: list[ExpertFixture],
    rubric: Rubric,
    *,
    chat_model: str,
    lede_model: str,
    judge_model: str,
) -> str:
    """sha256 of everything the score depends on; unchanged means the last
    gated run still stands."""
    payload = {
        "chat": {e.name: chat_system_prompt(e) for e in experts},
        "prefix": {e.name: user_prefix(e, experts) for e in experts},
        "autopilot": chat_system_prompt(None) + user_prefix(None, experts),
        "lede": {e.name: _system_prompt(e) for e in experts},
        "models": {"chat": chat_model, "lede": lede_model, "judge": judge_model},
        "fixtures": [f.model_dump() for f in fixtures],
        "rubric": rubric.model_dump(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_fixtures(names: list[str] | None = None) -> list[ExpertFixture]:
    wanted = {n.lower() for n in names} if names else None
    fixtures = [
        ExpertFixture.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(FIXTURES_DIR.glob("*.json"))
    ]
    return [f for f in fixtures if wanted is None or f.expert.lower() in wanted]


def load_rubric() -> Rubric:
    return Rubric.model_validate_json(RUBRIC_PATH.read_text(encoding="utf-8"))


def load_gate() -> GateConfig:
    return GateConfig.model_validate_json(GATE_PATH.read_text(encoding="utf-8"))


def save_gate(gate: GateConfig) -> None:
    GATE_PATH.write_text(gate.model_dump_json(indent=2) + "\n", encoding="utf-8")
