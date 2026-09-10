"""What production would send: the assembled system prompt and the routed model.

Every piece is imported from the engine modules rather than retyped, so a
change to any of them changes what the check measures — and shows up as a
changed fingerprint component.
"""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from backend.api.features.experts.models import (
    PROTECTED_SOUL_RULES,
    Expert,
    ExpertWorkflowRef,
)
from backend.api.features.experts.seed import ROSTER, RosterEntry
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
    get_expert_oversight_supplement,
    get_graphiti_supplement,
    get_sdk_supplement,
    get_team_building_supplement,
)
from backend.copilot.service import CACHEABLE_SYSTEM_PROMPT

from .models import Baseline, ExpertFixture, Rubric

STYLE_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = STYLE_DIR / "fixtures"
RUBRIC_PATH = STYLE_DIR / "rubric.json"
BASELINE_PATH = STYLE_DIR / "baseline.json"
# Everything between a prompt and a score: the tool stubs and round cap a turn
# runs under, the judge's own prompt, and the rules that decide which rows are
# scored at all.
HARNESS_MODULES = ("generation.py", "runner.py", "scorer.py")
# The state every scored turn is sent in. An expert exists only because
# hire-experts was on when it was hired, so that is the state its writing is
# measured in; turning the flag off afterwards leaves a session this check
# does not cover, which the baseline records rather than implies.
DELEGATION_ENABLED = True


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
        return render_team_context(roster, delegation_enabled=DELEGATION_ENABLED)
    return render_expert_workflows_block(expert) + render_team_context(
        roster,
        delegation_enabled=DELEGATION_ENABLED,
        exclude_expert_id=expert.id,
    )


def chat_system_prompt(expert: Expert | None) -> str:
    """The SDK engine's system prompt for an expert session (the assembly in
    ``sdk/service.py``) with the flags such a session has on in production:
    cloud sandbox, hire-experts, memory. ``None`` is plain AutoPilot."""
    suffix = render_expert_identity_suffix(expert) if expert else ""
    return (
        CACHEABLE_SYSTEM_PROMPT
        + get_sdk_supplement(use_e2b=True)
        + (get_delegation_supplement() if DELEGATION_ENABLED else "")
        + get_expert_oversight_supplement(
            experts_enabled=DELEGATION_ENABLED,
            expert_id=expert.id if expert else None,
        )
        + get_team_building_supplement(
            experts_enabled=DELEGATION_ENABLED,
            expert_id=expert.id if expert else None,
        )
        + get_graphiti_supplement()
        + suffix
    )


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


def fingerprint(parts: dict[str, str]) -> str:
    """One hash over the components below: unchanged means the baseline still
    describes what the experts are being asked, so there is nothing to re-run."""
    return _sha(json.dumps(parts, sort_keys=True))


def fingerprint_parts(
    experts: list[Expert],
    fixtures: list[ExpertFixture],
    rubric: Rubric,
    *,
    chat_model: str,
    judge_model: str,
) -> dict[str, str]:
    """Everything a score depends on, hashed component by component so a
    mismatch names what moved. The models are the ROUTED names: OpenRouter
    and direct Anthropic spell one model differently, and a fingerprint that
    changed with the transport would report a change nobody made."""
    by_name = {f"prompt:{e.name}": chat_system_prompt(e) for e in experts}
    by_name |= {f"context:{e.name}": user_prefix(e, experts) for e in experts}
    by_name |= {f"fixture:{f.expert}": f.model_dump_json() for f in fixtures}
    return {
        "autopilot": _sha(chat_system_prompt(None) + user_prefix(None, experts)),
        "rubric": _sha(rubric.model_dump_json()),
        "models": _sha(f"{chat_model}|{judge_model}"),
        "delegation": _sha(str(DELEGATION_ENABLED)),
        "harness": harness_fingerprint(STYLE_DIR / m for m in HARNESS_MODULES),
        **{key: _sha(value) for key, value in sorted(by_name.items())},
    }


def harness_fingerprint(paths: Iterable[Path]) -> str:
    """Hashes the source that turns a prompt into a score. Without it an edited
    tool stub or judge prompt reads as an unchanged baseline; an edited comment
    reads as a changed one, which is the safe direction to be wrong in."""
    return _sha("".join(path.read_text(encoding="utf-8") for path in sorted(paths)))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load_fixtures(names: list[str] | None = None) -> list[ExpertFixture]:
    wanted = {n.lower() for n in names} if names else None
    fixtures = [
        ExpertFixture.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(FIXTURES_DIR.glob("*.json"))
    ]
    return [f for f in fixtures if wanted is None or f.expert.lower() in wanted]


def load_rubric() -> Rubric:
    return Rubric.model_validate_json(RUBRIC_PATH.read_text(encoding="utf-8"))


def load_baseline() -> Baseline:
    return Baseline.model_validate_json(BASELINE_PATH.read_text(encoding="utf-8"))


def save_baseline(baseline: Baseline) -> None:
    BASELINE_PATH.write_text(
        baseline.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
