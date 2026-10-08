"""Experts in capability search: the roster a session can hire from and the
team it can hand work to, found through ``find_capability`` and run as
``hire_expert`` / ``delegate_to_expert``."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.experts.models import (
    Expert,
    ExpertBundledSkill,
    ExpertTemplate,
)
from backend.copilot.capabilities.dispatch import resolve_tool_dispatch
from backend.copilot.capabilities.ranking import ConnectionState
from backend.copilot.capabilities.registry import get_registry
from backend.copilot.capabilities.sources import expert_entries
from backend.copilot.gate import review as review_store
from backend.copilot.gate.content import ContentVerdict
from backend.copilot.model import ChatSessionMetadata
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.tools import (
    expert_tool_disabled_groups,
    origin_disabled_tools,
    tool_names_in_groups,
)

from ._test_data import make_session
from .base import BaseTool
from .describe_capability import DescribeCapabilityTool
from .find_capability import FindCapabilityTool
from .hire_expert import HireExpertTool
from .models import CapabilityDetailsResponse, CapabilityListResponse
from .run_capability import RunCapabilityTool
from .session_registry import _hireable_roster, may_hire, session_expert_entries

USER = "user-1"
_SR = "backend.copilot.tools.session_registry"


def _expert(expert_id: str, name: str, role: str, **extra) -> Expert:
    fields = {
        "avatar_url": None,
        "tagline": None,
        "bio": None,
        "skills": [],
        "identity": "",
        "voice_preferences": "",
        "boundaries": "",
        "protected_soul_rules": [],
        "is_template": False,
        "source_template_id": None,
        "is_archived": False,
        "workflows": [],
    }
    return Expert(id=expert_id, name=name, role=role, **(fields | extra))


JULES = ExpertTemplate(
    **_expert(
        "tpl-jules",
        "Jules",
        "Social & Content Repurposing",
        job_title="Social Media Manager",
        is_template=True,
    ).model_dump(),
    bundled_skills=[
        ExpertBundledSkill(
            id="s1", slug="social", title="Social media management", description=""
        )
    ],
)
VERA = ExpertTemplate(
    **_expert("tpl-vera", "Vera", "Vendor & Procurement", is_template=True).model_dump()
)
HIRED_VERA = _expert(
    "exp-vera", "Vera", "Vendor & Procurement", source_template_id="tpl-vera"
)


@pytest.fixture
def team():
    """Hire-experts on, Vera hired, Jules and Vera on the roster."""
    db = MagicMock()
    db.list_experts = AsyncMock(return_value=[HIRED_VERA])
    db.list_templates = AsyncMock(return_value=[JULES, VERA])
    db.with_bundled_skills = AsyncMock(side_effect=lambda templates, _user: templates)
    _hireable_roster.cache_clear()
    with (
        patch(f"{_SR}.is_feature_enabled", AsyncMock(return_value=True)),
        patch(f"{_SR}.experts_db", MagicMock(return_value=db)),
        patch(f"{_SR}.is_skills_feature_enabled", AsyncMock(return_value=False)),
        patch(
            "backend.copilot.tools.find_capability.load_connection_state",
            AsyncMock(return_value=ConnectionState()),
        ),
    ):
        yield db
    _hireable_roster.cache_clear()


async def test_find_capability_offers_the_roster_expert_toran_asked_for(team):
    result = await FindCapabilityTool()._execute(
        USER, make_session(USER), query="hire expert social media manager"
    )

    assert isinstance(result, CapabilityListResponse)
    experts = [c for c in result.capabilities if c["kind"] == "expert"]
    assert experts[0] == {
        "id": "expert:tpl-jules",
        "name": "Jules",
        "purpose": "Social Media Manager",
        "kind": "expert",
        "hired": False,
    }
    assert "kind=expert" in result.message


async def test_a_hired_template_is_offered_as_the_teammate_only(team):
    result = await FindCapabilityTool()._execute(
        USER, make_session(USER), query="vendor procurement", kind="expert"
    )

    assert isinstance(result, CapabilityListResponse)
    assert [(c["id"], c["hired"]) for c in result.capabilities] == [
        ("teammate:exp-vera", True)
    ]


@pytest.mark.parametrize(
    "capability_id, tool, bound",
    [
        ("expert:tpl-jules", "hire_expert", {"template_id": "tpl-jules"}),
        ("teammate:exp-vera", "delegate_to_expert", {"expert_id": "exp-vera"}),
    ],
)
def test_running_an_expert_id_is_the_call_it_names(capability_id, tool, bound):
    # The model's input cannot point the call at a different expert.
    call = resolve_tool_dispatch(
        "run_capability",
        {"id": capability_id, "input": {**dict.fromkeys(bound, "other"), "x": 1}},
    )

    assert call is not None
    assert call.name == tool
    assert call.args == {"x": 1, **bound}


async def test_describe_expert_asks_only_for_what_the_id_does_not_carry(team):
    result = await DescribeCapabilityTool()._execute(
        USER, make_session(USER), id="expert:tpl-jules"
    )

    assert isinstance(result, CapabilityDetailsResponse)
    assert "calls hire_expert" in result.message
    assert set(result.parameters["properties"]) == {"name"}
    assert result.parameters["required"] == []


@pytest.mark.parametrize("validate_only", [True, False])
async def test_run_capability_itself_describes_an_expert_and_hires_no_one(
    team, validate_only
):
    # The engines dispatch an expert id to hire_expert; a call that reaches
    # run_capability undispatched must not hire past the approval card.
    with patch.object(HireExpertTool, "_execute", AsyncMock()) as hire:
        result = await RunCapabilityTool()._execute(
            USER,
            make_session(USER),
            id="expert:tpl-jules",
            input={},
            validate_only=validate_only,
        )

    assert isinstance(result, CapabilityDetailsResponse)
    assert "calls hire_expert" in result.message
    hire.assert_not_awaited()


async def test_an_expert_ids_validate_only_answer_is_not_held_by_the_judge(team):
    # The answer quotes hire_expert's own instructions to the model; judged
    # whole, the content judge would hold the platform's words as a page's.
    session = make_session(USER)
    session.metadata.autopilot_mode = "auto"
    judge = AsyncMock(return_value=ContentVerdict(held=False))
    with (
        patch("backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=True)),
        patch.object(review_store, "find_review", AsyncMock(return_value=None)),
        patch.object(BaseTool, "_gate", AsyncMock(return_value=(None, False))),
        patch("backend.copilot.gate.reads.judge_content", judge),
    ):
        result = await RunCapabilityTool().execute(
            USER,
            session,
            "call-1",
            id="expert:tpl-jules",
            input={},
            validate_only=True,
        )

    assert result.success and "calls hire_expert" in result.output
    judge.assert_not_awaited()


@pytest.mark.parametrize("expert_id", [None, "exp-1"])
@pytest.mark.parametrize("origin", ["interactive", "automation", None])
def test_the_roster_shows_exactly_where_the_engines_let_hire_expert_run(
    expert_id, origin
):
    session = make_session(USER, expert_id=expert_id)
    session.metadata = ChatSessionMetadata(origin=origin)
    hidden = tool_names_in_groups(
        expert_tool_disabled_groups(experts_enabled=True, expert_id=expert_id)
    ) | origin_disabled_tools(origin)

    assert may_hire(session) == ("hire_expert" not in hidden)
    # Teammates show whenever the flag is on, as delegate_to_expert does.
    assert "delegate_to_expert" not in hidden


async def test_an_expert_session_sees_its_teammates_and_no_roster(team):
    team.list_experts.return_value = [HIRED_VERA, _expert("exp-me", "Me", "Self")]

    entries = await session_expert_entries(USER, make_session(USER, expert_id="exp-me"))

    assert [e.id for e in entries] == ["teammate:exp-vera"]
    team.list_templates.assert_not_called()


async def test_experts_are_hidden_when_the_turn_may_not_run_their_tool():
    index = get_registry().with_entries(expert_entries([JULES], [HIRED_VERA]))
    deny_hire = CopilotPermissions(tools=["hire_expert"])

    ids = index.search("social media vendor", kind="expert", permissions=deny_hire).ids

    assert ids == ["teammate:exp-vera"]
