"""Skill ownership: every skill belongs to one expert or to personal
AutoPilot. Experts see and manage only their own; AutoPilot runs only its own
but may manage any expert's."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.skills import (
    DeleteSkillResponse,
    DeleteSkillTool,
    ListSkillsResponse,
    ListSkillsTool,
    ParsedSkill,
    ReadSkillResponse,
    ReadSkillTool,
    StoreSkillResponse,
    StoreSkillTool,
    build_skills_context,
    copy_skill_to_expert,
    render_skill_markdown,
)
from backend.copilot.tools.skills_test import _FakeWorkspaceManager, _patch_skills_path
from backend.data.workspace_scope import WorkspaceScope

AUTOPILOT = "/skills/mine/SKILL.md"
EXPERT_A = "/experts/expert-a/skills/own/SKILL.md"
EXPERT_B = "/experts/expert-b/skills/theirs/SKILL.md"


def _skill(name: str) -> bytes:
    return render_skill_markdown(
        ParsedSkill(name=name, description=f"{name} description", body="steps")
    ).encode()


@pytest.fixture
def world():
    fake = _FakeWorkspaceManager()
    fake.files[AUTOPILOT] = _skill("mine")
    fake.files[EXPERT_A] = _skill("own")
    fake.files[EXPERT_B] = _skill("theirs")
    workspace = MagicMock()
    workspace.resolve_expert_workspace_scope = AsyncMock(
        side_effect=lambda user_id, expert_id: WorkspaceScope(expert_id=expert_id)
    )
    experts = MagicMock()
    experts.get_expert = AsyncMock(
        side_effect=lambda user_id, expert_id, **_: (
            MagicMock(id=expert_id) if expert_id in {"expert-a", "expert-b"} else None
        )
    )
    experts.add_expert_skill_name = AsyncMock()
    experts.remove_expert_skill_name = AsyncMock()
    with (
        _patch_skills_path(fake),
        patch("backend.copilot.tools.skills.workspace_db", return_value=workspace),
        patch("backend.copilot.tools.skills.experts_db", return_value=experts),
        patch(
            "backend.copilot.tools.skills.is_skills_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
    ):
        yield fake, experts


def _expert_session(expert_id: str = "expert-a") -> ChatSession:
    return ChatSession.new("user-1", dry_run=False, expert_id=expert_id)


def _personal_session() -> ChatSession:
    return ChatSession.new("user-1", dry_run=False)


async def test_expert_index_holds_defaults_plus_its_own_skills_only(world):
    ctx = await build_skills_context("user-1", expert_id="expert-a")
    assert "agent_building_guide" in ctx
    assert "name: own" in ctx
    assert "mine" not in ctx
    assert "theirs" not in ctx


async def test_autopilot_index_holds_defaults_plus_its_own_skills_only(world):
    ctx = await build_skills_context("user-1")
    assert "name: mine" in ctx
    assert "own" not in ctx.replace("agent_building_guide", "")
    assert "theirs" not in ctx


async def test_expert_reads_its_own_skill_and_not_others(world):
    own = await ReadSkillTool()._execute("user-1", _expert_session(), name="own")
    assert isinstance(own, ReadSkillResponse) and own.expert_id == "expert-a"
    for other in ("mine", "theirs"):
        result = await ReadSkillTool()._execute("user-1", _expert_session(), name=other)
        assert isinstance(result, ErrorResponse)
        assert "not found" in result.message


async def test_expert_cannot_name_another_owner(world):
    for tool in (ReadSkillTool(), DeleteSkillTool(), ListSkillsTool()):
        result = await tool._execute(
            "user-1", _expert_session(), name="theirs", expert_id="expert-b"
        )
        assert isinstance(result, ErrorResponse) and result.error == "access_denied"
    stored = await StoreSkillTool()._execute(
        "user-1",
        _expert_session(),
        name="sneaky",
        description="d",
        body="b",
        expert_id="expert-b",
    )
    assert isinstance(stored, ErrorResponse) and stored.error == "access_denied"


async def test_expert_creates_and_deletes_skills_in_its_own_folder(world):
    fake, experts = world
    stored = await StoreSkillTool()._execute(
        "user-1", _expert_session(), name="learned", description="d", body="b"
    )
    assert isinstance(stored, StoreSkillResponse) and stored.expert_id == "expert-a"
    assert "/experts/expert-a/skills/learned/SKILL.md" in fake.files
    experts.add_expert_skill_name.assert_awaited_once_with(
        "user-1", "expert-a", "learned"
    )

    deleted = await DeleteSkillTool()._execute(
        "user-1", _expert_session(), name="learned"
    )
    assert isinstance(deleted, DeleteSkillResponse)
    assert "/experts/expert-a/skills/learned/SKILL.md" not in fake.files
    experts.remove_expert_skill_name.assert_awaited_once_with(
        "user-1", "expert-a", "learned"
    )
    assert AUTOPILOT in fake.files and EXPERT_B in fake.files


async def test_autopilot_runs_only_its_own_but_manages_any_expert(world):
    fake, _ = world
    listed = await ListSkillsTool()._execute("user-1", _personal_session())
    assert isinstance(listed, ListSkillsResponse)
    assert {s["name"] for s in listed.skills} >= {"mine", "agent_building_guide"}
    assert not {"own", "theirs"} & {s["name"] for s in listed.skills}

    theirs = await ReadSkillTool()._execute(
        "user-1", _personal_session(), name="theirs", expert_id="expert-b"
    )
    assert isinstance(theirs, ReadSkillResponse) and theirs.expert_id == "expert-b"

    stored = await StoreSkillTool()._execute(
        "user-1",
        _personal_session(),
        name="gift",
        description="d",
        body="b",
        expert_id="expert-b",
    )
    assert isinstance(stored, StoreSkillResponse)
    assert "/experts/expert-b/skills/gift/SKILL.md" in fake.files

    unknown = await ReadSkillTool()._execute(
        "user-1", _personal_session(), name="theirs", expert_id="expert-zzz"
    )
    assert isinstance(unknown, ErrorResponse) and unknown.error == "access_denied"


async def test_copy_gives_expert_its_own_copy_of_an_autopilot_skill(world):
    fake, experts = world
    fake.files["/skills/mine/references/notes.md"] = b"notes"
    assert await copy_skill_to_expert("user-1", "expert-a", "mine") == "mine"
    assert "/experts/expert-a/skills/mine/SKILL.md" in fake.files
    assert fake.files["/experts/expert-a/skills/mine/references/notes.md"] == b"notes"
    assert AUTOPILOT in fake.files
    experts.add_expert_skill_name.assert_awaited_once_with("user-1", "expert-a", "mine")
    assert await copy_skill_to_expert("user-1", "expert-a", "nope") is None
