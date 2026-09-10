"""Skill ownership: every skill belongs to one expert or to personal
AutoPilot. Experts see and manage only their own; AutoPilot runs only its own
but may manage any expert's."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import backend.copilot.tools.skills as skills
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
    find_user_skill_slugs,
    invalidate_skills_index_cache,
    list_user_skills,
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
        side_effect=lambda user_id, expert_id: WorkspaceScope(
            expert_id=expert_id, owns_skills_folder=True
        )
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


async def test_library_lookup_resolves_by_folder_and_by_frontmatter_name(world):
    """Only a hand-written skill can be listed under a name that differs from
    its folder, and it is the one case that costs a read."""
    fake, _ = world
    fake.files["/skills/deep-research/SKILL.md"] = _skill("Deep Research")

    found = await find_user_skill_slugs(
        "user-1", ["mine", "Deep Research", "never-stored"]
    )

    assert found["mine"] == "mine"
    assert found["deep research"] == "deep-research"
    assert "never-stored" not in found


async def test_library_lookup_skips_the_read_for_a_folder_carrying_metadata(world):
    """The store-time metadata means folder == name, so the index never pays a
    fetch to learn a name it already has."""
    fake, _ = world
    fake.metadata[AUTOPILOT] = {"kind": "copilot_skill", "description": "mine"}
    reads: list[str] = []
    original = fake.read_file

    async def _counted(path: str) -> bytes:
        reads.append(path)
        return await original(path)

    fake.read_file = _counted

    assert await find_user_skill_slugs("user-1", ["absent"]) == {}
    assert reads == []


async def test_expert_index_heals_an_assignment_made_before_it_owned_skills(world):
    """An expert hired before ownership existed lists names that live only in
    AutoPilot's folder; without a copy it would drop to the defaults."""
    fake, experts = world
    experts.get_expert = AsyncMock(
        side_effect=lambda user_id, expert_id, **_: MagicMock(
            id=expert_id, skills=["mine"]
        )
    )

    ctx = await build_skills_context("user-1", expert_id="expert-a")

    assert "name: mine" in ctx
    assert "/experts/expert-a/skills/mine/SKILL.md" in fake.files
    # AutoPilot keeps its own copy; ownership is a copy, never a move.
    assert AUTOPILOT in fake.files


async def test_storing_a_skill_does_not_backfill_under_its_own_write_lock(world):
    """store_skill's cap check lists the owner's skills while holding the
    per-owner write lock that copying would need, so it must not heal there."""
    fake, experts = world
    experts.get_expert = AsyncMock(
        side_effect=lambda user_id, expert_id, **_: MagicMock(
            id=expert_id, skills=["mine"]
        )
    )

    with patch(
        "backend.copilot.tools.skills.copy_skill_to_expert", new=AsyncMock()
    ) as copy:
        result = await StoreSkillTool()._execute(
            "user-1",
            _expert_session(),
            name="fresh",
            description="a fresh skill",
            body="steps",
        )

    assert isinstance(result, StoreSkillResponse)
    assert "/experts/expert-a/skills/fresh/SKILL.md" in fake.files
    copy.assert_not_awaited()


async def test_expert_index_leaves_a_name_with_no_library_folder_on_the_row(world):
    """A marketplace attachment has no folder to copy, and a storage blip is
    indistinguishable from one, so the name must survive either way."""
    fake, experts = world
    experts.get_expert = AsyncMock(
        side_effect=lambda user_id, expert_id, **_: MagicMock(
            id=expert_id, skills=["a-marketplace-skill"]
        )
    )

    ctx = await build_skills_context("user-1", expert_id="expert-a")

    assert "name: a-marketplace-skill" not in ctx
    experts.remove_expert_skill_name.assert_not_awaited()


async def test_a_name_that_can_never_resolve_is_scanned_for_once(world):
    """A marketplace attachment resolves to no folder on any turn, so without
    a memo every cache-cold turn re-scans AutoPilot's whole library for it."""
    _, experts = world
    experts.get_expert = AsyncMock(
        side_effect=lambda user_id, expert_id, **_: MagicMock(
            id=expert_id, skills=["a-marketplace-skill"]
        )
    )
    scans, counted = _counting_slug_lookup()

    with _cold_turns(_FakeRedis()), patch.object(
        skills, "find_user_skill_slugs", counted
    ):
        first = await list_user_skills("user-1", "expert-a")
        second = await list_user_skills("user-1", "expert-a")

    assert len(scans) == 1
    assert [s.name for s in first] == [s.name for s in second] == ["own"]
    experts.remove_expert_skill_name.assert_not_awaited()


async def test_a_folder_change_lets_the_heal_retry_a_remembered_name(world):
    """The memo is dropped whenever the expert's folder changes, so a name
    that only becomes copyable later still gets its backfill."""
    fake, experts = world
    experts.get_expert = AsyncMock(
        side_effect=lambda user_id, expert_id, **_: MagicMock(
            id=expert_id, skills=["late-arrival"]
        )
    )
    scans, counted = _counting_slug_lookup()
    redis = _FakeRedis()

    with _cold_turns(redis), patch.object(skills, "find_user_skill_slugs", counted):
        await list_user_skills("user-1", "expert-a")
        fake.files["/skills/late-arrival/SKILL.md"] = _skill("late-arrival")
        await invalidate_skills_index_cache("user-1", "expert-a")
        await list_user_skills("user-1", "expert-a")

    assert len(scans) == 2
    assert "/experts/expert-a/skills/late-arrival/SKILL.md" in fake.files


class _FakeRedis:
    """In-memory stand-in: the heal backoff is the one cache these tests have
    to observe, and the fixture's MagicMock swallows every write."""

    def __init__(self):
        self.values: dict[str, str] = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value

    async def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)


@contextmanager
def _cold_turns(redis):
    """Make every listing a cache-cold turn, so the heal runs each time and
    only the backoff can stop the scan."""
    with (
        patch(
            "backend.copilot.tools.skills.get_redis_async",
            new=AsyncMock(return_value=redis),
        ),
        patch(
            "backend.copilot.tools.skills._read_skills_cache",
            new=AsyncMock(return_value=None),
        ),
    ):
        yield


def _counting_slug_lookup():
    scans: list[list[str]] = []
    original = skills.find_user_skill_slugs

    async def counted(user_id, names):
        scans.append(list(names))
        return await original(user_id, names)

    return scans, counted


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
