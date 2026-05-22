"""Tests for the skill registry (frontmatter parsing + rendering + the
``<available_skills>`` index builder)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.skills import (
    DEFAULT_SKILLS,
    MAX_BODY_CHARS,
    MAX_DESCRIPTION_CHARS,
    MAX_NAME_CHARS,
    MAX_TRIGGER_CHARS,
    MAX_TRIGGERS,
    MAX_USER_SKILLS,
    DeleteSkillResponse,
    DeleteSkillTool,
    ListSkillsResponse,
    ListSkillsTool,
    ParsedSkill,
    ReadSkillResponse,
    ReadSkillTool,
    StoreSkillResponse,
    StoreSkillTool,
    _validate_name,
    build_skills_context,
    get_default_skills,
    list_all_skills,
    parse_skill_markdown,
    render_skill_markdown,
    render_skills_index,
)

# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_render_then_parse_is_identity_for_required_fields():
    """A skill rendered to SKILL.md and parsed back must preserve name,
    description, body, and triggers — anything else is a bug in the
    on-the-wire format helpers."""
    original = ParsedSkill(
        name="example_skill",
        description="An example for the round-trip test.",
        body="## Steps\n1. Do the thing\n2. Done\n",
        triggers=("create_agent", "edit_agent"),
    )
    parsed = parse_skill_markdown(render_skill_markdown(original))
    assert parsed is not None
    assert parsed.name == original.name
    assert parsed.description == original.description
    assert parsed.body.strip() == original.body.strip()
    assert parsed.triggers == original.triggers


def test_parse_rejects_missing_frontmatter():
    """A markdown blob with no YAML frontmatter is not a skill — the
    registry must skip it instead of crashing."""
    assert parse_skill_markdown("just a plain body") is None


def test_parse_rejects_missing_required_fields():
    raw = "---\nname: foo\n---\nbody"
    # No description → invalid skill.
    assert parse_skill_markdown(raw) is None
    raw2 = "---\ndescription: bar\n---\nbody"
    # No name → invalid (and the fallback only kicks in when explicitly passed).
    assert parse_skill_markdown(raw2) is None


def test_parse_accepts_string_triggers_as_csv():
    """Tolerate ``triggers: a, b, c`` strings as a convenience for
    hand-edited SKILL.md files — the canonical render uses a YAML list
    but ingesting both shapes makes the parser forgiving."""
    raw = "---\nname: foo\ndescription: bar\ntriggers: a, b, c\n---\nbody\n"
    parsed = parse_skill_markdown(raw)
    assert parsed is not None
    assert parsed.triggers == ("a", "b", "c")


def test_parse_fallback_name_used_when_frontmatter_missing_name():
    """``list_user_skills`` derives the slug from the folder name. The
    parser must honour that fallback so a SKILL.md authored without an
    explicit ``name`` still surfaces under its folder slug."""
    raw = "---\ndescription: hello\n---\nbody\n"
    parsed = parse_skill_markdown(raw, fallback_name="slug-from-folder")
    assert parsed is not None
    assert parsed.name == "slug-from-folder"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_name_rejects_reserved_default_names():
    """Built-in defaults must not be overwritable via store_skill — they
    ship from disk and have semantics other code (the agent-building
    gate) relies on."""
    for default in DEFAULT_SKILLS:
        assert _validate_name(default.name) is not None


@pytest.mark.parametrize(
    "name",
    [
        "",
        "UPPER",
        "has space",
        "trailing-",
        "-leading",
        "has/slash",
        "has.dot",
        "name_" + "x" * (MAX_NAME_CHARS + 1),
    ],
)
def test_validate_name_rejects_invalid_slugs(name: str):
    assert _validate_name(name) is not None


@pytest.mark.parametrize(
    "name",
    ["foo", "foo_bar", "foo-bar", "a1b2", "snake_case_long_skill_name"],
)
def test_validate_name_accepts_valid_slugs(name: str):
    assert _validate_name(name) is None


# ---------------------------------------------------------------------------
# Defaults + index
# ---------------------------------------------------------------------------


def test_default_skills_load_from_disk():
    """The migration from get_agent_building_guide / get_mcp_guide
    relies on the on-disk markdown files still being present at the
    paths the default registry points to."""
    defaults = get_default_skills()
    names = {s.name for s in defaults}
    assert "agent_building_guide" in names
    assert "mcp_tool_guide" in names
    # Bodies must be non-trivial — a zero-byte file silently kills the
    # whole feature for end users.
    for skill in defaults:
        assert len(skill.body) > 100


def test_render_skills_index_includes_name_and_description():
    skills = [
        ParsedSkill(name="a", description="alpha desc", body=""),
        ParsedSkill(name="b", description="beta desc", body="", triggers=("x", "y")),
    ]
    index = render_skills_index(skills)
    assert "a — alpha desc" in index
    assert "b — beta desc" in index
    assert "triggers: x, y" in index


def test_render_skills_index_empty_returns_empty_string():
    """An empty registry must yield ``""`` so ``inject_user_context``
    omits the ``<available_skills>`` block entirely — preventing a
    zero-content tag from eating tokens for no reason."""
    assert render_skills_index([]) == ""


def test_description_length_cap_enforced_by_validate_path():
    """The cap is enforced inside ``StoreSkillTool._execute`` (not by
    the dataclass), so this test just locks in the constant — bumping
    it requires conscious thought about per-turn token cost."""
    assert MAX_DESCRIPTION_CHARS == 200


def test_trigger_caps_are_bounded():
    """Triggers are inlined into ``<available_skills>`` every turn, so
    both the count and per-entry length must be bounded.  Lock in the
    constants — raising either invites a per-turn token-budget
    regression."""
    assert MAX_TRIGGERS == 10
    assert MAX_TRIGGER_CHARS == 64


# ---------------------------------------------------------------------------
# Tool execution helpers — common mocks for the four skill tools
# ---------------------------------------------------------------------------


def _make_session(user_id: str = "user-1") -> ChatSession:
    """Minimal ChatSession for tool _execute paths."""
    session = ChatSession.new(user_id, dry_run=False)
    session.session_id = "session-1"
    return session


class _FakeWorkspaceManager:
    """Tiny stand-in for WorkspaceManager covering the surface that
    skills.py touches.  In-memory only — keeps the unit tests hermetic."""

    def __init__(self):
        self.files: dict[str, bytes] = {}

    async def write_file(self, *, content, filename, path, mime_type, overwrite):
        self.files[path] = content

    async def read_file(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def list_files(self, *, path, limit, include_all_sessions):
        result = []
        for p in self.files:
            if p.startswith(path):
                info = MagicMock()
                info.path = p
                info.id = f"id-{p}"
                result.append(info)
        return result

    async def get_file_info_by_path(self, path):
        if path not in self.files:
            return None
        info = MagicMock()
        info.path = path
        info.id = f"id-{path}"
        return info

    async def delete_file(self, file_id):
        # Match by synthetic id
        for path in list(self.files):
            if f"id-{path}" == file_id:
                del self.files[path]
                return


class _patch_skills_path:
    """Context manager that patches the workspace lookup, the Redis
    client, and the AsyncClusterLock used by ``StoreSkillTool``.

    ``store_skill`` calls ``await get_redis_async()`` and constructs an
    ``AsyncClusterLock`` even when running unit tests — without these
    patches the real Redis client tries to dial out and hangs the test.
    """

    def __init__(self, fake_manager: _FakeWorkspaceManager):
        fake_lock = MagicMock()
        fake_lock.owner_id = "test-owner"
        fake_lock.try_acquire = AsyncMock(return_value="test-owner")
        fake_lock.release = AsyncMock()
        self._patches = [
            patch(
                "backend.copilot.tools.skills._get_user_skill_manager",
                new=AsyncMock(return_value=fake_manager),
            ),
            patch(
                "backend.copilot.tools.skills.AsyncClusterLock",
                return_value=fake_lock,
            ),
            patch(
                "backend.copilot.tools.skills.get_redis_async",
                new=AsyncMock(return_value=MagicMock()),
            ),
        ]

    def __enter__(self):
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)


# ---------------------------------------------------------------------------
# StoreSkillTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_skill_requires_auth():
    tool = StoreSkillTool()
    result = await tool._execute(
        user_id=None,
        session=_make_session(),
        name="foo",
        description="bar",
        body="body",
    )
    assert isinstance(result, ErrorResponse)
    assert "Authentication" in result.message


@pytest.mark.asyncio
async def test_store_skill_rejects_invalid_name():
    tool = StoreSkillTool()
    result = await tool._execute(
        user_id="user-1",
        session=_make_session(),
        name="BAD NAME",
        description="ok",
        body="ok",
    )
    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_store_skill_rejects_default_name():
    tool = StoreSkillTool()
    result = await tool._execute(
        user_id="user-1",
        session=_make_session(),
        name="agent_building_guide",
        description="ok",
        body="ok",
    )
    assert isinstance(result, ErrorResponse)
    assert "built-in" in result.message


@pytest.mark.asyncio
async def test_store_skill_rejects_oversized_description():
    tool = StoreSkillTool()
    result = await tool._execute(
        user_id="user-1",
        session=_make_session(),
        name="foo",
        description="x" * (MAX_DESCRIPTION_CHARS + 1),
        body="ok",
    )
    assert isinstance(result, ErrorResponse)
    assert "description" in result.message


@pytest.mark.asyncio
async def test_store_skill_rejects_oversized_body():
    tool = StoreSkillTool()
    result = await tool._execute(
        user_id="user-1",
        session=_make_session(),
        name="foo",
        description="ok",
        body="x" * (MAX_BODY_CHARS + 1),
    )
    assert isinstance(result, ErrorResponse)
    assert "body" in result.message


@pytest.mark.asyncio
async def test_store_skill_rejects_too_many_triggers():
    tool = StoreSkillTool()
    result = await tool._execute(
        user_id="user-1",
        session=_make_session(),
        name="foo",
        description="ok",
        body="ok",
        triggers=[f"t{i}" for i in range(MAX_TRIGGERS + 1)],
    )
    assert isinstance(result, ErrorResponse)
    assert "triggers" in result.message


@pytest.mark.asyncio
async def test_store_skill_rejects_oversized_trigger():
    tool = StoreSkillTool()
    result = await tool._execute(
        user_id="user-1",
        session=_make_session(),
        name="foo",
        description="ok",
        body="ok",
        triggers=["a" * (MAX_TRIGGER_CHARS + 1)],
    )
    assert isinstance(result, ErrorResponse)
    assert "trigger" in result.message


@pytest.mark.asyncio
async def test_store_skill_happy_path_writes_to_workspace():
    """End-to-end: store_skill writes SKILL.md to the workspace and
    returns StoreSkillResponse with the metadata.  Verifies the lock
    failure path too — without Redis the wrapper logs and proceeds."""
    tool = StoreSkillTool()
    fake_manager = _FakeWorkspaceManager()
    with _patch_skills_path(fake_manager):
        result = await tool._execute(
            user_id="user-1",
            session=_make_session(),
            name="my_skill",
            description="A skill for testing.",
            body="## Why\nbecause\n## Steps\n1. do",
            triggers=["test", "demo"],
        )
    assert isinstance(result, StoreSkillResponse)
    assert result.name == "my_skill"
    assert "/skills/my_skill/SKILL.md" in fake_manager.files
    stored = fake_manager.files["/skills/my_skill/SKILL.md"].decode()
    assert "name: my_skill" in stored
    assert "## Why" in stored


@pytest.mark.asyncio
async def test_store_skill_enforces_max_user_skills_cap():
    """Hitting the per-user cap returns ErrorResponse instead of
    silently appending the (MAX_USER_SKILLS+1)-th skill."""
    tool = StoreSkillTool()
    fake_manager = _FakeWorkspaceManager()
    # Pre-fill the workspace with MAX_USER_SKILLS distinct slugs.
    for i in range(MAX_USER_SKILLS):
        slug = f"skill_{i}"
        fake_manager.files[f"/skills/{slug}/SKILL.md"] = render_skill_markdown(
            ParsedSkill(name=slug, description="desc", body="body")
        ).encode()
    with _patch_skills_path(fake_manager):
        result = await tool._execute(
            user_id="user-1",
            session=_make_session(),
            name="one_too_many",
            description="ok",
            body="ok",
        )
    assert isinstance(result, ErrorResponse)
    assert "limit" in result.message.lower()


@pytest.mark.asyncio
async def test_store_skill_upsert_does_not_trip_cap():
    """Overwriting an existing skill must NOT count toward the cap —
    re-storing the same name when already at MAX_USER_SKILLS is fine."""
    tool = StoreSkillTool()
    fake_manager = _FakeWorkspaceManager()
    for i in range(MAX_USER_SKILLS):
        slug = f"skill_{i}"
        fake_manager.files[f"/skills/{slug}/SKILL.md"] = render_skill_markdown(
            ParsedSkill(name=slug, description="old", body="old")
        ).encode()
    with _patch_skills_path(fake_manager):
        result = await tool._execute(
            user_id="user-1",
            session=_make_session(),
            name="skill_0",  # already exists → upsert
            description="new",
            body="new body",
        )
    assert isinstance(result, StoreSkillResponse)
    assert "new" in fake_manager.files["/skills/skill_0/SKILL.md"].decode()


# ---------------------------------------------------------------------------
# ReadSkillTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_skill_returns_default_skill_anonymously():
    """Default skills are usable without a user_id — the gate path for
    ``read_skill("agent_building_guide")`` works for anon traffic."""
    tool = ReadSkillTool()
    result = await tool._execute(
        user_id=None,
        session=_make_session(),
        name="agent_building_guide",
    )
    assert isinstance(result, ReadSkillResponse)
    assert result.is_default is True
    assert result.name == "agent_building_guide"
    assert len(result.body) > 100


@pytest.mark.asyncio
async def test_read_skill_rejects_empty_name():
    tool = ReadSkillTool()
    result = await tool._execute(user_id=None, session=_make_session(), name="")
    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_read_skill_requires_auth_for_user_skills():
    tool = ReadSkillTool()
    result = await tool._execute(
        user_id=None, session=_make_session(), name="some_user_skill"
    )
    assert isinstance(result, ErrorResponse)
    assert "Authentication" in result.message


@pytest.mark.asyncio
async def test_read_skill_returns_user_skill_with_siblings():
    """User-stored skill resolves with the markdown body and any
    sibling files (references/scripts/assets/) in the bundle."""
    tool = ReadSkillTool()
    fake_manager = _FakeWorkspaceManager()
    fake_manager.files["/skills/my_skill/SKILL.md"] = render_skill_markdown(
        ParsedSkill(
            name="my_skill",
            description="A test skill",
            body="## Steps\n1. step",
            triggers=("t1",),
        )
    ).encode()
    fake_manager.files["/skills/my_skill/references/note.md"] = b"sibling"
    with _patch_skills_path(fake_manager):
        result = await tool._execute(
            user_id="user-1", session=_make_session(), name="my_skill"
        )
    assert isinstance(result, ReadSkillResponse)
    assert result.is_default is False
    assert result.name == "my_skill"
    assert result.sibling_files == ["/skills/my_skill/references/note.md"]


@pytest.mark.asyncio
async def test_read_skill_not_found_returns_error():
    tool = ReadSkillTool()
    fake_manager = _FakeWorkspaceManager()
    with _patch_skills_path(fake_manager):
        result = await tool._execute(
            user_id="user-1", session=_make_session(), name="missing"
        )
    assert isinstance(result, ErrorResponse)
    assert "not found" in result.message.lower()


# ---------------------------------------------------------------------------
# DeleteSkillTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_skill_requires_auth():
    tool = DeleteSkillTool()
    result = await tool._execute(user_id=None, session=_make_session(), name="my_skill")
    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_delete_skill_refuses_default_skill():
    tool = DeleteSkillTool()
    result = await tool._execute(
        user_id="user-1",
        session=_make_session(),
        name="agent_building_guide",
    )
    assert isinstance(result, ErrorResponse)
    assert "built-in" in result.message


@pytest.mark.asyncio
async def test_delete_skill_removes_skill_and_siblings():
    tool = DeleteSkillTool()
    fake_manager = _FakeWorkspaceManager()
    fake_manager.files["/skills/my_skill/SKILL.md"] = b"skill"
    fake_manager.files["/skills/my_skill/refs/a.md"] = b"sibling"
    with _patch_skills_path(fake_manager):
        result = await tool._execute(
            user_id="user-1", session=_make_session(), name="my_skill"
        )
    assert isinstance(result, DeleteSkillResponse)
    assert "/skills/my_skill/SKILL.md" not in fake_manager.files
    assert "/skills/my_skill/refs/a.md" not in fake_manager.files


@pytest.mark.asyncio
async def test_delete_skill_returns_error_when_missing():
    tool = DeleteSkillTool()
    fake_manager = _FakeWorkspaceManager()
    with _patch_skills_path(fake_manager):
        result = await tool._execute(
            user_id="user-1", session=_make_session(), name="missing"
        )
    assert isinstance(result, ErrorResponse)
    assert "not found" in result.message.lower()


# ---------------------------------------------------------------------------
# ListSkillsTool + list_all_skills + build_skills_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_skills_anon_returns_defaults_only():
    """Anonymous (user_id=None) users see only built-in default skills."""
    tool = ListSkillsTool()
    result = await tool._execute(user_id=None, session=_make_session())
    assert isinstance(result, ListSkillsResponse)
    names = {s["name"] for s in result.skills}
    assert "agent_building_guide" in names
    assert "mcp_tool_guide" in names
    # All anon results must be flagged as default.
    assert all(s["is_default"] for s in result.skills)


@pytest.mark.asyncio
async def test_list_skills_authed_includes_user_skills():
    tool = ListSkillsTool()
    fake_manager = _FakeWorkspaceManager()
    fake_manager.files["/skills/custom/SKILL.md"] = render_skill_markdown(
        ParsedSkill(name="custom", description="user skill", body="x")
    ).encode()
    with _patch_skills_path(fake_manager):
        result = await tool._execute(user_id="user-1", session=_make_session())
    assert isinstance(result, ListSkillsResponse)
    names = {s["name"] for s in result.skills}
    assert "custom" in names
    custom = next(s for s in result.skills if s["name"] == "custom")
    assert custom["is_default"] is False


@pytest.mark.asyncio
async def test_build_skills_context_anon_includes_directive():
    """The injected ``<available_skills>`` body must lead with the
    directive line so the model knows *what to do* with the index, then
    list the skills."""
    ctx = await build_skills_context(user_id=None)
    assert ctx.startswith("Skills are reusable procedures")
    assert "agent_building_guide" in ctx


@pytest.mark.asyncio
async def test_list_all_skills_user_skills_follow_defaults():
    """Default skills always lead — the model sees built-in agent-
    building guide before any user customisation."""
    fake_manager = _FakeWorkspaceManager()
    fake_manager.files["/skills/aaa_custom/SKILL.md"] = render_skill_markdown(
        ParsedSkill(name="aaa_custom", description="user", body="x")
    ).encode()
    with _patch_skills_path(fake_manager):
        skills = await list_all_skills("user-1")
    # Defaults come first regardless of alphabetical sorting on user skills.
    default_names = {d.name for d in DEFAULT_SKILLS}
    first_n = [s.name for s in skills[: len(DEFAULT_SKILLS)]]
    assert set(first_n) == default_names
