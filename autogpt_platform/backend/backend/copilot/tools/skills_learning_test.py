"""Skill registry ↔ learning integration.

Every registry write appends a version, content checks run on the whole
bundle before a write, paused skills never reach the index or the model,
an unreadable registry fails closed for the owner's skills only, and a
``read_skill`` reports the exact version it loaded as a load event.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.learning import retrieval
from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.skills import (
    ExpectedHead,
    ReadSkillResponse,
    ReadSkillTool,
    SkillContentBlockedError,
    SkillVersionConflictError,
    SkillWriteLockError,
    build_skills_context,
    build_skills_refresh_context,
    list_all_skills,
    store_user_skill,
)
from backend.copilot.tools.skills_test import _FakeWorkspaceManager, _patch_skills_path

USER = "user-1"
EXPERT = "expert-1"


def _session(expert_id: str | None = None) -> ChatSession:
    now = datetime.now(timezone.utc)
    return ChatSession(
        session_id="sess-1",
        user_id=USER,
        usage=[],
        started_at=now,
        updated_at=now,
        expert_id=expert_id,
        messages=[],
    )


@pytest.fixture
def workspace():
    manager = _FakeWorkspaceManager()
    with _patch_skills_path(manager), patch(
        "backend.copilot.tools.skills.experts_db"
    ) as experts:
        experts.return_value.add_expert_skill_name = AsyncMock()
        experts.return_value.remove_expert_skill_name = AsyncMock()
        yield manager


@pytest.fixture(autouse=True)
def _no_redis_markers(monkeypatch):
    seen: dict[str, str] = {}

    async def read(session_id: str):
        return seen.get(session_id)

    async def mark(session_id: str, revision: str):
        seen[session_id] = revision

    monkeypatch.setattr("backend.copilot.tools.skills.read_seen_index_revision", read)
    monkeypatch.setattr("backend.copilot.tools.skills.mark_index_revision_seen", mark)
    monkeypatch.setattr("backend.copilot.tools.skills.record_skill_loaded", AsyncMock())
    monkeypatch.setattr(
        "backend.copilot.tools.skills.is_skills_feature_enabled",
        AsyncMock(return_value=True),
    )


@pytest.mark.asyncio
async def test_registry_writes_append_versions_with_their_origin(
    workspace, fake_learning_store
):
    await store_user_skill(
        USER, name="deploy-notes", description="d", body="## Steps\n1. go\n"
    )
    head = await fake_learning_store.get_head(USER, "personal", "deploy-notes")
    assert head is not None and head.current_version == 1 and head.auto_improve
    versions = await fake_learning_store.list_versions(USER, "personal", "deploy-notes")
    assert versions[0].origin == "saved_during_work" and versions[0].state == "ready"

    await store_user_skill(
        USER,
        name="deploy-notes",
        description="d",
        body="## Steps\n1. go carefully\n",
        version_origin="imported",
        summary="Uploaded",
    )
    head = await fake_learning_store.get_head(USER, "personal", "deploy-notes")
    assert head.current_version == 2 and head.auto_improve is False
    # Identical content is not a new version.
    await store_user_skill(
        USER,
        name="deploy-notes",
        description="d",
        body="## Steps\n1. go carefully\n",
        version_origin="imported",
    )
    assert (
        await fake_learning_store.get_head(USER, "personal", "deploy-notes")
    ).current_version == 2


@pytest.mark.asyncio
async def test_content_checks_block_the_write_and_the_bundle(workspace):
    with pytest.raises(SkillContentBlockedError) as blocked:
        await store_user_skill(
            USER,
            name="leaky",
            description="d",
            body="## Steps\n1. export TOKEN=ghp_" + "q" * 40 + "\n",
        )
    assert blocked.value.failure.pattern_class == "github_token"
    assert "ghp_" not in str(blocked.value)
    assert "/skills/leaky/SKILL.md" not in workspace.files

    await store_user_skill(USER, name="ok", description="d", body="## Steps\n1. go\n")
    workspace.files["/skills/ok/references/env.md"] = (
        b"password: hunter2hunter2hunter2\n"
    )
    with pytest.raises(SkillContentBlockedError) as bundle_block:
        await store_user_skill(
            USER, name="ok", description="d", body="## Steps\n1. go again\n"
        )
    assert bundle_block.value.failure.file == "bundle file 1 (references/)"
    assert bundle_block.value.failure.pattern_class == "assigned_secret"


@pytest.mark.asyncio
async def test_paused_skill_is_hidden_from_the_index_and_refused_on_read(
    workspace, fake_learning_store
):
    await store_user_skill(
        USER, name="csv-import", description="Import CSV", body="## Steps\n1. go\n"
    )
    await store_user_skill(
        USER, name="other", description="Other", body="## Steps\n1. go\n"
    )
    assert {s.name for s in await list_all_skills(USER)} >= {"csv-import", "other"}
    await fake_learning_store.update_head_policy(
        USER, "personal", "csv-import", use_paused=True
    )
    names = {s.name for s in await list_all_skills(USER)}
    assert "csv-import" not in names and "other" in names
    assert "csv-import" not in await build_skills_context(USER)

    result = await ReadSkillTool()._execute(
        user_id=USER, session=_session(), name="csv-import"
    )
    assert isinstance(result, ErrorResponse) and result.error == "skill_use_paused"


@pytest.mark.asyncio
async def test_registry_outage_fails_closed_for_owner_skills_only(
    workspace, monkeypatch
):
    await store_user_skill(
        USER, name="csv-import", description="Import CSV", body="## Steps\n1. go\n"
    )

    class Broken:
        async def list_use_paused_skill_names(self, *_args):
            raise RuntimeError("registry down")

        async def get_head(self, *_args):
            raise RuntimeError("registry down")

    monkeypatch.setattr(retrieval, "skill_versions_db", lambda: Broken())
    names = {s.name for s in await list_all_skills(USER)}
    assert "csv-import" not in names
    assert "agent_building_guide" in names  # built-ins stay available
    result = await ReadSkillTool()._execute(
        user_id=USER, session=_session(), name="csv-import"
    )
    assert (
        isinstance(result, ErrorResponse)
        and result.error == "skill_registry_unavailable"
    )
    default = await ReadSkillTool()._execute(
        user_id=USER, session=_session(), name="agent_building_guide"
    )
    assert isinstance(default, ReadSkillResponse)


@pytest.mark.asyncio
async def test_read_skill_reports_the_exact_loaded_version(
    workspace, fake_learning_store, monkeypatch
):
    loads = AsyncMock()
    monkeypatch.setattr("backend.copilot.tools.skills.record_skill_loaded", loads)
    await store_user_skill(
        USER, name="csv-import", description="Import CSV", body="## Steps\n1. go\n"
    )
    head = await fake_learning_store.get_head(USER, "personal", "csv-import")
    result = await ReadSkillTool()._execute(
        user_id=USER, session=_session(), name="csv-import"
    )
    assert isinstance(result, ReadSkillResponse)
    assert result.version == 1 and result.version_id == head.current_version_id
    assert (
        result.origin == "saved_during_work"
        and result.origin_label == "Saved during work"
    )
    assert "not a success record" in result.message
    loads.assert_awaited_once()
    assert loads.await_args.kwargs["version_id"] == head.current_version_id

    # A write that bypassed version tracking is reported unversioned.
    workspace.files["/skills/csv-import/SKILL.md"] = (
        b"---\nname: csv-import\ndescription: Import CSV\n---\n\n## Steps\n1. edited by hand\n"
    )
    untracked = await ReadSkillTool()._execute(
        user_id=USER, session=_session(), name="csv-import"
    )
    assert isinstance(untracked, ReadSkillResponse) and untracked.version is None


@pytest.mark.asyncio
async def test_existing_conversation_sees_index_changes_at_the_next_turn(workspace):
    await store_user_skill(
        USER, name="csv-import", description="Import CSV", body="## Steps\n1. go\n"
    )
    first_turn = await build_skills_context(USER, session_id="sess-1")
    assert "csv-import" in first_turn
    assert await build_skills_refresh_context(USER, None, "sess-1") == ""
    await store_user_skill(
        USER,
        name="csv-import",
        description="Import CSV v2",
        body="## Steps\n1. go\n",
        version="2",
    )
    refresh = await build_skills_refresh_context(USER, None, "sess-1")
    assert "changed since this conversation last saw it" in refresh
    assert "csv-import" in refresh
    assert await build_skills_refresh_context(USER, None, "sess-1") == ""


@pytest.mark.asyncio
async def test_versioned_write_checks_the_head_inside_the_write_lock(
    workspace, fake_learning_store
):
    await store_user_skill(
        USER, name="deploy-notes", description="d", body="## Steps\n1. go\n"
    )
    v1 = await fake_learning_store.get_head(USER, "personal", "deploy-notes")
    assert v1.current_version_id is not None

    # A tracked skill: ``None`` is not an acceptable base.
    with pytest.raises(SkillVersionConflictError):
        await store_user_skill(
            USER,
            name="deploy-notes",
            description="d",
            body="## Steps\n1. go faster\n",
            version_origin="edited",
            expected_head=ExpectedHead(version_id=None),
        )
    await store_user_skill(
        USER,
        name="deploy-notes",
        description="d",
        body="## Steps\n1. go faster\n",
        version_origin="edited",
        expected_head=ExpectedHead(version_id=v1.current_version_id),
    )
    v2 = await fake_learning_store.get_head(USER, "personal", "deploy-notes")
    assert v2.current_version == 2

    # A writer prepared against v1 arrives after v2: nothing is written.
    with pytest.raises(SkillVersionConflictError) as conflict:
        await store_user_skill(
            USER,
            name="deploy-notes",
            description="d",
            body="## Steps\n1. stale bytes\n",
            version_origin=None,
            expected_head=ExpectedHead(version_id=v1.current_version_id),
        )
    assert conflict.value.actual == v2.current_version_id
    stored = workspace.files["/skills/deploy-notes/SKILL.md"].decode()
    assert "go faster" in stored and "stale bytes" not in stored
    assert (
        await fake_learning_store.get_head(USER, "personal", "deploy-notes")
    ).current_version == 2


@pytest.mark.asyncio
async def test_versioned_write_has_no_unlocked_fallback(workspace, fake_learning_store):
    await store_user_skill(
        USER, name="deploy-notes", description="d", body="## Steps\n1. go\n"
    )
    head = await fake_learning_store.get_head(USER, "personal", "deploy-notes")
    busy = MagicMock()
    busy.owner_id = "me"
    busy.try_acquire = AsyncMock(return_value="someone-else")
    busy.release = AsyncMock()
    with patch(
        "backend.copilot.tools.skills.AsyncClusterLock", return_value=busy
    ), patch("backend.copilot.tools.skills.asyncio.sleep", AsyncMock()):
        # The ordinary path keeps its best-effort behaviour...
        await store_user_skill(
            USER, name="deploy-notes", description="d", body="## Steps\n1. go on\n"
        )
        # ...but a versioned write refuses to proceed without the lock.
        with pytest.raises(SkillWriteLockError):
            await store_user_skill(
                USER,
                name="deploy-notes",
                description="d",
                body="## Steps\n1. never written\n",
                version_origin=None,
                expected_head=ExpectedHead(version_id=head.current_version_id),
            )
    stored = workspace.files["/skills/deploy-notes/SKILL.md"].decode()
    assert "go on" in stored and "never written" not in stored
