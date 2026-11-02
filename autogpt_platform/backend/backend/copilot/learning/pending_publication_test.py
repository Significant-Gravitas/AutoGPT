from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools import skills
from backend.copilot.tools.skills import SkillFile
from backend.copilot.tools.skills_test import _FakeWorkspaceManager, _patch_skills_path
from backend.data.skill_publication import VersionDraft
from backend.data.skill_version_files import version_package_hash

from . import nightly_test, publish
from .nightly_test import EXPERT, _run, _source
from .packages import snapshot_files
from .publish_test import BODY_V1, BODY_V2, NAME, USER, _content

adapter = nightly_test.adapter
boundaries = nightly_test.boundaries


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_state", ["old", "partial", "written"])
async def test_another_source_cannot_promote_an_unsettled_package(
    fake_store, monkeypatch, workspace_state
):
    manager = _FakeWorkspaceManager()
    monkeypatch.setattr(skills, "skill_versions_db", lambda: fake_store)
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", skills.read_user_skill_markdown
    )
    monkeypatch.setattr(publish, "read_user_skill_files", skills.read_user_skill_files)
    monkeypatch.setattr(publish, "invalidate_skills_index_cache", AsyncMock())
    monkeypatch.setattr(skills, "invalidate_skills_index_cache", AsyncMock())
    with _patch_skills_path(manager):
        await skills.store_user_skill(
            USER,
            name=NAME,
            description="Import a CSV",
            body=BODY_V1,
            triggers=["csv"],
            files=[SkillFile(relative_path="references/notes.md", content=b"old")],
        )
        head = await fake_store.get_head(USER, "personal", NAME)
        pending = await fake_store.commit_version_safe(
            USER,
            head=head,
            draft=VersionDraft(
                content=_content(BODY_V2),
                files=[],
                description="Import a CSV",
                triggers=["csv"],
                origin="saved_overnight",
                base_version_id=head.current_version_id,
            ),
            expected_current_version=head.current_version,
        )
        if workspace_state != "old":
            await skills.store_user_skill(
                USER,
                name=NAME,
                description="Import a CSV",
                body=BODY_V2 if workspace_state == "written" else BODY_V1,
                triggers=["csv"],
                files=[],
                version_origin=None,
                expected_head=skills.ExpectedHead(version_id=pending.version.id),
            )
        raw = await skills.read_user_skill_markdown(USER, NAME)
        files = await skills.read_user_skill_files(USER, NAME)
        request = publish.PublishRequest(
            user_id=USER,
            expert_id=None,
            skill_name=NAME,
            description="Import a CSV",
            triggers=["csv"],
            body=BODY_V2,
            expected_package_hash=version_package_hash(raw, snapshot_files(files)),
            summary="second source",
            origin="saved_overnight",
        )
        outcome = await publish.publish_learned_version(request)
        assert outcome.status == "write_failed"
        head = await fake_store.get_head(USER, "personal", NAME)
        assert head.current_version_id == pending.version.id
        assert head.auto_improve
        assert len(await fake_store.list_versions(USER, "personal", NAME)) == 2
        assert (
            await fake_store.get_version(USER, pending.version.id)
        ).state == "pending_write"
        assert await skills.read_user_skill_markdown(USER, NAME) == raw
        assert await skills.read_user_skill_files(USER, NAME) == files

        assert await publish.reconcile_pending(USER) == 1
        assert (await fake_store.get_version(USER, pending.version.id)).state == "ready"
        after_recovery = await publish.publish_learned_version(request)
        assert after_recovery.status == (
            "applied" if workspace_state == "written" else "conflict"
        )
        assert (await fake_store.get_head(USER, "personal", NAME)).auto_improve
        assert not any(
            v.origin == "edited"
            for v in await fake_store.list_versions(USER, "personal", NAME)
        )


@pytest.mark.asyncio
async def test_pending_target_keeps_the_next_sources_paid_replay(
    fake_store, adapter, boundaries
):
    source = await _source(fake_store)
    head = await fake_store.ensure_head(USER, EXPERT, NAME)
    pending = await fake_store.commit_version_safe(
        USER,
        head=head,
        draft=VersionDraft(
            content=_content(BODY_V1),
            files=[],
            description="Import a CSV",
            origin="saved_overnight",
        ),
        expected_current_version=0,
    )
    boundaries["write"].side_effect = OSError("workspace offline")
    for _ in range(2):
        assert (await _run()).dispositions == {"write_failed": 1}
        row = await fake_store.get_review_for_revision(
            USER, source.id, source.revision, 1
        )
        assert row.replay is not None
        assert row.model_attempts == 1
        assert (
            await fake_store.get_head(USER, EXPERT, NAME)
        ).current_version_id == pending.version.id
    boundaries["review"].assert_awaited_once()
    boundaries["write"].side_effect = None
    assert (await _run()).dispositions == {"conflict": 1}
    assert (await fake_store.get_version(USER, pending.version.id)).state == "ready"
    assert (await fake_store.get_head(USER, EXPERT, NAME)).auto_improve
    boundaries["review"].assert_awaited_once()
