"""Deleting a skill ends its history: neither a source revocation nor
overnight learning brings the deleted skill back (#15540).

The probe from the issue: an independent v1, a v2 learned from chat A, then
the owner deletes the skill and excludes chat A. Revocation used to restore
v1 as a new ready version and write the deleted skill back, and a learned
``create`` for the name conflicted with the head for good.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools import skills
from backend.copilot.tools.skills import SkillFile
from backend.copilot.tools.skills_test import _FakeWorkspaceManager, _patch_skills_path
from backend.data.skill_publication import VersionDraft

from . import owner_actions, publish, publish_test, revocation
from ._fake_store import FakeLearningStore
from .contract import register_source_adapter
from .publish_test import BODY_V1, BODY_V2, KIND, NAME, USER, _Adapter, _content

workspace = publish_test.workspace

OWNER = "personal"
NEAR_COPY = BODY_V2.replace(
    "2. Validate the row count\n",
    "2. Validate the row count\n3. Report the row count\n",
)
DIFFERENT = "## Steps\n1. Parse each line as JSON\n2. Drop duplicate ids\n"
STOPWORDS_ONLY = "## Steps\n1. Use the\n2. And then\n"
SCRIPT = b"print('rows match')\n"
SCRIPT_PATH = f"/skills/{NAME}/scripts/check.py"


def _snapshot(source) -> publish.SourceSnapshot:
    return publish.SourceSnapshot(
        source_id=source.id,
        source_kind=source.source_kind,
        source_ref=source.source_id,
        revision=source.revision,
        epoch=source.epoch,
    )


async def _commit(store: FakeLearningStore, body: str, origin: str, sources):
    head = await store.ensure_head(USER, None, NAME)
    result = await store.commit_version_safe(
        USER,
        head=head,
        expected_current_version=head.current_version,
        draft=VersionDraft(
            content=_content(body),
            description="Import a CSV",
            triggers=["csv"],
            origin=origin,
            base_version_id=head.current_version_id,
            sources=sources,
        ),
    )
    await store.complete_publication(USER, version_id=result.version.id, review_id=None)
    return result.version


class _Skill:
    """An independent v1 and a v2 learned from the ``withdrawn`` chat, with
    v2 in the workspace."""

    def __init__(self, store: FakeLearningStore, monkeypatch) -> None:
        self.store = store
        self.manager = _FakeWorkspaceManager()
        self.manager.files[f"/skills/{NAME}/SKILL.md"] = _content(BODY_V2).encode()
        monkeypatch.setattr(skills, "skill_versions_db", lambda: store)
        monkeypatch.setattr(skills, "skill_deletion_db", lambda: store)

    async def seed(self, *, independent: bool = True, learned: bool = True) -> None:
        register_source_adapter(_Adapter(self.store))
        self.withdrawn, self.later = [
            await self.store.upsert_source_revision(
                USER,
                expert_id=None,
                source_kind=KIND,
                source_id=ref,
                revision="1",
                evidence_refs=[],
                outcome_signals=[],
            )
            for ref in ("withdrawn", "later")
        ]
        if independent:
            self.independent = await _commit(self.store, BODY_V1, "edited", [])
        if learned:
            self.learned = await _commit(
                self.store,
                BODY_V2,
                "saved_overnight",
                [_snapshot(self.withdrawn).as_dependency()],
            )

    async def delete(self) -> None:
        with _patch_skills_path(self.manager):
            await skills.delete_user_skill(USER, NAME)

    async def exclude_withdrawn(self) -> list[str]:
        excluded = await self.store.set_source_eligibility(
            USER, self.withdrawn.id, "excluded", excluded_by_user_id=USER
        )
        return await revocation.invalidate_versions_for_source(
            USER, excluded, "source excluded"
        )


@pytest.mark.asyncio
async def test_excluding_a_source_of_a_deleted_skill_does_not_bring_it_back(
    fake_store, workspace, monkeypatch
):
    skill = _Skill(fake_store, monkeypatch)
    await skill.seed()
    await skill.delete()

    assert await skill.exclude_withdrawn() == [skill.learned.id]

    head = await fake_store.get_head(USER, OWNER, NAME)
    assert head.current_version_id is None
    assert await fake_store.latest_independent_version_id(USER, head.id) is None
    states = {
        v.version: v.state for v in await fake_store.list_versions(USER, OWNER, NAME)
    }
    assert states == {1: "archived", 2: "invalidated", 3: "deleted"}
    workspace.assert_not_awaited()
    assert f"/skills/{NAME}/SKILL.md" not in skill.manager.files


@pytest.mark.asyncio
async def test_a_delete_landing_mid_revocation_is_not_undone(
    fake_store, workspace, monkeypatch
):
    """Revocation reads the head, then restores; a delete in between wins."""
    skill = _Skill(fake_store, monkeypatch)
    await skill.seed()
    restore = owner_actions.restore_version

    async def delete_then_restore(**kwargs):
        await skill.delete()
        return await restore(**kwargs)

    monkeypatch.setattr(revocation, "restore_version", delete_then_restore)

    await skill.exclude_withdrawn()

    head = await fake_store.get_head(USER, OWNER, NAME)
    assert head.current_version_id is None
    workspace.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "status"),
    [(BODY_V2, "suppressed"), (NEAR_COPY, "needs_decision"), (DIFFERENT, "applied")],
    ids=["same", "near-copy", "different"],
)
async def test_learning_brings_back_only_something_different(
    fake_store, workspace, monkeypatch, body, status
):
    skill = _Skill(fake_store, monkeypatch)
    await skill.seed()
    await skill.delete()
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", AsyncMock(return_value=None)
    )

    outcome = await publish.publish_learned_version(
        publish.PublishRequest(
            user_id=USER,
            expert_id=None,
            skill_name=NAME,
            description="Import a CSV",
            triggers=["csv"],
            body=body,
            summary="Learned again",
            origin="saved_overnight",
            sources=[_snapshot(skill.later)],
            create_only=True,
        )
    )

    assert outcome.status == status
    head = await fake_store.get_head(USER, OWNER, NAME)
    if status == "suppressed":
        assert outcome.version is None
        assert head.current_version_id is None
        return
    # A new chain: nothing links it to the deleted history, so revoking a
    # source of that history can never reach it.
    assert outcome.version.base_version_id is None
    assert (head.current_version_id == outcome.version.id) == (status == "applied")
    assert await skill.exclude_withdrawn() == [skill.learned.id]
    after = await fake_store.get_head(USER, OWNER, NAME)
    assert after.current_version_id == head.current_version_id


def _request(body: str, **kwargs) -> publish.PublishRequest:
    return publish.PublishRequest(
        user_id=USER,
        expert_id=None,
        skill_name=NAME,
        description="Import a CSV",
        triggers=["csv"],
        body=body,
        summary="Learned again",
        origin="saved_overnight",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_a_run_that_read_the_package_before_the_delete_does_not_import_it(
    fake_store, workspace, monkeypatch
):
    skill = _Skill(fake_store, monkeypatch)
    await skill.seed()
    await skill.delete()
    monkeypatch.setattr(
        publish,
        "read_user_skill_markdown",
        AsyncMock(return_value=_content(BODY_V2)),
    )

    outcome = await publish.publish_learned_version(
        _request(DIFFERENT, sources=[_snapshot(skill.later)])
    )

    assert outcome.status == "conflict"
    assert (await fake_store.get_head(USER, OWNER, NAME)).deleted
    origins = [v.origin for v in await fake_store.list_versions(USER, OWNER, NAME)]
    assert "imported" not in origins
    workspace.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_import_that_loses_to_a_deletion_records_nothing(
    fake_store, workspace, monkeypatch
):
    """A skill saved before version history is imported before learning
    changes it. A deletion landing between the run's read and the import's
    commit wins: the stale bytes are not imported over it."""
    monkeypatch.setattr(
        publish,
        "read_user_skill_markdown",
        AsyncMock(return_value=_content(BODY_V1)),
    )
    commit = fake_store.commit_version_safe

    async def delete_first(*args, **kwargs):
        monkeypatch.setattr(fake_store, "commit_version_safe", commit)
        await fake_store.record_skill_deletion(
            USER,
            expert_id=None,
            skill_name=NAME,
            expected_current_version_id=None,
            behaviors=[],
            actor_user_id=USER,
        )
        return await commit(*args, **kwargs)

    monkeypatch.setattr(fake_store, "commit_version_safe", delete_first)

    outcome = await publish.publish_learned_version(_request(BODY_V2))

    assert outcome.status == "write_failed"
    assert (await fake_store.get_head(USER, OWNER, NAME)).deleted
    origins = [v.origin for v in await fake_store.list_versions(USER, OWNER, NAME)]
    assert origins == ["deleted"]
    workspace.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_delete_landing_mid_revocation_is_not_paused(
    fake_store, workspace, monkeypatch
):
    """With no version to fall back to, revocation pauses the skill. A delete
    landing between its read and the pause must not inherit the pause, or a
    later skill under the name would start out hidden."""
    skill = _Skill(fake_store, monkeypatch)
    await skill.seed(independent=False)
    fallback = fake_store.latest_independent_version_id

    async def delete_then_look(*args, **kwargs):
        await skill.delete()
        return await fallback(*args, **kwargs)

    monkeypatch.setattr(fake_store, "latest_independent_version_id", delete_then_look)

    await skill.exclude_withdrawn()

    head = await fake_store.get_head(USER, OWNER, NAME)
    assert head.deleted
    assert head.use_paused_at is None


def _script() -> SkillFile:
    return SkillFile(relative_path="scripts/check.py", content=SCRIPT)


@pytest.mark.asyncio
async def test_the_deleted_package_with_its_files_is_suppressed(
    fake_store, workspace, monkeypatch
):
    skill = _Skill(fake_store, monkeypatch)
    skill.manager.files[SCRIPT_PATH] = SCRIPT
    await skill.seed()
    await skill.delete()
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", AsyncMock(return_value=None)
    )

    outcome = await publish.publish_learned_version(
        _request(
            BODY_V2,
            files=[_script()],
            sources=[_snapshot(skill.later)],
            create_only=True,
        )
    )

    assert outcome.status == "suppressed"


@pytest.mark.asyncio
async def test_a_package_unreadable_at_deletion_still_needs_the_owner(
    fake_store, workspace, monkeypatch
):
    """If the package's files cannot be read when it is deleted, its body
    alone is recorded. The same package proposed again then waits for the
    owner, even when its steps are all stopwords."""
    skill = _Skill(fake_store, monkeypatch)
    skill.manager.files[f"/skills/{NAME}/SKILL.md"] = _content(STOPWORDS_ONLY).encode()
    skill.manager.files[SCRIPT_PATH] = SCRIPT
    await skill.seed(independent=False, learned=False)
    read = skill.manager.read_file

    async def lose_the_script(path: str) -> bytes:
        if path == SCRIPT_PATH:
            raise OSError("blob missing")
        return await read(path)

    monkeypatch.setattr(skill.manager, "read_file", lose_the_script)
    await skill.delete()
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", AsyncMock(return_value=None)
    )

    outcome = await publish.publish_learned_version(
        _request(
            STOPWORDS_ONLY,
            files=[_script()],
            sources=[_snapshot(skill.later)],
            create_only=True,
        )
    )

    assert outcome.status == "needs_decision"
    workspace.assert_not_awaited()
