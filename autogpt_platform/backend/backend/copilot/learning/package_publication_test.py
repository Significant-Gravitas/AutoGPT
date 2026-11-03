from unittest.mock import AsyncMock

import pytest

from backend.copilot.learning import owner_actions, publish
from backend.copilot.tools.skills import ParsedSkill, SkillFile

from .package_test import BODY, SCRIPT, request


@pytest.fixture
def publication_files(monkeypatch):
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", AsyncMock(return_value=None)
    )
    write = AsyncMock()
    monkeypatch.setattr(publish, "store_user_skill", write)
    monkeypatch.setattr(publish, "invalidate_skills_index_cache", AsyncMock())
    return write


@pytest.mark.asyncio
async def test_package_retry_update_and_restore_preserve_file_versions(
    fake_store, publication_files
):
    files = [
        SkillFile(
            relative_path="scripts/normalize.py",
            content=SCRIPT.encode(),
            is_executable=True,
        ),
        SkillFile(relative_path="references/cases.json", content=b'[" A ","B"]'),
    ]
    publication_files.side_effect = OSError("temporary storage failure")
    pending = await publish.publish_learned_version(request(files))
    assert pending.status == "write_failed"
    pending_summary = (await fake_store.list_pending_publications("user-1"))[0]
    pending_version = await fake_store.get_version("user-1", pending_summary.id)
    assert pending_version.files[0].content == SCRIPT.encode()
    publication_files.side_effect = None
    assert await publish.reconcile_pending("user-1") == 1
    assert publication_files.call_args.kwargs["files"] == files
    updated = [
        SkillFile(
            relative_path="scripts/normalize.py",
            content=SCRIPT.replace("lower()", "upper()").encode(),
            is_executable=True,
        )
    ]
    result = await publish.publish_learned_version(request(updated))
    assert result.status == "applied"
    restored = await owner_actions.restore_version(
        user_id="user-1",
        expert_id=None,
        skill_name="normalize-input",
        version_id=pending_version.id,
        actor_user_id="user-1",
    )
    assert restored.status == "applied"
    assert publication_files.call_args.kwargs["files"] == files
    old = await fake_store.get_version("user-1", pending_version.id)
    assert old.files[0].content == SCRIPT.encode()
    assert result.version.files[0].content != old.files[0].content
    again = await publish.publish_learned_version(request(updated))
    assert again.status == "suppressed"
    assert await fake_store.get_version("another-user", old.id) is None
    foreign = await owner_actions.restore_version(
        user_id="another-user",
        expert_id=None,
        skill_name="normalize-input",
        version_id=old.id,
        actor_user_id="another-user",
    )
    assert foreign.status == "conflict"


@pytest.mark.asyncio
async def test_secret_in_generated_file_is_not_persisted(fake_store, publication_files):
    secret = "sk-" + "x" * 30
    files = [SkillFile(relative_path="scripts/normalize.py", content=secret.encode())]
    result = await publish.publish_learned_version(request(files))
    assert result.status == "blocked_content"
    publication_files.assert_not_awaited()
    for version in await fake_store.list_versions(
        "user-1", "personal", "normalize-input"
    ):
        assert not version.files and secret not in version.content


@pytest.mark.asyncio
async def test_permanent_pending_metadata_failure_is_settled(
    fake_store, publication_files
):
    head = await fake_store.ensure_head("user-1", None, "normalize-input")
    pending = await fake_store.commit_version_safe(
        "user-1",
        head=head,
        draft=publish.VersionDraft(
            content="---\nname: normalize-input\ndescription: Normalize input\ntriggers:\n- "
            + "x" * 513
            + "\n---\n"
            + BODY,
            description="Normalize input",
            origin="saved_overnight",
        ),
        expected_current_version=0,
    )
    result = await publish.write_committed_version("user-1", pending.version, None)
    assert result.status == "invalid_proposal"
    assert await fake_store.list_pending_publications("user-1") == []
    publication_files.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_long_trigger_recovers_without_changing_the_saved_proposal(
    fake_store, publication_files
):
    head = await fake_store.ensure_head("user-1", None, "normalize-input")
    trigger = "Replay cursor pagination with rate-limit retries and deduplication by item version"
    content = publish.render_skill_markdown(
        ParsedSkill(
            name="normalize-input",
            description="Normalize input",
            body=BODY,
            triggers=(trigger,),
        )
    )
    pending = await fake_store.commit_version_safe(
        "user-1",
        head=head,
        draft=publish.VersionDraft(
            content=content,
            description="Normalize input",
            triggers=[trigger],
            origin="saved_overnight",
        ),
        expected_current_version=0,
    )
    assert await publish.reconcile_pending("user-1") == 1
    ready = await fake_store.get_version("user-1", pending.version.id)
    assert ready.state == "ready" and ready.content == content
    assert publication_files.call_args.kwargs["triggers"] == [trigger]
