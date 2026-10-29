from unittest.mock import AsyncMock, Mock

import pytest

from . import nightly, nightly_test, publish
from .nightly_test import USER, _run, _source

adapter = nightly_test.adapter
boundaries = nightly_test.boundaries


@pytest.mark.asyncio
async def test_paid_review_survives_repeated_workspace_read_failure(
    fake_store, adapter, boundaries, monkeypatch
):
    source = await _source(fake_store)
    reader = AsyncMock(side_effect=OSError("workspace temporarily unavailable"))
    charge = AsyncMock(return_value=1000)
    monkeypatch.setattr(publish, "read_user_skill_markdown", reader)
    monkeypatch.setattr(nightly, "record_review_cost", charge)
    for attempt in range(3):
        result = await _run()
        assert result.dispositions == {"provider_error": 1}
        assert result.model_calls == int(attempt == 0)
        boundaries["budget"].return_value = (False, "insufficient_credits")
        row = await fake_store.get_review_for_revision(
            USER, source.id, source.revision, 1
        )
        assert row.input_tokens == 10 and row.output_tokens == 5
        assert row.cost_microdollars == 1000
    reader.side_effect = None
    reader.return_value = None
    recovered = await _run()
    assert recovered.applied == 1 and recovered.model_calls == 0
    boundaries["review"].assert_awaited_once()
    charge.assert_awaited_once()


@pytest.mark.asyncio
async def test_uncacheable_post_review_failure_has_a_durable_attempt_limit(
    fake_store, adapter, boundaries, monkeypatch
):
    source = await _source(fake_store)
    monkeypatch.setattr(
        nightly,
        "build_publish_request",
        Mock(side_effect=RuntimeError("post-review defect")),
    )
    for _ in range(3):
        assert (await _run()).model_calls == 1
        row = await fake_store.get_review_for_revision(
            USER, source.id, source.revision, 1
        )
        assert row is not None and row.cost_microdollars == 1000
    exhausted = await _run()
    assert exhausted.model_calls == 0 and exhausted.dispositions == {"skipped": 1}
    assert boundaries["review"].await_count == 3
    row = await fake_store.get_review_for_revision(USER, source.id, source.revision, 1)
    assert row.cost_microdollars == 1000 and "attempt" in row.reason
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == source.revision


@pytest.mark.asyncio
async def test_rejected_model_content_is_never_saved_in_the_replay(
    fake_store, adapter, boundaries, monkeypatch
):
    from .nightly_test import GOOD_BODY, _completion, _proposal

    source = await _source(fake_store)
    secret = "ghp_" + "s" * 40
    boundaries["review"].return_value = _completion(
        _proposal(body=GOOD_BODY + "\n" + secret)
    )
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", AsyncMock(side_effect=OSError("offline"))
    )
    result = await _run()
    assert result.model_calls == 1
    row = await fake_store.get_review_for_revision(USER, source.id, source.revision, 1)
    assert row.replay is None and secret not in row.model_dump_json()
    assert row.cost_microdollars == 1000


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["epoch", "expired", "wrong-owner"])
async def test_changed_or_expired_replay_is_not_published(
    fake_store, adapter, boundaries, monkeypatch, change
):
    source = await _source(fake_store)
    reader = AsyncMock(side_effect=OSError("offline"))
    monkeypatch.setattr(publish, "read_user_skill_markdown", reader)
    assert (await _run()).model_calls == 1
    row = await fake_store.get_review_for_revision(USER, source.id, source.revision, 1)
    assert row.replay is not None
    if change == "epoch":
        await fake_store.set_source_eligibility(USER, source.id, "eligible")
    else:
        replay = dict(row.replay)
        if change == "expired":
            replay["created_at"] = "2020-01-01T00:00:00Z"
        else:
            replay["request"] = {**replay["request"], "user_id": "other-account"}
        await fake_store.set_review_replay(USER, row.id, row.run_id, replay)
    reader.side_effect = None
    reader.return_value = None
    applied = await _run()
    assert applied.applied == 1 and applied.model_calls == 1
    assert boundaries["review"].await_count == 2
    assert boundaries["write"].await_args.args[0] == USER


@pytest.mark.asyncio
async def test_paid_replay_cannot_overwrite_a_new_target_package(
    fake_store, adapter, boundaries, monkeypatch
):
    from backend.copilot.tools.skills import ParsedSkill, render_skill_markdown

    from .nightly_test import EXPERT

    source = await _source(fake_store)
    reader = AsyncMock(side_effect=OSError("offline"))
    monkeypatch.setattr(publish, "read_user_skill_markdown", reader)
    assert (await _run()).model_calls == 1
    reader.side_effect = None
    reader.return_value = render_skill_markdown(
        ParsedSkill(
            name="csv-import-checks",
            description="New owner procedure",
            body="Owner correction",
        )
    )
    head = await fake_store.ensure_head(USER, EXPERT, "csv-import-checks")
    result = await _run()
    assert result.dispositions == {"conflict": 1} and result.model_calls == 0
    assert (
        await fake_store.get_head(USER, EXPERT, head.skill_name)
    ).current_version_id is None
    boundaries["write"].assert_not_awaited()
    row = await fake_store.get_review_for_revision(USER, source.id, source.revision, 1)
    assert row.replay is None and row.cost_microdollars == 1000


@pytest.mark.asyncio
async def test_lost_review_claim_still_accounts_for_completed_provider_usage(
    fake_store, adapter, boundaries, monkeypatch
):
    source = await _source(fake_store)
    charge = AsyncMock(return_value=1000)
    monkeypatch.setattr(nightly, "record_review_cost", charge)
    monkeypatch.setattr(
        fake_store, "checkpoint_review_usage", AsyncMock(return_value=False)
    )
    result = await _run()
    assert result.dispositions == {"deferred": 1} and result.model_calls == 1
    charge.assert_awaited_once()
    boundaries["write"].assert_not_awaited()
    assert (await fake_store.get_source(USER, source.id)).processed_revision is None


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", ["", "Unparseable skill contents", None])
async def test_create_review_preserves_existing_file_or_pending_target(
    fake_store, adapter, boundaries, monkeypatch, existing
):
    from backend.data.skill_publication import VersionDraft

    from .nightly_test import EXPERT, _completion, _proposal
    from .publish_test import BODY_V1, NAME, _content

    source = await _source(fake_store)
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", AsyncMock(return_value=existing)
    )

    async def create_during_review(*_args):
        if existing is None:
            head = await fake_store.ensure_head(USER, EXPERT, NAME)
            await fake_store.commit_version_safe(
                USER,
                head=head,
                draft=VersionDraft(
                    content=_content(BODY_V1),
                    description="Owner draft",
                    origin="edited",
                ),
                expected_current_version=head.current_version,
            )
        return _completion(_proposal())

    boundaries["review"].side_effect = create_during_review
    result = await _run()
    expected = "write_failed" if existing is None else "conflict"
    assert result.dispositions == {expected: 1} and result.model_calls == 1
    boundaries["write"].assert_not_awaited()
    assert (await fake_store.get_source(USER, source.id)).processed_revision is None
    if existing is None:
        saved = await fake_store.list_versions(USER, EXPERT, NAME)
        assert (
            len(saved) == 1
            and saved[0].origin == "edited"
            and saved[0].state == "pending_write"
        )


@pytest.mark.asyncio
async def test_create_does_not_overwrite_an_undecodable_skill(
    fake_store, adapter, boundaries, monkeypatch
):
    from backend.copilot.tools import skills
    from backend.copilot.tools.skills_test import (
        _FakeWorkspaceManager,
        _patch_skills_path,
    )

    from .nightly_test import EXPERT

    await _source(fake_store)
    manager = _FakeWorkspaceManager()
    path = f"{skills.skill_folder(EXPERT)}/csv-import-checks/SKILL.md"
    manager.files[path] = b"\xff"
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", skills.read_user_skill_markdown
    )
    with _patch_skills_path(manager):
        result = await _run()
    assert result.dispositions == {"skipped": 1} and result.model_calls == 1
    assert manager.files[path] == b"\xff"
    boundaries["write"].assert_not_awaited()
