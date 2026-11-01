from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools.skills import ParsedSkill, render_skill_markdown
from backend.data.skill_publication import VersionDraft

from . import nightly, nightly_test, publish, revocation
from .nightly_test import EXPERT, GOOD_BODY, USER, _completion, _proposal, _run, _source
from .publication_state import WITHDRAWN_REASON

adapter = nightly_test.adapter
boundaries = nightly_test.boundaries
NAME = "csv-import-checks"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, EXPERT])
async def test_paused_skill_reuses_its_ledger_until_resumed(
    fake_store, adapter, boundaries, monkeypatch, scope
):
    charge = AsyncMock(return_value=1000)
    monkeypatch.setattr(nightly, "record_review_cost", charge)
    head = await fake_store.ensure_head(USER, scope, NAME)
    await fake_store.update_head_policy(
        USER, head.owner_key, NAME, learning_paused=True
    )
    source = await _source(fake_store, expert_id=scope)
    assert (await _run()).dispositions == {"paused": 1}
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, 1
    )
    for _ in range(3):
        result = await _run()
        assert result.dispositions == {"paused": 1}
        assert result.model_calls == 0
    assert (
        await fake_store.get_review_for_revision(USER, source.id, source.revision, 1)
        == ledger
    )
    assert (await fake_store.get_source(USER, source.id)).processed_revision is None
    boundaries["review"].assert_awaited_once()
    charge.assert_awaited_once()
    boundaries["write"].assert_not_awaited()

    await fake_store.update_head_policy(
        USER, head.owner_key, NAME, learning_paused=False, use_paused=True
    )
    assert (await _run()).applied == 1
    assert boundaries["review"].await_count == charge.await_count == 1
    assert (
        await fake_store.get_head(USER, head.owner_key, NAME)
    ).use_paused_at is not None


@pytest.mark.asyncio
async def test_new_revision_can_learn_a_different_skill_while_old_target_is_paused(
    fake_store, adapter, boundaries
):
    await fake_store.ensure_head(USER, EXPERT, NAME)
    await fake_store.update_head_policy(USER, EXPERT, NAME, learning_paused=True)
    await _source(fake_store)
    assert (await _run()).dispositions == {"paused": 1}
    await _source(fake_store, revision="5")
    boundaries["review"].return_value = _completion(
        _proposal(skill_name="csv-column-checks")
    )
    assert (await _run()).applied == 1
    assert boundaries["review"].await_count == 2


@pytest.mark.asyncio
async def test_paused_personal_target_does_not_block_the_same_name_for_an_expert(
    fake_store, adapter, boundaries
):
    await fake_store.ensure_head(USER, None, NAME)
    await fake_store.update_head_policy(USER, "personal", NAME, learning_paused=True)
    await _source(fake_store, expert_id=None)
    assert (await _run()).dispositions == {"paused": 1}
    await _source(fake_store, ref="expert-session", expert_id=EXPERT)
    result = await _run()
    assert result.dispositions == {"paused": 1, "applied": 1}
    assert result.model_calls == 1
    assert boundaries["review"].await_count == 2


@pytest.mark.asyncio
async def test_pause_returning_during_a_paid_retry_keeps_its_cost_stamp(
    fake_store, adapter, boundaries
):
    await fake_store.ensure_head(USER, EXPERT, NAME)
    await fake_store.update_head_policy(USER, EXPERT, NAME, learning_paused=True)
    source = await _source(fake_store)
    assert (await _run()).dispositions == {"paused": 1}
    await fake_store.update_head_policy(USER, EXPERT, NAME, learning_paused=False)
    previous = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, 1
    )
    await fake_store.set_review_replay(USER, previous.id, previous.run_id, None)

    async def pause_during_review(*_args):
        await fake_store.update_head_policy(USER, EXPERT, NAME, learning_paused=True)
        return _completion(_proposal())

    boundaries["review"].side_effect = pause_during_review
    assert (await _run()).dispositions == {"paused": 1}
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, 1
    )
    assert ledger.attempts == 2
    assert boundaries["review"].await_count == 2
    assert (await _run()).model_calls == 0


@pytest.mark.asyncio
async def test_a_withdrawn_skill_waits_for_its_owner_without_paying_again(
    fake_store, adapter, boundaries, monkeypatch
):
    """Revocation left the invalidated version current. Later passes reuse
    the ledger instead of paying for a review, even after its kept replay
    expires, until the owner edits the skill."""
    learned_from = await _source(fake_store, ref="withdrawn")
    assert (await _run()).applied == 1
    excluded = await fake_store.set_source_eligibility(
        USER, learned_from.id, "excluded", excluded_by_user_id=USER
    )
    await revocation.invalidate_versions_for_source(USER, excluded, "excluded")
    source = await _source(fake_store, ref="later")

    assert (await _run()).dispositions == {"paused": 1}
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, nightly.POLICY_VERSION
    )
    assert ledger.reason == WITHDRAWN_REASON
    await fake_store.set_review_replay(USER, ledger.id, ledger.run_id, None)
    for _ in range(3):
        result = await _run()
        assert result.dispositions == {"paused": 1}
        assert result.model_calls == 0
    assert (await fake_store.get_source(USER, source.id)).processed_revision is None
    assert boundaries["review"].await_count == 2

    head = await fake_store.get_head(USER, EXPERT, NAME)
    edited_skill = ParsedSkill(
        name=NAME,
        description="Import a CSV with an encoding check",
        body=GOOD_BODY.replace("2. Validate the row count\n", ""),
        triggers=("csv import",),
    )
    content = render_skill_markdown(edited_skill)
    edited = await fake_store.commit_version_safe(
        USER,
        head=head,
        expected_current_version=head.current_version,
        draft=VersionDraft(
            content=content,
            description=edited_skill.description,
            triggers=list(edited_skill.triggers),
            origin="edited",
        ),
    )
    await fake_store.complete_publication(
        USER, version_id=edited.version.id, review_id=None
    )
    monkeypatch.setattr(
        nightly, "_existing_skills", AsyncMock(return_value=[edited_skill])
    )
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", AsyncMock(return_value=content)
    )
    boundaries["review"].return_value = _completion(_proposal(decision="update"))

    resumed = await _run()

    assert resumed.applied == 1 and resumed.model_calls == 1
