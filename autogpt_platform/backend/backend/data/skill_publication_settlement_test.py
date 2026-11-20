from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from backend.data import skill_publication as publication
from backend.data import skill_reviews as reviews
from backend.data import skill_versions as versions
from backend.data.skill_learning_test import _cleanup, _create_user, _source
from backend.data.skill_publication import ReviewStamp, VersionDraft


async def _pending(user):
    source = await _source(user)
    head = await versions.ensure_head(user, None, "settlement-check")
    committed = await publication.commit_version_safe(
        user,
        head=head,
        draft=VersionDraft(
            content="body", description="Check settlement", origin="saved_overnight"
        ),
        expected_current_version=0,
        review=ReviewStamp(
            source=source,
            source_revision="1",
            policy_version=1,
            run_id="settlement-test",
        ),
    )
    assert committed.committed and committed.version and committed.review_id
    return committed.version, committed.review_id


async def _settle(user, version, review, action):
    if action == "complete":
        await publication.complete_publication(
            user, version_id=version.id, review_id=review, reason="written"
        )
    else:
        await publication.abandon_publication(
            user, version_id=version.id, review_id=review, reason="superseded"
        )


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("winner", ["complete", "abandon"])
async def test_late_settlement_cannot_overwrite_the_winner(server, winner):
    user = str(uuid4())
    await _create_user(user)
    try:
        version, review = await _pending(user)
        await _settle(user, version, review, winner)
        await _settle(
            user, version, review, "abandon" if winner == "complete" else "complete"
        )
        saved = await versions.get_version(user, version.id)
        ledger = await reviews.get_review(user, review)
        assert saved and ledger
        assert saved.state == ("ready" if winner == "complete" else "stale")
        assert ledger.disposition == ("applied" if winner == "complete" else "conflict")
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("action", ["complete", "abandon"])
async def test_review_failure_rolls_back_version_settlement(
    server, monkeypatch, action
):
    user = str(uuid4())
    await _create_user(user)
    try:
        version, review = await _pending(user)
        monkeypatch.setattr(
            "prisma.actions.SkillLearningReviewActions.update_many",
            AsyncMock(side_effect=RuntimeError("ledger unavailable")),
        )
        with pytest.raises(RuntimeError, match="ledger unavailable"):
            await _settle(user, version, review, action)
        saved = await versions.get_version(user, version.id)
        ledger = await reviews.get_review(user, review)
        assert saved and ledger
        assert saved.state == "pending_write"
        assert ledger.disposition == "applied_pending"
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_settlement_does_not_change_another_users_version_or_review(server):
    owner, intruder = str(uuid4()), str(uuid4())
    await _create_user(owner)
    await _create_user(intruder)
    try:
        version, review = await _pending(owner)
        await _settle(intruder, version, review, "complete")
        await _settle(intruder, version, review, "abandon")
        saved = await versions.get_version(owner, version.id)
        ledger = await reviews.get_review(owner, review)
        assert saved and ledger
        assert saved.state == "pending_write"
        assert ledger.disposition == "applied_pending"
    finally:
        await _cleanup(owner, intruder)
