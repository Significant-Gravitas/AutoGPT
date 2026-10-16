import asyncio
from uuid import uuid4

import pytest

from backend.data import skill_learning as learning
from backend.data import skill_publication as publication
from backend.data import skill_reviews as reviews
from backend.data import skill_versions as versions
from backend.data.skill_learning import LearningAccessError
from backend.data.skill_learning_test import _cleanup, _create_user, _source
from backend.data.skill_publication import ReviewStamp, VersionDraft


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "disposition", sorted(reviews.TERMINAL_REVIEW_DISPOSITIONS) + ["reviewing"]
)
async def test_replaced_worker_cannot_publish_after_newer_reservation(
    server, disposition
):
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user)
        head = await versions.ensure_head(user, None, "claim-check")
        params = dict(
            source_id=source.id,
            source_revision=source.revision,
            policy_version=1,
            max_attempts=3,
            expected_epoch=source.epoch,
        )
        await reviews.reserve_model_review(user, run_id="old", **params)
        await reviews.reserve_model_review(user, run_id="new", **params)
        if disposition != "reviewing":
            await reviews.upsert_review(
                user,
                source_id=source.id,
                source_revision=source.revision,
                policy_version=1,
                run_id="new",
                disposition=disposition,
            )
            await learning.advance_source_cursor(user, source.id, source.revision)
        before = await reviews.get_review_for_revision(
            user, source.id, source.revision, 1
        )
        late = await publication.commit_version_safe(
            user,
            head=head,
            expected_current_version=0,
            draft=VersionDraft(
                content="body", description="Claim check", origin="saved_overnight"
            ),
            review=ReviewStamp(
                source=source,
                source_revision=source.revision,
                policy_version=1,
                run_id="old",
            ),
        )
        assert not late.committed
        assert (
            await reviews.get_review_for_revision(user, source.id, source.revision, 1)
            == before
        )
        assert (
            await versions.get_head(user, "personal", "claim-check")
        ).current_version_id is None
        assert await versions.list_versions(user, "personal", "claim-check") == []
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_review_claim_records_when_it_was_taken(server):
    """The nightly pass reads updated_at to tell a claim another run may
    still be using from one a dead run left behind."""
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user)
        params = dict(
            source_id=source.id,
            source_revision=source.revision,
            policy_version=1,
            max_attempts=3,
            expected_epoch=source.epoch,
        )
        first = await reviews.reserve_model_review(user, run_id="first", **params)
        await asyncio.sleep(0.01)
        taken_over = await reviews.reserve_model_review(user, run_id="second", **params)
        assert first is not None and taken_over is not None
        assert taken_over.updated_at > first.updated_at
        stored = await reviews.get_review_for_revision(
            user, source.id, source.revision, 1
        )
        assert stored.updated_at == taken_over.updated_at
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_late_standalone_write_cannot_reopen_a_terminal_review(server):
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user)
        params = dict(
            source_id=source.id, source_revision=source.revision, policy_version=1
        )
        terminal = await reviews.upsert_review(
            user, **params, run_id="winner", disposition="skipped"
        )
        late = await reviews.upsert_review(
            user, **params, run_id="late", disposition="provider_error", completed=False
        )
        assert late == terminal
        assert (
            await reviews.get_review_for_revision(user, source.id, source.revision, 1)
            == terminal
        )
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_lost_claim_cannot_settle_or_charge_the_replacement(server):
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user)
        params = dict(
            source_id=source.id, source_revision=source.revision, policy_version=1
        )
        reservation = dict(**params, max_attempts=3, expected_epoch=source.epoch)
        await reviews.reserve_model_review(user, run_id="old", **reservation)
        current = await reviews.reserve_model_review(user, run_id="new", **reservation)
        late = await reviews.upsert_review(
            user,
            **params,
            run_id="old",
            expected_run_id="old",
            disposition="skipped",
            cost_microdollars=2000
        )
        assert late == current
        head = await versions.ensure_head(user, None, "claim-check")
        result = await publication.commit_version_safe(
            user,
            head=head,
            expected_current_version=0,
            draft=VersionDraft(
                content="body", description="Claim check", origin="saved_overnight"
            ),
            review=ReviewStamp(
                source=source,
                source_revision=source.revision,
                policy_version=1,
                run_id="new",
            ),
        )
        assert result.committed
        await publication.complete_publication(
            user, version_id=result.version.id, review_id=result.review_id
        )
        assert (await reviews.get_review(user, current.id)).disposition == "applied"
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_settlement_waits_for_atomic_publication_and_preserves_its_winner(
    server, monkeypatch
):
    user = str(uuid4())
    await _create_user(user)
    locked, release, updating = asyncio.Event(), asyncio.Event(), asyncio.Event()
    check_review = publication._require_no_terminal_review
    lock_source = reviews.lock_review_source

    async def hold_publication(*args):
        await check_review(*args)
        locked.set()
        await release.wait()

    async def signal_update(*args):
        updating.set()
        return await lock_source(*args)

    try:
        source = await _source(user)
        head = await versions.ensure_head(user, None, "claim-race")
        monkeypatch.setattr(
            publication, "_require_no_terminal_review", hold_publication
        )
        writer = asyncio.create_task(
            publication.commit_version_safe(
                user,
                head=head,
                expected_current_version=0,
                draft=VersionDraft(
                    content="body", description="Claim check", origin="saved_overnight"
                ),
                review=ReviewStamp(
                    source=source,
                    source_revision=source.revision,
                    policy_version=1,
                    run_id="winner",
                ),
            )
        )
        await asyncio.wait_for(locked.wait(), 5)
        monkeypatch.setattr(reviews, "lock_review_source", signal_update)
        settler = asyncio.create_task(
            reviews.upsert_review(
                user,
                source_id=source.id,
                source_revision=source.revision,
                policy_version=1,
                run_id="late",
                disposition="skipped",
            )
        )
        try:
            await asyncio.wait_for(updating.wait(), 5)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(settler), 0.05)
        finally:
            release.set()
        committed, settled = await asyncio.gather(writer, settler)
        assert committed.committed
        assert settled.disposition == "applied_pending"
        assert settled.applied_version_id == committed.version.id
    finally:
        release.set()
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_deferral_keeps_the_paid_replay_claim_and_missing_claim_cannot_be_created(
    server,
):
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user)
        params = dict(
            source_id=source.id, source_revision=source.revision, policy_version=1
        )
        with pytest.raises(LearningAccessError, match="claim"):
            await reviews.upsert_review(
                user,
                **params,
                run_id="missing",
                expected_run_id="missing",
                disposition="provider_error",
                completed=False
            )
        paid = await reviews.reserve_model_review(
            user, **params, run_id="paid", max_attempts=3, expected_epoch=source.epoch
        )
        assert await reviews.set_review_replay(user, paid.id, "paid", {"version": 1})
        deferred = await reviews.upsert_review(
            user, **params, run_id="later-pass", disposition="deferred", completed=False
        )
        assert deferred.run_id == "paid" and deferred.replay == {"version": 1}
        assert deferred.model_attempts == 1
        head = await versions.ensure_head(user, None, "paid-replay")
        committed = await publication.commit_version_safe(
            user,
            head=head,
            expected_current_version=0,
            draft=VersionDraft(
                content="body", description="Replay", origin="saved_overnight"
            ),
            review=ReviewStamp(
                source=source,
                source_revision=source.revision,
                policy_version=1,
                run_id="paid",
            ),
        )
        assert committed.committed
    finally:
        await _cleanup(user)
