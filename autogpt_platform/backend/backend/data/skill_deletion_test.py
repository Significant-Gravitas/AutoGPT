"""Recording an owner's deletion against real Postgres (#15540)."""

import asyncio
from uuid import uuid4

import pytest

from backend.data import skill_deletion as deletion
from backend.data import skill_publication as publication
from backend.data import skill_reviews as reviews
from backend.data import skill_use as use
from backend.data import skill_versions as versions
from backend.data.skill_learning_test import _cleanup, _create_user, _source
from backend.data.skill_publication import ReviewStamp, VersionDraft

OWNER = "personal"
BEHAVIOR = deletion.DeletedBehavior(
    behavior_fingerprint="deleted-procedure", behavior_tokens=["learned", "procedure"]
)


async def _publish(user: str, name: str, content: str, **draft):
    head = await versions.ensure_head(user, None, name)
    result = await publication.commit_version_safe(
        user,
        head=head,
        draft=VersionDraft(
            content=content,
            description="Test",
            origin="saved_overnight",
            base_version_id=head.current_version_id,
            **draft,
        ),
        expected_current_version=head.current_version,
    )
    await publication.complete_publication(
        user, version_id=result.version.id, review_id=None
    )
    return result.version


async def _delete(user: str, name: str, expected: str | None):
    return await deletion.record_skill_deletion(
        user,
        expert_id=None,
        skill_name=name,
        expected_current_version_id=expected,
        behaviors=[BEHAVIOR, BEHAVIOR],
        actor_user_id=user,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_deletion_ends_the_history_and_the_next_write_starts_a_new_chain(
    server,
):
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user)
        independent = await _publish(user, "deleted-check", "Independent procedure")
        learned = await _publish(
            user,
            "deleted-check",
            "Learned procedure",
            sources=[{"source_id": source.id}],
        )
        before = await versions.get_head(user, OWNER, "deleted-check")
        proposal = await versions.create_version(
            user,
            head=before,
            content="Proposed procedure",
            description="Test",
            triggers=[],
            origin="saved_overnight",
            state="needs_decision",
            base_version_id=learned.id,
        )
        await versions.update_head_policy(
            user,
            OWNER,
            "deleted-check",
            auto_improve=False,
            learning_paused=True,
            use_paused=True,
        )

        result = await _delete(user, "deleted-check", learned.id)

        assert result.committed
        assert set(result.archived_version_ids) == {
            independent.id,
            learned.id,
            proposal.id,
        }
        marker = result.marker
        assert (marker.state, marker.origin, marker.content, marker.summary) == (
            "deleted",
            "deleted",
            "",
            f"Deleted v{learned.version}",
        )
        head = await versions.get_head(user, OWNER, "deleted-check")
        assert head.current_version_id is None
        assert head.current_version == marker.version
        assert head.content_hash is None
        assert head.auto_improve
        assert head.learning_paused_at is None and head.use_paused_at is None
        assert await versions.latest_independent_version_id(user, head.id) is None
        [suppressed] = await use.list_suppressions(user, OWNER, "deleted-check")
        assert suppressed.behavior_fingerprint == "deleted-procedure"
        assert suppressed.behavior_tokens == ["learned", "procedure"]

        revoked = await versions.invalidate_source_versions(
            user, source.id, "source excluded"
        )
        assert {v.id for v in revoked} == {learned.id, proposal.id}
        assert (
            await versions.get_head(user, OWNER, "deleted-check")
        ).current_version_id is None

        stale = await publication.commit_version_safe(
            user,
            head=before,
            draft=VersionDraft(
                content="Read before the delete",
                description="Test",
                origin="saved_overnight",
                base_version_id=learned.id,
            ),
            expected_current_version=before.current_version,
        )
        assert not stale.committed
        fresh = await _publish(user, "deleted-check", "A different procedure")
        assert fresh.version == marker.version + 1
        assert fresh.base_version_id is None
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_deletion_settles_a_publication_waiting_for_its_write(server):
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user)
        head = await versions.ensure_head(user, None, "pending-check")
        pending = await publication.commit_version(
            user,
            head=head,
            draft=VersionDraft(
                content="Pending procedure",
                description="Test",
                origin="saved_overnight",
                sources=[{"source_id": source.id}],
            ),
            expected_current_version=head.current_version,
            review=ReviewStamp(
                source=source,
                source_revision=source.revision,
                policy_version=1,
                run_id="run-1",
            ),
        )
        assert pending.committed and pending.review_id is not None

        result = await _delete(user, "pending-check", pending.version.id)

        assert result.abandoned_version_ids == [pending.version.id]
        assert (await versions.get_version(user, pending.version.id)).state == "stale"
        review = await reviews.get_review(user, pending.review_id)
        assert review.disposition == "suppressed"
        assert review.completed_at is not None
        assert await publication.list_pending_publications(user) == []
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_publication_landing_mid_deletion_wins_without_a_deadlock(
    server, monkeypatch
):
    """A publication commits the version number the deletion just read. The
    deletion must not hold the head while the publication's insert checks it
    (a lock-order deadlock): the publication completes, and the deletion
    rolls back for its caller to read the head again."""
    user = str(uuid4())
    await _create_user(user)
    try:
        current = await _publish(user, "race-check", "Current procedure")
        head = await versions.get_head(user, OWNER, "race-check")
        allocate = deletion.next_version_number
        raced = {}

        async def publish_meanwhile(head_id: str) -> int:
            number = await allocate(head_id)
            raced["result"] = await asyncio.wait_for(
                publication.commit_version_safe(
                    user,
                    head=head,
                    draft=VersionDraft(
                        content="Published meanwhile",
                        description="Test",
                        origin="saved_overnight",
                        base_version_id=current.id,
                    ),
                    expected_current_version=head.current_version,
                ),
                timeout=10,
            )
            return number

        monkeypatch.setattr(deletion, "next_version_number", publish_meanwhile)

        result = await _delete(user, "race-check", current.id)

        assert raced["result"].committed
        assert not result.committed
        after = await versions.get_head(user, OWNER, "race-check")
        assert after.current_version_id == raced["result"].version.id
        assert await use.list_suppressions(user, OWNER, "race-check") == []
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_publication_completing_mid_deletion_is_still_archived(
    server, monkeypatch
):
    """``complete_publication`` takes no write lock, so a pending version can
    turn ready while the deletion runs. It must still end up archived, not
    left ready for a revocation to restore."""
    user = str(uuid4())
    await _create_user(user)
    try:
        head = await versions.ensure_head(user, None, "completing-check")
        pending = await publication.commit_version_safe(
            user,
            head=head,
            draft=VersionDraft(
                content="Pending procedure", description="Test", origin="edited"
            ),
            expected_current_version=head.current_version,
        )
        settle = deletion.settle_publication_in

        async def complete_meanwhile(tx, user_id, **kwargs):
            await publication.complete_publication(
                user_id, version_id=kwargs["version_id"], review_id=None
            )
            return await settle(tx, user_id, **kwargs)

        monkeypatch.setattr(deletion, "settle_publication_in", complete_meanwhile)

        result = await _delete(user, "completing-check", pending.version.id)

        assert result.committed
        assert result.abandoned_version_ids == []
        assert result.archived_version_ids == [pending.version.id]
        version = await versions.get_version(user, pending.version.id)
        assert version.state == "archived"
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_use_pause_lands_only_while_the_head_names_the_version(server):
    """Revocation's pause for a version with no fallback is a compare-and-swap:
    once a deletion has ended the head, the pause changes nothing."""
    user = str(uuid4())
    await _create_user(user)
    try:
        current = await _publish(user, "pause-check", "Current procedure")
        head = await versions.get_head(user, OWNER, "pause-check")
        assert await versions.pause_use_if_current(user, head.id, current.id)
        assert (await versions.get_head(user, OWNER, "pause-check")).use_paused_at

        await versions.update_head_policy(user, OWNER, "pause-check", use_paused=False)
        assert (await _delete(user, "pause-check", current.id)).committed
        assert not await versions.pause_use_if_current(user, head.id, current.id)
        after = await versions.get_head(user, OWNER, "pause-check")
        assert after.deleted and after.use_paused_at is None
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_deletion_against_a_moved_head_records_nothing(server):
    user = str(uuid4())
    await _create_user(user)
    try:
        current = await _publish(user, "moved-check", "Current procedure")

        result = await _delete(user, "moved-check", "a-version-read-earlier")

        assert not result.committed
        assert await use.list_suppressions(user, OWNER, "moved-check") == []
        head = await versions.get_head(user, OWNER, "moved-check")
        assert head.current_version_id == current.id
        assert [
            v.id for v in await versions.list_versions(user, OWNER, "moved-check")
        ] == [current.id]
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_deleting_a_skill_whose_head_never_named_a_version(server):
    """A head can exist without a current version, e.g. after a blocked
    learning attempt; deleting the skill still ends its history."""
    user = str(uuid4())
    await _create_user(user)
    try:
        head = await versions.ensure_head(user, None, "unpublished-check")
        blocked = await versions.create_version(
            user,
            head=head,
            content="",
            description="Blocked",
            triggers=[],
            origin="saved_overnight",
            state="blocked_content",
        )

        result = await _delete(user, "unpublished-check", None)

        assert result.committed
        assert result.marker.summary == "Deleted"
        after = await versions.get_head(user, OWNER, "unpublished-check")
        assert after.current_version == result.marker.version
        assert after.current_version_id is None
        assert (await versions.get_version(user, blocked.id)).state == "blocked_content"
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_deleting_a_skill_with_no_history_gives_it_a_deleted_head(server):
    """A skill saved before version history has no head. Its deletion still
    leaves one, ended by the marker, so the deletion is on record."""
    user = str(uuid4())
    await _create_user(user)
    try:
        result = await _delete(user, "never-versioned", None)

        assert result.committed
        assert (result.marker.version, result.marker.summary) == (1, "Deleted")
        head = await versions.get_head(user, OWNER, "never-versioned")
        assert head.deleted
        assert head.current_version == 1
        assert len(await use.list_suppressions(user, OWNER, "never-versioned")) == 1
    finally:
        await _cleanup(user)
