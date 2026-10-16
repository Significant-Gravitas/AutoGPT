"""Recording an owner's deletion against real Postgres (#15540)."""

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
async def test_deleting_a_skill_with_no_history_only_suppresses_it(server):
    user = str(uuid4())
    await _create_user(user)
    try:
        result = await _delete(user, "never-versioned", None)

        assert result.committed and result.marker is None
        assert await versions.get_head(user, OWNER, "never-versioned") is None
        assert len(await use.list_suppressions(user, OWNER, "never-versioned")) == 1
    finally:
        await _cleanup(user)
