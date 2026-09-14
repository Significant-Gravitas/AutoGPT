"""Integration tests for the skill-learning data layer (real Postgres).

Pins the invariants the runtime relies on: pending selection never starves
behind completed rows, cursors never regress, epoch bumps are atomic,
every mutation enforces ownership, evidence merges survive out-of-order
turns, scope changes are not silently re-owned, and publication commits
are transactional with a recoverable outbox.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import pytest
from prisma.errors import UniqueViolationError
from prisma.models import User

from backend.data import skill_learning as learning
from backend.data import skill_publication as publication
from backend.data import skill_reviews as reviews
from backend.data import skill_use as use
from backend.data import skill_versions as versions
from backend.data.skill_learning import LearningAccessError, canonical_revision
from backend.data.skill_publication import ReviewStamp, VersionDraft
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)
KIND = "probe_kind"


async def _create_user(user_id: str) -> None:
    try:
        await User.prisma().create(
            data={"id": user_id, "email": f"{user_id}@example.invalid", "name": "L"}
        )
    except UniqueViolationError:
        pass


async def _cleanup(*user_ids: str) -> None:
    for user_id in user_ids:
        try:
            await User.prisma().delete_many(where={"id": user_id})
        except Exception as exc:
            logger.warning("cleanup for %s failed: %s", user_id, exc)


async def _source(user_id: str, *, expert_id: str | None = None, revision: str = "1"):
    return await learning.upsert_source_revision(
        user_id,
        expert_id=expert_id,
        source_kind=KIND,
        source_id=str(uuid4()),
        revision=revision,
        evidence_refs=[{"kind": "tool", "ref": "msg:1"}],
        outcome_signals=[],
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_pending_sources_are_not_starved_by_completed_rows(
    server: SpinTestServer,
):
    user = str(uuid4())
    await _create_user(user)
    try:
        completed = [await _source(user) for _ in range(5)]
        for source in completed:
            await learning.advance_source_cursor(user, source.id, "1")
        fresh = await _source(user)
        pending = await learning.list_pending_sources(user, limit=2)
        assert [s.id for s in pending] == [fresh.id]
        assert await learning.list_pending_owner_keys(user) == ["personal"]
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_cursor_is_monotonic_and_revisions_never_regress(server: SpinTestServer):
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user, revision="7")
        assert source.revision == canonical_revision("7")
        assert await learning.advance_source_cursor(user, source.id, "7")
        assert not await learning.advance_source_cursor(user, source.id, "3")
        current = await learning.get_source(user, source.id)
        assert current is not None
        assert current.processed_revision == canonical_revision("7")
        # An out-of-order turn merges evidence but keeps the newer revision.
        merged = await learning.upsert_source_revision(
            user,
            expert_id=None,
            source_kind=KIND,
            source_id=source.source_id,
            revision="5",
            evidence_refs=[{"kind": "user", "ref": "msg:5"}],
            outcome_signals=[{"kind": "tool_result", "ref": "msg:5", "label": "ok"}],
        )
        assert merged.revision == canonical_revision("7")
        assert {r["ref"] for r in merged.evidence_refs} == {"msg:1", "msg:5"}
        assert merged.outcome_signals[0]["ref"] == "msg:5"
        assert not merged.has_unprocessed_revision
        newer = await learning.upsert_source_revision(
            user,
            expert_id=None,
            source_kind=KIND,
            source_id=source.source_id,
            revision="12",
            evidence_refs=[],
            outcome_signals=[],
            origin="requested",
        )
        assert newer.revision == canonical_revision("12")
        assert newer.origin == "requested" and newer.has_unprocessed_revision
        ordinary = await learning.upsert_source_revision(
            user,
            expert_id=None,
            source_kind=KIND,
            source_id=source.source_id,
            revision="12",
            evidence_refs=[],
            outcome_signals=[],
        )
        assert ordinary.origin == "requested"
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_epoch_bumps_are_atomic_and_scope_changes_are_not_reowned(
    server: SpinTestServer,
):
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user, expert_id=None)
        excluded = await learning.set_source_eligibility(
            user, source.id, "excluded", excluded_by_user_id=user
        )
        assert excluded is not None and excluded.epoch == 1
        assert excluded.excluded_at is not None
        approved = await learning.set_source_approval(
            user,
            source.id,
            approval_event_id="ev-1",
            approval_actor_id=user,
            approved_revision="1",
            eligibility="eligible",
        )
        assert approved is not None and approved.epoch == 2
        moved = await learning.upsert_source_revision(
            user,
            expert_id="expert-other",
            source_kind=KIND,
            source_id=source.source_id,
            revision="2",
            evidence_refs=[],
            outcome_signals=[],
        )
        assert moved.eligibility == "inaccessible"
        assert moved.owner_key == "personal" and moved.epoch == 3
        assert moved.revision == canonical_revision("1")
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_another_user_cannot_mutate_known_ids(server: SpinTestServer):
    owner, intruder = str(uuid4()), str(uuid4())
    await _create_user(owner)
    await _create_user(intruder)
    try:
        source = await _source(owner)
        mine = await reviews.upsert_review(
            owner,
            source_id=source.id,
            source_revision="1",
            policy_version=1,
            run_id="owner-run",
            disposition="skipped",
        )
        with pytest.raises(LearningAccessError):
            await reviews.upsert_review(
                intruder,
                source_id=source.id,
                source_revision="1",
                policy_version=1,
                run_id="intruder-run",
                disposition="applied",
            )
        assert await reviews.get_review(intruder, mine.id) is None
        assert await learning.get_source(intruder, source.id) is None
        assert not await learning.advance_source_cursor(intruder, source.id, "1")
        assert (
            await learning.set_source_eligibility(intruder, source.id, "excluded")
            is None
        )
        reloaded = await reviews.get_review(owner, mine.id)
        assert reloaded is not None and reloaded.run_id == "owner-run"
        head = await versions.ensure_head(owner, None, "probe-skill")
        assert await versions.get_head(intruder, "personal", "probe-skill") is None
        assert (
            await versions.update_head_policy(
                intruder, "personal", "probe-skill", use_paused=True
            )
        ) is None
        draft = VersionDraft(
            content="---\nname: probe-skill\ndescription: d\n---\n\nbody\n",
            description="d",
            origin="edited",
        )
        stolen = await publication.commit_version_safe(
            intruder, head=head, draft=draft, expected_current_version=0
        )
        assert not stolen.committed
        assert (
            await versions.get_head(owner, "personal", "probe-skill")
        ).current_version == 0
    finally:
        await _cleanup(owner, intruder)


@pytest.mark.asyncio(loop_scope="session")
async def test_publication_commit_is_transactional_and_recoverable(
    server: SpinTestServer,
):
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user)
        head = await versions.ensure_head(user, None, "probe-skill")
        content = "---\nname: probe-skill\ndescription: d\n---\n\n## Steps\n1. go\n"
        stamp = ReviewStamp(
            source=source, source_revision="1", policy_version=1, run_id="run-1"
        )
        first = await publication.commit_version_safe(
            user,
            head=head,
            draft=VersionDraft(
                content=content, description="d", origin="saved_overnight"
            ),
            expected_current_version=0,
            review=stamp,
        )
        assert first.committed and first.version is not None
        assert first.version.state == "pending_write" and first.version.version == 1
        ledger = await reviews.get_review_for_revision(user, source.id, "1", 1)
        assert ledger is not None and ledger.disposition == "applied_pending"
        assert ledger.applied_version_id == first.version.id
        # A stale writer (still expecting version 0) loses: nothing is written.
        lost = await publication.commit_version_safe(
            user,
            head=head,
            draft=VersionDraft(content=content + "x", description="d", origin="edited"),
            expected_current_version=0,
        )
        assert not lost.committed
        assert len(await versions.list_versions(user, "personal", "probe-skill")) == 1
        pending = await publication.list_pending_publications(user)
        assert [v.id for v in pending] == [first.version.id]
        await publication.complete_publication(
            user, version_id=first.version.id, review_id=first.review_id, reason="ok"
        )
        assert (await versions.get_version(user, first.version.id)).state == "ready"
        assert (
            await reviews.get_review_for_revision(user, source.id, "1", 1)
        ).disposition == "applied"
        assert await publication.list_pending_publications(user) == []
        # Retry on the same revision updates the same ledger row (no duplicate).
        again = await reviews.upsert_review(
            user,
            source_id=source.id,
            source_revision="1",
            policy_version=1,
            run_id="run-2",
            disposition="applied",
        )
        assert again.id == ledger.id and again.attempts == 2
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_suppressions_and_use_events_round_trip(server: SpinTestServer):
    user = str(uuid4())
    await _create_user(user)
    try:
        record = await use.add_suppression(
            user,
            expert_id=None,
            skill_name="probe-skill",
            behavior_fingerprint="fp-1",
            behavior_tokens=["encoding", "not", "retry"],
            evidence_fingerprints=["ev-1"],
            actor_user_id=user,
            reason="undone",
        )
        found = await use.find_suppression(user, "personal", "probe-skill", "fp-1")
        assert found is not None and found.behavior_tokens == [
            "encoding",
            "not",
            "retry",
        ]
        duplicate = await use.add_suppression(
            user,
            expert_id=None,
            skill_name="probe-skill",
            behavior_fingerprint="fp-1",
            behavior_tokens=[],
            evidence_fingerprints=[],
            actor_user_id=user,
            reason="again",
        )
        assert duplicate.id == record.id
        await use.record_use_event(
            user,
            expert_id=None,
            skill_name="probe-skill",
            kind="loaded",
            version_id="v",
        )
        events = await use.list_use_events(user, "personal", "probe-skill")
        assert [e.kind for e in events] == ["loaded"]
        summary = await reviews.summarize_learning(user)
        assert summary.pending_sources == 0
    finally:
        await _cleanup(user)
