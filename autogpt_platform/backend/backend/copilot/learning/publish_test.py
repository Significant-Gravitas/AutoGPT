"""Owner actions on learned versions: edit, restore, decide, revoke."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools.skills import ParsedSkill, render_skill_markdown
from backend.data.skill_publication import VersionDraft

from . import owner_actions, publish, revocation
from ._fake_store import FakeLearningStore
from .contract import (
    Eligibility,
    EligibilityState,
    EvidenceBundle,
    register_source_adapter,
)
from .fingerprint import behavior_fingerprint

USER = "user-1"
EXPERT = "expert-1"
NAME = "csv-import-checks"
KIND = "unit_chat_publish"

BODY_V1 = "## Steps\n1. Open the file as utf-8\n\n## Verification\nrows match\n"
BODY_V2 = "## Steps\n1. Open the file as utf-8\n2. Validate the row count\n\n## Verification\nrows match\n"


def _content(body: str) -> str:
    return render_skill_markdown(
        ParsedSkill(name=NAME, description="Import a CSV", body=body, triggers=("csv",))
    )


class _Adapter:
    kind = KIND
    requires_approval = False

    def __init__(self, store: FakeLearningStore) -> None:
        self.store = store

    async def revalidate(self, *, source_id, revision, scope, approval_event_id):
        record = await self.store.get_source(scope.user_id, source_id)
        if record is None:
            return Eligibility(
                state=EligibilityState.INACCESSIBLE, revision=revision, epoch=-1
            )
        state = (
            EligibilityState.EXCLUDED
            if record.eligibility == "excluded"
            else EligibilityState.ELIGIBLE
        )
        return Eligibility(state=state, revision=record.revision, epoch=record.epoch)

    async def load_evidence(self, source, *, max_chars):
        return EvidenceBundle(source=source)


@pytest.fixture
def workspace(monkeypatch):
    """The workspace registry is a mocked boundary; the store is real."""
    write = AsyncMock()
    monkeypatch.setattr(publish, "store_user_skill", write)
    monkeypatch.setattr(owner_actions, "store_user_skill", write)
    monkeypatch.setattr(publish, "read_skill_bundle_files", AsyncMock(return_value={}))
    monkeypatch.setattr(
        owner_actions, "read_user_skill_with_body", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(publish, "invalidate_skills_index_cache", AsyncMock())
    monkeypatch.setattr(revocation, "invalidate_skills_index_cache", AsyncMock())
    return write


async def _seed(store: FakeLearningStore, *, sources: list[dict] | None = None):
    """Two automated versions: v1 then v2 (current)."""
    head = await store.ensure_head(USER, EXPERT, NAME)
    for body in (BODY_V1, BODY_V2):
        head = await store.get_head(USER, EXPERT, NAME)
        result = await store.commit_version_safe(
            USER,
            head=head,
            draft=VersionDraft(
                content=_content(body),
                description="Import a CSV",
                triggers=["csv"],
                origin="saved_overnight",
                summary=f"auto {len(body)}",
                sources=sources or [],
            ),
            expected_current_version=head.current_version,
        )
        await store.complete_publication(
            USER, version_id=result.version.id, review_id=None
        )
    return await store.list_versions(USER, EXPERT, NAME)


@pytest.mark.asyncio
async def test_restore_creates_a_new_version_and_blocks_reappearance(
    fake_store, workspace
):
    versions = await _seed(fake_store)
    v1, v2 = versions[1], versions[0]
    outcome = await owner_actions.restore_version(
        user_id=USER,
        expert_id=EXPERT,
        skill_name=NAME,
        version_id=v1.id,
        actor_user_id=USER,
    )
    assert outcome.status == "applied" and outcome.version is not None
    restored = outcome.version
    assert restored.version == 3 and restored.origin == "restored"
    assert (
        restored.restored_from_version_id == v1.id and restored.base_version_id == v2.id
    )
    assert restored.content == v1.content
    head = await fake_store.get_head(USER, EXPERT, NAME)
    assert head.current_version_id == restored.id
    assert head.auto_improve is False
    workspace.assert_awaited_once()
    # The undone behaviour is suppressed, so the nightly cannot reapply it.
    suppressed = await fake_store.find_suppression(
        USER, EXPERT, NAME, behavior_fingerprint(NAME, BODY_V2)
    )
    assert suppressed is not None and suppressed.reason == "undone by restore"
    # Audit is preserved: v2 keeps its state and content.
    assert (await fake_store.get_version(USER, v2.id)).state == "ready"


@pytest.mark.asyncio
async def test_restore_refuses_invalidated_or_unavailable_versions(
    fake_store, workspace
):
    versions = await _seed(fake_store)
    await fake_store.set_version_state(USER, versions[1].id, "invalidated", "revoked")
    outcome = await owner_actions.restore_version(
        user_id=USER,
        expert_id=EXPERT,
        skill_name=NAME,
        version_id=versions[1].id,
        actor_user_id=USER,
    )
    assert outcome.status == "stale_eligibility"
    workspace.assert_not_called()


@pytest.mark.asyncio
async def test_owner_edit_is_immediate_and_turns_auto_improve_off(
    fake_store, workspace, monkeypatch
):
    await _seed(fake_store)
    # ``store_user_skill`` is mocked; simulate the registry's version record.
    from backend.copilot.learning import history

    async def fake_write(user_id, **kwargs):
        await history.record_registry_write(
            user_id,
            expert_id=kwargs["expert_id"],
            skill_name=kwargs["name"],
            rendered=_content(kwargs["body"]),
            description=kwargs["description"],
            triggers=kwargs["triggers"],
            origin=kwargs["version_origin"],
            actor_user_id=kwargs["actor_user_id"],
            summary=kwargs["summary"],
            keep_auto_improve=kwargs["keep_auto_improve"],
        )

    workspace.side_effect = fake_write
    seen = (await fake_store.get_head(USER, EXPERT, NAME)).current_version_id
    outcome = await owner_actions.apply_owner_edit(
        user_id=USER,
        expert_id=EXPERT,
        skill_name=NAME,
        description="Import a CSV",
        body=BODY_V1,
        triggers=["csv"],
        keep_auto_improve=False,
        allowed_pattern_classes=[],
        expected_version_id=seen,
    )
    assert outcome.status == "applied" and outcome.version.origin == "edited"
    head = await fake_store.get_head(USER, EXPERT, NAME)
    assert head.auto_improve is False and head.current_version == 3
    assert await fake_store.find_suppression(
        USER, EXPERT, NAME, behavior_fingerprint(NAME, BODY_V2)
    )

    outcome = await owner_actions.apply_owner_edit(
        user_id=USER,
        expert_id=EXPERT,
        skill_name=NAME,
        description="Import a CSV",
        body=BODY_V2,
        triggers=["csv"],
        keep_auto_improve=True,
        allowed_pattern_classes=[],
        expected_version_id=head.current_version_id,
    )
    assert outcome.status == "applied"
    assert (await fake_store.get_head(USER, EXPERT, NAME)).auto_improve is True


@pytest.mark.asyncio
async def test_decisions_apply_keep_or_go_stale(fake_store, workspace):
    register_source_adapter(_Adapter(fake_store))
    source = await fake_store.upsert_source_revision(
        USER,
        expert_id=EXPERT,
        source_kind=KIND,
        source_id="s",
        revision="3",
        evidence_refs=[],
        outcome_signals=[],
    )
    dependency = {
        "source_id": source.id,
        "source_kind": KIND,
        "source_ref": "s",
        "revision": source.revision,
    }
    versions = await _seed(fake_store, sources=[dependency])
    head = await fake_store.get_head(USER, EXPERT, NAME)
    proposal = await fake_store.create_version(
        USER,
        head=head,
        content=_content(BODY_V2 + "3. Log it\n"),
        description="Import a CSV",
        triggers=["csv"],
        origin="saved_overnight",
        state="needs_decision",
        base_version_id=head.current_version_id,
        sources=[dependency],
    )
    kept = await owner_actions.decide_proposal(
        user_id=USER,
        expert_id=EXPERT,
        skill_name=NAME,
        version_id=proposal.id,
        action="keep_current",
        edited_body=None,
        actor_user_id=USER,
    )
    assert (
        kept.status == "applied"
        and (await fake_store.get_version(USER, proposal.id)).state == "archived"
    )
    assert (await fake_store.get_head(USER, EXPERT, NAME)).current_version == 2

    second = await fake_store.create_version(
        USER,
        head=head,
        content=_content(BODY_V2 + "3. Log it twice\n"),
        description="Import a CSV",
        triggers=["csv"],
        origin="saved_overnight",
        state="needs_decision",
        base_version_id=head.current_version_id,
        sources=[dependency],
    )
    applied = await owner_actions.decide_proposal(
        user_id=USER,
        expert_id=EXPERT,
        skill_name=NAME,
        version_id=second.id,
        action="apply",
        edited_body=None,
        actor_user_id=USER,
    )
    assert applied.status == "applied" and applied.version.version == 5
    assert (await fake_store.get_version(USER, second.id)).state == "archived"

    third = await fake_store.create_version(
        USER,
        head=await fake_store.get_head(USER, EXPERT, NAME),
        content=_content(BODY_V1),
        description="Import a CSV",
        triggers=["csv"],
        origin="saved_overnight",
        state="needs_decision",
        base_version_id=versions[0].id,
        sources=[dependency],
    )
    stale = await owner_actions.decide_proposal(
        user_id=USER,
        expert_id=EXPERT,
        skill_name=NAME,
        version_id=third.id,
        action="apply",
        edited_body=None,
        actor_user_id=USER,
    )
    assert stale.status == "conflict"
    assert (await fake_store.get_version(USER, third.id)).state == "stale"

    fourth = await fake_store.create_version(
        USER,
        head=await fake_store.get_head(USER, EXPERT, NAME),
        content=_content(BODY_V1),
        description="Import a CSV",
        triggers=["csv"],
        origin="saved_overnight",
        state="needs_decision",
        base_version_id=(
            await fake_store.get_head(USER, EXPERT, NAME)
        ).current_version_id,
        sources=[dependency],
    )
    await fake_store.set_source_eligibility(
        USER, source.id, "excluded", excluded_by_user_id=USER
    )
    refused = await owner_actions.decide_proposal(
        user_id=USER,
        expert_id=EXPERT,
        skill_name=NAME,
        version_id=fourth.id,
        action="apply",
        edited_body=None,
        actor_user_id=USER,
    )
    assert refused.status == "stale_eligibility"


@pytest.mark.asyncio
async def test_revoking_a_source_invalidates_dependents_and_restores_or_pauses(
    fake_store, workspace
):
    register_source_adapter(_Adapter(fake_store))
    source_a = await fake_store.upsert_source_revision(
        USER,
        expert_id=EXPERT,
        source_kind=KIND,
        source_id="a",
        revision="1",
        evidence_refs=[],
        outcome_signals=[],
    )
    dep_a = {
        "source_id": source_a.id,
        "source_kind": KIND,
        "source_ref": "a",
        "revision": source_a.revision,
    }
    head = await fake_store.ensure_head(USER, EXPERT, NAME)
    independent = await fake_store.commit_version_safe(
        USER,
        head=head,
        draft=VersionDraft(
            content=_content(BODY_V1), description="d", origin="saved_during_work"
        ),
        expected_current_version=0,
    )
    await fake_store.complete_publication(
        USER, version_id=independent.version.id, review_id=None
    )
    head = await fake_store.get_head(USER, EXPERT, NAME)
    dependent = await fake_store.commit_version_safe(
        USER,
        head=head,
        draft=VersionDraft(
            content=_content(BODY_V2),
            description="d",
            origin="saved_overnight",
            sources=[dep_a],
            base_version_id=head.current_version_id,
        ),
        expected_current_version=1,
    )
    await fake_store.complete_publication(
        USER, version_id=dependent.version.id, review_id=None
    )
    head = await fake_store.get_head(USER, EXPERT, NAME)
    edited = await fake_store.commit_version_safe(
        USER,
        head=head,
        draft=VersionDraft(
            content=_content(BODY_V2 + "3. tweak\n"),
            description="d",
            origin="edited",
            base_version_id=head.current_version_id,
        ),
        expected_current_version=2,
    )
    await fake_store.complete_publication(
        USER, version_id=edited.version.id, review_id=None
    )

    excluded = await fake_store.set_source_eligibility(
        USER, source_a.id, "excluded", excluded_by_user_id=USER
    )
    invalidated = await revocation.invalidate_versions_for_source(
        USER, excluded, "source excluded"
    )
    assert set(invalidated) == {dependent.version.id, edited.version.id}
    head = await fake_store.get_head(USER, EXPERT, NAME)
    current = await fake_store.get_version(USER, head.current_version_id)
    assert (
        current.origin == "restored"
        and current.restored_from_version_id == independent.version.id
    )
    assert head.use_paused_at is None

    # No independent fallback: the skill is paused with an explanation.
    other = await fake_store.upsert_source_revision(
        USER,
        expert_id=EXPERT,
        source_kind=KIND,
        source_id="b",
        revision="1",
        evidence_refs=[],
        outcome_signals=[],
    )
    dep_b = {
        "source_id": other.id,
        "source_kind": KIND,
        "source_ref": "b",
        "revision": other.revision,
    }
    head2 = await fake_store.ensure_head(USER, EXPERT, "other-skill")
    only = await fake_store.commit_version_safe(
        USER,
        head=head2,
        draft=VersionDraft(
            content=_content(BODY_V1),
            description="d",
            origin="saved_overnight",
            sources=[dep_b],
        ),
        expected_current_version=0,
    )
    await fake_store.complete_publication(
        USER, version_id=only.version.id, review_id=None
    )
    excluded_b = await fake_store.set_source_eligibility(
        USER, other.id, "excluded", excluded_by_user_id=USER
    )
    await revocation.invalidate_versions_for_source(USER, excluded_b, "source excluded")
    head2 = await fake_store.get_head(USER, EXPERT, "other-skill")
    assert head2.use_paused_at is not None
    assert (await fake_store.get_version(USER, only.version.id)).state == "invalidated"
