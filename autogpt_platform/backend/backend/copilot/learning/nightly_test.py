"""End-to-end unit tests for the nightly pass against the in-memory store.

The reviewer model, the workspace write, the lease, and billing are the
only mocked boundaries; the ledger, cursor, head pointer, versions, and
suppressions are exercised for real through ``FakeLearningStore``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, ConfigDict, Field

from backend.copilot.dream.llm import (
    CompletionUsage,
    DreamLLMError,
    StructuredCompletion,
)
from backend.copilot.tools.skills import ParsedSkill
from backend.data.skill_learning import canonical_revision

from . import dispositions, nightly, publish
from ._fake_store import FakeLearningStore
from .contract import (
    Eligibility,
    EligibilityState,
    EvidenceBundle,
    EvidenceSpan,
    LearningScope,
    SourceRevision,
    register_source_adapter,
)
from .prompts import LearningProposal

USER = "user-1"
EXPERT = "expert-1"
KIND = "unit_chat"

GOOD_BODY = (
    "## Why\nImports failed on encoding.\n\n## Trigger\ncsv import\n\n"
    "## Prerequisites\nA file path.\n\n## Steps\n1. Open the file as utf-8\n"
    "2. Validate the row count\n\n## Verification\nRow count matches.\n\n## Limits\nNone.\n"
)


class UnitAdapter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = KIND
    requires_approval: bool = False
    store: FakeLearningStore | None = None
    spans: list[EvidenceSpan] = Field(default_factory=list)
    clipped: list[str] = Field(default_factory=list)
    on_load: list = Field(default_factory=list)

    async def revalidate(self, *, source_id, revision, scope, approval_event_id):
        assert self.store is not None
        record = await self.store.get_source(scope.user_id, source_id)
        if record is None or record.owner_key != scope.owner_key:
            return Eligibility(
                state=EligibilityState.INACCESSIBLE, revision=revision, epoch=-1
            )
        if record.eligibility == "excluded":
            return Eligibility(
                state=EligibilityState.EXCLUDED,
                revision=record.revision,
                epoch=record.epoch,
            )
        return Eligibility(
            state=EligibilityState.ELIGIBLE,
            revision=record.revision,
            epoch=record.epoch,
        )

    async def load_evidence(
        self, source: SourceRevision, *, max_chars: int
    ) -> EvidenceBundle:
        for hook in self.on_load:
            await hook(source)
        return EvidenceBundle(
            source=source, spans=list(self.spans), clipped_refs=list(self.clipped)
        )


def _proposal(**overrides) -> LearningProposal:
    base = dict(
        decision="create",
        reason="verified import",
        skill_name="csv-import-checks",
        description="Import a CSV with an encoding check",
        triggers=["csv import"],
        body=GOOD_BODY,
        summary="Added an encoding check after the previous import failed.",
        supported_by=["msg:3"],
        verification="row count matched the tool output",
        limits=[],
    )
    base.update(overrides)
    return LearningProposal(**base)


def _completion(proposal: LearningProposal) -> StructuredCompletion[LearningProposal]:
    return StructuredCompletion(
        value=proposal,
        usage=CompletionUsage(
            model="test-model", input_tokens=10, output_tokens=5, cost_usd=0.001
        ),
    )


@pytest.fixture
def adapter(fake_store: FakeLearningStore) -> UnitAdapter:
    unit = UnitAdapter(store=fake_store)
    unit.spans = [
        EvidenceSpan(ref="msg:1", role="user", text="import this csv"),
        EvidenceSpan(
            ref="msg:3",
            role="tool",
            text='{"type":"block_output","success":true}',
            outcome="tool_result",
        ),
        EvidenceSpan(ref="msg:4", role="assistant", text="Imported 42 rows"),
    ]
    register_source_adapter(unit)
    return unit


@pytest.fixture
def boundaries(monkeypatch):
    """Mock everything outside the learner: lease, flag, billing, model, workspace."""
    review = AsyncMock(return_value=_completion(_proposal()))
    write = AsyncMock()
    memory = AsyncMock(return_value=True)
    budget = AsyncMock(return_value=(True, None))
    monkeypatch.setattr(nightly, "is_feature_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(nightly, "_acquire_lease", AsyncMock(return_value=object()))
    monkeypatch.setattr(nightly, "_release_lease", AsyncMock())
    monkeypatch.setattr(nightly, "check_dream_budget", budget)
    monkeypatch.setattr(nightly, "review_evidence", review)
    monkeypatch.setattr(nightly, "record_review_cost", AsyncMock(return_value=1000))
    monkeypatch.setattr(
        dispositions, "record_review_cost", AsyncMock(return_value=1000)
    )
    monkeypatch.setattr(nightly, "_existing_skills", AsyncMock(return_value=[]))
    monkeypatch.setattr(dispositions, "link_version_to_memory", memory)
    monkeypatch.setattr(publish, "store_user_skill", write)
    monkeypatch.setattr(publish, "read_skill_bundle_files", AsyncMock(return_value={}))
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(publish, "invalidate_skills_index_cache", AsyncMock())
    return {"review": review, "write": write, "memory": memory, "budget": budget}


async def _source(
    store: FakeLearningStore,
    *,
    ref: str = "session-1",
    expert_id: str | None = EXPERT,
    verified: bool = True,
    revision: str = "4",
):
    signals = (
        [{"kind": "tool_result", "ref": "msg:3", "label": "checked outcome"}]
        if verified
        else []
    )
    return await store.upsert_source_revision(
        USER,
        expert_id=expert_id,
        source_kind=KIND,
        source_id=ref,
        revision=revision,
        evidence_refs=[{"kind": "tool", "ref": "msg:3", "label": "tool"}],
        outcome_signals=signals,
    )


async def _run(**kwargs):
    return await nightly.run_skill_learning_pass(USER, **kwargs)


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_eligible_work_means_no_model_call(fake_store, adapter, boundaries):
    result = await _run()
    assert result.skipped and result.skip_reason == "no_eligible_work"
    boundaries["review"].assert_not_called()
    boundaries["budget"].assert_not_called()


@pytest.mark.asyncio
async def test_unverified_source_is_skipped_without_a_model_call(
    fake_store, adapter, boundaries
):
    source = await _source(fake_store, verified=False)
    result = await _run()
    assert result.dispositions == {"skipped": 1}
    boundaries["review"].assert_not_called()
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == canonical_revision("4")
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, "4", nightly.POLICY_VERSION
    )
    assert ledger is not None and "no concrete outcome" in ledger.reason


@pytest.mark.asyncio
async def test_verified_source_becomes_a_ready_version(fake_store, adapter, boundaries):
    source = await _source(fake_store)
    result = await _run()
    assert result.applied == 1 and result.model_calls == 1
    head = await fake_store.get_head(USER, EXPERT, "csv-import-checks")
    assert head is not None and head.current_version == 1
    version = await fake_store.get_version(USER, head.current_version_id)
    assert version.state == "ready" and version.origin == "saved_overnight"
    assert version.sources[0]["source_id"] == source.id
    assert any(e["label"] == "Worked once in the source" for e in version.evidence)
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, "4", nightly.POLICY_VERSION
    )
    assert ledger.disposition == "applied" and ledger.applied_version_id == version.id
    assert ledger.cost_microdollars == 1000
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == canonical_revision("4")
    boundaries["write"].assert_awaited_once()
    boundaries["memory"].assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_error_is_retryable_and_keeps_the_cursor(
    fake_store, adapter, boundaries
):
    source = await _source(fake_store)
    boundaries["review"].side_effect = DreamLLMError("LLM call failed: timeout")
    result = await _run()
    assert result.dispositions == {"provider_error": 1}
    assert (await fake_store.get_source(USER, source.id)).processed_revision is None
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, "4", nightly.POLICY_VERSION
    )
    assert ledger.disposition == "provider_error" and ledger.completed_at is None
    boundaries["review"].side_effect = None
    again = await _run()
    assert again.applied == 1
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, "4", nightly.POLICY_VERSION
    )
    assert ledger.disposition == "applied" and ledger.attempts == 2


@pytest.mark.asyncio
async def test_completed_ledger_row_prevents_a_second_model_call(
    fake_store, adapter, boundaries
):
    """Crash window: write completed and ledger says applied, but the cursor
    never moved. The next run must settle the cursor without paying again."""
    source = await _source(fake_store)
    await fake_store.upsert_review(
        USER,
        source_id=source.id,
        source_revision="4",
        policy_version=nightly.POLICY_VERSION,
        run_id="earlier",
        disposition="applied",
        applied_version_id="v-earlier",
    )
    result = await _run()
    assert result.dispositions == {"applied": 1}
    boundaries["review"].assert_not_called()
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == canonical_revision("4")
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, "4", nightly.POLICY_VERSION
    )
    assert ledger.run_id == "earlier" and ledger.applied_version_id == "v-earlier"


@pytest.mark.asyncio
async def test_budget_exhaustion_stops_further_paid_reviews(
    fake_store, adapter, boundaries
):
    first = await _source(fake_store, ref="s1")
    second = await _source(fake_store, ref="s2")
    third = await _source(fake_store, ref="s3")
    boundaries["budget"].side_effect = [
        (True, None),
        (True, None),
        (False, "insufficient_credits"),
    ]
    result = await _run()
    assert result.applied == 1 and result.model_calls == 1
    assert result.dispositions == {"applied": 1, "budget_exhausted": 2}
    assert (await fake_store.get_source(USER, first.id)).processed_revision is not None
    for source in (second, third):
        assert (await fake_store.get_source(USER, source.id)).processed_revision is None
    ledger = await fake_store.get_review_for_revision(
        USER, second.id, "4", nightly.POLICY_VERSION
    )
    assert ledger.disposition == "budget_exhausted"


@pytest.mark.asyncio
async def test_selection_is_fair_across_owners(fake_store, adapter, boundaries):
    for i in range(nightly.MAX_REVIEWS_PER_OWNER * 3):
        await _source(fake_store, ref=f"a-{i}", expert_id="expert-a")
    await _source(fake_store, ref="b-fresh", expert_id="expert-b")
    selected = await nightly._select_sources(USER, None)
    owners = {s.owner_key for s in selected}
    assert owners == {"expert-a", "expert-b"}
    assert (
        sum(1 for s in selected if s.owner_key == "expert-a")
        == nightly.MAX_REVIEWS_PER_OWNER
    )


@pytest.mark.asyncio
async def test_clipped_verification_span_defers_the_review(
    fake_store, adapter, boundaries
):
    source = await _source(fake_store)
    adapter.clipped = ["msg:3"]
    result = await _run()
    assert result.dispositions == {"deferred": 1}
    boundaries["review"].assert_not_called()
    assert (await fake_store.get_source(USER, source.id)).processed_revision is None


@pytest.mark.asyncio
async def test_exclusion_during_review_refuses_the_stale_proposal(
    fake_store, adapter, boundaries
):
    source = await _source(fake_store)

    async def exclude_while_loading(_source: SourceRevision) -> None:
        await fake_store.set_source_eligibility(
            USER, source.id, "excluded", excluded_by_user_id=USER
        )

    adapter.on_load.append(exclude_while_loading)
    result = await _run()
    assert result.dispositions == {"stale_eligibility": 1}
    boundaries["write"].assert_not_called()
    assert (
        await fake_store.get_head(USER, EXPERT, "csv-import-checks") is None
        or (await fake_store.list_versions(USER, EXPERT, "csv-import-checks")) == []
    )


@pytest.mark.asyncio
async def test_human_controlled_skill_gets_a_single_open_proposal(
    fake_store, adapter, boundaries
):
    head = await fake_store.ensure_head(USER, EXPERT, "csv-import-checks")
    await fake_store.update_head_policy(
        USER, EXPERT, "csv-import-checks", auto_improve=False
    )
    await _source(fake_store, ref="s1")
    await _source(fake_store, ref="s2")
    result = await _run()
    assert result.proposed == 2 and result.applied == 0
    open_versions = await fake_store.list_open_decisions(USER, EXPERT)
    assert len(open_versions) == 1
    archived = [
        v
        for v in await fake_store.list_versions(USER, EXPERT, "csv-import-checks")
        if v.state == "archived"
    ]
    assert len(archived) == 1 and "superseded" in archived[0].state_reason
    assert (
        await fake_store.get_head(USER, EXPERT, "csv-import-checks")
    ).current_version == head.current_version
    boundaries["write"].assert_not_called()


@pytest.mark.asyncio
async def test_removed_behavior_is_not_silently_reapplied(
    fake_store, adapter, boundaries
):
    from .fingerprint import behavior_fingerprint, behavior_tokens

    await fake_store.ensure_head(USER, EXPERT, "csv-import-checks")
    await fake_store.add_suppression(
        USER,
        expert_id=EXPERT,
        skill_name="csv-import-checks",
        behavior_fingerprint=behavior_fingerprint("csv-import-checks", GOOD_BODY),
        behavior_tokens=behavior_tokens(GOOD_BODY),
        evidence_fingerprints=[],
        actor_user_id=USER,
        reason="undone",
    )
    await _source(fake_store)
    result = await _run()
    assert result.dispositions == {"suppressed": 1}
    boundaries["write"].assert_not_called()


@pytest.mark.asyncio
async def test_near_identical_behavior_after_undo_needs_a_decision(
    fake_store, adapter, boundaries
):
    from .fingerprint import behavior_fingerprint, behavior_tokens

    paraphrased = GOOD_BODY.replace(
        "Open the file as utf-8", "Open the file using utf-8 first"
    )
    await fake_store.ensure_head(USER, EXPERT, "csv-import-checks")
    await fake_store.add_suppression(
        USER,
        expert_id=EXPERT,
        skill_name="csv-import-checks",
        behavior_fingerprint=behavior_fingerprint("csv-import-checks", paraphrased),
        behavior_tokens=behavior_tokens(paraphrased),
        evidence_fingerprints=[],
        actor_user_id=USER,
        reason="undone",
    )
    await _source(fake_store)
    result = await _run()
    assert result.dispositions == {"needs_decision": 1}
    open_versions = await fake_store.list_open_decisions(USER, EXPERT)
    assert len(open_versions) == 1 and "resembles" in open_versions[0].state_reason


@pytest.mark.asyncio
async def test_seeded_credential_never_reaches_the_skill_or_history(
    fake_store, adapter, boundaries
):
    secret = "ghp_" + "s" * 40
    boundaries["review"].return_value = _completion(
        _proposal(
            body=GOOD_BODY.replace(
                "Open the file as utf-8", f"Open the file with token {secret}"
            )
        )
    )
    await _source(fake_store)
    result = await _run()
    assert result.dispositions == {"blocked_content": 1}
    boundaries["write"].assert_not_called()
    versions = await fake_store.list_versions(USER, EXPERT, "csv-import-checks")
    assert len(versions) == 1 and versions[0].state == "blocked_content"
    assert versions[0].content == "" and secret not in versions[0].model_dump_json()
    assert versions[0].blocked_pattern_class == "github_token"
    assert (
        versions[0].blocked_step is not None and secret not in versions[0].blocked_step
    )
    for review in fake_store.reviews.values():
        assert secret not in review.model_dump_json()


@pytest.mark.asyncio
async def test_secret_in_model_metadata_creates_no_head_or_record(
    fake_store, adapter, boundaries
):
    """A credential in the model-written name or description is blocked
    before any head, version, or ledger row exists under that value."""
    secret = "sk-" + "m" * 30
    for overrides in ({"skill_name": secret}, {"description": f"use {secret}"}):
        boundaries["review"].return_value = _completion(_proposal(**overrides))
        await _source(fake_store, ref=f"s-{len(overrides)}-{list(overrides)[0]}")
        result = await _run()
        assert result.dispositions == {"blocked_content": 1}
        assert fake_store.heads == {} and fake_store.versions == {}
        assert all(
            secret not in r.model_dump_json() for r in fake_store.reviews.values()
        )
        assert all(secret not in r.reason for r in fake_store.reviews.values())


@pytest.mark.asyncio
async def test_unsupported_or_questionable_proposals_are_rejected_deterministically(
    fake_store, adapter, boundaries
):
    from .contract import OutcomeSignal

    bundle = EvidenceBundle(
        source=SourceRevision(
            source_id="x",
            source_kind=KIND,
            source_ref="x",
            scope=LearningScope(user_id=USER, expert_id=EXPERT, owner_key=EXPERT),
            revision="4",
            epoch=0,
            outcome_signals=[
                OutcomeSignal(kind="tool_result", ref="msg:3", label="ok")
            ],
        ),
        spans=adapter.spans,
    )
    assert nightly.validate_proposal(_proposal(supported_by=["msg:1"]), bundle, []) == (
        "proposal cites no span with a checked outcome"
    )
    assert nightly.validate_proposal(
        _proposal(supported_by=["msg:99"]), bundle, []
    ) == ("proposal cites no evidence spans that exist")
    assert nightly.validate_proposal(_proposal(verification=""), bundle, []) == (
        "proposal names no concrete verification"
    )
    assert nightly.validate_proposal(_proposal(body="## Steps\n1. x"), bundle, []) == (
        "proposal lacks required sections: verification"
    )
    existing = [ParsedSkill(name="csv-import-checks", description="d", body="b")]
    assert nightly.validate_proposal(
        _proposal(decision="create"), bundle, existing
    ) == ("create would duplicate an existing skill")
    assert nightly.validate_proposal(_proposal(decision="update"), bundle, []) == (
        "update names a skill that does not exist"
    )
    assert (
        nightly.validate_proposal(
            _proposal(decision="skip", reason="only a plan"), bundle, []
        )
        == "only a plan"
    )


@pytest.mark.asyncio
async def test_reconcile_finishes_an_interrupted_publication(
    fake_store, adapter, boundaries
):
    from backend.data.skill_publication import VersionDraft

    head = await fake_store.ensure_head(USER, EXPERT, "csv-import-checks")
    source = await _source(fake_store)
    from backend.data.skill_publication import ReviewStamp

    stamp = ReviewStamp(
        source=source, source_revision="4", policy_version=1, run_id="crashed"
    )
    commit = await fake_store.commit_version_safe(
        USER,
        head=head,
        draft=VersionDraft(
            content=publish.render_skill_markdown(
                ParsedSkill(name="csv-import-checks", description="d", body=GOOD_BODY)
            ),
            description="d",
            origin="saved_overnight",
        ),
        expected_current_version=0,
        review=stamp,
    )
    assert commit.committed and commit.version.state == "pending_write"
    completed = await publish.reconcile_pending(USER)
    assert completed == 1
    boundaries["write"].assert_awaited_once()
    assert (await fake_store.get_version(USER, commit.version.id)).state == "ready"
    ledger = await fake_store.get_review_for_revision(USER, source.id, "4", 1)
    assert ledger.disposition == "applied"
    # The next pass sees the completed ledger row and settles the cursor
    # without another model call.
    result = await _run()
    boundaries["review"].assert_not_called()
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == canonical_revision("4")
    assert result.dispositions == {"applied": 1}


@pytest.mark.asyncio
async def test_flag_off_and_held_lease_do_nothing(
    fake_store, adapter, boundaries, monkeypatch
):
    await _source(fake_store)
    monkeypatch.setattr(nightly, "is_feature_enabled", AsyncMock(return_value=False))
    assert (await _run()).skip_reason == "skill_learning_disabled"
    monkeypatch.setattr(nightly, "is_feature_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(nightly, "_acquire_lease", AsyncMock(return_value=None))
    assert (await _run()).skip_reason == "lease_held"
    boundaries["review"].assert_not_called()


@pytest.mark.asyncio
async def test_requested_learning_labels_origin(fake_store, adapter, boundaries):
    source = await fake_store.upsert_source_revision(
        USER,
        expert_id=EXPERT,
        source_kind=KIND,
        source_id="req-1",
        revision="4",
        evidence_refs=[],
        outcome_signals=[{"kind": "tool_result", "ref": "msg:3", "label": "ok"}],
        origin="requested",
    )
    result = await _run(source_ids=[source.id], trigger="requested")
    assert result.applied == 1
    head = await fake_store.get_head(USER, EXPERT, "csv-import-checks")
    version = await fake_store.get_version(USER, head.current_version_id)
    assert version.origin == "requested"
    assert any(e["label"] == "Requested by user" for e in version.evidence)


@pytest.mark.asyncio
async def test_pass_never_raises(fake_store, adapter, boundaries, monkeypatch):
    await _source(fake_store)
    monkeypatch.setattr(
        nightly, "reconcile_pending", AsyncMock(side_effect=RuntimeError("db down"))
    )
    with patch.object(nightly.logger, "exception"):
        result = await _run()
    assert result.error is not None and "db down" in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["proposal_limit", "missing_evidence"])
async def test_completed_review_survives_later_deferral_gates(
    fake_store, adapter, boundaries, monkeypatch, gate
):
    source = await _source(fake_store)
    await fake_store.upsert_review(
        USER,
        source_id=source.id,
        source_revision="4",
        policy_version=nightly.POLICY_VERSION,
        run_id="earlier",
        disposition="applied",
        applied_version_id="v-earlier",
    )
    if gate == "proposal_limit":
        monkeypatch.setattr(nightly, "_open_proposals", AsyncMock(return_value=10))
    else:
        adapter.spans = []
    result = await _run()
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, "4", nightly.POLICY_VERSION
    )
    assert ledger.disposition == "applied"
    assert ledger.run_id == "earlier"
    assert ledger.applied_version_id == "v-earlier"
    assert result.dispositions == {"applied": 1}
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == canonical_revision("4")
    boundaries["review"].assert_not_called()
    boundaries["write"].assert_not_called()
