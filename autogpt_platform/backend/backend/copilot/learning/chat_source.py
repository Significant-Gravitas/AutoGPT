"""Ordinary-chat source adapter.

Captures a bounded source revision at the end of each user turn and answers
the learner's live eligibility questions. Ordinary conversations need no
approval workflow: eligibility is "the session still exists, still belongs
to the owner, is not excluded, and its scope still accepts learning".

Evidence is read as an immutable snapshot bounded by the revision under
review: messages persisted after that revision are never included, and
every captured reference that could not be included in full is reported
as omitted.
"""

from __future__ import annotations

import logging

from backend.copilot.model import ChatMessage, ChatSession
from backend.data.db_accessors import chat_db, experts_db, skill_learning_db
from backend.data.skill_learning import (
    LearningSourceRecord,
    canonical_revision,
    owner_key_for,
)

from .contract import (
    ApprovalCheckpoint,
    Eligibility,
    EligibilityState,
    EvidenceBundle,
    EvidenceRef,
    EvidenceSpan,
    LearningScope,
    OutcomeSignal,
    SourceRevision,
    register_source_adapter,
)
from .signals import extract_turn_signals, is_learn_request, message_ref

logger = logging.getLogger(__name__)

CHAT_SOURCE_KIND = "chat_session"
MAX_EVIDENCE_MESSAGES = 120
MAX_SPAN_CHARS = 2_000


class CapturedTurn(EvidenceRef):
    """Marker type kept for readability of adapter code."""


def source_revision_from_record(record: LearningSourceRecord) -> SourceRevision:
    return SourceRevision(
        source_id=record.id,
        source_kind=record.source_kind,
        source_ref=record.source_id,
        scope=LearningScope(
            user_id=record.user_id,
            expert_id=record.expert_id,
            owner_key=record.owner_key,
        ),
        revision=record.revision,
        epoch=record.epoch,
        origin="requested" if record.origin == "requested" else "ordinary",
        evidence_refs=[EvidenceRef.model_validate(r) for r in record.evidence_refs],
        outcome_signals=[
            OutcomeSignal.model_validate(s) for s in record.outcome_signals
        ],
        approval=(
            ApprovalCheckpoint(
                event_id=record.approval_event_id,
                actor_id=record.approval_actor_id,
                approved_revision=record.approved_revision or "",
            )
            if record.approval_event_id
            else None
        ),
    )


def revision_sequence(revision: str) -> int | None:
    """Message sequence a canonical chat revision stands for."""
    text = revision.strip().lstrip("0") or "0"
    return int(text) if text.isdigit() else None


class ChatSessionSourceAdapter:
    kind = CHAT_SOURCE_KIND
    requires_approval = False

    async def revalidate(
        self,
        *,
        source_id: str,
        revision: str,
        scope: LearningScope,
        approval_event_id: str | None,
    ) -> Eligibility:
        record = await skill_learning_db().get_source(scope.user_id, source_id)
        if record is None or record.owner_key != scope.owner_key:
            return Eligibility(
                state=EligibilityState.INACCESSIBLE,
                revision=canonical_revision(revision),
                epoch=-1,
                reason="source not visible in this scope",
            )
        if record.eligibility == "excluded":
            return _eligibility(record, EligibilityState.EXCLUDED, "excluded by owner")
        if record.eligibility == "inaccessible":
            return _eligibility(
                record, EligibilityState.INACCESSIBLE, "source no longer accessible"
            )
        if record.revision != canonical_revision(revision):
            return _eligibility(record, EligibilityState.STALE, "newer revision exists")
        session = await chat_db().get_chat_session_metadata(record.source_id)
        if session is None or session.user_id != scope.user_id:
            return _eligibility(
                record,
                EligibilityState.INACCESSIBLE,
                "conversation no longer accessible",
            )
        if session.expert_id != scope.expert_id:
            return _eligibility(
                record, EligibilityState.INACCESSIBLE, "conversation moved scope"
            )
        if scope.expert_id is not None:
            expert = await experts_db().get_expert(
                scope.user_id, scope.expert_id, include_workflows=False
            )
            if expert is None:
                return _eligibility(
                    record, EligibilityState.INACCESSIBLE, "expert no longer owned"
                )
            if expert.learning_paused_at is not None:
                return _eligibility(
                    record, EligibilityState.PAUSED, "learning paused for this expert"
                )
        return _eligibility(record, EligibilityState.ELIGIBLE, "")

    async def load_evidence(
        self, source: SourceRevision, *, max_chars: int
    ) -> EvidenceBundle:
        last_sequence = revision_sequence(source.revision)
        if last_sequence is None:
            return EvidenceBundle(source=source)
        page = await chat_db().get_chat_messages_paginated(
            session_id=source.source_ref,
            limit=MAX_EVIDENCE_MESSAGES,
            user_id=source.scope.user_id,
            after_sequence=0,
            before_sequence=last_sequence + 1,
        )
        if page is None or page.session.user_id != source.scope.user_id:
            return EvidenceBundle(source=source)
        outcome_by_ref = {s.ref: s.kind for s in source.outcome_signals}
        spans: list[EvidenceSpan] = []
        clipped: list[str] = []
        used = 0
        for message in page.messages:
            if message.sequence is None or message.sequence > last_sequence:
                continue
            text = (message.content or "").strip()
            if not text or message.role not in ("user", "assistant", "tool"):
                continue
            ref = message_ref(message)
            body = text[:MAX_SPAN_CHARS]
            if len(body) < len(text):
                clipped.append(ref)
            if used + len(body) > max_chars:
                clipped.append(ref)
                continue
            used += len(body)
            spans.append(
                EvidenceSpan(
                    ref=ref,
                    role=message.role,
                    text=body,
                    outcome=outcome_by_ref.get(ref),
                )
            )
        present = {span.ref for span in spans}
        omitted = sorted(
            {r.ref for r in source.evidence_refs if r.ref not in present} | set(clipped)
        )
        return EvidenceBundle(
            source=source, spans=spans, omitted_refs=omitted, clipped_refs=clipped
        )


def _eligibility(
    record: LearningSourceRecord, state: EligibilityState, reason: str
) -> Eligibility:
    return Eligibility(
        state=state, revision=record.revision, epoch=record.epoch, reason=reason
    )


register_source_adapter(ChatSessionSourceAdapter())


async def record_chat_turn(
    user_id: str,
    session: ChatSession,
    turn_messages: list[ChatMessage],
    user_message: str,
) -> LearningSourceRecord:
    """Persist this turn as a source revision (one bounded upsert)."""
    refs, signals = extract_turn_signals(turn_messages)
    sequences = [m.sequence for m in turn_messages if m.sequence is not None]
    revision = (
        str(max(sequences)) if sequences else str(max(len(session.messages) - 1, 0))
    )
    origin = "requested" if is_learn_request(user_message) else "ordinary"
    return await skill_learning_db().upsert_source_revision(
        user_id,
        expert_id=session.expert_id,
        source_kind=CHAT_SOURCE_KIND,
        source_id=session.session_id,
        revision=revision,
        evidence_refs=[r.model_dump() for r in refs],
        outcome_signals=[s.model_dump() for s in signals],
        origin=origin,
    )


def scope_for(user_id: str, expert_id: str | None) -> LearningScope:
    return LearningScope(
        user_id=user_id, expert_id=expert_id, owner_key=owner_key_for(expert_id)
    )
