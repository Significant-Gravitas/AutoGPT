"""Unit tests for dream-pass learned-note promotion and demotion.

Covers:
- Only high-confidence, corroborated ``memory_kind=rule`` proposals graduate.
- Promotions carry the durable Graphiti edge uuid as ``source_rule_id``.
- A rule the user already has as a note is not re-promoted as a paraphrase.
- Notes whose underlying edge this pass retired are archived.
- The whole hook is gated by the feature flag and never raises.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.experts.models import ExpertLearnedNote
from backend.copilot.dream.learned_notes import (
    LEARNED_NOTE_MIN_CONFIDENCE,
    MAX_PROMOTIONS_PER_PASS,
    dedupe_candidates,
    invalidated_rule_ids,
    promote_pass_learned_notes,
    select_rule_candidates,
)
from backend.copilot.dream.schemas import (
    DemotionSummary,
    DreamDemotion,
    DreamOperations,
    DreamOperationsSnapshot,
    EntityInvalidationSummary,
    ProposedFinding,
    WriteSummary,
)
from backend.copilot.graphiti.memory_model import MemoryKind

_LN = "backend.copilot.dream.learned_notes"


def _rule(
    content: str = "Always send drafts to the user before publishing.",
    confidence: float = 0.95,
    citations: int = 2,
    memory_kind: MemoryKind = MemoryKind.rule,
) -> ProposedFinding:
    return ProposedFinding(
        content=content,
        memory_kind=memory_kind,
        confidence=confidence,
        rationale="Corrected twice in the window.",
        source_episode_uuids=[f"ep-{i}" for i in range(citations)],
    )


def _note(text: str, note_id: str = "note-1") -> ExpertLearnedNote:
    return ExpertLearnedNote(
        id=note_id,
        expert_id="exp-1",
        text=text,
        learned_at="2026-08-01T00:00:00Z",  # type: ignore[arg-type]
        source_session_id=None,
        source_rule_id=None,
        status="active",
    )


class TestSelectRuleCandidates:
    def test_promotes_a_corroborated_high_confidence_rule(self):
        ops = DreamOperations(proposals=[_rule()])
        candidates = select_rule_candidates(ops, None)

        assert [c.text for c in candidates] == [
            "Always send drafts to the user before publishing."
        ]

    def test_carries_the_graphiti_edge_uuid_as_source_rule_id(self):
        proposal = _rule()
        snapshot = DreamOperationsSnapshot(
            proposals=[
                WriteSummary(
                    edge_uuid="edge-42", content=proposal.content, status="tentative"
                )
            ]
        )
        candidates = select_rule_candidates(
            DreamOperations(proposals=[proposal]), snapshot
        )

        assert candidates[0].source_rule_id == "edge-42"

    def test_ignores_non_rule_kinds(self):
        ops = DreamOperations(proposals=[_rule(memory_kind=MemoryKind.finding)])

        assert select_rule_candidates(ops, None) == []

    def test_ignores_low_confidence_rules(self):
        ops = DreamOperations(
            proposals=[_rule(confidence=LEARNED_NOTE_MIN_CONFIDENCE - 0.05)]
        )

        assert select_rule_candidates(ops, None) == []

    def test_ignores_a_rule_with_a_single_citation(self):
        """One passing remark is a finding; a standing instruction needs
        corroboration before it lands in every future prompt."""
        ops = DreamOperations(proposals=[_rule(citations=1)])

        assert select_rule_candidates(ops, None) == []

    def test_promotable_rule_survives_a_snapshot_without_an_edge_uuid(self):
        """Losing provenance is better than losing the note."""
        snapshot = DreamOperationsSnapshot(
            proposals=[WriteSummary(edge_uuid=None, content=_rule().content)]
        )
        candidates = select_rule_candidates(
            DreamOperations(proposals=[_rule()]), snapshot
        )

        assert len(candidates) == 1
        assert candidates[0].source_rule_id is None


class TestDedupeCandidates:
    def test_drops_a_paraphrase_of_an_existing_note(self):
        candidates = select_rule_candidates(
            DreamOperations(
                proposals=[_rule("Always send drafts to the user before publishing.")]
            ),
            None,
        )

        kept = dedupe_candidates(
            candidates, ["Send drafts to the user before publishing, always."]
        )

        assert kept == []

    def test_keeps_a_genuinely_new_rule(self):
        candidates = select_rule_candidates(
            DreamOperations(proposals=[_rule("Never email the client on weekends.")]),
            None,
        )

        kept = dedupe_candidates(
            candidates, ["Always send drafts to the user before publishing."]
        )

        assert [c.text for c in kept] == ["Never email the client on weekends."]

    def test_collapses_duplicates_within_one_pass(self):
        candidates = select_rule_candidates(
            DreamOperations(
                proposals=[
                    _rule("Always send drafts to the user before publishing."),
                    _rule("Always send the drafts to the user before publishing."),
                ]
            ),
            None,
        )

        assert len(dedupe_candidates(candidates, [])) == 1


class TestInvalidatedRuleIds:
    def test_reads_applied_demotions_and_swept_entity_edges(self):
        snapshot = DreamOperationsSnapshot(
            demotions=[
                DemotionSummary(
                    edge_uuid="edge-1", reason="contradicted", new_status="contradicted"
                ),
                DemotionSummary(
                    edge_uuid="edge-2",
                    reason="stale_fact",
                    new_status="superseded",
                    applied=False,
                ),
            ],
            entity_invalidations=[
                EntityInvalidationSummary(
                    entity_uuid="ent-1", reason="gone", edges_touched=["edge-3"]
                )
            ],
        )

        assert invalidated_rule_ids(DreamOperations(), snapshot) == [
            "edge-1",
            "edge-3",
        ]

    def test_falls_back_to_requested_demotions_without_a_snapshot(self):
        ops = DreamOperations(
            demotions=[DreamDemotion(edge_uuid="edge-9", reason="user_signal")]
        )

        assert invalidated_rule_ids(ops, None) == ["edge-9"]


class TestPromotePassLearnedNotes:
    @staticmethod
    def _notes_db(existing: list[ExpertLearnedNote] | None = None) -> MagicMock:
        notes_db = MagicMock()
        notes_db.archive_notes_for_rules = AsyncMock(return_value=0)
        notes_db.list_learned_notes = AsyncMock(return_value=existing or [])
        notes_db.promote_learned_notes = AsyncMock(return_value=[])
        return notes_db

    @pytest.mark.asyncio
    async def test_writes_promotions_and_archives_retired_rules(self):
        notes_db = self._notes_db()
        ops = DreamOperations(
            proposals=[_rule()],
            demotions=[DreamDemotion(edge_uuid="edge-old", reason="user_signal")],
        )
        with (
            patch(f"{_LN}.is_feature_enabled", AsyncMock(return_value=True)),
            patch(f"{_LN}.expert_learned_notes_db", MagicMock(return_value=notes_db)),
        ):
            await promote_pass_learned_notes(
                "user-1", "exp-1", ops, {"session_id": "dream-session"}
            )

        notes_db.archive_notes_for_rules.assert_awaited_once_with(
            "user-1", "exp-1", ["edge-old"]
        )
        _, _, written = notes_db.promote_learned_notes.await_args.args
        assert [c.text for c in written] == [
            "Always send drafts to the user before publishing."
        ]
        assert written[0].source_session_id == "dream-session"

    @pytest.mark.asyncio
    async def test_skips_a_rule_the_expert_already_learned(self):
        notes_db = self._notes_db(
            [_note("Always send drafts to the user before publishing.")]
        )
        with (
            patch(f"{_LN}.is_feature_enabled", AsyncMock(return_value=True)),
            patch(f"{_LN}.expert_learned_notes_db", MagicMock(return_value=notes_db)),
        ):
            await promote_pass_learned_notes(
                "user-1", "exp-1", DreamOperations(proposals=[_rule()]), {}
            )

        _, _, written = notes_db.promote_learned_notes.await_args.args
        assert written == []

    # Each rule must be genuinely unrelated, not a numbered template: two
    # sentences differing only by an index share ~0.71 of their content words
    # and are correctly collapsed by the dedupe pass, which would mask the cap.
    _UNRELATED_RULES = (
        "Always deploy through the staging pipeline first.",
        "Never email investors without a second reviewer.",
        "Prefer Codex over Claude for refactoring work.",
        "Keep weekly spend under two hundred credits.",
        "Write release notes in British English.",
        "Schedule standups at nine in the morning.",
        "Archive Zendesk tickets once a fortnight.",
        "Use Postgres advisory locks for cron jobs.",
    )

    @pytest.mark.asyncio
    async def test_caps_promotions_per_pass(self):
        notes_db = self._notes_db()
        assert len(self._UNRELATED_RULES) > MAX_PROMOTIONS_PER_PASS
        ops = DreamOperations(proposals=[_rule(text) for text in self._UNRELATED_RULES])
        with (
            patch(f"{_LN}.is_feature_enabled", AsyncMock(return_value=True)),
            patch(f"{_LN}.expert_learned_notes_db", MagicMock(return_value=notes_db)),
        ):
            await promote_pass_learned_notes("user-1", "exp-1", ops, {})

        _, _, written = notes_db.promote_learned_notes.await_args.args
        assert len(written) == MAX_PROMOTIONS_PER_PASS

    @pytest.mark.asyncio
    async def test_does_nothing_when_the_flag_is_off(self):
        notes_db = self._notes_db()
        with (
            patch(f"{_LN}.is_feature_enabled", AsyncMock(return_value=False)),
            patch(f"{_LN}.expert_learned_notes_db", MagicMock(return_value=notes_db)),
        ):
            await promote_pass_learned_notes(
                "user-1", "exp-1", DreamOperations(proposals=[_rule()]), {}
            )

        notes_db.promote_learned_notes.assert_not_awaited()
        notes_db.archive_notes_for_rules.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_notes_failure_never_fails_the_pass(self):
        notes_db = self._notes_db()
        notes_db.promote_learned_notes = AsyncMock(side_effect=RuntimeError("db down"))
        with (
            patch(f"{_LN}.is_feature_enabled", AsyncMock(return_value=True)),
            patch(f"{_LN}.expert_learned_notes_db", MagicMock(return_value=notes_db)),
        ):
            await promote_pass_learned_notes(
                "user-1", "exp-1", DreamOperations(proposals=[_rule()]), {}
            )
