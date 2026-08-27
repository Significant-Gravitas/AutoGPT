from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.dream.schemas import DreamOperations, ProposedFinding
from backend.copilot.graphiti.memory_model import MemoryKind

from .learned_notes import (
    LEARNED_NOTE_MIN_CONFIDENCE,
    dedupe_candidates,
    promote_pass_learned_notes,
    select_rule_candidates,
)

_MODULE = "backend.copilot.dream.learned_notes"


def _rule(
    text: str = "Always send drafts before publishing.",
    *,
    confidence: float = 0.95,
    citations: int = 1,
) -> ProposedFinding:
    return ProposedFinding(
        content=text,
        memory_kind=MemoryKind.rule,
        confidence=confidence,
        rationale="Explicit user correction",
        source_episode_uuids=[f"episode-{index}" for index in range(citations)],
    )


def test_selects_only_supported_high_confidence_rules():
    operations = DreamOperations(
        proposals=[
            _rule(),
            _rule("Unsupported", citations=0),
            _rule("Uncertain", confidence=LEARNED_NOTE_MIN_CONFIDENCE - 0.01),
            ProposedFinding(
                content="A one-off finding",
                memory_kind=MemoryKind.finding,
                confidence=0.99,
                rationale="Not a rule",
                source_episode_uuids=["episode-x"],
            ),
        ]
    )

    assert [item.text for item in select_rule_candidates(operations, None)] == [
        "Always send drafts before publishing."
    ]


def test_equivalent_learning_is_not_duplicated():
    candidates = select_rule_candidates(
        DreamOperations(proposals=[_rule("Always send drafts before publishing.")]),
        None,
    )

    assert (
        dedupe_candidates(candidates, ["Send drafts before publishing, always."]) == []
    )


def _notes_db(existing=None):
    db = MagicMock()
    db.archive_notes_for_rules = AsyncMock(return_value=0)
    db.list_learned_notes = AsyncMock(return_value=existing or [])
    db.promote_learned_notes = AsyncMock(return_value=[])
    return db


@pytest.mark.asyncio
async def test_expert_pass_promotes_learning(monkeypatch):
    db = _notes_db()
    monkeypatch.setattr(f"{_MODULE}.is_feature_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(f"{_MODULE}.expert_learned_notes_db", lambda: db)

    await promote_pass_learned_notes(
        "user-1",
        "expert-1",
        DreamOperations(proposals=[_rule()]),
        {"session_id": "session-1"},
    )

    _, _, candidates = db.promote_learned_notes.await_args.args
    assert candidates[0].source_session_id == "session-1"


@pytest.mark.asyncio
async def test_flag_off_and_autopilot_scope_do_not_promote(monkeypatch):
    db = _notes_db()
    flag = AsyncMock(return_value=False)
    monkeypatch.setattr(f"{_MODULE}.is_feature_enabled", flag)
    monkeypatch.setattr(f"{_MODULE}.expert_learned_notes_db", lambda: db)

    await promote_pass_learned_notes(
        "user-1", "expert-1", DreamOperations(proposals=[_rule()]), {}
    )
    await promote_pass_learned_notes(
        "user-1", None, DreamOperations(proposals=[_rule()]), {}
    )

    db.promote_learned_notes.assert_not_awaited()
    assert flag.await_count == 1


@pytest.mark.asyncio
async def test_learning_failure_never_fails_dream_pass(monkeypatch):
    db = _notes_db()
    db.promote_learned_notes = AsyncMock(side_effect=RuntimeError("database down"))
    monkeypatch.setattr(f"{_MODULE}.is_feature_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(f"{_MODULE}.expert_learned_notes_db", lambda: db)

    await promote_pass_learned_notes(
        "user-1", "expert-1", DreamOperations(proposals=[_rule()]), {}
    )

    db.promote_learned_notes.assert_awaited_once()
