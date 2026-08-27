from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.enums
import prisma.errors
import prisma.models
import pytest

from backend.api.features.experts import learned_notes_db
from backend.api.features.experts.models import ExpertLearnedNote
from backend.util.exceptions import ExpertNotFoundError


def _row(
    note_id: str = "note-1",
    *,
    text: str = "Send drafts before publishing",
    source_rule_id: str | None = "rule-1",
    status: prisma.enums.ExpertLearnedNoteStatus = prisma.enums.ExpertLearnedNoteStatus.ACTIVE,
):
    return SimpleNamespace(
        id=note_id,
        expertId="expert-1",
        text=text,
        learnedAt=datetime(2026, 8, 1, tzinfo=timezone.utc),
        sourceSessionId="session-1",
        sourceRuleId=source_rule_id,
        status=status,
    )


def _note(
    *,
    text: str = "Send drafts before publishing",
    source_rule_id: str | None = "rule-1",
):
    return ExpertLearnedNote(
        id="note-existing",
        expert_id="expert-1",
        text=text,
        learned_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        source_session_id="session-1",
        source_rule_id=source_rule_id,
        status="active",
    )


def test_learning_equivalence_uses_meaningful_tokens_and_empty_fallback():
    assert learned_notes_db.content_tokens("The user should send a draft") == {
        "send",
        "draft",
    }
    assert learned_notes_db.is_equivalent_learning(
        "Always send the launch draft", "Send a launch draft"
    )
    assert learned_notes_db.is_equivalent_learning("Always", " always ")
    assert not learned_notes_db.is_equivalent_learning("Always", "Never")


@pytest.mark.parametrize("include_archived", [False, True])
async def test_list_learned_notes_maps_rows_and_applies_status_filter(
    include_archived: bool,
):
    manager = SimpleNamespace(
        find_many=AsyncMock(
            return_value=[
                _row(
                    status=(
                        prisma.enums.ExpertLearnedNoteStatus.ARCHIVED
                        if include_archived
                        else prisma.enums.ExpertLearnedNoteStatus.ACTIVE
                    )
                )
            ]
        )
    )
    with patch.object(prisma.models.ExpertLearnedNote, "prisma", return_value=manager):
        notes = await learned_notes_db.list_learned_notes(
            "user-1", "expert-1", include_archived=include_archived, limit=7
        )

    assert notes[0].status == ("archived" if include_archived else "active")
    where = manager.find_many.await_args.kwargs["where"]
    if include_archived:
        assert "status" not in where
    else:
        assert where["status"] == prisma.enums.ExpertLearnedNoteStatus.ACTIVE
    assert manager.find_many.await_args.kwargs["take"] == 7


async def test_archive_learned_note_returns_the_archived_note():
    manager = SimpleNamespace(
        update_many=AsyncMock(return_value=1),
        find_first=AsyncMock(
            return_value=_row(status=prisma.enums.ExpertLearnedNoteStatus.ARCHIVED)
        ),
    )
    with patch.object(prisma.models.ExpertLearnedNote, "prisma", return_value=manager):
        note = await learned_notes_db.archive_learned_note(
            "user-1", "note-1", "expert-1"
        )

    assert note.status == "archived"
    manager.update_many.assert_awaited_once()
    manager.find_first.assert_awaited_once()


@pytest.mark.parametrize(("updated", "row"), [(0, None), (1, None)])
async def test_archive_learned_note_rejects_missing_notes(updated: int, row: object):
    manager = SimpleNamespace(
        update_many=AsyncMock(return_value=updated),
        find_first=AsyncMock(return_value=row),
    )
    with (
        patch.object(prisma.models.ExpertLearnedNote, "prisma", return_value=manager),
        pytest.raises(ExpertNotFoundError),
    ):
        await learned_notes_db.archive_learned_note(
            "user-1", "note-missing", "expert-1"
        )

    if updated == 0:
        manager.find_first.assert_not_awaited()


async def test_archive_notes_for_rules_skips_empty_input_and_archives_matches():
    manager = SimpleNamespace(update_many=AsyncMock(return_value=2))
    with patch.object(prisma.models.ExpertLearnedNote, "prisma", return_value=manager):
        assert (
            await learned_notes_db.archive_notes_for_rules("user-1", "expert-1", [])
            == 0
        )
        archived = await learned_notes_db.archive_notes_for_rules(
            "user-1", "expert-1", ["rule-1", "rule-2"]
        )

    assert archived == 2
    manager.update_many.assert_awaited_once()


async def test_promote_learned_notes_requires_candidates_and_owned_expert():
    assert await learned_notes_db.promote_learned_notes("user-1", "expert-1", []) == []

    expert_manager = SimpleNamespace(count=AsyncMock(return_value=0))
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=expert_manager),
        pytest.raises(ExpertNotFoundError),
    ):
        await learned_notes_db.promote_learned_notes(
            "user-1",
            "expert-1",
            [learned_notes_db.LearnedNoteCandidate(text="Keep launch drafts concise")],
        )


async def test_promote_learned_notes_deduplicates_and_tolerates_insert_races():
    expert_manager = SimpleNamespace(count=AsyncMock(return_value=1))
    note_manager = SimpleNamespace(
        create=AsyncMock(
            side_effect=[
                _row(
                    note_id="note-new",
                    text="Store the final launch plan",
                    source_rule_id="rule-new",
                ),
                prisma.errors.UniqueViolationError({}),
            ]
        )
    )
    candidates = [
        learned_notes_db.LearnedNoteCandidate(text="   "),
        learned_notes_db.LearnedNoteCandidate(
            text="Use a different format", source_rule_id="rule-1"
        ),
        learned_notes_db.LearnedNoteCandidate(
            text="Always send drafts before publishing"
        ),
        learned_notes_db.LearnedNoteCandidate(
            text="Store the final launch plan", source_rule_id="rule-new"
        ),
        learned_notes_db.LearnedNoteCandidate(
            text="Keep the campaign brief concise", source_rule_id="rule-race"
        ),
    ]
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=expert_manager),
        patch.object(
            prisma.models.ExpertLearnedNote, "prisma", return_value=note_manager
        ),
        patch.object(
            learned_notes_db,
            "list_learned_notes",
            new_callable=AsyncMock,
            return_value=[_note()],
        ),
        patch.object(
            learned_notes_db, "_trim_to_cap", new_callable=AsyncMock
        ) as trim_to_cap,
    ):
        created = await learned_notes_db.promote_learned_notes(
            "user-1", "expert-1", candidates
        )

    assert [note.id for note in created] == ["note-new"]
    assert note_manager.create.await_count == 2
    trim_to_cap.assert_awaited_once_with("user-1", "expert-1")


@pytest.mark.parametrize("stale_ids", [[], ["note-21", "note-22"]])
async def test_trim_to_cap_archives_only_stale_notes(stale_ids: list[str]):
    manager = SimpleNamespace(
        find_many=AsyncMock(
            return_value=[_row(note_id=note_id) for note_id in stale_ids]
        ),
        update_many=AsyncMock(),
    )
    with patch.object(prisma.models.ExpertLearnedNote, "prisma", return_value=manager):
        await learned_notes_db._trim_to_cap("user-1", "expert-1")

    assert (
        manager.find_many.await_args.kwargs["skip"]
        == learned_notes_db.MAX_ACTIVE_LEARNED_NOTES
    )
    if stale_ids:
        manager.update_many.assert_awaited_once_with(
            where={"id": {"in": stale_ids}, "userId": "user-1"},
            data={"status": prisma.enums.ExpertLearnedNoteStatus.ARCHIVED},
        )
    else:
        manager.update_many.assert_not_awaited()
