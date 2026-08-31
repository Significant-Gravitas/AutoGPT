"""Pure composers for the overseer's daily briefing cards.

Called at briefing-generation time (`briefing/generate.py`) — these
describe a day of task state, not a 15-minute window, so they live with
the composer rather than the 15-minute pass. Pure functions over already-
fetched tasks: no I/O, trivially testable with frozen time.
"""

from datetime import datetime, timedelta
from difflib import SequenceMatcher
from itertools import combinations

from backend.api.features.tasks.models import DelegatedTask
from backend.copilot.briefing.models import BriefingMergeItem, BriefingNudgeItem

NUDGE_AFTER = timedelta(hours=24)

# Normalized-title similarity at or above this reads as "probably the same
# ask" — high enough that generic titles ("Run Agent") need a near-exact
# match before we suggest merging.
MERGE_SIMILARITY_THRESHOLD = 0.75

_MAX_NUDGE_ITEMS = 5
_MAX_MERGE_ITEMS = 3


def compose_nudge_items(
    tasks: list[DelegatedTask], now: datetime
) -> list[BriefingNudgeItem]:
    """WAITING_USER tasks the user has sat on for over a day, oldest first —
    the briefing nudges rather than Home shouting immediately."""
    waiting = sorted(
        (
            task
            for task in tasks
            if task.status == "WAITING_USER" and task.updated_at < now - NUDGE_AFTER
        ),
        key=lambda task: task.updated_at,
    )
    return [
        BriefingNudgeItem(
            task_id=task.id,
            title=task.title,
            waiting_since=task.updated_at,
            question=_latest_question(task),
            is_stale=task.stale_at is not None,
        )
        for task in waiting[:_MAX_NUDGE_ITEMS]
    ]


def compose_merge_items(tasks: list[DelegatedTask]) -> list[BriefingMergeItem]:
    """Pairs of open root tasks whose titles look like the same ask — a
    "merge?" suggestion only; nothing is ever merged automatically."""
    roots = [task for task in tasks if task.parent_task_id is None]
    suggested: set[str] = set()
    items: list[BriefingMergeItem] = []
    for a, b in combinations(roots, 2):
        if len(items) >= _MAX_MERGE_ITEMS:
            break
        if a.id in suggested or b.id in suggested:
            continue
        if _title_similarity(a.title, b.title) >= MERGE_SIMILARITY_THRESHOLD:
            suggested.update((a.id, b.id))
            items.append(
                BriefingMergeItem(task_ids=[a.id, b.id], titles=[a.title, b.title])
            )
    return items


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _normalize(title: str) -> str:
    return " ".join(title.lower().split())


def _latest_question(task: DelegatedTask) -> str | None:
    for amendment in reversed(task.amendments):
        if amendment.kind == "escalation":
            return amendment.question or amendment.note
    return None
