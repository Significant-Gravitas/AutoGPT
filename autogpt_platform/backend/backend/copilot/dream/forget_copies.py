"""What a hard forget blanks outside the graph: the dream's copies, in
Postgres, of the text it derived ("accept names, blank the copies").

A hard forget erases, in the graph, the text the dream derived from what it
forgot (``graphiti/recall_erase.py``). The dream kept that text in Postgres
too: on its pass's ``DreamPass`` row (each phase's output, the operations
apply was handed and what apply reported) and in the narrative its dream
session opens with. ``blank_dream_copies`` blanks them for every closed pass
of the scope whose writes the forget erased (a dream write's episode is
named after its pass, ``apply._episode_name``) or whose row names a fact or
episode the forget reached (what its writes and proposals cite).

Every sentence of such a pass goes, each ``content``, ``rationale``,
``reason`` and ``summary_for_user`` reading ``recall.FORGOTTEN_FACT``, not
only those derived from the forgotten fact: the dream may restate it
anywhere, in its own words. The uuids, scopes, counts and usage stay. Every
pass of the scope still running is left as it is and reported: its next
phase reads its row, and one that gathered its input before the forget may
not name what it read yet. Forgetting again once it has finished blanks it.
A pass whose row has aged out (``retention.py``) keeps only its narrative,
which is blanked through its pass id. Entity names and name embeddings are
kept.
"""

import asyncio
import re
from collections.abc import Collection
from typing import Any, TypeVar

from pydantic import BaseModel

from backend.copilot.graphiti.recall import FORGOTTEN_FACT
from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import chat_db
from backend.data.dream_pass_models import OPEN_STATUSES, DreamPassRecord

from .store import blank_pass_text, read_scope_passes

# How many of a scope's passes a hard forget looks through, newest first:
# more than the retention keeps.
SCOPE_PASS_LIMIT = 1000
# How long blanking the narratives may take, the RPC included.
NARRATIVE_TIMEOUT_SECONDS = 10.0
# A dream write's episode name (``apply._episode_name``).
_DREAM_WRITE = re.compile(r"^dream_(?P<pass_id>.+)_(?:consolidate|recombine)_\d{3}$")
# The fields holding the dream's text in a pass's phase outputs and operations.
_TEXT_FIELDS = frozenset({"content", "rationale", "reason", "summary_for_user"})

_Part = TypeVar("_Part", bound=BaseModel)


async def blank_dream_copies(
    scope: MemoryScope, *, reached: Collection[str], dream_writes: Collection[str]
) -> list[str]:
    """Blank the dream's text in Postgres for the passes of ``scope`` a hard
    forget reached: those whose dream writes it erased (``dream_writes``,
    their episodes' names) or whose row names one of ``reached``. Returns
    the ids of the scope's passes still running, left as they are. Raises
    when a read or a write fails or runs out of time."""
    user_id = scope.owner_user_id
    named = {pass_id for name in dream_writes if (pass_id := pass_of(name))}
    rows = await read_scope_passes(user_id, scope.scope_key, limit=SCOPE_PASS_LIMIT)
    running = [row.id for row in rows if row.status in OPEN_STATUSES]
    affected = [
        row
        for row in rows
        if row.id not in running and (row.id in named or _names(row, reached))
    ]
    for row in affected:
        await blank_pass_text(
            row.id, user_id, _blanked(row.phase_outputs), _blanked(row.operations)
        )
    narratives = sorted((named | {row.id for row in affected}) - set(running))
    if narratives:
        async with asyncio.timeout(NARRATIVE_TIMEOUT_SECONDS):
            await chat_db().blank_dream_narratives(user_id, narratives, FORGOTTEN_FACT)
    return running


def pass_of(episode_name: str) -> str | None:
    """The pass a dream write's episode is named after; None for any other
    episode."""
    match = _DREAM_WRITE.match(episode_name)
    return match.group("pass_id") if match else None


def _names(row: DreamPassRecord, uuids: Collection[str]) -> bool:
    """Whether the pass's phase outputs or operations name one of ``uuids``:
    something one of its writes or proposals cites, or a fact it demoted."""
    text = row.phase_outputs.model_dump_json() + row.operations.model_dump_json()
    return any(uuid in text for uuid in uuids)


def _blanked(part: _Part) -> _Part:
    """``part`` with every text field holding the dream's words blanked."""
    return type(part).model_validate(_blank(part.model_dump(mode="json")))


def _blank(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                FORGOTTEN_FACT
                if key in _TEXT_FIELDS and isinstance(item, str)
                else _blank(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_blank(item) for item in value]
    return value
