"""What a hard forget blanks outside the graph: the dream's copies, in
Postgres, of the text it derived ("accept names, blank the copies").

A hard forget erases, in the graph, the text the dream derived from what it
forgot (``graphiti/recall_erase.py``). The dream kept that text in Postgres
too: on its pass's ``DreamPass`` row (each phase's output, the operations
apply was handed and what apply reported, an error quoting the model) and in
the narrative its dream session opens with.

``blank_dream_copies`` marks with ``erasedAt``, never cleared, every pass of
the scope the forget may have reached, in one statement over all of the
scope's passes (``data/dream_pass.erase_dream_passes``): one whose dream
writes the forget erased (a dream write's episode is named after its pass,
``apply._episode_name``), one whose row names a fact or episode the forget
reached, one still open (it may have read the fact without naming it yet),
and one closed with its cleanup pending (a stop from outside closes the row
while the pass may still be applying). A marked pass that is closed has its
text blanked in that statement; an open one in the statement that closes it,
however it closes (``data/dream_pass_update.py``), the reaper's expiry being
the backstop for one that never ends. Then the narrative of every pass
marked or named is blanked, and one written after the mark is written blank
(``copilot/db.add_dream_narrative``). Every failed pass of the scope keeps
only the code of its error: one written before errors were codes may quote
the model.

Every sentence of a marked pass goes, each ``content``, ``rationale``,
``reason`` and ``summary_for_user`` reading ``recall.FORGOTTEN_FACT``, not
only those derived from the forgotten fact: the dream may restate it
anywhere, in its own words. The uuids, scopes, counts and usage stay. A
pass whose row has aged out (``retention.py``) keeps only its narrative,
which is blanked through its pass id. Entity names and name embeddings are
kept.
"""

import asyncio
import re
from collections.abc import Collection

from backend.copilot.graphiti.recall import FORGOTTEN_FACT
from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import chat_db

from .store import erase_passes

# How long blanking the narratives may take, the RPC included.
NARRATIVE_TIMEOUT_SECONDS = 10.0
# A dream write's episode name (``apply._episode_name``).
_DREAM_WRITE = re.compile(r"^dream_(?P<pass_id>.+)_(?:consolidate|recombine)_\d{3}$")


async def blank_dream_copies(
    scope: MemoryScope, *, reached: Collection[str], dream_writes: Collection[str]
) -> list[str]:
    """Mark the passes of ``scope`` a hard forget may have reached (the uuids
    it ``reached``, the episode names of the ``dream_writes`` it erased),
    blanking what they keep now or as they close, and blank their
    narratives. Returns the ids marked. Raises when a write fails or runs
    out of time; repeating it is harmless."""
    named = sorted({pass_id for name in dream_writes if (pass_id := pass_of(name))})
    marked = await erase_passes(scope, reached=sorted(reached), named=named)
    narratives = sorted({*marked, *named})
    if narratives:
        async with asyncio.timeout(NARRATIVE_TIMEOUT_SECONDS):
            await chat_db().blank_dream_narratives(
                scope.owner_user_id, narratives, FORGOTTEN_FACT
            )
    return marked


def pass_of(episode_name: str) -> str | None:
    """The pass a dream write's episode is named after; None for any other
    episode."""
    match = _DREAM_WRITE.match(episode_name)
    return match.group("pass_id") if match else None
