"""The chat sessions a dream must not read, because a forget hid their memory.

A dream reads recent chat history straight from the chat store, which a
forget never touches: the forgotten fact still sits in the message that said
it. So every session that a hidden episode came from (one the recall policy
will not return, see ``recall.recallable_episode_predicate``) is left out of
the dream's input whole. Coarse on purpose: a forgotten sentence cannot be
cut out of a transcript reliably, and a session is only background there.

An episode names its session in up to three places, all written at
ingestion: a chat turn is named ``conversation_<session>`` and a derived
finding ``finding_<session>``, both with a ``source_description`` ending
``in session <session>``; a stored memory's envelope has
``provenance='session:<session>#msg:<n>'``. A hard forget empties an episode
into a tombstone but keeps all three (the provenance in its own property,
since the envelope went with the text), so its session stays excluded.
"""

import logging
import re

from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.copilot.graphiti.memory_model import envelope_provenance
from backend.copilot.graphiti.recall import (
    forgotten_facts_clause,
    recallable_episode_predicate,
)

logger = logging.getLogger(__name__)

_EPISODE_NAME = re.compile(r"^(?:conversation|finding)_(?P<session>.+)$")
_SOURCE_DESCRIPTION = re.compile(r" in session (?P<session>\S+)$")
_PROVENANCE = re.compile(r"^session:(?P<session>[^#]+)")


async def hidden_session_ids(
    driver: AutoGPTFalkorDriver, group_id: str
) -> set[str] | None:
    """Every session a hidden episode in the graph came from.

    ``None`` when the graph could not be read: the caller cannot tell which
    sessions are safe and must read none.
    """
    try:
        result = await driver.execute_query(_HIDDEN_EPISODES_QUERY, g=group_id)
    except Exception:
        logger.warning(
            f"Hidden-episode read failed for group {group_id[:12]}; "
            "the dream reads no chat sessions",
            exc_info=True,
        )
        return None
    rows = result[0] if result else []
    return {
        session
        for row in rows
        for session in episode_session_ids(
            row["name"], row["source_description"], row["content"], row["provenance"]
        )
    }


def episode_session_ids(
    name: str | None,
    source_description: str | None,
    content: str | None,
    provenance: str | None = None,
) -> set[str]:
    """The chat sessions an episode says it came from (usually one).

    ``provenance`` is what a tombstone kept of its envelope; a live episode
    has it in ``content`` instead.
    """
    kept = provenance or envelope_provenance(content) or ""
    matches = [
        _EPISODE_NAME.search(name or ""),
        _SOURCE_DESCRIPTION.search(source_description or ""),
        _PROVENANCE.match(kept),
    ]
    return {match.group("session") for match in matches if match}


_HIDDEN_EPISODES_QUERY = (
    forgotten_facts_clause()
    + f"""
MATCH (n:Episodic {{group_id: $g}})
WHERE NOT ({recallable_episode_predicate("n")})
RETURN n.name AS name, n.source_description AS source_description,
       n.content AS content, n.provenance AS provenance
"""
)
