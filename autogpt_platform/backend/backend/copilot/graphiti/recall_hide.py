"""What a forget hides beyond the edge: the fact's sentence, wherever graphiti
would read it back, and the episodes it came from.

The edge's ``fact`` and ``name`` move to ``fact_redacted`` /
``name_redacted`` for audit and both read ``recall.FORGOTTEN_FACT``:
graphiti offers every edge, forgotten ones included, as a duplicate or
contradiction candidate in its own prompts. The ``name`` goes too because
graphiti picks an edge's attribute prompt by it, and that prompt lists
every stored property of the edge, ``fact_redacted`` included; no edge type
is named ``[forgotten]``, so the prompt never runs for a forgotten edge. An
audit copy, once written, is never replaced by the placeholder.

graphiti also keeps what it read out of the sentence on entities: a summary
built from fact sentences, and typed attributes (``Person.role``) it sends
to its attribute, summary and entity resolution prompts. So every entity
the fact joins, and every entity an episode citing it mentions, loses its
summary and every property but its identity (``_CORE_ENTITY_FIELDS``), and
so does the summary of every community one of them belongs to. They are
blanked, not rewritten: graphiti extracts them again from the next episode
that mentions the entity, and the weekly rebuild restores the communities.
Every episode citing the fact is stamped ``redacted_at``.

Each write is idempotent, so forgetting again finishes a forget whose
clean-up failed. The legacy-forget backfill hides through ``scrub`` too.
"""

import logging
from typing import Any

from graphiti_core.driver.driver import GraphDriver

from .memory_model import ForgetResult, MemoryForgetFailure
from .recall import FORGOTTEN_FACT, forgotten_facts_clause, recallable_episode_predicate

logger = logging.getLogger(__name__)

# What an entity keeps when a forget scrubs it: its identity. Everything
# else graphiti stored on it (typed attributes, an ``attributes`` map) goes.
_CORE_ENTITY_FIELDS = frozenset(
    {"uuid", "name", "group_id", "labels", "created_at", "name_embedding", "summary"}
)


async def hide(
    driver: GraphDriver,
    group_id: str,
    uuids: list[str],
    now: str,
    result: ForgetResult,
) -> bool:
    """Scrub the retracted facts' text, then redact every episode citing
    one; False when a write failed, each edge then carrying a
    ``cleanup_error`` (recall hides the text regardless)."""
    if not uuids:
        return True
    try:
        await scrub(driver, uuids)
        records = await driver.execute_query(
            REDACT_EPISODES_QUERY, uuids=uuids, now=now
        )
    except Exception as exc:
        logger.warning(
            f"Edges retracted but hiding their text failed in graph {group_id[:20]}",
            exc_info=True,
        )
        result.failures.extend(
            MemoryForgetFailure.cleanup_error(uuid, exc) for uuid in uuids
        )
        return False
    result.redacted_episodes = [row["uuid"] for row in _rows(records)]
    return True


async def scrub(driver: GraphDriver, uuids: list[str]) -> None:
    """Move the facts' text to their audit copies, then clear what graphiti
    read out of it onto entities and communities."""
    rows = _rows(
        await driver.execute_query(
            SCRUB_FACTS_QUERY, uuids=uuids, placeholder=FORGOTTEN_FACT
        )
    )
    await scrub_entities(driver, uuids, rows[0]["ends"] if rows else [])


async def scrub_entities(
    driver: GraphDriver, uuids: list[str], entities: list[str]
) -> None:
    """Clear the summary and attributes of ``entities`` and of every entity
    an episode citing one of ``uuids`` mentions, and blank their
    communities' summaries. A read picks the properties to drop, since
    Cypher cannot name them from data."""
    found = _rows(
        await driver.execute_query(_ENTITY_KEYS_QUERY, uuids=uuids, entities=entities)
    )
    cleared = [
        {
            "uuid": row["uuid"],
            "cleared": {k: None for k in row["keys"] if k not in _CORE_ENTITY_FIELDS},
        }
        for row in found
    ]
    if cleared:
        await driver.execute_query(_SCRUB_ENTITIES_QUERY, entities=cleared)


def _rows(result: Any) -> list[dict[str, Any]]:
    return result[0] if result else []


# ``sentence`` and ``relation`` are the audit copy already written, else the
# edge's own text unless it already reads the placeholder: a repeat changes
# nothing, and never files the placeholder as the original.
SCRUB_FACTS_QUERY = """
MATCH (source)-[e:RELATES_TO]->(target)
WHERE e.uuid IN $uuids
WITH e, source, target,
     coalesce(e.fact_redacted,
              CASE WHEN e.fact <> $placeholder THEN e.fact END) AS sentence,
     coalesce(e.name_redacted,
              CASE WHEN e.name <> $placeholder THEN e.name END) AS relation
SET e.fact_redacted = sentence,
    e.name_redacted = relation,
    e.fact = $placeholder,
    e.name = $placeholder
RETURN collect(DISTINCT source.uuid) + collect(DISTINCT target.uuid) AS ends
"""

# The endpoints passed in, plus every entity an episode citing a forgotten
# fact mentions (the episodes the redaction below hides).
_ENTITY_KEYS_QUERY = """
OPTIONAL MATCH (ep:Episodic)-[:MENTIONS]->(mentioned:Entity)
WHERE any(x IN coalesce(ep.entity_edges, []) WHERE x IN $uuids)
WITH collect(DISTINCT mentioned.uuid) + $entities AS targets
MATCH (n:Entity)
WHERE n.uuid IN targets
RETURN n.uuid AS uuid, keys(n) AS keys
"""

# ``+=`` with a null value removes that property. A community is matched
# through its ``HAS_MEMBER`` edge to a scrubbed entity.
_SCRUB_ENTITIES_QUERY = """
UNWIND $entities AS entity
MATCH (n:Entity {uuid: entity.uuid})
SET n += entity.cleared, n.summary = ''
WITH collect(n) AS scrubbed
OPTIONAL MATCH (c:Community)-[:HAS_MEMBER]->(member)
WHERE member IN scrubbed
SET c.summary = ''
"""

# Every episode naming one of ``$uuids`` that the recall policy now hides,
# which after the retraction is all of them: the stamp and the read-side test
# cannot disagree. ``coalesce`` keeps the time of the first redaction. ``via``
# names the facts among ``$uuids`` it cites (the forget's cascade follows it,
# ``recall_cascade.py``).
REDACT_EPISODES_QUERY = (
    forgotten_facts_clause()
    + f"""
MATCH (ep:Episodic)
WHERE any(x IN coalesce(ep.entity_edges, []) WHERE x IN $uuids)
  AND NOT ({recallable_episode_predicate("ep")})
SET ep.redacted_at = coalesce(ep.redacted_at, $now)
RETURN ep.uuid AS uuid, [x IN ep.entity_edges WHERE x IN $uuids] AS via
"""
)
