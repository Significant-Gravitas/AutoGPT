"""What a forget hides beyond the edge: the fact's sentence, wherever graphiti
would read it back, and the episodes it came from.

The edge's ``fact`` and ``name`` move to ``fact_redacted`` /
``name_redacted`` for audit and both read ``recall.FORGOTTEN_FACT``:
graphiti offers every edge, forgotten ones included, as a duplicate or
contradiction candidate in its own prompts. The ``name`` goes too because
graphiti picks an edge's attribute prompt by it, and that prompt lists
every stored property of the edge, ``fact_redacted`` included; no edge type
is named ``[forgotten]``, so the prompt never runs for a forgotten edge.
The summaries of both endpoint entities, and of every community either
belongs to, are blanked rather than rewritten: graphiti built them from
fact sentences and reads them into its entity resolution prompt; it grows
an empty entity summary back from the next episode that mentions the
entity, and the weekly community rebuild restores the rest. Every episode
citing the fact is stamped ``redacted_at``.

Each write is idempotent, so forgetting again finishes a forget whose
clean-up failed. The legacy-forget backfill reuses both queries.
"""

import logging

from .falkordb_driver import AutoGPTFalkorDriver
from .memory_model import ForgetResult, MemoryForgetFailure
from .recall import FORGOTTEN_FACT, forgotten_facts_clause, recallable_episode_predicate
from .scope import MemoryScope

logger = logging.getLogger(__name__)


async def hide(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
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
        await driver.execute_query(
            SCRUB_FACTS_QUERY, uuids=uuids, placeholder=FORGOTTEN_FACT
        )
        records = await driver.execute_query(
            REDACT_EPISODES_QUERY, uuids=uuids, now=now
        )
    except Exception as exc:
        logger.warning(
            f"Edges retracted but hiding their text failed for user "
            f"{scope.owner_user_id[:12]}",
            exc_info=True,
        )
        result.failures.extend(
            MemoryForgetFailure.cleanup_error(edge_uuid, exc) for edge_uuid in uuids
        )
        return False
    result.redacted_episodes = [row["uuid"] for row in (records[0] if records else [])]
    return True


# ``sentence`` and ``relation`` are the original text on a first run and the
# audit copies on a repeat, so a second scrub changes nothing. A community is
# matched through its ``HAS_MEMBER`` edge to either endpoint.
SCRUB_FACTS_QUERY = """
MATCH (source)-[e:RELATES_TO]->(target)
WHERE e.uuid IN $uuids
WITH e, source, target,
     coalesce(e.fact_redacted, e.fact) AS sentence,
     coalesce(e.name_redacted, e.name) AS relation
SET e.fact_redacted = sentence,
    e.name_redacted = relation,
    e.fact = $placeholder,
    e.name = $placeholder,
    source.summary = '',
    target.summary = ''
WITH collect(DISTINCT source) + collect(DISTINCT target) AS ends
OPTIONAL MATCH (c:Community)-[:HAS_MEMBER]->(member)
WHERE member IN ends
SET c.summary = ''
"""

# Every episode naming one of ``$uuids`` that the recall policy now hides,
# which after the retraction is all of them: the stamp and the read-side test
# cannot disagree. ``coalesce`` keeps the time of the first redaction.
REDACT_EPISODES_QUERY = (
    forgotten_facts_clause()
    + f"""
MATCH (ep:Episodic)
WHERE any(x IN coalesce(ep.entity_edges, []) WHERE x IN $uuids)
  AND NOT ({recallable_episode_predicate("ep")})
SET ep.redacted_at = coalesce(ep.redacted_at, $now)
RETURN ep.uuid AS uuid
"""
)
