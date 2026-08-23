"""Graphiti side-effect of deleting a learned note.

Archiving the Postgres row only stops the note being *rendered*. The rule it
was promoted from is still an active edge in the user's graph, so the next
nightly dream pass would re-derive and re-promote it — the user would delete
the same note over and over. Deleting a note therefore also demotes the source
edge, which is what makes the deletion durable.
"""

import logging

from backend.copilot.graphiti.client import derive_memory_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.copilot.tools.graphiti_forget import mark_edges_superseded

logger = logging.getLogger(__name__)

# Recorded on the demoted edge so an audit can tell a user-driven retraction
# from a dream-pass staleness demotion.
LEARNED_NOTE_DELETED_REASON = "user_signal:learned_note_deleted"


async def invalidate_learned_rule(
    user_id: str, expert_id: str | None, rule_id: str | None
) -> bool:
    """Demote the Graphiti edge a deleted note came from.

    Best-effort: the note is already archived by the time this runs, so a
    graph error costs at most a re-promotion on some later pass — worth far
    less than failing the user's delete. Returns whether an edge was demoted.
    """
    if not rule_id:
        return False
    try:
        group_id = derive_memory_group_id(user_id, expert_id)
    except ValueError:
        logger.warning(
            "Cannot derive a memory scope for user %s — learned rule left active",
            user_id[:12],
        )
        return False

    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        build_indices=False,
    )
    try:
        try:
            demoted, _ = await mark_edges_superseded(
                driver,
                [rule_id],
                LEARNED_NOTE_DELETED_REASON,
                new_status="contradicted",
                user_id=user_id,
                group_id=group_id,
            )
            return bool(demoted)
        finally:
            await driver.close()
    except Exception:
        logger.warning(
            "Failed to invalidate learned rule for user %s — the note stays "
            "archived but may be re-promoted",
            user_id[:12],
            exc_info=True,
        )
        return False
