import logging

from backend.copilot.graphiti.client import derive_memory_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.copilot.tools.graphiti_forget import mark_edges_superseded

logger = logging.getLogger(__name__)

LEARNED_NOTE_DELETED_REASON = "user_signal:learned_note_deleted"


async def invalidate_learned_rule(
    user_id: str, expert_id: str, rule_id: str | None
) -> bool:
    if not rule_id:
        return False
    try:
        group_id = derive_memory_group_id(user_id, expert_id)
    except ValueError:
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
            "Failed to invalidate a deleted expert learned note",
            exc_info=True,
        )
        return False
