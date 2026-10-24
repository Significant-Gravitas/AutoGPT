"""AutoPilot's long-term memory of the account: one FalkorDB graph for Otto's
memory of the user and one per expert they own (`copilot/graphiti/client.py`).
"""

from falkordb.asyncio import FalkorDB

from backend.copilot.graphiti.client import derive_memory_group_id, evict_client
from backend.copilot.graphiti.config import graphiti_config
from backend.data.account_erasure.models import StepReport, StepStatus, Subject


async def delete_memory(subject: Subject, *, dry_run: bool) -> StepReport:
    """Delete the account's memory graphs that exist. A graph is only ever
    created for an account the memory flag was on for, so most have none.

    Skipped only on a stack with no FalkorDB host configured. A configured
    one that cannot be reached fails the step, so the deletion stops before
    the account's rows go rather than leave its memory behind unnoticed."""
    if not graphiti_config.falkordb_host.strip():
        return StepReport(
            name="memory",
            status=StepStatus.SKIPPED,
            note="no FalkorDB on this stack (GRAPHITI_FALKORDB_HOST is empty)",
        )
    wanted = memory_graph_names(subject)
    db = FalkorDB(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
    )
    try:
        present = [name for name in await db.list_graphs() if name in wanted]
        if not dry_run:
            for name in present:
                await db.select_graph(name).delete()
                await evict_client(name)
    finally:
        await db.aclose()
    return StepReport(
        name="memory",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"memory_graphs": len(present)},
    )


def memory_graph_names(subject: Subject) -> set[str]:
    return {
        derive_memory_group_id(subject.user_id, expert_id)
        for expert_id in [None, *subject.expert_ids]
    }
