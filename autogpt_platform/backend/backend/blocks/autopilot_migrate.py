"""Backfill `transport` on AutoPilot nodes saved before the field existed.

Those nodes encode "use my ChatGPT subscription" as the mere presence of a
codex credential. Once `transport` exists, pydantic fills its `platform`
default for them, so the builder would show a transport the node does not
actually use. The block honours the credential regardless (see
`AutoPilotBlock.run`), so this migration corrects what is displayed and
removes the divergence — it does not change which account pays.

Runs automatically at API startup (see `api/rest_api.py`), and is idempotent —
nodes already carrying the right transport are skipped, so repeated boots are
a no-op. Also runnable by hand; dry-run by default, pass --apply to write.
"""

import argparse
import asyncio
import logging
from typing import Any, cast

from prisma.models import AgentNode

from backend.blocks.autopilot import AutoPilotBlock, AutoPilotTransport
from backend.data.db import connect, disconnect
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

AUTOPILOT_BLOCK_ID = "c069dc6b-c3ed-4c12-b6e5-d47361e64ce6"


async def migrate_autopilot_transport(*, apply: bool) -> int:
    """Set transport=codex_app_server on nodes carrying a codex credential.

    Returns the number of nodes that needed the backfill.
    """
    nodes = await AgentNode.prisma().find_many(
        where={"agentBlockId": AUTOPILOT_BLOCK_ID}
    )

    stale = []
    for node in nodes:
        constants = dict(node.constantInput or {})
        credential = constants.get("codex_credentials")
        # Only a real selection counts; an id-less meta means nothing was set.
        if not isinstance(credential, dict):
            continue
        if not cast(dict[str, Any], credential).get("id"):
            continue
        if constants.get("transport") == AutoPilotTransport.CODEX_APP_SERVER.value:
            continue
        stale.append((node, constants))

    for node, constants in stale:
        constants["transport"] = AutoPilotTransport.CODEX_APP_SERVER.value
        logger.info(
            "%s node #%s -> transport=codex_app_server",
            "migrating" if apply else "would migrate",
            node.id,
        )
        if apply:
            # SafeJson, not the raw dict: prisma rejects a plain dict for a
            # Json column ("should be of any of the following types:
            # JsonNullValueInput, Json").
            await AgentNode.prisma().update(
                where={"id": node.id}, data={"constantInput": SafeJson(constants)}
            )

    # Silent when there is nothing to do: this runs on every boot, and a
    # steady "0 nodes" line would be pure noise.
    if stale:
        logger.info(
            "%d AutoPilot node(s) %s backfill",
            len(stale),
            "given" if apply else "need (dry run — pass --apply to write)",
        )
    return len(stale)


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Import for its side effect of registering the block id used above.
    assert AutoPilotBlock().id == AUTOPILOT_BLOCK_ID

    await connect()
    try:
        await migrate_autopilot_transport(apply=args.apply)
    finally:
        await disconnect()


if __name__ == "__main__":
    asyncio.run(_main())
