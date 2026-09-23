"""Backfill `transport` on AutoPilot nodes saved before the field existed.

Those nodes encode "use my ChatGPT subscription" as the mere presence of a
codex credential, and carry no transport at all. The block reads that absence
correctly (see `AutoPilotBlock.run`), so this migration changes what the
builder *displays*, not which account pays.

Runs automatically at API startup (see `api/rest_api.py`) and is idempotent:
the statement only matches rows that still need it, so repeated boots are a
no-op. Also runnable by hand; dry-run by default, pass --apply to write.
"""

import argparse
import asyncio
import logging

from backend.blocks.autopilot import AUTOPILOT_BLOCK_ID, AutoPilotTransport
from backend.data.db import (
    connect,
    disconnect,
    execute_raw_with_schema,
    query_raw_with_schema,
)

logger = logging.getLogger(__name__)


# A single atomic statement rather than read-modify-write. Sibling startup
# migrations either lock (org_migration) or use jsonb_set (migrate_llm_models);
# reading every node and writing the whole blob back would let two booting pods
# clobber each other, and any concurrent user edit in between.
#
# The predicate is the idempotency guarantee: it matches only nodes with a real
# connection (an id-less meta means nothing was selected) and no transport yet,
# so re-running selects nothing.
def _match(block_id_parameter: int) -> str:
    return f"""
    "agentBlockId" = ${block_id_parameter}
    AND "constantInput"->'codex_credentials'->>'id' IS NOT NULL
    AND "constantInput"->>'transport' IS NULL
"""


_COUNT_QUERY = f"""
    SELECT COUNT(*)::int AS count
    FROM {{schema_prefix}}"AgentNode"
    WHERE {_match(1)}
"""

# ARRAY['transport'] rather than the '{transport}' path literal: these
# templates are run through str.format() to inject the schema prefix, so a
# literal brace becomes a format field and raises KeyError at apply time.
_UPDATE_QUERY = f"""
    UPDATE {{schema_prefix}}"AgentNode"
    SET "constantInput" = jsonb_set("constantInput", ARRAY['transport'], to_jsonb($1::text), true)
    WHERE {_match(2)}
"""


async def migrate_autopilot_transport(*, apply: bool) -> int:
    """Set transport=codex_app_server on nodes carrying a codex connection.

    Returns the number of nodes that needed the backfill.
    """
    if apply:
        updated = await execute_raw_with_schema(
            _UPDATE_QUERY,
            AutoPilotTransport.CODEX_APP_SERVER.value,
            AUTOPILOT_BLOCK_ID,
        )
        if updated:
            logger.info(
                "%d AutoPilot node(s) given transport=codex_app_server", updated
            )
        return updated

    rows = await query_raw_with_schema(_COUNT_QUERY, AUTOPILOT_BLOCK_ID)
    pending = int(rows[0]["count"]) if rows else 0
    if pending:
        logger.info(
            "%d AutoPilot node(s) need backfill (dry run — pass --apply to write)",
            pending,
        )
    return pending


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    await connect()
    try:
        await migrate_autopilot_transport(apply=args.apply)
    finally:
        await disconnect()


if __name__ == "__main__":
    asyncio.run(_main())
