"""
Record each Exa Agent run's cost exactly once, whichever block first sees it finish.

Several blocks can see the same finished run: Exa Agent (when it waits), Wait,
Get, Cancel and Stop, and the legacy Research blocks. Recording the cost in
each of them would count a workflow that waits and then reads a run twice, so
a per-user claim on the run ID makes only the first one count.

All of these blocks refuse the platform's shared Exa key, so runs only happen
on the user's own key, and a run on the user's own key is never charged
credits: the Exa base cost only applies to the platform's keys. What is
recorded is the Exa-reported cost in the node's execution stats. The claim
keeps that figure from being counted twice; it does not guard a charge.
"""

import logging
from asyncio import timeout

from redis.exceptions import RedisClusterException, RedisError

from backend.data.redis_client import get_redis_async
from backend.sdk import Block

from .agent_models import AgentRun
from .helpers import merge_exa_cost

logger = logging.getLogger(__name__)

BILLED_KEY_PREFIX = "exa:agent_run_billed"
# Comfortably longer than any workflow keeps re-reading the same run.
BILLED_KEY_TTL_SECONDS = 90 * 24 * 60 * 60
# An unreachable Redis must not stall the block that is reporting the run.
CLAIM_TIMEOUT_SECONDS = 5


async def bill_finished_run(block: Block, run: AgentRun, user_id: str) -> None:
    """Record a finished run's cost on ``block`` the first time it is seen.

    Unfinished runs are skipped: their cost has not settled yet.
    """
    if not run.is_terminal:
        return
    if not await claim_run_billing(user_id, run.run_id):
        return
    merge_exa_cost(block, {"costDollars": {"total": run.cost.total}})


async def claim_run_billing(user_id: str, run_id: str) -> bool:
    """Claim ``run_id``'s cost for ``user_id``; False if already claimed.

    If Redis can't be reached in time the cost is recorded anyway: the figure
    is only reported, so counting it twice now and then is better than
    leaving it out.
    """
    key = f"{BILLED_KEY_PREFIX}:{user_id}:{run_id}"
    try:
        async with timeout(CLAIM_TIMEOUT_SECONDS):
            redis = await get_redis_async()
            return bool(await redis.set(key, "1", nx=True, ex=BILLED_KEY_TTL_SECONDS))
    except (
        RedisError,
        RedisClusterException,
        ConnectionError,
        OSError,
        TimeoutError,
    ) as exc:
        logger.warning(
            f"Could not check whether Exa Agent run {run_id}'s cost was already "
            f"recorded, recording it: {exc}"
        )
        return True
