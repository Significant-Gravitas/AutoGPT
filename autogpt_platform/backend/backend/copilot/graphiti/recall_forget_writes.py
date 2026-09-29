"""The dream writes a forget answers for (``recall_forget.py``).

Before anything else a forget completes the dream citation markers its
graph still holds (``recall_reconcile.reconcile``), so its cascade sees the
provenance of every dream write that landed. After its cascade it asks
whether a dream write that could still land cites anything it reached
(``recall_reconcile.in_flight``): a root, a derived fact it retracted or
walked through, an episode it hid. Where the graph's write lock holds, none
can; where it does not (Redis unreachable, a lost lease), one can land
after the forget, and only its own settle (``recall_landing.py``) retracts
it. Either way, when something may be missing, each fact the forget
retracted is a ``cleanup_error``, which asks for the forget again: the
retry reconciles what landed meanwhile, and reports the forget done once no
such write is left.
"""

import logging

from graphiti_core.driver.driver import GraphDriver

from .memory_model import ForgetResult, MemoryForgetFailure
from .recall_reconcile import in_flight, reconcile

logger = logging.getLogger(__name__)


async def unreconciled(driver: GraphDriver, group_id: str) -> Exception | None:
    """Complete the graph's pending dream records before anything else; what
    went wrong, when some may still be missing: the error, or ``None``."""
    try:
        done = await reconcile(driver, group_id)
    except Exception as exc:
        logger.warning(
            f"Could not reconcile graph {group_id[:20]} before a forget",
            exc_info=True,
        )
        return exc
    if done.incomplete():
        return RuntimeError("dream writes that landed are still without a record")
    return None


async def still_landing(
    driver: GraphDriver, group_id: str, result: ForgetResult, roots: list[str]
) -> Exception | None:
    """Why the forget is not done when a dream write that could still land
    cites something it reached (a root, a derived fact it retracted or
    walked through, an episode it hid), else ``None``."""
    if not roots:
        return None
    facts = [*roots, *result.derived, *result.passed]
    try:
        count = await in_flight(driver, facts, result.redacted_episodes)
    except Exception as exc:
        logger.warning(
            f"Could not read the dream writes in flight in graph {group_id[:20]}",
            exc_info=True,
        )
        return exc
    if not count:
        return None
    return RuntimeError(f"{count} dream write(s) citing what it reached may still land")


def provenance_incomplete(
    result: ForgetResult, retracted: list[str], exc: Exception
) -> None:
    """A ``cleanup_error`` on each fact forgotten that has no failure yet: its
    cascade may have missed a dream fact whose record was still pending."""
    failed = {failure.uuid for failure in result.failures}
    result.failures.extend(
        MemoryForgetFailure.cleanup_error(uuid, exc)
        for uuid in retracted
        if uuid not in failed
    )
