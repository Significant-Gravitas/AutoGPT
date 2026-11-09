"""Local compaction measurements. Observability must never break a turn."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from backend.copilot.sdk.compaction import CompactionStats

logger = logging.getLogger(__name__)

CompactionPath = Literal["pre_query", "sdk_internal"]

# Event metadata schema version. Bump when a shipped event gains, loses,
# or renames keys so dashboards can split rather than average across the
# boundary. (Schema 1 = the unversioned first deploy.)
EVENT_SCHEMA_VERSION = 2


def emit_compaction_event(
    *,
    path: CompactionPath,
    stats: CompactionStats | None,
    after_source: str | None = None,
    trace_id: str | None = None,
    log_prefix: str = "",
) -> None:
    """Log one completed compaction cycle, including cycles without counts."""
    metadata: dict[str, Any] = {"schema": EVENT_SCHEMA_VERSION, "path": path}
    if stats is not None:
        # to_wire() carries the camelCase names the tool row's JSON output
        # uses, so the same cycle reads identically in the DB and local log.
        metadata.update(stats.to_wire())
    metadata["dropped"] = stats.dropped if stats is not None else False
    if after_source is not None:
        metadata["after_source"] = after_source
    logger.info("%s [copilot-compaction] %s", log_prefix, metadata)


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "CompactionPath",
    "emit_compaction_event",
]
