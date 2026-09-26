"""Putting a forgotten edge back after graphiti's ``add_episode`` rewrote it.

A restore writes only what a forget owns: the marker, status, reason and
times, the placeholder text and its audit copies. The fields a forget owns
are set back exactly; an audit copy or another ``MemoryFact`` field is only
filled in where graphiti's rewrite left none, so a value another writer
changed meanwhile is kept. The episode that stated the fact again after it
was forgotten is taken off the edge's sources, and nothing else is.

A restore that fails twice is left as an obligation in the forget stash,
which the next ingestion or the next forget of the edge carries out.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel

from .memory_model import MemoryStatus
from .recall import FORGOTTEN_FACT, USER_FORGET_REASON
from .recall_stash import ForgetRecord, stash_forgets
from .types import MemoryFact

logger = logging.getLogger(__name__)

# What a forget sets and a restore puts back exactly.
EXACT_FIELDS = (
    "forgotten_at",
    "status",
    "expiration_reason",
    "expired_at",
    "invalid_at",
    "valid_at",
    "fact",
    "name",
)
AUDIT_FIELDS = ("fact_redacted", "name_redacted")
# The other ``MemoryFact`` fields: only filled in where they went missing.
FILL_FIELDS = tuple(
    f for f in MemoryFact.model_fields if f not in ("status", "expiration_reason")
)
# Every field read off a forgotten edge, before ingestion and after it.
FIELDS = (*EXACT_FIELDS, *AUDIT_FIELDS, *FILL_FIELDS, "episodes")


class ForgottenEdge(BaseModel):
    """A forgotten edge as read from the graph."""

    uuid: str
    source: str
    target: str
    fields: dict[str, Any]


class RestoreSpec(BaseModel):
    """What a restore writes on one edge."""

    uuid: str
    source: str | None = None
    target: str | None = None
    exact: dict[str, Any]
    audit: dict[str, Any]
    fill: dict[str, Any] = {}
    dropped: list[str] = []


def spec_for(
    record: ForgetRecord | None, snapshot: ForgottenEdge | None
) -> RestoreSpec | None:
    """The restore for an edge, from its stashed forget if there is one
    (what the forget set), else from its pre-ingestion snapshot."""
    if record is not None:
        return _from_record(record, snapshot)
    if snapshot is not None:
        return _from_snapshot(snapshot)
    return None


def needs_restore(state: ForgottenEdge, spec: RestoreSpec) -> bool:
    """Whether the edge differs from what the restore would leave."""
    fields = state.fields
    if any(fields.get(key) != value for key, value in spec.exact.items()):
        return True
    missing = {**spec.audit, **spec.fill}
    if any(fields.get(k) is None and v is not None for k, v in missing.items()):
        return True
    return bool(set(spec.dropped) & set(fields.get("episodes") or []))


async def restore(driver: GraphDriver, group_id: str, spec: RestoreSpec) -> bool:
    """Write ``spec``, retrying once; after a second failure, stash it as an
    obligation and report False."""
    for attempt in (1, 2):
        try:
            await driver.execute_query(_RESTORE_QUERY, **_params(spec))
            return True
        except Exception:
            logger.warning(
                f"Restoring forgotten edge {spec.uuid} failed (attempt {attempt})",
                exc_info=True,
            )
    logger.error(
        f"Forgotten edge {spec.uuid} in graph {group_id[:20]} left for the next "
        "ingestion or forget to restore"
    )
    await stash_forgets(group_id, [obligation(spec)])
    return False


def obligation(spec: RestoreSpec) -> ForgetRecord:
    """``spec`` as a stash record, for whoever carries it out next."""
    now = datetime.now(timezone.utc).isoformat()
    exact = spec.exact
    return ForgetRecord(
        uuid=spec.uuid,
        forgotten_at=exact["forgotten_at"] or exact["expired_at"] or now,
        status=exact["status"] or MemoryStatus.retracted.value,
        expiration_reason=exact["expiration_reason"] or USER_FORGET_REASON,
        expired_at=exact["expired_at"] or now,
        invalid_at=exact["invalid_at"],
        valid_at=exact["valid_at"],
        fact=exact["fact"] or FORGOTTEN_FACT,
        fact_redacted=spec.audit["fact_redacted"],
        name=exact["name"] or FORGOTTEN_FACT,
        name_redacted=spec.audit["name_redacted"],
        dropped_episodes=spec.dropped,
        source=spec.source,
        target=spec.target,
        stashed_at=now,
    )


def _from_record(record: ForgetRecord, snapshot: ForgottenEdge | None) -> RestoreSpec:
    before = snapshot.fields if snapshot else {}
    return RestoreSpec(
        uuid=record.uuid,
        source=record.source or (snapshot.source if snapshot else None),
        target=record.target or (snapshot.target if snapshot else None),
        exact=record.model_dump(include=set(EXACT_FIELDS)),
        audit={
            "fact_redacted": record.fact_redacted or before.get("fact_redacted"),
            "name_redacted": record.name_redacted or before.get("name_redacted"),
        },
        fill={key: before.get(key) for key in FILL_FIELDS},
        dropped=record.dropped_episodes,
    )


def _from_snapshot(snapshot: ForgottenEdge) -> RestoreSpec:
    fields = snapshot.fields
    return RestoreSpec(
        uuid=snapshot.uuid,
        source=snapshot.source,
        target=snapshot.target,
        exact={key: fields.get(key) for key in EXACT_FIELDS},
        audit={key: fields.get(key) for key in AUDIT_FIELDS},
        fill={key: fields.get(key) for key in FILL_FIELDS},
    )


def _params(spec: RestoreSpec) -> dict[str, Any]:
    return {
        "uuid": spec.uuid,
        "exact": spec.exact,
        "audit": spec.audit,
        "fill": spec.fill,
        "dropped": spec.dropped,
    }


# ``+=`` sets the forget's own fields back (a null removes one graphiti
# added); the rest only fill a gap. The embedding is left alone.
_RESTORE_QUERY = (
    """
MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
SET e += $exact,
    e.fact_redacted = coalesce(e.fact_redacted, $audit.fact_redacted),
    e.name_redacted = coalesce(e.name_redacted, $audit.name_redacted),
    e.episodes = [x IN coalesce(e.episodes, []) WHERE NOT x IN $dropped],
"""
    + ",\n".join(f"    e.{key} = coalesce(e.{key}, $fill.{key})" for key in FILL_FIELDS)
    + "\n"
)
