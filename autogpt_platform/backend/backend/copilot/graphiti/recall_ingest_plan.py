"""What ``recall_ingest.keep_forgotten`` repairs after one ``add_episode``,
decided from the forgotten edges before it (``snapshot_forgotten``), the
forget stash and the graph as graphiti left it.

A forget covers an episode said before it (``IngestRun.said_at``: its
reference time, else when ingestion began): a statement graphiti merged into
the forgotten edge stays one of its sources, and a new edge saying the
forgotten sentence again is forgotten too. A forget that covers the episode
and was stamped after ingestion began (less a clock-skew margin) landed
while graphiti held its older copy, so it is applied again whole.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.nodes import EpisodicNode
from pydantic import BaseModel

from .recall import forgotten_fact_predicate
from .recall_restate import normalized
from .recall_restore import FIELDS, ForgottenEdge, RestoreSpec, needs_restore, spec_for
from .recall_stash import ForgetRecord

logger = logging.getLogger(__name__)

# A forget stamped this long before the episode started may still have been
# missed by it: the forget and the ingestion can run on different hosts.
_CLOCK_SKEW = timedelta(seconds=5)


class IngestRun(BaseModel):
    """One ``add_episode`` as the forget repair needs it. ``said_at`` is when
    the episode was said (its reference time, else when ingestion began)."""

    group_id: str
    started_at: datetime
    reference_time: datetime | None = None
    previous: list[str]
    instructions: str | None = None

    @property
    def said_at(self) -> datetime:
        return self.reference_time or self.started_at


class Reapply(BaseModel):
    uuid: str
    hard: bool
    reason: str


class Plan(BaseModel):
    """What ``keep_forgotten`` does after one ``add_episode``."""

    restores: list[RestoreSpec] = []
    merged: list[RestoreSpec] = []
    covered: list[RestoreSpec] = []
    unlink: list[str] = []
    reapply: list[Reapply] = []
    scrub: list[str] = []
    forgotten: set[str] = set()


async def snapshot_forgotten(driver: GraphDriver) -> dict[str, ForgottenEdge]:
    """Every forgotten fact as it stands before an ``add_episode``."""
    try:
        rows = _rows(await driver.execute_query(_SNAPSHOT_QUERY))
    except Exception:
        logger.warning("Forgotten-fact snapshot failed; not guarded", exc_info=True)
        return {}
    return {row["uuid"]: _edge(row) for row in rows}


async def make_plan(
    driver: GraphDriver,
    run: IngestRun,
    before: dict[str, ForgottenEdge],
    stashed: dict[str, ForgetRecord],
    result: AddEpisodeResults,
) -> Plan:
    """Decide, from the graph as ``add_episode`` left it, what to repair."""
    touched = {edge.uuid for edge in result.edges}
    states = await _read(driver, set(stashed) | (touched & set(before)))
    plan = Plan(forgotten=set(before) | set(stashed))
    for record in stashed.values():
        _plan_landed(plan, run, record, record.uuid in states)
    again = {entry.uuid for entry in plan.reapply}
    for uuid in ((touched & set(before)) | set(stashed)) - again:
        spec = spec_for(stashed.get(uuid), before.get(uuid))
        state = states.get(uuid)
        if spec is not None and state is not None:
            _plan_restore(plan, run, spec, state, result.episode)
    plan.reapply += _restated_forgotten(run, before, stashed, result)
    plan.reapply = list({entry.uuid: entry for entry in plan.reapply}.values())
    plan.forgotten |= {entry.uuid for entry in plan.reapply}
    return plan


def _plan_landed(
    plan: Plan, run: IngestRun, record: ForgetRecord, exists: bool
) -> None:
    """A forget that landed while the episode ran is applied again."""
    forgotten_at = _when(record.forgotten_at)
    if forgotten_at is None or forgotten_at < run.started_at - _CLOCK_SKEW:
        return
    if forgotten_at < run.said_at:
        return
    if exists:
        reason = record.expiration_reason
        plan.reapply.append(Reapply(uuid=record.uuid, hard=record.hard, reason=reason))
    else:
        plan.scrub += [uuid for uuid in (record.source, record.target) if uuid]


def _plan_restore(
    plan: Plan,
    run: IngestRun,
    spec: RestoreSpec,
    state: ForgottenEdge,
    episode: EpisodicNode,
) -> None:
    """Put back a forgotten edge the episode changed, and decide whether the
    episode's statement is a source of the forgotten fact or states it
    again."""
    absorbed = episode.uuid in (state.fields.get("episodes") or [])
    covered = _covers(spec.exact.get("forgotten_at"), run.said_at)
    if absorbed and covered:
        plan.covered.append(spec)
    elif absorbed:
        spec.dropped = [*spec.dropped, episode.uuid]
        plan.merged.append(spec)
        plan.unlink.append(spec.uuid)
    elif spec.uuid in episode.entity_edges:
        plan.unlink.append(spec.uuid)
    if needs_restore(state, spec):
        plan.restores.append(spec)


class _Sentence(BaseModel):
    """A forgotten fact's sentence and ends, to recognise it said again."""

    text: str
    source: str | None
    target: str | None
    names: tuple[str, str] | None
    hard: bool
    reason: str


def _restated_forgotten(
    run: IngestRun,
    before: dict[str, ForgottenEdge],
    stashed: dict[str, ForgetRecord],
    result: AddEpisodeResults,
) -> list[Reapply]:
    """The new edges that say again a fact forgotten after the episode was
    said: the forget covers them."""
    sentences = _covered_sentences(run, before, stashed)
    if not sentences:
        return []
    names = {node.uuid: node.name.lower() for node in result.nodes}
    new = [
        edge
        for edge in result.edges
        if edge.uuid not in before
        and edge.uuid not in stashed
        and edge.episodes == [result.episode.uuid]
    ]
    return [
        Reapply(uuid=edge.uuid, hard=sentence.hard, reason=sentence.reason)
        for edge in new
        for sentence in sentences
        if _says(edge, sentence, names)
    ]


def _covered_sentences(
    run: IngestRun,
    before: dict[str, ForgottenEdge],
    stashed: dict[str, ForgetRecord],
) -> list[_Sentence]:
    records = [
        _Sentence(
            text=record.fact_redacted,
            source=record.source,
            target=record.target,
            names=_names(record),
            hard=record.hard,
            reason=record.expiration_reason,
        )
        for record in stashed.values()
        if record.fact_redacted and _covers(record.forgotten_at, run.said_at)
    ]
    snapshots = [
        _Sentence(
            text=edge.fields["fact_redacted"],
            source=edge.source,
            target=edge.target,
            names=None,
            hard=False,
            reason=edge.fields.get("expiration_reason") or "",
        )
        for edge in before.values()
        if edge.fields.get("fact_redacted")
        and _covers(edge.fields.get("forgotten_at"), run.said_at)
    ]
    return records + snapshots


def _says(edge: EntityEdge, sentence: _Sentence, names: dict[str, str]) -> bool:
    if normalized(edge.fact) != normalized(sentence.text):
        return False
    ends = (edge.source_node_uuid, edge.target_node_uuid)
    if ends == (sentence.source, sentence.target):
        return True
    return sentence.names == (names.get(ends[0]), names.get(ends[1]))


def _names(record: ForgetRecord) -> tuple[str, str] | None:
    if record.source_name is None or record.target_name is None:
        return None
    return record.source_name.lower(), record.target_name.lower()


def _covers(forgotten_at: str | None, said_at: datetime) -> bool:
    """Whether a forget came after the episode was said, so covers it."""
    when = _when(forgotten_at)
    return when is not None and when >= said_at


def _when(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


async def _read(driver: GraphDriver, uuids: set[str]) -> dict[str, ForgottenEdge]:
    if not uuids:
        return {}
    rows = _rows(await driver.execute_query(_READ_QUERY, uuids=sorted(uuids)))
    return {row["uuid"]: _edge(row) for row in rows}


def _edge(row: dict[str, Any]) -> ForgottenEdge:
    return ForgottenEdge(
        uuid=row["uuid"],
        source=row["source"],
        target=row["target"],
        fields={field: row[field] for field in FIELDS},
    )


def _rows(result: Any) -> list[dict[str, Any]]:
    return result[0] if result else []


_RETURN = "e.uuid AS uuid, source.uuid AS source, target.uuid AS target, " + ", ".join(
    f"e.{field} AS {field}" for field in FIELDS
)

_SNAPSHOT_QUERY = f"""
MATCH (source)-[e:RELATES_TO]->(target)
WHERE {forgotten_fact_predicate("e")}
RETURN {_RETURN}
"""

# By uuid, not by the forgotten test: graphiti's rewrite can strip the
# markers that test reads.
_READ_QUERY = f"""
MATCH (source)-[e:RELATES_TO]->(target)
WHERE e.uuid IN $uuids
RETURN {_RETURN}
"""
