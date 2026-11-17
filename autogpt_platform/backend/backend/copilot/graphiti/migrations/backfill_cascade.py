"""The derivation backfill's cascade from the forgets already made
(``--cascade-existing-forgets``, ``backfill_derivations.py``).

It finds every root a forget left in the graph and runs a forget's cascade
from it (``recall_cascade.py``):

- every fact still in the graph that is forgotten, softly;
- every fact that is gone but that something still names as what it rests
  on or was hidden for: a derivation record (a fact's or an episode's), a
  pending citation marker, an episode's ``redacted_for``, or an earlier
  cascade's reason. A hard forget purged it, perhaps before its cascade ran
  or finished, so the cascade from it erases (``recall_erase.py``);
- every hidden episode a record names as a source, so what rests on an
  episode a forget hid is reached even when the forgotten fact is gone and
  nothing links the two any more (a forget made before ``redacted_for``). It
  names the root it was hidden for when it remembers one, else itself; one
  a hard forget emptied (``hard_deleted_at``) is erased from.

The hard roots cascade first, then the soft. A fact both reach is erased
either way: a hard walk erases the derived facts it only walks through. A
dry run counts the roots and writes nothing.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.copilot.graphiti.memory_model import ForgetResult
from backend.copilot.graphiti.recall import (
    forgotten_fact_predicate,
    forgotten_facts_clause,
    recallable_episode_predicate,
)
from backend.copilot.graphiti.recall_cascade import cascade
from backend.copilot.graphiti.recall_cascade_walk import DERIVED_FROM_FORGOTTEN
from backend.copilot.graphiti.recall_derivation import MARKER_LABEL

from .backfill_pages import BATCH_SIZE, pages, rows

_PREFIX = f"{DERIVED_FROM_FORGOTTEN}:"


class Roots(BaseModel):
    """The roots of one graph's forgets: facts and hidden episodes (each to
    the root it names), those to erase from (``hard``) and the rest."""

    soft: list[str] = Field(default_factory=list)
    hard: list[str] = Field(default_factory=list)
    soft_seeds: dict[str, str] = Field(default_factory=dict)
    hard_seeds: dict[str, str] = Field(default_factory=dict)

    def count(self) -> int:
        return sum(
            len(part)
            for part in (self.soft, self.hard, self.soft_seeds, self.hard_seeds)
        )


class Cascaded(BaseModel):
    """What the cascade found and did: its ``roots``, the facts it retracted
    (``derived``), and whether a cascade stopped short (``failed``)."""

    roots: int = 0
    derived: int = 0
    failed: bool = False


async def cascade_existing_forgets(
    driver: AutoGPTFalkorDriver, *, apply: bool
) -> Cascaded:
    """Cascade from every root the graph's forgets left, hard ones first,
    erasing; count them only on a dry run."""
    roots = await find_roots(driver)
    done = Cascaded(roots=roots.count())
    if not (apply and done.roots):
        return done
    result = ForgetResult()
    now = datetime.now(timezone.utc).isoformat()
    graph = driver.graph_name
    for facts, seeds, erase in (
        (roots.hard, roots.hard_seeds, True),
        (roots.soft, roots.soft_seeds, False),
    ):
        if facts or seeds:
            await cascade(driver, graph, facts, now, result, erase=erase, seeds=seeds)
    done.derived = len(result.derived)
    done.failed = bool(result.failures)
    return done


async def find_roots(driver: AutoGPTFalkorDriver) -> Roots:
    """Every fact a user forgot that is still there (one a cascade retracted
    is picked up from its root, whose name its reason keeps), every gone
    fact still named, and every hidden episode a record names, as
    ``Roots``."""
    named_facts: dict[str, None] = {}
    named_episodes: dict[str, None] = {}
    rooted: dict[str, str] = {}
    for row in [
        *await pages(driver, FACT_NAMES_QUERY),
        *await pages(driver, EPISODE_NAMES_QUERY),
        *rows(await driver.execute_query(MARKER_NAMES_QUERY)),
    ]:
        named_facts |= dict.fromkeys([*row["facts"], *row["hidden_for"]])
        named_episodes |= dict.fromkeys(row["episodes"])
        if (row["reason"] or "").startswith(_PREFIX):
            rooted[row["uuid"]] = row["reason"].removeprefix(_PREFIX)
            named_facts[rooted[row["uuid"]]] = None
    forgotten = await pages(driver, FORGOTTEN_FACTS_QUERY)
    gone = await _gone(driver, list(named_facts))
    hidden = await _hidden(driver, list(named_episodes))
    seeds = {row["uuid"]: (_name(row, rooted), row["hard"]) for row in hidden}
    return Roots(
        soft=[row["uuid"] for row in forgotten if row["uuid"] not in rooted],
        hard=gone,
        soft_seeds={uuid: name for uuid, (name, hard) in seeds.items() if not hard},
        hard_seeds={uuid: name for uuid, (name, hard) in seeds.items() if hard},
    )


async def _gone(driver: AutoGPTFalkorDriver, uuids: list[str]) -> list[str]:
    """Those of ``uuids`` no fact in the graph has."""
    found: set[str] = set()
    for start in range(0, len(uuids), BATCH_SIZE):
        chunk = uuids[start : start + BATCH_SIZE]
        result = await driver.execute_query(EXISTING_FACTS_QUERY, uuids=chunk)
        found |= {row["uuid"] for row in rows(result)}
    return [uuid for uuid in uuids if uuid not in found]


async def _hidden(driver: AutoGPTFalkorDriver, uuids: list[str]) -> list[dict]:
    """Those of ``uuids`` that are episodes the recall policy hides."""
    hidden: list[dict[str, Any]] = []
    for start in range(0, len(uuids), BATCH_SIZE):
        chunk = uuids[start : start + BATCH_SIZE]
        result = await driver.execute_query(HIDDEN_EPISODES_QUERY, uuids=chunk)
        hidden.extend(rows(result))
    return hidden


def _name(row: dict[str, Any], rooted: dict[str, str]) -> str:
    """The root a hidden episode's retractions name: that of the first
    forgotten fact it remembers it was hidden for (the fact itself, or the
    root a cascade that retracted it named), else the episode itself."""
    hidden_for: list[str] = row["hidden_for"] or []
    if not hidden_for:
        return str(row["uuid"])
    return rooted.get(hidden_for[0], hidden_for[0])


FORGOTTEN_FACTS_QUERY = f"""
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid > $after AND {forgotten_fact_predicate("e")}
RETURN e.uuid AS uuid
ORDER BY uuid
LIMIT $limit
"""

# Every fact's record and an earlier cascade's reason.
FACT_NAMES_QUERY = f"""
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid > $after
  AND (e.derived_from_facts IS NOT NULL
       OR e.expiration_reason STARTS WITH '{_PREFIX}')
RETURN e.uuid AS uuid, coalesce(e.derived_from_facts, []) AS facts,
       coalesce(e.derived_from_episodes, []) AS episodes, [] AS hidden_for,
       e.expiration_reason AS reason
ORDER BY uuid
LIMIT $limit
"""

# Every episode's record and what it was hidden for.
EPISODE_NAMES_QUERY = """
MATCH (ep:Episodic)
WHERE ep.uuid > $after
  AND (ep.derived_from_facts IS NOT NULL OR ep.redacted_for IS NOT NULL)
RETURN ep.uuid AS uuid, coalesce(ep.derived_from_facts, []) AS facts,
       coalesce(ep.derived_from_episodes, []) AS episodes,
       coalesce(ep.redacted_for, []) AS hidden_for, NULL AS reason
ORDER BY uuid
LIMIT $limit
"""

MARKER_NAMES_QUERY = f"""
MATCH (m:{MARKER_LABEL})
RETURN coalesce(m.derived_from_facts, []) AS facts,
       coalesce(m.derived_from_episodes, []) AS episodes, [] AS hidden_for,
       NULL AS reason
"""

EXISTING_FACTS_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid IN $uuids
RETURN e.uuid AS uuid
"""

HIDDEN_EPISODES_QUERY = (
    forgotten_facts_clause()
    + f"""
MATCH (ep:Episodic)
WHERE ep.uuid IN $uuids AND NOT ({recallable_episode_predicate("ep")})
RETURN ep.uuid AS uuid, ep.hard_deleted_at IS NOT NULL AS hard,
       coalesce(ep.redacted_for, []) AS hidden_for
"""
)
