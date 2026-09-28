"""A forget reaches what the dream derived from what it forgot.

Each dream write records what it was derived from, on its episode and on the
facts only dream episodes state (``recall_derivation.py``). After a forget
has retracted its facts and hidden their episodes, still holding the graph's
write lock (``recall_forget.py``), ``cascade`` retracts every live fact whose
record names a forgotten fact or a hidden episode, and hides every dream
episode whose record does; then again over what that retracted and hid,
until a round finds nothing new. Any one forgotten source is enough: a fact
derived from a forgotten one carries its content. A derived fact that is no
longer live (superseded, contradicted, or forgotten already) is left as it
is, but the walk goes on through it: what was derived from it rests on the
forgotten fact too. A fact the user stated has no record, and a derived fact
a user's own episode states too is passed over, as is what rests on it: it
has a source the forget did not reach.

A derived fact is retracted as a soft forget retracts one: ``forgotten_at``,
``status='retracted'``, its sentence moved to its audit copy, its entities'
summaries and attributes and their communities' summaries blanked, and every
episode citing it redacted. Its ``expiration_reason`` is
``derived_from_forgotten:<uuid>``, naming the fact the user forgot that it
descends from. A dream episode reached without a live fact of its own (its
fact was superseded, merged into a fact a user stated, or never extracted)
is redacted and the entities it mentions blanked, since the dream reads its
text. A hard forget cascades the same way, softly: the derived facts are the
assistant's inferences, not text the user asked to erase, retraction takes
them out of every read, and deleting them would purge what the model's
citations name, which can be more than a fact truly rests on, beyond undoing.

At most ``CASCADE_MAX_ROUNDS`` rounds and ``CASCADE_MAX_ITEMS`` derived facts
and dream episodes per forget. A forget stopped by a bound or a failed step
reports a ``cleanup_error`` on each fact it forgot, and forgetting them again
picks up the facts already retracted for them (their reason names them) and
goes on from there.
"""

import logging
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel, Field

from .memory_model import ForgetResult, MemoryForgetFailure, MemoryStatus
from .recall_cascade_queries import (
    DERIVED_EPISODES_QUERY,
    DERIVED_FACTS_QUERY,
    EARLIER_QUERY,
    REDACT_DERIVED_QUERY,
    RETRACT_QUERY,
)
from .recall_hide import REDACT_EPISODES_QUERY, scrub, scrub_entities

logger = logging.getLogger(__name__)

# ``expiration_reason`` of a fact the cascade retracted, before the colon.
DERIVED_FROM_FORGOTTEN = "derived_from_forgotten"
CASCADE_MAX_ITEMS = 500
CASCADE_MAX_ROUNDS = 10


def derived_reason(root: str) -> str:
    """The reason recorded on a fact derived from forgotten fact ``root``."""
    return f"{DERIVED_FROM_FORGOTTEN}:{root}"


async def cascade(
    driver: GraphDriver,
    group_id: str,
    roots: list[str],
    now: str,
    result: ForgetResult,
) -> None:
    """Retract what the dream derived from ``roots``, the facts this forget
    has just retracted and hidden: ``result.derived`` gets each fact it
    retracts and ``result.redacted_episodes`` each episode it hides. A failed
    step or a bound reached is a failure on every root."""
    if not roots:
        return
    walk = _Walk(roots=roots, root_of={r: r for r in roots}, budget=CASCADE_MAX_ITEMS)
    try:
        finished = await _run(driver, group_id, walk, now, result)
    except Exception as exc:
        logger.warning(f"Forget cascade failed in graph {group_id[:20]}", exc_info=True)
        result.failures.extend(
            MemoryForgetFailure.cleanup_error(root, exc) for root in roots
        )
        return
    if not finished:
        logger.warning(
            f"Forget cascade in graph {group_id[:20]} stopped at its bound "
            f"after {len(result.derived)} derived facts"
        )
        result.failures.extend(MemoryForgetFailure.derived_left(root) for root in roots)


class _Walk(BaseModel):
    """Every fact and episode the cascade has reached, mapped to the root it
    descends from, and how many more derived items it may retire."""

    roots: list[str]
    root_of: dict[str, str]
    budget: int

    def root(self, via: list[str]) -> str:
        """The root of the first item in ``via`` the walk has reached."""
        return next((self.root_of[x] for x in via if x in self.root_of), self.roots[0])

    def reach(self, rows: list[dict[str, Any]]) -> list[str]:
        """Record each row's ``uuid`` as reached ``via`` its items; the uuids
        not reached before."""
        new = [row for row in rows if row["uuid"] not in self.root_of]
        for row in new:
            self.root_of[row["uuid"]] = self.root(row["via"])
        return [row["uuid"] for row in new]


class _Frontier(BaseModel):
    """What the last round retracted and hid: the next round's search."""

    facts: list[str] = Field(default_factory=list)
    episodes: list[str] = Field(default_factory=list)


class _Found(BaseModel):
    """One round's finds, rows of ``uuid`` and ``via`` (and, for a fact,
    ``live``); ``truncated`` when the walk's budget cut them short."""

    facts: list[dict[str, Any]] = Field(default_factory=list)
    episodes: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False


async def _run(
    driver: GraphDriver,
    group_id: str,
    walk: _Walk,
    now: str,
    result: ForgetResult,
) -> bool:
    """Rounds until one finds nothing (True) or a bound stops them (False)."""
    frontier = await _resume(driver, walk, now, result)
    for _ in range(CASCADE_MAX_ROUNDS):
        found = await _derived(driver, walk, frontier)
        if not (found.facts or found.episodes):
            return True
        frontier = await _retire(driver, group_id, walk, found, now, result)
        if found.truncated:
            return False
    last = await _derived(driver, walk, frontier)
    return not (last.facts or last.episodes)


async def _resume(
    driver: GraphDriver, walk: _Walk, now: str, result: ForgetResult
) -> _Frontier:
    """The first frontier: the roots, the facts an earlier try of this forget
    retracted for them (their clean-up finished again), and every episode
    citing one."""
    reasons = [derived_reason(root) for root in walk.roots]
    earlier = _rows(await driver.execute_query(EARLIER_QUERY, reasons=reasons))
    for row in earlier:
        walk.root_of[row["uuid"]] = row["reason"].removeprefix(
            f"{DERIVED_FROM_FORGOTTEN}:"
        )
    again = [row["uuid"] for row in earlier]
    if again:
        await scrub(driver, again)
    facts = [*walk.roots, *again]
    hidden = await _redact_citing(driver, walk, facts, now, result)
    return _Frontier(facts=facts, episodes=hidden)


async def _derived(driver: GraphDriver, walk: _Walk, frontier: _Frontier) -> _Found:
    """The facts and the dream episodes not reached yet whose record names
    something in ``frontier``, no more than the walk's budget."""
    if not (frontier.facts or frontier.episodes):
        return _Found()
    params = {
        "facts": frontier.facts,
        "episodes": frontier.episodes,
        "seen": list(walk.root_of),
        "limit": walk.budget + 1,
    }
    facts = _rows(await driver.execute_query(DERIVED_FACTS_QUERY, **params))
    episodes = _rows(await driver.execute_query(DERIVED_EPISODES_QUERY, **params))
    if len(facts) + len(episodes) <= walk.budget:
        return _Found(facts=facts, episodes=episodes)
    kept = facts[: walk.budget]
    rest = episodes[: walk.budget - len(kept)]
    return _Found(facts=kept, episodes=rest, truncated=True)


async def _retire(
    driver: GraphDriver,
    group_id: str,
    walk: _Walk,
    found: _Found,
    now: str,
    result: ForgetResult,
) -> _Frontier:
    """Retract ``found``'s live facts and hide its episodes, each marker
    before any clean-up; the next frontier: every fact it reached, retracted
    or passed through, and every episode it hid."""
    targets = [
        {"uuid": row["uuid"], "reason": derived_reason(walk.root(row["via"]))}
        for row in found.facts
        if row["live"]
    ]
    landed = await _retract(driver, group_id, targets, now)
    retracted = walk.reach([row for row in found.facts if row["uuid"] in landed])
    passed = walk.reach([row for row in found.facts if row["uuid"] not in landed])
    result.derived.extend(retracted)
    tainted = walk.reach(found.episodes)
    mentioned: list[str] = []
    if tainted:
        rows = _rows(
            await driver.execute_query(REDACT_DERIVED_QUERY, uuids=tainted, now=now)
        )
        _note_redacted(result, tainted)
        mentioned = rows[0]["mentioned"] if rows else []
    walk.budget -= len(retracted) + len(passed) + len(tainted)
    if mentioned:
        await scrub_entities(driver, [], mentioned)
    if retracted:
        await scrub(driver, retracted)
    reached = [*retracted, *passed]
    hidden = await _redact_citing(driver, walk, reached, now, result)
    return _Frontier(facts=reached, episodes=[*tainted, *hidden])


async def _retract(
    driver: GraphDriver, group_id: str, targets: list[dict[str, str]], now: str
) -> set[str]:
    """Retract each target still live, as a soft forget does, with its own
    reason; the uuids retracted."""
    if not targets:
        return set()
    rows = _rows(
        await driver.execute_query(
            RETRACT_QUERY,
            targets=targets,
            group_id=group_id,
            now=now,
            status=MemoryStatus.retracted.value,
        )
    )
    return {row["uuid"] for row in rows}


async def _redact_citing(
    driver: GraphDriver,
    walk: _Walk,
    facts: list[str],
    now: str,
    result: ForgetResult,
) -> list[str]:
    """Redact every episode citing one of ``facts`` that the recall policy
    hides (all of them, for a fact forgotten now), as a forget does; the
    episodes the walk had not reached."""
    if not facts:
        return []
    rows = _rows(
        await driver.execute_query(REDACT_EPISODES_QUERY, uuids=facts, now=now)
    )
    _note_redacted(result, [row["uuid"] for row in rows])
    return walk.reach(rows)


def _note_redacted(result: ForgetResult, episodes: list[str]) -> None:
    result.redacted_episodes = list(
        dict.fromkeys([*result.redacted_episodes, *episodes])
    )


def _rows(result: Any) -> list[dict[str, Any]]:
    return result[0] if result else []
