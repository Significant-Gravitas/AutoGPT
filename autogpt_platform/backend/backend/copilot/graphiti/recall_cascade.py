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
``derived_from_forgotten:<uuid>``, naming the root it descends from: the
fact the user forgot (``recall_cascade_walk.py``). A dream episode reached
without a live fact of its own (its fact was superseded, merged into a fact
a user stated, or never extracted) is redacted and the entities it mentions
blanked, since the dream reads its text. A hard forget cascades the same
way, softly: retraction takes the derived facts out of every read, and
deleting them would purge what the model's citations name, which can be
more than a fact truly rests on, beyond undoing. It also erases the text it
reaches (``recall_erase.py``): every derived fact's, retracted or walked
through, and every hidden dream episode's.

At most ``CASCADE_MAX_ROUNDS`` rounds and ``CASCADE_MAX_ITEMS`` derived facts
and dream episodes per forget. A forget stopped by a bound or a failed step
reports a ``cleanup_error`` on each root, and forgetting it again picks up
the facts already retracted for it (their reason names it) and goes on from
there, erasing when the root is gone: a hard forget purges its facts even
when its cascade stops short, and a repeated forget of one of them starts
from the records that name it (``recall_forget.py``). The episodes a forget
hid remember which forgotten facts they were hidden for (``redacted_for``,
``recall_hide.py``), so a cascade resumed after the purge still starts from
the root's own episodes. The derivation backfill starts one from hidden
episodes too (``seeds``).
"""

import logging
from collections.abc import Mapping
from typing import Any

from graphiti_core.driver.driver import GraphDriver

from .memory_model import ForgetResult, MemoryForgetFailure, MemoryStatus
from .recall_cascade_queries import (
    DERIVED_EPISODES_QUERY,
    DERIVED_FACTS_QUERY,
    EARLIER_QUERY,
    REDACT_DERIVED_QUERY,
    RETRACT_QUERY,
)
from .recall_cascade_walk import (
    DERIVED_FROM_FORGOTTEN,
    Found,
    Frontier,
    Walk,
    derived_reason,
)
from .recall_erase import erase as erase_text
from .recall_hide import REDACT_EPISODES_QUERY, scrub, scrub_entities

logger = logging.getLogger(__name__)

CASCADE_MAX_ITEMS = 500
CASCADE_MAX_ROUNDS = 10


async def cascade(
    driver: GraphDriver,
    group_id: str,
    roots: list[str],
    now: str,
    result: ForgetResult,
    *,
    erase: bool = False,
    seeds: Mapping[str, str] | None = None,
    named: Mapping[str, str] | None = None,
) -> None:
    """Retract what the dream derived from ``roots``, the facts this forget
    has just retracted and hidden, or that are gone: ``result.derived`` gets
    each fact it retracts, ``result.passed`` each derived fact it only walks
    through and ``result.redacted_episodes`` each episode it hides; with
    ``erase`` (a hard forget, or a root that is gone), their text goes too.
    ``seeds`` are hidden episodes to start from as well, each mapped to the
    root it names, and ``named`` maps a root that is not itself the fact the
    user forgot (a derived fact a cascade retracted) to the one it names. A
    failed step or a bound reached is a failure on every root."""
    walk = Walk.start(
        roots,
        dict(seeds or {}),
        named=dict(named or {}),
        budget=CASCADE_MAX_ITEMS,
        erase=erase,
    )
    if not walk.names:
        return
    try:
        finished = await _run(driver, group_id, walk, now, result)
    except Exception as exc:
        logger.warning(f"Forget cascade failed in graph {group_id[:20]}", exc_info=True)
        result.failures.extend(
            MemoryForgetFailure.cleanup_error(name, exc) for name in walk.names
        )
        return
    if not finished:
        logger.warning(
            f"Forget cascade in graph {group_id[:20]} stopped at its bound "
            f"after {len(result.derived)} derived facts"
        )
        result.failures.extend(
            MemoryForgetFailure.derived_left(name) for name in walk.names
        )


async def _run(
    driver: GraphDriver,
    group_id: str,
    walk: Walk,
    now: str,
    result: ForgetResult,
) -> bool:
    """Rounds until one finds nothing (True) or a bound stops them (False).
    A round the budget cut short, to nothing at all once it is spent, still
    left something behind."""
    frontier = await _resume(driver, walk, now, result)
    for _ in range(CASCADE_MAX_ROUNDS):
        found = await _derived(driver, walk, frontier)
        if not (found.facts or found.episodes):
            return not found.truncated
        frontier = await _retire(driver, group_id, walk, found, now, result)
        if found.truncated:
            return False
    last = await _derived(driver, walk, frontier)
    return not (last.facts or last.episodes or last.truncated)


async def _resume(
    driver: GraphDriver, walk: Walk, now: str, result: ForgetResult
) -> Frontier:
    """The first frontier: the roots, the facts an earlier try retracted for
    them (their clean-up finished again), every episode citing one or hidden
    for one, and the seeds."""
    reasons = [derived_reason(name) for name in walk.names]
    earlier = _rows(await driver.execute_query(EARLIER_QUERY, reasons=reasons))
    for row in earlier:
        walk.root_of[row["uuid"]] = row["reason"].removeprefix(
            f"{DERIVED_FROM_FORGOTTEN}:"
        )
    again = [row["uuid"] for row in earlier]
    if again:
        await scrub(driver, again)
    facts = [*walk.roots, *again]
    hidden = [*walk.seeds, *await _redact_citing(driver, walk, facts, now, result)]
    if walk.erase:
        await erase_text(driver, again, hidden)
    return Frontier(facts=facts, episodes=hidden)


async def _derived(driver: GraphDriver, walk: Walk, frontier: Frontier) -> Found:
    """The facts and the dream episodes not reached yet whose record names
    something in ``frontier``, no more than the walk's budget."""
    if not (frontier.facts or frontier.episodes):
        return Found()
    params = {
        "facts": frontier.facts,
        "episodes": frontier.episodes,
        "seen": list(walk.root_of),
        "limit": walk.budget + 1,
    }
    facts = _rows(await driver.execute_query(DERIVED_FACTS_QUERY, **params))
    episodes = _rows(await driver.execute_query(DERIVED_EPISODES_QUERY, **params))
    if len(facts) + len(episodes) <= walk.budget:
        return Found(facts=facts, episodes=episodes)
    kept = facts[: walk.budget]
    rest = episodes[: walk.budget - len(kept)]
    return Found(facts=kept, episodes=rest, truncated=True)


async def _retire(
    driver: GraphDriver,
    group_id: str,
    walk: Walk,
    found: Found,
    now: str,
    result: ForgetResult,
) -> Frontier:
    """Retract ``found``'s live facts and hide its episodes, each marker
    before any clean-up (erasing their text, and that of the facts passed
    through, on a hard forget); the next frontier: every fact it reached,
    retracted or passed through, and every episode it hid."""
    targets = [
        {"uuid": row["uuid"], "reason": derived_reason(walk.root(row["via"]))}
        for row in found.facts
        if row["live"]
    ]
    landed = await _retract(driver, group_id, targets, now)
    retracted = walk.reach([row for row in found.facts if row["uuid"] in landed])
    passed = walk.reach([row for row in found.facts if row["uuid"] not in landed])
    result.derived.extend(retracted)
    result.passed.extend(passed)
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
    reached = [*retracted, *passed]
    scrubbed = reached if walk.erase else retracted
    if scrubbed:
        await scrub(driver, scrubbed)
    hidden = await _redact_citing(driver, walk, reached, now, result)
    if walk.erase:
        await erase_text(driver, reached, [*tainted, *hidden])
    return Frontier(facts=reached, episodes=[*tainted, *hidden])


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
    walk: Walk,
    facts: list[str],
    now: str,
    result: ForgetResult,
) -> list[str]:
    """Redact every episode citing one of ``facts``, or hidden for one, that
    the recall policy hides (all of them, for a fact forgotten now), as a
    forget does; the episodes the walk had not reached."""
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
