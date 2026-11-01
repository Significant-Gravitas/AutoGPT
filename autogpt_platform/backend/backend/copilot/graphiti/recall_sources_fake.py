"""An in-memory answer to the source-state reads of ``recall_sources.py``,
for the unit tests of the walk up the records, the check before a dream
write and the settle after one lands. Not collected by pytest.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from . import recall_sources


def fact(
    *,
    forgotten: bool = False,
    hard: bool = False,
    reason: str | None = None,
    live: bool = False,
    derived: bool = True,
    facts: tuple[str, ...] = (),
    episodes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """A fact's ``FACT_STATES_QUERY`` row, without its uuid: a derived fact
    no longer live unless told otherwise, naming ``facts`` and
    ``episodes``."""
    return {
        "forgotten": forgotten,
        "hard": hard,
        "reason": reason,
        "live": live,
        "derived": derived,
        "facts": list(facts),
        "episodes": list(episodes),
    }


def episode(
    *, hidden: bool = False, hard: bool = False, hidden_for: tuple[str, ...] = ()
) -> dict[str, Any]:
    """An episode's ``EPISODE_STATES_QUERY`` row, without its uuid."""
    return {"hidden": hidden, "hard": hard, "hidden_for": list(hidden_for)}


def sources(
    facts: dict[str, dict[str, Any]], episodes: dict[str, dict[str, Any]] | None = None
) -> MagicMock:
    """A driver answering the two state reads from ``facts`` and
    ``episodes`` by uuid (a uuid in neither is gone), and every other query
    with nothing."""
    known = episodes or {}

    async def answer(query: str, **params: Any) -> tuple[list, list, None]:
        if query == recall_sources.FACT_STATES_QUERY:
            rows = [{"uuid": u, **facts[u]} for u in params["uuids"] if u in facts]
        elif query == recall_sources.EPISODE_STATES_QUERY:
            rows = [{"uuid": u, **known[u]} for u in params["uuids"] if u in known]
        else:
            rows = []
        return rows, [], None

    driver = MagicMock()
    driver.execute_query = AsyncMock(side_effect=answer)
    return driver


def reads(driver: MagicMock, query: str) -> list[list[str]]:
    """The uuids each read of ``query`` asked for, in order."""
    return [
        call.kwargs["uuids"]
        for call in driver.execute_query.await_args_list
        if call.args[0] == query
    ]
