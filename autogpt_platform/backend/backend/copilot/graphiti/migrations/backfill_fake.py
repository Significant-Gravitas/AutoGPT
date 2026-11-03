"""A graph as the derivation backfill's reads see it, recording its writes,
for ``backfill_derivations_test.py``, ``legacy_citations_test.py`` and
``backfill_cascade_test.py``. Not collected by pytest.
"""

from typing import Any
from unittest.mock import AsyncMock

from backend.copilot.graphiti.recall_reconcile import MARKERS_QUERY

from . import backfill_cascade
from . import backfill_derivations as backfill
from . import legacy_citations


class BackfillGraph:
    """A graph as the backfill's reads see it, recording its writes."""

    graph_name = "user_a"

    def __init__(
        self,
        episodes: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        forgotten: list[str] | None = None,
    ) -> None:
        self.answers = {
            backfill.DREAM_EPISODES_QUERY: episodes,
            backfill.UNSTAMPED_FACTS_QUERY: facts,
            backfill_cascade.FORGOTTEN_FACTS_QUERY: [
                {"uuid": u} for u in forgotten or []
            ],
            backfill_cascade.FACT_NAMES_QUERY: [],
            backfill_cascade.EPISODE_NAMES_QUERY: [],
        }
        self.writes: list[tuple[str, list[dict[str, Any]]]] = []
        # Whether it read the pending dream records (none here).
        self.reconciled = False
        # The sources the graph has, all unscoped, for the citations read
        # back from descriptions.
        self.sources = {"e0", "e9", "f1", "f2"}
        self.close = AsyncMock()

    async def execute_query(self, query: str, **params: Any):
        if query == MARKERS_QUERY:
            self.reconciled = True
            return [], [], None
        if query == backfill_cascade.MARKER_NAMES_QUERY:
            return [], [], None
        if query == legacy_citations.FACT_SCOPES_QUERY:
            found = [{"uuid": u, "scope": None} for u in params["uuids"]]
            return [row for row in found if row["uuid"] in self.sources], [], None
        if query == legacy_citations.CITED_EPISODES_QUERY:
            found = [{"uuid": u} for u in params["uuids"]]
            return [row for row in found if row["uuid"] in self.sources], [], None
        if query in self.answers:
            rows = [r for r in self.answers[query] if r["uuid"] > params["after"]]
            return rows[: params["limit"]], [], None
        self.writes.append((query, params["rows"]))
        return [], [], None


def dream_row(uuid: str, description: str | None, **record: list[str]) -> dict:
    return {
        "uuid": uuid,
        "description": description,
        "content": None,
        "facts": record.get("facts"),
        "episodes": record.get("episodes"),
    }


def older_dreams() -> BackfillGraph:
    """Two dream episodes written before records (a consolidation listing
    episodes, a proposal listing facts), one recorded since, one listing
    nothing; and the facts they and a user's chat turn produced."""
    return BackfillGraph(
        episodes=[
            dream_row("d1", "dream-pass consolidation; src_episodes=e0,e9"),
            dream_row("d2", "dream-pass proposal; rationale=r; src_facts=f1"),
            dream_row("d3", "dream-pass consolidation", facts=["f2"], episodes=[]),
            dream_row("d4", "dream-pass consolidation; src_episodes="),
        ],
        facts=[
            {"uuid": "c1", "episodes": ["d1"]},
            {"uuid": "c2", "episodes": ["d2", "d3"]},
            {"uuid": "c4", "episodes": ["d4"]},
            {"uuid": "merged", "episodes": ["d1", "chat"]},
            {"uuid": "user", "episodes": ["chat"]},
        ],
    )
