"""A small in-memory graph that answers a forget's cascade as FalkorDB would,
for ``recall_cascade_test.py``: each of the cascade's queries (by identity)
and the forget's own scrub and redaction, over facts and episodes that carry
a dream's record or not. Not collected by pytest.
"""

from typing import Any

from pydantic import BaseModel, Field

from . import recall_cascade, recall_hide


class FakeFact(BaseModel):
    """A ``RELATES_TO`` fact: its record (``facts``, ``episodes``), the
    episodes that state it (``sources``), whether it is live, and its
    ``expiration_reason``."""

    facts: list[str] = Field(default_factory=list)
    episodes: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    live: bool = True
    reason: str | None = None


class FakeEpisode(BaseModel):
    """An episode: the facts it cites (``entity_edges``), a dream's record
    (``facts`` is None on a user's episode), and whether it is redacted."""

    cites: list[str] = Field(default_factory=list)
    facts: list[str] | None = None
    episodes: list[str] = Field(default_factory=list)
    redacted: bool = False


class CascadeGraph:
    """What the cascade's queries see and change, by query."""

    def __init__(
        self, facts: dict[str, FakeFact], episodes: dict[str, FakeEpisode]
    ) -> None:
        self.facts = facts
        self.episodes = episodes
        self.queries: list[str] = []
        self.scrubbed: list[str] = []
        # A query that raises, and facts the dream demotes between the
        # cascade's read and its retraction.
        self.fail_on: str | None = None
        self.demoted_meanwhile: set[str] = set()

    async def execute_query(
        self, query: str, **params: Any
    ) -> tuple[list[dict[str, Any]], list[str], None]:
        self.queries.append(query)
        if query == self.fail_on:
            raise RuntimeError("down")
        return self._answer(query, params), [], None

    def _answer(self, query: str, p: dict[str, Any]) -> list[dict[str, Any]]:
        if query == recall_cascade.EARLIER_QUERY:
            return [
                {"uuid": uuid, "reason": fact.reason}
                for uuid, fact in self.facts.items()
                if fact.reason in p["reasons"]
            ]
        if query == recall_hide.REDACT_EPISODES_QUERY:
            return self._redact_citing(p["uuids"])
        if query == recall_cascade.DERIVED_FACTS_QUERY:
            return self._derived_facts(p)[: p["limit"]]
        if query == recall_cascade.DERIVED_EPISODES_QUERY:
            return self._derived_episodes(p)[: p["limit"]]
        if query == recall_cascade.RETRACT_QUERY:
            return self._retract(p["targets"])
        if query == recall_cascade.REDACT_DERIVED_QUERY:
            for uuid in p["uuids"]:
                self.episodes[uuid].redacted = True
            return [{"mentioned": []}]
        if query == recall_hide.SCRUB_FACTS_QUERY:
            self.scrubbed.extend(p["uuids"])
            return [{"ends": []}]
        return []

    def _redact_citing(self, uuids: list[str]) -> list[dict[str, Any]]:
        rows = []
        for uuid, episode in sorted(self.episodes.items()):
            via = [x for x in episode.cites if x in uuids]
            if via:
                episode.redacted = True
                rows.append({"uuid": uuid, "via": via})
        return rows

    def _derived_facts(self, p: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for uuid, fact in sorted(self.facts.items()):
            via = [x for x in fact.facts if x in p["facts"]]
            via += [x for x in fact.episodes if x in p["episodes"]]
            stated = [s for s in fact.sources if self.episodes[s].facts is None]
            if via and not stated and uuid not in p["seen"]:
                rows.append({"uuid": uuid, "via": via, "live": fact.live})
        return rows

    def _derived_episodes(self, p: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for uuid, episode in sorted(self.episodes.items()):
            via = [x for x in episode.facts or [] if x in p["facts"]]
            via += [x for x in episode.episodes if x in p["episodes"]]
            if via and uuid not in p["seen"]:
                rows.append({"uuid": uuid, "via": via})
        return rows

    def _retract(self, targets: list[dict[str, str]]) -> list[dict[str, Any]]:
        for uuid in self.demoted_meanwhile:
            self.facts[uuid].live = False
        rows = []
        for target in targets:
            fact = self.facts[target["uuid"]]
            if fact.live:
                fact.live, fact.reason = False, target["reason"]
                rows.append({"uuid": target["uuid"]})
        return rows


def chain() -> CascadeGraph:
    """User fact ``f`` (said in ``e0``); a consolidation ``c`` citing it (its
    dream episode ``dc``); a proposal ``p`` citing ``c`` (``dp``); a
    proposal ``q`` citing the dream episode ``dp``; and ``u``, a user fact
    that shares ``e0`` but was derived from nothing."""
    return CascadeGraph(
        facts={
            "f": FakeFact(live=False, reason="user_signal", sources=["e0"]),
            "u": FakeFact(sources=["e0"]),
            "c": FakeFact(facts=["f"], episodes=["e0"], sources=["dc"]),
            "p": FakeFact(facts=["c"], sources=["dp"]),
            "q": FakeFact(episodes=["dp"], sources=["dq"]),
        },
        episodes={
            "e0": FakeEpisode(cites=["f", "u"]),
            "dc": FakeEpisode(cites=["c"], facts=["f"], episodes=["e0"]),
            "dp": FakeEpisode(cites=["p"], facts=["c"]),
            "dq": FakeEpisode(cites=["q"], facts=[], episodes=["dp"]),
        },
    )
