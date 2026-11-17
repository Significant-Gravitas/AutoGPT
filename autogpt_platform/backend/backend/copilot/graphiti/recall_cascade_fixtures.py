"""A bakery's memory and what three dream passes derived from it, for the
live cascade tests (``recall_cascade*_integration_test.py``).

The user's facts go in through graphiti's ``add_episode``, the dream's
through ``dream/apply.py`` and the production ingestion worker, each dream
write extracting the fact its test scripts: so each derived fact carries the
record ingestion writes in production (``recall_derivation.py``). Not
collected by pytest.
"""

import ast
import re
from contextlib import AbstractContextManager, nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from graphiti_core import Graphiti
from pydantic import BaseModel

from backend.copilot.dream import apply, fetch
from backend.copilot.dream.citations import fact_scopes
from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.schemas import (
    ConsolidatedFact,
    DreamOperations,
    ProposedFinding,
)

from . import ingest
from .falkordb_driver import AutoGPTFalkorDriver
from .recall_integration_fixtures import (
    BuildClient,
    Fact,
    ingest_facts,
    model_of,
    rows,
    scripted_responses,
)
from .scope import MemoryScope

FLOUR: Fact = (
    "Sunrise Bakery",
    "Hill Country Mills",
    "Sunrise Bakery uses Hill Country Mills as its flour supplier",
)
# A fact the user stated about the same bakery, and one about something else.
BOULE: Fact = (
    "Sunrise Bakery",
    "Sourdough Boule",
    "Sunrise Bakery bakes a sourdough boule",
)
CAFE: Fact = ("Maria", "Oak Street Cafe", "Maria runs the Oak Street Cafe")
# What the dream derives: a consolidation of FLOUR, a proposal from it and
# BOULE, and a proposal from the consolidation's own dream text.
SUPPLIES: Fact = (
    "Hill Country Mills",
    "Sunrise Bakery",
    "Hill Country Mills supplies the flour Sunrise Bakery bakes with",
)
BOULE_FLOUR: Fact = (
    "Hill Country Mills",
    "Sourdough Boule",
    "Hill Country Mills supplies the flour for the sourdough boule",
)
WEEKLY: Fact = (
    "Sunrise Bakery",
    "Hill Country Mills",
    "Sunrise Bakery orders flour weekly from Hill Country Mills",
)

_SECTION = re.compile(r"<EXISTING FACTS>\s*(.*?)\s*</EXISTING FACTS>", re.DOTALL)


class Bakery(BaseModel):
    """The user's facts and the chat turn that named the flour supplier;
    each dream fact and the dream episode that wrote it."""

    flour: str
    boule: str
    cafe: str
    said: str
    supplies: str
    supplies_episode: str
    boule_flour: str
    boule_flour_episode: str
    weekly: str
    weekly_episode: str

    def derived(self) -> list[str]:
        return sorted([self.supplies, self.boule_flour, self.weekly])


async def build_bakery(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient
) -> Bakery:
    """The user's three facts, then three dream passes, each reading the
    graph first: ``supplies`` consolidates FLOUR (citing its fact and chat
    turn); ``boule_flour`` rests on ``supplies`` and BOULE; ``weekly`` rests
    on the dream episode that wrote ``supplies``."""
    said, flour = await ingest_facts(driver, scope, build, [FLOUR], session_id="s-1")
    _, boule = await ingest_facts(driver, scope, build, [BOULE], session_id="s-2")
    _, cafe = await ingest_facts(driver, scope, build, [CAFE], session_id="s-3")
    consolidation = ConsolidatedFact(
        content=SUPPLIES[2],
        confidence=0.9,
        source_fact_uuids=[flour[FLOUR[2]]],
        source_episode_uuids=[said],
    )
    await dream_once(driver, scope, build, consolidation, SUPPLIES, "pass-1")
    supplies = await fact_uuid(driver, SUPPLIES[2])
    supplies_episode = await dream_episode(driver, "pass-1")
    proposal = _proposal(BOULE_FLOUR[2], facts=[supplies, boule[BOULE[2]]])
    await dream_once(driver, scope, build, proposal, BOULE_FLOUR, "pass-2")
    weekly = _proposal(WEEKLY[2], episodes=[supplies_episode])
    await dream_once(driver, scope, build, weekly, WEEKLY, "pass-3")
    return Bakery(
        flour=flour[FLOUR[2]],
        boule=boule[BOULE[2]],
        cafe=cafe[CAFE[2]],
        said=said,
        supplies=supplies,
        supplies_episode=supplies_episode,
        boule_flour=await fact_uuid(driver, BOULE_FLOUR[2]),
        boule_flour_episode=await dream_episode(driver, "pass-2"),
        weekly=await fact_uuid(driver, WEEKLY[2]),
        weekly_episode=await dream_episode(driver, "pass-3"),
    )


def _proposal(
    content: str, *, facts: list[str] | None = None, episodes: list[str] | None = None
) -> ProposedFinding:
    return ProposedFinding(
        content=content,
        confidence=0.6,
        rationale="the bakery's flour",
        source_fact_uuids=facts or [],
        source_episode_uuids=episodes or [],
    )


async def dream_once(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    build: BuildClient,
    write: ConsolidatedFact | ProposedFinding,
    extracted: Fact,
    pass_id: str,
    *,
    merge_into: str | None = None,
) -> dict[str, Any]:
    """One dream pass that reads the graph, then writes ``write``, graphiti
    extracting ``extracted`` from it (and, with ``merge_into``, its model
    naming the fact of that sentence a duplicate); its stats, after
    checking it wrote."""
    read = await gather(scope)
    ops = (
        DreamOperations(writes=[write])
        if isinstance(write, ConsolidatedFact)
        else DreamOperations(proposals=[write])
    )
    stats = await dream(driver, scope, build, ops, read, extracted, pass_id, merge_into)
    written = stats["consolidated_count"] + stats["proposal_count"]
    assert (written, stats["dropped_forgotten"]) == (1, 0), f"{pass_id}: {stats}"
    return stats


async def dream(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    build: BuildClient,
    ops: DreamOperations,
    read: DreamInput,
    extracted: Fact,
    pass_id: str,
    merge_into: str | None = None,
) -> dict[str, Any]:
    """``ops`` through ``apply_operations`` and the worker, as a pass that
    read ``read``; its stats."""
    client = build(driver, scripted_responses([extracted]))
    with (
        patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)),
        _merging(client, merge_into),
    ):
        return await apply.apply_operations(
            scope,
            pass_id,
            ops,
            known_fact_uuids=read.known_fact_uuids,
            known_episode_uuids=read.known_episode_uuids,
            fact_scopes=fact_scopes(read),
            ingestion_drain_timeout=25,
        )


def _merging(client: Graphiti, sentence: str | None) -> AbstractContextManager:
    """While it is open, graphiti's model names the existing fact
    ``sentence`` a duplicate of whatever a write states, as it would a
    paraphrase; nothing changes when ``sentence`` is None."""
    if sentence is None:
        return nullcontext()
    model = model_of(client)
    answer = model._generate_response

    async def resolve(messages, response_model=None, *args: Any, **kwargs: Any):
        if response_model is not None and response_model.__name__ == "EdgeDuplicate":
            found = _SECTION.search(messages[-1].content)
            existing = ast.literal_eval(found.group(1)) if found else []
            same = [c["idx"] for c in existing if c["fact"] == sentence]
            return {"duplicate_facts": same[:1], "contradicted_facts": []}
        return await answer(messages, response_model, *args, **kwargs)

    return patch.object(model, "_generate_response", side_effect=resolve)


async def gather(scope: MemoryScope) -> DreamInput:
    """What a dream pass reads now (no chat sessions)."""
    sessions = SimpleNamespace(get_user_chat_sessions=AsyncMock(return_value=[]))
    with patch.object(fetch, "chat_db", return_value=sessions):
        return await fetch.gather_dream_input(scope)


async def fact_uuid(driver: AutoGPTFalkorDriver, sentence: str) -> str:
    """The one fact whose sentence (or audit copy) is ``sentence``."""
    [row] = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO]->() "
        "WHERE coalesce(e.fact_redacted, e.fact) = $sentence "
        "RETURN e.uuid AS uuid",
        sentence=sentence,
    )
    return row["uuid"]


async def dream_episode(driver: AutoGPTFalkorDriver, pass_id: str) -> str:
    [row] = await rows(
        driver,
        "MATCH (ep:Episodic) WHERE ep.name STARTS WITH $prefix "
        "RETURN ep.uuid AS uuid",
        prefix=f"dream_{pass_id}_",
    )
    return row["uuid"]


async def derivation(driver: AutoGPTFalkorDriver, uuid: str) -> dict[str, Any]:
    """What fact or episode ``uuid`` records it was derived from, its
    sources and, for a fact, the reason it was retired."""
    found = await rows(
        driver,
        """
        OPTIONAL MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
        OPTIONAL MATCH (ep:Episodic {uuid: $uuid})
        RETURN coalesce(e.derived_from_facts, ep.derived_from_facts) AS facts,
               coalesce(e.derived_from_episodes, ep.derived_from_episodes)
                   AS episodes,
               e.episodes AS sources, e.expiration_reason AS reason,
               e.status AS status, ep.redacted_at AS redacted_at
        """,
        uuid=uuid,
    )
    return found[0]
