"""Pin graphiti-core's episode-provenance behavior on edges.

``_stamp_edge_metadata`` (``ingest.py``) relies on an upstream invariant
to select ONLY edges solely sourced by the dream episode it just wrote:

- a freshly EXTRACTED edge is created with ``episodes == [episode.uuid]``;
- when graphiti DEDUPES an extracted fact into a pre-existing edge, it
  APPENDS the new episode uuid to that edge's ``episodes`` (len >= 2).

The stamping unit tests hand-build edges, so they cannot detect upstream
drift. ``pyproject.toml`` pins ``graphiti-core = "^0.28.2"`` — minor/patch
bumps are allowed — so this file exercises the REAL graphiti-core
extraction/resolution code (only the LLM boundary is stubbed) and fails
loudly if a bump changes the invariant, instead of letting
``_stamp_edge_metadata`` silently mis-stamp user-authored edges.
"""

from datetime import datetime, timezone

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.prompts.models import Message
from graphiti_core.utils.maintenance.edge_operations import (
    extract_edges,
    resolve_extracted_edge,
)

GROUP_ID = "test_group"


class ScriptedLLMClient(LLMClient):
    """Real ``LLMClient`` whose responses are canned per response_model.

    Records every requested model so tests can assert which graphiti
    code path ran (e.g. the exact-match fast path must not call the LLM).
    """

    def __init__(self, responses: dict[str, dict] | None = None) -> None:
        super().__init__(config=None, cache=False)
        self.responses = responses or {}
        self.prompts_requested: list[str] = []

    async def _generate_response(
        self,
        messages: list[Message],
        response_model=None,
        max_tokens: int = 16384,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict:
        name = response_model.__name__ if response_model else "raw"
        self.prompts_requested.append(name)
        if name not in self.responses:
            raise AssertionError(f"Unexpected LLM call for {name}")
        return self.responses[name]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _entity(name: str) -> EntityNode:
    return EntityNode(
        name=name, group_id=GROUP_ID, labels=["Entity"], created_at=_now()
    )


def _episode(content: str) -> EpisodicNode:
    return EpisodicNode(
        name="test-episode",
        group_id=GROUP_ID,
        source=EpisodeType.json,
        source_description="test",
        content=content,
        valid_at=_now(),
        created_at=_now(),
    )


def _existing_edge(
    source: EntityNode, target: EntityNode, fact: str, episodes: list[str]
) -> EntityEdge:
    return EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name="works_on",
        group_id=GROUP_ID,
        fact=fact,
        episodes=episodes,
        created_at=_now(),
    )


@pytest.mark.asyncio
async def test_extracted_edge_is_sole_sourced_by_its_episode():
    """Creation leg: real ``extract_edges`` builds every new edge with
    ``episodes == [episode.uuid]`` — the exact predicate
    ``_stamp_edge_metadata`` uses to recognize edges this write created."""
    alice, atlas = _entity("Alice"), _entity("Atlas")
    episode = _episode('{"content": "Alice works on Atlas"}')
    llm = ScriptedLLMClient(
        {
            "ExtractedEdges": {
                "edges": [
                    {
                        "source_entity_name": "Alice",
                        "target_entity_name": "Atlas",
                        "relation_type": "works_on",
                        "fact": "Alice works on Atlas",
                        "valid_at": None,
                        "invalid_at": None,
                    }
                ]
            }
        }
    )
    clients = GraphitiClients.model_construct(llm_client=llm)

    edges = await extract_edges(
        clients,
        episode,
        [alice, atlas],
        previous_episodes=[],
        edge_type_map={},
        group_id=GROUP_ID,
    )

    assert len(edges) == 1
    assert list(edges[0].episodes) == [episode.uuid]


@pytest.mark.asyncio
async def test_exact_match_dedup_appends_episode_to_existing_edge():
    """Fast-path dedup (identical fact text): graphiti reuses the existing
    edge and APPENDS the new episode uuid — len >= 2, so the stamp
    selector must skip it. No LLM involved on this path."""
    alice, atlas = _entity("Alice"), _entity("Atlas")
    episode = _episode('{"content": "Alice works on Atlas"}')
    existing = _existing_edge(
        alice, atlas, "Alice works on Atlas", episodes=["older-episode"]
    )
    extracted = _existing_edge(
        alice, atlas, "Alice works on Atlas", episodes=[episode.uuid]
    )
    llm = ScriptedLLMClient()

    resolved, invalidated, duplicates = await resolve_extracted_edge(
        llm,
        extracted,
        related_edges=[existing],
        existing_edges=[],
        episode=episode,
    )

    assert resolved.uuid == existing.uuid
    assert list(resolved.episodes) == ["older-episode", episode.uuid]
    assert llm.prompts_requested == [], "fast path must not call the LLM"


@pytest.mark.asyncio
async def test_llm_dedup_appends_episode_to_existing_edge():
    """LLM-judged dedup (paraphrased fact): the resolved edge is the
    pre-existing one with the new episode uuid appended — len >= 2."""
    alice, atlas = _entity("Alice"), _entity("Atlas")
    episode = _episode('{"content": "Alice is a member of the Atlas project"}')
    existing = _existing_edge(
        alice, atlas, "Alice works on Atlas", episodes=["older-episode"]
    )
    extracted = _existing_edge(
        alice, atlas, "Alice is on the Atlas project", episodes=[episode.uuid]
    )
    llm = ScriptedLLMClient(
        {"EdgeDuplicate": {"duplicate_facts": [0], "contradicted_facts": []}}
    )

    resolved, invalidated, duplicates = await resolve_extracted_edge(
        llm,
        extracted,
        related_edges=[existing],
        existing_edges=[],
        episode=episode,
    )

    assert resolved.uuid == existing.uuid
    assert list(resolved.episodes) == ["older-episode", episode.uuid]
    assert duplicates == [existing]


@pytest.mark.asyncio
async def test_novel_edge_survives_resolution_sole_sourced():
    """When the LLM finds no duplicate among candidates, the extracted
    edge passes through resolution still sole-sourced by its episode —
    eligible for stamping."""
    alice, atlas = _entity("Alice"), _entity("Atlas")
    episode = _episode('{"content": "Alice leads the Atlas launch"}')
    existing = _existing_edge(
        alice, atlas, "Alice works on Atlas", episodes=["older-episode"]
    )
    extracted = _existing_edge(
        alice, atlas, "Alice leads the Atlas launch", episodes=[episode.uuid]
    )
    llm = ScriptedLLMClient(
        {"EdgeDuplicate": {"duplicate_facts": [], "contradicted_facts": []}}
    )

    resolved, invalidated, duplicates = await resolve_extracted_edge(
        llm,
        extracted,
        related_edges=[existing],
        existing_edges=[],
        episode=episode,
    )

    assert resolved.uuid == extracted.uuid
    assert list(resolved.episodes) == [episode.uuid]
    assert list(existing.episodes) == ["older-episode"], "existing edge untouched"
