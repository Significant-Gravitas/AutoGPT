"""Unit tests for ``fact_attributes``: graphiti's two saves of a fact run as
merges and nothing else is rewritten, checked against the queries graphiti
itself generates, and an attribute prompt reads as a stored fact's unless it
shows empty attributes.

The live runs are ``fact_attributes_integration_test.py`` and
``ingest_restatement_integration_test.py``.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.edges.edge_db_queries import (
    get_community_edge_save_query,
    get_entity_edge_save_bulk_query,
    get_entity_edge_save_query,
    get_episodic_edge_save_bulk_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    get_community_node_save_query,
    get_entity_node_save_query,
    get_episode_node_save_bulk_query,
    get_episode_node_save_query,
)
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.models import Message

from . import fact_attributes
from .fact_attributes import (
    MERGING_SAVES,
    REPLACING_CLAUSES,
    EdgeMergingSession,
    merging_save,
    stored_fact,
)
from .falkordb_driver import AutoGPTFalkorDriver

_FALKOR = GraphProvider.FALKORDB
_SAVE = get_entity_edge_save_query(_FALKOR)
_BULK_SAVE = get_entity_edge_save_bulk_query(_FALKOR)


@pytest.fixture(autouse=True)
def reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with no unreadable prompt logged yet."""
    monkeypatch.setattr(fact_attributes, "_REPORTED", set())


def _attribute_prompt(existing: dict[str, Any]) -> list[Message]:
    """graphiti's attribute prompt for a fact holding ``existing``."""
    return prompt_library.extract_edges.extract_attributes(
        {
            "fact": "Alice works on Atlas",
            "reference_time": None,
            "existing_attributes": existing,
        }
    )


def test_graphitis_fact_saves_each_replace_the_edge_with_one_clause() -> None:
    """The shape the rewrite depends on: a graphiti upgrade that changes either
    save fails here instead of silently saving as a replace again."""
    assert set(REPLACING_CLAUSES) == {_SAVE, _BULK_SAVE}
    for query, clause in REPLACING_CLAUSES.items():
        assert query.count(clause) == 1, clause


def test_both_saves_run_as_merges() -> None:
    assert merging_save(_SAVE) == _SAVE.replace(
        "SET e = $edge_data", "SET e += $edge_data"
    )
    assert merging_save(_BULK_SAVE) == _BULK_SAVE.replace(
        "SET r = edge", "SET r += edge"
    )
    for save, merge in MERGING_SAVES.items():
        assert REPLACING_CLAUSES[save] not in merge


@pytest.mark.parametrize(
    "query",
    [
        get_entity_node_save_query(_FALKOR, "Entity"),
        get_episode_node_save_query(_FALKOR),
        get_episode_node_save_bulk_query(_FALKOR),
        get_community_node_save_query(_FALKOR),
        get_episodic_edge_save_bulk_query(_FALKOR),
        get_community_edge_save_query(_FALKOR),
        get_entity_edge_save_query(GraphProvider.NEO4J),
        _SAVE.strip(),
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e = $props",
    ],
)
def test_no_other_query_is_rewritten(query: str) -> None:
    assert merging_save(query) == query


@pytest.mark.asyncio
async def test_the_session_runs_graphitis_bulk_save_as_a_merge() -> None:
    graph = MagicMock()
    graph.query = AsyncMock()
    session = EdgeMergingSession(graph)
    labelled = [("MATCH (n) SET n:Person", {"uuid": "a"})]

    await session.run(_BULK_SAVE, entity_edges=[{"uuid": "x"}])
    await session.run("MATCH (n) RETURN n", uuid="a")
    await session.run(labelled)

    queries = [call.args[0] for call in graph.query.await_args_list]
    assert queries == [MERGING_SAVES[_BULK_SAVE], "MATCH (n) RETURN n", labelled[0][0]]
    assert graph.query.await_args_list[0].args[1] == {"entity_edges": [{"uuid": "x"}]}


def test_the_driver_opens_the_merging_session() -> None:
    driver = AutoGPTFalkorDriver(falkor_db=MagicMock())

    assert isinstance(driver.session(), EdgeMergingSession)


def test_a_new_facts_prompt_shows_no_stored_attributes() -> None:
    assert stored_fact(_attribute_prompt({})) is False


def test_a_stored_facts_prompt_shows_its_attributes() -> None:
    stored = {"status": "active", "recall_count": 3, "scope": "project:atlas"}

    assert stored_fact(_attribute_prompt(stored)) is True


@pytest.mark.parametrize(
    "edit",
    [
        lambda text: text.replace("EXISTING ATTRIBUTES", "STORED ATTRIBUTES"),
        lambda text: text.replace("{}", "{not json"),
        lambda text: text.replace("{}", "[]"),
        lambda text: text.replace(
            "<FACT>", "<EXISTING ATTRIBUTES>{}</EXISTING ATTRIBUTES>\n<FACT>"
        ),
    ],
    ids=["renamed", "unparseable", "not-an-object", "listed-twice"],
)
def test_a_prompt_it_cannot_read_counts_as_a_stored_facts(
    edit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A new fact then gets no attributes rather than a stored one new
    values, and graphiti's upgrade is logged once, not on every fact."""
    prompt = [
        m.model_copy(update={"content": edit(m.content)}) for m in _attribute_prompt({})
    ]

    assert stored_fact(prompt) is True
    assert stored_fact(prompt) is True
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert "no fact gets attributes from extraction" in errors[0].getMessage()
