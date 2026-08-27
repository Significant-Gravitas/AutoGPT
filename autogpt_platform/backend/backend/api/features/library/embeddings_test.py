import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.library.embeddings import (
    _build_searchable_text,
    _run_embedding,
    schedule_library_agent_embedding,
)


def _mock_graph(name="A", description="B", instructions="C"):
    graph = MagicMock()
    graph.id = "graph-1"
    graph.version = 7
    graph.name = name
    graph.description = description
    graph.instructions = instructions
    return graph


def test_build_searchable_text_concatenates_present_fields():
    text = _build_searchable_text(_mock_graph("Email Bot", "Sends emails", ""))
    assert "Email Bot" in text
    assert "Sends emails" in text


def test_build_searchable_text_skips_empty_fields():
    assert _build_searchable_text(_mock_graph("", "", "")) == ""


@pytest.mark.asyncio
async def test_run_embedding_skips_when_text_is_empty():
    with patch(
        "backend.api.features.library.embeddings.ensure_live_library_content_embedding",
        new=AsyncMock(return_value=True),
    ) as ensure:
        await _run_embedding(
            "la-1", "user-1", _mock_graph("", "", ""), "org-1", "team-1"
        )
    ensure.assert_not_called()


@pytest.mark.asyncio
async def test_run_embedding_forwards_exact_resource_scope():
    with patch(
        "backend.api.features.library.embeddings.ensure_live_library_content_embedding",
        new=AsyncMock(return_value=True),
    ) as ensure:
        await _run_embedding(
            "la-1", "user-1", _mock_graph("X", "Y", ""), "org-1", "team-1"
        )

    ensure.assert_awaited_once_with(
        content_id="la-1",
        user_id="user-1",
        organization_id="org-1",
        team_id="team-1",
        source_graph_id="graph-1",
        source_graph_version=7,
        searchable_text="X Y",
        metadata={"name": "X"},
    )


@pytest.mark.asyncio
async def test_run_embedding_swallows_failures():
    with patch(
        "backend.api.features.library.embeddings.ensure_live_library_content_embedding",
        new=AsyncMock(side_effect=RuntimeError("openai down")),
    ):
        await _run_embedding("la-1", "user-1", _mock_graph(), "org-1", "team-1")


@pytest.mark.asyncio
async def test_schedule_returns_task_and_runs_in_scope_free_background():
    with patch(
        "backend.api.features.library.embeddings.ensure_live_library_content_embedding",
        new=AsyncMock(return_value=True),
    ) as ensure:
        task = schedule_library_agent_embedding(
            "la-1", "user-1", _mock_graph("X", "Y", ""), "org-1", "team-1"
        )
        assert isinstance(task, asyncio.Task)
        await task
    ensure.assert_awaited_once()
