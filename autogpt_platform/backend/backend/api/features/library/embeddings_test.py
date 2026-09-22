"""Tests for the library-agent embedding scheduler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.library.embeddings import (
    _build_searchable_text,
    _run_embedding,
    schedule_library_agent_embedding,
)


def _mock_graph(name="A", description="B", instructions="C"):
    g = MagicMock()
    g.name = name
    g.description = description
    g.instructions = instructions
    return g


def test_build_searchable_text_concatenates_present_fields():
    text = _build_searchable_text(_mock_graph("Email Bot", "Sends emails", ""))
    assert "Email Bot" in text
    assert "Sends emails" in text


def test_build_searchable_text_skips_empty_fields():
    text = _build_searchable_text(_mock_graph("", "", ""))
    assert text == ""


@pytest.mark.asyncio
async def test_run_embedding_skips_when_text_is_empty():
    """No call to ensure_content_embedding when there's nothing to embed."""
    with patch(
        "backend.api.features.library.embeddings.ensure_content_embedding",
        new=AsyncMock(return_value=True),
    ) as mock_ensure:
        await _run_embedding("la-1", "user-1", _mock_graph("", "", ""))
    mock_ensure.assert_not_called()


@pytest.mark.asyncio
async def test_run_embedding_forwards_user_scope_and_force_true():
    with (
        patch(
            "backend.api.features.library.embeddings.get_content_embedding",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "backend.api.features.library.embeddings.ensure_content_embedding",
            new=AsyncMock(return_value=True),
        ) as mock_ensure,
    ):
        await _run_embedding("la-1", "user-1", _mock_graph("X", "Y", ""))
    kwargs = mock_ensure.call_args.kwargs
    assert kwargs["content_id"] == "la-1"
    assert kwargs["user_id"] == "user-1"
    assert kwargs["force"] is True


@pytest.mark.asyncio
async def test_run_embedding_skips_when_searchable_text_unchanged():
    """If the existing embedding row already carries the same text, skip
    the OpenAI call — covers settings-only updates that bumped the graph
    version but didn't touch name/description/instructions."""
    graph = _mock_graph("X", "Y", "Z")
    existing = {"searchableText": "X Y Z"}
    with (
        patch(
            "backend.api.features.library.embeddings.get_content_embedding",
            new=AsyncMock(return_value=existing),
        ),
        patch(
            "backend.api.features.library.embeddings.ensure_content_embedding",
            new=AsyncMock(return_value=True),
        ) as mock_ensure,
    ):
        await _run_embedding("la-1", "user-1", graph)
    mock_ensure.assert_not_called()


@pytest.mark.asyncio
async def test_run_embedding_refreshes_when_text_changed():
    """When the existing embedding's text differs, re-embed."""
    graph = _mock_graph("X", "Y", "Z")
    existing = {"searchableText": "old text"}
    with (
        patch(
            "backend.api.features.library.embeddings.get_content_embedding",
            new=AsyncMock(return_value=existing),
        ),
        patch(
            "backend.api.features.library.embeddings.ensure_content_embedding",
            new=AsyncMock(return_value=True),
        ) as mock_ensure,
    ):
        await _run_embedding("la-1", "user-1", graph)
    mock_ensure.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_embedding_swallows_failures():
    """Failure to embed must not propagate — it's a best-effort background task."""
    with (
        patch(
            "backend.api.features.library.embeddings.get_content_embedding",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "backend.api.features.library.embeddings.ensure_content_embedding",
            new=AsyncMock(side_effect=RuntimeError("openai down")),
        ),
    ):
        # Must not raise.
        await _run_embedding("la-1", "user-1", _mock_graph())


@pytest.mark.asyncio
async def test_schedule_returns_task_and_runs_in_background():
    with (
        patch(
            "backend.api.features.library.embeddings.get_content_embedding",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "backend.api.features.library.embeddings.ensure_content_embedding",
            new=AsyncMock(return_value=True),
        ) as mock_ensure,
    ):
        task = schedule_library_agent_embedding(
            "la-1", "user-1", _mock_graph("X", "Y", "")
        )
        assert isinstance(task, asyncio.Task)
        await task
    mock_ensure.assert_awaited_once()
