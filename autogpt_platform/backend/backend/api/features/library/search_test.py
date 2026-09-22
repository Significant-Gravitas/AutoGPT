"""Tests for hybrid_search_library_agents."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.library.search import (
    _extract_lexical_keywords,
    hybrid_search_library_agents,
)


def _patch_search(return_value):
    """Patch the db_accessors.search() shim to return a mock module/client
    whose ``unified_hybrid_search`` returns ``return_value``."""
    mock_shim = MagicMock()
    mock_shim.unified_hybrid_search = AsyncMock(return_value=return_value)
    return (
        patch(
            "backend.api.features.library.search.search",
            return_value=mock_shim,
        ),
        mock_shim,
    )


@pytest.mark.asyncio
async def test_returns_empty_list_for_empty_query():
    """Empty/whitespace query short-circuits without calling the DB."""
    patcher, mock_shim = _patch_search(([], 0))
    with patcher:
        result = await hybrid_search_library_agents(query="   ", user_id="u1")
    assert result == []
    mock_shim.unified_hybrid_search.assert_not_called()


@pytest.mark.asyncio
async def test_delegates_to_unified_hybrid_search_with_user_scope():
    """Delegates to unified_hybrid_search with user_id + an unfiltered
    min_score=0 query (the wrapper does its own threshold filtering so
    sub-threshold scores can be logged for retuning)."""
    rows = [{"content_id": "lib-1", "combined_score": 0.82}]
    patcher, mock_shim = _patch_search((rows, 1))
    with patcher:
        result = await hybrid_search_library_agents(
            query="summarise my email", user_id="user-42", limit=3
        )

    assert result == rows
    mock_shim.unified_hybrid_search.assert_awaited_once()
    kwargs = mock_shim.unified_hybrid_search.call_args.kwargs
    assert kwargs["user_id"] == "user-42"
    assert kwargs["page_size"] == 3
    assert kwargs["min_score"] == 0.0
    assert len(kwargs["content_types"]) == 1


@pytest.mark.asyncio
async def test_filters_sub_threshold_results_after_querying():
    """Underlying query returns all candidates; wrapper filters to
    ``min_score`` (default LIBRARY_SIMILARITY_THRESHOLD) so the caller
    only sees the strong matches."""
    rows = [
        {"content_id": "strong", "combined_score": 0.80},
        {"content_id": "weak", "combined_score": 0.30},
    ]
    patcher, _ = _patch_search((rows, 2))
    with patcher:
        result = await hybrid_search_library_agents(
            query="summarise my email", user_id="user-42"
        )

    assert [r["content_id"] for r in result] == ["strong"]


@pytest.mark.asyncio
async def test_respects_explicit_min_score_override():
    """Caller-supplied ``min_score`` overrides the default threshold."""
    rows = [
        {"content_id": "a", "combined_score": 0.40},
        {"content_id": "b", "combined_score": 0.20},
    ]
    patcher, _ = _patch_search((rows, 2))
    with patcher:
        result = await hybrid_search_library_agents(
            query="x", user_id="u", min_score=0.35
        )
    assert [r["content_id"] for r in result] == ["a"]


@pytest.mark.asyncio
async def test_forwards_keyword_extracted_lexical_query():
    """Semantic uses the full sentence; lexical gets the stopword-stripped
    keyword form so plainto_tsquery's AND-of-terms doesn't zero out every
    match against an agent description."""
    rows: list[dict] = []
    patcher, mock_shim = _patch_search((rows, 0))
    with patcher:
        await hybrid_search_library_agents(
            query=(
                "Summarize a YouTube video with timestamped bullet points "
                "and topic summary from a URL input"
            ),
            user_id="u",
        )

    kwargs = mock_shim.unified_hybrid_search.call_args.kwargs
    # Full sentence preserved for embedding
    assert kwargs["query"].startswith("Summarize a YouTube video")
    # Lexical query: short keyword string, stopwords + short tokens dropped
    lex = kwargs["lexical_query"].split()
    assert len(lex) <= 5
    assert "the" not in lex and "a" not in lex and "from" not in lex
    assert "summarize" in lex
    assert "youtube" in lex
    assert "video" in lex


def test_extract_lexical_keywords_drops_stopwords_and_short_tokens():
    out = _extract_lexical_keywords("Build me an agent to summarize my emails")
    tokens = out.split()
    # 'Build', 'me', 'an', 'to', 'my' are stopwords/short → dropped
    assert "agent" in tokens
    assert "summarize" in tokens
    assert "emails" in tokens
    assert "build" not in tokens and "me" not in tokens and "to" not in tokens


def test_extract_lexical_keywords_caps_count():
    out = _extract_lexical_keywords(
        "alpha beta gamma delta epsilon zeta eta theta", max_keywords=5
    )
    assert out.split() == ["alpha", "beta", "gamma", "delta", "epsilon"]


def test_extract_lexical_keywords_dedupes_preserving_order():
    out = _extract_lexical_keywords("youtube video youtube summarizer video")
    assert out.split() == ["youtube", "video", "summarizer"]


def test_extract_lexical_keywords_empty_when_only_stopwords():
    assert _extract_lexical_keywords("a the of to and") == ""


# Note: caller-supplied ``min_score`` is now enforced inside the wrapper
# (so sub-threshold scores can be logged); see
# ``test_respects_explicit_min_score_override`` above.
