"""Tests for SearchDocsTool URL construction.

The doc URL shape (extension STRIPPED) is load-bearing: agpt.co serves
rendered pages at the extension-less path, while the .md variant returns a
soft-404 (HTTP 200 + "Page Not Found" body). A regression re-introducing
the extension must not ship silently.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.tools.search_docs import SearchDocsTool
from backend.util.docs import DOCS_BASE_URL, make_doc_url


def test_doc_url_strips_md_extension():
    url = make_doc_url("platform/block-sdk-guide.md")
    assert url == f"{DOCS_BASE_URL}/platform/block-sdk-guide"


@pytest.mark.asyncio
async def test_search_result_doc_url_shape(mocker):
    """Behavioral check through the tool: a search hit's doc_url must be the
    extension-less rendered-page URL."""
    row = {
        "metadata": {
            "path": "platform/block-sdk-guide.md",
            "doc_title": "Block SDK Guide",
            "section_title": "Intro",
        },
        "searchable_text": "how to build blocks",
        "combined_score": 0.9,
    }
    db = MagicMock()
    db.unified_hybrid_search = AsyncMock(return_value=([row], 1))
    mocker.patch(
        "backend.copilot.tools.search_docs.search",
        return_value=db,
    )
    tool = SearchDocsTool()
    session = MagicMock()
    session.session_id = "sess-1"
    result = await tool._execute(user_id="user-1", session=session, query="blocks")

    assert result.results, getattr(result, "message", result)
    assert result.results[0].doc_url == f"{DOCS_BASE_URL}/platform/block-sdk-guide"
