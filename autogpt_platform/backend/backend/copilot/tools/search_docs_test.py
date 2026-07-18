"""Tests for SearchDocsTool URL construction.

The doc URL shape (extension KEPT) is load-bearing: agpt.co serves the raw
repo docs verbatim, and the extension-less variant 308-redirects to the
wrong page. A regression re-introducing extension stripping must not ship
silently.
"""

from backend.copilot.tools.search_docs import SearchDocsTool
from backend.util.docs import DOCS_BASE_URL


def test_doc_url_keeps_md_extension():
    url = SearchDocsTool()._make_doc_url("platform/block-sdk-guide.md")
    assert url == f"{DOCS_BASE_URL}/platform/block-sdk-guide.md"
