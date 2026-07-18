"""Tests for SearchDocsTool URL construction.

The doc URL shape (extension KEPT) is load-bearing: agpt.co serves the raw
repo docs verbatim, and the extension-less variant 308-redirects to the
wrong page. A regression re-introducing extension stripping must not ship
silently.
"""

import backend.copilot.tools.search_docs as search_docs_module
from backend.util.docs import DOCS_BASE_URL, make_doc_url


def test_doc_url_keeps_md_extension():
    url = make_doc_url("platform/block-sdk-guide.md")
    assert url == f"{DOCS_BASE_URL}/platform/block-sdk-guide.md"


def test_search_docs_uses_shared_url_helper():
    assert search_docs_module.make_doc_url is make_doc_url
