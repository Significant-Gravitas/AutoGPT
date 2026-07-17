"""Tests for GetDocPageTool (and the shared docs-root resolution)."""

import pytest

from backend.copilot.tools.get_doc_page import DOCS_BASE_URL, GetDocPageTool
from backend.copilot.tools.models import DocPageResponse, ErrorResponse
from backend.util.docs import get_docs_root

from ._test_data import make_session

_TEST_USER_ID = "test-user-get-doc-page"


@pytest.fixture
def tool():
    return GetDocPageTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


def test_docs_root_resolves_to_bundled_docs():
    """Regression: the old parent-chain arithmetic resolved OUTSIDE the repo
    (two levels too far), making every page read 404 in dev and cloud."""
    root = get_docs_root()
    assert root.is_dir()
    assert root.name == "docs"
    assert any(root.rglob("*.md"))


@pytest.mark.asyncio
async def test_fetches_real_doc_page(tool, session):
    """Any indexed-style relative path under docs/ must be readable."""
    root = get_docs_root()
    doc_file = next(f for f in root.rglob("*.md") if f.stat().st_size > 0)
    rel_path = str(doc_file.relative_to(root))

    result = await tool._execute(user_id=None, session=session, path=rel_path)

    assert isinstance(result, DocPageResponse)
    assert result.content
    assert result.path == rel_path
    # The docs site serves raw repo paths verbatim, extension included.
    assert result.doc_url == f"{DOCS_BASE_URL}/{rel_path}"


@pytest.mark.asyncio
async def test_missing_page_returns_not_found(tool, session):
    result = await tool._execute(user_id=None, session=session, path="no/such/page.md")
    assert isinstance(result, ErrorResponse)
    assert result.error == "not_found"


@pytest.mark.asyncio
async def test_traversal_is_blocked(tool, session):
    result = await tool._execute(
        user_id=None, session=session, path="../backend/pyproject.toml"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "invalid_path"
