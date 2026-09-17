"""Tests for GetDocPageTool (and the shared docs-root resolution)."""

import pytest

from backend.copilot.tools.get_doc_page import GetDocPageTool
from backend.copilot.tools.models import DocPageResponse, ErrorResponse
from backend.util.docs import DOCS_BASE_URL, get_docs_root

from ._test_data import make_session

_TEST_USER_ID = "test-user-get-doc-page"


@pytest.fixture
def tool():
    return GetDocPageTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


requires_docs_bundle = pytest.mark.skipif(
    get_docs_root() is None,
    reason="integration: needs the repo/image docs bundle on disk",
)


@requires_docs_bundle
def test_docs_root_resolves_to_bundled_docs():
    """Regression: the old parent-chain arithmetic resolved OUTSIDE the repo
    (two levels too far), making every page read 404 in dev and cloud."""
    root = get_docs_root()
    assert root.is_dir()
    assert root.name == "docs"
    assert any(root.rglob("*.md"))


@pytest.mark.asyncio
@requires_docs_bundle
async def test_fetches_real_doc_page(tool, session):
    """Any indexed-style relative path under docs/ must be readable."""
    root = get_docs_root()
    doc_file = next(f for f in root.rglob("*.md") if f.stat().st_size > 0)
    rel_path = str(doc_file.relative_to(root))

    result = await tool._execute(user_id=None, session=session, path=rel_path)

    assert isinstance(result, DocPageResponse)
    assert result.content
    assert result.path == rel_path
    # The docs site serves rendered pages at the extension-less path;
    # the .md variant is a soft-404 (HTTP 200 + "Page Not Found" body).
    expected_slug = rel_path.removesuffix(".md").replace("_", "-")
    assert result.doc_url == f"{DOCS_BASE_URL}/{expected_slug}"


@pytest.mark.asyncio
async def test_happy_path_hermetic(tool, session, tmp_path, mocker):
    """Unconditional (mocked tmp docs root, no bundle needed): a valid
    in-root path returns a DocPageResponse with content, title, and a
    canonicalized doc_url — locks the repaired resolution regression in
    even on docs-less environments."""
    docs = tmp_path / "docs"
    (docs / "platform").mkdir(parents=True)
    (docs / "platform" / "getting_started.md").write_text("# Getting Started\n\nHello.")
    mocker.patch(
        "backend.copilot.tools.get_doc_page.get_docs_root",
        return_value=docs.resolve(),
    )

    result = await tool._execute(
        user_id=None, session=session, path="platform/getting_started.md"
    )

    assert isinstance(result, DocPageResponse)
    assert result.title == "Getting Started"
    assert result.content == "# Getting Started\n\nHello."
    assert result.path == "platform/getting_started.md"
    assert result.doc_url == f"{DOCS_BASE_URL}/platform/getting-started"


@pytest.mark.asyncio
@requires_docs_bundle
async def test_missing_page_returns_not_found(tool, session):
    result = await tool._execute(user_id=None, session=session, path="no/such/page.md")
    assert isinstance(result, ErrorResponse)
    assert result.error == "not_found"


@pytest.mark.asyncio
async def test_traversal_is_blocked(tool, session, tmp_path, mocker):
    """Unconditional (mocked tmp docs root, no bundle needed): a ../ path
    targeting a real out-of-root file is rejected before existence leaks."""
    docs = tmp_path / "docs"
    (docs / "platform").mkdir(parents=True)
    (tmp_path / "secret.toml").write_text("nope")
    mocker.patch(
        "backend.copilot.tools.get_doc_page.get_docs_root",
        return_value=docs.resolve(),
    )
    result = await tool._execute(user_id=None, session=session, path="../secret.toml")
    assert isinstance(result, ErrorResponse)
    assert result.error == "invalid_path"


@pytest.mark.asyncio
async def test_docs_unavailable_returns_dedicated_error(tool, session, mocker):
    """A deployment without bundled docs degrades to a clear error instead
    of a misleading not_found."""
    mocker.patch(
        "backend.copilot.tools.get_doc_page.get_docs_root",
        return_value=None,
    )
    result = await tool._execute(
        user_id=None, session=session, path="platform/getting-started.md"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "docs_unavailable"


@pytest.mark.asyncio
async def test_symlink_escape_blocked_by_containment(tool, session, tmp_path, mocker):
    """A path with no ../ that resolves OUTSIDE docs_root via an in-tree
    symlink must hit the containment check (the string guard can't see it)."""
    docs = tmp_path / "docs"
    (docs / "platform").mkdir(parents=True)
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    (docs / "platform" / "leak.md").symlink_to(outside)
    mocker.patch(
        "backend.copilot.tools.get_doc_page.get_docs_root",
        return_value=docs.resolve(),
    )
    result = await tool._execute(user_id=None, session=session, path="platform/leak.md")
    assert isinstance(result, ErrorResponse)
    assert result.error == "invalid_path"


@pytest.mark.asyncio
async def test_directory_path_returns_read_failed(tool, session, tmp_path, mocker):
    docs = tmp_path / "docs"
    (docs / "platform").mkdir(parents=True)
    mocker.patch(
        "backend.copilot.tools.get_doc_page.get_docs_root",
        return_value=docs.resolve(),
    )
    result = await tool._execute(user_id=None, session=session, path="platform")
    assert isinstance(result, ErrorResponse)
    assert result.error == "read_failed"


@pytest.mark.asyncio
async def test_empty_path_rejected(tool, session):
    result = await tool._execute(user_id=None, session=session, path="  ")
    assert isinstance(result, ErrorResponse)
    assert "path" in result.message.lower()
