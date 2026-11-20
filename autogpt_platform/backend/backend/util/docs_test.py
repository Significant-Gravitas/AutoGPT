"""Tests for the docs-root walk-up resolution."""

from pathlib import Path

import pytest

from backend.util.docs import DOCS_BASE_URL, get_docs_root, make_doc_url


@pytest.fixture(autouse=True)
def _clear_cache():
    get_docs_root.cache_clear()
    yield
    get_docs_root.cache_clear()


def _make_docs_tree(root: Path) -> Path:
    docs = root / "docs" / "platform"
    docs.mkdir(parents=True)
    return root / "docs"


def test_walks_up_to_docs_root(tmp_path: Path):
    docs = _make_docs_tree(tmp_path)
    start = tmp_path / "app" / "backend" / "util" / "docs.py"
    start.parent.mkdir(parents=True)
    start.touch()
    assert get_docs_root(start) == docs


def test_sentinel_skips_shadowing_docs_dir(tmp_path: Path):
    """A closer docs/ WITHOUT the platform sentinel must not shadow the
    real documentation root further up."""
    real_docs = _make_docs_tree(tmp_path)
    shadow = tmp_path / "app" / "backend" / "docs"
    shadow.mkdir(parents=True)
    start = tmp_path / "app" / "backend" / "util" / "docs.py"
    start.parent.mkdir(parents=True)
    start.touch()
    assert get_docs_root(start) == real_docs


def test_missing_docs_returns_none(tmp_path: Path):
    start = tmp_path / "app" / "backend" / "util" / "docs.py"
    start.parent.mkdir(parents=True)
    start.touch()
    assert get_docs_root(start) is None


def test_negative_result_memoized(tmp_path: Path):
    start = tmp_path / "app" / "backend" / "util" / "docs.py"
    start.parent.mkdir(parents=True)
    start.touch()
    assert get_docs_root(start) is None
    assert get_docs_root.cache_info().misses == 1
    # Negative result is cached — a second identical call must not re-walk.
    assert get_docs_root(start) is None
    assert get_docs_root.cache_info().hits == 1
    assert get_docs_root.cache_info().misses == 1


def test_make_doc_url_strips_extension_and_leading_slash():
    """agpt.co serves rendered pages extension-less; the .md variant is a
    soft-404 (HTTP 200 + "Page Not Found" body)."""
    assert make_doc_url("a/b.md").endswith("/docs/a/b")
    assert make_doc_url("a/b.mdx").endswith("/docs/a/b")
    assert make_doc_url("/a/b.md") == make_doc_url("a/b.md")


def test_docs_base_url_pins_live_host():
    """Guards against a silent revert to the dead docs.agpt.co host — the
    shape tests are all relative to this constant."""
    assert DOCS_BASE_URL == "https://agpt.co/docs"


def test_make_doc_url_passthrough_without_extension():
    assert make_doc_url("a/b").endswith("/docs/a/b")
    # Dots inside names are not extensions — only .md/.mdx strip.
    assert make_doc_url("platform/v1.2").endswith("/docs/platform/v1.2")
    assert make_doc_url("platform/v1.2-guide.md").endswith("/docs/platform/v1.2-guide")


def test_sentinel_rejects_platform_file(tmp_path: Path):
    """docs/platform must be a DIRECTORY — a stray file must not satisfy
    the sentinel."""
    fake = tmp_path / "docs"
    fake.mkdir()
    (fake / "platform").touch()
    start = tmp_path / "app" / "backend" / "util" / "docs.py"
    start.parent.mkdir(parents=True)
    start.touch()
    assert get_docs_root(start) is None


def test_make_doc_url_canonicalizes_underscores():
    """The site's edge worker rewrites _ to - (cloudflare_worker.js);
    verified live: platform/new-blocks renders, platform/new_blocks 404s."""
    assert make_doc_url("platform/new_blocks.md").endswith("/docs/platform/new-blocks")
