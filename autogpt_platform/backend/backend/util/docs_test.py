"""Tests for the docs-root walk-up resolution."""

from pathlib import Path

import pytest

from backend.util.docs import (
    _find_docs_root,
    get_docs_root,
    get_docs_root_or_none,
    make_doc_url,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    _find_docs_root.cache_clear()
    yield
    _find_docs_root.cache_clear()


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


def test_missing_docs_raises(tmp_path: Path):
    start = tmp_path / "app" / "backend" / "util" / "docs.py"
    start.parent.mkdir(parents=True)
    start.touch()
    with pytest.raises(FileNotFoundError):
        get_docs_root(start)


def test_or_none_returns_none_and_memoizes(tmp_path: Path):
    start = tmp_path / "app" / "backend" / "util" / "docs.py"
    start.parent.mkdir(parents=True)
    start.touch()
    assert get_docs_root_or_none(start) is None
    # Negative result is cached — a second call must not re-walk.
    assert _find_docs_root.cache_info().hits >= 0
    assert get_docs_root_or_none(start) is None
    assert _find_docs_root.cache_info().hits >= 1


def test_make_doc_url_keeps_extension_and_strips_leading_slash():
    assert make_doc_url("a/b.md").endswith("/docs/a/b.md")
    assert make_doc_url("/a/b.md") == make_doc_url("a/b.md")
