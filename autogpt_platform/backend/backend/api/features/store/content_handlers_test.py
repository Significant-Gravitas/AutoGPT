"""
Tests for content handlers (blocks, store agents, documentation).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from prisma.enums import ContentType

from backend.api.features.store.content_handlers import (
    CONTENT_HANDLERS,
    BlockHandler,
    DocumentationHandler,
    StoreAgentHandler,
    _get_enabled_blocks,
)


@pytest.fixture(autouse=True)
def _clear_block_cache():
    """Clear the lru_cache on _get_enabled_blocks before each test."""
    _get_enabled_blocks.cache_clear()
    yield
    _get_enabled_blocks.cache_clear()


# ---------------------------------------------------------------------------
# Helper to build a mock block class that returns a pre-configured instance
# ---------------------------------------------------------------------------


def _make_block_class(
    *,
    name: str = "Block",
    description: str = "",
    disabled: bool = False,
    categories: list[MagicMock] | None = None,
    fields: dict[str, str] | None = None,
    raise_on_init: Exception | None = None,
) -> MagicMock:
    cls = MagicMock()
    if raise_on_init is not None:
        cls.side_effect = raise_on_init
        return cls
    inst = MagicMock()
    inst.name = name
    inst.disabled = disabled
    inst.description = description
    inst.categories = categories or []
    field_mocks = {
        fname: MagicMock(description=fdesc) for fname, fdesc in (fields or {}).items()
    }
    inst.input_schema.model_fields = field_mocks
    inst.input_schema.get_credentials_fields_info.return_value = {}
    cls.return_value = inst
    return cls


# ---------------------------------------------------------------------------
# _get_enabled_blocks
# ---------------------------------------------------------------------------


def test_get_enabled_blocks_filters_disabled():
    """Disabled blocks are excluded."""
    blocks = {
        "enabled": _make_block_class(name="E", disabled=False),
        "disabled": _make_block_class(name="D", disabled=True),
    }
    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ):
        result = _get_enabled_blocks()
    assert list(result.keys()) == ["enabled"]


def test_get_enabled_blocks_skips_broken():
    """Blocks that raise on init are skipped without crashing."""
    blocks = {
        "good": _make_block_class(name="Good"),
        "bad": _make_block_class(raise_on_init=RuntimeError("boom")),
    }
    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ):
        result = _get_enabled_blocks()
    assert list(result.keys()) == ["good"]


def test_get_enabled_blocks_cached():
    """_get_enabled_blocks() calls get_blocks() only once across multiple calls."""
    blocks = {"b1": _make_block_class(name="B1")}
    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ) as mock_get_blocks:
        result1 = _get_enabled_blocks()
        result2 = _get_enabled_blocks()
    assert result1 is result2
    mock_get_blocks.assert_called_once()


# ---------------------------------------------------------------------------
# StoreAgentHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_store_agent_handler_get_missing_items(mocker):
    """Test StoreAgentHandler fetches approved agents without embeddings."""
    handler = StoreAgentHandler()

    mock_missing = [
        {
            "id": "agent-1",
            "name": "Test Agent",
            "description": "A test agent",
            "subHeading": "Test heading",
            "categories": ["AI", "Testing"],
        }
    ]

    with patch(
        "backend.api.features.store.content_handlers.query_raw_with_schema",
        return_value=mock_missing,
    ):
        items = await handler.get_missing_items(batch_size=10)

        assert len(items) == 1
        assert items[0].content_id == "agent-1"
        assert items[0].content_type == ContentType.STORE_AGENT
        assert "Test Agent" in items[0].searchable_text
        assert "A test agent" in items[0].searchable_text
        assert items[0].metadata["name"] == "Test Agent"
        assert items[0].user_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_store_agent_handler_get_stats(mocker):
    """Test StoreAgentHandler returns correct stats."""
    handler = StoreAgentHandler()

    mock_approved = [{"count": 50}]
    mock_embedded = [{"count": 30}]

    with patch(
        "backend.api.features.store.content_handlers.query_raw_with_schema",
        side_effect=[mock_approved, mock_embedded],
    ):
        stats = await handler.get_stats()

        assert stats["total"] == 50
        assert stats["with_embeddings"] == 30
        assert stats["without_embeddings"] == 20


# ---------------------------------------------------------------------------
# BlockHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_get_missing_items():
    """Test BlockHandler discovers blocks without embeddings."""
    handler = BlockHandler()

    blocks = {
        "block-uuid-1": _make_block_class(
            name="CalculatorBlock",
            description="Performs calculations",
            categories=[MagicMock(value="MATH")],
            fields={"expression": "Math expression to evaluate"},
        ),
    }

    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=[],
        ):
            items = await handler.get_missing_items(batch_size=10)

            assert len(items) == 1
            assert items[0].content_id == "block-uuid-1"
            assert items[0].content_type == ContentType.BLOCK
            # CamelCase should be split in searchable text and metadata name
            assert "Calculator Block" in items[0].searchable_text
            assert "Performs calculations" in items[0].searchable_text
            assert "MATH" in items[0].searchable_text
            assert "expression: Math expression" in items[0].searchable_text
            assert items[0].metadata["name"] == "Calculator Block"
            assert items[0].user_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_get_missing_items_splits_camelcase():
    """CamelCase block names are split for better search indexing."""
    handler = BlockHandler()

    blocks = {
        "ai-block": _make_block_class(name="AITextGeneratorBlock"),
    }

    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=[],
        ):
            items = await handler.get_missing_items(batch_size=10)

            assert len(items) == 1
            assert "AI Text Generator Block" in items[0].searchable_text


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_get_missing_items_batch_size_zero():
    """batch_size=0 returns an empty list; the DB is still queried to find missing IDs."""
    handler = BlockHandler()

    blocks = {"b1": _make_block_class(name="B1")}

    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=[],
        ) as mock_query:
            items = await handler.get_missing_items(batch_size=0)
            assert items == []
            # DB query is still issued to learn which blocks lack embeddings;
            # the empty result comes from itertools.islice limiting to 0 items.
            mock_query.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_disabled_dont_exhaust_batch():
    """Disabled blocks don't consume batch budget, so enabled blocks get indexed."""
    handler = BlockHandler()

    # 5 disabled + 3 enabled, batch_size=2
    blocks = {
        **{
            f"dis-{i}": _make_block_class(name=f"D{i}", disabled=True) for i in range(5)
        },
        **{f"en-{i}": _make_block_class(name=f"E{i}") for i in range(3)},
    }

    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=[],
        ):
            items = await handler.get_missing_items(batch_size=2)

            assert len(items) == 2
            assert all(item.content_id.startswith("en-") for item in items)


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_get_stats():
    """Test BlockHandler returns correct stats."""
    handler = BlockHandler()

    blocks = {
        "block-1": _make_block_class(name="B1"),
        "block-2": _make_block_class(name="B2"),
        "block-3": _make_block_class(name="B3"),
    }

    mock_embedded = [{"count": 2}]

    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=mock_embedded,
        ):
            stats = await handler.get_stats()

            assert stats["total"] == 3
            assert stats["with_embeddings"] == 2
            assert stats["without_embeddings"] == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_get_stats_skips_broken():
    """get_stats skips broken blocks instead of crashing."""
    handler = BlockHandler()

    blocks = {
        "good": _make_block_class(name="Good"),
        "bad": _make_block_class(raise_on_init=RuntimeError("boom")),
    }

    mock_embedded = [{"count": 1}]

    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=mock_embedded,
        ):
            stats = await handler.get_stats()

            assert stats["total"] == 1  # only the good block
            assert stats["with_embeddings"] == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_handles_none_name():
    """When block.name is None the fallback display name logic is used."""
    handler = BlockHandler()

    blocks = {
        "none-name-block": _make_block_class(
            name="placeholder",  # will be overridden to None below
            description="A block with no name",
        ),
    }
    # Override the name to None after construction so _make_block_class
    # doesn't interfere with the mock wiring.
    blocks["none-name-block"].return_value.name = None

    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=[],
        ):
            items = await handler.get_missing_items(batch_size=10)

            assert len(items) == 1
            # display_name should be "" because block.name is None
            # searchable_text should still contain the description
            assert "A block with no name" in items[0].searchable_text
            # metadata["name"] falls back to block_id when both display_name
            # and block.name are falsy, ensuring it is always a non-empty string.
            assert items[0].metadata["name"] == "none-name-block"


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_handles_empty_attributes():
    """Test BlockHandler handles blocks with empty/falsy attribute values."""
    handler = BlockHandler()

    blocks = {"block-minimal": _make_block_class(name="Minimal Block")}

    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=[],
        ):
            items = await handler.get_missing_items(batch_size=10)

            assert len(items) == 1
            assert items[0].searchable_text == "Minimal Block"


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_skips_failed_blocks():
    """Test BlockHandler skips blocks that fail to instantiate."""
    handler = BlockHandler()

    blocks = {
        "good-block": _make_block_class(name="Good Block", description="Works fine"),
        "bad-block": _make_block_class(raise_on_init=Exception("Instantiation failed")),
    }

    with patch(
        "backend.api.features.store.content_handlers.get_blocks", return_value=blocks
    ):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=[],
        ):
            items = await handler.get_missing_items(batch_size=10)

            assert len(items) == 1
            assert items[0].content_id == "good-block"


# ---------------------------------------------------------------------------
# DocumentationHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_documentation_handler_get_missing_items(tmp_path, mocker):
    """Test DocumentationHandler discovers docs without embeddings."""
    handler = DocumentationHandler()

    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "guide.md").write_text("# Getting Started\n\nThis is a guide.")
    (docs_root / "api.mdx").write_text("# API Reference\n\nAPI documentation.")

    with patch.object(handler, "_get_docs_root", return_value=docs_root):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=[],
        ):
            items = await handler.get_missing_items(batch_size=10)

            assert len(items) == 2

            guide_item = next(
                (item for item in items if item.content_id == "guide.md::0"), None
            )
            assert guide_item is not None
            assert guide_item.content_type == ContentType.DOCUMENTATION
            assert "Getting Started" in guide_item.searchable_text
            assert "This is a guide" in guide_item.searchable_text
            assert guide_item.metadata["doc_title"] == "Getting Started"
            assert guide_item.user_id is None

            api_item = next(
                (item for item in items if item.content_id == "api.mdx::0"), None
            )
            assert api_item is not None
            assert "API Reference" in api_item.searchable_text


@pytest.mark.asyncio(loop_scope="session")
async def test_documentation_handler_get_stats(tmp_path, mocker):
    """Test DocumentationHandler returns correct stats."""
    handler = DocumentationHandler()

    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "doc1.md").write_text("# Doc 1")
    (docs_root / "doc2.md").write_text("# Doc 2")
    (docs_root / "doc3.mdx").write_text("# Doc 3")

    mock_embedded = [{"count": 1}]

    with patch.object(handler, "_get_docs_root", return_value=docs_root):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=mock_embedded,
        ):
            stats = await handler.get_stats()

            assert stats["total"] == 3
            assert stats["with_embeddings"] == 1
            assert stats["without_embeddings"] == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_documentation_handler_title_extraction(tmp_path):
    """Test DocumentationHandler extracts title from markdown heading."""
    handler = DocumentationHandler()

    doc_with_heading = tmp_path / "with_heading.md"
    doc_with_heading.write_text("# My Title\n\nContent here")
    title = handler._extract_doc_title(doc_with_heading)
    assert title == "My Title"

    doc_without_heading = tmp_path / "no-heading.md"
    doc_without_heading.write_text("Just content, no heading")
    title = handler._extract_doc_title(doc_without_heading)
    assert title == "No Heading"  # Uses filename


@pytest.mark.asyncio(loop_scope="session")
async def test_documentation_handler_markdown_chunking(tmp_path):
    """Test DocumentationHandler chunks markdown by headings."""
    handler = DocumentationHandler()

    doc_with_sections = tmp_path / "sections.md"
    doc_with_sections.write_text(
        "# Document Title\n\n"
        "Intro paragraph.\n\n"
        "## Section One\n\n"
        "Content for section one.\n\n"
        "## Section Two\n\n"
        "Content for section two.\n"
    )
    sections = handler._chunk_markdown_by_headings(doc_with_sections)

    assert len(sections) == 3
    assert sections[0].title == "Document Title"
    assert sections[0].index == 0
    assert "Intro paragraph" in sections[0].content

    assert sections[1].title == "Section One"
    assert sections[1].index == 1
    assert "Content for section one" in sections[1].content

    assert sections[2].title == "Section Two"
    assert sections[2].index == 2
    assert "Content for section two" in sections[2].content

    doc_no_sections = tmp_path / "no-sections.md"
    doc_no_sections.write_text("Just plain content without any headings.")
    sections = handler._chunk_markdown_by_headings(doc_no_sections)
    assert len(sections) == 1
    assert sections[0].index == 0
    assert "Just plain content" in sections[0].content


@pytest.mark.asyncio(loop_scope="session")
async def test_documentation_handler_section_content_ids():
    """Test DocumentationHandler creates and parses section content IDs."""
    handler = DocumentationHandler()

    content_id = handler._make_section_content_id("docs/guide.md", 2)
    assert content_id == "docs/guide.md::2"

    doc_path, section_index = handler._parse_section_content_id("docs/guide.md::2")
    assert doc_path == "docs/guide.md"
    assert section_index == 2

    doc_path, section_index = handler._parse_section_content_id("docs/old-format.md")
    assert doc_path == "docs/old-format.md"
    assert section_index == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_documentation_handler_missing_docs_directory():
    """Test DocumentationHandler handles missing docs directory gracefully."""
    handler = DocumentationHandler()

    fake_path = Path("/nonexistent/docs")
    with patch.object(handler, "_get_docs_root", return_value=fake_path):
        items = await handler.get_missing_items(batch_size=10)
        assert items == []

        stats = await handler.get_stats()
        assert stats["total"] == 0
        assert stats["with_embeddings"] == 0
        assert stats["without_embeddings"] == 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_content_handlers_registry():
    """Test all content types are registered."""
    assert ContentType.STORE_AGENT in CONTENT_HANDLERS
    assert ContentType.BLOCK in CONTENT_HANDLERS
    assert ContentType.DOCUMENTATION in CONTENT_HANDLERS

    assert isinstance(CONTENT_HANDLERS[ContentType.STORE_AGENT], StoreAgentHandler)
    assert isinstance(CONTENT_HANDLERS[ContentType.BLOCK], BlockHandler)
    assert isinstance(CONTENT_HANDLERS[ContentType.DOCUMENTATION], DocumentationHandler)
