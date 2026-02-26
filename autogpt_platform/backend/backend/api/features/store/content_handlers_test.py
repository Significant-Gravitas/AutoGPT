"""
E2E tests for content handlers (blocks, store agents, documentation).

Tests the full flow: discovering content → generating embeddings → storing.
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
)


@pytest.mark.asyncio(loop_scope="session")
async def test_store_agent_handler_get_missing_items(mocker):
    """Test StoreAgentHandler fetches approved agents without embeddings."""
    handler = StoreAgentHandler()

    # Mock database query
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

    # Mock approved count query
    mock_approved = [{"count": 50}]
    # Mock embedded count query
    mock_embedded = [{"count": 30}]

    with patch(
        "backend.api.features.store.content_handlers.query_raw_with_schema",
        side_effect=[mock_approved, mock_embedded],
    ):
        stats = await handler.get_stats()

        assert stats["total"] == 50
        assert stats["with_embeddings"] == 30
        assert stats["without_embeddings"] == 20


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_get_missing_items(mocker):
    """Test BlockHandler discovers blocks without embeddings."""
    handler = BlockHandler()

    # Mock get_blocks to return test blocks
    mock_block_class = MagicMock()
    mock_block_instance = MagicMock()
    mock_block_instance.name = "Calculator Block"
    mock_block_instance.description = "Performs calculations"
    mock_block_instance.categories = [MagicMock(value="MATH")]
    mock_block_instance.disabled = False
    mock_field = MagicMock()
    mock_field.description = "Math expression to evaluate"
    mock_block_instance.input_schema.model_fields = {"expression": mock_field}
    mock_block_instance.input_schema.get_credentials_fields_info.return_value = {}
    mock_block_class.return_value = mock_block_instance

    mock_blocks = {"block-uuid-1": mock_block_class}

    # Mock existing embeddings query (no embeddings exist)
    mock_existing = []

    with patch(
        "backend.blocks.get_blocks",
        return_value=mock_blocks,
    ):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=mock_existing,
        ):
            items = await handler.get_missing_items(batch_size=10)

            assert len(items) == 1
            assert items[0].content_id == "block-uuid-1"
            assert items[0].content_type == ContentType.BLOCK
            assert "Calculator Block" in items[0].searchable_text
            assert "Performs calculations" in items[0].searchable_text
            assert "MATH" in items[0].searchable_text
            assert "expression: Math expression" in items[0].searchable_text
            assert items[0].user_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_get_stats(mocker):
    """Test BlockHandler returns correct stats."""
    handler = BlockHandler()

    # Mock get_blocks - each block class returns an instance with disabled=False
    def make_mock_block_class():
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.disabled = False
        mock_class.return_value = mock_instance
        return mock_class

    mock_blocks = {
        "block-1": make_mock_block_class(),
        "block-2": make_mock_block_class(),
        "block-3": make_mock_block_class(),
    }

    # Mock embedded count query (2 blocks have embeddings)
    mock_embedded = [{"count": 2}]

    with patch(
        "backend.blocks.get_blocks",
        return_value=mock_blocks,
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
async def test_documentation_handler_get_missing_items(tmp_path, mocker):
    """Test DocumentationHandler discovers docs without embeddings."""
    handler = DocumentationHandler()

    # Create temporary docs directory with test files
    docs_root = tmp_path / "docs"
    docs_root.mkdir()

    (docs_root / "guide.md").write_text("# Getting Started\n\nThis is a guide.")
    (docs_root / "api.mdx").write_text("# API Reference\n\nAPI documentation.")

    # Mock _get_docs_root to return temp dir
    with patch.object(handler, "_get_docs_root", return_value=docs_root):
        # Mock existing embeddings query (no embeddings exist)
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=[],
        ):
            items = await handler.get_missing_items(batch_size=10)

            assert len(items) == 2

            # Check guide.md (content_id format: doc_path::section_index)
            guide_item = next(
                (item for item in items if item.content_id == "guide.md::0"), None
            )
            assert guide_item is not None
            assert guide_item.content_type == ContentType.DOCUMENTATION
            assert "Getting Started" in guide_item.searchable_text
            assert "This is a guide" in guide_item.searchable_text
            assert guide_item.metadata["doc_title"] == "Getting Started"
            assert guide_item.user_id is None

            # Check api.mdx (content_id format: doc_path::section_index)
            api_item = next(
                (item for item in items if item.content_id == "api.mdx::0"), None
            )
            assert api_item is not None
            assert "API Reference" in api_item.searchable_text


@pytest.mark.asyncio(loop_scope="session")
async def test_documentation_handler_get_stats(tmp_path, mocker):
    """Test DocumentationHandler returns correct stats."""
    handler = DocumentationHandler()

    # Create temporary docs directory
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "doc1.md").write_text("# Doc 1")
    (docs_root / "doc2.md").write_text("# Doc 2")
    (docs_root / "doc3.mdx").write_text("# Doc 3")

    # Mock embedded count query (1 doc has embedding)
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

    # Test with heading
    doc_with_heading = tmp_path / "with_heading.md"
    doc_with_heading.write_text("# My Title\n\nContent here")
    title = handler._extract_doc_title(doc_with_heading)
    assert title == "My Title"

    # Test without heading
    doc_without_heading = tmp_path / "no-heading.md"
    doc_without_heading.write_text("Just content, no heading")
    title = handler._extract_doc_title(doc_without_heading)
    assert title == "No Heading"  # Uses filename


@pytest.mark.asyncio(loop_scope="session")
async def test_documentation_handler_markdown_chunking(tmp_path):
    """Test DocumentationHandler chunks markdown by headings."""
    handler = DocumentationHandler()

    # Test document with multiple sections
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

    # Should have 3 sections: intro (with doc title), section one, section two
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

    # Test document without headings
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

    # Test making content ID
    content_id = handler._make_section_content_id("docs/guide.md", 2)
    assert content_id == "docs/guide.md::2"

    # Test parsing content ID
    doc_path, section_index = handler._parse_section_content_id("docs/guide.md::2")
    assert doc_path == "docs/guide.md"
    assert section_index == 2

    # Test parsing legacy format (no section index)
    doc_path, section_index = handler._parse_section_content_id("docs/old-format.md")
    assert doc_path == "docs/old-format.md"
    assert section_index == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_content_handlers_registry():
    """Test all content types are registered."""
    assert ContentType.STORE_AGENT in CONTENT_HANDLERS
    assert ContentType.BLOCK in CONTENT_HANDLERS
    assert ContentType.DOCUMENTATION in CONTENT_HANDLERS

    assert isinstance(CONTENT_HANDLERS[ContentType.STORE_AGENT], StoreAgentHandler)
    assert isinstance(CONTENT_HANDLERS[ContentType.BLOCK], BlockHandler)
    assert isinstance(CONTENT_HANDLERS[ContentType.DOCUMENTATION], DocumentationHandler)


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_handles_empty_attributes():
    """Test BlockHandler handles blocks with empty/falsy attribute values."""
    handler = BlockHandler()

    # Mock block with empty values (all attributes exist but are falsy)
    mock_block_class = MagicMock()
    mock_block_instance = MagicMock()
    mock_block_instance.name = "Minimal Block"
    mock_block_instance.disabled = False
    mock_block_instance.description = ""
    mock_block_instance.categories = set()
    mock_block_instance.input_schema.model_fields = {}
    mock_block_instance.input_schema.get_credentials_fields_info.return_value = {}
    mock_block_class.return_value = mock_block_instance

    mock_blocks = {"block-minimal": mock_block_class}

    with patch(
        "backend.blocks.get_blocks",
        return_value=mock_blocks,
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

    # Mock one good block and one bad block
    good_block = MagicMock()
    good_instance = MagicMock()
    good_instance.name = "Good Block"
    good_instance.description = "Works fine"
    good_instance.categories = []
    good_instance.disabled = False
    good_instance.input_schema.model_fields = {}
    good_instance.input_schema.get_credentials_fields_info.return_value = {}
    good_block.return_value = good_instance

    bad_block = MagicMock()
    bad_block.side_effect = Exception("Instantiation failed")

    mock_blocks = {"good-block": good_block, "bad-block": bad_block}

    with patch(
        "backend.blocks.get_blocks",
        return_value=mock_blocks,
    ):
        with patch(
            "backend.api.features.store.content_handlers.query_raw_with_schema",
            return_value=[],
        ):
            items = await handler.get_missing_items(batch_size=10)

            # Should only get the good block
            assert len(items) == 1
            assert items[0].content_id == "good-block"


@pytest.mark.asyncio(loop_scope="session")
async def test_documentation_handler_missing_docs_directory():
    """Test DocumentationHandler handles missing docs directory gracefully."""
    handler = DocumentationHandler()

    # Mock _get_docs_root to return non-existent path
    fake_path = Path("/nonexistent/docs")
    with patch.object(handler, "_get_docs_root", return_value=fake_path):
        items = await handler.get_missing_items(batch_size=10)
        assert items == []

        stats = await handler.get_stats()
        assert stats["total"] == 0
        assert stats["with_embeddings"] == 0
        assert stats["without_embeddings"] == 0
