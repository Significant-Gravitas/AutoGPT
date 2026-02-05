"""
Content Type Handlers for Unified Embeddings

Pluggable system for different content sources (store agents, blocks, docs).
Each handler knows how to fetch and process its content type for embedding.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prisma.enums import ContentType

from backend.data.db import query_raw_with_schema

logger = logging.getLogger(__name__)


@dataclass
class ContentItem:
    """Represents a piece of content to be embedded."""

    content_id: str  # Unique identifier (DB ID or file path)
    content_type: ContentType
    searchable_text: str  # Combined text for embedding
    metadata: dict[str, Any]  # Content-specific metadata
    user_id: str | None = None  # For user-scoped content


class ContentHandler(ABC):
    """Base handler for fetching and processing content for embeddings."""

    @property
    @abstractmethod
    def content_type(self) -> ContentType:
        """The ContentType this handler manages."""
        pass

    @abstractmethod
    async def get_missing_items(self, batch_size: int) -> list[ContentItem]:
        """
        Fetch items that don't have embeddings yet.

        Args:
            batch_size: Maximum number of items to return

        Returns:
            List of ContentItem objects ready for embedding
        """
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, int]:
        """
        Get statistics about embedding coverage.

        Returns:
            Dict with keys: total, with_embeddings, without_embeddings
        """
        pass


class StoreAgentHandler(ContentHandler):
    """Handler for marketplace store agent listings."""

    @property
    def content_type(self) -> ContentType:
        return ContentType.STORE_AGENT

    async def get_missing_items(self, batch_size: int) -> list[ContentItem]:
        """Fetch approved store listings without embeddings."""
        from backend.api.features.store.embeddings import build_searchable_text

        missing = await query_raw_with_schema(
            """
            SELECT
                slv.id,
                slv.name,
                slv.description,
                slv."subHeading",
                slv.categories
            FROM {schema_prefix}"StoreListingVersion" slv
            LEFT JOIN {schema_prefix}"UnifiedContentEmbedding" uce
                ON slv.id = uce."contentId" AND uce."contentType" = 'STORE_AGENT'::{schema_prefix}"ContentType"
            WHERE slv."submissionStatus" = 'APPROVED'
            AND slv."isDeleted" = false
            AND uce."contentId" IS NULL
            LIMIT $1
            """,
            batch_size,
        )

        return [
            ContentItem(
                content_id=row["id"],
                content_type=ContentType.STORE_AGENT,
                searchable_text=build_searchable_text(
                    name=row["name"],
                    description=row["description"],
                    sub_heading=row["subHeading"],
                    categories=row["categories"] or [],
                ),
                metadata={
                    "name": row["name"],
                    "categories": row["categories"] or [],
                },
                user_id=None,  # Store agents are public
            )
            for row in missing
        ]

    async def get_stats(self) -> dict[str, int]:
        """Get statistics about store agent embedding coverage."""
        # Count approved versions
        approved_result = await query_raw_with_schema(
            """
            SELECT COUNT(*) as count
            FROM {schema_prefix}"StoreListingVersion"
            WHERE "submissionStatus" = 'APPROVED'
            AND "isDeleted" = false
            """
        )
        total_approved = approved_result[0]["count"] if approved_result else 0

        # Count versions with embeddings
        embedded_result = await query_raw_with_schema(
            """
            SELECT COUNT(*) as count
            FROM {schema_prefix}"StoreListingVersion" slv
            JOIN {schema_prefix}"UnifiedContentEmbedding" uce ON slv.id = uce."contentId" AND uce."contentType" = 'STORE_AGENT'::{schema_prefix}"ContentType"
            WHERE slv."submissionStatus" = 'APPROVED'
            AND slv."isDeleted" = false
            """
        )
        with_embeddings = embedded_result[0]["count"] if embedded_result else 0

        return {
            "total": total_approved,
            "with_embeddings": with_embeddings,
            "without_embeddings": total_approved - with_embeddings,
        }


class BlockHandler(ContentHandler):
    """Handler for block definitions (Python classes)."""

    @property
    def content_type(self) -> ContentType:
        return ContentType.BLOCK

    async def get_missing_items(self, batch_size: int) -> list[ContentItem]:
        """Fetch blocks without embeddings."""
        from backend.data.block import get_blocks

        # Get all available blocks
        all_blocks = get_blocks()

        # Check which ones have embeddings
        if not all_blocks:
            return []

        block_ids = list(all_blocks.keys())

        # Query for existing embeddings
        placeholders = ",".join([f"${i+1}" for i in range(len(block_ids))])
        existing_result = await query_raw_with_schema(
            f"""
            SELECT "contentId"
            FROM {{schema_prefix}}"UnifiedContentEmbedding"
            WHERE "contentType" = 'BLOCK'::{{schema_prefix}}"ContentType"
            AND "contentId" = ANY(ARRAY[{placeholders}])
            """,
            *block_ids,
        )

        existing_ids = {row["contentId"] for row in existing_result}
        missing_blocks = [
            (block_id, block_cls)
            for block_id, block_cls in all_blocks.items()
            if block_id not in existing_ids
        ]

        # Convert to ContentItem
        items = []
        for block_id, block_cls in missing_blocks[:batch_size]:
            try:
                block_instance = block_cls()

                # Skip disabled blocks - they shouldn't be indexed
                if block_instance.disabled:
                    continue

                # Build searchable text from block metadata
                parts = []
                if hasattr(block_instance, "name") and block_instance.name:
                    parts.append(block_instance.name)
                if (
                    hasattr(block_instance, "description")
                    and block_instance.description
                ):
                    parts.append(block_instance.description)
                if hasattr(block_instance, "categories") and block_instance.categories:
                    # Convert BlockCategory enum to strings
                    parts.append(
                        " ".join(str(cat.value) for cat in block_instance.categories)
                    )

                # Add input/output schema info
                if hasattr(block_instance, "input_schema"):
                    schema = block_instance.input_schema
                    if hasattr(schema, "model_json_schema"):
                        schema_dict = schema.model_json_schema()
                        if "properties" in schema_dict:
                            for prop_name, prop_info in schema_dict[
                                "properties"
                            ].items():
                                if "description" in prop_info:
                                    parts.append(
                                        f"{prop_name}: {prop_info['description']}"
                                    )

                searchable_text = " ".join(parts)

                # Convert categories set of enums to list of strings for JSON serialization
                categories = getattr(block_instance, "categories", set())
                categories_list = (
                    [cat.value for cat in categories] if categories else []
                )

                items.append(
                    ContentItem(
                        content_id=block_id,
                        content_type=ContentType.BLOCK,
                        searchable_text=searchable_text,
                        metadata={
                            "name": getattr(block_instance, "name", ""),
                            "categories": categories_list,
                        },
                        user_id=None,  # Blocks are public
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process block {block_id}: {e}")
                continue

        return items

    async def get_stats(self) -> dict[str, int]:
        """Get statistics about block embedding coverage."""
        from backend.data.block import get_blocks

        all_blocks = get_blocks()

        # Filter out disabled blocks - they're not indexed
        enabled_block_ids = [
            block_id
            for block_id, block_cls in all_blocks.items()
            if not block_cls().disabled
        ]
        total_blocks = len(enabled_block_ids)

        if total_blocks == 0:
            return {"total": 0, "with_embeddings": 0, "without_embeddings": 0}

        block_ids = enabled_block_ids
        placeholders = ",".join([f"${i+1}" for i in range(len(block_ids))])

        embedded_result = await query_raw_with_schema(
            f"""
            SELECT COUNT(*) as count
            FROM {{schema_prefix}}"UnifiedContentEmbedding"
            WHERE "contentType" = 'BLOCK'::{{schema_prefix}}"ContentType"
            AND "contentId" = ANY(ARRAY[{placeholders}])
            """,
            *block_ids,
        )

        with_embeddings = embedded_result[0]["count"] if embedded_result else 0

        return {
            "total": total_blocks,
            "with_embeddings": with_embeddings,
            "without_embeddings": total_blocks - with_embeddings,
        }


@dataclass
class MarkdownSection:
    """Represents a section of a markdown document."""

    title: str  # Section heading text
    content: str  # Section content (including the heading line)
    level: int  # Heading level (1 for #, 2 for ##, etc.)
    index: int  # Section index within the document


class DocumentationHandler(ContentHandler):
    """Handler for documentation files (.md/.mdx).

    Chunks documents by markdown headings to create multiple embeddings per file.
    Each section (## heading) becomes a separate embedding for better retrieval.
    """

    @property
    def content_type(self) -> ContentType:
        return ContentType.DOCUMENTATION

    def _get_docs_root(self) -> Path:
        """Get the documentation root directory."""
        # content_handlers.py is at: backend/backend/api/features/store/content_handlers.py
        # Need to go up to project root then into docs/
        # In container: /app/autogpt_platform/backend/backend/api/features/store -> /app/docs
        # In development: /repo/autogpt_platform/backend/backend/api/features/store -> /repo/docs
        this_file = Path(
            __file__
        )  # .../backend/backend/api/features/store/content_handlers.py
        project_root = (
            this_file.parent.parent.parent.parent.parent.parent.parent
        )  # -> /app or /repo
        docs_root = project_root / "docs"
        return docs_root

    def _extract_doc_title(self, file_path: Path) -> str:
        """Extract the document title from a markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Try to extract title from first # heading
            for line in lines:
                if line.startswith("# "):
                    return line[2:].strip()

            # If no title found, use filename
            return file_path.stem.replace("-", " ").replace("_", " ").title()
        except Exception as e:
            logger.warning(f"Failed to read title from {file_path}: {e}")
            return file_path.stem.replace("-", " ").replace("_", " ").title()

    def _chunk_markdown_by_headings(
        self, file_path: Path, min_heading_level: int = 2
    ) -> list[MarkdownSection]:
        """
        Split a markdown file into sections based on headings.

        Args:
            file_path: Path to the markdown file
            min_heading_level: Minimum heading level to split on (default: 2 for ##)

        Returns:
            List of MarkdownSection objects, one per section.
            If no headings found, returns a single section with all content.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []

        lines = content.split("\n")
        sections: list[MarkdownSection] = []
        current_section_lines: list[str] = []
        current_title = ""
        current_level = 0
        section_index = 0
        doc_title = ""

        for line in lines:
            # Check if line is a heading
            if line.startswith("#"):
                # Count heading level
                level = 0
                for char in line:
                    if char == "#":
                        level += 1
                    else:
                        break

                heading_text = line[level:].strip()

                # Track document title (level 1 heading)
                if level == 1 and not doc_title:
                    doc_title = heading_text
                    # Don't create a section for just the title - add it to first section
                    current_section_lines.append(line)
                    continue

                # Check if this heading should start a new section
                if level >= min_heading_level:
                    # Save previous section if it has content
                    if current_section_lines:
                        section_content = "\n".join(current_section_lines).strip()
                        if section_content:
                            # Use doc title for first section if no specific title
                            title = current_title if current_title else doc_title
                            if not title:
                                title = file_path.stem.replace("-", " ").replace(
                                    "_", " "
                                )
                            sections.append(
                                MarkdownSection(
                                    title=title,
                                    content=section_content,
                                    level=current_level if current_level else 1,
                                    index=section_index,
                                )
                            )
                            section_index += 1

                    # Start new section
                    current_section_lines = [line]
                    current_title = heading_text
                    current_level = level
                else:
                    # Lower level heading (e.g., # when splitting on ##)
                    current_section_lines.append(line)
            else:
                current_section_lines.append(line)

        # Don't forget the last section
        if current_section_lines:
            section_content = "\n".join(current_section_lines).strip()
            if section_content:
                title = current_title if current_title else doc_title
                if not title:
                    title = file_path.stem.replace("-", " ").replace("_", " ")
                sections.append(
                    MarkdownSection(
                        title=title,
                        content=section_content,
                        level=current_level if current_level else 1,
                        index=section_index,
                    )
                )

        # If no sections were created (no headings found), create one section with all content
        if not sections and content.strip():
            title = (
                doc_title
                if doc_title
                else file_path.stem.replace("-", " ").replace("_", " ")
            )
            sections.append(
                MarkdownSection(
                    title=title,
                    content=content.strip(),
                    level=1,
                    index=0,
                )
            )

        return sections

    def _make_section_content_id(self, doc_path: str, section_index: int) -> str:
        """Create a unique content ID for a document section.

        Format: doc_path::section_index
        Example: 'platform/getting-started.md::0'
        """
        return f"{doc_path}::{section_index}"

    def _parse_section_content_id(self, content_id: str) -> tuple[str, int]:
        """Parse a section content ID back into doc_path and section_index.

        Returns: (doc_path, section_index)
        """
        if "::" in content_id:
            parts = content_id.rsplit("::", 1)
            return parts[0], int(parts[1])
        # Legacy format (whole document)
        return content_id, 0

    async def get_missing_items(self, batch_size: int) -> list[ContentItem]:
        """Fetch documentation sections without embeddings.

        Chunks each document by markdown headings and creates embeddings for each section.
        Content IDs use the format: 'path/to/doc.md::section_index'
        """
        docs_root = self._get_docs_root()

        if not docs_root.exists():
            logger.warning(f"Documentation root not found: {docs_root}")
            return []

        # Find all .md and .mdx files
        all_docs = list(docs_root.rglob("*.md")) + list(docs_root.rglob("*.mdx"))

        if not all_docs:
            return []

        # Build list of all sections from all documents
        all_sections: list[tuple[str, Path, MarkdownSection]] = []
        for doc_file in all_docs:
            doc_path = str(doc_file.relative_to(docs_root))
            sections = self._chunk_markdown_by_headings(doc_file)
            for section in sections:
                all_sections.append((doc_path, doc_file, section))

        if not all_sections:
            return []

        # Generate content IDs for all sections
        section_content_ids = [
            self._make_section_content_id(doc_path, section.index)
            for doc_path, _, section in all_sections
        ]

        # Check which ones have embeddings
        placeholders = ",".join([f"${i+1}" for i in range(len(section_content_ids))])
        existing_result = await query_raw_with_schema(
            f"""
            SELECT "contentId"
            FROM {{schema_prefix}}"UnifiedContentEmbedding"
            WHERE "contentType" = 'DOCUMENTATION'::{{schema_prefix}}"ContentType"
            AND "contentId" = ANY(ARRAY[{placeholders}])
            """,
            *section_content_ids,
        )

        existing_ids = {row["contentId"] for row in existing_result}

        # Filter to missing sections
        missing_sections = [
            (doc_path, doc_file, section, content_id)
            for (doc_path, doc_file, section), content_id in zip(
                all_sections, section_content_ids
            )
            if content_id not in existing_ids
        ]

        # Convert to ContentItem (up to batch_size)
        items = []
        for doc_path, doc_file, section, content_id in missing_sections[:batch_size]:
            try:
                # Get document title for context
                doc_title = self._extract_doc_title(doc_file)

                # Build searchable text with context
                # Include doc title and section title for better search relevance
                searchable_text = f"{doc_title} - {section.title}\n\n{section.content}"

                items.append(
                    ContentItem(
                        content_id=content_id,
                        content_type=ContentType.DOCUMENTATION,
                        searchable_text=searchable_text,
                        metadata={
                            "doc_title": doc_title,
                            "section_title": section.title,
                            "section_index": section.index,
                            "heading_level": section.level,
                            "path": doc_path,
                        },
                        user_id=None,  # Documentation is public
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process section {content_id}: {e}")
                continue

        return items

    def _get_all_section_content_ids(self, docs_root: Path) -> set[str]:
        """Get all current section content IDs from the docs directory.

        Used for stats and cleanup to know what sections should exist.
        """
        all_docs = list(docs_root.rglob("*.md")) + list(docs_root.rglob("*.mdx"))
        content_ids = set()

        for doc_file in all_docs:
            doc_path = str(doc_file.relative_to(docs_root))
            sections = self._chunk_markdown_by_headings(doc_file)
            for section in sections:
                content_ids.add(self._make_section_content_id(doc_path, section.index))

        return content_ids

    async def get_stats(self) -> dict[str, int]:
        """Get statistics about documentation embedding coverage.

        Counts sections (not documents) since each section gets its own embedding.
        """
        docs_root = self._get_docs_root()

        if not docs_root.exists():
            return {"total": 0, "with_embeddings": 0, "without_embeddings": 0}

        # Get all section content IDs
        all_section_ids = self._get_all_section_content_ids(docs_root)
        total_sections = len(all_section_ids)

        if total_sections == 0:
            return {"total": 0, "with_embeddings": 0, "without_embeddings": 0}

        # Count embeddings in database for DOCUMENTATION type
        embedded_result = await query_raw_with_schema(
            """
            SELECT COUNT(*) as count
            FROM {schema_prefix}"UnifiedContentEmbedding"
            WHERE "contentType" = 'DOCUMENTATION'::{schema_prefix}"ContentType"
            """
        )

        with_embeddings = embedded_result[0]["count"] if embedded_result else 0

        return {
            "total": total_sections,
            "with_embeddings": with_embeddings,
            "without_embeddings": total_sections - with_embeddings,
        }


# Content handler registry
CONTENT_HANDLERS: dict[ContentType, ContentHandler] = {
    ContentType.STORE_AGENT: StoreAgentHandler(),
    ContentType.BLOCK: BlockHandler(),
    ContentType.DOCUMENTATION: DocumentationHandler(),
}
