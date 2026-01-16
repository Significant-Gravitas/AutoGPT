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
        total_blocks = len(all_blocks)

        if total_blocks == 0:
            return {"total": 0, "with_embeddings": 0, "without_embeddings": 0}

        block_ids = list(all_blocks.keys())
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


class DocumentationHandler(ContentHandler):
    """Handler for documentation files (.md/.mdx)."""

    @property
    def content_type(self) -> ContentType:
        return ContentType.DOCUMENTATION

    def _get_docs_root(self) -> Path:
        """Get the documentation root directory."""
        # content_handlers.py is at: backend/backend/api/features/store/content_handlers.py
        # Need to go up to project root then into docs/
        # In container: /app/autogpt_platform/backend/backend/api/features/store -> /app/docs
        # In development: /repo/autogpt_platform/backend/backend/api/features/store -> /repo/docs
        this_file = Path(__file__)  # .../backend/backend/api/features/store/content_handlers.py
        project_root = this_file.parent.parent.parent.parent.parent.parent.parent  # -> /app or /repo
        docs_root = project_root / "docs"
        return docs_root

    def _extract_title_and_content(self, file_path: Path) -> tuple[str, str]:
        """Extract title and content from markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Try to extract title from first # heading
            lines = content.split("\n")
            title = ""
            body_lines = []

            for line in lines:
                if line.startswith("# ") and not title:
                    title = line[2:].strip()
                else:
                    body_lines.append(line)

            # If no title found, use filename
            if not title:
                title = file_path.stem.replace("-", " ").replace("_", " ").title()

            body = "\n".join(body_lines)

            return title, body
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return file_path.stem, ""

    async def get_missing_items(self, batch_size: int) -> list[ContentItem]:
        """Fetch documentation files without embeddings."""
        docs_root = self._get_docs_root()

        if not docs_root.exists():
            logger.warning(f"Documentation root not found: {docs_root}")
            return []

        # Find all .md and .mdx files
        all_docs = list(docs_root.rglob("*.md")) + list(docs_root.rglob("*.mdx"))

        # Get relative paths for content IDs
        doc_paths = [str(doc.relative_to(docs_root)) for doc in all_docs]

        if not doc_paths:
            return []

        # Check which ones have embeddings
        placeholders = ",".join([f"${i+1}" for i in range(len(doc_paths))])
        existing_result = await query_raw_with_schema(
            f"""
            SELECT "contentId"
            FROM {{schema_prefix}}"UnifiedContentEmbedding"
            WHERE "contentType" = 'DOCUMENTATION'::{{schema_prefix}}"ContentType"
            AND "contentId" = ANY(ARRAY[{placeholders}])
            """,
            *doc_paths,
        )

        existing_ids = {row["contentId"] for row in existing_result}
        missing_docs = [
            (doc_path, doc_file)
            for doc_path, doc_file in zip(doc_paths, all_docs)
            if doc_path not in existing_ids
        ]

        # Convert to ContentItem
        items = []
        for doc_path, doc_file in missing_docs[:batch_size]:
            try:
                title, content = self._extract_title_and_content(doc_file)

                # Build searchable text
                searchable_text = f"{title} {content}"

                items.append(
                    ContentItem(
                        content_id=doc_path,
                        content_type=ContentType.DOCUMENTATION,
                        searchable_text=searchable_text,
                        metadata={
                            "title": title,
                            "path": doc_path,
                        },
                        user_id=None,  # Documentation is public
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process doc {doc_path}: {e}")
                continue

        return items

    async def get_stats(self) -> dict[str, int]:
        """Get statistics about documentation embedding coverage."""
        docs_root = self._get_docs_root()

        if not docs_root.exists():
            return {"total": 0, "with_embeddings": 0, "without_embeddings": 0}

        # Count all .md and .mdx files
        all_docs = list(docs_root.rglob("*.md")) + list(docs_root.rglob("*.mdx"))
        total_docs = len(all_docs)

        if total_docs == 0:
            return {"total": 0, "with_embeddings": 0, "without_embeddings": 0}

        doc_paths = [str(doc.relative_to(docs_root)) for doc in all_docs]
        placeholders = ",".join([f"${i+1}" for i in range(len(doc_paths))])

        embedded_result = await query_raw_with_schema(
            f"""
            SELECT COUNT(*) as count
            FROM {{schema_prefix}}"UnifiedContentEmbedding"
            WHERE "contentType" = 'DOCUMENTATION'::{{schema_prefix}}"ContentType"
            AND "contentId" = ANY(ARRAY[{placeholders}])
            """,
            *doc_paths,
        )

        with_embeddings = embedded_result[0]["count"] if embedded_result else 0

        return {
            "total": total_docs,
            "with_embeddings": with_embeddings,
            "without_embeddings": total_docs - with_embeddings,
        }


# Content handler registry
CONTENT_HANDLERS: dict[ContentType, ContentHandler] = {
    ContentType.STORE_AGENT: StoreAgentHandler(),
    ContentType.BLOCK: BlockHandler(),
    ContentType.DOCUMENTATION: DocumentationHandler(),
}
