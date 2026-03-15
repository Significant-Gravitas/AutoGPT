"""GetDocPageTool - Fetch full content of a documentation page."""

import logging
from pathlib import Path
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import DocPageResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)

# Base URL for documentation (can be configured)
DOCS_BASE_URL = "https://docs.agpt.co"


class GetDocPageTool(BaseTool):
    """Tool for fetching full content of a documentation page."""

    @property
    def name(self) -> str:
        return "get_doc_page"

    @property
    def description(self) -> str:
        return (
            "Get the full content of a documentation page by its path. "
            "Use this after search_docs to read the complete content of a relevant page."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "The path to the documentation file, as returned by search_docs. "
                        "Example: 'platform/block-sdk-guide.md'"
                    ),
                },
            },
            "required": ["path"],
        }

    @property
    def requires_auth(self) -> bool:
        return False  # Documentation is public

    def _get_docs_root(self) -> Path:
        """Get the documentation root directory."""
        this_file = Path(__file__)
        project_root = this_file.parent.parent.parent.parent.parent.parent.parent.parent
        return project_root / "docs"

    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract title from markdown content."""
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
        return fallback

    def _make_doc_url(self, path: str) -> str:
        """Create a URL for a documentation page."""
        url_path = path.rsplit(".", 1)[0] if "." in path else path
        return f"{DOCS_BASE_URL}/{url_path}"

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Fetch full content of a documentation page.

        Args:
            user_id: User ID (not required for docs)
            session: Chat session
            path: Path to the documentation file

        Returns:
            DocPageResponse: Full document content
            ErrorResponse: Error message
        """
        path = kwargs.get("path", "").strip()
        session_id = session.session_id if session else None

        if not path:
            return ErrorResponse(
                message="Please provide a documentation path.",
                error="Missing path parameter",
                session_id=session_id,
            )

        # Sanitize path to prevent directory traversal
        if ".." in path or path.startswith("/"):
            return ErrorResponse(
                message="Invalid documentation path.",
                error="invalid_path",
                session_id=session_id,
            )

        docs_root = self._get_docs_root()
        full_path = docs_root / path

        if not full_path.exists():
            return ErrorResponse(
                message=f"Documentation page not found: {path}",
                error="not_found",
                session_id=session_id,
            )

        # Ensure the path is within docs root
        try:
            full_path.resolve().relative_to(docs_root.resolve())
        except ValueError:
            return ErrorResponse(
                message="Invalid documentation path.",
                error="invalid_path",
                session_id=session_id,
            )

        try:
            content = full_path.read_text(encoding="utf-8")
            title = self._extract_title(content, path)

            return DocPageResponse(
                message=f"Retrieved documentation page: {title}",
                title=title,
                path=path,
                content=content,
                doc_url=self._make_doc_url(path),
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"Failed to read documentation page {path}: {e}")
            return ErrorResponse(
                message=f"Failed to read documentation page: {str(e)}",
                error="read_failed",
                session_id=session_id,
            )
