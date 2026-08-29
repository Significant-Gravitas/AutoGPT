"""GetDocPageTool - Fetch full content of a documentation page."""

import asyncio
import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.util.docs import get_docs_root, make_doc_url

from .base import BaseTool
from .models import DocPageResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class GetDocPageTool(BaseTool):
    """Tool for fetching full content of a documentation page."""

    @property
    def name(self) -> str:
        return "get_doc_page"

    @property
    def description(self) -> str:
        return (
            "Read full documentation page content by path (from search_docs results)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Doc file path (e.g. 'platform/block-sdk-guide.md').",
                },
            },
            "required": ["path"],
        }

    @property
    def requires_auth(self) -> bool:
        return False  # Documentation is public

    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract title from markdown content."""
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
        return fallback

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        path: str = "",
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
        path = path.strip()
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

        docs_root = get_docs_root()
        if docs_root is None:
            return ErrorResponse(
                message="Documentation is not available in this deployment.",
                error="docs_unavailable",
                session_id=session_id,
            )
        # Containment before existence: a uniform error for out-of-root
        # paths avoids leaking whether a file exists outside docs_root.
        try:
            # resolve() both sides: get_docs_root guarantees a resolved root,
            # but the guard stays self-contained rather than coupling a
            # security boundary to another module's invariant (idempotent,
            # one metadata syscall). All later access goes through the
            # resolved path so the validated path is the one read.
            full_path = (docs_root / path).resolve()
            full_path.relative_to(docs_root.resolve())
        except ValueError:
            return ErrorResponse(
                message="Invalid documentation path.",
                error="invalid_path",
                session_id=session_id,
            )

        if not full_path.exists():
            return ErrorResponse(
                message=f"Documentation page not found: {path}",
                error="not_found",
                session_id=session_id,
            )

        try:
            content = await asyncio.to_thread(full_path.read_text, encoding="utf-8")
            title = self._extract_title(content, path)

            return DocPageResponse(
                message=f"Retrieved documentation page: {title}",
                title=title,
                path=path,
                content=content,
                doc_url=make_doc_url(path),
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"Failed to read documentation page {path}: {e}")
            return ErrorResponse(
                message=f"Failed to read documentation page: {str(e)}",
                error="read_failed",
                session_id=session_id,
            )
