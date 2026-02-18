"""
WorkspaceManager for managing user workspace file operations.

This module provides a high-level interface for workspace file operations,
combining the storage backend and database layer.
"""

import logging
import mimetypes
import uuid
from typing import Optional

from prisma.errors import UniqueViolationError

from backend.data.db_accessors import workspace_db
from backend.data.workspace import WorkspaceFile
from backend.util.settings import Config
from backend.util.virus_scanner import scan_content_safe
from backend.util.workspace_storage import compute_file_checksum, get_workspace_storage

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Manages workspace file operations.

    Combines storage backend operations with database record management.
    Supports session-scoped file segmentation where files are stored in
    session-specific virtual paths: /sessions/{session_id}/{filename}
    """

    def __init__(
        self, user_id: str, workspace_id: str, session_id: Optional[str] = None
    ):
        """
        Initialize WorkspaceManager.

        Args:
            user_id: The user's ID
            workspace_id: The workspace ID
            session_id: Optional session ID for session-scoped file access
        """
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.session_id = session_id
        # Session path prefix for file isolation
        self.session_path = f"/sessions/{session_id}" if session_id else ""

    def _resolve_path(self, path: str) -> str:
        """
        Resolve a path, defaulting to session folder if session_id is set.

        Cross-session access is allowed by explicitly using /sessions/other-session-id/...

        Args:
            path: Virtual path (e.g., "/file.txt" or "/sessions/abc123/file.txt")

        Returns:
            Resolved path with session prefix if applicable
        """
        # If path explicitly references a session folder, use it as-is
        if path.startswith("/sessions/"):
            return path

        # If we have a session context, prepend session path
        if self.session_path:
            # Normalize the path
            if not path.startswith("/"):
                path = f"/{path}"
            return f"{self.session_path}{path}"

        # No session context, use path as-is
        return path if path.startswith("/") else f"/{path}"

    def _get_effective_path(
        self, path: Optional[str], include_all_sessions: bool
    ) -> Optional[str]:
        """
        Get effective path for list/count operations based on session context.

        Args:
            path: Optional path prefix to filter
            include_all_sessions: If True, don't apply session scoping

        Returns:
            Effective path prefix for database query
        """
        if include_all_sessions:
            # Normalize path to ensure leading slash (stored paths are normalized)
            if path is not None and not path.startswith("/"):
                return f"/{path}"
            return path
        elif path is not None:
            # Resolve the provided path with session scoping
            return self._resolve_path(path)
        elif self.session_path:
            # Default to session folder with trailing slash to prevent prefix collisions
            # e.g., "/sessions/abc" should not match "/sessions/abc123"
            return self.session_path.rstrip("/") + "/"
        else:
            # No session context, use path as-is
            return path

    async def read_file(self, path: str) -> bytes:
        """
        Read file from workspace by virtual path.

        When session_id is set, paths are resolved relative to the session folder
        unless they explicitly reference /sessions/...

        Args:
            path: Virtual path (e.g., "/documents/report.pdf")

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        db = workspace_db()
        resolved_path = self._resolve_path(path)
        file = await db.get_workspace_file_by_path(self.workspace_id, resolved_path)
        if file is None:
            raise FileNotFoundError(f"File not found at path: {resolved_path}")

        storage = await get_workspace_storage()
        return await storage.retrieve(file.storage_path)

    async def read_file_by_id(self, file_id: str) -> bytes:
        """
        Read file from workspace by file ID.

        Args:
            file_id: The file's ID

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        db = workspace_db()
        file = await db.get_workspace_file(file_id, self.workspace_id)
        if file is None:
            raise FileNotFoundError(f"File not found: {file_id}")

        storage = await get_workspace_storage()
        return await storage.retrieve(file.storage_path)

    async def write_file(
        self,
        content: bytes,
        filename: str,
        path: Optional[str] = None,
        mime_type: Optional[str] = None,
        overwrite: bool = False,
    ) -> WorkspaceFile:
        """
        Write file to workspace.

        When session_id is set, files are written to /sessions/{session_id}/...
        by default. Use explicit /sessions/... paths for cross-session access.

        Args:
            content: File content as bytes
            filename: Filename for the file
            path: Virtual path (defaults to "/{filename}", session-scoped if session_id set)
            mime_type: MIME type (auto-detected if not provided)
            overwrite: Whether to overwrite existing file at path

        Returns:
            Created WorkspaceFile instance

        Raises:
            ValueError: If file exceeds size limit or path already exists
        """
        # Enforce file size limit
        max_file_size = Config().max_file_size_mb * 1024 * 1024
        if len(content) > max_file_size:
            raise ValueError(
                f"File too large: {len(content)} bytes exceeds "
                f"{Config().max_file_size_mb}MB limit"
            )

        # Virus scan content before persisting (defense in depth)
        await scan_content_safe(content, filename=filename)

        # Determine path with session scoping
        if path is None:
            path = f"/{filename}"
        elif not path.startswith("/"):
            path = f"/{path}"

        # Resolve path with session prefix
        path = self._resolve_path(path)

        # Check if file exists at path (only error for non-overwrite case)
        # For overwrite=True, we let the write proceed and handle via UniqueViolationError
        # This ensures the new file is written to storage BEFORE the old one is deleted,
        # preventing data loss if the new write fails
        db = workspace_db()

        if not overwrite:
            existing = await db.get_workspace_file_by_path(self.workspace_id, path)
            if existing is not None:
                raise ValueError(f"File already exists at path: {path}")

        # Auto-detect MIME type if not provided
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or "application/octet-stream"

        # Compute checksum
        checksum = compute_file_checksum(content)

        # Generate unique file ID for storage
        file_id = str(uuid.uuid4())

        # Store file in storage backend
        storage = await get_workspace_storage()
        storage_path = await storage.store(
            workspace_id=self.workspace_id,
            file_id=file_id,
            filename=filename,
            content=content,
        )

        # Create database record - handle race condition where another request
        # created a file at the same path between our check and create
        try:
            file = await db.create_workspace_file(
                workspace_id=self.workspace_id,
                file_id=file_id,
                name=filename,
                path=path,
                storage_path=storage_path,
                mime_type=mime_type,
                size_bytes=len(content),
                checksum=checksum,
            )
        except UniqueViolationError:
            # Race condition: another request created a file at this path
            if overwrite:
                # Re-fetch and delete the conflicting file, then retry
                existing = await db.get_workspace_file_by_path(self.workspace_id, path)
                if existing:
                    await self.delete_file(existing.id)
                # Retry the create - if this also fails, clean up storage file
                try:
                    file = await db.create_workspace_file(
                        workspace_id=self.workspace_id,
                        file_id=file_id,
                        name=filename,
                        path=path,
                        storage_path=storage_path,
                        mime_type=mime_type,
                        size_bytes=len(content),
                        checksum=checksum,
                    )
                except Exception:
                    # Clean up orphaned storage file on retry failure
                    try:
                        await storage.delete(storage_path)
                    except Exception as e:
                        logger.warning(f"Failed to clean up orphaned storage file: {e}")
                    raise
            else:
                # Clean up the orphaned storage file before raising
                try:
                    await storage.delete(storage_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up orphaned storage file: {e}")
                raise ValueError(f"File already exists at path: {path}")
        except Exception:
            # Any other database error (connection, validation, etc.) - clean up storage
            try:
                await storage.delete(storage_path)
            except Exception as e:
                logger.warning(f"Failed to clean up orphaned storage file: {e}")
            raise

        logger.info(
            f"Wrote file {file.id} ({filename}) to workspace {self.workspace_id} "
            f"at path {path}, size={len(content)} bytes"
        )

        return file

    async def list_files(
        self,
        path: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        include_all_sessions: bool = False,
    ) -> list[WorkspaceFile]:
        """
        List files in workspace.

        When session_id is set and include_all_sessions is False (default),
        only files in the current session's folder are listed.

        Args:
            path: Optional path prefix to filter (e.g., "/documents/")
            limit: Maximum number of files to return
            offset: Number of files to skip
            include_all_sessions: If True, list files from all sessions.
                                  If False (default), only list current session's files.

        Returns:
            List of WorkspaceFile instances
        """
        effective_path = self._get_effective_path(path, include_all_sessions)
        db = workspace_db()

        return await db.list_workspace_files(
            workspace_id=self.workspace_id,
            path_prefix=effective_path,
            limit=limit,
            offset=offset,
        )

    async def delete_file(self, file_id: str) -> bool:
        """
        Delete a file (soft-delete).

        Args:
            file_id: The file's ID

        Returns:
            True if deleted, False if not found
        """
        db = workspace_db()
        file = await db.get_workspace_file(file_id, self.workspace_id)
        if file is None:
            return False

        # Delete from storage
        storage = await get_workspace_storage()
        try:
            await storage.delete(file.storage_path)
        except Exception as e:
            logger.warning(f"Failed to delete file from storage: {e}")
            # Continue with database soft-delete even if storage delete fails

        # Soft-delete database record
        result = await db.soft_delete_workspace_file(file_id, self.workspace_id)
        return result is not None

    async def get_download_url(self, file_id: str, expires_in: int = 3600) -> str:
        """
        Get download URL for a file.

        Args:
            file_id: The file's ID
            expires_in: URL expiration in seconds (default 1 hour)

        Returns:
            Download URL (signed URL for GCS, API endpoint for local)

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        db = workspace_db()
        file = await db.get_workspace_file(file_id, self.workspace_id)
        if file is None:
            raise FileNotFoundError(f"File not found: {file_id}")

        storage = await get_workspace_storage()
        return await storage.get_download_url(file.storage_path, expires_in)

    async def get_file_info(self, file_id: str) -> Optional[WorkspaceFile]:
        """
        Get file metadata.

        Args:
            file_id: The file's ID

        Returns:
            WorkspaceFile instance or None
        """
        db = workspace_db()
        return await db.get_workspace_file(file_id, self.workspace_id)

    async def get_file_info_by_path(self, path: str) -> Optional[WorkspaceFile]:
        """
        Get file metadata by path.

        When session_id is set, paths are resolved relative to the session folder
        unless they explicitly reference /sessions/...

        Args:
            path: Virtual path

        Returns:
            WorkspaceFile instance or None
        """
        db = workspace_db()
        resolved_path = self._resolve_path(path)
        return await db.get_workspace_file_by_path(self.workspace_id, resolved_path)

    async def get_file_count(
        self,
        path: Optional[str] = None,
        include_all_sessions: bool = False,
    ) -> int:
        """
        Get number of files in workspace.

        When session_id is set and include_all_sessions is False (default),
        only counts files in the current session's folder.

        Args:
            path: Optional path prefix to filter (e.g., "/documents/")
            include_all_sessions: If True, count all files in workspace.
                                  If False (default), only count current session's files.

        Returns:
            Number of files
        """
        effective_path = self._get_effective_path(path, include_all_sessions)
        db = workspace_db()

        return await db.count_workspace_files(
            self.workspace_id, path_prefix=effective_path
        )
