"""
WorkspaceManager for managing user workspace file operations.

This module provides a high-level interface for workspace file operations,
combining the storage backend and database layer.
"""

import logging
import mimetypes
import uuid
from typing import Optional

from prisma.enums import WorkspaceFileSource
from prisma.models import UserWorkspaceFile

from backend.data.workspace import (
    count_workspace_files,
    create_workspace_file,
    get_workspace_file,
    get_workspace_file_by_path,
    list_workspace_files,
    soft_delete_workspace_file,
    workspace_file_exists,
)
from backend.util.workspace_storage import compute_file_checksum, get_workspace_storage

logger = logging.getLogger(__name__)

# Maximum file size: 100MB per file
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


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
        resolved_path = self._resolve_path(path)
        file = await get_workspace_file_by_path(self.workspace_id, resolved_path)
        if file is None:
            raise FileNotFoundError(f"File not found at path: {resolved_path}")

        storage = await get_workspace_storage()
        return await storage.retrieve(file.storagePath)

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
        file = await get_workspace_file(file_id, self.workspace_id)
        if file is None:
            raise FileNotFoundError(f"File not found: {file_id}")

        storage = await get_workspace_storage()
        return await storage.retrieve(file.storagePath)

    async def write_file(
        self,
        content: bytes,
        filename: str,
        path: Optional[str] = None,
        mime_type: Optional[str] = None,
        source: WorkspaceFileSource = WorkspaceFileSource.UPLOAD,
        source_exec_id: Optional[str] = None,
        source_session_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> UserWorkspaceFile:
        """
        Write file to workspace.

        When session_id is set, files are written to /sessions/{session_id}/...
        by default. Use explicit /sessions/... paths for cross-session access.

        Args:
            content: File content as bytes
            filename: Filename for the file
            path: Virtual path (defaults to "/{filename}", session-scoped if session_id set)
            mime_type: MIME type (auto-detected if not provided)
            source: How the file was created
            source_exec_id: Graph execution ID if from execution
            source_session_id: Chat session ID if from CoPilot
            overwrite: Whether to overwrite existing file at path

        Returns:
            Created UserWorkspaceFile instance

        Raises:
            ValueError: If file exceeds size limit or path already exists
        """
        # Enforce file size limit
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {len(content)} bytes exceeds "
                f"{MAX_FILE_SIZE_BYTES // (1024*1024)}MB limit"
            )

        # Determine path with session scoping
        if path is None:
            path = f"/{filename}"
        elif not path.startswith("/"):
            path = f"/{path}"

        # Resolve path with session prefix
        path = self._resolve_path(path)

        # Check if file exists at path
        existing = await get_workspace_file_by_path(self.workspace_id, path)
        if existing is not None:
            if overwrite:
                # Delete existing file first
                await self.delete_file(existing.id)
            else:
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

        # Create database record
        file = await create_workspace_file(
            workspace_id=self.workspace_id,
            name=filename,
            path=path,
            storage_path=storage_path,
            mime_type=mime_type,
            size_bytes=len(content),
            checksum=checksum,
            source=source,
            source_exec_id=source_exec_id,
            source_session_id=source_session_id,
        )

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
    ) -> list[UserWorkspaceFile]:
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
            List of UserWorkspaceFile instances
        """
        # Determine the effective path prefix
        if include_all_sessions:
            # Use provided path as-is (or None for all files)
            effective_path = path
        elif path is not None:
            # Resolve the provided path with session scoping
            effective_path = self._resolve_path(path)
        elif self.session_path:
            # Default to session folder
            effective_path = self.session_path
        else:
            # No session context, list all
            effective_path = path

        return await list_workspace_files(
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
        file = await get_workspace_file(file_id, self.workspace_id)
        if file is None:
            return False

        # Delete from storage
        storage = await get_workspace_storage()
        try:
            await storage.delete(file.storagePath)
        except Exception as e:
            logger.warning(f"Failed to delete file from storage: {e}")
            # Continue with database soft-delete even if storage delete fails

        # Soft-delete database record
        result = await soft_delete_workspace_file(file_id, self.workspace_id)
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
        file = await get_workspace_file(file_id, self.workspace_id)
        if file is None:
            raise FileNotFoundError(f"File not found: {file_id}")

        storage = await get_workspace_storage()
        return await storage.get_download_url(file.storagePath, expires_in)

    async def get_file_info(self, file_id: str) -> Optional[UserWorkspaceFile]:
        """
        Get file metadata.

        Args:
            file_id: The file's ID

        Returns:
            UserWorkspaceFile instance or None
        """
        return await get_workspace_file(file_id, self.workspace_id)

    async def get_file_info_by_path(self, path: str) -> Optional[UserWorkspaceFile]:
        """
        Get file metadata by path.

        When session_id is set, paths are resolved relative to the session folder
        unless they explicitly reference /sessions/...

        Args:
            path: Virtual path

        Returns:
            UserWorkspaceFile instance or None
        """
        resolved_path = self._resolve_path(path)
        return await get_workspace_file_by_path(self.workspace_id, resolved_path)

    async def file_exists(self, path: str) -> bool:
        """
        Check if a file exists at the given path.

        When session_id is set, paths are resolved relative to the session folder
        unless they explicitly reference /sessions/...

        Args:
            path: Virtual path

        Returns:
            True if file exists
        """
        resolved_path = self._resolve_path(path)
        return await workspace_file_exists(self.workspace_id, resolved_path)

    async def get_file_count(self) -> int:
        """
        Get number of files in workspace.

        Returns:
            Number of files
        """
        return await count_workspace_files(self.workspace_id)
