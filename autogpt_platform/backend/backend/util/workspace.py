"""
WorkspaceManager for managing user workspace file operations.

This module provides a high-level interface for workspace file operations,
combining the storage backend and database layer.
"""

import asyncio
import inspect
import logging
import mimetypes
import uuid
from typing import Awaitable, Optional, TypeVar

from prisma.errors import UniqueViolationError

from backend.copilot.rate_limit import get_workspace_storage_limit_bytes
from backend.data.db_accessors import (
    LiveResourceAccessRevoked,
    live_resource_lease,
    require_exact_chat_session_scope,
    workspace_db,
)
from backend.data.tenancy import ResourceAccess
from backend.data.workspace import WorkspaceFile
from backend.util.settings import Config
from backend.util.virus_scanner import scan_content_safe
from backend.util.workspace_storage import compute_file_checksum, get_workspace_storage


def format_bytes(n: int) -> str:
    """Format bytes as a human-readable string (e.g. 250 MB, 1.0 GB)."""
    KB, MB, GB = 1024, 1024**2, 1024**3
    if n < KB:
        return f"{n} B"
    if n < MB:
        kb = round(n / KB)
        return f"{n / MB:.1f} MB" if kb >= 1024 else f"{kb} KB"
    if n < GB:
        mb = round(n / MB)
        return f"{n / GB:.1f} GB" if mb >= 1024 else f"{mb} MB"
    return f"{n / GB:.1f} GB"


logger = logging.getLogger(__name__)
T = TypeVar("T")


class WorkspaceManager:
    """
    Manages workspace file operations.

    Combines storage backend operations with database record management.
    Supports session-scoped file segmentation where files are stored in
    session-specific virtual paths: /sessions/{session_id}/{filename}
    """

    def __init__(
        self,
        user_id: str,
        workspace_id: str,
        session_id: Optional[str] = None,
        *,
        organization_id: str | None = None,
        team_id: str | None = None,
        execution_id: str | None = None,
        access: ResourceAccess = "execute",
        user_global_config: bool = False,
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
        self.organization_id = organization_id
        self.team_id = team_id
        self.execution_id = execution_id
        self.access: ResourceAccess = access
        self.user_global_config = user_global_config
        if user_global_config and any(
            (organization_id, team_id, session_id, execution_id)
        ):
            raise ValueError("User-global workspace files cannot carry tenant scope")
        # Session path prefix for file isolation
        self.session_path = f"/sessions/{session_id}" if session_id else ""

    def _scope_kwargs(self) -> dict:
        return {
            "organization_id": self.organization_id,
            "team_id": self.team_id,
            "user_global_config": self.user_global_config,
        }

    def _matches_session(self, file: WorkspaceFile | None) -> bool:
        return file is not None and (
            self.session_id is None or file.session_id == self.session_id
        )

    async def _run_scoped(self, action: Awaitable[T]) -> T:
        if self.user_global_config or self.organization_id is None:
            return await action
        async with live_resource_lease(
            self.user_id,
            self.organization_id,
            self.team_id,
            self.access,
        ) as guard:
            if not guard:
                if inspect.iscoroutine(action):
                    action.close()
                raise LiveResourceAccessRevoked("workspace_access_revoked")
            if self.session_id is not None:
                await require_exact_chat_session_scope(
                    self.session_id,
                    self.user_id,
                    self.organization_id,
                    self.team_id,
                )
            return await guard.run(action)

    def _resolve_path(self, path: str) -> str:
        """
        Resolve a path, defaulting to session folder if session_id is set.

        Cross-session access is allowed by explicitly using /sessions/other-session-id/...

        Args:
            path: Virtual path (e.g., "/file.txt" or "/sessions/abc123/file.txt")

        Returns:
            Resolved path with session prefix if applicable
        """
        # Explicit session paths must stay inside the manager's persisted session.
        if path.startswith("/sessions/"):
            if self.session_path and not path.startswith(self.session_path + "/"):
                raise ValueError("Cross-session workspace paths are not allowed")
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
        if include_all_sessions and self.session_id is not None:
            raise ValueError("Session-scoped managers cannot access other sessions")
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
        return await self._run_scoped(self._read_file(path))

    async def _read_file(self, path: str) -> bytes:
        db = workspace_db()
        resolved_path = self._resolve_path(path)
        file = await db.get_workspace_file_by_path(
            self.workspace_id,
            resolved_path,
            **self._scope_kwargs(),
        )
        if not self._matches_session(file):
            raise FileNotFoundError(f"File not found at path: {resolved_path}")
        assert file is not None

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
        return await self._run_scoped(self._read_file_by_id(file_id))

    async def _read_file_by_id(self, file_id: str) -> bytes:
        db = workspace_db()
        file = await db.get_workspace_file(
            file_id,
            self.workspace_id,
            **self._scope_kwargs(),
        )
        if not self._matches_session(file):
            raise FileNotFoundError(f"File not found: {file_id}")
        assert file is not None

        storage = await get_workspace_storage()
        return await storage.retrieve(file.storage_path)

    async def write_file(
        self,
        content: bytes,
        filename: str,
        path: Optional[str] = None,
        mime_type: Optional[str] = None,
        overwrite: bool = False,
        metadata: Optional[dict] = None,
    ) -> WorkspaceFile:
        return await self._run_scoped(
            self._write_file(
                content,
                filename,
                path=path,
                mime_type=mime_type,
                overwrite=overwrite,
                metadata=metadata,
            )
        )

    async def _write_file(
        self,
        content: bytes,
        filename: str,
        path: Optional[str] = None,
        mime_type: Optional[str] = None,
        overwrite: bool = False,
        metadata: Optional[dict] = None,
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
            metadata: Optional metadata dict (e.g., origin tracking)

        Returns:
            Created WorkspaceFile instance

        Raises:
            ValueError: If file exceeds size limit, exceeds the user's
                tier-based storage quota, or path already exists.
            VirusDetectedError: If the virus scanner flagged the content as
                malicious. Callers that want a generic failure should still
                catch ``Exception``, but a specific handler lets you log this
                as a security event rather than a generic write failure.
            VirusScanError: If the virus scanner is unreachable or otherwise
                fails. Distinct from ``VirusDetectedError`` — typically maps
                to a 5xx response (infrastructure issue, retry-able).
        """
        # Enforce file size limit
        max_file_size = Config().max_file_size_mb * 1024 * 1024
        if len(content) > max_file_size:
            raise ValueError(
                f"File too large: {len(content)} bytes exceeds "
                f"{Config().max_file_size_mb}MB limit"
            )

        # Determine path with session scoping (needed before quota check for overwrites)
        if path is None:
            path = f"/{filename}"
        elif not path.startswith("/"):
            path = f"/{path}"

        # Resolve path with session prefix
        path = self._resolve_path(path)

        # Enforce per-user workspace storage quota (tier-based).
        # For overwrites, subtract the existing file's size so replacing a file
        # with a same-size or smaller file is not rejected near the cap.
        storage_limit, current_usage = await asyncio.gather(
            get_workspace_storage_limit_bytes(self.user_id),
            workspace_db().get_workspace_total_size(self.workspace_id, all_scopes=True),
        )
        if overwrite:
            db = workspace_db()
            existing = await db.get_workspace_file_by_path(
                self.workspace_id,
                path,
                **self._scope_kwargs(),
            )
            if existing is not None:
                current_usage = max(0, current_usage - existing.size_bytes)

        projected_usage = current_usage + len(content)
        if storage_limit > 0 and projected_usage > storage_limit:
            raise ValueError(
                f"Storage limit exceeded. "
                f"You've used {format_bytes(current_usage)} of your "
                f"{format_bytes(storage_limit)} quota. "
                f"Delete some files or upgrade your plan for more storage."
            )

        # Check if file exists at path (only error for non-overwrite case).
        # Done before virus scanning so a cheap duplicate-path check isn't
        # blocked by a potentially slow or unavailable scanner.
        db = workspace_db()

        if not overwrite:
            existing = await db.get_workspace_file_by_path(
                self.workspace_id,
                path,
                **self._scope_kwargs(),
            )
            if existing is not None:
                raise ValueError(f"File already exists at path: {path}")

        # Scan here — callers must NOT duplicate this scan.
        # WorkspaceManager owns virus scanning for all persisted files.
        await scan_content_safe(content, filename=filename)

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
        async def _persist_db_record(
            retries: int = 2 if overwrite else 0,
        ) -> WorkspaceFile:
            """Create DB record, retrying on conflict if overwrite=True.

            Cleans up the orphaned storage file on any failure.
            """
            try:
                return await db.create_workspace_file(
                    workspace_id=self.workspace_id,
                    file_id=file_id,
                    name=filename,
                    path=path,
                    storage_path=storage_path,
                    mime_type=mime_type,
                    size_bytes=len(content),
                    checksum=checksum,
                    metadata=metadata,
                    organization_id=self.organization_id,
                    team_id=self.team_id,
                    session_id=self.session_id,
                    execution_id=self.execution_id,
                    user_global_config=self.user_global_config,
                )
            except UniqueViolationError:
                if retries > 0:
                    # Delete conflicting file and retry
                    existing = await db.get_workspace_file_by_path(
                        self.workspace_id,
                        path,
                        **self._scope_kwargs(),
                    )
                    if existing:
                        await self.delete_file(existing.id)
                    return await _persist_db_record(retries=retries - 1)
                if overwrite:
                    raise ValueError(
                        f"Unable to overwrite file at path: {path} "
                        f"(concurrent write conflict)"
                    ) from None
                raise ValueError(f"File already exists at path: {path}")

        try:
            file = await _persist_db_record()
        except Exception:
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
        name_contains: Optional[str] = None,
        path_not_starts_with: Optional[str] = None,
        metadata_equals: Optional[dict] = None,
        metadata_not_equals: Optional[dict] = None,
        folder_id: Optional[str] = None,
        root_only: bool = False,
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
            name_contains: Case-insensitive substring filter on the file name.
            path_not_starts_with: Optional path prefix to exclude from results.
                Generic path filter; origin-based filtering for the Artifacts
                page is handled separately via ``metadata_equals`` /
                ``metadata_not_equals``.
            metadata_equals: Match files whose ``metadata`` equals this object
                exactly (Artifacts "Uploaded" filter).
            metadata_not_equals: Match files whose ``metadata`` does not equal
                this object (Artifacts "Generated" filter).
            folder_id: If set, only return files in this folder.
            root_only: If True, only return root-level files (folderId IS NULL).

        Returns:
            List of WorkspaceFile instances
        """
        return await self._run_scoped(
            self._list_files(
                path=path,
                limit=limit,
                offset=offset,
                include_all_sessions=include_all_sessions,
                name_contains=name_contains,
                path_not_starts_with=path_not_starts_with,
                metadata_equals=metadata_equals,
                metadata_not_equals=metadata_not_equals,
                folder_id=folder_id,
                root_only=root_only,
            )
        )

    async def _list_files(
        self,
        path: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        include_all_sessions: bool = False,
        name_contains: Optional[str] = None,
        path_not_starts_with: Optional[str] = None,
        metadata_equals: Optional[dict] = None,
        metadata_not_equals: Optional[dict] = None,
        folder_id: Optional[str] = None,
        root_only: bool = False,
    ) -> list[WorkspaceFile]:
        effective_path = self._get_effective_path(path, include_all_sessions)
        db = workspace_db()

        return await db.list_workspace_files(
            workspace_id=self.workspace_id,
            path_prefix=effective_path,
            path_not_starts_with=path_not_starts_with,
            limit=limit,
            offset=offset,
            name_contains=name_contains,
            metadata_equals=metadata_equals,
            metadata_not_equals=metadata_not_equals,
            folder_id=folder_id,
            root_only=root_only,
            **self._scope_kwargs(),
        )

    async def delete_file(self, file_id: str) -> bool:
        """
        Delete a file (soft-delete).

        Args:
            file_id: The file's ID

        Returns:
            True if deleted, False if not found
        """
        return await self._run_scoped(self._delete_file(file_id))

    async def _delete_file(self, file_id: str) -> bool:
        db = workspace_db()
        file = await db.get_workspace_file(
            file_id,
            self.workspace_id,
            **self._scope_kwargs(),
        )
        if not self._matches_session(file):
            return False
        assert file is not None

        # Delete from storage
        storage = await get_workspace_storage()
        try:
            await storage.delete(file.storage_path)
        except Exception as e:
            logger.warning(f"Failed to delete file from storage: {e}")
            # Continue with database soft-delete even if storage delete fails

        # Soft-delete database record
        result = await db.soft_delete_workspace_file(
            file_id,
            self.workspace_id,
            **self._scope_kwargs(),
        )

        # Best-effort cleanup of the search index so deleted files don't
        # keep showing up in /search/global hits.
        if result is not None:
            try:
                from backend.api.features.workspace.embeddings import (
                    delete_workspace_file_embedding,
                )

                await delete_workspace_file_embedding(
                    file_id=file_id, user_id=self.user_id
                )
            except Exception as e:
                logger.warning(f"Failed to delete file embedding for {file_id}: {e}")

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
        return await self._run_scoped(
            self._get_download_url(file_id, expires_in=expires_in)
        )

    async def _get_download_url(self, file_id: str, expires_in: int = 3600) -> str:
        db = workspace_db()
        file = await db.get_workspace_file(
            file_id,
            self.workspace_id,
            **self._scope_kwargs(),
        )
        if not self._matches_session(file):
            raise FileNotFoundError(f"File not found: {file_id}")
        assert file is not None

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
        return await self._run_scoped(self._get_file_info(file_id))

    async def _get_file_info(self, file_id: str) -> Optional[WorkspaceFile]:
        db = workspace_db()
        file = await db.get_workspace_file(
            file_id,
            self.workspace_id,
            **self._scope_kwargs(),
        )
        return file if self._matches_session(file) else None

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
        return await self._run_scoped(self._get_file_info_by_path(path))

    async def _get_file_info_by_path(self, path: str) -> Optional[WorkspaceFile]:
        db = workspace_db()
        resolved_path = self._resolve_path(path)
        file = await db.get_workspace_file_by_path(
            self.workspace_id,
            resolved_path,
            **self._scope_kwargs(),
        )
        return file if self._matches_session(file) else None

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
        return await self._run_scoped(
            self._get_file_count(
                path=path,
                include_all_sessions=include_all_sessions,
            )
        )

    async def _get_file_count(
        self,
        path: Optional[str] = None,
        include_all_sessions: bool = False,
    ) -> int:
        effective_path = self._get_effective_path(path, include_all_sessions)
        db = workspace_db()

        return await db.count_workspace_files(
            self.workspace_id,
            path_prefix=effective_path,
            **self._scope_kwargs(),
        )
