"""
WorkspaceManager for managing user workspace file operations.

This module provides a high-level interface for workspace file operations,
combining the storage backend and database layer.
"""

import asyncio
import logging
import mimetypes
import posixpath
import uuid
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Optional

from prisma.errors import UniqueViolationError
from pydantic import BaseModel

from backend.copilot.rate_limit import get_workspace_storage_limit_bytes
from backend.data.db_accessors import (
    workspace_db,
    workspace_folder_db,
    workspace_skill_db,
)
from backend.data.skill_capacity import skill_owner_folder
from backend.data.workspace import WorkspaceFile
from backend.data.workspace_internal import is_internal_workspace_file
from backend.data.workspace_scope import (
    EXPERT_FILE_ACCESS_DENIED,
    SHARED_ROOTS,
    WorkspaceAccessDeniedError,
    WorkspaceScope,
)
from backend.data.workspace_skill import (
    WorkspaceSkillFile,
    WorkspaceSkillPublication,
    WorkspaceSkillWrite,
    publication_refusal,
)
from backend.util.settings import Config
from backend.util.virus_scanner import scan_content_safe
from backend.util.workspace_storage import (
    WorkspaceStorageBackend,
    compute_file_checksum,
    get_workspace_storage,
)

if TYPE_CHECKING:
    from backend.data.workspace_folder import WorkspaceFolder


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


class SkillPackageFile(BaseModel):
    """One file of a skill package written by :meth:`WorkspaceManager.write_skill_package`."""

    path: str
    content: bytes
    mime_type: str | None = None
    metadata: dict | None = None


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
        scope: Optional[WorkspaceScope] = None,
    ):
        """
        Initialize WorkspaceManager.

        Args:
            user_id: The user's ID
            workspace_id: The workspace ID
            session_id: Optional session ID for session-scoped file access
            scope: Resolved grants for an expert session. ``None`` means the
                account owner is acting and every file in the workspace is
                reachable. When set, every operation — by path or by file ID,
                including listings and counts — is confined to the scope.
        """
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.session_id = session_id
        self.scope = scope
        # Session path prefix for file isolation
        self.session_path = f"/sessions/{session_id}" if session_id else ""

    def _authorize_path(self, path: str, *, write: bool = False) -> None:
        if self.scope is not None and not self.scope.allows_path(path, write=write):
            logger.warning(
                "Workspace access denied for expert %s (write=%s): %s",
                self.scope.expert_id,
                write,
                path,
            )
            raise WorkspaceAccessDeniedError(EXPERT_FILE_ACCESS_DENIED)

    def _authorize_file(self, file: WorkspaceFile, *, write: bool = False) -> None:
        self._authorize_path(file.path, write=write)

    def _allowed_prefixes(
        self, requested: Optional[list[str]] = None
    ) -> Optional[list[str]]:
        """Prefixes a listing may span: the scope's, narrowed to ``requested``.

        A requested prefix survives only if it lies inside a scope prefix, so
        this argument can narrow an expert's reach but never widen it.
        """
        if self.scope is None:
            return requested
        allowed = self.scope.read_prefixes
        if requested is None:
            return allowed
        return [p for p in requested if any(p.startswith(a) for a in allowed)]

    def _include_user_files(self, requested: bool) -> bool:
        """Whether a listing also spans the owner's own files.

        A scoped manager answers from its own grant: the scope is resolved
        from persisted attribution, so it decides both directions and a
        caller's argument is only consulted when there is no scope to ask.
        """
        return self.scope.reads_user_files if self.scope is not None else requested

    def _resolve_path(self, path: str) -> str:
        """
        Resolve a path, defaulting to session folder if session_id is set.

        An absolute path into a shared root (:data:`SHARED_ROOTS`) is taken
        as written: another session's folder, or a skill package, which every
        session of the account shares. :meth:`_authorize_path` still applies,
        so an expert reaches only the roots its scope grants.

        Args:
            path: Virtual path (e.g., "/file.txt" or "/sessions/abc123/file.txt")

        Returns:
            Resolved path with session prefix if applicable
        """
        if path.startswith(SHARED_ROOTS):
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
        self._authorize_path(resolved_path)
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
        self._authorize_file(file)

        storage = await get_workspace_storage()
        return await storage.retrieve(file.storage_path)

    async def read_file_head_by_id(self, file_id: str, max_bytes: int) -> bytes:
        """Read at most ``max_bytes`` from the start of a file, by file ID.

        For a preview or a slice near the start, so a large file is not
        loaded whole. Authorized exactly as ``read_file_by_id``.
        """
        db = workspace_db()
        file = await db.get_workspace_file(file_id, self.workspace_id)
        if file is None:
            raise FileNotFoundError(f"File not found: {file_id}")
        self._authorize_file(file)

        storage = await get_workspace_storage()
        return await storage.retrieve_partial(file.storage_path, max_bytes)

    async def write_file(
        self,
        content: bytes,
        filename: str,
        path: Optional[str] = None,
        mime_type: Optional[str] = None,
        overwrite: bool = False,
        metadata: Optional[dict] = None,
        scanned_checksums: Collection[str] = (),
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
            metadata: Optional metadata dict (e.g., origin tracking). A
                ``purpose`` of ``tool-output`` or ``working-file`` marks the
                file as Otto's own; see ``is_internal_workspace_file``.
            scanned_checksums: SHA-256s of bytes the server has already
                scanned clean; content hashing to one of them is not scanned
                again. Never pass hashes a client supplied.

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
        self._authorize_path(path, write=True)
        is_internal = is_internal_workspace_file(
            name=filename, path=path, metadata=metadata
        )

        # Enforce per-user workspace storage quota (tier-based).
        # For overwrites, subtract the existing file's size so replacing a file
        # with a same-size or smaller file is not rejected near the cap.
        storage_limit, current_usage = await asyncio.gather(
            get_workspace_storage_limit_bytes(self.user_id),
            workspace_db().get_workspace_total_size(self.workspace_id),
        )
        if overwrite:
            db = workspace_db()
            existing = await db.get_workspace_file_by_path(self.workspace_id, path)
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
            existing = await db.get_workspace_file_by_path(self.workspace_id, path)
            if existing is not None:
                raise ValueError(f"File already exists at path: {path}")

        # Scan here — callers must NOT duplicate this scan.
        # WorkspaceManager owns virus scanning for all persisted files.
        checksum = compute_file_checksum(content)
        if checksum not in scanned_checksums:
            await scan_content_safe(content, filename=filename)

        # Auto-detect MIME type if not provided
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or "application/octet-stream"

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
                    is_internal=is_internal,
                )
            except UniqueViolationError:
                if retries > 0:
                    # Delete conflicting file and retry
                    existing = await db.get_workspace_file_by_path(
                        self.workspace_id, path
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

        replaced_storage_path: str | None = None
        replaced_file_id: str | None = None
        try:
            if skill_owner_folder(path) is not None:
                write = WorkspaceSkillWrite(
                    workspace_id=self.workspace_id,
                    file_id=file_id,
                    name=filename,
                    path=path,
                    storage_path=storage_path,
                    mime_type=mime_type,
                    size_bytes=len(content),
                    overwrite=overwrite,
                    checksum=checksum,
                    metadata=metadata,
                )
                publication = await workspace_skill_db().publish_workspace_skill_file(
                    write
                )
                refusal = publication_refusal(publication, write)
                if refusal is not None:
                    raise refusal
                assert publication.file is not None
                file = publication.file
                replaced_storage_path = publication.replaced_storage_path
                replaced_file_id = publication.replaced_file_id
            else:
                file = await _persist_db_record()
        except Exception:
            try:
                await storage.delete(storage_path)
            except Exception as e:
                logger.warning(f"Failed to clean up orphaned storage file: {e}")
            raise

        if replaced_storage_path and replaced_storage_path != storage_path:
            try:
                await storage.delete(replaced_storage_path)
            except Exception:
                logger.warning("Failed to clean up replaced skill blob", exc_info=True)

        if replaced_file_id and replaced_file_id != file.id:
            try:
                from backend.api.features.workspace.embeddings import (
                    delete_workspace_file_embedding,
                )

                await delete_workspace_file_embedding(
                    file_id=replaced_file_id, user_id=self.user_id
                )
            except Exception:
                logger.warning(
                    "Failed to clean up replaced skill embedding", exc_info=True
                )

        logger.info(
            f"Wrote file {file.id} ({filename}) to workspace {self.workspace_id} "
            f"at path {path}, size={len(content)} bytes"
        )

        # Fire-and-forget: index this file in the hybrid-search store so
        # the user can find it from /search/global by name. No-ops cheaply
        # when the existing embedding's text is unchanged. Otto's own working
        # files are not the user's to find.
        if file.is_internal:
            return file
        try:
            from backend.api.features.workspace.embeddings import (
                schedule_workspace_file_embedding,
            )

            schedule_workspace_file_embedding(
                file_id=file.id, user_id=self.user_id, name=file.name, path=file.path
            )
        except Exception as e:
            # Embedding is purely a search-quality concern — never block writes.
            logger.warning(f"Failed to schedule file embedding for {file.id}: {e}")

        return file

    async def write_skill_package(
        self,
        root: SkillPackageFile,
        files: Sequence[SkillPackageFile],
        removed_paths: Collection[str],
        scanned_checksums: Collection[str] = (),
        *,
        config: Config | None = None,
    ) -> WorkspaceFile:
        """Write a skill package as one change: ``root`` is its SKILL.md,
        ``files`` the other files that changed, and ``removed_paths`` the
        files it no longer has.

        Every blob is checked, scanned and stored before one transaction
        swaps the rows, so a failure anywhere leaves the previous package
        exactly as it was, rather than new files beside an old SKILL.md.
        """
        package = [
            item.model_copy(update={"path": self._resolve_path(item.path)})
            for item in [root, *files]
        ]
        removed = [self._resolve_path(path) for path in removed_paths]
        for path in [*(item.path for item in package), *removed]:
            self._authorize_path(path, write=True)
        config = config if config is not None else Config()
        await self._require_package_room(package, removed, config.max_file_size_mb)
        for item in package:
            if compute_file_checksum(item.content) not in scanned_checksums:
                await scan_content_safe(
                    item.content, filename=posixpath.basename(item.path)
                )
        storage = await get_workspace_storage(config=config)
        stored: list[WorkspaceSkillFile] = []
        try:
            for item in package:
                stored.append(await self._store_package_blob(storage, item))
            write = WorkspaceSkillWrite(
                workspace_id=self.workspace_id,
                overwrite=True,
                files=stored[1:],
                removed_paths=removed,
                **stored[0].model_dump(),
            )
            publication = await workspace_skill_db().publish_workspace_skill_file(write)
        except BaseException:
            # Cancellation included: a save stopped part-way must not leave
            # blobs no row points at. Shielded so a second cancel cannot
            # interrupt the cleanup itself.
            await asyncio.shield(self._discard_unpublished(storage, stored))
            raise
        refusal = publication_refusal(publication, write)
        if refusal is not None:
            await asyncio.shield(self._discard_unpublished(storage, stored))
            raise refusal
        assert publication.file is not None
        await self._retire_replaced(storage, publication)
        logger.info(
            f"Wrote skill package {publication.file.path} ({len(stored)} files, "
            f"{len(removed)} removed) to workspace {self.workspace_id}"
        )
        # Package files are the skills registry's, not the user's to find.
        for blob in stored:
            if not is_internal_workspace_file(
                name=blob.name, path=blob.path, metadata=blob.metadata
            ):
                self._schedule_embedding(blob.file_id, blob.name, blob.path)
        return publication.file

    async def _discard_unpublished(
        self, storage: WorkspaceStorageBackend, stored: list[WorkspaceSkillFile]
    ) -> None:
        """Delete blobs a failed package write staged. A publication whose
        reply was lost may have committed anyway; its root row then exists
        and every blob it names must stay."""
        if not stored:
            return
        try:
            if await workspace_db().get_workspace_file(
                stored[0].file_id, self.workspace_id
            ):
                return
        except Exception:
            logger.warning(
                "Could not tell whether a skill package was published; "
                "keeping its blobs",
                exc_info=True,
            )
            return
        for blob in stored:
            try:
                await storage.delete(blob.storage_path)
            except Exception:
                logger.warning("Failed to clean up unpublished skill blob")

    async def _require_package_room(
        self, package: list[SkillPackageFile], removed: list[str], max_file_size_mb: int
    ) -> None:
        max_file_size = max_file_size_mb * 1024 * 1024
        for item in package:
            if len(item.content) > max_file_size:
                raise ValueError(
                    f"File too large: {len(item.content)} bytes exceeds "
                    f"{max_file_size_mb}MB limit"
                )
        db = workspace_db()
        storage_limit, current_usage = await asyncio.gather(
            get_workspace_storage_limit_bytes(self.user_id),
            db.get_workspace_total_size(self.workspace_id),
        )
        for path in {item.path for item in package} | set(removed):
            existing = await db.get_workspace_file_by_path(self.workspace_id, path)
            if existing is not None:
                current_usage -= existing.size_bytes
        current_usage = max(0, current_usage)
        projected = current_usage + sum(len(item.content) for item in package)
        if storage_limit > 0 and projected > storage_limit:
            raise ValueError(
                f"Storage limit exceeded. "
                f"You've used {format_bytes(current_usage)} of your "
                f"{format_bytes(storage_limit)} quota. "
                f"Delete some files or upgrade your plan for more storage."
            )

    async def _store_package_blob(
        self, storage: WorkspaceStorageBackend, item: SkillPackageFile
    ) -> WorkspaceSkillFile:
        file_id = str(uuid.uuid4())
        name = posixpath.basename(item.path)
        mime_type = item.mime_type or (
            mimetypes.guess_type(name)[0] or "application/octet-stream"
        )
        storage_path = await storage.store(
            workspace_id=self.workspace_id,
            file_id=file_id,
            filename=name,
            content=item.content,
        )
        return WorkspaceSkillFile(
            file_id=file_id,
            name=name,
            path=item.path,
            storage_path=storage_path,
            mime_type=mime_type,
            size_bytes=len(item.content),
            checksum=compute_file_checksum(item.content),
            metadata=item.metadata,
        )

    async def _retire_replaced(
        self, storage: WorkspaceStorageBackend, publication: WorkspaceSkillPublication
    ) -> None:
        """Best-effort cleanup once the new package is live: the blobs and
        search entries of every row it replaced."""
        retired = [(f.file_id, f.storage_path) for f in publication.retired_files]
        if publication.replaced_file_id and publication.replaced_storage_path:
            retired.append(
                (publication.replaced_file_id, publication.replaced_storage_path)
            )
        for file_id, storage_path in retired:
            try:
                await storage.delete(storage_path)
            except Exception:
                logger.warning("Failed to clean up replaced skill blob", exc_info=True)
            try:
                from backend.api.features.workspace.embeddings import (
                    delete_workspace_file_embedding,
                )

                await delete_workspace_file_embedding(
                    file_id=file_id, user_id=self.user_id
                )
            except Exception:
                logger.warning(
                    "Failed to clean up replaced skill embedding", exc_info=True
                )

    def _schedule_embedding(self, file_id: str, name: str, path: str) -> None:
        try:
            from backend.api.features.workspace.embeddings import (
                schedule_workspace_file_embedding,
            )

            schedule_workspace_file_embedding(
                file_id=file_id, user_id=self.user_id, name=name, path=path
            )
        except Exception as e:
            # Embedding is purely a search-quality concern — never block writes.
            logger.warning(f"Failed to schedule file embedding for {file_id}: {e}")

    async def list_folders(self) -> list["WorkspaceFolder"]:
        """The folder tree; empty for a scope without the owner's files, whose
        folders these are."""
        if self.scope is not None and not self.scope.reads_user_files:
            return []
        return await workspace_folder_db().list_workspace_folders(self.workspace_id)

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
        folder_ids: Optional[list[str]] = None,
        root_only: bool = False,
        allowed_path_prefixes: Optional[list[str]] = None,
        include_user_files: bool = False,
        include_internal: bool = True,
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
            folder_ids: If set, only return files in any of these folders.
            root_only: If True, only return root-level files (folderId IS NULL).
                Every folder filter is a workspace-wide axis, so any of them
                lifts the current-session default the way
                ``include_all_sessions`` does; an expert's scope still confines
                the result.
            allowed_path_prefixes: Only list files under these prefixes. When
                the manager carries a scope the prefixes are intersected with
                it (see :meth:`_allowed_prefixes`).
            include_user_files: Also list the owner's own files. A scoped
                manager takes this from its scope instead
                (see :meth:`_include_user_files`).
            include_internal: Whether to list Otto's own working files. Lists
                the user sees pass ``False``.

        Returns:
            List of WorkspaceFile instances
        """
        effective_path = self._get_effective_path(
            path,
            include_all_sessions
            or folder_id is not None
            or folder_ids is not None
            or root_only,
        )
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
            folder_ids=folder_ids,
            root_only=root_only,
            allowed_path_prefixes=self._allowed_prefixes(allowed_path_prefixes),
            include_user_files=self._include_user_files(include_user_files),
            include_internal=include_internal,
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
        self._authorize_file(file, write=True)

        # Delete from storage
        storage = await get_workspace_storage()
        try:
            await storage.delete(file.storage_path)
        except Exception as e:
            logger.warning(f"Failed to delete file from storage: {e}")
            # Continue with database soft-delete even if storage delete fails

        # Soft-delete database record
        result = await db.soft_delete_workspace_file(file_id, self.workspace_id)

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
        db = workspace_db()
        file = await db.get_workspace_file(file_id, self.workspace_id)
        if file is None:
            raise FileNotFoundError(f"File not found: {file_id}")
        self._authorize_file(file)

        storage = await get_workspace_storage()
        return await storage.get_download_url(file.storage_path, expires_in)

    async def get_file_info(self, file_id: str) -> Optional[WorkspaceFile]:
        """
        Get file metadata.

        Args:
            file_id: The file's ID

        Returns:
            WorkspaceFile instance, or None when no such file exists.

        Raises:
            WorkspaceAccessDeniedError: the file exists but lies outside this
                manager's scope. Deliberately not folded into ``None`` so an
                expert sees "access denied" rather than "not found" for a
                file it cannot reach; every caller either surfaces it as such
                or treats it like any other failure to read the file.
        """
        db = workspace_db()
        file = await db.get_workspace_file(file_id, self.workspace_id)
        if file is not None:
            self._authorize_file(file)
        return file

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
        self._authorize_path(resolved_path)
        return await db.get_workspace_file_by_path(self.workspace_id, resolved_path)

    async def get_file_count(
        self,
        path: Optional[str] = None,
        include_all_sessions: bool = False,
        folder_id: Optional[str] = None,
        folder_ids: Optional[list[str]] = None,
        root_only: bool = False,
        allowed_path_prefixes: Optional[list[str]] = None,
        include_user_files: bool = False,
    ) -> int:
        """
        Get number of files in workspace.

        When session_id is set and include_all_sessions is False (default),
        only counts files in the current session's folder.

        Args:
            path: Optional path prefix to filter (e.g., "/documents/")
            include_all_sessions: If True, count all files in workspace.
                                  If False (default), only count current session's files.
            folder_id: See :meth:`list_files`.
            folder_ids: See :meth:`list_files`.
            root_only: See :meth:`list_files`.

        Returns:
            Number of files
        """
        effective_path = self._get_effective_path(
            path,
            include_all_sessions
            or folder_id is not None
            or folder_ids is not None
            or root_only,
        )
        db = workspace_db()

        return await db.count_workspace_files(
            self.workspace_id,
            path_prefix=effective_path,
            folder_id=folder_id,
            folder_ids=folder_ids,
            root_only=root_only,
            allowed_path_prefixes=self._allowed_prefixes(allowed_path_prefixes),
            include_user_files=self._include_user_files(include_user_files),
        )
