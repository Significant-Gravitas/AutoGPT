"""
Workspace storage backend abstraction for supporting both cloud and local deployments.

This module provides a unified interface for storing workspace files, with implementations
for Google Cloud Storage (cloud deployments) and local filesystem (self-hosted deployments).
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from gcloud.aio import storage as async_gcs_storage
from google.cloud import storage as gcs_storage

from backend.util.data import get_data_path
from backend.util.gcs_utils import (
    download_with_fresh_session,
    generate_signed_url,
    parse_gcs_path,
)
from backend.util.settings import Config

logger = logging.getLogger(__name__)


class WorkspaceStorageBackend(ABC):
    """Abstract interface for workspace file storage."""

    @abstractmethod
    async def store(
        self,
        workspace_id: str,
        file_id: str,
        filename: str,
        content: bytes,
    ) -> str:
        """
        Store file content, return storage path.

        Args:
            workspace_id: The workspace ID
            file_id: Unique file ID for storage
            filename: Original filename
            content: File content as bytes

        Returns:
            Storage path string (cloud path or local path)
        """
        pass

    @abstractmethod
    async def retrieve(self, storage_path: str) -> bytes:
        """
        Retrieve file content from storage.

        Args:
            storage_path: The storage path returned from store()

        Returns:
            File content as bytes
        """
        pass

    @abstractmethod
    async def delete(self, storage_path: str) -> None:
        """
        Delete file from storage.

        Args:
            storage_path: The storage path to delete
        """
        pass

    @abstractmethod
    async def get_download_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """
        Get URL for downloading the file.

        Args:
            storage_path: The storage path
            expires_in: URL expiration time in seconds (default 1 hour)

        Returns:
            Download URL (signed URL for GCS, direct API path for local)
        """
        pass


class GCSWorkspaceStorage(WorkspaceStorageBackend):
    """Google Cloud Storage implementation for workspace storage.

    Each instance owns a single ``aiohttp.ClientSession`` and GCS async
    client.  Because ``ClientSession`` is bound to the event loop on which it
    was created, callers that run on separate loops (e.g. copilot executor
    worker threads) **must** obtain their own ``GCSWorkspaceStorage`` instance
    via :func:`get_workspace_storage` which is event-loop-aware.
    """

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self._async_client: Optional[async_gcs_storage.Storage] = None
        self._sync_client: Optional[gcs_storage.Client] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_async_client(self) -> async_gcs_storage.Storage:
        """Get or create async GCS client."""
        if self._async_client is None:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=100, force_close=False)
            )
            self._async_client = async_gcs_storage.Storage(session=self._session)
        return self._async_client

    def _get_sync_client(self) -> gcs_storage.Client:
        """Get or create sync GCS client (for signed URLs)."""
        if self._sync_client is None:
            self._sync_client = gcs_storage.Client()
        return self._sync_client

    async def close(self) -> None:
        """Close all client connections."""
        if self._async_client is not None:
            try:
                await self._async_client.close()
            except Exception as e:
                logger.warning(f"Error closing GCS client: {e}")
            self._async_client = None

        if self._session is not None:
            try:
                await self._session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            self._session = None

    def _build_blob_name(self, workspace_id: str, file_id: str, filename: str) -> str:
        """Build the blob path for workspace files."""
        return f"workspaces/{workspace_id}/{file_id}/{filename}"

    async def store(
        self,
        workspace_id: str,
        file_id: str,
        filename: str,
        content: bytes,
    ) -> str:
        """Store file in GCS."""
        client = await self._get_async_client()
        blob_name = self._build_blob_name(workspace_id, file_id, filename)

        # Upload with metadata
        upload_time = datetime.now(timezone.utc)
        await client.upload(
            self.bucket_name,
            blob_name,
            content,
            metadata={
                "uploaded_at": upload_time.isoformat(),
                "workspace_id": workspace_id,
                "file_id": file_id,
            },
        )

        return f"gcs://{self.bucket_name}/{blob_name}"

    async def retrieve(self, storage_path: str) -> bytes:
        """Retrieve file from GCS."""
        bucket_name, blob_name = parse_gcs_path(storage_path)
        return await download_with_fresh_session(bucket_name, blob_name)

    async def delete(self, storage_path: str) -> None:
        """Delete file from GCS."""
        bucket_name, blob_name = parse_gcs_path(storage_path)
        client = await self._get_async_client()

        try:
            await client.delete(bucket_name, blob_name)
        except Exception as e:
            if "404" not in str(e) and "Not Found" not in str(e):
                raise
            # File already deleted, that's fine

    async def get_download_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """
        Generate download URL for GCS file.

        Attempts to generate a signed URL if running with service account credentials.
        Falls back to an API proxy endpoint if signed URL generation fails
        (e.g., when running locally with user OAuth credentials).
        """
        bucket_name, blob_name = parse_gcs_path(storage_path)

        # Extract file_id from blob_name for fallback: workspaces/{workspace_id}/{file_id}/{filename}
        blob_parts = blob_name.split("/")
        file_id = blob_parts[2] if len(blob_parts) >= 3 else None

        # Try to generate signed URL (requires service account credentials)
        try:
            sync_client = self._get_sync_client()
            return await generate_signed_url(
                sync_client, bucket_name, blob_name, expires_in
            )
        except AttributeError as e:
            # Signed URL generation requires service account with private key.
            # When running with user OAuth credentials, fall back to API proxy.
            if "private key" in str(e) and file_id:
                logger.debug(
                    "Cannot generate signed URL (no service account credentials), "
                    "falling back to API proxy endpoint"
                )
                return f"/api/workspace/files/{file_id}/download"
            raise


class LocalWorkspaceStorage(WorkspaceStorageBackend):
    """Local filesystem implementation for workspace storage (self-hosted deployments)."""

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize local storage backend.

        Args:
            base_dir: Base directory for workspace storage.
                     If None, defaults to {app_data}/workspaces
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path(get_data_path()) / "workspaces"

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _build_file_path(self, workspace_id: str, file_id: str, filename: str) -> Path:
        """Build the local file path with path traversal protection."""
        # Import here to avoid circular import
        # (file.py imports workspace.py which imports workspace_storage.py)
        from backend.util.file import sanitize_filename

        # Sanitize filename to prevent path traversal (removes / and \ among others)
        safe_filename = sanitize_filename(filename)
        file_path = (self.base_dir / workspace_id / file_id / safe_filename).resolve()

        # Verify the resolved path is still under base_dir
        if not file_path.is_relative_to(self.base_dir.resolve()):
            raise ValueError("Invalid filename: path traversal detected")

        return file_path

    def _parse_storage_path(self, storage_path: str) -> Path:
        """Parse local storage path to filesystem path."""
        if storage_path.startswith("local://"):
            relative_path = storage_path[8:]  # Remove "local://"
        else:
            relative_path = storage_path

        full_path = (self.base_dir / relative_path).resolve()

        # Security check: ensure path is under base_dir
        # Use is_relative_to() for robust path containment check
        # (handles case-insensitive filesystems and edge cases)
        if not full_path.is_relative_to(self.base_dir.resolve()):
            raise ValueError("Invalid storage path: path traversal detected")

        return full_path

    async def store(
        self,
        workspace_id: str,
        file_id: str,
        filename: str,
        content: bytes,
    ) -> str:
        """Store file locally."""
        file_path = self._build_file_path(workspace_id, file_id, filename)

        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file asynchronously
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        # Return relative path as storage path
        relative_path = file_path.relative_to(self.base_dir)
        return f"local://{relative_path}"

    async def retrieve(self, storage_path: str) -> bytes:
        """Retrieve file from local storage."""
        file_path = self._parse_storage_path(storage_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {storage_path}")

        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()

    async def delete(self, storage_path: str) -> None:
        """Delete file from local storage."""
        file_path = self._parse_storage_path(storage_path)

        if file_path.exists():
            # Remove file
            file_path.unlink()

            # Clean up empty parent directories
            parent = file_path.parent
            while parent != self.base_dir:
                try:
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                    else:
                        break
                except OSError:
                    break
                parent = parent.parent

    async def get_download_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """
        Get download URL for local file.

        For local storage, this returns an API endpoint path.
        The actual serving is handled by the API layer.
        """
        # Parse the storage path to get the components
        if storage_path.startswith("local://"):
            relative_path = storage_path[8:]
        else:
            relative_path = storage_path

        # Return the API endpoint for downloading
        # The file_id is extracted from the path: {workspace_id}/{file_id}/{filename}
        parts = relative_path.split("/")
        if len(parts) >= 2:
            file_id = parts[1]  # Second component is file_id
            return f"/api/workspace/files/{file_id}/download"
        else:
            raise ValueError(f"Invalid storage path format: {storage_path}")


# ---------------------------------------------------------------------------
# Storage instance management
# ---------------------------------------------------------------------------
# ``aiohttp.ClientSession`` is bound to the event loop where it is created.
# The copilot executor runs each worker in its own thread with a dedicated
# event loop, so a single global ``GCSWorkspaceStorage`` instance would break.
#
# For **local storage** a single shared instance is fine (no async I/O).
# For **GCS storage** we keep one instance *per event loop* so every loop
# gets its own ``ClientSession``.
# ---------------------------------------------------------------------------

_local_storage: Optional[LocalWorkspaceStorage] = None
_gcs_storages: dict[int, GCSWorkspaceStorage] = {}
_storage_lock = asyncio.Lock()


async def get_workspace_storage() -> WorkspaceStorageBackend:
    """Return a workspace storage backend for the **current** event loop.

    * Local storage → single shared instance (no event-loop affinity).
    * GCS storage   → one instance per event loop to avoid cross-loop
      ``aiohttp`` errors.
    """
    global _local_storage

    config = Config()

    # --- Local storage (shared) ---
    if not config.media_gcs_bucket_name:
        if _local_storage is None:
            storage_dir = (
                config.workspace_storage_dir if config.workspace_storage_dir else None
            )
            logger.info(f"Using local workspace storage: {storage_dir or 'default'}")
            _local_storage = LocalWorkspaceStorage(storage_dir)
        return _local_storage

    # --- GCS storage (per event loop) ---
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _gcs_storages:
        logger.info(
            f"Creating GCS workspace storage for loop {loop_id}: "
            f"{config.media_gcs_bucket_name}"
        )
        _gcs_storages[loop_id] = GCSWorkspaceStorage(config.media_gcs_bucket_name)
    return _gcs_storages[loop_id]


async def shutdown_workspace_storage() -> None:
    """Shut down workspace storage for the **current** event loop.

    Closes the ``aiohttp`` session owned by the current loop's GCS instance.
    Each worker thread should call this on its own loop before the loop is
    destroyed.  The REST API lifespan hook calls it for the main server loop.
    """
    global _local_storage

    loop_id = id(asyncio.get_running_loop())
    storage = _gcs_storages.pop(loop_id, None)
    if storage is not None:
        await storage.close()

    # Clear local storage only when the last GCS instance is gone
    # (i.e. full shutdown, not just a single worker stopping).
    if not _gcs_storages:
        _local_storage = None


def compute_file_checksum(content: bytes) -> str:
    """Compute SHA256 checksum of file content."""
    return hashlib.sha256(content).hexdigest()
