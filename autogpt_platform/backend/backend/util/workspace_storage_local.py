import logging
import tempfile
from pathlib import Path
from typing import Optional

import aiofiles

from backend.util.data import get_data_path
from backend.util.filename import sanitize_filename
from backend.util.workspace_storage_base import WorkspaceStorageBackend

logger = logging.getLogger(__name__)


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

        self.base_dir = self.base_dir.resolve()

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _build_file_path(self, workspace_id: str, file_id: str, filename: str) -> Path:
        """Build the local file path with path traversal protection."""
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

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=file_path.parent, prefix=".upload-", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            async with aiofiles.open(temporary_path, "wb") as f:
                await f.write(content)
            temporary_path.replace(file_path)
        except BaseException:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Failed to remove incomplete workspace upload")
            self._remove_empty_parents(file_path.parent)
            raise

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

    async def retrieve_partial(self, storage_path: str, max_bytes: int) -> bytes:
        """Retrieve the first ``max_bytes`` of a local file."""
        file_path = self._parse_storage_path(storage_path)

        try:
            async with aiofiles.open(file_path, "rb") as f:
                return await f.read(max_bytes)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {storage_path}") from e

    async def delete(self, storage_path: str) -> None:
        """Delete file from local storage."""
        file_path = self._parse_storage_path(storage_path)

        if file_path.exists():
            # Remove file
            file_path.unlink()

            # Clean up empty parent directories
            self._remove_empty_parents(file_path.parent)

    def _remove_empty_parents(self, parent: Path) -> None:
        """Remove ``parent`` and its ancestors while empty, stopping at base_dir."""
        while parent != self.base_dir:
            try:
                parent.rmdir()
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
