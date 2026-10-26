from abc import ABC, abstractmethod


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
    async def retrieve_partial(self, storage_path: str, max_bytes: int) -> bytes:
        """
        Retrieve up to ``max_bytes`` from the start of the file.

        Used for cheap text previews so large files are not loaded in full.

        Args:
            storage_path: The storage path returned from store()
            max_bytes: Maximum number of leading bytes to return

        Returns:
            At most ``max_bytes`` of file content as bytes
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
