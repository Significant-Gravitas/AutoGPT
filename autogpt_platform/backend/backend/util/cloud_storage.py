"""
Cloud storage utilities for handling various cloud storage providers.
"""

import asyncio
import logging
import os.path
import uuid
from datetime import datetime, timedelta, timezone
from typing import Tuple

from gcloud.aio import storage as async_gcs_storage
from google.cloud import storage as gcs_storage

from backend.util.settings import Config

logger = logging.getLogger(__name__)


class CloudStorageConfig:
    """Configuration for cloud storage providers."""

    def __init__(self):
        config = Config()

        # GCS configuration from settings - uses Application Default Credentials
        self.gcs_bucket_name = config.media_gcs_bucket_name

        # Future providers can be added here
        # self.aws_bucket_name = config.aws_bucket_name
        # self.azure_container_name = config.azure_container_name


class CloudStorageHandler:
    """Generic cloud storage handler that can work with multiple providers."""

    def __init__(self, config: CloudStorageConfig):
        self.config = config
        self._async_gcs_client = None
        self._sync_gcs_client = None  # Only for signed URLs

    def _get_async_gcs_client(self):
        """Lazy initialization of async GCS client."""
        if self._async_gcs_client is None:
            # Use Application Default Credentials (ADC)
            self._async_gcs_client = async_gcs_storage.Storage()
        return self._async_gcs_client

    def _get_sync_gcs_client(self):
        """Lazy initialization of sync GCS client (only for signed URLs)."""
        if self._sync_gcs_client is None:
            # Use Application Default Credentials (ADC) - same as media.py
            self._sync_gcs_client = gcs_storage.Client()
        return self._sync_gcs_client

    def parse_cloud_path(self, path: str) -> Tuple[str, str]:
        """
        Parse a cloud storage path and return provider and actual path.

        Args:
            path: Cloud storage path (e.g., "gcs://bucket/path/to/file")

        Returns:
            Tuple of (provider, actual_path)
        """
        if path.startswith("gcs://"):
            return "gcs", path[6:]  # Remove "gcs://" prefix
        # Future providers:
        # elif path.startswith("s3://"):
        #     return "s3", path[5:]
        # elif path.startswith("azure://"):
        #     return "azure", path[8:]
        else:
            raise ValueError(f"Unsupported cloud storage path: {path}")

    def is_cloud_path(self, path: str) -> bool:
        """Check if a path is a cloud storage path."""
        return path.startswith(("gcs://", "s3://", "azure://"))

    async def store_file(
        self,
        content: bytes,
        filename: str,
        provider: str = "gcs",
        expiration_hours: int = 48,
        user_id: str | None = None,
        graph_exec_id: str | None = None,
    ) -> str:
        """
        Store file content in cloud storage.

        Args:
            content: File content as bytes
            filename: Desired filename
            provider: Cloud storage provider ("gcs", "s3", "azure")
            expiration_hours: Hours until expiration (1-48, default: 48)
            user_id: User ID for user-scoped files (optional)
            graph_exec_id: Graph execution ID for execution-scoped files (optional)

        Note:
            Provide either user_id OR graph_exec_id, not both. If neither is provided,
            files will be stored as system uploads.

        Returns:
            Cloud storage path (e.g., "gcs://bucket/path/to/file")
        """
        if provider == "gcs":
            return await self._store_file_gcs(
                content, filename, expiration_hours, user_id, graph_exec_id
            )
        else:
            raise ValueError(f"Unsupported cloud storage provider: {provider}")

    async def _store_file_gcs(
        self,
        content: bytes,
        filename: str,
        expiration_hours: int,
        user_id: str | None = None,
        graph_exec_id: str | None = None,
    ) -> str:
        """Store file in Google Cloud Storage."""
        if not self.config.gcs_bucket_name:
            raise ValueError("GCS_BUCKET_NAME not configured")

        # Validate that only one scope is provided
        if user_id and graph_exec_id:
            raise ValueError("Provide either user_id OR graph_exec_id, not both")

        async_client = self._get_async_gcs_client()

        # Generate unique path with appropriate scope
        unique_id = str(uuid.uuid4())
        if user_id:
            # User-scoped uploads
            blob_name = f"uploads/users/{user_id}/{unique_id}/{filename}"
        elif graph_exec_id:
            # Execution-scoped uploads
            blob_name = f"uploads/executions/{graph_exec_id}/{unique_id}/{filename}"
        else:
            # System uploads (for backwards compatibility)
            blob_name = f"uploads/system/{unique_id}/{filename}"

        # Upload content with metadata using pure async client
        upload_time = datetime.now(timezone.utc)
        expiration_time = upload_time + timedelta(hours=expiration_hours)

        await async_client.upload(
            self.config.gcs_bucket_name,
            blob_name,
            content,
            metadata={
                "uploaded_at": upload_time.isoformat(),
                "expires_at": expiration_time.isoformat(),
                "expiration_hours": str(expiration_hours),
            },
        )

        return f"gcs://{self.config.gcs_bucket_name}/{blob_name}"

    async def retrieve_file(
        self,
        cloud_path: str,
        user_id: str | None = None,
        graph_exec_id: str | None = None,
    ) -> bytes:
        """
        Retrieve file content from cloud storage.

        Args:
            cloud_path: Cloud storage path (e.g., "gcs://bucket/path/to/file")
            user_id: User ID for authorization of user-scoped files (optional)
            graph_exec_id: Graph execution ID for authorization of execution-scoped files (optional)

        Returns:
            File content as bytes

        Raises:
            PermissionError: If user tries to access files they don't own
        """
        provider, path = self.parse_cloud_path(cloud_path)

        if provider == "gcs":
            return await self._retrieve_file_gcs(path, user_id, graph_exec_id)
        else:
            raise ValueError(f"Unsupported cloud storage provider: {provider}")

    async def _retrieve_file_gcs(
        self, path: str, user_id: str | None = None, graph_exec_id: str | None = None
    ) -> bytes:
        """Retrieve file from Google Cloud Storage with authorization."""
        # Parse bucket and blob name from path
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid GCS path: {path}")

        bucket_name, blob_name = parts

        # Authorization check
        self._validate_file_access(blob_name, user_id, graph_exec_id)

        async_client = self._get_async_gcs_client()

        try:
            # Download content using pure async client
            content = await async_client.download(bucket_name, blob_name)
            return content
        except Exception as e:
            # Convert gcloud-aio exceptions to standard ones
            if "404" in str(e) or "Not Found" in str(e):
                raise FileNotFoundError(f"File not found: gcs://{path}")
            raise

    def _validate_file_access(
        self,
        blob_name: str,
        user_id: str | None = None,
        graph_exec_id: str | None = None,
    ) -> None:
        """
        Validate that a user can access a specific file path.

        Args:
            blob_name: The blob path in GCS
            user_id: The requesting user ID (optional)
            graph_exec_id: The requesting graph execution ID (optional)

        Raises:
            PermissionError: If access is denied
        """

        # Normalize the path to prevent path traversal attacks
        normalized_path = os.path.normpath(blob_name)

        # Ensure the normalized path doesn't contain any path traversal attempts
        if ".." in normalized_path or normalized_path.startswith("/"):
            raise PermissionError("Invalid file path: path traversal detected")

        # Split into components and validate each part
        path_parts = normalized_path.split("/")

        # Validate path structure: must start with "uploads/"
        if not path_parts or path_parts[0] != "uploads":
            raise PermissionError("Invalid file path: must be under uploads/")

        # System uploads (uploads/system/*) can be accessed by anyone for backwards compatibility
        if len(path_parts) >= 2 and path_parts[1] == "system":
            return

        # User-specific uploads (uploads/users/{user_id}/*) require matching user_id
        if len(path_parts) >= 2 and path_parts[1] == "users":
            if not user_id or len(path_parts) < 3:
                raise PermissionError(
                    "User ID required to access user files"
                    if not user_id
                    else "Invalid user file path format"
                )

            file_owner_id = path_parts[2]
            # Validate user_id format (basic validation) - no need to check ".." again since we already did
            if not file_owner_id or "/" in file_owner_id:
                raise PermissionError("Invalid user ID in path")

            if file_owner_id != user_id:
                raise PermissionError(
                    f"Access denied: file belongs to user {file_owner_id}"
                )
            return

        # Execution-specific uploads (uploads/executions/{graph_exec_id}/*) require matching graph_exec_id
        if len(path_parts) >= 2 and path_parts[1] == "executions":
            if not graph_exec_id or len(path_parts) < 3:
                raise PermissionError(
                    "Graph execution ID required to access execution files"
                    if not graph_exec_id
                    else "Invalid execution file path format"
                )

            file_exec_id = path_parts[2]
            # Validate execution_id format (basic validation) - no need to check ".." again since we already did
            if not file_exec_id or "/" in file_exec_id:
                raise PermissionError("Invalid execution ID in path")

            if file_exec_id != graph_exec_id:
                raise PermissionError(
                    f"Access denied: file belongs to execution {file_exec_id}"
                )
            return

        # Legacy uploads directory (uploads/*) - allow for backwards compatibility with warning
        # Note: We already validated it starts with "uploads/" above, so this is guaranteed to match
        logger.warning(f"Accessing legacy upload path: {blob_name}")
        return

    async def generate_signed_url(
        self,
        cloud_path: str,
        expiration_hours: int = 1,
        user_id: str | None = None,
        graph_exec_id: str | None = None,
    ) -> str:
        """
        Generate a signed URL for temporary access to a cloud storage file.

        Args:
            cloud_path: Cloud storage path
            expiration_hours: URL expiration in hours
            user_id: User ID for authorization (required for user files)
            graph_exec_id: Graph execution ID for authorization (required for execution files)

        Returns:
            Signed URL string

        Raises:
            PermissionError: If user tries to access files they don't own
        """
        provider, path = self.parse_cloud_path(cloud_path)

        if provider == "gcs":
            return await self._generate_signed_url_gcs(
                path, expiration_hours, user_id, graph_exec_id
            )
        else:
            raise ValueError(f"Unsupported cloud storage provider: {provider}")

    async def _generate_signed_url_gcs(
        self,
        path: str,
        expiration_hours: int,
        user_id: str | None = None,
        graph_exec_id: str | None = None,
    ) -> str:
        """Generate signed URL for GCS with authorization."""

        # Parse bucket and blob name from path
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid GCS path: {path}")

        bucket_name, blob_name = parts

        # Authorization check
        self._validate_file_access(blob_name, user_id, graph_exec_id)

        # Use sync client for signed URLs since gcloud-aio doesn't support them
        sync_client = self._get_sync_gcs_client()
        bucket = sync_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Generate signed URL asynchronously using sync client
        url = await asyncio.to_thread(
            blob.generate_signed_url,
            version="v4",
            expiration=datetime.now(timezone.utc) + timedelta(hours=expiration_hours),
            method="GET",
        )

        return url

    async def delete_expired_files(self, provider: str = "gcs") -> int:
        """
        Delete files that have passed their expiration time.

        Args:
            provider: Cloud storage provider

        Returns:
            Number of files deleted
        """
        if provider == "gcs":
            return await self._delete_expired_files_gcs()
        else:
            raise ValueError(f"Unsupported cloud storage provider: {provider}")

    async def _delete_expired_files_gcs(self) -> int:
        """Delete expired files from GCS based on metadata."""
        if not self.config.gcs_bucket_name:
            raise ValueError("GCS_BUCKET_NAME not configured")

        async_client = self._get_async_gcs_client()
        current_time = datetime.now(timezone.utc)

        try:
            # List all blobs in the uploads directory using pure async client
            list_response = await async_client.list_objects(
                self.config.gcs_bucket_name, params={"prefix": "uploads/"}
            )

            items = list_response.get("items", [])
            deleted_count = 0

            # Process deletions in parallel with limited concurrency
            semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent deletions

            async def delete_if_expired(blob_info):
                async with semaphore:
                    blob_name = blob_info.get("name", "")
                    try:
                        # Get blob metadata - need to fetch it separately
                        if not blob_name:
                            return 0

                        # Get metadata for this specific blob using pure async client
                        metadata_response = await async_client.download_metadata(
                            self.config.gcs_bucket_name, blob_name
                        )
                        metadata = metadata_response.get("metadata", {})

                        if metadata and "expires_at" in metadata:
                            expires_at = datetime.fromisoformat(metadata["expires_at"])
                            if current_time > expires_at:
                                # Delete using pure async client
                                await async_client.delete(
                                    self.config.gcs_bucket_name, blob_name
                                )
                                return 1
                    except Exception as e:
                        # Log specific errors for debugging
                        logger.warning(
                            f"Failed to process file {blob_name} during cleanup: {e}"
                        )
                        # Skip files with invalid metadata or delete errors
                        pass
                    return 0

            if items:
                results = await asyncio.gather(
                    *[delete_if_expired(blob) for blob in items]
                )
                deleted_count = sum(results)

            return deleted_count

        except Exception as e:
            # Log the error for debugging but continue operation
            logger.error(f"Cleanup operation failed: {e}")
            # Return 0 - we'll try again next cleanup cycle
            return 0

    async def check_file_expired(self, cloud_path: str) -> bool:
        """
        Check if a file has expired based on its metadata.

        Args:
            cloud_path: Cloud storage path

        Returns:
            True if file has expired, False otherwise
        """
        provider, path = self.parse_cloud_path(cloud_path)

        if provider == "gcs":
            return await self._check_file_expired_gcs(path)
        else:
            raise ValueError(f"Unsupported cloud storage provider: {provider}")

    async def _check_file_expired_gcs(self, path: str) -> bool:
        """Check if a GCS file has expired."""
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid GCS path: {path}")

        bucket_name, blob_name = parts

        async_client = self._get_async_gcs_client()

        try:
            # Get object metadata using pure async client
            metadata_info = await async_client.download_metadata(bucket_name, blob_name)
            metadata = metadata_info.get("metadata", {})

            if metadata and "expires_at" in metadata:
                expires_at = datetime.fromisoformat(metadata["expires_at"])
                return datetime.now(timezone.utc) > expires_at

        except Exception as e:
            # If file doesn't exist or we can't read metadata
            if "404" in str(e) or "Not Found" in str(e):
                logger.debug(f"File not found during expiration check: {blob_name}")
                return True  # File doesn't exist, consider it expired

            # Log other types of errors for debugging
            logger.warning(f"Failed to check expiration for {blob_name}: {e}")
            # If we can't read metadata for other reasons, assume not expired
            return False

        return False


# Global instance with thread safety
_cloud_storage_handler = None
_handler_lock = asyncio.Lock()
_cleanup_lock = asyncio.Lock()


async def get_cloud_storage_handler() -> CloudStorageHandler:
    """Get the global cloud storage handler instance with proper locking."""
    global _cloud_storage_handler

    if _cloud_storage_handler is None:
        async with _handler_lock:
            # Double-check pattern to avoid race conditions
            if _cloud_storage_handler is None:
                config = CloudStorageConfig()
                _cloud_storage_handler = CloudStorageHandler(config)

    return _cloud_storage_handler


async def cleanup_expired_files_async() -> int:
    """
    Clean up expired files from cloud storage.

    This function uses a lock to prevent concurrent cleanup operations.

    Returns:
        Number of files deleted
    """
    # Use cleanup lock to prevent concurrent cleanup operations
    async with _cleanup_lock:
        try:
            logger.info("Starting cleanup of expired cloud storage files")
            handler = await get_cloud_storage_handler()
            deleted_count = await handler.delete_expired_files()
            logger.info(f"Cleaned up {deleted_count} expired files from cloud storage")
            return deleted_count
        except Exception as e:
            logger.error(f"Error during cloud storage cleanup: {e}")
            return 0
