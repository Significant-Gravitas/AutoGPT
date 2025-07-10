"""
Cloud storage utilities for handling various cloud storage providers.
"""

import uuid
from typing import Tuple


class CloudStorageConfig:
    """Configuration for cloud storage providers."""

    def __init__(self):
        import os

        # GCS configuration from environment variables
        self.gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
        self.gcs_credentials_path = os.getenv("GCS_CREDENTIALS_PATH")
        self.gcs_project_id = os.getenv("GCS_PROJECT_ID")

        # Future providers can be added here
        # self.aws_bucket_name = self.settings.get("AWS_BUCKET_NAME")
        # self.azure_container_name = self.settings.get("AZURE_CONTAINER_NAME")


class CloudStorageHandler:
    """Generic cloud storage handler that can work with multiple providers."""

    def __init__(self, config: CloudStorageConfig):
        self.config = config
        self._gcs_client = None

    def _get_gcs_client(self):
        """Lazy initialization of GCS client."""
        if self._gcs_client is None:
            try:
                from google.cloud import storage

                if self.config.gcs_credentials_path:
                    self._gcs_client = storage.Client.from_service_account_json(
                        self.config.gcs_credentials_path
                    )
                else:
                    # Use default credentials (useful for GCE, Cloud Run, etc.)
                    self._gcs_client = storage.Client(
                        project=self.config.gcs_project_id
                    )
            except ImportError:
                raise ImportError(
                    "Google Cloud Storage client not available. "
                    "Install with: pip install google-cloud-storage"
                )
        return self._gcs_client

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
        self, content: bytes, filename: str, provider: str = "gcs"
    ) -> str:
        """
        Store file content in cloud storage.

        Args:
            content: File content as bytes
            filename: Desired filename
            provider: Cloud storage provider ("gcs", "s3", "azure")

        Returns:
            Cloud storage path (e.g., "gcs://bucket/path/to/file")
        """
        if provider == "gcs":
            return await self._store_file_gcs(content, filename)
        else:
            raise ValueError(f"Unsupported cloud storage provider: {provider}")

    async def _store_file_gcs(self, content: bytes, filename: str) -> str:
        """Store file in Google Cloud Storage."""
        if not self.config.gcs_bucket_name:
            raise ValueError("GCS_BUCKET_NAME not configured")

        client = self._get_gcs_client()
        bucket = client.bucket(self.config.gcs_bucket_name)

        # Generate unique path to avoid conflicts
        unique_id = str(uuid.uuid4())
        blob_name = f"uploads/{unique_id}/{filename}"

        blob = bucket.blob(blob_name)
        blob.upload_from_string(content)

        # Set expiration time (optional - can be configured via bucket lifecycle)
        # blob.metadata = {"uploaded_at": datetime.utcnow().isoformat()}

        return f"gcs://{self.config.gcs_bucket_name}/{blob_name}"

    async def retrieve_file(self, cloud_path: str) -> bytes:
        """
        Retrieve file content from cloud storage.

        Args:
            cloud_path: Cloud storage path (e.g., "gcs://bucket/path/to/file")

        Returns:
            File content as bytes
        """
        provider, path = self.parse_cloud_path(cloud_path)

        if provider == "gcs":
            return await self._retrieve_file_gcs(path)
        else:
            raise ValueError(f"Unsupported cloud storage provider: {provider}")

    async def _retrieve_file_gcs(self, path: str) -> bytes:
        """Retrieve file from Google Cloud Storage."""
        # Parse bucket and blob name from path
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid GCS path: {path}")

        bucket_name, blob_name = parts

        client = self._get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            raise FileNotFoundError(f"File not found: gcs://{path}")

        return blob.download_as_bytes()

    async def generate_signed_url(
        self, cloud_path: str, expiration_hours: int = 1
    ) -> str:
        """
        Generate a signed URL for temporary access to a cloud storage file.

        Args:
            cloud_path: Cloud storage path
            expiration_hours: URL expiration in hours

        Returns:
            Signed URL string
        """
        provider, path = self.parse_cloud_path(cloud_path)

        if provider == "gcs":
            return await self._generate_signed_url_gcs(path, expiration_hours)
        else:
            raise ValueError(f"Unsupported cloud storage provider: {provider}")

    async def _generate_signed_url_gcs(self, path: str, expiration_hours: int) -> str:
        """Generate signed URL for GCS."""
        from datetime import datetime, timedelta

        # Parse bucket and blob name from path
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid GCS path: {path}")

        bucket_name, blob_name = parts

        client = self._get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Generate signed URL
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.utcnow() + timedelta(hours=expiration_hours),
            method="GET",
        )

        return url


# Global instance
_cloud_storage_handler = None


def get_cloud_storage_handler() -> CloudStorageHandler:
    """Get the global cloud storage handler instance."""
    global _cloud_storage_handler
    if _cloud_storage_handler is None:
        config = CloudStorageConfig()
        _cloud_storage_handler = CloudStorageHandler(config)
    return _cloud_storage_handler
