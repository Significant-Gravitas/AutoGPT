"""
Tests for cloud storage utilities.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.util.cloud_storage import CloudStorageConfig, CloudStorageHandler


class TestCloudStorageHandler:
    """Test cases for CloudStorageHandler."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = CloudStorageConfig()
        config.gcs_bucket_name = "test-bucket"
        config.gcs_project_id = "test-project"
        return config

    @pytest.fixture
    def handler(self, config):
        """Create a test handler."""
        return CloudStorageHandler(config)

    def test_parse_cloud_path_gcs(self, handler):
        """Test parsing GCS paths."""
        provider, path = handler.parse_cloud_path("gcs://bucket/path/to/file.txt")
        assert provider == "gcs"
        assert path == "bucket/path/to/file.txt"

    def test_parse_cloud_path_invalid(self, handler):
        """Test parsing invalid cloud paths."""
        with pytest.raises(ValueError, match="Unsupported cloud storage path"):
            handler.parse_cloud_path("invalid://path")

    def test_is_cloud_path(self, handler):
        """Test cloud path detection."""
        assert handler.is_cloud_path("gcs://bucket/file.txt")
        assert handler.is_cloud_path("s3://bucket/file.txt")
        assert handler.is_cloud_path("azure://container/file.txt")
        assert not handler.is_cloud_path("http://example.com/file.txt")
        assert not handler.is_cloud_path("/local/path/file.txt")
        assert not handler.is_cloud_path("data:text/plain;base64,SGVsbG8=")

    @patch("backend.util.cloud_storage.storage")
    @pytest.mark.asyncio
    async def test_store_file_gcs(self, mock_storage, handler):
        """Test storing file in GCS."""
        # Mock GCS client and operations
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        handler._gcs_client = mock_client

        content = b"test file content"
        filename = "test.txt"

        result = await handler.store_file(content, filename, "gcs")

        # Verify the result format
        assert result.startswith("gcs://test-bucket/uploads/")
        assert result.endswith("/test.txt")

        # Verify GCS operations were called
        mock_bucket.blob.assert_called_once()
        mock_blob.upload_from_string.assert_called_once_with(content)

    @patch("backend.util.cloud_storage.storage")
    @pytest.mark.asyncio
    async def test_retrieve_file_gcs(self, mock_storage, handler):
        """Test retrieving file from GCS."""
        # Mock GCS client and operations
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"test content"

        handler._gcs_client = mock_client

        result = await handler.retrieve_file("gcs://test-bucket/path/to/file.txt")

        assert result == b"test content"
        mock_bucket.blob.assert_called_once_with("path/to/file.txt")
        mock_blob.download_as_bytes.assert_called_once()

    @patch("backend.util.cloud_storage.storage")
    @pytest.mark.asyncio
    async def test_retrieve_file_not_found(self, mock_storage, handler):
        """Test retrieving non-existent file from GCS."""
        # Mock GCS client and operations
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        handler._gcs_client = mock_client

        with pytest.raises(FileNotFoundError):
            await handler.retrieve_file("gcs://test-bucket/nonexistent.txt")

    @patch("backend.util.cloud_storage.storage")
    @pytest.mark.asyncio
    async def test_generate_signed_url_gcs(self, mock_storage, handler):
        """Test generating signed URL for GCS."""
        # Mock GCS client and operations
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.generate_signed_url.return_value = "https://signed-url.example.com"

        handler._gcs_client = mock_client

        result = await handler.generate_signed_url("gcs://test-bucket/file.txt", 1)

        assert result == "https://signed-url.example.com"
        mock_blob.generate_signed_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsupported_provider(self, handler):
        """Test unsupported provider error."""
        with pytest.raises(ValueError, match="Unsupported cloud storage provider"):
            await handler.store_file(b"content", "file.txt", "unsupported")

        with pytest.raises(ValueError, match="Unsupported cloud storage provider"):
            await handler.retrieve_file("unsupported://bucket/file.txt")

        with pytest.raises(ValueError, match="Unsupported cloud storage provider"):
            await handler.generate_signed_url("unsupported://bucket/file.txt")
