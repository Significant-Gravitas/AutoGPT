"""
Tests for cloud storage utilities.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.cloud_storage import CloudStorageConfig, CloudStorageHandler


class TestCloudStorageHandler:
    """Test cases for CloudStorageHandler."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = CloudStorageConfig()
        config.gcs_bucket_name = "test-bucket"
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

    @patch.object(CloudStorageHandler, "_get_async_gcs_client")
    @pytest.mark.asyncio
    async def test_store_file_gcs(self, mock_get_async_client, handler):
        """Test storing file in GCS."""
        # Mock async GCS client
        mock_async_client = AsyncMock()
        mock_get_async_client.return_value = mock_async_client

        # Mock the upload method
        mock_async_client.upload = AsyncMock()

        content = b"test file content"
        filename = "test.txt"

        result = await handler.store_file(content, filename, "gcs", expiration_hours=24)

        # Verify the result format
        assert result.startswith("gcs://test-bucket/uploads/")
        assert result.endswith("/test.txt")

        # Verify upload was called with correct parameters
        mock_async_client.upload.assert_called_once()
        call_args = mock_async_client.upload.call_args
        assert call_args[0][0] == "test-bucket"  # bucket name
        assert call_args[0][1].startswith("uploads/system/")  # blob name
        assert call_args[0][2] == content  # file content
        assert "metadata" in call_args[1]  # metadata argument

    @patch.object(CloudStorageHandler, "_get_async_gcs_client")
    @pytest.mark.asyncio
    async def test_retrieve_file_gcs(self, mock_get_async_client, handler):
        """Test retrieving file from GCS."""
        # Mock async GCS client
        mock_async_client = AsyncMock()
        mock_get_async_client.return_value = mock_async_client

        # Mock the download method
        mock_async_client.download = AsyncMock(return_value=b"test content")

        result = await handler.retrieve_file(
            "gcs://test-bucket/uploads/system/uuid123/file.txt"
        )

        assert result == b"test content"
        mock_async_client.download.assert_called_once_with(
            "test-bucket", "uploads/system/uuid123/file.txt"
        )

    @patch.object(CloudStorageHandler, "_get_async_gcs_client")
    @pytest.mark.asyncio
    async def test_retrieve_file_not_found(self, mock_get_async_client, handler):
        """Test retrieving non-existent file from GCS."""
        # Mock async GCS client
        mock_async_client = AsyncMock()
        mock_get_async_client.return_value = mock_async_client

        # Mock the download method to raise a 404 exception
        mock_async_client.download = AsyncMock(side_effect=Exception("404 Not Found"))

        with pytest.raises(FileNotFoundError):
            await handler.retrieve_file(
                "gcs://test-bucket/uploads/system/uuid123/nonexistent.txt"
            )

    @patch.object(CloudStorageHandler, "_get_sync_gcs_client")
    @pytest.mark.asyncio
    async def test_generate_signed_url_gcs(self, mock_get_sync_client, handler):
        """Test generating signed URL for GCS."""
        # Mock sync GCS client for signed URLs
        mock_sync_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_get_sync_client.return_value = mock_sync_client
        mock_sync_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.generate_signed_url.return_value = "https://signed-url.example.com"

        result = await handler.generate_signed_url(
            "gcs://test-bucket/uploads/system/uuid123/file.txt", 1
        )

        assert result == "https://signed-url.example.com"
        mock_blob.generate_signed_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsupported_provider(self, handler):
        """Test unsupported provider error."""
        with pytest.raises(ValueError, match="Unsupported cloud storage provider"):
            await handler.store_file(b"content", "file.txt", "unsupported")

        with pytest.raises(ValueError, match="Unsupported cloud storage path"):
            await handler.retrieve_file("unsupported://bucket/file.txt")

        with pytest.raises(ValueError, match="Unsupported cloud storage path"):
            await handler.generate_signed_url("unsupported://bucket/file.txt")

    @patch.object(CloudStorageHandler, "_get_async_gcs_client")
    @pytest.mark.asyncio
    async def test_delete_expired_files_gcs(self, mock_get_async_client, handler):
        """Test deleting expired files from GCS."""
        from datetime import datetime, timedelta, timezone

        # Mock async GCS client
        mock_async_client = AsyncMock()
        mock_get_async_client.return_value = mock_async_client

        # Mock list_objects response with expired and valid files
        expired_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        valid_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

        mock_list_response = {
            "items": [
                {"name": "uploads/expired-file.txt"},
                {"name": "uploads/valid-file.txt"},
            ]
        }
        mock_async_client.list_objects = AsyncMock(return_value=mock_list_response)

        # Mock download_metadata responses
        async def mock_download_metadata(bucket, blob_name):
            if "expired-file" in blob_name:
                return {"metadata": {"expires_at": expired_time}}
            else:
                return {"metadata": {"expires_at": valid_time}}

        mock_async_client.download_metadata = AsyncMock(
            side_effect=mock_download_metadata
        )
        mock_async_client.delete = AsyncMock()

        result = await handler.delete_expired_files("gcs")

        assert result == 1  # Only one file should be deleted
        # Verify delete was called once (for expired file)
        assert mock_async_client.delete.call_count == 1

    @patch.object(CloudStorageHandler, "_get_async_gcs_client")
    @pytest.mark.asyncio
    async def test_check_file_expired_gcs(self, mock_get_async_client, handler):
        """Test checking if a file has expired."""
        from datetime import datetime, timedelta, timezone

        # Mock async GCS client
        mock_async_client = AsyncMock()
        mock_get_async_client.return_value = mock_async_client

        # Test with expired file
        expired_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mock_async_client.download_metadata = AsyncMock(
            return_value={"metadata": {"expires_at": expired_time}}
        )

        result = await handler.check_file_expired("gcs://test-bucket/expired-file.txt")
        assert result is True

        # Test with valid file
        valid_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        mock_async_client.download_metadata = AsyncMock(
            return_value={"metadata": {"expires_at": valid_time}}
        )

        result = await handler.check_file_expired("gcs://test-bucket/valid-file.txt")
        assert result is False

    @patch("backend.util.cloud_storage.get_cloud_storage_handler")
    @pytest.mark.asyncio
    async def test_cleanup_expired_files_async(self, mock_get_handler):
        """Test the async cleanup function."""
        from backend.util.cloud_storage import cleanup_expired_files_async

        # Mock the handler
        mock_handler = mock_get_handler.return_value
        mock_handler.delete_expired_files = AsyncMock(return_value=3)

        result = await cleanup_expired_files_async()

        assert result == 3
        mock_get_handler.assert_called_once()
        mock_handler.delete_expired_files.assert_called_once()

    @patch("backend.util.cloud_storage.get_cloud_storage_handler")
    @pytest.mark.asyncio
    async def test_cleanup_expired_files_async_error(self, mock_get_handler):
        """Test the async cleanup function with error."""
        from backend.util.cloud_storage import cleanup_expired_files_async

        # Mock the handler to raise an exception
        mock_handler = mock_get_handler.return_value
        mock_handler.delete_expired_files = AsyncMock(
            side_effect=Exception("GCS error")
        )

        result = await cleanup_expired_files_async()

        assert result == 0  # Should return 0 on error
        mock_get_handler.assert_called_once()
        mock_handler.delete_expired_files.assert_called_once()

    def test_validate_file_access_system_files(self, handler):
        """Test access validation for system files."""
        # System files should be accessible by anyone
        handler._validate_file_access("uploads/system/uuid123/file.txt", None)
        handler._validate_file_access("uploads/system/uuid123/file.txt", "user123")

    def test_validate_file_access_user_files_success(self, handler):
        """Test successful access validation for user files."""
        # User should be able to access their own files
        handler._validate_file_access(
            "uploads/users/user123/uuid456/file.txt", "user123"
        )

    def test_validate_file_access_user_files_no_user_id(self, handler):
        """Test access validation failure when no user_id provided for user files."""
        with pytest.raises(
            PermissionError, match="User ID required to access user files"
        ):
            handler._validate_file_access(
                "uploads/users/user123/uuid456/file.txt", None
            )

    def test_validate_file_access_user_files_wrong_user(self, handler):
        """Test access validation failure when accessing another user's files."""
        with pytest.raises(
            PermissionError, match="Access denied: file belongs to user user123"
        ):
            handler._validate_file_access(
                "uploads/users/user123/uuid456/file.txt", "user456"
            )

    def test_validate_file_access_legacy_files(self, handler):
        """Test access validation for legacy files."""
        # Legacy files should be accessible with a warning
        handler._validate_file_access("uploads/uuid789/file.txt", None)
        handler._validate_file_access("uploads/uuid789/file.txt", "user123")

    def test_validate_file_access_invalid_path(self, handler):
        """Test access validation failure for invalid paths."""
        with pytest.raises(
            PermissionError, match="Invalid file path: must be under uploads/"
        ):
            handler._validate_file_access("invalid/path/file.txt", "user123")

    @patch.object(CloudStorageHandler, "_get_async_gcs_client")
    @pytest.mark.asyncio
    async def test_retrieve_file_with_authorization(self, mock_get_client, handler):
        """Test file retrieval with authorization."""
        # Mock async GCS client
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.download = AsyncMock(return_value=b"test content")

        # Test successful retrieval of user's own file
        result = await handler.retrieve_file(
            "gcs://test-bucket/uploads/users/user123/uuid456/file.txt",
            user_id="user123",
        )
        assert result == b"test content"
        mock_client.download.assert_called_once_with(
            "test-bucket", "uploads/users/user123/uuid456/file.txt"
        )

        # Test authorization failure
        with pytest.raises(PermissionError):
            await handler.retrieve_file(
                "gcs://test-bucket/uploads/users/user123/uuid456/file.txt",
                user_id="user456",
            )

    @patch.object(CloudStorageHandler, "_get_async_gcs_client")
    @pytest.mark.asyncio
    async def test_store_file_with_user_id(self, mock_get_client, handler):
        """Test file storage with user ID."""
        # Mock async GCS client
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.upload = AsyncMock()

        content = b"test file content"
        filename = "test.txt"

        # Test with user_id
        result = await handler.store_file(
            content, filename, "gcs", expiration_hours=24, user_id="user123"
        )

        # Verify the result format includes user path
        assert result.startswith("gcs://test-bucket/uploads/users/user123/")
        assert result.endswith("/test.txt")
        mock_client.upload.assert_called()

        # Test without user_id (system upload)
        result = await handler.store_file(
            content, filename, "gcs", expiration_hours=24, user_id=None
        )

        # Verify the result format includes system path
        assert result.startswith("gcs://test-bucket/uploads/system/")
        assert result.endswith("/test.txt")
        assert mock_client.upload.call_count == 2

    @patch.object(CloudStorageHandler, "_get_async_gcs_client")
    @pytest.mark.asyncio
    async def test_store_file_with_graph_exec_id(self, mock_get_async_client, handler):
        """Test file storage with graph execution ID."""
        # Mock async GCS client
        mock_async_client = AsyncMock()
        mock_get_async_client.return_value = mock_async_client

        # Mock the upload method
        mock_async_client.upload = AsyncMock()

        content = b"test file content"
        filename = "test.txt"

        # Test with graph_exec_id
        result = await handler.store_file(
            content, filename, "gcs", expiration_hours=24, graph_exec_id="exec123"
        )

        # Verify the result format includes execution path
        assert result.startswith("gcs://test-bucket/uploads/executions/exec123/")
        assert result.endswith("/test.txt")

    @pytest.mark.asyncio
    async def test_store_file_with_both_user_and_exec_id(self, handler):
        """Test file storage fails when both user_id and graph_exec_id are provided."""
        content = b"test file content"
        filename = "test.txt"

        with pytest.raises(
            ValueError, match="Provide either user_id OR graph_exec_id, not both"
        ):
            await handler.store_file(
                content,
                filename,
                "gcs",
                expiration_hours=24,
                user_id="user123",
                graph_exec_id="exec123",
            )

    def test_validate_file_access_execution_files_success(self, handler):
        """Test successful access validation for execution files."""
        # Graph execution should be able to access their own files
        handler._validate_file_access(
            "uploads/executions/exec123/uuid456/file.txt", graph_exec_id="exec123"
        )

    def test_validate_file_access_execution_files_no_exec_id(self, handler):
        """Test access validation failure when no graph_exec_id provided for execution files."""
        with pytest.raises(
            PermissionError,
            match="Graph execution ID required to access execution files",
        ):
            handler._validate_file_access(
                "uploads/executions/exec123/uuid456/file.txt", user_id="user123"
            )

    def test_validate_file_access_execution_files_wrong_exec_id(self, handler):
        """Test access validation failure when accessing another execution's files."""
        with pytest.raises(
            PermissionError, match="Access denied: file belongs to execution exec123"
        ):
            handler._validate_file_access(
                "uploads/executions/exec123/uuid456/file.txt", graph_exec_id="exec456"
            )

    @patch.object(CloudStorageHandler, "_get_async_gcs_client")
    @pytest.mark.asyncio
    async def test_retrieve_file_with_exec_authorization(
        self, mock_get_async_client, handler
    ):
        """Test file retrieval with execution authorization."""
        # Mock async GCS client
        mock_async_client = AsyncMock()
        mock_get_async_client.return_value = mock_async_client

        # Mock the download method
        mock_async_client.download = AsyncMock(return_value=b"test content")

        # Test successful retrieval of execution's own file
        result = await handler.retrieve_file(
            "gcs://test-bucket/uploads/executions/exec123/uuid456/file.txt",
            graph_exec_id="exec123",
        )
        assert result == b"test content"

        # Test authorization failure
        with pytest.raises(PermissionError):
            await handler.retrieve_file(
                "gcs://test-bucket/uploads/executions/exec123/uuid456/file.txt",
                graph_exec_id="exec456",
            )

    @patch.object(CloudStorageHandler, "_get_sync_gcs_client")
    @pytest.mark.asyncio
    async def test_generate_signed_url_with_exec_authorization(
        self, mock_get_sync_client, handler
    ):
        """Test signed URL generation with execution authorization."""
        # Mock sync GCS client for signed URLs
        mock_sync_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_get_sync_client.return_value = mock_sync_client
        mock_sync_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.generate_signed_url.return_value = "https://signed-url.example.com"

        # Test successful signed URL generation for execution's own file
        result = await handler.generate_signed_url(
            "gcs://test-bucket/uploads/executions/exec123/uuid456/file.txt",
            1,
            graph_exec_id="exec123",
        )
        assert result == "https://signed-url.example.com"

        # Test authorization failure
        with pytest.raises(PermissionError):
            await handler.generate_signed_url(
                "gcs://test-bucket/uploads/executions/exec123/uuid456/file.txt",
                1,
                graph_exec_id="exec456",
            )
