"""
Tests for cloud storage integration in file utilities.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.execution import ExecutionContext
from backend.util.file import store_media_file
from backend.util.type import MediaFileType


def make_test_context(
    graph_exec_id: str = "test-exec-123",
    user_id: str = "test-user-123",
) -> ExecutionContext:
    """Helper to create test ExecutionContext."""
    return ExecutionContext(
        user_id=user_id,
        graph_exec_id=graph_exec_id,
    )


class TestFileCloudIntegration:
    """Test cases for cloud storage integration in file utilities."""

    @pytest.mark.asyncio
    async def test_store_media_file_cloud_path(self):
        """Test storing a file from cloud storage path."""
        graph_exec_id = "test-exec-123"
        cloud_path = "gcs://test-bucket/uploads/456/source.txt"
        cloud_content = b"cloud file content"

        with patch(
            "backend.util.file.get_cloud_storage_handler"
        ) as mock_handler_getter, patch(
            "backend.util.file.scan_content_safe"
        ) as mock_scan, patch(
            "backend.util.file.Path"
        ) as mock_path_class:

            # Mock cloud storage handler
            mock_handler = MagicMock()
            mock_handler.is_cloud_path.return_value = True
            mock_handler.parse_cloud_path.return_value = (
                "gcs",
                "test-bucket/uploads/456/source.txt",
            )
            mock_handler.retrieve_file = AsyncMock(return_value=cloud_content)
            mock_handler_getter.return_value = mock_handler

            # Mock virus scanner
            mock_scan.return_value = None

            # Mock file system operations
            mock_base_path = MagicMock()
            mock_target_path = MagicMock()
            mock_resolved_path = MagicMock()

            mock_path_class.return_value = mock_base_path
            mock_base_path.mkdir = MagicMock()
            mock_base_path.__truediv__ = MagicMock(return_value=mock_target_path)
            mock_target_path.resolve.return_value = mock_resolved_path
            mock_resolved_path.is_relative_to.return_value = True
            mock_resolved_path.write_bytes = MagicMock()
            mock_resolved_path.relative_to.return_value = Path("source.txt")

            # Configure the main Path mock to handle filename extraction
            # When Path(path_part) is called, it should return a mock with .name = "source.txt"
            mock_path_for_filename = MagicMock()
            mock_path_for_filename.name = "source.txt"

            # The Path constructor should return different mocks for different calls
            def path_constructor(*args, **kwargs):
                if len(args) == 1 and "source.txt" in str(args[0]):
                    return mock_path_for_filename
                else:
                    return mock_base_path

            mock_path_class.side_effect = path_constructor

            result = await store_media_file(
                file=MediaFileType(cloud_path),
                execution_context=make_test_context(graph_exec_id=graph_exec_id),
                return_format="for_local_processing",
            )

            # Verify cloud storage operations
            mock_handler.is_cloud_path.assert_called_once_with(cloud_path)
            mock_handler.parse_cloud_path.assert_called_once_with(cloud_path)
            mock_handler.retrieve_file.assert_called_once_with(
                cloud_path, user_id="test-user-123", graph_exec_id=graph_exec_id
            )

            # Verify virus scan
            mock_scan.assert_called_once_with(cloud_content, filename="source.txt")

            # Verify file operations
            mock_resolved_path.write_bytes.assert_called_once_with(cloud_content)

            # Result should be the relative path
            assert str(result) == "source.txt"

    @pytest.mark.asyncio
    async def test_store_media_file_cloud_path_return_content(self):
        """Test storing a file from cloud storage and returning content."""
        graph_exec_id = "test-exec-123"
        cloud_path = "gcs://test-bucket/uploads/456/image.png"
        cloud_content = b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR"  # PNG header

        with patch(
            "backend.util.file.get_cloud_storage_handler"
        ) as mock_handler_getter, patch(
            "backend.util.file.scan_content_safe"
        ) as mock_scan, patch(
            "backend.util.file.get_mime_type"
        ) as mock_mime, patch(
            "backend.util.file.base64.b64encode"
        ) as mock_b64, patch(
            "backend.util.file.Path"
        ) as mock_path_class:

            # Mock cloud storage handler
            mock_handler = MagicMock()
            mock_handler.is_cloud_path.return_value = True
            mock_handler.parse_cloud_path.return_value = (
                "gcs",
                "test-bucket/uploads/456/image.png",
            )
            mock_handler.retrieve_file = AsyncMock(return_value=cloud_content)
            mock_handler_getter.return_value = mock_handler

            # Mock other operations
            mock_scan.return_value = None
            mock_mime.return_value = "image/png"
            mock_b64.return_value.decode.return_value = "iVBORw0KGgoAAAANSUhEUgA="

            # Mock file system operations
            mock_base_path = MagicMock()
            mock_target_path = MagicMock()
            mock_resolved_path = MagicMock()

            mock_path_class.return_value = mock_base_path
            mock_base_path.mkdir = MagicMock()
            mock_base_path.__truediv__ = MagicMock(return_value=mock_target_path)
            mock_target_path.resolve.return_value = mock_resolved_path
            mock_resolved_path.is_relative_to.return_value = True
            mock_resolved_path.write_bytes = MagicMock()
            mock_resolved_path.read_bytes.return_value = cloud_content

            # Mock Path constructor for filename extraction
            mock_path_obj = MagicMock()
            mock_path_obj.name = "image.png"
            with patch("backend.util.file.Path", return_value=mock_path_obj):
                result = await store_media_file(
                    file=MediaFileType(cloud_path),
                    execution_context=make_test_context(graph_exec_id=graph_exec_id),
                    return_format="for_external_api",
                )

            # Verify result is a data URI
            assert str(result).startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_store_media_file_non_cloud_path(self):
        """Test that non-cloud paths are handled normally."""
        graph_exec_id = "test-exec-123"
        data_uri = "data:text/plain;base64,SGVsbG8gd29ybGQ="

        with patch(
            "backend.util.file.get_cloud_storage_handler"
        ) as mock_handler_getter, patch(
            "backend.util.file.scan_content_safe"
        ) as mock_scan, patch(
            "backend.util.file.base64.b64decode"
        ) as mock_b64decode, patch(
            "backend.util.file.uuid.uuid4"
        ) as mock_uuid, patch(
            "backend.util.file.Path"
        ) as mock_path_class:

            # Mock cloud storage handler
            mock_handler = MagicMock()
            mock_handler.is_cloud_path.return_value = False
            mock_handler.retrieve_file = (
                AsyncMock()
            )  # Add this even though it won't be called
            mock_handler_getter.return_value = mock_handler

            # Mock other operations
            mock_scan.return_value = None
            mock_b64decode.return_value = b"Hello world"
            mock_uuid.return_value = "test-uuid-789"

            # Mock file system operations
            mock_base_path = MagicMock()
            mock_target_path = MagicMock()
            mock_resolved_path = MagicMock()

            mock_path_class.return_value = mock_base_path
            mock_base_path.mkdir = MagicMock()
            mock_base_path.__truediv__ = MagicMock(return_value=mock_target_path)
            mock_target_path.resolve.return_value = mock_resolved_path
            mock_resolved_path.is_relative_to.return_value = True
            mock_resolved_path.write_bytes = MagicMock()
            mock_resolved_path.relative_to.return_value = Path("test-uuid-789.txt")

            await store_media_file(
                file=MediaFileType(data_uri),
                execution_context=make_test_context(graph_exec_id=graph_exec_id),
                return_format="for_local_processing",
            )

            # Verify cloud handler was checked but not used for retrieval
            mock_handler.is_cloud_path.assert_called_once_with(data_uri)
            mock_handler.retrieve_file.assert_not_called()

            # Verify normal data URI processing occurred
            mock_b64decode.assert_called_once()
            mock_resolved_path.write_bytes.assert_called_once_with(b"Hello world")

    @pytest.mark.asyncio
    async def test_store_media_file_cloud_retrieval_error(self):
        """Test error handling when cloud retrieval fails."""
        graph_exec_id = "test-exec-123"
        cloud_path = "gcs://test-bucket/nonexistent.txt"

        with patch(
            "backend.util.file.get_cloud_storage_handler"
        ) as mock_handler_getter:

            # Mock cloud storage handler to raise error
            mock_handler = AsyncMock()
            mock_handler.is_cloud_path.return_value = True
            mock_handler.retrieve_file.side_effect = FileNotFoundError(
                "File not found in cloud storage"
            )
            mock_handler_getter.return_value = mock_handler

            with pytest.raises(
                FileNotFoundError, match="File not found in cloud storage"
            ):
                await store_media_file(
                    file=MediaFileType(cloud_path),
                    execution_context=make_test_context(graph_exec_id=graph_exec_id),
                    return_format="for_local_processing",
                )
