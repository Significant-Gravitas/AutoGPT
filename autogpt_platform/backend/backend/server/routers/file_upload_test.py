"""
Tests for file upload API endpoint.
"""

from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import UploadFile

from backend.server.routers.v1 import upload_file


class TestFileUploadAPI:
    """Test cases for file upload API."""

    @pytest.mark.asyncio
    async def test_upload_file_success(self):
        """Test successful file upload."""
        # Create mock upload file
        file_content = b"test file content"
        file_obj = BytesIO(file_content)
        upload_file_mock = UploadFile(filename="test.txt", file=file_obj)
        upload_file_mock.content_type = "text/plain"

        # Mock dependencies
        with patch("backend.server.routers.v1.scan_content_safe") as mock_scan, patch(
            "backend.server.routers.v1.get_cloud_storage_handler"
        ) as mock_handler_getter:

            mock_scan.return_value = None
            mock_handler = AsyncMock()
            mock_handler.store_file.return_value = (
                "gcs://test-bucket/uploads/123/test.txt"
            )
            mock_handler.generate_signed_url.return_value = (
                "https://signed-url.example.com"
            )
            mock_handler_getter.return_value = mock_handler

            # Mock file.read()
            upload_file_mock.read = AsyncMock(return_value=file_content)

            result = await upload_file(
                file=upload_file_mock,
                user_id="test-user-123",
                provider="gcs",
                expiration_hours=24,
            )

            # Verify result
            assert result["storage_key"] == "gcs://test-bucket/uploads/123/test.txt"
            assert result["signed_url"] == "https://signed-url.example.com"
            assert result["filename"] == "test.txt"
            assert result["size"] == len(file_content)
            assert result["content_type"] == "text/plain"
            assert result["expires_in_hours"] == 24

            # Verify virus scan was called
            mock_scan.assert_called_once_with(file_content, filename="test.txt")

            # Verify cloud storage operations
            mock_handler.store_file.assert_called_once_with(
                content=file_content, filename="test.txt", provider="gcs"
            )
            mock_handler.generate_signed_url.assert_called_once_with(
                "gcs://test-bucket/uploads/123/test.txt", expiration_hours=24
            )

    @pytest.mark.asyncio
    async def test_upload_file_no_filename(self):
        """Test file upload without filename."""
        file_content = b"test content"
        file_obj = BytesIO(file_content)
        upload_file_mock = UploadFile(filename=None, file=file_obj)
        upload_file_mock.content_type = "application/octet-stream"

        with patch("backend.server.routers.v1.scan_content_safe") as mock_scan, patch(
            "backend.server.routers.v1.get_cloud_storage_handler"
        ) as mock_handler_getter:

            mock_scan.return_value = None
            mock_handler = AsyncMock()
            mock_handler.store_file.return_value = (
                "gcs://test-bucket/uploads/123/uploaded_file"
            )
            mock_handler.generate_signed_url.return_value = (
                "https://signed-url.example.com"
            )
            mock_handler_getter.return_value = mock_handler

            upload_file_mock.read = AsyncMock(return_value=file_content)

            result = await upload_file(file=upload_file_mock, user_id="test-user-123")

            assert result["filename"] == "uploaded_file"
            assert result["content_type"] == "application/octet-stream"

            # Verify virus scan was called with default filename
            mock_scan.assert_called_once_with(file_content, filename="uploaded_file")

    @pytest.mark.asyncio
    async def test_upload_file_invalid_expiration(self):
        """Test file upload with invalid expiration hours."""
        from fastapi import HTTPException

        file_obj = BytesIO(b"content")
        upload_file_mock = UploadFile(filename="test.txt", file=file_obj)

        # Test expiration too short
        with pytest.raises(HTTPException) as exc_info:
            await upload_file(
                file=upload_file_mock, user_id="test-user-123", expiration_hours=0
            )
        assert exc_info.value.status_code == 400
        assert "between 1 and 48" in exc_info.value.detail

        # Test expiration too long
        with pytest.raises(HTTPException) as exc_info:
            await upload_file(
                file=upload_file_mock, user_id="test-user-123", expiration_hours=49
            )
        assert exc_info.value.status_code == 400
        assert "between 1 and 48" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_upload_file_virus_scan_failure(self):
        """Test file upload when virus scan fails."""
        file_content = b"malicious content"
        file_obj = BytesIO(file_content)
        upload_file_mock = UploadFile(filename="virus.txt", file=file_obj)

        with patch("backend.server.routers.v1.scan_content_safe") as mock_scan:
            # Mock virus scan to raise exception
            mock_scan.side_effect = RuntimeError("Virus detected!")

            upload_file_mock.read = AsyncMock(return_value=file_content)

            with pytest.raises(RuntimeError, match="Virus detected!"):
                await upload_file(file=upload_file_mock, user_id="test-user-123")

    @pytest.mark.asyncio
    async def test_upload_file_cloud_storage_failure(self):
        """Test file upload when cloud storage fails."""
        file_content = b"test content"
        file_obj = BytesIO(file_content)
        upload_file_mock = UploadFile(filename="test.txt", file=file_obj)

        with patch("backend.server.routers.v1.scan_content_safe") as mock_scan, patch(
            "backend.server.routers.v1.get_cloud_storage_handler"
        ) as mock_handler_getter:

            mock_scan.return_value = None
            mock_handler = AsyncMock()
            mock_handler.store_file.side_effect = RuntimeError("Storage error!")
            mock_handler_getter.return_value = mock_handler

            upload_file_mock.read = AsyncMock(return_value=file_content)

            with pytest.raises(RuntimeError, match="Storage error!"):
                await upload_file(file=upload_file_mock, user_id="test-user-123")
