"""
Tests for GCS File Store blocks.
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.cloud import storage

from backend.blocks.gcs_file_store import GCSFileStoreBlock
from backend.blocks.gcs_file_retrieve import GCSFileRetrieveBlock


@pytest.fixture
def mock_gcs_client():
    """Mock GCS client and bucket."""
    with patch('backend.blocks.gcs_file_store.storage.Client') as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Configure mocks
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        mock_blob.public_url = "https://storage.googleapis.com/test-bucket/test-file.txt"
        mock_blob.upload_from_string = MagicMock()
        mock_blob.make_public = MagicMock()
        
        yield {
            'client': mock_client,
            'bucket': mock_bucket,
            'blob': mock_blob
        }


@pytest.fixture
def mock_requests():
    """Mock requests for URL downloads."""
    mock_response = MagicMock()
    mock_response.content = b"test file content"
    mock_response.headers = {"content-type": "text/plain"}
    mock_response.raise_for_status = MagicMock()
    
    with patch('backend.blocks.gcs_file_store.Requests') as mock_requests_class:
        mock_requests_instance = AsyncMock()
        mock_requests_instance.get.return_value = mock_response
        mock_requests_class.return_value = mock_requests_instance
        yield mock_requests_instance


class TestGCSFileStoreBlock:
    """Test the GCS File Store block."""
    
    def test_block_initialization(self, mock_gcs_client):
        """Test that the block initializes correctly."""
        with patch.dict(os.environ, {'MEDIA_GCS_BUCKET_NAME': 'test-bucket'}):
            block = GCSFileStoreBlock()
            assert block.id == "f47ac10b-58cc-4372-a567-0e02b2c3d479"
            assert "Store files permanently" in block.description
    
    def test_parse_data_uri(self, mock_gcs_client):
        """Test parsing data URIs."""
        with patch.dict(os.environ, {'MEDIA_GCS_BUCKET_NAME': 'test-bucket'}):
            block = GCSFileStoreBlock()
            
            # Test base64 data URI
            data_uri = "data:text/plain;base64,SGVsbG8gV29ybGQ="
            content, mime_type, filename = block._parse_data_uri(data_uri)
            
            assert content == b"Hello World"
            assert mime_type == "text/plain"
            assert filename.endswith(".txt")
    
    def test_generate_file_path(self, mock_gcs_client):
        """Test file path generation."""
        with patch.dict(os.environ, {'MEDIA_GCS_BUCKET_NAME': 'test-bucket'}):
            block = GCSFileStoreBlock()
            
            # Test custom path
            path = block._generate_file_path("documents/test", "example.txt")
            assert "autogpt-temp/" in path
            assert "documents/test" in path
            assert "example" in path
            assert path.endswith(".txt")
            
            # Test auto-generated path
            path = block._generate_file_path("", "example.txt")
            assert "autogpt-temp/" in path
            assert "example" in path
            assert path.endswith(".txt")
    
    @pytest.mark.asyncio
    async def test_run_with_data_uri(self, mock_gcs_client, mock_requests):
        """Test running the block with a data URI."""
        with patch.dict(os.environ, {'MEDIA_GCS_BUCKET_NAME': 'test-bucket'}):
            block = GCSFileStoreBlock()
            
            input_data = block.Input(
                file_in="data:text/plain;base64,SGVsbG8gV29ybGQ=",
                custom_path="test",
                expiration_hours=24
            )
            
            result = await block.run(input_data)
            
            assert "file_url" in result.data
            assert "file_path" in result.data
            assert "expiration_time" in result.data
            assert result.data["expires_in_hours"] == 24
            
            # Verify GCS calls
            mock_gcs_client['blob'].upload_from_string.assert_called_once()
            mock_gcs_client['blob'].make_public.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_with_url(self, mock_gcs_client, mock_requests):
        """Test running the block with a URL."""
        with patch.dict(os.environ, {'MEDIA_GCS_BUCKET_NAME': 'test-bucket'}):
            block = GCSFileStoreBlock()
            
            input_data = block.Input(
                file_in="https://example.com/test.txt",
                custom_path="downloads",
                expiration_hours=48
            )
            
            result = await block.run(input_data)
            
            assert "file_url" in result.data
            assert "file_path" in result.data
            assert result.data["expires_in_hours"] == 48
            
            # Verify URL was fetched
            mock_requests.get.assert_called_once_with("https://example.com/test.txt")
    
    @pytest.mark.asyncio
    async def test_run_with_local_file(self, mock_gcs_client):
        """Test running the block with a local file."""
        with patch.dict(os.environ, {'MEDIA_GCS_BUCKET_NAME': 'test-bucket'}):
            block = GCSFileStoreBlock()
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write("test content")
                temp_file_path = temp_file.name
            
            try:
                input_data = block.Input(
                    file_in=temp_file_path,
                    custom_path="",
                    expiration_hours=12
                )
                
                result = await block.run(input_data)
                
                assert "file_url" in result.data
                assert "file_path" in result.data
                assert result.data["expires_in_hours"] == 12
                
            finally:
                # Clean up
                os.unlink(temp_file_path)


class TestGCSFileRetrieveBlock:
    """Test the GCS File Retrieve block."""
    
    def test_block_initialization(self):
        """Test that the block initializes correctly."""
        with patch.dict(os.environ, {'MEDIA_GCS_BUCKET_NAME': 'test-bucket'}):
            with patch('backend.blocks.gcs_file_retrieve.storage.Client'):
                block = GCSFileRetrieveBlock()
                assert block.id == "e93d4c7f-8b2a-4e5d-9f3c-1a8b7e6d5c4b"
                assert "Retrieve files from" in block.description
    
    @pytest.mark.asyncio
    async def test_run_existing_file(self):
        """Test retrieving an existing file."""
        with patch.dict(os.environ, {'MEDIA_GCS_BUCKET_NAME': 'test-bucket'}):
            with patch('backend.blocks.gcs_file_retrieve.storage.Client') as mock_client:
                # Setup mocks
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_client.return_value.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob
                
                # Configure blob to exist
                mock_blob.exists.return_value = True
                mock_blob.size = 1024
                mock_blob.content_type = "text/plain"
                mock_blob.time_created = datetime.utcnow()
                mock_blob.metadata = {"uploaded_at": "2024-01-01T12:00:00"}
                mock_blob.public_url = "https://storage.googleapis.com/test-bucket/test.txt"
                mock_blob.generate_signed_url.return_value = "https://storage.googleapis.com/test-bucket/test.txt?signed=true"
                
                block = GCSFileRetrieveBlock()
                
                input_data = block.Input(
                    file_path="test/file.txt",
                    access_duration_minutes=60,
                    action="GET"
                )
                
                result = await block.run(input_data)
                
                assert result.data["file_exists"] is True
                assert result.data["file_size"] == 1024
                assert result.data["file_type"] == "text/plain"
                assert "presigned_url" in result.data
                assert "public_url" in result.data
                assert "expires_at" in result.data
    
    @pytest.mark.asyncio
    async def test_run_nonexistent_file(self):
        """Test retrieving a non-existent file."""
        with patch.dict(os.environ, {'MEDIA_GCS_BUCKET_NAME': 'test-bucket'}):
            with patch('backend.blocks.gcs_file_retrieve.storage.Client') as mock_client:
                # Setup mocks
                mock_bucket = MagicMock()
                mock_blob = MagicMock()
                mock_client.return_value.bucket.return_value = mock_bucket
                mock_bucket.blob.return_value = mock_blob
                
                # Configure blob to not exist
                mock_blob.exists.return_value = False
                
                block = GCSFileRetrieveBlock()
                
                input_data = block.Input(
                    file_path="nonexistent/file.txt",
                    access_duration_minutes=30,
                    action="GET"
                )
                
                result = await block.run(input_data)
                
                assert result.data["file_exists"] is False
                assert result.data["presigned_url"] == ""
                assert result.data["public_url"] == ""
                assert result.data["expires_at"] == ""