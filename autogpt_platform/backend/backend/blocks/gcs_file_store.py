"""
GCS File Store Block for permanent file storage with automatic expiration.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict
from urllib.parse import urlparse

from google.cloud import storage
from google.cloud.storage.blob import Blob
from pydantic import BaseModel, Field

from backend.blocks.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.block import BlockInput, BlockOutput as BlockOutputData
from backend.util.request import Requests
from backend.util.settings import Config
from backend.util.type import MediaFileType

logger = logging.getLogger(__name__)


class GCSFileStoreInput(BlockSchema):
    file_in: MediaFileType = Field(
        description="The file to store permanently in GCS. Can be a URL, data URI, or local path."
    )
    custom_path: str = Field(
        default="",
        description="Optional custom path within the bucket (e.g., 'documents/myfile'). If empty, uses auto-generated path."
    )
    expiration_hours: int = Field(
        default=48,
        ge=1,
        le=48,  # Max 2 days
        description="Hours until the file expires and is automatically deleted (1-48 hours, default 48)"
    )


class GCSFileStoreOutput(BlockSchema):
    file_url: str = Field(description="The permanent URL to access the stored file")
    file_path: str = Field(description="The GCS path where the file is stored")
    expiration_time: str = Field(description="ISO timestamp when the file will expire")
    expires_in_hours: int = Field(description="Hours until the file expires")


class GCSFileStoreBlock(Block):
    """
    Store files permanently in Google Cloud Storage with automatic expiration.
    
    This block stores files in GCS with:
    - Automatic expiration (1-48 hours, default 48 hours/2 days)
    - Unique paths to prevent conflicts
    - Public read access via URLs
    - Automatic cleanup via GCS bucket lifecycle policies
    """
    
    class Input(GCSFileStoreInput):
        pass
    
    class Output(GCSFileStoreOutput):
        pass
    
    def __init__(self):
        super().__init__(
            id="f47ac10b-58cc-4372-a567-0e02b2c3d479",
            description="Store files permanently in Google Cloud Storage with automatic expiration",
            categories=[BlockCategory.MULTIMEDIA, BlockCategory.STORAGE],
            input_schema=GCSFileStoreInput,
            output_schema=GCSFileStoreOutput,
            test_input={
                "file_in": "data:text/plain;base64,SGVsbG8gV29ybGQ=",
                "custom_path": "test/hello.txt",
                "expiration_hours": 48
            },
            test_output={
                "file_url": "https://storage.googleapis.com/bucket/autogpt-temp/2024/01/01/test/hello.txt",
                "file_path": "autogpt-temp/2024/01/01/test/hello.txt",
                "expiration_time": "2024-01-03T12:00:00Z",
                "expires_in_hours": 48
            }
        )
        self.config = Config()
        self.requests = Requests()
        
        # Initialize GCS client
        self.storage_client = None
        self.bucket = None
        self._init_gcs_client()
    
    def _init_gcs_client(self):
        """Initialize Google Cloud Storage client and bucket."""
        try:
            if not self.config.media_gcs_bucket_name:
                raise ValueError("GCS bucket name not configured. Set MEDIA_GCS_BUCKET_NAME environment variable.")
            
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.config.media_gcs_bucket_name)
            
            # Test bucket access
            if not self.bucket.exists():
                raise ValueError(f"GCS bucket '{self.config.media_gcs_bucket_name}' does not exist or is not accessible.")
            
            logger.info(f"Successfully initialized GCS client for bucket: {self.config.media_gcs_bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    def _generate_file_path(self, custom_path: str, original_filename: str = None) -> str:
        """Generate a unique file path with timestamp and UUID."""
        now = datetime.utcnow()
        date_path = now.strftime("%Y/%m/%d")
        
        if custom_path:
            # Use custom path but ensure it's unique
            unique_id = str(uuid.uuid4())[:8]
            if original_filename:
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{unique_id}{ext}"
            else:
                filename = f"{os.path.basename(custom_path)}_{unique_id}"
                custom_path = os.path.dirname(custom_path)
            
            return f"autogpt-temp/{date_path}/{custom_path}/{filename}".replace("//", "/")
        else:
            # Auto-generate path
            unique_id = str(uuid.uuid4())
            if original_filename:
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{unique_id}{ext}"
            else:
                filename = f"file_{unique_id}"
            
            return f"autogpt-temp/{date_path}/{filename}"
    
    def _parse_data_uri(self, data_uri: str) -> tuple[bytes, str, str]:
        """Parse data URI and extract content, mime type, and filename."""
        try:
            if not data_uri.startswith("data:"):
                raise ValueError("Invalid data URI format")
            
            # Parse data URI: data:mime/type;base64,content
            header, data = data_uri.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "")
            
            if ";base64" in header:
                import base64
                content = base64.b64decode(data)
            else:
                content = data.encode("utf-8")
            
            # Generate filename from mime type
            ext_map = {
                "text/plain": ".txt",
                "text/html": ".html",
                "application/json": ".json",
                "application/pdf": ".pdf",
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "image/webp": ".webp",
                "video/mp4": ".mp4",
                "video/webm": ".webm",
            }
            
            ext = ext_map.get(mime_type, "")
            filename = f"file_{str(uuid.uuid4())[:8]}{ext}"
            
            return content, mime_type, filename
        
        except Exception as e:
            logger.error(f"Failed to parse data URI: {e}")
            raise ValueError(f"Invalid data URI: {e}")
    
    async def _download_from_url(self, url: str) -> tuple[bytes, str, str]:
        """Download file from URL and return content, mime type, and filename."""
        try:
            response = await self.requests.get(url)
            response.raise_for_status()
            
            content = response.content
            mime_type = response.headers.get("content-type", "application/octet-stream")
            
            # Extract filename from URL or generate one
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path) or f"download_{str(uuid.uuid4())[:8]}"
            
            return content, mime_type, filename
        
        except Exception as e:
            logger.error(f"Failed to download from URL {url}: {e}")
            raise ValueError(f"Failed to download file from URL: {e}")
    
    def _read_local_file(self, file_path: str) -> tuple[bytes, str, str]:
        """Read local file and return content, mime type, and filename."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Local file not found: {file_path}")
            
            with open(file_path, "rb") as f:
                content = f.read()
            
            # Determine mime type
            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/octet-stream"
            
            filename = os.path.basename(file_path)
            
            return content, mime_type, filename
        
        except Exception as e:
            logger.error(f"Failed to read local file {file_path}: {e}")
            raise ValueError(f"Failed to read local file: {e}")
    
    async def _upload_to_gcs(self, content: bytes, file_path: str, mime_type: str, expiration_hours: int) -> str:
        """Upload content to GCS. Files are automatically deleted after 2 days by bucket lifecycle policy."""
        try:
            # Create blob
            blob = self.bucket.blob(file_path)
            
            # Set metadata (for informational purposes only - cleanup handled by bucket lifecycle)
            blob.metadata = {
                "uploaded_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(hours=expiration_hours)).isoformat(),
                "autogpt_temp": "true"
            }
            
            # Upload content
            blob.upload_from_string(content, content_type=mime_type)
            
            # Make blob publicly readable
            blob.make_public()
            
            # Return public URL
            return blob.public_url
        
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            raise ValueError(f"Failed to upload to GCS: {e}")
    
    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """Execute the GCS file store block."""
        try:
            # Determine file source type and get content
            file_in = input_data.file_in
            
            if file_in.startswith("data:"):
                # Data URI
                content, mime_type, filename = self._parse_data_uri(file_in)
            elif file_in.startswith(("http://", "https://")):
                # URL
                content, mime_type, filename = await self._download_from_url(file_in)
            else:
                # Local file path
                content, mime_type, filename = self._read_local_file(file_in)
            
            # Generate unique file path
            file_path = self._generate_file_path(input_data.custom_path, filename)
            
            # Upload to GCS
            file_url = await self._upload_to_gcs(content, file_path, mime_type, input_data.expiration_hours)
            
            # Calculate expiration
            expiration_time = datetime.utcnow() + timedelta(hours=input_data.expiration_hours)
            
            logger.info(f"Successfully stored file in GCS: {file_path}")
            
            return BlockOutput(
                data={
                    "file_url": file_url,
                    "file_path": file_path,
                    "expiration_time": expiration_time.isoformat() + "Z",
                    "expires_in_hours": input_data.expiration_hours
                }
            )
        
        except Exception as e:
            logger.error(f"GCS file store failed: {e}")
            raise RuntimeError(f"Failed to store file in GCS: {e}")


# Register the block
GCSFileStoreBlock()