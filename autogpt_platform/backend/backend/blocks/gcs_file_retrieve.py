"""
GCS File Retrieve Block for generating presigned URLs and accessing stored files.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from google.cloud import storage
from pydantic import BaseModel, Field

from backend.blocks.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.block import BlockInput, BlockOutput as BlockOutputData
from backend.util.settings import Config

logger = logging.getLogger(__name__)


class GCSFileRetrieveInput(BlockSchema):
    file_path: str = Field(
        description="The GCS path of the file to retrieve (e.g., 'autogpt-temp/2024/01/01/document.pdf')"
    )
    access_duration_minutes: int = Field(
        default=60,
        ge=1,
        le=10080,  # Max 7 days
        description="Duration in minutes for which the presigned URL will be valid (1-10080 minutes, default 60)"
    )
    action: str = Field(
        default="GET",
        description="HTTP method for the presigned URL (GET for download, PUT for upload)"
    )


class GCSFileRetrieveOutput(BlockSchema):
    presigned_url: str = Field(description="The presigned URL to access the file")
    file_path: str = Field(description="The GCS path of the file")
    public_url: str = Field(description="The public URL if the file is publicly accessible")
    expires_at: str = Field(description="ISO timestamp when the presigned URL expires")
    file_exists: bool = Field(description="Whether the file exists in GCS")
    file_size: Optional[int] = Field(description="Size of the file in bytes", default=None)
    file_type: Optional[str] = Field(description="MIME type of the file", default=None)
    uploaded_at: Optional[str] = Field(description="When the file was uploaded", default=None)


class GCSFileRetrieveBlock(Block):
    """
    Retrieve files from Google Cloud Storage with presigned URLs.
    
    This block provides:
    - Presigned URLs for secure file access
    - File metadata and existence checking
    - Configurable access duration
    - Support for both download and upload URLs
    """
    
    class Input(GCSFileRetrieveInput):
        pass
    
    class Output(GCSFileRetrieveOutput):
        pass
    
    def __init__(self):
        super().__init__(
            id="e93d4c7f-8b2a-4e5d-9f3c-1a8b7e6d5c4b",
            description="Retrieve files from Google Cloud Storage with presigned URLs",
            categories=[BlockCategory.MULTIMEDIA, BlockCategory.STORAGE],
            input_schema=GCSFileRetrieveInput,
            output_schema=GCSFileRetrieveOutput,
            test_input={
                "file_path": "autogpt-temp/2024/01/01/test/hello.txt",
                "access_duration_minutes": 120,
                "action": "GET"
            },
            test_output={
                "presigned_url": "https://storage.googleapis.com/bucket/autogpt-temp/2024/01/01/test/hello.txt?X-Goog-Algorithm=...",
                "file_path": "autogpt-temp/2024/01/01/test/hello.txt",
                "public_url": "https://storage.googleapis.com/bucket/autogpt-temp/2024/01/01/test/hello.txt",
                "expires_at": "2024-01-01T14:00:00Z",
                "file_exists": True,
                "file_size": 1024,
                "file_type": "text/plain",
                "uploaded_at": "2024-01-01T12:00:00Z"
            }
        )
        self.config = Config()
        
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
            
            logger.info(f"Successfully initialized GCS client for bucket: {self.config.media_gcs_bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    def _get_file_metadata(self, blob) -> Dict[str, Any]:
        """Extract metadata from GCS blob."""
        try:
            metadata = {
                "file_size": blob.size,
                "file_type": blob.content_type,
                "uploaded_at": None
            }
            
            # Get upload time from blob metadata or time_created
            if blob.metadata and "uploaded_at" in blob.metadata:
                metadata["uploaded_at"] = blob.metadata["uploaded_at"]
            elif blob.time_created:
                metadata["uploaded_at"] = blob.time_created.isoformat()
            
            return metadata
        
        except Exception as e:
            logger.warning(f"Failed to get file metadata: {e}")
            return {
                "file_size": None,
                "file_type": None,
                "uploaded_at": None
            }
    
    def _generate_presigned_url(self, blob, action: str, duration_minutes: int) -> str:
        """Generate a presigned URL for the blob."""
        try:
            expiration = datetime.utcnow() + timedelta(minutes=duration_minutes)
            
            if action.upper() == "GET":
                # Generate signed URL for download
                url = blob.generate_signed_url(
                    expiration=expiration,
                    method="GET",
                    version="v4"
                )
            elif action.upper() == "PUT":
                # Generate signed URL for upload
                url = blob.generate_signed_url(
                    expiration=expiration,
                    method="PUT",
                    version="v4"
                )
            else:
                raise ValueError(f"Unsupported action: {action}. Use 'GET' or 'PUT'.")
            
            return url
        
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise ValueError(f"Failed to generate presigned URL: {e}")
    
    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """Execute the GCS file retrieve block."""
        try:
            # Get blob reference
            blob = self.bucket.blob(input_data.file_path)
            
            # Check if file exists
            file_exists = blob.exists()
            
            # Initialize output data
            output_data = {
                "file_path": input_data.file_path,
                "file_exists": file_exists,
                "file_size": None,
                "file_type": None,
                "uploaded_at": None
            }
            
            if file_exists:
                # Reload blob to get metadata
                blob.reload()
                
                # Get file metadata
                metadata = self._get_file_metadata(blob)
                output_data.update(metadata)
                
                # Generate presigned URL
                presigned_url = self._generate_presigned_url(
                    blob, 
                    input_data.action, 
                    input_data.access_duration_minutes
                )
                
                # Generate public URL (may not be accessible if blob is private)
                public_url = blob.public_url
                
                # Calculate expiration time
                expires_at = datetime.utcnow() + timedelta(minutes=input_data.access_duration_minutes)
                
                output_data.update({
                    "presigned_url": presigned_url,
                    "public_url": public_url,
                    "expires_at": expires_at.isoformat() + "Z"
                })
                
                logger.info(f"Successfully generated presigned URL for: {input_data.file_path}")
                
            else:
                # File doesn't exist, provide empty URLs
                output_data.update({
                    "presigned_url": "",
                    "public_url": "",
                    "expires_at": ""
                })
                
                logger.warning(f"File not found in GCS: {input_data.file_path}")
            
            return BlockOutput(data=output_data)
        
        except Exception as e:
            logger.error(f"GCS file retrieve failed: {e}")
            raise RuntimeError(f"Failed to retrieve file from GCS: {e}")


# Register the block
GCSFileRetrieveBlock()