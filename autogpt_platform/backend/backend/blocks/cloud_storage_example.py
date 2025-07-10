"""
Example usage of cloud storage integration with AutoGPT blocks.

This example demonstrates how to:
1. Upload files to cloud storage via API
2. Use the returned storage keys with FileStoreBlock and AgentFileInputBlock
3. Work with cloud storage paths in agent workflows
"""

# Example 1: Upload a file via API and get storage key
"""
POST /api/v1/files/upload
Headers:
  Authorization: Bearer <your-token>
  Content-Type: multipart/form-data

Body:
  file: <your-file>
  provider: "gcs"  # optional, defaults to "gcs"
  expiration_hours: 24  # optional, defaults to 24

Response:
{
  "storage_key": "gcs://my-bucket/uploads/uuid-123/myfile.pdf",
  "signed_url": "https://storage.googleapis.com/my-bucket/uploads/uuid-123/myfile.pdf?X-Goog-Algorithm=...",
  "filename": "myfile.pdf",
  "size": 12345,
  "content_type": "application/pdf",
  "expires_in_hours": 24
}
"""

# Example 2: Use storage key with FileStoreBlock
"""
In your agent workflow, you can now use the storage_key directly:

{
  "id": "file_store_node",
  "block_id": "cbb50872-625b-42f0-8203-a2ae78242d8a",  # FileStoreBlock ID
  "input_default": {
    "file_in": "gcs://my-bucket/uploads/uuid-123/myfile.pdf"  # Use the storage_key
  }
}

The FileStoreBlock will:
1. Detect this is a cloud storage path (starts with "gcs://")
2. Download the file from cloud storage
3. Store it locally in the temp directory for the execution
4. Return the local path for use by other blocks
"""

# Example 3: Use storage key with AgentFileInputBlock
"""
{
  "id": "file_input_node", 
  "block_id": "95ead23f-8283-4654-aef3-10c053b74a31",  # AgentFileInputBlock ID
  "input_default": {
    "value": "gcs://my-bucket/uploads/uuid-123/image.png"  # Use the storage_key
  }
}

The AgentFileInputBlock will similarly:
1. Download from cloud storage
2. Store locally 
3. Return the local path
"""

# Example 4: Complete workflow with cloud storage
"""
Here's a complete agent workflow that uses cloud storage:

1. User uploads file via API: POST /api/v1/files/upload
   → Gets storage_key: "gcs://bucket/uploads/123/document.pdf"

2. Agent workflow processes the file:
   - FileStoreBlock downloads and stores locally
   - TextExtractionBlock processes the local file
   - OutputBlock returns results

3. Benefits:
   - Files persist across agent runs (until expiration)
   - Secure access via signed URLs
   - Automatic virus scanning
   - Support for large files
   - Cross-execution file sharing
"""

# Example 5: Environment configuration
"""
To enable cloud storage, set these environment variables:

# For Google Cloud Storage
GCS_BUCKET_NAME=your-bucket-name
GCS_PROJECT_ID=your-project-id
GCS_CREDENTIALS_PATH=/path/to/service-account.json  # Optional, uses default auth if not set

# Future providers (not yet implemented)
# AWS_BUCKET_NAME=your-s3-bucket
# AZURE_CONTAINER_NAME=your-azure-container
"""

# Example 6: Programmatic usage
"""
You can also use the cloud storage handler directly in your code:

from backend.util.cloud_storage import get_cloud_storage_handler

async def example_usage():
    handler = get_cloud_storage_handler()
    
    # Store a file
    storage_path = await handler.store_file(
        content=b"file content",
        filename="example.txt", 
        provider="gcs"
    )
    print(f"Stored at: {storage_path}")
    
    # Retrieve the file
    content = await handler.retrieve_file(storage_path)
    print(f"Retrieved: {content}")
    
    # Generate signed URL for sharing
    signed_url = await handler.generate_signed_url(storage_path, expiration_hours=1)
    print(f"Share via: {signed_url}")
"""
