# GCS Permanent File Storage

This document describes the new Google Cloud Storage (GCS) based permanent file storage functionality for the AutoGPT platform.

## Overview

The GCS file storage system provides permanent file storage with automatic expiration, replacing the need for base64 data transfer with efficient presigned URLs. Files are stored in Google Cloud Storage and can persist between runs, unlike the existing FileStore block which only provides execution-scoped temporary storage.

## Key Features

- **Permanent Storage**: Files persist between agent runs
- **Automatic Expiration**: Files expire after 1-48 hours (configurable, default 48 hours/2 days)
- **Presigned URLs**: Secure, time-limited access URLs instead of base64 data
- **Multiple Input Types**: Supports data URIs, HTTP URLs, and local file paths
- **Organized Storage**: Files stored in timestamped directories with unique identifiers
- **Lifecycle Management**: GCS bucket lifecycle policies automatically delete expired files
- **Security**: Virus scanning and file validation (inherits from existing security measures)

## Blocks

### 1. GCSFileStoreBlock

**ID**: `f47ac10b-58cc-4372-a567-0e02b2c3d479`

Stores files permanently in Google Cloud Storage with automatic expiration.

**Input Schema**:
```json
{
  "file_in": "MediaFileType - The file to store (URL, data URI, or local path)",
  "custom_path": "string - Optional custom path within bucket (default: auto-generated)",
  "expiration_hours": "integer - Hours until expiration (1-48, default: 48)"
}
```

**Output Schema**:
```json
{
  "file_url": "string - The permanent URL to access the stored file",
  "file_path": "string - The GCS path where the file is stored",
  "expiration_time": "string - ISO timestamp when the file will expire",
  "expires_in_hours": "integer - Hours until the file expires"
}
```

**Example Usage**:
```json
{
  "file_in": "data:text/plain;base64,SGVsbG8gV29ybGQ=",
  "custom_path": "documents/myfile",
  "expiration_hours": 24
}
```

### 2. GCSFileRetrieveBlock

**ID**: `e93d4c7f-8b2a-4e5d-9f3c-1a8b7e6d5c4b`

Retrieves files from Google Cloud Storage with presigned URLs.

**Input Schema**:
```json
{
  "file_path": "string - The GCS path of the file to retrieve",
  "access_duration_minutes": "integer - Duration for presigned URL validity (1-2880, default: 60)",
  "action": "string - HTTP method for presigned URL (GET/PUT, default: GET)"
}
```

**Output Schema**:
```json
{
  "presigned_url": "string - The presigned URL to access the file",
  "file_path": "string - The GCS path of the file",
  "public_url": "string - The public URL if file is publicly accessible",
  "expires_at": "string - ISO timestamp when the presigned URL expires",
  "file_exists": "boolean - Whether the file exists in GCS",
  "file_size": "integer - Size of the file in bytes (optional)",
  "file_type": "string - MIME type of the file (optional)",
  "uploaded_at": "string - When the file was uploaded (optional)"
}
```

## Storage Structure

Files are stored in GCS with the following path structure:

```
autogpt-temp/
├── 2024/
│   ├── 01/
│   │   ├── 01/
│   │   │   ├── custom_path/
│   │   │   │   └── filename_uuid.ext
│   │   │   └── auto_generated_uuid.ext
│   │   └── 02/
│   └── 02/
└── 2025/
```

**Path Components**:
- `autogpt-temp/`: Root prefix for temporary files
- `YYYY/MM/DD/`: Date-based organization
- `custom_path/`: User-specified path (if provided)
- `filename_uuid.ext`: Original filename with unique identifier

## Configuration

### Environment Variables

```bash
# Required: GCS bucket name for media storage
MEDIA_GCS_BUCKET_NAME=your-gcs-bucket-name

# Optional: Google Cloud credentials (if not using default service account)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### GCS Bucket Setup

1. **Create GCS Bucket**:
   ```bash
   gsutil mb gs://your-bucket-name
   ```

2. **Set Bucket Permissions**:
   ```bash
   # For service account
   gsutil iam ch serviceAccount:your-service-account@project.iam.gserviceaccount.com:objectAdmin gs://your-bucket-name
   
   # For public read access (if needed)
   gsutil iam ch allUsers:objectViewer gs://your-bucket-name
   ```

3. **Set Lifecycle Rules** (Required for automatic cleanup):
   ```json
   {
     "rule": [
       {
         "action": {"type": "Delete"},
         "condition": {
           "age": 2,
           "matchesPrefix": ["autogpt-temp/"]
         }
       }
     ]
   }
   ```
   
   Apply with:
   ```bash
   # Save the above JSON to lifecycle.json
   gsutil lifecycle set lifecycle.json gs://your-bucket-name
   ```

### Authentication

The blocks use Google Cloud Storage client libraries with Application Default Credentials (ADC):

1. **Local Development**: 
   ```bash
   gcloud auth application-default login
   ```

2. **Production**: Use service account with appropriate permissions
3. **Google Cloud**: Workload Identity or attached service account


## Security Considerations

### File Validation
- **Virus Scanning**: Integration with existing ClamAV scanning (inherits from FileStore)
- **Content Validation**: MIME type verification and magic byte checking
- **Size Limits**: Configurable file size limits
- **Path Validation**: Prevents directory traversal attacks

### Access Control
- **Presigned URLs**: Time-limited access with configurable duration
- **Unique Paths**: UUID-based naming prevents path guessing
- **Metadata**: Files include upload timestamp and expiration metadata

### Best Practices
1. **Minimize Expiration Time**: Use shortest necessary expiration period
2. **Regular Cleanup**: Monitor and clean up expired files
3. **Access Logging**: Enable GCS access logging for audit trails
4. **Bucket Policies**: Implement appropriate IAM policies
5. **Network Security**: Use HTTPS for all file transfers

## Migration from FileStore

### Key Differences

| Feature | FileStore (Local) | GCSFileStore (Cloud) |
|---------|------------------|---------------------|
| **Persistence** | Execution-scoped | Permanent (until expiration) |
| **Storage Location** | Local temp directory | Google Cloud Storage |
| **File Access** | Local file paths | HTTP URLs / Presigned URLs |
| **Data Transfer** | Base64 encoding | Direct URL access |
| **Cleanup** | Automatic on execution end | GCS bucket lifecycle policies |
| **Scalability** | Limited by local storage | Cloud-scale storage |
| **Cross-run Access** | No | Yes |

### Migration Steps

1. **Update Workflows**: Replace FileStore blocks with GCSFileStore blocks
2. **Update File Handling**: Change from local paths to URLs
3. **Configure GCS**: Set up bucket and authentication
4. **Test Integration**: Verify file upload/download functionality
5. **Monitor Usage**: Set up monitoring and verify lifecycle policies

### Compatibility Notes

- **Input Compatibility**: GCSFileStore accepts the same input types as FileStore
- **Output Changes**: Returns URLs instead of local paths
- **Block Chaining**: Other blocks may need updates to handle URLs instead of paths

## Monitoring and Observability

### Metrics to Monitor

1. **File Storage**:
   - Number of files stored per day
   - Total storage usage
   - Average file size
   - Storage costs

2. **Lifecycle Management**:
   - Files automatically deleted by lifecycle policies
   - Storage space reclaimed
   - Lifecycle policy effectiveness

3. **Access Patterns**:
   - Presigned URL generation frequency
   - File access patterns
   - Error rates

### Logging

The system provides comprehensive logging:

```python
# Storage operations
logger.info(f"Successfully stored file in GCS: {file_path}")

# Lifecycle management
logger.info(f"GCS lifecycle policies managing file expiration automatically")

# Error handling
logger.error(f"Failed to upload to GCS: {error}")
```

### Monitoring

Monitor GCS bucket lifecycle policies for:
- Automatic file deletion effectiveness
- Storage usage patterns
- Lifecycle policy compliance

## Cost Optimization

### Storage Costs
- **Storage Class**: Use Standard storage for active files
- **Lifecycle Rules**: Automatic deletion after expiration
- **Geographic Location**: Choose appropriate region for performance/cost

### Transfer Costs
- **Presigned URLs**: Reduce server bandwidth usage
- **Direct Access**: Clients download directly from GCS
- **Caching**: Consider CDN for frequently accessed files

### Best Practices
1. **Monitor Usage**: Regular cost analysis
2. **Optimize Expiration**: Balance functionality vs. storage costs
3. **Monitor Lifecycle Policies**: Ensure automatic cleanup is working effectively
4. **Use Appropriate Storage Classes**: Consider Nearline/Coldline for archival

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   ```
   Error: Failed to initialize GCS client
   Solution: Check GOOGLE_APPLICATION_CREDENTIALS and service account permissions
   ```

2. **Bucket Access Errors**:
   ```
   Error: GCS bucket 'bucket-name' does not exist or is not accessible
   Solution: Verify bucket name and IAM permissions
   ```

3. **File Upload Failures**:
   ```
   Error: Failed to upload to GCS
   Solution: Check file size limits, network connectivity, and permissions
   ```

4. **Presigned URL Errors**:
   ```
   Error: Failed to generate presigned URL
   Solution: Verify service account has signing permissions
   ```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger('backend.blocks.gcs_file_store').setLevel(logging.DEBUG)
logging.getLogger('backend.util.gcs_lifecycle').setLevel(logging.DEBUG)
```

### Health Checks

Implement health checks to verify GCS connectivity:

```python
def check_gcs_health():
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        return bucket.exists()
    except Exception as e:
        logger.error(f"GCS health check failed: {e}")
        return False
```

## Future Enhancements

### Planned Features

1. **Advanced Lifecycle Management**:
   - Multiple storage classes
   - Automated archival
   - Custom retention policies

2. **Enhanced Security**:
   - File encryption at rest
   - Advanced access controls
   - Audit logging integration

3. **Performance Optimization**:
   - CDN integration
   - Regional replication
   - Caching strategies

4. **User Experience**:
   - Progress tracking for uploads
   - Batch operations
   - File management UI

### Integration Opportunities

1. **Workflow Builder**: Visual file storage configuration
2. **Analytics**: Storage usage analytics and reporting  
3. **Backup**: Integration with backup systems
4. **Search**: File content indexing and search capabilities

## Support and Resources

- **Documentation**: This file and inline code comments
- **Tests**: Comprehensive test suite in `test/blocks/test_gcs_file_store.py`
- **Examples**: Test cases demonstrate usage patterns
- **Monitoring**: Built-in logging and Discord notifications