# ClamAV Virus Scanning Integration

This document describes the ClamAV virus scanning integration for the AutoGPT Platform.

## Overview

The platform now includes comprehensive virus scanning for all file uploads using ClamAV antivirus engine. This provides defense-in-depth security by scanning files before they are stored or processed.

## Architecture

### Components

1. **ClamAV Docker Service**: Runs as a containerized service alongside the platform infrastructure
2. **VirusScannerService**: Python service that interfaces with ClamAV daemon
3. **Integration Points**: File upload endpoints and utilities are integrated with virus scanning

### Security Model

- **Fail-Safe Design**: If virus scanning is enabled but ClamAV is unavailable, file uploads are rejected
- **Pre-Storage Scanning**: Files are scanned before being saved to storage or filesystem
- **Comprehensive Coverage**: All file upload paths are protected (store media, execution files, data URIs, downloads)

## Configuration

### Environment Variables

Add these to your `.env` file in the backend directory:

```bash
# ClamAV Configuration
CLAMAV_HOST=localhost
CLAMAV_PORT=3310
CLAMAV_TIMEOUT=60
VIRUS_SCANNING_ENABLED=true
MAX_SCAN_SIZE=104857600  # 100MB
```

### Docker Services

ClamAV is configured in `docker-compose.yml` with environment variables for easy Kubernetes deployment:

```yaml
clamav:
  image: clamav/clamav-debian:latest
  ports:
    - "3310:3310"
  volumes:
    - clamav-data:/var/lib/clamav
  environment:
    - CLAMAV_NO_FRESHCLAMD=false
    - CLAMD_CONF_StreamMaxLength=50M      # Max stream size for network scanning
    - CLAMD_CONF_MaxFileSize=100M         # Max individual file size
    - CLAMD_CONF_MaxScanSize=100M         # Max total scan size
    - CLAMD_CONF_MaxThreads=12            # Concurrent scan threads
    - CLAMD_CONF_ReadTimeout=300          # Socket read timeout (seconds)
  healthcheck:
    test: ["CMD-SHELL", "clamdscan --version || exit 1"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Kubernetes Configuration

For Kubernetes deployments, use the same environment variables:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: clamav
spec:
  template:
    spec:
      containers:
      - name: clamav
        image: clamav/clamav-debian:latest
        env:
        - name: CLAMAV_NO_FRESHCLAMD
          value: "false"
        - name: CLAMD_CONF_StreamMaxLength
          value: "50M"
        - name: CLAMD_CONF_MaxFileSize
          value: "100M"
        - name: CLAMD_CONF_MaxScanSize
          value: "100M"
        - name: CLAMD_CONF_MaxThreads
          value: "12"
        - name: CLAMD_CONF_ReadTimeout
          value: "300"
```

## Implementation Details

### Service Integration

The `VirusScannerService` is integrated at the following points:

1. **Store Media Uploads** (`/backend/server/v2/store/media.py`)
   - Scans files before GCS upload
   - Validates file content and signatures

2. **Execution File Handling** (`/backend/util/file.py`)
   - Scans data URIs, downloaded files, and local files
   - Protects graph execution from malicious files

3. **API Error Handling** (`/backend/server/v2/store/routes.py`)
   - Returns appropriate HTTP status codes for virus detection
   - Provides detailed error information to clients

### Exception Handling

Custom exceptions are defined for virus-related errors:

- `VirusDetectedError`: Raised when a virus is found
- `VirusScanError`: Raised when scanning fails

### API Responses

When a virus is detected, the API returns:

```json
{
  "detail": "File rejected due to virus detection: Win.Test.EICAR_HDB-1",
  "error_type": "virus_detected", 
  "threat_name": "Win.Test.EICAR_HDB-1"
}
```

Status codes:
- `400 Bad Request`: Virus detected in file
- `503 Service Unavailable`: Virus scanning service unavailable

## Usage

### Development Setup

1. Start all services including ClamAV:
   ```bash
   docker compose up -d
   ```

2. Verify ClamAV is running:
   ```bash
   docker compose logs clamav
   ```

3. Install Python dependencies:
   ```bash
   cd backend && poetry install
   ```

### Testing

Run virus scanner tests:
```bash
poetry run pytest backend/services/virus_scanner_test.py -v
```

Test file uploads with EICAR test file:
```bash
curl -X POST http://localhost:8000/api/v2/store/submissions/media \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@eicar.com"
```

### Production Deployment

1. Ensure ClamAV service is included in production docker-compose
2. Configure appropriate resource limits for ClamAV container
3. Set up monitoring for ClamAV service health
4. Configure log aggregation for virus detection events

## Security Considerations

### Threat Detection

- Scans against ClamAV's virus database (updated automatically)
- Detects malware, trojans, and other malicious files
- Provides threat identification for security logging

### Performance Impact

- File scanning adds latency to uploads
- Large files are scanned in chunks to avoid stream limits
- Chunking starts at 25MB and reduces to 128KB minimum if needed
- Files larger than 100MB skip scanning with a warning
- ClamAV database updates require periodic container restarts

### Chunked Scanning Strategy

The virus scanner automatically handles large files using an adaptive chunking strategy:

1. **Initial Chunk Size**: 25MB (safe for 50MB stream limit)
2. **Size Limit Handling**: If ClamAV rejects a chunk due to size limits:
   - Chunk size is halved automatically
   - Scanning retries with smaller chunks
   - Up to 8 retry attempts
3. **Minimum Chunk Size**: 128KB (configurable)
4. **Fallback**: If minimum chunk size still fails, the file is allowed with a warning

This ensures compatibility with different ClamAV configurations and stream limits.

### Limitations

- Only scans file content, not behavioral analysis
- Effectiveness depends on ClamAV signature database freshness
- May have false positives with certain file types

## Monitoring and Logging

### Metrics to Monitor

- ClamAV service availability
- Scan success/failure rates
- Virus detection frequency
- Scan performance (latency)

### Log Events

- Virus detections (WARNING level)
- Scan failures (ERROR level)
- Service availability issues (ERROR level)
- Successful scans (INFO level)

## Troubleshooting

### Common Issues

1. **ClamAV Service Not Starting**
   - Check Docker logs: `docker compose logs clamav`
   - Verify port 3310 is available
   - Ensure sufficient disk space for virus definitions

2. **Scans Timing Out**
   - Increase `CLAMAV_TIMEOUT` setting
   - Check ClamAV service performance
   - Verify network connectivity between services

3. **False Positives**
   - Review ClamAV logs for specific threat signatures
   - Consider whitelisting specific file types if appropriate
   - Update ClamAV definitions

### Disabling Virus Scanning

For development or testing, you can disable virus scanning:

```bash
VIRUS_SCANNING_ENABLED=false
```

**Warning**: Never disable virus scanning in production environments.

## Dependencies

- `pyclamd`: Python ClamAV client library
- `clamav/clamav:latest`: Official ClamAV Docker image
- Docker Compose for orchestration

## Testing Files

For testing virus detection, use the EICAR test file:
```
X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*
```

This harmless test file is recognized by all antivirus engines and will trigger virus detection.