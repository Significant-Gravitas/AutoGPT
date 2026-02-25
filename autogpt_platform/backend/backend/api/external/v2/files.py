"""
V2 External API - Files Endpoints

Provides file upload functionality for agent inputs.
"""

import base64
import logging

from fastapi import APIRouter, File, HTTPException, Query, Security, UploadFile
from prisma.enums import APIKeyPermission

from backend.api.external.middleware import require_permission
from backend.data.auth.base import APIAuthorizationInfo
from backend.util.cloud_storage import get_cloud_storage_handler
from backend.util.settings import Settings
from backend.util.virus_scanner import scan_content_safe

from .models import UploadFileResponse

logger = logging.getLogger(__name__)
settings = Settings()

files_router = APIRouter()


# ============================================================================
# Endpoints
# ============================================================================


def _create_file_size_error(size_bytes: int, max_size_mb: int) -> HTTPException:
    """Create standardized file size error response."""
    return HTTPException(
        status_code=400,
        detail=f"File size ({size_bytes} bytes) exceeds the maximum allowed size of {max_size_mb}MB",
    )


@files_router.post(
    path="/upload",
    summary="Upload a file",
    response_model=UploadFileResponse,
)
async def upload_file(
    file: UploadFile = File(...),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.UPLOAD_FILES)
    ),
    provider: str = Query(
        default="gcs", description="Storage provider (gcs, s3, azure)"
    ),
    expiration_hours: int = Query(
        default=24, ge=1, le=48, description="Hours until file expires (1-48)"
    ),
) -> UploadFileResponse:
    """
    Upload a file to cloud storage for use with agents.

    The returned `file_uri` can be used as input to agents that accept file inputs
    (e.g., FileStoreBlock, AgentFileInputBlock).

    Files are automatically scanned for viruses before storage.
    """
    # Check file size limit
    max_size_mb = settings.config.upload_file_size_limit_mb
    max_size_bytes = max_size_mb * 1024 * 1024

    # Try to get file size from headers first
    if hasattr(file, "size") and file.size is not None and file.size > max_size_bytes:
        raise _create_file_size_error(file.size, max_size_mb)

    # Read file content
    content = await file.read()
    content_size = len(content)

    # Double-check file size after reading
    if content_size > max_size_bytes:
        raise _create_file_size_error(content_size, max_size_mb)

    # Extract file info
    file_name = file.filename or "uploaded_file"
    content_type = file.content_type or "application/octet-stream"

    # Virus scan the content
    await scan_content_safe(content, filename=file_name)

    # Check if cloud storage is configured
    cloud_storage = await get_cloud_storage_handler()
    if not cloud_storage.config.gcs_bucket_name:
        # Fallback to base64 data URI when GCS is not configured
        base64_content = base64.b64encode(content).decode("utf-8")
        data_uri = f"data:{content_type};base64,{base64_content}"

        return UploadFileResponse(
            file_uri=data_uri,
            file_name=file_name,
            size=content_size,
            content_type=content_type,
            expires_in_hours=expiration_hours,
        )

    # Store in cloud storage
    storage_path = await cloud_storage.store_file(
        content=content,
        filename=file_name,
        provider=provider,
        expiration_hours=expiration_hours,
        user_id=auth.user_id,
    )

    return UploadFileResponse(
        file_uri=storage_path,
        file_name=file_name,
        size=content_size,
        content_type=content_type,
        expires_in_hours=expiration_hours,
    )
