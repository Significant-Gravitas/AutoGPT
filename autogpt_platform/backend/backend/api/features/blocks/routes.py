import base64
import time
from collections import defaultdict
from typing import Annotated

from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    Security,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from starlette.status import HTTP_402_PAYMENT_REQUIRED

from backend.api.model import CloudStorageUploadResponse
from backend.blocks import get_block, get_blocks
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data.block import BlockInput, CompletedBlockOutput
from backend.data.execution import ExecutionContext
from backend.data.user import get_user_by_id
from backend.executor import utils as execution_utils
from backend.monitoring.instrumentation import record_block_execution
from backend.util.cache import cached
from backend.util.cloud_storage import get_cloud_storage_handler
from backend.util.exceptions import InsufficientBalanceError
from backend.util.json import dumps
from backend.util.settings import Settings
from backend.util.timezone_utils import get_user_timezone_or_utc
from backend.util.virus_scanner import scan_content_safe

settings = Settings()

# Tags stay per-route: two are "blocks" and the upload is "files".
# execute_graph_block keeps its extra Depends(enforce_payment_paywall).
router = APIRouter(dependencies=[Security(requires_user)])


def _compute_blocks_sync() -> str:
    """
    Synchronous function to compute blocks data.
    This does the heavy lifting: instantiate 226+ blocks, compute costs, serialize.
    """
    from backend.data.credit import get_block_cost

    block_classes = get_blocks()
    result = []

    for block_class in block_classes.values():
        block_instance = block_class()
        if not block_instance.disabled:
            costs = get_block_cost(block_instance)
            # Convert BlockCost BaseModel objects to dictionaries for JSON serialization
            costs_dict = [
                cost.model_dump() if isinstance(cost, BaseModel) else cost
                for cost in costs
            ]
            result.append({**block_instance.to_dict(), "costs": costs_dict})

    # Use our JSON utility which properly handles complex types through to_dict conversion
    return dumps(result)


@cached(ttl_seconds=3600)
async def _get_cached_blocks() -> str:
    """
    Async cached function with thundering herd protection.
    On cache miss: runs heavy work in thread pool
    On cache hit: returns cached string immediately (no thread pool needed)
    """
    # Only run in thread pool on cache miss - cache hits return immediately
    return await run_in_threadpool(_compute_blocks_sync)


@router.get(
    path="/blocks",
    summary="List available blocks",
    tags=["blocks"],
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "items": {"additionalProperties": True, "type": "object"},
                        "type": "array",
                        "title": "Response Getv1List Available Blocks",
                    }
                }
            },
        }
    },
)
async def get_graph_blocks() -> Response:
    # Cache hit: returns immediately, Cache miss: runs in thread pool
    content = await _get_cached_blocks()
    return Response(
        content=content,
        media_type="application/json",
    )


@router.post(
    path="/blocks/{block_id}/execute",
    summary="Execute graph block",
    tags=["blocks"],
    dependencies=[Depends(enforce_payment_paywall)],
    responses={
        402: {"description": "Subscription required (NO_TIER user, paywall on)"},
    },
)
async def execute_graph_block(
    block_id: str,
    data: BlockInput,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> CompletedBlockOutput:
    obj = get_block(block_id)
    if not obj:
        raise HTTPException(status_code=404, detail=f"Block #{block_id} not found.")
    if obj.disabled:
        raise HTTPException(status_code=403, detail=f"Block #{block_id} is disabled.")

    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    try:
        await execution_utils.charge_for_direct_block_execution(
            user_id=user_id, block=obj, input_data=data, source="internal"
        )
    except InsufficientBalanceError as e:
        raise HTTPException(status_code=HTTP_402_PAYMENT_REQUIRED, detail=str(e)) from e

    # Direct block execution has no graph; build a minimal ExecutionContext
    # carrying the caller's identity + timezone so blocks that depend on
    # those (e.g. time blocks) get correct data.
    execution_context = ExecutionContext(
        user_id=user_id,
        user_timezone=get_user_timezone_or_utc(user.timezone),
    )

    start_time = time.time()
    try:
        output = defaultdict(list)
        async for name, data in obj.execute(
            data,
            user_id=user_id,
            execution_context=execution_context,
        ):
            output[name].append(data)

        # Record successful block execution with duration
        duration = time.time() - start_time
        block_type = obj.__class__.__name__
        record_block_execution(
            block_type=block_type, status="success", duration=duration
        )

        return output
    except Exception:
        # Record failed block execution
        duration = time.time() - start_time
        block_type = obj.__class__.__name__
        record_block_execution(block_type=block_type, status="error", duration=duration)
        raise


@router.post(
    path="/files/upload",
    summary="Upload file to cloud storage",
    tags=["files"],
)
async def upload_file(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    file: UploadFile = File(...),
    expiration_hours: int = 24,
) -> CloudStorageUploadResponse:
    """
    Upload a file to cloud storage and return a storage key that can be used
    with FileStoreBlock and AgentFileInputBlock.

    Args:
        file: The file to upload
        user_id: The user ID
        provider: Cloud storage provider ("gcs", "s3", "azure")
        expiration_hours: Hours until file expires (1-48)

    Returns:
        Dict containing the cloud storage path and signed URL
    """
    if expiration_hours < 1 or expiration_hours > 48:
        raise HTTPException(
            status_code=400, detail="Expiration hours must be between 1 and 48"
        )

    # Check file size limit before reading content to avoid memory issues
    max_size_mb = settings.config.upload_file_size_limit_mb
    max_size_bytes = max_size_mb * 1024 * 1024

    # Try to get file size from headers first
    if hasattr(file, "size") and file.size is not None and file.size > max_size_bytes:
        raise _create_file_size_error(file.size, max_size_mb)

    # Read file content
    content = await file.read()
    content_size = len(content)

    # Double-check file size after reading (in case header was missing/incorrect)
    if content_size > max_size_bytes:
        raise _create_file_size_error(content_size, max_size_mb)

    # Extract common variables
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

        return CloudStorageUploadResponse(
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
        expiration_hours=expiration_hours,
        user_id=user_id,
    )

    return CloudStorageUploadResponse(
        file_uri=storage_path,
        file_name=file_name,
        size=content_size,
        content_type=content_type,
        expires_in_hours=expiration_hours,
    )


def _create_file_size_error(size_bytes: int, max_size_mb: int) -> HTTPException:
    """Create standardized file size error response."""
    return HTTPException(
        status_code=400,
        detail=f"File size ({size_bytes} bytes) exceeds the maximum allowed size of {max_size_mb}MB",
    )
