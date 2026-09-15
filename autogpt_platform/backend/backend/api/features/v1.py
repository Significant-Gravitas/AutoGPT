import asyncio
import base64
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Annotated, get_args

import pydantic
from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    Security,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from prisma.enums import BriefingFrequency
from pydantic import BaseModel
from starlette.status import HTTP_402_PAYMENT_REQUIRED

from backend.api.model import (
    TimezoneResponse,
    UpdateTimezoneRequest,
    UploadFileResponse,
)
from backend.blocks import get_block, get_blocks
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data.block import BlockInput, CompletedBlockOutput
from backend.data.execution import ExecutionContext
from backend.data.model import UserOnboarding
from backend.data.notifications import NotificationPreference, NotificationPreferenceDTO
from backend.data.onboarding import (
    FrontendOnboardingStep,
    OnboardingStep,
    UserOnboardingUpdate,
    complete_onboarding_step,
    format_onboarding_for_extraction,
    get_recommended_agents,
    get_user_onboarding,
    reset_user_onboarding,
    update_user_onboarding,
)
from backend.data.tally import extract_business_understanding
from backend.data.understanding import (
    BusinessUnderstandingInput,
    upsert_business_understanding,
)
from backend.data.user import (
    get_or_create_user,
    get_or_create_user_with_status,
    get_user_by_id,
    get_user_notification_preference,
    update_user_email,
    update_user_notification_preference,
    update_user_timezone,
    verify_preference_token,
)
from backend.executor import utils as execution_utils
from backend.monitoring.instrumentation import record_block_execution
from backend.util.cache import cached
from backend.util.cloud_storage import get_cloud_storage_handler
from backend.util.exceptions import InsufficientBalanceError
from backend.util.json import dumps
from backend.util.settings import Settings
from backend.util.timezone_utils import get_user_timezone_or_utc
from backend.util.virus_scanner import scan_content_safe

from .store.model import StoreAgentDetails


def _create_file_size_error(size_bytes: int, max_size_mb: int) -> HTTPException:
    """Create standardized file size error response."""
    return HTTPException(
        status_code=400,
        detail=f"File size ({size_bytes} bytes) exceeds the maximum allowed size of {max_size_mb}MB",
    )


settings = Settings()
logger = logging.getLogger(__name__)


# Define the API routes
v1_router = APIRouter()


########################################################
##################### Auth #############################
########################################################


_tally_background_tasks: set[asyncio.Task] = set()
USER_CREATED_HEADER = "X-AutoGPT-User-Created"


@v1_router.post(
    "/auth/user",
    summary="Get or create user",
    tags=["auth"],
    responses={
        200: {
            "description": "Successful Response",
            "headers": {
                USER_CREATED_HEADER: {
                    "description": "Whether this request created a new user",
                    "schema": {"type": "string", "enum": ["true", "false"]},
                }
            },
        }
    },
    dependencies=[Security(requires_user)],
)
async def get_or_create_user_route(
    response: Response, user_data: dict = Security(get_jwt_payload)
):
    result = await get_or_create_user_with_status(user_data)
    response.headers[USER_CREATED_HEADER] = str(result.was_created).lower()
    user = result.user

    # Fire-and-forget: populate business understanding from Tally form.
    age_seconds = (datetime.now(timezone.utc) - user.created_at).total_seconds()
    if age_seconds < 30:
        try:
            from backend.data.tally import populate_understanding_from_tally

            task = asyncio.create_task(
                populate_understanding_from_tally(user.id, user.email)
            )
            _tally_background_tasks.add(task)
            task.add_done_callback(_tally_background_tasks.discard)
        except Exception:
            logger.debug("Failed to start Tally population task", exc_info=True)

    return user.model_dump()


@v1_router.post(
    "/auth/user/email",
    summary="Update user email",
    tags=["auth"],
    dependencies=[Security(requires_user)],
)
async def update_user_email_route(
    user_id: Annotated[str, Security(get_user_id)], email: str = Body(...)
) -> dict[str, str]:
    await update_user_email(user_id, email)

    return {"email": email}


@v1_router.get(
    "/auth/user/timezone",
    summary="Get user timezone",
    tags=["auth"],
    dependencies=[Security(requires_user)],
)
async def get_user_timezone_route(
    user_data: dict = Security(get_jwt_payload),
) -> TimezoneResponse:
    """Get user timezone setting."""
    user = await get_or_create_user(user_data)
    return TimezoneResponse(timezone=user.timezone)


@v1_router.post(
    "/auth/user/timezone",
    summary="Update user timezone",
    tags=["auth"],
    dependencies=[Security(requires_user)],
)
async def update_user_timezone_route(
    user_id: Annotated[str, Security(get_user_id)], request: UpdateTimezoneRequest
) -> TimezoneResponse:
    """Update user timezone. The timezone should be a valid IANA timezone identifier."""
    user = await update_user_timezone(user_id, str(request.timezone))
    return TimezoneResponse(timezone=user.timezone)


@v1_router.get(
    "/auth/user/preferences",
    summary="Get notification preferences",
    tags=["auth"],
    dependencies=[Security(requires_user)],
)
async def get_preferences(
    user_id: Annotated[str, Security(get_user_id)],
) -> NotificationPreference:
    preferences = await get_user_notification_preference(user_id)
    return preferences


@v1_router.post(
    "/auth/user/preferences",
    summary="Update notification preferences",
    tags=["auth"],
    dependencies=[Security(requires_user)],
)
async def update_preferences(
    user_id: Annotated[str, Security(get_user_id)],
    preferences: NotificationPreferenceDTO = Body(...),
) -> NotificationPreference:
    output = await update_user_notification_preference(user_id, preferences)
    return output


@v1_router.post(
    "/auth/user/preferences/from-email",
    summary="Apply a volume-knob choice from a Briefing footer link",
    tags=["auth"],
)
async def apply_email_preference_choice(
    choice: Annotated[str, Query()],
    token: Annotated[str, Query()],
) -> NotificationPreference:
    """Apply one footer choice, authorised by the token in the link.

    Deliberately not `Security(requires_user)`. The session is the wrong
    authority here: the settings page applies this on arrival, so a
    session-authenticated write would let any third party change a logged-in
    reader's preferences just by getting them to follow a link. The HMAC binds
    the choice to the recipient we sent it to, exactly as the unsubscribe link
    does, and works whether or not they happen to be signed in.
    """
    user_id = verify_preference_token(token, choice)
    if user_id is None:
        raise HTTPException(status_code=400, detail="Invalid or expired link")

    current = await get_user_notification_preference(user_id)
    updated = _preference_with_choice(current, choice)
    if updated is None:
        raise HTTPException(status_code=400, detail="Unknown preference choice")
    return await update_user_notification_preference(user_id, updated)


def _preference_with_choice(
    current: NotificationPreference, choice: str
) -> NotificationPreferenceDTO | None:
    """The volume knob, server-side. "alerts" and "off" both stop the digest;
    they differ in whether alerts survive."""
    mapping: dict[str, tuple[BriefingFrequency, bool]] = {
        "daily": (BriefingFrequency.DAILY, current.alerts_enabled),
        "weekly": (BriefingFrequency.WEEKLY, current.alerts_enabled),
        "monthly": (BriefingFrequency.MONTHLY, current.alerts_enabled),
        "alerts": (BriefingFrequency.OFF, True),
        "off": (BriefingFrequency.OFF, False),
    }
    if choice not in mapping:
        return None
    frequency, alerts = mapping[choice]
    return NotificationPreferenceDTO(
        email=current.email,
        briefing_frequency=frequency,
        alerts_enabled=alerts,
        store_verdicts_enabled=current.store_verdicts_enabled,
        daily_limit=current.daily_limit,
    )


########################################################
##################### Onboarding #######################
########################################################


@v1_router.get(
    "/onboarding",
    summary="Onboarding state",
    tags=["onboarding"],
    dependencies=[Security(requires_user)],
    response_model=UserOnboarding,
)
async def get_onboarding(user_id: Annotated[str, Security(get_user_id)]):
    return await get_user_onboarding(user_id)


@v1_router.patch(
    "/onboarding",
    summary="Update onboarding state",
    tags=["onboarding"],
    dependencies=[Security(requires_user)],
    response_model=UserOnboarding,
)
async def update_onboarding(
    user_id: Annotated[str, Security(get_user_id)], data: UserOnboardingUpdate
):
    return await update_user_onboarding(user_id, data)


@v1_router.post(
    "/onboarding/step",
    summary="Complete onboarding step",
    tags=["onboarding"],
    dependencies=[Security(requires_user)],
)
async def onboarding_complete_step(
    user_id: Annotated[str, Security(get_user_id)], step: FrontendOnboardingStep
):
    if step not in get_args(FrontendOnboardingStep):
        raise HTTPException(status_code=400, detail="Invalid onboarding step")
    return await complete_onboarding_step(user_id, step)


@v1_router.get(
    "/onboarding/agents",
    summary="Recommended onboarding agents",
    tags=["onboarding"],
    dependencies=[Security(requires_user)],
)
async def get_onboarding_agents(
    user_id: Annotated[str, Security(get_user_id)],
) -> list[StoreAgentDetails]:
    return await get_recommended_agents(user_id)


class OnboardingProfileRequest(pydantic.BaseModel):
    """Request body for onboarding profile submission."""

    user_name: str = pydantic.Field(min_length=1, max_length=100)
    user_role: str = pydantic.Field(min_length=1, max_length=100)
    pain_points: list[str] = pydantic.Field(default_factory=list, max_length=20)


class OnboardingStatusResponse(pydantic.BaseModel):
    """Response for onboarding completion check."""

    is_completed: bool


@v1_router.get(
    "/onboarding/completed",
    summary="Check if onboarding is completed",
    tags=["onboarding", "public"],
    response_model=OnboardingStatusResponse,
    dependencies=[Security(requires_user)],
)
async def is_onboarding_completed(
    user_id: Annotated[str, Security(get_user_id)],
) -> OnboardingStatusResponse:
    user_onboarding = await get_user_onboarding(user_id)
    return OnboardingStatusResponse(
        is_completed=OnboardingStep.ONBOARDING_COMPLETE
        in user_onboarding.completedSteps,
    )


@v1_router.post(
    "/onboarding/reset",
    summary="Reset onboarding progress",
    tags=["onboarding"],
    dependencies=[Security(requires_user)],
    response_model=UserOnboarding,
)
async def reset_onboarding(user_id: Annotated[str, Security(get_user_id)]):
    return await reset_user_onboarding(user_id)


@v1_router.post(
    "/onboarding/profile",
    summary="Submit onboarding profile",
    tags=["onboarding"],
    dependencies=[Security(requires_user)],
)
async def submit_onboarding_profile(
    data: OnboardingProfileRequest,
    user_id: Annotated[str, Security(get_user_id)],
):
    formatted = format_onboarding_for_extraction(
        user_name=data.user_name,
        user_role=data.user_role,
        pain_points=data.pain_points,
    )

    try:
        understanding_input = await extract_business_understanding(formatted)
    except Exception:
        understanding_input = BusinessUnderstandingInput.model_construct()

    # Ensure the direct fields are set even if LLM missed them
    understanding_input.user_name = data.user_name
    understanding_input.user_role = data.user_role
    if not understanding_input.pain_points:
        understanding_input.pain_points = data.pain_points

    await upsert_business_understanding(user_id, understanding_input)

    return {"status": "ok"}


########################################################
##################### Blocks ###########################
########################################################


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


@v1_router.get(
    path="/blocks",
    summary="List available blocks",
    tags=["blocks"],
    dependencies=[Security(requires_user)],
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


@v1_router.post(
    path="/blocks/{block_id}/execute",
    summary="Execute graph block",
    tags=["blocks"],
    dependencies=[Security(requires_user), Depends(enforce_payment_paywall)],
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


@v1_router.post(
    path="/files/upload",
    summary="Upload file to cloud storage",
    tags=["files"],
    dependencies=[Security(requires_user)],
)
async def upload_file(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    file: UploadFile = File(...),
    expiration_hours: int = 24,
) -> UploadFileResponse:
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
        expiration_hours=expiration_hours,
        user_id=user_id,
    )

    return UploadFileResponse(
        file_uri=storage_path,
        file_name=file_name,
        size=content_size,
        content_type=content_type,
        expires_in_hours=expiration_hours,
    )
