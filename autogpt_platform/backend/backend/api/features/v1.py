import asyncio
import base64
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Sequence, cast, get_args
from urllib.parse import urlparse

import pydantic
import stripe
from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    Security,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from prisma.enums import BriefingFrequency, SubscriptionTier
from pydantic import BaseModel, Field
from starlette.status import HTTP_402_PAYMENT_REQUIRED
from typing_extensions import Optional, TypedDict

from backend.api.features.credits_rate_limit import (
    enforce_subscription_status_rate_limit,
)
from backend.api.model import (
    CreateGraph,
    GraphExecutionSource,
    RequestTopUp,
    SetGraphActiveVersion,
    TimezoneResponse,
    UpdateTimezoneRequest,
    UploadFileResponse,
)
from backend.blocks import get_block, get_blocks
from backend.copilot.rate_limit import enforce_payment_paywall, get_tier_multipliers
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data.block import BlockInput, CompletedBlockOutput
from backend.data.credit import (
    AutoTopUpConfig,
    InvoiceListItem,
    PendingChangeUnknown,
    RefundRequest,
    TransactionHistory,
    UserCredit,
    cancel_stripe_subscription,
    create_subscription_checkout,
    get_active_subscription_period_end,
    get_auto_top_up,
    get_credit_model,
    get_pending_subscription_change,
    get_proration_credit_cents,
    get_subscription_price_id,
    get_user_billing_cycle,
    handle_subscription_payment_failure,
    handle_subscription_payment_success,
    modify_stripe_subscription_for_tier,
    release_pending_subscription_schedule,
    set_auto_top_up,
    set_subscription_tier,
    sync_subscription_from_stripe,
    sync_subscription_schedule_from_stripe,
    sync_tier_from_checkout_session,
)
from backend.data.execution import ExecutionContext
from backend.data.graph import GraphSettings
from backend.data.model import CredentialsMetaInput, UserOnboarding
from backend.data.notifications import (
    NotificationPreference,
    NotificationPreferenceDTO,
    PassWorkEvent,
    PassWorkKind,
)
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
from backend.data.redis_client import get_redis_async
from backend.data.stripe_client import stripe_call
from backend.data.subscription_trial_billing import (
    TRIAL_BILLING_EVENTS,
    sync_trials_for_billing_event,
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
from backend.integrations.webhooks.graph_lifecycle_hooks import (
    before_graph_activate,
    on_graph_deactivate,
)
from backend.monitoring.instrumentation import (
    record_block_execution,
    record_graph_operation,
)
from backend.notifications import lifecycle
from backend.notifications.queue import queue_pass_work
from backend.notifications.trial import notify_trial, on_trial_invoice
from backend.util.cache import cached
from backend.util.cloud_storage import get_cloud_storage_handler
from backend.util.exceptions import GraphValidationError, InsufficientBalanceError
from backend.util.feature_flag import Flag, evaluate_feature_flag
from backend.util.json import dumps
from backend.util.settings import Settings
from backend.util.timezone_utils import get_user_timezone_or_utc
from backend.util.virus_scanner import scan_content_safe

from .library import db as library_db
from .library import model as library_model
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


########################################################
##################### Credits ##########################
########################################################


@v1_router.get(
    path="/credits",
    tags=["credits"],
    summary="Get user credits",
    dependencies=[Security(requires_user)],
)
async def get_user_credits(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> dict[str, int]:
    credit_model = await get_credit_model(user_id, ctx.org_id)
    return {"credits": await credit_model.get_credits(user_id)}


@v1_router.post(
    path="/credits",
    summary="Request credit top up",
    tags=["credits"],
    dependencies=[Security(requires_user)],
)
async def request_top_up(
    request: RequestTopUp,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    x_datafast_visitor_id: Annotated[
        str | None, Header(include_in_schema=False)
    ] = None,
    x_datafast_session_id: Annotated[
        str | None, Header(include_in_schema=False)
    ] = None,
):
    credit_model = await get_credit_model(user_id, ctx.org_id)
    checkout_url = await credit_model.top_up_intent(
        user_id,
        request.credit_amount,
        datafast_visitor_id=x_datafast_visitor_id,
        datafast_session_id=x_datafast_session_id,
    )
    return {"checkout_url": checkout_url}


@v1_router.post(
    path="/credits/{transaction_key}/refund",
    summary="Refund credit transaction",
    tags=["credits"],
    dependencies=[Security(requires_user)],
)
async def refund_top_up(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    transaction_key: str,
    metadata: dict[str, str],
) -> int:
    credit_model = await get_credit_model(user_id, ctx.org_id)
    return await credit_model.top_up_refund(user_id, transaction_key, metadata)


@v1_router.patch(
    path="/credits",
    summary="Fulfill checkout session",
    tags=["credits"],
    dependencies=[Security(requires_user)],
)
async def fulfill_checkout(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
):
    credit_model = await get_credit_model(user_id, ctx.org_id)
    await credit_model.fulfill_checkout(user_id=user_id)
    return Response(status_code=200)


@v1_router.post(
    path="/credits/auto-top-up",
    summary="Configure auto top up",
    tags=["credits"],
    dependencies=[Security(requires_user)],
)
async def configure_user_auto_top_up(
    request: AutoTopUpConfig,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> str:
    """Configure auto top-up settings and perform an immediate top-up if needed.

    Raises HTTPException(422) if the request parameters are invalid or if
    the credit top-up fails.
    """
    if request.threshold < 0:
        raise HTTPException(status_code=422, detail="Threshold must be greater than 0")
    if request.amount < 500 and request.amount != 0:
        raise HTTPException(
            status_code=422, detail="Amount must be greater than or equal to 500"
        )
    if request.amount != 0 and request.amount < request.threshold:
        raise HTTPException(
            status_code=422, detail="Amount must be greater than or equal to threshold"
        )

    credit_model = await get_credit_model(user_id, ctx.org_id)
    current_balance = await credit_model.get_credits(user_id)

    try:
        if current_balance < request.threshold:
            await credit_model.top_up_credits(user_id, request.amount)
        else:
            await credit_model.top_up_credits(user_id, 0)
    except NotImplementedError as e:
        raise HTTPException(
            status_code=501, detail="Auto top-up is not available in this context"
        ) from e
    except ValueError as e:
        known_messages = (
            "must not be negative",
            "already exists for user",
            "No payment method found",
        )
        if any(msg in str(e) for msg in known_messages):
            raise HTTPException(status_code=422, detail=str(e))
        raise

    try:
        await set_auto_top_up(
            user_id, AutoTopUpConfig(threshold=request.threshold, amount=request.amount)
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return "Auto top-up settings updated"


@v1_router.get(
    path="/credits/auto-top-up",
    summary="Get auto top up",
    tags=["credits"],
    dependencies=[Security(requires_user)],
)
async def get_user_auto_top_up(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> AutoTopUpConfig:
    return await get_auto_top_up(user_id)


class SubscriptionTierRequest(BaseModel):
    tier: Literal["NO_TIER", "BASIC", "PRO", "MAX", "BUSINESS"]
    success_url: str = ""
    cancel_url: str = ""
    billing_cycle: Literal["monthly", "yearly"] = "monthly"


class SubscriptionStatusResponse(BaseModel):
    tier: Literal["NO_TIER", "TRIAL", "BASIC", "PRO", "MAX", "BUSINESS", "ENTERPRISE"]
    monthly_cost: int  # amount in cents (Stripe convention)
    tier_costs: dict[str, int]  # tier name -> monthly amount in cents
    tier_costs_yearly: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Tier → yearly amount in cents. Populated only for tiers with a"
            " yearly Stripe price configured in LaunchDarkly. Empty for"
            " monthly-only configurations."
        ),
    )
    billing_cycle: Literal["monthly", "yearly"] = Field(
        default="monthly",
        description=(
            "Billing cycle of the user's active Stripe subscription. Defaults"
            " to ``monthly`` for users without an active sub. ``monthly_cost``"
            " above reflects this cycle's actual price (so a yearly subscriber"
            " sees their yearly amount, not the monthly equivalent)."
        ),
    )
    tier_multipliers: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Tier → rate-limit multiplier. Covers the same tiers listed in"
            " ``tier_costs`` so the frontend can render rate-limit badges"
            " relative to the lowest visible tier without knowing backend"
            " defaults."
        ),
    )
    proration_credit_cents: int  # unused portion of current sub to convert on upgrade
    has_active_stripe_subscription: bool = Field(
        default=False,
        description=(
            "True when the user has an active/trialing Stripe subscription. The"
            " frontend uses this to branch upgrade UX: modify-in-place + saved-card"
            " auto-charge when True, redirect to Stripe Checkout when False."
        ),
    )
    current_period_end: Optional[int] = Field(
        default=None,
        description=(
            "Unix timestamp of the active subscription's current_period_end. Used"
            " to show the date Stripe will issue the next invoice (with prorated"
            " upgrade charges, if any). None when no active sub."
        ),
    )
    pending_tier: Optional[Literal["NO_TIER", "BASIC", "PRO", "MAX", "BUSINESS"]] = None
    pending_tier_effective_at: Optional[datetime] = None
    pending_billing_cycle: Optional[Literal["monthly", "yearly"]] = Field(
        default=None,
        description=(
            "Billing cycle of the queued change, when resolvable. Set alongside"
            " ``pending_tier`` for tier downgrades and same-tier cycle"
            " switches (yearly→monthly). The frontend uses this to differentiate"
            " a cycle-only schedule (``pending_tier == current tier``) from a"
            " real tier downgrade so the UI copy can describe the actual"
            " change. ``None`` for cancellations and unconfigured legacy prices."
        ),
    )
    url: str = Field(
        default="",
        description=(
            "Populated only when POST /credits/subscription starts a Stripe Checkout"
            " Session (BASIC → paid upgrade). Empty string in all other branches —"
            " the client redirects to this URL when non-empty."
        ),
    )


def _validate_checkout_redirect_url(url: str) -> bool:
    """Return True if `url` matches the configured frontend origin.

    Prevents open-redirect: attackers must not be able to supply arbitrary
    success_url/cancel_url that Stripe will redirect users to after checkout.

    Pre-parse rejection rules (applied before urlparse):
    - Backslashes (``\\``) are normalised differently across parsers/browsers.
    - Control characters (U+0000–U+001F) are not valid in URLs and may confuse
      some URL-parsing implementations.
    """
    # Reject characters that can confuse URL parsers before any parsing.
    if "\\" in url:
        return False
    if any(ord(c) < 0x20 for c in url):
        return False

    allowed = settings.config.frontend_base_url or settings.config.platform_base_url
    if not allowed:
        # No configured origin — refuse to validate rather than allow arbitrary URLs.
        return False
    try:
        parsed = urlparse(url)
        allowed_parsed = urlparse(allowed)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    # Reject ``user:pass@host`` authority tricks — ``@`` in the netloc component
    # can trick browsers into connecting to a different host than displayed.
    # ``@`` in query/fragment is harmless and must be allowed.
    if "@" in parsed.netloc:
        return False
    return (
        parsed.scheme == allowed_parsed.scheme
        and parsed.netloc == allowed_parsed.netloc
    )


@cached(ttl_seconds=300, maxsize=32, cache_none=False)
async def _get_stripe_price_amount(price_id: str) -> int | None:
    """Return the unit_amount (cents) for a Stripe Price ID, cached for 5 minutes.

    Returns ``None`` on transient Stripe errors. ``cache_none=False`` opts out
    of caching the ``None`` sentinel so the next request retries Stripe instead
    of being served a stale "no price" for the rest of the TTL window. Callers
    should treat ``None`` as an unknown price and fall back to 0.

    Stripe prices rarely change; caching avoids a ~200-600 ms Stripe round-trip on
    every GET /credits/subscription page load and reduces quota consumption.
    """
    try:
        price = await stripe_call(stripe.Price.retrieve_async, price_id)
        return price.unit_amount or 0
    except stripe.StripeError:
        logger.warning(
            "Failed to retrieve Stripe price %s — returning None (not cached)",
            price_id,
        )
        return None


@v1_router.get(
    path="/credits/subscription",
    summary="Get subscription tier, current cost, and all tier costs",
    operation_id="getSubscriptionStatus",
    tags=["credits"],
    dependencies=[
        Security(requires_user),
        Depends(enforce_subscription_status_rate_limit),
    ],
    responses={429: {"description": "Rate limit exceeded"}},
)
async def get_subscription_status(
    user_id: Annotated[str, Security(get_user_id)],
) -> SubscriptionStatusResponse:
    user = await get_user_by_id(user_id)
    tier = user.subscription_tier or SubscriptionTier.NO_TIER

    # Tiers that *can* have a Stripe price configured (and therefore appear
    # in the tier picker if the LD flag exposes a price-id). NO_TIER is not
    # priceable — it's the implicit "no active subscription" state.
    priceable_tiers = [
        SubscriptionTier.BASIC,
        SubscriptionTier.PRO,
        SubscriptionTier.MAX,
        SubscriptionTier.BUSINESS,
    ]
    monthly_price_ids, yearly_price_ids = await asyncio.gather(
        asyncio.gather(
            *[get_subscription_price_id(t, "monthly") for t in priceable_tiers]
        ),
        asyncio.gather(
            *[get_subscription_price_id(t, "yearly") for t in priceable_tiers]
        ),
    )

    async def _cost(pid: str | None) -> int:
        return (await _get_stripe_price_amount(pid) or 0) if pid else 0

    monthly_costs, yearly_costs = await asyncio.gather(
        asyncio.gather(*[_cost(pid) for pid in monthly_price_ids]),
        asyncio.gather(*[_cost(pid) for pid in yearly_price_ids]),
    )

    # Row visibility: include a tier if EITHER cycle is configured. Monthly
    # cost falls back to 0 when only yearly is configured so the frontend can
    # still render the card and surface yearly via ``tier_costs_yearly``.
    tier_costs: dict[str, int] = {}
    tier_costs_yearly: dict[str, int] = {}
    for t, m_pid, y_pid, m_cost, y_cost in zip(
        priceable_tiers,
        monthly_price_ids,
        yearly_price_ids,
        monthly_costs,
        yearly_costs,
    ):
        if m_pid or y_pid:
            tier_costs[t.value] = m_cost if m_pid else 0
        if y_pid:
            tier_costs_yearly[t.value] = y_cost

    # Expose the effective rate-limit multipliers alongside prices so the
    # frontend can render "Nx rate limits" relative to the lowest visible
    # tier without hard-coding backend defaults.  Only emit entries for tiers
    # that land in ``tier_costs`` — rows hidden at the price layer must stay
    # hidden in the multiplier layer too.
    multipliers = await get_tier_multipliers()
    # get_tier_multipliers() keys by tier string value (see its docstring),
    # so the lookup must use t.value — passing the enum t silently misses
    # every tier and falls back to 1.0, ignoring LD-configured multipliers.
    tier_multipliers: dict[str, float] = {
        t.value: multipliers.get(t.value, 1.0)
        for t in priceable_tiers
        if t.value in tier_costs
    }

    user_cycle = await get_user_billing_cycle(user_id) or "monthly"
    if user_cycle == "yearly":
        current_monthly_cost = tier_costs_yearly.get(tier.value, 0)
    else:
        current_monthly_cost = tier_costs.get(tier.value, 0)
    proration_credit, current_period_end = await asyncio.gather(
        get_proration_credit_cents(user_id, current_monthly_cost),
        get_active_subscription_period_end(user_id),
    )

    try:
        pending = await get_pending_subscription_change(user_id)
    except (stripe.StripeError, PendingChangeUnknown):
        # Swallow Stripe-side failures (rate limits, transient network) AND
        # PendingChangeUnknown (LaunchDarkly price-id lookup failed). Both
        # propagate past the cache so the next request retries fresh instead
        # of serving a stale None for the TTL window. Let real bugs (KeyError,
        # AttributeError, etc.) propagate so they surface in Sentry.
        logger.exception(
            "get_subscription_status: failed to resolve pending change for user %s",
            user_id,
        )
        pending = None

    response = SubscriptionStatusResponse(
        tier=tier.value,
        monthly_cost=current_monthly_cost,
        tier_costs=tier_costs,
        tier_costs_yearly=tier_costs_yearly,
        billing_cycle=user_cycle,
        tier_multipliers=tier_multipliers,
        proration_credit_cents=proration_credit,
        has_active_stripe_subscription=current_period_end is not None,
        current_period_end=current_period_end,
    )
    if pending is not None:
        pending_tier_enum, pending_effective_at, pending_cycle = pending
        if pending_tier_enum in (
            SubscriptionTier.NO_TIER,
            SubscriptionTier.BASIC,
            SubscriptionTier.PRO,
            SubscriptionTier.MAX,
            SubscriptionTier.BUSINESS,
        ):
            response.pending_tier = pending_tier_enum.value
            response.pending_tier_effective_at = pending_effective_at
            response.pending_billing_cycle = pending_cycle
    return response


@v1_router.post(
    path="/credits/subscription",
    summary="Update subscription tier or start a Stripe Checkout session",
    operation_id="updateSubscriptionTier",
    tags=["credits"],
    dependencies=[Security(requires_user)],
)
async def update_subscription_tier(
    request: SubscriptionTierRequest,
    user_id: Annotated[str, Security(get_user_id)],
    x_datafast_visitor_id: Annotated[
        str | None, Header(include_in_schema=False)
    ] = None,
    x_datafast_session_id: Annotated[
        str | None, Header(include_in_schema=False)
    ] = None,
) -> SubscriptionStatusResponse:
    # Pydantic validates tier is one of BASIC/PRO/MAX/BUSINESS via Literal type.
    tier = SubscriptionTier(request.tier)

    # ENTERPRISE tier is admin-managed — block self-service changes from ENTERPRISE users.
    user = await get_user_by_id(user_id)
    if (
        user.subscription_tier or SubscriptionTier.NO_TIER
    ) == SubscriptionTier.ENTERPRISE:
        raise HTTPException(
            status_code=403,
            detail="ENTERPRISE subscription changes must be managed by an administrator",
        )

    # Same-tier + same-cycle request = "stay on my current tier" = cancel any
    # pending scheduled change (paid→paid downgrade or paid→BASIC cancel). This
    # replaces the old /credits/subscription/cancel-pending route. Safe when no
    # pending change exists: release_pending_subscription_schedule returns
    # False and we simply return the current status.
    #
    # Same-tier-DIFFERENT-cycle (monthly Pro → yearly Pro, or vice versa) must
    # fall through to modify_stripe_subscription_for_tier so Stripe swaps the
    # price ID for the cycle the user actually requested.
    #
    # Gate the short-circuit on an actual active/trialing Stripe subscription:
    # admin-granted tiers (DB tier set, no Stripe sub) must fall through to the
    # Checkout flow so "start paying for my current tier" is not a no-op.
    current_tier = user.subscription_tier or SubscriptionTier.NO_TIER
    if current_tier == SubscriptionTier.TRIAL and tier != SubscriptionTier.NO_TIER:
        raise HTTPException(
            409,
            "Your accepted plan starts after your trial. Manage the trial in billing.",
        )
    current_cycle = await get_user_billing_cycle(user_id) or "monthly"
    has_active_stripe_subscription = (
        await get_active_subscription_period_end(user_id) is not None
    )
    if (
        current_tier == tier
        and current_cycle == request.billing_cycle
        and has_active_stripe_subscription
    ):
        try:
            await release_pending_subscription_schedule(user_id)
        except stripe.StripeError as e:
            logger.exception(
                "Stripe error releasing pending subscription change for user %s: %s",
                user_id,
                e,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    "Unable to cancel the pending subscription change right now. "
                    "Please try again or contact support."
                ),
            )
        return await get_subscription_status(user_id)

    payment_enabled, payment_flag_authoritative = await evaluate_feature_flag(
        Flag.ENABLE_PLATFORM_PAYMENT, user_id, default=False
    )

    target_price_id = await get_subscription_price_id(tier, request.billing_cycle)

    # Cancel: target NO_TIER. Schedule Stripe cancellation at period end;
    # cancel_at_period_end=True lets the webhook flip the DB tier. No active
    # sub (admin-granted or never-paid) or payment disabled → DB flip.
    # NO_TIER is never priceable, so this branch always fires for cancel
    # requests regardless of LD config.
    if tier == SubscriptionTier.NO_TIER:
        if payment_enabled:
            try:
                had_subscription = await cancel_stripe_subscription(user_id)
            except stripe.StripeError as e:
                logger.exception(
                    "Stripe error cancelling subscription for user %s: %s",
                    user_id,
                    e,
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Unable to cancel your subscription right now. "
                        "Please try again or contact support."
                    ),
                )
            if not had_subscription:
                # No Stripe subscription drove this change (admin-granted or
                # never-paid).
                await set_subscription_tier(user_id, tier)
            return await get_subscription_status(user_id)
        if not payment_flag_authoritative:
            # An unreadable flag reads False exactly like payment being off, and
            # the DB flip below would strand a still-billing Stripe subscription.
            logger.error(
                f"Refusing to cancel subscription for user {user_id}: "
                f"{Flag.ENABLE_PLATFORM_PAYMENT} could not be evaluated"
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    "Unable to cancel your subscription right now. "
                    "Please try again or contact support."
                ),
            )
        await set_subscription_tier(user_id, tier)
        return await get_subscription_status(user_id)

    if not payment_enabled:
        raise HTTPException(
            status_code=422,
            detail=f"Subscription not available for tier {tier.value}",
        )

    # Target has no LD price — not provisionable (matches the GET hiding).
    if target_price_id is None:
        raise HTTPException(
            status_code=422,
            detail=f"Subscription not available for tier {tier.value}",
        )

    # Modify in place if there's a sub; else fall through to Checkout below.
    try:
        modified = await modify_stripe_subscription_for_tier(
            user_id, tier, request.billing_cycle
        )
        if modified:
            return await get_subscription_status(user_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except stripe.CardError as e:
        # Auto-charge failed under payment_behavior=error_if_incomplete: the
        # modify was rolled back, so 402 lets the UI prompt for a new card or
        # surface SCA. SCA codes mean the card is fine but the bank wants 3DS —
        # different message so the user doesn't try a new card. Stripe emits
        # ``authentication_required`` for raw PaymentIntent confirms but
        # ``subscription_payment_intent_requires_action`` for Subscription.modify
        # under ``error_if_incomplete``; both must map to the SCA branch.
        if e.code in {
            "authentication_required",
            "subscription_payment_intent_requires_action",
        }:
            logger.warning(
                "SCA required on subscription upgrade for user %s: %s", user_id, e
            )
            raise HTTPException(
                status_code=402,
                detail=(
                    "Your bank requires extra authentication for this charge."
                    " The plan was not changed; please retry from the billing"
                    " portal so you can complete authentication, or contact"
                    " support."
                ),
            )
        logger.warning(
            "Card declined on subscription upgrade for user %s: %s", user_id, e
        )
        raise HTTPException(
            status_code=402,
            detail=(
                "Your card was declined. The plan was not changed; please"
                " update your payment method and try again."
            ),
        )
    except stripe.InvalidRequestError as e:
        # Stripe's e.param is documented as nullable, so we match by typed
        # field first and fall back to substring when param is absent.
        msg_lower = (e.user_message or str(e)).lower()
        # "No payment method" presents as InvalidRequestError (not CardError)
        # when error_if_incomplete fires with no default PM. Stripe signals
        # this with code=resource_missing/missing — sometimes with a typed
        # param, sometimes without (the raw "no attached payment source"
        # message has empty param). Map it to 402 either way.
        if e.code in {"resource_missing", "missing"} and (
            e.param
            in {
                "default_payment_method",
                "payment_method",
                "invoice_settings.default_payment_method",
            }
            or "no attached payment source" in msg_lower
            or "default payment method" in msg_lower
            or "no payment method" in msg_lower
        ):
            logger.warning(
                "No payment method on subscription upgrade for user %s: %s",
                user_id,
                e,
            )
            raise HTTPException(
                status_code=402,
                detail=(
                    "No payment method on file. The plan was not changed;"
                    " please add a payment method and try again."
                ),
            )
        # Stripe rejects schedule modify when phases mix currencies, e.g. the
        # active sub was checked out in GBP but the target tier's Price is
        # USD-only. e.param is "currency" on the schedule API but may be
        # "phases" or absent on older error shapes — substring fallback keeps
        # the 422 firing instead of dropping to the generic 502.
        if e.param == "currency" or "currency" in msg_lower:
            logger.warning(
                "Currency mismatch on tier change for user %s: %s", user_id, e
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    "Tier change unavailable for your current billing currency."
                    " Please contact support — the target tier needs to be"
                    " configured for your currency in Stripe before this"
                    " change can go through."
                ),
            )
        logger.exception(
            "Stripe error modifying subscription for user %s: %s", user_id, e
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to update your subscription right now. "
                "Please try again or contact support."
            ),
        )
    except stripe.StripeError as e:
        logger.exception(
            "Stripe error modifying subscription for user %s: %s", user_id, e
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to update your subscription right now. "
                "Please try again or contact support."
            ),
        )

    # No active Stripe subscription → create Stripe Checkout Session.
    if not request.success_url or not request.cancel_url:
        raise HTTPException(
            status_code=422,
            detail="success_url and cancel_url are required for paid tier upgrades",
        )
    # Open-redirect protection: both URLs must point to the configured frontend
    # origin, otherwise an attacker could use our Stripe integration as a
    # redirector to arbitrary phishing sites.
    #
    # Fail early with a clear 503 if the server is misconfigured (neither
    # frontend_base_url nor platform_base_url set), so operators get an
    # actionable error instead of the misleading "must match the platform
    # frontend origin" 422 that _validate_checkout_redirect_url would otherwise
    # produce when `allowed` is empty.
    if not (settings.config.frontend_base_url or settings.config.platform_base_url):
        logger.error(
            "update_subscription_tier: neither frontend_base_url nor "
            "platform_base_url is configured; cannot validate checkout redirect URLs"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Payment redirect URLs cannot be validated: "
                "frontend_base_url or platform_base_url must be set on the server."
            ),
        )
    if not _validate_checkout_redirect_url(
        request.success_url
    ) or not _validate_checkout_redirect_url(request.cancel_url):
        raise HTTPException(
            status_code=422,
            detail="success_url and cancel_url must match the platform frontend origin",
        )
    try:
        url = await create_subscription_checkout(
            user_id=user_id,
            tier=tier,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            billing_cycle=request.billing_cycle,
            datafast_visitor_id=x_datafast_visitor_id,
            datafast_session_id=x_datafast_session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except stripe.StripeError as e:
        logger.exception(
            "Stripe error creating checkout session for user %s: %s", user_id, e
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to start checkout right now. "
                "Please try again or contact support."
            ),
        )

    status = await get_subscription_status(user_id)
    status.url = url
    return status


def _stripe_event_dedup_key(event_id: str) -> str:
    return f"stripe_webhook_event:{event_id}"


async def _claim_stripe_event(event_id: str) -> bool:
    """Mark a Stripe webhook event as claimed via Redis SETNX.

    Returns ``True`` when the caller acquired the claim (first time we've seen
    ``event_id``) and should proceed with handler dispatch. Returns ``False``
    when the event was already processed in a prior delivery — Stripe retries
    the same ``event.id`` on non-2xx responses, and we don't want downstream
    handlers (some of which only carry per-resource idempotency) to fire twice.

    Pair with ``_release_stripe_event`` in a try/except around handler dispatch:
    on handler failure we DELete the key so Stripe's retry isn't no-op'd, but
    a retry that arrives *during* in-flight processing still hits the live
    claim and is deduped.

    TTL of 24h comfortably exceeds Stripe's retry window. On Redis failure we
    fall open and let processing continue — better to risk a rare duplicate
    than to drop a real event.
    """
    if not event_id:
        # Malformed event without an id — fall open so the rest of the
        # handler can decide what to do (it'll log and 200 anyway).
        return True
    try:
        redis_client = await get_redis_async()
        claimed = await redis_client.set(
            _stripe_event_dedup_key(event_id), "1", nx=True, ex=86400
        )
        return bool(claimed)
    except Exception:
        logger.warning(
            "stripe_webhook: dedup claim failed for event %s; processing anyway",
            event_id,
            exc_info=True,
        )
        return True


async def _notify_checkout_completed(session: dict) -> None:
    """Hand the completed checkout to the lifecycle emails.

    Failures here must not fail the webhook: the customer has paid, and Stripe
    retrying the whole event would re-run `fulfill_checkout`, which grants
    credits. But swallowing the failure lost the welcome permanently — Stripe
    does not retry a 200, and `customer.subscription.created` deliberately does
    not send it either.

    So the work is queued rather than done here. The consumer re-reads the
    session and subscription from Stripe and sends the welcome, with the same
    retry-with-backoff and dead-letter queue every other notification gets.
    Publishing is one small call that either succeeds or is logged; the Stripe
    API round-trip and the email now sit behind a retry instead of a warning.
    """
    if session.get("mode") != "subscription":
        return
    if not session.get("subscription"):
        return
    session_id = session.get("id")
    if not session_id:
        return
    result = await queue_pass_work(
        PassWorkKind.WELCOME.value,
        str(session_id),
        PassWorkEvent(
            kind=PassWorkKind.WELCOME,
            user_id="",
            scheduled_for=datetime.now(tz=timezone.utc),
            context={"session_id": str(session_id)},
        ).model_dump_json(),
    )
    if not result.success:
        if (session.get("metadata") or {}).get("trial_enrollment_id"):
            raise RuntimeError("Could not queue the trial welcome notice")
        logger.warning(
            "stripe_webhook: could not queue the welcome email for session %s: %s",
            session_id,
            result.message,
        )


async def _release_stripe_event(event_id: str) -> None:
    """Release a previously-claimed dedup key so Stripe's retry can rerun."""
    if not event_id:
        return
    try:
        redis_client = await get_redis_async()
        await redis_client.delete(_stripe_event_dedup_key(event_id))
    except Exception:
        logger.warning(
            "stripe_webhook: dedup release failed for event %s",
            event_id,
            exc_info=True,
        )


@v1_router.post(
    path="/credits/stripe_webhook", summary="Handle Stripe webhooks", tags=["credits"]
)
async def stripe_webhook(request: Request):
    webhook_secret = settings.secrets.stripe_webhook_secret
    if not webhook_secret:
        # Guard: an empty secret allows HMAC forgery (attacker can compute a valid
        # signature over the same empty key). Reject all webhook calls when unconfigured.
        logger.error(
            "stripe_webhook: STRIPE_WEBHOOK_SECRET is not configured — "
            "rejecting request to prevent signature bypass"
        )
        raise HTTPException(status_code=503, detail="Webhook not configured")

    # Get the raw request body
    payload = await request.body()
    # Get the signature header
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except ValueError:
        # Invalid payload
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.SignatureVerificationError:
        # Invalid signature
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Defensive payload extraction. A malformed payload (missing/non-dict
    # `data.object`, missing `id`) would otherwise raise KeyError/TypeError
    # AFTER signature verification — which Stripe interprets as a delivery
    # failure and retries forever, while spamming Sentry with no useful info.
    # Acknowledge with 200 and a warning so Stripe stops retrying.
    event_id = event.get("id", "")
    event_type = event.get("type", "")

    if event_type in TRIAL_BILLING_EVENTS:
        # This idempotent path only reconciles current Stripe state. Do not let
        # a claim left by a crashed delivery suppress card removal/restoration.
        await sync_trials_for_billing_event(event_type, event.get("data"))
        return Response(status_code=200)

    # Event-level dedup: short-circuit identical re-deliveries before any
    # handler runs. Stripe retries the same event.id on non-2xx responses, and
    # not every downstream handler is independently idempotent.
    if not await _claim_stripe_event(event_id):
        logger.info(
            "stripe_webhook: event %s (%s) already processed; skipping",
            event_id,
            event_type,
        )
        return Response(status_code=200)

    event_data = event.get("data") or {}
    data_object = event_data.get("object") if isinstance(event_data, dict) else None
    if not isinstance(data_object, dict):
        logger.warning(
            "stripe_webhook: %s missing or non-dict data.object; ignoring",
            event_type,
        )
        return Response(status_code=200)

    # Wrap handler dispatch so a downstream failure releases the dedup claim;
    # otherwise Stripe's retry would hit the live key and silently drop the
    # event. Concurrent retries that arrive *during* in-flight processing
    # still hit the live claim and are deduped.
    try:
        if event_type in (
            "checkout.session.completed",
            "checkout.session.async_payment_succeeded",
        ):
            session_id = data_object.get("id")
            if not session_id:
                logger.warning(
                    "stripe_webhook: %s missing data.object.id; ignoring", event_type
                )
                return Response(status_code=200)
            await UserCredit().fulfill_checkout(session_id=session_id)
            await sync_tier_from_checkout_session(data_object)
            # Only `checkout.session.completed` drives the welcome email.
            # `customer.subscription.created` fires at signup too; listening to
            # both would double-send.
            if event_type == "checkout.session.completed":
                await _notify_checkout_completed(data_object)

        if event_type in (
            "customer.subscription.created",
            "customer.subscription.updated",
            "customer.subscription.deleted",
        ):
            await sync_subscription_from_stripe(data_object)
            if event_type == "customer.subscription.updated":
                await lifecycle.on_subscription_updated(
                    data_object, event_data.get("previous_attributes") or {}
                )
            elif event_type == "customer.subscription.deleted":
                await lifecycle.on_subscription_deleted(data_object)

        # `subscription_schedule.updated` is deliberately omitted: our own
        # `SubscriptionSchedule.create` + `.modify` calls in
        # `_schedule_downgrade_at_period_end` would fire that event right back
        # at us and loop redundant traffic through this handler. We only care
        # about state transitions (released / completed); phase advance to
        # the new price is already covered by `customer.subscription.updated`.
        if event_type in (
            "subscription_schedule.released",
            "subscription_schedule.completed",
        ):
            await sync_subscription_schedule_from_stripe(data_object)

        if event_type == "invoice.payment_succeeded":
            await handle_subscription_payment_success(data_object)
            await on_trial_invoice(data_object, paid=True)

        if event_type == "customer.subscription.trial_will_end":
            await sync_subscription_from_stripe(data_object)
            await notify_trial(data_object, "ending")

        if event_type == "invoice.payment_failed":
            await handle_subscription_payment_failure(data_object)
            if not await on_trial_invoice(data_object, paid=False):
                await lifecycle.on_payment_failed(data_object)

        # New Stripe API (≥2025-04-01) split the per-payment events off the
        # Invoice resource. data.object is an InvoicePayment, not an Invoice,
        # so we hydrate the underlying Invoice before delegating to the
        # existing handlers. A transient ``stripe.StripeError`` here propagates
        # to the outer handler so the dedup claim is released and Stripe sees
        # a 5xx + retries — swallowing it with a 200 would silently drop the
        # event and leave the dedup key blocking the next delivery.
        if event_type in ("invoice_payment.paid", "invoice_payment.payment_failed"):
            invoice_id = data_object.get("invoice")
            if invoice_id:
                invoice = await stripe_call(stripe.Invoice.retrieve_async, invoice_id)
                invoice_payload = cast(dict, invoice)
                if event_type == "invoice_payment.paid":
                    await handle_subscription_payment_success(invoice_payload)
                    await on_trial_invoice(invoice_payload, paid=True)
                else:
                    await handle_subscription_payment_failure(invoice_payload)
                    if not await on_trial_invoice(invoice_payload, paid=False):
                        await lifecycle.on_payment_failed(invoice_payload)

        # `handle_dispute` and `deduct_credits` expect Stripe SDK typed objects
        # (Dispute/Refund). The Stripe webhook payload's `data.object` is a
        # StripeObject (a dict subclass) carrying that runtime shape, so we
        # cast to satisfy the type checker without changing runtime behaviour.
        if event_type == "charge.dispute.created":
            await UserCredit().handle_dispute(cast(stripe.Dispute, data_object))

        if event_type == "refund.created" or event_type == "charge.dispute.closed":
            await UserCredit().deduct_credits(
                cast("stripe.Refund | stripe.Dispute", data_object)
            )
    except Exception:
        # Release the dedup claim so Stripe's retry isn't no-op'd. Re-raise
        # so the webhook returns 500 and Stripe retries (the normal contract).
        await _release_stripe_event(event_id)
        raise

    return Response(status_code=200)


@v1_router.get(
    path="/credits/manage",
    tags=["credits"],
    summary="Manage payment methods",
    dependencies=[Security(requires_user)],
)
async def manage_payment_method(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> dict[str, str]:
    credit_model = await get_credit_model(user_id, ctx.org_id)
    return {"url": await credit_model.create_billing_portal_session(user_id)}


@v1_router.get(
    path="/credits/transactions",
    tags=["credits"],
    summary="Get credit history",
    dependencies=[Security(requires_user)],
)
async def get_credit_history(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    transaction_time: datetime | None = None,
    transaction_type: str | None = None,
    transaction_count_limit: int = 100,
    cursor: str | None = None,
) -> TransactionHistory:
    if transaction_count_limit < 1 or transaction_count_limit > 1000:
        raise ValueError("Transaction count limit must be between 1 and 1000")

    credit_model = await get_credit_model(user_id, ctx.org_id)
    return await credit_model.get_transaction_history(
        user_id=user_id,
        transaction_time_ceiling=transaction_time,
        transaction_count_limit=transaction_count_limit,
        transaction_type=transaction_type,
        cursor=cursor,
        viewer_organization_id=ctx.org_id,
    )


@v1_router.get(
    path="/credits/refunds",
    tags=["credits"],
    summary="Get refund requests",
    dependencies=[Security(requires_user)],
)
async def get_refund_requests(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[RefundRequest]:
    credit_model = await get_credit_model(user_id, ctx.org_id)
    return await credit_model.get_refund_requests(user_id)


@v1_router.get(
    path="/credits/invoices",
    tags=["credits"],
    summary="List Stripe invoices",
    dependencies=[Security(requires_user)],
)
async def list_invoices(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    limit: int = Query(24, ge=1, le=100),
) -> list[InvoiceListItem]:
    """Recent Stripe invoices for the current user.

    Each item includes ``hosted_invoice_url`` (Stripe-hosted view) and
    ``invoice_pdf_url`` (direct PDF download). Returns an empty list when
    the credit system is disabled or the user has no Stripe customer yet.
    """
    credit_model = await get_credit_model(user_id, ctx.org_id)
    return await credit_model.list_invoices(user_id, limit=limit)


########################################################
##################### Graphs ###########################
########################################################


class DeleteGraphResponse(TypedDict):
    version_counts: int


class UpdateGraphResponse(BaseModel):
    """Response for creating/activating a new graph version.

    Carries the new graph version plus any webhook presets that were left
    pinned to their old version because the new version's trigger block is
    incompatible and needs to be reconfigured.
    """

    graph: graph_db.GraphModel
    skipped_webhook_presets: list[library_model.SkippedWebhookPreset] = Field(
        default_factory=list
    )


class SetActiveGraphVersionResponse(BaseModel):
    """Response for activating an existing graph version.

    Carries any webhook presets that were left pinned to their old version
    because the newly activated version's trigger block is incompatible and
    needs to be reconfigured.
    """

    skipped_webhook_presets: list[library_model.SkippedWebhookPreset] = Field(
        default_factory=list
    )


@v1_router.get(
    path="/graphs",
    summary="List user graphs",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def list_graphs(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> Sequence[graph_db.GraphMeta]:
    paginated_result = await graph_db.list_graphs_paginated(
        user_id=user_id,
        page=1,
        page_size=250,
        filter_by="active",
        organization_id=ctx.org_id,
    )
    return paginated_result.graphs


@v1_router.get(
    path="/graphs/{graph_id}",
    summary="Get specific graph",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
@v1_router.get(
    path="/graphs/{graph_id}/versions/{version}",
    summary="Get graph version",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def get_graph(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    version: int | None = None,
    for_export: bool = False,
) -> graph_db.GraphModel:
    graph = await graph_db.get_graph(
        graph_id,
        version,
        user_id=user_id,
        for_export=for_export,
        include_subgraphs=True,  # needed to construct full credentials input schema
        organization_id=ctx.org_id,
    )
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
    return graph


@v1_router.get(
    path="/graphs/{graph_id}/versions",
    summary="Get all graph versions",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def get_graph_all_versions(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> Sequence[graph_db.GraphModel]:
    graphs = await graph_db.get_graph_all_versions(
        graph_id, user_id=user_id, organization_id=ctx.org_id
    )
    if not graphs:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
    return graphs


@v1_router.post(
    path="/graphs",
    summary="Create new graph",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def create_new_graph(
    create_graph: CreateGraph,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> graph_db.GraphModel:
    graph = graph_db.make_graph_model(create_graph.graph, user_id)
    graph.reassign_ids(user_id=user_id, reassign_graph_id=True)
    graph.validate_graph(for_run=False)

    # Validate node credentials (and clear stale optional ones) BEFORE
    # persisting, so a credential issue can't leave the graph/library agent
    # half-saved. before_graph_activate may also mutate input_default; those
    # edits need to be persisted, so it must run before create_graph.
    graph = await before_graph_activate(graph, user_id=user_id)

    await graph_db.create_graph(
        graph,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=ctx.team_id,
    )
    await library_db.create_library_agent(
        graph, user_id, organization_id=ctx.org_id, team_id=ctx.team_id
    )

    return graph


@v1_router.delete(
    path="/graphs/{graph_id}",
    summary="Delete graph permanently",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def delete_graph(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> DeleteGraphResponse:
    if active_version := await graph_db.get_graph(
        graph_id=graph_id, version=None, user_id=user_id
    ):
        await on_graph_deactivate(active_version, user_id=user_id)

    return {
        "version_counts": await graph_db.delete_graph(
            graph_id, user_id=user_id, organization_id=ctx.org_id
        )
    }


@v1_router.put(
    path="/graphs/{graph_id}",
    summary="Update graph version",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def update_graph(
    graph_id: str,
    graph: graph_db.Graph,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> UpdateGraphResponse:
    if graph.id and graph.id != graph_id:
        raise HTTPException(400, detail="Graph ID does not match ID in URI")

    existing_versions = await graph_db.get_graph_all_versions(graph_id, user_id=user_id)
    if not existing_versions:
        raise HTTPException(404, detail=f"Graph #{graph_id} not found")

    graph.version = max(g.version for g in existing_versions) + 1
    current_active_version = next((v for v in existing_versions if v.is_active), None)

    graph = graph_db.make_graph_model(graph, user_id)
    graph.reassign_ids(user_id=user_id, reassign_graph_id=False)
    graph.validate_graph(for_run=False)

    # If this new version is going to be active, validate node credentials
    # BEFORE persisting so a credential issue can't leave a half-saved version
    # behind. before_graph_activate may also clear stale optional credentials —
    # those edits must be persisted, hence the pre-save call.
    if graph.is_active:
        graph = await before_graph_activate(graph, user_id=user_id)

    new_graph_version = await graph_db.create_graph(
        graph,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=ctx.team_id,
    )

    skipped_webhook_presets: list[library_model.SkippedWebhookPreset] = []
    if new_graph_version.is_active:
        await library_db.update_library_agent_version_and_settings(
            user_id, new_graph_version
        )
        await graph_db.set_graph_active_version(
            graph_id=graph_id, version=new_graph_version.version, user_id=user_id
        )
        if current_active_version:
            await on_graph_deactivate(current_active_version, user_id=user_id)

        # Migrate webhook-attached presets to the new version so that
        # existing webhook URLs continue to trigger the latest agent version.
        if new_graph_version.webhook_input_node:
            migration = await library_db.migrate_webhook_presets_to_new_version(
                user_id=user_id,
                new_graph=new_graph_version,
            )
            skipped_webhook_presets = migration.skipped_presets

    new_graph_version_with_subgraphs = await graph_db.get_graph(
        graph_id,
        new_graph_version.version,
        user_id=user_id,
        include_subgraphs=True,
    )
    assert new_graph_version_with_subgraphs
    return UpdateGraphResponse(
        graph=new_graph_version_with_subgraphs,
        skipped_webhook_presets=skipped_webhook_presets,
    )


@v1_router.put(
    path="/graphs/{graph_id}/versions/active",
    summary="Set active graph version",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def set_graph_active_version(
    graph_id: str,
    request_body: SetGraphActiveVersion,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> SetActiveGraphVersionResponse:
    new_active_version = request_body.active_graph_version
    new_active_graph = await graph_db.get_graph(
        graph_id, new_active_version, user_id=user_id
    )
    if not new_active_graph:
        raise HTTPException(404, f"Graph #{graph_id} v{new_active_version} not found")

    current_active_graph = await graph_db.get_graph(
        graph_id=graph_id,
        version=None,
        user_id=user_id,
    )

    # Validate the new graph's credentials before flipping the active version.
    # Capture the returned graph: before_graph_activate may clear stale
    # optional credential references, which we want propagated to the library
    # agent's settings sync below.
    new_active_graph = await before_graph_activate(new_active_graph, user_id=user_id)
    # Ensure new version is the only active version
    await graph_db.set_graph_active_version(
        graph_id=graph_id,
        version=new_active_version,
        user_id=user_id,
    )

    # Keep the library agent up to date with the new active version
    await library_db.update_library_agent_version_and_settings(
        user_id, new_active_graph
    )

    if current_active_graph and current_active_graph.version != new_active_version:
        # Handle deactivation of the previously active version
        await on_graph_deactivate(current_active_graph, user_id=user_id)

    # Migrate webhook-attached presets to the new active version so that
    # existing webhook URLs continue to trigger the latest agent version.
    skipped_webhook_presets: list[library_model.SkippedWebhookPreset] = []
    if new_active_graph.webhook_input_node:
        migration = await library_db.migrate_webhook_presets_to_new_version(
            user_id=user_id,
            new_graph=new_active_graph,
        )
        skipped_webhook_presets = migration.skipped_presets

    return SetActiveGraphVersionResponse(
        skipped_webhook_presets=skipped_webhook_presets
    )


@v1_router.patch(
    path="/graphs/{graph_id}/settings",
    summary="Update graph settings",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def update_graph_settings(
    graph_id: str,
    settings: GraphSettings,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> GraphSettings:
    """Update graph settings for the user's library agent."""
    library_agent = await library_db.get_library_agent_by_graph_id(
        graph_id=graph_id, user_id=user_id
    )
    if not library_agent:
        raise HTTPException(404, f"Graph #{graph_id} not found in user's library")

    updated_agent = await library_db.update_library_agent(
        library_agent_id=library_agent.id,
        user_id=user_id,
        settings=settings,
    )

    return GraphSettings.model_validate(updated_agent.settings)


@v1_router.post(
    path="/graphs/{graph_id}/execute/{graph_version}",
    summary="Execute graph agent",
    tags=["graphs"],
    dependencies=[Security(requires_user), Depends(enforce_payment_paywall)],
    # The route dep enforces fail-closed (503-on-blip) so a transient
    # Supabase outage surfaces as a retryable error, not a free run
    # for a paywalled user. The deep gate inside ``add_graph_execution``
    # still covers scheduled / webhook / copilot-internal runs that
    # don't pass through this route — those callers prefer fail-open
    # so background work doesn't abandon valid jobs during a blip.
    responses={
        402: {
            "description": "Payment required: NO_TIER paywall, or insufficient credit balance"
        },
        503: {"description": "Subscription state temporarily unavailable"},
    },
)
async def execute_graph(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    inputs: Annotated[dict[str, Any], Body(..., embed=True, default_factory=dict)],
    credentials_inputs: Annotated[
        dict[str, CredentialsMetaInput], Body(..., embed=True, default_factory=dict)
    ],
    source: Annotated[GraphExecutionSource | None, Body(embed=True)] = None,
    graph_version: Optional[int] = None,
    preset_id: Optional[str] = None,
    dry_run: Annotated[bool, Body(embed=True)] = False,
) -> execution_db.GraphExecutionMeta:
    if not dry_run:
        credit_model = await get_credit_model(user_id, ctx.org_id)
        current_balance = await credit_model.get_credits(user_id)
        if current_balance <= 0:
            raise HTTPException(
                status_code=402,
                detail="Insufficient balance to execute the agent. Please top up your account.",
            )

    try:
        result = await execution_utils.add_graph_execution(
            graph_id=graph_id,
            user_id=user_id,
            inputs=inputs,
            preset_id=preset_id,
            graph_version=graph_version,
            graph_credentials_inputs=credentials_inputs,
            dry_run=dry_run,
            organization_id=ctx.org_id,
            team_id=ctx.team_id,
        )
        record_graph_operation(operation="execute", status="success")
        if source == "library":
            await complete_onboarding_step(user_id, OnboardingStep.LIBRARY_RUN_AGENT)
        elif source == "builder":
            await complete_onboarding_step(user_id, OnboardingStep.BUILDER_RUN_AGENT)
        return result
    except GraphValidationError as e:
        record_graph_operation(operation="execute", status="validation_error")
        # Return structured validation errors that the frontend can parse
        raise HTTPException(
            status_code=400,
            detail={
                "type": "validation_error",
                "message": e.message,
                # TODO: only return node-specific errors if user has access to graph
                "node_errors": e.node_errors,
            },
        )
    except Exception:
        record_graph_operation(operation="execute", status="error")
        raise
