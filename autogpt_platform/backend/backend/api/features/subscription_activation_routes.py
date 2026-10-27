"""Explicit Pro/Max trial conversion; pro-activation is the historical URL."""

import logging
from typing import Annotated

import stripe
from autogpt_libs.auth import get_user_id, requires_user
from fastapi import APIRouter, Depends, HTTPException, Security

from backend.api.features.billing.credits_rate_limit import (
    enforce_subscription_status_rate_limit,
)
from backend.data.subscription_activation_checkout import (
    confirm_activation,
    current_activation,
    get_activation,
    preview_activation,
)
from backend.data.subscription_activation_models import (
    ActivationConfirmRequest,
    ActivationNotFound,
    ActivationPreviewRequest,
    ActivationResponse,
    ActivationUnavailable,
)
from backend.data.subscription_checkout import SubscriptionCheckoutUnavailable

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/credits/pro-activation",
    tags=["subscriptions"],
    dependencies=[
        Security(requires_user),
        Depends(enforce_subscription_status_rate_limit),
    ],
    responses={
        404: {"description": "No owned activation"},
        409: {"description": "Review fresh terms or retry a concurrent confirmation"},
        502: {"description": "Preview temporarily unavailable; no payment started"},
    },
)
CurrentUser = Annotated[str, Security(get_user_id)]


@router.post("/preview")
async def preview_pro_activation(
    body: ActivationPreviewRequest,
    user_id: CurrentUser,
) -> ActivationResponse:
    try:
        return await preview_activation(user_id, body.return_to, body.plan)
    except (ActivationUnavailable, SubscriptionCheckoutUnavailable) as exc:
        raise HTTPException(409, str(exc)) from exc
    except stripe.StripeError as exc:
        raise HTTPException(502, "Unable to preview the charge. Please retry.") from exc


@router.post("/{attempt_id}/confirm")
async def confirm_pro_activation(
    attempt_id: str,
    body: ActivationConfirmRequest,
    user_id: CurrentUser,
) -> ActivationResponse:
    try:
        return await confirm_activation(user_id, attempt_id, body)
    except ActivationNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ActivationUnavailable, SubscriptionCheckoutUnavailable) as exc:
        raise HTTPException(409, str(exc)) from exc
    except stripe.StripeError as exc:
        raise HTTPException(
            502, "Unable to validate terms. No payment was started."
        ) from exc
    except Exception:
        logger.exception("Confirmation outcome needs recovery for %s", attempt_id)
        return ActivationResponse(
            id=attempt_id, status="processing", retry_after_seconds=3
        )


@router.get("/current")
async def get_current_pro_activation(user_id: CurrentUser) -> ActivationResponse:
    try:
        return await current_activation(user_id)
    except ActivationNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ActivationUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception:
        logger.exception("Current activation state is temporarily unavailable")
        return ActivationResponse(status="processing", retry_after_seconds=3)


@router.get("/{attempt_id}")
async def get_pro_activation(
    attempt_id: str, user_id: CurrentUser
) -> ActivationResponse:
    try:
        return await get_activation(user_id, attempt_id)
    except ActivationNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ActivationUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception:
        logger.exception("Activation state needs recovery for %s", attempt_id)
        return ActivationResponse(
            id=attempt_id, status="processing", retry_after_seconds=3
        )
