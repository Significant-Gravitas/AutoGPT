import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated, Literal, cast
from urllib.parse import urlparse

import stripe
from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
    Security,
)
from prisma.enums import SubscriptionTier
from pydantic import BaseModel, Field
from typing_extensions import Optional

from backend.api.features.credits_rate_limit import (
    enforce_subscription_status_rate_limit,
)
from backend.copilot.rate_limit import get_tier_multipliers
from backend.data.credit import (
    PendingChangeUnknown,
    UserCredit,
    cancel_stripe_subscription,
    create_subscription_checkout,
    get_active_subscription_period_end,
    get_credit_model,
    get_pending_subscription_change,
    get_proration_credit_cents,
    get_subscription_price_id,
    get_user_billing_cycle,
    handle_subscription_payment_failure,
    handle_subscription_payment_success,
    modify_stripe_subscription_for_tier,
    release_pending_subscription_schedule,
    set_subscription_tier,
    sync_subscription_from_stripe,
    sync_subscription_schedule_from_stripe,
    sync_tier_from_checkout_session,
)
from backend.data.notifications import PassWorkEvent, PassWorkKind
from backend.data.redis_client import get_redis_async
from backend.data.stripe_client import stripe_call
from backend.data.subscription_trial_billing import (
    TRIAL_BILLING_EVENTS,
    sync_trials_for_billing_event,
)
from backend.data.user import get_user_by_id
from backend.notifications import lifecycle
from backend.notifications.queue import queue_pass_work
from backend.notifications.trial import notify_trial, on_trial_invoice
from backend.util.cache import cached
from backend.util.feature_flag import Flag, evaluate_feature_flag
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# No router-level auth: /credits/stripe_webhook is authenticated by Stripe's
# signature, not by a user, so each route keeps its own dependency.
router = APIRouter()


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


@router.get(
    path="/credits/subscription",
    summary="Get subscription tier, current cost, and all tier costs",
    operation_id="getSubscriptionStatus",
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


@router.post(
    path="/credits/subscription",
    summary="Update subscription tier or start a Stripe Checkout session",
    operation_id="updateSubscriptionTier",
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


@router.post(path="/credits/stripe_webhook", summary="Handle Stripe webhooks")
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


@router.get(
    path="/credits/manage",
    summary="Manage payment methods",
    dependencies=[Security(requires_user)],
)
async def manage_payment_method(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> dict[str, str]:
    credit_model = await get_credit_model(user_id, ctx.org_id)
    return {"url": await credit_model.create_billing_portal_session(user_id)}
