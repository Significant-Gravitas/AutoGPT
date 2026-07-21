"""Periodic Stripe → DB subscription-tier reconciliation sweep.

Webhooks give low-latency tier updates; this sweep is the safety net that makes
Stripe the eventual source of truth. It pages through every active/trialing
Stripe subscription once, builds an authoritative ``{customer_id -> tier}`` map,
then for every reconcilable user (has a Stripe customer, not ENTERPRISE) sets the
tier from the map (NO_TIER when the customer is absent). Manual grants (no Stripe
customer) and ENTERPRISE rows are never touched.

Cost: one Stripe ``Subscription.list`` pass (not one call per user). The sweep
skips users whose tier already matches the map (``target_tier == current_tier``
in ``_reconcile_one``), so a steady-state run does no DB writes.
"""

import logging

import stripe
from fastapi.concurrency import run_in_threadpool
from prisma.enums import SubscriptionTier
from prisma.models import User
from pydantic import BaseModel

from backend.data.credit import (
    alert_tier_reconciliation_discrepancy,
    build_price_to_tier_map,
    log_tier_reconciliation_discrepancy,
    set_subscription_tier,
)

logger = logging.getLogger(__name__)

# Bound the Stripe pagination so a runaway/unexpected dataset can't loop forever.
# At 100 subs/page this caps a single sweep at 500k subscriptions; if it's ever
# hit the sweep logs and stops rather than silently truncating without notice.
_MAX_SUBSCRIPTION_PAGES = 5000
_PAGE_SIZE = 100


class TierDiscrepancy(BaseModel):
    """One Stripe<->DB tier mismatch the sweep had to correct."""

    user_id: str
    stripe_customer_id: str | None
    previous_tier: SubscriptionTier
    new_tier: SubscriptionTier
    direction: str


class ReconciliationSummary(BaseModel):
    """Counts produced by one reconciliation sweep."""

    stripe_active_subscriptions: int = 0
    candidate_users: int = 0
    upgrades: int = 0
    downgrades: int = 0
    unchanged: int = 0
    errors: int = 0
    skipped_incomplete: int = 0
    pagination_capped: bool = False
    discrepancies: list[TierDiscrepancy] = []


async def reconcile_all_stripe_tiers() -> ReconciliationSummary:
    """Reconcile every reconcilable user against live Stripe subscriptions."""
    customer_to_tier = await _build_customer_tier_map()
    summary = ReconciliationSummary(
        stripe_active_subscriptions=len(customer_to_tier.tiers),
        pagination_capped=customer_to_tier.capped,
    )
    # Match `_is_stripe_reconcilable`: every user with a Stripe customer that is
    # not on ENTERPRISE. This covers paid payers as well as paid-but-NO_TIER rows
    # left by a missed upgrade webhook. Manual grants (no customer) and ENTERPRISE
    # stay excluded.
    candidates = await User.prisma().find_many(
        where={
            "stripeCustomerId": {"not": None},
            "subscriptionTier": {"not": SubscriptionTier.ENTERPRISE},
        },
    )
    summary.candidate_users = len(candidates)
    map_complete = not customer_to_tier.capped
    for user in candidates:
        await _reconcile_one(user, customer_to_tier.tiers, summary, map_complete)
    logger.info(
        "reconcile_all_stripe_tiers: active_subs=%d candidates=%d upgrades=%d"
        " downgrades=%d unchanged=%d skipped_incomplete=%d errors=%d capped=%s",
        summary.stripe_active_subscriptions,
        summary.candidate_users,
        summary.upgrades,
        summary.downgrades,
        summary.unchanged,
        summary.skipped_incomplete,
        summary.errors,
        summary.pagination_capped,
    )
    if summary.discrepancies:
        await _alert_sweep_discrepancies(summary)
    return summary


async def _alert_sweep_discrepancies(summary: ReconciliationSummary) -> None:
    """Post a Discord (PLATFORM) system alert summarizing the sweep's corrections.

    A non-empty sweep means Stripe webhooks were dropped or missed — a payments-
    integrity signal, not a routine correction. ONE aggregate ops notification:
    a count header (how many reconciled, up vs down) followed by the FULL
    affected-user list — each line ``user_id  from_tier → to_tier (direction)``.
    Sent as a single message; ``SendDiscordMessageBlock`` splits anything over
    Discord's 2000-char limit on its own. Discord is the alert surface; this path
    logs at WARNING and never raises to Sentry.
    """
    n = len(summary.discrepancies)
    logger.warning(
        "Stripe tier reconciliation sweep reconciled %d account(s) "
        "(%d upgraded, %d downgraded); webhooks likely dropping events — "
        "steady state should be ZERO.",
        n,
        summary.upgrades,
        summary.downgrades,
    )
    user_lines = "\n".join(
        f"- `{d.user_id}`  {d.previous_tier.value} → {d.new_tier.value} ({d.direction})"
        for d in summary.discrepancies
    )
    await alert_tier_reconciliation_discrepancy(
        f"🔁 **Stripe tier reconciliation: reconciled {n} account(s)** — "
        f"{summary.upgrades} upgraded, {summary.downgrades} downgraded.\n"
        f"Each means a Stripe webhook was likely missed — investigate the webhook "
        f"pipeline; steady state should be ZERO.\nAffected users:\n{user_lines}"
    )


async def _reconcile_one(
    user: User,
    customer_to_tier: dict[str, SubscriptionTier],
    summary: ReconciliationSummary,
    map_complete: bool,
) -> None:
    """Set one user's tier from the Stripe map, updating the summary counts."""
    current_tier = SubscriptionTier(user.subscriptionTier or SubscriptionTier.NO_TIER)
    target_tier = SubscriptionTier.NO_TIER
    if user.stripeCustomerId:
        target_tier = customer_to_tier.get(
            user.stripeCustomerId, SubscriptionTier.NO_TIER
        )
    if target_tier == current_tier:
        summary.unchanged += 1
        return
    # When the Stripe snapshot is incomplete (a failed list page or the
    # pagination cap), absence from the map is unreliable — the user may sit on
    # a page we never fetched. Never revoke a tier off a partial snapshot;
    # upgrades (the customer IS in the map) stay safe.
    in_map = bool(user.stripeCustomerId) and user.stripeCustomerId in customer_to_tier
    if not map_complete and not in_map:
        summary.skipped_incomplete += 1
        return
    try:
        await set_subscription_tier(user.id, target_tier)
    except Exception:
        summary.errors += 1
        logger.exception(
            "reconcile_all_stripe_tiers: failed to set tier for user %s",
            user.id[:8],
        )
        return
    direction = log_tier_reconciliation_discrepancy(
        user_id=user.id,
        stripe_customer_id=user.stripeCustomerId,
        previous_tier=current_tier,
        new_tier=target_tier,
        via="sweep",
    )
    summary.discrepancies.append(
        TierDiscrepancy(
            user_id=user.id,
            stripe_customer_id=user.stripeCustomerId,
            previous_tier=current_tier,
            new_tier=target_tier,
            direction=direction,
        )
    )
    # Reuse the rank-based direction (not a NO_TIER check) so a paid->paid
    # downgrade (e.g. MAX->PRO) is counted as a downgrade, not an upgrade.
    if direction == "downgrade":
        summary.downgrades += 1
    else:
        summary.upgrades += 1


class _CustomerTierMap(BaseModel):
    tiers: dict[str, SubscriptionTier]
    capped: bool


async def _build_customer_tier_map() -> _CustomerTierMap:
    """Page through all active+trialing Stripe subs into a customer→tier map.

    A customer with multiple active subs resolves to their highest tier so a
    leftover lower-tier sub never downgrades a user mid-upgrade.
    """
    price_to_tier = await build_price_to_tier_map()
    tiers: dict[str, SubscriptionTier] = {}
    capped = False
    for status in ("active", "trialing"):
        page_capped = await _collect_status_page(status, price_to_tier, tiers)
        capped = capped or page_capped
    return _CustomerTierMap(tiers=tiers, capped=capped)


async def _collect_status_page(
    status: str,
    price_to_tier: dict[str, SubscriptionTier],
    tiers: dict[str, SubscriptionTier],
) -> bool:
    """Accumulate one Stripe status's subscriptions into ``tiers``. Returns
    True if pagination was capped before exhausting the dataset."""
    starting_after: str | None = None
    for _ in range(_MAX_SUBSCRIPTION_PAGES):
        list_kwargs: dict = {"status": status, "limit": _PAGE_SIZE}
        if starting_after:
            list_kwargs["starting_after"] = starting_after
        try:
            subs = await run_in_threadpool(stripe.Subscription.list, **list_kwargs)
        except stripe.StripeError:
            # A Stripe outage/rate-limit must not abort the whole sweep. Stop
            # this status and flag the run incomplete so we don't downgrade users
            # off a partial map (treated like the pagination cap below).
            logger.exception(
                "reconcile_all_stripe_tiers: Stripe list failed for status=%s;"
                " reconciliation is incomplete this run",
                status,
            )
            return True
        for sub in subs.data:
            _record_subscription(sub, price_to_tier, tiers)
        if not subs.has_more or not subs.data:
            return False
        starting_after = subs.data[-1].id
    logger.warning(
        "reconcile_all_stripe_tiers: hit %d-page cap for status=%s; remaining"
        " subscriptions not reconciled this run",
        _MAX_SUBSCRIPTION_PAGES,
        status,
    )
    return True


def _record_subscription(
    sub: stripe.Subscription,
    price_to_tier: dict[str, SubscriptionTier],
    tiers: dict[str, SubscriptionTier],
) -> None:
    """Map one subscription's customer to its tier, keeping the highest tier."""
    customer = sub.get("customer")
    if not isinstance(customer, str) or not customer:
        return
    items = sub.get("items", {}).get("data", [])
    if not items:
        return
    price_id = items[0].get("price", {}).get("id", "")
    tier = price_to_tier.get(price_id) if price_id else None
    if tier is None:
        return
    existing = tiers.get(customer)
    if existing is None or _TIER_RANK[tier] > _TIER_RANK[existing]:
        tiers[customer] = tier


_TIER_RANK: dict[SubscriptionTier, int] = {
    SubscriptionTier.NO_TIER: 0,
    SubscriptionTier.BASIC: 1,
    SubscriptionTier.PRO: 2,
    SubscriptionTier.MAX: 3,
    SubscriptionTier.BUSINESS: 4,
    SubscriptionTier.ENTERPRISE: 5,
}
