"""Periodic Stripe → DB subscription-tier reconciliation sweep.

Webhooks give low-latency tier updates; this sweep is the safety net that makes
Stripe the eventual source of truth. It pages through every active/trialing
Stripe subscription once, builds a candidate ``{customer_id -> tier}`` map,
then for every reconcilable user (has a Stripe customer, not ENTERPRISE) sets the
tier from the map (NO_TIER when the customer is absent). Manual grants (no Stripe
customer) and ENTERPRISE rows are never touched.

Pro/Max candidates without a completed activation for their current subscription pass
through subscription and settled-invoice reconciliation: tier assignment alone
does not establish that first paid activation and its usage generation completed.
Other unchanged tiers need no writes. Manual grants (no customer) remain excluded.
"""

import logging

import stripe
from prisma.enums import SubscriptionTier
from prisma.models import PaidUsageActivation, User
from pydantic import BaseModel, Field

from backend.data.credit import (
    alert_tier_reconciliation_discrepancy,
    build_price_to_tier_map,
    log_tier_reconciliation_discrepancy,
    set_subscription_tier,
    sync_subscription_from_stripe,
)
from backend.data.stripe_client import stripe_call

logger = logging.getLogger(__name__)

# Bound the Stripe pagination so a runaway/unexpected dataset can't loop forever.
# At 100 subs/page this caps a single sweep at 500k subscriptions; if it's ever
# hit the sweep logs and stops rather than silently truncating without notice.
_MAX_SUBSCRIPTION_PAGES = 5000
_PAGE_SIZE = 100
# How many failed trial subscription ids the run-health alert lists by name.
_MAX_ALERT_SUBSCRIPTION_IDS = 20


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
    failed_trial_subscriptions: list[str] = []
    discrepancies: list[TierDiscrepancy] = []


async def reconcile_all_stripe_tiers() -> ReconciliationSummary:
    """Reconcile every reconcilable user against live Stripe subscriptions."""
    customer_to_tier = await _build_customer_tier_map()
    summary = ReconciliationSummary(
        stripe_active_subscriptions=len(customer_to_tier.tiers),
        pagination_capped=customer_to_tier.capped,
        failed_trial_subscriptions=list(customer_to_tier.failed_trials),
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
    unknown_tier_customers = {c for c in customer_to_tier.failed_trials.values() if c}
    for user in candidates:
        await _reconcile_one(
            user,
            customer_to_tier.tiers,
            summary,
            map_complete,
            customer_to_tier.pro_subscriptions,
            unknown_tier_customers,
        )
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
    if (
        summary.pagination_capped
        or summary.failed_trial_subscriptions
        or summary.errors
    ):
        await _alert_sweep_incomplete(summary)
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


async def _alert_sweep_incomplete(summary: ReconciliationSummary) -> None:
    """Post one Discord (PLATFORM) alert when the sweep could not reconcile
    everyone: the Stripe snapshot was incomplete, a trial subscription failed its
    re-sync, or a user errored. Each leaves some tiers unreconciled, possibly on
    every run, so it must not stay buried in the logs."""
    problems = []
    if summary.pagination_capped:
        problems.append(
            f"Stripe snapshot incomplete (a list call failed or the page cap was "
            f"hit): {summary.skipped_incomplete} absence-based downgrade(s) held back."
        )
    failed = summary.failed_trial_subscriptions
    if failed:
        listed = ", ".join(f"`{s}`" for s in failed[:_MAX_ALERT_SUBSCRIPTION_IDS])
        if len(failed) > _MAX_ALERT_SUBSCRIPTION_IDS:
            listed += f" and {len(failed) - _MAX_ALERT_SUBSCRIPTION_IDS} more"
        problems.append(
            f"{len(failed)} trial subscription(s) failed their re-sync; their "
            f"customers' tiers were left unchanged: {listed}"
        )
    if summary.errors:
        problems.append(
            f"{summary.errors} user(s) were not reconciled, including any "
            f"skipped for a failed trial re-sync (see the logs for "
            f"reconcile_all_stripe_tiers)."
        )
    message = "\n".join(f"- {p}" for p in problems)
    logger.warning(
        "Stripe tier reconciliation sweep did not reconcile every user:\n%s", message
    )
    await alert_tier_reconciliation_discrepancy(
        f"⚠️ **Stripe tier reconciliation: run did not reconcile every user**\n"
        f"{message}"
    )


async def _reconcile_one(
    user: User,
    customer_to_tier: dict[str, SubscriptionTier],
    summary: ReconciliationSummary,
    map_complete: bool,
    pro_subscriptions: dict[str, dict] | None = None,
    unknown_tier_customers: set[str] | None = None,
) -> None:
    """Set one user's tier from the Stripe map, updating the summary counts."""
    # A trial subscription that failed its re-sync was still listed as active or
    # trialing, so this customer has a live subscription whose tier we don't
    # know. Leave their tier alone this run rather than downgrade them.
    if user.stripeCustomerId and user.stripeCustomerId in (
        unknown_tier_customers or set()
    ):
        summary.errors += 1
        logger.warning(
            "reconcile_all_stripe_tiers: skipped user %s; a trial subscription"
            " failed its re-sync",
            user.id[:8],
        )
        return
    current_tier = SubscriptionTier(user.subscriptionTier or SubscriptionTier.NO_TIER)
    target_tier = SubscriptionTier.NO_TIER
    if user.stripeCustomerId:
        target_tier = customer_to_tier.get(
            user.stripeCustomerId, SubscriptionTier.NO_TIER
        )
    # When the Stripe snapshot is incomplete (a failed list page or the
    # pagination cap), absence from the map is unreliable — the user may sit on
    # a page we never fetched. Never revoke a tier off a partial snapshot;
    # upgrades (the customer IS in the map) stay safe.
    in_map = bool(user.stripeCustomerId) and user.stripeCustomerId in customer_to_tier
    if not map_complete and not in_map:
        summary.skipped_incomplete += 1
        return
    try:
        pro_candidate = target_tier in (SubscriptionTier.PRO, SubscriptionTier.MAX)
        if pro_candidate:
            subscription = (pro_subscriptions or {}).get(user.stripeCustomerId or "")
            if subscription is None:
                raise ValueError("Paid reconciliation requires a subscription identity")
            # The list price is only a routing hint. Reconciliation refetches
            # Stripe under the activation lock and requires settled evidence.
            # Run this even for an unchanged tier to recover incomplete usage
            # activation; never grant Pro/Max by calling set_subscription_tier.
            if await _requires_pro_reconciliation(user, subscription, target_tier):
                target_tier = await _reconcile_pro_tier(user, subscription)
        if target_tier == current_tier:
            summary.unchanged += 1
            return
        if not pro_candidate:
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


async def _requires_pro_reconciliation(
    user: User, subscription: dict, target_tier: SubscriptionTier
) -> bool:
    if user.subscriptionTier != target_tier:
        return True
    activation = await PaidUsageActivation.prisma().find_unique(
        where={"userId": user.id}
    )
    return not (
        activation
        and activation.readyAt is not None
        and activation.stripeSubscriptionId == subscription.get("id")
    )


async def _reconcile_pro_tier(user: User, subscription: dict) -> SubscriptionTier:
    await sync_subscription_from_stripe(subscription, track_lifecycle=False)
    reconciled = await User.prisma().find_unique_or_raise(where={"id": user.id})
    return SubscriptionTier(reconciled.subscriptionTier)


class _CustomerTierMap(BaseModel):
    tiers: dict[str, SubscriptionTier]
    capped: bool
    pro_subscriptions: dict[str, dict] = Field(default_factory=dict)
    # Trial subscriptions whose re-sync failed: subscription id -> customer id.
    failed_trials: dict[str, str] = Field(default_factory=dict)


async def _build_customer_tier_map() -> _CustomerTierMap:
    """Page through all active+trialing Stripe subs into a customer→tier map.

    A customer with multiple active subs resolves to their highest tier so a
    leftover lower-tier sub never downgrades a user mid-upgrade.
    """
    price_to_tier = await build_price_to_tier_map()
    tiers: dict[str, SubscriptionTier] = {}
    pro_subscriptions: dict[str, dict] = {}
    failed_trials: dict[str, str] = {}
    capped = False
    for status in ("active", "trialing"):
        page_capped = await _collect_status_page(
            status, price_to_tier, tiers, pro_subscriptions, failed_trials
        )
        capped = capped or page_capped
    return _CustomerTierMap(
        tiers=tiers,
        capped=capped,
        pro_subscriptions=pro_subscriptions,
        failed_trials=failed_trials,
    )


async def _collect_status_page(
    status: str,
    price_to_tier: dict[str, SubscriptionTier],
    tiers: dict[str, SubscriptionTier],
    pro_subscriptions: dict[str, dict] | None = None,
    failed_trials: dict[str, str] | None = None,
) -> bool:
    """Accumulate one Stripe status's subscriptions into ``tiers``. Returns
    True if a list call failed or pagination was capped. A trial subscription
    whose re-sync fails is recorded in ``failed_trials`` instead: it only leaves
    that one customer's tier unknown, not the snapshot as a whole."""
    starting_after: str | None = None
    for _ in range(_MAX_SUBSCRIPTION_PAGES):
        list_kwargs: dict = {"status": status, "limit": _PAGE_SIZE}
        if starting_after:
            list_kwargs["starting_after"] = starting_after
        try:
            subs = await stripe_call(stripe.Subscription.list_async, **list_kwargs)
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
            if (sub.get("metadata") or {}).get("trial_enrollment_id"):
                try:
                    await sync_subscription_from_stripe(
                        dict(sub), track_lifecycle=False
                    )
                except Exception:
                    logger.exception(
                        "Trial reconciliation failed for subscription %s; its"
                        " customer is skipped this run",
                        sub.id,
                    )
                    if failed_trials is not None:
                        failed_trials[sub.id] = str(sub.get("customer") or "")
                    continue
                user = await User.prisma().find_first(
                    where={"stripeCustomerId": str(sub.customer)}
                )
                if user:
                    tier = SubscriptionTier(user.subscriptionTier)
                    existing = tiers.get(str(sub.customer))
                    if existing is None or _TIER_RANK[tier] > _TIER_RANK[existing]:
                        tiers[str(sub.customer)] = tier
                        if (
                            tier in (SubscriptionTier.PRO, SubscriptionTier.MAX)
                            and pro_subscriptions is not None
                        ):
                            pro_subscriptions[str(sub.customer)] = dict(sub)
                continue
            _record_subscription(sub, price_to_tier, tiers, pro_subscriptions)
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
    pro_subscriptions: dict[str, dict] | None = None,
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
    if (
        tier in (SubscriptionTier.PRO, SubscriptionTier.MAX)
        and pro_subscriptions is not None
        and (existing is None or _TIER_RANK[tier] > _TIER_RANK[existing])
    ):
        pro_subscriptions[customer] = dict(sub)


_TIER_RANK: dict[SubscriptionTier, int] = {
    SubscriptionTier.NO_TIER: 0,
    SubscriptionTier.TRIAL: 0,
    SubscriptionTier.BASIC: 1,
    SubscriptionTier.PRO: 2,
    SubscriptionTier.MAX: 3,
    SubscriptionTier.BUSINESS: 4,
    SubscriptionTier.ENTERPRISE: 5,
}
