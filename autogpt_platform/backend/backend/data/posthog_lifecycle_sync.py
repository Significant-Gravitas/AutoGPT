"""Keep subscription status and lifecycle dates on the PostHog person.

These are person properties, not events: each sync sends one ``$set`` on the
platform user id, the same id the browser passes to ``identify``, so they land
on the canonical person. The mapping itself is ``posthog_lifecycle``.

Three ways in, all best-effort (a failure is logged, never raised):

- ``schedule_posthog_lifecycle_sync`` after a write that can change the
  snapshot (signup, tier changes, ``sync_subscription_from_stripe``). It runs
  in the background, so a webhook or a signup never waits on Stripe or
  PostHog.
- ``sync_all_posthog_lifecycles``, daily from the scheduler: the safety net
  for a missed hook and for a trial that runs out on the clock.
- ``scripts/backfill_posthog_lifecycle.py``, the same sweep run by hand.
"""

import asyncio
import logging
import weakref
from collections import Counter
from collections.abc import Coroutine, Sequence
from datetime import UTC, datetime
from typing import Any

import stripe
from prisma.models import SubscriptionTrial, User
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema
from backend.data.posthog_lifecycle import (
    LifecycleSnapshot,
    LifecycleUser,
    StripeSubscriptionFacts,
    compute_lifecycle_snapshot,
)
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_trial import TrialState
from backend.util.posthog_client import get_posthog_client
from backend.util.posthog_events import PostHogEvent

logger = logging.getLogger(__name__)

_SYNC_TIMEOUT_SECONDS = 60
_MAX_CONCURRENT_SYNCS = 4


def schedule_posthog_lifecycle_sync(
    user_id: str | None = None, *, stripe_customer_id: str | None = None
) -> None:
    """Sync one user in the background, after the caller's write.

    Never raises and never makes the caller wait. Pass the user id, or the
    Stripe customer id when that is all the caller has; a customer id is
    resolved to its user first, so both end up coordinated per user:

    - a sync that is queued and hasn't started covers this call, because it
      reads the state when it starts;
    - a sync that is already running is followed by exactly one more, so the
      last one sent always read the state after this write;
    - two syncs for one user never run at the same time in this process, so
      an older read can't be sent after a newer one here. Another pod, or the
      daily sweep (which reads Stripe once at its start), can still send an
      older read; the next sync or sweep corrects it.
    """
    try:
        if get_posthog_client() is None:
            return
        if user_id:
            _schedule_user(user_id)
        elif stripe_customer_id:
            _spawn(_resolve_customer(stripe_customer_id))
    except Exception:
        logger.warning("Failed to schedule a lifecycle sync", exc_info=True)


# Per user id: "queued", "running", or "rerun" (running, then run once more).
_syncs: dict[str, str] = {}
_background_tasks: set[asyncio.Task] = set()
# One per event loop: an asyncio.Semaphore binds to the loop it is first used in.
_slots: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _schedule_user(user_id: str) -> None:
    state = _syncs.get(user_id)
    if state == "running":
        _syncs[user_id] = "rerun"
    elif state is None:
        _spawn(_run_user_sync(user_id))
        _syncs[user_id] = "queued"


def _spawn(coro: Coroutine[Any, Any, None]) -> None:
    try:
        task = asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        coro.close()
        raise
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _resolve_customer(stripe_customer_id: str) -> None:
    try:
        user = await User.prisma().find_first(
            where={"stripeCustomerId": stripe_customer_id}
        )
        if user is not None:
            _schedule_user(user.id)
    except Exception:
        logger.warning(
            f"Lifecycle sync: can't resolve customer {stripe_customer_id}",
            exc_info=True,
        )


async def _run_user_sync(user_id: str) -> None:
    # Cleanup wraps the slot as well: a task cancelled while waiting for a
    # slot must not leave its entry behind, or every later request for this
    # user would see "queued" and never run.
    cancelled = False
    try:
        # Bounded so a burst of hooks can't fan out into a burst of Stripe
        # calls that rate-limits the billing code sharing the Stripe account.
        async with _sync_slot():
            while True:
                _syncs[user_id] = "running"
                try:
                    await asyncio.wait_for(
                        sync_posthog_lifecycle(user_id), timeout=_SYNC_TIMEOUT_SECONDS
                    )
                except Exception:
                    logger.warning(
                        f"Lifecycle sync failed for user {user_id}", exc_info=True
                    )
                if _syncs.get(user_id) != "rerun":
                    return
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        state = _syncs.pop(user_id, None)
        if cancelled and state in ("queued", "rerun"):
            # Requests absorbed by this task would otherwise be lost.
            _requeue(user_id)


def _requeue(user_id: str) -> None:
    try:
        _schedule_user(user_id)
    except Exception:
        logger.warning(f"Lifecycle sync: can't requeue user {user_id}", exc_info=True)


def _sync_slot() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    slot = _slots.get(loop)
    if slot is None:
        slot = _slots[loop] = asyncio.Semaphore(_MAX_CONCURRENT_SYNCS)
    return slot


async def sync_posthog_lifecycle(
    user_id: str,
    *,
    stripe_subscriptions: Sequence[StripeSubscriptionFacts] | None = None,
) -> LifecycleSnapshot | None:
    """Send the user's current lifecycle snapshot. Never raises.

    ``stripe_subscriptions`` is the customer's complete list, for a caller
    that already has it. Otherwise it is read from Stripe, and only for a
    user with a Stripe customer. A webhook payload is deliberately not used
    in its place: it is one subscription as of one event, which is exactly
    the "last event received" the status must not depend on.
    """
    try:
        if not user_id or get_posthog_client() is None:
            return None
        snapshot = await load_lifecycle_snapshot(
            user_id, stripe_subscriptions=stripe_subscriptions
        )
        if snapshot is not None:
            send_lifecycle_snapshot(user_id, snapshot)
        return snapshot
    except Exception:
        logger.warning(
            f"Failed to sync lifecycle properties for user {user_id}", exc_info=True
        )
        return None


async def load_lifecycle_snapshot(
    user_id: str,
    *,
    stripe_subscriptions: Sequence[StripeSubscriptionFacts] | None = None,
    now: datetime | None = None,
) -> LifecycleSnapshot | None:
    """Read the user's current state and map it. None if the user is gone."""
    row = await User.prisma().find_unique(
        where={"id": user_id}, include={"subscriptionTrial": True}
    )
    if row is None:
        return None
    if stripe_subscriptions is None:
        stripe_subscriptions = (
            await list_customer_subscriptions(row.stripeCustomerId)
            if row.stripeCustomerId
            else []
        )
    trial_row = row.subscriptionTrial
    return compute_lifecycle_snapshot(
        user=LifecycleUser(
            id=row.id,
            created_at=row.createdAt,
            stripe_customer_id=row.stripeCustomerId,
            subscription_tier=row.subscriptionTier,
        ),
        trial=TrialState.from_db(trial_row) if trial_row else None,
        subscriptions=stripe_subscriptions,
        now=now or datetime.now(UTC),
    )


async def list_customer_subscriptions(
    customer_id: str,
) -> list[StripeSubscriptionFacts]:
    page = await stripe_call(
        stripe.Subscription.list_async, customer=customer_id, status="all", limit=100
    )
    return [
        StripeSubscriptionFacts.model_validate(sub)
        async for sub in stripe_list_items(page)
    ]


def send_lifecycle_snapshot(user_id: str, snapshot: LifecycleSnapshot) -> bool:
    """Queue the ``$set``. False when analytics is off or queueing failed."""
    try:
        client = get_posthog_client()
        if not user_id or client is None:
            return False
        to_set, to_unset = snapshot.person_update()
        properties: dict[str, object] = {"$set": to_set}
        if to_unset:
            properties["$unset"] = to_unset
        client.capture(
            PostHogEvent.SET_PERSON_PROPERTIES.value,
            distinct_id=user_id,
            properties=properties,
        )
        return True
    except Exception:
        logger.warning(
            f"Failed to send lifecycle properties for user {user_id}", exc_info=True
        )
        return False


async def start_posthog_lifecycle_sweep() -> bool:
    """Start the daily sweep in the background and return at once.

    The scheduler reaches this over RPC, whose call timeout is shorter than a
    large sweep can take and which retries on a timeout; waiting for the sweep
    there would start a second one alongside the first. Returns False when a
    sweep is already running in this process.
    """
    if any(not task.done() for task in _sweeps):
        return False
    task = asyncio.get_running_loop().create_task(_run_sweep())
    _sweeps.add(task)
    task.add_done_callback(_sweeps.discard)
    return True


_sweeps: set[asyncio.Task] = set()


async def _run_sweep() -> None:
    try:
        summary = await sync_all_posthog_lifecycles()
    except Exception:
        logger.exception("Lifecycle sweep failed")
        return
    if summary.aborted:
        logger.warning(f"Lifecycle sweep aborted: {summary.aborted}")


class LifecycleSyncSummary(BaseModel):
    dry_run: bool
    stripe_subscriptions: int = 0
    users: int = 0
    sent: int = 0
    errors: int = 0
    status_counts: dict[str, int] = {}
    aborted: str | None = None


async def sync_all_posthog_lifecycles(
    *,
    dry_run: bool = False,
    all_users: bool = False,
    batch_size: int = 500,
    now: datetime | None = None,
) -> LifecycleSyncSummary:
    """Recompute and send the snapshot of every user with billing history.

    Billing history means a Stripe customer or a trial row; ``all_users``
    widens it to everyone (a ``signed`` user gets ``signup_at`` and the
    status). Stripe is read once, as a full ``Subscription.list(status=all)``,
    not per user. A failed listing aborts the run: a partial list would mark
    people whose subscriptions sit on the missing pages as ``signed``.

    Sends are flushed after each batch so a large run can't overflow the
    PostHog client's in-memory queue, which drops events once it is full.
    """
    summary = LifecycleSyncSummary(dry_run=dry_run)
    client = None if dry_run else get_posthog_client()
    if not dry_run and client is None:
        summary.aborted = "PostHog is not configured"
        return summary
    try:
        by_customer = await _all_subscriptions_by_customer()
    except Exception:
        logger.exception("Lifecycle sync: listing Stripe subscriptions failed")
        summary.aborted = "Stripe listing failed"
        return summary
    summary.stripe_subscriptions = sum(len(subs) for subs in by_customer.values())
    counts: Counter[str] = Counter()
    after_id = ""
    while users := await _lifecycle_user_batch(after_id, batch_size, all_users):
        await _sync_batch(users, by_customer, now or datetime.now(UTC), summary, counts)
        after_id = users[-1].id
        if client is not None:
            await asyncio.to_thread(client.flush)
    summary.status_counts = dict(sorted(counts.items()))
    logger.info(f"Lifecycle sync finished: {summary.model_dump_json()}")
    return summary


async def _sync_batch(
    users: list[LifecycleUser],
    by_customer: dict[str, list[StripeSubscriptionFacts]],
    now: datetime,
    summary: LifecycleSyncSummary,
    counts: Counter[str],
) -> None:
    trials = await _trials_by_user([user.id for user in users])
    for user in users:
        summary.users += 1
        try:
            if user.id in trials and trials[user.id] is None:
                # Mapping without the trial would call them ``signed``.
                raise ValueError("unreadable trial row")
            snapshot = compute_lifecycle_snapshot(
                user=user,
                trial=trials.get(user.id),
                subscriptions=by_customer.get(user.stripe_customer_id or "", []),
                now=now,
            )
        except Exception:
            summary.errors += 1
            logger.warning(f"Lifecycle sync: can't map user {user.id}", exc_info=True)
            continue
        counts[snapshot.subscription_status] += 1
        if summary.dry_run:
            continue
        if send_lifecycle_snapshot(user.id, snapshot):
            summary.sent += 1
        else:
            summary.errors += 1


async def _all_subscriptions_by_customer() -> dict[str, list[StripeSubscriptionFacts]]:
    page = await stripe_call(stripe.Subscription.list_async, status="all", limit=100)
    by_customer: dict[str, list[StripeSubscriptionFacts]] = {}
    async for sub in stripe_list_items(page):
        facts = StripeSubscriptionFacts.model_validate(sub)
        if facts.customer:
            by_customer.setdefault(facts.customer, []).append(facts)
    return by_customer


_USER_BATCH = """
    SELECT u."id", u."createdAt" AS created_at,
           u."stripeCustomerId" AS stripe_customer_id,
           u."subscriptionTier"::text AS subscription_tier
    FROM {schema_prefix}"User" u
    WHERE u."id" > $1 AND ($3::boolean OR u."stripeCustomerId" IS NOT NULL
        OR EXISTS (SELECT 1 FROM {schema_prefix}"SubscriptionTrial" t
                   WHERE t."userId" = u."id"))
    ORDER BY u."id" LIMIT $2
"""


async def _lifecycle_user_batch(
    after_id: str, batch_size: int, all_users: bool
) -> list[LifecycleUser]:
    return await query_raw_with_schema(
        _USER_BATCH, after_id, batch_size, all_users, model=LifecycleUser
    )


async def _trials_by_user(user_ids: list[str]) -> dict[str, TrialState | None]:
    """Trial per user id. None marks a row that exists but can't be read."""
    rows = await SubscriptionTrial.prisma().find_many(
        where={"userId": {"in": user_ids}}
    )
    trials: dict[str, TrialState | None] = {}
    for row in rows:
        try:
            trials[row.userId] = TrialState.from_db(row)
        except Exception:
            trials[row.userId] = None
            logger.warning(
                f"Lifecycle sync: unreadable trial row for user {row.userId}",
                exc_info=True,
            )
    return trials
