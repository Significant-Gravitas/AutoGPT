"""The two scheduled passes, and the per-user work they fan out.

`flush_matured_alerts` runs every minute and empties the debounce window;
`send_due_briefings` runs hourly and catches each user's local ~07:30.

**A pass decides who is due and publishes; it never assembles.** Each due user
becomes one message on the work queue, and a consumer does the reading and
rendering. That keeps a tick O(1) in the number of users, so a pass physically
cannot run past its own interval no matter how many users are due — which is
what the previous shape got wrong, silently falling further behind as the user
base grew. It also buys per-user retry, failure isolation and a dead-letter
queue from the same machinery the emails already use.

The work is claimed before it is acted on, because queue delivery is
at-least-once and the service can run more than one replica.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from zoneinfo import available_timezones

from backend.data.db_accessors import (
    agent_graph_attachment_lease,
    alert_condition_delivery_lease,
)
from backend.data.notifications import (
    NotificationEventModel,
    NotificationScope,
    PassWorkEvent,
    PassWorkKind,
)
from backend.data.user import BriefingCandidate
from backend.notifications import alerts, briefing, lifecycle
from backend.notifications.briefing_period import (
    BRIEFING_HOUR,
    is_briefing_due,
    period_window,
    resolve_zone,
)
from backend.notifications.dedupe import (
    PASS_CLAIM_TTL_SECONDS,
    claim_once,
    release_claim,
)
from backend.notifications.queue import queue_pass_work
from backend.notifications.tenancy import (
    authorized_notification_scopes,
    normalize_scopes,
    run_with_notification_guards,
    scope_keys,
)
from backend.util.clients import get_database_manager_async_client
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[NotificationRunner]")


def _db():
    """This service owns no Prisma connection; the DatabaseManager does."""
    return get_database_manager_async_client()


# Page size for the candidate walk. Bounds how many users are held in memory
# at once; it does not bound how many are considered.
CANDIDATE_PAGE_SIZE = 5000
DeliverNotification = Callable[[NotificationEventModel], Awaitable[bool]]


async def flush_matured_alerts() -> None:
    """Publish one flush job per user whose conditions have matured."""
    now = datetime.now(tz=timezone.utc)
    published = 0

    async for user_id in _matured_alert_user_ids():
        published += 1
        await queue_pass_work(
            PassWorkKind.ALERT_FLUSH.value,
            user_id,
            PassWorkEvent(
                kind=PassWorkKind.ALERT_FLUSH, user_id=user_id, scheduled_for=now
            ).model_dump_json(),
        )

    if published:
        logger.info(f"Fanned out {published} alert flushes")


async def send_due_briefings() -> None:
    """Publish one briefing job per user whose local briefing hour this is.

    The cadence check stays here because it is cheap and needs no extra read:
    the candidate row already carries the frequency and the last send. Only
    users that pass it become messages.
    """
    now = datetime.now(tz=timezone.utc)
    zones = _briefing_hour_timezones(now)
    considered = published = 0

    async for user in _briefing_candidates(zones):
        considered += 1
        if not is_briefing_due(
            user.briefing_frequency, user.timezone, now, user.last_briefing_at
        ):
            continue
        published += 1
        await queue_pass_work(
            PassWorkKind.BRIEFING.value,
            user.id,
            PassWorkEvent(
                kind=PassWorkKind.BRIEFING, user_id=user.id, scheduled_for=now
            ).model_dump_json(),
        )

    logger.info(f"Considered {considered} users, fanned out {published} briefings")


# ── the work itself, run by the queue consumer ──────────────────────────


async def run_pass_work(
    event: PassWorkEvent, deliver_notification: DeliverNotification
) -> None:
    """Do one user's share of a scheduled pass.

    Claimed first: delivery is at-least-once and there may be several
    replicas, so without this a redelivery is a second email. The claim key
    carries the period, so the next period is never suppressed by this one.
    """
    key = _claim_key(event)
    if not await claim_once(key, ttl_seconds=PASS_CLAIM_TTL_SECONDS):
        logger.debug(f"{event.kind.value} for {event.user_id} already claimed")
        return

    try:
        if event.kind is PassWorkKind.ALERT_FLUSH:
            await _flush_user_alerts(event.user_id, key, deliver_notification)
        elif event.kind is PassWorkKind.WELCOME:
            await lifecycle.send_welcome_for_session(event.context["session_id"])
        else:
            await _build_and_send_briefing(
                event.user_id, event.scheduled_for, key, deliver_notification
            )
    except Exception:
        # The consumer retries this message. Holding the claim would make the
        # retry find its own claim, return early, and drop the work for the
        # whole period — the claim outlives the retries by design.
        await release_claim(key)
        raise


def _claim_key(event: PassWorkEvent) -> str:
    """Identify the work's slot, not the moment it was published.

    Every publisher of one slot has to arrive at the same key or the claim
    stops being a claim: two schedulers reading the clock a microsecond apart
    would each get their own key and both send. WELCOME has no slot — Stripe
    redelivers at a fresh timestamp, so it keys on the checkout session, which
    is also the only identifier it carries (`user_id` is empty).
    """
    if event.kind is PassWorkKind.WELCOME:
        return f"{event.kind.value}:{event.context['session_id']}"
    slot = event.scheduled_for.replace(second=0, microsecond=0)
    if event.kind is PassWorkKind.BRIEFING:
        # Briefings are scheduled by the hour; alert flushes run every minute.
        slot = slot.replace(minute=0)
    return f"{event.kind.value}:{event.user_id}:{slot.isoformat()}"


async def _flush_user_alerts(
    user_id: str, delivery_id: str, deliver_notification: DeliverNotification
) -> None:
    # Raises for an unknown user rather than returning None, and that is left
    # to propagate — a transient database failure is what the retry is for.
    preference = await _db().get_user_notification_preference(user_id)
    candidates = await _db().get_pending_alert_condition_scopes(user_id)
    async with authorized_notification_scopes(user_id, candidates) as (
        authorized,
        guards,
    ):
        denied_keys = scope_keys(candidates) - scope_keys(authorized)
        if denied_keys:
            await _db().resolve_alert_conditions_for_scopes(
                user_id,
                [
                    NotificationScope(organization_id=org_id, team_id=team_id)
                    for org_id, team_id in denied_keys
                ],
            )
        built = await alerts.build_alert_email(
            user_id, preference.alerts_enabled, authorized
        )
        if built is None:
            return
        source_graph_ids = await _db().get_alert_condition_source_graph_ids(
            user_id, built.condition_ids
        )
        delivered = False
        stale_ids: list[str] = []
        async with AsyncExitStack() as stack:
            source_guard = None
            if source_graph_ids:
                source_guard = await stack.enter_async_context(
                    agent_graph_attachment_lease(source_graph_ids)
                )
            condition_guard = await stack.enter_async_context(
                alert_condition_delivery_lease(built.condition_ids)
            )
            stale_ids = await _db().get_stale_alert_condition_ids(
                user_id,
                built.condition_ids,
                built.authorization_scopes,
                False,
            )
            if stale_ids:
                logger.info("Dropping alert whose source condition changed")
            else:
                delivery = condition_guard.run(
                    deliver_notification(
                        alerts.alert_event(
                            user_id,
                            built.data,
                            built.authorization_scopes,
                            delivery_id,
                        )
                    )
                )
                if source_guard is not None:
                    delivery = source_guard.run(delivery)
                delivered = await run_with_notification_guards(delivery, guards)
                if delivered:
                    await _db().finalize_alert_delivery(
                        user_id,
                        built.condition_ids,
                        built.authorization_scopes,
                        datetime.now(tz=timezone.utc),
                    )


async def _build_and_send_briefing(
    user_id: str,
    now: datetime,
    delivery_id: str,
    deliver_notification: DeliverNotification,
) -> None:
    # No preference read here: `send_due_briefings` already gated on frequency
    # via `is_briefing_due`, and a user who has since vanished falls out on the
    # candidate lookup below rather than raising.
    user = await _db().get_briefing_candidate(user_id)
    if user is None:
        return
    window = period_window(user.briefing_frequency, user.timezone, now)
    resource_scopes = await _db().get_briefing_resource_scopes(
        user_id, window.start, window.end
    )
    alert_scopes = await _db().get_briefing_alert_condition_scopes(user_id)
    candidates = normalize_scopes([*resource_scopes, *alert_scopes])

    async with authorized_notification_scopes(user_id, candidates) as (
        authorized,
        guards,
    ):
        denied_alert_keys = scope_keys(alert_scopes) - scope_keys(authorized)
        if denied_alert_keys:
            await _db().resolve_alert_conditions_for_scopes(
                user_id,
                [
                    NotificationScope(organization_id=org_id, team_id=team_id)
                    for org_id, team_id in denied_alert_keys
                ],
            )
        built = await briefing.build_briefing(
            user_id,
            user.briefing_frequency,
            user.timezone,
            now,
            authorized,
        )
        if built is None:
            logger.debug(f"Nothing to brief for user {user_id}")
            return
        source_graph_ids = await _db().get_alert_condition_source_graph_ids(
            user_id, built.attention_condition_ids
        )
        delivered = False
        stale_ids: list[str] = []
        async with AsyncExitStack() as stack:
            source_guard = None
            if source_graph_ids:
                source_guard = await stack.enter_async_context(
                    agent_graph_attachment_lease(source_graph_ids)
                )
            condition_guard = None
            if built.attention_condition_ids:
                condition_guard = await stack.enter_async_context(
                    alert_condition_delivery_lease(built.attention_condition_ids)
                )
                stale_ids = await _db().get_stale_alert_condition_ids(
                    user_id,
                    built.attention_condition_ids,
                    built.authorization_scopes,
                    True,
                )
            if stale_ids:
                logger.info("Dropping briefing whose attention condition changed")
            else:
                delivery = deliver_notification(
                    briefing.briefing_event(
                        user_id,
                        built.data,
                        built.authorization_scopes,
                        delivery_id,
                    )
                )
                if condition_guard is not None:
                    delivery = condition_guard.run(delivery)
                if source_guard is not None:
                    delivery = source_guard.run(delivery)
                delivered = await run_with_notification_guards(delivery, guards)
                if delivered:
                    await _db().finalize_briefing_delivery(
                        user_id,
                        built.attention_condition_ids,
                        built.authorization_scopes,
                        datetime.now(tz=timezone.utc),
                        now,
                    )


async def _matured_alert_user_ids() -> AsyncIterator[str]:
    """Every user with a matured PENDING condition, paged.

    The underlying table holds one row per live condition for the whole
    platform and this runs every minute, so the read is bounded and walked
    rather than taken in one go.
    """
    cursor: str | None = None
    while True:
        page = await alerts.matured_alert_user_ids(after_user_id=cursor)
        for user_id in page.user_ids:
            yield user_id
        # Deduplication makes `user_ids` shorter than the rows read, so only
        # the page itself can say whether the walk is done.
        if page.exhausted or not page.user_ids:
            return
        cursor = page.user_ids[-1]


async def _briefing_candidates(zones: list[str]) -> AsyncIterator[BriefingCandidate]:
    """Users for whom it is currently the briefing hour, locally.

    Filtering on timezone in SQL keeps each hourly pass to roughly a
    twenty-fourth of the user base; `is_briefing_due` then applies the weekly
    and monthly cadence rules.

    Walks the whole set one page at a time, keyed on the last id seen, so the
    pass is bounded in memory without being bounded in coverage.
    """
    cursor: str | None = None
    while True:
        page = await _db().get_briefing_candidates(zones, cursor, CANDIDATE_PAGE_SIZE)
        if not page:
            return
        for user in page:
            yield user
        if len(page) < CANDIDATE_PAGE_SIZE:
            return
        cursor = page[-1].id


def _briefing_hour_timezones(now: datetime) -> list[str]:
    """Every timezone whose local clock currently reads the briefing hour.

    Users who never set one are treated as UTC, matching `resolve_zone`, so
    they are included when it is the briefing hour in UTC.
    """
    zones = [
        name
        for name in available_timezones()
        if now.astimezone(resolve_zone(name)).hour == BRIEFING_HOUR
    ]
    if now.astimezone(timezone.utc).hour == BRIEFING_HOUR:
        zones.append("not-set")
    return zones
