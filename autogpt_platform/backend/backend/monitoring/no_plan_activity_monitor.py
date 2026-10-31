"""Daily report of API activity by accounts without an active plan.

Once a day, posts one system alert listing, for the previous UTC day, the
accounts the paywall refuses (`is_user_paywalled`) whose requests to the
internal and external API passed a threshold, and groups of them that signed up
or signed in from one IP. It reports only: banning stays an admin's click on
the admin users page, which also shows the emails and IPs this alert leaves out.
"""

import asyncio
import logging
from datetime import UTC, date, datetime, time, timedelta

from pydantic import BaseModel

from backend.copilot.rate_limit import is_user_paywalled
from backend.data.api_activity import (
    APIRequestCounts,
    find_busy_accounts,
    get_request_counts,
)
from backend.data.db_accessors import auth_ip_events_db, user_db
from backend.util.clients import get_notification_manager_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Keeps one alert readable; the admin users page has the rest.
_MAX_ACCOUNTS_LISTED = 20
_MAX_GROUPS_LISTED = 5
_PAYWALL_READS_IN_FLIGHT = 10


class NoPlanAccount(BaseModel):
    user_id: str
    requests: APIRequestCounts
    created_at: datetime | None


class NoPlanActivity(BaseModel):
    busy: list[NoPlanAccount]
    groups: list[list[NoPlanAccount]]


async def report_no_plan_api_activity() -> str:
    """Post yesterday's report, or nothing when no account crossed a threshold."""
    if not settings.config.no_plan_activity_report_enabled:
        return "The no-plan API activity report is disabled."
    day = datetime.now(UTC).date() - timedelta(days=1)
    activity = await find_no_plan_activity(day)
    if not activity.busy and not activity.groups:
        return f"No account without a plan crossed a threshold on {day}."
    message = _format_report(day, activity)
    alerts = get_notification_manager_client()
    await asyncio.to_thread(alerts.discord_system_alert, message)
    return message


async def find_no_plan_activity(day: date) -> NoPlanActivity:
    config = settings.config
    since = datetime.combine(day, time.min, tzinfo=UTC)
    busy = await find_busy_accounts(day, config.no_plan_activity_min_requests)
    groups = await auth_ip_events_db().find_accounts_sharing_an_ip(
        since, since + timedelta(days=1), config.no_plan_activity_min_accounts_per_ip
    )
    no_plan = await _without_plan({*busy, *(uid for group in groups for uid in group)})

    busy_ids = sorted(
        (uid for uid in busy if uid in no_plan), key=lambda uid: -busy[uid].total
    )
    group_ids = _distinct(
        [sorted(uid for uid in group if uid in no_plan) for group in groups],
        config.no_plan_activity_min_accounts_per_ip,
    )
    grouped = {uid for group in group_ids for uid in group}
    requests = {**await get_request_counts(day, sorted(grouped - busy.keys())), **busy}
    created = await user_db().get_account_created_at(sorted({*busy_ids, *grouped}))

    def account(uid: str) -> NoPlanAccount:
        return NoPlanAccount(
            user_id=uid, requests=requests[uid], created_at=created.get(uid)
        )

    return NoPlanActivity(
        busy=[account(uid) for uid in busy_ids],
        groups=[[account(uid) for uid in group] for group in group_ids],
    )


async def _without_plan(user_ids: set[str]) -> set[str]:
    in_flight = asyncio.Semaphore(_PAYWALL_READS_IN_FLIGHT)

    async def paywalled(user_id: str) -> bool:
        async with in_flight:
            try:
                return await is_user_paywalled(user_id)
            except Exception as e:
                logger.warning(
                    f"Plan lookup for {user_id} failed; left out of the no-plan "
                    f"activity report: {e}"
                )
                return False

    ids = sorted(user_ids)
    results = await asyncio.gather(*(paywalled(uid) for uid in ids))
    return {uid for uid, refused in zip(ids, results) if refused}


def _distinct(groups: list[list[str]], min_accounts: int) -> list[list[str]]:
    """Groups still large enough once paying accounts are dropped, each once:
    an account's IPv4 and IPv6 addresses yield the same group twice."""
    seen: set[frozenset[str]] = set()
    kept = []
    for group in groups:
        key = frozenset(group)
        if len(group) >= min_accounts and key not in seen:
            seen.add(key)
            kept.append(group)
    return kept


def _format_report(day: date, activity: NoPlanActivity) -> str:
    config = settings.config
    lines = [f"**API activity by accounts without an active plan on {day} (UTC)**"]
    if activity.busy:
        lines.append(
            f"{_count(len(activity.busy), 'account')} with "
            f"{config.no_plan_activity_min_requests:,} or more requests:"
        )
        lines += _account_lines(activity.busy, indent="")
    if activity.groups:
        lines.append(
            f"{_count(len(activity.groups), 'group')} of "
            f"{config.no_plan_activity_min_accounts_per_ip} or more accounts that "
            "signed up or signed in from one IP address:"
        )
        for group in activity.groups[:_MAX_GROUPS_LISTED]:
            lines.append(f"- {len(group)} accounts:")
            lines += _account_lines(group, indent="  ")
        if len(activity.groups) > _MAX_GROUPS_LISTED:
            lines.append(f"- and {len(activity.groups) - _MAX_GROUPS_LISTED} more")
    return "\n".join(lines)


def _account_lines(accounts: list[NoPlanAccount], indent: str) -> list[str]:
    lines = [
        f"{indent}- `{a.user_id}`: {a.requests.total:,} requests "
        f"({a.requests.internal:,} internal, {a.requests.external:,} external), "
        f"{_age(a.created_at)}, <{_admin_users_url(a.user_id)}>"
        for a in accounts[:_MAX_ACCOUNTS_LISTED]
    ]
    if len(accounts) > _MAX_ACCOUNTS_LISTED:
        lines.append(f"{indent}- and {len(accounts) - _MAX_ACCOUNTS_LISTED} more")
    return lines


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _age(created_at: datetime | None) -> str:
    if created_at is None:
        return "age unknown"
    age = datetime.now(UTC) - created_at
    if age < timedelta(days=1):
        return f"{int(age.total_seconds() // 3600)}h old"
    return f"{age.days}d old"


def _admin_users_url(user_id: str) -> str:
    base = settings.config.frontend_base_url or settings.config.platform_base_url
    return f"{base}/admin/users?user={user_id}"
