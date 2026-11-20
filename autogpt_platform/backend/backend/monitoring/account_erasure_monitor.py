"""Account erasures that started and did not finish.

An erasure claims its request (`startedAt`) before any step and sets
`completedAt` once everything is gone. A step that keeps failing, a service
left unset or a CLI run that crashed leaves the request claimed and
unfinished: the account is closed, past its 14 days and not erased, which the
30 days we promise don't allow for long. The hourly job logs its failures, but
a CLI run only prints to its terminal, and the job is off by default. This
check reads the request rows, so it catches both, and pages.
"""

import logging
from time import monotonic

from backend.util.clients import (
    get_database_manager_client,
    get_notification_manager_client,
)
from backend.util.metrics import sentry_capture_error
from backend.util.settings import Config

logger = logging.getLogger(__name__)
config = Config()

# A stalled erasure comes back on every check until someone runs it again.
# Page for each one at most this often (per scheduler process), and log the
# repeats at WARNING, so the alert channel is not trained into noise.
_REPAGE_SECS = 24 * 60 * 60
_paged_at: dict[str, float] = {}


def _unpaged(user_ids: list[str]) -> list[str]:
    now = monotonic()
    for stale in [
        user_id
        for user_id, paged_at in _paged_at.items()
        if now - paged_at >= _REPAGE_SECS
    ]:
        del _paged_at[stale]
    return [user_id for user_id in user_ids if user_id not in _paged_at]


def _mark_paged(user_ids: list[str]) -> None:
    now = monotonic()
    for user_id in user_ids:
        _paged_at[user_id] = now


class StalledAccountErasureException(Exception):
    """Raised (to Sentry) when an account erasure started and did not finish."""


def report_stalled_account_erasures() -> str:
    """Page for the erasures claimed over `account_erasure_stall_alert_secs`
    ago and not finished, each at most once a day. User IDs only: the alert
    goes to Discord and Sentry, neither of which should carry an email."""
    older_than_secs = config.account_erasure_stall_alert_secs
    user_ids = get_database_manager_client().stalled_account_erasures(
        older_than_secs=older_than_secs
    )
    if not user_ids:
        return "No stalled account erasure."

    summary = (
        f"account erasure(s) started over {older_than_secs / 3600:g}h ago and "
        "did not finish: a failed step, a service left unset, or a CLI run that "
        "crashed. Run `erase-account <user ID>` (dry run) to see where each one "
        "stops, fix that and run it with --apply; see "
        "docs/engineering/account-deletion.md."
    )
    msg = f"{len(user_ids)} {summary} User IDs: {', '.join(user_ids)}"
    # First, so the IDs are on record even when a page below fails.
    logger.warning(msg)
    if fresh := _unpaged(user_ids):
        # The IDs stay out of the Sentry issue's text, so every stall groups
        # into one issue.
        sentry_capture_error(StalledAccountErasureException(f"Stalled {summary}"))
        get_notification_manager_client().discord_system_alert(
            f"{len(fresh)} {summary} User IDs: {', '.join(fresh)}"
        )
        # Only once the page went out: a failed one is tried again next check.
        _mark_paged(fresh)
    return msg
