"""Shared fixtures for the notification service's tests.

The important one is `db_client`. Everything this service reads or writes goes
through the DatabaseManager RPC, so that client is the boundary these tests
should mock — and mocking it is what keeps them honest.

The suite previously mocked `User.prisma()` directly. That looked equivalent
and was not: it meant the tests supplied a working database to a process that
has no connection to one, so a hundred-plus green tests sat on top of a service
whose scheduled passes failed on every single tick. Mocking here instead means
a module that reaches past the RPC fails its test rather than passing it.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import AsyncMock

import pytest
from prisma.enums import BriefingFrequency

from backend.data.alerts import MaturedAlertPage
from backend.data.notifications import NotificationPreference

NOW = datetime(2026, 8, 3, 7, 30, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def server() -> None:
    """The notification suite is pure logic; it needs no live stack."""
    return None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup() -> Iterator[None]:
    yield


def make_db_client(**overrides) -> SimpleNamespace:
    """A stand-in `DatabaseManagerAsyncClient`.

    Defaults are the empty/no-op answer for every method the notification
    service calls, so a test overrides only what it is actually about. Passing
    an unknown name is deliberately allowed — a new RPC method should not
    require touching every test that predates it.
    """
    client = SimpleNamespace(
        # Preferences and users
        # The whole model, not just the fields one test happened to read:
        # the consumer gates on `daily_limit` and `wants_notification` reads
        # the frequency and the verdict switch, so a partial stub turns a real
        # regression into an AttributeError somewhere unrelated.
        get_user_notification_preference=AsyncMock(
            return_value=NotificationPreference(
                user_id="user-1",
                email="sam@example.com",
                briefing_frequency=BriefingFrequency.WEEKLY,
                alerts_enabled=True,
                store_verdicts_enabled=True,
                daily_limit=3,
            )
        ),
        get_user_by_id=AsyncMock(return_value=None),
        get_user_email_verification=AsyncMock(return_value=True),
        # Alert conditions
        get_users_with_matured_alerts=AsyncMock(
            return_value=MaturedAlertPage(user_ids=[], exhausted=True)
        ),
        get_pending_alert_conditions=AsyncMock(return_value=[]),
        count_alerts_sent_since=AsyncMock(return_value=0),
        mark_alert_conditions_sent=AsyncMock(),
        mark_alert_conditions_deferred=AsyncMock(),
        get_briefing_alert_conditions=AsyncMock(return_value=[]),
        mark_alert_conditions_briefed=AsyncMock(),
        raise_alert_condition=AsyncMock(),
        resolve_alert_condition=AsyncMock(return_value=True),
        # Briefing assembly
        get_agent_period_stats=AsyncMock(return_value=[]),
        get_top_scored_runs=AsyncMock(return_value=[]),
        count_active_agents=AsyncMock(return_value=0),
        get_briefing_credit_balance=AsyncMock(return_value=0.0),
        get_briefing_candidates=AsyncMock(return_value=[]),
        get_briefing_candidate=AsyncMock(return_value=None),
        set_last_briefing_at=AsyncMock(),
        get_graph_execution=AsyncMock(return_value=None),
    )
    for name, value in overrides.items():
        setattr(client, name, value)
    return client


@pytest.fixture
def db_client() -> SimpleNamespace:
    return make_db_client()
