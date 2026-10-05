"""Exercise the offer backfill against transaction-local PostgreSQL tables."""

import os
from pathlib import Path
from urllib.parse import urlparse

import psycopg2
import pytest
from psycopg2.extras import Json, RealDictCursor

BACKFILL = (
    Path(__file__).resolve().parents[2]
    / "scripts/sql/update_free_trial_usage_limits.sql"
)
LIMITS = {
    "daily_cost_limit": 100_000_000,
    "weekly_cost_limit": 20_000_000,
    "total_cost_limit": 20_000_000,
}
ORIGINAL_OFFER = {
    "version": "original-offer",
    "new_users_from": "2026-09-01T00:00:00Z",
    "duration_days": 14,
    "tier": "PRO",
    "billing_cycle": "monthly",
    "daily_cost_limit": 1_500_000,
    "weekly_cost_limit": 20_000_000,
    "total_cost_limit": 40_000_000,
    "onboarding_credit_amount": 300,
    "price_id": "price_original",
    "unit_amount": 2000,
    "currency": "usd",
}

pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires explicitly selected disposable trial database",
)


@pytest.mark.parametrize(
    "status,cost,ends_in_days,verified,converted,consumed,changed",
    [
        ("trialing", 0, 7, True, False, True, True),
        ("trialing", 5_000_000, 7, True, False, True, True),
        ("trialing", 20_000_000, 7, True, False, True, True),
        ("trialing", 25_000_000, 7, True, False, True, True),
        ("checkout_pending", 0, None, False, False, False, True),
        ("checkout_pending", 0, None, False, False, True, False),
        ("trialing", 5_000_000, 0, True, False, True, False),
        ("trialing", 5_000_000, -1, True, False, True, False),
        ("trialing", 5_000_000, None, True, False, True, False),
        ("trialing", 5_000_000, 7, False, False, True, True),
        ("trialing", 5_000_000, 7, True, True, True, False),
        ("active", 5_000_000, 7, True, True, True, False),
        ("active", 5_000_000, 7, True, False, True, False),
        ("checkout_pending", 0, None, False, True, False, False),
        ("canceled", 5_000_000, 7, True, False, True, False),
        ("past_due", 5_000_000, 7, True, False, True, False),
        ("incomplete_expired", 0, None, False, False, False, False),
    ],
)
def test_free_trial_limit_backfill_preserves_state(
    trial_cursor, status, cost, ends_in_days, verified, converted, consumed, changed
):
    trial_cursor.execute(
        """INSERT INTO "SubscriptionTrial" VALUES (
            'trial-1', %s, %s, %s,
            CASE WHEN %s THEN CURRENT_TIMESTAMP - INTERVAL '7 days' END,
            CURRENT_TIMESTAMP - INTERVAL '7 days',
            CURRENT_TIMESTAMP + %s * INTERVAL '1 day',
            CASE WHEN %s THEN CURRENT_TIMESTAMP - INTERVAL '7 days' END,
            CASE WHEN %s THEN CURRENT_TIMESTAMP - INTERVAL '1 day' END,
            TRUE, CURRENT_TIMESTAMP - INTERVAL '8 days',
            CURRENT_TIMESTAMP - INTERVAL '1 hour'
        )""",
        (
            Json(ORIGINAL_OFFER),
            status,
            cost,
            verified,
            ends_in_days,
            consumed,
            converted,
        ),
    )
    trial_cursor.execute('SELECT * FROM "SubscriptionTrial"')
    before = dict(trial_cursor.fetchone())
    trial_cursor.execute(BACKFILL.read_text().format(schema_prefix=""))
    assert trial_cursor.rowcount == int(changed)
    trial_cursor.execute('SELECT * FROM "SubscriptionTrial"')
    after = dict(trial_cursor.fetchone())
    expected = {**before, "offer": {**ORIGINAL_OFFER, **LIMITS}} if changed else before
    assert after == expected
    if changed:
        assert max(0, after["offer"]["total_cost_limit"] - cost) == max(
            0, 20_000_000 - cost
        )
    trial_cursor.execute(BACKFILL.read_text().format(schema_prefix=""))
    assert trial_cursor.rowcount == 0
    trial_cursor.execute('SELECT * FROM "SubscriptionTrial"')
    assert dict(trial_cursor.fetchone()) == after


@pytest.fixture
def trial_cursor():
    target = urlparse(os.environ["DATABASE_URL"])
    local = (target.hostname, target.port, target.path) == (
        "127.0.0.1",
        15432,
        "/trial_test",
    )
    ci = os.environ.get("GITHUB_ACTIONS") == "true" and (
        target.hostname,
        target.port,
    ) == ("localhost", 5432)
    assert local or ci, "Trial migration tests require a disposable database"
    connection = psycopg2.connect(target._replace(query="").geturl())
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """CREATE TEMP TABLE "SubscriptionTrial" (
                    "id" TEXT PRIMARY KEY,
                    "offer" JSONB NOT NULL,
                    "status" TEXT NOT NULL,
                    "costMicrodollars" BIGINT NOT NULL,
                    "cardVerifiedAt" TIMESTAMPTZ,
                    "startedAt" TIMESTAMPTZ,
                    "endsAt" TIMESTAMPTZ,
                    "consumedAt" TIMESTAMPTZ,
                    "convertedAt" TIMESTAMPTZ,
                    "cancelAtPeriodEnd" BOOLEAN NOT NULL,
                    "createdAt" TIMESTAMPTZ NOT NULL,
                    "updatedAt" TIMESTAMPTZ NOT NULL
                ) ON COMMIT DROP"""
            )
            yield cursor
    finally:
        connection.rollback()
        connection.close()
