"""Unit tests for CoPilot rate limiting."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisClusterException, RedisError

from .rate_limit import (
    _DEFAULT_TIER_MULTIPLIERS,
    _DEFAULT_TIER_WORKSPACE_STORAGE_MB,
    DEFAULT_TIER,
    TIER_MULTIPLIERS,
    CoPilotUsagePublic,
    CoPilotUsageStatus,
    RateLimitExceeded,
    RateLimitUnavailable,
    SubscriptionTier,
    UsageWindow,
    _daily_key,
    _daily_reset_time,
    _fetch_cost_limits_flag,
    _fetch_tier_multipliers_flag,
    _fetch_workspace_storage_limits_flag,
    _weekly_key,
    _weekly_reset_time,
    acquire_reset_lock,
    build_budget_ctx,
    check_rate_limit,
    get_daily_reset_count,
    get_global_rate_limits,
    get_remaining_usd_budget,
    get_tier_multipliers,
    get_usage_status,
    get_user_tier,
    get_workspace_storage_limit_bytes,
    get_workspace_storage_limits_mb,
    increment_daily_reset_count,
    record_cost_usage,
    release_reset_lock,
    reset_daily_usage,
    reset_user_usage,
    set_user_tier,
)

_USER = "test-user-rl"


# ---------------------------------------------------------------------------
# RateLimitExceeded
# ---------------------------------------------------------------------------


class TestRateLimitExceeded:
    def test_message_contains_window_name(self):
        exc = RateLimitExceeded("daily", datetime.now(UTC) + timedelta(hours=1))
        assert "daily" in str(exc)

    def test_message_contains_reset_time(self):
        exc = RateLimitExceeded(
            "weekly", datetime.now(UTC) + timedelta(hours=2, minutes=30)
        )
        msg = str(exc)
        # Allow for slight timing drift (29m or 30m)
        assert "2h " in msg
        assert "Resets in" in msg

    def test_message_minutes_only_when_under_one_hour(self):
        exc = RateLimitExceeded("daily", datetime.now(UTC) + timedelta(minutes=15))
        msg = str(exc)
        assert "Resets in" in msg
        # Should not have "0h"
        assert "0h" not in msg

    def test_message_says_now_when_resets_at_is_in_the_past(self):
        """Negative delta (clock skew / stale TTL) should say 'now', not '-1h -30m'."""
        exc = RateLimitExceeded("daily", datetime.now(UTC) - timedelta(minutes=5))
        assert "Resets in now" in str(exc)


# ---------------------------------------------------------------------------
# get_usage_status
# ---------------------------------------------------------------------------


class TestGetUsageStatus:
    @pytest.mark.asyncio
    async def test_returns_redis_values(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["500", "2000"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            status = await get_usage_status(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

        assert isinstance(status, CoPilotUsageStatus)
        assert status.daily.used == 500
        assert status.daily.limit == 10000
        assert status.weekly.used == 2000
        assert status.weekly.limit == 50000

    @pytest.mark.asyncio
    async def test_returns_zeros_when_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis down"),
        ):
            status = await get_usage_status(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

        assert status.daily.used == 0
        assert status.weekly.used == 0

    @pytest.mark.asyncio
    async def test_partial_none_daily_counter(self):
        """Daily counter is None (new day), weekly has usage."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[None, "3000"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            status = await get_usage_status(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

        assert status.daily.used == 0
        assert status.weekly.used == 3000

    @pytest.mark.asyncio
    async def test_partial_none_weekly_counter(self):
        """Weekly counter is None (start of week), daily has usage."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["500", None])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            status = await get_usage_status(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

        assert status.daily.used == 500
        assert status.weekly.used == 0

    @pytest.mark.asyncio
    async def test_resets_at_daily_is_next_midnight_utc(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["0", "0"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            status = await get_usage_status(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

        now = datetime.now(UTC)
        # Daily reset should be within 24h
        assert status.daily.resets_at > now
        assert status.daily.resets_at <= now + timedelta(hours=24, seconds=5)


# ---------------------------------------------------------------------------
# check_rate_limit
# ---------------------------------------------------------------------------


class TestCheckRateLimit:
    @pytest.mark.asyncio
    async def test_allows_when_under_limit(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["100", "200"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            # Should not raise
            await check_rate_limit(
                _USER, daily_cost_limit=10000, weekly_cost_limit=50000
            )

    @pytest.mark.asyncio
    async def test_raises_when_daily_limit_exceeded(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["10000", "200"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(
                    _USER, daily_cost_limit=10000, weekly_cost_limit=50000
                )
            assert exc_info.value.window == "daily"

    @pytest.mark.asyncio
    async def test_raises_when_weekly_limit_exceeded(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["100", "50000"])

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(
                    _USER, daily_cost_limit=10000, weekly_cost_limit=50000
                )
            assert exc_info.value.window == "weekly"

    @pytest.mark.asyncio
    async def test_raises_unavailable_when_redis_connection_error(self):
        """Fail-closed: ConnectionError must surface as RateLimitUnavailable so
        the API layer can map it to HTTP 503 instead of silently letting the
        request through and bypassing the per-user USD cap."""
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis down"),
        ):
            with pytest.raises(RateLimitUnavailable):
                await check_rate_limit(
                    _USER, daily_cost_limit=10000, weekly_cost_limit=50000
                )

    @pytest.mark.asyncio
    async def test_raises_unavailable_when_redis_redis_error(self):
        """Fail-closed for redis-py RedisError (covers cluster-down /
        CROSSSLOT-style failures during a brown-out)."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=RedisError("cluster down"))
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            with pytest.raises(RateLimitUnavailable):
                await check_rate_limit(
                    _USER, daily_cost_limit=10000, weekly_cost_limit=50000
                )

    @pytest.mark.asyncio
    async def test_raises_unavailable_when_redis_cluster_exception(self):
        """Fail-closed for RedisClusterException — covers SlotNotCoveredError
        raised during a GKE rolling restart when slot coverage is briefly
        incomplete. This subclass does NOT inherit from RedisError, so it
        must be caught explicitly."""
        try:
            from redis.exceptions import SlotNotCoveredError as _ClusterExc
        except ImportError:
            _ClusterExc = RedisClusterException
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=_ClusterExc("slot 1234 uncovered"))
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            with pytest.raises(RateLimitUnavailable):
                await check_rate_limit(
                    _USER, daily_cost_limit=10000, weekly_cost_limit=50000
                )

    @pytest.mark.asyncio
    async def test_raises_unavailable_when_os_error(self):
        """Fail-closed for OSError (DNS / TCP-level failures during a node
        rolling-restart)."""
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=OSError("ECONNREFUSED"),
        ):
            with pytest.raises(RateLimitUnavailable):
                await check_rate_limit(
                    _USER, daily_cost_limit=10000, weekly_cost_limit=50000
                )

    @pytest.mark.asyncio
    async def test_raises_unavailable_when_counter_is_corrupt(self):
        """Fail-closed when a Redis counter holds a non-numeric value — the
        ``int(...)`` cast would otherwise raise ValueError and surface as a
        500, but the same fail-closed reasoning applies (we cannot prove the
        user is under their cap)."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["not-a-number", "200"])
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            with pytest.raises(RateLimitUnavailable):
                await check_rate_limit(
                    _USER, daily_cost_limit=10000, weekly_cost_limit=50000
                )

    @pytest.mark.asyncio
    async def test_zero_limit_means_blocked_not_unlimited(self):
        """A limit of 0 means "no spend allowed", not "unlimited" — there
        is no real-world unlimited tier. This is the regression behind the
        autopilot paywall bypass: the previous ``> 0`` check let the
        multiplier-collapse path (which returns 0 for paywalled users) slip
        through silently."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["0", "0"])
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(_USER, daily_cost_limit=0, weekly_cost_limit=0)
            assert exc_info.value.window == "daily"


class TestCoPilotUsagePublicFromStatus:
    """Public-shape projection must surface a 0-limit window as fully
    exhausted (the user is paywalled / blocked), not as `None`. Hiding
    the window would tell the UI "no cap configured" — the same silent
    bypass that motivated this PR."""

    @staticmethod
    def _status(daily_limit: int, weekly_limit: int) -> CoPilotUsageStatus:
        from .rate_limit import UsageWindow

        return CoPilotUsageStatus(
            daily=UsageWindow(
                used=0,
                limit=daily_limit,
                resets_at=_daily_reset_time(),
            ),
            weekly=UsageWindow(
                used=0,
                limit=weekly_limit,
                resets_at=_weekly_reset_time(),
            ),
            tier=DEFAULT_TIER,
            reset_cost=0,
        )

    def test_zero_limit_renders_as_fully_exhausted(self):
        public = CoPilotUsagePublic.from_status(self._status(0, 0))
        assert public.daily is not None
        assert public.daily.percent_used == 100.0
        assert public.weekly is not None
        assert public.weekly.percent_used == 100.0

    def test_positive_limit_renders_normally(self):
        # daily limit $10, used $0 → 0% used
        from .rate_limit import UsageWindow

        status = CoPilotUsageStatus(
            daily=UsageWindow(used=0, limit=10_000_000, resets_at=_daily_reset_time()),
            weekly=UsageWindow(
                used=5_000_000, limit=10_000_000, resets_at=_weekly_reset_time()
            ),
            tier=DEFAULT_TIER,
            reset_cost=0,
        )
        public = CoPilotUsagePublic.from_status(status)
        assert public.daily is not None
        assert public.daily.percent_used == 0.0
        assert public.weekly is not None
        assert public.weekly.percent_used == 50.0


class TestEnforcePaymentPaywall:
    """The dep bypasses ``get_user_tier``'s fail-open default and goes
    through ``_fetch_user_tier`` directly so a transient DB blip raises
    503 instead of silently 402-ing every paid user."""

    @pytest.mark.asyncio
    async def test_blocks_no_tier_user_when_flag_on(self, mocker):
        """Headline behaviour: NO_TIER + ENABLE_PLATFORM_PAYMENT on raises
        :class:`UserPaywalledError`. The app-level exception handler in
        ``rest_api.py`` (and the external API's ``fastapi_app.py``) maps
        this to HTTP 402 — the dep itself doesn't construct an
        HTTPException so all gates share the same exception type."""
        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=SubscriptionTier.NO_TIER),
        )
        mocker.patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=True),
        )
        from .rate_limit import UserPaywalledError, enforce_payment_paywall

        with pytest.raises(UserPaywalledError) as exc_info:
            await enforce_payment_paywall(_USER)
        assert "subscription" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_allows_no_tier_user_when_flag_off(self, mocker):
        """Beta cohort: flag off → NO_TIER user passes through (no exception)."""
        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=SubscriptionTier.NO_TIER),
        )
        mocker.patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=False),
        )
        from .rate_limit import enforce_payment_paywall

        await enforce_payment_paywall(_USER)  # must not raise

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tier",
        [
            SubscriptionTier.BASIC,
            SubscriptionTier.PRO,
            SubscriptionTier.MAX,
            SubscriptionTier.BUSINESS,
            SubscriptionTier.ENTERPRISE,
        ],
    )
    async def test_allows_paid_tier_even_when_flag_on(self, mocker, tier):
        """Even with the flag on, paid tiers must never be blocked."""
        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=tier),
        )
        is_feature_enabled_mock = mocker.patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=True),
        )
        from .rate_limit import enforce_payment_paywall

        await enforce_payment_paywall(_USER)  # must not raise
        # Short-circuits on tier check; flag lookup never happens for paid
        # tiers (avoids an unnecessary LD round-trip per paywalled route).
        is_feature_enabled_mock.assert_not_called()


class TestIsUserPaywalled:
    """``is_user_paywalled`` is the shared core check used by both the
    route-level dep and the function-level gate inside ``add_graph_execution``.
    Strict: propagates lookup failures so the caller decides how to respond
    (route → 503; background job → fail-open)."""

    @pytest.mark.asyncio
    async def test_no_tier_with_flag_on_is_paywalled(self, mocker):
        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=SubscriptionTier.NO_TIER),
        )
        mocker.patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=True),
        )
        from .rate_limit import is_user_paywalled

        assert await is_user_paywalled(_USER) is True

    @pytest.mark.asyncio
    async def test_no_tier_with_flag_off_is_not_paywalled(self, mocker):
        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=SubscriptionTier.NO_TIER),
        )
        mocker.patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=False),
        )
        from .rate_limit import is_user_paywalled

        assert await is_user_paywalled(_USER) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tier",
        [
            SubscriptionTier.BASIC,
            SubscriptionTier.PRO,
            SubscriptionTier.MAX,
            SubscriptionTier.BUSINESS,
            SubscriptionTier.ENTERPRISE,
        ],
    )
    async def test_paid_tier_short_circuits_without_flag_lookup(self, mocker, tier):
        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=tier),
        )
        flag_mock = mocker.patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=True),
        )
        from .rate_limit import is_user_paywalled

        assert await is_user_paywalled(_USER) is False
        flag_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_propagates_lookup_failure(self, mocker):
        """The helper does NOT swallow generic tier-lookup errors — callers
        decide. The route-dep maps to 503; background callers
        (add_graph_execution) fail open."""
        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        )
        from .rate_limit import is_user_paywalled

        with pytest.raises(RuntimeError):
            await is_user_paywalled(_USER)

    @pytest.mark.asyncio
    async def test_user_not_found_treated_as_no_tier_when_flag_on(self, mocker):
        """``_UserNotFoundError`` (no DB row, or no ``subscription_tier``
        set yet on a fresh signup) is treated as NO_TIER so callers without
        a generic ``except`` (e.g. the external API ``execute_graph_block``
        route) get a real 402 instead of a leaked 500."""
        from .rate_limit import _UserNotFoundError, is_user_paywalled

        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(side_effect=_UserNotFoundError(_USER)),
        )
        mocker.patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=True),
        )
        assert await is_user_paywalled(_USER) is True

    @pytest.mark.asyncio
    async def test_user_not_found_treated_as_no_tier_when_flag_off(self, mocker):
        """Beta cohort: missing tier + flag off → not paywalled (same
        passthrough as a real NO_TIER user with the flag off)."""
        from .rate_limit import _UserNotFoundError, is_user_paywalled

        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(side_effect=_UserNotFoundError(_USER)),
        )
        mocker.patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=False),
        )
        assert await is_user_paywalled(_USER) is False


class TestUserPaywalledError:
    def test_default_message(self):
        from .rate_limit import UserPaywalledError

        assert "subscription" in str(UserPaywalledError()).lower()

    def test_custom_message(self):
        from .rate_limit import UserPaywalledError

        assert str(UserPaywalledError("custom")) == "custom"


class TestFetchUserTierErrorPropagation:
    """``_fetch_user_tier`` must distinguish "row missing" (raises
    ``ValueError`` via ``get_user_by_id``) from "transient DB error"
    (Prisma connection failure etc). Without this distinction a
    Supabase blip would silently degrade every paying user to NO_TIER
    and ``enforce_payment_paywall`` would 402 them — contradicting its
    503-on-blip contract. Regression locked here."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        from .rate_limit import _fetch_user_tier

        _fetch_user_tier.cache_clear()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_value_error_collapses_to_user_not_found(self, mocker):
        """ValueError from get_user_by_id is the documented "missing user"
        signal — it's correct to map this to NO_TIER (via
        _UserNotFoundError → caught in is_user_paywalled)."""
        mock_db = mocker.MagicMock()
        mock_db.get_user_by_id = AsyncMock(side_effect=ValueError("User not found"))
        mocker.patch("backend.copilot.rate_limit.user_db", return_value=mock_db)
        from .rate_limit import _fetch_user_tier, _UserNotFoundError

        with pytest.raises(_UserNotFoundError):
            await _fetch_user_tier(_USER)

    @pytest.mark.asyncio
    async def test_db_outage_propagates(self, mocker):
        """Non-ValueError exceptions (Prisma connection failure, generic
        RuntimeError) propagate as-is so route layer can map to 503."""
        mock_db = mocker.MagicMock()
        mock_db.get_user_by_id = AsyncMock(side_effect=RuntimeError("Prisma down"))
        mocker.patch("backend.copilot.rate_limit.user_db", return_value=mock_db)
        from .rate_limit import _fetch_user_tier

        with pytest.raises(RuntimeError, match="Prisma down"):
            await _fetch_user_tier(_USER)

    @pytest.mark.asyncio
    async def test_db_outage_does_not_collapse_to_402_for_paying_user(self, mocker):
        """End-to-end: a Supabase blip during enforce_payment_paywall
        for any user (paying or not) maps to 503, NOT 402. Locks the
        contract advertised in enforce_payment_paywall's docstring."""
        mock_db = mocker.MagicMock()
        mock_db.get_user_by_id = AsyncMock(side_effect=RuntimeError("Supabase blip"))
        mocker.patch("backend.copilot.rate_limit.user_db", return_value=mock_db)
        from fastapi import HTTPException

        from .rate_limit import enforce_payment_paywall

        with pytest.raises(HTTPException) as exc_info:
            await enforce_payment_paywall(_USER)
        assert exc_info.value.status_code == 503  # not 402
        assert exc_info.value.headers.get("Retry-After") == "30"


class TestEnforcePaymentPaywallInline:
    """``enforce_payment_paywall`` doubles as both a JWT route dep AND an
    inline call from non-JWT routes (external API graph + block execute).
    These tests cover the inline form — the dep form is exercised
    transitively by the chat / internal-block route tests via FastAPI."""

    @pytest.mark.asyncio
    async def test_blocks_paywalled_user(self, mocker):
        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=SubscriptionTier.NO_TIER),
        )
        mocker.patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=True),
        )
        from .rate_limit import UserPaywalledError, enforce_payment_paywall

        with pytest.raises(UserPaywalledError):
            await enforce_payment_paywall(_USER)

    @pytest.mark.asyncio
    async def test_passes_paid_user(self, mocker):
        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=SubscriptionTier.PRO),
        )
        from .rate_limit import enforce_payment_paywall

        await enforce_payment_paywall(_USER)  # must not raise

    @pytest.mark.asyncio
    async def test_db_failure_raises_503(self, mocker):
        """Lookup failure → 503 + Retry-After (not 402, not 500). This is
        the asymmetry fix: external API + internal graph routes used to
        fail-open via the deep gate; now they fail-closed at the route."""
        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        )
        from fastapi import HTTPException

        from .rate_limit import enforce_payment_paywall

        with pytest.raises(HTTPException) as exc_info:
            await enforce_payment_paywall(_USER)
        assert exc_info.value.status_code == 503
        assert exc_info.value.headers.get("Retry-After") == "30"


class TestEnforcePaymentPaywallContinued:
    """Additional ``enforce_payment_paywall`` cases that depend on the
    helpers above. Kept separate so the related ``TestIsUserPaywalled``
    block stays compact above."""

    @pytest.mark.asyncio
    async def test_db_failure_raises_503_not_402(self, mocker):
        """Transient DB outage during tier lookup must NOT 402 every paid
        user (the silent-bypass concern raised by sentry/CodeRabbit on
        the previous round). It must surface as 503 with a Retry-After
        header so the client retries shortly instead of treating the
        outage as a permanent paywall."""
        mocker.patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        )
        is_feature_enabled_mock = mocker.patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=True),
        )
        from fastapi import HTTPException

        from .rate_limit import enforce_payment_paywall

        with pytest.raises(HTTPException) as exc_info:
            await enforce_payment_paywall(_USER)
        assert exc_info.value.status_code == 503
        assert exc_info.value.headers.get("Retry-After") == "30"
        # Flag lookup never runs when tier is indeterminate — we don't
        # know which cohort to apply.
        is_feature_enabled_mock.assert_not_called()


# ---------------------------------------------------------------------------
# record_cost_usage
# ---------------------------------------------------------------------------


class TestRecordCostUsage:
    @staticmethod
    def _make_pipeline_mock() -> MagicMock:
        """Create a pipeline mock with sync methods and async execute."""
        pipe = MagicMock()
        pipe.execute = AsyncMock(return_value=[])
        return pipe

    @pytest.mark.asyncio
    async def test_increments_redis_counters(self):
        mock_pipe = self._make_pipeline_mock()
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_cost_usage(_USER, cost_microdollars=123_456)

        # Should call incrby twice (daily + weekly) with the same cost
        incrby_calls = mock_pipe.incrby.call_args_list
        assert len(incrby_calls) == 2
        assert incrby_calls[0].args[1] == 123_456  # daily
        assert incrby_calls[1].args[1] == 123_456  # weekly

    @pytest.mark.asyncio
    async def test_skips_when_cost_is_zero(self):
        mock_redis = AsyncMock()

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_cost_usage(_USER, cost_microdollars=0)

        # Should not call pipeline at all
        mock_redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_cost_is_negative(self):
        """Negative costs are clamped to zero and skip the pipeline."""
        mock_redis = AsyncMock()

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_cost_usage(_USER, cost_microdollars=-10)

        mock_redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_sets_expire_on_both_keys(self):
        """Pipeline should call expire for both daily and weekly keys."""
        mock_pipe = self._make_pipeline_mock()
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await record_cost_usage(_USER, cost_microdollars=5_000)

        expire_calls = mock_pipe.expire.call_args_list
        assert len(expire_calls) == 2

        # Daily key TTL should be positive (seconds until next midnight)
        daily_ttl = expire_calls[0].args[1]
        assert daily_ttl >= 1

        # Weekly key TTL should be positive (seconds until next Monday)
        weekly_ttl = expire_calls[1].args[1]
        assert weekly_ttl >= 1

    @pytest.mark.asyncio
    async def test_handles_redis_failure_gracefully(self):
        """Should not raise when Redis is unavailable."""
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis down"),
        ):
            # Should not raise
            await record_cost_usage(_USER, cost_microdollars=5_000)

    @pytest.mark.asyncio
    async def test_handles_redis_error_during_pipeline_execute(self):
        """Should not raise when pipeline.execute() fails with RedisError."""
        mock_pipe = self._make_pipeline_mock()
        mock_pipe.execute = AsyncMock(side_effect=RedisError("Pipeline failed"))
        mock_redis = AsyncMock()
        mock_redis.pipeline = lambda **_kw: mock_pipe

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            # Should not raise — fail-open
            await record_cost_usage(_USER, cost_microdollars=5_000)


# ---------------------------------------------------------------------------
# SubscriptionTier and tier multipliers
# ---------------------------------------------------------------------------


class TestSubscriptionTier:
    def test_tier_values(self):
        assert SubscriptionTier.BASIC.value == "BASIC"
        assert SubscriptionTier.PRO.value == "PRO"
        assert SubscriptionTier.MAX.value == "MAX"
        assert SubscriptionTier.BUSINESS.value == "BUSINESS"
        assert SubscriptionTier.ENTERPRISE.value == "ENTERPRISE"

    def test_tier_multipliers(self):
        # Float-typed so LD-provided fractional multipliers compose naturally;
        # equality against int literals still holds for the whole defaults.
        # NO_TIER is 0.0 — explicit "no active subscription" state;
        # rate-limited routes refuse with 429 (backend half of the paywall).
        assert TIER_MULTIPLIERS[SubscriptionTier.NO_TIER] == 0.0
        assert TIER_MULTIPLIERS[SubscriptionTier.BASIC] == 1.0
        assert TIER_MULTIPLIERS[SubscriptionTier.PRO] == 5.0
        assert TIER_MULTIPLIERS[SubscriptionTier.MAX] == 20.0
        assert TIER_MULTIPLIERS[SubscriptionTier.BUSINESS] == 60.0
        assert TIER_MULTIPLIERS[SubscriptionTier.ENTERPRISE] == 60.0
        assert TIER_MULTIPLIERS is _DEFAULT_TIER_MULTIPLIERS

    def test_default_tier_is_no_tier(self):
        assert DEFAULT_TIER == SubscriptionTier.NO_TIER

    def test_usage_status_includes_tier(self):
        now = datetime.now(UTC)
        status = CoPilotUsageStatus(
            daily=UsageWindow(used=0, limit=100, resets_at=now + timedelta(hours=1)),
            weekly=UsageWindow(used=0, limit=500, resets_at=now + timedelta(days=1)),
        )
        assert status.tier == SubscriptionTier.NO_TIER

    def test_usage_status_with_custom_tier(self):
        now = datetime.now(UTC)
        status = CoPilotUsageStatus(
            daily=UsageWindow(used=0, limit=100, resets_at=now + timedelta(hours=1)),
            weekly=UsageWindow(used=0, limit=500, resets_at=now + timedelta(days=1)),
            tier=SubscriptionTier.PRO,
        )
        assert status.tier == SubscriptionTier.PRO


# ---------------------------------------------------------------------------
# get_tier_multipliers (LD-backed resolver)
# ---------------------------------------------------------------------------


class TestGetTierMultipliers:
    @pytest.fixture(autouse=True)
    def _clear_flag_cache(self):
        """Clear the LD flag cache between tests so patches don't leak."""
        _fetch_tier_multipliers_flag.cache_clear()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_defaults_when_flag_unset(self):
        """With no LD override, the resolver returns the default map."""
        with patch(
            "backend.copilot.rate_limit.get_feature_flag_value",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await get_tier_multipliers()
        assert result == {t.value: m for t, m in _DEFAULT_TIER_MULTIPLIERS.items()}

    @pytest.mark.asyncio
    async def test_ld_override(self):
        """LD override populates the targeted tiers; others inherit defaults."""
        with patch(
            "backend.copilot.rate_limit.get_feature_flag_value",
            new_callable=AsyncMock,
            return_value={"PRO": 7.5, "BUSINESS": 25},
        ):
            result = await get_tier_multipliers()
        assert result["PRO"] == 7.5
        assert result["BUSINESS"] == 25.0
        # Untouched tiers inherit defaults.
        assert result["BASIC"] == _DEFAULT_TIER_MULTIPLIERS[SubscriptionTier.BASIC]
        assert result["MAX"] == _DEFAULT_TIER_MULTIPLIERS[SubscriptionTier.MAX]
        assert (
            result["ENTERPRISE"]
            == _DEFAULT_TIER_MULTIPLIERS[SubscriptionTier.ENTERPRISE]
        )

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back(self):
        """A non-object LD value (string, list, bool) falls back to defaults."""
        with patch(
            "backend.copilot.rate_limit.get_feature_flag_value",
            new_callable=AsyncMock,
            return_value="broken",
        ):
            result = await get_tier_multipliers()
        assert result == {t.value: m for t, m in _DEFAULT_TIER_MULTIPLIERS.items()}

    @pytest.mark.asyncio
    async def test_unknown_tier_key_skipped(self):
        """Unknown tier keys and non-positive values are silently ignored."""
        with patch(
            "backend.copilot.rate_limit.get_feature_flag_value",
            new_callable=AsyncMock,
            return_value={"PRO": 3, "BOGUS": 99, "MAX": -1, "BUSINESS": "nope"},
        ):
            result = await get_tier_multipliers()
        assert result["PRO"] == 3.0
        # MAX had a non-positive override → falls back to default.
        assert result["MAX"] == _DEFAULT_TIER_MULTIPLIERS[SubscriptionTier.MAX]
        # BUSINESS had an unparseable override → falls back to default.
        assert (
            result["BUSINESS"] == _DEFAULT_TIER_MULTIPLIERS[SubscriptionTier.BUSINESS]
        )

    @pytest.mark.asyncio
    async def test_ld_failure_falls_back(self):
        """LD lookup raising propagates to defaults, not up the call stack."""
        with patch(
            "backend.copilot.rate_limit.get_feature_flag_value",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LD SDK not initialized"),
        ):
            result = await get_tier_multipliers()
        assert result == {t.value: m for t, m in _DEFAULT_TIER_MULTIPLIERS.items()}


class TestGetWorkspaceStorageLimits:
    @pytest.fixture(autouse=True)
    def _clear_flag_cache(self):
        """Clear the LD flag cache between tests so patches don't leak."""
        _fetch_workspace_storage_limits_flag.cache_clear()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_defaults_when_flag_unset(self):
        """With no LD override, the resolver returns the default map."""
        with patch(
            "backend.copilot.rate_limit.get_feature_flag_value",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await get_workspace_storage_limits_mb()
        assert result == {
            t.value: mb for t, mb in _DEFAULT_TIER_WORKSPACE_STORAGE_MB.items()
        }

    @pytest.mark.asyncio
    async def test_ld_override(self):
        """LD override populates targeted tiers; others inherit defaults."""
        with patch(
            "backend.copilot.rate_limit.get_feature_flag_value",
            new_callable=AsyncMock,
            return_value={"NO_TIER": 300, "PRO": 2048},
        ):
            result = await get_workspace_storage_limits_mb()
        assert result["NO_TIER"] == 300
        assert result["PRO"] == 2048
        assert (
            result["BASIC"]
            == _DEFAULT_TIER_WORKSPACE_STORAGE_MB[SubscriptionTier.BASIC]
        )
        assert result["MAX"] == _DEFAULT_TIER_WORKSPACE_STORAGE_MB[SubscriptionTier.MAX]

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back(self):
        """A non-object LD value falls back to defaults."""
        with patch(
            "backend.copilot.rate_limit.get_feature_flag_value",
            new_callable=AsyncMock,
            return_value="broken",
        ):
            result = await get_workspace_storage_limits_mb()
        assert result == {
            t.value: mb for t, mb in _DEFAULT_TIER_WORKSPACE_STORAGE_MB.items()
        }

    @pytest.mark.asyncio
    async def test_unknown_tier_key_and_invalid_values_skipped(self):
        """Unknown tiers and invalid values degrade to defaults per key."""
        with patch(
            "backend.copilot.rate_limit.get_feature_flag_value",
            new_callable=AsyncMock,
            return_value={"NO_TIER": 300, "BOGUS": 999, "MAX": -1, "BUSINESS": "nope"},
        ):
            result = await get_workspace_storage_limits_mb()
        assert result["NO_TIER"] == 300
        assert result["MAX"] == _DEFAULT_TIER_WORKSPACE_STORAGE_MB[SubscriptionTier.MAX]
        assert (
            result["BUSINESS"]
            == _DEFAULT_TIER_WORKSPACE_STORAGE_MB[SubscriptionTier.BUSINESS]
        )

    @pytest.mark.asyncio
    async def test_ld_failure_falls_back(self):
        """LD lookup raising propagates to defaults, not up the call stack."""
        with patch(
            "backend.copilot.rate_limit.get_feature_flag_value",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LD SDK not initialized"),
        ):
            result = await get_workspace_storage_limits_mb()
        assert result == {
            t.value: mb for t, mb in _DEFAULT_TIER_WORKSPACE_STORAGE_MB.items()
        }


# ---------------------------------------------------------------------------
# get_global_rate_limits — LD-flag cost limits parsing
# ---------------------------------------------------------------------------


class TestGetGlobalRateLimitsCostLimitsFlag:
    """Coverage for the ``copilot-cost-limits`` JSON flag parsing path."""

    _CONFIG_DAILY = 625_000
    _CONFIG_WEEKLY = 3_125_000

    @pytest.fixture(autouse=True)
    def _clear_flag_cache(self):
        _fetch_tier_multipliers_flag.cache_clear()  # type: ignore[attr-defined]
        _fetch_cost_limits_flag.cache_clear()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_flag_unset_uses_config_defaults(self):
        async def _ld(_flag_key: str, _uid: str, default):
            return default  # LD returns default → None for cost-limits

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BASIC,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=_ld,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._CONFIG_DAILY, self._CONFIG_WEEKLY
            )
        assert daily == self._CONFIG_DAILY
        assert weekly == self._CONFIG_WEEKLY
        assert tier == SubscriptionTier.BASIC

    @pytest.mark.asyncio
    async def test_flag_with_both_keys_honoured(self):
        async def _ld(flag_key: str, _uid: str, default):
            if "cost-limits" in flag_key.lower():
                return {"daily": 2_000_000, "weekly": 10_000_000}
            return default

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BASIC,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=_ld,
            ),
        ):
            daily, weekly, _ = await get_global_rate_limits(
                _USER, self._CONFIG_DAILY, self._CONFIG_WEEKLY
            )
        assert daily == 2_000_000
        assert weekly == 10_000_000

    @pytest.mark.asyncio
    async def test_flag_with_only_daily_weekly_defaults(self):
        async def _ld(flag_key: str, _uid: str, default):
            if "cost-limits" in flag_key.lower():
                return {"daily": 9_999_999}
            return default

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BASIC,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=_ld,
            ),
        ):
            daily, weekly, _ = await get_global_rate_limits(
                _USER, self._CONFIG_DAILY, self._CONFIG_WEEKLY
            )
        assert daily == 9_999_999
        assert weekly == self._CONFIG_WEEKLY

    @pytest.mark.asyncio
    async def test_non_dict_payload_falls_back_and_warns(self, caplog):
        async def _ld(flag_key: str, _uid: str, default):
            if "cost-limits" in flag_key.lower():
                return "not-a-dict"
            return default

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BASIC,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=_ld,
            ),
            caplog.at_level("WARNING"),
        ):
            daily, weekly, _ = await get_global_rate_limits(
                _USER, self._CONFIG_DAILY, self._CONFIG_WEEKLY
            )
        assert daily == self._CONFIG_DAILY
        assert weekly == self._CONFIG_WEEKLY
        assert any("copilot-cost-limits" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_invalid_per_key_values_fall_back(self):
        """Negative / non-int per-key values resolve to the config default for
        that key while any valid key survives."""

        async def _ld(flag_key: str, _uid: str, default):
            if "cost-limits" in flag_key.lower():
                return {"daily": -5, "weekly": "oops"}
            return default

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BASIC,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=_ld,
            ),
        ):
            daily, weekly, _ = await get_global_rate_limits(
                _USER, self._CONFIG_DAILY, self._CONFIG_WEEKLY
            )
        assert daily == self._CONFIG_DAILY
        assert weekly == self._CONFIG_WEEKLY

    @pytest.mark.asyncio
    async def test_partial_invalid_key_preserves_valid_key(self):
        """A valid daily + invalid weekly → daily honoured, weekly defaults."""

        async def _ld(flag_key: str, _uid: str, default):
            if "cost-limits" in flag_key.lower():
                return {"daily": 1_234_567, "weekly": -1}
            return default

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BASIC,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=_ld,
            ),
        ):
            daily, weekly, _ = await get_global_rate_limits(
                _USER, self._CONFIG_DAILY, self._CONFIG_WEEKLY
            )
        assert daily == 1_234_567
        assert weekly == self._CONFIG_WEEKLY

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_value",
        [True, False, "100", 1.9, [1, 2], None],
        ids=["bool-true", "bool-false", "str-numeric", "float", "list", "null"],
    )
    async def test_non_strict_int_values_rejected(self, bad_value):
        """Strings like '100', booleans, floats, lists — none should coerce.

        Docstring promises "non-int values are skipped"; this asserts the
        strict-check (``isinstance(x, int) and not isinstance(x, bool)``)
        actually rejects values ``int()`` would silently coerce.
        """

        async def _ld(flag_key: str, _uid: str, default):
            if "cost-limits" in flag_key.lower():
                return {"daily": bad_value}
            return default

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BASIC,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=_ld,
            ),
        ):
            daily, weekly, _ = await get_global_rate_limits(
                _USER, self._CONFIG_DAILY, self._CONFIG_WEEKLY
            )
        assert daily == self._CONFIG_DAILY
        assert weekly == self._CONFIG_WEEKLY


# ---------------------------------------------------------------------------
# get_user_tier
# ---------------------------------------------------------------------------


class TestGetUserTier:
    @pytest.fixture(autouse=True)
    def _clear_tier_cache(self):
        """Clear the get_user_tier cache before each test."""
        get_user_tier.cache_clear()  # type: ignore[attr-defined]

    def _mock_user_db(
        self, subscription_tier: str | None = None, raises: Exception | None = None
    ):
        """Return a patched user_db() whose get_user_by_id behaves as specified."""
        mock_db = AsyncMock()
        if raises is not None:
            mock_db.get_user_by_id = AsyncMock(side_effect=raises)
        else:
            mock_user = MagicMock()
            mock_user.subscription_tier = subscription_tier
            mock_db.get_user_by_id = AsyncMock(return_value=mock_user)
        return mock_db

    @pytest.mark.asyncio
    async def test_returns_tier_from_db(self):
        """Should return the tier stored in the user record."""
        mock_db = self._mock_user_db(subscription_tier="PRO")
        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db):
            tier = await get_user_tier(_USER)
        assert tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_returns_default_when_user_not_found(self):
        """Should return DEFAULT_TIER when user is not in the DB."""
        mock_db = self._mock_user_db(raises=Exception("not found"))
        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db):
            tier = await get_user_tier(_USER)
        assert tier == DEFAULT_TIER

    @pytest.mark.asyncio
    async def test_returns_default_when_tier_is_none(self):
        """Should return DEFAULT_TIER when subscription_tier is None."""
        mock_db = self._mock_user_db(subscription_tier=None)
        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db):
            tier = await get_user_tier(_USER)
        assert tier == DEFAULT_TIER

    @pytest.mark.asyncio
    async def test_returns_default_on_db_error(self):
        """Should fall back to DEFAULT_TIER when DB raises."""
        mock_db = self._mock_user_db(raises=Exception("DB down"))
        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db):
            tier = await get_user_tier(_USER)
        assert tier == DEFAULT_TIER

    @pytest.mark.asyncio
    async def test_db_error_is_not_cached(self):
        """Transient DB errors should NOT cache the default tier.

        Regression test: a transient DB failure previously cached DEFAULT_TIER
        for 5 minutes, incorrectly downgrading higher-tier users until expiry.
        """
        failing_db = self._mock_user_db(raises=Exception("DB down"))
        with patch("backend.copilot.rate_limit.user_db", return_value=failing_db):
            tier1 = await get_user_tier(_USER)
        assert tier1 == DEFAULT_TIER

        # Now DB recovers and returns PRO
        ok_db = self._mock_user_db(subscription_tier="PRO")
        with patch("backend.copilot.rate_limit.user_db", return_value=ok_db):
            tier2 = await get_user_tier(_USER)

        # Should get PRO now — the error result was not cached
        assert tier2 == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_returns_default_on_invalid_tier_value(self):
        """Should fall back to DEFAULT_TIER when stored value is invalid."""
        mock_db = self._mock_user_db(subscription_tier="invalid-tier")
        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db):
            tier = await get_user_tier(_USER)
        assert tier == DEFAULT_TIER

    @pytest.mark.asyncio
    async def test_user_not_found_is_not_cached(self):
        """Non-existent user should NOT cache DEFAULT_TIER.

        Regression test: when ``get_user_tier`` is called before a user record
        exists, the DEFAULT_TIER fallback must not be cached.  Otherwise, a
        newly created user with a higher tier (e.g. PRO) would receive the
        stale cached BASIC tier for up to 5 minutes.
        """
        # First call: user does not exist yet
        missing_db = self._mock_user_db(raises=Exception("not found"))
        with patch("backend.copilot.rate_limit.user_db", return_value=missing_db):
            tier1 = await get_user_tier(_USER)
        assert tier1 == DEFAULT_TIER

        # Second call: user now exists with PRO tier
        ok_db = self._mock_user_db(subscription_tier="PRO")
        with patch("backend.copilot.rate_limit.user_db", return_value=ok_db):
            tier2 = await get_user_tier(_USER)

        # Should get PRO — the not-found result was not cached
        assert tier2 == SubscriptionTier.PRO


# ---------------------------------------------------------------------------
# set_user_tier
# ---------------------------------------------------------------------------


class TestSetUserTier:
    @pytest.fixture(autouse=True)
    def _clear_tier_cache(self):
        """Clear the get_user_tier cache before each test."""
        get_user_tier.cache_clear()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_updates_db_and_invalidates_cache(self):
        """set_user_tier should persist to DB and invalidate the tier cache."""
        mock_prisma = AsyncMock()
        mock_prisma.update = AsyncMock(return_value=None)

        with patch(
            "backend.copilot.rate_limit.PrismaUser.prisma",
            return_value=mock_prisma,
        ):
            await set_user_tier(_USER, SubscriptionTier.PRO)

        mock_prisma.update.assert_awaited_once_with(
            where={"id": _USER},
            data={"subscriptionTier": "PRO"},
        )

    @pytest.mark.asyncio
    async def test_record_not_found_propagates(self):
        """RecordNotFoundError from Prisma should propagate to callers."""
        import prisma.errors

        mock_prisma = AsyncMock()
        mock_prisma.update = AsyncMock(
            side_effect=prisma.errors.RecordNotFoundError(
                {"error": "Record not found"}
            ),
        )

        with patch(
            "backend.copilot.rate_limit.PrismaUser.prisma",
            return_value=mock_prisma,
        ):
            with pytest.raises(prisma.errors.RecordNotFoundError):
                await set_user_tier(_USER, SubscriptionTier.ENTERPRISE)

    @pytest.mark.asyncio
    async def test_cache_invalidated_after_set(self):
        """After set_user_tier, get_user_tier should query DB again (not cache)."""
        # First, populate the cache with BUSINESS via user_db() mock
        mock_db_biz = AsyncMock()
        mock_user_biz = MagicMock()
        mock_user_biz.subscription_tier = "BUSINESS"
        mock_db_biz.get_user_by_id = AsyncMock(return_value=mock_user_biz)

        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db_biz):
            tier_before = await get_user_tier(_USER)
        assert tier_before == SubscriptionTier.BUSINESS

        # Now set tier to ENTERPRISE via PrismaUser.prisma (set_user_tier still
        # uses Prisma directly since it's only called from admin API where Prisma
        # is connected).
        mock_prisma_set = AsyncMock()
        mock_prisma_set.update = AsyncMock(return_value=None)

        with patch(
            "backend.copilot.rate_limit.PrismaUser.prisma",
            return_value=mock_prisma_set,
        ):
            await set_user_tier(_USER, SubscriptionTier.ENTERPRISE)

        # Now get_user_tier should hit DB again (cache was invalidated)
        mock_db_ent = AsyncMock()
        mock_user_ent = MagicMock()
        mock_user_ent.subscription_tier = "ENTERPRISE"
        mock_db_ent.get_user_by_id = AsyncMock(return_value=mock_user_ent)

        with patch("backend.copilot.rate_limit.user_db", return_value=mock_db_ent):
            tier_after = await get_user_tier(_USER)

        assert tier_after == SubscriptionTier.ENTERPRISE

    @pytest.mark.asyncio
    async def test_drift_check_swallows_launchdarkly_failure(self):
        """LaunchDarkly price-id lookup failures inside the drift check must
        never bubble up and 500 the admin tier write — the DB update is
        already committed by the time we check drift."""
        mock_prisma = AsyncMock()
        mock_prisma.update = AsyncMock(return_value=None)

        mock_user = MagicMock()
        mock_user.stripe_customer_id = "cus_abc"

        mock_sub = MagicMock()
        mock_sub.id = "sub_abc"
        mock_sub["items"].data = [MagicMock(price=MagicMock(id="price_mismatch"))]

        with (
            patch(
                "backend.copilot.rate_limit.PrismaUser.prisma",
                return_value=mock_prisma,
            ),
            patch(
                "backend.copilot.rate_limit.get_user_by_id",
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
            patch(
                "backend.data.credit._get_active_subscription",
                new_callable=AsyncMock,
                return_value=mock_sub,
            ),
            patch(
                "backend.data.credit.get_subscription_price_id",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LD SDK not initialized"),
            ),
        ):
            # Must NOT raise — drift check is best-effort diagnostic only.
            await set_user_tier(_USER, SubscriptionTier.PRO)

        mock_prisma.update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drift_check_timeout_is_bounded(self):
        """A Stripe call that stalls on the 80s SDK default must not block the
        admin tier write — set_user_tier wraps the drift check in a 5s timeout
        and logs + returns on TimeoutError."""
        import asyncio as _asyncio

        mock_prisma = AsyncMock()
        mock_prisma.update = AsyncMock(return_value=None)

        async def _never_returns(_user_id: str, _tier):
            await _asyncio.sleep(60)

        with (
            patch(
                "backend.copilot.rate_limit.PrismaUser.prisma",
                return_value=mock_prisma,
            ),
            patch(
                "backend.copilot.rate_limit._warn_if_stripe_subscription_drifts",
                side_effect=_never_returns,
            ),
            patch(
                "backend.copilot.rate_limit.asyncio.wait_for",
                new_callable=AsyncMock,
                side_effect=_asyncio.TimeoutError,
            ),
        ):
            await set_user_tier(_USER, SubscriptionTier.PRO)

        # Set_user_tier still completed — the drift timeout did not propagate.
        mock_prisma.update.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_global_rate_limits with tiers
# ---------------------------------------------------------------------------


class TestGetGlobalRateLimitsWithTiers:
    @pytest.fixture(autouse=True)
    def _clear_flag_cache(self):
        """Clear the LD flag caches between tests so patches don't leak."""
        _fetch_tier_multipliers_flag.cache_clear()  # type: ignore[attr-defined]
        _fetch_cost_limits_flag.cache_clear()  # type: ignore[attr-defined]

    @staticmethod
    def _ld_side_effect(daily: int, weekly: int):
        """Return an async side_effect that dispatches by flag_key.

        Returns the cost-limits JSON shape for ``copilot-cost-limits`` and
        the raw default for the tier-multipliers flag so existing tests
        continue to exercise the default multiplier map.
        """

        async def _side_effect(flag_key: str, _uid: str, default):
            if "cost-limits" in flag_key.lower():
                return {"daily": daily, "weekly": weekly}
            return default

        return _side_effect

    @pytest.mark.asyncio
    async def test_ld_override_applies_fractional_multiplier(self):
        """A fractional LD multiplier is applied and truncated back to int."""

        async def _ld(flag_key: str, _uid: str, default):
            if "cost-limits" in flag_key.lower():
                return {"daily": 1_000_000, "weekly": 5_000_000}
            if "tier-multipliers" in flag_key.lower():
                return {"PRO": 8.5}
            return default

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.PRO,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=_ld,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, 1_000_000, 5_000_000
            )

        assert tier == SubscriptionTier.PRO
        assert daily == 8_500_000  # 1_000_000 * 8.5
        assert weekly == 42_500_000  # 5_000_000 * 8.5
        # Both results are plain ints so microdollar math stays integer.
        assert isinstance(daily, int)
        assert isinstance(weekly, int)

    @pytest.mark.asyncio
    async def test_free_tier_no_multiplier(self):
        """Free tier should not change limits."""
        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BASIC,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(2_500_000, 12_500_000),
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, 2_500_000, 12_500_000
            )

        assert daily == 2_500_000
        assert weekly == 12_500_000
        assert tier == SubscriptionTier.BASIC

    @pytest.mark.asyncio
    async def test_pro_tier_5x_multiplier(self):
        """Pro tier should multiply limits by 5."""
        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.PRO,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(2_500_000, 12_500_000),
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, 2_500_000, 12_500_000
            )

        assert daily == 12_500_000
        assert weekly == 62_500_000
        assert tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_max_tier_20x_multiplier(self):
        """Max tier should multiply limits by 20 (self-service $320 tier)."""
        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.MAX,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(2_500_000, 12_500_000),
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, 2_500_000, 12_500_000
            )

        assert daily == 50_000_000
        assert weekly == 250_000_000
        assert tier == SubscriptionTier.MAX

    @pytest.mark.asyncio
    async def test_business_tier_60x_multiplier(self):
        """Business tier should multiply limits by 60 (matches Enterprise capacity)."""
        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BUSINESS,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(2_500_000, 12_500_000),
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, 2_500_000, 12_500_000
            )

        assert daily == 150_000_000
        assert weekly == 750_000_000
        assert tier == SubscriptionTier.BUSINESS

    @pytest.mark.asyncio
    async def test_enterprise_tier_60x_multiplier(self):
        """Enterprise tier should multiply limits by 60."""
        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.ENTERPRISE,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(2_500_000, 12_500_000),
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, 2_500_000, 12_500_000
            )

        assert daily == 150_000_000
        assert weekly == 750_000_000
        assert tier == SubscriptionTier.ENTERPRISE

    @pytest.mark.asyncio
    async def test_no_tier_with_payment_disabled_upgrades_to_basic(self):
        """Beta cohort regression test: a NO_TIER user passing through
        the route-level paywall (because ENABLE_PLATFORM_PAYMENT is off)
        must NOT then hit (0, 0) at check_rate_limit. The fallback to
        BASIC's multiplier (1.0) keeps daily/weekly intact so beta
        testers retain access without bypassing usage caps."""

        async def _ld(flag_key: str, _uid: str, default):
            if "cost-limits" in flag_key.lower():
                return {"daily": 1_000_000, "weekly": 5_000_000}
            return default

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new=AsyncMock(return_value=SubscriptionTier.NO_TIER),
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=_ld,
            ),
            patch(
                "backend.copilot.rate_limit.is_feature_enabled",
                new=AsyncMock(return_value=False),  # beta cohort
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, 1_000_000, 5_000_000
            )

        # BASIC's multiplier is 1.0 → limits arrive non-zero, not the
        # NO_TIER multiplier-collapsed (0, 0) that would slip past
        # check_rate_limit's pre-PR `> 0` short-circuit.
        assert daily == 1_000_000
        assert weekly == 5_000_000
        assert tier == SubscriptionTier.NO_TIER


# ---------------------------------------------------------------------------
# End-to-end: tier limits are respected by check_rate_limit
# ---------------------------------------------------------------------------


class TestTierLimitsRespected:
    """Verify that tier-adjusted limits from get_global_rate_limits flow
    correctly into check_rate_limit, so higher tiers allow more usage and
    lower tiers are blocked when they would exceed their allocation."""

    _BASE_DAILY = 2_500_000
    _BASE_WEEKLY = 12_500_000

    @pytest.fixture(autouse=True)
    def _clear_flag_cache(self):
        _fetch_tier_multipliers_flag.cache_clear()  # type: ignore[attr-defined]
        _fetch_cost_limits_flag.cache_clear()  # type: ignore[attr-defined]

    @staticmethod
    def _ld_side_effect(daily: int, weekly: int):

        async def _side_effect(flag_key: str, _uid: str, default):
            if "cost-limits" in flag_key.lower():
                return {"daily": daily, "weekly": weekly}
            return default

        return _side_effect

    @pytest.mark.asyncio
    async def test_pro_user_allowed_above_basic_limit(self):
        """A PRO user with usage above the BASIC limit should be allowed."""
        # Usage: 3M tokens (above BASIC limit of 2.5M, below PRO limit of 12.5M)
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["3000000", "3000000"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.PRO,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            # PRO: 5x multiplier
            assert daily == 12_500_000
            assert tier == SubscriptionTier.PRO
            # Should NOT raise — 3M < 12.5M
            await check_rate_limit(
                _USER, daily_cost_limit=daily, weekly_cost_limit=weekly
            )

    @pytest.mark.asyncio
    async def test_basic_user_blocked_at_basic_limit(self):
        """A BASIC user at or above the base limit should be blocked."""
        # Usage: 2.5M tokens (at BASIC limit of 2.5M)
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["2500000", "2500000"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BASIC,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            # BASIC: 1x multiplier
            assert daily == 2_500_000
            assert tier == SubscriptionTier.BASIC
            # Should raise — 2.5M >= 2.5M
            with pytest.raises(RateLimitExceeded):
                await check_rate_limit(
                    _USER, daily_cost_limit=daily, weekly_cost_limit=weekly
                )

    @pytest.mark.asyncio
    async def test_enterprise_user_has_highest_headroom(self):
        """An ENTERPRISE user should have 60x the base limit."""
        # Usage: 100M tokens (huge, but below ENTERPRISE daily of 150M)
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["100000000", "100000000"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.ENTERPRISE,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert daily == 150_000_000
            assert tier == SubscriptionTier.ENTERPRISE
            # Should NOT raise — 100M < 150M
            await check_rate_limit(
                _USER, daily_cost_limit=daily, weekly_cost_limit=weekly
            )


# ---------------------------------------------------------------------------
# reset_daily_usage
# ---------------------------------------------------------------------------


class TestResetDailyUsage:
    @pytest.mark.asyncio
    async def test_deletes_daily_key(self):
        mock_redis = AsyncMock()

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            result = await reset_daily_usage(_USER, daily_cost_limit=10000)

        assert result is True
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_reduces_weekly_usage_via_eval(self):
        """Weekly counter should be decremented via the atomic Lua script."""
        mock_redis = AsyncMock()

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await reset_daily_usage(_USER, daily_cost_limit=10000)

        # The Lua script handles both decrement and floor-to-zero in a single
        # call — no separate SET is expected for the clamp branch any more.
        # Pin the call shape so a regression that targets the wrong key or
        # delta (e.g. the daily key, or a sign-flip) fails loudly.
        mock_redis.eval.assert_called_once()
        eval_args = mock_redis.eval.call_args.args
        # eval(script, numkeys, KEYS[1], ARGV[1])
        assert eval_args[1] == 1
        assert eval_args[2] == _weekly_key(_USER)
        assert int(eval_args[3]) == 10000
        mock_redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_weekly_reduction_when_daily_limit_zero(self):
        """When daily_cost_limit is 0, weekly counter should not be touched."""
        mock_redis = AsyncMock()

        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            await reset_daily_usage(_USER, daily_cost_limit=0)

        mock_redis.delete.assert_called_once()
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_false_when_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis down"),
        ):
            result = await reset_daily_usage(_USER, daily_cost_limit=10000)

        assert result is False

    @pytest.mark.asyncio
    async def test_decr_counter_floor_zero_invokes_lua_script(self):
        """The atomic DECRBY+floor helper routes through redis.eval with the
        expected single-key, single-arg call shape."""
        from backend.copilot.rate_limit import (
            _DECR_FLOOR_ZERO_SCRIPT,
            _decr_counter_floor_zero,
        )

        mock_redis = AsyncMock()

        await _decr_counter_floor_zero(mock_redis, "weekly:user1", 42)

        mock_redis.eval.assert_called_once_with(
            _DECR_FLOOR_ZERO_SCRIPT, 1, "weekly:user1", 42
        )


# ---------------------------------------------------------------------------
# Tier-limit enforcement (integration-style)
# ---------------------------------------------------------------------------


class TestTierLimitsEnforced:
    """Verify that tier-multiplied limits are actually respected by
    ``check_rate_limit`` — i.e. that usage within the tier allowance passes
    and usage at/above the tier allowance is rejected."""

    _BASE_DAILY = 1_000_000
    _BASE_WEEKLY = 5_000_000

    @pytest.fixture(autouse=True)
    def _clear_flag_cache(self):
        _fetch_tier_multipliers_flag.cache_clear()  # type: ignore[attr-defined]
        _fetch_cost_limits_flag.cache_clear()  # type: ignore[attr-defined]

    @staticmethod
    def _ld_side_effect(daily: int, weekly: int):
        """Mock LD flag lookup returning the given raw limits."""

        async def _side_effect(flag_key: str, _uid: str, default):
            if "cost-limits" in flag_key.lower():
                return {"daily": daily, "weekly": weekly}
            return default

        return _side_effect

    @pytest.mark.asyncio
    async def test_pro_within_limit_allowed(self):
        """Usage under PRO daily limit should not raise."""
        pro_daily = int(self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.PRO])
        mock_redis = AsyncMock()
        # Simulate usage just under the PRO daily limit
        mock_redis.get = AsyncMock(side_effect=[str(pro_daily - 1), "0"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.PRO,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert tier == SubscriptionTier.PRO
            assert daily == pro_daily
            # Should not raise — usage is under the limit
            await check_rate_limit(_USER, daily, weekly)

    @pytest.mark.asyncio
    async def test_pro_at_limit_rejected(self):
        """Usage at exactly the PRO daily limit should raise."""
        pro_daily = int(self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.PRO])
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[str(pro_daily), "0"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.PRO,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(_USER, daily, weekly)
            assert exc_info.value.window == "daily"

    @pytest.mark.asyncio
    async def test_business_higher_limit_allows_pro_overflow(self):
        """Usage exceeding PRO but under BUSINESS should pass for BUSINESS."""
        pro_daily = int(self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.PRO])
        biz_daily = int(self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.BUSINESS])
        # Usage between PRO and BUSINESS limits
        usage = pro_daily + 1_000_000
        assert usage < biz_daily, "test sanity: usage must be under BUSINESS limit"

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[str(usage), "0"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BUSINESS,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert tier == SubscriptionTier.BUSINESS
            assert daily == biz_daily
            # Should not raise — BUSINESS tier can handle this
            await check_rate_limit(_USER, daily, weekly)

    @pytest.mark.asyncio
    async def test_weekly_limit_enforced_for_tier(self):
        """Weekly limit should also be tier-multiplied and enforced."""
        pro_weekly = int(self._BASE_WEEKLY * TIER_MULTIPLIERS[SubscriptionTier.PRO])
        mock_redis = AsyncMock()
        # Daily usage fine, weekly at limit
        mock_redis.get = AsyncMock(side_effect=["0", str(pro_weekly)])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.PRO,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(_USER, daily, weekly)
            assert exc_info.value.window == "weekly"

    @pytest.mark.asyncio
    async def test_free_tier_base_limit_enforced(self):
        """Free tier (1x multiplier) should enforce the base limit exactly."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[str(self._BASE_DAILY), "0"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BASIC,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert daily == self._BASE_DAILY  # 1x multiplier
            with pytest.raises(RateLimitExceeded):
                await check_rate_limit(_USER, daily, weekly)

    @pytest.mark.asyncio
    async def test_basic_tier_cannot_bypass_pro_limit(self):
        """A BASIC-tier user whose usage is within PRO limits but over BASIC
        limits must still be rejected.

        Negative test: ensures the tier multiplier is applied *before* the
        rate-limit check, so a lower-tier user cannot 'bypass' limits that
        would be acceptable for a higher tier.
        """
        basic_daily = int(self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.BASIC])
        pro_daily = int(self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.PRO])
        # Usage above BASIC limit but below PRO limit
        usage = basic_daily + 500_000
        assert usage < pro_daily, "test sanity: usage must be under PRO limit"

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[str(usage), "0"])

        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BASIC,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert tier == SubscriptionTier.BASIC
            assert daily == basic_daily  # 1x, not 5x
            with pytest.raises(RateLimitExceeded) as exc_info:
                await check_rate_limit(_USER, daily, weekly)
            assert exc_info.value.window == "daily"

    @pytest.mark.asyncio
    async def test_tier_change_updates_effective_limits(self):
        """After upgrading from BASIC to BUSINESS, the effective limits must
        increase accordingly.

        Verifies that the tier multiplier is correctly applied after a tier
        change, and that usage that was over the BASIC limit is within the new
        BUSINESS limit.
        """
        basic_daily = int(self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.BASIC])
        biz_daily = int(self._BASE_DAILY * TIER_MULTIPLIERS[SubscriptionTier.BUSINESS])
        # Usage above BASIC limit but below BUSINESS limit
        usage = basic_daily + 500_000
        assert usage < biz_daily, "test sanity: usage must be under BUSINESS limit"

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=[str(usage), "0"])

        # Simulate the user having been upgraded to BUSINESS
        with (
            patch(
                "backend.copilot.rate_limit.get_user_tier",
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BUSINESS,
            ),
            patch(
                "backend.copilot.rate_limit.get_feature_flag_value",
                side_effect=self._ld_side_effect(self._BASE_DAILY, self._BASE_WEEKLY),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            daily, weekly, tier = await get_global_rate_limits(
                _USER, self._BASE_DAILY, self._BASE_WEEKLY
            )
            assert tier == SubscriptionTier.BUSINESS
            assert daily == biz_daily  # 60x
            # Should NOT raise — usage is within the BUSINESS tier allowance
            await check_rate_limit(_USER, daily, weekly)


# ---------------------------------------------------------------------------
# Private key/reset helpers
# ---------------------------------------------------------------------------


class TestKeyHelpers:
    def test_daily_key_format(self):
        now = datetime(2026, 4, 3, 12, 0, 0, tzinfo=UTC)
        key = _daily_key("user-1", now=now)
        assert "daily" in key
        assert "user-1" in key
        assert "2026-04-03" in key

    def test_daily_key_defaults_to_now(self):
        key = _daily_key("user-1")
        assert "daily" in key
        assert "user-1" in key

    def test_weekly_key_format(self):
        now = datetime(2026, 4, 3, 12, 0, 0, tzinfo=UTC)
        key = _weekly_key("user-1", now=now)
        assert "weekly" in key
        assert "user-1" in key
        assert "2026-W" in key

    def test_weekly_key_defaults_to_now(self):
        key = _weekly_key("user-1")
        assert "weekly" in key

    def test_daily_reset_time_is_next_midnight(self):
        now = datetime(2026, 4, 3, 15, 30, 0, tzinfo=UTC)
        reset = _daily_reset_time(now=now)
        assert reset == datetime(2026, 4, 4, 0, 0, 0, tzinfo=UTC)

    def test_daily_reset_time_defaults_to_now(self):
        reset = _daily_reset_time()
        assert reset.hour == 0
        assert reset.minute == 0

    def test_weekly_reset_time_is_next_monday(self):
        # 2026-04-03 is a Friday
        now = datetime(2026, 4, 3, 15, 30, 0, tzinfo=UTC)
        reset = _weekly_reset_time(now=now)
        assert reset.weekday() == 0  # Monday
        assert reset == datetime(2026, 4, 6, 0, 0, 0, tzinfo=UTC)

    def test_weekly_reset_time_defaults_to_now(self):
        reset = _weekly_reset_time()
        assert reset.weekday() == 0  # Monday


# ---------------------------------------------------------------------------
# acquire_reset_lock / release_reset_lock
# ---------------------------------------------------------------------------


class TestResetLock:
    @pytest.mark.asyncio
    async def test_acquire_lock_success(self):
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            result = await acquire_reset_lock("user-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_lock_already_held(self):
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=False)
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            result = await acquire_reset_lock("user-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_lock_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=RedisError("down"),
        ):
            result = await acquire_reset_lock("user-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_release_lock_success(self):
        mock_redis = AsyncMock()
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            await release_reset_lock("user-1")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_lock_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=RedisError("down"),
        ):
            # Should not raise
            await release_reset_lock("user-1")


# ---------------------------------------------------------------------------
# get_daily_reset_count / increment_daily_reset_count
# ---------------------------------------------------------------------------


class TestDailyResetCount:
    @pytest.mark.asyncio
    async def test_get_count_returns_value(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="3")
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            count = await get_daily_reset_count("user-1")
        assert count == 3

    @pytest.mark.asyncio
    async def test_get_count_returns_zero_when_no_key(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            count = await get_daily_reset_count("user-1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_count_returns_none_when_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=RedisError("down"),
        ):
            count = await get_daily_reset_count("user-1")
        assert count is None

    @pytest.mark.asyncio
    async def test_increment_count(self):
        mock_pipe = MagicMock()
        mock_pipe.incr = MagicMock()
        mock_pipe.expire = MagicMock()
        mock_pipe.execute = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)

        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            await increment_daily_reset_count("user-1")
        mock_pipe.incr.assert_called_once()
        mock_pipe.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_increment_count_redis_unavailable(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=RedisError("down"),
        ):
            # Should not raise
            await increment_daily_reset_count("user-1")


# ---------------------------------------------------------------------------
# reset_user_usage
# ---------------------------------------------------------------------------


class TestResetUserUsage:
    @pytest.mark.asyncio
    async def test_resets_daily_key(self):
        mock_redis = AsyncMock()
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            await reset_user_usage("user-1")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_resets_daily_and_weekly(self):
        mock_redis = AsyncMock()
        with patch(
            "backend.copilot.rate_limit.get_redis_async", return_value=mock_redis
        ):
            await reset_user_usage("user-1", reset_weekly=True)
        # Daily and weekly keys hash to different cluster slots, so they are
        # deleted via two separate DELETE calls (not a single multi-key one).
        assert mock_redis.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_on_redis_failure(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=RedisError("down"),
        ):
            with pytest.raises(RedisError):
                await reset_user_usage("user-1")


class TestWorkspaceStorageLimits:
    """Tests for tier-based workspace storage limits."""

    @pytest.fixture(autouse=True)
    def _clear_flag_cache(self):
        _fetch_workspace_storage_limits_flag.cache_clear()  # type: ignore[attr-defined]

    def test_every_subscription_tier_has_storage_limit(self):
        """Adding a new SubscriptionTier without a storage limit should fail."""
        for tier in SubscriptionTier:
            assert tier in _DEFAULT_TIER_WORKSPACE_STORAGE_MB, (
                f"SubscriptionTier.{tier.name} has no entry in "
                f"_DEFAULT_TIER_WORKSPACE_STORAGE_MB — add one"
            )
            assert _DEFAULT_TIER_WORKSPACE_STORAGE_MB[tier] > 0

    def test_every_subscription_tier_has_rate_limit_multiplier(self):
        """Adding a new SubscriptionTier without a rate limit multiplier should fail."""
        for tier in SubscriptionTier:
            assert tier in _DEFAULT_TIER_MULTIPLIERS, (
                f"SubscriptionTier.{tier.name} has no entry in "
                f"_DEFAULT_TIER_MULTIPLIERS — add one"
            )
            if tier == SubscriptionTier.NO_TIER:
                assert _DEFAULT_TIER_MULTIPLIERS[tier] == 0.0
            else:
                assert _DEFAULT_TIER_MULTIPLIERS[tier] > 0

    @pytest.mark.parametrize(
        "tier,expected_mb",
        [
            (SubscriptionTier.NO_TIER, 250),
            (SubscriptionTier.BASIC, 250),
            (SubscriptionTier.PRO, 1024),
            (SubscriptionTier.MAX, 5 * 1024),
            (SubscriptionTier.BUSINESS, 15 * 1024),
            (SubscriptionTier.ENTERPRISE, 15 * 1024),
        ],
    )
    def test_tier_workspace_storage_mapping_covers_all_tiers(self, tier, expected_mb):
        """Every tier has an explicit storage limit in the mapping."""
        assert tier in _DEFAULT_TIER_WORKSPACE_STORAGE_MB
        assert _DEFAULT_TIER_WORKSPACE_STORAGE_MB[tier] == expected_mb

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tier,expected_bytes",
        [
            (SubscriptionTier.NO_TIER, 250 * 1024 * 1024),
            (SubscriptionTier.BASIC, 250 * 1024 * 1024),
            (SubscriptionTier.PRO, 1024 * 1024 * 1024),
            (SubscriptionTier.MAX, 5 * 1024 * 1024 * 1024),
            (SubscriptionTier.BUSINESS, 15 * 1024 * 1024 * 1024),
            (SubscriptionTier.ENTERPRISE, 15 * 1024 * 1024 * 1024),
        ],
    )
    async def test_get_workspace_storage_limit_bytes_per_tier(
        self, tier, expected_bytes
    ):
        """get_workspace_storage_limit_bytes returns correct bytes for each tier."""
        with patch(
            "backend.copilot.rate_limit.get_user_tier",
            return_value=tier,
        ):
            result = await get_workspace_storage_limit_bytes("user-1")
        assert result == expected_bytes

    @pytest.mark.asyncio
    async def test_get_workspace_storage_limit_bytes_defaults_to_default_tier_on_unknown(
        self,
    ):
        """Unknown tier falls back to the default tier limit."""
        with patch(
            "backend.copilot.rate_limit.get_user_tier",
            return_value="UNKNOWN_TIER",
        ):
            result = await get_workspace_storage_limit_bytes("user-1")
        assert result == 250 * 1024 * 1024


# _warn_if_stripe_subscription_drifts — yearly billing
# ---------------------------------------------------------------------------


class TestWarnIfStripeSubscriptionDriftsYearly:
    @pytest.mark.asyncio
    async def test_yearly_subscription_does_not_log_drift(self, caplog):
        """A user on the yearly price for their admin-set tier must NOT trigger
        a drift warning — the drift check now matches either monthly OR yearly."""
        from backend.copilot.rate_limit import _warn_if_stripe_subscription_drifts

        mock_user = MagicMock()
        mock_user.stripe_customer_id = "cus_yearly"

        mock_sub = MagicMock()
        mock_sub.id = "sub_yearly"
        mock_sub["items"].data = [MagicMock(price=MagicMock(id="price_pro_yearly"))]

        async def price_lookup(tier, billing_cycle="monthly"):
            if tier == SubscriptionTier.PRO and billing_cycle == "monthly":
                return "price_pro_monthly"
            if tier == SubscriptionTier.PRO and billing_cycle == "yearly":
                return "price_pro_yearly"
            return None

        with (
            patch(
                "backend.copilot.rate_limit.get_user_by_id",
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
            patch(
                "backend.data.credit._get_active_subscription",
                new_callable=AsyncMock,
                return_value=mock_sub,
            ),
            patch(
                "backend.data.credit.get_subscription_price_id",
                side_effect=price_lookup,
            ),
        ):
            with caplog.at_level("WARNING", logger="backend.copilot.rate_limit"):
                await _warn_if_stripe_subscription_drifts(_USER, SubscriptionTier.PRO)

        drift_records = [
            r for r in caplog.records if "will drift from Stripe" in r.getMessage()
        ]
        assert drift_records == []

    @pytest.mark.asyncio
    async def test_genuine_drift_still_logs(self, caplog):
        """When the Stripe price doesn't match either monthly or yearly for the
        admin-set tier, the drift warning still fires."""
        from backend.copilot.rate_limit import _warn_if_stripe_subscription_drifts

        mock_user = MagicMock()
        mock_user.stripe_customer_id = "cus_drift"

        mock_sub = MagicMock()
        mock_sub.id = "sub_drift"
        mock_sub["items"].data = [MagicMock(price=MagicMock(id="price_max_yearly"))]

        async def price_lookup(tier, billing_cycle="monthly"):
            if tier == SubscriptionTier.PRO and billing_cycle == "monthly":
                return "price_pro_monthly"
            if tier == SubscriptionTier.PRO and billing_cycle == "yearly":
                return "price_pro_yearly"
            return None

        with (
            patch(
                "backend.copilot.rate_limit.get_user_by_id",
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
            patch(
                "backend.data.credit._get_active_subscription",
                new_callable=AsyncMock,
                return_value=mock_sub,
            ),
            patch(
                "backend.data.credit.get_subscription_price_id",
                side_effect=price_lookup,
            ),
        ):
            with caplog.at_level("WARNING", logger="backend.copilot.rate_limit"):
                await _warn_if_stripe_subscription_drifts(_USER, SubscriptionTier.PRO)

        drift_records = [
            r for r in caplog.records if "will drift from Stripe" in r.getMessage()
        ]
        assert len(drift_records) == 1


# ---------------------------------------------------------------------------
# get_remaining_usd_budget
# ---------------------------------------------------------------------------


class TestGetRemainingUsdBudget:
    @pytest.mark.asyncio
    async def test_zero_limits_return_floor_not_unlimited(self):
        """With no real-world unlimited tier, both limits at 0 means
        "no spend allowed" — remaining is 0 → floor_usd is returned."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["0", "0"])
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            result = await get_remaining_usd_budget(
                _USER, daily_cost_limit=0, weekly_cost_limit=0, floor_usd=0.5
            )
        assert result == 0.5

    @pytest.mark.asyncio
    async def test_smaller_of_daily_and_weekly_remaining(self):
        # daily=$10 used $3 → $7 remaining.  weekly=$50 used $48 → $2 remaining.
        # Returns the smaller (weekly $2).
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["3000000", "48000000"])
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            result = await get_remaining_usd_budget(
                _USER,
                daily_cost_limit=10_000_000,
                weekly_cost_limit=50_000_000,
            )
        assert result == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_floor_applies_when_user_at_or_over_cap(self):
        # daily=$5 used $5 → 0 remaining → floor returned. Weekly cap kept
        # well above usage so it doesn't drive the result.
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["5000000", "0"])
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            result = await get_remaining_usd_budget(
                _USER,
                daily_cost_limit=5_000_000,
                weekly_cost_limit=50_000_000,
                floor_usd=0.5,
            )
        assert result == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_returns_floor_on_redis_error(self):
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=RedisError("boom"),
        ):
            result = await get_remaining_usd_budget(
                _USER,
                daily_cost_limit=10_000_000,
                weekly_cost_limit=50_000_000,
                floor_usd=0.5,
            )
        assert result == 0.5

    @pytest.mark.asyncio
    async def test_daily_drives_when_weekly_is_loose(self):
        """Both limits constrain; whichever leaves less budget wins. Here a
        tight daily ($10 - $1 = $9) wins over a loose weekly."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["1000000", None])
        with patch(
            "backend.copilot.rate_limit.get_redis_async",
            return_value=mock_redis,
        ):
            result = await get_remaining_usd_budget(
                _USER,
                daily_cost_limit=10_000_000,
                weekly_cost_limit=1_000_000_000,
            )
        assert result == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# build_budget_ctx
# ---------------------------------------------------------------------------


class TestBuildBudgetCtx:
    """The helper combines ``get_global_rate_limits`` + ``get_remaining_usd_budget``
    into a single call so callers don't have to compose them by hand on
    every turn, and returns the *inner* text only — ``inject_user_context``
    wraps it in the ``<budget_context>`` tag."""

    @pytest.mark.asyncio
    async def test_returns_empty_without_user_id(self):
        result = await build_budget_ctx(
            user_id=None,
            default_daily_cost_limit=10_000_000,
            default_weekly_cost_limit=0,
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_when_remaining_is_zero(self):
        """No real-world unlimited tier — when the multiplier-collapse path
        returns (0, 0, NO_TIER) the user has $0 remaining → empty hint
        (the paywall dep should already have rejected this turn at 402;
        we just defend against drift)."""
        with patch(
            "backend.copilot.rate_limit.get_global_rate_limits",
            new=AsyncMock(return_value=(0, 0, DEFAULT_TIER)),
        ):
            result = await build_budget_ctx(
                user_id=_USER,
                default_daily_cost_limit=10_000_000,
                default_weekly_cost_limit=50_000_000,
            )
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_inner_text_with_remaining_in_dollars(self):
        # daily=$10 used $4.50 → $5.50 daily remaining; weekly cap kept
        # well above usage so daily drives the result.
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["4500000", "0"])
        with (
            patch(
                "backend.copilot.rate_limit.get_global_rate_limits",
                new=AsyncMock(return_value=(10_000_000, 1_000_000_000, DEFAULT_TIER)),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            block = await build_budget_ctx(
                user_id=_USER,
                default_daily_cost_limit=10_000_000,
                default_weekly_cost_limit=1_000_000_000,
            )
        # No tag wrap — that's inject_user_context's job.
        assert "<budget_context>" not in block
        assert "</budget_context>" not in block
        assert "$5.50" in block

    @pytest.mark.asyncio
    async def test_returns_empty_on_redis_brownout(self):
        """Redis brown-out → empty string, not a misleading ``$0.00`` hint.
        Pre-turn rate-limit gate has already failed closed at 503 in this
        case, so the model would never see the hint anyway."""
        with (
            patch(
                "backend.copilot.rate_limit.get_global_rate_limits",
                new=AsyncMock(return_value=(10_000_000, 1_000_000_000, DEFAULT_TIER)),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                side_effect=RedisError("down"),
            ),
        ):
            block = await build_budget_ctx(
                user_id=_USER,
                default_daily_cost_limit=10_000_000,
                default_weekly_cost_limit=1_000_000_000,
            )
        assert block == ""

    @pytest.mark.asyncio
    async def test_returns_empty_when_at_cap(self):
        """At-or-over cap → no hint (the pre-turn gate has already
        rejected this turn at 429; we only emit a hint when there's
        actual headroom to communicate)."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=["10000000", "0"])
        with (
            patch(
                "backend.copilot.rate_limit.get_global_rate_limits",
                new=AsyncMock(return_value=(10_000_000, 1_000_000_000, DEFAULT_TIER)),
            ),
            patch(
                "backend.copilot.rate_limit.get_redis_async",
                return_value=mock_redis,
            ),
        ):
            block = await build_budget_ctx(
                user_id=_USER,
                default_daily_cost_limit=10_000_000,
                default_weekly_cost_limit=1_000_000_000,
            )
        assert block == ""
