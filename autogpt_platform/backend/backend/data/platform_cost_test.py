"""Unit tests for helpers and async functions in platform_cost module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma import Json

from backend.util.json import SafeJson

from .platform_cost import (
    PlatformCostEntry,
    _build_prisma_where,
    _build_raw_where,
    _build_where,
    _mask_email,
    get_platform_cost_dashboard,
    get_platform_cost_logs,
    get_platform_cost_logs_for_export,
    log_platform_cost,
    log_platform_cost_safe,
    usd_to_microdollars,
)


class TestUsdToMicrodollars:
    def test_none_returns_none(self):
        assert usd_to_microdollars(None) is None

    def test_converts_usd_to_microdollars(self):
        assert usd_to_microdollars(1.0) == 1_000_000

    def test_fractional_usd(self):
        assert usd_to_microdollars(0.0042) == 4200

    def test_zero_returns_zero(self):
        assert usd_to_microdollars(0.0) == 0

    def test_positive_value(self):
        assert usd_to_microdollars(0.001) == 1000

    def test_large_value(self):
        assert usd_to_microdollars(1.0) == 1_000_000


class TestMaskEmail:
    def test_typical_email(self):
        assert _mask_email("user@example.com") == "us***@example.com"

    def test_short_local_part(self):
        assert _mask_email("a@b.com") == "a***@b.com"

    def test_none_returns_none(self):
        assert _mask_email(None) is None

    def test_empty_string_returns_empty(self):
        assert _mask_email("") == ""

    def test_no_at_sign_returns_stars(self):
        assert _mask_email("notanemail") == "***"

    def test_two_char_local(self):
        assert _mask_email("ab@domain.org") == "ab***@domain.org"


class TestBuildWhere:
    def test_no_filters_returns_true(self):
        sql, params = _build_where(None, None, None, None)
        assert sql == "TRUE"
        assert params == []

    def test_start_only(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sql, params = _build_where(dt, None, None, None)
        assert '"createdAt" >= $1::timestamptz' in sql
        assert params == [dt]

    def test_end_only(self):
        dt = datetime(2026, 6, 1, tzinfo=timezone.utc)
        sql, params = _build_where(None, dt, None, None)
        assert '"createdAt" <= $1::timestamptz' in sql
        assert params == [dt]

    def test_provider_only(self):
        # Provider names are normalized to lowercase at write time, so the
        # filter uses a plain equality check. The input is also lowercased so
        # "OpenAI" and "openai" both match stored rows.
        sql, params = _build_where(None, None, "OpenAI", None)
        assert '"provider" = $1' in sql
        assert params == ["openai"]

    def test_user_id_only(self):
        sql, params = _build_where(None, None, None, "user-123")
        assert '"userId" = $1' in sql
        assert params == ["user-123"]

    def test_all_filters(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        sql, params = _build_where(start, end, "Anthropic", "u1")
        assert "$1" in sql
        assert "$2" in sql
        assert "$3" in sql
        assert "$4" in sql
        assert len(params) == 4
        # Provider is lowercased at filter time to match stored lowercase values.
        assert params == [start, end, "anthropic", "u1"]

    def test_table_alias(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sql, params = _build_where(dt, None, None, None, table_alias="p")
        assert 'p."createdAt"' in sql
        assert params == [dt]

    def test_clauses_joined_with_and(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        sql, _ = _build_where(start, end, None, None)
        assert " AND " in sql

    def test_model_only(self):
        sql, params = _build_where(None, None, None, None, model="gpt-4")
        assert '"model" = $1' in sql
        assert params == ["gpt-4"]

    def test_block_name_only(self):
        sql, params = _build_where(None, None, None, None, block_name="LLMBlock")
        assert 'LOWER("blockName") = LOWER($1)' in sql
        assert params == ["LLMBlock"]

    def test_tracking_type_only(self):
        sql, params = _build_where(None, None, None, None, tracking_type="tokens")
        assert '"trackingType" = $1' in sql
        assert params == ["tokens"]

    def test_all_new_filters_combined(self):
        sql, params = _build_where(
            None,
            None,
            None,
            None,
            model="gpt-4",
            block_name="LLM",
            tracking_type="tokens",
        )
        assert len(params) == 3
        assert params[0] == "gpt-4"
        assert params[1] == "LLM"
        assert params[2] == "tokens"

    def test_new_filters_with_alias(self):
        sql, params = _build_where(
            None,
            None,
            None,
            None,
            table_alias="p",
            model="gpt-4",
            block_name="MyBlock",
            tracking_type="cost_usd",
        )
        assert 'p."model" = $1' in sql
        assert 'LOWER(p."blockName") = LOWER($2)' in sql
        assert 'p."trackingType" = $3' in sql


class TestBuildPrismaWhere:
    def test_both_start_and_end(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        where = _build_prisma_where(start, end, None, None)
        assert where["createdAt"] == {"gte": start, "lte": end}

    def test_end_only(self):
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        where = _build_prisma_where(None, end, None, None)
        assert where["createdAt"] == {"lte": end}

    def test_start_only(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        where = _build_prisma_where(start, None, None, None)
        assert where["createdAt"] == {"gte": start}

    def test_no_filters(self):
        where = _build_prisma_where(None, None, None, None)
        assert "createdAt" not in where

    def test_provider_lowercased(self):
        where = _build_prisma_where(None, None, "OpenAI", None)
        assert where["provider"] == "openai"

    def test_model_filter(self):
        where = _build_prisma_where(None, None, None, None, model="gpt-4")
        assert where["model"] == "gpt-4"

    def test_block_name_case_insensitive(self):
        where = _build_prisma_where(None, None, None, None, block_name="LLMBlock")
        assert where["blockName"] == {"equals": "LLMBlock", "mode": "insensitive"}

    def test_tracking_type(self):
        where = _build_prisma_where(None, None, None, None, tracking_type="tokens")
        assert where["trackingType"] == "tokens"

    def test_graph_exec_id_filter(self):
        where = _build_prisma_where(None, None, None, None, graph_exec_id="exec-123")
        assert where["graphExecId"] == "exec-123"

    def test_graph_exec_id_none_not_included(self):
        where = _build_prisma_where(None, None, None, None, graph_exec_id=None)
        assert "graphExecId" not in where


class TestBuildRawWhere:
    def test_end_filter(self):
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        sql, params = _build_raw_where(None, end, None, None)
        assert '"createdAt" <= $2::timestamptz' in sql
        assert end in params

    def test_model_filter(self):
        sql, params = _build_raw_where(None, None, None, None, model="gpt-4")
        assert '"model" = $' in sql
        assert "gpt-4" in params

    def test_block_name_filter(self):
        sql, params = _build_raw_where(None, None, None, None, block_name="LLMBlock")
        assert 'LOWER("blockName") = LOWER($' in sql
        assert "LLMBlock" in params

    def test_all_filters_combined(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        sql, params = _build_raw_where(
            start, end, "anthropic", "u1", model="claude-3", block_name="LLM"
        )
        # trackingType (default), start, end, provider, user_id, model, block_name
        assert len(params) == 7
        assert "anthropic" in params
        assert "u1" in params
        assert "claude-3" in params
        assert "LLM" in params

    def test_default_tracking_type_is_cost_usd(self):
        sql, params = _build_raw_where(None, None, None, None)
        assert '"trackingType" = $1' in sql
        assert params[0] == "cost_usd"

    def test_explicit_tracking_type_overrides_default(self):
        sql, params = _build_raw_where(None, None, None, None, tracking_type="tokens")
        assert params[0] == "tokens"

    def test_graph_exec_id_filter(self):
        sql, params = _build_raw_where(None, None, None, None, graph_exec_id="exec-abc")
        assert '"graphExecId" = $' in sql
        assert "exec-abc" in params

    def test_graph_exec_id_not_included_when_none(self):
        sql, params = _build_raw_where(None, None, None, None)
        assert "graphExecId" not in sql


def _make_entry(**overrides: object) -> PlatformCostEntry:
    return PlatformCostEntry.model_validate(
        {
            "user_id": "user-1",
            "block_id": "block-1",
            "block_name": "TestBlock",
            "provider": "openai",
            "credential_id": "cred-1",
            **overrides,
        }
    )


class TestLogPlatformCost:
    @pytest.mark.asyncio
    async def test_creates_prisma_record(self):
        mock_create = AsyncMock()
        with patch("backend.data.platform_cost.PrismaLog.prisma") as mock_prisma:
            mock_prisma.return_value.create = mock_create
            entry = _make_entry(
                input_tokens=100,
                output_tokens=50,
                cost_microdollars=5000,
                model="gpt-4",
                metadata={"key": "val"},
            )
            await log_platform_cost(entry)
        mock_create.assert_awaited_once()
        data = mock_create.call_args[1]["data"]
        assert data["userId"] == "user-1"
        assert data["blockName"] == "TestBlock"
        assert data["provider"] == "openai"
        # metadata must be wrapped in SafeJson (a prisma.Json subclass), not a plain dict
        assert isinstance(data["metadata"], Json)

    @pytest.mark.asyncio
    async def test_metadata_none_passes_none(self):
        mock_create = AsyncMock()
        with patch("backend.data.platform_cost.PrismaLog.prisma") as mock_prisma:
            mock_prisma.return_value.create = mock_create
            entry = _make_entry(metadata=None)
            await log_platform_cost(entry)
        data = mock_create.call_args[1]["data"]
        # None falls back to SafeJson({}) so Prisma always gets a valid Json value
        assert isinstance(data["metadata"], Json)
        assert data["metadata"] == SafeJson({})


class TestLogPlatformCostSafe:
    @pytest.mark.asyncio
    async def test_does_not_raise_on_error(self):
        with patch("backend.data.platform_cost.PrismaLog.prisma") as mock_prisma:
            mock_prisma.return_value.create = AsyncMock(
                side_effect=RuntimeError("DB down")
            )
            entry = _make_entry()
            await log_platform_cost_safe(entry)

    @pytest.mark.asyncio
    async def test_succeeds_when_no_error(self):
        mock_create = AsyncMock()
        with patch("backend.data.platform_cost.PrismaLog.prisma") as mock_prisma:
            mock_prisma.return_value.create = mock_create
            entry = _make_entry()
            await log_platform_cost_safe(entry)
        mock_create.assert_awaited_once()


def _make_group_by_row(
    provider: str = "openai",
    tracking_type: str | None = "tokens",
    model: str | None = None,
    cost: int = 5000,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    duration: float = 10.5,
    tracking_amount: float = 0.0,
    count: int = 3,
    user_id: str | None = None,
) -> dict:
    row: dict = {
        "_sum": {
            "costMicrodollars": cost,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "cacheReadTokens": cache_read_tokens,
            "cacheCreationTokens": cache_creation_tokens,
            "duration": duration,
            "trackingAmount": tracking_amount,
        },
        "_count": {"_all": count},
    }
    if user_id is not None:
        row["userId"] = user_id
    else:
        row["provider"] = provider
        row["trackingType"] = tracking_type
        row["model"] = model
    return row


class TestGetPlatformCostDashboard:
    def setup_method(self):
        # @cached stores results in-process; clear between tests to avoid bleed.
        get_platform_cost_dashboard.cache_clear()

    @pytest.mark.asyncio
    async def test_returns_dashboard_with_data(self):
        provider_row = _make_group_by_row(
            provider="openai",
            tracking_type="tokens",
            cost=5000,
            input_tokens=1000,
            output_tokens=500,
            duration=10.5,
            count=3,
        )
        user_row = _make_group_by_row(user_id="u1", cost=5000, count=3)

        mock_user = MagicMock()
        mock_user.id = "u1"
        mock_user.email = "a@b.com"

        mock_actions = MagicMock()
        mock_actions.group_by = AsyncMock(
            side_effect=[
                [provider_row],  # by_provider
                [user_row],  # by_user
                [],  # by_user_tracking_groups (no cost_usd rows for this user)
                [{"userId": "u1"}],  # distinct users
                [provider_row],  # total agg (tracking_type=None → same as unfiltered)
            ]
        )
        mock_actions.find_many = AsyncMock(return_value=[mock_user])

        with (
            patch(
                "backend.data.platform_cost.PrismaLog.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.PrismaUser.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.query_raw_with_schema",
                new_callable=AsyncMock,
                side_effect=[
                    [{"p50": 1000, "p75": 2000, "p95": 4000, "p99": 5000}],
                    [{"bucket": "$0-0.50", "count": 3}],
                ],
            ),
        ):
            dashboard = await get_platform_cost_dashboard()

        assert dashboard.total_cost_microdollars == 5000
        assert dashboard.total_requests == 3
        assert dashboard.total_users == 1
        assert len(dashboard.by_provider) == 1
        assert dashboard.by_provider[0].provider == "openai"
        assert dashboard.by_provider[0].tracking_type == "tokens"
        assert dashboard.by_provider[0].total_duration_seconds == 10.5
        assert len(dashboard.by_user) == 1
        assert dashboard.by_user[0].email == "a***@b.com"
        assert dashboard.cost_p50_microdollars == 1000
        assert dashboard.cost_p75_microdollars == 2000
        assert dashboard.cost_p95_microdollars == 4000
        assert dashboard.cost_p99_microdollars == 5000
        assert len(dashboard.cost_buckets) == 1
        # total_input/output_tokens come from total_agg_no_tracking_type_groups
        # (provider_row has 1000/500)
        assert dashboard.total_input_tokens == 1000
        assert dashboard.total_output_tokens == 500
        # Token averages must use token_bearing_requests (3) not cost_bearing (0)
        assert dashboard.avg_input_tokens_per_request == pytest.approx(1000 / 3)
        assert dashboard.avg_output_tokens_per_request == pytest.approx(500 / 3)
        # No cost_usd rows in total_agg → avg_cost should be 0
        assert dashboard.avg_cost_microdollars_per_request == 0.0

    @pytest.mark.asyncio
    async def test_cost_bearing_request_count_nonzero_when_filtering_by_tokens(self):
        """When filtering by tracking_type='tokens', cost_bearing_request_count
        must still reflect cost_usd rows because by_user_tracking_groups is
        queried without the tracking_type constraint."""
        # total_agg only has a tokens row (because of the tracking_type filter)
        total_row = _make_group_by_row(
            provider="openai", tracking_type="tokens", cost=0, count=5
        )
        # by_user_tracking_groups returns BOTH rows (no tracking_type filter)
        user_tracking_cost_usd_row = {
            "_count": {"_all": 7},
            "userId": "u1",
            "trackingType": "cost_usd",
        }
        user_tracking_tokens_row = {
            "_count": {"_all": 5},
            "userId": "u1",
            "trackingType": "tokens",
        }

        mock_actions = MagicMock()
        mock_actions.group_by = AsyncMock(
            side_effect=[
                [total_row],  # by_provider
                [{"_sum": {}, "_count": {"_all": 5}, "userId": "u1"}],  # by_user
                [
                    user_tracking_cost_usd_row,
                    user_tracking_tokens_row,
                ],  # by_user_tracking
                [{"userId": "u1"}],  # distinct users
                [total_row],  # total agg (filtered)
                [total_row],  # total agg (no tracking_type filter)
            ]
        )
        mock_actions.find_many = AsyncMock(return_value=[])

        with (
            patch(
                "backend.data.platform_cost.PrismaLog.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.PrismaUser.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.query_raw_with_schema",
                new_callable=AsyncMock,
                side_effect=[[], []],
            ),
        ):
            dashboard = await get_platform_cost_dashboard(tracking_type="tokens")

        # by_user has 1 user with 5 total requests (tokens rows only due to filter)
        # but per-user cost_bearing count should be 7 (from cost_usd rows in
        # by_user_tracking_groups which uses where_no_tracking_type)
        assert len(dashboard.by_user) == 1
        assert dashboard.by_user[0].cost_bearing_request_count == 7

    @pytest.mark.asyncio
    async def test_global_avg_cost_nonzero_when_filtering_by_tokens(self):
        """When filtering by tracking_type='tokens', avg_cost_microdollars_per_request
        must still reflect cost_usd rows from total_agg_no_tracking_type_groups,
        not the filtered total_agg_groups which only has tokens rows."""
        # filtered total_agg only has tokens rows (zero cost)
        tokens_row = _make_group_by_row(
            provider="openai", tracking_type="tokens", cost=0, count=5
        )
        # unfiltered total_agg has both rows (cost_usd carries the actual cost)
        cost_usd_row = _make_group_by_row(
            provider="openai", tracking_type="cost_usd", cost=10_000, count=4
        )

        mock_actions = MagicMock()
        mock_actions.group_by = AsyncMock(
            side_effect=[
                [tokens_row],  # by_provider
                [{"_sum": {}, "_count": {"_all": 5}, "userId": "u1"}],  # by_user
                [],  # by_user_tracking_groups
                [{"userId": "u1"}],  # distinct users
                [tokens_row],  # total agg (filtered — tokens only)
                [tokens_row, cost_usd_row],  # total agg (no tracking_type filter)
            ]
        )
        mock_actions.find_many = AsyncMock(return_value=[])

        with (
            patch(
                "backend.data.platform_cost.PrismaLog.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.PrismaUser.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.query_raw_with_schema",
                new_callable=AsyncMock,
                side_effect=[[], []],
            ),
        ):
            dashboard = await get_platform_cost_dashboard(tracking_type="tokens")

        # avg_cost_microdollars_per_request must be non-zero: cost_usd row
        # (10_000 microdollars, 4 requests) is present in the unfiltered agg.
        assert dashboard.avg_cost_microdollars_per_request == pytest.approx(10_000 / 4)
        # avg token stats use token_bearing_requests from unfiltered agg (5)
        assert dashboard.avg_input_tokens_per_request == pytest.approx(1000 / 5)
        assert dashboard.avg_output_tokens_per_request == pytest.approx(500 / 5)

    @pytest.mark.asyncio
    async def test_cache_tokens_aggregated_not_hardcoded(self):
        """cache_read_tokens and cache_creation_tokens must be read from the
        DB aggregation, not hardcoded to 0 (regression guard for Sentry report)."""
        provider_row = _make_group_by_row(
            provider="anthropic",
            tracking_type="tokens",
            cost=1000,
            input_tokens=800,
            output_tokens=200,
            cache_read_tokens=400,
            cache_creation_tokens=100,
            count=1,
        )
        user_row = _make_group_by_row(user_id="u2", cost=1000, count=1)

        mock_actions = MagicMock()
        mock_actions.group_by = AsyncMock(
            side_effect=[
                [provider_row],  # by_provider
                [user_row],  # by_user
                [],  # by_user_tracking_groups
                [{"userId": "u2"}],  # distinct users
                [provider_row],  # total agg (tracking_type=None → same as unfiltered)
            ]
        )
        mock_actions.find_many = AsyncMock(return_value=[])

        with (
            patch(
                "backend.data.platform_cost.PrismaLog.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.PrismaUser.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.query_raw_with_schema",
                new_callable=AsyncMock,
                side_effect=[
                    [{"p50": 0, "p75": 0, "p95": 0, "p99": 0}],
                    [],
                ],
            ),
        ):
            dashboard = await get_platform_cost_dashboard()

        assert len(dashboard.by_provider) == 1
        row = dashboard.by_provider[0]
        assert row.total_cache_read_tokens == 400
        assert row.total_cache_creation_tokens == 100

    @pytest.mark.asyncio
    async def test_returns_empty_dashboard(self):
        mock_actions = MagicMock()
        mock_actions.group_by = AsyncMock(side_effect=[[], [], [], [], []])
        mock_actions.find_many = AsyncMock(return_value=[])

        with (
            patch(
                "backend.data.platform_cost.PrismaLog.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.PrismaUser.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.query_raw_with_schema",
                new_callable=AsyncMock,
                side_effect=[[], []],
            ),
        ):
            dashboard = await get_platform_cost_dashboard()

        assert dashboard.total_cost_microdollars == 0
        assert dashboard.total_requests == 0
        assert dashboard.total_users == 0
        assert dashboard.by_provider == []
        assert dashboard.by_user == []
        assert dashboard.cost_p50_microdollars == 0
        assert dashboard.cost_buckets == []

    @pytest.mark.asyncio
    async def test_passes_filters_to_queries(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        mock_actions = MagicMock()
        mock_actions.group_by = AsyncMock(side_effect=[[], [], [], [], []])
        mock_actions.find_many = AsyncMock(return_value=[])

        raw_mock = AsyncMock(side_effect=[[], []])
        with (
            patch(
                "backend.data.platform_cost.PrismaLog.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.PrismaUser.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.query_raw_with_schema",
                raw_mock,
            ),
        ):
            await get_platform_cost_dashboard(
                start=start, provider="openai", user_id="u1"
            )

        # group_by called 5 times (by_provider, by_user, by_user_tracking, distinct users,
        # total agg filtered); the 6th call (total agg no-tracking-type) only runs
        # when tracking_type is set.
        assert mock_actions.group_by.await_count == 5
        # The where dict passed to the first call should include createdAt
        first_call_kwargs = mock_actions.group_by.call_args_list[0][1]
        assert "createdAt" in first_call_kwargs.get("where", {})
        # Raw SQL queries should receive provider and user_id as parameters
        assert raw_mock.await_count == 2
        raw_call_args = raw_mock.call_args_list[0][0]  # positional args of 1st call
        raw_params = raw_call_args[1:]  # first arg is the query template
        assert "openai" in raw_params
        assert "u1" in raw_params

    @pytest.mark.asyncio
    async def test_user_tracking_groups_excludes_tracking_type_filter(self):
        """by_user_tracking_groups must NOT apply the tracking_type filter so that
        cost_usd rows are always included even when the caller filters by 'tokens'."""
        mock_actions = MagicMock()
        mock_actions.group_by = AsyncMock(side_effect=[[], [], [], [], [], []])
        mock_actions.find_many = AsyncMock(return_value=[])

        with (
            patch(
                "backend.data.platform_cost.PrismaLog.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.PrismaUser.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.query_raw_with_schema",
                new_callable=AsyncMock,
                side_effect=[[], []],
            ),
        ):
            await get_platform_cost_dashboard(tracking_type="tokens")

        # Call index 2 is by_user_tracking_groups (0=by_provider, 1=by_user,
        # 2=by_user_tracking, 3=distinct_users, 4=total_agg, 5=total_agg_no_tt).
        tracking_call_where = mock_actions.group_by.call_args_list[2][1]["where"]
        # The main filter applies trackingType; by_user_tracking must NOT.
        assert "trackingType" not in tracking_call_where
        # Other filters (e.g., date range, provider) are still passed through.
        # The first call (by_provider) should have trackingType in its where dict.
        provider_call_where = mock_actions.group_by.call_args_list[0][1]["where"]
        assert "trackingType" in provider_call_where

    @pytest.mark.asyncio
    async def test_graph_exec_id_filter_passed_to_queries(self):
        """graph_exec_id must be forwarded to both prisma and raw SQL queries."""
        mock_actions = MagicMock()
        mock_actions.group_by = AsyncMock(side_effect=[[], [], [], [], []])
        mock_actions.find_many = AsyncMock(return_value=[])
        raw_mock = AsyncMock(side_effect=[[], []])

        with (
            patch(
                "backend.data.platform_cost.PrismaLog.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.PrismaUser.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.platform_cost.query_raw_with_schema",
                raw_mock,
            ),
        ):
            await get_platform_cost_dashboard(graph_exec_id="exec-xyz")

        # Prisma groupBy where must include graphExecId
        first_call_where = mock_actions.group_by.call_args_list[0][1]["where"]
        assert first_call_where.get("graphExecId") == "exec-xyz"
        # Raw SQL params must include the exec id
        raw_params = raw_mock.call_args_list[0][0][1:]
        assert "exec-xyz" in raw_params


def _make_prisma_log_row(
    i: int = 0,
    user_email: str | None = None,
) -> MagicMock:
    row = MagicMock()
    row.id = f"log-{i}"
    row.createdAt = datetime(2026, 3, 1, tzinfo=timezone.utc)
    row.userId = "u1"
    row.graphExecId = None
    row.nodeExecId = None
    row.blockName = "TestBlock"
    row.provider = "openai"
    row.trackingType = "tokens"
    row.costMicrodollars = 1000
    row.inputTokens = 10
    row.outputTokens = 5
    row.duration = 0.5
    row.model = "gpt-4"
    # cacheReadTokens / cacheCreationTokens may not exist on older Prisma clients
    row.configure_mock(**{"cacheReadTokens": None, "cacheCreationTokens": None})
    if user_email is not None:
        row.User = MagicMock()
        row.User.email = user_email
    else:
        row.User = None
    return row


class TestGetPlatformCostLogs:
    @pytest.mark.asyncio
    async def test_returns_logs_and_total(self):
        row = _make_prisma_log_row(0, user_email="a@b.com")
        mock_actions = MagicMock()
        mock_actions.count = AsyncMock(return_value=1)
        mock_actions.find_many = AsyncMock(return_value=[row])

        with patch(
            "backend.data.platform_cost.PrismaLog.prisma",
            return_value=mock_actions,
        ):
            logs, total = await get_platform_cost_logs(page=1, page_size=10)

        assert total == 1
        assert len(logs) == 1
        assert logs[0].id == "log-0"
        assert logs[0].provider == "openai"
        assert logs[0].model == "gpt-4"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_data(self):
        mock_actions = MagicMock()
        mock_actions.count = AsyncMock(return_value=0)
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.platform_cost.PrismaLog.prisma",
            return_value=mock_actions,
        ):
            logs, total = await get_platform_cost_logs()

        assert total == 0
        assert logs == []

    @pytest.mark.asyncio
    async def test_pagination_offset(self):
        mock_actions = MagicMock()
        mock_actions.count = AsyncMock(return_value=100)
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.platform_cost.PrismaLog.prisma",
            return_value=mock_actions,
        ):
            logs, total = await get_platform_cost_logs(page=3, page_size=25)

        assert total == 100
        find_many_call = mock_actions.find_many.call_args[1]
        assert find_many_call["take"] == 25
        assert find_many_call["skip"] == 50  # (3-1) * 25

    @pytest.mark.asyncio
    async def test_explicit_start_skips_default(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mock_actions = MagicMock()
        mock_actions.count = AsyncMock(return_value=0)
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.platform_cost.PrismaLog.prisma",
            return_value=mock_actions,
        ):
            logs, total = await get_platform_cost_logs(start=start)

        assert total == 0
        where = mock_actions.count.call_args[1]["where"]
        # start provided — should appear in the where filter
        assert "createdAt" in where

    @pytest.mark.asyncio
    async def test_graph_exec_id_filter(self):
        mock_actions = MagicMock()
        mock_actions.count = AsyncMock(return_value=0)
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.platform_cost.PrismaLog.prisma",
            return_value=mock_actions,
        ):
            logs, total = await get_platform_cost_logs(graph_exec_id="exec-abc")

        where = mock_actions.count.call_args[1]["where"]
        assert where.get("graphExecId") == "exec-abc"


class TestGetPlatformCostLogsForExport:
    @pytest.mark.asyncio
    async def test_returns_logs_not_truncated(self):
        row = _make_prisma_log_row(0)
        mock_actions = MagicMock()
        mock_actions.find_many = AsyncMock(return_value=[row])

        with patch(
            "backend.data.platform_cost.PrismaLog.prisma",
            return_value=mock_actions,
        ):
            logs, truncated = await get_platform_cost_logs_for_export()

        assert len(logs) == 1
        assert truncated is False
        assert logs[0].id == "log-0"

    @pytest.mark.asyncio
    async def test_returns_empty_not_truncated(self):
        mock_actions = MagicMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.platform_cost.PrismaLog.prisma",
            return_value=mock_actions,
        ):
            logs, truncated = await get_platform_cost_logs_for_export()

        assert logs == []
        assert truncated is False

    @pytest.mark.asyncio
    async def test_truncates_at_export_max_rows(self):
        rows = [_make_prisma_log_row(i) for i in range(3)]
        mock_actions = MagicMock()
        mock_actions.find_many = AsyncMock(return_value=rows)

        with (
            patch(
                "backend.data.platform_cost.PrismaLog.prisma",
                return_value=mock_actions,
            ),
            patch("backend.data.platform_cost.EXPORT_MAX_ROWS", 2),
        ):
            logs, truncated = await get_platform_cost_logs_for_export()

        assert len(logs) == 2
        assert truncated is True

    @pytest.mark.asyncio
    async def test_passes_model_block_tracking_filters(self):
        mock_actions = MagicMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.platform_cost.PrismaLog.prisma",
            return_value=mock_actions,
        ):
            await get_platform_cost_logs_for_export(
                model="gpt-4", block_name="LLMBlock", tracking_type="tokens"
            )

        where = mock_actions.find_many.call_args[1]["where"]
        assert where.get("model") == "gpt-4"
        assert where.get("trackingType") == "tokens"
        # blockName uses a dict filter for case-insensitive match
        assert "blockName" in where

    @pytest.mark.asyncio
    async def test_maps_cache_tokens(self):
        row = _make_prisma_log_row(0)
        row.configure_mock(**{"cacheReadTokens": 50, "cacheCreationTokens": 25})
        mock_actions = MagicMock()
        mock_actions.find_many = AsyncMock(return_value=[row])

        with patch(
            "backend.data.platform_cost.PrismaLog.prisma",
            return_value=mock_actions,
        ):
            logs, _ = await get_platform_cost_logs_for_export()

        assert logs[0].cache_read_tokens == 50
        assert logs[0].cache_creation_tokens == 25

    @pytest.mark.asyncio
    async def test_graph_exec_id_filter(self):
        mock_actions = MagicMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.platform_cost.PrismaLog.prisma",
            return_value=mock_actions,
        ):
            logs, truncated = await get_platform_cost_logs_for_export(
                graph_exec_id="exec-xyz"
            )

        where = mock_actions.find_many.call_args[1]["where"]
        assert where.get("graphExecId") == "exec-xyz"
        assert logs == []
        assert truncated is False

    @pytest.mark.asyncio
    async def test_explicit_start_skips_default(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mock_actions = MagicMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.platform_cost.PrismaLog.prisma",
            return_value=mock_actions,
        ):
            logs, truncated = await get_platform_cost_logs_for_export(start=start)

        assert logs == []
        assert truncated is False
        where = mock_actions.find_many.call_args[1]["where"]
        assert "createdAt" in where
