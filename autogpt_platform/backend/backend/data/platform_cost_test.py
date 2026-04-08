"""Unit tests for helpers and async functions in platform_cost module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from prisma import Json

from backend.util.json import SafeJson

from .platform_cost import (
    PlatformCostEntry,
    _build_where,
    _mask_email,
    get_platform_cost_dashboard,
    get_platform_cost_logs,
    log_platform_cost,
    log_platform_cost_safe,
)


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


class TestGetPlatformCostDashboard:
    def setup_method(self):
        # @cached stores results in-process; clear between tests to avoid bleed.
        get_platform_cost_dashboard.cache_clear()

    @pytest.mark.asyncio
    async def test_returns_dashboard_with_data(self):
        provider_rows = [
            {
                "provider": "openai",
                "tracking_type": "tokens",
                "total_cost": 5000,
                "total_input_tokens": 1000,
                "total_output_tokens": 500,
                "total_duration": 10.5,
                "request_count": 3,
            }
        ]
        user_rows = [
            {
                "user_id": "u1",
                "email": "a@b.com",
                "total_cost": 5000,
                "total_input_tokens": 1000,
                "total_output_tokens": 500,
                "request_count": 3,
            }
        ]
        # Dashboard runs 3 queries: by_provider, by_user, COUNT(DISTINCT userId).
        mock_query = AsyncMock(side_effect=[provider_rows, user_rows, [{"cnt": 1}]])
        with patch("backend.data.platform_cost.query_raw_with_schema", new=mock_query):
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

    @pytest.mark.asyncio
    async def test_returns_empty_dashboard(self):
        mock_query = AsyncMock(side_effect=[[], [], []])
        with patch("backend.data.platform_cost.query_raw_with_schema", new=mock_query):
            dashboard = await get_platform_cost_dashboard()
        assert dashboard.total_cost_microdollars == 0
        assert dashboard.total_requests == 0
        assert dashboard.total_users == 0
        assert dashboard.by_provider == []
        assert dashboard.by_user == []

    @pytest.mark.asyncio
    async def test_passes_filters_to_queries(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mock_query = AsyncMock(side_effect=[[], [], []])
        with patch("backend.data.platform_cost.query_raw_with_schema", new=mock_query):
            await get_platform_cost_dashboard(
                start=start, provider="openai", user_id="u1"
            )
        assert mock_query.await_count == 3
        first_call_sql = mock_query.call_args_list[0][0][0]
        assert "createdAt" in first_call_sql


class TestGetPlatformCostLogs:
    @pytest.mark.asyncio
    async def test_returns_logs_and_total(self):
        count_rows = [{"cnt": 1}]
        log_rows = [
            {
                "id": "log-1",
                "created_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
                "user_id": "u1",
                "email": "a@b.com",
                "graph_exec_id": "g1",
                "node_exec_id": "n1",
                "block_name": "TestBlock",
                "provider": "openai",
                "tracking_type": "tokens",
                "cost_microdollars": 5000,
                "input_tokens": 100,
                "output_tokens": 50,
                "duration": 1.5,
                "model": "gpt-4",
            }
        ]
        mock_query = AsyncMock(side_effect=[count_rows, log_rows])
        with patch("backend.data.platform_cost.query_raw_with_schema", new=mock_query):
            logs, total = await get_platform_cost_logs(page=1, page_size=10)
        assert total == 1
        assert len(logs) == 1
        assert logs[0].id == "log-1"
        assert logs[0].provider == "openai"
        assert logs[0].model == "gpt-4"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_data(self):
        mock_query = AsyncMock(side_effect=[[{"cnt": 0}], []])
        with patch("backend.data.platform_cost.query_raw_with_schema", new=mock_query):
            logs, total = await get_platform_cost_logs()
        assert total == 0
        assert logs == []

    @pytest.mark.asyncio
    async def test_pagination_offset(self):
        mock_query = AsyncMock(side_effect=[[{"cnt": 100}], []])
        with patch("backend.data.platform_cost.query_raw_with_schema", new=mock_query):
            logs, total = await get_platform_cost_logs(page=3, page_size=25)
        assert total == 100
        second_call_args = mock_query.call_args_list[1][0]
        assert 25 in second_call_args  # page_size
        assert 50 in second_call_args  # offset = (3-1) * 25

    @pytest.mark.asyncio
    async def test_empty_count_returns_zero(self):
        mock_query = AsyncMock(side_effect=[[], []])
        with patch("backend.data.platform_cost.query_raw_with_schema", new=mock_query):
            logs, total = await get_platform_cost_logs()
        assert total == 0
