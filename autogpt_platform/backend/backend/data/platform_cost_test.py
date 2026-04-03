"""Unit tests for helpers and async functions in platform_cost module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from .platform_cost import (
    PlatformCostEntry,
    _build_where,
    _json_or_none,
    get_platform_cost_dashboard,
    get_platform_cost_logs,
    log_platform_cost,
    log_platform_cost_safe,
)


class TestJsonOrNone:
    def test_returns_none_for_none(self):
        assert _json_or_none(None) is None

    def test_returns_json_string_for_dict(self):
        result = _json_or_none({"key": "value", "num": 42})
        assert result is not None
        assert '"key"' in result
        assert '"value"' in result

    def test_returns_json_for_empty_dict(self):
        assert _json_or_none({}) == "{}"


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
        sql, params = _build_where(None, None, "openai", None)
        assert 'LOWER("provider") = LOWER($1)' in sql
        assert params == ["openai"]

    def test_user_id_only(self):
        sql, params = _build_where(None, None, None, "user-123")
        assert '"userId" = $1' in sql
        assert params == ["user-123"]

    def test_all_filters(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 6, 1, tzinfo=timezone.utc)
        sql, params = _build_where(start, end, "anthropic", "u1")
        assert "$1" in sql
        assert "$2" in sql
        assert "$3" in sql
        assert "$4" in sql
        assert len(params) == 4
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
    async def test_calls_execute_raw_with_schema(self):
        mock_exec = AsyncMock()
        with patch("backend.data.platform_cost.execute_raw_with_schema", new=mock_exec):
            entry = _make_entry(
                input_tokens=100,
                output_tokens=50,
                cost_microdollars=5000,
                model="gpt-4",
                metadata={"key": "val"},
            )
            await log_platform_cost(entry)
        mock_exec.assert_awaited_once()
        args = mock_exec.call_args
        assert args[0][1] == "user-1"  # user_id is first param
        assert args[0][6] == "block-1"  # block_id
        assert args[0][7] == "TestBlock"  # block_name

    @pytest.mark.asyncio
    async def test_metadata_none_passes_none(self):
        mock_exec = AsyncMock()
        with patch("backend.data.platform_cost.execute_raw_with_schema", new=mock_exec):
            entry = _make_entry(metadata=None)
            await log_platform_cost(entry)
        args = mock_exec.call_args
        assert args[0][-1] is None  # last arg is metadata json


class TestLogPlatformCostSafe:
    @pytest.mark.asyncio
    async def test_does_not_raise_on_error(self):
        with patch(
            "backend.data.platform_cost.execute_raw_with_schema",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        ):
            entry = _make_entry()
            await log_platform_cost_safe(entry)

    @pytest.mark.asyncio
    async def test_succeeds_when_no_error(self):
        mock_exec = AsyncMock()
        with patch("backend.data.platform_cost.execute_raw_with_schema", new=mock_exec):
            entry = _make_entry()
            await log_platform_cost_safe(entry)
        mock_exec.assert_awaited_once()


class TestGetPlatformCostDashboard:
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
        user_count_rows = [{"cnt": 2}]
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
        mock_query = AsyncMock(side_effect=[provider_rows, user_count_rows, user_rows])
        with patch("backend.data.platform_cost.query_raw_with_schema", new=mock_query):
            dashboard = await get_platform_cost_dashboard()
        assert dashboard.total_cost_microdollars == 5000
        assert dashboard.total_requests == 3
        assert dashboard.total_users == 2
        assert len(dashboard.by_provider) == 1
        assert dashboard.by_provider[0].provider == "openai"
        assert dashboard.by_provider[0].tracking_type == "tokens"
        assert dashboard.by_provider[0].total_duration_seconds == 10.5
        assert len(dashboard.by_user) == 1
        assert dashboard.by_user[0].email == "a@b.com"

    @pytest.mark.asyncio
    async def test_returns_empty_dashboard(self):
        mock_query = AsyncMock(side_effect=[[], [{"cnt": 0}], []])
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
        mock_query = AsyncMock(side_effect=[[], [{"cnt": 0}], []])
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
