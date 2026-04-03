"""Unit tests for pure helpers in platform_cost module."""

from datetime import datetime, timezone

from .platform_cost import _build_where, _json_or_none


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
