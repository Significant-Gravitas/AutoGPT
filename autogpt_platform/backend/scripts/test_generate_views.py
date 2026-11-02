"""Tests for the analytics view generator (no database needed)."""

import argparse
import re

import pytest

import scripts.generate_views as generate_views


def _args(**overrides) -> argparse.Namespace:
    defaults = {"dry_run": True, "db_url": None, "only": None, "recreate": False}
    return argparse.Namespace(**{**defaults, **overrides})


def test_build_drop_sql_drops_all_views_in_one_statement() -> None:
    assert generate_views.build_drop_sql(["user_lifecycle", "platform_cost_log"]) == (
        "DROP VIEW IF EXISTS analytics.user_lifecycle, analytics.platform_cost_log;\n"
    )


def test_recreate_drops_the_selected_views_before_creating_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    generate_views.cmd_views(
        _args(only="user_lifecycle_funnel_weekly,user_lifecycle", recreate=True)
    )

    out = capsys.readouterr().out
    drop = out.index(
        "DROP VIEW IF EXISTS analytics.user_lifecycle, "
        "analytics.user_lifecycle_funnel_weekly;"
    )
    assert "CASCADE" not in out
    assert out.index("SET LOCAL lock_timeout") < drop
    assert drop < out.index("CREATE OR REPLACE VIEW analytics.user_lifecycle ")
    assert out.index("CREATE OR REPLACE VIEW analytics.user_lifecycle ") < out.index(
        "CREATE OR REPLACE VIEW analytics.user_lifecycle_funnel_weekly "
    )


def test_recreate_restores_the_grants_the_views_had(
    capsys: pytest.CaptureFixture[str],
) -> None:
    generate_views.cmd_views(
        _args(only="user_lifecycle_funnel_weekly,user_lifecycle", recreate=True)
    )

    out = capsys.readouterr().out
    save = out.index("CREATE TEMP TABLE analytics_view_grants ON COMMIT DROP")
    assert "c.relname IN ('user_lifecycle', 'user_lifecycle_funnel_weekly')" in out
    assert save < out.index("DROP VIEW IF EXISTS")
    assert "WHEN a.grantee = 0 THEN 'PUBLIC'" in out
    assert "a.grantee <> c.relowner" in out
    restore = out.index("FOR g IN SELECT * FROM analytics_view_grants")
    assert "'GRANT %s ON analytics.%I TO %s%s'" in out
    assert "WHEN g.is_grantable THEN ' WITH GRANT OPTION'" in out
    assert restore > out.index(
        "CREATE OR REPLACE VIEW analytics.user_lifecycle_funnel_weekly "
    )


def test_without_recreate_nothing_is_dropped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    generate_views.cmd_views(_args(only="user_lifecycle"))

    assert "DROP VIEW" not in capsys.readouterr().out


def test_no_view_exposes_user_email() -> None:
    for path in sorted(generate_views.QUERIES_DIR.glob("*.sql")):
        sql = re.sub(r"--[^\n]*", "", path.read_text())
        assert not re.search(r"\bemail\b", sql, re.IGNORECASE), path.name
