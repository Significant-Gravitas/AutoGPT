from unittest.mock import AsyncMock

import pytest

from scripts import update_free_trial_usage_limits as backfill


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    connect = AsyncMock()
    disconnect = AsyncMock()
    query = AsyncMock(return_value=[backfill.OfferCount(count=3)])
    execute = AsyncMock(return_value=3)
    monkeypatch.setattr(backfill, "connect", connect)
    monkeypatch.setattr(backfill, "disconnect", disconnect)
    monkeypatch.setattr(backfill, "query_raw_with_schema", query)
    monkeypatch.setattr(backfill, "execute_raw_with_schema", execute)
    return connect, disconnect, query, execute


@pytest.mark.asyncio
async def test_default_dry_run_counts_matching_offers_without_writes(database, capsys):
    connect, disconnect, query, execute = database

    assert await backfill.main() == 0

    connect.assert_awaited_once_with()
    disconnect.assert_awaited_once_with()
    execute.assert_not_awaited()
    count_sql = query.await_args.args[0]
    update_sql = backfill.SQL_PATH.read_text(encoding="utf-8")
    assert count_sql.startswith("SELECT COUNT(*)::int AS count")
    assert count_sql.partition("\nWHERE ")[2] == update_sql.partition("\nWHERE ")[2]
    assert "DRY RUN: 3 free-trial offers need an update" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_apply_uses_schema_aware_conditional_update(database, capsys):
    connect, disconnect, query, execute = database

    assert await backfill.main(apply=True) == 0

    connect.assert_awaited_once_with()
    disconnect.assert_awaited_once_with()
    query.assert_not_awaited()
    execute.assert_awaited_once_with(backfill.SQL_PATH.read_text(encoding="utf-8"))
    assert "Updated 3 free-trial offers" in capsys.readouterr().out


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.asyncio
async def test_failed_backfill_disconnects(database, apply):
    _, disconnect, query, execute = database
    query.side_effect = execute.side_effect = RuntimeError("Database unavailable")

    with pytest.raises(RuntimeError, match="Database unavailable"):
        await backfill.main(apply=apply)

    disconnect.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_missing_database_target_is_rejected(database, monkeypatch):
    connect, _, query, execute = database
    monkeypatch.delenv("DATABASE_URL")

    with pytest.raises(SystemExit, match="DATABASE_URL must be set"):
        await backfill.main(apply=True)

    connect.assert_not_awaited()
    query.assert_not_awaited()
    execute.assert_not_awaited()


def test_cli_defaults_to_dry_run():
    assert backfill.parse_args([]).apply is False
    assert backfill.parse_args(["--apply"]).apply is True
