import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import db as db_module
from backend.data.db import DISCONNECT_TIMEOUT, connect, disconnect, session_pooler_url


# The root conftest spins a full test server for every test via an autouse
# session fixture. These two tests are pure unit tests over a mocked client,
# so shadow it here the way `backend/util/conftest.py` does for its directory
# -- narrowed to this module so the rest of `backend/data` is unaffected.
@pytest.fixture(scope="session")
def server():
    yield None


async def test_disconnect_bounds_the_engine_shutdown(monkeypatch):
    """Prisma's shutdown wait is unbounded unless a timeout is handed to it.

    The blocking wait sits inside an async function, so an engine that ignores
    the SIGINT takes the event loop down with it.
    """
    prisma = MagicMock()
    prisma.is_connected.side_effect = [True, False]
    prisma.disconnect = AsyncMock()
    monkeypatch.setattr(db_module, "prisma", prisma)

    await disconnect()

    prisma.disconnect.assert_awaited_once_with(DISCONNECT_TIMEOUT)


async def test_disconnect_is_a_noop_when_not_connected(monkeypatch):
    prisma = MagicMock()
    prisma.is_connected.return_value = False
    prisma.disconnect = AsyncMock()
    monkeypatch.setattr(db_module, "prisma", prisma)

    await disconnect()

    prisma.disconnect.assert_not_awaited()


@pytest.mark.parametrize(
    ("pool_mode", "level", "says"),
    [
        ("session", "INFO", "session pooler"),
        ("Session", "WARNING", "Ignored DB_POOL_MODE='Session'"),
        ("", None, None),
    ],
    ids=["session mode", "a mistyped mode", "unset"],
)
async def test_connect_says_which_pooler_mode_it_took(
    monkeypatch, caplog, pool_mode, level, says
):
    """A rollout that flips the setting can read in the logs whether it took."""
    prisma = MagicMock()
    prisma.is_connected.side_effect = [False, True]
    prisma.connect = AsyncMock()
    monkeypatch.setattr(db_module, "prisma", prisma)
    monkeypatch.setattr(db_module, "DB_POOL_MODE", pool_mode)
    caplog.set_level("INFO", logger=db_module.__name__)

    await connect()

    said = [r for r in caplog.records if r.name == db_module.__name__]
    if says is None:
        assert said == []
    else:
        assert [r.levelname for r in said] == [level]
        assert says in said[0].getMessage()


class TestSessionPoolerUrl:
    def test_the_transaction_pooler_becomes_the_session_pooler(self):
        url = (
            "postgresql://postgres.ref:p%40ss@aws-0-us-east-2.pooler.supabase.com:6543"
            "/postgres?pgbouncer=true&schema=platform&connection_limit=16"
        )

        assert session_pooler_url(url) == (
            "postgresql://postgres.ref:p%40ss@aws-0-us-east-2.pooler.supabase.com:5432"
            "/postgres?schema=platform&connection_limit=16"
        )

    def test_a_url_on_another_port_keeps_it(self):
        url = "postgresql://postgres:postgres@localhost:5432/postgres?schema=platform"

        assert session_pooler_url(url) == url

    def test_pgbouncer_mode_is_dropped_wherever_the_url_points(self):
        url = "postgresql://u:p@db.example.com:6432/postgres?pgbouncer=true"

        assert (
            session_pooler_url(url) == "postgresql://u:p@db.example.com:6432/postgres"
        )


@pytest.mark.parametrize(
    ("pool_mode", "expected"),
    [
        (
            "session",
            (
                "postgresql://u:p@pooler.example.com:5432/postgres"
                "?schema=platform&connection_limit=4"
            ),
        ),
        (
            "",
            (
                "postgresql://u:p@pooler.example.com:6543/postgres"
                "?pgbouncer=true&schema=platform&connection_limit=4"
            ),
        ),
    ],
    ids=["session mode", "default"],
)
def test_db_pool_mode_decides_the_url_prisma_connects_with(pool_mode, expected):
    """In a process of its own: the URL is built when the module is imported."""
    env = {
        **os.environ,
        "DATABASE_URL": "postgresql://u:p@pooler.example.com:6543/postgres"
        "?pgbouncer=true&schema=platform",
        "DB_POOL_MODE": pool_mode,
        "DB_CONNECTION_LIMIT": "4",
        # Set empty so a developer's .env can't add its own.
        "DB_CONNECT_TIMEOUT": "",
        "DB_POOL_TIMEOUT": "",
        "DB_STATEMENT_CACHE_SIZE": "",
    }
    printed = subprocess.run(
        [sys.executable, "-c", "from backend.data import db; print(db.DATABASE_URL)"],
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    assert printed.stdout.strip().splitlines()[-1] == expected
