"""
Test fixtures for store integration tests.

Why this file exists: ``content_handlers_integration_test.py`` and
``embeddings_e2e_test.py`` issue real ``query_raw`` calls through the
prisma client and run with ``@pytest.mark.asyncio(loop_scope="session")``.

Earlier in the suite, function-scoped async tests run on their own
event loops. Prisma's underlying httpx.AsyncClient is created lazily on
the first request, binding its connection pool to whatever loop was
current at the time of that first request. When the binding happens on
a function loop and that loop closes, the pool dies and every later
``query_raw`` (including ours on the session loop) raises
``RuntimeError: Event loop is closed``.

Fix: before the first session-scoped integration test in this directory,
close prisma's httpx session and let prisma lazy-reopen it on the
current (session) event loop on the next query.
"""

from __future__ import annotations

import prisma
import pytest_asyncio


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def _reopen_prisma_http_session_on_session_loop():
    """Reset prisma's lazy httpx pool so it rebinds to the session loop.

    Idempotent. Runs once per session at first integration-test setup.
    Close-then-open forces a fresh ``httpx.AsyncClient`` whose connection
    pool will bind to whatever loop is current — i.e., the session loop
    these integration tests run on.
    """
    client = prisma.get_client()
    engine = getattr(client, "_internal_engine", None)
    http = getattr(engine, "session", None) if engine is not None else None

    if http is not None and getattr(http, "_session", None) is not None:
        await http.close()
        # http.session.setter accepts None, but accessing http.session
        # after a close raises HTTPClientClosedError. Re-open immediately.
        http.open()

    yield
