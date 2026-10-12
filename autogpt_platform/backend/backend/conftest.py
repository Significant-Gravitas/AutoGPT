import inspect
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from multiprocessing import get_all_start_methods, get_context, set_forkserver_preload
from threading import Thread
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import uuid4

import pytest
import pytest_asyncio
import sentry_sdk
from dotenv import load_dotenv
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport

from backend.data import versioned_cache
from backend.data.subscription_lookup_cache import ACTIVE_SUBSCRIPTIONS
from backend.data.subscription_lookup_unconfirmed import UnconfirmedInvalidations
from backend.data.subscription_pending_cache import PENDING_CHANGES
from backend.util.logging import configure_logging
from backend.util.tracing import traces_sampler

load_dotenv()


def _with_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query[key] = value
    return urlunparse(parsed._replace(query=urlencode(query)))


# Every database connection this session opens, the server fixture's service
# processes included, carries this name, so no_leaked_db_transaction can tell
# them from any other client of the same database. Set before anything imports
# backend.data.db, which reads the URL once; the default is that module's own.
os.environ["DATABASE_URL"] = _with_query_param(
    os.environ.get("DATABASE_URL", "postgresql://localhost:5432"),
    "application_name",
    f"autogpt-tests-{uuid4().hex[:8]}",
)

#  NOTE: You can run tests like with the --log-cli-level=INFO to see the logs
# Set up logging
configure_logging()
logger = logging.getLogger(__name__)
_preload_started = False

# Reduce Prisma log spam unless PRISMA_DEBUG is set
if not os.getenv("PRISMA_DEBUG"):
    prisma_logger = logging.getLogger("prisma")
    prisma_logger.setLevel(logging.INFO)


def pytest_itemcollected(item: pytest.Function) -> None:
    global _preload_started
    if (
        _preload_started
        or os.getenv("CI")
        or item.config.getoption("collectonly")
        or "server" not in item.fixturenames
        or "forkserver" not in get_all_start_methods()
    ):
        return
    set_forkserver_preload(["scripts.server_preload"])
    Thread(target=_warm_test_forkserver, daemon=True).start()
    _preload_started = True


def _warm_test_forkserver() -> None:
    process = get_context("forkserver").Process(target=os.getpid)
    process.start()
    process.join()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    from backend.util.test import SpinTestServer

    async with SpinTestServer() as server:
        async with _track_test_graphs(server):
            yield server


@pytest_asyncio.fixture(autouse=True)
async def no_leaked_db_transaction(request: pytest.FixtureRequest):
    """Fail a test that leaves a database transaction open.

    A connection back in the pool inside an open transaction runs every later
    test's statements inside it: their writes are never committed, and the
    first one that fails aborts the rest. Checked while the session's client
    is connected, i.e. once a test has started the server.
    """
    started = datetime.now(timezone.utc)
    yield
    # Nothing can leak through a database layer this session never loaded,
    # and checking that keeps Prisma out of unit-only runs.
    if "backend.data.db" not in sys.modules:
        return
    from prisma import get_client

    from backend.data.db_transaction_guard import end_leaked_transactions

    # The registered client, not backend.data.db.prisma: a test's patch of that
    # attribute can still be in place while this teardown runs.
    client = get_client()
    if not client.is_connected():
        return
    report = await end_leaked_transactions(client)
    if not report.leaks:
        return
    pytest.fail(
        f"{request.node.nodeid} left a database transaction open (test began "
        f"{started.isoformat()}); every later test would have run inside it:\n"
        + "\n".join(
            f"  pid {leak.pid}, {leak.state}, opened {leak.xact_start.isoformat()},"
            f" last statement: {leak.last_query}"
            for leak in report.leaks
        )
        + (
            f"\nBackends {report.survivors} could not be confirmed ended, so "
            "later tests may still see their transactions."
            if report.survivors
            else ""
        ),
        pytrace=False,
    )


@pytest.fixture(autouse=True)
def no_force_all_flags(monkeypatch: pytest.MonkeyPatch):
    """``load_dotenv()`` above copies ``.env`` into ``os.environ``, so a developer
    who turned on ``FORCE_ALL_FLAGS`` locally would otherwise fail every
    flag-off test. Tests that want the switch on set it themselves."""
    monkeypatch.delenv("FORCE_ALL_FLAGS", raising=False)
    monkeypatch.delenv("NEXT_PUBLIC_FORCE_ALL_FLAGS", raising=False)


@pytest.fixture(autouse=True)
def isolated_versioned_caches(monkeypatch: pytest.MonkeyPatch):
    """Versioned caches live in Redis, which outlives a test and is shared by
    parallel runs: give every test keys of its own, empty process-local state,
    and a Redis that hasn't backed off after an earlier test's failure."""
    run = uuid4().hex
    monkeypatch.setattr(versioned_cache._backoff, "until", 0.0)
    for cache in (ACTIVE_SUBSCRIPTIONS, PENDING_CHANGES):
        monkeypatch.setattr(cache, "prefix", f"test:{run}:{cache.prefix}")
        monkeypatch.setattr(cache, "_in_flight", {})
        monkeypatch.setattr(cache, "_fallback", {})
        monkeypatch.setattr(
            cache,
            "_unconfirmed",
            UnconfirmedInvalidations(lifetime=cache.ttl_seconds + 1),
        )


@pytest.fixture
def test_user_id() -> str:
    """Test user ID fixture."""
    return "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


@pytest.fixture
def admin_user_id() -> str:
    """Admin user ID fixture."""
    return "4e53486c-cf57-477e-ba2a-cb02dc828e1b"


@pytest.fixture
def target_user_id() -> str:
    """Target user ID fixture."""
    return "5e53486c-cf57-477e-ba2a-cb02dc828e1c"


class _SentryTransactionCapture(Transport):
    """Keeps the transactions a test's Sentry client would have sent."""

    def __init__(self) -> None:
        super().__init__()
        self.transactions: list[dict] = []

    def capture_envelope(self, envelope: Envelope) -> None:
        for item in envelope.items:
            if item.type == "transaction" and item.payload.json is not None:
                self.transactions.append(item.payload.json)


@pytest.fixture
def sentry_transactions() -> Iterator[list[dict]]:
    """A live Sentry client for the test, sampling as production does. Every
    transaction it finishes is appended, as sent, to the yielded list."""
    capture = _SentryTransactionCapture()
    client = sentry_sdk.Client(
        dsn="https://key@o1.ingest.us.sentry.io/1",
        transport=capture,
        traces_sampler=traces_sampler,
        default_integrations=False,
        auto_enabling_integrations=False,
        send_client_reports=False,
    )
    global_scope = sentry_sdk.get_global_scope()
    previous = global_scope.client
    global_scope.set_client(client)
    try:
        yield capture.transactions
    finally:
        global_scope.set_client(previous)
        client.close()


async def _create_user_with_loop_retry(user_data: dict) -> None:
    """Create a user, retrying once on a transient ``Event loop is closed``.

    Fire-and-forget background tasks elsewhere can leave the Prisma pool
    bound to a now-closed test function loop. The first session-loop DB
    call after that surfaces as ``RuntimeError: Event loop is closed``;
    the pool re-establishes itself on the retry.
    """
    from backend.data.user import get_or_create_user
    from backend.util.exceptions import DatabaseError

    try:
        await get_or_create_user(user_data)
    except DatabaseError as e:
        if "Event loop is closed" not in str(e):
            raise
        await get_or_create_user(user_data)


@pytest.fixture
async def setup_test_user(server, test_user_id):
    """Create test user in database before tests."""
    user_data = {
        "sub": test_user_id,
        "email": "test@example.com",
        "user_metadata": {"name": "Test User"},
    }
    await _create_user_with_loop_retry(user_data)
    return test_user_id


@pytest.fixture
async def setup_admin_user(server, admin_user_id):
    """Create admin user in database before tests."""
    user_data = {
        "sub": admin_user_id,
        "email": "test-admin@example.com",
        "user_metadata": {"name": "Test Admin"},
    }
    await _create_user_with_loop_retry(user_data)
    return admin_user_id


@asynccontextmanager
async def _track_test_graphs(server):
    """Delete the graphs and store listings that tests created through the test
    server, at the end of the session, so they don't pile up in the test DB."""
    created_graphs: list[tuple[str, str]] = []
    created_listing_ids: list[str] = []
    agent_server = server.agent_server
    original_create_graph = agent_server.test_create_graph
    original_create_store_listing = agent_server.test_create_store_listing

    async def create_graph_wrapper(*args, **kwargs):
        created_graph = await original_create_graph(*args, **kwargs)
        # Callers pass user_id both positionally and by keyword.
        call = inspect.signature(original_create_graph).bind(*args, **kwargs)
        user_id = call.arguments["user_id"]
        created_graphs.extend(
            (graph.id, user_id) for graph in [created_graph, *created_graph.sub_graphs]
        )
        return created_graph

    async def create_store_listing_wrapper(*args, **kwargs):
        from fastapi.responses import JSONResponse

        store_listing = await original_create_store_listing(*args, **kwargs)
        if not isinstance(store_listing, JSONResponse):
            created_listing_ids.append(store_listing.listing_id)
        return store_listing

    try:
        agent_server.test_create_graph = create_graph_wrapper
        agent_server.test_create_store_listing = create_store_listing_wrapper
        yield  # This runs the test function
    finally:
        agent_server.test_create_graph = original_create_graph
        agent_server.test_create_store_listing = original_create_store_listing
        await _delete_test_graphs(agent_server, created_graphs, created_listing_ids)


async def _delete_test_graphs(
    agent_server, created_graphs: list[tuple[str, str]], listing_ids: list[str]
) -> None:
    from prisma.models import (
        AgentGraph,
        AgentNodeExecutionInputOutput,
        AgentPreset,
        LibraryAgent,
        StoreListing,
    )

    graph_ids = [graph_id for graph_id, _ in created_graphs]
    # Listing versions, presets and every user's library entries hold the graph
    # (onDelete: Restrict), so they go first. Deleting a listing cascades to
    # its versions and their reviews; a preset's saved inputs would only be
    # orphaned, so they are deleted with it.
    await StoreListing.prisma().delete_many(where={"id": {"in": listing_ids}})
    await LibraryAgent.prisma().delete_many(where={"agentGraphId": {"in": graph_ids}})
    presets = await AgentPreset.prisma().find_many(
        where={"agentGraphId": {"in": graph_ids}}
    )
    preset_ids = [preset.id for preset in presets]
    await AgentNodeExecutionInputOutput.prisma().delete_many(
        where={"agentPresetId": {"in": preset_ids}}
    )
    await AgentPreset.prisma().delete_many(where={"id": {"in": preset_ids}})
    for graph_id, user_id in created_graphs:
        await agent_server.test_delete_graph(graph_id, user_id)

    remaining = await AgentGraph.prisma().find_many(where={"id": {"in": graph_ids}})
    leftover = sorted({graph.id for graph in remaining})
    assert not leftover, f"Test graphs were not deleted: {leftover}"
