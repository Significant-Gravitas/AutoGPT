"""Local conftest for copilot/graphiti tests.

Two responsibilities:

1. **Opt out of the full SpinTestServer.** Mirrors
   ``backend/copilot/tools/conftest.py`` — these tests don't need
   postgres + rabbitmq + the platform-wide ``graph_cleanup`` autouse.

2. **Provide FalkorDB integration fixtures.** Tests marked
   ``@pytest.mark.integration`` use the ``clean_graph`` and
   ``seeded_graph`` fixtures to talk to a live FalkorDB. The
   ``falkordb_available`` session fixture skips the integration suite
   gracefully when FalkorDB is not reachable, so local unit-test runs
   stay fast.

Run integration tests with the platform docker-compose stack up::

    cd autogpt_platform
    docker compose up -d falkordb
    cd backend
    poetry run pytest -m integration backend/copilot/graphiti

The fixtures connect using ``GraphitiConfig`` defaults
(``localhost:6380`` host-side) — override with the standard
``GRAPHITI_FALKORDB_*`` env vars to point at a different host.
"""

import socket
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator

import pytest
import pytest_asyncio

from .config import graphiti_config
from .falkordb_driver import AutoGPTFalkorDriver


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():  # type: ignore[override]
    """No-op server stub — graphiti tests don't need the full backend."""
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():  # type: ignore[override]
    """No-op graph cleanup stub."""
    yield


@pytest.fixture(scope="session")
def falkordb_available() -> bool:
    """Skip the calling integration suite if FalkorDB is unreachable.

    Resolves host + port from ``GraphitiConfig`` so a developer who has
    customized the env stays in sync. A 1-second TCP probe is enough to
    confirm the port is open — Graphiti's own client will surface any
    deeper protocol mismatch when the test actually runs.
    """
    host = graphiti_config.falkordb_host
    port = graphiti_config.falkordb_port
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as exc:
        pytest.skip(
            f"FalkorDB not reachable at {host}:{port} ({exc.__class__.__name__}). "
            "Bring it up with `docker compose up -d falkordb` from autogpt_platform/, "
            "or set GRAPHITI_FALKORDB_HOST / GRAPHITI_FALKORDB_PORT to a reachable instance."
        )
    return True


def _new_test_group_id() -> str:
    """Mint a unique group_id for the current test.

    The ``test_`` prefix matches the project convention for ephemeral
    Graphiti databases and is easy to grep / sweep. UUID is truncated
    to 16 chars to stay well under the 128-char limit enforced by
    ``derive_group_id``.
    """
    return f"test_{uuid.uuid4().hex[:16]}"


async def _drop_database(driver: AutoGPTFalkorDriver) -> None:
    """Best-effort full wipe of the test database via Cypher.

    DETACH DELETE everything covers entities, episodes, communities, and
    all edges between them. Falls back silently — if the wipe fails on
    cleanup we don't want to fail the test that already passed.
    """
    try:
        await driver.execute_query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass


@pytest_asyncio.fixture(loop_scope="function")
async def clean_graph(
    falkordb_available: bool,
) -> AsyncIterator[tuple[AutoGPTFalkorDriver, str]]:
    """A per-test FalkorDB driver scoped to a fresh group_id.

    Yields ``(driver, group_id)``. After the test, every node + edge in
    the database is detach-deleted; the database itself is left
    in place (FalkorDB doesn't expose a "drop database" Cypher and the
    test FalkorDB is ephemeral anyway).

    Passes ``build_indices=False`` so the driver does NOT spawn the
    background ``build_indices_and_constraints`` task graphiti-core's
    ``FalkorDriver.__init__`` fires by default. That task runs many
    sequential ``CREATE INDEX`` statements that race the test's own
    queries — the integration tests hang indefinitely on the test
    runner with ``Connection closed by server`` retries piling up.
    Indexes are irrelevant to these tests anyway (they assert against
    Cypher we issue directly), so opting out is harmless. See the
    ``AutoGPTFalkorDriver`` docstring for the production rationale.
    """
    group_id = _new_test_group_id()
    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        build_indices=False,
    )
    try:
        yield driver, group_id
    finally:
        await _drop_database(driver)
        await driver.close()


@pytest_asyncio.fixture(loop_scope="function")
async def seeded_graph(
    clean_graph: tuple[AutoGPTFalkorDriver, str],
) -> tuple[AutoGPTFalkorDriver, str]:
    """A canonical small graph for tests that need preexisting state.

    Three entities, two ``:RELATES_TO`` edges, all with the new edge
    properties from ``MemoryFact`` already populated so tests that
    exercise post-P-1 behavior can run without a separate ingestion
    step. The shape::

        (Alice {uuid:'alice'})-[:RELATES_TO {uuid:'e1', status:'active'}]->(Atlas {uuid:'atlas'})
        (Bob   {uuid:'bob'})  -[:RELATES_TO {uuid:'e2', status:'active'}]->(Atlas {uuid:'atlas'})
    """
    driver, group_id = clean_graph
    # FalkorDB doesn't implement Cypher's no-arg ``datetime()`` function
    # (it errors with ``Unknown function 'datetime'``). Generate the
    # timestamp in Python and bind it as a parameter so the seed query
    # runs cleanly against FalkorDB — same workaround the production
    # ``dream/fetch.py`` and ``tools/graphiti_forget.py`` use against
    # the same backend.
    now_iso = datetime.now(timezone.utc).isoformat()
    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'alice', name: 'Alice', group_id: $gid, created_at: $now}),
          (b:Entity {uuid: 'bob',   name: 'Bob',   group_id: $gid, created_at: $now}),
          (t:Entity {uuid: 'atlas', name: 'Atlas', group_id: $gid, created_at: $now}),
          (a)-[:RELATES_TO {
              uuid: 'e1',
              group_id: $gid,
              fact: 'Alice works on Atlas',
              name: 'works_on',
              status: 'active',
              confidence: 0.9,
              source_kind: 'user_asserted',
              scope: 'real:global',
              provenance: 'session:test#msg:1',
              created_at: $now
          }]->(t),
          (b)-[:RELATES_TO {
              uuid: 'e2',
              group_id: $gid,
              fact: 'Bob works on Atlas',
              name: 'works_on',
              status: 'active',
              confidence: 0.8,
              source_kind: 'user_asserted',
              scope: 'real:global',
              provenance: 'session:test#msg:2',
              created_at: $now
          }]->(t)
        """,
        gid=group_id,
        now=now_iso,
    )
    return driver, group_id
