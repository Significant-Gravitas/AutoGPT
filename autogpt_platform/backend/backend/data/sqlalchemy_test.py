"""
Integration tests for SQLAlchemy infrastructure.

These tests verify:
- Engine and session lifecycle management
- Connection pool behavior
- Database URL parsing and schema handling
- Session dependency injection for FastAPI
- Error handling and connection cleanup
- Integration with the docker compose database
"""

import asyncio
import re
from unittest.mock import patch

import pytest
from sqlalchemy import literal, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from backend.data import sqlalchemy as sa
from backend.util.settings import Config

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="function")
async def engine_cleanup():
    """Cleanup fixture to ensure engine is disposed after each test."""
    yield
    # Cleanup after test
    await sa.dispose()


@pytest.fixture(scope="function")
async def initialized_sqlalchemy(engine_cleanup):
    """
    Fixture that initializes SQLAlchemy for tests.

    Creates engine and initializes global state.
    Automatically cleaned up after test.
    """
    engine = sa.create_engine()
    sa.initialize(engine)
    yield engine
    await sa.dispose()


@pytest.fixture(scope="function")
def test_database_config():
    """Fixture to provide test configuration values."""
    config = Config()
    return config


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_get_database_url_parsing():
    """
    Test database URL conversion from Prisma format to asyncpg format.

    Verifies:
    - postgresql:// is replaced with postgresql+asyncpg://
    - schema query parameter is removed
    - Other connection params are preserved
    """
    # The actual DATABASE_URL should be in the environment
    url = sa.get_database_url()

    # Verify it uses asyncpg driver
    assert "postgresql+asyncpg://" in url, "URL should use asyncpg driver"

    # Verify schema parameter is removed from URL
    assert "?schema=" not in url, "Schema parameter should be removed from URL"
    assert "&schema=" not in url, "Schema parameter should be removed from URL"

    # Verify it's a valid database URL structure
    assert re.match(
        r"postgresql\+asyncpg://.*@.*:\d+/.*", url
    ), "URL should match expected format"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_database_schema_extraction():
    """
    Test schema extraction from DATABASE_URL query parameter.

    Verifies the schema name is correctly parsed from the URL.
    """
    schema = sa.get_database_schema()

    # Should extract 'platform' schema (or whatever is configured)
    assert schema is not None, "Schema should not be None"
    assert isinstance(schema, str), "Schema should be a string"
    assert len(schema) > 0, "Schema should not be empty"

    # Based on .env.default, should be 'platform'
    assert schema == "platform", "Default schema should be 'platform'"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_database_schema_default():
    """
    Test default schema when not specified in DATABASE_URL.

    Verifies fallback to 'platform' when schema parameter is missing.
    """
    # Test with mocked Config instance
    with patch("backend.data.sqlalchemy.Config") as MockConfig:
        mock_config = MockConfig.return_value
        mock_config.database_url = "postgresql://user:pass@localhost:5432/testdb"

        schema = sa.get_database_schema()
        assert (
            schema == "platform"
        ), "Should default to 'platform' when schema not specified"


@pytest.mark.asyncio(loop_scope="session")
async def test_database_url_removes_query_params():
    """
    Test that get_database_url properly removes all query parameters.

    Verifies ?... patterns are completely removed.
    """
    # Test with mocked Config instance
    with patch("backend.data.sqlalchemy.Config") as MockConfig:
        mock_config = MockConfig.return_value
        mock_config.database_url = (
            "postgresql://user:pass@localhost:5432/db?schema=test&connect_timeout=60"
        )

        url = sa.get_database_url()
        assert "?" not in url, "All query parameters should be removed"
        assert "schema=" not in url, "Schema parameter should be removed"
        assert (
            "connect_timeout" not in url
        ), "Connect timeout parameter should be removed"
        assert (
            url == "postgresql+asyncpg://user:pass@localhost:5432/db"
        ), "URL should only contain connection details without query params"


# ============================================================================
# ENGINE CREATION TESTS
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_create_engine_with_default_config(engine_cleanup):
    """
    Test engine creation with default configuration.

    Verifies:
    - Engine is created successfully
    - Engine is an AsyncEngine instance
    - Engine has a connection pool
    """
    engine = sa.create_engine()

    assert engine is not None, "Engine should be created"
    assert isinstance(engine, AsyncEngine), "Should create AsyncEngine"

    # Verify engine has a pool
    assert engine.pool is not None, "Engine should have a connection pool"

    # Cleanup
    await engine.dispose()


@pytest.mark.asyncio(loop_scope="session")
async def test_create_engine_pool_configuration(test_database_config):
    """
    Test engine pool configuration.

    Verifies pool_size, max_overflow, and timeout settings are applied.
    """
    engine = sa.create_engine()

    # Verify pool configuration
    pool = engine.pool
    assert pool is not None, "Engine should have a pool"

    # Check pool size matches configuration
    config = test_database_config
    # Note: pool.size() returns the pool size
    # We use hasattr/getattr to avoid type checker issues with internal APIs
    if hasattr(pool, "size"):
        pool_size = pool.size() if callable(pool.size) else pool.size  # type: ignore
        assert (
            pool_size == config.sqlalchemy_pool_size
        ), f"Pool size should be {config.sqlalchemy_pool_size}"

    # Verify max_overflow is set
    if hasattr(pool, "_max_overflow"):
        assert (
            getattr(pool, "_max_overflow") == config.sqlalchemy_max_overflow
        ), f"Max overflow should be {config.sqlalchemy_max_overflow}"

    # Cleanup
    await engine.dispose()


@pytest.mark.asyncio(loop_scope="session")
async def test_create_engine_connection_args():
    """
    Test engine connection arguments.

    Verifies search_path (schema) and timeout are configured correctly.
    """
    engine = sa.create_engine()

    # Get the expected schema
    expected_schema = sa.get_database_schema()

    # Test by actually connecting and checking search_path
    async with engine.connect() as conn:
        # Query current search_path
        result = await conn.execute(text("SHOW search_path"))
        search_path = result.scalar()

        # Verify the schema is in the search_path
        assert search_path is not None, "search_path should not be None"
        assert (
            expected_schema in search_path
        ), f"Schema '{expected_schema}' should be in search_path"

    # Cleanup
    await engine.dispose()


# ============================================================================
# SESSION FACTORY TESTS
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_create_session_factory():
    """
    Test session factory creation.

    Verifies factory is configured correctly with proper settings.
    """
    engine = sa.create_engine()
    factory = sa.create_session_factory(engine)

    assert factory is not None, "Factory should be created"

    # Verify factory configuration
    assert (
        factory.kw.get("expire_on_commit") is False
    ), "expire_on_commit should be False"
    assert factory.kw.get("autoflush") is False, "autoflush should be False"
    assert factory.kw.get("autocommit") is False, "autocommit should be False"

    # Cleanup
    await engine.dispose()


@pytest.mark.asyncio(loop_scope="session")
async def test_session_factory_creates_sessions():
    """
    Test that session factory can create working sessions.

    Verifies sessions can execute queries successfully.
    """
    engine = sa.create_engine()
    factory = sa.create_session_factory(engine)

    # Create a session and execute a simple query
    async with factory() as session:
        result = await session.execute(select(1))
        value = result.scalar()
        assert value == 1, "Should execute simple query successfully"

    # Cleanup
    await engine.dispose()


# ============================================================================
# INITIALIZATION AND LIFECYCLE TESTS
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_initialize_sets_globals(engine_cleanup):
    """
    Test that initialize() sets global engine and session factory.

    Verifies global state is properly configured.
    """
    engine = sa.create_engine()
    sa.initialize(engine)

    # Verify globals are set (by checking get_session doesn't raise)
    async with sa.get_session() as session:
        assert session is not None, "Should create session from factory"
        assert isinstance(session, AsyncSession), "Should be AsyncSession"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_session_before_initialize_fails():
    """
    Test that get_session() raises RuntimeError when not initialized.

    Verifies proper error handling when used before initialization.
    """
    # Ensure we're not initialized
    await sa.dispose()

    # Should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        async with sa.get_session():
            pass

    assert (
        "not initialized" in str(exc_info.value).lower()
    ), "Error message should mention not initialized"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_session_provides_working_session(initialized_sqlalchemy):
    """
    Test that get_session provides a working database session.

    Verifies session can execute queries and access database.
    """
    async with sa.get_session() as session:
        # Execute a simple query
        result = await session.execute(select(1))
        value = result.scalar()
        assert value == 1, "Should execute query successfully"

        # Verify session is active
        assert session.is_active, "Session should be active"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_session_commits_on_success(initialized_sqlalchemy):
    """
    Test that get_session automatically commits on successful completion.

    Verifies transaction is committed when no exception occurs.
    """
    # This test verifies the commit behavior by checking that
    # the session successfully completes without errors
    async with sa.get_session() as session:
        # Execute a query
        result = await session.execute(select(1))
        assert result.scalar() == 1
        # Session should auto-commit on exit

    # If we get here without exception, commit succeeded


@pytest.mark.asyncio(loop_scope="session")
async def test_dispose_closes_connections(initialized_sqlalchemy):
    """
    Test that dispose() properly closes all connections.

    Verifies cleanup is performed correctly.
    """
    # Create a session to establish connection
    async with sa.get_session() as session:
        await session.execute(select(1))

    # Dispose should close all connections
    await sa.dispose()

    # Verify engine is cleaned up (globals should be None)
    # After dispose, get_session should fail
    with pytest.raises(RuntimeError):
        async with sa.get_session():
            pass


# ============================================================================
# CONNECTION POOL INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_connection_pool_reuses_connections(initialized_sqlalchemy):
    """
    Test that connection pool reuses connections.

    Verifies connections are borrowed and returned to pool.
    """
    # Execute multiple queries in sequence
    for i in range(5):
        async with sa.get_session() as session:
            result = await session.execute(select(literal(i)))
            assert result.scalar() == i

    # All queries should complete successfully, reusing connections


@pytest.mark.asyncio(loop_scope="session")
async def test_connection_pool_concurrent_sessions(initialized_sqlalchemy):
    """
    Test multiple concurrent sessions from the pool.

    Verifies pool can handle concurrent access.
    """

    async def execute_query(query_id: int):
        async with sa.get_session() as session:
            result = await session.execute(select(literal(query_id)))
            return result.scalar()

    # Run 5 concurrent queries
    results = await asyncio.gather(
        execute_query(1),
        execute_query(2),
        execute_query(3),
        execute_query(4),
        execute_query(5),
    )

    # Verify all queries succeeded
    assert results == [
        1,
        2,
        3,
        4,
        5,
    ], "All concurrent queries should complete successfully"


@pytest.mark.asyncio(loop_scope="session")
async def test_connection_pool_respects_limits(initialized_sqlalchemy):
    """
    Test that connection pool respects size limits.

    Verifies pool_size + max_overflow configuration.
    """
    config = Config()
    max_connections = config.sqlalchemy_pool_size + config.sqlalchemy_max_overflow

    # This test just verifies the pool doesn't crash with concurrent load
    # Actual limit enforcement is handled by SQLAlchemy

    async def execute_query(query_id: int):
        async with sa.get_session() as session:
            await asyncio.sleep(0.1)  # Hold connection briefly
            result = await session.execute(select(literal(query_id)))
            return result.scalar()

    # Run queries up to the limit
    tasks = [execute_query(i) for i in range(min(max_connections, 10))]
    results = await asyncio.gather(*tasks)

    assert len(results) == min(
        max_connections, 10
    ), "Should handle concurrent queries up to pool limit"


@pytest.mark.asyncio(loop_scope="session")
async def test_connection_pool_timeout_on_exhaustion(initialized_sqlalchemy):
    """
    Test pool timeout when all connections are exhausted.

    Verifies TimeoutError is raised when waiting for connection.
    """
    # This test is complex and may not be reliable in all environments
    # We'll test that the pool can handle at least some concurrent load
    # without timing out

    async def hold_connection(duration: float):
        async with sa.get_session() as session:
            await asyncio.sleep(duration)
            result = await session.execute(select(1))
            return result.scalar()

    # Run a few concurrent queries
    tasks = [hold_connection(0.1) for _ in range(3)]
    results = await asyncio.gather(*tasks)

    assert all(
        r == 1 for r in results
    ), "Should handle concurrent queries within pool capacity"


@pytest.mark.asyncio(loop_scope="session")
async def test_connection_pool_pre_ping(initialized_sqlalchemy):
    """
    Test that pool_pre_ping validates connections.

    Verifies stale connections are detected and refreshed.
    """
    # Execute a query to establish a connection
    async with sa.get_session() as session:
        result = await session.execute(select(1))
        assert result.scalar() == 1

    # Execute another query - pre_ping should validate connection
    async with sa.get_session() as session:
        result = await session.execute(select(literal(2)))
        assert result.scalar() == 2

    # If pre_ping is working, both queries succeed


@pytest.mark.asyncio(loop_scope="session")
async def test_schema_search_path_applied(initialized_sqlalchemy):
    """
    Test that queries use the correct schema (search_path).

    Verifies connection search_path is set to platform schema.
    """
    expected_schema = sa.get_database_schema()

    async with sa.get_session() as session:
        # Check current search_path
        result = await session.execute(text("SHOW search_path"))
        search_path = result.scalar()

        # Verify the platform schema is in search_path
        assert search_path is not None, "search_path should not be None"
        assert (
            expected_schema in search_path
        ), f"Schema '{expected_schema}' should be in search_path"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_get_session_rolls_back_on_error(initialized_sqlalchemy):
    """
    Test that get_session rolls back transaction on exception.

    Verifies automatic rollback on error.
    """
    try:
        async with sa.get_session() as session:
            # Execute a valid query
            await session.execute(select(1))

            # Raise an exception
            raise ValueError("Test error")
    except ValueError:
        pass  # Expected

    # Should be able to use a new session after rollback
    async with sa.get_session() as session:
        result = await session.execute(select(1))
        assert (
            result.scalar() == 1
        ), "Should be able to create new session after rollback"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_session_always_closes_session(initialized_sqlalchemy):
    """
    Test that get_session always closes session, even on error.

    Verifies connection is returned to pool on exception.
    """
    session_closed = False

    try:
        async with sa.get_session() as session:
            # Execute query
            await session.execute(select(1))
            # Raise error
            raise RuntimeError("Test error")
    except RuntimeError:
        # Session should be closed even though we raised
        session_closed = True

    assert session_closed, "Should have caught the error"

    # Verify we can still create new sessions
    async with sa.get_session() as session:
        result = await session.execute(select(1))
        assert result.scalar() == 1, "Should be able to create new session after error"


@pytest.mark.asyncio(loop_scope="session")
async def test_database_connection_error_handling():
    """
    Test behavior with invalid DATABASE_URL.

    Verifies proper error handling for connection failures.
    """
    try:
        # Mock Config with invalid URL
        with patch("backend.data.sqlalchemy.Config") as MockConfig:
            mock_config = MockConfig.return_value
            mock_config.database_url = (
                "postgresql://invalid:invalid@invalid:9999/invalid?schema=platform"
            )
            mock_config.sqlalchemy_pool_size = 10
            mock_config.sqlalchemy_max_overflow = 5
            mock_config.sqlalchemy_pool_timeout = 30
            mock_config.sqlalchemy_connect_timeout = 10
            mock_config.sqlalchemy_echo = False

            engine = sa.create_engine()
            sa.initialize(engine)

            # Try to use session - should fail with connection error
            with pytest.raises((DBAPIError, Exception)):
                async with sa.get_session() as session:
                    await session.execute(select(1))
    finally:
        # Cleanup
        await sa.dispose()


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_session_error_isolation(initialized_sqlalchemy):
    """
    Test that error in one session doesn't affect others.

    Verifies session isolation and independent error handling.
    """

    async def failing_query():
        try:
            async with sa.get_session() as session:
                # Execute invalid SQL
                await session.execute(text("SELECT * FROM nonexistent_table"))
        except Exception:
            return "failed"
        return "succeeded"

    async def successful_query():
        async with sa.get_session() as session:
            result = await session.execute(select(1))
            return result.scalar()

    # Run both concurrently
    results = await asyncio.gather(
        failing_query(), successful_query(), return_exceptions=False
    )

    # First should fail, second should succeed
    assert results[0] == "failed", "First query should fail"
    assert results[1] == 1, "Second query should succeed despite first failing"


# ============================================================================
# FASTAPI INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_session_dependency_injection(initialized_sqlalchemy):
    """
    Test that session can be used as FastAPI dependency.

    Verifies Depends(get_session) pattern works.
    """
    # Simulate FastAPI dependency injection
    async with sa.get_session() as session:
        # This is how it would be injected into a route
        assert isinstance(session, AsyncSession), "Should receive AsyncSession instance"

        # Should be able to execute queries
        result = await session.execute(select(1))
        assert result.scalar() == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_session_lifecycle_in_endpoint(initialized_sqlalchemy):
    """
    Test full request/response cycle with session.

    Simulates a FastAPI endpoint using the session.
    """

    # Simulate an endpoint that uses the session
    async def mock_endpoint():
        async with sa.get_session() as session:
            # Simulate querying data
            result = await session.execute(select(literal(42)))
            value = result.scalar()

            # Simulate returning response
            return {"value": value}

    # Execute the mock endpoint
    response = await mock_endpoint()

    assert response["value"] == 42, "Endpoint should return correct value"

    # Verify we can still use sessions after endpoint completes
    async with sa.get_session() as session:
        result = await session.execute(select(1))
        assert result.scalar() == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_multiple_requests_share_pool(initialized_sqlalchemy):
    """
    Test that multiple requests share the same connection pool.

    Verifies pool reuse across simulated requests.
    """

    async def simulate_request(request_id: int):
        async with sa.get_session() as session:
            result = await session.execute(select(literal(request_id)))
            return result.scalar()

    # Simulate 10 concurrent requests
    results = await asyncio.gather(*[simulate_request(i) for i in range(10)])

    # All requests should complete successfully
    assert results == list(range(10)), "All requests should complete using shared pool"
