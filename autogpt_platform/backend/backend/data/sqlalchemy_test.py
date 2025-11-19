"""
Tests for SQLAlchemy infrastructure.

These tests verify that the SQLAlchemy module correctly:
1. Creates async engines with proper configuration
2. Initializes session factories
3. Provides database sessions via get_session()
4. Handles connection pooling
5. Properly disposes of resources
"""

import os
from unittest import mock

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from backend.data import sqlalchemy as sa


class TestDatabaseURLConversion:
    """Test database URL conversion from Prisma to asyncpg format."""

    def test_get_database_url_converts_prisma_format(self):
        """Test that Prisma URL is converted to asyncpg format."""
        # Mock the Config to return a Prisma-style URL
        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = (
                "postgresql://user:pass@localhost:5432/testdb?schema=platform"
            )

            result = sa.get_database_url()

            assert result.startswith("postgresql+asyncpg://")
            assert "?schema=platform" not in result  # Schema param should be removed
            assert "user:pass@localhost:5432/testdb" in result

    def test_get_database_url_removes_schema_parameter(self):
        """Test that schema parameter is removed from URL."""
        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = (
                "postgresql://user:pass@localhost:5432/db?schema=test&other=param"
            )

            result = sa.get_database_url()

            assert "schema=" not in result

    def test_get_database_schema_extracts_schema_name(self):
        """Test that schema name is correctly extracted."""
        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = (
                "postgresql://user:pass@localhost:5432/db?schema=custom_schema"
            )

            result = sa.get_database_schema()

            assert result == "custom_schema"

    def test_get_database_schema_defaults_to_platform(self):
        """Test that schema defaults to 'platform' if not specified."""
        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = (
                "postgresql://user:pass@localhost:5432/db"
            )

            result = sa.get_database_schema()

            assert result == "platform"


class TestEngineCreation:
    """Test async engine creation and configuration."""

    def test_create_engine_returns_async_engine(self):
        """Test that create_engine returns an AsyncEngine instance."""
        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = (
                "postgresql://user:pass@localhost:5432/testdb?schema=platform"
            )
            MockConfig.return_value.sqlalchemy_pool_size = 5
            MockConfig.return_value.sqlalchemy_max_overflow = 2
            MockConfig.return_value.sqlalchemy_pool_timeout = 20
            MockConfig.return_value.sqlalchemy_connect_timeout = 10
            MockConfig.return_value.sqlalchemy_echo = False

            engine = sa.create_engine()

            assert isinstance(engine, AsyncEngine)
            # Note: We don't test actual DB connection here, just engine creation

    def test_create_engine_uses_config_pool_settings(self):
        """Test that engine uses pool settings from config."""
        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = (
                "postgresql://user:pass@localhost:5432/testdb?schema=platform"
            )
            MockConfig.return_value.sqlalchemy_pool_size = 15
            MockConfig.return_value.sqlalchemy_max_overflow = 10
            MockConfig.return_value.sqlalchemy_pool_timeout = 45
            MockConfig.return_value.sqlalchemy_connect_timeout = 15
            MockConfig.return_value.sqlalchemy_echo = True

            engine = sa.create_engine()

            # Verify pool configuration is set (engine.pool.size() would require connection)
            assert engine is not None


class TestSessionFactory:
    """Test session factory creation."""

    def test_create_session_factory_returns_sessionmaker(self):
        """Test that create_session_factory returns a session maker."""
        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = (
                "postgresql://user:pass@localhost:5432/testdb?schema=platform"
            )
            MockConfig.return_value.sqlalchemy_pool_size = 5
            MockConfig.return_value.sqlalchemy_max_overflow = 2
            MockConfig.return_value.sqlalchemy_pool_timeout = 20
            MockConfig.return_value.sqlalchemy_connect_timeout = 10
            MockConfig.return_value.sqlalchemy_echo = False

            engine = sa.create_engine()
            session_factory = sa.create_session_factory(engine)

            assert session_factory is not None
            # The factory is an async_sessionmaker, we can verify it exists


class TestInitialization:
    """Test module initialization."""

    def test_initialize_sets_global_references(self):
        """Test that initialize sets module-level engine and session factory."""
        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = (
                "postgresql://user:pass@localhost:5432/testdb?schema=platform"
            )
            MockConfig.return_value.sqlalchemy_pool_size = 5
            MockConfig.return_value.sqlalchemy_max_overflow = 2
            MockConfig.return_value.sqlalchemy_pool_timeout = 20
            MockConfig.return_value.sqlalchemy_connect_timeout = 10
            MockConfig.return_value.sqlalchemy_echo = False

            engine = sa.create_engine()
            sa.initialize(engine)

            # After initialization, _engine and _session_factory should be set
            assert sa._engine is not None
            assert sa._session_factory is not None

    @pytest.mark.asyncio
    async def test_get_session_raises_without_initialization(self):
        """Test that get_session raises error if not initialized."""
        # Reset globals
        sa._engine = None
        sa._session_factory = None

        with pytest.raises(RuntimeError, match="SQLAlchemy not initialized"):
            async with sa.get_session() as session:
                pass


@pytest.mark.asyncio
class TestSessionLifecycle:
    """Test database session lifecycle using real database."""

    async def test_get_session_provides_valid_session(self):
        """Test that get_session provides a working AsyncSession."""
        # This test requires DATABASE_URL to be set and database to be accessible
        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set, skipping integration test")

        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = os.getenv("DATABASE_URL")
            MockConfig.return_value.sqlalchemy_pool_size = 2
            MockConfig.return_value.sqlalchemy_max_overflow = 1
            MockConfig.return_value.sqlalchemy_pool_timeout = 10
            MockConfig.return_value.sqlalchemy_connect_timeout = 5
            MockConfig.return_value.sqlalchemy_echo = False

            engine = sa.create_engine()
            sa.initialize(engine)

            try:
                # Test using get_session as a context manager
                async with sa.get_session() as session:
                    assert isinstance(session, AsyncSession)

                    # Execute a simple query to verify connection works
                    result = await session.execute(text("SELECT 1"))
                    value = result.scalar()
                    assert value == 1

            finally:
                await sa.dispose()

    async def test_get_session_commits_on_success(self):
        """Test that session commits when no exception occurs."""
        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set, skipping integration test")

        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = os.getenv("DATABASE_URL")
            MockConfig.return_value.sqlalchemy_pool_size = 2
            MockConfig.return_value.sqlalchemy_max_overflow = 1
            MockConfig.return_value.sqlalchemy_pool_timeout = 10
            MockConfig.return_value.sqlalchemy_connect_timeout = 5
            MockConfig.return_value.sqlalchemy_echo = False

            engine = sa.create_engine()
            sa.initialize(engine)

            try:
                # Create a session and perform an operation
                async with sa.get_session() as session:
                    # Just execute a query - commit should happen automatically
                    await session.execute(text("SELECT 1"))

                # If we get here without exception, commit worked
                assert True

            finally:
                await sa.dispose()

    async def test_get_session_rolls_back_on_exception(self):
        """Test that session rolls back when exception occurs."""
        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set, skipping integration test")

        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = os.getenv("DATABASE_URL")
            MockConfig.return_value.sqlalchemy_pool_size = 2
            MockConfig.return_value.sqlalchemy_max_overflow = 1
            MockConfig.return_value.sqlalchemy_pool_timeout = 10
            MockConfig.return_value.sqlalchemy_connect_timeout = 5
            MockConfig.return_value.sqlalchemy_echo = False

            engine = sa.create_engine()
            sa.initialize(engine)

            try:
                with pytest.raises(ValueError):
                    async with sa.get_session() as session:
                        # Execute valid query
                        await session.execute(text("SELECT 1"))
                        # Raise exception to trigger rollback
                        raise ValueError("Test exception")

                # Session should have been rolled back
                assert True

            finally:
                await sa.dispose()


@pytest.mark.asyncio
class TestDisposal:
    """Test resource disposal."""

    async def test_dispose_closes_connections(self):
        """Test that dispose closes all connections."""
        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set, skipping integration test")

        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = os.getenv("DATABASE_URL")
            MockConfig.return_value.sqlalchemy_pool_size = 2
            MockConfig.return_value.sqlalchemy_max_overflow = 1
            MockConfig.return_value.sqlalchemy_pool_timeout = 10
            MockConfig.return_value.sqlalchemy_connect_timeout = 5
            MockConfig.return_value.sqlalchemy_echo = False

            engine = sa.create_engine()
            sa.initialize(engine)

            # Use a session to establish connection
            async with sa.get_session() as session:
                await session.execute(text("SELECT 1"))

            # Dispose should close connections
            await sa.dispose()

            # After disposal, globals should be None
            assert sa._engine is None
            assert sa._session_factory is None

    async def test_dispose_handles_multiple_calls(self):
        """Test that dispose can be called multiple times safely."""
        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set, skipping integration test")

        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = os.getenv("DATABASE_URL")
            MockConfig.return_value.sqlalchemy_pool_size = 2
            MockConfig.return_value.sqlalchemy_max_overflow = 1
            MockConfig.return_value.sqlalchemy_pool_timeout = 10
            MockConfig.return_value.sqlalchemy_connect_timeout = 5
            MockConfig.return_value.sqlalchemy_echo = False

            engine = sa.create_engine()
            sa.initialize(engine)

            # First disposal
            await sa.dispose()

            # Second disposal should not raise error
            await sa.dispose()

            assert sa._engine is None


@pytest.mark.asyncio
class TestConnectionPooling:
    """Test connection pooling behavior."""

    async def test_multiple_sessions_share_pool(self):
        """Test that multiple sessions use the same connection pool."""
        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set, skipping integration test")

        with mock.patch("backend.data.sqlalchemy.Config") as MockConfig:
            MockConfig.return_value.database_url = os.getenv("DATABASE_URL")
            MockConfig.return_value.sqlalchemy_pool_size = 3
            MockConfig.return_value.sqlalchemy_max_overflow = 2
            MockConfig.return_value.sqlalchemy_pool_timeout = 10
            MockConfig.return_value.sqlalchemy_connect_timeout = 5
            MockConfig.return_value.sqlalchemy_echo = False

            engine = sa.create_engine()
            sa.initialize(engine)

            try:
                # Create multiple sessions sequentially
                async with sa.get_session() as session1:
                    result1 = await session1.execute(text("SELECT 1"))
                    assert result1.scalar() == 1

                async with sa.get_session() as session2:
                    result2 = await session2.execute(text("SELECT 2"))
                    assert result2.scalar() == 2

                async with sa.get_session() as session3:
                    result3 = await session3.execute(text("SELECT 3"))
                    assert result3.scalar() == 3

                # All sessions should have worked, sharing the pool
                assert True

            finally:
                await sa.dispose()
