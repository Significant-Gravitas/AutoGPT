"""
SQLAlchemy infrastructure for AutoGPT Platform.

This module provides:
1. Async engine creation with connection pooling
2. Session factory for dependency injection
3. Database lifecycle management
"""

import logging
import re
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import QueuePool

from backend.util.settings import Config

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================


def get_database_url() -> str:
    """
    Extract database URL from environment and convert to async format.

    Prisma URL: postgresql://user:pass@host:port/db?schema=platform
    Async URL:  postgresql+asyncpg://user:pass@host:port/db

    Returns the async-compatible URL without schema parameter (handled separately).
    """
    prisma_url = Config().database_url

    # Replace postgresql:// with postgresql+asyncpg://
    async_url = prisma_url.replace("postgresql://", "postgresql+asyncpg://")

    # Remove schema parameter (we'll handle via MetaData)
    async_url = re.sub(r"\?schema=\w+", "", async_url)

    # Remove any remaining query parameters that might conflict
    async_url = re.sub(r"&schema=\w+", "", async_url)

    return async_url


def get_database_schema() -> str:
    """
    Extract schema name from DATABASE_URL query parameter.

    Returns 'platform' by default (matches Prisma configuration).
    """
    prisma_url = Config().database_url
    match = re.search(r"schema=(\w+)", prisma_url)
    return match.group(1) if match else "platform"


# ============================================================================
# ENGINE CREATION
# ============================================================================


def create_engine() -> AsyncEngine:
    """
    Create async SQLAlchemy engine with connection pooling.

    This should be called ONCE per process at startup.
    The engine is long-lived and thread-safe.

    Connection Pool Configuration:
    - pool_size: Number of persistent connections (default: 10)
    - max_overflow: Additional connections when pool exhausted (default: 5)
    - pool_timeout: Seconds to wait for connection (default: 30)
    - pool_pre_ping: Test connections before using (prevents stale connections)

    Total max connections = pool_size + max_overflow = 15
    """
    url = get_database_url()
    config = Config()

    engine = create_async_engine(
        url,
        # Connection pool configuration
        poolclass=QueuePool,  # Standard connection pool
        pool_size=config.sqlalchemy_pool_size,  # Persistent connections
        max_overflow=config.sqlalchemy_max_overflow,  # Burst capacity
        pool_timeout=config.sqlalchemy_pool_timeout,  # Wait time for connection
        pool_pre_ping=True,  # Validate connections before use
        # Async configuration
        echo=config.sqlalchemy_echo,  # Log SQL statements (dev/debug only)
        future=True,  # Use SQLAlchemy 2.0 style
        # Connection arguments (passed to asyncpg)
        connect_args={
            "server_settings": {
                "search_path": get_database_schema(),  # Use 'platform' schema
            },
            "timeout": config.sqlalchemy_connect_timeout,  # Connection timeout
        },
    )

    logger.info(
        f"SQLAlchemy engine created: pool_size={config.sqlalchemy_pool_size}, "
        f"max_overflow={config.sqlalchemy_max_overflow}, "
        f"schema={get_database_schema()}"
    )

    return engine


# ============================================================================
# SESSION FACTORY
# ============================================================================


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    Create session factory for creating AsyncSession instances.

    The factory is configured once, then used to create sessions on-demand.
    Each session represents a single database transaction.

    Args:
        engine: The async engine (with connection pool)

    Returns:
        Session factory that creates properly configured AsyncSession instances
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,  # Don't expire objects after commit
        autoflush=False,  # Manual control over when to flush
        autocommit=False,  # Explicit transaction control
    )


# ============================================================================
# DEPENDENCY INJECTION FOR FASTAPI
# ============================================================================

# Global references (set during app startup)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def initialize(engine: AsyncEngine) -> None:
    """
    Initialize global engine and session factory.

    Called during FastAPI lifespan startup.

    Args:
        engine: The async engine to use for this process
    """
    global _engine, _session_factory
    _engine = engine
    _session_factory = create_session_factory(engine)
    logger.info("SQLAlchemy session factory initialized")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides database session.

    Usage in routes:
        @router.get("/users/{user_id}")
        async def get_user(
            user_id: int,
            session: AsyncSession = Depends(get_session)
        ):
            result = await session.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()

    Usage in DatabaseManager RPC methods:
        @expose
        async def get_user(user_id: int):
            async with get_session() as session:
                result = await session.execute(select(User).where(User.id == user_id))
                return result.scalar_one_or_none()

    Lifecycle:
    1. Request arrives
    2. FastAPI calls this function (or used as context manager)
    3. Session is created (borrows connection from pool)
    4. Session is injected into route handler
    5. Route executes (may commit/rollback)
    6. Route returns
    7. Session is closed (returns connection to pool)

    Error handling:
    - If exception occurs, session is rolled back
    - Connection is always returned to pool (even on error)
    """
    if _session_factory is None:
        raise RuntimeError(
            "SQLAlchemy not initialized. Call initialize() in lifespan context."
        )

    # Create session (borrows connection from pool)
    async with _session_factory() as session:
        try:
            yield session  # Inject into route handler or context manager
            # If we get here, route succeeded - commit any pending changes
            await session.commit()
        except Exception:
            # Error occurred - rollback transaction
            await session.rollback()
            raise
        finally:
            # Always close session (returns connection to pool)
            await session.close()


async def dispose() -> None:
    """
    Dispose of engine and close all connections.

    Called during FastAPI lifespan shutdown.
    Closes all connections in the pool gracefully.
    """
    global _engine, _session_factory

    if _engine is not None:
        logger.info("Disposing SQLAlchemy engine...")
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("SQLAlchemy engine disposed")
