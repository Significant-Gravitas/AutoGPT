"""E2B sandbox manager for CoPilot sessions.

Manages e2b sandbox lifecycle: create, reuse via Redis, dispose with GCS sync.
One sandbox per session, cached in-memory on the worker thread and stored in
Redis for cross-pod reconnection.
"""

import asyncio
import logging
import time
from typing import Any

from backend.util.settings import Config

logger = logging.getLogger(__name__)

_REDIS_KEY_PREFIX = "copilot:sandbox:"
_SANDBOX_HOME = "/home/user"


class CoPilotSandboxManager:
    """Manages e2b sandbox lifecycle for CoPilot sessions.

    Each session gets a single sandbox. The sandbox_id is stored in Redis
    so another pod can reconnect to it if the original pod dies.
    """

    def __init__(self) -> None:
        self._sandboxes: dict[str, Any] = {}  # session_id -> AsyncSandbox
        self._last_activity: dict[str, float] = {}  # session_id -> timestamp
        self._cleanup_task: asyncio.Task[None] | None = None
        config = Config()
        self._timeout: int = config.copilot_sandbox_timeout
        self._template: str = config.copilot_sandbox_template
        self._api_key: str = config.e2b_api_key

    async def get_or_create(self, session_id: str, user_id: str) -> Any:
        """Get existing sandbox or create a new one for this session.

        Args:
            session_id: CoPilot chat session ID.
            user_id: User ID for workspace sync.

        Returns:
            An e2b AsyncSandbox instance.
        """
        self._last_activity[session_id] = time.monotonic()

        # 1. Check in-memory cache
        if session_id in self._sandboxes:
            sandbox = self._sandboxes[session_id]
            if await _is_sandbox_alive(sandbox):
                return sandbox
            # Sandbox died — clean up stale reference
            del self._sandboxes[session_id]

        # 2. Check Redis for sandbox_id (cross-pod reconnection)
        sandbox = await self._try_reconnect_from_redis(session_id)
        if sandbox is not None:
            self._sandboxes[session_id] = sandbox
            return sandbox

        # 3. Create new sandbox
        sandbox = await self._create_sandbox(session_id, user_id)
        self._sandboxes[session_id] = sandbox
        await _store_sandbox_id_in_redis(session_id, sandbox.sandbox_id)

        # 4. Start cleanup task if not running
        self._ensure_cleanup_task()

        return sandbox

    async def dispose(self, session_id: str) -> None:
        """Persist workspace files to GCS, then kill sandbox.

        Args:
            session_id: CoPilot chat session ID.
        """
        sandbox = self._sandboxes.pop(session_id, None)
        self._last_activity.pop(session_id, None)

        if sandbox is None:
            return

        try:
            await sandbox.kill()
        except Exception as e:
            logger.warning(f"[E2B] Failed to kill sandbox for {session_id}: {e}")

        await _remove_sandbox_id_from_redis(session_id)
        logger.info(f"[E2B] Disposed sandbox for session {session_id}")

    async def dispose_all(self) -> None:
        """Dispose all sandboxes (called on processor shutdown)."""
        session_ids = list(self._sandboxes.keys())
        for sid in session_ids:
            await self.dispose(sid)
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _create_sandbox(self, session_id: str, user_id: str) -> Any:
        """Create a new e2b sandbox."""
        from e2b import AsyncSandbox

        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._template:
            kwargs["template"] = self._template
        if self._timeout:
            kwargs["timeout"] = self._timeout

        sandbox = await AsyncSandbox.create(**kwargs)
        logger.info(
            f"[E2B] Created sandbox {sandbox.sandbox_id} for session={session_id}, "
            f"user={user_id}"
        )
        return sandbox

    async def _try_reconnect_from_redis(self, session_id: str) -> Any | None:
        """Attempt to reconnect to a sandbox stored in Redis."""
        from e2b import AsyncSandbox

        sandbox_id = await _load_sandbox_id_from_redis(session_id)
        if not sandbox_id:
            return None

        try:
            sandbox = await AsyncSandbox.connect(
                sandbox_id=sandbox_id, api_key=self._api_key
            )
            logger.info(
                f"[E2B] Reconnected to sandbox {sandbox_id} for session={session_id}"
            )
            return sandbox
        except Exception as e:
            logger.warning(f"[E2B] Failed to reconnect to sandbox {sandbox_id}: {e}")
            await _remove_sandbox_id_from_redis(session_id)
            return None

    def _ensure_cleanup_task(self) -> None:
        """Start the idle cleanup background task if not already running."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.ensure_future(self._idle_cleanup_loop())

    async def _idle_cleanup_loop(self) -> None:
        """Periodically check for idle sandboxes and dispose them."""
        while True:
            await asyncio.sleep(60)
            if not self._sandboxes:
                continue
            now = time.monotonic()
            to_dispose: list[str] = []
            for sid, last in list(self._last_activity.items()):
                if now - last > self._timeout:
                    to_dispose.append(sid)
            for sid in to_dispose:
                logger.info(f"[E2B] Disposing idle sandbox for session {sid}")
                await self.dispose(sid)


# ------------------------------------------------------------------
# Module-level helpers (placed after classes that call them)
# ------------------------------------------------------------------


async def _is_sandbox_alive(sandbox: Any) -> bool:
    """Check if an e2b sandbox is still running."""
    try:
        result = await sandbox.commands.run("echo ok", timeout=5)
        return result.exit_code == 0
    except Exception:
        return False


async def _store_sandbox_id_in_redis(session_id: str, sandbox_id: str) -> None:
    """Store sandbox_id in Redis keyed by session_id."""
    try:
        from backend.data import redis as redis_client

        redis = redis_client.get_redis()
        key = f"{_REDIS_KEY_PREFIX}{session_id}"
        config = Config()
        ttl = max(config.copilot_sandbox_timeout * 2, 3600)  # At least 1h, 2x timeout
        await redis.set(key, sandbox_id, ex=ttl)
    except Exception as e:
        logger.warning(f"[E2B] Failed to store sandbox_id in Redis: {e}")


async def _load_sandbox_id_from_redis(session_id: str) -> str | None:
    """Load sandbox_id from Redis."""
    try:
        from backend.data import redis as redis_client

        redis = redis_client.get_redis()
        key = f"{_REDIS_KEY_PREFIX}{session_id}"
        value = await redis.get(key)
        return value.decode() if isinstance(value, bytes) else value
    except Exception as e:
        logger.warning(f"[E2B] Failed to load sandbox_id from Redis: {e}")
        return None


async def _remove_sandbox_id_from_redis(session_id: str) -> None:
    """Remove sandbox_id from Redis."""
    try:
        from backend.data import redis as redis_client

        redis = redis_client.get_redis()
        key = f"{_REDIS_KEY_PREFIX}{session_id}"
        await redis.delete(key)
    except Exception as e:
        logger.warning(f"[E2B] Failed to remove sandbox_id from Redis: {e}")
