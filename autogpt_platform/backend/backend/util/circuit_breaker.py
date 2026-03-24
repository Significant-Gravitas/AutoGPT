"""Lightweight circuit breaker for external service calls (e.g. LLM providers).

The breaker tracks consecutive failures per *key* (typically a provider name).
When the failure count exceeds the threshold the circuit **opens** and all
subsequent calls are rejected immediately with ``CircuitOpenError`` until the
``recovery_timeout`` has elapsed.  After the timeout the circuit enters a
**half-open** state, allowing a single probe call through:

* If the probe succeeds → circuit **closes** (reset).
* If the probe fails   → circuit **re-opens** for another timeout period.

Thread-safety: guarded by a ``threading.Lock`` so it can be shared across
asyncio tasks running on the same event loop and across sync call sites.

Usage::

    from backend.util.circuit_breaker import circuit_breaker_registry

    breaker = circuit_breaker_registry.get("openai")
    breaker.pre_call()        # raises CircuitOpenError if open
    try:
        result = await external_call()
        breaker.record_success()
    except Exception:
        breaker.record_failure()
        raise
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, key: str, retry_after: float):
        self.key = key
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker open for '{key}'. "
            f"Retry after {retry_after:.1f}s."
        )


class CircuitBreaker:
    """Per-key circuit breaker."""

    def __init__(
        self,
        key: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.key = key
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._effective_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pre_call(self) -> None:
        """Call before making the external request.

        Raises ``CircuitOpenError`` if the circuit is open and the recovery
        timeout has not yet elapsed.
        """
        with self._lock:
            state = self._effective_state()
            if state == CircuitState.OPEN:
                retry_after = self.recovery_timeout - (
                    time.monotonic() - self._last_failure_time
                )
                raise CircuitOpenError(self.key, max(retry_after, 0))
            # CLOSED or HALF_OPEN → allow the call through
            if state == CircuitState.HALF_OPEN:
                logger.info(
                    f"Circuit breaker '{self.key}' half-open: allowing probe call"
                )

    def record_success(self) -> None:
        """Record a successful call – resets the breaker."""
        with self._lock:
            if self._failure_count > 0:
                logger.info(
                    f"Circuit breaker '{self.key}' reset after success "
                    f"(was at {self._failure_count} failures)"
                )
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call – may trip the breaker open."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    logger.warning(
                        f"Circuit breaker '{self.key}' OPEN after "
                        f"{self._failure_count} consecutive failures"
                    )
                self._state = CircuitState.OPEN

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _effective_state(self) -> CircuitState:
        """Return the effective state, transitioning OPEN → HALF_OPEN when
        the recovery timeout has elapsed.  Must be called under ``_lock``."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state


class CircuitBreakerRegistry:
    """Global registry of per-key circuit breakers."""

    def __init__(
        self,
        default_failure_threshold: int = 5,
        default_recovery_timeout: float = 60.0,
    ):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        self._default_failure_threshold = default_failure_threshold
        self._default_recovery_timeout = default_recovery_timeout

    def get(self, key: str) -> CircuitBreaker:
        with self._lock:
            if key not in self._breakers:
                self._breakers[key] = CircuitBreaker(
                    key,
                    failure_threshold=self._default_failure_threshold,
                    recovery_timeout=self._default_recovery_timeout,
                )
            return self._breakers[key]


# Module-level singleton used by LLM call paths and any other external service.
circuit_breaker_registry = CircuitBreakerRegistry(
    default_failure_threshold=5,
    default_recovery_timeout=60.0,
)
