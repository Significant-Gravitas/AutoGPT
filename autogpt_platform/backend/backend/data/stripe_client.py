"""One door for every Stripe SDK call.

The SDK's native ``*_async`` methods are used directly (no thread pool), a
timeout well under the SDK's 80s default is applied, and every call is
counted by resource, method and outcome. Nothing about Stripe semantics is
changed: the original exception is always re-raised unchanged, so callers
that branch on ``stripe.InvalidRequestError`` etc. keep working.

Retries are deliberately left to the SDK (``stripe.max_network_retries``),
which only retries with idempotency keys. Wrapping this in our own retry
decorator would re-issue non-idempotent writes such as PaymentIntent.create.

On timeout the coroutine is cancelled on our side and a
``stripe.APIConnectionError`` is raised, the same type the SDK uses for its own
network timeouts, so existing ``except stripe.StripeError`` handling keeps
working. For a write that Stripe may
already have applied (a charge, a refund) that leaves the caller unsure, which
is inherent to any client-side timeout on a non-idempotent call; the outcome is
recorded as ``timeout`` and logged at error level so it is never silent.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import ParamSpec, TypeVar

import stripe

from backend.monitoring.instrumentation import record_stripe_request

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")
StripeResource = TypeVar("StripeResource", bound=stripe.StripeObject)

# Stripe's own p99 is low single-digit seconds; 80s (the SDK default) only
# ever means a stalled socket, and holding a request that long serves nobody.
DEFAULT_TIMEOUT_SECONDS = 30.0


async def stripe_call(
    fn: Callable[P, Awaitable[T]], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Await ``fn(*args, **kwargs)`` with the default timeout, recording the outcome.

    ``fn`` must be one of the SDK's ``*_async`` methods, e.g.
    ``stripe.Subscription.list_async``. The original exception is re-raised.
    """
    return await _call(DEFAULT_TIMEOUT_SECONDS, fn, *args, **kwargs)


async def stripe_call_timeout(
    timeout_seconds: float,
    fn: Callable[P, Awaitable[T]],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """As :func:`stripe_call`, with an explicit timeout. Rarely needed."""
    return await _call(timeout_seconds, fn, *args, **kwargs)


async def stripe_list_items(
    page: stripe.ListObject[StripeResource],
) -> AsyncIterator[StripeResource]:
    """Keep subsequent SDK list requests inside the same timeout/metrics boundary."""
    while True:
        for item in page.data:
            yield item
        if not page.has_more:
            return
        if not page.data:
            raise ValueError("Stripe returned an empty page with more results")
        page = await stripe_call(page.next_page_async)


async def _call(
    timeout_seconds: float,
    fn: Callable[P, Awaitable[T]],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    resource, method = _labels(fn)
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        record_stripe_request(
            resource, method, "timeout", time.perf_counter() - started
        )
        logger.error(
            f"Stripe {resource}.{method} exceeded {timeout_seconds}s and was cancelled"
        )
        raise stripe.APIConnectionError(
            f"Stripe {resource}.{method} exceeded {timeout_seconds}s and was cancelled.",
            should_retry=False,
        ) from exc
    except Exception as exc:
        record_stripe_request(
            resource, method, _outcome(exc), time.perf_counter() - started
        )
        raise
    record_stripe_request(resource, method, "ok", time.perf_counter() - started)
    return result


def _labels(fn: Callable[..., object]) -> tuple[str, str]:
    """``stripe.Subscription.list_async`` -> ("Subscription", "list")."""
    # Test doubles (AsyncMock) refuse to synthesise dunder attributes, so a
    # default is needed here; this is a label lookup, not type dispatch.
    qual = getattr(fn, "__qualname__", "") or ""
    resource, _, method = qual.rpartition(".")
    resource = resource.rsplit(".", 1)[-1] or "unknown"
    return resource, method.removesuffix("_async") or "unknown"


def _outcome(exc: BaseException) -> str:
    if isinstance(exc, stripe.RateLimitError):
        return "rate_limited"
    if isinstance(exc, stripe.APIConnectionError):
        return "connection_error"
    return "api_error"
