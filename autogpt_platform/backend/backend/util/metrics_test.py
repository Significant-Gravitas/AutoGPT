"""Tests for the Sentry ``before_send`` filter in ``metrics.py``.

The filter exists to keep noise out of Sentry without dropping load-bearing
errors. We pin the "drop" list at the call-site level so a regression that
changes ``_before_send`` semantics — e.g. accidentally dropping all pika
ERRORs, or letting StreamLostError through — surfaces as a test failure
rather than a Sentry alert flood.
"""

from __future__ import annotations

import sys

from backend.util.metrics import (
    _FALKORDB_DRIVER_LOGGER,
    _FALKORDB_TEARDOWN_SIGNATURES,
    _PIKA_RECONNECT_LOGGERS,
    _PIKA_RECONNECT_SIGNATURES,
    _before_send,
)


def _log_event(logger: str, message: str) -> dict:
    return {
        "logger": logger,
        "logentry": {"formatted": message, "message": message},
        "level": "error",
    }


class _RedisConnectionError(Exception):
    """Exception whose *type* module mimics redis — so tests can tell the old
    module-based scoping (checked ``exc_type.__module__``) apart from the new
    traceback-based scoping (checks the raising frame's module)."""

    __module__ = "redis.exceptions"


def _exc_info_raised_from(module: str, message: str, exc_type=ConnectionError):
    """Build an ``exc_info`` triple whose traceback contains a frame in
    ``module`` and whose exception is ``exc_type`` — lets tests control both
    the traceback origin (what the new scoping checks) and the exception
    type's module (what the old module-based scoping checked)."""
    namespace: dict = {"__name__": module, "_Exc": exc_type}
    exec("def _raise(msg):\n    raise _Exc(msg)", namespace)
    try:
        namespace["_raise"](message)
    except exc_type:
        return sys.exc_info()
    raise AssertionError("expected exception")


# ---------- pika reconnect noise → dropped ----------


def test_pika_streamlost_error_dropped() -> None:
    """AUTOGPT-SERVER-6JC: ``StreamLostError: Transport indicated EOF`` from
    ``pika.adapters.blocking_connection`` is benign reconnect noise."""
    for logger in _PIKA_RECONNECT_LOGGERS:
        evt = _log_event(logger, "StreamLostError: Transport indicated EOF")
        assert _before_send(evt, hint={}) is None, logger


def test_pika_socket_eof_dropped() -> None:
    """AUTOGPT-SERVER-6JD: ``Socket EOF`` from
    ``pika.adapters.utils.io_services_utils`` reconnect path."""
    evt = _log_event(
        "pika.adapters.utils.io_services_utils",
        "Socket EOF on fd=12",
    )
    assert _before_send(evt, hint={}) is None


def test_pika_connection_lost_dropped() -> None:
    """AUTOGPT-SERVER-6JE: ``connection_lost`` callback firing during a
    rolling deploy."""
    evt = _log_event(
        "pika.adapters.base_connection",
        "connection_lost: Stream connection lost: ConnectionResetError(...)",
    )
    assert _before_send(evt, hint={}) is None


def test_pika_transport_eof_dropped() -> None:
    """AUTOGPT-SERVER-6JF: ``Transport indicated EOF`` standalone string."""
    evt = _log_event("pika.adapters.blocking_connection", "Transport indicated EOF")
    assert _before_send(evt, hint={}) is None


# ---------- pika ERRORs that must still get through ----------


def test_pika_authentication_failure_kept() -> None:
    """A real auth failure on the AMQP connection is load-bearing and must
    NOT be filtered out by the reconnect-noise rule."""
    evt = _log_event(
        "pika.adapters.blocking_connection",
        "ProbableAuthenticationError: Server closed connection",
    )
    assert _before_send(evt, hint={}) is not None


def test_pika_channel_declare_error_kept() -> None:
    """PRECONDITION_FAILED on a queue declare (e.g. quorum-type mismatch) is
    a real bug, not reconnect noise — must be kept."""
    evt = _log_event(
        "pika.adapters.blocking_connection",
        "PRECONDITION_FAILED - inequivalent arg 'x-queue-type' for queue 'foo'",
    )
    assert _before_send(evt, hint={}) is not None


def test_non_pika_logger_with_streamlost_kept() -> None:
    """Reconnect signatures are only suppressed for the three known pika
    loggers; any other logger emitting the same string is kept (e.g. a
    custom wrapper that re-raises)."""
    evt = _log_event(
        "backend.data.rabbitmq",
        "StreamLostError: Transport indicated EOF",
    )
    assert _before_send(evt, hint={}) is not None


def test_pika_reconnect_signatures_cover_all_four_known_patterns() -> None:
    """Sanity check: the signatures list still covers all four AUTOGPT-
    SERVER-6JC/6JD/6JE/6JF patterns from the prod Sentry issues."""
    expected = {
        "streamlosterror",
        "transport indicated eof",
        "socket eof",
        "connection_lost",
    }
    assert expected == set(_PIKA_RECONNECT_SIGNATURES)


# ---------- FalkorDB connection-teardown noise → dropped ----------


def test_falkordb_buffer_is_closed_log_dropped() -> None:
    """SENTRY-1387: ``Buffer is closed`` logged by graphiti-core's FalkorDB
    driver is a benign connection-teardown race — a query racing the cache
    eviction close or a per-request ``driver.close()``."""
    evt = _log_event(
        _FALKORDB_DRIVER_LOGGER,
        "Error executing FalkorDB query: Buffer is closed.\nMATCH (n) RETURN n\n{}",
    )
    assert _before_send(evt, hint={}) is None


def test_falkordb_connection_closed_by_server_log_dropped() -> None:
    """The sibling teardown message from the same race — the driver docstring
    pairs it with ``Buffer is closed``."""
    evt = _log_event(
        _FALKORDB_DRIVER_LOGGER,
        "Error executing FalkorDB query: Connection closed by server.",
    )
    assert _before_send(evt, hint={}) is None


def test_falkordb_buffer_is_closed_exc_from_graphiti_dropped() -> None:
    """If the re-raised teardown error reaches Sentry as an exception (a caller
    that doesn't swallow it), a ``Buffer is closed`` raised from the graphiti
    FalkorDB driver is still benign and must be dropped."""
    exc_info = _exc_info_raised_from(_FALKORDB_DRIVER_LOGGER, "Buffer is closed.")
    assert _before_send({"level": "error"}, hint={"exc_info": exc_info}) is None


def test_main_redis_connection_closed_exc_kept() -> None:
    """A genuine ``Connection closed by server`` from the platform's main redis
    (NOT raised from the graphiti driver) is a real incident and must NOT be
    swallowed by the teardown-noise rule. Uses a redis-module exception type so
    the OLD module-based scoping would have dropped it — this test fails if the
    traceback-based narrowing is reverted."""
    exc_info = _exc_info_raised_from(
        "redis.asyncio.connection",
        "Connection closed by server.",
        _RedisConnectionError,
    )
    assert _before_send({"level": "error"}, hint={"exc_info": exc_info}) is not None


def test_falkordb_real_query_error_kept() -> None:
    """A genuine Cypher/query failure from the same driver logger is load-
    bearing and must NOT be filtered out by the teardown-noise rule."""
    evt = _log_event(
        _FALKORDB_DRIVER_LOGGER,
        "Error executing FalkorDB query: Invalid input 'RETRUN': expected...",
    )
    assert _before_send(evt, hint={}) is not None


def test_falkordb_buffer_is_closed_from_other_logger_kept() -> None:
    """The teardown signatures are only suppressed for the graphiti FalkorDB
    driver logger; the same string from any other logger is kept."""
    evt = _log_event(
        "backend.data.redis_client",
        "Buffer is closed.",
    )
    assert _before_send(evt, hint={}) is not None


def test_falkordb_teardown_signatures_cover_known_patterns() -> None:
    """Sanity check: the teardown-signature list still covers both messages the
    graphiti FalkorDB driver docstring pairs together."""
    expected = {"buffer is closed", "connection closed by server"}
    assert expected == set(_FALKORDB_TEARDOWN_SIGNATURES)
