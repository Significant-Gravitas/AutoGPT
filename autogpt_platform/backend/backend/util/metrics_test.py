"""Tests for the Sentry ``before_send`` filter in ``metrics.py``.

The filter exists to keep noise out of Sentry without dropping load-bearing
errors. We pin the "drop" list at the call-site level so a regression that
changes ``_before_send`` semantics — e.g. accidentally dropping all pika
ERRORs, or letting StreamLostError through — surfaces as a test failure
rather than a Sentry alert flood.
"""

from __future__ import annotations

import json
import sys

from sentry_sdk.consts import DEFAULT_OPTIONS
from sentry_sdk.utils import event_from_exception

from backend.util.metrics import (
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


def test_before_send_scrubs_secrets_from_actual_exception_event() -> None:
    secrets = {
        "id": "FAKE-ID-SECRET-991",
        "access": "FAKE-ACCESS-SECRET-992",
        "refresh": "FAKE-REFRESH-SECRET-993",
        "bearer": "FAKE-BEARER-SECRET-994",
        "provider": "FAKE-PROVIDER-SECRET-995",
        "device_code": "FAKE-DEVICE-CODE-996",
    }
    payload = {
        "tokens": {
            "id_token": secrets["id"],
            "access_token": secrets["access"],
            "refresh_token": secrets["refresh"],
        },
        "safe": "safe-frame-value",
    }
    encoded = json.dumps(payload).encode()
    assert encoded

    try:
        raise RuntimeError("materialization failed")
    except RuntimeError:
        event, hint = event_from_exception(
            sys.exc_info(),
            client_options=DEFAULT_OPTIONS,
        )

    event.update(
        {
            "breadcrumbs": {
                "values": [
                    {
                        "data": {
                            "message": f"Authorization: Bearer {secrets['bearer']}",
                            "safe": "safe-breadcrumb-value",
                        }
                    }
                ]
            },
            "contexts": {
                "codex": {
                    "provider_state": secrets["provider"],
                    "user_code": secrets["device_code"],
                    "safe": "safe-context-value",
                }
            },
            "request": {
                "data": {"access_token": secrets["access"]},
                "headers": {"Authorization": f"Bearer {secrets['bearer']}"},
            },
        }
    )

    scrubbed = _before_send(event, hint)

    assert scrubbed is event
    serialized = json.dumps(scrubbed, default=str)
    for name, secret in secrets.items():
        assert secret not in serialized, name
    assert "safe-frame-value" in serialized
    assert "safe-breadcrumb-value" in serialized
    assert "safe-context-value" in serialized
