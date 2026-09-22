import logging
from contextvars import Context

import pytest
from typesafe_sdk._core import transport as sdk_transport

from backend.blocks.typesafe._wire import capture_wire

WIRE_MESSAGE = "%(method)s %(url)s %(arrow)s headers=%(headers)s body=%(body)r"


def emit_wire(arrow: str, body: bytes) -> None:
    sdk_transport.logger.debug(
        WIRE_MESSAGE,
        {"method": "POST", "url": "/test", "arrow": arrow, "headers": {}, "body": body},
    )


def test_wire_capture_suppresses_bodies_from_unrelated_contexts(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.DEBUG):
        with capture_wire() as capture:
            emit_wire("->", b"own-private-request")
            Context().run(emit_wire, "<-", b"unrelated-private-response")
            emit_wire("<-", b"own-private-response")
    assert capture.request == b"own-private-request"
    assert capture.response == b"own-private-response"
    assert "private" not in caplog.text


def test_nested_captures_restore_the_outer_context():
    with capture_wire() as outer:
        emit_wire("->", b"outer-request")
        with capture_wire() as inner:
            emit_wire("->", b"inner-request")
            emit_wire("<-", b"inner-response")
        emit_wire("<-", b"outer-response")
    assert outer.request == b"outer-request"
    assert outer.response == b"outer-response"
    assert inner.request == b"inner-request"
    assert inner.response == b"inner-response"


def test_previously_disabled_logger_is_restored(monkeypatch: pytest.MonkeyPatch):
    logger = logging.getLogger("typesafe_sdk")
    monkeypatch.setattr(logger, "disabled", True)
    with capture_wire() as capture:
        emit_wire("->", b"request")
    assert capture.request == b"request"
    assert logger.disabled


def test_global_logging_disable_is_preserved_without_losing_capture():
    previous = logging.root.manager.disable
    try:
        logging.disable(logging.INFO)
        with capture_wire() as capture:
            emit_wire("->", b"request")
        assert capture.request == b"request"
        assert logging.root.manager.disable == logging.INFO
    finally:
        logging.disable(previous)
