import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Protocol, TypedDict, cast

from pydantic import BaseModel
from typesafe_sdk._core import transport as sdk_transport

_SDK_WIRE_MESSAGE = "%(method)s %(url)s %(arrow)s headers=%(headers)s body=%(body)r"


class WireCapture(BaseModel):
    request: bytes | None = None
    response: bytes | None = None
    request_id: str = ""


@contextmanager
def capture_wire() -> Iterator[WireCapture]:
    """Capture SDK 0.6.0 _log_wire records without changing global logging."""
    capture = WireCapture()
    token = _current_capture.set(capture)
    try:
        _enable_capture()
        try:
            yield capture
        finally:
            _disable_capture()
    finally:
        _current_capture.reset(token)


class _WireArguments(TypedDict):
    arrow: str
    body: bytes | None
    headers: dict[str, str]


class _SDKTransportLogging(Protocol):
    logger: logging.Logger


class _CaptureLogger(logging.Logger):
    def __init__(self, delegate: logging.Logger):
        super().__init__(delegate.name)
        self.delegate = delegate

    def isEnabledFor(self, level: int) -> bool:
        if level == logging.DEBUG and _current_capture.get() is not None:
            return True
        return self.delegate.isEnabledFor(level)

    def handle(self, record: logging.LogRecord) -> None:
        if record.msg != _SDK_WIRE_MESSAGE:
            self.delegate.handle(record)
            return
        capture = _current_capture.get()
        if capture is not None:
            args = cast(_WireArguments, record.args)
            if args["arrow"] == "->":
                capture.request = args["body"]
            elif args["arrow"] == "<-":
                capture.response = args["body"]
                capture.request_id = args["headers"].get("x-typesafe-request-id", "")
        # Wire bodies stay in this context and never reach logging handlers.


_current_capture: ContextVar[WireCapture | None] = ContextVar(
    "jev_wire_capture", default=None
)
_lock = Lock()
_active = 0
_previous_logger: logging.Logger | None = None
_transport_logging = cast(_SDKTransportLogging, sdk_transport)


def _enable_capture() -> None:
    global _active, _previous_logger
    with _lock:
        if _active == 0:
            _previous_logger = _transport_logging.logger
            _transport_logging.logger = _CaptureLogger(_previous_logger)
        _active += 1


def _disable_capture() -> None:
    global _active, _previous_logger
    with _lock:
        _active -= 1
        if _active == 0 and _previous_logger is not None:
            _transport_logging.logger = _previous_logger
            _previous_logger = None
