import logging

"""
_logging.py
websocket - WebSocket client library for Python

Copyright 2022 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

_logger = logging.getLogger('websocket')
try:
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

_logger.addHandler(NullHandler())

_traceEnabled = False

__all__ = ["enableTrace", "dump", "error", "warning", "debug", "trace",
           "isEnabledForError", "isEnabledForDebug", "isEnabledForTrace"]


def enableTrace(traceable, handler=logging.StreamHandler(), level="DEBUG"):
    """
    Turn on/off the traceability.

    Parameters
    ----------
    traceable: bool
        If set to True, traceability is enabled.
    """
    global _traceEnabled
    _traceEnabled = traceable
    if traceable:
        _logger.addHandler(handler)
        _logger.setLevel(getattr(logging, level))


def dump(title, message):
    if _traceEnabled:
        _logger.debug("--- " + title + " ---")
        _logger.debug(message)
        _logger.debug("-----------------------")


def error(msg):
    _logger.error(msg)


def warning(msg):
    _logger.warning(msg)


def debug(msg):
    _logger.debug(msg)


def info(msg):
    _logger.info(msg)


def trace(msg):
    if _traceEnabled:
        _logger.debug(msg)


def isEnabledForError():
    return _logger.isEnabledFor(logging.ERROR)


def isEnabledForDebug():
    return _logger.isEnabledFor(logging.DEBUG)


def isEnabledForTrace():
    return _traceEnabled
