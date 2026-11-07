"""Normalize prisma-client-py 0.15.0 error payloads that crash error mapping.

Pinned prisma 0.15.0 is the last release (upstream is archived; see
https://github.com/RobertCraigie/prisma-client-py/issues/1070).
The query engine serializes P1017 ("Server has closed the connection") as a
``user_facing_error`` object with ``meta: null``. ``handle_response_errors``
then calls ``.get`` on that null ``meta``. The upstream guard catches only
``KeyError``/``TypeError``, so an ``AttributeError`` escapes instead of the
intended ``DataError``.
"""

import logging
import sys
from functools import wraps

import prisma
from prisma.engine import _http, _query
from prisma.engine import utils as engine_utils

logger = logging.getLogger(__name__)

_PINNED_VERSION = "0.15.0"
_PATCH_ATTR = "_autogpt_null_meta_patch"
_ENGINE_MODULES = (_http, _query, engine_utils)


def patch_handle_response_errors() -> None:
    if prisma.__version__ != _PINNED_VERSION:
        logger.warning(
            f"Skipping prisma error-payload patch: installed prisma "
            f"{prisma.__version__} is not {_PINNED_VERSION}"
        )
        return

    original = engine_utils.handle_response_errors
    if getattr(original, _PATCH_ATTR, False):
        return

    wrapped = _wrap_handle_response_errors(original)
    _rebind_handle_response_errors(original, wrapped)


def _wrap_handle_response_errors(original):
    @wraps(original)
    def handle_response_errors(resp, data):
        return original(resp, _normalize_errors(data))

    setattr(handle_response_errors, _PATCH_ATTR, True)
    return handle_response_errors


def _rebind_handle_response_errors(original, wrapped) -> None:
    for module in _ENGINE_MODULES:
        if getattr(module, "handle_response_errors", None) is original:
            module.handle_response_errors = wrapped

    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("prisma"):
            continue
        if getattr(module, "handle_response_errors", None) is original:
            module.handle_response_errors = wrapped


def _normalize_errors(data):
    if type(data) is not list:
        return data
    return [_normalize_error(error) for error in data]


def _normalize_error(error):
    if type(error) is not dict:
        return error
    user_facing = error.get("user_facing_error", {})
    if user_facing is None:
        return {**error, "user_facing_error": {}}
    if type(user_facing) is not dict:
        return error
    if user_facing.get("meta") is not None:
        return error
    return {**error, "user_facing_error": {**user_facing, "meta": {}}}


patch_handle_response_errors()
