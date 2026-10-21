"""Do-not-merge probe (#15739 integration preview): where a turn's time goes
before the SDK engine starts.

Two processes, two installs, same ``perf-step <label> <ms> task=<task>`` lines
as ``sdk/perf_step_probe.py``:

* the API server (``install_route``, called at the end of ``routes.py``):
  every awaited step of ``stream_chat_post`` up to the enqueue, and inside
  ``schedule_chat_turn`` the turn slot, the message append (its session lock,
  cache read, next-sequence read, DB save and cache write) and the dispatch;
* the copilot executor (``install_executor``, called at the end of
  ``processor.py``): the ban check, the session load, the expert check, the
  allowance gate, the hire-setup wait, the engine choice and the trial cost
  context, i.e. everything between the cluster lock and "Using SDK service".
"""

import importlib
import inspect
import logging
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

_ROUTE_FUNCTIONS = (
    ("backend.api.features.chat.routes", "_validate_and_get_writable_session"),
    ("backend.api.features.chat.routes", "_apply_autopilot_mode"),
    ("backend.api.features.chat.routes", "enforce_model_entitlement"),
    ("backend.api.features.chat.routes", "enforce_payment_paywall"),
    ("backend.api.features.chat.routes", "_read_paywall_state"),
    ("backend.api.features.chat.routes", "is_turn_in_flight"),
    ("backend.api.features.chat.routes", "record_card_decisions"),
    ("backend.api.features.chat.routes", "enforce_platform_allowance"),
    ("backend.api.features.chat.routes", "claim_client_message"),
    ("backend.api.features.chat.routes", "accept_client_message"),
    ("backend.api.features.chat.routes", "schedule_chat_turn"),
    ("backend.copilot.executor.utils", "dispatch_turn"),
    ("backend.copilot.model", "append_and_save_message"),
    ("backend.copilot.model", "get_chat_session"),
    ("backend.copilot.model", "_save_session_to_db"),
    ("backend.copilot.model", "cache_chat_session"),
)
_ROUTE_CONTEXTS = (
    ("backend.copilot.executor.utils", "acquire_turn_slot"),
    ("backend.copilot.model", "_get_session_lock"),
)
_EXECUTOR_FUNCTIONS = (
    ("backend.copilot.executor.processor", "is_user_banned"),
    ("backend.copilot.executor.processor", "_normalize_private_expert_session_tenancy"),
    ("backend.copilot.executor.processor", "refusal_for"),
    ("backend.copilot.executor.processor", "bill_to_credits"),
    ("backend.copilot.executor.processor", "_wait_for_expert_setup"),
    ("backend.copilot.executor.processor", "resolve_use_sdk"),
    ("backend.copilot.model", "get_chat_session"),
)
_EXECUTOR_CONTEXTS = (
    ("backend.copilot.executor.processor", "trial_cost_context"),
)
_installed: set[str] = set()


def install_route() -> None:
    _install("route", _ROUTE_FUNCTIONS, _ROUTE_CONTEXTS)


def install_executor() -> None:
    _install("pre", _EXECUTOR_FUNCTIONS, _EXECUTOR_CONTEXTS)


def _install(prefix: str, functions, contexts) -> None:
    if prefix in _installed:
        return
    _installed.add(prefix)
    for module_name, attribute in functions:
        _wrap_function(importlib.import_module(module_name), attribute, prefix)
    for module_name, attribute in contexts:
        _wrap_context(importlib.import_module(module_name), attribute, prefix)
    logger.info("perf-step %s probe installed", prefix)


def _wrap_function(module, attribute: str, prefix: str) -> None:
    original = getattr(module, attribute, None)
    if original is None or not inspect.iscoroutinefunction(original):
        logger.info("perf-step skip %s:%s (missing or not async)", prefix, attribute)
        return
    label = f"{prefix}:{attribute.lstrip('_')}"

    async def timed(*args, **kwargs):
        started = time.perf_counter()
        try:
            return await original(*args, **kwargs)
        finally:
            _log(label, started)

    setattr(module, attribute, timed)


def _wrap_context(module, attribute: str, prefix: str) -> None:
    original = getattr(module, attribute, None)
    if original is None:
        logger.info("perf-step skip %s:%s (missing)", prefix, attribute)
        return
    label = f"{prefix}:{attribute.lstrip('_')}.enter"

    @asynccontextmanager
    async def timed(*args, **kwargs):
        started = time.perf_counter()
        async with original(*args, **kwargs) as value:
            _log(label, started)
            yield value

    setattr(module, attribute, timed)


def _log(label: str, started: float) -> None:
    import asyncio

    try:
        task = asyncio.current_task()
        task_name = task.get_name() if task is not None else "-"
    except RuntimeError:
        task_name = "-"
    logger.info(
        "perf-step %s %d task=%s",
        label,
        round((time.perf_counter() - started) * 1000),
        task_name,
    )
