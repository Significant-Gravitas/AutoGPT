"""Warn when every graph worker is parked on a slow LLM call.

Raising the per-call deadline to 600s means one degraded provider can occupy
every worker for ten minutes. Nothing else reports that state: the executor's
utilization gauge shows the workers as busy, which is indistinguishable from
healthy load.
"""

import contextlib
import logging
import threading
import time
from collections import Counter as _Counter
from typing import Iterator

from prometheus_client import Counter, Gauge
from pydantic import BaseModel

from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# A worker holds one graph run, so distinct graph runs with a long call in
# flight is the closest proxy for "workers occupied" reachable from this seam.
SATURATION_CALL_AGE_SECONDS = 60.0

# Saturation is reached by calls AGEING, not by new ones arriving, so a fully
# stuck fleet would never re-evaluate on its own.
_TICK_SECONDS = 15.0

saturated_workers_gauge = Gauge(
    "llm_saturated_graph_workers",
    "Graph runs blocked on an LLM call older than the saturation age",
)
saturation_episodes_counter = Counter(
    "llm_worker_saturation_episodes_total",
    "Times every graph worker was simultaneously blocked on a slow LLM call",
)


@contextlib.contextmanager
def track_llm_call(
    *, graph_exec_id: str | None, provider: str, now: float | None = None
) -> Iterator[None]:
    """Register one in-flight LLM call for the duration of the block.

    ``graph_exec_id`` is None for callers that do not occupy a graph worker
    (copilot, dream, briefing); those are excluded from the count.
    """
    call = _Call(
        graph_exec_id=graph_exec_id,
        provider=provider,
        started=now if now is not None else time.monotonic(),
    )
    token = object()
    with _lock:
        _in_flight[id(token)] = call
        _ensure_ticker()
    try:
        yield
    finally:
        with _lock:
            _in_flight.pop(id(token), None)
        # Keep `token` alive to here: id() is only unique while it is.
        del token


def evaluate_saturation(now: float | None = None) -> int:
    """Refresh the gauge; log once on entering saturation. Returns the count."""
    global _saturated
    now = now if now is not None else time.monotonic()
    cutoff = now - SATURATION_CALL_AGE_SECONDS

    with _lock:
        slow: dict[str, list[_Call]] = {}
        for call in _in_flight.values():
            if call.graph_exec_id is None or call.started > cutoff:
                continue
            slow.setdefault(call.graph_exec_id, []).append(call)

        count = len(slow)
        saturated = count > 0 and count >= settings.config.num_graph_workers
        episode: str | None = None
        if saturated and not _saturated:
            calls = [c for group in slow.values() for c in group]
            mix = _Counter(c.provider for c in calls).most_common()
            episode = (
                f"LLM worker saturation: all {count} graph workers on executor "
                f"{_executor_id} are blocked on an LLM call older than "
                f"{SATURATION_CALL_AGE_SECONDS:.0f}s. "
                f"providers={', '.join(f'{p}={n}' for p, n in mix)} "
                f"longest_call_age={now - min(c.started for c in calls):.0f}s"
            )
        _saturated = saturated

    saturated_workers_gauge.set(count)
    if episode:
        saturation_episodes_counter.inc()
        logger.warning(episode)
    return count


def set_executor_id(executor_id: str) -> None:
    """Name the pod in the warning; without it an alert can't be routed."""
    global _executor_id
    _executor_id = executor_id


class _Call(BaseModel, frozen=True):
    graph_exec_id: str | None
    provider: str
    started: float


def _ensure_ticker() -> None:
    """Start the re-evaluation thread on first use; caller holds ``_lock``."""
    global _ticker
    if _ticker is not None:
        return
    _ticker = threading.Thread(target=_tick_forever, daemon=True, name="llm-saturation")
    _ticker.start()


def _tick_forever() -> None:
    while True:
        time.sleep(_TICK_SECONDS)
        try:
            evaluate_saturation()
        except Exception:  # a metrics thread must never take the process down
            logger.exception("LLM saturation check failed")


_lock = threading.Lock()
_in_flight: dict[int, _Call] = {}
_saturated = False
_executor_id = "unknown"
_ticker: threading.Thread | None = None
