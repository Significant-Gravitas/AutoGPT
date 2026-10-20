"""The cancel consumer must consume the per-pod queue it just declared."""

from unittest.mock import MagicMock

from backend.executor.manager import ExecutionManager
from backend.executor.utils import GRAPH_EXECUTION_CANCEL_EXCHANGE


def test_cancel_consumer_consumes_the_queue_it_just_declared() -> None:
    """Pins the consumer to the per-pod queue the declare helper returned.

    Re-pinning ``basic_consume`` to a fixed name restores the original defect —
    one shared queue, so the broker hands each cancel to one arbitrary pod —
    and no other test in the suite would notice.
    """
    manager = ExecutionManager()
    channel = MagicMock()
    manager._cancel_client = MagicMock(is_ready=True)
    manager._cancel_client.get_channel.return_value = channel
    channel.start_consuming.side_effect = lambda: manager.stop_consuming.set()

    ExecutionManager._consume_execution_cancel.__wrapped__(manager)

    declared = channel.queue_declare.call_args.kwargs["queue"]
    assert declared.startswith(f"{GRAPH_EXECUTION_CANCEL_EXCHANGE.name}.instance.")
    assert channel.basic_consume.call_args.kwargs["queue"] == declared
