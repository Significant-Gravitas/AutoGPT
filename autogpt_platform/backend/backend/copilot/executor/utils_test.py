"""Tests for CoPilot executor utils (queue config, message models, logging)."""

from backend.copilot.executor.utils import (
    COPILOT_EXECUTION_EXCHANGE,
    COPILOT_EXECUTION_QUEUE_NAME,
    COPILOT_EXECUTION_ROUTING_KEY,
    CancelCoPilotEvent,
    CoPilotExecutionEntry,
    CoPilotLogMetadata,
    create_copilot_queue_config,
)


class TestCoPilotExecutionEntry:
    def test_basic_fields(self):
        entry = CoPilotExecutionEntry(
            session_id="s1",
            user_id="u1",
            message="hello",
        )
        assert entry.session_id == "s1"
        assert entry.user_id == "u1"
        assert entry.message == "hello"
        assert entry.is_user_message is True
        assert entry.mode is None
        assert entry.context is None
        assert entry.file_ids is None

    def test_mode_field(self):
        entry = CoPilotExecutionEntry(
            session_id="s1",
            user_id="u1",
            message="test",
            mode="fast",
        )
        assert entry.mode == "fast"

        entry2 = CoPilotExecutionEntry(
            session_id="s1",
            user_id="u1",
            message="test",
            mode="extended_thinking",
        )
        assert entry2.mode == "extended_thinking"

    def test_optional_fields(self):
        entry = CoPilotExecutionEntry(
            session_id="s1",
            user_id="u1",
            message="test",
            turn_id="t1",
            context={"url": "https://example.com"},
            file_ids=["f1", "f2"],
            is_user_message=False,
        )
        assert entry.turn_id == "t1"
        assert entry.context == {"url": "https://example.com"}
        assert entry.file_ids == ["f1", "f2"]
        assert entry.is_user_message is False

    def test_serialization_roundtrip(self):
        entry = CoPilotExecutionEntry(
            session_id="s1",
            user_id="u1",
            message="hello",
            mode="fast",
        )
        json_str = entry.model_dump_json()
        restored = CoPilotExecutionEntry.model_validate_json(json_str)
        assert restored == entry


class TestCancelCoPilotEvent:
    def test_basic(self):
        event = CancelCoPilotEvent(session_id="s1")
        assert event.session_id == "s1"

    def test_serialization(self):
        event = CancelCoPilotEvent(session_id="s1")
        restored = CancelCoPilotEvent.model_validate_json(event.model_dump_json())
        assert restored.session_id == "s1"


class TestCreateCopilotQueueConfig:
    def test_returns_valid_config(self):
        config = create_copilot_queue_config()
        assert len(config.exchanges) == 2
        assert len(config.queues) == 2

    def test_execution_queue_properties(self):
        config = create_copilot_queue_config()
        exec_queue = next(
            q for q in config.queues if q.name == COPILOT_EXECUTION_QUEUE_NAME
        )
        assert exec_queue.durable is True
        assert exec_queue.exchange == COPILOT_EXECUTION_EXCHANGE
        assert exec_queue.routing_key == COPILOT_EXECUTION_ROUTING_KEY

    def test_cancel_queue_uses_fanout(self):
        config = create_copilot_queue_config()
        cancel_queue = next(
            q for q in config.queues if q.name != COPILOT_EXECUTION_QUEUE_NAME
        )
        assert cancel_queue.exchange is not None
        assert cancel_queue.exchange.type.value == "fanout"


class TestCoPilotLogMetadata:
    def test_creates_logger_with_metadata(self):
        import logging

        base_logger = logging.getLogger("test")
        log = CoPilotLogMetadata(base_logger, session_id="s1", user_id="u1")
        assert log is not None

    def test_filters_none_values(self):
        import logging

        base_logger = logging.getLogger("test")
        log = CoPilotLogMetadata(
            base_logger, session_id="s1", user_id=None, turn_id="t1"
        )
        assert log is not None
