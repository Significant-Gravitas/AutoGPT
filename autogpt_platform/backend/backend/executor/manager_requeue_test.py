import json
import threading
from unittest.mock import Mock, patch

import pytest
from pika import BasicProperties
from pika.spec import Basic

from backend.data.execution import GraphExecutionEntry, UserContext
from backend.executor.manager import ExecutionManager


class TestExecutionManagerRequeue:
    @pytest.fixture
    def mock_manager(self):
        manager = Mock(spec=ExecutionManager)
        manager.service_name = "test-executor"
        manager.stop_consuming = threading.Event()
        manager.active_graph_runs = {}
        manager.pool_size = 10
        manager.run_client = Mock()
        manager.run_client.get_channel = Mock()
        manager._cleanup_completed_runs = Mock()
        manager._run_graph = Mock()

        manager._handle_run_message = ExecutionManager._handle_run_message.__get__(
            manager, ExecutionManager
        )

        return manager

    @pytest.fixture
    def mock_channel(self):
        channel = Mock()
        channel.connection = Mock()
        channel.connection.add_callback_threadsafe = Mock()
        channel.basic_ack = Mock()
        channel.basic_nack = Mock()
        return channel

    @pytest.fixture
    def mock_method(self):
        method = Mock(spec=Basic.Deliver)
        method.delivery_tag = 123
        return method

    @pytest.fixture
    def mock_properties(self):
        return Mock(spec=BasicProperties)

    def test_malformed_message_not_requeued(
        self, mock_manager, mock_channel, mock_method, mock_properties
    ):
        mock_manager.run_client.get_channel.return_value = mock_channel
        malformed_body = b"not valid json {{"

        mock_manager._handle_run_message(
            mock_channel, mock_method, mock_properties, malformed_body
        )

        mock_channel.connection.add_callback_threadsafe.assert_called_once()
        callback = mock_channel.connection.add_callback_threadsafe.call_args[0][0]
        callback()
        mock_channel.basic_nack.assert_called_once_with(123, requeue=False)
        mock_channel.basic_ack.assert_not_called()

    def test_invalid_schema_message_not_requeued(
        self, mock_manager, mock_channel, mock_method, mock_properties
    ):
        mock_manager.run_client.get_channel.return_value = mock_channel
        invalid_schema_body = json.dumps({"some": "data"}).encode()

        mock_manager._handle_run_message(
            mock_channel, mock_method, mock_properties, invalid_schema_body
        )

        mock_channel.connection.add_callback_threadsafe.assert_called_once()
        callback = mock_channel.connection.add_callback_threadsafe.call_args[0][0]
        callback()
        mock_channel.basic_nack.assert_called_once_with(123, requeue=False)
        mock_channel.basic_ack.assert_not_called()

    def test_duplicate_execution_not_requeued(
        self, mock_manager, mock_channel, mock_method, mock_properties
    ):
        mock_manager.run_client.get_channel.return_value = mock_channel
        graph_exec_id = "test-exec-123"
        valid_entry = GraphExecutionEntry(
            graph_exec_id=graph_exec_id,
            user_id="user-123",
            graph_id="graph-456",
            graph_version=1,
            user_context=UserContext(timezone="UTC"),
        )
        valid_body = valid_entry.model_dump_json().encode()
        mock_manager.active_graph_runs[graph_exec_id] = Mock()

        mock_manager._handle_run_message(
            mock_channel, mock_method, mock_properties, valid_body
        )

        mock_channel.connection.add_callback_threadsafe.assert_called_once()
        callback = mock_channel.connection.add_callback_threadsafe.call_args[0][0]
        callback()
        mock_channel.basic_nack.assert_called_once_with(123, requeue=False)
        mock_channel.basic_ack.assert_not_called()

    def test_shutdown_message_requeued(
        self, mock_manager, mock_channel, mock_method, mock_properties
    ):
        mock_manager.run_client.get_channel.return_value = mock_channel
        mock_manager.stop_consuming.set()

        valid_entry = GraphExecutionEntry(
            graph_exec_id="test-exec-456",
            user_id="user-123",
            graph_id="graph-456",
            graph_version=1,
            user_context=UserContext(timezone="UTC"),
        )
        valid_body = valid_entry.model_dump_json().encode()

        mock_manager._handle_run_message(
            mock_channel, mock_method, mock_properties, valid_body
        )

        mock_channel.connection.add_callback_threadsafe.assert_called_once()
        callback = mock_channel.connection.add_callback_threadsafe.call_args[0][0]
        callback()
        mock_channel.basic_nack.assert_called_once_with(123, requeue=True)
        mock_channel.basic_ack.assert_not_called()

    def test_pool_full_message_requeued(
        self, mock_manager, mock_channel, mock_method, mock_properties
    ):
        mock_manager.run_client.get_channel.return_value = mock_channel
        mock_manager.pool_size = 2
        mock_manager.active_graph_runs = {"exec-1": Mock(), "exec-2": Mock()}

        valid_entry = GraphExecutionEntry(
            graph_exec_id="test-exec-789",
            user_id="user-123",
            graph_id="graph-456",
            graph_version=1,
            user_context=UserContext(timezone="UTC"),
        )
        valid_body = valid_entry.model_dump_json().encode()

        mock_manager._handle_run_message(
            mock_channel, mock_method, mock_properties, valid_body
        )

        mock_channel.connection.add_callback_threadsafe.assert_called_once()
        callback = mock_channel.connection.add_callback_threadsafe.call_args[0][0]
        callback()
        mock_channel.basic_nack.assert_called_once_with(123, requeue=True)
        mock_channel.basic_ack.assert_not_called()

    def test_successful_message_acknowledged(
        self, mock_manager, mock_channel, mock_method, mock_properties
    ):
        mock_manager.run_client.get_channel.return_value = mock_channel

        valid_entry = GraphExecutionEntry(
            graph_exec_id="test-exec-999",
            user_id="user-123",
            graph_id="graph-456",
            graph_version=1,
            user_context=UserContext(timezone="UTC"),
        )
        valid_body = valid_entry.model_dump_json().encode()
        mock_manager._run_graph.return_value = (None, None)

        with patch.object(ExecutionManager, "_run_graph", return_value=(None, None)):
            mock_manager._handle_run_message(
                mock_channel, mock_method, mock_properties, valid_body
            )

        assert mock_manager.active_graph_runs.get("test-exec-999") is not None
