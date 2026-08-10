from __future__ import annotations

import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "probe.py"
SPEC = importlib.util.spec_from_file_location("single_container_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ChunkedStream(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        return super().read(min(size, 1) if size >= 0 else 1)


class DuplexStream:
    def __init__(self, response: bytes) -> None:
        self.reader = io.BytesIO(response)
        self.written = io.BytesIO()

    def read(self, size: int = -1) -> bytes:
        return self.reader.read(size)

    def readline(self, size: int = -1) -> bytes:
        return self.reader.readline(size)

    def write(self, value: bytes) -> int:
        return self.written.write(value)


class FakeConnection:
    def __init__(self, response: bytes) -> None:
        self.stream = DuplexStream(response)
        self.response = response
        self.sent = b""

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def makefile(self, _mode: str, buffering: int) -> DuplexStream:
        del buffering
        return self.stream

    def sendall(self, value: bytes) -> None:
        self.sent = value

    def recv(self, size: int) -> bytes:
        return self.response[:size]


class RespTest(unittest.TestCase):
    def test_reads_fragmented_bulk_response(self) -> None:
        self.assertEqual(probe._read_resp(ChunkedStream(b"$5\r\nhello\r\n")), "hello")

    def test_rejects_truncated_bulk_response(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "closed the connection"):
            probe._read_resp(ChunkedStream(b"$5\r\nhel"))

    def test_parses_supported_response_types(self) -> None:
        cases = [
            (b"+PONG\r\n", "PONG"),
            (b":42\r\n", 42),
            (b"$-1\r\n", None),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(probe._read_resp(io.BytesIO(response)), expected)

    def test_rejects_error_response(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "NOAUTH"):
            probe._read_resp(io.BytesIO(b"-NOAUTH authentication required\r\n"))

    def test_rejects_invalid_negative_bulk_length(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid Redis bulk length"):
            probe._read_resp(io.BytesIO(b"$-2\r\n"))

    def test_encodes_resp_command(self) -> None:
        stream = io.BytesIO()
        probe._send_resp_command(stream, "AUTH", "secret")
        self.assertEqual(stream.getvalue(), b"*2\r\n$4\r\nAUTH\r\n$6\r\nsecret\r\n")


class ServiceProbeTest(unittest.TestCase):
    def test_http_rejects_error_status(self) -> None:
        for status in (302, 503):
            with self.subTest(status=status):
                response = mock.MagicMock()
                response.status = status
                response.__enter__.return_value = response
                with (
                    mock.patch.object(
                        probe.urllib.request, "urlopen", return_value=response
                    ),
                    self.assertRaisesRegex(RuntimeError, f"HTTP {status}"),
                ):
                    probe.probe_http("http://127.0.0.1/health", 1)

    def test_http_many_checks_every_url(self) -> None:
        urls = ["http://127.0.0.1/one", "http://127.0.0.1/two"]
        with mock.patch.object(probe, "probe_http") as probe_http:
            probe.probe_http_many(urls, 1)
        probe_http.assert_has_calls(
            [mock.call(urls[0], 1), mock.call(urls[1], 1)], any_order=True
        )

    def test_amqp_connects_with_runtime_credentials(self) -> None:
        connection = mock.Mock()
        pika = types.SimpleNamespace(
            PlainCredentials=mock.Mock(return_value="credentials"),
            ConnectionParameters=mock.Mock(return_value="parameters"),
            BlockingConnection=mock.Mock(return_value=connection),
        )
        with mock.patch.dict(sys.modules, {"pika": pika}):
            probe.probe_amqp("127.0.0.1", 5672, 1, "autogpt", "secret")
        pika.PlainCredentials.assert_called_once_with("autogpt", "secret")
        pika.BlockingConnection.assert_called_once_with("parameters")
        connection.close.assert_called_once_with()

    def test_redis_rejects_failed_authentication(self) -> None:
        connection = FakeConnection(b"+NOPE\r\n")
        with (
            mock.patch.object(
                probe.socket, "create_connection", return_value=connection
            ),
            self.assertRaisesRegex(RuntimeError, "authentication failed"),
        ):
            probe.probe_redis("127.0.0.1", 6380, 1, "secret", False)

    def test_redis_rejects_unhealthy_cluster(self) -> None:
        connection = FakeConnection(b"$18\r\ncluster_state:fail\r\n")
        with (
            mock.patch.object(
                probe.socket, "create_connection", return_value=connection
            ),
            self.assertRaisesRegex(RuntimeError, "cluster is not healthy"),
        ):
            probe.probe_redis("127.0.0.1", 17000, 1, "", True)

    def test_redis_rejects_wrong_ping_response(self) -> None:
        connection = FakeConnection(b"+NOPE\r\n")
        with (
            mock.patch.object(
                probe.socket, "create_connection", return_value=connection
            ),
            self.assertRaisesRegex(RuntimeError, "did not return PONG"),
        ):
            probe.probe_redis("127.0.0.1", 17000, 1, "", False)

    def test_redis_accepts_pong(self) -> None:
        connection = FakeConnection(b"+PONG\r\n")
        with mock.patch.object(
            probe.socket, "create_connection", return_value=connection
        ):
            probe.probe_redis("127.0.0.1", 17000, 1, "", False)

    def test_redis_accepts_healthy_cluster(self) -> None:
        connection = FakeConnection(b"$16\r\ncluster_state:ok\r\n")
        with mock.patch.object(
            probe.socket, "create_connection", return_value=connection
        ):
            probe.probe_redis("127.0.0.1", 17000, 1, "", True)

    def test_clam_rejects_wrong_ping_response(self) -> None:
        connection = FakeConnection(b"NOPE\0")
        with (
            mock.patch.object(
                probe.socket, "create_connection", return_value=connection
            ),
            self.assertRaisesRegex(RuntimeError, "did not return PONG"),
        ):
            probe.probe_clam("127.0.0.1", 3310, 1)
        self.assertEqual(connection.sent, b"zPING\0")

    def test_clam_accepts_pong(self) -> None:
        connection = FakeConnection(b"PONG\0")
        with mock.patch.object(
            probe.socket, "create_connection", return_value=connection
        ):
            probe.probe_clam("127.0.0.1", 3310, 1)
        self.assertEqual(connection.sent, b"zPING\0")


if __name__ == "__main__":
    unittest.main()
