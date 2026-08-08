from __future__ import annotations

import importlib.util
import io
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "probe.py"
SPEC = importlib.util.spec_from_file_location("single_container_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ChunkedStream(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        return super().read(min(size, 1) if size >= 0 else 1)


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


if __name__ == "__main__":
    unittest.main()
