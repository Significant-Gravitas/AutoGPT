import importlib.util
import logging
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "python" / "sitecustomize.py"
SPEC = importlib.util.spec_from_file_location(
    "single_container_sitecustomize", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
sitecustomize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sitecustomize)


class SiteCustomizeTest(unittest.TestCase):
    def test_disables_uvicorn_http_access_logger(self) -> None:
        self.assertTrue(logging.getLogger("uvicorn.access").disabled)

    def test_redacts_websocket_query_string(self) -> None:
        record = logging.LogRecord(
            name="uvicorn.error",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='%s - "WebSocket %s" [accepted]',
            args=(("127.0.0.1", 1234), "/ws?token=do-not-log&other=value"),
            exc_info=None,
        )

        for log_filter in logging.getLogger("uvicorn.error").filters:
            log_filter.filter(record)

        rendered = record.getMessage()
        self.assertIn('"WebSocket /ws"', rendered)
        self.assertNotIn("do-not-log", rendered)
        self.assertNotIn("other=value", rendered)

    def test_leaves_websocket_target_without_query_unchanged(self) -> None:
        record = self._record(
            name="uvicorn.error",
            msg='%s - "WebSocket %s" [accepted]',
            args=(("127.0.0.1", 1234), "/ws"),
        )

        self._apply_filters(record)

        self.assertEqual(record.args[1], "/ws")

    def test_malformed_records_are_safe_no_ops(self) -> None:
        cases = (
            self._record(
                name="application",
                msg='%s - "WebSocket %s" [accepted]',
                args=(("127.0.0.1", 1234), "/ws?token=secret"),
            ),
            self._record(name="uvicorn.error", msg=123, args=()),
            self._record(name="uvicorn.error", msg="WebSocket %s", args=("/ws",)),
            self._record(
                name="uvicorn.error",
                msg='%(client)s - "WebSocket %(path)s" [accepted]',
                args={"client": "local", "path": "/ws?token=secret"},
            ),
        )
        original_args = [record.args for record in cases]

        for record in cases:
            with self.subTest(name=record.name, msg=record.msg, args=record.args):
                self.assertTrue(self._apply_filters(record))

        self.assertEqual([record.args for record in cases], original_args)

    def _apply_filters(self, record: logging.LogRecord) -> bool:
        return all(
            log_filter.filter(record)
            for log_filter in logging.getLogger("uvicorn.error").filters
        )

    def _record(
        self,
        *,
        name: str,
        msg: object,
        args: tuple[object, ...] | dict[str, object],
    ) -> logging.LogRecord:
        return logging.LogRecord(
            name=name,
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=args,
            exc_info=None,
        )
