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
