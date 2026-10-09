import logging

import pytest

from backend.util.logging import TruncatedLogger


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error", "exception"])
def test_truncated_logger_hands_exc_info_to_logging(level, caplog):
    base = logging.getLogger(f"truncated_logger_test.{level}")
    logger = TruncatedLogger(base, prefix="[Test]")
    try:
        raise ValueError("kaput")
    except ValueError as e:
        error = e

    with caplog.at_level(logging.DEBUG, logger=base.name):
        getattr(logger, level)("failed", exc_info=error, user_id="u1")

    (record,) = caplog.records
    assert record.exc_info is not None and record.exc_info[1] is error
    assert "Traceback" in caplog.text
    assert record.getMessage() == "[Test] failed {'user_id': 'u1'}"
    assert record.__dict__["json_fields"] == {"user_id": "u1"}
