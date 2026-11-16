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


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error", "exception"])
def test_truncated_logger_records_its_callers_location(level, caplog):
    logger = TruncatedLogger(logging.getLogger(f"truncated_logger_test.at.{level}"))

    with caplog.at_level(logging.DEBUG, logger=logger.logger.name):
        _log_here(logger, level)
        _log_one_level_up(logger, level)

    assert [(r.funcName, r.lineno) for r in caplog.records] == [
        ("_log_here", _log_here.__code__.co_firstlineno + 1),
        ("_log_one_level_up", _log_one_level_up.__code__.co_firstlineno + 1),
    ]


def _log_here(logger: TruncatedLogger, level: str, **kwargs):
    getattr(logger, level)("logged", **kwargs)


def _log_one_level_up(logger: TruncatedLogger, level: str):
    _log_here(logger, level, stacklevel=2)
