import logging

from colorama import Fore, Style

from .formatters import AGPTFormatter


def test_formatter_does_not_mutate_log_record() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="failed %s",
        args=("safely",),
        exc_info=None,
    )
    formatter = AGPTFormatter("%(levelname)s %(title)s%(message)s")

    output = formatter.format(record)

    assert output == (
        f"{Fore.RED}ERROR{Style.RESET_ALL} {Fore.RED}failed safely{Style.RESET_ALL}"
    )
    assert record.levelname == "ERROR"
    assert record.msg == "failed %s"
    assert record.args == ("safely",)
    assert not hasattr(record, "title")
