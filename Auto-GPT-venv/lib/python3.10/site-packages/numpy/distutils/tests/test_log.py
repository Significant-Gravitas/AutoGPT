import io
import re
from contextlib import redirect_stdout

import pytest

from numpy.distutils import log


def setup_module():
    f = io.StringIO()  # changing verbosity also logs here, capture that
    with redirect_stdout(f):
        log.set_verbosity(2, force=True)  # i.e. DEBUG


def teardown_module():
    log.set_verbosity(0, force=True)  # the default


r_ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@pytest.mark.parametrize("func_name", ["error", "warn", "info", "debug"])
def test_log_prefix(func_name):
    func = getattr(log, func_name)
    msg = f"{func_name} message"
    f = io.StringIO()
    with redirect_stdout(f):
        func(msg)
    out = f.getvalue()
    assert out  # sanity check
    clean_out = r_ansi.sub("", out)
    line = next(line for line in clean_out.splitlines())
    assert line == f"{func_name.upper()}: {msg}"
