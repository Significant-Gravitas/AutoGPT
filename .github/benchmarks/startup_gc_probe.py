import json
import os
from pathlib import Path

import pytest

from startup_gc_control import observe_startup


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_load_initial_conftests():
    observation = None
    try:
        with observe_startup(
            os.environ["BENCHMARK_DISABLE_STARTUP_GC"] == "1"
        ) as observation:
            return (yield)
    finally:
        if observation is not None:
            Path(os.environ["BACKEND_TEST_STARTUP_GC_METRICS"]).write_text(
                json.dumps(observation, indent=2), encoding="utf-8"
            )
