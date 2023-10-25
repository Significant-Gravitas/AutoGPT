import json
import os

from unittest.mock import MagicMock

import pytest
import rudderstack.analytics as rudder_analytics

from gpt_engineer.cli.collect import collect_learnings, steps_file_hash
from gpt_engineer.core.db import DB, DBs
from gpt_engineer.cli.learning import collect_consent, extract_learning
from gpt_engineer.core.steps import simple_gen


def test_collect_learnings(monkeypatch):
    monkeypatch.setattr(os, "environ", {"COLLECT_LEARNINGS_OPT_IN": "true"})
    monkeypatch.setattr(rudder_analytics, "track", MagicMock())

    model = "test_model"
    temperature = 0.5
    steps = [simple_gen]
    dbs = DBs(
        DB("/tmp"), DB("/tmp"), DB("/tmp"), DB("/tmp"), DB("/tmp"), DB("/tmp"), DB("/tmp")
    )
    dbs.input = {
        "prompt": "test prompt\n with newlines",
        "feedback": "test feedback",
    }
    code = "this is output\n\nit contains code"
    dbs.logs = {steps[0].__name__: json.dumps([{"role": "system", "content": code}])}
    dbs.memory = {"all_output.txt": "test workspace\n" + code}

    collect_learnings(model, temperature, steps, dbs)

    learnings = extract_learning(
        model, temperature, steps, dbs, steps_file_hash=steps_file_hash()
    )
    assert rudder_analytics.track.call_count == 1
    assert rudder_analytics.track.call_args[1]["event"] == "learning"
    a = {
        k: v
        for k, v in rudder_analytics.track.call_args[1]["properties"].items()
        if k != "timestamp"
    }
    b = {k: v for k, v in learnings.to_dict().items() if k != "timestamp"}
    assert a == b

    assert json.dumps(code) in learnings.logs
    assert code in learnings.workspace


if __name__ == "__main__":
    pytest.main(["-v"])
