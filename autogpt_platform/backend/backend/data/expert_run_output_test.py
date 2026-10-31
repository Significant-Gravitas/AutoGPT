from datetime import datetime, timedelta, timezone

from backend.data.expert_run_output import reconstruct_run_outputs

_T0 = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def _entry(seconds: int, name: str, value, output: list | None = None):
    output_data = {"output": output} if output is not None else {}
    return (
        _T0 + timedelta(seconds=seconds),
        _T0,
        {"name": name, "value": value},
        output_data,
    )


def test_yielded_output_wins_over_input_value():
    outputs = reconstruct_run_outputs([_entry(0, "shout", "hi", output=["HI"])])
    assert outputs == {"shout": ["HI"]}


def test_falls_back_to_input_value_when_nothing_was_yielded():
    outputs = reconstruct_run_outputs(
        [_entry(0, "a", "raw-a"), _entry(1, "b", "raw-b", output=[])]
    )
    assert outputs == {"a": ["raw-a"], "b": ["raw-b"]}


def test_keeps_queue_order_and_every_yielded_value():
    outputs = reconstruct_run_outputs(
        [
            _entry(5, "rows", "raw-3", output=["3"]),
            _entry(1, "rows", "raw-1", output=["1", "2"]),
        ]
    )
    assert outputs == {"rows": ["1", "2", "3"]}


def test_name_comes_from_input_even_when_not_yielded():
    # The format branch of AgentOutputBlock yields only ``output``.
    outputs = reconstruct_run_outputs(
        [(None, _T0, {"name": "summary", "value": {"k": 1}}, {"output": ["k=1"]})]
    )
    assert outputs == {"summary": ["k=1"]}


def test_skips_entries_without_a_name():
    outputs = reconstruct_run_outputs([(None, _T0, {"value": 1}, {"output": [1]})])
    assert outputs == {}
