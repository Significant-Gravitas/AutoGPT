"""Unit tests for expert run output classification — no DB required."""

from backend.data.expert_run_output import classify_output_type, classify_run_output


def test_classify_output_type_table_from_list_of_dicts():
    assert classify_output_type([{"name": "A"}, {"name": "B"}]) == "table"


def test_classify_output_type_table_from_single_dict():
    assert classify_output_type({"name": "A"}) == "table"


def test_classify_output_type_image_from_url_with_extension():
    assert classify_output_type("https://cdn.example.com/report.PNG") == "image"
    assert classify_output_type("https://cdn.example.com/chart.svg?v=2") == "image"
    assert classify_output_type("http://cdn.example.com/chart.png") == "unknown"


def test_classify_output_type_doc_from_long_text():
    assert classify_output_type("word " * 100) == "doc"


def test_classify_output_type_handles_multi_value_string_pins():
    assert classify_output_type(["word " * 25, "word " * 25]) == "doc"
    assert (
        classify_output_type(
            [
                "https://cdn.example.com/first.png",
                "https://cdn.example.com/second.webp",
            ]
        )
        == "image"
    )


def test_classify_output_type_unknown_for_short_text_and_scalars():
    assert classify_output_type("ok") == "unknown"
    assert classify_output_type(42) == "unknown"
    assert classify_output_type([1, 2, 3]) == "unknown"
    assert classify_output_type([]) == "unknown"


def test_classify_run_output_picks_first_renderable_pin_with_key():
    outputs = {"skipped": [], "result": [[{"row": 1}]]}
    assert classify_run_output(outputs) == ("table", "result")


def test_classify_run_output_skips_unrenderable_pin_for_later_table():
    """A short status string on the first pin must not mask a table on the
    second — the first *renderable* pin wins, not the first non-empty one."""
    outputs = {"status": ["ok"], "results": [[{"metric": "signups"}]]}
    assert classify_run_output(outputs) == ("table", "results")


def test_classify_run_output_empty_is_unknown():
    assert classify_run_output({}) == ("unknown", None)
    assert classify_run_output({"result": []}) == ("unknown", None)
    assert classify_run_output({"status": ["ok"]}) == ("unknown", None)
