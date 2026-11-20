"""YAML is parsed with the safe loader within the limits a caller sets."""

import pytest
import yaml

from backend.util.bounded_yaml import (
    MAX_YAML_DEPTH,
    MAX_YAML_NODES,
    YAMLLimitError,
    load_bounded_yaml,
)


def _load(text: str, *, allow_anchors: bool = True, **limits: int):
    return load_bounded_yaml(
        text,
        subject="Test YAML",
        max_bytes=limits.get("max_bytes", 64 * 1024),
        max_expanded_bytes=limits.get("max_expanded_bytes", 64 * 1024),
        allow_anchors=allow_anchors,
    )


def test_an_ordinary_document_parses() -> None:
    assert _load("name: tidy\ntags: [a, b]\nnested: {x: 1}") == {
        "name": "tidy",
        "tags": ["a", "b"],
        "nested": {"x": 1},
    }


def test_an_empty_document_is_none() -> None:
    assert _load("") is None


def test_text_over_the_byte_limit_is_refused() -> None:
    with pytest.raises(YAMLLimitError, match="Test YAML is larger than 1 KiB"):
        _load(f"key: {'x' * 1024}", max_bytes=1024)


def test_a_value_reused_through_an_alias_counts_each_time_it_is_used() -> None:
    uses = ", ".join(["*b"] * 40)
    reused = f"base: &b [xxxxxxxx, xxxxxxxx, xxxxxxxx]\nuses: [{uses}]"

    assert len(_load(reused)["uses"]) == 40
    with pytest.raises(YAMLLimitError, match="expands to more than 1 KiB"):
        _load(reused, max_expanded_bytes=1024)


def test_anchors_parse_when_allowed_and_are_refused_otherwise() -> None:
    text = "base: &b {x: 1}\nuse: *b"

    assert _load(text) == {"base": {"x": 1}, "use": {"x": 1}}
    with pytest.raises(YAMLLimitError, match="cannot use YAML anchors"):
        _load(text, allow_anchors=False)


def test_a_document_that_refers_to_itself_is_refused() -> None:
    with pytest.raises(YAMLLimitError, match="refers to itself"):
        _load("a: &a [1, *a]")


def _nested(levels: int) -> str:
    return f"{'[' * levels}x{']' * levels}"


def test_nesting_up_to_the_depth_limit_parses() -> None:
    value = _load(_nested(MAX_YAML_DEPTH))
    for _ in range(MAX_YAML_DEPTH):
        value = value[0]
    assert value == "x"


def test_nesting_past_the_depth_limit_is_refused() -> None:
    with pytest.raises(YAMLLimitError, match="nested more than 32 levels deep"):
        _load(_nested(MAX_YAML_DEPTH + 1))


def test_nesting_built_up_through_aliases_counts_toward_the_depth_limit() -> None:
    lines = ["l0: &l0 x"]
    lines += [f"l{i}: &l{i} {'[' * 15}*l{i - 1}{']' * 15}" for i in (1, 2, 3)]

    with pytest.raises(YAMLLimitError, match="nested more than 32 levels deep"):
        _load("\n".join(lines))


def test_a_very_deeply_nested_document_is_refused() -> None:
    with pytest.raises(YAMLLimitError, match="nested more than 32 levels deep"):
        _load(_nested(5000))


@pytest.mark.parametrize("allow_anchors", [True, False])
def test_nesting_is_refused_once_it_passes_the_limit_before_the_rest_is_read(
    allow_anchors: bool,
) -> None:
    # Never closed, so reading to the end of the document would be a syntax error.
    unfinished = "[" * (MAX_YAML_DEPTH + 1) + "x"

    with pytest.raises(YAMLLimitError, match="nested more than 32 levels deep"):
        _load(unfinished, allow_anchors=allow_anchors)


@pytest.mark.parametrize("allow_anchors", [True, False])
def test_a_document_over_the_node_budget_is_refused_while_composing(
    allow_anchors: bool,
) -> None:
    # Never closed, so only a refusal made while composing avoids a syntax error.
    unfinished = "[" + "1, " * (MAX_YAML_NODES + 1)

    with pytest.raises(YAMLLimitError, match="more than 250,000 nodes"):
        _load(
            unfinished,
            allow_anchors=allow_anchors,
            max_bytes=len(unfinished) + 1,
            max_expanded_bytes=len(unfinished) * 8,
        )


@pytest.mark.parametrize("allow_anchors", [True, False])
def test_a_document_within_the_node_budget_parses(allow_anchors: bool) -> None:
    items = ", ".join(["1"] * 1000)

    assert len(_load(f"[{items}]", allow_anchors=allow_anchors)) == 1000


def test_invalid_yaml_raises_a_yaml_error() -> None:
    with pytest.raises(yaml.YAMLError):
        _load("key: [unclosed")
