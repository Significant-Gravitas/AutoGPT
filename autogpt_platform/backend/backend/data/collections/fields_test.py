"""What each field type stores, converts and refuses, and what a new field is inferred as."""

from typing import Any

import pytest

from .fields import FieldSet, key_text
from .model import MAX_FIELDS, CollectionError, CollectionField, FieldType

CASES = [
    # type, fits, converts (given, stored), refused
    (FieldType.TEXT, "Harbour Pallets BV", (640, "640"), {"a": 1}),
    (FieldType.NUMBER, 1180.5, ("42", 42), "forty-two"),
    (
        FieldType.DATE,
        "2026-11-02",
        ("2026-11-02T14:00:00+01:00", "2026-11-02T13:00:00Z"),
        "2 Nov",
    ),
    (FieldType.SELECT, "Booked", (7, "7"), ["Booked"]),
    (
        FieldType.MULTI_SELECT,
        ["Brightline"],
        ("Oko Studio", ["Oko Studio"]),
        [{"x": 1}],
    ),
    (FieldType.CHECKBOX, True, ("yes", True), "maybe"),
    (
        FieldType.URL,
        "https://example.com/p/1",
        (" https://example.com ", "https://example.com"),
        "javascript:alert(1)",
    ),
    (
        FieldType.EMAIL,
        "sam@example.com",
        (" sam@example.com", "sam@example.com"),
        "sam at example",
    ),
]


@pytest.mark.parametrize(
    "kind, fits, converts, refused", CASES, ids=[c[0].value for c in CASES]
)
def test_each_type_stores_converts_and_refuses(
    kind: FieldType, fits: Any, converts: tuple[Any, Any], refused: Any
) -> None:
    fields = FieldSet([CollectionField(name="value", type=kind)])

    assert fields.coerce_row({"value": fits}, grow=False) == {"value": fits}
    assert fields.coerce_row({"value": converts[0]}, grow=False) == {
        "value": converts[1]
    }
    with pytest.raises(CollectionError, match="value"):
        fields.coerce_row({"other": "x", "value": refused}, grow=True)


@pytest.mark.parametrize(
    "value, inferred",
    [
        (True, FieldType.CHECKBOX),
        (1900, FieldType.NUMBER),
        (0.21, FieldType.NUMBER),
        ("2026-11-21", FieldType.DATE),
        ("2026-11-21T09:30:00Z", FieldType.DATE),
        ("https://example.com/in/jordan", FieldType.URL),
        (["Brightline", "Oko Studio"], FieldType.MULTI_SELECT),
        ("cohort retention chart", FieldType.TEXT),
        ("sam@example.com", FieldType.TEXT),
    ],
)
def test_an_unknown_field_is_added_with_the_inferred_type(
    value: Any, inferred: FieldType
) -> None:
    fields = FieldSet([])

    fields.coerce_row({"new": value}, grow=True)

    assert fields.require("new").type is inferred
    assert fields.changed is True


def test_the_field_past_the_limit_is_refused() -> None:
    fields = FieldSet(
        [CollectionField(name=f"f{i}", type=FieldType.TEXT) for i in range(MAX_FIELDS)]
    )

    with pytest.raises(CollectionError, match=f"at most {MAX_FIELDS} fields"):
        fields.coerce_row({"one_more": "x"}, grow=True)


def test_a_select_learns_options_and_keeps_their_first_spelling() -> None:
    fields = FieldSet(
        [CollectionField(name="status", type=FieldType.SELECT, options=["New"])]
    )

    assert fields.coerce_row({"status": "drafted"}, grow=False) == {"status": "drafted"}
    assert fields.coerce_row({"Status": "NEW"}, grow=False) == {"status": "New"}
    assert fields.require("status").options == ["New", "drafted"]


def test_an_old_name_writes_to_the_renamed_field() -> None:
    fields = FieldSet(
        [CollectionField(name="amount", type=FieldType.NUMBER, aliases=["total"])]
    )

    assert fields.coerce_row({"total": 12}, grow=True) == {"amount": 12}
    assert fields.changed is False


@pytest.mark.parametrize(
    "value, text", [(42.0, "42"), (True, "true"), ("INV-7", "INV-7"), (None, None)]
)
def test_key_text_matches_equal_values(value: Any, text: str | None) -> None:
    assert key_text(value) == text
