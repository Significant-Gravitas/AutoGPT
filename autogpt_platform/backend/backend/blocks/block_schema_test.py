from typing import Annotated, Any

import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import pytest
from pydantic import Field

from backend.blocks._base import BlockSchemaInput


class _Input(BlockSchemaInput):
    union: dict | list[dict | list] | None = None
    counts: list[Annotated[int, Field(ge=1)]] = []
    names: list[str] = []
    limit: Annotated[int, Field(le=10)] | None = None
    tags: dict[str, Annotated[str, Field(max_length=3)]] = {}
    pages: Annotated[list[Annotated[int, Field(le=5)]], Field(max_length=2)] = []


@pytest.mark.parametrize(
    "data, expected",
    [
        pytest.param({"union": [0] * 5_000}, {}, id="union"),
        pytest.param({"names": list(range(5_000))}, {}, id="wrong-type"),
        pytest.param(
            {"counts": [0] * 5_000},
            {"counts": "0 is less than the minimum of 1"},
            id="out-of-bounds",
        ),
    ],
)
def test_a_long_invalid_list_builds_at_most_one_error(
    data: dict[str, Any], expected: dict[str, str], errors_built: list[None]
):
    assert _Input.get_field_errors(data) == expected
    assert len(errors_built) <= 1


@pytest.mark.parametrize(
    "data",
    [
        {"counts": [3, 0, -1]},
        {"pages": [9, 1, 7]},
        {"limit": 20},
        {"tags": {"a": "long", "b": "ok"}},
        {"union": [0], "names": [1]},
        {"counts": [5], "pages": [1], "tags": {}, "limit": None},
    ],
)
def test_field_errors_match_a_full_validation(data: dict[str, Any]):
    assert _Input.get_field_errors(data) == _full_validation_field_errors(data)


def _full_validation_field_errors(data: dict[str, Any]) -> dict[str, str]:
    """What get_field_errors reported when it validated the whole schema."""
    schema = _Input.jsonschema()
    validator = jsonschema.validators.validator_for(schema)(schema)
    errors: dict[str, str] = {}
    for error in validator.iter_errors(
        {k: v for k, v in data.items() if v is not None}
    ):
        if error.validator in _Input._INLINE_FIELD_ERROR_KEYWORDS and error.path:
            errors.setdefault(str(error.path[0]), error.message)
    return errors


@pytest.fixture
def errors_built(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    built: list[None] = []
    init = jsonschema.exceptions.ValidationError.__init__

    def counting_init(self, *args, **kwargs):
        built.append(None)
        init(self, *args, **kwargs)

    monkeypatch.setattr(
        jsonschema.exceptions.ValidationError, "__init__", counting_init
    )
    return built
