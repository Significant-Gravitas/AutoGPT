import json

import pytest
from forge.json import json_loads

_JSON_FIXABLE: list[tuple[str, str]] = [
    # Missing comma
    ('{"name": "John Doe"   "age": 30,}', '{"name": "John Doe", "age": 30}'),
    ("[1, 2 3]", "[1, 2, 3]"),
    # Trailing comma
    ('{"name": "John Doe", "age": 30,}', '{"name": "John Doe", "age": 30}'),
    ("[1, 2, 3,]", "[1, 2, 3]"),
    # Extra comma in object
    ('{"name": "John Doe",, "age": 30}', '{"name": "John Doe", "age": 30}'),
    # Extra newlines
    ('{"name": "John Doe",\n"age": 30}', '{"name": "John Doe", "age": 30}'),
    ("[1, 2,\n3]", "[1, 2, 3]"),
    # Missing closing brace or bracket
    ('{"name": "John Doe", "age": 30', '{"name": "John Doe", "age": 30}'),
    ("[1, 2, 3", "[1, 2, 3]"),
    # Different numerals
    ("[+1, ---2, .5, +-4.5, 123.]", "[1, -2, 0.5, -4.5, 123]"),
    ('{"bin": 0b1001, "hex": 0x1A, "oct": 0o17}', '{"bin": 9, "hex": 26, "oct": 15}'),
    # Broken array
    (
        '[1, 2 3, "yes" true, false null, 25, {"obj": "var"}',
        '[1, 2, 3, "yes", true, false, null, 25, {"obj": "var"}]',
    ),
    # Codeblock
    (
        '```json\n{"name": "John Doe", "age": 30}\n```',
        '{"name": "John Doe", "age": 30}',
    ),
    # Mutliple problems
    (
        '{"name":"John Doe" "age": 30\n "empty": "","address": '
        "// random comment\n"
        '{"city": "New York", "state": "NY"},'
        '"skills": ["Python" "C++", "Java",""],',
        '{"name": "John Doe", "age": 30, "empty": "", "address": '
        '{"city": "New York", "state": "NY"}, '
        '"skills": ["Python", "C++", "Java", ""]}',
    ),
    # All good
    (
        '{"name": "John Doe", "age": 30, "address": '
        '{"city": "New York", "state": "NY"}, '
        '"skills": ["Python", "C++", "Java"]}',
        '{"name": "John Doe", "age": 30, "address": '
        '{"city": "New York", "state": "NY"}, '
        '"skills": ["Python", "C++", "Java"]}',
    ),
    ("true", "true"),
    ("false", "false"),
    ("null", "null"),
    ("123.5", "123.5"),
    ('"Hello, World!"', '"Hello, World!"'),
    ("{}", "{}"),
    ("[]", "[]"),
]

_JSON_UNFIXABLE: list[tuple[str, str]] = [
    # Broken booleans and null
    ("[TRUE, False, NULL]", "[true, false, null]"),
    # Missing values in array
    ("[1, , 3]", "[1, 3]"),
    # Leading zeros (are treated as octal)
    ("[0023, 015]", "[23, 15]"),
    # Missing quotes
    ('{"name": John Doe}', '{"name": "John Doe"}'),
    # Missing opening braces or bracket
    ('"name": "John Doe"}', '{"name": "John Doe"}'),
    ("1, 2, 3]", "[1, 2, 3]"),
]


@pytest.fixture(params=_JSON_FIXABLE)
def fixable_json(request: pytest.FixtureRequest) -> tuple[str, str]:
    return request.param


@pytest.fixture(params=_JSON_UNFIXABLE)
def unfixable_json(request: pytest.FixtureRequest) -> tuple[str, str]:
    return request.param


def test_json_loads_fixable(fixable_json: tuple[str, str]):
    assert json_loads(fixable_json[0]) == json.loads(fixable_json[1])


def test_json_loads_unfixable(unfixable_json: tuple[str, str]):
    assert json_loads(unfixable_json[0]) != json.loads(unfixable_json[1])
