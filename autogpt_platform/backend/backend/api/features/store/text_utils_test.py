"""Tests for text_utils helpers."""

import pytest

from backend.api.features.store.text_utils import split_camelcase

# ---------------------------------------------------------------------------
# split_camelcase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("AITextGeneratorBlock", "AI Text Generator Block"),
        ("HTTPRequestBlock", "HTTP Request Block"),
        ("simpleWord", "simple Word"),
        ("already spaced", "already spaced"),
        ("XMLParser", "XML Parser"),
        ("getHTTPResponse", "get HTTP Response"),
        ("Block", "Block"),
        ("", ""),
        ("OAuth2Block", "OAuth2 Block"),
        ("IOError", "IO Error"),
        ("getHTTPSResponse", "get HTTPS Response"),
        # Known limitation: single-letter uppercase prefixes are NOT split.
        # "ABlock" stays "ABlock" because the algorithm requires the left
        # part of an uppercase run to retain at least 2 uppercase chars.
        ("ABlock", "ABlock"),
    ],
)
def test_split_camelcase(input_text: str, expected: str):
    assert split_camelcase(input_text) == expected
