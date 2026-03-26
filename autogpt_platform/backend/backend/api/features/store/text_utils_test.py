"""Tests for split_camelcase (now in backend.util.text)."""

import pytest

from backend.util.text import split_camelcase

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
        # Digit-to-uppercase transitions
        ("Base64Encoder", "Base64 Encoder"),
        ("UTF8Decoder", "UTF8 Decoder"),
        # Pure digits — no camelCase boundaries to split
        ("123", "123"),
        # Known limitation: single-letter uppercase segments after digits
        # are not split from the following word.  "3D" is only 1 uppercase
        # char so the uppercase-run rule cannot fire, producing "3 DRenderer"
        # rather than the ideal "3D Renderer".
        ("3DRenderer", "3 DRenderer"),
        # Exception list — compound terms that should stay together
        ("YouTubeBlock", "YouTube Block"),
        ("OpenAIBlock", "OpenAI Block"),
        ("AutoGPTAgent", "AutoGPT Agent"),
        ("GitHubIntegration", "GitHub Integration"),
        ("LinkedInBlock", "LinkedIn Block"),
    ],
)
def test_split_camelcase(input_text: str, expected: str):
    assert split_camelcase(input_text) == expected
