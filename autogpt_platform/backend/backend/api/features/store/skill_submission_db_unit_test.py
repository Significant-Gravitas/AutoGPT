import pytest

from backend.copilot.tools.skills import ParsedSkill

from .skill_submission_db import _MAX_LICENSE_CHARS, _license_of


def _skill_with_license(value: object) -> ParsedSkill:
    return ParsedSkill(
        name="demo",
        description="Demo",
        body="# Demo\n",
        extra={"license": value},
    )


def test_license_is_trimmed():
    assert _license_of(_skill_with_license("  Apache-2.0  ")) == "Apache-2.0"


@pytest.mark.parametrize("value", [{"name": "MIT"}, ["MIT"], 1])
def test_license_must_be_text(value: object):
    with pytest.raises(ValueError, match="license must be a string"):
        _license_of(_skill_with_license(value))


def test_license_length_is_bounded():
    with pytest.raises(ValueError, match=f"license must be ≤{_MAX_LICENSE_CHARS}"):
        _license_of(_skill_with_license("x" * (_MAX_LICENSE_CHARS + 1)))
