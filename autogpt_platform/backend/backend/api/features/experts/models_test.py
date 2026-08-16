import pytest
from pydantic import ValidationError

from backend.api.features.experts.models import ExpertSoulUpdate


def test_soul_update_strips_optional_fields():
    soul = ExpertSoulUpdate(
        name="  Mara  ",
        identity="  You are Mara.  ",
        voice_preferences="   ",
        boundaries="  Ask before sending.  ",
    )

    assert soul.name == "Mara"
    assert soul.identity == "You are Mara."
    # Whitespace-only optional fields collapse to "", so the prompt builder
    # falls back to "Not specified." instead of emitting blank sections.
    assert soul.voice_preferences == ""
    assert soul.boundaries == "Ask before sending."


@pytest.mark.parametrize("field", ["name", "identity"])
def test_soul_update_rejects_blank_required_fields(field: str):
    payload = {
        "name": "Mara",
        "identity": "You are Mara.",
        "voice_preferences": "",
        "boundaries": "",
        field: "   ",
    }

    with pytest.raises(ValidationError):
        ExpertSoulUpdate(**payload)
