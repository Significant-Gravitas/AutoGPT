import pytest
from pydantic import ValidationError

from backend.api.features.experts.models import (
    ExpertSoulUpdate,
    VoiceSample,
    decode_voice_preferences,
    encode_voice_preferences,
)


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


def test_encode_decode_voice_preferences_round_trips():
    samples = [
        VoiceSample(label="Punchy", text="Ship it."),
        VoiceSample(label="Warm", text="Let's start with a story."),
    ]

    description, decoded = decode_voice_preferences(
        encode_voice_preferences("Clear and direct.", samples)
    )

    assert description == "Clear and direct."
    assert decoded == samples


def test_decode_plain_string_returns_it_with_no_samples():
    # A hired copy stores the user's plain-text pick, never the envelope.
    description, samples = decode_voice_preferences("Warm, concise, and direct.")

    assert description == "Warm, concise, and direct."
    assert samples == []


def test_decode_empty_string_is_empty():
    assert decode_voice_preferences("") == ("", [])


def test_decode_non_envelope_json_degrades_to_plain_string():
    # Valid JSON without the sample envelope shape is treated as a plain value
    # rather than raising.
    description, samples = decode_voice_preferences('{"foo": "bar"}')

    assert description == '{"foo": "bar"}'
    assert samples == []
