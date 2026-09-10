import json

import pytest
from pydantic import ValidationError

from backend.api.features.experts.models import (
    EXPERT_IDENTITY_MAX_LENGTH,
    ExpertSoulFieldsPatch,
    ExpertSoulUpdate,
    RaiseAttachment,
    VoiceSample,
    decode_voice_preferences,
    encode_voice_preferences,
    validate_avatar_url,
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


@pytest.mark.parametrize(
    ("raw", "expected_description"),
    [
        # One malformed template row must degrade, never fail template
        # listing or hiring.
        ('{"description": "x", "samples": null}', ""),
        ('{"description": "x", "samples": "not-a-list"}', ""),
        ('{"description": "x", "samples": [{"label": 1, "text": null}]}', "x"),
        ('{"description": 42, "samples": []}', ""),
        ('{"description": null, "samples": []}', ""),
    ],
)
def test_decode_malformed_envelope_degrades_without_raising(
    raw: str, expected_description: str
):
    assert decode_voice_preferences(raw) == (expected_description, [])


def test_decode_mixed_sample_list_keeps_only_valid_samples():
    raw = json.dumps(
        {
            "description": "Clear and direct.",
            "samples": [
                {"label": "Punchy", "text": "Ship it."},
                {"label": 1, "text": None},
                "not-an-object",
                {"label": "Warm"},
            ],
        }
    )

    description, samples = decode_voice_preferences(raw)

    assert description == "Clear and direct."
    assert samples == [VoiceSample(label="Punchy", text="Ship it.")]


def test_raise_attachment_strips_id():
    attachment = RaiseAttachment(kind="workflow", source="library", id="  agent-1  ")
    assert attachment.id == "agent-1"


def test_raise_attachment_rejects_blank_id():
    with pytest.raises(ValidationError):
        RaiseAttachment(kind="skill", source="library", id="   ")


def test_raise_attachment_strips_before_length_bounds():
    # The trim runs ahead of max_length, so padding around a full-length id
    # cannot push it over the limit.
    padded = f"  {'a' * 100}  "

    attachment = RaiseAttachment(kind="skill", source="library", id=padded)

    assert attachment.id == "a" * 100


def test_raise_attachment_blank_id_reports_the_blank_message():
    # Whitespace satisfies min_length, so the blank check must own the error.
    with pytest.raises(ValidationError) as error:
        RaiseAttachment(kind="skill", source="library", id="   ")

    assert "Attachment id must not be blank" in str(error.value)


def test_raise_attachment_rejects_non_string_id():
    # The "before" validator hands a non-str through so Pydantic still owns
    # the type error.
    with pytest.raises(ValidationError):
        RaiseAttachment.model_validate({"kind": "skill", "source": "library", "id": 7})


def test_soul_update_strips_before_length_bounds():
    padded_name = f"  {'n' * 100}  "
    padded_identity = f"  {'i' * EXPERT_IDENTITY_MAX_LENGTH}  "

    soul = ExpertSoulUpdate(
        name=padded_name,
        identity=padded_identity,
        voice_preferences="",
        boundaries="",
    )

    assert soul.name == "n" * 100
    assert soul.identity == "i" * EXPERT_IDENTITY_MAX_LENGTH


def test_soul_patch_strips_before_length_bounds():
    patch = ExpertSoulFieldsPatch(identity=f"  {'i' * EXPERT_IDENTITY_MAX_LENGTH}  ")

    assert patch.identity == "i" * EXPERT_IDENTITY_MAX_LENGTH


def test_soul_patch_leaves_omitted_fields_none():
    patch = ExpertSoulFieldsPatch()

    assert patch.identity is None
    assert patch.voice_preferences is None
    assert patch.boundaries is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("   ", None),
        ("  https://cdn.example.com/a.png  ", "https://cdn.example.com/a.png"),
        ("/experts/maria.svg", "/experts/maria.svg"),
    ],
)
def test_validate_avatar_url_accepts_https_and_relative_paths(
    value: str | None, expected: str | None
):
    assert validate_avatar_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        # Plaintext http would fetch the avatar unencrypted.
        "http://cdn.example.com/a.png",
        # Browsers read "\" as "/" for special schemes, so these look
        # site-relative but resolve to a third-party origin.
        "/\\tracker.example/avatar.png",
        "https:/\\tracker.example/avatar.png",
        # Tab/CR/LF are stripped by the URL parser, turning this into
        # "//evil.example/x.png" — protocol-relative to another origin.
        "/\t/evil.example/x.png",
        "//evil.example/x.png",
        "javascript:alert(1)",
        "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
        "ftp://example.com/x.png",
        # https with no host is not fetchable from our origin.
        "https:///a.png",
        "https://",
    ],
)
def test_validate_avatar_url_rejects_unsafe_values(value: str):
    with pytest.raises(ValueError):
        validate_avatar_url(value)
