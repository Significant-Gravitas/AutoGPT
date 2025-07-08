"""
ElevenLabs speech generation (text-to-speech) blocks.
"""

from typing import Optional

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import elevenlabs


class ElevenLabsGenerateSpeechBlock(Block):
    """
    Turn text into audio (binary).
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = elevenlabs.credentials_field(
            description="ElevenLabs API credentials"
        )
        voice_id: str = SchemaField(description="ID of the voice to use")
        text: str = SchemaField(description="Text to convert to speech")
        model_id: str = SchemaField(
            description="Model ID to use for generation",
            default="eleven_multilingual_v2",
        )
        output_format: str = SchemaField(
            description="Audio format (e.g., mp3_44100_128)",
            default="mp3_44100_128",
        )
        voice_settings: Optional[dict] = SchemaField(
            description="Override voice settings (stability, similarity_boost, etc.)",
            default=None,
        )
        language_code: Optional[str] = SchemaField(
            description="Language code to enforce output language", default=None
        )
        seed: Optional[int] = SchemaField(
            description="Seed for reproducible output", default=None
        )

    class Output(BlockSchema):
        audio: str = SchemaField(description="Base64-encoded audio data")

    def __init__(self):
        super().__init__(
            id="c5d6e7f8-a9b0-c1d2-e3f4-a5b6c7d8e9f0",
            description="Generate speech audio from text using a specified voice",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        import base64

        api_key = credentials.api_key.get_secret_value()

        # Build request body
        body: dict[str, str | int | dict] = {
            "text": input_data.text,
            "model_id": input_data.model_id,
        }

        # Add optional fields
        if input_data.voice_settings:
            body["voice_settings"] = input_data.voice_settings
        if input_data.language_code:
            body["language_code"] = input_data.language_code
        if input_data.seed is not None:
            body["seed"] = input_data.seed

        # Generate speech
        response = await Requests().post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{input_data.voice_id}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=body,
            params={"output_format": input_data.output_format},
        )

        # Get audio data and encode to base64
        audio_data = response.content
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        yield "audio", audio_base64


class ElevenLabsGenerateSpeechWithTimestampsBlock(Block):
    """
    Text to audio AND per-character timing data.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = elevenlabs.credentials_field(
            description="ElevenLabs API credentials"
        )
        voice_id: str = SchemaField(description="ID of the voice to use")
        text: str = SchemaField(description="Text to convert to speech")
        model_id: str = SchemaField(
            description="Model ID to use for generation",
            default="eleven_multilingual_v2",
        )
        output_format: str = SchemaField(
            description="Audio format (e.g., mp3_44100_128)",
            default="mp3_44100_128",
        )
        voice_settings: Optional[dict] = SchemaField(
            description="Override voice settings (stability, similarity_boost, etc.)",
            default=None,
        )
        language_code: Optional[str] = SchemaField(
            description="Language code to enforce output language", default=None
        )

    class Output(BlockSchema):
        audio_base64: str = SchemaField(description="Base64-encoded audio data")
        alignment: dict = SchemaField(
            description="Character-level timing alignment data"
        )
        normalized_alignment: dict = SchemaField(
            description="Normalized text alignment data"
        )

    def __init__(self):
        super().__init__(
            id="d6e7f8a9-b0c1-d2e3-f4a5-b6c7d8e9f0a1",
            description="Generate speech with character-level timestamp information",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build request body
        body: dict[str, str | dict] = {
            "text": input_data.text,
            "model_id": input_data.model_id,
        }

        # Add optional fields
        if input_data.voice_settings:
            body["voice_settings"] = input_data.voice_settings
        if input_data.language_code:
            body["language_code"] = input_data.language_code

        # Generate speech with timestamps
        response = await Requests().post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{input_data.voice_id}/with-timestamps",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=body,
            params={"output_format": input_data.output_format},
        )

        data = response.json()

        yield "audio_base64", data.get("audio_base64", "")
        yield "alignment", data.get("alignment", {})
        yield "normalized_alignment", data.get("normalized_alignment", {})
