"""
ElevenLabs speech-to-text (transcription) blocks.
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


class ElevenLabsTranscribeAudioSyncBlock(Block):
    """
    Synchronously convert audio to text (+ word timestamps, diarization).
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = elevenlabs.credentials_field(
            description="ElevenLabs API credentials"
        )
        model_id: str = SchemaField(
            description="Model ID for transcription", default="scribe_v1"
        )
        file: Optional[str] = SchemaField(
            description="Base64-encoded audio file", default=None
        )
        cloud_storage_url: Optional[str] = SchemaField(
            description="URL to audio file in cloud storage", default=None
        )
        language_code: Optional[str] = SchemaField(
            description="Language code (ISO 639-1 or -3) to improve accuracy",
            default=None,
        )
        diarize: bool = SchemaField(
            description="Enable speaker diarization", default=False
        )
        num_speakers: Optional[int] = SchemaField(
            description="Expected number of speakers (max 32)", default=None
        )
        timestamps_granularity: str = SchemaField(
            description="Timestamp detail level: word, character, or none",
            default="word",
        )
        tag_audio_events: bool = SchemaField(
            description="Tag non-speech sounds (laughter, noise)", default=True
        )

    class Output(BlockSchema):
        text: str = SchemaField(description="Full transcribed text")
        words: list[dict] = SchemaField(
            description="Array with word timing and speaker info"
        )
        language_code: str = SchemaField(description="Detected language code")
        language_probability: float = SchemaField(
            description="Confidence in language detection"
        )

    def __init__(self):
        super().__init__(
            id="e7f8a9b0-c1d2-e3f4-a5b6-c7d8e9f0a1b2",
            description="Transcribe audio to text with timing and speaker information",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        import base64
        from io import BytesIO

        api_key = credentials.api_key.get_secret_value()

        # Validate input - must have either file or URL
        if not input_data.file and not input_data.cloud_storage_url:
            raise ValueError("Either 'file' or 'cloud_storage_url' must be provided")
        if input_data.file and input_data.cloud_storage_url:
            raise ValueError(
                "Only one of 'file' or 'cloud_storage_url' should be provided"
            )

        # Build form data
        form_data = {
            "model_id": input_data.model_id,
            "diarize": str(input_data.diarize).lower(),
            "timestamps_granularity": input_data.timestamps_granularity,
            "tag_audio_events": str(input_data.tag_audio_events).lower(),
        }

        if input_data.language_code:
            form_data["language_code"] = input_data.language_code
        if input_data.num_speakers is not None:
            form_data["num_speakers"] = str(input_data.num_speakers)

        # Handle file or URL
        files = None
        if input_data.file:
            # Decode base64 file
            file_data = base64.b64decode(input_data.file)
            files = [("file", ("audio.wav", BytesIO(file_data), "audio/wav"))]
        elif input_data.cloud_storage_url:
            form_data["cloud_storage_url"] = input_data.cloud_storage_url

        # Transcribe audio
        response = await Requests().post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": api_key},
            data=form_data,
            files=files,
        )

        data = response.json()

        yield "text", data.get("text", "")
        yield "words", data.get("words", [])
        yield "language_code", data.get("language_code", "")
        yield "language_probability", data.get("language_probability", 0.0)


class ElevenLabsTranscribeAudioAsyncBlock(Block):
    """
    Kick off transcription that returns quickly; result arrives via webhook.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = elevenlabs.credentials_field(
            description="ElevenLabs API credentials"
        )
        model_id: str = SchemaField(
            description="Model ID for transcription", default="scribe_v1"
        )
        file: Optional[str] = SchemaField(
            description="Base64-encoded audio file", default=None
        )
        cloud_storage_url: Optional[str] = SchemaField(
            description="URL to audio file in cloud storage", default=None
        )
        language_code: Optional[str] = SchemaField(
            description="Language code (ISO 639-1 or -3) to improve accuracy",
            default=None,
        )
        diarize: bool = SchemaField(
            description="Enable speaker diarization", default=False
        )
        num_speakers: Optional[int] = SchemaField(
            description="Expected number of speakers (max 32)", default=None
        )
        timestamps_granularity: str = SchemaField(
            description="Timestamp detail level: word, character, or none",
            default="word",
        )
        webhook_url: str = SchemaField(
            description="URL to receive transcription result",
            default="",
        )

    class Output(BlockSchema):
        tracking_id: str = SchemaField(description="ID to track the transcription job")

    def __init__(self):
        super().__init__(
            id="f8a9b0c1-d2e3-f4a5-b6c7-d8e9f0a1b2c3",
            description="Start async transcription with webhook callback",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        import base64
        import uuid
        from io import BytesIO

        api_key = credentials.api_key.get_secret_value()

        # Validate input
        if not input_data.file and not input_data.cloud_storage_url:
            raise ValueError("Either 'file' or 'cloud_storage_url' must be provided")
        if input_data.file and input_data.cloud_storage_url:
            raise ValueError(
                "Only one of 'file' or 'cloud_storage_url' should be provided"
            )

        # Build form data
        form_data = {
            "model_id": input_data.model_id,
            "diarize": str(input_data.diarize).lower(),
            "timestamps_granularity": input_data.timestamps_granularity,
            "webhook": "true",  # Enable async mode
        }

        if input_data.language_code:
            form_data["language_code"] = input_data.language_code
        if input_data.num_speakers is not None:
            form_data["num_speakers"] = str(input_data.num_speakers)
        if input_data.webhook_url:
            form_data["webhook_url"] = input_data.webhook_url

        # Handle file or URL
        files = None
        if input_data.file:
            # Decode base64 file
            file_data = base64.b64decode(input_data.file)
            files = [("file", ("audio.wav", BytesIO(file_data), "audio/wav"))]
        elif input_data.cloud_storage_url:
            form_data["cloud_storage_url"] = input_data.cloud_storage_url

        # Start async transcription
        response = await Requests().post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": api_key},
            data=form_data,
            files=files,
        )

        # Generate tracking ID (API might return one)
        data = response.json()
        tracking_id = data.get("tracking_id", str(uuid.uuid4()))

        yield "tracking_id", tracking_id
