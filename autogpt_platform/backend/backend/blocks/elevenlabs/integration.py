"""
ElevenLabs integration for text-to-speech capabilities.

This module provides blocks for interacting with the ElevenLabs API,
which offers high-quality text-to-speech and speech-to-speech conversion.
"""

import base64
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import APIKeyCredentials, CredentialsField, SchemaField

from ._api import ElevenLabsClient, ElevenLabsException, VoiceSettings
from ._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, ElevenLabsCredentialsInput

logger = logging.getLogger(__name__)


class Voice(BaseModel):
    """Model representing an ElevenLabs voice."""

    voice_id: str
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    preview_url: Optional[str] = None


class TextToSpeechBlock(Block):
    """Block for converting text to speech using ElevenLabs API."""

    class Input(BlockSchema):
        text: str = SchemaField(
            description="The text to convert to speech.",
            placeholder="Enter your text here...",
        )
        voice_id: str = SchemaField(
            description="The ID of the voice to use.",
            placeholder="21m00Tcm4TlvDq8ikWAM",
        )
        model_id: Optional[str] = SchemaField(
            description="The ID of the model to use (e.g., eleven_multilingual_v2).",
            placeholder="eleven_multilingual_v2",
            default=None,
            advanced=True,
        )
        stability: float = SchemaField(
            description="Voice stability (0.0 to 1.0). Higher values make voice more consistent.",
            placeholder="0.75",
            default=0.75,
            advanced=True,
        )
        similarity_boost: float = SchemaField(
            description="Similarity boost (0.0 to 1.0). Higher values make voice sound more like the original.",
            placeholder="0.85",
            default=0.85,
            advanced=True,
        )
        stream: bool = SchemaField(
            description="Whether to stream the audio response.",
            default=False,
            advanced=True,
        )
        credentials: ElevenLabsCredentialsInput = CredentialsField(
            description="ElevenLabs API credentials."
        )

    class Output(BlockSchema):
        audio_data: str = SchemaField(
            description="The generated audio data in Base64 format."
        )
        content_type: str = SchemaField(
            description="The MIME type of the audio (e.g., audio/mpeg)."
        )
        text: str = SchemaField(description="The text that was converted to speech.")
        voice_id: str = SchemaField(description="The ID of the voice used.")
        error: str = SchemaField(
            description="Error message if the text-to-speech conversion failed."
        )

    def __init__(self):
        super().__init__(
            id="d923f6a8-beb2-4a57-90e2-b9c2f7e30f91",
            description="Convert text to speech using ElevenLabs' high-quality voices.",
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=TextToSpeechBlock.Input,
            output_schema=TextToSpeechBlock.Output,
            test_input={
                "text": "Hello, this is a test of the ElevenLabs text-to-speech API.",
                "voice_id": "21m00Tcm4TlvDq8ikWAM",
                "model_id": "eleven_multilingual_v2",
                "stability": 0.75,
                "similarity_boost": 0.85,
                "stream": False,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("audio_data", "base64_encoded_audio_data"),
                ("content_type", "audio/mpeg"),
                ("text", "Hello, this is a test of the ElevenLabs text-to-speech API."),
                ("voice_id", "21m00Tcm4TlvDq8ikWAM"),
            ],
            test_mock={
                "generate_speech": lambda *args, **kwargs: (
                    base64.b64encode(b"mock_audio_data").decode("utf-8"),
                    "audio/mpeg",
                )
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def generate_speech(
        self,
        client: ElevenLabsClient,
        text: str,
        voice_id: str,
        model_id: Optional[str] = None,
        voice_settings: Optional[VoiceSettings] = None,
        stream: bool = False,
    ) -> tuple[str, str]:
        """
        Generate speech from text using ElevenLabs API.

        Args:
            client: Initialized ElevenLabsClient.
            text: Text to convert to speech.
            voice_id: ID of the voice to use.
            model_id: Optional model ID.
            voice_settings: Optional voice settings.
            stream: Whether to use streaming endpoint.

        Returns:
            Tuple of (base64_encoded_audio_data, content_type).
        """
        try:
            if stream:
                audio_data = client.text_to_speech_stream(
                    voice_id=voice_id,
                    text=text,
                    model_id=model_id,
                    voice_settings=voice_settings,
                )
            else:
                audio_data = client.text_to_speech(
                    voice_id=voice_id,
                    text=text,
                    model_id=model_id,
                    voice_settings=voice_settings,
                )

            # Encode the binary audio data to base64 for transmission
            base64_audio = base64.b64encode(audio_data).decode("utf-8")
            return base64_audio, "audio/mpeg"
        except ElevenLabsException as e:
            logger.error(f"ElevenLabs API error: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in speech generation: {str(e)}")
            raise ElevenLabsException(f"Failed to generate speech: {str(e)}", 500)

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Run the text-to-speech conversion.

        Args:
            input_data: Input data containing text and voice settings.
            credentials: ElevenLabs API credentials.

        Yields:
            Audio data and metadata.
        """
        try:
            client = ElevenLabsClient(credentials=credentials)

            # Create voice settings if provided
            voice_settings = None
            if hasattr(input_data, "stability") and hasattr(
                input_data, "similarity_boost"
            ):
                voice_settings = VoiceSettings(
                    stability=input_data.stability,
                    similarity_boost=input_data.similarity_boost,
                )

            # Generate speech
            audio_data, content_type = self.generate_speech(
                client=client,
                text=input_data.text,
                voice_id=input_data.voice_id,
                model_id=input_data.model_id,
                voice_settings=voice_settings,
                stream=input_data.stream,
            )

            # Yield results
            yield "audio_data", audio_data
            yield "content_type", content_type
            yield "text", input_data.text
            yield "voice_id", input_data.voice_id

        except ElevenLabsException as e:
            yield "error", f"ElevenLabs API error: {str(e)}"
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"


class ListVoicesBlock(Block):
    """Block for listing available voices from ElevenLabs API."""

    class Input(BlockSchema):
        credentials: ElevenLabsCredentialsInput = CredentialsField(
            description="ElevenLabs API credentials."
        )

    class Output(BlockSchema):
        voices: List[Voice] = SchemaField(description="List of available voices.")
        voice_ids: List[str] = SchemaField(description="List of voice IDs only.")
        error: str = SchemaField(
            description="Error message if the operation failed."
        )

    def __init__(self):
        super().__init__(
            id="4eaa8b1e-c0bc-45d2-a566-5fd4a5ce738d",
            description="List all available voices from your ElevenLabs account.",
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=ListVoicesBlock.Input,
            output_schema=ListVoicesBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                (
                    "voices",
                    [
                        Voice(
                            voice_id="21m00Tcm4TlvDq8ikWAM",
                            name="Rachel",
                            category="premade",
                        ),
                        Voice(
                            voice_id="AZnzlk1XvdvUeBnXmlld",
                            name="Domi",
                            category="premade",
                        ),
                    ],
                ),
                ("voice_ids", ["21m00Tcm4TlvDq8ikWAM", "AZnzlk1XvdvUeBnXmlld"]),
            ],
            test_mock={
                "get_voices_list": lambda *args, **kwargs: [
                    Voice(
                        voice_id="21m00Tcm4TlvDq8ikWAM",
                        name="Rachel",
                        category="premade",
                    ),
                    Voice(
                        voice_id="AZnzlk1XvdvUeBnXmlld",
                        name="Domi",
                        category="premade",
                    ),
                ]
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def get_voices_list(self, client: ElevenLabsClient) -> List[Voice]:
        """
        Get list of voices from ElevenLabs.

        Args:
            client: Initialized ElevenLabsClient.

        Returns:
            List of Voice objects.
        """
        try:
            response = client.get_voices()
            voices = []
            for voice_data in response.get("voices", []):
                voice = Voice(
                    voice_id=voice_data.get("voice_id"),
                    name=voice_data.get("name"),
                    category=voice_data.get("category"),
                    description=voice_data.get("description"),
                    preview_url=voice_data.get("preview_url"),
                )
                voices.append(voice)
            return voices
        except ElevenLabsException as e:
            logger.error(f"ElevenLabs API error when listing voices: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error when listing voices: {str(e)}")
            raise ElevenLabsException(f"Failed to list voices: {str(e)}", 500)

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Run the list voices operation.

        Args:
            input_data: Input data (mainly credentials).
            credentials: ElevenLabs API credentials.

        Yields:
            List of voices and voice IDs.
        """
        try:
            client = ElevenLabsClient(credentials=credentials)
            voices = self.get_voices_list(client)

            yield "voices", voices
            yield "voice_ids", [voice.voice_id for voice in voices]

        except ElevenLabsException as e:
            yield "error", f"ElevenLabs API error: {str(e)}"
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"


class VoiceSettingsBlock(Block):
    """Block for managing voice settings in ElevenLabs."""

    class Input(BlockSchema):
        voice_id: str = SchemaField(
            description="The ID of the voice to manage.",
            placeholder="21m00Tcm4TlvDq8ikWAM",
        )
        action: str = SchemaField(
            description="Action to perform on the voice settings.",
            placeholder="get",
            options=["get", "update"],
        )
        stability: Optional[float] = SchemaField(
            description="Voice stability (0.0 to 1.0) for update action.",
            placeholder="0.75",
            default=None,
            advanced=True,
        )
        similarity_boost: Optional[float] = SchemaField(
            description="Similarity boost (0.0 to 1.0) for update action.",
            placeholder="0.85",
            default=None,
            advanced=True,
        )
        credentials: ElevenLabsCredentialsInput = CredentialsField(
            description="ElevenLabs API credentials."
        )

    class Output(BlockSchema):
        stability: Optional[float] = SchemaField(
            description="Current voice stability setting."
        )
        similarity_boost: Optional[float] = SchemaField(
            description="Current voice similarity boost setting."
        )
        success: bool = SchemaField(description="Whether the operation was successful.")
        message: str = SchemaField(description="Operation result message.")
        error: str = SchemaField(
            description="Error message if the operation failed."
        )

    def __init__(self):
        super().__init__(
            id="5f3c4b87-1a9d-47bc-9fb3-d7c4e63a10e9",
            description="Get or update voice settings in ElevenLabs.",
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=VoiceSettingsBlock.Input,
            output_schema=VoiceSettingsBlock.Output,
            test_input={
                "voice_id": "21m00Tcm4TlvDq8ikWAM",
                "action": "get",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("stability", 0.75),
                ("similarity_boost", 0.85),
                ("success", True),
                ("message", "Voice settings retrieved successfully."),
            ],
            test_mock={
                "get_voice_settings": lambda *args, **kwargs: {
                    "stability": 0.75,
                    "similarity_boost": 0.85,
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def get_voice_settings(
        self, client: ElevenLabsClient, voice_id: str
    ) -> Dict[str, float]:
        """
        Get current settings for a voice.

        Args:
            client: Initialized ElevenLabsClient.
            voice_id: ID of the voice.

        Returns:
            Dictionary with stability and similarity_boost values.
        """
        try:
            return client.get_voice_settings(voice_id)
        except Exception as e:
            logger.error(f"Error getting voice settings: {str(e)}")
            raise

    def update_voice_settings(
        self,
        client: ElevenLabsClient,
        voice_id: str,
        stability: float,
        similarity_boost: float,
    ) -> None:
        """
        Update settings for a voice.

        Args:
            client: Initialized ElevenLabsClient.
            voice_id: ID of the voice.
            stability: Stability setting (0.0 to 1.0).
            similarity_boost: Similarity boost setting (0.0 to 1.0).
        """
        try:
            client.edit_voice_settings(voice_id, stability, similarity_boost)
        except Exception as e:
            logger.error(f"Error updating voice settings: {str(e)}")
            raise

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Run the voice settings operation.

        Args:
            input_data: Input data with voice ID and action.
            credentials: ElevenLabs API credentials.

        Yields:
            Current settings or operation result.
        """
        try:
            client = ElevenLabsClient(credentials=credentials)

            if input_data.action == "get":
                settings = self.get_voice_settings(client, input_data.voice_id)
                yield "stability", settings.get("stability")
                yield "similarity_boost", settings.get("similarity_boost")
                yield "success", True
                yield "message", "Voice settings retrieved successfully."

            elif input_data.action == "update":
                if input_data.stability is None or input_data.similarity_boost is None:
                    yield "error", "Both stability and similarity_boost must be provided for update action."
                    yield "success", False
                    yield "message", "Update failed: missing parameters."
                    return

                self.update_voice_settings(
                    client,
                    input_data.voice_id,
                    input_data.stability,
                    input_data.similarity_boost,
                )

                # Get updated settings
                updated_settings = self.get_voice_settings(client, input_data.voice_id)
                yield "stability", updated_settings.get("stability")
                yield "similarity_boost", updated_settings.get("similarity_boost")
                yield "success", True
                yield "message", "Voice settings updated successfully."

            else:
                yield "error", f"Unknown action: {input_data.action}"
                yield "success", False
                yield "message", f"Failed: {input_data.action} is not a valid action."

        except ElevenLabsException as e:
            yield "error", f"ElevenLabs API error: {str(e)}"
            yield "success", False
            yield "message", f"Operation failed: {str(e)}"
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"
            yield "success", False
            yield "message", f"Operation failed with unexpected error: {str(e)}"
