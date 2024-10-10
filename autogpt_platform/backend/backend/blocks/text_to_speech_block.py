from typing import Any

import requests

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


class UnrealTextToSpeechBlock(Block):
    class Input(BlockSchema):
        text: str = SchemaField(
            description="The text to be converted to speech",
            placeholder="Enter the text you want to convert to speech",
        )
        voice_id: str = SchemaField(
            description="The voice ID to use for text-to-speech conversion",
            placeholder="Scarlett",
            default="Scarlett",
        )
        api_key: BlockSecret = SecretField(
            key="unreal_speech_api_key", description="Your Unreal Speech API key"
        )

    class Output(BlockSchema):
        mp3_url: str = SchemaField(description="The URL of the generated MP3 file")
        error: str = SchemaField(description="Error message if the API call failed")

    def __init__(self):
        super().__init__(
            id="4ff1ff6d-cc40-4caa-ae69-011daa20c378",
            description="Converts text to speech using the Unreal Speech API",
            categories={BlockCategory.AI, BlockCategory.TEXT},
            input_schema=UnrealTextToSpeechBlock.Input,
            output_schema=UnrealTextToSpeechBlock.Output,
            test_input={
                "text": "This is a test of the text to speech API.",
                "voice_id": "Scarlett",
                "api_key": "test_api_key",
            },
            test_output=[("mp3_url", "https://example.com/test.mp3")],
            test_mock={
                "call_unreal_speech_api": lambda *args, **kwargs: {
                    "OutputUri": "https://example.com/test.mp3"
                }
            },
        )

    @staticmethod
    def call_unreal_speech_api(
        api_key: str, text: str, voice_id: str
    ) -> dict[str, Any]:
        url = "https://api.v7.unrealspeech.com/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "Text": text,
            "VoiceId": voice_id,
            "Bitrate": "192k",
            "Speed": "0",
            "Pitch": "1",
            "TimestampType": "sentence",
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        api_response = self.call_unreal_speech_api(
            input_data.api_key.get_secret_value(),
            input_data.text,
            input_data.voice_id,
        )
        yield "mp3_url", api_response["OutputUri"]
