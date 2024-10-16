import time
from typing import Literal

import requests

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


class CreateTalkingAvatarVideoBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(
            key="did_api_key", description="D-ID API Key"
        )
        script_input: str = SchemaField(
            description="The text input for the script", default="Welcome to AutoGPT"
        )
        provider: Literal["microsoft", "elevenlabs", "amazon"] = SchemaField(
            description="The voice provider to use", default="microsoft"
        )
        voice_id: str = SchemaField(
            description="The voice ID to use, get list of voices [here](https://docs.agpt.co/server/d_id)",
            default="en-US-JennyNeural",
        )
        presenter_id: str = SchemaField(
            description="The presenter ID to use", default="amy-Aq6OmGZnMt"
        )
        driver_id: str = SchemaField(
            description="The driver ID to use", default="Vcq0R4a8F0"
        )
        result_format: Literal["mp4", "gif", "wav"] = SchemaField(
            description="The desired result format", default="mp4"
        )
        crop_type: Literal["wide", "square", "vertical"] = SchemaField(
            description="The crop type for the presenter", default="wide"
        )
        subtitles: bool = SchemaField(
            description="Whether to include subtitles", default=False
        )
        ssml: bool = SchemaField(description="Whether the input is SSML", default=False)
        max_polling_attempts: int = SchemaField(
            description="Maximum number of polling attempts", default=30, ge=5
        )
        polling_interval: int = SchemaField(
            description="Interval between polling attempts in seconds", default=10, ge=5
        )

    class Output(BlockSchema):
        video_url: str = SchemaField(description="The URL of the created video")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="98c6f503-8c47-4b1c-a96d-351fc7c87dab",
            description="This block integrates with D-ID to create video clips and retrieve their URLs.",
            categories={BlockCategory.AI},
            input_schema=CreateTalkingAvatarVideoBlock.Input,
            output_schema=CreateTalkingAvatarVideoBlock.Output,
            test_input={
                "api_key": "your_test_api_key",
                "script_input": "Welcome to AutoGPT",
                "voice_id": "en-US-JennyNeural",
                "presenter_id": "amy-Aq6OmGZnMt",
                "driver_id": "Vcq0R4a8F0",
                "result_format": "mp4",
                "crop_type": "wide",
                "subtitles": False,
                "ssml": False,
                "max_polling_attempts": 5,
                "polling_interval": 5,
            },
            test_output=[
                (
                    "video_url",
                    "https://d-id.com/api/clips/abcd1234-5678-efgh-ijkl-mnopqrstuvwx/video",
                ),
            ],
            test_mock={
                "create_clip": lambda *args, **kwargs: {
                    "id": "abcd1234-5678-efgh-ijkl-mnopqrstuvwx",
                    "status": "created",
                },
                "get_clip_status": lambda *args, **kwargs: {
                    "status": "done",
                    "result_url": "https://d-id.com/api/clips/abcd1234-5678-efgh-ijkl-mnopqrstuvwx/video",
                },
            },
        )

    def create_clip(self, api_key: str, payload: dict) -> dict:
        url = "https://api.d-id.com/clips"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Basic {api_key}",
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_clip_status(self, api_key: str, clip_id: str) -> dict:
        url = f"https://api.d-id.com/clips/{clip_id}"
        headers = {"accept": "application/json", "authorization": f"Basic {api_key}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Create the clip
        payload = {
            "script": {
                "type": "text",
                "subtitles": str(input_data.subtitles).lower(),
                "provider": {
                    "type": input_data.provider,
                    "voice_id": input_data.voice_id,
                },
                "ssml": str(input_data.ssml).lower(),
                "input": input_data.script_input,
            },
            "config": {"result_format": input_data.result_format},
            "presenter_config": {"crop": {"type": input_data.crop_type}},
            "presenter_id": input_data.presenter_id,
            "driver_id": input_data.driver_id,
        }

        response = self.create_clip(input_data.api_key.get_secret_value(), payload)
        clip_id = response["id"]

        # Poll for clip status
        for _ in range(input_data.max_polling_attempts):
            status_response = self.get_clip_status(
                input_data.api_key.get_secret_value(), clip_id
            )
            if status_response["status"] == "done":
                yield "video_url", status_response["result_url"]
                return
            elif status_response["status"] == "error":
                raise RuntimeError(
                    f"Clip creation failed: {status_response.get('error', 'Unknown error')}"
                )

            time.sleep(input_data.polling_interval)

        raise TimeoutError("Clip creation timed out")
