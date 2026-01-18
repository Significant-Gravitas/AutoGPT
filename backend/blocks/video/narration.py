"""
VideoNarrationBlock - Generate AI voice narration and add to video
"""
import uuid
from typing import Literal

from backend.data.block import Block, BlockCategory, BlockOutput
from backend.data.block import BlockSchemaInput, BlockSchemaOutput
from backend.data.model import SchemaField, CredentialsMetaInput, APIKeyCredentials
from backend.integrations.providers import ProviderName
from backend.util.exceptions import BlockExecutionError


class VideoNarrationBlock(Block):
    """Generate AI narration and add to video."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.ELEVENLABS], Literal["api_key"]
        ] = SchemaField(
            description="ElevenLabs API key for voice synthesis"
        )
        video_in: str = SchemaField(
            description="Input video file",
            json_schema_extra={"format": "file"}
        )
        script: str = SchemaField(
            description="Narration script text"
        )
        voice_id: str = SchemaField(
            description="ElevenLabs voice ID",
            default="21m00Tcm4TlvDq8ikWAM"  # Rachel
        )
        mix_mode: Literal["replace", "mix", "ducking"] = SchemaField(
            description="How to combine with original audio",
            default="ducking"
        )
        narration_volume: float = SchemaField(
            description="Narration volume (0.0 to 2.0)",
            default=1.0,
            ge=0.0,
            le=2.0,
            advanced=True
        )
        original_volume: float = SchemaField(
            description="Original audio volume when mixing (0.0 to 1.0)",
            default=0.3,
            ge=0.0,
            le=1.0,
            advanced=True
        )

    class Output(BlockSchemaOutput):
        video_out: str = SchemaField(
            description="Video with narration",
            json_schema_extra={"format": "file"}
        )
        audio_file: str = SchemaField(
            description="Generated audio file",
            json_schema_extra={"format": "file"}
        )

    def __init__(self):
        super().__init__(
            id="e5f6a7b8-c9d0-1234-ef56-789012345678",
            description="Generate AI narration and add to video",
            categories={BlockCategory.MULTIMEDIA, BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "video_in": "/tmp/test.mp4",
                "script": "Hello world",
                "credentials": {"provider": "elevenlabs", "id": "test", "type": "api_key"}
            },
            test_output=[("video_out", str), ("audio_file", str)],
            test_mock={"_generate_narration": lambda *args: ("/tmp/narrated.mp4", "/tmp/audio.mp3")}
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs
    ) -> BlockOutput:
        try:
            import requests
            from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
        except ImportError as e:
            raise BlockExecutionError(
                message=f"Missing dependency: {e}. Install moviepy and requests.",
                block_name=self.name,
                block_id=str(self.id)
            ) from e

        video = None
        final = None
        narration = None
        try:
            # Generate narration via ElevenLabs
            response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{input_data.voice_id}",
                headers={
                    "xi-api-key": credentials.api_key.get_secret_value(),
                    "Content-Type": "application/json"
                },
                json={
                    "text": input_data.script,
                    "model_id": "eleven_monolingual_v1"
                },
                timeout=120
            )
            response.raise_for_status()

            audio_path = f"/tmp/narration_{uuid.uuid4()}.mp3"
            with open(audio_path, "wb") as f:
                f.write(response.content)

            # Combine with video
            video = VideoFileClip(input_data.video_in)
            narration = AudioFileClip(audio_path)
            narration = narration.volumex(input_data.narration_volume)

            if input_data.mix_mode == "replace":
                final_audio = narration
            elif input_data.mix_mode == "mix":
                if video.audio:
                    original = video.audio.volumex(input_data.original_volume)
                    final_audio = CompositeAudioClip([original, narration])
                else:
                    final_audio = narration
            else:  # ducking - lower original volume more when narration plays
                if video.audio:
                    # Apply stronger attenuation for ducking effect
                    ducking_volume = input_data.original_volume * 0.3
                    original = video.audio.volumex(ducking_volume)
                    final_audio = CompositeAudioClip([original, narration])
                else:
                    final_audio = narration

            final = video.set_audio(final_audio)

            output_path = f"/tmp/narrated_{uuid.uuid4()}.mp4"
            final.write_videofile(output_path, logger=None)

            yield "video_out", output_path
            yield "audio_file", audio_path

        except requests.exceptions.RequestException as e:
            raise BlockExecutionError(
                message=f"ElevenLabs API error: {e}",
                block_name=self.name,
                block_id=str(self.id)
            ) from e
        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to add narration: {e}",
                block_name=self.name,
                block_id=str(self.id)
            ) from e
        finally:
            if narration:
                narration.close()
            if final:
                final.close()
            if video:
                video.close()
