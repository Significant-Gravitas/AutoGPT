"""VideoNarrationBlock - Generate AI voice narration and add to video."""

import os
import tempfile
from typing import Literal

from elevenlabs import ElevenLabs
from moviepy import CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import APIKeyCredentials, CredentialsMetaInput, SchemaField
from backend.integrations.providers import ProviderName
from backend.util.exceptions import BlockExecutionError


class VideoNarrationBlock(Block):
    """Generate AI narration and add to video."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.ELEVENLABS],
            Literal["api_key"],
        ] = SchemaField(description="ElevenLabs API key for voice synthesis")
        video_in: str = SchemaField(
            description="Input video file", json_schema_extra={"format": "file"}
        )
        script: str = SchemaField(description="Narration script text")
        voice_id: str = SchemaField(
            description="ElevenLabs voice ID", default="21m00Tcm4TlvDq8ikWAM"  # Rachel
        )
        mix_mode: Literal["replace", "mix", "ducking"] = SchemaField(
            description="How to combine with original audio. 'ducking' applies stronger attenuation than 'mix'.",
            default="ducking",
        )
        narration_volume: float = SchemaField(
            description="Narration volume (0.0 to 2.0)",
            default=1.0,
            ge=0.0,
            le=2.0,
            advanced=True,
        )
        original_volume: float = SchemaField(
            description="Original audio volume when mixing (0.0 to 1.0)",
            default=0.3,
            ge=0.0,
            le=1.0,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        video_out: str = SchemaField(
            description="Video with narration", json_schema_extra={"format": "file"}
        )
        audio_file: str = SchemaField(
            description="Generated audio file", json_schema_extra={"format": "file"}
        )

    def __init__(self):
        super().__init__(
            id="3d036b53-859c-4b17-9826-ca340f736e0e",
            description="Generate AI narration and add to video",
            categories={BlockCategory.MULTIMEDIA, BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "video_in": "/tmp/test.mp4",
                "script": "Hello world",
                "credentials": {
                    "provider": "elevenlabs",
                    "id": "test",
                    "type": "api_key",
                },
            },
            test_output=[("video_out", str), ("audio_file", str)],
            test_mock={
                "_generate_narration_audio": lambda *args: b"mock audio content",
                "_add_narration_to_video": lambda *args: "/tmp/narrated.mp4",
            },
        )

    def _generate_narration_audio(
        self, api_key: str, script: str, voice_id: str
    ) -> bytes:
        """Generate narration audio via ElevenLabs API."""
        client = ElevenLabs(api_key=api_key)
        audio_generator = client.text_to_speech.convert(
            voice_id=voice_id,
            text=script,
            model_id="eleven_monolingual_v1",
        )
        # The SDK returns a generator, collect all chunks
        return b"".join(audio_generator)

    def _add_narration_to_video(
        self,
        video_in: str,
        audio_path: str,
        mix_mode: str,
        narration_volume: float,
        original_volume: float,
    ) -> str:
        """Add narration audio to video. Extracted for testability."""
        video = None
        final = None
        narration_original = None
        narration_scaled = None
        original = None

        try:
            video = VideoFileClip(video_in)
            narration_original = AudioFileClip(audio_path)
            narration_scaled = narration_original.with_volume_scaled(narration_volume)
            narration = narration_scaled

            if mix_mode == "replace":
                final_audio = narration
            elif mix_mode == "mix":
                if video.audio:
                    original = video.audio.with_volume_scaled(original_volume)
                    final_audio = CompositeAudioClip([original, narration])
                else:
                    final_audio = narration
            else:  # ducking - apply stronger attenuation
                if video.audio:
                    # Ducking uses a much lower volume for original audio
                    ducking_volume = original_volume * 0.3
                    original = video.audio.with_volume_scaled(ducking_volume)
                    final_audio = CompositeAudioClip([original, narration])
                else:
                    final_audio = narration

            final = video.with_audio(final_audio)

            fd, output_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            final.write_videofile(output_path, logger=None)

            return output_path

        finally:
            if original:
                original.close()
            if narration_scaled:
                narration_scaled.close()
            if narration_original:
                narration_original.close()
            if final:
                final.close()
            if video:
                video.close()

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            # Generate narration audio via ElevenLabs
            audio_content = self._generate_narration_audio(
                credentials.api_key.get_secret_value(),
                input_data.script,
                input_data.voice_id,
            )

            # Save audio to temp file
            fd, audio_path = tempfile.mkstemp(suffix=".mp3")
            with os.fdopen(fd, "wb") as f:
                f.write(audio_content)

            # Add narration to video
            output_path = self._add_narration_to_video(
                input_data.video_in,
                audio_path,
                input_data.mix_mode,
                input_data.narration_volume,
                input_data.original_volume,
            )

            yield "video_out", output_path
            yield "audio_file", audio_path

        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to add narration: {e}",
                block_name=self.name,
                block_id=str(self.id),
            ) from e
