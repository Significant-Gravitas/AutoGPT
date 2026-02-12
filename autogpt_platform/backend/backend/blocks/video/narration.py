"""VideoNarrationBlock - Generate AI voice narration and add to video."""

import os
from typing import Literal

from elevenlabs import ElevenLabs
from moviepy import CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.elevenlabs._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ElevenLabsCredentials,
    ElevenLabsCredentialsInput,
)
from backend.blocks.video._utils import (
    extract_source_name,
    get_video_codecs,
    strip_chapters_inplace,
)
from backend.data.execution import ExecutionContext
from backend.data.model import CredentialsField, SchemaField
from backend.util.exceptions import BlockExecutionError
from backend.util.file import MediaFileType, get_exec_file_path, store_media_file


class VideoNarrationBlock(Block):
    """Generate AI narration and add to video."""

    class Input(BlockSchemaInput):
        credentials: ElevenLabsCredentialsInput = CredentialsField(
            description="ElevenLabs API key for voice synthesis"
        )
        video_in: MediaFileType = SchemaField(
            description="Input video (URL, data URI, or local path)"
        )
        script: str = SchemaField(description="Narration script text")
        voice_id: str = SchemaField(
            description="ElevenLabs voice ID", default="21m00Tcm4TlvDq8ikWAM"  # Rachel
        )
        model_id: Literal[
            "eleven_multilingual_v2",
            "eleven_flash_v2_5",
            "eleven_turbo_v2_5",
            "eleven_turbo_v2",
        ] = SchemaField(
            description="ElevenLabs TTS model",
            default="eleven_multilingual_v2",
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
        video_out: MediaFileType = SchemaField(
            description="Video with narration (path or data URI)"
        )
        audio_file: MediaFileType = SchemaField(
            description="Generated audio file (path or data URI)"
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
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("video_out", str), ("audio_file", str)],
            test_mock={
                "_generate_narration_audio": lambda *args: b"mock audio content",
                "_add_narration_to_video": lambda *args: None,
                "_store_input_video": lambda *args, **kwargs: "test.mp4",
                "_store_output_video": lambda *args, **kwargs: "narrated_test.mp4",
            },
        )

    async def _store_input_video(
        self, execution_context: ExecutionContext, file: MediaFileType
    ) -> MediaFileType:
        """Store input video. Extracted for testability."""
        return await store_media_file(
            file=file,
            execution_context=execution_context,
            return_format="for_local_processing",
        )

    async def _store_output_video(
        self, execution_context: ExecutionContext, file: MediaFileType
    ) -> MediaFileType:
        """Store output video. Extracted for testability."""
        return await store_media_file(
            file=file,
            execution_context=execution_context,
            return_format="for_block_output",
        )

    def _generate_narration_audio(
        self, api_key: str, script: str, voice_id: str, model_id: str
    ) -> bytes:
        """Generate narration audio via ElevenLabs API."""
        client = ElevenLabs(api_key=api_key)
        audio_generator = client.text_to_speech.convert(
            voice_id=voice_id,
            text=script,
            model_id=model_id,
        )
        # The SDK returns a generator, collect all chunks
        return b"".join(audio_generator)

    def _add_narration_to_video(
        self,
        video_abspath: str,
        audio_abspath: str,
        output_abspath: str,
        mix_mode: str,
        narration_volume: float,
        original_volume: float,
    ) -> None:
        """Add narration audio to video. Extracted for testability."""
        video = None
        final = None
        narration_original = None
        narration_scaled = None
        original = None

        try:
            strip_chapters_inplace(video_abspath)
            video = VideoFileClip(video_abspath)
            narration_original = AudioFileClip(audio_abspath)
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
            video_codec, audio_codec = get_video_codecs(output_abspath)
            final.write_videofile(
                output_abspath, codec=video_codec, audio_codec=audio_codec
            )

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
        self,
        input_data: Input,
        *,
        credentials: ElevenLabsCredentials,
        execution_context: ExecutionContext,
        node_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            assert execution_context.graph_exec_id is not None

            # Store the input video locally
            local_video_path = await self._store_input_video(
                execution_context, input_data.video_in
            )
            video_abspath = get_exec_file_path(
                execution_context.graph_exec_id, local_video_path
            )

            # Generate narration audio via ElevenLabs
            audio_content = self._generate_narration_audio(
                credentials.api_key.get_secret_value(),
                input_data.script,
                input_data.voice_id,
                input_data.model_id,
            )

            # Save audio to exec file path
            audio_filename = MediaFileType(f"{node_exec_id}_narration.mp3")
            audio_abspath = get_exec_file_path(
                execution_context.graph_exec_id, audio_filename
            )
            os.makedirs(os.path.dirname(audio_abspath), exist_ok=True)
            with open(audio_abspath, "wb") as f:
                f.write(audio_content)

            # Add narration to video
            source = extract_source_name(local_video_path)
            output_filename = MediaFileType(f"{node_exec_id}_narrated_{source}.mp4")
            output_abspath = get_exec_file_path(
                execution_context.graph_exec_id, output_filename
            )

            self._add_narration_to_video(
                video_abspath,
                audio_abspath,
                output_abspath,
                input_data.mix_mode,
                input_data.narration_volume,
                input_data.original_volume,
            )

            # Return as workspace path or data URI based on context
            video_out = await self._store_output_video(
                execution_context, output_filename
            )
            audio_out = await self._store_output_video(
                execution_context, audio_filename
            )

            yield "video_out", video_out
            yield "audio_file", audio_out

        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to add narration: {e}",
                block_name=self.name,
                block_id=str(self.id),
            ) from e
