"""VideoClipBlock - Extract a segment from a video file."""

from typing import Literal

from moviepy.video.io.VideoFileClip import VideoFileClip

from backend.blocks.video._utils import (
    extract_source_name,
    get_video_codecs,
    strip_chapters_inplace,
)
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.exceptions import BlockExecutionError
from backend.util.file import MediaFileType, get_exec_file_path, store_media_file


class VideoClipBlock(Block):
    """Extract a time segment from a video."""

    class Input(BlockSchemaInput):
        video_in: MediaFileType = SchemaField(
            description="Input video (URL, data URI, or local path)"
        )
        start_time: float = SchemaField(description="Start time in seconds", ge=0.0)
        end_time: float = SchemaField(description="End time in seconds", ge=0.0)
        output_format: Literal["mp4", "webm", "mkv", "mov"] = SchemaField(
            description="Output format", default="mp4", advanced=True
        )

    class Output(BlockSchemaOutput):
        video_out: MediaFileType = SchemaField(
            description="Clipped video file (path or data URI)"
        )
        duration: float = SchemaField(description="Clip duration in seconds")

    def __init__(self):
        super().__init__(
            id="8f539119-e580-4d86-ad41-86fbcb22abb1",
            description="Extract a time segment from a video",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "video_in": "/tmp/test.mp4",
                "start_time": 0.0,
                "end_time": 10.0,
            },
            test_output=[("video_out", str), ("duration", float)],
            test_mock={
                "_clip_video": lambda *args: 10.0,
                "_store_input_video": lambda *args, **kwargs: "test.mp4",
                "_store_output_video": lambda *args, **kwargs: "clip_test.mp4",
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

    def _clip_video(
        self,
        video_abspath: str,
        output_abspath: str,
        start_time: float,
        end_time: float,
    ) -> float:
        """Extract a clip from a video. Extracted for testability."""
        clip = None
        subclip = None
        try:
            strip_chapters_inplace(video_abspath)
            clip = VideoFileClip(video_abspath)
            subclip = clip.subclipped(start_time, end_time)
            video_codec, audio_codec = get_video_codecs(output_abspath)
            subclip.write_videofile(
                output_abspath, codec=video_codec, audio_codec=audio_codec
            )
            return subclip.duration
        finally:
            if subclip:
                subclip.close()
            if clip:
                clip.close()

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        node_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        # Validate time range
        if input_data.end_time <= input_data.start_time:
            raise BlockExecutionError(
                message=f"end_time ({input_data.end_time}) must be greater than start_time ({input_data.start_time})",
                block_name=self.name,
                block_id=str(self.id),
            )

        try:
            assert execution_context.graph_exec_id is not None

            # Store the input video locally
            local_video_path = await self._store_input_video(
                execution_context, input_data.video_in
            )
            video_abspath = get_exec_file_path(
                execution_context.graph_exec_id, local_video_path
            )

            # Build output path
            source = extract_source_name(local_video_path)
            output_filename = MediaFileType(
                f"{node_exec_id}_clip_{source}.{input_data.output_format}"
            )
            output_abspath = get_exec_file_path(
                execution_context.graph_exec_id, output_filename
            )

            duration = self._clip_video(
                video_abspath,
                output_abspath,
                input_data.start_time,
                input_data.end_time,
            )

            # Return as workspace path or data URI based on context
            video_out = await self._store_output_video(
                execution_context, output_filename
            )

            yield "video_out", video_out
            yield "duration", duration

        except BlockExecutionError:
            raise
        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to clip video: {e}",
                block_name=self.name,
                block_id=str(self.id),
            ) from e
