"""VideoClipBlock - Extract a segment from a video file."""

import os
import tempfile
import uuid
from typing import Literal

from moviepy.video.io.VideoFileClip import VideoFileClip

from backend.data.block import Block, BlockCategory, BlockOutput
from backend.data.block import BlockSchemaInput, BlockSchemaOutput
from backend.data.model import SchemaField
from backend.util.exceptions import BlockExecutionError


class VideoClipBlock(Block):
    """Extract a time segment from a video."""

    class Input(BlockSchemaInput):
        video_in: str = SchemaField(
            description="Input video (URL, data URI, or file path)",
            json_schema_extra={"format": "file"}
        )
        start_time: float = SchemaField(
            description="Start time in seconds",
            ge=0.0
        )
        end_time: float = SchemaField(
            description="End time in seconds",
            ge=0.0
        )
        output_format: Literal["mp4", "webm", "mkv", "mov"] = SchemaField(
            description="Output format",
            default="mp4",
            advanced=True
        )

    class Output(BlockSchemaOutput):
        video_out: str = SchemaField(
            description="Clipped video file",
            json_schema_extra={"format": "file"}
        )
        duration: float = SchemaField(description="Clip duration in seconds")

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-bcde-f23456789012",
            description="Extract a time segment from a video",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"video_in": "/tmp/test.mp4", "start_time": 0.0, "end_time": 10.0},
            test_output=[("video_out", str), ("duration", float)],
            test_mock={"_clip_video": lambda *args: ("/tmp/clip.mp4", 10.0)}
        )

    def _clip_video(
        self,
        video_in: str,
        start_time: float,
        end_time: float,
        output_format: str,
    ) -> tuple[str, float]:
        """Extract a clip from a video. Extracted for testability."""
        clip = None
        subclip = None
        try:
            clip = VideoFileClip(video_in)
            subclip = clip.subclip(start_time, end_time)

            fd, output_path = tempfile.mkstemp(suffix=f".{output_format}")
            os.close(fd)
            subclip.write_videofile(output_path, logger=None)

            return output_path, subclip.duration
        finally:
            if subclip:
                subclip.close()
            if clip:
                clip.close()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Validate time range
        if input_data.end_time <= input_data.start_time:
            raise BlockExecutionError(
                message=f"end_time ({input_data.end_time}) must be greater than start_time ({input_data.start_time})",
                block_name=self.name,
                block_id=str(self.id)
            )

        try:
            output_path, duration = self._clip_video(
                input_data.video_in,
                input_data.start_time,
                input_data.end_time,
                input_data.output_format,
            )
            yield "video_out", output_path
            yield "duration", duration

        except BlockExecutionError:
            raise
        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to clip video: {e}",
                block_name=self.name,
                block_id=str(self.id)
            ) from e
