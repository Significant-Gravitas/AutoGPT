"""VideoTextOverlayBlock - Add text overlay to video."""

import os
import tempfile
from typing import Literal

from moviepy import CompositeVideoClip, TextClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.exceptions import BlockExecutionError


class VideoTextOverlayBlock(Block):
    """Add text overlay/caption to video."""

    class Input(BlockSchemaInput):
        video_in: str = SchemaField(
            description="Input video file", json_schema_extra={"format": "file"}
        )
        text: str = SchemaField(description="Text to overlay on video")
        position: Literal[
            "top",
            "center",
            "bottom",
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ] = SchemaField(description="Position of text on screen", default="bottom")
        start_time: float | None = SchemaField(
            description="When to show text (seconds). None = entire video",
            default=None,
            advanced=True,
        )
        end_time: float | None = SchemaField(
            description="When to hide text (seconds). None = until end",
            default=None,
            advanced=True,
        )
        font_size: int = SchemaField(
            description="Font size", default=48, ge=12, le=200, advanced=True
        )
        font_color: str = SchemaField(
            description="Font color (hex or name)", default="white", advanced=True
        )
        bg_color: str | None = SchemaField(
            description="Background color behind text (None for transparent)",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        video_out: str = SchemaField(
            description="Video with text overlay", json_schema_extra={"format": "file"}
        )

    def __init__(self):
        super().__init__(
            id="8ef14de6-cc90-430a-8cfa-3a003be92454",
            description="Add text overlay/caption to video",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"video_in": "/tmp/test.mp4", "text": "Hello World"},
            test_output=[("video_out", str)],
            test_mock={"_add_text_overlay": lambda *args: "/tmp/overlay.mp4"},
        )

    def _add_text_overlay(
        self,
        video_in: str,
        text: str,
        position: str,
        start_time: float | None,
        end_time: float | None,
        font_size: int,
        font_color: str,
        bg_color: str | None,
    ) -> str:
        """Add text overlay to video. Extracted for testability."""
        video = None
        final = None
        txt_clip = None
        try:
            video = VideoFileClip(video_in)

            txt_clip = TextClip(
                text=text,
                font_size=font_size,
                color=font_color,
                bg_color=bg_color,
            )

            # Position mapping
            pos_map = {
                "top": ("center", "top"),
                "center": ("center", "center"),
                "bottom": ("center", "bottom"),
                "top-left": ("left", "top"),
                "top-right": ("right", "top"),
                "bottom-left": ("left", "bottom"),
                "bottom-right": ("right", "bottom"),
            }

            txt_clip = txt_clip.with_position(pos_map[position])

            # Set timing
            start = start_time or 0
            end = end_time or video.duration
            duration = max(0, end - start)
            txt_clip = txt_clip.with_start(start).with_end(end).with_duration(duration)

            final = CompositeVideoClip([video, txt_clip])

            fd, output_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            final.write_videofile(output_path, logger=None)

            return output_path
        finally:
            if txt_clip:
                txt_clip.close()
            if final:
                final.close()
            if video:
                video.close()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Validate time range if both are provided
        if (
            input_data.start_time is not None
            and input_data.end_time is not None
            and input_data.end_time <= input_data.start_time
        ):
            raise BlockExecutionError(
                message=f"end_time ({input_data.end_time}) must be greater than start_time ({input_data.start_time})",
                block_name=self.name,
                block_id=str(self.id),
            )

        try:
            output_path = self._add_text_overlay(
                input_data.video_in,
                input_data.text,
                input_data.position,
                input_data.start_time,
                input_data.end_time,
                input_data.font_size,
                input_data.font_color,
                input_data.bg_color,
            )
            yield "video_out", output_path

        except BlockExecutionError:
            raise
        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to add text overlay: {e}",
                block_name=self.name,
                block_id=str(self.id),
            ) from e
