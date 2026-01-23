"""VideoTextOverlayBlock - Add text overlay to video."""

import os
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
from backend.util.file import MediaFileType, get_exec_file_path, store_media_file


class VideoTextOverlayBlock(Block):
    """Add text overlay/caption to video."""

    class Input(BlockSchemaInput):
        video_in: MediaFileType = SchemaField(
            description="Input video (URL, data URI, or local path)"
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
        output_return_type: Literal["file_path", "data_uri"] = SchemaField(
            description="Return the output as a relative path or base64 data URI.",
            default="file_path",
        )

    class Output(BlockSchemaOutput):
        video_out: MediaFileType = SchemaField(
            description="Video with text overlay (path or data URI)"
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
            test_mock={
                "_add_text_overlay": lambda *args: None,
                "_store_input_video": lambda *args, **kwargs: "test.mp4",
                "_store_output_video": lambda *args, **kwargs: "overlay_test.mp4",
            },
        )

    async def _store_input_video(
        self, graph_exec_id: str, file: MediaFileType, user_id: str
    ) -> MediaFileType:
        """Store input video. Extracted for testability."""
        return await store_media_file(
            graph_exec_id=graph_exec_id,
            file=file,
            user_id=user_id,
            return_content=False,
        )

    async def _store_output_video(
        self,
        graph_exec_id: str,
        file: MediaFileType,
        user_id: str,
        return_content: bool,
    ) -> MediaFileType:
        """Store output video. Extracted for testability."""
        return await store_media_file(
            graph_exec_id=graph_exec_id,
            file=file,
            user_id=user_id,
            return_content=return_content,
        )

    def _add_text_overlay(
        self,
        video_abspath: str,
        output_abspath: str,
        text: str,
        position: str,
        start_time: float | None,
        end_time: float | None,
        font_size: int,
        font_color: str,
        bg_color: str | None,
    ) -> None:
        """Add text overlay to video. Extracted for testability."""
        video = None
        final = None
        txt_clip = None
        try:
            video = VideoFileClip(video_abspath)

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
            final.write_videofile(output_abspath, codec="libx264", audio_codec="aac")

        finally:
            if txt_clip:
                txt_clip.close()
            if final:
                final.close()
            if video:
                video.close()

    async def run(
        self,
        input_data: Input,
        *,
        node_exec_id: str,
        graph_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
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
            # Store the input video locally
            local_video_path = await self._store_input_video(
                graph_exec_id, input_data.video_in, user_id
            )
            video_abspath = get_exec_file_path(graph_exec_id, local_video_path)

            # Build output path
            output_filename = MediaFileType(
                f"{node_exec_id}_overlay_{os.path.basename(local_video_path)}"
            )
            output_abspath = get_exec_file_path(graph_exec_id, output_filename)

            self._add_text_overlay(
                video_abspath,
                output_abspath,
                input_data.text,
                input_data.position,
                input_data.start_time,
                input_data.end_time,
                input_data.font_size,
                input_data.font_color,
                input_data.bg_color,
            )

            # Return as data URI or path
            video_out = await self._store_output_video(
                graph_exec_id,
                output_filename,
                user_id,
                input_data.output_return_type == "data_uri",
            )

            yield "video_out", video_out

        except BlockExecutionError:
            raise
        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to add text overlay: {e}",
                block_name=self.name,
                block_id=str(self.id),
            ) from e
