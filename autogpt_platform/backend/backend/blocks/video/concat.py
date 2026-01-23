"""VideoConcatBlock - Concatenate multiple video clips into one."""

import os
import tempfile
from typing import Literal

from moviepy import concatenate_videoclips
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, FadeIn, FadeOut
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


class VideoConcatBlock(Block):
    """Merge multiple video clips into one continuous video."""

    class Input(BlockSchemaInput):
        videos: list[str] = SchemaField(
            description="List of video files to concatenate (in order)"
        )
        transition: Literal["none", "crossfade", "fade_black"] = SchemaField(
            description="Transition between clips", default="none"
        )
        transition_duration: int = SchemaField(
            description="Transition duration in seconds",
            default=1,
            ge=0,
            advanced=True,
        )
        output_format: Literal["mp4", "webm", "mkv", "mov"] = SchemaField(
            description="Output format", default="mp4", advanced=True
        )

    class Output(BlockSchemaOutput):
        video_out: str = SchemaField(
            description="Concatenated video file", json_schema_extra={"format": "file"}
        )
        total_duration: float = SchemaField(description="Total duration in seconds")

    def __init__(self):
        super().__init__(
            id="9b0f531a-1118-487f-aeec-3fa63ea8900a",
            description="Merge multiple video clips into one continuous video",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"videos": ["/tmp/a.mp4", "/tmp/b.mp4"]},
            test_output=[("video_out", str), ("total_duration", float)],
            test_mock={"_concat_videos": lambda *args: ("/tmp/concat.mp4", 20.0)},
        )

    def _concat_videos(
        self,
        videos: list[str],
        transition: str,
        transition_duration: int,
        output_format: str,
    ) -> tuple[str, float]:
        """Concatenate videos. Extracted for testability."""
        clips = []
        faded_clips = []
        final = None
        try:
            # Load clips
            for v in videos:
                clips.append(VideoFileClip(v))

            if transition == "crossfade":
                for i, clip in enumerate(clips):
                    effects = []
                    if i > 0:
                        effects.append(CrossFadeIn(transition_duration))
                    if i < len(clips) - 1:
                        effects.append(CrossFadeOut(transition_duration))
                    if effects:
                        clip = clip.with_effects(effects)
                    faded_clips.append(clip)
                final = concatenate_videoclips(
                    faded_clips,
                    method="compose",
                    padding=-transition_duration,
                )
            elif transition == "fade_black":
                for clip in clips:
                    faded = clip.with_effects(
                        [FadeIn(transition_duration), FadeOut(transition_duration)]
                    )
                    faded_clips.append(faded)
                final = concatenate_videoclips(faded_clips)
            else:
                final = concatenate_videoclips(clips)

            fd, output_path = tempfile.mkstemp(suffix=f".{output_format}")
            os.close(fd)
            final.write_videofile(output_path, logger=None)

            return output_path, final.duration
        finally:
            if final:
                final.close()
            for clip in faded_clips:
                clip.close()
            for clip in clips:
                clip.close()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Validate minimum clips
        if len(input_data.videos) < 2:
            raise BlockExecutionError(
                message="At least 2 videos are required for concatenation",
                block_name=self.name,
                block_id=str(self.id),
            )

        try:
            output_path, total_duration = self._concat_videos(
                input_data.videos,
                input_data.transition,
                input_data.transition_duration,
                input_data.output_format,
            )
            yield "video_out", output_path
            yield "total_duration", total_duration

        except BlockExecutionError:
            raise
        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to concatenate videos: {e}",
                block_name=self.name,
                block_id=str(self.id),
            ) from e
