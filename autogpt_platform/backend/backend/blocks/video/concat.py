"""VideoConcatBlock - Concatenate multiple video clips into one."""

from typing import Literal

from moviepy import concatenate_videoclips
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, FadeIn, FadeOut
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


class VideoConcatBlock(Block):
    """Merge multiple video clips into one continuous video."""

    class Input(BlockSchemaInput):
        videos: list[MediaFileType] = SchemaField(
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
        video_out: MediaFileType = SchemaField(
            description="Concatenated video file (path or data URI)"
        )
        total_duration: float = SchemaField(description="Total duration in seconds")

    def __init__(self):
        super().__init__(
            id="9b0f531a-1118-487f-aeec-3fa63ea8900a",
            description="Merge multiple video clips into one continuous video",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "videos": ["/tmp/a.mp4", "/tmp/b.mp4"],
            },
            test_output=[
                ("video_out", str),
                ("total_duration", float),
            ],
            test_mock={
                "_concat_videos": lambda *args: 20.0,
                "_store_input_video": lambda *args, **kwargs: "test.mp4",
                "_store_output_video": lambda *args, **kwargs: "concat_test.mp4",
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

    def _concat_videos(
        self,
        video_abspaths: list[str],
        output_abspath: str,
        transition: str,
        transition_duration: int,
    ) -> float:
        """Concatenate videos. Extracted for testability.

        Returns:
            Total duration of the concatenated video.
        """
        clips = []
        faded_clips = []
        final = None
        try:
            # Load clips
            for v in video_abspaths:
                strip_chapters_inplace(v)
                clips.append(VideoFileClip(v))

            # Validate transition_duration against shortest clip
            if transition in {"crossfade", "fade_black"} and transition_duration > 0:
                min_duration = min(c.duration for c in clips)
                if transition_duration >= min_duration:
                    raise BlockExecutionError(
                        message=(
                            f"transition_duration ({transition_duration}s) must be "
                            f"shorter than the shortest clip ({min_duration:.2f}s)"
                        ),
                        block_name=self.name,
                        block_id=str(self.id),
                    )

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

            video_codec, audio_codec = get_video_codecs(output_abspath)
            final.write_videofile(
                output_abspath, codec=video_codec, audio_codec=audio_codec
            )

            return final.duration
        finally:
            if final:
                final.close()
            for clip in faded_clips:
                clip.close()
            for clip in clips:
                clip.close()

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        node_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        # Validate minimum clips
        if len(input_data.videos) < 2:
            raise BlockExecutionError(
                message="At least 2 videos are required for concatenation",
                block_name=self.name,
                block_id=str(self.id),
            )

        try:
            assert execution_context.graph_exec_id is not None

            # Store all input videos locally
            video_abspaths = []
            for video in input_data.videos:
                local_path = await self._store_input_video(execution_context, video)
                video_abspaths.append(
                    get_exec_file_path(execution_context.graph_exec_id, local_path)
                )

            # Build output path
            source = (
                extract_source_name(video_abspaths[0]) if video_abspaths else "video"
            )
            output_filename = MediaFileType(
                f"{node_exec_id}_concat_{source}.{input_data.output_format}"
            )
            output_abspath = get_exec_file_path(
                execution_context.graph_exec_id, output_filename
            )

            total_duration = self._concat_videos(
                video_abspaths,
                output_abspath,
                input_data.transition,
                input_data.transition_duration,
            )

            # Return as workspace path or data URI based on context
            video_out = await self._store_output_video(
                execution_context, output_filename
            )

            yield "video_out", video_out
            yield "total_duration", total_duration

        except BlockExecutionError:
            raise
        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to concatenate videos: {e}",
                block_name=self.name,
                block_id=str(self.id),
            ) from e
