"""
VideoConcatBlock - Concatenate multiple video clips into one
"""
import uuid

from backend.data.block import Block, BlockCategory, BlockOutput
from backend.data.block import BlockSchemaInput, BlockSchemaOutput
from backend.data.model import SchemaField
from backend.util.exceptions import BlockExecutionError


class VideoConcatBlock(Block):
    """Merge multiple video clips into one continuous video."""

    class Input(BlockSchemaInput):
        videos: list[str] = SchemaField(
            description="List of video files to concatenate (in order)"
        )
        transition: str = SchemaField(
            description="Transition between clips",
            default="none",
            enum=["none", "crossfade", "fade_black"]
        )
        transition_duration: float = SchemaField(
            description="Transition duration in seconds",
            default=0.5,
            advanced=True
        )
        output_format: str = SchemaField(
            description="Output format",
            default="mp4",
            advanced=True
        )

    class Output(BlockSchemaOutput):
        video_out: str = SchemaField(
            description="Concatenated video file",
            json_schema_extra={"format": "file"}
        )
        total_duration: float = SchemaField(description="Total duration in seconds")

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-a7b8-9012-cdef-345678901234",
            description="Merge multiple video clips into one continuous video",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"videos": ["/tmp/a.mp4", "/tmp/b.mp4"]},
            test_output=[("video_out", str), ("total_duration", float)],
            test_mock={"_concat_videos": lambda *args: ("/tmp/concat.mp4", 20.0)}
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            from moviepy.editor import VideoFileClip, concatenate_videoclips
        except ImportError as e:
            raise BlockExecutionError(
                message="moviepy is not installed. Please install it with: pip install moviepy",
                block_name=self.name,
                block_id=str(self.id)
            ) from e

        # Validate minimum clips
        if len(input_data.videos) < 2:
            raise BlockExecutionError(
                message="At least 2 videos are required for concatenation",
                block_name=self.name,
                block_id=str(self.id)
            )

        clips = []
        faded_clips = []
        final = None
        try:
            # Load clips one by one to handle partial failures
            for v in input_data.videos:
                clips.append(VideoFileClip(v))

            if input_data.transition == "crossfade":
                # Apply crossfade between clips using crossfadein/crossfadeout
                transition_dur = input_data.transition_duration
                for i, clip in enumerate(clips):
                    if i > 0:
                        clip = clip.crossfadein(transition_dur)
                    if i < len(clips) - 1:
                        clip = clip.crossfadeout(transition_dur)
                    faded_clips.append(clip)
                final = concatenate_videoclips(
                    faded_clips,
                    method="compose",
                    padding=-transition_dur
                )
            elif input_data.transition == "fade_black":
                # Fade to black between clips
                for clip in clips:
                    faded = clip.fadein(input_data.transition_duration).fadeout(
                        input_data.transition_duration
                    )
                    faded_clips.append(faded)
                final = concatenate_videoclips(faded_clips)
            else:
                final = concatenate_videoclips(clips)

            output_path = f"/tmp/concat_{uuid.uuid4()}.{input_data.output_format}"
            final.write_videofile(output_path, logger=None)

            yield "video_out", output_path
            yield "total_duration", final.duration

        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to concatenate videos: {e}",
                block_name=self.name,
                block_id=str(self.id)
            ) from e
        finally:
            if final:
                final.close()
            for clip in faded_clips:
                clip.close()
            for clip in clips:
                clip.close()
