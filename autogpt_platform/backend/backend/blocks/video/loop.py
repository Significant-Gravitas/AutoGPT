"""LoopVideoBlock - Loop a video to a given duration or number of repeats."""

from typing import Optional

from moviepy.video.fx.Loop import Loop
from moviepy.video.io.VideoFileClip import VideoFileClip

from backend.blocks.video._utils import extract_source_name, strip_chapters_inplace
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.file import MediaFileType, get_exec_file_path, store_media_file


class LoopVideoBlock(Block):
    """Loop (repeat) a video clip until a given duration or number of loops."""

    class Input(BlockSchemaInput):
        video_in: MediaFileType = SchemaField(
            description="The input video (can be a URL, data URI, or local path)."
        )
        duration: Optional[float] = SchemaField(
            description="Target duration (in seconds) to loop the video to. Either duration or n_loops must be provided.",
            default=None,
            ge=0.0,
            le=3600.0,  # Max 1 hour to prevent disk exhaustion
        )
        n_loops: Optional[int] = SchemaField(
            description="Number of times to repeat the video. Either n_loops or duration must be provided.",
            default=None,
            ge=1,
            le=10,  # Max 10 loops to prevent disk exhaustion
        )

    class Output(BlockSchemaOutput):
        video_out: MediaFileType = SchemaField(
            description="Looped video returned either as a relative path or a data URI."
        )

    def __init__(self):
        super().__init__(
            id="8bf9eef6-5451-4213-b265-25306446e94b",
            description="Block to loop a video to a given duration or number of repeats.",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=LoopVideoBlock.Input,
            output_schema=LoopVideoBlock.Output,
        )

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        assert execution_context.graph_exec_id is not None
        assert execution_context.node_exec_id is not None
        graph_exec_id = execution_context.graph_exec_id
        node_exec_id = execution_context.node_exec_id

        # 1) Store the input video locally
        local_video_path = await store_media_file(
            file=input_data.video_in,
            execution_context=execution_context,
            return_format="for_local_processing",
        )
        input_abspath = get_exec_file_path(graph_exec_id, local_video_path)

        # 2) Load the clip
        strip_chapters_inplace(input_abspath)
        clip = None
        looped_clip = None
        try:
            clip = VideoFileClip(input_abspath)

            # 3) Apply the loop effect
            if input_data.duration:
                # Loop until we reach the specified duration
                looped_clip = clip.with_effects([Loop(duration=input_data.duration)])
            elif input_data.n_loops:
                looped_clip = clip.with_effects([Loop(n=input_data.n_loops)])
            else:
                raise ValueError("Either 'duration' or 'n_loops' must be provided.")

            assert isinstance(looped_clip, VideoFileClip)

            # 4) Save the looped output
            source = extract_source_name(local_video_path)
            output_filename = MediaFileType(f"{node_exec_id}_looped_{source}.mp4")
            output_abspath = get_exec_file_path(graph_exec_id, output_filename)

            looped_clip = looped_clip.with_audio(clip.audio)
            looped_clip.write_videofile(
                output_abspath, codec="libx264", audio_codec="aac"
            )
        finally:
            if looped_clip:
                looped_clip.close()
            if clip:
                clip.close()

        # Return output - for_block_output returns workspace:// if available, else data URI
        video_out = await store_media_file(
            file=output_filename,
            execution_context=execution_context,
            return_format="for_block_output",
        )

        yield "video_out", video_out
