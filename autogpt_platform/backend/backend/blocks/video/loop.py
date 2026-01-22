"""LoopVideoBlock - Loop a video to a given duration or number of repeats."""

import os
from typing import Literal, Optional

from moviepy.video.fx.Loop import Loop
from moviepy.video.io.VideoFileClip import VideoFileClip

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.file import MediaFileType, get_exec_file_path, store_media_file


class LoopVideoBlock(Block):
    """Loop (repeat) a video clip until a given duration or number of loops."""

    class Input(BlockSchemaInput):
        video_in: MediaFileType = SchemaField(
            description="The input video (can be a URL, data URI, or local path)."
        )
        duration: Optional[float] = SchemaField(
            description="Target duration (in seconds) to loop the video to. If omitted, defaults to no looping.",
            default=None,
            ge=0.0,
        )
        n_loops: Optional[int] = SchemaField(
            description="Number of times to repeat the video. If omitted, defaults to 1 (no repeat).",
            default=None,
            ge=1,
        )
        output_return_type: Literal["file_path", "data_uri"] = SchemaField(
            description="How to return the output video. Either a relative path or base64 data URI.",
            default="file_path",
        )

    class Output(BlockSchemaOutput):
        video_out: str = SchemaField(
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
        node_exec_id: str,
        graph_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        # 1) Store the input video locally
        local_video_path = await store_media_file(
            graph_exec_id=graph_exec_id,
            file=input_data.video_in,
            user_id=user_id,
            return_content=False,
        )
        input_abspath = get_exec_file_path(graph_exec_id, local_video_path)

        clip = None
        looped_clip = None
        try:
            # 2) Load the clip
            clip = VideoFileClip(input_abspath)

            # 3) Apply the loop effect
            looped_clip = clip
            if input_data.duration:
                looped_clip = looped_clip.with_effects([Loop(duration=input_data.duration)])
            elif input_data.n_loops:
                looped_clip = looped_clip.with_effects([Loop(n=input_data.n_loops)])
            else:
                raise ValueError("Either 'duration' or 'n_loops' must be provided.")

            assert isinstance(looped_clip, VideoFileClip)

            # 4) Save the looped output
            output_filename = MediaFileType(
                f"{node_exec_id}_looped_{os.path.basename(local_video_path)}"
            )
            output_abspath = get_exec_file_path(graph_exec_id, output_filename)

            looped_clip = looped_clip.with_audio(clip.audio)
            looped_clip.write_videofile(output_abspath, codec="libx264", audio_codec="aac")

            # Return as data URI or path
            video_out = await store_media_file(
                graph_exec_id=graph_exec_id,
                file=output_filename,
                user_id=user_id,
                return_content=input_data.output_return_type == "data_uri",
            )

            yield "video_out", video_out
        finally:
            if looped_clip and looped_clip is not clip:
                looped_clip.close()
            if clip:
                clip.close()
