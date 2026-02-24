"""MediaDurationBlock - Get the duration of a media file."""

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.video._utils import strip_chapters_inplace
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.file import MediaFileType, get_exec_file_path, store_media_file


class MediaDurationBlock(Block):
    """Get the duration of a media file (video or audio)."""

    class Input(BlockSchemaInput):
        media_in: MediaFileType = SchemaField(
            description="Media input (URL, data URI, or local path)."
        )
        is_video: bool = SchemaField(
            description="Whether the media is a video (True) or audio (False).",
            default=True,
        )

    class Output(BlockSchemaOutput):
        duration: float = SchemaField(
            description="Duration of the media file (in seconds)."
        )

    def __init__(self):
        super().__init__(
            id="d8b91fd4-da26-42d4-8ecb-8b196c6d84b6",
            description="Block to get the duration of a media file.",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=MediaDurationBlock.Input,
            output_schema=MediaDurationBlock.Output,
        )

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        # 1) Store the input media locally
        local_media_path = await store_media_file(
            file=input_data.media_in,
            execution_context=execution_context,
            return_format="for_local_processing",
        )
        assert execution_context.graph_exec_id is not None
        media_abspath = get_exec_file_path(
            execution_context.graph_exec_id, local_media_path
        )

        # 2) Strip chapters to avoid MoviePy crash, then load the clip
        strip_chapters_inplace(media_abspath)
        clip = None
        try:
            if input_data.is_video:
                clip = VideoFileClip(media_abspath)
            else:
                clip = AudioFileClip(media_abspath)

            duration = clip.duration
        finally:
            if clip:
                clip.close()

        yield "duration", duration
