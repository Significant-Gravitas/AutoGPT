"""AddAudioToVideoBlock - Attach an audio track to a video file."""

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.video._render import add_audio_to_video
from backend.blocks.video._utils import extract_source_name, run_edit
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.file import MediaFileType, get_exec_file_path, store_media_file


class AddAudioToVideoBlock(Block):
    """Add (attach) an audio track to an existing video."""

    class Input(BlockSchemaInput):
        video_in: MediaFileType = SchemaField(
            description="Video input (URL, data URI, or local path)."
        )
        audio_in: MediaFileType = SchemaField(
            description="Audio input (URL, data URI, or local path)."
        )
        volume: float = SchemaField(
            description="Volume scale for the newly attached audio track (1.0 = original).",
            default=1.0,
        )

    class Output(BlockSchemaOutput):
        video_out: MediaFileType = SchemaField(
            description="Final video (with attached audio), as a path or data URI."
        )

    def __init__(self):
        super().__init__(
            id="3503748d-62b6-4425-91d6-725b064af509",
            description="Block to attach an audio file to a video file using moviepy.",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=AddAudioToVideoBlock.Input,
            output_schema=AddAudioToVideoBlock.Output,
            effect=BlockEffect.WORKSPACE,
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

        # 1) Store the inputs locally
        local_video_path = await store_media_file(
            file=input_data.video_in,
            execution_context=execution_context,
            return_format="for_local_processing",
        )
        local_audio_path = await store_media_file(
            file=input_data.audio_in,
            execution_context=execution_context,
            return_format="for_local_processing",
        )

        video_abspath = get_exec_file_path(graph_exec_id, local_video_path)
        audio_abspath = get_exec_file_path(graph_exec_id, local_audio_path)

        # 2) Attach the audio track, writing the output file
        source = extract_source_name(local_video_path)
        output_filename = MediaFileType(f"{node_exec_id}_with_audio_{source}.mp4")
        output_abspath = get_exec_file_path(graph_exec_id, output_filename)
        await run_edit(
            self,
            add_audio_to_video,
            [video_abspath, audio_abspath],
            output_abspath,
            input_data.volume,
        )

        # 3) Return output - for_block_output returns workspace:// if available, else data URI
        video_out = await store_media_file(
            file=output_filename,
            execution_context=execution_context,
            return_format="for_block_output",
        )

        yield "video_out", video_out
