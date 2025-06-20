from __future__ import annotations

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.type import MediaFileType


class TranscribeVideoBlock(Block):
    class Input(BlockSchema):
        video_in: MediaFileType = SchemaField(
            description="Video file to transcribe",
        )

    class Output(BlockSchema):
        transcription: str = SchemaField(
            description="Text transcription of the video",
        )
        error: str = SchemaField(
            description="Error message if something fails",
            default="",
        )

    def __init__(self) -> None:
        super().__init__(
            id="fa49dad0-a5fc-441c-ba04-2ac206e392d8",
            description="Transcribes speech from a video file.",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=TranscribeVideoBlock.Input,
            output_schema=TranscribeVideoBlock.Output,
            test_input={"video_in": "data:video/mp4;base64,AAAA"},
            test_output=("transcription", "example transcript"),
            test_mock={"transcribe": lambda path: "example transcript"},
        )

    def transcribe(self, file_path: str) -> str:
        """Placeholder transcription implementation."""
        raise NotImplementedError

    def run(
        self,
        input_data: Input,
        *,
        graph_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        local_path = store_media_file(
            graph_exec_id=graph_exec_id, file=input_data.video_in, return_content=False
        )
        abs_path = get_exec_file_path(graph_exec_id, local_path)
        try:
            transcript = self.transcribe(abs_path)
            yield "transcription", transcript
        except Exception as e:
            yield "error", str(e)
