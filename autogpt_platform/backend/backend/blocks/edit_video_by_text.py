from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.type import MediaFileType


class EditVideoByTextBlock(Block):
    class Input(BlockSchema):
        video_in: MediaFileType = SchemaField(
            description="Video file to edit",
        )
        transcription: str = SchemaField(
            description="Desired transcript for the output video",
        )
        split_at: str = SchemaField(
            description="Granularity for transcript matching",
            default="word",
        )

    class Output(BlockSchema):
        video: str = SchemaField(
            description="Edited video file path",
        )
        transcription: str = SchemaField(
            description="Transcription used for editing",
        )
        error: str = SchemaField(
            description="Error message if something fails",
            default="",
        )

    def __init__(self) -> None:
        super().__init__(
            id="98d40049-a1de-465f-bba1-47411298ad1a",
            description="Edits a video by modifying its transcript.",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=EditVideoByTextBlock.Input,
            output_schema=EditVideoByTextBlock.Output,
            test_input={
                "video_in": "data:video/mp4;base64,AAAA",
                "transcription": "edited transcript",
            },
            test_output=[
                ("video", "edited_video.mp4"),
                ("transcription", "edited transcript"),
            ],
            test_mock={"edit_video": lambda path, t, s: "edited_video.mp4"},
        )

    def edit_video(self, file_path: str, transcription: str, split_at: str) -> str:
        """Placeholder editing implementation copying the source video."""
        directory = Path(file_path).parent
        output_path = directory / f"{uuid.uuid4()}_edited{Path(file_path).suffix}"
        shutil.copy(file_path, output_path)
        return str(output_path)

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
        output_path = self.edit_video(
            abs_path, input_data.transcription, input_data.split_at
        )
        if os.path.isabs(output_path):
            output_path = os.path.relpath(
                output_path, get_exec_file_path(graph_exec_id, "")
            )
        yield "video", output_path
        yield "transcription", input_data.transcription
