"""AddAudioToVideoBlock - Attach an audio track to a video."""

import os
from typing import Literal

from moviepy.audio.io.AudioFileClip import AudioFileClip
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


class AddAudioToVideoBlock(Block):
    """Attach an audio track to an existing video."""

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
        output_return_type: Literal["file_path", "data_uri"] = SchemaField(
            description="Return the final output as a relative path or base64 data URI.",
            default="file_path",
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
        # 1) Store the inputs locally
        local_video_path = await store_media_file(
            graph_exec_id=graph_exec_id,
            file=input_data.video_in,
            user_id=user_id,
            return_content=False,
        )
        local_audio_path = await store_media_file(
            graph_exec_id=graph_exec_id,
            file=input_data.audio_in,
            user_id=user_id,
            return_content=False,
        )

        video_abspath = get_exec_file_path(graph_exec_id, local_video_path)
        audio_abspath = get_exec_file_path(graph_exec_id, local_audio_path)

        video_clip = None
        audio_clip_original = None
        audio_clip_scaled = None
        final_clip = None
        try:
            # 2) Load video + audio with moviepy
            video_clip = VideoFileClip(video_abspath)
            audio_clip_original = AudioFileClip(audio_abspath)

            # Optionally scale volume
            audio_to_use = audio_clip_original
            if input_data.volume != 1.0:
                audio_clip_scaled = audio_clip_original.with_volume_scaled(
                    input_data.volume
                )
                audio_to_use = audio_clip_scaled

            # 3) Attach the new audio track
            final_clip = video_clip.with_audio(audio_to_use)

            # 4) Write to output file
            output_filename = MediaFileType(
                f"{node_exec_id}_audio_attached_{os.path.basename(local_video_path)}"
            )
            output_abspath = get_exec_file_path(graph_exec_id, output_filename)
            final_clip.write_videofile(
                output_abspath, codec="libx264", audio_codec="aac"
            )

            # 5) Return either path or data URI
            video_out = await store_media_file(
                graph_exec_id=graph_exec_id,
                file=output_filename,
                user_id=user_id,
                return_content=input_data.output_return_type == "data_uri",
            )

            yield "video_out", video_out
        finally:
            if final_clip:
                final_clip.close()
            if audio_clip_scaled:
                audio_clip_scaled.close()
            if audio_clip_original:
                audio_clip_original.close()
            if video_clip:
                video_clip.close()
