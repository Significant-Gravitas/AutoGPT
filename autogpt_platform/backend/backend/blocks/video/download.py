"""VideoDownloadBlock - Download video from URL (YouTube, Vimeo, news sites, direct links)."""

import os
import typing
from typing import Literal

import yt_dlp

if typing.TYPE_CHECKING:
    from yt_dlp import _Params

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


class VideoDownloadBlock(Block):
    """Download video from URL using yt-dlp."""

    class Input(BlockSchemaInput):
        url: str = SchemaField(
            description="URL of the video to download (YouTube, Vimeo, direct link, etc.)",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        quality: Literal["best", "1080p", "720p", "480p", "audio_only"] = SchemaField(
            description="Video quality preference", default="720p"
        )
        output_format: Literal["mp4", "webm", "mkv"] = SchemaField(
            description="Output video format", default="mp4", advanced=True
        )

    class Output(BlockSchemaOutput):
        video_file: MediaFileType = SchemaField(
            description="Downloaded video (path or data URI)"
        )
        duration: float = SchemaField(description="Video duration in seconds")
        title: str = SchemaField(description="Video title from source")
        source_url: str = SchemaField(description="Original source URL")

    def __init__(self):
        super().__init__(
            id="c35daabb-cd60-493b-b9ad-51f1fe4b50c4",
            description="Download video from URL (YouTube, Vimeo, news sites, direct links)",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=self.Input,
            output_schema=self.Output,
            disabled=True,  # Disable until we can sandbox yt-dlp and handle security implications
            test_input={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "quality": "480p",
            },
            test_output=[
                ("video_file", str),
                ("duration", float),
                ("title", str),
                ("source_url", str),
            ],
            test_mock={
                "_download_video": lambda *args: (
                    "video.mp4",
                    212.0,
                    "Test Video",
                ),
                "_store_output_video": lambda *args, **kwargs: "video.mp4",
            },
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

    def _get_format_string(self, quality: str) -> str:
        formats = {
            "best": "bestvideo+bestaudio/best",
            "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "audio_only": "bestaudio/best",
        }
        return formats.get(quality, formats["720p"])

    def _download_video(
        self,
        url: str,
        quality: str,
        output_format: str,
        output_dir: str,
        node_exec_id: str,
    ) -> tuple[str, float, str]:
        """Download video. Extracted for testability."""
        output_template = os.path.join(
            output_dir, f"{node_exec_id}_%(title).50s.%(ext)s"
        )

        ydl_opts: "_Params" = {
            "format": f"{self._get_format_string(quality)}/best",
            "outtmpl": output_template,
            "merge_output_format": output_format,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)

            # Handle format conversion in filename
            if not video_path.endswith(f".{output_format}"):
                video_path = video_path.rsplit(".", 1)[0] + f".{output_format}"

            # Return just the filename, not the full path
            filename = os.path.basename(video_path)

            return (
                filename,
                info.get("duration") or 0.0,
                info.get("title") or "Unknown",
            )

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        node_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            assert execution_context.graph_exec_id is not None

            # Get the exec file directory
            output_dir = get_exec_file_path(execution_context.graph_exec_id, "")
            os.makedirs(output_dir, exist_ok=True)

            filename, duration, title = self._download_video(
                input_data.url,
                input_data.quality,
                input_data.output_format,
                output_dir,
                node_exec_id,
            )

            # Return as workspace path or data URI based on context
            video_out = await self._store_output_video(
                execution_context, MediaFileType(filename)
            )

            yield "video_file", video_out
            yield "duration", duration
            yield "title", title
            yield "source_url", input_data.url

        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to download video: {e}",
                block_name=self.name,
                block_id=str(self.id),
            ) from e
