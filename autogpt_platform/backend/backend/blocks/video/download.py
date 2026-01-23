"""VideoDownloadBlock - Download video from URL (YouTube, Vimeo, news sites, direct links)."""

import os
import tempfile
import uuid
from typing import Literal

import yt_dlp
from yt_dlp import _Params

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.exceptions import BlockExecutionError


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
        video_file: str = SchemaField(
            description="Path or data URI of downloaded video",
            json_schema_extra={"format": "file"},
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
                "_download_video": lambda *args: ("/tmp/video.mp4", 212.0, "Test Video")
            },
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
    ) -> tuple[str, float, str]:
        """Download video. Extracted for testability."""
        video_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        output_template = os.path.join(temp_dir, f"{video_id}.%(ext)s")

        ydl_opts: _Params = {
            "format": self._get_format_string(quality),
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

            return (
                video_path,
                info.get("duration") or 0.0,
                info.get("title") or "Unknown",
            )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            video_path, duration, title = self._download_video(
                input_data.url,
                input_data.quality,
                input_data.output_format,
            )
            yield "video_file", video_path
            yield "duration", duration
            yield "title", title
            yield "source_url", input_data.url

        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to download video: {e}",
                block_name=self.name,
                block_id=str(self.id),
            ) from e
