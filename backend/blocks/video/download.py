"""
VideoDownloadBlock - Download video from URL (YouTube, Vimeo, news sites, direct links)
"""
import uuid
from typing import Literal

from backend.data.block import Block, BlockCategory, BlockOutput
from backend.data.block import BlockSchemaInput, BlockSchemaOutput
from backend.data.model import SchemaField
from backend.util.exceptions import BlockExecutionError


class VideoDownloadBlock(Block):
    """Download video from URL using yt-dlp."""

    class Input(BlockSchemaInput):
        url: str = SchemaField(
            description="URL of the video to download (YouTube, Vimeo, direct link, etc.)",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        quality: Literal["best", "1080p", "720p", "480p", "audio_only"] = SchemaField(
            description="Video quality preference",
            default="720p"
        )
        output_format: Literal["mp4", "webm", "mkv"] = SchemaField(
            description="Output video format",
            default="mp4",
            advanced=True
        )

    class Output(BlockSchemaOutput):
        video_file: str = SchemaField(
            description="Path or data URI of downloaded video",
            json_schema_extra={"format": "file"}
        )
        duration: float = SchemaField(description="Video duration in seconds")
        title: str = SchemaField(description="Video title from source")
        source_url: str = SchemaField(description="Original source URL")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description="Download video from URL (YouTube, Vimeo, news sites, direct links)",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "quality": "480p"},
            test_output=[("video_file", str), ("duration", float), ("title", str), ("source_url", str)],
            test_mock={"_download_video": lambda *args: ("/tmp/video.mp4", 212.0, "Test Video")}
        )

    def _get_format_string(self, quality: str) -> str:
        formats = {
            "best": "bestvideo+bestaudio/best",
            "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "audio_only": "bestaudio/best"
        }
        return formats.get(quality, formats["720p"])

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            import yt_dlp
        except ImportError as e:
            raise BlockExecutionError(
                message="yt-dlp is not installed. Please install it with: pip install yt-dlp",
                block_name=self.name,
                block_id=str(self.id)
            ) from e

        video_id = str(uuid.uuid4())[:8]
        output_template = f"/tmp/{video_id}.%(ext)s"

        ydl_opts = {
            "format": self._get_format_string(input_data.quality),
            "outtmpl": output_template,
            "merge_output_format": input_data.output_format,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(input_data.url, download=True)
                video_path = ydl.prepare_filename(info)
                
                # Handle format conversion in filename
                if not video_path.endswith(f".{input_data.output_format}"):
                    video_path = video_path.rsplit(".", 1)[0] + f".{input_data.output_format}"

                yield "video_file", video_path
                yield "duration", info.get("duration") or 0.0
                yield "title", info.get("title") or "Unknown"
                yield "source_url", input_data.url

        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to download video: {e}",
                block_name=self.name,
                block_id=str(self.id)
            ) from e
