from urllib.parse import parse_qs, urlparse

from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
from youtube_transcript_api._transcripts import FetchedTranscript
from youtube_transcript_api.formatters import TextFormatter

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField


class TranscribeYoutubeVideoBlock(Block):
    class Input(BlockSchemaInput):
        youtube_url: str = SchemaField(
            title="YouTube URL",
            description="The URL of the YouTube video to transcribe",
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )

    class Output(BlockSchemaOutput):
        video_id: str = SchemaField(description="The extracted YouTube video ID")
        transcript: str = SchemaField(description="The transcribed text of the video")
        error: str = SchemaField(
            description="Any error message if the transcription fails"
        )

    def __init__(self):
        super().__init__(
            id="f3a8f7e1-4b1d-4e5f-9f2a-7c3d5a2e6b4c",
            input_schema=TranscribeYoutubeVideoBlock.Input,
            output_schema=TranscribeYoutubeVideoBlock.Output,
            description="Transcribes a YouTube video.",
            categories={BlockCategory.SOCIAL},
            test_input={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            test_output=[
                ("video_id", "dQw4w9WgXcQ"),
                (
                    "transcript",
                    "Never gonna give you up\nNever gonna let you down",
                ),
            ],
            test_mock={
                "get_transcript": lambda video_id: [
                    {"text": "Never gonna give you up"},
                    {"text": "Never gonna let you down"},
                ],
                "format_transcript": lambda transcript: "Never gonna give you up\nNever gonna let you down",
            },
        )

    @staticmethod
    def extract_video_id(url: str) -> str:
        parsed_url = urlparse(url)
        if parsed_url.netloc == "youtu.be":
            return parsed_url.path[1:]
        if parsed_url.netloc in ("www.youtube.com", "youtube.com"):
            if parsed_url.path == "/watch":
                p = parse_qs(parsed_url.query)
                return p["v"][0]
            if parsed_url.path[:7] == "/embed/":
                return parsed_url.path.split("/")[2]
            if parsed_url.path[:3] == "/v/":
                return parsed_url.path.split("/")[2]
        raise ValueError(f"Invalid YouTube URL: {url}")

    @staticmethod
    def get_transcript(video_id: str) -> FetchedTranscript:
        """
        Get transcript for a video, preferring English but falling back to any available language.

        :param video_id: The YouTube video ID
        :return: The fetched transcript
        :raises: Any exception except NoTranscriptFound for requested languages
        """
        api = YouTubeTranscriptApi()
        try:
            # Try to get English transcript first (default behavior)
            return api.fetch(video_id=video_id)
        except NoTranscriptFound:
            # If English is not available, get the first available transcript
            transcript_list = api.list(video_id)
            # Try manually created transcripts first, then generated ones
            available_transcripts = list(
                transcript_list._manually_created_transcripts.values()
            ) + list(transcript_list._generated_transcripts.values())
            if available_transcripts:
                # Fetch the first available transcript
                return available_transcripts[0].fetch()
            # If no transcripts at all, re-raise the original error
            raise

    @staticmethod
    def format_transcript(transcript: FetchedTranscript) -> str:
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)
        return transcript_text

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        video_id = self.extract_video_id(input_data.youtube_url)
        yield "video_id", video_id

        transcript = self.get_transcript(video_id)
        transcript_text = self.format_transcript(transcript=transcript)

        yield "transcript", transcript_text
