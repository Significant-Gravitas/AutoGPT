from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TranscribeYoutubeVideoBlock(Block):
    class Input(BlockSchema):
        youtube_url: str = SchemaField(
            title="YouTube URL",
            description="The URL of the YouTube video to transcribe",
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )

    class Output(BlockSchema):
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
    def get_transcript(video_id: str):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            if not transcript_list:
                raise ValueError(f"No transcripts found for the video: {video_id}")

            for transcript in transcript_list:
                first_transcript = transcript_list.find_transcript(
                    [transcript.language_code]
                )
                return YouTubeTranscriptApi.get_transcript(
                    video_id, languages=[first_transcript.language_code]
                )

        except Exception:
            raise ValueError(f"No transcripts found for the video: {video_id}")

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        video_id = self.extract_video_id(input_data.youtube_url)
        yield "video_id", video_id

        transcript = self.get_transcript(video_id)
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)

        yield "transcript", transcript_text
