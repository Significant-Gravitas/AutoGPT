import re
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

from autogpt_server.data.block import Block, BlockSchema, BlockOutput

class YouTubeTranscriber(Block):
    class Input(BlockSchema):
        youtube_url: str

    class Output(BlockSchema):
        video_id: str
        transcript: str
        error: str

    def __init__(self):
        super().__init__(
            id="f3a8f7e1-4b1d-4e5f-9f2a-7c3d5a2e6b4c",
            input_schema=YouTubeTranscriber.Input,
            output_schema=YouTubeTranscriber.Output,
            test_input={
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            },
            test_output=("transcript", "Never gonna give you up\nNever gonna let you down\n...")
        )

    @staticmethod
    def extract_video_id(url: str) -> str:
        # Try to extract the video ID from the URL
        parsed_url = urlparse(url)
        if parsed_url.netloc == 'youtu.be':
            return parsed_url.path[1:]
        if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                p = parse_qs(parsed_url.query)
                return p['v'][0]
            if parsed_url.path[:7] == '/embed/':
                return parsed_url.path.split('/')[2]
            if parsed_url.path[:3] == '/v/':
                return parsed_url.path.split('/')[2]
        # If we get here, we can't handle the URL
        raise ValueError(f"Invalid YouTube URL: {url}")

    def run(self, input_data: Input) -> BlockOutput:
        try:
            video_id = self.extract_video_id(input_data.youtube_url)
            yield "video_id", video_id

            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            formatter = TextFormatter()
            transcript_text = formatter.format_transcript(transcript)

            yield "transcript", transcript_text
        except Exception as e:
            yield "error", str(e)