"""
YouTube Transcript Summarizer Block
Fetches a YouTube transcript and processes it with an LLM.
Does not require a proxy - uses the youtube-transcript-api directly.
"""
import logging
from typing import Optional
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from backend.blocks._base import (
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.llm import (
    DEFAULT_LLM_MODEL,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    AIBlockBase,
    AICredentials,
    AICredentialsField,
    LlmModel,
    LLMResponse,
    llm_call,
)
from backend.data.model import APIKeyCredentials, NodeExecutionStats, SchemaField

logger = logging.getLogger(__name__)

MAX_TRANSCRIPT_CHARS = 12_000


class YouTubeTranscriptSummarizerBlock(AIBlockBase):
    """
    Fetches the transcript of a YouTube video and summarizes it using an LLM.

    Input  : YouTube URL + optional custom prompt + LLM model choice
    Output : video_id, transcript (raw), summary (LLM output), error
    """

    class Input(BlockSchemaInput):
        youtube_url: str = SchemaField(
            title="YouTube URL",
            description="URL of the YouTube video to summarize.",
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
        custom_prompt: Optional[str] = SchemaField(
            title="Custom Prompt (optional)",
            description=(
                "Override the default summarization instruction. "
                "Leave blank for a standard bullet-point summary."
            ),
            default=None,
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default=DEFAULT_LLM_MODEL,
            description="Language model used to summarize the transcript.",
            advanced=False,
        )
        credentials: AICredentials = AICredentialsField()

    class Output(BlockSchemaOutput):
        video_id: str = SchemaField(description="Extracted YouTube video ID.")
        transcript: str = SchemaField(description="Raw transcript text from YouTube.")
        summary: str = SchemaField(description="LLM-generated summary of the video.")
        error: str = SchemaField(description="Error message if the block fails.")

    def __init__(self):
        super().__init__(
            id="297863b7-bc21-44a5-803b-ebb5c456f9ae",
            input_schema=YouTubeTranscriptSummarizerBlock.Input,
            output_schema=YouTubeTranscriptSummarizerBlock.Output,
            description=(
                "Fetches a YouTube video transcript and summarizes it with an LLM. "
                "No proxy required."
            ),
            categories={BlockCategory.AI, BlockCategory.SOCIAL},
            test_input={
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "custom_prompt": None,
                "model": DEFAULT_LLM_MODEL,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("video_id", "dQw4w9WgXcQ"),
                ("transcript", "Never gonna give you up Never gonna let you down"),
                ("summary", "The video is a classic 80s pop song by Rick Astley."),
            ],
            test_mock={
                "fetch_transcript": lambda video_id: (
                    "Never gonna give you up Never gonna let you down"
                ),
                "llm_call": lambda *args, **kwargs: LLMResponse(
                    raw_response="",
                    prompt=[],
                    response="The video is a classic 80s pop song by Rick Astley.",
                    tool_calls=None,
                    prompt_tokens=80,
                    completion_tokens=15,
                    reasoning=None,
                ),
            },
        )

    # ------------------------------------------------------------------
    # Helpers (mockable for tests)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_video_id(url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc == "youtu.be":
            return parsed.path[1:]
        if parsed.netloc in ("www.youtube.com", "youtube.com"):
            if parsed.path == "/watch":
                return parse_qs(parsed.query)["v"][0]
            if parsed.path.startswith(("/embed/", "/v/", "/shorts/")):
                return parsed.path.split("/")[2]
        raise ValueError(f"Cannot extract video ID from: {url}")

    def fetch_transcript(self, video_id: str) -> str:
        """Fetch and concatenate all transcript snippets for a video."""
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id)
            return " ".join(snippet.text for snippet in fetched)
        except TranscriptsDisabled:
            raise RuntimeError("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            raise RuntimeError("No transcript available for this video.")
        except VideoUnavailable:
            raise RuntimeError("Video is unavailable or private.")
        except CouldNotRetrieveTranscript as exc:
            raise RuntimeError(f"Could not retrieve transcript: {exc}")

    async def llm_call(
        self,
        credentials: APIKeyCredentials,
        llm_model: LlmModel,
        prompt: list,
        max_tokens: int,
    ) -> LLMResponse:
        return await llm_call(
            credentials=credentials,
            llm_model=llm_model,
            prompt=prompt,
            force_json_output=False,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            video_id = self.extract_video_id(input_data.youtube_url)
            transcript = self.fetch_transcript(video_id)
        except Exception as exc:
            yield "error", str(exc)
            return

        truncated = transcript[:MAX_TRANSCRIPT_CHARS]
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            truncated += "\n\n[Transcript truncated for length]"

        system_instruction = input_data.custom_prompt or (
            "You are a helpful assistant. Read the YouTube video transcript below "
            "and produce a concise summary with clear bullet points covering the "
            "main ideas, key takeaways, and any action items mentioned."
        )

        prompt = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Transcript:\n\n{truncated}"},
        ]

        try:
            response = await self.llm_call(
                credentials=credentials,
                llm_model=input_data.model,
                prompt=prompt,
                max_tokens=1024,
            )
        except Exception as exc:
            yield "error", f"LLM error: {exc}"
            return

        self.merge_stats(
            NodeExecutionStats(
                input_token_count=response.prompt_tokens,
                output_token_count=response.completion_tokens,
                cache_read_token_count=response.cache_read_tokens,
                cache_creation_token_count=response.cache_creation_tokens,
                provider_cost=response.provider_cost,
            )
        )

        yield "video_id", video_id
        yield "transcript", transcript
        yield "summary", response.response
