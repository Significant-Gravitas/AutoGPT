"""
YouTube Transcript Summarizer Block
Fetches a YouTube transcript via Supadata API and summarizes it with an LLM.
"""
import logging
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests

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
SUPADATA_ENDPOINT = "https://api.supadata.ai/v1/transcript"


class YouTubeTranscriptSummarizerBlock(AIBlockBase):
    """
    Fetches the transcript of a YouTube video via Supadata API
    and summarizes it using an LLM.

    Input  : YouTube URL + Supadata API key (optional) + LLM model
    Output : video_id, transcript (raw), summary (LLM output), error
    """

    class Input(BlockSchemaInput):
        youtube_url: str = SchemaField(
            title="YouTube URL",
            description="URL of the YouTube video to summarize.",
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
        supadata_api_key: Optional[str] = SchemaField(
            title="Supadata API Key (optional)",
            description=(
                "Required for self-hosted users. "
                "Get a free key (100 requests/month, no credit card) at supadata.ai. "
                "AutoGPT Cloud users can leave this blank."
            ),
            placeholder="Leave blank (AutoGPT Cloud) or enter your Supadata key",
            default=None,
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
                "Fetches a YouTube video transcript via Supadata API "
                "and summarizes it with an LLM. "
                "Self-hosted users need a free Supadata API key from supadata.ai."
            ),
            categories={BlockCategory.AI, BlockCategory.SOCIAL},
            test_input={
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "supadata_api_key": None,
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
                "fetch_transcript": lambda video_id, api_key: (
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

    def fetch_transcript(self, video_id: str, api_key: Optional[str]) -> str:
        """Fetch transcript via Supadata API."""
        if not api_key:
            raise RuntimeError(
                "Supadata API key is required. "
                "Get a free key at supadata.ai and enter it in the block settings."
            )

        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(
            SUPADATA_ENDPOINT,
            params={"url": url, "text": "true", "lang": "en"},
            headers={"x-api-key": api_key},
            timeout=30,
        )

        if response.status_code == 401:
            raise RuntimeError("Invalid Supadata API key.")
        if response.status_code == 404:
            raise RuntimeError("No transcript found for this video.")
        if not response.ok:
            raise RuntimeError(
                f"Supadata API error {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        content = data.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if item.get("text")
            )

        if not content.strip():
            raise RuntimeError("Transcript is empty for this video.")

        return content

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
            transcript = self.fetch_transcript(video_id, input_data.supadata_api_key)
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
