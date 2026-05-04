"""
Unit tests for YouTubeTranscriptSummarizerBlock.

Tests cover:
- URL parsing (various YouTube URL formats)
- Transcript fetch error handling
- Full block execution via execute_block_test (mocked)
"""

import pytest

from backend.blocks.youtube_summarizer import YouTubeTranscriptSummarizerBlock
from backend.util.test import execute_block_test

TEST_API_KEY = "test-api-key"

# ---------------------------------------------------------------------------
# extract_video_id
# ---------------------------------------------------------------------------


class TestExtractVideoId:
    """Tests for the static URL-parsing helper."""

    def _extract(self, url: str) -> str:
        return YouTubeTranscriptSummarizerBlock.extract_video_id(url)

    def test_standard_watch_url(self):
        assert (
            self._extract("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            == "dQw4w9WgXcQ"
        )

    def test_short_youtu_be_url(self):
        assert self._extract("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert (
            self._extract("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        )

    def test_v_url(self):
        assert self._extract("https://www.youtube.com/v/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        assert (
            self._extract("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        )

    def test_url_without_www(self):
        assert self._extract("https://youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PLxxx"
        assert self._extract(url) == "dQw4w9WgXcQ"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Cannot extract video ID"):
            self._extract("https://vimeo.com/12345678")

    def test_empty_string_raises(self):
        with pytest.raises((ValueError, KeyError)):
            self._extract("")


# ---------------------------------------------------------------------------
# fetch_transcript — error paths
# ---------------------------------------------------------------------------


class _MockResponse:
    """Minimal requests.Response stand-in for monkeypatching."""

    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = ""
        self._body = body or {}

    def json(self) -> dict:
        return self._body


class TestFetchTranscriptErrors:
    """Verify that Supadata HTTP errors are translated into RuntimeError."""

    def _block(self) -> YouTubeTranscriptSummarizerBlock:
        return YouTubeTranscriptSummarizerBlock()

    def test_invalid_api_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(
            "backend.blocks.youtube_summarizer.requests.get",
            lambda *a, **kw: _MockResponse(401),
        )
        block = self._block()
        with pytest.raises(RuntimeError, match="Invalid"):
            block.fetch_transcript("vid123", TEST_API_KEY)

    def test_no_transcript_found_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(
            "backend.blocks.youtube_summarizer.requests.get",
            lambda *a, **kw: _MockResponse(404),
        )
        block = self._block()
        with pytest.raises(RuntimeError, match="No transcript"):
            block.fetch_transcript("vid123", TEST_API_KEY)

    def test_supadata_api_error_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(
            "backend.blocks.youtube_summarizer.requests.get",
            lambda *a, **kw: _MockResponse(503),
        )
        block = self._block()
        with pytest.raises(RuntimeError, match="Supadata API error"):
            block.fetch_transcript("vid123", TEST_API_KEY)


# ---------------------------------------------------------------------------
# Full block — mock-based execution
# ---------------------------------------------------------------------------


async def test_block_with_mocks():
    """
    Run the full block pipeline using the built-in test_input / test_output /
    test_mock defined on the block — no real network or API calls made.
    """
    block = YouTubeTranscriptSummarizerBlock()
    await execute_block_test(block)


async def test_block_fetch_error_yields_error_field():
    """
    If fetch_transcript raises, the block must yield an 'error' output
    and must NOT yield video_id / transcript / summary.
    """
    from backend.blocks.llm import TEST_CREDENTIALS, LLMResponse

    block = YouTubeTranscriptSummarizerBlock()

    # Mock fetch_transcript to simulate a failure
    def bad_fetch(video_id: str, api_key: str | None) -> str:
        raise RuntimeError("Transcripts are disabled for this video.")

    block.fetch_transcript = bad_fetch  # type: ignore[method-assign]

    # Also mock llm_call so it is never called
    llm_called = False

    async def should_not_call(*args, **kwargs) -> LLMResponse:
        nonlocal llm_called
        llm_called = True
        return LLMResponse(
            raw_response="",
            prompt=[],
            response="",
            tool_calls=None,
            prompt_tokens=0,
            completion_tokens=0,
            reasoning=None,
        )

    block.llm_call = should_not_call  # type: ignore[method-assign]

    import uuid

    from backend.data.execution import ExecutionContext

    graph_id = str(uuid.uuid4())
    node_id = str(uuid.uuid4())
    graph_exec_id = str(uuid.uuid4())
    node_exec_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    input_data = YouTubeTranscriptSummarizerBlock.Input(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        custom_prompt=None,
        credentials={
            "provider": "openai",
            "id": str(uuid.uuid4()),
            "type": "api_key",
            "title": "test",
        },  # type: ignore[arg-type]
    )

    outputs = {}
    async for key, value in block.run(
        input_data,
        credentials=TEST_CREDENTIALS,
        graph_id=graph_id,
        node_id=node_id,
        graph_exec_id=graph_exec_id,
        node_exec_id=node_exec_id,
        user_id=user_id,
        graph_version=1,
        execution_context=ExecutionContext(
            user_id=user_id,
            graph_id=graph_id,
            graph_exec_id=graph_exec_id,
            graph_version=1,
            node_id=node_id,
            node_exec_id=node_exec_id,
        ),
    ):
        outputs[key] = value

    assert "error" in outputs, "Expected 'error' output when fetch fails"
    assert "Transcripts are disabled" in outputs["error"]
    assert "video_id" not in outputs
    assert "summary" not in outputs
    assert not llm_called, "LLM should not be called when transcript fetch fails"
