"""Tests for the ``web_search`` copilot tool.

Covers the annotation extractor + cost extractor as pure units (fed
with real ``openai`` SDK types — no duck-typed ``SimpleNamespace``
stand-ins), plus integration tests exercising both the quick
(``perplexity/sonar``) and deep (``perplexity/sonar-deep-research``)
paths — mocking ``AsyncOpenAI.chat.completions.create`` and confirming
the handler plumbs through to ``persist_and_record_usage`` with
``provider='open_router'`` and the real ``usage.cost`` value.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import (
    Annotation,
    AnnotationURLCitation,
    ChatCompletionMessage,
)

from backend.copilot.model import ChatSession

from .models import ErrorResponse, WebSearchResponse
from .web_search import (
    WebSearchTool,
    _extract_answer,
    _extract_cost_usd,
    _extract_results,
)


def _usage(
    *,
    prompt_tokens: int = 120,
    completion_tokens: int = 40,
    cost: object = 0.01,
) -> CompletionUsage:
    """Typed ``CompletionUsage`` with OpenRouter's ``cost`` extension
    parked in ``model_extra`` — the same channel the production code
    reads it from.  ``model_construct`` preserves unknown fields;
    ``model_validate`` would drop them because ``CompletionUsage``
    treats the schema as strict."""
    payload: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cost is not None:
        payload["cost"] = cost
    return CompletionUsage.model_construct(None, **payload)


def _citation(*, url: str, title: str, content: str | None = None) -> Annotation:
    """Typed ``Annotation`` for a URL citation.  ``content`` is an
    OpenRouter extension on the otherwise-typed schema — goes into
    ``url_citation.model_extra`` when model_construct preserves it."""
    payload: dict[str, Any] = {
        "url": url,
        "title": title,
        "start_index": 0,
        "end_index": len(title),
    }
    if content is not None:
        payload["content"] = content
    url_citation = AnnotationURLCitation.model_construct(None, **payload)
    return Annotation(type="url_citation", url_citation=url_citation)


def _fake_response(
    *,
    citations: list[dict] | None = None,
    answer: str = "ok",
    prompt_tokens: int = 120,
    completion_tokens: int = 40,
    cost: object = 0.01,
) -> ChatCompletion:
    """Build a typed ``ChatCompletion`` shaped like an OpenRouter
    response — typed end-to-end so the production code's attribute
    access runs under the real SDK types in tests."""
    annotations = [
        _citation(
            url=c.get("url", ""),
            title=c.get("title", "untitled"),
            content=c.get("content"),
        )
        for c in citations or []
    ]
    message = ChatCompletionMessage.model_construct(
        None,
        role="assistant",
        content=answer,
        annotations=annotations,
    )
    choice = Choice.model_construct(
        None,
        index=0,
        finish_reason="stop",
        message=message,
    )
    return ChatCompletion.model_construct(
        None,
        id="cmpl-test",
        object="chat.completion",
        created=0,
        model="perplexity/sonar",
        choices=[choice],
        usage=_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
        ),
    )


class TestExtractResults:
    """Pin the annotation shape — a schema bump in the OpenAI SDK or
    OpenRouter surfaces here first.  Same extractor serves both tiers
    because OpenRouter normalises annotations across models."""

    def test_extracts_title_url_and_content_snippet(self):
        resp = _fake_response(
            citations=[
                {
                    "title": "Kimi K2.6 launch",
                    "url": "https://example.com/kimi",
                    "content": "Moonshot released K2.6 on 2026-04-20.",
                },
                {
                    "title": "OpenRouter pricing",
                    "url": "https://openrouter.ai/moonshotai/kimi-k2.6",
                },
            ]
        )
        out = _extract_results(resp, limit=10)
        assert len(out) == 2
        assert out[0].title == "Kimi K2.6 launch"
        assert out[0].url == "https://example.com/kimi"
        assert out[0].snippet.startswith("Moonshot released")
        # Missing ``content`` extension → empty snippet rather than crash.
        assert out[1].snippet == ""

    def test_limit_caps_returned_results(self):
        resp = _fake_response(
            citations=[{"title": f"r{i}", "url": f"https://e/{i}"} for i in range(10)]
        )
        out = _extract_results(resp, limit=3)
        assert len(out) == 3
        assert [r.title for r in out] == ["r0", "r1", "r2"]

    def test_missing_choices_returns_empty(self):
        resp = ChatCompletion.model_construct(
            None,
            id="cmpl-test",
            object="chat.completion",
            created=0,
            model="perplexity/sonar",
            choices=[],
            usage=_usage(),
        )
        assert _extract_results(resp, limit=10) == []

    def test_extract_answer_returns_message_content(self):
        resp = _fake_response(
            answer="Sonar's synthesised, web-grounded answer text.",
            citations=[{"title": "t", "url": "https://e"}],
        )
        assert _extract_answer(resp) == "Sonar's synthesised, web-grounded answer text."

    def test_extract_answer_returns_empty_when_no_choices(self):
        resp = ChatCompletion.model_construct(
            None,
            id="cmpl-test",
            object="chat.completion",
            created=0,
            model="perplexity/sonar",
            choices=[],
            usage=_usage(),
        )
        assert _extract_answer(resp) == ""

    def test_snippet_clamped_to_max_chars(self):
        long_body = "x" * 5000
        resp = _fake_response(
            citations=[{"title": "t", "url": "https://e", "content": long_body}]
        )
        out = _extract_results(resp, limit=1)
        assert len(out) == 1
        assert len(out[0].snippet) == 500


class TestExtractCostUsd:
    """Read real ``usage.cost`` via typed ``model_extra`` — no
    hard-coded rates, so a future provider price change is reflected
    automatically.  Error handling mirrors the baseline service's
    ``_extract_usage_cost``."""

    def test_returns_cost_value(self):
        assert _extract_cost_usd(_usage(cost=0.023456)) == pytest.approx(0.023456)

    def test_returns_none_when_usage_missing(self):
        assert _extract_cost_usd(None) is None

    def test_returns_none_when_cost_field_missing(self):
        assert _extract_cost_usd(_usage(cost=None)) is None

    def test_returns_none_when_cost_is_explicit_null(self):
        usage = CompletionUsage.model_construct(
            None, prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=None
        )
        assert _extract_cost_usd(usage) is None

    def test_returns_none_when_cost_is_negative(self):
        usage = CompletionUsage.model_construct(
            None, prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=-1.0
        )
        assert _extract_cost_usd(usage) is None

    def test_accepts_numeric_string(self):
        usage = CompletionUsage.model_construct(
            None, prompt_tokens=0, completion_tokens=0, total_tokens=0, cost="0.017"
        )
        assert _extract_cost_usd(usage) == pytest.approx(0.017)


class TestWebSearchToolDispatch:
    """Integration test: mock the OpenAI client, confirm both paths
    dispatch the right Sonar model + track cost."""

    def _session(self) -> ChatSession:
        s = ChatSession.new("test-user", dry_run=False)
        s.session_id = "sess-1"
        return s

    def _mock_client(self, fake_resp: ChatCompletion) -> Any:
        return type(
            "MC",
            (),
            {
                "chat": type(
                    "C",
                    (),
                    {
                        "completions": type(
                            "CC",
                            (),
                            {"create": AsyncMock(return_value=fake_resp)},
                        )()
                    },
                )()
            },
        )()

    @pytest.mark.asyncio
    async def test_quick_path_uses_sonar_base(self, monkeypatch):
        fake_resp = _fake_response(
            citations=[
                {
                    "title": "hello",
                    "url": "https://example.com",
                    "content": "greeting",
                }
            ],
            answer="Kimi K2.6 launched 2026-04-20 [1].",
            cost=0.01,
        )
        mock_client = self._mock_client(fake_resp)

        monkeypatch.setattr(
            "backend.copilot.tools.web_search._chat_config",
            type(
                "C",
                (),
                {
                    "api_key": "sk-test",
                    "base_url": "https://openrouter.ai/api/v1",
                },
            )(),
        )

        with (
            patch(
                "backend.copilot.tools.web_search.AsyncOpenAI",
                return_value=mock_client,
            ),
            patch(
                "backend.copilot.tools.web_search.persist_and_record_usage",
                new=AsyncMock(return_value=160),
            ) as mock_track,
        ):
            tool = WebSearchTool()
            result = await tool._execute(
                user_id="u1",
                session=self._session(),
                query="kimi k2.6 launch",
                max_results=5,
                deep=False,
            )

        assert isinstance(result, WebSearchResponse)
        assert result.answer == "Kimi K2.6 launched 2026-04-20 [1]."
        assert len(result.results) == 1
        assert result.results[0].snippet == "greeting"

        create_call = mock_client.chat.completions.create.call_args
        assert create_call.kwargs["model"] == "perplexity/sonar"
        # Sonar searches natively — no server-tool extras.
        assert create_call.kwargs["extra_body"] == {"usage": {"include": True}}

        kwargs = mock_track.await_args.kwargs
        assert kwargs["provider"] == "open_router"
        assert kwargs["model"] == "perplexity/sonar"
        assert kwargs["cost_usd"] == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_deep_path_uses_sonar_deep_research(self, monkeypatch):
        fake_resp = _fake_response(
            citations=[
                {
                    "title": "deep find",
                    "url": "https://example.com/deep",
                    "content": "research body",
                }
            ],
            cost=0.087,
        )
        mock_client = self._mock_client(fake_resp)

        monkeypatch.setattr(
            "backend.copilot.tools.web_search._chat_config",
            type(
                "C",
                (),
                {
                    "api_key": "sk-test",
                    "base_url": "https://openrouter.ai/api/v1",
                },
            )(),
        )

        with (
            patch(
                "backend.copilot.tools.web_search.AsyncOpenAI",
                return_value=mock_client,
            ),
            patch(
                "backend.copilot.tools.web_search.persist_and_record_usage",
                new=AsyncMock(return_value=160),
            ) as mock_track,
        ):
            tool = WebSearchTool()
            result = await tool._execute(
                user_id="u1",
                session=self._session(),
                query="research question",
                deep=True,
            )

        assert isinstance(result, WebSearchResponse)
        create_call = mock_client.chat.completions.create.call_args
        assert create_call.kwargs["model"] == "perplexity/sonar-deep-research"

        kwargs = mock_track.await_args.kwargs
        assert kwargs["provider"] == "open_router"
        assert kwargs["model"] == "perplexity/sonar-deep-research"
        assert kwargs["cost_usd"] == pytest.approx(0.087)

    @pytest.mark.asyncio
    async def test_missing_credentials_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            "backend.copilot.tools.web_search._chat_config",
            type("C", (), {"api_key": "", "base_url": ""})(),
        )
        openai_stub = AsyncMock()
        with (
            patch(
                "backend.copilot.tools.web_search.AsyncOpenAI",
                return_value=openai_stub,
            ),
            patch(
                "backend.copilot.tools.web_search.persist_and_record_usage",
                new=AsyncMock(),
            ) as mock_track,
        ):
            tool = WebSearchTool()
            assert tool.is_available is False
            result = await tool._execute(
                user_id="u1",
                session=self._session(),
                query="anything",
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "web_search_not_configured"
        openai_stub.chat.completions.create.assert_not_called()
        mock_track.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_query_rejected_without_api_call(self, monkeypatch):
        monkeypatch.setattr(
            "backend.copilot.tools.web_search._chat_config",
            type(
                "C",
                (),
                {
                    "api_key": "sk-test",
                    "base_url": "https://openrouter.ai/api/v1",
                },
            )(),
        )
        openai_stub = AsyncMock()
        with patch(
            "backend.copilot.tools.web_search.AsyncOpenAI",
            return_value=openai_stub,
        ):
            tool = WebSearchTool()
            result = await tool._execute(
                user_id="u1", session=self._session(), query="   "
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_query"
        openai_stub.chat.completions.create.assert_not_called()


class TestToolRegistryIntegration:
    """The tool must be registered under the ``web_search`` name so the
    MCP layer exposes it as ``mcp__copilot__web_search`` — which is
    what the SDK path dispatches to (see
    ``sdk/tool_adapter.py::SDK_DISALLOWED_TOOLS`` which blocks the CLI's
    native ``WebSearch`` in favour of the MCP route)."""

    def test_web_search_is_in_tool_registry(self):
        from backend.copilot.tools import TOOL_REGISTRY

        assert "web_search" in TOOL_REGISTRY
        assert isinstance(TOOL_REGISTRY["web_search"], WebSearchTool)

    def test_sdk_native_websearch_is_disallowed(self):
        from backend.copilot.sdk.tool_adapter import SDK_DISALLOWED_TOOLS

        assert "WebSearch" in SDK_DISALLOWED_TOOLS
