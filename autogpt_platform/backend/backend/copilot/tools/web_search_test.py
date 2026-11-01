"""Tests for the ``web_search`` copilot tool.

Covers the annotation extractor + cost extractor as pure units (fed
with real ``openai`` SDK types — no duck-typed ``SimpleNamespace``
stand-ins), plus integration tests exercising both the quick
(``perplexity/sonar``) and deep (``perplexity/sonar-deep-research``)
paths — mocking ``AsyncOpenAI.chat.completions.create`` and confirming
the handler plumbs through to ``persist_and_record_usage`` with
``provider='open_router'`` and the real ``usage.cost`` value.
"""

import json
import re
from pathlib import Path
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

from .models import ErrorResponse, WebSearchResponse, WebSearchResult
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


def _citation(*, url: str, title: str) -> Annotation:
    """Typed ``Annotation`` for a URL citation, shaped like OpenRouter's
    Sonar annotations (indices always 0, no snippet)."""
    url_citation = AnnotationURLCitation(
        url=url, title=title, start_index=0, end_index=0
    )
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
        _citation(url=c.get("url", ""), title=c.get("title", "untitled"))
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

    def test_extracts_number_title_and_url(self):
        resp = _fake_response(
            citations=[
                {"title": "Kimi K2.6 launch", "url": "https://example.com/kimi"},
                {
                    "title": "OpenRouter pricing",
                    "url": "https://openrouter.ai/moonshotai/kimi-k2.6",
                },
            ],
            answer="K2.6 launched on 2026-04-20.[1] Pricing is listed.[2]",
        )
        out = _extract_results(resp, limit=10)
        assert [r.model_dump() for r in out] == [
            {"n": 1, "title": "Kimi K2.6 launch", "url": "https://example.com/kimi"},
            {
                "n": 2,
                "title": "OpenRouter pricing",
                "url": "https://openrouter.ai/moonshotai/kimi-k2.6",
            },
        ]

    def test_answer_without_markers_returns_first_limit_sources(self):
        resp = _fake_response(
            citations=[{"title": f"r{i}", "url": f"https://e/{i}"} for i in range(10)]
        )
        out = _extract_results(resp, limit=3)
        assert [(r.n, r.title) for r in out] == [(1, "r0"), (2, "r1"), (3, "r2")]

    def test_cited_sources_with_gaps_are_padded_to_limit(self):
        resp = _fake_response(
            citations=[
                {"title": f"r{i}", "url": f"https://e/{i}"} for i in range(1, 25)
            ],
            answer="The rate is 3.75%.[10][20]",
        )
        out = _extract_results(resp, limit=5)
        assert [r.n for r in out] == [1, 2, 3, 10, 20]
        assert {r.n: r.url for r in out}[20] == "https://e/20"

    def test_cited_sources_are_not_cut_to_limit(self):
        resp = _fake_response(
            citations=[
                {"title": f"r{i}", "url": f"https://e/{i}"} for i in range(1, 25)
            ],
            answer="".join(f"Fact.[{n}] " for n in (2, 4, 7, 9, 13, 15, 21)),
        )
        out = _extract_results(resp, limit=1)
        assert [r.n for r in out] == [2, 4, 7, 9, 13, 15, 21]

    def test_markers_with_no_annotation_are_dropped(self):
        resp = _fake_response(
            citations=[
                {"title": f"r{i}", "url": f"https://e/{i}"} for i in range(1, 4)
            ],
            answer="Known.[2] Unknown.[9] Not a source.[0]",
        )
        out = _extract_results(resp, limit=1)
        assert [r.n for r in out] == [2]

    def test_duplicate_urls_keep_each_cited_number(self):
        resp = _fake_response(
            citations=[
                {"title": "Page", "url": "https://e/same"},
                {"title": "Other", "url": "https://e/other"},
                {"title": "Page again", "url": "https://e/same"},
            ],
            answer="One.[1] Two.[3]",
        )
        out = _extract_results(resp, limit=1)
        assert [(r.n, r.url) for r in out] == [
            (1, "https://e/same"),
            (3, "https://e/same"),
        ]

    def test_repeated_markers_return_one_entry_per_number(self):
        resp = _fake_response(
            citations=[
                {"title": f"r{i}", "url": f"https://e/{i}"} for i in range(1, 4)
            ],
            answer="A.[2] B.[2][2] C.[3]",
        )
        assert [r.n for r in _extract_results(resp, limit=1)] == [2, 3]

    def test_more_than_twenty_cited_sources_are_capped_at_twenty(self):
        resp = _fake_response(
            citations=[
                {"title": f"r{i}", "url": f"https://e/{i}"} for i in range(1, 31)
            ],
            answer="".join(f"Fact.[{n}] " for n in range(1, 31)),
        )
        out = _extract_results(resp, limit=5)
        assert [r.n for r in out] == list(range(1, 21))

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


_SONAR_FIXTURES = json.loads(
    (Path(__file__).parent / "testdata" / "web_search_sonar_responses.json").read_text()
)


def _recorded_response(name: str) -> ChatCompletion:
    """A real ``perplexity/sonar`` response recorded through OpenRouter."""
    return ChatCompletion.model_validate(_SONAR_FIXTURES[name])


def _cited_numbers(answer: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


class TestCitedSources:
    """Sonar's answer cites its sources as ``[n]`` markers that index
    the annotation list (annotation i is ``[i+1]``).  Every number the
    answer cites must come back with that number, or the caller cannot
    say where a claim came from."""

    @pytest.mark.asyncio
    async def test_max_results_pads_the_cited_sources_up_to_it(self, monkeypatch):
        sources = [
            {"title": f"Source {i}", "url": f"https://news.example.com/story-{i}"}
            for i in range(1, 13)
        ]
        dispatch = TestWebSearchToolDispatch()
        mock_client = dispatch._mock_client(
            _fake_response(citations=sources, answer="A.[11] B.[12]")
        )
        monkeypatch.setattr(
            "backend.copilot.tools.web_search._chat_config",
            type(
                "C",
                (),
                {"api_key": "sk-test", "base_url": "https://openrouter.ai/api/v1"},
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
            ),
        ):
            result = await WebSearchTool()._execute(
                user_id="u1",
                session=dispatch._session(),
                query="twelve sources",
                max_results=4,
            )

        assert isinstance(result, WebSearchResponse)
        assert [r.n for r in result.results] == [1, 2, 11, 12]

    @pytest.mark.parametrize("name", ["jwst_news", "central_bank_rates"])
    def test_recorded_sonar_response_every_cited_number_resolves(self, name):
        resp = _recorded_response(name)
        annotations = resp.choices[0].message.annotations or []
        cited = _cited_numbers(_extract_answer(resp))
        assert cited, "fixture answer should cite sources"

        by_n = {r.n: r for r in _extract_results(resp, limit=5)}
        assert cited <= set(by_n)
        for n in cited:
            assert by_n[n].url == annotations[n - 1].url_citation.url
            assert by_n[n].title == annotations[n - 1].url_citation.title

    def test_description_and_result_promise_no_snippet(self):
        assert "snippet" not in WebSearchTool().description
        assert "snippet" not in WebSearchResult.model_fields


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
            citations=[{"title": "hello", "url": "https://example.com"}],
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
        assert result.results[0].model_dump() == {
            "n": 1,
            "title": "hello",
            "url": "https://example.com",
        }

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
            citations=[{"title": "deep find", "url": "https://example.com/deep"}],
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
