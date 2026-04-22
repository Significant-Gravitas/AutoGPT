"""Tests for the ``web_search`` copilot tool.

Covers the result extractor + cost estimator as pure units (fed with
synthetic Anthropic response objects), plus light integration tests that
mock ``AsyncAnthropic.messages.create`` and confirm the handler plumbs
through to ``persist_and_record_usage`` with the right provider tag.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.model import ChatSession

from .models import ErrorResponse, WebSearchResponse, WebSearchResult
from .web_search import (
    _COST_PER_SEARCH_USD,
    WebSearchTool,
    _estimate_cost_usd,
    _extract_results,
)


def _fake_anthropic_response(
    *,
    results: list[dict] | None = None,
    search_requests: int = 1,
    input_tokens: int = 120,
    output_tokens: int = 40,
) -> SimpleNamespace:
    """Build a synthetic Anthropic Messages response.

    Matches the shape produced by ``client.messages.create`` when the
    response includes a ``web_search_tool_result`` content block and
    ``usage.server_tool_use.web_search_requests`` on the turn meter.
    """
    content = []
    if results is not None:
        content.append(
            SimpleNamespace(
                type="web_search_tool_result",
                content=[
                    SimpleNamespace(
                        type="web_search_result",
                        title=r.get("title", "untitled"),
                        url=r.get("url", ""),
                        encrypted_content=r.get("snippet", ""),
                        page_age=r.get("page_age"),
                    )
                    for r in results
                ],
            )
        )
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        server_tool_use=SimpleNamespace(web_search_requests=search_requests),
    )
    return SimpleNamespace(content=content, usage=usage)


class TestExtractResults:
    """The extractor is the only Anthropic-response-shape contact point;
    pin its behaviour so an API shape change surfaces here first."""

    def test_extracts_title_url_page_age_and_drops_encrypted_snippet(self):
        # Anthropic's ``web_search_result`` ships an opaque
        # ``encrypted_content`` blob that is not safe to surface —
        # the extractor must drop it (snippet=="") regardless of
        # whether the blob is non-empty.
        resp = _fake_anthropic_response(
            results=[
                {
                    "title": "Kimi K2.6 launch",
                    "url": "https://example.com/kimi",
                    "snippet": "EiJjbGF1ZGUtZW5jcnlwdGVkLWJsb2I=",
                    "page_age": "1 day",
                },
                {
                    "title": "OpenRouter pricing",
                    "url": "https://openrouter.ai/moonshotai/kimi-k2.6",
                    "snippet": "",
                },
            ]
        )
        out, requests = _extract_results(resp, limit=10)
        assert requests == 1
        assert len(out) == 2
        assert out[0].title == "Kimi K2.6 launch"
        assert out[0].url == "https://example.com/kimi"
        assert out[0].snippet == ""
        assert out[0].page_age == "1 day"
        assert out[1].snippet == ""

    def test_limit_caps_returned_results(self):
        resp = _fake_anthropic_response(
            results=[{"title": f"r{i}", "url": f"https://e/{i}"} for i in range(10)]
        )
        out, _ = _extract_results(resp, limit=3)
        assert len(out) == 3
        assert [r.title for r in out] == ["r0", "r1", "r2"]

    def test_missing_content_returns_empty(self):
        resp = SimpleNamespace(content=[], usage=None)
        out, requests = _extract_results(resp, limit=10)
        assert out == []
        assert requests == 0

    def test_non_search_blocks_are_ignored(self):
        resp = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Here's what I found..."),
                SimpleNamespace(
                    type="web_search_tool_result",
                    content=[
                        SimpleNamespace(
                            type="web_search_result",
                            title="real",
                            url="https://real.example",
                            encrypted_content="body",
                            page_age=None,
                        )
                    ],
                ),
            ],
            usage=None,
        )
        out, _ = _extract_results(resp, limit=10)
        assert len(out) == 1 and out[0].title == "real"


class TestEstimateCostUsd:
    """Pin the per-search fee + Haiku inference math — the pricing
    constants in ``web_search.py`` are hard-coded (no live lookup) so a
    drift between Anthropic's schedule and our constants must surface
    in this test for the next reader to notice."""

    def test_zero_searches_still_charges_inference(self):
        resp = _fake_anthropic_response(results=[], search_requests=0)
        cost = _estimate_cost_usd(resp, search_requests=0)
        # Haiku at 1000 input / 5000 output tokens = tiny but non-zero.
        assert 0 < cost < 0.001

    def test_single_search_fee_dominates(self):
        resp = _fake_anthropic_response(
            results=[{"title": "x", "url": "https://e"}],
            search_requests=1,
            input_tokens=100,
            output_tokens=20,
        )
        cost = _estimate_cost_usd(resp, search_requests=1)
        # ~$0.010 search + trivial inference — total still ~1 cent.
        assert cost >= _COST_PER_SEARCH_USD
        assert cost < _COST_PER_SEARCH_USD + 0.001

    def test_three_searches_linear_in_count(self):
        resp = _fake_anthropic_response(
            results=[], search_requests=3, input_tokens=0, output_tokens=0
        )
        cost = _estimate_cost_usd(resp, search_requests=3)
        assert cost == pytest.approx(3 * _COST_PER_SEARCH_USD)


class TestWebSearchToolDispatch:
    """Lightweight integration test: mock the Anthropic client, confirm
    the handler returns a ``WebSearchResponse`` and the usage tracker is
    called with ``provider='anthropic'`` (not 'open_router', even on the
    baseline path — server-side web_search bills Anthropic regardless of
    the calling LLM's route)."""

    def _session(self) -> ChatSession:
        s = ChatSession.new("test-user", dry_run=False)
        s.session_id = "sess-1"
        return s

    @pytest.mark.asyncio
    async def test_returns_response_with_results_and_tracks_cost(self, monkeypatch):
        fake_resp = _fake_anthropic_response(
            results=[
                {
                    "title": "hello",
                    "url": "https://example.com",
                    "snippet": "greeting",
                }
            ],
            search_requests=1,
        )
        mock_client = type(
            "MC",
            (),
            {
                "messages": type(
                    "M", (), {"create": AsyncMock(return_value=fake_resp)}
                )()
            },
        )()

        # Stub the Anthropic API key so ``is_available`` is True.
        monkeypatch.setattr(
            "backend.copilot.tools.web_search.Settings",
            lambda: SimpleNamespace(
                secrets=SimpleNamespace(anthropic_api_key="sk-test")
            ),
        )

        with (
            patch(
                "backend.copilot.tools.web_search.AsyncAnthropic",
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
            )

        assert isinstance(result, WebSearchResponse)
        assert result.query == "kimi k2.6 launch"
        assert len(result.results) == 1
        assert isinstance(result.results[0], WebSearchResult)
        assert result.search_requests == 1

        # Cost tracker must have been called with provider="anthropic".
        assert mock_track.await_count == 1
        kwargs = mock_track.await_args.kwargs
        assert kwargs["provider"] == "anthropic"
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["user_id"] == "u1"
        assert kwargs["cost_usd"] >= _COST_PER_SEARCH_USD

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error_without_calling_anthropic(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "backend.copilot.tools.web_search.Settings",
            lambda: SimpleNamespace(secrets=SimpleNamespace(anthropic_api_key="")),
        )
        anthropic_stub = AsyncMock()
        with (
            patch(
                "backend.copilot.tools.web_search.AsyncAnthropic",
                return_value=anthropic_stub,
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
        anthropic_stub.messages.create.assert_not_called()
        mock_track.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_query_rejected_without_api_call(self, monkeypatch):
        monkeypatch.setattr(
            "backend.copilot.tools.web_search.Settings",
            lambda: SimpleNamespace(
                secrets=SimpleNamespace(anthropic_api_key="sk-test")
            ),
        )
        anthropic_stub = AsyncMock()
        with patch(
            "backend.copilot.tools.web_search.AsyncAnthropic",
            return_value=anthropic_stub,
        ):
            tool = WebSearchTool()
            result = await tool._execute(
                user_id="u1", session=self._session(), query="   "
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_query"
        anthropic_stub.messages.create.assert_not_called()


class TestToolRegistryIntegration:
    """The tool must be registered under the ``web_search`` name so the
    MCP layer exposes it as ``mcp__copilot__web_search`` — which is
    what the SDK path now dispatches to (see
    ``sdk/tool_adapter.py::SDK_DISALLOWED_TOOLS`` which blocks the CLI's
    native ``WebSearch`` in favour of the MCP route)."""

    def test_web_search_is_in_tool_registry(self):
        from backend.copilot.tools import TOOL_REGISTRY

        assert "web_search" in TOOL_REGISTRY
        assert isinstance(TOOL_REGISTRY["web_search"], WebSearchTool)

    def test_sdk_native_websearch_is_disallowed(self):
        from backend.copilot.sdk.tool_adapter import SDK_DISALLOWED_TOOLS

        assert "WebSearch" in SDK_DISALLOWED_TOOLS
