"""Tests for cost tracking in Exa blocks.

Covers the cost_dollars → provider_cost → merge_stats path for both
ExaContentsBlock and ExaCodeContextBlock.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.data.model import NodeExecutionStats


class TestExaCodeContextCostTracking:
    """ExaCodeContextBlock parses cost_dollars (string) and calls merge_stats."""

    @pytest.mark.asyncio
    async def test_valid_cost_string_is_parsed_and_merged(self):
        """A numeric cost string like '0.005' is merged as provider_cost."""
        from backend.blocks.exa.code_context import ExaCodeContextBlock

        block = ExaCodeContextBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        api_response = {
            "requestId": "req-1",
            "query": "test query",
            "response": "some code",
            "resultsCount": 3,
            "costDollars": "0.005",
            "searchTime": 1.2,
            "outputTokens": 100,
        }

        with patch("backend.blocks.exa.code_context.Requests") as mock_requests_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = api_response
            mock_requests_cls.return_value.post = AsyncMock(return_value=mock_resp)

            outputs = []
            async for key, value in block.run(
                block.Input(query="test query", credentials=TEST_CREDENTIALS_INPUT),  # type: ignore[arg-type]
                credentials=TEST_CREDENTIALS,
            ):
                outputs.append((key, value))

        assert any(k == "cost_dollars" for k, _ in outputs)
        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.005)

    @pytest.mark.asyncio
    async def test_invalid_cost_string_does_not_raise(self):
        """A non-numeric cost_dollars value is swallowed silently."""
        from backend.blocks.exa.code_context import ExaCodeContextBlock

        block = ExaCodeContextBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        api_response = {
            "requestId": "req-2",
            "query": "test",
            "response": "code",
            "resultsCount": 0,
            "costDollars": "N/A",
            "searchTime": 0.5,
            "outputTokens": 0,
        }

        with patch("backend.blocks.exa.code_context.Requests") as mock_requests_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = api_response
            mock_requests_cls.return_value.post = AsyncMock(return_value=mock_resp)

            outputs = []
            async for key, value in block.run(
                block.Input(query="test", credentials=TEST_CREDENTIALS_INPUT),  # type: ignore[arg-type]
                credentials=TEST_CREDENTIALS,
            ):
                outputs.append((key, value))

        # No merge_stats call because float() raised ValueError
        assert len(merged) == 0

    @pytest.mark.asyncio
    async def test_zero_cost_string_is_merged(self):
        """'0.0' is a valid cost — should still be tracked."""
        from backend.blocks.exa.code_context import ExaCodeContextBlock

        block = ExaCodeContextBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        api_response = {
            "requestId": "req-3",
            "query": "free query",
            "response": "result",
            "resultsCount": 1,
            "costDollars": "0.0",
            "searchTime": 0.1,
            "outputTokens": 10,
        }

        with patch("backend.blocks.exa.code_context.Requests") as mock_requests_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = api_response
            mock_requests_cls.return_value.post = AsyncMock(return_value=mock_resp)

            async for _ in block.run(
                block.Input(query="free query", credentials=TEST_CREDENTIALS_INPUT),  # type: ignore[arg-type]
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.0)


class TestExaContentsCostTracking:
    """ExaContentsBlock merges cost_dollars.total as provider_cost."""

    @pytest.mark.asyncio
    async def test_cost_dollars_total_is_merged(self):
        """When the SDK response includes cost_dollars, its total is merged."""
        from backend.blocks.exa.contents import ExaContentsBlock
        from backend.blocks.exa.helpers import CostDollars

        block = ExaContentsBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        mock_sdk_response = MagicMock()
        mock_sdk_response.results = []
        mock_sdk_response.context = None
        mock_sdk_response.statuses = None
        mock_sdk_response.cost_dollars = CostDollars(total=0.012)

        with patch("backend.blocks.exa.contents.AsyncExa") as mock_exa_cls:
            mock_exa = MagicMock()
            mock_exa.get_contents = AsyncMock(return_value=mock_sdk_response)
            mock_exa_cls.return_value = mock_exa

            async for _ in block.run(
                block.Input(
                    urls=["https://example.com"], credentials=TEST_CREDENTIALS_INPUT
                ),  # type: ignore[arg-type]
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.012)

    @pytest.mark.asyncio
    async def test_no_cost_dollars_skips_merge(self):
        """When cost_dollars is absent, merge_stats is not called."""
        from backend.blocks.exa.contents import ExaContentsBlock

        block = ExaContentsBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        mock_sdk_response = MagicMock()
        mock_sdk_response.results = []
        mock_sdk_response.context = None
        mock_sdk_response.statuses = None
        mock_sdk_response.cost_dollars = None

        with patch("backend.blocks.exa.contents.AsyncExa") as mock_exa_cls:
            mock_exa = MagicMock()
            mock_exa.get_contents = AsyncMock(return_value=mock_sdk_response)
            mock_exa_cls.return_value = mock_exa

            async for _ in block.run(
                block.Input(
                    urls=["https://example.com"], credentials=TEST_CREDENTIALS_INPUT
                ),  # type: ignore[arg-type]
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 0

    @pytest.mark.asyncio
    async def test_zero_cost_dollars_is_merged(self):
        """A total of 0.0 (free tier) should still be merged."""
        from backend.blocks.exa.contents import ExaContentsBlock
        from backend.blocks.exa.helpers import CostDollars

        block = ExaContentsBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        mock_sdk_response = MagicMock()
        mock_sdk_response.results = []
        mock_sdk_response.context = None
        mock_sdk_response.statuses = None
        mock_sdk_response.cost_dollars = CostDollars(total=0.0)

        with patch("backend.blocks.exa.contents.AsyncExa") as mock_exa_cls:
            mock_exa = MagicMock()
            mock_exa.get_contents = AsyncMock(return_value=mock_sdk_response)
            mock_exa_cls.return_value = mock_exa

            async for _ in block.run(
                block.Input(
                    urls=["https://example.com"], credentials=TEST_CREDENTIALS_INPUT
                ),  # type: ignore[arg-type]
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.0)


class TestExaSearchCostTracking:
    """ExaSearchBlock merges cost_dollars.total as provider_cost."""

    @pytest.mark.asyncio
    async def test_cost_dollars_total_is_merged(self):
        """When the SDK response includes cost_dollars, its total is merged."""
        from backend.blocks.exa.helpers import CostDollars
        from backend.blocks.exa.search import ExaSearchBlock

        block = ExaSearchBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        mock_sdk_response = MagicMock()
        mock_sdk_response.results = []
        mock_sdk_response.context = None
        mock_sdk_response.resolved_search_type = None
        mock_sdk_response.cost_dollars = CostDollars(total=0.008)

        with patch("backend.blocks.exa.search.AsyncExa") as mock_exa_cls:
            mock_exa = MagicMock()
            mock_exa.search = AsyncMock(return_value=mock_sdk_response)
            mock_exa_cls.return_value = mock_exa

            async for _ in block.run(
                block.Input(query="test query", credentials=TEST_CREDENTIALS_INPUT),  # type: ignore[arg-type]
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.008)

    @pytest.mark.asyncio
    async def test_no_cost_dollars_skips_merge(self):
        """When cost_dollars is absent, merge_stats is not called."""
        from backend.blocks.exa.search import ExaSearchBlock

        block = ExaSearchBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        mock_sdk_response = MagicMock()
        mock_sdk_response.results = []
        mock_sdk_response.context = None
        mock_sdk_response.resolved_search_type = None
        mock_sdk_response.cost_dollars = None

        with patch("backend.blocks.exa.search.AsyncExa") as mock_exa_cls:
            mock_exa = MagicMock()
            mock_exa.search = AsyncMock(return_value=mock_sdk_response)
            mock_exa_cls.return_value = mock_exa

            async for _ in block.run(
                block.Input(query="test query", credentials=TEST_CREDENTIALS_INPUT),  # type: ignore[arg-type]
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 0


class TestExaSimilarCostTracking:
    """ExaFindSimilarBlock merges cost_dollars.total as provider_cost."""

    @pytest.mark.asyncio
    async def test_cost_dollars_total_is_merged(self):
        """When the SDK response includes cost_dollars, its total is merged."""
        from backend.blocks.exa.helpers import CostDollars
        from backend.blocks.exa.similar import ExaFindSimilarBlock

        block = ExaFindSimilarBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        mock_sdk_response = MagicMock()
        mock_sdk_response.results = []
        mock_sdk_response.context = None
        mock_sdk_response.request_id = "req-1"
        mock_sdk_response.cost_dollars = CostDollars(total=0.015)

        with patch("backend.blocks.exa.similar.AsyncExa") as mock_exa_cls:
            mock_exa = MagicMock()
            mock_exa.find_similar = AsyncMock(return_value=mock_sdk_response)
            mock_exa_cls.return_value = mock_exa

            async for _ in block.run(
                block.Input(
                    url="https://example.com", credentials=TEST_CREDENTIALS_INPUT
                ),  # type: ignore[arg-type]
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.015)

    @pytest.mark.asyncio
    async def test_no_cost_dollars_skips_merge(self):
        """When cost_dollars is absent, merge_stats is not called."""
        from backend.blocks.exa.similar import ExaFindSimilarBlock

        block = ExaFindSimilarBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        mock_sdk_response = MagicMock()
        mock_sdk_response.results = []
        mock_sdk_response.context = None
        mock_sdk_response.request_id = "req-2"
        mock_sdk_response.cost_dollars = None

        with patch("backend.blocks.exa.similar.AsyncExa") as mock_exa_cls:
            mock_exa = MagicMock()
            mock_exa.find_similar = AsyncMock(return_value=mock_sdk_response)
            mock_exa_cls.return_value = mock_exa

            async for _ in block.run(
                block.Input(
                    url="https://example.com", credentials=TEST_CREDENTIALS_INPUT
                ),  # type: ignore[arg-type]
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 0


# ---------------------------------------------------------------------------
# ExaCreateResearchBlock — cost_dollars from completed poll response
# ---------------------------------------------------------------------------


COMPLETED_RESEARCH_RESPONSE = {
    "researchId": "test-research-id",
    "status": "completed",
    "model": "exa-research",
    "instructions": "test instructions",
    "createdAt": 1700000000000,
    "finishedAt": 1700000060000,
    "costDollars": {
        "total": 0.05,
        "numSearches": 3,
        "numPages": 10,
        "reasoningTokens": 500,
    },
    "output": {"content": "Research findings...", "parsed": None},
}

PENDING_RESEARCH_RESPONSE = {
    "researchId": "test-research-id",
    "status": "pending",
    "model": "exa-research",
    "instructions": "test instructions",
    "createdAt": 1700000000000,
}


class TestExaCreateResearchBlockCostTracking:
    """ExaCreateResearchBlock merges cost from completed poll response."""

    @pytest.mark.asyncio
    async def test_cost_merged_when_research_completes(self):
        """merge_stats called with provider_cost=total when poll returns completed."""
        from backend.blocks.exa.research import ExaCreateResearchBlock

        block = ExaCreateResearchBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        create_resp = MagicMock()
        create_resp.json.return_value = PENDING_RESEARCH_RESPONSE

        poll_resp = MagicMock()
        poll_resp.json.return_value = COMPLETED_RESEARCH_RESPONSE

        mock_instance = MagicMock()
        mock_instance.post = AsyncMock(return_value=create_resp)
        mock_instance.get = AsyncMock(return_value=poll_resp)

        with (
            patch("backend.blocks.exa.research.Requests", return_value=mock_instance),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            async for _ in block.run(
                block.Input(
                    instructions="test instructions",
                    wait_for_completion=True,
                    credentials=TEST_CREDENTIALS_INPUT,  # type: ignore[arg-type]
                ),
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_no_merge_when_no_cost_dollars(self):
        """When completed response has no costDollars, merge_stats is not called."""
        from backend.blocks.exa.research import ExaCreateResearchBlock

        block = ExaCreateResearchBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        no_cost_response = {**COMPLETED_RESEARCH_RESPONSE, "costDollars": None}
        create_resp = MagicMock()
        create_resp.json.return_value = PENDING_RESEARCH_RESPONSE
        poll_resp = MagicMock()
        poll_resp.json.return_value = no_cost_response

        mock_instance = MagicMock()
        mock_instance.post = AsyncMock(return_value=create_resp)
        mock_instance.get = AsyncMock(return_value=poll_resp)

        with (
            patch("backend.blocks.exa.research.Requests", return_value=mock_instance),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            async for _ in block.run(
                block.Input(
                    instructions="test instructions",
                    wait_for_completion=True,
                    credentials=TEST_CREDENTIALS_INPUT,  # type: ignore[arg-type]
                ),
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert merged == []


# ---------------------------------------------------------------------------
# ExaGetResearchBlock — cost_dollars from single GET response
# ---------------------------------------------------------------------------


class TestExaGetResearchBlockCostTracking:
    """ExaGetResearchBlock merges cost when the fetched research has cost_dollars."""

    @pytest.mark.asyncio
    async def test_cost_merged_from_completed_research(self):
        """merge_stats called with provider_cost=total when research has costDollars."""
        from backend.blocks.exa.research import ExaGetResearchBlock

        block = ExaGetResearchBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        get_resp = MagicMock()
        get_resp.json.return_value = COMPLETED_RESEARCH_RESPONSE

        mock_instance = MagicMock()
        mock_instance.get = AsyncMock(return_value=get_resp)

        with patch("backend.blocks.exa.research.Requests", return_value=mock_instance):
            async for _ in block.run(
                block.Input(
                    research_id="test-research-id",
                    credentials=TEST_CREDENTIALS_INPUT,  # type: ignore[arg-type]
                ),
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_no_merge_when_no_cost_dollars(self):
        """When research has no costDollars, merge_stats is not called."""
        from backend.blocks.exa.research import ExaGetResearchBlock

        block = ExaGetResearchBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        no_cost_response = {**COMPLETED_RESEARCH_RESPONSE, "costDollars": None}
        get_resp = MagicMock()
        get_resp.json.return_value = no_cost_response

        mock_instance = MagicMock()
        mock_instance.get = AsyncMock(return_value=get_resp)

        with patch("backend.blocks.exa.research.Requests", return_value=mock_instance):
            async for _ in block.run(
                block.Input(
                    research_id="test-research-id",
                    credentials=TEST_CREDENTIALS_INPUT,  # type: ignore[arg-type]
                ),
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert merged == []


# ---------------------------------------------------------------------------
# ExaWaitForResearchBlock — cost_dollars from polling response
# ---------------------------------------------------------------------------


class TestExaWaitForResearchBlockCostTracking:
    """ExaWaitForResearchBlock merges cost when the polled research has cost_dollars."""

    @pytest.mark.asyncio
    async def test_cost_merged_when_research_completes(self):
        """merge_stats called with provider_cost=total once polling returns completed."""
        from backend.blocks.exa.research import ExaWaitForResearchBlock

        block = ExaWaitForResearchBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        poll_resp = MagicMock()
        poll_resp.json.return_value = COMPLETED_RESEARCH_RESPONSE

        mock_instance = MagicMock()
        mock_instance.get = AsyncMock(return_value=poll_resp)

        with (
            patch("backend.blocks.exa.research.Requests", return_value=mock_instance),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            async for _ in block.run(
                block.Input(
                    research_id="test-research-id",
                    credentials=TEST_CREDENTIALS_INPUT,  # type: ignore[arg-type]
                ),
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_no_merge_when_no_cost_dollars(self):
        """When completed research has no costDollars, merge_stats is not called."""
        from backend.blocks.exa.research import ExaWaitForResearchBlock

        block = ExaWaitForResearchBlock()
        merged: list[NodeExecutionStats] = []
        block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]

        no_cost_response = {**COMPLETED_RESEARCH_RESPONSE, "costDollars": None}
        poll_resp = MagicMock()
        poll_resp.json.return_value = no_cost_response

        mock_instance = MagicMock()
        mock_instance.get = AsyncMock(return_value=poll_resp)

        with (
            patch("backend.blocks.exa.research.Requests", return_value=mock_instance),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            async for _ in block.run(
                block.Input(
                    research_id="test-research-id",
                    credentials=TEST_CREDENTIALS_INPUT,  # type: ignore[arg-type]
                ),
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert merged == []
