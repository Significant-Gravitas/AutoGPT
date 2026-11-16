"""Tests for cost tracking in Exa blocks.

Covers the cost_dollars → provider_cost → merge_stats path for the Search API
blocks (contents, search, find similar, answer), ExaCodeContextBlock and the
research blocks.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.exa import agent_api
from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.exa.agent_fixtures import (
    SAMPLE_COMPLETED_RUN,
    SAMPLE_QUEUED_RUN,
    SAMPLE_RUN_ID,
)
from backend.blocks.exa.answers import ExaAnswerBlock
from backend.blocks.exa.contents import ExaContentsBlock
from backend.blocks.exa.research import (
    ExaCreateResearchBlock,
    ExaGetResearchBlock,
    ExaWaitForResearchBlock,
)
from backend.blocks.exa.search import ExaSearchBlock
from backend.blocks.exa.similar import ExaFindSimilarBlock
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

        with patch.object(
            block, "_fetch_context", AsyncMock(return_value=api_response)
        ):

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

        with patch.object(
            block, "_fetch_context", AsyncMock(return_value=api_response)
        ):

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

        with patch.object(
            block, "_fetch_context", AsyncMock(return_value=api_response)
        ):

            async for _ in block.run(
                block.Input(query="free query", credentials=TEST_CREDENTIALS_INPUT),  # type: ignore[arg-type]
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.0)


@pytest.mark.parametrize(
    "block_cls, input_kwargs",
    [
        (ExaContentsBlock, {"urls": ["https://example.com"]}),
        (ExaSearchBlock, {"query": "test query"}),
        (ExaFindSimilarBlock, {"url": "https://example.com"}),
        (ExaAnswerBlock, {"query": "test question"}),
    ],
)
class TestExaSearchApiCostTracking:
    """Search API blocks merge ``costDollars.total`` as provider_cost."""

    @staticmethod
    async def _run(block_cls, input_kwargs, response: dict) -> list:
        block = block_cls()
        merged: list[NodeExecutionStats] = []
        with (
            patch.object(block, "merge_stats", side_effect=merged.append),
            patch.object(block, "call_exa", AsyncMock(return_value=response)),
        ):
            input_data = block_cls.Input(
                credentials=TEST_CREDENTIALS_INPUT, **input_kwargs
            )
            async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
                pass
        return merged

    @pytest.mark.asyncio
    async def test_cost_dollars_total_is_merged(self, block_cls, input_kwargs):
        response = {"results": [], "costDollars": {"total": 0.012}}
        merged = await self._run(block_cls, input_kwargs, response)
        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.012)

    @pytest.mark.asyncio
    async def test_no_cost_dollars_skips_merge(self, block_cls, input_kwargs):
        merged = await self._run(block_cls, input_kwargs, {"results": []})
        assert merged == []

    @pytest.mark.asyncio
    async def test_zero_cost_dollars_is_merged(self, block_cls, input_kwargs):
        response = {"results": [], "costDollars": {"total": 0.0}}
        merged = await self._run(block_cls, input_kwargs, response)
        assert len(merged) == 1
        assert merged[0].provider_cost == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Research blocks — now on Exa Agent: cost comes from the run's costDollars
# ---------------------------------------------------------------------------
#
# The research blocks call /agent/runs through backend.blocks.exa.agent_api.
# A finished run carries costDollars.total (0.1 in SAMPLE_COMPLETED_RUN); a
# queued or running run has not settled, so nothing is billed for it.


class TestExaCreateResearchBlockCostTracking:
    """ExaCreateResearchBlock merges the finished run's costDollars.total."""

    @pytest.mark.asyncio
    async def test_cost_merged_when_research_completes(self):
        block = ExaCreateResearchBlock()
        with (
            patch.object(
                agent_api,
                "create_agent_run",
                AsyncMock(return_value=SAMPLE_QUEUED_RUN),
            ),
            patch.object(
                agent_api,
                "get_agent_run",
                AsyncMock(side_effect=[SAMPLE_QUEUED_RUN, SAMPLE_COMPLETED_RUN]),
            ),
            patch.object(agent_api, "sleep", AsyncMock()),
            patch.object(block, "merge_stats") as merge_stats,
        ):
            async for _ in block.run(
                block.Input(
                    instructions="test instructions",
                    wait_for_completion=True,
                    credentials=TEST_CREDENTIALS_INPUT,
                ),
                credentials=TEST_CREDENTIALS,
                user_id="user-1",
            ):
                pass

        merge_stats.assert_called_once()
        merged: NodeExecutionStats = merge_stats.call_args.args[0]
        assert merged.provider_cost == pytest.approx(
            SAMPLE_COMPLETED_RUN["costDollars"]["total"]
        )
        assert merged.provider_cost_type == "cost_usd"

    @pytest.mark.asyncio
    async def test_no_merge_without_waiting(self):
        """A run that was only queued has not settled; nothing is billed yet."""
        block = ExaCreateResearchBlock()
        with (
            patch.object(
                agent_api,
                "create_agent_run",
                AsyncMock(return_value=SAMPLE_QUEUED_RUN),
            ),
            patch.object(block, "merge_stats") as merge_stats,
        ):
            async for _ in block.run(
                block.Input(
                    instructions="test instructions",
                    wait_for_completion=False,
                    credentials=TEST_CREDENTIALS_INPUT,
                ),
                credentials=TEST_CREDENTIALS,
                user_id="user-1",
            ):
                pass

        merge_stats.assert_not_called()


class TestExaGetResearchBlockCostTracking:
    """ExaGetResearchBlock merges cost once the fetched run has finished."""

    @pytest.mark.asyncio
    async def test_cost_merged_from_completed_research(self):
        block = ExaGetResearchBlock()
        with (
            patch.object(
                agent_api,
                "get_agent_run",
                AsyncMock(return_value=SAMPLE_COMPLETED_RUN),
            ),
            patch.object(block, "merge_stats") as merge_stats,
        ):
            async for _ in block.run(
                block.Input(
                    research_id=SAMPLE_RUN_ID,
                    credentials=TEST_CREDENTIALS_INPUT,
                ),
                credentials=TEST_CREDENTIALS,
                user_id="user-1",
            ):
                pass

        merge_stats.assert_called_once()
        assert merge_stats.call_args.args[0].provider_cost == pytest.approx(
            SAMPLE_COMPLETED_RUN["costDollars"]["total"]
        )

    @pytest.mark.asyncio
    async def test_no_merge_while_research_is_running(self):
        block = ExaGetResearchBlock()
        with (
            patch.object(
                agent_api,
                "get_agent_run",
                AsyncMock(return_value=SAMPLE_QUEUED_RUN),
            ),
            patch.object(block, "merge_stats") as merge_stats,
        ):
            async for _ in block.run(
                block.Input(
                    research_id=SAMPLE_RUN_ID,
                    credentials=TEST_CREDENTIALS_INPUT,
                ),
                credentials=TEST_CREDENTIALS,
                user_id="user-1",
            ):
                pass

        merge_stats.assert_not_called()


class TestExaWaitForResearchBlockCostTracking:
    """ExaWaitForResearchBlock merges cost when the polled run finishes."""

    @pytest.mark.asyncio
    async def test_cost_merged_when_research_completes(self):
        block = ExaWaitForResearchBlock()
        with (
            patch.object(
                agent_api,
                "get_agent_run",
                AsyncMock(side_effect=[SAMPLE_QUEUED_RUN, SAMPLE_COMPLETED_RUN]),
            ),
            patch.object(agent_api, "sleep", AsyncMock()),
            patch.object(block, "merge_stats") as merge_stats,
        ):
            async for _ in block.run(
                block.Input(
                    research_id=SAMPLE_RUN_ID,
                    credentials=TEST_CREDENTIALS_INPUT,
                ),
                credentials=TEST_CREDENTIALS,
                user_id="user-1",
            ):
                pass

        merge_stats.assert_called_once()
        assert merge_stats.call_args.args[0].provider_cost == pytest.approx(
            SAMPLE_COMPLETED_RUN["costDollars"]["total"]
        )

    @pytest.mark.asyncio
    async def test_no_merge_when_polling_times_out(self):
        block = ExaWaitForResearchBlock()
        clock = iter([0.0, 30.0])
        with (
            patch.object(
                agent_api,
                "get_agent_run",
                AsyncMock(return_value=SAMPLE_QUEUED_RUN),
            ),
            patch.object(agent_api, "monotonic", side_effect=lambda: next(clock)),
            patch.object(block, "merge_stats") as merge_stats,
        ):
            async for _ in block.run(
                block.Input(
                    research_id=SAMPLE_RUN_ID,
                    timeout=10,
                    credentials=TEST_CREDENTIALS_INPUT,
                ),
                credentials=TEST_CREDENTIALS,
                user_id="user-1",
            ):
                pass

        merge_stats.assert_not_called()
