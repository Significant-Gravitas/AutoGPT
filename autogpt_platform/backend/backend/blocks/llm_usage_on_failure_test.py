"""A failed or stopped AI block keeps the usage its model calls incurred.

The node is billed on its block's stats, so usage a call incurred before it
failed, and the estimated input of a request a stop cut off, are recorded
there.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from backend.blocks import llm
from backend.data.model import NodeExecutionStats

CALL_TOKENS = 100


def _structured_run(*, fail_on_call: int):
    """Stands in for the structured block's run: each call records its usage,
    then answers, until call number ``fail_on_call``, which fails after
    recording it, as a call whose reply did not parse does."""
    calls = 0

    async def run(self: llm.AIStructuredResponseGeneratorBlock, input_data, **_):
        nonlocal calls
        calls += 1
        self.merge_stats(NodeExecutionStats(input_token_count=CALL_TOKENS))
        if calls == fail_on_call:
            raise RuntimeError("The reply did not match the expected format.")
        yield "response", {
            "response": "text",
            "summary": "a summary",
            "final_summary": "the final summary",
            "list": ["a", "b"],
        }
        yield "prompt", []

    return run


def _structured_input(**overrides: Any) -> llm.AIStructuredResponseGeneratorBlock.Input:
    fields: dict[str, Any] = {
        "prompt": "Hello",
        "expected_format": {},
        "model": llm.DEFAULT_LLM_MODEL,
        "credentials": llm.TEST_CREDENTIALS_INPUT,
    }
    return llm.AIStructuredResponseGeneratorBlock.Input(**(fields | overrides))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block",
    [
        llm.AITextGeneratorBlock(),
        llm.AITextSummarizerBlock(),
        llm.AIConversationBlock(),
        llm.AIListGeneratorBlock(),
    ],
    ids=lambda block: block.name,
)
async def test_a_failed_call_keeps_the_usage_it_incurred(block: llm.AIBlockBase):
    with (
        patch.object(
            llm.AIStructuredResponseGeneratorBlock,
            "run",
            _structured_run(fail_on_call=1),
        ),
        pytest.raises(RuntimeError),
    ):
        await block.llm_call(_structured_input(), credentials=llm.TEST_CREDENTIALS)

    assert block.execution_stats.input_token_count == CALL_TOKENS


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on_call", [0, 3], ids=["succeeds", "fails"])
async def test_a_nested_summary_is_billed_to_the_summarizer(fail_on_call: int):
    """Summaries too long to combine are summarized again by a nested
    summarizer, whose calls are the outer block's usage too, whether or not
    they succeed."""
    block = llm.AITextSummarizerBlock()
    input_data = llm.AITextSummarizerBlock.Input(
        text="",
        model=llm.DEFAULT_LLM_MODEL,
        credentials=llm.TEST_CREDENTIALS_INPUT,
        max_tokens=5,
        chunk_overlap=0,
    )
    summaries = ["one two three", "four five six"]  # 6 words over a budget of 5

    with patch.object(
        llm.AIStructuredResponseGeneratorBlock,
        "run",
        _structured_run(fail_on_call=fail_on_call),
    ):
        if fail_on_call:
            with pytest.raises(RuntimeError):
                await block._combine_summaries(
                    summaries, input_data, credentials=llm.TEST_CREDENTIALS
                )
        else:
            await block._combine_summaries(
                summaries, input_data, credentials=llm.TEST_CREDENTIALS
            )

    # The nested summarizer splits the 6 words into two chunks and combines
    # their summaries: three calls, all billed here.
    assert block.execution_stats.input_token_count == 3 * CALL_TOKENS


async def _hanging_provider(**_: Any) -> None:
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_call_cancelled_mid_request_records_its_estimated_input():
    """A stop cancels the request after it went out: no reply reports its
    usage, so its estimated input tokens are recorded, apart from the
    reported usage."""
    block = llm.AIStructuredResponseGeneratorBlock()
    request_sent = asyncio.Event()

    async def provider(**kwargs: Any) -> None:
        request_sent.set()
        await _hanging_provider(**kwargs)

    with patch("backend.util.llm.providers.call_provider", provider):
        run = asyncio.ensure_future(
            block.run_once(
                _structured_input(prompt="Summarize this text. " * 50),
                "response",
                credentials=llm.TEST_CREDENTIALS,
            )
        )
        await asyncio.wait_for(request_sent.wait(), timeout=30)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    assert block.execution_stats.abandoned_input_token_count > 100
    assert block.execution_stats.input_token_count == 0


@pytest.mark.asyncio
async def test_a_wrapper_block_keeps_the_estimate_of_a_cancelled_call():
    block = llm.AITextGeneratorBlock()
    request_sent = asyncio.Event()

    async def provider(**kwargs: Any) -> None:
        request_sent.set()
        await _hanging_provider(**kwargs)

    with patch("backend.util.llm.providers.call_provider", provider):
        run = asyncio.ensure_future(
            block.llm_call(
                _structured_input(prompt="Write a long story. " * 50),
                credentials=llm.TEST_CREDENTIALS,
            )
        )
        await asyncio.wait_for(request_sent.wait(), timeout=30)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    assert block.execution_stats.abandoned_input_token_count > 100


@pytest.mark.asyncio
async def test_a_call_cancelled_before_the_request_records_no_estimate():
    """Cancelled while the prompt is still being prepared, the request never
    reached the provider, so no estimate is recorded for it."""
    block = llm.AIStructuredResponseGeneratorBlock()
    preparing = asyncio.Event()

    async def compress(**_: Any) -> None:
        preparing.set()
        await asyncio.Event().wait()

    provider = AsyncMock()
    with (
        patch.object(llm, "compress_context", compress),
        patch("backend.util.llm.providers.call_provider", provider),
    ):
        run = asyncio.ensure_future(
            block.run_once(
                _structured_input(), "response", credentials=llm.TEST_CREDENTIALS
            )
        )
        await asyncio.wait_for(preparing.wait(), timeout=30)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    provider.assert_not_awaited()
    assert block.execution_stats.abandoned_input_token_count == 0
    assert block.execution_stats.input_token_count == 0


@pytest.mark.asyncio
async def test_a_timed_out_call_records_only_reported_usage():
    """A timeout carries no estimate: the node's usage is what the provider
    reported."""
    block = llm.AIStructuredResponseGeneratorBlock()
    timeout = openai.APITimeoutError(
        request=httpx.Request("POST", "https://provider.invalid")
    )

    with (
        patch(
            "backend.util.llm.providers.call_provider",
            AsyncMock(side_effect=timeout),
        ),
        pytest.raises(RuntimeError),
    ):
        await block.run_once(
            _structured_input(), "response", credentials=llm.TEST_CREDENTIALS
        )

    assert block.execution_stats.abandoned_input_token_count == 0
    assert block.execution_stats.input_token_count == 0
