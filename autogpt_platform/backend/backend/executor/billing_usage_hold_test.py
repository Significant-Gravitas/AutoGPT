"""A usage-priced block is billed after it runs, so ``charge_usage`` holds what
it may cost first, through the same checked spend as any priced block."""

import math
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import SecretStr

from backend.blocks._base import Block, BlockCost, BlockCostType
from backend.blocks.baas.bots import BaasBotFetchMeetingDataBlock
from backend.blocks.claude_code import ClaudeCodeBlock
from backend.blocks.code_executor import (
    RESUMED_SANDBOX_SECONDS,
    ExecuteCodeBlock,
    ExecuteCodeStepBlock,
    InstantiateCodeSandboxBlock,
)
from backend.blocks.codex import CodeGenerationBlock, CodexModel
from backend.blocks.elevenlabs._auth import TEST_CREDENTIALS as ELEVENLABS_KEY
from backend.blocks.jina.search import SearchTheWebBlock
from backend.blocks.pinecone import PineconeInsertBlock
from backend.blocks.text_to_speech_block import TEST_CREDENTIALS as UNREAL_KEY
from backend.blocks.text_to_speech_block import UnrealTextToSpeechBlock
from backend.blocks.video.narration import VideoNarrationBlock
from backend.blocks.zerobounce._auth import TEST_CREDENTIALS as ZEROBOUNCE_KEY
from backend.blocks.zerobounce.validate_emails import ValidateEmailsBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.model import APIKeyCredentials, NodeExecutionStats
from backend.executor import utils as executor_utils
from backend.executor.billing import (
    DEFAULT_USAGE_HOLD,
    charge_usage,
    redispatch_reconcile_baseline,
    usage_hold,
)
from backend.integrations.credentials_store import (
    e2b_credentials,
    elevenlabs_credentials,
    openai_credentials,
    unreal_credentials,
    zerobounce_credentials,
)
from backend.util.exceptions import InsufficientBalanceError
from backend.util.type import coerce_inputs_to_schema

_OPENAI = {
    "id": openai_credentials.id,
    "provider": openai_credentials.provider,
    "type": openai_credentials.type,
}
_E2B = {
    "id": e2b_credentials.id,
    "provider": e2b_credentials.provider,
    "type": e2b_credentials.type,
}
_UNREAL = {
    "id": unreal_credentials.id,
    "provider": unreal_credentials.provider,
    "type": unreal_credentials.type,
}
_ELEVENLABS = {
    "id": elevenlabs_credentials.id,
    "provider": elevenlabs_credentials.provider,
    "type": elevenlabs_credentials.type,
}
_ZEROBOUNCE = {
    "id": zerobounce_credentials.id,
    "provider": zerobounce_credentials.provider,
    "type": zerobounce_credentials.type,
}


@pytest.fixture(autouse=True)
def _no_preflight_estimates(monkeypatch):
    monkeypatch.setattr(executor_utils, "get_preflight_estimate", lambda _bid: 0)


@pytest.fixture
def usd_priced_search(monkeypatch):
    def configure(**cost_kwargs):
        monkeypatch.setitem(
            BLOCK_COSTS,
            SearchTheWebBlock,
            [
                BlockCost(
                    cost_amount=150, cost_type=BlockCostType.COST_USD, **cost_kwargs
                )
            ],
        )

    configure()
    return configure


class _Wallet:
    """Spends come out of one balance and fail when it can't cover them."""

    def __init__(self, balance: int) -> None:
        self.balance = balance

    def spend_credits(self, user_id: str, cost: int, metadata) -> int:
        if cost > self.balance:
            raise InsufficientBalanceError(
                user_id=user_id, message="no", balance=self.balance, amount=cost
            )
        self.balance -= cost
        return self.balance


def _node_exec(block_id: str, inputs: dict | None = None) -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id=str(uuid4()),
        graph_id=str(uuid4()),
        graph_version=1,
        node_exec_id=str(uuid4()),
        node_id=str(uuid4()),
        block_id=block_id,
        inputs=inputs or {},
        execution_context=ExecutionContext(),
    )


@pytest.mark.usefixtures("usd_priced_search")
def test_each_hold_is_taken_from_the_balance_and_refused_when_it_cant_be():
    wallet = _Wallet(balance=DEFAULT_USAGE_HOLD + 1)
    charged: list[int | None] = []
    with (
        patch("backend.executor.billing.get_db_client", return_value=wallet),
        patch("backend.executor.billing.get_block", return_value=SearchTheWebBlock()),
    ):
        for _ in range(3):
            try:
                charged.append(charge_usage(_node_exec(SearchTheWebBlock().id), 0)[2])
            except InsufficientBalanceError:
                charged.append(None)

    assert charged == [DEFAULT_USAGE_HOLD, None, None]
    assert wallet.balance == 1


def test_the_hold_is_the_entrys_ceiling_when_inputs_bound_nothing(usd_priced_search):
    usd_priced_search(max_cost=77)

    assert usage_hold(SearchTheWebBlock(), {}) == 77


@pytest.mark.parametrize("max_output_tokens", [2_048, 20_000])
def test_code_generation_holds_the_price_of_its_largest_answer(max_output_tokens):
    prompt = "write a function"
    inputs = {
        "prompt": prompt,
        "model": CodexModel.GPT5_6_SOL,
        "credentials": _OPENAI,
        "max_output_tokens": max_output_tokens,
    }

    hold = usage_hold(CodeGenerationBlock(), inputs)

    usd = (len(prompt) * 5.0 + max_output_tokens * 30.0) / 1_000_000
    assert hold == math.ceil(usd * 150)


def test_code_generation_counts_a_prompt_by_its_bytes():
    # Three bytes a character, and a token can be one byte.
    prompt = "你好" * 50_000
    inputs = {
        "prompt": prompt,
        "model": CodexModel.GPT5_6_SOL,
        "credentials": _OPENAI,
        "max_output_tokens": 1,
    }

    hold = usage_hold(CodeGenerationBlock(), inputs)

    assert hold == math.ceil((len(prompt.encode()) * 5.0 + 30.0) / 1_000_000 * 150)


@pytest.mark.parametrize(
    "block_cls, inputs, hold",
    [
        (UnrealTextToSpeechBlock, {"text": "x" * 1_000, "credentials": _UNREAL}, 3),
        (VideoNarrationBlock, {"script": "x" * 1_000, "credentials": _ELEVENLABS}, 26),
        (
            VideoNarrationBlock,
            {
                "script": "x" * 1_000,
                "model_id": "eleven_flash_v2_5",
                "credentials": _ELEVENLABS,
            },
            13,
        ),
        (ValidateEmailsBlock, {"email": "a@b.co", "credentials": _ZEROBOUNCE}, 2),
        (PineconeInsertBlock, {"chunks": ["c"] * 37, "embeddings": [[0.1]] * 37}, 37),
        (PineconeInsertBlock, {"chunks": ["c"] * 5, "embeddings": [[0.1]] * 3}, 3),
        (BaasBotFetchMeetingDataBlock, {"bot_id": "bot-1"}, 52),
    ],
    ids=[
        "unreal-1k-chars",
        "narration-1k-chars",
        "narration-1k-chars-half-rate",
        "zerobounce",
        "pinecone-37-vectors",
        "pinecone-unpaired-chunks",
        "meeting-fetch",
    ],
)
def test_each_family_holds_its_expected_cost_per_call(block_cls, inputs, hold):
    assert usage_hold(block_cls(), inputs) == hold


@pytest.mark.parametrize("case", ["unreal", "narration", "zerobounce", "pinecone"])
async def test_a_block_priced_by_its_inputs_holds_what_its_run_charges(case):
    block, inputs, charged = await _CHARGED_BY_RUN[case]()

    hold = usage_hold(block, inputs)

    assert hold > 0
    assert (
        hold
        == executor_utils.block_usage_cost(
            block=block, input_data=inputs, stats=charged
        )[0]
    )


def test_a_node_saved_on_a_retired_codex_model_holds_what_its_replacement_holds():
    inputs = {"prompt": "write a function", "credentials": _OPENAI}

    retired = usage_hold(CodeGenerationBlock(), {**inputs, "model": "gpt-5.1-codex"})

    assert retired == usage_hold(
        CodeGenerationBlock(), {**inputs, "model": CodexModel.GPT5_3_CODEX}
    )


@pytest.mark.parametrize("timeout", [60, 600])
def test_a_sandbox_holds_its_timeout(timeout):
    inputs = {"code": "print(1)", "timeout": timeout, "credentials": _E2B}

    # 1 credit per 10 seconds, the sandbox's price.
    assert usage_hold(ExecuteCodeBlock(), inputs) == timeout // 10


@pytest.mark.parametrize(
    "block_cls, inputs, held_seconds",
    [
        (ExecuteCodeBlock, {"timeout": 0}, None),
        (InstantiateCodeSandboxBlock, {"timeout": -5}, None),
        (ExecuteCodeBlock, {"timeout": 86_400}, None),
        (ClaudeCodeBlock, {"timeout": 60, "sandbox_id": "sbx-1"}, None),
        (ClaudeCodeBlock, {"timeout": 60}, 60),
        (ExecuteCodeStepBlock, {"sandbox_id": "sbx-1"}, RESUMED_SANDBOX_SECONDS),
    ],
    ids=[
        "no timeout",
        "negative timeout",
        "past the cap",
        "reused box",
        "own box",
        "step in a reused box",
    ],
)
def test_a_sandbox_is_held_for_as_long_as_its_run_can_last(
    block_cls, inputs, held_seconds
):
    block = block_cls()
    run_on_platform_key = dict(BLOCK_COSTS[block_cls][0].cost_filter or {})
    seconds = held_seconds or block.execution_timeout_seconds

    # 1 credit per 10 seconds; with no usable timeout, the executor's cap ends the run.
    assert usage_hold(block, {**run_on_platform_key, **inputs}) == seconds // 10


@pytest.mark.parametrize(
    "block_cls, inputs",
    [
        (UnrealTextToSpeechBlock, {"text": ["x" * 1_000], "credentials": _UNREAL}),
        (
            VideoNarrationBlock,
            {"script": {"line": "x" * 1_000}, "credentials": _ELEVENLABS},
        ),
        (ExecuteCodeBlock, {"code": "1", "timeout": "600", "credentials": _E2B}),
    ],
    ids=["unreal-text-as-list", "narration-script-as-dict", "sandbox-timeout-as-text"],
)
def test_a_tool_calls_raw_arguments_hold_what_its_coerced_run_charges(
    block_cls, inputs
):
    coerced = dict(inputs)
    coerce_inputs_to_schema(coerced, block_cls().input_schema)

    assert usage_hold(block_cls(), inputs) == usage_hold(block_cls(), coerced)


@pytest.mark.usefixtures("usd_priced_search")
def test_a_resumed_block_is_held_again_and_a_restarted_one_keeps_its_hold():
    wallet = _Wallet(balance=100)
    resumed = _node_exec(SearchTheWebBlock().id)
    resumed.redispatched_from = ExecutionStatus.REVIEW
    restarted = _node_exec(SearchTheWebBlock().id)
    restarted.redispatched_from = ExecutionStatus.RUNNING
    with (
        patch("backend.executor.billing.get_db_client", return_value=wallet),
        patch("backend.executor.billing.get_block", return_value=SearchTheWebBlock()),
    ):
        assert redispatch_reconcile_baseline(resumed) == DEFAULT_USAGE_HOLD
        assert redispatch_reconcile_baseline(restarted) == DEFAULT_USAGE_HOLD

    assert wallet.balance == 100 - DEFAULT_USAGE_HOLD


def test_a_resumed_per_run_block_is_not_charged_again(monkeypatch):
    monkeypatch.setitem(BLOCK_COSTS, SearchTheWebBlock, [BlockCost(cost_amount=5)])
    db = MagicMock()
    resumed = _node_exec(SearchTheWebBlock().id)
    resumed.redispatched_from = ExecutionStatus.REVIEW
    with (
        patch("backend.executor.billing.get_db_client", return_value=db),
        patch("backend.executor.billing.get_block", return_value=SearchTheWebBlock()),
    ):
        assert redispatch_reconcile_baseline(resumed) == 5

    db.spend_credits.assert_not_called()


async def _unreal_charged() -> tuple[Block, dict, NodeExecutionStats]:
    block = UnrealTextToSpeechBlock()
    inputs = {"text": "Hello " * 400, "credentials": _UNREAL}
    with patch.object(
        UnrealTextToSpeechBlock,
        "call_unreal_speech_api",
        new_callable=AsyncMock,
        return_value={"OutputUri": "https://example.com/a.mp3"},
    ):
        async for _ in block.run(block.input_schema(**inputs), credentials=UNREAL_KEY):
            pass
    return block, inputs, block.execution_stats


async def _narration_charged() -> tuple[Block, dict, NodeExecutionStats]:
    block = VideoNarrationBlock()
    inputs = {
        "video_in": "v.mp4",
        "script": "Narrate this. " * 300,
        "model_id": "eleven_turbo_v2_5",
        "credentials": _ELEVENLABS,
    }
    with (
        tempfile.TemporaryDirectory() as run_dir,
        patch(
            "backend.blocks.video.narration.get_exec_file_path",
            side_effect=lambda _run, path: os.path.join(run_dir, path),
        ),
        patch.multiple(
            VideoNarrationBlock,
            _store_input_video=AsyncMock(side_effect=lambda _ctx, file: file),
            _check_input_video=AsyncMock(),
            _generate_narration_audio=MagicMock(return_value=b"mp3"),
            _add_narration_to_video=AsyncMock(),
            _store_output_video=AsyncMock(side_effect=lambda _ctx, file: file),
        ),
    ):
        async for _ in block.run(
            block.input_schema(**inputs),
            credentials=ELEVENLABS_KEY,
            execution_context=ExecutionContext(
                user_id="u", graph_exec_id="run-1", node_exec_id="n"
            ),
            node_exec_id="n",
        ):
            pass
    return block, inputs, block.execution_stats


async def _zerobounce_charged() -> tuple[Block, dict, NodeExecutionStats]:
    block = ValidateEmailsBlock()
    inputs = {"email": "a@b.co", "credentials": _ZEROBOUNCE}
    with patch.object(block, "validate_email", block.test_mock["validate_email"]):
        async for _ in block.run(
            block.input_schema(**inputs), credentials=ZEROBOUNCE_KEY
        ):
            pass
    return block, inputs, block.execution_stats


async def _pinecone_charged() -> tuple[Block, dict, NodeExecutionStats]:
    block = PineconeInsertBlock()
    inputs = {
        "index": "i",
        "chunks": ["one", "two", "three", "four"],
        "embeddings": [[0.1], [0.2], [0.3]],
        "credentials": {"id": "k", "provider": "pinecone", "type": "api_key"},
    }
    key = APIKeyCredentials(
        id="k", provider="pinecone", api_key=SecretStr("k"), title="k"
    )
    with patch("backend.blocks.pinecone.Pinecone", MagicMock()):
        async for _ in block.run(block.input_schema(**inputs), credentials=key):
            pass
    return block, inputs, block.execution_stats


_CHARGED_BY_RUN = {
    "unreal": _unreal_charged,
    "narration": _narration_charged,
    "zerobounce": _zerobounce_charged,
    "pinecone": _pinecone_charged,
}
