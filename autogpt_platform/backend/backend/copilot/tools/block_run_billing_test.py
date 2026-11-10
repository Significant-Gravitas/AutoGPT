"""A block AutoPilot runs is billed what the run cost, as a workflow node is.

Each block here is a real one with its real price rule, run on the platform's
own credential; only its provider is stood in for. The block records what a
real run leaves behind (the provider's cost, a model call cut off mid-request)
and the clock says how long the run took. The wallet takes or refuses each
debit in one step, as the credit store does. The expected charges are worked
out by hand from the price rules, not by the code under test.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.api.features.experts.spend_approval import SpendApprovalNeeded
from backend.blocks._base import AnyBlockSchema
from backend.blocks.ai_condition import AIConditionBlock
from backend.blocks.code_executor import ExecuteCodeBlock, InstantiateCodeSandboxBlock
from backend.blocks.llm import AITextGeneratorBlock, LLMModel, abandoned_call_stats
from backend.copilot.constants import COPILOT_NODE_PREFIX, COPILOT_SESSION_PREFIX
from backend.copilot.tools import helpers
from backend.copilot.tools.helpers import (
    BlockPreparation,
    check_spend_approval,
    execute_block,
)
from backend.copilot.tools.models import (
    BlockOutputResponse,
    ErrorResponse,
    ReviewRequiredResponse,
    ToolResponseBase,
)
from backend.data.credit import UsageTransactionMetadata
from backend.data.model import (
    APIKeyCredentials,
    CredentialsMetaInput,
    NodeExecutionStats,
)
from backend.integrations.credentials_store import (
    e2b_credentials,
    open_router_credentials,
)
from backend.integrations.providers import ProviderName
from backend.util.exceptions import BlockExecutionError, InsufficientBalanceError

from ._test_data import make_session

_USER = "block-run-billing-user"
_SESSION = "block-run-billing-session"
# Billed at 150 credits per dollar of the cost OpenRouter reports; its input
# rate is 375 credits per million tokens.
_MODEL = LLMModel("cohere/command-a-03-2025")


class _Wallet:
    """The user's balance. A debit is checked and taken in one step, and
    refused whole when it would overdraw unless it may (a settlement
    records spend the provider has already billed), as the store does."""

    def __init__(self, balance: int) -> None:
        self.balance = balance
        self.debits: list[int] = []

    @property
    def paid(self) -> int:
        return sum(self.debits)

    async def spend_credits(
        self,
        *,
        user_id: str,
        cost: int,
        metadata: UsageTransactionMetadata,
        fail_insufficient_credits: bool = True,
    ) -> int:
        if fail_insufficient_credits and cost > self.balance:
            raise InsufficientBalanceError(
                "Insufficient balance", user_id, self.balance, cost
            )
        self.balance -= cost
        self.debits.append(cost)
        return self.balance


class _Clock:
    """The clock execute_block times a run with; the provider moves it."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _Provider:
    """Stands in for a block's provider: each call takes ``seconds`` on the
    clock and bills ``cost_usd``, then fails if ``fails``. With
    ``abandons_input_tokens`` it takes the request and never answers, and
    the block records the request's input as abandoned once it is cut off,
    as the LLM blocks do."""

    def __init__(
        self,
        block: AnyBlockSchema,
        *,
        clock: _Clock | None = None,
        seconds: float = 0.0,
        cost_usd: float | None = None,
        fails: bool = False,
        abandons_input_tokens: int = 0,
    ) -> None:
        self.block = block
        self.clock = clock or _Clock()
        self.seconds = seconds
        self.cost_usd = cost_usd
        self.fails = fails
        self.abandons_input_tokens = abandons_input_tokens
        self.calls = 0
        self.request_sent = asyncio.Event()

    def serving(self):
        return patch.object(self.block, "execute", self._execute)

    async def _execute(
        self, input_data: dict, **kwargs: Any
    ) -> AsyncIterator[tuple[str, Any]]:
        self.calls += 1
        self.clock.now += self.seconds
        if self.abandons_input_tokens:
            self.request_sent.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.block.merge_stats(abandoned_call_stats(self.abandons_input_tokens))
                raise
        if self.cost_usd is not None:
            self.block.merge_stats(
                NodeExecutionStats(
                    provider_cost=self.cost_usd, provider_cost_type="cost_usd"
                )
            )
        if self.fails:
            raise BlockExecutionError(
                "The provider failed after billing the call.",
                self.block.name,
                self.block.id,
            )
        yield "response", "done"


def _meta(credentials: APIKeyCredentials) -> CredentialsMetaInput:
    return CredentialsMetaInput(
        id=credentials.id,
        provider=ProviderName(credentials.provider),
        type=credentials.type,
        title=credentials.title,
    )


# Each block with an input it accepts and the platform credential it runs on.
_RUNS: dict[type, tuple[dict[str, Any], APIKeyCredentials]] = {
    AITextGeneratorBlock: ({"prompt": "Hi.", "model": _MODEL}, open_router_credentials),
    AIConditionBlock: (
        {"input_value": "7", "condition": "is prime", "model": _MODEL},
        open_router_credentials,
    ),
    ExecuteCodeBlock: ({"code": "print(1)"}, e2b_credentials),
    InstantiateCodeSandboxBlock: ({"setup_code": ""}, e2b_credentials),
}


async def _run_block(
    block: AnyBlockSchema, wallet: _Wallet, clock: _Clock | None = None
) -> ToolResponseBase:
    """Run ``block`` on its platform credential, as run_block does."""
    input_data, credentials = _RUNS[type(block)]
    creds_manager = MagicMock(get=AsyncMock(return_value=credentials))
    workspace = MagicMock(
        get_or_create_workspace=AsyncMock(return_value=MagicMock(id="ws"))
    )
    users = MagicMock(get_user_by_id=AsyncMock(return_value=MagicMock(timezone="UTC")))
    with (
        patch.object(
            helpers, "IntegrationCredentialsManager", return_value=creds_manager
        ),
        patch.object(helpers, "workspace_db", return_value=workspace),
        patch.object(helpers, "user_db", return_value=users),
        patch.object(helpers, "credit_db", return_value=wallet),
        patch.object(helpers, "perf_counter", clock or _Clock()),
    ):
        return await execute_block(
            block=block,
            block_id=block.id,
            input_data=dict(input_data),
            user_id=_USER,
            session_id=_SESSION,
            node_exec_id=f"{COPILOT_NODE_PREFIX}{block.id}:{uuid4().hex[:8]}",
            matched_credentials={"credentials": _meta(credentials)},
            dry_run=False,
        )


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("cost_usd", "fails", "billed"),
    [
        # $0.075 at 150 credits per dollar is 11.25: 12 credits.
        (0.075, False, 12),
        # A run that fails after the provider billed the call pays the same.
        (0.075, True, 12),
        # $0.01 is 1.5: 2 credits, so a larger debit before the run is
        # partly given back.
        (0.01, False, 2),
    ],
    ids=["succeeds", "fails-after-the-call", "costs-less-than-the-debit"],
)
async def test_a_run_priced_on_provider_cost_is_billed_that_cost(
    cost_usd: float, fails: bool, billed: int
):
    block = AITextGeneratorBlock()
    wallet = _Wallet(balance=100)
    provider = _Provider(block, cost_usd=cost_usd, fails=fails)

    with provider.serving():
        response = await _run_block(block, wallet)

    assert isinstance(response, ErrorResponse if fails else BlockOutputResponse)
    assert provider.calls == 1
    assert wallet.paid == billed
    assert wallet.balance == 100 - billed


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("block_cls", [ExecuteCodeBlock, InstantiateCodeSandboxBlock])
@pytest.mark.parametrize(
    ("seconds", "billed"),
    [
        # One credit per ten seconds started: 25 seconds is 3 credits.
        (25.0, 3),
        # A run shorter than ten seconds still started one.
        (0.5, 1),
    ],
)
async def test_a_run_priced_by_the_second_is_billed_its_wall_time(
    block_cls: type, seconds: float, billed: int
):
    block = block_cls()
    wallet = _Wallet(balance=100)
    clock = _Clock()
    provider = _Provider(block, clock=clock, seconds=seconds)

    with provider.serving():
        response = await _run_block(block, wallet, clock)

    assert isinstance(response, BlockOutputResponse)
    assert wallet.paid == billed


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "block_cls",
    # The first has a pre-run estimate of its cost; the other two have none.
    [AITextGeneratorBlock, AIConditionBlock, InstantiateCodeSandboxBlock],
)
async def test_an_empty_balance_cannot_start_a_block_priced_on_usage(block_cls: type):
    block = block_cls()
    wallet = _Wallet(balance=0)
    provider = _Provider(block, seconds=25.0, cost_usd=0.075)

    with provider.serving():
        response = await _run_block(block, wallet)

    assert isinstance(response, ErrorResponse)
    assert "Insufficient credits" in response.message
    assert provider.calls == 0
    assert wallet.debits == []
    assert wallet.balance == 0


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("ended_by", "billed"),
    [
        # 100,000 input tokens at 375 credits per million is 37.5: 38
        # credits, as a workflow node ended by a stop request is billed.
        ("stop", 38),
        # Cut off by the tool's time cap instead, it pays only the usage the
        # provider reported, as a node the executor times out does.
        ("time-cap", 0),
    ],
)
async def test_a_run_stopped_during_a_model_call_is_billed_the_requests_input(
    ended_by: str, billed: int
):
    block = AITextGeneratorBlock()
    wallet = _Wallet(balance=100)
    provider = _Provider(block, abandons_input_tokens=100_000)

    with (
        provider.serving(),
        patch.object(
            helpers, "MAX_TOOL_WAIT_SECONDS", 0.05 if ended_by == "time-cap" else 600
        ),
    ):
        run = asyncio.ensure_future(_run_block(block, wallet))
        if ended_by == "stop":
            await asyncio.wait_for(provider.request_sent.wait(), 60)
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
        else:
            assert isinstance(await run, ErrorResponse)

    assert provider.calls == 1
    assert wallet.paid == billed


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("block_cls", [AITextGeneratorBlock, AIConditionBlock])
async def test_an_experts_spend_threshold_holds_a_block_priced_on_usage(
    block_cls: type,
):
    """The block's cost is not known until it has run, but it is paid, so an
    expert past her spend threshold waits for the user before running it."""
    block = block_cls()
    input_data, credentials = _RUNS[block_cls]
    prep = BlockPreparation(
        block=block,
        block_id=block.id,
        input_data=dict(input_data),
        matched_credentials={"credentials": _meta(credentials)},
        input_schema={},
        credentials_fields={"credentials"},
        required_non_credential_keys=set(input_data),
        provided_input_keys=set(input_data),
        synthetic_graph_id=f"{COPILOT_SESSION_PREFIX}{_SESSION}",
        synthetic_node_id=f"{COPILOT_NODE_PREFIX}{block.id}",
    )
    approvals = MagicMock(
        spend_approval_required=AsyncMock(
            return_value=SpendApprovalNeeded(
                expert_id="expert-1",
                expert_name="Ada",
                spent=250,
                threshold=250,
                window="week",
            )
        ),
        open_chat_spend_review=AsyncMock(return_value="spend-review-1"),
    )

    with patch.object(helpers, "spend_approval_db", return_value=approvals):
        response = await check_spend_approval(
            prep, _USER, make_session(_USER, expert_id="expert-1")
        )

    assert isinstance(response, ReviewRequiredResponse)
    assert response.review_id == "spend-review-1"
