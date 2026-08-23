from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest
from prisma.enums import ReviewStatus

from backend.blocks.taskmarket.blocks import CreateTaskMarketTaskBlock
from backend.blocks.taskmarket.cli import SettlementUnknownError, TaskMarketCLI
from backend.blocks.taskmarket.models import (
    BASE_USDC_ADDRESS,
    TaskMarketTaskPreview,
)
from backend.blocks.taskmarket.review import consume_approved_review
from backend.data.human_review import ReviewResult
from backend.util.exceptions import BlockExecutionError


def test_preview_binds_exact_spend_and_deliverables():
    preview = TaskMarketTaskPreview.build(
        description="Implement an accessibility audit",
        deliverables=["report.md", "screenshots.zip"],
        reward_usdc=Decimal("2.5"),
        maximum_spend_usdc=Decimal(3),
        deadline=datetime.now(timezone.utc) + timedelta(hours=4),
        mode="bounty",
        tags=["audit", "accessibility"],
    )

    assert preview.network.chain_id == 8453
    assert preview.reward_usdc == Decimal("2.500000")
    assert preview.maximum_spend_usdc == Decimal("3.000000")
    assert preview.deliverables == ["report.md", "screenshots.zip"]
    assert preview.fingerprint == preview.calculate_fingerprint()


def test_preview_rejects_reward_above_maximum_spend():
    with pytest.raises(ValueError, match="maximum spend"):
        TaskMarketTaskPreview.build(
            description="Write a report",
            deliverables=["report.md"],
            reward_usdc=Decimal(4),
            maximum_spend_usdc=Decimal(3),
            deadline=datetime.now(timezone.utc) + timedelta(hours=4),
            mode="bounty",
            tags=[],
        )


@pytest.mark.asyncio
async def test_create_preflight_enforces_base_usdc_and_balance():
    runner = AsyncMock(
        side_effect=[
            {
                "address": "0x38fA0E8373649d90c34E7D4228254B4795a5a8b5",
                "chainId": 8453,
                "currency": "USDC",
                "usdcContract": BASE_USDC_ADDRESS,
            },
            {"accepted": True, "enforcementEnabled": True},
            {"balanceUsdc": "5.0", "balanceBaseUnits": "5000000"},
        ]
    )
    cli = TaskMarketCLI(command_runner=runner)

    preflight = await cli.preflight(Decimal(3))

    assert preflight.chain_id == 8453
    assert preflight.balance_usdc == Decimal("5.000000")
    assert runner.await_count == 3


@pytest.mark.asyncio
async def test_create_timeout_is_unknown_and_never_retried():
    runner = AsyncMock(side_effect=TimeoutError("relayer timed out"))
    cli = TaskMarketCLI(command_runner=runner)

    with pytest.raises(SettlementUnknownError, match="must not be retried"):
        await cli.create_task(
            description="Write a report\n\nDeliverables:\n- report.md",
            reward_usdc=Decimal(2),
            duration_hours=Decimal(4),
            mode="bounty",
            tags=["report"],
            idempotency_key="a" * 64,
        )

    assert runner.await_count == 1


@pytest.mark.asyncio
async def test_create_requires_exact_fresh_review_before_cli_call():
    preview = TaskMarketTaskPreview.build(
        description="Write a report",
        deliverables=["report.md"],
        reward_usdc=Decimal(2),
        maximum_spend_usdc=Decimal(2),
        deadline=datetime.now(timezone.utc) + timedelta(hours=4),
        mode="bounty",
        tags=["report"],
    )
    block = CreateTaskMarketTaskBlock()
    block.request_funding_approval = AsyncMock(return_value=False)
    block.create_task = AsyncMock()
    input_data = block.Input(preview=preview)

    with pytest.raises(BlockExecutionError, match="rejected"):
        _ = [
            item
            async for item in block.run(
                input_data,
                user_id="user",
                node_id="node",
                node_exec_id="node-exec",
                graph_exec_id="graph-exec",
                graph_id="graph",
                graph_version=1,
                execution_context=type(
                    "Context", (), {"organization_id": None, "team_id": None}
                )(),
            )
        ]

    block.create_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_rejects_review_bound_to_different_execution():
    preview = TaskMarketTaskPreview.build(
        description="Write a report",
        deliverables=["report.md"],
        reward_usdc=Decimal(2),
        maximum_spend_usdc=Decimal(2),
        deadline=datetime.now(timezone.utc) + timedelta(hours=4),
        mode="bounty",
        tags=[],
    )
    block = CreateTaskMarketTaskBlock()
    wrong_review = ReviewResult(
        data=preview.model_dump(mode="json"),
        status=ReviewStatus.APPROVED,
        message="approved",
        processed=False,
        node_exec_id="old-execution",
    )

    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        pytest.raises(BlockExecutionError, match="exact execution"),
    ):
        get_review = AsyncMock(return_value=wrong_review)
        check_approval = AsyncMock(return_value=None)
        monkeypatch.setattr(
            "backend.blocks.taskmarket.blocks.HITLReviewHelper.check_approval",
            check_approval,
        )
        monkeypatch.setattr(
            "backend.blocks.taskmarket.blocks.HITLReviewHelper.get_or_create_human_review",
            get_review,
        )
        await block.request_funding_approval(
            preview=preview,
            user_id="user",
            node_id="node",
            node_exec_id="node-exec",
            graph_exec_id="graph-exec",
            graph_id="graph",
            graph_version=1,
            execution_context=type(
                "Context", (), {"organization_id": None, "team_id": None}
            )(),
        )


def test_preview_rejects_comma_delimited_tag():
    with pytest.raises(ValueError, match="commas"):
        TaskMarketTaskPreview.build(
            description="Write a report",
            deliverables=["report.md"],
            reward_usdc=Decimal(2),
            maximum_spend_usdc=Decimal(2),
            deadline=datetime.now(timezone.utc) + timedelta(hours=4),
            mode="bounty",
            tags=["audit,urgent"],
        )


@pytest.mark.asyncio
async def test_approved_review_must_be_claimed_atomically():
    preview = TaskMarketTaskPreview.build(
        description="Write a report",
        deliverables=["report.md"],
        reward_usdc=Decimal(2),
        maximum_spend_usdc=Decimal(2),
        deadline=datetime.now(timezone.utc) + timedelta(hours=4),
        mode="bounty",
        tags=[],
    )
    review = ReviewResult(
        data=preview.model_dump(mode="json"),
        status=ReviewStatus.APPROVED,
        message="approved",
        processed=False,
        node_exec_id="node-exec",
    )
    block = CreateTaskMarketTaskBlock()

    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        pytest.raises(BlockExecutionError, match="already consumed"),
    ):
        monkeypatch.setattr(
            "backend.blocks.taskmarket.blocks.HITLReviewHelper.check_approval",
            AsyncMock(return_value=review),
        )
        monkeypatch.setattr(
            "backend.blocks.taskmarket.blocks.consume_approved_review",
            AsyncMock(return_value=False),
        )
        await block.request_funding_approval(
            preview=preview,
            user_id="user",
            node_id="node",
            node_exec_id="node-exec",
            graph_exec_id="graph-exec",
            graph_id="graph",
            graph_version=1,
            execution_context=type(
                "Context", (), {"organization_id": None, "team_id": None}
            )(),
        )


@pytest.mark.asyncio
async def test_consume_approved_review_uses_conditional_update():
    model = Mock()
    model.update_many = AsyncMock(return_value=1)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "backend.blocks.taskmarket.review.PendingHumanReview.prisma",
            Mock(return_value=model),
        )
        claimed = await consume_approved_review("node-exec", "user")

    assert claimed is True
    model.update_many.assert_awaited_once_with(
        where={
            "nodeExecId": "node-exec",
            "userId": "user",
            "status": ReviewStatus.APPROVED,
            "processed": False,
        },
        data={"processed": True},
    )
