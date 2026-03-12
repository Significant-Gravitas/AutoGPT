"""Tests for block credit charging in execute_block()."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockType

from .helpers import execute_block
from .models import BlockOutputResponse, ErrorResponse

_USER = "test-user-helpers"
_SESSION = "test-session-helpers"


def _make_block(block_id: str = "block-1", name: str = "TestBlock"):
    """Create a minimal mock block for execute_block()."""
    mock = MagicMock()
    mock.id = block_id
    mock.name = name
    mock.block_type = BlockType.STANDARD

    mock.input_schema = MagicMock()
    mock.input_schema.get_credentials_fields_info.return_value = {}

    async def _execute(
        input_data: dict, **kwargs: Any
    ) -> AsyncIterator[tuple[str, Any]]:
        yield "result", "ok"

    mock.execute = _execute
    return mock


def _patch_workspace():
    """Patch workspace_db to return a mock workspace."""
    mock_workspace = MagicMock()
    mock_workspace.id = "ws-1"
    mock_ws_db = MagicMock()
    mock_ws_db.get_or_create_workspace = AsyncMock(return_value=mock_workspace)
    return patch("backend.copilot.tools.helpers.workspace_db", return_value=mock_ws_db)


@pytest.mark.asyncio
class TestExecuteBlockCreditCharging:
    async def test_charges_credits_when_cost_is_positive(self):
        """Block with cost > 0 should call spend_credits after execution."""
        block = _make_block()
        mock_credit = AsyncMock()
        mock_credit.get_credits = AsyncMock(return_value=100)
        mock_credit.spend_credits = AsyncMock()

        with (
            _patch_workspace(),
            patch(
                "backend.copilot.tools.helpers.block_usage_cost",
                return_value=(10, {"key": "val"}),
            ),
            patch(
                "backend.copilot.tools.helpers.get_user_credit_model",
                return_value=mock_credit,
            ),
        ):
            result = await execute_block(
                block=block,
                block_id="block-1",
                input_data={"text": "hello"},
                user_id=_USER,
                session_id=_SESSION,
                node_exec_id="exec-1",
                matched_credentials={},
            )

        assert isinstance(result, BlockOutputResponse)
        assert result.success is True
        mock_credit.spend_credits.assert_awaited_once()
        call_kwargs = mock_credit.spend_credits.call_args.kwargs
        assert call_kwargs["cost"] == 10
        assert call_kwargs["metadata"].reason == "copilot_block_execution"

    async def test_returns_error_when_insufficient_credits_before_exec(self):
        """Pre-execution check should return ErrorResponse when balance < cost."""
        block = _make_block()
        mock_credit = AsyncMock()
        mock_credit.get_credits = AsyncMock(return_value=5)  # balance < cost (10)

        with (
            _patch_workspace(),
            patch(
                "backend.copilot.tools.helpers.block_usage_cost",
                return_value=(10, {}),
            ),
            patch(
                "backend.copilot.tools.helpers.get_user_credit_model",
                return_value=mock_credit,
            ),
        ):
            result = await execute_block(
                block=block,
                block_id="block-1",
                input_data={},
                user_id=_USER,
                session_id=_SESSION,
                node_exec_id="exec-1",
                matched_credentials={},
            )

        assert isinstance(result, ErrorResponse)
        assert "Insufficient credits" in result.message

    async def test_no_charge_when_cost_is_zero(self):
        """Block with cost 0 should not call spend_credits."""
        block = _make_block()

        with (
            _patch_workspace(),
            patch(
                "backend.copilot.tools.helpers.block_usage_cost",
                return_value=(0, {}),
            ),
            patch(
                "backend.copilot.tools.helpers.get_user_credit_model",
            ) as mock_get_credit,
        ):
            result = await execute_block(
                block=block,
                block_id="block-1",
                input_data={},
                user_id=_USER,
                session_id=_SESSION,
                node_exec_id="exec-1",
                matched_credentials={},
            )

        assert isinstance(result, BlockOutputResponse)
        assert result.success is True
        # get_user_credit_model should not be called at all for zero-cost blocks
        mock_get_credit.assert_not_awaited()

    async def test_returns_error_on_post_exec_insufficient_balance(self):
        """If charging fails after execution (concurrent spend race), return error."""
        from backend.util.exceptions import InsufficientBalanceError

        block = _make_block()
        mock_credit = AsyncMock()
        mock_credit.get_credits = AsyncMock(return_value=15)  # passes pre-check
        mock_credit.spend_credits = AsyncMock(
            side_effect=InsufficientBalanceError("Low balance", _USER, 5, 10)
        )  # fails during actual charge (race with concurrent spend)

        with (
            _patch_workspace(),
            patch(
                "backend.copilot.tools.helpers.block_usage_cost",
                return_value=(10, {}),
            ),
            patch(
                "backend.copilot.tools.helpers.get_user_credit_model",
                return_value=mock_credit,
            ),
        ):
            result = await execute_block(
                block=block,
                block_id="block-1",
                input_data={},
                user_id=_USER,
                session_id=_SESSION,
                node_exec_id="exec-1",
                matched_credentials={},
            )

        # Post-exec charge failure is treated as fatal (matches executor behavior)
        assert isinstance(result, ErrorResponse)
        assert "Insufficient credits" in result.message
