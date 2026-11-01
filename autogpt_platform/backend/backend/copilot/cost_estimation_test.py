"""Tests for CoPilot cost-estimation helpers."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_mock

from backend.copilot import cost_approval, cost_estimation
from backend.copilot.model import ChatMessage, ChatSession


@pytest.mark.asyncio
async def test_threshold_source_is_default_when_the_vendor_does_not_answer(
    mocker: pytest_mock.MockerFixture,
):
    mocker.patch(
        "backend.copilot.cost_estimation.evaluate_feature_flag_value",
        new=AsyncMock(return_value=(10.0, False)),
    )

    threshold, source = await cost_estimation.resolve_cost_confirmation_threshold(
        "user-1"
    )

    assert threshold == cost_estimation.config.copilot_cost_confirmation_threshold_usd
    assert source == "default"


@pytest.mark.asyncio
async def test_threshold_source_is_launchdarkly_when_the_served_value_matches_fallback(
    mocker: pytest_mock.MockerFixture,
):
    fallback = cost_estimation.config.copilot_cost_confirmation_threshold_usd
    mocker.patch(
        "backend.copilot.cost_estimation.evaluate_feature_flag_value",
        new=AsyncMock(return_value=(fallback, True)),
    )

    threshold, source = await cost_estimation.resolve_cost_confirmation_threshold(
        "user-1"
    )

    assert threshold == fallback
    assert source == "launchdarkly"


@pytest.mark.asyncio
async def test_threshold_without_a_user_does_not_read_the_flag(
    mocker: pytest_mock.MockerFixture,
):
    read_flag = mocker.patch(
        "backend.copilot.cost_estimation.evaluate_feature_flag_value",
        new=AsyncMock(),
    )

    threshold, source = await cost_estimation.resolve_cost_confirmation_threshold(None)

    assert source == "default"
    assert threshold == cost_estimation.config.copilot_cost_confirmation_threshold_usd
    read_flag.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("served", [float("nan"), float("inf"), float("-inf")])
async def test_non_finite_served_threshold_uses_the_configured_fallback(
    mocker: pytest_mock.MockerFixture, served: float
):
    mocker.patch(
        "backend.copilot.cost_estimation.evaluate_feature_flag_value",
        new=AsyncMock(return_value=(served, True)),
    )

    threshold, source = await cost_estimation.resolve_cost_confirmation_threshold(
        "user-1"
    )

    assert threshold == cost_estimation.config.copilot_cost_confirmation_threshold_usd
    assert source == "default"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sdk,model_name,threshold,expected_confirmation,expected_cost",
    [
        (False, "claude-sonnet-4-5", 10.0, False, 0.02538),
        (True, "claude-opus-4-1", 0.001, True, 0.2868),
    ],
)
async def test_estimate_prices_a_turn_and_applies_the_confirmation_threshold(
    mocker: pytest_mock.MockerFixture,
    cost_session: ChatSession,
    sdk: bool,
    model_name: str,
    threshold: float,
    expected_confirmation: bool,
    expected_cost: float,
):
    route = _mock_estimate_dependencies(mocker, sdk, model_name, threshold)
    estimate = await cost_estimation.estimate_copilot_turn_cost(
        session_id=cost_session.session_id,
        user_id=cost_session.user_id,
        session=cost_session,
        message="Please analyze this task",
        is_user_message=True,
        context={"url": "https://example.com"},
        file_ids=["file-1"],
        mode="extended_thinking" if sdk else "fast",
        model="advanced" if sdk else "standard",
    )

    assert estimate.resolved_path == ("sdk" if sdk else "baseline")
    assert estimate.resolved_model == model_name
    assert estimate.estimated_llm_calls == (4 if sdk else 3)
    assert estimate.estimated_total_tokens == (
        estimate.estimated_prompt_tokens + estimate.estimated_completion_tokens
    )
    assert estimate.estimated_cost_usd == pytest.approx(expected_cost)
    assert estimate.requires_confirmation is expected_confirmation
    assert estimate.confirmation_threshold_usd == threshold
    assert estimate.threshold_source == "launchdarkly"
    assert "attached files" in estimate.rationale
    route.assert_awaited_once_with(
        "thinking" if sdk else "fast",
        "advanced" if sdk else "standard",
        "user-1",
        config=cost_estimation.config,
    )


@pytest.fixture
def cost_session() -> ChatSession:
    now = datetime.now(UTC)
    return ChatSession(
        session_id="session-1",
        user_id="user-1",
        usage=[],
        started_at=now,
        updated_at=now,
        messages=[ChatMessage(role="user", content="Earlier question")],
    )


def _mock_estimate_dependencies(
    mocker: pytest_mock.MockerFixture,
    sdk: bool,
    model_name: str,
    threshold: float,
) -> AsyncMock:
    mocker.patch.object(cost_estimation.config, "test_mode", False)
    mocker.patch.object(cost_estimation, "estimate_token_count_str", return_value=20)
    mocker.patch.object(cost_estimation, "estimate_token_count", return_value=10)
    mocker.patch.object(
        cost_estimation, "resolve_use_sdk", new=AsyncMock(return_value=sdk)
    )
    route = AsyncMock(return_value=SimpleNamespace(model=model_name))
    mocker.patch.object(cost_estimation, "resolve_model_route", new=route)
    mocker.patch.object(
        cost_estimation,
        "resolve_cost_confirmation_threshold",
        new=AsyncMock(return_value=(threshold, "launchdarkly")),
    )
    return route


def test_longer_turns_and_history_raise_the_call_budget_without_exceeding_cap():
    short = cost_estimation._estimate_llm_calls(
        resolved_path="baseline",
        message_tokens=100,
        history_tokens=0,
        file_count=0,
        has_context=False,
        is_user_message=True,
    )
    long = cost_estimation._estimate_llm_calls(
        resolved_path="sdk",
        message_tokens=2_001,
        history_tokens=20_001,
        file_count=9,
        has_context=True,
        is_user_message=True,
    )
    tool_reply = cost_estimation._estimate_llm_calls(
        resolved_path="sdk",
        message_tokens=2_001,
        history_tokens=20_001,
        file_count=9,
        has_context=True,
        is_user_message=False,
    )

    assert short == 1
    assert short < tool_reply < long <= 12


def _confirmation_quote(*, required: bool) -> cost_estimation.CoPilotTurnCostEstimate:
    return cost_estimation.CoPilotTurnCostEstimate(
        resolved_path="baseline",
        resolved_model="claude-sonnet-4-5",
        estimated_llm_calls=1,
        estimated_prompt_tokens=10,
        estimated_completion_tokens=10,
        estimated_total_tokens=20,
        estimated_cost_usd=2.0 if required else 0.01,
        confirmation_threshold_usd=1.0,
        threshold_source="default",
        requires_confirmation=required,
        rationale="baseline path estimated at 1 model calls",
    )


@pytest.fixture
def _signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        (
            cost_estimation.generate_cost_approval_token.__wrapped__
            if False
            else __import__(
                "backend.copilot.cost_approval", fromlist=["settings"]
            ).settings.secrets
        ),
        "cost_approval_signing_key",
        "test-signing-key",
    )


def _patch_pricing(
    mocker: pytest_mock.MockerFixture,
    session: ChatSession,
    *,
    required: bool,
) -> None:
    mocker.patch.object(
        cost_estimation,
        "get_chat_session",
        new=AsyncMock(return_value=session),
    )
    mocker.patch.object(
        cost_estimation,
        "estimate_copilot_turn_cost",
        new=AsyncMock(return_value=_confirmation_quote(required=required)),
    )


@pytest.mark.asyncio
async def test_a_turn_under_the_threshold_can_be_scheduled(
    mocker: pytest_mock.MockerFixture,
    cost_session: ChatSession,
) -> None:
    _patch_pricing(mocker, cost_session, required=False)

    blocked = await cost_estimation.turn_cost_blocks_scheduling(
        session=cost_session,
        message="Please analyze this task",
        is_user_message=True,
        context=None,
        file_ids=None,
        model="standard",
        approval_token=None,
    )

    assert blocked is None


@pytest.mark.asyncio
async def test_an_over_threshold_turn_is_held_until_a_matching_token_arrives(
    mocker: pytest_mock.MockerFixture,
    cost_session: ChatSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cost_approval.settings.secrets, "cost_approval_signing_key", "test-signing-key"
    )
    _patch_pricing(mocker, cost_session, required=True)

    held = await cost_estimation.turn_cost_blocks_scheduling(
        session=cost_session,
        message="Please analyze this task",
        is_user_message=True,
        context=None,
        file_ids=None,
        model="standard",
        approval_token=None,
    )

    assert held is not None
    assert held.requires_confirmation is True
    assert held.approval_token

    allowed = await cost_estimation.turn_cost_blocks_scheduling(
        session=cost_session,
        message="Please analyze this task",
        is_user_message=True,
        context=None,
        file_ids=None,
        model="standard",
        approval_token=held.approval_token,
    )

    assert allowed is None


@pytest.mark.asyncio
async def test_a_token_for_a_different_message_does_not_schedule_the_turn(
    mocker: pytest_mock.MockerFixture,
    cost_session: ChatSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cost_approval.settings.secrets, "cost_approval_signing_key", "test-signing-key"
    )
    _patch_pricing(mocker, cost_session, required=True)
    held = await cost_estimation.turn_cost_blocks_scheduling(
        session=cost_session,
        message="Please analyze this task",
        is_user_message=True,
        context=None,
        file_ids=None,
        model="standard",
        approval_token=None,
    )
    assert held is not None and held.approval_token

    blocked = await cost_estimation.turn_cost_blocks_scheduling(
        session=cost_session,
        message="A different task",
        is_user_message=True,
        context=None,
        file_ids=None,
        model="standard",
        approval_token=held.approval_token,
    )

    assert blocked is not None
    assert blocked.approval_token != held.approval_token
