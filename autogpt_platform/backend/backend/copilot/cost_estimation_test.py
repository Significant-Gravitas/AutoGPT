"""Tests for CoPilot cost-estimation helpers."""

from unittest.mock import AsyncMock

import pytest
import pytest_mock

from backend.copilot import cost_estimation


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
