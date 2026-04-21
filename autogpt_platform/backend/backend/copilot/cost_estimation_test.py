"""Tests for CoPilot cost-estimation helpers."""

import pytest

from backend.copilot import cost_approval


@pytest.mark.asyncio
async def test_approval_token_round_trip_valid():
    token = await cost_approval.generate_cost_approval_token(
        user_id="user-1",
        session_id="sess-1",
        message="Analyze this task",
        is_user_message=True,
        context={"url": "https://example.com", "content": "ctx"},
        file_ids=["11111111-1111-1111-1111-111111111111"],
        mode="extended_thinking",
        model="advanced",
        ttl_seconds=600,
    )

    is_valid = await cost_approval.validate_cost_approval_token(
        token=token,
        user_id="user-1",
        session_id="sess-1",
        message="Analyze this task",
        is_user_message=True,
        context={"url": "https://example.com", "content": "ctx"},
        file_ids=["11111111-1111-1111-1111-111111111111"],
        mode="extended_thinking",
        model="advanced",
    )

    assert is_valid is True


@pytest.mark.asyncio
async def test_approval_token_invalid_when_message_changes():
    token = await cost_approval.generate_cost_approval_token(
        user_id="user-1",
        session_id="sess-1",
        message="Original message",
        is_user_message=True,
        context=None,
        file_ids=None,
        mode=None,
        model=None,
        ttl_seconds=600,
    )

    is_valid = await cost_approval.validate_cost_approval_token(
        token=token,
        user_id="user-1",
        session_id="sess-1",
        message="Tampered message",
        is_user_message=True,
        context=None,
        file_ids=None,
        mode=None,
        model=None,
    )

    assert is_valid is False


@pytest.mark.asyncio
async def test_approval_token_invalid_when_expired():
    token = await cost_approval.generate_cost_approval_token(
        user_id="user-1",
        session_id="sess-1",
        message="Original message",
        is_user_message=True,
        context=None,
        file_ids=None,
        mode=None,
        model=None,
        ttl_seconds=-1,
    )

    is_valid = await cost_approval.validate_cost_approval_token(
        token=token,
        user_id="user-1",
        session_id="sess-1",
        message="Original message",
        is_user_message=True,
        context=None,
        file_ids=None,
        mode=None,
        model=None,
    )

    assert is_valid is False


@pytest.mark.asyncio
async def test_approval_token_invalid_for_malformed_payload():
    is_valid = await cost_approval.validate_cost_approval_token(
        token="not-a-valid-token.payload",
        user_id="user-1",
        session_id="sess-1",
        message="Original message",
        is_user_message=True,
        context=None,
        file_ids=None,
        mode=None,
        model=None,
    )

    assert is_valid is False
