"""Tests for CoPilot cost-approval tokens."""

import hashlib
import logging

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
        message_count=4,
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
        message_count=4,
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
        message_count=2,
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
        message_count=2,
    )

    assert is_valid is False


@pytest.mark.asyncio
async def test_approval_token_invalid_when_history_grows():
    token = await cost_approval.generate_cost_approval_token(
        user_id="user-1",
        session_id="sess-1",
        message="Original message",
        is_user_message=True,
        context=None,
        file_ids=None,
        mode=None,
        model=None,
        message_count=30,
        ttl_seconds=600,
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
        message_count=31,
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
        message_count=1,
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
        message_count=1,
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
        message_count=0,
    )

    assert is_valid is False


def test_dedicated_signing_key_is_used(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        cost_approval.settings.secrets, "cost_approval_signing_key", "dedicated-key"
    )

    assert cost_approval._approval_secret() == b"dedicated-key"


def test_missing_dedicated_key_derives_a_stable_secret_from_encryption_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cost_approval.settings.secrets, "cost_approval_signing_key", "")
    monkeypatch.setattr(cost_approval.settings.secrets, "encryption_key", "enc-key")

    expected = hashlib.sha256(b"autogpt-cost-approval-v1:enc-key").digest()
    assert cost_approval._approval_secret() == expected
    assert cost_approval._approval_secret() == expected


def test_missing_signing_material_warns_and_stays_stable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(cost_approval.settings.secrets, "cost_approval_signing_key", "")
    monkeypatch.setattr(cost_approval.settings.secrets, "encryption_key", "")
    monkeypatch.setattr(cost_approval, "_warned_missing_signing_key", False)

    with caplog.at_level(logging.WARNING, logger="backend.copilot.cost_approval"):
        first = cost_approval._approval_secret()
        second = cost_approval._approval_secret()

    assert first == second == cost_approval._DEV_FALLBACK_SECRET
    assert "[SECURITY]" in caplog.text
    assert caplog.text.count("[SECURITY]") == 1
