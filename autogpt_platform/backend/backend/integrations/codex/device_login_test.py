import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.integrations.codex.chatgpt_auth import (
    ChatGPTDeviceCode,
    ChatGPTPollResult,
    CodexAuthError,
)
from backend.integrations.codex.device_login import CodexHttpDeviceLogin

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _device(interval: int = 5, expires_in: int | None = 900) -> ChatGPTDeviceCode:
    return ChatGPTDeviceCode(
        device_auth_id="deviceauth_abc",
        user_code="FWK1-91A0I",
        interval=interval,
        expires_at=_NOW + timedelta(seconds=expires_in) if expires_in else None,
    )


class _Clock:
    """A clock that only advances when the login sleeps."""

    def __init__(self) -> None:
        self.now = _NOW
        self.slept: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)

    def __call__(self) -> datetime:
        return self.now


def _login(
    clock: _Clock, device: ChatGPTDeviceCode | None = None, timeout: float = 900
):
    return CodexHttpDeviceLogin(
        device or _device(),
        timeout_seconds=timeout,
        sleep=clock.sleep,
        now=clock,
    )


def _patch_poll(results: list[ChatGPTPollResult]):
    return patch(
        "backend.integrations.codex.device_login.poll_device_code",
        new=AsyncMock(side_effect=results),
    )


def test_details_carry_the_code_and_the_verify_url() -> None:
    login = _login(_Clock())
    assert login.details.user_code == "FWK1-91A0I"
    assert login.details.verification_url == "https://auth.openai.com/codex/device"
    # The coordinator overwrites this with the id it minted.
    assert login.details.login_id == ""


@pytest.mark.asyncio
async def test_polling_continues_while_the_user_is_still_approving() -> None:
    clock = _Clock()
    results = [
        ChatGPTPollResult(status="pending"),
        ChatGPTPollResult(status="pending"),
        ChatGPTPollResult(
            status="approved", authorization_code="ac", code_verifier="cv"
        ),
    ]
    tokens = AsyncMock()
    with (
        _patch_poll(results),
        patch(
            "backend.integrations.codex.device_login.exchange_authorization_code",
            new=tokens,
        ),
        patch(
            "backend.integrations.codex.device_login.bundle_from_tokens",
            return_value="bundle",
        ),
        patch(
            "backend.integrations.codex.device_login.credentials_from_bundle",
            return_value="creds",
        ),
        patch(
            "backend.integrations.codex.device_login.account_snapshot",
            return_value=None,
        ),
        patch(
            "backend.integrations.codex.device_login.CodexLoginCompletion",
            lambda **kw: kw,
        ),
    ):
        completion = await _login(clock).wait()

    assert clock.slept == [5, 5]
    assert completion["bundle"] == "bundle"
    tokens.assert_awaited_once_with("ac", "cv")


@pytest.mark.asyncio
async def test_slow_down_backs_off_instead_of_holding_the_same_cadence() -> None:
    """Keeping the old interval means the server answers slow_down forever."""
    clock = _Clock()
    results = [
        ChatGPTPollResult(status="slow_down"),
        ChatGPTPollResult(status="slow_down"),
        ChatGPTPollResult(status="pending"),
        ChatGPTPollResult(status="denied"),
    ]
    with _patch_poll(results):
        with pytest.raises(CodexAuthError):
            await _login(clock).wait()

    assert clock.slept == [10, 20, 20]


@pytest.mark.asyncio
async def test_backoff_is_capped() -> None:
    clock = _Clock()
    results = [ChatGPTPollResult(status="slow_down") for _ in range(8)]
    results.append(ChatGPTPollResult(status="denied"))
    with _patch_poll(results):
        with pytest.raises(CodexAuthError):
            await _login(clock).wait()

    assert max(clock.slept) == 60


@pytest.mark.asyncio
async def test_a_declined_sign_in_stops_rather_than_waiting_out_the_clock() -> None:
    with _patch_poll([ChatGPTPollResult(status="denied")]):
        with pytest.raises(CodexAuthError, match="declined"):
            await _login(_Clock()).wait()


@pytest.mark.asyncio
async def test_an_expired_code_stops_immediately() -> None:
    with _patch_poll([ChatGPTPollResult(status="expired")]):
        with pytest.raises(CodexAuthError, match="expired"):
            await _login(_Clock()).wait()


@pytest.mark.asyncio
async def test_polling_stops_at_the_codes_own_expiry_not_our_timeout() -> None:
    """The configured timeout is long; the code's expiry is what binds here."""
    clock = _Clock()
    login = _login(clock, device=_device(expires_in=12), timeout=900)
    with _patch_poll([ChatGPTPollResult(status="pending")] * 20):
        with pytest.raises(CodexAuthError, match="expired"):
            await login.wait()

    # The final wait is trimmed to what is left of the code's life, so the
    # expiry is reported at 12s rather than at the next 5s boundary.
    assert clock.slept == [5, 5, 2]


@pytest.mark.asyncio
async def test_cancelling_stops_the_loop() -> None:
    clock = _Clock()
    login = _login(clock)
    await login.cancel()

    with _patch_poll([ChatGPTPollResult(status="pending")]):
        with pytest.raises(CodexAuthError, match="canceled"):
            await login.wait()

    assert clock.slept == []


@pytest.mark.asyncio
async def test_cancelling_interrupts_an_in_flight_poll() -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    async def blocked_poll(_device_auth_id: str, _user_code: str):
        started.set()
        await never.wait()

    login = _login(_Clock())
    with patch(
        "backend.integrations.codex.device_login.poll_device_code",
        new=blocked_poll,
    ):
        waiting = asyncio.create_task(login.wait())
        await started.wait()
        await login.cancel()

        with pytest.raises(CodexAuthError, match="canceled"):
            await asyncio.wait_for(waiting, timeout=0.1)
