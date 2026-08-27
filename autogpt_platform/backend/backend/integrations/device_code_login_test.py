"""The device flow is mostly about not giving up too early.

A user approving a sign-in on their phone takes as long as it takes, and
the provider says so with two codes that read like errors and are not:
``authorization_pending`` and ``slow_down``. Treating either as a failure
abandons a sign-in that was going fine, and the user sees "sign-in failed"
while looking at the approval screen that was about to work.

These tests pin that, plus the outcomes that genuinely are failures and
have to be told apart -- declining and expiring are different things to say
on screen.
"""

from typing import Any

import pytest

from backend.integrations.device_code_login import (
    DeviceCodeConfig,
    DeviceCodeDenied,
    DeviceCodeError,
    DeviceCodeExpired,
    DeviceCodeGrant,
    poll_for_tokens,
    request_device_code,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeRequests:
    """Answers each POST with the next scripted payload."""

    def __init__(self, *payloads: dict[str, Any]) -> None:
        self._payloads = list(payloads)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def post(self, url: str, *args, **kwargs) -> FakeResponse:
        self.calls.append((url, kwargs.get("data") or {}))
        payload = self._payloads.pop(0) if self._payloads else {}
        return FakeResponse(payload)


class FakeClock:
    """Advances only when the code under test sleeps."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def _config(**over) -> DeviceCodeConfig:
    fields: dict = {
        "device_authorization_url": "https://provider.invalid/device/code",
        "token_url": "https://provider.invalid/token",
        "client_id": "a-client",
    }
    fields.update(over)
    return DeviceCodeConfig(**fields)


def _grant(**over) -> DeviceCodeGrant:
    fields: dict = {
        "device_code": "dev-code",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://provider.invalid/activate",
        "interval_seconds": 5,
    }
    fields.update(over)
    return DeviceCodeGrant(**fields)


class TestAskingForACode:
    @pytest.mark.asyncio
    async def test_returns_what_the_user_needs_to_approve(self) -> None:
        http = FakeRequests(
            {
                "device_code": "dev-code",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://provider.invalid/activate",
                "verification_uri_complete": (
                    "https://provider.invalid/activate?code=ABCD-EFGH"
                ),
                "interval": 7,
                "expires_in": 600,
            }
        )

        grant = await request_device_code(_config(), requests=http)  # type: ignore[arg-type]

        assert grant.user_code == "ABCD-EFGH"
        assert grant.verification_uri_complete is not None
        assert grant.interval_seconds == 7
        assert grant.expires_in_seconds == 600

    @pytest.mark.asyncio
    async def test_reads_the_other_spelling_of_the_verification_url(self) -> None:
        """RFC 8628 says ``verification_uri``; several providers ship
        ``verification_url``. Reading only one sends the user nowhere."""
        http = FakeRequests(
            {
                "device_code": "dev-code",
                "user_code": "WXYZ",
                "verification_url": "https://provider.invalid/activate",
            }
        )

        grant = await request_device_code(_config(), requests=http)  # type: ignore[arg-type]

        assert grant.verification_uri == "https://provider.invalid/activate"

    @pytest.mark.asyncio
    async def test_a_response_with_no_device_code_is_an_error(self) -> None:
        http = FakeRequests({"error": "invalid_client"})

        with pytest.raises(DeviceCodeError) as raised:
            await request_device_code(_config(), requests=http)  # type: ignore[arg-type]

        assert raised.value.code == "invalid_client"

    @pytest.mark.asyncio
    async def test_the_device_code_is_not_in_the_repr(self) -> None:
        grant = _grant()
        assert "dev-code" not in repr(grant)


class TestWaitingForApproval:
    @pytest.mark.asyncio
    async def test_keeps_waiting_while_the_user_is_still_approving(self) -> None:
        """The bug this module exists to not have written twice."""
        clock = FakeClock()
        http = FakeRequests(
            {"error": "authorization_pending"},
            {"error": "authorization_pending"},
            {"access_token": "at", "refresh_token": "rt", "expires_in": 3600},
        )

        tokens = await poll_for_tokens(
            _config(),
            _grant(),
            requests=http,  # type: ignore[arg-type]
            sleep=clock.sleep,
            now=clock.now,
        )

        assert tokens.access_token.get_secret_value() == "at"
        assert tokens.refresh_token is not None
        assert len(http.calls) == 3

    @pytest.mark.asyncio
    async def test_slows_down_when_asked_instead_of_giving_up(self) -> None:
        """``slow_down`` means "you are polling too fast", not "stop"."""
        clock = FakeClock()
        http = FakeRequests(
            {"error": "slow_down"},
            {"access_token": "at"},
        )

        await poll_for_tokens(
            _config(),
            _grant(interval_seconds=5),
            requests=http,  # type: ignore[arg-type]
            sleep=clock.sleep,
            now=clock.now,
        )

        # First poll at the server interval, second five seconds slower.
        assert clock.slept == [5, 10]

    @pytest.mark.asyncio
    async def test_an_expiry_is_told_apart_from_a_refusal(self) -> None:
        """Two different things to say on screen: "you ran out of time, try
        again" versus "you declined"."""
        clock = FakeClock()
        expired = FakeRequests({"error": "expired_token"})
        with pytest.raises(DeviceCodeExpired):
            await poll_for_tokens(
                _config(),
                _grant(),
                requests=expired,  # type: ignore[arg-type]
                sleep=clock.sleep,
                now=clock.now,
            )

        denied = FakeRequests({"error": "access_denied"})
        with pytest.raises(DeviceCodeDenied):
            await poll_for_tokens(
                _config(),
                _grant(),
                requests=denied,  # type: ignore[arg-type]
                sleep=FakeClock().sleep,
                now=FakeClock().now,
            )

    @pytest.mark.asyncio
    async def test_gives_up_at_the_deadline_rather_than_polling_forever(self) -> None:
        """A user who walked away must not hold a request open. Driven by
        the clock, so this asserts the behaviour without waiting for it."""
        clock = FakeClock()
        http = FakeRequests(*[{"error": "authorization_pending"}] * 100)

        with pytest.raises(DeviceCodeExpired):
            await poll_for_tokens(
                _config(max_wait_seconds=30),
                _grant(interval_seconds=5),
                requests=http,  # type: ignore[arg-type]
                sleep=clock.sleep,
                now=clock.now,
            )

        assert clock.t - 1000.0 >= 30

    @pytest.mark.asyncio
    async def test_the_providers_own_expiry_wins_when_it_is_shorter(self) -> None:
        """Polling a code the provider has already retired is just noise."""
        clock = FakeClock()
        http = FakeRequests(*[{"error": "authorization_pending"}] * 100)

        with pytest.raises(DeviceCodeExpired):
            await poll_for_tokens(
                _config(max_wait_seconds=900),
                _grant(interval_seconds=5, expires_in_seconds=20),
                requests=http,  # type: ignore[arg-type]
                sleep=clock.sleep,
                now=clock.now,
            )

        assert clock.t - 1000.0 < 900

    @pytest.mark.asyncio
    async def test_an_unrecognised_error_is_raised_rather_than_retried(self) -> None:
        """Retrying something we do not understand turns a clear failure
        into a hang."""
        clock = FakeClock()
        http = FakeRequests({"error": "invalid_grant", "error_description": "nope"})

        with pytest.raises(DeviceCodeError) as raised:
            await poll_for_tokens(
                _config(),
                _grant(),
                requests=http,  # type: ignore[arg-type]
                sleep=clock.sleep,
                now=clock.now,
            )

        assert raised.value.code == "invalid_grant"
        assert raised.value.description == "nope"


class TestWhatIsStored:
    @pytest.mark.asyncio
    async def test_expiry_is_absolute_not_relative(self) -> None:
        """``expires_in`` is only meaningful beside the moment it was
        issued, and that moment is not carried anywhere once the credential
        is persisted."""
        clock = FakeClock()
        http = FakeRequests({"access_token": "at", "expires_in": 3600})

        tokens = await poll_for_tokens(
            _config(),
            _grant(interval_seconds=1),
            requests=http,  # type: ignore[arg-type]
            sleep=clock.sleep,
            now=clock.now,
        )

        assert tokens.access_token_expires_at == int(clock.t) + 3600

    @pytest.mark.asyncio
    async def test_the_access_token_is_not_in_the_repr(self) -> None:
        clock = FakeClock()
        http = FakeRequests({"access_token": "super-secret", "scope": "a b"})

        tokens = await poll_for_tokens(
            _config(),
            _grant(interval_seconds=1),
            requests=http,  # type: ignore[arg-type]
            sleep=clock.sleep,
            now=clock.now,
        )

        assert "super-secret" not in repr(tokens)
        assert tokens.scopes == ("a", "b")
