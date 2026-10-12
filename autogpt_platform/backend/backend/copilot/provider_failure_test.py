import time

import httpx
import pytest
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from backend.copilot.provider_failure import ProviderFailureKind, classify
from backend.copilot.rate_limit import UserPaywalledError
from backend.util.exceptions import ExecutionFailureReason


def _api_error(cls, status: int, headers: dict[str, str] | None = None):
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("boom", response=response, body=None)


class TestWhatTheFailureIs:
    @pytest.mark.parametrize(
        "exc, expected",
        [
            (_api_error(AuthenticationError, 401), ProviderFailureKind.AUTH_EXPIRED),
            (_api_error(PermissionDeniedError, 403), ProviderFailureKind.POLICY_DENIED),
            (_api_error(RateLimitError, 429), ProviderFailureKind.USAGE_LIMIT),
            (_api_error(NotFoundError, 404), ProviderFailureKind.MODEL_UNAVAILABLE),
            (_api_error(InternalServerError, 500), ProviderFailureKind.TRANSIENT),
            (_api_error(BadRequestError, 502), ProviderFailureKind.TRANSIENT),
        ],
    )
    def test_each_provider_refusal_gets_its_own_name(self, exc, expected) -> None:
        failure = classify(exc)
        assert failure is not None
        assert failure.kind is expected

    def test_a_connection_failure_is_worth_retrying(self) -> None:
        exc = APIConnectionError(request=httpx.Request("POST", "https://x"))
        failure = classify(exc)
        assert failure is not None
        assert failure.kind is ProviderFailureKind.TRANSIENT
        assert failure.retryable is True

    def test_a_paywall_is_the_same_denial_a_graph_run_reports(self) -> None:
        # Two names for one denial would split every downstream count in half.
        failure = classify(UserPaywalledError("Max plan required"))
        assert failure is not None
        assert failure.kind.value == ExecutionFailureReason.ENTITLEMENT_REQUIRED.value


class TestWhatIsNotTheProvidersFault:
    def test_our_own_bad_request_is_not_dressed_up_as_a_provider_refusal(self) -> None:
        # A 4xx we did not name is a request we built wrong. Calling it a
        # provider failure would send the user to reconnect a working account.
        assert classify(_api_error(BadRequestError, 400)) is None

    def test_an_ordinary_bug_is_left_alone(self) -> None:
        assert classify(ValueError("our bug")) is None


class TestWhatTheUserShouldDo:
    @pytest.mark.parametrize(
        "exc, retryable, reconnect",
        [
            (_api_error(AuthenticationError, 401), False, True),
            (_api_error(RateLimitError, 429), False, False),
            (_api_error(NotFoundError, 404), False, False),
            (_api_error(InternalServerError, 500), True, False),
        ],
    )
    def test_the_advice_matches_the_failure(self, exc, retryable, reconnect) -> None:
        failure = classify(exc)
        assert failure is not None
        assert failure.retryable is retryable
        assert failure.reconnect_fixes_it is reconnect

    def test_the_advice_travels_with_the_envelope(self) -> None:
        # Computed server-side so a second "is this retryable" cannot drift.
        part = classify(_api_error(AuthenticationError, 401)).as_part()
        assert part["retryable"] is False
        assert part["reconnectFixesIt"] is True
        assert part["kind"] == "auth_expired"


class TestWhenTheLimitLifts:
    def test_a_reported_duration_becomes_a_real_timestamp(self) -> None:
        exc = _api_error(RateLimitError, 429, {"x-ratelimit-reset-requests": "6m0s"})
        failure = classify(exc)
        assert failure is not None
        assert failure.resets_at is not None
        assert 300 < failure.resets_at - int(time.time()) <= 360

    def test_a_reported_epoch_is_passed_through(self) -> None:
        exc = _api_error(
            RateLimitError, 429, {"x-ratelimit-reset-requests": "1900000000"}
        )
        assert classify(exc).resets_at == 1900000000

    def test_token_resets_are_read_when_request_resets_are_absent(self) -> None:
        exc = _api_error(RateLimitError, 429, {"x-ratelimit-reset-tokens": "30s"})
        assert classify(exc).resets_at is not None

    @pytest.mark.parametrize("raw", ["", "soon", "12x", "tomorrow"])
    def test_an_unreadable_reset_is_dropped_rather_than_approximated(
        self, raw: str
    ) -> None:
        # A wrong time is worse than no time: the user schedules around it.
        exc = _api_error(RateLimitError, 429, {"x-ratelimit-reset-requests": raw})
        assert classify(exc).resets_at is None

    def test_no_reset_is_invented_when_the_provider_reported_none(self) -> None:
        assert classify(_api_error(RateLimitError, 429)).resets_at is None


class TestWhichConnectionFailed:
    def test_the_envelope_names_the_connection_that_refused(self) -> None:
        failure = classify(
            _api_error(AuthenticationError, 401),
            auth_provider="codex",
            credential_id="cred-1",
        )
        assert failure is not None
        assert failure.as_part()["authProvider"] == "codex"
        assert failure.as_part()["credentialId"] == "cred-1"

    def test_a_humanized_message_survives_classification(self) -> None:
        # The baseline path already turns a bare connection error into
        # operator-useful advice; classifying must not throw that away.
        failure = classify(
            APIConnectionError(request=httpx.Request("POST", "https://x")),
            message="Can't reach the local LLM backend at http://x/v1.",
        )
        assert failure is not None
        assert "local LLM backend" in failure.message


class TestCodexFailuresSpeakToThePerson:
    """Codex failures carry an internal code as their message.

    ``codex_credential_busy`` is a log line. Shown to a user it explains
    nothing and suggests nothing.
    """

    def test_a_busy_connection_says_so_and_says_waiting_works(self) -> None:
        from backend.integrations.codex.transport import CodexCredentialBusyError

        failure = classify(
            CodexCredentialBusyError("codex_credential_busy"), auth_provider="codex"
        )
        assert failure is not None
        assert "codex_credential_busy" not in failure.message
        assert "briefly unavailable" in failure.message
        # Waiting is what clears it, so the retry affordance is honest.
        assert failure.retryable is True

    def test_the_message_does_not_blame_concurrent_chats(self) -> None:
        # Measured: concurrent chats on one credential do not contend. The
        # lease is contended by credential writes (token refresh), so telling
        # the user to close another chat would point at the wrong thing.
        from backend.integrations.codex.transport import CodexCredentialBusyError

        message = classify(CodexCredentialBusyError("x"), auth_provider="codex").message
        assert "only one" not in message.lower()

    def test_an_unusable_credential_points_at_reconnecting(self) -> None:
        from backend.integrations.codex.transport import CodexCredentialIntegrityError

        failure = classify(CodexCredentialIntegrityError("bad"), auth_provider="codex")
        assert failure is not None
        assert "Reconnect" in failure.message
        assert failure.reconnect_fixes_it is True

    def test_an_explicit_message_still_wins(self) -> None:
        from backend.integrations.codex.transport import CodexCredentialBusyError

        failure = classify(
            CodexCredentialBusyError("x"), auth_provider="codex", message="caller says"
        )
        assert failure.message == "caller says"
