"""The SDK path reports provider failures the way the baseline does.

Most turns run on the SDK engine, and the codex route always does. Until
this, a provider failure there reached the chat as CLI text: the marker
carried no envelope and the retry affordance came from a list of
engine-level codes that knows nothing about expired logins or spent quotas.
"""

from datetime import UTC, datetime

from backend.copilot.markers import provider_failure_of
from backend.copilot.model import ChatSession
from backend.copilot.provider_failure import ProviderFailureKind, classify
from backend.copilot.sdk.service import (
    _append_error_marker,
    _InterruptedAttempt,
    _provider_failure_for,
)
from backend.integrations.codex.transport import CodexCredentialIntegrityError


class _Gateway:
    def __init__(self, failure=None):
        self.last_failure = failure


class _Ctx:
    def __init__(self, gateway=None):
        self.codex_gateway = gateway


def _session() -> ChatSession:
    now = datetime(2026, 8, 21, tzinfo=UTC)
    return ChatSession(
        session_id="s1",
        user_id="u1",
        usage=[],
        started_at=now,
        updated_at=now,
        messages=[],
    )


class TestReadingTheGateway:
    def test_a_named_codex_failure_is_found(self) -> None:
        failure = classify(CodexCredentialIntegrityError("bad"), auth_provider="codex")
        assert _provider_failure_for(_Ctx(_Gateway(failure))) is failure

    def test_the_platform_route_has_no_gateway_to_ask(self) -> None:
        assert _provider_failure_for(_Ctx(None)) is None

    def test_a_codex_turn_the_gateway_could_not_name_reports_nothing(self) -> None:
        # Caller keeps its existing behaviour rather than inventing a kind.
        assert _provider_failure_for(_Ctx(_Gateway(None))) is None


class TestTheMarkerMatchesTheBaseline:
    def test_the_sdk_marker_carries_the_envelope(self) -> None:
        # Same row shape both engines produce, so one reader serves both.
        session = _session()
        failure = classify(CodexCredentialIntegrityError("bad"), auth_provider="codex")
        _append_error_marker(
            session,
            failure.message,
            retryable=failure.retryable,
            failure=failure.as_part(),
        )

        recorded = provider_failure_of(session.messages[-1])
        assert recorded is not None
        assert recorded["kind"] == ProviderFailureKind.INVALID_CREDENTIAL.value
        assert recorded["reconnectFixesIt"] is True

    def test_an_unclassified_sdk_error_still_writes_a_marker(self) -> None:
        session = _session()
        _append_error_marker(session, "the assistant stopped", retryable=True)

        assert session.messages, "an unclassified failure must still be recorded"
        assert provider_failure_of(session.messages[-1]) is None

    def test_retry_finalizer_keeps_the_provider_envelope(self) -> None:
        session = _session()
        failure = classify(CodexCredentialIntegrityError("bad"), auth_provider="codex")

        _InterruptedAttempt().finalize(
            session,
            state=None,
            display_msg=failure.message,
            retryable=failure.retryable,
            failure=failure.as_part(),
        )

        recorded = provider_failure_of(session.messages[-1])
        assert recorded is not None
        assert recorded["kind"] == ProviderFailureKind.INVALID_CREDENTIAL.value
        assert recorded["reconnectFixesIt"] is True
