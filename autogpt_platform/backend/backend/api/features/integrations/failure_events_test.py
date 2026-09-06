"""The credential-failure paths raise a Sentry event, tagged with their class.

`caplog` is useless here: the app's logging config captures nothing for
`backend.*` and rewrites `record.msg` with ANSI colour codes, so every test
below attaches its own plain handler and matches with `in`.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest
import sentry_sdk
from pydantic import SecretStr
from sentry_sdk.integrations.logging import LoggingIntegration

from backend.api.features.integrations.failure_events import (
    CredentialFailure,
    report_credential_failure,
)
from backend.api.features.integrations.router import router
from backend.data.model import OAuth2Credentials, OAuthState
from backend.integrations.providers import ProviderName

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

ROUTER_LOGGER = "backend.api.features.integrations.router"


class RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def failures(self) -> list[logging.LogRecord]:
        return [r for r in self.records if hasattr(r, "failure_class")]

    def only_failure(self) -> logging.LogRecord:
        found = self.failures()
        assert len(found) == 1, f"expected 1 failure event, got {len(found)}"
        return found[0]


@pytest.fixture
def router_log():
    yield from _attach(ROUTER_LOGGER)


def _attach(name: str):
    logger = logging.getLogger(name)
    handler = RecordingHandler()
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


@pytest.fixture(autouse=True)
def setup_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


class TestReportCredentialFailure:
    """The mechanism: an ERROR Sentry keeps, carrying a tag a rule can match."""

    def test_emits_at_error_so_the_default_event_level_lets_it_through(
        self, router_log
    ):
        # LoggingIntegration() defaults to event_level=ERROR; a warning here
        # would be dropped before it ever reached Sentry.
        report_credential_failure(
            logging.getLogger(ROUTER_LOGGER),
            CredentialFailure.SCOPES_TOO_NARROW,
            "granted_scopes_narrower",
            "granted scopes are short",
            provider="github",
        )

        record = router_log.only_failure()
        assert record.levelno == logging.ERROR
        assert "granted scopes are short" in record.getMessage()
        assert record.failure_class == "class_08_scopes_too_narrow"
        assert record.reason == "granted_scopes_narrower"

    def test_the_sentry_event_carries_the_tags_and_does_not_leak_them(self):
        captured: list[dict] = []

        class Recorder(sentry_sdk.transport.Transport):
            def capture_envelope(self, envelope):
                for item in envelope.items:
                    if item.type == "event":
                        captured.append(item.payload.json)

        logger = logging.getLogger(ROUTER_LOGGER)
        sentry_sdk.init(
            dsn="https://public@example.invalid/1",
            transport=Recorder(),
            integrations=[LoggingIntegration()],
            default_integrations=False,
        )
        try:
            report_credential_failure(
                logger,
                CredentialFailure.DEVICE_CODE_RACE,
                "throttle_unavailable",
                "throttle unavailable",
                provider="stripe",
                user_id="user-1",
            )
            logger.error("an unrelated later error")
        finally:
            sentry_sdk.init(dsn=None)

        assert len(captured) == 2
        tagged, unrelated = captured
        assert tagged["tags"] == {
            "failure_class": "class_07_device_code_race",
            "reason": "throttle_unavailable",
            "provider": "stripe",
        }
        # user_id is per-user: an extra, never an indexed tag.
        assert tagged["extra"]["user_id"] == "user-1"
        assert not unrelated.get("tags")


class TestCallbackFailureEvents:
    def test_an_invalid_state_token_reports_class_06(self, router_log):
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.store.verify_state_token = AsyncMock(return_value=None)
            resp = client.post(
                "/github/callback",
                json={"code": "auth-code", "state_token": "stale-token"},
            )

        assert resp.status_code == 400
        record = router_log.only_failure()
        assert record.failure_class == "class_06_provider_registration_wrong"
        assert record.reason == "invalid_state_token"
        assert record.provider == "github"

    def test_a_granted_scope_shortfall_reports_class_08(self, router_log):
        state = OAuthState(
            token="state-token",
            provider="github",
            expires_at=9999999999,
            scopes=["repo", "admin:org"],
        )
        short = OAuth2Credentials(
            id="github-cred-1",
            provider="github",
            title="My GitHub",
            access_token=SecretStr("gho_token"),
            refresh_token=None,
            scopes=["repo"],
            username="alice",
        )
        handler = MagicMock()
        handler.handle_default_scopes.return_value = state.scopes
        handler.exchange_code_for_tokens = AsyncMock(return_value=short)

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=None)
            mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_mgr.create = AsyncMock()
            resp = client.post(
                "/github/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        # The credential is stored regardless — that is the defect being made
        # visible, not one being fixed here.
        assert resp.status_code == 200
        record = router_log.only_failure()
        assert record.failure_class == "class_08_scopes_too_narrow"
        assert record.reason == "granted_scopes_narrower"
        assert record.provider == "github"


class TestDeviceAuthFailureEvents:
    def test_an_unavailable_throttle_reports_class_07(self, router_log):
        from backend.api.features.integrations.router import _throttle_upstream

        with patch(
            "backend.data.redis_client.get_redis_async",
            AsyncMock(side_effect=ConnectionError("redis is gone")),
        ):
            throttled = asyncio.run(
                _throttle_upstream(
                    "user-1", ProviderName.GITHUB, seconds=5, scope="initiate"
                )
            )

        # Fails open, and says so where someone can see it.
        assert throttled is False
        record = router_log.only_failure()
        assert record.failure_class == "class_07_device_code_race"
        assert record.reason == "throttle_unavailable"

    def test_an_unreadable_stored_credential_reports_class_07(self, router_log):
        from backend.api.features.integrations.router import _credential_for_grant

        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.store.get_creds_by_id = AsyncMock(
                side_effect=RuntimeError("decrypt failed")
            )
            result = asyncio.run(
                _credential_for_grant("user-1", ProviderName.GITHUB, "cred-1")
            )

        assert result is None
        record = router_log.only_failure()
        assert record.failure_class == "class_07_device_code_race"
        assert record.reason == "credential_unreadable"


class TestProvisioningAndDiscoveryFailureEvents:
    def test_a_managed_sweep_timeout_reports_class_12(self, router_log):
        from backend.api.features.integrations.router import (
            _ensure_managed_credentials_bounded,
        )

        async def never_finishes(*_args, **_kwargs):
            await asyncio.sleep(3600)

        async def run() -> None:
            with (
                patch(
                    "backend.api.features.integrations.router.ensure_managed_credentials",
                    never_finishes,
                ),
                patch(
                    "backend.api.features.integrations.router._MANAGED_PROVISION_TIMEOUT_S",
                    0.01,
                ),
            ):
                await _ensure_managed_credentials_bounded("user-1")
                # The fire-and-forget retry the handler schedules is not the
                # subject here; cancel it so the loop closes cleanly.
                for task in asyncio.all_tasks() - {asyncio.current_task()}:
                    task.cancel()

        asyncio.run(run())

        record = router_log.only_failure()
        assert record.failure_class == "class_12_managed_provisioning_late"
        assert record.reason == "sweep_timeout"
        assert record.user_id == "user-1"

    def test_a_failed_block_load_reports_class_03(self, router_log):
        with patch(
            "backend.blocks.load_all_blocks",
            side_effect=ImportError("a provider _config.py is broken"),
        ):
            resp = client.get("/providers")

        # The list still returns, one provider short — which is exactly why
        # nothing else reports this.
        assert resp.status_code == 200
        record = router_log.only_failure()
        assert record.failure_class == "class_03_provider_unknown_to_frontend"
        assert record.reason == "block_load_failed"
