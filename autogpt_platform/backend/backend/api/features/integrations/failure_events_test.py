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

    def test_the_fields_reach_gcp_cloud_logging(self):
        """`json_fields` is the only extra GCP's handler carries through."""
        import io as _io
        import json

        from google.cloud.logging.handlers import StructuredLogHandler

        stream = _io.StringIO()
        logger = logging.getLogger("backend.credential_failure_gcp_probe")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        handler = StructuredLogHandler(stream=stream)
        logger.addHandler(handler)
        try:
            report_credential_failure(
                logger,
                CredentialFailure.MANAGED_PROVISIONING_LATE,
                "sweep_timeout",
                "sweep timed out",
                user_id="user-1",
            )
        finally:
            logger.removeHandler(handler)

        entry = json.loads(stream.getvalue().strip().splitlines()[-1])
        assert entry["failure_class"] == "class_12_managed_provisioning_late"
        assert entry["reason"] == "sweep_timeout"
        assert entry["user_id"] == "user-1"

    def test_a_reserved_context_key_does_not_raise_out_of_the_caller(self):
        logger = logging.getLogger(ROUTER_LOGGER)
        handler = RecordingHandler()
        logger.addHandler(handler)
        try:
            report_credential_failure(
                logger,
                CredentialFailure.DEVICE_CODE_RACE,
                "credential_unreadable",
                "unreadable",
                module="github",
            )
        finally:
            logger.removeHandler(handler)

        record = handler.only_failure()
        assert record.ctx_module == "github"
        # GCP keeps the original name, so a query on it does not change.
        assert record.json_fields["module"] == "github"

    def test_the_record_points_at_the_call_site_not_this_helper(self):
        logger = logging.getLogger(ROUTER_LOGGER)
        handler = RecordingHandler()
        logger.addHandler(handler)
        try:
            report_credential_failure(
                logger,
                CredentialFailure.DEVICE_CODE_RACE,
                "credential_unreadable",
                "unreadable",
            )
        finally:
            logger.removeHandler(handler)

        assert handler.only_failure().filename == "failure_events_test.py"

    def test_the_sentry_event_carries_the_tags_and_does_not_leak_them(self):
        captured: list[dict] = []

        class Recorder(sentry_sdk.transport.Transport):
            def capture_envelope(self, envelope):
                for item in envelope.items:
                    if item.type == "event":
                        captured.append(item.payload.json)

        logger = logging.getLogger(ROUTER_LOGGER)
        # `init(dsn=None)` leaves the client bound and `is_active()` True, so
        # restoring the original is the only teardown that does not leak into
        # `metrics_test.py::test_no_sentry_client_is_active_under_pytest`.
        original_client = sentry_sdk.get_client()
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
            sentry_sdk.get_global_scope().set_client(original_client)

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
        # The message is the Sentry grouping key: one issue for the provider's
        # broken config, not one per user who hit it.
        assert "Invalid or expired state token" in record.getMessage()
        # The message is the Sentry grouping key, so the id belongs in the
        # context and nowhere else — otherwise class 06 is one issue per user.
        assert record.user_id
        assert record.user_id not in record.getMessage()

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
    async def test_an_unavailable_throttle_stays_a_warning(self, router_log):
        from backend.api.features.integrations.router import _throttle_upstream

        with patch(
            "backend.data.redis_client.get_redis_async",
            AsyncMock(side_effect=ConnectionError("redis is gone")),
        ):
            throttled = await _throttle_upstream(
                "user-1", ProviderName.GITHUB, seconds=5, scope="initiate"
            )

        # Fails open. A Redis outage is reported by infrastructure monitoring
        # and every ~5s poll re-enters here, so this must not raise an event.
        assert throttled is False
        assert router_log.failures() == []
        assert any("throttle unavailable" in r.getMessage() for r in router_log.records)

    async def test_an_unreadable_stored_credential_reports_class_07(self, router_log):
        from backend.api.features.integrations.router import _credential_for_grant

        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.store.get_creds_by_id = AsyncMock(
                side_effect=RuntimeError("decrypt failed")
            )
            result = await _credential_for_grant(
                "user-1", ProviderName.GITHUB, "cred-1"
            )

        assert result is None
        record = router_log.only_failure()
        assert record.failure_class == "class_07_device_code_race"
        assert record.reason == "credential_unreadable"


class TestProvisioningAndDiscoveryFailureEvents:
    async def test_a_managed_sweep_timeout_reports_class_12(self, router_log):
        from backend.api.features.integrations.router import (
            _ensure_managed_credentials_bounded,
        )

        async def never_finishes(*_args, **_kwargs):
            await asyncio.sleep(3600)

        before = asyncio.all_tasks()
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

        # Cancel only the fire-and-forget retry this call scheduled — the loop
        # is session-scoped and its other tasks belong to the whole suite.
        scheduled = asyncio.all_tasks() - before
        for task in scheduled:
            task.cancel()
        await asyncio.gather(*scheduled, return_exceptions=True)

        record = router_log.only_failure()
        assert record.failure_class == "class_12_managed_provisioning_late"
        assert record.reason == "sweep_timeout"
        assert record.user_id == "user-1"

    def test_a_failed_block_load_stays_a_warning(self, router_log):
        with patch(
            "backend.blocks.load_all_blocks",
            side_effect=ImportError("a provider _config.py is broken"),
        ):
            resp = client.get("/providers")

        # `load_all_blocks` caches values, not exceptions, so a broken
        # `_config.py` re-raises on every page load fleet-wide. That volume
        # belongs nowhere near the event quota.
        assert resp.status_code == 200
        assert router_log.failures() == []
        assert any(
            "Failed to load blocks" in r.getMessage() for r in router_log.records
        )
