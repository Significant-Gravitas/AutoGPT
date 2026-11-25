"""Without a Postmark token, production refuses to start the notification
service, and every other install starts and skips the email."""

import pytest
from prisma.enums import NotificationType

from backend.notifications import email as delivery
from backend.notifications.notifications import NotificationManager
from backend.notifications.renderer import RenderedEmail, build_urls
from backend.util.exceptions import MissingConfigError
from backend.util.settings import AppEnvironment, BehaveAs


def _configure(monkeypatch, app_env, behave_as, token=""):
    monkeypatch.setattr(delivery.settings.config, "app_env", app_env)
    monkeypatch.setattr(delivery.settings.config, "behave_as", behave_as)
    monkeypatch.setattr(delivery.settings.secrets, "postmark_server_api_token", token)


def test_production_without_postmark_refuses_to_start(monkeypatch):
    _configure(monkeypatch, AppEnvironment.PRODUCTION, BehaveAs.CLOUD)

    with pytest.raises(MissingConfigError, match="POSTMARK_SERVER_API_TOKEN"):
        NotificationManager()


def test_production_with_postmark_starts(monkeypatch):
    _configure(
        monkeypatch, AppEnvironment.PRODUCTION, BehaveAs.CLOUD, token="server-token"
    )

    assert NotificationManager().email_sender.postmark is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "app_env,behave_as",
    [
        pytest.param(AppEnvironment.LOCAL, BehaveAs.LOCAL, id="self-hosted"),
        pytest.param(
            AppEnvironment.DEVELOPMENT, BehaveAs.CLOUD, id="multi-user-single-container"
        ),
        pytest.param(AppEnvironment.PRODUCTION, BehaveAs.LOCAL, id="prod-app-env-only"),
        pytest.param(AppEnvironment.LOCAL, BehaveAs.CLOUD, id="cloud-behaviour-only"),
    ],
)
async def test_other_installs_start_and_skip_email(monkeypatch, app_env, behave_as):
    _configure(monkeypatch, app_env, behave_as)
    manager = NotificationManager()

    await manager.email_sender._deliver(
        NotificationType.BRIEFING,
        "user@example.org",
        RenderedEmail(subject="Brief", preheader="Brief", html="<p>B</p>", text="B"),
        build_urls("https://example.org/unsubscribe"),
    )

    assert manager.email_sender.postmark is None
