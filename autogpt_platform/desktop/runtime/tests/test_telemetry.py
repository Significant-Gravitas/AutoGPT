import sys
from pathlib import Path

import pytest

from autogpt_desktop import ports, settings
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.process import base_env

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build"))
import build_runtime  # noqa: E402


@pytest.mark.parametrize("spelling", [str.upper, str.lower, str.title])
def test_desktop_settings_cannot_enable_broad_reporting(tmp_path, spelling):
    data = DataDir(tmp_path / "profile")
    data.prepare()
    bundle = Bundle(tmp_path / "runtime")
    secret = settings.ensure_secrets(bundle, data)
    allocated = ports.allocate(data.ports_file)
    telemetry = {
        spelling(name): "enabled-private-canary"
        for name in settings.TELEMETRY_DISABLED_ENV
    }
    telemetry.update(
        {
            spelling("OTEL_EXPORTER_OTLP_HEADERS"): "Authorization=private-canary",
            spelling("LANGSMITH_API_KEY"): "private-canary",
            spelling("NEXT_PUBLIC_POSTHOG_KEY"): "private-canary",
            spelling("SENTRY_AUTH_TOKEN"): "private-canary",
        }
    )
    providers = {
        "OPENAI_API_KEY": "fake-openai",
        "ANTHROPIC_API_KEY": "fake-anthropic",
        "OPEN_ROUTER_API_KEY": "fake-openrouter",
        "CHAT_BASE_URL": "http://127.0.0.1:11434/v1",
    }
    backend = settings.backend_environment(
        bundle, data, allocated, secret, telemetry | providers, defaults=telemetry
    )
    for name, expected in settings.TELEMETRY_DISABLED_ENV.items():
        assert backend[name] == expected
    assert not any("private-canary" in value for value in backend.values())
    for name, expected in providers.items():
        assert backend[name] == expected
    assert backend["ENABLE_AUTH"] == "true"
    assert backend["BETTER_AUTH_SECRET"] == secret["BETTER_AUTH_SECRET"]
    frontend = settings.frontend_environment(backend, allocated, secret, data)
    assert frontend["OPENAI_API_KEY"] == "fake-openai"
    assert frontend["DISABLE_SENTRY"] == frontend["OTEL_SDK_DISABLED"] == "true"
    assert frontend["NEXT_TELEMETRY_DISABLED"] == "1"


def test_ambient_reporting_settings_are_not_inherited(monkeypatch):
    for name in [
        "SENTRY_DSN",
        "POSTHOG_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ]:
        monkeypatch.setenv(name, "private-canary")
    assert not any("private-canary" in value for value in base_env().values())


@pytest.mark.parametrize("spelling", [str.upper, str.lower, str.title])
def test_frontend_build_clears_ambient_reporting_and_web_consent(monkeypatch, spelling):
    for name in build_runtime.FRONTEND_BUILD_ENV:
        monkeypatch.setenv(spelling(name), "private-canary")
    monkeypatch.setenv(spelling("OTEL_EXPORTER_OTLP_HEADERS"), "private-canary")
    monkeypatch.setenv(spelling("LANGSMITH_API_KEY"), "private-canary")
    monkeypatch.setenv("DESKTOP_TEST_UNRELATED", "preserved")
    env = build_runtime.frontend_build_environment()
    assert not any("private-canary" in value for value in env.values())
    assert env["DESKTOP_TEST_UNRELATED"] == "preserved"
    assert env["NEXT_PUBLIC_BEHAVE_AS"] == "LOCAL"
    assert env["NEXT_PUBLIC_APP_ENV"] == "local"
    assert env["NEXT_PUBLIC_POSTHOG_KEY"] == env["NEXT_PUBLIC_POSTHOG_HOST"] == ""
    assert env["NEXT_PUBLIC_COOKIEBOT_CBID"] == ""
    assert env["NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID"] == ""
    assert env["SENTRY_AUTH_TOKEN"] == env["DATAFAST_API_KEY"] == ""
    assert env["DISABLE_SENTRY"] == env["OTEL_SDK_DISABLED"] == "true"
    assert env["NEXT_TELEMETRY_DISABLED"] == "1"
