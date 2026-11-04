from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from posthog.ai.prompts import PromptResult

from backend.util import posthog_prompts
from backend.util.posthog_prompts import (
    compile_prompt,
    fetch_prompt,
    get_posthog_prompts_client,
    posthog_prompts_configured,
    prompt_label,
)
from backend.util.settings import AppEnvironment


@pytest.fixture(autouse=True)
def fresh_client() -> Iterator[None]:
    get_posthog_prompts_client.cache_clear()
    try:
        yield
    finally:
        get_posthog_prompts_client.cache_clear()


def _settings(
    *,
    prompts_key: str = "phx_test",
    prompts_project_key: str = "",
    project_key: str = "phc_events",
    label: str = "",
    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT,
) -> MagicMock:
    return MagicMock(
        secrets=MagicMock(
            posthog_prompts_api_key=prompts_key,
            posthog_prompts_project_api_key=prompts_project_key,
            posthog_api_key=project_key,
            posthog_prompts_host="http://127.0.0.1:9",
        ),
        config=MagicMock(posthog_prompts_label=label, app_env=app_env),
    )


@pytest.mark.parametrize(
    "settings, configured",
    [
        (_settings(), True),
        (_settings(prompts_key=""), False),
        (_settings(project_key=""), False),
        (_settings(project_key="", prompts_project_key="phc_prod"), True),
    ],
    ids=["keys set", "no personal key", "no project key", "prompts project key"],
)
def test_configured_needs_a_personal_key_and_a_project(
    settings: MagicMock, configured: bool
):
    with patch.object(posthog_prompts, "Settings", return_value=settings):
        assert posthog_prompts_configured() is configured


@pytest.mark.parametrize(
    "prompts_project_key, expected",
    [("", "phc_events"), ("phc_prod", "phc_prod")],
    ids=["events project", "another project"],
)
def test_the_client_reads_the_configured_project(
    prompts_project_key: str, expected: str
):
    settings = _settings(prompts_project_key=prompts_project_key)
    with (
        patch.object(posthog_prompts, "Settings", return_value=settings),
        patch.object(posthog_prompts, "Prompts") as prompts,
    ):
        get_posthog_prompts_client()

    prompts.assert_called_once_with(
        personal_api_key="phx_test",
        project_api_key=expected,
        host="http://127.0.0.1:9",
    )


def test_the_client_is_built_once():
    with (
        patch.object(posthog_prompts, "Settings", return_value=_settings()),
        patch.object(posthog_prompts, "Prompts") as prompts,
    ):
        assert get_posthog_prompts_client() is get_posthog_prompts_client()
    prompts.assert_called_once()


@pytest.mark.parametrize(
    "app_env, label, expected",
    [
        (AppEnvironment.PRODUCTION, "", "production"),
        (AppEnvironment.DEVELOPMENT, "", None),
        (AppEnvironment.LOCAL, "", None),
        (AppEnvironment.DEVELOPMENT, "production", "production"),
        (AppEnvironment.PRODUCTION, "latest", None),
        (AppEnvironment.DEVELOPMENT, " staging ", "staging"),
    ],
)
def test_prompt_label(app_env: AppEnvironment, label: str, expected: str | None):
    settings = _settings(app_env=app_env, label=label)
    with patch.object(posthog_prompts, "Settings", return_value=settings):
        assert prompt_label() == expected


@pytest.mark.asyncio
async def test_fetch_reads_the_prompt_at_the_environments_label():
    client = MagicMock()
    result = PromptResult(source="api", prompt="text", name="p", version=1)
    client.get.return_value = result
    settings = _settings(app_env=AppEnvironment.PRODUCTION)
    with (
        patch.object(posthog_prompts, "Settings", return_value=settings),
        patch.object(posthog_prompts, "get_posthog_prompts_client", lambda: client),
    ):
        assert await fetch_prompt("p", cache_ttl_seconds=0, timeout_seconds=5) is result

    client.get.assert_called_once_with(
        "p", with_metadata=True, cache_ttl_seconds=0, label="production"
    )


def test_compile_fills_double_brace_variables_only():
    template = "Hello{{users_information}} {{other}} {users_information}"
    with patch.object(posthog_prompts, "Settings", return_value=_settings()):
        compiled = compile_prompt(template, {"users_information": ""})
    assert compiled == "Hello {{other}} {users_information}"
