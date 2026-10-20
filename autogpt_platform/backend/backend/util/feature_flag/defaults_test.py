import os
import re
from pathlib import Path

import pytest
from fastapi import HTTPException
from ldclient import Context
from ldclient.evaluation import EvaluationDetail
from posthog.types import FeatureFlagResult

import backend.util.feature_flag as ff
from backend.util.feature_flag import Flag, frontend
from backend.util.feature_flag.defaults import DEFAULT_ON_FLAGS
from backend.util.settings import FeatureFlagBackend

MATURE_FLAGS = [
    "autogpt-new-layout",
    "chat-search",
    "artifacts-page",
    "hire-experts",
    "skills-hub",
    "onboarding-brain-dump",
    "onboarding-expert-team",
    "graphiti-memory",
    "graphiti-communities-enabled",
    "expert-spend-approval",
    "ai-agent-execution-summary",
]


def test_mature_boolean_defaults_match_frontend():
    source = (
        Path(__file__).resolve().parents[4]
        / "frontend/src/services/feature-flags/use-get-flag.ts"
    ).read_text(encoding="utf-8")
    declaration = re.search(
        r"^const defaultFlags = \{(.*?)^\} satisfies Record<Flag, unknown>;",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert declaration is not None, "Cannot locate the frontend defaultFlags map"
    frontend_defaults = {
        frontend.FrontendFlag[name].value
        for name in re.findall(
            r"^\s*\[Flag\.([A-Z0-9_]+)\]: true,\s*$",
            declaration.group(1),
            re.MULTILINE,
        )
    }
    assert frontend_defaults == DEFAULT_ON_FLAGS & {
        flag.value for flag in frontend.FrontendFlag
    }


@pytest.fixture(autouse=True)
def isolated_flags(monkeypatch, mocker):
    for name in list(os.environ):
        if name.startswith(("FORCE_FLAG_", "NEXT_PUBLIC_FORCE_FLAG_", "FORCE_ALL_")):
            monkeypatch.delenv(name)
    mocker.patch.object(
        ff.settings.config, "feature_flag_backend", FeatureFlagBackend.POSTHOG
    )
    mocker.patch.object(ff.posthog, "get_flag_client", return_value=None)
    mocker.patch.object(
        ff,
        "_fetch_user_context_status",
        return_value=(Context.create("test-user"), True),
    )


@pytest.mark.parametrize("key", MATURE_FLAGS)
async def test_mature_defaults_without_vendor_are_not_authoritative(key):
    assert await ff._evaluate_boolean_flag(key, "test-user") == (True, False)
    assert await ff._evaluate_boolean_flag(key, "test-user", False) == (False, False)


@pytest.mark.parametrize(
    "key",
    [
        "enable-platform-payment",
        "copilot-stream-runtime",
        "copilot-sdk",
        "dream-pass-enabled",
        "dream-pass-invalidate-entity",
        "dream-skill-learning-enabled",
        "unknown-flag",
    ],
)
async def test_other_flags_keep_false_default(key):
    assert await ff._evaluate_boolean_flag(key, "test-user") == (False, False)
    assert await ff._evaluate_boolean_flag(key, "test-user", True) == (True, False)


@pytest.mark.parametrize("key", MATURE_FLAGS)
@pytest.mark.parametrize(
    "backend", [FeatureFlagBackend.POSTHOG, FeatureFlagBackend.LAUNCHDARKLY]
)
@pytest.mark.parametrize("value", [False, None, "not-a-boolean", "failure"])
async def test_vendor_false_wins_but_unavailable_and_invalid_answers_use_default(
    key, backend, value, mocker
):
    mocker.patch.object(ff.settings.config, "feature_flag_backend", backend)
    client = mocker.Mock()
    if backend is FeatureFlagBackend.POSTHOG:
        mocker.patch.object(ff.posthog, "get_flag_client", return_value=client)
        client.get_feature_flag_result.return_value = (
            FeatureFlagResult.from_value_and_payload(key, value, None)
            if value is not None
            else None
        )
        if value == "failure":
            client.get_feature_flag_result.side_effect = RuntimeError("vendor failed")
    else:
        mocker.patch.object(ff, "get_client", return_value=client)
        client.is_initialized.return_value = True
        client.variation_detail.return_value = EvaluationDetail(
            value, 0, {"kind": "FALLTHROUGH"}
        )
        if value == "failure":
            client.variation_detail.side_effect = RuntimeError("vendor failed")
    expected = (False, True) if value is False else (True, False)
    assert await ff._evaluate_boolean_flag(key, "test-user") == expected


@pytest.mark.parametrize("key", MATURE_FLAGS)
@pytest.mark.parametrize("prefix", ["FORCE_FLAG_", "NEXT_PUBLIC_FORCE_FLAG_"])
async def test_operator_false_wins_over_mature_default(
    key, prefix, monkeypatch, mocker
):
    monkeypatch.setenv(prefix + key.upper().replace("-", "_"), "false")
    vendor = mocker.patch.object(ff, "_evaluate_flag_value")
    assert await ff._evaluate_boolean_flag(key, "test-user") == (False, True)
    vendor.assert_not_called()


@pytest.mark.parametrize("flag", [Flag.SKILLS_HUB, Flag.ONBOARDING_BRAIN_DUMP])
async def test_route_dependency_and_decorator_share_default_and_explicit_override(
    flag, mocker
):
    mocker.patch.object(ff, "_flag_backend_initialized", return_value=False)
    mocker.patch.object(ff, "_flag_backend_configured", return_value=False)
    assert await ff.is_feature_enabled(flag, "test-user") is True
    await ff.create_feature_flag_dependency(flag)(None)

    @ff.feature_flag(flag.value)
    async def available(user_id):
        return "available"

    assert await available(user_id="test-user") == "available"
    with pytest.raises(ValueError, match="user_id"):
        await available(user_id=None)
    with pytest.raises(HTTPException) as dependency_off:
        await ff.create_feature_flag_dependency(flag, default=False)("test-user")
    assert dependency_off.value.status_code == 404

    @ff.feature_flag(flag.value, default=False)
    async def disabled(user_id):
        pytest.fail("disabled route executed")

    with pytest.raises(HTTPException) as decorator_off:
        await disabled(user_id="test-user")
    assert decorator_off.value.status_code == 404


async def test_frontend_omits_unavailable_values_but_serves_operator_false(
    mocker, monkeypatch
):
    mocker.patch.object(frontend, "_defined_flags", return_value=None)
    assert await frontend._read(frontend.FrontendFlag.HIRE_EXPERTS, "test-user") == (
        True,
        False,
    )
    assert frontend.FrontendFlag.HIRE_EXPERTS not in await frontend._evaluate(
        "test-user"
    )
    monkeypatch.setenv("FORCE_FLAG_HIRE_EXPERTS", "false")
    assert (await frontend._evaluate("test-user"))[
        frontend.FrontendFlag.HIRE_EXPERTS
    ] is False
