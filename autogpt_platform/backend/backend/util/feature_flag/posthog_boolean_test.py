import json
import os

import pytest
from ldclient import Context
from posthog import Posthog
from posthog.request import _flags_session
from posthog.types import normalize_flags_response
from requests import Response

import backend.util.feature_flag as ff
from backend.util.feature_flag import Flag, frontend
from backend.util.settings import FeatureFlagBackend


@pytest.fixture
def remote_flags(mocker, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("FORCE_FLAG_", "NEXT_PUBLIC_FORCE_FLAG_")):
            monkeypatch.delenv(name)
    client = Posthog("phc-fake", enable_local_evaluation=False)
    mocker.patch.object(ff.posthog, "get_flag_client", return_value=client)
    mocker.patch.object(ff.posthog.settings.secrets, "posthog_api_key", "phc-fake")
    mocker.patch.object(ff.posthog.settings.secrets, "posthog_secret_key", "")
    mocker.patch.object(
        ff.settings.config, "feature_flag_backend", FeatureFlagBackend.POSTHOG
    )
    mocker.patch.object(
        ff, "_fetch_user_context_status", return_value=(Context.create("u-1"), True)
    )
    response = Response()
    response.status_code = 200
    request = mocker.patch.object(_flags_session, "post", return_value=response)
    yield client, request, response
    client.shutdown()


@pytest.mark.parametrize(
    "body, status, expected",
    [
        ({"flags": {}, "errorsWhileComputingFlags": False}, 200, (False, True)),
        ({"featureFlags": {}, "errorsWhileComputingFlags": False}, 200, (False, True)),
        ({"flags": {}, "errorsWhileComputingFlags": True}, 200, (True, False)),
        ({}, 200, (True, False)),
        ({"detail": "unavailable"}, 500, (True, False)),
        (
            {
                "featureFlags": {"hire-experts": False},
                "errorsWhileComputingFlags": False,
            },
            200,
            (False, True),
        ),
        (
            {
                "featureFlags": {"hire-experts": True},
                "errorsWhileComputingFlags": False,
            },
            200,
            (True, True),
        ),
    ],
)
async def test_real_sdk_remote_boolean_response(remote_flags, body, status, expected):
    _, request, response = remote_flags
    response.status_code = status
    response._content = json.dumps(body).encode()
    assert await ff.evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == expected
    request.assert_called_once()
    sent = json.loads(request.call_args.kwargs["data"])
    assert sent["flag_keys_to_evaluate"] == ["hire-experts"]


@pytest.mark.parametrize(
    "flag, default",
    [
        (Flag.COPILOT_SDK, True),
        (Flag.COPILOT_SKILLS, True),
        (Flag.DREAM_PASS_ENABLED, False),
    ],
)
async def test_unselected_missing_flags_keep_caller_defaults_and_authority(
    remote_flags, flag, default
):
    _, request, response = remote_flags
    response._content = json.dumps(
        {"flags": {}, "errorsWhileComputingFlags": False}
    ).encode()
    assert await ff.evaluate_feature_flag(flag, "u-1", default=default) == (
        default,
        False,
    )
    request.assert_called_once()


async def test_remote_only_read_does_not_warn_about_local_definitions(
    remote_flags, caplog
):
    _, _, response = remote_flags
    response._content = json.dumps(
        {"flags": {}, "errorsWhileComputingFlags": False}
    ).encode()
    assert await ff.evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, True)
    assert "Local evaluation called" not in caplog.text


async def test_frontend_gets_inactive_remote_flag_as_off(remote_flags):
    _, request, response = remote_flags
    response._content = json.dumps(
        {"flags": {}, "errorsWhileComputingFlags": False}
    ).encode()
    frontend._defined_flags.cache_clear()
    try:
        assert ff.posthog.defined_flag_keys() is None
        values = (await frontend.evaluate_frontend_flags("u-1")).model_dump(mode="json")
        assert values["configured"] is True
        assert values["flags"]["hire-experts"] is False
        assert values["flags"]["onboarding-brain-dump"] is False
        assert "dream-pass-enabled" not in values["flags"]
        assert request.call_count == len(frontend.FrontendFlag)
    finally:
        frontend._defined_flags.cache_clear()


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"flags": {}, "errorsWhileComputingFlags": False}, (False, True)),
        ({"flags": {}, "errorsWhileComputingFlags": True}, (True, False)),
        (
            {
                "featureFlags": {"hire-experts": True},
                "errorsWhileComputingFlags": False,
            },
            (True, True),
        ),
    ],
)
async def test_frontend_matches_route_gate_when_mature_flag_missing_locally(
    remote_flags, body, expected
):
    client, _, response = remote_flags
    client.feature_flags = [
        {"id": 1, "key": "unrelated-flag", "active": False, "filters": {"groups": []}}
    ]
    response._content = json.dumps(body).encode()
    frontend._defined_flags.cache_clear()
    try:
        assert ff.posthog.defined_flag_keys() == frozenset({"unrelated-flag"})
        assert await ff.evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == expected
        values = (await frontend.evaluate_frontend_flags("u-1")).model_dump(mode="json")
        assert values["configured"] is True
        flags = values["flags"]
        if expected[1]:
            assert flags["hire-experts"] is expected[0]
        else:
            assert "hire-experts" not in flags
        assert flags["dream-pass-enabled"] is None
        assert flags["marketplace-search-terms"] is None
    finally:
        frontend._defined_flags.cache_clear()


@pytest.mark.parametrize("forced", [False, True])
async def test_frontend_honors_env_for_mature_flag_missing_locally(
    remote_flags, monkeypatch, forced
):
    client, request, response = remote_flags
    client.feature_flags = [
        {"id": 1, "key": "unrelated-flag", "active": False, "filters": {"groups": []}}
    ]
    response._content = json.dumps(
        {"flags": {}, "errorsWhileComputingFlags": False}
    ).encode()
    monkeypatch.setenv("FORCE_FLAG_HIRE_EXPERTS", str(forced).lower())
    frontend._defined_flags.cache_clear()
    try:
        assert await ff.evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (
            forced,
            True,
        )
        values = await frontend.evaluate_frontend_flags("u-1")
        assert values.flags[frontend.FrontendFlag.HIRE_EXPERTS] is forced
        assert all(
            "hire-experts"
            not in json.loads(call.kwargs["data"])["flag_keys_to_evaluate"]
            for call in request.call_args_list
        )
    finally:
        frontend._defined_flags.cache_clear()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("completeness", [{"errorsWhileComputingFlags": True}, {}])
async def test_present_boolean_survives_partial_or_legacy_response(
    remote_flags, enabled, completeness
):
    _, request, response = remote_flags
    response._content = json.dumps(
        {"featureFlags": {"hire-experts": enabled}, **completeness}
    ).encode()
    assert await ff.evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (enabled, True)
    request.assert_called_once()


@pytest.mark.parametrize("enabled", [None, "false"])
async def test_malformed_boolean_is_not_an_authoritative_off(remote_flags, enabled):
    _, request, response = remote_flags
    response._content = json.dumps(
        {
            "flags": {"hire-experts": {"key": "hire-experts", "enabled": enabled}},
            "errorsWhileComputingFlags": False,
        }
    ).encode()
    assert await ff.evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (True, False)
    request.assert_called_once()


async def test_inactive_local_definition_needs_no_remote_request(remote_flags):
    client, request, _ = remote_flags
    client.feature_flags = [
        {
            "id": 1,
            "key": "hire-experts",
            "active": False,
            "filters": {"groups": [{"properties": [], "rollout_percentage": 100}]},
        }
    ]
    assert await ff.evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, True)
    request.assert_not_called()


async def test_first_read_loads_local_definitions_before_remote_fallback(
    remote_flags, mocker
):
    client, request, _ = remote_flags
    client.enable_local_evaluation = True
    client.personal_api_key = "phx-fake"

    def load_definitions():
        client.feature_flags = [
            {
                "id": 1,
                "key": "hire-experts",
                "active": False,
                "filters": {"groups": []},
            }
        ]

    load = mocker.patch.object(
        client, "load_feature_flags", side_effect=load_definitions
    )
    assert client.feature_flags is None
    assert await ff.evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, True)
    load.assert_called_once()
    request.assert_not_called()


def test_sdk_failure_normalization_does_not_claim_complete_evaluation():
    assert "errorsWhileComputingFlags" not in normalize_flags_response({})
