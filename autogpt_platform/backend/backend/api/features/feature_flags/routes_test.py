import os

import fastapi
import httpx
import pytest
from autogpt_libs.auth import get_optional_user_id
from ldclient import LDClient
from ldclient.config import Config as LDConfig
from ldclient.integrations.test_data import TestData

import backend.util.feature_flag as ff
from backend.util.feature_flag import Flag, get_feature_flag_value, is_feature_enabled
from backend.util.feature_flag.frontend import FrontendFlag, _anonymous_flags
from backend.util.settings import FeatureFlagBackend

from .routes import router

SEARCH_TERMS = ["Research", "Marketing"]

app = fastapi.FastAPI()
app.include_router(router, prefix="/api")


@pytest.fixture(autouse=True)
def launchdarkly(mocker, monkeypatch: pytest.MonkeyPatch):
    """A real LaunchDarkly client serving fixed targeting, and no env overrides."""
    for name in list(os.environ):
        if name.startswith(
            ("FORCE_FLAG_", "NEXT_PUBLIC_FORCE_FLAG_", "FORCE_ALL_FLAGS")
        ):
            monkeypatch.delenv(name)
    td = TestData.data_source()
    td.update(
        td.flag(FrontendFlag.COPILOT_VOICE_MODE.value)
        .boolean_flag()
        .variation_for_user("u-staff", True)
        .fallthrough_variation(False)
    )
    td.update(
        td.flag(FrontendFlag.HIRE_EXPERTS.value).boolean_flag().variation_for_all(True)
    )
    td.update(
        td.flag(FrontendFlag.MARKETPLACE_SEARCH_TERMS.value)
        .variations(SEARCH_TERMS)
        .fallthrough_variation(0)
    )
    client = LDClient(
        LDConfig("sdk-test", update_processor_class=td, send_events=False)
    )
    mocker.patch.object(
        ff.settings.config, "feature_flag_backend", FeatureFlagBackend.LAUNCHDARKLY
    )
    mocker.patch.object(ff.settings.secrets, "launch_darkly_sdk_key", "sdk-test")
    monkeypatch.setattr(ff, "_init_attempted", True)
    mocker.patch("backend.util.feature_flag.ldclient.get", return_value=client)
    _anonymous_flags.cache_clear()
    yield
    _anonymous_flags.cache_clear()
    app.dependency_overrides.clear()
    client.close()


@pytest.mark.parametrize("caller", ["u-staff", "u-other", None])
async def test_each_flag_answers_what_the_backends_own_read_answers(caller):
    body = await _get_flags(caller)

    as_backend = caller or "anonymous"
    assert body["configured"] is True
    assert body["flags"][
        FrontendFlag.COPILOT_VOICE_MODE.value
    ] is await is_feature_enabled(Flag.COPILOT_VOICE_MODE, as_backend)
    assert body["flags"][
        FrontendFlag.MARKETPLACE_SEARCH_TERMS.value
    ] == await get_feature_flag_value(
        FrontendFlag.MARKETPLACE_SEARCH_TERMS.value, as_backend
    )


async def test_a_flag_the_vendor_did_not_answer_is_left_out_not_defaulted():
    flags = (await _get_flags("u-staff"))["flags"]

    # A default here would read as an answer and 404 a gated route for a user who has it.
    assert FrontendFlag.CHAT_SEARCH.value not in flags
    assert FrontendFlag.COPILOT_STREAM_SHADOW.value not in flags
    assert flags[FrontendFlag.HIRE_EXPERTS.value] is True


async def test_with_no_vendor_configured_every_flag_is_the_frontends_default(mocker):
    mocker.patch.object(ff.settings.secrets, "launch_darkly_sdk_key", "")

    body = await _get_flags("u-staff")

    assert body["configured"] is False


async def test_an_env_override_forces_a_switch_but_never_a_payload(monkeypatch):
    monkeypatch.setenv("FORCE_FLAG_COPILOT_VOICE_MODE", "true")
    monkeypatch.setenv("FORCE_FLAG_MARKETPLACE_SEARCH_TERMS", "true")

    flags = (await _get_flags("u-other"))["flags"]

    assert flags[FrontendFlag.COPILOT_VOICE_MODE.value] is True
    assert flags[FrontendFlag.MARKETPLACE_SEARCH_TERMS.value] == SEARCH_TERMS


async def _get_flags(user_id: str | None) -> dict:
    app.dependency_overrides[get_optional_user_id] = lambda: user_id
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/feature-flags")
    assert response.status_code == 200
    return response.json()
