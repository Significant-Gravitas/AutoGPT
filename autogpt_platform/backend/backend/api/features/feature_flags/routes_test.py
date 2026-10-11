import os
from unittest.mock import AsyncMock

import fastapi
import httpx
import pytest
from autogpt_libs.auth import get_optional_user_id
from ldclient import Context, LDClient
from ldclient.config import Config as LDConfig
from ldclient.integrations.test_data import TestData

import backend.util.feature_flag as ff
from backend.util.feature_flag import (
    Flag,
    frontend,
    get_feature_flag_value,
    is_feature_enabled,
)
from backend.util.feature_flag.frontend import FrontendFlag, _anonymous_flags
from backend.util.settings import FeatureFlagBackend

from .routes import router

SEARCH_TERMS = ["Research", "Marketing"]
VISITOR = "0b6c1f3e-visitor-0001"
PINNING = FrontendFlag.CHAT_PINNING.value

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
    td.update(
        td.flag(PINNING)
        .boolean_flag()
        .variation_for_user(VISITOR, True)
        .variation_for_user("v" * 65, True)
        .variation_for_user("visitor:bad$id", True)
        .if_match("email_domain", "agpt.co")
        .then_return(True)
        .fallthrough_variation(False)
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
    frontend._visitor_flags_cache.clear()
    yield
    _anonymous_flags.cache_clear()
    frontend._visitor_flags_cache.clear()
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


async def test_a_consenting_visitor_is_bucketed_by_their_own_id():
    own = (await _get_flags(None, VISITOR))["flags"]
    shared = (await _get_flags(None))["flags"]

    assert own[PINNING] is True
    assert shared[PINNING] is False


async def test_a_signed_in_caller_is_never_bucketed_by_a_visitor_id():
    flags = (await _get_flags("u-other", VISITOR))["flags"]

    assert flags[PINNING] is False


@pytest.mark.parametrize("malformed", ["v" * 65, "visitor:bad$id"])
async def test_a_malformed_visitor_id_counts_as_none(malformed):
    flags = (await _get_flags(None, malformed))["flags"]

    assert flags[PINNING] is False


async def test_a_users_id_sent_as_a_visitor_id_is_never_looked_up_as_that_user(
    mocker,
):
    # The lookup would hand back an agpt.co user, whom the flag's rule turns on.
    employee = Context.builder("u-staff").set("email_domain", "agpt.co").build()
    lookup = mocker.patch(
        "backend.util.feature_flag._fetch_user_context_status",
        AsyncMock(return_value=(employee, True)),
    )

    flags = (await _get_flags(None, "3f2b8c1e-7a4d-4e9b-9c2a-5d6e7f8a9b0c"))["flags"]

    assert flags[PINNING] is False
    lookup.assert_not_awaited()


async def test_fresh_visitor_ids_cannot_grow_the_cache_past_its_bound(mocker):
    mocker.patch.object(frontend, "_evaluate", AsyncMock(return_value={}))

    for i in range(frontend.VISITOR_CACHE_SIZE + 10):
        await frontend.evaluate_frontend_flags(None, f"{i:08d}-visitor")

    assert len(frontend._visitor_flags_cache) <= frontend.VISITOR_CACHE_SIZE


async def _get_flags(user_id: str | None, visitor_id: str | None = None) -> dict:
    app.dependency_overrides[get_optional_user_id] = lambda: user_id
    params = {"visitor_id": visitor_id} if visitor_id else {}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/feature-flags", params=params)
    assert response.status_code == 200
    return response.json()
