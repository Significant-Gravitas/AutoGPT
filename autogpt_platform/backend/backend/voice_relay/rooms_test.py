"""A call's LiveKit room: who may join it, with which grants, for how long,
and when a deployment has a LiveKit project at all."""

import json
import subprocess
import sys
from pathlib import Path

import jwt
import pytest
from livekit import api
from livekit.api.access_token import Claims

from backend.util.settings import AppEnvironment, Settings
from backend.voice_relay import rooms
from backend.voice_relay.config import (
    CAPS,
    RELAY_ROOM_GRACE_S,
    ROOM_REVOCATION_S,
    ROOM_TOKEN_TTL_S,
)

CALL_ID = "c" * 32
KEY = "APIkey"
SECRET = "s" * 48  # pragma: allowlist secret
PROJECT = rooms.LiveKitProject(
    url="wss://agpt.livekit.cloud", api_key=KEY, api_secret=SECRET
)


def _claims(token: str) -> Claims:
    return api.TokenVerifier(KEY, SECRET).verify(token)


def _lifetime_s(token: str) -> int:
    # The SDK's claims leave the token's times out; the JWT carries both.
    decoded = jwt.decode(token, SECRET, algorithms=["HS256"], issuer=KEY)
    return decoded["exp"] - decoded["nbf"]


def test_the_browser_may_join_only_its_calls_room_publish_its_microphone_and_listen():
    token = rooms.person_token(user_id="user-1", call_id=CALL_ID, project=PROJECT)
    claims = _claims(token)
    assert claims.identity == "user:user-1"
    video = claims.video
    assert video is not None
    assert video.room == f"voice-{CALL_ID}"
    assert video.room_join is True
    assert video.can_publish is True
    assert video.can_publish_sources == ["microphone"]
    assert video.can_subscribe is True
    assert video.can_publish_data is False
    assert video.can_update_own_metadata is False
    # No grant over the room itself, nor any other room.
    assert not video.room_admin
    assert not video.room_create
    assert not video.room_list
    assert not video.room_record
    assert not video.hidden
    assert _lifetime_s(token) == 600


def test_the_persons_token_never_outlives_the_revocation_closing_the_room_puts_on_it():
    # Closing a room revokes the person's tokens for ten minutes; a token that
    # lived longer could open the room again after that.
    assert ROOM_REVOCATION_S == 600
    assert ROOM_TOKEN_TTL_S <= ROOM_REVOCATION_S
    token = rooms.person_token(user_id="user-1", call_id=CALL_ID, project=PROJECT)
    assert _lifetime_s(token) <= ROOM_REVOCATION_S


def test_the_relay_joins_the_same_room_as_otto_for_as_long_as_the_call_can_run():
    ttl_s = CAPS.max_call_s + RELAY_ROOM_GRACE_S
    token = rooms.relay_token(call_id=CALL_ID, project=PROJECT, ttl_s=ttl_s)
    claims = _claims(token)
    assert claims.identity == "otto"
    video = claims.video
    assert video is not None
    assert video.room == f"voice-{CALL_ID}"
    assert video.room_join is True
    assert video.can_publish is True
    assert video.can_subscribe is True
    assert video.can_publish_data is False
    assert not video.room_admin
    assert _lifetime_s(token) == ttl_s == 360


@pytest.mark.parametrize("mint", ["person", "relay"])
def test_whichever_join_makes_the_room_makes_it_for_two_and_short_lived(mint):
    if mint == "person":
        token = rooms.person_token(user_id="user-1", call_id=CALL_ID, project=PROJECT)
    else:
        token = rooms.relay_token(call_id=CALL_ID, project=PROJECT, ttl_s=360)
    config = _claims(token).room_config
    assert config is not None
    assert config.max_participants == 2
    assert config.empty_timeout == 60
    assert config.departure_timeout == 20


def test_a_token_signed_with_another_secret_is_not_the_projects():
    other = PROJECT.model_copy(update={"api_secret": "t" * 48})
    token = rooms.person_token(user_id="user-1", call_id=CALL_ID, project=other)
    with pytest.raises(jwt.InvalidSignatureError):
        _claims(token)


def _settings(
    monkeypatch, *, url: str, key: str = KEY, secret: str = SECRET, env=None
) -> Settings:
    settings = Settings()
    monkeypatch.setattr(settings.config, "livekit_url", url)
    monkeypatch.setattr(settings.config, "app_env", env or AppEnvironment.PRODUCTION)
    monkeypatch.setattr(settings.secrets, "livekit_api_key", key)
    monkeypatch.setattr(settings.secrets, "livekit_api_secret", secret)
    return settings


def test_a_deployment_with_a_secure_url_and_both_keys_has_a_project(monkeypatch):
    project = rooms.livekit_project(
        _settings(monkeypatch, url=" wss://agpt.livekit.cloud/ ")
    )
    assert project is not None
    assert project.url == "wss://agpt.livekit.cloud"
    assert project.api_key == KEY
    # Its keys stay out of anything that prints it.
    assert SECRET not in repr(project) and KEY not in repr(project)


@pytest.mark.parametrize(
    "url,key,secret",
    [
        ("", KEY, SECRET),
        ("wss://agpt.livekit.cloud", "", SECRET),
        ("wss://agpt.livekit.cloud", KEY, ""),
        ("ws://agpt.livekit.cloud", KEY, SECRET),
        ("https://agpt.livekit.cloud", KEY, SECRET),
    ],
    ids=["no-url", "no-key", "no-secret", "plain-socket", "not-a-socket"],
)
def test_a_deployment_missing_any_part_of_its_project_has_none(
    monkeypatch, url, key, secret
):
    settings = _settings(monkeypatch, url=url, key=key, secret=secret)
    assert rooms.livekit_project(settings) is None


def test_a_local_install_may_use_a_plain_socket(monkeypatch):
    settings = _settings(
        monkeypatch, url="ws://localhost:7880", env=AppEnvironment.LOCAL
    )
    project = rooms.livekit_project(settings)
    assert project is not None and project.url == "ws://localhost:7880"


LOAD_PROBE = """
import json, sys
import backend.api.features.onboarding_dump.routes
from backend.voice_relay import rooms
def livekit():
    return sorted(m for m in sys.modules if m == "livekit" or m.startswith("livekit."))
print("AT_LOAD=" + json.dumps(livekit()))
print("REAPER=" + json.dumps("backend.voice_relay.room_reaper" in sys.modules))
rooms.relay_token(
    call_id="c" * 32,
    project=rooms.LiveKitProject(url="wss://x.livekit.cloud", api_key="k", api_secret="s" * 48),
    ttl_s=60,
)
print("AFTER_MINT=" + json.dumps("livekit.api" in livekit()))
"""


def test_the_rest_api_loads_no_room_reaper_and_the_livekit_sdk_only_once_it_mints():
    run = subprocess.run(
        [sys.executable, "-c", LOAD_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert run.returncode == 0, run.stderr[-2000:]
    reported = dict(
        line.split("=", 1)
        for line in run.stdout.splitlines()
        if line.startswith(("AT_LOAD=", "REAPER=", "AFTER_MINT="))
    )
    assert json.loads(reported["AT_LOAD"]) == []
    # The REST API only registers rooms: the relay alone closes them.
    assert json.loads(reported["REAPER"]) is False
    assert json.loads(reported["AFTER_MINT"]) is True
