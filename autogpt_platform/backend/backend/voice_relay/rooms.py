"""A call's LiveKit room: its name, who may join it, and their tokens.

Every call on the LiveKit leg has a room of its own, ``voice-{call_id}``,
which LiveKit Cloud makes when the first participant joins. LiveKit keeps a
room open while anyone is in it, so the relay closes it (``room_reaper``).
Two participants join it and no one else can:

- the person's browser, as ``user:{user_id}``, on a token the REST API mints
  with the call's start, once the room is registered (:func:`person_token`).
  It may join that one room, publish only sources declared as a microphone,
  and subscribe; no data channel, no room admin. The token only has to let
  the browser in, so it is short, and never outlives the revocation that
  closing the room puts on the person's tokens.
- the relay, as Otto, on a token it mints for itself (:func:`relay_token`).
  It may publish Otto's speech and subscribe to the person's, for as long as
  the call can run.

Both are signed with the LiveKit project's API secret, which only the REST
API and the relay hold. The token helpers take the project itself, never its
keys on their own. The LiveKit server SDK is loaded the first time a token is
minted, so a process that mints none never loads it.
"""

from datetime import timedelta

from pydantic import BaseModel, ConfigDict, Field

from backend.util.settings import AppEnvironment, Settings
from backend.voice_relay.config import (
    PERSON_IDENTITY_PREFIX,
    RELAY_IDENTITY,
    ROOM_DEPARTURE_TIMEOUT_S,
    ROOM_EMPTY_TIMEOUT_S,
    ROOM_MAX_PARTICIPANTS,
    ROOM_PREFIX,
    ROOM_TOKEN_TTL_S,
)

# The one kind of track either participant may publish.
MICROPHONE = "microphone"


class LiveKitProject(BaseModel):
    """The LiveKit Cloud project the calls' rooms are in."""

    model_config = ConfigDict(frozen=True)

    url: str
    api_key: str = Field(repr=False)
    api_secret: str = Field(repr=False)


def livekit_project(settings: Settings) -> LiveKitProject | None:
    """This deployment's LiveKit project, or None while it has none: its URL
    must be a secure socket, or a plain one for a local install, and both of
    its keys set."""
    url = settings.config.livekit_url.strip().rstrip("/")
    secure = url.startswith("wss://")
    local = url.startswith("ws://") and settings.config.app_env == AppEnvironment.LOCAL
    keyed = bool(
        settings.secrets.livekit_api_key and settings.secrets.livekit_api_secret
    )
    if not (secure or local) or not keyed:
        return None
    return LiveKitProject(
        url=url,
        api_key=settings.secrets.livekit_api_key,
        api_secret=settings.secrets.livekit_api_secret,
    )


def room_name(call_id: str) -> str:
    """The LiveKit room of call *call_id*."""
    return f"{ROOM_PREFIX}{call_id}"


def person_identity(user_id: str) -> str:
    """The identity *user_id*'s browser joins their call's room under."""
    return f"{PERSON_IDENTITY_PREFIX}{user_id}"


def person_token(*, user_id: str, call_id: str, project: LiveKitProject) -> str:
    """The browser's token for call *call_id*'s room, for
    ``ROOM_TOKEN_TTL_S`` and never longer."""
    return _token(
        identity=person_identity(user_id),
        name="You",
        call_id=call_id,
        project=project,
        ttl_s=ROOM_TOKEN_TTL_S,
    )


def relay_token(*, call_id: str, project: LiveKitProject, ttl_s: float) -> str:
    """The relay's own token for call *call_id*'s room, as Otto."""
    return _token(
        identity=RELAY_IDENTITY,
        name="Otto",
        call_id=call_id,
        project=project,
        ttl_s=ttl_s,
    )


def _token(
    *,
    identity: str,
    name: str,
    call_id: str,
    project: LiveKitProject,
    ttl_s: float,
) -> str:
    from livekit import api

    grants = api.VideoGrants(
        room_join=True,
        room=room_name(call_id),
        can_publish=True,
        can_publish_sources=[MICROPHONE],
        can_subscribe=True,
        can_publish_data=False,
        can_update_own_metadata=False,
    )
    return (
        api.AccessToken(project.api_key, project.api_secret)
        .with_identity(identity)
        .with_name(name)
        .with_ttl(timedelta(seconds=ttl_s))
        .with_grants(grants)
        # Applied by whichever join makes the room: the browser's, as a rule.
        .with_room_config(
            api.RoomConfiguration(
                max_participants=ROOM_MAX_PARTICIPANTS,
                empty_timeout=ROOM_EMPTY_TIMEOUT_S,
                departure_timeout=ROOM_DEPARTURE_TIMEOUT_S,
            )
        )
        .to_jwt()
    )
