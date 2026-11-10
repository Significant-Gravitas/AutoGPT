"""The LiveKit rooms of calls: closing each one, and sweeping for any left open.

LiveKit makes a call's room when the browser joins, and keeps a room someone
is in open for as long as they stay, handing their client fresh tokens. So
every room has an owner here from before it can exist to after it is gone:

- The REST API registers the room (``locks.register_room``) before it mints
  the browser's token, and only while a relay's sweep has run lately
  (``locks.rooms_swept``).
- The relay that runs the call closes the room once the call is done with it
  (:meth:`RoomReaper.close`): as the call ends, after Otto's play-out and the
  relay's own leave, and at once when the call's audio moves to the socket.
  A close removes the person, revoking every token of theirs issued until
  then, deletes the room, and stamps the close on the room's entry.
- Every relay with a LiveKit project sweeps the registry every
  ``ROOM_SWEEP_S`` against one listing of the project's rooms
  (:meth:`RoomReaper.sweep`). A room is left alone while it is in use: while
  its call may still start, while the call runs, and for ``ROOM_END_GRACE_S``
  after it ended. Any other room not yet closed is closed: a call that never
  started, a call whose pod died, a close that failed. A closed room listed
  again is closed again, and an entry is forgotten once its room has been
  closed longer than any token for it lives, and is not listed.

Closing is idempotent, so relays need no claim between them. A sweep touches
only the rooms the registry names: a listed ``voice-`` room with no entry is
another deployment's, or a leak, and is only counted.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from backend.data.redis_client import AsyncRedisClient
from backend.data.redis_helpers import as_str
from backend.voice_relay import locks, rooms
from backend.voice_relay.config import (
    ROOM_END_GRACE_S,
    ROOM_LEAVE_TIMEOUT_S,
    ROOM_LIST_TIMEOUT_S,
    ROOM_PREFIX,
    ROOM_REVOCATION_S,
    ROOM_START_GRACE_S,
    ROOM_SWEEP_S,
    RoomCloseWhy,
)
from backend.voice_relay.metering import (
    LIVEKIT_OLDEST_ROOM_SECONDS,
    LIVEKIT_PARTICIPANTS,
    LIVEKIT_ROOMS,
    LIVEKIT_ROOMS_CLOSED,
    LIVEKIT_SWEEPS,
)

if TYPE_CHECKING:
    from livekit import api

logger = logging.getLogger(__name__)


class RoomService(Protocol):
    """The calls the reaper makes on LiveKit's room service
    (``livekit.api.RoomService``); tests stand in for it."""

    async def remove_participant(
        self, remove: "api.RoomParticipantIdentity", /
    ) -> object: ...

    async def delete_room(self, delete: "api.DeleteRoomRequest", /) -> object: ...

    async def list_rooms(
        self, request: "api.ListRoomsRequest", /
    ) -> "api.ListRoomsResponse": ...


class RoomReaper:
    """Closes this deployment's call rooms, and sweeps for the ones left open.

    The room service is made the first time it is needed, from the LiveKit
    project, and closed with :meth:`aclose`.
    """

    def __init__(
        self,
        project: rooms.LiveKitProject,
        *,
        redis: Callable[[], Awaitable[AsyncRedisClient]],
        service: RoomService | None = None,
        clock: Callable[[], float] = time.time,
        sweep_s: float = ROOM_SWEEP_S,
    ) -> None:
        self._project = project
        self._redis = redis
        self._service = service
        self._api: "api.LiveKitAPI | None" = None
        self._clock = clock
        self._sweep_s = sweep_s
        # Closes under way, which a drain waits for.
        self.closing = 0

    def _rooms(self) -> RoomService:
        if self._service is None:
            from livekit import api

            self._api = api.LiveKitAPI(
                self._project.url, self._project.api_key, self._project.api_secret
            )
            self._service = self._api.room
        return self._service

    async def aclose(self) -> None:
        made, self._api = self._api, None
        if made is not None:
            self._service = None
            await made.aclose()

    # --- closing a room ----------------------------------------------------

    async def close(self, call_id: str, user_id: str, why: RoomCloseWhy) -> bool:
        """Close *call_id*'s room: remove *user_id* from it with every token
        of theirs issued until now revoked, delete it, and stamp the close.

        True once closed. A close that fails, or takes longer than
        ``ROOM_LEAVE_TIMEOUT_S``, is logged and left to the sweep.
        """
        self.closing += 1
        try:
            await asyncio.wait_for(
                self._close(call_id, user_id), timeout=ROOM_LEAVE_TIMEOUT_S
            )
        except Exception as error:
            LIVEKIT_ROOMS_CLOSED.labels(why, "failed").inc()
            logger.warning(
                f"Voice call {call_id}: its room was not closed ({why}, "
                f"{type(error).__name__})"
            )
            return False
        finally:
            self.closing -= 1
        LIVEKIT_ROOMS_CLOSED.labels(why, "closed").inc()
        logger.info(f"Voice call {call_id}: its room is closed ({why})")
        return True

    async def _close(self, call_id: str, user_id: str) -> None:
        from livekit import api

        service = self._rooms()
        room = rooms.room_name(call_id)
        if user_id:
            await _gone_counts_as_done(
                service.remove_participant(
                    api.RoomParticipantIdentity(
                        room=room,
                        identity=rooms.person_identity(user_id),
                        revoke_token_ts=int(self._clock()),
                    )
                )
            )
        await _gone_counts_as_done(
            service.delete_room(api.DeleteRoomRequest(room=room))
        )
        await locks.mark_room_closed(await self._redis(), call_id, self._clock())

    # --- the sweep ---------------------------------------------------------

    async def sweep_forever(self) -> None:
        while True:
            await self.sweep()
            await asyncio.sleep(self._sweep_s)

    async def sweep(self) -> bool:
        """Close every registered room no longer in use, against one listing
        of the project's rooms, and write the heartbeat.

        False, with no heartbeat, when the registry or the listing could not
        be read.
        """
        from livekit import api

        try:
            redis = await self._redis()
            entries = await locks.registered_rooms(redis)
            listing = await asyncio.wait_for(
                self._rooms().list_rooms(api.ListRoomsRequest()),
                timeout=ROOM_LIST_TIMEOUT_S,
            )
            now = self._clock()
            await locks.mark_rooms_swept(redis, now)
        except Exception:
            LIVEKIT_SWEEPS.labels("failed").inc()
            logger.warning("Voice rooms: the sweep could not run", exc_info=True)
            return False
        listed = {room.name: room for room in listing.rooms}
        ours = [room for name, room in listed.items() if _call_id_of(name) in entries]
        LIVEKIT_ROOMS.labels("ours").set(len(ours))
        LIVEKIT_ROOMS.labels("other").set(
            sum(
                1
                for name in listed
                if name.startswith(ROOM_PREFIX) and _call_id_of(name) not in entries
            )
        )
        LIVEKIT_PARTICIPANTS.set(sum(room.num_participants for room in listing.rooms))
        LIVEKIT_OLDEST_ROOM_SECONDS.set(
            max((now - room.creation_time for room in ours), default=0.0)
        )
        for entry in entries.values():
            try:
                await self._settle(
                    redis, entry, rooms.room_name(entry.call_id) in listed, now
                )
            except Exception:
                logger.warning(
                    f"Voice call {entry.call_id}: its room could not be swept",
                    exc_info=True,
                )
        LIVEKIT_SWEEPS.labels("ok").inc()
        return True

    async def _settle(
        self,
        redis: AsyncRedisClient,
        entry: locks.RoomEntry,
        is_listed: bool,
        now: float,
    ) -> None:
        if entry.closed_at is not None:
            if is_listed:
                await self.close(entry.call_id, entry.uid, "reopened")
            elif now - entry.closed_at > ROOM_REVOCATION_S:
                await locks.forget_room(redis, entry.call_id)
            return
        why = await _unused(redis, entry, now)
        if why is not None:
            await self.close(entry.call_id, entry.uid, why)


async def _unused(
    redis: AsyncRedisClient, entry: locks.RoomEntry, now: float
) -> RoomCloseWhy | None:
    """Why *entry*'s room is no longer in use, or None while it is: while
    its call may still start, while the call runs, and for a while after it
    ended."""
    status, ended_at = await redis.hmget(
        locks.call_record_key(entry.call_id), ["status", "ended_at"]
    )
    if status is None and ended_at is None:
        if now - entry.minted_at < ROOM_START_GRACE_S:
            return None
        return "never_started"
    if as_str(status) in locks.RUNNING and await locks.is_live(
        redis, entry.call_id, now
    ):
        return None
    ended = _seconds(ended_at)
    if ended is not None and now - ended < ROOM_END_GRACE_S:
        return None
    return "orphaned"


async def _gone_counts_as_done(call: Awaitable[object]) -> None:
    """*call*, a room service request; a room or a participant that is not
    there is as good as removed."""
    from livekit import api

    try:
        await call
    except api.TwirpError as error:
        if error.code != api.TwirpErrorCode.NOT_FOUND:
            raise


def _call_id_of(room_name: str) -> str | None:
    if not room_name.startswith(ROOM_PREFIX):
        return None
    return room_name.removeprefix(ROOM_PREFIX)


def _seconds(value: bytes | str | None) -> float | None:
    try:
        return float(as_str(value) or "")
    except ValueError:
        return None
