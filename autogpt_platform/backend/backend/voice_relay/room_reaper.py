"""The LiveKit rooms of calls: closing each one, and sweeping for any left open.

LiveKit makes a call's room when the browser joins, and keeps a room someone
is in open for as long as they stay, handing their client fresh tokens. So
every room has an owner here from before it can exist to after it is gone:

- The REST API registers the room (``locks.register_room``) before it mints
  the browser's token, and only while a relay's sweep has lately run and
  closed all it had to (``locks.rooms_swept``).
- The relay that runs the call closes the room once the call is done with it
  (:meth:`RoomReaper.close`): as the call ends, after Otto's play-out and the
  relay's own leave, and at once when the call's audio moves to the socket.
  A close removes the person, revoking every token of theirs issued until a
  moment after it (``ROOM_REVOKE_AHEAD_S``), and deletes the room, even when
  the removal failed; only a close that did both is stamped on the room's
  entry.
- Every relay with a LiveKit project sweeps the registry every
  ``ROOM_SWEEP_S`` against one listing of the project's rooms
  (:meth:`RoomReaper.sweep`). A room is left alone while it is in use: while
  its call may still start, while the call runs, and for ``ROOM_END_GRACE_S``
  after it ended. Any other room not yet closed is closed: a call that never
  started, a call whose pod died, a close that failed. A closed room listed
  again is closed again.
- A removal revokes the person's tokens for ``ROOM_REVOCATION_S``, but
  LiveKit still takes a token for ``ROOM_TOKEN_LEEWAY_S`` past its expiry,
  so one issued just before the close can outlast the revocation. The sweep
  revokes the person's tokens once more ``ROOM_REVOKE_AGAIN_AFTER_S`` after
  the close, before the first revocation lapses, and forgets the entry only
  once its room is not listed and ``ROOM_FORGET_AFTER_REVOKED_S`` have passed
  since the second revocation.
- Where the LiveKit project is this deployment's alone (``exclusive``), a
  listed ``voice-`` room with no entry is a leak, whatever made it: once it
  is older than ``ROOM_UNREGISTERED_GRACE_S``, longer than any call's room
  lives, the sweep enters it in the registry under the person in it and
  closes it like any other. On a project other deployments share, such a
  room is theirs, and is only counted.

Closing is idempotent, so relays need no claim between them. A sweep writes
the heartbeat only when it read the registry and the listing, and every
close and revocation it made went through: while they keep failing, the REST
API puts calls on the socket.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, TypeVar

from backend.data.redis_client import AsyncRedisClient
from backend.data.redis_helpers import as_str
from backend.voice_relay import locks, rooms
from backend.voice_relay.config import (
    PERSON_IDENTITY_PREFIX,
    ROOM_END_GRACE_S,
    ROOM_FORGET_AFTER_REVOKED_S,
    ROOM_LEAVE_TIMEOUT_S,
    ROOM_LIST_TIMEOUT_S,
    ROOM_PREFIX,
    ROOM_REVOKE_AGAIN_AFTER_S,
    ROOM_REVOKE_AHEAD_S,
    ROOM_START_GRACE_S,
    ROOM_SWEEP_S,
    ROOM_UNREGISTERED_GRACE_S,
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

T = TypeVar("T")


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

    async def list_participants(
        self, request: "api.ListParticipantsRequest", /
    ) -> "api.ListParticipantsResponse": ...


class RoomReaper:
    """Closes this deployment's call rooms, and sweeps for the ones left open.

    The room service is made the first time it is needed, from the LiveKit
    project, and closed with :meth:`aclose`. With *exclusive*, the project is
    this deployment's alone, and the sweep also closes the ``voice-`` rooms
    the registry has no entry for.
    """

    def __init__(
        self,
        project: rooms.LiveKitProject,
        *,
        redis: Callable[[], Awaitable[AsyncRedisClient]],
        service: RoomService | None = None,
        clock: Callable[[], float] = time.time,
        sweep_s: float = ROOM_SWEEP_S,
        exclusive: bool = False,
        unregistered_grace_s: float = ROOM_UNREGISTERED_GRACE_S,
    ) -> None:
        self._project = project
        self._redis = redis
        self._service = service
        self._api: "api.LiveKitAPI | None" = None
        self._clock = clock
        self._sweep_s = sweep_s
        self._exclusive = exclusive
        self._unregistered_grace_s = unregistered_grace_s
        # Closes and revocations under way, which a drain waits for.
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
        """Close *call_id*'s room: remove *user_id* from it, revoking every
        token of theirs issued until a moment from now, delete the room, and
        stamp the close.

        True once both the removal and the deletion went through, each within
        ``ROOM_LEAVE_TIMEOUT_S``. The room is deleted even when the removal
        failed; a close that did not do both is logged, left unstamped, and
        left to the sweep.
        """
        self.closing += 1
        try:
            await self._close(call_id, user_id)
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
        removal: Exception | None = None
        if user_id:
            try:
                await self._revoke(service, room, user_id)
            except Exception as error:
                removal = error
        await _request(service.delete_room(api.DeleteRoomRequest(room=room)))
        if removal is not None:
            raise removal
        redis = await self._redis()
        await _bounded(locks.mark_room_closed(redis, call_id, user_id, self._clock()))

    async def revoke_again(self, call_id: str, user_id: str, closed_at: float) -> bool:
        """Revoke *user_id*'s tokens for *call_id*'s room, closed at
        *closed_at*, once more before the close's revocation lapses, and stamp
        it. True once done."""
        self.closing += 1
        try:
            if user_id:
                await self._revoke(self._rooms(), rooms.room_name(call_id), user_id)
            redis = await self._redis()
            await _bounded(
                locks.mark_room_revoked_again(
                    redis,
                    call_id,
                    user_id,
                    closed_at=closed_at,
                    revoked_at=self._clock(),
                )
            )
        except Exception as error:
            LIVEKIT_ROOMS_CLOSED.labels("revoke_again", "failed").inc()
            logger.warning(
                f"Voice call {call_id}: its person's tokens were not revoked "
                f"again ({type(error).__name__})"
            )
            return False
        finally:
            self.closing -= 1
        LIVEKIT_ROOMS_CLOSED.labels("revoke_again", "closed").inc()
        return True

    async def _revoke(self, service: RoomService, room: str, user_id: str) -> None:
        """Remove *user_id* from *room*, with every token of theirs issued
        until a moment from now revoked: a little ahead of now, so a token
        issued in this same second is caught too."""
        from livekit import api

        await _request(
            service.remove_participant(
                api.RoomParticipantIdentity(
                    room=room,
                    identity=rooms.person_identity(user_id),
                    revoke_token_ts=int(self._clock()) + ROOM_REVOKE_AHEAD_S,
                )
            )
        )

    # --- the sweep ---------------------------------------------------------

    async def sweep_forever(self) -> None:
        while True:
            await self.sweep()
            await asyncio.sleep(self._sweep_s)

    async def sweep(self) -> bool:
        """Settle every registered room against one listing of the project's
        rooms, adopt the unregistered ones where the project is exclusive,
        and write the heartbeat.

        False, with no heartbeat, when the registry or the listing could not
        be read, or a close or a revocation the sweep made failed.
        """
        from livekit import api

        try:
            redis = await self._redis()
            entries = await locks.registered_rooms(redis)
            listing = await asyncio.wait_for(
                self._rooms().list_rooms(api.ListRoomsRequest()),
                timeout=ROOM_LIST_TIMEOUT_S,
            )
        except Exception:
            LIVEKIT_SWEEPS.labels("failed").inc()
            logger.warning("Voice rooms: the sweep could not run", exc_info=True)
            return False
        now = self._clock()
        listed = {room.name: room for room in listing.rooms}
        ours = [room for name, room in listed.items() if _call_id_of(name) in entries]
        unregistered = [
            room
            for name, room in listed.items()
            if name.startswith(ROOM_PREFIX) and _call_id_of(name) not in entries
        ]
        LIVEKIT_ROOMS.labels("ours").set(len(ours))
        LIVEKIT_ROOMS.labels("other").set(len(unregistered))
        LIVEKIT_PARTICIPANTS.set(sum(room.num_participants for room in listing.rooms))
        LIVEKIT_OLDEST_ROOM_SECONDS.set(
            max((now - room.creation_time for room in ours), default=0.0)
        )
        settled = True
        for entry in entries.values():
            try:
                done = await self._settle(
                    redis, entry, rooms.room_name(entry.call_id) in listed, now
                )
            except Exception:
                logger.warning(
                    f"Voice call {entry.call_id}: its room could not be swept",
                    exc_info=True,
                )
                done = False
            settled = settled and done
        if self._exclusive:
            for room in unregistered:
                if now - room.creation_time > self._unregistered_grace_s:
                    settled = await self._adopt(redis, room) and settled
        if not settled:
            LIVEKIT_SWEEPS.labels("failed").inc()
            logger.warning(
                "Voice rooms: a close or a revocation failed, so this sweep "
                "writes no heartbeat"
            )
            return False
        try:
            await _bounded(locks.mark_rooms_swept(redis, now))
        except Exception:
            LIVEKIT_SWEEPS.labels("failed").inc()
            logger.warning("Voice rooms: the heartbeat was not written", exc_info=True)
            return False
        LIVEKIT_SWEEPS.labels("ok").inc()
        return True

    async def _settle(
        self,
        redis: AsyncRedisClient,
        entry: locks.RoomEntry,
        is_listed: bool,
        now: float,
    ) -> bool:
        """Close, revoke again or forget *entry*'s room, as far as its state
        asks. False when a close or a revocation failed."""
        closed_at = entry.closed_at
        if closed_at is None:
            why = await _unused(redis, entry, now)
            return why is None or await self.close(entry.call_id, entry.uid, why)
        if is_listed:
            return await self.close(entry.call_id, entry.uid, "reopened")
        if entry.revoked_again_at is None:
            if now - closed_at < ROOM_REVOKE_AGAIN_AFTER_S:
                return True
            return await self.revoke_again(entry.call_id, entry.uid, closed_at)
        if now - entry.revoked_again_at > ROOM_FORGET_AFTER_REVOKED_S:
            await _bounded(locks.forget_room(redis, entry.call_id))
        return True

    async def _adopt(self, redis: AsyncRedisClient, room: "api.Room") -> bool:
        """Enter a ``voice-`` room the registry has no entry for under the
        person in it, as made when the room was, and close it. False when its
        participants could not be listed or its close failed; a later sweep
        tries again."""
        from livekit import api

        call_id = _call_id_of(room.name) or ""
        try:
            try:
                listing = await asyncio.wait_for(
                    self._rooms().list_participants(
                        api.ListParticipantsRequest(room=room.name)
                    ),
                    timeout=ROOM_LIST_TIMEOUT_S,
                )
            except api.TwirpError as error:
                if error.code == api.TwirpErrorCode.NOT_FOUND:
                    # Gone since the listing: nothing is left to close.
                    return True
                raise
            uid = next(
                (
                    participant.identity.removeprefix(PERSON_IDENTITY_PREFIX)
                    for participant in listing.participants
                    if participant.identity.startswith(PERSON_IDENTITY_PREFIX)
                ),
                "",
            )
            await _bounded(
                locks.register_room(redis, call_id, uid, float(room.creation_time))
            )
        except Exception:
            logger.warning(
                f"Voice rooms: unregistered room {room.name} could not be adopted",
                exc_info=True,
            )
            return False
        logger.warning(
            f"Voice rooms: room {room.name} had no entry; adopted, and closed"
        )
        return await self.close(call_id, uid, "unregistered")


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


async def _request(call: Awaitable[object]) -> None:
    """*call*, one room service request, within ``ROOM_LEAVE_TIMEOUT_S``; a
    room or a participant that is not there is as good as removed."""
    from livekit import api

    try:
        await asyncio.wait_for(call, timeout=ROOM_LEAVE_TIMEOUT_S)
    except api.TwirpError as error:
        if error.code != api.TwirpErrorCode.NOT_FOUND:
            raise


async def _bounded(operation: Awaitable[T]) -> T:
    return await asyncio.wait_for(operation, timeout=locks.REDIS_TIMEOUT_S)


def _call_id_of(room_name: str) -> str | None:
    if not room_name.startswith(ROOM_PREFIX):
        return None
    return room_name.removeprefix(ROOM_PREFIX)


def _seconds(value: bytes | str | None) -> float | None:
    try:
        return float(as_str(value) or "")
    except ValueError:
        return None
