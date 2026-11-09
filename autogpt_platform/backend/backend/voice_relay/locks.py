"""A voice call's shared state in Redis.

Every operation touches one key, so the relay and the REST API run against the
production Redis cluster unchanged:

- ``voice:live:<user id>`` names the user's live call: one live interview per
  user. It is taken with SET NX and a TTL longer than any call, so a dead
  pod's lock clears itself, and released by compare-and-delete, so a call
  never frees a lock that a later call holds.
- ``voice:call:<call id>``, a hash, is the call's record for a day: its owner,
  pod, leg and status, how it ended, where Otto's turns ended in the
  recording, the transcript of record and what the hand-off answered
  (:class:`CallRecord`).
- ``voice:concurrent``, a sorted set of live call ids scored by when each was
  last refreshed, counts live calls across the platform. A call whose pod died
  stops counting once its entry goes stale.
- ``voice:end:<call id>`` asks the pod that owns a call to end it; the pod
  looks every second.
- ``voice:spend:<UTC date>`` is what voice calls spent across the platform
  that day, which the start route holds to the daily budget.
- ``voice:postcall:pending``, a set, holds every call whose post-call job has
  not finished, from the moment the call starts. ``voice:postcall:claim:<call
  id>`` names the process running a call's job, so each job runs once across
  pods (``postcall``).
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Mapping
from datetime import UTC, datetime, timedelta
from typing import TypeVar

from prisma.enums import BrainDumpStatus
from pydantic import BaseModel, ConfigDict, ValidationError
from redis.typing import EncodableT, FieldT

from backend.api.features.onboarding_dump.models import (
    FinalizeResponse,
    VoiceCallStatus,
)
from backend.data.redis_client import AsyncRedisClient
from backend.data.redis_helpers import (
    SlotAdmission,
    as_str,
    string_compare_and_set,
    try_acquire_concurrency_slot,
)
from backend.voice_relay.config import (
    CALL_RECORD_TTL_S,
    END_REQUEST_TTL_S,
    LIVE_CALL_STALE_S,
    POSTCALL_CLAIM_TTL_S,
    USER_LOCK_TTL_S,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

LIVE_CALLS_KEY = "voice:concurrent"
POSTCALL_PENDING_KEY = "voice:postcall:pending"
# How long one Redis operation of a live call may take before it is skipped.
REDIS_TIMEOUT_S = 0.5
DAILY_SPEND_TTL_S = 2 * 24 * 3600

# Why a second call, or the typed interview, is refused while a call of the
# person's is live. The refusal names the call.
CALL_IN_PROGRESS = "call_in_progress"

# A call's status on its record, as the call routes report it.
CallStatus = VoiceCallStatus
CONNECTING: CallStatus = "connecting"
LIVE: CallStatus = "live"
ENDED: CallStatus = "ended"
TRANSCRIBING: CallStatus = "transcribing"
TRANSCRIBED: CallStatus = "transcribed"
HANDED_OFF: CallStatus = "handed_off"
FAILED: CallStatus = "failed"


def user_lock_key(user_id: str) -> str:
    return f"voice:live:{user_id}"


def call_record_key(call_id: str) -> str:
    return f"voice:call:{call_id}"


def end_flag_key(call_id: str) -> str:
    return f"voice:end:{call_id}"


async def take_user_lock(
    redis: AsyncRedisClient, user_id: str, call_id: str
) -> str | None:
    """Make *call_id* the user's live call.

    Returns None once it is, or the id of the live call that already holds
    the lock (blank in the rare case it kept changing hands while read).
    """
    key = user_lock_key(user_id)
    for _ in range(2):
        if await redis.set(key, call_id, nx=True, ex=USER_LOCK_TTL_S):
            return None
        holder = as_str(await redis.get(key))
        if holder is not None:
            return holder
        # The holder's lock expired between the two reads: try once more.
    return ""


async def live_call_of(redis: AsyncRedisClient, user_id: str) -> str | None:
    """The user's live call, or None."""
    return as_str(await redis.get(user_lock_key(user_id)))


async def release_user_lock(
    redis: AsyncRedisClient, user_id: str, call_id: str
) -> bool:
    """Free the user's lock if *call_id* still holds it. True when it did."""
    return await string_compare_and_set(
        redis, user_lock_key(user_id), expected=call_id, new=None
    )


async def write_call_record(
    redis: AsyncRedisClient, call_id: str, fields: Mapping[str, object]
) -> None:
    """Set *fields* on the call's record and give it a day to live.

    A None value is left out; anything that is not a string is stored as
    JSON.
    """
    mapping: dict[FieldT, EncodableT] = {
        name: value if isinstance(value, str) else json.dumps(value)
        for name, value in fields.items()
        if value is not None
    }
    if not mapping:
        return
    key = call_record_key(call_id)
    pipe = redis.pipeline(transaction=True)
    pipe.hset(key, mapping=mapping)
    pipe.expire(key, CALL_RECORD_TTL_S)
    await pipe.execute()


async def read_call_record(redis: AsyncRedisClient, call_id: str) -> dict[str, str]:
    raw = await redis.hgetall(call_record_key(call_id))
    return {
        as_str(name) or "": as_str(value) or "" for name, value in (raw or {}).items()
    }


async def forget_call_fields(
    redis: AsyncRedisClient, call_id: str, set_fields: Mapping[str, object], *names: str
) -> None:
    """Set *set_fields* on the call's record and drop *names* from it, in one
    write."""
    mapping: dict[FieldT, EncodableT] = {
        name: value if isinstance(value, str) else json.dumps(value)
        for name, value in set_fields.items()
        if value is not None
    }
    key = call_record_key(call_id)
    pipe = redis.pipeline(transaction=True)
    if mapping:
        pipe.hset(key, mapping=mapping)
    if names:
        pipe.hdel(key, *names)
    pipe.expire(key, CALL_RECORD_TTL_S)
    await pipe.execute()


class TurnMark(BaseModel):
    """Where one of Otto's turns ended in the person's recording."""

    model_config = ConfigDict(frozen=True)

    index: int
    # Samples of the person's audio recorded when the turn completed.
    at: int
    text: str = ""


class CallRecord(BaseModel):
    """A call's record, read back from Redis."""

    call_id: str
    uid: str = ""
    status: CallStatus | None = None
    leg: str = "websocket"
    end_reason: str | None = None
    # The task Otto's end tool carried.
    first_task: str | None = None
    # The latest upload of the person's audio, until its transcript is made.
    audio_path: str | None = None
    turns: list[TurnMark] = []
    # The person's words from the call, one turn per line.
    transcript_of_record: str | None = None
    # What the onboarding pipeline answered the call's hand-off.
    hand_off: FinalizeResponse | None = None
    postcall_attempts: int = 0

    @property
    def took_the_interview(self) -> bool:
        """Whether the call's words became the interview's take: handed off
        and accepted, not superseded by another take."""
        return (
            self.status == HANDED_OFF
            and self.hand_off is not None
            and self.hand_off.status != BrainDumpStatus.failed
        )


async def read_call(redis: AsyncRedisClient, call_id: str) -> CallRecord | None:
    """The call's record, or None once it has expired or was never written.

    A field that cannot be read is left at its default.
    """
    raw = await read_call_record(redis, call_id)
    if not raw:
        return None
    fields: dict[str, object] = {"call_id": call_id}
    for name, value in raw.items():
        if name not in CallRecord.model_fields or name == "call_id":
            continue
        if name in ("turns", "hand_off", "postcall_attempts"):
            try:
                fields[name] = json.loads(value)
            except ValueError:
                logger.warning(f"Voice call {call_id}: {name} unreadable on its record")
            continue
        fields[name] = value
    try:
        return CallRecord.model_validate(fields)
    except ValidationError:
        logger.warning(f"Voice call {call_id}: record unreadable, read as bare")
        return CallRecord(call_id=call_id, uid=raw.get("uid", ""))


async def acquire_live_call(
    redis: AsyncRedisClient, call_id: str, now: float, capacity: int
) -> bool:
    """Count *call_id* as live across the platform if there is room for it.

    False, with nothing written, once *capacity* live calls are counted.
    Entries a dead pod left behind are dropped first, so they hold no room.
    """
    admission = await try_acquire_concurrency_slot(
        redis,
        pool_key=LIVE_CALLS_KEY,
        slot_id=call_id,
        score=now,
        capacity=capacity,
        stale_before_score=now - LIVE_CALL_STALE_S,
        ttl_seconds=int(LIVE_CALL_STALE_S * 3),
    )
    return admission != SlotAdmission.REJECTED


async def register_live_call(redis: AsyncRedisClient, call_id: str, now: float) -> None:
    """Count *call_id* as live across the platform, or refresh its entry.

    Entries a dead pod left behind are dropped on the way.
    """
    pipe = redis.pipeline(transaction=True)
    pipe.zremrangebyscore(LIVE_CALLS_KEY, "-inf", now - LIVE_CALL_STALE_S)
    pipe.zadd(LIVE_CALLS_KEY, {call_id: now})
    pipe.expire(LIVE_CALLS_KEY, int(LIVE_CALL_STALE_S * 3))
    await pipe.execute()


async def unregister_live_call(redis: AsyncRedisClient, call_id: str) -> None:
    await redis.zrem(LIVE_CALLS_KEY, call_id)


async def live_call_count(redis: AsyncRedisClient, now: float) -> int:
    """Live calls across the platform, not counting entries gone stale."""
    return int(
        await redis.zcount(LIVE_CALLS_KEY, f"({now - LIVE_CALL_STALE_S}", "+inf")
    )


async def request_end(redis: AsyncRedisClient, call_id: str) -> None:
    """Ask the pod that owns *call_id* to end it."""
    await redis.set(end_flag_key(call_id), "1", ex=END_REQUEST_TTL_S)


async def end_requested(redis: AsyncRedisClient, call_id: str) -> bool:
    return bool(await redis.exists(end_flag_key(call_id)))


async def is_live(redis: AsyncRedisClient, call_id: str, now: float) -> bool:
    """Whether *call_id* still counts as live: its pod refreshed it lately."""
    score = await redis.zscore(LIVE_CALLS_KEY, call_id)
    return score is not None and float(score) > now - LIVE_CALL_STALE_S


def postcall_claim_key(call_id: str) -> str:
    return f"voice:postcall:claim:{call_id}"


async def add_postcall_pending(redis: AsyncRedisClient, call_id: str) -> None:
    await redis.sadd(POSTCALL_PENDING_KEY, call_id)


async def remove_postcall_pending(redis: AsyncRedisClient, call_id: str) -> None:
    await redis.srem(POSTCALL_PENDING_KEY, call_id)


async def postcall_pending(redis: AsyncRedisClient) -> list[str]:
    """The calls whose post-call job has not finished."""
    members = await redis.smembers(POSTCALL_PENDING_KEY)
    return sorted(as_str(member) or "" for member in members or ())


async def claim_postcall(
    redis: AsyncRedisClient,
    call_id: str,
    holder: str,
    ttl_s: int = POSTCALL_CLAIM_TTL_S,
) -> bool:
    """Make *holder* the one that runs *call_id*'s post-call job. False when
    another holds it."""
    return bool(await redis.set(postcall_claim_key(call_id), holder, nx=True, ex=ttl_s))


async def release_postcall(redis: AsyncRedisClient, call_id: str, holder: str) -> None:
    """Give up *holder*'s claim on *call_id*'s post-call job, if it still has it."""
    await string_compare_and_set(
        redis, postcall_claim_key(call_id), expected=holder, new=None
    )


def daily_spend_key(now: datetime) -> str:
    return f"voice:spend:{now.astimezone(UTC).date().isoformat()}"


async def add_daily_spend(redis: AsyncRedisClient, usd: float, now: datetime) -> None:
    key = daily_spend_key(now)
    pipe = redis.pipeline(transaction=True)
    pipe.incrbyfloat(key, usd)
    pipe.expire(key, DAILY_SPEND_TTL_S)
    await pipe.execute()


async def daily_spend_usd(redis: AsyncRedisClient, now: datetime) -> float:
    """What voice calls have spent across the platform today (UTC)."""
    raw = as_str(await redis.get(daily_spend_key(now)))
    return float(raw) if raw else 0.0


def seconds_until_tomorrow(now: datetime) -> int:
    """Seconds until the UTC day turns and the daily budget starts over."""
    utc = now.astimezone(UTC)
    tomorrow = datetime.combine(
        utc.date() + timedelta(days=1), datetime.min.time(), UTC
    )
    return max(1, int((tomorrow - utc).total_seconds()))


class CallStore:
    """One live call's shared state, as the pod that owns the call keeps it.

    A Redis outage never ends or stalls a call: each operation is bounded,
    and a failed one is logged and skipped.

    *holder* names the process the call runs on, which claims the call's
    post-call job when the call ends (:meth:`claim_postcall`). Without one,
    the call leaves its job to the sweep.
    """

    def __init__(
        self,
        redis: AsyncRedisClient,
        *,
        call_id: str,
        user_id: str,
        holder: str | None = None,
    ) -> None:
        self.redis = redis
        self.call_id = call_id
        self.user_id = user_id
        self.holder = holder

    async def record(self, **fields: object) -> bool:
        """Set *fields* on the call's record. True once written."""

        async def write() -> bool:
            await write_call_record(self.redis, self.call_id, fields)
            return True

        return bool(await self._bounded("record", write()))

    async def pend_postcall(self) -> None:
        """Count the call as one whose post-call job has yet to finish, so a
        call whose pod dies is still found and finished."""
        await self._bounded(
            "post-call pending", add_postcall_pending(self.redis, self.call_id)
        )

    async def claim_postcall(self) -> bool:
        """Claim the call's post-call job for this process. False when there
        is no holder, another holds it, or Redis did not answer."""
        if self.holder is None:
            return False
        return bool(
            await self._bounded(
                "post-call claim",
                claim_postcall(self.redis, self.call_id, self.holder),
            )
        )

    async def refresh_live(self, now: float) -> None:
        await self._bounded(
            "live-call refresh", register_live_call(self.redis, self.call_id, now)
        )

    async def end_requested(self) -> bool:
        return bool(
            await self._bounded("end flag", end_requested(self.redis, self.call_id))
        )

    async def release(self) -> None:
        """Free the user's lock and stop counting the call as live. Idempotent."""
        await self._bounded(
            "lock release", release_user_lock(self.redis, self.user_id, self.call_id)
        )
        await self._bounded(
            "live-call release", unregister_live_call(self.redis, self.call_id)
        )

    async def _bounded(self, what: str, operation: Awaitable[T]) -> T | None:
        try:
            return await asyncio.wait_for(operation, timeout=REDIS_TIMEOUT_S)
        except Exception:
            logger.warning(f"Voice call {self.call_id}: {what} failed", exc_info=True)
            return None
