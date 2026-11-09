"""A voice call's shared state in Redis.

Every operation touches one key, so the relay and the REST API run against the
production Redis cluster unchanged:

- ``voice:live:<user id>`` names the user's live call: one live interview per
  user. It is taken with SET NX and a TTL longer than any call, and released
  by compare-and-delete, so a call never frees a lock that a later call
  holds. It is honoured only while its call runs (:func:`call_is_live`), so a
  dead pod's call holds nobody back once its pod stops refreshing it.
- ``voice:call:<call id>``, a hash, is the call's record for a day: its owner,
  pod, leg and status, how it ended, where Otto's turns ended in the
  recording, the transcript of record, what the hand-off answered, and whether
  the call's words took the person's interview or their typing superseded
  them (:class:`CallRecord`, :func:`claim_interview`).
- ``voice:concurrent``, a sorted set of live call ids scored by when each was
  last refreshed, counts live calls across the platform. A call whose pod died
  stops counting once its entry goes stale.
- ``voice:end:<call id>`` asks the pod that owns a call to end it; the pod
  looks every second.
- ``voice:spend:<UTC date>`` is what voice calls spent across the platform
  that day, which the start route holds to the daily budget.
- ``voice:postcall:pending``, a set, holds every call whose words are not yet
  settled, from the moment the call starts: the relay's transcript job
  (``postcall``) leaves a call there once its transcript is made, and the REST
  API's hand-off (``onboarding_dump.voice_handoff``) takes it out once the
  pipeline answers. ``voice:postcall:claim:<call id>`` is the one mutex over
  both: the pod running a call claims it as the call starts and holds it with
  the call's refresh until the transcript is stored, and the REST API claims
  it for each hand-off.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar, cast

from prisma.enums import BrainDumpStatus
from pydantic import BaseModel, ConfigDict, ValidationError
from redis.typing import EncodableT, FieldT

from backend.api.features.onboarding_dump.models import (
    FinalizeResponse,
    VoiceCallInterview,
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
# The statuses of a call still running.
RUNNING: tuple[CallStatus, ...] = (CONNECTING, LIVE)

# Who owns the person's interview as far as a call goes: its words took it, or
# the person's typing after the call superseded them. Whichever side claims
# first keeps it (:func:`claim_interview`).
INTERVIEW_TAKEN: VoiceCallInterview = "taken"
INTERVIEW_SUPERSEDED: VoiceCallInterview = "superseded"


def user_lock_key(user_id: str) -> str:
    return f"voice:live:{user_id}"


def call_record_key(call_id: str) -> str:
    return f"voice:call:{call_id}"


def end_flag_key(call_id: str) -> str:
    return f"voice:end:{call_id}"


async def take_user_lock(
    redis: AsyncRedisClient, user_id: str, call_id: str, now: float | None = None
) -> str | None:
    """Make *call_id* the user's live call.

    Returns None once it is, or the id of the live call that already holds
    the lock (blank in the rare case it kept changing hands while read). A
    lock whose call is no longer running (:func:`call_is_live`) is taken over
    from it.
    """
    key = user_lock_key(user_id)
    for _ in range(2):
        if await redis.set(key, call_id, nx=True, ex=USER_LOCK_TTL_S):
            return None
        holder = as_str(await redis.get(key))
        if holder is None:
            # The holder's lock expired between the two reads.
            continue
        if await call_is_live(redis, holder, time.time() if now is None else now):
            return holder
        if await string_compare_and_set(
            redis, key, expected=holder, new=call_id, ttl_seconds=USER_LOCK_TTL_S
        ):
            return None
    return ""


async def live_call_of(redis: AsyncRedisClient, user_id: str) -> str | None:
    """The call holding the user's lock, or None, whether or not it still
    runs. Whose a call is; :func:`live_call_holding` is whether it holds the
    person back."""
    return as_str(await redis.get(user_lock_key(user_id)))


async def live_call_holding(
    redis: AsyncRedisClient, user_id: str, now: float
) -> str | None:
    """The user's live call: the call holding their lock, while it runs. A
    lock left by a call that ended, or whose pod died, holds nothing."""
    holder = await live_call_of(redis, user_id)
    if holder is None or not await call_is_live(redis, holder, now):
        return None
    return holder


async def call_is_live(redis: AsyncRedisClient, call_id: str, now: float) -> bool:
    """Whether *call_id* still runs: its record is not past ``live``, and its
    pod refreshed its place in the platform's count lately.

    A call that has neither yet is starting: the lock is taken before the
    other two are written. A record still ``live`` whose place is gone was
    dropped as stale, so its pod died.
    """
    status = as_str(await redis.hget(call_record_key(call_id), "status"))
    if status is not None and status not in RUNNING:
        return False
    score = await redis.zscore(LIVE_CALLS_KEY, call_id)
    if score is None:
        return status is None
    return float(score) > now - LIVE_CALL_STALE_S


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
    mapping = _encoded(fields)
    if not mapping:
        return
    key = call_record_key(call_id)
    pipe = redis.pipeline(transaction=True)
    pipe.hset(key, mapping=mapping)
    pipe.expire(key, CALL_RECORD_TTL_S)
    await pipe.execute()


def _encoded(fields: Mapping[str, object]) -> dict[FieldT, EncodableT]:
    """*fields* as the record stores them: strings as they are, anything else
    as JSON, and a None left out."""
    return {
        name: value if isinstance(value, str) else json.dumps(value)
        for name, value in fields.items()
        if value is not None
    }


# Moves the call's status, with the fields that move with it, only from the
# statuses given; the other fields are written whatever the status.
# KEYS[1]: the record. ARGV[1]: its TTL. ARGV[2]: the new status. ARGV[3]: the
# statuses it moves from, comma-separated, an empty entry for a record with
# none. ARGV[4]: how many field/value pairs after it move with the status.
_ADVANCE_LUA = """
local current = redis.call('HGET', KEYS[1], 'status') or ''
local moves = false
for allowed in string.gmatch(ARGV[3] .. ',', '([^,]*),') do
    if allowed == current then moves = true end
end
if moves then
    redis.call('HSET', KEYS[1], 'status', ARGV[2])
end
local guarded_end = 4 + tonumber(ARGV[4]) * 2
for i = 5, #ARGV, 2 do
    if moves or i > guarded_end then
        redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1])
    end
end
redis.call('EXPIRE', KEYS[1], ARGV[1])
if moves then return 1 end
return 0
"""


async def advance_call(
    redis: AsyncRedisClient,
    call_id: str,
    status: CallStatus,
    *,
    moves_from: Iterable[CallStatus | None],
    guarded: Mapping[str, object] | None = None,
    fields: Mapping[str, object] | None = None,
) -> bool:
    """Move the call to *status*, and set *guarded* with it, only while its
    status is one of *moves_from* (None for a record with none yet); set
    *fields* whatever it is. One script on the one key, so a status written
    elsewhere since is never moved back. True when the status moved."""
    moving = _encoded(guarded or {})
    rest = _encoded(fields or {})
    args: list[EncodableT] = [
        CALL_RECORD_TTL_S,
        status,
        ",".join(state or "" for state in moves_from),
        len(moving),
    ]
    for pairs in (moving, rest):
        for name, value in pairs.items():
            args.extend((cast(EncodableT, name), value))
    moved = await cast(
        Any, redis.eval(_ADVANCE_LUA, 1, call_record_key(call_id), *args)
    )
    return int(moved) == 1


# Stores the transcript of record, moves the call to ``transcribed`` and
# forgets the recording, in one write. A call whose hand-off answered is left
# as it is, and a transcript already stored is never replaced by an empty one.
# KEYS[1]: the record. ARGV[1]: the transcript. ARGV[2]: whether it was made
# from the last upload, as JSON. ARGV[3]: the owner. ARGV[4]: when, in
# seconds since the epoch. ARGV[5]: the TTL.
_STORE_TRANSCRIPT_LUA = """
local stored = redis.call('HGET', KEYS[1], 'transcript_of_record')
if redis.call('HEXISTS', KEYS[1], 'hand_off') == 1 then
    return stored
end
if ARGV[1] ~= '' or not stored then
    redis.call('HSET', KEYS[1], 'transcript_of_record', ARGV[1],
        'transcribed_from_upload', ARGV[2])
    stored = ARGV[1]
end
redis.call('HSET', KEYS[1], 'status', 'transcribed', 'transcribed_at', ARGV[4],
    'uid', ARGV[3])
redis.call('HDEL', KEYS[1], 'audio_path')
redis.call('EXPIRE', KEYS[1], ARGV[5])
return stored
"""


async def store_transcript(
    redis: AsyncRedisClient,
    call_id: str,
    transcript: str,
    *,
    uid: str,
    from_upload: bool,
    now: float | None = None,
) -> str | None:
    """Store the call's transcript of record and move the call to
    ``transcribed``, in the write that forgets its recording.

    A call whose hand-off has answered is left as it is, and an empty
    transcript never replaces one stored. Returns the transcript the record
    holds.
    """
    stored = await cast(
        Any,
        redis.eval(
            _STORE_TRANSCRIPT_LUA,
            1,
            call_record_key(call_id),
            transcript,
            json.dumps(from_upload),
            uid,
            round(time.time() if now is None else now, 3),
            CALL_RECORD_TTL_S,
        ),
    )
    return as_str(stored)


async def record_answer(
    redis: AsyncRedisClient,
    call_id: str,
    *,
    uid: str,
    status: CallStatus,
    answer: FinalizeResponse,
) -> None:
    """Write an answer on the call's record without a hand-off: the relay's
    own, when no transcript could be made."""
    await write_call_record(
        redis,
        call_id,
        {"uid": uid, "status": status, "hand_off": answer.model_dump(mode="json")},
    )


# Sets the record's ``interview`` field unless it holds one; never creates a
# record. Returns the field as it stands, or nothing for no record.
_CLAIM_INTERVIEW_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then return false end
redis.call('HSETNX', KEYS[1], 'interview', ARGV[1])
return redis.call('HGET', KEYS[1], 'interview')
"""


async def claim_interview(
    redis: AsyncRedisClient, call_id: str, claim: VoiceCallInterview
) -> VoiceCallInterview | None:
    """Claim the person's interview for *claim* on the call's record: taken by
    the call's words, or superseded by the person's typing.

    Set-if-absent on the one record, so of two claims exactly one wins.
    Returns the claim that stands, the winner's, or None when the call has no
    record.
    """
    standing = as_str(
        await cast(
            Any,
            redis.eval(_CLAIM_INTERVIEW_LUA, 1, call_record_key(call_id), claim),
        )
    )
    if standing == INTERVIEW_TAKEN:
        return INTERVIEW_TAKEN
    if standing == INTERVIEW_SUPERSEDED:
        return INTERVIEW_SUPERSEDED
    return None


async def interview_of(
    redis: AsyncRedisClient, call_id: str
) -> VoiceCallInterview | None:
    """The interview claim on the call's record, if one was made."""
    standing = as_str(await redis.hget(call_record_key(call_id), "interview"))
    if standing == INTERVIEW_TAKEN:
        return INTERVIEW_TAKEN
    if standing == INTERVIEW_SUPERSEDED:
        return INTERVIEW_SUPERSEDED
    return None


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
    mapping = _encoded(set_fields)
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
    # When the call started, in seconds since the epoch.
    started: float | None = None
    end_reason: str | None = None
    # The person's arm of the voice ending experiment, read by the start route
    # and carried in the start token; None for anyone not enrolled.
    ending: str | None = None
    # The task Otto's end tool carried.
    first_task: str | None = None
    # The latest upload of the person's audio, until its transcript is made.
    audio_path: str | None = None
    turns: list[TurnMark] = []
    # The person's words from the call, one turn per line.
    transcript_of_record: str | None = None
    # When the transcript was stored, in seconds since the epoch.
    transcribed_at: float | None = None
    # Whether the transcript was made from the call's last upload, which a
    # call whose pod died can leave short of its end, rather than from the
    # whole recording.
    transcribed_from_upload: bool = False
    # What the onboarding pipeline answered the call's hand-off.
    hand_off: FinalizeResponse | None = None
    # Whether the call's words took the person's interview or their typing
    # superseded them, as first claimed (:func:`claim_interview`).
    interview: VoiceCallInterview | None = None
    # Attempts at the transcript job, and at the hand-off.
    postcall_attempts: int = 0
    handoff_attempts: int = 0

    @property
    def took_interview(self) -> bool:
        return self.interview == INTERVIEW_TAKEN

    @property
    def accepted(self) -> bool:
        """Whether the pipeline took the call's words: handed off, neither
        refused nor superseded."""
        return (
            self.status == HANDED_OFF
            and self.hand_off is not None
            and self.hand_off.status != BrainDumpStatus.failed
        )

    @property
    def heard(self) -> bool:
        """Whether the relay took any of the person's audio by the time Otto's
        last turn ended."""
        return bool(self.turns) and self.turns[-1].at > 0


# The record's fields stored as JSON; every other one is a plain string.
_JSON_FIELDS = frozenset(
    {
        "turns",
        "hand_off",
        "transcribed_from_upload",
        "postcall_attempts",
        "handoff_attempts",
    }
)


async def read_call(redis: AsyncRedisClient, call_id: str) -> CallRecord | None:
    """The call's record, or None once it has expired or was never written.

    A field that cannot be read is left at its default; the others are read
    as they are.
    """
    raw = await read_call_record(redis, call_id)
    if not raw:
        return None
    fields: dict[str, object] = {}
    for name, value in raw.items():
        if name not in CallRecord.model_fields or name == "call_id":
            continue
        if name not in _JSON_FIELDS:
            fields[name] = value
            continue
        try:
            fields[name] = json.loads(value)
        except ValueError:
            logger.warning(f"Voice call {call_id}: {name} unreadable on its record")
    while True:
        try:
            return CallRecord.model_validate({**fields, "call_id": call_id})
        except ValidationError as error:
            unreadable = {
                str(problem["loc"][0])
                for problem in error.errors()
                if problem["loc"] and str(problem["loc"][0]) in fields
            }
            if not unreadable:
                logger.warning(f"Voice call {call_id}: record unreadable")
                return CallRecord(call_id=call_id, uid=raw.get("uid", ""))
            logger.warning(
                f"Voice call {call_id}: {', '.join(sorted(unreadable))} unreadable "
                f"on its record"
            )
            for name in unreadable:
                del fields[name]


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
    """The calls whose words are not settled yet."""
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


# Gives a claim its full TTL again, only while its holder still holds it.
_HOLD_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 0
"""


async def hold_postcall(
    redis: AsyncRedisClient,
    call_id: str,
    holder: str,
    ttl_s: int = POSTCALL_CLAIM_TTL_S,
) -> bool:
    """Give *holder*'s claim on *call_id*'s post-call job its full TTL again.
    False when *holder* no longer holds it."""
    held = await cast(
        Any, redis.eval(_HOLD_LUA, 1, postcall_claim_key(call_id), holder, ttl_s)
    )
    return int(held) == 1


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
    and a failed one is logged and skipped. Every write names the call's
    owner, so no write leaves a record without one.

    *holder* names the process the call runs on, which claims the call's
    post-call job as the call starts and holds it with each refresh
    (:meth:`claim_postcall`). Without one, the call leaves its job to the
    sweep.
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
        # Whether this process holds the call's post-call job.
        self.claimed = False

    async def record(self, **fields: object) -> bool:
        """Set *fields* on the call's record. True once written."""

        async def write() -> bool:
            await write_call_record(
                self.redis, self.call_id, {**fields, "uid": self.user_id}
            )
            return True

        return bool(await self._bounded("record", write()))

    async def advance(
        self,
        status: CallStatus,
        *,
        moves_from: Iterable[CallStatus | None],
        guarded: Mapping[str, object] | None = None,
        **fields: object,
    ) -> bool:
        """Move the call to *status*, with *guarded*, unless its status has
        moved on since (:func:`advance_call`); *fields* are set either way.
        True when the status moved."""
        return bool(
            await self._bounded(
                "status",
                advance_call(
                    self.redis,
                    self.call_id,
                    status,
                    moves_from=moves_from,
                    guarded=guarded,
                    fields={**fields, "uid": self.user_id},
                ),
            )
        )

    async def pend_postcall(self) -> None:
        """Count the call as one whose post-call job has yet to finish, so a
        call whose pod dies is still found and finished."""
        await self._bounded(
            "post-call pending", add_postcall_pending(self.redis, self.call_id)
        )

    async def claim_postcall(self) -> bool:
        """Claim the call's post-call job for this process, unless it holds it
        already. False when there is no holder, another holds it, or Redis did
        not answer."""
        if self.holder is None:
            return False
        if not self.claimed:
            self.claimed = bool(
                await self._bounded(
                    "post-call claim",
                    claim_postcall(self.redis, self.call_id, self.holder),
                )
            )
        return self.claimed

    async def refresh_live(self, now: float) -> None:
        """Count the call as live again, and hold its post-call job for
        another full TTL."""
        await self._bounded(
            "live-call refresh", register_live_call(self.redis, self.call_id, now)
        )
        if self.claimed and self.holder is not None:
            held = await self._bounded(
                "post-call hold",
                hold_postcall(self.redis, self.call_id, self.holder),
            )
            if held is False:
                logger.warning(f"Voice call {self.call_id}: post-call claim lost")
                self.claimed = False

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
