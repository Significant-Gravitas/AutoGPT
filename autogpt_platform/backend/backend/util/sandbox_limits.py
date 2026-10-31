"""How many E2B boxes may run at once: per deployment, and per account.

Every start of a box on the platform's E2B key, a create or a connect that
may wake a paused box, takes a slot here first (``start_slot``, called from
``backend.util.e2b_network``).  The slot counts the boxes E2B lists as running
with this deployment's ``autogpt_env`` stamp, and those of the account's
``autogpt_user``, and refuses a start that would take either past its limit
(``Config.e2b_deployment_sandbox_limit``, ``Config.e2b_account_sandbox_limit``).

Concurrent starts cannot overshoot: each start's reservation is added and the
held reservations read back in one Redis script, so of several starts racing
for the last place only the first to reserve sees room.  Once its box is up,
a reservation becomes the box's id and is kept for ``_STARTED_GRACE_SECONDS``,
so a box E2B does not list yet still counts, and counts once when it does.
"""

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator

from e2b import AsyncSandbox, SandboxInfo, SandboxQuery, SandboxState
from pydantic import BaseModel, ConfigDict

from backend.data.redis_client import get_redis_async
from backend.util.sandbox_metadata import (
    METADATA_PREFIX,
    SERVICE,
    SERVICE_FIELD,
    deployment_env,
)
from backend.util.settings import Config

logger = logging.getLogger(__name__)

ENV_KEY = f"{METADATA_PREFIX}env"
USER_KEY = f"{METADATA_PREFIX}user"

# Hash-tagged by deployment, so one script can touch both keys on the cluster.
_KEY_PREFIX = "e2b:sandbox_starts:"
# Longer than any single create or connect attempt: a start that dies without
# releasing its reservation stops counting after this.
_START_TTL_SECONDS = 120
# How long a started box counts from its reservation, whether or not E2B lists it.
_STARTED_GRACE_SECONDS = 30
_LIST_TIMEOUT_SECONDS = 10
# E2B lists at most 100 per page; a team runs no more than a few hundred.
_LIST_MAX_PAGES = 10

# Drop expired reservations, add this one, and return what every key holds.
_RESERVE_SCRIPT = (
    "local held = {} "
    "for i, key in ipairs(KEYS) do "
    'redis.call("zremrangebyscore", key, "-inf", ARGV[1]) '
    'redis.call("zadd", key, ARGV[2], ARGV[3]) '
    'redis.call("expire", key, ARGV[4]) '
    'held[i] = redis.call("zrange", key, 0, -1) '
    "end "
    "return held"
)
# Replace a start's reservation with its box's, for the grace period.
_STARTED_SCRIPT = (
    "for _, key in ipairs(KEYS) do "
    'redis.call("zrem", key, ARGV[1]) '
    'redis.call("zadd", key, ARGV[2], ARGV[3]) '
    'redis.call("expire", key, ARGV[4]) '
    "end "
    "return 1"
)


class SandboxLimitError(Exception):
    """A box start refused under a limit.  The message is written for the user."""

    code = "sandbox_limit_reached"


class _Scope(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    limit: int
    user_id: str | None = None
    refusal: str


class StartSlot:
    """A held place for one box start; ``started`` once the box is up."""

    def __init__(self, scopes: list[_Scope], member: str) -> None:
        self._scopes = scopes
        self._member = member

    async def started(self, sandbox_id: str) -> None:
        if not self._scopes:
            return
        try:
            redis = await get_redis_async()
            await redis.eval(
                _STARTED_SCRIPT,
                len(self._scopes),
                *(scope.key for scope in self._scopes),
                self._member,
                time.time() + _STARTED_GRACE_SECONDS,
                _box_member(sandbox_id),
                _START_TTL_SECONDS,
            )
        except Exception:
            # The reservation still counts the box until it expires.
            logger.warning(
                f"[E2B] Could not record started box {sandbox_id[:12]}", exc_info=True
            )
        self._scopes = []

    async def release(self) -> None:
        if not self._scopes:
            return
        await _release(self._scopes, self._member)
        self._scopes = []


@contextlib.asynccontextmanager
async def start_slot(
    api_key: str | None,
    user_id: str | None,
    *,
    sandbox_id: str | None = None,
) -> AsyncIterator[StartSlot]:
    """Hold a place for one box start, or raise :class:`SandboxLimitError`.

    *sandbox_id* names the box a connect may wake; a create has none yet.  The
    body calls ``started`` once the box is up; leaving without it, by an error
    or a cancellation, gives the place back.
    """
    scopes = _scopes(user_id)
    member = _box_member(sandbox_id) if sandbox_id else f"start:{uuid.uuid4().hex}"
    slot = await _admit(scopes, member, api_key)
    try:
        yield slot
    finally:
        with contextlib.suppress(BaseException):
            await slot.release()


def tagged(metadata: dict[str, str] | None, user_id: str | None) -> dict[str, str]:
    """*metadata* with the stamps the limits count by: service, deployment, account."""
    tags = {SERVICE_FIELD: SERVICE, ENV_KEY: deployment_env()}
    if user_id:
        tags[USER_KEY] = user_id
    return tags | (metadata or {})


def _scopes(user_id: str | None) -> list[_Scope]:
    # Read per start, as ``deployment_env`` does: ``Settings`` holds the values
    # its process started with.
    config = Config()
    env = deployment_env()
    scopes: list[_Scope] = []
    if config.e2b_deployment_sandbox_limit > 0:
        scopes.append(
            _Scope(
                key=f"{_KEY_PREFIX}{{{env}}}",
                limit=config.e2b_deployment_sandbox_limit,
                refusal="Too many sandboxes are running right now. "
                "Please try again in a few minutes.",
            )
        )
    if user_id and config.e2b_account_sandbox_limit > 0:
        limit = config.e2b_account_sandbox_limit
        scopes.append(
            _Scope(
                key=f"{_KEY_PREFIX}{{{env}}}:{user_id}",
                limit=limit,
                user_id=user_id,
                refusal=f"Too many sandboxes are running on your account: "
                f"{limit} at once is the most. Wait for one of your tasks to "
                "finish, then try again.",
            )
        )
    return scopes


async def _admit(scopes: list[_Scope], member: str, api_key: str | None) -> StartSlot:
    if not scopes:
        return StartSlot([], member)
    try:
        held = await _reserve(scopes, member)
    except Exception as exc:
        return await _uncounted(scopes, StartSlot([], member), exc)
    slot = StartSlot(scopes, member)
    try:
        # Past a limit on reservations alone, E2B need not be asked.
        _check(scopes, held, running=None)
        _check(scopes, held, running=await _running(scopes, api_key))
    except SandboxLimitError as refusal:
        await slot.release()
        logger.info(f"[E2B] Refused a sandbox start: {refusal}")
        raise
    except Exception as exc:
        return await _uncounted(scopes, slot, exc)
    return slot


def _check(
    scopes: list[_Scope],
    held: list[list[str]],
    running: dict[str, set[str]] | None,
) -> None:
    for scope, members in zip(scopes, held):
        boxes = {m for m in members if m.startswith("box:")}
        starts = len(members) - len(boxes)
        listed = (
            {_box_member(i) for i in running[scope.key]}
            if running is not None
            else set()
        )
        if len(listed | boxes) + starts > scope.limit:
            raise SandboxLimitError(scope.refusal)


async def _uncounted(
    scopes: list[_Scope], slot: StartSlot, error: Exception
) -> StartSlot:
    """A start whose count could not be read: refused under a deployment
    ceiling, which exists to hold, and let through under the account cap alone,
    keeping whatever reservation it made."""
    logger.warning("[E2B] Could not count running sandboxes", exc_info=error)
    if any(scope.user_id is None for scope in scopes):
        await slot.release()
        raise SandboxLimitError(
            "Couldn't check how many sandboxes are running, so none was started. "
            "Please try again in a minute."
        )
    return slot


async def _reserve(scopes: list[_Scope], member: str) -> list[list[str]]:
    now = time.time()
    redis = await get_redis_async()
    held = await redis.eval(
        _RESERVE_SCRIPT,
        len(scopes),
        *(scope.key for scope in scopes),
        now,
        now + _START_TTL_SECONDS,
        member,
        _START_TTL_SECONDS,
    )
    return [[_text(m) for m in members] for members in held]


async def _release(scopes: list[_Scope], member: str) -> None:
    try:
        redis = await get_redis_async()
        for scope in scopes:
            await redis.zrem(scope.key, member)
    except Exception:
        # It stops counting once its TTL passes.
        logger.warning("[E2B] Could not release a sandbox start", exc_info=True)


async def _running(scopes: list[_Scope], api_key: str | None) -> dict[str, set[str]]:
    """The running boxes each scope counts, from one E2B listing.

    The listing's metadata filter only narrows the pages; each box's stamps
    are read again here, so the count never rests on how E2B matches them.
    """
    env = deployment_env()
    account = next((scope for scope in scopes if scope.user_id), None)
    wanted = {SERVICE_FIELD: SERVICE, ENV_KEY: env}
    if account and len(scopes) == 1:
        wanted[USER_KEY] = account.user_id or ""
    infos = await _list_running(wanted, api_key)
    return {
        scope.key: {
            info.sandbox_id
            for info in infos
            if _stamped(info, wanted)
            and (scope.user_id is None or _stamp(info, USER_KEY) == scope.user_id)
        }
        for scope in scopes
    }


async def _list_running(
    metadata: dict[str, str], api_key: str | None
) -> list[SandboxInfo]:
    paginator = AsyncSandbox.list(
        query=SandboxQuery(metadata=metadata, state=[SandboxState.RUNNING]),
        limit=100,
        **({"api_key": api_key} if api_key else {}),
    )
    infos: list[SandboxInfo] = []
    for _ in range(_LIST_MAX_PAGES):
        if not paginator.has_next:
            break
        infos += await asyncio.wait_for(
            paginator.next_items(), timeout=_LIST_TIMEOUT_SECONDS
        )
    return [info for info in infos if info.state == SandboxState.RUNNING]


def _stamped(info: SandboxInfo, wanted: dict[str, str]) -> bool:
    return all(_stamp(info, key) == value for key, value in wanted.items())


def _stamp(info: SandboxInfo, key: str) -> str | None:
    return (info.metadata or {}).get(key)


def _box_member(sandbox_id: str) -> str:
    return f"box:{sandbox_id}"


def _text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value
