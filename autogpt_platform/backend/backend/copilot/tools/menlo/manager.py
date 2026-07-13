"""Lifecycle + connection management for the copilot's Menlo robot tools.

The Menlo SDK has two planes (see the SDK's ``llms-full.txt``):

* **control plane** (HTTP): create/list/delete robots, mint viewer tokens.
* **runtime plane** (LiveKit SFU): ``invoke`` / ``state`` / ``get_vision`` /
  ``discover_skills`` — these require a *runtime worker* (rcw) in the room. For
  the SimpleSim model the rcw is the browser 3D viewer tab, so the user must
  open the viewer link before runtime calls succeed.

Persistence mirrors ``e2b_sandbox.py``: the durable, cross-turn handle is the
``robot_id`` stored in Redis keyed by ``session_id``; the live LiveKit
``MenloSession`` is cached process-locally and re-established from the stored
``robot_id`` when a later turn lands on a fresh process.

The SDK is an optional dependency (extra ``menlo``, Python >=3.12), so every
import of it is lazy and guarded by :func:`menlo_available`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from backend.data.redis_client import get_redis_async
from backend.util.settings import Settings

if TYPE_CHECKING:
    from menlo_robot_sdk import AsyncClient, MenloSession

logger = logging.getLogger(__name__)

# The SimpleSim viewer tab joins the room as ``simplesim:<robot_id>``; this
# prefix is how the SDK resolves it as the runtime worker for runtime calls.
_RCW_IDENTITY_PREFIX = "simplesim"
_DEFAULT_ROBOT_MODEL = "asimov-v0"

_ROBOT_KEY_PREFIX = "copilot:menlo:robot:"
# Match the viewer room-key lifetime (4h) so a stored robot_id never outlives
# the link the user is holding.
_ROBOT_ID_TTL = 4 * 3600

# Process-local cache of the live connection, keyed by session_id. Holds the
# non-serializable AsyncClient + MenloSession so multiple tool calls in one turn
# reuse a single LiveKit room join.
_LIVE: dict[str, "LiveConnection"] = {}


class MenloConfigError(RuntimeError):
    """MENLO_API_KEY is not configured (or the SDK extra is not installed)."""


class MenloNotConnectedError(RuntimeError):
    """No robot is connected for this chat session — call menlo_connect_robot first."""


class LiveConnection:
    """A live Menlo connection for one chat session (not serializable)."""

    def __init__(self, robot_id: str, client: "AsyncClient", session: "MenloSession"):
        self.robot_id = robot_id
        self.client = client
        self.session = session


def menlo_available() -> bool:
    """True when the copilot robot tools are usable: key configured + SDK present."""
    if not Settings().secrets.menlo_api_key:
        return False
    try:
        import menlo_robot_sdk  # noqa: F401
    except ImportError:
        return False
    return True


def _menlo_config() -> tuple[str, str, str]:
    """Return ``(api_key, rcs_url, viewer_url)`` or raise :class:`MenloConfigError`."""
    settings = Settings()
    api_key = settings.secrets.menlo_api_key
    if not api_key:
        raise MenloConfigError("MENLO_API_KEY is not configured")
    return (
        api_key,
        settings.config.menlo_rcs_url,
        settings.config.menlo_robot_viewer_url,
    )


def _new_client() -> "AsyncClient":
    from menlo_robot_sdk import AsyncClient

    api_key, rcs_url, _ = _menlo_config()
    return AsyncClient(rcs_url=rcs_url, api_key=api_key)


def _robot_key(session_id: str) -> str:
    return f"{_ROBOT_KEY_PREFIX}{session_id}"


async def get_stored_robot_id(session_id: str) -> str | None:
    redis = await get_redis_async()
    raw = await redis.get(_robot_key(session_id))
    return raw.decode() if isinstance(raw, bytes) else raw


async def set_stored_robot_id(session_id: str, robot_id: str) -> None:
    redis = await get_redis_async()
    await redis.set(_robot_key(session_id), robot_id, ex=_ROBOT_ID_TTL)


async def clear_stored_robot_id(session_id: str) -> None:
    redis = await get_redis_async()
    await redis.delete(_robot_key(session_id))


async def _create_robot(client: "AsyncClient", *, name: str, model: str | None) -> str:
    """Create a robot and return its id.

    Works around a menlo-robot-sdk 0.2.2 vs live-API mismatch: the SDK's
    ``RobotCreateResponse`` requires ``pin_code``, but the platform no longer
    returns it for virtual robots, so ``robots.create`` raises a
    ``ValidationError`` even though the robot WAS created. In that case we
    recover the id by finding the just-created robot (unique name) via
    ``robots.list`` — which uses the lenient ``RobotOut`` model.
    """
    resolved_model = model or _DEFAULT_ROBOT_MODEL
    try:
        created = await client.robots.create(name=name, model=resolved_model)
        return created.robot.id
    except ValidationError:
        logger.warning(
            "[menlo] robots.create response failed SDK validation; "
            "recovering robot id via list() by name %r",
            name,
        )
        listing = await client.robots.list(limit=50)
        match = next((r for r in listing.robots if r.name == name), None)
        if match is None:
            raise
        return match.id


async def _connect(client: "AsyncClient", robot_id: str) -> "MenloSession":
    """Join the robot's runtime room (browser viewer fills the rcw role)."""
    from menlo_robot_sdk import connect

    return await connect(
        client,
        robot_id,
        worker_names=[],  # no server-side worker — the browser viewer is the runtime
        rcw_identity_prefix=_RCW_IDENTITY_PREFIX,
        join_livekit=True,
    )


async def connect_new_robot(
    session_id: str, *, model: str | None = None, name: str
) -> LiveConnection:
    """Create a fresh robot, join its room, and persist it for this session."""
    client = _new_client()
    try:
        robot_id = await _create_robot(client, name=name, model=model)
        session = await _connect(client, robot_id)
    except BaseException:
        await _safe_aclose(client)
        raise

    await set_stored_robot_id(session_id, robot_id)
    conn = LiveConnection(robot_id, client, session)
    _LIVE[session_id] = conn
    logger.info("[menlo] connected robot %s for session %s", robot_id, session_id)
    return conn


async def resolve_connection(session_id: str) -> LiveConnection:
    """Return the live connection for this session, reconnecting if needed.

    Raises :class:`MenloNotConnectedError` when no robot is associated with the
    session (the copilot must call ``menlo_connect_robot`` first).
    """
    cached = _LIVE.get(session_id)
    if cached is not None:
        return cached

    robot_id = await get_stored_robot_id(session_id)
    if not robot_id:
        raise MenloNotConnectedError(
            "No robot is connected for this session. Call menlo_connect_robot first."
        )

    client = _new_client()
    try:
        session = await _connect(client, robot_id)
    except BaseException:
        await _safe_aclose(client)
        # A 404 means the robot/session was deleted out from under us — forget it
        # so the next connect starts clean.
        await _forget_if_gone(session_id)
        raise

    conn = LiveConnection(robot_id, client, session)
    _LIVE[session_id] = conn
    return conn


async def generate_viewer_url(conn: LiveConnection) -> str:
    """Mint a fresh room-key viewer link the user opens in Chrome."""
    from menlo_robot_sdk.experimental import generate_room_key_url

    _, _, viewer_url = _menlo_config()
    return await generate_room_key_url(
        conn.client, conn.robot_id, viewer_base_url=viewer_url
    )


async def disconnect_robot(session_id: str, *, delete_robot: bool = True) -> str | None:
    """Tear down the session's robot. Returns the robot_id that was removed."""
    robot_id = await get_stored_robot_id(session_id)
    conn = _LIVE.pop(session_id, None)

    if conn is not None:
        robot_id = conn.robot_id
        # disconnect() deletes the RCS session and closes the LiveKit room.
        await _safe_call(conn.session.disconnect())
        if delete_robot:
            await _safe_call(conn.client.robots.delete(conn.robot_id))
        await _safe_aclose(conn.client)
    elif robot_id and delete_robot:
        # No live handle in this process — still delete the robot record.
        client = _new_client()
        try:
            await _safe_call(client.robots.delete(robot_id))
        finally:
            await _safe_aclose(client)

    await clear_stored_robot_id(session_id)
    return robot_id


async def _forget_if_gone(session_id: str) -> None:
    """Drop the stored robot_id (best-effort) after a failed reconnect."""
    _LIVE.pop(session_id, None)
    await clear_stored_robot_id(session_id)


async def _safe_call(coro) -> None:
    try:
        await coro
    except Exception:
        logger.warning("[menlo] cleanup call failed", exc_info=True)


async def _safe_aclose(client: "AsyncClient") -> None:
    try:
        await client.aclose()
    except Exception:
        logger.warning("[menlo] client aclose failed", exc_info=True)
