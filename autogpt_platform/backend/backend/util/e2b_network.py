"""One door for every E2B box: where its egress is pinned.

Every ``AsyncSandbox.create`` and ``.connect`` in the backend goes through
``create_sandbox`` / ``connect_sandbox`` here; ``e2b_network_test`` fails on
any that does not.  That is the chokepoint at which a box's network is set:
the credential swap proxy (SECRT-2651) that every box will egress through,
so that the model never holds a real credential and nothing dials out
around the proxy.

Off by default.  With ``E2B_EGRESS_PROXY_ADDRESS`` empty, boxes are created
and reconnected exactly as before, egress direct.  With an address set,
every create pins the box's outbound TCP to that SOCKS5 proxy and every
reconnect re-applies it: ``update_network`` replaces the whole
configuration, so a box created before the address was set is pinned on
its first reconnect, and one created while it was set stays pinned only
because every reconnect says so again.  Do not set the address before the
proxy exists: E2B's proxy option fails closed, so a box pointed at nothing
has no egress at all, pip, npm and gh included.

The per-box proxy credential is how the proxy knows whose connection it is
serving, and that mapping is the tenant boundary on the proxy's side.  A
credential is minted for one running stretch of a box: fresh at create,
fresh again at every reconnect, the previous one forgotten.  The proxy
looks the username up (``e2b:egress:cred:<username>``) and checks the
secret against the stored digest; the record names the owner and the user
whose credentials may be swapped in.  A box cannot claim another owner:
usernames are random, and the secret is handed to E2B's host, which dials
the proxy, never to anything running inside the box.

Known gap, to be settled by the spike on SECRT-2615: E2B tunnels TCP only.
DNS and QUIC leave the box directly.  QUIC is turned off in the browsers by
policy in the image; DNS stays an accepted side channel.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import secrets
from typing import Any, Literal, Optional, TypeVar, cast

from e2b import AsyncSandbox
from e2b.sandbox.sandbox_api import (
    SandboxEgressProxyOpts,
    SandboxNetworkOpts,
    SandboxNetworkUpdate,
)
from pydantic import BaseModel, ConfigDict

from backend.data.redis_client import get_redis_async
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
# One JSON line per pin: which box, for whom, and which of the user's accounts
# its requests may be given.  Never a value; the proxy's own audit has the
# requests themselves.
audit_logger = logging.getLogger("backend.egress_audit")

_CREDENTIAL_KEY_PREFIX = "e2b:egress:cred:"
_BOX_KEY_PREFIX = "e2b:egress:box:"
# A credential lives as long as a paused box can (E2B's paused-sandbox
# lifetime); every reconnect replaces it anyway.
_CREDENTIAL_TTL = 48 * 3600
_KILL_TIMEOUT_SECONDS = 10
_LOCK_KEY_PREFIX = "e2b:egress:lock:"
# A rotation is one E2B call and three Redis writes; the TTL only matters if
# the holder dies mid-way.
_ROTATION_LOCK_TTL = 60
_ROTATION_LOCK_WAIT = 30

S = TypeVar("S", bound=AsyncSandbox)


class EgressOwner(BaseModel):
    """Whose box a connection comes from: what the proxy audits and swaps for.

    ``session`` and ``expert`` are CoPilot boxes (``SandboxOwner``); ``block``
    is a graph execution's, keyed by the user.  *user_id* is who the box runs
    for.  Only a CoPilot box gets that user's credentials swapped in
    (``swaps``): a block runs a graph someone else may have written, and a
    marketplace agent must not get to act with the GitHub account of whoever
    runs it.

    *providers* is the ceiling on which of the user's providers the box may
    use (the turn's ``CopilotPermissions``); ``None`` means every one.  It is
    recorded with the credential, and the backend's swap service refuses a
    value for any provider outside it.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["session", "expert", "block"]
    id: str
    user_id: Optional[str] = None
    providers: Optional[tuple[str, ...]] = None

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.id}"

    @property
    def swaps(self) -> bool:
        return self.kind != "block" and self.user_id is not None


class ProxyCredential(BaseModel):
    """What E2B's host presents to the proxy for one running stretch of a box.

    The secret is 256 random bits, not a password anyone chose: it goes into
    the SOCKS5 ``password`` field, but there is nothing to dictionary-attack,
    so the record keeps a plain SHA-256 digest of it, as for any API token.
    """

    model_config = ConfigDict(frozen=True)

    username: str
    secret: str


def proxy_address() -> Optional[str]:
    """The swap proxy's ``host:port``, or ``None`` when egress is left direct."""
    return Settings().config.e2b_egress_proxy_address.strip() or None


def secret_digest(secret: str) -> str:
    """What the record stores instead of the secret; the proxy compares to it."""
    return hashlib.sha256(secret.encode()).hexdigest()


async def create_sandbox(sandbox_cls: type[S], owner: EgressOwner, **kwargs: Any) -> S:
    """Create a box the way the caller asked, pinned to the proxy if one is set.

    *sandbox_cls* is the SDK class the caller uses (``e2b.AsyncSandbox`` or
    ``e2b_code_interpreter``'s); every other keyword goes to its ``create``.
    The credential is recorded before the box exists so that its very first
    connection already resolves at the proxy.
    """
    if "network" in kwargs:
        raise ValueError("A box's network is decided here, not by the caller")
    address = proxy_address()
    if address is None:
        return await sandbox_cls.create(**kwargs)
    credential = _mint()
    await _remember(credential, owner, sandbox_id=None)
    try:
        sandbox = await sandbox_cls.create(
            network=_network(address, credential), **kwargs
        )
    except BaseException:
        # No box will ever present this credential: do not leave it
        # resolving at the proxy for its TTL.
        with contextlib.suppress(BaseException):
            await _forget(credential.username)
        raise
    try:
        await _remember(credential, owner, sandbox_id=sandbox.sandbox_id)
        await _bind(sandbox.sandbox_id, credential.username)
    except BaseException:
        # The box is on the meter and its handle is about to be lost: kill
        # it rather than leak it, and take its credential with it.
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(sandbox.kill(), timeout=_KILL_TIMEOUT_SECONDS)
        with contextlib.suppress(BaseException):
            await _forget(credential.username)
        raise
    logger.info(
        "[E2B] Created %.12s for %s pinned to the egress proxy",
        sandbox.sandbox_id,
        owner.label,
    )
    _audit_pin("created", sandbox.sandbox_id, owner)
    return sandbox


async def connect_sandbox(
    sandbox_cls: type[S],
    sandbox_id: str,
    owner: EgressOwner,
    *,
    apply_network: bool = True,
    **kwargs: Any,
) -> S:
    """Reconnect to a box, re-pinning its egress under a fresh credential.

    *apply_network* is ``False`` for a connect that only pauses or kills the
    box: nothing will egress before it is gone, so there is nothing to pin.
    The new credential is recorded before the update and the old one is
    forgotten after it, so a failed update leaves the box on a credential
    that still resolves.

    The rotation is serialized per box.  Several turns share an expert's box
    and may reconnect at once; the order two ``update_network`` calls land at
    E2B is not the order their awaits return, so without the lock the box
    could end up presenting a credential the other reconnect just forgot, and
    E2B's proxy option fails closed.
    """
    # The SDK types ``connect`` as the base class; it returns *sandbox_cls*.
    sandbox = cast(S, await sandbox_cls.connect(sandbox_id, **kwargs))
    address = proxy_address()
    if address is None or not apply_network:
        return sandbox
    async with _rotation_lock(sandbox_id):
        credential = _mint()
        await _remember(credential, owner, sandbox_id=sandbox_id)
        try:
            await sandbox.update_network(_network_update(address, credential))
        except BaseException:
            # The box never got this credential: do not leave it resolving.
            with contextlib.suppress(BaseException):
                await _forget(credential.username)
            # ``connect`` has already resumed the box.  One that was pinned
            # before stays pinned, on its previous credential; one that never
            # was (it predates the proxy) is awake with direct egress and its
            # handle is about to be lost, so put it back to sleep.
            with contextlib.suppress(BaseException):
                if not await _bound_username(sandbox_id):
                    await asyncio.wait_for(
                        sandbox.pause(), timeout=_KILL_TIMEOUT_SECONDS
                    )
            raise
        await _bind(sandbox_id, credential.username)
    logger.info(
        "[E2B] Reconnected %.12s for %s, egress re-pinned", sandbox_id, owner.label
    )
    _audit_pin("reconnected", sandbox_id, owner)
    return sandbox


async def kill_sandbox(sandbox: AsyncSandbox) -> None:
    """Kill a box and revoke its proxy credential: the kill for any caller
    that holds the handle.  A kill that raises leaves the credential alone,
    since the box may still be running."""
    await sandbox.kill()
    await forget_sandbox(sandbox.sandbox_id)


async def forget_sandbox(sandbox_id: str) -> None:
    """Revoke a box's proxy credential: call when the box is paused or killed.

    Nothing will present it again (a resume mints a fresh one), and until its
    TTL it would keep naming this owner to the proxy.  Best effort: a failure
    here must not fail the pause or kill it follows.
    """
    # Not skipped while no proxy is configured: a box pinned before the
    # address was removed still has a credential on record.
    try:
        redis = await get_redis_async()
        username = await _bound_username(sandbox_id)
        if username:
            await redis.delete(_CREDENTIAL_KEY_PREFIX + username)
        await redis.delete(_BOX_KEY_PREFIX + sandbox_id)
    except Exception:
        logger.warning(
            "[E2B] Could not revoke the proxy credential of %.12s",
            sandbox_id,
            exc_info=True,
        )


async def credential_record(username: str) -> Optional[dict[str, Any]]:
    """What the proxy sees for a username: owner, user, box, secret digest."""
    redis = await get_redis_async()
    raw = await redis.get(_CREDENTIAL_KEY_PREFIX + username)
    if not raw:
        return None
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


async def recorded_providers(sandbox_id: str) -> Optional[tuple[str, ...]]:
    """The provider ceiling on the box's current egress record: what a re-pin
    that has no ceiling of its own to give keeps, so that it never widens one.

    ``None`` (every provider) only for a box with no record at all, which has
    no ceiling to keep.  If the record cannot be read, no provider: the safe
    side, until the next turn pins the box with its own.
    """
    try:
        username = await _bound_username(sandbox_id)
        record = await credential_record(username) if username else None
    except Exception:
        logger.warning(
            "[E2B] Could not read the ceiling of %.12s; keeping none",
            sandbox_id,
            exc_info=True,
        )
        return ()
    if record is None:
        return None
    providers = record.get("providers")
    return None if providers is None else tuple(providers)


def _audit_pin(event: str, sandbox_id: str, owner: EgressOwner) -> None:
    """Record whose credentials a box's requests may be given from now on:
    the per-box half of the audit trail, the proxy's lines being the
    per-request half."""
    audit_logger.info(
        json.dumps(
            {
                "event": event,
                "sandbox_id": sandbox_id,
                "owner": owner.label,
                "user_id": owner.user_id,
                "swaps": owner.swaps,
                "providers": (
                    sorted(owner.providers) if owner.providers is not None else "all"
                ),
            },
            sort_keys=True,
        )
    )


def _mint() -> ProxyCredential:
    return ProxyCredential(
        username=f"box-{secrets.token_hex(8)}", secret=secrets.token_urlsafe(32)
    )


def _network(address: str, credential: ProxyCredential) -> SandboxNetworkOpts:
    return {"egress_proxy": _proxy(address, credential)}


def _network_update(address: str, credential: ProxyCredential) -> SandboxNetworkUpdate:
    # The update replaces the whole configuration: this is the complete set.
    return {"egress_proxy": _proxy(address, credential)}


def _proxy(address: str, credential: ProxyCredential) -> SandboxEgressProxyOpts:
    return {
        "address": address,
        "username": credential.username,
        "password": credential.secret,
    }


async def _remember(
    credential: ProxyCredential, owner: EgressOwner, *, sandbox_id: Optional[str]
) -> None:
    redis = await get_redis_async()
    record = {
        "owner": owner.label,
        "user_id": owner.user_id,
        # Absent or false means the proxy swaps nothing for this box.
        "swaps": owner.swaps,
        # None: every provider.  Read by the backend, never by the box.
        "providers": list(owner.providers) if owner.providers is not None else None,
        "sandbox_id": sandbox_id,
        "secret_sha256": secret_digest(credential.secret),
    }
    await redis.set(
        _CREDENTIAL_KEY_PREFIX + credential.username,
        json.dumps(record),
        ex=_CREDENTIAL_TTL,
    )


async def _forget(username: str) -> None:
    redis = await get_redis_async()
    await redis.delete(_CREDENTIAL_KEY_PREFIX + username)


async def _bound_username(sandbox_id: str) -> Optional[str]:
    redis = await get_redis_async()
    raw = await redis.get(_BOX_KEY_PREFIX + sandbox_id)
    return raw.decode() if isinstance(raw, bytes) else raw


@contextlib.asynccontextmanager
async def _rotation_lock(sandbox_id: str):
    """One credential rotation per box at a time, across processes."""
    # Imported here: backend.executor's package import reaches back to the
    # blocks, which import this module.
    from backend.executor.cluster_lock import AsyncClusterLock

    lock = AsyncClusterLock(
        await get_redis_async(),
        _LOCK_KEY_PREFIX + sandbox_id,
        owner_id=secrets.token_hex(8),
        timeout=_ROTATION_LOCK_TTL,
    )
    deadline = asyncio.get_running_loop().time() + _ROTATION_LOCK_WAIT
    while await lock.try_acquire() != lock.owner_id:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"Another reconnect is still re-pinning sandbox {sandbox_id[:12]}"
            )
        await asyncio.sleep(0.2)
    try:
        yield
    finally:
        await lock.release()


async def _bind(sandbox_id: str, username: str) -> None:
    """Make *username* the box's current credential and forget its previous one."""
    redis = await get_redis_async()
    key = _BOX_KEY_PREFIX + sandbox_id
    previous = await _bound_username(sandbox_id)
    await redis.set(key, username, ex=_CREDENTIAL_TTL)
    if previous and previous != username:
        await redis.delete(_CREDENTIAL_KEY_PREFIX + previous)
