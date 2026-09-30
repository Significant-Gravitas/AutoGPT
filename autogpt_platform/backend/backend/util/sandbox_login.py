"""The login chain every E2B command runs first (``bash -l -c``), watched so an
edit to it can neither run unseen nor hide from the supervisor.

A sandbox's startup files are hashed before any agent action. The supervisor is
shown every one that differs since, and a command the platform runs itself skips
the user's files and refuses to run when ``/etc``'s have changed.
"""

import asyncio
import hashlib
import json
import logging
import shlex
from typing import Any

from e2b import AsyncSandbox, FileType, NotFoundException

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

USER_HOME = "/home/user"
# What `bash -l` reads for the user, and what stock ~/.profile and ~/.bashrc source.
USER_FILES = tuple(
    f"{USER_HOME}/{name}"
    for name in (".bash_profile", ".bash_login", ".profile", ".bashrc", ".bash_aliases")
)
# Root-owned, but the sandbox user has passwordless sudo.
SYSTEM_FILES = ("/etc/profile", "/etc/bash.bashrc")
PROFILE_D = "/etc/profile.d"
# A file past this is cut, which puts a judged call past every supervisor ceiling.
READ_CAP = 64_000

_NO_HOME = "/nonexistent"
_BASELINE_TTL_SECONDS = 30 * 24 * 3600
_READ_TIMEOUT_SECONDS = 5


class LoginChainChanged(RuntimeError):
    """``/etc``'s login files differ from the sandbox's baseline."""


class _TooLong(Exception):
    """A login file past ``READ_CAP``, which cannot be compared whole."""


async def run_internal(
    sandbox: AsyncSandbox,
    command: str,
    *,
    keep_home: bool = False,
    user: str | None = None,
    envs: dict[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """``sandbox.commands.run`` for a command the platform runs, not the agent.

    Every platform command in a chat's sandbox goes through here: ``bash -l`` finds
    no user files under ``_NO_HOME``, and a changed ``/etc`` chain refuses the run.
    ``keep_home`` hands the real home back to a program that reads it.
    """
    if changed := await changed_login_files(sandbox, system_only=True):
        raise LoginChainChanged(
            f"The sandbox's login files changed ({', '.join(changed)}), so the "
            "platform ran nothing in it."
        )
    if keep_home:
        home = "/root" if user == "root" else USER_HOME
        command = f"export HOME={shlex.quote(home)}; {command}"
    return await sandbox.commands.run(
        command, user=user, envs={**(envs or {}), "HOME": _NO_HOME}, **kwargs
    )


async def take_baseline(
    sandbox: AsyncSandbox, *, only_if_missing: bool = False
) -> None:
    """Called by whatever creates or reconnects a sandbox, before any agent action.
    Logged, not raised: without a baseline every login file reads as changed, so
    judged commands carry them all and internal ones refuse."""
    try:
        if only_if_missing:
            await ensure_baseline(sandbox)
        else:
            await record_baseline(sandbox)
    except Exception:
        logger.error(
            f"Could not record the login baseline for {sandbox.sandbox_id[:12]}",
            exc_info=True,
        )


async def record_baseline(sandbox: AsyncSandbox) -> None:
    """Hash the login files; called before any agent action touches the sandbox."""
    snapshot = await _snapshot(sandbox, system_only=False)
    digests: dict[str, str | None] = {}
    for path, raw in snapshot.items():
        if isinstance(raw, BaseException):
            raise RuntimeError(f"Could not read login file {path}") from raw
        digests[path] = _digest(raw)
    redis = await get_redis_async()
    await redis.set(_key(sandbox), json.dumps(digests), ex=_BASELINE_TTL_SECONDS)


async def ensure_baseline(sandbox: AsyncSandbox) -> None:
    """On a reconnect: a sandbox created before baselines existed gets one now."""
    redis = await get_redis_async()
    if await redis.expire(_key(sandbox), _BASELINE_TTL_SECONDS):
        return
    logger.warning(
        f"No login baseline for sandbox {sandbox.sandbox_id[:12]}; taking one"
    )
    await record_baseline(sandbox)


async def changed_login_files(
    sandbox: AsyncSandbox, *, system_only: bool = False
) -> dict[str, str | None]:
    """Each login file whose content differs from the baseline, by path, with None
    for one that could not be read. A new file counts; a deleted one runs nothing.
    Without a baseline every file counts: one taken now would bake a change in."""
    snapshot = await _snapshot(sandbox, system_only=system_only)
    raw_baseline = await (await get_redis_async()).get(_key(sandbox))
    if not raw_baseline:
        logger.warning(f"No login baseline for sandbox {sandbox.sandbox_id[:12]}")
    baseline: dict[str, str | None] = json.loads(raw_baseline) if raw_baseline else {}
    changed: dict[str, str | None] = {}
    for path, raw in snapshot.items():
        if isinstance(raw, BaseException):
            changed[path] = None
        elif raw is not None and _digest(raw) != baseline.get(path):
            changed[path] = judged_text(raw)
    return changed


async def read_capped(sandbox: AsyncSandbox, path: str) -> bytes | None:
    """At most ``READ_CAP + 1`` bytes of ``path``, streamed so a huge file is never
    held whole; None when it does not exist. Any other failure raises."""
    try:
        stream = await sandbox.files.read(
            path, format="stream", request_timeout=_READ_TIMEOUT_SECONDS
        )
        data = bytearray()
        async with stream:
            async for chunk in stream:
                data += chunk
                if len(data) > READ_CAP:
                    break
    except NotFoundException:
        return None
    return bytes(data[: READ_CAP + 1])


def judged_text(raw: bytes) -> str:
    """A file as the supervisor reads it."""
    if b"\0" in raw[:READ_CAP]:
        return "[binary file]"
    return raw[: READ_CAP + 1].decode("utf-8", errors="replace")


async def _snapshot(
    sandbox: AsyncSandbox, *, system_only: bool
) -> dict[str, bytes | BaseException | None]:
    """Every login file's bytes; None where it does not exist, the error where it
    could not be read."""
    listing: dict[str, bytes | BaseException | None] = {}
    try:
        entries = await asyncio.wait_for(
            sandbox.files.list(PROFILE_D, depth=1), _READ_TIMEOUT_SECONDS
        )
    except NotFoundException:
        entries = []
    except Exception as error:
        # Unlisted, a new file there could run unseen: read as unreadable.
        entries = []
        listing[PROFILE_D] = error
    profile_d = [entry.path for entry in entries if entry.type is not FileType.DIR]
    paths = [*SYSTEM_FILES, *profile_d, *([] if system_only else USER_FILES)]
    contents = await asyncio.gather(
        *(read_capped(sandbox, path) for path in paths), return_exceptions=True
    )
    # Only a prefix would be hashed, and a change past it would go unseen.
    return {
        **listing,
        **{
            path: (
                _TooLong(path)
                if isinstance(raw, bytes) and len(raw) > READ_CAP
                else raw
            )
            for path, raw in zip(paths, contents)
        },
    }


def _key(sandbox: AsyncSandbox) -> str:
    return f"copilot:sandbox:login-baseline:{sandbox.sandbox_id}"


def _digest(raw: bytes | None) -> str | None:
    return hashlib.sha256(raw).hexdigest() if raw is not None else None
