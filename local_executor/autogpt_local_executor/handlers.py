"""
Per-message handlers.

Each handler takes a parsed inbound message (one of the *Message types
from `protocol`) and returns the response message — never raises (any
internal error becomes an ERROR message with INTERNAL_ERROR).

Every FILE_* handler MUST call `assert_inside_jail()` before touching
the filesystem, in a worker thread like every other filesystem call
(`_jailed`, `_read_in_a_thread`): a stalled mount holds a stat for as long
as it stalls, and the event loop carries every session.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import contextvars
import hashlib
import json
import logging
import math
import mimetypes
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import weakref
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any, Literal, TypeVar

from . import local_llm_limits, platform_info, windows_jobs
from .audit import AuditWriter
from .computer_use import (
    APPROVAL_ARGUMENTS,
    ApprovalRequired,
    ArgumentRefused,
    CallNotStarted,
    ClipboardConcealedError,
    ClipboardWritebackRegistry,
    ComputerUseError,
    CuaDriverBridge,
    DriverSnapshot,
    DriverUnavailable,
    check_arguments,
    offered_tools,
)
from .computer_use.catalog import (
    is_aumid,
    linux_app_name_matches,
    linux_launch_uses_name,
    takes_session,
    windows_app_name_matches,
    without_session,
)
from .computer_use.concealed import ClipboardState, clipboard_state, marker_in
from .computer_use.mcp_client import INVALID_PARAMS, McpError
from .config import ShimConfig
from .directory_browser import _is_reparse_point
from .fsops import rename_no_replace
from .local_llm_limits import bounded_chunks, read_json_lines
from .path_jail import (
    PathJailError,
    assert_entry_inside_jail,
    assert_inside_jail,
    from_root,
    is_inside_jail,
    open_jail_file,
    write_jail_file,
)
from .protocol import (
    MAX_WEBSOCKET_MESSAGE_BYTES,
    MAX_WIRE_BASE64_CONTENT_BYTES,
    MAX_WIRE_TEXT_CONTENT_BYTES,
    AckMessage,
    ApplyRecordingReviewMessage,
    CommandResultMessage,
    CommandResultPayload,
    ComputerImageContent,
    ComputerTextContent,
    ComputerToolCallMessage,
    ComputerToolResultMessage,
    ComputerToolResultPayload,
    ComputerToolSpec,
    ComputerToolsRequestMessage,
    ComputerToolsResponseMessage,
    ComputerToolsResponsePayload,
    Encoding,
    ErrorCode,
    ErrorMessage,
    ExecuteCommandMessage,
    FileContentsMessage,
    FileContentsPayload,
    FileDeleteMessage,
    FileEntry,
    FileFormat,
    FileListMessage,
    FileListResponseMessage,
    FileListResponsePayload,
    FileMoveMessage,
    FileReadMessage,
    FileStatMessage,
    FileStatResponseMessage,
    FileStatResponsePayload,
    FileWriteMessage,
    LocalLLMCompletionChunkMessage,
    LocalLLMCompletionChunkPayload,
    LocalLLMCompletionMessage,
    LocalLLMCompletionResponseMessage,
    LocalLLMCompletionResponsePayload,
    LocalLLMFinishReason,
    LocalLLMTokensUsage,
    RecordingConsentResultMessage,
    RecordingConsentResultPayload,
    RecordingDataMessage,
    RecordingDataPayload,
    RecordingFetchMessage,
    RecordingReviewAppliedMessage,
    RecordingReviewAppliedPayload,
    RecordingStartedMessage,
    RecordingStartedPayload,
    RecordingStepMessage,
    RecordingStepPayload,
    RecordingSummaryMessage,
    RecordingSummaryPayload,
    RequestRecordingConsentMessage,
    StartRecordingMessage,
    StopRecordingMessage,
    make_ack,
    make_error,
    new_id,
    now_ts,
)
from .recording import (
    A11yEnricher,
    CaptureSource,
    ConsentBroker,
    ConsentError,
    RecordingError,
    RecordingSession,
    ScreenshotActionFloor,
    probe_interpretation_route,
    request_user_consent,
)
from .recording.consent import CONSENT_TOKEN_TTL_SECONDS
from .recording.crypto import RecordingCipher

logger = logging.getLogger(__name__)


class _SlidingWindowRateLimiter:
    def __init__(self, limit: int, *, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()
        if len(self._events) >= self.limit:
            return False
        self._events.append(now)
        return True

    def available(self) -> int:
        """How many more the window takes now, without taking one."""
        cutoff = time.monotonic() - self.window_seconds
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()
        return max(self.limit - len(self._events), 0)


class _BoundedCombinedOutput:
    """Retain a bounded stdout+stderr prefix while callers keep draining pipes."""

    _TRUNCATION_MARKER = b"\n[command output truncated]\n"

    def __init__(self, limit: int) -> None:
        self.limit = max(limit, 0)
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.truncated = False

    def feed(self, channel: Literal["stdout", "stderr"], chunk: bytes) -> None:
        remaining = self.limit - len(self.stdout) - len(self.stderr)
        kept = chunk[: max(remaining, 0)]
        getattr(self, channel).extend(kept)
        if len(kept) != len(chunk):
            self.truncated = True

    def finish(self) -> tuple[bytes, bytes, bool]:
        if self.truncated and self.limit:
            marker = self._TRUNCATION_MARKER[: self.limit]
            excess = len(self.stdout) + len(self.stderr) + len(marker) - self.limit
            if excess > 0:
                trim_stdout = min(excess, len(self.stdout))
                if trim_stdout:
                    del self.stdout[-trim_stdout:]
                    excess -= trim_stdout
            if excess > 0:
                del self.stderr[-min(excess, len(self.stderr)) :]
            self.stderr.extend(marker)
        return bytes(self.stdout), bytes(self.stderr), self.truncated


# ── Helpers ──────────────────────────────────────────────────────────────────


def _jail_error_to_message(msg_id: str, exc: PathJailError) -> ErrorMessage:
    return make_error(msg_id, exc.code, exc.message)


def _ok_result(duration_ms: int, *, exit_code: int | None = None) -> dict:
    return {"ok": True, "exit_code": exit_code, "duration_ms": duration_ms, "error_code": None}


# The audit `error_code` of a command stopped because its session ended or
# was revoked. Not a wire ErrorCode: no reply is sent for it.
_CANCELLED_ERROR_CODE = "CANCELLED"


def _err_result(duration_ms: int, error_code: str, *, exit_code: int | None = None) -> dict:
    return {
        "ok": False,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "error_code": error_code,
    }


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _sendable(text: str) -> bool:
    """Whether `text` can go on the wire: JSON is UTF-8."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _jailed_link_target_mode(link: Path, allowed_root: str | os.PathLike[str]) -> int | None:
    """The mode of what `link` points to, when that's inside the jail; None
    for a target outside it, a missing one, or one that can't be stat'ed."""
    if not is_inside_jail(link, allowed_root):
        return None
    try:
        return link.stat().st_mode
    except OSError:
        return None


# How long a command's process group has to exit after SIGTERM before SIGKILL.
_TERMINATE_GRACE_SECONDS = 2.0
# How long a command's output is still read for its result once it has
# exited: a child it left running can hold its output pipes open, and what it
# writes after that is read and dropped (CommandHandler._keep_draining).
_OUTPUT_DRAIN_SECONDS = 2.0
# The most programs left running by commands whose output is read and
# dropped so, each holding two of the shim's file descriptors: a launchd
# agent may open 256, a systemd service 1024. Past it, the pipes of the
# program whose command went over are closed.
_MAX_OUTPUT_DRAINS = 32
# How long a Windows command has to exit after CTRL_BREAK_EVENT before its
# job, every process it started, is ended.
_WINDOWS_BREAK_GRACE_SECONDS = 2.0
# Each running Windows command's job (windows_jobs), held from its start
# until its request is done.
_command_jobs: weakref.WeakKeyDictionary[asyncio.subprocess.Process, windows_jobs.CommandJob] = (
    weakref.WeakKeyDictionary()
)
# The most entries one FILE_LIST returns, whatever max_entries asks for.
MAX_FILE_LIST_ENTRIES = 10_000
# The most directory entries one FILE_LIST looks at. A glob that matches
# little would otherwise walk a whole tree, holding a request slot and a
# worker thread until the walk ended.
MAX_FILE_LIST_SCANNED = 100_000
# The most bytes of entries one FILE_LIST reply carries. Long paths can take
# 10,000 entries past the frame limit, which would refuse the whole reply as
# FILE_TOO_LARGE; what is left is room for the envelope.
FILE_LIST_REPLY_BUDGET_BYTES = MAX_WEBSOCKET_MESSAGE_BYTES - 64 * 1024


async def _terminate_process_group(pgid: int) -> None:
    """SIGTERM a command's process group, then SIGKILL whatever is left of it.

    The group is signalled by its ID, never through the command's own process,
    which may exit on SIGTERM while members that ignore it live on.

    Cancelled during the grace period (its own task cancelled, as the shim's
    event loop ends), it sends SIGKILL then rather than never.
    """
    if not _signal_process_group(pgid, signal.SIGTERM):
        return
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    try:
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            if not _signal_process_group(pgid, 0):
                return
    except asyncio.CancelledError:
        _signal_process_group(pgid, signal.SIGKILL)
        raise
    _signal_process_group(pgid, signal.SIGKILL)


async def _exited(proc: asyncio.subprocess.Process) -> None:
    """Return once `proc` has exited, whether or not its output pipes have
    closed. Process.wait() returns only once they have too, and a child the
    command started can keep them open (a server started in the
    background, say), so a command that had exited was waited for until
    its timeout, and stopped with the child. The exit is seen by its return
    code, looked at every 50 ms while wait() hasn't returned."""
    waiting = asyncio.ensure_future(proc.wait())
    try:
        while not waiting.done() and proc.returncode is None:
            await asyncio.wait({waiting}, timeout=0.05)
    finally:
        if not waiting.done():
            waiting.cancel()


async def _discard_until_closed(streams: tuple[asyncio.StreamReader, ...]) -> None:
    """Read what `streams` carry and drop it, until each one closes."""

    async def discard(stream: asyncio.StreamReader) -> None:
        while await stream.read(64 * 1024):
            pass

    await asyncio.gather(*(discard(stream) for stream in streams), return_exceptions=True)


def _close_output_pipes(proc: asyncio.subprocess.Process) -> None:
    """Close the shim's ends of a command's stdout and stderr now, through
    the transport `proc` holds, so their descriptors are released at once,
    not whenever the transport is collected: a program holding the other
    ends fails its next write to them."""
    transport = getattr(proc, "_transport", None)
    if not isinstance(transport, asyncio.SubprocessTransport):
        return
    for fd in (1, 2):
        pipe = transport.get_pipe_transport(fd)
        if pipe is not None:
            pipe.close()


def _end_windows_command(
    proc: asyncio.subprocess.Process, job: windows_jobs.CommandJob | None
) -> None:
    """End a Windows command and every process it started (its job), or the
    command alone where the job can't be ended: TerminateProcess ends one
    process. Every command that runs has a job (_spawn_argv)."""
    if job is not None:
        try:
            job.terminate()
            return
        except OSError:
            logger.warning("A command's job could not be ended", exc_info=True)
    if proc.returncode is None:
        with contextlib.suppress(OSError):
            proc.kill()


def _signal_process_group(pgid: int, sig: int) -> bool:
    """Send `sig` to a process group; False once no process is left in it."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Only members this user cannot signal are left (a setuid program).
        return True
    return True


def _resolved_root(config: ShimConfig) -> Path:
    return Path(os.path.realpath(config.allowed_root))


def _is_link(path: Path) -> bool:
    """A symlink, or a Windows junction: removing either leaves its target."""
    isjunction = getattr(os.path, "isjunction", None)  # Python 3.12+
    return path.is_symlink() or bool(isjunction and isjunction(path))


def _is_own_directory(path: Path) -> bool:
    """A directory itself. FILE_DELETE removes a link to one as a file,
    leaving the directory it links to."""
    return path.is_dir() and not _is_link(path)


_T = TypeVar("_T")


class _ClipboardCheckOutOfTime(TimeoutError):
    """A foreign clipboard read whose deadline passed before its checks for
    a password manager's mark were done: what it read is withheld. The
    driver ran nothing past the deadline, so it isn't stopped."""


class _NeverStarted(Exception):
    """The job was cancelled while it still waited for a worker thread."""


# Set for a request whose session is ending, or was revoked, before its task
# is cancelled (the daemon's _cancel_dispatch_requests). A cancel reaches the
# task only when it next runs, and a worker thread freed up in between would
# start a change still queued for one: a write, a move or a delete after
# revocation. Checked as the thread starts, so a change already running
# finishes and is recorded, and one still queued never starts.
work_refused: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "work_refused", default=None
)


async def _run_to_completion(func: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Run a side effect in a worker thread, and finish it even if the task
    awaiting it is cancelled (SESSION_REVOKED, session teardown).

    The thread can't be stopped once it starts, so a cancelled task would
    let the change land unseen: the audit record its handler writes next
    would never be written. The cancellation is held until the thread is
    done, then dropped, so the handler records what happened. Its reply
    goes nowhere: a revoked session refuses sends and a closed one fails
    them. A job still waiting for a thread when the cancel lands, or when
    the daemon refuses its request's work (work_refused) just before the
    cancel, never starts, and the task stays cancelled: nothing ran, so
    there is nothing to record. Requests still queued for a slot never get
    here either.
    """
    cancelled = threading.Event()
    refused = work_refused.get()

    def run() -> _T:
        if cancelled.is_set() or (refused is not None and refused.is_set()):
            raise _NeverStarted
        return func(*args, **kwargs)

    work = asyncio.ensure_future(asyncio.to_thread(run))
    held = 0
    try:
        while True:
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                # Only `work` itself being cancelled (the loop shutting down)
                # ends the wait; a change that finished as the cancel landed
                # is returned.
                if work.cancelled():
                    held = 0  # the cancellation goes on, so it stays counted
                    raise
                cancelled.set()
                held += 1
            except _NeverStarted:
                held = 0
                raise asyncio.CancelledError from None
    finally:
        # Released however the thread ended, returned or raised: a held
        # cancel left counted would mislead asyncio.timeout and TaskGroup.
        task = asyncio.current_task()
        if task is not None:
            for _ in range(held):
                task.uncancel()


async def _jailed(
    requested_path: str | os.PathLike[str], allowed_root: str | os.PathLike[str]
) -> Path:
    """`assert_inside_jail`, in a worker thread: resolving a path stats each
    folder on the way, and a mount that stalls (NFS, FUSE, a cloud-synced
    folder) holds a stat for as long as it stalls. On the event loop it would
    hold every session's work and keepalive with it."""
    return await _read_in_a_thread(assert_inside_jail, requested_path, allowed_root)


async def _read_in_a_thread(func: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Run a read (FILE_READ, FILE_STAT, FILE_LIST, a request's path checks)
    in a worker thread, and keep its task until the thread is done even if
    the task is cancelled meanwhile, then let the cancel go on. The thread can't be stopped, and a
    request holds its machine slot until its task ends: one let go as the
    cancel landed, with its read still blocked (on a network filesystem,
    say), let repeated disconnects pile up reads past the limit, holding the
    worker threads every chat's requests share."""
    return await _finish_then_cancel(asyncio.to_thread(func, *args, **kwargs))


async def _finish_then_cancel(work: Awaitable[_T]) -> _T:
    """Await `work` to the end even if this task is cancelled meanwhile, then
    let the cancellation go on.

    For the audit record of a change that has happened: the write first
    waits for the writer's lock, and a cancel landing there would stop it,
    leaving the change with no record. Unlike `_run_to_completion`, the
    cancellation isn't dropped, so the handler stops once its record is in.
    """
    task = asyncio.ensure_future(work)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except Exception:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
    if cancelled:
        raise asyncio.CancelledError
    return result


class _CancelledOnceStarted(asyncio.CancelledError):
    """A request cancelled while its command started, once the command was
    running (POSIX): the request stops it and records it, as one cancelled
    later is."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        super().__init__()
        self.proc = proc


async def _started_through_cancel(
    starting: Awaitable[asyncio.subprocess.Process],
) -> asyncio.subprocess.Process:
    """The process `starting` creates, though the task waiting for it is
    cancelled meanwhile: it then raises _CancelledOnceStarted, holding it.

    A POSIX command runs from the moment it is created, and asyncio connects
    its output pipes after that. Cancelled there, asyncio kills the command
    alone, so what it had started in its process group ran on, with no
    record, and the request never had the process to stop. A start that
    fails started nothing to stop, and the cancel goes on."""
    task = asyncio.ensure_future(starting)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if task.cancelled():
            raise
        while not task.done():
            # Cancelled again: the start still has a process to hand over.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait({task})
        if task.cancelled() or task.exception() is not None:
            raise
        raise _CancelledOnceStarted(task.result()) from None


async def _audit_jail_violation(
    audit: AuditWriter | None,
    op: str,
    raw_path: str,
    exc: PathJailError,
) -> None:
    """Mirror a path-jail rejection into the audit log as JAIL_VIOLATION."""
    if audit is None:
        return
    try:
        await audit.jail_violation(exc.code, raw_path, op=op)
    except Exception:
        logger.debug("Failed to write JAIL_VIOLATION audit record", exc_info=True)


def _safe_env_baseline() -> dict[str, str]:
    """Per CROSS_PLATFORM.md "Environment variables" → safe-env baseline row."""
    plat = platform_info.detect_platform()
    if plat == "windows":
        keep = {
            "USERPROFILE",
            "HOMEDRIVE",
            "HOMEPATH",
            "PATH",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "COMSPEC",
            "OS",
            "WINDIR",
            "LOCALAPPDATA",
            "APPDATA",
        }
        # Case-fold on Windows (env names are case-insensitive in Win32).
        out: dict[str, str] = {}
        for k, v in os.environ.items():
            if k.upper() in keep:
                out[k] = v
        return out
    keep = {"HOME", "PATH", "LANG", "TZ", "TMPDIR", "USER", "SHELL"}
    out = {k: v for k, v in os.environ.items() if k in keep or k.startswith("LC_")}
    return out


def _merge_env(base: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    """Merge wire-supplied env into the safe baseline.

    On Windows, env-var names are case-insensitive: incoming "Path" should
    replace the baseline's "PATH", not coexist. We case-fold names by
    treating uppercase as canonical when there's a collision.
    """
    if platform_info.detect_platform() != "windows":
        return {**base, **extra}
    # Build a case-folded index over the base so we can drop colliding keys.
    folded: dict[str, str] = {k.upper(): k for k in base}
    result = dict(base)
    for k, v in extra.items():
        existing = folded.get(k.upper())
        if existing is not None:
            del result[existing]
        result[k] = v
    return result


# ── EXECUTE_COMMAND ──────────────────────────────────────────────────────────


class _CommandConcurrency:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0

    def acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active = max(self.active - 1, 0)


class CommandHandler:
    def __init__(
        self,
        config: ShimConfig,
        audit: AuditWriter | None = None,
    ) -> None:
        self.config = config
        self.audit = audit
        self._command_rate = _SlidingWindowRateLimiter(config.max_commands_per_minute)
        self._concurrency = _CommandConcurrency(config.max_concurrent_commands)
        # The reads of what commands left running write (_keep_draining),
        # each held with its command's process until its pipes close, at
        # most _MAX_OUTPUT_DRAINS of them.
        self._output_drains: dict[asyncio.Task[None], asyncio.subprocess.Process] = {}
        self._owns_output_drains = True

    def share_with(self, other: CommandHandler) -> None:
        """Make `other`, a child session's handler, use this one's command
        limits, and leave the reads of what its commands left running to
        this one: a server a chat's command started outlives the chat."""
        other._command_rate = self._command_rate
        other._concurrency = self._concurrency
        other._output_drains = self._output_drains
        other._owns_output_drains = False

    async def close(self) -> None:
        """End the reads of what commands left running, if this handler holds
        them: the shim is stopping, and a program that never closes its
        output (a server) would keep them going until the event loop ends."""
        if not self._owns_output_drains:
            return
        reads = tuple(self._output_drains)
        for read in reads:
            read.cancel()
        await asyncio.gather(*reads, return_exceptions=True)

    def _keep_draining(self, proc: asyncio.subprocess.Process) -> None:
        """Go on reading a command's stdout and stderr once its result is
        made, dropping what they carry, until they close: a program it left
        running (a server started in the background) still holds them.
        Unread, they would fill, and the program would stop on its next
        write, or end on a broken pipe once they were collected. The read is
        held with `proc`, which holds the pipes, until it ends.

        Each read keeps two of the shim's file descriptors open, and a
        command that starts a program in the background, run again and
        again, would use them all up, so at most _MAX_OUTPUT_DRAINS programs'
        output is read. Past that, the pipes of the program whose command
        went over are closed, so the programs already read keep running, and
        that one's next write to them fails."""
        reads = self._output_drains
        if len(reads) >= _MAX_OUTPUT_DRAINS:
            logger.warning(
                "A command left a program holding its output while the output of %d "
                "others was read, the most there may be: its pipes are closed, so its "
                "next write to them fails. A program meant to outlive its command "
                "should send its output to a file.",
                len(reads),
            )
            _close_output_pipes(proc)
            return
        streams = tuple(stream for stream in (proc.stdout, proc.stderr) if stream is not None)
        read = asyncio.create_task(_discard_until_closed(streams))
        reads[read] = proc
        read.add_done_callback(lambda done: reads.pop(done, None))

    async def handle(self, msg: ExecuteCommandMessage) -> CommandResultMessage | ErrorMessage:
        if not self.config.enable_shell:
            return make_error(
                msg.id,
                ErrorCode.CAPABILITY_NOT_GRANTED,
                "Shell execution is disabled. Restart the shim with --enable-shell to opt in.",
            )
        if not self._command_rate.allow():
            await self._audit(
                msg.id,
                msg.payload,
                cwd=msg.payload.cwd or str(self.config.allowed_root),
                result=_err_result(0, ErrorCode.SHIM_OVERLOADED.value),
            )
            return make_error(
                msg.id,
                ErrorCode.SHIM_OVERLOADED,
                f"Command rate limit exceeded ({self.config.max_commands_per_minute}/minute).",
            )
        if not self._concurrency.acquire():
            await self._audit(
                msg.id,
                msg.payload,
                cwd=msg.payload.cwd or str(self.config.allowed_root),
                result=_err_result(0, ErrorCode.SHIM_OVERLOADED.value),
            )
            return make_error(
                msg.id,
                ErrorCode.SHIM_OVERLOADED,
                "Concurrent command limit exceeded.",
            )
        try:
            return await self._execute(msg)
        finally:
            self._concurrency.release()

    async def _execute(self, msg: ExecuteCommandMessage) -> CommandResultMessage | ErrorMessage:
        payload = msg.payload
        start = time.monotonic()

        # Validate command vs argv mutual exclusion.
        if (payload.command is None) == (payload.argv is None):
            err = "Exactly one of `command` or `argv` must be set."
            await self._audit(
                msg.id,
                payload,
                cwd=payload.cwd or str(self.config.allowed_root),
                result=_err_result(_elapsed_ms(start), ErrorCode.INTERNAL_ERROR.value),
            )
            return make_error(msg.id, ErrorCode.INTERNAL_ERROR, err)

        cwd_str = payload.cwd or str(self.config.allowed_root)
        try:
            # No cwd is the root itself, `.` from it: the jail takes a relative
            # path from the root, so a relative root named again would be joined
            # to itself (workspace/workspace).
            cwd = await _jailed(payload.cwd or ".", self.config.allowed_root)
        except PathJailError as exc:
            await _audit_jail_violation(self.audit, "EXECUTE_COMMAND", cwd_str, exc)
            return _jail_error_to_message(msg.id, exc)

        timeout = min(
            payload.timeout_seconds or self.config.command_timeout_seconds,
            self.config.command_timeout_seconds,
        )
        env = _merge_env(_safe_env_baseline(), payload.env or {})
        # Always force UTF-8 on Python subprocess output to keep stdout/stderr decodable.
        env["PYTHONIOENCODING"] = "utf-8"

        plat = platform_info.detect_platform()
        creation_flags = 0
        new_session = False
        if plat == "windows":
            # CREATE_NEW_PROCESS_GROUP so we can deliver CTRL_BREAK_EVENT.
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            # New session = new pgid so we can kill the whole tree. Not
            # preexec_fn=os.setsid: Python code run in the forked child of a
            # process with threads (the jail's checks run in worker threads)
            # can deadlock it before exec.
            new_session = True

        exit_code = -1

        try:
            if payload.argv is not None:
                proc = await self._spawn_argv(
                    payload.argv,
                    cwd=str(cwd),
                    env=env,
                    creation_flags=creation_flags,
                    new_session=new_session,
                )
            else:
                resolved = platform_info.resolve_shell(payload.shell.value)
                if resolved is None:
                    await self._audit(
                        msg.id,
                        payload,
                        cwd=str(cwd),
                        result=_err_result(_elapsed_ms(start), ErrorCode.SHELL_NOT_AVAILABLE.value),
                    )
                    return make_error(
                        msg.id,
                        ErrorCode.SHELL_NOT_AVAILABLE,
                        f"Shell {payload.shell.value!r} is not available on this OS.",
                    )
                shell_exe, shell_args = resolved
                # On Windows, ensure UTF-8 codepage for the subprocess.
                full_command = payload.command or ""
                if plat == "windows" and shell_exe.lower().endswith("cmd.exe"):
                    full_command = f"chcp 65001 >NUL & {full_command}"
                proc = await self._spawn_argv(
                    [shell_exe, *shell_args, full_command],
                    cwd=str(cwd),
                    env=env,
                    creation_flags=creation_flags,
                    new_session=new_session,
                )
        except _CancelledOnceStarted as cancelled:
            # Cancelled as its command started: stopped and recorded as one
            # cancelled while it runs is (below).
            await _finish_then_cancel(
                self._stop_cancelled(
                    cancelled.proc, msg.id, payload, cwd=str(cwd), env=env, start=start
                )
            )
            raise
        except FileNotFoundError as exc:
            await self._audit(
                msg.id,
                payload,
                cwd=str(cwd),
                result=_err_result(_elapsed_ms(start), ErrorCode.SHELL_NOT_AVAILABLE.value),
            )
            return make_error(
                msg.id,
                ErrorCode.SHELL_NOT_AVAILABLE,
                f"Executable not found: {exc}",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to spawn subprocess")
            await self._audit(
                msg.id,
                payload,
                cwd=str(cwd),
                result=_err_result(_elapsed_ms(start), ErrorCode.INTERNAL_ERROR.value),
            )
            return make_error(msg.id, ErrorCode.INTERNAL_ERROR, str(exc))

        try:
            stdout_b, stderr_b, timed_out, output_truncated = await self._collect_process_output(
                proc,
                timeout=timeout,
            )
        except asyncio.CancelledError:
            # Its session ended or was revoked. The stop runs as one task that
            # a further cancel (a detach and a revocation both stopping the
            # chat) waits for: one landing in the grace period ended the stop
            # before SIGKILL, and a command that ignores SIGTERM ran on after
            # its session, with no record.
            await _finish_then_cancel(
                self._stop_cancelled(proc, msg.id, payload, cwd=str(cwd), env=env, start=start)
            )
            raise
        finally:
            # Done with the command, however its request ended: what it left
            # running keeps running, as on Linux and macOS.
            job = _command_jobs.pop(proc, None)
            if job is not None:
                job.close()
        exit_code = proc.returncode if proc.returncode is not None else -1

        elapsed_ms = _elapsed_ms(start)
        duration = elapsed_ms / 1000.0

        if timed_out:
            result = _err_result(elapsed_ms, ErrorCode.COMMAND_TIMEOUT.value, exit_code=exit_code)
        else:
            result = _ok_result(elapsed_ms, exit_code=exit_code)
        result["output_truncated"] = output_truncated
        await self._audit(msg.id, payload, cwd=str(cwd), result=result, env=env)

        return CommandResultMessage(
            id=msg.id,
            ts=now_ts(),
            payload=CommandResultPayload(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=exit_code,
                timed_out=timed_out,
                duration_seconds=round(duration, 3),
                output_truncated=output_truncated,
            ),
        )

    async def _collect_process_output(
        self,
        proc: asyncio.subprocess.Process,
        *,
        timeout: float,
    ) -> tuple[bytes, bytes, bool, bool]:
        """Drain stdout/stderr concurrently while retaining a shared bounded prefix."""
        output = _BoundedCombinedOutput(
            min(self.config.max_file_size_bytes, MAX_WIRE_TEXT_CONTENT_BYTES)
        )
        stdout = getattr(proc, "stdout", None)
        stderr = getattr(proc, "stderr", None)

        # Lightweight process doubles use communicate(); real asyncio
        # subprocesses expose StreamReaders and take the bounded drain path.
        if not hasattr(stdout, "read") or not hasattr(stderr, "read"):
            timed_out = False
            stdout_b = stderr_b = b""
            try:
                async with asyncio.timeout(timeout):
                    stdout_b, stderr_b = await proc.communicate()
            except TimeoutError:
                timed_out = True
                await self._terminate_tree(proc)
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(2.0):
                        stdout_b, stderr_b = await proc.communicate()
            output.feed("stdout", stdout_b)
            output.feed("stderr", stderr_b)
            kept_stdout, kept_stderr, truncated = output.finish()
            return kept_stdout, kept_stderr, timed_out, truncated

        async def drain(channel: Literal["stdout", "stderr"], stream: Any) -> None:
            while chunk := await stream.read(64 * 1024):
                output.feed(channel, chunk)

        drain_tasks = [
            asyncio.create_task(drain("stdout", stdout)),
            asyncio.create_task(drain("stderr", stderr)),
        ]
        timed_out = False
        try:
            try:
                # The command's exit, not its pipes' closing (_exited): a
                # command that has exited isn't timed out, and what it left
                # running keeps running, as on any other exit. What is still
                # in the pipes is read for _OUTPUT_DRAIN_SECONDS (below).
                # Both waits are under asyncio.timeout, not wait_for: Python
                # 3.11's wait_for drops a cancel that lands as the exit is
                # seen, or as the read ends, and a session revoked then left
                # the command recorded as done, and what it started running.
                async with asyncio.timeout(timeout):
                    await _exited(proc)
            except TimeoutError:
                timed_out = True
                await self._terminate_tree(proc)
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(2.0):
                        await _exited(proc)
            try:
                async with asyncio.timeout(_OUTPUT_DRAIN_SECONDS):
                    await asyncio.gather(*drain_tasks, return_exceptions=True)
            except TimeoutError:
                for task in drain_tasks:
                    task.cancel()
                await asyncio.gather(*drain_tasks, return_exceptions=True)
                # A program the command left running holds the pipes: the
                # result has what came by now, and what comes later is read
                # and dropped, never kept, or, past _MAX_OUTPUT_DRAINS such
                # programs, the pipes are closed.
                self._keep_draining(proc)
        except asyncio.CancelledError:
            for task in drain_tasks:
                task.cancel()
            await asyncio.gather(*drain_tasks, return_exceptions=True)
            raise
        kept_stdout, kept_stderr, truncated = output.finish()
        return kept_stdout, kept_stderr, timed_out, truncated

    async def _stop_cancelled(
        self,
        proc: asyncio.subprocess.Process,
        request_id: str,
        payload,
        *,
        cwd: str,
        env: dict[str, str],
        start: float,
    ) -> None:
        """Stop a cancelled request's command, wait for it to exit, and
        record it as one that was stopped: it ran."""
        await self._terminate_tree(proc)
        wait = getattr(proc, "wait", None)
        if callable(wait):
            try:
                await asyncio.wait_for(wait(), timeout=2.0)
            except (TimeoutError, ProcessLookupError):
                pass
        await self._audit(
            request_id,
            payload,
            cwd=cwd,
            result=_err_result(
                _elapsed_ms(start), _CANCELLED_ERROR_CODE, exit_code=proc.returncode
            ),
            env=env,
        )

    @staticmethod
    async def _spawn_argv(
        argv: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        creation_flags: int,
        new_session: bool,
    ) -> asyncio.subprocess.Process:
        # On Windows the command starts suspended, and runs once it is in a
        # job of its own, which a stop ends whole (windows_jobs).
        in_job = sys.platform == "win32"
        starting = asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creation_flags | (windows_jobs.CREATE_SUSPENDED if in_job else 0),
            start_new_session=new_session,
        )
        if in_job:
            # Cancelled while it starts, asyncio ends the command while it is
            # still suspended: nothing of it ran.
            proc = await starting
        else:
            proc = await _started_through_cancel(starting)
        if in_job and isinstance(proc, asyncio.subprocess.Process):
            try:
                job = windows_jobs.run_in_job(proc.pid)
            except OSError:
                # Still suspended: it never ran. Ended, and waited for,
                # before the request is refused: without a job its stop could
                # end it alone, and what it started would run on.
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(2.0):
                        await proc.wait()
                raise
            _command_jobs[proc] = job
        return proc

    @staticmethod
    async def _terminate_tree(proc: asyncio.subprocess.Process) -> None:
        """Graceful → forceful kill per CROSS_PLATFORM.md "Process control" row."""
        plat = platform_info.detect_platform()
        try:
            if plat == "windows":
                job = _command_jobs.get(proc)
                if proc.returncode is None:
                    # CTRL_BREAK to the process group, then end the job.
                    try:
                        proc.send_signal(getattr(signal, "CTRL_BREAK_EVENT"))
                    except (AttributeError, OSError):
                        pass
                    try:
                        await asyncio.sleep(_WINDOWS_BREAK_GRACE_SECONDS)
                    except asyncio.CancelledError:
                        # Cut short, as _terminate_process_group is: ended now.
                        _end_windows_command(proc, job)
                        raise
                elif job is None or not job.has_processes():
                    # Nothing it started is left, or nothing can reach it:
                    # the command's own handle can't once it has exited.
                    return
                # What is left: the command, and the processes it started
                # that outlived the break or the command itself.
                _end_windows_command(proc, job)
            else:
                # setsid at spawn made the command its group's leader, so the
                # group's ID is the command's PID. Signalled even once the
                # command has exited: a child it started can live on in the
                # group, holding the output pipes the shim reads, and the ID
                # isn't given to another process while the group has members.
                await _terminate_process_group(proc.pid)
        except Exception:  # pragma: no cover - best-effort
            logger.exception("Error terminating subprocess tree")

    async def _audit(
        self,
        request_id: str,
        payload,
        *,
        cwd: str,
        result: dict,
        env: dict[str, str] | None = None,
    ) -> None:
        """Emit an EXECUTE_COMMAND record. Never logs env *values* — only the
        list of keys, per AUDIT_LOG.md "What's NEVER logged"."""
        if self.audit is None:
            return
        # Prefer the merged env if we got that far; otherwise the wire env.
        env_keys = sorted((env or payload.env or {}).keys())
        details = {
            "command": payload.command,
            "argv": payload.argv,
            "shell": payload.shell.value,
            "cwd": cwd,
            "env_keys": env_keys,
            "timeout_seconds": min(
                payload.timeout_seconds or self.config.command_timeout_seconds,
                self.config.command_timeout_seconds,
            ),
        }
        try:
            await _finish_then_cancel(
                self.audit.write(
                    "EXECUTE_COMMAND",
                    request_id=request_id,
                    details=details,
                    result=result,
                )
            )
        except Exception:
            logger.debug("Failed to write EXECUTE_COMMAND audit record", exc_info=True)


# ── FILE_* handlers ──────────────────────────────────────────────────────────


class FileHandler:
    def __init__(
        self,
        config: ShimConfig,
        audit: AuditWriter | None = None,
    ) -> None:
        self.config = config
        self.audit = audit

    # -- FILE_READ -----------------------------------------------------------

    async def handle_read(self, msg: FileReadMessage) -> FileContentsMessage | ErrorMessage:
        payload = msg.payload
        start = time.monotonic()
        try:
            path = await _jailed(payload.path, self.config.allowed_root)
        except PathJailError as exc:
            await _audit_jail_violation(self.audit, "FILE_READ", payload.path, exc)
            return _jail_error_to_message(msg.id, exc)

        details_base = {
            "path": str(path),
            "encoding": payload.encoding.value,
            "offset": payload.offset,
            "length": payload.length,
        }

        text_mode = payload.format != FileFormat.BYTES and payload.encoding != Encoding.BASE64
        response_limit = self.config.max_file_size_bytes
        if text_mode:
            response_limit = min(response_limit, MAX_WIRE_TEXT_CONTENT_BYTES)
        else:
            response_limit = min(response_limit, MAX_WIRE_BASE64_CONTENT_BYTES)

        def read_chunk() -> tuple[bytes | None, int]:
            with open_jail_file(path, self.config.allowed_root) as file:
                size = os.fstat(file.fileno()).st_size
                remaining = max(size - payload.offset, 0)
                requested_size = payload.length if payload.length is not None else remaining
                if requested_size > response_limit:
                    return None, size
                file.seek(payload.offset)
                return file.read(requested_size), size

        try:
            chunk, size = await _read_in_a_thread(read_chunk)
        except PathJailError as exc:
            await _audit_jail_violation(self.audit, "FILE_READ", payload.path, exc)
            return _jail_error_to_message(msg.id, exc)
        except OSError as exc:
            code = (
                ErrorCode.PATH_NOT_FOUND
                if isinstance(exc, FileNotFoundError)
                else ErrorCode.INTERNAL_ERROR
            )
            await self._emit(
                "FILE_READ",
                msg.id,
                {**details_base, "size_bytes_returned": 0},
                _err_result(_elapsed_ms(start), code.value),
            )
            return make_error(msg.id, code, f"Could not read file: {exc}")

        if chunk is None:
            requested_size = (
                payload.length if payload.length is not None else max(size - payload.offset, 0)
            )
            await self._emit(
                "FILE_READ",
                msg.id,
                {**details_base, "size_bytes_returned": 0},
                _err_result(_elapsed_ms(start), ErrorCode.FILE_TOO_LARGE.value),
            )
            return make_error(
                msg.id,
                ErrorCode.FILE_TOO_LARGE,
                f"Requested read of {requested_size} bytes exceeds limit {response_limit}",
            )

        truncated = payload.offset + len(chunk) < size

        # encoding ↔ format mapping per PROTOCOL.md FILE_READ table.
        if payload.format == FileFormat.BYTES or payload.encoding == Encoding.BASE64:
            content = base64.b64encode(chunk).decode("ascii")
            wire_encoding = Encoding.BASE64
        else:
            # text mode: decode as UTF-8, do NOT translate CRLF.
            content = chunk.decode("utf-8", errors="replace")
            wire_encoding = Encoding.UTF8

        await self._emit(
            "FILE_READ",
            msg.id,
            {**details_base, "size_bytes_returned": len(chunk)},
            _ok_result(_elapsed_ms(start)),
        )

        return FileContentsMessage(
            id=msg.id,
            ts=now_ts(),
            payload=FileContentsPayload(
                content=content,
                encoding=wire_encoding,
                size_bytes=len(chunk),
                truncated=truncated,
            ),
        )

    # -- FILE_WRITE ----------------------------------------------------------

    async def handle_write(self, msg: FileWriteMessage) -> AckMessage | ErrorMessage:
        payload = msg.payload
        start = time.monotonic()
        try:
            path = await _jailed(payload.path, self.config.allowed_root)
        except PathJailError as exc:
            await _audit_jail_violation(self.audit, "FILE_WRITE", payload.path, exc)
            return _jail_error_to_message(msg.id, exc)

        details_base = {
            "path": str(path),
            "encoding": payload.encoding.value,
            "create_parents": payload.create_parents,
        }

        if payload.encoding == Encoding.BASE64:
            try:
                raw = base64.b64decode(payload.content, validate=True)
            except (ValueError, binascii.Error) as exc:
                await self._emit(
                    "FILE_WRITE",
                    msg.id,
                    {**details_base, "size_bytes_written": 0},
                    _err_result(_elapsed_ms(start), ErrorCode.INTERNAL_ERROR.value),
                )
                return make_error(msg.id, ErrorCode.INTERNAL_ERROR, f"Bad base64: {exc}")
        else:
            raw = payload.content.encode("utf-8")

        if len(raw) > self.config.max_file_size_bytes:
            await self._emit(
                "FILE_WRITE",
                msg.id,
                {**details_base, "size_bytes_written": 0},
                _err_result(_elapsed_ms(start), ErrorCode.FILE_TOO_LARGE.value),
            )
            return make_error(
                msg.id,
                ErrorCode.FILE_TOO_LARGE,
                f"{len(raw)} bytes exceeds limit {self.config.max_file_size_bytes}",
            )

        def write_content() -> None:
            if payload.create_parents:
                path.parent.mkdir(parents=True, exist_ok=True)
            write_jail_file(path, self.config.allowed_root, raw)

        try:
            await _run_to_completion(write_content)
        except PathJailError as exc:
            await _audit_jail_violation(self.audit, "FILE_WRITE", payload.path, exc)
            return _jail_error_to_message(msg.id, exc)
        except OSError as exc:
            code = (
                ErrorCode.PATH_NOT_FOUND
                if isinstance(exc, FileNotFoundError)
                else ErrorCode.INTERNAL_ERROR
            )
            await self._emit(
                "FILE_WRITE",
                msg.id,
                {**details_base, "size_bytes_written": 0},
                _err_result(_elapsed_ms(start), code.value),
            )
            return make_error(msg.id, code, f"Could not write file: {exc}")
        await self._emit(
            "FILE_WRITE",
            msg.id,
            {**details_base, "size_bytes_written": len(raw)},
            _ok_result(_elapsed_ms(start)),
        )
        return make_ack(msg.id)

    # -- FILE_STAT -----------------------------------------------------------

    async def handle_stat(self, msg: FileStatMessage) -> FileStatResponseMessage | ErrorMessage:
        payload = msg.payload
        start = time.monotonic()

        def jailed() -> Path:
            path = assert_inside_jail(payload.path, self.config.allowed_root)
            # Without following, describe the entry the path names, a symlink
            # included. The root has no entry inside itself, so it is
            # described as resolved.
            if not payload.follow_symlinks and path != _resolved_root(self.config):
                return assert_entry_inside_jail(payload.path, self.config.allowed_root)
            return path

        try:
            path = await _read_in_a_thread(jailed)
        except PathJailError as exc:
            await _audit_jail_violation(self.audit, "FILE_STAT", payload.path, exc)
            return _jail_error_to_message(msg.id, exc)

        details = {"path": str(path), "follow_symlinks": payload.follow_symlinks}

        try:
            if payload.follow_symlinks:
                st = await _read_in_a_thread(os.stat, str(path))
            else:
                st = await _read_in_a_thread(os.lstat, str(path))
        except FileNotFoundError:
            await self._emit("FILE_STAT", msg.id, details, _ok_result(_elapsed_ms(start)))
            return FileStatResponseMessage(
                id=msg.id,
                ts=now_ts(),
                payload=FileStatResponsePayload(exists=False),
            )
        except OSError as exc:
            await self._emit(
                "FILE_STAT",
                msg.id,
                details,
                _err_result(_elapsed_ms(start), ErrorCode.INTERNAL_ERROR.value),
            )
            return make_error(msg.id, ErrorCode.INTERNAL_ERROR, str(exc))

        is_file = stat.S_ISREG(st.st_mode)
        is_dir = stat.S_ISDIR(st.st_mode)
        is_symlink = False
        if not payload.follow_symlinks:
            is_symlink = stat.S_ISLNK(st.st_mode)
        else:
            # Even with follow=True we can report whether the *original*
            # path was a symlink, via a separate lstat of the path the jail
            # checked (a relative one is the root's, not the working
            # directory's).
            try:
                lst = await _read_in_a_thread(
                    os.lstat, from_root(payload.path, self.config.allowed_root)
                )
                is_symlink = stat.S_ISLNK(lst.st_mode)
            except OSError:
                is_symlink = False

        is_windows = sys.platform == "win32"
        mode_str: str
        if is_windows:
            # Derive a POSIX-flavored mode from FILE_ATTRIBUTE_READONLY.
            ro = bool(st.st_mode & stat.S_IWRITE) is False
            if is_dir:
                mode_str = "0555" if ro else "0755"
            else:
                mode_str = "0444" if ro else "0644"
        else:
            mode_str = f"{stat.S_IMODE(st.st_mode):04o}"

        mime, _ = mimetypes.guess_type(str(path))

        await self._emit("FILE_STAT", msg.id, details, _ok_result(_elapsed_ms(start)))
        return FileStatResponseMessage(
            id=msg.id,
            ts=now_ts(),
            payload=FileStatResponsePayload(
                exists=True,
                is_file=is_file,
                is_dir=is_dir,
                is_symlink=is_symlink,
                size_bytes=st.st_size,
                mtime=st.st_mtime,
                ctime=st.st_ctime,
                mode=mode_str,
                owner_uid=None if is_windows else st.st_uid,
                owner_gid=None if is_windows else st.st_gid,
                mime_type=mime,
                path=str(path),
            ),
        )

    # -- FILE_LIST -----------------------------------------------------------

    async def handle_list(self, msg: FileListMessage) -> FileListResponseMessage | ErrorMessage:
        payload = msg.payload
        start = time.monotonic()
        try:
            base = await _jailed(payload.path, self.config.allowed_root)
        except PathJailError as exc:
            await _audit_jail_violation(self.audit, "FILE_LIST", payload.path, exc)
            return _jail_error_to_message(msg.id, exc)

        details_base = {
            "path": str(base),
            "glob": payload.glob,
            "recursive": payload.recursive,
            "include_hidden": payload.include_hidden,
            "max_entries": payload.max_entries,
        }

        if not await _read_in_a_thread(base.exists):
            await self._emit(
                "FILE_LIST",
                msg.id,
                {**details_base, "entries_returned": 0},
                _err_result(_elapsed_ms(start), ErrorCode.PATH_NOT_FOUND.value),
            )
            return make_error(msg.id, ErrorCode.PATH_NOT_FOUND, f"{base} does not exist")
        if not await _read_in_a_thread(base.is_dir):
            await self._emit(
                "FILE_LIST",
                msg.id,
                {**details_base, "entries_returned": 0},
                _err_result(_elapsed_ms(start), ErrorCode.INTERNAL_ERROR.value),
            )
            return make_error(msg.id, ErrorCode.INTERNAL_ERROR, f"{base} is not a directory")

        # max_entries has no bound on the wire; cap it so a huge value can't
        # walk a whole tree only to have the reply refused as too large. Zero
        # or less asks for no entries.
        limit = max(0, min(payload.max_entries, MAX_FILE_LIST_ENTRIES))

        def _walk() -> tuple[list[FileEntry], bool]:
            entries: list[FileEntry] = []
            truncated = False
            # A subfolder that couldn't be read leaves its entries out.
            unread = False
            case_fold = platform_info.is_case_insensitive_fs(str(base))
            glob = payload.glob

            def _matches(name: str) -> bool:
                if glob is None:
                    return True
                if case_fold:
                    from fnmatch import fnmatchcase

                    return fnmatchcase(name.lower(), glob.lower())
                from fnmatch import fnmatchcase

                return fnmatchcase(name, glob)

            def _candidates() -> Iterator[Path]:
                # os.scandir yields entries as the directory is read, so the
                # scan budget stops a huge directory part way; os.walk and
                # iterdir read each directory whole before yielding any of it.
                # Depth first, a directory's entries before its subfolders',
                # and like os.walk it never follows a symlink into a folder,
                # nor a Windows junction, which is_dir() takes for one.
                nonlocal unread
                pending = [base]
                while pending:
                    folder = pending.pop()
                    subfolders: list[Path] = []
                    try:
                        with os.scandir(folder) as entries:
                            for entry in entries:
                                path = Path(entry.path)
                                yield path
                                if not payload.recursive:
                                    continue
                                # Hidden folders are pruned, never entered.
                                if not payload.include_hidden and entry.name.startswith("."):
                                    continue
                                try:
                                    folder_here = entry.is_dir(follow_symlinks=False)
                                    if folder_here and not _is_reparse_point(path):
                                        subfolders.append(path)
                                except OSError:
                                    pass
                    except OSError:
                        # The folder asked for is an error (handle_list
                        # answers it); one under it leaves the listing short.
                        if folder == base:
                            raise
                        unread = True
                        continue
                    pending.extend(reversed(subfolders))

            # `truncated` is set only once something is known to be left out:
            # an entry not looked at, or a match that didn't fit.
            scanned = 0
            reply_bytes = 0
            for p in _candidates():
                scanned += 1
                if scanned > MAX_FILE_LIST_SCANNED:
                    truncated = True
                    break
                # A recursive walk reaches the contents of hidden folders too.
                hidden = any(part.startswith(".") for part in p.relative_to(base).parts)
                if hidden and not payload.include_hidden:
                    continue
                if not _matches(p.name):
                    continue
                if not _sendable(str(p)):
                    # A POSIX name whose bytes aren't UTF-8 decodes to lone
                    # surrogates, which JSON can't carry; no request could
                    # name the file either.
                    continue
                if len(entries) >= limit:
                    truncated = True
                    break
                try:
                    st = p.lstat()
                except OSError:
                    continue
                link = stat.S_ISLNK(st.st_mode)
                # The entry's own type, from its lstat. A link is followed only
                # to a target inside the jail: whether one outside exists, and
                # what it is, isn't the platform's to learn.
                mode = _jailed_link_target_mode(p, self.config.allowed_root) if link else st.st_mode
                entry = FileEntry(
                    name=p.name,
                    path=str(p),
                    is_file=mode is not None and stat.S_ISREG(mode),
                    is_dir=mode is not None and stat.S_ISDIR(mode),
                    is_symlink=link,
                    size_bytes=st.st_size,
                    mtime=st.st_mtime,
                )
                # Measured as it goes on the wire, with the comma after it.
                size = len(entry.model_dump_json().encode("utf-8")) + 1
                if reply_bytes + size > FILE_LIST_REPLY_BUDGET_BYTES:
                    truncated = True
                    break
                reply_bytes += size
                entries.append(entry)
            return entries, truncated or unread

        try:
            entries, truncated = await _read_in_a_thread(_walk)
        except OSError as exc:
            # The folder can be entered but not read (no read permission, a
            # macOS privacy prompt nobody answered), or went away: listing it
            # as empty would say it is.
            code = (
                ErrorCode.PATH_NOT_FOUND
                if isinstance(exc, FileNotFoundError)
                else ErrorCode.INTERNAL_ERROR
            )
            await self._emit(
                "FILE_LIST",
                msg.id,
                {**details_base, "entries_returned": 0},
                _err_result(_elapsed_ms(start), code.value),
            )
            return make_error(msg.id, code, f"Could not list {base}: {exc}")
        await self._emit(
            "FILE_LIST",
            msg.id,
            {**details_base, "entries_returned": len(entries)},
            _ok_result(_elapsed_ms(start)),
        )
        return FileListResponseMessage(
            id=msg.id,
            ts=now_ts(),
            payload=FileListResponsePayload(entries=entries, truncated=truncated),
        )

    # -- FILE_DELETE ---------------------------------------------------------

    async def handle_delete(self, msg: FileDeleteMessage) -> AckMessage | ErrorMessage:
        payload = msg.payload
        start = time.monotonic()

        def jailed() -> Path:
            assert_inside_jail(payload.path, self.config.allowed_root)
            # A symlink is removed itself; its target is left alone.
            return assert_entry_inside_jail(payload.path, self.config.allowed_root)

        try:
            path = await _read_in_a_thread(jailed)
        except PathJailError as exc:
            await _audit_jail_violation(self.audit, "FILE_DELETE", payload.path, exc)
            return _jail_error_to_message(msg.id, exc)

        details = {
            "path": str(path),
            "recursive": payload.recursive,
            "missing_ok": payload.missing_ok,
        }

        if not await _read_in_a_thread(os.path.lexists, path):
            if payload.missing_ok:
                await self._emit("FILE_DELETE", msg.id, details, _ok_result(_elapsed_ms(start)))
                return make_ack(msg.id)
            await self._emit(
                "FILE_DELETE",
                msg.id,
                details,
                _err_result(_elapsed_ms(start), ErrorCode.PATH_NOT_FOUND.value),
            )
            return make_error(msg.id, ErrorCode.PATH_NOT_FOUND, f"{path} does not exist")

        try:
            if await _read_in_a_thread(_is_own_directory, path):
                if payload.recursive:
                    await _run_to_completion(shutil.rmtree, str(path))
                else:
                    try:
                        await _run_to_completion(os.rmdir, str(path))
                    except OSError as exc:
                        # Dir not empty.
                        await self._emit(
                            "FILE_DELETE",
                            msg.id,
                            details,
                            _err_result(_elapsed_ms(start), ErrorCode.PATH_NOT_EMPTY.value),
                        )
                        return make_error(msg.id, ErrorCode.PATH_NOT_EMPTY, str(exc))
            else:
                await _run_to_completion(os.unlink, str(path))
        except OSError as exc:
            await self._emit(
                "FILE_DELETE",
                msg.id,
                details,
                _err_result(_elapsed_ms(start), ErrorCode.INTERNAL_ERROR.value),
            )
            return make_error(msg.id, ErrorCode.INTERNAL_ERROR, str(exc))

        await self._emit("FILE_DELETE", msg.id, details, _ok_result(_elapsed_ms(start)))
        return make_ack(msg.id)

    # -- FILE_MOVE -----------------------------------------------------------

    async def handle_move(self, msg: FileMoveMessage) -> AckMessage | ErrorMessage:
        payload = msg.payload
        start = time.monotonic()

        def jailed() -> tuple[Path, Path]:
            assert_inside_jail(payload.src, self.config.allowed_root)
            assert_inside_jail(payload.dst, self.config.allowed_root)
            # Renaming moves a symlink itself, and overwriting replaces one,
            # never its target.
            return (
                assert_entry_inside_jail(payload.src, self.config.allowed_root),
                assert_entry_inside_jail(payload.dst, self.config.allowed_root),
            )

        try:
            src, dst = await _read_in_a_thread(jailed)
        except PathJailError as exc:
            await _audit_jail_violation(
                self.audit, "FILE_MOVE", f"{payload.src} -> {payload.dst}", exc
            )
            return _jail_error_to_message(msg.id, exc)

        details = {"src": str(src), "dst": str(dst), "overwrite": payload.overwrite}

        if not await _read_in_a_thread(os.path.lexists, src):
            await self._emit(
                "FILE_MOVE",
                msg.id,
                details,
                _err_result(_elapsed_ms(start), ErrorCode.PATH_NOT_FOUND.value),
            )
            return make_error(msg.id, ErrorCode.PATH_NOT_FOUND, f"{src} does not exist")
        if not payload.overwrite and await _read_in_a_thread(os.path.lexists, dst):
            await self._emit(
                "FILE_MOVE",
                msg.id,
                details,
                _err_result(_elapsed_ms(start), ErrorCode.PATH_EXISTS.value),
            )
            return make_error(msg.id, ErrorCode.PATH_EXISTS, f"{dst} already exists")

        try:
            # Without overwrite, a destination created after the check above
            # must not be replaced, so the rename itself refuses one.
            move = os.replace if payload.overwrite else rename_no_replace
            await _run_to_completion(move, str(src), str(dst))
        except FileExistsError:
            await self._emit(
                "FILE_MOVE",
                msg.id,
                details,
                _err_result(_elapsed_ms(start), ErrorCode.PATH_EXISTS.value),
            )
            return make_error(msg.id, ErrorCode.PATH_EXISTS, f"{dst} already exists")
        except OSError as exc:
            await self._emit(
                "FILE_MOVE",
                msg.id,
                details,
                _err_result(_elapsed_ms(start), ErrorCode.INTERNAL_ERROR.value),
            )
            return make_error(msg.id, ErrorCode.INTERNAL_ERROR, str(exc))

        await self._emit("FILE_MOVE", msg.id, details, _ok_result(_elapsed_ms(start)))
        return make_ack(msg.id)

    # -- emit helper ---------------------------------------------------------

    async def _emit(
        self,
        op: str,
        request_id: str,
        details: dict,
        result: dict,
    ) -> None:
        if self.audit is None:
            return
        try:
            await _finish_then_cancel(
                self.audit.write(op, request_id=request_id, details=details, result=result)
            )
        except Exception:
            logger.debug("Failed to write %s audit record", op, exc_info=True)


# ── Computer use ─────────────────────────────────────────────────────────────


def _session_ended(raw: dict[str, Any]) -> bool:
    """Whether cua-driver refused a call because its session had ended
    (cua-driver 0.38: `refusal.code` "session_ended")."""
    structured = raw.get("structuredContent")
    refusal = structured.get("refusal") if isinstance(structured, dict) else None
    return (
        bool(raw.get("isError"))
        and isinstance(refusal, dict)
        and refusal.get("code") == "session_ended"
    )


# How many chats' driver label renewals a machine keeps (session_label): a
# chat renewed less recently than that many others starts from its first.
MAX_RENEWED_LABELS = 1024


def _program_on_path(name: str) -> str | None:
    """The program `name` runs from PATH (shutil.which): what a Linux
    launch's name runs when its first word is a program's (check_arguments)."""
    return shutil.which(name)


def _seen(future: asyncio.Future[Any]) -> None:
    """Mark a lookup's failure seen: the callers that waited for it may all
    have given up."""
    if not future.cancelled():
        future.exception()


class _ProgramLookups:
    """A machine's lookups of a Linux launch's name on PATH
    (_program_on_path), made in a worker thread one at a time.

    shutil.which stats a file in each PATH folder, and a stalled mount there
    (NFS, FUSE) holds it for as long as it stalls. A caller waits for a
    lookup no later than its own deadline, and leaves it to finish in its
    thread. The next caller waits for that lookup too, no later than its
    deadline, rather than start one that would stall as well: one looking
    the same name up takes its answer, and one looking another up starts
    its own once it has ended. So a stall holds one worker thread, however
    many launches come in."""

    def __init__(self) -> None:
        self._name: str | None = None
        self._lookup: asyncio.Future[str | None] | None = None

    async def program_by(self, name: str, deadline: float) -> str | None:
        """The program `name` runs, by `deadline` (loop time), else
        TimeoutError."""
        loop = asyncio.get_running_loop()
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            lookup = self._lookup
            if lookup is None or lookup.done():
                lookup = self._lookup = asyncio.ensure_future(
                    asyncio.to_thread(_program_on_path, name)
                )
                lookup.add_done_callback(_seen)
                self._name = name
            ours = self._name == name
            done, _ = await asyncio.wait({lookup}, timeout=remaining)
            if not done:
                raise TimeoutError
            if ours:
                return lookup.result()


class _ClipboardChecks:
    """A machine's checks of the clipboard for a password manager's mark
    (clipboard_state), made in a worker thread one at a time.

    On Linux the check looks for wl-paste or xclip on PATH before its own
    timeout applies, and a stalled mount there (NFS, FUSE) holds that look
    for as long as it stalls. A foreign clipboard read waits for its check
    no later than its deadline and leaves it to finish in its thread. The
    next check waits for that one to end, no later than its own deadline,
    then looks for itself: what the clipboard held then says nothing of what
    it holds now, and a read is checked before and after with looks of its
    own. So a stall holds one worker thread, however many reads come in."""

    def __init__(self) -> None:
        self._check: asyncio.Future[ClipboardState] | None = None

    async def state_by(self, deadline: float) -> ClipboardState:
        """The clipboard's marker, by `deadline` (loop time), else
        TimeoutError."""
        loop = asyncio.get_running_loop()
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            earlier = self._check
            check: asyncio.Future[ClipboardState]
            if earlier is None or earlier.done():
                check = self._check = asyncio.ensure_future(
                    asyncio.to_thread(clipboard_state, remaining)
                )
                check.add_done_callback(_seen)
            else:
                check = earlier
            done, _ = await asyncio.wait({check}, timeout=remaining)
            if not done:
                raise TimeoutError
            if check is not earlier:
                return check.result()


class ComputerUseHandler:
    """Relays computer use to cua-driver (see docs/CUA_DRIVER.md).

    COMPUTER_TOOLS_REQUEST answers with the driver's guidance and the tools
    this machine offers. COMPUTER_TOOL_CALL checks the call against the
    catalog, sets the session label, applies the clipboard rules, relays it,
    and returns what the driver said: its own errors as a result the model
    reads, the shim's refusals and failures as ERROR.
    """

    def __init__(
        self,
        config: ShimConfig,
        audit: AuditWriter | None = None,
        driver: CuaDriverBridge | None = None,
        machine_turn: asyncio.Lock | None = None,
    ) -> None:
        self.config = config
        self.audit = audit
        self._driver = driver
        # One GUI: tool calls from every session on the machine take turns.
        self._machine_turn = machine_turn or asyncio.Lock()
        # The machine's lookups of launch names on PATH, made one at a time.
        self._program_lookups = _ProgramLookups()
        # The machine's checks of the clipboard for a password manager's
        # mark, made one at a time.
        self._clipboard_checks = _ClipboardChecks()
        # A child session's handler uses its parent's driver and never stops it.
        self._owns_driver = True
        self._screenshot_rate = _SlidingWindowRateLimiter(config.max_screenshots_per_minute)
        self.clipboard_writeback = ClipboardWritebackRegistry()
        # How many times cua-driver ended each chat's label, by the label the
        # chat started from (session_label). The machine's, shared like the
        # driver (share_with): a chat's later sessions use the same driver,
        # which refuses the labels it ended until it restarts.
        self._label_renewals: dict[str, int] = {}

    @property
    def shares_a_driver(self) -> bool:
        """Whether this is a child session's handler, using its parent's
        driver (share_with)."""
        return not self._owns_driver

    async def hello_snapshot(self) -> DriverSnapshot:
        """What this session's HELLO advertises. HELLO waits for the driver at
        most _hello_wait: after a chat's call overran, the driver can take up
        to a minute to start again, longer than the platform waits for HELLO.
        Past that it goes on with what the driver offered when it last ran,
        which its next start lists again."""
        wait = self._hello_wait
        if wait is None:
            return await self.driver.snapshot()
        loop = asyncio.get_running_loop()
        try:
            return await self.driver.snapshot(deadline=loop.time() + wait)
        except CallNotStarted:
            last = self.driver.last_snapshot
            if last is None:
                raise DriverUnavailable("cua-driver is still starting") from None
            return last

    async def driver_found_for_hello(self) -> bool:
        """Whether the driver is installed, for HELLO, from the bridge's
        look (CuaDriverBridge.find_binary, one look at a time), within
        HELLO's wait and DRIVER_LOOKUP_SECONDS at most.

        A stalled mount on sys.path or PATH holds the look for as long as it
        stalls, and HELLO, which looked in a worker thread of its own and
        waited for it to the end, never went out: the machine couldn't
        connect, even for file work. Past the wait HELLO goes without
        computer use, as when the driver isn't installed, and the look goes
        on: the next HELLO waits for that same look."""
        wait = self._hello_wait
        timeout = DRIVER_LOOKUP_SECONDS if wait is None else min(DRIVER_LOOKUP_SECONDS, wait)
        try:
            async with asyncio.timeout(timeout):
                return await self.driver.find_binary() is not None
        except TimeoutError:
            logger.warning(
                "Not advertising computer_use: looking for cua-driver took longer than %g s",
                timeout,
            )
            return False

    @property
    def _hello_wait(self) -> float | None:
        """How long HELLO waits for the driver: CHILD_DRIVER_WAIT_SECONDS for
        a chat's session, which the platform undoes in about eight seconds,
        and for the owner's once the driver has run, since the platform
        closes a connection whose HELLO hasn't come in ten. The owner's first
        start is waited for in full: nothing earlier can stand in for it."""
        if self.shares_a_driver or self.driver.last_snapshot is not None:
            return CHILD_DRIVER_WAIT_SECONDS
        return None

    @property
    def driver(self) -> CuaDriverBridge:
        if self._driver is None:
            self._driver = CuaDriverBridge(
                tools=offered_tools(enable_clipboard=self.config.enable_clipboard),
                binary=self.config.cua_driver_path,
                client_version=_shim_version(),
            )
        return self._driver

    def share_with(self, other: ComputerUseHandler) -> None:
        """Make `other`, a child session's handler, use this one's driver
        process, turn, lookups on PATH, clipboard checks, screenshot budget,
        clipboard record and label renewals: they are the machine's, not the
        session's."""
        other._driver = self.driver
        other._machine_turn = self._machine_turn
        other._program_lookups = self._program_lookups
        other._clipboard_checks = self._clipboard_checks
        other._owns_driver = False
        other._screenshot_rate = self._screenshot_rate
        other.clipboard_writeback = self.clipboard_writeback
        other._label_renewals = self._label_renewals

    async def screen_size(self) -> tuple[int, int] | None:
        """The main display's logical size, from the driver, for HELLO. The
        whole call fits in HELLO's wait, its run as well as its start, and
        the stop of a driver that overran it isn't waited for: a driver that
        takes the call and stalls, or is slow to go, would otherwise hold
        HELLO past the point the platform gives up on it, for a size HELLO
        can leave out."""
        wait = self._hello_wait
        try:
            raw = await self.call_driver(
                "get_screen_size",
                {"session": self.session_label},
                timeout=SCREEN_SIZE_SECONDS if wait is None else min(SCREEN_SIZE_SECONDS, wait),
                start_within=wait,
                wait_for_stop=False,
            )
        except (DriverUnavailable, McpError, TimeoutError):
            # HELLO goes out without the size rather than not at all.
            return None
        structured = raw.get("structuredContent")
        if raw.get("isError") or not isinstance(structured, dict):
            return None
        width, height = structured.get("width"), structured.get("height")
        # Python's json reads NaN and Infinity, which int() refuses.
        if (
            isinstance(width, int | float)
            and isinstance(height, int | float)
            and math.isfinite(width)
            and math.isfinite(height)
        ):
            return int(width), int(height)
        return None

    async def close(self) -> None:
        """Stop the driver process, if this handler started one."""
        if self._owns_driver and self._driver is not None:
            await self._driver.close()

    @property
    def session_label(self) -> str:
        """The cua-driver session every call from this shim session carries,
        so each gets its own agent cursor and cleanup. A new one each time
        cua-driver ends the last (call_driver), which the chat's later
        sessions on this machine go on from: the driver they share refuses
        an ended label until it restarts."""
        label = self._first_label
        renewals = self._label_renewals.get(label, 0)
        return f"{label}.{renewals}" if renewals else label

    @property
    def _first_label(self) -> str:
        session_id = self.config.session_id or "default"
        return "autogpt-" + hashlib.sha256(session_id.encode()).hexdigest()[:12]

    def _renew_label(self) -> None:
        """Move this chat to its next label, as the machine's most recently
        renewed (MAX_RENEWED_LABELS)."""
        label = self._first_label
        self._label_renewals[label] = self._label_renewals.pop(label, 0) + 1
        while len(self._label_renewals) > MAX_RENEWED_LABELS:
            del self._label_renewals[next(iter(self._label_renewals))]

    async def call_driver(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        timeout: float,
        start_within: float | None = None,
        wait_for_stop: bool = True,
    ) -> dict[str, Any]:
        """One driver call, as `CuaDriverBridge.call_tool`. cua-driver ends a
        named session left idle for five minutes and refuses every later call
        that names it until the driver restarts, so a call refused for that
        moves this session to a new label and goes again, once: the refused
        call never ran. It goes again within what is left of the timeout and
        of `start_within`, so it waits no longer for the driver than the
        first try would have."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        start_by = None if start_within is None else loop.time() + start_within
        raw = await self.driver.call_tool(
            tool,
            arguments,
            timeout=timeout,
            start_within=start_within,
            wait_for_stop=wait_for_stop,
        )
        if arguments.get("session") != self.session_label or not _session_ended(raw):
            return raw
        self._renew_label()
        arguments["session"] = self.session_label
        logger.info(
            f"cua-driver ended this session's driver session; going on as {arguments['session']}"
        )
        now = loop.time()
        remaining = deadline - now
        if remaining <= 0:
            raise CallNotStarted(f"{tool}'s deadline passed while its driver session was renewed")
        return await self.driver.call_tool(
            tool,
            arguments,
            timeout=remaining,
            start_within=None if start_by is None else max(start_by - now, 0.0),
            wait_for_stop=wait_for_stop,
        )

    async def handle(self, msg: Any, *, received_at: float | None = None) -> Any:
        """`received_at` (loop time) is when the shim received the message;
        a tool call's deadline counts from it."""
        if not self.config.enable_computer_use:
            return make_error(
                msg.id,
                ErrorCode.CAPABILITY_NOT_GRANTED,
                "Computer use is not enabled on this shim.",
            )
        if isinstance(msg, ComputerToolsRequestMessage):
            return await self._tools(msg)
        if isinstance(msg, ComputerToolCallMessage):
            return await self._call(msg, received_at)
        return make_error(
            msg.id,
            ErrorCode.INTERNAL_ERROR,
            f"Unknown computer-use message: {type(msg).__name__}",
        )

    # ── COMPUTER_TOOLS_REQUEST ───────────────────────────────────────

    async def _tools(self, msg: ComputerToolsRequestMessage) -> Any:
        try:
            snapshot = await self.driver.snapshot()
        except DriverUnavailable as exc:
            return make_error(msg.id, exc.code, exc.message, details=exc.details or None)
        tools = [
            ComputerToolSpec(
                name=tool["name"],
                description=str(tool.get("description") or ""),
                input_schema=without_session(tool.get("inputSchema") or {"type": "object"}),
                annotations=tool.get("annotations") or None,
                approval_arguments=APPROVAL_ARGUMENTS.get(tool["name"], {}),
            )
            for tool in snapshot.tools
        ]
        return ComputerToolsResponseMessage(
            id=msg.id,
            ts=now_ts(),
            payload=ComputerToolsResponsePayload(
                driver=snapshot.info,
                instructions=_instructions(snapshot, self.config),
                tools=tools,
            ),
        )

    # ── COMPUTER_TOOL_CALL ───────────────────────────────────────────

    async def _call(self, msg: ComputerToolCallMessage, received_at: float | None = None) -> Any:
        """One call, within one deadline: `timeout_seconds` from when the
        shim received it (`received_at`, loop time), covering its wait for
        the computer, which every session's calls take turns at, and the
        driver's work. A call whose deadline passes while it waits never
        reaches the driver. The computer is let go once the driver is done,
        before the call's audit record is written: a record held up (the
        log locked by `autogpt-shim audit verify`, a slow disk) holds up
        this call's answer, which goes out after it, not other chats' calls.
        What the call's check looks up on disk is looked up before it waits
        for the computer, within the same deadline (_programs_named)."""
        deadline = self.call_deadline(msg, received_at)
        start = time.monotonic()
        details = self._call_details(msg)
        try:
            programs = await self._programs_named(msg, deadline)
            async with asyncio.timeout_at(deadline):
                await self._machine_turn.acquire()
        except TimeoutError:
            return await self._not_started(msg, details, start)
        try:
            try:
                response, record, result = await self._run_call(
                    msg, details, start, deadline, programs
                )
            finally:
                self._machine_turn.release()
        except asyncio.CancelledError:
            # The action may have happened: the bridge stopped the driver to
            # end it, but nothing takes back a click it already made.
            await self._emit(
                msg.id, details, _err_result(_elapsed_ms(start), _CANCELLED_ERROR_CODE)
            )
            raise
        await self._emit(msg.id, record, result)
        return response

    @staticmethod
    def call_deadline(msg: ComputerToolCallMessage, received_at: float | None = None) -> float:
        """When a call's `timeout_seconds` runs out (loop time), counted from
        when the shim received it."""
        seconds = float(msg.payload.timeout_seconds or DEFAULT_COMPUTER_TOOL_TIMEOUT_SECONDS)
        return (asyncio.get_running_loop().time() if received_at is None else received_at) + seconds

    async def _programs_named(
        self, msg: ComputerToolCallMessage, deadline: float
    ) -> dict[str, str | None]:
        """What the call's check (check_arguments) looks up on disk, by the
        call's deadline, else TimeoutError: the program a Linux launch's
        name runs when its first word is a program's, from PATH.

        It is looked up before the call waits for the computer, through the
        machine's lookups (_ProgramLookups). Looked up while the call held
        the computer, with its wait for the worker thread held to the end, a
        stalled mount on PATH held every chat's calls behind it for as long
        as it stalled. A call the check refuses looks nothing up, and _relay
        refuses it."""
        names: list[str] = []

        def noted(name: str) -> str | None:
            names.append(name)
            return None

        try:
            check_arguments(
                msg.payload.tool,
                msg.payload.arguments,
                msg.payload.approved_arguments,
                on_path=noted,
            )
        except (ApprovalRequired, ArgumentRefused):
            return {}
        programs: dict[str, str | None] = {}
        for name in names:
            programs[name] = await self._program_lookups.program_by(name, deadline)
        return programs

    @staticmethod
    def _call_details(msg: ComputerToolCallMessage) -> dict[str, Any]:
        tool = msg.payload.tool
        approved = msg.payload.approved_arguments
        # Only a name the tool can have approved counts: a call naming
        # another is refused, and its record keeps the usual cut.
        approves = set(approved) & set(APPROVAL_ARGUMENTS.get(tool, {}))
        details: dict[str, Any] = {
            "tool": tool,
            # What launches, and the paths the person approved, are kept
            # whole, as EXECUTE_COMMAND keeps argv.
            "arguments": _audit_arguments(
                dict(msg.payload.arguments), truncate=tool != "launch_app" and not approves
            ),
        }
        if approved:
            details["approved_arguments"] = sorted(approved)
        return details

    async def _run_call(
        self,
        msg: ComputerToolCallMessage,
        details: dict[str, Any],
        start: float,
        deadline: float,
        programs: dict[str, str | None],
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        """Relay the call, and return its answer with its audit record's
        details and result, which _call writes once the computer is free.
        `programs` is what its check looks up on disk (_programs_named)."""
        tool = msg.payload.tool
        arguments = dict(msg.payload.arguments)
        approved = msg.payload.approved_arguments
        try:
            result = await self._relay(tool, arguments, deadline, approved, programs)
        except ComputerUseError as exc:
            return (
                make_error(msg.id, exc.code, exc.message, details=exc.details or None),
                details,
                _err_result(_elapsed_ms(start), exc.code.value),
            )
        except DriverUnavailable as exc:
            return (
                make_error(msg.id, exc.code, exc.message, details=exc.details or None),
                details,
                _err_result(_elapsed_ms(start), exc.code.value),
            )
        except McpError as exc:
            if exc.code != INVALID_PARAMS:
                return (
                    make_error(msg.id, ErrorCode.INTERNAL_ERROR, f"cua-driver: {exc.message}"),
                    details,
                    _err_result(_elapsed_ms(start), ErrorCode.INTERNAL_ERROR.value),
                )
            # Arguments the driver rejected are the model's to fix.
            result = self._result_payload(
                tool, {"isError": True, "content": [{"type": "text", "text": exc.message}]}
            )
        except CallNotStarted:
            return (
                self._not_started_response(msg),
                details,
                _err_result(_elapsed_ms(start), ErrorCode.COMMAND_TIMEOUT.value),
            )
        except TimeoutError as exc:
            timed_out = make_error(
                msg.id,
                ErrorCode.COMMAND_TIMEOUT,
                (
                    f"{tool} ran out of time before the clipboard could be checked for a "
                    "password manager's mark, so nothing it read is returned. Read it again."
                )
                if isinstance(exc, _ClipboardCheckOutOfTime)
                else f"{tool} did not finish in time, so the local executor stopped cua-driver "
                "to end it. What it already did stays done, and the next call starts the "
                "driver afresh: look at the window again before you act on it.",
                details={
                    "tool": tool,
                    "timeout_seconds": msg.payload.timeout_seconds
                    or DEFAULT_COMPUTER_TOOL_TIMEOUT_SECONDS,
                    "started": True,
                },
            )
            return (
                timed_out,
                details,
                _err_result(_elapsed_ms(start), ErrorCode.COMMAND_TIMEOUT.value),
            )
        payload, summary = result
        return (
            ComputerToolResultMessage(id=msg.id, ts=now_ts(), payload=payload),
            {**details, **summary},
            _ok_result(_elapsed_ms(start)),
        )

    async def _not_started(
        self, msg: ComputerToolCallMessage, details: dict[str, Any], start: float
    ) -> Any:
        """Audit, and answer, a call whose deadline passed before it had the
        computer."""
        await self._emit(
            msg.id, details, _err_result(_elapsed_ms(start), ErrorCode.COMMAND_TIMEOUT.value)
        )
        return self._not_started_response(msg)

    @staticmethod
    def _not_started_response(msg: ComputerToolCallMessage) -> Any:
        """The deadline passed before the driver took the call: it waited
        behind other calls on this computer, or for the driver to start."""
        return make_error(
            msg.id,
            ErrorCode.COMMAND_TIMEOUT,
            f"{msg.payload.tool} didn't start: this computer was busy with another call, "
            "or cua-driver was still starting, until its deadline passed. Nothing ran; "
            "call it again.",
            details={
                "tool": msg.payload.tool,
                "timeout_seconds": msg.payload.timeout_seconds
                or DEFAULT_COMPUTER_TOOL_TIMEOUT_SECONDS,
                "started": False,
            },
        )

    async def _relay(
        self,
        tool: str,
        arguments: dict[str, Any],
        deadline: float,
        approved: list[str],
        programs: dict[str, str | None],
    ) -> tuple[ComputerToolResultPayload, dict[str, Any]]:
        snapshot = await self.driver.snapshot(deadline=deadline)
        spec = snapshot.tool(tool)
        if spec is None:
            raise ComputerUseError(
                ErrorCode.FEATURE_NOT_SUPPORTED,
                f"{tool} is not available on this machine.",
                details={"tool": tool},
            )
        try:
            # On the event loop: the one thing the check looks at on disk,
            # the program a Linux launch's name runs, was looked up on PATH
            # before the call took the computer (_programs_named). A name
            # not looked up then can't come up here: the check is the same.
            check_arguments(tool, arguments, approved, on_path=programs.__getitem__)
        except ApprovalRequired as exc:
            raise ComputerUseError(
                ErrorCode.APPROVAL_REQUIRED,
                str(exc),
                details={"tool": tool, "argument": exc.argument},
            ) from exc
        except ArgumentRefused as exc:
            raise ComputerUseError(
                ErrorCode.FEATURE_NOT_SUPPORTED,
                str(exc),
                details={"tool": tool, "argument": exc.argument},
            ) from exc
        if tool == "launch_app":
            await self._check_launch_name(arguments, deadline)
        if takes_session(spec.get("inputSchema") or {}):
            arguments["session"] = self.session_label
        timeout = deadline - asyncio.get_running_loop().time()
        if timeout <= 0:
            raise CallNotStarted(f"{tool}'s deadline passed before the driver took it")

        if tool == "clipboard_read":
            raw = await self._clipboard_read(arguments, timeout)
        else:
            raw = await self.call_driver(tool, arguments, timeout=timeout)
            if tool == "clipboard_write" and not raw.get("isError"):
                if isinstance(arguments.get("text"), str):
                    self.clipboard_writeback.record_write(arguments["text"])
                else:
                    # A file or image replaced the text this shim wrote.
                    self.clipboard_writeback.clear()
        return self._result_payload(tool, raw)

    async def _check_launch_name(self, arguments: dict[str, Any], deadline: float) -> None:
        """A launch_app `name` goes ahead only when an installed app, as
        list_apps lists it, would match it, on the systems where cua-driver
        acts on a name nothing matches.

        On Windows the driver looks the name up among the installed apps
        and, when none matches, hands it to ShellExecuteEx, which runs a
        program of that name from PATH, out of sight: `logoff` signs the
        person out. A launch_path, path or AUMID takes precedence over the
        name in the driver's routing, so those calls aren't checked.

        On Linux, once no program has the name's first word
        (check_arguments), the driver looks for an installed app and, when
        none matches, hands the name to xdg-open if a file or folder of that
        name is in its working folder: in the person's home, where it ran
        under systemd, `.bash_history` opened in an editor, for a screenshot
        to read, past the path jail and the approval `urls` needs. It runs in
        an empty folder of its own now (CuaDriverBridge._new_working_folder),
        where no name finds anything, even one whose app went between this
        check and the driver's own look. Links to open, and then a
        launch_path, take precedence over the name, so those calls aren't
        checked."""
        name = arguments.get("name")
        platform = self.driver.platform
        if not isinstance(name, str):
            return
        if platform == "win32":
            if any(arguments.get(key) for key in ("launch_path", "path", "aumid")):
                return
            bundle_id = arguments.get("bundle_id")
            if isinstance(bundle_id, str) and is_aumid(bundle_id):
                return
            apps = await self._installed_apps_listed(deadline)
            if not windows_app_name_matches(name, [app_name for app_name, _ in apps]):
                raise ComputerUseError(
                    ErrorCode.FEATURE_NOT_SUPPORTED,
                    f"No installed app here is called {name!r}, so cua-driver would run a "
                    "program of that name from PATH, out of sight. Name the app as list_apps "
                    "shows it, pass its AUMID as aumid, or pass the program as launch_path, "
                    "which the person approves.",
                    details={"tool": "launch_app", "argument": "name"},
                )
        elif platform.startswith("linux") and linux_launch_uses_name(arguments):
            if not linux_app_name_matches(name, await self._installed_apps_listed(deadline)):
                raise ComputerUseError(
                    ErrorCode.FEATURE_NOT_SUPPORTED,
                    f"No installed app here has the name or bundle_id {name!r}, so "
                    "cua-driver would open a file or folder of that name, if there is one, "
                    "without the person's approval. Name the app exactly as list_apps shows "
                    "it, or pass the program as launch_path or the file as urls, which the "
                    "person approves.",
                    details={"tool": "launch_app", "argument": "name"},
                )

    async def _installed_apps_listed(self, deadline: float) -> list[tuple[str, str | None]]:
        """Each installed app's name and bundle_id (on Linux its desktop-file
        id) from list_apps, listed for each launch, in the same turn of the
        computer as the launch: apps kept from an earlier listing could hold
        one removed since, whose name the driver, finding no app, would hand
        to ShellExecuteEx or xdg-open. A running process no installed app
        matched has no launch_path, and the driver's name lookup doesn't find
        it either."""
        loop = asyncio.get_running_loop()
        timeout = deadline - loop.time()
        if timeout <= 0:
            raise CallNotStarted(
                "launch_app's deadline passed before the installed apps were listed"
            )
        raw = await self.call_driver("list_apps", {}, timeout=timeout)
        structured = raw.get("structuredContent")
        apps = structured.get("apps") if isinstance(structured, dict) else None
        if raw.get("isError") or not isinstance(apps, list):
            aumid = "the app's AUMID as aumid, or " if self.driver.platform == "win32" else ""
            raise ComputerUseError(
                ErrorCode.FEATURE_NOT_SUPPORTED,
                "The installed apps couldn't be listed to check this launch's name. Pass "
                f"{aumid}the program as launch_path, which the person approves.",
                details={"tool": "launch_app", "argument": "name"},
            )
        return [
            (app["name"], app["bundle_id"] if isinstance(app.get("bundle_id"), str) else None)
            for app in apps
            if isinstance(app, dict) and isinstance(app.get("name"), str) and app.get("launch_path")
        ]

    async def _clipboard_read(self, arguments: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Relay a clipboard read under the shim's rules: by default only
        text this shim wrote in the last 30 s comes back; with
        --enable-clipboard-read-foreign anything does, unless a password
        manager marked it private."""
        arguments["include_text"] = True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        foreign = self.config.enable_clipboard_read_foreign
        # Each look at the OS gets what is left of the deadline (xclip and
        # wl-paste can take seconds), and so does each read: the call holds
        # the computer's turn throughout, which every chat's calls share.
        # The looks are the machine's, one at a time (_ClipboardChecks).
        before = None
        if foreign:
            try:
                before = await self._clipboard_checks.state_by(deadline)
            except TimeoutError:
                raise CallNotStarted(
                    "clipboard_read's deadline passed in the marker check"
                ) from None
        timeout = deadline - loop.time()
        if timeout <= 0:
            # None left means the read never starts.
            raise CallNotStarted("clipboard_read's deadline passed in the marker check")
        if before is not None and before.marker:
            raise ClipboardConcealedError("concealed_type", marker=before.marker)
        raw = await self.call_driver("clipboard_read", arguments, timeout=timeout)
        if raw.get("isError"):
            return raw
        structured = raw.get("structuredContent")
        structured = structured if isinstance(structured, dict) else {}
        text = structured.get("text")
        if before is not None:
            # The types the driver saw with the text, then the OS again: a
            # copy can land between the first look and the read.
            listed = marker_in(structured.get("types"))
            if listed:
                raise ClipboardConcealedError("concealed_type", marker=listed)
            if before.sequence is None:
                # No sequence number to tell a copy made meanwhile (Linux),
                # and the driver reads the types, then the text: a password
                # manager's copy between the two would pass as clean. Read
                # again; the same text with no marker belongs to one copy.
                # Not past the deadline: a read that overran would have the
                # bridge stop the driver every chat shares.
                left = deadline - loop.time()
                if left <= 0:
                    raise _ClipboardCheckOutOfTime
                again = await self.call_driver("clipboard_read", arguments, timeout=left)
                checked = again.get("structuredContent")
                checked = checked if isinstance(checked, dict) else {}
                listed = marker_in(checked.get("types"))
                if listed:
                    raise ClipboardConcealedError("concealed_type", marker=listed)
                if again.get("isError") or checked.get("text") != text:
                    raise ClipboardConcealedError(
                        "changed",
                        message="The clipboard changed while it was read. Read it again.",
                    )
            left = deadline - loop.time()
            if left <= 0:
                raise _ClipboardCheckOutOfTime
            try:
                after = await self._clipboard_checks.state_by(deadline)
            except TimeoutError:
                raise _ClipboardCheckOutOfTime from None
            if loop.time() >= deadline:
                raise _ClipboardCheckOutOfTime
            if after.marker:
                raise ClipboardConcealedError("concealed_type", marker=after.marker)
            if before.sequence is not None and after.sequence != before.sequence:
                raise ClipboardConcealedError(
                    "changed",
                    message="The clipboard changed while it was read. Read it again.",
                )
            return raw
        if self.clipboard_writeback.check(
            live_content=text if isinstance(text, str) else None, live_sequence=None
        ):
            return raw
        record = self.clipboard_writeback.snapshot()
        if record is None:
            raise ClipboardConcealedError("writeback_only")
        age = record.age()
        if age > self.clipboard_writeback.window_seconds:
            raise ClipboardConcealedError("writeback_only", writeback_age_seconds=age)
        raise ClipboardConcealedError("writeback_overwritten")

    def _result_payload(
        self, tool: str, raw: dict[str, Any]
    ) -> tuple[ComputerToolResultPayload, dict[str, Any]]:
        """The wire result: text and image blocks within the frame limit, at
        most `max_screenshots_per_minute` images, and a note for anything
        left out."""
        content: list[ComputerTextContent | ComputerImageContent] = []
        notes: list[str] = []
        withheld = 0
        # Images past the rate limit are left out here without spending any
        # of it; the limiter is charged at the end, for the images sent. One
        # left out below for size spent nothing, so it can't withhold the
        # next screenshot, this session's or another's.
        room = self._screenshot_rate.available()
        for block in raw.get("content") or []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                content.append(ComputerTextContent(text=block["text"]))
            elif kind == "image" and isinstance(block.get("data"), str):
                if room > 0:
                    room -= 1
                    content.append(
                        ComputerImageContent(
                            data=block["data"],
                            mime_type=str(block.get("mimeType") or "image/png"),
                        )
                    )
                else:
                    withheld += 1
            else:
                notes.append(f"A {kind or 'untyped'} block was left out.")
        if withheld:
            notes.append(
                f"{withheld} screenshot(s) left out: this machine sends at most "
                f"{self.config.max_screenshots_per_minute} a minute. Read the "
                "accessibility tree instead, or ask again shortly."
            )
        structured = raw.get("structuredContent")
        structured = structured if isinstance(structured, dict) else None
        is_error = bool(raw.get("isError"))

        # Every operation's payload stays within the negotiated
        # max_file_size_bytes (PROTOCOL.md), and within the frame: the whole
        # payload as it is sent, its fields, each block's wrapper and the
        # notes added below included.
        budget = min(self.config.max_file_size_bytes, MAX_WIRE_BASE64_CONTENT_BYTES)

        def built() -> ComputerToolResultPayload:
            return ComputerToolResultPayload(
                tool=tool,
                is_error=is_error,
                content=[*content, *(ComputerTextContent(text=note) for note in notes)],
                structured_content=structured,
            )

        def size() -> int:
            return _wire_size(built())

        if size() > budget:
            images = [block for block in content if isinstance(block, ComputerImageContent)]
            content = [block for block in content if isinstance(block, ComputerTextContent)]
            withheld += len(images)
            if images:
                notes.append(
                    f"{len(images)} screenshot(s) left out: the result was too large to "
                    "send. Ask for a smaller image (max_image_dimension) or zoom in."
                )
        if size() > budget:
            structured = None
            first_text = next(
                (block for block in content if isinstance(block, ComputerTextContent)), None
            )
            content = []
            notes.append(_CUT_DOWN_NOTE)
            if first_text is not None:
                # As much of it as fits beside the notes, measured as JSON
                # writes it: a control character takes six bytes there, so
                # its UTF-8 length can be far less.
                content = [ComputerTextContent(text="")]
                room = min(budget - size() + _json_text_size(""), MAX_WIRE_TEXT_CONTENT_BYTES)
                text = _truncate_json_text(first_text.text, room)
                content = [ComputerTextContent(text=text)] if text else []
        if size() > budget:
            # Too small a budget for every note: the one that says the
            # result was cut down, else none. That leaves the smallest result
            # the protocol allows, the tool's name, is_error and no content,
            # which goes even past a budget smaller than itself.
            notes = notes[-1:]
            if size() > budget:
                notes = []
        for block in content:
            if isinstance(block, ComputerImageContent):
                self._screenshot_rate.allow()
        payload = built()
        summary = {
            "is_error": payload.is_error,
            "images": sum(isinstance(block, ComputerImageContent) for block in payload.content),
            "images_withheld": withheld,
            "text_bytes": sum(
                len(block.text.encode("utf-8"))
                for block in payload.content
                if isinstance(block, ComputerTextContent)
            ),
        }
        return payload, summary

    async def _emit(self, request_id: str, details: dict, result: dict) -> None:
        if self.audit is None:
            return
        try:
            await _finish_then_cancel(
                self.audit.write(
                    "COMPUTER_TOOL_CALL", request_id=request_id, details=details, result=result
                )
            )
        except Exception:
            logger.debug("Failed to write COMPUTER_TOOL_CALL audit record", exc_info=True)


# How long HELLO waits for the driver's screen size.
SCREEN_SIZE_SECONDS = 10.0
# How long HELLO waits for the look for the driver (CuaDriverBridge.find_binary)
# at most: well within the ten seconds the platform waits for HELLO.
DRIVER_LOOKUP_SECONDS = 5.0
# How long a chat's session, starting, waits for the driver another chat's
# call may hold (its permission check, HELLO's screen size) before going on
# without it: the platform gives a new chat's session a few seconds (eight)
# to connect, and undoes the chat when it doesn't.
CHILD_DRIVER_WAIT_SECONDS = 2.0

# How long one computer tool call may run unless the platform asks for less
# or more. The driver's own bounded waits (verify_state, run_script) fit.
DEFAULT_COMPUTER_TOOL_TIMEOUT_SECONDS = 120

# The note a computer-use result cut down to its size limit ends with.
_CUT_DOWN_NOTE = "The result was cut down to fit the connection's message limit."

# Arguments whose values the audit log keeps only the length of: what the
# agent typed, pasted, or scripted.
_AUDIT_REDACTED_ARGUMENTS = frozenset({"prompt_text", "script", "text", "value"})
_AUDIT_STRING_LIMIT = 200

# The names cua-driver 0.38's hotkey holds as modifiers, matched lowercased
# but not trimmed: is_modifier in its macOS and Windows hotkey tools, and on
# Linux modifier_keysym, which its XTest input uses, skipping any other name.
# A name outside these holds nothing, so the chord's character is typed.
_DRIVER_MODIFIERS = {
    "darwin": frozenset(
        {
            "cmd",
            "command",
            "super",
            "meta",
            "win",
            "shift",
            "option",
            "alt",
            "ctrl",
            "control",
            "fn",
        }
    ),
    "win32": frozenset({"ctrl", "control", "shift", "alt", "win", "windows", "cmd", "command"}),
}
_LINUX_DRIVER_MODIFIERS = frozenset(
    {"shift", "control", "ctrl", "alt", "option", "command", "cmd", "super", "win", "meta"}
)


def _audit_arguments(
    value: Any, key: str | None = None, *, truncate: bool = True, platform: str | None = None
) -> Any:
    """`value` as the audit log keeps it. `platform` is the machine's
    (sys.platform) unless a test says otherwise: it decides which chords
    type a character."""
    platform = sys.platform if platform is None else platform
    if isinstance(value, dict):
        return {
            k: _audit_arguments(v, k, truncate=truncate, platform=platform)
            for k, v in value.items()
        }
    if isinstance(value, list):
        if key == "keys" and _types_a_character(value, platform):
            return [
                {"redacted_chars": len(item)} if _is_character_key(item) else item for item in value
            ]
        # A `keys` list holds one key each, none a chord written as one.
        item_key = None if key == "keys" else key
        return [
            _audit_arguments(item, item_key, truncate=truncate, platform=platform) for item in value
        ]
    if isinstance(value, str):
        # press_key takes any single character, so a secret can be typed a
        # key at a time; named keys (enter, tab) stay readable. The driver
        # trims the key first.
        if key in _AUDIT_REDACTED_ARGUMENTS or (key == "key" and _is_character(value.strip())):
            return {"redacted_chars": len(value)}
        chord = _chord_written_as_one(value, key)
        if chord is not None and _types_a_character(chord, platform):
            return [{"redacted_chars": 1} if _is_character(part) else part for part in chord]
        if truncate and len(value) > _AUDIT_STRING_LIMIT:
            return value[:_AUDIT_STRING_LIMIT] + f"… ({len(value)} chars)"
    return value


def _is_character(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 1


def _is_character_key(key: Any) -> bool:
    """Whether hotkey presses `key` as a character: one, as given or once
    trimmed, since cua-driver on a Mac trims the key it presses."""
    return _is_character(key) or (isinstance(key, str) and _is_character(key.strip()))


def _chord_written_as_one(value: str, key: str | None) -> list[str] | None:
    """The keys of a chord written as one string, split at "+" as cua-driver
    0.38 splits it before the call (normalize_key_args in its tool.rs):
    `hotkey`'s `keys` always, leaving out empty parts, and `press_key`'s
    `key` once trimmed when it is longer than one character, holds a "+"
    that isn't its last, and has no empty part. None where the driver takes
    the string whole."""
    if key == "keys":
        return [part.strip() for part in value.split("+") if part.strip()]
    if key == "key":
        trimmed = value.strip()
        if len(trimmed) > 1 and "+" in trimmed and not trimmed.endswith("+"):
            parts = [part.strip() for part in trimmed.split("+")]
            if all(parts):
                return parts
    return None


def _types_a_character(keys: list[Any], platform: str) -> bool:
    """Whether a hotkey chord types a character, as against a command chord
    (ctrl+c) whose keys stay readable: Shift and a character on every OS;
    Option too on macOS (Option+L is @ on a German layout); Ctrl with Alt
    on Windows, which reads the pair as AltGr (Ctrl+Alt+Q is @ there).
    Only the names the driver holds as modifiers count (_DRIVER_MODIFIERS):
    it presses or skips any other, so ["capslock", "a"] types a."""
    if not any(_is_character_key(key) for key in keys):
        return False
    modifiers = _DRIVER_MODIFIERS.get(platform, _LINUX_DRIVER_MODIFIERS)
    held = {key.lower() for key in keys if isinstance(key, str)} & modifiers
    if held <= {"shift"}:
        return True
    alt = bool(held & {"alt", "option"})
    if platform == "darwin":
        return alt and held <= {"shift", "alt", "option"}
    if platform == "win32":
        ctrl = bool(held & {"ctrl", "control"})
        return alt and ctrl and held <= {"shift", "alt", "option", "ctrl", "control"}
    return False


def _wire_size(payload: ComputerToolResultPayload) -> int:
    """The payload's bytes as the frame carries them (dump_message writes
    model_dump_json): a control character takes six, a quote or backslash
    two, and its fields and each block's wrapper count too."""
    return len(payload.model_dump_json().encode("utf-8"))


def _json_text_size(text: str) -> int:
    """The bytes `text` takes as a JSON string, quotes and escapes included."""
    return len(json.dumps(text, ensure_ascii=False).encode("utf-8"))


def _truncate_json_text(text: str, limit: int) -> str:
    """The longest start of `text` that takes at most `limit` bytes as a JSON
    string. Every character takes at least one byte, so it is no longer than
    `limit` characters."""
    if _json_text_size(text) <= limit:
        return text
    low, high = 0, min(len(text), max(limit, 0))
    while low < high:
        middle = (low + high + 1) // 2
        if _json_text_size(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _shim_version() -> str:
    try:
        from importlib.metadata import version

        return version("autogpt-local-executor")
    except Exception:
        return "0"


def _instructions(snapshot: DriverSnapshot, config: ShimConfig) -> str:
    """cua-driver's own guidance, then what differs on this machine."""
    offered = {tool["name"] for tool in snapshot.tools}
    if "clipboard_read" not in offered:
        clipboard = "Clipboard tools are off on this machine."
    elif config.enable_clipboard_read_foreign:
        clipboard = (
            "clipboard_read returns what is on the clipboard, except content a password "
            "manager marked private."
        )
    else:
        clipboard = (
            "clipboard_read returns only text clipboard_write put on this computer's "
            "clipboard, in this chat or another, in the last 30 seconds."
        )
    notes = [
        "",
        "On this machine (AutoGPT local executor):",
        "- Calls arrive through the AutoGPT platform. The executor sets `session` on "
        "every call; leave it out.",
        "- Screenshots come back in results, at most "
        f"{config.max_screenshots_per_minute} images a minute.",
        "- Arguments that read or write this computer's disk (each tool's "
        "`approval_arguments`) need approval on every call: the platform asks the person, "
        "unless the chat runs Unsupervised or a standing rule the person set approves the "
        "call without asking. Use one only when the task needs that file. A `run_script` "
        "script needs approval too, since its calls can start a program or save a "
        "screenshot to a file: use `run_steps` or single calls unless the flow needs a "
        "script's logic.",
        "- Open an app with `launch_app` by its `name` as `list_apps` shows it (or its "
        "`bundle_id`) alone. A command "
        "line, a path, links or files to open, extra arguments or an inspector port need "
        "the same approval, since the program they start runs out of sight.",
        f"- {clipboard}",
        "- Only the tools listed are available. Driver configuration, sessions, "
        "recording, process control, downloads and page JavaScript are not.",
    ]
    return snapshot.instructions.rstrip() + "\n" + "\n".join(notes) + "\n"


class OllamaBackendError(Exception):
    """Raised by OllamaBackend when the upstream HTTP call fails.

    Carries the ErrorCode the LocalLLMHandler should surface plus the raw
    backend error string for `details.backend_error` on the wire.
    """

    def __init__(self, code: ErrorCode, backend_error: str) -> None:
        self.code = code
        self.backend_error = backend_error
        super().__init__(f"{code.value}: {backend_error}")


class OllamaBackend:
    """Thin async wrapper over Ollama's HTTP API.

    Only the two endpoints the shim needs: GET /api/tags for model probing
    at HELLO time, and POST /api/chat (streaming or one-shot) for completion
    requests. Mocked via httpx.AsyncClient in tests.
    """

    def __init__(self, base_url: str, *, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def list_models(self) -> list[str]:
        """Return the list of currently-loaded model names.

        Raises OllamaBackendError on transport failure. Caller (HELLO probe)
        catches and translates to "no local_llm capability".
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
        except httpx.RequestError as exc:
            raise OllamaBackendError(ErrorCode.LOCAL_LLM_FAILED, str(exc)) from exc
        if r.status_code != 200:
            raise OllamaBackendError(
                ErrorCode.LOCAL_LLM_FAILED,
                f"HTTP {r.status_code} from /api/tags",
            )
        try:
            data = r.json()
        except ValueError as exc:
            raise OllamaBackendError(ErrorCode.LOCAL_LLM_FAILED, f"bad JSON: {exc}") from exc
        models = data.get("models") or []
        return [m["name"] for m in models if isinstance(m, dict) and "name" in m]

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """Yield streaming response chunks from /api/chat.

        Each yielded value is the parsed JSON line dict. Caller stitches
        `chunk["message"]["content"]` deltas together and uses the final
        line's `done: true` + token counters for the terminal frame.

        Raises OllamaBackendError on connection errors (LOCAL_LLM_FAILED)
        and 404 (MODEL_NOT_AVAILABLE). 503 also maps to LOCAL_LLM_FAILED.
        """
        import httpx

        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        # Ollama sends its response headers with the first token, once it has
        # loaded the model and read the prompt, so a read may wait as long as
        # the whole request. bounded_chunks bounds the gaps after that.
        timeout = httpx.Timeout(self.timeout_seconds, read=local_llm_limits.TOTAL_TIMEOUT_SECONDS)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=body) as r:
                    if r.status_code == 404:
                        raise OllamaBackendError(ErrorCode.MODEL_NOT_AVAILABLE, "model not found")
                    if r.status_code != 200:
                        raise OllamaBackendError(
                            ErrorCode.LOCAL_LLM_FAILED, f"HTTP {r.status_code}"
                        )
                    # What has arrived, as it arrives: with a chunk_size, httpx
                    # holds bytes back until it has that many, and a model
                    # sending short lines slowly looked idle (bounded_chunks).
                    # read_json_lines bounds the lines.
                    async for chunk in read_json_lines(r.aiter_bytes()):
                        yield chunk
        except httpx.RequestError as exc:
            raise OllamaBackendError(ErrorCode.LOCAL_LLM_FAILED, str(exc)) from exc


class LocalLLMHandler:
    """Dispatch LOCAL_LLM_COMPLETION requests to the local backend.

    Streams response chunks back as LOCAL_LLM_COMPLETION_CHUNK frames and
    emits a terminal LOCAL_LLM_COMPLETION_RESPONSE with the assembled
    output + token accounting.

    Concurrency: Ollama serializes requests by default. We track active
    requests via asyncio.Lock with a try-acquire; a second request that
    arrives while one is in flight gets LOCAL_LLM_BUSY immediately rather
    than queueing (the platform's auto-retry layer is the better place
    for that decision).
    """

    def __init__(
        self,
        config: ShimConfig,
        audit: AuditWriter | None = None,
        backend: OllamaBackend | None = None,
    ) -> None:
        self.config = config
        self.audit = audit
        self._backend = backend or OllamaBackend(config.ollama_url)
        # One-at-a-time semaphore for the underlying backend. Ollama is
        # single-threaded by default; the LOCAL_LLM_BUSY error code is the
        # contract the platform retries against.
        self._inflight_lock = asyncio.Lock()
        # Cached model list from the most-recent probe(). Populated at HELLO
        # emission time so we can reject unknown models without a round trip.
        self._models: list[str] = []

    @property
    def models(self) -> list[str]:
        return list(self._models)

    async def probe(self) -> list[str]:
        """Query Ollama for loaded models. Called at HELLO emission.

        Returns the model list on success; returns [] AND swallows the
        OllamaBackendError on failure. Caller (daemon._build_hello) uses
        an empty list to mean "omit local_llm capability + no advertised
        models" per LOCAL_LLM.md.
        """
        try:
            models = await self._backend.list_models()
        except OllamaBackendError as exc:
            logger.debug("Ollama probe failed: %s", exc)
            self._models = []
            return []
        self._models = models
        return models

    async def handle(
        self,
        msg: LocalLLMCompletionMessage,
        *,
        send: Any | None = None,
    ) -> LocalLLMCompletionResponseMessage | ErrorMessage:
        """Process a completion request.

        `send` is an async callable that takes a single envelope and pushes
        it on the WS — used to emit CHUNK frames mid-stream. Returns the
        terminal RESPONSE (or ERROR) for the standard dispatch path to send.
        When `send` is None (tests), CHUNK emissions are silently dropped
        and only the terminal frame is returned.
        """
        payload = msg.payload
        start = time.monotonic()
        details_base = {
            "model": payload.model,
            "stream": payload.stream,
            "prompt_chars": sum(len(m.content) for m in payload.messages),
        }

        # Pre-check model against our cached probe. Backend will re-confirm
        # (404 → MODEL_NOT_AVAILABLE) but failing fast here saves a round
        # trip when the platform's local_llm_models list is stale.
        if self._models and payload.model not in self._models:
            await self._emit(
                msg.id,
                {**details_base, "response_chars": 0, "finish_reason": None},
                _err_result(_elapsed_ms(start), ErrorCode.MODEL_NOT_AVAILABLE.value),
            )
            return make_error(
                msg.id,
                ErrorCode.MODEL_NOT_AVAILABLE,
                f"Model {payload.model!r} not in advertised list.",
                details={
                    "requested_model": payload.model,
                    "available_models": list(self._models),
                },
            )

        # Backpressure: Ollama only serves one request at a time. A second
        # caller while another is in-flight gets LOCAL_LLM_BUSY.
        if self._inflight_lock.locked():
            await self._emit(
                msg.id,
                {**details_base, "response_chars": 0, "finish_reason": None},
                _err_result(_elapsed_ms(start), ErrorCode.LOCAL_LLM_BUSY.value),
            )
            return make_error(
                msg.id,
                ErrorCode.LOCAL_LLM_BUSY,
                "Local LLM backend is already processing another request.",
            )

        async with self._inflight_lock:
            try:
                async with asyncio.timeout(local_llm_limits.TOTAL_TIMEOUT_SECONDS):
                    return await self._run_completion(
                        msg, send=send, details_base=details_base, start=start
                    )
            except TimeoutError:
                await self._emit(
                    msg.id,
                    {**details_base, "response_chars": 0, "finish_reason": None},
                    _err_result(_elapsed_ms(start), ErrorCode.LOCAL_LLM_FAILED.value),
                )
                return make_error(msg.id, ErrorCode.LOCAL_LLM_FAILED, "Local model timed out")

    async def _run_completion(
        self,
        msg: LocalLLMCompletionMessage,
        *,
        send: Any | None,
        details_base: dict,
        start: float,
    ) -> LocalLLMCompletionResponseMessage | ErrorMessage:
        payload = msg.payload
        messages_wire = [{"role": m.role, "content": m.content} for m in payload.messages]
        content_buf: list[str] = []
        finish_reason: LocalLLMFinishReason = "stop"
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        stream = bounded_chunks(
            self._backend.chat_stream(
                model=payload.model,
                messages=messages_wire,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
            )
        )
        try:
            async for chunk in stream:
                delta = ""
                msg_obj = chunk.get("message") or {}
                if isinstance(msg_obj, dict):
                    delta = msg_obj.get("content") or ""
                done = bool(chunk.get("done"))

                # Final-chunk token accounting from Ollama. Field names per
                # the documented Ollama /api/chat response.
                if done:
                    if "prompt_eval_count" in chunk:
                        prompt_tokens = int(chunk["prompt_eval_count"])
                    if "eval_count" in chunk:
                        completion_tokens = int(chunk["eval_count"])
                    # Ollama uses `done_reason` ("stop" | "length") on the
                    # terminal frame. Default to "stop" when unspecified.
                    raw_reason = chunk.get("done_reason") or "stop"
                    if raw_reason in ("stop", "length", "content_filter"):
                        finish_reason = raw_reason  # type: ignore[assignment]
                    else:
                        finish_reason = "stop"

                if delta:
                    content_buf.append(delta)
                    if payload.stream and send is not None:
                        chunk_msg = LocalLLMCompletionChunkMessage(
                            id=msg.id,
                            ts=now_ts(),
                            payload=LocalLLMCompletionChunkPayload(
                                delta=delta,
                                finish_reason=None,
                            ),
                        )
                        async with asyncio.timeout(local_llm_limits.IDLE_TIMEOUT_SECONDS):
                            await send(chunk_msg)

                if done:
                    # Emit terminal-marker chunk (delta="", finish_reason=stop/length)
                    # before the response. Platform consumers that only watch
                    # CHUNK frames see the stream close before the metadata.
                    if payload.stream and send is not None:
                        chunk_msg = LocalLLMCompletionChunkMessage(
                            id=msg.id,
                            ts=now_ts(),
                            payload=LocalLLMCompletionChunkPayload(
                                delta="",
                                finish_reason=finish_reason,
                            ),
                        )
                        async with asyncio.timeout(local_llm_limits.IDLE_TIMEOUT_SECONDS):
                            await send(chunk_msg)
                    break
        except OllamaBackendError as exc:
            details = {
                **details_base,
                "response_chars": sum(len(c) for c in content_buf),
                "finish_reason": None,
            }
            await self._emit(
                msg.id,
                details,
                _err_result(_elapsed_ms(start), exc.code.value),
            )
            return make_error(
                msg.id,
                exc.code,
                _local_llm_error_message(exc.code, exc.backend_error),
                details={"backend_error": exc.backend_error},
            )
        except Exception as exc:
            logger.exception("LocalLLMHandler crashed")
            await self._emit(
                msg.id,
                {**details_base, "response_chars": 0, "finish_reason": None},
                _err_result(_elapsed_ms(start), ErrorCode.LOCAL_LLM_FAILED.value),
            )
            return make_error(
                msg.id,
                ErrorCode.LOCAL_LLM_FAILED,
                str(exc),
                details={"backend_error": str(exc)},
            )

        finally:
            await stream.aclose()

        duration = time.monotonic() - start
        content = "".join(content_buf)
        total_tokens: int | None = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        details = {
            **details_base,
            "response_chars": len(content),
            "finish_reason": finish_reason,
            "tokens_prompt": prompt_tokens,
            "tokens_completion": completion_tokens,
            "tokens_total": total_tokens,
        }
        await self._emit(msg.id, details, _ok_result(_elapsed_ms(start)))

        return LocalLLMCompletionResponseMessage(
            id=msg.id,
            ts=now_ts(),
            payload=LocalLLMCompletionResponsePayload(
                content="" if payload.stream and send is not None else content,
                finish_reason=finish_reason,
                tokens=LocalLLMTokensUsage(
                    prompt=prompt_tokens,
                    completion=completion_tokens,
                    total=total_tokens,
                ),
                duration_seconds=round(duration, 3),
            ),
        )

    async def _emit(self, request_id: str, details: dict, result: dict) -> None:
        if self.audit is None:
            return
        try:
            await _finish_then_cancel(
                self.audit.write(
                    "LOCAL_LLM_COMPLETION",
                    request_id=request_id,
                    details=details,
                    result=result,
                )
            )
        except Exception:
            logger.debug("Failed to write LOCAL_LLM_COMPLETION audit record", exc_info=True)


def _local_llm_error_message(code: ErrorCode, backend_error: str) -> str:
    """Human-readable wire message that the platform-side translator turns
    into a user-facing message. Keep terse — the backend_error in details
    carries the verbose context."""
    if code == ErrorCode.MODEL_NOT_AVAILABLE:
        return "Requested local LLM model is not loaded on the shim."
    if code == ErrorCode.LOCAL_LLM_BUSY:
        return "Local LLM backend is already serving another request."
    return f"Local LLM backend failed: {backend_error[:200]}"


# ── Workflow recording ────────────────────────────────────────────────────────


class RecordingHandler:
    """Handles START / STOP / FETCH recording wire ops (§6).

    Enforces one-recording-at-a-time (RECORDING_ALREADY_ACTIVE), validates the
    shim-issued consent token (CONSENT_REQUIRED), probes/honors the
    interpretation route, spins a RecordingSession, and consumes the
    CaptureSource:

      * co-pilot mode → streams RECORDING_STEP frames via the STATUS-frame send
        path, so they're exempt from in-flight accounting (§6).
      * demonstration mode → buffers silently; the platform pulls via
        RECORDING_FETCH after STOP + user approval (§6).

    Audit is content-free per §9: recording_id, channels, step_count,
    interpretation_route — NEVER step content.

    The CaptureSource is injected (`capture_factory`) so tests drive a scripted
    MockCaptureSource and production wires the real floor + a11y enricher.
    """

    def __init__(
        self,
        config: ShimConfig,
        audit: AuditWriter | None = None,
        *,
        consent_broker: ConsentBroker | None = None,
        capture_factory: Any | None = None,
        recording_cipher: RecordingCipher | None = None,
        consent_prompt: Any | None = None,
    ) -> None:
        self.config = config
        self.audit = audit
        self.consent = consent_broker or ConsentBroker()
        # capture_factory(session) -> CaptureSource. Defaults to the real floor
        # (+ a11y enricher) wrapping the computer-use backend. Tests inject a
        # scripted MockCaptureSource here.
        self._capture_factory = capture_factory or self._default_capture_factory
        self._using_injected_capture = capture_factory is not None
        self._recording_cipher = recording_cipher
        self._consent_prompt = consent_prompt or request_user_consent
        initial_channels = (
            list(config.recording_channels)
            if self._using_injected_capture
            else [
                channel
                for channel in config.recording_channels
                if channel in set(platform_info.available_recording_channels())
            ]
        )
        self._available_channels = list(dict.fromkeys(initial_channels))
        self._local_llm_models: list[str] = []
        self._sessions: OrderedDict[str, RecordingSession] = OrderedDict()
        self._active_recording_id: str | None = None
        # The background task draining the CaptureSource (co-pilot streaming or
        # demonstration buffering).
        self._consume_task: asyncio.Task[None] | None = None

    # ── Capture wiring (the seam) ─────────────────────────────────────────

    def _default_capture_factory(self, session: RecordingSession) -> CaptureSource:
        """Build the production macOS chain: Quartz input tap → floor → a11y."""
        if platform_info.detect_platform() != "darwin":
            raise RecordingError("workflow recording has no native capture source on this OS")
        try:
            from .recording.macos_capture import MacInputCaptureSource
        except ImportError as exc:
            raise RecordingError("macOS recording dependencies are unavailable") from exc

        input_events = MacInputCaptureSource()
        if not input_events.start():
            raise RecordingError(
                "macOS Input Monitoring permission is required for workflow recording"
            )
        floor = ScreenshotActionFloor(input_events=input_events, config=self.config)
        return A11yEnricher(floor=floor)

    def set_available_channels(self, channels: list[str]) -> None:
        """Update channels from the daemon's authoritative HELLO probe."""
        self._available_channels = list(dict.fromkeys(channels))

    def set_local_llm_models(self, models: list[str]) -> None:
        """Expose the daemon's successful Ollama probe to route selection."""
        self._local_llm_models = list(dict.fromkeys(models))

    def _validate_channels(self, msg_id: str, channels: list[str]) -> ErrorMessage | None:
        requested = list(dict.fromkeys(channels))
        available = set(self._available_channels)
        unavailable = [channel for channel in requested if channel not in available]
        if requested and not unavailable:
            return None
        return make_error(
            msg_id,
            ErrorCode.RECORDING_CHANNEL_UNAVAILABLE,
            "Requested recording channels are unavailable on this shim.",
            details={
                "requested_channels": requested,
                "available_channels": self._available_channels,
            },
        )

    # ── Dispatch ──────────────────────────────────────────────────────────

    async def handle(self, msg: Any, *, send: Any | None = None) -> Any:
        if not self.config.enable_recording:
            return make_error(
                msg.id,
                ErrorCode.CAPABILITY_NOT_GRANTED,
                "Workflow recording is not enabled on this shim.",
            )
        if isinstance(msg, RequestRecordingConsentMessage):
            return await self._request_consent(msg)
        if isinstance(msg, StartRecordingMessage):
            return await self._start(msg, send=send)
        if isinstance(msg, StopRecordingMessage):
            return await self._stop(msg)
        if isinstance(msg, ApplyRecordingReviewMessage):
            return await self._apply_review(msg)
        if isinstance(msg, RecordingFetchMessage):
            return await self._fetch(msg)
        return make_error(
            msg.id,
            ErrorCode.INTERNAL_ERROR,
            f"Unknown recording message: {type(msg).__name__}",
        )

    async def _request_consent(self, msg: RequestRecordingConsentMessage) -> Any:
        p = msg.payload
        channel_error = self._validate_channels(msg.id, list(p.channels))
        if channel_error is not None:
            return channel_error
        decision = probe_interpretation_route(
            channels=p.channels,
            local_llm_models=list(self._advertised_models()),
            requested=p.interpretation_route,
        )
        token = await asyncio.to_thread(
            self._consent_prompt,
            self.consent,
            mode=p.mode,
            interpretation_route=decision.route,
        )
        approved = bool(token)
        return RecordingConsentResultMessage(
            id=msg.id,
            ts=now_ts(),
            payload=RecordingConsentResultPayload(
                approved=approved,
                mode=p.mode,
                interpretation_route=decision.route,
                consent_token=token if approved else None,
                expires_at=(time.time() + CONSENT_TOKEN_TTL_SECONDS if approved else None),
            ),
        )

    # ── START ─────────────────────────────────────────────────────────────

    async def _start(self, msg: StartRecordingMessage, *, send: Any | None) -> Any:
        p = msg.payload

        channel_error = self._validate_channels(msg.id, list(p.channels))
        if channel_error is not None:
            return channel_error

        # One-at-a-time. A START while a session is live → RECORDING_ALREADY_ACTIVE.
        if self._active_recording_id is not None:
            return make_error(
                msg.id,
                ErrorCode.RECORDING_ALREADY_ACTIVE,
                "A recording is already in progress.",
                details={"recording_id": self._active_recording_id},
            )

        # Probe the interpretation route first. The consent token approves one
        # route, and a machine that can't honor it must not fall back to
        # another under that approval: the fallback can be
        # screenshots_to_cloud, which sends frames off the machine. Probing
        # before the token is checked also leaves the single-use token unspent.
        decision = probe_interpretation_route(
            channels=p.channels,
            local_llm_models=list(self._advertised_models()),
            requested=p.interpretation_route,
        )
        if decision.route != p.interpretation_route:
            return make_error(
                msg.id,
                ErrorCode.CONSENT_REQUIRED,
                f"This machine can't use the approved interpretation route "
                f"{p.interpretation_route!r}; {decision.route!r} needs its own consent.",
                details={"available_route": decision.route},
            )

        # Consent: the token must be a valid, shim-issued, single-use token.
        # A platform that self-asserts (no/forged token) gets CONSENT_REQUIRED.
        try:
            self.consent.validate_consent_token(
                p.consent_token,
                mode=p.mode,
                interpretation_route=p.interpretation_route,
            )
        except ConsentError as exc:
            return make_error(
                msg.id,
                ErrorCode.CONSENT_REQUIRED,
                f"Valid shim-issued consent token required: {exc}",
            )

        session = RecordingSession(
            machine_id=self.config.machine_id,
            mode=p.mode,
            interpretation_route=decision.route,
            channels=list(p.channels),
            buffer_dir=self.config.derived_recording_buffer_dir,
            cipher=self._recording_cipher,
        )
        try:
            source = self._capture_factory(session)
        except Exception as exc:
            logger.warning("Could not start recording capture: %s", exc)
            return make_error(
                msg.id,
                ErrorCode.RECORDING_CHANNEL_UNAVAILABLE,
                f"Could not start recording capture: {exc}",
                details={"available_channels": self._available_channels},
            )
        session.start()
        self._sessions[session.recording_id] = session
        self._active_recording_id = session.recording_id
        self._prune_recordings()

        await self._audit_lifecycle("RECORDING_STARTED", session)

        # Begin consuming the capture source. Co-pilot streams; demonstration
        # buffers silently.
        stream = p.mode == "copilot"
        self._consume_task = asyncio.create_task(
            self._consume(
                session,
                source=source,
                send=send if stream else None,
                stream=stream,
            )
        )

        return RecordingStartedMessage(
            id=msg.id,
            ts=now_ts(),
            payload=RecordingStartedPayload(recording_id=session.recording_id),
        )

    def _advertised_models(self) -> list[str]:
        return list(self._local_llm_models)

    def _prune_recordings(self) -> None:
        while len(self._sessions) > self.config.recording_retention_limit:
            recording_id, session = self._sessions.popitem(last=False)
            if recording_id == self._active_recording_id:
                self._sessions[recording_id] = session
                break
            session.close()

    async def _consume(
        self,
        session: RecordingSession,
        *,
        source: CaptureSource,
        send: Any | None,
        stream: bool,
    ) -> None:
        """Drain the CaptureSource into the session.

        In co-pilot mode (`stream=True`, `send` provided) each step is emitted
        as an unsolicited RECORDING_STEP frame via the STATUS-frame send path —
        exempt from in-flight / max_concurrent accounting (§6). In
        demonstration mode the step is only buffered; the platform fetches
        after STOP + approval.
        """
        try:
            async for step in source.steps():
                session.append(step)
                if stream and send is not None:
                    frame = RecordingStepMessage(
                        id=new_id(),
                        ts=now_ts(),
                        payload=RecordingStepPayload(
                            recording_id=session.recording_id,
                            step=step,
                        ),
                    )
                    try:
                        await send(frame)
                    except Exception:
                        logger.debug("Failed to send RECORDING_STEP frame", exc_info=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Recording capture loop crashed for %s", session.recording_id)

    # ── STOP ──────────────────────────────────────────────────────────────

    async def _stop(self, msg: StopRecordingMessage) -> Any:
        session = self._sessions.get(msg.payload.recording_id)
        if (
            session is None
            or self._active_recording_id != msg.payload.recording_id
            or not session.is_active
        ):
            return make_error(
                msg.id,
                ErrorCode.RECORDING_NOT_FOUND,
                f"No active recording with id {msg.payload.recording_id!r}.",
            )

        # Stop capture and finalize. Steps are buffered synchronously by
        # `append` as they're captured, so by STOP time everything the source
        # has produced is already in the session. A finite source (a
        # fully-played demonstration, or the test MockCaptureSource) may not
        # have had a turn yet, so we yield once to let it run to completion;
        # then we cancel whatever is still running (an infinite real OS
        # input-hook source) to halt further capture.
        if self._consume_task is not None:
            # Yield up to a bounded number of turns to let a finite source drain
            # fully; stop early as soon as it completes. Bounded so an infinite
            # real source falls through to cancellation rather than spinning.
            for _ in range(64):
                if self._consume_task.done():
                    break
                await asyncio.sleep(0)
            if not self._consume_task.done():
                self._consume_task.cancel()
            try:
                await self._consume_task
            except (asyncio.CancelledError, Exception):
                pass
            self._consume_task = None

        session.stop()
        self._active_recording_id = None
        await self._audit_lifecycle("RECORDING_STOPPED", session)

        return RecordingSummaryMessage(
            id=msg.id,
            ts=now_ts(),
            payload=RecordingSummaryPayload(
                recording_id=session.recording_id,
                step_count=session.step_count,
                enrichment_coverage=session.enrichment_coverage(),
                duration_seconds=round(session.duration_seconds(), 3),
            ),
        )

    # ── FETCH ─────────────────────────────────────────────────────────────

    async def _apply_review(self, msg: ApplyRecordingReviewMessage) -> Any:
        session = self._sessions.get(msg.payload.recording_id)
        if session is None:
            return make_error(
                msg.id,
                ErrorCode.RECORDING_NOT_FOUND,
                f"No recording with id {msg.payload.recording_id!r}.",
            )
        try:
            step_count = session.apply_review(
                removed_step_seqs=msg.payload.removed_step_seqs,
                redacted_step_seqs=msg.payload.redacted_step_seqs,
            )
        except RecordingError as exc:
            return make_error(msg.id, ErrorCode.RECORDING_REVIEW_INVALID, str(exc))
        await self._audit_lifecycle("RECORDING_REVIEW_APPLIED", session)
        return RecordingReviewAppliedMessage(
            id=msg.id,
            ts=now_ts(),
            payload=RecordingReviewAppliedPayload(
                recording_id=session.recording_id,
                step_count=step_count,
            ),
        )

    async def _fetch(self, msg: RecordingFetchMessage) -> Any:
        session = self._sessions.get(msg.payload.recording_id)
        if session is None:
            return make_error(
                msg.id,
                ErrorCode.RECORDING_NOT_FOUND,
                f"No recording with id {msg.payload.recording_id!r}.",
            )
        await self._audit_lifecycle("RECORDING_FETCHED", session)
        return RecordingDataMessage(
            id=msg.id,
            ts=now_ts(),
            payload=RecordingDataPayload(recording=session.to_recording()),
        )

    # ── Audit (content-free, §9) ──────────────────────────────────────────

    async def _audit_lifecycle(self, op: str, session: RecordingSession) -> None:
        """Emit a content-free recording-lifecycle audit record (§9).

        Records recording_id, channels, step_count, interpretation_route, mode
        — NEVER step content (no actions, values, screenshots, narration).
        """
        if self.audit is None:
            return
        details = {
            "recording_id": session.recording_id,
            "channels": list(session.channels),
            "step_count": session.step_count,
            "interpretation_route": session.interpretation_route,
            "mode": session.mode,
        }
        try:
            await _finish_then_cancel(self.audit.write(op, request_id=None, details=details))
        except Exception:
            logger.debug("Failed to write %s audit record", op, exc_info=True)


__all__ = [
    "CommandHandler",
    "ComputerUseHandler",
    "FileHandler",
    "LocalLLMHandler",
    "OllamaBackend",
    "OllamaBackendError",
    "RecordingHandler",
]
