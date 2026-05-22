"""LocalPCShim-side file operations.

When the active executor is a :class:`LocalPCShim`, the platform's file
tools delegate here instead of running POSIX shell commands. Each helper
maps directly onto a wire-level FILE_* message (defined in PROTOCOL.md)
and trusts the shim to enforce the real path jail (CROSS_PLATFORM.md
"Path Jail Strategy"). Platform-side validation in :func:`resolve_executor_path`
is a best-effort lexical check, not a security boundary.
"""

from __future__ import annotations

import os
from typing import Any

from backend.copilot.context import resolve_executor_path
from backend.copilot.tools.local_pc_shim import LocalPCShim


async def read_file_via_shim(
    sandbox: LocalPCShim,
    file_path: str,
    *,
    as_bytes: bool = True,
) -> bytes | str:
    """FILE_READ — returns bytes by default to mirror E2B's `format="bytes"`.

    Raises :class:`OSError` on shim ERROR (e.g. PATH_OUTSIDE_ALLOWED_ROOT,
    PATH_NOT_FOUND).
    """
    remote = resolve_executor_path(file_path, sandbox)
    return await sandbox.files.read(remote, format="bytes" if as_bytes else "text")


async def write_file_via_shim(
    sandbox: LocalPCShim,
    file_path: str,
    content: str | bytes,
) -> None:
    """FILE_WRITE — no `/tmp` base64 trick; the shim runs file and command
    ops as the same OS user, so the E2B sticky-bit workaround in
    :func:`_sandbox_write` doesn't apply."""
    remote = resolve_executor_path(file_path, sandbox)
    await sandbox.files.write(remote, content)


async def stat_via_shim(
    sandbox: LocalPCShim,
    file_path: str,
    *,
    follow_symlinks: bool = True,
) -> dict:
    """FILE_STAT — portable replacement for `readlink -f`, `stat`, `test -e`."""
    remote = resolve_executor_path(file_path, sandbox)
    return await sandbox.files.stat(remote, follow_symlinks=follow_symlinks)


async def list_via_shim(
    sandbox: LocalPCShim,
    path: str,
    *,
    glob: str | None = None,
    recursive: bool = False,
    include_hidden: bool = False,
    max_entries: int = 1000,
) -> dict:
    """FILE_LIST — portable replacement for shell `ls` / `find`."""
    remote = resolve_executor_path(path, sandbox)
    return await sandbox.files.list(
        remote,
        glob=glob,
        recursive=recursive,
        include_hidden=include_hidden,
        max_entries=max_entries,
    )


async def delete_via_shim(
    sandbox: LocalPCShim,
    path: str,
    *,
    recursive: bool = False,
    missing_ok: bool = False,
) -> None:
    """FILE_DELETE — portable replacement for shell `rm` / `del`."""
    remote = resolve_executor_path(path, sandbox)
    await sandbox.files.delete(remote, recursive=recursive, missing_ok=missing_ok)


async def move_via_shim(
    sandbox: LocalPCShim,
    src: str,
    dst: str,
    *,
    overwrite: bool = False,
) -> None:
    """FILE_MOVE — portable replacement for shell `mv` / `move`."""
    remote_src = resolve_executor_path(src, sandbox)
    remote_dst = resolve_executor_path(dst, sandbox)
    await sandbox.files.move(remote_src, remote_dst, overwrite=overwrite)


def is_local_pc(sandbox: Any) -> bool:
    """Type-guard helper. Cheap isinstance check.

    Use this at the top of e2b_file_tools handlers to branch on executor
    kind before falling into E2B-specific shell quirks.
    """
    return isinstance(sandbox, LocalPCShim)


def describe_workspace(sandbox: Any) -> str:
    """Render a human-readable workspace description for MCP tool prompts.

    E2B: "cloud sandbox (/home/user or /tmp)".
    LocalPCShim: the shim's allowed_root, with the OS surfaced so the LLM
    knows it's writing to a real machine.

    Used by tool description renderers to avoid hard-coding `/home/user`
    in prompts that may target a shim with allowed_root=`C:\\workspace`.
    """
    if isinstance(sandbox, LocalPCShim):
        platform_display = {
            "darwin": "macOS",
            "linux": "Linux",
            "windows": "Windows",
            "wsl2": "Windows (WSL2)",
        }.get(sandbox.platform, sandbox.platform or "local machine")
        root = sandbox.allowed_root or "<unconfigured workspace>"
        return f"local workspace on {platform_display} ({root})"
    return "cloud sandbox (/home/user or /tmp)"
