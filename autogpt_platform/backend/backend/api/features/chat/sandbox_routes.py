"""REST + WebSocket endpoints that proxy a chat session's E2B sandbox.

Everything here is lazy and on-demand — we never sync the whole sandbox
folder. The E2B API key stays server-side; the browser only talks to these
endpoints. All routes are gated behind the ``autogpt-new-layout-ide`` flag.
"""

import asyncio
import json
import logging
import posixpath
import shlex
from typing import Annotated, Literal

from autogpt_libs import auth
from autogpt_libs.auth.jwt_utils import parse_jwt_token
from e2b import AsyncSandbox, CommandExitException, FileType, PtySize
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    Security,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from backend.copilot.config import ChatConfig
from backend.copilot.context import E2B_WORKDIR
from backend.copilot.model import get_chat_session_metadata
from backend.copilot.tools.e2b_sandbox import get_or_create_sandbox
from backend.util.exceptions import NotFoundError
from backend.util.feature_flag import (
    Flag,
    create_feature_flag_dependency,
    is_feature_enabled,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Stored in a module-level variable so tests can override the exact
# dependency object via ``app.dependency_overrides``.
sandbox_flag_dependency = create_feature_flag_dependency(Flag.AUTOGPT_NEW_LAYOUT_IDE)

_MAX_FILE_BYTES = 1024 * 1024
_MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
_DOWNLOAD_TMP_PATH = "/tmp/__ws.tar.gz"


class SandboxTreeEntry(BaseModel):
    name: str
    path: str  # workspace-relative
    type: Literal["file", "dir"]


class SandboxTreeResponse(BaseModel):
    entries: list[SandboxTreeEntry]


class SandboxFileResponse(BaseModel):
    path: str
    content: str
    truncated: bool = False


class SandboxWriteFileRequest(BaseModel):
    path: str = Field(max_length=4096)
    content: str = Field(max_length=1_000_000)


class SandboxChangedFile(BaseModel):
    path: str
    status: Literal["M", "A", "D", "R", "?"]


class SandboxChangesResponse(BaseModel):
    is_git_repo: bool
    files: list[SandboxChangedFile]


class SandboxDiffResponse(BaseModel):
    path: str
    original: str
    modified: str


def _resolve_workspace_path(rel_path: str) -> str:
    """Resolve a client-supplied relative path inside E2B_WORKDIR; reject escapes."""
    cleaned = posixpath.normpath(posixpath.join(E2B_WORKDIR, rel_path.lstrip("/")))
    if cleaned != E2B_WORKDIR and not cleaned.startswith(E2B_WORKDIR + "/"):
        raise ValueError(f"Path escapes workspace: {rel_path}")
    return cleaned


def _safe_resolve(rel_path: str) -> str:
    """``_resolve_workspace_path`` but mapping the escape to a 400."""
    try:
        return _resolve_workspace_path(rel_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _to_workspace_relative(abs_path: str) -> str:
    if abs_path == E2B_WORKDIR:
        return ""
    prefix = E2B_WORKDIR + "/"
    return abs_path[len(prefix) :] if abs_path.startswith(prefix) else abs_path


def _parse_porcelain(text: str) -> list[SandboxChangedFile]:
    """Parse ``git status --porcelain`` output into changed-file records."""
    files: list[SandboxChangedFile] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        xy = line[:2]
        rest = line[3:]
        if xy == "??":
            files.append(SandboxChangedFile(path=rest, status="?"))
            continue
        path = rest.split(" -> ")[-1] if "->" in rest else rest
        files.append(SandboxChangedFile(path=path, status=_first_status(xy)))
    return files


def _first_status(xy: str) -> Literal["M", "A", "D", "R", "?"]:
    for ch in xy:
        if ch == "M":
            return "M"
        if ch == "A":
            return "A"
        if ch == "D":
            return "D"
        if ch == "R":
            return "R"
    return "M"


async def _run_command(
    sandbox: AsyncSandbox, command: str, timeout: float = 30
) -> tuple[str, str, int]:
    """Run a bash command in the workspace, returning (stdout, stderr, exit_code).

    Non-zero exits are captured rather than raised so callers can branch on them.
    """
    try:
        result = await sandbox.commands.run(
            f"bash -c {shlex.quote(command)}", cwd=E2B_WORKDIR, timeout=timeout
        )
        return result.stdout, result.stderr, result.exit_code
    except CommandExitException as exc:
        return exc.stdout, exc.stderr, exc.exit_code


async def _validate_session(session_id: str, user_id: str) -> None:
    session = await get_chat_session_metadata(session_id, user_id)
    if not session:
        raise NotFoundError(f"Session {session_id} not found.")


async def _get_sandbox_for_session(session_id: str, user_id: str) -> AsyncSandbox:
    await _validate_session(session_id, user_id)
    cfg = ChatConfig()
    if not cfg.e2b_active:
        raise NotFoundError("Sandbox not available for this session.")
    assert cfg.e2b_api_key  # guaranteed by e2b_active check
    return await get_or_create_sandbox(
        session_id,
        cfg.e2b_api_key,
        cfg.e2b_sandbox_timeout,
        template=cfg.e2b_sandbox_template,
        on_timeout=cfg.e2b_sandbox_on_timeout,
    )


@router.get(
    "/sessions/{session_id}/sandbox/tree",
    dependencies=[Depends(sandbox_flag_dependency)],
)
async def get_sandbox_tree(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
    path: str = Query(default=""),
) -> SandboxTreeResponse:
    """List a single directory in the sandbox workspace (no recursion)."""
    abs_path = _safe_resolve(path)
    sandbox = await _get_sandbox_for_session(session_id, user_id)
    raw_entries = await sandbox.files.list(abs_path)
    entries = [
        SandboxTreeEntry(
            name=entry.name,
            path=_to_workspace_relative(entry.path),
            type="dir" if entry.type == FileType.DIR else "file",
        )
        for entry in raw_entries
    ]
    entries.sort(key=lambda e: (e.type != "dir", e.name.lower()))
    return SandboxTreeResponse(entries=entries)


@router.get(
    "/sessions/{session_id}/sandbox/file",
    dependencies=[Depends(sandbox_flag_dependency)],
)
async def get_sandbox_file(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
    path: str = Query(...),
) -> SandboxFileResponse:
    """Read a file's UTF-8 content (first 1 MB, ``truncated`` set if larger)."""
    abs_path = _safe_resolve(path)
    sandbox = await _get_sandbox_for_session(session_id, user_id)
    raw = bytes(await sandbox.files.read(abs_path, format="bytes"))
    truncated = len(raw) > _MAX_FILE_BYTES
    content = raw[:_MAX_FILE_BYTES].decode("utf-8", errors="replace")
    return SandboxFileResponse(path=path, content=content, truncated=truncated)


@router.put(
    "/sessions/{session_id}/sandbox/file",
    dependencies=[Depends(sandbox_flag_dependency)],
)
async def write_sandbox_file(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
    body: SandboxWriteFileRequest,
) -> SandboxFileResponse:
    """Write UTF-8 content to a file, echoing back what was written."""
    abs_path = _safe_resolve(body.path)
    sandbox = await _get_sandbox_for_session(session_id, user_id)
    await sandbox.files.write(abs_path, body.content)
    return SandboxFileResponse(path=body.path, content=body.content, truncated=False)


@router.get(
    "/sessions/{session_id}/sandbox/changes",
    dependencies=[Depends(sandbox_flag_dependency)],
)
async def get_sandbox_changes(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> SandboxChangesResponse:
    """List git-changed files via ``git status --porcelain`` (empty if not a repo)."""
    sandbox = await _get_sandbox_for_session(session_id, user_id)
    stdout, _, exit_code = await _run_command(sandbox, "git status --porcelain")
    if exit_code != 0:
        return SandboxChangesResponse(is_git_repo=False, files=[])
    return SandboxChangesResponse(is_git_repo=True, files=_parse_porcelain(stdout))


@router.get(
    "/sessions/{session_id}/sandbox/diff",
    dependencies=[Depends(sandbox_flag_dependency)],
)
async def get_sandbox_diff(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
    path: str = Query(...),
) -> SandboxDiffResponse:
    """Return committed (HEAD) vs current content for a file."""
    abs_path = _safe_resolve(path)
    rel_path = _to_workspace_relative(abs_path)
    sandbox = await _get_sandbox_for_session(session_id, user_id)

    show_stdout, _, show_exit = await _run_command(
        sandbox, f"git show {shlex.quote('HEAD:' + rel_path)}"
    )
    original = show_stdout if show_exit == 0 else ""

    try:
        modified = bytes(await sandbox.files.read(abs_path, format="bytes")).decode(
            "utf-8", errors="replace"
        )
    except Exception:
        modified = ""

    return SandboxDiffResponse(path=path, original=original, modified=modified)


@router.get(
    "/sessions/{session_id}/sandbox/download",
    dependencies=[Depends(sandbox_flag_dependency)],
)
async def download_sandbox_files(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> Response:
    """Download the workspace as a gzipped tarball (VCS/deps excluded)."""
    sandbox = await _get_sandbox_for_session(session_id, user_id)
    tmp = shlex.quote(_DOWNLOAD_TMP_PATH)
    await _run_command(
        sandbox,
        f"tar --exclude-vcs --exclude=node_modules --exclude=.venv "
        f"-czf {tmp} -C {shlex.quote(E2B_WORKDIR)} .",
        timeout=120,
    )

    size_out, _, size_exit = await _run_command(sandbox, f"stat -c %s {tmp}")
    if size_exit == 0 and size_out.strip().isdigit():
        if int(size_out.strip()) > _MAX_DOWNLOAD_BYTES:
            await _run_command(sandbox, f"rm -f {tmp}")
            raise HTTPException(
                status_code=413, detail="Workspace archive too large to download."
            )

    data = bytes(await sandbox.files.read(_DOWNLOAD_TMP_PATH, format="bytes"))
    await _run_command(sandbox, f"rm -f {tmp}")
    return Response(
        content=data,
        media_type="application/gzip",
        headers={
            "Content-Disposition": 'attachment; filename="workspace.tar.gz"',
        },
    )


@router.websocket("/sessions/{session_id}/sandbox/terminal")
async def sandbox_terminal(websocket: WebSocket, session_id: str) -> None:
    """Interactive PTY over WebSocket.

    Client→server: JSON text frames ``{"type":"input","data":...}`` and
    ``{"type":"resize","cols":N,"rows":N}``. Server→client: raw PTY bytes as
    binary frames.
    """
    await websocket.accept()

    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing authentication token")
        return
    try:
        payload = parse_jwt_token(token)
        user_id = payload.get("sub")
    except ValueError:
        await websocket.close(code=4003, reason="Invalid token")
        return
    if not user_id:
        await websocket.close(code=4002, reason="Invalid token")
        return

    if not await is_feature_enabled(Flag.AUTOGPT_NEW_LAYOUT_IDE, user_id):
        await websocket.close(code=4005, reason="Feature not available")
        return

    session = await get_chat_session_metadata(session_id, user_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    cfg = ChatConfig()
    if not cfg.e2b_active or not cfg.e2b_api_key:
        await websocket.close(code=4005, reason="Sandbox not available")
        return

    sandbox = await get_or_create_sandbox(
        session_id,
        cfg.e2b_api_key,
        cfg.e2b_sandbox_timeout,
        template=cfg.e2b_sandbox_template,
        on_timeout=cfg.e2b_sandbox_on_timeout,
    )

    async def on_pty(data: bytes) -> None:
        await websocket.send_bytes(bytes(data))

    handle = await sandbox.pty.create(
        size=PtySize(rows=24, cols=80),
        on_data=on_pty,
        cwd=E2B_WORKDIR,
        timeout=0,
    )
    pump = asyncio.create_task(handle.wait())
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            if message.get("type") == "input":
                await sandbox.pty.send_stdin(
                    handle.pid, str(message.get("data", "")).encode()
                )
            elif message.get("type") == "resize":
                await sandbox.pty.resize(
                    handle.pid,
                    PtySize(rows=int(message["rows"]), cols=int(message["cols"])),
                )
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await sandbox.pty.kill(handle.pid)
        except Exception:
            logger.debug("[E2B] pty kill failed for session %s", session_id[:12])
        pump.cancel()
