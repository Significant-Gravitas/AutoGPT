"""E2B file tools — MCP tools that proxy filesystem operations to the e2b sandbox.

These replace the SDK built-in Read/Write/Edit/Glob/Grep tools when e2b is
enabled, ensuring all file operations go through the sandbox VM.
"""

import logging
import posixpath
import shlex
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import BashExecResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_SANDBOX_HOME = "/home/user"


class E2BReadTool(BaseTool):
    """Read a file from the e2b sandbox filesystem."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a file from the sandbox filesystem. "
            "The sandbox is the shared working environment — files created by "
            "any tool (bash_exec, write_file, etc.) are accessible here. "
            "Returns the file content as text. "
            "Use offset and limit for large files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to the file to read (relative to /home/user/ "
                        "or absolute within /home/user/)."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "Line number to start reading from (0-indexed). Default: 0"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read. Default: 2000",
                },
            },
            "required": ["path"],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        path = kwargs.get("path", "")
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 2000)

        sandbox = await _get_sandbox(session)
        if sandbox is None:
            return _sandbox_unavailable(session)

        resolved = _resolve_path(path)
        if resolved is None:
            return _path_error(path, session)

        try:
            content = await sandbox.files.read(resolved)
            lines = content.splitlines(keepends=True)
            selected = lines[offset : offset + limit]
            text = "".join(selected)
            return BashExecResponse(
                message=f"Read {len(selected)} lines from {resolved}",
                stdout=text,
                stderr="",
                exit_code=0,
                timed_out=False,
                session_id=session.session_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to read {resolved}: {e}",
                error=str(e),
                session_id=session.session_id,
            )


class E2BWriteTool(BaseTool):
    """Write a file to the e2b sandbox filesystem."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write or create a file in the sandbox filesystem. "
            "This is the shared working environment — files are accessible "
            "to bash_exec and other tools. "
            "Creates parent directories automatically."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path for the file (relative to /home/user/ "
                        "or absolute within /home/user/)."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["path", "content"],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")

        sandbox = await _get_sandbox(session)
        if sandbox is None:
            return _sandbox_unavailable(session)

        resolved = _resolve_path(path)
        if resolved is None:
            return _path_error(path, session)

        try:
            # Ensure parent directory exists
            parent = posixpath.dirname(resolved)
            if parent and parent != _SANDBOX_HOME:
                await sandbox.commands.run(f"mkdir -p {parent}", timeout=5)
            await sandbox.files.write(resolved, content)
            return BashExecResponse(
                message=f"Wrote {len(content)} bytes to {resolved}",
                stdout=f"Successfully wrote to {resolved}",
                stderr="",
                exit_code=0,
                timed_out=False,
                session_id=session.session_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to write {resolved}: {e}",
                error=str(e),
                session_id=session.session_id,
            )


class E2BEditTool(BaseTool):
    """Edit a file in the e2b sandbox using search/replace."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file in the sandbox by replacing exact text. "
            "Provide old_text (the exact text to find) and new_text "
            "(what to replace it with). The old_text must match exactly."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to the file (relative to /home/user/ "
                        "or absolute within /home/user/)."
                    ),
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find in the file.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace old_text with.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        path = kwargs.get("path", "")
        old_text = kwargs.get("old_text", "")
        new_text = kwargs.get("new_text", "")

        sandbox = await _get_sandbox(session)
        if sandbox is None:
            return _sandbox_unavailable(session)

        resolved = _resolve_path(path)
        if resolved is None:
            return _path_error(path, session)

        try:
            content = await sandbox.files.read(resolved)
            occurrences = content.count(old_text)
            if occurrences == 0:
                return ErrorResponse(
                    message=f"old_text not found in {resolved}",
                    error="text_not_found",
                    session_id=session.session_id,
                )
            if occurrences > 1:
                return ErrorResponse(
                    message=(
                        f"old_text found {occurrences} times in {resolved}. "
                        "Please provide more context to make the match unique."
                    ),
                    error="ambiguous_match",
                    session_id=session.session_id,
                )
            new_content = content.replace(old_text, new_text, 1)
            await sandbox.files.write(resolved, new_content)
            return BashExecResponse(
                message=f"Edited {resolved}",
                stdout=f"Successfully edited {resolved}",
                stderr="",
                exit_code=0,
                timed_out=False,
                session_id=session.session_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to edit {resolved}: {e}",
                error=str(e),
                session_id=session.session_id,
            )


class E2BGlobTool(BaseTool):
    """List files matching a pattern in the e2b sandbox."""

    @property
    def name(self) -> str:
        return "glob_files"

    @property
    def description(self) -> str:
        return (
            "List files in the sandbox matching a glob pattern. "
            "Uses find under the hood. Default directory is /home/user/."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern to match (e.g., '*.py', '**/*.json')."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": ("Directory to search in (default: /home/user/)."),
                },
            },
            "required": ["pattern"],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        pattern = kwargs.get("pattern", "*")
        path = kwargs.get("path", _SANDBOX_HOME)

        sandbox = await _get_sandbox(session)
        if sandbox is None:
            return _sandbox_unavailable(session)

        resolved = _resolve_path(path)
        if resolved is None:
            return _path_error(path, session)

        try:
            result = await sandbox.commands.run(
                f"find {resolved} -name {shlex.quote(pattern)} -type f 2>/dev/null",
                timeout=15,
            )
            return BashExecResponse(
                message="Glob results",
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                timed_out=False,
                session_id=session.session_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to glob: {e}",
                error=str(e),
                session_id=session.session_id,
            )


class E2BGrepTool(BaseTool):
    """Search file contents in the e2b sandbox."""

    @property
    def name(self) -> str:
        return "grep_files"

    @property
    def description(self) -> str:
        return (
            "Search for a pattern in files within the sandbox. "
            "Uses grep -rn under the hood. Returns matching lines with "
            "file paths and line numbers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": ("Directory to search in (default: /home/user/)."),
                },
                "include": {
                    "type": "string",
                    "description": "File glob to include (e.g., '*.py').",
                },
            },
            "required": ["pattern"],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        pattern = kwargs.get("pattern", "")
        path = kwargs.get("path", _SANDBOX_HOME)
        include = kwargs.get("include", "")

        sandbox = await _get_sandbox(session)
        if sandbox is None:
            return _sandbox_unavailable(session)

        resolved = _resolve_path(path)
        if resolved is None:
            return _path_error(path, session)

        include_flag = f" --include={shlex.quote(include)}" if include else ""
        try:
            result = await sandbox.commands.run(
                f"grep -rn{include_flag} {shlex.quote(pattern)} {resolved} 2>/dev/null",
                timeout=15,
            )
            return BashExecResponse(
                message="Grep results",
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                timed_out=False,
                session_id=session.session_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to grep: {e}",
                error=str(e),
                session_id=session.session_id,
            )


class SaveToWorkspaceTool(BaseTool):
    """Copy a file from e2b sandbox to the persistent GCS workspace."""

    @property
    def name(self) -> str:
        return "save_to_workspace"

    @property
    def description(self) -> str:
        return (
            "Save a file from the sandbox to the persistent workspace "
            "(cloud storage). Files saved to workspace survive across sessions. "
            "Provide the sandbox file path and optional workspace path."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sandbox_path": {
                    "type": "string",
                    "description": "Path of the file in the sandbox to save.",
                },
                "workspace_path": {
                    "type": "string",
                    "description": (
                        "Path in the workspace to save to "
                        "(defaults to the sandbox filename)."
                    ),
                },
            },
            "required": ["sandbox_path"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        sandbox_path = kwargs.get("sandbox_path", "")
        workspace_path = kwargs.get("workspace_path", "")

        if not user_id:
            return ErrorResponse(
                message="Authentication required",
                session_id=session.session_id,
            )

        sandbox = await _get_sandbox(session)
        if sandbox is None:
            return _sandbox_unavailable(session)

        resolved = _resolve_path(sandbox_path)
        if resolved is None:
            return _path_error(sandbox_path, session)

        try:
            content_bytes = await sandbox.files.read(resolved, format="bytes")

            # Determine workspace path
            filename = resolved.rsplit("/", 1)[-1]
            wp = workspace_path or f"/{filename}"

            from backend.data.db_accessors import workspace_db
            from backend.util.workspace import WorkspaceManager

            workspace = await workspace_db().get_or_create_workspace(user_id)
            manager = WorkspaceManager(user_id, workspace.id, session.session_id)
            file_record = await manager.write_file(
                content=content_bytes,
                filename=filename,
                path=wp,
                overwrite=True,
            )

            return BashExecResponse(
                message=f"Saved {resolved} to workspace at {file_record.path}",
                stdout=(
                    f"Saved to workspace: {file_record.path} "
                    f"({file_record.size_bytes} bytes)"
                ),
                stderr="",
                exit_code=0,
                timed_out=False,
                session_id=session.session_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to save to workspace: {e}",
                error=str(e),
                session_id=session.session_id,
            )


class LoadFromWorkspaceTool(BaseTool):
    """Copy a file from the persistent GCS workspace into the e2b sandbox."""

    @property
    def name(self) -> str:
        return "load_from_workspace"

    @property
    def description(self) -> str:
        return (
            "Load a file from the persistent workspace (cloud storage) into "
            "the sandbox. Use this to bring workspace files into the sandbox "
            "for editing or processing."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workspace_path": {
                    "type": "string",
                    "description": ("Path of the file in the workspace to load."),
                },
                "sandbox_path": {
                    "type": "string",
                    "description": (
                        "Path in the sandbox to write to "
                        "(defaults to /home/user/<filename>)."
                    ),
                },
            },
            "required": ["workspace_path"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        workspace_path = kwargs.get("workspace_path", "")
        sandbox_path = kwargs.get("sandbox_path", "")

        if not user_id:
            return ErrorResponse(
                message="Authentication required",
                session_id=session.session_id,
            )

        sandbox = await _get_sandbox(session)
        if sandbox is None:
            return _sandbox_unavailable(session)

        try:
            from backend.data.db_accessors import workspace_db
            from backend.util.workspace import WorkspaceManager

            workspace = await workspace_db().get_or_create_workspace(user_id)
            manager = WorkspaceManager(user_id, workspace.id, session.session_id)
            file_info = await manager.get_file_info_by_path(workspace_path)
            if file_info is None:
                return ErrorResponse(
                    message=f"File not found in workspace: {workspace_path}",
                    session_id=session.session_id,
                )
            content = await manager.read_file_by_id(file_info.id)

            # Determine sandbox path
            filename = workspace_path.rsplit("/", 1)[-1]
            target = sandbox_path or f"{_SANDBOX_HOME}/{filename}"
            resolved = _resolve_path(target)
            if resolved is None:
                return _path_error(target, session)

            # Ensure parent directory exists
            parent = posixpath.dirname(resolved)
            if parent and parent != _SANDBOX_HOME:
                await sandbox.commands.run(f"mkdir -p {parent}", timeout=5)
            await sandbox.files.write(resolved, content)

            return BashExecResponse(
                message=f"Loaded {workspace_path} into sandbox at {resolved}",
                stdout=(f"Loaded from workspace: {resolved} ({len(content)} bytes)"),
                stderr="",
                exit_code=0,
                timed_out=False,
                session_id=session.session_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to load from workspace: {e}",
                error=str(e),
                session_id=session.session_id,
            )


# ------------------------------------------------------------------
# Module-level helpers (placed after functions that call them)
# ------------------------------------------------------------------


def _resolve_path(path: str) -> str | None:
    """Resolve a path to an absolute path within /home/user/.

    Returns None if the path escapes the sandbox home.
    """
    if not path:
        return None

    # Handle relative paths
    if not path.startswith("/"):
        path = f"{_SANDBOX_HOME}/{path}"

    # Normalize to prevent traversal
    resolved = posixpath.normpath(path)

    if not resolved.startswith(_SANDBOX_HOME):
        return None

    return resolved


async def _get_sandbox(session: ChatSession) -> Any | None:
    """Get the sandbox for the current session from the execution context."""
    try:
        from backend.copilot.sdk.tool_adapter import get_sandbox_manager

        manager = get_sandbox_manager()
        if manager is None:
            return None
        user_id, _ = _get_user_from_context()
        return await manager.get_or_create(session.session_id, user_id or "anonymous")
    except Exception as e:
        logger.error(f"[E2B] Failed to get sandbox: {e}")
        return None


def _get_user_from_context() -> tuple[str | None, Any]:
    """Get user_id from execution context."""
    from backend.copilot.sdk.tool_adapter import get_execution_context

    return get_execution_context()


def _sandbox_unavailable(session: ChatSession) -> ErrorResponse:
    """Return an error response for unavailable sandbox."""
    return ErrorResponse(
        message="E2B sandbox is not available. Try again or contact support.",
        error="sandbox_unavailable",
        session_id=session.session_id,
    )


def _path_error(path: str, session: ChatSession) -> ErrorResponse:
    """Return an error response for invalid paths."""
    return ErrorResponse(
        message=f"Invalid path: {path}. Paths must be within /home/user/.",
        error="invalid_path",
        session_id=session.session_id,
    )
