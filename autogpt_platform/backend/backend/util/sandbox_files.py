"""
Shared utilities for extracting and storing files from E2B sandboxes.

This module provides common file extraction and workspace storage functionality
for blocks that run code in E2B sandboxes (Claude Code, Code Executor, etc.).
"""

import base64
import logging
import mimetypes
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from backend.util.file import store_media_file
from backend.util.type import MediaFileType

if TYPE_CHECKING:
    from e2b import AsyncSandbox as BaseAsyncSandbox

    from backend.executor.utils import ExecutionContext

logger = logging.getLogger(__name__)

# Text file extensions that can be safely read and stored as text
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".css",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".py",
    ".rb",
    ".php",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".graphql",
    ".env",
    ".gitignore",
    ".dockerfile",
    "Dockerfile",
    ".vue",
    ".svelte",
    ".astro",
    ".mdx",
    ".rst",
    ".tex",
    ".csv",
    ".log",
    ".svg",  # SVG is XML-based text
}

# Binary file extensions we explicitly support extracting
BINARY_EXTENSIONS = {
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bmp",
    ".tiff",
    ".tif",
    # Documents
    ".pdf",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    # Audio
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    # Video
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    # Fonts
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
}

# Maximum file size for binary extraction (50MB)
# Prevents OOM from accidentally extracting huge files
MAX_BINARY_FILE_SIZE = 50 * 1024 * 1024


class SandboxFileOutput(BaseModel):
    """A file extracted from a sandbox and optionally stored in workspace."""

    path: str
    """Full path in the sandbox."""

    relative_path: str
    """Path relative to the working directory."""

    name: str
    """Filename only."""

    content: str
    """File content as text (for backward compatibility)."""

    workspace_ref: str | None = None
    """Workspace reference (workspace://{id}#mime) if stored, None otherwise."""


@dataclass
class ExtractedFile:
    """Internal representation of an extracted file before storage."""

    path: str
    relative_path: str
    name: str
    content: bytes
    is_text: bool


async def extract_sandbox_files(
    sandbox: "BaseAsyncSandbox",
    working_directory: str,
    since_timestamp: str | None = None,
    text_only: bool = True,
) -> list[ExtractedFile]:
    """
    Extract files from an E2B sandbox.

    Args:
        sandbox: The E2B sandbox instance
        working_directory: Directory to search for files
        since_timestamp: ISO timestamp - only return files modified after this time
        text_only: If True, only extract text files. If False, also extract
                   supported binary files (images, PDFs, etc.).

    Returns:
        List of ExtractedFile objects with path, content, and metadata
    """
    files: list[ExtractedFile] = []

    try:
        # Build find command
        safe_working_dir = shlex.quote(working_directory)
        timestamp_filter = ""
        if since_timestamp:
            timestamp_filter = f"-newermt {shlex.quote(since_timestamp)} "

        find_result = await sandbox.commands.run(
            f"find {safe_working_dir} -type f "
            f"{timestamp_filter}"
            f"-not -path '*/node_modules/*' "
            f"-not -path '*/.git/*' "
            f"2>/dev/null"
        )

        if not find_result.stdout:
            return files

        for file_path in find_result.stdout.strip().split("\n"):
            if not file_path:
                continue

            # Check file type (case-insensitive for extensions)
            file_path_lower = file_path.lower()
            is_text = any(
                file_path_lower.endswith(ext.lower()) for ext in TEXT_EXTENSIONS
            )
            is_binary = any(
                file_path_lower.endswith(ext.lower()) for ext in BINARY_EXTENSIONS
            )

            # Skip files with unrecognized extensions
            if not is_text and not is_binary:
                continue

            # In text_only mode, skip binary files
            if text_only and not is_text:
                continue

            try:
                # For binary files, check size before reading to prevent OOM
                if is_binary:
                    stat_result = await sandbox.commands.run(
                        f"stat -c %s {shlex.quote(file_path)} 2>/dev/null"
                    )
                    if stat_result.exit_code != 0 or not stat_result.stdout:
                        logger.debug(
                            f"Skipping {file_path}: could not determine file size"
                        )
                        continue

                    try:
                        file_size = int(stat_result.stdout.strip())
                    except ValueError:
                        logger.debug(
                            f"Skipping {file_path}: unexpected stat output "
                            f"{stat_result.stdout.strip()!r}"
                        )
                        continue

                    if file_size > MAX_BINARY_FILE_SIZE:
                        logger.info(
                            f"Skipping {file_path}: size {file_size} bytes "
                            f"exceeds limit {MAX_BINARY_FILE_SIZE}"
                        )
                        continue

                content = await sandbox.files.read(file_path, format="bytes")
                if isinstance(content, str):
                    content = content.encode("utf-8")
                elif isinstance(content, bytearray):
                    content = bytes(content)

                # Extract filename from path
                file_name = file_path.split("/")[-1]

                # Calculate relative path
                relative_path = file_path
                if file_path.startswith(working_directory):
                    relative_path = file_path[len(working_directory) :]
                    if relative_path.startswith("/"):
                        relative_path = relative_path[1:]

                files.append(
                    ExtractedFile(
                        path=file_path,
                        relative_path=relative_path,
                        name=file_name,
                        content=content,
                        is_text=is_text,
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to read file {file_path}: {e}")
                continue

    except Exception as e:
        logger.warning(f"File extraction failed: {e}")

    return files


async def store_sandbox_files(
    extracted_files: list[ExtractedFile],
    execution_context: "ExecutionContext",
) -> list[SandboxFileOutput]:
    """
    Store extracted sandbox files to workspace and return output objects.

    Args:
        extracted_files: List of files extracted from sandbox
        execution_context: Execution context for workspace storage

    Returns:
        List of SandboxFileOutput objects with workspace refs
    """
    outputs: list[SandboxFileOutput] = []

    for file in extracted_files:
        # Decode content for text files (for backward compat content field)
        if file.is_text:
            try:
                content_str = file.content.decode("utf-8", errors="replace")
            except Exception:
                content_str = ""
        else:
            content_str = f"[Binary file: {len(file.content)} bytes]"

        # Build data URI (needed for storage and as binary fallback)
        mime_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        data_uri = f"data:{mime_type};base64,{base64.b64encode(file.content).decode()}"

        # Try to store in workspace
        workspace_ref: str | None = None
        try:
            result = await store_media_file(
                file=MediaFileType(data_uri),
                execution_context=execution_context,
                return_format="for_block_output",
            )
            if result.startswith("workspace://"):
                workspace_ref = result
            elif not file.is_text:
                # Non-workspace context (graph execution): store_media_file
                # returned a data URI — use it as content so binary data isn't lost.
                content_str = result
        except Exception as e:
            logger.warning(f"Failed to store file {file.name} to workspace: {e}")
            # For binary files, fall back to data URI to prevent data loss
            if not file.is_text:
                content_str = data_uri

        outputs.append(
            SandboxFileOutput(
                path=file.path,
                relative_path=file.relative_path,
                name=file.name,
                content=content_str,
                workspace_ref=workspace_ref,
            )
        )

    return outputs


async def extract_and_store_sandbox_files(
    sandbox: "BaseAsyncSandbox",
    working_directory: str,
    execution_context: "ExecutionContext",
    since_timestamp: str | None = None,
    text_only: bool = True,
) -> list[SandboxFileOutput]:
    """
    Extract files from sandbox and store them in workspace.

    This is the main entry point combining extraction and storage.

    Args:
        sandbox: The E2B sandbox instance
        working_directory: Directory to search for files
        execution_context: Execution context for workspace storage
        since_timestamp: ISO timestamp - only return files modified after this time
        text_only: If True, only extract text files

    Returns:
        List of SandboxFileOutput objects with content and workspace refs
    """
    extracted = await extract_sandbox_files(
        sandbox=sandbox,
        working_directory=working_directory,
        since_timestamp=since_timestamp,
        text_only=text_only,
    )

    return await store_sandbox_files(extracted, execution_context)
