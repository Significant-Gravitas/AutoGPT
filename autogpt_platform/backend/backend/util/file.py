import base64
import mimetypes
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from backend.util.cloud_storage import get_cloud_storage_handler
from backend.util.request import Requests
from backend.util.settings import Config
from backend.util.type import MediaFileType
from backend.util.virus_scanner import scan_content_safe

if TYPE_CHECKING:
    from backend.data.execution import ExecutionContext

# Return format options for store_media_file
# - "for_local_processing": Returns local file path - use with ffmpeg, MoviePy, PIL, etc.
# - "for_external_api": Returns data URI (base64) - use when sending content to external APIs
# - "for_block_output": Returns best format for output - workspace:// in CoPilot, data URI in graphs
MediaReturnFormat = Literal[
    "for_local_processing", "for_external_api", "for_block_output"
]

TEMP_DIR = Path(tempfile.gettempdir()).resolve()

# Maximum filename length (conservative limit for most filesystems)
MAX_FILENAME_LENGTH = 200


def sanitize_filename(filename: str) -> str:
    """
    Sanitize and truncate filename to prevent filesystem errors.
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", filename)

    # Truncate if too long
    if len(sanitized) > MAX_FILENAME_LENGTH:
        # Keep the extension if possible
        if "." in sanitized:
            name, ext = sanitized.rsplit(".", 1)
            max_name_length = MAX_FILENAME_LENGTH - len(ext) - 1
            sanitized = name[:max_name_length] + "." + ext
        else:
            sanitized = sanitized[:MAX_FILENAME_LENGTH]

    # Ensure it's not empty or just dots
    if not sanitized or sanitized.strip(".") == "":
        sanitized = f"file_{uuid.uuid4().hex[:8]}"

    return sanitized


def get_exec_file_path(graph_exec_id: str, path: str) -> str:
    """
    Utility to build an absolute path in the {temp}/exec_file/{exec_id}/... folder.
    """
    try:
        full_path = TEMP_DIR / "exec_file" / graph_exec_id / path
        return str(full_path)
    except OSError as e:
        if "File name too long" in str(e):
            raise ValueError(
                f"File path too long: {len(path)} characters. Maximum path length exceeded."
            ) from e
        raise ValueError(f"Invalid file path: {e}") from e


def clean_exec_files(graph_exec_id: str, file: str = "") -> None:
    """
    Utility to remove the {temp}/exec_file/{exec_id} folder and its contents.
    """
    exec_path = Path(get_exec_file_path(graph_exec_id, file))
    if exec_path.exists() and exec_path.is_dir():
        shutil.rmtree(exec_path)


async def store_media_file(
    file: MediaFileType,
    execution_context: "ExecutionContext",
    *,
    return_format: MediaReturnFormat,
) -> MediaFileType:
    """
    Safely handle 'file' (a data URI, a URL, a workspace:// reference, or a local path
    relative to {temp}/exec_file/{exec_id}), placing or verifying it under:
        {tempdir}/exec_file/{exec_id}/...

    For each MediaFileType input:
    - Data URI: decode and store locally
    - URL: download and store locally
    - workspace:// reference: read from workspace, store locally
    - Local path: verify it exists in exec_file directory

    Return format options:
    - "for_local_processing": Returns local file path - use with ffmpeg, MoviePy, PIL, etc.
    - "for_external_api": Returns data URI (base64) - use when sending to external APIs
    - "for_block_output": Returns best format for output - workspace:// in CoPilot, data URI in graphs

    :param file:               Data URI, URL, workspace://, or local (relative) path.
    :param execution_context:  ExecutionContext with user_id, graph_exec_id, workspace_id.
    :param return_format:      What to return: "for_local_processing", "for_external_api", or "for_block_output".
    :return:                   The requested result based on return_format.
    """
    # Extract values from execution_context
    graph_exec_id = execution_context.graph_exec_id
    user_id = execution_context.user_id

    if not graph_exec_id:
        raise ValueError("execution_context.graph_exec_id is required")
    if not user_id:
        raise ValueError("execution_context.user_id is required")

    # Create workspace_manager if we have workspace_id (with session scoping)
    # Import here to avoid circular import (file.py → workspace.py → data → blocks → file.py)
    from backend.util.workspace import WorkspaceManager

    workspace_manager: WorkspaceManager | None = None
    if execution_context.workspace_id:
        workspace_manager = WorkspaceManager(
            user_id, execution_context.workspace_id, execution_context.session_id
        )
    # Build base path
    base_path = Path(get_exec_file_path(graph_exec_id, ""))
    base_path.mkdir(parents=True, exist_ok=True)

    # Security fix: Add disk space limits to prevent DoS
    MAX_FILE_SIZE_BYTES = Config().max_file_size_mb * 1024 * 1024
    MAX_TOTAL_DISK_USAGE = 1024 * 1024 * 1024  # 1GB total per execution directory

    # Check total disk usage in base_path
    if base_path.exists():
        current_usage = get_dir_size(base_path)
        if current_usage > MAX_TOTAL_DISK_USAGE:
            raise ValueError(
                f"Disk usage limit exceeded: {current_usage} bytes > {MAX_TOTAL_DISK_USAGE} bytes"
            )

    # Helper functions
    def _extension_from_mime(mime: str) -> str:
        ext = mimetypes.guess_extension(mime, strict=False)
        return ext if ext else ".bin"

    def _file_to_data_uri(path: Path) -> str:
        mime_type = get_mime_type(str(path))
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def _ensure_inside_base(path_candidate: Path, base: Path) -> Path:
        """
        Resolve symlinks via resolve() and ensure the result is still under base.
        """
        real_candidate = path_candidate.resolve()
        real_base = base.resolve()

        if not real_candidate.is_relative_to(real_base):
            raise ValueError(
                "Local file path is outside the temp_base directory. Access denied."
            )
        return real_candidate

    def _strip_base_prefix(absolute_path: Path, base: Path) -> str:
        """
        Strip base prefix and normalize path.
        """
        return str(absolute_path.relative_to(base))

    # Get cloud storage handler for checking cloud paths
    cloud_storage = await get_cloud_storage_handler()

    # Track if the input came from workspace (don't re-save it)
    is_from_workspace = file.startswith("workspace://")

    # Check if this is a workspace file reference
    if is_from_workspace:
        if workspace_manager is None:
            raise ValueError(
                "Workspace file reference requires workspace context. "
                "This file type is only available in CoPilot sessions."
            )

        # Parse workspace reference
        # workspace://abc123 - by file ID
        # workspace:///path/to/file.txt - by virtual path
        file_ref = file[12:]  # Remove "workspace://"

        if file_ref.startswith("/"):
            # Path reference
            workspace_content = await workspace_manager.read_file(file_ref)
            file_info = await workspace_manager.get_file_info_by_path(file_ref)
            filename = sanitize_filename(
                file_info.name if file_info else f"{uuid.uuid4()}.bin"
            )
        else:
            # ID reference
            workspace_content = await workspace_manager.read_file_by_id(file_ref)
            file_info = await workspace_manager.get_file_info(file_ref)
            filename = sanitize_filename(
                file_info.name if file_info else f"{uuid.uuid4()}.bin"
            )

        try:
            target_path = _ensure_inside_base(base_path / filename, base_path)
        except OSError as e:
            raise ValueError(f"Invalid file path '{filename}': {e}") from e

        # Check file size limit
        if len(workspace_content) > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {len(workspace_content)} bytes > {MAX_FILE_SIZE_BYTES} bytes"
            )

        # Virus scan the workspace content before writing locally
        await scan_content_safe(workspace_content, filename=filename)
        target_path.write_bytes(workspace_content)

    # Check if this is a cloud storage path
    elif cloud_storage.is_cloud_path(file):
        # Download from cloud storage and store locally
        cloud_content = await cloud_storage.retrieve_file(
            file, user_id=user_id, graph_exec_id=graph_exec_id
        )

        # Generate filename from cloud path
        _, path_part = cloud_storage.parse_cloud_path(file)
        filename = sanitize_filename(Path(path_part).name or f"{uuid.uuid4()}.bin")
        try:
            target_path = _ensure_inside_base(base_path / filename, base_path)
        except OSError as e:
            raise ValueError(f"Invalid file path '{filename}': {e}") from e

        # Check file size limit
        if len(cloud_content) > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {len(cloud_content)} bytes > {MAX_FILE_SIZE_BYTES} bytes"
            )

        # Virus scan the cloud content before writing locally
        await scan_content_safe(cloud_content, filename=filename)
        target_path.write_bytes(cloud_content)

    # Process file
    elif file.startswith("data:"):
        # Data URI
        match = re.match(r"^data:([^;]+);base64,(.*)$", file, re.DOTALL)
        if not match:
            raise ValueError(
                "Invalid data URI format. Expected data:<mime>;base64,<data>"
            )
        mime_type = match.group(1).strip().lower()
        b64_content = match.group(2).strip()

        # Generate filename and decode
        extension = _extension_from_mime(mime_type)
        filename = f"{uuid.uuid4()}{extension}"
        try:
            target_path = _ensure_inside_base(base_path / filename, base_path)
        except OSError as e:
            raise ValueError(f"Invalid file path '{filename}': {e}") from e
        content = base64.b64decode(b64_content)

        # Check file size limit
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {len(content)} bytes > {MAX_FILE_SIZE_BYTES} bytes"
            )

        # Virus scan the base64 content before writing
        await scan_content_safe(content, filename=filename)
        target_path.write_bytes(content)

    elif file.startswith(("http://", "https://")):
        # URL - download first to get Content-Type header
        resp = await Requests().get(file)

        # Check file size limit
        if len(resp.content) > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {len(resp.content)} bytes > {MAX_FILE_SIZE_BYTES} bytes"
            )

        # Extract filename from URL path
        parsed_url = urlparse(file)
        filename = sanitize_filename(Path(parsed_url.path).name or f"{uuid.uuid4()}")

        # If filename lacks extension, add one from Content-Type header
        if "." not in filename:
            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
            if content_type:
                ext = _extension_from_mime(content_type)
                filename = f"{filename}{ext}"

        try:
            target_path = _ensure_inside_base(base_path / filename, base_path)
        except OSError as e:
            raise ValueError(f"Invalid file path '{filename}': {e}") from e

        # Virus scan the downloaded content before writing
        await scan_content_safe(resp.content, filename=filename)
        target_path.write_bytes(resp.content)

    else:
        # Local path - sanitize the filename part to prevent long filename errors
        sanitized_file = sanitize_filename(file)
        try:
            target_path = _ensure_inside_base(base_path / sanitized_file, base_path)
        except OSError as e:
            raise ValueError(f"Invalid file path '{sanitized_file}': {e}") from e
        if not target_path.is_file():
            raise ValueError(f"Local file does not exist: {target_path}")

    # Return based on requested format
    if return_format == "for_local_processing":
        # Use when processing files locally with tools like ffmpeg, MoviePy, PIL
        # Returns: relative path in exec_file directory (e.g., "image.png")
        return MediaFileType(_strip_base_prefix(target_path, base_path))

    elif return_format == "for_external_api":
        # Use when sending content to external APIs that need base64
        # Returns: data URI (e.g., "data:image/png;base64,iVBORw0...")
        return MediaFileType(_file_to_data_uri(target_path))

    elif return_format == "for_block_output":
        # Use when returning output from a block to user/next block
        # Returns: workspace:// ref (CoPilot) or data URI (graph execution)
        if workspace_manager is None:
            # No workspace available (graph execution without CoPilot)
            # Fallback to data URI so the content can still be used/displayed
            return MediaFileType(_file_to_data_uri(target_path))

        # Don't re-save if input was already from workspace
        if is_from_workspace:
            # Return original workspace reference
            return MediaFileType(file)

        # Save new content to workspace
        content = target_path.read_bytes()
        filename = target_path.name

        file_record = await workspace_manager.write_file(
            content=content,
            filename=filename,
            overwrite=True,
        )
        return MediaFileType(f"workspace://{file_record.id}")

    else:
        raise ValueError(f"Invalid return_format: {return_format}")


def get_dir_size(path: Path) -> int:
    """Get total size of directory."""
    total = 0
    try:
        for entry in path.glob("**/*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return total


def get_mime_type(file: str) -> str:
    """
    Get the MIME type of a file, whether it's a data URI, URL, or local path.
    """
    if file.startswith("data:"):
        match = re.match(r"^data:([^;]+);base64,", file)
        return match.group(1) if match else "application/octet-stream"

    elif file.startswith(("http://", "https://")):
        parsed_url = urlparse(file)
        mime_type, _ = mimetypes.guess_type(parsed_url.path)
        return mime_type or "application/octet-stream"

    else:
        mime_type, _ = mimetypes.guess_type(file)
        return mime_type or "application/octet-stream"
