import base64
import mimetypes
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

from backend.util.cloud_storage import get_cloud_storage_handler
from backend.util.request import Requests
from backend.util.type import MediaFileType
from backend.util.virus_scanner import scan_content_safe

TEMP_DIR = Path(tempfile.gettempdir()).resolve()


def get_exec_file_path(graph_exec_id: str, path: str) -> str:
    """
    Utility to build an absolute path in the {temp}/exec_file/{exec_id}/... folder.
    """
    return str(TEMP_DIR / "exec_file" / graph_exec_id / path)


def clean_exec_files(graph_exec_id: str, file: str = "") -> None:
    """
    Utility to remove the {temp}/exec_file/{exec_id} folder and its contents.
    """
    exec_path = Path(get_exec_file_path(graph_exec_id, file))
    if exec_path.exists() and exec_path.is_dir():
        shutil.rmtree(exec_path)


async def store_media_file(
    graph_exec_id: str,
    file: MediaFileType,
    user_id: str,
    return_content: bool = False,
) -> MediaFileType:
    """
    Safely handle 'file' (a data URI, a URL, or a local path relative to {temp}/exec_file/{exec_id}),
    placing or verifying it under:
        {tempdir}/exec_file/{exec_id}/...

    If 'return_content=True', return a data URI (data:<mime>;base64,<content>).
    Otherwise, returns the file media path relative to the exec_id folder.

    For each MediaFileType type:
    - Data URI:
      -> decode and store in a new random file in that folder
    - URL:
      -> download and store in that folder
    - Local path:
      -> interpret as relative to that folder; verify it exists
         (no copying, as it's presumably already there).
         We realpath-check so no symlink or '..' can escape the folder.


    :param graph_exec_id:  The unique ID of the graph execution.
    :param file:           Data URI, URL, or local (relative) path.
    :param return_content: If True, return a data URI of the file content.
                           If False, return the *relative* path inside the exec_id folder.
    :return:               The requested result: data URI or relative path of the media.
    """
    # Build base path
    base_path = Path(get_exec_file_path(graph_exec_id, ""))
    base_path.mkdir(parents=True, exist_ok=True)

    # Security fix: Add disk space limits to prevent DoS
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB per file
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

    # Check if this is a cloud storage path
    cloud_storage = await get_cloud_storage_handler()
    if cloud_storage.is_cloud_path(file):
        # Download from cloud storage and store locally
        cloud_content = await cloud_storage.retrieve_file(
            file, user_id=user_id, graph_exec_id=graph_exec_id
        )

        # Generate filename from cloud path
        _, path_part = cloud_storage.parse_cloud_path(file)
        filename = Path(path_part).name or f"{uuid.uuid4()}.bin"
        target_path = _ensure_inside_base(base_path / filename, base_path)

        # Check file size limit
        if len(cloud_content) > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {len(cloud_content)} bytes > {MAX_FILE_SIZE} bytes"
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
        target_path = _ensure_inside_base(base_path / filename, base_path)
        content = base64.b64decode(b64_content)

        # Check file size limit
        if len(content) > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {len(content)} bytes > {MAX_FILE_SIZE} bytes"
            )

        # Virus scan the base64 content before writing
        await scan_content_safe(content, filename=filename)
        target_path.write_bytes(content)

    elif file.startswith(("http://", "https://")):
        # URL
        parsed_url = urlparse(file)
        filename = Path(parsed_url.path).name or f"{uuid.uuid4()}"
        target_path = _ensure_inside_base(base_path / filename, base_path)

        # Download and save
        resp = await Requests().get(file)

        # Check file size limit
        if len(resp.content) > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {len(resp.content)} bytes > {MAX_FILE_SIZE} bytes"
            )

        # Virus scan the downloaded content before writing
        await scan_content_safe(resp.content, filename=filename)
        target_path.write_bytes(resp.content)

    else:
        # Local path
        target_path = _ensure_inside_base(base_path / file, base_path)
        if not target_path.is_file():
            raise ValueError(f"Local file does not exist: {target_path}")

    # Return result
    if return_content:
        return MediaFileType(_file_to_data_uri(target_path))
    else:
        return MediaFileType(_strip_base_prefix(target_path, base_path))


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
