import base64
import mimetypes
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

# This "requests" presumably has additional checks against internal networks for SSRF.
from backend.util.request import requests

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


"""
MediaFile is a string that represents a file. It can be one of the following:
    - Data URI: base64 encoded media file. See https://developer.mozilla.org/en-US/docs/Web/URI/Schemes/data/
    - URL: Media file hosted on the internet, it starts with http:// or https://.
    - Local path (anything else): A temporary file path living within graph execution time.
    
Note: Replace this type alias into a proper class, when more information is needed.
"""
MediaFile = str


def store_media_file(
    graph_exec_id: str, file: MediaFile, return_content: bool = False
) -> MediaFile:
    """
    Safely handle 'file' (a data URI, a URL, or a local path relative to {temp}/exec_file/{exec_id}),
    placing or verifying it under:
        {tempdir}/exec_file/{exec_id}/...

    If 'return_content=True', return a data URI (data:<mime>;base64,<content>).
    Otherwise, returns the file media path relative to the exec_id folder.

    For each MediaFile type:
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

    # Helper functions
    def _extension_from_mime(mime: str) -> str:
        ext = mimetypes.guess_extension(mime, strict=False)
        return ext if ext else ".bin"

    def _file_to_data_uri(path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        mime_type = mime_type or "application/octet-stream"
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

    # Process file
    if file.startswith("data:"):
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
        target_path.write_bytes(base64.b64decode(b64_content))

    elif file.startswith(("http://", "https://")):
        # URL
        parsed_url = urlparse(file)
        filename = Path(parsed_url.path).name or f"{uuid.uuid4()}"
        target_path = _ensure_inside_base(base_path / filename, base_path)

        # Download and save
        resp = requests.get(file)
        resp.raise_for_status()
        target_path.write_bytes(resp.content)

    else:
        # Local path
        target_path = _ensure_inside_base(base_path / file, base_path)
        if not target_path.is_file():
            raise ValueError(f"Local file does not exist: {target_path}")

    # Return result
    if return_content:
        return MediaFile(_file_to_data_uri(target_path))
    else:
        return MediaFile(_strip_base_prefix(target_path, base_path))
