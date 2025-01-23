import base64
import mimetypes
import os
import re
import tempfile
import uuid
from urllib.parse import urlparse

# This "requests" presumably has additional checks against internal networks for SSRF.
from backend.util.request import requests

TEMP_DIR = tempfile.gettempdir()


def get_path(exec_id: str, path: str) -> str:
    """
    Utility to build an absolute path in the {temp}/exec_file/{exec_id}/... folder.
    """
    rel_path = os.path.join(TEMP_DIR, "exec_file", exec_id, path)
    return os.path.realpath(rel_path)


def store_temp_file(exec_id: str, file: str, return_content: bool = False) -> str:
    """
    Safely handle 'file' (a data URI, a URL, or a local path relative to {temp}/exec_file/{exec_id}),
    placing or verifying it under:
        {tempdir}/exec_file/{exec_id}/...

    If 'return_content=True', return a data URI (data:<mime>;base64,<content>).
    Otherwise, return the *relative path* (prefix stripped) inside that folder.

    For each 'file' type:
      - Data URI (starting with "data:"):
          -> decode and store in a new random file in that folder
      - URL (http:// or https://):
          -> download and store in that folder
      - Local path (anything else):
          -> interpret as relative to that folder; verify it exists
             (no copying, as it's presumably already there).
             We realpath-check so no symlink or '..' can escape the folder.

    :param exec_id:        Unique identifier for the execution context.
    :param file:           Data URI, URL, or local (relative) path.
    :param return_content: If True, return a data URI of the file content.
                           If False, return the *relative* path inside the exec_id folder.
    :return:               The requested result: data URI or relative path.
    """

    # 1) Build the absolute base path for this exec_id
    temp_base = get_path(exec_id, "")
    os.makedirs(temp_base, exist_ok=True)

    # 2) Helper functions
    def _extension_from_mime(mime: str) -> str:
        ext = mimetypes.guess_extension(mime, strict=False)
        return ext if ext else ".bin"

    def _file_to_data_uri(path: str) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "application/octet-stream"
        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def _strip_base_prefix(absolute_path: str) -> str:
        # Stripe temp_base prefix and normalize path
        return absolute_path.removeprefix(temp_base).removeprefix(os.sep)

    def _ensure_inside_base(path_candidate: str) -> str:
        """
        Resolve symlinks via realpath and ensure the result is still under temp_base.
        If valid, returns the real, absolute path.
        Otherwise, raises ValueError.
        """
        real_candidate = os.path.realpath(path_candidate)
        real_base = os.path.realpath(temp_base)
        # Must be either exactly the folder or inside it
        if (
            not real_candidate.startswith(real_base + os.sep)
            and real_candidate != real_base
        ):
            raise ValueError(
                "Local file path is outside the temp_base directory. Access denied."
            )
        return real_candidate

    # 3) Distinguish between data URI, URL, or local path
    if file.startswith("data:"):
        # === Data URI ===
        match = re.match(r"^data:([^;]+);base64,(.*)$", file, re.DOTALL)
        if not match:
            raise ValueError(
                "Invalid data URI format. Expected data:<mime>;base64,<data>"
            )

        mime_type = match.group(1).strip().lower()
        b64_content = match.group(2).strip()

        # Generate random filename with guessed extension
        extension = _extension_from_mime(mime_type)
        local_filename = str(uuid.uuid4()) + extension
        # Our intended path
        intended_path = os.path.join(temp_base, local_filename)
        absolute_path = _ensure_inside_base(intended_path)

        # Decode and write
        raw_bytes = base64.b64decode(b64_content)
        with open(absolute_path, "wb") as f:
            f.write(raw_bytes)

    elif file.startswith("http://") or file.startswith("https://"):
        # === URL ===
        parsed_url = urlparse(file)
        basename = os.path.basename(parsed_url.path) or str(uuid.uuid4())

        intended_path = os.path.join(temp_base, basename)
        absolute_path = _ensure_inside_base(intended_path)

        # Download
        resp = requests.get(file)
        resp.raise_for_status()
        with open(absolute_path, "wb") as f:
            f.write(resp.content)

    else:
        # === Local path (relative to temp_base) ===
        # interpret 'file' as a sub-path, then realpath-check it
        intended_path = os.path.join(temp_base, file)
        absolute_path = _ensure_inside_base(intended_path)

        # Check file actually exists
        if not os.path.isfile(absolute_path):
            raise ValueError(f"Local file does not exist: {absolute_path}")

    # 4) Return result
    if return_content:
        return _file_to_data_uri(absolute_path)
    else:
        return _strip_base_prefix(absolute_path)
