import base64
import mimetypes
import os
import re
import tempfile
import uuid
from urllib.parse import urlparse

import requests

TEMP_DIR = tempfile.gettempdir()


def get_path(exec_id: str, path: str) -> str:
    return os.path.join(TEMP_DIR, "exec_file", exec_id, path)


def store_temp_file(exec_id: str, file: str, return_content: bool = False) -> str:
    """
    Safely handle 'file' (a data URI, a URL, or a local path relative to {temp}/exec_file/{exec_id}),
    placing or verifying it under:
        {tempdir}/exec_file/{exec_id}/...

    If 'return_content=True', return a data URI (data:<mime>;base64,<content>).
    Otherwise, return the *relative path* (prefix stripped) inside that folder.

    What happens for each 'file' type:
      - Data URI (starting with "data:"):
          -> decode and store in a new random file in that folder
      - URL (http:// or https://):
          -> download and store in that folder
      - Local path (anything else):
          -> interpret as relative to that folder, verify it is inside & file exists
             (no copying, as it's presumably already there)

    :param exec_id:        Unique identifier for the execution context.
    :param file:           Data URI, URL, or local (relative) path.
    :param return_content: If True, return a data URI of the file content.
                           If False, return the *relative* path inside the exec_id folder.
    :return:               The requested result: data URI or relative path.
    """

    # 1) Build the absolute base path for this exec_id
    temp_base = get_path(exec_id, "")
    os.makedirs(temp_base, exist_ok=True)

    # Helper to guess an extension from MIME
    def _extension_from_mime(mime: str) -> str:
        ext = mimetypes.guess_extension(mime, strict=False)
        return ext if ext else ".bin"

    # Helper to create a data URI from a local file
    def _file_to_data_uri(path: str) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "application/octet-stream"
        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    # Helper to convert an absolute path (inside temp_base) to a relative path
    def _strip_base_prefix(absolute_path: str) -> str:
        # This will give a relative path from temp_base to absolute_path
        rel = os.path.relpath(absolute_path, start=temp_base)
        return rel

    # 2) Distinguish between data URI, URL, or local path
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
        absolute_path = os.path.join(temp_base, local_filename)

        # Decode and write
        raw_bytes = base64.b64decode(b64_content)
        with open(absolute_path, "wb") as f:
            f.write(raw_bytes)

    elif file.startswith("http://") or file.startswith("https://"):
        # === URL ===
        parsed_url = urlparse(file)
        basename = os.path.basename(parsed_url.path)
        if not basename:
            # If the URL path doesn't provide a usable name, use a UUID
            basename = str(uuid.uuid4())

        absolute_path = os.path.join(temp_base, basename)

        # Download
        resp = requests.get(file)
        resp.raise_for_status()
        with open(absolute_path, "wb") as f:
            f.write(resp.content)

    else:
        # === Local path (relative to temp_base) ===
        # We do NOT allow absolute external paths. We interpret 'file' as a sub-path of temp_base.
        # Combine them
        combined_path = os.path.join(temp_base, file)  # might be "subdir/image.png"

        # Normalize to remove ".." or such
        absolute_path = os.path.normpath(combined_path)
        base_dir = os.path.normpath(temp_base)

        # Verify it's still inside temp_base
        if (
            not absolute_path.startswith(base_dir + os.sep)
            and absolute_path != base_dir
        ):
            raise ValueError(
                "Local file path is outside the temp_base directory. Access denied."
            )

        # Check the file actually exists
        if not os.path.isfile(absolute_path):
            raise ValueError(f"Local file does not exist: {absolute_path}")

    if return_content:
        return _file_to_data_uri(absolute_path)
    else:
        return _strip_base_prefix(absolute_path)


# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    exec_id_example = "test"
    file = "lips_movement.mp4"

    print("File path:", store_temp_file(exec_id_example, file, return_content=False))
    print("File path:", store_temp_file(exec_id_example, file, return_content=True))

    # # A) Data URI example
    # data_uri_input = "data:text/plain;base64," + base64.b64encode(
    #     b"Hello from data URI!"
    # ).decode("utf-8")
    # result_data_uri = store_temp_file(
    #     exec_id_example, data_uri_input, return_content=False
    # )
    # print("[Data URI] => Relative path:", result_data_uri)
    #
    # # B) URL example
    # url_input = "https://example.com/index.html"
    # result_url = store_temp_file(exec_id_example, url_input, return_content=True)
    # print("[URL] => Data URI (truncated):", result_url[:100], "...")
    #
    # # C) Local path example
    # # Suppose we previously stored a file named "somefile.txt" in that folder.
    # # Let's simulate that by making a file ourselves:
    # temp_base_for_exec = os.path.join(
    #     tempfile.gettempdir(), "exec_file", exec_id_example
    # )
    # test_local_file = os.path.join(temp_base_for_exec, "somefile.txt")
    # with open(test_local_file, "w") as f:
    #     f.write("Hello from a local file!\n")
    #
    # # Now refer to it by relative path:
    # result_local = store_temp_file(exec_id_example, "somefile.txt", return_content=True)
    # print("[Local path] => Data URI:", result_local)
