import mimetypes
import uuid
from pathlib import Path
from typing import Any, Literal

from backend.data.model import GoogleDriveFile, GoogleDrivePickerField
from backend.util.file import get_exec_file_path
from backend.util.request import Requests
from backend.util.type import MediaFileType
from backend.util.virus_scanner import scan_content_safe

DRIVE_API_URL = "https://www.googleapis.com/drive/v3/files"
_requests = Requests(trusted_origins=["https://www.googleapis.com"])
AttachmentView = Literal[
    "DOCS",
    "DOCUMENTS",
    "SPREADSHEETS",
    "PRESENTATIONS",
    "DOCS_IMAGES",
    "FOLDERS",
]
ATTACHMENT_VIEWS: tuple[AttachmentView, ...] = (
    "DOCS",
    "DOCUMENTS",
    "SPREADSHEETS",
    "PRESENTATIONS",
    "DOCS_IMAGES",
    "FOLDERS",
)


def GoogleDriveAttachmentField(
    *,
    title: str,
    description: str | None = None,
    placeholder: str | None = None,
    multiselect: bool = True,
    allowed_mime_types: list[str] | None = None,
    **extra: Any,
) -> Any:
    return GoogleDrivePickerField(
        multiselect=multiselect,
        allowed_views=list(ATTACHMENT_VIEWS),
        allowed_mime_types=allowed_mime_types,
        title=title,
        description=description,
        placeholder=placeholder or "Choose files from Google Drive",
        **extra,
    )


async def drive_file_to_media_file(
    drive_file: GoogleDriveFile, *, graph_exec_id: str, user_id: str
) -> MediaFileType:
    if drive_file.is_folder:
        raise ValueError("Google Drive selection must be a file.")
    if not drive_file.access_token:
        raise ValueError(
            "Google Drive selection is missing an access token. Please re-select the file."
        )

    url = f"{DRIVE_API_URL}/{drive_file.id}?alt=media"
    response = await _requests.get(
        url, headers={"Authorization": f"Bearer {drive_file.access_token}"}
    )

    mime_type = drive_file.mime_type or response.headers.get(
        "content-type", "application/octet-stream"
    )

    MAX_FILE_SIZE = 100 * 1024 * 1024
    if len(response.content) > MAX_FILE_SIZE:
        raise ValueError(
            f"File too large: {len(response.content)} bytes > {MAX_FILE_SIZE} bytes"
        )

    base_path = Path(get_exec_file_path(graph_exec_id, ""))
    base_path.mkdir(parents=True, exist_ok=True)

    extension = mimetypes.guess_extension(mime_type, strict=False) or ".bin"
    filename = f"{uuid.uuid4()}{extension}"
    target_path = base_path / filename

    await scan_content_safe(response.content, filename=filename)
    target_path.write_bytes(response.content)

    return MediaFileType(str(target_path.relative_to(base_path)))
