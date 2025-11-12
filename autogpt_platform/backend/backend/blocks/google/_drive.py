from __future__ import annotations

import base64
from typing import Any, Literal

from backend.data.model import GoogleDriveFile, GoogleDrivePickerField
from backend.util.request import Requests
from backend.util.type import MediaFileType

DRIVE_API_URL = "https://www.googleapis.com/drive/v3/files"
TRUSTED_ORIGINS = ["https://www.googleapis.com"]
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


async def drive_file_to_media_file(drive_file: GoogleDriveFile) -> MediaFileType:
    if drive_file.is_folder:
        raise ValueError("Google Drive selection must be a file.")
    if not drive_file.access_token:
        raise ValueError(
            "Google Drive selection is missing an access token. Please re-select the file."
        )

    url = f"{DRIVE_API_URL}/{drive_file.id}?alt=media"
    response = await Requests(
        trusted_origins=TRUSTED_ORIGINS,
        extra_headers={"Authorization": f"Bearer {drive_file.access_token}"},
    ).get(url)

    mime_type = drive_file.mime_type or response.headers.get(
        "content-type", "application/octet-stream"
    )
    encoded = base64.b64encode(response.content).decode("utf-8")

    return MediaFileType(f"data:{mime_type};base64,{encoded}")
