import asyncio
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.data.model import SchemaField
from backend.util.file import get_exec_file_path
from backend.util.request import Requests
from backend.util.type import MediaFileType
from backend.util.virus_scanner import scan_content_safe

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


class GoogleDriveFile(BaseModel):
    """Represents a single file/folder picked from Google Drive"""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(description="Google Drive file/folder ID")
    name: Optional[str] = Field(None, description="File/folder name")
    mime_type: Optional[str] = Field(
        None,
        alias="mimeType",
        description="MIME type (e.g., application/vnd.google-apps.document)",
    )
    url: Optional[str] = Field(None, description="URL to open the file")
    icon_url: Optional[str] = Field(None, alias="iconUrl", description="Icon URL")
    is_folder: Optional[bool] = Field(
        None, alias="isFolder", description="Whether this is a folder"
    )


def GoogleDrivePickerField(
    multiselect: bool = False,
    allow_folder_selection: bool = False,
    allowed_views: Optional[list[AttachmentView]] = None,
    allowed_mime_types: Optional[list[str]] = None,
    scopes: Optional[list[str]] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    placeholder: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Creates a Google Drive Picker input field.

    Args:
        multiselect: Allow selecting multiple files/folders (default: False)
        allow_folder_selection: Allow selecting folders (default: False)
        allowed_views: List of view types to show in picker (default: ["DOCS"])
        allowed_mime_types: Filter by MIME types (e.g., ["application/pdf"])
        title: Field title shown in UI
        description: Field description/help text
        placeholder: Placeholder text for the button
        **kwargs: Additional SchemaField arguments (advanced, hidden, etc.)

    Returns:
        Field definition that produces:
        - Single GoogleDriveFile when multiselect=False
        - list[GoogleDriveFile] when multiselect=True

    Example:
        >>> class MyBlock(Block):
        ...     class Input(BlockSchema):
        ...         document: GoogleDriveFile = GoogleDrivePickerField(
        ...             title="Select Document",
        ...             allowed_views=["DOCUMENTS"],
        ...         )
        ...
        ...         files: list[GoogleDriveFile] = GoogleDrivePickerField(
        ...             title="Select Multiple Files",
        ...             multiselect=True,
        ...             allow_folder_selection=True,
        ...         )
    """
    # Build configuration that will be sent to frontend
    picker_config = {
        "multiselect": multiselect,
        "allow_folder_selection": allow_folder_selection,
        "allowed_views": list(allowed_views) if allowed_views else ["DOCS"],
    }

    # Add optional configurations
    if allowed_mime_types:
        picker_config["allowed_mime_types"] = list(allowed_mime_types)

    # Determine required scopes based on config
    base_scopes = scopes if scopes is not None else []
    picker_scopes: set[str] = set(base_scopes)
    if allow_folder_selection:
        picker_scopes.add("https://www.googleapis.com/auth/drive")
    else:
        # Use drive.file for minimal scope - only access files selected by user in picker
        picker_scopes.add("https://www.googleapis.com/auth/drive.file")

    views = set(allowed_views or [])
    if "SPREADSHEETS" in views:
        picker_scopes.add("https://www.googleapis.com/auth/spreadsheets.readonly")
    if "DOCUMENTS" in views or "DOCS" in views:
        picker_scopes.add("https://www.googleapis.com/auth/documents.readonly")

    picker_config["scopes"] = sorted(picker_scopes)

    # Set appropriate default value
    default_value = [] if multiselect else None

    # Use SchemaField to handle format properly
    return SchemaField(
        default=default_value,
        title=title,
        description=description,
        placeholder=placeholder or "Choose from Google Drive",
        format="google-drive-picker",
        advanced=False,
        json_schema_extra={
            "google_drive_picker_config": picker_config,
            **kwargs,
        },
    )


DRIVE_API_URL = "https://www.googleapis.com/drive/v3/files"
_requests = Requests(trusted_origins=["https://www.googleapis.com"])


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
    drive_file: GoogleDriveFile, *, graph_exec_id: str, access_token: str
) -> MediaFileType:
    if drive_file.is_folder:
        raise ValueError("Google Drive selection must be a file.")
    if not access_token:
        raise ValueError("Google Drive access token is required for file download.")

    url = f"{DRIVE_API_URL}/{drive_file.id}?alt=media"
    response = await _requests.get(
        url, headers={"Authorization": f"Bearer {access_token}"}
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
    await asyncio.to_thread(target_path.write_bytes, response.content)

    return MediaFileType(str(target_path.relative_to(base_path)))
