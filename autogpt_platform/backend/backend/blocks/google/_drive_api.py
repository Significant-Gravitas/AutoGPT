"""Shared models and helpers for the Google Drive blocks."""

import re
from enum import Enum
from typing import Any, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from backend.util.exceptions import BlockExecutionError, BlockInputError
from backend.util.settings import Settings

from ._auth import GoogleCredentials
from ._drive import GoogleDriveFile

DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
FILE_FIELDS = (
    "id, name, mimeType, webViewLink, iconLink, size, createdTime, modifiedTime, "
    "owners(emailAddress), parents, description, shared, starred, trashed"
)
LIST_FIELDS = f"nextPageToken, incompleteSearch, files({FILE_FIELDS})"

_ID_IN_URL = re.compile(
    r"/(?:d|folders)/([A-Za-z0-9_-]{10,})|[?&]id=([A-Za-z0-9_-]{10,})"
)


class DriveFile(GoogleDriveFile):
    """A Drive file with the metadata Drive search and lookups return.

    It is a GoogleDriveFile, so it chains into the Drive, Docs and Sheets
    blocks with the credentials that found it.
    """

    size: Optional[int] = Field(
        default=None,
        description="Size in bytes (empty for Google Docs, Sheets and Slides)",
    )
    created_time: Optional[str] = Field(
        default=None,
        alias="createdTime",
        description="When the file was created (RFC 3339)",
    )
    modified_time: Optional[str] = Field(
        default=None,
        alias="modifiedTime",
        description="When the file was last modified (RFC 3339)",
    )
    owners: list[str] = Field(
        default_factory=list, description="Email addresses of the file's owners"
    )
    parents: list[str] = Field(
        default_factory=list, description="IDs of the folders that contain the file"
    )
    description: Optional[str] = Field(
        default=None, description="The file's description"
    )
    shared: Optional[bool] = Field(
        default=None, description="Whether the file is shared"
    )
    starred: Optional[bool] = Field(
        default=None, description="Whether the file is starred"
    )
    trashed: Optional[bool] = Field(
        default=None, description="Whether the file is in the trash"
    )


class DrivePermission(BaseModel):
    """Who can access a Drive file, and how."""

    id: str = Field(description="Permission ID")
    type: str = Field(description="user, group, domain or anyone")
    role: str = Field(
        description="owner, organizer, fileOrganizer, writer, commenter or reader"
    )
    email_address: Optional[str] = Field(
        default=None,
        description="Email of the user or group (for user and group permissions)",
    )
    domain: Optional[str] = Field(
        default=None, description="Domain (for domain permissions)"
    )
    display_name: Optional[str] = Field(
        default=None, description="Name of the user or group"
    )


class GoogleFileType(str, Enum):
    ANY = "any"
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    FOLDER = "folder"
    PDF = "pdf"
    IMAGE = "image"
    VIDEO = "video"


FILE_TYPE_CLAUSES: dict[GoogleFileType, str] = {
    GoogleFileType.DOCUMENT: "mimeType = 'application/vnd.google-apps.document'",
    GoogleFileType.SPREADSHEET: "mimeType = 'application/vnd.google-apps.spreadsheet'",
    GoogleFileType.PRESENTATION: "mimeType = 'application/vnd.google-apps.presentation'",
    GoogleFileType.FOLDER: f"mimeType = '{FOLDER_MIME_TYPE}'",
    GoogleFileType.PDF: "mimeType = 'application/pdf'",
    GoogleFileType.IMAGE: "mimeType contains 'image/'",
    GoogleFileType.VIDEO: "mimeType contains 'video/'",
}


def build_drive_service(credentials: GoogleCredentials):
    settings = Settings()
    creds = Credentials(
        token=(
            credentials.access_token.get_secret_value()
            if credentials.access_token
            else None
        ),
        refresh_token=(
            credentials.refresh_token.get_secret_value()
            if credentials.refresh_token
            else None
        ),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.secrets.google_client_id,
        client_secret=settings.secrets.google_client_secret,
        scopes=credentials.scopes,
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def to_drive_file(item: dict[str, Any], credentials_id: str | None) -> DriveFile:
    """Map a Drive API file resource to a chainable DriveFile."""
    mime_type = item.get("mimeType")
    return DriveFile(
        id=item["id"],
        name=item.get("name"),
        mimeType=mime_type,
        url=item.get("webViewLink"),
        iconUrl=item.get("iconLink"),
        isFolder=mime_type == FOLDER_MIME_TYPE,
        size=int(item["size"]) if item.get("size") else None,
        createdTime=item.get("createdTime"),
        modifiedTime=item.get("modifiedTime"),
        owners=[
            owner["emailAddress"]
            for owner in item.get("owners", [])
            if owner.get("emailAddress")
        ],
        parents=item.get("parents", []),
        description=item.get("description"),
        shared=item.get("shared"),
        starred=item.get("starred"),
        trashed=item.get("trashed"),
        _credentials_id=credentials_id,
    )


def parse_folder_id(value: str) -> str:
    """Accept a folder ID, a Drive folder URL, or ``root`` for My Drive."""
    value = value.strip()
    match = _ID_IN_URL.search(value)
    if match:
        return match.group(1) or match.group(2)
    return value


def quote_drive_value(value: str) -> str:
    """Quote a string for a Drive search clause."""
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def require_file(
    file: GoogleDriveFile | None, block_name: str, block_id: str
) -> GoogleDriveFile:
    if file is None or not file.id:
        raise BlockInputError(
            message="Pick a Drive file, or connect one from a Drive search.",
            block_name=block_name,
            block_id=block_id,
        )
    return file


def drive_error(exc: HttpError, block_name: str, block_id: str) -> BlockExecutionError:
    """Turn a Drive API error into a message the user can act on."""
    if exc.status_code == 404:
        message = (
            "Google Drive couldn't find that file, or the connected Google "
            "account can't open it."
        )
    elif exc.status_code == 403 and "insufficient" in str(exc.reason).lower():
        message = (
            "The connected Google account hasn't granted the Drive access this "
            "block needs. Reconnect Google and approve Drive access."
        )
    else:
        message = f"Google Drive API error {exc.status_code}: {exc.reason}"
    return BlockExecutionError(
        message=message, block_name=block_name, block_id=block_id
    )
