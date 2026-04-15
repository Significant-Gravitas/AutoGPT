from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.data.model import SchemaField

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


class _GoogleDriveFileBase(BaseModel):
    """Internal base class for Google Drive file representation."""

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


class GoogleDriveFile(_GoogleDriveFileBase):
    """
    Represents a Google Drive file/folder with optional credentials for chaining.

    Used for both inputs and outputs in Google Drive blocks. The `_credentials_id`
    field enables chaining between blocks - when one block outputs a file, the
    next block can use the same credentials to access it.

    When used with GoogleDriveFileField(), the frontend renders a combined
    auth + file picker UI that automatically populates `_credentials_id`.
    """

    # Hidden field for credential ID - populated by frontend, preserved in outputs
    credentials_id: Optional[str] = Field(
        None,
        alias="_credentials_id",
        description="Internal: credential ID for authentication",
    )


def GoogleDriveFileField(
    *,
    title: str,
    description: str | None = None,
    credentials_kwarg: str = "credentials",
    credentials_scopes: list[str] | None = None,
    allowed_views: list[AttachmentView] | None = None,
    allowed_mime_types: list[str] | None = None,
    placeholder: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Creates a Google Drive file input field with auto-generated credentials.

    This field type produces a single UI element that handles both:
    1. Google OAuth authentication
    2. File selection via Google Drive Picker

    The system automatically generates a credentials field, and the credentials
    are passed to the run() method using the specified kwarg name.

    Args:
        title: Field title shown in UI
        description: Field description/help text
        credentials_kwarg: Name of the kwarg that will receive GoogleCredentials
                          in the run() method (default: "credentials")
        credentials_scopes: OAuth scopes required (default: drive.file)
        allowed_views: List of view types to show in picker (default: ["DOCS"])
        allowed_mime_types: Filter by MIME types
        placeholder: Placeholder text for the button
        **kwargs: Additional SchemaField arguments

    Returns:
        Field definition that produces GoogleDriveFile

    Example:
        >>> class MyBlock(Block):
        ...     class Input(BlockSchemaInput):
        ...         spreadsheet: GoogleDriveFile = GoogleDriveFileField(
        ...             title="Select Spreadsheet",
        ...             credentials_kwarg="creds",
        ...             allowed_views=["SPREADSHEETS"],
        ...         )
        ...
        ...     async def run(
        ...         self, input_data: Input, *, creds: GoogleCredentials, **kwargs
        ...     ):
        ...         # creds is automatically populated
        ...         file = input_data.spreadsheet
    """

    # Determine scopes - drive.file is sufficient for picker-selected files
    scopes = credentials_scopes or ["https://www.googleapis.com/auth/drive.file"]

    # Build picker configuration with auto_credentials embedded
    picker_config = {
        "multiselect": False,
        "allow_folder_selection": False,
        "allowed_views": list(allowed_views) if allowed_views else ["DOCS"],
        "scopes": scopes,
        # Auto-credentials config tells frontend to include _credentials_id in output
        "auto_credentials": {
            "provider": "google",
            "type": "oauth2",
            "scopes": scopes,
            "kwarg_name": credentials_kwarg,
        },
    }

    if allowed_mime_types:
        picker_config["allowed_mime_types"] = list(allowed_mime_types)

    return SchemaField(
        default=None,
        title=title,
        description=description,
        placeholder=placeholder or "Select from Google Drive",
        # Use google-drive-picker format so frontend renders existing component
        format="google-drive-picker",
        advanced=False,
        json_schema_extra={
            "google_drive_picker_config": picker_config,
            # Also keep auto_credentials at top level for backend detection
            "auto_credentials": {
                "provider": "google",
                "type": "oauth2",
                "scopes": scopes,
                "kwarg_name": credentials_kwarg,
            },
            **kwargs,
        },
    )
