"""Shared models and helpers for the Google Slides blocks."""

import re
from pathlib import Path
from typing import Any, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError, BlockInputError
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.settings import Settings
from backend.util.type import MediaFileType

from ._auth import GoogleCredentials
from ._drive import GoogleDriveFile, GoogleDriveFileField

DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
PRESENTATIONS_SCOPE = "https://www.googleapis.com/auth/presentations"

PRESENTATION_MIME_TYPE = "application/vnd.google-apps.presentation"
SLIDES_ICON_URL = "https://www.gstatic.com/images/branding/product/1x/slides_48dp.png"
TITLE_PLACEHOLDERS = ("TITLE", "CENTERED_TITLE")

_SLIDE_IN_URL = re.compile(r"[#?&]slide=id\.([A-Za-z0-9_:-]+)")
_ELEMENT_TYPES = {
    "shape": "shape",
    "table": "table",
    "image": "image",
    "video": "video",
    "line": "line",
    "sheetsChart": "sheets_chart",
    "wordArt": "word_art",
    "elementGroup": "group",
    "speakerSpotlight": "speaker_spotlight",
}


class SlideElement(BaseModel):
    """One thing on a slide: a text box, shape, table, image and so on."""

    element_id: str = Field(description="The element's object ID")
    type: str = Field(
        description=(
            "shape, table, image, video, line, sheets_chart, word_art, group "
            "or speaker_spotlight"
        )
    )
    shape_type: Optional[str] = Field(
        default=None, description="For shapes: TEXT_BOX, RECTANGLE, ELLIPSE and so on"
    )
    placeholder: Optional[str] = Field(
        default=None,
        description="For layout placeholders: TITLE, CENTERED_TITLE, SUBTITLE, BODY and so on",
    )
    text: str = Field(
        default="", description="The element's text (table cells separated by ' | ')"
    )


class SlideSummary(BaseModel):
    """A slide's ID, position, text and speaker notes."""

    slide_id: str = Field(description="The slide's object ID")
    index: int = Field(description="The slide's position, counting from 0")
    title: str = Field(default="", description="The text of the slide's title")
    text: str = Field(
        default="", description="All text on the slide, from shapes and tables"
    )
    speaker_notes: str = Field(default="", description="The slide's speaker notes")


def presentation_field(description: str, *, edit: bool = False) -> GoogleDriveFile:
    """A Drive picker for Google Slides files. Edits also need the Slides scope."""
    scopes = [DRIVE_FILE_SCOPE, PRESENTATIONS_SCOPE] if edit else [DRIVE_FILE_SCOPE]
    return GoogleDriveFileField(
        title="Presentation",
        description=description,
        allowed_views=["PRESENTATIONS"],
        credentials_scopes=scopes,
    )


def build_slides_service(credentials: GoogleCredentials):
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
    return build("slides", "v1", credentials=creds, cache_discovery=False)


def get_page(service, presentation_id: str, page_id: str) -> dict:
    return (
        service.presentations()
        .pages()
        .get(presentationId=presentation_id, pageObjectId=page_id)
        .execute()
    )


def batch_update(service, presentation_id: str, requests: list[dict]) -> dict:
    return (
        service.presentations()
        .batchUpdate(presentationId=presentation_id, body={"requests": requests})
        .execute()
    )


def presentation_file(
    presentation_id: str, name: Optional[str], credentials_id: Optional[str]
) -> GoogleDriveFile:
    """A chainable reference to a presentation, carrying the credentials used."""
    return GoogleDriveFile(
        id=presentation_id,
        name=name,
        mimeType=PRESENTATION_MIME_TYPE,
        url=f"https://docs.google.com/presentation/d/{presentation_id}/edit",
        iconUrl=SLIDES_ICON_URL,
        isFolder=False,
        _credentials_id=credentials_id,
    )


def require_presentation(
    file: GoogleDriveFile | None, block_name: str, block_id: str
) -> GoogleDriveFile:
    if file is None or not file.id:
        raise BlockInputError(
            message="Pick a Google Slides presentation, or connect one from another block.",
            block_name=block_name,
            block_id=block_id,
        )
    if file.mime_type and file.mime_type != PRESENTATION_MIME_TYPE:
        raise BlockInputError(
            message=(
                f"'{file.name or file.id}' is not a Google Slides presentation "
                f"({file.mime_type}). Pick a Google Slides file. To use a "
                "PowerPoint file, open it in Google Slides and choose File > "
                "Save as Google Slides."
            ),
            block_name=block_name,
            block_id=block_id,
        )
    return file


def require_slide_id(value: str, block_name: str, block_id: str) -> str:
    """Accept a slide's object ID or a link to the slide (...#slide=id.<ID>)."""
    value = value.strip()
    match = _SLIDE_IN_URL.search(value)
    if match:
        return match.group(1)
    if value and "://" not in value:
        return value
    raise BlockInputError(
        message=(
            "Give a slide ID from Google Slides Read Presentation, or a link "
            "copied while the slide is selected (it ends in #slide=id.<ID>)."
        ),
        block_name=block_name,
        block_id=block_id,
    )


def slides_error(
    exc: HttpError, block_name: str, block_id: str, bad_request_hint: str = ""
) -> BlockExecutionError:
    """Turn a Slides API error into a message the user can act on."""
    reason = str(exc.reason).rstrip(".")
    if exc.status_code == 404:
        message = (
            "Google Slides couldn't find that presentation or slide, or the "
            "connected Google account can't open it."
        )
    elif exc.status_code == 403 and "insufficient" in reason.lower():
        message = (
            "The connected Google account hasn't granted the Google Slides access "
            "this block needs. Reconnect Google and approve access to Google Slides."
        )
    elif exc.status_code == 403:
        message = (
            f"Google Slides denied access ({reason}). Check that the connected "
            "Google account can open the presentation, and can edit it for changes."
        )
    else:
        message = f"Google Slides API error {exc.status_code}: {reason}."
        if exc.status_code == 400 and bad_request_hint:
            message = f"{message} {bad_request_hint}"
    return BlockExecutionError(
        message=message, block_name=block_name, block_id=block_id
    )


def parse_slide(slide: dict[str, Any], index: int) -> SlideSummary:
    elements = flatten_elements(slide.get("pageElements", []))
    return SlideSummary(
        slide_id=slide.get("objectId", ""),
        index=index,
        title=next(
            (
                e.text
                for e in elements
                if e.placeholder in TITLE_PLACEHOLDERS and e.text
            ),
            "",
        ),
        text=slide_text(elements),
        speaker_notes=speaker_notes(slide),
    )


def presentation_text(title: str, slides: list[SlideSummary]) -> str:
    """A whole deck as one Markdown document, for AI blocks."""
    sections = [f"# {title}"] if title else []
    for slide in slides:
        parts = [f"## Slide {slide.index + 1} (id: {slide.slide_id})"]
        if slide.text:
            parts.append(slide.text)
        if slide.speaker_notes:
            parts.append(f"Speaker notes:\n{slide.speaker_notes}")
        sections.append("\n\n".join(parts))
    return "\n\n".join(sections)


def slide_text(elements: list[SlideElement]) -> str:
    return "\n".join(element.text for element in elements if element.text)


def flatten_elements(elements: list[dict[str, Any]]) -> list[SlideElement]:
    """List a slide's elements, each group followed by the elements inside it."""
    flat: list[SlideElement] = []
    for element in elements:
        flat.append(to_slide_element(element))
        flat.extend(
            flatten_elements(element.get("elementGroup", {}).get("children", []))
        )
    return flat


def to_slide_element(element: dict[str, Any]) -> SlideElement:
    shape = element.get("shape", {})
    return SlideElement(
        element_id=element.get("objectId", ""),
        type=next(
            (name for key, name in _ELEMENT_TYPES.items() if key in element),
            "unknown",
        ),
        shape_type=shape.get("shapeType"),
        placeholder=shape.get("placeholder", {}).get("type"),
        text=element_text(element),
    )


def element_text(element: dict[str, Any]) -> str:
    if "shape" in element:
        return plain_text(element["shape"].get("text"))
    if "table" in element:
        rows = [
            " | ".join(
                plain_text(cell.get("text")) for cell in row.get("tableCells", [])
            )
            for row in element["table"].get("tableRows", [])
        ]
        return "\n".join(row for row in rows if row.strip(" |"))
    if "wordArt" in element:
        return element["wordArt"].get("renderedText", "")
    return ""


def plain_text(text: Optional[dict[str, Any]]) -> str:
    """Join a shape's text runs. Slides marks soft line breaks with a vertical tab."""
    if not text:
        return ""
    runs = (
        (item.get("textRun") or item.get("autoText") or {}).get("content", "")
        for item in text.get("textElements", [])
    )
    return "".join(runs).replace("\x0b", "\n").strip()


def speaker_notes(slide: dict[str, Any]) -> str:
    notes_id = speaker_notes_id(slide)
    notes_page = slide.get("slideProperties", {}).get("notesPage", {})
    return next(
        (
            plain_text(element["shape"].get("text"))
            for element in notes_page.get("pageElements", [])
            if element.get("objectId") == notes_id and "shape" in element
        ),
        "",
    )


def speaker_notes_id(slide: dict[str, Any]) -> Optional[str]:
    """The ID of the shape that holds a slide's speaker notes. Google creates
    the shape on the first insert if it doesn't exist yet."""
    notes_page = slide.get("slideProperties", {}).get("notesPage", {})
    return notes_page.get("notesProperties", {}).get("speakerNotesObjectId")


async def save_to_file_store(
    data: bytes, file_name: str, execution_context: ExecutionContext
) -> MediaFileType:
    """Write bytes under the execution's file folder and return a block output
    reference (a workspace file in CoPilot, a data URI in agents)."""
    if not execution_context.graph_exec_id:
        raise ValueError("execution_context.graph_exec_id is required")
    path = Path(get_exec_file_path(execution_context.graph_exec_id, file_name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return await store_media_file(
        file=MediaFileType(file_name),
        execution_context=execution_context,
        return_format="for_block_output",
    )
