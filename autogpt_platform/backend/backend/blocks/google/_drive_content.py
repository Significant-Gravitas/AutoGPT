"""Read Drive file content as text or bytes, and move bytes in and out of
the execution's file store."""

import io
import mimetypes
from enum import Enum
from pathlib import Path

import pypdfium2
from googleapiclient.http import MediaIoBaseDownload

from backend.data.execution import ExecutionContext
from backend.util.file import get_exec_file_path, sanitize_filename, store_media_file
from backend.util.type import MediaFileType

MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
GOOGLE_TYPE_PREFIX = "application/vnd.google-apps."
DOCUMENT = "application/vnd.google-apps.document"
SPREADSHEET = "application/vnd.google-apps.spreadsheet"
PRESENTATION = "application/vnd.google-apps.presentation"
PDF = "application/pdf"

TEXT_EXPORTS = {
    DOCUMENT: "text/markdown",
    SPREADSHEET: "text/csv",
    PRESENTATION: "text/plain",
}
TEXT_LIKE_TYPES = {
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-yaml",
    "application/yaml",
    "application/x-ndjson",
    "application/sql",
}


class ExportFormat(str, Enum):
    """What to turn a Google Doc, Sheet or Slides file into when downloading."""

    PDF = "pdf"
    OFFICE = "office"
    TEXT = "text"


EXPORT_TYPES: dict[str, dict[ExportFormat, str]] = {
    DOCUMENT: {
        ExportFormat.PDF: PDF,
        ExportFormat.OFFICE: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ExportFormat.TEXT: "text/markdown",
    },
    SPREADSHEET: {
        ExportFormat.PDF: PDF,
        ExportFormat.OFFICE: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ExportFormat.TEXT: "text/csv",
    },
    PRESENTATION: {
        ExportFormat.PDF: PDF,
        ExportFormat.OFFICE: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ExportFormat.TEXT: "text/plain",
    },
}


def is_google_type(mime_type: str) -> bool:
    return mime_type.startswith(GOOGLE_TYPE_PREFIX)


def read_as_text(service, metadata: dict) -> str:
    """Return a Drive file's content as text, exporting Google files first."""
    mime_type = metadata.get("mimeType", "")
    if mime_type in TEXT_EXPORTS:
        data = export_file(service, metadata["id"], TEXT_EXPORTS[mime_type])
        return data.decode("utf-8", errors="replace")
    if not (
        mime_type == PDF
        or mime_type.startswith("text/")
        or mime_type in TEXT_LIKE_TYPES
    ):
        raise ValueError(
            f"Can't read a {mime_type or 'file of unknown type'} as text. "
            "Use the Google Drive Download File block to get the file itself."
        )
    data = download_file(service, metadata)
    if mime_type == PDF:
        return pdf_to_text(data)
    return data.decode("utf-8", errors="replace")


def fetch_file_bytes(
    service, metadata: dict, export_format: ExportFormat
) -> tuple[bytes, str]:
    """Return a file's bytes and MIME type, exporting Google files."""
    mime_type = metadata.get("mimeType", "")
    if not is_google_type(mime_type):
        return download_file(service, metadata), mime_type
    formats = EXPORT_TYPES.get(mime_type, {ExportFormat.PDF: PDF})
    export_type = formats.get(export_format)
    if export_type is None:
        raise ValueError(
            f"{mime_type} files can only be downloaded as PDF. Set the format to PDF."
        )
    return export_file(service, metadata["id"], export_type), export_type


def export_file(service, file_id: str, mime_type: str) -> bytes:
    return service.files().export(fileId=file_id, mimeType=mime_type).execute()


def download_file(service, metadata: dict) -> bytes:
    size = int(metadata.get("size") or 0)
    if size > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            f"The file is {size // (1024 * 1024)} MB; the limit is "
            f"{MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB."
        )
    request = service.files().get_media(fileId=metadata["id"], supportsAllDrives=True)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request, chunksize=8 * 1024 * 1024)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def pdf_to_text(data: bytes) -> str:
    pdf = pypdfium2.PdfDocument(data)
    try:
        pages = []
        for page in pdf:
            text_page = page.get_textpage()
            pages.append(text_page.get_text_range())
            text_page.close()
            page.close()
        return "\n\n".join(pages)
    finally:
        pdf.close()


def file_name_for(name: str, mime_type: str) -> str:
    """Give an exported file a matching extension (``Report`` -> ``Report.pdf``)."""
    extension = mimetypes.guess_extension(mime_type, strict=False) or ""
    if extension and not name.lower().endswith(extension):
        name += extension
    return sanitize_filename(name)


async def save_to_file_store(
    data: bytes, file_name: str, execution_context: ExecutionContext
) -> MediaFileType:
    """Write bytes under the execution's file folder and return a block output
    reference (a workspace file in CoPilot, a data URI in graphs)."""
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


async def load_from_file_store(
    file: MediaFileType, execution_context: ExecutionContext
) -> tuple[str, bytes]:
    """Materialise a MediaFileType input and return its (name, bytes)."""
    local_path = await store_media_file(
        file=file,
        execution_context=execution_context,
        return_format="for_local_processing",
    )
    if not execution_context.graph_exec_id:
        raise ValueError("execution_context.graph_exec_id is required")
    path = Path(get_exec_file_path(execution_context.graph_exec_id, local_path))
    return path.name, path.read_bytes()
