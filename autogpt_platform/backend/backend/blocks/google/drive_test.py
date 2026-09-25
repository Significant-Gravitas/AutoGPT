"""Unit tests for the Google Drive blocks' query, content and chaining logic.

The blocks' own test_input/test_mock cases mock the API calls away; these
cover what those mocks skip.
"""

from datetime import datetime, timezone

import httplib2
import pytest
from googleapiclient.errors import HttpError

from backend.blocks.google import _drive_content
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._drive import GoogleDriveFile
from backend.blocks.google._drive_api import (
    drive_error,
    parse_folder_id,
    quote_drive_value,
    to_drive_file,
)
from backend.blocks.google._drive_content import (
    ExportFormat,
    fetch_file_bytes,
    pdf_to_text,
    read_as_text,
)
from backend.blocks.google.drive_manage import (
    GoogleDriveCreateFileBlock,
    GoogleDriveMoveFileBlock,
)
from backend.blocks.google.drive_search import (
    GoogleDriveSearchFilesBlock,
    build_search_query,
)
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockInputError
from backend.util.json import to_dict

DOC = "application/vnd.google-apps.document"
DRAWING = "application/vnd.google-apps.drawing"


def _search_input(**fields) -> GoogleDriveSearchFilesBlock.Input:
    return GoogleDriveSearchFilesBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **fields}
    )


def test_search_query_defaults_to_skipping_trash():
    assert build_search_query(_search_input()) == "trashed = false"


def test_search_query_combines_filters():
    query = build_search_query(
        _search_input(
            name_contains="Q3 Report",
            text_contains="revenue",
            file_type="spreadsheet",
            folder_id="https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp",
            modified_after=datetime(2026, 9, 1, tzinfo=timezone.utc),
            custom_query="'me' in owners",
        )
    )
    assert query == (
        "name contains 'Q3 Report' and fullText contains 'revenue' and "
        "mimeType = 'application/vnd.google-apps.spreadsheet' and "
        "'1AbCdEfGhIjKlMnOp' in parents and "
        "modifiedTime > '2026-09-01T00:00:00+00:00' and "
        "('me' in owners) and trashed = false"
    )


def test_search_query_can_include_trash():
    query = build_search_query(_search_input(name_contains="x", include_trashed=True))
    assert query == "name contains 'x'"


def test_quote_drive_value_escapes_quotes_and_backslashes():
    assert quote_drive_value("O'Brien \\ Co") == "'O\\'Brien \\\\ Co'"


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp?usp=sharing",
            "1AbCdEfGhIjKlMnOp",
        ),
        ("https://drive.google.com/open?id=1AbCdEfGhIjKlMnOp", "1AbCdEfGhIjKlMnOp"),
        (
            "https://docs.google.com/document/d/1AbCdEfGhIjKlMnOp/edit",
            "1AbCdEfGhIjKlMnOp",
        ),
        (" 1AbCdEfGhIjKlMnOp ", "1AbCdEfGhIjKlMnOp"),
        ("root", "root"),
    ],
)
def test_parse_folder_id(value: str, expected: str):
    assert parse_folder_id(value) == expected


def test_drive_file_chains_into_picker_fields():
    """A DriveFile output must feed GoogleDriveFile inputs with its credentials."""
    drive_file = to_drive_file(
        {
            "id": "1AbCdEfGhIjKlMnOp",
            "name": "Q3 Report",
            "mimeType": DOC,
            "size": "2048",
            "owners": [{"emailAddress": "a@example.com"}, {"displayName": "no email"}],
            "modifiedTime": "2026-09-20T15:30:00.000Z",
        },
        "cred-123",
    )
    assert drive_file.size == 2048
    assert drive_file.owners == ["a@example.com"]
    assert drive_file.is_folder is False

    serialized = to_dict(drive_file)
    assert serialized["_credentials_id"] == "cred-123"
    assert serialized["modifiedTime"] == "2026-09-20T15:30:00.000Z"

    picked = GoogleDriveFile.model_validate(serialized)
    assert picked.id == "1AbCdEfGhIjKlMnOp"
    assert picked.credentials_id == "cred-123"


class _Request:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _Files:
    def __init__(self, get_result: dict | None = None):
        self.calls: list[tuple[str, dict]] = []
        self._get_result = get_result or {}

    def export(self, **kwargs):
        self.calls.append(("export", kwargs))
        return _Request(b"# Exported")

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return _Request(self._get_result)

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))
        return _Request({"id": kwargs["fileId"], "parents": [kwargs["addParents"]]})


class _Service:
    def __init__(self, files: _Files):
        self._files = files

    def files(self):
        return self._files


def test_read_as_text_exports_google_docs_as_markdown():
    files = _Files()
    text = read_as_text(_Service(files), {"id": "doc1", "mimeType": DOC})
    assert text == "# Exported"
    assert files.calls == [("export", {"fileId": "doc1", "mimeType": "text/markdown"})]


def test_read_as_text_decodes_text_files(monkeypatch):
    monkeypatch.setattr(_drive_content, "download_file", lambda *a: b"a,b\n1,2\n")
    text = read_as_text(_Service(_Files()), {"id": "f", "mimeType": "text/csv"})
    assert text == "a,b\n1,2\n"


def test_read_as_text_extracts_pdf_text(monkeypatch):
    monkeypatch.setattr(
        _drive_content, "download_file", lambda *a: _minimal_pdf("Hello Drive")
    )
    text = read_as_text(_Service(_Files()), {"id": "f", "mimeType": "application/pdf"})
    assert "Hello Drive" in text


def test_read_as_text_rejects_binary_files():
    with pytest.raises(ValueError, match="Download File"):
        read_as_text(_Service(_Files()), {"id": "f", "mimeType": "image/png"})


def test_pdf_to_text_reads_every_page():
    assert "Hello Drive" in pdf_to_text(_minimal_pdf("Hello Drive"))


def test_fetch_file_bytes_exports_google_files_in_the_chosen_format():
    files = _Files()
    data, mime_type = fetch_file_bytes(
        _Service(files), {"id": "doc1", "mimeType": DOC}, ExportFormat.OFFICE
    )
    assert data == b"# Exported"
    assert mime_type.endswith("wordprocessingml.document")


def test_fetch_file_bytes_only_allows_pdf_for_other_google_types():
    with pytest.raises(ValueError, match="only be downloaded as PDF"):
        fetch_file_bytes(
            _Service(_Files()), {"id": "d", "mimeType": DRAWING}, ExportFormat.TEXT
        )
    _, mime_type = fetch_file_bytes(
        _Service(_Files()), {"id": "d", "mimeType": DRAWING}, ExportFormat.PDF
    )
    assert mime_type == "application/pdf"


def test_fetch_file_bytes_downloads_other_files_as_is(monkeypatch):
    monkeypatch.setattr(_drive_content, "download_file", lambda *a: b"PK\x03\x04")
    data, mime_type = fetch_file_bytes(
        _Service(_Files()), {"id": "z", "mimeType": "application/zip"}, ExportFormat.PDF
    )
    assert (data, mime_type) == (b"PK\x03\x04", "application/zip")


def test_download_file_refuses_files_over_the_limit():
    with pytest.raises(ValueError, match="limit"):
        _drive_content.download_file(
            _Service(_Files()),
            {"id": "big", "size": str(_drive_content.MAX_DOWNLOAD_BYTES + 1)},
        )


def test_move_replaces_every_current_parent():
    files = _Files(get_result={"parents": ["old1", "old2"]})
    GoogleDriveMoveFileBlock._move(_Service(files), "file1", "new")
    update = files.calls[-1]
    assert update[0] == "update"
    assert update[1]["addParents"] == "new"
    assert update[1]["removeParents"] == "old1,old2"


@pytest.mark.parametrize(
    "status, reason, expected",
    [
        (404, "File not found: x.", "couldn't find that file"),
        (403, "Request had insufficient authentication scopes.", "Reconnect Google"),
        (500, "Backend Error", "Google Drive API error 500"),
    ],
)
def test_drive_error_messages(status: int, reason: str, expected: str):
    content = f'{{"error": {{"code": {status}, "message": "{reason}"}}}}'.encode()
    exc = HttpError(httplib2.Response({"status": status}), content)
    assert expected in str(drive_error(exc, "block", "id"))


@pytest.mark.asyncio
async def test_create_file_content_rules():
    block = GoogleDriveCreateFileBlock()
    context = ExecutionContext()

    def make_input(**fields):
        return GoogleDriveCreateFileBlock.Input.model_validate(
            {"credentials": TEST_CREDENTIALS_INPUT, **fields}
        )

    empty = await block._content(make_input(name="notes.txt"), context)
    assert empty.data is None

    text = await block._content(
        make_input(name="notes.txt", text_content="Hi"), context
    )
    assert (text.data, text.content_type) == (b"Hi", "text/plain")

    csv = await block._content(make_input(name="rows.csv", text_content="a,b"), context)
    assert csv.content_type == "text/csv"

    with pytest.raises(BlockInputError, match="not both"):
        await block._content(
            make_input(text_content="x", file_to_upload="data:text/plain;base64,eA=="),
            context,
        )


@pytest.mark.asyncio
async def test_create_file_needs_a_name_or_an_upload():
    block = GoogleDriveCreateFileBlock()
    input_data = GoogleDriveCreateFileBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "text_content": "Hi"}
    )
    with pytest.raises(BlockInputError, match="name"):
        async for _ in block.run(
            input_data,
            credentials=TEST_CREDENTIALS,
            execution_context=ExecutionContext(),
        ):
            pass


def _minimal_pdf(text: str) -> bytes:
    """Build a one-page PDF with correct xref offsets."""
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % xref_at
    return bytes(out)
