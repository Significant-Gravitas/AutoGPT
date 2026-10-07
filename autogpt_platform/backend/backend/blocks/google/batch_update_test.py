"""Unit tests for the raw batch-update blocks for Google Docs and Sheets."""

import httplib2
import pytest
from googleapiclient.errors import HttpError

from backend.blocks.google import docs_batch_update, sheets_batch_update
from backend.blocks.google._auth import TEST_CREDENTIALS
from backend.blocks.google._batch_update import batch_update_error
from backend.blocks.google.docs_batch_update import GoogleDocsBatchUpdateBlock
from backend.blocks.google.sheets_batch_update import GoogleSheetsBatchUpdateBlock
from backend.util.exceptions import BlockInputError

DOC = {
    "id": "1abc123def456",
    "name": "Proposal",
    "mimeType": "application/vnd.google-apps.document",
}
SHEET = {
    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    "name": "Pipeline",
    "mimeType": "application/vnd.google-apps.spreadsheet",
}


def _http_error(status: int, message: str) -> HttpError:
    content = f'{{"error": {{"code": {status}, "message": "{message}"}}}}'.encode()
    return HttpError(httplib2.Response({"status": status}), content)


@pytest.mark.parametrize(
    "status, reason, expected",
    [
        (
            400,
            "Invalid requests[0].insertText: Index 10 must be less than 5",
            "Index 10 must be less than 5",
        ),
        (404, "Requested entity was not found.", "Couldn't find that Docs file"),
        (403, "Request had insufficient authentication scopes.", "Reconnect Google"),
        (403, "The caller does not have permission", "can't edit this Docs file"),
        (500, "Internal error", "Google Docs API error 500"),
    ],
)
def test_batch_update_error_messages(status: int, reason: str, expected: str):
    error = batch_update_error(_http_error(status, reason), "Docs", "block", "id")
    assert expected in str(error)


async def _run(block, input_data):
    return [
        output
        async for output in block.run(
            block.input_schema.model_validate(input_data), credentials=TEST_CREDENTIALS
        )
    ]


@pytest.mark.asyncio
async def test_docs_batch_update_sends_write_control(monkeypatch):
    calls = []
    monkeypatch.setattr(docs_batch_update, "_build_docs_service", lambda creds: "svc")
    block = GoogleDocsBatchUpdateBlock()
    monkeypatch.setattr(
        block,
        "_batch_update",
        lambda service, doc_id, body: calls.append((service, doc_id, body)) or {},
    )
    outputs = await _run(
        block,
        {
            "document": DOC,
            "requests": [{"insertText": {"location": {"index": 1}, "text": "Hi"}}],
            "required_revision_id": "rev-7",
        },
    )
    assert calls[0][1] == DOC["id"]
    assert calls[0][2]["writeControl"] == {"requiredRevisionId": "rev-7"}
    assert outputs[0] == ("replies", [])
    assert outputs[1] == ("revision_id", "")


@pytest.mark.asyncio
async def test_docs_batch_update_rejects_empty_requests_and_non_docs():
    block = GoogleDocsBatchUpdateBlock()
    with pytest.raises(BlockInputError, match="at least one request"):
        await _run(block, {"document": DOC, "requests": []})
    with pytest.raises(BlockInputError, match="not a Google Doc"):
        await _run(
            block,
            {"document": {**DOC, "mimeType": "application/pdf"}, "requests": [{}]},
        )


@pytest.mark.asyncio
async def test_sheets_batch_update_passes_requests_through(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sheets_batch_update, "_build_sheets_service", lambda creds: "svc"
    )
    block = GoogleSheetsBatchUpdateBlock()
    monkeypatch.setattr(
        block,
        "_batch_update",
        lambda service, sheet_id, body: calls.append(body) or {"replies": [{}, {}]},
    )
    requests = [
        {"addSheet": {"properties": {"title": "Q4"}}},
        {"autoResizeDimensions": {}},
    ]
    outputs = await _run(block, {"spreadsheet": SHEET, "requests": requests})
    assert calls == [{"requests": requests}]
    assert outputs[0] == ("replies", [{}, {}])
    assert outputs[1][1].id == SHEET["id"]


@pytest.mark.asyncio
async def test_sheets_batch_update_rejects_csv_files():
    block = GoogleSheetsBatchUpdateBlock()
    with pytest.raises(BlockInputError, match="CSV"):
        await _run(
            block,
            {"spreadsheet": {**SHEET, "mimeType": "text/csv"}, "requests": [{}]},
        )
