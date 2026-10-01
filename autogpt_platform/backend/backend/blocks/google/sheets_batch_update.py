import asyncio
from typing import Any

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.exceptions import BlockInputError

from ._auth import GOOGLE_OAUTH_IS_CONFIGURED, TEST_CREDENTIALS, GoogleCredentials
from ._batch_update import batch_update_error
from ._drive import GoogleDriveFile, GoogleDriveFileField
from .sheets import _build_sheets_service, _validate_spreadsheet_file

_TEST_SPREADSHEET = {
    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    "name": "Pipeline",
    "mimeType": "application/vnd.google-apps.spreadsheet",
}
_TEST_REQUESTS = [
    {
        "mergeCells": {
            "range": {"sheetId": 0, "startRowIndex": 0, "endRowIndex": 1},
            "mergeType": "MERGE_ALL",
        }
    }
]


def spreadsheet_output(file: GoogleDriveFile) -> GoogleDriveFile:
    """The spreadsheet as a chainable GoogleDriveFile."""
    return GoogleDriveFile(
        id=file.id,
        name=file.name,
        mimeType="application/vnd.google-apps.spreadsheet",
        url=f"https://docs.google.com/spreadsheets/d/{file.id}/edit",
        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
        isFolder=False,
        _credentials_id=file.credentials_id,
    )


class GoogleSheetsBatchUpdateBlock(Block):
    """Apply a raw list of Google Sheets API requests to a spreadsheet."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="The Google Sheets spreadsheet to update",
            allowed_views=["SPREADSHEETS"],
        )
        requests: list[dict[str, Any]] = SchemaField(
            description=(
                "Google Sheets API batchUpdate requests, applied in order and all "
                'or nothing, e.g. [{"addChart": {...}}] or '
                '[{"repeatCell": {...}}]. Request types: '
                "https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/request"
            ),
        )

    class Output(BlockSchemaOutput):
        replies: list[dict[str, Any]] = SchemaField(
            description="One reply per request, in order (empty for requests that return nothing)"
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet, for chaining"
        )

    def __init__(self):
        super().__init__(
            id="e7919065-6ecb-4411-bbd5-94623bf69a68",
            description=(
                "Apply any Google Sheets API batchUpdate requests to a spreadsheet "
                "in one all-or-nothing call: charts, conditional formatting, "
                "merges, filters and anything the other Google Sheets blocks "
                "don't cover."
            ),
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsBatchUpdateBlock.Input,
            output_schema=GoogleSheetsBatchUpdateBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"spreadsheet": _TEST_SPREADSHEET, "requests": _TEST_REQUESTS},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("replies", [{}]),
                (
                    "spreadsheet",
                    spreadsheet_output(
                        GoogleDriveFile.model_validate(_TEST_SPREADSHEET)
                    ),
                ),
            ],
            test_mock={"_batch_update": lambda *args, **kwargs: {"replies": [{}]}},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        spreadsheet = input_data.spreadsheet
        problem = (
            _validate_spreadsheet_file(spreadsheet)
            if spreadsheet
            else "Pick a Google Sheets spreadsheet."
        )
        if problem:
            raise BlockInputError(
                message=problem, block_name=self.name, block_id=self.id
            )
        if not input_data.requests:
            raise BlockInputError(
                message="Add at least one request.",
                block_name=self.name,
                block_id=self.id,
            )
        service = _build_sheets_service(credentials)
        try:
            result = await asyncio.to_thread(
                self._batch_update,
                service,
                spreadsheet.id,
                {"requests": input_data.requests},
            )
        except HttpError as e:
            raise batch_update_error(e, "Sheets", self.name, self.id) from e
        yield "replies", result.get("replies", [])
        yield "spreadsheet", spreadsheet_output(spreadsheet)

    @staticmethod
    def _batch_update(service, spreadsheet_id: str, body: dict) -> dict:
        return (
            service.spreadsheets()
            .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
            .execute()
        )
