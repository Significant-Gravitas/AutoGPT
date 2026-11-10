"""Google Sheets Set Layout: freeze, size, wrap, filter and band a sheet."""

import asyncio

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.google._drive import GoogleDriveFile, GoogleDriveFileField
from backend.data.model import SchemaField

from ._auth import TEST_CREDENTIALS, GoogleCredentials
from ._sheets_layout import (
    MAX_COLUMN_WIDTH_PX,
    MAX_COLUMN_WIDTHS,
    SheetColor,
    layout_requests,
    target_sheet_name,
)
from .sheets import (
    GOOGLE_SHEETS_DISABLED,
    _build_sheets_service,
    _validate_spreadsheet_file,
)


class GoogleSheetsSetLayoutBlock(Block):
    """Freeze, size, wrap, filter and band a sheet in one call."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet to lay out. Optional: defaults to the sheet the ranges name, else the first sheet",
            default="",
        )
        frozen_rows: int = SchemaField(
            description="Rows to keep in view at the top, e.g. 1 for a header row. -1 leaves it as is.",
            default=-1,
            ge=-1,
        )
        frozen_columns: int = SchemaField(
            description="Columns to keep in view on the left. -1 leaves it as is.",
            default=-1,
            ge=-1,
        )
        column_widths: list[int] = SchemaField(
            description=f"Widths in pixels for columns A, B, C and on, up to {MAX_COLUMN_WIDTH_PX}. 0 leaves that column as is.",
            default_factory=list,
            max_length=MAX_COLUMN_WIDTHS,
        )
        wrap_text_range: str = SchemaField(
            description="Range whose text wraps inside the cell, aligned to the top, e.g. 'A1:L1000'",
            default="",
        )
        filter_range: str = SchemaField(
            description="Range to add filter buttons to, header row first, e.g. 'A1:L1000'",
            default="",
        )
        banding_range: str = SchemaField(
            description="Range to give alternating row colours, header row first, e.g. 'A1:L1000'. Replaces alternating colours that overlap it.",
            default="",
        )
        header_color: SheetColor = SchemaField(
            description="Header row colour for the alternating colours",
            default=SheetColor(red=0.13, green=0.2, blue=0.33),
            advanced=True,
        )
        band_color: SheetColor = SchemaField(
            description="Colour of every second row",
            default=SheetColor(red=0.95, green=0.96, blue=0.98),
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of the operation")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="a64d1c3c-915b-4064-a5c6-44dc5cba5c12",
            description="Lay out a Google Sheet: freeze header rows or columns, set column widths, wrap text, add filter buttons and alternating row colours.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsSetLayoutBlock.Input,
            output_schema=GoogleSheetsSetLayoutBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "frozen_rows": 1,
                "column_widths": [180, 320],
                "filter_range": "A1:B100",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "requests": 4}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_set_layout": lambda *args, **kwargs: {"success": True, "requests": 4},
            },
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._set_layout, service, input_data.spreadsheet.id, input_data
            )
            yield "result", result
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to lay out sheet: {e!s}"

    def _set_layout(self, service, spreadsheet_id: str, input_data: Input) -> dict:
        wanted = target_sheet_name(
            input_data.sheet_name,
            [
                input_data.wrap_text_range,
                input_data.filter_range,
                input_data.banding_range,
            ],
        )
        # One read gives the sheet ids, titles and bandings.
        meta = (
            service.spreadsheets()
            .get(
                spreadsheetId=spreadsheet_id,
                fields="sheets(properties(sheetId,title),bandedRanges(bandedRangeId,range))",
            )
            .execute()
        )
        sheets = meta.get("sheets", [])
        if not sheets:
            raise ValueError("The spreadsheet has no sheets")
        sheet = (
            sheets[0]
            if wanted is None
            else next(
                (s for s in sheets if s.get("properties", {}).get("title") == wanted),
                None,
            )
        )
        if sheet is None:
            raise ValueError(f"Sheet '{wanted}' not found")

        requests = layout_requests(
            sheet["properties"]["sheetId"],
            frozen_rows=input_data.frozen_rows,
            frozen_columns=input_data.frozen_columns,
            column_widths=input_data.column_widths,
            wrap_text_range=input_data.wrap_text_range,
            filter_range=input_data.filter_range,
            banding_range=input_data.banding_range,
            header_color=input_data.header_color.model_dump(),
            band_color=input_data.band_color.model_dump(),
            existing_bandings=sheet.get("bandedRanges", []),
        )
        if not requests:
            raise ValueError("Nothing to change: set at least one layout option")

        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": requests}
        ).execute()
        return {"success": True, "requests": len(requests)}
