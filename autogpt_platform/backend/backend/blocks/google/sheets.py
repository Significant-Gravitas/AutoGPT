import asyncio
from enum import Enum

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.settings import AppEnvironment, Settings

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)


class SheetOperation(str, Enum):
    CREATE = "create"
    DELETE = "delete"
    COPY = "copy"


class BatchOperationType(str, Enum):
    UPDATE = "update"
    CLEAR = "clear"


class BatchOperation(BlockSchema):
    type: BatchOperationType = SchemaField(
        description="The type of operation to perform"
    )
    range: str = SchemaField(description="The A1 notation range for the operation")
    values: list[list[str]] = SchemaField(
        description="The values to update (only for UPDATE operations)",
        default=[],
    )


class GoogleSheetsReadBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the spreadsheet to read from",
        )
        range: str = SchemaField(
            description="The A1 notation of the range to read",
        )

    class Output(BlockSchema):
        result: list[list[str]] = SchemaField(
            description="The data read from the spreadsheet",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        settings = Settings()
        super().__init__(
            id="5724e902-3635-47e9-a108-aaa0263a4988",
            description="This block reads data from a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsReadBlock.Input,
            output_schema=GoogleSheetsReadBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED
            or settings.config.app_env == AppEnvironment.PRODUCTION,
            test_input={
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "range": "Sheet1!A1:B2",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    [
                        ["Name", "Score"],
                        ["Alice", "85"],
                    ],
                ),
            ],
            test_mock={
                "_read_sheet": lambda *args, **kwargs: [
                    ["Name", "Score"],
                    ["Alice", "85"],
                ],
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        data = await asyncio.to_thread(
            self._read_sheet, service, input_data.spreadsheet_id, input_data.range
        )
        yield "result", data

    @staticmethod
    def _build_service(credentials: GoogleCredentials, **kwargs):
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
            client_id=Settings().secrets.google_client_id,
            client_secret=Settings().secrets.google_client_secret,
            scopes=credentials.scopes,
        )
        return build("sheets", "v4", credentials=creds)

    def _read_sheet(self, service, spreadsheet_id: str, range: str) -> list[list[str]]:
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range).execute()
        return result.get("values", [])


class GoogleSheetsWriteBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/spreadsheets"]
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the spreadsheet to write to",
        )
        range: str = SchemaField(
            description="The A1 notation of the range to write",
        )
        values: list[list[str]] = SchemaField(
            description="The data to write to the spreadsheet",
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
            description="The result of the write operation",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="d9291e87-301d-47a8-91fe-907fb55460e5",
            description="This block writes data to a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsWriteBlock.Input,
            output_schema=GoogleSheetsWriteBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "range": "Sheet1!A1:B2",
                "values": [
                    ["Name", "Score"],
                    ["Bob", "90"],
                ],
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    {"updatedCells": 4, "updatedColumns": 2, "updatedRows": 2},
                ),
            ],
            test_mock={
                "_write_sheet": lambda *args, **kwargs: {
                    "updatedCells": 4,
                    "updatedColumns": 2,
                    "updatedRows": 2,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GoogleSheetsReadBlock._build_service(credentials, **kwargs)
        result = await asyncio.to_thread(
            self._write_sheet,
            service,
            input_data.spreadsheet_id,
            input_data.range,
            input_data.values,
        )
        yield "result", result

    def _write_sheet(
        self, service, spreadsheet_id: str, range: str, values: list[list[str]]
    ) -> dict:
        body = {"values": values}
        result = (
            service.spreadsheets()
            .values()
            .update(
                spreadsheetId=spreadsheet_id,
                range=range,
                valueInputOption="USER_ENTERED",
                body=body,
            )
            .execute()
        )
        return result


class GoogleSheetsAppendBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/spreadsheets"]
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the spreadsheet to append to",
        )
        sheet_name: str = SchemaField(
            description="The name of the sheet to append to",
            default="Sheet1",
        )
        values: list[list[str]] = SchemaField(
            description="The data to append to the spreadsheet",
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
            description="The result of the append operation",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="531d50c0-d6b9-4cf9-a013-7bf783d313c7",
            description="This block appends data to the end of a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsAppendBlock.Input,
            output_schema=GoogleSheetsAppendBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "sheet_name": "Sheet1",
                "values": [["Charlie", "95"]],
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    {"updatedCells": 2, "updatedColumns": 2, "updatedRows": 1},
                ),
            ],
            test_mock={
                "_append_sheet": lambda *args, **kwargs: {
                    "updatedCells": 2,
                    "updatedColumns": 2,
                    "updatedRows": 1,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GoogleSheetsReadBlock._build_service(credentials, **kwargs)
        result = await asyncio.to_thread(
            self._append_sheet,
            service,
            input_data.spreadsheet_id,
            input_data.sheet_name,
            input_data.values,
        )
        yield "result", result

    def _append_sheet(
        self, service, spreadsheet_id: str, sheet_name: str, values: list[list[str]]
    ) -> dict:
        body = {"values": values}
        result = (
            service.spreadsheets()
            .values()
            .append(
                spreadsheetId=spreadsheet_id,
                range=f"{sheet_name}!A:A",
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body=body,
            )
            .execute()
        )
        return result


class GoogleSheetsClearBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/spreadsheets"]
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the spreadsheet to clear",
        )
        range: str = SchemaField(
            description="The A1 notation of the range to clear",
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
            description="The result of the clear operation",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="84938266-0fc7-46e5-9369-adb0f6ae8015",
            description="This block clears data from a specified range in a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsClearBlock.Input,
            output_schema=GoogleSheetsClearBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "range": "Sheet1!A1:B2",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"clearedRange": "Sheet1!A1:B2"}),
            ],
            test_mock={
                "_clear_range": lambda *args, **kwargs: {
                    "clearedRange": "Sheet1!A1:B2"
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GoogleSheetsReadBlock._build_service(credentials, **kwargs)
        result = await asyncio.to_thread(
            self._clear_range,
            service,
            input_data.spreadsheet_id,
            input_data.range,
        )
        yield "result", result

    def _clear_range(self, service, spreadsheet_id: str, range: str) -> dict:
        result = (
            service.spreadsheets()
            .values()
            .clear(spreadsheetId=spreadsheet_id, range=range)
            .execute()
        )
        return result


class GoogleSheetsMetadataBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the spreadsheet to get metadata for",
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
            description="The metadata of the spreadsheet including sheets info",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="6a0be6ee-7a0d-4c92-819b-500630846ad0",
            description="This block retrieves metadata about a Google Sheets spreadsheet including sheet names and properties.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsMetadataBlock.Input,
            output_schema=GoogleSheetsMetadataBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    {
                        "title": "Test Spreadsheet",
                        "sheets": [{"title": "Sheet1", "sheetId": 0}],
                    },
                ),
            ],
            test_mock={
                "_get_metadata": lambda *args, **kwargs: {
                    "title": "Test Spreadsheet",
                    "sheets": [{"title": "Sheet1", "sheetId": 0}],
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GoogleSheetsReadBlock._build_service(credentials, **kwargs)
        result = await asyncio.to_thread(
            self._get_metadata,
            service,
            input_data.spreadsheet_id,
        )
        yield "result", result

    def _get_metadata(self, service, spreadsheet_id: str) -> dict:
        result = (
            service.spreadsheets()
            .get(spreadsheetId=spreadsheet_id, includeGridData=False)
            .execute()
        )
        return {
            "title": result.get("properties", {}).get("title"),
            "sheets": [
                {
                    "title": sheet.get("properties", {}).get("title"),
                    "sheetId": sheet.get("properties", {}).get("sheetId"),
                    "gridProperties": sheet.get("properties", {}).get(
                        "gridProperties", {}
                    ),
                }
                for sheet in result.get("sheets", [])
            ],
        }


class GoogleSheetsManageSheetBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/spreadsheets"]
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the spreadsheet to manage",
        )
        operation: SheetOperation = SchemaField(
            description="The operation to perform",
        )
        sheet_name: str = SchemaField(
            description="The name of the sheet (for create/delete operations)",
            default="",
        )
        source_sheet_id: int = SchemaField(
            description="The ID of the sheet to copy (for copy operations)",
            default=0,
        )
        destination_sheet_name: str = SchemaField(
            description="The name for the copied sheet (for copy operations)",
            default="",
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
            description="The result of the sheet management operation",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="7940189d-b137-4ef1-aa18-3dd9a5bde9f3",
            description="This block manages sheets in a Google Sheets spreadsheet (create, delete, copy).",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsManageSheetBlock.Input,
            output_schema=GoogleSheetsManageSheetBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "operation": SheetOperation.CREATE,
                "sheet_name": "NewSheet",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "sheetId": 123}),
            ],
            test_mock={
                "_manage_sheet": lambda *args, **kwargs: {
                    "success": True,
                    "sheetId": 123,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GoogleSheetsReadBlock._build_service(credentials, **kwargs)
        result = await asyncio.to_thread(
            self._manage_sheet,
            service,
            input_data.spreadsheet_id,
            input_data.operation,
            input_data.sheet_name,
            input_data.source_sheet_id,
            input_data.destination_sheet_name,
        )
        yield "result", result

    def _manage_sheet(
        self,
        service,
        spreadsheet_id: str,
        operation: SheetOperation,
        sheet_name: str,
        source_sheet_id: int,
        destination_sheet_name: str,
    ) -> dict:
        requests = []

        if operation == SheetOperation.CREATE:
            requests.append({"addSheet": {"properties": {"title": sheet_name}}})
        elif operation == SheetOperation.DELETE:
            # Find sheet ID by name first
            metadata = (
                service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            )
            sheet_id = None
            for sheet in metadata.get("sheets", []):
                if sheet.get("properties", {}).get("title") == sheet_name:
                    sheet_id = sheet.get("properties", {}).get("sheetId")
                    break

            if sheet_id is not None:
                requests.append({"deleteSheet": {"sheetId": sheet_id}})
            else:
                return {"error": f"Sheet '{sheet_name}' not found"}

        elif operation == SheetOperation.COPY:
            requests.append(
                {
                    "duplicateSheet": {
                        "sourceSheetId": source_sheet_id,
                        "newSheetName": destination_sheet_name,
                    }
                }
            )
        else:
            return {"error": f"Unknown operation: {operation}"}

        body = {"requests": requests}
        result = (
            service.spreadsheets()
            .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
            .execute()
        )

        return {"success": True, "result": result}


class GoogleSheetsBatchOperationsBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/spreadsheets"]
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the spreadsheet to perform batch operations on",
        )
        operations: list[BatchOperation] = SchemaField(
            description="List of operations to perform",
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
            description="The result of the batch operations",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="a4078584-6fe5-46e0-997e-d5126cdd112a",
            description="This block performs multiple operations on a Google Sheets spreadsheet in a single batch request.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsBatchOperationsBlock.Input,
            output_schema=GoogleSheetsBatchOperationsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "operations": [
                    {
                        "type": BatchOperationType.UPDATE,
                        "range": "A1:B1",
                        "values": [["Header1", "Header2"]],
                    },
                    {
                        "type": BatchOperationType.UPDATE,
                        "range": "A2:B2",
                        "values": [["Data1", "Data2"]],
                    },
                ],
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"totalUpdatedCells": 4, "replies": []}),
            ],
            test_mock={
                "_batch_operations": lambda *args, **kwargs: {
                    "totalUpdatedCells": 4,
                    "replies": [],
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GoogleSheetsReadBlock._build_service(credentials, **kwargs)
        result = await asyncio.to_thread(
            self._batch_operations,
            service,
            input_data.spreadsheet_id,
            input_data.operations,
        )
        yield "result", result

    def _batch_operations(
        self, service, spreadsheet_id: str, operations: list[BatchOperation]
    ) -> dict:
        update_data = []
        clear_ranges = []

        for op in operations:
            if op.type == BatchOperationType.UPDATE:
                update_data.append(
                    {
                        "range": op.range,
                        "values": op.values,
                    }
                )
            elif op.type == BatchOperationType.CLEAR:
                clear_ranges.append(op.range)

        results = {}

        # Perform updates if any
        if update_data:
            update_body = {
                "valueInputOption": "USER_ENTERED",
                "data": update_data,
            }
            update_result = (
                service.spreadsheets()
                .values()
                .batchUpdate(spreadsheetId=spreadsheet_id, body=update_body)
                .execute()
            )
            results["updateResult"] = update_result

        # Perform clears if any
        if clear_ranges:
            clear_body = {"ranges": clear_ranges}
            clear_result = (
                service.spreadsheets()
                .values()
                .batchClear(spreadsheetId=spreadsheet_id, body=clear_body)
                .execute()
            )
            results["clearResult"] = clear_result

        return results


class GoogleSheetsFindReplaceBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/spreadsheets"]
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the spreadsheet to perform find/replace on",
        )
        find_text: str = SchemaField(
            description="The text to find",
        )
        replace_text: str = SchemaField(
            description="The text to replace with",
        )
        sheet_id: int = SchemaField(
            description="The ID of the specific sheet to search (optional, searches all sheets if not provided)",
            default=-1,
        )
        match_case: bool = SchemaField(
            description="Whether to match case",
            default=False,
        )
        match_entire_cell: bool = SchemaField(
            description="Whether to match entire cell",
            default=False,
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
            description="The result of the find/replace operation including number of replacements",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="accca760-8174-4656-b55e-5f0e82fee986",
            description="This block finds and replaces text in a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsFindReplaceBlock.Input,
            output_schema=GoogleSheetsFindReplaceBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "find_text": "old_value",
                "replace_text": "new_value",
                "match_case": False,
                "match_entire_cell": False,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"occurrencesChanged": 5}),
            ],
            test_mock={
                "_find_replace": lambda *args, **kwargs: {"occurrencesChanged": 5},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GoogleSheetsReadBlock._build_service(credentials, **kwargs)
        result = await asyncio.to_thread(
            self._find_replace,
            service,
            input_data.spreadsheet_id,
            input_data.find_text,
            input_data.replace_text,
            input_data.sheet_id,
            input_data.match_case,
            input_data.match_entire_cell,
        )
        yield "result", result

    def _find_replace(
        self,
        service,
        spreadsheet_id: str,
        find_text: str,
        replace_text: str,
        sheet_id: int,
        match_case: bool,
        match_entire_cell: bool,
    ) -> dict:
        find_replace_request = {
            "find": find_text,
            "replacement": replace_text,
            "matchCase": match_case,
            "matchEntireCell": match_entire_cell,
        }

        if sheet_id >= 0:
            find_replace_request["sheetId"] = sheet_id

        requests = [{"findReplace": find_replace_request}]
        body = {"requests": requests}

        result = (
            service.spreadsheets()
            .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
            .execute()
        )

        return result


class GoogleSheetsFormatBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/spreadsheets"]
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the spreadsheet to format",
        )
        range: str = SchemaField(
            description="The A1 notation of the range to format",
        )
        background_color: dict = SchemaField(
            description="Background color as RGB dict (e.g., {'red': 1.0, 'green': 0.0, 'blue': 0.0})",
            default={},
        )
        text_color: dict = SchemaField(
            description="Text color as RGB dict (e.g., {'red': 0.0, 'green': 0.0, 'blue': 1.0})",
            default={},
        )
        bold: bool = SchemaField(
            description="Whether to make text bold",
            default=False,
        )
        italic: bool = SchemaField(
            description="Whether to make text italic",
            default=False,
        )
        font_size: int = SchemaField(
            description="Font size in points",
            default=10,
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
            description="The result of the formatting operation",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="270f2384-8089-4b5b-b2e3-fe2ea3d87c02",
            description="This block applies formatting to cells in a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsFormatBlock.Input,
            output_schema=GoogleSheetsFormatBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "range": "A1:B2",
                "background_color": {"red": 1.0, "green": 0.9, "blue": 0.9},
                "text_color": {"red": 0.0, "green": 0.0, "blue": 0.0},
                "bold": True,
                "italic": False,
                "font_size": 12,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
            ],
            test_mock={
                "_format_cells": lambda *args, **kwargs: {"success": True},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GoogleSheetsReadBlock._build_service(credentials, **kwargs)
        result = await asyncio.to_thread(
            self._format_cells,
            service,
            input_data.spreadsheet_id,
            input_data.range,
            input_data.background_color,
            input_data.text_color,
            input_data.bold,
            input_data.italic,
            input_data.font_size,
        )
        yield "result", result

    def _format_cells(
        self,
        service,
        spreadsheet_id: str,
        range: str,
        background_color: dict,
        text_color: dict,
        bold: bool,
        italic: bool,
        font_size: int,
    ) -> dict:
        # Parse the range to get sheet name and grid range
        if "!" in range:
            sheet_name, cell_range = range.split("!")
        else:
            sheet_name = "Sheet1"
            cell_range = range

        # Get sheet metadata to find sheet ID
        metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_id = None
        for sheet in metadata.get("sheets", []):
            if sheet.get("properties", {}).get("title") == sheet_name:
                sheet_id = sheet.get("properties", {}).get("sheetId")
                break

        if sheet_id is None:
            return {"error": f"Sheet '{sheet_name}' not found"}

        # Parse cell range (simplified for A1:B2 format)
        try:
            start_cell, end_cell = cell_range.split(":")
            start_col = ord(start_cell[0]) - ord("A")
            start_row = int(start_cell[1:]) - 1
            end_col = ord(end_cell[0]) - ord("A") + 1
            end_row = int(end_cell[1:])
        except (ValueError, IndexError):
            return {"error": f"Invalid range format: {cell_range}"}

        # Build format request
        format_request = {"userEnteredFormat": {}}

        if background_color:
            format_request["userEnteredFormat"]["backgroundColor"] = background_color

        if text_color or bold or italic or font_size != 10:
            text_format = {}
            if text_color:
                text_format["foregroundColor"] = text_color
            if bold:
                text_format["bold"] = True
            if italic:
                text_format["italic"] = True
            if font_size != 10:
                text_format["fontSize"] = font_size
            format_request["userEnteredFormat"]["textFormat"] = text_format

        requests = [
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": start_row,
                        "endRowIndex": end_row,
                        "startColumnIndex": start_col,
                        "endColumnIndex": end_col,
                    },
                    "cell": format_request,
                    "fields": "userEnteredFormat(backgroundColor,textFormat)",
                }
            }
        ]

        body = {"requests": requests}
        result = (
            service.spreadsheets()
            .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
            .execute()
        )

        return {"success": True, "result": result}


class GoogleSheetsCreateSpreadsheetBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/spreadsheets"]
        )
        title: str = SchemaField(
            description="The title of the new spreadsheet",
        )
        sheet_names: list[str] = SchemaField(
            description="List of sheet names to create (optional, defaults to single 'Sheet1')",
            default=["Sheet1"],
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
            description="The result containing spreadsheet ID and URL",
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the created spreadsheet",
        )
        spreadsheet_url: str = SchemaField(
            description="The URL of the created spreadsheet",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="c8d4c0d3-c76e-4c2a-8c66-4119817ea3d1",
            description="This block creates a new Google Sheets spreadsheet with specified sheets.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsCreateSpreadsheetBlock.Input,
            output_schema=GoogleSheetsCreateSpreadsheetBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "title": "Test Spreadsheet",
                "sheet_names": ["Sheet1", "Data", "Summary"],
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("spreadsheet_id", "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"),
                (
                    "spreadsheet_url",
                    "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                ),
                ("result", {"success": True}),
            ],
            test_mock={
                "_create_spreadsheet": lambda *args, **kwargs: {
                    "spreadsheetId": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "spreadsheetUrl": "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GoogleSheetsReadBlock._build_service(credentials, **kwargs)
        result = await asyncio.to_thread(
            self._create_spreadsheet,
            service,
            input_data.title,
            input_data.sheet_names,
        )

        if "error" in result:
            yield "error", result["error"]
        else:
            yield "spreadsheet_id", result["spreadsheetId"]
            yield "spreadsheet_url", result["spreadsheetUrl"]
            yield "result", {"success": True}

    def _create_spreadsheet(self, service, title: str, sheet_names: list[str]) -> dict:
        try:
            # Create the initial spreadsheet
            spreadsheet_body = {
                "properties": {"title": title},
                "sheets": [
                    {
                        "properties": {
                            "title": sheet_names[0] if sheet_names else "Sheet1"
                        }
                    }
                ],
            }

            result = service.spreadsheets().create(body=spreadsheet_body).execute()
            spreadsheet_id = result["spreadsheetId"]
            spreadsheet_url = result["spreadsheetUrl"]

            # Add additional sheets if requested
            if len(sheet_names) > 1:
                requests = []
                for sheet_name in sheet_names[1:]:
                    requests.append({"addSheet": {"properties": {"title": sheet_name}}})

                if requests:
                    batch_body = {"requests": requests}
                    service.spreadsheets().batchUpdate(
                        spreadsheetId=spreadsheet_id, body=batch_body
                    ).execute()

            return {
                "spreadsheetId": spreadsheet_id,
                "spreadsheetUrl": spreadsheet_url,
            }
        except Exception as e:
            return {"error": str(e)}
