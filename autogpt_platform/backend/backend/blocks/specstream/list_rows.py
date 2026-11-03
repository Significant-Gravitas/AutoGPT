"""Read the rows of a run, each value with its source URL and read time."""

from backend.blocks._platform_credentials import refuse_platform_credentials
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import ROWS_PER_PAGE, SpecStreamClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, specstream
from ._models import DatasetRow, RowsPage
from ._testdata import TEST_FIND_ID, TEST_ROWS_PAGE


class SpecStreamListRowsBlock(Block):
    """Read the matching rows of a run. Free, and safe to repeat."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = specstream.credentials_field(
            description="The SpecStream integration requires an API Key."
        )
        run_id: str = SchemaField(
            description="The run ID from Approve and Run", placeholder="fnd_..."
        )
        max_rows: int = SchemaField(
            description="The most rows to return in this call",
            default=100,
            ge=1,
            le=1000,
        )
        cursor: str = SchemaField(
            description="The next_cursor of an earlier call, to read the following rows",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        row: DatasetRow = SchemaField(
            description="One row: its values by field, and each value with its source URL and read time"
        )
        rows: list[DatasetRow] = SchemaField(description="All the rows of this call")
        row_count: int = SchemaField(description="How many rows this call returned")
        status: str = SchemaField(
            description="The status of the run. While it runs, the rows found so far are returned."
        )
        next_cursor: str = SchemaField(
            description="Pass this as cursor to read the following rows. Empty when there are no more."
        )

    def __init__(self):
        super().__init__(
            id="e138c5c8-e1b1-4b44-9609-e6a170389ae5",
            description=(
                "Reads the rows of a SpecStream dataset run. Source URL and read "
                "time on every value."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            display_name="SpecStream List Rows",
            input_schema=SpecStreamListRowsBlock.Input,
            output_schema=SpecStreamListRowsBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "run_id": TEST_FIND_ID,
                "max_rows": 1,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("row", DatasetRow),
                ("rows", list),
                ("row_count", 1),
                ("status", "completed"),
                ("next_cursor", "cur_2"),
            ],
            test_mock={"read_page": lambda *args, **kwargs: TEST_ROWS_PAGE},
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def read_page(
        credentials: APIKeyCredentials,
        run_id: str,
        cursor: str | None,
        limit: int,
    ) -> RowsPage:
        return await SpecStreamClient(credentials).list_rows(run_id, cursor, limit)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        refuse_platform_credentials(self, credentials)
        page = await self.read_page(
            credentials,
            input_data.run_id,
            input_data.cursor or None,
            min(ROWS_PER_PAGE, input_data.max_rows),
        )
        rows = [DatasetRow.from_api(row) for row in page.rows]
        while page.next_cursor and page.rows and len(rows) < input_data.max_rows:
            page = await self.read_page(
                credentials,
                input_data.run_id,
                page.next_cursor,
                min(ROWS_PER_PAGE, input_data.max_rows - len(rows)),
            )
            rows.extend(DatasetRow.from_api(row) for row in page.rows)

        for row in rows:
            yield "row", row
        yield "rows", rows
        yield "row_count", len(rows)
        yield "status", page.status
        yield "next_cursor", page.next_cursor or ""
