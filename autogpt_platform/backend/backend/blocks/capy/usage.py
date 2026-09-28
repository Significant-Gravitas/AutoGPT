"""Block that reports what a Capy organization has spent."""

from typing import Any

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

from ._api import CapyClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, capy_credentials_field
from ._testdata import TEST_USAGE_REPORT


class CapyGetUsageBlock(Block):
    """Report Capy spend in dollars for a date range."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        start: str = SchemaField(
            description=(
                "Start of the range as an ISO date or timestamp. Empty means the "
                "start of the current month."
            ),
            default="",
        )
        end: str = SchemaField(
            description="End of the range as an ISO date or timestamp. Empty means now.",
            default="",
        )

    class Output(BlockSchemaOutput):
        total_dollars: float = SchemaField(description="Total spend in the range")
        report: dict[str, Any] = SchemaField(
            description=(
                "The full report: totals by kind (LLM, image, VM), token counts, "
                "and breakdowns by member, model and image"
            )
        )

    def __init__(self):
        super().__init__(
            id="4ebe2817-9628-432c-a520-034efd2e95bd",
            description=(
                "Reports how much your Capy organization spent in a date range, "
                "broken down by member, model and kind of usage."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyGetUsageBlock.Input,
            output_schema=CapyGetUsageBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("total_dollars", 1.75),
                ("report", TEST_USAGE_REPORT),
            ],
            test_mock={"get_usage": lambda *args, **kwargs: TEST_USAGE_REPORT},
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def get_usage(
        credentials: APIKeyCredentials, start: str, end: str
    ) -> dict[str, Any]:
        return await CapyClient(credentials).get_usage(start, end)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        report = await self.get_usage(credentials, input_data.start, input_data.end)
        total = (report.get("totals") or {}).get("totalDollars") or 0
        yield "total_dollars", float(total)
        yield "report", report
