import asyncio

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

from ._analytics_api import (
    ANALYTICS_READONLY_SCOPE,
    DATA_API,
    PROPERTY_ID_DESCRIPTION,
    analytics_error,
    build_data_service,
    property_resource_name,
)
from ._analytics_report import (
    FILTERS_DESCRIPTION,
    LIMIT_DESCRIPTION,
    MAX_ROWS,
    ORDER_BY_DESCRIPTION,
    ROW_COUNT_DESCRIPTION,
    ROWS_DESCRIPTION,
    TOTALS_DESCRIPTION,
    GoogleAnalyticsDimensionFilter,
    parse_report,
    report_body,
    report_date,
)
from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)

_TEST_REPORT = {
    "dimensionHeaders": [{"name": "sessionSource"}],
    "metricHeaders": [
        {"name": "activeUsers", "type": "TYPE_INTEGER"},
        {"name": "sessions", "type": "TYPE_INTEGER"},
        {"name": "engagementRate", "type": "TYPE_FLOAT"},
    ],
    "rows": [
        {
            "dimensionValues": [{"value": "google"}],
            "metricValues": [{"value": "1204"}, {"value": "1530"}, {"value": "0.64"}],
        },
        {
            "dimensionValues": [{"value": "(direct)"}],
            "metricValues": [{"value": "651"}, {"value": "702"}, {"value": "0.52"}],
        },
    ],
    "totals": [
        {
            "dimensionValues": [{"value": "RESERVED_TOTAL"}],
            "metricValues": [{"value": "1810"}, {"value": "2232"}, {"value": "0.6"}],
        }
    ],
    "rowCount": 2,
    "metadata": {"currencyCode": "USD", "timeZone": "America/New_York"},
    "kind": "analyticsData#runReport",
}
_TEST_ROWS = [
    {
        "sessionSource": "google",
        "activeUsers": 1204,
        "sessions": 1530,
        "engagementRate": 0.64,
    },
    {
        "sessionSource": "(direct)",
        "activeUsers": 651,
        "sessions": 702,
        "engagementRate": 0.52,
    },
]


class GoogleAnalyticsRunReportBlock(Block):
    """Run a Google Analytics 4 report over a date range."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [ANALYTICS_READONLY_SCOPE]
        )
        property_id: str = SchemaField(
            description=PROPERTY_ID_DESCRIPTION, placeholder="123456789"
        )
        metrics: list[str] = SchemaField(
            description=(
                "Metric API names, 1 to 10, such as activeUsers, sessions, "
                "screenPageViews, keyEvents, engagementRate or totalRevenue. Custom "
                "metrics look like customEvent:name."
            ),
            default=["activeUsers", "sessions"],
            advanced=False,
        )
        dimensions: list[str] = SchemaField(
            description=(
                "Dimension API names to break the numbers down by, up to 9, such as "
                "date, country, sessionSource, sessionDefaultChannelGroup or "
                "pagePath. Leave empty for a single row of totals."
            ),
            default_factory=list,
            advanced=False,
        )
        start_date: str = SchemaField(
            description=(
                "First day of the report: YYYY-MM-DD, today, yesterday or NdaysAgo "
                "(such as 28daysAgo). Relative dates follow the property's time zone."
            ),
            default="28daysAgo",
            advanced=False,
        )
        end_date: str = SchemaField(
            description=(
                "Last day of the report, included: YYYY-MM-DD, today, yesterday or "
                "NdaysAgo"
            ),
            default="yesterday",
            advanced=False,
        )
        dimension_filters: list[GoogleAnalyticsDimensionFilter] = SchemaField(
            description=FILTERS_DESCRIPTION, default_factory=list, advanced=True
        )
        order_by: str = SchemaField(
            description=ORDER_BY_DESCRIPTION, default="", advanced=False
        )
        limit: int = SchemaField(
            description=LIMIT_DESCRIPTION, default=100, ge=1, le=MAX_ROWS, advanced=True
        )

    class Output(BlockSchemaOutput):
        rows: list[dict[str, str | int | float]] = SchemaField(
            description=ROWS_DESCRIPTION
        )
        row: dict[str, str | int | float] = SchemaField(description="Each row")
        totals: dict[str, int | float] = SchemaField(description=TOTALS_DESCRIPTION)
        row_count: int = SchemaField(description=ROW_COUNT_DESCRIPTION)
        time_zone: str = SchemaField(
            description="The property's time zone, which the report's dates are in"
        )
        currency_code: str = SchemaField(
            description="The currency of money metrics such as totalRevenue"
        )

    def __init__(self):
        super().__init__(
            id="09455cb7-1fb8-433b-8d0e-e0d89f1e0ca3",
            description=(
                "Run a Google Analytics 4 report of metrics such as users, sessions or "
                "page views over a date range, split by dimensions such as date, page "
                "or traffic source. It can filter on dimension values, sort and limit "
                "the rows, and returns totals too. Google Analytics List Dimensions "
                "and Metrics gives a property's custom names."
            ),
            categories={BlockCategory.DATA, BlockCategory.MARKETING},
            input_schema=GoogleAnalyticsRunReportBlock.Input,
            output_schema=GoogleAnalyticsRunReportBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "property_id": "123456789",
                "metrics": ["activeUsers", "sessions", "engagementRate"],
                "dimensions": ["sessionSource"],
                "order_by": "-sessions",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("rows", _TEST_ROWS),
                ("row", _TEST_ROWS[0]),
                ("row", _TEST_ROWS[1]),
                (
                    "totals",
                    {"activeUsers": 1810, "sessions": 2232, "engagementRate": 0.6},
                ),
                ("row_count", 2),
                ("time_zone", "America/New_York"),
                ("currency_code", "USD"),
            ],
            test_mock={"_run_report": lambda *args, **kwargs: _TEST_REPORT},
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        name = property_resource_name(input_data.property_id, self.name, self.id)
        start = report_date(input_data.start_date, "start_date", self.name, self.id)
        end = report_date(input_data.end_date, "end_date", self.name, self.id)
        body = report_body(
            metrics=input_data.metrics,
            dimensions=input_data.dimensions,
            filters=input_data.dimension_filters,
            order_by=input_data.order_by,
            limit=input_data.limit,
            block_name=self.name,
            block_id=self.id,
        )
        body["dateRanges"] = [{"startDate": start, "endDate": end}]
        service = build_data_service(credentials)
        try:
            response = await asyncio.to_thread(self._run_report, service, name, body)
        except HttpError as e:
            raise analytics_error(
                e, self.name, self.id, api=DATA_API, property_name=name
            ) from e

        report = parse_report(response)
        yield "rows", report.rows
        for row in report.rows:
            yield "row", row
        yield "totals", report.totals
        yield "row_count", report.row_count
        if report.time_zone:
            yield "time_zone", report.time_zone
        if report.currency_code:
            yield "currency_code", report.currency_code

    @staticmethod
    def _run_report(service, property_name: str, body: dict) -> dict:
        return (
            service.properties().runReport(property=property_name, body=body).execute()
        )
