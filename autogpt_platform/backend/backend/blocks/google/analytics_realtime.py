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
)
from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)

# How far back a standard property's realtime report can start. Google
# Analytics 360 properties can go back 59 minutes.
STANDARD_PROPERTY_MINUTES = 29
# From Google's realtime API schema. Only hints: Google checks the names.
REALTIME_DIMENSIONS = (
    "appVersion, audienceId, audienceName, audienceResourceName, city, cityId, "
    "country, countryId, deviceCategory, eventName, minutesAgo, platform, streamId, "
    "streamName, unifiedScreenName"
)
REALTIME_METRICS = "activeUsers, eventCount, keyEvents, screenPageViews"
REALTIME_HINT = (
    f"Realtime reports take only these dimensions: {REALTIME_DIMENSIONS}, plus "
    "user-scoped custom dimensions (customUser:...), and only these metrics: "
    f"{REALTIME_METRICS}."
)

_TEST_REALTIME_REPORT = {
    "dimensionHeaders": [{"name": "country"}],
    "metricHeaders": [{"name": "activeUsers", "type": "TYPE_INTEGER"}],
    "rows": [
        {
            "dimensionValues": [{"value": "United States"}],
            "metricValues": [{"value": "42"}],
        },
        {"dimensionValues": [{"value": "Germany"}], "metricValues": [{"value": "7"}]},
    ],
    "totals": [
        {
            "dimensionValues": [{"value": "RESERVED_TOTAL"}],
            "metricValues": [{"value": "49"}],
        }
    ],
    "rowCount": 2,
    "kind": "analyticsData#runRealtimeReport",
}
_TEST_ROWS = [
    {"country": "United States", "activeUsers": 42},
    {"country": "Germany", "activeUsers": 7},
]


class GoogleAnalyticsRunRealtimeReportBlock(Block):
    """Run a Google Analytics 4 realtime report of the last minutes."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [ANALYTICS_READONLY_SCOPE]
        )
        property_id: str = SchemaField(
            description=PROPERTY_ID_DESCRIPTION, placeholder="123456789"
        )
        metrics: list[str] = SchemaField(
            description=f"Realtime metric API names: {REALTIME_METRICS}",
            default=["activeUsers"],
            advanced=False,
        )
        dimensions: list[str] = SchemaField(
            description=(
                "Realtime dimension API names to break the numbers down by, up to 9, "
                "such as country, city, deviceCategory, unifiedScreenName, eventName, "
                "platform or minutesAgo. Leave empty for a single row of totals."
            ),
            default_factory=list,
            advanced=False,
        )
        minutes_ago: int = SchemaField(
            description=(
                "How many minutes back to look. 29, the most a standard property "
                "allows, covers the last 30 minutes. Google Analytics 360 properties "
                "allow up to 59."
            ),
            default=STANDARD_PROPERTY_MINUTES,
            ge=0,
            le=59,
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

    def __init__(self):
        super().__init__(
            id="16d953ee-670a-4d2d-9c24-16554f8646aa",
            description=(
                "Run a Google Analytics 4 realtime report of the last 30 minutes of "
                "activity (60 on Analytics 360), such as active users by country, "
                f"device or page. {REALTIME_HINT}"
            ),
            categories={BlockCategory.DATA, BlockCategory.MARKETING},
            input_schema=GoogleAnalyticsRunRealtimeReportBlock.Input,
            output_schema=GoogleAnalyticsRunRealtimeReportBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "property_id": "123456789",
                "dimensions": ["country"],
                "order_by": "-activeUsers",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("rows", _TEST_ROWS),
                ("row", _TEST_ROWS[0]),
                ("row", _TEST_ROWS[1]),
                ("totals", {"activeUsers": 49}),
                ("row_count", 2),
            ],
            test_mock={
                "_run_realtime_report": lambda *args, **kwargs: _TEST_REALTIME_REPORT
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        name = property_resource_name(input_data.property_id, self.name, self.id)
        body = report_body(
            metrics=input_data.metrics,
            dimensions=input_data.dimensions,
            filters=input_data.dimension_filters,
            order_by=input_data.order_by,
            limit=input_data.limit,
            block_name=self.name,
            block_id=self.id,
        )
        body["minuteRanges"] = [
            {"startMinutesAgo": input_data.minutes_ago, "endMinutesAgo": 0}
        ]
        service = build_data_service(credentials)
        try:
            response = await asyncio.to_thread(
                self._run_realtime_report, service, name, body
            )
        except HttpError as e:
            raise analytics_error(
                e,
                self.name,
                self.id,
                api=DATA_API,
                property_name=name,
                hint=realtime_hint(input_data.minutes_ago),
            ) from e

        report = parse_report(response)
        yield "rows", report.rows
        for row in report.rows:
            yield "row", row
        yield "totals", report.totals
        yield "row_count", report.row_count

    @staticmethod
    def _run_realtime_report(service, property_name: str, body: dict) -> dict:
        return (
            service.properties()
            .runRealtimeReport(property=property_name, body=body)
            .execute()
        )


def realtime_hint(minutes_ago: int) -> str:
    """What to add when Google rejects a realtime request as invalid."""
    if minutes_ago <= STANDARD_PROPERTY_MINUTES:
        return REALTIME_HINT
    return (
        f"{REALTIME_HINT} A standard property can look back at most "
        f"{STANDARD_PROPERTY_MINUTES} minutes; more needs Google Analytics 360."
    )
