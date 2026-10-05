import asyncio
from typing import Any

from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

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
    ADMIN_API,
    ANALYTICS_READONLY_SCOPE,
    DATA_API,
    PROPERTY_ID_DESCRIPTION,
    analytics_error,
    build_admin_service,
    build_data_service,
    property_resource_name,
)
from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)

ACCOUNT_SUMMARIES_PAGE_SIZE = 200  # the most the Admin API returns per page


class GoogleAnalyticsProperty(BaseModel):
    """A Google Analytics 4 property the connected account can read."""

    property_id: str = Field(
        description=(
            "The numeric property ID, which the other Google Analytics blocks take"
        )
    )
    name: str = Field(description="The property's resource name (properties/...)")
    display_name: str = Field(default="", description="The property's name")
    property_type: str = Field(description="ordinary, subproperty or rollup")
    account_id: str = Field(
        description="Numeric ID of the Google Analytics account the property is in"
    )
    account_name: str = Field(default="", description="The account's name")


class _Definition(BaseModel):
    api_name: str = Field(
        description=(
            "The name the report blocks take, such as country or customEvent:plan"
        )
    )
    ui_name: str = Field(default="", description="The name shown in Google Analytics")
    description: str = Field(default="", description="What it measures or describes")
    category: str = Field(
        default="", description="The group Google Analytics lists it under"
    )
    custom: bool = Field(
        default=False,
        description="Whether it's one of the property's own custom definitions",
    )


class GoogleAnalyticsDimension(_Definition):
    """A dimension a Google Analytics property can report on."""


class GoogleAnalyticsMetric(_Definition):
    """A metric a Google Analytics property can report on."""

    type: str = Field(
        description=(
            "What its values are: integer, float, seconds, milliseconds, minutes, "
            "hours, currency, standard (a custom metric's plain number) or a "
            "distance (feet, miles, meters, kilometers)"
        )
    )


_TEST_ACCOUNT_SUMMARIES = [
    {
        "name": "accountSummaries/2000",
        "account": "accounts/2000",
        "displayName": "Example Corp",
        "propertySummaries": [
            {
                "property": "properties/123456789",
                "displayName": "example.com",
                "propertyType": "PROPERTY_TYPE_ORDINARY",
                "parent": "accounts/2000",
            },
            {
                "property": "properties/987654321",
                "displayName": "Example app",
                "propertyType": "PROPERTY_TYPE_ORDINARY",
                "parent": "accounts/2000",
            },
        ],
    }
]
_TEST_METADATA = {
    "name": "properties/123456789/metadata",
    "dimensions": [
        {
            "apiName": "country",
            "uiName": "Country",
            "description": "The country from which the user activity originated.",
            "category": "Geography",
        },
        {
            "apiName": "customEvent:plan",
            "uiName": "Plan",
            "description": "An event scoped custom dimension.",
            "category": "Custom",
            "customDefinition": True,
        },
    ],
    "metrics": [
        {
            "apiName": "activeUsers",
            "uiName": "Active users",
            "description": "The number of distinct users who visited your site.",
            "type": "TYPE_INTEGER",
            "category": "User",
        },
        {
            "apiName": "customEvent:credits_used",
            "uiName": "Credits used",
            "description": "An event scoped custom metric.",
            "type": "TYPE_STANDARD",
            "category": "Custom",
            "customDefinition": True,
        },
    ],
}


class GoogleAnalyticsListPropertiesBlock(Block):
    """List the Google Analytics 4 properties the connected account can read."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [ANALYTICS_READONLY_SCOPE]
        )

    class Output(BlockSchemaOutput):
        properties: list[GoogleAnalyticsProperty] = SchemaField(
            description="The properties the account can read, account by account"
        )
        property: GoogleAnalyticsProperty = SchemaField(description="Each property")

    def __init__(self):
        properties = to_properties(_TEST_ACCOUNT_SUMMARIES)
        super().__init__(
            id="8f667593-bd61-4385-9d2e-85e8a77180e9",
            description=(
                "List the Google Analytics 4 properties the connected Google account "
                "can read, with their numeric property IDs and account names. The "
                "other Google Analytics blocks take one of these property IDs."
            ),
            categories={BlockCategory.DATA, BlockCategory.MARKETING},
            input_schema=GoogleAnalyticsListPropertiesBlock.Input,
            output_schema=GoogleAnalyticsListPropertiesBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("properties", properties),
                ("property", properties[0]),
                ("property", properties[1]),
            ],
            test_mock={
                "_list_account_summaries": lambda *args, **kwargs: _TEST_ACCOUNT_SUMMARIES
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = build_admin_service(credentials)
        try:
            summaries = await asyncio.to_thread(self._list_account_summaries, service)
        except HttpError as e:
            raise analytics_error(e, self.name, self.id, api=ADMIN_API) from e

        properties = to_properties(summaries)
        yield "properties", properties
        for item in properties:
            yield "property", item

    @staticmethod
    def _list_account_summaries(service) -> list[dict]:
        summaries: list[dict] = []
        request = service.accountSummaries().list(pageSize=ACCOUNT_SUMMARIES_PAGE_SIZE)
        while request is not None:
            response = request.execute()
            summaries += response.get("accountSummaries", [])
            request = service.accountSummaries().list_next(request, response)
        return summaries


class GoogleAnalyticsListDimensionsAndMetricsBlock(Block):
    """List the dimensions and metrics a Google Analytics property can report on."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [ANALYTICS_READONLY_SCOPE]
        )
        property_id: str = SchemaField(
            description=PROPERTY_ID_DESCRIPTION, placeholder="123456789"
        )
        custom_only: bool = SchemaField(
            description=(
                "List only the property's custom dimensions and metrics. Turn off to "
                "list the standard ones too, several hundred in all."
            ),
            default=True,
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        dimensions: list[GoogleAnalyticsDimension] = SchemaField(
            description=(
                "Dimensions the property's reports can be split by (only its custom "
                "ones unless custom_only is off)"
            )
        )
        metrics: list[GoogleAnalyticsMetric] = SchemaField(
            description=(
                "Metrics the property's reports can show (only its custom ones unless "
                "custom_only is off)"
            )
        )

    def __init__(self):
        super().__init__(
            id="720597ae-c0e1-4fe1-8811-6d4d460b07ab",
            description=(
                "List the dimensions and metrics a Google Analytics 4 property can "
                "report on, with the API names the report blocks take. By default it "
                "lists only the property's custom ones; turn off custom_only to "
                "include the standard ones too."
            ),
            categories={BlockCategory.DATA, BlockCategory.MARKETING},
            input_schema=GoogleAnalyticsListDimensionsAndMetricsBlock.Input,
            output_schema=GoogleAnalyticsListDimensionsAndMetricsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "property_id": "123456789",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("dimensions", [to_dimension(_TEST_METADATA["dimensions"][1])]),
                ("metrics", [to_metric(_TEST_METADATA["metrics"][1])]),
            ],
            test_mock={"_get_metadata": lambda *args, **kwargs: _TEST_METADATA},
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        name = property_resource_name(input_data.property_id, self.name, self.id)
        service = build_data_service(credentials)
        try:
            metadata = await asyncio.to_thread(self._get_metadata, service, name)
        except HttpError as e:
            raise analytics_error(
                e, self.name, self.id, api=DATA_API, property_name=name
            ) from e

        dimensions = _pick(metadata.get("dimensions", []), input_data.custom_only)
        metrics = _pick(metadata.get("metrics", []), input_data.custom_only)
        yield "dimensions", [to_dimension(item) for item in dimensions]
        yield "metrics", [to_metric(item) for item in metrics]

    @staticmethod
    def _get_metadata(service, property_name: str) -> dict:
        return (
            service.properties().getMetadata(name=f"{property_name}/metadata").execute()
        )


def to_properties(summaries: list[dict[str, Any]]) -> list[GoogleAnalyticsProperty]:
    """Flatten Admin API account summaries into the properties they list."""
    return [
        GoogleAnalyticsProperty(
            property_id=item.get("property", "").removeprefix("properties/"),
            name=item.get("property", ""),
            display_name=item.get("displayName", ""),
            property_type=_enum_word(item.get("propertyType"), "PROPERTY_TYPE_"),
            account_id=account.get("account", "").removeprefix("accounts/"),
            account_name=account.get("displayName", ""),
        )
        for account in summaries
        for item in account.get("propertySummaries", [])
    ]


def to_dimension(item: dict[str, Any]) -> GoogleAnalyticsDimension:
    """Map a Data API DimensionMetadata (or MetricMetadata) to its common fields."""
    return GoogleAnalyticsDimension(
        api_name=item.get("apiName", ""),
        ui_name=item.get("uiName", ""),
        description=item.get("description", ""),
        category=item.get("category", ""),
        custom=bool(item.get("customDefinition")),
    )


def to_metric(item: dict[str, Any]) -> GoogleAnalyticsMetric:
    return GoogleAnalyticsMetric(
        **to_dimension(item).model_dump(),
        type=_enum_word(item.get("type"), "TYPE_"),
    )


def _pick(items: list[dict[str, Any]], custom_only: bool) -> list[dict[str, Any]]:
    return [item for item in items if item.get("customDefinition") or not custom_only]


def _enum_word(value: str | None, prefix: str) -> str:
    """Turn an API enum value such as PROPERTY_TYPE_ORDINARY into 'ordinary'."""
    return (value or f"{prefix}UNSPECIFIED").removeprefix(prefix).lower()
