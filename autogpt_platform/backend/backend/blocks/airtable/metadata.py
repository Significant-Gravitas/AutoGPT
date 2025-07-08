"""
Airtable metadata blocks for bases and views.
"""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import airtable


class AirtableListBasesBlock(Block):
    """
    Lists all Airtable bases accessible by the API token.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )

    class Output(BlockSchema):
        bases: list[dict] = SchemaField(
            description="Array of base objects with id and name"
        )

    def __init__(self):
        super().__init__(
            id="613f9907-bef8-468a-be6d-2dd7a53f96e7",
            description="List all accessible Airtable bases",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # List bases
        response = await Requests().get(
            "https://api.airtable.com/v0/meta/bases",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        data = response.json()

        yield "bases", data.get("bases", [])


class AirtableListViewsBlock(Block):
    """
    Lists all views in an Airtable base with their associated tables.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")

    class Output(BlockSchema):
        views: list[dict] = SchemaField(
            description="Array of view objects with tableId"
        )

    def __init__(self):
        super().__init__(
            id="3878cf82-d384-40c2-aace-097042233f6a",
            description="List all views in an Airtable base",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Get base schema which includes views
        response = await Requests().get(
            f"https://api.airtable.com/v0/meta/bases/{input_data.base_id}/tables",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        data = response.json()

        # Extract all views from all tables
        all_views = []
        for table in data.get("tables", []):
            table_id = table.get("id")
            for view in table.get("views", []):
                view_with_table = {**view, "tableId": table_id}
                all_views.append(view_with_table)

        yield "views", all_views


class AirtableGetViewBlock(Block):
    """
    Gets detailed information about a specific view in an Airtable base.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        view_id: str = SchemaField(description="The view ID to retrieve")

    class Output(BlockSchema):
        view: dict = SchemaField(description="Full view object with configuration")

    def __init__(self):
        super().__init__(
            id="ad0dd9f3-b3f4-446b-8142-e81a566797c4",
            description="Get details of a specific Airtable view",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Get specific view
        response = await Requests().get(
            f"https://api.airtable.com/v0/meta/bases/{input_data.base_id}/views/{input_data.view_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        view_data = response.json()

        yield "view", view_data
