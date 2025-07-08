"""
Airtable schema and table management blocks.
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


class AirtableListSchemaBlock(Block):
    """
    Retrieves the complete schema of an Airtable base, including all tables,
    fields, and views.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")

    class Output(BlockSchema):
        base_schema: dict = SchemaField(
            description="Complete base schema with tables, fields, and views"
        )
        tables: list[dict] = SchemaField(description="Array of table objects")

    def __init__(self):
        super().__init__(
            id="64291d3c-99b5-47b7-a976-6d94293cdb2d",
            description="Get the complete schema of an Airtable base",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Get base schema
        response = await Requests().get(
            f"https://api.airtable.com/v0/meta/bases/{input_data.base_id}/tables",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        data = response.json()

        yield "base_schema", data
        yield "tables", data.get("tables", [])


class AirtableCreateTableBlock(Block):
    """
    Creates a new table in an Airtable base with specified fields and views.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_definition: dict = SchemaField(
            description="Table definition with name, description, fields, and views",
            default={
                "name": "New Table",
                "fields": [{"name": "Name", "type": "singleLineText"}],
            },
        )

    class Output(BlockSchema):
        table: dict = SchemaField(description="Created table object")
        table_id: str = SchemaField(description="ID of the created table")

    def __init__(self):
        super().__init__(
            id="fcc20ced-d817-42ea-9b40-c35e7bf34b4f",
            description="Create a new table in an Airtable base",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Create table
        response = await Requests().post(
            f"https://api.airtable.com/v0/meta/bases/{input_data.base_id}/tables",
            headers={"Authorization": f"Bearer {api_key}"},
            json=input_data.table_definition,
        )

        table_data = response.json()

        yield "table", table_data
        yield "table_id", table_data.get("id", "")


class AirtableUpdateTableBlock(Block):
    """
    Updates an existing table's properties such as name or description.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_id: str = SchemaField(description="The table ID to update")
        patch: dict = SchemaField(
            description="Properties to update (name, description)", default={}
        )

    class Output(BlockSchema):
        table: dict = SchemaField(description="Updated table object")

    def __init__(self):
        super().__init__(
            id="34077c5f-f962-49f2-9ec6-97c67077013a",
            description="Update table properties",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Update table
        response = await Requests().patch(
            f"https://api.airtable.com/v0/meta/bases/{input_data.base_id}/tables/{input_data.table_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=input_data.patch,
        )

        table_data = response.json()

        yield "table", table_data


class AirtableDeleteTableBlock(Block):
    """
    Deletes a table from an Airtable base.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_id: str = SchemaField(description="The table ID to delete")

    class Output(BlockSchema):
        deleted: bool = SchemaField(
            description="Confirmation that the table was deleted"
        )

    def __init__(self):
        super().__init__(
            id="6b96c196-d0ad-4fb2-981f-7a330549bc22",
            description="Delete a table from an Airtable base",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Delete table
        response = await Requests().delete(
            f"https://api.airtable.com/v0/meta/bases/{input_data.base_id}/tables/{input_data.table_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        deleted = response.status in [200, 204]

        yield "deleted", deleted


class AirtableAddFieldBlock(Block):
    """
    Adds a new field (column) to an existing Airtable table.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_id: str = SchemaField(description="The table ID to add field to")
        field_definition: dict = SchemaField(
            description="Field definition with name, type, and options",
            default={"name": "New Field", "type": "singleLineText"},
        )

    class Output(BlockSchema):
        field: dict = SchemaField(description="Created field object")
        field_id: str = SchemaField(description="ID of the created field")

    def __init__(self):
        super().__init__(
            id="6c98a32f-dbf9-45d8-a2a8-5e97e8326351",
            description="Add a new field to an Airtable table",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Add field
        response = await Requests().post(
            f"https://api.airtable.com/v0/meta/bases/{input_data.base_id}/tables/{input_data.table_id}/fields",
            headers={"Authorization": f"Bearer {api_key}"},
            json=input_data.field_definition,
        )

        field_data = response.json()

        yield "field", field_data
        yield "field_id", field_data.get("id", "")


class AirtableUpdateFieldBlock(Block):
    """
    Updates an existing field's properties in an Airtable table.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_id: str = SchemaField(description="The table ID containing the field")
        field_id: str = SchemaField(description="The field ID to update")
        patch: dict = SchemaField(description="Field properties to update", default={})

    class Output(BlockSchema):
        field: dict = SchemaField(description="Updated field object")

    def __init__(self):
        super().__init__(
            id="f46ac716-3b18-4da1-92e4-34ca9a464d48",
            description="Update field properties in an Airtable table",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Update field
        response = await Requests().patch(
            f"https://api.airtable.com/v0/meta/bases/{input_data.base_id}/tables/{input_data.table_id}/fields/{input_data.field_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=input_data.patch,
        )

        field_data = response.json()

        yield "field", field_data


class AirtableDeleteFieldBlock(Block):
    """
    Deletes a field from an Airtable table.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_id: str = SchemaField(description="The table ID containing the field")
        field_id: str = SchemaField(description="The field ID to delete")

    class Output(BlockSchema):
        deleted: bool = SchemaField(
            description="Confirmation that the field was deleted"
        )

    def __init__(self):
        super().__init__(
            id="ca6ebacb-be8b-4c54-80a3-1fb519ad51c6",
            description="Delete a field from an Airtable table",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Delete field
        response = await Requests().delete(
            f"https://api.airtable.com/v0/meta/bases/{input_data.base_id}/tables/{input_data.table_id}/fields/{input_data.field_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        deleted = response.status in [200, 204]

        yield "deleted", deleted
