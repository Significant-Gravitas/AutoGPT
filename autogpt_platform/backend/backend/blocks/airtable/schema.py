"""
Airtable schema and table management blocks.
"""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._api import TableFieldType, create_field, create_table, update_field, update_table
from ._config import airtable


class AirtableListSchemaBlock(Block):
    """
    Retrieves the complete schema of an Airtable base, including all tables,
    fields, and views.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")

    class Output(BlockSchemaOutput):
        base_schema: dict = SchemaField(
            description="Complete base schema with tables, fields, and views"
        )
        tables: list[dict] = SchemaField(description="Array of table objects")

    def __init__(self):
        super().__init__(
            id="64291d3c-99b5-47b7-a976-6d94293cdb2d",
            description="Get the complete schema of an Airtable base",
            categories={BlockCategory.DATA},
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

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        table_name: str = SchemaField(description="The name of the table to create")
        table_fields: list[dict] = SchemaField(
            description="Table fields with name, type, and options",
            default=[{"name": "Name", "type": "singleLineText"}],
        )

    class Output(BlockSchemaOutput):
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
        table_data = await create_table(
            credentials,
            input_data.base_id,
            input_data.table_name,
            input_data.table_fields,
        )

        yield "table", table_data
        yield "table_id", table_data.get("id", "")


class AirtableUpdateTableBlock(Block):
    """
    Updates an existing table's properties such as name or description.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        table_id: str = SchemaField(description="The table ID to update")
        table_name: str | None = SchemaField(
            description="The name of the table to update", default=None
        )
        table_description: str | None = SchemaField(
            description="The description of the table to update", default=None
        )
        date_dependency: dict | None = SchemaField(
            description="The date dependency of the table to update", default=None
        )

    class Output(BlockSchemaOutput):
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
        table_data = await update_table(
            credentials,
            input_data.base_id,
            input_data.table_id,
            input_data.table_name,
            input_data.table_description,
            input_data.date_dependency,
        )

        yield "table", table_data


class AirtableCreateFieldBlock(Block):
    """
    Adds a new field (column) to an existing Airtable table.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        table_id: str = SchemaField(description="The table ID to add field to")
        field_type: TableFieldType = SchemaField(
            description="The type of the field to create",
            default=TableFieldType.SINGLE_LINE_TEXT,
            advanced=False,
        )
        name: str = SchemaField(description="The name of the field to create")
        description: str | None = SchemaField(
            description="The description of the field to create", default=None
        )
        options: dict[str, str] | None = SchemaField(
            description="The options of the field to create", default=None
        )

    class Output(BlockSchemaOutput):
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
        field_data = await create_field(
            credentials,
            input_data.base_id,
            input_data.table_id,
            input_data.field_type,
            input_data.name,
        )

        yield "field", field_data
        yield "field_id", field_data.get("id", "")


class AirtableUpdateFieldBlock(Block):
    """
    Updates an existing field's properties in an Airtable table.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        table_id: str = SchemaField(description="The table ID containing the field")
        field_id: str = SchemaField(description="The field ID to update")
        name: str | None = SchemaField(
            description="The name of the field to update", default=None, advanced=False
        )
        description: str | None = SchemaField(
            description="The description of the field to update",
            default=None,
            advanced=False,
        )

    class Output(BlockSchemaOutput):
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
        field_data = await update_field(
            credentials,
            input_data.base_id,
            input_data.table_id,
            input_data.field_id,
            input_data.name,
            input_data.description,
        )

        yield "field", field_data
