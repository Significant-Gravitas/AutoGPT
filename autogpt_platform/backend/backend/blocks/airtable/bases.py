"""
Airtable base operation blocks.
"""

from typing import Optional

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import create_base, list_bases
from ._config import airtable


class AirtableCreateBaseBlock(Block):
    """
    Creates a new base in an Airtable workspace.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        workspace_id: str = SchemaField(
            description="The workspace ID where the base will be created"
        )
        name: str = SchemaField(description="The name of the new base")
        tables: list[dict] = SchemaField(
            description="At least one table and field must be specified. Array of table objects to create in the base. Each table should have 'name' and 'fields' properties",
            default=[
                {
                    "description": "Default table",
                    "name": "Default table",
                    "fields": [
                        {
                            "name": "ID",
                            "type": "number",
                            "description": "Auto-incrementing ID field",
                            "options": {"precision": 0},
                        }
                    ],
                }
            ],
        )

    class Output(BlockSchema):
        base_id: str = SchemaField(description="The ID of the created base")
        tables: list[dict] = SchemaField(description="Array of table objects")
        table: dict = SchemaField(description="A single table object")

    def __init__(self):
        super().__init__(
            id="f59b88a8-54ce-4676-a508-fd614b4e8dce",
            description="Create a new base in Airtable",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await create_base(
            credentials,
            input_data.workspace_id,
            input_data.name,
            input_data.tables,
        )

        yield "base_id", data.get("id", None)
        yield "tables", data.get("tables", [])
        for table in data.get("tables", []):
            yield "table", table


class AirtableListBasesBlock(Block):
    """
    Lists all bases in an Airtable workspace that the user has access to.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        trigger: str = SchemaField(
            description="Trigger the block to run - value is ignored", default="manual"
        )
        offset: str = SchemaField(
            description="Pagination offset from previous request", default=""
        )

    class Output(BlockSchema):
        bases: list[dict] = SchemaField(description="Array of base objects")
        offset: Optional[str] = SchemaField(
            description="Offset for next page (null if no more bases)", default=None
        )

    def __init__(self):
        super().__init__(
            id="4bd8d466-ed5d-4e44-8083-97f25a8044e7",
            description="List all bases in Airtable",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await list_bases(
            credentials,
            offset=input_data.offset if input_data.offset else None,
        )

        yield "bases", data.get("bases", [])
        yield "offset", data.get("offset", None)
