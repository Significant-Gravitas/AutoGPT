"""
Airtable record operation blocks.
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

from ._api import (
    create_record,
    delete_multiple_records,
    get_record,
    list_records,
    update_multiple_records,
)
from ._config import airtable


class AirtableListRecordsBlock(Block):
    """
    Lists records from an Airtable table with optional filtering, sorting, and pagination.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        table_id_or_name: str = SchemaField(description="Table ID or name")
        filter_formula: str = SchemaField(
            description="Airtable formula to filter records", default=""
        )
        view: str = SchemaField(description="View ID or name to use", default="")
        sort: list[dict] = SchemaField(
            description="Sort configuration (array of {field, direction})", default=[]
        )
        max_records: int = SchemaField(
            description="Maximum number of records to return", default=100
        )
        page_size: int = SchemaField(
            description="Number of records per page (max 100)", default=100
        )
        offset: str = SchemaField(
            description="Pagination offset from previous request", default=""
        )
        return_fields: list[str] = SchemaField(
            description="Specific fields to return (comma-separated)", default=[]
        )

    class Output(BlockSchema):
        records: list[dict] = SchemaField(description="Array of record objects")
        offset: Optional[str] = SchemaField(
            description="Offset for next page (null if no more records)", default=None
        )

    def __init__(self):
        super().__init__(
            id="588a9fde-5733-4da7-b03c-35f5671e960f",
            description="List records from an Airtable table",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await list_records(
            credentials,
            input_data.base_id,
            input_data.table_id_or_name,
            filter_by_formula=(
                input_data.filter_formula if input_data.filter_formula else None
            ),
            view=input_data.view if input_data.view else None,
            sort=input_data.sort if input_data.sort else None,
            max_records=input_data.max_records if input_data.max_records else None,
            page_size=min(input_data.page_size, 100) if input_data.page_size else None,
            offset=input_data.offset if input_data.offset else None,
            fields=input_data.return_fields if input_data.return_fields else None,
        )

        yield "records", data.get("records", [])
        yield "offset", data.get("offset", None)


class AirtableGetRecordBlock(Block):
    """
    Retrieves a single record from an Airtable table by its ID.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        table_id_or_name: str = SchemaField(description="Table ID or name")
        record_id: str = SchemaField(description="The record ID to retrieve")

    class Output(BlockSchema):
        id: str = SchemaField(description="The record ID")
        fields: dict = SchemaField(description="The record fields")
        created_time: str = SchemaField(description="The record created time")

    def __init__(self):
        super().__init__(
            id="c29c5cbf-0aff-40f9-bbb5-f26061792d2b",
            description="Get a single record from Airtable",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        record = await get_record(
            credentials,
            input_data.base_id,
            input_data.table_id_or_name,
            input_data.record_id,
        )

        yield "id", record.get("id", None)
        yield "fields", record.get("fields", None)
        yield "created_time", record.get("createdTime", None)


class AirtableCreateRecordsBlock(Block):
    """
    Creates one or more records in an Airtable table.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        table_id_or_name: str = SchemaField(description="Table ID or name")
        records: list[dict] = SchemaField(
            description="Array of records to create (each with 'fields' object)"
        )
        typecast: bool = SchemaField(
            description="Automatically convert string values to appropriate types",
            default=False,
        )
        return_fields_by_field_id: bool | None = SchemaField(
            description="Return fields by field ID",
            default=None,
        )

    class Output(BlockSchema):
        records: list[dict] = SchemaField(description="Array of created record objects")
        details: dict = SchemaField(description="Details of the created records")

    def __init__(self):
        super().__init__(
            id="42527e98-47b6-44ce-ac0e-86b4883721d3",
            description="Create records in an Airtable table",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # The create_record API expects records in a specific format
        data = await create_record(
            credentials,
            input_data.base_id,
            input_data.table_id_or_name,
            records=[{"fields": record} for record in input_data.records],
            typecast=input_data.typecast if input_data.typecast else None,
            return_fields_by_field_id=input_data.return_fields_by_field_id,
        )

        yield "records", data.get("records", [])
        details = data.get("details", None)
        if details:
            yield "details", details


class AirtableUpdateRecordsBlock(Block):
    """
    Updates one or more existing records in an Airtable table.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        table_id_or_name: str = SchemaField(
            description="Table ID or name - It's better to use the table ID instead of the name"
        )
        records: list[dict] = SchemaField(
            description="Array of records to update (each with 'id' and 'fields')"
        )
        typecast: bool | None = SchemaField(
            description="Automatically convert string values to appropriate types",
            default=None,
        )

    class Output(BlockSchema):
        records: list[dict] = SchemaField(description="Array of updated record objects")

    def __init__(self):
        super().__init__(
            id="6e7d2590-ac2b-4b5d-b08c-fc039cd77e1f",
            description="Update records in an Airtable table",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # The update_multiple_records API expects records with id and fields
        data = await update_multiple_records(
            credentials,
            input_data.base_id,
            input_data.table_id_or_name,
            records=input_data.records,
            typecast=input_data.typecast if input_data.typecast else None,
            return_fields_by_field_id=False,  # Use field names, not IDs
        )

        yield "records", data.get("records", [])


class AirtableDeleteRecordsBlock(Block):
    """
    Deletes one or more records from an Airtable table.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        table_id_or_name: str = SchemaField(
            description="Table ID or name - It's better to use the table ID instead of the name"
        )
        record_ids: list[str] = SchemaField(
            description="Array of upto 10 record IDs to delete"
        )

    class Output(BlockSchema):
        records: list[dict] = SchemaField(description="Array of deletion results")

    def __init__(self):
        super().__init__(
            id="93e22b8b-3642-4477-aefb-1c0929a4a3a6",
            description="Delete records from an Airtable table",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if len(input_data.record_ids) > 10:
            yield "error", "Only upto 10 record IDs can be deleted at a time"
        else:
            data = await delete_multiple_records(
                credentials,
                input_data.base_id,
                input_data.table_id_or_name,
                input_data.record_ids,
            )

            yield "records", data.get("records", [])
