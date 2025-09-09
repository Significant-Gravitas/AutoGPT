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
        normalize_output: bool = SchemaField(
            description="Normalize output to include all fields with proper empty values (adds extra API call for schema)",
            default=False,
        )
        include_field_metadata: bool = SchemaField(
            description="Include field type and configuration metadata (requires normalize_output=true)",
            default=False,
        )

    class Output(BlockSchema):
        records: list[dict] = SchemaField(description="Array of record objects")
        offset: Optional[str] = SchemaField(
            description="Offset for next page (null if no more records)", default=None
        )
        field_metadata: Optional[dict] = SchemaField(
            description="Field type and configuration metadata (only when include_field_metadata=true)",
            default=None,
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
        from ._api import get_table_schema, normalize_records

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

        records = data.get("records", [])

        # Normalize output if requested
        if input_data.normalize_output:
            # Fetch table schema
            table_schema = await get_table_schema(
                credentials, input_data.base_id, input_data.table_id_or_name
            )

            # Normalize the records
            normalized_data = await normalize_records(
                records,
                table_schema,
                include_field_metadata=input_data.include_field_metadata,
            )

            yield "records", normalized_data["records"]
            yield "offset", data.get("offset", None)

            if (
                input_data.include_field_metadata
                and "field_metadata" in normalized_data
            ):
                yield "field_metadata", normalized_data["field_metadata"]
        else:
            yield "records", records
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
        normalize_output: bool = SchemaField(
            description="Normalize output to include all fields with proper empty values (adds extra API call for schema)",
            default=False,
        )
        include_field_metadata: bool = SchemaField(
            description="Include field type and configuration metadata (requires normalize_output=true)",
            default=False,
        )

    class Output(BlockSchema):
        id: str = SchemaField(description="The record ID")
        fields: dict = SchemaField(description="The record fields")
        created_time: str = SchemaField(description="The record created time")
        field_metadata: Optional[dict] = SchemaField(
            description="Field type and configuration metadata (only when include_field_metadata=true)",
            default=None,
        )

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
        from ._api import get_table_schema, normalize_records

        record = await get_record(
            credentials,
            input_data.base_id,
            input_data.table_id_or_name,
            input_data.record_id,
        )

        # Normalize output if requested
        if input_data.normalize_output:
            # Fetch table schema
            table_schema = await get_table_schema(
                credentials, input_data.base_id, input_data.table_id_or_name
            )

            # Normalize the single record (wrap in list and unwrap result)
            normalized_data = await normalize_records(
                [record],
                table_schema,
                include_field_metadata=input_data.include_field_metadata,
            )

            normalized_record = normalized_data["records"][0]
            yield "id", normalized_record.get("id", None)
            yield "fields", normalized_record.get("fields", None)
            yield "created_time", normalized_record.get("createdTime", None)

            if (
                input_data.include_field_metadata
                and "field_metadata" in normalized_data
            ):
                yield "field_metadata", normalized_data["field_metadata"]
        else:
            yield "id", record.get("id", None)
            yield "fields", record.get("fields", None)
            yield "created_time", record.get("createdTime", None)


class AirtableCreateRecordsBlock(Block):
    """
    Creates one or more records in an Airtable table, or finds existing records based on unique fields.
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
        find_by_fields: list[str] = SchemaField(
            description="Field names to use for finding existing records before creating (e.g., ['email', 'id'])",
            default=[],
        )
        update_if_found: bool = SchemaField(
            description="If true, update existing records when found; if false, skip creation",
            default=False,
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
        records: list[dict] = SchemaField(
            description="Array of created/found/updated record objects"
        )
        created_count: int = SchemaField(
            description="Number of records created", default=0
        )
        found_count: int = SchemaField(
            description="Number of existing records found", default=0
        )
        updated_count: int = SchemaField(
            description="Number of records updated", default=0
        )
        details: dict = SchemaField(description="Details of the operation")

    def __init__(self):
        super().__init__(
            id="42527e98-47b6-44ce-ac0e-86b4883721d3",
            description="Create or find records in an Airtable table",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        result_records = []
        created_count = 0
        found_count = 0
        updated_count = 0

        # If find_by_fields is specified, check for existing records first
        if input_data.find_by_fields:
            for record_data in input_data.records:
                # Build filter formula for finding existing records
                filter_parts = []
                for field_name in input_data.find_by_fields:
                    if field_name in record_data:
                        value = record_data[field_name]
                        if isinstance(value, str):
                            # Escape single quotes in the value
                            escaped_value = value.replace("'", "\\'")
                            filter_parts.append(f"{{{field_name}}} = '{escaped_value}'")
                        elif isinstance(value, bool):
                            filter_parts.append(
                                f"{{{field_name}}} = "
                                + ("TRUE()" if value else "FALSE()")
                            )
                        elif isinstance(value, (int, float)):
                            filter_parts.append(f"{{{field_name}}} = {value}")

                if filter_parts:
                    filter_formula = (
                        "AND(" + ", ".join(filter_parts) + ")"
                        if len(filter_parts) > 1
                        else filter_parts[0]
                    )

                    # Search for existing record
                    existing_records = await list_records(
                        credentials,
                        input_data.base_id,
                        input_data.table_id_or_name,
                        filter_by_formula=filter_formula,
                        max_records=1,
                    )

                    if existing_records.get("records"):
                        # Record exists
                        existing_record = existing_records["records"][0]
                        found_count += 1

                        if input_data.update_if_found:
                            # Update the existing record
                            update_data = await update_multiple_records(
                                credentials,
                                input_data.base_id,
                                input_data.table_id_or_name,
                                records=[
                                    {"id": existing_record["id"], "fields": record_data}
                                ],
                                typecast=input_data.typecast,
                                return_fields_by_field_id=input_data.return_fields_by_field_id,
                            )
                            result_records.extend(update_data.get("records", []))
                            updated_count += 1
                        else:
                            # Just return the existing record
                            result_records.append(existing_record)
                    else:
                        # Record doesn't exist, create it
                        create_data = await create_record(
                            credentials,
                            input_data.base_id,
                            input_data.table_id_or_name,
                            records=[{"fields": record_data}],
                            typecast=input_data.typecast,
                            return_fields_by_field_id=input_data.return_fields_by_field_id,
                        )
                        result_records.extend(create_data.get("records", []))
                        created_count += 1
                else:
                    # No fields to match on, just create
                    create_data = await create_record(
                        credentials,
                        input_data.base_id,
                        input_data.table_id_or_name,
                        records=[{"fields": record_data}],
                        typecast=input_data.typecast,
                        return_fields_by_field_id=input_data.return_fields_by_field_id,
                    )
                    result_records.extend(create_data.get("records", []))
                    created_count += 1
        else:
            # No find_by_fields specified, use original behavior (backwards compatible)
            data = await create_record(
                credentials,
                input_data.base_id,
                input_data.table_id_or_name,
                records=[{"fields": record} for record in input_data.records],
                typecast=input_data.typecast if input_data.typecast else None,
                return_fields_by_field_id=input_data.return_fields_by_field_id,
            )
            result_records = data.get("records", [])
            created_count = len(result_records)

        yield "records", result_records
        yield "created_count", created_count
        yield "found_count", found_count
        yield "updated_count", updated_count
        yield "details", {
            "total_processed": len(input_data.records),
            "created": created_count,
            "found": found_count,
            "updated": updated_count,
        }


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
