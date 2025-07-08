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
    Requests,
    SchemaField,
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
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_id_or_name: str = SchemaField(description="Table ID or name", default="")
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
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build query parameters
        params = {}
        if input_data.filter_formula:
            params["filterByFormula"] = input_data.filter_formula
        if input_data.view:
            params["view"] = input_data.view
        if input_data.sort:
            for i, sort_config in enumerate(input_data.sort):
                params[f"sort[{i}][field]"] = sort_config.get("field", "")
                params[f"sort[{i}][direction]"] = sort_config.get("direction", "asc")
        if input_data.max_records:
            params["maxRecords"] = input_data.max_records
        if input_data.page_size:
            params["pageSize"] = min(input_data.page_size, 100)
        if input_data.offset:
            params["offset"] = input_data.offset
        if input_data.return_fields:
            for i, field in enumerate(input_data.return_fields):
                params[f"fields[{i}]"] = field

        # Make request
        response = await Requests().get(
            f"https://api.airtable.com/v0/{input_data.base_id}/{input_data.table_id_or_name}",
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
        )

        data = response.json()

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
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_id_or_name: str = SchemaField(description="Table ID or name", default="")
        record_id: str = SchemaField(description="The record ID to retrieve")
        return_fields: list[str] = SchemaField(
            description="Specific fields to return", default=[]
        )

    class Output(BlockSchema):
        record: dict = SchemaField(description="The record object")

    def __init__(self):
        super().__init__(
            id="c29c5cbf-0aff-40f9-bbb5-f26061792d2b",
            description="Get a single record from Airtable",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build query parameters
        params = {}
        if input_data.return_fields:
            for i, field in enumerate(input_data.return_fields):
                params[f"fields[{i}]"] = field

        # Make request
        response = await Requests().get(
            f"https://api.airtable.com/v0/{input_data.base_id}/{input_data.table_id_or_name}/{input_data.record_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
        )

        record = response.json()

        yield "record", record


class AirtableCreateRecordsBlock(Block):
    """
    Creates one or more records in an Airtable table.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_id_or_name: str = SchemaField(description="Table ID or name", default="")
        records: list[dict] = SchemaField(
            description="Array of records to create (each with 'fields' object)"
        )
        typecast: bool = SchemaField(
            description="Automatically convert string values to appropriate types",
            default=False,
        )
        return_fields: list[str] = SchemaField(
            description="Specific fields to return in created records", default=[]
        )

    class Output(BlockSchema):
        records: list[dict] = SchemaField(description="Array of created record objects")

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
        api_key = credentials.api_key.get_secret_value()

        # Build request body
        body = {"records": input_data.records, "typecast": input_data.typecast}

        # Build query parameters for return fields
        params = {}
        if input_data.return_fields:
            for i, field in enumerate(input_data.return_fields):
                params[f"fields[{i}]"] = field

        # Make request
        response = await Requests().post(
            f"https://api.airtable.com/v0/{input_data.base_id}/{input_data.table_id_or_name}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            params=params,
        )

        data = response.json()

        yield "records", data.get("records", [])


class AirtableUpdateRecordsBlock(Block):
    """
    Updates one or more existing records in an Airtable table.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_id_or_name: str = SchemaField(description="Table ID or name", default="")
        records: list[dict] = SchemaField(
            description="Array of records to update (each with 'id' and 'fields')"
        )
        typecast: bool = SchemaField(
            description="Automatically convert string values to appropriate types",
            default=False,
        )
        return_fields: list[str] = SchemaField(
            description="Specific fields to return in updated records", default=[]
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
        api_key = credentials.api_key.get_secret_value()

        # Build request body
        body = {"records": input_data.records, "typecast": input_data.typecast}

        # Build query parameters for return fields
        params = {}
        if input_data.return_fields:
            for i, field in enumerate(input_data.return_fields):
                params[f"fields[{i}]"] = field

        # Make request
        response = await Requests().patch(
            f"https://api.airtable.com/v0/{input_data.base_id}/{input_data.table_id_or_name}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            params=params,
        )

        data = response.json()

        yield "records", data.get("records", [])


class AirtableUpsertRecordsBlock(Block):
    """
    Creates or updates records in an Airtable table based on a merge field.

    If a record with the same value in the merge field exists, it will be updated.
    Otherwise, a new record will be created.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_id_or_name: str = SchemaField(description="Table ID or name", default="")
        records: list[dict] = SchemaField(
            description="Array of records to upsert (each with 'fields' object)"
        )
        merge_field: str = SchemaField(
            description="Field to use for matching existing records"
        )
        typecast: bool = SchemaField(
            description="Automatically convert string values to appropriate types",
            default=False,
        )
        return_fields: list[str] = SchemaField(
            description="Specific fields to return in upserted records", default=[]
        )

    class Output(BlockSchema):
        records: list[dict] = SchemaField(
            description="Array of created/updated record objects"
        )

    def __init__(self):
        super().__init__(
            id="99f78a9d-3418-429f-a6fb-9d2166638e99",
            description="Create or update records based on a merge field",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build request body
        body = {
            "performUpsert": {"fieldsToMergeOn": [input_data.merge_field]},
            "records": input_data.records,
            "typecast": input_data.typecast,
        }

        # Build query parameters for return fields
        params = {}
        if input_data.return_fields:
            for i, field in enumerate(input_data.return_fields):
                params[f"fields[{i}]"] = field

        # Make request
        response = await Requests().post(
            f"https://api.airtable.com/v0/{input_data.base_id}/{input_data.table_id_or_name}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            params=params,
        )

        data = response.json()

        yield "records", data.get("records", [])


class AirtableDeleteRecordsBlock(Block):
    """
    Deletes one or more records from an Airtable table.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID", default="")
        table_id_or_name: str = SchemaField(description="Table ID or name", default="")
        record_ids: list[str] = SchemaField(description="Array of record IDs to delete")

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
        api_key = credentials.api_key.get_secret_value()

        # Build query parameters
        params = {}
        for i, record_id in enumerate(input_data.record_ids):
            params[f"records[{i}]"] = record_id

        # Make request
        response = await Requests().delete(
            f"https://api.airtable.com/v0/{input_data.base_id}/{input_data.table_id_or_name}",
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
        )

        data = response.json()

        yield "records", data.get("records", [])
