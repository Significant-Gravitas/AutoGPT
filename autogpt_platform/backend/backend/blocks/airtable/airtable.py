# """
# Airtable API integration blocks.

# This module provides blocks for interacting with the Airtable API,
# including operations for tables, fields, and records.
# """

# import enum
# import logging
# from typing import Dict, List, Optional

# from pydantic import BaseModel

# from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
# from backend.data.model import APIKeyCredentials, CredentialsField, SchemaField

# from ._api import AirtableAPIException, AirtableClient
# from ._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, AirtableCredentialsInput

# logger = logging.getLogger(__name__)


# # Common response models
# class AirtableTable(BaseModel):
#     id: str
#     name: str
#     description: Optional[str] = None


# class AirtableField(BaseModel):
#     id: str
#     name: str
#     type: str


# class AirtableRecord(BaseModel):
#     id: str
#     fields: Dict
#     created_time: Optional[str] = None


# class AirtableTablesBlock(Block):
#     """Block for listing, getting, and creating tables in Airtable."""

#     class Input(BlockSchema):
#         base_id: str = SchemaField(
#             description="The ID of the Airtable base",
#             placeholder="appXXXXXXXXXXXXXX",
#         )
#         operation: str = SchemaField(
#             description="The operation to perform on tables",
#             placeholder="list",
#             choices=["list", "get", "create"],
#         )
#         table_id: Optional[str] = SchemaField(
#             description="The ID of the table (required for 'get' operation)",
#             placeholder="tblXXXXXXXXXXXXXX",
#             advanced=True,
#         )
#         table_name: Optional[str] = SchemaField(
#             description="The name of the new table (required for 'create' operation)",
#             placeholder="My New Table",
#             advanced=True,
#         )
#         table_description: Optional[str] = SchemaField(
#             description="The description of the new table (for 'create' operation)",
#             placeholder="Description of my table",
#             advanced=True,
#         )
#         fields: Optional[List[Dict[str, str]]] = SchemaField(
#             description="The fields to create in the new table (for 'create' operation)",
#             placeholder='[{"name": "Name", "type": "text"}]',
#             advanced=True,
#         )
#         credentials: AirtableCredentialsInput = CredentialsField(
#             description="The credentials for the Airtable API"
#         )

#     class Output(BlockSchema):
#         tables: Optional[List[AirtableTable]] = SchemaField(
#             description="List of tables in the base"
#         )
#         table: Optional[AirtableTable] = SchemaField(
#             description="The retrieved or created table"
#         )
#         error: Optional[str] = SchemaField(description="Error message if any")

#     def __init__(self):
#         super().__init__(
#             id="da53b48c-6e97-4c1c-afb9-4ecf10c81856",
#             description="List, get, or create tables in an Airtable base",
#             categories={BlockCategory.DATA},
#             input_schema=AirtableTablesBlock.Input,
#             output_schema=AirtableTablesBlock.Output,
#             test_input={
#                 "base_id": "appXXXXXXXXXXXXXX",
#                 "operation": "list",
#                 "credentials": TEST_CREDENTIALS_INPUT,
#             },
#             test_output=[
#                 ("tables", [AirtableTable(id="tbl123", name="Example Table")])
#             ],
#             test_mock={
#                 "list_tables": lambda *args, **kwargs: {
#                     "tables": [{"id": "tbl123", "name": "Example Table"}]
#                 }
#             },
#             test_credentials=TEST_CREDENTIALS,
#         )

#     def run(
#         self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
#     ) -> BlockOutput:
#         """
#         Perform operations on Airtable tables.

#         Args:
#             input_data: The input parameters for the block.
#             credentials: The Airtable API credentials.

#         Yields:
#             BlockOutput: The result of the table operation.
#         """
#         try:
#             client = AirtableClient(credentials=credentials)

#             if input_data.operation == "list":
#                 # List all tables in the base
#                 response = client.list_tables(input_data.base_id)
#                 tables = [
#                     AirtableTable(
#                         id=table.id, name=table.name, description=table.description
#                     )
#                     for table in response.tables
#                 ]
#                 yield "tables", tables

#             elif input_data.operation == "get":
#                 # Get a specific table
#                 if not input_data.table_id:
#                     yield "error", "Table ID is required for 'get' operation"
#                     return

#                 table = client.get_table(input_data.base_id, input_data.table_id)
#                 yield "table", AirtableTable(
#                     id=table.id, name=table.name, description=table.description
#                 )

#             elif input_data.operation == "create":
#                 # Create a new table
#                 if not input_data.table_name:
#                     yield "error", "Table name is required for 'create' operation"
#                     return
#                 if not input_data.fields or len(input_data.fields) == 0:
#                     yield "error", "At least one field is required for 'create' operation"
#                     return

#                 table = client.create_table(
#                     input_data.base_id,
#                     input_data.table_name,
#                     input_data.table_description or "",
#                     input_data.fields,
#                 )
#                 yield "table", AirtableTable(
#                     id=table.id, name=table.name, description=table.description
#                 )

#             else:
#                 yield "error", f"Unknown operation: {input_data.operation}"

#         except AirtableAPIException as e:
#             yield "error", f"Airtable API error: {str(e)}"
#         except Exception as e:
#             logger.exception("Error in AirtableTablesBlock")
#             yield "error", f"Error: {str(e)}"


# class AirtableFieldsBlock(Block):
#     """Block for listing, getting, and creating fields in Airtable tables."""

#     class Input(BlockSchema):
#         base_id: str = SchemaField(
#             description="The ID of the Airtable base",
#             placeholder="appXXXXXXXXXXXXXX",
#         )
#         table_id: str = SchemaField(
#             description="The ID of the table",
#             placeholder="tblXXXXXXXXXXXXXX",
#         )
#         operation: str = SchemaField(
#             description="The operation to perform on fields",
#             placeholder="list",
#             choices=["list", "get", "create"],
#         )
#         field_id: Optional[str] = SchemaField(
#             description="The ID of the field (required for 'get' operation)",
#             placeholder="fldXXXXXXXXXXXXXX",
#             advanced=True,
#         )
#         field_name: Optional[str] = SchemaField(
#             description="The name of the new field (required for 'create' operation)",
#             placeholder="My New Field",
#             advanced=True,
#         )
#         field_type: Optional[str] = SchemaField(
#             description="The type of the new field (required for 'create' operation)",
#             placeholder="text",
#             advanced=True,
#             choices=[
#                 "text",
#                 "number",
#                 "checkbox",
#                 "singleSelect",
#                 "multipleSelects",
#                 "date",
#                 "dateTime",
#                 "attachment",
#                 "link",
#                 "multipleRecordLinks",
#                 "formula",
#                 "rollup",
#                 "count",
#                 "lookup",
#                 "currency",
#                 "percent",
#                 "duration",
#                 "rating",
#                 "richText",
#                 "barcode",
#                 "button",
#             ],
#         )
#         credentials: AirtableCredentialsInput = CredentialsField(
#             description="The credentials for the Airtable API"
#         )

#     class Output(BlockSchema):
#         fields: Optional[List[AirtableField]] = SchemaField(
#             description="List of fields in the table"
#         )
#         field: Optional[AirtableField] = SchemaField(
#             description="The retrieved or created field"
#         )
#         error: Optional[str] = SchemaField(description="Error message if any")

#     def __init__(self):
#         super().__init__(
#             id="c27a6a11-8c09-4f8c-afeb-82c7a0c81857",
#             description="List, get, or create fields in an Airtable table",
#             categories={BlockCategory.DATA},
#             input_schema=AirtableFieldsBlock.Input,
#             output_schema=AirtableFieldsBlock.Output,
#             test_input={
#                 "base_id": "appXXXXXXXXXXXXXX",
#                 "table_id": "tblXXXXXXXXXXXXXX",
#                 "operation": "list",
#                 "credentials": TEST_CREDENTIALS_INPUT,
#             },
#             test_output=[
#                 ("fields", [AirtableField(id="fld123", name="Name", type="text")])
#             ],
#             test_mock={
#                 "list_fields": lambda *args, **kwargs: [
#                     {"id": "fld123", "name": "Name", "type": "text"}
#                 ]
#             },
#             test_credentials=TEST_CREDENTIALS,
#         )

#     def run(
#         self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
#     ) -> BlockOutput:
#         """
#         Perform operations on Airtable fields.

#         Args:
#             input_data: The input parameters for the block.
#             credentials: The Airtable API credentials.

#         Yields:
#             BlockOutput: The result of the field operation.
#         """
#         try:
#             client = AirtableClient(credentials=credentials)

#             if input_data.operation == "list":
#                 # List all fields in the table
#                 fields_list = client.list_fields(
#                     input_data.base_id, input_data.table_id
#                 )
#                 fields = [
#                     AirtableField(id=field.id, name=field.name, type=field.type)
#                     for field in fields_list
#                 ]
#                 yield "fields", fields

#             elif input_data.operation == "get":
#                 # Get a specific field
#                 if not input_data.field_id:
#                     yield "error", "Field ID is required for 'get' operation"
#                     return

#                 field = client.get_field(
#                     input_data.base_id, input_data.table_id, input_data.field_id
#                 )
#                 yield "field", AirtableField(
#                     id=field.id, name=field.name, type=field.type
#                 )

#             elif input_data.operation == "create":
#                 # Create a new field
#                 if not input_data.field_name:
#                     yield "error", "Field name is required for 'create' operation"
#                     return
#                 if not input_data.field_type:
#                     yield "error", "Field type is required for 'create' operation"
#                     return

#                 field = client.create_field(
#                     input_data.base_id,
#                     input_data.table_id,
#                     input_data.field_name,
#                     input_data.field_type,
#                 )
#                 yield "field", AirtableField(
#                     id=field.id, name=field.name, type=field.type
#                 )

#             else:
#                 yield "error", f"Unknown operation: {input_data.operation}"

#         except AirtableAPIException as e:
#             yield "error", f"Airtable API error: {str(e)}"
#         except Exception as e:
#             logger.exception("Error in AirtableFieldsBlock")
#             yield "error", f"Error: {str(e)}"


# class OperationChoices(enum.Enum):
#     LIST = "LIST"
#     GET = "GET"
#     CREATE = "CREATE"
#     UPDATE = "UPDATE"
#     DELETE = "DELETE"


# class AirtableRecordsBlock(Block):
#     """Block for creating, reading, updating, and deleting records in Airtable."""

#     class Input(BlockSchema):
#         base_id: str = SchemaField(
#             description="The ID of the Airtable base",
#             placeholder="appXXXXXXXXXXXXXX",
#         )
#         table_id: str = SchemaField(
#             description="The ID of the table",
#             placeholder="tblXXXXXXXXXXXXXX",
#         )
#         operation: OperationChoices = SchemaField(
#             description="The operation to perform on records",
#             default=OperationChoices.LIST,
#         )
#         record_id: Optional[str] = SchemaField(
#             description="The ID of the record (required for 'get', 'update', and 'delete' operations)",
#             placeholder="recXXXXXXXXXXXXXX",
#             advanced=True,
#         )
#         filter_formula: Optional[str] = SchemaField(
#             description="Filter formula for listing records (optional for 'list' operation)",
#             placeholder="{Field}='Value'",
#             advanced=True,
#         )
#         fields: Optional[Dict] = SchemaField(
#             description="The field values (required for 'create' and 'update' operations)",
#             placeholder='{"Name": "John Doe", "Email": "john@example.com"}',
#             advanced=True,
#         )
#         credentials: AirtableCredentialsInput = CredentialsField(
#             description="The credentials for the Airtable API"
#         )

#     class Output(BlockSchema):
#         records: Optional[List[AirtableRecord]] = SchemaField(
#             description="List of records in the table"
#         )
#         record: Optional[AirtableRecord] = SchemaField(
#             description="The retrieved, created, or updated record"
#         )
#         success: Optional[bool] = SchemaField(
#             description="Success status for delete operation"
#         )
#         error: Optional[str] = SchemaField(description="Error message if any")
