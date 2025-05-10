# """
# API module for Airtable API integration.

# This module provides a client for interacting with the Airtable API,
# including methods for working with tables, fields, records, and webhooks.
# """

# from json import JSONDecodeError
# from typing import Any, Dict, List, Optional

# from pydantic import BaseModel

# from backend.data.model import APIKeyCredentials
# from backend.util.request import Requests


# class AirtableAPIException(Exception):
#     def __init__(self, message: str, status_code: int):
#         super().__init__(message)
#         self.status_code = status_code


# # Response Models
# class TableField(BaseModel):
#     id: str
#     name: str
#     type: str
#     options: Optional[Dict[str, Any]] = None


# class Table(BaseModel):
#     id: str
#     name: str
#     description: Optional[str] = None
#     fields: List[TableField]


# class Record(BaseModel):
#     id: str
#     fields: Dict[str, Any]
#     createdTime: Optional[str] = None


# class RecordAttachment(BaseModel):
#     id: str
#     url: str
#     filename: str
#     size: Optional[int] = None
#     type: Optional[str] = None


# class Webhook(BaseModel):
#     id: str
#     url: str
#     event: str
#     notification_url: Optional[str] = None
#     active: bool


# class ListTablesResponse(BaseModel):
#     tables: List[Table]


# class ListRecordsResponse(BaseModel):
#     records: List[Record]
#     offset: Optional[str] = None


# class ListAttachmentsResponse(BaseModel):
#     attachments: List[RecordAttachment]
#     offset: Optional[str] = None


# class ListWebhooksResponse(BaseModel):
#     webhooks: List[Webhook]
#     offset: Optional[str] = None


# class AirtableClient:
#     """Client for the Airtable API"""

#     API_BASE_URL = "https://api.airtable.com/v0"

#     def __init__(
#         self,
#         credentials: Optional[APIKeyCredentials] = None,
#         custom_requests: Optional[Requests] = None,
#     ):
#         if custom_requests:
#             self._requests = custom_requests
#         else:
#             headers: dict[str, str] = {
#                 "Content-Type": "application/json",
#             }
#             if credentials:
#                 headers["Authorization"] = (
#                     f"Bearer {credentials.api_key.get_secret_value()}"
#                 )

#             self._requests = Requests(
#                 extra_headers=headers,
#                 raise_for_status=False,
#             )

#     @staticmethod
#     def _handle_response(response) -> Any:
#         """
#         Handles API response and checks for errors.

#         Args:
#             response: The response object from the request.

#         Returns:
#             The parsed JSON response data.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         if not response.ok:
#             try:
#                 error_data = response.json()
#                 error_message = error_data.get("error", {}).get("message", "")
#             except JSONDecodeError:
#                 error_message = response.text

#             raise AirtableAPIException(
#                 f"Airtable API request failed ({response.status_code}): {error_message}",
#                 response.status_code,
#             )

#         return response.json()

#     # Table Methods
#     def list_tables(self, base_id: str) -> ListTablesResponse:
#         """
#         List all tables in a base.

#         Args:
#             base_id: The ID of the base to list tables from.

#         Returns:
#             ListTablesResponse: Object containing the list of tables.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         try:
#             response = self._requests.get(f"{self.API_BASE_URL}/bases/{base_id}/tables")
#             data = self._handle_response(response)
#             return ListTablesResponse(**data)
#         except Exception as e:
#             raise AirtableAPIException(f"Failed to list tables: {str(e)}", 500)

#     def get_table(self, base_id: str, table_id: str) -> Table:
#         """
#         Get a specific table schema.

#         Args:
#             base_id: The ID of the base containing the table.
#             table_id: The ID of the table to retrieve.

#         Returns:
#             Table: The table object.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         try:
#             response = self._requests.get(
#                 f"{self.API_BASE_URL}/bases/{base_id}/tables/{table_id}"
#             )
#             data = self._handle_response(response)
#             return Table(**data)
#         except Exception as e:
#             raise AirtableAPIException(f"Failed to get table: {str(e)}", 500)

#     def create_table(
#         self, base_id: str, name: str, description: str, fields: List[Dict[str, Any]]
#     ) -> Table:
#         """
#         Create a new table in a base.

#         Args:
#             base_id: The ID of the base to create the table in.
#             name: The name of the new table.
#             description: The description of the new table.
#             fields: The fields to create in the new table.

#         Returns:
#             Table: The created table object.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         try:
#             payload = {
#                 "name": name,
#                 "description": description,
#                 "fields": fields,
#             }
#             response = self._requests.post(
#                 f"{self.API_BASE_URL}/meta/bases/{base_id}/tables", json=payload
#             )
#             data = self._handle_response(response)
#             return Table(**data)
#         except Exception as e:
#             raise AirtableAPIException(f"Failed to create table: {str(e)}", 500)

#     # Field Methods
#     def list_fields(self, base_id: str, table_id: str) -> List[TableField]:
#         """
#         List all fields in a table.

#         Args:
#             base_id: The ID of the base containing the table.
#             table_id: The ID of the table to list fields from.

#         Returns:
#             List[TableField]: List of field objects.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         try:
#             response = self._requests.get(
#                 f"{self.API_BASE_URL}/bases/{base_id}/tables/{table_id}/fields"
#             )
#             data = self._handle_response(response)
#             return [TableField(**field) for field in data.get("fields", [])]
#         except Exception as e:
#             raise AirtableAPIException(f"Failed to list fields: {str(e)}", 500)

#     def get_field(self, base_id: str, table_id: str, field_id: str) -> TableField:
#         """
#         Get a specific field.

#         Args:
#             base_id: The ID of the base containing the table.
#             table_id: The ID of the table containing the field.
#             field_id: The ID of the field to retrieve.

#         Returns:
#             TableField: The field object.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         try:
#             response = self._requests.get(
#                 f"{self.API_BASE_URL}/bases/{base_id}/tables/{table_id}/fields/{field_id}"
#             )
#             data = self._handle_response(response)
#             return TableField(**data)
#         except Exception as e:
#             raise AirtableAPIException(f"Failed to get field: {str(e)}", 500)

#     def create_field(
#         self,
#         base_id: str,
#         table_id: str,
#         name: str,
#         field_type: str,
#         options: Optional[Dict[str, Any]] = None,
#     ) -> TableField:
#         """
#         Create a new field in a table.

#         Args:
#             base_id: The ID of the base containing the table.
#             table_id: The ID of the table to create the field in.
#             name: The name of the new field.
#             field_type: The type of the new field.
#             options: Optional field type options.

#         Returns:
#             TableField: The created field object.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         try:
#             payload = {
#                 "name": name,
#                 "type": field_type,
#             }
#             if options:
#                 payload["options"] = options

#             response = self._requests.post(
#                 f"{self.API_BASE_URL}/meta/bases/{base_id}/tables/{table_id}/fields",
#                 json=payload,
#             )
#             data = self._handle_response(response)
#             return TableField(**data)
#         except Exception as e:
#             raise AirtableAPIException(f"Failed to create field: {str(e)}", 500)

#     # Record Methods
#     def list_records(
#         self,
#         base_id: str,
#         table_id: str,
#         filter_formula: Optional[str] = None,
#         offset: Optional[str] = None,
#     ) -> ListRecordsResponse:
#         """
#         List records in a table, with optional filtering.

#         Args:
#             base_id: The ID of the base containing the table.
#             table_id: The ID of the table to list records from.
#             filter_formula: Optional formula to filter records.
#             offset: Optional pagination offset.

#         Returns:
#             ListRecordsResponse: Object containing the list of records.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         try:
#             params = {}
#             if filter_formula:
#                 params["filterByFormula"] = filter_formula
#             if offset:
#                 params["offset"] = offset

#             response = self._requests.get(
#                 f"{self.API_BASE_URL}/bases/{base_id}/tables/{table_id}/records",
#                 params=params,
#             )
#             data = self._handle_response(response)
#             return ListRecordsResponse(**data)
#         except Exception as e:
#             raise AirtableAPIException(f"Failed to list records: {str(e)}", 500)

#     def get_record(self, base_id: str, table_id: str, record_id: str) -> Record:
#         """
#         Get a specific record.

#         Args:
#             base_id: The ID of the base containing the table.
#             table_id: The ID of the table containing the record.
#             record_id: The ID of the record to retrieve.

#         Returns:
#             Record: The record object.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         try:
#             response = self._requests.get(
#                 f"{self.API_BASE_URL}/bases/{base_id}/tables/{table_id}/records/{record_id}"
#             )
#             data = self._handle_response(response)
#             return Record(**data)
#         except Exception as e:
#             raise AirtableAPIException(f"Failed to get record: {str(e)}", 500)

#     def create_record(
#         self, base_id: str, table_id: str, fields: Dict[str, Any]
#     ) -> Record:
#         """
#         Create a new record in a table.

#         Args:
#             base_id: The ID of the base containing the table.
#             table_id: The ID of the table to create the record in.
#             fields: The field values for the new record.

#         Returns:
#             Record: The created record object.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         try:
#             payload = {"fields": fields}
#             response = self._requests.post(
#                 f"{self.API_BASE_URL}/bases/{base_id}/tables/{table_id}/records",
#                 json=payload,
#             )
#             data = self._handle_response(response)
#             return Record(**data)
#         except Exception as e:
#             raise AirtableAPIException(f"Failed to create record: {str(e)}", 500)

#     def update_record(
#         self, base_id: str, table_id: str, record_id: str, fields: Dict[str, Any]
#     ) -> Record:
#         """
#         Update a record in a table.

#         Args:
#             base_id: The ID of the base containing the table.
#             table_id: The ID of the table containing the record.
#             record_id: The ID of the record to update.
#             fields: The field values to update.

#         Returns:
#             Record: The updated record object.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         try:
#             payload = {"fields": fields}
#             response = self._requests.patch(
#                 f"{self.API_BASE_URL}/bases/{base_id}/tables/{table_id}/records/{record_id}",
#                 json=payload,
#             )
#             data = self._handle_response(response)
#             return Record(**data)
#         except Exception as e:
#             raise AirtableAPIException(f"Failed to update record: {str(e)}", 500)

#     def delete_record(self, base_id: str, table_id: str, record_id: str) -> bool:
#         """
#         Delete a record from a table.

#         Args:
#             base_id: The ID of the base containing the table.
#             table_id: The ID of the table containing the record.
#             record_id: The ID of the record to delete.

#         Returns:
#             bool: True if the deletion was successful.

#         Raises:
#             AirtableAPIException: If the API request fails.
#         """
#         try:
#             response = self._requests.delete(
#                 f"{self.API_BASE_URL}/bases/{base_id}/tables/{table_id}/records/{record_id}"
#             )
#             self._handle_response(response)
#             return True
#         except Exception as e:
#             raise AirtableAPIException(f"Failed to delete record: {str(e)}", 500)
