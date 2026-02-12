from os import getenv
from uuid import uuid4

import pytest

from backend.sdk import APIKeyCredentials, SecretStr

from ._api import (
    TableFieldType,
    WebhookFilters,
    WebhookSpecification,
    create_base,
    create_field,
    create_record,
    create_table,
    create_webhook,
    delete_multiple_records,
    delete_record,
    delete_webhook,
    get_record,
    list_bases,
    list_records,
    list_webhook_payloads,
    update_field,
    update_multiple_records,
    update_record,
    update_table,
)


@pytest.mark.asyncio
async def test_create_update_table():

    key = getenv("AIRTABLE_API_KEY")
    if not key:
        return pytest.skip("AIRTABLE_API_KEY is not set")

    credentials = APIKeyCredentials(
        provider="airtable",
        api_key=SecretStr(key),
    )
    postfix = uuid4().hex[:4]
    workspace_id = "wsphuHmfllg7V3Brd"
    response = await create_base(credentials, workspace_id, "API Testing Base")
    assert response is not None, f"Checking create base response: {response}"
    assert (
        response.get("id") is not None
    ), f"Checking create base response id: {response}"
    base_id = response.get("id")
    assert base_id is not None, f"Checking create base response id: {base_id}"

    response = await list_bases(credentials)
    assert response is not None, f"Checking list bases response: {response}"
    assert "API Testing Base" in [
        base.get("name") for base in response.get("bases", [])
    ], f"Checking list bases response bases: {response}"

    table_name = f"test_table_{postfix}"
    table_fields = [{"name": "test_field", "type": "singleLineText"}]
    table = await create_table(credentials, base_id, table_name, table_fields)
    assert table.get("name") == table_name

    table_id = table.get("id")

    assert table_id is not None

    table_name = f"test_table_updated_{postfix}"
    table_description = "test_description_updated"
    table = await update_table(
        credentials,
        base_id,
        table_id,
        table_name=table_name,
        table_description=table_description,
    )
    assert table.get("name") == table_name
    assert table.get("description") == table_description


@pytest.mark.asyncio
async def test_invalid_field_type():

    key = getenv("AIRTABLE_API_KEY")
    if not key:
        return pytest.skip("AIRTABLE_API_KEY is not set")

    credentials = APIKeyCredentials(
        provider="airtable",
        api_key=SecretStr(key),
    )
    postfix = uuid4().hex[:4]
    base_id = "appZPxegHEU3kDc1S"
    table_name = f"test_table_{postfix}"
    table_fields = [{"name": "test_field", "type": "notValid"}]
    with pytest.raises(AssertionError):
        await create_table(credentials, base_id, table_name, table_fields)


@pytest.mark.asyncio
async def test_create_and_update_field():
    key = getenv("AIRTABLE_API_KEY")
    if not key:
        return pytest.skip("AIRTABLE_API_KEY is not set")

    credentials = APIKeyCredentials(
        provider="airtable",
        api_key=SecretStr(key),
    )
    postfix = uuid4().hex[:4]
    base_id = "appZPxegHEU3kDc1S"
    table_name = f"test_table_{postfix}"
    table_fields = [{"name": "test_field", "type": "singleLineText"}]
    table = await create_table(credentials, base_id, table_name, table_fields)
    assert table.get("name") == table_name

    table_id = table.get("id")

    assert table_id is not None

    field_name = f"test_field_{postfix}"
    field_type = TableFieldType.SINGLE_LINE_TEXT
    field = await create_field(credentials, base_id, table_id, field_type, field_name)
    assert field.get("name") == field_name

    field_id = field.get("id")

    assert field_id is not None
    assert isinstance(field_id, str)

    field_name = f"test_field_updated_{postfix}"
    field = await update_field(credentials, base_id, table_id, field_id, field_name)
    assert field.get("name") == field_name

    field_description = "test_description_updated"
    field = await update_field(
        credentials, base_id, table_id, field_id, description=field_description
    )
    assert field.get("description") == field_description


@pytest.mark.asyncio
async def test_record_management():
    key = getenv("AIRTABLE_API_KEY")
    if not key:
        return pytest.skip("AIRTABLE_API_KEY is not set")

    credentials = APIKeyCredentials(
        provider="airtable",
        api_key=SecretStr(key),
    )
    postfix = uuid4().hex[:4]
    base_id = "appZPxegHEU3kDc1S"
    table_name = f"test_table_{postfix}"
    table_fields = [{"name": "test_field", "type": "singleLineText"}]
    table = await create_table(credentials, base_id, table_name, table_fields)
    assert table.get("name") == table_name

    table_id = table.get("id")
    assert table_id is not None

    # Create a record
    record_fields = {"test_field": "test_value"}
    record = await create_record(credentials, base_id, table_id, fields=record_fields)
    fields = record.get("fields")
    assert fields is not None
    assert isinstance(fields, dict)
    assert fields.get("test_field") == "test_value"

    record_id = record.get("id")

    assert record_id is not None
    assert isinstance(record_id, str)

    # Get a record
    record = await get_record(credentials, base_id, table_id, record_id)
    fields = record.get("fields")
    assert fields is not None
    assert isinstance(fields, dict)
    assert fields.get("test_field") == "test_value"

    # Updata a record
    record_fields = {"test_field": "test_value_updated"}
    record = await update_record(
        credentials, base_id, table_id, record_id, fields=record_fields
    )
    fields = record.get("fields")
    assert fields is not None
    assert isinstance(fields, dict)
    assert fields.get("test_field") == "test_value_updated"

    # Delete a record
    record = await delete_record(credentials, base_id, table_id, record_id)
    assert record is not None
    assert record.get("id") == record_id
    assert record.get("deleted")

    # Create 2 records
    records = [
        {"fields": {"test_field": "test_value_1"}},
        {"fields": {"test_field": "test_value_2"}},
    ]
    response = await create_record(credentials, base_id, table_id, records=records)
    created_records = response.get("records")
    assert created_records is not None
    assert isinstance(created_records, list)
    assert len(created_records) == 2, f"Created records: {created_records}"
    first_record = created_records[0]  # type: ignore
    second_record = created_records[1]  # type: ignore
    first_record_id = first_record.get("id")
    second_record_id = second_record.get("id")
    assert first_record_id is not None
    assert second_record_id is not None
    assert first_record_id != second_record_id
    first_fields = first_record.get("fields")
    second_fields = second_record.get("fields")
    assert first_fields is not None
    assert second_fields is not None
    assert first_fields.get("test_field") == "test_value_1"  # type: ignore
    assert second_fields.get("test_field") == "test_value_2"  # type: ignore

    # List records
    response = await list_records(credentials, base_id, table_id)
    records = response.get("records")
    assert records is not None
    assert len(records) == 2, f"Records: {records}"
    assert isinstance(records, list), f"Type of records: {type(records)}"

    # Update multiple records
    records = [
        {"id": first_record_id, "fields": {"test_field": "test_value_1_updated"}},
        {"id": second_record_id, "fields": {"test_field": "test_value_2_updated"}},
    ]
    response = await update_multiple_records(
        credentials, base_id, table_id, records=records
    )
    updated_records = response.get("records")
    assert updated_records is not None
    assert len(updated_records) == 2, f"Updated records: {updated_records}"
    assert isinstance(
        updated_records, list
    ), f"Type of updated records: {type(updated_records)}"
    first_updated = updated_records[0]  # type: ignore
    second_updated = updated_records[1]  # type: ignore
    first_updated_fields = first_updated.get("fields")
    second_updated_fields = second_updated.get("fields")
    assert first_updated_fields is not None
    assert second_updated_fields is not None
    assert first_updated_fields.get("test_field") == "test_value_1_updated"  # type: ignore
    assert second_updated_fields.get("test_field") == "test_value_2_updated"  # type: ignore

    # Delete multiple records
    assert isinstance(first_record_id, str)
    assert isinstance(second_record_id, str)
    response = await delete_multiple_records(
        credentials, base_id, table_id, records=[first_record_id, second_record_id]
    )
    deleted_records = response.get("records")
    assert deleted_records is not None
    assert len(deleted_records) == 2, f"Deleted records: {deleted_records}"
    assert isinstance(
        deleted_records, list
    ), f"Type of deleted records: {type(deleted_records)}"
    first_deleted = deleted_records[0]  # type: ignore
    second_deleted = deleted_records[1]  # type: ignore
    assert first_deleted.get("deleted")
    assert second_deleted.get("deleted")


@pytest.mark.asyncio
async def test_webhook_management():
    key = getenv("AIRTABLE_API_KEY")
    if not key:
        return pytest.skip("AIRTABLE_API_KEY is not set")

    credentials = APIKeyCredentials(
        provider="airtable",
        api_key=SecretStr(key),
    )
    postfix = uuid4().hex[:4]
    base_id = "appZPxegHEU3kDc1S"
    table_name = f"test_table_{postfix}"
    table_fields = [{"name": "test_field", "type": "singleLineText"}]
    table = await create_table(credentials, base_id, table_name, table_fields)
    assert table.get("name") == table_name

    table_id = table.get("id")
    assert table_id is not None
    webhook_specification = WebhookSpecification(
        filters=WebhookFilters(
            dataTypes=["tableData", "tableFields", "tableMetadata"],
            changeTypes=["add", "update", "remove"],
        )
    )
    response = await create_webhook(credentials, base_id, webhook_specification)
    assert response is not None, f"Checking create webhook response: {response}"
    assert (
        response.get("id") is not None
    ), f"Checking create webhook response id: {response}"
    assert (
        response.get("macSecretBase64") is not None
    ), f"Checking create webhook response macSecretBase64: {response}"

    webhook_id = response.get("id")
    assert webhook_id is not None, f"Webhook ID: {webhook_id}"
    assert isinstance(webhook_id, str)

    response = await create_record(
        credentials, base_id, table_id, fields={"test_field": "test_value"}
    )
    assert response is not None, f"Checking create record response: {response}"
    assert (
        response.get("id") is not None
    ), f"Checking create record response id: {response}"
    fields = response.get("fields")
    assert fields is not None, f"Checking create record response fields: {response}"
    assert (
        fields.get("test_field") == "test_value"
    ), f"Checking create record response fields test_field: {response}"

    response = await list_webhook_payloads(credentials, base_id, webhook_id)
    assert response is not None, f"Checking list webhook payloads response: {response}"

    response = await delete_webhook(credentials, base_id, webhook_id)
