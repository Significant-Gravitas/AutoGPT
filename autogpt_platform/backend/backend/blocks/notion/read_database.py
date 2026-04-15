from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import OAuth2Credentials, SchemaField

from ._api import NotionClient, parse_rich_text
from ._auth import (
    NOTION_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    NotionCredentialsField,
    NotionCredentialsInput,
)


class NotionReadDatabaseBlock(Block):
    """Query a Notion database and retrieve entries with their properties."""

    class Input(BlockSchemaInput):
        credentials: NotionCredentialsInput = NotionCredentialsField()
        database_id: str = SchemaField(
            description="Notion database ID. Must be accessible by the connected integration.",
        )
        filter_property: Optional[str] = SchemaField(
            description="Property name to filter by (e.g., 'Status', 'Priority')",
            default=None,
        )
        filter_value: Optional[str] = SchemaField(
            description="Value to filter for in the specified property", default=None
        )
        sort_property: Optional[str] = SchemaField(
            description="Property name to sort by", default=None
        )
        sort_direction: Optional[str] = SchemaField(
            description="Sort direction: 'ascending' or 'descending'",
            default="ascending",
        )
        limit: int = SchemaField(
            description="Maximum number of entries to retrieve",
            default=100,
            ge=1,
            le=100,
        )

    class Output(BlockSchemaOutput):
        entries: List[Dict[str, Any]] = SchemaField(
            description="List of database entries with their properties."
        )
        entry: Dict[str, Any] = SchemaField(
            description="Individual database entry (yields one per entry found)."
        )
        entry_ids: List[str] = SchemaField(
            description="List of entry IDs for batch operations."
        )
        entry_id: str = SchemaField(
            description="Individual entry ID (yields one per entry found)."
        )
        count: int = SchemaField(description="Number of entries retrieved.")
        database_title: str = SchemaField(description="Title of the database.")

    def __init__(self):
        super().__init__(
            id="fcd53135-88c9-4ba3-be50-cc6936286e6c",
            description="Query a Notion database with optional filtering and sorting, returning structured entries.",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=NotionReadDatabaseBlock.Input,
            output_schema=NotionReadDatabaseBlock.Output,
            disabled=not NOTION_OAUTH_IS_CONFIGURED,
            test_input={
                "database_id": "00000000-0000-0000-0000-000000000000",
                "limit": 10,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                (
                    "entries",
                    [{"Name": "Test Entry", "Status": "Active", "_id": "test-123"}],
                ),
                ("entry_ids", ["test-123"]),
                (
                    "entry",
                    {"Name": "Test Entry", "Status": "Active", "_id": "test-123"},
                ),
                ("entry_id", "test-123"),
                ("count", 1),
                ("database_title", "Test Database"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "query_database": lambda *args, **kwargs: (
                    [{"Name": "Test Entry", "Status": "Active", "_id": "test-123"}],
                    1,
                    "Test Database",
                )
            },
        )

    @staticmethod
    def _parse_property_value(prop: dict) -> Any:
        """Parse a Notion property value into a simple Python type."""
        prop_type = prop.get("type")

        if prop_type == "title":
            return parse_rich_text(prop.get("title", []))
        elif prop_type == "rich_text":
            return parse_rich_text(prop.get("rich_text", []))
        elif prop_type == "number":
            return prop.get("number")
        elif prop_type == "select":
            select = prop.get("select")
            return select.get("name") if select else None
        elif prop_type == "multi_select":
            return [item.get("name") for item in prop.get("multi_select", [])]
        elif prop_type == "date":
            date = prop.get("date")
            if date:
                return date.get("start")
            return None
        elif prop_type == "checkbox":
            return prop.get("checkbox", False)
        elif prop_type == "url":
            return prop.get("url")
        elif prop_type == "email":
            return prop.get("email")
        elif prop_type == "phone_number":
            return prop.get("phone_number")
        elif prop_type == "people":
            return [
                person.get("name", person.get("id"))
                for person in prop.get("people", [])
            ]
        elif prop_type == "files":
            files = prop.get("files", [])
            return [
                f.get(
                    "name",
                    f.get("external", {}).get("url", f.get("file", {}).get("url")),
                )
                for f in files
            ]
        elif prop_type == "relation":
            return [rel.get("id") for rel in prop.get("relation", [])]
        elif prop_type == "formula":
            formula = prop.get("formula", {})
            return formula.get(formula.get("type"))
        elif prop_type == "rollup":
            rollup = prop.get("rollup", {})
            return rollup.get(rollup.get("type"))
        elif prop_type == "created_time":
            return prop.get("created_time")
        elif prop_type == "created_by":
            return prop.get("created_by", {}).get(
                "name", prop.get("created_by", {}).get("id")
            )
        elif prop_type == "last_edited_time":
            return prop.get("last_edited_time")
        elif prop_type == "last_edited_by":
            return prop.get("last_edited_by", {}).get(
                "name", prop.get("last_edited_by", {}).get("id")
            )
        else:
            # Return the raw value for unknown types
            return prop

    @staticmethod
    def _build_filter(property_name: str, value: str) -> dict:
        """Build a simple filter object for a property."""
        # This is a simplified filter - in reality, you'd need to know the property type
        # For now, we'll try common filter types
        return {
            "or": [
                {"property": property_name, "rich_text": {"contains": value}},
                {"property": property_name, "title": {"contains": value}},
                {"property": property_name, "select": {"equals": value}},
                {"property": property_name, "multi_select": {"contains": value}},
            ]
        }

    @staticmethod
    async def query_database(
        credentials: OAuth2Credentials,
        database_id: str,
        filter_property: Optional[str] = None,
        filter_value: Optional[str] = None,
        sort_property: Optional[str] = None,
        sort_direction: str = "ascending",
        limit: int = 100,
    ) -> tuple[List[Dict[str, Any]], int, str]:
        """
        Query a Notion database and parse the results.

        Returns:
            Tuple of (entries_list, count, database_title)
        """
        client = NotionClient(credentials)

        # Build filter if specified
        filter_obj = None
        if filter_property and filter_value:
            filter_obj = NotionReadDatabaseBlock._build_filter(
                filter_property, filter_value
            )

        # Build sorts if specified
        sorts = None
        if sort_property:
            sorts = [{"property": sort_property, "direction": sort_direction}]

        # Query the database
        result = await client.query_database(
            database_id, filter_obj=filter_obj, sorts=sorts, page_size=limit
        )

        # Parse the entries
        entries = []
        for page in result.get("results", []):
            entry = {}
            properties = page.get("properties", {})

            for prop_name, prop_value in properties.items():
                entry[prop_name] = NotionReadDatabaseBlock._parse_property_value(
                    prop_value
                )

            # Add metadata
            entry["_id"] = page.get("id")
            entry["_url"] = page.get("url")
            entry["_created_time"] = page.get("created_time")
            entry["_last_edited_time"] = page.get("last_edited_time")

            entries.append(entry)

        # Get database title (we need to make a separate call for this)
        try:
            database_url = f"https://api.notion.com/v1/databases/{database_id}"
            db_response = await client.requests.get(
                database_url, headers=client.headers
            )
            if db_response.ok:
                db_data = db_response.json()
                db_title = parse_rich_text(db_data.get("title", []))
            else:
                db_title = "Unknown Database"
        except Exception:
            db_title = "Unknown Database"

        return entries, len(entries), db_title

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            entries, count, db_title = await self.query_database(
                credentials,
                input_data.database_id,
                input_data.filter_property,
                input_data.filter_value,
                input_data.sort_property,
                input_data.sort_direction or "ascending",
                input_data.limit,
            )
            # Yield the complete list for batch operations
            yield "entries", entries

            # Extract and yield IDs as a list for batch operations
            entry_ids = [entry["_id"] for entry in entries if "_id" in entry]
            yield "entry_ids", entry_ids

            # Yield each individual entry and its ID for single connections
            for entry in entries:
                yield "entry", entry
                if "_id" in entry:
                    yield "entry_id", entry["_id"]

            yield "count", count
            yield "database_title", db_title
        except Exception as e:
            yield "error", str(e) if str(e) else "Unknown error"
