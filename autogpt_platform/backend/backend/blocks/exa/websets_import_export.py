"""
Exa Websets Import/Export Management Blocks

This module provides blocks for importing data into websets from CSV files
and exporting webset data in various formats.
"""

import csv
import json
from enum import Enum
from io import StringIO
from typing import Any, Optional

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

from ._api import ExaApiUrls, build_headers, yield_paginated_results
from ._config import exa


class ImportFormat(str, Enum):
    """Supported import formats."""

    CSV = "csv"
    # JSON = "json"  # Future support


class ImportEntityType(str, Enum):
    """Entity types for imports."""

    COMPANY = "company"
    PERSON = "person"
    ARTICLE = "article"
    RESEARCH_PAPER = "research_paper"
    CUSTOM = "custom"


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    JSON_LINES = "jsonl"


class ExaCreateImportBlock(Block):
    """Create an import to load external data that can be used with websets."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        title: str = SchemaField(
            description="Title for this import",
            placeholder="Customer List Import",
        )
        csv_data: str = SchemaField(
            description="CSV data to import (as a string)",
            placeholder="name,url\nAcme Corp,https://acme.com\nExample Inc,https://example.com",
        )
        entity_type: ImportEntityType = SchemaField(
            default=ImportEntityType.COMPANY,
            description="Type of entities being imported",
        )
        entity_description: Optional[str] = SchemaField(
            default=None,
            description="Description for custom entity type",
            advanced=True,
        )
        identifier_column: int = SchemaField(
            default=0,
            description="Column index containing the identifier (0-based)",
            ge=0,
        )
        url_column: Optional[int] = SchemaField(
            default=None,
            description="Column index containing URLs (optional)",
            ge=0,
            advanced=True,
        )
        metadata: Optional[dict] = SchemaField(
            default=None,
            description="Metadata to attach to the import",
            advanced=True,
        )

    class Output(BlockSchema):
        import_id: str = SchemaField(
            description="The unique identifier for the created import"
        )
        status: str = SchemaField(description="Current status of the import")
        title: str = SchemaField(description="Title of the import")
        count: int = SchemaField(
            description="Number of items in the import",
            default=0,
        )
        entity_type: str = SchemaField(description="Type of entities imported")
        created_at: str = SchemaField(description="When the import was created")
        error: str = SchemaField(
            description="Error message if the import failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="f2a3b4c5-d6e7-8901-2345-678901234567",
            description="Import CSV data to use with websets for targeted searches",
            categories={BlockCategory.DATA},
            input_schema=ExaCreateImportBlock.Input,
            output_schema=ExaCreateImportBlock.Output,
            test_input={
                "title": "Test Import",
                "csv_data": "name,url\nAcme,https://acme.com",
                "entity_type": ImportEntityType.COMPANY,
                "identifier_column": 0,
            },
            test_output=[
                ("import_id", str),
                ("status", str),
                ("title", "Test Import"),
                ("count", int),
                ("entity_type", "company"),
                ("created_at", str),
            ],
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = ExaApiUrls.imports()
        headers = build_headers(
            credentials.api_key.get_secret_value(), include_content_type=True
        )

        try:
            # Parse CSV data to count rows
            csv_reader = csv.reader(StringIO(input_data.csv_data))
            rows = list(csv_reader)
            count = len(rows) - 1 if len(rows) > 1 else 0  # Subtract header row

            # Calculate size in bytes
            size = len(input_data.csv_data.encode("utf-8"))

            # Build the payload
            payload = {
                "title": input_data.title,
                "format": ImportFormat.CSV.value,
                "count": count,
                "size": size,
                "csv": {
                    "identifier": input_data.identifier_column,
                },
            }

            # Add URL column if specified
            if input_data.url_column is not None:
                payload["csv"]["url"] = input_data.url_column

            # Add entity configuration
            entity = {"type": input_data.entity_type.value}
            if (
                input_data.entity_type == ImportEntityType.CUSTOM
                and input_data.entity_description
            ):
                entity["description"] = input_data.entity_description
            payload["entity"] = entity

            # Add metadata if provided
            if input_data.metadata:
                payload["metadata"] = input_data.metadata

            # Note: The actual CSV data would need to be uploaded separately
            # This is a placeholder for the API structure
            # In a real implementation, you'd need to handle file upload

            # Create the import
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()

            yield "import_id", data.get("id", "")
            yield "status", data.get("status", "")
            yield "title", data.get("title", "")
            yield "count", data.get("count", count)
            yield "entity_type", input_data.entity_type.value
            yield "created_at", data.get("createdAt", "")

        except ValueError as e:
            # Re-raise user input validation errors
            raise ValueError(f"Failed to create import: {e}") from e
        # Let all other exceptions propagate naturally


class ExaGetImportBlock(Block):
    """Get the status and details of an import."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        import_id: str = SchemaField(
            description="The ID of the import to retrieve",
            placeholder="import-id",
        )

    class Output(BlockSchema):
        import_id: str = SchemaField(description="The unique identifier for the import")
        status: str = SchemaField(description="Current status of the import")
        title: str = SchemaField(description="Title of the import")
        format: str = SchemaField(description="Format of the imported data")
        entity_type: str = SchemaField(description="Type of entities imported")
        count: int = SchemaField(
            description="Number of items imported",
            default=0,
        )
        failed_reason: Optional[str] = SchemaField(
            description="Reason for failure (if applicable)",
            default=None,
        )
        failed_message: Optional[str] = SchemaField(
            description="Detailed failure message (if applicable)",
            default=None,
        )
        created_at: str = SchemaField(description="When the import was created")
        updated_at: str = SchemaField(description="When the import was last updated")
        metadata: dict = SchemaField(
            description="Metadata attached to the import",
            default_factory=dict,
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="a3b4c5d6-e7f8-9012-3456-789012345678",
            description="Get the status and details of an import",
            categories={BlockCategory.DATA},
            input_schema=ExaGetImportBlock.Input,
            output_schema=ExaGetImportBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = ExaApiUrls.import_(input_data.import_id)
        headers = build_headers(credentials.api_key.get_secret_value())

        response = await Requests().get(url, headers=headers)
        data = response.json()

        # Extract entity type
        entity = data.get("entity", {})
        entity_type = entity.get("type", "unknown")

        yield "import_id", data.get("id", "")
        yield "status", data.get("status", "")
        yield "title", data.get("title", "")
        yield "format", data.get("format", "")
        yield "entity_type", entity_type
        yield "count", data.get("count", 0)
        yield "failed_reason", data.get("failedReason")
        yield "failed_message", data.get("failedMessage")
        yield "created_at", data.get("createdAt", "")
        yield "updated_at", data.get("updatedAt", "")
        yield "metadata", data.get("metadata", {})

        # Let all exceptions propagate naturally
        # The API will return appropriate HTTP errors for invalid import IDs


class ExaListImportsBlock(Block):
    """List all imports with pagination."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        limit: int = SchemaField(
            default=25,
            description="Number of imports to return",
            ge=1,
            le=100,
        )
        cursor: Optional[str] = SchemaField(
            default=None,
            description="Cursor for pagination",
            advanced=True,
        )

    class Output(BlockSchema):
        imports: list[dict] = SchemaField(
            description="List of imports",
            default_factory=list,
        )
        import_item: dict = SchemaField(
            description="Individual import (yielded for each import)",
            default_factory=dict,
        )
        has_more: bool = SchemaField(
            description="Whether there are more imports to paginate through",
            default=False,
        )
        next_cursor: Optional[str] = SchemaField(
            description="Cursor for the next page of results",
            default=None,
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="b4c5d6e7-f8a9-0123-4567-890123456789",
            description="List all imports with pagination support",
            categories={BlockCategory.DATA},
            input_schema=ExaListImportsBlock.Input,
            output_schema=ExaListImportsBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = ExaApiUrls.imports()
        headers = build_headers(credentials.api_key.get_secret_value())

        params: dict[str, Any] = {
            "limit": input_data.limit,
        }
        if input_data.cursor:
            params["cursor"] = input_data.cursor

        response = await Requests().get(url, headers=headers, params=params)
        data = response.json()

        # Yield paginated results using helper
        for key, value in yield_paginated_results(data, "imports", "import_item"):
            yield key, value

        # Let all exceptions propagate naturally


class ExaDeleteImportBlock(Block):
    """Delete an import."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        import_id: str = SchemaField(
            description="The ID of the import to delete",
            placeholder="import-id",
        )

    class Output(BlockSchema):
        import_id: str = SchemaField(description="The ID of the deleted import")
        success: str = SchemaField(
            description="Whether the deletion was successful",
            default="true",
        )
        error: str = SchemaField(
            description="Error message if the deletion failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="c5d6e7f8-a9b0-1234-5678-901234567890",
            description="Delete an import",
            categories={BlockCategory.DATA},
            input_schema=ExaDeleteImportBlock.Input,
            output_schema=ExaDeleteImportBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = ExaApiUrls.import_(input_data.import_id)
        headers = build_headers(credentials.api_key.get_secret_value())

        await Requests().delete(url, headers=headers)

        # API returns 204 No Content on successful deletion
        yield "import_id", input_data.import_id
        yield "success", "true"

        # Let all exceptions propagate naturally
        # The API will return appropriate HTTP errors for invalid operations


class ExaExportWebsetBlock(Block):
    """Export all data from a webset in various formats."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to export",
            placeholder="webset-id-or-external-id",
        )
        format: ExportFormat = SchemaField(
            default=ExportFormat.JSON,
            description="Export format",
        )
        include_content: bool = SchemaField(
            default=True,
            description="Include full content in export",
        )
        include_enrichments: bool = SchemaField(
            default=True,
            description="Include enrichment data in export",
        )
        max_items: int = SchemaField(
            default=1000,
            description="Maximum number of items to export",
            ge=1,
            le=10000,
        )

    class Output(BlockSchema):
        export_data: str = SchemaField(
            description="Exported data in the requested format"
        )
        item_count: int = SchemaField(
            description="Number of items exported",
            default=0,
        )
        total_items: int = SchemaField(
            description="Total number of items in the webset",
            default=0,
        )
        truncated: bool = SchemaField(
            description="Whether the export was truncated due to max_items limit",
            default=False,
        )
        format: str = SchemaField(description="Format of the exported data")
        error: str = SchemaField(
            description="Error message if the export failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="d6e7f8a9-b0c1-2345-6789-012345678901",
            description="Export webset data in JSON, CSV, or JSON Lines format",
            categories={BlockCategory.DATA},
            input_schema=ExaExportWebsetBlock.Input,
            output_schema=ExaExportWebsetBlock.Output,
            test_input={
                "webset_id": "test-webset",
                "format": ExportFormat.JSON,
                "include_content": True,
                "include_enrichments": True,
                "max_items": 10,
            },
            test_output=[
                ("export_data", str),
                ("item_count", int),
                ("total_items", int),
                ("truncated", bool),
                ("format", "json"),
            ],
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        headers = build_headers(credentials.api_key.get_secret_value())

        try:
            all_items = []
            cursor = None
            has_more = True
            batch_size = min(100, input_data.max_items)
            data: dict[str, Any] = {}  # Initialize to handle empty websets

            # Fetch all items up to max_items
            while has_more and len(all_items) < input_data.max_items:
                url = ExaApiUrls.webset_items(input_data.webset_id)
                params: dict[str, Any] = {
                    "limit": min(batch_size, input_data.max_items - len(all_items)),
                }
                if cursor:
                    params["cursor"] = cursor

                response = await Requests().get(url, headers=headers, params=params)
                data = response.json()

                items = data.get("data", [])
                all_items.extend(items)

                has_more = data.get("hasMore", False)
                cursor = data.get("nextCursor")

                if len(all_items) >= input_data.max_items:
                    break

            # Get total count
            total_items = len(all_items)
            if "pagination" in data:
                total_items = data["pagination"].get("total", total_items)

            truncated = len(all_items) >= input_data.max_items and has_more

            # Process items based on include flags
            if not input_data.include_content:
                for item in all_items:
                    item.pop("content", None)

            if not input_data.include_enrichments:
                for item in all_items:
                    item.pop("enrichments", None)

            # Format the export data
            export_data = ""

            if input_data.format == ExportFormat.JSON:
                export_data = json.dumps(all_items, indent=2, default=str)

            elif input_data.format == ExportFormat.JSON_LINES:
                lines = [json.dumps(item, default=str) for item in all_items]
                export_data = "\n".join(lines)

            elif input_data.format == ExportFormat.CSV:
                # Extract all unique keys for CSV headers
                all_keys = set()
                for item in all_items:
                    all_keys.update(self._flatten_dict(item).keys())

                # Create CSV
                output = StringIO()
                writer = csv.DictWriter(output, fieldnames=sorted(all_keys))
                writer.writeheader()

                for item in all_items:
                    flat_item = self._flatten_dict(item)
                    writer.writerow(flat_item)

                export_data = output.getvalue()

            yield "export_data", export_data
            yield "item_count", len(all_items)
            yield "total_items", total_items
            yield "truncated", truncated
            yield "format", input_data.format.value

        except ValueError as e:
            # Re-raise user input validation errors
            raise ValueError(f"Failed to export webset: {e}") from e
        # Let all other exceptions propagate naturally

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = "_") -> dict:
        """Flatten nested dictionaries for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings for CSV
                items.append((new_key, json.dumps(v, default=str)))
            else:
                items.append((new_key, v))
        return dict(items)
