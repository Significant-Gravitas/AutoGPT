"""
Exa Websets Import/Export Management Blocks

This module provides blocks for importing data into websets from CSV files
and exporting webset data in various formats.
"""

import csv
import json
from enum import Enum
from io import StringIO
from typing import Optional, Union

from exa_py import AsyncExa
from exa_py.websets.types import CreateImportResponse
from exa_py.websets.types import Import as SdkImport
from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT


# Mirrored model for stability - don't use SDK types directly in block outputs
class ImportModel(BaseModel):
    """Stable output model mirroring SDK Import."""

    id: str
    status: str
    title: str
    format: str
    entity_type: str
    count: int
    upload_url: Optional[str]  # Only in CreateImportResponse
    upload_valid_until: Optional[str]  # Only in CreateImportResponse
    failed_reason: str
    failed_message: str
    metadata: dict
    created_at: str
    updated_at: str

    @classmethod
    def from_sdk(
        cls, import_obj: Union[SdkImport, CreateImportResponse]
    ) -> "ImportModel":
        """Convert SDK Import or CreateImportResponse to our stable model."""
        # Extract entity type from union (may be None)
        entity_type = "unknown"
        if import_obj.entity:
            entity_dict = import_obj.entity.model_dump(by_alias=True, exclude_none=True)
            entity_type = entity_dict.get("type", "unknown")

        # Handle status enum
        status_str = (
            import_obj.status.value
            if hasattr(import_obj.status, "value")
            else str(import_obj.status)
        )

        # Handle format enum
        format_str = (
            import_obj.format.value
            if hasattr(import_obj.format, "value")
            else str(import_obj.format)
        )

        # Handle failed_reason enum (may be None or enum)
        failed_reason_str = ""
        if import_obj.failed_reason:
            failed_reason_str = (
                import_obj.failed_reason.value
                if hasattr(import_obj.failed_reason, "value")
                else str(import_obj.failed_reason)
            )

        return cls(
            id=import_obj.id,
            status=status_str,
            title=import_obj.title or "",
            format=format_str,
            entity_type=entity_type,
            count=int(import_obj.count or 0),
            upload_url=getattr(
                import_obj, "upload_url", None
            ),  # Only in CreateImportResponse
            upload_valid_until=getattr(
                import_obj, "upload_valid_until", None
            ),  # Only in CreateImportResponse
            failed_reason=failed_reason_str,
            failed_message=import_obj.failed_message or "",
            metadata=import_obj.metadata or {},
            created_at=(
                import_obj.created_at.isoformat() if import_obj.created_at else ""
            ),
            updated_at=(
                import_obj.updated_at.isoformat() if import_obj.updated_at else ""
            ),
        )


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

    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        import_id: str = SchemaField(
            description="The unique identifier for the created import"
        )
        status: str = SchemaField(description="Current status of the import")
        title: str = SchemaField(description="Title of the import")
        count: int = SchemaField(description="Number of items in the import")
        entity_type: str = SchemaField(description="Type of entities imported")
        upload_url: Optional[str] = SchemaField(
            description="Upload URL for CSV data (only if csv_data not provided in request)"
        )
        upload_valid_until: Optional[str] = SchemaField(
            description="Expiration time for upload URL (only if upload_url is provided)"
        )
        created_at: str = SchemaField(description="When the import was created")

    def __init__(self):
        super().__init__(
            id="020a35d8-8a53-4e60-8b60-1de5cbab1df3",
            description="Import CSV data to use with websets for targeted searches",
            categories={BlockCategory.DATA},
            input_schema=ExaCreateImportBlock.Input,
            output_schema=ExaCreateImportBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "title": "Test Import",
                "csv_data": "name,url\nAcme,https://acme.com",
                "entity_type": ImportEntityType.COMPANY,
                "identifier_column": 0,
            },
            test_output=[
                ("import_id", "import-123"),
                ("status", "pending"),
                ("title", "Test Import"),
                ("count", 1),
                ("entity_type", "company"),
                ("upload_url", None),
                ("upload_valid_until", None),
                ("created_at", "2024-01-01T00:00:00"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock=self._create_test_mock(),
        )

    @staticmethod
    def _create_test_mock():
        """Create test mocks for the AsyncExa SDK."""
        from datetime import datetime
        from unittest.mock import MagicMock

        # Create mock SDK import object
        mock_import = MagicMock()
        mock_import.id = "import-123"
        mock_import.status = MagicMock(value="pending")
        mock_import.title = "Test Import"
        mock_import.format = MagicMock(value="csv")
        mock_import.count = 1
        mock_import.upload_url = None
        mock_import.upload_valid_until = None
        mock_import.failed_reason = None
        mock_import.failed_message = ""
        mock_import.metadata = {}
        mock_import.created_at = datetime.fromisoformat("2024-01-01T00:00:00")
        mock_import.updated_at = datetime.fromisoformat("2024-01-01T00:00:00")

        # Mock entity
        mock_entity = MagicMock()
        mock_entity.model_dump = MagicMock(return_value={"type": "company"})
        mock_import.entity = mock_entity

        return {
            "_get_client": lambda *args, **kwargs: MagicMock(
                websets=MagicMock(
                    imports=MagicMock(create=lambda *args, **kwargs: mock_import)
                )
            )
        }

    def _get_client(self, api_key: str) -> AsyncExa:
        """Get Exa client (separated for testing)."""
        return AsyncExa(api_key=api_key)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        aexa = self._get_client(credentials.api_key.get_secret_value())

        csv_reader = csv.reader(StringIO(input_data.csv_data))
        rows = list(csv_reader)
        count = len(rows) - 1 if len(rows) > 1 else 0

        size = len(input_data.csv_data.encode("utf-8"))

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

        sdk_import = aexa.websets.imports.create(
            params=payload, csv_data=input_data.csv_data
        )

        import_obj = ImportModel.from_sdk(sdk_import)

        yield "import_id", import_obj.id
        yield "status", import_obj.status
        yield "title", import_obj.title
        yield "count", import_obj.count
        yield "entity_type", import_obj.entity_type
        yield "upload_url", import_obj.upload_url
        yield "upload_valid_until", import_obj.upload_valid_until
        yield "created_at", import_obj.created_at


class ExaGetImportBlock(Block):
    """Get the status and details of an import."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        import_id: str = SchemaField(
            description="The ID of the import to retrieve",
            placeholder="import-id",
        )

    class Output(BlockSchemaOutput):
        import_id: str = SchemaField(description="The unique identifier for the import")
        status: str = SchemaField(description="Current status of the import")
        title: str = SchemaField(description="Title of the import")
        format: str = SchemaField(description="Format of the imported data")
        entity_type: str = SchemaField(description="Type of entities imported")
        count: int = SchemaField(description="Number of items imported")
        upload_url: Optional[str] = SchemaField(
            description="Upload URL for CSV data (if import not yet uploaded)"
        )
        upload_valid_until: Optional[str] = SchemaField(
            description="Expiration time for upload URL (if applicable)"
        )
        failed_reason: Optional[str] = SchemaField(
            description="Reason for failure (if applicable)"
        )
        failed_message: Optional[str] = SchemaField(
            description="Detailed failure message (if applicable)"
        )
        created_at: str = SchemaField(description="When the import was created")
        updated_at: str = SchemaField(description="When the import was last updated")
        metadata: dict = SchemaField(description="Metadata attached to the import")

    def __init__(self):
        super().__init__(
            id="236663c8-a8dc-45f7-a050-2676bb0a3dd2",
            description="Get the status and details of an import",
            categories={BlockCategory.DATA},
            input_schema=ExaGetImportBlock.Input,
            output_schema=ExaGetImportBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        sdk_import = aexa.websets.imports.get(import_id=input_data.import_id)

        import_obj = ImportModel.from_sdk(sdk_import)

        # Yield all fields
        yield "import_id", import_obj.id
        yield "status", import_obj.status
        yield "title", import_obj.title
        yield "format", import_obj.format
        yield "entity_type", import_obj.entity_type
        yield "count", import_obj.count
        yield "upload_url", import_obj.upload_url
        yield "upload_valid_until", import_obj.upload_valid_until
        yield "failed_reason", import_obj.failed_reason
        yield "failed_message", import_obj.failed_message
        yield "created_at", import_obj.created_at
        yield "updated_at", import_obj.updated_at
        yield "metadata", import_obj.metadata


class ExaListImportsBlock(Block):
    """List all imports with pagination."""

    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        imports: list[dict] = SchemaField(description="List of imports")
        import_item: dict = SchemaField(
            description="Individual import (yielded for each import)"
        )
        has_more: bool = SchemaField(
            description="Whether there are more imports to paginate through"
        )
        next_cursor: Optional[str] = SchemaField(
            description="Cursor for the next page of results"
        )

    def __init__(self):
        super().__init__(
            id="65323630-f7e9-4692-a624-184ba14c0686",
            description="List all imports with pagination support",
            categories={BlockCategory.DATA},
            input_schema=ExaListImportsBlock.Input,
            output_schema=ExaListImportsBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        response = aexa.websets.imports.list(
            cursor=input_data.cursor,
            limit=input_data.limit,
        )

        # Convert SDK imports to our stable models
        imports = [ImportModel.from_sdk(i) for i in response.data]

        yield "imports", [i.model_dump() for i in imports]

        for import_obj in imports:
            yield "import_item", import_obj.model_dump()

        yield "has_more", response.has_more
        yield "next_cursor", response.next_cursor


class ExaDeleteImportBlock(Block):
    """Delete an import."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        import_id: str = SchemaField(
            description="The ID of the import to delete",
            placeholder="import-id",
        )

    class Output(BlockSchemaOutput):
        import_id: str = SchemaField(description="The ID of the deleted import")
        success: str = SchemaField(description="Whether the deletion was successful")

    def __init__(self):
        super().__init__(
            id="81ae30ed-c7ba-4b5d-8483-b726846e570c",
            description="Delete an import",
            categories={BlockCategory.DATA},
            input_schema=ExaDeleteImportBlock.Input,
            output_schema=ExaDeleteImportBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        deleted_import = aexa.websets.imports.delete(import_id=input_data.import_id)

        yield "import_id", deleted_import.id
        yield "success", "true"


class ExaExportWebsetBlock(Block):
    """Export all data from a webset in various formats."""

    class Input(BlockSchemaInput):
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
            default=100,
            description="Maximum number of items to export",
            ge=1,
            le=100,
        )

    class Output(BlockSchemaOutput):
        export_data: str = SchemaField(
            description="Exported data in the requested format"
        )
        item_count: int = SchemaField(description="Number of items exported")
        total_items: int = SchemaField(
            description="Total number of items in the webset"
        )
        truncated: bool = SchemaField(
            description="Whether the export was truncated due to max_items limit"
        )
        format: str = SchemaField(description="Format of the exported data")

    def __init__(self):
        super().__init__(
            id="5da9d0fd-4b5b-4318-8302-8f71d0ccce9d",
            description="Export webset data in JSON, CSV, or JSON Lines format",
            categories={BlockCategory.DATA},
            input_schema=ExaExportWebsetBlock.Input,
            output_schema=ExaExportWebsetBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "webset_id": "test-webset",
                "format": ExportFormat.JSON,
                "include_content": True,
                "include_enrichments": True,
                "max_items": 10,
            },
            test_output=[
                ("export_data", str),
                ("item_count", 2),
                ("total_items", 2),
                ("truncated", False),
                ("format", "json"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock=self._create_test_mock(),
        )

    @staticmethod
    def _create_test_mock():
        """Create test mocks for the AsyncExa SDK."""
        from unittest.mock import MagicMock

        # Create mock webset items
        mock_item1 = MagicMock()
        mock_item1.model_dump = MagicMock(
            return_value={
                "id": "item-1",
                "url": "https://example.com",
                "title": "Test Item 1",
            }
        )

        mock_item2 = MagicMock()
        mock_item2.model_dump = MagicMock(
            return_value={
                "id": "item-2",
                "url": "https://example.org",
                "title": "Test Item 2",
            }
        )

        # Create mock iterator
        mock_items = [mock_item1, mock_item2]

        return {
            "_get_client": lambda *args, **kwargs: MagicMock(
                websets=MagicMock(
                    items=MagicMock(list_all=lambda *args, **kwargs: iter(mock_items))
                )
            )
        }

    def _get_client(self, api_key: str) -> AsyncExa:
        """Get Exa client (separated for testing)."""
        return AsyncExa(api_key=api_key)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = self._get_client(credentials.api_key.get_secret_value())

        try:
            all_items = []

            # Use SDK's list_all iterator to fetch items
            item_iterator = aexa.websets.items.list_all(
                webset_id=input_data.webset_id, limit=input_data.max_items
            )

            for sdk_item in item_iterator:
                if len(all_items) >= input_data.max_items:
                    break

                # Convert to dict for export
                item_dict = sdk_item.model_dump(by_alias=True, exclude_none=True)
                all_items.append(item_dict)

            # Calculate total and truncated
            total_items = len(all_items)  # SDK doesn't provide total count
            truncated = len(all_items) >= input_data.max_items

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
