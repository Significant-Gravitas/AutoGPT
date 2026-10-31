"""
Exa Websets Import Update Block

``PATCH /v0/imports/{id}`` can change an import's ``title`` and ``metadata``;
the data itself can't be changed after upload (create a new import instead).
"""

from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .websets_api import ExaWebsetsBlockBase, segment
from .websets_fixtures import IMPORT_ID, sample_import
from .websets_import_export import ImportModel


class ExaUpdateImportBlock(ExaWebsetsBlockBase):
    """Rename an import or replace its metadata."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        import_id: str = SchemaField(
            description="The ID of the import to update",
            placeholder="import-id",
        )
        title: str | None = SchemaField(
            default=None,
            description="New title for the import. Leave empty to keep the current one.",
            placeholder="Customer list (Q3)",
            advanced=False,
        )
        metadata: dict[str, str] | None = SchemaField(
            default=None,
            description="Key-value pairs that replace the import's metadata. Leave empty to keep the current metadata.",
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        import_id: str = SchemaField(description="The ID of the updated import")
        status: str = SchemaField(
            description="Import status: pending, processing, completed, failed or canceled"
        )
        title: str = SchemaField(description="The import's title after the update")
        metadata: dict = SchemaField(
            description="The import's metadata after the update"
        )
        updated_at: str = SchemaField(description="When the import was last updated")

    def __init__(self):
        super().__init__(
            id="61095bbe-4de0-44b9-aad5-d4a14c0badb6",
            description="Update the title or metadata of an Exa Websets import",
            categories={BlockCategory.DATA},
            input_schema=ExaUpdateImportBlock.Input,
            output_schema=ExaUpdateImportBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "import_id": IMPORT_ID,
                "title": "Customer list",
            },
            test_output=[
                ("import_id", IMPORT_ID),
                ("status", "completed"),
                ("title", "Customer list"),
                ("metadata", {}),
                ("updated_at", "2024-01-01T00:00:00Z"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_api": lambda *args, **kwargs: sample_import()},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        payload: dict[str, Any] = {}
        if input_data.title:
            payload["title"] = input_data.title
        if input_data.metadata is not None:
            payload["metadata"] = input_data.metadata
        if not payload:
            raise ValueError("Set a new title or metadata to update the import.")

        updated = ImportModel.from_api(
            await self._api(
                credentials,
                "PATCH",
                f"/imports/{segment(input_data.import_id)}",
                json=payload,
            )
        )

        yield "import_id", updated.id
        yield "status", updated.status
        yield "title", updated.title
        yield "metadata", updated.metadata
        yield "updated_at", updated.updated_at
