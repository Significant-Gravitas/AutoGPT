from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError

from ._api import ConductorClient
from ._config import conductor


class ConductorSearchTranscriptsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description="Conductor API key from app.conductor.build/users/api-keys"
        )
        query: str = SchemaField(
            description="Read-only SQL over the transcripts of the workspaces you "
            "can access, e.g. SELECT * FROM messages WHERE content LIKE '%bug%' "
            "LIMIT 20",
            placeholder="SELECT ... LIMIT 20",
        )

    class Output(BlockSchemaOutput):
        rows: list[dict] = SchemaField(description="Result rows as objects")
        row_count: int = SchemaField(description="Number of rows returned")
        truncated: bool = SchemaField(
            description="True when the server cut the result short"
        )

    def __init__(self):
        super().__init__(
            id="d2f58c9b-c65f-41bf-93fb-c852be03fe41",
            description="Search Conductor session transcripts with a read-only SQL "
            "query. Useful for finding what an agent said or did across workspaces.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.READ,
            capability_kind="service",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": conductor.get_test_credentials().model_dump(),
                "query": "SELECT 1 AS one",
            },
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("rows", [{"one": 1}]),
                ("row_count", 1),
                ("truncated", False),
            ],
            test_mock={
                "_query": lambda *args, **kwargs: {
                    "rows": [{"one": 1}],
                    "rowCount": 1,
                    "truncated": False,
                }
            },
        )

    async def _query(
        self, credentials: APIKeyCredentials, query: str
    ) -> dict[str, Any]:
        return await ConductorClient(credentials).sql(query)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self._query(credentials, input_data.query)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Transcript search failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        rows = result.get("rows") or []
        yield "rows", rows
        yield "row_count", int(result.get("rowCount") or len(rows))
        yield "truncated", bool(result.get("truncated", False))
