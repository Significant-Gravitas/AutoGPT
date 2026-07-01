from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.dataforb2b._api import DataForB2BClient
from backend.blocks.dataforb2b._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    DataForB2BCredentials,
    DataForB2BCredentialsField,
    DataForB2BCredentialsInput,
)
from backend.data.model import SchemaField


class SearchFilterTypeaheadBlock(Block):
    """Resolve the exact filter value for LinkedIn searches with DataForB2B."""

    class Input(BlockSchemaInput):
        type: str = SchemaField(
            description="Filter type to resolve (company, industry, title, skill, school, investor, location, category)",
            advanced=False,
        )
        q: str = SchemaField(description="Free-text query to resolve", advanced=False)
        limit: int = SchemaField(
            description="Max suggestions (1-20)", default=20, advanced=False
        )
        credentials: DataForB2BCredentialsInput = DataForB2BCredentialsField()

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Full typeahead response")
        results: list = SchemaField(
            description="List of suggestions", default_factory=list
        )
        values: list = SchemaField(
            description="Resolved stored values", default_factory=list
        )
        error: str = SchemaField(
            description="Error message if the lookup failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="554b0945-a2f7-4ee0-bee5-1ffb298a2e30",
            description=(
                "Resolve the exact filter value (company, industry, job title, skill, "
                "school, location) for people and company searches with DataForB2B."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.SOCIAL},
            input_schema=SearchFilterTypeaheadBlock.Input,
            output_schema=SearchFilterTypeaheadBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "type": "company",
                "q": "google",
                "limit": 3,
            },
            test_output=[
                ("result", {"results": [{"value": "Google"}]}),
                ("results", [{"value": "Google"}]),
                ("values", ["Google"]),
            ],
            test_mock={
                "typeahead": lambda type_, q, limit, credentials: {
                    "results": [{"value": "Google"}]
                }
            },
        )

    @staticmethod
    async def typeahead(
        type_: str, q: str, limit: int, credentials: DataForB2BCredentials
    ) -> dict:
        client = DataForB2BClient(credentials)
        return await client.typeahead(type_, q, limit)

    async def run(
        self, input_data: Input, *, credentials: DataForB2BCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.type:
            raise ValueError("'type' is required.")
        if not input_data.q:
            raise ValueError("'q' (query) is required.")

        limit = max(1, min(int(input_data.limit or 20), 20))
        data = await self.typeahead(input_data.type, input_data.q, limit, credentials)
        results = data.get("results", []) or []
        yield "result", data
        yield "results", results
        yield "values", [r.get("value") for r in results if r.get("value")]
