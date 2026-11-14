from urllib.parse import quote

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.helpers.http import GetRequest
from backend.data.model import SchemaField


class DatacircleBestApiBlock(Block, GetRequest):
    """Finds the best GTM API for a kind of data, from Datacircle's open API (no key)."""

    class Input(BlockSchemaInput):
        slug: str = SchemaField(
            description="The kind of API, such as email-finder or web-scraping. "
            "The full list is at https://api.datacircle.dev/v1/apis",
            placeholder="email-finder",
        )
        limit: int = SchemaField(
            description="How many plans to return", default=3, ge=1, le=20
        )

    class Output(BlockSchemaOutput):
        plans: list = SchemaField(
            description="The best plans of that kind, the best first"
        )
        error: str = SchemaField(description="Error message if the call fails")

    def __init__(self):
        super().__init__(
            id="6f1d0a52-8c3e-4b7a-9d41-2e5b7c9a1f30",
            description="Finds the best GTM API for a kind of data with Datacircle "
            "(https://datacircle.dev), which benchmarks all the GTM APIs.",
            categories={BlockCategory.SEARCH},
            input_schema=DatacircleBestApiBlock.Input,
            output_schema=DatacircleBestApiBlock.Output,
            test_input={"slug": "email-finder", "limit": 1},
            test_output=("plans", [{"provider": "Example", "plan": "Pro"}]),
            test_mock={
                "get_request": lambda url, headers, json: {
                    "plans": [{"provider": "Example", "plan": "Pro"}]
                }
            },
            effect=BlockEffect.READ,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        url = (
            f"https://api.datacircle.dev/v1/apis/{quote(input_data.slug, safe='')}"
            f"/best?limit={input_data.limit}"
        )
        try:
            response = await self.get_request(url, headers={}, json=True)
        except Exception as e:
            raise ValueError(f"Failed to fetch from Datacircle: {e}") from e
        yield "plans", response.get("plans", [])
