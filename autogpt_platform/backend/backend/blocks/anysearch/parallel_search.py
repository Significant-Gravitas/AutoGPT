import asyncio
from typing import Any

from pydantic import model_validator

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

from ._api import (
    AnySearchClient,
    AnySearchDomain,
    AnySearchQueryResults,
    result_from_dict,
    unwrap_envelope,
)
from ._config import anysearch


class AnySearchParallelSearchBlock(Block):
    """Runs up to 5 AnySearch queries concurrently (client-side fan-out)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = anysearch.credentials_field(
            description="The AnySearch integration requires an API Key."
        )
        queries: list[str] = SchemaField(
            description="The search queries to run in parallel (max 5)",
            min_length=1,
            max_length=5,
        )
        max_results: int = SchemaField(
            description="Maximum number of results per query",
            default=5,
            ge=1,
            le=10,
            advanced=True,
        )
        domain: AnySearchDomain | None = SchemaField(
            description="Restrict every query to a vertical domain",
            default=None,
            advanced=True,
        )
        sub_domain: str | None = SchemaField(
            description="Sub-domain inside the domain, applied to every query "
            "(e.g. finance.quote)",
            default=None,
            advanced=True,
        )
        sub_domain_params: dict[str, Any] | None = SchemaField(
            description="Structured parameters for the shared sub_domain",
            default=None,
            advanced=True,
        )

        @model_validator(mode="after")
        def _check_vertical_inputs(self):
            if self.domain and not self.sub_domain:
                raise ValueError("sub_domain is required when domain is set")
            if (
                self.domain
                and self.sub_domain
                and not self.sub_domain.startswith(f"{self.domain.value}.")
            ):
                raise ValueError(
                    "sub_domain must belong to the selected domain "
                    f"({self.domain.value}.*)"
                )
            return self

    class Output(BlockSchemaOutput):
        results: list[AnySearchQueryResults] = SchemaField(
            description="One result group per query, in input order"
        )

    def __init__(self):
        super().__init__(
            id="611523ee-ba2e-43b5-bd62-4fe1b2c42044",
            description="Runs several AnySearch queries in parallel "
            "(client-side concurrency via asyncio)",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": anysearch.get_test_credentials().model_dump(),
                "queries": ["What is AutoGPT?", "AutoGPT documentation"],
                "max_results": 1,
            },
            test_credentials=anysearch.get_test_credentials(),
            test_output=[
                (
                    "results",
                    lambda groups: isinstance(groups, list)
                    and len(groups) == 2
                    and all(g.results and not g.error for g in groups),
                ),
            ],
            test_mock={
                "_search": lambda *args, **kwargs: {
                    "code": 0,
                    "message": "success",
                    "data": {
                        "results": [
                            {
                                "title": "AutoGPT",
                                "url": "https://agpt.co",
                                "snippet": "AutoGPT is a platform for building "
                                "AI agents.",
                                "content": None,
                            }
                        ],
                        "metadata": {"total_results": 1, "search_time_ms": 10},
                    },
                }
            },
        )

    async def _search(
        self, credentials: APIKeyCredentials, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /v1/search and return the raw response envelope (mockable)."""
        return await AnySearchClient(credentials).search(payload)

    async def _search_one(
        self, credentials: APIKeyCredentials, payload: dict[str, Any]
    ) -> AnySearchQueryResults:
        query = payload["query"]
        try:
            data = unwrap_envelope(await self._search(credentials, payload))
            results = [result_from_dict(r) for r in data.get("results", [])]
        except Exception as e:
            return AnySearchQueryResults(query=query, error=str(e) or type(e).__name__)
        return AnySearchQueryResults(query=query, results=results)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        shared: dict[str, Any] = {"max_results": input_data.max_results}
        if input_data.sub_domain:
            shared["tag"] = input_data.sub_domain
        if input_data.sub_domain_params:
            shared["params"] = input_data.sub_domain_params

        groups = await asyncio.gather(
            *(
                self._search_one(credentials, {"query": q, **shared})
                for q in input_data.queries
            )
        )

        yield "results", list(groups)
