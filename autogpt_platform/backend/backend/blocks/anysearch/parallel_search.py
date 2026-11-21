import asyncio
from typing import Any, Literal

import aiohttp
from pydantic import model_validator

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
from backend.util.request import HTTPClientError, HTTPServerError

from ._api import (
    AnySearchAuth,
    AnySearchClient,
    AnySearchDomain,
    AnySearchQueryResults,
    check_vertical_inputs,
    result_from_dict,
    sanitize_upstream_text,
    unwrap_envelope,
)
from ._config import anysearch
from ._http import AnySearchHTTPClientError, AnySearchHTTPServerError


class AnySearchParallelSearchBlock(Block):
    """Runs up to 5 AnySearch queries concurrently (client-side fan-out)."""

    class Input(BlockSchemaInput):
        auth: AnySearchAuth = SchemaField(
            title="Authentication",
            description="Anonymous tier (lower rate limit, no key) or an "
            "AnySearch API key credential",
            default=AnySearchAuth.ANONYMOUS,
            advanced=False,
        )
        credentials: CredentialsMetaInput = anysearch.credentials_field(
            title="AnySearch API key",
            description="The AnySearch integration requires an API Key.",
            discriminator="auth",
            discriminator_mapping={AnySearchAuth.API_KEY.value: "anysearch"},
            credential_free_discriminator_values={AnySearchAuth.ANONYMOUS.value},
            default=None,
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
        zone: Literal["cn", "intl"] | None = SchemaField(
            description="Regional recall-pool routing (cn=China, "
            "intl=international) applied to every query; the effect "
            "varies by query",
            default=None,
            advanced=True,
        )
        language: str | None = SchemaField(
            description="Preferred result language as a BCP 47 tag "
            "(e.g. zh-CN, en), applied to every query",
            default=None,
            advanced=True,
        )
        format: Literal["json", "markdown"] | None = SchemaField(
            description="Content encoding inside each result - 'markdown' "
            "makes AnySearchResult.content Markdown text; the outer "
            "response stays JSON",
            default=None,
            advanced=True,
        )

        @model_validator(mode="after")
        def _check_vertical_inputs(self):
            check_vertical_inputs(self.domain, self.sub_domain)
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
            effect=BlockEffect.READ,
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
                    and all(g.results and not g.error for g in groups)
                    and all(g.request_id == "req-test" for g in groups),
                ),
            ],
            test_mock={
                "_search": lambda *args, **kwargs: {
                    "code": 0,
                    "message": "success",
                    "request_id": "req-test",
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
        self, credentials: APIKeyCredentials | None, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /v1/search and return the raw response envelope (mockable)."""
        return await AnySearchClient(credentials).search(
            payload
        )  # pragma: no cover - mocked seam, covered live

    async def _search_one(
        self, credentials: APIKeyCredentials | None, payload: dict[str, Any]
    ) -> AnySearchQueryResults:
        query = payload["query"]
        request_id = ""
        try:
            response = await self._search(credentials, payload)
            if not isinstance(response, dict):
                raise ValueError("malformed AnySearch response envelope")
            request_id = str(response.get("request_id") or "")
            env = unwrap_envelope(response, required="results")
            raw_results = env.data["results"]
            if not isinstance(raw_results, list):
                raise ValueError("AnySearch response data.results is not a list")
            results = [result_from_dict(r) for r in raw_results]
        except (AnySearchHTTPClientError, AnySearchHTTPServerError) as e:
            return AnySearchQueryResults(
                query=query, error=str(e), request_id=e.request_id
            )
        except (
            ValueError,
            HTTPClientError,
            HTTPServerError,
            aiohttp.ClientError,
            TimeoutError,
            asyncio.TimeoutError,
        ) as e:
            # Expected failure modes only - envelope errors, HTTP status
            # (4xx/5xx), transport, and timeouts degrade to an error group;
            # anything else propagates and fails the whole block.
            return AnySearchQueryResults(
                query=query,
                error=sanitize_upstream_text(str(e) or type(e).__name__),
                request_id=request_id,
            )
        return AnySearchQueryResults(
            query=query, results=results, request_id=request_id
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials | None = None,
        **kwargs
    ) -> BlockOutput:
        if input_data.auth == AnySearchAuth.ANONYMOUS:
            # Anonymous wins over a still-selected credential, the same way
            # AutoPilot honours an explicit credential-free transport.
            credentials = None
        elif credentials is None:
            raise ValueError(
                "AnySearch API key credentials are required when auth is api_key."
            )

        shared: dict[str, Any] = {"max_results": input_data.max_results}
        if input_data.sub_domain:
            shared["tag"] = input_data.sub_domain
        if input_data.sub_domain_params:
            shared["params"] = input_data.sub_domain_params
        if input_data.zone:
            shared["zone"] = input_data.zone
        if input_data.language:
            shared["language"] = input_data.language
        if input_data.format:
            shared["format"] = input_data.format

        tasks = [
            asyncio.ensure_future(self._search_one(credentials, {"query": q, **shared}))
            for q in input_data.queries
        ]
        try:
            groups = await asyncio.gather(*tasks)
        finally:
            # A propagated unexpected failure must not leave sibling calls
            # running detached - they would burn quota and log unretrieved
            # exceptions. cancel() only schedules cancellation; draining
            # the tasks lets their cleanup finish before the block exits.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        yield "results", list(groups)
