from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.dataforb2b._api import DataForB2BClient, api_error_message
from backend.blocks.dataforb2b._config import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    DataForB2BCredentials,
    DataForB2BCredentialsInput,
    dataforb2b,
)
from backend.blocks.dataforb2b._enums import SearchCategory
from backend.data.model import SchemaField
from backend.util.request import HTTPClientError, HTTPServerError

# Conservative safety ceiling for `max_results`, mirroring search.py's MAX_COUNT.
MAX_RESULTS = 100


class SmartSearchBlock(Block):
    """Natural-language LinkedIn people & company search with DataForB2B."""

    class Input(BlockSchemaInput):
        query: str = SchemaField(
            description="Plain-English LinkedIn search / ICP (e.g. 'marketing directors at Series A SaaS startups in France')",
            default="",
            advanced=False,
        )
        category: SearchCategory = SchemaField(
            description="What to search for: 'people' or 'company'",
            default=SearchCategory.PEOPLE,
            advanced=False,
        )
        session_id: str = SchemaField(
            description="Session id to resolve a previous 'needs_input' turn",
            default="",
            advanced=True,
        )
        answers: dict = SchemaField(
            description="Answers to clarifying questions {question_id: answer}",
            default_factory=dict,
            advanced=True,
        )
        max_results: int = SchemaField(
            description=f"Maximum results to return (1-{MAX_RESULTS})",
            default=25,
            advanced=False,
        )
        enrich_live: bool = SchemaField(
            description="Fetch fresh live data (uses more credits)",
            default=False,
            advanced=True,
        )
        credentials: DataForB2BCredentialsInput = dataforb2b.credentials_field(
            description="DataForB2B API key"
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Full reasoning-search response")
        status: str = SchemaField(description="'ok' or 'needs_input'", default="")
        results: list = SchemaField(
            description="Matching results when status is ok", default_factory=list
        )
        questions: list = SchemaField(
            description="Clarifying questions when status is needs_input",
            default_factory=list,
        )
        session_id: str = SchemaField(
            description="Session id to continue the search", default=""
        )
        applied_filters: dict = SchemaField(
            description=(
                "The structured filters the search applied. Feed this into "
                "People Search or Company Search 'filters_json' with an offset to "
                "paginate beyond the first page."
            ),
            default_factory=dict,
        )
        category: str = SchemaField(
            description="Category searched ('people' or 'company', echoed from the input) — route pagination to the matching search block",
            default="",
        )
        error: str = SchemaField(
            description="Error message if the search failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="3a835db4-e60d-4e5f-aa7b-3b1ef14ff4a6",
            description=(
                "Natural-language search for people, leads or companies using DataForB2B's "
                "B2B database — describe your ideal lead or ICP in plain English and get "
                "matching profiles. Handles clarifying questions."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.SOCIAL, BlockCategory.AI},
            input_schema=SmartSearchBlock.Input,
            output_schema=SmartSearchBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "software engineers in Paris",
                "category": "people",
                "max_results": 1,
            },
            test_output=[
                (
                    "result",
                    {"status": "ok", "total": 1, "count": 1, "results": [{"id": "1"}]},
                ),
                ("status", "ok"),
                ("results", [{"id": "1"}]),
                ("questions", []),
                ("session_id", ""),
                ("applied_filters", {}),
                ("category", "people"),
            ],
            test_mock={
                "reasoning_search": lambda payload, credentials: {
                    "status": "ok",
                    "total": 1,
                    "count": 1,
                    "results": [{"id": "1"}],
                }
            },
        )

    @staticmethod
    async def reasoning_search(
        payload: dict, credentials: DataForB2BCredentials
    ) -> dict:
        client = DataForB2BClient(credentials)
        return await client.reasoning_search(payload)

    async def run(
        self, input_data: Input, *, credentials: DataForB2BCredentials, **kwargs
    ) -> BlockOutput:
        query = input_data.query.strip()
        session_id = input_data.session_id.strip()
        has_continuation_data = bool(session_id or input_data.answers)
        if has_continuation_data:
            if query or not session_id or not input_data.answers:
                raise ValueError(
                    "Provide either 'query' or both 'session_id' and 'answers'."
                )
        elif not query:
            raise ValueError(
                "Provide 'query' (first call) or 'session_id' + 'answers' (to "
                "resolve a needs_input turn)."
            )

        category = input_data.category
        max_results = max(1, min(int(input_data.max_results), MAX_RESULTS))
        payload: dict = {
            "category": category.value,
            "max_results": max_results,
            "enrich_live": bool(input_data.enrich_live),
        }
        if query:
            payload["query"] = query
        if session_id:
            payload["session_id"] = session_id
        if input_data.answers:
            payload["answers"] = input_data.answers

        try:
            data = await self.reasoning_search(payload, credentials)
        except HTTPClientError as e:
            yield "error", api_error_message(e, "smart search")
            return
        except HTTPServerError as e:
            yield "error", api_error_message(e, "smart search")
            return
        yield "result", data
        yield "status", data.get("status", "ok")
        yield "results", data.get("results", []) or []
        yield "questions", data.get("questions", []) or []
        yield "session_id", data.get("session_id", "") or ""
        yield "applied_filters", data.get("applied_filters", {}) or {}
        yield "category", category.value
