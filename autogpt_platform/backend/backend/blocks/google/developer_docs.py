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
from backend.util.exceptions import BlockInputError

from ._developer_knowledge_api import (
    MAX_DOCUMENTS,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    DeveloperDoc,
    DeveloperDocPassage,
    DeveloperKnowledgeError,
    build_filter,
    call_developer_knowledge,
    developer_docs_credentials_field,
    developer_docs_error,
    document_names,
    to_developer_doc,
    to_document_name,
    to_passage,
)

_SITES_EXAMPLE = (
    "e.g. firebase.google.com, developer.android.com or docs.cloud.google.com"
)
_ANSWER_QUOTA_HINT = (
    "Answers have a small daily quota (50 per project by default). Use Search "
    "Google Developer Docs instead, or ask Google for more quota."
)

_TEST_NAME = "documents/docs.cloud.google.com/storage/docs/creating-buckets"
_TEST_DOCUMENT_META = {
    "name": _TEST_NAME,
    "uri": "https://docs.cloud.google.com/storage/docs/creating-buckets",
    "title": "Create buckets",
    "dataSource": "docs.cloud.google.com",
    "updateTime": "2026-09-01T12:00:00Z",
}
_TEST_CHUNK = {
    "parent": _TEST_NAME,
    "id": "chunk_0",
    "content": "To create a bucket, run `gcloud storage buckets create gs://BUCKET`.",
    "relevanceScore": 0.92,
    "document": _TEST_DOCUMENT_META,
}
_TEST_PASSAGE = to_passage(_TEST_CHUNK)
_TEST_DOCUMENT = {
    **_TEST_DOCUMENT_META,
    "description": "Create a Cloud Storage bucket.",
    "content": "# Create buckets\n\nThis page shows you how to create a bucket.",
}
_TEST_ANSWER = "Run `gcloud storage buckets create gs://BUCKET`."


class SearchGoogleDeveloperDocsBlock(Block):
    """Search Google's developer documentation for passages matching a query."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = developer_docs_credentials_field()
        query: str = SchemaField(
            description="What to look for, e.g. 'How do I create a Cloud Storage bucket?'",
            max_length=500,
        )
        sites: list[str] = SchemaField(
            description=f"Only search these documentation sites ({_SITES_EXAMPLE})",
            default_factory=list,
        )
        custom_filter: str = SchemaField(
            description=(
                "Extra filter in Google's filter syntax, ANDed with the sites, "
                "e.g. 'update_time >= \"2026-01-01T00:00:00Z\"'"
            ),
            default="",
            advanced=True,
        )
        max_results: int = SchemaField(
            description="Maximum number of passages to return",
            default=5,
            ge=1,
            le=100,
        )
        page_token: str = SchemaField(
            description="Page token from a previous search, to get the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[DeveloperDocPassage] = SchemaField(
            description="Matching passages, most relevant first"
        )
        result: DeveloperDocPassage = SchemaField(description="Each matching passage")
        document_names: list[str] = SchemaField(
            description="The pages the passages come from, for Get Google Developer Docs"
        )
        next_page_token: str = SchemaField(
            description="Token for the next page, when there are more results"
        )

    def __init__(self):
        super().__init__(
            id="f45b5056-0955-4200-8a57-e328d065f165",
            description=(
                "Search Google's developer documentation and return the "
                "best-matching passages with links to their pages. Covers Android, "
                "Firebase, Google Cloud, Maps, Chrome, Flutter, Go, Google AI and "
                "more."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DEVELOPER_TOOLS},
            input_schema=SearchGoogleDeveloperDocsBlock.Input,
            output_schema=SearchGoogleDeveloperDocsBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "How do I create a Cloud Storage bucket?",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("results", [_TEST_PASSAGE]),
                ("result", _TEST_PASSAGE),
                ("document_names", [_TEST_NAME]),
                ("next_page_token", "next-page"),
            ],
            test_mock={
                "_search": lambda *args, **kwargs: {
                    "results": [_TEST_CHUNK],
                    "nextPageToken": "next-page",
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self._search(
                credentials.api_key.get_secret_value(), search_params(input_data)
            )
        except DeveloperKnowledgeError as e:
            raise developer_docs_error(e, self.name, self.id) from e

        passages = [to_passage(chunk) for chunk in result.get("results", [])]
        yield "results", passages
        for passage in passages:
            yield "result", passage
        yield "document_names", document_names(passages)
        if next_page_token := result.get("nextPageToken"):
            yield "next_page_token", next_page_token

    @staticmethod
    async def _search(api_key: str, params: dict[str, str]) -> dict:
        return await call_developer_knowledge(
            "GET", "v1/documents:searchDocumentChunks", api_key, params=params
        )


class AskGoogleDeveloperDocsBlock(Block):
    """Answer a question from Google's developer documentation, with sources."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = developer_docs_credentials_field()
        question: str = SchemaField(
            description="The question, e.g. 'How do I create a BigQuery dataset?'"
        )
        sites: list[str] = SchemaField(
            description=f"Only answer from these documentation sites ({_SITES_EXAMPLE})",
            default_factory=list,
        )
        custom_filter: str = SchemaField(
            description=(
                "Extra filter in Google's filter syntax, ANDed with the sites, "
                "e.g. 'update_time >= \"2026-01-01T00:00:00Z\"'"
            ),
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        answer: str = SchemaField(
            description="The answer, written from Google's documentation"
        )
        sources: list[DeveloperDocPassage] = SchemaField(
            description="The passages the answer is based on"
        )
        source: DeveloperDocPassage = SchemaField(
            description="Each passage the answer is based on"
        )
        document_names: list[str] = SchemaField(
            description="The pages the answer cites, for Get Google Developer Docs"
        )

    def __init__(self):
        super().__init__(
            id="7a0968c1-b988-4cf8-831c-c4a2deadcd16",
            description=(
                "Answer a question about Google developer products with an answer "
                "Google writes from its official documentation, plus the passages "
                "it used. Covers Android, Firebase, Google Cloud, Maps and more. "
                "Answers have a small daily quota; use Search Google Developer "
                "Docs for many questions."
            ),
            categories={BlockCategory.AI, BlockCategory.DEVELOPER_TOOLS},
            input_schema=AskGoogleDeveloperDocsBlock.Input,
            output_schema=AskGoogleDeveloperDocsBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "question": "How do I create a Cloud Storage bucket?",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("answer", _TEST_ANSWER),
                ("sources", [_TEST_PASSAGE]),
                ("source", _TEST_PASSAGE),
                ("document_names", [_TEST_NAME]),
            ],
            test_mock={
                "_answer": lambda *args, **kwargs: {
                    "answer": {
                        "answerText": _TEST_ANSWER,
                        "references": [
                            {"documentReference": {"documentChunk": _TEST_CHUNK}}
                        ],
                    }
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        body = {"query": input_data.question}
        if query_filter := build_filter(input_data.sites, input_data.custom_filter):
            body["filter"] = query_filter
        try:
            result = await self._answer(credentials.api_key.get_secret_value(), body)
        except DeveloperKnowledgeError as e:
            raise developer_docs_error(
                e, self.name, self.id, quota_hint=_ANSWER_QUOTA_HINT
            ) from e

        answer = result.get("answer") or {}
        sources = [
            to_passage(chunk)
            for reference in answer.get("references", [])
            if (
                chunk := (reference.get("documentReference") or {}).get("documentChunk")
            )
        ]
        yield "answer", answer.get("answerText") or ""
        yield "sources", sources
        for source in sources:
            yield "source", source
        yield "document_names", document_names(sources)

    @staticmethod
    async def _answer(api_key: str, body: dict[str, str]) -> dict:
        # One attempt: a 429 here is the daily answer quota, which a retry can't fix.
        return await call_developer_knowledge(
            "POST", "v1:answerQuery", api_key, body=body, max_attempts=1
        )


class GetGoogleDeveloperDocsBlock(Block):
    """Get whole pages of Google's developer documentation as Markdown."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = developer_docs_credentials_field()
        document_names: list[str] = SchemaField(
            description=(
                f"Up to {MAX_DOCUMENTS} pages: document names from Search or Ask "
                "Google Developer Docs (documents/...), or links to the pages"
            ),
        )

    class Output(BlockSchemaOutput):
        documents: list[DeveloperDoc] = SchemaField(
            description="The pages, in the order asked for"
        )
        document: DeveloperDoc = SchemaField(description="Each page")

    def __init__(self):
        super().__init__(
            id="a877a10c-a32e-47ea-86ed-65a27e2c5b10",
            description=(
                "Get whole pages of Google's developer documentation as Markdown, "
                "up to 20 at a time. Takes document names from Search Google "
                "Developer Docs, or links to the pages."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DEVELOPER_TOOLS},
            input_schema=GetGoogleDeveloperDocsBlock.Input,
            output_schema=GetGoogleDeveloperDocsBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "document_names": [
                    "https://docs.cloud.google.com/storage/docs/creating-buckets"
                ],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("documents", [to_developer_doc(_TEST_DOCUMENT)]),
                ("document", to_developer_doc(_TEST_DOCUMENT)),
            ],
            test_mock={
                "_batch_get": lambda *args, **kwargs: {"documents": [_TEST_DOCUMENT]}
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        names = self._names(input_data.document_names)
        try:
            result = await self._batch_get(
                credentials.api_key.get_secret_value(), names
            )
        except DeveloperKnowledgeError as e:
            raise developer_docs_error(e, self.name, self.id) from e

        documents = [to_developer_doc(doc) for doc in result.get("documents", [])]
        yield "documents", documents
        for document in documents:
            yield "document", document

    def _names(self, values: list[str]) -> list[str]:
        names = list(dict.fromkeys(to_document_name(v) for v in values if v.strip()))
        if not names:
            raise BlockInputError(
                message="Give at least one document name or page link.",
                block_name=self.name,
                block_id=self.id,
            )
        if len(names) > MAX_DOCUMENTS:
            raise BlockInputError(
                message=(
                    f"Google returns at most {MAX_DOCUMENTS} pages per call, and "
                    f"{len(names)} were given. Split them across several runs."
                ),
                block_name=self.name,
                block_id=self.id,
            )
        return names

    @staticmethod
    async def _batch_get(api_key: str, names: list[str]) -> dict:
        return await call_developer_knowledge(
            "GET",
            "v1/documents:batchGet",
            api_key,
            params=[("names", name) for name in names],
        )


def search_params(input_data: SearchGoogleDeveloperDocsBlock.Input) -> dict[str, str]:
    """Query parameters for ``documents.searchDocumentChunks``."""
    params = {"query": input_data.query, "pageSize": str(input_data.max_results)}
    if query_filter := build_filter(input_data.sites, input_data.custom_filter):
        params["filter"] = query_filter
    if input_data.page_token:
        params["pageToken"] = input_data.page_token
    return params
