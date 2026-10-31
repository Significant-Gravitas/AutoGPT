"""Unit tests for the Google developer docs blocks' requests, parsing and errors.

The blocks' own test_input/test_mock cases mock the API calls away; these
cover what those mocks skip.
"""

import json

import pytest

from backend.blocks.google._developer_knowledge_api import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    DeveloperKnowledgeError,
    build_filter,
    call_developer_knowledge,
    developer_docs_error,
    document_names,
    to_developer_doc,
    to_document_name,
    to_passage,
)
from backend.blocks.google.developer_docs import (
    AskGoogleDeveloperDocsBlock,
    GetGoogleDeveloperDocsBlock,
    SearchGoogleDeveloperDocsBlock,
    search_params,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError
from backend.util.request import Requests

API = "https://developerknowledge.googleapis.com"


class _Response:
    def __init__(self, status: int, body: dict | bytes, reason: str = "OK"):
        self.status = status
        self.reason = reason
        self.content = body if isinstance(body, bytes) else json.dumps(body).encode()

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def _patch_requests(monkeypatch, response: _Response) -> list[dict]:
    calls: list[dict] = []

    async def fake_request(self, method, url, **kwargs):
        calls.append(
            {
                "method": method,
                "url": url,
                "attempts": self.retry_max_attempts,
                "trusted": self.trusted_origins,
                **kwargs,
            }
        )
        return response

    monkeypatch.setattr(Requests, "request", fake_request)
    return calls


def _google_error(status: int, message: str, reason: str = "") -> dict:
    details = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}]
    return {
        "error": {
            "code": status,
            "message": message,
            "details": details if reason else [],
        }
    }


@pytest.mark.asyncio
async def test_call_sends_the_key_in_a_header_not_the_url(monkeypatch):
    calls = _patch_requests(monkeypatch, _Response(200, {"results": []}))

    result = await call_developer_knowledge(
        "GET", "v1/documents:searchDocumentChunks", "secret-key", params={"query": "x"}
    )

    assert result == {"results": []}
    call = calls[0]
    assert call["url"] == f"{API}/v1/documents:searchDocumentChunks"
    assert call["headers"] == {"X-Goog-Api-Key": "secret-key"}
    assert "secret-key" not in call["url"] + json.dumps(call["params"])
    assert call["attempts"] == 3
    assert call["trusted"] == ["developerknowledge.googleapis.com"]


@pytest.mark.asyncio
async def test_call_reads_the_reason_from_google_errors(monkeypatch):
    _patch_requests(
        monkeypatch,
        _Response(
            400,
            _google_error(400, "API key not valid.", "API_KEY_INVALID"),
            "Bad Request",
        ),
    )
    with pytest.raises(DeveloperKnowledgeError) as exc_info:
        await call_developer_knowledge("GET", "v1/documents:batchGet", "bad-key")
    assert exc_info.value.status == 400
    assert exc_info.value.reason == "API_KEY_INVALID"
    assert exc_info.value.message == "API key not valid."


@pytest.mark.asyncio
async def test_call_survives_errors_without_a_json_body(monkeypatch):
    _patch_requests(monkeypatch, _Response(502, b"<html>oops</html>", "Bad Gateway"))
    with pytest.raises(DeveloperKnowledgeError) as exc_info:
        await call_developer_knowledge("POST", "v1:answerQuery", "key", body={})
    assert (exc_info.value.status, exc_info.value.message) == (502, "Bad Gateway")
    assert exc_info.value.reason == ""


@pytest.mark.parametrize(
    "status, reason, message, error_type, expected",
    [
        (
            400,
            "API_KEY_INVALID",
            "API key not valid.",
            BlockExecutionError,
            "rejected the API key",
        ),
        (401, "", "Unauthenticated.", BlockExecutionError, "rejected the API key"),
        (
            403,
            "API_KEY_SERVICE_BLOCKED",
            "Blocked.",
            BlockExecutionError,
            "API restrictions",
        ),
        (
            403,
            "SERVICE_DISABLED",
            "Developer Knowledge API has not been used in project 1 before",
            BlockExecutionError,
            "error 403: Developer Knowledge API has not been used",
        ),
        (
            400,
            "",
            "Invalid filter.",
            BlockInputError,
            "rejected the request: Invalid filter.",
        ),
        (
            404,
            "",
            "Document not found.",
            BlockExecutionError,
            "no such page (Document not found.)",
        ),
        (
            429,
            "RATE_LIMIT_EXCEEDED",
            "Quota exceeded.",
            BlockExecutionError,
            "is used up: Quota exceeded.",
        ),
        (
            500,
            "",
            "Internal error.",
            BlockExecutionError,
            "API error 500: Internal error.",
        ),
    ],
)
def test_error_messages(status, reason, message, error_type, expected):
    error = developer_docs_error(
        DeveloperKnowledgeError(status, message, reason), "block", "id"
    )
    assert type(error) is error_type
    assert expected in str(error)


def test_quota_hint_is_only_added_to_quota_errors():
    hint = "Use Search Google Developer Docs instead."
    quota = developer_docs_error(DeveloperKnowledgeError(429, "Quota."), "b", "i", hint)
    other = developer_docs_error(DeveloperKnowledgeError(500, "Boom."), "b", "i", hint)
    assert str(quota).endswith(hint)
    assert hint not in str(other)


@pytest.mark.parametrize(
    "sites, custom_filter, expected",
    [
        ([], "", ""),
        (["", "  "], " ", ""),
        (["firebase.google.com"], "", 'data_source = "firebase.google.com"'),
        (
            [
                "https://Firebase.google.com/docs/auth",
                "firebase.google.com",
                " web.dev ",
            ],
            "",
            'data_source = "firebase.google.com" OR data_source = "web.dev"',
        ),
        ([], 'uri != "x"', 'uri != "x"'),
        (["web.dev"], 'uri != "x"', '(data_source = "web.dev") AND (uri != "x")'),
    ],
)
def test_build_filter(sites, custom_filter, expected):
    assert build_filter(sites, custom_filter) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            "documents/firebase.google.com/docs/auth",
            "documents/firebase.google.com/docs/auth",
        ),
        (
            "https://firebase.google.com/docs/auth?hl=en#setup",
            "documents/firebase.google.com/docs/auth",
        ),
        (
            "  https://Developer.Android.com/guide  ",
            "documents/developer.android.com/guide",
        ),
        (
            "docs.cloud.google.com/storage/docs",
            "documents/docs.cloud.google.com/storage/docs",
        ),
        (
            "https://developer.chrome.com/docs/extensions/",
            "documents/developer.chrome.com/docs/extensions/",
        ),
    ],
)
def test_to_document_name(value: str, expected: str):
    assert to_document_name(value) == expected


def test_to_passage_falls_back_to_the_document_name():
    passage = to_passage(
        {"content": "Text", "document": {"name": "documents/web.dev/a", "title": "A"}}
    )
    assert passage.document_name == "documents/web.dev/a"
    assert (passage.title, passage.url, passage.relevance_score) == ("A", None, None)

    bare = to_passage({})
    assert (bare.document_name, bare.content) == ("", "")


def test_document_names_are_unique_and_keep_rank_order():
    passages = [
        to_passage({"parent": name})
        for name in ["documents/b", "documents/a", "documents/b"]
    ]
    passages.append(to_passage({}))
    assert document_names(passages) == ["documents/b", "documents/a"]


def test_to_developer_doc_maps_every_field():
    doc = to_developer_doc(
        {
            "name": "documents/web.dev/vitals",
            "uri": "https://web.dev/vitals",
            "title": "Web Vitals",
            "description": "Quality signals",
            "content": "# Web Vitals",
            "dataSource": "web.dev",
            "updateTime": "2026-09-01T00:00:00Z",
            "contentLengthBytes": 12,
        }
    )
    assert doc.model_dump() == {
        "name": "documents/web.dev/vitals",
        "title": "Web Vitals",
        "url": "https://web.dev/vitals",
        "description": "Quality signals",
        "content": "# Web Vitals",
        "site": "web.dev",
        "updated_at": "2026-09-01T00:00:00Z",
    }


def _search_input(**fields) -> SearchGoogleDeveloperDocsBlock.Input:
    return SearchGoogleDeveloperDocsBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "query": "bucket", **fields}
    )


def test_search_params():
    assert search_params(_search_input()) == {"query": "bucket", "pageSize": "5"}
    assert search_params(
        _search_input(sites=["docs.cloud.google.com"], max_results=20, page_token="p2")
    ) == {
        "query": "bucket",
        "pageSize": "20",
        "filter": 'data_source = "docs.cloud.google.com"',
        "pageToken": "p2",
    }


def test_search_query_is_capped_at_google_limit():
    with pytest.raises(ValueError):
        _search_input(query="x" * 501)


async def _outputs(block, input_data) -> dict[str, list]:
    outputs: dict[str, list] = {}
    async for name, value in block.run(input_data, credentials=TEST_CREDENTIALS):
        outputs.setdefault(name, []).append(value)
    return outputs


def _get_input(names: list[str]) -> GetGoogleDeveloperDocsBlock.Input:
    return GetGoogleDeveloperDocsBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "document_names": names}
    )


@pytest.mark.asyncio
async def test_get_normalizes_and_dedupes_names():
    block = GetGoogleDeveloperDocsBlock()
    requested: list[list[str]] = []

    async def batch_get(api_key: str, names: list[str]) -> dict:
        requested.append(names)
        return {"documents": [{"name": name} for name in names]}

    block._batch_get = batch_get
    outputs = await _outputs(
        block,
        _get_input(
            [
                "https://web.dev/articles/vitals#lcp",
                "documents/web.dev/articles/vitals",
                " ",
                "firebase.google.com/docs/auth",
            ]
        ),
    )
    assert requested == [
        ["documents/web.dev/articles/vitals", "documents/firebase.google.com/docs/auth"]
    ]
    assert [doc.name for doc in outputs["documents"][0]] == requested[0]
    assert len(outputs["document"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "names, expected",
    [
        ([" "], "at least one"),
        ([f"documents/web.dev/{i}" for i in range(21)], "at most 20"),
    ],
)
async def test_get_rejects_empty_and_oversized_requests(names, expected):
    with pytest.raises(BlockInputError, match=expected):
        await _outputs(GetGoogleDeveloperDocsBlock(), _get_input(names))


@pytest.mark.asyncio
async def test_batch_get_repeats_the_names_parameter(monkeypatch):
    calls = _patch_requests(monkeypatch, _Response(200, {"documents": []}))
    await GetGoogleDeveloperDocsBlock._batch_get("key", ["documents/a", "documents/b"])
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == f"{API}/v1/documents:batchGet"
    assert calls[0]["params"] == [("names", "documents/a"), ("names", "documents/b")]


@pytest.mark.asyncio
async def test_answer_posts_once_without_retries(monkeypatch):
    calls = _patch_requests(monkeypatch, _Response(200, {"answer": {}}))
    await AskGoogleDeveloperDocsBlock._answer("key", {"query": "q"})
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"] == f"{API}/v1:answerQuery"
    assert calls[0]["json"] == {"query": "q"}
    assert calls[0]["attempts"] == 1


def _ask_input(**fields) -> AskGoogleDeveloperDocsBlock.Input:
    return AskGoogleDeveloperDocsBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "question": "How?", **fields}
    )


@pytest.mark.asyncio
async def test_ask_filters_sites_and_skips_references_without_a_chunk():
    block = AskGoogleDeveloperDocsBlock()
    bodies: list[dict] = []

    async def answer(api_key: str, body: dict) -> dict:
        bodies.append(body)
        return {
            "answer": {
                "answerText": "Like this.",
                "references": [
                    {"documentReference": {"documentChunk": {"parent": "documents/a"}}},
                    {"documentReference": {}},
                    {},
                ],
            }
        }

    block._answer = answer
    outputs = await _outputs(block, _ask_input(sites=["firebase.google.com"]))
    assert bodies == [
        {"query": "How?", "filter": 'data_source = "firebase.google.com"'}
    ]
    assert outputs["answer"] == ["Like this."]
    assert [s.document_name for s in outputs["sources"][0]] == ["documents/a"]
    assert outputs["document_names"] == [["documents/a"]]


@pytest.mark.asyncio
async def test_ask_points_to_search_when_the_answer_quota_runs_out():
    block = AskGoogleDeveloperDocsBlock()

    async def answer(api_key: str, body: dict) -> dict:
        raise DeveloperKnowledgeError(429, "Quota exceeded.", "RATE_LIMIT_EXCEEDED")

    block._answer = answer
    with pytest.raises(BlockExecutionError, match="Search Google Developer Docs"):
        await _outputs(block, _ask_input())
