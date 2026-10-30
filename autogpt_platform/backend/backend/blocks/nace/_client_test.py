import base64
import json
from typing import Any

import aiohttp
import pytest

from backend.blocks.nace import _client
from backend.blocks.nace._client import (
    MAX_IMAGES,
    MAX_REQUEST_BYTES,
    STATE_BUDGET_BYTES,
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    call_drex,
)

from ._test import TEST_IMAGE

ANSWERS: dict[str, dict[str, Any]] = {
    "choice": {
        "type": "choice",
        "choice": "a",
        "probabilities": {"a": 0.9, "b": 0.1},
        "confidence": 0.8,
    },
    "score": {
        "type": "score",
        "score": 0.75,
        "legend": ["Low", "High"],
        "probabilities": {"0": 0.25, "1": 0.75},
        "confidence": 0.6,
    },
    "noul": {"type": "noul", "noul": 0.82},
}
RESPONSE = json.dumps(
    {
        "model": "drex-max",
        "answers": ANSWERS,
        "usage": {"input_tokens": 123, "output_tokens": 45},
        "request_id": "req_body",
    }
).encode()
QUESTIONS = {
    "choice": ChoiceQuestion(instructions="Pick", criteria={"a": "A", "b": None}),
    "score": ScoreQuestion(instructions="Rate", criteria=["Low", "High"]),
    "noul": NoulQuestion(instructions="Yes?"),
}


class FakeResponse:
    def __init__(self, status: int, content: bytes, request_id: str | None = None):
        self.status = status
        self.content = content
        self.headers = {"x-request-id": request_id} if request_id else {}


class FakeRequests:
    """Stands in for backend.util.request.Requests at the client's boundary."""

    def __init__(self, reply: FakeResponse | Exception):
        self.reply = reply
        self.init_kwargs: dict[str, Any] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "FakeRequests":
        def factory(**kwargs: Any) -> "FakeRequests":
            self.init_kwargs = kwargs
            return self

        monkeypatch.setattr(_client, "Requests", factory)
        return self

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def sent_body(self) -> dict[str, Any]:
        return json.loads(self.calls[0][1]["data"])


async def test_success_sends_a_system_one_body_and_parses_typed_answers(monkeypatch):
    fake = FakeRequests(FakeResponse(200, RESPONSE, "req-123")).install(monkeypatch)
    result = await call_drex("key", "evidence", QUESTIONS, [TEST_IMAGE])
    assert fake.init_kwargs["extra_headers"] == {"Authorization": "Bearer key"}
    assert fake.init_kwargs["raise_for_status"] is False
    assert fake.init_kwargs["retry_max_attempts"] == 1
    url, kwargs = fake.calls[0]
    assert url == "https://console.nace.ai/v1/systemone"
    assert kwargs["headers"] == {"Content-Type": "application/json"}
    assert fake.sent_body() == {
        "model": "drex-max",
        "state": "evidence",
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": {"a": "A", "b": None},
            },
            "score": {
                "type": "score",
                "instructions": "Rate",
                "criteria": ["Low", "High"],
            },
            "noul": {"type": "noul", "instructions": "Yes?"},
        },
        "images": [TEST_IMAGE],
    }
    assert result.answers == ANSWERS
    assert (result.input_tokens, result.output_tokens) == (123, 45)
    assert result.request_id == "req-123"
    assert result.response == RESPONSE.decode()
    assert result.error == ""
    assert result.latency_ms >= 0
    assert result.truncated is False
    sent = json.loads(result.request)
    assert sent["images"] == [f"<Image 1: image/png, {len(TEST_IMAGE):,} characters>"]
    assert sent["questions"] == fake.sent_body()["questions"]


@pytest.mark.parametrize(
    "state,expected", [({"a": [1, 2]}, '{"a":[1,2]}'), (None, ""), ("text", "text")]
)
async def test_state_is_text_or_compact_json(monkeypatch, state, expected):
    fake = FakeRequests(FakeResponse(200, RESPONSE)).install(monkeypatch)
    result = await call_drex("key", state, QUESTIONS)
    assert fake.sent_body()["state"] == expected
    assert "images" not in fake.sent_body()
    assert result.request_id == "req_body"


@pytest.mark.parametrize("state", [{"not-json"}, float("nan")])
async def test_invalid_structured_state_is_rejected_before_http(monkeypatch, state):
    """Unserializable state and non-finite numbers cannot reach the provider."""
    fake = FakeRequests(FakeResponse(200, RESPONSE)).install(monkeypatch)

    with pytest.raises(ValueError, match="State must be text or JSON-serializable"):
        await call_drex("key", state, QUESTIONS)

    assert fake.calls == []


async def test_oversized_state_is_truncated_with_a_note(monkeypatch):
    fake = FakeRequests(FakeResponse(200, RESPONSE)).install(monkeypatch)
    result = await call_drex("key", {"text": "é" * STATE_BUDGET_BYTES}, QUESTIONS)
    sent_state = fake.sent_body()["state"]
    assert len(sent_state.encode("utf-8")) <= STATE_BUDGET_BYTES
    assert sent_state.startswith('{"text":"éé')
    assert result.truncated is True
    assert f"{STATE_BUDGET_BYTES:,} bytes" in result.truncation_note
    assert "incomplete JSON" in result.truncation_note


@pytest.mark.parametrize(
    "images,message",
    [
        ([TEST_IMAGE] * (MAX_IMAGES + 1), "at most 8 images"),
        ([TEST_IMAGE, "data:text/plain;base64,aGk="], "Image 2 must be"),
        (["https://example.com/a.png"], "Image 1 must be"),
        (["data:image/gif;base64,R0lGODlh"], "Image 1 must be"),
    ],
)
async def test_bad_images_are_rejected_before_any_request(monkeypatch, images, message):
    fake = FakeRequests(FakeResponse(200, RESPONSE)).install(monkeypatch)
    with pytest.raises(ValueError, match=message):
        await call_drex("key", "evidence", QUESTIONS, images)
    assert fake.calls == []


async def test_oversized_body_is_rejected_before_any_request(monkeypatch):
    fake = FakeRequests(FakeResponse(200, RESPONSE)).install(monkeypatch)
    huge = "data:image/png;base64," + "A" * MAX_REQUEST_BYTES
    with pytest.raises(ValueError, match="4,194,304"):
        await call_drex("key", "evidence", QUESTIONS, [huge])
    assert fake.calls == []


async def test_api_errors_report_status_message_and_issues(monkeypatch):
    body = {
        "error": {
            "type": "invalid_request_error",
            "message": "Invalid request.",
            "issues": [
                {"path": "questions.q.criteria", "message": "must be an object"}
            ],
        },
        "request_id": "req_err",
    }
    FakeRequests(FakeResponse(422, json.dumps(body).encode())).install(monkeypatch)
    result = await call_drex("key", "evidence", QUESTIONS)
    assert result.error == (
        "nace API request failed (HTTP 422). Invalid request. "
        "questions.q.criteria: must be an object"
    )
    assert result.answers == {}
    assert (result.input_tokens, result.output_tokens) == (None, None)
    assert result.response == json.dumps(body)
    assert result.request_id == "req_err"


async def test_issues_already_in_the_message_are_not_repeated(monkeypatch):
    body = {
        "error": {
            "message": "images.0: Image could not be read.",
            "issues": [{"path": "images.0", "message": "Image could not be read."}],
        }
    }
    FakeRequests(FakeResponse(422, json.dumps(body).encode())).install(monkeypatch)
    result = await call_drex("key", "evidence", QUESTIONS)
    assert result.error == (
        "nace API request failed (HTTP 422). images.0: Image could not be read."
    )


async def test_overload_is_reported_without_a_retry(monkeypatch):
    fake = FakeRequests(FakeResponse(529, b"overloaded")).install(monkeypatch)
    result = await call_drex("key", "evidence", QUESTIONS)
    assert result.error == "nace API request failed (HTTP 529)."
    assert result.response == "overloaded"
    assert len(fake.calls) == 1
    assert fake.init_kwargs["retry_max_attempts"] == 1


@pytest.mark.parametrize(
    "failure", [aiohttp.ClientConnectionError("down"), TimeoutError()]
)
async def test_connection_failures_report_no_response(monkeypatch, failure):
    FakeRequests(failure).install(monkeypatch)
    result = await call_drex("key", "evidence", QUESTIONS)
    assert result.error == (
        "nace connection failed or timed out; no HTTP response was received."
    )
    assert result.response is None
    assert result.answers == {}
    assert json.loads(result.request)["state"] == "evidence"


async def test_invalid_utf8_response_is_preserved_as_base64(monkeypatch):
    FakeRequests(FakeResponse(200, b"\xff\xfe{}", "req-1")).install(monkeypatch)
    result = await call_drex("key", "evidence", QUESTIONS)
    assert result.answers == {}
    assert "invalid response" in result.error
    assert "not valid UTF-8" in result.error
    assert result.response is not None
    prefix = "data:application/octet-stream;base64,"
    assert result.response.startswith(prefix)
    assert base64.b64decode(result.response[len(prefix) :]) == b"\xff\xfe{}"


async def test_success_without_usage_keeps_answers_and_null_tokens(monkeypatch):
    body = json.dumps({"answers": ANSWERS}).encode()
    FakeRequests(FakeResponse(200, body)).install(monkeypatch)
    result = await call_drex("key", "evidence", QUESTIONS)
    assert result.answers == ANSWERS
    assert (result.input_tokens, result.output_tokens) == (None, None)
    assert result.request_id == ""
    assert result.error == ""


@pytest.mark.parametrize("missing", list(QUESTIONS))
async def test_missing_requested_answer_is_an_invalid_response(monkeypatch, missing):
    """An incomplete successful HTTP response cannot produce a decision."""
    body = {"answers": {key: value for key, value in ANSWERS.items() if key != missing}}
    response = json.dumps(body).encode()
    FakeRequests(FakeResponse(200, response, "req-malformed")).install(monkeypatch)

    result = await call_drex("key", "evidence", QUESTIONS)

    assert result.error == "nace returned an invalid response (HTTP 200)."
    assert result.answers == {}
    assert result.response == response.decode()
    assert result.request_id == "req-malformed"


@pytest.mark.parametrize(
    "question,answer",
    [
        ("choice", {"type": "choice"}),
        ("choice", {**ANSWERS["choice"], "choice": "unknown"}),
        ("choice", {**ANSWERS["choice"], "confidence": "high"}),
        ("choice", {**ANSWERS["choice"], "probabilities": {"a": "bad"}}),
        ("choice", {**ANSWERS["choice"], "confidence": float("nan")}),
        ("choice", {**ANSWERS["choice"], "probabilities": {"a": 1.1, "b": -0.1}}),
        ("choice", {**ANSWERS["choice"], "probabilities": {}}),
        ("choice", {**ANSWERS["choice"], "probabilities": {"other": 1}}),
        ("choice", {**ANSWERS["choice"], "probabilities": {"a": 0.9, "b": 0.9}}),
        ("choice", {**ANSWERS["choice"], "probabilities": {"a": 1}}),
        (
            "choice",
            {**ANSWERS["choice"], "probabilities": {"a": 0.9, "b": 0.1, "c": 0}},
        ),
        ("choice", {**ANSWERS["choice"], "probabilities": {"a": 0.8, "b": 0.1}}),
        ("score", {"type": "score", "score": 0.5}),
        ("score", {**ANSWERS["score"], "legend": []}),
        ("score", {**ANSWERS["score"], "legend": ["High", "Low"]}),
        ("score", {**ANSWERS["score"], "legend": ["Low", "High", "Extra"]}),
        ("score", {**ANSWERS["score"], "probabilities": {}}),
        ("score", {**ANSWERS["score"], "probabilities": {"0": 0.5, "2": 0.5}}),
        ("score", {**ANSWERS["score"], "probabilities": {"0": 0.9, "1": 0.9}}),
        ("score", {**ANSWERS["score"], "probabilities": {"0": 1.1, "1": -0.1}}),
        ("score", {**ANSWERS["score"], "score": "good"}),
        ("score", {**ANSWERS["score"], "score": 2}),
        ("score", {**ANSWERS["score"], "legend": {"0": "Low", "1": "High"}}),
        ("score", {**ANSWERS["score"], "confidence": None}),
        ("noul", {"type": "noul"}),
        ("noul", {"type": "noul", "noul": "yes"}),
        ("noul", {"type": "noul", "noul": True}),
        ("noul", {"type": "noul", "noul": 1.1}),
        ("noul", {"type": "noul", "noul": float("inf")}),
        ("choice", ANSWERS["noul"]),
    ],
)
async def test_malformed_requested_answer_is_an_invalid_response(
    monkeypatch, question, answer
):
    """Typed answer fields are checked before blocks can consume them."""
    response = json.dumps({"answers": {question: answer}}).encode()
    FakeRequests(FakeResponse(200, response)).install(monkeypatch)

    result = await call_drex("key", "evidence", {question: QUESTIONS[question]})

    assert result.error == "nace returned an invalid response (HTTP 200)."
    assert result.answers == {}
    assert (result.input_tokens, result.output_tokens) == (None, None)
    assert result.response == response.decode()


async def test_valid_boundary_probabilities_and_extra_metadata_are_retained(
    monkeypatch,
):
    """Strict answer validation still accepts integer endpoints and future metadata."""
    answer = {"type": "noul", "noul": 1, "confidence": 1, "details": {"flag": True}}
    response = json.dumps({"answers": {"noul": answer}}).encode()
    FakeRequests(FakeResponse(200, response)).install(monkeypatch)

    result = await call_drex("key", "evidence", {"noul": QUESTIONS["noul"]})

    assert result.answers == {"noul": answer}
    assert result.error == ""


async def test_probabilities_summing_to_one_within_rounding_are_accepted(
    monkeypatch,
):
    """Contract: probabilities sum to 1 within rounding (4 decimal places)."""
    answer = {**ANSWERS["choice"], "probabilities": {"a": 0.9001, "b": 0.0999}}
    response = json.dumps({"answers": {"choice": answer}}).encode()
    FakeRequests(FakeResponse(200, response)).install(monkeypatch)

    result = await call_drex("key", "evidence", {"choice": QUESTIONS["choice"]})

    assert result.error == ""
    assert result.answers == {"choice": answer}
