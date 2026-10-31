"""Client for nace System One decisions, answered by the Drex Max model."""

import json
from time import perf_counter
from typing import Any

import aiohttp
from pydantic import BaseModel

from backend.util.request import Requests

from ._wire import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    Usage,
    describe,
    error_response,
    parse_response,
    response_body,
)

__all__ = ["ChoiceQuestion", "NoulQuestion", "Question", "ScoreQuestion"]

NACE_API_URL = "https://console.nace.ai"
SYSTEMONE_URL = f"{NACE_API_URL}/v1/systemone"
DEFAULT_MODEL = "drex-max"
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_IMAGES = 8
IMAGE_DATA_URL_PREFIXES = (
    "data:image/png;base64,",
    "data:image/jpeg;base64,",
    "data:image/webp;base64,",
)
# nace accepts request bodies up to 4 MiB; inline images are what fills one.
MAX_REQUEST_BYTES = 4 * 1024 * 1024
# Conservative cap on state size, well under drex-max's 262,144 input tokens per
# question. No Drex tokenizer is published, and a token never covers less than
# one byte, so UTF-8 bytes bound the token count.
STATE_BUDGET_BYTES = 131_072


class DrexCallResult(BaseModel):
    answers: dict[str, dict[str, Any]]
    request: str
    response: str | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    request_id: str
    truncated: bool
    truncation_note: str
    error: str = ""


class PreparedState(BaseModel):
    text: str
    truncated: bool
    truncation_note: str


async def call_drex(
    api_key: str,
    state: Any,
    questions: dict[str, Question],
    images: list[str] | None = None,
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> DrexCallResult:
    """Make one stateless System One request and report it verbatim.

    ``images`` are PNG, JPEG or WebP data URLs, at most ``MAX_IMAGES``; the
    question text refers to them as "Image 1", "Image 2" in list order.
    """
    check_images(images or [])
    prepared = prepare_state(state)
    body = _body(model, prepared.text, questions, images or [])
    payload = _encode(body)
    sent = _redacted_request(body)
    started = perf_counter()
    try:
        response = await _requests(api_key).post(
            SYSTEMONE_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=timeout),
        )
    except (TimeoutError, aiohttp.ClientError):
        return _failure(
            sent,
            None,
            prepared,
            started,
            request_id="",
            error="nace connection failed or timed out; no HTTP response was received.",
        )
    return _interpret(
        response.status,
        response.content,
        response.headers.get("x-request-id", ""),
        sent,
        prepared,
        started,
        questions,
    )


def check_images(images: list[str]) -> None:
    if len(images) > MAX_IMAGES:
        raise ValueError(
            f"nace accepts at most {MAX_IMAGES} images per request; "
            f"{len(images)} were given."
        )
    for index, image in enumerate(images, start=1):
        if not image.startswith(IMAGE_DATA_URL_PREFIXES):
            raise ValueError(f"Image {index} must be a PNG, JPEG or WebP image.")


def prepare_state(state: Any) -> PreparedState:
    text = _state_text(state)
    encoded = text.encode("utf-8")
    if len(encoded) <= STATE_BUDGET_BYTES:
        return PreparedState(text=text, truncated=False, truncation_note="")
    prefix = encoded[:STATE_BUDGET_BYTES].decode("utf-8", errors="ignore")
    note = (
        f"State truncated from {len(encoded):,} to "
        f"{len(prefix.encode('utf-8')):,} UTF-8 bytes, "
        f"a conservative limit of {STATE_BUDGET_BYTES:,} bytes that stays well under "
        "the model's token limits because a token never covers less than a byte."
    )
    if not isinstance(state, str):
        note += " Structured state was compact JSON text; its retained prefix may be incomplete JSON."
    return PreparedState(text=prefix, truncated=True, truncation_note=note)


def _interpret(
    status: int,
    content: bytes,
    header_request_id: str,
    sent: str,
    prepared: PreparedState,
    started: float,
    questions: dict[str, Question],
) -> DrexCallResult:
    if status != 200:
        failure = error_response(content)
        return _failure(
            sent,
            content,
            prepared,
            started,
            request_id=header_request_id or failure.request_id,
            error=f"nace API request failed (HTTP {status}). {describe(failure)}".strip(),
        )
    parsed = parse_response(content, questions)
    if parsed is None:
        return _failure(
            sent,
            content,
            prepared,
            started,
            request_id=header_request_id,
            error="nace returned an invalid response (HTTP 200).",
        )
    return _result(
        sent,
        content,
        prepared,
        started,
        answers={name: answer.model_dump() for name, answer in parsed.answers.items()},
        usage=parsed.usage,
        request_id=header_request_id or parsed.request_id,
        error="",
    )


def _failure(
    sent: str,
    content: bytes | None,
    prepared: PreparedState,
    started: float,
    *,
    request_id: str,
    error: str,
) -> DrexCallResult:
    """Represent unsuccessful responses without decision values or usage."""
    return _result(
        sent,
        content,
        prepared,
        started,
        answers={},
        usage=Usage(),
        request_id=request_id,
        error=error,
    )


def _result(
    sent: str,
    content: bytes | None,
    prepared: PreparedState,
    started: float,
    *,
    answers: dict[str, dict[str, Any]],
    usage: Usage,
    request_id: str,
    error: str,
) -> DrexCallResult:
    response, encoded = response_body(content)
    if encoded:
        error = (
            f"{error} Response body is not valid UTF-8; "
            "response holds the exact bytes as a Base64 data URL."
        ).strip()
    return DrexCallResult(
        answers=answers,
        request=sent,
        response=response,
        latency_ms=(perf_counter() - started) * 1_000,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        request_id=request_id,
        truncated=prepared.truncated,
        truncation_note=prepared.truncation_note,
        error=error,
    )


def _requests(api_key: str) -> Requests:
    # One attempt only: an overloaded (529) or failed call is reported, not retried.
    return Requests(
        trusted_origins=[NACE_API_URL],
        raise_for_status=False,
        extra_headers={"Authorization": f"Bearer {api_key}"},
        retry_max_attempts=1,
    )


def _body(
    model: str, state: str, questions: dict[str, Question], images: list[str]
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "state": state,
        "questions": {
            name: question.model_dump() for name, question in questions.items()
        },
    }
    if images:
        body["images"] = images
    return body


def _encode(body: dict[str, Any]) -> bytes:
    payload = _compact(body).encode("utf-8")
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError(
            f"Request body is {len(payload):,} bytes; nace accepts at most "
            f"{MAX_REQUEST_BYTES:,}. Send fewer or smaller images, or less state."
        )
    return payload


def _redacted_request(body: dict[str, Any]) -> str:
    """The request as sent, with inline image data replaced by its size."""
    images: list[str] = body.get("images", [])
    if not images:
        return _compact(body)
    summary = [
        f"<Image {index}: {image[5:image.index(';')]}, {len(image):,} characters>"
        for index, image in enumerate(images, start=1)
    ]
    return _compact({**body, "images": summary})


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _state_text(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    try:
        return _compact(state)
    except (TypeError, ValueError) as error:
        raise ValueError("State must be text or JSON-serializable data.") from error
