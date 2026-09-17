import base64
import json
from time import perf_counter
from typing import Any

from pydantic import BaseModel
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy, Score
from typesafe_sdk._core.errors import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
)
from typesafe_sdk._core.json import serialize
from typesafe_sdk._core.response_types import SystemOneResponse
from typesafe_sdk.constants import DEFAULT_BASE_URL

from ._budget import PreparedState, prepare_state
from ._wire import WireCapture, capture_wire


class JevCallResult(BaseModel):
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


async def call_jev(
    api_key: str, state: Any, questions: dict[str, Choice | Score | Noul]
) -> JevCallResult:
    prepared = prepare_state(state, questions)
    async with AsyncTypeSafeClient(
        api_key=api_key, base_url=DEFAULT_BASE_URL, retry=RetryPolicy(max_retries=0)
    ) as client:
        with capture_wire() as wire:
            started = perf_counter()
            try:
                result = await client.system_one(
                    state=prepared.state, questions=questions
                )
            except TypeSafeAPIResponseValidationError as error:
                return _failure(
                    wire,
                    prepared,
                    started,
                    f"Jev returned an invalid response (HTTP {error.status}).",
                    error.request_id or "",
                )
            except TypeSafeAPIError as error:
                return _failure(
                    wire,
                    prepared,
                    started,
                    f"Jev API request failed (HTTP {error.status}).",
                    error.request_id or "",
                )
            except TypeSafeAPIConnectionError:
                return _failure(
                    wire,
                    prepared,
                    started,
                    "Jev connection failed or timed out; no HTTP response was received.",
                )
            except UnicodeDecodeError:
                if wire.response is None:
                    raise
                return _failure(
                    wire,
                    prepared,
                    started,
                    "Jev returned an invalid response.",
                    wire.request_id,
                )
            return _success(result, wire, prepared, started)


def _success(
    result: SystemOneResponse,
    wire: WireCapture,
    prepared: PreparedState,
    started: float,
) -> JevCallResult:
    request_id = result.raw_http_response.headers.get("x-typesafe-request-id", "")
    errors = [
        message
        for missing, message in (
            (
                result.usage.input_tokens is None or result.usage.output_tokens is None,
                "Jev response did not report token usage.",
            ),
            (not request_id, "Jev response did not include a request ID."),
        )
        if missing
    ]
    return _result(
        wire,
        prepared,
        started,
        answers=json.loads(serialize(result.answers)),
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        request_id=request_id,
        error=" ".join(errors),
    )


def _failure(
    wire: WireCapture,
    prepared: PreparedState,
    started: float,
    error: str,
    request_id: str = "",
) -> JevCallResult:
    return _result(
        wire,
        prepared,
        started,
        answers={},
        input_tokens=None,
        output_tokens=None,
        request_id=request_id,
        error=error,
    )


def _result(
    wire: WireCapture,
    prepared: PreparedState,
    started: float,
    *,
    answers: dict[str, dict[str, Any]],
    input_tokens: int | None,
    output_tokens: int | None,
    request_id: str,
    error: str,
) -> JevCallResult:
    if wire.request is None or (wire.response is None and not error):
        raise RuntimeError(
            "Jev SDK wire capture failed; request transparency is unavailable."
        )
    response, encoded = _response_body(wire.response)
    if encoded:
        error = (
            f"{error} Response body is not valid UTF-8; "
            "response contains the exact bytes as a Base64 data URL."
        ).strip()
    return JevCallResult(
        answers=answers,
        request=wire.request.decode("utf-8"),
        response=response,
        latency_ms=(perf_counter() - started) * 1_000,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        request_id=request_id,
        truncated=prepared.truncated,
        truncation_note=prepared.truncation_note,
        error=error,
    )


def _response_body(body: bytes | None) -> tuple[str | None, bool]:
    if body is None:
        return None, False
    try:
        return body.decode("utf-8"), False
    except UnicodeDecodeError:
        encoded = base64.b64encode(body).decode("ascii")
        return f"data:application/octet-stream;base64,{encoded}", True
