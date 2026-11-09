import json
import os
from typing import Any

from pydantic import BaseModel
from typesafe_sdk import Choice, Noul, Score
from typesafe_sdk._core.json import serialize
from typesafe_sdk.constants import DEFAULT_MODEL, DEFAULT_MODEL_ENV

INPUT_TOKEN_BUDGET = 32_000
TOKEN_RESERVE = 1_024
MAX_REQUEST_BYTES = INPUT_TOKEN_BUDGET - TOKEN_RESERVE


class PreparedState(BaseModel):
    state: str
    truncated: bool
    truncation_note: str


def prepare_state(
    state: Any, questions: dict[str, Choice | Score | Noul]
) -> PreparedState:
    if not questions:
        raise ValueError("At least one Jev question is required.")
    text, json_state = _state_text(state)
    framing_bytes = len(
        serialize(
            {
                "state": "",
                "model": os.environ.get(DEFAULT_MODEL_ENV, "").strip() or DEFAULT_MODEL,
                "questions": questions,
            }
        )
    )
    if framing_bytes > MAX_REQUEST_BYTES:
        raise ValueError(
            "Jev questions and request framing exceed the shared request budget "
            f"of {MAX_REQUEST_BYTES:,} UTF-8 bytes. Shorten the questions or criteria."
        )
    available = MAX_REQUEST_BYTES - framing_bytes
    if len(serialize(text)) - 2 <= available:
        return PreparedState(state=text, truncated=False, truncation_note="")
    prefix = _fitting_prefix(text, available)
    return PreparedState(
        state=prefix,
        truncated=True,
        truncation_note=_truncation_note(text, prefix, json_state),
    )


def _state_text(state: Any) -> tuple[str, bool]:
    match state:
        case str():
            return state, False
        case _:
            try:
                return (
                    json.dumps(
                        state,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    True,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Jev state must be text or JSON-serializable data."
                ) from error


def _fitting_prefix(text: str, available: int) -> str:
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(serialize(text[:middle])) - 2 <= available:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _truncation_note(original: str, prefix: str, json_state: bool) -> str:
    note = (
        f"State truncated from {len(original):,} to {len(prefix):,} characters "
        f"({len(original.encode('utf-8')):,} to {len(prefix.encode('utf-8')):,} UTF-8 bytes). "
        f"The entire serialized request is limited to {MAX_REQUEST_BYTES:,} UTF-8 bytes, "
        f"including all questions and JSON framing, with a {TOKEN_RESERVE:,}-token reserve "
        f"against the approximately {INPUT_TOKEN_BUDGET:,}-token shared context. "
        "This conservative byte proxy is not an exact Jev token count; "
        "no tokenizer is documented."
    )
    if json_state:
        note += " Structured state was compact JSON text; its retained prefix may be incomplete JSON."
    return note
