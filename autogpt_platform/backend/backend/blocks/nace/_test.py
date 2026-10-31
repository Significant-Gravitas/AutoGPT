from typing import Any

from ._client import DrexCallResult

TEST_OPTIONS = {"yes": "Evidence is present", "no": "Evidence is absent"}
TEST_LEVELS = ["No relevant evidence", "Concrete evidence meets the requirement"]
TEST_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
CHOICE_ANSWER = {
    "type": "choice",
    "choice": "yes",
    "probabilities": {"yes": 0.8, "no": 0.2},
    "confidence": 0.8,
}
SCORE_ANSWER = {
    "type": "score",
    "score": 0.8,
    "legend": TEST_LEVELS,
    "probabilities": {"0": 0.2, "1": 0.8},
    "confidence": 0.8,
}
NOUL_ANSWER = {"type": "noul", "noul": 0.8}
TEST_TRANSPARENCY = [
    ("request", '{"model":"drex-max","state":"test","questions":{}}'),
    ("response", '{"answers":{}}'),
    ("latency_ms", 12.5),
    ("input_tokens", 10),
    ("output_tokens", 3),
    ("request_id", "req_test"),
    ("truncated", False),
    ("truncation_note", ""),
]


def mock_result(answers: dict[str, dict[str, Any]]) -> DrexCallResult:
    return DrexCallResult(answers=answers, **dict(TEST_TRANSPARENCY))
