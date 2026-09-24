from typing import Any

from ._client import JevCallResult

TEST_OPTIONS = {"yes": "Evidence is present", "no": "Evidence is absent"}
TEST_LEVELS = ["No relevant evidence", "Concrete evidence meets the requirement"]
CHOICE_ANSWER = {
    "type": "choice",
    "choice": "yes",
    "probabilities": {"yes": 0.8, "no": 0.2},
    "confidence": 0.8,
}
SCORE_ANSWER = {
    "type": "score",
    "score": 0.8,
    "legend": {"0": TEST_LEVELS[0], "1": TEST_LEVELS[1]},
    "probabilities": {"0": 0.2, "1": 0.8},
    "confidence": 0.8,
}
TEST_TRANSPARENCY = [
    ("request", '{"state":"test","questions":{}}'),
    ("response", '{"answers":{}}'),
    ("latency_ms", 12.5),
    ("input_tokens", 10),
    ("output_tokens", 3),
    ("request_id", "test-request"),
    ("truncated", False),
    ("truncation_note", ""),
]


def mock_result(answers: dict[str, dict[str, Any]]) -> JevCallResult:
    return JevCallResult(answers=answers, **dict(TEST_TRANSPARENCY))
