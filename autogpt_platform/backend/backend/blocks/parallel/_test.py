from typing import Any

from ._schemas import ParallelCitation

TEST_CONTEXT = {"company": "Clay", "website": "https://www.clay.com"}
TEST_OPTIONS = {
    "documented": "Clay publicly documents an official CLI.",
    "unavailable": "Clay explicitly states that no official CLI is offered.",
}
TEST_LEVELS = [
    "No official CLI is documented.",
    "An official CLI is documented.",
]
NOUL_ANSWER = {"type": "noul", "noul": 0.9}
CHOICE_ANSWER = {
    "type": "choice",
    "choice": "documented",
    "probabilities": {"documented": 0.97, "unavailable": 0.03},
}
SCORE_ANSWER = {
    "type": "score",
    "score": 0.8,
    "legend": {"0": TEST_LEVELS[0], "1": TEST_LEVELS[1]},
    "probabilities": {"0": 0.2, "1": 0.8},
}
REFUSAL_ANSWER = {"type": "refusal"}
TEST_CITATION = {
    "url": "https://developers.clay.com/concepts/cli-basics",
    "title": "CLI basics",
    "excerpts": ["The Clay CLI (clay) is JSON-first."],
}
TEST_DECISION_ID = "decisions_34b4be1b1405e19ba5516d9bb76c56ad"
TEST_EVIDENCE = [
    ("citations", [ParallelCitation(**TEST_CITATION)]),
    ("decision_id", TEST_DECISION_ID),
]


def mock_response(answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": "fast",
        "answers": answers,
        "usage": {"input_tokens": 4212, "output_tokens": 0},
        "citations": [TEST_CITATION],
        "decision_id": TEST_DECISION_ID,
    }
