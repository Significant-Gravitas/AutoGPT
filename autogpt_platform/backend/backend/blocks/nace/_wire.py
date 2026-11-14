"""Wire models for nace System One responses."""

import base64
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROBABILITY_SUM_TOLERANCE = 1e-3

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Answer(BaseModel):
    """Validate decision fields while retaining additional upstream metadata."""

    model_config = ConfigDict(strict=True, extra="allow")


class ChoiceAnswer(Answer):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class ScoreAnswer(Answer):
    type: Literal["score"]
    score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    legend: list[str]
    probabilities: dict[str, Probability]
    confidence: Probability


class NoulAnswer(Answer):
    type: Literal["noul"]
    noul: Probability


TypedAnswer = Annotated[
    ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")
]


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str | None]


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: str


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class SystemOneResponse(BaseModel):
    answers: dict[str, TypedAnswer]
    usage: Usage = Usage()
    request_id: str = ""


def parse_response(
    content: bytes, questions: dict[str, Question]
) -> SystemOneResponse | None:
    """Accept only typed responses that answer every requested question."""
    try:
        parsed = SystemOneResponse.model_validate_json(content)
    except ValueError:
        return None
    if not all(
        name in parsed.answers and _answer_matches(question, parsed.answers[name])
        for name, question in questions.items()
    ):
        return None
    return parsed


def _answer_matches(question: Question, answer: TypedAnswer) -> bool:
    """Require an answer for the requested type and choice/score domain."""
    match question, answer:
        case ChoiceQuestion(), ChoiceAnswer():
            return answer.choice in question.criteria and _is_distribution(
                answer.probabilities, list(question.criteria)
            )
        case ScoreQuestion(), ScoreAnswer():
            return (
                answer.score <= len(question.criteria) - 1
                and answer.legend == question.criteria
                and _is_distribution(
                    answer.probabilities,
                    [str(index) for index in range(len(question.criteria))],
                )
            )
        case NoulQuestion(), NoulAnswer():
            return True
        case _:
            return False


def _is_distribution(probabilities: dict[str, float], keys: list[str]) -> bool:
    """One probability per requested option or level, summing to 1 within rounding."""
    return (
        sorted(probabilities) == sorted(keys)
        and abs(sum(probabilities.values()) - 1) <= PROBABILITY_SUM_TOLERANCE
    )


class Issue(BaseModel):
    path: str = ""
    message: str = ""


class APIError(BaseModel):
    message: str = ""
    issues: list[Issue] = []


class ErrorResponse(BaseModel):
    error: APIError = APIError()
    request_id: str = ""


def error_response(content: bytes) -> ErrorResponse:
    try:
        return ErrorResponse.model_validate_json(content)
    except ValidationError:
        return ErrorResponse()


def describe(failure: ErrorResponse) -> str:
    """The API's message plus any issue it does not already spell out."""
    message = failure.error.message
    issues = "; ".join(
        f"{issue.path}: {issue.message}".strip(": ")
        for issue in failure.error.issues
        if issue.message not in message
    )
    return " ".join(part for part in (message, issues) if part)


def response_body(content: bytes | None) -> tuple[str | None, bool]:
    if content is None:
        return None, False
    try:
        return content.decode("utf-8"), False
    except UnicodeDecodeError:
        encoded = base64.b64encode(content).decode("ascii")
        return f"data:application/octet-stream;base64,{encoded}", True
