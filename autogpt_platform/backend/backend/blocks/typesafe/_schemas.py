from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from typesafe_sdk import Choice, Noul
from typesafe_sdk import NoulCriteria as SDKNoulCriteria
from typesafe_sdk import Score

from backend.blocks._base import BlockSchemaInput, BlockSchemaOutput
from backend.data.model import SchemaField

from ._config import TypeSafeCredentialsField, TypeSafeCredentialsInput

CALIBRATION = (
    "Levels define the scale: describe concrete evidence for each tier so a very "
    "good case does not reach the top. For hiring, distinguish terrible, bad, "
    "okay, excellent, brilliant, and one-in-a-thousand by what qualifies."
)


class JevInput(BlockSchemaInput):
    credentials: TypeSafeCredentialsInput = TypeSafeCredentialsField()
    state: Any = SchemaField(
        default=None,
        advanced=False,
        description=(
            "Text or JSON context shared by every question in this stateless call. "
            "JSON is compactly serialized; oversized context is truncated with a note."
        ),
    )


class JevOutput(BlockSchemaOutput):
    request: str = SchemaField(description="Verbatim JSON request body sent to Jev.")
    response: str | None = SchemaField(
        description=(
            "Verbatim UTF-8 response body from Jev, or null if no response arrived. "
            "Invalid UTF-8 is preserved as a lossless Base64 data URL with an error."
        )
    )
    latency_ms: float = SchemaField(description="Wall-clock API call duration in ms.")
    input_tokens: int | None = SchemaField(
        description="Input tokens reported by the API, or null if unavailable on failure."
    )
    output_tokens: int | None = SchemaField(
        description="Output tokens reported by the API, or null if unavailable on failure."
    )
    request_id: str = SchemaField(description="TypeSafe request ID response header.")
    truncated: bool = SchemaField(description="Whether any input text was truncated.")
    truncation_note: str = SchemaField(
        description="Truncation details, or an empty string."
    )
    error: str = SchemaField(
        default="", description="API failure details; emitted only when a call fails."
    )


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    question: str = Field(min_length=1)
    options: dict[str, str] = Field(min_length=2)

    def to_question(self) -> Choice:
        return Choice(instructions=self.question, criteria=self.options)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    question: str = Field(min_length=1)
    levels: list[str] = Field(min_length=2, description=CALIBRATION)

    def to_question(self) -> Score:
        return Score(instructions=self.question, criteria=self.levels)


class NoulCriteria(BaseModel):
    true: str
    false: str


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    question: str = Field(min_length=1)
    criteria: NoulCriteria | None = None

    def to_question(self) -> Noul:
        criteria: SDKNoulCriteria | None = (
            {"true": self.criteria.true, "false": self.criteria.false}
            if self.criteria is not None
            else None
        )
        return Noul(instructions=self.question, criteria=criteria)


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")
]
