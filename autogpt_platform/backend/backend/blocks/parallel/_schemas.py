from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from backend.sdk import (
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import parallel

GROUPING = (
    "Questions in one request share one set of web evidence, so group questions "
    "about the same entity or topic and use separate blocks for unrelated ones."
)


class ParallelCitation(BaseModel):
    """A web source included in the evidence for a Decisions request."""

    url: str = Field(description="Source page URL")
    title: str | None = Field(default=None, description="Page title, if available")
    excerpts: list[str] = Field(
        default_factory=list,
        description="Passages used as evidence; empty when excerpts are turned off",
    )


class ParallelDecisionInput(BlockSchemaInput):
    credentials: CredentialsMetaInput = parallel.credentials_field(
        description="Your Parallel API key, from platform.parallel.ai"
    )
    context: Any = SchemaField(
        default=None,
        advanced=False,
        description=(
            "Text or JSON describing what the question is about, e.g. "
            '{"company": "Clay", "website": "https://www.clay.com"}. '
            "Parallel searches the web for current evidence about it."
        ),
    )
    include_excerpts: bool = SchemaField(
        default=True,
        advanced=True,
        description=(
            "Return the evidence passages with each citation. Turn off to keep "
            "outputs small; the decision still uses the passages."
        ),
    )


class ParallelDecisionOutput(BlockSchemaOutput):
    citations: list[ParallelCitation] = SchemaField(
        description=(
            "Web sources used as evidence for the whole request. They are not "
            "mapped to individual answers."
        )
    )
    decision_id: str = SchemaField(description="Parallel's ID for this request")


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    question: str = Field(min_length=1)
    options: dict[str, str] = Field(min_length=2, max_length=255)

    def to_api(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.question,
            "criteria": self.options,
        }


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    question: str = Field(min_length=1)
    levels: list[str] = Field(min_length=2, max_length=10)

    def to_api(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.question, "criteria": self.levels}


class NoulCriteria(BaseModel):
    true: str
    false: str


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    question: str = Field(min_length=1)
    criteria: NoulCriteria | None = None

    def to_api(self) -> dict[str, Any]:
        body: dict[str, Any] = {"type": "noul", "instructions": self.question}
        if self.criteria is not None:
            body["criteria"] = self.criteria.model_dump()
        return body


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")
]
