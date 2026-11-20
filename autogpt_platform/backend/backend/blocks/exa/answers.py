import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    MediaFileType,
    SchemaField,
)

from ._config import exa
from ._exa_rest import EXA_API_BASE, ExaRestMixin
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .helpers import CostDollars, merge_exa_cost, parse_cost_dollars
from .request_checks import checked_country_code, checked_user_identifier


class AnswerModel(str, Enum):
    EXA = "exa"
    EXA_PRO = "exa-pro"
    EXA_FAST = "exa-fast"
    EXA_RESEARCH = "exa-research"


class AnswerCitation(BaseModel):
    """Citation model for answer endpoint."""

    model_config = ConfigDict(extra="ignore")

    id: str = SchemaField(default="", description="The temporary ID for the document")
    url: str = SchemaField(description="The URL of the search result")
    title: str | None = SchemaField(
        default=None, description="The title of the search result"
    )
    author: str | None = SchemaField(
        default=None, description="The author of the content"
    )
    publishedDate: str | None = SchemaField(
        default=None, description="An estimate of the creation date"
    )
    text: str | None = SchemaField(
        default=None, description="The full text content of the source"
    )
    image: MediaFileType | None = SchemaField(
        default=None, description="The URL of the image associated with the result"
    )
    favicon: MediaFileType | None = SchemaField(
        default=None, description="The URL of the favicon for the domain"
    )


class ExaAnswerResponse(BaseModel):
    requestId: str = ""
    answer: str | dict[str, Any] = ""
    citations: list[AnswerCitation] = []


class ExaAnswerBlock(ExaRestMixin, Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        query: str = SchemaField(
            description="The question or query to answer",
            placeholder="What is the latest valuation of SpaceX?",
        )
        text: bool = SchemaField(
            description="Include full text content in the search results used for the answer",
            default=True,
        )
        model: AnswerModel = SchemaField(
            title="Model",
            description="The Exa model that writes the answer: 'exa' (default), "
            "'exa-pro', 'exa-fast' or 'exa-research'. Exa does not document how "
            "they differ; all are billed at the /answer rate.",
            default=AnswerModel.EXA,
            advanced=True,
        )
        output_schema: dict[str, Any] | None = SchemaField(
            description="A JSON Schema (draft 7) for a structured answer, e.g. "
            '{"type": "object", "properties": {"valuation": {"type": "string"}}}. '
            "When set, the answer arrives as an object in Answer Structured (and "
            "as JSON text in Answer).",
            default=None,
            advanced=True,
        )
        system_prompt: str | None = SchemaField(
            description="Extra instructions for the answer, e.g. source "
            "preferences, tone or length",
            placeholder="Answer in two sentences and prefer official sources.",
            default=None,
            advanced=True,
        )
        user_location: str | None = SchemaField(
            description="The end user's two-letter ISO country code (e.g. 'US') "
            "to localize the search",
            placeholder="US",
            default=None,
            advanced=True,
        )
        user_identifier: str | None = SchemaField(
            description="Opaque ID of the end user this request is for (1-64 "
            "letters, digits, '_' or '-', e.g. a UUID or hash). Never an email "
            "or name.",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        answer: str = SchemaField(
            description="The generated answer. With an Output Schema, the "
            "structured answer as JSON text."
        )
        answer_structured: dict[str, Any] = SchemaField(
            description="The answer as an object, when Exa returns one (normally "
            "only when Output Schema is set)"
        )
        citations: list[AnswerCitation] = SchemaField(
            description="Search results used to generate the answer"
        )
        citation: AnswerCitation = SchemaField(
            description="Individual citation from the answer"
        )
        request_id: str = SchemaField(description="Exa's ID for this request")
        cost_dollars: CostDollars = SchemaField(
            title="Cost", description="Cost breakdown for the request"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="b79ca4cc-9d5e-47d1-9d4f-e3a2d7f28df5",
            description="Get an LLM answer to a question informed by Exa search "
            "results, with citations; optionally as structured JSON",
            categories={BlockCategory.SEARCH, BlockCategory.AI},
            input_schema=ExaAnswerBlock.Input,
            output_schema=ExaAnswerBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "What is the capital of France?",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("answer", "Paris"),
                ("citations", [AnswerCitation(id="c1", url="https://example.com")]),
                ("citation", AnswerCitation(id="c1", url="https://example.com")),
                ("request_id", "req-1"),
                ("cost_dollars", CostDollars(total=0.005)),
            ],
            test_mock={
                "call_exa": lambda *args, **kwargs: {
                    "requestId": "req-1",
                    "answer": "Paris",
                    "citations": [{"id": "c1", "url": "https://example.com"}],
                    "costDollars": {"total": 0.005},
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials,
            "POST",
            f"{EXA_API_BASE}/answer",
            body=build_answer_payload(input_data),
        )
        response = ExaAnswerResponse.model_validate(data)

        if isinstance(response.answer, dict):
            yield "answer", json.dumps(response.answer)
            yield "answer_structured", response.answer
        else:
            yield "answer", response.answer
        yield "citations", response.citations
        for citation in response.citations:
            yield "citation", citation
        if response.requestId:
            yield "request_id", response.requestId
        cost = parse_cost_dollars(data)
        if cost is not None:
            yield "cost_dollars", cost
        merge_exa_cost(self, data)


def build_answer_payload(input_data: ExaAnswerBlock.Input) -> dict[str, Any]:
    """Build the non-streaming ``/answer`` body, omitting unset options."""
    optional: dict[str, Any] = {
        "model": (
            None if input_data.model == AnswerModel.EXA else input_data.model.value
        ),
        "outputSchema": input_data.output_schema,
        "systemPrompt": input_data.system_prompt,
        "userLocation": checked_country_code(input_data.user_location),
        "userIdentifier": checked_user_identifier(input_data.user_identifier),
    }
    return {
        "query": input_data.query,
        "text": input_data.text,
        **{key: value for key, value in optional.items() if value},
    }
