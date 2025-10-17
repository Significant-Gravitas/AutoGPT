from typing import Optional

from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    MediaFileType,
    Requests,
    SchemaField,
)

from ._config import exa
from .helpers import CostDollars


class AnswerCitation(BaseModel):
    """Citation model for answer endpoint."""

    id: str = SchemaField(description="The temporary ID for the document")
    url: str = SchemaField(description="The URL of the search result")
    title: Optional[str] = SchemaField(
        description="The title of the search result", default=None
    )
    author: Optional[str] = SchemaField(
        description="The author of the content", default=None
    )
    publishedDate: Optional[str] = SchemaField(
        description="An estimate of the creation date", default=None
    )
    text: Optional[str] = SchemaField(
        description="The full text content of the source", default=None
    )
    image: Optional[MediaFileType] = SchemaField(
        description="The URL of the image associated with the result", default=None
    )
    favicon: Optional[MediaFileType] = SchemaField(
        description="The URL of the favicon for the domain", default=None
    )


class ExaAnswerBlock(Block):
    class Input(BlockSchema):
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

    class Output(BlockSchema):
        answer: str = SchemaField(
            description="The generated answer based on search results"
        )
        citations: list[AnswerCitation] = SchemaField(
            description="Search results used to generate the answer",
            default_factory=list,
        )
        citation: AnswerCitation = SchemaField(
            description="Individual citation from the answer"
        )
        cost_dollars: Optional[CostDollars] = SchemaField(
            description="Cost breakdown for the request", default=None
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="b79ca4cc-9d5e-47d1-9d4f-e3a2d7f28df5",
            description="Get an LLM answer to a question informed by Exa search results",
            categories={BlockCategory.SEARCH, BlockCategory.AI},
            input_schema=ExaAnswerBlock.Input,
            output_schema=ExaAnswerBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/answer"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        payload = {
            "query": input_data.query,
            "text": input_data.text,
            # We don't support streaming in blocks
            "stream": False,
        }

        try:
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()

            # Yield the answer
            if "answer" in data:
                yield "answer", data["answer"]

            # Yield citations as a list
            if "citations" in data:
                yield "citations", data["citations"]

                # Also yield individual citations
                for citation in data["citations"]:
                    yield "citation", citation

            # Yield cost information if present
            if "costDollars" in data:
                yield "cost_dollars", data["costDollars"]

        except Exception as e:
            yield "error", str(e)
