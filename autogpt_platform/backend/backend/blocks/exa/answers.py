from typing import Optional

from exa_py import AsyncExa
from exa_py.api import AnswerResponse
from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    MediaFileType,
    SchemaField,
)

from ._config import exa


class AnswerCitation(BaseModel):
    """Citation model for answer endpoint."""

    id: str = SchemaField(description="The temporary ID for the document")
    url: str = SchemaField(description="The URL of the search result")
    title: Optional[str] = SchemaField(description="The title of the search result")
    author: Optional[str] = SchemaField(description="The author of the content")
    publishedDate: Optional[str] = SchemaField(
        description="An estimate of the creation date"
    )
    text: Optional[str] = SchemaField(description="The full text content of the source")
    image: Optional[MediaFileType] = SchemaField(
        description="The URL of the image associated with the result"
    )
    favicon: Optional[MediaFileType] = SchemaField(
        description="The URL of the favicon for the domain"
    )

    @classmethod
    def from_sdk(cls, sdk_citation) -> "AnswerCitation":
        """Convert SDK AnswerResult (dataclass) to our Pydantic model."""
        return cls(
            id=getattr(sdk_citation, "id", ""),
            url=getattr(sdk_citation, "url", ""),
            title=getattr(sdk_citation, "title", None),
            author=getattr(sdk_citation, "author", None),
            publishedDate=getattr(sdk_citation, "published_date", None),
            text=getattr(sdk_citation, "text", None),
            image=getattr(sdk_citation, "image", None),
            favicon=getattr(sdk_citation, "favicon", None),
        )


class ExaAnswerBlock(Block):
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

    class Output(BlockSchemaOutput):
        answer: str = SchemaField(
            description="The generated answer based on search results"
        )
        citations: list[AnswerCitation] = SchemaField(
            description="Search results used to generate the answer"
        )
        citation: AnswerCitation = SchemaField(
            description="Individual citation from the answer"
        )
        error: str = SchemaField(description="Error message if the request failed")

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
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        # Get answer using SDK (stream=False for blocks) - this IS async, needs await
        response = await aexa.answer(
            query=input_data.query, text=input_data.text, stream=False
        )

        # this should remain true as long as they don't start defaulting to streaming only.
        # provides a bit of safety for sdk updates.
        assert type(response) is AnswerResponse

        yield "answer", response.answer

        citations = [
            AnswerCitation.from_sdk(sdk_citation)
            for sdk_citation in response.citations or []
        ]

        yield "citations", citations
        for citation in citations:
            yield "citation", citation
