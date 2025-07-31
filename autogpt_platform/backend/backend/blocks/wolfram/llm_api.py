from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockCostType,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    ProviderBuilder,
    SchemaField,
)

from ._api import llm_api_call

wolfram = (
    ProviderBuilder("wolfram")
    .with_api_key("WOLFRAM_APP_ID", "Wolfram Alpha App ID")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)


class AskWolframBlock(Block):
    """
    Ask Wolfram Alpha a question.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = wolfram.credentials_field(
            description="Wolfram Alpha API credentials"
        )
        question: str = SchemaField(description="The question to ask")

    class Output(BlockSchema):
        answer: str = SchemaField(description="The answer to the question")

    def __init__(self):
        super().__init__(
            id="b7710ce4-68ef-4e82-9a2f-f0b874ef9c7d",
            description="Ask Wolfram Alpha a question",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        answer = await llm_api_call(credentials, input_data.question)
        yield "answer", answer
