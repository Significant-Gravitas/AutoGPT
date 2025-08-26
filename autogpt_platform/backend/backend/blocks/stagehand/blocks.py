import logging
from enum import Enum
from typing import Optional

from stagehand import Stagehand

from backend.blocks.llm import (
    MODEL_METADATA,
    AICredentials,
    AICredentialsField,
    LlmModel,
    ModelMetadata,
)
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import stagehand as stagehand_provider

logger = logging.getLogger(__name__)


class StagehandRecommendedLlmModel(str, Enum):
    """
    This is subset of LLModel from autogpt_platform/backend/backend/blocks/llm.py

    It contains only the models recommended by Stagehand
    """

    # OpenAI
    GPT41 = "gpt-4.1-2025-04-14"
    GPT41_MINI = "gpt-4.1-mini-2025-04-14"

    # Anthropic
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"

    @property
    def provider_name(self) -> str:
        """
        Returns the provider name for the model in the required format for Stagehand:
        provider/model_name
        """
        model_metadata = MODEL_METADATA[LlmModel(self.value)]
        model_name = self.value

        if len(model_name.split("/")) == 1 and not self.value.startswith(
            model_metadata.provider
        ):
            assert (
                model_metadata.provider != "open_router"
            ), "Logic failed and open_router provider attempted to be prepended to model name! in stagehand/_config.py"
            model_name = f"{model_metadata.provider}/{model_name}"

        return model_name

    @property
    def provider(self) -> str:
        return MODEL_METADATA[LlmModel(self.value)].provider

    @property
    def metadata(self) -> ModelMetadata:
        return MODEL_METADATA[LlmModel(self.value)]

    @property
    def context_window(self) -> int:
        return MODEL_METADATA[LlmModel(self.value)].context_window

    @property
    def max_output_tokens(self) -> int | None:
        return MODEL_METADATA[LlmModel(self.value)].max_output_tokens


class StagehandObserveBlock(Block):
    class Input(BlockSchema):
        # Browserbase credentials (Stagehand provider) or raw API key
        stagehand_credentials: CredentialsMetaInput = (
            stagehand_provider.credentials_field(
                description="Stagehand/Browserbase API key"
            )
        )
        browserbase_project_id: str = SchemaField(
            description="Browserbase project ID (required if using Browserbase)",
        )
        # Model selection and credentials (provider-discriminated like llm.py)
        model: StagehandRecommendedLlmModel = SchemaField(
            title="LLM Model",
            description="LLM to use for Stagehand (provider is inferred)",
            default=StagehandRecommendedLlmModel.CLAUDE_3_7_SONNET,
            advanced=False,
        )
        model_credentials: AICredentials = AICredentialsField()
        url: str = SchemaField(
            description="URL to navigate to.",
        )
        instruction: str = SchemaField(
            description="Natural language description of elements or actions to discover.",
        )
        iframes: bool = SchemaField(
            description="Whether to search within iframes. If True, Stagehand will search for actions within iframes.",
            default=True,
        )
        domSettleTimeoutMs: int = SchemaField(
            description="Timeout in milliseconds for DOM settlement.Wait longer for dynamic content",
            default=45000,
        )

    class Output(BlockSchema):
        selector: str = SchemaField(description="XPath selector to locate element.")
        description: str = SchemaField(description="Human-readable description")
        method: str | None = SchemaField(description="Suggested action method")
        arguments: list[str] | None = SchemaField(
            description="Additional action parameters"
        )

    def __init__(self):
        super().__init__(
            id="d3863944-0eaf-45c4-a0c9-63e0fe1ee8b9",
            description="Find suggested actions for your workflows",
            categories={BlockCategory.AI, BlockCategory.DEVELOPER_TOOLS},
            input_schema=StagehandObserveBlock.Input,
            output_schema=StagehandObserveBlock.Output,
        )

    async def run(
        self,
        input_data: Input,
        *,
        stagehand_credentials: APIKeyCredentials,
        model_credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:

        stagehand = Stagehand(
            api_key=stagehand_credentials.api_key.get_secret_value(),
            project_id=input_data.browserbase_project_id,
            model=input_data.model.provider_name,
            model_api_key=model_credentials.api_key.get_secret_value(),
        )

        await stagehand.init()

        page = stagehand.page

        assert page is not None, "Stagehand page is not initialized"

        await page.goto(input_data.url)

        observe_results = await page.observe(
            input_data.instruction,
            iframes=input_data.iframes,
            domSettleTimeoutMs=input_data.domSettleTimeoutMs,
        )
        for result in observe_results:
            yield "selector", result.selector
            yield "description", result.description
            yield "method", result.method
            yield "arguments", result.arguments


class StagehandActBlock(Block):
    class Input(BlockSchema):
        # Browserbase credentials (Stagehand provider) or raw API key
        stagehand_credentials: CredentialsMetaInput = (
            stagehand_provider.credentials_field(
                description="Stagehand/Browserbase API key"
            )
        )
        browserbase_project_id: str = SchemaField(
            description="Browserbase project ID (required if using Browserbase)",
        )
        # Model selection and credentials (provider-discriminated like llm.py)
        model: StagehandRecommendedLlmModel = SchemaField(
            title="LLM Model",
            description="LLM to use for Stagehand (provider is inferred)",
            default=StagehandRecommendedLlmModel.CLAUDE_3_7_SONNET,
            advanced=False,
        )
        model_credentials: AICredentials = AICredentialsField()
        url: str = SchemaField(
            description="URL to navigate to.",
        )
        action: list[str] = SchemaField(
            description="Action to perform. Suggested actions are: click, fill, type, press, scroll, select from dropdown. For multi-step actions, add an entry for each step.",
        )
        variables: dict[str, str] = SchemaField(
            description="Variables to use in the action. Variables contains data you want the action to use.",
            default_factory=dict,
        )
        iframes: bool = SchemaField(
            description="Whether to search within iframes. If True, Stagehand will search for actions within iframes.",
            default=True,
        )
        domSettleTimeoutMs: int = SchemaField(
            description="Timeout in milliseconds for DOM settlement.Wait longer for dynamic content",
            default=45000,
        )
        timeoutMs: int = SchemaField(
            description="Timeout in milliseconds for DOM ready. Extended timeout for slow-loading forms",
            default=60000,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the action was completed successfully"
        )
        message: str = SchemaField(description="Details about the action’s execution.")
        action: str = SchemaField(description="Action performed")

    def __init__(self):
        super().__init__(
            id="86eba68b-9549-4c0b-a0db-47d85a56cc27",
            description="Interact with a web page by performing actions on a web page. Use it to build self-healing and deterministic automations that adapt to website chang.",
            categories={BlockCategory.AI, BlockCategory.DEVELOPER_TOOLS},
            input_schema=StagehandActBlock.Input,
            output_schema=StagehandActBlock.Output,
        )

    async def run(
        self,
        input_data: Input,
        *,
        stagehand_credentials: APIKeyCredentials,
        model_credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:

        stagehand = Stagehand(
            api_key=stagehand_credentials.api_key.get_secret_value(),
            project_id=input_data.browserbase_project_id,
            model=input_data.model.provider_name,
            model_api_key=model_credentials.api_key.get_secret_value(),
        )

        await stagehand.init()

        page = stagehand.page

        assert page is not None, "Stagehand page is not initialized"

        await page.goto(input_data.url)
        for action in input_data.action:
            action_results = await page.act(
                action,
                variables=input_data.variables,
                iframes=input_data.iframes,
                domSettleTimeoutMs=input_data.domSettleTimeoutMs,
                timeoutMs=input_data.timeoutMs,
            )
            yield "success", action_results.success
            yield "message", action_results.message
            yield "action", action_results.action


class StagehandExtractBlock(Block):
    class Input(BlockSchema):
        # Browserbase credentials (Stagehand provider) or raw API key
        stagehand_credentials: CredentialsMetaInput = (
            stagehand_provider.credentials_field(
                description="Stagehand/Browserbase API key"
            )
        )
        browserbase_project_id: str = SchemaField(
            description="Browserbase project ID (required if using Browserbase)",
        )
        # Model selection and credentials (provider-discriminated like llm.py)
        model: StagehandRecommendedLlmModel = SchemaField(
            title="LLM Model",
            description="LLM to use for Stagehand (provider is inferred)",
            default=StagehandRecommendedLlmModel.CLAUDE_3_7_SONNET,
            advanced=False,
        )
        model_credentials: AICredentials = AICredentialsField()
        url: str = SchemaField(
            description="URL to navigate to.",
        )
        instruction: str = SchemaField(
            description="Natural language description of elements or actions to discover.",
        )
        iframes: bool = SchemaField(
            description="Whether to search within iframes. If True, Stagehand will search for actions within iframes.",
            default=True,
        )
        domSettleTimeoutMs: int = SchemaField(
            description="Timeout in milliseconds for DOM settlement.Wait longer for dynamic content",
            default=45000,
        )

    class Output(BlockSchema):
        extraction: str = SchemaField(description="Extracted data from the page.")

    def __init__(self):
        super().__init__(
            id="fd3c0b18-2ba6-46ae-9339-fcb40537ad98",
            description="Extract structured data from a webpage.",
            categories={BlockCategory.AI, BlockCategory.DEVELOPER_TOOLS},
            input_schema=StagehandExtractBlock.Input,
            output_schema=StagehandExtractBlock.Output,
        )

    async def run(
        self,
        input_data: Input,
        *,
        stagehand_credentials: APIKeyCredentials,
        model_credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:

        stagehand = Stagehand(
            api_key=stagehand_credentials.api_key.get_secret_value(),
            project_id=input_data.browserbase_project_id,
            model=input_data.model.provider_name,
            model_api_key=model_credentials.api_key.get_secret_value(),
        )

        await stagehand.init()

        page = stagehand.page

        assert page is not None, "Stagehand page is not initialized"

        await page.goto(input_data.url)
        extraction = await page.extract(
            input_data.instruction,
            iframes=input_data.iframes,
            domSettleTimeoutMs=input_data.domSettleTimeoutMs,
        )
        yield "extraction", extraction["extraction"]
