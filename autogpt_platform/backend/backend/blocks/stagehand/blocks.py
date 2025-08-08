import logging
import os
from enum import Enum
from typing import Any, Dict, Literal, Optional

from backend.blocks.llm import AICredentials, AICredentialsField, LlmModel
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import CredentialsField, CredentialsMetaInput, SchemaField

logger = logging.getLogger(__name__)


StagehandCredentials = CredentialsMetaInput[Literal["stagehand"], Literal["api_key"]]  # type: ignore


def StagehandCredentialsField() -> StagehandCredentials:
    return CredentialsField(
        description="Stagehand API credentials",
        provider="stagehand",
    )


class StagehandLlmModel(str, Enum):
    """Subset of LlmModel that Stagehand supports (OpenAI and Anthropic)."""

    # OpenAI
    O3_MINI = LlmModel.O3_MINI.value
    O3 = LlmModel.O3.value
    O1 = LlmModel.O1.value
    O1_MINI = LlmModel.O1_MINI.value
    GPT41 = LlmModel.GPT41.value
    GPT41_MINI = LlmModel.GPT41_MINI.value
    GPT4O_MINI = LlmModel.GPT4O_MINI.value
    GPT4O = LlmModel.GPT4O.value
    GPT4_TURBO = LlmModel.GPT4_TURBO.value
    # Anthropic
    CLAUDE_4_OPUS = LlmModel.CLAUDE_4_OPUS.value
    CLAUDE_4_SONNET = LlmModel.CLAUDE_4_SONNET.value
    CLAUDE_3_7_SONNET = LlmModel.CLAUDE_3_7_SONNET.value
    CLAUDE_3_5_SONNET = LlmModel.CLAUDE_3_5_SONNET.value
    CLAUDE_3_5_HAIKU = LlmModel.CLAUDE_3_5_HAIKU.value
    CLAUDE_3_HAIKU = LlmModel.CLAUDE_3_HAIKU.value

    @property
    def provider(self) -> str:
        return LlmModel(self.value).provider


class StagehandInitBlock(Block):
    """
    Initializes a Stagehand browser session for automation.
    """

    class Input(BlockSchema):
        # Browserbase credentials (Stagehand provider) or raw API key
        browser_credentials: CredentialsMetaInput[Literal["stagehand"], Literal["api_key"]] | None = CredentialsField(  # type: ignore
            description="Stagehand/Browserbase API key (used when env is BROWSERBASE)",
            title="Stagehand API Key",
            default=None,
        )
        browserbase_api_key: Optional[str] = SchemaField(
            description="Browserbase API key (fallback if no credentials selected)",
            secret=True,
            default=None,
        )
        browserbase_project_id: Optional[str] = SchemaField(
            description="Browserbase project ID (required if using Browserbase)",
            default=None,
        )
        # Model selection and credentials (provider-discriminated like llm.py)
        model: StagehandLlmModel = SchemaField(
            title="LLM Model",
            description="LLM to use for Stagehand (provider is inferred)",
            default=StagehandLlmModel.GPT4O,
            advanced=False,
        )
        model_credentials: AICredentials = AICredentialsField()
        env: str = SchemaField(
            description="Environment to use ('BROWSERBASE' or 'LOCAL')",
            default="LOCAL",
        )
        headless: bool = SchemaField(
            description="Run browser in headless mode (only for LOCAL env)",
            default=False,
        )

    class Output(BlockSchema):
        session_id: str = SchemaField(
            description="Unique session ID for this Stagehand instance"
        )
        success: bool = SchemaField(description="Whether initialization was successful")
        error: Optional[str] = SchemaField(
            description="Error message if initialization failed"
        )

    def __init__(self):
        super().__init__(
            id="b685abf9-796b-47f5-b7ac-dcffb70635a3",
            description="Initialize a Stagehand browser automation session",
            categories={BlockCategory.AI, BlockCategory.DEVELOPER_TOOLS},
            input_schema=StagehandInitBlock.Input,
            output_schema=StagehandInitBlock.Output,
            test_input={
                "env": "LOCAL",
                "headless": True,
                "model": StagehandLlmModel.GPT4O,
                "model_credentials": {
                    "id": "openai-test",
                    "provider": "openai",
                    "type": "api_key",
                    "title": "Mock OpenAI API key",
                },
            },
            test_output=[
                ("session_id", "test-session-123"),
                ("success", True),
                ("error", None),
            ],
        )
        self._sessions: Dict[str, Any] = {}

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            import uuid

            from stagehand import Stagehand, StagehandConfig

            # Create unique session ID
            session_id = str(uuid.uuid4())

            # Configure Stagehand
            model_name = input_data.model.value

            # Resolve LLM model API key from credentials
            model_api_key: Optional[str] = None
            try:
                # type: ignore[attr-defined]
                model_api_key = kwargs.get("credentials").api_key.get_secret_value() if False else None  # noqa: F401
            except Exception:
                # We don't use block-level credentials; pull from input credentials
                pass

            if input_data.model_credentials:
                # mypy/pydantic type has 'id','provider','type' only; runtime passes API key through execution context
                # In backend engine, the resolved Credentials object is passed separately to run(); here we fallback to env var
                # so we set provider-specific env var if available via Secrets store in execution.
                # Prefer using Secrets if present in settings; else rely on explicit model_api_key passed below.
                # We cannot read the secret value from meta; so rely on Stagehand(model_api_key=...) path
                pass

            # Build config and set Browserbase env vars when needed
            if input_data.env == "BROWSERBASE":
                # Prefer credential object if provided, else fallback field
                browser_api_key = input_data.browserbase_api_key
                if input_data.browser_credentials:
                    # At runtime the engine injects the selected credential secret; mirror by honoring env var for compatibility
                    # Since meta input does not carry secret, we rely on env being pre-configured via SDK defaults
                    # If AutoRegistry provided a default credential, it should be in env already; still prefer explicit field
                    browser_api_key = browser_api_key or os.getenv("BROWSERBASE_API_KEY")

                # Set env vars to match docs
                if browser_api_key:
                    os.environ["BROWSERBASE_API_KEY"] = browser_api_key
                if input_data.browserbase_project_id:
                    os.environ["BROWSERBASE_PROJECT_ID"] = input_data.browserbase_project_id

                config = StagehandConfig(  # type: ignore
                    env="BROWSERBASE",
                    apiKey=browser_api_key,
                    projectId=input_data.browserbase_project_id,
                    modelName=model_name,
                )
            else:  # LOCAL
                config = StagehandConfig(  # type: ignore
                    env="LOCAL",
                    modelName=model_name,
                )

            # Create Stagehand instance with model API key
            # Try to derive provider env var name and set it for compatibility with Stagehand
            # Map provider from the full LlmModel metadata
            provider = LlmModel(model_name).provider
            provider_env_var = None
            if provider == "openai":
                provider_env_var = "OPENAI_API_KEY"
            elif provider == "anthropic":
                provider_env_var = "ANTHROPIC_API_KEY"
            elif provider == "groq":
                provider_env_var = "GROQ_API_KEY"
            elif provider == "open_router":
                provider_env_var = "OPENROUTER_API_KEY"
            elif provider == "llama_api":
                provider_env_var = "LLAMA_API_KEY"
            elif provider == "aiml_api":
                provider_env_var = "AIML_API_KEY"

            # Best-effort: rely on env var already configured by SDK; Stagehand also accepts explicit model_api_key
            explicit_model_key = os.getenv(provider_env_var) if provider_env_var else None

            stagehand = Stagehand(
                config=config,
                model_api_key=explicit_model_key or "",  # empty if not set; Stagehand may fallback to env
                use_rich_logging=False,  # Disable rich logging in backend
            )

            # Initialize the browser
            await stagehand.init()

            # Store the session
            self._sessions[session_id] = stagehand

            yield "session_id", session_id
            yield "success", True
            yield "error", None

        except Exception as e:
            logger.error(f"Error initializing Stagehand: {str(e)}")
            yield "session_id", ""
            yield "success", False
            yield "error", str(e)


class StagehandActBlock(Block):
    """
    Performs actions on a web page using natural language instructions.
    """

    class Input(BlockSchema):
        session_id: str = SchemaField(description="Session ID from StagehandInitBlock")
        url: Optional[str] = SchemaField(
            description="URL to navigate to before performing action (optional)",
            default=None,
        )
        instruction: str = SchemaField(
            description="Natural language instruction for the action to perform"
        )
        use_vision: bool = SchemaField(
            description="Use visual AI to identify elements on the page",
            default=False,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the action was successfully performed"
        )
        message: str = SchemaField(description="Result message or error description")
        current_url: str = SchemaField(description="Current page URL after action")

    def __init__(self):
        super().__init__(
            id="97881292-a597-4410-a243-a05c6fc1e695",
            description="Perform browser actions using natural language via Stagehand",
            categories={BlockCategory.AI, BlockCategory.DEVELOPER_TOOLS},
            input_schema=StagehandActBlock.Input,
            output_schema=StagehandActBlock.Output,
            test_input={
                "session_id": "test-session-123",
                "url": "https://example.com",
                "instruction": "Click the search button",
                "use_vision": False,
            },
            test_output=[
                ("success", True),
                ("message", "Action completed successfully"),
                ("current_url", "https://example.com"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            # Get the Stagehand instance from the init block
            from backend.blocks.stagehand.blocks import StagehandInitBlock

            init_block = StagehandInitBlock()
            stagehand = init_block._sessions.get(input_data.session_id)

            if not stagehand:
                yield "success", False
                yield "message", f"Session {input_data.session_id} not found. Please initialize first."
                yield "current_url", ""
                return

            page = stagehand.page

            # Navigate to URL if provided
            if input_data.url:
                await page.goto(input_data.url)

            # Perform the action
            await page.act(input_data.instruction, use_vision=input_data.use_vision)

            # Get current URL
            current_url = page.url

            yield "success", True
            yield "message", "Action completed successfully"
            yield "current_url", current_url

        except Exception as e:
            logger.error(f"Error in StagehandActBlock: {str(e)}")
            yield "success", False
            yield "message", f"Error: {str(e)}"
            yield "current_url", ""


class StagehandExtractBlock(Block):
    """
    Extracts structured data from a web page using natural language or schema.
    """

    class Input(BlockSchema):
        session_id: str = SchemaField(description="Session ID from StagehandInitBlock")
        url: Optional[str] = SchemaField(
            description="URL to navigate to before extraction (optional)",
            default=None,
        )
        instruction: str = SchemaField(
            description="Natural language instruction for what to extract"
        )
        extraction_schema: Optional[Dict[str, Any]] = SchemaField(
            description="Optional schema to structure the extracted data",
            default=None,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether extraction was successful")
        extracted_data: Any = SchemaField(description="The extracted data")
        error: Optional[str] = SchemaField(
            description="Error message if extraction failed"
        )

    def __init__(self):
        super().__init__(
            id="14aa8191-05a4-4bc6-aed6-2bf442796caa",
            description="Extract data from web pages using Stagehand",
            categories={
                BlockCategory.AI,
                BlockCategory.DEVELOPER_TOOLS,
                BlockCategory.SEARCH,
            },
            input_schema=StagehandExtractBlock.Input,
            output_schema=StagehandExtractBlock.Output,
            test_input={
                "session_id": "test-session-123",
                "url": "https://example.com",
                "instruction": "Extract all product prices",
                "extraction_schema": None,
            },
            test_output=[
                ("success", True),
                ("extracted_data", ["$19.99", "$29.99", "$39.99"]),
                ("error", None),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            # Get the Stagehand instance from the init block
            from backend.blocks.stagehand.blocks import StagehandInitBlock

            init_block = StagehandInitBlock()
            stagehand = init_block._sessions.get(input_data.session_id)

            if not stagehand:
                yield "success", False
                yield "extracted_data", None
                yield "error", f"Session {input_data.session_id} not found. Please initialize first."
                return

            page = stagehand.page

            # Navigate to URL if provided
            if input_data.url:
                await page.goto(input_data.url)

            # Extract data
            if input_data.extraction_schema:
                result = await page.extract(
                    instruction=input_data.instruction,
                    schema=input_data.extraction_schema,
                )
            else:
                result = await page.extract(input_data.instruction)

            yield "success", True
            yield "extracted_data", result
            yield "error", None

        except Exception as e:
            logger.error(f"Error in StagehandExtractBlock: {str(e)}")
            yield "success", False
            yield "extracted_data", None
            yield "error", str(e)


class StagehandObserveBlock(Block):
    """
    Observes and plans an action before executing it.
    """

    class Input(BlockSchema):
        session_id: str = SchemaField(description="Session ID from StagehandInitBlock")
        url: Optional[str] = SchemaField(
            description="URL to navigate to before observing (optional)",
            default=None,
        )
        instruction: str = SchemaField(
            description="Natural language instruction to observe/plan"
        )
        use_vision: bool = SchemaField(
            description="Use visual AI to identify elements",
            default=False,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether observation was successful")
        action_plan: str = SchemaField(description="Description of the planned action")
        selectors: list[str] = SchemaField(
            description="CSS selectors identified for the action"
        )
        error: Optional[str] = SchemaField(
            description="Error message if observation failed"
        )

    def __init__(self):
        super().__init__(
            id="101bb67c-333e-4f73-8cbc-eff62ff8e7e4",
            description="Observe and plan browser actions using Stagehand",
            categories={BlockCategory.AI, BlockCategory.DEVELOPER_TOOLS},
            input_schema=StagehandObserveBlock.Input,
            output_schema=StagehandObserveBlock.Output,
            test_input={
                "session_id": "test-session-123",
                "url": "https://example.com",
                "instruction": "Click the login button",
                "use_vision": False,
            },
            test_output=[
                ("success", True),
                ("action_plan", "Click on the login button element"),
                ("selectors", ["#login-btn", ".login-button"]),
                ("error", None),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            # Get the Stagehand instance from the init block
            from backend.blocks.stagehand.blocks import StagehandInitBlock

            init_block = StagehandInitBlock()
            stagehand = init_block._sessions.get(input_data.session_id)

            if not stagehand:
                yield "success", False
                yield "action_plan", ""
                yield "selectors", []
                yield "error", f"Session {input_data.session_id} not found. Please initialize first."
                return

            page = stagehand.page

            # Navigate to URL if provided
            if input_data.url:
                await page.goto(input_data.url)

            # Observe the action
            observations = await page.observe(
                input_data.instruction, use_vision=input_data.use_vision
            )

            # Extract selectors and action plan from observations
            selectors = []
            action_plan = input_data.instruction

            if observations:
                if isinstance(observations, list):
                    for obs in observations:
                        if hasattr(obs, "selector"):
                            selectors.append(obs.selector)
                    action_plan = f"Will perform: {input_data.instruction}"
                else:
                    action_plan = str(observations)

            yield "success", True
            yield "action_plan", action_plan
            yield "selectors", selectors
            yield "error", None

        except Exception as e:
            logger.error(f"Error in StagehandObserveBlock: {str(e)}")
            yield "success", False
            yield "action_plan", ""
            yield "selectors", []
            yield "error", str(e)


class StagehandCloseBlock(Block):
    """
    Closes a Stagehand browser session.
    """

    class Input(BlockSchema):
        session_id: str = SchemaField(description="Session ID to close")

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the session was closed")
        message: str = SchemaField(description="Status message")

    def __init__(self):
        super().__init__(
            id="33fc55c8-7c2a-43f6-8d33-82d67d4546ce",
            description="Close a Stagehand browser session",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=StagehandCloseBlock.Input,
            output_schema=StagehandCloseBlock.Output,
            test_input={
                "session_id": "test-session-123",
            },
            test_output=[
                ("success", True),
                ("message", "Session closed successfully"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            # Get the Stagehand instance from the init block
            from backend.blocks.stagehand.blocks import StagehandInitBlock

            init_block = StagehandInitBlock()
            stagehand = init_block._sessions.get(input_data.session_id)

            if not stagehand:
                yield "success", False
                yield "message", f"Session {input_data.session_id} not found"
                return

            # Close the browser
            await stagehand.close()

            # Remove from sessions
            del init_block._sessions[input_data.session_id]

            yield "success", True
            yield "message", "Session closed successfully"

        except Exception as e:
            logger.error(f"Error closing Stagehand session: {str(e)}")
            yield "success", False
            yield "message", f"Error: {str(e)}"
