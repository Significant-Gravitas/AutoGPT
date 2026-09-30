"""Block that starts a Capy agent thread."""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import CapyClient, with_capy_balance_fallback
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, capy_credentials_field
from ._models import (
    ModelRoute,
    billed_via,
    capy_balance_fallback_field,
    model_route_field,
    resolve_model_id,
)
from ._testdata import TEST_PROJECT, TEST_THREAD
from ._types import MachineSize, ReasoningEffort, Thread


class CapyCreateThreadBlock(Block):
    """Start a Capy agent thread: a cloud coding agent working on a project's repos."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        project_id: str = SchemaField(
            description="The Capy project to run in (see Capy List Projects)"
        )
        message: str = SchemaField(
            description=(
                "The task for the agent, written as you would brief an engineer: "
                "the goal, where to look, what done looks like, and whether to "
                "open a pull request"
            ),
        )
        title: str = SchemaField(
            description="Thread title. Leave empty to let Capy name it.",
            default="",
        )
        model_id: str = SchemaField(
            description=(
                "Capy model ID from docs.capy.ai/models-and-pricing, e.g. "
                "openai/gpt-6-astra, or a bare name like gpt-6-astra to combine "
                "with model_route. Leave empty for the project's default model."
            ),
            default="",
            advanced=True,
        )
        model_route: ModelRoute = model_route_field()
        fall_back_to_capy_balance: bool = capy_balance_fallback_field()
        reasoning: ReasoningEffort = SchemaField(
            description="Reasoning effort for the chosen model. Needs model_id.",
            default=ReasoningEffort.DEFAULT,
            advanced=True,
        )
        machine_size: MachineSize = SchemaField(
            description="Machine size for the agent's VM. Empty uses Capy's default.",
            default=MachineSize.DEFAULT,
            advanced=True,
        )
        request_id: str = SchemaField(
            description=(
                "Idempotency key. Re-sending the same request_id returns the "
                "thread it already created instead of starting a second run. "
                "Leave empty to generate one."
            ),
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        thread: Thread = SchemaField(description="The created thread")
        thread_id: str = SchemaField(description="ID of the created thread")
        thread_url: str = SchemaField(
            description=(
                "The thread in the Capy app, where the agent's plan, commands "
                "and diff show live. Share it when you hand the task off."
            )
        )
        status: str = SchemaField(description="The thread's status right after start")
        model_id: str = SchemaField(
            description=(
                "The model ID the thread was started with; differs from the "
                "input when the route rewrote it or the balance fallback ran. "
                "Empty means the project's default model."
            )
        )
        billed_via: str = SchemaField(
            description="Who pays for that model: the Capy balance or a linked provider"
        )

    def __init__(self):
        super().__init__(
            id="f8ffe722-ceba-4fb3-81df-a3fd465ff29c",
            description=(
                "Hands a task to Capy, an AI software engineer: a background "
                "coding agent that works on its own cloud machine against a "
                "GitHub repo in one of your Capy projects. Use it to write code, "
                "fix a bug, build a feature or open a pull request. The model "
                "can run on your Capy balance or on a provider linked in Capy "
                "(Codex, Copilot, SuperGrok, Azure). Returns immediately with "
                "the thread ID and a link where the work shows live; use Capy "
                "Wait For Thread to wait for the result. Once the agent opens "
                "a pull request it follows it by itself, fixing failing CI and "
                "answering review comments."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.AGENT},
            input_schema=CapyCreateThreadBlock.Input,
            output_schema=CapyCreateThreadBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": TEST_PROJECT.id,
                "message": "Upgrade the CI pipeline to Node 24 and open a PR.",
                "model_id": "grok-4.5",
                "model_route": ModelRoute.SUPERGROK.value,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("thread", TEST_THREAD),
                ("thread_id", TEST_THREAD.id),
                ("thread_url", TEST_THREAD.url),
                ("status", "working"),
                ("model_id", "supergrok/grok-4.5"),
                ("billed_via", "SuperGrok subscription"),
            ],
            test_mock={
                "create_thread": lambda *args, **kwargs: (
                    TEST_THREAD,
                    "supergrok/grok-4.5",
                )
            },
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    async def create_thread(
        credentials: APIKeyCredentials, input_data: "CapyCreateThreadBlock.Input"
    ) -> tuple[Thread, str]:
        client = CapyClient(credentials)
        requested = resolve_model_id(input_data.model_id, input_data.model_route)
        request_id = input_data.request_id

        async def create(model_id: str) -> Thread:
            # A fallback run is a different request from the rejected one, so
            # it gets its own (still deterministic) idempotency key.
            key = request_id
            if key and model_id != requested:
                key = f"{key}-capy-balance"
            return await client.create_thread(
                project_id=input_data.project_id,
                message=input_data.message,
                title=input_data.title,
                model_id=model_id,
                reasoning=input_data.reasoning.value,
                machine_size=input_data.machine_size.value,
                request_id=key,
            )

        return await with_capy_balance_fallback(
            create, requested, input_data.fall_back_to_capy_balance
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        thread, model_id = await self.create_thread(credentials, input_data)
        yield "thread", thread
        yield "thread_id", thread.id
        yield "thread_url", thread.url
        yield "status", thread.status
        yield "model_id", model_id
        yield "billed_via", billed_via(model_id)
