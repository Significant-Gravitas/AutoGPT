"""
Exa Agent blocks that manage runs: list, cancel, stop, delete, list events.
"""

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

from . import agent_api
from ._config import exa
from ._platform_key import require_own_exa_key
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .agent import RUN_ID_PLACEHOLDER, AgentRunOutputs, run_outputs
from .agent_billing import bill_finished_run
from .agent_fixtures import (
    SAMPLE_CANCELLED_RUN,
    SAMPLE_DELETED_RUN,
    SAMPLE_EVENT_LIST,
    SAMPLE_RUN_ID,
    SAMPLE_RUN_LIST,
    SAMPLE_STOPPED_RUN,
)
from .agent_models import AgentRun, AgentRunEvent, parse_event, parse_page, parse_run


def _credentials_field() -> CredentialsMetaInput:
    return exa.credentials_field(description="The Exa integration requires an API Key.")


def _run_id_field(description: str) -> str:
    return SchemaField(description=description, placeholder=RUN_ID_PLACEHOLDER)


def _limit_field(noun: str) -> int:
    return SchemaField(
        default=20, description=f"How many {noun} to return (1-100)", ge=1, le=100
    )


def _cursor_field() -> str | None:
    return SchemaField(
        default=None,
        description="Cursor from a previous call's next_cursor, to fetch the next page",
    )


class ExaListAgentRunsBlock(Block):
    """List your team's Exa Agent runs, newest first."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = _credentials_field()
        limit: int = _limit_field("runs")
        cursor: str | None = _cursor_field()

    class Output(BlockSchemaOutput):
        runs: list[AgentRun] = SchemaField(description="Runs, newest first")
        run: AgentRun = SchemaField(title="Run", description="Each run, one at a time")
        has_more: bool = SchemaField(description="Whether more runs are available")
        next_cursor: str | None = SchemaField(
            description="Pass as cursor to get the next page"
        )

    def __init__(self):
        sample_runs = [parse_run(r) for r in SAMPLE_RUN_LIST["data"]]
        super().__init__(
            id="77cfb6ba-a083-4c9d-8e18-a28d52ddfc09",
            description="List your Exa Agent runs, newest first, with paging",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListAgentRunsBlock.Input,
            output_schema=ExaListAgentRunsBlock.Output,
            effect=BlockEffect.READ,
            display_name="List Exa Agent Runs",
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "limit": 1},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("runs", sample_runs),
                ("run", sample_runs[0]),
                ("has_more", True),
                ("next_cursor", SAMPLE_RUN_LIST["nextCursor"]),
            ],
            test_mock={"list_runs": lambda *args, **kwargs: SAMPLE_RUN_LIST},
        )

    @staticmethod
    async def list_runs(
        credentials: APIKeyCredentials, limit: int, cursor: str | None
    ) -> dict:
        return await agent_api.list_agent_runs(credentials, limit, cursor)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        require_own_exa_key(credentials, "Listing Exa Agent runs")
        data = await self.list_runs(credentials, input_data.limit, input_data.cursor)
        runs = [parse_run(r) for r in data.get("data") or []]
        page = parse_page(data)
        yield "runs", runs
        for run in runs:
            yield "run", run
        yield "has_more", page.has_more
        yield "next_cursor", page.next_cursor


class ExaCancelAgentRunBlock(Block):
    """Cancel a queued or running Exa Agent run, discarding its results."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = _credentials_field()
        run_id: str = _run_id_field("ID of the Exa Agent run to cancel")

    class Output(AgentRunOutputs):
        pass

    def __init__(self):
        super().__init__(
            id="b5b99203-5b43-48be-a77b-3906fcfd0ef8",
            description="Cancel an Exa Agent run immediately. No results are kept; "
            "usage so far is still billed. To keep partial results of an ultra "
            "run, use 'Stop Exa Agent Run' instead.",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCancelAgentRunBlock.Input,
            output_schema=ExaCancelAgentRunBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
            display_name="Cancel Exa Agent Run",
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID},
            test_credentials=TEST_CREDENTIALS,
            test_output=list(run_outputs(parse_run(SAMPLE_CANCELLED_RUN))),
            test_mock={"cancel_run": lambda *args, **kwargs: SAMPLE_CANCELLED_RUN},
        )

    @staticmethod
    async def cancel_run(credentials: APIKeyCredentials, run_id: str) -> dict:
        return await agent_api.cancel_agent_run(credentials, run_id)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        data = await self.cancel_run(credentials, input_data.run_id)
        run = parse_run(data)
        await bill_finished_run(self, run, user_id)
        for name, value in run_outputs(run):
            yield name, value


class ExaStopAgentRunBlock(Block):
    """Finish an Exa Agent run early and keep what it has found."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = _credentials_field()
        run_id: str = _run_id_field("ID of the running Exa Agent run to stop")

    class Output(AgentRunOutputs):
        pass

    def __init__(self):
        super().__init__(
            id="8e96eb38-32b8-47f6-81c3-2f89aa488b14",
            description="Stop an Exa Agent run early and return the results "
            "gathered so far (stop reason 'stopped'). Exa currently supports this "
            "for ultra-effort runs only.",
            categories={BlockCategory.SEARCH},
            input_schema=ExaStopAgentRunBlock.Input,
            output_schema=ExaStopAgentRunBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
            display_name="Stop Exa Agent Run",
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID},
            test_credentials=TEST_CREDENTIALS,
            test_output=list(run_outputs(parse_run(SAMPLE_STOPPED_RUN))),
            test_mock={"stop_run": lambda *args, **kwargs: SAMPLE_STOPPED_RUN},
        )

    @staticmethod
    async def stop_run(credentials: APIKeyCredentials, run_id: str) -> dict:
        return await agent_api.stop_agent_run(credentials, run_id)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        data = await self.stop_run(credentials, input_data.run_id)
        run = parse_run(data)
        await bill_finished_run(self, run, user_id)
        for name, value in run_outputs(run):
            yield name, value


class ExaDeleteAgentRunBlock(Block):
    """Permanently delete a stored Exa Agent run."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = _credentials_field()
        run_id: str = _run_id_field("ID of the Exa Agent run to delete")

    class Output(BlockSchemaOutput):
        run_id: str = SchemaField(description="ID of the deleted run")
        deleted: bool = SchemaField(description="True when Exa deleted the run")

    def __init__(self):
        super().__init__(
            id="de99219e-14cb-420b-b3d8-978c8d4ccd83",
            description="Permanently delete a stored Exa Agent run and its output. "
            "It can no longer be fetched or used as a previous run.",
            categories={BlockCategory.SEARCH},
            input_schema=ExaDeleteAgentRunBlock.Input,
            output_schema=ExaDeleteAgentRunBlock.Output,
            is_irreversible_action=True,
            effect=BlockEffect.EXTERNAL,
            display_name="Delete Exa Agent Run",
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("run_id", SAMPLE_RUN_ID), ("deleted", True)],
            test_mock={"delete_run": lambda *args, **kwargs: SAMPLE_DELETED_RUN},
        )

    @staticmethod
    async def delete_run(credentials: APIKeyCredentials, run_id: str) -> dict:
        return await agent_api.delete_agent_run(credentials, run_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.delete_run(credentials, input_data.run_id)
        yield "run_id", data.get("id", input_data.run_id)
        yield "deleted", bool(data.get("deleted", False))


class ExaListAgentRunEventsBlock(Block):
    """List the stored lifecycle events of an Exa Agent run."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = _credentials_field()
        run_id: str = _run_id_field("ID of the Exa Agent run")
        limit: int = _limit_field("events")
        cursor: str | None = _cursor_field()

    class Output(BlockSchemaOutput):
        events: list[AgentRunEvent] = SchemaField(
            description="Events in the order Exa stored them"
        )
        event: AgentRunEvent = SchemaField(
            title="Event", description="Each event, one at a time"
        )
        has_more: bool = SchemaField(description="Whether more events are available")
        next_cursor: str | None = SchemaField(
            description="Pass as cursor to get the next page"
        )

    def __init__(self):
        sample_events = [parse_event(e) for e in SAMPLE_EVENT_LIST["data"]]
        super().__init__(
            id="0d8cf328-ff84-4bc6-b4ae-fa6b9a728d5d",
            description="List an Exa Agent run's stored events (created, started, "
            "completed, failed, cancelled, plus progress). Not available for "
            "Zero Data Retention teams.",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListAgentRunEventsBlock.Input,
            output_schema=ExaListAgentRunEventsBlock.Output,
            effect=BlockEffect.READ,
            display_name="List Exa Agent Run Events",
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("events", sample_events),
                ("event", sample_events[0]),
                ("event", sample_events[1]),
                ("has_more", False),
                ("next_cursor", None),
            ],
            test_mock={"list_events": lambda *args, **kwargs: SAMPLE_EVENT_LIST},
        )

    @staticmethod
    async def list_events(
        credentials: APIKeyCredentials, run_id: str, limit: int, cursor: str | None
    ) -> dict:
        return await agent_api.list_agent_run_events(credentials, run_id, limit, cursor)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.list_events(
            credentials,
            input_data.run_id,
            input_data.limit,
            input_data.cursor,
        )
        events = [parse_event(e) for e in data.get("data") or []]
        page = parse_page(data)
        yield "events", events
        for event in events:
            yield "event", event
        yield "has_more", page.has_more
        yield "next_cursor", page.next_cursor
