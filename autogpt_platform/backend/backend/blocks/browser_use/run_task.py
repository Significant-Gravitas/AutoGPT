import asyncio
import logging
import time
from typing import Any

from backend.data.model import NodeExecutionStats
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
from backend.util.exceptions import BlockExecutionError

from ._api import (
    BrowserUseClient,
    BrowserUseRun,
    BrowserUseRunStatus,
    BrowserUseSessionCost,
    build_run_payload,
)
from ._config import browser_use

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 3


class BrowserUseRunTaskBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = browser_use.credentials_field(
            description="The Browser Use integration requires an API Key."
        )
        task: str = SchemaField(
            description="What the browser agent should do, in plain language",
            placeholder="Find the top story on Hacker News and return its title",
        )
        output_schema: dict[str, Any] | None = SchemaField(
            description="Optional JSON Schema for the final answer. When set, "
            "the output pin holds JSON that matches it.",
            default=None,
            advanced=True,
        )
        model: str = SchemaField(
            description="Browser Use agent model, e.g. gpt-6-luna. Leave empty "
            "for the Browser Use default.",
            default="",
            advanced=True,
        )
        proxy_country_code: str = SchemaField(
            description="Two-letter country code for the residential proxy. "
            "Leave empty to run without a proxy.",
            default="us",
            advanced=True,
        )
        profile_id: str = SchemaField(
            description="Browser Use profile ID whose cookies and logins the "
            "browser should load",
            default="",
            advanced=True,
        )
        max_cost_usd: float | None = SchemaField(
            description="Stop the run once it has cost this many USD. Leave "
            "empty for no cap.",
            default=None,
            ge=0,
            advanced=True,
        )
        timeout: int = SchemaField(
            description="Seconds to wait for the run before cancelling it",
            default=600,
            ge=30,
            le=3600,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        result: str = SchemaField(description="The agent's final answer as text")
        output: Any = SchemaField(
            description="Parsed JSON answer matching output_schema (null "
            "without one)"
        )
        run_id: str = SchemaField(description="Browser Use run ID")
        session_id: str = SchemaField(
            description="Browser Use session ID, for follow-up runs and replay"
        )
        cost_usd: float = SchemaField(
            description="Recorded cost in USD: agent LLM usage plus browser "
            "time and proxy traffic"
        )

    def __init__(self):
        super().__init__(
            id="3a874687-35e1-4167-a321-a18d0d6d9f21",
            description="Runs a natural-language task with a Browser Use agent "
            "in a cloud browser and returns the result",
            categories={BlockCategory.AI, BlockCategory.SEARCH},
            effect=BlockEffect.EXTERNAL,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": browser_use.get_test_credentials().model_dump(),
                "task": "Find the top story on Hacker News and return its title",
            },
            test_credentials=browser_use.get_test_credentials(),
            test_output=[
                ("result", "Show HN: AutoGPT"),
                ("output", None),
                ("run_id", "run-1"),
                ("session_id", "session-1"),
                ("cost_usd", lambda x: abs(x - 0.015) < 1e-9),
            ],
            test_mock={
                "_create_run": lambda *args, **kwargs: "run-1",
                "_get_status": lambda *args, **kwargs: BrowserUseRunStatus.COMPLETED,
                "_get_run": lambda *args, **kwargs: BrowserUseRun(
                    id="run-1",
                    status=BrowserUseRunStatus.COMPLETED,
                    session_id="session-1",
                    result="Show HN: AutoGPT",
                    total_cost_usd=0.012,
                ),
                "_get_session_cost": lambda *args, **kwargs: BrowserUseSessionCost(
                    browser_cost_usd=0.002, proxy_cost_usd=0.001
                ),
                "_cancel_run": lambda *args, **kwargs: None,
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        payload = build_run_payload(
            task=input_data.task,
            model=input_data.model,
            output_schema=input_data.output_schema,
            proxy_country_code=input_data.proxy_country_code,
            profile_id=input_data.profile_id,
            max_cost_usd=input_data.max_cost_usd,
        )
        try:
            run_id = await self._create_run(credentials, payload)
        except Exception as e:
            raise self._error(f"Could not start the run: {e}") from e

        run = await self._wait_or_cancel(credentials, run_id, input_data.timeout)
        cost_usd = await self._record_cost(credentials, run)
        if run.status != BrowserUseRunStatus.COMPLETED:
            raise self._error(
                f"Run {run.id} {run.status.value}: {run.error or 'no error message'}"
            )

        yield "result", run.result or ""
        yield "output", run.output
        yield "run_id", run.id
        yield "session_id", run.session_id
        yield "cost_usd", cost_usd

    async def _wait_or_cancel(
        self, credentials: APIKeyCredentials, run_id: str, timeout: int
    ) -> BrowserUseRun:
        """Poll until the run is terminal. On timeout, a poll failure or this
        node being stopped, cancel the run so it stops billing."""
        try:
            return await self._wait_for_run(credentials, run_id, timeout)
        except asyncio.CancelledError:
            await self._cancel_and_record_cost(credentials, run_id)
            raise
        except Exception as e:
            await self._cancel_and_record_cost(credentials, run_id)
            raise self._error(f"Stopped waiting for run {run_id}: {e}") from e

    async def _wait_for_run(
        self, credentials: APIKeyCredentials, run_id: str, timeout: int
    ) -> BrowserUseRun:
        deadline = time.monotonic() + timeout
        while not (await self._get_status(credentials, run_id)).is_terminal:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"not finished after {timeout}s")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        return await self._get_run(credentials, run_id)

    async def _cancel_and_record_cost(
        self, credentials: APIKeyCredentials, run_id: str
    ) -> None:
        """Best effort: cancel the run, then bill whatever it already spent."""
        try:
            await self._cancel_run(credentials, run_id)
            await self._record_cost(
                credentials, await self._get_run(credentials, run_id)
            )
        except Exception as e:
            logger.warning(f"Could not cancel Browser Use run {run_id}: {e}")

    async def _record_cost(
        self, credentials: APIKeyCredentials, run: BrowserUseRun
    ) -> float:
        """Bill the run's LLM spend plus its session's browser and proxy use.

        Each block run starts a fresh session, so the session's browser and
        proxy costs belong to this run. If they cannot be read, bill the run
        cost alone rather than fail a finished run.
        """
        cost = run.total_cost_usd
        try:
            session = await self._get_session_cost(credentials, run.session_id)
            cost += session.browser_cost_usd + session.proxy_cost_usd
        except Exception as e:
            logger.warning(f"Could not read Browser Use session cost: {e}")
        self.merge_stats(
            NodeExecutionStats(provider_cost=cost, provider_cost_type="cost_usd")
        )
        return cost

    def _error(self, message: str) -> BlockExecutionError:
        return BlockExecutionError(
            message=message, block_name=self.name, block_id=self.id
        )

    # Thin wrappers over the client so tests and test_mock can stub the network.
    async def _create_run(
        self, credentials: APIKeyCredentials, payload: dict[str, Any]
    ) -> str:
        return await BrowserUseClient(credentials).create_run(payload)

    async def _get_status(
        self, credentials: APIKeyCredentials, run_id: str
    ) -> BrowserUseRunStatus:
        return await BrowserUseClient(credentials).get_status(run_id)

    async def _get_run(
        self, credentials: APIKeyCredentials, run_id: str
    ) -> BrowserUseRun:
        return await BrowserUseClient(credentials).get_run(run_id)

    async def _get_session_cost(
        self, credentials: APIKeyCredentials, session_id: str
    ) -> BrowserUseSessionCost:
        return await BrowserUseClient(credentials).get_session_cost(session_id)

    async def _cancel_run(self, credentials: APIKeyCredentials, run_id: str) -> None:
        await BrowserUseClient(credentials).cancel_run(run_id)
