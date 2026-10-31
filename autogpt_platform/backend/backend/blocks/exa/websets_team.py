"""
Exa Websets Team Block

Reports the team's Websets concurrency: how many requests are running and
queued against the plan's limits. Check it before launching a large webset to
see whether it will start now or wait in the queue.
"""

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .websets_api import ExaWebsetsBlockBase
from .websets_fixtures import sample_team


class ExaGetWebsetsTeamInfoBlock(ExaWebsetsBlockBase):
    """Get the team's current Websets concurrency usage and limits."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )

    class Output(BlockSchemaOutput):
        team_id: str = SchemaField(description="The team's unique identifier")
        name: str = SchemaField(description="The team's name")
        active: int = SchemaField(
            description="Websets requests being processed right now"
        )
        queued: int = SchemaField(description="Websets requests waiting to start")
        max_concurrent: int | None = SchemaField(
            description="Most requests the team can run at once (empty means unlimited)"
        )
        max_queued: int | None = SchemaField(
            description="Most requests the team can queue (empty means unlimited)"
        )
        available_slots: int | None = SchemaField(
            description="Requests that can start immediately: max concurrent minus active (empty means unlimited)"
        )
        has_capacity: bool = SchemaField(
            description="True if a new request would start now instead of queueing"
        )

    def __init__(self):
        super().__init__(
            id="0e1db79a-a419-4cc9-ab52-0a3133043198",
            description="Get your team's Websets concurrency usage and limits, to check capacity before launching large websets",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetWebsetsTeamInfoBlock.Input,
            output_schema=ExaGetWebsetsTeamInfoBlock.Output,
            effect=BlockEffect.READ,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[
                ("team_id", "team_test123"),
                ("name", "Test team"),
                ("active", 2),
                ("queued", 0),
                ("max_concurrent", 10),
                ("max_queued", None),
                ("available_slots", 8),
                ("has_capacity", True),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_api": lambda *args, **kwargs: sample_team()},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        team = await self._api(credentials, "GET", "/teams/me")
        concurrency = team.get("concurrency") or {}
        limits = team.get("limits") or {}
        active = int(concurrency.get("active") or 0)
        max_concurrent = limits.get("maxConcurrent")
        available = None if max_concurrent is None else max(0, max_concurrent - active)

        yield "team_id", team.get("id", "")
        yield "name", team.get("name", "")
        yield "active", active
        yield "queued", int(concurrency.get("queued") or 0)
        yield "max_concurrent", max_concurrent
        yield "max_queued", limits.get("maxQueued")
        yield "available_slots", available
        yield "has_capacity", available is None or available > 0
