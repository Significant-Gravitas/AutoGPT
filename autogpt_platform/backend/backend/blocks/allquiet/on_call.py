"""Block that reads who is currently on call."""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import AllQuietClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, allquiet, region_field
from ._testdata import TEST_SHIFT
from ._types import AllQuietRegion, AllQuietUser, OnCallShift


class AllQuietGetOnCallBlock(Block):
    """Look up who is on call, now or at a chosen time."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = allquiet.credentials_field(
            description="The All Quiet integration requires an API Key."
        )
        team_ids: list[str] = SchemaField(
            description="Limit to these teams. Leave empty for every team.",
            default_factory=list,
            advanced=False,
        )
        user_ids: list[str] = SchemaField(
            description="Limit to these users. Leave empty for every user.",
            default_factory=list,
            advanced=True,
        )
        timestamp: str = SchemaField(
            description=(
                "ISO-8601 timestamp to evaluate the rotation at. "
                "Leave empty for right now."
            ),
            default="",
            advanced=True,
        )
        region: AllQuietRegion = region_field()

    class Output(BlockSchemaOutput):
        shifts: list[OnCallShift] = SchemaField(
            description="Every matching on-call assignment"
        )
        shift: OnCallShift = SchemaField(
            description="Each on-call assignment, emitted one at a time"
        )
        users: list[AllQuietUser] = SchemaField(
            description="The on-call users, deduplicated across teams"
        )
        user_ids: list[str] = SchemaField(description="IDs of the on-call users")
        emails: list[str] = SchemaField(
            description="Email addresses of the on-call users"
        )
        users_without_email: list[AllQuietUser] = SchemaField(
            description=(
                "On-call users carrying no email address. These are counted in "
                "users/has_coverage but absent from emails, so a graph that "
                "notifies by email alone would silently skip them"
            )
        )
        has_coverage: bool = SchemaField(
            description=(
                "False when nobody is on call for the requested teams/time, so a "
                "graph can branch to a fallback instead of silently paging no one"
            )
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="8e5d3b27-1a94-4c6f-b7e0-5f9c2d8a4b16",
            description="Looks up who is on call in All Quiet, now or at a given time",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.ISSUE_TRACKING},
            input_schema=AllQuietGetOnCallBlock.Input,
            output_schema=AllQuietGetOnCallBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("shifts", [TEST_SHIFT]),
                ("shift", TEST_SHIFT),
                ("users", [TEST_SHIFT.user]),
                ("user_ids", [TEST_SHIFT.user.id if TEST_SHIFT.user else ""]),
                ("emails", ["ada@example.com"]),
                ("users_without_email", []),
                ("has_coverage", True),
            ],
            test_mock={"get_on_call": lambda *args, **kwargs: [TEST_SHIFT]},
        )

    @staticmethod
    async def get_on_call(
        credentials: APIKeyCredentials, region: AllQuietRegion, input_data: Input
    ) -> list[OnCallShift]:
        client = AllQuietClient(credentials, region)
        return await client.get_on_call(
            team_ids=input_data.team_ids,
            user_ids=input_data.user_ids,
            timestamp=input_data.timestamp,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        shifts = await self.get_on_call(credentials, input_data.region, input_data)
        yield "shifts", shifts
        for shift in shifts:
            yield "shift", shift

        # A user on call for several teams appears once per team; collapse them
        # so downstream notification steps do not page the same person twice.
        users_by_id: dict[str, AllQuietUser] = {}
        for shift in shifts:
            if shift.user and shift.user.id:
                users_by_id.setdefault(shift.user.id, shift.user)

        users = list(users_by_id.values())
        yield "users", users
        yield "user_ids", list(users_by_id)
        yield "emails", [user.email for user in users if user.email]
        # Surfaced separately so an email-only notification step can tell
        # "nobody on call" apart from "on call but unreachable by email".
        yield "users_without_email", [user for user in users if not user.email]
        yield "has_coverage", bool(users)
