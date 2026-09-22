"""Block that lists All Quiet teams, to resolve the team IDs other blocks take."""

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
from ._testdata import TEST_TEAM
from ._types import AllQuietRegion, Team


class AllQuietListTeamsBlock(Block):
    """List teams so other blocks can be pointed at them by ID."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = allquiet.credentials_field(
            description="The All Quiet integration requires an API Key."
        )
        display_name: str = SchemaField(
            description="Filter to teams whose name matches this text.",
            default="",
            advanced=False,
        )
        limit: int = SchemaField(
            description="Maximum number of teams to return.",
            default=50,
            ge=1,
            le=100,
            advanced=False,
        )
        offset: int = SchemaField(
            description="Number of teams to skip, for paging.",
            default=0,
            ge=0,
            advanced=True,
        )
        region: AllQuietRegion = region_field()

    class Output(BlockSchemaOutput):
        teams: list[Team] = SchemaField(description="All matching teams")
        team: Team = SchemaField(
            description="Each matching team, emitted one at a time"
        )
        team_ids: list[str] = SchemaField(
            description="IDs of the matching teams, for use as team_ids elsewhere"
        )
        has_more: bool = SchemaField(
            description="Whether more teams are available beyond this page"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="2a6f8d13-4b70-4e29-8c5d-9e1b3f7a0c48",
            description="Lists All Quiet teams and their IDs",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.ISSUE_TRACKING},
            input_schema=AllQuietListTeamsBlock.Input,
            output_schema=AllQuietListTeamsBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("teams", [TEST_TEAM]),
                ("team", TEST_TEAM),
                ("team_ids", [TEST_TEAM.id]),
                ("has_more", False),
            ],
            test_mock={"list_teams": lambda *args, **kwargs: ([TEST_TEAM], False)},
        )

    @staticmethod
    async def list_teams(
        credentials: APIKeyCredentials, region: AllQuietRegion, input_data: Input
    ) -> tuple[list[Team], bool]:
        client = AllQuietClient(credentials, region)
        return await client.list_teams(
            display_name=input_data.display_name,
            limit=input_data.limit,
            offset=input_data.offset,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        teams, has_more = await self.list_teams(
            credentials, input_data.region, input_data
        )
        yield "teams", teams
        for team in teams:
            yield "team", team
        yield "team_ids", [team.id for team in teams]
        yield "has_more", has_more
