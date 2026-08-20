import logging

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

from ._api import GITHUB_API_URL, get_api
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentials,
    GithubCredentialsField,
    GithubCredentialsInput,
)

logger = logging.getLogger(__name__)

TEST_USER_PAYLOAD = {
    "login": "octocat",
    "name": "The Octocat",
    "html_url": "https://github.com/octocat",
    "avatar_url": "https://avatars.githubusercontent.com/u/583231",
}


class GithubGetUserInfoBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("read:user")
        username: str = SchemaField(
            description="Username of the GitHub user to look up. "
            "Leave empty to get the authenticated user (yourself).",
            placeholder="octocat",
            default="",
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        username: str = SchemaField(description="Login (username) of the user")
        name: str = SchemaField(description="Display name of the user")
        profile_url: str = SchemaField(description="URL of the user's GitHub profile")
        avatar_url: str = SchemaField(description="URL of the user's avatar image")
        user: dict = SchemaField(description="The full user object from the API")
        error: str = SchemaField(
            description="Error message if fetching the user info failed"
        )

    def __init__(self):
        super().__init__(
            id="046050c6-96cd-44fa-b73f-af5d790a43b6",
            description="This block fetches information about a GitHub user, "
            "or about the authenticated user (yourself) if no username is given.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubGetUserInfoBlock.Input,
            output_schema=GithubGetUserInfoBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("username", "octocat"),
                ("name", "The Octocat"),
                ("profile_url", "https://github.com/octocat"),
                ("avatar_url", "https://avatars.githubusercontent.com/u/583231"),
                ("user", TEST_USER_PAYLOAD),
            ],
            test_mock={"get_user": lambda *args, **kwargs: TEST_USER_PAYLOAD},
        )

    @staticmethod
    async def get_user(credentials: GithubCredentials, username: str) -> dict:
        api = get_api(credentials, convert_urls=False)
        url = (
            f"{GITHUB_API_URL}/users/{username}"
            if username
            else f"{GITHUB_API_URL}/user"
        )
        response = await api.get(url)
        return response.json()

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        user = await self.get_user(credentials, input_data.username)
        yield "username", user["login"]
        yield "name", user.get("name") or ""
        yield "profile_url", user["html_url"]
        yield "avatar_url", user["avatar_url"]
        yield "user", user
