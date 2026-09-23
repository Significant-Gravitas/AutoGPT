from datetime import datetime, timezone
from typing import Literal, Optional

from typing_extensions import TypedDict

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.request import Requests

from ._api import get_api, get_paginated
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentials,
    GithubCredentialsField,
    GithubCredentialsInput,
)
from ._utils import github_repo_path


class GithubListTagsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        limit: int = SchemaField(
            description="Maximum number of tags to fetch",
            default=100,
            ge=1,
            le=1000,
        )

    class Output(BlockSchemaOutput):
        class TagItem(TypedDict):
            name: str
            url: str

        tag: TagItem = SchemaField(
            title="Tag", description="Tags with their name and file tree browser URL"
        )
        tags: list[TagItem] = SchemaField(
            description="List of tags with their name and file tree browser URL"
        )

    def __init__(self):
        super().__init__(
            id="358924e7-9a11-4d1a-a0f2-13c67fe59e2e",
            description="This block lists all tags for a specified GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListTagsBlock.Input,
            output_schema=GithubListTagsBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "tags",
                    [
                        {
                            "name": "v1.0.0",
                            "url": "https://github.com/owner/repo/tree/v1.0.0",
                        }
                    ],
                ),
                (
                    "tag",
                    {
                        "name": "v1.0.0",
                        "url": "https://github.com/owner/repo/tree/v1.0.0",
                    },
                ),
            ],
            test_mock={
                "list_tags": lambda *args, **kwargs: [
                    {
                        "name": "v1.0.0",
                        "url": "https://github.com/owner/repo/tree/v1.0.0",
                    }
                ]
            },
        )

    @staticmethod
    async def list_tags(
        credentials: GithubCredentials, repo_url: str, limit: int
    ) -> list[Output.TagItem]:
        api = get_api(credentials)
        tags_url = repo_url + "/tags"
        data = await get_paginated(api, tags_url, limit=limit)
        repo_path = github_repo_path(repo_url)
        tags: list[GithubListTagsBlock.Output.TagItem] = [
            {
                "name": tag["name"],
                "url": f"https://github.com/{repo_path}/tree/{tag['name']}",
            }
            for tag in data
        ]
        return tags

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        tags = await self.list_tags(
            credentials,
            input_data.repo_url,
            input_data.limit,
        )
        yield "tags", tags
        for tag in tags:
            yield "tag", tag


class GithubListDiscussionsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        num_discussions: int = SchemaField(
            title="Limit",
            description="Maximum number of discussions to fetch",
            default=5,
        )
        category: str = SchemaField(
            description="Only include discussions in the category with this name",
            default="",
        )
        answered: Literal["all", "answered", "unanswered"] = SchemaField(
            description="Only include discussions that are (un)answered",
            default="all",
            advanced=True,
        )
        state: Literal["all", "open", "closed"] = SchemaField(
            description="Only include discussions in this state",
            default="all",
            advanced=True,
        )
        order_by: Literal["created", "updated"] = SchemaField(
            description="What to sort the discussions by",
            default="updated",
            advanced=True,
        )
        direction: Literal["asc", "desc"] = SchemaField(
            description="Sort direction",
            default="desc",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        class DiscussionItem(TypedDict):
            title: str
            url: str

        discussion: DiscussionItem = SchemaField(
            title="Discussion", description="Discussions with their title and URL"
        )
        discussions: list[DiscussionItem] = SchemaField(
            description="List of discussions with their title and URL"
        )
        error: str = SchemaField(
            description="Error message if listing discussions failed"
        )

    def __init__(self):
        super().__init__(
            id="3ef1a419-3d76-4e07-b761-de9dad4d51d7",
            description="This block lists recent discussions for a specified GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListDiscussionsBlock.Input,
            output_schema=GithubListDiscussionsBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "num_discussions": 3,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "discussions",
                    [
                        {
                            "title": "Discussion 1",
                            "url": "https://github.com/owner/repo/discussions/1",
                        }
                    ],
                ),
                (
                    "discussion",
                    {
                        "title": "Discussion 1",
                        "url": "https://github.com/owner/repo/discussions/1",
                    },
                ),
            ],
            test_mock={
                "list_discussions": lambda *args, **kwargs: [
                    {
                        "title": "Discussion 1",
                        "url": "https://github.com/owner/repo/discussions/1",
                    }
                ]
            },
        )

    @staticmethod
    async def list_discussions(
        credentials: GithubCredentials, input_data: Input
    ) -> list[Output.DiscussionItem]:
        api = get_api(credentials)
        # GitHub GraphQL API endpoint is different; we'll use api.post with custom URL
        repo_path = github_repo_path(input_data.repo_url)
        owner, repo = repo_path.split("/")
        query = """
        query(
            $owner: String!, $repo: String!, $num: Int!, $after: String,
            $categoryId: ID, $answered: Boolean,
            $states: [DiscussionState!], $orderBy: DiscussionOrder
        ) {
            repository(owner: $owner, name: $repo) {
                discussions(
                    first: $num, after: $after, categoryId: $categoryId,
                    answered: $answered, states: $states, orderBy: $orderBy
                ) {
                    nodes {
                        title
                        url
                    }
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                }
            }
        }
        """
        limit = input_data.num_discussions
        variables = {
            "owner": owner,
            "repo": repo,
            "num": min(limit, 100),
            "after": None,
            "categoryId": (
                await GithubListDiscussionsBlock._get_category_id(
                    api, owner, repo, input_data.category
                )
                if input_data.category
                else None
            ),
            "answered": {"all": None, "answered": True, "unanswered": False}[
                input_data.answered
            ],
            "states": (
                [input_data.state.upper()] if input_data.state != "all" else None
            ),
            "orderBy": {
                "field": {"created": "CREATED_AT", "updated": "UPDATED_AT"}[
                    input_data.order_by
                ],
                "direction": input_data.direction.upper(),
            },
        }

        discussions: list[GithubListDiscussionsBlock.Output.DiscussionItem] = []
        while len(discussions) < limit:
            response = await api.post(
                "https://api.github.com/graphql",
                json={"query": query, "variables": variables},
            )
            data = response.json()["data"]["repository"]["discussions"]
            discussions.extend(
                {"title": discussion["title"], "url": discussion["url"]}
                for discussion in data["nodes"]
            )
            if not data["pageInfo"]["hasNextPage"]:
                break
            variables["after"] = data["pageInfo"]["endCursor"]
        return discussions[:limit]

    @staticmethod
    async def _get_category_id(
        api: Requests, owner: str, repo: str, category_name: str
    ) -> str:
        query = """
        query($owner: String!, $repo: String!) {
            repository(owner: $owner, name: $repo) {
                discussionCategories(first: 100) {
                    nodes {
                        id
                        name
                    }
                }
            }
        }
        """
        response = await api.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": {"owner": owner, "repo": repo}},
        )
        categories = response.json()["data"]["repository"]["discussionCategories"][
            "nodes"
        ]
        for category in categories:
            if category["name"].lower() == category_name.lower():
                return category["id"]
        raise ValueError(
            f"Discussion category '{category_name}' does not exist in {owner}/{repo}"
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        discussions = await self.list_discussions(credentials, input_data)
        yield "discussions", discussions
        for discussion in discussions:
            yield "discussion", discussion


class GithubListReleasesBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        release_type: Literal["all", "stable", "prerelease", "draft"] = SchemaField(
            description="Only include releases of this type",
            default="all",
        )
        limit: int = SchemaField(
            description="Maximum number of releases to fetch",
            default=100,
            ge=1,
            le=1000,
        )
        published_after: str = SchemaField(
            description="Only include releases published after the given "
            "ISO 8601 timestamp",
            placeholder="2026-01-01T00:00:00Z",
            default="",
            advanced=True,
        )
        published_before: str = SchemaField(
            description="Only include releases published before the given "
            "ISO 8601 timestamp",
            placeholder="2026-01-01T00:00:00Z",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        class ReleaseItem(TypedDict):
            name: str
            tag_name: str
            url: str
            published_at: str
            prerelease: bool
            draft: bool

        release: ReleaseItem = SchemaField(
            title="Release",
            description="Releases with their name, tag, URL, publish date, and type",
        )
        releases: list[ReleaseItem] = SchemaField(
            description="List of releases with their name, tag, URL, publish date, "
            "and type"
        )

    def __init__(self):
        super().__init__(
            id="3460367a-6ba7-4645-8ce6-47b05d040b92",
            description="This block lists releases for a specified GitHub repository, "
            "optionally filtered by type and date range.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListReleasesBlock.Input,
            output_schema=GithubListReleasesBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "releases",
                    [
                        {
                            "name": "v1.0.0",
                            "tag_name": "v1.0.0",
                            "url": "https://github.com/owner/repo/releases/tag/v1.0.0",
                            "published_at": "2026-01-01T00:00:00Z",
                            "prerelease": False,
                            "draft": False,
                        }
                    ],
                ),
                (
                    "release",
                    {
                        "name": "v1.0.0",
                        "tag_name": "v1.0.0",
                        "url": "https://github.com/owner/repo/releases/tag/v1.0.0",
                        "published_at": "2026-01-01T00:00:00Z",
                        "prerelease": False,
                        "draft": False,
                    },
                ),
            ],
            test_mock={
                "list_releases": lambda *args, **kwargs: [
                    {
                        "name": "v1.0.0",
                        "tag_name": "v1.0.0",
                        "url": "https://github.com/owner/repo/releases/tag/v1.0.0",
                        "published_at": "2026-01-01T00:00:00Z",
                        "prerelease": False,
                        "draft": False,
                    }
                ]
            },
        )

    @staticmethod
    async def list_releases(
        credentials: GithubCredentials, input_data: Input
    ) -> list[Output.ReleaseItem]:
        api = get_api(credentials)
        releases_url = input_data.repo_url + "/releases"

        # The releases endpoint has no filter params, so we filter client-side
        published_after = _parse_timestamp(input_data.published_after)
        published_before = _parse_timestamp(input_data.published_before)

        def keep(release: dict) -> bool:
            if input_data.release_type == "stable" and (
                release["prerelease"] or release["draft"]
            ):
                return False
            if input_data.release_type == "prerelease" and not release["prerelease"]:
                return False
            if input_data.release_type == "draft" and not release["draft"]:
                return False
            if published_after or published_before:
                # Drafts have no publish date; use their creation date instead
                published = _parse_timestamp(
                    release.get("published_at") or release["created_at"]
                )
                if published is None:
                    return False
                if published_after and published < published_after:
                    return False
                if published_before and published > published_before:
                    return False
            return True

        is_filtered = (
            input_data.release_type != "all"
            or published_after is not None
            or published_before is not None
        )
        data = await get_paginated(
            api,
            releases_url,
            limit=input_data.limit,
            keep=keep if is_filtered else None,
        )
        releases: list[GithubListReleasesBlock.Output.ReleaseItem] = [
            {
                "name": release["name"] or release["tag_name"],
                "tag_name": release["tag_name"],
                "url": release["html_url"],
                "published_at": release.get("published_at") or "",
                "prerelease": release["prerelease"],
                "draft": release["draft"],
            }
            for release in data
        ]
        return releases

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        releases = await self.list_releases(credentials, input_data)
        yield "releases", releases
        for release in releases:
            yield "release", release


class GithubCreateRepositoryBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        name: str = SchemaField(
            description="Name of the repository to create",
            placeholder="my-new-repo",
        )
        description: str = SchemaField(
            description="Description of the repository",
            placeholder="A description of the repository",
            default="",
        )
        private: bool = SchemaField(
            description="Whether the repository should be private",
            default=False,
        )
        auto_init: bool = SchemaField(
            description="Whether to initialize the repository with a README",
            default=True,
        )
        gitignore_template: str = SchemaField(
            description="Git ignore template to use (e.g., Python, Node, Java)",
            default="",
        )

    class Output(BlockSchemaOutput):
        url: str = SchemaField(description="URL of the created repository")
        clone_url: str = SchemaField(description="Git clone URL of the repository")
        error: str = SchemaField(
            description="Error message if the repository creation failed"
        )

    def __init__(self):
        super().__init__(
            id="029ec3b8-1cfd-46d3-b6aa-28e4a706efd1",
            description="This block creates a new GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCreateRepositoryBlock.Input,
            output_schema=GithubCreateRepositoryBlock.Output,
            test_input={
                "name": "test-repo",
                "description": "A test repository",
                "private": False,
                "auto_init": True,
                "gitignore_template": "Python",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("url", "https://github.com/owner/test-repo"),
                ("clone_url", "https://github.com/owner/test-repo.git"),
            ],
            test_mock={
                "create_repository": lambda *args, **kwargs: (
                    "https://github.com/owner/test-repo",
                    "https://github.com/owner/test-repo.git",
                )
            },
        )

    @staticmethod
    async def create_repository(
        credentials: GithubCredentials,
        name: str,
        description: str,
        private: bool,
        auto_init: bool,
        gitignore_template: str,
    ) -> tuple[str, str]:
        api = get_api(credentials)
        data = {
            "name": name,
            "description": description,
            "private": private,
            "auto_init": auto_init,
            "gitignore_template": gitignore_template,
        }
        response = await api.post("https://api.github.com/user/repos", json=data)
        data = response.json()
        return data["html_url"], data["clone_url"]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            url, clone_url = await self.create_repository(
                credentials,
                input_data.name,
                input_data.description,
                input_data.private,
                input_data.auto_init,
                input_data.gitignore_template,
            )
            yield "url", url
            yield "clone_url", clone_url
        except Exception as e:
            yield "error", str(e)


class GithubListStargazersBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        limit: int = SchemaField(
            description="Maximum number of stargazers to fetch",
            default=100,
            ge=1,
            le=1000,
        )

    class Output(BlockSchemaOutput):
        class StargazerItem(TypedDict):
            username: str
            url: str

        stargazer: StargazerItem = SchemaField(
            title="Stargazer",
            description="Stargazers with their username and profile URL",
        )
        stargazers: list[StargazerItem] = SchemaField(
            description="List of stargazers with their username and profile URL"
        )
        error: str = SchemaField(
            description="Error message if listing stargazers failed"
        )

    def __init__(self):
        super().__init__(
            id="e96d01ec-b55e-4a99-8ce8-c8776dce850b",  # Generated unique UUID
            description="This block lists all users who have starred a specified GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListStargazersBlock.Input,
            output_schema=GithubListStargazersBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "stargazers",
                    [
                        {
                            "username": "octocat",
                            "url": "https://github.com/octocat",
                        }
                    ],
                ),
                (
                    "stargazer",
                    {
                        "username": "octocat",
                        "url": "https://github.com/octocat",
                    },
                ),
            ],
            test_mock={
                "list_stargazers": lambda *args, **kwargs: [
                    {
                        "username": "octocat",
                        "url": "https://github.com/octocat",
                    }
                ]
            },
        )

    @staticmethod
    async def list_stargazers(
        credentials: GithubCredentials, repo_url: str, limit: int
    ) -> list[Output.StargazerItem]:
        api = get_api(credentials)
        stargazers_url = repo_url + "/stargazers"
        data = await get_paginated(api, stargazers_url, limit=limit)
        stargazers: list[GithubListStargazersBlock.Output.StargazerItem] = [
            {
                "username": stargazer["login"],
                "url": stargazer["html_url"],
            }
            for stargazer in data
        ]
        return stargazers

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        stargazers = await self.list_stargazers(
            credentials,
            input_data.repo_url,
            input_data.limit,
        )
        yield "stargazers", stargazers
        for stargazer in stargazers:
            yield "stargazer", stargazer


class GithubGetRepositoryInfoBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchemaOutput):
        name: str = SchemaField(description="Repository name")
        full_name: str = SchemaField(description="Full repository name (owner/repo)")
        description: str = SchemaField(description="Repository description")
        default_branch: str = SchemaField(description="Default branch name (e.g. main)")
        private: bool = SchemaField(description="Whether the repository is private")
        html_url: str = SchemaField(description="Web URL of the repository")
        clone_url: str = SchemaField(description="Git clone URL")
        stars: int = SchemaField(description="Number of stars")
        forks: int = SchemaField(description="Number of forks")
        open_issues: int = SchemaField(description="Number of open issues")
        error: str = SchemaField(
            description="Error message if fetching repo info failed"
        )

    def __init__(self):
        super().__init__(
            id="59d4f241-968a-4040-95da-348ac5c5ce27",
            description="This block retrieves metadata about a GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubGetRepositoryInfoBlock.Input,
            output_schema=GithubGetRepositoryInfoBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("name", "repo"),
                ("full_name", "owner/repo"),
                ("description", "A test repo"),
                ("default_branch", "main"),
                ("private", False),
                ("html_url", "https://github.com/owner/repo"),
                ("clone_url", "https://github.com/owner/repo.git"),
                ("stars", 42),
                ("forks", 5),
                ("open_issues", 3),
            ],
            test_mock={
                "get_repo_info": lambda *args, **kwargs: {
                    "name": "repo",
                    "full_name": "owner/repo",
                    "description": "A test repo",
                    "default_branch": "main",
                    "private": False,
                    "html_url": "https://github.com/owner/repo",
                    "clone_url": "https://github.com/owner/repo.git",
                    "stargazers_count": 42,
                    "forks_count": 5,
                    "open_issues_count": 3,
                }
            },
        )

    @staticmethod
    async def get_repo_info(credentials: GithubCredentials, repo_url: str) -> dict:
        api = get_api(credentials)
        response = await api.get(repo_url)
        return response.json()

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            data = await self.get_repo_info(credentials, input_data.repo_url)
            yield "name", data["name"]
            yield "full_name", data["full_name"]
            yield "description", data.get("description", "") or ""
            yield "default_branch", data["default_branch"]
            yield "private", data["private"]
            yield "html_url", data["html_url"]
            yield "clone_url", data["clone_url"]
            yield "stars", data["stargazers_count"]
            yield "forks", data["forks_count"]
            yield "open_issues", data["open_issues_count"]
        except Exception as e:
            yield "error", str(e)


class GithubForkRepositoryBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository to fork",
            placeholder="https://github.com/owner/repo",
        )
        organization: str = SchemaField(
            description="Organization to fork into (leave empty to fork to your account)",
            default="",
        )

    class Output(BlockSchemaOutput):
        url: str = SchemaField(description="URL of the forked repository")
        clone_url: str = SchemaField(description="Git clone URL of the fork")
        full_name: str = SchemaField(description="Full name of the fork (owner/repo)")
        error: str = SchemaField(description="Error message if the fork failed")

    def __init__(self):
        super().__init__(
            id="a439f2f4-835f-4dae-ba7b-0205ffa70be6",
            description="This block forks a GitHub repository to your account or an organization.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubForkRepositoryBlock.Input,
            output_schema=GithubForkRepositoryBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "organization": "",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("url", "https://github.com/myuser/repo"),
                ("clone_url", "https://github.com/myuser/repo.git"),
                ("full_name", "myuser/repo"),
            ],
            test_mock={
                "fork_repo": lambda *args, **kwargs: (
                    "https://github.com/myuser/repo",
                    "https://github.com/myuser/repo.git",
                    "myuser/repo",
                )
            },
        )

    @staticmethod
    async def fork_repo(
        credentials: GithubCredentials,
        repo_url: str,
        organization: str,
    ) -> tuple[str, str, str]:
        api = get_api(credentials)
        forks_url = repo_url + "/forks"
        data: dict[str, str] = {}
        if organization:
            data["organization"] = organization
        response = await api.post(forks_url, json=data)
        result = response.json()
        return result["html_url"], result["clone_url"], result["full_name"]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            url, clone_url, full_name = await self.fork_repo(
                credentials,
                input_data.repo_url,
                input_data.organization,
            )
            yield "url", url
            yield "clone_url", clone_url
            yield "full_name", full_name
        except Exception as e:
            yield "error", str(e)


class GithubStarRepositoryBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository to star",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="Status of the star operation")
        error: str = SchemaField(description="Error message if starring failed")

    def __init__(self):
        super().__init__(
            id="bd700764-53e3-44dd-a969-d1854088458f",
            description="This block stars a GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubStarRepositoryBlock.Input,
            output_schema=GithubStarRepositoryBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Repository starred successfully")],
            test_mock={
                "star_repo": lambda *args, **kwargs: "Repository starred successfully"
            },
        )

    @staticmethod
    async def star_repo(credentials: GithubCredentials, repo_url: str) -> str:
        api = get_api(credentials, convert_urls=False)
        repo_path = github_repo_path(repo_url)
        owner, repo = repo_path.split("/")
        await api.put(
            f"https://api.github.com/user/starred/{owner}/{repo}",
            headers={"Content-Length": "0"},
        )
        return "Repository starred successfully"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            status = await self.star_repo(credentials, input_data.repo_url)
            yield "status", status
        except Exception as e:
            yield "error", str(e)


def _parse_timestamp(timestamp: str) -> Optional[datetime]:
    if not timestamp:
        return None
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
