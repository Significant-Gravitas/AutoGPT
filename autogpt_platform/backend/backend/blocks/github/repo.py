import base64
from enum import StrEnum

from typing_extensions import TypedDict

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

from ._api import get_api
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentials,
    GithubCredentialsField,
    GithubCredentialsInput,
)


class GithubListTagsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
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
        credentials: GithubCredentials, repo_url: str
    ) -> list[Output.TagItem]:
        api = get_api(credentials)
        tags_url = repo_url + "/tags"
        response = await api.get(tags_url)
        data = response.json()
        repo_path = repo_url.replace("https://github.com/", "")
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
        )
        yield "tags", tags
        for tag in tags:
            yield "tag", tag


class GithubListBranchesBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchemaOutput):
        class BranchItem(TypedDict):
            name: str
            url: str

        branch: BranchItem = SchemaField(
            title="Branch",
            description="Branches with their name and file tree browser URL",
        )
        branches: list[BranchItem] = SchemaField(
            description="List of branches with their name and file tree browser URL"
        )

    def __init__(self):
        super().__init__(
            id="74243e49-2bec-4916-8bf4-db43d44aead5",
            description="This block lists all branches for a specified GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListBranchesBlock.Input,
            output_schema=GithubListBranchesBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "branches",
                    [
                        {
                            "name": "main",
                            "url": "https://github.com/owner/repo/tree/main",
                        }
                    ],
                ),
                (
                    "branch",
                    {
                        "name": "main",
                        "url": "https://github.com/owner/repo/tree/main",
                    },
                ),
            ],
            test_mock={
                "list_branches": lambda *args, **kwargs: [
                    {
                        "name": "main",
                        "url": "https://github.com/owner/repo/tree/main",
                    }
                ]
            },
        )

    @staticmethod
    async def list_branches(
        credentials: GithubCredentials, repo_url: str
    ) -> list[Output.BranchItem]:
        api = get_api(credentials)
        branches_url = repo_url + "/branches"
        response = await api.get(branches_url)
        data = response.json()
        repo_path = repo_url.replace("https://github.com/", "")
        branches: list[GithubListBranchesBlock.Output.BranchItem] = [
            {
                "name": branch["name"],
                "url": f"https://github.com/{repo_path}/tree/{branch['name']}",
            }
            for branch in data
        ]
        return branches

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        branches = await self.list_branches(
            credentials,
            input_data.repo_url,
        )
        yield "branches", branches
        for branch in branches:
            yield "branch", branch


class GithubListDiscussionsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        num_discussions: int = SchemaField(
            description="Number of discussions to fetch", default=5
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
        credentials: GithubCredentials, repo_url: str, num_discussions: int
    ) -> list[Output.DiscussionItem]:
        api = get_api(credentials)
        # GitHub GraphQL API endpoint is different; we'll use api.post with custom URL
        repo_path = repo_url.replace("https://github.com/", "")
        owner, repo = repo_path.split("/")
        query = """
        query($owner: String!, $repo: String!, $num: Int!) {
            repository(owner: $owner, name: $repo) {
                discussions(first: $num) {
                    nodes {
                        title
                        url
                    }
                }
            }
        }
        """
        variables = {"owner": owner, "repo": repo, "num": num_discussions}
        response = await api.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": variables},
        )
        data = response.json()
        discussions: list[GithubListDiscussionsBlock.Output.DiscussionItem] = [
            {"title": discussion["title"], "url": discussion["url"]}
            for discussion in data["data"]["repository"]["discussions"]["nodes"]
        ]
        return discussions

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        discussions = await self.list_discussions(
            credentials,
            input_data.repo_url,
            input_data.num_discussions,
        )
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

    class Output(BlockSchemaOutput):
        class ReleaseItem(TypedDict):
            name: str
            url: str

        release: ReleaseItem = SchemaField(
            title="Release",
            description="Releases with their name and file tree browser URL",
        )
        releases: list[ReleaseItem] = SchemaField(
            description="List of releases with their name and file tree browser URL"
        )

    def __init__(self):
        super().__init__(
            id="3460367a-6ba7-4645-8ce6-47b05d040b92",
            description="This block lists all releases for a specified GitHub repository.",
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
                            "url": "https://github.com/owner/repo/releases/tag/v1.0.0",
                        }
                    ],
                ),
                (
                    "release",
                    {
                        "name": "v1.0.0",
                        "url": "https://github.com/owner/repo/releases/tag/v1.0.0",
                    },
                ),
            ],
            test_mock={
                "list_releases": lambda *args, **kwargs: [
                    {
                        "name": "v1.0.0",
                        "url": "https://github.com/owner/repo/releases/tag/v1.0.0",
                    }
                ]
            },
        )

    @staticmethod
    async def list_releases(
        credentials: GithubCredentials, repo_url: str
    ) -> list[Output.ReleaseItem]:
        api = get_api(credentials)
        releases_url = repo_url + "/releases"
        response = await api.get(releases_url)
        data = response.json()
        releases: list[GithubListReleasesBlock.Output.ReleaseItem] = [
            {"name": release["name"], "url": release["html_url"]} for release in data
        ]
        return releases

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        releases = await self.list_releases(
            credentials,
            input_data.repo_url,
        )
        yield "releases", releases
        for release in releases:
            yield "release", release


class GithubReadFileBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        file_path: str = SchemaField(
            description="Path to the file in the repository",
            placeholder="path/to/file",
        )
        branch: str = SchemaField(
            description="Branch to read from",
            placeholder="branch_name",
            default="master",
        )

    class Output(BlockSchemaOutput):
        text_content: str = SchemaField(
            description="Content of the file (decoded as UTF-8 text)"
        )
        raw_content: str = SchemaField(
            description="Raw base64-encoded content of the file"
        )
        size: int = SchemaField(description="The size of the file (in bytes)")

    def __init__(self):
        super().__init__(
            id="87ce6c27-5752-4bbc-8e26-6da40a3dcfd3",
            description="This block reads the content of a specified file from a GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadFileBlock.Input,
            output_schema=GithubReadFileBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "file_path": "path/to/file",
                "branch": "master",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("raw_content", "RmlsZSBjb250ZW50"),
                ("text_content", "File content"),
                ("size", 13),
            ],
            test_mock={"read_file": lambda *args, **kwargs: ("RmlsZSBjb250ZW50", 13)},
        )

    @staticmethod
    async def read_file(
        credentials: GithubCredentials, repo_url: str, file_path: str, branch: str
    ) -> tuple[str, int]:
        api = get_api(credentials)
        content_url = repo_url + f"/contents/{file_path}?ref={branch}"
        response = await api.get(content_url)
        data = response.json()

        if isinstance(data, list):
            # Multiple entries of different types exist at this path
            if not (file := next((f for f in data if f["type"] == "file"), None)):
                raise TypeError("Not a file")
            data = file

        if data["type"] != "file":
            raise TypeError("Not a file")

        return data["content"], data["size"]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        content, size = await self.read_file(
            credentials,
            input_data.repo_url,
            input_data.file_path,
            input_data.branch,
        )
        yield "raw_content", content
        yield "text_content", base64.b64decode(content).decode("utf-8")
        yield "size", size


class GithubReadFolderBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        folder_path: str = SchemaField(
            description="Path to the folder in the repository",
            placeholder="path/to/folder",
        )
        branch: str = SchemaField(
            description="Branch name to read from (defaults to master)",
            placeholder="branch_name",
            default="master",
        )

    class Output(BlockSchemaOutput):
        class DirEntry(TypedDict):
            name: str
            path: str

        class FileEntry(TypedDict):
            name: str
            path: str
            size: int

        file: FileEntry = SchemaField(description="Files in the folder")
        dir: DirEntry = SchemaField(description="Directories in the folder")
        error: str = SchemaField(
            description="Error message if reading the folder failed"
        )

    def __init__(self):
        super().__init__(
            id="1355f863-2db3-4d75-9fba-f91e8a8ca400",
            description="This block reads the content of a specified folder from a GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadFolderBlock.Input,
            output_schema=GithubReadFolderBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "folder_path": "path/to/folder",
                "branch": "master",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "file",
                    {
                        "name": "file1.txt",
                        "path": "path/to/folder/file1.txt",
                        "size": 1337,
                    },
                ),
                ("dir", {"name": "dir2", "path": "path/to/folder/dir2"}),
            ],
            test_mock={
                "read_folder": lambda *args, **kwargs: (
                    [
                        {
                            "name": "file1.txt",
                            "path": "path/to/folder/file1.txt",
                            "size": 1337,
                        }
                    ],
                    [{"name": "dir2", "path": "path/to/folder/dir2"}],
                )
            },
        )

    @staticmethod
    async def read_folder(
        credentials: GithubCredentials, repo_url: str, folder_path: str, branch: str
    ) -> tuple[list[Output.FileEntry], list[Output.DirEntry]]:
        api = get_api(credentials)
        contents_url = repo_url + f"/contents/{folder_path}?ref={branch}"
        response = await api.get(contents_url)
        data = response.json()

        if not isinstance(data, list):
            raise TypeError("Not a folder")

        files: list[GithubReadFolderBlock.Output.FileEntry] = [
            GithubReadFolderBlock.Output.FileEntry(
                name=entry["name"],
                path=entry["path"],
                size=entry["size"],
            )
            for entry in data
            if entry["type"] == "file"
        ]

        dirs: list[GithubReadFolderBlock.Output.DirEntry] = [
            GithubReadFolderBlock.Output.DirEntry(
                name=entry["name"],
                path=entry["path"],
            )
            for entry in data
            if entry["type"] == "dir"
        ]

        return files, dirs

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        files, dirs = await self.read_folder(
            credentials,
            input_data.repo_url,
            input_data.folder_path.lstrip("/"),
            input_data.branch,
        )
        for file in files:
            yield "file", file
        for dir in dirs:
            yield "dir", dir


class GithubMakeBranchBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        new_branch: str = SchemaField(
            description="Name of the new branch",
            placeholder="new_branch_name",
        )
        source_branch: str = SchemaField(
            description="Name of the source branch",
            placeholder="source_branch_name",
        )

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="Status of the branch creation operation")
        error: str = SchemaField(
            description="Error message if the branch creation failed"
        )

    def __init__(self):
        super().__init__(
            id="944cc076-95e7-4d1b-b6b6-b15d8ee5448d",
            description="This block creates a new branch from a specified source branch.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubMakeBranchBlock.Input,
            output_schema=GithubMakeBranchBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "new_branch": "new_branch_name",
                "source_branch": "source_branch_name",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Branch created successfully")],
            test_mock={
                "create_branch": lambda *args, **kwargs: "Branch created successfully"
            },
        )

    @staticmethod
    async def create_branch(
        credentials: GithubCredentials,
        repo_url: str,
        new_branch: str,
        source_branch: str,
    ) -> str:
        api = get_api(credentials)
        ref_url = repo_url + f"/git/refs/heads/{source_branch}"
        response = await api.get(ref_url)
        data = response.json()
        sha = data["object"]["sha"]

        # Create the new branch
        new_ref_url = repo_url + "/git/refs"
        data = {
            "ref": f"refs/heads/{new_branch}",
            "sha": sha,
        }
        response = await api.post(new_ref_url, json=data)
        return "Branch created successfully"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = await self.create_branch(
            credentials,
            input_data.repo_url,
            input_data.new_branch,
            input_data.source_branch,
        )
        yield "status", status


class GithubDeleteBranchBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        branch: str = SchemaField(
            description="Name of the branch to delete",
            placeholder="branch_name",
        )

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="Status of the branch deletion operation")
        error: str = SchemaField(
            description="Error message if the branch deletion failed"
        )

    def __init__(self):
        super().__init__(
            id="0d4130f7-e0ab-4d55-adc3-0a40225e80f4",
            description="This block deletes a specified branch.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubDeleteBranchBlock.Input,
            output_schema=GithubDeleteBranchBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "branch": "branch_name",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Branch deleted successfully")],
            test_mock={
                "delete_branch": lambda *args, **kwargs: "Branch deleted successfully"
            },
        )

    @staticmethod
    async def delete_branch(
        credentials: GithubCredentials, repo_url: str, branch: str
    ) -> str:
        api = get_api(credentials)
        ref_url = repo_url + f"/git/refs/heads/{branch}"
        await api.delete(ref_url)
        return "Branch deleted successfully"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = await self.delete_branch(
            credentials,
            input_data.repo_url,
            input_data.branch,
        )
        yield "status", status


class GithubCreateFileBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        file_path: str = SchemaField(
            description="Path where the file should be created",
            placeholder="path/to/file.txt",
        )
        content: str = SchemaField(
            description="Content to write to the file",
            placeholder="File content here",
        )
        branch: str = SchemaField(
            description="Branch where the file should be created",
            default="main",
        )
        commit_message: str = SchemaField(
            description="Message for the commit",
            default="Create new file",
        )

    class Output(BlockSchemaOutput):
        url: str = SchemaField(description="URL of the created file")
        sha: str = SchemaField(description="SHA of the commit")
        error: str = SchemaField(
            description="Error message if the file creation failed"
        )

    def __init__(self):
        super().__init__(
            id="8fd132ac-b917-428a-8159-d62893e8a3fe",
            description="This block creates a new file in a GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCreateFileBlock.Input,
            output_schema=GithubCreateFileBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "file_path": "test/file.txt",
                "content": "Test content",
                "branch": "main",
                "commit_message": "Create test file",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("url", "https://github.com/owner/repo/blob/main/test/file.txt"),
                ("sha", "abc123"),
            ],
            test_mock={
                "create_file": lambda *args, **kwargs: (
                    "https://github.com/owner/repo/blob/main/test/file.txt",
                    "abc123",
                )
            },
        )

    @staticmethod
    async def create_file(
        credentials: GithubCredentials,
        repo_url: str,
        file_path: str,
        content: str,
        branch: str,
        commit_message: str,
    ) -> tuple[str, str]:
        api = get_api(credentials)
        contents_url = repo_url + f"/contents/{file_path}"
        content_base64 = base64.b64encode(content.encode()).decode()
        data = {
            "message": commit_message,
            "content": content_base64,
            "branch": branch,
        }
        response = await api.put(contents_url, json=data)
        data = response.json()
        return data["content"]["html_url"], data["commit"]["sha"]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            url, sha = await self.create_file(
                credentials,
                input_data.repo_url,
                input_data.file_path,
                input_data.content,
                input_data.branch,
                input_data.commit_message,
            )
            yield "url", url
            yield "sha", sha
        except Exception as e:
            yield "error", str(e)


class GithubUpdateFileBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        file_path: str = SchemaField(
            description="Path to the file to update",
            placeholder="path/to/file.txt",
        )
        content: str = SchemaField(
            description="New content for the file",
            placeholder="Updated content here",
        )
        branch: str = SchemaField(
            description="Branch containing the file",
            default="main",
        )
        commit_message: str = SchemaField(
            description="Message for the commit",
            default="Update file",
        )

    class Output(BlockSchemaOutput):
        url: str = SchemaField(description="URL of the updated file")
        sha: str = SchemaField(description="SHA of the commit")

    def __init__(self):
        super().__init__(
            id="30be12a4-57cb-4aa4-baf5-fcc68d136076",
            description="This block updates an existing file in a GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubUpdateFileBlock.Input,
            output_schema=GithubUpdateFileBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "file_path": "test/file.txt",
                "content": "Updated content",
                "branch": "main",
                "commit_message": "Update test file",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("url", "https://github.com/owner/repo/blob/main/test/file.txt"),
                ("sha", "def456"),
            ],
            test_mock={
                "update_file": lambda *args, **kwargs: (
                    "https://github.com/owner/repo/blob/main/test/file.txt",
                    "def456",
                )
            },
        )

    @staticmethod
    async def update_file(
        credentials: GithubCredentials,
        repo_url: str,
        file_path: str,
        content: str,
        branch: str,
        commit_message: str,
    ) -> tuple[str, str]:
        api = get_api(credentials)
        contents_url = repo_url + f"/contents/{file_path}"
        params = {"ref": branch}
        response = await api.get(contents_url, params=params)
        data = response.json()

        # Convert new content to base64
        content_base64 = base64.b64encode(content.encode()).decode()
        data = {
            "message": commit_message,
            "content": content_base64,
            "sha": data["sha"],
            "branch": branch,
        }
        response = await api.put(contents_url, json=data)
        data = response.json()
        return data["content"]["html_url"], data["commit"]["sha"]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            url, sha = await self.update_file(
                credentials,
                input_data.repo_url,
                input_data.file_path,
                input_data.content,
                input_data.branch,
                input_data.commit_message,
            )
            yield "url", url
            yield "sha", sha
        except Exception as e:
            yield "error", str(e)


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
            id="a4b9c2d1-e5f6-4g7h-8i9j-0k1l2m3n4o5p",  # Generated unique UUID
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
        credentials: GithubCredentials, repo_url: str
    ) -> list[Output.StargazerItem]:
        api = get_api(credentials)
        stargazers_url = repo_url + "/stargazers"
        response = await api.get(stargazers_url)
        data = response.json()
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


class GithubListCommitsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        branch: str = SchemaField(
            description="Branch name to list commits from",
            default="main",
        )
        per_page: int = SchemaField(
            description="Number of commits to return (max 100)",
            default=30,
        )

    class Output(BlockSchemaOutput):
        class CommitItem(TypedDict):
            sha: str
            message: str
            author: str
            date: str
            url: str

        commit: CommitItem = SchemaField(
            title="Commit", description="A commit with its details"
        )
        commits: list[CommitItem] = SchemaField(
            description="List of commits with their details"
        )
        error: str = SchemaField(description="Error message if listing commits failed")

    def __init__(self):
        super().__init__(
            id="8b13f579-d8b6-4dc2-a140-f770428805de",
            description="This block lists commits on a branch in a GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListCommitsBlock.Input,
            output_schema=GithubListCommitsBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "branch": "main",
                "per_page": 30,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "commits",
                    [
                        {
                            "sha": "abc123",
                            "message": "Initial commit",
                            "author": "octocat",
                            "date": "2024-01-01T00:00:00Z",
                            "url": "https://github.com/owner/repo/commit/abc123",
                        }
                    ],
                ),
                (
                    "commit",
                    {
                        "sha": "abc123",
                        "message": "Initial commit",
                        "author": "octocat",
                        "date": "2024-01-01T00:00:00Z",
                        "url": "https://github.com/owner/repo/commit/abc123",
                    },
                ),
            ],
            test_mock={
                "list_commits": lambda *args, **kwargs: [
                    {
                        "sha": "abc123",
                        "message": "Initial commit",
                        "author": "octocat",
                        "date": "2024-01-01T00:00:00Z",
                        "url": "https://github.com/owner/repo/commit/abc123",
                    }
                ]
            },
        )

    @staticmethod
    async def list_commits(
        credentials: GithubCredentials,
        repo_url: str,
        branch: str,
        per_page: int,
    ) -> list[Output.CommitItem]:
        api = get_api(credentials)
        commits_url = repo_url + "/commits"
        params = {"sha": branch, "per_page": str(per_page)}
        response = await api.get(commits_url, params=params)
        data = response.json()
        repo_path = repo_url.replace("https://github.com/", "")
        return [
            GithubListCommitsBlock.Output.CommitItem(
                sha=c["sha"],
                message=c["commit"]["message"],
                author=c["commit"]["author"]["name"],
                date=c["commit"]["author"]["date"],
                url=f"https://github.com/{repo_path}/commit/{c['sha']}",
            )
            for c in data
        ]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            commits = await self.list_commits(
                credentials,
                input_data.repo_url,
                input_data.branch,
                input_data.per_page,
            )
            yield "commits", commits
            for commit in commits:
                yield "commit", commit
        except Exception as e:
            yield "error", str(e)


class GithubSearchCodeBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        query: str = SchemaField(
            description="Search query (GitHub code search syntax)",
            placeholder="className language:python",
        )
        repo: str = SchemaField(
            description="Restrict search to a repository (owner/repo format, optional)",
            default="",
            placeholder="owner/repo",
        )
        per_page: int = SchemaField(
            description="Number of results to return (max 100)",
            default=30,
        )

    class Output(BlockSchemaOutput):
        class SearchResult(TypedDict):
            name: str
            path: str
            repository: str
            url: str
            score: float

        result: SearchResult = SchemaField(
            title="Result", description="A code search result"
        )
        results: list[SearchResult] = SchemaField(
            description="List of code search results"
        )
        total_count: int = SchemaField(description="Total number of matching results")
        error: str = SchemaField(description="Error message if search failed")

    def __init__(self):
        super().__init__(
            id="47f94891-a2b1-4f1c-b5f2-573c043f721e",
            description="This block searches for code in GitHub repositories.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubSearchCodeBlock.Input,
            output_schema=GithubSearchCodeBlock.Output,
            test_input={
                "query": "addClass",
                "repo": "owner/repo",
                "per_page": 30,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("total_count", 1),
                (
                    "results",
                    [
                        {
                            "name": "file.py",
                            "path": "src/file.py",
                            "repository": "owner/repo",
                            "url": "https://github.com/owner/repo/blob/main/src/file.py",
                            "score": 1.0,
                        }
                    ],
                ),
                (
                    "result",
                    {
                        "name": "file.py",
                        "path": "src/file.py",
                        "repository": "owner/repo",
                        "url": "https://github.com/owner/repo/blob/main/src/file.py",
                        "score": 1.0,
                    },
                ),
            ],
            test_mock={
                "search_code": lambda *args, **kwargs: (
                    1,
                    [
                        {
                            "name": "file.py",
                            "path": "src/file.py",
                            "repository": "owner/repo",
                            "url": "https://github.com/owner/repo/blob/main/src/file.py",
                            "score": 1.0,
                        }
                    ],
                )
            },
        )

    @staticmethod
    async def search_code(
        credentials: GithubCredentials,
        query: str,
        repo: str,
        per_page: int,
    ) -> tuple[int, list[Output.SearchResult]]:
        api = get_api(credentials, convert_urls=False)
        full_query = f"{query} repo:{repo}" if repo else query
        params = {"q": full_query, "per_page": str(per_page)}
        response = await api.get("https://api.github.com/search/code", params=params)
        data = response.json()
        results: list[GithubSearchCodeBlock.Output.SearchResult] = [
            GithubSearchCodeBlock.Output.SearchResult(
                name=item["name"],
                path=item["path"],
                repository=item["repository"]["full_name"],
                url=item["html_url"],
                score=item["score"],
            )
            for item in data["items"]
        ]
        return data["total_count"], results

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            total_count, results = await self.search_code(
                credentials,
                input_data.query,
                input_data.repo,
                input_data.per_page,
            )
            yield "total_count", total_count
            yield "results", results
            for result in results:
                yield "result", result
        except Exception as e:
            yield "error", str(e)


class GithubCompareBranchesBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        base: str = SchemaField(
            description="Base branch or commit SHA",
            placeholder="main",
        )
        head: str = SchemaField(
            description="Head branch or commit SHA to compare against base",
            placeholder="feature-branch",
        )

    class Output(BlockSchemaOutput):
        class FileChange(TypedDict):
            filename: str
            status: str
            additions: int
            deletions: int
            patch: str

        status: str = SchemaField(
            description="Comparison status: ahead, behind, diverged, or identical"
        )
        ahead_by: int = SchemaField(
            description="Number of commits head is ahead of base"
        )
        behind_by: int = SchemaField(
            description="Number of commits head is behind base"
        )
        total_commits: int = SchemaField(
            description="Total number of commits in the comparison"
        )
        diff: str = SchemaField(description="Unified diff of all file changes")
        file: FileChange = SchemaField(
            title="Changed File", description="A changed file with its diff"
        )
        files: list[FileChange] = SchemaField(
            description="List of changed files with their diffs"
        )
        error: str = SchemaField(description="Error message if comparison failed")

    def __init__(self):
        super().__init__(
            id="2e4faa8c-6086-4546-ba77-172d1d560186",
            description="This block compares two branches or commits in a GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCompareBranchesBlock.Input,
            output_schema=GithubCompareBranchesBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "base": "main",
                "head": "feature",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("status", "ahead"),
                ("ahead_by", 2),
                ("behind_by", 0),
                ("total_commits", 2),
                ("diff", "+++ b/file.py\n+new line"),
                (
                    "files",
                    [
                        {
                            "filename": "file.py",
                            "status": "modified",
                            "additions": 1,
                            "deletions": 0,
                            "patch": "+new line",
                        }
                    ],
                ),
                (
                    "file",
                    {
                        "filename": "file.py",
                        "status": "modified",
                        "additions": 1,
                        "deletions": 0,
                        "patch": "+new line",
                    },
                ),
            ],
            test_mock={
                "compare_branches": lambda *args, **kwargs: {
                    "status": "ahead",
                    "ahead_by": 2,
                    "behind_by": 0,
                    "total_commits": 2,
                    "files": [
                        {
                            "filename": "file.py",
                            "status": "modified",
                            "additions": 1,
                            "deletions": 0,
                            "patch": "+new line",
                        }
                    ],
                }
            },
        )

    @staticmethod
    async def compare_branches(
        credentials: GithubCredentials,
        repo_url: str,
        base: str,
        head: str,
    ) -> dict:
        api = get_api(credentials)
        compare_url = repo_url + f"/compare/{base}...{head}"
        response = await api.get(compare_url)
        return response.json()

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            data = await self.compare_branches(
                credentials,
                input_data.repo_url,
                input_data.base,
                input_data.head,
            )
            yield "status", data["status"]
            yield "ahead_by", data["ahead_by"]
            yield "behind_by", data["behind_by"]
            yield "total_commits", data["total_commits"]

            files: list[GithubCompareBranchesBlock.Output.FileChange] = [
                GithubCompareBranchesBlock.Output.FileChange(
                    filename=f["filename"],
                    status=f["status"],
                    additions=f["additions"],
                    deletions=f["deletions"],
                    patch=f.get("patch", ""),
                )
                for f in data.get("files", [])
            ]

            # Build unified diff
            diff_parts = []
            for f in data.get("files", []):
                patch = f.get("patch", "")
                if patch:
                    diff_parts.append(f"+++ b/{f['filename']}\n{patch}")
            yield "diff", "\n".join(diff_parts)

            yield "files", files
            for file in files:
                yield "file", file
        except Exception as e:
            yield "error", str(e)


class GithubGetRepositoryTreeBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        branch: str = SchemaField(
            description="Branch name to get the tree from",
            default="main",
        )
        recursive: bool = SchemaField(
            description="Whether to recursively list the entire tree",
            default=True,
        )

    class Output(BlockSchemaOutput):
        class TreeEntry(TypedDict):
            path: str
            type: str
            size: int
            sha: str

        entry: TreeEntry = SchemaField(
            title="Tree Entry", description="A file or directory in the tree"
        )
        entries: list[TreeEntry] = SchemaField(
            description="List of all files and directories in the tree"
        )
        truncated: bool = SchemaField(
            description="Whether the tree was truncated due to size"
        )
        error: str = SchemaField(description="Error message if getting tree failed")

    def __init__(self):
        super().__init__(
            id="89c5c0ec-172e-4001-a32c-bdfe4d0c9e81",
            description="This block lists the entire file tree of a GitHub repository recursively.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubGetRepositoryTreeBlock.Input,
            output_schema=GithubGetRepositoryTreeBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "branch": "main",
                "recursive": True,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("truncated", False),
                (
                    "entries",
                    [
                        {
                            "path": "src/main.py",
                            "type": "blob",
                            "size": 1234,
                            "sha": "abc123",
                        }
                    ],
                ),
                (
                    "entry",
                    {
                        "path": "src/main.py",
                        "type": "blob",
                        "size": 1234,
                        "sha": "abc123",
                    },
                ),
            ],
            test_mock={
                "get_tree": lambda *args, **kwargs: (
                    False,
                    [
                        {
                            "path": "src/main.py",
                            "type": "blob",
                            "size": 1234,
                            "sha": "abc123",
                        }
                    ],
                )
            },
        )

    @staticmethod
    async def get_tree(
        credentials: GithubCredentials,
        repo_url: str,
        branch: str,
        recursive: bool,
    ) -> tuple[bool, list[Output.TreeEntry]]:
        api = get_api(credentials)
        tree_url = repo_url + f"/git/trees/{branch}"
        params = {"recursive": "1"} if recursive else {}
        response = await api.get(tree_url, params=params)
        data = response.json()
        entries: list[GithubGetRepositoryTreeBlock.Output.TreeEntry] = [
            GithubGetRepositoryTreeBlock.Output.TreeEntry(
                path=item["path"],
                type=item["type"],
                size=item.get("size", 0),
                sha=item["sha"],
            )
            for item in data["tree"]
        ]
        return data.get("truncated", False), entries

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            truncated, entries = await self.get_tree(
                credentials,
                input_data.repo_url,
                input_data.branch,
                input_data.recursive,
            )
            yield "truncated", truncated
            yield "entries", entries
            for entry in entries:
                yield "entry", entry
        except Exception as e:
            yield "error", str(e)


class FileOperation(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class FileOperationInput(TypedDict):
    path: str
    content: str
    operation: str


class GithubMultiFileCommitBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        branch: str = SchemaField(
            description="Branch to commit to",
            placeholder="feature-branch",
        )
        commit_message: str = SchemaField(
            description="Commit message",
            placeholder="Add new feature",
        )
        files: list[FileOperationInput] = SchemaField(
            description=(
                "List of file operations. Each item has: "
                "'path' (file path), 'content' (file content, omit for delete), "
                "'operation' (create/update/delete)"
            ),
        )

    class Output(BlockSchemaOutput):
        sha: str = SchemaField(description="SHA of the new commit")
        url: str = SchemaField(description="URL of the new commit")
        error: str = SchemaField(description="Error message if the commit failed")

    def __init__(self):
        super().__init__(
            id="76c7c716-94f0-4bc1-8551-0cccbea8e443",
            description=(
                "This block creates a single commit with multiple file "
                "create/update/delete operations using the Git Trees API."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubMultiFileCommitBlock.Input,
            output_schema=GithubMultiFileCommitBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "branch": "feature",
                "commit_message": "Add files",
                "files": [
                    {
                        "path": "src/new.py",
                        "content": "print('hello')",
                        "operation": "create",
                    },
                    {
                        "path": "src/old.py",
                        "content": "",
                        "operation": "delete",
                    },
                ],
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("sha", "newcommitsha"),
                ("url", "https://github.com/owner/repo/commit/newcommitsha"),
            ],
            test_mock={
                "multi_file_commit": lambda *args, **kwargs: (
                    "newcommitsha",
                    "https://github.com/owner/repo/commit/newcommitsha",
                )
            },
        )

    @staticmethod
    async def multi_file_commit(
        credentials: GithubCredentials,
        repo_url: str,
        branch: str,
        commit_message: str,
        files: list[FileOperationInput],
    ) -> tuple[str, str]:
        api = get_api(credentials)

        # 1. Get the latest commit SHA for the branch
        ref_url = repo_url + f"/git/refs/heads/{branch}"
        response = await api.get(ref_url)
        ref_data = response.json()
        latest_commit_sha = ref_data["object"]["sha"]

        # 2. Get the tree SHA of the latest commit
        commit_url = repo_url + f"/git/commits/{latest_commit_sha}"
        response = await api.get(commit_url)
        commit_data = response.json()
        base_tree_sha = commit_data["tree"]["sha"]

        # 3. Build tree entries for each file operation
        tree_entries = []
        for file_op in files:
            path = file_op["path"]
            operation = file_op.get("operation", "create")

            if operation == FileOperation.DELETE:
                tree_entries.append(
                    {
                        "path": path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": None,  # null SHA = delete
                    }
                )
            else:
                # Create a blob for the file content
                blob_url = repo_url + "/git/blobs"
                blob_response = await api.post(
                    blob_url,
                    json={
                        "content": file_op.get("content", ""),
                        "encoding": "utf-8",
                    },
                )
                blob_sha = blob_response.json()["sha"]
                tree_entries.append(
                    {
                        "path": path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob_sha,
                    }
                )

        # 4. Create a new tree
        tree_url = repo_url + "/git/trees"
        tree_response = await api.post(
            tree_url,
            json={"base_tree": base_tree_sha, "tree": tree_entries},
        )
        new_tree_sha = tree_response.json()["sha"]

        # 5. Create a new commit
        new_commit_url = repo_url + "/git/commits"
        commit_response = await api.post(
            new_commit_url,
            json={
                "message": commit_message,
                "tree": new_tree_sha,
                "parents": [latest_commit_sha],
            },
        )
        new_commit_sha = commit_response.json()["sha"]

        # 6. Update the branch reference
        await api.patch(
            ref_url,
            json={"sha": new_commit_sha},
        )

        repo_path = repo_url.replace("https://github.com/", "")
        commit_web_url = f"https://github.com/{repo_path}/commit/{new_commit_sha}"
        return new_commit_sha, commit_web_url

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            sha, url = await self.multi_file_commit(
                credentials,
                input_data.repo_url,
                input_data.branch,
                input_data.commit_message,
                input_data.files,
            )
            yield "sha", sha
            yield "url", url
        except Exception as e:
            yield "error", str(e)
