import base64

from typing_extensions import TypedDict

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
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
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        class TagItem(TypedDict):
            name: str
            url: str

        tag: TagItem = SchemaField(
            title="Tag", description="Tags with their name and file tree browser URL"
        )
        error: str = SchemaField(description="Error message if listing tags failed")

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
                    "tag",
                    {
                        "name": "v1.0.0",
                        "url": "https://github.com/owner/repo/tree/v1.0.0",
                    },
                )
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
    def list_tags(
        credentials: GithubCredentials, repo_url: str
    ) -> list[Output.TagItem]:
        api = get_api(credentials)
        tags_url = repo_url + "/tags"
        response = api.get(tags_url)
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

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        tags = self.list_tags(
            credentials,
            input_data.repo_url,
        )
        yield from (("tag", tag) for tag in tags)


class GithubListBranchesBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        class BranchItem(TypedDict):
            name: str
            url: str

        branch: BranchItem = SchemaField(
            title="Branch",
            description="Branches with their name and file tree browser URL",
        )
        error: str = SchemaField(description="Error message if listing branches failed")

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
                    "branch",
                    {
                        "name": "main",
                        "url": "https://github.com/owner/repo/tree/main",
                    },
                )
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
    def list_branches(
        credentials: GithubCredentials, repo_url: str
    ) -> list[Output.BranchItem]:
        api = get_api(credentials)
        branches_url = repo_url + "/branches"
        response = api.get(branches_url)
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

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        branches = self.list_branches(
            credentials,
            input_data.repo_url,
        )
        yield from (("branch", branch) for branch in branches)


class GithubListDiscussionsBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        num_discussions: int = SchemaField(
            description="Number of discussions to fetch", default=5
        )

    class Output(BlockSchema):
        class DiscussionItem(TypedDict):
            title: str
            url: str

        discussion: DiscussionItem = SchemaField(
            title="Discussion", description="Discussions with their title and URL"
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
                    "discussion",
                    {
                        "title": "Discussion 1",
                        "url": "https://github.com/owner/repo/discussions/1",
                    },
                )
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
    def list_discussions(
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
        response = api.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": variables},
        )
        data = response.json()
        discussions: list[GithubListDiscussionsBlock.Output.DiscussionItem] = [
            {"title": discussion["title"], "url": discussion["url"]}
            for discussion in data["data"]["repository"]["discussions"]["nodes"]
        ]
        return discussions

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        discussions = self.list_discussions(
            credentials, input_data.repo_url, input_data.num_discussions
        )
        yield from (("discussion", discussion) for discussion in discussions)


class GithubListReleasesBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        class ReleaseItem(TypedDict):
            name: str
            url: str

        release: ReleaseItem = SchemaField(
            title="Release",
            description="Releases with their name and file tree browser URL",
        )
        error: str = SchemaField(description="Error message if listing releases failed")

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
                    "release",
                    {
                        "name": "v1.0.0",
                        "url": "https://github.com/owner/repo/releases/tag/v1.0.0",
                    },
                )
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
    def list_releases(
        credentials: GithubCredentials, repo_url: str
    ) -> list[Output.ReleaseItem]:
        api = get_api(credentials)
        releases_url = repo_url + "/releases"
        response = api.get(releases_url)
        data = response.json()
        releases: list[GithubListReleasesBlock.Output.ReleaseItem] = [
            {"name": release["name"], "url": release["html_url"]} for release in data
        ]
        return releases

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        releases = self.list_releases(
            credentials,
            input_data.repo_url,
        )
        yield from (("release", release) for release in releases)


class GithubReadFileBlock(Block):
    class Input(BlockSchema):
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

    class Output(BlockSchema):
        text_content: str = SchemaField(
            description="Content of the file (decoded as UTF-8 text)"
        )
        raw_content: str = SchemaField(
            description="Raw base64-encoded content of the file"
        )
        size: int = SchemaField(description="The size of the file (in bytes)")
        error: str = SchemaField(description="Error message if the file reading failed")

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
    def read_file(
        credentials: GithubCredentials, repo_url: str, file_path: str, branch: str
    ) -> tuple[str, int]:
        api = get_api(credentials)
        content_url = repo_url + f"/contents/{file_path}?ref={branch}"
        response = api.get(content_url)
        content = response.json()

        if isinstance(content, list):
            # Multiple entries of different types exist at this path
            if not (file := next((f for f in content if f["type"] == "file"), None)):
                raise TypeError("Not a file")
            content = file

        if content["type"] != "file":
            raise TypeError("Not a file")

        return content["content"], content["size"]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        raw_content, size = self.read_file(
            credentials,
            input_data.repo_url,
            input_data.file_path.lstrip("/"),
            input_data.branch,
        )
        yield "raw_content", raw_content
        yield "text_content", base64.b64decode(raw_content).decode("utf-8")
        yield "size", size


class GithubReadFolderBlock(Block):
    class Input(BlockSchema):
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

    class Output(BlockSchema):
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
    def read_folder(
        credentials: GithubCredentials, repo_url: str, folder_path: str, branch: str
    ) -> tuple[list[Output.FileEntry], list[Output.DirEntry]]:
        api = get_api(credentials)
        contents_url = repo_url + f"/contents/{folder_path}?ref={branch}"
        response = api.get(contents_url)
        content = response.json()

        if not isinstance(content, list):
            raise TypeError("Not a folder")

        files = [
            GithubReadFolderBlock.Output.FileEntry(
                name=entry["name"],
                path=entry["path"],
                size=entry["size"],
            )
            for entry in content
            if entry["type"] == "file"
        ]
        dirs = [
            GithubReadFolderBlock.Output.DirEntry(
                name=entry["name"],
                path=entry["path"],
            )
            for entry in content
            if entry["type"] == "dir"
        ]

        return files, dirs

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        files, dirs = self.read_folder(
            credentials,
            input_data.repo_url,
            input_data.folder_path.lstrip("/"),
            input_data.branch,
        )
        yield from (("file", file) for file in files)
        yield from (("dir", dir) for dir in dirs)


class GithubMakeBranchBlock(Block):
    class Input(BlockSchema):
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

    class Output(BlockSchema):
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
    def create_branch(
        credentials: GithubCredentials,
        repo_url: str,
        new_branch: str,
        source_branch: str,
    ) -> str:
        api = get_api(credentials)
        # Get the SHA of the source branch
        ref_url = repo_url + f"/git/refs/heads/{source_branch}"
        response = api.get(ref_url)
        sha = response.json()["object"]["sha"]

        # Create the new branch
        create_ref_url = repo_url + "/git/refs"
        data = {"ref": f"refs/heads/{new_branch}", "sha": sha}
        response = api.post(create_ref_url, json=data)
        return "Branch created successfully"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.create_branch(
            credentials,
            input_data.repo_url,
            input_data.new_branch,
            input_data.source_branch,
        )
        yield "status", status


class GithubDeleteBranchBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        branch: str = SchemaField(
            description="Name of the branch to delete",
            placeholder="branch_name",
        )

    class Output(BlockSchema):
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
    def delete_branch(
        credentials: GithubCredentials, repo_url: str, branch: str
    ) -> str:
        api = get_api(credentials)
        ref_url = repo_url + f"/git/refs/heads/{branch}"
        api.delete(ref_url)
        return "Branch deleted successfully"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.delete_branch(
            credentials,
            input_data.repo_url,
            input_data.branch,
        )
        yield "status", status


class GithubCreateFileBlock(Block):
    class Input(BlockSchema):
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

    class Output(BlockSchema):
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
    def create_file(
        credentials: GithubCredentials,
        repo_url: str,
        file_path: str,
        content: str,
        branch: str,
        commit_message: str,
    ) -> tuple[str, str]:
        api = get_api(credentials)
        # Convert content to base64
        content_bytes = content.encode("utf-8")
        content_base64 = base64.b64encode(content_bytes).decode("utf-8")

        # Create the file using the GitHub API
        contents_url = f"{repo_url}/contents/{file_path}"
        data = {
            "message": commit_message,
            "content": content_base64,
            "branch": branch,
        }
        response = api.put(contents_url, json=data)
        result = response.json()

        return result["content"]["html_url"], result["commit"]["sha"]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            url, sha = self.create_file(
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
    class Input(BlockSchema):
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

    class Output(BlockSchema):
        url: str = SchemaField(description="URL of the updated file")
        sha: str = SchemaField(description="SHA of the commit")
        error: str = SchemaField(description="Error message if the file update failed")

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
    def update_file(
        credentials: GithubCredentials,
        repo_url: str,
        file_path: str,
        content: str,
        branch: str,
        commit_message: str,
    ) -> tuple[str, str]:
        api = get_api(credentials)

        # First get the current file to get its SHA
        contents_url = f"{repo_url}/contents/{file_path}"
        params = {"ref": branch}
        response = api.get(contents_url, params=params)
        current_file = response.json()

        # Convert new content to base64
        content_bytes = content.encode("utf-8")
        content_base64 = base64.b64encode(content_bytes).decode("utf-8")

        # Update the file
        data = {
            "message": commit_message,
            "content": content_base64,
            "sha": current_file["sha"],
            "branch": branch,
        }
        response = api.put(contents_url, json=data)
        result = response.json()

        return result["content"]["html_url"], result["commit"]["sha"]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            url, sha = self.update_file(
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
    class Input(BlockSchema):
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

    class Output(BlockSchema):
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
    def create_repository(
        credentials: GithubCredentials,
        name: str,
        description: str,
        private: bool,
        auto_init: bool,
        gitignore_template: str,
    ) -> tuple[str, str]:
        api = get_api(credentials, convert_urls=False)  # Disable URL conversion
        data = {
            "name": name,
            "description": description,
            "private": private,
            "auto_init": auto_init,
        }

        if gitignore_template:
            data["gitignore_template"] = gitignore_template

        # Create repository using the user endpoint
        response = api.post("https://api.github.com/user/repos", json=data)
        result = response.json()

        return result["html_url"], result["clone_url"]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            url, clone_url = self.create_repository(
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
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        class StargazerItem(TypedDict):
            username: str
            url: str

        stargazer: StargazerItem = SchemaField(
            title="Stargazer",
            description="Stargazers with their username and profile URL",
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
                    "stargazer",
                    {
                        "username": "octocat",
                        "url": "https://github.com/octocat",
                    },
                )
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
    def list_stargazers(
        credentials: GithubCredentials, repo_url: str
    ) -> list[Output.StargazerItem]:
        api = get_api(credentials)
        # Add /stargazers to the repo URL to get stargazers endpoint
        stargazers_url = f"{repo_url}/stargazers"
        # Set accept header to get starred_at timestamp
        headers = {"Accept": "application/vnd.github.star+json"}
        response = api.get(stargazers_url, headers=headers)
        data = response.json()

        stargazers: list[GithubListStargazersBlock.Output.StargazerItem] = [
            {
                "username": stargazer["login"],
                "url": stargazer["html_url"],
            }
            for stargazer in data
        ]
        return stargazers

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            stargazers = self.list_stargazers(
                credentials,
                input_data.repo_url,
            )
            yield from (("stargazer", stargazer) for stargazer in stargazers)
        except Exception as e:
            yield "error", str(e)
