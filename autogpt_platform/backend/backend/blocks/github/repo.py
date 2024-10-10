import base64

import requests
from typing_extensions import TypedDict

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

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
        repo_path = repo_url.replace("https://github.com/", "")
        api_url = f"https://api.github.com/repos/{repo_path}/tags"
        headers = {
            "Authorization": credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        data = response.json()
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
        api_url = repo_url.replace("github.com", "api.github.com/repos") + "/branches"
        headers = {
            "Authorization": credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        data = response.json()
        branches: list[GithubListBranchesBlock.Output.BranchItem] = [
            {"name": branch["name"], "url": branch["commit"]["url"]} for branch in data
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
        headers = {
            "Authorization": credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": variables},
            headers=headers,
        )
        response.raise_for_status()

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
        repo_path = repo_url.replace("https://github.com/", "")
        api_url = f"https://api.github.com/repos/{repo_path}/releases"
        headers = {
            "Authorization": credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

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
        repo_path = repo_url.replace("https://github.com/", "")
        api_url = f"https://api.github.com/repos/{repo_path}/contents/{file_path}?ref={branch}"
        headers = {
            "Authorization": credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

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
        repo_path = repo_url.replace("https://github.com/", "")
        api_url = f"https://api.github.com/repos/{repo_path}/contents/{folder_path}?ref={branch}"
        headers = {
            "Authorization": credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        content = response.json()

        if isinstance(content, list):
            # Multiple entries of different types exist at this path
            if not (dir := next((d for d in content if d["type"] == "dir"), None)):
                raise TypeError("Not a folder")
            content = dir

        if content["type"] != "dir":
            raise TypeError("Not a folder")

        return (
            [
                GithubReadFolderBlock.Output.FileEntry(
                    name=entry["name"],
                    path=entry["path"],
                    size=entry["size"],
                )
                for entry in content["entries"]
                if entry["type"] == "file"
            ],
            [
                GithubReadFolderBlock.Output.DirEntry(
                    name=entry["name"],
                    path=entry["path"],
                )
                for entry in content["entries"]
                if entry["type"] == "dir"
            ],
        )

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
        repo_path = repo_url.replace("https://github.com/", "")
        ref_api_url = (
            f"https://api.github.com/repos/{repo_path}/git/refs/heads/{source_branch}"
        )
        headers = {
            "Authorization": credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.get(ref_api_url, headers=headers)
        response.raise_for_status()

        sha = response.json()["object"]["sha"]

        create_branch_api_url = f"https://api.github.com/repos/{repo_path}/git/refs"
        data = {"ref": f"refs/heads/{new_branch}", "sha": sha}

        response = requests.post(create_branch_api_url, headers=headers, json=data)
        response.raise_for_status()

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
        repo_path = repo_url.replace("https://github.com/", "")
        api_url = f"https://api.github.com/repos/{repo_path}/git/refs/heads/{branch}"
        headers = {
            "Authorization": credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.delete(api_url, headers=headers)
        response.raise_for_status()

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
