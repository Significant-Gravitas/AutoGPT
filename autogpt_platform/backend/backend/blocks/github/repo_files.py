import base64
from urllib.parse import quote

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
            default="main",
        )

    class Output(BlockSchemaOutput):
        text_content: str = SchemaField(
            description="Content of the file (decoded as UTF-8 text)"
        )
        raw_content: str = SchemaField(
            description="Raw base64-encoded content of the file"
        )
        size: int = SchemaField(description="The size of the file (in bytes)")
        error: str = SchemaField(description="Error message if reading the file failed")

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
                "branch": "main",
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
        content_url = (
            repo_url
            + f"/contents/{quote(file_path, safe='')}?ref={quote(branch, safe='')}"
        )
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
        try:
            content, size = await self.read_file(
                credentials,
                input_data.repo_url,
                input_data.file_path,
                input_data.branch,
            )
            yield "raw_content", content
            yield "text_content", base64.b64decode(content).decode("utf-8")
            yield "size", size
        except Exception as e:
            yield "error", str(e)


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
            description="Branch name to read from (defaults to main)",
            placeholder="branch_name",
            default="main",
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
                "branch": "main",
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
        contents_url = (
            repo_url
            + f"/contents/{quote(folder_path, safe='/')}?ref={quote(branch, safe='')}"
        )
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
        try:
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
        except Exception as e:
            yield "error", str(e)


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
        contents_url = repo_url + f"/contents/{quote(file_path, safe='/')}"
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
        contents_url = repo_url + f"/contents/{quote(file_path, safe='/')}"
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
            ge=1,
            le=100,
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
        tree_url = repo_url + f"/git/trees/{quote(branch, safe='')}"
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
