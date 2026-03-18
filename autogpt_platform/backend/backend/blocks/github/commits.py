import asyncio
from enum import StrEnum
from urllib.parse import quote

from typing_extensions import TypedDict

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.file import parse_data_uri, resolve_media_content
from backend.util.type import MediaFileType

from ._api import get_api
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentials,
    GithubCredentialsField,
    GithubCredentialsInput,
)
from ._utils import github_repo_path


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
            ge=1,
            le=100,
        )
        page: int = SchemaField(
            description="Page number for pagination",
            default=1,
            ge=1,
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
                "page": 1,
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
        page: int,
    ) -> list[Output.CommitItem]:
        api = get_api(credentials)
        commits_url = repo_url + "/commits"
        params = {"sha": branch, "per_page": str(per_page), "page": str(page)}
        response = await api.get(commits_url, params=params)
        data = response.json()
        repo_path = github_repo_path(repo_url)
        return [
            GithubListCommitsBlock.Output.CommitItem(
                sha=c["sha"],
                message=c["commit"]["message"],
                author=(c["commit"].get("author") or {}).get("name", "Unknown"),
                date=(c["commit"].get("author") or {}).get("date", ""),
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
                input_data.page,
            )
            yield "commits", commits
            for commit in commits:
                yield "commit", commit
        except Exception as e:
            yield "error", str(e)


class FileOperation(StrEnum):
    """File operations for GithubMultiFileCommitBlock.

    UPSERT creates or overwrites a file (the Git Trees API does not distinguish
    between creation and update — the blob is placed at the given path regardless
    of whether a file already exists there).

    DELETE removes a file from the tree.
    """

    UPSERT = "upsert"
    DELETE = "delete"


class FileOperationInput(TypedDict):
    path: str
    # MediaFileType is a str NewType — no runtime breakage for existing callers.
    content: MediaFileType
    operation: FileOperation


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
                "'path' (file path), 'content' (file content, ignored for delete), "
                "'operation' (upsert/delete)"
            ),
        )

    class Output(BlockSchemaOutput):
        sha: str = SchemaField(description="SHA of the new commit")
        url: str = SchemaField(description="URL of the new commit")
        error: str = SchemaField(description="Error message if the commit failed")

    def __init__(self):
        super().__init__(
            id="389eee51-a95e-4230-9bed-92167a327802",
            description=(
                "This block creates a single commit with multiple file "
                "upsert/delete operations using the Git Trees API."
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
                        "operation": "upsert",
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
        safe_branch = quote(branch, safe="")

        # 1. Get the latest commit SHA for the branch
        ref_url = repo_url + f"/git/refs/heads/{safe_branch}"
        response = await api.get(ref_url)
        ref_data = response.json()
        latest_commit_sha = ref_data["object"]["sha"]

        # 2. Get the tree SHA of the latest commit
        commit_url = repo_url + f"/git/commits/{latest_commit_sha}"
        response = await api.get(commit_url)
        commit_data = response.json()
        base_tree_sha = commit_data["tree"]["sha"]

        # 3. Build tree entries for each file operation (blobs created concurrently)
        async def _create_blob(content: str, encoding: str = "utf-8") -> str:
            blob_url = repo_url + "/git/blobs"
            blob_response = await api.post(
                blob_url,
                json={"content": content, "encoding": encoding},
            )
            return blob_response.json()["sha"]

        tree_entries: list[dict] = []
        upsert_files = []
        for file_op in files:
            path = file_op["path"]
            operation = FileOperation(file_op.get("operation", "upsert"))

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
                upsert_files.append((path, file_op.get("content", "")))

        # Create all blobs concurrently. Data URIs (from store_media_file)
        # are sent as base64 blobs to preserve binary content.
        if upsert_files:

            async def _make_blob(content: str) -> str:
                parsed = parse_data_uri(content)
                if parsed is not None:
                    _, b64_payload = parsed
                    return await _create_blob(b64_payload, encoding="base64")
                return await _create_blob(content)

            blob_shas = await asyncio.gather(
                *[_make_blob(content) for _, content in upsert_files]
            )
            for (path, _), blob_sha in zip(upsert_files, blob_shas):
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
        try:
            await api.patch(
                ref_url,
                json={"sha": new_commit_sha},
            )
        except Exception as e:
            raise RuntimeError(
                f"Commit {new_commit_sha} was created but failed to update "
                f"ref heads/{branch}: {e}. "
                f"You can recover by manually updating the branch to {new_commit_sha}."
            ) from e

        repo_path = github_repo_path(repo_url)
        commit_web_url = f"https://github.com/{repo_path}/commit/{new_commit_sha}"
        return new_commit_sha, commit_web_url

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        try:
            # Resolve media references (workspace://, data:, URLs) to data
            # URIs so _make_blob can send binary content correctly.
            resolved_files: list[FileOperationInput] = []
            for file_op in input_data.files:
                content = file_op.get("content", "")
                operation = FileOperation(file_op.get("operation", "upsert"))
                if operation != FileOperation.DELETE:
                    content = await resolve_media_content(
                        MediaFileType(content),
                        execution_context,
                        return_format="for_external_api",
                    )
                resolved_files.append(
                    FileOperationInput(
                        path=file_op["path"],
                        content=MediaFileType(content),
                        operation=operation,
                    )
                )

            sha, url = await self.multi_file_commit(
                credentials,
                input_data.repo_url,
                input_data.branch,
                input_data.commit_message,
                resolved_files,
            )
            yield "sha", sha
            yield "url", url
        except Exception as e:
            yield "error", str(e)
