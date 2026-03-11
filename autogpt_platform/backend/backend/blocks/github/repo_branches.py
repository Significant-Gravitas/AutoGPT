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
from ._utils import github_repo_path


class GithubListBranchesBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        per_page: int = SchemaField(
            description="Number of branches to return per page (max 100)",
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
                "per_page": 30,
                "page": 1,
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
        credentials: GithubCredentials, repo_url: str, per_page: int, page: int
    ) -> list[Output.BranchItem]:
        api = get_api(credentials)
        branches_url = repo_url + "/branches"
        response = await api.get(
            branches_url, params={"per_page": str(per_page), "page": str(page)}
        )
        data = response.json()
        repo_path = github_repo_path(repo_url)
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
        try:
            branches = await self.list_branches(
                credentials,
                input_data.repo_url,
                input_data.per_page,
                input_data.page,
            )
            yield "branches", branches
            for branch in branches:
                yield "branch", branch
        except Exception as e:
            yield "error", str(e)


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
        ref_url = repo_url + f"/git/refs/heads/{quote(source_branch, safe='')}"
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
        try:
            status = await self.create_branch(
                credentials,
                input_data.repo_url,
                input_data.new_branch,
                input_data.source_branch,
            )
            yield "status", status
        except Exception as e:
            yield "error", str(e)


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
            is_sensitive_action=True,
        )

    @staticmethod
    async def delete_branch(
        credentials: GithubCredentials, repo_url: str, branch: str
    ) -> str:
        api = get_api(credentials)
        ref_url = repo_url + f"/git/refs/heads/{quote(branch, safe='')}"
        await api.delete(ref_url)
        return "Branch deleted successfully"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            status = await self.delete_branch(
                credentials,
                input_data.repo_url,
                input_data.branch,
            )
            yield "status", status
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
        safe_base = quote(base, safe="")
        safe_head = quote(head, safe="")
        compare_url = repo_url + f"/compare/{safe_base}...{safe_head}"
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
