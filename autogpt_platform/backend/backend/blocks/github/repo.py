import base64
from typing_extensions import TypedDict

import requests

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
            description="Tags with their name and file tree browser URL"
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
        try:
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
        except Exception as e:
            return [{"name": "Error", "url": f"Failed to list tags: {str(e)}"}]

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
        if any("Failed" in tag["url"] for tag in tags):
            yield "error", tags[0]["url"]
        else:
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
            description="Branches with their name and file tree browser URL"
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
        try:
            api_url = (
                repo_url.replace("github.com", "api.github.com/repos") + "/branches"
            )
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            branches: list[GithubListBranchesBlock.Output.BranchItem] = [
                {"name": branch["name"], "url": branch["commit"]["url"]}
                for branch in data
            ]

            return branches
        except Exception as e:
            return [{"name": "Error", "url": f"Failed to list branches: {str(e)}"}]

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
        if any("Failed" in branch["url"] for branch in branches):
            yield "error", branches[0]["url"]
        else:
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
            description="Discussions with their title and URL"
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
        try:
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
        except Exception as e:
            return [{"title": "Error", "url": f"Failed to list discussions: {str(e)}"}]

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
        if any("Failed" in discussion["url"] for discussion in discussions):
            yield "error", discussions[0]["url"]
        else:
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
            description="Releases with their name and file tree browser URL"
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
        try:
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
                {"name": release["name"], "url": release["html_url"]}
                for release in data
            ]

            return releases
        except Exception as e:
            return [{"name": "Error", "url": f"Failed to list releases: {str(e)}"}]

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
        if any("Failed" in release["url"] for release in releases):
            yield "error", releases[0]["url"]
        else:
            yield from (("release", release) for release in releases)


class GithubReadCodeownersFileBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        codeowners_content: str = SchemaField(
            description="Content of the CODEOWNERS file"
        )
        error: str = SchemaField(description="Error message if the file reading failed")

    def __init__(self):
        super().__init__(
            id="ea64bb61-30c0-4693-9273-04081a8f955b",
            description="This block reads the CODEOWNERS file from the master branch of a specified GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadCodeownersFileBlock.Input,
            output_schema=GithubReadCodeownersFileBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("codeowners_content", "# CODEOWNERS content")],
            test_mock={
                "read_codeowners": lambda *args, **kwargs: "# CODEOWNERS content"
            },
        )

    @staticmethod
    def read_codeowners(credentials: GithubCredentials, repo_url: str) -> str:
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            api_url = f"https://api.github.com/repos/{repo_path}/contents/.github/CODEOWNERS?ref=master"
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            content = response.json()
            return base64.b64decode(content["content"]).decode("utf-8")
        except requests.exceptions.HTTPError as http_err:
            return f"Failed to read CODEOWNERS file: {str(http_err)}"
        except Exception as e:
            return f"Failed to read CODEOWNERS file: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        content = self.read_codeowners(
            credentials,
            input_data.repo_url,
        )
        if "Failed" not in content:
            yield "codeowners_content", content
        else:
            yield "error", content


class GithubReadFileFromMasterBlock(Block):
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

    class Output(BlockSchema):
        file_content: str = SchemaField(
            description="Content of the file from the master branch"
        )
        error: str = SchemaField(description="Error message if the file reading failed")

    def __init__(self):
        super().__init__(
            id="87ce6c27-5752-4bbc-8e26-6da40a3dcfd3",
            description="This block reads the content of a specified file from the master branch of a GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadFileFromMasterBlock.Input,
            output_schema=GithubReadFileFromMasterBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "file_path": "path/to/file",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("file_content", "File content")],
            test_mock={"read_file": lambda *args, **kwargs: "File content"},
        )

    @staticmethod
    def read_file(credentials: GithubCredentials, repo_url: str, file_path: str) -> str:
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            api_url = f"https://api.github.com/repos/{repo_path}/contents/{file_path}?ref=master"
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            content = response.json()
            return base64.b64decode(content["content"]).decode("utf-8")
        except requests.exceptions.HTTPError as http_err:
            return f"Failed to read file: {str(http_err)}"
        except Exception as e:
            return f"Failed to read file: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        content = self.read_file(
            credentials,
            input_data.repo_url,
            input_data.file_path,
        )
        if "Failed" not in content:
            yield "file_content", content
        else:
            yield "error", content


class GithubReadFileFolderRepoBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        path: str = SchemaField(
            description="Path to the file/folder in the repository",
            placeholder="path/to/file_or_folder",
        )
        branch: str = SchemaField(
            description="Branch name to read from",
            placeholder="branch_name",
        )

    class Output(BlockSchema):
        content: str = SchemaField(
            description="Content of the file/folder/repo from the specified branch"
        )
        error: str = SchemaField(description="Error message if the reading failed")

    def __init__(self):
        super().__init__(
            id="1355f863-2db3-4d75-9fba-f91e8a8ca400",
            description="This block reads the content of a specified file, folder, or repository from a specified branch.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadFileFolderRepoBlock.Input,
            output_schema=GithubReadFileFolderRepoBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "path": "path/to/file_or_folder",
                "branch": "branch_name",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("content", "File or folder content")],
            test_mock={
                "read_content": lambda *args, **kwargs: "File or folder content"
            },
        )

    @staticmethod
    def read_content(
        credentials: GithubCredentials, repo_url: str, path: str, branch: str
    ) -> str:
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            api_url = (
                f"https://api.github.com/repos/{repo_path}/contents/{path}?ref={branch}"
            )
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            content = response.json()
            if "content" in content:
                return base64.b64decode(content["content"]).decode("utf-8")
            else:
                return content  # Return the directory content as JSON

        except requests.exceptions.HTTPError as http_err:
            return f"Failed to read content: {str(http_err)}"
        except Exception as e:
            return f"Failed to read content: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        content = self.read_content(
            credentials,
            input_data.repo_url,
            input_data.path,
            input_data.branch,
        )
        if "Failed" not in content:
            yield "content", content
        else:
            yield "error", content


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
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            ref_api_url = f"https://api.github.com/repos/{repo_path}/git/refs/heads/{source_branch}"
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
        except requests.exceptions.HTTPError as http_err:
            return f"Failed to create branch: {str(http_err)}"
        except Exception as e:
            return f"Failed to create branch: {str(e)}"

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
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status


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
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            api_url = (
                f"https://api.github.com/repos/{repo_path}/git/refs/heads/{branch}"
            )
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.delete(api_url, headers=headers)
            response.raise_for_status()

            return "Branch deleted successfully"
        except requests.exceptions.HTTPError as http_err:
            return f"Failed to delete branch: {str(http_err)}"
        except Exception as e:
            return f"Failed to delete branch: {str(e)}"

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
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status
