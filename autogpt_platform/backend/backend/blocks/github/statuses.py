from enum import Enum
from typing import Optional

from pydantic import BaseModel

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

from ._api import get_api
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubFineGrainedAPICredentials,
    GithubFineGrainedAPICredentialsField,
    GithubFineGrainedAPICredentialsInput,
)


class StatusState(Enum):
    ERROR = "error"
    FAILURE = "failure"
    PENDING = "pending"
    SUCCESS = "success"


class GithubCreateStatusBlock(Block):
    """Block for creating a commit status on a GitHub repository."""

    class Input(BlockSchema):
        credentials: GithubFineGrainedAPICredentialsInput = (
            GithubFineGrainedAPICredentialsField("repo:status")
        )
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        sha: str = SchemaField(
            description="The SHA of the commit to set status for",
        )
        state: StatusState = SchemaField(
            description="The state of the status (error, failure, pending, success)",
        )
        target_url: Optional[str] = SchemaField(
            description="URL with additional details about this status",
            default=None,
        )
        description: Optional[str] = SchemaField(
            description="Short description of the status",
            default=None,
        )
        check_name: Optional[str] = SchemaField(
            description="Label to differentiate this status from others",
            default="AutoGPT Platform Checks",
            advanced=False,
        )

    class Output(BlockSchema):
        class StatusResult(BaseModel):
            id: int
            url: str
            state: str
            context: str
            description: Optional[str]
            target_url: Optional[str]
            created_at: str
            updated_at: str

        status: StatusResult = SchemaField(description="Details of the created status")
        error: str = SchemaField(description="Error message if status creation failed")

    def __init__(self):
        super().__init__(
            id="3d67f123-a4b5-4c89-9d01-2e34f5c67890",  # Generated UUID
            description="Creates a new commit status in a GitHub repository",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCreateStatusBlock.Input,
            output_schema=GithubCreateStatusBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "sha": "ce587453ced02b1526dfb4cb910479d431683101",
                "state": StatusState.SUCCESS.value,
                "target_url": "https://example.com/build/status",
                "description": "The build succeeded!",
                "check_name": "continuous-integration/jenkins",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "status",
                    {
                        "id": 1234567890,
                        "url": "https://api.github.com/repos/owner/repo/statuses/ce587453ced02b1526dfb4cb910479d431683101",
                        "state": "success",
                        "context": "continuous-integration/jenkins",
                        "description": "The build succeeded!",
                        "target_url": "https://example.com/build/status",
                        "created_at": "2024-01-21T10:00:00Z",
                        "updated_at": "2024-01-21T10:00:00Z",
                    },
                ),
            ],
            test_mock={
                "create_status": lambda *args, **kwargs: {
                    "id": 1234567890,
                    "url": "https://api.github.com/repos/owner/repo/statuses/ce587453ced02b1526dfb4cb910479d431683101",
                    "state": "success",
                    "context": "continuous-integration/jenkins",
                    "description": "The build succeeded!",
                    "target_url": "https://example.com/build/status",
                    "created_at": "2024-01-21T10:00:00Z",
                    "updated_at": "2024-01-21T10:00:00Z",
                }
            },
        )

    @staticmethod
    def create_status(
        credentials: GithubFineGrainedAPICredentials,
        repo_url: str,
        sha: str,
        state: StatusState,
        target_url: Optional[str] = None,
        description: Optional[str] = None,
        context: str = "default",
    ) -> dict:
        api = get_api(credentials)

        class StatusData(BaseModel):
            state: str
            target_url: Optional[str] = None
            description: Optional[str] = None
            context: str

        data = StatusData(
            state=state.value,
            context=context,
        )

        if target_url:
            data.target_url = target_url

        if description:
            data.description = description

        status_url = f"{repo_url}/statuses/{sha}"
        response = api.post(status_url, data=data.model_dump_json(exclude_none=True))
        result = response.json()

        return {
            "id": result["id"],
            "url": result["url"],
            "state": result["state"],
            "context": result["context"],
            "description": result.get("description"),
            "target_url": result.get("target_url"),
            "created_at": result["created_at"],
            "updated_at": result["updated_at"],
        }

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubFineGrainedAPICredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            result = self.create_status(
                credentials=credentials,
                repo_url=input_data.repo_url,
                sha=input_data.sha,
                state=input_data.state,
                target_url=input_data.target_url,
                description=input_data.description,
                context=input_data.check_name or "AutoGPT Platform Checks",
            )
            yield "status", result
        except Exception as e:
            yield "error", str(e)
