from enum import Enum
from typing import Optional

from pydantic import BaseModel

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


# queued, in_progress, completed, waiting, requested, pending
class ChecksStatus(Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    WAITING = "waiting"
    REQUESTED = "requested"
    PENDING = "pending"


class ChecksConclusion(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    ACTION_REQUIRED = "action_required"
    SKIPPED = "skipped"


class GithubCreateCheckRunBlock(Block):
    """Block for creating a new check run on a GitHub repository."""

    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo:status")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        name: str = SchemaField(
            description="The name of the check run (e.g., 'code-coverage')",
        )
        head_sha: str = SchemaField(
            description="The SHA of the commit to check",
        )
        status: ChecksStatus = SchemaField(
            description="Current status of the check run",
            default=ChecksStatus.QUEUED,
        )
        conclusion: Optional[ChecksConclusion] = SchemaField(
            description="The final conclusion of the check (required if status is completed)",
            default=None,
        )
        details_url: str = SchemaField(
            description="The URL for the full details of the check",
            default="",
        )
        output_title: str = SchemaField(
            description="Title of the check run output",
            default="",
        )
        output_summary: str = SchemaField(
            description="Summary of the check run output",
            default="",
        )
        output_text: str = SchemaField(
            description="Detailed text of the check run output",
            default="",
        )

    class Output(BlockSchema):
        class CheckRunResult(BaseModel):
            id: int
            html_url: str
            status: str

        check_run: CheckRunResult = SchemaField(
            description="Details of the created check run"
        )
        error: str = SchemaField(
            description="Error message if check run creation failed"
        )

    def __init__(self):
        super().__init__(
            id="2f45e89a-3b7d-4f22-b89e-6c4f5c7e1234",
            description="Creates a new check run for a specific commit in a GitHub repository",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCreateCheckRunBlock.Input,
            output_schema=GithubCreateCheckRunBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "name": "test-check",
                "head_sha": "ce587453ced02b1526dfb4cb910479d431683101",
                "status": ChecksStatus.COMPLETED.value,
                "conclusion": ChecksConclusion.SUCCESS.value,
                "output_title": "Test Results",
                "output_summary": "All tests passed",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            # requires a github app not available to oauth in our current system
            disabled=True,
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "check_run",
                    {
                        "id": 4,
                        "html_url": "https://github.com/owner/repo/runs/4",
                        "status": "completed",
                    },
                ),
            ],
            test_mock={
                "create_check_run": lambda *args, **kwargs: {
                    "id": 4,
                    "html_url": "https://github.com/owner/repo/runs/4",
                    "status": "completed",
                }
            },
        )

    @staticmethod
    def create_check_run(
        credentials: GithubCredentials,
        repo_url: str,
        name: str,
        head_sha: str,
        status: ChecksStatus,
        conclusion: Optional[ChecksConclusion] = None,
        details_url: Optional[str] = None,
        output_title: Optional[str] = None,
        output_summary: Optional[str] = None,
        output_text: Optional[str] = None,
    ) -> dict:
        api = get_api(credentials)

        class CheckRunData(BaseModel):
            name: str
            head_sha: str
            status: str
            conclusion: Optional[str] = None
            details_url: Optional[str] = None
            output: Optional[dict[str, str]] = None

        data = CheckRunData(
            name=name,
            head_sha=head_sha,
            status=status.value,
        )

        if conclusion:
            data.conclusion = conclusion.value

        if details_url:
            data.details_url = details_url

        if output_title or output_summary or output_text:
            output_data = {
                "title": output_title or "",
                "summary": output_summary or "",
                "text": output_text or "",
            }
            data.output = output_data

        check_runs_url = f"{repo_url}/check-runs"
        response = api.post(
            check_runs_url, data=data.model_dump_json(exclude_none=True)
        )
        result = response.json()

        return {
            "id": result["id"],
            "html_url": result["html_url"],
            "status": result["status"],
        }

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            result = self.create_check_run(
                credentials=credentials,
                repo_url=input_data.repo_url,
                name=input_data.name,
                head_sha=input_data.head_sha,
                status=input_data.status,
                conclusion=input_data.conclusion,
                details_url=input_data.details_url,
                output_title=input_data.output_title,
                output_summary=input_data.output_summary,
                output_text=input_data.output_text,
            )
            yield "check_run", result
        except Exception as e:
            yield "error", str(e)


class GithubUpdateCheckRunBlock(Block):
    """Block for updating an existing check run on a GitHub repository."""

    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo:status")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        check_run_id: int = SchemaField(
            description="The ID of the check run to update",
        )
        status: ChecksStatus = SchemaField(
            description="New status of the check run",
        )
        conclusion: ChecksConclusion = SchemaField(
            description="The final conclusion of the check (required if status is completed)",
        )
        output_title: Optional[str] = SchemaField(
            description="New title of the check run output",
            default=None,
        )
        output_summary: Optional[str] = SchemaField(
            description="New summary of the check run output",
            default=None,
        )
        output_text: Optional[str] = SchemaField(
            description="New detailed text of the check run output",
            default=None,
        )

    class Output(BlockSchema):
        class CheckRunResult(BaseModel):
            id: int
            html_url: str
            status: str
            conclusion: Optional[str]

        check_run: CheckRunResult = SchemaField(
            description="Details of the updated check run"
        )
        error: str = SchemaField(description="Error message if check run update failed")

    def __init__(self):
        super().__init__(
            id="8a23c567-9d01-4e56-b789-0c12d3e45678",  # Generated UUID
            description="Updates an existing check run in a GitHub repository",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubUpdateCheckRunBlock.Input,
            output_schema=GithubUpdateCheckRunBlock.Output,
            # requires a github app not available to oauth in our current system
            disabled=True,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "check_run_id": 4,
                "status": ChecksStatus.COMPLETED.value,
                "conclusion": ChecksConclusion.SUCCESS.value,
                "output_title": "Updated Results",
                "output_summary": "All tests passed after retry",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "check_run",
                    {
                        "id": 4,
                        "html_url": "https://github.com/owner/repo/runs/4",
                        "status": "completed",
                        "conclusion": "success",
                    },
                ),
            ],
            test_mock={
                "update_check_run": lambda *args, **kwargs: {
                    "id": 4,
                    "html_url": "https://github.com/owner/repo/runs/4",
                    "status": "completed",
                    "conclusion": "success",
                }
            },
        )

    @staticmethod
    def update_check_run(
        credentials: GithubCredentials,
        repo_url: str,
        check_run_id: int,
        status: ChecksStatus,
        conclusion: Optional[ChecksConclusion] = None,
        output_title: Optional[str] = None,
        output_summary: Optional[str] = None,
        output_text: Optional[str] = None,
    ) -> dict:
        api = get_api(credentials)

        class UpdateCheckRunData(BaseModel):
            status: str
            conclusion: Optional[str] = None
            output: Optional[dict[str, str]] = None

        data = UpdateCheckRunData(
            status=status.value,
        )

        if conclusion:
            data.conclusion = conclusion.value

        if output_title or output_summary or output_text:
            output_data = {
                "title": output_title or "",
                "summary": output_summary or "",
                "text": output_text or "",
            }
            data.output = output_data

        check_run_url = f"{repo_url}/check-runs/{check_run_id}"
        response = api.patch(
            check_run_url, data=data.model_dump_json(exclude_none=True)
        )
        result = response.json()

        return {
            "id": result["id"],
            "html_url": result["html_url"],
            "status": result["status"],
            "conclusion": result.get("conclusion"),
        }

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            result = self.update_check_run(
                credentials=credentials,
                repo_url=input_data.repo_url,
                check_run_id=input_data.check_run_id,
                status=input_data.status,
                conclusion=input_data.conclusion,
                output_title=input_data.output_title,
                output_summary=input_data.output_summary,
                output_text=input_data.output_text,
            )
            yield "check_run", result
        except Exception as e:
            yield "error", str(e)
