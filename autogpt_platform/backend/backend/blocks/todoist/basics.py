#  These are some basic blocks for todoist
from backend.blocks.jina._auth import TEST_CREDENTIALS
from backend.blocks.todoist._auth import TEST_CREDENTIALS_INPUT, TodoistCredentials, TodoistCredentialsField, TodoistCredentialsInput
from todoist_api_python.api import TodoistAPI

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TodoistGetProjectsBlock(Block):
    """Gets projects for a Todoist user"""

    class Input(BlockSchema):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])

    class Output(BlockSchema):
        project_ids: list = SchemaField(description="All project IDs")
        project_names: list = SchemaField(description="All project names")
        data: list[dict] = SchemaField(description="Complete project data")
        error: str = SchemaField(description="Error message if request failed")

    def __init__(self):
        super().__init__(
            id="e141bce6-e0fa-4f3f-bbda-8a2bddc2c659",
            description="Gets all projects for a Todoist user",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistGetProjectsBlock.Input,
            output_schema=TodoistGetProjectsBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("project_ids", ["220474322"]),
                ("project_names", ["Inbox"]),
                ("data", [{"id": "220474322", "name": "Inbox"}]),
            ],
            test_mock={
                "get_projects": lambda *args, **kwargs: (
                    ["220474322"],
                    ["Inbox"],
                    [{"id": "220474322", "name": "Inbox"}],
                    None,
                )
            },
        )

    @staticmethod
    def get_projects(credentials: TodoistCredentials):
        try:
            api = TodoistAPI(credentials.access_token.get_secret_value())
            projects = api.get_projects()

            project_ids = []
            project_names = []
            project_data = []

            for project in projects:
                project_ids.append(project.id)
                project_names.append(project.name)
                project_data.append(project.__dict__)

            return project_ids, project_names, project_data, None

        except Exception as e:
            raise e

    def run(
        self,
        input_data: Input,
        *,
        credentials: TodoistCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            ids, names, data, error = self.get_projects(credentials)

            if ids:
                yield "project_ids", ids
            if names:
                yield "project_names", names
            if data:
                yield "data", data

        except Exception as e:
            yield "error", str(e)
