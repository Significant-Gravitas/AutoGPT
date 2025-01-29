from typing_extensions import Optional
from todoist_api_python.api import TodoistAPI

from backend.blocks.todoist._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    TodoistCredentials,
    TodoistCredentialsInput,
    TodoistCredentialsField,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TodoistCreateTaskBlock(Block):
    """Creates a new task in a Todoist project"""

    class Input(BlockSchema):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        content: str = SchemaField(description="Task content")
        description: Optional[str] = SchemaField(description="Task description", default=None)
        project_id: Optional[str] = SchemaField(description="Project ID this task should belong to", default=None)
        section_id: Optional[str] = SchemaField(description="Section ID this task should belong to", default=None)
        parent_id: Optional[str] = SchemaField(description="Parent task ID", default=None)
        order: Optional[int] = SchemaField(description="Optional order among other tasks", default=None)
        labels: Optional[list[str]] = SchemaField(description="Task labels", default=None)
        priority: Optional[int] = SchemaField(description="Task priority (1-4)", default=None)
        due_string: Optional[str] = SchemaField(description="Human defined task due date", default=None)
        due_date: Optional[str] = SchemaField(description="Due date in YYYY-MM-DD format", default=None)
        due_datetime: Optional[str] = SchemaField(description="Due date and time in RFC3339 format", default=None)
        due_lang: Optional[str] = SchemaField(description="2-letter language code for due_string", default=None)
        assignee_id: Optional[str] = SchemaField(description="Responsible user ID", default=None)
        duration: Optional[int] = SchemaField(description="Task duration amount", default=None)
        duration_unit: Optional[str] = SchemaField(description="Task duration unit (minute/day)", default=None)

    class Output(BlockSchema):
        project_id: str = SchemaField(description="Project ID")
        id: str = SchemaField(description="ID of created task")
        url: str = SchemaField(description="Task URL")
        is_completed: bool = SchemaField(description="Task completion status")

    def __init__(self):
        super().__init__(
            id="fde4f458-de14-11ef-bf0c-32d3674e8b7e",
            description="Creates a new task in a Todoist project",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistCreateTaskBlock.Input,
            output_schema=TodoistCreateTaskBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "content": "Buy groceries",
                "project_id": "2203306141",
                "priority": 4
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("project_id", "2203306141"),
                ("id", "2995104339"),
                ("url", "https://todoist.com/showTask?id=2995104339"),
                ("is_completed", False)
            ],
            test_mock={
                "create_task": lambda *args, **kwargs: (
                    {
                        "project_id": "2203306141",
                        "id": "2995104339",
                        "url": "https://todoist.com/showTask?id=2995104339",
                        "is_completed": False
                    },
                    None,
                )
            },
        )

    @staticmethod
    def create_task(credentials: TodoistCredentials, content: str, **kwargs):
        try:
            api = TodoistAPI(credentials.access_token.get_secret_value())
            task = api.add_task(content=content, **kwargs)
            return task.__dict__

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
            task_args = {
                "description": input_data.description,
                "project_id": input_data.project_id,
                "section_id": input_data.section_id,
                "parent_id": input_data.parent_id,
                "order": input_data.order,
                "labels": input_data.labels,
                "priority": input_data.priority,
                "due_string": input_data.due_string,
                "due_date": input_data.due_date,
                "due_datetime": input_data.due_datetime,
                "due_lang": input_data.due_lang,
                "assignee_id": input_data.assignee_id,
                "duration": input_data.duration,
                "duration_unit": input_data.duration_unit
            }

            task_data = self.create_task(
                credentials,
                input_data.content,
                **{k:v for k,v in task_args.items() if v is not None}
            )

            if task_data:
                yield "project_id", task_data["project_id"]
                yield "id", task_data["id"]
                yield "url", task_data["url"]
                yield "is_completed", False

        except Exception as e:
            yield "error", str(e)

class TodoistGetTasksBlock(Block):
    """Get active tasks from Todoist"""

    class Input(BlockSchema):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        project_id: Optional[str] = SchemaField(description="Filter tasks by project ID", default=None)
        section_id: Optional[str] = SchemaField(description="Filter tasks by section ID", default=None)
        label: Optional[str] = SchemaField(description="Filter tasks by label name", default=None)
        filter: Optional[str] = SchemaField(description="Filter by any supported filter", default=None)
        lang: Optional[str] = SchemaField(description="IETF language tag for filter language", default=None)
        ids: Optional[list[str]] = SchemaField(description="List of task IDs to retrieve", default=None)

    class Output(BlockSchema):
        project_id: str = SchemaField(description="Project ID containing the task")
        id: str = SchemaField(description="Task ID")
        url: str = SchemaField(description="Task URL")
        is_completed: bool = SchemaField(description="Task completion status")

    def __init__(self):
        super().__init__(
            id="0b706e86-de15-11ef-a113-32d3674e8b7e",
            description="Get active tasks from Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistGetTasksBlock.Input,
            output_schema=TodoistGetTasksBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": "2203306141"
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("project_id", "2203306141"),
                ("id", "2995104339"),
                ("url", "https://todoist.com/showTask?id=2995104339"),
                ("is_completed", False)
            ],
            test_mock={
                "get_tasks": lambda *args, **kwargs: (
                    [{
                        "project_id": "2203306141",
                        "id": "2995104339",
                        "url": "https://todoist.com/showTask?id=2995104339",
                        "is_completed": False
                    }],
                    None
                )
            }
        )

    @staticmethod
    def get_tasks(credentials: TodoistCredentials, **kwargs):
        try:
            api = TodoistAPI(credentials.access_token.get_secret_value())
            tasks = api.get_tasks(**kwargs)
            return [task.__dict__ for task in tasks]
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
            task_filters = {
                "project_id": input_data.project_id,
                "section_id": input_data.section_id,
                "label": input_data.label,
                "filter": input_data.filter,
                "lang": input_data.lang,
                "ids": input_data.ids
            }

            tasks = self.get_tasks(
                credentials,
                **{k:v for k,v in task_filters.items() if v is not None}
            )

            for task in tasks:
                yield "project_id", task["project_id"]
                yield "id", task["id"]
                yield "url", task["url"]
                yield "is_completed", task["is_completed"]

        except Exception as e:
            yield "error", str(e)

class TodoistGetTaskBlock(Block):
    """Get an active task from Todoist"""

    class Input(BlockSchema):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        task_id: str = SchemaField(description="Task ID to retrieve")

    class Output(BlockSchema):
        project_id: str = SchemaField(description="Project ID containing the task")
        url: str = SchemaField(description="Task URL")
        complete_data: dict = SchemaField(description="Complete task data as dictionary")

    def __init__(self):
        super().__init__(
            id="16d7dc8c-de15-11ef-8ace-32d3674e8b7e",
            description="Get an active task from Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistGetTaskBlock.Input,
            output_schema=TodoistGetTaskBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "task_id": "2995104339"
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("project_id", "2203306141"),
                ("url", "https://todoist.com/showTask?id=2995104339"),
                ("complete_data", {
                    "id": "2995104339",
                    "project_id": "2203306141",
                    "url": "https://todoist.com/showTask?id=2995104339"
                })
            ],
            test_mock={
                "get_task": lambda *args, **kwargs: (
                    {
                        "project_id": "2203306141",
                        "id": "2995104339",
                        "url": "https://todoist.com/showTask?id=2995104339"
                    },
                    None,
                )
            },
        )

    @staticmethod
    def get_task(credentials: TodoistCredentials, task_id: str):
        try:
            api = TodoistAPI(credentials.access_token.get_secret_value())
            task = api.get_task(task_id=task_id)
            return task.__dict__
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
            task_data = self.get_task(credentials, input_data.task_id)

            if task_data:
                yield "project_id", task_data["project_id"]
                yield "url", task_data["url"]
                yield "complete_data", task_data

        except Exception as e:
            yield "error", str(e)

class TodoistUpdateTaskBlock(Block):
    """Updates an existing task in Todoist"""

    class Input(BlockSchema):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        task_id: str = SchemaField(description="Task ID to update")
        content: Optional[str] = SchemaField(description="Task content", default=None)
        description: Optional[str] = SchemaField(description="Task description", default=None)
        labels: Optional[list[str]] = SchemaField(description="Task labels", default=None)
        priority: Optional[int] = SchemaField(description="Task priority (1-4)", default=None)
        due_string: Optional[str] = SchemaField(description="Human defined task due date", default=None)
        due_date: Optional[str] = SchemaField(description="Due date in YYYY-MM-DD format", default=None)
        due_datetime: Optional[str] = SchemaField(description="Due date and time in RFC3339 format", default=None)
        due_lang: Optional[str] = SchemaField(description="2-letter language code for due_string", default=None)
        assignee_id: Optional[str] = SchemaField(description="Responsible user ID", default=None)
        duration: Optional[int] = SchemaField(description="Task duration amount", default=None)
        duration_unit: Optional[str] = SchemaField(description="Task duration unit (minute/day)", default=None)
        deadline_date: Optional[str] = SchemaField(description="Deadline date in YYYY-MM-DD format", default=None)
        deadline_lang: Optional[str] = SchemaField(description="2-letter language code for deadline", default=None)

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the update was successful")
        error: str = SchemaField(description="Error message if request failed")

    def __init__(self):
        super().__init__(
            id="1eee6d32-de15-11ef-a2ff-32d3674e8b7e",
            description="Updates an existing task in Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistUpdateTaskBlock.Input,
            output_schema=TodoistUpdateTaskBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "task_id": "2995104339",
                "content": "Buy Coffee"
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
                ("error", None)
            ],
            test_mock={
                "update_task": lambda *args, **kwargs: (True, None)
            },
        )

    @staticmethod
    def update_task(credentials: TodoistCredentials, task_id: str, **kwargs):
        try:
            api = TodoistAPI(credentials.access_token.get_secret_value())
            is_success = api.update_task(task_id=task_id, **kwargs)
            return is_success
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
            task_updates = {
                "content": input_data.content,
                "description": input_data.description,
                "labels": input_data.labels,
                "priority": input_data.priority,
                "due_string": input_data.due_string,
                "due_date": input_data.due_date,
                "due_datetime": input_data.due_datetime,
                "due_lang": input_data.due_lang,
                "assignee_id": input_data.assignee_id,
                "duration": input_data.duration,
                "duration_unit": input_data.duration_unit,
                "deadline_date": input_data.deadline_date,
                "deadline_lang": input_data.deadline_lang
            }

            is_success = self.update_task(
                credentials,
                input_data.task_id,
                **{k:v for k,v in task_updates.items() if v is not None}
            )

            yield "success", is_success
            yield "error", None

        except Exception as e:
            yield "success", False
            yield "error", str(e)

class TodoistCloseTaskBlock(Block):
    """Closes a task in Todoist"""

    class Input(BlockSchema):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        task_id: str = SchemaField(description="Task ID to close")

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the task was successfully closed")
        error: str = SchemaField(description="Error message if request failed")

    def __init__(self):
        super().__init__(
            id="29fac798-de15-11ef-b839-32d3674e8b7e",
            description="Closes a task in Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistCloseTaskBlock.Input,
            output_schema=TodoistCloseTaskBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "task_id": "2995104339"
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
                ("error", None)
            ],
            test_mock={
                "close_task": lambda *args, **kwargs: (True, None)
            },
        )

    @staticmethod
    def close_task(credentials: TodoistCredentials, task_id: str):
        try:
            api = TodoistAPI(credentials.access_token.get_secret_value())
            is_success = api.close_task(task_id=task_id)
            return is_success
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
            is_success = self.close_task(credentials, input_data.task_id)
            yield "success", is_success
            yield "error", None

        except Exception as e:
            yield "success", False
            yield "error", str(e)

class TodoistReopenTaskBlock(Block):
    """Reopens a task in Todoist"""

    class Input(BlockSchema):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        task_id: str = SchemaField(description="Task ID to reopen")

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the task was successfully reopened")
        error: str = SchemaField(description="Error message if request failed")

    def __init__(self):
        super().__init__(
            id="2e6bf6f8-de15-11ef-ae7c-32d3674e8b7e",
            description="Reopens a task in Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistReopenTaskBlock.Input,
            output_schema=TodoistReopenTaskBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "task_id": "2995104339"
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
                ("error", None)
            ],
            test_mock={
                "reopen_task": lambda *args, **kwargs: (True, None)
            },
        )

    @staticmethod
    def reopen_task(credentials: TodoistCredentials, task_id: str):
        try:
            api = TodoistAPI(credentials.access_token.get_secret_value())
            is_success = api.reopen_task(task_id=task_id)
            return is_success
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
            is_success = self.reopen_task(credentials, input_data.task_id)
            yield "success", is_success
            yield "error", None

        except Exception as e:
            yield "success", False
            yield "error", str(e)

class TodoistDeleteTaskBlock(Block):
    """Deletes a task in Todoist"""

    class Input(BlockSchema):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        task_id: str = SchemaField(description="Task ID to delete")

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the task was successfully deleted")
        error: str = SchemaField(description="Error message if request failed")

    def __init__(self):
        super().__init__(
            id="33c29ada-de15-11ef-bcbb-32d3674e8b7e",
            description="Deletes a task in Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistDeleteTaskBlock.Input,
            output_schema=TodoistDeleteTaskBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "task_id": "2995104339"
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
                ("error", None)
            ],
            test_mock={
                "delete_task": lambda *args, **kwargs: (True, None)
            },
        )

    @staticmethod
    def delete_task(credentials: TodoistCredentials, task_id: str):
        try:
            api = TodoistAPI(credentials.access_token.get_secret_value())
            is_success = api.delete_task(task_id=task_id)
            return is_success
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
            is_success = self.delete_task(credentials, input_data.task_id)
            yield "success", is_success
            yield "error", None

        except Exception as e:
            yield "success", False
            yield "error", str(e)
