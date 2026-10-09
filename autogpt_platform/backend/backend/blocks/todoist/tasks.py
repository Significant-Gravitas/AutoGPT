from datetime import date, datetime

from typing_extensions import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.todoist._api import (
    flatten_pages,
    get_api,
    parse_duration_unit,
    task_to_dict,
)
from backend.blocks.todoist._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    TODOIST_OAUTH_IS_CONFIGURED,
    TodoistCredentials,
    TodoistCredentialsField,
    TodoistCredentialsInput,
)
from backend.data.model import SchemaField


class TodoistCreateTaskBlock(Block):
    """Creates a new task in a Todoist project"""

    class Input(BlockSchemaInput):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        content: str = SchemaField(description="Task content", advanced=False)
        description: Optional[str] = SchemaField(
            description="Task description", default=None, advanced=False
        )
        project_id: Optional[str] = SchemaField(
            description="Project ID this task should belong to",
            default=None,
            advanced=False,
        )
        section_id: Optional[str] = SchemaField(
            description="Section ID this task should belong to",
            default=None,
            advanced=False,
        )
        parent_id: Optional[str] = SchemaField(
            description="Parent task ID", default=None, advanced=True
        )
        order: Optional[int] = SchemaField(
            description="Optional order among other tasks,[Non-zero integer value used by clients to sort tasks under the same parent]",
            default=None,
            advanced=True,
        )
        labels: Optional[list[str]] = SchemaField(
            description="Task labels", default=None, advanced=True
        )
        priority: Optional[int] = SchemaField(
            description="Task priority from 1 (normal) to 4 (urgent)",
            default=None,
            advanced=True,
        )
        due_date: Optional[datetime] = SchemaField(
            description="Due date in YYYY-MM-DD format", advanced=True, default=None
        )
        deadline_date: Optional[datetime] = SchemaField(
            description="Specific date in YYYY-MM-DD format relative to user's timezone",
            default=None,
            advanced=True,
        )
        assignee_id: Optional[str] = SchemaField(
            description="Responsible user ID", default=None, advanced=True
        )
        duration_unit: Optional[str] = SchemaField(
            description="Task duration unit (minute/day)", default=None, advanced=True
        )
        duration: Optional[int] = SchemaField(
            description="Task duration amount, You need to selecct the duration unit first",
            depends_on=["duration_unit"],
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        id: str = SchemaField(description="Task ID")
        url: str = SchemaField(description="Task URL")
        complete_data: dict = SchemaField(
            description="Complete task data as dictionary"
        )

    def __init__(self):
        super().__init__(
            id="fde4f458-de14-11ef-bf0c-32d3674e8b7e",
            description="Creates a new task in a Todoist project",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistCreateTaskBlock.Input,
            output_schema=TodoistCreateTaskBlock.Output,
            disabled=not TODOIST_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "content": "Buy groceries",
                "project_id": "2203306141",
                "priority": 4,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("id", "2995104339"),
                ("url", "https://todoist.com/showTask?id=2995104339"),
                (
                    "complete_data",
                    {
                        "id": "2995104339",
                        "project_id": "2203306141",
                        "url": "https://todoist.com/showTask?id=2995104339",
                    },
                ),
            ],
            test_mock={
                "create_task": lambda *args, **kwargs: (
                    "2995104339",
                    "https://todoist.com/showTask?id=2995104339",
                    {
                        "id": "2995104339",
                        "project_id": "2203306141",
                        "url": "https://todoist.com/showTask?id=2995104339",
                    },
                )
            },
        )

    @staticmethod
    def create_task(
        credentials: TodoistCredentials,
        content: str,
        *,
        description: Optional[str] = None,
        project_id: Optional[str] = None,
        section_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        order: Optional[int] = None,
        labels: Optional[list[str]] = None,
        priority: Optional[int] = None,
        due_date: Optional[date] = None,
        deadline_date: Optional[date] = None,
        assignee_id: Optional[str] = None,
        duration: Optional[int] = None,
        duration_unit: Optional[str] = None,
    ):
        with get_api(credentials) as api:
            task = api.add_task(
                content,
                description=description,
                project_id=project_id,
                section_id=section_id,
                parent_id=parent_id,
                order=order,
                labels=labels,
                priority=priority,
                due_date=due_date,
                deadline_date=deadline_date,
                assignee_id=assignee_id,
                duration=duration,
                duration_unit=parse_duration_unit(duration_unit),
            )
            return task.id, task.url, task_to_dict(task)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: TodoistCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            id, url, complete_data = self.create_task(
                credentials,
                input_data.content,
                description=input_data.description,
                project_id=input_data.project_id,
                section_id=input_data.section_id,
                parent_id=input_data.parent_id,
                order=input_data.order,
                labels=input_data.labels,
                priority=input_data.priority,
                due_date=input_data.due_date.date() if input_data.due_date else None,
                deadline_date=(
                    input_data.deadline_date.date()
                    if input_data.deadline_date
                    else None
                ),
                assignee_id=input_data.assignee_id,
                duration=input_data.duration,
                duration_unit=input_data.duration_unit,
            )

            yield "id", id
            yield "url", url
            yield "complete_data", complete_data

        except Exception as e:
            yield "error", str(e)


class TodoistGetTasksBlock(Block):
    """Get active tasks from Todoist"""

    class Input(BlockSchemaInput):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        project_id: Optional[str] = SchemaField(
            description="Filter tasks by project ID", default=None, advanced=False
        )
        section_id: Optional[str] = SchemaField(
            description="Filter tasks by section ID", default=None, advanced=True
        )
        label: Optional[str] = SchemaField(
            description="Filter tasks by label name", default=None, advanced=True
        )
        filter: Optional[str] = SchemaField(
            description="Filter by any supported filter, You can see How to use filters or create one of your one here - https://todoist.com/help/articles/introduction-to-filters-V98wIH",
            default=None,
            advanced=True,
        )
        lang: Optional[str] = SchemaField(
            description="IETF language tag for filter language", default=None
        )
        ids: Optional[list[str]] = SchemaField(
            description="List of task IDs to retrieve", default=None, advanced=False
        )

    class Output(BlockSchemaOutput):
        ids: list[str] = SchemaField(description="Task IDs")
        urls: list[str] = SchemaField(description="Task URLs")
        complete_data: list[dict] = SchemaField(
            description="Complete task data as dictionary"
        )

    def __init__(self):
        super().__init__(
            id="0b706e86-de15-11ef-a113-32d3674e8b7e",
            description="Get active tasks from Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistGetTasksBlock.Input,
            output_schema=TodoistGetTasksBlock.Output,
            disabled=not TODOIST_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": "2203306141",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["2995104339"]),
                ("urls", ["https://todoist.com/showTask?id=2995104339"]),
                (
                    "complete_data",
                    [
                        {
                            "id": "2995104339",
                            "project_id": "2203306141",
                            "url": "https://todoist.com/showTask?id=2995104339",
                            "is_completed": False,
                        }
                    ],
                ),
            ],
            test_mock={
                "get_tasks": lambda *args, **kwargs: [
                    {
                        "id": "2995104339",
                        "project_id": "2203306141",
                        "url": "https://todoist.com/showTask?id=2995104339",
                        "is_completed": False,
                    }
                ]
            },
        )

    @staticmethod
    def get_tasks(
        credentials: TodoistCredentials,
        *,
        project_id: Optional[str] = None,
        section_id: Optional[str] = None,
        label: Optional[str] = None,
        filter: Optional[str] = None,
        lang: Optional[str] = None,
        ids: Optional[list[str]] = None,
    ):
        with get_api(credentials) as api:
            # A filter query takes precedence over the other filters, as it did
            # on the REST v2 endpoint; API v1 serves it from its own endpoint.
            if filter is not None:
                pages = api.filter_tasks(query=filter, lang=lang)
            else:
                pages = api.get_tasks(
                    project_id=project_id,
                    section_id=section_id,
                    label=label,
                    ids=ids,
                )
            return [task_to_dict(task) for task in flatten_pages(pages)]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: TodoistCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            tasks = self.get_tasks(
                credentials,
                project_id=input_data.project_id,
                section_id=input_data.section_id,
                label=input_data.label,
                filter=input_data.filter,
                lang=input_data.lang,
                ids=input_data.ids,
            )

            yield "ids", [task["id"] for task in tasks]
            yield "urls", [task["url"] for task in tasks]
            yield "complete_data", tasks

        except Exception as e:
            yield "error", str(e)


class TodoistGetTaskBlock(Block):
    """Get an active task from Todoist"""

    class Input(BlockSchemaInput):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        task_id: str = SchemaField(description="Task ID to retrieve")

    class Output(BlockSchemaOutput):
        project_id: str = SchemaField(description="Project ID containing the task")
        url: str = SchemaField(description="Task URL")
        complete_data: dict = SchemaField(
            description="Complete task data as dictionary"
        )

    def __init__(self):
        super().__init__(
            id="16d7dc8c-de15-11ef-8ace-32d3674e8b7e",
            description="Get an active task from Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistGetTaskBlock.Input,
            output_schema=TodoistGetTaskBlock.Output,
            disabled=not TODOIST_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "task_id": "2995104339"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("project_id", "2203306141"),
                ("url", "https://todoist.com/showTask?id=2995104339"),
                (
                    "complete_data",
                    {
                        "id": "2995104339",
                        "project_id": "2203306141",
                        "url": "https://todoist.com/showTask?id=2995104339",
                    },
                ),
            ],
            test_mock={
                "get_task": lambda *args, **kwargs: {
                    "project_id": "2203306141",
                    "id": "2995104339",
                    "url": "https://todoist.com/showTask?id=2995104339",
                }
            },
        )

    @staticmethod
    def get_task(credentials: TodoistCredentials, task_id: str):
        with get_api(credentials) as api:
            return task_to_dict(api.get_task(task_id=task_id))

    async def run(
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

    class Input(BlockSchemaInput):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        task_id: str = SchemaField(description="Task ID to update")
        content: str = SchemaField(description="Task content", advanced=False)
        description: Optional[str] = SchemaField(
            description="Task description", default=None, advanced=False
        )
        project_id: Optional[str] = SchemaField(
            description="Project ID this task should belong to",
            default=None,
            advanced=False,
        )
        section_id: Optional[str] = SchemaField(
            description="Section ID this task should belong to",
            default=None,
            advanced=False,
        )
        parent_id: Optional[str] = SchemaField(
            description="Parent task ID", default=None, advanced=True
        )
        order: Optional[int] = SchemaField(
            description="Optional order among other tasks,[Non-zero integer value used by clients to sort tasks under the same parent]",
            default=None,
            advanced=True,
        )
        labels: Optional[list[str]] = SchemaField(
            description="Task labels", default=None, advanced=True
        )
        priority: Optional[int] = SchemaField(
            description="Task priority from 1 (normal) to 4 (urgent)",
            default=None,
            advanced=True,
        )
        due_date: Optional[datetime] = SchemaField(
            description="Due date in YYYY-MM-DD format", advanced=True, default=None
        )
        deadline_date: Optional[datetime] = SchemaField(
            description="Specific date in YYYY-MM-DD format relative to user's timezone",
            default=None,
            advanced=True,
        )
        assignee_id: Optional[str] = SchemaField(
            description="Responsible user ID", default=None, advanced=True
        )
        duration_unit: Optional[str] = SchemaField(
            description="Task duration unit (minute/day)", default=None, advanced=True
        )
        duration: Optional[int] = SchemaField(
            description="Task duration amount, You need to selecct the duration unit first",
            depends_on=["duration_unit"],
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the update was successful")

    def __init__(self):
        super().__init__(
            id="1eee6d32-de15-11ef-a2ff-32d3674e8b7e",
            description="Updates an existing task in Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistUpdateTaskBlock.Input,
            output_schema=TodoistUpdateTaskBlock.Output,
            disabled=not TODOIST_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "task_id": "2995104339",
                "content": "Buy Coffee",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"update_task": lambda *args, **kwargs: True},
        )

    @staticmethod
    def update_task(
        credentials: TodoistCredentials,
        task_id: str,
        *,
        content: Optional[str] = None,
        description: Optional[str] = None,
        project_id: Optional[str] = None,
        section_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        order: Optional[int] = None,
        labels: Optional[list[str]] = None,
        priority: Optional[int] = None,
        due_date: Optional[date] = None,
        deadline_date: Optional[date] = None,
        assignee_id: Optional[str] = None,
        duration: Optional[int] = None,
        duration_unit: Optional[str] = None,
    ):
        with get_api(credentials) as api:
            api.update_task(
                task_id,
                content=content,
                description=description,
                order=order,
                labels=labels,
                priority=priority,
                due_date=due_date,
                deadline_date=deadline_date,
                assignee_id=assignee_id,
                duration=duration,
                duration_unit=parse_duration_unit(duration_unit),
            )
            # project_id/section_id/parent_id are not sent: the update endpoint
            # never moved tasks (REST v2 ignored them too), and moving is a
            # separate move_task call this block has not made so far.
            return True

    async def run(
        self,
        input_data: Input,
        *,
        credentials: TodoistCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            self.update_task(
                credentials,
                input_data.task_id,
                content=input_data.content,
                description=input_data.description,
                project_id=input_data.project_id,
                section_id=input_data.section_id,
                parent_id=input_data.parent_id,
                order=input_data.order,
                labels=input_data.labels,
                priority=input_data.priority,
                due_date=input_data.due_date.date() if input_data.due_date else None,
                deadline_date=(
                    input_data.deadline_date.date()
                    if input_data.deadline_date
                    else None
                ),
                assignee_id=input_data.assignee_id,
                duration=input_data.duration,
                duration_unit=input_data.duration_unit,
            )

            yield "success", True

        except Exception as e:
            yield "error", str(e)


class TodoistCloseTaskBlock(Block):
    """Closes a task in Todoist"""

    class Input(BlockSchemaInput):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        task_id: str = SchemaField(description="Task ID to close")

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="Whether the task was successfully closed"
        )

    def __init__(self):
        super().__init__(
            id="29fac798-de15-11ef-b839-32d3674e8b7e",
            description="Closes a task in Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistCloseTaskBlock.Input,
            output_schema=TodoistCloseTaskBlock.Output,
            disabled=not TODOIST_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "task_id": "2995104339"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"close_task": lambda *args, **kwargs: True},
        )

    @staticmethod
    def close_task(credentials: TodoistCredentials, task_id: str):
        with get_api(credentials) as api:
            return api.complete_task(task_id=task_id)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: TodoistCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            is_success = self.close_task(credentials, input_data.task_id)
            yield "success", is_success

        except Exception as e:
            yield "error", str(e)


class TodoistReopenTaskBlock(Block):
    """Reopens a task in Todoist"""

    class Input(BlockSchemaInput):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        task_id: str = SchemaField(description="Task ID to reopen")

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="Whether the task was successfully reopened"
        )

    def __init__(self):
        super().__init__(
            id="2e6bf6f8-de15-11ef-ae7c-32d3674e8b7e",
            description="Reopens a task in Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistReopenTaskBlock.Input,
            output_schema=TodoistReopenTaskBlock.Output,
            disabled=not TODOIST_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "task_id": "2995104339"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"reopen_task": lambda *args, **kwargs: (True)},
        )

    @staticmethod
    def reopen_task(credentials: TodoistCredentials, task_id: str):
        with get_api(credentials) as api:
            return api.uncomplete_task(task_id=task_id)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: TodoistCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            is_success = self.reopen_task(credentials, input_data.task_id)
            yield "success", is_success

        except Exception as e:
            yield "error", str(e)


class TodoistDeleteTaskBlock(Block):
    """Deletes a task in Todoist"""

    class Input(BlockSchemaInput):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        task_id: str = SchemaField(description="Task ID to delete")

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="Whether the task was successfully deleted"
        )

    def __init__(self):
        super().__init__(
            id="33c29ada-de15-11ef-bcbb-32d3674e8b7e",
            description="Deletes a task in Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistDeleteTaskBlock.Input,
            output_schema=TodoistDeleteTaskBlock.Output,
            disabled=not TODOIST_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "task_id": "2995104339"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"delete_task": lambda *args, **kwargs: (True)},
            is_irreversible_action=True,
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    def delete_task(credentials: TodoistCredentials, task_id: str):
        with get_api(credentials) as api:
            return api.delete_task(task_id=task_id)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: TodoistCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            is_success = self.delete_task(credentials, input_data.task_id)
            yield "success", is_success

        except Exception as e:
            yield "error", str(e)
