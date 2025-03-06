from todoist_api_python.api import TodoistAPI
from typing_extensions import Optional

from backend.blocks.todoist._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    TodoistCredentials,
    TodoistCredentialsField,
    TodoistCredentialsInput,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TodoistListSectionsBlock(Block):
    """Gets all sections for a Todoist project"""

    class Input(BlockSchema):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        project_id: Optional[str] = SchemaField(
            description="Optional project ID to filter sections"
        )

    class Output(BlockSchema):
        names_list: list[str] = SchemaField(description="List of section names")
        ids_list: list[str] = SchemaField(description="List of section IDs")
        complete_data: list[dict] = SchemaField(
            description="Complete section data including all fields"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="d6a116d8-de14-11ef-a94c-32d3674e8b7e",
            description="Gets all sections and their details from Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistListSectionsBlock.Input,
            output_schema=TodoistListSectionsBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": "2203306141",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("names_list", ["Groceries"]),
                ("ids_list", ["7025"]),
                (
                    "complete_data",
                    [
                        {
                            "id": "7025",
                            "project_id": "2203306141",
                            "order": 1,
                            "name": "Groceries",
                        }
                    ],
                ),
            ],
            test_mock={
                "get_section_lists": lambda *args, **kwargs: (
                    ["Groceries"],
                    ["7025"],
                    [
                        {
                            "id": "7025",
                            "project_id": "2203306141",
                            "order": 1,
                            "name": "Groceries",
                        }
                    ],
                )
            },
        )

    @staticmethod
    def get_section_lists(
        credentials: TodoistCredentials, project_id: Optional[str] = None
    ):
        try:
            api = TodoistAPI(credentials.access_token.get_secret_value())
            sections = api.get_sections(project_id=project_id)

            names = []
            ids = []
            complete_data = []

            for section in sections:
                names.append(section.name)
                ids.append(section.id)
                complete_data.append(section.__dict__)

            return names, ids, complete_data

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
            names, ids, data = self.get_section_lists(
                credentials, input_data.project_id
            )

            if names:
                yield "names_list", names
            if ids:
                yield "ids_list", ids
            if data:
                yield "complete_data", data

        except Exception as e:
            yield "error", str(e)


# Error in official todoist SDK. Will add this block using sync_api
# class TodoistCreateSectionBlock(Block):
#     """Creates a new section in a Todoist project"""

#     class Input(BlockSchema):
#         credentials: TodoistCredentialsInput = TodoistCredentialsField([])
#         name: str = SchemaField(description="Section name")
#         project_id: str = SchemaField(description="Project ID this section should belong to")
#         order: Optional[int] = SchemaField(description="Optional order among other sections", default=None)

#     class Output(BlockSchema):
#         success: bool = SchemaField(description="Whether section was successfully created")
#         error: str = SchemaField(description="Error message if the request failed")

#     def __init__(self):
#         super().__init__(
#             id="e3025cfc-de14-11ef-b9f2-32d3674e8b7e",
#             description="Creates a new section in a Todoist project",
#             categories={BlockCategory.PRODUCTIVITY},
#             input_schema=TodoistCreateSectionBlock.Input,
#             output_schema=TodoistCreateSectionBlock.Output,
#             test_input={
#                 "credentials": TEST_CREDENTIALS_INPUT,
#                 "name": "Groceries",
#                 "project_id": "2203306141"
#             },
#             test_credentials=TEST_CREDENTIALS,
#             test_output=[
#                 ("success", True)
#             ],
#             test_mock={
#                 "create_section": lambda *args, **kwargs: (
#                     {"id": "7025", "project_id": "2203306141", "order": 1, "name": "Groceries"},
#                 )
#             },
#         )

#     @staticmethod
#     def create_section(credentials: TodoistCredentials, name: str, project_id: str, order: Optional[int] = None):
#         try:
#             api = TodoistAPI(credentials.access_token.get_secret_value())
#             section = api.add_section(name=name, project_id=project_id, order=order)
#             return section.__dict__

#         except Exception as e:
#             raise e

#     def run(
#         self,
#         input_data: Input,
#         *,
#         credentials: TodoistCredentials,
#         **kwargs,
#     ) -> BlockOutput:
#         try:
#             section_data = self.create_section(
#                 credentials,
#                 input_data.name,
#                 input_data.project_id,
#                 input_data.order
#             )

#             if section_data:
#                 yield "success", True

#         except Exception as e:
#             yield "error", str(e)


class TodoistGetSectionBlock(Block):
    """Gets a single section from Todoist by ID"""

    class Input(BlockSchema):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        section_id: str = SchemaField(description="ID of section to fetch")

    class Output(BlockSchema):
        id: str = SchemaField(description="ID of section")
        project_id: str = SchemaField(description="Project ID the section belongs to")
        order: int = SchemaField(description="Order of the section")
        name: str = SchemaField(description="Name of the section")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="ea5580e2-de14-11ef-a5d3-32d3674e8b7e",
            description="Gets a single section by ID from Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistGetSectionBlock.Input,
            output_schema=TodoistGetSectionBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "section_id": "7025"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("id", "7025"),
                ("project_id", "2203306141"),
                ("order", 1),
                ("name", "Groceries"),
            ],
            test_mock={
                "get_section": lambda *args, **kwargs: {
                    "id": "7025",
                    "project_id": "2203306141",
                    "order": 1,
                    "name": "Groceries",
                }
            },
        )

    @staticmethod
    def get_section(credentials: TodoistCredentials, section_id: str):
        try:
            api = TodoistAPI(credentials.access_token.get_secret_value())
            section = api.get_section(section_id=section_id)
            return section.__dict__

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
            section_data = self.get_section(credentials, input_data.section_id)

            if section_data:
                yield "id", section_data["id"]
                yield "project_id", section_data["project_id"]
                yield "order", section_data["order"]
                yield "name", section_data["name"]

        except Exception as e:
            yield "error", str(e)


class TodoistDeleteSectionBlock(Block):
    """Deletes a section and all its tasks from Todoist"""

    class Input(BlockSchema):
        credentials: TodoistCredentialsInput = TodoistCredentialsField([])
        section_id: str = SchemaField(description="ID of section to delete")

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether section was successfully deleted"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="f0e52eee-de14-11ef-9b12-32d3674e8b7e",
            description="Deletes a section and all its tasks from Todoist",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=TodoistDeleteSectionBlock.Input,
            output_schema=TodoistDeleteSectionBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "section_id": "7025"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"delete_section": lambda *args, **kwargs: (True)},
        )

    @staticmethod
    def delete_section(credentials: TodoistCredentials, section_id: str):
        try:
            api = TodoistAPI(credentials.access_token.get_secret_value())
            success = api.delete_section(section_id=section_id)
            return success

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
            success = self.delete_section(credentials, input_data.section_id)
            yield "success", success

        except Exception as e:
            yield "error", str(e)
