from typing_extensions import Optional
from todoist_api_python.api import TodoistAPI

from backend.blocks.jina._auth import TEST_CREDENTIALS
from backend.blocks.todoist._auth import (
    TEST_CREDENTIALS_INPUT,
    TodoistCredentials,
    TodoistCredentialsInput,
    TodoistCredentialsField,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


# class TodoistListProjectsBlock(Block):
#     """Gets all projects for a Todoist user"""

#     class Input(BlockSchema):
#         credentials: TodoistCredentialsInput = TodoistCredentialsField([])

#     class Output(BlockSchema):
#         names_list: list[str] = SchemaField(description="List of project names")
#         ids_list: list[str] = SchemaField(description="List of project IDs")
#         url_list: list[str] = SchemaField(description="List of project URLs")
#         complete_data: list[dict] = SchemaField(description="Complete project data including all fields")

#     def __init__(self):
#         super().__init__(
#             id="e141bce6-e0fa-4f3f-bbda-8a2bddc2c659",
#             description="Gets all projects and their details from Todoist",
#             categories={BlockCategory.PRODUCTIVITY},
#             input_schema=TodoistListProjectsBlock.Input,
#             output_schema=TodoistListProjectsBlock.Output,
#             test_input={
#                 "credentials": TEST_CREDENTIALS_INPUT,
#             },
#             test_credentials=TEST_CREDENTIALS,
#             test_output=[
#                 ("names_list", ["Inbox"]),
#                 ("ids_list", ["220474322"]),
#                 ("url_list", ["https://todoist.com/showProject?id=220474322"]),
#                 ("complete_data", [{
#                     "id": "220474322",
#                     "name": "Inbox",
#                     "url": "https://todoist.com/showProject?id=220474322"
#                 }])
#             ],
#             test_mock={
#                 "get_project_lists": lambda *args, **kwargs: (
#                     ["Inbox"],
#                     ["220474322"],
#                     ["https://todoist.com/showProject?id=220474322"],
#                     [{"id": "220474322", "name": "Inbox", "url": "https://todoist.com/showProject?id=220474322"}],
#                     None,
#                 )
#             },
#         )

#     @staticmethod
#     def get_project_lists(credentials: TodoistCredentials):
#         try:
#             api = TodoistAPI(credentials.access_token.get_secret_value())
#             projects = api.get_projects()

#             names = []
#             ids = []
#             urls = []
#             complete_data = []

#             for project in projects:
#                 names.append(project.name)
#                 ids.append(project.id)
#                 urls.append(project.url)
#                 complete_data.append(project.__dict__)

#             return names, ids, urls, complete_data, None

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
#             names, ids, urls, data, error = self.get_project_lists(credentials)

#             if names:
#                 yield "names_list", names
#             if ids:
#                 yield "ids_list", ids
#             if urls:
#                 yield "url_list", urls
#             if data:
#                 yield "complete_data", data

#         except Exception as e:
#             yield "error", str(e)

# class TodoistCreateProjectBlock(Block):
#     """Creates a new project in Todoist"""

#     class Input(BlockSchema):
#         credentials: TodoistCredentialsInput = TodoistCredentialsField([])
#         name: str = SchemaField(description="Name of the project", advanced=False)
#         parent_id: Optional[str] = SchemaField(description="Parent project ID", default=None, advanced=True)
#         color: Optional[str] = SchemaField(description="Color of the project icon", default=None, advanced=True)
#         is_favorite: bool = SchemaField(description="Whether the project is a favorite", default=False ,advanced=True)
#         view_style: Optional[str] = SchemaField(description="Display style (list or board)", default=None, advanced=True)

#     class Output(BlockSchema):
#         project_id: str = SchemaField(description="ID of created project")
#         project_name: str = SchemaField(description="Name of created project")
#         project_url: str = SchemaField(description="URL of created project")
#         complete_data: dict = SchemaField(description="Complete project data including all fields")

#     def __init__(self):
#         super().__init__(
#             id="f252bde7-e1fa-4f3f-ccda-9a2bddc2c770",
#             description="Creates a new project in Todoist",
#             categories={BlockCategory.PRODUCTIVITY},
#             input_schema=TodoistCreateProjectBlock.Input,
#             output_schema=TodoistCreateProjectBlock.Output,
#             test_input={
#                 "credentials": TEST_CREDENTIALS_INPUT,
#                 "name": "Test Project"
#             },
#             test_credentials=TEST_CREDENTIALS,
#             test_output=[
#                 ("project_id", "2203306141"),
#                 ("project_name", "Test Project"),
#                 ("project_url", "https://todoist.com/showProject?id=2203306141"),
#                 ("complete_data", {
#                     "id": "2203306141",
#                     "name": "Test Project",
#                     "url": "https://todoist.com/showProject?id=2203306141"
#                 })
#             ],
#             test_mock={
#                 "create_project": lambda *args, **kwargs: (
#                     "2203306141",
#                     "Test Project",
#                     "https://todoist.com/showProject?id=2203306141",
#                     {"id": "2203306141", "name": "Test Project", "url": "https://todoist.com/showProject?id=2203306141"},
#                     None
#                 )
#             },
#         )

#     @staticmethod
#     def create_project(credentials: TodoistCredentials, name: str, parent_id: Optional[str],
#                       color: Optional[str], is_favorite: bool , view_style: Optional[str]):
#         try:
#             api = TodoistAPI(credentials.access_token.get_secret_value())
#             params = {"name": name,"is_favorite":is_favorite}

#             if parent_id is not None:
#                 params["parent_id"] = parent_id
#             if color is not None:
#                 params["color"] = color
#             if view_style is not None:
#                 params["view_style"] = view_style

#             project = api.add_project(**params)
#             return project.id, project.name, project.url, project.__dict__, None

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
#             project_id, project_name, project_url, data, error = self.create_project(
#                 credentials=credentials,
#                 name=input_data.name,
#                 parent_id=input_data.parent_id,
#                 color=input_data.color,
#                 is_favorite=input_data.is_favorite,
#                 view_style=input_data.view_style
#             )

#             if project_id:
#                 yield "project_id", project_id
#             if project_name:
#                 yield "project_name", project_name
#             if project_url:
#                 yield "project_url", project_url
#             if data:
#                 yield "complete_data", data

#         except Exception as e:
#             yield "error", str(e)
