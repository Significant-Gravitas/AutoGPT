from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    SchemaField,
    String,
    OAuth2Credentials,
    APIKeyCredentials,
    CredentialsMetaInput,
)
from ._api import LinearAPIException, LinearClient
from .models import Project
from ._config import linear


class LinearSearchProjectsBlock(Block):
    """Block for searching projects on Linear"""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = linear.credentials_field(
            description="Linear credentials with read permissions",
            required_scopes={"read"},
        )
        term: String = SchemaField(description="Term to search for projects")

    class Output(BlockSchema):
        projects: list[Project] = SchemaField(description="List of projects")
        error: String = SchemaField(
            description="Error message if search failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="446a1d35-9d8f-4ac5-83ea-7684ec50e6af",
            description="Searches for projects on Linear",
            input_schema=self.Input,
            output_schema=self.Output,
            categories={BlockCategory.PRODUCTIVITY},
        )

    @staticmethod
    async def search_projects(
        credentials: OAuth2Credentials | APIKeyCredentials,
        term: str,
    ) -> list[Project]:
        client = LinearClient(credentials=credentials)
        response: list[Project] = await client.try_search_projects(term=term)
        return response

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials | APIKeyCredentials,
        **kwargs
    ) -> BlockOutput:
        """Execute the project search"""
        try:
            projects = await self.search_projects(
                credentials=credentials,
                term=input_data.term,
            )

            yield "projects", projects

        except LinearAPIException as e:
            yield "error", str(e)
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"
