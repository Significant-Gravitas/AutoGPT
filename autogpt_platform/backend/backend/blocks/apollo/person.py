from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.apollo._api import ApolloClient
from backend.blocks.apollo._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ApolloCredentials,
    ApolloCredentialsInput,
)
from backend.blocks.apollo.models import Contact, EnrichPersonRequest
from backend.data.model import CredentialsField, SchemaField


class GetPersonDetailBlock(Block):
    """Get detailed person data with Apollo API, including email reveal"""

    class Input(BlockSchemaInput):
        person_id: str = SchemaField(
            description="Apollo person ID to enrich (most accurate method)",
            default="",
            advanced=False,
        )
        first_name: str = SchemaField(
            description="First name of the person to enrich",
            default="",
            advanced=False,
        )
        last_name: str = SchemaField(
            description="Last name of the person to enrich",
            default="",
            advanced=False,
        )
        name: str = SchemaField(
            description="Full name of the person to enrich (alternative to first_name + last_name)",
            default="",
            advanced=False,
        )
        email: str = SchemaField(
            description="Known email address of the person (helps with matching)",
            default="",
            advanced=False,
        )
        domain: str = SchemaField(
            description="Company domain of the person (e.g., 'google.com')",
            default="",
            advanced=False,
        )
        company: str = SchemaField(
            description="Company name of the person",
            default="",
            advanced=False,
        )
        linkedin_url: str = SchemaField(
            description="LinkedIn URL of the person",
            default="",
            advanced=False,
        )
        organization_id: str = SchemaField(
            description="Apollo organization ID of the person's company",
            default="",
            advanced=True,
        )
        title: str = SchemaField(
            description="Job title of the person to enrich",
            default="",
            advanced=True,
        )
        credentials: ApolloCredentialsInput = CredentialsField(
            description="Apollo credentials",
        )

    class Output(BlockSchemaOutput):
        contact: Contact = SchemaField(
            description="Enriched contact information",
        )
        error: str = SchemaField(
            description="Error message if enrichment failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="3b18d46c-3db6-42ae-a228-0ba441bdd176",
            description="Get detailed person data with Apollo API, including email reveal",
            categories={BlockCategory.SEARCH},
            input_schema=GetPersonDetailBlock.Input,
            output_schema=GetPersonDetailBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "first_name": "John",
                "last_name": "Doe",
                "company": "Google",
            },
            test_output=[
                (
                    "contact",
                    Contact(
                        id="1",
                        name="John Doe",
                        first_name="John",
                        last_name="Doe",
                        email="john.doe@gmail.com",
                        title="Software Engineer",
                        organization_name="Google",
                        linkedin_url="https://www.linkedin.com/in/johndoe",
                    ),
                ),
            ],
            test_mock={
                "enrich_person": lambda query, credentials: Contact(
                    id="1",
                    name="John Doe",
                    first_name="John",
                    last_name="Doe",
                    email="john.doe@gmail.com",
                    title="Software Engineer",
                    organization_name="Google",
                    linkedin_url="https://www.linkedin.com/in/johndoe",
                )
            },
        )

    @staticmethod
    async def enrich_person(
        query: EnrichPersonRequest, credentials: ApolloCredentials
    ) -> Contact:
        client = ApolloClient(credentials)
        return await client.enrich_person(query)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: ApolloCredentials,
        **kwargs,
    ) -> BlockOutput:
        query = EnrichPersonRequest(**input_data.model_dump())
        yield "contact", await self.enrich_person(query, credentials)
