from backend.blocks.apollo._api import ApolloClient
from backend.blocks.apollo._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ApolloCredentials,
    ApolloCredentialsInput,
)
from backend.blocks.apollo.models import (
    Contact,
    ContactEmailStatuses,
    SearchPeopleRequest,
    SenorityLevels,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class SearchPeopleBlock(Block):
    """Search for people in Apollo"""

    class Input(BlockSchema):
        person_titles: list[str] = SchemaField(
            description="""Job titles held by the people you want to find. For a person to be included in search results, they only need to match 1 of the job titles you add. Adding more job titles expands your search results.

        Results also include job titles with the same terms, even if they are not exact matches. For example, searching for marketing manager might return people with the job title content marketing manager.

        Use this parameter in combination with the person_seniorities[] parameter to find people based on specific job functions and seniority levels.
        """,
            default=[],
            advanced=False,
        )
        person_locations: list[str] = SchemaField(
            description="""The location where people live. You can search across cities, US states, and countries.

        To find people based on the headquarters locations of their current employer, use the organization_locations parameter.""",
            default=[],
            advanced=False,
        )
        person_seniorities: list[SenorityLevels] = SchemaField(
            description="""The job seniority that people hold within their current employer. This enables you to find people that currently hold positions at certain reporting levels, such as Director level or senior IC level.

        For a person to be included in search results, they only need to match 1 of the seniorities you add. Adding more seniorities expands your search results.

        Searches only return results based on their current job title, so searching for Director-level employees only returns people that currently hold a Director-level title. If someone was previously a Director, but is currently a VP, they would not be included in your search results.

        Use this parameter in combination with the person_titles[] parameter to find people based on specific job functions and seniority levels.""",
            default=[],
            advanced=False,
        )
        organization_locations: list[str] = SchemaField(
            description="""The location of the company headquarters for a person's current employer. You can search across cities, US states, and countries.

        If a company has several office locations, results are still based on the headquarters location. For example, if you search chicago but a company's HQ location is in boston, people that work for the Boston-based company will not appear in your results, even if they match other parameters.

        To find people based on their personal location, use the person_locations parameter.""",
            default=[],
            advanced=False,
        )
        q_organization_domains: list[str] = SchemaField(
            description="""The domain name for the person's employer. This can be the current employer or a previous employer. Do not include www., the @ symbol, or similar.

        You can add multiple domains to search across companies.

          Examples: apollo.io and microsoft.com""",
            default=[],
            advanced=False,
        )
        contact_email_statuses: list[ContactEmailStatuses] = SchemaField(
            description="""The email statuses for the people you want to find. You can add multiple statuses to expand your search.""",
            default=[],
            advanced=False,
        )
        organization_ids: list[str] = SchemaField(
            description="""The Apollo IDs for the companies (employers) you want to include in your search results. Each company in the Apollo database is assigned a unique ID.

        To find IDs, call the Organization Search endpoint and identify the values for organization_id.""",
            default=[],
            advanced=False,
        )
        organization_num_empoloyees_range: list[int] = SchemaField(
            description="""The number range of employees working for the company. This enables you to find companies based on headcount. You can add multiple ranges to expand your search results.

        Each range you add needs to be a string, with the upper and lower numbers of the range separated only by a comma.""",
            default=[],
            advanced=False,
        )
        q_keywords: str = SchemaField(
            description="""A string of words over which we want to filter the results""",
            default="",
            advanced=False,
        )
        max_results: int = SchemaField(
            description="""The maximum number of results to return. If you don't specify this parameter, the default is 100.""",
            default=100,
            ge=1,
            le=50000,
            advanced=True,
        )
        # page: int = SchemaField(
        #     description="""The page number of the Apollo data that you want to retrieve.

        # Use this parameter in combination with the per_page parameter to make search results for navigable and improve the performance of the endpoint.""",
        #     default=1,
        #     advanced=True,
        # )
        # per_page: int = SchemaField(
        #     description="""The number of search results that should be returned for each page. Limited the number of results per page improves the endpoint's performance.

        # Use the page parameter to search the different pages of data.""",
        #     default=100,
        #     advanced=True,
        # )
        credentials: ApolloCredentialsInput = SchemaField(
            description="Apollo credentials",
        )

    class Output(BlockSchema):
        people: list[Contact] = SchemaField(
            description="List of people found",
            default=[],
        )
        person: Contact = SchemaField(
            description="Each found person, one at a time",
        )
        error: str = SchemaField(
            description="Error message if the search failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="c2adb3aa-5aae-488d-8a6e-4eb8c23e2ed6",
            description="Search for people in Apollo",
            categories={BlockCategory.SEARCH},
            input_schema=SearchPeopleBlock.Input,
            output_schema=SearchPeopleBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[
                (
                    "people",
                    [
                        {
                            "id": "1",
                            "name": "John Doe",
                        }
                    ],
                )
            ],
        )

    def run(
        self,
        input_data: Input,
        *,
        credentials: ApolloCredentials,
        **kwargs,
    ) -> BlockOutput:
        client = ApolloClient(credentials)

        query = SearchPeopleRequest(**input_data.model_dump(exclude={"credentials"}))
        people = client.search_people(query)
        for person in people:
            yield "person", person
        yield "people", people
