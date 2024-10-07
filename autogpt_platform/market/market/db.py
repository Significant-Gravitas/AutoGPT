import datetime
import typing

import fuzzywuzzy.fuzz
import prisma.enums
import prisma.errors
import prisma.models
import prisma.types
import pydantic

import market.model
import market.utils.extension_types


class AgentQueryError(Exception):
    """Custom exception for agent query errors"""

    pass


class TopAgentsDBResponse(pydantic.BaseModel):
    """
    Represents a response containing a list of top agents.

    Attributes:
        analytics (list[AgentResponse]): The list of top agents.
        total_count (int): The total count of agents.
        page (int): The current page number.
        page_size (int): The number of agents per page.
        total_pages (int): The total number of pages.
    """

    analytics: list[prisma.models.AnalyticsTracker]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class FeaturedAgentResponse(pydantic.BaseModel):
    """
    Represents a response containing a list of featured agents.

    Attributes:
        featured_agents (list[FeaturedAgent]): The list of featured agents.
        total_count (int): The total count of featured agents.
        page (int): The current page number.
        page_size (int): The number of agents per page.
        total_pages (int): The total number of pages.
    """

    featured_agents: list[prisma.models.FeaturedAgent]
    total_count: int
    page: int
    page_size: int
    total_pages: int

async def delete_agent(agent_id: str) -> prisma.models.Agents | None:
    """
    Delete an agent from the database.

    Args:
        agent_id (str): The ID of the agent to delete.

    Returns:
        prisma.models.Agents | None: The deleted agent if found, None otherwise.

    Raises:
        AgentQueryError: If there is an error deleting the agent from the database.
    """
    try:
        deleted_agent = await prisma.models.Agents.prisma().delete(
            where={"id": agent_id}
        )
        return deleted_agent
    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Database query failed: {str(e)}")
    except Exception as e:
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")


async def create_agent_entry(
    name: str,
    description: str,
    author: str,
    keywords: typing.List[str],
    categories: typing.List[str],
    graph: prisma.Json,
    submission_state: prisma.enums.SubmissionStatus = prisma.enums.SubmissionStatus.PENDING,
):
    """
    Create a new agent entry in the database.

    Args:
        name (str): The name of the agent.
        description (str): The description of the agent.
        author (str): The author of the agent.
        keywords (List[str]): The keywords associated with the agent.
        categories (List[str]): The categories associated with the agent.
        graph (dict): The graph data of the agent.

    Returns:
        dict: The newly created agent entry.

    Raises:
        AgentQueryError: If there is an error creating the agent entry.
    """
    try:
        agent = await prisma.models.Agents.prisma().create(
            data={
                "name": name,
                "description": description,
                "author": author,
                "keywords": keywords,
                "categories": categories,
                "graph": graph,
                "AnalyticsTracker": {"create": {"downloads": 0, "views": 0}},
                "submissionStatus": submission_state,
            }
        )

        return agent

    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Database query failed: {str(e)}")
    except Exception as e:
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")


async def update_agent_entry(
    agent_id: str,
    version: int,
    submission_state: prisma.enums.SubmissionStatus,
    comments: str | None = None,
) -> prisma.models.Agents | None:
    """
    Update an existing agent entry in the database.

    Args:
        agent_id (str): The ID of the agent.
        version (int): The version of the agent.
        submission_state (prisma.enums.SubmissionStatus): The submission state of the agent.
    """

    try:
        agent = await prisma.models.Agents.prisma().update(
            where={"id": agent_id},
            data={
                "version": version,
                "submissionStatus": submission_state,
                "submissionReviewDate": datetime.datetime.now(datetime.timezone.utc),
                "submissionReviewComments": comments,
            },
        )

        return agent
    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Agent Update Failed Database query failed: {str(e)}")
    except Exception as e:
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")


async def get_agents(
    page: int = 1,
    page_size: int = 10,
    name: str | None = None,
    keyword: str | None = None,
    category: str | None = None,
    description: str | None = None,
    description_threshold: int = 60,
    submission_status: prisma.enums.SubmissionStatus = prisma.enums.SubmissionStatus.APPROVED,
    sort_by: str = "createdAt",
    sort_order: typing.Literal["desc"] | typing.Literal["asc"] = "desc",
):
    """
    Retrieve a list of agents from the database based on the provided filters and pagination parameters.

    Args:
        page (int, optional): The page number to retrieve. Defaults to 1.
        page_size (int, optional): The number of agents per page. Defaults to 10.
        name (str, optional): Filter agents by name. Defaults to None.
        keyword (str, optional): Filter agents by keyword. Defaults to None.
        category (str, optional): Filter agents by category. Defaults to None.
        description (str, optional): Filter agents by description. Defaults to None.
        description_threshold (int, optional): The minimum fuzzy search threshold for the description. Defaults to 60.
        sort_by (str, optional): The field to sort the agents by. Defaults to "createdAt".
        sort_order (str, optional): The sort order ("asc" or "desc"). Defaults to "desc".

    Returns:
        dict: A dictionary containing the list of agents, total count, current page number, page size, and total number of pages.
    """
    try:
        # Define the base query
        query = {}

        # Add optional filters
        if name:
            query["name"] = {"contains": name, "mode": "insensitive"}
        if keyword:
            query["keywords"] = {"has": keyword}
        if category:
            query["categories"] = {"has": category}

        query["submissionStatus"] = submission_status

        # Define sorting
        order = {sort_by: sort_order}

        # Calculate pagination
        skip = (page - 1) * page_size

        # Execute the query
        try:
            agents = await prisma.models.Agents.prisma().find_many(
                where=query,  # type: ignore
                order=order,  # type: ignore
                skip=skip,
                take=page_size,
            )
        except prisma.errors.PrismaError as e:
            raise AgentQueryError(f"Database query failed: {str(e)}")

        # Apply fuzzy search on description if provided
        if description:
            try:
                filtered_agents = []
                for agent in agents:
                    if (
                        agent.description
                        and fuzzywuzzy.fuzz.partial_ratio(
                            description.lower(), agent.description.lower()
                        )
                        >= description_threshold
                    ):
                        filtered_agents.append(agent)
                agents = filtered_agents
            except AttributeError as e:
                raise AgentQueryError(f"Error during fuzzy search: {str(e)}")

        # Get total count for pagination info
        total_count = len(agents)

        return {
            "agents": agents,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size,
        }

    except AgentQueryError as e:
        # Log the error or handle it as needed
        raise e
    except ValueError as e:
        raise AgentQueryError(f"Invalid input parameter: {str(e)}")
    except Exception as e:
        # Catch any other unexpected exceptions
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")


async def get_agent_details(agent_id: str, version: int | None = None):
    """
    Retrieve agent details from the database.

    Args:
        agent_id (str): The ID of the agent.
        version (int | None, optional): The version of the agent. Defaults to None.

    Returns:
        dict: The agent details.

    Raises:
        AgentQueryError: If the agent is not found or if there is an error querying the database.
    """
    try:
        query = {"id": agent_id}
        if version is not None:
            query["version"] = version  # type: ignore

        agent = await prisma.models.Agents.prisma().find_first(where=query)  # type: ignore

        if not agent:
            raise AgentQueryError("Agent not found")

        return agent

    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Database query failed: {str(e)}")
    except Exception as e:
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")


async def search_db(
    query: str,
    page: int = 1,
    page_size: int = 10,
    categories: typing.List[str] | None = None,
    description_threshold: int = 60,
    sort_by: str = "rank",
    sort_order: typing.Literal["desc"] | typing.Literal["asc"] = "desc",
    submission_status: prisma.enums.SubmissionStatus = prisma.enums.SubmissionStatus.APPROVED,
) -> typing.List[market.utils.extension_types.AgentsWithRank]:
    """Perform a search for agents based on the provided query string.

    Args:
        query (str): the search string
        page (int, optional): page for searching. Defaults to 1.
        page_size (int, optional): the number of results to return. Defaults to 10.
        categories (List[str] | None, optional): list of category filters. Defaults to None.
        description_threshold (int, optional): number of characters to return. Defaults to 60.
        sort_by (str, optional): sort by option. Defaults to "rank".
        sort_order ("asc" | "desc", optional): the sort order. Defaults to "desc".

    Raises:
        AgentQueryError: Raises an error if the query fails.
        AgentQueryError: Raises if an unexpected error occurs.

    Returns:
        List[AgentsWithRank]: List of agents matching the search criteria.
    """
    try:
        offset = (page - 1) * page_size

        category_filter = ""
        if categories:
            category_conditions = [f"'{cat}' = ANY(categories)" for cat in categories]
            category_filter = "AND (" + " OR ".join(category_conditions) + ")"

        # Construct the ORDER BY clause based on the sort_by parameter
        if sort_by in ["createdAt", "updatedAt"]:
            order_by_clause = f'"{sort_by}" {sort_order.upper()}, rank DESC'
        elif sort_by == "name":
            order_by_clause = f"name {sort_order.upper()}, rank DESC"
        else:
            order_by_clause = 'rank DESC, "createdAt" DESC'

        submission_status_filter = f""""submissionStatus" = '{submission_status}'"""

        sql_query = f"""
        WITH query AS (
            SELECT to_tsquery(string_agg(lexeme || ':*', ' & ' ORDER BY positions)) AS q 
            FROM unnest(to_tsvector('{query}'))
        )
        SELECT 
            id, 
            "createdAt", 
            "updatedAt", 
            version, 
            name, 
            LEFT(description, {description_threshold}) AS description, 
            author, 
            keywords, 
            categories, 
            graph,
            "submissionStatus",
            "submissionDate",
            ts_rank(CAST(search AS tsvector), query.q) AS rank
        FROM "Agents", query
        WHERE 1=1 {category_filter} AND {submission_status_filter}
        ORDER BY {order_by_clause}
        LIMIT {page_size}
        OFFSET {offset};
        """

        results = await prisma.client.get_client().query_raw(
            query=sql_query,
            model=market.utils.extension_types.AgentsWithRank,
        )

        return results

    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Database query failed: {str(e)}")
    except Exception as e:
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")


async def get_top_agents_by_downloads(
    page: int = 1,
    page_size: int = 10,
    submission_status: prisma.enums.SubmissionStatus = prisma.enums.SubmissionStatus.APPROVED,
) -> TopAgentsDBResponse:
    """Retrieve the top agents by download count.

    Args:
        page (int, optional): The page number. Defaults to 1.
        page_size (int, optional): The number of agents per page. Defaults to 10.

    Returns:
        dict: A dictionary containing the list of agents, total count, current page number, page size, and total number of pages.
    """
    try:
        # Calculate pagination
        skip = (page - 1) * page_size

        # Execute the query
        try:
            # Agents with no downloads will not be included in the results... is this the desired behavior?
            analytics = await prisma.models.AnalyticsTracker.prisma().find_many(
                include={"agent": True},
                order={"downloads": "desc"},
                where={"agent": {"is": {"submissionStatus": submission_status}}},
                skip=skip,
                take=page_size,
            )
        except prisma.errors.PrismaError as e:
            raise AgentQueryError(f"Database query failed: {str(e)}")

        # Get total count for pagination info
        total_count = len(analytics)

        return TopAgentsDBResponse(
            analytics=analytics,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=(total_count + page_size - 1) // page_size,
        )

    except AgentQueryError as e:
        # Log the error or handle it as needed
        raise e from e
    except ValueError as e:
        raise AgentQueryError(f"Invalid input parameter: {str(e)}") from e
    except Exception as e:
        # Catch any other unexpected exceptions
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}") from e


async def set_agent_featured(
    agent_id: str, is_active: bool = True, featured_categories: list[str] = ["featured"]
) -> prisma.models.FeaturedAgent:
    """Set an agent as featured in the database.

    Args:
        agent_id (str): The ID of the agent.
        category (str, optional): The category to set the agent as featured. Defaults to "featured".

    Raises:
        AgentQueryError: If there is an error setting the agent as featured.
    """
    try:
        agent = await prisma.models.Agents.prisma().find_unique(where={"id": agent_id})
        if not agent:
            raise AgentQueryError(f"Agent with ID {agent_id} not found.")

        featured = await prisma.models.FeaturedAgent.prisma().upsert(
            where={"agentId": agent_id},
            data={
                "update": {
                    "featuredCategories": featured_categories,
                    "isActive": is_active,
                },
                "create": {
                    "featuredCategories": featured_categories,
                    "isActive": is_active,
                    "agent": {"connect": {"id": agent_id}},
                },
            },
        )
        return featured

    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Database query failed: {str(e)}")
    except Exception as e:
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")


async def get_featured_agents(
    category: str = "featured",
    page: int = 1,
    page_size: int = 10,
    submission_status: prisma.enums.SubmissionStatus = prisma.enums.SubmissionStatus.APPROVED,
) -> FeaturedAgentResponse:
    """Retrieve a list of featured agents from the database based on the provided category.

    Args:
        category (str, optional): The category of featured agents to retrieve. Defaults to "featured".
        page (int, optional): The page number to retrieve. Defaults to 1.
        page_size (int, optional): The number of agents per page. Defaults to 10.

    Returns:
        dict: A dictionary containing the list of featured agents, total count, current page number, page size, and total number of pages.
    """
    try:
        # Calculate pagination
        skip = (page - 1) * page_size

        # Execute the query
        try:
            featured_agents = await prisma.models.FeaturedAgent.prisma().find_many(
                where={
                    "featuredCategories": {"has": category},
                    "isActive": True,
                    "agent": {"is": {"submissionStatus": submission_status}},
                },
                include={"agent": {"include": {"AnalyticsTracker": True}}},
                skip=skip,
                take=page_size,
            )
        except prisma.errors.PrismaError as e:
            raise AgentQueryError(f"Database query failed: {str(e)}")

        # Get total count for pagination info
        total_count = len(featured_agents)

        return FeaturedAgentResponse(
            featured_agents=featured_agents,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=(total_count + page_size - 1) // page_size,
        )

    except AgentQueryError as e:
        # Log the error or handle it as needed
        raise e from e
    except ValueError as e:
        raise AgentQueryError(f"Invalid input parameter: {str(e)}") from e
    except Exception as e:
        # Catch any other unexpected exceptions
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}") from e


async def remove_featured_category(
    agent_id: str, category: str
) -> prisma.models.FeaturedAgent | None:
    """Adds a featured category to an agent.

    Args:
        agent_id (str): The ID of the agent.
        category (str): The category to add to the agent.

    Returns:
        FeaturedAgentResponse: The updated list of featured agents.
    """
    try:
        # get the existing categories
        featured_agent = await prisma.models.FeaturedAgent.prisma().find_unique(
            where={"agentId": agent_id},
            include={"agent": True},
        )

        if not featured_agent:
            raise AgentQueryError(f"Agent with ID {agent_id} not found.")

        # remove the category from the list
        featured_agent.featuredCategories.remove(category)

        featured_agent = await prisma.models.FeaturedAgent.prisma().update(
            where={"agentId": agent_id},
            data={"featuredCategories": featured_agent.featuredCategories},
        )

        return featured_agent

    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Database query failed: {str(e)}")
    except Exception as e:
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")


async def add_featured_category(
    agent_id: str, category: str
) -> prisma.models.FeaturedAgent | None:
    """Removes a featured category from an agent.

    Args:
        agent_id (str): The ID of the agent.
        category (str): The category to remove from the agent.

    Returns:
        FeaturedAgentResponse: The updated list of featured agents.
    """
    try:
        featured_agent = await prisma.models.FeaturedAgent.prisma().update(
            where={"agentId": agent_id},
            data={"featuredCategories": {"push": [category]}},
        )

        return featured_agent

    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Database query failed: {str(e)}")
    except Exception as e:
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")


async def get_agent_featured(agent_id: str) -> prisma.models.FeaturedAgent | None:
    """Retrieve an agent's featured categories from the database.

    Args:
        agent_id (str): The ID of the agent.

    Returns:
        FeaturedAgentResponse: The list of featured agents.
    """
    try:
        featured_agent = await prisma.models.FeaturedAgent.prisma().find_unique(
            where={"agentId": agent_id},
        )
        return featured_agent
    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Database query failed: {str(e)}")
    except Exception as e:
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")


async def get_not_featured_agents(
    page: int = 1, page_size: int = 10
) -> typing.List[prisma.models.Agents]:
    """
    Retrieve a list of not featured agents from the database.
    """
    try:
        agents = await prisma.client.get_client().query_raw(
            query=f"""
            SELECT 
                "market"."Agents".id, 
                "market"."Agents"."createdAt", 
                "market"."Agents"."updatedAt", 
                "market"."Agents".version, 
                "market"."Agents".name, 
                LEFT("market"."Agents".description, 500) AS description, 
                "market"."Agents".author, 
                "market"."Agents".keywords, 
                "market"."Agents".categories, 
                "market"."Agents".graph,
                "market"."Agents"."submissionStatus",
                "market"."Agents"."submissionDate",
                "market"."Agents".search::text AS search
            FROM "market"."Agents"
            LEFT JOIN "market"."FeaturedAgent" ON "market"."Agents"."id" = "market"."FeaturedAgent"."agentId"
            WHERE ("market"."FeaturedAgent"."agentId" IS NULL OR "market"."FeaturedAgent"."featuredCategories" = '{{}}')
                AND "market"."Agents"."submissionStatus" = 'APPROVED'
            ORDER BY "market"."Agents"."createdAt" DESC
            LIMIT {page_size} OFFSET {page_size * (page - 1)}
            """,
            model=prisma.models.Agents,
        )
        return agents
    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Database query failed: {str(e)}")
    except Exception as e:
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")


async def get_all_categories() -> market.model.CategoriesResponse:
    """
    Retrieve all unique categories from the database.

    Returns:
        CategoriesResponse: A list of unique categories.
    """
    try:
        agents = await prisma.models.Agents.prisma().find_many(distinct=["categories"])

        # Aggregate categories on the Python side
        all_categories = set()
        for agent in agents:
            all_categories.update(agent.categories)

        unique_categories = sorted(list(all_categories))

        return market.model.CategoriesResponse(unique_categories=unique_categories)
    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Database query failed: {str(e)}")
    except Exception as e:
        # Return an empty list of categories in case of unexpected errors
        return market.model.CategoriesResponse(unique_categories=[])


async def create_agent_installed_event(
    event_data: market.model.AgentInstalledFromMarketplaceEventData,
):
    try:
        await prisma.models.InstallTracker.prisma().create(
            data={
                "installedAgentId": event_data.installed_agent_id,
                "marketplaceAgentId": event_data.marketplace_agent_id,
                "installationLocation": prisma.enums.InstallationLocation(
                    event_data.installation_location.name
                ),
            }
        )
    except prisma.errors.PrismaError as e:
        raise AgentQueryError(f"Database query failed: {str(e)}")
    except Exception as e:
        raise AgentQueryError(f"Unexpected error occurred: {str(e)}")
