import logging

import prisma.enums
import prisma.errors
import prisma.models
import prisma.types

import backend.server.v2.store.exceptions
import backend.server.v2.store.model

logger = logging.getLogger(__name__)


async def get_store_agents(
    featured: bool = False,
    creator: str | None = None,
    sorted_by: str | None = None,
    search_query: str | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> backend.server.v2.store.model.StoreAgentsResponse:
    logger.debug(
        f"Getting store agents. featured={featured}, creator={creator}, sorted_by={sorted_by}, search={search_query}, category={category}, page={page}"
    )

    where_clause = {}
    if featured:
        where_clause["featured"] = featured
    if creator:
        where_clause["creator_username"] = creator
    if category:
        where_clause["categories"] = {"has": category}
    if search_query:
        where_clause["OR"] = [
            {"agent_name": {"contains": search_query, "mode": "insensitive"}},
            {"description": {"contains": search_query, "mode": "insensitive"}},
        ]

    order_by = []
    if sorted_by == "rating":
        order_by.append({"rating": "desc"})
    elif sorted_by == "runs":
        order_by.append({"runs": "desc"})
    elif sorted_by == "name":
        order_by.append({"agent_name": "asc"})

    try:
        agents = await prisma.models.StoreAgent.prisma().find_many(
            where=prisma.types.StoreAgentWhereInput(**where_clause),
            order=order_by,
            skip=(page - 1) * page_size,
            take=page_size,
        )

        total = await prisma.models.StoreAgent.prisma().count(
            where=prisma.types.StoreAgentWhereInput(**where_clause)
        )
        total_pages = (total + page_size - 1) // page_size

        store_agents = [
            backend.server.v2.store.model.StoreAgent(
                slug=agent.slug,
                agent_name=agent.agent_name,
                agent_image=agent.agent_image[0] if agent.agent_image else "",
                creator=agent.creator_username,
                creator_avatar=agent.creator_avatar,
                sub_heading=agent.sub_heading,
                description=agent.description,
                runs=agent.runs,
                rating=agent.rating,
            )
            for agent in agents
        ]

        logger.debug(f"Found {len(store_agents)} agents")
        return backend.server.v2.store.model.StoreAgentsResponse(
            agents=store_agents,
            pagination=backend.server.v2.store.model.Pagination(
                current_page=page,
                total_items=total,
                total_pages=total_pages,
                page_size=page_size,
            ),
        )
    except Exception as e:
        logger.error(f"Error getting store agents: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch store agents"
        ) from e


async def get_store_agent_details(
    username: str, agent_name: str
) -> backend.server.v2.store.model.StoreAgentDetails:
    logger.debug(f"Getting store agent details for {username}/{agent_name}")

    try:
        agent = await prisma.models.StoreAgent.prisma().find_first(
            where={"creator_username": username, "slug": agent_name}
        )

        if not agent:
            logger.warning(f"Agent not found: {username}/{agent_name}")
            raise backend.server.v2.store.exceptions.AgentNotFoundError(
                f"Agent {username}/{agent_name} not found"
            )

        logger.debug(f"Found agent details for {username}/{agent_name}")
        return backend.server.v2.store.model.StoreAgentDetails(
            slug=agent.slug,
            agent_name=agent.agent_name,
            agent_video=agent.agent_video or "",
            agent_image=agent.agent_image,
            creator=agent.creator_username,
            creator_avatar=agent.creator_avatar,
            sub_heading=agent.sub_heading,
            description=agent.description,
            categories=agent.categories,
            runs=agent.runs,
            rating=agent.rating,
            versions=agent.versions,
            last_updated=agent.updated_at,
        )
    except backend.server.v2.store.exceptions.AgentNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error getting store agent details: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch agent details"
        ) from e


async def get_store_creators(
    featured: bool = False,
    search_query: str | None = None,
    sorted_by: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> backend.server.v2.store.model.CreatorsResponse:
    logger.debug(
        f"Getting store creators. featured={featured}, search={search_query}, sorted_by={sorted_by}, page={page}"
    )

    # Build where clause
    where = {}

    # Add search filter if provided
    if search_query:
        where["OR"] = [
            {"username": {"contains": search_query, "mode": "insensitive"}},
            {"name": {"contains": search_query, "mode": "insensitive"}},
            {"description": {"contains": search_query, "mode": "insensitive"}},
        ]

    try:
        # Get total count for pagination
        total = await prisma.models.Creator.prisma().count(
            where=prisma.types.CreatorWhereInput(**where)
        )
        total_pages = (total + page_size - 1) // page_size

        # Add pagination
        skip = (page - 1) * page_size
        take = page_size

        # Add sorting
        order = []
        if sorted_by == "agent_rating":
            order.append({"agent_rating": "desc"})
        elif sorted_by == "agent_runs":
            order.append({"agent_runs": "desc"})
        elif sorted_by == "num_agents":
            order.append({"num_agents": "desc"})
        else:
            order.append({"username": "asc"})

        # Execute query
        creators = await prisma.models.Creator.prisma().find_many(
            where=prisma.types.CreatorWhereInput(**where),
            skip=skip,
            take=take,
            order=order,
        )

        # Convert to response model
        creator_models = [
            backend.server.v2.store.model.Creator(
                username=creator.username,
                name=creator.name,
                description=creator.description,
                avatar_url=creator.avatar_url,
                num_agents=creator.num_agents,
            )
            for creator in creators
        ]

        logger.debug(f"Found {len(creator_models)} creators")
        return backend.server.v2.store.model.CreatorsResponse(
            creators=creator_models,
            pagination=backend.server.v2.store.model.Pagination(
                current_page=page,
                total_items=total,
                total_pages=total_pages,
                page_size=page_size,
            ),
        )
    except Exception as e:
        logger.error(f"Error getting store creators: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch store creators"
        ) from e


async def get_store_creator_details(
    username: str,
) -> backend.server.v2.store.model.CreatorDetails:
    logger.debug(f"Getting store creator details for {username}")

    try:
        # Query creator details from database
        creator = await prisma.models.Creator.prisma().find_unique(
            where={"username": username}
        )

        if not creator:
            logger.warning(f"Creator not found: {username}")
            raise backend.server.v2.store.exceptions.CreatorNotFoundError(
                f"Creator {username} not found"
            )

        logger.debug(f"Found creator details for {username}")
        return backend.server.v2.store.model.CreatorDetails(
            name=creator.name,
            username=creator.username,
            description=creator.description,
            links=creator.links,
            avatar_url=creator.avatar_url,
            agent_rating=creator.agent_rating,
            agent_runs=creator.agent_runs,
            top_categories=creator.top_categories,
        )
    except backend.server.v2.store.exceptions.CreatorNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error getting store creator details: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch creator details"
        ) from e


async def get_store_submissions(
    user_id: str, page: int = 1, page_size: int = 20
) -> backend.server.v2.store.model.StoreSubmissionsResponse:
    logger.debug(f"Getting store submissions for user {user_id}, page={page}")

    try:
        # Calculate pagination values
        skip = (page - 1) * page_size

        where = prisma.types.StoreSubmissionWhereInput(user_id=user_id)
        # Query submissions from database
        submissions = await prisma.models.StoreSubmission.prisma().find_many(
            where=where, skip=skip, take=page_size, order=[{"date_submitted": "desc"}]
        )

        # Get total count for pagination
        total = await prisma.models.StoreSubmission.prisma().count(where=where)

        total_pages = (total + page_size - 1) // page_size

        # Convert to response models
        submission_models = [
            backend.server.v2.store.model.StoreSubmission(
                name=sub.name,
                description=sub.description,
                image_urls=sub.image_urls or [],
                date_submitted=sub.date_submitted,
                status=sub.status,
                runs=sub.runs or 0,
                rating=sub.rating or 0.0,
            )
            for sub in submissions
        ]

        logger.debug(f"Found {len(submission_models)} submissions")
        return backend.server.v2.store.model.StoreSubmissionsResponse(
            submissions=submission_models,
            pagination=backend.server.v2.store.model.Pagination(
                current_page=page,
                total_items=total,
                total_pages=total_pages,
                page_size=page_size,
            ),
        )

    except Exception as e:
        logger.error(f"Error fetching store submissions: {str(e)}")
        # Return empty response rather than exposing internal errors
        return backend.server.v2.store.model.StoreSubmissionsResponse(
            submissions=[],
            pagination=backend.server.v2.store.model.Pagination(
                current_page=page,
                total_items=0,
                total_pages=0,
                page_size=page_size,
            ),
        )


async def create_store_submission(
    user_id: str,
    agent_id: str,
    agent_version: int,
    slug: str,
    name: str,
    video_url: str | None = None,
    image_urls: list[str] = [],
    description: str = "",
    sub_heading: str = "",
    categories: list[str] = [],
) -> backend.server.v2.store.model.StoreSubmission:
    """
    Create a new store listing submission.

    Args:
        user_id: ID of the authenticated user submitting the listing
        agent_id: ID of the agent being submitted
        agent_version: Version of the agent being submitted
        slug: URL slug for the listing
        name: Name of the agent
        video_url: Optional URL to video demo
        image_urls: List of image URLs for the listing
        description: Description of the agent
        categories: List of categories for the agent

    Returns:
        StoreSubmission: The created store submission
    """
    logger.debug(
        f"Creating store submission for user {user_id}, agent {agent_id} v{agent_version}"
    )

    try:
        # First verify the agent belongs to this user
        agent = await prisma.models.AgentGraph.prisma().find_first(
            where=prisma.types.AgentGraphWhereInput(
                id=agent_id, version=agent_version, userId=user_id
            )
        )

        if not agent:
            logger.warning(
                f"Agent not found for user {user_id}: {agent_id} v{agent_version}"
            )
            raise backend.server.v2.store.exceptions.AgentNotFoundError(
                "Agent not found for this user"
            )

        listing = await prisma.models.StoreListing.prisma().find_first(
            where=prisma.types.StoreListingWhereInput(
                agentId=agent_id, owningUserId=user_id
            )
        )
        if listing is not None:
            logger.warning(f"Listing already exists for agent {agent_id}")
            raise backend.server.v2.store.exceptions.ListingExistsError(
                "Listing already exists for this agent"
            )

        # Create the store listing
        listing = await prisma.models.StoreListing.prisma().create(
            data={
                "agentId": agent_id,
                "agentVersion": agent_version,
                "owningUserId": user_id,
                "StoreListingVersions": {
                    "create": {
                        "agentId": agent_id,
                        "agentVersion": agent_version,
                        "slug": slug,
                        "name": name,
                        "videoUrl": video_url,
                        "imageUrls": image_urls,
                        "description": description,
                        "categories": categories,
                        "subHeading": sub_heading,
                    }
                },
            }
        )

        logger.debug(f"Created store listing for agent {agent_id}")
        # Return submission details
        return backend.server.v2.store.model.StoreSubmission(
            name=name,
            description=description,
            image_urls=image_urls,
            date_submitted=listing.createdAt,
            status=prisma.enums.SubmissionStatus.PENDING,
            runs=0,
            rating=0.0,
        )

    except (
        backend.server.v2.store.exceptions.AgentNotFoundError,
        backend.server.v2.store.exceptions.ListingExistsError,
    ):
        raise
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error creating store submission: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to create store submission"
        ) from e


async def get_user_profile(
    user_id: str,
) -> backend.server.v2.store.model.ProfileDetails:
    logger.debug(f"Getting user profile for {user_id}")

    try:
        profile = await prisma.models.Profile.prisma().find_unique(
            where={"userId": user_id}  # type: ignore
        )

        if not profile:
            logger.warning(f"Profile not found for user {user_id}")
            raise backend.server.v2.store.exceptions.ProfileNotFoundError(
                f"Profile not found for user {user_id}"
            )

        return backend.server.v2.store.model.ProfileDetails(
            name=profile.name,
            username=profile.username,
            description=profile.description,
            links=profile.links,
            avatar_url=profile.avatarUrl,
        )
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        return backend.server.v2.store.model.ProfileDetails(
            name="No Profile Data",
            username="No Profile Data",
            description="No Profile Data",
            links=[],
            avatar_url="",
        )


async def update_or_create_profile(
    user_id: str, profile: backend.server.v2.store.model.CreatorDetails
) -> backend.server.v2.store.model.CreatorDetails:
    """
    Update the store profile for a user. Creates a new profile if one doesn't exist.
    Only allows updating if the user_id matches the owning user.

    Args:
        user_id: ID of the authenticated user
        profile: Updated profile details

    Returns:
        CreatorDetails: The updated profile

    Raises:
        HTTPException: If user is not authorized to update this profile
    """
    logger.debug(f"Updating profile for user {user_id}")

    try:
        # Check if profile exists for user
        existing_profile = await prisma.models.Profile.prisma().find_first(
            where={"userId": user_id}
        )

        # If no profile exists, create a new one
        if not existing_profile:
            logger.debug(f"Creating new profile for user {user_id}")
            # Create new profile since one doesn't exist
            new_profile = await prisma.models.Profile.prisma().create(
                data={
                    "userId": user_id,
                    "name": profile.name,
                    "username": profile.username,
                    "description": profile.description,
                    "links": profile.links,
                    "avatarUrl": profile.avatar_url,
                }
            )

            return backend.server.v2.store.model.CreatorDetails(
                name=new_profile.name,
                username=new_profile.username,
                description=new_profile.description,
                links=new_profile.links,
                avatar_url=new_profile.avatarUrl or "",
                agent_rating=0.0,
                agent_runs=0,
                top_categories=[],
            )
        else:
            logger.debug(f"Updating existing profile for user {user_id}")
            # Update the existing profile
            updated_profile = await prisma.models.Profile.prisma().update(
                where={"id": existing_profile.id},
                data=prisma.types.ProfileUpdateInput(
                    username=profile.username,
                    description=profile.description,
                    links=profile.links,
                    avatarUrl=profile.avatar_url,
                ),
            )
            if updated_profile is None:
                logger.error(f"Failed to update profile for user {user_id}")
                raise backend.server.v2.store.exceptions.DatabaseError(
                    "Failed to update profile"
                )

            return backend.server.v2.store.model.CreatorDetails(
                name=updated_profile.name,
                username=updated_profile.username,
                description=updated_profile.description,
                links=updated_profile.links,
                avatar_url=updated_profile.avatarUrl or "",
                agent_rating=0.0,
                agent_runs=0,
                top_categories=[],
            )

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error updating profile: {str(e)}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to update profile"
        ) from e
