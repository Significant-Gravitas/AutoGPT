import logging
from datetime import datetime, timezone

import fastapi
import prisma.enums
import prisma.errors
import prisma.models
import prisma.types

import backend.server.v2.store.exceptions
import backend.server.v2.store.model
from backend.data.graph import (
    GraphMeta,
    GraphModel,
    get_graph,
    get_graph_as_admin,
    get_sub_graphs,
)
from backend.data.includes import AGENT_GRAPH_INCLUDE

logger = logging.getLogger(__name__)


def sanitize_query(query: str | None) -> str | None:
    if query is None:
        return query
    query = query.strip()[:100]
    return (
        query.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace(";", "\\;")
        .replace("--", "\\--")
        .replace("/*", "\\/*")
        .replace("*/", "\\*/")
    )


async def get_store_agents(
    featured: bool = False,
    creator: str | None = None,
    sorted_by: str | None = None,
    search_query: str | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> backend.server.v2.store.model.StoreAgentsResponse:
    """
    Get PUBLIC store agents from the StoreAgent view
    """
    logger.debug(
        f"Getting store agents. featured={featured}, creator={creator}, sorted_by={sorted_by}, search={search_query}, category={category}, page={page}"
    )
    sanitized_query = sanitize_query(search_query)

    where_clause = {}
    if featured:
        where_clause["featured"] = featured
    if creator:
        where_clause["creator_username"] = creator
    if category:
        where_clause["categories"] = {"has": category}

    if sanitized_query:
        where_clause["OR"] = [
            {"agent_name": {"contains": sanitized_query, "mode": "insensitive"}},
            {"description": {"contains": sanitized_query, "mode": "insensitive"}},
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

        store_agents: list[backend.server.v2.store.model.StoreAgent] = []
        for agent in agents:
            try:
                # Create the StoreAgent object safely
                store_agent = backend.server.v2.store.model.StoreAgent(
                    slug=agent.slug,
                    agent_name=agent.agent_name,
                    agent_image=agent.agent_image[0] if agent.agent_image else "",
                    creator=agent.creator_username or "Needs Profile",
                    creator_avatar=agent.creator_avatar or "",
                    sub_heading=agent.sub_heading,
                    description=agent.description,
                    runs=agent.runs,
                    rating=agent.rating,
                )
                # Add to the list only if creation was successful
                store_agents.append(store_agent)
            except Exception as e:
                # Skip this agent if there was an error
                # You could log the error here if needed
                logger.error(
                    f"Error parsing Store agent when getting store agents from db: {e}"
                )
                continue

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
        logger.error(f"Error getting store agents: {e}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch store agents"
        ) from e


async def get_store_agent_details(
    username: str, agent_name: str
) -> backend.server.v2.store.model.StoreAgentDetails:
    """Get PUBLIC store agent details from the StoreAgent view"""
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

        profile = await prisma.models.Profile.prisma().find_first(
            where={"username": username}
        )
        user_id = profile.userId if profile else None

        # Retrieve StoreListing to get active_version_id and has_approved_version
        store_listing = await prisma.models.StoreListing.prisma().find_first(
            where=prisma.types.StoreListingWhereInput(
                slug=agent_name,
                owningUserId=user_id or "",
            ),
            include={"ActiveVersion": True},
        )

        active_version_id = store_listing.activeVersionId if store_listing else None
        has_approved_version = (
            store_listing.hasApprovedVersion if store_listing else False
        )

        logger.debug(f"Found agent details for {username}/{agent_name}")
        return backend.server.v2.store.model.StoreAgentDetails(
            store_listing_version_id=agent.storeListingVersionId,
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
            active_version_id=active_version_id,
            has_approved_version=has_approved_version,
        )
    except backend.server.v2.store.exceptions.AgentNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error getting store agent details: {e}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch agent details"
        ) from e


async def get_available_graph(store_listing_version_id: str) -> GraphMeta:
    try:
        # Get avaialble, non-deleted store listing version
        store_listing_version = (
            await prisma.models.StoreListingVersion.prisma().find_first(
                where={
                    "id": store_listing_version_id,
                    "isAvailable": True,
                    "isDeleted": False,
                },
                include={"AgentGraph": {"include": {"Nodes": True}}},
            )
        )

        if not store_listing_version or not store_listing_version.AgentGraph:
            raise fastapi.HTTPException(
                status_code=404,
                detail=f"Store listing version {store_listing_version_id} not found",
            )

        return GraphModel.from_db(store_listing_version.AgentGraph).meta()

    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch agent"
        ) from e


async def get_store_agent_by_version_id(
    store_listing_version_id: str,
) -> backend.server.v2.store.model.StoreAgentDetails:
    logger.debug(f"Getting store agent details for {store_listing_version_id}")

    try:
        agent = await prisma.models.StoreAgent.prisma().find_first(
            where={"storeListingVersionId": store_listing_version_id}
        )

        if not agent:
            logger.warning(f"Agent not found: {store_listing_version_id}")
            raise backend.server.v2.store.exceptions.AgentNotFoundError(
                f"Agent {store_listing_version_id} not found"
            )

        logger.debug(f"Found agent details for {store_listing_version_id}")
        return backend.server.v2.store.model.StoreAgentDetails(
            store_listing_version_id=agent.storeListingVersionId,
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
        logger.error(f"Error getting store agent details: {e}")
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
    """Get PUBLIC store creators from the Creator view"""
    logger.debug(
        f"Getting store creators. featured={featured}, search={search_query}, sorted_by={sorted_by}, page={page}"
    )

    # Build where clause with sanitized inputs
    where = {}

    if featured:
        where["is_featured"] = featured

    # Add search filter if provided, using parameterized queries
    if search_query:
        # Sanitize and validate search query by escaping special characters
        sanitized_query = search_query.strip()
        if not sanitized_query or len(sanitized_query) > 100:  # Reasonable length limit
            raise backend.server.v2.store.exceptions.DatabaseError(
                "Invalid search query"
            )

        # Escape special SQL characters
        sanitized_query = (
            sanitized_query.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace(";", "\\;")
            .replace("--", "\\--")
            .replace("/*", "\\/*")
            .replace("*/", "\\*/")
        )

        where["OR"] = [
            {"username": {"contains": sanitized_query, "mode": "insensitive"}},
            {"name": {"contains": sanitized_query, "mode": "insensitive"}},
            {"description": {"contains": sanitized_query, "mode": "insensitive"}},
        ]

    try:
        # Validate pagination parameters
        if not isinstance(page, int) or page < 1:
            raise backend.server.v2.store.exceptions.DatabaseError(
                "Invalid page number"
            )
        if not isinstance(page_size, int) or page_size < 1 or page_size > 100:
            raise backend.server.v2.store.exceptions.DatabaseError("Invalid page size")

        # Get total count for pagination using sanitized where clause
        total = await prisma.models.Creator.prisma().count(
            where=prisma.types.CreatorWhereInput(**where)
        )
        total_pages = (total + page_size - 1) // page_size

        # Add pagination with validated parameters
        skip = (page - 1) * page_size
        take = page_size

        # Add sorting with validated sort parameter
        order = []
        valid_sort_fields = {"agent_rating", "agent_runs", "num_agents"}
        if sorted_by in valid_sort_fields:
            order.append({sorted_by: "desc"})
        else:
            order.append({"username": "asc"})

        # Execute query with sanitized parameters
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
                agent_rating=creator.agent_rating,
                agent_runs=creator.agent_runs,
                is_featured=creator.is_featured,
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
        logger.error(f"Error getting store creators: {e}")
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
        logger.error(f"Error getting store creator details: {e}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch creator details"
        ) from e


async def get_store_submissions(
    user_id: str, page: int = 1, page_size: int = 20
) -> backend.server.v2.store.model.StoreSubmissionsResponse:
    """Get store submissions for the authenticated user -- not an admin"""
    logger.debug(f"Getting store submissions for user {user_id}, page={page}")

    try:
        # Calculate pagination values
        skip = (page - 1) * page_size

        where = prisma.types.StoreSubmissionWhereInput(user_id=user_id)
        # Query submissions from database
        submissions = await prisma.models.StoreSubmission.prisma().find_many(
            where=where,
            skip=skip,
            take=page_size,
            order=[{"date_submitted": "desc"}],
        )

        # Get total count for pagination
        total = await prisma.models.StoreSubmission.prisma().count(where=where)

        total_pages = (total + page_size - 1) // page_size

        # Convert to response models
        submission_models = []
        for sub in submissions:
            submission_model = backend.server.v2.store.model.StoreSubmission(
                agent_id=sub.agent_id,
                agent_version=sub.agent_version,
                name=sub.name,
                sub_heading=sub.sub_heading,
                slug=sub.slug,
                description=sub.description,
                image_urls=sub.image_urls or [],
                date_submitted=sub.date_submitted or datetime.now(tz=timezone.utc),
                status=sub.status,
                runs=sub.runs or 0,
                rating=sub.rating or 0.0,
                store_listing_version_id=sub.store_listing_version_id,
                reviewer_id=sub.reviewer_id,
                review_comments=sub.review_comments,
                # internal_comments omitted for regular users
                reviewed_at=sub.reviewed_at,
                changes_summary=sub.changes_summary,
            )
            submission_models.append(submission_model)

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
        logger.error(f"Error fetching store submissions: {e}")
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


async def delete_store_submission(
    user_id: str,
    submission_id: str,
) -> bool:
    """
    Delete a store listing submission as the submitting user.

    Args:
        user_id: ID of the authenticated user
        submission_id: ID of the submission to be deleted

    Returns:
        bool: True if the submission was successfully deleted, False otherwise
    """
    logger.debug(f"Deleting store submission {submission_id} for user {user_id}")

    try:
        # Verify the submission belongs to this user
        submission = await prisma.models.StoreListing.prisma().find_first(
            where={"agentGraphId": submission_id, "owningUserId": user_id}
        )

        if not submission:
            logger.warning(f"Submission not found for user {user_id}: {submission_id}")
            raise backend.server.v2.store.exceptions.SubmissionNotFoundError(
                f"Submission not found for this user. User ID: {user_id}, Submission ID: {submission_id}"
            )

        # Delete the submission
        await prisma.models.StoreListing.prisma().delete(where={"id": submission.id})

        logger.debug(
            f"Successfully deleted submission {submission_id} for user {user_id}"
        )
        return True

    except Exception as e:
        logger.error(f"Error deleting store submission: {e}")
        return False


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
    changes_summary: str = "Initial Submission",
) -> backend.server.v2.store.model.StoreSubmission:
    """
    Create the first (and only) store listing and thus submission as a normal user

    Args:
        user_id: ID of the authenticated user submitting the listing
        agent_id: ID of the agent being submitted
        agent_version: Version of the agent being submitted
        slug: URL slug for the listing
        name: Name of the agent
        video_url: Optional URL to video demo
        image_urls: List of image URLs for the listing
        description: Description of the agent
        sub_heading: Optional sub-heading for the agent
        categories: List of categories for the agent
        changes_summary: Summary of changes made in this submission

    Returns:
        StoreSubmission: The created store submission
    """
    logger.debug(
        f"Creating store submission for user {user_id}, agent {agent_id} v{agent_version}"
    )

    try:
        # Sanitize slug to only allow letters and hyphens
        slug = "".join(
            c if c.isalpha() or c == "-" or c.isnumeric() else "" for c in slug
        ).lower()

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
                f"Agent not found for this user. User ID: {user_id}, Agent ID: {agent_id}, Version: {agent_version}"
            )

        # Check if listing already exists for this agent
        existing_listing = await prisma.models.StoreListing.prisma().find_first(
            where=prisma.types.StoreListingWhereInput(
                agentGraphId=agent_id, owningUserId=user_id
            )
        )

        if existing_listing is not None:
            logger.info(
                f"Listing already exists for agent {agent_id}, creating new version instead"
            )

            # Delegate to create_store_version which already handles this case correctly
            return await create_store_version(
                user_id=user_id,
                agent_id=agent_id,
                agent_version=agent_version,
                store_listing_id=existing_listing.id,
                name=name,
                video_url=video_url,
                image_urls=image_urls,
                description=description,
                sub_heading=sub_heading,
                categories=categories,
                changes_summary=changes_summary,
            )

        # If no existing listing, create a new one
        data = prisma.types.StoreListingCreateInput(
            slug=slug,
            agentGraphId=agent_id,
            agentGraphVersion=agent_version,
            owningUserId=user_id,
            createdAt=datetime.now(tz=timezone.utc),
            Versions={
                "create": [
                    prisma.types.StoreListingVersionCreateInput(
                        agentGraphId=agent_id,
                        agentGraphVersion=agent_version,
                        name=name,
                        videoUrl=video_url,
                        imageUrls=image_urls,
                        description=description,
                        categories=categories,
                        subHeading=sub_heading,
                        submissionStatus=prisma.enums.SubmissionStatus.PENDING,
                        submittedAt=datetime.now(tz=timezone.utc),
                        changesSummary=changes_summary,
                    )
                ]
            },
        )
        listing = await prisma.models.StoreListing.prisma().create(
            data=data,
            include=prisma.types.StoreListingInclude(Versions=True),
        )

        store_listing_version_id = (
            listing.Versions[0].id
            if listing.Versions is not None and len(listing.Versions) > 0
            else None
        )

        logger.debug(f"Created store listing for agent {agent_id}")
        # Return submission details
        return backend.server.v2.store.model.StoreSubmission(
            agent_id=agent_id,
            agent_version=agent_version,
            name=name,
            slug=slug,
            sub_heading=sub_heading,
            description=description,
            image_urls=image_urls,
            date_submitted=listing.createdAt,
            status=prisma.enums.SubmissionStatus.PENDING,
            runs=0,
            rating=0.0,
            store_listing_version_id=store_listing_version_id,
            changes_summary=changes_summary,
        )

    except (
        backend.server.v2.store.exceptions.AgentNotFoundError,
        backend.server.v2.store.exceptions.ListingExistsError,
    ):
        raise
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error creating store submission: {e}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to create store submission"
        ) from e


async def create_store_version(
    user_id: str,
    agent_id: str,
    agent_version: int,
    store_listing_id: str,
    name: str,
    video_url: str | None = None,
    image_urls: list[str] = [],
    description: str = "",
    sub_heading: str = "",
    categories: list[str] = [],
    changes_summary: str = "Update Submission",
) -> backend.server.v2.store.model.StoreSubmission:
    """
    Create a new version for an existing store listing

    Args:
        user_id: ID of the authenticated user submitting the version
        agent_id: ID of the agent being submitted
        agent_version: Version of the agent being submitted
        store_listing_id: ID of the existing store listing
        name: Name of the agent
        video_url: Optional URL to video demo
        image_urls: List of image URLs for the listing
        description: Description of the agent
        categories: List of categories for the agent
        changes_summary: Summary of changes from the previous version

    Returns:
        StoreSubmission: The created store submission
    """
    logger.debug(
        f"Creating new version for store listing {store_listing_id} for user {user_id}, agent {agent_id} v{agent_version}"
    )

    try:
        # First verify the listing belongs to this user
        listing = await prisma.models.StoreListing.prisma().find_first(
            where=prisma.types.StoreListingWhereInput(
                id=store_listing_id, owningUserId=user_id
            ),
            include={"Versions": {"order_by": {"version": "desc"}, "take": 1}},
        )

        if not listing:
            raise backend.server.v2.store.exceptions.ListingNotFoundError(
                f"Store listing not found. User ID: {user_id}, Listing ID: {store_listing_id}"
            )

        # Verify the agent belongs to this user
        agent = await prisma.models.AgentGraph.prisma().find_first(
            where=prisma.types.AgentGraphWhereInput(
                id=agent_id, version=agent_version, userId=user_id
            )
        )

        if not agent:
            raise backend.server.v2.store.exceptions.AgentNotFoundError(
                f"Agent not found for this user. User ID: {user_id}, Agent ID: {agent_id}, Version: {agent_version}"
            )

        # Get the latest version number
        latest_version = listing.Versions[0] if listing.Versions else None

        next_version = (latest_version.version + 1) if latest_version else 1

        # Create a new version for the existing listing
        new_version = await prisma.models.StoreListingVersion.prisma().create(
            data=prisma.types.StoreListingVersionCreateInput(
                version=next_version,
                agentGraphId=agent_id,
                agentGraphVersion=agent_version,
                name=name,
                videoUrl=video_url,
                imageUrls=image_urls,
                description=description,
                categories=categories,
                subHeading=sub_heading,
                submissionStatus=prisma.enums.SubmissionStatus.PENDING,
                submittedAt=datetime.now(),
                changesSummary=changes_summary,
                storeListingId=store_listing_id,
            )
        )

        logger.debug(
            f"Created new version for listing {store_listing_id} of agent {agent_id}"
        )
        # Return submission details
        return backend.server.v2.store.model.StoreSubmission(
            agent_id=agent_id,
            agent_version=agent_version,
            name=name,
            slug=listing.slug,
            sub_heading=sub_heading,
            description=description,
            image_urls=image_urls,
            date_submitted=datetime.now(),
            status=prisma.enums.SubmissionStatus.PENDING,
            runs=0,
            rating=0.0,
            store_listing_version_id=new_version.id,
            changes_summary=changes_summary,
            version=next_version,
        )

    except prisma.errors.PrismaError as e:
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to create new store version"
        ) from e


async def create_store_review(
    user_id: str,
    store_listing_version_id: str,
    score: int,
    comments: str | None = None,
) -> backend.server.v2.store.model.StoreReview:
    """Create a review for a store listing as a user to detail their experience"""
    try:
        data = prisma.types.StoreListingReviewUpsertInput(
            update=prisma.types.StoreListingReviewUpdateInput(
                score=score,
                comments=comments,
            ),
            create=prisma.types.StoreListingReviewCreateInput(
                reviewByUserId=user_id,
                storeListingVersionId=store_listing_version_id,
                score=score,
                comments=comments,
            ),
        )
        review = await prisma.models.StoreListingReview.prisma().upsert(
            where={
                "storeListingVersionId_reviewByUserId": {
                    "storeListingVersionId": store_listing_version_id,
                    "reviewByUserId": user_id,
                }
            },
            data=data,
        )

        return backend.server.v2.store.model.StoreReview(
            score=review.score,
            comments=review.comments,
        )

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error creating store review: {e}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to create store review"
        ) from e


async def get_user_profile(
    user_id: str,
) -> backend.server.v2.store.model.ProfileDetails | None:
    logger.debug(f"Getting user profile for {user_id}")

    try:
        profile = await prisma.models.Profile.prisma().find_first(
            where={"userId": user_id}
        )

        if not profile:
            return None
        return backend.server.v2.store.model.ProfileDetails(
            name=profile.name,
            username=profile.username,
            description=profile.description,
            links=profile.links,
            avatar_url=profile.avatarUrl,
        )
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to get user profile"
        ) from e


async def update_profile(
    user_id: str, profile: backend.server.v2.store.model.Profile
) -> backend.server.v2.store.model.CreatorDetails:
    """
    Update the store profile for a user or create a new one if it doesn't exist.
    Args:
        user_id: ID of the authenticated user
        profile: Updated profile details
    Returns:
        CreatorDetails: The updated or created profile details
    Raises:
        DatabaseError: If there's an issue updating or creating the profile
    """
    logger.info(f"Updating profile for user {user_id} with data: {profile}")
    try:
        # Sanitize username to allow only letters, numbers, and hyphens
        username = "".join(
            c if c.isalpha() or c == "-" or c.isnumeric() else ""
            for c in profile.username
        ).lower()
        # Check if profile exists for the given user_id
        existing_profile = await prisma.models.Profile.prisma().find_first(
            where={"userId": user_id}
        )
        if not existing_profile:
            raise backend.server.v2.store.exceptions.ProfileNotFoundError(
                f"Profile not found for user {user_id}. This should not be possible."
            )

        # Verify that the user is authorized to update this profile
        if existing_profile.userId != user_id:
            logger.error(
                f"Unauthorized update attempt for profile {existing_profile.id} by user {user_id}"
            )
            raise backend.server.v2.store.exceptions.DatabaseError(
                f"Unauthorized update attempt for profile {existing_profile.id} by user {user_id}"
            )

        logger.debug(f"Updating existing profile for user {user_id}")
        # Prepare update data, only including non-None values
        update_data = {}
        if profile.name is not None:
            update_data["name"] = profile.name
        if profile.username is not None:
            update_data["username"] = username
        if profile.description is not None:
            update_data["description"] = profile.description
        if profile.links is not None:
            update_data["links"] = profile.links
        if profile.avatar_url is not None:
            update_data["avatarUrl"] = profile.avatar_url

        # Update the existing profile
        updated_profile = await prisma.models.Profile.prisma().update(
            where={"id": existing_profile.id},
            data=prisma.types.ProfileUpdateInput(**update_data),
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
        logger.error(f"Database error updating profile: {e}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to update profile"
        ) from e


async def get_my_agents(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
) -> backend.server.v2.store.model.MyAgentsResponse:
    """Get the agents for the authenticated user"""
    logger.debug(f"Getting my agents for user {user_id}, page={page}")

    try:
        search_filter: prisma.types.LibraryAgentWhereInput = {
            "userId": user_id,
            "AgentGraph": {"is": {"StoreListings": {"none": {"isDeleted": False}}}},
            "isArchived": False,
            "isDeleted": False,
        }

        library_agents = await prisma.models.LibraryAgent.prisma().find_many(
            where=search_filter,
            order=[{"updatedAt": "desc"}],
            skip=(page - 1) * page_size,
            take=page_size,
            include={"AgentGraph": True},
        )

        total = await prisma.models.LibraryAgent.prisma().count(where=search_filter)
        total_pages = (total + page_size - 1) // page_size

        my_agents = [
            backend.server.v2.store.model.MyAgent(
                agent_id=graph.id,
                agent_version=graph.version,
                agent_name=graph.name or "",
                last_edited=graph.updatedAt or graph.createdAt,
                description=graph.description or "",
                agent_image=library_agent.imageUrl,
            )
            for library_agent in library_agents
            if (graph := library_agent.AgentGraph)
        ]

        return backend.server.v2.store.model.MyAgentsResponse(
            agents=my_agents,
            pagination=backend.server.v2.store.model.Pagination(
                current_page=page,
                total_items=total,
                total_pages=total_pages,
                page_size=page_size,
            ),
        )
    except Exception as e:
        logger.error(f"Error getting my agents: {e}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to fetch my agents"
        ) from e


async def get_agent(
    user_id: str | None,
    store_listing_version_id: str,
) -> GraphModel:
    """Get agent using the version ID and store listing version ID."""
    store_listing_version = (
        await prisma.models.StoreListingVersion.prisma().find_unique(
            where={"id": store_listing_version_id}
        )
    )

    if not store_listing_version:
        raise ValueError(f"Store listing version {store_listing_version_id} not found")

    graph = await get_graph(
        user_id=user_id,
        graph_id=store_listing_version.agentGraphId,
        version=store_listing_version.agentGraphVersion,
        for_export=True,
    )
    if not graph:
        raise ValueError(
            f"Agent {store_listing_version.agentGraphId} v{store_listing_version.agentGraphVersion} not found"
        )

    return graph


#####################################################
################## ADMIN FUNCTIONS ##################
#####################################################


async def _get_missing_sub_store_listing(
    graph: prisma.models.AgentGraph,
) -> list[prisma.models.AgentGraph]:
    """
    Agent graph can have sub-graphs, and those sub-graphs also need to be store listed.
    This method fetches the sub-graphs, and returns the ones not listed in the store.
    """
    sub_graphs = await get_sub_graphs(graph)
    if not sub_graphs:
        return []

    # Fetch all the sub-graphs that are listed, and return the ones missing.
    store_listed_sub_graphs = {
        (listing.agentGraphId, listing.agentGraphVersion)
        for listing in await prisma.models.StoreListingVersion.prisma().find_many(
            where={
                "OR": [
                    {
                        "agentGraphId": sub_graph.id,
                        "agentGraphVersion": sub_graph.version,
                    }
                    for sub_graph in sub_graphs
                ],
                "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
                "isDeleted": False,
            }
        )
    }

    return [
        sub_graph
        for sub_graph in sub_graphs
        if (sub_graph.id, sub_graph.version) not in store_listed_sub_graphs
    ]


async def review_store_submission(
    store_listing_version_id: str,
    is_approved: bool,
    external_comments: str,
    internal_comments: str,
    reviewer_id: str,
) -> backend.server.v2.store.model.StoreSubmission:
    """Review a store listing submission as an admin."""
    try:
        store_listing_version = (
            await prisma.models.StoreListingVersion.prisma().find_unique(
                where={"id": store_listing_version_id},
                include={
                    "StoreListing": True,
                    "AgentGraph": {"include": AGENT_GRAPH_INCLUDE},
                },
            )
        )

        if not store_listing_version or not store_listing_version.StoreListing:
            raise fastapi.HTTPException(
                status_code=404,
                detail=f"Store listing version {store_listing_version_id} not found",
            )

        # If approving, update the listing to indicate it has an approved version
        if is_approved and store_listing_version.AgentGraph:
            heading = f"Sub-graph of {store_listing_version.name}v{store_listing_version.agentGraphVersion}"

            sub_store_listing_versions = [
                prisma.types.StoreListingVersionCreateWithoutRelationsInput(
                    agentGraphId=sub_graph.id,
                    agentGraphVersion=sub_graph.version,
                    name=sub_graph.name or heading,
                    submissionStatus=prisma.enums.SubmissionStatus.APPROVED,
                    subHeading=heading,
                    description=f"{heading}: {sub_graph.description}",
                    changesSummary=f"This listing is added as a {heading} / #{store_listing_version.agentGraphId}.",
                    isAvailable=False,  # Hide sub-graphs from the store by default.
                    submittedAt=datetime.now(tz=timezone.utc),
                )
                for sub_graph in await _get_missing_sub_store_listing(
                    store_listing_version.AgentGraph
                )
            ]

            await prisma.models.StoreListing.prisma().update(
                where={"id": store_listing_version.StoreListing.id},
                data={
                    "hasApprovedVersion": True,
                    "ActiveVersion": {"connect": {"id": store_listing_version_id}},
                    "Versions": {"create": sub_store_listing_versions},
                },
            )

        submission_status = (
            prisma.enums.SubmissionStatus.APPROVED
            if is_approved
            else prisma.enums.SubmissionStatus.REJECTED
        )

        # Update the version with review information
        update_data: prisma.types.StoreListingVersionUpdateInput = {
            "submissionStatus": submission_status,
            "reviewComments": external_comments,
            "internalComments": internal_comments,
            "Reviewer": {"connect": {"id": reviewer_id}},
            "StoreListing": {"connect": {"id": store_listing_version.StoreListing.id}},
            "reviewedAt": datetime.now(tz=timezone.utc),
        }

        # Update the version
        submission = await prisma.models.StoreListingVersion.prisma().update(
            where={"id": store_listing_version_id},
            data=update_data,
            include={"StoreListing": True},
        )

        if not submission:
            raise backend.server.v2.store.exceptions.DatabaseError(
                f"Failed to update store listing version {store_listing_version_id}"
            )

        # Convert to Pydantic model for consistency
        return backend.server.v2.store.model.StoreSubmission(
            agent_id=submission.agentGraphId,
            agent_version=submission.agentGraphVersion,
            name=submission.name,
            sub_heading=submission.subHeading,
            slug=(
                submission.StoreListing.slug
                if hasattr(submission, "storeListing") and submission.StoreListing
                else ""
            ),
            description=submission.description,
            image_urls=submission.imageUrls or [],
            date_submitted=submission.submittedAt or submission.createdAt,
            status=submission.submissionStatus,
            runs=0,  # Default values since we don't have this data here
            rating=0.0,
            store_listing_version_id=submission.id,
            reviewer_id=submission.reviewerId,
            review_comments=submission.reviewComments,
            internal_comments=submission.internalComments,
            reviewed_at=submission.reviewedAt,
            changes_summary=submission.changesSummary,
        )

    except Exception as e:
        logger.error(f"Could not create store submission review: {e}")
        raise backend.server.v2.store.exceptions.DatabaseError(
            "Failed to create store submission review"
        ) from e


async def get_admin_listings_with_versions(
    status: prisma.enums.SubmissionStatus | None = None,
    search_query: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> backend.server.v2.store.model.StoreListingsWithVersionsResponse:
    """
    Get store listings for admins with all their versions.

    Args:
        status: Filter by submission status (PENDING, APPROVED, REJECTED)
        search_query: Search by name, description, or user email
        page: Page number for pagination
        page_size: Number of items per page

    Returns:
        StoreListingsWithVersionsResponse with listings and their versions
    """
    logger.debug(
        f"Getting admin store listings with status={status}, search={search_query}, page={page}"
    )

    try:
        # Build the where clause for StoreListing
        where_dict: prisma.types.StoreListingWhereInput = {
            "isDeleted": False,
        }
        if status:
            where_dict["Versions"] = {"some": {"submissionStatus": status}}

        sanitized_query = sanitize_query(search_query)
        if sanitized_query:
            # Find users with matching email
            matching_users = await prisma.models.User.prisma().find_many(
                where={"email": {"contains": sanitized_query, "mode": "insensitive"}},
            )

            user_ids = [user.id for user in matching_users]

            # Set up OR conditions
            where_dict["OR"] = [
                {"slug": {"contains": sanitized_query, "mode": "insensitive"}},
                {
                    "Versions": {
                        "some": {
                            "name": {"contains": sanitized_query, "mode": "insensitive"}
                        }
                    }
                },
                {
                    "Versions": {
                        "some": {
                            "description": {
                                "contains": sanitized_query,
                                "mode": "insensitive",
                            }
                        }
                    }
                },
                {
                    "Versions": {
                        "some": {
                            "subHeading": {
                                "contains": sanitized_query,
                                "mode": "insensitive",
                            }
                        }
                    }
                },
            ]

            # Add user_id condition if any users matched
            if user_ids:
                where_dict["OR"].append({"owningUserId": {"in": user_ids}})

        # Calculate pagination
        skip = (page - 1) * page_size

        # Create proper Prisma types for the query
        where = prisma.types.StoreListingWhereInput(**where_dict)
        include = prisma.types.StoreListingInclude(
            Versions=prisma.types.FindManyStoreListingVersionArgsFromStoreListing(
                order_by=prisma.types._StoreListingVersion_version_OrderByInput(
                    version="desc"
                )
            ),
            OwningUser=True,
        )

        # Query listings with their versions
        listings = await prisma.models.StoreListing.prisma().find_many(
            where=where,
            skip=skip,
            take=page_size,
            include=include,
            order=[{"createdAt": "desc"}],
        )

        # Get total count for pagination
        total = await prisma.models.StoreListing.prisma().count(where=where)
        total_pages = (total + page_size - 1) // page_size

        # Convert to response models
        listings_with_versions = []
        for listing in listings:
            versions: list[backend.server.v2.store.model.StoreSubmission] = []
            # If we have versions, turn them into StoreSubmission models
            for version in listing.Versions or []:
                version_model = backend.server.v2.store.model.StoreSubmission(
                    agent_id=version.agentGraphId,
                    agent_version=version.agentGraphVersion,
                    name=version.name,
                    sub_heading=version.subHeading,
                    slug=listing.slug,
                    description=version.description,
                    image_urls=version.imageUrls or [],
                    date_submitted=version.submittedAt or version.createdAt,
                    status=version.submissionStatus,
                    runs=0,  # Default values since we don't have this data here
                    rating=0.0,  # Default values since we don't have this data here
                    store_listing_version_id=version.id,
                    reviewer_id=version.reviewerId,
                    review_comments=version.reviewComments,
                    internal_comments=version.internalComments,
                    reviewed_at=version.reviewedAt,
                    changes_summary=version.changesSummary,
                    version=version.version,
                )
                versions.append(version_model)

            # Get the latest version (first in the sorted list)
            latest_version = versions[0] if versions else None

            creator_email = listing.OwningUser.email if listing.OwningUser else None

            listing_with_versions = (
                backend.server.v2.store.model.StoreListingWithVersions(
                    listing_id=listing.id,
                    slug=listing.slug,
                    agent_id=listing.agentGraphId,
                    agent_version=listing.agentGraphVersion,
                    active_version_id=listing.activeVersionId,
                    has_approved_version=listing.hasApprovedVersion,
                    creator_email=creator_email,
                    latest_version=latest_version,
                    versions=versions,
                )
            )

            listings_with_versions.append(listing_with_versions)

        logger.debug(f"Found {len(listings_with_versions)} listings for admin")
        return backend.server.v2.store.model.StoreListingsWithVersionsResponse(
            listings=listings_with_versions,
            pagination=backend.server.v2.store.model.Pagination(
                current_page=page,
                total_items=total,
                total_pages=total_pages,
                page_size=page_size,
            ),
        )
    except Exception as e:
        logger.error(f"Error fetching admin store listings: {e}")
        # Return empty response rather than exposing internal errors
        return backend.server.v2.store.model.StoreListingsWithVersionsResponse(
            listings=[],
            pagination=backend.server.v2.store.model.Pagination(
                current_page=page,
                total_items=0,
                total_pages=0,
                page_size=page_size,
            ),
        )


async def get_agent_as_admin(
    user_id: str | None,
    store_listing_version_id: str,
) -> GraphModel:
    """Get agent using the version ID and store listing version ID."""
    store_listing_version = (
        await prisma.models.StoreListingVersion.prisma().find_unique(
            where={"id": store_listing_version_id}
        )
    )

    if not store_listing_version:
        raise ValueError(f"Store listing version {store_listing_version_id} not found")

    graph = await get_graph_as_admin(
        user_id=user_id,
        graph_id=store_listing_version.agentGraphId,
        version=store_listing_version.agentGraphVersion,
        for_export=True,
    )
    if not graph:
        raise ValueError(
            f"Agent {store_listing_version.agentGraphId} v{store_listing_version.agentGraphVersion} not found"
        )

    return graph
