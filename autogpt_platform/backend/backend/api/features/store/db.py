import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Literal, overload

import fastapi
import prisma.enums
import prisma.errors
import prisma.models
import prisma.types

from backend.data.db import transaction
from backend.data.graph import (
    GraphModel,
    GraphModelWithoutNodes,
    get_graph,
    get_graph_as_admin,
    get_sub_graphs,
)
from backend.data.includes import AGENT_GRAPH_INCLUDE
from backend.data.notifications import (
    AgentApprovalData,
    AgentRejectionData,
    NotificationEventModel,
    WaitlistLaunchData,
)
from backend.notifications.notifications import queue_notification_async
from backend.util.exceptions import DatabaseError
from backend.util.settings import Settings

from . import exceptions as store_exceptions
from . import model as store_model
from .embeddings import ensure_embedding
from .hybrid_search import hybrid_search

logger = logging.getLogger(__name__)
settings = Settings()


# Constants for default admin values
DEFAULT_ADMIN_NAME = "AutoGPT Admin"
DEFAULT_ADMIN_EMAIL = "admin@autogpt.co"


async def get_store_agents(
    featured: bool = False,
    creators: list[str] | None = None,
    sorted_by: Literal["rating", "runs", "name", "updated_at"] | None = None,
    search_query: str | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> store_model.StoreAgentsResponse:
    """
    Get PUBLIC store agents from the StoreAgent view.

    Search behavior:
    - With search_query: Uses hybrid search (semantic + lexical)
    - Fallback: If embeddings unavailable, gracefully degrades to lexical-only
    - Rationale: User-facing endpoint prioritizes availability over accuracy

    Note: Admin operations (approval) use fail-fast to prevent inconsistent state.
    """
    logger.debug(
        f"Getting store agents. featured={featured}, creators={creators}, sorted_by={sorted_by}, search={search_query}, category={category}, page={page}"
    )

    search_used_hybrid = False
    store_agents: list[store_model.StoreAgent] = []
    agents: list[dict[str, Any]] = []
    total = 0
    total_pages = 0

    try:
        # If search_query is provided, use hybrid search (embeddings + tsvector)
        if search_query:
            # Try hybrid search combining semantic and lexical signals
            # Falls back to lexical-only if OpenAI unavailable (user-facing, high SLA)
            try:
                agents, total = await hybrid_search(
                    query=search_query,
                    featured=featured,
                    creators=creators,
                    category=category,
                    sorted_by="relevance",  # Use hybrid scoring for relevance
                    page=page,
                    page_size=page_size,
                )
                search_used_hybrid = True
            except Exception as e:
                # Log error but fall back to lexical search for better UX
                logger.error(
                    f"Hybrid search failed (likely OpenAI unavailable), "
                    f"falling back to lexical search: {e}"
                )
                # search_used_hybrid remains False, will use fallback path below

            # Convert hybrid search results (dict format) if hybrid succeeded
            if search_used_hybrid:
                total_pages = (total + page_size - 1) // page_size
                store_agents: list[store_model.StoreAgent] = []
                for agent in agents:
                    try:
                        store_agent = store_model.StoreAgent(
                            slug=agent["slug"],
                            agent_name=agent["agent_name"],
                            agent_image=(
                                agent["agent_image"][0] if agent["agent_image"] else ""
                            ),
                            creator=agent["creator_username"] or "Needs Profile",
                            creator_avatar=agent["creator_avatar"] or "",
                            sub_heading=agent["sub_heading"],
                            description=agent["description"],
                            runs=agent["runs"],
                            rating=agent["rating"],
                            agent_graph_id=agent.get("agentGraphId", ""),
                        )
                        store_agents.append(store_agent)
                    except Exception as e:
                        logger.error(
                            f"Error parsing Store agent from hybrid search results: {e}"
                        )
                        continue

        if not search_used_hybrid:
            # Fallback path - use basic search or no search
            where_clause: prisma.types.StoreAgentWhereInput = {"is_available": True}
            if featured:
                where_clause["featured"] = featured
            if creators:
                where_clause["creator_username"] = {"in": creators}
            if category:
                where_clause["categories"] = {"has": category}

            # Add basic text search if search_query provided but hybrid failed
            if search_query:
                where_clause["OR"] = [
                    {"agent_name": {"contains": search_query, "mode": "insensitive"}},
                    {"sub_heading": {"contains": search_query, "mode": "insensitive"}},
                    {"description": {"contains": search_query, "mode": "insensitive"}},
                ]

            order_by = []
            if sorted_by == "rating":
                order_by.append({"rating": "desc"})
            elif sorted_by == "runs":
                order_by.append({"runs": "desc"})
            elif sorted_by == "name":
                order_by.append({"agent_name": "asc"})

            db_agents = await prisma.models.StoreAgent.prisma().find_many(
                where=where_clause,
                order=order_by,
                skip=(page - 1) * page_size,
                take=page_size,
            )

            total = await prisma.models.StoreAgent.prisma().count(where=where_clause)
            total_pages = (total + page_size - 1) // page_size

            store_agents: list[store_model.StoreAgent] = []
            for agent in db_agents:
                try:
                    # Create the StoreAgent object safely
                    store_agent = store_model.StoreAgent(
                        slug=agent.slug,
                        agent_name=agent.agent_name,
                        agent_image=agent.agent_image[0] if agent.agent_image else "",
                        creator=agent.creator_username or "Needs Profile",
                        creator_avatar=agent.creator_avatar or "",
                        sub_heading=agent.sub_heading,
                        description=agent.description,
                        runs=agent.runs,
                        rating=agent.rating,
                        agent_graph_id=agent.agentGraphId,
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
        return store_model.StoreAgentsResponse(
            agents=store_agents,
            pagination=store_model.Pagination(
                current_page=page,
                total_items=total,
                total_pages=total_pages,
                page_size=page_size,
            ),
        )
    except Exception as e:
        logger.error(f"Error getting store agents: {e}")
        raise DatabaseError("Failed to fetch store agents") from e
    # TODO: commenting this out as we concerned about potential db load issues
    # finally:
    #     if search_term:
    #         await log_search_term(search_query=search_term)


async def log_search_term(search_query: str):
    """Log a search term to the database"""

    # Anonymize the data by preventing correlation with other logs
    date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        await prisma.models.SearchTerms.prisma().create(
            data={"searchTerm": search_query, "createdDate": date}
        )
    except Exception as e:
        # Fail silently here so that logging search terms doesn't break the app
        logger.error(f"Error logging search term: {e}")


async def get_store_agent_details(
    username: str, agent_name: str, include_changelog: bool = False
) -> store_model.StoreAgentDetails:
    """Get PUBLIC store agent details from the StoreAgent view"""
    logger.debug(f"Getting store agent details for {username}/{agent_name}")

    try:
        agent = await prisma.models.StoreAgent.prisma().find_first(
            where={"creator_username": username, "slug": agent_name}
        )

        if not agent:
            logger.warning(f"Agent not found: {username}/{agent_name}")
            raise store_exceptions.AgentNotFoundError(
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

        if active_version_id:
            agent_by_active = await prisma.models.StoreAgent.prisma().find_first(
                where={"storeListingVersionId": active_version_id}
            )
            if agent_by_active:
                agent = agent_by_active
        elif store_listing:
            latest_approved = (
                await prisma.models.StoreListingVersion.prisma().find_first(
                    where={
                        "storeListingId": store_listing.id,
                        "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
                    },
                    order=[{"version": "desc"}],
                )
            )
            if latest_approved:
                agent_latest = await prisma.models.StoreAgent.prisma().find_first(
                    where={"storeListingVersionId": latest_approved.id}
                )
                if agent_latest:
                    agent = agent_latest

        if store_listing and store_listing.ActiveVersion:
            recommended_schedule_cron = (
                store_listing.ActiveVersion.recommendedScheduleCron
            )
        else:
            recommended_schedule_cron = None

        # Fetch changelog data if requested
        changelog_data = None
        if include_changelog and store_listing:
            changelog_versions = (
                await prisma.models.StoreListingVersion.prisma().find_many(
                    where={
                        "storeListingId": store_listing.id,
                        "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
                    },
                    order=[{"version": "desc"}],
                )
            )
            changelog_data = [
                store_model.ChangelogEntry(
                    version=str(version.version),
                    changes_summary=version.changesSummary or "No changes recorded",
                    date=version.createdAt,
                )
                for version in changelog_versions
            ]

        logger.debug(f"Found agent details for {username}/{agent_name}")
        return store_model.StoreAgentDetails(
            store_listing_version_id=agent.storeListingVersionId,
            slug=agent.slug,
            agent_name=agent.agent_name,
            agent_video=agent.agent_video or "",
            agent_output_demo=agent.agent_output_demo or "",
            agent_image=agent.agent_image,
            creator=agent.creator_username or "",
            creator_avatar=agent.creator_avatar or "",
            sub_heading=agent.sub_heading,
            description=agent.description,
            categories=agent.categories,
            runs=agent.runs,
            rating=agent.rating,
            versions=agent.versions,
            agentGraphVersions=agent.agentGraphVersions,
            agentGraphId=agent.agentGraphId,
            last_updated=agent.updated_at,
            active_version_id=active_version_id,
            has_approved_version=has_approved_version,
            recommended_schedule_cron=recommended_schedule_cron,
            changelog=changelog_data,
        )
    except store_exceptions.AgentNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error getting store agent details: {e}")
        raise DatabaseError("Failed to fetch agent details") from e


@overload
async def get_available_graph(
    store_listing_version_id: str, hide_nodes: Literal[False]
) -> GraphModel: ...


@overload
async def get_available_graph(
    store_listing_version_id: str, hide_nodes: Literal[True] = True
) -> GraphModelWithoutNodes: ...


async def get_available_graph(
    store_listing_version_id: str,
    hide_nodes: bool = True,
) -> GraphModelWithoutNodes | GraphModel:
    try:
        # Get avaialble, non-deleted store listing version
        store_listing_version = (
            await prisma.models.StoreListingVersion.prisma().find_first(
                where={
                    "id": store_listing_version_id,
                    "isAvailable": True,
                    "isDeleted": False,
                },
                include={"AgentGraph": {"include": AGENT_GRAPH_INCLUDE}},
            )
        )

        if not store_listing_version or not store_listing_version.AgentGraph:
            raise fastapi.HTTPException(
                status_code=404,
                detail=f"Store listing version {store_listing_version_id} not found",
            )

        return (GraphModelWithoutNodes if hide_nodes else GraphModel).from_db(
            store_listing_version.AgentGraph
        )

    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        raise DatabaseError("Failed to fetch agent") from e


async def get_store_agent_by_version_id(
    store_listing_version_id: str,
) -> store_model.StoreAgentDetails:
    logger.debug(f"Getting store agent details for {store_listing_version_id}")

    try:
        agent = await prisma.models.StoreAgent.prisma().find_first(
            where={"storeListingVersionId": store_listing_version_id}
        )

        if not agent:
            logger.warning(f"Agent not found: {store_listing_version_id}")
            raise store_exceptions.AgentNotFoundError(
                f"Agent {store_listing_version_id} not found"
            )

        logger.debug(f"Found agent details for {store_listing_version_id}")
        return store_model.StoreAgentDetails(
            store_listing_version_id=agent.storeListingVersionId,
            slug=agent.slug,
            agent_name=agent.agent_name,
            agent_video=agent.agent_video or "",
            agent_output_demo=agent.agent_output_demo or "",
            agent_image=agent.agent_image,
            creator=agent.creator_username or "",
            creator_avatar=agent.creator_avatar or "",
            sub_heading=agent.sub_heading,
            description=agent.description,
            categories=agent.categories,
            runs=agent.runs,
            rating=agent.rating,
            versions=agent.versions,
            agentGraphVersions=agent.agentGraphVersions,
            agentGraphId=agent.agentGraphId,
            last_updated=agent.updated_at,
        )
    except store_exceptions.AgentNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error getting store agent details: {e}")
        raise DatabaseError("Failed to fetch agent details") from e


async def get_store_creators(
    featured: bool = False,
    search_query: str | None = None,
    sorted_by: Literal["agent_rating", "agent_runs", "num_agents"] | None = None,
    page: int = 1,
    page_size: int = 20,
) -> store_model.CreatorsResponse:
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
            raise DatabaseError("Invalid search query")

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
            raise DatabaseError("Invalid page number")
        if not isinstance(page_size, int) or page_size < 1 or page_size > 100:
            raise DatabaseError("Invalid page size")

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
            store_model.Creator(
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
        return store_model.CreatorsResponse(
            creators=creator_models,
            pagination=store_model.Pagination(
                current_page=page,
                total_items=total,
                total_pages=total_pages,
                page_size=page_size,
            ),
        )
    except Exception as e:
        logger.error(f"Error getting store creators: {e}")
        raise DatabaseError("Failed to fetch store creators") from e


async def get_store_creator_details(
    username: str,
) -> store_model.CreatorDetails:
    logger.debug(f"Getting store creator details for {username}")

    try:
        # Query creator details from database
        creator = await prisma.models.Creator.prisma().find_unique(
            where={"username": username}
        )

        if not creator:
            logger.warning(f"Creator not found: {username}")
            raise store_exceptions.CreatorNotFoundError(f"Creator {username} not found")

        logger.debug(f"Found creator details for {username}")
        return store_model.CreatorDetails(
            name=creator.name,
            username=creator.username,
            description=creator.description,
            links=creator.links,
            avatar_url=creator.avatar_url,
            agent_rating=creator.agent_rating,
            agent_runs=creator.agent_runs,
            top_categories=creator.top_categories,
        )
    except store_exceptions.CreatorNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error getting store creator details: {e}")
        raise DatabaseError("Failed to fetch creator details") from e


async def get_store_submissions(
    user_id: str, page: int = 1, page_size: int = 20
) -> store_model.StoreSubmissionsResponse:
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
            submission_model = store_model.StoreSubmission(
                listing_id=sub.listing_id,
                agent_id=sub.agent_id,
                agent_version=sub.agent_version,
                name=sub.name,
                sub_heading=sub.sub_heading,
                slug=sub.slug,
                description=sub.description,
                instructions=getattr(sub, "instructions", None),
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
                video_url=sub.video_url,
                categories=sub.categories,
            )
            submission_models.append(submission_model)

        logger.debug(f"Found {len(submission_models)} submissions")
        return store_model.StoreSubmissionsResponse(
            submissions=submission_models,
            pagination=store_model.Pagination(
                current_page=page,
                total_items=total,
                total_pages=total_pages,
                page_size=page_size,
            ),
        )

    except Exception as e:
        logger.error(f"Error fetching store submissions: {e}")
        # Return empty response rather than exposing internal errors
        return store_model.StoreSubmissionsResponse(
            submissions=[],
            pagination=store_model.Pagination(
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
    Delete a store submission version as the submitting user.

    Args:
        user_id: ID of the authenticated user
        submission_id: StoreListingVersion ID to delete

    Returns:
        bool: True if successfully deleted
    """
    try:
        # Find the submission version with ownership check
        version = await prisma.models.StoreListingVersion.prisma().find_first(
            where={"id": submission_id}, include={"StoreListing": True}
        )

        if (
            not version
            or not version.StoreListing
            or version.StoreListing.owningUserId != user_id
        ):
            raise store_exceptions.SubmissionNotFoundError("Submission not found")

        # Prevent deletion of approved submissions
        if version.submissionStatus == prisma.enums.SubmissionStatus.APPROVED:
            raise store_exceptions.InvalidOperationError(
                "Cannot delete approved submissions"
            )

        # Delete the version
        await prisma.models.StoreListingVersion.prisma().delete(
            where={"id": version.id}
        )

        # Clean up empty listing if this was the last version
        remaining = await prisma.models.StoreListingVersion.prisma().count(
            where={"storeListingId": version.storeListingId}
        )
        if remaining == 0:
            await prisma.models.StoreListing.prisma().delete(
                where={"id": version.storeListingId}
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
    agent_output_demo_url: str | None = None,
    image_urls: list[str] = [],
    description: str = "",
    instructions: str | None = None,
    sub_heading: str = "",
    categories: list[str] = [],
    changes_summary: str | None = "Initial Submission",
    recommended_schedule_cron: str | None = None,
) -> store_model.StoreSubmission:
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
            # Provide more user-friendly error message when agent_id is empty
            if not agent_id or agent_id.strip() == "":
                raise store_exceptions.AgentNotFoundError(
                    "No agent selected. Please select an agent before submitting to the store."
                )
            else:
                raise store_exceptions.AgentNotFoundError(
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
                instructions=instructions,
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
                        agentOutputDemoUrl=agent_output_demo_url,
                        imageUrls=image_urls,
                        description=description,
                        instructions=instructions,
                        categories=categories,
                        subHeading=sub_heading,
                        submissionStatus=prisma.enums.SubmissionStatus.PENDING,
                        submittedAt=datetime.now(tz=timezone.utc),
                        changesSummary=changes_summary,
                        recommendedScheduleCron=recommended_schedule_cron,
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
        return store_model.StoreSubmission(
            listing_id=listing.id,
            agent_id=agent_id,
            agent_version=agent_version,
            name=name,
            slug=slug,
            sub_heading=sub_heading,
            description=description,
            instructions=instructions,
            image_urls=image_urls,
            date_submitted=listing.createdAt,
            status=prisma.enums.SubmissionStatus.PENDING,
            runs=0,
            rating=0.0,
            store_listing_version_id=store_listing_version_id,
            changes_summary=changes_summary,
        )
    except prisma.errors.UniqueViolationError as exc:
        # Attempt to check if the error was due to the slug field being unique
        error_str = str(exc)
        if "slug" in error_str.lower():
            logger.debug(
                f"Slug '{slug}' is already in use by another agent (agent_id: {agent_id}) for user {user_id}"
            )
            raise store_exceptions.SlugAlreadyInUseError(
                f"The URL slug '{slug}' is already in use by another one of your agents. Please choose a different slug."
            ) from exc
        else:
            # Reraise as a generic database error for other unique violations
            raise DatabaseError(
                f"Unique constraint violated (not slug): {error_str}"
            ) from exc
    except (
        store_exceptions.AgentNotFoundError,
        store_exceptions.ListingExistsError,
    ):
        raise
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error creating store submission: {e}")
        raise DatabaseError("Failed to create store submission") from e


async def edit_store_submission(
    user_id: str,
    store_listing_version_id: str,
    name: str,
    video_url: str | None = None,
    agent_output_demo_url: str | None = None,
    image_urls: list[str] = [],
    description: str = "",
    sub_heading: str = "",
    categories: list[str] = [],
    changes_summary: str | None = "Update submission",
    recommended_schedule_cron: str | None = None,
    instructions: str | None = None,
) -> store_model.StoreSubmission:
    """
    Edit an existing store listing submission.

    Args:
        user_id: ID of the authenticated user editing the submission
        store_listing_version_id: ID of the store listing version to edit
        agent_id: ID of the agent being submitted
        agent_version: Version of the agent being submitted
        slug: URL slug for the listing (only changeable for PENDING submissions)
        name: Name of the agent
        video_url: Optional URL to video demo
        image_urls: List of image URLs for the listing
        description: Description of the agent
        sub_heading: Optional sub-heading for the agent
        categories: List of categories for the agent
        changes_summary: Summary of changes made in this submission

    Returns:
        StoreSubmission: The updated store submission

    Raises:
        SubmissionNotFoundError: If the submission is not found
        UnauthorizedError: If the user doesn't own the submission
        InvalidOperationError: If trying to edit a submission that can't be edited
    """
    try:
        # Get the current version and verify ownership
        current_version = await prisma.models.StoreListingVersion.prisma().find_first(
            where=prisma.types.StoreListingVersionWhereInput(
                id=store_listing_version_id
            ),
            include={
                "StoreListing": {
                    "include": {
                        "Versions": {"order_by": {"version": "desc"}, "take": 1}
                    }
                }
            },
        )

        if not current_version:
            raise store_exceptions.SubmissionNotFoundError(
                f"Store listing version not found: {store_listing_version_id}"
            )

        # Verify the user owns this submission
        if (
            not current_version.StoreListing
            or current_version.StoreListing.owningUserId != user_id
        ):
            raise store_exceptions.UnauthorizedError(
                f"User {user_id} does not own submission {store_listing_version_id}"
            )

        # Currently we are not allowing user to update the agent associated with a submission
        # If we allow it in future, then we need a check here to verify the agent belongs to this user.

        # Only allow editing of PENDING submissions
        if current_version.submissionStatus != prisma.enums.SubmissionStatus.PENDING:
            raise store_exceptions.InvalidOperationError(
                f"Cannot edit a {current_version.submissionStatus.value.lower()} submission. Only pending submissions can be edited."
            )

        # For PENDING submissions, we can update the existing version
        # Update the existing version
        updated_version = await prisma.models.StoreListingVersion.prisma().update(
            where={"id": store_listing_version_id},
            data=prisma.types.StoreListingVersionUpdateInput(
                name=name,
                videoUrl=video_url,
                agentOutputDemoUrl=agent_output_demo_url,
                imageUrls=image_urls,
                description=description,
                categories=categories,
                subHeading=sub_heading,
                changesSummary=changes_summary,
                recommendedScheduleCron=recommended_schedule_cron,
                instructions=instructions,
            ),
        )

        logger.debug(
            f"Updated existing version {store_listing_version_id} for agent {current_version.agentGraphId}"
        )

        if not updated_version:
            raise DatabaseError("Failed to update store listing version")
        return store_model.StoreSubmission(
            listing_id=current_version.StoreListing.id,
            agent_id=current_version.agentGraphId,
            agent_version=current_version.agentGraphVersion,
            name=name,
            sub_heading=sub_heading,
            slug=current_version.StoreListing.slug,
            description=description,
            instructions=instructions,
            image_urls=image_urls,
            date_submitted=updated_version.submittedAt or updated_version.createdAt,
            status=updated_version.submissionStatus,
            runs=0,
            rating=0.0,
            store_listing_version_id=updated_version.id,
            changes_summary=changes_summary,
            video_url=video_url,
            categories=categories,
            version=updated_version.version,
        )

    except (
        store_exceptions.SubmissionNotFoundError,
        store_exceptions.UnauthorizedError,
        store_exceptions.AgentNotFoundError,
        store_exceptions.ListingExistsError,
        store_exceptions.InvalidOperationError,
    ):
        raise
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error editing store submission: {e}")
        raise DatabaseError("Failed to edit store submission") from e


async def create_store_version(
    user_id: str,
    agent_id: str,
    agent_version: int,
    store_listing_id: str,
    name: str,
    video_url: str | None = None,
    agent_output_demo_url: str | None = None,
    image_urls: list[str] = [],
    description: str = "",
    instructions: str | None = None,
    sub_heading: str = "",
    categories: list[str] = [],
    changes_summary: str | None = "Initial submission",
    recommended_schedule_cron: str | None = None,
) -> store_model.StoreSubmission:
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
            raise store_exceptions.ListingNotFoundError(
                f"Store listing not found. User ID: {user_id}, Listing ID: {store_listing_id}"
            )

        # Verify the agent belongs to this user
        agent = await prisma.models.AgentGraph.prisma().find_first(
            where=prisma.types.AgentGraphWhereInput(
                id=agent_id, version=agent_version, userId=user_id
            )
        )

        if not agent:
            raise store_exceptions.AgentNotFoundError(
                f"Agent not found for this user. User ID: {user_id}, Agent ID: {agent_id}, Version: {agent_version}"
            )

        # Check if there's already a PENDING submission for this agent (any version)
        existing_pending_submission = (
            await prisma.models.StoreListingVersion.prisma().find_first(
                where=prisma.types.StoreListingVersionWhereInput(
                    storeListingId=store_listing_id,
                    agentGraphId=agent_id,
                    submissionStatus=prisma.enums.SubmissionStatus.PENDING,
                    isDeleted=False,
                )
            )
        )

        # Handle existing pending submission and create new one atomically
        async with transaction() as tx:
            # Get the latest version number first
            latest_listing = await prisma.models.StoreListing.prisma(tx).find_first(
                where=prisma.types.StoreListingWhereInput(
                    id=store_listing_id, owningUserId=user_id
                ),
                include={"Versions": {"order_by": {"version": "desc"}, "take": 1}},
            )

            if not latest_listing:
                raise store_exceptions.ListingNotFoundError(
                    f"Store listing not found. User ID: {user_id}, Listing ID: {store_listing_id}"
                )

            latest_version = (
                latest_listing.Versions[0] if latest_listing.Versions else None
            )
            next_version = (latest_version.version + 1) if latest_version else 1

            # If there's an existing pending submission, delete it atomically before creating new one
            if existing_pending_submission:
                logger.info(
                    f"Found existing PENDING submission for agent {agent_id} (was v{existing_pending_submission.agentGraphVersion}, now v{agent_version}), replacing existing submission instead of creating duplicate"
                )
                await prisma.models.StoreListingVersion.prisma(tx).delete(
                    where={"id": existing_pending_submission.id}
                )
                logger.debug(
                    f"Deleted existing pending submission {existing_pending_submission.id}"
                )

            # Create a new version for the existing listing
            new_version = await prisma.models.StoreListingVersion.prisma(tx).create(
                data=prisma.types.StoreListingVersionCreateInput(
                    version=next_version,
                    agentGraphId=agent_id,
                    agentGraphVersion=agent_version,
                    name=name,
                    videoUrl=video_url,
                    agentOutputDemoUrl=agent_output_demo_url,
                    imageUrls=image_urls,
                    description=description,
                    instructions=instructions,
                    categories=categories,
                    subHeading=sub_heading,
                    submissionStatus=prisma.enums.SubmissionStatus.PENDING,
                    submittedAt=datetime.now(),
                    changesSummary=changes_summary,
                    recommendedScheduleCron=recommended_schedule_cron,
                    storeListingId=store_listing_id,
                )
            )

        logger.debug(
            f"Created new version for listing {store_listing_id} of agent {agent_id}"
        )
        # Return submission details
        return store_model.StoreSubmission(
            listing_id=listing.id,
            agent_id=agent_id,
            agent_version=agent_version,
            name=name,
            slug=listing.slug,
            sub_heading=sub_heading,
            description=description,
            instructions=instructions,
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
        raise DatabaseError("Failed to create new store version") from e


async def create_store_review(
    user_id: str,
    store_listing_version_id: str,
    score: int,
    comments: str | None = None,
) -> store_model.StoreReview:
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

        return store_model.StoreReview(
            score=review.score,
            comments=review.comments,
        )

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error creating store review: {e}")
        raise DatabaseError("Failed to create store review") from e


async def get_user_profile(
    user_id: str,
) -> store_model.ProfileDetails | None:
    logger.debug(f"Getting user profile for {user_id}")

    try:
        profile = await prisma.models.Profile.prisma().find_first(
            where={"userId": user_id}
        )

        if not profile:
            return None
        return store_model.ProfileDetails(
            name=profile.name,
            username=profile.username,
            description=profile.description,
            links=profile.links,
            avatar_url=profile.avatarUrl,
        )
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise DatabaseError("Failed to get user profile") from e


async def update_profile(
    user_id: str, profile: store_model.Profile
) -> store_model.CreatorDetails:
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
            raise store_exceptions.ProfileNotFoundError(
                f"Profile not found for user {user_id}. This should not be possible."
            )

        # Verify that the user is authorized to update this profile
        if existing_profile.userId != user_id:
            logger.error(
                f"Unauthorized update attempt for profile {existing_profile.id} by user {user_id}"
            )
            raise DatabaseError(
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
            raise DatabaseError("Failed to update profile")

        return store_model.CreatorDetails(
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
        raise DatabaseError("Failed to update profile") from e


async def get_my_agents(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
) -> store_model.MyAgentsResponse:
    """Get the agents for the authenticated user"""
    logger.debug(f"Getting my agents for user {user_id}, page={page}")

    try:
        search_filter: prisma.types.LibraryAgentWhereInput = {
            "userId": user_id,
            "AgentGraph": {
                "is": {
                    "StoreListings": {
                        "none": {
                            "isDeleted": False,
                            "Versions": {
                                "some": {
                                    "isAvailable": True,
                                }
                            },
                        }
                    }
                }
            },
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
            store_model.MyAgent(
                agent_id=graph.id,
                agent_version=graph.version,
                agent_name=graph.name or "",
                last_edited=graph.updatedAt or graph.createdAt,
                description=graph.description or "",
                agent_image=library_agent.imageUrl,
                recommended_schedule_cron=graph.recommendedScheduleCron,
            )
            for library_agent in library_agents
            if (graph := library_agent.AgentGraph)
        ]

        return store_model.MyAgentsResponse(
            agents=my_agents,
            pagination=store_model.Pagination(
                current_page=page,
                total_items=total,
                total_pages=total_pages,
                page_size=page_size,
            ),
        )
    except Exception as e:
        logger.error(f"Error getting my agents: {e}")
        raise DatabaseError("Failed to fetch my agents") from e


async def get_agent(store_listing_version_id: str) -> GraphModel:
    """Get agent using the version ID and store listing version ID."""
    store_listing_version = (
        await prisma.models.StoreListingVersion.prisma().find_unique(
            where={"id": store_listing_version_id}
        )
    )

    if not store_listing_version:
        raise ValueError(f"Store listing version {store_listing_version_id} not found")

    graph = await get_graph(
        graph_id=store_listing_version.agentGraphId,
        version=store_listing_version.agentGraphVersion,
        user_id=None,
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


async def _approve_sub_agent(
    tx,
    sub_graph: prisma.models.AgentGraph,
    main_agent_name: str,
    main_agent_version: int,
    main_agent_user_id: str,
) -> None:
    """Approve a single sub-agent by creating/updating store listings as needed"""
    heading = f"Sub-agent of {main_agent_name} v{main_agent_version}"

    # Find existing listing for this sub-agent
    listing = await prisma.models.StoreListing.prisma(tx).find_first(
        where={"agentGraphId": sub_graph.id, "isDeleted": False},
        include={"Versions": True},
    )

    # Early return: Create new listing if none exists
    if not listing:
        await prisma.models.StoreListing.prisma(tx).create(
            data=prisma.types.StoreListingCreateInput(
                slug=f"sub-agent-{sub_graph.id[:8]}",
                agentGraphId=sub_graph.id,
                agentGraphVersion=sub_graph.version,
                owningUserId=main_agent_user_id,
                hasApprovedVersion=True,
                Versions={
                    "create": [
                        _create_sub_agent_version_data(
                            sub_graph, heading, main_agent_name
                        )
                    ]
                },
            )
        )
        return

    # Find version matching this sub-graph
    matching_version = next(
        (
            v
            for v in listing.Versions or []
            if v.agentGraphId == sub_graph.id
            and v.agentGraphVersion == sub_graph.version
        ),
        None,
    )

    # Early return: Approve existing version if found and not already approved
    if matching_version:
        if matching_version.submissionStatus == prisma.enums.SubmissionStatus.APPROVED:
            return  # Already approved, nothing to do

        await prisma.models.StoreListingVersion.prisma(tx).update(
            where={"id": matching_version.id},
            data={
                "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
                "reviewedAt": datetime.now(tz=timezone.utc),
            },
        )
        await prisma.models.StoreListing.prisma(tx).update(
            where={"id": listing.id}, data={"hasApprovedVersion": True}
        )
        return

    # Create new version if no matching version found
    next_version = max((v.version for v in listing.Versions or []), default=0) + 1
    await prisma.models.StoreListingVersion.prisma(tx).create(
        data={
            **_create_sub_agent_version_data(sub_graph, heading, main_agent_name),
            "version": next_version,
            "storeListingId": listing.id,
        }
    )
    await prisma.models.StoreListing.prisma(tx).update(
        where={"id": listing.id}, data={"hasApprovedVersion": True}
    )


def _create_sub_agent_version_data(
    sub_graph: prisma.models.AgentGraph, heading: str, main_agent_name: str
) -> prisma.types.StoreListingVersionCreateInput:
    """Create store listing version data for a sub-agent"""
    return prisma.types.StoreListingVersionCreateInput(
        agentGraphId=sub_graph.id,
        agentGraphVersion=sub_graph.version,
        name=sub_graph.name or heading,
        submissionStatus=prisma.enums.SubmissionStatus.APPROVED,
        subHeading=heading,
        description=(
            f"{heading}: {sub_graph.description}" if sub_graph.description else heading
        ),
        changesSummary=f"Auto-approved as sub-agent of {main_agent_name}",
        isAvailable=False,
        submittedAt=datetime.now(tz=timezone.utc),
        imageUrls=[],  # Sub-agents don't need images
        categories=[],  # Sub-agents don't need categories
    )


async def review_store_submission(
    store_listing_version_id: str,
    is_approved: bool,
    external_comments: str,
    internal_comments: str,
    reviewer_id: str,
) -> store_model.StoreSubmission:
    """Review a store listing submission as an admin."""
    try:
        store_listing_version = (
            await prisma.models.StoreListingVersion.prisma().find_unique(
                where={"id": store_listing_version_id},
                include={
                    "StoreListing": True,
                    "AgentGraph": {"include": {**AGENT_GRAPH_INCLUDE, "User": True}},
                    "Reviewer": True,
                },
            )
        )

        if not store_listing_version or not store_listing_version.StoreListing:
            raise fastapi.HTTPException(
                status_code=404,
                detail=f"Store listing version {store_listing_version_id} not found",
            )

        # Check if we're rejecting an already approved agent
        is_rejecting_approved = (
            not is_approved
            and store_listing_version.submissionStatus
            == prisma.enums.SubmissionStatus.APPROVED
        )

        # If approving, update the listing to indicate it has an approved version
        if is_approved and store_listing_version.AgentGraph:
            async with transaction() as tx:
                # Handle sub-agent approvals in transaction
                await asyncio.gather(
                    *[
                        _approve_sub_agent(
                            tx,
                            sub_graph,
                            store_listing_version.name,
                            store_listing_version.agentGraphVersion,
                            store_listing_version.StoreListing.owningUserId,
                        )
                        for sub_graph in await get_sub_graphs(
                            store_listing_version.AgentGraph
                        )
                    ]
                )

                # Update the AgentGraph with store listing data
                await prisma.models.AgentGraph.prisma(tx).update(
                    where={
                        "graphVersionId": {
                            "id": store_listing_version.agentGraphId,
                            "version": store_listing_version.agentGraphVersion,
                        }
                    },
                    data={
                        "name": store_listing_version.name,
                        "description": store_listing_version.description,
                        "recommendedScheduleCron": store_listing_version.recommendedScheduleCron,
                        "instructions": store_listing_version.instructions,
                    },
                )

                # Generate embedding for approved listing (blocking - admin operation)
                # Inside transaction: if embedding fails, entire transaction rolls back
                await ensure_embedding(
                    version_id=store_listing_version_id,
                    name=store_listing_version.name,
                    description=store_listing_version.description,
                    sub_heading=store_listing_version.subHeading,
                    categories=store_listing_version.categories or [],
                    tx=tx,
                )

                await prisma.models.StoreListing.prisma(tx).update(
                    where={"id": store_listing_version.StoreListing.id},
                    data={
                        "hasApprovedVersion": True,
                        "ActiveVersion": {"connect": {"id": store_listing_version_id}},
                    },
                )

        # If rejecting an approved agent, update the StoreListing accordingly
        if is_rejecting_approved:
            # Check if there are other approved versions
            other_approved = (
                await prisma.models.StoreListingVersion.prisma().find_first(
                    where={
                        "storeListingId": store_listing_version.StoreListing.id,
                        "id": {"not": store_listing_version_id},
                        "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
                    }
                )
            )

            if not other_approved:
                # No other approved versions, update hasApprovedVersion to False
                await prisma.models.StoreListing.prisma().update(
                    where={"id": store_listing_version.StoreListing.id},
                    data={
                        "hasApprovedVersion": False,
                        "ActiveVersion": {"disconnect": True},
                    },
                )
            else:
                # Set the most recent other approved version as active
                await prisma.models.StoreListing.prisma().update(
                    where={"id": store_listing_version.StoreListing.id},
                    data={
                        "ActiveVersion": {"connect": {"id": other_approved.id}},
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
            raise DatabaseError(
                f"Failed to update store listing version {store_listing_version_id}"
            )

        # Send email notification to the agent creator
        if store_listing_version.AgentGraph and store_listing_version.AgentGraph.User:
            agent_creator = store_listing_version.AgentGraph.User
            reviewer = (
                store_listing_version.Reviewer
                if store_listing_version.Reviewer
                else None
            )

            try:
                base_url = (
                    settings.config.frontend_base_url
                    or settings.config.platform_base_url
                )

                if is_approved:
                    store_agent = (
                        await prisma.models.StoreAgent.prisma().find_first_or_raise(
                            where={"storeListingVersionId": submission.id}
                        )
                    )

                    # Send approval notification
                    notification_data = AgentApprovalData(
                        agent_name=submission.name,
                        agent_id=submission.agentGraphId,
                        agent_version=submission.agentGraphVersion,
                        reviewer_name=(
                            reviewer.name
                            if reviewer and reviewer.name
                            else DEFAULT_ADMIN_NAME
                        ),
                        reviewer_email=(
                            reviewer.email if reviewer else DEFAULT_ADMIN_EMAIL
                        ),
                        comments=external_comments,
                        reviewed_at=submission.reviewedAt
                        or datetime.now(tz=timezone.utc),
                        store_url=f"{base_url}/marketplace/agent/{store_agent.creator_username}/{store_agent.slug}",
                    )

                    notification_event = NotificationEventModel[AgentApprovalData](
                        user_id=agent_creator.id,
                        type=prisma.enums.NotificationType.AGENT_APPROVED,
                        data=notification_data,
                    )
                else:
                    # Send rejection notification
                    notification_data = AgentRejectionData(
                        agent_name=submission.name,
                        agent_id=submission.agentGraphId,
                        agent_version=submission.agentGraphVersion,
                        reviewer_name=(
                            reviewer.name
                            if reviewer and reviewer.name
                            else DEFAULT_ADMIN_NAME
                        ),
                        reviewer_email=(
                            reviewer.email if reviewer else DEFAULT_ADMIN_EMAIL
                        ),
                        comments=external_comments,
                        reviewed_at=submission.reviewedAt
                        or datetime.now(tz=timezone.utc),
                        resubmit_url=f"{base_url}/build?flowID={submission.agentGraphId}",
                    )

                    notification_event = NotificationEventModel[AgentRejectionData](
                        user_id=agent_creator.id,
                        type=prisma.enums.NotificationType.AGENT_REJECTED,
                        data=notification_data,
                    )

                # Queue the notification for immediate sending
                await queue_notification_async(notification_event)
                logger.info(
                    f"Queued {'approval' if is_approved else 'rejection'} notification for user {agent_creator.id} and agent {submission.name}"
                )

            except Exception as e:
                logger.error(f"Failed to send email notification for agent review: {e}")
                # Don't fail the review process if email sending fails
                pass

        # Notify waitlist users if this is an approval and has a linked waitlist
        if is_approved and submission.StoreListing:
            try:
                frontend_base_url = (
                    settings.config.frontend_base_url
                    or settings.config.platform_base_url
                )
                store_agent = (
                    await prisma.models.StoreAgent.prisma().find_first_or_raise(
                        where={"storeListingVersionId": submission.id}
                    )
                )
                creator_username = store_agent.creator_username or "unknown"
                store_url = f"{frontend_base_url}/marketplace/agent/{creator_username}/{store_agent.slug}"
                await notify_waitlist_users_on_launch(
                    store_listing_id=submission.StoreListing.id,
                    agent_name=submission.name,
                    store_url=store_url,
                )
            except Exception as e:
                logger.error(f"Failed to notify waitlist users on agent approval: {e}")
                # Don't fail the approval process

        # Convert to Pydantic model for consistency
        return store_model.StoreSubmission(
            listing_id=(submission.StoreListing.id if submission.StoreListing else ""),
            agent_id=submission.agentGraphId,
            agent_version=submission.agentGraphVersion,
            name=submission.name,
            sub_heading=submission.subHeading,
            slug=(submission.StoreListing.slug if submission.StoreListing else ""),
            description=submission.description,
            instructions=submission.instructions,
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
        raise DatabaseError("Failed to create store submission review") from e


async def get_admin_listings_with_versions(
    status: prisma.enums.SubmissionStatus | None = None,
    search_query: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> store_model.StoreListingsWithVersionsResponse:
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

        if search_query:
            # Find users with matching email
            matching_users = await prisma.models.User.prisma().find_many(
                where={"email": {"contains": search_query, "mode": "insensitive"}},
            )

            user_ids = [user.id for user in matching_users]

            # Set up OR conditions
            where_dict["OR"] = [
                {"slug": {"contains": search_query, "mode": "insensitive"}},
                {
                    "Versions": {
                        "some": {
                            "name": {"contains": search_query, "mode": "insensitive"}
                        }
                    }
                },
                {
                    "Versions": {
                        "some": {
                            "description": {
                                "contains": search_query,
                                "mode": "insensitive",
                            }
                        }
                    }
                },
                {
                    "Versions": {
                        "some": {
                            "subHeading": {
                                "contains": search_query,
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
                order_by={"version": "desc"}
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
            versions: list[store_model.StoreSubmission] = []
            # If we have versions, turn them into StoreSubmission models
            for version in listing.Versions or []:
                version_model = store_model.StoreSubmission(
                    listing_id=listing.id,
                    agent_id=version.agentGraphId,
                    agent_version=version.agentGraphVersion,
                    name=version.name,
                    sub_heading=version.subHeading,
                    slug=listing.slug,
                    description=version.description,
                    instructions=version.instructions,
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

            listing_with_versions = store_model.StoreListingWithVersions(
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

            listings_with_versions.append(listing_with_versions)

        logger.debug(f"Found {len(listings_with_versions)} listings for admin")
        return store_model.StoreListingsWithVersionsResponse(
            listings=listings_with_versions,
            pagination=store_model.Pagination(
                current_page=page,
                total_items=total,
                total_pages=total_pages,
                page_size=page_size,
            ),
        )
    except Exception as e:
        logger.error(f"Error fetching admin store listings: {e}")
        # Return empty response rather than exposing internal errors
        return store_model.StoreListingsWithVersionsResponse(
            listings=[],
            pagination=store_model.Pagination(
                current_page=page,
                total_items=0,
                total_pages=0,
                page_size=page_size,
            ),
        )


async def check_submission_already_approved(
    store_listing_version_id: str,
) -> bool:
    """Check the submission status of a store listing version."""
    try:
        store_listing_version = (
            await prisma.models.StoreListingVersion.prisma().find_unique(
                where={"id": store_listing_version_id}
            )
        )
        if not store_listing_version:
            return False
        return (
            store_listing_version.submissionStatus
            == prisma.enums.SubmissionStatus.APPROVED
        )
    except Exception as e:
        logger.error(f"Error checking submission status: {e}")
        return False


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


def _waitlist_to_store_entry(
    waitlist: prisma.models.WaitlistEntry,
) -> store_model.StoreWaitlistEntry:
    """Convert a WaitlistEntry to StoreWaitlistEntry for public display."""
    return store_model.StoreWaitlistEntry(
        waitlistId=waitlist.id,
        slug=waitlist.slug,
        name=waitlist.name,
        subHeading=waitlist.subHeading,
        videoUrl=waitlist.videoUrl,
        agentOutputDemoUrl=waitlist.agentOutputDemoUrl,
        imageUrls=waitlist.imageUrls or [],
        description=waitlist.description,
        categories=waitlist.categories or [],
    )


async def get_waitlist() -> list[store_model.StoreWaitlistEntry]:
    """Get all active waitlists for public display."""
    try:
        waitlists = await prisma.models.WaitlistEntry.prisma().find_many(
            where=prisma.types.WaitlistEntryWhereInput(isDeleted=False),
        )

        # Filter out closed/done waitlists and sort by votes (descending)
        excluded_statuses = {
            prisma.enums.WaitlistExternalStatus.CANCELED,
            prisma.enums.WaitlistExternalStatus.DONE,
        }
        active_waitlists = [w for w in waitlists if w.status not in excluded_statuses]
        sorted_list = sorted(active_waitlists, key=lambda x: x.votes, reverse=True)

        return [_waitlist_to_store_entry(w) for w in sorted_list]
    except Exception as e:
        logger.error(f"Error fetching waitlists: {e}")
        raise DatabaseError("Failed to fetch waitlists") from e


async def get_user_waitlist_memberships(user_id: str) -> list[str]:
    """Get all waitlist IDs that a user has joined."""
    try:
        user = await prisma.models.User.prisma().find_unique(
            where={"id": user_id},
            include={"joinedWaitlists": True},
        )
        if not user or not user.joinedWaitlists:
            return []
        return [w.id for w in user.joinedWaitlists]
    except Exception as e:
        logger.error(f"Error fetching user waitlist memberships: {e}")
        raise DatabaseError("Failed to fetch waitlist memberships") from e


async def add_user_to_waitlist(
    waitlist_id: str, user_id: str | None, email: str | None
) -> store_model.StoreWaitlistEntry:
    """
    Add a user to a waitlist.

    For logged-in users: connects via joinedUsers relation
    For anonymous users: adds email to unaffiliatedEmailUsers array
    """
    if not user_id and not email:
        raise ValueError("Either user_id or email must be provided")

    try:
        # Find the waitlist
        waitlist = await prisma.models.WaitlistEntry.prisma().find_unique(
            where={"id": waitlist_id},
            include={"joinedUsers": True},
        )

        if not waitlist:
            raise ValueError(f"Waitlist {waitlist_id} not found")

        if waitlist.isDeleted:
            raise ValueError(f"Waitlist {waitlist_id} is no longer available")

        if waitlist.status in [
            prisma.enums.WaitlistExternalStatus.CANCELED,
            prisma.enums.WaitlistExternalStatus.DONE,
        ]:
            raise ValueError(f"Waitlist {waitlist_id} is closed")

        if user_id:
            # Check if user already joined
            joined_user_ids = [u.id for u in (waitlist.joinedUsers or [])]
            if user_id in joined_user_ids:
                # Already joined - return waitlist info
                logger.debug(f"User {user_id} already joined waitlist {waitlist_id}")
            else:
                # Connect user to waitlist
                await prisma.models.WaitlistEntry.prisma().update(
                    where={"id": waitlist_id},
                    data={"joinedUsers": {"connect": [{"id": user_id}]}},
                )
                logger.info(f"User {user_id} joined waitlist {waitlist_id}")

            # If user was previously in email list, remove them
            # Use transaction to prevent race conditions
            if email:
                async with transaction() as tx:
                    current_waitlist = await tx.waitlistentry.find_unique(
                        where={"id": waitlist_id}
                    )
                    if current_waitlist and email in (
                        current_waitlist.unaffiliatedEmailUsers or []
                    ):
                        updated_emails: list[str] = [
                            e
                            for e in (current_waitlist.unaffiliatedEmailUsers or [])
                            if e != email
                        ]
                        await tx.waitlistentry.update(
                            where={"id": waitlist_id},
                            data={"unaffiliatedEmailUsers": updated_emails},
                        )
        elif email:
            # Add email to unaffiliated list if not already present
            # Use transaction to prevent race conditions with concurrent signups
            async with transaction() as tx:
                # Re-fetch within transaction to get latest state
                current_waitlist = await tx.waitlistentry.find_unique(
                    where={"id": waitlist_id}
                )
                if current_waitlist:
                    current_emails: list[str] = list(
                        current_waitlist.unaffiliatedEmailUsers or []
                    )
                    if email not in current_emails:
                        current_emails.append(email)
                        await tx.waitlistentry.update(
                            where={"id": waitlist_id},
                            data={"unaffiliatedEmailUsers": current_emails},
                        )
                        # Mask email for logging to avoid PII exposure
                        parts = email.split("@") if "@" in email else []
                        local = parts[0] if len(parts) > 0 else ""
                        domain = parts[1] if len(parts) > 1 else "unknown"
                        masked = (local[0] if local else "?") + "***@" + domain
                        logger.info(f"Email {masked} added to waitlist {waitlist_id}")
                    else:
                        logger.debug(f"Email already exists on waitlist {waitlist_id}")

        # Re-fetch to return updated data
        updated_waitlist = await prisma.models.WaitlistEntry.prisma().find_unique(
            where={"id": waitlist_id}
        )
        return _waitlist_to_store_entry(updated_waitlist or waitlist)

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error adding user to waitlist: {e}")
        raise DatabaseError("Failed to add user to waitlist") from e


# ============== Admin Waitlist Functions ==============


def _waitlist_to_admin_response(
    waitlist: prisma.models.WaitlistEntry,
) -> store_model.WaitlistAdminResponse:
    """Convert a WaitlistEntry to WaitlistAdminResponse."""
    joined_count = len(waitlist.joinedUsers) if waitlist.joinedUsers else 0
    email_count = (
        len(waitlist.unaffiliatedEmailUsers) if waitlist.unaffiliatedEmailUsers else 0
    )

    return store_model.WaitlistAdminResponse(
        id=waitlist.id,
        createdAt=waitlist.createdAt.isoformat() if waitlist.createdAt else "",
        updatedAt=waitlist.updatedAt.isoformat() if waitlist.updatedAt else "",
        slug=waitlist.slug,
        name=waitlist.name,
        subHeading=waitlist.subHeading,
        description=waitlist.description,
        categories=waitlist.categories or [],
        imageUrls=waitlist.imageUrls or [],
        videoUrl=waitlist.videoUrl,
        agentOutputDemoUrl=waitlist.agentOutputDemoUrl,
        status=waitlist.status or prisma.enums.WaitlistExternalStatus.NOT_STARTED,
        votes=waitlist.votes,
        signupCount=joined_count + email_count,
        storeListingId=waitlist.storeListingId,
        owningUserId=waitlist.owningUserId,
    )


async def create_waitlist_admin(
    admin_user_id: str,
    data: store_model.WaitlistCreateRequest,
) -> store_model.WaitlistAdminResponse:
    """Create a new waitlist (admin only)."""
    logger.info(f"Admin {admin_user_id} creating waitlist: {data.name}")

    try:
        waitlist = await prisma.models.WaitlistEntry.prisma().create(
            data=prisma.types.WaitlistEntryCreateInput(
                name=data.name,
                slug=data.slug,
                subHeading=data.subHeading,
                description=data.description,
                categories=data.categories,
                imageUrls=data.imageUrls,
                videoUrl=data.videoUrl,
                agentOutputDemoUrl=data.agentOutputDemoUrl,
                owningUserId=admin_user_id,
                status=prisma.enums.WaitlistExternalStatus.NOT_STARTED,
            ),
            include={"joinedUsers": True},
        )

        return _waitlist_to_admin_response(waitlist)
    except Exception as e:
        logger.error(f"Error creating waitlist: {e}")
        raise DatabaseError("Failed to create waitlist") from e


async def get_waitlists_admin() -> store_model.WaitlistAdminListResponse:
    """Get all waitlists with admin details."""
    try:
        waitlists = await prisma.models.WaitlistEntry.prisma().find_many(
            where=prisma.types.WaitlistEntryWhereInput(isDeleted=False),
            include={"joinedUsers": True},
            order={"createdAt": "desc"},
        )

        return store_model.WaitlistAdminListResponse(
            waitlists=[_waitlist_to_admin_response(w) for w in waitlists],
            totalCount=len(waitlists),
        )
    except Exception as e:
        logger.error(f"Error fetching waitlists for admin: {e}")
        raise DatabaseError("Failed to fetch waitlists") from e


async def get_waitlist_admin(
    waitlist_id: str,
) -> store_model.WaitlistAdminResponse:
    """Get a single waitlist with admin details."""
    try:
        waitlist = await prisma.models.WaitlistEntry.prisma().find_unique(
            where={"id": waitlist_id},
            include={"joinedUsers": True},
        )

        if not waitlist:
            raise ValueError(f"Waitlist {waitlist_id} not found")

        if waitlist.isDeleted:
            raise ValueError(f"Waitlist {waitlist_id} has been deleted")

        return _waitlist_to_admin_response(waitlist)
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error fetching waitlist {waitlist_id}: {e}")
        raise DatabaseError("Failed to fetch waitlist") from e


async def update_waitlist_admin(
    waitlist_id: str,
    data: store_model.WaitlistUpdateRequest,
) -> store_model.WaitlistAdminResponse:
    """Update a waitlist (admin only)."""
    logger.info(f"Updating waitlist {waitlist_id}")

    try:
        # Check if waitlist exists first
        existing = await prisma.models.WaitlistEntry.prisma().find_unique(
            where={"id": waitlist_id}
        )

        if not existing:
            raise ValueError(f"Waitlist {waitlist_id} not found")

        if existing.isDeleted:
            raise ValueError(f"Waitlist {waitlist_id} has been deleted")

        # Build update data from explicitly provided fields
        # Use model_fields_set to allow clearing fields by setting them to None
        field_mappings = {
            "name": data.name,
            "slug": data.slug,
            "subHeading": data.subHeading,
            "description": data.description,
            "categories": data.categories,
            "imageUrls": data.imageUrls,
            "videoUrl": data.videoUrl,
            "agentOutputDemoUrl": data.agentOutputDemoUrl,
            "storeListingId": data.storeListingId,
        }
        update_data: dict[str, Any] = {
            k: v for k, v in field_mappings.items() if k in data.model_fields_set
        }

        # Add status if provided (already validated as enum by Pydantic)
        if "status" in data.model_fields_set and data.status is not None:
            update_data["status"] = data.status

        if not update_data:
            # No updates, just return current data
            return await get_waitlist_admin(waitlist_id)

        waitlist = await prisma.models.WaitlistEntry.prisma().update(
            where={"id": waitlist_id},
            data=prisma.types.WaitlistEntryUpdateInput(**update_data),
            include={"joinedUsers": True},
        )

        # We already verified existence above, so this should never be None
        assert waitlist is not None
        return _waitlist_to_admin_response(waitlist)
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error updating waitlist {waitlist_id}: {e}")
        raise DatabaseError("Failed to update waitlist") from e


async def delete_waitlist_admin(waitlist_id: str) -> None:
    """Soft delete a waitlist (admin only)."""
    logger.info(f"Soft deleting waitlist {waitlist_id}")

    try:
        # Check if waitlist exists first
        waitlist = await prisma.models.WaitlistEntry.prisma().find_unique(
            where={"id": waitlist_id},
        )

        if not waitlist:
            raise ValueError(f"Waitlist {waitlist_id} not found")

        if waitlist.isDeleted:
            raise ValueError(f"Waitlist {waitlist_id} has already been deleted")

        await prisma.models.WaitlistEntry.prisma().update(
            where={"id": waitlist_id},
            data={"isDeleted": True},
        )
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error deleting waitlist {waitlist_id}: {e}")
        raise DatabaseError("Failed to delete waitlist") from e


async def get_waitlist_signups_admin(
    waitlist_id: str,
) -> store_model.WaitlistSignupListResponse:
    """Get all signups for a waitlist (admin only)."""
    try:
        waitlist = await prisma.models.WaitlistEntry.prisma().find_unique(
            where={"id": waitlist_id},
            include={"joinedUsers": True},
        )

        if not waitlist:
            raise ValueError(f"Waitlist {waitlist_id} not found")

        signups: list[store_model.WaitlistSignup] = []

        # Add user signups
        for user in waitlist.joinedUsers or []:
            signups.append(
                store_model.WaitlistSignup(
                    type="user",
                    userId=user.id,
                    email=user.email,
                    username=user.name,
                )
            )

        # Add email signups
        for email in waitlist.unaffiliatedEmailUsers or []:
            signups.append(
                store_model.WaitlistSignup(
                    type="email",
                    email=email,
                )
            )

        return store_model.WaitlistSignupListResponse(
            waitlistId=waitlist_id,
            signups=signups,
            totalCount=len(signups),
        )
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error fetching signups for waitlist {waitlist_id}: {e}")
        raise DatabaseError("Failed to fetch waitlist signups") from e


async def link_waitlist_to_listing_admin(
    waitlist_id: str,
    store_listing_id: str,
) -> store_model.WaitlistAdminResponse:
    """Link a waitlist to a store listing (admin only)."""
    logger.info(f"Linking waitlist {waitlist_id} to listing {store_listing_id}")

    try:
        # Verify the waitlist exists
        waitlist = await prisma.models.WaitlistEntry.prisma().find_unique(
            where={"id": waitlist_id}
        )

        if not waitlist:
            raise ValueError(f"Waitlist {waitlist_id} not found")

        if waitlist.isDeleted:
            raise ValueError(f"Waitlist {waitlist_id} has been deleted")

        # Verify the store listing exists
        listing = await prisma.models.StoreListing.prisma().find_unique(
            where={"id": store_listing_id}
        )

        if not listing:
            raise ValueError(f"Store listing {store_listing_id} not found")

        updated_waitlist = await prisma.models.WaitlistEntry.prisma().update(
            where={"id": waitlist_id},
            data={"StoreListing": {"connect": {"id": store_listing_id}}},
            include={"joinedUsers": True},
        )

        # We already verified existence above, so this should never be None
        assert updated_waitlist is not None
        return _waitlist_to_admin_response(updated_waitlist)
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error linking waitlist to listing: {e}")
        raise DatabaseError("Failed to link waitlist to listing") from e


async def notify_waitlist_users_on_launch(
    store_listing_id: str,
    agent_name: str,
    store_url: str,
) -> int:
    """
    Notify all users on waitlists linked to a store listing when the agent is launched.

    Args:
        store_listing_id: The ID of the store listing that was approved
        agent_name: The name of the approved agent
        store_url: The URL to the agent's store page

    Returns:
        The number of notifications sent
    """
    logger.info(f"Notifying waitlist users for store listing {store_listing_id}")

    try:
        # Find all active waitlists linked to this store listing
        # Exclude DONE and CANCELED to prevent duplicate notifications on re-approval
        waitlists = await prisma.models.WaitlistEntry.prisma().find_many(
            where={
                "storeListingId": store_listing_id,
                "isDeleted": False,
                "status": {
                    "not_in": [
                        prisma.enums.WaitlistExternalStatus.DONE,
                        prisma.enums.WaitlistExternalStatus.CANCELED,
                    ]
                },
            },
            include={"joinedUsers": True},
        )

        if not waitlists:
            logger.info(
                f"No active waitlists found for store listing {store_listing_id}"
            )
            return 0

        notification_count = 0
        launched_at = datetime.now(tz=timezone.utc)

        for waitlist in waitlists:
            # Track notification results for this waitlist
            users_to_notify = waitlist.joinedUsers or []
            failed_user_ids: list[str] = []

            # Notify registered users
            for user in users_to_notify:
                try:
                    notification_data = WaitlistLaunchData(
                        agent_name=agent_name,
                        waitlist_name=waitlist.name,
                        store_url=store_url,
                        launched_at=launched_at,
                    )

                    notification_event = NotificationEventModel[WaitlistLaunchData](
                        user_id=user.id,
                        type=prisma.enums.NotificationType.WAITLIST_LAUNCH,
                        data=notification_data,
                    )

                    await queue_notification_async(notification_event)
                    notification_count += 1
                except Exception as e:
                    logger.error(
                        f"Failed to send waitlist launch notification to user {user.id}: {e}"
                    )
                    failed_user_ids.append(user.id)

            # Note: For unaffiliated email users, you would need to send emails directly
            # since they don't have user IDs for the notification system.
            # This could be done via a separate email service.
            # For now, we log these for potential manual follow-up or future implementation.
            has_pending_email_users = bool(waitlist.unaffiliatedEmailUsers)
            if has_pending_email_users:
                logger.info(
                    f"Waitlist {waitlist.id} has {len(waitlist.unaffiliatedEmailUsers)} "
                    f"unaffiliated email users that need email notifications"
                )

            # Only mark waitlist as DONE if all registered user notifications succeeded
            # AND there are no unaffiliated email users still waiting for notifications
            if not failed_user_ids and not has_pending_email_users:
                await prisma.models.WaitlistEntry.prisma().update(
                    where={"id": waitlist.id},
                    data={"status": prisma.enums.WaitlistExternalStatus.DONE},
                )
                logger.info(f"Updated waitlist {waitlist.id} status to DONE")
            elif failed_user_ids:
                logger.warning(
                    f"Waitlist {waitlist.id} not marked as DONE due to "
                    f"{len(failed_user_ids)} failed notifications"
                )
            elif has_pending_email_users:
                logger.warning(
                    f"Waitlist {waitlist.id} not marked as DONE due to "
                    f"{len(waitlist.unaffiliatedEmailUsers)} pending email-only users"
                )

        logger.info(
            f"Sent {notification_count} waitlist launch notifications for store listing {store_listing_id}"
        )
        return notification_count

    except Exception as e:
        logger.error(
            f"Error notifying waitlist users for store listing {store_listing_id}: {e}"
        )
        # Don't raise - we don't want to fail the approval process
        return 0
