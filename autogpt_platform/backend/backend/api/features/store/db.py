import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, overload

import prisma.enums
import prisma.errors
import prisma.models
import prisma.types

from backend.data.db import query_raw_with_schema, transaction
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
)
from backend.notifications.notifications import queue_notification_async
from backend.util.exceptions import DatabaseError, NotFoundError, PreconditionFailed
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


class StoreAgentsSortOptions(Enum):
    RATING = "rating"
    RUNS = "runs"
    NAME = "name"
    UPDATED_AT = "updated_at"


async def get_store_agents(
    featured: bool = False,
    creators: list[str] | None = None,
    sorted_by: StoreAgentsSortOptions | None = None,
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
        "Getting store agents: "
        f"featured={featured}, creators={creators}, sorted_by={sorted_by}, "
        f"query={search_query}, category={category}, page={page}"
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
            # Fall through to direct DB search if hybrid returned nothing
            if search_used_hybrid and agents:
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
                            agent_graph_id=agent.get("graph_id", ""),
                        )
                        store_agents.append(store_agent)
                    except Exception as e:
                        logger.error(
                            f"Error parsing Store agent from hybrid search results: {e}"
                        )
                        continue

        if not search_used_hybrid or not agents:
            # Fallback path: direct DB query with optional tsvector search.
            # This mirrors the original pre-hybrid-search implementation.
            store_agents, total = await _fallback_store_agent_search(
                search_query=search_query,
                featured=featured,
                creators=creators,
                category=category,
                sorted_by=sorted_by,
                page=page,
                page_size=page_size,
            )
            total_pages = (total + page_size - 1) // page_size

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


async def _fallback_store_agent_search(
    *,
    search_query: str | None,
    featured: bool,
    creators: list[str] | None,
    category: str | None,
    sorted_by: StoreAgentsSortOptions | None,
    page: int,
    page_size: int,
) -> tuple[list[store_model.StoreAgent], int]:
    """Direct DB search fallback when hybrid search is unavailable or empty.

    Uses ad-hoc to_tsvector/plainto_tsquery with ts_rank_cd for text search,
    matching the quality of the original pre-hybrid-search implementation.
    Falls back to simple listing when no search query is provided.
    """
    if not search_query:
        # No search query — use Prisma for simple filtered listing
        where_clause: prisma.types.StoreAgentWhereInput = {"is_available": True}
        if featured:
            where_clause["featured"] = featured
        if creators:
            where_clause["creator_username"] = {"in": creators}
        if category:
            where_clause["categories"] = {"has": category}

        order_by = []
        if sorted_by == StoreAgentsSortOptions.RATING:
            order_by.append({"rating": "desc"})
        elif sorted_by == StoreAgentsSortOptions.RUNS:
            order_by.append({"runs": "desc"})
        elif sorted_by == StoreAgentsSortOptions.NAME:
            order_by.append({"agent_name": "asc"})
        elif sorted_by == StoreAgentsSortOptions.UPDATED_AT:
            order_by.append({"updated_at": "desc"})

        db_agents = await prisma.models.StoreAgent.prisma().find_many(
            where=where_clause,
            order=order_by,
            skip=(page - 1) * page_size,
            take=page_size,
        )
        total = await prisma.models.StoreAgent.prisma().count(where=where_clause)
        return [store_model.StoreAgent.from_db(a) for a in db_agents], total

    # Text search using ad-hoc tsvector on StoreAgent view fields
    params: list[Any] = [search_query]
    filters = ["sa.is_available = true"]
    param_idx = 2

    if featured:
        filters.append("sa.featured = true")
    if creators:
        params.append(creators)
        filters.append(f"sa.creator_username = ANY(${param_idx})")
        param_idx += 1
    if category:
        params.append(category)
        filters.append(f"${param_idx} = ANY(sa.categories)")
        param_idx += 1

    where_sql = " AND ".join(filters)

    params.extend([page_size, (page - 1) * page_size])
    limit_param = f"${param_idx}"
    param_idx += 1
    offset_param = f"${param_idx}"

    sql = f"""
        WITH ranked AS (
            SELECT sa.*,
                ts_rank_cd(
                    to_tsvector('english',
                        COALESCE(sa.agent_name, '') || ' ' ||
                        COALESCE(sa.sub_heading, '') || ' ' ||
                        COALESCE(sa.description, '')
                    ),
                    plainto_tsquery('english', $1)
                ) AS rank,
                COUNT(*) OVER () AS total_count
            FROM {{schema_prefix}}"StoreAgent" sa
            WHERE {where_sql}
            AND to_tsvector('english',
                    COALESCE(sa.agent_name, '') || ' ' ||
                    COALESCE(sa.sub_heading, '') || ' ' ||
                    COALESCE(sa.description, '')
                ) @@ plainto_tsquery('english', $1)
        )
        SELECT * FROM ranked
        ORDER BY rank DESC
        LIMIT {limit_param} OFFSET {offset_param}
    """

    results = await query_raw_with_schema(sql, *params)
    total = results[0]["total_count"] if results else 0

    store_agents = []
    for row in results:
        try:
            store_agents.append(
                store_model.StoreAgent(
                    slug=row["slug"],
                    agent_name=row["agent_name"],
                    agent_image=row["agent_image"][0] if row["agent_image"] else "",
                    creator=row["creator_username"] or "Needs Profile",
                    creator_avatar=row["creator_avatar"] or "",
                    sub_heading=row["sub_heading"],
                    description=row["description"],
                    runs=row["runs"],
                    rating=row["rating"],
                    agent_graph_id=row.get("graph_id", ""),
                )
            )
        except Exception as e:
            logger.error(f"Error parsing StoreAgent from fallback search: {e}")
            continue

    return store_agents, total


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
            raise NotFoundError(f"Agent {username}/{agent_name} not found")

        # Fetch changelog data if requested
        changelog_data = None
        if include_changelog:
            changelog_versions = (
                await prisma.models.StoreListingVersion.prisma().find_many(
                    where={
                        "storeListingId": agent.listing_id,
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
        details = store_model.StoreAgentDetails.from_db(agent)
        details.changelog = changelog_data
        return details
    except NotFoundError:
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
            raise NotFoundError(
                f"Store listing version {store_listing_version_id} not found",
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
    """Get agent details from the StoreAgent view (APPROVED agents only).

    See also: `get_store_agent_details_as_admin()` which bypasses the
    APPROVED-only StoreAgent view for admin preview of pending submissions.
    """
    logger.debug(f"Getting store agent details for {store_listing_version_id}")

    try:
        agent = await prisma.models.StoreAgent.prisma().find_first(
            where={"listing_version_id": store_listing_version_id}
        )

        if not agent:
            logger.warning(f"Agent not found: {store_listing_version_id}")
            raise NotFoundError(f"Agent {store_listing_version_id} not found")

        logger.debug(f"Found agent details for {store_listing_version_id}")
        return store_model.StoreAgentDetails.from_db(agent)
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error getting store agent details: {e}")
        raise DatabaseError("Failed to fetch agent details") from e


async def get_store_agent_details_as_admin(
    store_listing_version_id: str,
) -> store_model.StoreAgentDetails:
    """Get agent details for admin preview, bypassing the APPROVED-only
    StoreAgent view. Queries StoreListingVersion directly so pending
    submissions are visible."""
    slv = await prisma.models.StoreListingVersion.prisma().find_unique(
        where={"id": store_listing_version_id},
        include={
            "StoreListing": {"include": {"CreatorProfile": True}},
        },
    )
    if not slv or not slv.StoreListing:
        raise NotFoundError(
            f"Store listing version {store_listing_version_id} not found"
        )

    listing = slv.StoreListing
    # CreatorProfile is a required FK relation — should always exist.
    # If it's None, the DB is in a bad state.
    profile = listing.CreatorProfile
    if not profile:
        raise DatabaseError(
            f"StoreListing {listing.id} has no CreatorProfile — FK violated"
        )

    return store_model.StoreAgentDetails(
        store_listing_version_id=slv.id,
        slug=listing.slug,
        agent_name=slv.name,
        agent_video=slv.videoUrl or "",
        agent_output_demo=slv.agentOutputDemoUrl or "",
        agent_image=slv.imageUrls,
        creator=profile.username,
        creator_avatar=profile.avatarUrl or "",
        sub_heading=slv.subHeading,
        description=slv.description,
        instructions=slv.instructions,
        categories=slv.categories,
        runs=0,
        rating=0.0,
        versions=[str(slv.version)],
        graph_id=slv.agentGraphId,
        graph_versions=[str(slv.agentGraphVersion)],
        last_updated=slv.updatedAt,
        recommended_schedule_cron=slv.recommendedScheduleCron,
        active_version_id=listing.activeVersionId or slv.id,
        has_approved_version=listing.hasApprovedVersion,
    )


class StoreCreatorsSortOptions(Enum):
    # NOTE: values correspond 1:1 to columns of the Creator view
    AGENT_RATING = "agent_rating"
    AGENT_RUNS = "agent_runs"
    NUM_AGENTS = "num_agents"


async def get_store_creators(
    featured: bool = False,
    search_query: str | None = None,
    sorted_by: StoreCreatorsSortOptions | None = None,
    page: int = 1,
    page_size: int = 20,
) -> store_model.CreatorsResponse:
    """Get PUBLIC store creators from the Creator view"""
    logger.debug(
        "Getting store creators: "
        f"featured={featured}, query={search_query}, sorted_by={sorted_by}, page={page}"
    )

    # Build where clause with sanitized inputs
    where = {}

    # Only return creators with approved agents
    where["num_agents"] = {"gt": 0}

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

        order: prisma.types.CreatorOrderByInput = (
            {"agent_rating": "desc"}
            if sorted_by == StoreCreatorsSortOptions.AGENT_RATING
            else (
                {"agent_runs": "desc"}
                if sorted_by == StoreCreatorsSortOptions.AGENT_RUNS
                else (
                    {"num_agents": "desc"}
                    if sorted_by == StoreCreatorsSortOptions.NUM_AGENTS
                    else {"username": "asc"}
                )
            )
        )

        # Execute query with sanitized parameters
        creators = await prisma.models.Creator.prisma().find_many(
            where=prisma.types.CreatorWhereInput(**where),
            skip=skip,
            take=take,
            order=order,
        )

        # Convert to response model
        creator_models = [
            store_model.CreatorDetails.from_db(creator) for creator in creators
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


async def get_store_creator(
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
        return store_model.CreatorDetails.from_db(creator)
    except store_exceptions.CreatorNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error getting store creator details: {e}")
        raise DatabaseError("Failed to fetch creator details") from e


async def _get_submission_stats(user_id: str) -> store_model.SubmissionStats:
    """Compute creator-wide submission aggregates in a single round-trip.

    Uses Postgres FILTER clauses so all five aggregates land in one query —
    cheaper than five separate counts/sums and immune to the pagination
    undercount that client-side aggregation suffers from.
    """
    # average_rating is weighted by review_count so a submission with 1,000
    # reviews counts proportionally more than one with a single review;
    # straight AVG would over-represent low-volume submissions.
    sql = """
        SELECT
            COUNT(*)::int                                          AS total,
            COUNT(*) FILTER (WHERE status = 'APPROVED')::int       AS approved,
            COUNT(*) FILTER (WHERE status = 'PENDING')::int        AS pending,
            COALESCE(SUM(run_count), 0)::bigint                    AS total_runs,
            (
                SUM(review_avg_rating * review_count)
                FILTER (WHERE review_count > 0 AND review_avg_rating > 0)
            )::double precision
            / NULLIF(
                SUM(review_count) FILTER (
                    WHERE review_count > 0 AND review_avg_rating > 0
                ),
                0
            )                                                      AS average_rating
        FROM {schema_prefix}"StoreSubmission"
        WHERE user_id = $1 AND is_deleted = false
    """
    rows = await query_raw_with_schema(
        sql,
        user_id,
        model=store_model.SubmissionStats,
    )
    return (
        rows[0]
        if rows
        else store_model.SubmissionStats(
            total=0,
            approved=0,
            pending=0,
            total_runs=0,
            average_rating=None,
        )
    )


async def get_store_submissions(
    user_id: str, page: int = 1, page_size: int = 20
) -> store_model.StoreSubmissionsResponse:
    """Get store submissions for the authenticated user -- not an admin"""
    logger.debug(f"Getting store submissions for user {user_id}, page={page}")

    try:
        skip = (page - 1) * page_size

        where: prisma.types.StoreSubmissionWhereInput = {
            "user_id": user_id,
            "is_deleted": False,
        }

        # Page rows and creator-wide stats run concurrently — stats already
        # returns COUNT(*) over the same filter, so we reuse it for pagination
        # instead of issuing a redundant count query.
        submissions, stats = await asyncio.gather(
            prisma.models.StoreSubmission.prisma().find_many(
                where=where,
                skip=skip,
                take=page_size,
                order=[{"submitted_at": "desc"}],
            ),
            _get_submission_stats(user_id),
        )

        total = stats.total
        total_pages = (total + page_size - 1) // page_size

        submission_models = [
            store_model.StoreSubmission.from_db(sub) for sub in submissions
        ]

        logger.debug(f"Found {len(submission_models)} submissions")
        return store_model.StoreSubmissionsResponse(
            submissions=submission_models,
            pagination=store_model.Pagination(
                current_page=page,
                total_items=total,
                total_pages=total_pages,
                page_size=page_size,
            ),
            stats=stats,
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
            stats=store_model.SubmissionStats(
                total=0,
                approved=0,
                pending=0,
                total_runs=0,
                average_rating=None,
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
    graph_id: str,
    graph_version: int,
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
        graph_id: ID of the agent graph being submitted
        graph_version: Version of the agent graph being submitted
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
        f"Creating store submission for user #{user_id}, "
        f"graph #{graph_id} v{graph_version}"
    )

    try:
        # Sanitize slug to only allow letters and hyphens
        slug = "".join(
            c if c.isalpha() or c == "-" or c.isnumeric() else "" for c in slug
        ).lower()

        # First verify the agent graph belongs to this user
        graph = await prisma.models.AgentGraph.prisma().find_first(
            where={"id": graph_id, "version": graph_version, "userId": user_id},
            include={"User": {"include": {"Profile": True}}},
        )

        if not graph:
            logger.warning(
                f"Agent graph {graph_id} v{graph_version} not found for user {user_id}"
            )
            # Provide more user-friendly error message when graph_id is empty
            if not graph_id or graph_id.strip() == "":
                raise ValueError(
                    "No agent selected. "
                    "Please select an agent before submitting to the store."
                )
            else:
                raise NotFoundError(
                    f"Agent #{graph_id} v{graph_version} not found "
                    f"for this user (#{user_id})"
                )

        if not graph.User or not graph.User.Profile:
            logger.warning(f"User #{user_id} does not have a Profile")
            raise PreconditionFailed(
                "User must create a Marketplace Profile before submitting an agent"
            )

        async with transaction() as tx:
            # Determine next version number for this listing
            existing_listing = await prisma.models.StoreListing.prisma(tx).find_unique(
                where={"agentGraphId": graph_id},
                include={
                    "Versions": {
                        # We just need the latest version and one of each status:
                        "order_by": {"version": "desc"},
                        "distinct": ["submissionStatus"],
                        "where": {"isDeleted": False},
                    }
                },
            )
            next_version = 1
            graph_has_pending_submissions = False
            if existing_listing and existing_listing.Versions:
                current_latest_version = max(
                    (slv.version for slv in existing_listing.Versions), default=0
                )
                next_version = current_latest_version + 1

                graph_has_pending_submissions = any(
                    slv.submissionStatus == prisma.enums.SubmissionStatus.PENDING
                    for slv in existing_listing.Versions
                )

            # Delete any currently PENDING submissions for the same graph
            # in favor of the new submission
            if graph_has_pending_submissions:
                await prisma.models.StoreListingVersion.prisma(tx).update_many(
                    where={
                        "agentGraphId": graph.id,
                        "submissionStatus": prisma.enums.SubmissionStatus.PENDING,
                        "isDeleted": False,
                    },
                    data={"isDeleted": True},
                )

            new_submission = await prisma.models.StoreListingVersion.prisma(tx).create(
                data={
                    "AgentGraph": {
                        "connect": {
                            "graphVersionId": {
                                "id": graph_id,
                                "version": graph_version,
                            }
                        }
                    },
                    "name": name,
                    "version": next_version,
                    "videoUrl": video_url,
                    "agentOutputDemoUrl": agent_output_demo_url,
                    "imageUrls": image_urls,
                    "description": description,
                    "instructions": instructions,
                    "categories": categories,
                    "subHeading": sub_heading,
                    "submissionStatus": prisma.enums.SubmissionStatus.PENDING,
                    "submittedAt": datetime.now(tz=timezone.utc),
                    "changesSummary": changes_summary,
                    "recommendedScheduleCron": recommended_schedule_cron,
                    "StoreListing": {
                        "connect_or_create": {
                            "where": {"agentGraphId": graph_id},
                            "create": {
                                "slug": slug,
                                "agentGraphId": graph_id,
                                "OwningUser": {"connect": {"id": user_id}},
                                "CreatorProfile": {"connect": {"userId": user_id}},
                            },
                        }
                    },
                },
                include={"StoreListing": True},
            )

        if not new_submission:
            raise DatabaseError("Failed to create store listing version")

        logger.debug(f"Created store listing for agent {graph_id}")
        return store_model.StoreSubmission.from_listing_version(new_submission)
    except prisma.errors.UniqueViolationError as exc:
        # Attempt to check if the error was due to the slug field being unique
        error_str = str(exc)
        if "slug" in error_str.lower():
            logger.debug(f"Slug '{slug}' is already in use by graph #{graph_id}")
            raise store_exceptions.SlugAlreadyInUseError(
                f"The slug '{slug}' is already in use by another one of your agents. "
                "Please choose a different slug."
            ) from exc
        else:
            # Reraise as a generic database error for other unique violations
            raise DatabaseError(
                f"Unique constraint violated (not slug): {error_str}"
            ) from exc
    except (
        NotFoundError,
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
            include={"StoreListing": True},
        )

        if not current_version:
            raise store_exceptions.SubmissionNotFoundError(
                f"Store listing version not found: {store_listing_version_id}"
            )

        # Verify the user owns this listing (submission)
        if (
            not current_version.StoreListing
            or current_version.StoreListing.owningUserId != user_id
        ):
            raise store_exceptions.UnauthorizedError(
                f"User {user_id} does not own submission {store_listing_version_id}"
            )

        # Only allow editing of PENDING submissions
        if current_version.submissionStatus != prisma.enums.SubmissionStatus.PENDING:
            display_status = current_version.submissionStatus.value.lower()
            raise store_exceptions.InvalidOperationError(
                f"Cannot edit a {display_status} submission. "
                "Only pending submissions can be edited."
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
            include={"StoreListing": True},
        )
        if not updated_version:
            raise DatabaseError("Failed to update store listing version")

        logger.debug(
            f"Updated existing listing version {store_listing_version_id} "
            f"for graph {current_version.agentGraphId}"
        )

        return store_model.StoreSubmission.from_listing_version(updated_version)

    except (
        store_exceptions.SubmissionNotFoundError,
        store_exceptions.UnauthorizedError,
        NotFoundError,
        store_exceptions.ListingExistsError,
        store_exceptions.InvalidOperationError,
    ):
        raise
    except prisma.errors.PrismaError as e:
        logger.error(f"Database error editing store submission: {e}")
        raise DatabaseError("Failed to edit store submission") from e


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
        return store_model.ProfileDetails.from_db(profile)
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise DatabaseError("Failed to get user profile") from e


async def update_profile(
    user_id: str, profile: store_model.Profile
) -> store_model.ProfileDetails:
    """
    Update the store profile for a user or create a new one if it doesn't exist.
    Args:
        user_id: ID of the authenticated user
        profile: Updated profile details
    Returns:
        ProfileDetails: The updated or created profile details
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
                f"Unauthorized update attempt for profile {existing_profile.id} "
                f"by user {user_id}"
            )
            raise DatabaseError(
                f"Unauthorized update attempt for profile {existing_profile.id} "
                f"by user {user_id}"
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

        return store_model.ProfileDetails.from_db(updated_profile)

    except prisma.errors.PrismaError as e:
        logger.error(f"Database error updating profile: {e}")
        raise DatabaseError("Failed to update profile") from e


async def get_my_agents(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
    sort_by: store_model.MyAgentsSortBy = store_model.MyAgentsSortBy.MOST_RECENT,
) -> store_model.MyUnpublishedAgentsResponse:
    """Get the agents for the authenticated user"""
    logger.debug(
        f"Getting my agents for user {user_id}, page={page}, "
        f"sort_by={sort_by.value}"
    )

    try:
        search_filter: prisma.types.LibraryAgentWhereInput = {
            "userId": user_id,
            # Filter for unpublished agents only:
            "AgentGraph": {
                "is": {
                    "StoreListingVersions": {
                        "none": {
                            "isAvailable": True,
                            "StoreListing": {"is": {"isDeleted": False}},
                        }
                    }
                }
            },
            "isArchived": False,
            "isDeleted": False,
        }

        if sort_by == store_model.MyAgentsSortBy.NAME:
            order: list = [
                {"AgentGraph": {"name": "asc"}},
                {"updatedAt": "desc"},
            ]
        else:
            order = [{"updatedAt": "desc"}]

        library_agents = await prisma.models.LibraryAgent.prisma().find_many(
            where=search_filter,
            order=order,
            skip=(page - 1) * page_size,
            take=page_size,
            include={"AgentGraph": True},
        )

        total = await prisma.models.LibraryAgent.prisma().count(where=search_filter)
        total_pages = (total + page_size - 1) // page_size

        my_agents = [
            store_model.MyUnpublishedAgent(
                graph_id=graph.id,
                graph_version=graph.version,
                agent_name=graph.name or "",
                last_edited=graph.updatedAt or graph.createdAt,
                description=graph.description or "",
                agent_image=library_agent.imageUrl,
                recommended_schedule_cron=graph.recommendedScheduleCron,
            )
            for library_agent in library_agents
            if (graph := library_agent.AgentGraph)
        ]

        return store_model.MyUnpublishedAgentsResponse(
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
    slv = await prisma.models.StoreListingVersion.prisma().find_unique(
        where={"id": store_listing_version_id}
    )

    if not slv:
        raise NotFoundError(
            f"Store listing version {store_listing_version_id} not found"
        )

    graph = await get_graph(
        graph_id=slv.agentGraphId,
        version=slv.agentGraphVersion,
        user_id=None,
        for_export=True,
    )
    if not graph:
        raise NotFoundError(
            f"Graph {slv.agentGraphId} v{slv.agentGraphVersion} not found"
        )
    return graph


#####################################################
################## ADMIN FUNCTIONS ##################
#####################################################


async def review_store_submission(
    store_listing_version_id: str,
    is_approved: bool,
    external_comments: str,
    internal_comments: str,
    reviewer_id: str,
) -> store_model.StoreSubmissionAdminView:
    """Review a store listing submission as an admin."""
    try:
        submission = await prisma.models.StoreListingVersion.prisma().find_unique(
            where={"id": store_listing_version_id},
            include={"AgentGraph": {"include": AGENT_GRAPH_INCLUDE}},
        )

        if not submission:
            raise NotFoundError(
                f"Store listing version {store_listing_version_id} not found"
            )
        assert submission.AgentGraph is not None
        creator_user_id = submission.AgentGraph.userId

        # Check if we're rejecting an already approved agent
        is_rejecting_approved = (
            not is_approved
            and submission.submissionStatus == prisma.enums.SubmissionStatus.APPROVED
        )

        # If approving, update the listing to indicate it has an approved version
        if is_approved:
            async with transaction() as tx:
                # Handle sub-agent approvals in transaction
                await asyncio.gather(
                    *[
                        _approve_sub_agent(
                            tx,
                            sub_graph,
                            submission.name,
                            submission.agentGraphVersion,
                            creator_user_id,
                        )
                        for sub_graph in await get_sub_graphs(submission.AgentGraph)
                    ]
                )

                # Update the AgentGraph with store listing data
                await prisma.models.AgentGraph.prisma(tx).update(
                    where={
                        "graphVersionId": {
                            "id": submission.agentGraphId,
                            "version": submission.agentGraphVersion,
                        }
                    },
                    data={
                        "name": submission.name,
                        "description": submission.description,
                        "recommendedScheduleCron": submission.recommendedScheduleCron,
                        "instructions": submission.instructions,
                    },
                )

                # Generate embedding for approved listing (best-effort)
                try:
                    await ensure_embedding(
                        version_id=store_listing_version_id,
                        name=submission.name,
                        description=submission.description,
                        sub_heading=submission.subHeading,
                        categories=submission.categories,
                        tx=tx,
                    )
                except Exception as emb_err:
                    logger.warning(
                        f"Could not generate embedding for listing "
                        f"{store_listing_version_id}: {emb_err}"
                    )

                await prisma.models.StoreListing.prisma(tx).update(
                    where={"id": submission.storeListingId},
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
                        "storeListingId": submission.storeListingId,
                        "id": {"not": store_listing_version_id},
                        "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
                    }
                )
            )

            if not other_approved:
                # No other approved versions, update hasApprovedVersion to False
                await prisma.models.StoreListing.prisma().update(
                    where={"id": submission.storeListingId},
                    data={
                        "hasApprovedVersion": False,
                        "ActiveVersion": {"disconnect": True},
                    },
                )
            else:
                # Set the most recent other approved version as active
                await prisma.models.StoreListing.prisma().update(
                    where={"id": submission.storeListingId},
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
            "reviewedAt": datetime.now(tz=timezone.utc),
            "Reviewer": {"connect": {"id": reviewer_id}},
            "reviewComments": external_comments,
            "internalComments": internal_comments,
        }

        # Update the version
        reviewed_submission = await prisma.models.StoreListingVersion.prisma().update(
            where={"id": store_listing_version_id},
            data=update_data,
            include={
                "StoreListing": True,  # required for StoreSubmissionAdminView
                "Reviewer": True,  # used in _send_submission_review_notification
            },
        )

        if not reviewed_submission:
            raise DatabaseError(
                f"Failed to update store listing version {store_listing_version_id}"
            )

        try:
            await _send_submission_review_notification(
                creator_user_id,
                is_approved,
                external_comments,
                reviewed_submission,
            )
        except Exception as e:
            logger.error(f"Failed to send email notification for agent review: {e}")
            # Don't fail the review process if email sending fails

        return store_model.StoreSubmissionAdminView.from_listing_version(
            reviewed_submission
        )

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Could not create store submission review: {e}")
        raise DatabaseError("Failed to create store submission review") from e


async def _approve_sub_agent(
    tx,
    sub_graph: prisma.models.AgentGraph,
    main_agent_name: str,
    main_agent_graph_version: int,
    main_agent_user_id: str,
) -> None:
    """Approve a single sub-agent by creating/updating store listings as needed"""
    heading = f"Sub-agent of {main_agent_name} v{main_agent_graph_version}"

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


async def _send_submission_review_notification(
    creator_user_id: str,
    is_approved: bool,
    external_comments: str,
    reviewed_listing_version: prisma.models.StoreListingVersion,
):
    """Send email notification to the agent creator"""
    reviewer = (
        reviewed_listing_version.Reviewer if reviewed_listing_version.Reviewer else None
    )

    base_url = settings.config.frontend_base_url or settings.config.platform_base_url

    if is_approved:
        store_agent = await prisma.models.StoreAgent.prisma().find_first_or_raise(
            where={"listing_version_id": reviewed_listing_version.id}
        )

        # Send approval notification
        creator_username = store_agent.creator_username
        notification_data = AgentApprovalData(
            agent_name=reviewed_listing_version.name,
            graph_id=reviewed_listing_version.agentGraphId,
            graph_version=reviewed_listing_version.agentGraphVersion,
            reviewer_name=(
                reviewer.name if reviewer and reviewer.name else DEFAULT_ADMIN_NAME
            ),
            reviewer_email=(reviewer.email if reviewer else DEFAULT_ADMIN_EMAIL),
            comments=external_comments,
            reviewed_at=(
                reviewed_listing_version.reviewedAt or datetime.now(tz=timezone.utc)
            ),
            store_url=(
                f"{base_url}/marketplace/agent/{creator_username}/{store_agent.slug}"
            ),
        )

        notification_event = NotificationEventModel[AgentApprovalData](
            user_id=creator_user_id,
            type=prisma.enums.NotificationType.AGENT_APPROVED,
            data=notification_data,
        )
    else:
        # Send rejection notification
        graph_id = reviewed_listing_version.agentGraphId
        notification_data = AgentRejectionData(
            agent_name=reviewed_listing_version.name,
            graph_id=reviewed_listing_version.agentGraphId,
            graph_version=reviewed_listing_version.agentGraphVersion,
            reviewer_name=(
                reviewer.name if reviewer and reviewer.name else DEFAULT_ADMIN_NAME
            ),
            reviewer_email=(reviewer.email if reviewer else DEFAULT_ADMIN_EMAIL),
            comments=external_comments,
            reviewed_at=reviewed_listing_version.reviewedAt
            or datetime.now(tz=timezone.utc),
            resubmit_url=f"{base_url}/build?flowID={graph_id}",
        )

        notification_event = NotificationEventModel[AgentRejectionData](
            user_id=creator_user_id,
            type=prisma.enums.NotificationType.AGENT_REJECTED,
            data=notification_data,
        )

    # Queue the notification for immediate sending
    await queue_notification_async(notification_event)
    logger.info(
        f"Queued {'approval' if is_approved else 'rejection'} notification "
        f"for agent '{reviewed_listing_version.name}' of user #{creator_user_id}"
    )


async def get_admin_listings_with_versions(
    status: prisma.enums.SubmissionStatus | None = None,
    search_query: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> store_model.StoreListingsWithVersionsAdminViewResponse:
    """
    Get store listings for admins with all their versions.

    Args:
        status: Filter by submission status (PENDING, APPROVED, REJECTED)
        search_query: Search by name, description, or user email
        page: Page number for pagination
        page_size: Number of items per page

    Returns:
        Paginated listings with their versions
    """
    logger.debug(
        "Getting admin store listings: "
        f"status={status}, search={search_query}, page={page}"
    )

    # Build the where clause for StoreListing
    store_listing_filter: prisma.types.StoreListingWhereInput = {
        "isDeleted": False,
    }
    if status:
        store_listing_filter["Versions"] = {"some": {"submissionStatus": status}}

    if search_query:
        # Find users with matching email
        matching_users = await prisma.models.User.prisma().find_many(
            where={"email": {"contains": search_query, "mode": "insensitive"}},
        )

        user_ids = [user.id for user in matching_users]

        # Set up OR conditions
        store_listing_filter["OR"] = [
            {"slug": {"contains": search_query, "mode": "insensitive"}},
            {
                "Versions": {
                    "some": {"name": {"contains": search_query, "mode": "insensitive"}}
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
            store_listing_filter["OR"].append({"owningUserId": {"in": user_ids}})

    # Calculate pagination
    skip = (page - 1) * page_size

    # Create proper Prisma types for the query
    include: prisma.types.StoreListingInclude = {
        "Versions": {
            "order_by": {"version": "desc"},
            "where": {"isDeleted": False},
        },
        "OwningUser": True,
    }

    # Query listings with their versions
    listings = await prisma.models.StoreListing.prisma().find_many(
        where=store_listing_filter,
        skip=skip,
        take=page_size,
        include=include,
        order=[{"createdAt": "desc"}],
    )

    # Get total count for pagination
    total = await prisma.models.StoreListing.prisma().count(where=store_listing_filter)
    total_pages = (total + page_size - 1) // page_size

    # Convert to response models
    listings_with_versions = []
    for listing in listings:
        versions: list[store_model.StoreSubmissionAdminView] = []
        # If we have versions, turn them into StoreSubmissionAdminView models
        for version in listing.Versions or []:
            # .StoreListing is required for StoreSubmission.from_listing_version(v)
            version.StoreListing = listing.model_copy(update={"Versions": None})

            versions.append(
                store_model.StoreSubmissionAdminView.from_listing_version(version)
            )

        # Get the latest version (first in the sorted list)
        latest_version = versions[0] if versions else None

        creator_email = listing.OwningUser.email if listing.OwningUser else None

        listing_with_versions = store_model.StoreListingWithVersionsAdminView(
            listing_id=listing.id,
            slug=listing.slug,
            graph_id=listing.agentGraphId,
            active_listing_version_id=listing.activeVersionId,
            has_approved_version=listing.hasApprovedVersion,
            creator_email=creator_email,
            latest_version=latest_version,
            versions=versions,
        )

        listings_with_versions.append(listing_with_versions)

    logger.debug(f"Found {len(listings_with_versions)} listings for admin")
    return store_model.StoreListingsWithVersionsAdminViewResponse(
        listings=listings_with_versions,
        pagination=store_model.Pagination(
            current_page=page,
            total_items=total,
            total_pages=total_pages,
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
    slv = await prisma.models.StoreListingVersion.prisma().find_unique(
        where={"id": store_listing_version_id}
    )

    if not slv:
        raise NotFoundError(
            f"Store listing version {store_listing_version_id} not found"
        )

    graph = await get_graph_as_admin(
        user_id=user_id,
        graph_id=slv.agentGraphId,
        version=slv.agentGraphVersion,
        for_export=True,
    )
    if not graph:
        raise NotFoundError(
            f"Graph {slv.agentGraphId} v{slv.agentGraphVersion} not found"
        )

    return graph
