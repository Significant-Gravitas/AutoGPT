import datetime
from typing import TYPE_CHECKING, List, Self

import prisma.enums
import pydantic

from backend.util.models import Pagination

if TYPE_CHECKING:
    import prisma.models


class ChangelogEntry(pydantic.BaseModel):
    version: str
    changes_summary: str
    date: datetime.datetime


class MyUnpublishedAgent(pydantic.BaseModel):
    graph_id: str
    graph_version: int
    agent_name: str
    agent_image: str | None = None
    description: str
    last_edited: datetime.datetime
    recommended_schedule_cron: str | None = None


class MyUnpublishedAgentsResponse(pydantic.BaseModel):
    agents: list[MyUnpublishedAgent]
    pagination: Pagination


class StoreAgent(pydantic.BaseModel):
    slug: str
    agent_name: str
    agent_image: str
    creator: str
    creator_avatar: str
    sub_heading: str
    description: str
    runs: int
    rating: float
    agent_graph_id: str

    @classmethod
    def from_db(cls, agent: "prisma.models.StoreAgent") -> "StoreAgent":
        return cls(
            slug=agent.slug,
            agent_name=agent.agent_name,
            agent_image=agent.agent_image[0] if agent.agent_image else "",
            creator=agent.creator_username or "Needs Profile",
            creator_avatar=agent.creator_avatar or "",
            sub_heading=agent.sub_heading,
            description=agent.description,
            runs=agent.runs,
            rating=agent.rating,
            agent_graph_id=agent.graph_id,
        )


class StoreAgentsResponse(pydantic.BaseModel):
    agents: list[StoreAgent]
    pagination: Pagination


class StoreAgentDetails(pydantic.BaseModel):
    store_listing_version_id: str
    slug: str
    agent_name: str
    agent_video: str
    agent_output_demo: str
    agent_image: list[str]
    creator: str
    creator_avatar: str
    sub_heading: str
    description: str
    instructions: str | None = None
    categories: list[str]
    runs: int
    rating: float
    versions: list[str]
    graph_id: str
    graph_versions: list[str]
    last_updated: datetime.datetime
    recommended_schedule_cron: str | None = None

    active_version_id: str
    has_approved_version: bool

    # Optional changelog data when include_changelog=True
    changelog: list[ChangelogEntry] | None = None

    @classmethod
    def from_db(cls, agent: "prisma.models.StoreAgent") -> "StoreAgentDetails":
        return cls(
            store_listing_version_id=agent.listing_version_id,
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
            graph_id=agent.graph_id,
            graph_versions=agent.graph_versions,
            last_updated=agent.updated_at,
            recommended_schedule_cron=agent.recommended_schedule_cron,
            active_version_id=agent.listing_version_id,
            has_approved_version=True,  # StoreAgent view only has approved agents
        )


class Profile(pydantic.BaseModel):
    """Marketplace user profile (only attributes that the user can update)"""

    username: str
    name: str
    description: str
    avatar_url: str | None
    links: list[str]


class ProfileDetails(Profile):
    """Marketplace user profile (including read-only fields)"""

    is_featured: bool

    @classmethod
    def from_db(cls, profile: "prisma.models.Profile") -> "ProfileDetails":
        return cls(
            name=profile.name,
            username=profile.username,
            avatar_url=profile.avatarUrl,
            description=profile.description,
            links=profile.links,
            is_featured=profile.isFeatured,
        )


class CreatorDetails(ProfileDetails):
    """Marketplace creator profile details, including aggregated stats"""

    num_agents: int
    agent_runs: int
    agent_rating: float
    top_categories: list[str]

    @classmethod
    def from_db(cls, creator: "prisma.models.Creator") -> "CreatorDetails":  # type: ignore[override]
        return cls(
            name=creator.name,
            username=creator.username,
            avatar_url=creator.avatar_url,
            description=creator.description,
            links=creator.links,
            is_featured=creator.is_featured,
            num_agents=creator.num_agents,
            agent_runs=creator.agent_runs,
            agent_rating=creator.agent_rating,
            top_categories=creator.top_categories,
        )


class CreatorsResponse(pydantic.BaseModel):
    creators: List[CreatorDetails]
    pagination: Pagination


class StoreSubmission(pydantic.BaseModel):
    # From StoreListing:
    listing_id: str
    user_id: str
    slug: str

    # From StoreListingVersion:
    listing_version_id: str
    listing_version: int
    graph_id: str
    graph_version: int
    name: str
    sub_heading: str
    description: str
    instructions: str | None
    categories: list[str]
    image_urls: list[str]
    video_url: str | None
    agent_output_demo_url: str | None

    submitted_at: datetime.datetime | None
    changes_summary: str | None
    status: prisma.enums.SubmissionStatus
    reviewed_at: datetime.datetime | None = None
    reviewer_id: str | None = None
    review_comments: str | None = None  # External comments visible to creator

    # Aggregated from AgentGraphExecutions and StoreListingReviews:
    run_count: int = 0
    review_count: int = 0
    review_avg_rating: float = 0.0

    @classmethod
    def from_db(cls, _sub: "prisma.models.StoreSubmission") -> Self:
        """Construct from the StoreSubmission Prisma view."""
        return cls(
            listing_id=_sub.listing_id,
            user_id=_sub.user_id,
            slug=_sub.slug,
            listing_version_id=_sub.listing_version_id,
            listing_version=_sub.listing_version,
            graph_id=_sub.graph_id,
            graph_version=_sub.graph_version,
            name=_sub.name,
            sub_heading=_sub.sub_heading,
            description=_sub.description,
            instructions=_sub.instructions,
            categories=_sub.categories,
            image_urls=_sub.image_urls,
            video_url=_sub.video_url,
            agent_output_demo_url=_sub.agent_output_demo_url,
            submitted_at=_sub.submitted_at,
            changes_summary=_sub.changes_summary,
            status=_sub.status,
            reviewed_at=_sub.reviewed_at,
            reviewer_id=_sub.reviewer_id,
            review_comments=_sub.review_comments,
            run_count=_sub.run_count,
            review_count=_sub.review_count,
            review_avg_rating=_sub.review_avg_rating,
        )

    @classmethod
    def from_listing_version(cls, _lv: "prisma.models.StoreListingVersion") -> Self:
        """
        Construct from the StoreListingVersion Prisma model (with StoreListing included)
        """
        if not (_l := _lv.StoreListing):
            raise ValueError("StoreListingVersion must have included StoreListing")

        return cls(
            listing_id=_l.id,
            user_id=_l.owningUserId,
            slug=_l.slug,
            listing_version_id=_lv.id,
            listing_version=_lv.version,
            graph_id=_lv.agentGraphId,
            graph_version=_lv.agentGraphVersion,
            name=_lv.name,
            sub_heading=_lv.subHeading,
            description=_lv.description,
            instructions=_lv.instructions,
            categories=_lv.categories,
            image_urls=_lv.imageUrls,
            video_url=_lv.videoUrl,
            agent_output_demo_url=_lv.agentOutputDemoUrl,
            submitted_at=_lv.submittedAt,
            changes_summary=_lv.changesSummary,
            status=_lv.submissionStatus,
            reviewed_at=_lv.reviewedAt,
            reviewer_id=_lv.reviewerId,
            review_comments=_lv.reviewComments,
        )


class StoreSubmissionsResponse(pydantic.BaseModel):
    submissions: list[StoreSubmission]
    pagination: Pagination


class StoreSubmissionRequest(pydantic.BaseModel):
    graph_id: str = pydantic.Field(
        ..., min_length=1, description="Graph ID cannot be empty"
    )
    graph_version: int = pydantic.Field(
        ..., gt=0, description="Graph version must be greater than 0"
    )
    slug: str
    name: str
    sub_heading: str
    video_url: str | None = None
    agent_output_demo_url: str | None = None
    image_urls: list[str] = []
    description: str = ""
    instructions: str | None = None
    categories: list[str] = []
    changes_summary: str | None = None
    recommended_schedule_cron: str | None = None


class StoreSubmissionEditRequest(pydantic.BaseModel):
    name: str
    sub_heading: str
    video_url: str | None = None
    agent_output_demo_url: str | None = None
    image_urls: list[str] = []
    description: str = ""
    instructions: str | None = None
    categories: list[str] = []
    changes_summary: str | None = None
    recommended_schedule_cron: str | None = None


class StoreSubmissionAdminView(StoreSubmission):
    internal_comments: str | None  # Private admin notes

    @classmethod
    def from_db(cls, _sub: "prisma.models.StoreSubmission") -> Self:
        return cls(
            **StoreSubmission.from_db(_sub).model_dump(),
            internal_comments=_sub.internal_comments,
        )

    @classmethod
    def from_listing_version(cls, _lv: "prisma.models.StoreListingVersion") -> Self:
        return cls(
            **StoreSubmission.from_listing_version(_lv).model_dump(),
            internal_comments=_lv.internalComments,
        )


class StoreListingWithVersionsAdminView(pydantic.BaseModel):
    """A store listing with its version history"""

    listing_id: str
    graph_id: str
    slug: str
    active_listing_version_id: str | None = None
    has_approved_version: bool = False
    creator_email: str | None = None
    latest_version: StoreSubmissionAdminView | None = None
    versions: list[StoreSubmissionAdminView] = []


class StoreListingsWithVersionsAdminViewResponse(pydantic.BaseModel):
    """Response model for listings with version history"""

    listings: list[StoreListingWithVersionsAdminView]
    pagination: Pagination


class StoreReview(pydantic.BaseModel):
    score: int
    comments: str | None = None


class StoreReviewCreate(pydantic.BaseModel):
    store_listing_version_id: str
    score: int
    comments: str | None = None


class ReviewSubmissionRequest(pydantic.BaseModel):
    store_listing_version_id: str
    is_approved: bool
    comments: str  # External comments visible to creator
    internal_comments: str | None = None  # Private admin notes


class UnifiedSearchResult(pydantic.BaseModel):
    """A single result from unified hybrid search across all content types."""

    content_type: str  # STORE_AGENT, BLOCK, DOCUMENTATION
    content_id: str
    searchable_text: str
    metadata: dict | None = None
    updated_at: datetime.datetime | None = None
    combined_score: float | None = None
    semantic_score: float | None = None
    lexical_score: float | None = None


class UnifiedSearchResponse(pydantic.BaseModel):
    """Response model for unified search across all content types."""

    results: list[UnifiedSearchResult]
    pagination: Pagination
