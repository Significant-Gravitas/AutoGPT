"""Data models and access layer for user business understanding."""

import logging
from datetime import datetime
from typing import Any, Optional, cast

import pydantic
from prisma.models import CoPilotUnderstanding

from backend.data.redis_client import get_redis_async
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_KEY_PREFIX = "understanding"
CACHE_TTL_SECONDS = 48 * 60 * 60  # 48 hours


def _cache_key(user_id: str) -> str:
    """Generate cache key for user business understanding."""
    return f"{CACHE_KEY_PREFIX}:{user_id}"


def _json_to_list(value: Any) -> list[str]:
    """Convert Json field to list[str], handling None."""
    if value is None:
        return []
    if isinstance(value, list):
        return cast(list[str], value)
    return []


class BusinessUnderstandingInput(pydantic.BaseModel):
    """Input model for updating business understanding - all fields optional for incremental updates."""

    # User info
    user_name: Optional[str] = pydantic.Field(None, description="The user's name")
    job_title: Optional[str] = pydantic.Field(None, description="The user's job title")

    # Business basics
    business_name: Optional[str] = pydantic.Field(
        None, description="Name of the user's business"
    )
    industry: Optional[str] = pydantic.Field(None, description="Industry or sector")
    business_size: Optional[str] = pydantic.Field(
        None, description="Company size (e.g., '1-10', '11-50')"
    )
    user_role: Optional[str] = pydantic.Field(
        None,
        description="User's role in the organization (e.g., 'decision maker', 'implementer')",
    )

    # Processes & activities
    key_workflows: Optional[list[str]] = pydantic.Field(
        None, description="Key business workflows"
    )
    daily_activities: Optional[list[str]] = pydantic.Field(
        None, description="Daily activities performed"
    )

    # Pain points & goals
    pain_points: Optional[list[str]] = pydantic.Field(
        None, description="Current pain points"
    )
    bottlenecks: Optional[list[str]] = pydantic.Field(
        None, description="Process bottlenecks"
    )
    manual_tasks: Optional[list[str]] = pydantic.Field(
        None, description="Manual/repetitive tasks"
    )
    automation_goals: Optional[list[str]] = pydantic.Field(
        None, description="Desired automation goals"
    )

    # Current tools
    current_software: Optional[list[str]] = pydantic.Field(
        None, description="Software/tools currently used"
    )
    existing_automation: Optional[list[str]] = pydantic.Field(
        None, description="Existing automations"
    )

    # Additional context
    additional_notes: Optional[str] = pydantic.Field(
        None, description="Any additional context"
    )


class BusinessUnderstanding(pydantic.BaseModel):
    """Full business understanding model returned from database."""

    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    # User info
    user_name: Optional[str] = None
    job_title: Optional[str] = None

    # Business basics
    business_name: Optional[str] = None
    industry: Optional[str] = None
    business_size: Optional[str] = None
    user_role: Optional[str] = None

    # Processes & activities
    key_workflows: list[str] = pydantic.Field(default_factory=list)
    daily_activities: list[str] = pydantic.Field(default_factory=list)

    # Pain points & goals
    pain_points: list[str] = pydantic.Field(default_factory=list)
    bottlenecks: list[str] = pydantic.Field(default_factory=list)
    manual_tasks: list[str] = pydantic.Field(default_factory=list)
    automation_goals: list[str] = pydantic.Field(default_factory=list)

    # Current tools
    current_software: list[str] = pydantic.Field(default_factory=list)
    existing_automation: list[str] = pydantic.Field(default_factory=list)

    # Additional context
    additional_notes: Optional[str] = None

    @classmethod
    def from_db(cls, db_record: CoPilotUnderstanding) -> "BusinessUnderstanding":
        """Convert database record to Pydantic model."""
        data = db_record.data if isinstance(db_record.data, dict) else {}
        business = (
            data.get("business", {}) if isinstance(data.get("business"), dict) else {}
        )
        return cls(
            id=db_record.id,
            user_id=db_record.userId,
            created_at=db_record.createdAt,
            updated_at=db_record.updatedAt,
            user_name=data.get("name"),
            job_title=business.get("job_title"),
            business_name=business.get("business_name"),
            industry=business.get("industry"),
            business_size=business.get("business_size"),
            user_role=business.get("user_role"),
            key_workflows=_json_to_list(business.get("key_workflows")),
            daily_activities=_json_to_list(business.get("daily_activities")),
            pain_points=_json_to_list(business.get("pain_points")),
            bottlenecks=_json_to_list(business.get("bottlenecks")),
            manual_tasks=_json_to_list(business.get("manual_tasks")),
            automation_goals=_json_to_list(business.get("automation_goals")),
            current_software=_json_to_list(business.get("current_software")),
            existing_automation=_json_to_list(business.get("existing_automation")),
            additional_notes=business.get("additional_notes"),
        )


def _merge_lists(existing: list | None, new: list | None) -> list | None:
    """Merge two lists, removing duplicates while preserving order."""
    if new is None:
        return existing
    if existing is None:
        return new
    # Preserve order, add new items that don't exist
    merged = list(existing)
    for item in new:
        if item not in merged:
            merged.append(item)
    return merged


async def _get_from_cache(user_id: str) -> Optional[BusinessUnderstanding]:
    """Get business understanding from Redis cache."""
    try:
        redis = await get_redis_async()
        cached_data = await redis.get(_cache_key(user_id))
        if cached_data:
            return BusinessUnderstanding.model_validate_json(cached_data)
    except Exception as e:
        logger.warning(f"Failed to get understanding from cache: {e}")
    return None


async def _set_cache(user_id: str, understanding: BusinessUnderstanding) -> None:
    """Set business understanding in Redis cache with TTL."""
    try:
        redis = await get_redis_async()
        await redis.setex(
            _cache_key(user_id),
            CACHE_TTL_SECONDS,
            understanding.model_dump_json(),
        )
    except Exception as e:
        logger.warning(f"Failed to set understanding in cache: {e}")


async def _delete_cache(user_id: str) -> None:
    """Delete business understanding from Redis cache."""
    try:
        redis = await get_redis_async()
        await redis.delete(_cache_key(user_id))
    except Exception as e:
        logger.warning(f"Failed to delete understanding from cache: {e}")


async def get_business_understanding(
    user_id: str,
) -> Optional[BusinessUnderstanding]:
    """Get the business understanding for a user.

    Checks cache first, falls back to database if not cached.
    Results are cached for 48 hours.
    """
    # Try cache first
    cached = await _get_from_cache(user_id)
    if cached:
        logger.debug(f"Business understanding cache hit for user {user_id}")
        return cached

    # Cache miss - load from database
    logger.debug(f"Business understanding cache miss for user {user_id}")
    record = await CoPilotUnderstanding.prisma().find_unique(where={"userId": user_id})
    if record is None:
        return None

    understanding = BusinessUnderstanding.from_db(record)

    # Store in cache for next time
    await _set_cache(user_id, understanding)

    return understanding


async def upsert_business_understanding(
    user_id: str,
    input_data: BusinessUnderstandingInput,
) -> BusinessUnderstanding:
    """
    Create or update business understanding with incremental merge strategy.

    - String fields: new value overwrites if provided (not None)
    - List fields: new items are appended to existing (deduplicated)

    Data is stored as: {name: ..., business: {version: 1, ...}}
    """
    # Get existing record for merge
    existing = await CoPilotUnderstanding.prisma().find_unique(
        where={"userId": user_id}
    )

    # Get existing data structure or start fresh
    existing_data: dict[str, Any] = {}
    if existing and isinstance(existing.data, dict):
        existing_data = dict(existing.data)

    existing_business: dict[str, Any] = {}
    if isinstance(existing_data.get("business"), dict):
        existing_business = dict(existing_data["business"])

    # Business fields (stored inside business object)
    business_string_fields = [
        "job_title",
        "business_name",
        "industry",
        "business_size",
        "user_role",
        "additional_notes",
    ]
    business_list_fields = [
        "key_workflows",
        "daily_activities",
        "pain_points",
        "bottlenecks",
        "manual_tasks",
        "automation_goals",
        "current_software",
        "existing_automation",
    ]

    # Handle top-level name field
    if input_data.user_name is not None:
        existing_data["name"] = input_data.user_name

    # Business string fields - overwrite if provided
    for field in business_string_fields:
        value = getattr(input_data, field)
        if value is not None:
            existing_business[field] = value

    # Business list fields - merge with existing
    for field in business_list_fields:
        value = getattr(input_data, field)
        if value is not None:
            existing_list = _json_to_list(existing_business.get(field))
            merged = _merge_lists(existing_list, value)
            existing_business[field] = merged

    # Set version and nest business data
    existing_business["version"] = 1
    existing_data["business"] = existing_business

    # Upsert with the merged data
    record = await CoPilotUnderstanding.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id, "data": SafeJson(existing_data)},
            "update": {"data": SafeJson(existing_data)},
        },
    )

    understanding = BusinessUnderstanding.from_db(record)

    # Update cache with new understanding
    await _set_cache(user_id, understanding)

    return understanding


async def clear_business_understanding(user_id: str) -> bool:
    """Clear/delete business understanding for a user from both DB and cache."""
    # Delete from cache first
    await _delete_cache(user_id)

    try:
        await CoPilotUnderstanding.prisma().delete(where={"userId": user_id})
        return True
    except Exception:
        # Record might not exist
        return False


def format_understanding_for_prompt(understanding: BusinessUnderstanding) -> str:
    """Format business understanding as text for system prompt injection."""
    if not understanding:
        return ""
    sections = []

    # User info section
    user_info = []
    if understanding.user_name:
        user_info.append(f"Name: {understanding.user_name}")
    if understanding.job_title:
        user_info.append(f"Job Title: {understanding.job_title}")
    if user_info:
        sections.append("## User\n" + "\n".join(user_info))

    # Business section
    business_info = []
    if understanding.business_name:
        business_info.append(f"Company: {understanding.business_name}")
    if understanding.industry:
        business_info.append(f"Industry: {understanding.industry}")
    if understanding.business_size:
        business_info.append(f"Size: {understanding.business_size}")
    if understanding.user_role:
        business_info.append(f"Role Context: {understanding.user_role}")
    if business_info:
        sections.append("## Business\n" + "\n".join(business_info))

    # Processes section
    processes = []
    if understanding.key_workflows:
        processes.append(f"Key Workflows: {', '.join(understanding.key_workflows)}")
    if understanding.daily_activities:
        processes.append(
            f"Daily Activities: {', '.join(understanding.daily_activities)}"
        )
    if processes:
        sections.append("## Processes\n" + "\n".join(processes))

    # Pain points section
    pain_points = []
    if understanding.pain_points:
        pain_points.append(f"Pain Points: {', '.join(understanding.pain_points)}")
    if understanding.bottlenecks:
        pain_points.append(f"Bottlenecks: {', '.join(understanding.bottlenecks)}")
    if understanding.manual_tasks:
        pain_points.append(f"Manual Tasks: {', '.join(understanding.manual_tasks)}")
    if pain_points:
        sections.append("## Pain Points\n" + "\n".join(pain_points))

    # Goals section
    if understanding.automation_goals:
        sections.append(
            "## Automation Goals\n"
            + "\n".join(f"- {goal}" for goal in understanding.automation_goals)
        )

    # Current tools section
    tools_info = []
    if understanding.current_software:
        tools_info.append(
            f"Current Software: {', '.join(understanding.current_software)}"
        )
    if understanding.existing_automation:
        tools_info.append(
            f"Existing Automation: {', '.join(understanding.existing_automation)}"
        )
    if tools_info:
        sections.append("## Current Tools\n" + "\n".join(tools_info))

    # Additional notes
    if understanding.additional_notes:
        sections.append(f"## Additional Context\n{understanding.additional_notes}")

    if not sections:
        return ""

    return "# User Business Context\n\n" + "\n\n".join(sections)
