"""Public browse and install for marketplace skill listings."""

import autogpt_libs.auth
import fastapi
from fastapi import Path, Query, Security

from backend.copilot.tools.skills import SkillLimitError

from . import skill_db, skill_model, skill_submission_db

router = fastapi.APIRouter()


@router.get(
    "",
    summary="List marketplace skills",
    tags=["store", "public"],
)
async def list_marketplace_skills(
    category: str | None = Query(default=None),
    search_query: str | None = Query(default=None, alias="search_query"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> skill_model.MarketplaceSkillsResponse:
    """Approved skill listings, verified first."""
    return await skill_db.get_marketplace_skills(
        category=category,
        search_query=search_query,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/submissions",
    summary="Publish skill to marketplace",
    tags=["store", "private"],
    status_code=201,
    responses={
        404: {"description": "Skill not in the caller's library"},
        428: {"description": "Marketplace profile required, or slug taken"},
    },
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def submit_skill(
    request: skill_model.SkillSubmissionRequest,
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> skill_model.SkillSubmission:
    """Submit one of the caller's own library skills for marketplace review."""
    return await skill_submission_db.submit_skill(user_id, request)


@router.get(
    "/submissions",
    summary="List my skill submissions",
    tags=["store", "private"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def list_my_skill_submissions(
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> list[skill_model.SkillSubmission]:
    """Every version the caller has submitted, newest first per listing."""
    return await skill_submission_db.list_my_skill_submissions(user_id)


@router.put(
    "/submissions/{skill_listing_version_id}",
    summary="Edit skill submission",
    tags=["store", "private"],
    responses={
        404: {"description": "Submission not found"},
        428: {"description": "Only a pending submission can be edited"},
    },
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def edit_skill_submission(
    request: skill_model.SkillSubmissionRequest,
    skill_listing_version_id: str = Path(...),
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> skill_model.SkillSubmission:
    """Update a pending submission and re-snapshot the library skill."""
    return await skill_submission_db.edit_skill_submission(
        user_id, skill_listing_version_id, request
    )


@router.get(
    "/{slug}",
    summary="Get marketplace skill",
    tags=["store", "public"],
    responses={404: {"description": "Skill not found"}},
)
async def get_marketplace_skill(
    slug: str = Path(..., description="Slug of the skill listing"),
) -> skill_model.MarketplaceSkillDetails:
    """One listing with the full SKILL.md body the install will copy."""
    return await skill_db.get_marketplace_skill(slug)


@router.post(
    "/{slug}/install",
    summary="Install marketplace skill",
    tags=["store", "private"],
    responses={
        404: {"description": "Skill not found"},
        409: {"description": "Per-user skill limit reached"},
    },
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def install_marketplace_skill(
    slug: str = Path(..., description="Slug of the skill listing"),
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> skill_model.InstalledSkill:
    """Copy the listing's skill into the caller's AutoPilot skill library."""
    try:
        return await skill_db.install_marketplace_skill(user_id, slug)
    except SkillLimitError as exc:
        raise fastapi.HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=400, detail=str(exc))
