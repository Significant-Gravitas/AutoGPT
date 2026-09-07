"""Public browse and install for marketplace skill listings."""

import autogpt_libs.auth
import fastapi
from fastapi import Path, Query, Security

from backend.copilot.tools.skills import SkillLimitError

from . import skill_db, skill_model

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
