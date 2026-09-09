import logging
from typing import Annotated

from autogpt_libs.auth import get_user_id, requires_user
from fastapi import APIRouter, HTTPException, Path, Security
from starlette.status import HTTP_404_NOT_FOUND

from backend.api.features.skills.model import (
    CopilotSkillDetail,
    CopilotSkillInfo,
    UploadCopilotSkillRequest,
)
from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot.tools.skills import (
    BuiltInSkillError,
    SkillLimitError,
    SkillNotFoundError,
    delete_user_skill,
    get_default_skill_with_body,
    list_user_skill_sibling_paths,
    list_user_skills,
    parse_skill_markdown,
    read_user_skill_with_body,
    store_user_skill,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Security(requires_user)])


@router.get(
    path="",
    summary="List user-distilled copilot skills",
    operation_id="listCopilotSkills",
)
async def list_copilot_skills(
    user_id: Annotated[str, Security(get_user_id)],
) -> list[CopilotSkillInfo]:
    """Return user-stored skills for the current user.

    Reuses :func:`backend.copilot.tools.skills.list_user_skills` so the
    library UI sees the exact same set the copilot ``<available_skills>``
    block surfaces, minus the built-in defaults (which are read-only and
    handled separately by the copilot runtime).
    """
    skills = await list_user_skills(user_id)
    return [
        CopilotSkillInfo(
            name=s.name,
            description=s.description,
            triggers=list(s.triggers),
        )
        for s in skills
    ]


@router.post(
    path="",
    summary="Upload a copilot skill from a SKILL.md file",
    operation_id="uploadCopilotSkill",
    status_code=201,
    responses={
        400: {"description": "Malformed SKILL.md or validation error"},
        409: {"description": "Per-user skill limit reached"},
    },
)
async def upload_copilot_skill(
    user_id: Annotated[str, Security(get_user_id)],
    body: UploadCopilotSkillRequest,
) -> CopilotSkillInfo:
    """Create a user-distilled skill from an uploaded ``SKILL.md`` file.

    Parses the canonical frontmatter + body, then reuses
    :func:`backend.copilot.tools.skills.store_user_skill` so an uploaded skill
    is validated, capped, and persisted exactly like one the copilot distils
    via ``store_skill``.  Malformed files return 400, the per-user cap returns
    409, and an existing slug is overwritten (upsert).
    """
    parsed = parse_skill_markdown(body.content)
    if parsed is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "File is not a valid SKILL.md — expected YAML frontmatter with "
                "'name' and 'description' followed by a markdown body."
            ),
        )
    try:
        stored = await store_user_skill(
            user_id,
            name=parsed.name,
            description=parsed.description,
            body=parsed.body,
            triggers=list(parsed.triggers),
            version=parsed.version,
        )
    except SkillLimitError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (VirusDetectedError, VirusScanError) as exc:
        logger.warning("[skills] virus scan rejected uploaded skill: %s", exc)
        raise HTTPException(
            status_code=400, detail="Skill content rejected by virus scan"
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return CopilotSkillInfo(
        name=stored.name,
        description=stored.description,
        triggers=list(stored.triggers),
    )


@router.get(
    path="/{name}",
    summary="Read a single copilot skill with its full SKILL.md body",
    operation_id="readCopilotSkill",
    responses={404: {"description": "Skill not found"}},
)
async def read_copilot_skill(
    user_id: Annotated[str, Security(get_user_id)],
    name: str = Path(..., description="Slug of the skill to read"),
) -> CopilotSkillDetail:
    """Return full SKILL.md content (name, description, triggers, body)
    for the library UI's expand-to-view dialog.

    Built-in default skills are returned with ``is_default=True`` so the
    UI can hide destructive affordances; missing user skills return 404.
    """
    slug = name.strip().lower()
    try:
        default = get_default_skill_with_body(slug)
    except OSError:
        # Don't leak the on-disk path; operators trace via server logs.
        logger.exception("[skills] failed to load default skill body for %s", slug)
        raise HTTPException(
            status_code=500,
            detail="Failed to load default skill body",
        )
    if default is not None:
        return CopilotSkillDetail(
            name=default.name,
            description=default.description,
            triggers=list(default.triggers),
            body=default.body,
            is_default=True,
        )

    parsed = await read_user_skill_with_body(user_id, slug)
    if parsed is None:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"Skill '{slug}' not found"
        )
    sibling_files = await list_user_skill_sibling_paths(user_id, slug)
    return CopilotSkillDetail(
        name=parsed.name,
        description=parsed.description,
        triggers=list(parsed.triggers),
        body=parsed.body,
        version=parsed.version,
        is_default=False,
        sibling_files=sibling_files,
    )


@router.delete(
    path="/{name}",
    summary="Delete a user-distilled copilot skill",
    operation_id="deleteCopilotSkill",
)
async def delete_copilot_skill(
    user_id: Annotated[str, Security(get_user_id)],
    name: str = Path(..., description="Slug of the skill to delete"),
) -> dict[str, str]:
    """Delete a user-distilled skill by slug.

    Built-in defaults are not user-deletable — attempting to delete one
    returns 400.  Missing skills return 404 so the UI can reconcile a
    stale list.
    """
    try:
        slug = await delete_user_skill(user_id, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except BuiltInSkillError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except SkillNotFoundError as exc:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=str(exc))
    return {"name": slug}
