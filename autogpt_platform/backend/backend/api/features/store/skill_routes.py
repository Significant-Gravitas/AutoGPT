"""Public browse and install for marketplace skill listings."""

import autogpt_libs.auth
import fastapi
from fastapi import Depends, Path, Query, Security

from backend.api.features.experts import experts_db
from backend.copilot.tools.skills import SkillLimitError, SkillOwnedError
from backend.util.feature_flag import Flag, is_feature_enabled

from . import skill_db, skill_model, skill_submission_db

# Browse is anonymous, and LaunchDarkly needs a context key: a non-UUID key is
# evaluated as an anonymous context, which with the flag off answers False.
_ANONYMOUS_FLAG_KEY = "anonymous"

# This endpoint is a source viewer for someone deciding to install or approve a
# package, not a file server: a file past this is refused rather than streamed.
MAX_VIEWABLE_FILE_BYTES = 256 * 1024


async def require_skills_hub_flag(
    user_id: str | None = Security(autogpt_libs.auth.get_optional_user_id),
) -> None:
    """Gate every skills route on the skills-hub flag, fail-closed.

    Deliberately not ``create_feature_flag_dependency``: that helper 404s
    whenever LaunchDarkly has no SDK key, before consulting the
    ``FORCE_FLAG_SKILLS_HUB`` override every local environment relies on.
    """
    if not await is_feature_enabled(Flag.SKILLS_HUB, user_id or _ANONYMOUS_FLAG_KEY):
        raise fastapi.HTTPException(status_code=404, detail="Feature not available")


router = fastapi.APIRouter(dependencies=[Depends(require_skills_hub_flag)])


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
    """Approved skill listings, most recently updated first."""
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


@router.get(
    "/{slug}/files/{path:path}",
    summary="Read marketplace skill file",
    tags=["store", "public"],
    response_class=fastapi.responses.PlainTextResponse,
    responses={
        404: {"description": "Skill, version or file not found"},
        413: {"description": "File is too large to view"},
        415: {"description": "File is not text"},
    },
)
async def read_marketplace_skill_file(
    slug: str = Path(..., description="Slug of the skill listing"),
    path: str = Path(..., description="File's path relative to the skill folder"),
    version_id: str | None = Query(
        default=None,
        description=(
            "Read this version instead of the live one. A pending submission "
            "is readable only by the creator who submitted it."
        ),
    ),
    user_id: str | None = Security(autogpt_libs.auth.get_optional_user_id),
) -> str:
    """One file's text, for reading a package before installing it."""
    version = await skill_db.find_readable_version(
        slug, version_id=version_id, user_id=user_id
    )
    return await read_package_file(version.id, path)


async def read_package_file(skill_listing_version_id: str, relative_path: str) -> str:
    """The published file's text, or the refusal that applies to it.

    Shared with the admin reviewer's own route, which reaches a pending
    version through its router's admin check rather than through this one.
    """
    meta = await skill_db.get_package_file_meta(skill_listing_version_id, relative_path)
    # The path is matched against the published rows, so nothing here can
    # address storage — an unknown path is simply not part of the package.
    if meta is None:
        raise fastapi.HTTPException(
            status_code=404, detail=f"'{relative_path}' is not part of this package"
        )
    if meta.size_bytes > MAX_VIEWABLE_FILE_BYTES:
        raise fastapi.HTTPException(
            status_code=413,
            detail=(
                f"'{relative_path}' is {meta.size_bytes} bytes; this endpoint "
                f"serves up to {MAX_VIEWABLE_FILE_BYTES}"
            ),
        )
    if not _is_text_type(meta.mime_type):
        raise fastapi.HTTPException(
            status_code=415,
            detail=f"'{relative_path}' is {meta.mime_type}, which is not text",
        )
    content = await skill_db.read_package_file_bytes(
        skill_listing_version_id, relative_path
    )
    if content is None:
        raise fastapi.HTTPException(
            status_code=404, detail=f"'{relative_path}' is not part of this package"
        )
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        raise fastapi.HTTPException(
            status_code=415, detail=f"'{relative_path}' is not valid UTF-8 text"
        )


def _is_text_type(mime_type: str | None) -> bool:
    """Whether the viewer will serve a file of this type.

    ``None`` is an extension ``mimetypes`` cannot name — a ``Makefile``, a
    ``.toml`` — which a reviewer still has to read, so the decode below is
    what refuses those rather than the name.
    """
    if mime_type is None:
        return True
    return mime_type.startswith("text/") or mime_type in _TEXT_APPLICATION_TYPES


# `application/…` types a package ships as source a reviewer reads; every other
# non-`text/` type is a binary this will not serve. Listed rather than derived
# from `mimetypes`, which reads the publishing host's /etc/mime.types, so a row
# can carry a spelling this machine never produces.
_TEXT_APPLICATION_TYPES = frozenset(
    {
        "application/javascript",
        "application/json",
        "application/sql",
        "application/toml",
        "application/x-httpd-php",
        "application/x-sh",
        "application/x-yaml",
        "application/xml",
        "application/yaml",
    }
)


@router.post(
    "/{slug}/install",
    summary="Install marketplace skill",
    tags=["store", "private"],
    responses={
        404: {"description": "Skill or expert not found"},
        409: {
            "description": (
                "Skill limit reached, or the skill name is already used by an "
                "owner-saved skill"
            )
        },
    },
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def install_marketplace_skill(
    slug: str = Path(..., description="Slug of the skill listing"),
    expert_id: str | None = Query(
        default=None,
        description="Install into this expert's skills instead of the caller's own.",
    ),
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> skill_model.InstalledSkill:
    """Copy the listing's skill into an expert's skills, or the caller's own."""
    await _require_install_target(user_id, expert_id)
    try:
        return await skill_db.install_marketplace_skill(
            user_id, slug, expert_id=expert_id
        )
    except (SkillLimitError, SkillOwnedError) as exc:
        raise fastapi.HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=400, detail=str(exc))


async def _require_install_target(user_id: str, expert_id: str | None) -> None:
    """An expert named by the client must be one of the caller's own active
    hires; someone else's answers 404, which a 403 would confirm exists."""
    if expert_id is None:
        return
    if not await experts_db.owns_private_active_expert(user_id, expert_id):
        raise fastapi.HTTPException(
            status_code=404, detail=f"Expert '{expert_id}' not found"
        )
