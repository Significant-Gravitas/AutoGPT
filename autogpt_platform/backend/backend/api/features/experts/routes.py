import autogpt_libs.auth as autogpt_auth_lib
import fastapi
from fastapi import APIRouter, Security
from pydantic import BaseModel, Field, field_validator

from backend.api.features.experts import experts_db, scheduling
from backend.api.features.experts.models import (
    EXPERT_AVATAR_URL_MAX_LENGTH,
    EXPERT_COLOR_MAX_LENGTH,
    EXPERT_IDENTITY_MAX_LENGTH,
    MAX_RAISE_ATTACHMENTS,
    WEEKLY_BUDGET_MAX_CREDITS,
    Expert,
    ExpertDetachPreview,
    ExpertIdentity,
    ExpertPod,
    ExpertRun,
    ExpertSoulUpdate,
    ExpertWorkflowRef,
    HireResult,
    RaiseAttachment,
    RaiseResult,
    validate_avatar_url,
)

router = APIRouter(
    prefix="/experts",
    tags=["experts", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


class HireRequest(BaseModel):
    template_id: str
    name: str | None = Field(default=None, max_length=100)


class InstallWorkflowRequest(BaseModel):
    store_listing_version_id: str


class CreatePodRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Pod name must not be blank")
        return stripped


class AssignPodRequest(BaseModel):
    # Required but nullable: an explicit {"pod_id": null} detaches the
    # expert, while an omitted field is rejected rather than treated as
    # a silent detach.
    pod_id: str | None


class CreateRaisedExpertRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    role: str | None = Field(default=None, max_length=100)
    avatar_url: str | None = Field(
        default=None, max_length=EXPERT_AVATAR_URL_MAX_LENGTH
    )
    # Opaque design token (e.g. "sky-300"); the client maps it to a palette.
    color: str | None = Field(default=None, max_length=EXPERT_COLOR_MAX_LENGTH)
    voice_preferences: str | None = Field(default=None, max_length=4_000)
    # Free-text "about them" answer from the raise flow; becomes the identity.
    about: str | None = Field(default=None, max_length=EXPERT_IDENTITY_MAX_LENGTH)
    # Credits (100 = $1). Omitted/null keeps the platform default at read time.
    weekly_budget: int | None = Field(default=None, ge=0, le=WEEKLY_BUDGET_MAX_CREDITS)
    attachments: list[RaiseAttachment] = Field(
        default_factory=list, max_length=MAX_RAISE_ATTACHMENTS
    )

    # "before" so the length bounds apply to the trimmed name and a blank one
    # fails with the message below rather than the generic min_length error.
    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("Name must not be blank")
        return stripped

    @field_validator("avatar_url")
    @classmethod
    def check_avatar_url(cls, value: str | None) -> str | None:
        return validate_avatar_url(value)

    @field_validator("color", "about")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


@router.get("/templates", operation_id="list_expert_templates")
async def list_expert_templates() -> list[Expert]:
    return await experts_db.list_templates()


@router.post(
    "",
    operation_id="hire_expert",
    responses={
        404: {"description": "Expert or expert template not found"},
        409: {"description": "Active expert limit reached"},
        503: {"description": "Expert workspace unavailable"},
    },
)
async def hire_expert(
    request: HireRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> HireResult:
    try:
        return await experts_db.hire_expert(user_id, request.template_id, request.name)
    except experts_db.ExpertTemplateNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))
    except experts_db.ExpertNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail="Expert not found") from e
    except experts_db.ExpertHireUnavailableError as e:
        raise fastapi.HTTPException(
            status_code=503,
            detail="Your expert is temporarily unavailable. Try again shortly.",
        ) from e
    except experts_db.ExpertPrivateTenancyNotFoundError as e:
        raise fastapi.HTTPException(
            status_code=503,
            detail="Your expert workspace is still being set up. Try again shortly.",
        ) from e
    except experts_db.ExpertLimitExceededError as e:
        raise fastapi.HTTPException(
            status_code=409,
            detail={"code": "active_expert_limit", "limit": e.limit},
        )


@router.post(
    "/raise",
    operation_id="create_raised_expert",
    responses={
        404: {"description": "Attachment not found or unavailable"},
        409: {"description": "Active expert limit reached"},
    },
)
async def create_raised_expert(
    request: CreateRaisedExpertRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> RaiseResult:
    try:
        return await experts_db.create_raised_expert(
            user_id,
            request.name,
            request.role,
            request.voice_preferences,
            avatar_url=request.avatar_url,
            color=request.color,
            about=request.about,
            weekly_budget=request.weekly_budget,
            attachments=request.attachments,
        )
    except experts_db.FirstJobUnavailableError as e:
        raise fastapi.HTTPException(
            status_code=404,
            detail={
                "code": "attachment_unavailable",
                "kind": e.kind,
                "source": e.source,
                "id": e.id,
            },
        )
    except experts_db.ExpertLimitExceededError as e:
        raise fastapi.HTTPException(
            status_code=409,
            detail={"code": "active_expert_limit", "limit": e.limit},
        )
    except experts_db.RaisedExpertLifetimeLimitExceededError as e:
        raise fastapi.HTTPException(
            status_code=409,
            detail={"code": "raised_expert_lifetime_limit", "limit": e.limit},
        )


@router.get("", operation_id="list_experts")
async def list_experts(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> list[Expert]:
    """List the user's active hired experts."""
    return await experts_db.list_experts(user_id)


# Pod routes are declared before "/{expert_id}" so "/experts/pods" is not
# swallowed by the expert-detail path parameter.
@router.post(
    "/pods",
    operation_id="create_expert_pod",
    responses={409: {"description": "Duplicate pod name, or the pod limit is reached"}},
)
async def create_expert_pod(
    request: CreatePodRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ExpertPod:
    try:
        return await experts_db.create_pod(user_id, request.name)
    except (
        experts_db.ExpertPodNameTakenError,
        experts_db.ExpertPodLimitReachedError,
    ) as e:
        # 409 rather than 422: the request is well-formed, it conflicts with
        # the caller's existing pods. Keeping 422 for schema validation alone
        # preserves the generated HTTPValidationError shape on this route.
        raise fastapi.HTTPException(status_code=409, detail=str(e))


@router.get("/pods", operation_id="list_expert_pods")
async def list_expert_pods(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> list[ExpertPod]:
    return await experts_db.list_pods(user_id)


@router.patch(
    "/{expert_id}/pod",
    operation_id="assign_expert_pod",
    responses={404: {"description": "Expert or pod not found"}},
)
async def assign_expert_pod(
    expert_id: str,
    request: AssignPodRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Expert:
    try:
        return await experts_db.assign_pod(user_id, expert_id, request.pod_id)
    except (
        experts_db.ExpertNotFoundError,
        experts_db.ExpertPodNotFoundError,
    ):
        raise fastapi.HTTPException(status_code=404, detail="Expert or pod not found")


@router.get("/identities", operation_id="list_expert_identities")
async def list_expert_identities(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> list[ExpertIdentity]:
    """List the lightweight active and archived identity projection for chat."""
    return await experts_db.list_expert_identities(user_id)


@router.get(
    "/{expert_id}",
    operation_id="get_expert",
    responses={404: {"description": "Expert not found"}},
)
async def get_expert(
    expert_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Expert:
    expert = await experts_db.get_expert(user_id, expert_id)
    if expert is None:
        raise fastapi.HTTPException(status_code=404, detail="Expert not found")
    return expert


@router.get(
    "/{expert_id}/runs",
    operation_id="list_expert_runs",
    responses={404: {"description": "Expert not found"}},
)
async def list_expert_runs(
    expert_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> list[ExpertRun]:
    """Recent executions attributed to this expert, with a classified output
    type for the Work surface's typed viewer."""
    try:
        return await experts_db.list_expert_runs(user_id, expert_id)
    except experts_db.ExpertNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))


@router.patch(
    "/{expert_id}/soul",
    operation_id="update_expert_soul",
    responses={404: {"description": "Expert not found"}},
)
async def update_expert_soul(
    expert_id: str,
    request: ExpertSoulUpdate,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Expert:
    try:
        return await experts_db.update_soul(user_id, expert_id, request)
    except experts_db.ExpertNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))


@router.post(
    "/{expert_id}/workflows",
    operation_id="install_expert_workflow",
    responses={404: {"description": "Expert not found"}},
)
async def install_expert_workflow(
    expert_id: str,
    request: InstallWorkflowRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ExpertWorkflowRef:
    try:
        return await experts_db.install_workflow(
            user_id, expert_id, request.store_listing_version_id
        )
    except experts_db.ExpertNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{expert_id}/detach-preview",
    operation_id="get_expert_detach_preview",
    responses={404: {"description": "Expert not found"}},
)
async def get_expert_detach_preview(
    expert_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ExpertDetachPreview:
    """What archiving this expert would pause — schedules and triggers —
    so the client can show a clear confirmation prompt."""
    expert = await experts_db.get_expert(user_id, expert_id, include_workflows=False)
    if expert is None:
        raise fastapi.HTTPException(status_code=404, detail="Expert not found")
    return await scheduling.get_detach_preview(user_id, expert_id)


@router.post(
    "/{expert_id}/schedules/resume",
    operation_id="resume_expert_schedules",
    responses={404: {"description": "Expert not found"}},
)
async def resume_expert_schedules(
    expert_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Expert:
    """One-click reversal of a budget/archive pause."""
    resumed = await scheduling.resume_expert_schedules(user_id, expert_id)
    if not resumed:
        raise fastapi.HTTPException(status_code=404, detail="Expert not found")
    expert = await experts_db.get_expert(user_id, expert_id)
    if expert is None:
        raise fastapi.HTTPException(status_code=404, detail="Expert not found")
    return expert


@router.delete(
    "/{expert_id}",
    operation_id="archive_expert",
    status_code=204,
    responses={404: {"description": "Expert not found"}},
)
async def archive_expert(
    expert_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> fastapi.Response:
    try:
        await experts_db.archive_expert(user_id, expert_id)
    except experts_db.ExpertNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))
    return fastapi.Response(status_code=204)
