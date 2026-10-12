import uuid

from autogpt_libs.auth import get_user_id, requires_user
from fastapi import APIRouter, BackgroundTasks, HTTPException, Security

from backend.api.features.experts import avatar_jobs
from backend.api.features.experts.avatar_generation import ExpertAvatarRequest
from backend.api.features.experts.avatar_jobs import ExpertAvatarJob
from backend.util.settings import Settings

router = APIRouter(
    prefix="/avatars", tags=["experts"], dependencies=[Security(requires_user)]
)


@router.post("/generations", operation_id="generate_expert_avatar", status_code=202)
async def generate_expert_avatar(
    request: ExpertAvatarRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Security(get_user_id),
) -> ExpertAvatarJob:
    if not Settings().secrets.openai_api_key:
        raise HTTPException(
            503,
            "Avatar generation is not configured. Choose a catalog avatar or upload a picture.",
        )
    await avatar_jobs.reserve_generation(user_id)
    job = ExpertAvatarJob()
    try:
        await avatar_jobs.save_job(user_id, job)
    except Exception as exc:
        raise HTTPException(
            503, "Avatar generation is temporarily unavailable"
        ) from exc
    background_tasks.add_task(avatar_jobs.run_generation, user_id, job, request)
    return job


@router.get("/generations/{job_id}", operation_id="get_expert_avatar_generation")
async def get_expert_avatar_generation(
    job_id: uuid.UUID, user_id: str = Security(get_user_id)
) -> ExpertAvatarJob:
    return await avatar_jobs.load_job(user_id, job_id)
