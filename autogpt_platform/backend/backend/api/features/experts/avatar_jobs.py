import asyncio
import logging
import time
import uuid
from typing import Literal

from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, Field
from redis_lua_py import Key, redis, script
from starlette.datastructures import Headers

from backend.api.features.experts.avatar_generation import (
    ExpertAvatarRequest,
    generate_avatar,
)
from backend.api.features.store.media import upload_media
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)
JOB_TTL = 3600
GENERATION_TIMEOUT = 240


@script
def _reserve(key: Key, now: int) -> int:
    count = int(redis.hget(key, "count") or "0")
    if count >= 5:
        return max(1, redis.ttl(key))
    next_at = int(redis.hget(key, "next") or "0")
    if next_at > now:
        return next_at - now
    redis.hset(key, "count", count + 1, "next", now + 240)
    if count == 0:
        redis.expire(key, 86400)
    return 0


class ExpertAvatarJob(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    status: Literal["pending", "complete", "failed"] = "pending"
    avatar_url: str | None = None
    error: str | None = None
    created_at: float = Field(default_factory=time.time)


async def reserve_generation(user_id: str) -> None:
    try:
        async with asyncio.timeout(3):
            redis = await get_redis_async()
            retry_after = await _reserve(
                redis, key=f"expert-avatar-limit:{user_id}", now=int(time.time())
            )
    except Exception as exc:
        raise HTTPException(
            503, "Avatar generation is temporarily unavailable"
        ) from exc
    if retry_after:
        raise HTTPException(
            429,
            "Please wait before generating again. Limit: five avatars per day.",
            headers={"Retry-After": str(retry_after)},
        )


async def save_job(user_id: str, job: ExpertAvatarJob) -> None:
    async with asyncio.timeout(3):
        redis = await get_redis_async()
        await redis.setex(
            f"expert-avatar-job:{user_id}:{job.id}", JOB_TTL, job.model_dump_json()
        )


async def load_job(user_id: str, job_id: uuid.UUID) -> ExpertAvatarJob:
    try:
        async with asyncio.timeout(3):
            redis = await get_redis_async()
            value = await redis.get(f"expert-avatar-job:{user_id}:{job_id}")
    except Exception as exc:
        raise HTTPException(503, "Could not check avatar generation") from exc
    if not value:
        raise HTTPException(404, "Avatar generation not found")
    job = ExpertAvatarJob.model_validate_json(value)
    if job.status == "pending" and time.time() - job.created_at > GENERATION_TIMEOUT:
        job.status = "failed"
        job.error = "Generation timed out. Choose a catalog avatar or try again."
    return job


async def run_generation(
    user_id: str, job: ExpertAvatarJob, request: ExpertAvatarRequest
) -> None:
    try:
        async with asyncio.timeout(GENERATION_TIMEOUT - 5):
            image = await generate_avatar(request)
            file = UploadFile(
                file=image,
                filename=f"expert-avatar-{job.id}.png",
                headers=Headers({"content-type": "image/png"}),
            )
            try:
                job.avatar_url = await upload_media(
                    user_id=user_id, file=file, is_avatar=True
                )
            finally:
                await file.close()
            job.status = "complete"
    except Exception:
        logger.exception("Expert avatar generation failed")
        job.status = "failed"
        job.error = (
            "Could not generate this avatar. Choose a catalog avatar or try again."
        )
    try:
        await save_job(user_id, job)
    except Exception:
        logger.exception("Could not save expert avatar generation result")
