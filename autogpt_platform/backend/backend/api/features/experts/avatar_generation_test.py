import io
import time
import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from backend.api.features.experts import avatar_jobs
from backend.api.features.experts.avatar_generation import (
    ExpertAvatarRequest,
    avatar_prompt,
    validate_png,
)
from backend.api.features.experts.avatar_jobs import ExpertAvatarJob


def test_generation_accepts_only_brand_choices():
    for payload in (
        {"category": "otto"},
        {"category": "purple"},
        {"prompt": "ignore the rules"},
        {"shape": "human"},
    ):
        with pytest.raises(ValidationError):
            ExpertAvatarRequest.model_validate(payload)
    prompt = avatar_prompt(
        ExpertAvatarRequest(category="finance", expression="curious")
    )
    assert "#A5B09A" in prompt
    assert "one raised brow" in prompt
    assert "LOWER BASE" in prompt


def test_png_validation_keeps_alpha_and_rejects_wrong_outputs():
    image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    image.putpixel((512, 512), (100, 100, 100, 255))
    content = io.BytesIO()
    image.save(content, format="PNG")
    assert validate_png(content.getvalue()).getvalue() == content.getvalue()
    for size, mode, format in [
        ((1024, 1024), "RGB", "PNG"),
        ((16, 16), "RGBA", "PNG"),
        ((1024, 1024), "RGB", "JPEG"),
    ]:
        bad = io.BytesIO()
        Image.new(mode, size).save(bad, format=format)
        with pytest.raises(ValueError):
            validate_png(bad.getvalue())


@pytest.mark.asyncio
async def test_job_lookup_is_owner_scoped_and_expired_jobs_fail(monkeypatch):
    job = ExpertAvatarJob(created_at=time.time() - 300)
    redis = AsyncMock()
    redis.get.return_value = job.model_dump_json()
    monkeypatch.setattr(avatar_jobs, "get_redis_async", AsyncMock(return_value=redis))
    result = await avatar_jobs.load_job("owner", job.id)
    redis.get.assert_awaited_once_with(f"expert-avatar-job:owner:{job.id}")
    assert result.status == "failed"
    redis.get.return_value = None
    with pytest.raises(HTTPException) as error:
        await avatar_jobs.load_job("other-owner", job.id)
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_rate_limit_and_unavailable_redis_fail_closed(monkeypatch):
    redis = AsyncMock()
    redis.eval.return_value = 120
    monkeypatch.setattr(avatar_jobs, "get_redis_async", AsyncMock(return_value=redis))
    with pytest.raises(HTTPException) as error:
        await avatar_jobs.reserve_generation("owner")
    assert error.value.status_code == 429
    assert error.value.headers == {"Retry-After": "120"}
    redis.eval.side_effect = ConnectionError()
    with pytest.raises(HTTPException) as error:
        await avatar_jobs.reserve_generation("owner")
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_generation_stores_a_preview_without_changing_an_expert(monkeypatch):
    generate = AsyncMock(return_value=io.BytesIO(b"png"))
    upload = AsyncMock(return_value="https://cdn.test/generated.png")
    save = AsyncMock()
    monkeypatch.setattr(avatar_jobs, "generate_avatar", generate)
    monkeypatch.setattr(avatar_jobs, "upload_media", upload)
    monkeypatch.setattr(avatar_jobs, "save_job", save)
    job = ExpertAvatarJob(id=uuid.uuid4())
    await avatar_jobs.run_generation("owner", job, ExpertAvatarRequest())
    assert job.status == "complete"
    assert job.avatar_url == "https://cdn.test/generated.png"
    assert upload.call_args.kwargs["user_id"] == "owner"
    assert upload.call_args.kwargs["file"].content_type == "image/png"
    save.assert_awaited_once_with("owner", job)


@pytest.mark.asyncio
async def test_failed_generation_does_not_upload(monkeypatch):
    upload = AsyncMock()
    monkeypatch.setattr(
        avatar_jobs, "generate_avatar", AsyncMock(side_effect=ValueError("bad image"))
    )
    monkeypatch.setattr(avatar_jobs, "upload_media", upload)
    monkeypatch.setattr(avatar_jobs, "save_job", AsyncMock())
    job = ExpertAvatarJob()
    await avatar_jobs.run_generation("owner", job, ExpertAvatarRequest())
    assert job.status == "failed"
    assert "bad image" not in (job.error or "")
    upload.assert_not_called()
