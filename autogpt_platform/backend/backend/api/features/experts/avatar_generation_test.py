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
        {"color": "lavender"},
        {"base": "legs"},
        {"tilt": "upside-down"},
        {"inlay": "head"},
    ):
        with pytest.raises(ValidationError):
            ExpertAvatarRequest.model_validate(payload)
    prompt = avatar_prompt(
        ExpertAvatarRequest(category="finance", expression="curious")
    )
    assert "#A5B09A" in prompt
    assert "one raised brow" in prompt
    assert "BODY ONLY" in prompt


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


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["pebble", "bean", "shield"])
async def test_provider_edit_round_trip_uses_reference_and_returns_validated_png(
    monkeypatch,
    shape,
):
    import base64
    from types import SimpleNamespace

    import httpx
    from openai import AsyncOpenAI

    from backend.api.features.experts import avatar_generation

    png = io.BytesIO()
    image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    image.putpixel((512, 512), (100, 100, 100, 255))
    image.save(png, format="PNG")
    content = png.getvalue()
    requests = []

    def provider(request):
        requests.append(request)
        assert request.url.path == "/v1/images/edits"
        assert b'name="background"\r\n\r\ntransparent' in request.content
        assert b'name="output_format"\r\n\r\npng' in request.content
        assert b'name="size"\r\n\r\n1024x1024' in request.content
        assert b"reference.png" in request.content
        assert (
            avatar_generation.REFERENCE_FOLDER / f"{shape}.png"
        ).read_bytes() in request.content
        return httpx.Response(
            200,
            json={
                "created": 1,
                "data": [{"b64_json": base64.b64encode(content).decode()}],
            },
        )

    def client(**kwargs):
        return AsyncOpenAI(
            **kwargs,
            base_url="https://provider.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
        )

    monkeypatch.setattr(avatar_generation, "AsyncOpenAI", client)
    monkeypatch.setattr(
        avatar_generation,
        "Settings",
        lambda: SimpleNamespace(
            secrets=SimpleNamespace(openai_api_key="test-key"),
            config=SimpleNamespace(expert_avatar_model="gpt-image-1.5"),
        ),
    )
    result = await avatar_generation.generate_avatar(
        ExpertAvatarRequest.model_validate({"category": "finance", "shape": shape})
    )
    assert result.getvalue() == content
    assert len(requests) == 1


def test_generation_varies_color_and_full_outline_independently_of_category():
    request = ExpertAvatarRequest.model_validate(
        {
            "category": "marketing",
            "color": "pine",
            "shape": "bean",
            "base": "wide",
            "tilt": "left",
            "inlay": "curl",
            "expression": "curious",
        }
    )
    prompt = avatar_prompt(request)
    assert "#4F7968" in prompt
    assert "#C45F36" not in prompt
    assert "kidney" in prompt
    assert "wide" in prompt
    assert "left" in prompt
    assert "curl" in prompt


def test_every_shape_has_a_distinct_transparent_reference():
    from typing import get_args

    from backend.api.features.experts.avatar_design import SHAPES, AvatarShape
    from backend.api.features.experts.avatar_generation import REFERENCE_FOLDER

    assert set(get_args(AvatarShape)) == set(SHAPES)
    for shape in get_args(AvatarShape):
        with Image.open(REFERENCE_FOLDER / f"{shape}.png") as image:
            assert image.size == (256, 256)
            assert image.mode == "RGBA"
            assert image.getchannel("A").getextrema() == (0, 255)


def test_accents_can_move_to_head_and_repeat():
    prompt = avatar_prompt(
        ExpertAvatarRequest.model_validate(
            {
                "accent_placement": "head",
                "accent_count": "three",
                "inlay": "patch",
            }
        )
    )
    assert "HEAD ONLY" in prompt
    assert "three separate" in prompt
    assert "rounded irregular patch" in prompt
    assert "Head stays wholly main color" not in prompt
    assert "LOWER BASE" not in prompt


def test_shade_uses_category_hue_instead_of_an_unrelated_color():
    prompt = avatar_prompt(
        ExpertAvatarRequest.model_validate(
            {
                "category": "finance",
                "shade": "dark",
                "color": "terracotta",
            }
        )
    )
    assert "#8B9481" in prompt
    assert "#C45F36" not in prompt
