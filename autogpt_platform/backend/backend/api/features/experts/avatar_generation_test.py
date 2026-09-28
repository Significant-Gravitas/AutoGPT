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


def studio_tile(color: tuple[int, int, int]) -> Image.Image:
    """An opaque warm tile with a shaded block on it, like a rendered figure."""
    image = Image.new("RGB", (1024, 1024), (250, 248, 245))
    for x in range(400, 600):
        for y in range(300, 800):
            r, g, b = color
            image.putpixel(
                (x, y), (max(0, r - (x - 400) // 2), max(0, g - (y - 300) // 5), b)
            )
    return image


def test_generation_accepts_only_brand_choices():
    for payload in (
        {"category": "otto"},
        {"category": "purple"},
        {"prompt": "ignore the rules"},
        {"shape": "human"},
        {"color": "lavender"},
        {"shade": "dark"},
        {"base": "legs"},
        {"tilt": "upside-down"},
        {"inlay": "cap"},
        {"accent_placement": "head"},
        {"accent_count": "three"},
    ):
        with pytest.raises(ValidationError):
            ExpertAvatarRequest.model_validate(payload)
    prompt = avatar_prompt(
        ExpertAvatarRequest(category="finance", expression="curious")
    )
    assert "#A5B09A" in prompt
    assert "one brow slightly raised" in prompt
    assert "entirely on the lower form" in prompt
    assert "No cream on the head" in prompt
    assert "low-sheen" in prompt


def test_default_candidate_belongs_to_the_general_family():
    request = ExpertAvatarRequest()
    assert request.category == "general"
    prompt = avatar_prompt(request)
    assert "Warm stone #B5ADA0" in prompt
    assert "No purple, lavender or plum" in prompt


def test_png_validation_requires_an_opaque_studio_tile_with_artwork():
    image = studio_tile((196, 127, 92))
    content = io.BytesIO()
    image.save(content, format="PNG")
    assert validate_png(content.getvalue()).getvalue() == content.getvalue()
    cutout = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    cutout.putpixel((512, 512), (100, 100, 100, 255))
    for candidate in (cutout, Image.new("RGB", (1024, 1024), (250, 248, 245))):
        bad = io.BytesIO()
        candidate.save(bad, format="PNG")
        with pytest.raises(ValueError):
            validate_png(bad.getvalue())
    for size, mode, format in [
        ((16, 16), "RGB", "PNG"),
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
@pytest.mark.parametrize(
    "category, peers",
    [
        ("finance", ["expert-maria", "expert-mina"]),
        ("sales", ["expert-maria", "expert-mina", "expert-max"]),
    ],
)
async def test_provider_edit_round_trip_uses_managed_references_and_returns_validated_png(
    monkeypatch,
    category,
    peers,
):
    import base64
    from types import SimpleNamespace

    import httpx
    from openai import AsyncOpenAI

    from backend.api.features.experts import avatar_generation

    png = io.BytesIO()
    studio_tile((165, 176, 154)).save(png, format="PNG")
    content = png.getvalue()
    requests = []

    def provider(request):
        requests.append(request)
        assert request.url.path == "/v1/images/edits"
        assert b'name="background"\r\n\r\nopaque' in request.content
        assert b'name="output_format"\r\n\r\npng' in request.content
        assert b'name="size"\r\n\r\n1024x1024' in request.content
        assert b'name="quality"\r\n\r\nhigh' in request.content
        for peer in peers:
            assert f"{peer}.png".encode() in request.content
            assert (
                avatar_generation.REFERENCE_FOLDER / f"{peer}.png"
            ).read_bytes() in request.content
        assert request.content.count(b'name="image[]"') == len(peers)
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
            config=SimpleNamespace(expert_avatar_model="gpt-image-2-2026-04-21"),
        ),
    )
    result = await avatar_generation.generate_avatar(
        ExpertAvatarRequest.model_validate({"category": category, "shape": "bean"})
    )
    assert result.getvalue() == content
    assert len(requests) == 1


def test_generation_varies_shape_base_tilt_and_cream_route_only():
    request = ExpertAvatarRequest.model_validate(
        {
            "category": "marketing",
            "shape": "bean",
            "base": "wide",
            "tilt": "left",
            "inlay": "wrap",
            "expression": "curious",
        }
    )
    prompt = avatar_prompt(request)
    assert "Terracotta #C47F5C" in prompt
    assert "kidney bean" in prompt
    assert "wide low rounded base" in prompt
    assert "tilted gently left" in prompt
    assert "curved cream corner wrap on the lower form" in prompt
    assert "shade" not in prompt.lower()
    assert "Accent placement" not in prompt


def test_reference_set_follows_the_generation_standard():
    import hashlib
    import json
    from typing import get_args

    from backend.api.features.experts.avatar_design import SHAPES, AvatarShape
    from backend.api.features.experts.avatar_generation import (
        REFERENCE_FOLDER,
        GenerationCategory,
        reference_ids,
        reference_images,
    )

    assert set(get_args(AvatarShape)) == set(SHAPES)
    manifest = json.loads((REFERENCE_FOLDER / "manifest.json").read_text())
    for category in get_args(GenerationCategory):
        ids = reference_ids(category)
        assert ids[:2] == ["expert-maria", "expert-mina"]
        assert len(ids) == len(set(ids))
        for name, content, mime in reference_images(category):
            asset_id = name.removesuffix(".png")
            assert mime == "image/png"
            assert hashlib.sha256(content).hexdigest() == manifest[asset_id]["sha256"]
            with Image.open(io.BytesIO(content)) as image:
                assert image.size == (512, 512)
                assert image.mode == "RGB"
    assert reference_ids("content") == ["expert-maria", "expert-mina"]
    assert "hex anchor" in avatar_prompt(ExpertAvatarRequest(category="content"))
