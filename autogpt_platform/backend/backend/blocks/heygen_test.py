"""Unit tests for CreateHeyGenAvatarVideoBlock's HTTP and polling behavior.

Covers what the block-level test_mock scenario (backend/blocks/heygen.py)
doesn't exercise: the actual create_video/get_video_status request bodies,
and the failed/timeout branches of run().
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.heygen import CreateHeyGenAvatarVideoBlock


def _make_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    return response


@pytest.mark.asyncio
async def test_create_video_posts_avatar_payload_with_api_key_header():
    block = CreateHeyGenAvatarVideoBlock()
    response = _make_response({"data": {"video_id": "vid-123"}})

    requests_instance = MagicMock()
    requests_instance.post = AsyncMock(return_value=response)

    with patch("backend.blocks.heygen.Requests", return_value=requests_instance):
        result = await block.create_video(
            SecretStr("fake-key"), {"type": "avatar", "avatar_id": "a1"}
        )

    assert result == {"data": {"video_id": "vid-123"}}
    requests_instance.post.assert_awaited_once()
    call = requests_instance.post.await_args
    assert call.args[0] == "https://api.heygen.com/v3/videos"
    assert call.kwargs["json"] == {"type": "avatar", "avatar_id": "a1"}
    assert call.kwargs["headers"]["x-api-key"] == "fake-key"


@pytest.mark.asyncio
async def test_get_video_status_hits_video_id_path_with_api_key_header():
    block = CreateHeyGenAvatarVideoBlock()
    response = _make_response({"status": "processing"})

    requests_instance = MagicMock()
    requests_instance.get = AsyncMock(return_value=response)

    with patch("backend.blocks.heygen.Requests", return_value=requests_instance):
        result = await block.get_video_status(SecretStr("fake-key"), "vid-123")

    assert result == {"status": "processing"}
    requests_instance.get.assert_awaited_once()
    call = requests_instance.get.await_args
    assert call.args[0] == "https://api.heygen.com/v3/videos/vid-123"
    assert call.kwargs["headers"]["x-api-key"] == "fake-key"


@pytest.mark.asyncio
async def test_run_includes_optional_title_when_provided():
    """title is only added to the payload when set (issue: untested branch)."""
    block = CreateHeyGenAvatarVideoBlock()
    input_data = CreateHeyGenAvatarVideoBlock.Input(
        credentials={"provider": "heygen", "id": "x", "type": "api_key", "title": "x"},
        avatar_id="a1",
        script="hello",
        title="My Video",
    )

    captured_payload = {}

    async def fake_create_video(api_key, payload):
        captured_payload.update(payload)
        return {"data": {"video_id": "vid-123"}}

    async def fake_get_video_status(api_key, video_id):
        return {"status": "completed", "video_url": "data:video/mp4;base64,AAAA"}

    with (
        patch.object(block, "create_video", side_effect=fake_create_video),
        patch.object(block, "get_video_status", side_effect=fake_get_video_status),
        patch(
            "backend.blocks.heygen.store_media_file",
            new=AsyncMock(return_value="workspace://video.mp4"),
        ),
    ):
        outputs = [
            item
            async for item in block.run(
                input_data,
                credentials=MagicMock(api_key=SecretStr("fake-key")),
                execution_context=MagicMock(),
            )
        ]

    assert captured_payload["title"] == "My Video"
    assert ("video_url", "workspace://video.mp4") in outputs


@pytest.mark.asyncio
async def test_run_raises_on_failed_status():
    block = CreateHeyGenAvatarVideoBlock()
    input_data = CreateHeyGenAvatarVideoBlock.Input(
        credentials={"provider": "heygen", "id": "x", "type": "api_key", "title": "x"},
        avatar_id="a1",
        script="hello",
    )

    async def fake_create_video(api_key, payload):
        return {"data": {"video_id": "vid-123"}}

    async def fake_get_video_status(api_key, video_id):
        return {"status": "failed", "failure_message": "content policy violation"}

    with (
        patch.object(block, "create_video", side_effect=fake_create_video),
        patch.object(block, "get_video_status", side_effect=fake_get_video_status),
    ):
        with pytest.raises(RuntimeError, match="content policy violation"):
            async for _ in block.run(
                input_data,
                credentials=MagicMock(api_key=SecretStr("fake-key")),
                execution_context=MagicMock(),
            ):
                pass


@pytest.mark.asyncio
async def test_run_raises_timeout_error_after_max_polling_attempts():
    block = CreateHeyGenAvatarVideoBlock()
    input_data = CreateHeyGenAvatarVideoBlock.Input(
        credentials={"provider": "heygen", "id": "x", "type": "api_key", "title": "x"},
        avatar_id="a1",
        script="hello",
        max_polling_attempts=5,
        polling_interval=5,
    )

    async def fake_create_video(api_key, payload):
        return {"data": {"video_id": "vid-123"}}

    async def fake_get_video_status(api_key, video_id):
        return {"status": "processing"}

    with (
        patch.object(block, "create_video", side_effect=fake_create_video),
        patch.object(block, "get_video_status", side_effect=fake_get_video_status),
        patch("backend.blocks.heygen.asyncio.sleep", new=AsyncMock(return_value=None)),
    ):
        with pytest.raises(TimeoutError):
            async for _ in block.run(
                input_data,
                credentials=MagicMock(api_key=SecretStr("fake-key")),
                execution_context=MagicMock(),
            ):
                pass
