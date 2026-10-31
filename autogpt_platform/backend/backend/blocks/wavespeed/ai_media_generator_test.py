"""Tests for AIMediaGeneratorBlock: the submit/poll loop and every way it can
fail. The block's own test_mock replaces generate_media wholesale, so the
request building, the polling state machine and the error branches are only
reachable from here."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from backend.blocks.wavespeed._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.wavespeed.ai_media_generator import (
    WAVESPEED_API_BASE,
    AIMediaGeneratorBlock,
)


def _response(payload: dict) -> MagicMock:
    """Stand-in for backend.util.request.Response — json() is sync there."""
    response = MagicMock()
    response.json.return_value = payload
    return response


def _submitted(prediction_id: str = "pred-123") -> dict:
    return {"data": {"id": prediction_id}}


def _client_error(message: str = "boom") -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(
        request_info=MagicMock(), history=(), status=500, message=message
    )


def _input(**overrides) -> AIMediaGeneratorBlock.Input:
    kwargs: dict = {
        "credentials": TEST_CREDENTIALS_INPUT,
        "model": "bytedance/seedream-v5.0-pro",
        "prompt": "A serene mountain lake at sunrise.",
    }
    kwargs.update(overrides)
    return AIMediaGeneratorBlock.Input(**kwargs)


def _patch_requests(post=None, get=None):
    """Patch the Requests class so every Requests() instance shares our mocks."""
    instance = MagicMock()
    instance.post = post or AsyncMock()
    instance.get = get or AsyncMock()
    return (
        patch(
            "backend.blocks.wavespeed.ai_media_generator.Requests",
            return_value=instance,
        ),
        instance,
    )


def test_get_headers_carries_the_bearer_token():
    headers = AIMediaGeneratorBlock()._get_headers("secret-key")
    assert headers == {
        "Authorization": "Bearer secret-key",
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
async def test_generate_media_submits_prompt_merged_with_extra_inputs():
    """extra_inputs is spread into the body, but prompt must win — otherwise a
    stray "prompt" key in extra_inputs would silently replace the real one."""
    post = AsyncMock(return_value=_response(_submitted()))
    get = AsyncMock(
        return_value=_response(
            {"data": {"status": "completed", "outputs": ["https://cdn/a.png"]}}
        )
    )
    patcher, _ = _patch_requests(post, get)
    with patcher:
        outputs = await AIMediaGeneratorBlock().generate_media(
            _input(extra_inputs={"size": "2048*2048", "prompt": "ignored"}),
            TEST_CREDENTIALS,
        )

    assert outputs == ["https://cdn/a.png"]
    assert post.await_args.args[0] == (
        f"{WAVESPEED_API_BASE}/bytedance/seedream-v5.0-pro"
    )
    assert post.await_args.kwargs["json"] == {
        "size": "2048*2048",
        "prompt": "A serene mountain lake at sunrise.",
    }
    assert get.await_args.args[0] == (
        f"{WAVESPEED_API_BASE}/predictions/pred-123/result"
    )


@pytest.mark.asyncio
async def test_generate_media_polls_until_the_status_is_terminal():
    post = AsyncMock(return_value=_response(_submitted()))
    get = AsyncMock(
        side_effect=[
            _response({"data": {"status": "created"}}),
            _response({"data": {"status": "processing"}}),
            _response(
                {"data": {"status": "completed", "outputs": ["https://cdn/b.mp4"]}}
            ),
        ]
    )
    patcher, _ = _patch_requests(post, get)
    with patcher, patch(
        "backend.blocks.wavespeed.ai_media_generator.asyncio.sleep", new=AsyncMock()
    ) as sleep:
        outputs = await AIMediaGeneratorBlock().generate_media(
            _input(), TEST_CREDENTIALS
        )

    assert outputs == ["https://cdn/b.mp4"]
    assert get.await_count == 3
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_generate_media_keeps_only_string_outputs():
    post = AsyncMock(return_value=_response(_submitted()))
    get = AsyncMock(
        return_value=_response(
            {
                "data": {
                    "status": "completed",
                    "outputs": ["https://cdn/a.png", None, {"url": "x"}, 7],
                }
            }
        )
    )
    patcher, _ = _patch_requests(post, get)
    with patcher:
        outputs = await AIMediaGeneratorBlock().generate_media(
            _input(), TEST_CREDENTIALS
        )

    assert outputs == ["https://cdn/a.png"]


@pytest.mark.asyncio
async def test_generate_media_raises_when_the_submission_call_fails():
    post = AsyncMock(side_effect=_client_error("submit exploded"))
    patcher, _ = _patch_requests(post)
    with patcher, pytest.raises(RuntimeError, match="Failed to submit request"):
        await AIMediaGeneratorBlock().generate_media(_input(), TEST_CREDENTIALS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"message": "quota exceeded"},
        {"data": {}},
        {"data": {"id": 12345}},
        {"data": None},
    ],
)
async def test_generate_media_raises_without_a_usable_prediction_id(payload):
    """A non-string or absent id must fail before polling, not build a URL like
    /predictions/None/result and burn 120 polls on it."""
    post = AsyncMock(return_value=_response(payload))
    get = AsyncMock()
    patcher, _ = _patch_requests(post, get)
    with patcher, pytest.raises(ValueError, match="Missing prediction ID"):
        await AIMediaGeneratorBlock().generate_media(_input(), TEST_CREDENTIALS)

    get.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_media_raises_when_polling_fails():
    post = AsyncMock(return_value=_response(_submitted()))
    get = AsyncMock(side_effect=_client_error("poll exploded"))
    patcher, _ = _patch_requests(post, get)
    with patcher, pytest.raises(RuntimeError, match="Failed to get prediction result"):
        await AIMediaGeneratorBlock().generate_media(_input(), TEST_CREDENTIALS)


@pytest.mark.asyncio
async def test_generate_media_raises_when_a_completed_run_has_no_urls():
    post = AsyncMock(return_value=_response(_submitted()))
    get = AsyncMock(
        return_value=_response({"data": {"status": "completed", "outputs": []}})
    )
    patcher, _ = _patch_requests(post, get)
    with patcher, pytest.raises(ValueError, match="No valid output URLs"):
        await AIMediaGeneratorBlock().generate_media(_input(), TEST_CREDENTIALS)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "timeout"])
async def test_generate_media_surfaces_terminal_failure_statuses(status):
    post = AsyncMock(return_value=_response(_submitted()))
    get = AsyncMock(
        return_value=_response(
            {"data": {"status": status, "error": "upstream said no"}}
        )
    )
    patcher, _ = _patch_requests(post, get)
    with patcher, pytest.raises(RuntimeError, match=f"Generation {status}"):
        await AIMediaGeneratorBlock().generate_media(_input(), TEST_CREDENTIALS)


@pytest.mark.asyncio
async def test_generate_media_reports_missing_error_details():
    post = AsyncMock(return_value=_response(_submitted()))
    get = AsyncMock(return_value=_response({"data": {"status": "failed"}}))
    patcher, _ = _patch_requests(post, get)
    with patcher, pytest.raises(RuntimeError, match="No error details provided"):
        await AIMediaGeneratorBlock().generate_media(_input(), TEST_CREDENTIALS)


@pytest.mark.asyncio
async def test_generate_media_gives_up_after_the_polling_budget():
    post = AsyncMock(return_value=_response(_submitted()))
    get = AsyncMock(return_value=_response({"data": {"status": "processing"}}))
    patcher, _ = _patch_requests(post, get)
    with patcher, patch(
        "backend.blocks.wavespeed.ai_media_generator.asyncio.sleep", new=AsyncMock()
    ):
        with pytest.raises(RuntimeError, match="Maximum polling attempts reached"):
            await AIMediaGeneratorBlock().generate_media(_input(), TEST_CREDENTIALS)

    assert get.await_count == 120


@pytest.mark.asyncio
async def test_run_stores_every_output_and_yields_the_first():
    stored = AsyncMock(side_effect=lambda **kw: f"workspace://{kw['file']}")
    with patch.object(
        AIMediaGeneratorBlock,
        "generate_media",
        new=AsyncMock(return_value=["https://cdn/a.png", "https://cdn/b.png"]),
    ), patch(
        "backend.blocks.wavespeed.ai_media_generator.store_media_file", new=stored
    ):
        outputs = [
            item
            async for item in AIMediaGeneratorBlock().run(
                _input(),
                credentials=TEST_CREDENTIALS,
                execution_context=MagicMock(),
            )
        ]

    assert outputs == [
        ("media_url", "workspace://https://cdn/a.png"),
        (
            "media_urls",
            ["workspace://https://cdn/a.png", "workspace://https://cdn/b.png"],
        ),
    ]


@pytest.mark.asyncio
async def test_run_yields_error_instead_of_raising():
    """Errors have to come back on the error pin; a raise would abort the whole
    graph run rather than letting the user branch on it."""
    with patch.object(
        AIMediaGeneratorBlock,
        "generate_media",
        new=AsyncMock(side_effect=RuntimeError("Generation failed: upstream said no")),
    ):
        outputs = [
            item
            async for item in AIMediaGeneratorBlock().run(
                _input(),
                credentials=TEST_CREDENTIALS,
                execution_context=MagicMock(),
            )
        ]

    assert outputs == [("error", "Generation failed: upstream said no")]
