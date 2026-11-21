from unittest.mock import AsyncMock

from backend.blocks import ai_shortform_video_block as revid
from backend.data.execution import ExecutionContext


async def test_shortform_video_runs_when_webhook_site_is_unavailable(monkeypatch):
    block = revid.AIShortformVideoCreatorBlock()
    webhook = AsyncMock(side_effect=ConnectionError("Webhook.site unavailable"))
    render = AsyncMock(return_value={"pid": "test_pid"})
    ready_url = "data:video/mp4;base64,AAAA"
    status = AsyncMock(return_value={"status": "ready", "videoUrl": ready_url})
    store = AsyncMock(return_value="workspace://test_video")
    monkeypatch.setattr(block, "create_webhook", webhook)
    monkeypatch.setattr(block, "create_video", render)
    monkeypatch.setattr(block, "check_video_status", status)
    monkeypatch.setattr(revid, "store_media_file", store)

    context = ExecutionContext()
    outputs = [
        output
        async for output in block.run(
            block.Input.model_validate(block.test_input),
            credentials=revid.TEST_CREDENTIALS,
            execution_context=context,
        )
    ]

    assert outputs == [("video_url", "workspace://test_video")]
    webhook.assert_not_awaited()
    render.assert_awaited_once()
    assert render.await_args is not None
    assert render.await_args.args[1]["webhook"] is None
    status.assert_awaited_once_with(revid.TEST_CREDENTIALS.api_key, "test_pid")
    store.assert_awaited_once_with(
        file=ready_url,
        execution_context=context,
        return_format="for_block_output",
    )
