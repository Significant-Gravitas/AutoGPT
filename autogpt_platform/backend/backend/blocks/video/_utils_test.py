from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.video._utils import VideoMediaStorage
from backend.data.execution import ExecutionContext
from backend.util.file import MediaFileType


@pytest.mark.parametrize(
    "method, return_format",
    [
        ("_store_input_video", "for_local_processing"),
        ("_store_output_video", "for_block_output"),
    ],
)
async def test_storage_step_stores_in_its_return_format(method, return_format):
    exec_ctx = ExecutionContext(user_id="test-user", graph_exec_id="test-graph-exec")
    with patch(
        "backend.blocks.video._utils.store_media_file",
        new_callable=AsyncMock,
        return_value=MediaFileType("stored.mp4"),
    ) as store:
        stored = await getattr(VideoMediaStorage(), method)(
            exec_ctx, MediaFileType("video.mp4")
        )

    assert stored == "stored.mp4"
    store.assert_awaited_once_with(
        file="video.mp4", execution_context=exec_ctx, return_format=return_format
    )
