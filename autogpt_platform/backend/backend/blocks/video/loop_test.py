from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.video.loop import LoopVideoBlock
from backend.data.execution import ExecutionContext
from backend.util.type import MediaFileType


async def test_a_loop_with_neither_duration_nor_count_is_refused_before_storing():
    store_media_file = AsyncMock()

    with (
        patch("backend.blocks.video.loop.store_media_file", new=store_media_file),
        pytest.raises(ValueError, match="Either 'duration' or 'n_loops'"),
    ):
        await anext(
            LoopVideoBlock().run(
                LoopVideoBlock.Input(video_in=MediaFileType("clip.mp4")),
                execution_context=ExecutionContext(
                    user_id="user-1", graph_exec_id="run-1", node_exec_id="node-1"
                ),
            )
        )

    store_media_file.assert_not_called()
