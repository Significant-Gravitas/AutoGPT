from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.video._utils import (
    MAX_VIDEO_FRAME_PIXELS,
    VideoMediaStorage,
    check_video_frame_size,
)
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError
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


@pytest.mark.parametrize(
    "infos",
    [
        {"video_found": True, "video_size": [3840, 2160]},
        {"video_found": True, "video_size": [2160, 3840]},
        {"video_found": True, "video_size": [3840, 2160], "video_rotation": 90},
        {"video_found": True, "video_size": [2880, 2880]},
        {"video_found": True, "video_size": [7680, 1080]},
        {"video_found": True, "video_size": [1920, 1080]},
        {"video_found": False},
    ],
    ids=[
        "landscape_at_limit",
        "portrait_at_limit",
        "rotated_at_limit",
        "square_at_limit",
        "ultrawide_at_limit",
        "under_limit",
        "no_video_stream",
    ],
)
def test_frame_size_check_accepts_frames_within_limit(infos):
    with patch(
        "backend.blocks.video._utils.ffmpeg_parse_infos", return_value=infos
    ) as probe:
        check_video_frame_size("video.mp4", block_name="b", block_id="id")

    probe.assert_called_once_with("video.mp4")


@pytest.mark.parametrize(
    "infos, shown",
    [
        ({"video_size": [3841, 2160]}, "3841x2160"),
        ({"video_size": [2160, 3841]}, "2160x3841"),
        ({"video_size": [2881, 2880]}, "2881x2880"),
        ({"video_size": [3841, 2160], "video_rotation": -90}, "2160x3841"),
        ({"video_size": [6000, 6000]}, "6000x6000"),
    ],
    ids=["landscape", "portrait", "square", "rotated", "large_square"],
)
def test_frame_size_check_refuses_frames_over_limit(infos, shown):
    width, height = infos["video_size"]
    assert width * height > MAX_VIDEO_FRAME_PIXELS
    with (
        patch(
            "backend.blocks.video._utils.ffmpeg_parse_infos",
            return_value={"video_found": True, **infos},
        ),
        pytest.raises(BlockExecutionError, match=f"{shown} is over the limit"),
    ):
        check_video_frame_size("video.mp4", block_name="b", block_id="id")
