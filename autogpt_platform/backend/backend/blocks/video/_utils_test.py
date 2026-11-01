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


def _infos(*video_sizes: tuple[int, int]) -> dict:
    """A parse result with one video stream per size, the first as moviepy's default."""
    streams = [{"stream_type": "audio", "default": True}] + [
        {"stream_type": "video", "size": list(size), "default": i == 0}
        for i, size in enumerate(video_sizes)
    ]
    infos: dict = {"video_found": bool(video_sizes), "inputs": [{"streams": streams}]}
    if video_sizes:
        infos["video_size"] = list(video_sizes[0])
    return infos


@pytest.mark.parametrize(
    "sizes",
    [
        [(3840, 2160)],
        [(2160, 3840)],
        [(2880, 2880)],
        [(7680, 1080)],
        [(1920, 1080)],
        [(1920, 1080), (3840, 2160)],
        [],
    ],
    ids=[
        "landscape_at_limit",
        "portrait_at_limit",
        "square_at_limit",
        "ultrawide_at_limit",
        "under_limit",
        "two_streams_at_limit",
        "no_video_stream",
    ],
)
def test_frame_size_check_accepts_frames_within_limit(sizes):
    with patch(
        "backend.blocks.video._utils.ffmpeg_parse_infos", return_value=_infos(*sizes)
    ) as probe:
        check_video_frame_size("video.mp4", block_name="b", block_id="id")

    probe.assert_called_once_with("video.mp4")


@pytest.mark.parametrize(
    "sizes, shown",
    [
        ([(3841, 2160)], "3841x2160"),
        ([(2160, 3841)], "2160x3841"),
        ([(2881, 2880)], "2881x2880"),
        ([(6000, 6000)], "6000x6000"),
        ([(320, 240), (4096, 2160)], "4096x2160"),
        ([(4096, 2160), (320, 240)], "4096x2160"),
    ],
    ids=[
        "landscape",
        "portrait",
        "square",
        "large_square",
        "second_stream_over",
        "first_stream_over",
    ],
)
def test_frame_size_check_refuses_frames_over_limit(sizes, shown):
    width, height = (int(n) for n in shown.split("x"))
    assert width * height > MAX_VIDEO_FRAME_PIXELS
    with (
        patch(
            "backend.blocks.video._utils.ffmpeg_parse_infos",
            return_value=_infos(*sizes),
        ),
        pytest.raises(BlockExecutionError, match=f"{shown} is over the limit"),
    ):
        check_video_frame_size("video.mp4", block_name="b", block_id="id")
