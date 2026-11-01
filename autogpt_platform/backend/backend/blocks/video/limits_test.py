"""Frame-size and input-count limits on the video blocks.

Each test runs a block on a tiny synthetic video and spies on `VideoFileClip`
to check that a file over `MAX_VIDEO_FRAME_PIXELS` is refused before it is
opened.
"""

import subprocess
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import imageio_ffmpeg
import pytest

from backend.blocks.elevenlabs._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.video._utils import VideoMediaStorage
from backend.blocks.video.add_audio import AddAudioToVideoBlock
from backend.blocks.video.clip import VideoClipBlock
from backend.blocks.video.concat import MAX_CONCAT_INPUTS, VideoConcatBlock
from backend.blocks.video.duration import MediaDurationBlock
from backend.blocks.video.loop import LoopVideoBlock
from backend.blocks.video.narration import VideoNarrationBlock
from backend.blocks.video.text_overlay import VideoTextOverlayBlock
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError

SMALL = "small.mp4"
PORTRAIT_AT_LIMIT = "portrait_at_limit.mp4"
PORTRAIT_OVERSIZED = "portrait_oversized.mp4"
OVERSIZED = "oversized.mp4"


def _make_video(path: Path, size: str) -> None:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={size}:r=25",
            "-frames:v",
            "2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def video_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("videos")
    _make_video(d / SMALL, "320x240")
    _make_video(d / PORTRAIT_AT_LIMIT, "2160x3840")
    _make_video(d / PORTRAIT_OVERSIZED, "2160x4096")
    # DCI 4K: 4096x2160 is just over the 3840x2160 limit
    _make_video(d / OVERSIZED, "4096x2160")
    return d


@pytest.fixture
def exec_ctx() -> ExecutionContext:
    return ExecutionContext(
        user_id="test-user",
        graph_exec_id="test-graph-exec",
        node_exec_id="test-node-exec",
    )


def _patch_local_storage(stack: ExitStack, module: str, video_dir: Path) -> None:
    """Serve a block's inputs from `video_dir` by file name."""
    stack.enter_context(
        patch(
            f"{module}.get_exec_file_path",
            side_effect=lambda _graph_exec_id, path: str(video_dir / path),
        )
    )
    stack.enter_context(
        patch(
            f"{module}.store_media_file",
            new=AsyncMock(side_effect=lambda file, **_: file),
            create=True,
        )
    )
    stack.enter_context(
        patch.object(
            VideoMediaStorage,
            "_store_input_video",
            new=AsyncMock(side_effect=lambda _ctx, file: file),
        )
    )


async def _run(block, input_data, exec_ctx, **kwargs) -> dict:
    outputs = {}
    async for name, value in block.run(
        input_data,
        execution_context=exec_ctx,
        node_exec_id="test-node-exec",
        **kwargs,
    ):
        outputs[name] = value
    return outputs


async def _assert_refused_before_open(
    module, block, input_data, exec_ctx, video_dir, **kwargs
):
    with ExitStack() as stack:
        _patch_local_storage(stack, module, video_dir)
        video_file_clip = stack.enter_context(patch(f"{module}.VideoFileClip"))
        with pytest.raises(BlockExecutionError, match="over the limit"):
            await _run(block, input_data, exec_ctx, **kwargs)
    video_file_clip.assert_not_called()


async def test_media_duration_refuses_oversized_frame(video_dir, exec_ctx):
    block = MediaDurationBlock()
    await _assert_refused_before_open(
        "backend.blocks.video.duration",
        block,
        MediaDurationBlock.Input(media_in=OVERSIZED),
        exec_ctx,
        video_dir,
    )


async def test_media_duration_refuses_oversized_portrait_frame(video_dir, exec_ctx):
    block = MediaDurationBlock()
    await _assert_refused_before_open(
        "backend.blocks.video.duration",
        block,
        MediaDurationBlock.Input(media_in=PORTRAIT_OVERSIZED),
        exec_ctx,
        video_dir,
    )


@pytest.mark.parametrize("video", [SMALL, PORTRAIT_AT_LIMIT])
async def test_media_duration_reads_in_bounds_video(video_dir, exec_ctx, video):
    block = MediaDurationBlock()
    with ExitStack() as stack:
        _patch_local_storage(stack, "backend.blocks.video.duration", video_dir)
        outputs = await _run(block, MediaDurationBlock.Input(media_in=video), exec_ctx)
    assert outputs["duration"] > 0


async def test_clip_refuses_oversized_frame(video_dir, exec_ctx):
    block = VideoClipBlock()
    await _assert_refused_before_open(
        "backend.blocks.video.clip",
        block,
        VideoClipBlock.Input(video_in=OVERSIZED, start_time=0.0, end_time=0.04),
        exec_ctx,
        video_dir,
    )


async def test_loop_refuses_oversized_frame(video_dir, exec_ctx):
    block = LoopVideoBlock()
    await _assert_refused_before_open(
        "backend.blocks.video.loop",
        block,
        LoopVideoBlock.Input(video_in=OVERSIZED, n_loops=2),
        exec_ctx,
        video_dir,
    )


async def test_add_audio_refuses_oversized_frame(video_dir, exec_ctx):
    block = AddAudioToVideoBlock()
    await _assert_refused_before_open(
        "backend.blocks.video.add_audio",
        block,
        AddAudioToVideoBlock.Input(video_in=OVERSIZED, audio_in=SMALL),
        exec_ctx,
        video_dir,
    )


async def test_narration_refuses_oversized_frame(video_dir, exec_ctx):
    block = VideoNarrationBlock()
    with patch.object(
        VideoNarrationBlock, "_generate_narration_audio", return_value=b""
    ):
        await _assert_refused_before_open(
            "backend.blocks.video.narration",
            block,
            VideoNarrationBlock.Input.model_validate(
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "video_in": OVERSIZED,
                    "script": "Hello",
                }
            ),
            exec_ctx,
            video_dir,
            credentials=TEST_CREDENTIALS,
        )


async def test_text_overlay_refuses_oversized_frame(video_dir, exec_ctx):
    block = VideoTextOverlayBlock()
    await _assert_refused_before_open(
        "backend.blocks.video.text_overlay",
        block,
        VideoTextOverlayBlock.Input(video_in=OVERSIZED, text="Hello"),
        exec_ctx,
        video_dir,
    )


async def test_concat_checks_every_input(video_dir, exec_ctx):
    block = VideoConcatBlock()
    await _assert_refused_before_open(
        "backend.blocks.video.concat",
        block,
        VideoConcatBlock.Input(videos=[SMALL, SMALL, OVERSIZED]),
        exec_ctx,
        video_dir,
    )


async def test_concat_refuses_more_than_max_inputs(video_dir, exec_ctx):
    block = VideoConcatBlock()
    store_input = AsyncMock(side_effect=lambda _ctx, file: file)
    with (
        patch.object(VideoConcatBlock, "_store_input_video", new=store_input),
        patch("backend.blocks.video.concat.VideoFileClip") as video_file_clip,
        pytest.raises(BlockExecutionError, match=f"At most {MAX_CONCAT_INPUTS}"),
    ):
        await _run(
            block,
            VideoConcatBlock.Input(videos=[SMALL] * (MAX_CONCAT_INPUTS + 1)),
            exec_ctx,
        )
    store_input.assert_not_awaited()
    video_file_clip.assert_not_called()


async def test_concat_accepts_max_inputs(video_dir, exec_ctx):
    block = VideoConcatBlock()
    concat = MagicMock(return_value=1.0)
    with (
        patch(
            "backend.blocks.video.concat.get_exec_file_path",
            side_effect=lambda _graph_exec_id, path: str(video_dir / path),
        ),
        patch.object(
            VideoConcatBlock,
            "_store_input_video",
            new=AsyncMock(side_effect=lambda _ctx, file: file),
        ),
        patch.object(
            VideoConcatBlock,
            "_store_output_video",
            new=AsyncMock(return_value="out.mp4"),
        ),
        patch.object(VideoConcatBlock, "_concat_videos", new=concat),
    ):
        outputs = await _run(
            block,
            VideoConcatBlock.Input(videos=[SMALL] * MAX_CONCAT_INPUTS),
            exec_ctx,
        )
    assert outputs["total_duration"] == 1.0
    assert len(concat.call_args.args[0]) == MAX_CONCAT_INPUTS
