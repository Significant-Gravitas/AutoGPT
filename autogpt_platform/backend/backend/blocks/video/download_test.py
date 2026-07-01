"""Tests for VideoDownloadBlock: SSRF protection, download size limits,
and playlist restriction (SECRT-1898)."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.video.download import VideoDownloadBlock
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError
from backend.util.file import MAX_FILE_SIZE_BYTES


def _make_execution_context() -> ExecutionContext:
    return ExecutionContext(
        user_id="test-user",
        graph_exec_id="test-graph-exec-id",
        graph_id="test-graph-id",
    )


async def _collect_outputs(
    block: VideoDownloadBlock,
    input_data: VideoDownloadBlock.Input,
    exec_ctx: ExecutionContext,
    node_exec_id: str = "node-1",
) -> dict:
    outputs: dict = {}
    async for name, value in block.run(
        input_data, execution_context=exec_ctx, node_exec_id=node_exec_id
    ):
        outputs[name] = value
    return outputs


class TestSSRFProtection:
    """SSRF protection: blocked URLs must be rejected before reaching yt-dlp."""

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "http://127.0.0.1/video.mp4",
            "http://10.0.0.1/video.mp4",
            "http://169.254.169.254/latest/meta-data/",
            "http://172.16.0.1/video.mp4",
            "http://192.168.1.1/video.mp4",
            "http://[::1]/video.mp4",
        ],
        ids=[
            "file_protocol",
            "loopback_127",
            "private_10",
            "cloud_metadata",
            "private_172",
            "private_192",
            "ipv6_loopback",
        ],
    )
    async def test_blocked_urls_rejected(self, url: str, tmp_path: str):
        block = VideoDownloadBlock()
        block.validate_url = AsyncMock(  # type: ignore[method-assign]
            side_effect=ValueError(f"Blocked: {url}")
        )
        input_data = VideoDownloadBlock.Input(url=url, quality="480p")

        with patch("backend.blocks.video.download.os.makedirs"):
            with pytest.raises(BlockExecutionError, match="URL validation failed"):
                await _collect_outputs(block, input_data, _make_execution_context())


class TestDownloadSizeLimit:
    """yt-dlp must enforce max_filesize during download, not just after."""

    async def test_max_filesize_in_ydl_opts(self, tmp_path: str):
        with patch("backend.blocks.video.download.yt_dlp.YoutubeDL") as mock_cls:
            mock_ydl = MagicMock()
            mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = {"duration": 10, "title": "t"}
            mock_ydl.prepare_filename.return_value = os.path.join(
                str(tmp_path), "test.mp4"
            )
            mock_cls.return_value = mock_ydl

            block = VideoDownloadBlock()
            block._download_video(
                "https://example.com/v.mp4", "720p", "mp4", str(tmp_path), "n1"
            )

            opts = mock_cls.call_args[0][0]
            assert opts["max_filesize"] == MAX_FILE_SIZE_BYTES


class TestPlaylistRestriction:
    """yt-dlp must not download playlists — only the single requested video."""

    async def test_noplaylist_in_ydl_opts(self, tmp_path: str):
        with patch("backend.blocks.video.download.yt_dlp.YoutubeDL") as mock_cls:
            mock_ydl = MagicMock()
            mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl.__exit__ = MagicMock(return_value=False)
            mock_ydl.extract_info.return_value = {"duration": 10, "title": "t"}
            mock_ydl.prepare_filename.return_value = os.path.join(
                str(tmp_path), "test.mp4"
            )
            mock_cls.return_value = mock_ydl

            block = VideoDownloadBlock()
            block._download_video(
                "https://example.com/v.mp4", "720p", "mp4", str(tmp_path), "n1"
            )

            opts = mock_cls.call_args[0][0]
            assert opts.get("noplaylist") is True


class TestValidURLAccepted:
    """A valid public URL should pass validation and yield all outputs."""

    async def test_valid_public_url_produces_output(self):
        block = VideoDownloadBlock()
        block.validate_url = AsyncMock(return_value=None)  # type: ignore[method-assign]
        block._download_video = MagicMock(  # type: ignore[method-assign]
            return_value=("video.mp4", 120.0, "Test Video")
        )
        block._store_output_video = AsyncMock(return_value="video.mp4")  # type: ignore[method-assign]

        input_data = VideoDownloadBlock.Input(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            quality="480p",
        )

        with patch("backend.blocks.video.download.os.makedirs"):
            with patch(
                "backend.blocks.video.download.get_exec_file_path",
                return_value="/tmp/exec",
            ):
                outputs = await _collect_outputs(
                    block, input_data, _make_execution_context()
                )

        block.validate_url.assert_called_once_with(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        assert outputs["video_file"] == "video.mp4"
        assert outputs["duration"] == 120.0
        assert outputs["title"] == "Test Video"
        assert outputs["source_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
