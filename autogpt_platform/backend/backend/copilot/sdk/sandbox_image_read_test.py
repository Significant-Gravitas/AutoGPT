from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.sdk import e2b_file_tools
from backend.copilot.tool_image import MAX_IMAGE_BYTES, prepare_image


@pytest.fixture
def sandbox(monkeypatch):
    sandbox = SimpleNamespace(
        files=SimpleNamespace(
            get_info=AsyncMock(return_value=SimpleNamespace(size=1)),
            read=AsyncMock(),
        )
    )
    monkeypatch.setattr(e2b_file_tools, "_get_sandbox", lambda: sandbox)
    monkeypatch.setattr(e2b_file_tools, "is_sdk_tool_path", lambda _: False)
    monkeypatch.setattr(e2b_file_tools, "resolve_sandbox_path", lambda path: path)
    return sandbox


@pytest.mark.parametrize("chunk_size", [64 * 1024, MAX_IMAGE_BYTES + 100])
async def test_growing_sandbox_image_stops_at_byte_cap(
    monkeypatch, sandbox, chunk_size
):
    consumed = 0
    prepared_sizes = []

    async def chunks():
        nonlocal consumed
        while consumed <= MAX_IMAGE_BYTES:
            consumed += chunk_size
            yield b"x" * chunk_size
        raise AssertionError("Read beyond the image byte cap")

    def prepare(raw):
        prepared_sizes.append(len(raw))
        return prepare_image(raw)

    stream = MagicMock()
    stream.__aiter__.side_effect = chunks
    sandbox.files.read.side_effect = lambda _path, format: (
        stream if format == "stream" else b"x" * (MAX_IMAGE_BYTES + chunk_size)
    )
    monkeypatch.setattr(e2b_file_tools, "prepare_image", prepare)

    result = await e2b_file_tools._handle_read_file(
        {"file_path": "/home/user/growing.png"}
    )

    assert result["isError"]
    assert "exceeds 3 MiB" in result["content"][0]["text"]
    assert prepared_sizes == [MAX_IMAGE_BYTES + 1]
    assert MAX_IMAGE_BYTES < consumed <= MAX_IMAGE_BYTES + chunk_size
    sandbox.files.read.assert_awaited_once_with(
        "/home/user/growing.png", format="stream"
    )
    stream.__aexit__.assert_awaited_once()


async def test_failed_sandbox_image_stream_is_closed(sandbox):
    async def chunks():
        yield b"partial"
        raise OSError("download interrupted")

    stream = MagicMock()
    stream.__aiter__.side_effect = chunks
    sandbox.files.read.return_value = stream

    result = await e2b_file_tools._handle_read_file(
        {"file_path": "/home/user/interrupted.png"}
    )

    assert result["isError"]
    assert "download interrupted" in result["content"][0]["text"]
    assert all(block["type"] != "image" for block in result["content"])
    stream.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("filename", ["archive.zip", "program.exe"])
async def test_sandbox_binary_guidance_does_not_assume_pdf_or_video(sandbox, filename):
    result = await e2b_file_tools._handle_read_file(
        {"file_path": f"/home/user/{filename}"}
    )

    assert result["isError"]
    assert "has not been viewed" in result["content"][0]["text"]
    assert "xxd" in result["content"][0]["text"]
    assert "PDF" not in result["content"][0]["text"]
    sandbox.files.read.assert_not_awaited()
