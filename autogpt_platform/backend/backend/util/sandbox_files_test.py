"""Tests for sandbox file extraction and workspace storage."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.sandbox_files import (
    MAX_BINARY_FILE_SIZE,
    ExtractedFile,
    extract_sandbox_files,
    store_sandbox_files,
)

_WORKDIR = "/home/user"
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(256))


def _sandbox(
    file_listing: list[str],
    file_sizes: dict[str, str | None] | None = None,
    file_contents: dict[str, bytes] | None = None,
) -> MagicMock:
    """Fake E2B sandbox: one `find` call, then a `stat` call per file.

    ``file_sizes`` maps path -> stat stdout (None = stat fails with exit 1).
    """
    sizes = file_sizes or {}
    contents = file_contents or {}
    sandbox = MagicMock()

    async def run(command: str):
        result = MagicMock()
        if command.startswith("find "):
            result.exit_code = 0
            result.stdout = "\n".join(file_listing)
            return result
        for path, size in sizes.items():
            if path in command:
                result.exit_code = 1 if size is None else 0
                result.stdout = size or ""
                return result
        result.exit_code = 0
        result.stdout = "4"
        return result

    async def read(path: str, format: str = "bytes"):
        return contents.get(path, b"data")

    sandbox.commands.run = AsyncMock(side_effect=run)
    sandbox.files.read = AsyncMock(side_effect=read)
    return sandbox


def _ctx() -> MagicMock:
    return MagicMock()


# ── extract_sandbox_files ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_text_file():
    sandbox = _sandbox(
        [f"{_WORKDIR}/notes.md"], file_contents={f"{_WORKDIR}/notes.md": b"# hi"}
    )
    files = await extract_sandbox_files(sandbox, _WORKDIR)
    assert len(files) == 1
    assert files[0].is_text is True
    assert files[0].content == b"# hi"
    assert files[0].relative_path == "notes.md"
    assert files[0].name == "notes.md"


@pytest.mark.asyncio
async def test_extract_binary_file_when_allowed():
    sandbox = _sandbox(
        [f"{_WORKDIR}/chart.png"],
        file_contents={f"{_WORKDIR}/chart.png": _PNG_BYTES},
    )
    files = await extract_sandbox_files(sandbox, _WORKDIR, text_only=False)
    assert len(files) == 1
    assert files[0].is_text is False
    assert files[0].content == _PNG_BYTES


@pytest.mark.asyncio
async def test_binary_extension_matching_is_case_insensitive():
    sandbox = _sandbox(
        [f"{_WORKDIR}/PHOTO.JPG"],
        file_contents={f"{_WORKDIR}/PHOTO.JPG": b"\xff\xd8\xff"},
    )
    files = await extract_sandbox_files(sandbox, _WORKDIR, text_only=False)
    assert [f.name for f in files] == ["PHOTO.JPG"]


@pytest.mark.asyncio
async def test_text_only_mode_skips_binary_files():
    sandbox = _sandbox([f"{_WORKDIR}/chart.png", f"{_WORKDIR}/readme.txt"])
    files = await extract_sandbox_files(sandbox, _WORKDIR, text_only=True)
    assert [f.name for f in files] == ["readme.txt"]


@pytest.mark.asyncio
async def test_unrecognized_extension_is_skipped():
    sandbox = _sandbox([f"{_WORKDIR}/core.dump", f"{_WORKDIR}/binary.exe"])
    files = await extract_sandbox_files(sandbox, _WORKDIR, text_only=False)
    assert files == []


@pytest.mark.asyncio
async def test_oversized_file_is_skipped():
    path = f"{_WORKDIR}/huge.png"
    sandbox = _sandbox([path], file_sizes={path: str(MAX_BINARY_FILE_SIZE + 1)})
    files = await extract_sandbox_files(sandbox, _WORKDIR, text_only=False)
    assert files == []
    sandbox.files.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_stat_failure_skips_file():
    path = f"{_WORKDIR}/gone.png"
    sandbox = _sandbox([path], file_sizes={path: None})
    files = await extract_sandbox_files(sandbox, _WORKDIR, text_only=False)
    assert files == []
    sandbox.files.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_unparseable_stat_output_skips_file():
    path = f"{_WORKDIR}/odd.png"
    sandbox = _sandbox([path], file_sizes={path: "not-a-number"})
    files = await extract_sandbox_files(sandbox, _WORKDIR, text_only=False)
    assert files == []


@pytest.mark.asyncio
async def test_read_failure_skips_file_but_not_others():
    bad, good = f"{_WORKDIR}/bad.png", f"{_WORKDIR}/good.png"
    sandbox = _sandbox([bad, good], file_contents={good: b"ok"})
    sandbox.files.read = AsyncMock(
        side_effect=lambda path, format="bytes": (
            (_ for _ in ()).throw(OSError("io")) if path == bad else b"ok"
        )
    )
    files = await extract_sandbox_files(sandbox, _WORKDIR, text_only=False)
    assert [f.name for f in files] == ["good.png"]


@pytest.mark.asyncio
async def test_empty_find_output_returns_no_files():
    sandbox = _sandbox([])
    assert await extract_sandbox_files(sandbox, _WORKDIR) == []


# ── store_sandbox_files ────────────────────────────────────────────


def _extracted(name: str, content: bytes, is_text: bool) -> ExtractedFile:
    return ExtractedFile(
        path=f"{_WORKDIR}/{name}",
        relative_path=name,
        name=name,
        content=content,
        is_text=is_text,
    )


@pytest.mark.asyncio
async def test_store_text_file_sets_content_and_workspace_ref():
    store = AsyncMock(return_value="workspace://abc123")
    with patch("backend.util.sandbox_files.store_media_file", store):
        outputs = await store_sandbox_files(
            [_extracted("notes.md", b"# hi", is_text=True)], _ctx()
        )
    assert outputs[0].content == "# hi"
    assert outputs[0].workspace_ref == "workspace://abc123"


@pytest.mark.asyncio
async def test_store_binary_file_uses_placeholder_content():
    store = AsyncMock(return_value="workspace://img1")
    with patch("backend.util.sandbox_files.store_media_file", store):
        outputs = await store_sandbox_files(
            [_extracted("chart.png", _PNG_BYTES, is_text=False)], _ctx()
        )
    assert outputs[0].workspace_ref == "workspace://img1"
    assert outputs[0].content == f"[Binary file: {len(_PNG_BYTES)} bytes]"
    # The stored payload is the base64 data URI of the raw bytes.
    assert store.await_args is not None
    assert store.await_args.kwargs["file"] == (
        f"data:image/png;base64,{base64.b64encode(_PNG_BYTES).decode()}"
    )


@pytest.mark.asyncio
async def test_store_binary_in_graph_context_keeps_data_uri_content():
    # Non-workspace context: store_media_file returns a data URI instead of a
    # workspace ref — that URI must become the content so the bytes survive.
    data_uri = f"data:image/png;base64,{base64.b64encode(_PNG_BYTES).decode()}"
    store = AsyncMock(return_value=data_uri)
    with patch("backend.util.sandbox_files.store_media_file", store):
        outputs = await store_sandbox_files(
            [_extracted("chart.png", _PNG_BYTES, is_text=False)], _ctx()
        )
    assert outputs[0].workspace_ref is None
    assert outputs[0].content == data_uri


@pytest.mark.asyncio
async def test_store_failure_falls_back_to_data_uri_for_binary():
    store = AsyncMock(side_effect=RuntimeError("storage down"))
    with patch("backend.util.sandbox_files.store_media_file", store):
        outputs = await store_sandbox_files(
            [_extracted("chart.png", _PNG_BYTES, is_text=False)], _ctx()
        )
    assert outputs[0].workspace_ref is None
    assert outputs[0].content.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_store_failure_fallback_respects_size_cap():
    # A binary too large for the configured cap must NOT be inlined as a data
    # URI on storage failure — the placeholder stays, so a broken storage
    # backend can't balloon execution payloads.
    store = AsyncMock(side_effect=RuntimeError("storage down"))
    config = MagicMock(max_file_size_mb=1)
    big = bytes(2 * 1024 * 1024)  # 2MB raw -> ~2.7MB data URI > 1MB cap
    with (
        patch("backend.util.sandbox_files.store_media_file", store),
        patch("backend.util.sandbox_files.Config", return_value=config),
    ):
        outputs = await store_sandbox_files(
            [_extracted("huge.png", big, is_text=False)], _ctx()
        )
    assert outputs[0].workspace_ref is None
    assert outputs[0].content == f"[Binary file: {len(big)} bytes]"


@pytest.mark.asyncio
async def test_store_failure_keeps_decoded_text_content():
    store = AsyncMock(side_effect=RuntimeError("storage down"))
    with patch("backend.util.sandbox_files.store_media_file", store):
        outputs = await store_sandbox_files(
            [_extracted("notes.md", b"# hi", is_text=True)], _ctx()
        )
    assert outputs[0].workspace_ref is None
    assert outputs[0].content == "# hi"


@pytest.mark.asyncio
async def test_store_unknown_extension_defaults_octet_stream_mime():
    store = AsyncMock(return_value="workspace://f1")
    with patch("backend.util.sandbox_files.store_media_file", store):
        # mimetypes tables differ per platform (.7z is known on Linux but not
        # Windows) — use an extension no table knows.
        await store_sandbox_files(
            [_extracted("blob.zz9unknown", b"\x00\x01", is_text=False)], _ctx()
        )
    assert store.await_args is not None
    assert store.await_args.kwargs["file"].startswith(
        "data:application/octet-stream;base64,"
    )
