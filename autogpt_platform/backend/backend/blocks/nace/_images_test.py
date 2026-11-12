import asyncio
import base64
import random
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from backend.blocks.nace import _images
from backend.data.execution import ExecutionContext
from backend.util.file import MediaFileType

from ._client import MAX_REQUEST_BYTES

CONTEXT = ExecutionContext(user_id="test-user", graph_exec_id="test-exec")
# The largest raw image whose base64 form fits in an empty request.
FULL_RAW_CAP = MAX_REQUEST_BYTES // 4 * 3


def image_bytes(image_format: str, color: str = "red") -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", (2, 2), color=color)
    if image_format == "MPO":
        # Pillow only writes MPO with a second frame.
        image.save(
            buffer,
            format="MPO",
            save_all=True,
            append_images=[Image.new("RGB", (2, 2), color="blue")],
        )
        assert Image.open(BytesIO(buffer.getvalue())).format == "MPO"
    else:
        image.save(buffer, format=image_format)
    return buffer.getvalue()


def noise_png(width: int, height: int, seed: int) -> bytes:
    """An uncompressed PNG of random pixels, about width * height * 3 bytes."""
    pixels = random.Random(seed).randbytes(width * height * 3)
    buffer = BytesIO()
    Image.frombytes("RGB", (width, height), pixels).save(
        buffer, format="PNG", compress_level=0
    )
    return buffer.getvalue()


def corrupt_png() -> bytes:
    data = bytearray(image_bytes("PNG"))
    idat = data.index(b"IDAT")
    length = int.from_bytes(data[idat - 4 : idat], "big")
    data[idat + 4 + length] ^= 0xFF
    return bytes(data)


def response(content: bytes, content_type: str | None = None) -> MagicMock:
    headers = {"Content-Type": content_type} if content_type else {}
    return MagicMock(content=content, headers=headers)


@pytest.fixture
def downloads(tmp_path, monkeypatch) -> dict[str, MagicMock]:
    """The real store_media_file, with its storage, scan and HTTP boundaries mocked.

    Tests fill the returned dict with the response for each URL.
    """
    responses: dict[str, MagicMock] = {}
    cloud = MagicMock()
    cloud.is_cloud_path.return_value = False
    monkeypatch.setattr("backend.util.file.TEMP_DIR", tmp_path)
    monkeypatch.setattr(
        "backend.util.file.get_cloud_storage_handler", AsyncMock(return_value=cloud)
    )
    monkeypatch.setattr("backend.util.file.scan_content_safe", AsyncMock())
    monkeypatch.setattr(
        "backend.util.file.Requests.get",
        AsyncMock(side_effect=lambda url: responses[url]),
    )
    return responses


def decoded(url: str) -> tuple[str, bytes]:
    head, data = url.split(",", 1)
    return head, base64.b64decode(data)


@pytest.mark.parametrize(
    "image_format,mime",
    [
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("MPO", "image/jpeg"),
        ("WEBP", "image/webp"),
    ],
)
@pytest.mark.parametrize(
    "source",
    ["workspace-id", "workspace-path", "url-missing", "url-generic", "url-non-image"],
)
async def test_extensionless_images_are_typed_by_their_content(
    downloads, monkeypatch, image_format, mime, source
):
    """Names and Content-Type headers do not matter; the image bytes do."""
    content = image_bytes(image_format)
    manager = MagicMock()
    info = MagicMock()
    info.name = "image"
    info.mime_type = mime
    manager.read_file_by_id = AsyncMock(return_value=content)
    manager.read_file = AsyncMock(return_value=content)
    manager.get_file_info = AsyncMock(return_value=info)
    manager.get_file_info_by_path = AsyncMock(return_value=info)
    monkeypatch.setattr(
        "backend.util.workspace.WorkspaceManager", MagicMock(return_value=manager)
    )
    content_type = {
        "url-generic": "application/octet-stream",
        "url-non-image": "text/plain",
    }.get(source)
    downloads["https://example.com/image"] = response(content, content_type)
    ref = {
        "workspace-id": "workspace://file-1",
        "workspace-path": "workspace:///image",
    }.get(source, "https://example.com/image")
    context = CONTEXT.model_copy(
        update={"workspace_id": "ws-1" if source.startswith("workspace") else None}
    )

    [url] = await _images.resolve_images([MediaFileType(ref)], context)

    assert decoded(url) == (f"data:{mime};base64", content)


async def test_a_misnamed_image_is_typed_by_its_content(downloads):
    content = image_bytes("JPEG")
    downloads["https://example.com/photo.png"] = response(content, "image/png")

    [url] = await _images.resolve_images(
        [MediaFileType("https://example.com/photo.png")], CONTEXT
    )

    assert decoded(url) == ("data:image/jpeg;base64", content)


@pytest.mark.parametrize(
    "content",
    [image_bytes("GIF"), b"ordinary text", b"\x89PNG\r\n\x1a\n", corrupt_png()],
    ids=["gif", "text", "png-signature-only", "png-bad-crc"],
)
async def test_unsupported_or_broken_images_are_rejected(downloads, content):
    downloads["https://example.com/image"] = response(content, "image/png")

    with pytest.raises(ValueError, match="Image 1 must be a PNG, JPEG or WebP image"):
        await _images.resolve_images(
            [MediaFileType("https://example.com/image")], CONTEXT
        )


async def test_an_oversized_image_is_refused_before_it_is_read_whole_or_encoded(
    downloads, monkeypatch
):
    downloads["https://example.com/big"] = response(b"\0" * (FULL_RAW_CAP + 4096))
    sniff = MagicMock(side_effect=_images.image_mime_type)
    encode = MagicMock(side_effect=_images._data_url)
    monkeypatch.setattr(_images, "image_mime_type", sniff)
    monkeypatch.setattr(_images, "_data_url", encode)
    reads: list[int] = []

    def counting_open(path, mode):
        handle = open(path, mode)
        read = handle.read
        handle.read = lambda size=-1: reads.append(len(chunk := read(size))) or chunk
        return handle

    monkeypatch.setattr(_images, "open", counting_open, raising=False)

    with pytest.raises(ValueError, match="Image 1 is too large"):
        await _images.resolve_images(
            [MediaFileType("https://example.com/big")], CONTEXT
        )

    assert reads == [FULL_RAW_CAP + 1]
    sniff.assert_not_called()
    encode.assert_not_called()


async def test_images_past_the_request_size_are_refused_and_later_ones_not_fetched(
    downloads, monkeypatch
):
    # Each about 2.2 MB raw, 2.9 MB encoded: one fits the request, two do not.
    for index in range(1, 4):
        downloads[f"https://example.com/{index}.png"] = response(
            noise_png(1024, 716, seed=index)
        )
    store = AsyncMock(side_effect=_images.store_media_file)
    monkeypatch.setattr(_images, "store_media_file", store)

    with pytest.raises(ValueError, match="Image 2 brings the images past 4,194,304"):
        await _images.resolve_images(
            [MediaFileType(f"https://example.com/{i}.png") for i in range(1, 4)],
            CONTEXT,
        )

    assert store.await_count == 2


async def test_images_resolve_one_at_a_time_and_keep_their_own_bytes(
    downloads, monkeypatch
):
    """Two URLs with the same file name land on the same local path."""
    red, blue = image_bytes("PNG", "red"), image_bytes("PNG", "blue")
    downloads["https://a.example/x.png"] = response(red)
    downloads["https://b.example/x.png"] = response(blue)
    real_store = _images.store_media_file
    in_flight = peak = 0

    async def tracking_store(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0)
            return await real_store(**kwargs)
        finally:
            in_flight -= 1

    monkeypatch.setattr(_images, "store_media_file", tracking_store)

    urls = await _images.resolve_images(
        [
            MediaFileType("https://a.example/x.png"),
            MediaFileType("https://b.example/x.png"),
        ],
        CONTEXT,
    )

    assert [decoded(url)[1] for url in urls] == [red, blue]
    assert peak == 1


async def test_no_images_need_no_execution_ids(monkeypatch):
    store = AsyncMock()
    monkeypatch.setattr(_images, "store_media_file", store)

    assert await _images.resolve_images([], ExecutionContext()) == []
    store.assert_not_called()
