import base64
import io
import json
import random
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from backend.copilot import tool_image
from backend.copilot.tool_image import (
    MAX_IMAGE_BYTES,
    parse_held_image,
    prepare_image,
    truncate_mcp_result,
)


@pytest.mark.parametrize(
    "format,mime",
    [
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("GIF", "image/gif"),
        ("WEBP", "image/webp"),
    ],
)
def test_supported_images_are_validated_from_bytes(format, mime):
    output = io.BytesIO()
    Image.new("RGB", (40, 30), "blue").save(output, format=format)
    raw = output.getvalue()
    image = prepare_image(raw)
    assert image.mime_type == mime
    assert base64.b64decode(image.data_base64, validate=True) == raw


@pytest.mark.parametrize("raw", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_invalid_image_has_actionable_error(raw):
    with pytest.raises(ValueError, match="Image not shown.*corrupt"):
        prepare_image(raw)


def test_oversize_image_is_rejected_before_decoding():
    with pytest.raises(ValueError, match="exceeds 3 MiB"):
        prepare_image(b"x" * (MAX_IMAGE_BYTES + 1))


def test_large_dimensions_are_rejected():
    output = io.BytesIO()
    Image.new("RGB", (2001, 1)).save(output, format="PNG")
    with pytest.raises(ValueError, match="exceeds 2000 pixels"):
        prepare_image(output.getvalue())


def test_unsupported_image_format_is_rejected():
    output = io.BytesIO()
    Image.new("RGB", (20, 20)).save(output, format="BMP")
    with pytest.raises(ValueError, match="convert it to PNG"):
        prepare_image(output.getvalue())


def test_text_cap_does_not_truncate_image_bytes():
    output = io.BytesIO()
    Image.effect_noise((512, 512), 100).convert("RGB").save(output, format="PNG")
    image = prepare_image(output.getvalue())
    result = truncate_mcp_result(
        {"content": [{"type": "text", "text": "text" * 10000}, image.mcp()]}, 1000
    )
    assert len(result["content"][0]["text"]) < 1000
    assert result["content"][1] == image.mcp()


@pytest.mark.parametrize(
    "mime,data", [("image/svg+xml", "svg"), ("", ""), ("image/png", None)]
)
def test_unrecognized_held_image_does_not_crash_release(mime, data):
    output = json.dumps(
        {"content": [{"type": "image", "mimeType": mime, "data": data}]}
    )
    assert parse_held_image(output) is None


@pytest.mark.parametrize("format", ["MPO", "GIF", "WEBP", "PNG"])
def test_multi_frame_image_is_reduced_to_its_first_frame(format):
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "blue").save(
        output,
        format=format,
        save_all=True,
        append_images=[Image.new("RGB", (20, 20), "red")],
    )
    raw = output.getvalue()
    with Image.open(io.BytesIO(raw)) as original:
        expected = original.convert("RGBA").tobytes()
    image = prepare_image(raw)
    assert image.mime_type == ("image/jpeg" if format == "MPO" else "image/png")
    with Image.open(io.BytesIO(base64.b64decode(image.data_base64))) as viewed:
        pixels = viewed.convert("RGBA").tobytes()
        if format == "MPO":
            assert list(pixels) == pytest.approx(list(expected), abs=3)
        else:
            assert pixels == expected
        with pytest.raises(EOFError):
            viewed.seek(1)


@pytest.mark.parametrize("subsampling", [0, 2])
def test_photographic_mpo_remains_a_bounded_single_frame_jpeg(subsampling):
    first = Image.frombytes(
        "RGB", (1400, 1200), random.Random(0).randbytes(1400 * 1200 * 3)
    )
    first.paste("blue", (0, 0, 100, 100))
    output = io.BytesIO()
    first.save(
        output,
        format="MPO",
        quality=85,
        subsampling=subsampling,
        save_all=True,
        append_images=[Image.new("RGB", (20, 20), "red")],
    )
    raw = output.getvalue()
    assert len(raw) < MAX_IMAGE_BYTES
    image = prepare_image(raw)
    encoded = base64.b64decode(image.data_base64)
    assert image.mime_type == "image/jpeg"
    assert len(encoded) < MAX_IMAGE_BYTES
    with Image.open(io.BytesIO(encoded)) as viewed:
        assert viewed.format == "JPEG"
        assert viewed.size == first.size
        assert viewed.convert("RGB").getpixel((50, 50)) == pytest.approx(
            (0, 0, 255), abs=3
        )
        with pytest.raises(EOFError):
            viewed.seek(1)


def test_normalized_first_frame_still_obeys_byte_limit(monkeypatch):
    first = Image.frombytes("RGB", (100, 100), random.Random(0).randbytes(30_000))
    output = io.BytesIO()
    first.save(
        output,
        format="WEBP",
        quality=5,
        save_all=True,
        append_images=[Image.new("RGB", first.size, "red")],
    )
    raw = output.getvalue()
    monkeypatch.setattr(tool_image, "MAX_IMAGE_BYTES", len(raw))
    with pytest.raises(ValueError, match="exceeds 3 MiB"):
        prepare_image(raw)


@pytest.mark.parametrize("pixel_limit", [100, 170])
def test_decompression_bomb_warning_and_error_are_rejected(monkeypatch, pixel_limit):
    output = io.BytesIO()
    Image.new("RGB", (16, 16)).save(output, format="PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pixel_limit)
    with pytest.raises(ValueError, match="cannot be decoded"):
        prepare_image(output.getvalue())


def test_concurrent_reads_preserve_warning_filters(monkeypatch):
    first_output, second_output = io.BytesIO(), io.BytesIO()
    Image.new("RGB", (20, 20), "blue").save(first_output, format="PNG")
    Image.new("RGB", (20, 20), "red").save(second_output, format="PNG")
    first_raw, second_raw = first_output.getvalue(), second_output.getvalue()
    first_entered = threading.Event()
    second_entered = threading.Event()
    first_finished = threading.Event()
    original_open = Image.open

    def interleaved_open(file, *args, **kwargs):
        if file.getvalue() == first_raw:
            first_entered.set()
            assert second_entered.wait(timeout=10)
        else:
            second_entered.set()
            assert first_finished.wait(timeout=10)
        return original_open(file, *args, **kwargs)

    def prepare_first():
        try:
            return prepare_image(first_raw)
        finally:
            first_finished.set()

    monkeypatch.setattr(Image, "open", interleaved_open)
    with warnings.catch_warnings():
        original_filters = warnings.filters
        original_values = list(original_filters)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(prepare_first)
            assert first_entered.wait(timeout=10)
            second = pool.submit(prepare_image, second_raw)
            assert base64.b64decode(first.result(timeout=10).data_base64) == first_raw
            assert base64.b64decode(second.result(timeout=10).data_base64) == second_raw
        assert warnings.filters is original_filters
        assert warnings.filters == original_values


@pytest.mark.parametrize("warning_action", ["ignore", "error"])
def test_pixel_limit_is_enforced_before_decoding(monkeypatch, mocker, warning_action):
    output = io.BytesIO()
    Image.new("RGB", (16, 16)).save(output, format="PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 170)
    decode = mocker.patch("PIL.ImageFile.ImageFile.load")
    with warnings.catch_warnings():
        warnings.simplefilter(warning_action, Image.DecompressionBombWarning)
        with pytest.raises(ValueError, match="cannot be decoded"):
            prepare_image(output.getvalue())
    decode.assert_not_called()
