import base64
import io

import pytest
from PIL import Image

from backend.copilot.tool_image import (
    MAX_IMAGE_BYTES,
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
