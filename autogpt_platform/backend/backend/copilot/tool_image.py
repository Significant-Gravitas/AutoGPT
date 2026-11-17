"""Bounded image payloads, kept separate from textual tool output."""

import base64
import io
import json
from typing import Any, Literal, Protocol, Self, cast

from PIL import Image, JpegImagePlugin, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from backend.util.truncate import truncate

MAX_IMAGE_BYTES = 3 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2000
IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
IMAGE_READ_DESCRIPTION = (
    "Images: PNG/JPEG/GIF/WebP, up to 3 MiB and 2000px/side; first frame only. "
)
IMAGE_TOO_LARGE_MESSAGE = (
    "Image not shown: exceeds 3 MiB. Resize or compress it and read it again."
)
BINARY_IMAGE_READ_GUIDANCE = (
    "This file has not been viewed. Render PDF pages or extract video frames "
    "to PNG/JPEG, then read the images."
)


class ToolImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    mime_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
    data_base64: str = Field(min_length=1, max_length=4 * ((MAX_IMAGE_BYTES + 2) // 3))

    def mcp(self) -> dict[str, Any]:
        return {"type": "image", "mimeType": self.mime_type, "data": self.data_base64}

    def anthropic(self) -> dict[str, Any]:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.mime_type,
                "data": self.data_base64,
            },
        }

    def openai(self) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{self.mime_type};base64,{self.data_base64}"},
        }


class ImageCarrier(BaseModel):
    """In-process pixels; held and pending payloads serialize them explicitly."""

    _image: ToolImage | None = PrivateAttr(default=None)

    @property
    def image(self) -> ToolImage | None:
        return self._image

    def with_image(self, image: ToolImage | None) -> Self:
        self._image = image
        return self


class HeldImage(BaseModel):
    type: Literal["held_tool_image"] = "held_tool_image"
    text: str
    image: ToolImage


class _AnimatedImage(Protocol):
    is_animated: bool


class ImageReadError(ValueError):
    """Recovery guidance authored here, never interpolated from file contents."""


def prepare_image(raw: bytes) -> ToolImage:
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageReadError(IMAGE_TOO_LARGE_MESSAGE)
    try:
        with Image.open(io.BytesIO(raw)) as image:
            mime = (
                "image/jpeg"
                if image.format == "MPO"
                else Image.MIME.get(image.format or "", "")
            )
            if mime not in IMAGE_MIME_TYPES:
                raise ImageReadError(
                    "Image not shown: convert it to PNG, JPEG, GIF or WebP, then read it again."
                )
            if max(image.size) > MAX_IMAGE_DIMENSION:
                raise ImageReadError(
                    "Image not shown: exceeds 2000 pixels per side. Resize or crop it and read it again."
                )
            pixel_limit = Image.MAX_IMAGE_PIXELS
            if pixel_limit is not None and image.width * image.height > pixel_limit:
                raise Image.DecompressionBombError(
                    "Image exceeds the configured pixel limit"
                )
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if (
                image.format in {"MPO", "GIF", "WEBP", "PNG"}
                and cast(_AnimatedImage, image).is_animated
            ):
                raw, mime = _encode_first_frame(image)
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageReadError(
            "Image not shown: the file is corrupt or cannot be decoded. Re-export it as PNG or JPEG."
        ) from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageReadError(IMAGE_TOO_LARGE_MESSAGE)
    return ToolImage.model_validate(
        {"mime_type": mime, "data_base64": base64.b64encode(raw).decode("ascii")}
    )


def _encode_first_frame(image: Image.Image) -> tuple[bytes, str]:
    output = io.BytesIO()
    if image.format == "MPO":
        jpeg = cast(JpegImagePlugin.JpegImageFile, image)
        jpeg.save(
            output,
            format="JPEG",
            qtables=jpeg.quantization,
            subsampling=JpegImagePlugin.get_sampling(jpeg),
            optimize=True,
        )
        return output.getvalue(), "image/jpeg"
    image.convert("RGBA").save(output, format="PNG")
    return output.getvalue(), "image/png"


def parse_held_image(output: str | dict[str, Any]) -> HeldImage | None:
    try:
        return _parse_held_image(
            json.loads(output) if isinstance(output, str) else output
        )
    except (ValueError, ValidationError):
        return None


def _parse_held_image(data: object) -> HeldImage | None:
    if not isinstance(data, dict):
        return None
    if data.get("type") == "held_tool_image":
        return HeldImage.model_validate(data)
    content = data.get("content")
    if not isinstance(content, list):
        return None
    images = [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == "image"
    ]
    if len(images) != 1:
        return None
    image = images[0]
    return HeldImage(
        text="\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ),
        image=ToolImage.model_validate(
            {"mime_type": image.get("mimeType"), "data_base64": image.get("data")}
        ),
    )


def truncate_mcp_result(result: dict[str, Any], limit: int) -> dict[str, Any]:
    """Cap text without corrupting the separately bounded image payload."""
    blocks = result.get("content", [])
    images = [
        block
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "image"
    ]
    if not images:
        return truncate(result, limit)
    text_result = {
        **result,
        "content": [block for block in blocks if block not in images],
    }
    capped = truncate(text_result, limit)
    capped["content"].extend(images)
    return capped
