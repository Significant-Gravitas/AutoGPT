import asyncio
import base64
import io
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict

from backend.api.features.experts.avatar_catalog import (
    COLORS,
    PRESETS,
    AvatarCategory,
    AvatarColor,
)
from backend.api.features.experts.avatar_design import (
    BASES,
    INLAYS,
    SHAPES,
    TILTS,
    AvatarShape,
)
from backend.util.settings import Settings


class ExpertAvatarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: AvatarCategory = "content"
    color: AvatarColor | None = None
    shape: AvatarShape = "pebble"
    base: Literal["compact", "wide", "tall"] = "compact"
    tilt: Literal["level", "left", "right"] = "level"
    inlay: Literal["sweep", "pool", "curl"] = "sweep"
    expression: Literal["friendly", "curious", "focused", "pleased"] = "friendly"


EXPRESSIONS = {
    "friendly": "open oval eyes, relaxed curved brows, tiny closed smile",
    "curious": "open oval eyes, one raised brow, tiny round mouth",
    "focused": "compact oval eyes, low gently angled brows, short diagonal mouth; not angry",
    "pleased": "small upward-curved closed eyes, relaxed brows, small closed smile",
}
REFERENCE_FOLDER = Path(__file__).parent / "avatar_references"


async def generate_avatar(request: ExpertAvatarRequest) -> io.BytesIO:
    settings = Settings()
    async with AsyncOpenAI(
        api_key=settings.secrets.openai_api_key, timeout=180, max_retries=0
    ) as client:
        result = await client.images.edit(
            model=settings.config.expert_avatar_model,
            image=(
                "reference.png",
                (REFERENCE_FOLDER / f"{request.shape}.png").read_bytes(),
                "image/png",
            ),
            prompt=avatar_prompt(request),
            size="1024x1024",
            quality="medium",
            background="transparent",
            output_format="png",
            n=1,
        )
    if not result.data or not result.data[0].b64_json:
        raise ValueError("Image provider returned no image")
    content = base64.b64decode(result.data[0].b64_json, validate=True)
    return await asyncio.to_thread(validate_png, content)


def avatar_prompt(request: ExpertAvatarRequest) -> str:
    color = COLORS[request.color or PRESETS[request.category].color_id]
    return (
        "Create ONE new AutoGPT Clay & Rock specialist avatar. Use the attached image "
        "as a material, lighting and head outline reference. Render only ONE figure. "
        "Replace its color, base, tilt and inlay with the choices below. "
        f"Main mineral hue: {color.label} {color.hex} across head and base. "
        f"Head outline: {SHAPES[request.shape]}. Base: {BASES[request.base]}. "
        f"Tilt: {TILTS[request.tilt]}. Cream path: {INLAYS[request.inlay]}. "
        f"Face: {EXPRESSIONS[request.expression]}. "
        "Exactly two irregular masses: head 55–65% of total height, touching one "
        "stable base. Smooth matte clay, very fine grain, rounded corners. "
        "One small broad flowing cream #EAE2D5 inlay entirely on the LOWER BASE, "
        "8–20% of visible area, rounded boundaries with a narrow recessed material groove. "
        "Head stays wholly main color. No cream head patches, sharp wedges, thin piping, "
        "black collar, open gap, limbs, neck, octopus anatomy, purple, lavender, props, "
        "clothes, accessories, logos or text. Small charcoal eyes, brows and mouth only. "
        "No teeth, blush, highlights or theatrical reactions. Soft upper-left studio light, "
        "front or slight three-quarter view. Square transparent PNG, full figure centered "
        "at 80% of canvas height, safe margins, quiet contact shadow. No backdrop or pedestal."
    )


def validate_png(content: bytes) -> io.BytesIO:
    if len(content) > 5 * 1024 * 1024:
        raise ValueError("Generated image is too large")
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG" or image.size != (1024, 1024):
            raise ValueError("Expected a square 1024px PNG")
        image.load()
        if image.mode != "RGBA" or image.getchannel("A").getextrema() != (0, 255):
            raise ValueError("Expected a transparent PNG with visible artwork")
    return io.BytesIO(content)
