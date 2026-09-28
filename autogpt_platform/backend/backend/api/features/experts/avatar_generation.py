import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict

from backend.api.features.experts.avatar_catalog import PALETTE
from backend.api.features.experts.avatar_design import (
    BASES,
    EXPRESSIONS,
    INLAYS,
    SHAPES,
    TILTS,
    AvatarInlay,
    AvatarShape,
)
from backend.util.settings import Settings

# A candidate may belong to a work category or, with none chosen, to the
# warm-stone General family. Otto's lavender is never available.
GenerationCategory = Literal[
    "marketing",
    "sales",
    "finance",
    "support",
    "operations",
    "research",
    "content",
    "development",
    "general",
]


class ExpertAvatarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: GenerationCategory = "general"
    shape: AvatarShape = "pebble"
    base: Literal["compact", "wide", "tall"] = "compact"
    tilt: Literal["level", "left", "right"] = "level"
    inlay: AvatarInlay = "sweep"
    expression: Literal["friendly", "curious", "focused", "pleased"] = "friendly"


REFERENCE_FOLDER = Path(__file__).parent / "avatar_references"
# Reference order follows the generation standard: Maria fixes finish, light,
# face and the cream material; Mina fixes the silhouette range and where cream
# sits; an accepted peer of the requested family is the color reference.
FINISH_REFERENCE = "expert-maria"
SILHOUETTE_REFERENCE = "expert-mina"
COLOR_REFERENCES: dict[GenerationCategory, str | None] = {
    "marketing": "expert-maria",
    "sales": "expert-max",
    "finance": "expert-mina",
    "support": "expert-riley",
    "operations": "expert-harper",
    "research": "expert-nadia",
    "development": "expert-devon",
    "general": "expert-general-01",
    # Content has no accepted identity yet; the hex anchor carries the color.
    "content": None,
}


def reference_ids(category: GenerationCategory) -> list[str]:
    ids = [FINISH_REFERENCE, SILHOUETTE_REFERENCE]
    peer = COLOR_REFERENCES[category]
    if peer and peer not in ids:
        ids.append(peer)
    return ids


def reference_images(category: GenerationCategory) -> list[tuple[str, bytes, str]]:
    manifest = json.loads((REFERENCE_FOLDER / "manifest.json").read_text())
    images = []
    for asset_id in reference_ids(category):
        content = (REFERENCE_FOLDER / f"{asset_id}.png").read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest[asset_id]["sha256"]:
            raise ValueError(f"Reference {asset_id} does not match its recorded hash")
        images.append((f"{asset_id}.png", content, "image/png"))
    return images


async def generate_avatar(request: ExpertAvatarRequest) -> io.BytesIO:
    settings = Settings()
    async with AsyncOpenAI(
        api_key=settings.secrets.openai_api_key, timeout=180, max_retries=0
    ) as client:
        result = await client.images.edit(
            model=settings.config.expert_avatar_model,
            image=reference_images(request.category),
            prompt=avatar_prompt(request),
            size="1024x1024",
            quality="high",
            background="opaque",
            output_format="png",
            n=1,
        )
    if not result.data or not result.data[0].b64_json:
        raise ValueError("Image provider returned no image")
    content = base64.b64decode(result.data[0].b64_json, validate=True)
    return await asyncio.to_thread(validate_png, content)


def avatar_prompt(request: ExpertAvatarRequest) -> str:
    palette = PALETTE[request.category]
    peer = COLOR_REFERENCES[request.category]
    color_reference = (
        f"The third attached image is an accepted {request.category} peer: match its "
        "color under the same light; do not copy its identity."
        if peer and peer not in (FINISH_REFERENCE, SILHOUETTE_REFERENCE)
        else "No color peer is attached; take the color from the hex anchor."
    )
    return (
        "Create ONE new AutoGPT Clay & Rock specialist Expert as a new identity in "
        "the attached family. The first attached image (Maria) fixes the material, "
        "finish, studio light, face construction and the cream material. The second "
        "(Mina) fixes the permitted silhouette range and where the cream sits. "
        f"{color_reference} Render only one figure, no text, props, accessories, "
        "clothing, logos or extra stones.\n"
        f"Material: one mineral color over both masses, {palette.label} {palette.hex} "
        "as the material anchor, matched to the references under the same light. "
        "Smooth clay/rock with a soft, low-sheen finish, broad gentle highlights, "
        "slow tonal transitions and fine subdued mineral texture. No per-figure hue, "
        "saturation or gloss changes; no glossy plastic, wet glare, hard rims or "
        "coarse rubble.\n"
        "Construction: exactly two touching irregular primary masses, a head above a "
        "compact limbless base, no neck, arms, feet, ears or costume. "
        f"Head: {SHAPES[request.shape]}; a rounded sculptural volume with depth, "
        "broad convex surfaces, gently receding sides and generous transitions; no "
        "sharp faceting, spikes, fragile tips, horns or deep clefts. "
        f"Base: {BASES[request.base]}. {TILTS[request.tilt].capitalize()}. "
        "Narrow soft contact shadow between the masses; no black collar, gap or "
        "floating head. Avoid any elongated, bulbous, cleft or anatomical reading of "
        "the head, the base or the two together; keep a broad grounded mass with a "
        "stable footprint.\n"
        f"Cream: {INLAYS[request.inlay]}, cream #EAE2D5, entirely on the lower form "
        "below the head/body join, small and broad, with a gently curving boundary, "
        "generous radii and a narrow recessed material groove. No cream on the head, "
        "no patches, tips, sharp wedges, straight bands, thin piping or second "
        "accents; the underside stays the main color.\n"
        f"Face: {EXPRESSIONS[request.expression]}, drawn in charcoal only, placed "
        "optically on the usable front surface of the head, eyes and mouth on one "
        "local centerline with balanced clear space, gaze toward the viewer. No "
        "teeth, blush, eye sparkle or theatrical reaction.\n"
        "Scene: warm off-white studio sweep, large soft light from the upper left, "
        "quiet contact shadow to the right, near-frontal camera, the complete figure "
        "centered with comfortable margins at about 70 percent of the frame height. "
        "Opaque square PNG. No purple, lavender or plum, and no octopus anatomy."
    )


def validate_png(content: bytes) -> io.BytesIO:
    if len(content) > 5 * 1024 * 1024:
        raise ValueError("Generated image is too large")
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG" or image.size != (1024, 1024):
            raise ValueError("Expected a square 1024px PNG")
        image.load()
        if image.mode not in ("RGB", "RGBA"):
            raise ValueError("Expected an opaque color PNG")
        if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
            raise ValueError("Expected an opaque studio tile, not a cut-out")
        # getcolors() gives up (None) past its limit; a short list means a
        # flat or near-flat tile with no rendered figure on it.
        if image.convert("RGB").getcolors(256) is not None:
            raise ValueError("Expected visible artwork")
    return io.BytesIO(content)
