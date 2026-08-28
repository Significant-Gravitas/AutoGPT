import hashlib
import io
import logging
import re
import unicodedata
from enum import Enum

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from prisma.models import AgentGraph
from replicate.client import Client as ReplicateClient
from replicate.exceptions import ReplicateError
from replicate.helpers import FileOutput

from backend.data.graph import GraphBaseMeta
from backend.data.model import CredentialsMetaInput, ProviderName
from backend.integrations.credentials_store import ideogram_credentials
from backend.util.request import Requests
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class ImageSize(str, Enum):
    LANDSCAPE = "1024x768"


class ImageStyle(str, Enum):
    DIGITAL_ART = "digital art"


async def generate_agent_image(agent: GraphBaseMeta | AgentGraph) -> io.BytesIO:
    if settings.config.use_agent_image_generation_v2:
        if not ideogram_credentials.api_key.get_secret_value():
            return generate_local_agent_image(agent)
        return await generate_agent_image_v2(graph=agent)
    else:
        if not settings.secrets.replicate_api_key:
            return generate_local_agent_image(agent)
        return await generate_agent_image_v1(agent=agent)


def _normalized_graph_text(value: object, fallback: str, max_length: int) -> str:
    text = unicodedata.normalize("NFKC", value if isinstance(value, str) else "")
    text = " ".join(character if character.isprintable() else " " for character in text)
    text = re.sub(r"\s+", " ", text).strip()
    return (text or fallback)[:max_length]


def _display_text(value: str, fallback: str) -> str:
    text = "".join(character if character.isascii() else " " for character in value)
    text = re.sub(r"[^A-Za-z0-9 &+._'-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip() or fallback


def _font(
    size: int, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(font_name, size)
    except OSError:
        return ImageFont.load_default()


def generate_local_agent_image(agent: GraphBaseMeta | AgentGraph) -> io.BytesIO:
    name = _normalized_graph_text(getattr(agent, "name", None), "AI Agent", 160)
    description = _normalized_graph_text(
        getattr(agent, "description", None), "Automation workspace", 600
    )
    digest = hashlib.sha512(f"{name}\0{description}".encode()).digest()

    width, height = 1024, 768
    hue_a = (digest[0] * 0.7 + 30, digest[1] * 0.45 + 20, digest[2] * 0.55 + 35)
    hue_b = (digest[3] * 0.35 + 8, digest[4] * 0.45 + 12, digest[5] * 0.6 + 25)
    background = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(background)
    for y in range(height):
        ratio = y / (height - 1)
        color = tuple(
            round(start * (1 - ratio) + end * ratio) for start, end in zip(hue_a, hue_b)
        )
        draw.line((0, y, width, y), fill=color)

    glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow = ImageDraw.Draw(glow_layer)
    accent = (120 + digest[6] // 2, 90 + digest[7] // 2, 160 + digest[8] // 3)
    for index in range(4):
        radius = 130 + digest[9 + index] // 2
        x = (digest[13 + index] / 255) * width
        y = (digest[17 + index] / 255) * height
        glow.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(*accent, 70 - index * 8),
        )
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(70))
    image = Image.alpha_composite(background.convert("RGBA"), glow_layer)

    art_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    art = ImageDraw.Draw(art_layer)
    nodes: list[tuple[int, int]] = []
    for index in range(8):
        x = 90 + ((digest[21 + index] * 31 + index * 137) % 840)
        y = 90 + ((digest[29 + index] * 19 + index * 83) % 580)
        nodes.append((x, y))
    for index, start in enumerate(nodes):
        end = nodes[(index * 3 + 2) % len(nodes)]
        art.line((*start, *end), fill=(255, 255, 255, 42), width=3)
    for index, (x, y) in enumerate(nodes):
        radius = 8 + digest[37 + index] % 13
        art.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255, 70),
            outline=(255, 255, 255, 145),
            width=2,
        )

    card = (96, 132, 928, 636)
    art.rounded_rectangle(card, radius=46, fill=(5, 8, 28, 190))
    art.rounded_rectangle(card, radius=46, outline=(255, 255, 255, 75), width=2)
    art.rounded_rectangle(
        (142, 180, 258, 296),
        radius=30,
        fill=(*accent, 230),
        outline=(255, 255, 255, 125),
        width=2,
    )

    display_name = _display_text(name, "AI Agent")
    initials = "".join(part[0] for part in display_name.split()[:2]).upper() or "AI"
    initials_font = _font(48, bold=True)
    initials_box = art.textbbox((0, 0), initials, font=initials_font)
    initials_width = initials_box[2] - initials_box[0]
    initials_height = initials_box[3] - initials_box[1]
    art.text(
        (200 - initials_width / 2, 238 - initials_height / 2 - initials_box[1]),
        initials,
        font=initials_font,
        fill=(255, 255, 255, 255),
    )

    title = display_name
    title_font = _font(58, bold=True)
    for font_size in (58, 52, 46, 40, 34, 28):
        candidate_font = _font(font_size, bold=True)
        if art.textlength(title, font=candidate_font) <= 570:
            title_font = candidate_font
            break
    while art.textlength(title, font=title_font) > 570 and len(title) > 4:
        title = f"{title[:-4].rstrip()}..."
    art.text((304, 190), title, font=title_font, fill=(255, 255, 255, 255))
    art.text(
        (306, 270),
        "AUTONOMOUS WORKFLOW",
        font=_font(20, bold=True),
        fill=(*accent, 255),
    )

    art.line((142, 350, 882, 350), fill=(255, 255, 255, 50), width=2)
    for index, label in enumerate(("PLAN", "ACT", "DELIVER")):
        x = 148 + index * 245
        art.rounded_rectangle(
            (x, 414, x + 196, 510),
            radius=22,
            fill=(255, 255, 255, 18),
            outline=(255, 255, 255, 50),
            width=2,
        )
        art.ellipse((x + 22, 442, x + 42, 462), fill=(*accent, 255))
        art.text((x + 58, 438), label, font=_font(22, bold=True), fill="white")

    art.text(
        (145, 574),
        "AUTOGPT MARKETPLACE",
        font=_font(18, bold=True),
        fill=(255, 255, 255, 145),
    )
    image = Image.alpha_composite(image, art_layer).convert("RGB")

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=91, optimize=True, progressive=True)
    output.seek(0)
    return output


async def generate_agent_image_v2(graph: GraphBaseMeta | AgentGraph) -> io.BytesIO:
    """
    Generate an image for an agent using Ideogram model.
    Returns:
        str: The URL of the generated image
    """
    if not ideogram_credentials.api_key:
        raise ValueError("Missing Ideogram API key")

    from backend.blocks.ideogram import (
        AspectRatio,
        ColorPalettePreset,
        IdeogramModelBlock,
        IdeogramModelName,
        MagicPromptOption,
        StyleType,
        UpscaleOption,
    )

    name = graph.name
    description = f"{name} ({graph.description})" if graph.description else name

    prompt = (
        "Create a visually striking retro-futuristic vector pop art illustration "
        f'prominently featuring "{name}" in bold typography. The image clearly and '
        f"literally depicts a {description}, along with recognizable objects directly "
        f"associated with the primary function of a {name}. "
        f"Ensure the imagery is concrete, intuitive, and immediately understandable, "
        f"clearly conveying the purpose of a {name}. "
        "Maintain vibrant, limited-palette colors, sharp vector lines, "
        "geometric shapes, flat illustration techniques, and solid colors "
        "without gradients or shading. Preserve a retro-futuristic aesthetic "
        "influenced by mid-century futurism and 1960s psychedelia, "
        "prioritizing clear visual storytelling and thematic clarity above all else."
    )

    custom_colors = [
        "#000030",
        "#1C0C47",
        "#9900FF",
        "#4285F4",
        "#FFFFFF",
    ]

    # Run the Ideogram model block with the specified parameters
    url = await IdeogramModelBlock().run_once(
        IdeogramModelBlock.Input(
            credentials=CredentialsMetaInput(
                id=ideogram_credentials.id,
                provider=ProviderName.IDEOGRAM,
                title=ideogram_credentials.title,
                type=ideogram_credentials.type,
            ),
            prompt=prompt,
            ideogram_model_name=IdeogramModelName.V3,
            aspect_ratio=AspectRatio.ASPECT_16_9,
            magic_prompt_option=MagicPromptOption.OFF,
            style_type=StyleType.AUTO,
            upscale=UpscaleOption.NO_UPSCALE,
            color_palette_name=ColorPalettePreset.NONE,
            custom_color_palette=custom_colors,
            seed=None,
            negative_prompt=None,
        ),
        "result",
        credentials=ideogram_credentials,
    )
    response = await Requests().get(url)
    return io.BytesIO(response.content)


async def generate_agent_image_v1(agent: GraphBaseMeta | AgentGraph) -> io.BytesIO:
    """
    Generate an image for an agent using Flux model via Replicate API.

    Args:
        agent (GraphBaseMeta | AgentGraph): The agent to generate an image for

    Returns:
        io.BytesIO: The generated image as bytes
    """
    try:
        if not settings.secrets.replicate_api_key:
            raise ValueError("Missing Replicate API key in settings")

        # Construct prompt from agent details
        prompt = (
            "Create a visually engaging app store thumbnail for the AI agent "
            "that highlights what it does in a clear and captivating way:\n"
            f"- **Name**: {agent.name}\n"
            f"- **Description**: {agent.description}\n"
            f"Focus on showcasing its core functionality with an appealing design."
        )

        # Set up Replicate client
        client = ReplicateClient(api_token=settings.secrets.replicate_api_key)

        # Model parameters
        input_data = {
            "prompt": prompt,
            "width": 1024,
            "height": 768,
            "aspect_ratio": "4:3",
            "output_format": "jpg",
            "output_quality": 90,
            "num_inference_steps": 30,
            "guidance": 3.5,
            "negative_prompt": "blurry, low quality, distorted, deformed",
            "disable_safety_checker": True,
        }

        try:
            # Run model
            output = client.run("black-forest-labs/flux-1.1-pro", input=input_data)

            # Depending on the model output, extract the image URL or bytes
            # If the output is a list of FileOutput or URLs
            if isinstance(output, list) and output:
                if isinstance(output[0], FileOutput):
                    image_bytes = output[0].read()
                else:
                    # If it's a URL string, fetch the image bytes
                    result_url = output[0]
                    response = await Requests().get(result_url)
                    image_bytes = response.content
            elif isinstance(output, FileOutput):
                image_bytes = output.read()
            elif isinstance(output, str):
                # Output is a URL
                response = await Requests().get(output)
                image_bytes = response.content
            else:
                raise RuntimeError("Unexpected output format from the model.")

            return io.BytesIO(image_bytes)

        except ReplicateError as e:
            if e.status == 401:
                raise RuntimeError("Invalid Replicate API token") from e
            raise RuntimeError(f"Replicate API error: {str(e)}") from e

    except Exception as e:
        logger.exception("Failed to generate agent image")
        raise RuntimeError(f"Image generation failed: {str(e)}")
