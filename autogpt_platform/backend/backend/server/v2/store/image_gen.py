import asyncio
import io
import logging
from enum import Enum

import replicate
import replicate.exceptions
from prisma.models import AgentGraph
from replicate.helpers import FileOutput

from backend.blocks.ideogram import (
    AspectRatio,
    ColorPalettePreset,
    IdeogramModelBlock,
    IdeogramModelName,
    MagicPromptOption,
    StyleType,
    UpscaleOption,
)
from backend.data.graph import Graph
from backend.data.model import CredentialsMetaInput, ProviderName
from backend.integrations.credentials_store import ideogram_credentials
from backend.util.request import requests
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class ImageSize(str, Enum):
    LANDSCAPE = "1024x768"


class ImageStyle(str, Enum):
    DIGITAL_ART = "digital art"


async def generate_agent_image(agent: Graph | AgentGraph) -> io.BytesIO:
    if settings.config.use_agent_image_generation_v2:
        return await asyncio.to_thread(generate_agent_image_v2, graph=agent)
    else:
        return await generate_agent_image_v1(agent=agent)


def generate_agent_image_v2(graph: Graph | AgentGraph) -> io.BytesIO:
    """
    Generate an image for an agent using Ideogram model.
    Returns:
        str: The URL of the generated image
    """
    if not ideogram_credentials.api_key:
        raise ValueError("Missing Ideogram API key")

    name = graph.name
    description = f"{name} ({graph.description})" if graph.description else name

    prompt = (
        f"Create a visually striking retro-futuristic vector pop art illustration prominently featuring "
        f'"{name}" in bold typography. The image clearly and literally depicts a {description}, '
        f"along with recognizable objects directly associated with the primary function of a {name}. "
        f"Ensure the imagery is concrete, intuitive, and immediately understandable, clearly conveying the "
        f"purpose of a {name}. Maintain vibrant, limited-palette colors, sharp vector lines, geometric "
        f"shapes, flat illustration techniques, and solid colors without gradients or shading. Preserve a "
        f"retro-futuristic aesthetic influenced by mid-century futurism and 1960s psychedelia, "
        f"prioritizing clear visual storytelling and thematic clarity above all else."
    )

    custom_colors = [
        "#000030",
        "#1C0C47",
        "#9900FF",
        "#4285F4",
        "#FFFFFF",
    ]

    # Run the Ideogram model block with the specified parameters
    url = IdeogramModelBlock().run_once(
        IdeogramModelBlock.Input(
            credentials=CredentialsMetaInput(
                id=ideogram_credentials.id,
                provider=ProviderName.IDEOGRAM,
                title=ideogram_credentials.title,
                type=ideogram_credentials.type,
            ),
            prompt=prompt,
            ideogram_model_name=IdeogramModelName.V2,
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
    return io.BytesIO(requests.get(url).content)


async def generate_agent_image_v1(agent: Graph | AgentGraph) -> io.BytesIO:
    """
    Generate an image for an agent using Flux model via Replicate API.

    Args:
        agent (Graph): The agent to generate an image for

    Returns:
        io.BytesIO: The generated image as bytes
    """
    try:
        if not settings.secrets.replicate_api_key:
            raise ValueError("Missing Replicate API key in settings")

        # Construct prompt from agent details
        prompt = f"Create a visually engaging app store thumbnail for the AI agent that highlights what it does in a clear and captivating way:\n- **Name**: {agent.name}\n- **Description**: {agent.description}\nFocus on showcasing its core functionality with an appealing design."

        # Set up Replicate client
        client = replicate.Client(api_token=settings.secrets.replicate_api_key)

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
                    response = requests.get(result_url)
                    image_bytes = response.content
            elif isinstance(output, FileOutput):
                image_bytes = output.read()
            elif isinstance(output, str):
                # Output is a URL
                response = requests.get(output)
                image_bytes = response.content
            else:
                raise RuntimeError("Unexpected output format from the model.")

            return io.BytesIO(image_bytes)

        except replicate.exceptions.ReplicateError as e:
            if e.status == 401:
                raise RuntimeError("Invalid Replicate API token") from e
            raise RuntimeError(f"Replicate API error: {str(e)}") from e

    except Exception as e:
        logger.exception("Failed to generate agent image")
        raise RuntimeError(f"Image generation failed: {str(e)}")
