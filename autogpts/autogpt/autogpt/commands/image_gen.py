"""Commands to generate images based on text input"""

COMMAND_CATEGORY = "text_to_image"
COMMAND_CATEGORY_TITLE = "Text to Image"

import io
import json
import logging
import time
import uuid
from base64 import b64decode

import openai
import requests
from PIL import Image

from autogpt.agents.agent import Agent
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema

logger = logging.getLogger(__name__)


@command(
    "generate_image",
    "Generates an Image",
    {
        "prompt": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The prompt used to generate the image",
            required=True,
        ),
    },
    lambda config: bool(config.image_provider),
    "Requires a image provider to be set.",
)
def generate_image(prompt: str, agent: Agent, size: int = 256) -> str:
    """Generate an image from a prompt.

    Args:
        prompt (str): The prompt to use
        size (int, optional): The size of the image. Defaults to 256. (Not supported by HuggingFace)

    Returns:
        str: The filename of the image
    """
    filename = agent.legacy_config.workspace_path / f"{str(uuid.uuid4())}.jpg"

    # DALL-E
    if agent.legacy_config.image_provider == "dalle":
        return generate_image_with_dalle(prompt, filename, size, agent)
    # HuggingFace
    elif agent.legacy_config.image_provider == "huggingface":
        return generate_image_with_hf(prompt, filename, agent)
    # SD WebUI
    elif agent.legacy_config.image_provider == "sdwebui":
        return generate_image_with_sd_webui(prompt, filename, agent, size)
    return "No Image Provider Set"


def generate_image_with_hf(prompt: str, filename: str, agent: Agent) -> str:
    """Generate an image with HuggingFace's API.

    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to

    Returns:
        str: The filename of the image
    """
    API_URL = f"https://api-inference.huggingface.co/models/{agent.legacy_config.huggingface_image_model}"
    if agent.legacy_config.huggingface_api_token is None:
        raise ValueError(
            "You need to set your Hugging Face API token in the config file."
        )
    headers = {
        "Authorization": f"Bearer {agent.legacy_config.huggingface_api_token}",
        "X-Use-Cache": "false",
    }

    retry_count = 0
    while retry_count < 10:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
            },
        )

        if response.ok:
            try:
                image = Image.open(io.BytesIO(response.content))
                logger.info(f"Image Generated for prompt:{prompt}")
                image.save(filename)
                return f"Saved to disk:{filename}"
            except Exception as e:
                logger.error(e)
                break
        else:
            try:
                error = json.loads(response.text)
                if "estimated_time" in error:
                    delay = error["estimated_time"]
                    logger.debug(response.text)
                    logger.info("Retrying in", delay)
                    time.sleep(delay)
                else:
                    break
            except Exception as e:
                logger.error(e)
                break

        retry_count += 1

    return f"Error creating image."


def generate_image_with_dalle(
    prompt: str, filename: str, size: int, agent: Agent
) -> str:
    """Generate an image with DALL-E.

    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to
        size (int): The size of the image

    Returns:
        str: The filename of the image
    """

    # Check for supported image sizes
    if size not in [256, 512, 1024]:
        closest = min([256, 512, 1024], key=lambda x: abs(x - size))
        logger.info(
            f"DALL-E only supports image sizes of 256x256, 512x512, or 1024x1024. Setting to {closest}, was {size}."
        )
        size = closest

    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=f"{size}x{size}",
        response_format="b64_json",
        api_key=agent.legacy_config.openai_api_key,
    )

    logger.info(f"Image Generated for prompt:{prompt}")

    image_data = b64decode(response["data"][0]["b64_json"])

    with open(filename, mode="wb") as png:
        png.write(image_data)

    return f"Saved to disk:{filename}"


def generate_image_with_sd_webui(
    prompt: str,
    filename: str,
    agent: Agent,
    size: int = 512,
    negative_prompt: str = "",
    extra: dict = {},
) -> str:
    """Generate an image with Stable Diffusion webui.
    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to
        size (int, optional): The size of the image. Defaults to 256.
        negative_prompt (str, optional): The negative prompt to use. Defaults to "".
        extra (dict, optional): Extra parameters to pass to the API. Defaults to {}.
    Returns:
        str: The filename of the image
    """
    # Create a session and set the basic auth if needed
    s = requests.Session()
    if agent.legacy_config.sd_webui_auth:
        username, password = agent.legacy_config.sd_webui_auth.split(":")
        s.auth = (username, password or "")

    # Generate the images
    response = requests.post(
        f"{agent.legacy_config.sd_webui_url}/sdapi/v1/txt2img",
        json={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "sampler_index": "DDIM",
            "steps": 20,
            "config_scale": 7.0,
            "width": size,
            "height": size,
            "n_iter": 1,
            **extra,
        },
    )

    logger.info(f"Image Generated for prompt:{prompt}")

    # Save the image to disk
    response = response.json()
    b64 = b64decode(response["images"][0].split(",", 1)[0])
    image = Image.open(io.BytesIO(b64))
    image.save(filename)

    return f"Saved to disk:{filename}"
