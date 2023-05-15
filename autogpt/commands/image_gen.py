""" Image Generation Module for AutoGPT."""
import io
import uuid
from base64 import b64decode

import openai
import requests
from PIL import Image

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.logs import logger

CFG = Config()


@command("generate_image", "Generate Image", '"prompt": "<prompt>"', CFG.image_provider)
def generate_image(prompt: str, size: int = 256) -> str:
    """Generate an image from a prompt.

    Args:
        prompt (str): The prompt to use
        size (int, optional): The size of the image. Defaults to 256. (Not supported by HuggingFace)

    Returns:
        str: The filename of the image
    """
    filename = f"{CFG.workspace_path}/{str(uuid.uuid4())}.jpg"

    # DALL-E
    if CFG.image_provider == "dalle":
        return generate_image_with_dalle(prompt, filename, size)
    # HuggingFace
    elif CFG.image_provider == "huggingface":
        return generate_image_with_hf(prompt, filename)
    # SD WebUI
    elif CFG.image_provider == "sdwebui":
        return generate_image_with_sd_webui(prompt, filename, size)
    return "No Image Provider Set"


def generate_image_with_hf(prompt: str, filename: str) -> str:
    """Generate an image with HuggingFace's API.

    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to

    Returns:
        str: The filename of the image
    """
    API_URL = (
        f"https://api-inference.huggingface.co/models/{CFG.huggingface_image_model}"
    )
    if CFG.huggingface_api_token is None:
        raise ValueError(
            "You need to set your Hugging Face API token in the config file."
        )
    headers = {
        "Authorization": f"Bearer {CFG.huggingface_api_token}",
        "X-Use-Cache": "false",
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": prompt,
        },
    )

    image = Image.open(io.BytesIO(response.content))
    logger.info(f"Image Generated for prompt:{prompt}")

    image.save(filename)

    return f"Saved to disk:{filename}"


def generate_image_with_dalle(prompt: str, filename: str, size: int) -> str:
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
        api_key=CFG.openai_api_key,
    )

    logger.info(f"Image Generated for prompt:{prompt}")

    image_data = b64decode(response["data"][0]["b64_json"])

    with open(filename, mode="wb") as png:
        png.write(image_data)

    return f"Saved to disk:{filename}"


def generate_image_with_sd_webui(
    prompt: str,
    filename: str,
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
    if CFG.sd_webui_auth:
        username, password = CFG.sd_webui_auth.split(":")
        s.auth = (username, password or "")

    # Generate the images
    response = requests.post(
        f"{CFG.sd_webui_url}/sdapi/v1/txt2img",
        json={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "sampler_index": "DDIM",
            "steps": 20,
            "cfg_scale": 7.0,
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
