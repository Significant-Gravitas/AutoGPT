""" Image Generation Module for AutoGPT."""
import io
import os.path
import uuid
from base64 import b64decode

import openai
import requests
from PIL import Image
from autogpt.config import Config
from autogpt.workspace import path_in_workspace

CFG = Config()


def generate_image(prompt: str, image_size: str = "256x256") -> str:
    """Generate an image from a prompt.

    Args:
        prompt (str): The prompt to use

    Returns:
        str: The filename of the image
    """
    filename = f"{str(uuid.uuid4())}.jpg"

    # DALL-E
    if CFG.image_provider == "dalle":
        return generate_image_with_dalle(prompt, filename, image_size)
    elif CFG.image_provider == "sd":
        return generate_image_with_hf(prompt, filename)
    else:
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
        "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
    )
    if CFG.huggingface_api_token is None:
        raise ValueError(
            "You need to set your Hugging Face API token in the config file."
        )
    headers = {"Authorization": f"Bearer {CFG.huggingface_api_token}"}

    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": prompt,
        },
    )

    try:
        image = Image.open(io.BytesIO(response.content))
    except IOError:
        return "Error: cannot identify image file"

    image = Image.open(io.BytesIO(response.content))
    print(f"Image Generated for prompt:{prompt}")

    image.save(path_in_workspace(filename))

    return f"Saved to disk:{filename}"


def generate_image_with_dalle(prompt: str, filename: str, image_size: str) -> str:
    """Generate an image with DALL-E.

    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to
        image_size (str): The image_size

    Returns:
        str: The filename of the image
    """
    openai.api_key = CFG.openai_api_key

    allowed_sizes = ["256x256", "512x512", "1024x1024"]

    if image_size not in allowed_sizes:
        image_size = "256x256"

    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=image_size,
        response_format="b64_json",
    )

    print(f"Image Generated for prompt:{prompt}")

    image_data = b64decode(response["data"][0]["b64_json"])

    with open(path_in_workspace(filename), mode="wb") as png:
        png.write(image_data)

    return f"Saved to disk:{filename}"
