from pathlib import Path

import requests

from autogpt.agent.agent import Agent
from autogpt.command_decorator import command
from autogpt.config import Config


@command(
    "summarize_image_from_file",
    "Generates a caption for an image",
    {
        "filename": {
            "type": "string",
            "description": "The filename of the image to describe",
            "required": True,
        },
    },
    lambda config: bool(
        config.huggingface_api_token and config.huggingface_image_to_text_model
    ),
    "Requires a HuggingFace API token and image-to-text model to be set.",
)
def summarize_image_from_file(image_path: str | Path, agent: Agent):
    """Summarize an image from a file path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The summary of the image.
    """
    image_path = agent.workspace.get_path(image_path)
    with open(image_path, "rb") as image_file:
        image = image_file.read()
    return summarize_image(image, agent.config)


def summarize_image(image: bytes, config: Config):
    """Summarize a image as a binary.

    Args:
        image (bytes): The image as a binary.

    Returns:
        str: The summary of the image.
    """

    if config.huggingface_api_token is None:
        raise RuntimeError(
            "You need to set your Hugging Face API token in the config file."
        )

    model = config.huggingface_image_to_text_model
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {config.huggingface_api_token}"}

    response = requests.post(
        api_url,
        headers=headers,
        data=image,
    )

    return "The image is about: " + response.json()[0]["generated_text"]
