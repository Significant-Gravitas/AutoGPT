"""Commands for converting audio to text."""
import json
from typing import TYPE_CHECKING

import requests

from autogpt.commands.command import command
from autogpt.config import Config

if TYPE_CHECKING:
    from autogpt.config import Config


@command(
    "read_audio_from_file",
    "Convert Audio to text",
    '"filename": "<filename>"',
    lambda config: config.huggingface_audio_to_text_model
    and config.huggingface_api_token,
    "Configure huggingface_audio_to_text_model and Hugging Face api token.",
)
def read_audio_from_file(filename: str, config: Config) -> str:
    """
    Convert audio to text.

    Args:
        filename (str): The path to the audio file

    Returns:
        str: The text from the audio
    """
    with open(filename, "rb") as audio_file:
        audio = audio_file.read()
    return read_audio(audio, config)


def read_audio(audio: bytes, config: Config) -> str:
    """
    Convert audio to text.

    Args:
        audio (bytes): The audio to convert

    Returns:
        str: The text from the audio
    """
    model = config.huggingface_audio_to_text_model
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    api_token = config.huggingface_api_token
    headers = {"Authorization": f"Bearer {api_token}"}

    if api_token is None:
        raise ValueError(
            "You need to set your Hugging Face API token in the config file."
        )

    response = requests.post(
        api_url,
        headers=headers,
        data=audio,
    )

    text = json.loads(response.content.decode("utf-8"))["text"]
    return f"The audio says: {text}"
