"""Commands for converting audio to text."""
import json

import requests

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.workspace import path_in_workspace

CFG = Config()


@command(
    "read_audio_from_file",
    "Convert Audio to text",
    '"filename": "<filename>"',
    CFG.huggingface_audio_to_text_model,
    "Configure huggingface_audio_to_text_model.",
)
def read_audio_from_file(filename: str) -> str:
    """
    Convert audio to text.

    Args:
        audio_path (str): The path to the audio file

    Returns:
        str: The text from the audio
    """
    audio_path = path_in_workspace(filename)
    with open(audio_path, "rb") as audio_file:
        audio = audio_file.read()
    return read_audio(audio)


def read_audio(audio: bytes) -> str:
    """
    Convert audio to text.

    Args:
        audio (bytes): The audio to convert

    Returns:
        str: The text from the audio
    """
    model = CFG.huggingface_audio_to_text_model
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    api_token = CFG.huggingface_api_token
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
