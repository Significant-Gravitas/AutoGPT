"""Commands for converting audio to text."""
import json

import requests

from autogpt.agent.agent import Agent
from autogpt.commands.command import command


@command(
    "read_audio_from_file",
    "Convert Audio to text",
    '"filename": "<filename>"',
    lambda config: config.huggingface_audio_to_text_model
    and config.huggingface_api_token,
    "Configure huggingface_audio_to_text_model and Hugging Face api token.",
)
def read_audio_from_file(filename: str, agent: Agent) -> str:
    """
    Convert audio to text.

    Args:
        filename (str): The path to the audio file

    Returns:
        str: The text from the audio
    """
    with open(filename, "rb") as audio_file:
        audio = audio_file.read()
    return read_audio(audio, agent.config)


def read_audio(audio: bytes, agent: Agent) -> str:
    """
    Convert audio to text.

    Args:
        audio (bytes): The audio to convert

    Returns:
        str: The text from the audio
    """
    if agent.config.audio_to_text_provider == "huggingface":
        text = read_huggingface_audio(audio, agent.config)
        if text:
            return f"The audio says: {text}"
        else:
            return f"Error, couldn't convert audio to text"

    return "Error: No audio to text provider given"


def read_huggingface_audio(audio: bytes, agent: Agent) -> str:
    model = agent.config.huggingface_audio_to_text_model
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    api_token = agent.config.huggingface_api_token
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

    response_json = json.loads(response.content.decode("utf-8"))
    return response_json.get("text")
