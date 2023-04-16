import requests
import json

from autogpt.config import Config
from autogpt.workspace import path_in_workspace

cfg = Config()


def read_audio_from_file(audio_path):
    audio_path = path_in_workspace(audio_path)
    with open(audio_path, "rb") as audio_file:
        audio = audio_file.read()
    return read_audio(audio)


def read_audio(audio):
    model = cfg.huggingface_audio_to_text_model
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    api_token = cfg.huggingface_api_token
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
    return "The audio says: " + text
