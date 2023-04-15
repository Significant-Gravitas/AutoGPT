import io
import os.path
import uuid
from base64 import b64decode

import openai
import requests
from PIL import Image

from autogpt.config import Config

cfg = Config()

working_directory = "auto_gpt_workspace"


def generate_image(prompt):
    filename = str(uuid.uuid4()) + ".jpg"

    # DALL-E
    if cfg.image_provider == "dalle":
        openai.api_key = cfg.openai_api_key

        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="256x256",
            response_format="b64_json",
        )

        print("Image Generated for prompt:" + prompt)

        image_data = b64decode(response["data"][0]["b64_json"])

        with open(working_directory + "/" + filename, mode="wb") as png:
            png.write(image_data)

        return "Saved to disk:" + filename

    # STABLE DIFFUSION
    elif cfg.image_provider == "sd":
        API_URL = (
            "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
        )
        if cfg.huggingface_api_token is None:
            raise ValueError(
                "You need to set your Hugging Face API token in the config file."
            )
        headers = {"Authorization": "Bearer " + cfg.huggingface_api_token}

        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
            },
        )

        image = Image.open(io.BytesIO(response.content))
        print("Image Generated for prompt:" + prompt)

        image.save(os.path.join(working_directory, filename))

        return "Saved to disk:" + filename

    else:
        return "No Image Provider Set"
