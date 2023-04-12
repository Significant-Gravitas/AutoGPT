import io
import os.path
import uuid
from base64 import b64decode

import openai
import requests
from config import Config
from PIL import Image

cfg = Config()

working_directory = "auto_gpt_workspace"


def generate_image(prompt):
    filename = f"{str(uuid.uuid4())}.jpg"

    # DALL-E
    if cfg.image_provider == "dalle":
        openai.api_key = cfg.openai_api_key

        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="256x256",
            response_format="b64_json",
        )

        print(f"Image Generated for prompt:{prompt}")

        image_data = b64decode(response["data"][0]["b64_json"])

        with open(f"{working_directory}/{filename}", mode="wb") as png:
            png.write(image_data)

        return f"Saved to disk:{filename}"

    else:
        return "No Image Provider Set"
