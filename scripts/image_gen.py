import requests
import io
import os.path
from PIL import Image
from config import Config
import uuid

cfg = Config()

working_directory = "auto_gpt_workspace"

API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
headers = {"Authorization": "Bearer " + cfg.huggingface_api_token}

def generate_image(prompt):
    response = requests.post(API_URL, headers=headers, json={
        "inputs": prompt,
    })
    image = Image.open(io.BytesIO(response.content))
    print("Image Generated for prompt:" + prompt)

    filename = str(uuid.uuid4()) + ".jpg"

    image.save(os.path.join(working_directory, filename))

    print("Saved to disk:" + filename)

    return str("Image " + filename + " saved to disk for prompt: " + prompt)
