import requests

from autogpt.config import Config
from autogpt.commands.file_operations import safe_join

cfg = Config()

working_directory = "auto_gpt_workspace"


def summarize_image_from_file(image_path):
    image_path = safe_join(working_directory, image_path)
    with open(image_path, "rb") as image_file:
        image = image_file.read()
    return summarize_image(image)


def summarize_image(image):
    model = cfg.huggingface_image_to_text_model
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    api_token = cfg.huggingface_api_token
    headers = {"Authorization": f"Bearer {api_token}"}
    
    if api_token is None:
        raise ValueError("You need to set your Hugging Face API token in the config file.")
    
    response = requests.post(
        api_url, 
        headers=headers,
        data=image, 
    )
    
    return "The image is about: " + response.json()[0]["generated_text"]