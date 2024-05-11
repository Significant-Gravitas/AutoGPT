"""Commands to generate images based on text input"""

import io
import json
import logging
import time
import uuid
from base64 import b64decode
from pathlib import Path
from typing import Iterator

import requests
from openai import OpenAI
from PIL import Image

from forge.agent.protocols import CommandProvider
from forge.command import Command, command
from forge.config.config import Config
from forge.file_storage import FileStorage
from forge.json.model import JSONSchema

logger = logging.getLogger(__name__)


class ImageGeneratorComponent(CommandProvider):
    """A component that provides commands to generate images from text prompts."""

    def __init__(self, workspace: FileStorage, config: Config):
        self._enabled = bool(config.image_provider)
        self._disabled_reason = "No image provider set."
        self.workspace = workspace
        self.legacy_config = config

    def get_commands(self) -> Iterator[Command]:
        yield self.generate_image

    @command(
        parameters={
            "prompt": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The prompt used to generate the image",
                required=True,
            ),
            "size": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="The size of the image",
                required=False,
            ),
        },
    )
    def generate_image(self, prompt: str, size: int) -> str:
        """Generate an image from a prompt.

        Args:
            prompt (str): The prompt to use
            size (int, optional): The size of the image. Defaults to 256.
                Not supported by HuggingFace.

        Returns:
            str: The filename of the image
        """
        filename = self.workspace.root / f"{str(uuid.uuid4())}.jpg"

        # DALL-E
        if self.legacy_config.image_provider == "dalle":
            return self.generate_image_with_dalle(prompt, filename, size)
        # HuggingFace
        elif self.legacy_config.image_provider == "huggingface":
            return self.generate_image_with_hf(prompt, filename)
        # SD WebUI
        elif self.legacy_config.image_provider == "sdwebui":
            return self.generate_image_with_sd_webui(prompt, filename, size)
        return "No Image Provider Set"

    def generate_image_with_hf(self, prompt: str, output_file: Path) -> str:
        """Generate an image with HuggingFace's API.

        Args:
            prompt (str): The prompt to use
            filename (Path): The filename to save the image to

        Returns:
            str: The filename of the image
        """
        API_URL = f"https://api-inference.huggingface.co/models/{self.legacy_config.huggingface_image_model}"  # noqa: E501
        if self.legacy_config.huggingface_api_token is None:
            raise ValueError(
                "You need to set your Hugging Face API token in the config file."
            )
        headers = {
            "Authorization": f"Bearer {self.legacy_config.huggingface_api_token}",
            "X-Use-Cache": "false",
        }

        retry_count = 0
        while retry_count < 10:
            response = requests.post(
                API_URL,
                headers=headers,
                json={
                    "inputs": prompt,
                },
            )

            if response.ok:
                try:
                    image = Image.open(io.BytesIO(response.content))
                    logger.info(f"Image Generated for prompt:{prompt}")
                    image.save(output_file)
                    return f"Saved to disk: {output_file}"
                except Exception as e:
                    logger.error(e)
                    break
            else:
                try:
                    error = json.loads(response.text)
                    if "estimated_time" in error:
                        delay = error["estimated_time"]
                        logger.debug(response.text)
                        logger.info("Retrying in", delay)
                        time.sleep(delay)
                    else:
                        break
                except Exception as e:
                    logger.error(e)
                    break

            retry_count += 1

        return "Error creating image."

    def generate_image_with_dalle(
        self, prompt: str, output_file: Path, size: int
    ) -> str:
        """Generate an image with DALL-E.

        Args:
            prompt (str): The prompt to use
            filename (Path): The filename to save the image to
            size (int): The size of the image

        Returns:
            str: The filename of the image
        """

        # Check for supported image sizes
        if size not in [256, 512, 1024]:
            closest = min([256, 512, 1024], key=lambda x: abs(x - size))
            logger.info(
                "DALL-E only supports image sizes of 256x256, 512x512, or 1024x1024. "
                f"Setting to {closest}, was {size}."
            )
            size = closest

        response = OpenAI(
            api_key=self.legacy_config.openai_credentials.api_key.get_secret_value()
        ).images.generate(
            prompt=prompt,
            n=1,
            size=f"{size}x{size}",
            response_format="b64_json",
        )

        logger.info(f"Image Generated for prompt:{prompt}")

        image_data = b64decode(response.data[0].b64_json)

        with open(output_file, mode="wb") as png:
            png.write(image_data)

        return f"Saved to disk: {output_file}"

    def generate_image_with_sd_webui(
        self,
        prompt: str,
        output_file: Path,
        size: int = 512,
        negative_prompt: str = "",
        extra: dict = {},
    ) -> str:
        """Generate an image with Stable Diffusion webui.
        Args:
            prompt (str): The prompt to use
            filename (str): The filename to save the image to
            size (int, optional): The size of the image. Defaults to 256.
            negative_prompt (str, optional): The negative prompt to use. Defaults to "".
            extra (dict, optional): Extra parameters to pass to the API. Defaults to {}.
        Returns:
            str: The filename of the image
        """
        # Create a session and set the basic auth if needed
        s = requests.Session()
        if self.legacy_config.sd_webui_auth:
            username, password = self.legacy_config.sd_webui_auth.split(":")
            s.auth = (username, password or "")

        # Generate the images
        response = requests.post(
            f"{self.legacy_config.sd_webui_url}/sdapi/v1/txt2img",
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "sampler_index": "DDIM",
                "steps": 20,
                "config_scale": 7.0,
                "width": size,
                "height": size,
                "n_iter": 1,
                **extra,
            },
        )

        logger.info(f"Image Generated for prompt: '{prompt}'")

        # Save the image to disk
        response = response.json()
        b64 = b64decode(response["images"][0].split(",", 1)[0])
        image = Image.open(io.BytesIO(b64))
        image.save(output_file)

        return f"Saved to disk: {output_file}"
