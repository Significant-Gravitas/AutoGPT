import io
import json
import logging
import time
import uuid
from base64 import b64decode
from pathlib import Path
from typing import Iterator, Literal, Optional

import requests
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, SecretStr

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider
from forge.command import Command, command
from forge.file_storage import FileStorage
from forge.llm.providers.openai import OpenAICredentials
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema

logger = logging.getLogger(__name__)


class ImageGeneratorConfiguration(BaseModel):
    image_provider: Literal["dalle", "huggingface", "sdwebui"] = "dalle"
    huggingface_image_model: str = "CompVis/stable-diffusion-v1-4"
    huggingface_api_token: Optional[SecretStr] = UserConfigurable(
        None, from_env="HUGGINGFACE_API_TOKEN", exclude=True
    )
    sd_webui_url: str = "http://localhost:7860"
    sd_webui_auth: Optional[SecretStr] = UserConfigurable(
        None, from_env="SD_WEBUI_AUTH", exclude=True
    )


class ImageGeneratorComponent(
    CommandProvider, ConfigurableComponent[ImageGeneratorConfiguration]
):
    """A component that provides commands to generate images from text prompts."""

    config_class = ImageGeneratorConfiguration

    def __init__(
        self,
        workspace: FileStorage,
        config: Optional[ImageGeneratorConfiguration] = None,
        openai_credentials: Optional[OpenAICredentials] = None,
    ):
        """openai_credentials only needed for `dalle` provider."""
        ConfigurableComponent.__init__(self, config)
        self.openai_credentials = openai_credentials
        self._enabled = bool(self.config.image_provider)
        self._disabled_reason = "No image provider set."
        self.workspace = workspace

    def get_commands(self) -> Iterator[Command]:
        if (
            self.openai_credentials
            or self.config.huggingface_api_token
            or self.config.sd_webui_auth
        ):
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
                description="The size of the image [256, 512, 1024]",
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

        if self.openai_credentials and (
            self.config.image_provider == "dalle"
            or not (self.config.huggingface_api_token or self.config.sd_webui_url)
        ):
            return self.generate_image_with_dalle(prompt, filename, size)

        elif self.config.huggingface_api_token and (
            self.config.image_provider == "huggingface"
            or not (self.openai_credentials or self.config.sd_webui_url)
        ):
            return self.generate_image_with_hf(prompt, filename)

        elif self.config.sd_webui_url and (
            self.config.image_provider == "sdwebui" or self.config.sd_webui_auth
        ):
            return self.generate_image_with_sd_webui(prompt, filename, size)

        return "Error: No image generation provider available"

    def generate_image_with_hf(self, prompt: str, output_file: Path) -> str:
        """Generate an image with HuggingFace's API.

        Args:
            prompt (str): The prompt to use
            filename (Path): The filename to save the image to

        Returns:
            str: The filename of the image
        """
        API_URL = f"https://api-inference.huggingface.co/models/{self.config.huggingface_image_model}"  # noqa: E501
        if self.config.huggingface_api_token is None:
            raise ValueError(
                "You need to set your Hugging Face API token in the config file."
            )
        headers = {
            "Authorization": (
                f"Bearer {self.config.huggingface_api_token.get_secret_value()}"
            ),
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
        assert self.openai_credentials  # otherwise this tool is disabled

        # Check for supported image sizes
        if size not in [256, 512, 1024]:
            closest = min([256, 512, 1024], key=lambda x: abs(x - size))
            logger.info(
                "DALL-E only supports image sizes of 256x256, 512x512, or 1024x1024. "
                f"Setting to {closest}, was {size}."
            )
            size = closest

        # TODO: integrate in `forge.llm.providers`(?)
        response = OpenAI(
            api_key=self.openai_credentials.api_key.get_secret_value(),
            organization=self.openai_credentials.organization.get_secret_value()
            if self.openai_credentials.organization
            else None,
        ).images.generate(
            prompt=prompt,
            n=1,
            # TODO: improve typing of size config item(s)
            size=f"{size}x{size}",  # type: ignore
            response_format="b64_json",
        )
        assert response.data[0].b64_json is not None  # response_format = "b64_json"

        logger.info(f"Image Generated for prompt: {prompt}")

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
        if self.config.sd_webui_auth:
            username, password = self.config.sd_webui_auth.get_secret_value().split(":")
            s.auth = (username, password or "")

        # Generate the images
        response = requests.post(
            f"{self.config.sd_webui_url}/sdapi/v1/txt2img",
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
