import logging
import os

from requests import RequestException

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.workspace import Workspace


class GenerateImage(Ability):
    default_configuration = AbilityConfiguration(
        packages_required=["requests", "PIL", "openai"],
        workspace_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        workspace: Workspace,
        configuration: AbilityConfiguration,
    ):
        self._logger = logger
        self._workspace = workspace
        self._configuration = configuration

        # TODO: read this from some form of config?
        self.image_generator = "huggingface"

    @classmethod
    def description(self) -> str:
        return "Generate an Image from a prompt."

    @classmethod
    def arguments(self) -> dict:
        return {
            "prompt": {
                "type": "string",
                "description": "The prompt used to generate the image.",
            },
            "filename": {
                "type": "string",
                "description": "The desired name of the image generated.",
            },
        }

    @classmethod
    def required_arguments(cls) -> list[str]:
        return ["prompt"]

    def _check_preconditions(self, prompt: str, filename: str) -> AbilityResult | None:
        message = ""
        try:
            import requests
        except ImportError:
            message = "Package requests is not installed."
        try:
            import openai
        except ImportError:
            message = "Package openai is not installed."
        try:
            from PIL import Image
        except ImportError:
            message = "Package PIL is not installed."

        # TODO: check the config matches the passed in tokens to ensure we aren't wasting time
        try:
            # IF hugging face, check token
            raise PermissionError
        except PermissionError:
            message = "Huggingface token is not set."
        try:
            # IF dalle, check token
            raise PermissionError
        except PermissionError:
            message = "OpenAI API token is not set."
        try:
            # IF StableDiffusion WebAuth, check user/pass
            raise PermissionError
        except PermissionError:
            message = "Stable Diffusion username or password is not set."

        # TODO: Check if the file exists before overwriting
        try:
            file_path = self._workspace.get_path(filename)
            if file_path.exists():
                message = f"File {filename} already exists."
        except ValueError as e:
            message = str(e)

        if message:
            return AbilityResult(
                success=False,
                message=message,
            )

    def __call__(self, prompt: str, filename: str) -> AbilityResult:
        if result := self._check_preconditions(prompt, filename):
            return result

        file_path = self._workspace.get_path(filename)
        try:
            if self.image_generator is "huggingface":
                return self._generate_huggingface_image(
                    prompt=prompt,
                    filename=filename,
                    huggingface_api_token=""
                    huggingface_image_model=""
                )
            elif self.image_generator is "dalli":
                return self._generate_dalle_image(
                    prompt=prompt,
                    filename=filename,
                    api_key="",
                    size=512
                )
            elif self.image_generator is "stablediffusion":
                return self._generate_stablediffusion_image(
                    prompt=prompt,
                    filename=filename,
                    stable_diffusion_url="",
                    size=512,
                )

        except IOError as e:
            data = None
            success = False
            message = str(e)

        return AbilityResult(
            success=success,
            message=message,
            data=data,
        )

    # NOTE: How on earth do we combine a thousand APIs into one here? what if one requires more info 
    def _generate_huggingface_image(
        self,
        prompt: str,
        filename: str,
        huggingface_api_token: str,
        huggingface_image_model: str,
    ) -> AbilityResult:
        import io
        import json
        import time
        import requests
        from PIL import Image

        api_url = (
            f"https://api-inference.huggingface.co/models/{huggingface_image_model}"
        )

        headers = {
            "Authorization": f"Bearer {huggingface_api_token}",
            "X-Use-Cache": "false",
        }

        retry_count = 0
        while retry_count < 10:
            response = requests.post(
                api_url,
                headers=headers,
                json={
                    "inputs": prompt,
                },
            )

            if response.ok:
                try:
                    image = Image.open(io.BytesIO(response.content))
                    self._logger.info(f"Image Generated for prompt:{prompt}")
                    image.save(filename)
                    return AbilityResult(
                        success=True, message=f"Saved to disk:{filename}"
                    )
                except Exception as e:
                    message = "Failed to save image after generation. "
                    return AbilityResult(success=False, message=message)
            else:
                try:
                    error = json.loads(response.text)
                    if "estimated_time" in error:
                        delay = error["estimated_time"]
                        self._logger.debug(response.text)
                        self._logger.info("Retrying in", delay)
                        time.sleep(delay)
                    else:
                        break
                except Exception as e:
                    self._logger.error(e)
                    return AbilityResult(
                        success=False,
                        message=f"Failed to generate image for unknown reasons. {e}",
                    )

            retry_count += 1

        return AbilityResult(
            success=False,
            message=f"Failed to generate image after {retry_count} tries.",
        )

    def _generate_dalle_image(
        self,
        prompt: str,
        filename: str,
        api_key: str,
        size: int = 512
    ) -> AbilityResult:
        from base64 import b64decode
        import openai
        if size not in [256, 512, 1024]:
            closest = min([256, 512, 1024], key=lambda x: abs(x - size))
            self._logger.info(
                f"DALL-E only supports image sizes of 256x256, 512x512, or 1024x1024. Setting to {closest}, was {size}."
            )
            size = closest

        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size=f"{size}x{size}",
                response_format="b64_json",
                api_key=api_key
            )

            self._logger.info(f"Image Generated for prompt:{prompt}")

            image_data = b64decode(response["data"][0]["b64_json"])

            try:
                with open(filename, mode="wb") as png:
                    png.write(image_data)
            except IOError:
                return AbilityResult(
                    success=False,
                    message="Error while writing file."
                )
        except openai.OpenAIError as e:
            return AbilityResult(
                success=False,
                message=f"Error querying OpenAI. {e}"
            )

        return AbilityResult(
            success=True,
            message=f"Saved to disk:{filename}"
        )
    
    def _generate_stablediffusion_image(
        self,
        prompt: str,
        negative_prompt: str,
        filename: str,
        stable_diffusion_url: str,
        size: int = 512,
        username: str | None = None,
        password: str | None = None,
        extra: dict = {},
    ) -> AbilityResult:
        import io
        from base64 import b64decode
        import requests
        from PIL import Image
        s = requests.Session()
        if username and password:
            s.auth = (username, password or "")

        try:
            # Generate the images
            response = requests.post(
                f"{stable_diffusion_url}/sdapi/v1/txt2img",
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

            self._logger.info(f"Image Generated for prompt:{prompt}")

            # Save the image to disk
            response = response.json()
            b64 = b64decode(response["images"][0].split(",", 1)[0])
            image = Image.open(io.BytesIO(b64))
            image.save(filename)
        except RequestException as e:
            return AbilityResult(
                success=False,
                message=f"Failed to generate image: {e}"
            )

        return AbilityResult(success=True,message=f"Saved to disk:{filename}")
