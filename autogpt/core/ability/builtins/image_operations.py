import logging
import os

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

                )
            elif self.image_generator is "stablediffusion":
                return self._generate_stablediffusion_image(

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
    ) -> AbilityResult:
        return AbilityResult()
    
    def _generate_stablediffusion_image(
        self,
        prompt: str,
        filename: str,
    ) -> AbilityResult:
        return AbilityResult()