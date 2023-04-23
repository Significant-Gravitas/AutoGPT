import hashlib
import os
import shutil
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from autogpt.commands.image_gen import generate_image, generate_image_with_sd_webui
from autogpt.config import Config
from autogpt.workspace import Workspace
from tests.utils import requires_api_key


def lst(txt):
    return Path(txt.split(":")[1].strip())


@unittest.skip("Skipping image generation tests")
class TestImageGen(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        workspace_path = os.path.join(os.path.dirname(__file__), "workspace")
        self.workspace_path = Workspace.make_workspace(workspace_path)
        self.config.workspace_path = workspace_path
        self.workspace = Workspace(workspace_path, restrict_to_workspace=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_path)

    @requires_api_key("OPENAI_API_KEY")
    def test_dalle(self):
        self.config.image_provider = "dalle"

        # Test using size 256
        image_path = lst(generate_image("astronaut riding a horse", 256))
        self.assertTrue(image_path.exists())
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (256, 256))
        image_path.unlink()

        # Test using size 512
        image_path = lst(generate_image("astronaut riding a horse", 512))
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (512, 512))
        image_path.unlink()

    @requires_api_key("HUGGINGFACE_API_TOKEN")
    def test_huggingface(self):
        self.config.image_provider = "huggingface"

        # Test usin SD 1.4 model and size 512
        self.config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"
        image_path = lst(generate_image("astronaut riding a horse", 512))
        self.assertTrue(image_path.exists())
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (512, 512))
        image_path.unlink()

        # Test using SD 2.1 768 model and size 768
        self.config.huggingface_image_model = "stabilityai/stable-diffusion-2-1"
        image_path = lst(generate_image("astronaut riding a horse", 768))
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (768, 768))
        image_path.unlink()

    @requires_api_key("HUGGINGFACE_API_TOKEN")
    @patch("time.sleep")
    @patch("requests.post")
    def test_huggingface_fail_request_with_delay(self, mock_post, mock_sleep):
        mock_post.return_value.status_code = 500
        mock_post.return_value.ok = False
        mock_post.return_value.text = "{\"error\":\"Model CompVis/stable-diffusion-v1-4 is currently loading\",\"estimated_time\":0}"

        self.config.image_provider = "huggingface"

        # Verify request fails.
        self.config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"
        result = generate_image("astronaut riding a horse", 512)
        self.assertTrue(result == "Error creating image.")

        # Verify retry was called with delay.
        mock_sleep.assert_called_with(0)

    @requires_api_key("HUGGINGFACE_API_TOKEN")
    @patch("time.sleep")
    @patch("requests.post")
    def test_huggingface_fail_request_no_delay(self, mock_post, mock_sleep):
        mock_post.return_value.status_code = 500
        mock_post.return_value.ok = False
        mock_post.return_value.text = "{\"error\":\"Model CompVis/stable-diffusion-v1-4 is currently loading\"}"

        self.config.image_provider = "huggingface"

        # Verify request fails.
        self.config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"
        result = generate_image("astronaut riding a horse", 512)
        self.assertTrue(result == "Error creating image.")

        # Verify retry was not called.
        mock_sleep.assert_not_called()

    @requires_api_key("HUGGINGFACE_API_TOKEN")
    @patch("time.sleep")
    @patch("requests.post")
    def test_huggingface_fail_request_bad_json(self, mock_post, mock_sleep):
        mock_post.return_value.status_code = 500
        mock_post.return_value.ok = False
        mock_post.return_value.text = "{\"error:}"

        self.config.image_provider = "huggingface"

        # Verify request fails.
        self.config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"
        result = generate_image("astronaut riding a horse", 512)
        self.assertTrue(result == "Error creating image.")

        # Verify retry was not called.
        mock_sleep.assert_not_called()

    def test_sd_webui(self):
        self.config.image_provider = "sd_webui"
        return

        # Test using size 128
        image_path = lst(generate_image_with_sd_webui("astronaut riding a horse", 128))
        self.assertTrue(image_path.exists())
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (128, 128))
        image_path.unlink()

        # Test using size 64 and negative prompt
        result = lst(
            generate_image_with_sd_webui(
                "astronaut riding a horse",
                negative_prompt="horse",
                size=64,
                extra={"seed": 123},
            )
        )
        image_path = path_in_workspace(result)
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (64, 64))
            neg_image_hash = hashlib.md5(img.tobytes()).hexdigest()
        image_path.unlink()

        # Same test as above but without the negative prompt
        result = lst(
            generate_image_with_sd_webui(
                "astronaut riding a horse", image_size=64, size=1, extra={"seed": 123}
            )
        )
        image_path = path_in_workspace(result)
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (64, 64))
            image_hash = hashlib.md5(img.tobytes()).hexdigest()
        image_path.unlink()

        self.assertNotEqual(image_hash, neg_image_hash)


if __name__ == "__main__":
    unittest.main()
