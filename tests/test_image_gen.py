import hashlib
import os
import unittest

from PIL import Image

from autogpt.commands.image_gen import generate_image, generate_image_with_sd_webui
from autogpt.config import Config
from autogpt.workspace import path_in_workspace


def lst(txt):
    return txt.split(":")[1].strip()


@unittest.skipIf(os.getenv("CI"), "Skipping image generation tests")
class TestImageGen(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_dalle(self):
        self.config.image_provider = "dalle"

        # Test using size 256
        result = lst(generate_image("astronaut riding a horse", 256))
        image_path = path_in_workspace(result)
        self.assertTrue(image_path.exists())
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (256, 256))
        image_path.unlink()

        # Test using size 512
        result = lst(generate_image("astronaut riding a horse", 512))
        image_path = path_in_workspace(result)
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (512, 512))
        image_path.unlink()

    def test_huggingface(self):
        self.config.image_provider = "huggingface"

        # Test usin SD 1.4 model and size 512
        self.config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"
        result = lst(generate_image("astronaut riding a horse", 512))
        image_path = path_in_workspace(result)
        self.assertTrue(image_path.exists())
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (512, 512))
        image_path.unlink()

        # Test using SD 2.1 768 model and size 768
        self.config.huggingface_image_model = "stabilityai/stable-diffusion-2-1"
        result = lst(generate_image("astronaut riding a horse", 768))
        image_path = path_in_workspace(result)
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (768, 768))
        image_path.unlink()

    def test_sd_webui(self):
        self.config.image_provider = "sd_webui"
        return

        # Test using size 128
        result = lst(generate_image_with_sd_webui("astronaut riding a horse", 128))
        image_path = path_in_workspace(result)
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
