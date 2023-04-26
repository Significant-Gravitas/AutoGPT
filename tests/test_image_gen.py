import hashlib
from pathlib import Path

from PIL import Image

from autogpt.commands.image_gen import generate_image, generate_image_with_sd_webui
from tests.utils import requires_api_key


def lst(txt):
    """Extract the file path from the output of `generate_image()`"""
    return Path(txt.split(":")[1].strip())


@requires_api_key("OPENAI_API_KEY")
def test_dalle(request, config, workspace):
    """Test DALL-E image generation."""
    config.image_provider = "dalle"

    # Test using size 256
    image_path = lst(generate_image("astronaut riding a horse", 256))
    assert image_path.exists()
    with Image.open(image_path) as img:
        assert img.size == (256, 256)
    image_path.unlink()

    # Test using size 512
    image_path = lst(generate_image("astronaut riding a horse", 512))
    with Image.open(image_path) as img:
        assert img.size == (512, 512)
    image_path.unlink()


@requires_api_key("HUGGINGFACE_API_TOKEN")
def test_huggingface(request, config, workspace):
    """Test HuggingFace image generation."""
    config.image_provider = "huggingface"

    # Test using SD 1.4 model and size 512
    config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"
    image_path = lst(generate_image("astronaut riding a horse", 512))
    assert image_path.exists()
    with Image.open(image_path) as img:
        assert img.size == (512, 512)
    image_path.unlink()

    # Test using SD 2.1 768 model and size 768
    config.huggingface_image_model = "stabilityai/stable-diffusion-2-1"
    image_path = lst(generate_image("astronaut riding a horse", 768))
    with Image.open(image_path) as img:
        assert img.size == (768, 768)
    image_path.unlink()


def test_sd_webui(config, workspace):
    """Test SD WebUI image generation."""
    config.image_provider = "sd_webui"
    return

    # Test using size 128
    image_path = lst(generate_image_with_sd_webui("astronaut riding a horse", 128))
    assert image_path.exists()
    with Image.open(image_path) as img:
        assert img.size == (128, 128)
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
        assert img.size == (64, 64)

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
        assert img.size == (64, 64)
        image_hash = hashlib.md5(img.tobytes()).hexdigest()
    image_path.unlink()

    assert image_hash != neg_image_hash
