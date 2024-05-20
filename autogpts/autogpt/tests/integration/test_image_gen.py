import functools
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
from forge.components.image_gen import ImageGeneratorComponent
from PIL import Image

from autogpt.agents.agent import Agent


@pytest.fixture
def image_gen_component(agent: Agent):
    return agent.image_gen


@pytest.fixture(params=[256, 512, 1024])
def image_size(request):
    """Parametrize image size."""
    return request.param


@pytest.mark.requires_openai_api_key
@pytest.mark.vcr
def test_dalle(
    image_gen_component: ImageGeneratorComponent,
    agent: Agent,
    storage,
    image_size,
    cached_openai_client,
):
    """Test DALL-E image generation."""
    generate_and_validate(
        image_gen_component,
        agent,
        storage,
        image_provider="dalle",
        image_size=image_size,
    )


@pytest.mark.xfail(
    reason="The image is too big to be put in a cassette for a CI pipeline. "
    "We're looking into a solution."
)
@pytest.mark.requires_huggingface_api_key
@pytest.mark.parametrize(
    "image_model",
    ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"],
)
def test_huggingface(
    image_gen_component: ImageGeneratorComponent,
    agent: Agent,
    storage,
    image_size,
    image_model,
):
    """Test HuggingFace image generation."""
    generate_and_validate(
        image_gen_component,
        agent,
        storage,
        image_provider="huggingface",
        image_size=image_size,
        hugging_face_image_model=image_model,
    )


@pytest.mark.xfail(reason="SD WebUI call does not work.")
def test_sd_webui(
    image_gen_component: ImageGeneratorComponent, agent: Agent, storage, image_size
):
    """Test SD WebUI image generation."""
    generate_and_validate(
        image_gen_component,
        agent,
        storage,
        image_provider="sd_webui",
        image_size=image_size,
    )


@pytest.mark.xfail(reason="SD WebUI call does not work.")
def test_sd_webui_negative_prompt(
    image_gen_component: ImageGeneratorComponent, storage, image_size
):
    gen_image = functools.partial(
        image_gen_component.generate_image_with_sd_webui,
        prompt="astronaut riding a horse",
        size=image_size,
        extra={"seed": 123},
    )

    # Generate an image with a negative prompt
    image_path = lst(
        gen_image(negative_prompt="horse", output_file=Path("negative.jpg"))
    )
    with Image.open(image_path) as img:
        neg_image_hash = hashlib.md5(img.tobytes()).hexdigest()

    # Generate an image without a negative prompt
    image_path = lst(gen_image(output_file=Path("positive.jpg")))
    with Image.open(image_path) as img:
        image_hash = hashlib.md5(img.tobytes()).hexdigest()

    assert image_hash != neg_image_hash


def lst(txt):
    """Extract the file path from the output of `generate_image()`"""
    return Path(txt.split(": ", maxsplit=1)[1].strip())


def generate_and_validate(
    image_gen_component: ImageGeneratorComponent,
    agent: Agent,
    storage,
    image_size,
    image_provider,
    hugging_face_image_model=None,
    **kwargs,
):
    """Generate an image and validate the output."""
    agent.legacy_config.image_provider = image_provider
    if hugging_face_image_model:
        agent.legacy_config.huggingface_image_model = hugging_face_image_model
    prompt = "astronaut riding a horse"

    image_path = lst(image_gen_component.generate_image(prompt, image_size, **kwargs))
    assert image_path.exists()
    with Image.open(image_path) as img:
        assert img.size == (image_size, image_size)


@pytest.mark.parametrize(
    "return_text",
    [
        # Delay
        '{"error":"Model [model] is currently loading","estimated_time": [delay]}',
        '{"error":"Model [model] is currently loading"}',  # No delay
        '{"error:}',  # Bad JSON
        "",  # Bad Image
    ],
)
@pytest.mark.parametrize(
    "image_model",
    ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"],
)
@pytest.mark.parametrize("delay", [10, 0])
def test_huggingface_fail_request_with_delay(
    image_gen_component: ImageGeneratorComponent,
    agent: Agent,
    storage,
    image_size,
    image_model,
    return_text,
    delay,
):
    return_text = return_text.replace("[model]", image_model).replace(
        "[delay]", str(delay)
    )

    with patch("requests.post") as mock_post:
        if return_text == "":
            # Test bad image
            mock_post.return_value.status_code = 200
            mock_post.return_value.ok = True
            mock_post.return_value.content = b"bad image"
        else:
            # Test delay and bad json
            mock_post.return_value.status_code = 500
            mock_post.return_value.ok = False
            mock_post.return_value.text = return_text

        agent.legacy_config.image_provider = "huggingface"
        agent.legacy_config.huggingface_api_token = "mock-api-key"
        agent.legacy_config.huggingface_image_model = image_model
        prompt = "astronaut riding a horse"

        with patch("time.sleep") as mock_sleep:
            # Verify request fails.
            result = image_gen_component.generate_image(prompt, image_size)
            assert result == "Error creating image."

            # Verify retry was called with delay if delay is in return_text
            if "estimated_time" in return_text:
                mock_sleep.assert_called_with(delay)
            else:
                mock_sleep.assert_not_called()


def test_huggingface_fail_request_no_delay(
    mocker, image_gen_component: ImageGeneratorComponent, agent: Agent
):
    agent.legacy_config.huggingface_api_token = "1"

    # Mock requests.post
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 500
    mock_post.return_value.ok = False
    mock_post.return_value.text = (
        '{"error":"Model CompVis/stable-diffusion-v1-4 is currently loading"}'
    )

    # Mock time.sleep
    mock_sleep = mocker.patch("time.sleep")

    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    result = image_gen_component.generate_image("astronaut riding a horse", 512)

    assert result == "Error creating image."

    # Verify retry was not called.
    mock_sleep.assert_not_called()


def test_huggingface_fail_request_bad_json(
    mocker, image_gen_component: ImageGeneratorComponent, agent: Agent
):
    agent.legacy_config.huggingface_api_token = "1"

    # Mock requests.post
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 500
    mock_post.return_value.ok = False
    mock_post.return_value.text = '{"error:}'

    # Mock time.sleep
    mock_sleep = mocker.patch("time.sleep")

    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    result = image_gen_component.generate_image("astronaut riding a horse", 512)

    assert result == "Error creating image."

    # Verify retry was not called.
    mock_sleep.assert_not_called()


def test_huggingface_fail_request_bad_image(
    mocker, image_gen_component: ImageGeneratorComponent, agent: Agent
):
    agent.legacy_config.huggingface_api_token = "1"

    # Mock requests.post
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 200

    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    result = image_gen_component.generate_image("astronaut riding a horse", 512)

    assert result == "Error creating image."


def test_huggingface_fail_missing_api_token(
    mocker, image_gen_component: ImageGeneratorComponent, agent: Agent
):
    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    # Mock requests.post to raise ValueError
    mocker.patch("requests.post", side_effect=ValueError)

    # Verify request raises an error.
    with pytest.raises(ValueError):
        image_gen_component.generate_image("astronaut riding a horse", 512)
