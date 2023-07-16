from pathlib import Path
import requests
import shutil

import pytest

from autogpt.agents.agent import Agent
from autogpt.workspace import Workspace
from autogpt.commands.image_text import summarize_image_from_file


@pytest.fixture
def image_filenames(workspace: Workspace):
    addresses = [
        "https://cdn.pixabay.com/photo/2023/05/12/03/12/australian-king-parrot-7987514_1280.jpg",
        "https://cdn.pixabay.com/photo/2023/07/01/18/56/dog-8100754_1280.jpg",
        "https://cdn.pixabay.com/photo/2023/07/08/09/53/monastery-8114076_1280.jpg",
        "https://cdn.pixabay.com/photo/2022/08/17/15/46/labrador-7392840_1280.jpg"
    ]
    filenames = []
    for address in addresses:
        res = requests.get(address, stream=True)
        filename = address.rsplit('/', maxsplit=1)[-1]
        filenames.append(filename)
        filepath = workspace.get_path(filename)
        assert res.status_code == 200
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
    return filenames

def summarize_and_validate(
    image_filenames,
    agent: Agent,
    workspace,
    hugging_face_image_to_text_model=None,
    **kwargs,
):
    """Generate an image and validate the output."""
    agent.config.huggingface_image_to_text_model = hugging_face_image_to_text_model
    for image_filename in image_filenames:
        image_path = workspace.get_path(image_filename)
        assert image_path.exists()
        summary = summarize_image_from_file(image_path, agent)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert summary != "Error describing image."


@pytest.mark.requires_huggingface_api_key
@pytest.mark.parametrize(
    "image_to_text_model",
    ["nlpconnect/vit-gpt2-image-captioning"],
)
def test_huggingface(image_filenames, agent: Agent, workspace, image_to_text_model):
    """Test HuggingFace image generation."""
    summarize_and_validate(
        image_filenames,
        agent,
        workspace,
        hugging_face_image_to_text_model=image_to_text_model,
    )
