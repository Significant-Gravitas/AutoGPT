from dataclasses import dataclass
from gradio_tools import StableDiffusionTool, ImageCaptioningTool
from gradio_tools.tools import Job
from typing import Callable, Dict
from pathlib import Path

WORKSPACE_DIR = (".." / Path(__file__) / "auto_gpt_workspace").resolve()

@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[str], str]
    args: Dict[str, str]


class AutoGPTCaptioner(ImageCaptioningTool):

    def __init__(self, name="ImageCaptioner", description="An image captioner. Use this to create a caption for an image. " "Input will be a path to an image file. " "The output will be a caption of that image.", src="gradio-client-demos/comparing-captioning-models") -> None:
        super().__init__(name, description, src)
        self.args = {"img": "<full-path-to-image>"}

    def create_job(self, query: dict) -> Job:
        for v in query.values():
            if Path(v).exists():
                return super().create_job(v)
            elif Path(WORKSPACE_DIR / v).exists():
                return super().create_job(v)
        raise ValueError(f"Cannot create captioning job for query: {query}")


class AutoGPTStableDiffusion(StableDiffusionTool):
    def __init__(self, name="StableDiffusion", description="An image generator. Use this to generate images based on " "text input. Input should be a description of what the image should " "look like. The output will be a path to an image file.", src="gradio-client-demos/stable-diffusion") -> None:
        super().__init__(name, description, src)
        self.args = {"prompt": "text description of image"}


TOOLS = [AutoGPTStableDiffusion(), AutoGPTCaptioner()]


def get_tool(tool: str) -> Tool:
    return next(t for t in TOOLS if t.name == tool)


