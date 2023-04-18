from dataclasses import dataclass
from gradio_tools import (
    StableDiffusionTool,
    ImageCaptioningTool,
    TextToVideoTool,
    StableDiffusionPromptGeneratorTool,
    WhisperTool,
)
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
    def __init__(
        self,
        name="ImageCaptioner",
        description="An image captioner. Use this to create a caption for an image. "
        "Input will be a path to an image file. "
        "The output will be a caption of that image.",
        src="gradio-client-demos/comparing-captioning-models",
    ) -> None:
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
    def __init__(
        self,
        name="StableDiffusion",
        description="An image generator. Use this to generate images based on "
        "text input. Input should be a description of what the image should "
        "look like. The output will be a path to an image file.",
        src="gradio-client-demos/stable-diffusion",
    ) -> None:
        super().__init__(name, description, src)
        self.args = {"prompt": "text description of image"}


class AutoGPTWhisperTool(WhisperTool):
    def __init__(
        self,
        name="Whisper",
        description="A tool for transcribing audio. Use this tool to transcribe an audio file. "
        "track from an image. Input will be a path to an audio file. "
        "The output will the text transcript of that file.",
        src="abidlabs/whisper-large-v2",
        hf_token=None,
    ) -> None:
        super().__init__(name, description, src, hf_token)
        self.args = {"audio": "full path of audio file"}


class AutoGPTTextToVideoTool(TextToVideoTool):
    def __init__(
        self,
        name="TextToVideo",
        description="A tool for creating videos from text."
        "Use this tool to create videos from text prompts. "
        "Input will be a text prompt describing a video scene. "
        "The output will be a path to a video file.",
        src="damo-vilab/modelscope-text-to-video-synthesis",
        hf_token=None,
    ) -> None:
        super().__init__(name, description, src, hf_token)
        self.args = {"prompt": "text description of video"}


class AUtGPTPromptGeneratorTool(StableDiffusionPromptGeneratorTool):
    def __init__(
        self,
        name="StableDiffusionPromptGenerator",
        description="Use this tool to improve a prompt for stable diffusion and other image generators "
        "This tool will refine your prompt to include key words and phrases that make "
        "stable diffusion perform better. The input is a prompt text string "
        "and the output is a prompt text string",
        src="microsoft/Promptist",
        hf_token=None,
    ) -> None:
        super().__init__(name, description, src, hf_token)
        self.args = {"prompt": "text description of image"}


TOOLS = [
    AutoGPTStableDiffusion(),
    AutoGPTCaptioner(),
    AutoGPTWhisperTool(),
    AutoGPTTextToVideoTool(),
]


def get_tool(tool: str) -> Tool:
    return next(t for t in TOOLS if t.name == tool)
