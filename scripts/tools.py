from dataclasses import dataclass
from gradio_tools import StableDiffusionTool, ImageCaptioningTool, ImageToMusicTool
from gradio_tools.tools import Job
from typing import Callable
import os
from pathlib import Path

WORKSPACE_DIR = (".." / Path(__file__) / "auto_gpt_workspace").resolve()

@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[str], str]

class AutoGPTCaptioner(ImageCaptioningTool):

    def create_job(self, query: dict) -> Job:
        for v in query.values():
            if Path(v).exists():
                return super().create_job(v)
            elif Path(WORKSPACE_DIR / v).exists():
                return super().create_job(v)
        raise ValueError(f"Cannot create captioning job for query: {query}")

TOOLS = [StableDiffusionTool(), AutoGPTCaptioner(), ImageToMusicTool()]


def get_tool(tool: str) -> Tool:
    return next(t for t in TOOLS if t.name == tool)


