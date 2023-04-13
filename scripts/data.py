import os
from pathlib import Path
from jinja2 import Template, Environment, FileSystemLoader
from typing import List
import tools

def load_prompt(requested_tools: List[tools.Tool]) -> str:
    """Load the prompt from data/prompt.txt"""
    try:
        # get directory of this file:
        file_dir = Path(__file__).parent
        env = Environment(loader=FileSystemLoader(str(file_dir / "data")))
        template = env.get_template("prompt.txt")
        return template.render(tools=requested_tools)
    except FileNotFoundError:
        print("Error: Prompt file not found", flush=True)
        return ""
