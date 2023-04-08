import os
from pathlib import Path

# Constants
# file location: src/data/prompts/index.txt
SRC_DIR = Path(__file__).parent
PROMPT_FILE = SRC_DIR / "data" / "prompts" / "index.txt"


def load_prompt():
    try:
        # Load the prompt from src/data/prompts/index.txt
        with open(PROMPT_FILE, "r") as prompt_file:
            prompt = prompt_file.read()

        return prompt
    except FileNotFoundError:
        print("Error: Prompt file not found", flush=True)
        return ""
