import os
from pathlib import Path
from config import Config

cfg = Config()
SRC_DIR = Path(__file__).parent

def load_prompt():
    try:
        # get directory of this file:
        file_dir = Path(__file__).parent
        f cfg.db_access:
            prompt_file_path = file_dir / "data" / "prompt_db.txt"
        else:
            prompt_file_path = file_dir / "data" / "prompt.txt"
        
        # Load the prompt from data/prompt.txt
        with open(prompt_file_path, "r") as prompt_file:

            prompt = prompt_file.read()

        return prompt
    except FileNotFoundError:
        print("Error: Prompt file not found", flush=True)
        return ""
