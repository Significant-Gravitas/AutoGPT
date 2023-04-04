import os
from pathlib import Path
import chevron
from config import Config
SRC_DIR = Path(__file__).parent

def load_prompt(cfg: Config):
    try:
        # get directory of this file:
        file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        data_dir = file_dir / "data"
        prompt_file = data_dir / "prompt.txt"
        # Load the promt from data/prompt.txt
        with (SRC_DIR/ "data/qa_prompt.txt").open() as prompt_file:
            QA_PROMPT = prompt_file.read()

        with (SRC_DIR/ "data/prompt.txt.mustache").open() as prompt_file:
            prompt = prompt_file.read()
            if cfg.qa_mode:
                prompt = chevron.render(prompt, {"QA_PROMPT": QA_PROMPT})
            else:
                prompt = chevron.render(prompt, {"QA_PROMPT": ""})

        return prompt
    except FileNotFoundError:
        print("Error: Prompt file not found", flush=True)
        return ""
