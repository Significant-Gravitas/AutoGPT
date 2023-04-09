import os
from pathlib import Path
import chevron
from autogpt.config import Config
SRC_DIR = Path(__file__).parent

def load_prompt(cfg: Config):
    # get directory of this file:
    file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    data_dir = file_dir / "data"

    # Load the promt from data/prompt.txt
    with (data_dir/ "qa_prompt.txt").open() as prompt_file:
        QA_PROMPT = prompt_file.read()

    with (data_dir/ "prompt.txt.mustache").open() as prompt_file:
        prompt = prompt_file.read()
        if cfg.qa_mode:
            prompt = chevron.render(prompt, {"QA_PROMPT": QA_PROMPT})
        else:
            prompt = chevron.render(prompt, {"QA_PROMPT": ""})

    return prompt

