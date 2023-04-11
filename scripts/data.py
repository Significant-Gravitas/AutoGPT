import os
from pathlib import Path

from config import Config
from conversation_summary.summary import SummaryUtils

cfg = Config()


def load_prompt():
    """Load the prompt from data/prompt.txt"""
    try:
        # get directory of this file:
        file_dir = Path(__file__).parent
        prompt_file_path = file_dir / "data" / "prompt.txt"

        # Load the prompt from data/prompt.txt
        with open(prompt_file_path, "r") as prompt_file:
            prompt = prompt_file.read()

        if cfg.conversation_summary_mode:
            # Add the summary field to the response structure in the AutoGPT prompt
            prompt = SummaryUtils.add_summary_field_to_prompt(
                prompt, value=cfg.step_summarization_prompt
            )
        return prompt

    except FileNotFoundError:
        print("Error: Prompt file not found", flush=True)
        return ""
